//! USB keyboards and mice, on the HID boot protocol.
//!
//! The built-in keyboard and touchpad on the GF63 are i8042 devices and always
//! were. This is for everything a person actually plugs in, which on a laptop
//! with a dock is usually all of it, and for the machines where the i8042 does
//! not exist at all.
//!
//! ### Boot protocol, and why that is a decision rather than a stage
//!
//! A HID device describes its own reports with a *report descriptor*: a small
//! stack language of usage pages, logical minimums and bit-field sizes, which
//! has to be interpreted before a single byte of a report means anything. That
//! interpreter is most of a HID stack, and getting it subtly wrong produces a
//! keyboard that types the wrong letters rather than an error.
//!
//! The boot protocol exists so a BIOS never has to write one. A device with
//! `bInterfaceSubClass == 1` promises that, when asked, it will also speak a
//! fixed 8-byte keyboard report or 3-byte mouse report, and every keyboard and
//! mouse anybody plugs into a computer supports it. What it costs is the extra
//! keys a gaming keyboard puts in its own report, the fourth and fifth mouse
//! buttons, and high-resolution scrolling. That is a fair price for not having
//! a report-descriptor interpreter in ring 0, and the limit is stated rather
//! than discovered: a device that offers no boot interface is ignored, and
//! `usb hid` says so by name.
//!
//! ### One decoder, reached two ways
//!
//! A HID keyboard reports usage codes and this turns them into **PS/2
//! scancodes**, which go to `kbd::inject_scancode` and through the same
//! `decode` the i8042 uses. Shift, caps lock, control, Alt held against Alt
//! tapped, Alt-Tab, Alt-Space and Ctrl-Escape are policy, and policy written
//! twice is policy that disagrees with itself. The table below is data; there
//! is no second opinion about what a key means.
//!
//! The mouse is the same argument by a different route: `mouse::apply` takes a
//! movement and owns the arithmetic that turns it into a cursor, and the two
//! drivers differ only in unpacking their own wire format.
//!
//! ### Reports are state, and events are the difference between them
//!
//! A boot keyboard report is the set of keys held *right now*, up to six of
//! them, in no particular order. There is no press or release in it. So each
//! report is compared against the one before: a usage that appeared is a make
//! code, one that vanished is a break code, and a modifier bit that changed is
//! either. Losing a report therefore loses two events and can leave a key
//! stuck down, which is why the transfer is kept permanently armed rather than
//! queued per poll.

use super::xhci::{self, Device, HidIface};
use crate::sync::Racy;
use alloc::vec::Vec;

/// HID class requests, sent to the interface.
const SET_IDLE: u8 = 0x0A;
const SET_PROTOCOL: u8 = 0x0B;

/// Report sizes. The boot keyboard report is 8 bytes and the boot mouse report
/// is 3, with a fourth for the wheel on anything made since about 1996. The
/// buffer is larger than either so a device that sends its own longer report
/// anyway does not scribble past the end.
const REPORT_LEN: u32 = 16;

const KEYBOARD: u8 = 1;
const MOUSE: u8 = 2;

/// One attached input device.
struct Hid {
    dev: Device,
    protocol: u8,
    /// Where the controller writes reports.
    buf: u64,
    /// Whether a transfer is currently queued.
    armed: bool,
    /// The previous report, which is what makes events out of state.
    last: [u8; 8],
    /// Reports seen, for `usb hid`.
    reports: u64,
}

static DEVICES: Racy<Vec<Hid>> = Racy::new(Vec::new());

/// Devices that offered HID but not the boot protocol, so `usb hid` can say
/// what it declined instead of leaving a keyboard mysteriously dead.
static DECLINED: Racy<Vec<(u16, u16)>> = Racy::new(Vec::new());

pub fn attached() -> usize {
    unsafe { (*DEVICES.get()).len() }
}

/// Find and configure every boot-protocol keyboard and mouse on the bus.
///
/// Safe to call when the controller is already running and some other driver
/// owns a device on it: ports another driver claimed are skipped, because
/// enumerating one twice hands it a second slot and leaves the first driver
/// ringing a doorbell the controller no longer associates with it.
pub fn probe(ecam: u64) -> Result<usize, &'static str> {
    xhci::ensure_started(ecam)?;
    let mut found = 0usize;

    for port in xhci::free_ports() {
        // A port that will not enumerate is said out loud. It used to `continue`
        // in silence, so a keyboard that the controller could see and not
        // address looked exactly like no keyboard at all.
        let mut dev = match xhci::with_ctl(|c| c.enumerate(port)).unwrap_or(Err("no controller")) {
            Ok(d) => d,
            Err(e) => {
                crate::kprintln!("  hid    port {}: {}", port, e);
                continue;
            }
        };
        let mut claimed = false;

        for i in 0..dev.num_configs {
            let Some(Ok((buf, total))) = xhci::with_ctl(|c| c.config_descriptor(&mut dev, i))
            else {
                break;
            };
            let cfg = xhci::parse_config(buf, total);
            let Some(hid) = pick(&cfg.hids) else { continue };

            if xhci::with_ctl(|c| c.set_configuration(&mut dev, cfg.value))
                .unwrap_or(Err("no controller"))
                .is_err()
            {
                continue;
            }
            match setup(&mut dev, hid) {
                Ok(()) => {
                    claimed = true;
                    found += 1;
                }
                Err(e) => {
                    crate::kprintln!("  hid    port {} refused setup: {}", port, e);
                }
            }
            break;
        }

        if claimed {
            xhci::claim_port(port);
            continue;
        }
        if let Some(Ok((buf, total))) = xhci::with_ctl(|c| c.config_descriptor(&mut dev, 0))
        {
            // A device that has a HID interface but no boot one. Recorded so
            // the report can name it: a keyboard that does not work is worth a
            // sentence, and silence is what makes it look like the port is
            // broken.
            let cfg = xhci::parse_config(buf, total);
            if cfg.hids.is_empty() && looks_hid(buf, total) {
                unsafe { (*DECLINED.get()).push((dev.vid, dev.pid)) };
            }
        }
        // Not ours, so the slot goes back. A device left addressed to a port
        // makes that port unenumerable for everything that walks the bus
        // afterwards.
        xhci::with_ctl(|c| c.release(dev));
    }
    Ok(found)
}

/// Prefer a keyboard over a mouse when one interface has to be chosen.
///
/// Only one interface per device is driven, because each needs its own
/// interrupt endpoint and its own armed transfer, and a combined dongle
/// presents keyboard and mouse as two interfaces of one device. Driving both
/// is a second `Hid` sharing a slot, which the transfer ring supports and
/// nothing here has been able to test. So it is left undone and said out loud
/// rather than written blind.
fn pick(hids: &[HidIface]) -> Option<HidIface> {
    hids.iter()
        .find(|h| h.protocol == KEYBOARD)
        .or_else(|| hids.first())
        .copied()
}

/// Whether a descriptor block mentions the HID class at all.
fn looks_hid(buf: u64, total: usize) -> bool {
    let at = |o: usize| -> u8 { unsafe { core::ptr::read_volatile((buf + o as u64) as *const u8) } };
    let mut o = 0usize;
    while o + 9 <= total {
        let len = at(o) as usize;
        if len == 0 {
            return false;
        }
        if at(o + 1) == 4 && at(o + 5) == 0x03 {
            return true;
        }
        o += len;
    }
    false
}

fn setup(dev: &mut Device, hid: HidIface) -> Result<(), &'static str> {
    if hid.ep_in.addr == 0 {
        return Err("no interrupt endpoint on the boot interface");
    }

    // SET_PROTOCOL(0) is what actually asks for boot reports. A device that
    // supports the boot subclass still powers up in *report* protocol, so
    // skipping this gets a device-defined report that happens to look right on
    // simple keyboards and does not on anything else.
    xhci::with_ctl(|c| {
        c.control(dev, 0x21, SET_PROTOCOL, 0, hid.number as u16, 0, 0)
    })
    .unwrap_or(Err("no controller"))?;

    // SET_IDLE(0) means "report only when something changes". The default is
    // to repeat the current state forever at a fixed rate, which works and
    // wastes the poll on a keyboard nobody is touching.
    //
    // Not fatal if refused: some devices stall it, and a keyboard that reports
    // continuously is still a keyboard.
    let _ = xhci::with_ctl(|c| c.control(dev, 0x21, SET_IDLE, 0, hid.number as u16, 0, 0));

    // bInterval from the descriptor. Full-speed devices state it in
    // milliseconds and the controller wants log2 of 125 us units, so it is
    // converted rather than passed through -- handing 10 straight to a
    // controller that reads it as 2^10 * 125 us asks for one poll every eight
    // seconds.
    let interval = encode_interval(hid.interval);
    xhci::with_ctl(|c| c.configure_interrupt(dev, hid.ep_in, interval))
        .unwrap_or(Err("no controller"))?;

    let buf = xhci::dma(REPORT_LEN as usize, 16).ok_or("out of memory")?;
    let mut entry = Hid {
        dev: core::mem::replace(dev, xhci::Device::placeholder()),
        protocol: hid.protocol,
        buf,
        armed: false,
        last: [0; 8],
        reports: 0,
    };
    entry.armed = xhci::with_ctl(|c| c.arm_interrupt(&mut entry.dev, buf, REPORT_LEN))
        .unwrap_or(false);
    if hid.protocol == MOUSE {
        super::mouse::declare_present();
    }
    unsafe { (*DEVICES.get()).push(entry) };
    Ok(())
}

/// bInterval to the controller's `Interval` field.
///
/// Full and low speed state the period in milliseconds directly; high speed
/// and above already state it as a power of two of 125 us frames. Without
/// knowing the speed this takes the conservative reading and treats the value
/// as milliseconds, which for a high-speed device polls slower than it asked
/// for and never faster. Slower is the safe direction: too fast is a
/// bandwidth reservation the controller can refuse outright, and the endpoint
/// then does not exist at all.
fn encode_interval(b_interval: u8) -> u8 {
    let ms = b_interval.max(1) as u32;
    // log2 of (ms * 8), since one millisecond is eight 125 us frames.
    let frames = ms * 8;
    let mut e = 0u8;
    while (1u32 << (e + 1)) <= frames && e < 15 {
        e += 1;
    }
    e
}

/// Check every attached device for a report. Never waits.
///
/// Called from the shell's idle loop beside `tcp::service`, for the reason
/// that path already exists: there is no interrupt-driven USB in this kernel,
/// so anything that wants to hear from the bus has to ask.
pub fn poll() {
    let devices = unsafe { &mut *DEVICES.get() };
    if devices.is_empty() {
        return;
    }
    for h in devices.iter_mut() {
        if !h.armed {
            h.armed = xhci::with_ctl(|c| c.arm_interrupt(&mut h.dev, h.buf, REPORT_LEN))
                .unwrap_or(false);
            continue;
        }
        let Some(Some(n)) = xhci::with_ctl(|c| c.poll_interrupt(&h.dev, REPORT_LEN)) else {
            continue;
        };
        h.armed = false;
        h.reports += 1;
        // The first one, once, out loud. A device that enumerates and
        // configures perfectly and then never reports looks identical to one
        // that is working until somebody presses a key, and there is no way to
        // tell those apart from a boot log. This is also the only evidence
        // that reaches a headless test, since a keystroke injected through the
        // emulator arrives long after the last shell command.
        if h.reports == 1 {
            crate::kprintln!(
                "[usb] first report from {:04x}:{:04x}, the {} is live",
                h.dev.vid,
                h.dev.pid,
                if h.protocol == KEYBOARD { "keyboard" } else { "mouse" }
            );
        }

        let mut report = [0u8; 8];
        for i in 0..report.len().min(n as usize) {
            report[i] = unsafe { core::ptr::read_volatile((h.buf + i as u64) as *const u8) };
        }
        match h.protocol {
            KEYBOARD => keyboard(h, &report),
            MOUSE => mouse(&report, n),
            _ => {}
        }

        // Re-arm at once. A keyboard that is only listened to after being
        // asked drops the release of the key it just reported, and a key whose
        // release is lost is a key that stays down.
        h.armed = xhci::with_ctl(|c| c.arm_interrupt(&mut h.dev, h.buf, REPORT_LEN))
            .unwrap_or(false);
    }
}

/// Modifier bit to scancode.
///
/// The right-hand control and alt keys are given their *left-hand* scancodes
/// deliberately. Their real codes are `E0`-prefixed, and `decode` ignores
/// releases on the extended path, so a right control sent faithfully would
/// latch on and never let go. `decode` already treats an extended right
/// control as a left one for the same reason; this keeps the release working
/// too.
const MODIFIERS: [u8; 8] = [
    0x1D, // left control
    0x2A, // left shift
    0x38, // left alt
    0x00, // left GUI, which this kernel has no key for
    0x1D, // right control
    0x36, // right shift
    0x38, // right alt
    0x00, // right GUI
];

fn keyboard(h: &mut Hid, report: &[u8; 8]) {
    // 0x01 in the first key slot is ErrorRollOver: more keys held than the
    // report can carry. The key array is meaningless then, and treating it as
    // real would release everything currently down.
    if report[2] == 0x01 {
        return;
    }

    let (was, now) = (h.last[0], report[0]);
    if was != now {
        for bit in 0..8 {
            let before = was & (1 << bit) != 0;
            let after = now & (1 << bit) != 0;
            if before == after || MODIFIERS[bit] == 0 {
                continue;
            }
            let sc = MODIFIERS[bit];
            super::kbd::inject_scancode(if after { sc } else { sc | 0x80 });
        }
    }

    // Released: in the old report and not the new one.
    for &u in &h.last[2..8] {
        if u > 3 && !report[2..8].contains(&u) {
            if let Some(sc) = scancode(u) {
                emit(sc, false);
            }
        }
    }
    // Pressed: in the new report and not the old one.
    for &u in &report[2..8] {
        if u > 3 && !h.last[2..8].contains(&u) {
            if let Some(sc) = scancode(u) {
                emit(sc, true);
            }
        }
    }
    h.last = *report;
}

/// Send one scancode, with the `E0` prefix where the key needs it.
///
/// The prefix is carried in the high bit of the table entry, since a set-1
/// scancode is seven bits and the eighth is the release flag.
fn emit(sc: u16, pressed: bool) {
    if sc & 0x100 != 0 {
        super::kbd::inject_scancode(0xE0);
    }
    let code = (sc & 0xFF) as u8;
    super::kbd::inject_scancode(if pressed { code } else { code | 0x80 });
}

fn mouse(report: &[u8; 8], n: u32) {
    let buttons = report[0];
    let dx = report[1] as i8 as i32;
    let dy = report[2] as i8 as i32;
    // The wheel byte is optional and its absence is not an error, so it is
    // taken from the length the transfer actually returned rather than
    // assumed. A three-byte report read as four scrolls on every movement.
    let wheel = if n >= 4 { -(report[3] as i8 as i32) } else { 0 };
    super::mouse::apply(dx, dy, buttons & 0x01 != 0, buttons & 0x02 != 0, wheel);
}

/// HID usage to PS/2 set-1 scancode. Bit 8 marks the ones that need `E0`.
///
/// Pure data, and the reason this module has no opinion about what a key
/// means: everything from here goes through the same `decode` the built-in
/// keyboard does.
fn scancode(usage: u8) -> Option<u16> {
    const LETTERS: [u8; 26] = [
        0x1E, 0x30, 0x2E, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17, 0x24, 0x25, 0x26, 0x32, 0x31, 0x18,
        0x19, 0x10, 0x13, 0x1F, 0x14, 0x16, 0x2F, 0x11, 0x2D, 0x15, 0x2C,
    ];
    // 1234567890, in that order; '1' is scancode 2 and '0' is 11.
    const DIGITS: [u8; 10] = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B];

    let sc: u16 = match usage {
        0x04..=0x1D => LETTERS[(usage - 0x04) as usize] as u16,
        0x1E..=0x27 => DIGITS[(usage - 0x1E) as usize] as u16,
        0x28 => 0x1C, // enter
        0x29 => 0x01, // escape
        0x2A => 0x0E, // backspace
        0x2B => 0x0F, // tab
        0x2C => 0x39, // space
        0x2D => 0x0C, // - _
        0x2E => 0x0D, // = +
        0x2F => 0x1A, // [ {
        0x30 => 0x1B, // ] }
        0x31 | 0x32 => 0x2B,
        0x33 => 0x27, // ; :
        0x34 => 0x28, // ' "
        0x35 => 0x29, // ` ~
        0x36 => 0x33, // , <
        0x37 => 0x34, // . >
        0x38 => 0x35, // / ?
        0x39 => 0x3A, // caps lock
        0x3A..=0x43 => 0x3B + (usage - 0x3A) as u16, // F1..F10
        0x44 => 0x57,                                // F11
        0x45 => 0x58,                                // F12
        0x49 => 0x152,                               // insert
        0x4A => 0x147,                               // home
        0x4B => 0x149,                               // page up
        0x4C => 0x153,                               // delete
        0x4D => 0x14F,                               // end
        0x4E => 0x151,                               // page down
        0x4F => 0x14D,                               // right
        0x50 => 0x14B,                               // left
        0x51 => 0x150,                               // down
        0x52 => 0x148,                               // up
        _ => return None,
    };
    Some(sc)
}

pub fn report() {
    let devices = unsafe { &*DEVICES.get() };
    let declined = unsafe { &*DECLINED.get() };
    if devices.is_empty() && declined.is_empty() {
        crate::kprintln!("  no USB input devices");
        return;
    }
    for h in devices.iter() {
        let kind = if h.protocol == KEYBOARD { "keyboard" } else { "mouse" };
        crate::kprintln!(
            "  {:04x}:{:04x}  {}  port {}  slot {}  {} report(s){}",
            h.dev.vid,
            h.dev.pid,
            kind,
            h.dev.port,
            h.dev.slot,
            h.reports,
            if h.armed { "" } else { "  (not armed)" }
        );
    }
    for (vid, pid) in declined.iter() {
        crate::kprintln!("  {:04x}:{:04x}  HID without a boot interface, not driven", vid, pid);
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // The table is the whole driver, so it is checked rather than trusted.
    claim(&mut ok, scancode(0x04) == Some(0x1E), "usage 0x04 is 'a', which is scancode 0x1E");
    claim(&mut ok, scancode(0x1D) == Some(0x2C), "and usage 0x1D is 'z' at 0x2C");
    claim(&mut ok, scancode(0x1E) == Some(0x02), "'1' is scancode 2 and not scancode 1");
    claim(&mut ok, scancode(0x27) == Some(0x0B), "and '0' is 11, at the end of the row");
    claim(&mut ok, scancode(0x28) == Some(0x1C), "enter");
    claim(&mut ok, scancode(0x52) == Some(0x148), "an arrow key asks for the E0 prefix");
    claim(&mut ok, scancode(0x00) == None, "and a usage this font of keyboard has no key for is refused");

    // Every letter and digit maps somewhere distinct. A duplicated entry is
    // two keys that type the same character, which is exactly the kind of
    // thing that is invisible until somebody types the second one.
    let mut distinct = true;
    for a in 0x04u8..=0x27 {
        for b in (a + 1)..=0x27 {
            if scancode(a).is_some() && scancode(a) == scancode(b) {
                distinct = false;
            }
        }
    }
    claim(&mut ok, distinct, "no two letters or digits share a scancode");

    // The interval conversion, at the two ends that matter. 1 ms is eight
    // 125 us frames, so 2^3; 10 ms is eighty, and the exponent below it is 6.
    claim(&mut ok, encode_interval(1) == 3, "a 1 ms interval is 2^3 frames of 125 us");
    claim(&mut ok, encode_interval(10) == 6, "a 10 ms interval rounds down rather than up");
    claim(&mut ok, encode_interval(0) == 3, "and a device claiming zero is polled, not divided by it");

    // A boot report is state, so the same report twice must produce no
    // events. Checked through the real path, with nothing attached: what is
    // being claimed is the diffing, and `last` is where it lives.
    let mut h = Hid {
        dev: xhci::Device::placeholder(),
        protocol: KEYBOARD,
        buf: 0,
        armed: false,
        last: [0; 8],
        reports: 0,
    };
    let held = [0u8, 0, 0x04, 0, 0, 0, 0, 0];
    let before = crate::dev::kbd::pending();
    keyboard(&mut h, &held);
    let after_press = crate::dev::kbd::pending();
    keyboard(&mut h, &held);
    let after_repeat = crate::dev::kbd::pending();
    claim(&mut ok, after_press > before, "a key appearing in a report is a press");
    claim(&mut ok, after_repeat == after_press, "and the same report again is not a second one");

    // Rollover must not be read as "every key was released", which would
    // send a break code for a key the operator is still holding.
    let rollover = [0u8, 0, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01];
    keyboard(&mut h, &rollover);
    claim(&mut ok, h.last == held, "a rollover report is ignored rather than believed");
    while crate::dev::kbd::pop().is_some() {}
    ok
}
