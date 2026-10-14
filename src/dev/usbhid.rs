//! USB keyboards and mice, on the HID boot protocol.
//!
//! The GF63's built-in keyboard is an i8042 device, and its touchpad is *not*:
//! the trip found the touchpad on USB, which makes this driver the only
//! pointer that laptop has. So a dead endpoint here is a machine nobody can
//! use, and the failure paths below are held to that.
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
use crate::sync::{Racy, Spin};
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

/// One attached input device, and every boot interface on it that is driven.
struct Hid {
    dev: Device,
    ifaces: Vec<Iface>,
    /// HID interfaces on the device this driver cannot speak, for `usb hid`.
    other: u8,
}

/// One boot-protocol interface: a keyboard or a mouse, with its own
/// interrupt endpoint and its own transfer.
///
/// A device is not one interface. A gaming mouse presents a keyboard (its
/// macro keys) beside the mouse, and the driver took the keyboard and
/// stopped: the trip's mouse was lit, enumerated as `258a:0033 keyboard`,
/// and moved nothing.
struct Iface {
    protocol: u8,
    /// bEndpointAddress of its interrupt IN endpoint.
    ep: u8,
    /// Where the controller writes reports.
    buf: u64,
    /// Whether a transfer is currently queued.
    armed: bool,
    /// The previous report, which is what makes events out of state.
    last: [u8; 8],
    /// Where a mouse's fields sit when it was left in report protocol, read
    /// off its own report descriptor. `None` is the boot report.
    layout: Option<Layout>,
    /// Reports seen, for `usb hid`.
    reports: u64,
    /// Transfers that completed with an error, and how many of those the
    /// endpoint was brought back from.
    failures: u64,
    recovered: u64,
}

/// Where PCI configuration space is, kept from the first probe so the
/// compositor's poll can walk the bus again when a device comes back.
static ECAM: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// TSC of the last look at the ports, so it is done a few times a second
/// and not on every turn of the compositor's loop.
static LAST_PORT_LOOK: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// Devices that dropped off the bus and were picked up again, for `usb hid`.
static REJOINED: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// **A lock, because two tasks reach it.** `poll` runs on the compositor so a
/// USB mouse keeps moving while the shell is inside a command -- the same
/// reason the pointer moved there -- and `probe` runs on the shell when
/// somebody types `usb hid`. `probe` holds this for its whole walk of the bus,
/// so a poll landing mid-enumeration finds it taken and skips a turn rather
/// than reading a half-built entry. The controller underneath has its own lock
/// and stashes transfer events it was not asked for, so the two tasks' transfers
/// do not steal each other's completions.
static DEVICES: Spin<Vec<Hid>> = Spin::new(Vec::new());

/// Devices that offered HID but not the boot protocol, so `usb hid` can say
/// what it declined instead of leaving a keyboard mysteriously dead.
static DECLINED: Racy<Vec<(u16, u16)>> = Racy::new(Vec::new());

/// What the device manager shows of one USB input device.
pub struct Live {
    pub vid: u16,
    pub pid: u16,
    pub port: u8,
    pub ifaces: Vec<LiveIface>,
    pub other: u8,
}

pub struct LiveIface {
    pub mouse: bool,
    pub reports: u64,
    pub failures: u64,
    pub recovered: u64,
}

/// Every USB input device held now. Waits for a walk in progress rather
/// than skipping it, because an answer missing a device is the stale
/// inventory the manager exists to replace.
pub fn live() -> Vec<Live> {
    DEVICES
        .lock()
        .iter()
        .map(|h| Live {
            vid: h.dev.vid,
            pid: h.dev.pid,
            port: h.dev.port,
            ifaces: h
                .ifaces
                .iter()
                .map(|i| LiveIface { mouse: i.protocol == MOUSE, reports: i.reports, failures: i.failures, recovered: i.recovered })
                .collect(),
            other: h.other,
        })
        .collect()
}

pub fn attached() -> usize {
    DEVICES.lock().len()
}

/// Find and configure every boot-protocol keyboard and mouse on the bus.
///
/// Safe to call when the controller is already running and some other driver
/// owns a device on it: ports another driver claimed are skipped, because
/// enumerating one twice hands it a second slot and leaves the first driver
/// ringing a doorbell the controller no longer associates with it.
pub fn probe(ecam: u64) -> Result<usize, &'static str> {
    xhci::ensure_started(ecam)?;
    ECAM.store(ecam, core::sync::atomic::Ordering::Relaxed);
    let mut found = 0usize;
    // Taken before the controller and released after it, every time: the one
    // lock order this file has, and `poll` keeps the same one.
    let mut devices = DEVICES.lock();

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
        // Whatever the registry remembers on this port was a previous
        // device, or this one under a slot it no longer has.
        super::registry::forget_usb(port);
        xhci::note_device(&dev, (0, 0, 0));
        let mut claimed = false;

        for i in 0..dev.num_configs {
            let Some(Ok((buf, total))) = xhci::with_ctl(|c| c.config_descriptor(&mut dev, i))
            else {
                break;
            };
            let cfg = xhci::parse_config(buf, total);
            if cfg.hids.is_empty() {
                continue;
            }

            if xhci::with_ctl(|c| c.set_configuration(&mut dev, cfg.value))
                .unwrap_or(Err("no controller"))
                .is_err()
            {
                continue;
            }
            // Every boot interface, not the first: a mouse is often a
            // keyboard as well, and the half not driven is the half that
            // moves the pointer.
            let mut ifaces = Vec::new();
            for hid in cfg.hids.iter().copied() {
                match setup(&mut dev, hid) {
                    Ok(i) => ifaces.push(i),
                    Err(e) => crate::kprintln!(
                        "  hid    port {} interface {} refused setup: {}",
                        port, hid.number, e
                    ),
                }
            }
            if !ifaces.is_empty() {
                claimed = true;
                found += 1;
                // HID 0x03, boot subclass 0x01, and the protocol of the first
                // interface. Recorded on success only: a device whose setup
                // was refused is not a keyboard this machine has.
                xhci::note_device(&dev, (0x03, 0x01, ifaces[0].protocol));
                if ifaces.iter().any(|i| i.protocol == MOUSE) {
                    super::mouse::declare_present();
                }
                let mut entry = Hid {
                    dev: core::mem::replace(&mut dev, xhci::Device::placeholder()),
                    ifaces,
                    other: cfg.other_hid,
                };
                for i in entry.ifaces.iter_mut() {
                    i.armed = xhci::with_ctl(|c| c.arm_interrupt(&mut entry.dev, i.ep, i.buf, REPORT_LEN))
                        .unwrap_or(false);
                }
                devices.push(entry);
                super::manager::changed();
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
    // The connect changes up to now are the ones this walk just answered --
    // the boot's own arrivals among them -- so they are acknowledged here,
    // or the compositor's next look would read them as devices that left
    // and tear down what was just set up.
    drop(devices);
    let _ = xhci::with_ctl(|c| c.take_port_changes());
    Ok(found)
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

fn setup(dev: &mut Device, hid: HidIface) -> Result<Iface, &'static str> {
    if hid.ep_in.addr == 0 {
        return Err("no interrupt endpoint on the boot interface");
    }

    // A mouse whose own report describes a wheel is left in report protocol
    // and decoded by that description. The boot report is three bytes by the
    // spec, and a fourth carrying the wheel is a courtesy most mice do not
    // extend once they are asked for boot reports -- which is why the wheel
    // never turned anything on a USB mouse. Anything the reader cannot place
    // X, Y and a wheel in goes to the boot report exactly as before.
    let layout = if hid.protocol == MOUSE { read_layout(dev, hid.number).filter(|l| l.wheel.size > 0) } else { None };
    let layout = match layout {
        // SET_PROTOCOL(1), said rather than assumed: firmware with legacy USB
        // support leaves a mouse in boot protocol, and a device answers in
        // whichever it was last told.
        Some(l) => match xhci::with_ctl(|c| c.control(dev, 0x21, SET_PROTOCOL, 1, hid.number as u16, 0, 0)) {
            Some(Ok(_)) => Some(l),
            _ => None,
        },
        None => None,
    };

    // SET_PROTOCOL(0) is what actually asks for boot reports. A device that
    // supports the boot subclass still powers up in *report* protocol, so
    // skipping this gets a device-defined report that happens to look right on
    // simple keyboards and does not on anything else.
    if layout.is_none() {
        xhci::with_ctl(|c| {
            c.control(dev, 0x21, SET_PROTOCOL, 0, hid.number as u16, 0, 0)
        })
        .unwrap_or(Err("no controller"))?;
    }

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
    Ok(Iface {
        protocol: hid.protocol,
        ep: hid.ep_in.addr,
        buf,
        armed: false,
        last: [0; 8],
        layout,
        reports: 0,
        failures: 0,
        recovered: 0,
    })
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
/// Called from the compositor's loop beside the pointer, because that loop
/// runs whatever the shell is doing; it was the shell's idle loop, and a USB
/// mouse therefore stopped for the length of every command -- which read as
/// the machine freezing, since nothing else on screen moves during one either.
/// There is still no interrupt-driven USB in this kernel, so anything that
/// wants to hear from the bus has to ask. A miner image has no compositor and
/// its shell asks instead.
///
/// `try_lock`: a probe in progress holds the list, and waiting on it from the
/// task that paints the screen would stop the screen for the length of a bus
/// walk.
pub fn poll() {
    let rescan = {
        let Some(mut devices) = DEVICES.try_lock() else { return };
        if devices.is_empty() && ECAM.load(core::sync::atomic::Ordering::Relaxed) == 0 {
            return;
        }
        let rescan = watch_ports(&mut devices);
        for h in devices.iter_mut() {
            poll_one(h);
        }
        rescan
    };
    // Asked for, not done: enumerating is control transfers with timeouts,
    // and this is the task that draws the screen. `service` does it.
    if rescan {
        RESCAN.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

/// A walk of the bus is owed.
static RESCAN: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// Seconds to wait before the next walk, doubled by every walk that found
/// nothing and reset by one that found something; and when that wait ends.
static BACKOFF_S: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
static NEXT_WALK: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// The clock task's, once a second: walk the bus if a port changed.
///
/// Backing off is what makes this safe to leave running. A port whose device
/// will not enumerate changes again with every attempt, and retrying at the
/// rate the ports are watched would spend the machine on one broken device;
/// so a walk that finds nothing doubles the wait, up to a minute.
pub fn service() {
    use core::sync::atomic::Ordering::Relaxed;
    if !RESCAN.load(Relaxed) {
        return;
    }
    let now = crate::dev::lapic::ticks();
    if now < NEXT_WALK.load(Relaxed) {
        return;
    }
    RESCAN.store(false, Relaxed);
    let ecam = ECAM.load(Relaxed);
    let found = probe(ecam).unwrap_or(0);
    let wait = if found > 0 {
        REJOINED.fetch_add(found as u64, Relaxed);
        crate::kprintln!("[usb] {} input device(s) on the bus", found);
        1
    } else {
        (BACKOFF_S.load(Relaxed) * 2).min(60)
    };
    BACKOFF_S.store(wait, Relaxed);
    NEXT_WALK.store(now + wait * crate::TIMER_HZ as u64, Relaxed);
}

/// A few times a second: has any port's connection changed? A device that
/// left takes its slot with it, and one that came back -- a touchpad whose
/// own controller resets itself does exactly this -- is a new device the old
/// slot does not describe. Either way the old entry goes, and the caller
/// walks the bus again. Answers whether to.
///
/// This was not looked at at all: a device that dropped off and came back
/// was polled forever through a slot the controller had stopped associating
/// with it, which reads as a pointer that died for no reason.
fn watch_ports(devices: &mut Vec<Hid>) -> bool {
    use core::sync::atomic::Ordering::Relaxed;
    let now = crate::time::rdtsc();
    let gap = crate::time::tsc_mhz().max(1) as u64 * 250_000;
    if now.saturating_sub(LAST_PORT_LOOK.load(Relaxed)) < gap {
        return false;
    }
    LAST_PORT_LOOK.store(now, Relaxed);
    let Some(mut changed) = xhci::with_ctl(|c| c.take_port_changes()) else { return false };
    // And every port a device is held on, whether or not it says it
    // changed: a port can go disabled without a connect change, and a held
    // device on a dead port is the pointer that silently stopped.
    for h in devices.iter() {
        if !changed.contains(&h.dev.port) && !xhci::with_ctl(|c| c.port_live(h.dev.port)).unwrap_or(true) {
            changed.push(h.dev.port);
        }
    }
    let mut rescan = false;
    for port in changed {
        if let Some(i) = devices.iter().position(|h| h.dev.port == port) {
            let h = devices.remove(i);
            crate::kprintln!(
                "[usb] {:04x}:{:04x} on port {} dropped off; finding it again",
                h.dev.vid,
                h.dev.pid,
                port
            );
            xhci::with_ctl(|c| c.release(h.dev));
            xhci::unclaim_port(port);
            super::registry::forget_usb(port);
            super::manager::changed();
            rescan = true;
        } else if !xhci::port_claimed(port) {
            // Something new on a port nobody holds: maybe a mouse.
            rescan = true;
        }
    }
    rescan
}

/// One device: a report from each interface if there is one, and an
/// interface's endpoint brought back if its transfer failed.
fn poll_one(h: &mut Hid) {
    let (vid, pid) = (h.dev.vid, h.dev.pid);
    for i in h.ifaces.iter_mut() {
        poll_iface(&mut h.dev, i, vid, pid);
    }
}

fn poll_iface(dev: &mut Device, i: &mut Iface, vid: u16, pid: u16) {
    if !i.armed {
        i.armed = xhci::with_ctl(|c| c.arm_interrupt(dev, i.ep, i.buf, REPORT_LEN)).unwrap_or(false);
        return;
    }
    let n = match xhci::with_ctl(|c| c.take_interrupt(dev, i.ep, REPORT_LEN)) {
        Some(xhci::Rx::Data(n)) => n,
        Some(xhci::Rx::Failed(code)) => {
            // The transfer is over and the endpoint is probably halted, so
            // waiting for the next report would wait forever. Reset it and
            // queue a fresh one. Said out loud the first few times, so a
            // flaky device is visible rather than merely slow.
            i.failures += 1;
            let back = xhci::with_ctl(|c| c.recover_interrupt(dev, i.ep)).unwrap_or(false);
            if back {
                i.recovered += 1;
            }
            if i.failures <= 3 {
                crate::kprintln!(
                    "[usb] {:04x}:{:04x} transfer failed (code {}), endpoint {}",
                    vid,
                    pid,
                    code,
                    if back { "reset" } else { "did not reset" }
                );
            }
            i.armed = xhci::with_ctl(|c| c.arm_interrupt(dev, i.ep, i.buf, REPORT_LEN)).unwrap_or(false);
            return;
        }
        _ => return,
    };
    i.armed = false;
    i.reports += 1;
    // The first one, once, out loud. A device that enumerates and
    // configures perfectly and then never reports looks identical to one
    // that is working until somebody presses a key, and there is no way to
    // tell those apart from a boot log.
    if i.reports == 1 {
        crate::kprintln!(
            "[usb] first report from {:04x}:{:04x}, the {} is live",
            vid,
            pid,
            if i.protocol == KEYBOARD { "keyboard" } else { "mouse" }
        );
    }

    let mut wide = [0u8; REPORT_LEN as usize];
    for k in 0..wide.len().min(n as usize) {
        wide[k] = unsafe { core::ptr::read_volatile((i.buf + k as u64) as *const u8) };
    }
    let mut report = [0u8; 8];
    report.copy_from_slice(&wide[..8]);
    match (i.protocol, i.layout) {
        (KEYBOARD, _) => keyboard(i, &report),
        (MOUSE, Some(l)) => {
            if let Some((dx, dy, buttons, wheel)) = l.decode(&wide[..(n as usize).min(wide.len())]) {
                super::mouse::apply(dx, dy, buttons & 1 != 0, buttons & 2 != 0, wheel);
            }
        }
        (MOUSE, None) => mouse(&report, n),
        _ => {}
    }

    // Re-arm at once. A keyboard that is only listened to after being
    // asked drops the release of the key it just reported, and a key whose
    // release is lost is a key that stays down.
    i.armed = xhci::with_ctl(|c| c.arm_interrupt(dev, i.ep, i.buf, REPORT_LEN)).unwrap_or(false);
}

/// `usb hid reset`: let go of every input device and find them again. The
/// last resort when a pointer is dead and nothing above noticed why --
/// typed, so it needs no pointer.
pub fn reset() -> Result<usize, &'static str> {
    let ecam = ECAM.load(core::sync::atomic::Ordering::Relaxed);
    if ecam == 0 {
        return Err("the bus has not been walked this boot");
    }
    {
        let mut devices = DEVICES.lock();
        for h in devices.drain(..) {
            let port = h.dev.port;
            xhci::with_ctl(|c| c.release(h.dev));
            xhci::unclaim_port(port);
            super::registry::forget_usb(port);
            super::manager::changed();
        }
    }
    let _ = xhci::with_ctl(|c| c.take_port_changes());
    probe(ecam)
}

/// Modifier bit to scancode.
///
/// The right-hand control and alt keys are given their *left-hand* scancodes
/// deliberately. Their real codes are `E0`-prefixed, and `decode` ignores
/// releases on the extended path, so a right control sent faithfully would
/// latch on and never let go. `decode` already treats an extended right
/// control as a left one for the same reason; this keeps the release working
/// too.
///
/// The GUI keys are the exception, sent as their real `E0 5B`/`E0 5C`: that is
/// Super, and `decode` handles their releases on the extended path because
/// releasing Super alone is the gesture that opens Start. They were zero --
/// "no key for" -- so a USB keyboard had no Super at all, while the same key
/// on PS/2 had one. Entries carry `emit`'s `0x100` prefix flag.
const MODIFIERS: [u16; 8] = [
    0x1D,  // left control
    0x2A,  // left shift
    0x38,  // left alt
    0x15B, // left GUI: Super
    0x1D,  // right control
    0x36,  // right shift
    0x38,  // right alt
    0x15C, // right GUI: Super
];

fn keyboard(h: &mut Iface, report: &[u8; 8]) {
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
            emit(MODIFIERS[bit], after);
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

/// GET_DESCRIPTOR for a HID report descriptor, sent to the interface.
const REPORT_DESCRIPTOR: u16 = 0x2200;
/// Longer than any mouse's: a five-button mouse with a wheel and a tilt
/// describes itself in about eighty bytes, a gaming mouse in a few hundred.
const DESCRIPTOR_MAX: usize = 512;

/// One field of a report: where it starts, how wide each value is, and how
/// many values there are. `size` 0 is a field the descriptor never named.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Field {
    bit: u16,
    size: u8,
    count: u8,
}

/// Where a mouse keeps its buttons, its two axes and its wheel.
///
/// **This is not a HID stack, and the module doc's refusal of one still
/// stands.** It answers one question -- which bits of which report are the
/// buttons, X, Y and the wheel -- for one kind of device, and any descriptor it
/// cannot answer that for is handed back to the boot report, which is what
/// every mouse got before. The two failure shapes are therefore "no wheel",
/// as before, and never "a pointer that moves wrongly": `setup` keeps report
/// protocol only when all three of X, Y and a wheel were found.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Layout {
    /// The report ID carrying them, or 0 when the device uses none.
    id: u8,
    buttons: Field,
    x: Field,
    y: Field,
    wheel: Field,
}

impl Layout {
    /// Movement, buttons and wheel out of one report, in `mouse::apply`'s
    /// conventions: y and the wheel both count positive *downward*, where HID
    /// counts y downward and the wheel *upward*.
    ///
    /// `None` for a report of another ID -- a mouse with media keys sends
    /// those on its own ID through the same endpoint, and reading one as
    /// movement throws the pointer across the screen.
    fn decode(&self, report: &[u8]) -> Option<(i32, i32, u8, i32)> {
        let data = if self.id != 0 {
            if report.first() != Some(&self.id) {
                return None;
            }
            &report[1..]
        } else {
            report
        };
        let dx = bits(data, self.x, true)?;
        let dy = bits(data, self.y, true)?;
        let wheel = if self.wheel.size > 0 { bits(data, self.wheel, true).unwrap_or(0) } else { 0 };
        let mut buttons = 0u8;
        for k in 0..self.buttons.count.min(8) {
            let f = Field { bit: self.buttons.bit + k as u16 * self.buttons.size as u16, ..self.buttons };
            if bits(data, f, false).unwrap_or(0) != 0 {
                buttons |= 1 << k;
            }
        }
        // `apply` takes PS/2's upward y, and the wheel positive downward.
        Some((dx, -dy, buttons, -wheel))
    }
}

/// One value of `f` out of `data`, little-endian bit order as HID lays it.
/// `None` when the field runs past what the device sent.
fn bits(data: &[u8], f: Field, signed: bool) -> Option<i32> {
    let size = f.size as usize;
    if size == 0 || size > 32 {
        return None;
    }
    let start = f.bit as usize;
    if start + size > data.len() * 8 {
        return None;
    }
    let mut v: u64 = 0;
    for k in 0..size {
        let b = start + k;
        if data[b / 8] >> (b % 8) & 1 != 0 {
            v |= 1 << k;
        }
    }
    if signed && size < 64 && v >> (size - 1) & 1 != 0 {
        v |= !0u64 << size;
    }
    Some(v as i64 as i32)
}

/// Fetch and read an interface's report descriptor.
fn read_layout(dev: &mut Device, iface: u8) -> Option<Layout> {
    let lay = core::alloc::Layout::from_size_align(DESCRIPTOR_MAX, 64).ok()?;
    let buf = xhci::dma(DESCRIPTOR_MAX, 64)?;
    let got = xhci::with_ctl(|c| c.control(dev, 0x81, 6, REPORT_DESCRIPTOR, iface as u16, buf, DESCRIPTOR_MAX as u16));
    let out = match got {
        Some(Ok(n)) => {
            let n = (n as usize).min(DESCRIPTOR_MAX);
            let mut d = Vec::with_capacity(n);
            for k in 0..n {
                d.push(unsafe { core::ptr::read_volatile((buf + k as u64) as *const u8) });
            }
            parse_layout(&d)
        }
        _ => None,
    };
    unsafe { alloc::alloc::dealloc(buf as *mut u8, lay) };
    out
}

/// The report descriptor's short items, walked once with the one piece of
/// state that matters: how many input bits each report ID has used so far.
///
/// Global items that are not page, size, count or ID are skipped, and so is
/// Push/Pop -- no mouse this has seen uses it, and a descriptor that does gets
/// offsets this walk cannot vouch for, which `parse_layout`'s caller survives
/// by falling back to the boot report rather than by trusting a guess.
fn parse_layout(d: &[u8]) -> Option<Layout> {
    const GD: u32 = 0x01;
    const BUTTON: u32 = 0x09;
    let (mut page, mut size, mut count, mut id) = (0u32, 0u32, 0u32, 0u8);
    let mut usages: Vec<u32> = Vec::new();
    let (mut umin, mut umax): (Option<u32>, Option<u32>) = (None, None);
    // Bits used per report ID; 0 is "no ID".
    let mut used: Vec<(u8, u32)> = Vec::new();
    let mut found: Vec<Layout> = Vec::new();
    let full = |raw: u32, wide: bool, page: u32| if wide { raw } else { page << 16 | raw };
    let mut i = 0usize;
    while i < d.len() {
        let b = d[i];
        if b == 0xFE {
            // A long item: data size in the next byte, and nothing a mouse uses.
            let n = *d.get(i + 1)? as usize;
            i += 3 + n;
            continue;
        }
        let n = [0usize, 1, 2, 4][(b & 3) as usize];
        if i + 1 + n > d.len() {
            return None;
        }
        let mut data = 0u32;
        for k in 0..n {
            data |= (d[i + 1 + k] as u32) << (8 * k);
        }
        let (kind, tag) = ((b >> 2) & 3, b >> 4);
        match (kind, tag) {
            // Input
            (0, 0x8) => {
                let at = match used.iter_mut().find(|(r, _)| *r == id) {
                    Some(e) => e,
                    None => {
                        used.push((id, 0));
                        used.last_mut()?
                    }
                };
                let start = at.1;
                at.1 += size * count;
                let constant = data & 1 != 0;
                if !constant && size > 0 && size <= 32 && count > 0 {
                    let l = match found.iter_mut().find(|l| l.id == id) {
                        Some(l) => l,
                        None => {
                            found.push(Layout { id, ..Layout::default() });
                            found.last_mut()?
                        }
                    };
                    for k in 0..count {
                        let usage = if !usages.is_empty() {
                            usages[(k as usize).min(usages.len() - 1)]
                        } else if let Some(lo) = umin {
                            (lo + k).min(umax.unwrap_or(u32::MAX))
                        } else {
                            0
                        };
                        let f = Field { bit: (start + k * size) as u16, size: size as u8, count: 1 };
                        match (usage >> 16, usage & 0xFFFF) {
                            (BUTTON, _) if l.buttons.size == 0 => {
                                l.buttons = Field { count: count.min(8) as u8, ..f };
                            }
                            (GD, 0x30) if l.x.size == 0 => l.x = f,
                            (GD, 0x31) if l.y.size == 0 => l.y = f,
                            (GD, 0x38) if l.wheel.size == 0 => l.wheel = f,
                            _ => {}
                        }
                    }
                }
                usages.clear();
                (umin, umax) = (None, None);
            }
            // Output, Feature, Collection, End Collection: locals end here.
            (0, _) => {
                usages.clear();
                (umin, umax) = (None, None);
            }
            (1, 0x0) => page = data,
            (1, 0x7) => size = data,
            (1, 0x8) => id = data as u8,
            (1, 0x9) => count = data,
            (2, 0x0) => usages.push(full(data, n == 4, page)),
            (2, 0x1) => umin = Some(full(data, n == 4, page)),
            (2, 0x2) => umax = Some(full(data, n == 4, page)),
            _ => {}
        }
        i += 1 + n;
    }
    found.into_iter().find(|l| l.x.size > 0 && l.y.size > 0)
}

fn mouse(report: &[u8; 8], n: u32) {
    let buttons = report[0];
    let dx = report[1] as i8 as i32;
    // HID counts y downward, the way the screen does; PS/2 counts it upward
    // and `apply` takes PS/2's convention. Passed through raw, the pointer
    // went up when the mouse came toward you -- the trip's first finding
    // once a USB mouse moved at all.
    let dy = -(report[2] as i8 as i32);
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
    let devices = DEVICES.lock();
    let declined = unsafe { &*DECLINED.get() };
    if devices.is_empty() && declined.is_empty() {
        crate::kprintln!("  no USB input devices");
        return;
    }
    for h in devices.iter() {
        for i in h.ifaces.iter() {
            let kind = if i.protocol == KEYBOARD { "keyboard" } else { "mouse" };
            crate::kprintln!(
                "  {:04x}:{:04x}  {}  port {}  slot {}  {} report(s), {} failed, {} recovered{}{}",
                h.dev.vid,
                h.dev.pid,
                kind,
                h.dev.port,
                h.dev.slot,
                i.reports,
                i.failures,
                i.recovered,
                match (i.protocol, i.layout) {
                    (MOUSE, Some(_)) => "  report protocol, wheel",
                    (MOUSE, None) => "  boot report, no wheel",
                    _ => "",
                },
                if i.armed { "" } else { "  (not armed)" }
            );
        }
        if h.other > 0 {
            crate::kprintln!(
                "  {:04x}:{:04x}  and {} HID interface(s) with no boot form, not driven",
                h.dev.vid,
                h.dev.pid,
                h.other
            );
        }
    }
    let back = REJOINED.load(core::sync::atomic::Ordering::Relaxed);
    if back > 0 {
        crate::kprintln!("  {} device(s) left the bus and were found again", back);
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

    // The report-descriptor reader, against the two shapes a mouse comes in.
    // A plain wheel mouse: no report ID, three buttons padded to a byte, then
    // X, Y and the wheel as signed bytes.
    const PLAIN: [u8; 52] = [
        0x05, 0x01, 0x09, 0x02, 0xA1, 0x01, 0x09, 0x01, 0xA1, 0x00, 0x05, 0x09, 0x19, 0x01, 0x29,
        0x03, 0x15, 0x00, 0x25, 0x01, 0x95, 0x03, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x05,
        0x81, 0x01, 0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x09, 0x38, 0x15, 0x81, 0x25, 0x7F, 0x75,
        0x08, 0x95, 0x03, 0x81, 0x06, 0xC0, 0xC0,
    ];
    let plain = parse_layout(&PLAIN);
    claim(
        &mut ok,
        plain.map(|l| (l.id, l.buttons.count, l.x.bit, l.y.bit, l.wheel.bit, l.wheel.size)) == Some((0, 3, 8, 16, 24, 8)),
        "a plain wheel mouse: three buttons, then X, Y and the wheel a byte each",
    );
    // A gaming mouse: report ID 1, five buttons, 16-bit axes, a wheel byte.
    const GAMING: [u8; 66] = [
        0x05, 0x01, 0x09, 0x02, 0xA1, 0x01, 0x85, 0x01, 0x09, 0x01, 0xA1, 0x00, 0x05, 0x09, 0x19,
        0x01, 0x29, 0x05, 0x15, 0x00, 0x25, 0x01, 0x95, 0x05, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01,
        0x75, 0x03, 0x81, 0x01, 0x05, 0x01, 0x16, 0x01, 0x80, 0x26, 0xFF, 0x7F, 0x75, 0x10, 0x95,
        0x02, 0x09, 0x30, 0x09, 0x31, 0x81, 0x06, 0x15, 0x81, 0x25, 0x7F, 0x75, 0x08, 0x95, 0x01,
        0x09, 0x38, 0x81, 0x06, 0xC0, 0xC0,
    ];
    let gaming = parse_layout(&GAMING);
    claim(
        &mut ok,
        gaming.map(|l| (l.id, l.x.size, l.y.bit, l.wheel.bit)) == Some((1, 16, 24, 40)),
        "a gaming mouse: report ID 1, sixteen-bit axes, the wheel after them",
    );
    // Left held, 5 right, 3 up the desk, one notch away from the user.
    let moved = gaming.and_then(|l| l.decode(&[0x01, 0x01, 0x05, 0x00, 0xFD, 0xFF, 0x01]));
    claim(
        &mut ok,
        moved == Some((5, 3, 1, -1)),
        "and its report reads in apply's terms: y up, a notch away is negative",
    );
    claim(
        &mut ok,
        gaming.and_then(|l| l.decode(&[0x02, 0x01, 0x05, 0x00, 0xFD, 0xFF, 0x01])).is_none(),
        "a report under another ID is not read as movement",
    );
    // No wheel described: X and Y are found and the wheel is not, which is
    // the shape `setup` sends back to the boot report.
    let mut wheelless = PLAIN;
    wheelless[38] = 0x09;
    wheelless[39] = 0x00; // usage 0x38 becomes undefined
    claim(
        &mut ok,
        parse_layout(&wheelless).map(|l| l.wheel.size) == Some(0),
        "a mouse that describes no wheel has none, and stays on the boot report",
    );
    claim(&mut ok, parse_layout(&PLAIN[..20]).is_none(), "and a truncated descriptor answers nothing");

    // A boot report is state, so the same report twice must produce no
    // events. Checked through the real path, with nothing attached: what is
    // being claimed is the diffing, and `last` is where it lives.
    let mut h = Iface {
        protocol: KEYBOARD,
        ep: 0,
        buf: 0,
        armed: false,
        last: [0; 8],
        layout: None,
        reports: 0,
        failures: 0,
        recovered: 0,
    };
    // The compositor would otherwise route what this injects, as keystrokes.
    crate::dev::kbd::pause_routing(true);
    while crate::dev::kbd::pop().is_some() {}
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

    // Super, through the modifier byte: bit 3 is the left GUI key.
    keyboard(&mut h, &[0; 8]);
    while crate::dev::kbd::pop().is_some() {}
    keyboard(&mut h, &[0x08, 0, 0x07, 0, 0, 0, 0, 0]); // Super, then 'd'
    keyboard(&mut h, &[0x08, 0, 0, 0, 0, 0, 0, 0]);
    keyboard(&mut h, &[0; 8]);
    let mut got = alloc::vec::Vec::new();
    while let Some(b) = crate::dev::kbd::pop() {
        got.push(b);
    }
    claim(
        &mut ok,
        got == [crate::dev::kbd::super_chord(b'd', false).unwrap_or(0)],
        "the GUI modifier is Super: Super-D arrives as its chord and nothing else",
    );
    keyboard(&mut h, &[0x80, 0, 0, 0, 0, 0, 0, 0]); // right GUI, alone
    keyboard(&mut h, &[0; 8]);
    let tap = crate::dev::kbd::pop();
    claim(
        &mut ok,
        tap == Some(crate::dev::kbd::KEY_STARTMENU) && crate::dev::kbd::pop().is_none(),
        "and Super tapped alone opens Start",
    );
    while crate::dev::kbd::pop().is_some() {}
    crate::dev::kbd::pause_routing(false);
    ok
}
