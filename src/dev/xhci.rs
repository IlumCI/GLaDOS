//! xHCI: the USB 3 host controller.
//!
//! The first step toward a USB stack, which is the only route this machine has
//! to a wireless dongle. The built-in card is CNVi -- the MAC lives in the PCH
//! and the M.2 module is a radio -- and needs an undocumented signed-firmware
//! protocol. xHCI is the opposite kind of problem: laborious, and *specified*.
//! Intel publishes the register layout and the ring formats, so every question
//! here has an answer somewhere rather than requiring another driver to be
//! read as a spec.
//!
//! UEFI has a working USB stack and we throw it away at `ExitBootServices`.
//! That is the cost of being the kernel rather than a guest of the firmware,
//! and it is not recoverable: boot services are gone, and the protocols that
//! pointed into them are gone with them.
//!
//! ### What is here so far
//!
//! Discovery and the capability registers. That is deliberately where this
//! stops for now: the capability block says how many device slots and ports
//! the controller has and where the operational and runtime registers begin,
//! and every later step is sized by those numbers. Getting them onto the
//! screen -- from QEMU's `qemu-xhci` and from the GF63's real controller --
//! is what turns the rest from guesswork into arithmetic.
//!
//! ### Rings, and why they are the whole design
//!
//! xHCI is not register-poking like the e1000. Everything is a ring of 16-byte
//! Transfer Request Blocks shared with the controller, and each ring carries a
//! *cycle bit* that flips every time the producer wraps. That single bit is
//! how both sides know which entries are new without any other synchronisation
//! -- the controller consumes entries whose cycle matches its state and stops
//! at the first that does not. Getting it wrong does not corrupt anything; the
//! controller simply never sees the command, which presents as a device that
//! enumerates to silence.
//!
//! Three rings matter here: a command ring the driver writes and the
//! controller reads, an event ring the controller writes and the driver reads,
//! and one transfer ring per endpoint. Commands complete asynchronously by
//! posting a Command Completion event, so nothing is synchronous even though
//! it reads that way.
//!
//! ### Memory
//!
//! Every structure the controller touches is allocated from the kernel heap,
//! which comes from identity-mapped physical frames -- so a heap address *is*
//! its physical address and no translation is needed anywhere below. That is
//! true because of how `init_heap` gets its memory, and it is the assumption
//! that would break first if this kernel ever grew a higher-half mapping.
//!
//! None of it is ever freed. A host controller lives as long as the machine.

use super::pci;
use crate::time::delay_us;
use alloc::vec::Vec;
use core::ptr::write_volatile;
use crate::mem::paging;
use core::ptr::read_volatile;

/// PCI class 0x0C subclass 0x03 is a USB controller; prog-if 0x30 is xHCI
/// specifically. The earlier interfaces (0x00 UHCI, 0x10 OHCI, 0x20 EHCI) are
/// different controllers entirely and are not driven here -- on this laptop
/// everything is routed through xHCI anyway, which is what USB 3 requires.
const CLASS_SERIAL_BUS: u8 = 0x0C;
const SUBCLASS_USB: u8 = 0x03;
const PROGIF_XHCI: u8 = 0x30;

/// Capability register offsets, from the base of the MMIO block.
const CAPLENGTH: u64 = 0x00;
const HCIVERSION: u64 = 0x02;
const HCSPARAMS1: u64 = 0x04;
const HCSPARAMS2: u64 = 0x08;
const HCCPARAMS1: u64 = 0x10;
const DBOFF: u64 = 0x14;
const RTSOFF: u64 = 0x18;

pub struct Caps {
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
    pub vendor: u16,
    pub device: u16,
    /// Base of the MMIO block, already mapped uncacheable.
    pub base: u64,
    /// Where the operational registers start, relative to `base`.
    pub op_off: u64,
    /// Doorbell array and runtime registers, likewise relative.
    pub db_off: u64,
    pub rt_off: u64,
    pub version: u16,
    pub max_slots: u8,
    pub max_intrs: u16,
    pub max_ports: u8,
    /// 64-byte context structures rather than 32. The controller decides this
    /// and every context the driver builds has to match, so it is read here
    /// rather than assumed -- guessing wrong misaligns every field past the
    /// first and produces a controller that accepts commands and does nothing.
    pub ctx64: bool,
    /// 64-bit addressing. If clear, every ring and context must live below
    /// 4 GiB, which constrains the allocator rather than the driver.
    pub ac64: bool,
}

#[derive(Debug)]
pub enum InitError {
    NotFound,
    NoBar,
    NoMap,
    NoMem,
    ResetTimeout,
}

/// Find the first xHCI controller and read what it says about itself.
pub fn probe(ecam: u64) -> Result<Caps, InitError> {
    let mut found: Option<pci::Device> = None;
    pci::scan(ecam, 255, |d| {
        if d.class == CLASS_SERIAL_BUS
            && d.subclass == SUBCLASS_USB
            && d.prog_if == PROGIF_XHCI
            && found.is_none()
        {
            found = Some(d);
        }
    });
    let dev = found.ok_or(InitError::NotFound)?;

    let bar = pci::bar(ecam, &dev, 0).ok_or(InitError::NoBar)?;
    if bar == 0 {
        return Err(InitError::NoBar);
    }
    // Device memory, and generally outside the boot-time identity map, so it
    // has to be mapped uncacheable before the first register read -- the same
    // reasoning as the NVMe and e1000 BARs.
    if !paging::map_range(bar, 0x20000, true) {
        return Err(InitError::NoMap);
    }
    // The controller cannot touch host memory until it is a bus master, and a
    // controller that silently never DMAs is the failure this prevents. Set it
    // here rather than later: it costs nothing and the rings are useless
    // without it.
    pci::enable_bus_master(ecam, &dev);

    // CAPLENGTH and HCIVERSION are two halves of one 32-bit register, and this
    // block only answers dword reads: a 16-bit read at offset 2 returns zero
    // rather than the version, which showed up as a controller claiming to
    // implement xHCI 0.0. Read the dword and split it.
    let cap0 = unsafe { read_volatile((bar + CAPLENGTH) as *const u32) };
    let cap_len = (cap0 & 0xFF) as u64;
    let version = (cap0 >> 16) as u16;
    let _ = HCIVERSION;
    let hcs1 = unsafe { read_volatile((bar + HCSPARAMS1) as *const u32) };
    let _hcs2 = unsafe { read_volatile((bar + HCSPARAMS2) as *const u32) };
    let hcc1 = unsafe { read_volatile((bar + HCCPARAMS1) as *const u32) };
    let db = unsafe { read_volatile((bar + DBOFF) as *const u32) } as u64;
    let rt = unsafe { read_volatile((bar + RTSOFF) as *const u32) } as u64;

    Ok(Caps {
        bus: dev.bus,
        dev: dev.dev,
        func: dev.func,
        vendor: dev.vendor,
        device: dev.device,
        base: bar,
        op_off: cap_len,
        // The low bits of both offsets are reserved and must be masked, not
        // merely ignored: a doorbell array addressed two bytes off is a write
        // into a neighbouring register.
        db_off: db & !0x3,
        rt_off: rt & !0x1F,
        version,
        max_slots: (hcs1 & 0xFF) as u8,
        max_intrs: ((hcs1 >> 8) & 0x7FF) as u16,
        max_ports: ((hcs1 >> 24) & 0xFF) as u8,
        ctx64: hcc1 & (1 << 2) != 0,
        ac64: hcc1 & 1 != 0,
    })
}

/// Set once networking owns the controller.
///
/// Starting a second `Controller` on the same hardware resets it, which drops
/// every configured device -- so running `usb` after boot took eth0 down and
/// left it looking like a driver regression rather than like a command with a
/// side effect. There is no controller registry to consult, and one flag is
/// enough for the single-controller case this machine actually has.
static CLAIMED: crate::sync::Racy<bool> = crate::sync::Racy::new(false);

/// What the last USB enumeration saw in the way of wireless hardware.
///
/// Recorded when a scan happens rather than looked up when asked, because
/// asking costs a bus reset: enumerating the controller drops whatever link
/// is running on it. A settings page rebuilt on every navigation cannot pay
/// that, so the enumeration that already happens at boot leaves its answer
/// here for anything that wants to know.
///
/// The two layers of `Option` are the whole point. The outer `None` means no
/// enumeration has ever run, so nothing is known. `Some(None)` means one ran
/// and found no wireless device. Those are different claims, and a page that
/// collapses them tells an operator with an adapter plugged in that they have
/// no adapter.
static USB_WIRELESS: crate::sync::Racy<Option<Option<(u16, u16, &'static str)>>> =
    crate::sync::Racy::new(None);

pub fn usb_wireless() -> Option<Option<(u16, u16, &'static str)>> {
    unsafe { *USB_WIRELESS.get() }
}

/// The USB ethernet adapter driving `eth0`, if one is.
///
/// Recorded for the same reason as the wireless one and with more urgency: on
/// a machine with no PCI network card this is the part carrying every packet,
/// and a hardware inventory that omitted it listed nothing while the operator
/// was reading it over the network.
static USB_ETHERNET: crate::sync::Racy<Option<(u16, u16)>> = crate::sync::Racy::new(None);

pub fn usb_ethernet() -> Option<(u16, u16)> {
    unsafe { *USB_ETHERNET.get() }
}

/// A scan is starting: forget what the last one found.
fn usb_scan_begin() {
    unsafe { *USB_WIRELESS.get() = Some(None) };
}

/// One enumerated device, checked against the wireless id list.
fn usb_note(vid: u16, pid: u16) {
    if let Some(name) = super::rtl8188eu::identify(vid, pid) {
        unsafe { *USB_WIRELESS.get() = Some(Some((vid, pid, name))) };
    }
}

/// What `usb` prints.
pub fn report(ecam: u64) {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[usb]");
    console::set_color(LTGRAY);
    if unsafe { *CLAIMED.get() } {
        kprintln!("  eth0 is driving the controller -- scanning would reset it");
        kprintln!("  and drop the link. Nothing to do that would not break more.");
        return;
    }
    console::set_color(LTGRAY);

    match probe(ecam) {
        Err(e) => {
            console::set_color(LTRED);
            kprintln!("  no xHCI controller ({:?})", e);
            console::set_color(WHITE);
            kprintln!("  QEMU needs '-device qemu-xhci'; the GF63 has one on the chipset");
        }
        Ok(c) => {
            kprintln!(
                "  xhci {:04x}:{:04x} at {:02x}:{:02x}.{}  bar {:#x}",
                c.vendor, c.device, c.bus, c.dev, c.func, c.base
            );
            kprintln!(
                "  version {:x}.{:x}  slots {}  ports {}  interrupters {}",
                c.version >> 8,
                (c.version >> 4) & 0xF,
                c.max_slots,
                c.max_ports,
                c.max_intrs
            );
            kprintln!(
                "  op +{:#x}  doorbells +{:#x}  runtime +{:#x}",
                c.op_off, c.db_off, c.rt_off
            );
            kprintln!(
                "  {}-byte contexts, {}-bit addressing",
                if c.ctx64 { 64 } else { 32 },
                if c.ac64 { 64 } else { 32 }
            );
            console::set_color(WHITE);

            let mut ctl = match Controller::start(&c) {
                Ok(x) => x,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  controller did not start: {:?}", e);
                    console::set_color(WHITE);
                    return;
                }
            };
            let ports = ctl.connected();
            usb_scan_begin();
            if ports.is_empty() {
                kprintln!("  running; no devices attached");
                return;
            }
            for p in ports {
                let mut dev = match ctl.enumerate(p) {
                    Ok(d) => d,
                    Err(e) => {
                        console::set_color(LTRED);
                        kprintln!("  port {}  {}", p, e);
                        console::set_color(WHITE);
                        continue;
                    }
                };
                console::set_color(LTGREEN);
                kprintln!("  port {}  {:04x}:{:04x}  slot {}  {} config(s)", p, dev.vid, dev.pid, dev.slot, dev.num_configs);
                console::set_color(LTGRAY);
                // A vendor-specific interface has no class code to key off, so
                // the id list is the whole of the detection.
                usb_note(dev.vid, dev.pid);
                if let Some(name) = super::rtl8188eu::identify(dev.vid, dev.pid) {
                    kprintln!("    {} -- wireless", name);
                    // One register read is the whole point of getting this far:
                    // REG_SYS_CFG is readable from reset, so a plausible answer
                    // proves rings, enumeration, control transfers and vendor
                    // requests all at once, and a garbage one says the fault is
                    // below the driver rather than in its tables.
                    match dma(64, 16) {
                        None => kprintln!("    no memory for a register read"),
                        Some(scratch) => {
                            let mut regs =
                                super::rtl8188eu::Regs::new(&mut ctl, &mut dev, scratch);
                            match regs.chip_id() {
                                Err(e) => {
                                    console::set_color(LTRED);
                                    kprintln!("    chip id unreadable: {}", e);
                                    console::set_color(LTGRAY);
                                }
                                Ok(id) => {
                                    kprintln!(
                                        "    sys_cfg 0x{:08x}  version {}  {}{}",
                                        id.raw, id.version,
                                        if id.vendor_umc { "UMC" } else { "TSMC" },
                                        if id.test_chip { "  TEST CHIP -- suspect read" }
                                        else { "" }
                                    );
                                    // This writes to the chip, which `usb` is
                                    // otherwise too passive a name for -- but
                                    // scanning already reset the bus, nothing
                                    // else is using this device, and it is the
                                    // only place the sequence can be run at
                                    // all. Said out loud rather than done
                                    // quietly.
                                    kprintln!("    powering on and loading the MAC table...");
                                    match regs.bring_up() {
                                        Ok(()) => {
                                            console::set_color(LTGREEN);
                                            kprintln!("    MAC up -- power sequence and 92 registers accepted");
                                            console::set_color(LTGRAY);
                                            kprintln!("    (PHY, radio and firmware are not written yet)");
                                        }
                                        Err(e) => {
                                            console::set_color(LTRED);
                                            kprintln!("    bring-up failed: {}", e);
                                            console::set_color(LTGRAY);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let mut best: Option<Config> = None;
                // Every configuration, because the interesting one is rarely
                // the first: QEMU's usb-net puts RNDIS on configuration 1 and
                // CDC Ethernet on 2, and only the second is worth driving.
                for i in 0..dev.num_configs {
                    let (buf, total) = match ctl.config_descriptor(&mut dev, i) {
                        Ok(x) => x,
                        Err(_) => break,
                    };
                    let c = parse_config(buf, total);
                    kprintln!(
                        "    config {}  {} interface(s)  {} bytes{}",
                        c.value,
                        c.interfaces,
                        total,
                        if c.ecm { "  CDC data" } else { "" }
                    );
                    // Every descriptor is length-prefixed, so walking the
                    // records and printing type and length says what the device
                    // actually offers -- guessing which functional descriptors
                    // a config carries from its total size is how the MAC came
                    // back empty from a config that had no room for one.
                    let mut o = 0usize;
                    while o + 2 <= total {
                        let ln = unsafe { read_volatile((buf + o as u64) as *const u8) } as usize;
                        let ty = unsafe { read_volatile((buf + o as u64 + 1) as *const u8) };
                        if ln == 0 {
                            break;
                        }
                        let sub = if ty == 0x24 && o + 3 <= total {
                            unsafe { read_volatile((buf + o as u64 + 2) as *const u8) }
                        } else {
                            0xFF
                        };
                        kprintln!(
                            "      desc type 0x{:02x} len {}{}",
                            ty, ln,
                            if sub != 0xFF { alloc::format!("  subtype 0x{:02x}", sub) }
                            else { alloc::string::String::new() }
                        );
                        o += ln;
                    }
                    kprintln!(
                        "      imac string index {}", c.imac
                    );
                    if let Some((n, alt)) = c.data_iface {
                        kprintln!("      data interface {} alt {}", n, alt);
                    }
                    for (label, ep) in [("in", c.bulk_in), ("out", c.bulk_out)] {
                        if let Some(e) = ep {
                            kprintln!(
                                "      bulk {} ep {:#04x}  max packet {}",
                                label, e.addr, e.max_packet
                            );
                        }
                    }
                    if best.is_none() && c.ecm && c.bulk_in.is_some() && c.bulk_out.is_some() {
                        best = Some(c);
                    }
                }

                // Bring the CDC Ethernet configuration up and move a frame.
                if let Some(c) = best {
                    let (iface, alt) = c.data_iface.unwrap_or((0, 0));
                    let r = ctl
                        .set_configuration(&mut dev, c.value)
                        .and_then(|_| {
                            // Only when the endpoints live on a non-zero
                            // alternate setting. Sending SET_INTERFACE(0) to a
                            // device with no alternates stalls on some.
                            if alt > 0 {
                                ctl.set_interface(&mut dev, iface, alt)
                            } else {
                                Ok(())
                            }
                        })
                        .and_then(|_| {
                            ctl.configure_bulk(&mut dev, c.bulk_in.unwrap(), c.bulk_out.unwrap())
                        });
                    match r {
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("    bring-up failed: {}", e);
                            console::set_color(WHITE);
                        }
                        Ok(()) => {
                            console::set_color(LTGREEN);
                            kprintln!("    configuration {} up, bulk endpoints ready", c.value);
                            console::set_color(LTGRAY);
                            // A broadcast ARP-shaped frame. The point is the
                            // transfer completing, not the reply: this proves
                            // TRBs reach the device and the controller reports
                            // the bytes it moved.
                            if let Some(tx) = dma(64, 16) {
                                unsafe {
                                    for i in 0..6u64 {
                                        write_volatile((tx + i) as *mut u8, 0xFF);
                                    }
                                    write_volatile((tx + 12) as *mut u16, 0x0608);
                                }
                                match ctl.bulk_out(&mut dev, tx, 60) {
                                    Ok(n) => kprintln!("    bulk out moved {} bytes", n),
                                    Err(e) => {
                                        console::set_color(LTRED);
                                        kprintln!("    bulk out: {}", e);
                                        console::set_color(LTGRAY);
                                    }
                                }
                            }
                            if let Some(rx) = dma(1536, 16) {
                                match ctl.bulk_in(&mut dev, rx, 1536, 300) {
                                    Ok(n) => kprintln!("    bulk in received {} bytes", n),
                                    Err(e) => kprintln!("    bulk in: {}", e),
                                }
                            }
                            console::set_color(WHITE);
                        }
                    }
                }
                console::set_color(WHITE);
            }
        }
    }
    console::set_color(WHITE);
}

// --- operational, runtime and doorbell registers --------------------------

const USBCMD: u64 = 0x00;
const USBSTS: u64 = 0x04;
const CRCR: u64 = 0x18;
const DCBAAP: u64 = 0x30;
const CONFIG: u64 = 0x38;
/// Port register sets begin here, 0x10 bytes each, one-based.
const PORTSC_BASE: u64 = 0x400;

const CMD_RS: u32 = 1 << 0;
const CMD_HCRST: u32 = 1 << 1;

const STS_HCH: u32 = 1 << 0;
const STS_CNR: u32 = 1 << 11;

/// Port status bits that clear when written as one. Every read-modify-write of
/// PORTSC has to mask these off, or it clears status it never meant to touch --
/// the classic way to lose a connect event while enabling a port.
const PORTSC_RW1C: u32 = (1 << 17) | (1 << 18) | (1 << 20) | (1 << 21) | (1 << 22) | (1 << 23);
const PORTSC_CCS: u32 = 1 << 0;
const PORTSC_PED: u32 = 1 << 1;
const PORTSC_PR: u32 = 1 << 4;
const PORTSC_PP: u32 = 1 << 9;
const PORTSC_PRC: u32 = 1 << 21;

/// Interrupter 0's registers, relative to the runtime base.
const IR0: u64 = 0x20;
const ERSTSZ: u64 = 0x08;
const ERSTBA: u64 = 0x10;
const ERDP: u64 = 0x18;

// TRB types this driver produces.
const TRB_LINK: u32 = 6;
const TRB_ENABLE_SLOT: u32 = 9;
const TRB_ADDRESS_DEVICE: u32 = 11;
const TRB_SETUP: u32 = 2;
const TRB_DATA: u32 = 3;
const TRB_STATUS: u32 = 4;
const TRB_NORMAL: u32 = 1;
const TRB_CONFIG_EP: u32 = 12;
// ...and the ones the controller posts back.
const TRB_TRANSFER_EVENT: u32 = 32;
const TRB_CMD_COMPLETE: u32 = 33;

/// TRBs per ring. Far more than this driver ever has outstanding, and it keeps
/// a ring inside one page.
const RING_TRBS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Trb {
    lo: u32,
    hi: u32,
    status: u32,
    control: u32,
}

impl Trb {
    /// Endpoint ID (a DCI) from a transfer event.
    fn endpoint(&self) -> u32 {
        (self.control >> 16) & 0x1F
    }

    fn kind(&self) -> u32 {
        (self.control >> 10) & 0x3F
    }
    /// Completion code from an event TRB. 1 is success; everything else is a
    /// reason.
    fn code(&self) -> u32 {
        self.status >> 24
    }
    fn slot(&self) -> u8 {
        (self.control >> 24) as u8
    }
}

/// Allocate zeroed, aligned memory the controller may read.
///
/// Never freed, by design -- see the module note. The address returned is
/// simultaneously the virtual and the physical one.
pub(crate) fn dma(size: usize, align: usize) -> Option<u64> {
    let layout = core::alloc::Layout::from_size_align(size, align).ok()?;
    let p = unsafe { alloc::alloc::alloc_zeroed(layout) };
    if p.is_null() {
        None
    } else {
        Some(p as u64)
    }
}

/// A producer ring: TRBs, plus the cycle state that says which are new.
struct Ring {
    base: u64,
    idx: usize,
    cycle: u32,
}

impl Ring {
    fn new() -> Option<Ring> {
        let base = dma(RING_TRBS * 16, 64)?;
        // The last TRB is a Link back to the start carrying Toggle Cycle,
        // which is what makes a fixed-size ring behave like an endless queue.
        let link = Trb {
            lo: base as u32,
            hi: (base >> 32) as u32,
            status: 0,
            control: (TRB_LINK << 10) | (1 << 1),
        };
        unsafe { write_volatile((base as *mut Trb).add(RING_TRBS - 1), link) };
        Some(Ring { base, idx: 0, cycle: 1 })
    }

    fn push(&mut self, mut t: Trb) {
        t.control = (t.control & !1) | self.cycle;
        unsafe { write_volatile((self.base as *mut Trb).add(self.idx), t) };
        self.idx += 1;
        if self.idx == RING_TRBS - 1 {
            // Give the Link TRB the current cycle so the controller follows
            // it, then flip: everything after the wrap is the other polarity.
            let ctrl = (TRB_LINK << 10) | (1 << 1) | self.cycle;
            unsafe {
                let p = (self.base as *mut Trb).add(RING_TRBS - 1);
                write_volatile(&mut (*p).control, ctrl);
            }
            self.idx = 0;
            self.cycle ^= 1;
        }
    }
}

pub struct Controller {
    op: u64,
    rt: u64,
    db: u64,
    max_ports: u8,
    ctx_bytes: usize,
    dcbaa: u64,
    cmd: Ring,
    ev_base: u64,
    ev_idx: usize,
    ev_cycle: u32,
    /// Transfer events belonging to an endpoint other than the one being
    /// waited on. See `wait_event_ep`.
    pending: Vec<Trb>,
}

impl Controller {
    #[inline]
    pub(crate) fn r32(&self, off: u64) -> u32 {
        unsafe { read_volatile((self.op + off) as *const u32) }
    }
    #[inline]
    fn w32(&self, off: u64, v: u32) {
        unsafe { write_volatile((self.op + off) as *mut u32, v) }
    }
    #[inline]
    fn w64(&self, off: u64, v: u64) {
        // Low half first: several of these registers latch on the high write,
        // so the order is part of the protocol rather than a preference.
        unsafe {
            write_volatile((self.op + off) as *mut u32, v as u32);
            write_volatile((self.op + off + 4) as *mut u32, (v >> 32) as u32);
        }
    }
    #[inline]
    pub(crate) fn portsc(&self, port: u8) -> u64 {
        PORTSC_BASE + (port as u64 - 1) * 0x10
    }
    #[inline]
    fn doorbell(&self, slot: u8, target: u32) {
        unsafe { write_volatile((self.db + slot as u64 * 4) as *mut u32, target) };
    }

    /// Reset, build the rings, and start the controller.
    pub fn start(caps: &Caps) -> Result<Controller, InitError> {
        let mut c = Controller {
            op: caps.base + caps.op_off,
            rt: caps.base + caps.rt_off,
            db: caps.base + caps.db_off,
            max_ports: caps.max_ports,
            ctx_bytes: if caps.ctx64 { 64 } else { 32 },
            dcbaa: 0,
            cmd: Ring::new().ok_or(InitError::NoMem)?,
            ev_base: 0,
            ev_idx: 0,
            ev_cycle: 1,
            pending: Vec::new(),
        };

        // Halt before resetting. Resetting a running controller is undefined,
        // and this one *is* running -- UEFI was using it a moment ago.
        c.w32(USBCMD, c.r32(USBCMD) & !CMD_RS);
        for _ in 0..1000 {
            if c.r32(USBSTS) & STS_HCH != 0 {
                break;
            }
            delay_us(1000);
        }
        c.w32(USBCMD, CMD_HCRST);
        // Two waits, not one. HCRST clears when the reset finishes; CNR clears
        // when the controller is willing to be written to at all. Programming
        // registers between those two moments is ignored silently.
        for _ in 0..1000 {
            if c.r32(USBCMD) & CMD_HCRST == 0 {
                break;
            }
            delay_us(1000);
        }
        for _ in 0..1000 {
            if c.r32(USBSTS) & STS_CNR == 0 {
                break;
            }
            delay_us(1000);
        }
        if c.r32(USBSTS) & STS_CNR != 0 {
            return Err(InitError::ResetTimeout);
        }

        // Device Context Base Address Array: one pointer per slot, plus entry
        // zero, which the controller owns.
        let n = caps.max_slots as usize + 1;
        c.dcbaa = dma(n * 8, 64).ok_or(InitError::NoMem)?;
        c.w32(CONFIG, (c.r32(CONFIG) & !0xFF) | caps.max_slots as u32);
        c.w64(DCBAAP, c.dcbaa);

        // Command ring. Bit 0 is the ring cycle state and must agree with the
        // ring's own producer cycle, or the first command is never seen.
        c.w64(CRCR, c.cmd.base | 1);

        // Event ring: a segment table of one entry, pointing at the ring.
        c.ev_base = dma(RING_TRBS * 16, 64).ok_or(InitError::NoMem)?;
        let erst = dma(16, 64).ok_or(InitError::NoMem)?;
        unsafe {
            write_volatile(erst as *mut u32, c.ev_base as u32);
            write_volatile((erst + 4) as *mut u32, (c.ev_base >> 32) as u32);
            write_volatile((erst + 8) as *mut u32, RING_TRBS as u32);
            write_volatile((erst + 12) as *mut u32, 0);
            write_volatile((c.rt + IR0 + ERSTSZ) as *mut u32, 1);
            // Dequeue pointer before the table base: the controller may start
            // using the ring the moment ERSTBA is written.
            write_volatile((c.rt + IR0 + ERDP) as *mut u32, c.ev_base as u32);
            write_volatile((c.rt + IR0 + ERDP + 4) as *mut u32, (c.ev_base >> 32) as u32);
            write_volatile((c.rt + IR0 + ERSTBA) as *mut u32, erst as u32);
            write_volatile((c.rt + IR0 + ERSTBA + 4) as *mut u32, (erst >> 32) as u32);
        }

        c.w32(USBCMD, c.r32(USBCMD) | CMD_RS);
        for _ in 0..1000 {
            if c.r32(USBSTS) & STS_HCH == 0 {
                return Ok(c);
            }
            delay_us(1000);
        }
        Err(InitError::ResetTimeout)
    }

    /// Take the next event, if the controller has posted one.
    fn poll_event(&mut self) -> Option<Trb> {
        let t = unsafe { read_volatile((self.ev_base as *const Trb).add(self.ev_idx)) };
        // The cycle bit is the entire handshake: an entry whose cycle does not
        // match ours has not been written this time round.
        if t.control & 1 != self.ev_cycle {
            return None;
        }
        self.ev_idx += 1;
        if self.ev_idx == RING_TRBS {
            self.ev_idx = 0;
            self.ev_cycle ^= 1;
        }
        let dq = self.ev_base + (self.ev_idx * 16) as u64;
        unsafe {
            // Bit 3 is Event Handler Busy, write-one-to-clear.
            write_volatile((self.rt + IR0 + ERDP) as *mut u32, (dq as u32) | (1 << 3));
            write_volatile((self.rt + IR0 + ERDP + 4) as *mut u32, (dq >> 32) as u32);
        }
        Some(t)
    }

    /// Wait for an event of a given type, discarding others.
    ///
    /// Port change events arrive unbidden throughout and are not errors.
    fn wait_event(&mut self, kind: u32, ms: u64) -> Option<Trb> {
        self.wait_event_ep(kind, 0, ms)
    }

    /// Wait for an event, optionally for one endpoint only.
    ///
    /// The event ring is shared by every endpoint on every slot, so matching on
    /// the TRB type alone is only correct while exactly one transfer is in
    /// flight. It stops being correct the moment a receive is left armed: a
    /// send then completes against whichever transfer event arrives first, so
    /// `bulk_out` reports success on the *receive*, and the frame that actually
    /// arrived is dropped with the endpoint still looking armed. That is what a
    /// DHCP that sends fine and never sees an offer looks like from here.
    ///
    /// An event for somebody else is therefore set aside rather than discarded.
    fn wait_event_ep(&mut self, kind: u32, ep: u32, ms: u64) -> Option<Trb> {
        let hit = |t: &Trb| t.kind() == kind && (ep == 0 || t.endpoint() == ep);
        if let Some(i) = self.pending.iter().position(hit) {
            return Some(self.pending.remove(i));
        }
        // Drain the ring before consulting the clock, so a zero timeout still
        // means "check" rather than "do nothing". It read as the latter, and a
        // non-blocking receive that never looks at the ring only sees frames
        // some other waiter happened to stash -- which is why DHCP completed
        // (it alternates send and receive) while ARP timed out.
        let mut waited = 0u64;
        loop {
            while let Some(t) = self.poll_event() {
                if hit(&t) {
                    return Some(t);
                }
                // Only transfer events are worth keeping: they are the ones
                // another waiter is owed. Port changes are informational and
                // command completions are always awaited synchronously.
                if t.kind() == TRB_TRANSFER_EVENT {
                    // Bounded, so a device that spews cannot grow this without
                    // limit. Dropping the oldest loses a completion, which
                    // costs a frame -- unbounded growth would cost the heap.
                    if self.pending.len() >= 16 {
                        self.pending.remove(0);
                    }
                    self.pending.push(t);
                }
            }
            if waited >= ms {
                return None;
            }
            delay_us(1000);
            waited += 1;
        }
    }

    fn command(&mut self, t: Trb, ms: u64) -> Option<Trb> {
        self.cmd.push(t);
        // Slot 0, target 0 is the command doorbell.
        self.doorbell(0, 0);
        self.wait_event(TRB_CMD_COMPLETE, ms)
    }

    /// Ports with something plugged into them.
    pub fn connected(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for port in 1..=self.max_ports {
            if self.r32(self.portsc(port)) & PORTSC_CCS != 0 {
                out.push(port);
            }
        }
        out
    }

    /// Reset one port and wait for it to enable.
    fn reset_port(&mut self, port: u8) -> bool {
        let off = self.portsc(port);
        let v = self.r32(off) & !PORTSC_RW1C;
        self.w32(off, v | PORTSC_PP | PORTSC_PR);
        for _ in 0..500 {
            let s = self.r32(off);
            if s & PORTSC_PRC != 0 {
                // Acknowledge the reset-change bit, and only that bit.
                self.w32(off, (s & !PORTSC_RW1C) | PORTSC_PRC);
                return s & PORTSC_PED != 0;
            }
            delay_us(1000);
        }
        false
    }

    /// A control transfer with an IN data stage.
    ///
    /// Extracted from `enumerate` the moment a second caller existed. The
    /// three stages are not optional and their order is the protocol: Setup
    /// carries the request in the TRB itself (Immediate Data), Data moves the
    /// bytes, Status is the handshake -- and only the last carries Interrupt
    /// On Completion, because one event per transfer is what the caller waits
    /// for.
    fn control_in(
        &mut self,
        dev: &mut Device,
        setup_lo: u32,
        setup_hi: u32,
        buf: u64,
        len: u32,
    ) -> Result<u32, &'static str> {
        dev.ep0.push(Trb {
            lo: setup_lo,
            hi: setup_hi,
            status: 8,
            control: (TRB_SETUP << 10) | (1 << 6) | (3 << 16),
        });
        dev.ep0.push(Trb {
            lo: buf as u32,
            hi: (buf >> 32) as u32,
            status: len,
            control: (TRB_DATA << 10) | (1 << 16),
        });
        dev.ep0.push(Trb {
            control: (TRB_STATUS << 10) | (1 << 5),
            ..Default::default()
        });
        self.doorbell(dev.slot, 1);

        // Endpoint zero is DCI 1, and saying so matters as soon as anything
        // else is in flight: a driver that reads registers over the control
        // pipe while a receive is armed would otherwise complete its register
        // read against an arriving frame. Same bug as the bulk path had.
        let ev = self
            .wait_event_ep(TRB_TRANSFER_EVENT, 1, 500)
            .ok_or("no response to control transfer")?;
        // 13 is Short Packet: fewer bytes than asked for, which for a
        // descriptor read is success rather than failure -- the device is
        // telling you how long the thing actually is.
        if ev.code() != 1 && ev.code() != 13 {
            return Err("control transfer failed");
        }
        // The event reports bytes *not* transferred, so the length is what was
        // asked for minus that.
        Ok(len.saturating_sub(ev.status & 0xFFFFFF))
    }

    /// A control transfer with a host-to-device data stage.
    ///
    /// The three differences from `control_in` are all direction bits and all
    /// silent if got wrong: the Setup TRB's transfer type is 2 (OUT) rather
    /// than 3, the Data TRB's direction bit is clear, and the Status stage runs
    /// the *opposite* way to the data -- so it is IN here and OUT there.
    fn control_out(
        &mut self,
        dev: &mut Device,
        setup_lo: u32,
        setup_hi: u32,
        buf: u64,
        len: u32,
    ) -> Result<(), &'static str> {
        dev.ep0.push(Trb {
            lo: setup_lo,
            hi: setup_hi,
            status: 8,
            control: (TRB_SETUP << 10) | (1 << 6) | (2 << 16),
        });
        if len > 0 {
            dev.ep0.push(Trb {
                lo: buf as u32,
                hi: (buf >> 32) as u32,
                status: len,
                control: TRB_DATA << 10,
            });
        }
        dev.ep0.push(Trb {
            control: (TRB_STATUS << 10) | (1 << 5) | (1 << 16),
            ..Default::default()
        });
        self.doorbell(dev.slot, 1);
        let ev = self
            .wait_event_ep(TRB_TRANSFER_EVENT, 1, 500)
            .ok_or("no response to control transfer")?;
        if ev.code() != 1 && ev.code() != 13 {
            return Err("control transfer failed");
        }
        Ok(())
    }

    /// A vendor-defined control transfer, which is how every Realtek USB part
    /// exposes its register file. `read` picks the direction; `value` is the
    /// register offset and `len` is 1, 2 or 4.
    ///
    /// Public because the register layout belongs to the device driver, not
    /// here -- this module knows how to move the bytes and nothing about what
    /// they mean.
    pub fn vendor(
        &mut self,
        dev: &mut Device,
        read: bool,
        request: u8,
        value: u16,
        index: u16,
        buf: u64,
        len: u16,
    ) -> Result<u32, &'static str> {
        // bmRequestType: bit 7 direction, bits 6-5 = 2 for vendor, bits 4-0 = 0
        // for a device recipient. 0xC0 in, 0x40 out.
        let rt: u32 = if read { 0xC0 } else { 0x40 };
        let lo = rt | ((request as u32) << 8) | ((value as u32) << 16);
        let hi = (index as u32) | ((len as u32) << 16);
        if read {
            self.control_in(dev, lo, hi, buf, len as u32)
        } else {
            self.control_out(dev, lo, hi, buf, len as u32).map(|_| len as u32)
        }
    }

    /// Read a descriptor into `buf`. `value` is the type and index, as the
    /// standard packs them: 0x0100 device, 0x0200 configuration.
    pub fn descriptor(
        &mut self,
        dev: &mut Device,
        value: u16,
        buf: u64,
        len: u16,
    ) -> Result<u32, &'static str> {
        // bmRequestType 0x80 (device to host), bRequest 6 (GET_DESCRIPTOR).
        let lo = 0x0680 | ((value as u32) << 16);
        self.control_in(dev, lo, (len as u32) << 16, buf, len as u32)
    }

    /// Enumerate one port far enough to talk to the device on it.
    pub fn enumerate(&mut self, port: u8) -> Result<Device, &'static str> {
        if !self.reset_port(port) {
            return Err("port did not enable after reset");
        }

        let ev = self
            .command(Trb { control: TRB_ENABLE_SLOT << 10, ..Default::default() }, 200)
            .ok_or("no response to Enable Slot")?;
        if ev.code() != 1 {
            return Err("Enable Slot refused");
        }
        let slot = ev.slot();

        // The input context is one context larger than the device context: an
        // Input Control Context at the front says which of the following
        // entries the controller should consume.
        let cb = self.ctx_bytes;
        let dev_ctx = dma(cb * 32, 64).ok_or("out of memory")?;
        let inp = dma(cb * 33, 64).ok_or("out of memory")?;
        unsafe {
            write_volatile((self.dcbaa + slot as u64 * 8) as *mut u64, dev_ctx);
            // Add-context flags: bit 0 the slot context, bit 1 endpoint zero.
            write_volatile((inp + 4) as *mut u32, 0b11);
        }

        let ring = Ring::new().ok_or("out of memory")?;
        let speed = (self.r32(self.portsc(port)) >> 10) & 0xF;
        // The default endpoint's maximum packet size is fixed by speed and is
        // not negotiable. Guessing high on a low-speed device makes the first
        // control transfer fail with nothing useful to look at.
        let mps: u32 = match speed {
            4 => 512,
            2 => 8,
            _ => 64,
        };

        let slot_ctx = inp + cb as u64;
        let ep0_ctx = inp + 2 * cb as u64;
        unsafe {
            // Route string zero, one context entry, speed, root hub port.
            write_volatile(slot_ctx as *mut u32, (1 << 27) | (speed << 20));
            write_volatile((slot_ctx + 4) as *mut u32, (port as u32) << 16);
            // EP0: control endpoint, three retries, and its transfer ring.
            write_volatile((ep0_ctx + 4) as *mut u32, (4 << 3) | (3 << 1) | (mps << 16));
            write_volatile((ep0_ctx + 8) as *mut u32, (ring.base as u32) | 1);
            write_volatile((ep0_ctx + 12) as *mut u32, (ring.base >> 32) as u32);
        }

        let ev = self
            .command(
                Trb {
                    lo: inp as u32,
                    hi: (inp >> 32) as u32,
                    control: (TRB_ADDRESS_DEVICE << 10) | ((slot as u32) << 24),
                    ..Default::default()
                },
                200,
            )
            .ok_or("no response to Address Device")?;
        if ev.code() != 1 {
            return Err("Address Device refused");
        }

        let mut dev = Device {
            slot,
            ep0: ring,
            vid: 0,
            pid: 0,
            num_configs: 0,
            inp,
            port,
            bulk_in: None,
            bulk_out: None,
        };
        let buf = dma(18, 16).ok_or("out of memory")?;
        self.descriptor(&mut dev, 0x0100, buf, 18)?;
        dev.vid = unsafe { read_volatile((buf + 8) as *const u16) };
        dev.pid = unsafe { read_volatile((buf + 10) as *const u16) };
        dev.num_configs = unsafe { read_volatile((buf + 17) as *const u8) };
        Ok(dev)
    }

    /// Read the configuration descriptor and everything that follows it.
    ///
    /// Two reads, not one: the first nine bytes say how long the whole block
    /// is, and only then can a buffer the right size be asked for. Asking for
    /// a fixed large length instead works on most devices and returns a stall
    /// on the ones that take wLength literally.
    pub fn config_descriptor(
        &mut self,
        dev: &mut Device,
        index: u8,
    ) -> Result<(u64, usize), &'static str> {
        let head = dma(9, 16).ok_or("out of memory")?;
        self.descriptor(dev, 0x0200 | index as u16, head, 9)?;
        let total = unsafe { read_volatile((head + 2) as *const u16) } as usize;
        if total < 9 || total > 4096 {
            return Err("implausible configuration descriptor length");
        }
        let buf = dma(total, 16).ok_or("out of memory")?;
        self.descriptor(dev, 0x0200 | index as u16, buf, total as u16)?;
        Ok((buf, total))
    }
}

impl Controller {
    /// Read a string descriptor as ASCII.
    ///
    /// USB strings are UTF-16LE after a two-byte header. Everything this
    /// driver reads from one is hex digits, so the high byte is dropped rather
    /// than decoded -- a real UTF-16 reader would be code with no second
    /// caller.
    pub fn string(&mut self, dev: &mut Device, index: u8) -> Result<alloc::string::String, &'static str> {
        if index == 0 {
            return Err("no such string");
        }
        let buf = dma(256, 16).ok_or("out of memory")?;
        // wIndex is the language id. 0x0409 is US English, and asking for a
        // language the device does not have gets a stall -- which halts
        // endpoint zero, so it is worth getting right first time.
        let lo = 0x0680 | ((0x0300u32 | index as u32) << 16);
        let n = self.control_in(dev, lo, 0x0409 | (255u32 << 16), buf, 255)? as usize;
        let mut s = alloc::string::String::new();
        let mut i = 2usize;
        while i + 1 < n {
            let c = unsafe { read_volatile((buf + i as u64) as *const u8) };
            if c.is_ascii_graphic() {
                s.push(c as char);
            }
            i += 2;
        }
        Ok(s)
    }

    /// The adapter's MAC, from the string the ECM descriptor points at.
    pub fn ecm_mac(&mut self, dev: &mut Device, imac: u8) -> Result<[u8; 6], &'static str> {
        let s = self.string(dev, imac)?;
        let b = s.as_bytes();
        if b.len() < 12 {
            return Err("MAC string too short");
        }
        let hex = |c: u8| -> Option<u8> {
            match c {
                b'0'..=b'9' => Some(c - b'0'),
                b'a'..=b'f' => Some(c - b'a' + 10),
                b'A'..=b'F' => Some(c - b'A' + 10),
                _ => None,
            }
        };
        let mut mac = [0u8; 6];
        for i in 0..6 {
            let hi = hex(b[i * 2]).ok_or("MAC string is not hex")?;
            let lo = hex(b[i * 2 + 1]).ok_or("MAC string is not hex")?;
            mac[i] = (hi << 4) | lo;
        }
        Ok(mac)
    }

    /// Queue a receive without waiting for it.
    fn arm_rx(&mut self, dev: &mut Device, buf: u64, len: u32) -> bool {
        let Some((addr, mut ring)) = dev.bulk_in.take() else { return false };
        ring.push(Trb {
            lo: buf as u32,
            hi: (buf >> 32) as u32,
            status: len,
            control: (TRB_NORMAL << 10) | (1 << 5) | (1 << 2),
        });
        let d = dci(addr);
        self.doorbell(dev.slot, d);
        dev.bulk_in = Some((addr, ring));
        true
    }

    /// Bytes received, if the armed transfer has completed. Never waits.
    fn poll_rx(&mut self, ep: u32, len: u32) -> Option<u32> {
        let ev = self.wait_event_ep(TRB_TRANSFER_EVENT, ep, 0)?;
        if ev.code() != 1 && ev.code() != 13 {
            return None;
        }
        Some(len.saturating_sub(ev.status & 0xFFFFFF))
    }
}

/// A CDC Ethernet adapter behind USB, as the IP stack sees it.
///
/// The interesting part is `receive`. One IN transfer is kept permanently
/// armed and each poll checks whether it completed -- rather than queueing a
/// fresh one per call and abandoning it on a timeout, which piles up transfers
/// the controller will complete later into events nobody is expecting.
pub struct UsbNet {
    ctl: Controller,
    dev: Device,
    mac: [u8; 6],
    rx: u64,
    tx: u64,
    armed: bool,
    /// DCI of the bulk IN endpoint, so its completions can be told apart from
    /// the transmit side's on the shared event ring.
    rx_dci: u32,
}

/// Ethernet's maximum frame, plus room for the CRC some devices append.
const RX_LEN: u32 = 1600;

impl UsbNet {
    pub fn new(mut ctl: Controller, mut dev: Device, mac: [u8; 6]) -> Option<UsbNet> {
        let rx = dma(RX_LEN as usize, 16)?;
        let tx = dma(RX_LEN as usize, 16)?;
        let rx_dci = dev.bulk_in.as_ref().map(|(a, _)| dci(*a))?;
        let armed = ctl.arm_rx(&mut dev, rx, RX_LEN);
        Some(UsbNet { ctl, dev, mac, rx, tx, armed, rx_dci })
    }
}

impl crate::net::iface::Nic for UsbNet {
    fn mac(&self) -> crate::net::Mac {
        self.mac
    }

    fn link_up(&mut self) -> bool {
        // The port still reporting a connection is the closest thing to a link
        // state USB offers: CDC has a notification endpoint for it, and
        // reading that would mean a third transfer ring for one bit.
        let off = self.ctl.portsc(self.dev.port);
        self.ctl.r32(off) & PORTSC_CCS != 0
    }

    fn transmit(&mut self, frame: &[u8]) -> bool {
        if frame.len() > RX_LEN as usize {
            return false;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(frame.as_ptr(), self.tx as *mut u8, frame.len());
        }
        self.ctl.bulk_out(&mut self.dev, self.tx, frame.len() as u32).is_ok()
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        if !self.armed {
            self.armed = self.ctl.arm_rx(&mut self.dev, self.rx, RX_LEN);
            return None;
        }
        let ep = self.rx_dci;
        let n = self.ctl.poll_rx(ep, RX_LEN)?;
        self.armed = false;
        let mut out = Vec::new();
        if out.try_reserve_exact(n as usize).is_err() {
            return None;
        }
        for i in 0..n as u64 {
            out.push(unsafe { read_volatile((self.rx + i) as *const u8) });
        }
        // Re-arm immediately: a receiver that only listens after being asked
        // drops everything that arrives between polls.
        self.armed = self.ctl.arm_rx(&mut self.dev, self.rx, RX_LEN);
        Some(out)
    }

    fn kind(&self) -> crate::net::iface::Kind {
        crate::net::iface::Kind::Ethernet
    }
}

/// Bring up the first CDC Ethernet adapter on the bus, if there is one.
pub fn probe_net(ecam: u64) -> Result<UsbNet, &'static str> {
    let caps = probe(ecam).map_err(|_| "no xHCI controller")?;
    let mut ctl = Controller::start(&caps).map_err(|_| "controller did not start")?;
    usb_scan_begin();
    for port in ctl.connected() {
        let Ok(mut dev) = ctl.enumerate(port) else { continue };
        // This walk already visits every attached device, so noting the
        // wireless ones costs a comparison and saves a second bus reset.
        usb_note(dev.vid, dev.pid);
        for i in 0..dev.num_configs {
            let Ok((buf, total)) = ctl.config_descriptor(&mut dev, i) else { break };
            let c = parse_config(buf, total);
            let (Some(ep_in), Some(ep_out)) = (c.bulk_in, c.bulk_out) else { continue };
            if !c.ecm {
                continue;
            }
            let mac = match ctl.ecm_mac(&mut dev, c.imac) {
                Ok(m) => m,
                Err(e) => {
                    // Locally-administered and deliberately odd, so it is never
                    // mistaken for a real address in a capture -- and so a boot
                    // log distinguishes "read the descriptor" from "made one
                    // up", which an invented 02:00:00:00:00:01 did not.
                    crate::kprintln!("  eth0   MAC string unreadable ({}), inventing one", e);
                    [0x02, 0x47, 0x4C, 0x41, 0x44, 0x53]
                }
            };
            ctl.set_configuration(&mut dev, c.value)?;
            if let Some((iface, alt)) = c.data_iface {
                if alt > 0 {
                    ctl.set_interface(&mut dev, iface, alt)?;
                }
            }
            ctl.configure_bulk(&mut dev, ep_in, ep_out)?;
            let (vid, pid) = (dev.vid, dev.pid);
            let nic = UsbNet::new(ctl, dev, mac).ok_or("out of memory")?;
            unsafe { *CLAIMED.get() = true };
            unsafe { *USB_ETHERNET.get() = Some((vid, pid)) };
            return Ok(nic);
        }
    }
    Err("no CDC Ethernet adapter found")
}

/// One addressed device: its slot, its default endpoint, and what it is.
pub struct Device {
    pub slot: u8,
    ep0: Ring,
    pub vid: u16,
    pub pid: u16,
    /// How many configurations the device has. Read rather than discovered by
    /// asking for one too many: a request for a configuration that does not
    /// exist is answered with a stall, and a stall *halts endpoint zero* until
    /// it is explicitly reset -- so probing past the end does not merely fail,
    /// it disables the device for everything after.
    pub num_configs: u8,
    /// The input context, kept because Configure Endpoint reuses it.
    inp: u64,
    pub port: u8,
    /// Transfer rings for the bulk pair, once configured. Held here because a
    /// ring the controller is walking must not be dropped, and the device
    /// outlives the call that set it up.
    bulk_in: Option<(u8, Ring)>,
    bulk_out: Option<(u8, Ring)>,
}

/// Device Context Index for an endpoint address.
///
/// Endpoint N has two of these -- OUT at 2N, IN at 2N+1 -- because a USB
/// endpoint number names a *pair*, and the context array indexes directions
/// separately. Endpoint 0 is DCI 1, which is why the array starts at one and
/// the doorbell for the default endpoint is 1 rather than 0.
fn dci(addr: u8) -> u32 {
    let num = (addr & 0x0F) as u32;
    num * 2 + if addr & 0x80 != 0 { 1 } else { 0 }
}

impl Controller {
    /// A control transfer with no data stage.
    ///
    /// The status stage of a no-data transfer is always IN, regardless of the
    /// request's direction. That is not symmetry for its own sake: the status
    /// stage is the *device* acknowledging, so it flows device to host even
    /// when the request was host to device.
    fn control_nodata(&mut self, dev: &mut Device, lo: u32, hi: u32) -> Result<(), &'static str> {
        dev.ep0.push(Trb {
            lo,
            hi,
            status: 8,
            // TRT 0: no data stage.
            control: (TRB_SETUP << 10) | (1 << 6),
        });
        dev.ep0.push(Trb {
            control: (TRB_STATUS << 10) | (1 << 5) | (1 << 16),
            ..Default::default()
        });
        self.doorbell(dev.slot, 1);
        let ev = self
            .wait_event(TRB_TRANSFER_EVENT, 500)
            .ok_or("no response to control transfer")?;
        if ev.code() != 1 && ev.code() != 13 {
            return Err("control transfer refused");
        }
        Ok(())
    }

    /// SET_CONFIGURATION. bmRequestType 0, bRequest 9, wValue the value from
    /// the descriptor -- not its index.
    pub fn set_configuration(&mut self, dev: &mut Device, value: u8) -> Result<(), &'static str> {
        // 0x0900, not 0x0009: the setup packet is little-endian, so byte 0
        // is bmRequestType and byte 1 is bRequest. Writing them the way they
        // are spoken aloud swaps the pair, and the device answers a request
        // type of 9 with silence rather than a stall.
        self.control_nodata(dev, 0x0900 | ((value as u32) << 16), 0)
    }

    /// SET_INTERFACE, for a data interface whose endpoints live on a non-zero
    /// alternate setting. bmRequestType 1 (interface), bRequest 11.
    pub fn set_interface(&mut self, dev: &mut Device, iface: u8, alt: u8) -> Result<(), &'static str> {
        self.control_nodata(dev, 0x0B01 | ((alt as u32) << 16), iface as u32)
    }

    /// Add the bulk pair to the device context.
    ///
    /// Configure Endpoint reuses the input context built during addressing.
    /// The add-context flags say which entries the controller should read, and
    /// the slot context's Context Entries field has to reach the highest DCI
    /// being added -- a controller told about an endpoint at DCI 5 while the
    /// slot still claims one entry accepts the command and then never rings.
    pub fn configure_bulk(
        &mut self,
        dev: &mut Device,
        ep_in: Endpoint,
        ep_out: Endpoint,
    ) -> Result<(), &'static str> {
        let cb = self.ctx_bytes as u64;
        let inp = dev.inp;
        let din = dci(ep_in.addr);
        let dout = dci(ep_out.addr);
        let max_dci = din.max(dout);

        let rin = Ring::new().ok_or("out of memory")?;
        let rout = Ring::new().ok_or("out of memory")?;

        unsafe {
            // Add the slot context and both endpoints; drop nothing.
            write_volatile(inp as *mut u32, 0);
            write_volatile((inp + 4) as *mut u32, 1 | (1 << din) | (1 << dout));
            // Context Entries, in the slot context's first dword.
            let slot_ctx = inp + cb;
            let d0 = read_volatile(slot_ctx as *const u32);
            write_volatile(slot_ctx as *mut u32, (d0 & 0x07FF_FFFF) | (max_dci << 27));

            for (d, ep, ring) in [(din, ep_in, &rin), (dout, ep_out, &rout)] {
                let c = inp + cb * (d as u64 + 1);
                // EP Type: 2 is Bulk OUT, 6 is Bulk IN.
                let ep_type: u32 = if ep.input { 6 } else { 2 };
                write_volatile((c + 4) as *mut u32,
                    (ep_type << 3) | (3 << 1) | ((ep.max_packet as u32) << 16));
                write_volatile((c + 8) as *mut u32, (ring.base as u32) | 1);
                write_volatile((c + 12) as *mut u32, (ring.base >> 32) as u32);
                // Average TRB length. Zero is legal and some controllers
                // schedule badly on it; the packet size is the honest estimate.
                write_volatile((c + 16) as *mut u32, ep.max_packet as u32);
            }
        }

        let ev = self
            .command(
                Trb {
                    lo: inp as u32,
                    hi: (inp >> 32) as u32,
                    control: (TRB_CONFIG_EP << 10) | ((dev.slot as u32) << 24),
                    ..Default::default()
                },
                200,
            )
            .ok_or("no response to Configure Endpoint")?;
        if ev.code() != 1 {
            return Err("Configure Endpoint refused");
        }

        dev.bulk_in = Some((ep_in.addr, rin));
        dev.bulk_out = Some((ep_out.addr, rout));
        Ok(())
    }

    /// Queue one bulk transfer and wait for it.
    ///
    /// Interrupt On Short Packet as well as On Completion: an IN transfer that
    /// receives less than a full buffer is the normal case for a network
    /// endpoint, and without ISP the event only arrives when the buffer fills
    /// -- which for a 1514-byte read of a 60-byte frame is never.
    fn bulk(&mut self, slot: u8, ring: &mut Ring, addr: u8, buf: u64, len: u32, ms: u64)
        -> Result<u32, &'static str>
    {
        ring.push(Trb {
            lo: buf as u32,
            hi: (buf >> 32) as u32,
            status: len,
            control: (TRB_NORMAL << 10) | (1 << 5) | (1 << 2),
        });
        let d = dci(addr);
        self.doorbell(slot, d);
        let ev = self.wait_event_ep(TRB_TRANSFER_EVENT, d, ms).ok_or("bulk timed out")?;
        if ev.code() != 1 && ev.code() != 13 {
            return Err("bulk transfer failed");
        }
        Ok(len.saturating_sub(ev.status & 0xFFFFFF))
    }

    pub fn bulk_out(&mut self, dev: &mut Device, buf: u64, len: u32) -> Result<u32, &'static str> {
        let (addr, mut ring) = dev.bulk_out.take().ok_or("no bulk out endpoint")?;
        let r = self.bulk(dev.slot, &mut ring, addr, buf, len, 500);
        dev.bulk_out = Some((addr, ring));
        r
    }

    pub fn bulk_in(&mut self, dev: &mut Device, buf: u64, len: u32, ms: u64)
        -> Result<u32, &'static str>
    {
        let (addr, mut ring) = dev.bulk_in.take().ok_or("no bulk in endpoint")?;
        let r = self.bulk(dev.slot, &mut ring, addr, buf, len, ms);
        dev.bulk_in = Some((addr, ring));
        r
    }
}

/// A bulk endpoint found in a configuration descriptor.
#[derive(Clone, Copy)]
pub struct Endpoint {
    /// bEndpointAddress, including the direction bit.
    pub addr: u8,
    pub max_packet: u16,
    pub input: bool,
}

/// What a configuration offers, as far as this driver cares.
pub struct Config {
    pub value: u8,
    pub interfaces: usize,
    /// Interface number and alternate setting that carry the bulk pair.
    pub data_iface: Option<(u8, u8)>,
    pub bulk_in: Option<Endpoint>,
    pub bulk_out: Option<Endpoint>,
    /// True when this looks like CDC Ethernet rather than RNDIS.
    pub ecm: bool,
    /// String index of the adapter's MAC, from the Ethernet Networking
    /// Functional Descriptor. CDC puts the address in a *string*, in ASCII
    /// hex -- there is no binary field for it anywhere in the descriptors.
    pub imac: u8,
}

/// Walk a configuration descriptor block.
///
/// Descriptors are a flat sequence of length-prefixed records, so this is a
/// walk rather than a parse -- step by `bLength` and dispatch on `bDescriptorType`.
/// Trusting the length field is also the only defence against a malformed
/// block: a zero length would spin forever, so it terminates instead.
pub fn parse_config(buf: u64, total: usize) -> Config {
    const IFACE: u8 = 4;
    const ENDPOINT: u8 = 5;
    // USB class codes: 0x02 communications, 0x0A CDC data.
    const CLASS_CDC_DATA: u8 = 0x0A;

    let mut cfg = Config {
        value: 0,
        interfaces: 0,
        data_iface: None,
        bulk_in: None,
        bulk_out: None,
        ecm: false,
        imac: 0,
    };
    let at = |o: usize| -> u8 { unsafe { read_volatile((buf + o as u64) as *const u8) } };

    if total >= 9 {
        cfg.interfaces = at(4) as usize;
        cfg.value = at(5);
    }

    let mut o = 0usize;
    let mut in_data_iface = false;
    while o + 2 <= total {
        let len = at(o) as usize;
        let kind = at(o + 1);
        if len == 0 {
            break;
        }
        match kind {
            IFACE if o + 9 <= total => {
                let class = at(o + 5);
                in_data_iface = class == CLASS_CDC_DATA;
                if in_data_iface {
                    // An ECM data interface carries its endpoints on a
                    // non-zero alternate setting; alt 0 is deliberately empty
                    // so an unconfigured device consumes no bus bandwidth.
                    let alt = at(o + 3);
                    if cfg.data_iface.is_none() || alt > 0 {
                        cfg.data_iface = Some((at(o + 2), alt));
                    }
                }
            }
            // 0x24 is CS_INTERFACE, a class-specific interface descriptor;
            // subtype 0x0F is Ethernet Networking Functional, whose fourth
            // byte is the string index holding the MAC.
            // Descriptor 0x0F is what makes a configuration Ethernet, and the
            // only thing that does. A CDC *data* interface is not enough:
            // QEMU's usb-net offers two configurations that both have one, and
            // the first is RNDIS -- which accepts bulk writes and silently
            // passes nothing, because it wants a control protocol first. Taking
            // it cost a send that worked and a receive that never fired.
            0x24 if o + 4 <= total && at(o + 2) == 0x0F => {
                cfg.imac = at(o + 3);
                cfg.ecm = true;
            }
            ENDPOINT if o + 7 <= total && in_data_iface => {
                let addr = at(o + 2);
                let attrs = at(o + 3);
                let mps = unsafe { read_volatile((buf + o as u64 + 4) as *const u16) } & 0x7FF;
                // Transfer type is the low two bits; 2 is bulk.
                if attrs & 0x3 == 2 {
                    let ep = Endpoint { addr, max_packet: mps, input: addr & 0x80 != 0 };
                    if ep.input {
                        cfg.bulk_in = Some(ep);
                    } else {
                        cfg.bulk_out = Some(ep);
                    }
                }
            }
            _ => {}
        }
        o += len;
    }
    cfg
}
