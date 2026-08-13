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

/// What `usb` prints.
pub fn report(ecam: u64) {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[usb]");
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
                kprintln!("  port {}  {:04x}:{:04x}  slot {}", p, dev.vid, dev.pid, dev.slot);
                console::set_color(LTGRAY);

                // Every configuration, because the interesting one is rarely
                // the first: QEMU's usb-net puts RNDIS on configuration 1 and
                // CDC Ethernet on 2, and only the second is worth driving.
                for i in 0..4u8 {
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
fn dma(size: usize, align: usize) -> Option<u64> {
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
}

impl Controller {
    #[inline]
    fn r32(&self, off: u64) -> u32 {
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
    fn portsc(&self, port: u8) -> u64 {
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
        for _ in 0..ms {
            while let Some(t) = self.poll_event() {
                if t.kind() == kind {
                    return Some(t);
                }
            }
            delay_us(1000);
        }
        None
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

        let ev = self
            .wait_event(TRB_TRANSFER_EVENT, 500)
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

        let mut dev = Device { slot, ep0: ring, vid: 0, pid: 0, inp, port };
        let buf = dma(18, 16).ok_or("out of memory")?;
        self.descriptor(&mut dev, 0x0100, buf, 18)?;
        dev.vid = unsafe { read_volatile((buf + 8) as *const u16) };
        dev.pid = unsafe { read_volatile((buf + 10) as *const u16) };
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

/// One addressed device: its slot, its default endpoint, and what it is.
pub struct Device {
    pub slot: u8,
    ep0: Ring,
    pub vid: u16,
    pub pid: u16,
    /// The input context, kept because Configure Endpoint reuses it.
    inp: u64,
    pub port: u8,
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
                    cfg.ecm = true;
                }
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
