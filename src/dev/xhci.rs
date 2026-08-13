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
//! Still to come: the operational registers, a command ring, an event ring,
//! device slots and endpoint contexts, then USB enumeration on top.

use super::pci;
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
    use crate::gfx::console::{self, LTGRAY, LTRED, WHITE, YELLOW};
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
            kprintln!("  no enumeration yet -- rings and slots are the next step");
        }
    }
    console::set_color(WHITE);
}
