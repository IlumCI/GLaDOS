//! Realtek RTL8188EU(S): the USB wireless dongle.
//!
//! This is the part that matters for wireless on this machine. The built-in
//! card is CNVi -- the MAC lives in the PCH and the M.2 module is a radio
//! reachable only through an undocumented signed-firmware protocol -- so
//! `net/wifi.rs` identifies it and refuses to pretend. A USB dongle sidesteps
//! all of that: it is a complete radio *and* MAC behind a bus this kernel now
//! drives.
//!
//! The supplicant is already done. `net/wpa2.rs` implements the four-way
//! handshake and is checked against the IEEE 802.11i vectors at every boot; it
//! has simply never had hardware to run on. This module is what would give it
//! some.
//!
//! ### How the chip is addressed
//!
//! There is no MMIO here. Every register is reached by a vendor control
//! transfer -- `bRequest` 0x05, the register offset in `wValue`, one, two or
//! four bytes of data -- so a register read is a full USB round trip of a few
//! hundred microseconds. That single fact shapes the whole driver: the vendor
//! drivers batch aggressively and avoid read-modify-write in hot paths, and
//! anything written here that polls a register in a loop will be slow in a way
//! that is not obvious from the source.
//!
//! Frames move over bulk endpoints, with the TX queues mapped onto two or three
//! bulk OUT endpoints by traffic class.
//!
//! ### What is here, and what is deliberately not
//!
//! Here: device matching, the register file over vendor control transfers, and
//! chip identification.
//!
//! **Not here: the power-on sequence, the RF/PHY initialisation tables, the
//! efuse layout, and the firmware blob.** Those are several hundred specific
//! register/value pairs that exist in Realtek's vendor driver and in Linux's
//! `rtl8xxxu_8188e.c`, and they are not reproduced from memory on purpose. A
//! plausible-looking power sequence with three wrong values yields a chip that
//! accepts every write, reports sensible-looking registers, and never
//! transmits -- which is precisely the failure mode this project keeps getting
//! bitten by and now refuses to manufacture. They have to be transcribed from
//! the source, and until they are this driver identifies hardware and stops.
//!
//! ### Testing
//!
//! None of this can run in QEMU, which has no model of this chip. The only
//! verification available is the GF63 with the dongle plugged in, and the first
//! thing worth checking there is a single register read: if `REG_SYS_CFG` comes
//! back with a plausible chip version, the entire transport underneath -- xHCI
//! rings, enumeration, control transfers, vendor requests -- is proven in one
//! step. Everything after that is tables.

use super::xhci::{Controller, Device};

/// Realtek's vendor request for register access. The same value serves reads
/// and writes; the direction lives in `bmRequestType`.
const VENDOR_REQ: u8 = 0x05;

/// Chip version and manufacturer. The one register worth reading before
/// anything else is initialised, because it is readable before anything else
/// is initialised.
const REG_SYS_CFG: u16 = 0x00F0;

/// USB ids that are an RTL8188EU behind some other company's badge.
///
/// The dongle on this machine reports 2357:010c -- TP-Link, not Realtek -- and
/// matching on the Realtek vendor id alone would miss it entirely. There is no
/// class code to key off: it is a vendor-specific interface, so the id list
/// *is* the detection.
const KNOWN: &[(u16, u16, &str)] = &[
    (0x2357, 0x010C, "TP-Link TL-WN722N v2/v3"),
    (0x0BDA, 0x8179, "Realtek RTL8188EUS"),
    (0x0BDA, 0x0179, "Realtek RTL8188ETV"),
    (0x2357, 0x0111, "TP-Link TL-WN727N v5"),
];

/// The name of a known dongle, if this is one.
pub fn identify(vid: u16, pid: u16) -> Option<&'static str> {
    KNOWN.iter().find(|(v, p, _)| *v == vid && *p == pid).map(|(_, _, n)| *n)
}

/// The chip's register file, borrowed from whoever owns the USB device.
///
/// Borrowing rather than owning because the register layer is useful before
/// there is a driver to own anything -- `usb` reads the chip id through this
/// while still holding the controller for the rest of its scan. The eventual
/// driver owns the `Controller` and `Device` and constructs one of these per
/// access, which costs nothing.
pub struct Regs<'a> {
    ctl: &'a mut Controller,
    dev: &'a mut Device,
    /// A scratch DMA buffer for register access. One transfer is in flight at a
    /// time on the control pipe, so one buffer is enough and allocating per
    /// register read would be a heap round trip per USB round trip.
    scratch: u64,
}

impl<'a> Regs<'a> {
    pub fn new(ctl: &'a mut Controller, dev: &'a mut Device, scratch: u64) -> Regs<'a> {
        Regs { ctl, dev, scratch }
    }

    /// Read `n` bytes (1, 2 or 4) from a register, little-endian.
    pub fn read(&mut self, reg: u16, n: u16) -> Result<u32, &'static str> {
        let got = self.ctl.vendor(self.dev, true, VENDOR_REQ, reg, 0, self.scratch, n)?;
        if got < n as u32 {
            return Err("short read from register");
        }
        let mut v = 0u32;
        for i in 0..n as u64 {
            let b = unsafe { core::ptr::read_volatile((self.scratch + i) as *const u8) };
            v |= (b as u32) << (i * 8);
        }
        Ok(v)
    }

    /// Write `n` bytes (1, 2 or 4) to a register, little-endian.
    pub fn write(&mut self, reg: u16, n: u16, value: u32) -> Result<(), &'static str> {
        for i in 0..n as u64 {
            unsafe {
                core::ptr::write_volatile(
                    (self.scratch + i) as *mut u8,
                    (value >> (i * 8)) as u8,
                );
            }
        }
        self.ctl.vendor(self.dev, false, VENDOR_REQ, reg, 0, self.scratch, n)?;
        Ok(())
    }

    /// What the chip says it is.
    ///
    /// This is the single most informative thing the driver can do before any
    /// initialisation: `REG_SYS_CFG` is readable from reset, so a sane answer
    /// proves the whole path from PCI through xHCI rings and enumeration down
    /// to a vendor control transfer, and a garbage answer localises the fault
    /// to somewhere in that path rather than to the tables that come later.
    pub fn chip_id(&mut self) -> Result<ChipId, &'static str> {
        let v = self.read(REG_SYS_CFG, 4)?;
        Ok(ChipId {
            raw: v,
            // Bit 20 distinguishes the test chip from production silicon; a
            // dongle anyone can buy is always production, so this reading
            // `true` means the register read is wrong rather than that the
            // hardware is exotic. It is the cheapest sanity check available.
            test_chip: v & (1 << 20) == 0,
            // TSMC or UMC, recorded because the RF calibration tables differ
            // between them -- which is the first place this distinction will
            // matter, and it will matter silently.
            vendor_umc: v & (1 << 7) != 0,
            version: ((v >> 12) & 0xF) as u8,
        })
    }
}

pub struct ChipId {
    pub raw: u32,
    pub test_chip: bool,
    pub vendor_umc: bool,
    pub version: u8,
}
