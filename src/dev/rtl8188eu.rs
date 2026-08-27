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
//! Here: device matching, the register file over vendor control transfers,
//! chip identification, and initialisation as far as the baseband.
//! `bring_up` runs the power sequence and then the MAC, PHY and AGC tables.
//!
//! **Not here: the radio, and the datapath.** `RADIOA_INIT` is transcribed
//! next door and is not applied, because an RF table is addressed by register
//! index on the radio's own serial bus and those indices collide with the MAC
//! power-control registers -- writing it as direct register writes puts radio
//! values into the power sequencer, and every one of them returns success.
//! Reaching the radio needs the LSSI parameter registers and the index
//! encoding that goes into them, which have to be transcribed the way the
//! tables were. `apply_bb` refuses any table addressed below the baseband so
//! that mistake cannot be made by accident.
//!
//! Missing alongside it: the LLT page table, the receive FIFO boundary that
//! `CR_INIT` deliberately leaves the TX and RX enables waiting on, the efuse
//! layout, the firmware blob, and the datapath itself -- no bulk endpoint is
//! ever touched here, so not one frame goes out or comes in. Scanning is
//! sending probe requests and reading beacons, so a chip brought this far
//! still cannot find a network.
//!
//! The frames themselves are not the missing part. `net::ieee80211` builds
//! probe requests and parses beacons, and is checked at every boot against
//! frames it constructs on the spot. When something can carry bytes to and
//! from the air, the layer above it is already written and already tested.
//!
//! The tables that are here were transcribed rather than recalled, and the
//! ones that are missing must be too. A plausible-looking power sequence with
//! three wrong values yields a chip that accepts every write, reports
//! sensible-looking registers, and never transmits -- precisely the failure
//! mode this project keeps getting bitten by and refuses to manufacture.
//!
//! Keep this section honest. It was stale once already: it claimed the driver
//! stopped at chip identification long after the power sequence and MAC table
//! had landed, and a settings page written from it told an operator the wrong
//! thing about their own hardware.
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

// The rest of the register file this driver touches. Offsets and bit positions
// are from Linux's rtl8xxxu regs.h; see rtl8188eu_tables.rs on provenance.
const REG_SYS_FUNC: u16 = 0x0002;
const REG_APS_FSMCO: u16 = 0x0004;
const REG_LPLDO_CTRL: u16 = 0x0023;
const REG_AFE_XTAL_CTRL: u16 = 0x0024;
const REG_CR: u16 = 0x0100;

const SYS_FUNC_BBRSTB: u32 = 1 << 0;
const SYS_FUNC_BB_GLB_RSTN: u32 = 1 << 1;
const APS_FSMCO_MAC_ENABLE: u32 = 1 << 8;
const APS_FSMCO_HW_SUSPEND: u32 = 1 << 11;
const APS_FSMCO_PCIE: u32 = 1 << 12;
const APS_FSMCO_HW_POWERDOWN: u32 = 1 << 15;
/// Power ready, in the same register.
const APS_FSMCO_PFM_ALDN: u32 = 1 << 17;

/// DMA, protocol, scheduler, security and the 32k calibration timer.
///
/// Deliberately *not* including the MAC TX and RX enables: the 88E has a
/// hardware bug where setting them before `REG_TRXFF_BNDY` makes the receive
/// FIFO boundary come out larger than the buffer actually is. Linux carries
/// the same comment, and it is the kind of ordering constraint that corrupts
/// quietly rather than failing.
const CR_INIT: u32 = 0x063F;

/// How many times to poll a register before giving up, matching Linux. Each
/// poll here is a USB round trip rather than an MMIO read, so this is a much
/// longer wall-clock timeout than the same number would be on a PCI part.
const MAX_POLL: u32 = 500;

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

// --- power on -------------------------------------------------------------

impl Regs<'_> {
    /// Read, clear some bits, write back.
    ///
    /// Two USB round trips per call, which is why the sequence below is
    /// written as explicit steps rather than folded into a table: half of
    /// these are read-modify-write and a table of (register, value) pairs
    /// cannot express one.
    fn clear(&mut self, reg: u16, n: u16, bits: u32) -> Result<(), &'static str> {
        let v = self.read(reg, n)?;
        self.write(reg, n, v & !bits)
    }

    fn set(&mut self, reg: u16, n: u16, bits: u32) -> Result<(), &'static str> {
        let v = self.read(reg, n)?;
        self.write(reg, n, v | bits)
    }

    /// Poll a register until `f` is satisfied.
    fn poll(&mut self, reg: u16, mut f: impl FnMut(u32) -> bool) -> Result<(), &'static str> {
        for _ in 0..MAX_POLL {
            if f(self.read(reg, 4)?) {
                return Ok(());
            }
            crate::time::delay_us(10);
        }
        Err("register poll timed out")
    }

    /// Bring the chip from cold to a state where the MAC responds.
    ///
    /// This follows `rtl8188eu_power_on` in Linux step for step, including the
    /// order, which is the part that matters: the sequence walks the chip
    /// through disabled -> emulation -> active, and the analogue blocks need
    /// settling time between stages that is expressed only as "this write
    /// comes after that one". Reordering it produces a chip that acknowledges
    /// every write and does not work.
    pub fn power_on(&mut self) -> Result<(), &'static str> {
        // Disabled to emulation: drop the suspend bits.
        self.clear(REG_APS_FSMCO, 2, APS_FSMCO_HW_SUSPEND | APS_FSMCO_PCIE)?;

        // Emulation to active.
        self.poll(REG_APS_FSMCO, |v| v & APS_FSMCO_PFM_ALDN != 0)?;
        self.clear(REG_SYS_FUNC, 1, SYS_FUNC_BBRSTB | SYS_FUNC_BB_GLB_RSTN)?;
        // Schmitt trigger on the crystal input.
        self.set(REG_AFE_XTAL_CTRL, 4, 1 << 23)?;
        self.clear(REG_APS_FSMCO, 2, APS_FSMCO_HW_POWERDOWN)?;
        self.clear(REG_APS_FSMCO, 2, APS_FSMCO_HW_SUSPEND | APS_FSMCO_PCIE)?;

        // Setting MAC_ENABLE starts the power-up; the hardware clears the bit
        // when it has finished, so the same bit is both the request and the
        // completion flag.
        self.set(REG_APS_FSMCO, 4, APS_FSMCO_MAC_ENABLE)?;
        self.poll(REG_APS_FSMCO, |v| v & APS_FSMCO_MAC_ENABLE == 0)?;

        // LDO back to normal mode.
        self.clear(REG_LPLDO_CTRL, 1, 1 << 4)?;

        self.write(REG_CR, 2, CR_INIT)
    }

    /// Apply one of the initialisation tables.
    ///
    /// Order is preserved because order is the content -- see the note in
    /// `rtl8188eu_tables`.
    pub fn apply8(&mut self, table: &[(u16, u8)]) -> Result<(), &'static str> {
        for (reg, val) in table {
            self.write(*reg, 1, *val as u32)?;
        }
        Ok(())
    }

    pub fn apply32(&mut self, table: &[(u16, u32)]) -> Result<(), &'static str> {
        for (reg, val) in table {
            self.write(*reg, 4, *val)?;
        }
        Ok(())
    }

    /// MAC initialisation: power on, then the MAC register table.
    ///
    /// Stops here on purpose. The PHY and radio tables exist in
    /// `rtl8188eu_tables` but applying them needs the baseband brought up
    /// first and the RF writes go through a serial interface rather than
    /// straight to a register, neither of which is written yet. Applying them
    /// anyway would half-configure the chip, which is worse than not starting.
    /// Baseband registers start here. Everything below is the MAC control
    /// area, which is why the guard below exists rather than being paranoia.
    const BB_BASE: u16 = 0x800;

    /// Apply a baseband table, refusing anything that is not one.
    ///
    /// The radio's table is addressed by register *index* on the RF serial
    /// bus -- 0x00, 0x08, 0x18 -- and those indices collide with the MAC
    /// power-control registers. Writing `RADIOA_INIT` through this path would
    /// put radio values into the power sequencer, and the chip would take
    /// every write and report success. The tables are all `&[(u16, u32)]` and
    /// nothing in the type distinguishes them, so the address range is the
    /// only thing that can, and it is checked rather than remembered.
    fn apply_bb(&mut self, table: &[(u16, u32)]) -> Result<(), &'static str> {
        if table.iter().any(|(r, _)| *r < Self::BB_BASE) {
            return Err("not a baseband table -- refusing to write it as direct registers");
        }
        self.apply32(table)
    }

    /// Power on and initialise as far as this driver honestly can.
    ///
    /// Power sequence, MAC registers, then the baseband: PHY first because
    /// AGC writes tune what PHY set up, and AGC's table writes one register
    /// 130 times with the index carried in the value, so its order is its
    /// content.
    ///
    /// It stops before the radio. `RADIOA_INIT` is transcribed and sitting
    /// next door, and applying it needs the RF serial interface -- the LSSI
    /// parameter registers and the index encoding that goes into them --
    /// which is not written here and would have to be transcribed like the
    /// tables were. Between here and a working radio there is also the LLT
    /// page table, the receive FIFO boundary that `CR_INIT` deliberately
    /// leaves the TX and RX enables waiting on, and the whole datapath.
    ///
    /// So: this brings the chip up. It does not make it a radio, and the
    /// difference is worth stating because every register write below returns
    /// success either way.
    pub fn bring_up(&mut self) -> Result<(), &'static str> {
        self.power_on()?;
        self.apply8(super::rtl8188eu_tables::MAC_INIT)?;
        self.apply_bb(super::rtl8188eu_tables::PHY_INIT)?;
        self.apply_bb(super::rtl8188eu_tables::AGC_INIT)
    }
}
