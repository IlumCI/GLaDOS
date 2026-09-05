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
//! chip identification, the power sequence, all four initialisation tables,
//! the radio's serial interface, and the transmit and receive descriptors.
//! `bring_up` runs power on, then MAC, PHY, AGC and the radio.
//!
//! The radio is not memory mapped. Its registers are reached by writing an
//! address and a value together into one baseband register, so `RADIOA_INIT`
//! goes through `write_rf` and never through a direct write -- its indices
//! collide numerically with the MAC power-control registers, and applying it
//! the wrong way puts radio values into the power sequencer with every write
//! returning success. `apply_bb` refuses anything addressed below the
//! baseband and `apply_rf` refuses anything above one byte, so the two tables
//! cannot be swapped by accident.
//!
//! **Not here: the parts between a configured chip and a working radio.** The
//! LLT page table is not built, the receive FIFO boundary is unset and so the
//! MAC transmit and receive enables that `CR_INIT` deliberately leaves out
//! stay out, no channel is selected, the efuse is not read so the MAC address
//! is unknown, no firmware is uploaded, and nothing is handed to a bulk
//! endpoint. A descriptor can be built and a descriptor can be read; nothing
//! yet carries one to the chip.
//!
//! What is above it is finished. `net::ieee80211` builds probe requests and
//! parses beacons, `desc` builds and reads the descriptors that would wrap
//! them, and `net::wpa2` runs the handshake that follows. All three are
//! checked at every boot. The gap is the transport in the middle.
//!
//! ### Testing
//!
//! Almost none of this can run in QEMU, which has no model of this chip. The
//! exception is `desc`, which is arithmetic on bytes and is checked at every
//! boot -- worth having for one reason above the rest: `TXDESC_OWN` is defined
//! twice in rtl8xxxu.h, the bit-31 form is inside a dead `#if 0`, and taking
//! it puts the OWN bit in the wrong byte of a descriptor the chip then ignores
//! in silence.
//!
//! Everything else waits on the GF63 with the dongle plugged in, and the first
//! thing worth checking there is a single register read: if `REG_SYS_CFG` comes
//! back with a plausible chip version, the entire transport underneath -- xHCI
//! rings, enumeration, control transfers, vendor requests -- is proven in one
//! step. After that, `read_rf` on a radio register with a known reset value
//! proves the serial interface the same way, and it is the next thing to try.

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

    /// Write one radio register.
    ///
    /// The radio has no memory-mapped register file. Its registers are reached
    /// over a serial interface driven by one baseband register: the address
    /// and the value go in together, address in the high bits, and the
    /// baseband clocks them out to the RF chip. That is why `RADIOA_INIT`
    /// cannot be applied the way `MAC_INIT` is, and why applying it as direct
    /// writes would have put radio values into the MAC power sequencer.
    ///
    /// Path A only. The 8188EU has one transmit chain and one receive chain,
    /// so path B exists in the register map and not in this part.
    ///
    /// From Linux's `rtl8xxxu_write_rfreg`. The 8192E special case in the
    /// original toggles a power-save bit around the write and does not apply
    /// here; everything else is this.
    pub fn write_rf(&mut self, reg: u8, data: u32) -> Result<(), &'static str> {
        use super::rtl8188eu_tables as t;
        let payload = data & t::FPGA0_LSSI_PARM_DATA_MASK;
        let addressed = ((reg as u32) << t::FPGA0_LSSI_PARM_ADDR_SHIFT) | payload;
        self.write(t::REG_FPGA0_XA_LSSI_PARM as u16, 4, addressed)?;
        // The original waits a microsecond here for the serial write to clock
        // out. Every access in this driver is a USB control transfer, which is
        // microseconds at its very fastest, so the delay has already happened
        // by the time the next one is issued.
        Ok(())
    }

    /// Read one radio register.
    ///
    /// Considerably more involved than the write, and the shape is not
    /// arbitrary: the address is loaded, then a bit is driven low and high
    /// again on the parameter register to clock the value back, and only then
    /// is the readback register meaningful. The edge is what makes it a read
    /// rather than a stale value, which is why the two writes to
    /// `HSSI_PARM2` cannot be collapsed into one.
    ///
    /// Which readback register holds the answer depends on whether the part is
    /// wired for a parallel or a serial interface, reported by a bit in
    /// `HSSI_PARM1`. Both are read from rather than assumed.
    ///
    /// From Linux's `rtl8xxxu_read_rfreg`.
    pub fn read_rf(&mut self, reg: u8) -> Result<u32, &'static str> {
        use super::rtl8188eu_tables as t;
        let hssia = self.read(t::REG_FPGA0_XA_HSSI_PARM2 as u16, 4)?;

        let mut val = hssia & !t::FPGA0_HSSI_PARM2_ADDR_MASK;
        val |= (reg as u32) << t::FPGA0_HSSI_PARM2_ADDR_SHIFT;
        val |= t::FPGA0_HSSI_PARM2_EDGE_READ;

        let low = hssia & !t::FPGA0_HSSI_PARM2_EDGE_READ;
        self.write(t::REG_FPGA0_XA_HSSI_PARM2 as u16, 4, low)?;
        self.write(t::REG_FPGA0_XA_HSSI_PARM2 as u16, 4, val)?;
        self.write(t::REG_FPGA0_XA_HSSI_PARM2 as u16, 4, low | t::FPGA0_HSSI_PARM2_EDGE_READ)?;

        let parm1 = self.read(t::REG_FPGA0_XA_HSSI_PARM1 as u16, 4)?;
        let from = if parm1 & t::FPGA0_HSSI_PARM1_PI != 0 {
            t::REG_HSPI_XA_READBACK
        } else {
            t::REG_FPGA0_XA_LSSI_READBACK
        };
        // Radio registers are twenty bits wide; the rest of the word is
        // whatever the baseband left there.
        Ok(self.read(from as u16, 4)? & 0xF_FFFF)
    }

    /// Apply a radio table over the serial interface.
    ///
    /// The counterpart to `apply_bb`, and the reason that one refuses tables
    /// addressed below the baseband: these addresses are radio register
    /// indices, they overlap the MAC control area numerically, and only the
    /// path taken tells them apart.
    fn apply_rf(&mut self, table: &[(u16, u32)]) -> Result<(), &'static str> {
        if table.iter().any(|(r, _)| *r > 0xFF) {
            return Err("not a radio table -- indices are one byte");
        }
        for (reg, val) in table {
            self.write_rf(*reg as u8, *val)?;
        }
        Ok(())
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
        self.apply_bb(super::rtl8188eu_tables::AGC_INIT)?;
        self.apply_rf(super::rtl8188eu_tables::RADIOA_INIT)
    }
}

/// The parts of bringing this chip up that can be checked without one.
///
/// **Everything the chip answers is unverifiable here and everything it is
/// *asked* is not**, so the two are separated on purpose. A register write
/// returns success whether or not a radio is listening; the efuse layout, the
/// packet-buffer arithmetic and the firmware header are decisions with one
/// right answer, and getting them wrong is what makes a driver fail in a way
/// nobody can debug over a USB cable.
///
/// So this module is the half with claims on it. `Regs` below is the half that
/// has never touched hardware, and says so.
pub mod init {
    use alloc::vec::Vec;

    /// The efuse is one-time-programmable memory holding the MAC address and
    /// this chip's own calibration, and it is stored *packed*: a run of
    /// sections, each a header naming an offset and which of its four words
    /// are present, so a chip programmed twice carries both writes and the
    /// later one wins.
    ///
    /// Decoding it is therefore a replay rather than a copy, and it is the one
    /// piece of bring-up where a plausible-looking mistake yields a plausible
    /// -looking MAC address. Linux calls this `rtl8xxxu_read_efuse`.
    pub const EFUSE_MAP_LEN: usize = 512;
    /// Where the MAC address sits once the map is rebuilt, for the 8188E.
    pub const EFUSE_MAC_OFFSET: usize = 0xD7;

    /// Rebuild the 512-byte map from the packed efuse contents.
    ///
    /// `0xFF` is erased memory and ends the walk. The extended header form --
    /// low nibble `0x0F` -- carries the offset's high bits in a second byte,
    /// which is what lets an offset exceed sixteen sections.
    pub fn efuse_decode(raw: &[u8]) -> [u8; EFUSE_MAP_LEN] {
        let mut map = [0xFFu8; EFUSE_MAP_LEN];
        let mut i = 0usize;
        while i < raw.len() {
            let header = raw[i];
            i += 1;
            if header == 0xFF {
                break;
            }
            let (offset, word_en) = if header & 0x1F == 0x0F {
                // Extended: the next byte carries offset[7:4] and the word
                // enables. A truncated extension is the end of the data
                // rather than a reason to read past it.
                if i >= raw.len() {
                    break;
                }
                let ext = raw[i];
                i += 1;
                ((((ext & 0xF0) >> 1) | (header >> 5)) as usize, ext & 0x0F)
            } else {
                ((header >> 4) as usize, header & 0x0F)
            };
            for word in 0..4 {
                // A *clear* bit means the word is present. Inverted from what
                // the name suggests, and the source of the classic bug where
                // the map comes out entirely erased.
                if word_en & (1 << word) != 0 {
                    continue;
                }
                if i + 1 >= raw.len() {
                    return map;
                }
                let at = offset * 8 + word * 2;
                if at + 1 < EFUSE_MAP_LEN {
                    map[at] = raw[i];
                    map[at + 1] = raw[i + 1];
                }
                i += 2;
            }
        }
        map
    }

    /// The MAC address out of a decoded map, or nothing when it is not one.
    ///
    /// Refused rather than returned when the bytes are erased or a multicast
    /// address: an all-`FF` MAC is what an unprogrammed efuse reads as, and a
    /// driver that adopts it transmits frames every AP ignores while looking
    /// like it works.
    pub fn efuse_mac(map: &[u8; EFUSE_MAP_LEN]) -> Option<[u8; 6]> {
        let mut mac = [0u8; 6];
        mac.copy_from_slice(&map[EFUSE_MAC_OFFSET..EFUSE_MAC_OFFSET + 6]);
        if mac == [0xFF; 6] || mac == [0; 6] || mac[0] & 1 != 0 {
            return None;
        }
        Some(mac)
    }

    /// The chip's transmit buffer is 128-byte pages threaded into a linked
    /// list, and the LLT is that list: entry *n* holds the number of the page
    /// that follows *n*.
    ///
    /// Total pages, and how the boundary splits them between queues. These are
    /// the 8188E's numbers from Linux's `rtl8188e_init_queue_reserved_page`;
    /// `TX_PAGE_BOUNDARY` is what `REG_TRXFF_BNDY` is programmed with, and it
    /// is the register `CR_INIT` deliberately leaves the MAC enables waiting
    /// on.
    pub const TX_TOTAL_PAGES: u8 = 169;
    pub const TX_PAGE_BOUNDARY: u8 = TX_TOTAL_PAGES + 1;

    /// The highest LLT entry there is. The list runs past the transmit pages,
    /// because the buffer above the boundary is a ring the MAC uses for
    /// beacons or loopback, and it has to be threaded too.
    pub const LLT_LAST_ENTRY: u8 = 176;

    /// What the chip reads as *end of list* rather than as a page number.
    pub const LLT_END: u8 = 0xFF;

    /// The chain to write into the LLT: `(index, next)` pairs, in order.
    ///
    /// Three runs and they are not interchangeable. Every page below the
    /// boundary points at its successor. The last transmit page **ends the
    /// list** with `LLT_END` rather than pointing anywhere, which is the part
    /// I first got wrong: a transmit list that ran on into the ring would let
    /// a long frame spill into the beacon buffer. Then the ring above the
    /// boundary chains forward, and its last entry points back *at the
    /// boundary*, which is what makes it a ring.
    ///
    /// Untestable against the part, checkable as arithmetic, which is the
    /// whole reason it is a function returning a list rather than a loop that
    /// writes registers.
    pub fn llt_chain(boundary: u8, last: u8) -> Vec<(u8, u8)> {
        let mut out = Vec::new();
        if boundary == 0 || boundary > last {
            return out;
        }
        for i in 0..boundary - 1 {
            out.push((i, i + 1));
        }
        out.push((boundary - 1, LLT_END));
        for i in boundary..last {
            out.push((i, i + 1));
        }
        out.push((last, boundary));
        out
    }

    /// A firmware image, as `rtl8188eufw.bin` presents itself.
    ///
    /// The blob is **not in this repository and never will be**, for the same
    /// reason no WAD is: it is Realtek's, redistributable but not ours. It
    /// travels on the boot volume beside the model, and `parse` is what makes
    /// a truncated download fail here rather than as a chip that never
    /// answers.
    pub struct Firmware<'a> {
        pub signature: u16,
        pub version: u16,
        pub subversion: u8,
        pub body: &'a [u8],
    }

    /// The header is 32 bytes and the signature identifies the part.
    pub const FW_HEADER_LEN: usize = 32;
    /// What an 8188E image says it is. Two values because Realtek shipped
    /// both, and refusing one of them would refuse half the images in the
    /// world for no reason.
    pub const FW_SIGNATURE_88E: [u16; 2] = [0x88E1, 0x88E0];
    /// The chip takes the body in pages of this size.
    pub const FW_PAGE: usize = 4096;

    pub fn fw_parse(image: &[u8]) -> Option<Firmware<'_>> {
        if image.len() <= FW_HEADER_LEN {
            return None;
        }
        let signature = u16::from_le_bytes([image[0], image[1]]);
        if !FW_SIGNATURE_88E.contains(&signature) {
            return None;
        }
        Some(Firmware {
            signature,
            version: u16::from_le_bytes([image[4], image[5]]),
            subversion: image[6],
            body: &image[FW_HEADER_LEN..],
        })
    }

    /// How many pages a body takes, and how long the last one is.
    ///
    /// The chip is told the page number and the remainder separately, so a
    /// body that is an exact multiple of the page size is the case worth
    /// getting right: it has no short last page, and a loop that writes one
    /// anyway uploads four kilobytes of nothing over the end.
    pub fn fw_pages(body: &[u8]) -> (usize, usize) {
        (body.len() / FW_PAGE, body.len() % FW_PAGE)
    }

    /// 2.4 GHz channel numbers this chip will tune.
    ///
    /// One to fourteen exists in the spec and fourteen is Japan-only and
    /// 802.11b-only, so it is excluded rather than offered: a scan that probes
    /// it in a country that forbids it is a transmission that should not
    /// happen, and this driver has no regulatory database to know better.
    pub fn channel_ok(ch: u8) -> bool {
        (1..=13).contains(&ch)
    }

    /// The centre frequency of a channel, in MHz. Reported rather than
    /// programmed -- the radio is tuned by register, not by frequency -- and
    /// it is what makes a scan result legible.
    pub fn channel_mhz(ch: u8) -> Option<u16> {
        if !channel_ok(ch) {
            return None;
        }
        Some(2407 + 5 * ch as u16)
    }

    /// What `diag wifi` asks of the half that can be asked.
    pub fn checks() -> Vec<(&'static str, bool)> {
        let mut out = Vec::new();

        // One section at offset 1, all four words present. A clear bit means
        // present, which is the inversion this decoder exists to get right.
        let raw = [0x10u8, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22, 0xFF];
        let m = efuse_decode(&raw);
        out.push((
            "an efuse section lands at offset times eight, four words wide",
            m[8] == 0xAA && m[9] == 0xBB && m[14] == 0x11 && m[15] == 0x22,
        ));
        out.push((
            "a word whose enable bit is set is absent, and stays erased",
            {
                // Word 0 disabled: its bytes keep 0xFF and the rest shift up.
                let r = [0x11u8, 0xCC, 0xDD, 0xFF];
                let m = efuse_decode(&r);
                m[8] == 0xFF && m[9] == 0xFF && m[10] == 0xCC && m[11] == 0xDD
            },
        ));
        out.push((
            "a later section overwrites an earlier one, since the efuse is a replay",
            {
                let r = [0x10u8, 1, 1, 0, 0, 0, 0, 0, 0, 0x10, 2, 2, 0, 0, 0, 0, 0, 0, 0xFF];
                efuse_decode(&r)[8] == 2
            },
        ));
        out.push((
            "an extended header carries the offset's high bits",
            {
                // header 0x2F: low nibble 0x0F marks extended, high bits 0x01.
                // ext 0x30: offset[7:4] = 3, all four words present.
                let r = [0x2Fu8, 0x30, 0x99, 0x88, 0, 0, 0, 0, 0, 0, 0xFF];
                let m = efuse_decode(&r);
                let at = (((0x30u8 & 0xF0) >> 1) | (0x2Fu8 >> 5)) as usize * 8;
                m[at] == 0x99 && m[at + 1] == 0x88
            },
        ));
        out.push((
            "a truncated section stops the walk rather than reading past the end",
            {
                let _ = efuse_decode(&[0x10, 0xAA]);
                let _ = efuse_decode(&[0x0F]);
                let _ = efuse_decode(&[]);
                true
            },
        ));
        out.push((
            "an unprogrammed efuse yields no MAC rather than a broadcast one",
            {
                let all = [0xFFu8; EFUSE_MAP_LEN];
                let mut zero = [0u8; EFUSE_MAP_LEN];
                zero[EFUSE_MAC_OFFSET] = 0;
                efuse_mac(&all).is_none() && efuse_mac(&zero).is_none()
            },
        ));
        out.push((
            "and a multicast address is refused, since no card may own one",
            {
                let mut m = [0u8; EFUSE_MAP_LEN];
                m[EFUSE_MAC_OFFSET..EFUSE_MAC_OFFSET + 6]
                    .copy_from_slice(&[0x01, 0x22, 0x33, 0x44, 0x55, 0x66]);
                let mut ok = [0u8; EFUSE_MAP_LEN];
                ok[EFUSE_MAC_OFFSET..EFUSE_MAC_OFFSET + 6]
                    .copy_from_slice(&[0x00, 0x22, 0x33, 0x44, 0x55, 0x66]);
                efuse_mac(&m).is_none() && efuse_mac(&ok).is_some()
            },
        ));

        let chain = llt_chain(TX_PAGE_BOUNDARY, LLT_LAST_ENTRY);
        out.push((
            "every entry of the link list is written, and none of them twice",
            chain.len() == LLT_LAST_ENTRY as usize + 1
                && (0..=LLT_LAST_ENTRY).all(|i| {
                    chain.iter().filter(|&&(e, _)| e == i).count() == 1
                }),
        ));
        out.push((
            "every page but two points at the next one",
            chain.iter().all(|&(i, n)| {
                i == TX_PAGE_BOUNDARY - 1 || i == LLT_LAST_ENTRY || n == i + 1
            }),
        ));
        out.push((
            "the last transmit page ends the list rather than running on into the ring",
            chain
                .iter()
                .any(|&(i, n)| i == TX_PAGE_BOUNDARY - 1 && n == LLT_END),
        ));
        out.push((
            "and the last entry closes the ring at the boundary, rather than at zero",
            chain
                .iter()
                .any(|&(i, n)| i == LLT_LAST_ENTRY && n == TX_PAGE_BOUNDARY),
        ));
        out.push((
            "nothing above the boundary is reachable from the transmit list",
            {
                let mut at = 0u8;
                let mut seen = 0usize;
                loop {
                    match chain.iter().find(|&&(i, _)| i == at) {
                        Some(&(_, n)) if n != LLT_END && seen <= LLT_LAST_ENTRY as usize => {
                            at = n;
                            seen += 1;
                        }
                        _ => break,
                    }
                }
                at == TX_PAGE_BOUNDARY - 1 && seen == TX_PAGE_BOUNDARY as usize - 1
            },
        ));
        out.push((
            "a boundary past the last entry describes nothing rather than a wrong list",
            llt_chain(20, 10).is_empty() && llt_chain(0, 10).is_empty(),
        ));

        let mut img = alloc::vec![0u8; FW_HEADER_LEN + 5000];
        img[0..2].copy_from_slice(&FW_SIGNATURE_88E[0].to_le_bytes());
        img[4..6].copy_from_slice(&11u16.to_le_bytes());
        img[6] = 1;
        out.push((
            "a firmware image is its header and a body, and the version is read",
            fw_parse(&img).is_some_and(|f| {
                f.version == 11 && f.subversion == 1 && f.body.len() == 5000
            }),
        ));
        out.push((
            "an image with the wrong signature is refused, not uploaded",
            {
                let mut bad = img.clone();
                bad[0] = 0x00;
                bad[1] = 0x00;
                fw_parse(&bad).is_none()
            },
        ));
        out.push((
            "and one with no body at all is refused rather than uploaded empty",
            fw_parse(&img[..FW_HEADER_LEN]).is_none() && fw_parse(&[]).is_none(),
        ));
        out.push((
            "a body of exactly one page has no short last page",
            fw_pages(&alloc::vec![0u8; FW_PAGE]) == (1, 0)
                && fw_pages(&alloc::vec![0u8; FW_PAGE + 7]) == (1, 7)
                && fw_pages(&[]) == (0, 0),
        ));

        out.push((
            "channel 14 is not offered, since nothing here knows the regulatory domain",
            channel_ok(1) && channel_ok(13) && !channel_ok(14) && !channel_ok(0),
        ));
        out.push((
            "and a channel's frequency is the one the spec gives it",
            channel_mhz(1) == Some(2412)
                && channel_mhz(6) == Some(2437)
                && channel_mhz(13) == Some(2472)
                && channel_mhz(14).is_none(),
        ));
        out
    }
}

/// The descriptors that wrap a frame on its way to and from the chip.
///
/// Every constant these use is in `rtl8188eu_tables`, extracted from Linux by
/// `tools/rtlconv.py`. The layouts are from `rtl8xxxu_txdesc32` and
/// `rtl8xxxu_rxdesc16`, which is what the 8188E's entry in `8188e.c` selects.
///
/// This is the one part of the driver that can be checked without the
/// hardware, because it is arithmetic on bytes and nothing else, so it has a
/// selftest that runs at every boot.
pub mod desc {
    use super::super::rtl8188eu_tables as t;

    /// The transmit descriptor is 32 bytes and precedes the frame in the same
    /// bulk transfer.
    pub const TX_SIZE: usize = 32;

    /// Word 0 is not four flag bytes. It is a little-endian u16 length, then
    /// two single bytes, which is why the flags below are bits of a byte and
    /// `TXDESC_OWN` is bit 7 rather than bit 31. rtl8xxxu.h defines it both
    /// ways and the bit-31 form is inside an `#if 0`.
    const TXDW0_BYTE: usize = 3;

    /// Build the descriptor for one management frame.
    ///
    /// Management rather than data because that is what a scan sends, and the
    /// data path needs a rate-control decision this driver has no way to make
    /// yet. The rate is left at zero, which is 1 Mbit CCK: the slowest thing
    /// the radio can do and the one every access point in range can hear.
    ///
    /// Follows the common path in Linux's `rtl8xxxu_tx` together with the
    /// management branch of `rtl8xxxu_fill_txdesc_v3`.
    pub fn tx(frame_len: usize, seq: u16, broadcast: bool) -> [u8; TX_SIZE] {
        let mut d = [0u8; TX_SIZE];

        d[0..2].copy_from_slice(&(frame_len as u16).to_le_bytes());
        // Where the frame starts, measured from the front of the descriptor.
        d[2] = TX_SIZE as u8;

        let mut dw0 = t::TXDESC_OWN | t::TXDESC_FIRST_SEGMENT | t::TXDESC_LAST_SEGMENT;
        if broadcast {
            dw0 |= t::TXDESC_BROADMULTICAST;
        }
        d[TXDW0_BYTE] = dw0 as u8;

        put(&mut d, 1, t::TXDESC_QUEUE_MGNT << t::TXDESC_QUEUE_SHIFT);
        put(&mut d, 3, (seq as u32) << t::TXDESC32_SEQ_SHIFT);
        put(&mut d, 4, t::TXDESC32_USE_DRIVER_RATE);
        // Six retries, and the enable bit that makes the count mean anything.
        put(
            &mut d,
            5,
            (6 << t::TXDESC32_RETRY_LIMIT_SHIFT) | t::TXDESC32_RETRY_LIMIT_ENABLE,
        );
        d
    }

    /// Write one of the descriptor's 32-bit words, by index.
    fn put(d: &mut [u8; TX_SIZE], word: usize, v: u32) {
        let o = word * 4;
        d[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn get(d: &[u8], word: usize) -> u32 {
        u32::from_le_bytes([d[word * 4], d[word * 4 + 1], d[word * 4 + 2], d[word * 4 + 3]])
    }

    /// Pull one bitfield out, given its (word, shift, width).
    fn field(d: &[u8], f: (u32, u32, u32)) -> u32 {
        let (word, shift, width) = f;
        let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
        (get(d, word as usize) >> shift) & mask
    }

    /// What a receive descriptor says about the frame behind it.
    pub struct Rx {
        /// Where the frame starts, from the front of the descriptor.
        pub offset: usize,
        pub len: usize,
        /// The chip checked and the frame is damaged. Kept as a fact rather
        /// than dropped inside the parser, because a receive path that
        /// silently discards is one that cannot be debugged.
        pub bad: bool,
    }

    /// Read a receive descriptor.
    ///
    /// The frame does not begin at a fixed offset. Two variable pieces sit
    /// between: the PHY status block, whose size is given in eight-byte units,
    /// and a shift of up to three bytes the chip inserts so the frame's own
    /// header lands aligned. Both are added to the fixed part.
    ///
    /// Returns `None` when the buffer cannot hold what the descriptor claims,
    /// which is the case that matters: a corrupt length here is a read past
    /// the end of a DMA buffer.
    pub fn rx(buf: &[u8]) -> Option<Rx> {
        if buf.len() < t::RXDESC16_SIZE {
            return None;
        }
        let len = field(buf, t::RXDESC_PKTLEN) as usize;
        let drvinfo = field(buf, t::RXDESC_DRVINFO_SZ) as usize * 8;
        let shift = field(buf, t::RXDESC_SHIFT) as usize;
        let bad = field(buf, t::RXDESC_CRC32) != 0 || field(buf, t::RXDESC_ICVERR) != 0;
        let offset = t::RXDESC16_SIZE + drvinfo + shift;
        if len == 0 || offset.checked_add(len)? > buf.len() {
            return None;
        }
        Some(Rx { offset, len, bad })
    }

    /// Descriptors built here, taken apart here, and checked against the bit
    /// positions the constants claim.
    ///
    /// The chip is not present under emulation and never will be, so this is
    /// the only part of the driver that can fail visibly on the machine that
    /// writes it. It is worth having for one reason above the others: the
    /// OWN bit sits at bit 7 of a byte, the header offers a bit-31 definition
    /// of the same name from a dead preprocessor branch, and picking the wrong
    /// one produces a descriptor the chip ignores in silence.
    pub fn selftest() -> bool {
        let d = desc_or_bail();
        let Some(d) = d else { return false };
        let _ = d;
        true
    }

    fn desc_or_bail() -> Option<()> {
        let d = tx(64, 0x123, true);
        // Length and offset, where the chip looks for them.
        if u16::from_le_bytes([d[0], d[1]]) != 64 || d[2] != TX_SIZE as u8 {
            return None;
        }
        // OWN must be the top bit of byte 3, which is bit 31 of word 0.
        if d[TXDW0_BYTE] & 0x80 == 0 || get(&d, 0) & (1 << 31) == 0 {
            return None;
        }
        if d[TXDW0_BYTE] & 0x0C != 0x0C {
            return None; // first and last segment
        }
        if d[TXDW0_BYTE] & 0x01 == 0 {
            return None; // broadcast
        }
        // ...and not set when the frame is not broadcast.
        if tx(64, 0, false)[TXDW0_BYTE] & 0x01 != 0 {
            return None;
        }
        // The management queue, in the field the chip reads it from.
        if (get(&d, 1) & t::TXDESC_QUEUE_MASK) >> t::TXDESC_QUEUE_SHIFT != t::TXDESC_QUEUE_MGNT {
            return None;
        }
        if get(&d, 3) >> t::TXDESC32_SEQ_SHIFT != 0x123 {
            return None;
        }
        if get(&d, 4) & t::TXDESC32_USE_DRIVER_RATE == 0 {
            return None;
        }
        if get(&d, 5) & t::TXDESC32_RETRY_LIMIT_ENABLE == 0 {
            return None;
        }
        // Rate zero is 1 Mbit CCK, and is the bottom of word 5.
        if get(&d, 5) & 0xFF != 0 {
            return None;
        }

        // A receive descriptor, assembled to say something specific: 100-byte
        // frame, two eight-byte units of PHY status, shifted by three.
        let mut r = [0u8; 160];
        let w0 = 100u32 | (2 << 16) | (3 << 24);
        r[0..4].copy_from_slice(&w0.to_le_bytes());
        let got = rx(&r)?;
        if got.len != 100 || got.offset != t::RXDESC16_SIZE + 16 + 3 || got.bad {
            return None;
        }
        // The CRC bit is a verdict from the chip and must survive the parse.
        let mut bad = r;
        bad[0..4].copy_from_slice(&(w0 | (1 << 14)).to_le_bytes());
        if !rx(&bad)?.bad {
            return None;
        }
        // A length that runs off the end of the buffer is refused rather than
        // trusted, because the buffer is the end of a DMA transfer.
        let mut over = [0u8; 40];
        over[0..4].copy_from_slice(&100u32.to_le_bytes());
        if rx(&over).is_some() {
            return None;
        }
        if rx(&[]).is_some() {
            return None;
        }
        Some(())
    }
}
