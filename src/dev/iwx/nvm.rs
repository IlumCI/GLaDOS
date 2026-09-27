//! What the part knows about itself: its address, its bands, its channels.
//!
//! The NVM is not a region the host reads. It is a question put to firmware --
//! `NVM_GET_INFO`, in the regulatory group -- whose answer carries the version,
//! the number of addresses the part was fused with, which standards and bands the
//! SKU permits, how many antenna chains it has, and a flag word per channel.
//!
//! ### Except the MAC address, which has a register path and needs it
//!
//! The address is read out of two registers rather than taken from the response,
//! and upstream does the same: the response does not carry it. There are **two
//! pairs** -- a strap pair the OEM may have fused and an OTP pair -- and the rule
//! is to prefer the strap and fall back to OTP when the strap holds nothing
//! valid. A driver that read only one pair gets a working address on most machines
//! and a broadcast address on the rest.
//!
//! **And validity is four separate refusals**, which is why it is a function
//! rather than a null check: all zeroes, all ones, a multicast bit set, and one
//! specific reserved address Intel ships in unfused parts. A driver checking only
//! for zero accepts `02:cc:aa:ff:ee:00` and associates with it, and every such
//! machine has the same address.
//!
//! ### The register base is family-dependent and the two are far apart
//!
//! `CSR_ADDR_BASE` is 0x380 on this family and 0x30 on Bz. Not adjacent, not
//! derivable, and both are legal offsets inside the aperture -- so reading the
//! wrong one returns whatever lives there and produces an address rather than an
//! error.
//!
//! ### Provenance
//!
//! As the rest of `iwx`: numbers from OpenBSD's `iwx(4)`, Intel's dual BSD/GPLv2
//! headers underneath, BSD arm, `NOTICE.md`.

use alloc::string::String;
use alloc::vec::Vec;

use super::cmd;

/// Re-exported so a caller naming this command names one module, not two.
pub use super::cmd::REGULATORY_AND_NVM_GROUP;
use super::Family;

/// The opcode, in the regulatory group.
pub const NVM_GET_INFO: u8 = 0x02;

/// The payload: one reserved word. A command with nothing to say still has to say
/// four bytes of it.
pub const REQUEST: [u8; 4] = [0; 4];

/// Where the address registers begin, per family.
///
/// A function rather than a constant, because the two families put them 848 bytes
/// apart and both offsets are inside the aperture -- so the wrong one reads
/// something rather than failing.
pub fn addr_base(family: Family) -> u64 {
    match family {
        Family::Bz => 0x30,
        Family::F22000 | Family::Ax210 => 0x380,
    }
}

/// The reserved address an unfused part reports. Refusing it is the whole reason
/// validity is more than a null check: it is a perfectly well-formed unicast
/// address, and every machine that shipped unfused has this one.
pub const RESERVED_MAC: [u8; 6] = [0x02, 0xcc, 0xaa, 0xff, 0xee, 0x00];

/// Is this an address the part can use?
pub fn valid_mac(a: &[u8; 6]) -> bool {
    if *a == RESERVED_MAC {
        return false;
    }
    if a.iter().all(|&b| b == 0) || a.iter().all(|&b| b == 0xff) {
        return false;
    }
    // The multicast bit. A station with a group address set would be transmitting
    // as something no access point will ever reply to.
    if a[0] & 1 != 0 {
        return false;
    }
    true
}

/// Turn the two registers into an address.
///
/// **Both halves are byte-reversed and the second is only sixteen bits wide.** The
/// first register's four bytes come out in the opposite order, and the second
/// contributes its low two, also reversed. Upstream spells it as six indexed
/// assignments through a pointer; written as two big-endian conversions here,
/// which is the same permutation and says what it is.
pub fn flip(addr0: u32, addr1: u32) -> [u8; 6] {
    let a = addr0.to_be_bytes();
    let b = (addr1 as u16).to_be_bytes();
    [a[0], a[1], a[2], a[3], b[0], b[1]]
}

// --- the response ------------------------------------------------------------

/// Offsets in the response. General, SKU, PHY, then the regulatory block.
pub mod at {
    pub const FLAGS: usize = 0;
    pub const NVM_VERSION: usize = 4;
    pub const BOARD_TYPE: usize = 6;
    pub const N_HW_ADDRS: usize = 7;
    pub const MAC_SKU_FLAGS: usize = 8;
    pub const TX_CHAINS: usize = 12;
    pub const RX_CHAINS: usize = 16;
    pub const LAR_ENABLED: usize = 20;
    pub const N_CHANNELS: usize = 24;
    pub const CHANNEL_PROFILE: usize = 28;
}

/// How many channel words each version of the response carries.
pub const CHANNELS_V3: usize = 51;
pub const CHANNELS_V4: usize = 110;
/// The two payload lengths, which is what distinguishes the versions.
pub const RSP_V3: usize = at::CHANNEL_PROFILE + CHANNELS_V3 * 4;
pub const RSP_V4: usize = at::CHANNEL_PROFILE + CHANNELS_V4 * 4;

const SKU_BAND_24: u32 = 1 << 0;
const SKU_BAND_52: u32 = 1 << 1;
const SKU_11N: u32 = 1 << 2;
const SKU_11AC: u32 = 1 << 3;
const SKU_11AX: u32 = 1 << 4;
const SKU_MIMO_DISABLED: u32 = 1 << 5;

const CH_VALID: u32 = 1 << 0;
const CH_ACTIVE: u32 = 1 << 3;
const CH_RADAR: u32 = 1 << 4;
const CH_INDOOR_ONLY: u32 = 1 << 5;
const CH_40MHZ: u32 = 1 << 9;
const CH_80MHZ: u32 = 1 << 10;
const CH_160MHZ: u32 = 1 << 11;

/// What the part said about itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Nvm {
    /// 3 or 4, from the payload's length.
    pub version: u8,
    pub nvm_version: u16,
    pub board_type: u8,
    /// How many addresses the part was fused with. Zero is a part with none, which
    /// is a different fact from an address that will not pass validity.
    pub n_hw_addrs: u8,
    pub band_24: bool,
    pub band_52: bool,
    pub n11: bool,
    pub ac: bool,
    pub ax: bool,
    /// Note the sense: the flag means MIMO is *disabled*, so this is the flag and
    /// not the capability, named to match.
    pub mimo_disabled: bool,
    pub tx_chains: u8,
    pub rx_chains: u8,
    /// Whether the part expects the host to apply a learned regulatory profile.
    pub lar: bool,
    /// How many channel words the response declared, which need not be how many it
    /// carried.
    pub declared_channels: u32,
    /// How many are usable, by flag.
    pub valid_channels: u32,
    pub active_channels: u32,
    pub radar_channels: u32,
    pub indoor_channels: u32,
    pub wide_40: u32,
    pub wide_80: u32,
    pub wide_160: u32,
    /// Read from the registers rather than the response, and possibly not valid.
    pub mac: [u8; 6],
}

impl Nvm {
    pub fn say(&self) -> String {
        alloc::format!(
            "v{} nvm {:#06x} board {}, {} address(es), {}{}{}{}{} {}x{} chains",
            self.version,
            self.nvm_version,
            self.board_type,
            self.n_hw_addrs,
            if self.band_24 { "2.4 " } else { "" },
            if self.band_52 { "5 " } else { "" },
            if self.n11 { "n" } else { "" },
            if self.ac { "/ac" } else { "" },
            if self.ax { "/ax" } else { "" },
            self.tx_chains.count_ones(),
            self.rx_chains.count_ones()
        )
    }

    pub fn channels(&self) -> String {
        alloc::format!(
            "{} of {} channel(s) usable, {} active, {} radar, {} indoor-only; {}/{}/{} at 40/80/160",
            self.valid_channels,
            self.declared_channels,
            self.active_channels,
            self.radar_channels,
            self.indoor_channels,
            self.wide_40,
            self.wide_80,
            self.wide_160
        )
    }
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parse the response.
///
/// **Dispatched on the payload's length**, as the ALIVE notification is, and for
/// the same reason: upstream decides between v3 and v4 by looking up an API bit in
/// the firmware file, and the two lengths already answer it -- 232 and 468 -- so
/// the bytes in hand settle it with no second source to disagree with them. A
/// length that is neither is refused rather than parsed as the nearer, because the
/// channel block is the tail of the structure and a wrong version reads the wrong
/// number of words from it.
///
/// The declared channel count is **not** trusted for the walk. It is a number
/// inside a payload whose length is already known, so a response claiming more
/// channels than it carries would walk off the end; the walk uses the length and
/// the count is reported beside it, which is how a disagreement becomes visible
/// rather than fatal.
pub fn parse(payload: &[u8], mac: [u8; 6]) -> Option<Nvm> {
    let (version, n) = match payload.len() {
        RSP_V3 => (3u8, CHANNELS_V3),
        RSP_V4 => (4, CHANNELS_V4),
        _ => return None,
    };
    let sku = le32(payload, at::MAC_SKU_FLAGS);
    let mut v = Nvm {
        version,
        nvm_version: u16::from_le_bytes([payload[at::NVM_VERSION], payload[at::NVM_VERSION + 1]]),
        board_type: payload[at::BOARD_TYPE],
        n_hw_addrs: payload[at::N_HW_ADDRS],
        band_24: sku & SKU_BAND_24 != 0,
        band_52: sku & SKU_BAND_52 != 0,
        n11: sku & SKU_11N != 0,
        ac: sku & SKU_11AC != 0,
        ax: sku & SKU_11AX != 0,
        mimo_disabled: sku & SKU_MIMO_DISABLED != 0,
        // The chain masks are 32-bit fields holding a handful of bits; the low
        // byte is what upstream keeps, and `count_ones` on it is the antenna
        // count.
        tx_chains: le32(payload, at::TX_CHAINS) as u8,
        rx_chains: le32(payload, at::RX_CHAINS) as u8,
        lar: le32(payload, at::LAR_ENABLED) != 0,
        declared_channels: le32(payload, at::N_CHANNELS),
        valid_channels: 0,
        active_channels: 0,
        radar_channels: 0,
        indoor_channels: 0,
        wide_40: 0,
        wide_80: 0,
        wide_160: 0,
        mac,
    };
    for i in 0..n {
        let f = le32(payload, at::CHANNEL_PROFILE + i * 4);
        // Everything else is only meaningful on a valid channel. Counting an
        // 80 MHz flag on an invalid one would report width the part does not have.
        if f & CH_VALID == 0 {
            continue;
        }
        v.valid_channels += 1;
        if f & CH_ACTIVE != 0 {
            v.active_channels += 1;
        }
        if f & CH_RADAR != 0 {
            v.radar_channels += 1;
        }
        if f & CH_INDOOR_ONLY != 0 {
            v.indoor_channels += 1;
        }
        if f & CH_40MHZ != 0 {
            v.wide_40 += 1;
        }
        if f & CH_80MHZ != 0 {
            v.wide_80 += 1;
        }
        if f & CH_160MHZ != 0 {
            v.wide_160 += 1;
        }
    }
    Some(v)
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum NvmError {
    Cmd(cmd::CmdError),
    /// The part was never told the question was coming. Its own variant because
    /// this is a failure *before* the NVM was asked about, and reporting it as a
    /// command failure would send a reader to look at `NVM_GET_INFO`.
    Init(super::init::Fault),
    /// The answer was a length no version of the response has.
    UnknownVersion(usize),
    /// No usable address in either register pair. Carries what was read, because
    /// which invalid address it is says whether the part is unfused, asleep, or
    /// being read at the wrong offset.
    NoAddress([u8; 6]),
}

impl NvmError {
    pub fn why(&self) -> String {
        match self {
            NvmError::Cmd(c) => c.why(),
            NvmError::Init(f) => f.why(),
            NvmError::UnknownVersion(n) => {
                alloc::format!("the answer is {} bytes, which is no version of it", n)
            }
            NvmError::NoAddress(a) => alloc::format!(
                "no valid address in either register pair: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}{}",
                a[0], a[1], a[2], a[3], a[4], a[5],
                if *a == RESERVED_MAC {
                    " -- Intel's reserved address, so the part was never fused"
                } else if a.iter().all(|&b| b == 0xff) {
                    " -- all ones, so the registers are not answering"
                } else if a.iter().all(|&b| b == 0) {
                    " -- all zeroes"
                } else {
                    ""
                }
            ),
        }
    }
}

/// Read the address out of the part's registers.
///
/// **Both pairs, strap first.** The strap pair is what an OEM fuses and the OTP
/// pair is the fallback; upstream tries the strap, checks validity, and only then
/// reads OTP. Reading one pair works on most machines and returns a broadcast
/// address on the rest, which is the kind of thing that looks like a driver bug
/// on somebody else's laptop.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part holding the MAC access lock -- the
/// registers answer only while it is held, which is why this is not called
/// without one.
pub unsafe fn read_mac(bar0: u64, family: Family) -> [u8; 6] {
    let base = bar0 + addr_base(family);
    let rd = |off: u64| core::ptr::read_volatile((base + off) as *const u32);
    // Strap at +8/+12, OTP at +0/+4. Note the strap is the *later* pair and the
    // preferred one, so the offsets do not run in preference order.
    let strap = flip(rd(0x08), rd(0x0c));
    if valid_mac(&strap) {
        return strap;
    }
    flip(rd(0x00), rd(0x04))
}

/// Claims. No radio, and no register read.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    // --- the two lengths ----------------------------------------------------

    ok(at::CHANNEL_PROFILE == 28, "the channel block begins 28 bytes in");
    ok(RSP_V3 == 232, "version 3 carries 51 channels, so 232 bytes");
    ok(RSP_V4 == 468, "version 4 carries 110, so 468");
    ok(RSP_V3 != RSP_V4, "and the two differ, which is what lets the length pick the version");

    // --- the address --------------------------------------------------------

    // The two bases are far apart and both legal, so the wrong one reads
    // something rather than failing.
    ok(addr_base(Family::Ax210) == 0x380, "the address registers are at 0x380 on this family");
    ok(addr_base(Family::Bz) == 0x30, "and at 0x30 on Bz");
    ok(addr_base(Family::Ax210) != addr_base(Family::Bz), "which is a difference of 848 bytes, not an adjacent pair");
    ok(addr_base(Family::Ax210) + 12 < super::APERTURE, "and all four are inside the aperture");

    // The permutation. Both halves reversed and the second only sixteen bits: a
    // straight little-endian read gives the address backwards, which is a
    // perfectly plausible address.
    ok(
        flip(0x0011_2233, 0x0000_4455) == [0x00, 0x11, 0x22, 0x33, 0x44, 0x55],
        "the two registers unpack big-endian, four bytes then two",
    );
    ok(
        flip(0x3322_1100, 0x0000_5544) != [0x00, 0x11, 0x22, 0x33, 0x44, 0x55],
        "so a little-endian reading gives a different address, not an error",
    );
    // The high half of the second register is not part of the address.
    ok(
        flip(0x0011_2233, 0xffff_4455) == flip(0x0011_2233, 0x0000_4455),
        "and the second register's high half is ignored",
    );

    // Validity, which is four refusals and not a null check.
    ok(valid_mac(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]), "an ordinary address is valid");
    ok(!valid_mac(&[0; 6]), "all zeroes is not");
    ok(!valid_mac(&[0xff; 6]), "nor all ones");
    // The one that earns its place: a well-formed unicast address every unfused
    // part reports, which a null check accepts and every such machine shares.
    ok(!valid_mac(&RESERVED_MAC), "nor Intel's reserved address for an unfused part");
    ok(!valid_mac(&[0x01, 0x11, 0x22, 0x33, 0x44, 0x55]), "nor one with the multicast bit set");
    ok(valid_mac(&[0x02, 0x11, 0x22, 0x33, 0x44, 0x55]), "though the locally-administered bit is fine");

    // --- parsing an answer --------------------------------------------------

    let mut p = alloc::vec![0u8; RSP_V4];
    p[at::NVM_VERSION..at::NVM_VERSION + 2].copy_from_slice(&0x0a0bu16.to_le_bytes());
    p[at::BOARD_TYPE] = 5;
    p[at::N_HW_ADDRS] = 2;
    p[at::MAC_SKU_FLAGS..at::MAC_SKU_FLAGS + 4]
        .copy_from_slice(&(SKU_BAND_24 | SKU_BAND_52 | SKU_11N | SKU_11AC | SKU_11AX).to_le_bytes());
    p[at::TX_CHAINS..at::TX_CHAINS + 4].copy_from_slice(&3u32.to_le_bytes());
    p[at::RX_CHAINS..at::RX_CHAINS + 4].copy_from_slice(&3u32.to_le_bytes());
    p[at::LAR_ENABLED..at::LAR_ENABLED + 4].copy_from_slice(&1u32.to_le_bytes());
    p[at::N_CHANNELS..at::N_CHANNELS + 4].copy_from_slice(&(CHANNELS_V4 as u32).to_le_bytes());
    // Three channels: one plain and valid, one valid with every width and a radar
    // flag, one invalid but carrying width flags -- which must not be counted.
    let ch = |i: usize| at::CHANNEL_PROFILE + i * 4;
    p[ch(0)..ch(0) + 4].copy_from_slice(&(CH_VALID | CH_ACTIVE).to_le_bytes());
    p[ch(1)..ch(1) + 4].copy_from_slice(
        &(CH_VALID | CH_RADAR | CH_INDOOR_ONLY | CH_40MHZ | CH_80MHZ | CH_160MHZ).to_le_bytes(),
    );
    p[ch(2)..ch(2) + 4].copy_from_slice(&(CH_ACTIVE | CH_80MHZ | CH_160MHZ).to_le_bytes());
    let mac = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    match parse(&p, mac) {
        Some(n) => {
            ok(n.version == 4, "a 468-byte answer parses as version 4");
            ok(n.nvm_version == 0x0a0b && n.board_type == 5, "the version and board type are read");
            ok(n.n_hw_addrs == 2, "and how many addresses the part was fused with");
            ok(n.band_24 && n.band_52, "both bands are permitted");
            ok(n.n11 && n.ac && n.ax, "and all three standards");
            ok(!n.mimo_disabled, "with MIMO not disabled");
            ok(n.tx_chains.count_ones() == 2 && n.rx_chains.count_ones() == 2, "two chains each way");
            ok(n.lar, "and a learned regulatory profile is expected");
            ok(n.declared_channels == CHANNELS_V4 as u32, "the declared channel count is reported");
            // The counting rule: two valid of three, and the invalid one's width
            // flags are not counted even though they are set.
            ok(n.valid_channels == 2, "two of the three channels are valid");
            ok(n.active_channels == 1, "one is active");
            ok(n.radar_channels == 1 && n.indoor_channels == 1, "one carries radar and indoor-only");
            ok(
                n.wide_40 == 1 && n.wide_80 == 1 && n.wide_160 == 1,
                "and the widths of an invalid channel are not counted, though it declares them",
            );
            ok(n.mac == mac, "the address comes from the registers, not the answer");
        }
        None => ok(false, "a 468-byte answer parses"),
    }
    // Version 3, the shorter one.
    let mut q = alloc::vec![0u8; RSP_V3];
    q[ch(0)..ch(0) + 4].copy_from_slice(&CH_VALID.to_le_bytes());
    ok(
        parse(&q, mac).map(|n| (n.version, n.valid_channels)) == Some((3, 1)),
        "a 232-byte answer parses as version 3",
    );
    ok(parse(&alloc::vec![0u8; 300], mac).is_none(), "a length neither version has is refused");

    // A response claiming more channels than it carries. The walk uses the
    // length, so this is reported rather than fatal -- and it is the shape that
    // would walk off the end if the count were trusted.
    let mut lying = alloc::vec![0u8; RSP_V3];
    lying[at::N_CHANNELS..at::N_CHANNELS + 4].copy_from_slice(&9999u32.to_le_bytes());
    ok(
        parse(&lying, mac).map(|n| (n.declared_channels, n.valid_channels)) == Some((9999, 0)),
        "a count larger than the payload is reported, not walked",
    );

    ok(NVM_GET_INFO == 0x02 && REGULATORY_AND_NVM_GROUP == 0xc, "the command is opcode 2 of group 0xc");
    ok(REQUEST.len() == 4, "and carries four reserved bytes, because a command with nothing to say still says four");

    out
}
