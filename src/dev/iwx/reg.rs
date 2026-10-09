//! Which channels the part may use, as its firmware decides.
//!
//! **LAR is declared on the GF63's firmware, and LAR means the firmware owns
//! regulatory.** The NVM carries a channel profile, and on a part with
//! location-aware regulatory that profile is a starting point the firmware
//! replaces: the host asks with `MCC_UPDATE` and an alpha-2 code, and the reply is
//! the channel map that is actually in force. Upstream asks for `"ZZ"`, the world
//! domain, and lets the firmware narrow it from what it hears; so does this.
//!
//! The reply is a word per channel in the order of one of two fixed tables,
//! 51 entries without the 6 GHz band and 110 with it. **The table is chosen by
//! what the part supports, not by the reply's length**, because a reply may
//! carry fewer words than the table and the words that are there still mean the
//! table's channels in the table's order. Reading 6 GHz numbers as 2.4 GHz ones
//! is the failure: they reuse 1, 5, 9 and so on.

use alloc::vec::Vec;

pub const MCC_UPDATE_CMD: u8 = 0xc8;

/// The world domain, which firmware narrows from what it hears.
pub const WORLD: [u8; 2] = *b"ZZ";

const SOURCE_OLD_FW: u8 = 0;
const SOURCE_GET_CURRENT: u8 = 0x10;

/// The 2.4 and 5 GHz channels, in the order every channel profile uses.
pub const CHANNELS_8000: [u8; 51] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, //
    36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128,
    132, 136, 140, 144, 149, 153, 157, 161, 165, 169, 173, 177, 181,
];
/// How many of a profile's words are 2.4 GHz.
pub const N_24: usize = 14;
/// And how many of the remainder are 5 GHz. Past these is 6 GHz, which nothing
/// above this module can tune -- `dev::radio` numbers channels by one byte and
/// 6 GHz reuses the low numbers.
pub const N_5: usize = 37;

const CH_VALID: u32 = 1 << 0;
const CH_ACTIVE: u32 = 1 << 3;
const CH_RADAR: u32 = 1 << 4;

/// The request: an alpha-2 code and where it came from. Twenty-eight bytes,
/// `LAR_UPDATE_MCC_CMD_API_S_VER_2`.
///
/// `multi` is whether the firmware takes the newer source scheme -- the API bit
/// `WIFI_MCC_UPDATE` or the capability `LAR_MULTI_MCC`, either -- and it decides
/// the source byte, which is upstream's only branch here.
pub fn request(alpha2: [u8; 2], multi: bool) -> Vec<u8> {
    let mut v = alloc::vec![0u8; 28];
    v[0..2].copy_from_slice(&(((alpha2[0] as u16) << 8) | alpha2[1] as u16).to_le_bytes());
    v[2] = if multi { SOURCE_GET_CURRENT } else { SOURCE_OLD_FW };
    v
}

/// One usable channel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Channel {
    pub number: u8,
    /// Whether the part may transmit there unprompted. False means passive:
    /// listen for beacons, never probe, which is what DFS channels are.
    pub active: bool,
    pub radar: bool,
}

/// What the firmware said is in force.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Regulatory {
    pub status: u32,
    /// The code it settled on, as two ASCII letters where it is one.
    pub mcc: [u8; 2],
    pub channels: Vec<Channel>,
    /// Words the reply carried, before filtering.
    pub declared: u32,
}

impl Regulatory {
    pub fn say(&self) -> alloc::string::String {
        let printable = self.mcc.iter().all(|c| c.is_ascii_alphanumeric());
        let active = self.channels.iter().filter(|c| c.active).count();
        alloc::format!(
            "regulatory domain {}: {} usable channel(s) of {} declared, {} active, {} radar",
            if printable { core::str::from_utf8(&self.mcc).unwrap_or("??") } else { "unnamed" },
            self.channels.len(),
            self.declared,
            active,
            self.channels.iter().filter(|c| c.radar).count()
        )
    }

    /// The numbers, for a scan plan.
    pub fn numbers(&self) -> Vec<u8> {
        self.channels.iter().map(|c| c.number).collect()
    }
}

/// Which reply layout the firmware sends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// `LAR_UPDATE_MCC_CMD_RESP_S_VER_3`: sixteen bytes, the count at twelve.
    V3,
    /// Version 4: twenty bytes, the count at sixteen. Firmware declaring
    /// `MCC_UPDATE_11AX_SUPPORT` sends this one, the GF63's among them.
    V4,
}

impl Layout {
    /// Chosen by the capability, which is the rule Linux keeps. OpenBSD reads
    /// version 4 on both of its branches, which is right for every image it has
    /// met and wrong for one without the bit: four bytes of channel map read as
    /// header, and every channel after shifted by one.
    pub fn of(image: &super::fw::Image) -> Layout {
        if image.has_capa(super::fw::capa::MCC_UPDATE_11AX_SUPPORT) {
            Layout::V4
        } else {
            Layout::V3
        }
    }

    fn header(self) -> usize {
        match self {
            Layout::V3 => 16,
            Layout::V4 => 20,
        }
    }
}

/// The reply versions this reads. Upstream refuses eight and above.
pub const MAX_REPLY_VER: u8 = 7;

/// Read the reply: a header (`Layout`), then a word per channel.
///
/// The count in the header is checked against the length, as upstream checks it:
/// a reply whose words do not add up is refused rather than walked, because the
/// channel words are the tail and a wrong count is a map read from the wrong
/// bytes. `band_5` is the NVM's own SKU bit, and a part fused without 5 GHz has
/// its 5 GHz channels cleared whatever the reply says, which is upstream's rule.
pub fn response(payload: &[u8], band_5: bool, layout: Layout) -> Option<Regulatory> {
    let hdr = layout.header();
    if payload.len() < hdr {
        return None;
    }
    let le32 = |at: usize| u32::from_le_bytes([payload[at], payload[at + 1], payload[at + 2], payload[at + 3]]);
    let status = le32(0);
    let mcc = u16::from_le_bytes([payload[4], payload[5]]);
    let n = le32(hdr - 4);
    if (payload.len() - hdr) as u64 != n as u64 * 4 {
        return None;
    }
    let mut channels = Vec::new();
    for i in 0..(n as usize).min(N_24 + N_5) {
        let flags = le32(hdr + i * 4);
        if flags & CH_VALID == 0 || (i >= N_24 && !band_5) {
            continue;
        }
        channels.push(Channel {
            number: CHANNELS_8000[i],
            active: flags & CH_ACTIVE != 0,
            radar: flags & CH_RADAR != 0,
        });
    }
    Some(Regulatory { status, mcc: [(mcc >> 8) as u8, mcc as u8], channels, declared: n })
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let r = request(WORLD, true);
    out.push((
        "the request is twenty-eight bytes, ZZ high byte first, asking for the current source",
        r.len() == 28 && r[0..2] == [0x5A, 0x5A] && r[2] == SOURCE_GET_CURRENT && r[3..].iter().all(|&b| b == 0),
    ));
    out.push(("and an old firmware is asked the old way", request(WORLD, false)[2] == SOURCE_OLD_FW));
    out.push(("the channel table is 14 then 37, ending at 181", CHANNELS_8000.len() == N_24 + N_5 && CHANNELS_8000[N_24] == 36 && CHANNELS_8000[50] == 181));
    // A reply naming DE: channel 1 active, 13 valid and passive, 36 active, 52 radar.
    let mut p = alloc::vec![0u8; 20];
    p[4..6].copy_from_slice(&0x4445u16.to_le_bytes());
    let mut words = alloc::vec![0u32; 110];
    words[0] = CH_VALID | CH_ACTIVE;
    words[12] = CH_VALID;
    words[14] = CH_VALID | CH_ACTIVE;
    words[18] = CH_VALID | CH_RADAR;
    // A 6 GHz word that would read as channel 1 if the table were the wrong one.
    words[51] = CH_VALID | CH_ACTIVE;
    p[16..20].copy_from_slice(&110u32.to_le_bytes());
    for w in &words {
        p.extend_from_slice(&w.to_le_bytes());
    }
    let reg = response(&p, true, Layout::V4);
    out.push((
        "a reply's valid channels are read by position, with their flags",
        reg.as_ref().map(|r| r.channels.clone())
            == Some(alloc::vec![
                Channel { number: 1, active: true, radar: false },
                Channel { number: 13, active: false, radar: false },
                Channel { number: 36, active: true, radar: false },
                Channel { number: 52, active: false, radar: true },
            ]),
    ));
    out.push((
        "and its code is two letters, high byte first",
        reg.as_ref().map(|r| r.mcc) == Some(*b"DE"),
    ));
    out.push((
        "a 6 GHz word is not read as a 2.4 GHz channel that shares its number",
        reg.as_ref().map(|r| r.channels.iter().filter(|c| c.number == 1).count()) == Some(1),
    ));
    out.push((
        "a part fused without 5 GHz keeps none, whatever the reply says",
        response(&p, false, Layout::V4).map(|r| r.channels.iter().all(|c| c.number <= 14)) == Some(true),
    ));
    let mut short = p.clone();
    short.truncate(p.len() - 4);
    out.push(("a reply whose count and length disagree is refused", response(&short, true, Layout::V4).is_none()));
    out.push(("and one too short for its header", response(&p[..12], true, Layout::V4).is_none()));
    // The same map in the sixteen-byte layout: the count moves to twelve, and
    // every word moves up four.
    let mut v3 = p[..12].to_vec();
    v3.extend_from_slice(&p[16..]);
    out.push((
        "a version-3 reply is read with its own header, to the same channels",
        response(&v3, true, Layout::V3).map(|r| r.channels) == reg.as_ref().map(|r| r.channels.clone()),
    ));
    out.push((
        "and read as version 4 it does not add up, so it is refused rather than shifted",
        response(&v3, true, Layout::V4).is_none(),
    ));
    out
}
