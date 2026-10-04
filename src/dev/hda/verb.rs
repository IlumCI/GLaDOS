//! Verbs: how anything is said to a codec.
//!
//! A codec is a graph of numbered widgets -- converters, mixers, selectors,
//! pins -- and the only way to read or change any of it is a 32-bit command
//! naming a codec, a widget and what to do, answered by a 32-bit response.
//! Everything in this file is arithmetic on those words, so it is checked with
//! no controller at all.
//!
//!     31    28 27        20 19                                 0
//!     +-------+------------+------------------------------------+
//!     | codec |    node    |  12-bit verb + 8-bit payload, or   |
//!     |       |            |   4-bit verb + 16-bit payload      |
//!     +-------+------------+------------------------------------+
//!
//! **Two verb widths share one field**, told apart by the verb itself: the
//! amplifier and the converter format are the 4-bit kind, because their
//! payloads are sixteen bits wide, and everything else is 12-bit. Encoding a
//! 4-bit verb as a 12-bit one puts its payload's top nibble into the verb and
//! asks the codec something else entirely, and the codec answers it.
//!
//! The numbers are the Intel High Definition Audio Specification's, revision
//! 1.0a, which Intel publishes for exactly this.

/// Compose a command.
pub const fn command(codec: u8, node: u8, verb: u32) -> u32 {
    ((codec as u32 & 0xF) << 28) | ((node as u32) << 20) | (verb & 0xF_FFFF)
}

/// A 12-bit verb with its 8-bit payload.
pub const fn v12(verb: u32, payload: u8) -> u32 {
    ((verb & 0xFFF) << 8) | payload as u32
}

/// A 4-bit verb with its 16-bit payload.
pub const fn v4(verb: u32, payload: u16) -> u32 {
    ((verb & 0xF) << 16) | payload as u32
}

// 12-bit verbs.
pub const GET_PARAMETER: u32 = 0xF00;
pub const GET_CONNECTION_SELECT: u32 = 0xF01;
pub const SET_CONNECTION_SELECT: u32 = 0x701;
pub const GET_CONNECTION_ENTRY: u32 = 0xF02;
pub const SET_POWER_STATE: u32 = 0x705;
pub const SET_STREAM_CHANNEL: u32 = 0x706;
pub const SET_PIN_CONTROL: u32 = 0x707;
pub const SET_EAPD: u32 = 0x70C;
pub const GET_CONFIG_DEFAULT: u32 = 0xF1C;
pub const FUNCTION_RESET: u32 = 0x7FF;

// 4-bit verbs.
pub const SET_FORMAT: u32 = 0x2;
pub const SET_AMP: u32 = 0x3;

// Parameters, for GET_PARAMETER.
pub const P_VENDOR: u8 = 0x00;
pub const P_REVISION: u8 = 0x02;
pub const P_NODES: u8 = 0x04;
pub const P_FUNCTION_TYPE: u8 = 0x05;
pub const P_WIDGET_CAPS: u8 = 0x09;
pub const P_PIN_CAPS: u8 = 0x0C;
pub const P_CONN_LEN: u8 = 0x0E;
pub const P_AMP_OUT_CAPS: u8 = 0x12;

/// What a function group is.
pub const FG_AUDIO: u32 = 0x01;

/// Pin control: drive the output, and the headphone amplifier.
pub const PIN_OUT: u8 = 1 << 6;
pub const PIN_HP: u8 = 1 << 7;
/// EAPD: the external amplifier's enable, which on a laptop is usually the
/// speaker amplifier itself -- the ALC256 in the GF63 has it on both outputs.
pub const EAPD: u8 = 1 << 1;

/// A widget's type, from bits 23:20 of its capabilities.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Output,
    Input,
    Mixer,
    Selector,
    Pin,
    Power,
    Knob,
    Beep,
    Vendor,
    Other(u8),
}

impl Kind {
    pub fn of(caps: u32) -> Kind {
        match (caps >> 20) & 0xF {
            0 => Kind::Output,
            1 => Kind::Input,
            2 => Kind::Mixer,
            3 => Kind::Selector,
            4 => Kind::Pin,
            5 => Kind::Power,
            6 => Kind::Knob,
            7 => Kind::Beep,
            0xF => Kind::Vendor,
            n => Kind::Other(n as u8),
        }
    }
}

/// Widget capability bits worth naming.
pub const WCAP_STEREO: u32 = 1 << 0;
pub const WCAP_AMP_IN: u32 = 1 << 1;
pub const WCAP_AMP_OUT: u32 = 1 << 2;
pub const WCAP_CONN_LIST: u32 = 1 << 8;
pub const WCAP_DIGITAL: u32 = 1 << 9;
pub const WCAP_POWER: u32 = 1 << 10;

/// Pin capability bits worth naming.
pub const PCAP_OUT: u32 = 1 << 4;
pub const PCAP_HP: u32 = 1 << 3;
pub const PCAP_EAPD: u32 = 1 << 16;

/// What a pin's default configuration says it is wired to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PinDefault {
    /// 0 a jack, 1 nothing, 2 fixed (built in), 3 both.
    pub connectivity: u8,
    /// 0 line out, 1 speaker, 2 headphone, 5 SPDIF out, 0xA microphone, ...
    pub device: u8,
    pub assoc: u8,
    pub seq: u8,
}

impl PinDefault {
    pub fn of(cfg: u32) -> PinDefault {
        PinDefault {
            connectivity: (cfg >> 30) as u8,
            device: ((cfg >> 20) & 0xF) as u8,
            assoc: ((cfg >> 4) & 0xF) as u8,
            seq: (cfg & 0xF) as u8,
        }
    }

    pub fn connected(&self) -> bool {
        self.connectivity != 1
    }

    pub fn name(&self) -> &'static str {
        match self.device {
            0 => "line out",
            1 => "speaker",
            2 => "headphones",
            3 => "CD",
            4 => "SPDIF out",
            5 => "digital out",
            8 => "line in",
            0xA => "microphone",
            0xB => "SPDIF in",
            _ => "other",
        }
    }

    /// How much this pin is wanted as the machine's output, highest first: a
    /// built-in speaker, then headphones, then a line out. A jack nothing is
    /// plugged into still ranks -- presence detection is not asked here, so a
    /// laptop plays through its speaker and a desktop through its line out.
    pub fn rank(&self) -> Option<u8> {
        if !self.connected() {
            return None;
        }
        match (self.device, self.connectivity) {
            (1, 2) => Some(4),
            (1, _) => Some(3),
            (2, _) => Some(2),
            (0, _) => Some(1),
            _ => None,
        }
    }
}

/// The stream format word, both for the converter and the stream descriptor:
/// rate, depth and channel count. Only the rates on the 48 kHz and 44.1 kHz
/// bases with no multiplier or divider -- the two every codec takes.
pub fn format(rate: u32, bits: u8, channels: u8) -> Option<u16> {
    let base = match rate {
        48_000 => 0,
        44_100 => 1 << 14,
        _ => return None,
    };
    let depth = match bits {
        8 => 0,
        16 => 1,
        20 => 2,
        24 => 3,
        32 => 4,
        _ => return None,
    };
    if channels == 0 || channels > 16 {
        return None;
    }
    Some(base | (depth << 4) | (channels as u16 - 1))
}

/// Set an output amplifier: both channels, unmuted, at `gain` steps (clamped
/// by the caller to the widget's own range).
pub fn amp_out(gain: u8) -> u16 {
    (1 << 15) | (1 << 13) | (1 << 12) | (gain as u16 & 0x7F)
}

/// Set an input amplifier, at one connection index, unmuted.
pub fn amp_in(index: u8, gain: u8) -> u16 {
    (1 << 14) | (1 << 13) | (1 << 12) | ((index as u16 & 0xF) << 8) | (gain as u16 & 0x7F)
}

/// Connection list entries out of one GET_CONNECTION_ENTRY answer: four 8-bit
/// node ids in the short form, two 16-bit ones in the long. A range bit says
/// the entry is the end of a run starting at the one before it.
pub fn connections(word: u32, long: bool) -> alloc::vec::Vec<(u16, bool)> {
    let mut out = alloc::vec::Vec::new();
    if long {
        for i in 0..2 {
            let e = (word >> (i * 16)) & 0xFFFF;
            out.push(((e & 0x7FFF) as u16, e & 0x8000 != 0));
        }
    } else {
        for i in 0..4 {
            let e = (word >> (i * 8)) & 0xFF;
            out.push(((e & 0x7F) as u16, e & 0x80 != 0));
        }
    }
    out
}

pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();
    out.push((
        "a command is codec, node and verb, in that order down the word",
        command(2, 0x14, v12(GET_PARAMETER, P_WIDGET_CAPS)) == 0x214F_0009,
    ));
    out.push((
        "a 4-bit verb keeps its whole 16-bit payload: unmute both output channels at 0x40",
        command(0, 0x02, v4(SET_AMP, amp_out(0x40))) == 0x0023_B040,
    ));
    out.push((
        "48 kHz, 16 bits, stereo is 0x0011; 44.1 kHz sets the base bit",
        format(48_000, 16, 2) == Some(0x0011) && format(44_100, 16, 2) == Some(0x4011),
    ));
    out.push(("and a rate neither base reaches unscaled is refused", format(22_050, 16, 2).is_none()));
    // The GF63's own ALC256, as Linux read its defaults.
    let speaker = PinDefault::of(0x9017_0110);
    let phones = PinDefault::of(0x0321_4020);
    let unused = PinDefault::of(0x4111_11f0);
    out.push((
        "the GF63's internal speaker reads as a fixed speaker, its jack as headphones, and an unused pin as nothing",
        speaker.connectivity == 2 && speaker.device == 1 && phones.device == 2 && phones.connectivity == 0
            && !unused.connected(),
    ));
    out.push((
        "a built-in speaker outranks headphones, which outrank a line out",
        speaker.rank() > phones.rank() && phones.rank() > PinDefault::of(0x0101_4010).rank() && unused.rank().is_none(),
    ));
    out.push((
        "a connection list word reads four short entries or two long ones",
        connections(0x0306_0502, false) == alloc::vec![(2, false), (5, false), (6, false), (3, false)]
            && connections(0x8010_0002, true) == alloc::vec![(2, false), (0x10, true)],
    ));
    out.push((
        "a widget's type is bits 23 to 20 of its capabilities",
        Kind::of(0x0004_041d) == Kind::Output && Kind::of(0x0040_058d) == Kind::Pin && Kind::of(0x0020_0000) == Kind::Mixer,
    ));
    out
}
