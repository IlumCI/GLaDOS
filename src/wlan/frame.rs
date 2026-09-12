//! The 802.11 frame header, and the four questions everything asks of it.
//!
//! Ported from OpenBSD `sys/net80211/ieee80211.h`.
//! upstream-sha256: 4b10494ab0322392
//! Copyright (c) 2001 Atsushi Onoe
//! Copyright (c) 2002, 2003 Sam Leffler, Errno Consulting
//! Licence: BSD-2. See licenses/net80211-BSD-2-Clause.txt.
//!
//! Upstream is a 1,204-line header of every constant in the standard. This is
//! the subset the CCMP path needs, and it grows as the port does rather than
//! arriving whole -- a constant nothing references is a constant nobody has
//! checked against the standard.
//!
//! ### Why the header is a view rather than a struct
//!
//! An 802.11 header is not a fixed layout. It is 24 bytes, or 30 with a fourth
//! address, or 26 with a QoS control field, or 32 with both, and which one it
//! is depends on two bits and a subtype in the first two bytes. Upstream
//! expresses that with `struct ieee80211_frame`, `ieee80211_frame_addr4`,
//! `ieee80211_qosframe` and `ieee80211_qosframe_addr4` and casts between
//! them. A borrowed slice with accessors says the same thing and cannot read
//! past the end of a frame that is shorter than the struct somebody chose.

pub const FC0_TYPE_MASK: u8 = 0x0c;
pub const FC0_TYPE_MGT: u8 = 0x00;
pub const FC0_TYPE_DATA: u8 = 0x08;
pub const FC0_SUBTYPE_MASK: u8 = 0xf0;
pub const FC0_SUBTYPE_QOS: u8 = 0x80;

pub const FC1_DIR_MASK: u8 = 0x03;
pub const FC1_DIR_DSTODS: u8 = 0x03;
pub const FC1_RETRY: u8 = 0x08;
pub const FC1_PWR_MGT: u8 = 0x10;
pub const FC1_MORE_DATA: u8 = 0x20;
pub const FC1_PROTECTED: u8 = 0x40;
pub const FC1_ORDER: u8 = 0x80;

pub const QOS_TID: u8 = 0x0f;

/// The shortest header there is: two frame-control bytes, duration, and three
/// addresses.
pub const MIN_HDR: usize = 24;
/// Six more for a fourth address, two more for QoS control.
pub const ADDR4_LEN: usize = 6;
pub const QOS_LEN: usize = 2;

/// A borrowed 802.11 header.
///
/// Construction is the only place a length is checked, so every accessor below
/// can index without a bounds test and without being able to be wrong.
#[derive(Clone, Copy)]
pub struct Header<'a> {
    raw: &'a [u8],
}

impl<'a> Header<'a> {
    /// `None` when the slice is too short to be the header it claims to be.
    ///
    /// The length required depends on the flags *inside* the header, so this
    /// reads the first two bytes, works out what shape it is, and then checks
    /// -- rather than accepting 24 bytes and reading 32.
    pub fn new(raw: &'a [u8]) -> Option<Header<'a>> {
        if raw.len() < MIN_HDR {
            return None;
        }
        let h = Header { raw };
        if raw.len() < h.len() {
            return None;
        }
        Some(h)
    }

    pub fn fc0(&self) -> u8 {
        self.raw[0]
    }

    pub fn fc1(&self) -> u8 {
        self.raw[1]
    }

    pub fn addr1(&self) -> &'a [u8] {
        &self.raw[4..10]
    }

    pub fn addr2(&self) -> &'a [u8] {
        &self.raw[10..16]
    }

    pub fn addr3(&self) -> &'a [u8] {
        &self.raw[16..22]
    }

    /// The sequence-control field's low byte, which carries the fragment
    /// number in its bottom four bits.
    pub fn seq0(&self) -> u8 {
        self.raw[22]
    }

    /// Whether a fourth address is present: only when the frame is going
    /// from one distribution system to another.
    pub fn has_addr4(&self) -> bool {
        self.fc1() & FC1_DIR_MASK == FC1_DIR_DSTODS
    }

    /// Whether a QoS control field is present.
    ///
    /// Both halves of the test matter. The QoS bit is bit 7 of the subtype
    /// field, which only means QoS in a *data* frame -- in a management frame
    /// the same bit means something else entirely, and a test that checked
    /// only the bit would read two bytes of a beacon as a QoS field.
    pub fn has_qos(&self) -> bool {
        self.fc0() & (FC0_TYPE_MASK | FC0_SUBTYPE_QOS) == (FC0_TYPE_DATA | FC0_SUBTYPE_QOS)
    }

    pub fn addr4(&self) -> Option<&'a [u8]> {
        if self.has_addr4() {
            Some(&self.raw[24..30])
        } else {
            None
        }
    }

    /// The traffic identifier, zero when there is no QoS field.
    pub fn tid(&self) -> u8 {
        if !self.has_qos() {
            return 0;
        }
        let at = if self.has_addr4() { MIN_HDR + ADDR4_LEN } else { MIN_HDR };
        self.raw[at] & QOS_TID
    }

    /// How long this header actually is.
    pub fn len(&self) -> usize {
        MIN_HDR
            + if self.has_addr4() { ADDR4_LEN } else { 0 }
            + if self.has_qos() { QOS_LEN } else { 0 }
    }

    pub fn is_data(&self) -> bool {
        self.fc0() & FC0_TYPE_MASK == FC0_TYPE_DATA
    }

    pub fn is_mgmt(&self) -> bool {
        self.fc0() & FC0_TYPE_MASK == FC0_TYPE_MGT
    }

    pub fn protected(&self) -> bool {
        self.fc1() & FC1_PROTECTED != 0
    }

    pub fn bytes(&self) -> &'a [u8] {
        &self.raw[..self.len()]
    }
}
