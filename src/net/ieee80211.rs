//! 802.11 management frames: the part of wireless that is a standard.
//!
//! Everything here comes from IEEE 802.11 itself -- frame layout, information
//! element numbering, the capability bits -- and not from any vendor's driver.
//! That distinction matters in this directory. The register tables next door
//! had to be copied out of Linux because they describe one company's silicon
//! and exist nowhere else; a beacon has the same shape coming out of every
//! access point ever built, so it can be written from the specification and
//! checked against frames constructed on the spot.
//!
//! Which is what makes this worth having before there is a radio. A scan is
//! two halves: get frames off the air, and understand them. The second half
//! can be finished and proven now, so when the first half arrives there is
//! one new thing to debug rather than two.
//!
//! Byte order is little-endian throughout, which is what 802.11 specifies and
//! also what the machine is, but the conversions are written out rather than
//! assumed -- the one place a reader should not have to know the target.

use alloc::string::String;
use alloc::vec::Vec;

/// Management frames carry a 24-byte header: control, duration, three
/// addresses and a sequence number. Data frames can carry a fourth address;
/// nothing here parses those.
pub const MGMT_HDR: usize = 24;

/// Type 0 is management. The type field lives in bits 2 and 3 of the first
/// byte, and the subtype in the top four.
const TYPE_MGMT: u8 = 0;
const SUBTYPE_PROBE_REQ: u8 = 4;
const SUBTYPE_PROBE_RESP: u8 = 5;
const SUBTYPE_BEACON: u8 = 8;

/// Information element numbers this module knows.
const IE_SSID: u8 = 0;
const IE_RATES: u8 = 1;
const IE_DS_PARAM: u8 = 3;
const IE_RSN: u8 = 48;
const IE_VENDOR: u8 = 221;

/// Privacy, bit 4 of the capability field: the network requires some form of
/// encryption. It does not say which, which is why the RSN element is checked
/// as well -- an ancient WEP network sets exactly this bit and nothing else.
const CAP_PRIVACY: u16 = 1 << 4;

/// Broadcast, for a probe request that asks every access point in earshot.
pub const BROADCAST: [u8; 6] = [0xFF; 6];

fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// One information element: a number, and its bytes.
pub struct Ie<'a> {
    pub id: u8,
    pub data: &'a [u8],
}

/// Walk the elements in a frame body.
///
/// Returns what it could parse and stops at the first malformed length rather
/// than failing the whole frame. Frames come off the air corrupted, and a
/// beacon whose third element is truncated still has a usable SSID in its
/// first -- discarding it would lose networks for no gain.
pub fn elements(mut body: &[u8]) -> Vec<Ie<'_>> {
    let mut out = Vec::new();
    while body.len() >= 2 {
        let id = body[0];
        let len = body[1] as usize;
        if body.len() < 2 + len {
            break;
        }
        out.push(Ie { id, data: &body[2..2 + len] });
        body = &body[2 + len..];
    }
    out
}

/// What a beacon or probe response says about its network.
pub struct Beacon {
    pub ssid: String,
    pub bssid: [u8; 6],
    /// From the DS Parameter Set. Absent on 5 GHz frames that use a different
    /// element, so it is an option rather than a guess.
    pub channel: Option<u8>,
    pub secured: bool,
    /// True when an RSN element is present: WPA2 or later, as opposed to the
    /// privacy bit alone, which WEP also sets.
    pub rsn: bool,
}

/// True if this is a beacon or a probe response, the two frames a scan reads.
///
/// Checked before parsing rather than inside it: a scan sees every frame on
/// the channel, most of them data, and running the element walker over a data
/// frame's payload finds nonsense elements that parse.
pub fn is_beacon_like(frame: &[u8]) -> bool {
    if frame.len() < MGMT_HDR {
        return false;
    }
    let fc = frame[0];
    let ty = (fc >> 2) & 0x3;
    let sub = (fc >> 4) & 0xF;
    ty == TYPE_MGMT && (sub == SUBTYPE_BEACON || sub == SUBTYPE_PROBE_RESP)
}

/// Read a beacon or probe response.
///
/// Both carry the same body -- timestamp, interval, capability, then elements
/// -- which is why one parser serves both and why a scan can use whichever
/// arrives first.
pub fn parse_beacon(frame: &[u8]) -> Option<Beacon> {
    if !is_beacon_like(frame) {
        return None;
    }
    // 8 timestamp, 2 beacon interval, 2 capability.
    const FIXED: usize = 12;
    if frame.len() < MGMT_HDR + FIXED {
        return None;
    }
    let mut bssid = [0u8; 6];
    bssid.copy_from_slice(&frame[16..22]);
    let cap = u16le(&frame[MGMT_HDR + 10..]);

    let mut ssid = String::new();
    let mut channel = None;
    let mut rsn = false;
    for ie in elements(&frame[MGMT_HDR + FIXED..]) {
        match ie.id {
            // A zero-length SSID is a hidden network announcing itself. That
            // is a real answer and stays an empty string; the caller decides
            // whether to show it.
            IE_SSID => ssid = String::from_utf8_lossy(ie.data).into_owned(),
            IE_DS_PARAM if !ie.data.is_empty() => channel = Some(ie.data[0]),
            IE_RSN => rsn = true,
            // WPA1 lived in a vendor element before RSN existed: OUI 00:50:F2
            // with type 1. Still seen on old access points.
            IE_VENDOR if ie.data.len() >= 4 => {
                if ie.data[..4] == [0x00, 0x50, 0xF2, 0x01] {
                    rsn = true;
                }
            }
            _ => {}
        }
    }
    Some(Beacon {
        ssid,
        bssid,
        channel,
        secured: rsn || cap & CAP_PRIVACY != 0,
        rsn,
    })
}

/// Build a probe request.
///
/// An empty `ssid` is a wildcard probe: every access point in range answers,
/// which is how a scan finds networks it was not told to look for. A named
/// one is how a hidden network is found at all, since it does not put its
/// name in its own beacons.
pub fn probe_request(sa: [u8; 6], ssid: &str, rates: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(MGMT_HDR + 2 + ssid.len() + 2 + rates.len());
    // Frame control: subtype in the top nibble, type 0, version 0.
    f.push((SUBTYPE_PROBE_REQ << 4) | (TYPE_MGMT << 2));
    // Flags: none. Not to or from a distribution system.
    f.push(0);
    // Duration. The access point overwrites what matters; zero is what a
    // probe request from a station not yet associated carries.
    f.extend_from_slice(&0u16.to_le_bytes());
    f.extend_from_slice(&BROADCAST); // addr1, destination
    f.extend_from_slice(&sa); // addr2, us
    f.extend_from_slice(&BROADCAST); // addr3, BSSID
    // Sequence control. The hardware fills this in, and writing a number here
    // that the chip then overwrites would be a lie in a packet capture.
    f.extend_from_slice(&0u16.to_le_bytes());

    f.push(IE_SSID);
    f.push(ssid.len() as u8);
    f.extend_from_slice(ssid.as_bytes());

    f.push(IE_RATES);
    f.push(rates.len() as u8);
    f.extend_from_slice(rates);
    f
}

/// The 802.11b/g rates every access point understands, in the encoding the
/// element uses: half-megabit units, with the top bit marking a rate the
/// network requires rather than merely supports.
pub const BASIC_RATES: &[u8] = &[0x82, 0x84, 0x8B, 0x96, 0x0C, 0x12, 0x18, 0x24];

/// Frames built here, parsed here, and compared against what went in.
///
/// This is the whole verification available without a radio, and it is worth
/// more than it looks: it proves the header is the right length, that the
/// element walker agrees with the element writer, and that the capability bit
/// and the RSN element each independently mark a network as secured. When
/// frames do start arriving, a failure is in the radio and not in this.
///
/// Silent, returning only a verdict, because the registry that calls it prints
/// one line per check and a second opinion underneath would be noise.
pub fn selftest() -> bool {
    let sa = [0x02, 0x47, 0x4C, 0x41, 0x44, 0x53];
    let req = probe_request(sa, "glados", BASIC_RATES);
    if req.len() != MGMT_HDR + 2 + 6 + 2 + BASIC_RATES.len() {
        return false;
    }
    if (req[0] >> 2) & 0x3 != TYPE_MGMT || (req[0] >> 4) & 0xF != SUBTYPE_PROBE_REQ {
        return false;
    }
    if req[10..16] != sa {
        return false;
    }
    // A probe request must not parse as a beacon: a scan reads every frame on
    // the channel, and one that mistakes its own transmissions for networks
    // finds itself.
    if is_beacon_like(&req) {
        return false;
    }
    let ies = elements(&req[MGMT_HDR..]);
    if ies.len() != 2
        || ies[0].id != IE_SSID
        || ies[0].data != b"glados"
        || ies[1].id != IE_RATES
        || ies[1].data != BASIC_RATES
    {
        return false;
    }

    // A beacon assembled by hand, since nothing here builds one.
    let bssid = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    let mut b: Vec<u8> = Vec::new();
    b.push((SUBTYPE_BEACON << 4) | (TYPE_MGMT << 2));
    b.push(0);
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&BROADCAST);
    b.extend_from_slice(&bssid);
    b.extend_from_slice(&bssid);
    b.extend_from_slice(&0u16.to_le_bytes());
    b.extend_from_slice(&[0u8; 8]);
    b.extend_from_slice(&100u16.to_le_bytes());
    b.extend_from_slice(&CAP_PRIVACY.to_le_bytes());
    b.extend_from_slice(&[IE_SSID, 7]);
    b.extend_from_slice(b"testnet");
    b.extend_from_slice(&[IE_DS_PARAM, 1, 6]);

    if !is_beacon_like(&b) {
        return false;
    }
    match parse_beacon(&b) {
        None => return false,
        Some(p) => {
            // The privacy bit on its own says encrypted, not WPA2.
            if p.ssid != "testnet" || p.bssid != bssid || p.channel != Some(6) {
                return false;
            }
            if !p.secured || p.rsn {
                return false;
            }
        }
    }

    // The same beacon with an RSN element and the privacy bit cleared: still
    // secured, and now known to be WPA2 rather than merely encrypted somehow.
    let mut r = b.clone();
    r[MGMT_HDR + 10..MGMT_HDR + 12].copy_from_slice(&0u16.to_le_bytes());
    r.extend_from_slice(&[IE_RSN, 2, 0x01, 0x00]);
    match parse_beacon(&r) {
        None => return false,
        Some(p) => {
            if !p.secured || !p.rsn {
                return false;
            }
        }
    }

    // An open network: neither signal present.
    let mut o = b.clone();
    o[MGMT_HDR + 10..MGMT_HDR + 12].copy_from_slice(&0u16.to_le_bytes());
    match parse_beacon(&o) {
        None => return false,
        Some(p) => {
            if p.secured {
                return false;
            }
        }
    }

    // Truncation: a beacon cut off mid-element keeps the elements before it,
    // because frames come off the air damaged and a usable SSID is worth more
    // than a clean rejection.
    match parse_beacon(&b[..b.len() - 2]) {
        None => return false,
        Some(p) => {
            if p.ssid != "testnet" {
                return false;
            }
        }
    }
    // One cut into its fixed fields has no SSID to salvage and is rejected.
    if parse_beacon(&b[..MGMT_HDR + 4]).is_some() {
        return false;
    }
    parse_beacon(&[]).is_none()
}
