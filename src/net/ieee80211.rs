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
const TYPE_DATA: u8 = 2;
const SUBTYPE_DATA: u8 = 0;
const SUBTYPE_ASSOC_REQ: u8 = 0;
const SUBTYPE_ASSOC_RESP: u8 = 1;
const SUBTYPE_PROBE_REQ: u8 = 4;
const SUBTYPE_DISASSOC: u8 = 10;
const SUBTYPE_AUTH: u8 = 11;
const SUBTYPE_DEAUTH: u8 = 12;
const SUBTYPE_PROBE_RESP: u8 = 5;
const SUBTYPE_BEACON: u8 = 8;

/// LLC/SNAP: `AA AA 03` then a three-byte OUI of zero, then the EtherType.
///
/// **This is how an Ethernet payload rides on 802.11**, and it is why a
/// wireless frame is eight bytes longer than the Ethernet one it carries
/// rather than the fourteen an Ethernet header costs -- the addresses moved
/// into the 802.11 header and only the type came along.
pub const SNAP: [u8; 6] = [0xAA, 0xAA, 0x03, 0x00, 0x00, 0x00];

pub fn snap_wrap(ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + payload.len());
    b.extend_from_slice(&SNAP);
    b.extend_from_slice(&ethertype.to_be_bytes());
    b.extend_from_slice(payload);
    b
}

/// Take an EtherType and a payload back out. Nothing if it is not SNAP.
///
/// Refused rather than skipped: an 802.11 body that is not SNAP is some other
/// encapsulation, and reading its ninth byte onwards as an IP packet would
/// hand the stack something that parses and is not what was sent.
pub fn snap_unwrap(body: &[u8]) -> Option<(u16, &[u8])> {
    if body.len() < 8 || body[..6] != SNAP {
        return None;
    }
    Some((u16::from_be_bytes([body[6], body[7]]), &body[8..]))
}

/// A data frame from a station to its access point.
///
/// ToDS, so the addresses are BSSID, source, destination in that order -- which
/// is not the order they appear in on a frame coming the other way. That
/// reshuffling is the whole of `data_addrs` below, and it is the thing that
/// makes 802.11 addressing worth a function rather than three slices.
pub fn data_to_ds(bssid: &[u8; 6], sa: &[u8; 6], da: &[u8; 6], seq: u16, body: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(24 + body.len());
    f.push((SUBTYPE_DATA << 4) | (TYPE_DATA << 2));
    f.push(0x01); // ToDS
    f.extend_from_slice(&[0, 0]); // duration, filled by the radio
    f.extend_from_slice(bssid);
    f.extend_from_slice(sa);
    f.extend_from_slice(da);
    // Sequence number in the top twelve bits, fragment number in the low four.
    f.extend_from_slice(&(((seq & 0x0FFF) << 4) as u16).to_le_bytes());
    f.extend_from_slice(body);
    f
}

/// A data frame from an access point to one of its stations.
///
/// FromDS, so the addresses are destination, BSSID, source -- a different
/// order from `data_to_ds` for the same three parties, which is the whole
/// reason `data_addrs` exists and the reason both directions are built here
/// rather than one being assumed to be the other reversed.
pub fn data_from_ds(da: &[u8; 6], bssid: &[u8; 6], sa: &[u8; 6], seq: u16, body: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(24 + body.len());
    f.push((SUBTYPE_DATA << 4) | (TYPE_DATA << 2));
    f.push(0x02); // FromDS
    f.extend_from_slice(&[0, 0]); // duration
    f.extend_from_slice(da);
    f.extend_from_slice(bssid);
    f.extend_from_slice(sa);
    f.extend_from_slice(&((seq & 0x0FFF) << 4).to_le_bytes());
    f.extend_from_slice(body);
    f
}

/// Who a data frame is really from and to, whichever way it was going.
///
/// **Four layouts, and the addresses mean different things in each.** A station
/// sending to its access point puts the BSSID first; the access point sending
/// back puts the destination first. Reading A1 as the destination is correct
/// exactly half the time, which is the worst possible rate for a bug: it works
/// on everything you send and fails on everything you receive.
pub fn data_addrs(frame: &[u8]) -> Option<([u8; 6], [u8; 6])> {
    if frame.len() < 24 {
        return None;
    }
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    if (fc >> 2) & 0x3 != TYPE_DATA as u16 {
        return None;
    }
    let at = |off: usize| -> [u8; 6] {
        let mut m = [0u8; 6];
        m.copy_from_slice(&frame[off..off + 6]);
        m
    };
    let (a1, a2, a3) = (at(4), at(10), at(16));
    let to_ds = fc & 0x0100 != 0;
    let from_ds = fc & 0x0200 != 0;
    Some(match (to_ds, from_ds) {
        // Ad-hoc: destination, source, then the BSSID nobody routes by.
        (false, false) => (a1, a2),
        // To the access point: BSSID, source, destination.
        (true, false) => (a3, a2),
        // From the access point: destination, BSSID, source.
        (false, true) => (a1, a3),
        // Between two access points, where the real source is a fourth
        // address that sits *after* the sequence control rather than beside
        // the others -- which is why this cannot be an index into a loop.
        (true, true) => {
            if frame.len() < 30 {
                return None;
            }
            let mut a4 = [0u8; 6];
            a4.copy_from_slice(&frame[24..30]);
            (a3, a4)
        }
    })
}

/// Information element numbers this module knows.
const IE_SSID: u8 = 0;
const IE_RATES: u8 = 1;
const IE_DS_PARAM: u8 = 3;
const IE_TIM: u8 = 5;
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
    /// In time units of 1024 us; 100 is the ordinary answer.
    pub beacon_int: u16,
    /// From the TIM element, which only a beacon carries: a probe response
    /// answers `None`, and guessing 1 would tell a part's power table to wake
    /// for every beacon on a network that asked for every third.
    pub dtim: Option<u8>,
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
    let beacon_int = u16le(&frame[MGMT_HDR + 8..]);
    let cap = u16le(&frame[MGMT_HDR + 10..]);

    let mut ssid = String::new();
    let mut channel = None;
    let mut dtim = None;
    let mut rsn = false;
    for ie in elements(&frame[MGMT_HDR + FIXED..]) {
        match ie.id {
            // A zero-length SSID is a hidden network announcing itself. That
            // is a real answer and stays an empty string; the caller decides
            // whether to show it.
            IE_SSID => ssid = String::from_utf8_lossy(ie.data).into_owned(),
            IE_DS_PARAM if !ie.data.is_empty() => channel = Some(ie.data[0]),
            // Count, then period. A period of zero is reserved and refused.
            IE_TIM if ie.data.len() >= 2 && ie.data[1] != 0 => dtim = Some(ie.data[1]),
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
        beacon_int,
        dtim,
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

// ---------------------------------------------------------------------------
// The frames that get a station onto a network
// ---------------------------------------------------------------------------
//
// Authenticate, associate, and the two ways either end says it is leaving.
// All four are management frames with the same header and a short fixed body,
// and all four are *chip independent* -- a FullMAC part's firmware sends these
// itself, and every SoftMAC part in the world needs exactly this.

/// A status code of zero is success, and nothing else is.
pub const STATUS_SUCCESS: u16 = 0;
/// Open System, the only authentication algorithm worth implementing: WPA2
/// does its real authentication in the four-way handshake afterwards, and
/// Shared Key is WEP's, which is broken by design and whose challenge-response
/// hands an eavesdropper a known plaintext.
pub const AUTH_OPEN: u16 = 0;

pub const REASON_LEAVING: u16 = 3;

/// The RSN element for WPA2-Personal with CCMP, complete with its id and
/// length so it can be appended whole.
///
/// **An association request carries the ciphers the station is choosing**, and
/// an access point that offers CCMP refuses a station that asks for anything
/// else -- so this is not a description of what is supported, it is the
/// choice. Group CCMP, pairwise CCMP, AKM PSK, no RSN capabilities. TKIP is
/// deliberately absent: offering it lets an access point pick it, and a
/// downgrade a station volunteered for is still a downgrade.
pub const RSN_CCMP_PSK: [u8; 22] = [
    IE_RSN, 20, // element id and length
    0x01, 0x00, // version 1
    0x00, 0x0F, 0xAC, 0x04, // group cipher: CCMP
    0x01, 0x00, 0x00, 0x0F, 0xAC, 0x04, // one pairwise cipher: CCMP
    0x01, 0x00, 0x00, 0x0F, 0xAC, 0x02, // one AKM: PSK
    0x00, 0x00, // RSN capabilities
];

/// Capability bits a station claims in an association request. ESS, because
/// this is an infrastructure network and not ad-hoc; Privacy when the network
/// is encrypted, which has to agree with the RSN element or the request is
/// internally inconsistent and refused.
const CAP_ESS: u16 = 1 << 0;

/// The header every management frame shares.
///
/// Neither ToDS nor FromDS is set -- management frames are between a station
/// and an access point directly, so A1 is always the destination, A2 the
/// sender and A3 the BSSID, with none of the reshuffling `data_addrs` exists
/// for. `seq` is ours to fill here and not the chip's, which is the whole
/// difference between SoftMAC and FullMAC written as one argument.
fn mgmt_header(subtype: u8, da: &[u8; 6], sa: &[u8; 6], bssid: &[u8; 6], seq: u16) -> Vec<u8> {
    let mut f = Vec::with_capacity(MGMT_HDR);
    f.push((subtype << 4) | (TYPE_MGMT << 2));
    f.push(0);
    f.extend_from_slice(&0u16.to_le_bytes()); // duration
    f.extend_from_slice(da);
    f.extend_from_slice(sa);
    f.extend_from_slice(bssid);
    f.extend_from_slice(&((seq & 0x0FFF) << 4).to_le_bytes());
    f
}

/// The management subtype of a frame, or nothing if it is not management.
pub fn mgmt_subtype(frame: &[u8]) -> Option<u8> {
    if frame.len() < MGMT_HDR {
        return None;
    }
    let fc = frame[0];
    if (fc >> 2) & 0x3 != TYPE_MGMT {
        return None;
    }
    Some((fc >> 4) & 0xF)
}

/// Destination, sender and BSSID of a management frame.
pub fn mgmt_addrs(frame: &[u8]) -> Option<([u8; 6], [u8; 6], [u8; 6])> {
    mgmt_subtype(frame)?;
    let at = |off: usize| -> [u8; 6] {
        let mut m = [0u8; 6];
        m.copy_from_slice(&frame[off..off + 6]);
        m
    };
    Some((at(4), at(10), at(16)))
}

/// True when this frame is addressed to us or to everybody.
///
/// Worth a function because the broadcast case is the one that gets forgotten:
/// an access point tearing down a whole BSS sends one deauthentication to
/// `ff:ff:ff:ff:ff:ff`, and a station that only matched its own address stays
/// associated to a network that has stopped talking to it.
pub fn addressed_to(frame: &[u8], me: &[u8; 6]) -> bool {
    match mgmt_addrs(frame) {
        Some((da, _, _)) => &da == me || da == BROADCAST,
        None => false,
    }
}

/// Build an Open System authentication frame.
///
/// `seq_no` is the *transaction* sequence number and is not the frame's
/// sequence control -- 1 from the station, 2 from the access point. Two
/// counters with the same name in one frame, and swapping them produces an
/// authentication nobody answers.
pub fn auth_open(bssid: &[u8; 6], sa: &[u8; 6], seq: u16, seq_no: u16) -> Vec<u8> {
    let mut f = mgmt_header(SUBTYPE_AUTH, bssid, sa, bssid, seq);
    f.extend_from_slice(&AUTH_OPEN.to_le_bytes());
    f.extend_from_slice(&seq_no.to_le_bytes());
    f.extend_from_slice(&STATUS_SUCCESS.to_le_bytes());
    f
}

/// What an authentication frame says.
pub struct Auth {
    pub alg: u16,
    pub seq_no: u16,
    pub status: u16,
}

pub fn parse_auth(frame: &[u8]) -> Option<Auth> {
    if mgmt_subtype(frame)? != SUBTYPE_AUTH || frame.len() < MGMT_HDR + 6 {
        return None;
    }
    let b = &frame[MGMT_HDR..];
    Some(Auth { alg: u16le(b), seq_no: u16le(&b[2..]), status: u16le(&b[4..]) })
}

/// Build an association request.
///
/// The SSID goes in even though the frame is addressed to one BSSID, because
/// an access point carrying several networks on one radio tells them apart by
/// exactly this element.
pub fn assoc_request(
    bssid: &[u8; 6],
    sa: &[u8; 6],
    seq: u16,
    ssid: &str,
    rates: &[u8],
    rsn: bool,
) -> Vec<u8> {
    let mut f = mgmt_header(SUBTYPE_ASSOC_REQ, bssid, sa, bssid, seq);
    let cap = if rsn { CAP_ESS | CAP_PRIVACY } else { CAP_ESS };
    f.extend_from_slice(&cap.to_le_bytes());
    // Listen interval, in beacon periods: how long the station may sleep
    // before the access point may discard buffered frames for it. Ten is
    // ordinary, and nothing here sleeps, so it costs the access point a little
    // buffering it will never have to do.
    f.extend_from_slice(&10u16.to_le_bytes());

    f.push(IE_SSID);
    f.push(ssid.len() as u8);
    f.extend_from_slice(ssid.as_bytes());
    f.push(IE_RATES);
    f.push(rates.len() as u8);
    f.extend_from_slice(rates);
    if rsn {
        f.extend_from_slice(&RSN_CCMP_PSK);
    }
    f
}

/// What an association response says.
pub struct AssocResp {
    pub status: u16,
    /// Association identifier. The top two bits are set by convention and are
    /// not part of the number, so masking is required rather than tidy: an
    /// unmasked AID of 1 reads as 49153.
    pub aid: u16,
}

pub fn parse_assoc_resp(frame: &[u8]) -> Option<AssocResp> {
    if mgmt_subtype(frame)? != SUBTYPE_ASSOC_RESP || frame.len() < MGMT_HDR + 6 {
        return None;
    }
    let b = &frame[MGMT_HDR..];
    Some(AssocResp { status: u16le(&b[2..]), aid: u16le(&b[4..]) & 0x3FFF })
}

/// Build a deauthentication, which ends everything, or a disassociation,
/// which ends only the association and leaves the station authenticated.
pub fn deauth(bssid: &[u8; 6], sa: &[u8; 6], seq: u16, reason: u16) -> Vec<u8> {
    let mut f = mgmt_header(SUBTYPE_DEAUTH, bssid, sa, bssid, seq);
    f.extend_from_slice(&reason.to_le_bytes());
    f
}

pub fn disassoc(bssid: &[u8; 6], sa: &[u8; 6], seq: u16, reason: u16) -> Vec<u8> {
    let mut f = mgmt_header(SUBTYPE_DISASSOC, bssid, sa, bssid, seq);
    f.extend_from_slice(&reason.to_le_bytes());
    f
}

/// The reason in a deauthentication or disassociation, and nothing else.
pub fn parse_reason(frame: &[u8]) -> Option<(u8, u16)> {
    let sub = mgmt_subtype(frame)?;
    if sub != SUBTYPE_DEAUTH && sub != SUBTYPE_DISASSOC {
        return None;
    }
    if frame.len() < MGMT_HDR + 2 {
        return None;
    }
    Some((sub, u16le(&frame[MGMT_HDR..])))
}

/// Build an association response. Only an access point sends one, and the one
/// that exists here is the fake in `mlme`'s selftest -- which is exactly why
/// it belongs beside the parser rather than inside the test: a builder and a
/// parser written in one place agree with each other and prove nothing, and
/// these two are read by code that never sees the other.
pub fn assoc_response(sta: &[u8; 6], bssid: &[u8; 6], seq: u16, status: u16, aid: u16) -> Vec<u8> {
    let mut f = mgmt_header(SUBTYPE_ASSOC_RESP, sta, bssid, bssid, seq);
    f.extend_from_slice(&(CAP_ESS | CAP_PRIVACY).to_le_bytes());
    f.extend_from_slice(&status.to_le_bytes());
    f.extend_from_slice(&(0xC000 | (aid & 0x3FFF)).to_le_bytes());
    f
}

/// Build an authentication response: the access point's side, transaction
/// sequence number 2. A separate function from `auth_open` rather than a
/// direction flag, because the addresses go in a different order and a flag
/// is how that becomes a runtime decision nobody reads.
pub fn auth_response(sta: &[u8; 6], bssid: &[u8; 6], seq: u16, status: u16) -> Vec<u8> {
    let mut f = mgmt_header(SUBTYPE_AUTH, sta, bssid, bssid, seq);
    f.extend_from_slice(&AUTH_OPEN.to_le_bytes());
    f.extend_from_slice(&2u16.to_le_bytes());
    f.extend_from_slice(&status.to_le_bytes());
    f
}

/// Build a beacon, for the same reason `assoc_response` exists.
pub fn beacon(bssid: &[u8; 6], seq: u16, ssid: &str, channel: u8, rsn: bool) -> Vec<u8> {
    beacon_like(SUBTYPE_BEACON, &BROADCAST, bssid, seq, ssid, channel, rsn)
}

/// A probe response: the same body as a beacon, addressed to the station that
/// asked. `parse_beacon` reads either, which is why one scan can use whichever
/// arrives first -- and why the two builders differ only in the two fields
/// that genuinely differ.
pub fn probe_response(
    sta: &[u8; 6],
    bssid: &[u8; 6],
    seq: u16,
    ssid: &str,
    channel: u8,
    rsn: bool,
) -> Vec<u8> {
    beacon_like(SUBTYPE_PROBE_RESP, sta, bssid, seq, ssid, channel, rsn)
}

fn beacon_like(
    sub: u8,
    da: &[u8; 6],
    bssid: &[u8; 6],
    seq: u16,
    ssid: &str,
    channel: u8,
    rsn: bool,
) -> Vec<u8> {
    let mut f = mgmt_header(sub, da, bssid, bssid, seq);
    f.extend_from_slice(&0u64.to_le_bytes()); // timestamp
    f.extend_from_slice(&100u16.to_le_bytes()); // beacon interval
    let cap = if rsn { CAP_ESS | CAP_PRIVACY } else { CAP_ESS };
    f.extend_from_slice(&cap.to_le_bytes());

    f.push(IE_SSID);
    f.push(ssid.len() as u8);
    f.extend_from_slice(ssid.as_bytes());
    f.push(IE_RATES);
    f.push(BASIC_RATES.len() as u8);
    f.extend_from_slice(BASIC_RATES);
    f.push(IE_DS_PARAM);
    f.push(1);
    f.push(channel);
    if rsn {
        f.extend_from_slice(&RSN_CCMP_PSK);
    }
    f
}

/// Every frame this module builds, at fixed inputs, as hex.
///
/// **The one thing the boot selftests structurally cannot check.** Every
/// 802.11 claim in this tree builds a frame with these functions and reads it
/// back with the parsers beside them, so a field written in the wrong order is
/// read back in the wrong order and agrees with itself perfectly. The suites
/// pass and no access point in the world will answer.
///
/// So the bytes go out to where something that is not this can read them:
/// `tools/dot11check.py` puts every line through **scapy**, which is somebody
/// else's implementation of these formats and reads real captures. That is the
/// same bargain `tokenizer.py --verify` and `manifest.py --verify` make, and
/// the reason it is a shell verb rather than a suite is that the second
/// opinion cannot live in the kernel -- a checker compiled in here would be a
/// third thing written from the same understanding.
///
/// Inputs are constants so the host can rebuild the identical frame and
/// compare byte for byte rather than field by field.
pub fn dump() {
    use crate::kprintln;
    let me: [u8; 6] = [0x02, 0, 0, 0, 0, 0x11];
    let ap: [u8; 6] = [0x02, 0, 0, 0, 0, 0xAA];
    const SEQ: u16 = 7;

    let line = |name: &str, f: &[u8]| {
        let mut hex = alloc::string::String::with_capacity(f.len() * 2);
        for b in f {
            hex.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
            hex.push(char::from_digit((b & 0xF) as u32, 16).unwrap_or('0'));
        }
        kprintln!("frame {} {}", name, hex);
    };

    line("auth_req", &auth_open(&ap, &me, SEQ, 1));
    line("auth_resp", &auth_response(&me, &ap, SEQ, 0));
    line("assoc_req_rsn", &assoc_request(&ap, &me, SEQ, "glados", BASIC_RATES, true));
    line("assoc_req_open", &assoc_request(&ap, &me, SEQ, "glados", BASIC_RATES, false));
    line("assoc_resp", &assoc_response(&me, &ap, SEQ, 0, 7));
    line("deauth", &deauth(&me, &ap, SEQ, REASON_LEAVING));
    line("disassoc", &disassoc(&me, &ap, SEQ, 8));
    line("beacon", &beacon(&ap, SEQ, "glados", 6, true));
    line("probe_resp", &probe_response(&me, &ap, SEQ, "glados", 6, true));
    line("probe_req", &probe_request(me, "glados", BASIC_RATES));
    line("rsn_element", &RSN_CCMP_PSK);
    line("snap", &snap_wrap(0x0800, b"payload"));
    line("data_to_ds", &data_to_ds(&ap, &me, &ap, SEQ, &snap_wrap(0x0800, b"payload")));
    line("data_from_ds", &data_from_ds(&me, &ap, &ap, SEQ, &snap_wrap(0x0800, b"payload")));
}

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
