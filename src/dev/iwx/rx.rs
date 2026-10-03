//! What comes back from a running part: frames off the air, and notifications.
//!
//! Until now every packet the receive ring produced was a reply somebody was
//! waiting for, and anything else was acknowledged and dropped -- `cmd::ask`
//! says so, and for a part that only answers questions it is right. A part that
//! scans, or is associated, sends far more than replies: every beacon it hears
//! arrives as an `RX_MPDU`, and the end of a scan is a notification nobody asked
//! for. Dropping those is a scan that hears nothing.
//!
//! So there is one place a packet goes when it is not the reply being waited
//! for, the `Inbox`, and the service loop and the command path both feed it.
//! **AX210 parts put exactly one packet in a receive buffer**, upstream's
//! comment and the reason `alive::Rx` reads one per slot; so a buffer is a
//! packet is either a frame or a notification, never several of each.
//!
//! The frame half is a pure parse over the bytes, so every refusal is asserted
//! with no radio present.

use super::alive::Packet;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// A frame received, in the legacy group.
pub const REPLY_RX_MPDU_CMD: u8 = 0xc1;

/// The descriptor in front of each frame, on AX210-family parts: twenty bytes
/// common to every version, then the 44-byte v3 tail. The 22000 family uses the
/// 28-byte v1 tail, so 48. Getting this wrong by sixteen puts every frame's
/// first sixteen bytes inside the descriptor, and the header read out of the
/// rest is somebody's payload.
pub const DESC_V3: usize = 64;
pub const DESC_V1: usize = 48;

const STATUS_CRC_OK: u32 = 1 << 0;
const STATUS_OVERRUN_OK: u32 = 1 << 1;
const STATUS_SEC_MASK: u32 = 7 << 8;
const STATUS_SEC_CCM: u32 = 2 << 8;
const MFLG2_PAD: u8 = 0x20;
const MFLG2_AMSDU: u8 = 0x40;

/// The smallest frame worth handing up: a full three-address header.
const MIN_FRAME: usize = 24;
/// What a CCMP header adds after the 802.11 header.
const CCMP_HDR: usize = 8;

/// One frame, as `dev::radio::Rx` wants it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Frame {
    pub frame: Vec<u8>,
    pub rssi: i8,
    /// The channel the part says it heard this on.
    pub channel: u8,
    /// The descriptor's status word, kept for whoever installs keys later: it
    /// says whether the part decrypted this.
    pub status: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dropped {
    /// Shorter than the descriptor this family puts in front.
    NoDescriptor,
    /// The part's CRC or overrun check failed. Upstream drops these silently,
    /// and so does this, but counts them.
    Bad,
    /// Shorter than an 802.11 header.
    Runt,
    /// The declared length runs past what the buffer holds.
    Overrun,
}

/// The length of an 802.11 header, for the frames a station hands up.
///
/// Management is 24. Data is 24, plus six for a fourth address when both DS
/// bits are set, plus two for QoS. Control frames are short and odd and a
/// station never needs one, so they are answered as their minimum and the
/// caller drops them anyway.
fn header_len(f: &[u8]) -> usize {
    let fc0 = f[0];
    let fc1 = f[1];
    match (fc0 >> 2) & 3 {
        2 => {
            let mut n = 24;
            if fc1 & 3 == 3 {
                n += 6;
            }
            if fc0 & 0x80 != 0 {
                n += 2;
            }
            n
        }
        1 => 10,
        _ => 24,
    }
}

/// Read one `RX_MPDU` payload into a frame.
///
/// Three corrections upstream makes and every one is silent when skipped:
///
/// - **The pad.** When `MFLG2_PAD` is set the part inserted two bytes after the
///   header -- after the CCMP header, for an encrypted frame -- so the payload
///   lands aligned. Handed up unremoved, every data frame's LLC header is two
///   bytes late and nothing above reads a single packet.
/// - **The A-MSDU bit.** The part de-aggregates and copies the header onto
///   each subframe with the QoS A-MSDU-present bit still set, so a reader that
///   honoured it would try to de-aggregate an already de-aggregated frame.
/// - **Signal.** Two chains, each an energy that is zero for "no reading";
///   zero is not 0 dBm, it is absent, and the stronger chain is the answer.
pub fn mpdu(payload: &[u8], desc: usize) -> Result<Frame, Dropped> {
    if payload.len() < desc {
        return Err(Dropped::NoDescriptor);
    }
    let len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let mflg2 = payload[3];
    let status = u32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]);
    if status & STATUS_CRC_OK == 0 || status & STATUS_OVERRUN_OK == 0 {
        return Err(Dropped::Bad);
    }
    if len < MIN_FRAME {
        return Err(Dropped::Runt);
    }
    if len > payload.len() - desc {
        return Err(Dropped::Overrun);
    }
    let mut f = payload[desc..desc + len].to_vec();
    if mflg2 & MFLG2_PAD != 0 {
        let mut h = header_len(&f);
        if status & STATUS_SEC_MASK == STATUS_SEC_CCM {
            h += CCMP_HDR;
        }
        if h + 2 <= f.len() {
            f.drain(h..h + 2);
        }
    }
    if mflg2 & MFLG2_AMSDU != 0 && (f[0] >> 2) & 3 == 2 && f[0] & 0x80 != 0 {
        let qos = header_len(&f) - 2;
        if qos < f.len() {
            f[qos] &= !0x80;
        }
    }
    // The v3 tail's energies are at 20+20 and 20+21; v1's at 20+12 and 20+13.
    let at = if desc == DESC_V3 { 40 } else { 32 };
    let (ea, eb, ch) = (payload[at], payload[at + 1], payload[at + 2]);
    let dbm = |e: u8| if e == 0 { -256i32 } else { -(e as i32) };
    let rssi = dbm(ea).max(dbm(eb)).max(-128) as i8;
    Ok(Frame { frame: f, rssi, channel: ch, status })
}

/// A notification, owned: group, code and payload, copied out of the ring so
/// the buffer can go back to the part.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Notif {
    pub group: u8,
    pub code: u8,
    pub payload: Vec<u8>,
}

/// Where packets go that nobody was waiting for.
///
/// Bounded on both sides, oldest dropped, and counted: a station that stops
/// reading must not take the heap with it, and a scan heard during a burst of
/// beacons loses the oldest beacons rather than the newest ones -- which is the
/// half that is still true.
pub struct Inbox {
    pub frames: VecDeque<Frame>,
    pub notifs: VecDeque<Notif>,
    pub dropped: u32,
    pub bad: u32,
}

const MAX_FRAMES: usize = 64;
const MAX_NOTIFS: usize = 32;

impl Inbox {
    pub fn new() -> Inbox {
        Inbox { frames: VecDeque::new(), notifs: VecDeque::new(), dropped: 0, bad: 0 }
    }

    /// File one packet.
    pub fn take(&mut self, p: &Packet, desc: usize) {
        if p.group == 0 && p.code == REPLY_RX_MPDU_CMD {
            match mpdu(p.payload, desc) {
                Ok(f) => {
                    if self.frames.len() >= MAX_FRAMES {
                        self.frames.pop_front();
                        self.dropped += 1;
                    }
                    self.frames.push_back(f);
                }
                Err(_) => self.bad += 1,
            }
            return;
        }
        if self.notifs.len() >= MAX_NOTIFS {
            self.notifs.pop_front();
            self.dropped += 1;
        }
        self.notifs.push_back(Notif { group: p.group, code: p.code, payload: p.payload.to_vec() });
    }

    /// The first notification of this kind, removed.
    pub fn notif(&mut self, group: u8, code: u8) -> Option<Notif> {
        let i = self.notifs.iter().position(|n| n.group == group && n.code == code)?;
        self.notifs.remove(i)
    }
}

/// Claims, over synthetic descriptors.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    // A descriptor and a frame behind it.
    fn build(desc: usize, frame: &[u8], status: u32, mflg2: u8, ea: u8, eb: u8, ch: u8) -> Vec<u8> {
        let mut p = alloc::vec![0u8; desc];
        p[0..2].copy_from_slice(&(frame.len() as u16).to_le_bytes());
        p[3] = mflg2;
        p[12..16].copy_from_slice(&status.to_le_bytes());
        let at = if desc == DESC_V3 { 40 } else { 32 };
        p[at] = ea;
        p[at + 1] = eb;
        p[at + 2] = ch;
        p.extend_from_slice(frame);
        p
    }
    let ok = STATUS_CRC_OK | STATUS_OVERRUN_OK;
    // A beacon: management, 24-byte header, then a body.
    let mut beacon = alloc::vec![0x80u8, 0];
    beacon.extend_from_slice(&[0u8; 22]);
    beacon.extend_from_slice(b"body");
    let p = build(DESC_V3, &beacon, ok, 0, 40, 55, 6);
    let f = mpdu(&p, DESC_V3);
    out.push((
        "a frame behind a 64-byte descriptor comes out whole, with its channel",
        f.as_ref().map(|f| (&f.frame[..], f.channel)) == Ok((&beacon[..], 6)),
    ));
    out.push((
        "and the stronger chain is its signal, as a negative dBm",
        f.as_ref().map(|f| f.rssi) == Ok(-40),
    ));
    out.push((
        "an energy of zero is no reading, not 0 dBm",
        mpdu(&build(DESC_V3, &beacon, ok, 0, 0, 70, 1), DESC_V3).map(|f| f.rssi) == Ok(-70),
    ));
    out.push((
        "the 22000 family's shorter descriptor is read at its own offsets",
        mpdu(&build(DESC_V1, &beacon, ok, 0, 33, 0, 11), DESC_V1).map(|f| (f.frame.len(), f.rssi, f.channel))
            == Ok((beacon.len(), -33, 11)),
    ));
    out.push((
        "a frame whose CRC failed is dropped",
        mpdu(&build(DESC_V3, &beacon, STATUS_OVERRUN_OK, 0, 40, 0, 6), DESC_V3) == Err(Dropped::Bad),
    ));
    out.push((
        "and one shorter than a header",
        mpdu(&build(DESC_V3, &beacon[..20], ok, 0, 40, 0, 6), DESC_V3) == Err(Dropped::Runt),
    ));
    {
        let mut p = build(DESC_V3, &beacon, ok, 0, 40, 0, 6);
        p[0..2].copy_from_slice(&((beacon.len() + 1) as u16).to_le_bytes());
        out.push(("and one whose length runs past the buffer", mpdu(&p, DESC_V3) == Err(Dropped::Overrun)));
    }
    out.push(("and a buffer too short for the descriptor", mpdu(&[0u8; 40], DESC_V3) == Err(Dropped::NoDescriptor)));
    // A QoS data frame, padded after its 26-byte header, with the A-MSDU bit set.
    let mut qos = alloc::vec![0x88u8, 0x01];
    qos.extend_from_slice(&[0u8; 22]);
    qos.extend_from_slice(&[0x80, 0x00]); // QoS control, A-MSDU present
    let mut padded = qos.clone();
    padded.extend_from_slice(&[0xEE, 0xEE]); // the pad
    padded.extend_from_slice(b"\xAA\xAA\x03payload");
    let f = mpdu(&build(DESC_V3, &padded, ok, MFLG2_PAD | MFLG2_AMSDU, 40, 0, 1), DESC_V3);
    out.push((
        "the pad after a QoS header is removed, so the LLC header is where it belongs",
        f.as_ref().map(|f| f.frame.len() == padded.len() - 2 && &f.frame[26..29] == b"\xAA\xAA\x03") == Ok(true),
    ));
    out.push((
        "and the A-MSDU bit the part left set is cleared",
        f.as_ref().map(|f| f.frame[24] & 0x80 == 0) == Ok(true),
    ));
    // Encrypted: the pad comes after the CCMP header, eight bytes later.
    let mut enc = qos.clone();
    enc[24] = 0;
    enc.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut enc_padded = enc.clone();
    enc_padded.extend_from_slice(&[0xEE, 0xEE]);
    enc_padded.extend_from_slice(b"cipher");
    let f = mpdu(&build(DESC_V3, &enc_padded, ok | STATUS_SEC_CCM, MFLG2_PAD, 40, 0, 1), DESC_V3);
    out.push((
        "for a CCMP frame the pad is after the CCMP header, not the 802.11 one",
        f.as_ref().map(|f| &f.frame[26..34] == &[1, 2, 3, 4, 5, 6, 7, 8] && &f.frame[34..] == b"cipher") == Ok(true),
    ));
    out.push((
        "a four-address QoS header is thirty-two bytes",
        header_len(&[0x88, 0x03]) == 32 && header_len(&[0x08, 0x01]) == 24 && header_len(&[0x80, 0]) == 24,
    ));
    // The inbox: frames and notifications apart, and bounded.
    let mut ib = Inbox::new();
    let raw = build(DESC_V3, &beacon, ok, 0, 40, 0, 6);
    for _ in 0..MAX_FRAMES + 3 {
        ib.take(&Packet { group: 0, code: REPLY_RX_MPDU_CMD, idx: 0, qid: 0, payload: &raw }, DESC_V3);
    }
    ib.take(&Packet { group: 0x0c, code: 0x0d, idx: 0, qid: 0, payload: &[1, 2] }, DESC_V3);
    out.push((
        "the inbox keeps frames and notifications apart, bounded, counting what it dropped",
        ib.frames.len() == MAX_FRAMES && ib.dropped == 3 && ib.notifs.len() == 1,
    ));
    out.push((
        "and hands a notification back by kind, once",
        ib.notif(0x0c, 0x0d).map(|n| n.payload) == Some(alloc::vec![1, 2]) && ib.notif(0x0c, 0x0d).is_none(),
    ));
    out
}
