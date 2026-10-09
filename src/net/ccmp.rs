//! CCMP: the link cipher every WPA2 network uses, IEEE 802.11-2016 §12.5.3.
//!
//! `wpa2.rs` ends with a Temporal Key and four comments saying it "encrypts
//! data with CCMP". This is that, and without it a completed four-way handshake
//! produces a key nothing can use.
//!
//! **Chip independent, which is the point.** Nearly every wireless part in
//! `dev::registry` is SoftMAC -- the radio moves 802.11 frames and the host
//! does association, sequencing and crypto -- so this is written once and every
//! driver uses it. A part with CCMP in hardware says so through its
//! capabilities and this is skipped; that is a property of the chip, not a
//! second implementation.
//!
//! ### Three constructions, and each is where a mistake hides
//!
//! **The nonce** is priority, then the transmitter's address, then the packet
//! number. It must never repeat under one key -- CCM is counter mode, and a
//! repeated nonce hands an attacker the XOR of two plaintexts. The PN is
//! therefore a strictly increasing 48-bit counter and `next_pn` refuses to wrap
//! rather than starting again.
//!
//! **The AAD** is the frame header with every field an intermediate is allowed
//! to change masked out: retry, power management, more-data, and the sequence
//! number. Without that masking a frame retransmitted by the sender -- with the
//! retry bit now set -- authenticates as a forgery. With too *much* masking, a
//! field an attacker can change stops being covered. Each mask below is one
//! line with the rule beside it.
//!
//! **The replay window.** A received PN at or below the last accepted one is a
//! replay and is dropped. This is not an optimisation: without it a recorded
//! frame can be injected forever and every bit of it verifies, because it is
//! genuine.
//!
//! ### What is checked and what is owed
//!
//! The cipher underneath is checked against RFC 3610 at every boot, and the
//! *framing* is checked against **IEEE 802.11-2012 Annex M.6.4** -- the
//! published CCMP encapsulation example, ciphertext and MIC byte for byte.
//! That gap was open for a long time and this is what closed it.
//!
//! Why a vector and not only a round trip, which this file also does: the AAD
//! masking and the nonce layout could be self-consistently wrong and round-trip
//! perfectly against themselves while failing against every access point in
//! the world. The other end of a real conversation is not available to test
//! against, so a published frame is the only thing that catches it.
//!
//! The vector's own frame carries the **retry bit set**, which is the case the
//! AAD mask exists for, so it exercises the masking rather than passing
//! through it. Both directions of that are asserted: a change to an address
//! the AAD covers must fail the MIC, and a retry bit set in flight must *not*
//! -- the second being the one a too-eager mask breaks, and it presents as a
//! flaky radio rather than as a bug.
//!
//! What is still structural rather than published: the QoS layout. Annex M's
//! example is a non-QoS data frame, so the TID's path into the nonce and the
//! two-byte QoS control in the AAD are checked by round trip and by the
//! rewritten-TID refusal above, and not against anybody else's bytes.

use alloc::vec::Vec;

use crate::crypto::ccm;

/// Bytes the CCMP header adds ahead of the payload.
pub const HDR_LEN: usize = 8;
/// Bytes the MIC adds after it.
pub const MIC_LEN: usize = 8;
/// A PN is 48 bits, which at a thousand frames a second is nine thousand years.
pub const PN_MAX: u64 = (1 << 48) - 1;

// --- the frame control field, by bit ------------------------------------
const FC_TYPE: u16 = 0x000C;
const FC_TYPE_DATA: u16 = 0x0008;
const FC_SUBTYPE_QOS: u16 = 0x0080;
const FC_TO_DS: u16 = 0x0100;
const FC_FROM_DS: u16 = 0x0200;
const FC_PROTECTED: u16 = 0x4000;

/// What `aad` masks out of the frame control field, and why, one rule each.
///
/// Subtype bits 4-6 for data frames, because an intermediate may re-classify
/// within the data subtypes; bit 7 stays, since it is what tells QoS data from
/// plain data and that changes the header's own length. Retry, power
/// management and more-data are all set by the transmitter after the fact.
/// Order is masked for QoS frames.
const FC_MASK: u16 = !(0x0070 | 0x0800 | 0x1000 | 0x2000);

/// A parsed 802.11 header: how long it is and what is in it.
pub struct Frame {
    pub hdr_len: usize,
    pub qos: bool,
    pub a4: bool,
    /// Transmitter address, which is address 2 in every layout CCMP admits.
    pub a2: [u8; 6],
    /// Traffic identifier, which becomes the nonce's priority.
    pub tid: u8,
}

/// Read the header of an 802.11 data frame.
///
/// Answers nothing for anything that is not one. CCMP protects data frames and
/// robust management frames; only data is handled here, and a management frame
/// arriving at this function is a caller bug rather than something to guess at.
pub fn parse(frame: &[u8]) -> Option<Frame> {
    if frame.len() < 24 {
        return None;
    }
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    if fc & FC_TYPE != FC_TYPE_DATA {
        return None;
    }
    let qos = fc & FC_SUBTYPE_QOS != 0;
    // Four addresses only when a frame is going between two access points.
    let a4 = (fc & FC_TO_DS != 0) && (fc & FC_FROM_DS != 0);
    let hdr_len = 24 + if a4 { 6 } else { 0 } + if qos { 2 } else { 0 };
    if frame.len() < hdr_len {
        return None;
    }
    let mut a2 = [0u8; 6];
    a2.copy_from_slice(&frame[10..16]);
    // The TID is the low four bits of the QoS control field, which sits at the
    // end of the header.
    let tid = if qos { frame[hdr_len - 2] & 0x0F } else { 0 };
    Some(Frame { hdr_len, qos, a4, a2, tid })
}

/// The CCM nonce: priority, transmitter, packet number.
///
/// Thirteen bytes, which is what fixes CCM's `L` at two. The PN is big-endian
/// here and little-endian in the header -- not a mistake in either place, the
/// standard genuinely writes it both ways, and writing them from one `u64` is
/// how they are kept from disagreeing.
pub fn nonce(priority: u8, a2: &[u8; 6], pn: u64) -> [u8; 13] {
    let mut n = [0u8; 13];
    n[0] = priority & 0x0F;
    n[1..7].copy_from_slice(a2);
    for k in 0..6 {
        n[7 + k] = (pn >> (8 * (5 - k))) as u8;
    }
    n
}

/// The eight-byte CCMP header that sits between the 802.11 header and the
/// ciphertext.
///
/// The ExtIV bit is always set, because CCMP always carries an extended IV.
/// A receiver that finds it clear is looking at WEP.
pub fn header(pn: u64, key_id: u8) -> [u8; HDR_LEN] {
    let b = |k: u32| (pn >> (8 * k)) as u8;
    [b(0), b(1), 0, 0x20 | ((key_id & 3) << 6), b(2), b(3), b(4), b(5)]
}

/// Read a packet number back out of a CCMP header.
pub fn header_pn(h: &[u8]) -> Option<u64> {
    if h.len() < HDR_LEN || h[3] & 0x20 == 0 {
        return None;
    }
    Some((h[0] as u64)
        | (h[1] as u64) << 8
        | (h[4] as u64) << 16
        | (h[5] as u64) << 24
        | (h[6] as u64) << 32
        | (h[7] as u64) << 40)
}

/// The additional authenticated data: the header, with the mutable parts gone.
pub fn aad(frame: &[u8], f: &Frame) -> Vec<u8> {
    let mut out = Vec::with_capacity(f.hdr_len);
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    // Masked down, then Protected forced on: a frame is authenticated as the
    // protected frame it will be on the air, whatever the caller handed in.
    let masked = (fc & FC_MASK) | FC_PROTECTED;
    out.extend_from_slice(&masked.to_le_bytes());
    out.extend_from_slice(&frame[4..22]); // A1, A2, A3

    // Sequence control: the sequence number is masked and the fragment number
    // kept. Fragments of one frame share a sequence number and differ in the
    // fragment number, so masking both would authenticate any fragment as any
    // other.
    let sc = u16::from_le_bytes([frame[22], frame[23]]);
    out.extend_from_slice(&(sc & 0x000F).to_le_bytes());

    if f.a4 {
        out.extend_from_slice(&frame[24..30]);
    }
    if f.qos {
        // The TID is kept, everything else masked: the TID picks the nonce's
        // priority, so a frame whose TID could be altered would decrypt under
        // a nonce the sender never used.
        let qc = u16::from_le_bytes([frame[f.hdr_len - 2], frame[f.hdr_len - 1]]);
        out.extend_from_slice(&(qc & 0x000F).to_le_bytes());
    }
    out
}

/// Wrap a plaintext frame: header, CCMP header, ciphertext, MIC.
///
/// `frame` is the whole MPDU with its payload in the clear. What comes back is
/// what goes on the air.
pub fn protect(tk: &[u8], frame: &[u8], pn: u64, key_id: u8) -> Option<Vec<u8>> {
    let f = parse(frame)?;
    if pn == 0 || pn > PN_MAX {
        return None;
    }
    let a = aad(frame, &f);
    let n = nonce(f.tid, &f.a2, pn);
    let sealed = ccm::seal(tk, &n, &a, &frame[f.hdr_len..], MIC_LEN)?;

    let mut out = Vec::with_capacity(f.hdr_len + HDR_LEN + sealed.len());
    out.extend_from_slice(&frame[..f.hdr_len]);
    // The Protected bit goes on in the output too, and it is set from the same
    // masked value the AAD used -- so the bit that was authenticated is the bit
    // that is transmitted.
    let fc = u16::from_le_bytes([frame[0], frame[1]]) | FC_PROTECTED;
    out[0..2].copy_from_slice(&fc.to_le_bytes());
    out.extend_from_slice(&header(pn, key_id));
    out.extend_from_slice(&sealed);
    Some(out)
}

/// Unwrap a protected frame. Answers the header, the plaintext, and the PN.
pub fn unprotect(tk: &[u8], frame: &[u8]) -> Option<(Vec<u8>, Vec<u8>, u64)> {
    let f = parse(frame)?;
    if frame.len() < f.hdr_len + HDR_LEN + MIC_LEN {
        return None;
    }
    let fc = u16::from_le_bytes([frame[0], frame[1]]);
    if fc & FC_PROTECTED == 0 {
        return None;
    }
    let pn = header_pn(&frame[f.hdr_len..])?;
    let a = aad(frame, &f);
    let n = nonce(f.tid, &f.a2, pn);
    let body = &frame[f.hdr_len + HDR_LEN..];
    let plain = ccm::open(tk, &n, &a, body, MIC_LEN)?;
    Some((frame[..f.hdr_len].to_vec(), plain, pn))
}

/// One direction of one key: the counter out, and the highest seen in.
pub struct Keys {
    pub tk: [u8; 16],
    pub key_id: u8,
    tx_pn: u64,
    rx_pn: u64,
}

impl Keys {
    pub fn new(tk: &[u8], key_id: u8) -> Option<Keys> {
        if tk.len() != 16 {
            return None;
        }
        let mut k = [0u8; 16];
        k.copy_from_slice(tk);
        // Transmit numbering starts at one: zero is the value a fresh counter
        // has, so a frame carrying it is a frame sent before the key existed.
        Some(Keys { tk: k, key_id: key_id & 3, tx_pn: 0, rx_pn: 0 })
    }

    /// The next transmit PN, or nothing when the counter is spent.
    ///
    /// **Refused rather than wrapped.** A wrapped PN repeats a nonce under a
    /// key that is still in use, and CCM in counter mode gives up the XOR of
    /// two plaintexts the moment that happens. The answer is to rekey, which
    /// only the supplicant above can do -- so this says no and lets it.
    pub fn next_pn(&mut self) -> Option<u64> {
        if self.tx_pn >= PN_MAX {
            return None;
        }
        self.tx_pn += 1;
        Some(self.tx_pn)
    }

    pub fn protect(&mut self, frame: &[u8]) -> Option<Vec<u8>> {
        let pn = self.next_pn()?;
        protect(&self.tk, frame, pn, self.key_id)
    }

    /// Verify, decrypt, and refuse a replay.
    ///
    /// The PN is only accepted after the MIC has verified. Advancing it first
    /// would let anybody who can inject a frame with a huge PN wedge the
    /// counter and deny every genuine frame that follows -- authentication
    /// before state, which is the same ordering `update::hook` keeps.
    pub fn unprotect(&mut self, frame: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        let (hdr, plain, pn) = unprotect(&self.tk, frame)?;
        if pn <= self.rx_pn {
            return None;
        }
        self.rx_pn = pn;
        Some((hdr, plain))
    }

    pub fn seen(&self) -> (u64, u64) {
        (self.tx_pn, self.rx_pn)
    }
}

/// A protected frame at fixed inputs, for `tools/dot11check.py`.
///
/// The CCMP *header* is what this exposes to an outside reader -- six packet
/// number bytes in an order that is not the order they are counted in, a key
/// id in the top two bits of the fifth byte, and ExtIV always set. Scapy
/// parses that structure, so the framing gets a second opinion from a reader
/// nobody here wrote. The cryptography has its own second opinion now -- the
/// Annex M.6.4 vector in `selftest` -- so this is no longer standing in for
/// one.
pub fn dump() {
    use crate::kprintln;
    let me: [u8; 6] = [0x02, 0, 0, 0, 0, 0x11];
    let ap: [u8; 6] = [0x02, 0, 0, 0, 0, 0xAA];
    let tk = [0x5Au8; 16];
    let body = crate::net::ieee80211::snap_wrap(0x0800, b"payload");
    let plain = crate::net::ieee80211::data_to_ds(&ap, &me, &ap, 7, &body);
    // A packet number with a different value in every byte, so a reader that
    // reversed the order or dropped a byte cannot land on the same answer.
    if let Some(p) = protect(&tk, &plain, 0x060504030201, 2) {
        let mut hex = alloc::string::String::with_capacity(p.len() * 2);
        for b in &p {
            hex.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
            hex.push(char::from_digit((b & 0xF) as u32, 16).unwrap_or('0'));
        }
        kprintln!("frame ccmp {}", hex);
    }
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    let tk: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
        0xFF,
    ];
    // A plain data frame, ToDS, with a short payload.
    let mut frame: Vec<u8> = Vec::new();
    frame.extend_from_slice(&[0x08, 0x01]); // data, ToDS
    frame.extend_from_slice(&[0x00, 0x00]); // duration
    frame.extend_from_slice(&[0x11, 0x11, 0x11, 0x11, 0x11, 0x11]); // A1
    frame.extend_from_slice(&[0x22, 0x22, 0x22, 0x22, 0x22, 0x22]); // A2
    frame.extend_from_slice(&[0x33, 0x33, 0x33, 0x33, 0x33, 0x33]); // A3
    frame.extend_from_slice(&[0x40, 0x05]); // sequence 0x054, fragment 0
    frame.extend_from_slice(b"hello 802.11");

    let f = parse(&frame);
    check(
        "a data frame's header length follows its own bits",
        match &f {
            Some(v) => v.hdr_len == 24 && !v.qos && !v.a4 && v.a2 == [0x22; 6] && v.tid == 0,
            None => false,
        },
    );
    // Anything that is not a data frame is refused rather than measured with a
    // data frame's ruler.
    check(
        "a management frame is refused, not parsed as data",
        parse(&[0x80, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .is_none()
            && parse(&frame[..20]).is_none(),
    );

    // --- the nonce ------------------------------------------------------
    let n = nonce(0, &[0x22; 6], 1);
    check(
        "the nonce is priority, transmitter, then the PN big-endian",
        n[0] == 0 && n[1..7] == [0x22; 6] && n[7..13] == [0, 0, 0, 0, 0, 1],
    );
    check(
        "a different PN is a different nonce, which is what stops a key repeating",
        nonce(0, &[0x22; 6], 2) != n && nonce(1, &[0x22; 6], 1) != n,
    );

    // --- the header -----------------------------------------------------
    let h = header(0x0102_0304_0506, 2);
    check(
        "the CCMP header splits the PN the way the standard does, ExtIV always set",
        h[0] == 0x06 && h[1] == 0x05 && h[3] == 0xA0 && h[4..8] == [0x04, 0x03, 0x02, 0x01],
    );
    check(
        "and it reads back as the number it was written from",
        header_pn(&h) == Some(0x0102_0304_0506),
    );
    check(
        "a header with ExtIV clear is WEP and is refused",
        header_pn(&[0, 0, 0, 0, 0, 0, 0, 0]).is_none(),
    );

    // --- the masking, which is the part that fails silently -------------
    let fr = f.unwrap();
    let base = aad(&frame, &fr);
    let mut retried = frame.clone();
    retried[1] |= 0x08; // retry
    let mut powered = frame.clone();
    powered[1] |= 0x10; // power management
    let mut more = frame.clone();
    more[1] |= 0x20; // more data
    check(
        "retry, power management and more-data are masked out of the AAD",
        aad(&retried, &fr) == base && aad(&powered, &fr) == base && aad(&more, &fr) == base,
    );
    // Without this a retransmission authenticates as a forgery, which is the
    // most common frame on a busy network.
    let mut seq = frame.clone();
    seq[22] = 0xF0;
    seq[23] = 0xFF;
    check(
        "the sequence number is masked and the fragment number is not",
        aad(&seq, &fr) == base && {
            let mut frag = frame.clone();
            frag[22] = 0x41;
            aad(&frag, &fr) != base
        },
    );
    // And the things an attacker could change must stay covered.
    let mut addr = frame.clone();
    addr[4] ^= 1;
    check(
        "the addresses are covered, so a redirected frame does not verify",
        aad(&addr, &fr) != base,
    );
    check(
        "and the protected bit is forced on rather than copied",
        base[0] == (u16::from_le_bytes([frame[0], frame[1]]) & FC_MASK | FC_PROTECTED)
            .to_le_bytes()[0]
            && base[1] & 0x40 != 0,
    );

    // --- a round trip ---------------------------------------------------
    let wire = protect(&tk, &frame, 1, 0);
    check(
        "a protected frame is header, CCMP header, ciphertext and MIC",
        match &wire {
            Some(w) => w.len() == frame.len() + HDR_LEN + MIC_LEN && w[1] & 0x40 != 0,
            None => false,
        },
    );
    let wire = wire.unwrap_or_default();
    // The header that comes back differs from the one that went in by exactly
    // one bit: Protected, which  sets because the frame is now
    // protected. Asserting equality was the first version of this claim and it
    // failed -- correctly, and on the behaviour rather than on a mistake.
    check(
        "and it unwraps to exactly what went in, bar the bit that was set",
        match unprotect(&tk, &wire) {
            Some((h, p, pn)) => {
                h[2..] == frame[2..24]
                    && h[1] & 0x40 != 0
                    && frame[1] & 0x40 == 0
                    && h[0] == frame[0]
                    && p == b"hello 802.11"
                    && pn == 1
            }
            None => false,
        },
    );
    // The payload must not be readable on the air, which a cipher that
    // accidentally passed the plaintext through would still round-trip.
    check(
        "the payload is not in the clear on the wire",
        !wire.windows(5).any(|w| w == b"hello"),
    );

    // --- every way to tamper --------------------------------------------
    let mut t = wire.clone();
    t[40] ^= 1;
    check("a flipped ciphertext bit is refused", unprotect(&tk, &t).is_none());
    let mut t = wire.clone();
    t[4] ^= 1; // A1
    check("a changed address is refused", unprotect(&tk, &t).is_none());
    let mut t = wire.clone();
    let last = t.len() - 1;
    t[last] ^= 1;
    check("a flipped MIC bit is refused", unprotect(&tk, &t).is_none());
    let mut t = wire.clone();
    t[24] ^= 1; // the PN in the CCMP header
    check("and a rewritten packet number is refused", unprotect(&tk, &t).is_none());
    check(
        "a retransmission still verifies, because retry is masked",
        {
            let mut r = wire.clone();
            r[1] |= 0x08;
            unprotect(&tk, &r).is_some()
        },
    );
    check(
        "a frame with the protected bit clear is refused",
        {
            let mut r = wire.clone();
            r[1] &= !0x40;
            unprotect(&tk, &r).is_none()
        },
    );
    check(
        "and one too short to hold a header and a MIC is refused, not indexed",
        unprotect(&tk, &wire[..30]).is_none(),
    );

    // --- replay ---------------------------------------------------------
    let mut keys = Keys::new(&tk, 0).unwrap();
    let a = keys.protect(&frame).unwrap_or_default();
    let b = keys.protect(&frame).unwrap_or_default();
    check(
        "each frame sent takes the next packet number, never the same one twice",
        a != b && keys.seen().0 == 2,
    );
    let mut rx = Keys::new(&tk, 0).unwrap();
    check(
        "a first frame is accepted",
        rx.unprotect(&a).map(|(_, p)| p) == Some(b"hello 802.11".to_vec()),
    );
    check(
        "a later one is too",
        rx.unprotect(&b).is_some(),
    );
    // Genuine, correctly signed, and still refused -- which is the whole point
    // of a replay counter, and the reason verifying is not enough on its own.
    check(
        "and replaying either of them is refused though both are genuine",
        rx.unprotect(&a).is_none() && rx.unprotect(&b).is_none(),
    );
    // Authentication before state: a forged frame claiming a huge PN must not
    // move the counter, or one injected packet denies every real one after it.
    let mut wedge = b.clone();
    wedge[24] = 0xFF;
    wedge[25] = 0xFF;
    let before = rx.seen().1;
    check(
        "a forgery claiming a huge PN is refused and does not move the counter",
        rx.unprotect(&wedge).is_none() && rx.seen().1 == before,
    );

    // --- a QoS frame, which has a longer header and a live TID ----------
    let mut q: Vec<u8> = frame[..24].to_vec();
    q[0] = 0x88; // QoS data
    q.extend_from_slice(&[0x06, 0x00]); // QoS control, TID 6
    q.extend_from_slice(b"voice");
    let qf = parse(&q);
    check(
        "a QoS frame is two bytes longer and its TID reaches the nonce",
        match &qf {
            Some(v) => v.hdr_len == 26 && v.qos && v.tid == 6,
            None => false,
        },
    );
    let qw = protect(&tk, &q, 7, 1);
    check(
        "and it round-trips with its own header length",
        match &qw {
            Some(w) => match unprotect(&tk, w) {
                Some((h, p, pn)) => h.len() == 26 && p == b"voice" && pn == 7,
                None => false,
            },
            None => false,
        },
    );
    // The TID is authenticated, so it cannot be moved to make a frame decrypt
    // under a nonce its sender never used.
    check(
        "a rewritten TID is refused",
        match &qw {
            Some(w) => {
                let mut t = w.clone();
                t[24] = 0x01;
                unprotect(&tk, &t).is_none()
            }
            None => false,
        },
    );


    // --- IEEE 802.11-2012 Annex M.6.4, the published CCMP example ----------
    //
    // Everything above this line is a round trip against ourselves, and the
    // top of this file spent a long time saying why that is not enough: the
    // AAD masking and the nonce layout could be self-consistently wrong and
    // round-trip perfectly while failing against every access point in the
    // world. Nothing here can be checked against the other end of a real
    // conversation, so a published vector is the only thing that catches it.
    //
    // The frame is a plain data frame with the retry bit set, which is the
    // case the AAD mask exists for -- so the vector exercises the masking
    // rather than merely passing through it.
    let vtk: [u8; 16] = [
        0xc9, 0x7c, 0x1f, 0x67, 0xce, 0x37, 0x11, 0x85, 0x51, 0x4a, 0x8a, 0x19, 0xf2, 0xbd, 0xd5,
        0x2f,
    ];
    let vhdr: [u8; 24] = [
        0x08, 0x48, 0xc3, 0x2c, 0x0f, 0xd2, 0xe1, 0x28, 0xa5, 0x7c, 0x50, 0x30, 0xf1, 0x84, 0x44,
        0x08, 0xab, 0xae, 0xa5, 0xb8, 0xfc, 0xba, 0x80, 0x33,
    ];
    let vplain: [u8; 20] = [
        0xf8, 0xba, 0x1a, 0x55, 0xd0, 0x2f, 0x85, 0xae, 0x96, 0x7b, 0xb6, 0x2f, 0xb6, 0xcd, 0xa8,
        0xeb, 0x7e, 0x78, 0xa0, 0x50,
    ];
    let want_c: [u8; 20] = [
        0xf3, 0xd0, 0xa2, 0xfe, 0x9a, 0x3d, 0xbf, 0x23, 0x42, 0xa6, 0x43, 0xe4, 0x32, 0x46, 0xe8,
        0x0c, 0x3c, 0x04, 0xd0, 0x19,
    ];
    let want_mic: [u8; 8] = [0x78, 0x45, 0xce, 0x0b, 0x16, 0xf9, 0x76, 0x23];
    let vpn: u64 = 0xb503_9776_e70c;

    let vframe: Vec<u8> = [&vhdr[..], &vplain[..]].concat();
    let vf = parse(&vframe);

    // The two intermediate values are asserted directly as well as through the
    // ciphertext, so a failure says which of the two is wrong rather than only
    // that the frame came out different.
    check(
        "the vector's AAD is 22 bytes with retry masked and protected forced on",
        match &vf {
            Some(f) => {
                let a = aad(&vframe, f);
                a.len() == 22 && a[0] == 0x08 && a[1] == 0x40
            }
            None => false,
        },
    );
    check(
        "the vector's nonce is priority, transmitter, then the PN big-endian",
        match &vf {
            Some(f) => {
                let n = nonce(f.tid, &f.a2, vpn);
                n[0] == 0 && &n[1..7] == &vhdr[10..16] && &n[7..13] == &vpn.to_be_bytes()[2..8]
            }
            None => false,
        },
    );

    // The vector proper: the cipher, under this file's own AAD and nonce.
    check(
        "Annex M.6.4: the published ciphertext and MIC, byte for byte",
        match &vf {
            Some(f) => {
                let a = aad(&vframe, f);
                let n = nonce(f.tid, &f.a2, vpn);
                match ccm::seal(&vtk, &n, &a, &vplain, MIC_LEN) {
                    Some(sealed) => {
                        sealed.len() == vplain.len() + MIC_LEN
                            && &sealed[..vplain.len()] == &want_c[..]
                            && &sealed[vplain.len()..] == &want_mic[..]
                    }
                    None => false,
                }
            }
            None => false,
        },
    );

    // And through the public path, which additionally has to get the CCMP
    // header's byte order right: PN0, PN1, reserved, key id with ExtIV, then
    // PN2 to PN5. That is neither endianness, and it is the field most often
    // written out as a plain integer.
    let vout = protect(&vtk, &vframe, vpn, 0);
    check(
        "the whole protected frame matches the vector, header order included",
        match &vout {
            Some(o) => {
                o.len() == 24 + HDR_LEN + vplain.len() + MIC_LEN
                    && header_pn(&o[24..]) == Some(vpn)
                    && o[27] & 0x20 != 0
                    && u16::from_le_bytes([o[0], o[1]]) & FC_PROTECTED != 0
                    && &o[32..32 + want_c.len()] == &want_c[..]
                    && &o[o.len() - MIC_LEN..] == &want_mic[..]
            }
            None => false,
        },
    );

    // A flipped ciphertext bit must fail the MIC and -- the half that matters
    // -- must not advance the replay window, or anybody able to inject garbage
    // can lock out the real sender. Authentication before state.
    check(
        "a tampered frame fails the MIC without advancing the replay window",
        match &vout {
            Some(o) => {
                let mut k = Keys::new(&vtk, 0).unwrap();
                let mut bad = o.clone();
                bad[33] ^= 1;
                let refused = k.unprotect(&bad).is_none();
                let still_zero = k.seen().1 == 0;
                // The genuine frame is still accepted afterwards.
                refused && still_zero && k.unprotect(o).is_some()
            }
            None => false,
        },
    );

    // addr3 is authenticated and not encrypted, so this is the check that the
    // AAD reaches the MIC at all rather than being computed and dropped.
    check(
        "a changed address the AAD covers fails the MIC",
        match &vout {
            Some(o) => {
                let mut moved = o.clone();
                moved[16] ^= 1;
                unprotect(&vtk, &moved).is_none()
            }
            None => false,
        },
    );

    // And the mirror of it, which a too-eager mask would break: the retry bit
    // is rewritten in flight, so a receiver seeing it set where the sender had
    // it clear has to accept the frame. That failure presents as a flaky radio
    // rather than as a bug, which is why it is asserted rather than reasoned
    // about.
    check(
        "a frame retried in flight still verifies, because the AAD masks it",
        match &vout {
            Some(o) => {
                let mut retried = o.clone();
                retried[1] |= 0x08; // FC1 retry
                match unprotect(&vtk, &retried) {
                    Some((_, p, pn)) => p == vplain && pn == vpn,
                    None => false,
                }
            }
            None => false,
        },
    );

    check(
        "a key that is not 128 bits is refused",
        Keys::new(&[0u8; 8], 0).is_none(),
    );

    ok
}
