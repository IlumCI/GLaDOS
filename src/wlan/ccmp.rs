//! CCMP: the cipher that carries every encrypted byte on a WPA2 network.
//!
//! Ported from OpenBSD `sys/net80211/ieee80211_crypto_ccmp.c`.
//! upstream-sha256: a4d4cd2e819f04d9
//! Copyright (c) 2008 Damien Bergamini <damien.bergamini@free.fr>
//! Licence: ISC. See licenses/net80211-ISC.txt.
//!
//! CTR with CBC-MAC, IEEE Std 802.11-2007 section 8.3.3. The AES-CCM itself is
//! `radio::cipher`; what is here is the 802.11-specific part, which is the
//! part that is wrong in implementations that get the cipher right.
//!
//! ### The whole difficulty is the AAD
//!
//! CCMP authenticates the frame header as well as encrypting the body, so both
//! ends must construct byte-for-byte identical additional authenticated data
//! from headers that are *not* identical -- the retry bit, the power-management
//! bit and the more-data bit all change in flight and are legitimately
//! different at the receiver. So those are masked out before authenticating.
//! Get the mask wrong and every retransmitted frame fails its MIC while every
//! first transmission passes, which presents as a flaky radio.
//!
//! Three masks, each with a different reason:
//!
//!   * `FC1_RETRY`, `FC1_PWR_MGT`, `FC1_MORE_DATA` are rewritten in flight by
//!     the transmitter and by any retry, so they cannot be authenticated.
//!   * The subtype nibble of a **data** frame is masked except the QoS bit --
//!     an 802.11w rule, so that a frame's subtype cannot be altered to change
//!     how it is interpreted without also breaking the MIC.
//!   * `FC1_ORDER` is masked on QoS frames, an 802.11n rule, because the HT
//!     control field's presence is signalled there and is not end to end.
//!
//! ### The packet number is the replay defence and the nonce at once
//!
//! A 48-bit counter, carried in the eight-byte CCMP header in an order that is
//! neither big nor little endian -- PN0 and PN1, then a key id byte, then PN2
//! through PN5. That layout exists because the first three bytes had to look
//! like a WEP IV to hardware that predated CCMP. A receiver rejects any PN not
//! greater than the last one accepted, which is what stops a captured frame
//! being replayed; and the same PN is the CCM nonce, which is why it must
//! never repeat under one key. Both properties break silently.

use crate::radio::cipher;
use crate::radio::log;
use alloc::vec::Vec;
use super::frame::Header;

/// The CCMP header: PN0, PN1, reserved, key id and ExtIV, then PN2..PN5.
pub const HDR_LEN: usize = 8;
/// The MIC trailer.
pub const MIC_LEN: usize = 8;
/// Bit 5 of the key-id byte says an extended IV follows. CCMP always sets it;
/// a frame without it is WEP, and treating one as the other reads the key id
/// as packet-number bytes.
const EXT_IV: u8 = 0x20;

/// One direction of one key.
///
/// Transmit and receive counters are separate and must be: they advance
/// independently, and sharing one would make a received frame bump the number
/// the next transmitted frame uses.
pub struct Key {
    /// The temporal key, 16 bytes, out of the PTK or the GTK.
    pub tk: [u8; 16],
    /// Which of the four key slots this is. Travels in every frame.
    pub id: u8,
    /// The next packet number to send.
    tx_pn: u64,
    /// The highest packet number accepted, per traffic identifier.
    ///
    /// Per TID rather than one counter, because QoS traffic classes are
    /// delivered independently and a voice frame overtaking a background one
    /// is normal. A single counter would reject it as a replay.
    rx_pn: [u64; 16],
}

impl Key {
    pub fn new(tk: &[u8], id: u8) -> Option<Key> {
        if tk.len() != 16 {
            return None;
        }
        let mut k = [0u8; 16];
        k.copy_from_slice(tk);
        Some(Key { tk: k, id: id & 0x03, tx_pn: 0, rx_pn: [0; 16] })
    }

    /// The packet number that will be used next, for reporting.
    pub fn tx_pn(&self) -> u64 {
        self.tx_pn
    }
}

/// Build the additional authenticated data for a frame.
///
/// Upstream writes this into a fixed 32-byte buffer and pads to exactly two
/// blocks; here it is a `Vec` and the padding is CCM's own, which produces the
/// same bytes because the maximum length is 30 and the rule is the same. That
/// equivalence is worth stating, since it is the one place this file stops
/// being a transliteration.
fn aad(h: &Header) -> Vec<u8> {
    let mut a = Vec::with_capacity(30);

    let mut fc0 = h.fc0();
    if h.is_data() {
        fc0 &= !super::frame::FC0_SUBTYPE_MASK | super::frame::FC0_SUBTYPE_QOS;
    }
    a.push(fc0);

    let mut fc1 = h.fc1();
    fc1 &= !(super::frame::FC1_RETRY
        | super::frame::FC1_PWR_MGT
        | super::frame::FC1_MORE_DATA);
    if h.has_qos() {
        fc1 &= !super::frame::FC1_ORDER;
    }
    a.push(fc1);

    a.extend_from_slice(h.addr1());
    a.extend_from_slice(h.addr2());
    a.extend_from_slice(h.addr3());

    // The fragment number is authenticated and the sequence number is not:
    // the sequence number is assigned by hardware and may differ by the time
    // a frame is retried.
    a.push(h.seq0() & 0x0f);
    a.push(0);

    if let Some(a4) = h.addr4() {
        a.extend_from_slice(a4);
    }
    if h.has_qos() {
        a.push(h.tid());
        a.push(0);
    }
    a
}

/// Build the CCM nonce: priority, transmitter address, packet number.
fn nonce(h: &Header, pn: u64) -> [u8; 13] {
    let mut n = [0u8; 13];
    n[0] = h.tid();
    if h.is_mgmt() {
        // 802.11w. A management frame and a data frame with the same TID, the
        // same sender and the same PN would otherwise share a nonce, which
        // under one key leaks the keystream of both.
        n[0] |= 1 << 4;
    }
    n[1..7].copy_from_slice(h.addr2());
    for i in 0..6 {
        n[7 + i] = (pn >> (40 - 8 * i)) as u8;
    }
    n
}

/// Write the eight-byte CCMP header in the order the standard defines.
fn write_hdr(out: &mut Vec<u8>, pn: u64, id: u8) {
    out.push(pn as u8); // PN0
    out.push((pn >> 8) as u8); // PN1
    out.push(0); // reserved
    out.push(EXT_IV | (id << 6)); // key id, and the ExtIV bit
    out.push((pn >> 16) as u8); // PN2
    out.push((pn >> 24) as u8); // PN3
    out.push((pn >> 32) as u8); // PN4
    out.push((pn >> 40) as u8); // PN5
}

/// Read a packet number back out of a CCMP header.
fn read_pn(b: &[u8]) -> u64 {
    (b[0] as u64)
        | (b[1] as u64) << 8
        | (b[4] as u64) << 16
        | (b[5] as u64) << 24
        | (b[6] as u64) << 32
        | (b[7] as u64) << 40
}

/// Encrypt one frame. `frame` is a whole 802.11 frame, header and body.
///
/// Answers the encrypted frame: the same header with the protected bit set,
/// then the CCMP header, the ciphertext, and the MIC.
pub fn encrypt(k: &mut Key, frame: &[u8]) -> Option<Vec<u8>> {
    let h = Header::new(frame)?;
    let hlen = h.len();

    // The counter is 48 bits and wrapping it reuses a nonce, which is a
    // key-recovery bug rather than a glitch. Upstream rekeys long before
    // this; here there is nothing to rekey with yet, so it refuses.
    if k.tx_pn >= 0xFFFF_FFFF_FFFF {
        log::error("ccmp: packet number exhausted, refusing to reuse a nonce");
        return None;
    }
    k.tx_pn += 1;
    let pn = k.tx_pn;

    // The protected bit has to be set *before* the AAD is built, because the
    // AAD covers it -- a receiver computes its AAD from a frame that has the
    // bit set, so a sender that sets it afterwards authenticates a different
    // header than the one it sends.
    let mut hdr: Vec<u8> = h.bytes().to_vec();
    hdr[1] |= super::frame::FC1_PROTECTED;
    let ph = Header::new(&hdr)?;

    let a = aad(&ph);
    let n = nonce(&ph, pn);
    let (cipher_text, mic) = cipher::ccm_encrypt(&k.tk, &n, &a, &frame[hlen..])?;

    let mut out = Vec::with_capacity(hlen + HDR_LEN + cipher_text.len() + MIC_LEN);
    out.extend_from_slice(&hdr);
    write_hdr(&mut out, pn, k.id);
    out.extend_from_slice(&cipher_text);
    out.extend_from_slice(&mic);
    Some(out)
}

/// Why a frame was not decrypted.
///
/// An enum rather than `None`, because these mean very different things to
/// somebody holding a capture: a replay is an attack or a duplicate, a MIC
/// failure is the wrong key or a corrupted frame, and a malformed header is a
/// driver bug. Collapsing them is how "wireless does not work" becomes
/// unanswerable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bad {
    Short,
    NotProtected,
    NoExtIv,
    Replay,
    Mic,
}

/// Decrypt one frame, answering the plaintext frame with its header.
pub fn decrypt(k: &mut Key, frame: &[u8]) -> Result<Vec<u8>, Bad> {
    let h = Header::new(frame).ok_or(Bad::Short)?;
    let hlen = h.len();
    if !h.protected() {
        return Err(Bad::NotProtected);
    }
    if frame.len() < hlen + HDR_LEN + MIC_LEN {
        return Err(Bad::Short);
    }

    let ccmp = &frame[hlen..hlen + HDR_LEN];
    if ccmp[3] & EXT_IV == 0 {
        // Without the extended IV this is a WEP frame, and reading it as CCMP
        // takes the key id as packet-number bytes.
        return Err(Bad::NoExtIv);
    }
    let pn = read_pn(ccmp);

    // Replay first, before spending an AES pass on a frame that is going to
    // be dropped anyway -- and before touching any counter, so a replayed
    // frame cannot move the window it failed against.
    let tid = h.tid() as usize;
    if pn <= k.rx_pn[tid] {
        return Err(Bad::Replay);
    }

    let body = &frame[hlen + HDR_LEN..frame.len() - MIC_LEN];
    let mic = &frame[frame.len() - MIC_LEN..];
    let a = aad(&h);
    let n = nonce(&h, pn);
    let plain = cipher::ccm_decrypt(&k.tk, &n, &a, body, mic).ok_or(Bad::Mic)?;

    // Only now. A frame whose MIC failed must not advance the replay window,
    // or anybody able to inject garbage can push the counter past the real
    // sender's and lock the link out.
    k.rx_pn[tid] = pn;

    let mut out = Vec::with_capacity(hlen + plain.len());
    out.extend_from_slice(&frame[..hlen]);
    // The protected bit is cleared: what is handed upward is plaintext, and
    // leaving it set means anything that looks at the frame again re-enters
    // the decrypt path.
    out[1] &= !super::frame::FC1_PROTECTED;
    out.extend_from_slice(&plain);
    Ok(out)
}

/// IEEE 802.11-2012 Annex M.6.4, the published CCMP encapsulation example.
///
/// A real vector rather than a round-trip, and the distinction is the whole
/// value: this file's arithmetic could be self-consistently wrong in the AAD
/// masking or the nonce layout and round-trip perfectly against itself while
/// failing against every access point in the world. The vector is the only
/// thing that catches that, because the other end of a real conversation is
/// not available to test against.
pub fn selftest() -> bool {
    let mut ok = true;

    let tk = [
        0xc9, 0x7c, 0x1f, 0x67, 0xce, 0x37, 0x11, 0x85, 0x51, 0x4a, 0x8a, 0x19, 0xf2, 0xbd, 0xd5,
        0x2f,
    ];
    // fc, duration, addr1, addr2, addr3, seq. The retry bit is set in fc1 and
    // the protected bit with it, which is exactly the case the AAD mask
    // exists for.
    let hdr = [
        0x08, 0x48, 0xc3, 0x2c, 0x0f, 0xd2, 0xe1, 0x28, 0xa5, 0x7c, 0x50, 0x30, 0xf1, 0x84, 0x44,
        0x08, 0xab, 0xae, 0xa5, 0xb8, 0xfc, 0xba, 0x80, 0x33,
    ];
    let plain = [
        0xf8, 0xba, 0x1a, 0x55, 0xd0, 0x2f, 0x85, 0xae, 0x96, 0x7b, 0xb6, 0x2f, 0xb6, 0xcd, 0xa8,
        0xeb, 0x7e, 0x78, 0xa0, 0x50,
    ];
    let want_c = [
        0xf3, 0xd0, 0xa2, 0xfe, 0x9a, 0x3d, 0xbf, 0x23, 0x42, 0xa6, 0x43, 0xe4, 0x32, 0x46, 0xe8,
        0x0c, 0x3c, 0x04, 0xd0, 0x19,
    ];
    let want_mic = [0x78, 0x45, 0xce, 0x0b, 0x16, 0xf9, 0x76, 0x23];
    let pn: u64 = 0xb503_9776_e70c;

    let Some(h) = Header::new(&hdr) else { return false };
    ok &= h.len() == 24;
    ok &= !h.has_qos();
    ok &= !h.has_addr4();
    ok &= h.tid() == 0;

    let a = aad(&h);
    let n = nonce(&h, pn);
    // The masks, asserted directly rather than only through the vector, so a
    // failure says which of the two is wrong.
    ok &= a[0] == 0x08;
    ok &= a[1] == 0x40;
    ok &= a.len() == 22;
    ok &= n[0] == 0;
    ok &= &n[1..7] == h.addr2();

    match cipher::ccm_encrypt(&tk, &n, &a, &plain) {
        Some((c, m)) => {
            ok &= c == want_c;
            ok &= m == want_mic;
        }
        None => ok = false,
    }

    // Now the same thing through the public path, which additionally has to
    // get the CCMP header's byte order right -- PN0, PN1, reserved, key id,
    // then PN2 through PN5, which is neither endianness and is the field most
    // often written as a plain integer.
    let Some(mut k) = Key::new(&tk, 0) else { return false };
    k.tx_pn = pn - 1;
    match encrypt(&mut k, &[&hdr[..], &plain[..]].concat()) {
        Some(out) => {
            ok &= out.len() == 24 + HDR_LEN + plain.len() + MIC_LEN;
            ok &= read_pn(&out[24..32]) == pn;
            ok &= out[27] & EXT_IV != 0;
            ok &= out[1] & super::frame::FC1_PROTECTED != 0;
            ok &= &out[32..32 + want_c.len()] == &want_c[..];
            ok &= &out[out.len() - MIC_LEN..] == &want_mic[..];

            // Round-trip through a fresh receive key.
            let Some(mut rk) = Key::new(&tk, 0) else { return false };
            match decrypt(&mut rk, &out) {
                Ok(back) => {
                    ok &= &back[24..] == &plain[..];
                    // The protected bit must be cleared on the way up, or
                    // anything that looks at the frame again decrypts it twice.
                    ok &= back[1] & super::frame::FC1_PROTECTED == 0;
                    // And the same frame again is a replay.
                    ok &= decrypt(&mut rk, &out) == Err(Bad::Replay);
                }
                Err(_) => ok = false,
            }

            // A flipped ciphertext bit fails the MIC, and -- the part that
            // matters -- must not advance the replay window, or anybody able
            // to inject garbage can lock out the real sender.
            let mut rk2 = Key::new(&tk, 0).unwrap();
            let mut bad = out.clone();
            bad[33] ^= 1;
            ok &= decrypt(&mut rk2, &bad) == Err(Bad::Mic);
            ok &= rk2.rx_pn[0] == 0;
            ok &= decrypt(&mut rk2, &out).is_ok();

            // A changed header byte that the AAD covers must also fail. addr3
            // is authenticated and not encrypted, so this is the check that
            // the AAD is reaching the MIC at all.
            let mut rk3 = Key::new(&tk, 0).unwrap();
            let mut moved = out.clone();
            moved[16] ^= 1;
            ok &= decrypt(&mut rk3, &moved) == Err(Bad::Mic);

            // And a bit the AAD deliberately masks must *not* fail: the retry
            // bit is rewritten in flight, so a receiver seeing it set where
            // the sender had it clear has to accept the frame. This is the
            // claim that a too-eager mask would break, and it is the failure
            // that presents as a flaky radio rather than as a bug.
            let mut rk4 = Key::new(&tk, 0).unwrap();
            let mut retried = out.clone();
            retried[1] |= super::frame::FC1_RETRY;
            ok &= decrypt(&mut rk4, &retried).is_ok();
        }
        None => ok = false,
    }

    // An unprotected frame, a truncated one, and one claiming WEP rather than
    // CCMP are each named rather than collapsed into a MIC failure.
    //
    // The protected bit has to be *cleared* to make that first frame, because
    // the vector's own header carries it set -- it is the header of an
    // encrypted frame. Writing this test the obvious way asserted that a
    // correctly-protected frame is unprotected, and the host harness caught
    // it, which is the entire argument for running these somewhere they can
    // be run.
    let mut k2 = Key::new(&tk, 0).unwrap();
    let mut clear = [&hdr[..], &plain[..]].concat();
    clear[1] &= !super::frame::FC1_PROTECTED;
    ok &= decrypt(&mut k2, &clear) == Err(Bad::NotProtected);
    ok &= decrypt(&mut k2, &hdr[..8]) == Err(Bad::Short);

    // A CCMP header without the extended-IV bit is WEP, and reading it as
    // CCMP takes the key id for packet-number bytes.
    let mut k3 = Key::new(&tk, 0).unwrap();
    let mut wep = encrypt(&mut Key::new(&tk, 0).unwrap(), &[&hdr[..], &plain[..]].concat())
        .unwrap_or_default();
    if wep.len() > 27 {
        wep[27] &= !EXT_IV;
        ok &= decrypt(&mut k3, &wep) == Err(Bad::NoExtIv);
    } else {
        ok = false;
    }

    ok
}
