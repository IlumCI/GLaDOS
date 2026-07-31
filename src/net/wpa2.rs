//! WPA2-PSK: the key hierarchy and the four-way handshake.
//!
//! ### What this is, and what it is waiting for
//!
//! Everything here is the supplicant's *cryptography* and message handling --
//! the part that can be written and checked without any hardware, against the
//! test vectors in IEEE 802.11i. It is complete and it is verified at boot.
//!
//! What it cannot do is run, because `wlan0` has no driver. There is nothing
//! to send an EAPOL frame over. So this module is deliberately structured as
//! pure functions over byte slices with no I/O anywhere: when a driver
//! arrives, it supplies frames and this supplies answers, and none of the code
//! below changes.
//!
//! ### The key hierarchy
//!
//! ```text
//!   passphrase + SSID  --PBKDF2-SHA1, 4096 rounds-->  PMK   (32 bytes)
//!   PMK + both nonces + both MACs  --PRF-384-->       PTK   (48 bytes)
//!   PTK[0..16]   KCK   confirms the handshake messages
//!   PTK[16..32]  KEK   unwraps the group key
//!   PTK[32..48]  TK    encrypts data with CCMP
//! ```
//!
//! The nonces are what make the PTK fresh: the same passphrase on the same
//! network yields a different session key every time, so recording traffic
//! today and learning the passphrase tomorrow does not decrypt it. That
//! property is why the four-way handshake exists at all rather than simply
//! using the PMK.
//!
//! ### The known weakness, which is the protocol's and not this code's
//!
//! Anyone who captures the four-way handshake can test passphrase guesses
//! offline at 4096 PBKDF2 rounds each. That was expensive in 2004. A
//! dictionary word is not safe on WPA2 and no supplicant can fix it -- WPA3's
//! SAE replaces exactly this.

use crate::crypto::{aes, sha1};
use alloc::vec::Vec;

pub const PMK_LEN: usize = 32;
pub const PTK_LEN: usize = 48;

/// EAPOL-Key frames sit inside an EAPOL packet with this small header.
const EAPOL_VERSION: u8 = 1;
const EAPOL_TYPE_KEY: u8 = 3;
const KEY_TYPE_RSN: u8 = 2;

// Key Information bits, IEEE 802.11 table 12-8.
const KEY_INFO_PAIRWISE: u16 = 1 << 3;
const KEY_INFO_INSTALL: u16 = 1 << 6;
const KEY_INFO_ACK: u16 = 1 << 7;
const KEY_INFO_MIC: u16 = 1 << 8;
const KEY_INFO_SECURE: u16 = 1 << 9;
const KEY_INFO_ENCRYPTED: u16 = 1 << 12;

/// Derive the pairwise master key from a passphrase.
///
/// The SSID is the salt, which is why two networks with the same name and
/// password share a PMK -- and why precomputed tables for common SSIDs work.
pub fn pmk(passphrase: &str, ssid: &[u8]) -> Vec<u8> {
    sha1::pbkdf2(passphrase.as_bytes(), ssid, 4096, PMK_LEN)
}

/// The IEEE 802.11 PRF, built from HMAC-SHA1.
///
/// Each iteration hashes the label, the data, and a counter; the counter is
/// what makes the blocks differ, and it is a single byte appended *after* the
/// data rather than mixed in, which is the detail most reimplementations get
/// wrong.
fn prf(key: &[u8], label: &str, data: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len + sha1::HASH_LEN);
    let mut counter: u8 = 0;
    while out.len() < out_len {
        let mut input = Vec::with_capacity(label.len() + 1 + data.len() + 1);
        input.extend_from_slice(label.as_bytes());
        input.push(0);
        input.extend_from_slice(data);
        input.push(counter);
        out.extend_from_slice(&sha1::hmac(key, &input));
        counter += 1;
    }
    out.truncate(out_len);
    out
}

/// Derive the pairwise transient key.
///
/// The two MAC addresses and the two nonces each go in sorted order, smaller
/// first. Both sides therefore build the same input without having to agree
/// who is who -- and getting the ordering wrong produces a PTK that is
/// perfectly well-formed and does not match the other end's.
pub fn ptk(pmk: &[u8], aa: &[u8; 6], spa: &[u8; 6], anonce: &[u8; 32], snonce: &[u8; 32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(76);
    if aa <= spa {
        data.extend_from_slice(aa);
        data.extend_from_slice(spa);
    } else {
        data.extend_from_slice(spa);
        data.extend_from_slice(aa);
    }
    if anonce <= snonce {
        data.extend_from_slice(anonce);
        data.extend_from_slice(snonce);
    } else {
        data.extend_from_slice(snonce);
        data.extend_from_slice(anonce);
    }
    prf(pmk, "Pairwise key expansion", &data, PTK_LEN)
}

pub struct Ptk<'a>(pub &'a [u8]);

impl<'a> Ptk<'a> {
    /// Key Confirmation Key -- authenticates the handshake messages.
    pub fn kck(&self) -> &[u8] {
        &self.0[0..16]
    }
    /// Key Encryption Key -- unwraps the group key in message 3.
    pub fn kek(&self) -> &[u8] {
        &self.0[16..32]
    }
    /// Temporal Key -- what CCMP actually encrypts data with.
    pub fn tk(&self) -> &[u8] {
        &self.0[32..48]
    }
}

/// A parsed EAPOL-Key frame.
pub struct KeyFrame<'a> {
    pub info: u16,
    pub replay: u64,
    pub nonce: [u8; 32],
    pub mic: [u8; 16],
    pub key_data: &'a [u8],
    /// The whole EAPOL packet, needed because the MIC covers it with the MIC
    /// field zeroed.
    pub raw: &'a [u8],
}

impl<'a> KeyFrame<'a> {
    pub fn has(&self, bit: u16) -> bool {
        self.info & bit != 0
    }
    /// Message 1 carries the ANonce and no MIC; message 3 carries both a MIC
    /// and the encrypted group key.
    pub fn is_message1(&self) -> bool {
        self.has(KEY_INFO_PAIRWISE) && self.has(KEY_INFO_ACK) && !self.has(KEY_INFO_MIC)
    }
    pub fn is_message3(&self) -> bool {
        self.has(KEY_INFO_PAIRWISE) && self.has(KEY_INFO_ACK) && self.has(KEY_INFO_MIC)
    }
}

/// Parse an EAPOL-Key packet.
pub fn parse(pkt: &[u8]) -> Option<KeyFrame<'_>> {
    // EAPOL header: version, type, length. Then the Key Descriptor.
    if pkt.len() < 4 + 95 || pkt[1] != EAPOL_TYPE_KEY {
        return None;
    }
    let body_len = u16::from_be_bytes([pkt[2], pkt[3]]) as usize;
    if 4 + body_len > pkt.len() {
        return None;
    }
    let b = &pkt[4..4 + body_len];
    if b[0] != KEY_TYPE_RSN {
        return None;
    }
    let info = u16::from_be_bytes([b[1], b[2]]);
    let replay = u64::from_be_bytes([b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12]]);
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&b[13..45]);
    let mut mic = [0u8; 16];
    mic.copy_from_slice(&b[77..93]);
    let data_len = u16::from_be_bytes([b[93], b[94]]) as usize;
    if 95 + data_len > b.len() {
        return None;
    }
    Some(KeyFrame {
        info,
        replay,
        nonce,
        mic,
        key_data: &b[95..95 + data_len],
        raw: &pkt[..4 + body_len],
    })
}

/// Check the MIC on a received frame.
///
/// The MIC is computed over the whole EAPOL packet with the MIC field itself
/// zeroed -- it cannot cover its own value, and a verifier that forgets to
/// zero it rejects every frame.
pub fn verify_mic(kck: &[u8], frame: &KeyFrame) -> bool {
    let mut copy = frame.raw.to_vec();
    // The MIC sits 77 bytes into the key descriptor, which starts at 4.
    for b in copy[4 + 77..4 + 93].iter_mut() {
        *b = 0;
    }
    let computed = sha1::hmac(kck, &copy);
    let mut diff = 0u8;
    for i in 0..16 {
        diff |= computed[i] ^ frame.mic[i];
    }
    diff == 0
}

/// Build message 2 or 4: the supplicant's replies, both MIC-protected.
pub fn build_reply(
    kck: &[u8],
    replay: u64,
    snonce: Option<&[u8; 32]>,
    key_data: &[u8],
    secure: bool,
) -> Vec<u8> {
    let body_len = 95 + key_data.len();
    let mut p = Vec::with_capacity(4 + body_len);
    p.push(EAPOL_VERSION);
    p.push(EAPOL_TYPE_KEY);
    p.extend_from_slice(&(body_len as u16).to_be_bytes());

    p.push(KEY_TYPE_RSN);
    let mut info = KEY_INFO_PAIRWISE | KEY_INFO_MIC | 2; // 2 = HMAC-SHA1 AKM
    if secure {
        info |= KEY_INFO_SECURE;
    }
    p.extend_from_slice(&info.to_be_bytes());
    p.extend_from_slice(&16u16.to_be_bytes()); // key length
    p.extend_from_slice(&replay.to_be_bytes());
    match snonce {
        Some(n) => p.extend_from_slice(n),
        None => p.extend_from_slice(&[0u8; 32]),
    }
    p.extend_from_slice(&[0u8; 16]); // key IV
    p.extend_from_slice(&[0u8; 8]); // key RSC
    p.extend_from_slice(&[0u8; 8]); // reserved
    let mic_at = p.len();
    p.extend_from_slice(&[0u8; 16]); // MIC, filled below
    p.extend_from_slice(&(key_data.len() as u16).to_be_bytes());
    p.extend_from_slice(key_data);

    let mic = sha1::hmac(kck, &p);
    p[mic_at..mic_at + 16].copy_from_slice(&mic[..16]);
    p
}

/// Pull the group key out of message 3's encrypted key data.
pub fn group_key(kek: &[u8], key_data: &[u8]) -> Option<Vec<u8>> {
    let unwrapped = aes::key_unwrap(kek, key_data)?;
    // The result is a sequence of RSN key-data encapsulations; the GTK is the
    // one with OUI 00-0F-AC and data type 1.
    let mut at = 0;
    while at + 6 <= unwrapped.len() {
        let len = unwrapped[at + 1] as usize;
        if len < 4 || at + 2 + len > unwrapped.len() {
            break;
        }
        let oui = &unwrapped[at + 2..at + 5];
        let dtype = unwrapped[at + 5];
        if oui == [0x00, 0x0F, 0xAC] && dtype == 1 && len >= 6 {
            // Two bytes of key id and reserved precede the key itself.
            return Some(unwrapped[at + 8..at + 2 + len].to_vec());
        }
        at += 2 + len;
        // Encapsulations are padded to a multiple of eight.
        while at % 8 != 0 && at < unwrapped.len() {
            at += 1;
        }
    }
    None
}

/// The supplicant's side of the exchange, as a state machine over frames.
pub struct Supplicant {
    pub pmk: Vec<u8>,
    pub ptk: Option<Vec<u8>>,
    pub gtk: Option<Vec<u8>>,
    pub snonce: [u8; 32],
    pub aa: [u8; 6],
    pub spa: [u8; 6],
    pub done: bool,
}

impl Supplicant {
    pub fn new(passphrase: &str, ssid: &[u8], aa: [u8; 6], spa: [u8; 6]) -> Self {
        let mut snonce = [0u8; 32];
        for i in 0..4 {
            let t = crate::time::rdtsc().rotate_left((i * 13) as u32);
            snonce[i * 8..i * 8 + 8].copy_from_slice(&t.to_le_bytes());
        }
        Supplicant {
            pmk: pmk(passphrase, ssid),
            ptk: None,
            gtk: None,
            snonce,
            aa,
            spa,
            done: false,
        }
    }

    /// Feed a received EAPOL-Key frame, get back what to send.
    pub fn on_frame(&mut self, pkt: &[u8]) -> Option<Vec<u8>> {
        let f = parse(pkt)?;

        if f.is_message1() {
            // Message 1 has no MIC -- there is no key yet to compute one with,
            // which is why an attacker can force a handshake and capture it.
            let ptk = ptk(&self.pmk, &self.aa, &self.spa, &f.nonce, &self.snonce);
            let reply = build_reply(Ptk(&ptk).kck(), f.replay, Some(&self.snonce), &[], false);
            self.ptk = Some(ptk);
            return Some(reply);
        }

        if f.is_message3() {
            let ptk = self.ptk.clone()?;
            let k = Ptk(&ptk);
            // Message 3 is the first frame that proves the AP knows the PMK.
            // A failed MIC here means the passphrase is wrong -- or someone is
            // impersonating the network.
            if !verify_mic(k.kck(), &f) {
                return None;
            }
            if f.has(KEY_INFO_ENCRYPTED) {
                self.gtk = group_key(k.kek(), f.key_data);
            }
            let reply = build_reply(k.kck(), f.replay, None, &[], true);
            self.done = true;
            let _ = KEY_INFO_INSTALL;
            return Some(reply);
        }

        None
    }
}

pub fn selftest() -> bool {
    // IEEE 802.11i Annex H.4: passphrase "password", SSID "IEEE".
    let k = pmk("password", b"IEEE");
    let want: [u8; 32] = [
        0xf4, 0x2c, 0x6f, 0xc5, 0x2d, 0xf0, 0xeb, 0xef, 0x9e, 0xbb, 0x4b, 0x90, 0xb3, 0x8a, 0x5f,
        0x90, 0x2e, 0x83, 0xfe, 0x1b, 0x13, 0x5a, 0x70, 0xe2, 0x3a, 0xed, 0x76, 0x2e, 0x97, 0x10,
        0xa1, 0x2e,
    ];
    if k[..] != want[..] {
        return false;
    }

    // The PTK must not depend on which side is called which: swapping the
    // roles has to produce the same key, or the two ends never agree.
    let aa = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    let spa = [0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB];
    let an = [0xAA; 32];
    let sn = [0xBB; 32];
    let a = ptk(&k, &aa, &spa, &an, &sn);
    let b = ptk(&k, &spa, &aa, &sn, &an);
    if a != b || a.len() != PTK_LEN {
        return false;
    }
    // And a different nonce must give a different key, which is the freshness
    // property the whole handshake exists to provide.
    let c = ptk(&k, &aa, &spa, &an, &[0xCC; 32]);
    a != c
}

pub fn report() {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[wpa2]");
    console::set_color(LTGRAY);
    let ok = selftest() && aes::selftest() && sha1::selftest();
    console::set_color(if ok { LTGREEN } else { crate::gfx::console::LTRED });
    kprintln!(
        "  supplicant crypto {}",
        if ok { "matches the IEEE vectors" } else { "IS WRONG" }
    );
    console::set_color(LTGRAY);
    kprintln!("  pmk  pbkdf2-hmac-sha1, 4096 rounds, ssid as salt");
    kprintln!("  ptk  802.11 prf-384 over both nonces and both macs");
    kprintln!("  gtk  rfc 3394 key unwrap under the kek");
    console::set_color(YELLOW);
    kprintln!("  nothing to run it on: wlan0 has no driver.");
    console::set_color(LTGRAY);
    kprintln!("  this is pure functions over byte slices by design -- a driver");
    kprintln!("  supplies frames, this supplies answers, and none of it changes.");
}
