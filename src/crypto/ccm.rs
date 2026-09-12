//! AES-CCM and AES-CMAC: the two modes 802.11 needs and `aes.rs` did not have.
//!
//! `aes.rs` is a block cipher and RFC 3394 key wrap, which is what the WPA2
//! *handshake* needs. The data path needs more: CCMP is AES in CCM mode, so
//! without this the temporal key is derived, installed, and never used to
//! encrypt anything. `wpa2.rs` has been deriving a `tk()` nothing consumes.
//!
//! Two modes, and the second is not optional either:
//!
//!   * **CCM** (RFC 3610) is counter mode for confidentiality plus CBC-MAC for
//!     integrity, sharing one key. That sharing is the reason CCM is specified
//!     as a whole rather than left to be composed: getting the nonce formatting
//!     wrong between the two halves produces something that encrypts and
//!     authenticates and is not CCM.
//!   * **CMAC** (RFC 4493) is needed twice -- for BIP, which protects
//!     management frames under 802.11w, and for EAPOL-Key descriptor version
//!     3, which `wpa2.rs:223` currently hardcodes away by always claiming
//!     version 2.
//!
//! ### Why the constants are parameters and not literals
//!
//! CCMP always uses M=8 and L=2, so it would be tempting to bake those in. The
//! published test vectors do not: RFC 3610 varies both, and a CCM that is only
//! ever exercised at one length is one where an off-by-one in the block count
//! is invisible. `selftest` runs vectors #1 and #2, which differ in payload
//! length by one byte, specifically so a hardcoded block count fails.
//!
//! Checked at boot against RFC 3610 and RFC 4493, and `tools/wlan.py` is an
//! independent implementation of the same two algorithms for diffing against.

use super::aes::Aes;
use alloc::vec::Vec;

/// The MIC length CCMP uses, in bytes.
pub const CCMP_MIC_LEN: usize = 8;
/// The length field width CCMP uses. Nonce is then `15 - L` = 13 bytes.
pub const CCMP_L: usize = 2;

fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

/// The A_i counter block: flags, nonce, and a big-endian counter.
fn ctr_block(nonce: &[u8], l: usize, i: u64) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[0] = (l - 1) as u8;
    a[1..1 + nonce.len()].copy_from_slice(nonce);
    for k in 0..l {
        a[15 - k] = (i >> (8 * k)) as u8;
    }
    a
}

/// The CBC-MAC tag over B_0, the length-prefixed AAD, and the payload.
///
/// Separated from encryption because decryption needs exactly this and nothing
/// else -- recomputing the tag over the *recovered* plaintext is what makes
/// CCM's authentication cover the ciphertext at all.
fn mac(aes: &Aes, nonce: &[u8], aad: &[u8], plain: &[u8], m: usize, l: usize) -> [u8; 16] {
    let mut x = [0u8; 16];

    // B_0. The flags carry whether there is AAD, the MIC length and the length
    // field width, so a receiver that disagrees about any of the three
    // computes a different first block and every block after it.
    x[0] = (if aad.is_empty() { 0 } else { 0x40 }) | (((m - 2) / 2) << 3) as u8 | (l - 1) as u8;
    x[1..1 + nonce.len()].copy_from_slice(nonce);
    let n = plain.len() as u64;
    for k in 0..l {
        x[15 - k] = (n >> (8 * k)) as u8;
    }
    aes.encrypt_block(&mut x);

    if !aad.is_empty() {
        let mut b: Vec<u8> = Vec::with_capacity(aad.len() + 22);
        // Two length encodings, and 0xFF00 is the boundary. 802.11 AAD is
        // never near it, but a reader comparing this against the RFC should
        // find the RFC's rule rather than the subset this caller needs.
        if aad.len() < 0xFF00 {
            b.extend_from_slice(&(aad.len() as u16).to_be_bytes());
        } else {
            b.extend_from_slice(&[0xFF, 0xFE]);
            b.extend_from_slice(&(aad.len() as u32).to_be_bytes());
        }
        b.extend_from_slice(aad);
        while b.len() % 16 != 0 {
            b.push(0);
        }
        for chunk in b.chunks(16) {
            xor_into(&mut x, chunk);
            aes.encrypt_block(&mut x);
        }
    }

    for chunk in plain.chunks(16) {
        // Zero-padded rather than length-prefixed: the length is already in
        // B_0, so a short final block needs no further disambiguation.
        xor_into(&mut x[..chunk.len()], chunk);
        aes.encrypt_block(&mut x);
    }
    x
}

/// Encrypt and authenticate. Answers `(ciphertext, mic)`.
///
/// `None` when the key length is not one AES takes, or the nonce is not
/// `15 - l` bytes -- refused rather than padded, because a nonce silently
/// zero-extended to the right length is a nonce that collides with another.
pub fn encrypt(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    plain: &[u8],
    m: usize,
    l: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let aes = Aes::new(key)?;
    if nonce.len() != 15 - l || !(2..=16).contains(&m) || m % 2 != 0 || !(2..=8).contains(&l) {
        return None;
    }

    let tag = mac(&aes, nonce, aad, plain, m, l);

    let mut s0 = ctr_block(nonce, l, 0);
    aes.encrypt_block(&mut s0);
    let mut mic = Vec::with_capacity(m);
    for i in 0..m {
        mic.push(tag[i] ^ s0[i]);
    }

    let mut out = Vec::with_capacity(plain.len());
    for (i, chunk) in plain.chunks(16).enumerate() {
        let mut s = ctr_block(nonce, l, i as u64 + 1);
        aes.encrypt_block(&mut s);
        for (k, b) in chunk.iter().enumerate() {
            out.push(b ^ s[k]);
        }
    }
    Some((out, mic))
}

/// Decrypt and verify. `None` when the MIC does not match.
///
/// The plaintext is discarded on a MIC failure rather than returned with a
/// flag, because a caller that can reach the bytes will eventually use them.
/// That is the whole point of authenticated encryption and it is the exact
/// mistake `tls.rs` records the shape of: reporting a verdict and handing over
/// the data anyway.
pub fn decrypt(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    cipher: &[u8],
    mic: &[u8],
    l: usize,
) -> Option<Vec<u8>> {
    let aes = Aes::new(key)?;
    let m = mic.len();
    if nonce.len() != 15 - l || !(2..=16).contains(&m) || m % 2 != 0 {
        return None;
    }

    let mut plain = Vec::with_capacity(cipher.len());
    for (i, chunk) in cipher.chunks(16).enumerate() {
        let mut s = ctr_block(nonce, l, i as u64 + 1);
        aes.encrypt_block(&mut s);
        for (k, b) in chunk.iter().enumerate() {
            plain.push(b ^ s[k]);
        }
    }

    let tag = mac(&aes, nonce, aad, &plain, m, l);
    let mut s0 = ctr_block(nonce, l, 0);
    aes.encrypt_block(&mut s0);

    // Accumulate the difference rather than returning early, so the time taken
    // does not say which byte was wrong. The same idiom `wpa2::verify_mic`
    // uses, for the same reason.
    let mut diff = 0u8;
    for i in 0..m {
        diff |= (tag[i] ^ s0[i]) ^ mic[i];
    }
    if diff != 0 {
        return None;
    }
    Some(plain)
}

/// AES-CMAC (RFC 4493).
pub fn cmac(key: &[u8], msg: &[u8]) -> Option<[u8; 16]> {
    let aes = Aes::new(key)?;

    // The two subkeys, each a doubling in GF(2^128). The 0x87 is the
    // reduction polynomial; getting it wrong gives a MAC that is perfectly
    // consistent with itself and matches nobody else's.
    let dbl = |b: [u8; 16]| -> [u8; 16] {
        let msb = b[0] & 0x80;
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = (b[i] << 1) | if i + 1 < 16 { b[i + 1] >> 7 } else { 0 };
        }
        if msb != 0 {
            o[15] ^= 0x87;
        }
        o
    };

    let mut l = [0u8; 16];
    aes.encrypt_block(&mut l);
    let k1 = dbl(l);
    let k2 = dbl(k1);

    let whole = !msg.is_empty() && msg.len() % 16 == 0;
    let body_len = if whole { msg.len() - 16 } else { msg.len() - msg.len() % 16 };

    let mut last = [0u8; 16];
    if whole {
        last.copy_from_slice(&msg[body_len..]);
        xor_into(&mut last, &k1);
    } else {
        let tail = &msg[body_len..];
        last[..tail.len()].copy_from_slice(tail);
        last[tail.len()] = 0x80;
        xor_into(&mut last, &k2);
    }

    let mut x = [0u8; 16];
    for chunk in msg[..body_len].chunks(16) {
        xor_into(&mut x, chunk);
        aes.encrypt_block(&mut x);
    }
    xor_into(&mut x, &last);
    aes.encrypt_block(&mut x);
    Some(x)
}

pub fn selftest() -> bool {
    let mut ok = true;

    // RFC 3610 packet vector #1.
    let key = [
        0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce,
        0xcf,
    ];
    let nonce = [
        0x00, 0x00, 0x00, 0x03, 0x02, 0x01, 0x00, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5,
    ];
    let aad = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let plain: [u8; 23] = [
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
    ];
    let want_c: [u8; 23] = [
        0x58, 0x8c, 0x97, 0x9a, 0x61, 0xc6, 0x63, 0xd2, 0xf0, 0x66, 0xd0, 0xc2, 0xc0, 0xf9, 0x89,
        0x80, 0x6d, 0x5f, 0x6b, 0x61, 0xda, 0xc3, 0x84,
    ];
    let want_m = [0x17, 0xe8, 0xd1, 0x2c, 0xfd, 0xf9, 0x26, 0xe0];

    match encrypt(&key, &nonce, &aad, &plain, 8, 2) {
        Some((c, m)) => {
            ok &= c == want_c;
            ok &= m == want_m;
            // Round-trip, and then the thing that matters more: a single
            // flipped ciphertext bit must not decrypt. A CCM that encrypts
            // correctly and accepts anything is the failure this guards.
            ok &= decrypt(&key, &nonce, &aad, &c, &m, 2).as_deref() == Some(&plain[..]);
            let mut bad = c.clone();
            bad[0] ^= 1;
            ok &= decrypt(&key, &nonce, &aad, &bad, &m, 2).is_none();
            let mut badmic = m.clone();
            badmic[7] ^= 1;
            ok &= decrypt(&key, &nonce, &aad, &c, &badmic, 2).is_none();
            // AAD is authenticated but not encrypted, so a changed AAD must
            // also fail -- which a CCM that simply ignored the AAD would pass
            // every other check above.
            let mut badaad = aad;
            badaad[0] ^= 1;
            ok &= decrypt(&key, &nonce, &badaad, &c, &m, 2).is_none();
        }
        None => ok = false,
    }

    // RFC 3610 packet vector #2: one byte longer, so a hardcoded block count
    // that passes #1 fails here.
    let nonce2 = [
        0x00, 0x00, 0x00, 0x04, 0x03, 0x02, 0x01, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5,
    ];
    let plain2: [u8; 24] = [
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];
    let want_c2: [u8; 24] = [
        0x72, 0xc9, 0x1a, 0x36, 0xe1, 0x35, 0xf8, 0xcf, 0x29, 0x1c, 0xa8, 0x94, 0x08, 0x5c, 0x87,
        0xe3, 0xcc, 0x15, 0xc4, 0x39, 0xc9, 0xe4, 0x3a, 0x3b,
    ];
    let want_m2 = [0xa0, 0x91, 0xd5, 0x6e, 0x10, 0x40, 0x09, 0x16];
    match encrypt(&key, &nonce2, &aad, &plain2, 8, 2) {
        Some((c, m)) => {
            ok &= c == want_c2;
            ok &= m == want_m2;
        }
        None => ok = false,
    }

    // A nonce of the wrong length is refused rather than padded.
    ok &= encrypt(&key, &nonce[..12], &aad, &plain, 8, 2).is_none();

    // RFC 4493 AES-CMAC. The empty case is the one most often got wrong,
    // because it takes the padding branch with nothing to pad.
    let ck = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
        0x3c,
    ];
    ok &= cmac(&ck, &[])
        == Some([
            0xbb, 0x1d, 0x69, 0x29, 0xe9, 0x59, 0x37, 0x28, 0x7f, 0xa3, 0x7d, 0x12, 0x9b, 0x75,
            0x67, 0x46,
        ]);
    let m16 = [
        0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17,
        0x2a,
    ];
    ok &= cmac(&ck, &m16)
        == Some([
            0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44, 0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a,
            0x28, 0x7c,
        ]);
    // 40 bytes: a whole number of blocks plus a partial one, so it exercises
    // the k2 branch after a real body rather than instead of one.
    let mut m40 = [0u8; 40];
    m40[..16].copy_from_slice(&m16);
    m40[16..32].copy_from_slice(&[
        0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c, 0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e,
        0x51,
    ]);
    m40[32..].copy_from_slice(&[0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11]);
    ok &= cmac(&ck, &m40)
        == Some([
            0xdf, 0xa6, 0x67, 0x47, 0xde, 0x9a, 0xe6, 0x30, 0x30, 0xca, 0x32, 0x61, 0x14, 0x97,
            0xc8, 0x27,
        ]);

    ok
}
