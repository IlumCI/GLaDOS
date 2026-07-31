//! HMAC-SHA256, HKDF, and the TLS 1.3 label scheme on top of them.
//!
//! All three are short and mechanical, which is exactly why they are worth
//! writing rather than trusting: there is nothing here that a careful reader
//! cannot check against RFC 2104, RFC 5869 and RFC 8446 section 7.1 in an
//! afternoon.
//!
//! The key schedule is the part where a mistake is silent. Every secret in TLS
//! 1.3 is derived from the one before it, so a wrong label or a transcript
//! hashed at the wrong moment produces keys that are perfectly well-formed and
//! simply do not match the server's -- which presents as a decryption failure
//! several messages later, with nothing to say where it went wrong. The
//! transcript is therefore hashed at exactly the points the RFC names, and the
//! labels are written out rather than constructed.

use crate::store::sha256::Sha256;
use alloc::vec::Vec;

pub const HASH_LEN: usize = 32;
const BLOCK_LEN: usize = 64;

pub fn hmac(key: &[u8], data: &[u8]) -> [u8; HASH_LEN] {
    // A key longer than the block is replaced by its hash; a shorter one is
    // zero-padded. Both are the RFC 2104 rule and both matter here, because
    // HKDF passes keys of either shape.
    let mut k = [0u8; BLOCK_LEN];
    if key.len() > BLOCK_LEN {
        k[..HASH_LEN].copy_from_slice(&crate::store::sha256::hash(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; BLOCK_LEN];
    let mut opad = [0x5Cu8; BLOCK_LEN];
    for i in 0..BLOCK_LEN {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(data);
    let inner = inner.finish();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner);
    outer.finish()
}

pub fn extract(salt: &[u8], ikm: &[u8]) -> [u8; HASH_LEN] {
    hmac(salt, ikm)
}

pub fn expand(prk: &[u8], info: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut t: Vec<u8> = Vec::new();
    let mut counter: u8 = 1;
    while out.len() < len {
        let mut input = Vec::with_capacity(t.len() + info.len() + 1);
        input.extend_from_slice(&t);
        input.extend_from_slice(info);
        input.push(counter);
        let block = hmac(prk, &input);
        t = block.to_vec();
        out.extend_from_slice(&block);
        counter += 1;
    }
    out.truncate(len);
    out
}

/// HKDF-Expand-Label from RFC 8446.
///
/// The "tls13 " prefix is what stops a secret derived here from ever matching
/// one derived by another protocol using the same HKDF and the same label.
pub fn expand_label(secret: &[u8], label: &str, context: &[u8], len: usize) -> Vec<u8> {
    let mut info = Vec::with_capacity(4 + 6 + label.len() + context.len());
    info.extend_from_slice(&(len as u16).to_be_bytes());
    info.push((6 + label.len()) as u8);
    info.extend_from_slice(b"tls13 ");
    info.extend_from_slice(label.as_bytes());
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    expand(secret, &info, len)
}

/// Derive-Secret: Expand-Label over a transcript hash.
pub fn derive_secret(secret: &[u8], label: &str, transcript: &[u8; HASH_LEN]) -> [u8; HASH_LEN] {
    let v = expand_label(secret, label, transcript, HASH_LEN);
    let mut out = [0u8; HASH_LEN];
    out.copy_from_slice(&v);
    out
}

pub fn selftest() -> bool {
    // RFC 4231 test case 2: key "Jefe", data "what do ya want for nothing?".
    let mac = hmac(b"Jefe", b"what do ya want for nothing?");
    let want = [
        0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e, 0x6a, 0x04, 0x24, 0x26, 0x08, 0x95, 0x75,
        0xc7, 0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83, 0x9d, 0xec, 0x58, 0xb9, 0x64, 0xec,
        0x38, 0x43,
    ];
    if mac != want {
        return false;
    }

    // RFC 5869 test case 1.
    let ikm = [0x0bu8; 22];
    let salt: Vec<u8> = (0..13u8).collect();
    let info: [u8; 10] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9];
    let prk = extract(&salt, &ikm);
    let okm = expand(&prk, &info, 42);
    let want_okm = [
        0x3c, 0xb2, 0x5f, 0x25, 0xfa, 0xac, 0xd5, 0x7a, 0x90, 0x43, 0x4f, 0x64, 0xd0, 0x36, 0x2f,
        0x2a, 0x2d, 0x2d, 0x0a, 0x90, 0xcf, 0x1a, 0x5a, 0x4c, 0x5d, 0xb0, 0x2d, 0x56, 0xec, 0xc4,
        0xc5, 0xbf, 0x34, 0x00, 0x72, 0x08, 0xd5, 0xb8, 0x87, 0x18, 0x58, 0x65,
    ];
    okm[..] == want_okm[..]
}
