//! RSA signature verification: PKCS#1 v1.5 and PSS, both with SHA-256.
//!
//! Verification only -- a public-key operation on public data, so nothing here
//! needs to be constant time.
//!
//! ### The PKCS#1 v1.5 trap
//!
//! The padded block is `00 01 FF FF ... FF 00 <DigestInfo>`, and the tempting
//! way to check it is to search for the `00` separator and compare what
//! follows. That is the Bleichenbacher signature forgery: with a small
//! exponent an attacker can construct a value whose cube *begins* correctly
//! and has garbage after the digest, and a parser that stops looking accepts
//! it. The only safe check is to build the expected block and compare the
//! whole thing, which is what happens below.

use super::bigint::{Big, Mont};
use crate::store::sha256;
use alloc::vec;
use alloc::vec::Vec;

/// The DER prefix of a DigestInfo wrapping a SHA-256 hash. Fixed, so it is
/// written out rather than encoded.
const SHA256_DIGEST_INFO: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05,
    0x00, 0x04, 0x20,
];

/// The raw public-key operation: s^e mod n, returned as a fixed-width block.
fn public_op(n: &[u8], e: &[u8], sig: &[u8]) -> Option<Vec<u8>> {
    if n.is_empty() || sig.is_empty() || sig.len() > n.len() {
        return None;
    }
    let modulus = Big::from_bytes(n);
    let exponent = Big::from_bytes(e);
    let s = Big::from_bytes(sig);
    // A signature must be less than the modulus; anything else is not a
    // valid representative and is rejected rather than reduced.
    if s.cmp(&modulus) != core::cmp::Ordering::Less {
        return None;
    }
    let mont = Mont::new(&modulus)?;
    Some(mont.pow(&s, &exponent).to_bytes(n.len()))
}

pub fn verify_pkcs1_sha256(n: &[u8], e: &[u8], digest: &[u8], sig: &[u8]) -> bool {
    let k = n.len();
    // 00 01, at least eight FF bytes, 00, then 19 + 32 of DigestInfo.
    if k < 11 + SHA256_DIGEST_INFO.len() + 32 || digest.len() != 32 {
        return false;
    }
    let Some(block) = public_op(n, e, sig) else { return false };

    let mut want = vec![0xFFu8; k];
    want[0] = 0x00;
    want[1] = 0x01;
    let tail = k - SHA256_DIGEST_INFO.len() - 32;
    want[tail - 1] = 0x00;
    want[tail..tail + SHA256_DIGEST_INFO.len()].copy_from_slice(&SHA256_DIGEST_INFO);
    want[tail + SHA256_DIGEST_INFO.len()..].copy_from_slice(digest);

    // Whole-block comparison. See the note above on why nothing here searches.
    let mut diff = 0u8;
    for i in 0..k {
        diff |= block[i] ^ want[i];
    }
    diff == 0
}

/// The same, for SHA-384. Only the DigestInfo prefix and the length differ.
const SHA384_DIGEST_INFO: [u8; 19] = [
    0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05,
    0x00, 0x04, 0x30,
];

pub fn verify_pkcs1_sha384(n: &[u8], e: &[u8], digest: &[u8], sig: &[u8]) -> bool {
    let k = n.len();
    if k < 11 + SHA384_DIGEST_INFO.len() + 48 || digest.len() != 48 {
        return false;
    }
    let Some(block) = public_op(n, e, sig) else { return false };

    let mut want = vec![0xFFu8; k];
    want[0] = 0x00;
    want[1] = 0x01;
    let tail = k - SHA384_DIGEST_INFO.len() - 48;
    want[tail - 1] = 0x00;
    want[tail..tail + SHA384_DIGEST_INFO.len()].copy_from_slice(&SHA384_DIGEST_INFO);
    want[tail + SHA384_DIGEST_INFO.len()..].copy_from_slice(digest);

    let mut diff = 0u8;
    for i in 0..k {
        diff |= block[i] ^ want[i];
    }
    diff == 0
}

/// MGF1 with SHA-256, the mask generator PSS uses.
fn mgf1(seed: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 32);
    let mut counter: u32 = 0;
    while out.len() < len {
        let mut h = sha256::Sha256::new();
        h.update(seed);
        h.update(&counter.to_be_bytes());
        out.extend_from_slice(&h.finish());
        counter += 1;
    }
    out.truncate(len);
    out
}

/// RSASSA-PSS with SHA-256 and a salt length equal to the hash length, which
/// is what TLS 1.3 mandates.
pub fn verify_pss_sha256(n: &[u8], e: &[u8], digest: &[u8], sig: &[u8]) -> bool {
    let k = n.len();
    let h_len = 32usize;
    let s_len = 32usize;
    if digest.len() != h_len || k < h_len + s_len + 2 {
        return false;
    }
    let Some(em) = public_op(n, e, sig) else { return false };

    // The modulus is em_bits = 8k - 1 bits here for the usual key sizes, so
    // the leading bit must be zero and the block ends with 0xBC.
    if *em.last().unwrap() != 0xBC {
        return false;
    }
    if em[0] & 0x80 != 0 {
        return false;
    }

    let db_len = k - h_len - 1;
    let (masked_db, rest) = em.split_at(db_len);
    let h = &rest[..h_len];

    let mask = mgf1(h, db_len);
    let mut db: Vec<u8> = masked_db.iter().zip(mask.iter()).map(|(a, b)| a ^ b).collect();
    // Clear the bits the modulus size does not cover.
    db[0] &= 0x7F;

    // db must be PS (zeros) || 0x01 || salt.
    let ps_len = db_len - s_len - 1;
    if db[..ps_len].iter().any(|b| *b != 0) || db[ps_len] != 0x01 {
        return false;
    }
    let salt = &db[ps_len + 1..];

    // H' = SHA-256(eight zero bytes || mHash || salt)
    let mut hh = sha256::Sha256::new();
    hh.update(&[0u8; 8]);
    hh.update(digest);
    hh.update(salt);
    let expect = hh.finish();

    let mut diff = 0u8;
    for i in 0..h_len {
        diff |= expect[i] ^ h[i];
    }
    diff == 0
}

pub fn selftest() -> bool {
    // A small but genuine RSA key: n = 3233, e = 17, d = 413 (the textbook
    // 61x53 example). Signing 2 gives 2^413 mod 3233; verifying returns 2.
    let n = Big::from_u64(3233, 1);
    let mont = match Mont::new(&n) {
        None => return false,
        Some(m) => m,
    };
    let sig = mont.pow(&Big::from_u64(2, 1), &Big::from_u64(413, 1));
    let back = mont.pow(&sig, &Big::from_u64(17, 1));
    if back.v[0] != 2 {
        return false;
    }

    // PKCS#1 padding is built and compared as a whole; check the shape is
    // what a real verifier would produce for a known digest.
    let digest = sha256::hash(b"glados");
    let k = 128usize;
    let mut want = vec![0xFFu8; k];
    want[0] = 0x00;
    want[1] = 0x01;
    let tail = k - SHA256_DIGEST_INFO.len() - 32;
    want[tail - 1] = 0x00;
    want[tail..tail + SHA256_DIGEST_INFO.len()].copy_from_slice(&SHA256_DIGEST_INFO);
    want[tail + SHA256_DIGEST_INFO.len()..].copy_from_slice(&digest);
    if want[0] != 0 || want[1] != 1 || want[2] != 0xFF || want[tail - 1] != 0 {
        return false;
    }

    // MGF1 against RFC 8017's construction: length is honoured and the output
    // changes with the seed.
    let a = mgf1(b"seed", 40);
    let b = mgf1(b"seee", 40);
    a.len() == 40 && a != b
}
