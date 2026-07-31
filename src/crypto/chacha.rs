//! ChaCha20-Poly1305, RFC 8439.
//!
//! Chosen over AES-GCM for one reason: it can be written correctly here.
//! A software AES is a table lookup indexed by secret data, which leaks the
//! key through the cache to anything that can measure it, and the constant-time
//! alternatives are either bitsliced (large, fiddly) or AES-NI (fine on this
//! CPU, but then the code only runs on CPUs that have it). ChaCha20 is adds,
//! xors and rotates on 32-bit words -- no tables, no data-dependent branches,
//! constant time by construction.
//!
//! Poly1305 needs 130-bit arithmetic, done here in five 26-bit limbs so that
//! every product fits in a u64 without a carry chain that depends on the data.

use alloc::vec::Vec;

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

#[inline(always)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

fn block(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN]) -> [u8; 64] {
    let mut s = [0u32; 16];
    // "expand 32-byte k", as four little-endian words.
    s[0] = 0x61707865;
    s[1] = 0x3320646e;
    s[2] = 0x79622d32;
    s[3] = 0x6b206574;
    for i in 0..8 {
        s[4 + i] = u32::from_le_bytes([
            key[i * 4],
            key[i * 4 + 1],
            key[i * 4 + 2],
            key[i * 4 + 3],
        ]);
    }
    s[12] = counter;
    for i in 0..3 {
        s[13 + i] = u32::from_le_bytes([
            nonce[i * 4],
            nonce[i * 4 + 1],
            nonce[i * 4 + 2],
            nonce[i * 4 + 3],
        ]);
    }

    let mut w = s;
    // Ten double-rounds: four columns then four diagonals, twenty in total.
    for _ in 0..10 {
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }

    let mut out = [0u8; 64];
    for i in 0..16 {
        let v = w[i].wrapping_add(s[i]);
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// XOR a buffer with the keystream, starting at `counter`.
///
/// Encryption and decryption are the same operation, which is why there is one
/// function -- and also why a nonce must never repeat under one key. Two
/// messages encrypted with the same counter and nonce differ by the XOR of
/// their plaintexts, and the keystream is gone.
pub fn apply(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN], data: &mut [u8]) {
    let mut c = counter;
    for chunk in data.chunks_mut(64) {
        let ks = block(key, c, nonce);
        for (b, k) in chunk.iter_mut().zip(ks.iter()) {
            *b ^= *k;
        }
        c = c.wrapping_add(1);
    }
}

// --- Poly1305 -----------------------------------------------------------

struct Poly1305 {
    r: [u64; 5],
    h: [u64; 5],
    pad: [u32; 4],
    buffer: [u8; 16],
    used: usize,
}

impl Poly1305 {
    fn new(key: &[u8; 32]) -> Self {
        let le = |i: usize| u32::from_le_bytes([key[i], key[i + 1], key[i + 2], key[i + 3]]) as u64;
        // "Clamping": specific bits of r are cleared so that the multiply
        // cannot carry beyond what the limb arithmetic below can hold.
        let t0 = le(0);
        let t1 = le(4);
        let t2 = le(8);
        let t3 = le(12);
        Poly1305 {
            r: [
                t0 & 0x3ffffff,
                ((t0 >> 26) | (t1 << 6)) & 0x3ffff03,
                ((t1 >> 20) | (t2 << 12)) & 0x3ffc0ff,
                ((t2 >> 14) | (t3 << 18)) & 0x3f03fff,
                (t3 >> 8) & 0x00fffff,
            ],
            h: [0; 5],
            pad: [le(16) as u32, le(20) as u32, le(24) as u32, le(28) as u32],
            buffer: [0; 16],
            used: 0,
        }
    }

    fn block(&mut self, m: &[u8], final_block: bool) {
        let hibit = if final_block { 0 } else { 1u64 << 24 };
        let le = |i: usize| u32::from_le_bytes([m[i], m[i + 1], m[i + 2], m[i + 3]]) as u64;
        let t0 = le(0);
        let t1 = le(4);
        let t2 = le(8);
        let t3 = le(12);

        self.h[0] += t0 & 0x3ffffff;
        self.h[1] += ((t0 >> 26) | (t1 << 6)) & 0x3ffffff;
        self.h[2] += ((t1 >> 20) | (t2 << 12)) & 0x3ffffff;
        self.h[3] += ((t2 >> 14) | (t3 << 18)) & 0x3ffffff;
        self.h[4] += (t3 >> 8) | hibit;

        let r = self.r;
        // 5*r_i shows up because reduction mod 2^130-5 folds the top back in.
        let s: [u64; 4] = [r[1] * 5, r[2] * 5, r[3] * 5, r[4] * 5];
        let h = self.h;

        let d0 = h[0] * r[0] + h[1] * s[3] + h[2] * s[2] + h[3] * s[1] + h[4] * s[0];
        let d1 = h[0] * r[1] + h[1] * r[0] + h[2] * s[3] + h[3] * s[2] + h[4] * s[1];
        let d2 = h[0] * r[2] + h[1] * r[1] + h[2] * r[0] + h[3] * s[3] + h[4] * s[2];
        let d3 = h[0] * r[3] + h[1] * r[2] + h[2] * r[1] + h[3] * r[0] + h[4] * s[3];
        let d4 = h[0] * r[4] + h[1] * r[3] + h[2] * r[2] + h[3] * r[1] + h[4] * r[0];

        let mut c;
        c = d0 >> 26;
        self.h[0] = d0 & 0x3ffffff;
        let d1 = d1 + c;
        c = d1 >> 26;
        self.h[1] = d1 & 0x3ffffff;
        let d2 = d2 + c;
        c = d2 >> 26;
        self.h[2] = d2 & 0x3ffffff;
        let d3 = d3 + c;
        c = d3 >> 26;
        self.h[3] = d3 & 0x3ffffff;
        let d4 = d4 + c;
        c = d4 >> 26;
        self.h[4] = d4 & 0x3ffffff;
        self.h[0] += c * 5;
        c = self.h[0] >> 26;
        self.h[0] &= 0x3ffffff;
        self.h[1] += c;
    }

    fn update(&mut self, mut data: &[u8]) {
        if self.used > 0 {
            let take = core::cmp::min(16 - self.used, data.len());
            self.buffer[self.used..self.used + take].copy_from_slice(&data[..take]);
            self.used += take;
            data = &data[take..];
            if self.used == 16 {
                let b = self.buffer;
                self.block(&b, false);
                self.used = 0;
            }
        }
        while data.len() >= 16 {
            let (b, rest) = data.split_at(16);
            self.block(b, false);
            data = rest;
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.used = data.len();
        }
    }

    fn finish(mut self) -> [u8; TAG_LEN] {
        if self.used > 0 {
            let n = self.used;
            self.buffer[n] = 1;
            for b in self.buffer[n + 1..].iter_mut() {
                *b = 0;
            }
            let b = self.buffer;
            self.block(&b, true);
        }

        let mut c = self.h[1] >> 26;
        self.h[1] &= 0x3ffffff;
        self.h[2] += c;
        c = self.h[2] >> 26;
        self.h[2] &= 0x3ffffff;
        self.h[3] += c;
        c = self.h[3] >> 26;
        self.h[3] &= 0x3ffffff;
        self.h[4] += c;
        c = self.h[4] >> 26;
        self.h[4] &= 0x3ffffff;
        self.h[0] += c * 5;
        c = self.h[0] >> 26;
        self.h[0] &= 0x3ffffff;
        self.h[1] += c;

        // Subtract 2^130-5 and keep the result only if it did not borrow.
        // Written branchlessly: which of the two is correct depends on the
        // secret accumulator, so a branch here would leak it.
        let mut g = [0u64; 5];
        g[0] = self.h[0] + 5;
        c = g[0] >> 26;
        g[0] &= 0x3ffffff;
        for i in 1..4 {
            g[i] = self.h[i] + c;
            c = g[i] >> 26;
            g[i] &= 0x3ffffff;
        }
        g[4] = self.h[4] + c;
        let g4 = g[4].wrapping_sub(1 << 26);

        let mask = if (g4 >> 63) == 0 { u64::MAX } else { 0 };
        g[4] = g4;
        for i in 0..5 {
            self.h[i] = (self.h[i] & !mask) | (g[i] & mask);
        }

        let h0 = (self.h[0] | (self.h[1] << 26)) & 0xffffffff;
        let h1 = ((self.h[1] >> 6) | (self.h[2] << 20)) & 0xffffffff;
        let h2 = ((self.h[2] >> 12) | (self.h[3] << 14)) & 0xffffffff;
        let h3 = ((self.h[3] >> 18) | (self.h[4] << 8)) & 0xffffffff;

        let mut f = h0 + self.pad[0] as u64;
        let o0 = f as u32;
        f = h1 + self.pad[1] as u64 + (f >> 32);
        let o1 = f as u32;
        f = h2 + self.pad[2] as u64 + (f >> 32);
        let o2 = f as u32;
        f = h3 + self.pad[3] as u64 + (f >> 32);
        let o3 = f as u32;

        let mut tag = [0u8; TAG_LEN];
        tag[0..4].copy_from_slice(&o0.to_le_bytes());
        tag[4..8].copy_from_slice(&o1.to_le_bytes());
        tag[8..12].copy_from_slice(&o2.to_le_bytes());
        tag[12..16].copy_from_slice(&o3.to_le_bytes());
        tag
    }
}

fn pad16(n: usize) -> usize {
    if n % 16 == 0 {
        0
    } else {
        16 - (n % 16)
    }
}

fn tag_for(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[u8], ct: &[u8]) -> [u8; TAG_LEN] {
    // The one-time Poly1305 key is the first 32 bytes of ChaCha block zero.
    // Block one onward is the keystream, which is why encryption starts at 1.
    let b0 = block(key, 0, nonce);
    let mut pk = [0u8; 32];
    pk.copy_from_slice(&b0[..32]);

    let mut p = Poly1305::new(&pk);
    p.update(aad);
    p.update(&[0u8; 16][..pad16(aad.len())]);
    p.update(ct);
    p.update(&[0u8; 16][..pad16(ct.len())]);
    p.update(&(aad.len() as u64).to_le_bytes());
    p.update(&(ct.len() as u64).to_le_bytes());
    p.finish()
}

pub fn seal(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
    let mut out = plaintext.to_vec();
    apply(key, 1, nonce, &mut out);
    let tag = tag_for(key, nonce, aad, &out);
    out.extend_from_slice(&tag);
    out
}

/// Returns `None` if the tag does not match, and never returns plaintext in
/// that case -- the whole point of an AEAD is that unauthenticated bytes are
/// not handed upward, however tempting it is to "just look at them".
pub fn open(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[u8], sealed: &[u8]) -> Option<Vec<u8>> {
    if sealed.len() < TAG_LEN {
        return None;
    }
    let (ct, tag) = sealed.split_at(sealed.len() - TAG_LEN);
    let want = tag_for(key, nonce, aad, ct);
    // Constant time: a byte-at-a-time comparison that stops early tells an
    // attacker how much of a forged tag was right, which is enough to build
    // the rest one byte at a time.
    let mut diff = 0u8;
    for i in 0..TAG_LEN {
        diff |= want[i] ^ tag[i];
    }
    if diff != 0 {
        return None;
    }
    let mut out = ct.to_vec();
    apply(key, 1, nonce, &mut out);
    Some(out)
}

pub fn selftest() -> bool {
    // RFC 8439 section 2.8.2.
    let key: [u8; 32] = core::array::from_fn(|i| (0x80 + i) as u8);
    let nonce: [u8; 12] = [0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47];
    let aad: [u8; 12] = [0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7];
    let pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";

    let sealed = seal(&key, &nonce, &aad, pt);
    let want_tag = [
        0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60, 0x06,
        0x91,
    ];
    if sealed[sealed.len() - 16..] != want_tag {
        return false;
    }
    // First eight ciphertext bytes from the same vector.
    if sealed[..8] != [0xd3, 0x1a, 0x8d, 0x34, 0x64, 0x8e, 0x60, 0xdb] {
        return false;
    }
    match open(&key, &nonce, &aad, &sealed) {
        None => false,
        Some(out) => {
            if out != pt {
                return false;
            }
            // A flipped bit anywhere must be rejected.
            let mut bad = sealed.clone();
            bad[3] ^= 1;
            open(&key, &nonce, &aad, &bad).is_none()
        }
    }
}
