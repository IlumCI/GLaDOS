//! SHA-1, HMAC-SHA1, and PBKDF2 -- for WPA2 and nothing else.
//!
//! SHA-1 is broken for signatures: collisions are findable, and anything that
//! depends on two distinct messages hashing differently must not use it. That
//! is why nothing in the TLS path touches this module.
//!
//! It is here because WPA2 specifies it and a supplicant does not get a vote.
//! The uses that remain are HMAC and PBKDF2, which depend on SHA-1's
//! resistance to *preimage* attacks rather than collisions, and that is not
//! broken. An attacker who can find SHA-1 collisions still cannot recover a
//! WPA2 passphrase.

use alloc::vec::Vec;

pub const HASH_LEN: usize = 20;
const BLOCK_LEN: usize = 64;

pub struct Sha1 {
    h: [u32; 5],
    buf: [u8; BLOCK_LEN],
    used: usize,
    len: u64,
}

impl Sha1 {
    pub fn new() -> Self {
        Sha1 {
            h: [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0],
            buf: [0; BLOCK_LEN],
            used: 0,
            len: 0,
        }
    }

    fn block(&mut self, b: &[u8]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b_, mut c, mut d, mut e) =
            (self.h[0], self.h[1], self.h[2], self.h[3], self.h[4]);
        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b_ & c) | (!b_ & d), 0x5A827999),
                20..=39 => (b_ ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b_ & c) | (b_ & d) | (c & d), 0x8F1BBCDC),
                _ => (b_ ^ c ^ d, 0xCA62C1D6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b_.rotate_left(30);
            b_ = a;
            a = t;
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b_);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if self.used > 0 {
            let take = core::cmp::min(BLOCK_LEN - self.used, data.len());
            self.buf[self.used..self.used + take].copy_from_slice(&data[..take]);
            self.used += take;
            data = &data[take..];
            if self.used == BLOCK_LEN {
                let b = self.buf;
                self.block(&b);
                self.used = 0;
            }
        }
        while data.len() >= BLOCK_LEN {
            let (b, rest) = data.split_at(BLOCK_LEN);
            self.block(b);
            data = rest;
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.used = data.len();
        }
    }

    pub fn finish(mut self) -> [u8; HASH_LEN] {
        let bits = self.len * 8;
        self.update(&[0x80]);
        while self.used != 56 {
            self.update(&[0]);
        }
        let b = bits.to_be_bytes();
        self.update(&b);

        let mut out = [0u8; HASH_LEN];
        for i in 0..5 {
            out[i * 4..i * 4 + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
}

pub fn hash(data: &[u8]) -> [u8; HASH_LEN] {
    let mut h = Sha1::new();
    h.update(data);
    h.finish()
}

pub fn hmac(key: &[u8], data: &[u8]) -> [u8; HASH_LEN] {
    let mut k = [0u8; BLOCK_LEN];
    if key.len() > BLOCK_LEN {
        k[..HASH_LEN].copy_from_slice(&hash(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK_LEN];
    let mut opad = [0x5Cu8; BLOCK_LEN];
    for i in 0..BLOCK_LEN {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = Sha1::new();
    inner.update(&ipad);
    inner.update(data);
    let inner = inner.finish();
    let mut outer = Sha1::new();
    outer.update(&opad);
    outer.update(&inner);
    outer.finish()
}

/// PBKDF2-HMAC-SHA1, RFC 2898.
///
/// The iteration count is the whole security argument: WPA2 fixes it at 4096,
/// which in 2004 made a dictionary attack expensive and today makes it merely
/// annoying. That is a property of the standard, not of this code.
pub fn pbkdf2(password: &[u8], salt: &[u8], iterations: u32, out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len);
    let mut block: u32 = 1;
    while out.len() < out_len {
        let mut salted = Vec::with_capacity(salt.len() + 4);
        salted.extend_from_slice(salt);
        salted.extend_from_slice(&block.to_be_bytes());
        let mut u = hmac(password, &salted);
        let mut acc = u;
        for _ in 1..iterations {
            u = hmac(password, &u);
            for i in 0..HASH_LEN {
                acc[i] ^= u[i];
            }
        }
        out.extend_from_slice(&acc);
        block += 1;
    }
    out.truncate(out_len);
    out
}

pub fn selftest() -> bool {
    // FIPS 180-1.
    let want = [
        0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50, 0xc2,
        0x6c, 0x9c, 0xd0, 0xd8, 0x9d,
    ];
    if hash(b"abc") != want {
        return false;
    }

    // RFC 2202 HMAC-SHA1 test case 1.
    let mac = hmac(&[0x0b; 20], b"Hi There");
    let want = [
        0xb6, 0x17, 0x31, 0x86, 0x55, 0x05, 0x72, 0x64, 0xe2, 0x8b, 0xc0, 0xb6, 0xfb, 0x37, 0x8c,
        0x8e, 0xf1, 0x46, 0xbe, 0x00,
    ];
    if mac != want {
        return false;
    }

    // RFC 6070 PBKDF2 test cases 1 and 2. The iteration counts are small on
    // purpose; the 4096-round case is exercised by the WPA2 vector instead.
    let d = pbkdf2(b"password", b"salt", 1, 20);
    let want = [
        0x0c, 0x60, 0xc8, 0x0f, 0x96, 0x1f, 0x0e, 0x71, 0xf3, 0xa9, 0xb5, 0x24, 0xaf, 0x60, 0x12,
        0x06, 0x2f, 0xe0, 0x37, 0xa6,
    ];
    if d[..] != want[..] {
        return false;
    }
    let d = pbkdf2(b"password", b"salt", 2, 20);
    let want = [
        0xea, 0x6c, 0x01, 0x4d, 0xc7, 0x2d, 0x6f, 0x8c, 0xcd, 0x1e, 0xd9, 0x2a, 0xce, 0x1d, 0x41,
        0xf0, 0xd8, 0xde, 0x89, 0x57,
    ];
    d[..] == want[..]
}
