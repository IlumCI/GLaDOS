//! yespower, the proof-of-work scheme BitZeny, Yenten, Koto and friends use.
//!
//! Transliterated from `yespower-ref.c` in <https://github.com/openwall/yespower>,
//! which is 2-clause BSD:
//!
//! > Copyright 2009 Colin Percival
//! > Copyright 2012-2025 Alexander Peslyak
//! > All rights reserved.
//! >
//! > Redistribution and use in source and binary forms, with or without
//! > modification, are permitted provided that the following conditions are
//! > met: 1. Redistributions of source code must retain the above copyright
//! > notice, this list of conditions and the following disclaimer. [...]
//!
//! Same arrangement `src/doom/` has with room4doom: one file, marked at the
//! top, saying where it came from.
//!
//! **The reference and not the optimised implementation**, deliberately, and
//! upstream asks for exactly that: `yespower-opt.c` is faster and
//! `yespower-ref.c` calls itself "a simple human- and machine-readable
//! specification that implementations intended for actual use should be tested
//! against". A hash that is fast and wrong is worth nothing, and this tree has
//! no way to tell fast-and-wrong from fast-and-right except by comparing
//! against something. Speed comes after `tools/algocheck.py` agrees, and it
//! comes as a separate change with the vectors still passing.
//!
//! **And not from cpuminer-opt**, which is where a search for this lands and
//! which is GPL-2. Reading it to write this would put an obligation on the
//! whole kernel that nobody has decided to take on -- the reasoning `mkiso.py`
//! already applies to Xash3D. `design/mining.md` states the rule: each
//! algorithm's own upstream, never the multi-algo miner.
//!
//! ### Why it is worth having
//!
//! yespower is CPU-only by construction rather than by hoping ASICs do not
//! turn up: its working set is sized to hammer L2, which is what makes a GPU
//! bad at it. That same property is what bounds how many of these can run at
//! once here, and the bound is **L3 rather than cores or heap** -- see
//! `design/mining.md`.

use alloc::vec;
use alloc::vec::Vec;

use crate::crypto::hkdf;
use crate::store::sha256;

/// Which of the two incompatible schemes. Both are live on real coins, so
/// this is a parameter rather than a choice: yescrypt-era coins need 0.5 and
/// anything that hard-forked since needs 1.0.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Version {
    V0_5,
    V1_0,
}

const PWX_SIMPLE: usize = 2;
const PWX_GATHER: usize = 4;
/// `PWXgather * PWXsimple * 8`. Sixteen words, which is exactly one salsa
/// block, and that coincidence is why `blockmix_pwxform` and `blockmix_salsa`
/// step at the same granularity.
const PWX_BYTES: usize = PWX_GATHER * PWX_SIMPLE * 8;
const PWX_WORDS: usize = PWX_BYTES / 4;

#[inline(always)]
fn rotl(a: u32, b: u32) -> u32 {
    a.rotate_left(b)
}

/// The Salsa20 core over 16 words, with the SIMD shuffle upstream applies.
///
/// The shuffle is not decoration. `i * 5 % 16` is a permutation, and every
/// index in this file that looks like it is stepping through a block in order
/// is really stepping through the shuffled form. Getting it right in one place
/// and wrong in another produces a hash that is stable, fast and not yespower.
fn salsa20(b: &mut [u32; 16], rounds: u32) {
    let mut x = [0u32; 16];
    for i in 0..16 {
        x[i * 5 % 16] = b[i];
    }
    let mut n = 0;
    while n < rounds {
        // columns
        x[4] ^= rotl(x[0].wrapping_add(x[12]), 7);
        x[8] ^= rotl(x[4].wrapping_add(x[0]), 9);
        x[12] ^= rotl(x[8].wrapping_add(x[4]), 13);
        x[0] ^= rotl(x[12].wrapping_add(x[8]), 18);
        x[9] ^= rotl(x[5].wrapping_add(x[1]), 7);
        x[13] ^= rotl(x[9].wrapping_add(x[5]), 9);
        x[1] ^= rotl(x[13].wrapping_add(x[9]), 13);
        x[5] ^= rotl(x[1].wrapping_add(x[13]), 18);
        x[14] ^= rotl(x[10].wrapping_add(x[6]), 7);
        x[2] ^= rotl(x[14].wrapping_add(x[10]), 9);
        x[6] ^= rotl(x[2].wrapping_add(x[14]), 13);
        x[10] ^= rotl(x[6].wrapping_add(x[2]), 18);
        x[3] ^= rotl(x[15].wrapping_add(x[11]), 7);
        x[7] ^= rotl(x[3].wrapping_add(x[15]), 9);
        x[11] ^= rotl(x[7].wrapping_add(x[3]), 13);
        x[15] ^= rotl(x[11].wrapping_add(x[7]), 18);
        // rows
        x[1] ^= rotl(x[0].wrapping_add(x[3]), 7);
        x[2] ^= rotl(x[1].wrapping_add(x[0]), 9);
        x[3] ^= rotl(x[2].wrapping_add(x[1]), 13);
        x[0] ^= rotl(x[3].wrapping_add(x[2]), 18);
        x[6] ^= rotl(x[5].wrapping_add(x[4]), 7);
        x[7] ^= rotl(x[6].wrapping_add(x[5]), 9);
        x[4] ^= rotl(x[7].wrapping_add(x[6]), 13);
        x[5] ^= rotl(x[4].wrapping_add(x[7]), 18);
        x[11] ^= rotl(x[10].wrapping_add(x[9]), 7);
        x[8] ^= rotl(x[11].wrapping_add(x[10]), 9);
        x[9] ^= rotl(x[8].wrapping_add(x[11]), 13);
        x[10] ^= rotl(x[9].wrapping_add(x[8]), 18);
        x[12] ^= rotl(x[15].wrapping_add(x[14]), 7);
        x[13] ^= rotl(x[12].wrapping_add(x[15]), 9);
        x[14] ^= rotl(x[13].wrapping_add(x[12]), 13);
        x[15] ^= rotl(x[14].wrapping_add(x[13]), 18);
        n += 2;
    }
    for i in 0..16 {
        b[i] = b[i].wrapping_add(x[i * 5 % 16]);
    }
}

fn blockmix_salsa(b: &mut [u32], rounds: u32) {
    let mut x = [0u32; 16];
    x.copy_from_slice(&b[16..32]);
    for i in 0..2 {
        for j in 0..16 {
            x[j] ^= b[i * 16 + j];
        }
        salsa20(&mut x, rounds);
        b[i * 16..i * 16 + 16].copy_from_slice(&x);
    }
}

/// One instance, sized for a given `(version, N, r)` and reused across nonces.
///
/// **The allocation is the point of the type.** `V` is 128*r*N bytes -- 8 MiB
/// at N=2048, r=32 -- and a miner hashes millions of nonces against one
/// parameter set. Allocating per hash would spend more time in the heap than
/// in the algorithm, and this kernel's heap takes a lock on every allocation.
pub struct Yespower {
    version: Version,
    n: u32,
    r: u32,
    rounds: u32,
    pwx_rounds: u32,
    swidth: u32,
    smask: u32,
    /// The S-boxes, as words. `s0`/`s1`/`s2` are word offsets into it and
    /// rotate every pwxform call on 1.0.
    s: Vec<u32>,
    s0: usize,
    s1: usize,
    s2: usize,
    w: usize,
    v: Vec<u32>,
    x: Vec<u32>,
    b: Vec<u32>,
    /// A separate 32-word scratch for the S-box-filling `smix1`, which runs at
    /// r = 1 and must not share the main one.
    xs: Vec<u32>,
}

impl Yespower {
    /// `None` when the parameters are outside what the scheme admits, or when
    /// the working set will not fit.
    ///
    /// Refusing rather than clamping: every one of these bounds is part of
    /// what a coin's network agreed on, so a clamped parameter produces
    /// perfectly valid hashes of the wrong function.
    pub fn new(version: Version, n: u32, r: u32) -> Option<Yespower> {
        if !(1024..=512 * 1024).contains(&n) || !(8..=32).contains(&r) || n & (n - 1) != 0 {
            return None;
        }
        let (rounds, pwx_rounds, swidth, sboxes) = match version {
            Version::V0_5 => (8u32, 6u32, 8u32, 2usize),
            Version::V1_0 => (2, 3, 11, 3),
        };
        let sbytes = sboxes * (1usize << swidth) * PWX_SIMPLE * 8;
        let bw = 32 * r as usize; // words in one 128r-byte block
        let vw = bw.checked_mul(n as usize)?;
        let pairs = (1usize << swidth) * PWX_SIMPLE;
        Some(Yespower {
            version,
            n,
            r,
            rounds,
            pwx_rounds,
            swidth,
            smask: (((1u32 << swidth) - 1) * PWX_SIMPLE as u32 * 8),
            s: vec![0u32; sbytes / 4],
            s0: 0,
            s1: pairs * 2,
            s2: pairs * 4,
            w: 0,
            v: vec![0u32; vw],
            x: vec![0u32; bw],
            b: vec![0u32; bw],
            xs: vec![0u32; 32],
        })
    }

    /// Bytes of working memory this instance holds, for the resource budget.
    pub fn footprint(&self) -> usize {
        (self.s.len() + self.v.len() + self.x.len() + self.b.len() + self.xs.len()) * 4
    }

    pub fn params(&self) -> (u32, u32) {
        (self.n, self.r)
    }

    fn pwxform(&mut self, x: &mut [u32; PWX_WORDS]) {
        let (s0, s1) = (self.s0, self.s1);
        let smask = self.smask as usize;
        let mut w = self.w;
        for i in 0..self.pwx_rounds {
            for j in 0..PWX_GATHER {
                let xl = x[j * 4] as usize;
                let xh = x[j * 4 + 1] as usize;
                let p0 = s0 + (xl & smask) / 8 * 2;
                let p1 = s1 + (xh & smask) / 8 * 2;
                for k in 0..PWX_SIMPLE {
                    let sa = ((self.s[p0 + k * 2 + 1] as u64) << 32) + self.s[p0 + k * 2] as u64;
                    let sb = ((self.s[p1 + k * 2 + 1] as u64) << 32) + self.s[p1 + k * 2] as u64;
                    let lo = x[j * 4 + k * 2] as u64;
                    let hi = x[j * 4 + k * 2 + 1] as u64;
                    // Wrapping throughout: this is a 64-bit multiply of two
                    // 32-bit halves and it is *meant* to discard the top.
                    let mut v = hi.wrapping_mul(lo);
                    v = v.wrapping_add(sa);
                    v ^= sb;
                    x[j * 4 + k * 2] = v as u32;
                    x[j * 4 + k * 2 + 1] = (v >> 32) as u32;
                }
                // 1.0 writes back into the S-boxes as it goes, which is what
                // makes it data-dependent in a way 0.5 is not. The asymmetry
                // between the two branches is upstream's: the odd one advances
                // `w` per element, the even one does not advance it at all.
                if self.version != Version::V0_5 && (i == 0 || j < PWX_GATHER / 2) {
                    if j & 1 != 0 {
                        for k in 0..PWX_SIMPLE {
                            self.s[s1 + w * 2] = x[j * 4 + k * 2];
                            self.s[s1 + w * 2 + 1] = x[j * 4 + k * 2 + 1];
                            w += 1;
                        }
                    } else {
                        for k in 0..PWX_SIMPLE {
                            self.s[s0 + (w + k) * 2] = x[j * 4 + k * 2];
                            self.s[s0 + (w + k) * 2 + 1] = x[j * 4 + k * 2 + 1];
                        }
                    }
                }
            }
        }
        if self.version != Version::V0_5 {
            let (a, b, c) = (self.s0, self.s1, self.s2);
            self.s0 = c;
            self.s1 = a;
            self.s2 = b;
            self.w = w & ((1usize << self.swidth) * PWX_SIMPLE - 1);
        }
    }

    /// `blockmix_pwxform` over `buf[off .. off + 32r]`.
    fn blockmix_pwxform(&mut self, buf: &mut [u32], off: usize, r: usize) {
        let r1 = 128 * r / PWX_BYTES;
        let mut x = [0u32; PWX_WORDS];
        x.copy_from_slice(&buf[off + (r1 - 1) * PWX_WORDS..off + r1 * PWX_WORDS]);
        for i in 0..r1 {
            if r1 > 1 {
                for j in 0..PWX_WORDS {
                    x[j] ^= buf[off + i * PWX_WORDS + j];
                }
            }
            self.pwxform(&mut x);
            buf[off + i * PWX_WORDS..off + (i + 1) * PWX_WORDS].copy_from_slice(&x);
        }
        let mut i = (r1 - 1) * PWX_BYTES / 64;
        let mut blk = [0u32; 16];
        blk.copy_from_slice(&buf[off + i * 16..off + i * 16 + 16]);
        salsa20(&mut blk, self.rounds);
        buf[off + i * 16..off + i * 16 + 16].copy_from_slice(&blk);
        // A no-op at the current pwxform settings, since PWX_BYTES is 64 and
        // so `i` lands on the last block. Kept because upstream keeps it, and
        // because a settings change that made it reachable would otherwise
        // silently drop a step.
        i += 1;
        while i < 2 * r {
            let mut b2 = [0u32; 16];
            b2.copy_from_slice(&buf[off + i * 16..off + i * 16 + 16]);
            for j in 0..16 {
                b2[j] ^= buf[off + (i - 1) * 16 + j];
            }
            salsa20(&mut b2, self.rounds);
            buf[off + i * 16..off + i * 16 + 16].copy_from_slice(&b2);
            i += 1;
        }
    }
}

#[inline]
fn integerify(x: &[u32], r: usize) -> u32 {
    x[(2 * r - 1) * 16]
}

fn p2floor(mut x: u32) -> u32 {
    loop {
        let y = x & x.wrapping_sub(1);
        if y == 0 {
            return x;
        }
        x = y;
    }
}

fn wrap(x: u32, i: u32) -> u32 {
    let n = p2floor(i);
    (x & (n - 1)).wrapping_add(i - n)
}

/// PBKDF2-HMAC-SHA256 with one iteration, which is all yespower asks for.
///
/// Written out rather than reached for, because at `c = 1` the whole function
/// collapses to one HMAC per output block and a general implementation would
/// be a loop that always runs once.
fn pbkdf2_1(password: &[u8], salt: &[u8], out: &mut [u8]) {
    let mut block = 1u32;
    let mut at = 0;
    let mut msg = Vec::with_capacity(salt.len() + 4);
    while at < out.len() {
        msg.clear();
        msg.extend_from_slice(salt);
        msg.extend_from_slice(&block.to_be_bytes());
        let t = hkdf::hmac(password, &msg);
        let take = core::cmp::min(32, out.len() - at);
        out[at..at + take].copy_from_slice(&t[..take]);
        at += take;
        block += 1;
    }
}

impl Yespower {
    /// Hash `src` under this instance's parameters.
    ///
    /// `pers` is the coin's personalisation string, which is part of what makes
    /// one coin's proof-of-work not another's -- and note it enters at a
    /// *different point* in each version: 0.5 folds it in at the end, 1.0 uses
    /// it as the PBKDF2 salt at the start. A single implementation that put it
    /// in one place would be right for one version and wrong for the other.
    pub fn hash(&mut self, src: &[u8], pers: Option<&[u8]>) -> [u8; 32] {
        let r = self.r as usize;
        let bw = 32 * r;
        let b_size = 128 * r;

        // Reset the rotating S-box state. The buffers themselves need no
        // clearing: the first `smix1` writes the whole of S before anything
        // reads it, which is why upstream can leave it malloc'd and unzeroed.
        let pairs = (1usize << self.swidth) * PWX_SIMPLE;
        self.s0 = 0;
        self.s1 = pairs * 2;
        self.s2 = pairs * 4;
        self.w = 0;

        let sha = sha256::hash(src);
        let salt: &[u8] = match self.version {
            Version::V0_5 => src,
            Version::V1_0 => pers.unwrap_or(&[]),
        };
        let mut bbytes = vec![0u8; b_size];
        pbkdf2_1(&sha, salt, &mut bbytes);

        let mut b = core::mem::take(&mut self.b);
        for i in 0..bw {
            b[i] = u32::from_le_bytes([
                bbytes[i * 4],
                bbytes[i * 4 + 1],
                bbytes[i * 4 + 2],
                bbytes[i * 4 + 3],
            ]);
        }
        let mut sha2 = [0u8; 32];
        for i in 0..8 {
            sha2[i * 4..i * 4 + 4].copy_from_slice(&b[i].to_le_bytes());
        }

        self.smix(&mut b);

        for i in 0..bw {
            bbytes[i * 4..i * 4 + 4].copy_from_slice(&b[i].to_le_bytes());
        }
        self.b = b;

        match self.version {
            Version::V0_5 => {
                let mut dst = [0u8; 32];
                pbkdf2_1(&sha2, &bbytes, &mut dst);
                if let Some(p) = pers {
                    let h = hkdf::hmac(&dst, p);
                    dst = sha256::hash(&h);
                }
                dst
            }
            // Key is the *last 64 bytes of B* and the message is the digest
            // carried down from the start, not the other way round. Swapping
            // them is a perfectly good HMAC of the wrong thing.
            Version::V1_0 => hkdf::hmac(&bbytes[b_size - 64..], &sha2),
        }
        .into()
    }

    fn smix(&mut self, b: &mut [u32]) {
        let n = self.n;
        let r = self.r as usize;
        let mut nloop_all = (n + 2) / 3;
        let mut nloop_rw = nloop_all;
        nloop_all += 1;
        nloop_all &= !1u32;
        if self.version == Version::V0_5 {
            nloop_rw &= !1u32;
        } else {
            nloop_rw += 1;
            nloop_rw &= !1u32;
        }

        // Fills the S-boxes, at r = 1 and over S itself rather than V, which is
        // what selects the salsa blockmix instead of pwxform -- pwxform cannot
        // run yet because the boxes it would read are what this is writing.
        let sn = (self.s.len() * 4 / 128) as u32;
        self.smix1_into_s(b, sn);
        self.smix1(b, r, n);
        self.smix2(b, r, n, nloop_rw);
        self.smix2(b, r, n, nloop_all - nloop_rw);
    }

    fn smix1_into_s(&mut self, b: &mut [u32], n: u32) {
        let mut x = core::mem::take(&mut self.xs);
        for k in 0..2 {
            for i in 0..16 {
                x[k * 16 + i] = b[k * 16 + (i * 5 % 16)];
            }
        }
        for i in 0..n as usize {
            self.s[i * 32..(i + 1) * 32].copy_from_slice(&x[..32]);
            if i > 1 {
                let j = wrap(integerify(&x, 1), i as u32) as usize;
                for q in 0..32 {
                    x[q] ^= self.s[j * 32 + q];
                }
            }
            blockmix_salsa(&mut x, self.rounds);
        }
        for k in 0..2 {
            for i in 0..16 {
                b[k * 16 + (i * 5 % 16)] = x[k * 16 + i];
            }
        }
        self.xs = x;
    }

    fn smix1(&mut self, b: &mut [u32], r: usize, n: u32) {
        let s = 32 * r;
        let mut x = core::mem::take(&mut self.x);
        for k in 0..2 * r {
            for i in 0..16 {
                x[k * 16 + i] = b[k * 16 + (i * 5 % 16)];
            }
        }
        if self.version != Version::V0_5 {
            for k in 1..r {
                let (lo, hi) = x.split_at_mut(k * 32);
                hi[..32].copy_from_slice(&lo[(k - 1) * 32..k * 32]);
                self.blockmix_pwxform(&mut x, k * 32, 1);
            }
        }
        let mut v = core::mem::take(&mut self.v);
        for i in 0..n as usize {
            v[i * s..(i + 1) * s].copy_from_slice(&x[..s]);
            if i > 1 {
                let j = wrap(integerify(&x, r), i as u32) as usize;
                for q in 0..s {
                    x[q] ^= v[j * s + q];
                }
            }
            self.blockmix_pwxform(&mut x, 0, r);
        }
        self.v = v;
        for k in 0..2 * r {
            for i in 0..16 {
                b[k * 16 + (i * 5 % 16)] = x[k * 16 + i];
            }
        }
        self.x = x;
    }

    fn smix2(&mut self, b: &mut [u32], r: usize, n: u32, nloop: u32) {
        let s = 32 * r;
        let mut x = core::mem::take(&mut self.x);
        for k in 0..2 * r {
            for i in 0..16 {
                x[k * 16 + i] = b[k * 16 + (i * 5 % 16)];
            }
        }
        let mut v = core::mem::take(&mut self.v);
        for _ in 0..nloop {
            let j = (integerify(&x, r) & (n - 1)) as usize;
            for q in 0..s {
                x[q] ^= v[j * s + q];
            }
            // The one place the second loop writes back, and it is skipped for
            // the short tail pass. Upstream gates on `Nloop != 2` rather than
            // on which call this is, so the condition travels with the value.
            if nloop != 2 {
                v[j * s..(j + 1) * s].copy_from_slice(&x[..s]);
            }
            self.blockmix_pwxform(&mut x, 0, r);
        }
        self.v = v;
        for k in 0..2 * r {
            for i in 0..16 {
                b[k * 16 + (i * 5 % 16)] = x[k * 16 + i];
            }
        }
        self.x = x;
    }
}
