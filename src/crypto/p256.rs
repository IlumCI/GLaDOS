//! ECDSA verification on NIST P-256 (secp256r1).
//!
//! Verification only. Everything here operates on public values -- a
//! signature, a public key, a message hash -- so the non-constant-time
//! arithmetic underneath is not a leak. Signing would need entirely different
//! care and is not implemented.
//!
//! ### Jacobian coordinates, and why the obvious choice was wrong
//!
//! This started out affine, on the reasoning that Jacobian only saves an
//! inversion per point operation and that cannot matter when the job is
//! verifying two certificates. That reasoning was wrong, and expensively so.
//!
//! An inversion here is a full modular exponentiation -- for P-384, about 600
//! multiplies. A scalar multiply is roughly 768 point operations. So affine
//! coordinates meant ~460,000 modular multiplies per signature, each
//! allocating, and verifying one real certificate chain exhausted the heap and
//! panicked. Jacobian does one inversion at the very end: ~13,000 multiplies,
//! a factor of thirty-five.
//!
//! A point is (X, Y, Z) standing for the affine (X/Z^2, Y/Z^3), with Z = 0 as
//! the identity. Everything stays in Montgomery form for the whole ladder --
//! converting per operation was the second-largest cost -- and comes out once
//! at the end.

use super::bigint::{Big, Mont};
use alloc::vec::Vec;

/// p = 2^256 - 2^224 + 2^192 + 2^96 - 1
const P: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];
/// n, the order of the base point.
const N: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xBC, 0xE6, 0xFA, 0xAD, 0xA7, 0x17, 0x9E, 0x84, 0xF3, 0xB9, 0xCA, 0xC2, 0xFC, 0x63, 0x25, 0x51,
];
const B: [u8; 32] = [
    0x5A, 0xC6, 0x35, 0xD8, 0xAA, 0x3A, 0x93, 0xE7, 0xB3, 0xEB, 0xBD, 0x55, 0x76, 0x98, 0x86, 0xBC,
    0x65, 0x1D, 0x06, 0xB0, 0xCC, 0x53, 0xB0, 0xF6, 0x3B, 0xCE, 0x3C, 0x3E, 0x27, 0xD2, 0x60, 0x4B,
];
const GX: [u8; 32] = [
    0x6B, 0x17, 0xD1, 0xF2, 0xE1, 0x2C, 0x42, 0x47, 0xF8, 0xBC, 0xE6, 0xE5, 0x63, 0xA4, 0x40, 0xF2,
    0x77, 0x03, 0x7D, 0x81, 0x2D, 0xEB, 0x33, 0xA0, 0xF4, 0xA1, 0x39, 0x45, 0xD8, 0x98, 0xC2, 0x96,
];
const GY: [u8; 32] = [
    0x4F, 0xE3, 0x42, 0xE2, 0xFE, 0x1A, 0x7F, 0x9B, 0x8E, 0xE7, 0xEB, 0x4A, 0x7C, 0x0F, 0x9E, 0x16,
    0x2B, 0xCE, 0x33, 0x57, 0x6B, 0x31, 0x5E, 0xCE, 0xCB, 0xB6, 0x40, 0x68, 0x37, 0xBF, 0x51, 0xF5,
];

/// A point in Jacobian coordinates, with every value in Montgomery form.
/// `z` zero is the point at infinity, which removes the separate flag and the
/// special cases that went with it.
#[derive(Clone)]
struct Point {
    x: Big,
    y: Big,
    z: Big,
}

/// p = 2^384 - 2^128 - 2^96 + 2^32 - 1
const P384_P: [u8; 48] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
];
const P384_N: [u8; 48] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC7, 0x63, 0x4D, 0x81, 0xF4, 0x37, 0x2D, 0xDF,
    0x58, 0x1A, 0x0D, 0xB2, 0x48, 0xB0, 0xA7, 0x7A, 0xEC, 0xEC, 0x19, 0x6A, 0xCC, 0xC5, 0x29, 0x73,
];
const P384_B: [u8; 48] = [
    0xB3, 0x31, 0x2F, 0xA7, 0xE2, 0x3E, 0xE7, 0xE4, 0x98, 0x8E, 0x05, 0x6B, 0xE3, 0xF8, 0x2D, 0x19,
    0x18, 0x1D, 0x9C, 0x6E, 0xFE, 0x81, 0x41, 0x12, 0x03, 0x14, 0x08, 0x8F, 0x50, 0x13, 0x87, 0x5A,
    0xC6, 0x56, 0x39, 0x8D, 0x8A, 0x2E, 0xD1, 0x9D, 0x2A, 0x85, 0xC8, 0xED, 0xD3, 0xEC, 0x2A, 0xEF,
];
const P384_GX: [u8; 48] = [
    0xAA, 0x87, 0xCA, 0x22, 0xBE, 0x8B, 0x05, 0x37, 0x8E, 0xB1, 0xC7, 0x1E, 0xF3, 0x20, 0xAD, 0x74,
    0x6E, 0x1D, 0x3B, 0x62, 0x8B, 0xA7, 0x9B, 0x98, 0x59, 0xF7, 0x41, 0xE0, 0x82, 0x54, 0x2A, 0x38,
    0x55, 0x02, 0xF2, 0x5D, 0xBF, 0x55, 0x29, 0x6C, 0x3A, 0x54, 0x5E, 0x38, 0x72, 0x76, 0x0A, 0xB7,
];
const P384_GY: [u8; 48] = [
    0x36, 0x17, 0xDE, 0x4A, 0x96, 0x26, 0x2C, 0x6F, 0x5D, 0x9E, 0x98, 0xBF, 0x92, 0x92, 0xDC, 0x29,
    0xF8, 0xF4, 0x1D, 0xBD, 0x28, 0x9A, 0x14, 0x7C, 0xE9, 0xDA, 0x31, 0x13, 0xB5, 0xF0, 0xB8, 0xC0,
    0x0A, 0x60, 0xB1, 0xCE, 0x1D, 0x7E, 0x81, 0x9D, 0x7A, 0x43, 0x1D, 0x7C, 0x90, 0xEA, 0x0E, 0x5F,
];

/// Which NIST prime curve. Both have a = -3, which is the only property the
/// arithmetic below depends on beyond the constants themselves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Nist {
    P256,
    P384,
}

impl Nist {
    fn size(self) -> usize {
        match self {
            Nist::P256 => 32,
            Nist::P384 => 48,
        }
    }
    fn p(self) -> &'static [u8] {
        match self {
            Nist::P256 => &P,
            Nist::P384 => &P384_P,
        }
    }
    fn n(self) -> &'static [u8] {
        match self {
            Nist::P256 => &N,
            Nist::P384 => &P384_N,
        }
    }
    fn b(self) -> &'static [u8] {
        match self {
            Nist::P256 => &B,
            Nist::P384 => &P384_B,
        }
    }
    fn g(self) -> (&'static [u8], &'static [u8]) {
        match self {
            Nist::P256 => (&GX, &GY),
            Nist::P384 => (&P384_GX, &P384_GY),
        }
    }
}

struct Curve {
    fp: Mont,
    p: Big,
    /// a = -3, in Montgomery form.
    a_m: Big,
    b_m: Big,
    limbs: usize,
}

impl Curve {
    fn new(c: Nist) -> Curve {
        let p = Big::from_bytes(c.p());
        let fp = Mont::new(&p).expect("p is odd");
        let three = Big::from_u64(3, p.limbs());
        let a = p.sub_mod(&three, &p); // -3 mod p
        let a_m = fp.to_mont(&a);
        let b_m = fp.to_mont(&Big::from_bytes(c.b()));
        let limbs = p.limbs();
        Curve { fp, p, a_m, b_m, limbs }
    }

    fn zero(&self) -> Point {
        Point {
            x: self.one_m(),
            y: self.one_m(),
            z: Big::zero(self.limbs),
        }
    }

    fn one_m(&self) -> Big {
        self.fp.to_mont(&Big::from_u64(1, self.limbs))
    }

    /// Lift affine (x, y) into Montgomery-form Jacobian with Z = 1.
    fn affine(&self, x: &Big, y: &Big) -> Point {
        Point {
            x: self.fp.to_mont(x),
            y: self.fp.to_mont(y),
            z: self.one_m(),
        }
    }

    fn dbl_m(&self, a: &Big) -> Big {
        a.add_mod(a, &self.p)
    }

    /// Invert a value that is *already* in Montgomery form, returning one that
    /// still is.
    ///
    /// `Mont::inv_prime` takes and returns ordinary values -- it converts
    /// internally -- so handing it a Montgomery value computes
    /// `(z*R)^(p-2)`, which is not `z^-1 * R` and is not anything useful. The
    /// affine code got this right and the Jacobian rewrite dropped it, which
    /// broke every ECDSA verification while leaving RSA untouched: exactly the
    /// shape of failure the trust-store check surfaced.
    fn inv_m(&self, a_m: &Big) -> Big {
        self.fp.to_mont(&self.fp.inv_prime(&self.fp.from_mont(a_m)))
    }

    /// y^2 == x^3 - 3x + b, on the affine values as given.
    fn on_curve(&self, x: &Big, y: &Big) -> bool {
        let x = self.fp.to_mont(x);
        let y = self.fp.to_mont(y);
        let y2 = self.fp.mul(&y, &y);
        let x3 = self.fp.mul(&self.fp.mul(&x, &x), &x);
        let ax = self.fp.mul(&self.a_m, &x);
        y2 == x3.add_mod(&ax, &self.p).add_mod(&self.b_m, &self.p)
    }

    /// Point doubling, the a = -3 form. Both NIST curves have a = -3, which is
    /// what lets `alpha` be computed as 3(X-Z^2)(X+Z^2) instead of needing a
    /// separate multiply by a.
    fn double(&self, pt: &Point) -> Point {
        if pt.z.is_zero() {
            return self.zero();
        }
        let delta = self.fp.mul(&pt.z, &pt.z);
        let gamma = self.fp.mul(&pt.y, &pt.y);
        let beta = self.fp.mul(&pt.x, &gamma);

        let t1 = pt.x.sub_mod(&delta, &self.p);
        let t2 = pt.x.add_mod(&delta, &self.p);
        let t3 = self.fp.mul(&t1, &t2);
        let alpha = t3.add_mod(&t3, &self.p).add_mod(&t3, &self.p);

        let x3 = {
            let a2 = self.fp.mul(&alpha, &alpha);
            let eight_beta = self.dbl_m(&self.dbl_m(&self.dbl_m(&beta)));
            a2.sub_mod(&eight_beta, &self.p)
        };
        let z3 = {
            let yz = pt.y.add_mod(&pt.z, &self.p);
            let s = self.fp.mul(&yz, &yz);
            s.sub_mod(&gamma, &self.p).sub_mod(&delta, &self.p)
        };
        let y3 = {
            let four_beta = self.dbl_m(&self.dbl_m(&beta));
            let t = four_beta.sub_mod(&x3, &self.p);
            let g2 = self.fp.mul(&gamma, &gamma);
            let eight_g2 = self.dbl_m(&self.dbl_m(&self.dbl_m(&g2)));
            self.fp.mul(&alpha, &t).sub_mod(&eight_g2, &self.p)
        };
        Point { x: x3, y: y3, z: z3 }
    }

    /// Mixed addition: Jacobian + affine (Z2 = 1). Both operands added to the
    /// accumulator are fixed affine points, so the general Jacobian-Jacobian
    /// form is never needed.
    fn add_mixed(&self, a: &Point, bx: &Big, by: &Big) -> Point {
        if a.z.is_zero() {
            return Point {
                x: bx.clone(),
                y: by.clone(),
                z: self.one_m(),
            };
        }
        let z1z1 = self.fp.mul(&a.z, &a.z);
        let u2 = self.fp.mul(bx, &z1z1);
        let s2 = self.fp.mul(&self.fp.mul(by, &a.z), &z1z1);

        let h = u2.sub_mod(&a.x, &self.p);
        let r = self.dbl_m(&s2.sub_mod(&a.y, &self.p));

        if h.is_zero() {
            // Same x. Either the points are equal -- a doubling -- or they are
            // inverses and the sum is the identity. Getting this wrong is the
            // classic Jacobian bug: the general formula yields (0,0,0) here
            // and silently poisons the rest of the ladder.
            if r.is_zero() {
                return self.double(a);
            }
            return self.zero();
        }

        let hh = self.fp.mul(&h, &h);
        let i = self.dbl_m(&self.dbl_m(&hh));
        let j = self.fp.mul(&h, &i);
        let v = self.fp.mul(&a.x, &i);

        let x3 = {
            let r2 = self.fp.mul(&r, &r);
            r2.sub_mod(&j, &self.p).sub_mod(&v, &self.p).sub_mod(&v, &self.p)
        };
        let y3 = {
            let t = v.sub_mod(&x3, &self.p);
            let y1j = self.fp.mul(&a.y, &j);
            self.fp
                .mul(&r, &t)
                .sub_mod(&y1j, &self.p)
                .sub_mod(&y1j, &self.p)
        };
        let z3 = {
            let zh = a.z.add_mod(&h, &self.p);
            let s = self.fp.mul(&zh, &zh);
            s.sub_mod(&z1z1, &self.p).sub_mod(&hh, &self.p)
        };
        Point { x: x3, y: y3, z: z3 }
    }

    /// u1*G + u2*Q, by interleaved double-and-add ("Shamir's trick").
    ///
    /// One pass over the bits instead of two scalar multiplications. Safe here
    /// only because both scalars are public.
    ///
    /// Returns the affine x-coordinate, which is all ECDSA verification needs
    /// -- so the single inversion converts x and nothing else.
    fn mul_two_x(&self, u1: &Big, g: (&Big, &Big), u2: &Big, q: (&Big, &Big)) -> Option<Big> {
        // G + Q, precomputed once so a bit set in both scalars costs one add.
        let sum = {
            let gp = self.affine(g.0, g.1);
            let p = self.add_mixed(&gp, &self.fp.to_mont(q.0), &self.fp.to_mont(q.1));
            if p.z.is_zero() {
                None
            } else {
                // Bring it back to affine so it can be used with add_mixed.
                let zi = self.inv_m(&p.z);
                let zi2 = self.fp.mul(&zi, &zi);
                let zi3 = self.fp.mul(&zi2, &zi);
                Some((self.fp.mul(&p.x, &zi2), self.fp.mul(&p.y, &zi3)))
            }
        };

        let gm = (self.fp.to_mont(g.0), self.fp.to_mont(g.1));
        let qm = (self.fp.to_mont(q.0), self.fp.to_mont(q.1));

        let mut acc = self.zero();
        let bits = u1.bits().max(u2.bits());
        for i in (0..bits).rev() {
            acc = self.double(&acc);
            let (b1, b2) = (u1.bit(i), u2.bit(i));
            if b1 && b2 {
                match &sum {
                    // G + Q was the identity, so G = -Q and the two terms
                    // cancel: adding neither is correct.
                    None => {}
                    Some((sx, sy)) => acc = self.add_mixed(&acc, sx, sy),
                }
            } else if b1 {
                acc = self.add_mixed(&acc, &gm.0, &gm.1);
            } else if b2 {
                acc = self.add_mixed(&acc, &qm.0, &qm.1);
            }
        }

        if acc.z.is_zero() {
            return None;
        }
        // The one inversion in the whole operation.
        let zi = self.inv_m(&acc.z);
        let zi2 = self.fp.mul(&zi, &zi);
        Some(self.fp.from_mont(&self.fp.mul(&acc.x, &zi2)))
    }
}

/// Verify an ECDSA signature: `sig` is (r, s) as two 32-byte values, `pubkey`
/// is an uncompressed point (0x04 || X || Y), `hash` is the message digest.
pub fn verify(c: Nist, pubkey: &[u8], hash: &[u8], r: &[u8], s: &[u8]) -> bool {
    let sz = c.size();
    if pubkey.len() != 1 + 2 * sz || pubkey[0] != 0x04 {
        // Compressed points are legal and rare in certificates; refusing is
        // better than decompressing incorrectly.
        return false;
    }
    let curve = Curve::new(c);
    let n = Big::from_bytes(c.n());
    let fnq = match Mont::new(&n) {
        None => return false,
        Some(x) => x,
    };

    let r = Big::from_bytes(r);
    let s = Big::from_bytes(s);
    // Both must be in [1, n-1]. Zero is the classic forgery that a missing
    // range check lets through.
    if r.is_zero() || s.is_zero() {
        return false;
    }
    if r.cmp(&n) != core::cmp::Ordering::Less || s.cmp(&n) != core::cmp::Ordering::Less {
        return false;
    }

    let qx = Big::from_bytes(&pubkey[1..1 + sz]);
    let qy = Big::from_bytes(&pubkey[1 + sz..1 + 2 * sz]);
    // A public key off the curve can be used to extract information in some
    // protocols; checking is cheap and unconditional.
    if !curve.on_curve(&qx, &qy) {
        return false;
    }

    // e is the leftmost bits of the hash, up to the bit length of n: a
    // SHA-384 digest verified against P-256 is truncated, and a SHA-256
    // digest against P-384 is used whole.
    let e = Big::from_bytes(&hash[..hash.len().min(sz)]).rem(&n);

    let w = fnq.inv_prime(&s);
    let u1 = fnq.from_mont(&fnq.mul(&fnq.to_mont(&e), &fnq.to_mont(&w)));
    let u2 = fnq.from_mont(&fnq.mul(&fnq.to_mont(&r), &fnq.to_mont(&w)));

    let (gxb, gyb) = c.g();
    let gx = Big::from_bytes(gxb);
    let gy = Big::from_bytes(gyb);
    match curve.mul_two_x(&u1, (&gx, &gy), &u2, (&qx, &qy)) {
        None => false,
        Some(x) => x.rem(&n).cmp(&r) == core::cmp::Ordering::Equal,
    }
}

/// Parse the DER SEQUENCE { INTEGER r, INTEGER s } that carries an ECDSA
/// signature, into two fixed-width 32-byte values.
pub fn parse_der_signature(der: &[u8], sz: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut p = crate::net::x509::Der::new(der);
    let mut seq = p.sequence()?;
    let r = seq.integer()?;
    let s = seq.integer()?;
    Some((pad_to(r, sz)?, pad_to(s, sz)?))
}

fn pad_to(v: &[u8], sz: usize) -> Option<Vec<u8>> {
    // DER integers are signed, so a value with the high bit set carries a
    // leading zero byte that is not part of the number.
    let v = if v.len() > 1 && v[0] == 0 { &v[1..] } else { v };
    if v.len() > sz {
        return None;
    }
    let mut out = alloc::vec![0u8; sz];
    out[sz - v.len()..].copy_from_slice(v);
    Some(out)
}

pub fn selftest() -> bool {
    // FIPS 186-4 P-256 SHA-256 verification vector.
    let qx = [
        0x1c, 0xcb, 0xe9, 0x1c, 0x07, 0x5f, 0xc7, 0xf4, 0xf0, 0x33, 0xbf, 0xa2, 0x48, 0xdb, 0x8f,
        0xcc, 0xd3, 0x56, 0x5d, 0xe9, 0x4b, 0xbf, 0xb1, 0x2f, 0x3c, 0x59, 0xff, 0x46, 0xc2, 0x71,
        0xbf, 0x83,
    ];
    let qy = [
        0xce, 0x40, 0x14, 0xc6, 0x88, 0x11, 0xf9, 0xa2, 0x1a, 0x1f, 0xdb, 0x2c, 0x0e, 0x61, 0x13,
        0xe0, 0x6d, 0xb7, 0xca, 0x93, 0xb7, 0x40, 0x4e, 0x78, 0xdc, 0x7c, 0xcd, 0x5c, 0xa8, 0x9a,
        0x4c, 0xa9,
    ];
    let hash = [
        0x44, 0xac, 0xf6, 0xb7, 0xe3, 0x6c, 0x13, 0x42, 0xc2, 0xc5, 0x89, 0x72, 0x04, 0xfe, 0x09,
        0x50, 0x4e, 0x1e, 0x2e, 0xfb, 0x1a, 0x90, 0x03, 0x77, 0xdb, 0xc4, 0xe7, 0xa6, 0xa1, 0x33,
        0xec, 0x56,
    ];
    let r = [
        0xf3, 0xac, 0x80, 0x61, 0xb5, 0x14, 0x79, 0x5b, 0x88, 0x43, 0xe3, 0xd6, 0x62, 0x95, 0x27,
        0xed, 0x2a, 0xfd, 0x6b, 0x1f, 0x6a, 0x55, 0x5a, 0x7a, 0xca, 0xbb, 0x5e, 0x6f, 0x79, 0xc8,
        0xc2, 0xac,
    ];
    let s = [
        0x8b, 0xf7, 0x78, 0x19, 0xca, 0x05, 0xa6, 0xb2, 0x78, 0x6c, 0x76, 0x26, 0x2b, 0xf7, 0x37,
        0x1c, 0xef, 0x97, 0xb2, 0x18, 0xe9, 0x6f, 0x17, 0x5a, 0x3c, 0xcd, 0xda, 0x2a, 0xcc, 0x05,
        0x89, 0x03,
    ];

    let mut pubkey = alloc::vec![0x04u8];
    pubkey.extend_from_slice(&qx);
    pubkey.extend_from_slice(&qy);

    if !verify(Nist::P256, &pubkey, &hash, &r, &s) {
        return false;
    }
    // A tampered hash must be rejected -- a verifier that accepts everything
    // passes the positive test perfectly.
    let mut bad = hash;
    bad[0] ^= 1;
    !verify(Nist::P256, &pubkey, &bad, &r, &s)
}
