//! X25519 key agreement, RFC 7748.
//!
//! Field elements are five 51-bit limbs over 2^255 - 19. That representation
//! exists because it leaves slack: a product of two limbs is at most about
//! 2^102, and the largest sum in a multiply is under 2^111, so every
//! intermediate fits in a u128 with no carry chain that depends on the values.
//! A radix-2^64 representation would need carries whose count varies with the
//! data, and variable work on secret data is how a key leaks.
//!
//! The scalar multiply is a Montgomery ladder: every bit of the scalar does
//! exactly the same arithmetic, and which of the two points is which is
//! decided by a conditional swap implemented as a mask rather than a branch.
//! Both properties are the reason to use the ladder at all.
//!
//! ### What this is and is not
//!
//! It is a correct implementation of the RFC, checked against its test
//! vectors. It is not hardened against an attacker who can measure this
//! machine's power draw or electromagnetic emissions, and no pure-Rust
//! implementation is. It is constant-time with respect to instruction
//! sequence and memory access, which is the threat model that matters for a
//! network peer.

/// A field element, five limbs of 51 bits, little-endian.
type Fe = [u64; 5];

const MASK: u64 = (1 << 51) - 1;

fn fe_zero() -> Fe {
    [0; 5]
}
fn fe_one() -> Fe {
    [1, 0, 0, 0, 0]
}

fn fe_add(a: &Fe, b: &Fe) -> Fe {
    let mut r = [0u64; 5];
    for i in 0..5 {
        r[i] = a[i] + b[i];
    }
    r
}

/// a - b, computed as a + 2p - b so the result never goes negative.
///
/// 2p in this representation is [2^52 - 38, 2^52 - 2, 2^52 - 2, 2^52 - 2,
/// 2^52 - 2]; the first limb differs because p is 2^255 - 19 rather than a
/// clean power of two.
fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    let mut r = [0u64; 5];
    r[0] = a[0] + 0xFFFFFFFFFFFDA - b[0];
    for i in 1..5 {
        r[i] = a[i] + 0xFFFFFFFFFFFFE - b[i];
    }
    carry(&mut r);
    r
}

fn carry(r: &mut Fe) {
    let mut c = 0u64;
    for i in 0..5 {
        r[i] += c;
        c = r[i] >> 51;
        r[i] &= MASK;
    }
    // The carry out of the top limb re-enters at the bottom multiplied by 19,
    // because 2^255 = 19 mod p.
    r[0] += c * 19;
    c = r[0] >> 51;
    r[0] &= MASK;
    r[1] += c;
}

fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let m = |x: u64, y: u64| -> u128 { x as u128 * y as u128 };
    // Terms that would land above limb 4 are folded down by 19 as they are
    // formed, which is the whole trick of this representation.
    let d0 = m(a[0], b[0])
        + 19 * (m(a[1], b[4]) + m(a[2], b[3]) + m(a[3], b[2]) + m(a[4], b[1]));
    let d1 = m(a[0], b[1]) + m(a[1], b[0]) + 19 * (m(a[2], b[4]) + m(a[3], b[3]) + m(a[4], b[2]));
    let d2 = m(a[0], b[2]) + m(a[1], b[1]) + m(a[2], b[0]) + 19 * (m(a[3], b[4]) + m(a[4], b[3]));
    let d3 =
        m(a[0], b[3]) + m(a[1], b[2]) + m(a[2], b[1]) + m(a[3], b[0]) + 19 * m(a[4], b[4]);
    let d4 = m(a[0], b[4]) + m(a[1], b[3]) + m(a[2], b[2]) + m(a[3], b[1]) + m(a[4], b[0]);

    let mut r = [0u64; 5];
    let mut c: u128;
    r[0] = (d0 as u64) & MASK;
    c = d0 >> 51;
    let d1 = d1 + c;
    r[1] = (d1 as u64) & MASK;
    c = d1 >> 51;
    let d2 = d2 + c;
    r[2] = (d2 as u64) & MASK;
    c = d2 >> 51;
    let d3 = d3 + c;
    r[3] = (d3 as u64) & MASK;
    c = d3 >> 51;
    let d4 = d4 + c;
    r[4] = (d4 as u64) & MASK;
    c = d4 >> 51;
    r[0] += (c as u64) * 19;
    let cc = r[0] >> 51;
    r[0] &= MASK;
    r[1] += cc;
    r
}

fn fe_sq(a: &Fe) -> Fe {
    fe_mul(a, a)
}

fn fe_mul121666(a: &Fe) -> Fe {
    let mut r = [0u64; 5];
    let mut c: u128 = 0;
    for i in 0..5 {
        let t = a[i] as u128 * 121666 + c;
        r[i] = (t as u64) & MASK;
        c = t >> 51;
    }
    r[0] += (c as u64) * 19;
    let cc = r[0] >> 51;
    r[0] &= MASK;
    r[1] += cc;
    r
}

/// Swap a and b if `swap` is 1, using a mask so the timing does not depend on
/// which way it went.
fn cswap(swap: u64, a: &mut Fe, b: &mut Fe) {
    let mask = 0u64.wrapping_sub(swap);
    for i in 0..5 {
        let t = mask & (a[i] ^ b[i]);
        a[i] ^= t;
        b[i] ^= t;
    }
}

/// Inversion by Fermat: a^(p-2). The addition chain is the standard one from
/// the reference implementation -- 254 squarings and 11 multiplies.
fn fe_invert(a: &Fe) -> Fe {
    let z1 = *a;
    let z2 = fe_sq(&z1);
    let z8 = fe_sq(&fe_sq(&z2));
    let z9 = fe_mul(&z8, &z1);
    let z11 = fe_mul(&z9, &z2);
    let z22 = fe_sq(&z11);
    let z_5_0 = fe_mul(&z22, &z9);

    let mut t = fe_sq(&z_5_0);
    for _ in 1..5 {
        t = fe_sq(&t);
    }
    let z_10_0 = fe_mul(&t, &z_5_0);

    let mut t = fe_sq(&z_10_0);
    for _ in 1..10 {
        t = fe_sq(&t);
    }
    let z_20_0 = fe_mul(&t, &z_10_0);

    let mut t = fe_sq(&z_20_0);
    for _ in 1..20 {
        t = fe_sq(&t);
    }
    let z_40_0 = fe_mul(&t, &z_20_0);

    let mut t = fe_sq(&z_40_0);
    for _ in 1..10 {
        t = fe_sq(&t);
    }
    let z_50_0 = fe_mul(&t, &z_10_0);

    let mut t = fe_sq(&z_50_0);
    for _ in 1..50 {
        t = fe_sq(&t);
    }
    let z_100_0 = fe_mul(&t, &z_50_0);

    let mut t = fe_sq(&z_100_0);
    for _ in 1..100 {
        t = fe_sq(&t);
    }
    let z_200_0 = fe_mul(&t, &z_100_0);

    let mut t = fe_sq(&z_200_0);
    for _ in 1..50 {
        t = fe_sq(&t);
    }
    let z_250_0 = fe_mul(&t, &z_50_0);

    let mut t = fe_sq(&z_250_0);
    for _ in 1..5 {
        t = fe_sq(&t);
    }
    fe_mul(&t, &z11)
}

fn fe_from_bytes(b: &[u8; 32]) -> Fe {
    let load = |i: usize| -> u64 {
        let mut v = 0u64;
        for k in 0..8 {
            if i + k < 32 {
                v |= (b[i + k] as u64) << (8 * k);
            }
        }
        v
    };
    let mut r = [0u64; 5];
    r[0] = load(0) & MASK;
    r[1] = (load(6) >> 3) & MASK;
    r[2] = (load(12) >> 6) & MASK;
    r[3] = (load(19) >> 1) & MASK;
    // The top bit of the last byte is ignored, as RFC 7748 requires.
    r[4] = (load(24) >> 12) & MASK;
    r
}

fn fe_to_bytes(a: &Fe) -> [u8; 32] {
    let mut t = *a;
    carry(&mut t);
    carry(&mut t);
    carry(&mut t);

    // Conditionally subtract p once, branchlessly: the value may be in
    // [p, 2p) after carrying and must come out fully reduced.
    let mut q = (t[0] + 19) >> 51;
    q = (t[1] + q) >> 51;
    q = (t[2] + q) >> 51;
    q = (t[3] + q) >> 51;
    q = (t[4] + q) >> 51;
    t[0] += 19 * q;
    let mut c = t[0] >> 51;
    t[0] &= MASK;
    for i in 1..5 {
        t[i] += c;
        c = t[i] >> 51;
        t[i] &= MASK;
    }
    t[4] &= MASK;

    let mut out = [0u8; 32];
    let words = [
        t[0] | (t[1] << 51),
        (t[1] >> 13) | (t[2] << 38),
        (t[2] >> 26) | (t[3] << 25),
        (t[3] >> 39) | (t[4] << 12),
    ];
    for i in 0..4 {
        out[i * 8..i * 8 + 8].copy_from_slice(&words[i].to_le_bytes());
    }
    out
}

/// The core: scalar * point on Curve25519's x-coordinate.
pub fn scalarmult(scalar: &[u8; 32], point: &[u8; 32]) -> [u8; 32] {
    // Clamping, per RFC 7748: clear the low three bits so the scalar is a
    // multiple of the cofactor, and force bit 254 so the ladder always runs
    // the same number of iterations regardless of the key.
    let mut e = *scalar;
    e[0] &= 248;
    e[31] &= 127;
    e[31] |= 64;

    let x1 = fe_from_bytes(point);
    let mut x2 = fe_one();
    let mut z2 = fe_zero();
    let mut x3 = x1;
    let mut z3 = fe_one();
    let mut swap = 0u64;

    for pos in (0..255).rev() {
        let bit = ((e[pos / 8] >> (pos % 8)) & 1) as u64;
        swap ^= bit;
        cswap(swap, &mut x2, &mut x3);
        cswap(swap, &mut z2, &mut z3);
        swap = bit;

        let a = fe_add(&x2, &z2);
        let aa = fe_sq(&a);
        let b = fe_sub(&x2, &z2);
        let bb = fe_sq(&b);
        let e_ = fe_sub(&aa, &bb);
        let c = fe_add(&x3, &z3);
        let d = fe_sub(&x3, &z3);
        let da = fe_mul(&d, &a);
        let cb = fe_mul(&c, &b);
        x3 = fe_sq(&fe_add(&da, &cb));
        z3 = fe_mul(&x1, &fe_sq(&fe_sub(&da, &cb)));
        x2 = fe_mul(&aa, &bb);
        z2 = fe_mul(&e_, &fe_add(&bb, &fe_mul121666(&e_)));
    }

    cswap(swap, &mut x2, &mut x3);
    cswap(swap, &mut z2, &mut z3);
    fe_to_bytes(&fe_mul(&x2, &fe_invert(&z2)))
}

const BASE: [u8; 32] = {
    let mut b = [0u8; 32];
    b[0] = 9;
    b
};

pub fn public_key(secret: &[u8; 32]) -> [u8; 32] {
    scalarmult(secret, &BASE)
}

/// The shared secret. All-zero means the peer sent a low-order point, which
/// forces the result whatever our key is; RFC 7748 says a key agreement may
/// reject it, and TLS 1.3 says it must.
pub fn shared_secret(secret: &[u8; 32], peer: &[u8; 32]) -> Option<[u8; 32]> {
    let out = scalarmult(secret, peer);
    if out.iter().all(|b| *b == 0) {
        None
    } else {
        Some(out)
    }
}

pub fn selftest() -> bool {
    // RFC 7748 section 5.2, first vector.
    let scalar: [u8; 32] = [
        0xa5, 0x46, 0xe3, 0x6b, 0xf0, 0x52, 0x7c, 0x9d, 0x3b, 0x16, 0x15, 0x4b, 0x82, 0x46, 0x5e,
        0xdd, 0x62, 0x14, 0x4c, 0x0a, 0xc1, 0xfc, 0x5a, 0x18, 0x50, 0x6a, 0x22, 0x44, 0xba, 0x44,
        0x9a, 0xc4,
    ];
    let point: [u8; 32] = [
        0xe6, 0xdb, 0x68, 0x67, 0x58, 0x30, 0x30, 0xdb, 0x35, 0x94, 0xc1, 0xa4, 0x24, 0xb1, 0x5f,
        0x7c, 0x72, 0x66, 0x24, 0xec, 0x26, 0xb3, 0x35, 0x3b, 0x10, 0xa9, 0x03, 0xa6, 0xd0, 0xab,
        0x1c, 0x4c,
    ];
    let want: [u8; 32] = [
        0xc3, 0xda, 0x55, 0x37, 0x9d, 0xe9, 0xc6, 0x90, 0x8e, 0x94, 0xea, 0x4d, 0xf2, 0x8d, 0x08,
        0x4f, 0x32, 0xec, 0xcf, 0x03, 0x49, 0x1c, 0x71, 0xf7, 0x54, 0xb4, 0x07, 0x55, 0x77, 0xa2,
        0x85, 0x52,
    ];
    if scalarmult(&scalar, &point) != want {
        return false;
    }

    // The base-point vector, which exercises public_key.
    let a_sec: [u8; 32] = [
        0x77, 0x07, 0x6d, 0x0a, 0x73, 0x18, 0xa5, 0x7d, 0x3c, 0x16, 0xc1, 0x72, 0x51, 0xb2, 0x66,
        0x45, 0xdf, 0x4c, 0x2f, 0x87, 0xeb, 0xc0, 0x99, 0x2a, 0xb1, 0x77, 0xfb, 0xa5, 0x1d, 0xb9,
        0x2c, 0x2a,
    ];
    let a_pub: [u8; 32] = [
        0x85, 0x20, 0xf0, 0x09, 0x89, 0x30, 0xa7, 0x54, 0x74, 0x8b, 0x7d, 0xdc, 0xb4, 0x3e, 0xf7,
        0x5a, 0x0d, 0xbf, 0x3a, 0x0d, 0x26, 0x38, 0x1a, 0xf4, 0xeb, 0xa4, 0xa9, 0x8e, 0xaa, 0x9b,
        0x4e, 0x6a,
    ];
    if public_key(&a_sec) != a_pub {
        return false;
    }

    // And that both sides of an exchange agree, which is the property the
    // vectors alone do not check.
    let b_sec: [u8; 32] = [
        0x5d, 0xab, 0x08, 0x7e, 0x62, 0x4a, 0x8a, 0x4b, 0x79, 0xe1, 0x7f, 0x8b, 0x83, 0x80, 0x0e,
        0xe6, 0x6f, 0x3b, 0xb1, 0x29, 0x26, 0x18, 0xb6, 0xfd, 0x1c, 0x2f, 0x8b, 0x27, 0xff, 0x88,
        0xe0, 0xeb,
    ];
    let b_pub = public_key(&b_sec);
    scalarmult(&a_sec, &b_pub) == scalarmult(&b_sec, &a_pub)
}
