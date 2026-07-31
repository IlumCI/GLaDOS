//! Variable-width unsigned integers, enough for RSA and P-256.
//!
//! One implementation serves both because both want the same three things:
//! modular multiplication, modular exponentiation, and comparison. RSA works
//! at 2048 or 4096 bits and P-256 at 256; making the width a runtime property
//! costs a little speed and saves a second implementation with a second set of
//! bugs.
//!
//! ### Montgomery form, and why
//!
//! Reducing a 512-bit product modulo a 256-bit number by long division is slow
//! and awkward to write correctly. Montgomery multiplication replaces the
//! division with shifts: it computes `a*b*R^-1 mod m` where R is 2^(64*limbs),
//! so if values are kept pre-multiplied by R the extra factor cancels and
//! every step is multiply-and-shift. The conversion in and out costs two extra
//! multiplications, which is nothing next to a modexp.
//!
//! It requires an odd modulus. Both an RSA modulus and a prime are odd, so the
//! restriction never bites here.
//!
//! ### Constant time
//!
//! This is **not** constant time, and does not need to be. Every use is
//! verification: an RSA public-key operation and an ECDSA signature check,
//! both over data the attacker already has. There is no secret to leak. If
//! this is ever used for a private-key operation -- signing, or RSA
//! decryption -- that assumption breaks and the code must be revisited.

use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Big {
    /// Little-endian limbs. Trailing zero limbs are permitted; `trim` removes
    /// them where it matters.
    pub v: Vec<u64>,
}

impl Big {
    pub fn zero(limbs: usize) -> Self {
        Big { v: vec![0; limbs] }
    }

    pub fn from_u64(x: u64, limbs: usize) -> Self {
        let mut b = Big::zero(limbs.max(1));
        b.v[0] = x;
        b
    }

    /// Big-endian bytes, as every wire format in sight uses.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let limbs = (bytes.len() + 7) / 8;
        let mut v = vec![0u64; limbs.max(1)];
        for (i, b) in bytes.iter().rev().enumerate() {
            v[i / 8] |= (*b as u64) << (8 * (i % 8));
        }
        Big { v }
    }

    pub fn to_bytes(&self, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        for i in 0..len {
            let byte = (self.v.get(i / 8).copied().unwrap_or(0) >> (8 * (i % 8))) as u8;
            out[len - 1 - i] = byte;
        }
        out
    }

    pub fn limbs(&self) -> usize {
        self.v.len()
    }

    pub fn is_zero(&self) -> bool {
        self.v.iter().all(|x| *x == 0)
    }

    pub fn bit(&self, i: usize) -> bool {
        self.v.get(i / 64).map(|w| (w >> (i % 64)) & 1 == 1).unwrap_or(false)
    }

    pub fn bits(&self) -> usize {
        for i in (0..self.v.len()).rev() {
            if self.v[i] != 0 {
                return i * 64 + (64 - self.v[i].leading_zeros() as usize);
            }
        }
        0
    }

    fn resize(&self, n: usize) -> Big {
        let mut v = self.v.clone();
        v.resize(n, 0);
        Big { v }
    }

    pub fn cmp(&self, other: &Big) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        let n = self.v.len().max(other.v.len());
        for i in (0..n).rev() {
            let a = self.v.get(i).copied().unwrap_or(0);
            let b = other.v.get(i).copied().unwrap_or(0);
            if a != b {
                return if a < b { Ordering::Less } else { Ordering::Greater };
            }
        }
        Ordering::Equal
    }

    /// Returns the sum and the carry out.
    fn add_raw(&self, other: &Big) -> (Big, u64) {
        let n = self.v.len().max(other.v.len());
        let mut r = Big::zero(n);
        let mut carry = 0u128;
        for i in 0..n {
            let s = self.v.get(i).copied().unwrap_or(0) as u128
                + other.v.get(i).copied().unwrap_or(0) as u128
                + carry;
            r.v[i] = s as u64;
            carry = s >> 64;
        }
        (r, carry as u64)
    }

    /// Returns the difference and the borrow out.
    fn sub_raw(&self, other: &Big) -> (Big, u64) {
        let n = self.v.len().max(other.v.len());
        let mut r = Big::zero(n);
        let mut borrow = 0i128;
        for i in 0..n {
            let d = self.v.get(i).copied().unwrap_or(0) as i128
                - other.v.get(i).copied().unwrap_or(0) as i128
                - borrow;
            r.v[i] = d as u64;
            borrow = if d < 0 { 1 } else { 0 };
        }
        (r, borrow as u64)
    }

    pub fn add_mod(&self, other: &Big, m: &Big) -> Big {
        let (s, carry) = self.add_raw(other);
        // The carry matters: a + b can exceed the limb width even when both
        // are below m, and dropping it silently gives a wrong answer for
        // roughly one addition in every 2^64.
        if carry != 0 || s.cmp(m) != core::cmp::Ordering::Less {
            s.sub_raw(m).0
        } else {
            s
        }
    }

    pub fn sub_mod(&self, other: &Big, m: &Big) -> Big {
        let (d, borrow) = self.sub_raw(other);
        if borrow != 0 {
            d.add_raw(m).0
        } else {
            d
        }
    }

    /// Plain remainder by shift-and-subtract. Only used to normalise inputs,
    /// never in a loop that matters for speed.
    pub fn rem(&self, m: &Big) -> Big {
        if self.cmp(m) == core::cmp::Ordering::Less {
            return self.resize(m.v.len());
        }
        let n = m.v.len();
        let mut r = Big::zero(n);
        for i in (0..self.bits()).rev() {
            // r = r*2 + bit
            let mut carry = self.bit(i) as u64;
            for k in 0..n {
                let nv = (r.v[k] << 1) | carry;
                carry = r.v[k] >> 63;
                r.v[k] = nv;
            }
            // The shifted-out carry means r has exceeded the modulus width,
            // so a subtraction is owed regardless of the comparison.
            if carry != 0 || r.cmp(m) != core::cmp::Ordering::Less {
                r = r.sub_raw(m).0;
            }
        }
        r
    }
}

/// Montgomery parameters for a fixed odd modulus.
pub struct Mont {
    pub m: Big,
    /// -m^-1 mod 2^64.
    m_inv: u64,
    /// R^2 mod m, for converting into Montgomery form.
    r2: Big,
    n: usize,
}

impl Mont {
    pub fn new(m: &Big) -> Option<Mont> {
        if m.v.is_empty() || m.v[0] & 1 == 0 {
            return None; // Montgomery needs an odd modulus.
        }
        let n = m.v.len();

        // -m^-1 mod 2^64 by Newton iteration: the inverse doubles its correct
        // bit count each round, so five rounds cover 64 bits starting from a
        // 2-bit seed.
        let m0 = m.v[0];
        let mut inv: u64 = 1;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(inv)));
        }
        let m_inv = inv.wrapping_neg();

        // R mod m, then R^2 by doubling n*64 times. Computed rather than
        // tabulated so there is no constant to get wrong per modulus.
        let mut r = Big::zero(n + 1);
        r.v[n] = 1;
        let r_mod = r.rem(m);
        let mut r2 = r_mod.clone();
        for _ in 0..(n * 64) {
            r2 = r2.add_mod(&r2, m);
        }

        Some(Mont { m: m.clone(), m_inv, r2, n })
    }

    /// CIOS Montgomery multiplication: a*b*R^-1 mod m.
    pub fn mul(&self, a: &Big, b: &Big) -> Big {
        let n = self.n;
        // A stack buffer rather than a Vec: this is the hottest function in
        // the system by a wide margin -- a single ECDSA verification calls it
        // over ten thousand times -- and an allocation per call was measurable
        // heap churn for no reason. 66 limbs covers RSA-4096 with room.
        let mut stack = [0u64; 66];
        let mut heap;
        let t: &mut [u64] = if n + 2 <= stack.len() {
            &mut stack[..n + 2]
        } else {
            heap = vec![0u64; n + 2];
            &mut heap
        };
        for i in 0..n {
            let bi = b.v.get(i).copied().unwrap_or(0) as u128;
            let mut carry = 0u128;
            for j in 0..n {
                let s = t[j] as u128 + a.v.get(j).copied().unwrap_or(0) as u128 * bi + carry;
                t[j] = s as u64;
                carry = s >> 64;
            }
            let s = t[n] as u128 + carry;
            t[n] = s as u64;
            t[n + 1] = (s >> 64) as u64;

            // Add a multiple of m chosen to clear the bottom limb, then shift.
            let u = (t[0].wrapping_mul(self.m_inv)) as u128;
            let mut carry = 0u128;
            for j in 0..n {
                let s = t[j] as u128 + self.m.v[j] as u128 * u + carry;
                t[j] = s as u64;
                carry = s >> 64;
            }
            let s = t[n] as u128 + carry;
            t[n] = s as u64;
            t[n + 1] += (s >> 64) as u64;

            for j in 0..=n {
                t[j] = t[j + 1];
            }
            t[n + 1] = 0;
        }

        let mut r = Big { v: t[..n].to_vec() };
        // One conditional subtraction: the CIOS result is below 2m.
        if t[n] != 0 || r.cmp(&self.m) != core::cmp::Ordering::Less {
            r = r.sub_raw(&self.m).0;
        }
        r
    }

    pub fn to_mont(&self, a: &Big) -> Big {
        let a = a.rem(&self.m);
        self.mul(&a, &self.r2)
    }

    pub fn from_mont(&self, a: &Big) -> Big {
        self.mul(a, &Big::from_u64(1, self.n))
    }

    /// Modular exponentiation, square-and-multiply, most significant bit first.
    pub fn pow(&self, base: &Big, exp: &Big) -> Big {
        let one = self.to_mont(&Big::from_u64(1, self.n));
        let b = self.to_mont(base);
        let mut acc = one;
        let bits = exp.bits();
        if bits == 0 {
            return self.from_mont(&acc);
        }
        for i in (0..bits).rev() {
            acc = self.mul(&acc, &acc);
            if exp.bit(i) {
                acc = self.mul(&acc, &b);
            }
        }
        self.from_mont(&acc)
    }

    /// Inverse by Fermat: a^(m-2) mod m. Valid only for prime m, which is how
    /// it is used -- ECDSA needs inverses modulo the curve order.
    pub fn inv_prime(&self, a: &Big) -> Big {
        let two = Big::from_u64(2, self.n);
        let e = self.m.sub_raw(&two).0;
        self.pow(a, &e)
    }
}

pub fn selftest() -> bool {
    // A modexp with known small values: 5^117 mod 19 == 1 (5 has order 9
    // modulo 19, and 117 is a multiple of 9).
    let m = Big::from_u64(19, 1);
    let mont = match Mont::new(&m) {
        None => return false,
        Some(x) => x,
    };
    let r = mont.pow(&Big::from_u64(5, 1), &Big::from_u64(117, 1));
    if r.v[0] != 1 {
        return false;
    }

    // 2^10 mod 1000 == 24, exercising a modulus that is not prime.
    let m = Big::from_u64(1000 + 1, 1); // odd modulus required
    let mont = Mont::new(&m).unwrap();
    let r = mont.pow(&Big::from_u64(2, 1), &Big::from_u64(10, 1));
    if r.v[0] != 1024 % 1001 {
        return false;
    }

    // Round-tripping a big-endian byte string.
    let bytes: [u8; 9] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x11];
    let b = Big::from_bytes(&bytes);
    if b.to_bytes(9) != bytes {
        return false;
    }

    // rem() against a value known to need many shift-subtract rounds.
    let big = Big::from_bytes(&[0xFF; 32]);
    let m = Big::from_bytes(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF,
    ]);
    let r = big.rem(&m);
    // 2^256-1 mod p256 == 2^224 - 2^192 - 2^96 + 4, checked by adding m back.
    let sum = r.add_mod(&m, &Big::from_bytes(&[0xFF; 33]));
    sum.cmp(&big) == core::cmp::Ordering::Equal
}
