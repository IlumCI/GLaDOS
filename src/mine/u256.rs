//! A 256-bit unsigned integer, with only what mining targets need.
//!
//! Not a general bignum. Five operations, because a difficulty target is
//! produced two ways and compared once, and nothing here ever adds or
//! subtracts. `crypto::bigint` exists and is the wrong tool: it is Montgomery
//! modular arithmetic for ECDSA, allocates, and has no ordering.
//!
//! **Limbs are most-significant first, and that is load-bearing.** It makes
//! `Ord` derivable: comparing `[u32; 8]` element-wise from index 0 is exactly
//! numeric comparison when index 0 is the high word. Storing them the other way
//! round would compile identically and order wrongly, which is the failure this
//! module exists to make impossible rather than unlikely.

/// A 256-bit unsigned integer. `w[0]` is the most significant word.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct U256 {
    pub w: [u32; 8],
}

impl U256 {
    pub const ZERO: U256 = U256 { w: [0; 8] };
    /// The easiest target there is: every hash meets it.
    pub const MAX: U256 = U256 { w: [u32::MAX; 8] };

    pub fn from_be_bytes(b: &[u8; 32]) -> U256 {
        let mut w = [0u32; 8];
        for (i, word) in w.iter_mut().enumerate() {
            *word = u32::from_be_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
        }
        U256 { w }
    }

    pub fn to_be_bytes(self) -> [u8; 32] {
        let mut b = [0u8; 32];
        for i in 0..8 {
            b[i * 4..i * 4 + 4].copy_from_slice(&self.w[i].to_be_bytes());
        }
        b
    }

    pub fn is_zero(self) -> bool {
        self.w.iter().all(|&x| x == 0)
    }

    /// Multiply by a small factor. `None` on overflow rather than wrapping.
    ///
    /// A wrapped target is a *smaller* number, so it would silently reject
    /// every share the miner ever found and look exactly like a miner that is
    /// not working. Overflow has to be a refusal.
    pub fn mul_u32(self, m: u32) -> Option<U256> {
        let mut out = [0u32; 8];
        let mut carry: u64 = 0;
        for i in (0..8).rev() {
            let v = self.w[i] as u64 * m as u64 + carry;
            out[i] = v as u32;
            carry = v >> 32;
        }
        if carry == 0 {
            Some(U256 { w: out })
        } else {
            None
        }
    }

    /// Divide by a 64-bit divisor. `None` on division by zero.
    ///
    /// Schoolbook long division a word at a time. The running remainder is
    /// always below the divisor, so `rem << 32 | word` is under 2^96 and fits
    /// a `u128`, and the quotient word is under 2^32 by the same bound.
    pub fn div_u64(self, d: u64) -> Option<U256> {
        if d == 0 {
            return None;
        }
        let mut out = [0u32; 8];
        let mut rem: u128 = 0;
        for i in 0..8 {
            let cur = (rem << 32) | self.w[i] as u128;
            out[i] = (cur / d as u128) as u32;
            rem = cur % d as u128;
        }
        Some(U256 { w: out })
    }

    /// The target `nbits` encodes: `mantissa * 256^(exponent - 3)`.
    ///
    /// This is Bitcoin's compact form and every coin that inherited its header
    /// uses it. The mantissa is three bytes placed so its top byte lands at
    /// offset `32 - exponent`, which is the whole of the arithmetic once you
    /// see it as placement rather than as a shift.
    ///
    /// Refuses an exponent that would push the mantissa off either end instead
    /// of clamping. A clamped target is a legal-looking number that is not the
    /// one the network asked for, and it would be discovered as a rejected
    /// share with a confusing reason.
    pub fn from_nbits(nbits: u32) -> Option<U256> {
        let mant = nbits & 0x00ff_ffff;
        let exp = (nbits >> 24) as usize;
        if !(3..=32).contains(&exp) {
            return None;
        }
        let off = 32 - exp;
        let mut b = [0u8; 32];
        b[off] = (mant >> 16) as u8;
        b[off + 1] = (mant >> 8) as u8;
        b[off + 2] = mant as u8;
        Some(U256::from_be_bytes(&b))
    }
}

/// The difficulty-1 target, `0x00000000FFFF0000...0000`.
///
/// Derived from `nbits` rather than written out, so the two producers of a
/// target cannot disagree about this one value -- and a claim checks that the
/// literal encoding and the constant agree, which is the only way to catch
/// `from_nbits` being wrong in a way that happens to look plausible.
pub fn diff1() -> U256 {
    // 0x1d00ffff is the genesis block's own nbits.
    U256::from_nbits(0x1d00_ffff).unwrap_or(U256::ZERO)
}

/// The target for a share difficulty given as `mantissa / 10^scale`.
///
/// Pools send fractional difficulties routinely, and this kernel's JSON reader
/// truncates at the decimal point -- so the difficulty arrives here already
/// split rather than as an integer that lost its fraction. See
/// `stratum`'s decimal reader for why that split exists.
///
/// `target = diff1 * 10^scale / mantissa`. Multiplying first is what keeps the
/// precision: dividing first would floor to zero for every difficulty under
/// one, which is exactly the case the split was introduced to serve.
pub fn target_for(mantissa: u64, scale: u32) -> Option<U256> {
    if mantissa == 0 || scale > 8 {
        return None;
    }
    let mut t = diff1();
    // 10^8 exceeds u32 only above scale 9, and scale is capped at 8. Applied as
    // repeated small multiplications so each one can refuse on overflow.
    for _ in 0..scale {
        t = t.mul_u32(10)?;
    }
    t.div_u64(mantissa)
}
