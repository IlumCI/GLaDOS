//! Is this worker name an address, and if it is, does it check out?
//!
//! `boot.rs` argues that `worker` has no default because a default would mine
//! to whoever the default belongs to. A **typo** is the same failure arriving by
//! a different route and it is the more likely one: the operator pastes an
//! address into a file on read-only media, one character is wrong, and the image
//! is cut. The machine boots, connects, hashes all night and pays nobody.
//!
//! Every address form that matters here carries a checksum for exactly this
//! reason, so the question is answerable offline, before the first share:
//!
//! - **base58check** -- four bytes of double-SHA-256 over the payload, which is
//!   what the `1`/`3` Bitcoin and `L`/`M` Litecoin forms are.
//! - **bech32 and bech32m** -- a BCH code over the whole string, hrp included,
//!   which is why `bc1...` catches transpositions that base58 does not.
//! - **EIP-55** -- the *case* of an EVM address's hex digits is a 40-bit
//!   checksum over the Keccak-256 of its lowercase form.
//!
//! ### Four answers, because there are four facts
//!
//! A checker with two answers has to file "carries no checksum" under one of
//! them and both are wrong: as a pass it claims a verification that did not
//! happen, and as a failure it refuses an address that is perfectly valid. An
//! all-lowercase EVM address is exactly that case -- every wallet will accept
//! it and nothing on earth can tell whether it is the one the operator meant.
//!
//! And a plain worker name is a fourth fact rather than a failure. `pool`'s own
//! roster maps a name to an address, so a name is the *documented* arrangement
//! for a pool that keeps its own books; what is owed there is to say that
//! nothing here checked where it pays, not to refuse it.
//!
//! ### What is deliberately not checked
//!
//! Which coin it belongs to. A version byte says P2PKH on *some* chain, and the
//! set of chains sharing byte 0x00 is open, so a table of them would refuse a
//! valid address on the first coin nobody had heard of. The checksum is a
//! property of the string; the coin is a property of where it is sent, and the
//! pool is what knows that.
//!
//! Nor whether the address has ever been seen on a chain. That needs a node.

use alloc::string::String;
use alloc::vec::Vec;

/// Which form a name turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `1...` or `3...` and the Litecoin equivalents: base58check.
    Base58,
    /// `bc1...` and friends: bech32 for witness version 0, bech32m above it.
    Bech32,
    /// `0x` and forty hex digits.
    Evm,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Base58 => "base58check",
            Kind::Bech32 => "bech32",
            Kind::Evm => "EVM hex",
        }
    }
}

/// What could be established about where a name pays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Payout {
    /// A known form whose own checksum holds. A typo in it would have shown.
    Checked(Kind),
    /// A known form that carries no checksum at all -- an EVM address written
    /// in one case throughout. The shape is right and where it pays cannot be
    /// verified by anything that is not a chain.
    Unchecked(Kind),
    /// A known form whose checksum fails. This is not anybody's address.
    Broken(Kind),
    /// Not a form anything here knows: a worker name, which the pool's roster
    /// maps to an address. Allowed, and unverifiable from here.
    Name,
}

impl Payout {
    /// Whether a miner should start on this. Only a broken checksum says no --
    /// the other three are all things somebody may legitimately have meant.
    pub fn may_mine(self) -> bool {
        !matches!(self, Payout::Broken(_))
    }

    /// One clause for the boot line, in the register the rest of that line uses.
    pub fn say(self) -> String {
        use alloc::format;
        match self {
            Payout::Checked(k) => format!("payout address checks out ({})", k.as_str()),
            Payout::Unchecked(k) => {
                format!("payout address is a well-formed {} carrying no checksum", k.as_str())
            }
            Payout::Broken(k) => {
                format!("payout address is a {} whose checksum fails -- this pays nobody", k.as_str())
            }
            Payout::Name => String::from("worker is a name, so the pool's roster decides where this pays"),
        }
    }
}

/// Judge a worker name.
///
/// **The name is split at the first `.` and the head is judged.** `address.rig`
/// is the convention at every multi-rig venue, so judging the whole string would
/// answer `Name` for every address that had a rig suffix -- which is most of
/// them, and is the one input where being wrong costs the most.
pub fn judge(worker: &str) -> Payout {
    let head = match worker.split_once('.') {
        Some((h, _)) => h,
        None => worker,
    };
    if let Some(p) = evm(head) {
        return p;
    }
    if let Some(p) = bech32(head) {
        return p;
    }
    if let Some(p) = base58(head) {
        return p;
    }
    Payout::Name
}

// --- EVM -------------------------------------------------------------------

/// `0x` and forty hex digits. Mixed case is EIP-55 and is checked; one case
/// throughout carries no information and is reported as carrying none.
fn evm(s: &str) -> Option<Payout> {
    let body = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    if body.len() != 40 || !body.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    // Only letters carry the checksum -- a digit has no case -- so an address of
    // forty digits is unverifiable however it is written, and an address whose
    // letters are all one case is the operator's wallet having handed over the
    // lowercase form.
    let letters: Vec<u8> = body.bytes().filter(|b| b.is_ascii_alphabetic()).collect();
    if letters.is_empty()
        || letters.iter().all(|b| b.is_ascii_lowercase())
        || letters.iter().all(|b| b.is_ascii_uppercase())
    {
        return Some(Payout::Unchecked(Kind::Evm));
    }

    let lower: Vec<u8> = body.bytes().map(|b| b.to_ascii_lowercase()).collect();
    let h = crate::crypto::keccak::keccak256(&lower);
    // Nibble i of the digest decides the case of character i: four or above is
    // uppercase. Forty nibbles, which is where the forty bits come from.
    for (i, c) in body.bytes().enumerate() {
        if !c.is_ascii_alphabetic() {
            continue;
        }
        let nibble = if i % 2 == 0 { h[i / 2] >> 4 } else { h[i / 2] & 0x0f };
        let want_upper = nibble >= 8;
        if want_upper != c.is_ascii_uppercase() {
            return Some(Payout::Broken(Kind::Evm));
        }
    }
    Some(Payout::Checked(Kind::Evm))
}

// --- bech32 ----------------------------------------------------------------

const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// The human-readable parts this recognises.
///
/// **A closed list, and not because the code cares.** The polynomial would
/// accept any hrp, and accepting any would mean reading a worker name containing
/// a `1` as a malformed bech32 address and refusing the plan. Recognition has to
/// be narrow precisely because the verdict is a refusal -- `repair::ACTIONS`'s
/// argument about an allowlist, arriving on a string format.
const HRPS: &[&str] = &["bc", "tb", "bcrt", "ltc", "tltc", "rltc"];

fn polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [0x3b6a_57b2, 0x2650_8e6d, 0x1ea1_19fa, 0x3d42_33dd, 0x2a14_62b3];
    let mut chk: u32 = 1;
    for &v in values {
        let top = chk >> 25;
        chk = ((chk & 0x1ff_ffff) << 5) ^ v as u32;
        for (i, g) in GEN.iter().enumerate() {
            if (top >> i) & 1 == 1 {
                chk ^= g;
            }
        }
    }
    chk
}

fn bech32(s: &str) -> Option<Payout> {
    // Mixed case is invalid in bech32 by specification, because the checksum is
    // defined over one case. Refusing to *recognise* it rather than calling it
    // broken: a mixed-case string is much more likely to be somebody's worker
    // name than a mangled address.
    let lower = s.to_ascii_lowercase();
    if s != lower && s != s.to_ascii_uppercase() {
        return None;
    }
    if lower.len() < 8 || lower.len() > 90 {
        return None;
    }
    let sep = lower.rfind('1')?;
    let hrp = &lower[..sep];
    if !HRPS.contains(&hrp) {
        return None;
    }
    let data = &lower[sep + 1..];
    // Six characters of checksum plus at least one of payload.
    if data.len() < 7 {
        return None;
    }

    let mut values: Vec<u8> = Vec::with_capacity(hrp.len() * 2 + 1 + data.len());
    for b in hrp.bytes() {
        values.push(b >> 5);
    }
    values.push(0);
    for b in hrp.bytes() {
        values.push(b & 31);
    }
    let mut payload: Vec<u8> = Vec::with_capacity(data.len());
    for b in data.bytes() {
        let v = CHARSET.iter().position(|&c| c == b)? as u8;
        payload.push(v);
        values.push(v);
    }

    // The witness version picks which constant applies: bech32 for 0, bech32m
    // for 1 and above. Getting this backwards accepts every taproot address as
    // broken and every v0 address as broken, which is a checker that refuses
    // everything and looks like a strict one.
    let want = if payload[0] == 0 { 1u32 } else { 0x2bc8_30a3 };
    if polymod(&values) == want {
        Some(Payout::Checked(Kind::Bech32))
    } else {
        Some(Payout::Broken(Kind::Bech32))
    }
}

// --- base58check -----------------------------------------------------------

const B58: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

fn base58(s: &str) -> Option<Payout> {
    // A legacy address is 25 bytes -- version, twenty of hash, four of checksum
    // -- which spells as 26 to 35 characters. Both bounds are needed: the length
    // is what stops a short worker name of base58-legal characters being read as
    // an address at all.
    if s.len() < 26 || s.len() > 35 {
        return None;
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(32);
    for c in s.bytes() {
        let d = B58.iter().position(|&x| x == c)? as u32;
        // Multiply the accumulated big-endian number by 58 and add the digit.
        let mut carry = d;
        for b in bytes.iter_mut().rev() {
            let v = (*b as u32) * 58 + carry;
            *b = (v & 0xff) as u8;
            carry = v >> 8;
        }
        while carry > 0 {
            bytes.insert(0, (carry & 0xff) as u8);
            carry >>= 8;
        }
    }
    // Leading '1's are leading zero bytes and the arithmetic above cannot
    // produce them, so they are restored by counting. Without this a version
    // byte of zero -- which is every `1...` address -- is simply absent and the
    // checksum is computed over the wrong payload.
    for c in s.bytes() {
        if c == b'1' {
            bytes.insert(0, 0);
        } else {
            break;
        }
    }
    if bytes.len() != 25 {
        return None;
    }
    let sum = super::hash::sha256d(&bytes[..21]);
    if sum[..4] == bytes[21..] {
        Some(Payout::Checked(Kind::Base58))
    } else {
        Some(Payout::Broken(Kind::Base58))
    }
}

/// Claims. No network, no pool, and every address below is a published one or a
/// one-character edit of it.
pub fn checks() -> Vec<(bool, String)> {
    let mut out: Vec<(bool, String)> = Vec::new();
    let mut ok = |c: bool, w: &str| out.push((c, String::from(w)));

    // Satoshi's, the most published base58 address there is.
    let genesis = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";
    ok(judge(genesis) == Payout::Checked(Kind::Base58), "a real P2PKH address checks out");
    // One character moved. The checksum is four bytes, so this is the case the
    // whole encoding exists for.
    ok(
        judge("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb") == Payout::Broken(Kind::Base58),
        "and one character wrong in it is caught rather than mined to",
    );
    ok(
        judge("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy") == Payout::Checked(Kind::Base58),
        "a P2SH address checks out too",
    );
    // The leading-zero restoration: without it this address's version byte is
    // missing and the checksum is taken over twenty bytes of the wrong payload.
    ok(
        judge(genesis) != Payout::Broken(Kind::Base58),
        "a leading '1' is a zero byte, not a character to drop",
    );

    // BIP-173's own test vector, and BIP-350's for bech32m.
    ok(
        judge("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4") == Payout::Checked(Kind::Bech32),
        "a segwit v0 address checks out under bech32",
    );
    ok(
        judge("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0")
            == Payout::Checked(Kind::Bech32),
        "and a taproot address under bech32m, which is the other constant",
    );
    ok(
        judge("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5") == Payout::Broken(Kind::Bech32),
        "a bech32 address with one character wrong is caught",
    );
    // The two constants are not interchangeable, and swapping them would make a
    // checker that refuses every valid address while looking strict.
    ok(
        judge("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0")
            != Payout::Broken(Kind::Bech32),
        "so bech32m is not judged against bech32's constant",
    );

    // EIP-55's own vectors.
    ok(
        judge("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed") == Payout::Checked(Kind::Evm),
        "an EIP-55 address checks out",
    );
    ok(
        judge("0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359") == Payout::Checked(Kind::Evm),
        "and so does the second published one",
    );
    ok(
        judge("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1Beaed") == Payout::Broken(Kind::Evm),
        "one letter's case wrong in an EIP-55 address is caught",
    );
    ok(
        judge("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed") == Payout::Unchecked(Kind::Evm),
        "an all-lowercase EVM address is well-formed and carries no checksum",
    );
    ok(
        judge("0x5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED") == Payout::Unchecked(Kind::Evm),
        "and so is an all-uppercase one",
    );
    ok(judge("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAe") == Payout::Name, "thirty-nine hex digits is not an address");

    // The rig suffix, which is the input this gets wrong most expensively:
    // judging the whole string answers Name for every address that has one.
    ok(
        judge("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed.rig1") == Payout::Checked(Kind::Evm),
        "a rig suffix is split off rather than making the address unrecognisable",
    );
    ok(
        judge("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.gf63") == Payout::Checked(Kind::Base58),
        "and the same for a base58 address",
    );

    // Names, which must survive. Each of these is a shape somebody would
    // plausibly type, and each would be a refusal if recognition were loose.
    ok(judge("rig1") == Payout::Name, "a plain worker name is a name");
    ok(judge("gf63-miner") == Payout::Name, "and so is one with a hyphen");
    ok(judge("1rig") == Payout::Name, "a short name starting with '1' is not a base58 address");
    ok(judge("bc-alpha1") == Payout::Name, "and a name containing '1' is not a bech32 address");
    ok(
        judge("MixedCaseWorkerName123456789") == Payout::Name,
        "a 27-character mixed-case name is not read as an address",
    );
    ok(judge("") == Payout::Name, "and an empty name is a name rather than a panic");

    // The gate itself: exactly one of the four says no.
    ok(Payout::Checked(Kind::Evm).may_mine(), "a checked address may mine");
    ok(Payout::Unchecked(Kind::Evm).may_mine(), "so may one with no checksum to read");
    ok(Payout::Name.may_mine(), "so may a worker name");
    ok(!Payout::Broken(Kind::Evm).may_mine(), "and a broken checksum is the one that does not");

    out
}
