//! Mining: block headers, targets, and the hash loop.
//!
//! Stage one is arithmetic and nothing else. No network, no task, no shell
//! verb -- the pure half first, because every mistake available here is silent.
//! A header with two bytes swapped hashes at full speed and is rejected by the
//! pool forever; a target compared by counting zeros accepts and rejects the
//! wrong shares; a midstate taken at the wrong offset produces digests for a
//! header that never existed. None of that faults.
//!
//! So the fixture is a real block, decomposed into the fields a pool actually
//! sends, and the claim is that reassembling them reproduces the header byte
//! for byte and hashes to the published id. That single claim pins the version
//! reversal, the prevhash word swap, the ntime and nbits reversals, the merkle
//! root's *non*-reversal, every field offset, and the double hash.
//!
//! `tools/algocheck.py` holds the same header and computes the same digest with
//! `hashlib`, which is the bargain `tokenizer.py --verify` makes: the reader is
//! deliberately not the writer.

pub mod algo;
pub mod blake2s;
pub mod boot;
pub mod client;
pub mod ev;
pub mod hash;
pub mod proto;
pub mod header;
pub mod neoscrypt;
pub mod stratum;
pub mod u256;
pub mod work;
pub mod yespower;

use alloc::vec::Vec;

use algo::Algo;
use u256::U256;

/// Block 125552's header, as bytes. The same array `tools/algocheck.py` holds.
///
/// Its hash is public and checkable against any explorer, which is what makes
/// it worth more than a synthetic fixture: nothing here chose the answer.
const BTC_HEADER: [u8; 80] = [
    0x01, 0x00, 0x00, 0x00, 0x81, 0xcd, 0x02, 0xab, 0x7e, 0x56, 0x9e, 0x8b, 0xcd, 0x93, 0x17, 0xe2,
    0xfe, 0x99, 0xf2, 0xde, 0x44, 0xd4, 0x9a, 0xb2, 0xb8, 0x85, 0x1b, 0xa4, 0xa3, 0x08, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xe3, 0x20, 0xb6, 0xc2, 0xff, 0xfc, 0x8d, 0x75, 0x04, 0x23, 0xdb, 0x8b,
    0x1e, 0xb9, 0x42, 0xae, 0x71, 0x0e, 0x95, 0x1e, 0xd7, 0x97, 0xf7, 0xaf, 0xfc, 0x88, 0x92, 0xb0,
    0xf1, 0xfc, 0x12, 0x2b, 0xc7, 0xf5, 0xd7, 0x4d, 0xf2, 0xb9, 0x44, 0x1a, 0x42, 0xa1, 0x46, 0x95,
];

/// The same block's prevhash **as a pool sends it**: each 4-byte word of the
/// header's own bytes, reversed. Words in the same order.
const BTC_PREV_WIRE: [u8; 32] = [
    0xab, 0x02, 0xcd, 0x81, 0x8b, 0x9e, 0x56, 0x7e, 0xe2, 0x17, 0x93, 0xcd, 0xde, 0xf2, 0x99, 0xfe,
    0xb2, 0x9a, 0xd4, 0x44, 0xa4, 0x1b, 0x85, 0xb8, 0x00, 0x00, 0x08, 0xa3, 0x00, 0x00, 0x00, 0x00,
];

/// And its merkle root, which needs no transform at all.
const BTC_MERKLE: [u8; 32] = [
    0xe3, 0x20, 0xb6, 0xc2, 0xff, 0xfc, 0x8d, 0x75, 0x04, 0x23, 0xdb, 0x8b, 0x1e, 0xb9, 0x42, 0xae,
    0x71, 0x0e, 0x95, 0x1e, 0xd7, 0x97, 0xf7, 0xaf, 0xfc, 0x88, 0x92, 0xb0, 0xf1, 0xfc, 0x12, 0x2b,
];

const BTC_VERSION: u32 = 1;
const BTC_NTIME: u32 = 0x4dd7_f5c7;
const BTC_NBITS: u32 = 0x1a44_b9f2;
const BTC_NONCE: u32 = 0x9546_a142;

/// The raw digest, least-significant byte first. Bitcoin *displays* the reverse
/// of this, which is the trap the constant is written out to pin down.
const BTC_DIGEST: [u8; 32] = [
    0x1d, 0xbd, 0x98, 0x1f, 0xe6, 0x98, 0x57, 0x76, 0xb6, 0x44, 0xb1, 0x73, 0xa4, 0xd0, 0x38, 0x5d,
    0xdc, 0x1a, 0xa2, 0xa8, 0x29, 0x68, 0x8d, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Build a little-endian digest from a big-endian 256-bit value, for the
/// comparison claims. A digest is stored the opposite way round from a target,
/// which is the whole reason `below_target` reverses before comparing.
fn le_digest(be: &[u8; 32]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for i in 0..32 {
        d[i] = be[31 - i];
    }
    d
}

/// Parse a 64-character hex digest into bytes.
///
/// Only reachable from `checks`, and it refuses nothing: a malformed literal
/// here would be a typo in a claim rather than input from anywhere, and the
/// claim it belongs to fails, which is the report that is wanted.
fn hex32(s: &str) -> [u8; 32] {
    let b = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = (b[i * 2] as char).to_digit(16).unwrap_or(0) as u8;
        let lo = (b[i * 2 + 1] as char).to_digit(16).unwrap_or(0) as u8;
        out[i] = (hi << 4) | lo;
    }
    out
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    // --- the fixture, which pins six things at once ---
    let h = header::assemble(
        BTC_VERSION,
        &BTC_PREV_WIRE,
        &BTC_MERKLE,
        BTC_NTIME,
        BTC_NBITS,
        BTC_NONCE,
    );
    out.push((
        "block 125552 reassembles from pool fields, byte for byte",
        h == BTC_HEADER,
    ));
    out.push((
        "and hashes to its published digest",
        hash::sha256d(&h) == BTC_DIGEST,
    ));

    // The prevhash swap is not a whole-string reversal, and it is not a no-op.
    // Both of those are what somebody writes when the table is skimmed, and
    // both would pass a claim that only checked the length.
    let mut whole_rev = [0u8; 32];
    for i in 0..32 {
        whole_rev[i] = BTC_PREV_WIRE[31 - i];
    }
    out.push((
        "the prevhash swap is neither a whole reversal nor a no-op",
        h[4..36] != whole_rev[..] && h[4..36] != BTC_PREV_WIRE[..],
    ));

    // --- the midstate, which is the only optimisation in the tree ---
    let mid = hash::Midstate::new(&h);
    out.push((
        "a midstate reproduces the whole-header hash",
        mid.hash_with(BTC_NONCE) == BTC_DIGEST,
    ));
    out.push((
        "and a different nonce gives a different digest",
        mid.hash_with(BTC_NONCE.wrapping_add(1)) != BTC_DIGEST,
    ));

    // --- targets, produced two ways that must agree ---
    let d1 = u256::diff1();
    let mut want = [0u8; 32];
    want[4] = 0xff;
    want[5] = 0xff;
    out.push((
        "nbits 0x1d00ffff is the difficulty-1 target",
        d1.to_be_bytes() == want,
    ));
    out.push((
        "difficulty 1 gives the same target by the other route",
        u256::target_for(1, 0) == Some(d1),
    ));
    out.push((
        "an exponent off either end is refused, not clamped",
        U256::from_nbits(0x0200_ffff).is_none() && U256::from_nbits(0x2100_ffff).is_none(),
    ));

    // Fractional difficulty. `Json::as_i64` truncates at the decimal point, so
    // 0.001 would arrive as 0 and a target from zero either faults or accepts
    // everything until the worker is banned. This is the arithmetic that makes
    // the split worth carrying.
    let easy = u256::target_for(1, 3); // difficulty 0.001
    out.push((
        "difficulty 0.001 is a larger target than difficulty 1",
        matches!(easy, Some(t) if t > d1),
    ));
    let hard = u256::target_for(8192, 0);
    out.push((
        "and a large difficulty is a smaller one",
        matches!(hard, Some(t) if t < d1),
    ));
    out.push((
        "a zero difficulty is refused rather than divided by",
        u256::target_for(0, 0).is_none(),
    ));
    out.push((
        "overflow refuses instead of wrapping to a smaller target",
        U256::from_be_bytes(&[0xff; 32]).mul_u32(2).is_none(),
    ));

    // --- the comparison, and the case a zero count cannot express ---
    //
    // Both digests below have exactly 32 leading zero bits, so a leading-zero
    // implementation calls them equal. One is under the difficulty-1 target and
    // one is over it. This is the claim `cuda/algo.cuh`'s comment is about.
    let mut lo = [0u8; 32];
    lo[4] = 0xff;
    lo[5] = 0xfe;
    let mut hi = [0u8; 32];
    hi[4] = 0xff;
    hi[5] = 0xff;
    hi[31] = 0x01;
    out.push((
        "a digest under the target passes",
        hash::below_target(&le_digest(&lo), &d1),
    ));
    out.push((
        "one over it does not, though both have 32 leading zeros",
        !hash::below_target(&le_digest(&hi), &d1)
            && hash::leading_zero_bits(&le_digest(&lo)) == 32
            && hash::leading_zero_bits(&le_digest(&hi)) == 32,
    ));
    out.push((
        "a digest exactly on the target counts as below it",
        hash::below_target(&le_digest(&d1.to_be_bytes()), &d1),
    ));

    // --- the pieces the stratum client will lean on ---
    let cb = header::coinbase(b"\x01\x02", b"\xaa\xbb", b"\xcc\xdd", b"\x03\x04");
    out.push((
        "a coinbase is its four parts in order",
        cb == [0x01, 0x02, 0xaa, 0xbb, 0xcc, 0xdd, 0x03, 0x04],
    ));
    out.push((
        "an empty merkle branch leaves the coinbase hash alone",
        header::merkle_root(&cb, &[]) == hash::sha256d(&cb),
    ));
    let sib = [0x11u8; 32];
    out.push((
        "and one step folds the sibling in on the right",
        header::merkle_root(&cb, &[sib]) != hash::sha256d(&cb),
    ));

    // A real branch, four levels deep, from a tree with an odd node at one
    // level -- which is where Bitcoin duplicates the last hash and where a fold
    // goes wrong first.
    //
    // This exists because nothing else reaches it. The `mining.notify` fixture
    // carries an empty branch, so until this claim the loop body in
    // `merkle_root` had never executed against more than one element; and the
    // one real pool this has been driven at (public-pool.io) refuses to
    // authorize without a payout address and therefore sends no job at all.
    // The leaves are synthetic and that changes nothing: the fold is what is
    // under test, and the root comes from `hashlib` in `tools/algocheck.py`
    // rather than from anything here.
    const BRANCH: [[u8; 32]; 4] = [
        [
            0x72, 0x83, 0x38, 0xd9, 0x9f, 0x35, 0x61, 0x75, 0xc4, 0x94, 0x5e, 0xf5, 0xcc, 0xcf,
            0xa6, 0x1b, 0x7b, 0x56, 0x14, 0x3c, 0xbb, 0xf4, 0x26, 0xdd, 0xd0, 0xe0, 0xfc, 0x7c,
            0xfe, 0x8c, 0x3c, 0x23,
        ],
        [
            0x57, 0x55, 0x58, 0xcd, 0xf9, 0x72, 0x7a, 0xfc, 0x22, 0x50, 0x91, 0x27, 0x9e, 0x3b,
            0xb3, 0xbf, 0x79, 0x39, 0x6a, 0xc9, 0x12, 0x1d, 0x57, 0xc2, 0x52, 0x1e, 0xfd, 0x2e,
            0x0a, 0xa1, 0xd5, 0x51,
        ],
        [
            0x35, 0xd6, 0x3b, 0xaa, 0xaf, 0x85, 0x5f, 0x96, 0x62, 0x63, 0x46, 0x25, 0x9a, 0x47,
            0xd3, 0xf3, 0x58, 0x22, 0x9f, 0x6a, 0x2a, 0xc1, 0x4e, 0x10, 0xbd, 0x78, 0xa2, 0xe1,
            0x42, 0x7e, 0xd8, 0xd7,
        ],
        [
            0xdc, 0xa6, 0x61, 0x9d, 0xe0, 0x0d, 0x59, 0x05, 0x82, 0x21, 0x78, 0x75, 0x65, 0x46,
            0x65, 0x02, 0xd4, 0xa9, 0x2e, 0x0a, 0x80, 0xe9, 0x35, 0x8a, 0x07, 0xdb, 0xff, 0xd7,
            0xec, 0x6e, 0x6d, 0xac,
        ],
    ];
    const BRANCH_ROOT: [u8; 32] = [
        0x45, 0x45, 0xcd, 0xed, 0xf9, 0x42, 0xe8, 0x07, 0xcc, 0xdd, 0xc9, 0xba, 0xe7, 0xcd, 0x99,
        0xb6, 0xed, 0xd1, 0x02, 0x13, 0x3b, 0xc8, 0x79, 0x0d, 0xe0, 0x8f, 0x27, 0x19, 0x80, 0x08,
        0x8c, 0x16,
    ];
    out.push((
        "a four-level merkle branch folds to the root hashlib computes",
        header::merkle_root(&[0u8; 8], &BRANCH) == BRANCH_ROOT,
    ));
    // Order is the thing a fold gets wrong and still produces 32 plausible
    // bytes. Reversing the branch must not land on the same root.
    let mut rev = BRANCH;
    rev.reverse();
    out.push((
        "and the branch order matters, so the fold is not concatenating a set",
        header::merkle_root(&[0u8; 8], &rev) != BRANCH_ROOT,
    ));

    let (ntime_hex, nonce_hex) = header::submit_hex(&h);
    out.push((
        "submit sends ntime and nonce big-endian, the header's bytes reversed",
        ntime_hex == [0x4d, 0xd7, 0xf5, 0xc7] && nonce_hex == [0x95, 0x46, 0xa1, 0x42],
    ));

    out.extend(stratum_checks());
    out.extend(ev_checks());
    out.extend(yespower_checks());
    out
}

/// yespower against vectors this kernel did not compute.
///
/// The input is `src[i] = i * 3` over 80 bytes, which is what upstream's
/// `tests.c` hashes and is conveniently a header's length. Every digest below
/// comes from `tools/yespower.py`, which is checked against upstream's own
/// published TESTS-OK before it is trusted to produce anything -- so the chain
/// is upstream's vectors, then the Python, then this.
///
/// The small parameter sets are here rather than the large ones because the
/// implementation is the *reference* one, deliberately unoptimised, and a boot
/// selftest that takes seconds is one people stop reading. `diag mine` is where
/// a slower set belongs when there is one.
fn yespower_checks() -> Vec<(&'static str, bool)> {
    use yespower::{Version, Yespower};
    let mut out = Vec::new();

    let src: alloc::vec::Vec<u8> = (0..80u32).map(|i| (i * 3) as u8).collect();

    const V10_1024_8: [u8; 32] = [
        0xc9, 0x8f, 0x31, 0x9e, 0xa7, 0xdf, 0x5e, 0x7f, 0xd0, 0xd8, 0xac, 0xa4, 0xac, 0x14, 0xf7,
        0x69, 0x2a, 0x40, 0xf4, 0x88, 0x63, 0x15, 0xe2, 0x42, 0x09, 0x28, 0x5a, 0x42, 0x94, 0x91,
        0x23, 0xfd,
    ];
    const V05_1024_8: [u8; 32] = [
        0x62, 0xa5, 0x42, 0x28, 0x38, 0x76, 0x1e, 0x78, 0xb5, 0xbe, 0x2c, 0xa8, 0x47, 0xde, 0xc9,
        0x20, 0xf9, 0x17, 0x2a, 0x90, 0xeb, 0x29, 0x7c, 0x64, 0x97, 0x3e, 0x95, 0xf3, 0xd5, 0x00,
        0xce, 0x0e,
    ];
    // This one is upstream's own TESTS-OK line verbatim rather than something
    // the Python produced, so it closes the loop on the whole chain.
    const V10_1024_32: [u8; 32] = [
        0x50, 0x1b, 0x79, 0x2d, 0xb4, 0x2e, 0x38, 0x8f, 0x6e, 0x7d, 0x45, 0x3c, 0x95, 0xd0, 0x3a,
        0x12, 0xa3, 0x60, 0x16, 0xa5, 0x15, 0x4a, 0x68, 0x83, 0x90, 0xdd, 0xc6, 0x09, 0xa4, 0x0c,
        0x67, 0x99,
    ];

    match Yespower::new(Version::V1_0, 1024, 8) {
        Some(mut y) => {
            out.push((
                "yespower 1.0 at N=1024 r=8 matches the oracle",
                y.hash(&src, None) == V10_1024_8,
            ));
            // Same instance, twice. The S-boxes rotate and `w` advances during
            // a hash, so an instance that failed to reset them would answer
            // correctly once and differently ever after -- which is the worst
            // shape available here, since a miner reuses one instance for
            // millions of nonces and only the first would be right.
            out.push((
                "and again on the same instance, so the state resets",
                y.hash(&src, None) == V10_1024_8,
            ));
            let mut other = src.clone();
            other[0] ^= 1;
            out.push((
                "a one-bit change in the input changes the digest",
                y.hash(&other, None) != V10_1024_8,
            ));
        }
        None => out.push(("yespower 1.0 at N=1024 r=8 could be constructed", false)),
    }

    // The two versions are not variants of one function, they are different
    // proof-of-work schemes that live on different coins. Same input, same
    // parameters, different answer.
    match Yespower::new(Version::V0_5, 1024, 8) {
        Some(mut y) => out.push((
            "yespower 0.5 answers differently, and matches its own vector",
            y.hash(&src, None) == V05_1024_8 && V05_1024_8 != V10_1024_8,
        )),
        None => out.push(("yespower 0.5 at N=1024 r=8 could be constructed", false)),
    }

    match Yespower::new(Version::V1_0, 1024, 32) {
        Some(mut y) => {
            out.push((
                "and at r=32 it matches upstream's own published vector",
                y.hash(&src, None) == V10_1024_32,
            ));
            // 128 * r * N for V, plus the S-boxes and three 128r scratches.
            out.push((
                "the working set is the size the parameters imply",
                y.footprint() >= 128 * 32 * 1024 && y.footprint() < 128 * 32 * 1024 * 2,
            ));
        }
        None => out.push(("yespower 1.0 at N=1024 r=32 could be constructed", false)),
    }

    // Refused rather than clamped. Every one of these bounds is part of what a
    // coin's network agreed on, so a clamped parameter hashes a different
    // function perfectly correctly and every share is rejected.
    out.push((
        "a non-power-of-two N is refused",
        Yespower::new(Version::V1_0, 1500, 8).is_none(),
    ));
    out.push((
        "and an N or r outside the scheme's range",
        Yespower::new(Version::V1_0, 512, 8).is_none()
            && Yespower::new(Version::V1_0, 1024, 7).is_none()
            && Yespower::new(Version::V1_0, 1024, 33).is_none(),
    ));

    // --- the resource fingerprint, and the one part of it that can drift ---
    //
    // `Algo::working_set` is the formula and `Yespower::footprint` is the
    // allocation, written independently because asking the second costs eight
    // megabytes and a scheduler cannot pay that to decide what to run next.
    // Two expressions of one quantity is exactly the arrangement this tree
    // warns about, so it is checked rather than trusted: a table that said two
    // megabytes where the allocator takes eight would pack a device wrong and
    // report nothing at all.
    let mut agree = true;
    for (v10, n, r) in [
        (true, 1024u32, 8u32),
        (true, 2048, 8),
        (true, 1024, 32),
        (false, 1024, 8),
        (false, 2048, 16),
    ] {
        let a = Algo::Yespower { v10, n, r, pers: None };
        let ver = if v10 { Version::V1_0 } else { Version::V0_5 };
        match Yespower::new(ver, n, r) {
            Some(y) => agree = agree && a.working_set() == y.footprint(),
            None => agree = false,
        }
    }
    out.push((
        "the declared working set is the one yespower actually allocates",
        agree,
    ));

    // The whole point of the fingerprint: what two algorithms take from each
    // other on one device. Two arithmetic ones halve each other; an arithmetic
    // one beside a memory-bound one is close to free, which is the only shape
    // in which running several coins at once does more work rather than the
    // same work divided up.
    out.push((
        "two arithmetic algorithms contend and sha256d does not contend with yespower",
        Algo::Sha256d.contends_with(&Algo::Blake2s)
            && !Algo::Sha256d.contends_with(&Algo::Yespower {
                v10: true,
                n: 2048,
                r: 8,
                pers: None,
            }),
    ));
    // And the size gap is the reason the bound differs, so it is asserted
    // rather than left as a story: four orders of magnitude between them.
    out.push((
        "an arithmetic hash is cache-resident and a memory-bound one is megabytes",
        Algo::Sha256d.working_set() <= 4096
            && Algo::Yespower { v10: true, n: 2048, r: 8, pers: None }.working_set()
                > 2 * 1024 * 1024,
    ));

    // --- the cache budget, every branch of it, with no processor involved ---
    //
    // Slices run at once, so their working sets are resident at once, and a
    // memory-bound algorithm exists to exceed a core's private cache. Handing
    // out more slices than the last level holds buys thrashing -- a report
    // where the rate went *down* when more of the machine was given to it.
    let mib = 1024 * 1024;
    out.push((
        "a cache that holds twelve working sets does not cap four slices",
        work::budget_from(Some(24 * mib), 2 * mib, 4) == 4,
    ));
    out.push((
        "and one that holds three caps four to three",
        work::budget_from(Some(24 * mib), 8 * mib, 4) == 3,
    ));
    // The two refusals, which are the branches worth having claims for.
    out.push((
        "an unreadable cache leaves the request alone rather than throttling it",
        work::budget_from(None, 8 * mib, 4) == 4,
    ));
    out.push((
        "and nothing memory-bound means the cache is not the resource",
        work::budget_from(Some(24 * mib), 0, 4) == 4,
    ));
    // A working set larger than the whole cache still gets one slice. Zero
    // would turn a large-parameter coin into a silent no-op, which reads from
    // the report exactly like a pool that has gone quiet.
    out.push((
        "a working set larger than the cache still gets one slice, never none",
        work::budget_from(Some(4 * mib), 16 * mib, 4) == 1,
    ));

    // The processor's own answer, when it gives one. Not a claim about a
    // number -- this boots under an emulator and on a laptop and they differ --
    // but that what comes back is a cache rather than nonsense.
    out.push((
        "the last-level cache reads as a plausible size, or refuses",
        match crate::cpu::last_level_cache() {
            None => true,
            Some(n) => (256 * 1024..=512 * mib).contains(&n) && n % 4096 == 0,
        },
    ));

    out
}

/// The expected-value arithmetic, and the coinbase parser under it.
fn ev_checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    // A coinbase transaction, hand-built so every field is known. One input
    // with the null prevout a coinbase has, two outputs, and a locktime -- 76
    // bytes exactly, which is the number the parser has to land on.
    const CB: [u8; 76] = [
        0x01, 0x00, 0x00, 0x00, // version
        0x01, // one input
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // null prevout, 32 bytes
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
        0xff, 0xff, 0xff, 0xff, // prevout index
        0x04, 0xde, 0xad, 0xbe, 0xef, // script
        0xff, 0xff, 0xff, 0xff, // sequence
        0x02, // two outputs
        0x00, 0xf2, 0x05, 0x2a, 0x01, 0x00, 0x00, 0x00, 0x01, 0x51, // 50 coin
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x6a, 0x00, // zero
        0x00, 0x00, 0x00, 0x00, // locktime
    ];
    out.push((
        "a coinbase's outputs sum to what it pays",
        ev::coinbase_value(&CB) == Some(5_000_000_000),
    ));
    // Landing exactly on the last byte is the whole bargain, and it is what
    // separates a parse from a plausible number. Both directions.
    let mut long = alloc::vec::Vec::from(&CB[..]);
    long.push(0);
    out.push((
        "a transaction with a byte left over is refused, not summed",
        ev::coinbase_value(&long).is_none(),
    ));
    out.push((
        "and so is a truncated one",
        ev::coinbase_value(&CB[..CB.len() - 1]).is_none()
            && ev::coinbase_value(&CB[..20]).is_none(),
    ));
    // Stratum's coinbase is the non-witness serialisation by construction,
    // because it is what the merkle root is taken over. A marker here means
    // this is not the transaction the parser thinks it is.
    let mut segwit = alloc::vec::Vec::from(&CB[..4]);
    segwit.push(0x00);
    segwit.push(0x01);
    segwit.extend_from_slice(&CB[4..]);
    out.push((
        "a segwit marker is refused rather than skipped past",
        ev::coinbase_value(&segwit).is_none(),
    ));

    // The two producers of an expected-hash count have to agree. A share at
    // difficulty 1 costs 2^32 hashes by definition, and the difficulty-1
    // *target* has to imply the same thing -- it is 0xFFFF * 2^208, so the
    // exact answer is 2^48 / 0xFFFF, about 0.0015% above 2^32.
    let d1 = u256::diff1();
    let from_target = ev::expected_hashes(&d1).unwrap_or(0.0);
    let from_diff = ev::share_hashes(1, 0);
    let ratio = from_target / from_diff;
    out.push((
        "the difficulty-1 target implies the 2^32 hashes difficulty 1 costs",
        ratio > 0.999 && ratio < 1.001,
    ));
    out.push((
        "an easier difficulty costs proportionally fewer hashes",
        (ev::share_hashes(1, 3) * 1000.0 - from_diff).abs() < 1.0,
    ));
    out.push((
        "a zero target is refused rather than dividing by it",
        ev::expected_hashes(&U256::ZERO).is_none(),
    ));

    // The unit picker, because a figure in seconds that should be in millennia
    // is the one this whole block exists to make legible.
    out.push((
        "a duration is rendered in a unit that leaves it legible",
        ev::render_seconds(30.0).1 == "seconds"
            && ev::render_seconds(120.0).1 == "minutes"
            && ev::render_seconds(90_000.0).1 == "days"
            && ev::render_seconds(1.0e12).1 == "millennia",
    ));

    out
}

/// The protocol half: framing, classification, and the difficulty trap.
fn stratum_checks() -> Vec<(&'static str, bool)> {
    use crate::json::Json;
    use alloc::vec;
    use stratum::{classify, decimal, parse_job, subscribe_result, take_line, unhex, Error, Message};

    let mut out = Vec::new();

    // --- framing ---
    let mut buf: Vec<u8> = b"{\"id\":1}".to_vec();
    out.push((
        "a partial line is withheld rather than guessed at",
        matches!(take_line(&mut buf), Ok(None)),
    ));
    buf.extend_from_slice(b"\n{\"id\":2}\n");
    let first = take_line(&mut buf);
    out.push((
        "and completes when the rest of it arrives",
        matches!(&first, Ok(Some(s)) if s == "{\"id\":1}"),
    ));
    out.push((
        "a second line survives the first being taken",
        matches!(take_line(&mut buf), Ok(Some(s)) if s == "{\"id\":2}"),
    ));
    out.push((
        "and the buffer is empty afterwards",
        matches!(take_line(&mut buf), Ok(None)) && buf.is_empty(),
    ));
    let mut crlf: Vec<u8> = b"{\"a\":1}\r\n".to_vec();
    out.push((
        "a CRLF terminator leaves no carriage return behind",
        matches!(take_line(&mut crlf), Ok(Some(s)) if s == "{\"a\":1}"),
    ));
    let mut huge: Vec<u8> = vec![b'x'; stratum::MAX_LINE + 2];
    out.push((
        "a line with no terminator in sight is refused, not buffered forever",
        take_line(&mut huge) == Err(Error::Oversize),
    ));

    // --- the difficulty trap, and the contrast that explains it ---
    //
    // The first of these two is the bug. `as_i64` splits at the decimal point,
    // so a pool sending 0.001 hands the kernel a zero, and a target computed
    // from zero either faults or accepts everything until the worker is banned.
    // Asserting the wrong answer beside the right one is what stops somebody
    // "simplifying" `decimal` back into `as_i64`.
    out.push((
        "as_i64 really does read 0.001 as zero, which is why decimal exists",
        matches!(Json::parse("0.001"), Some(j) if j.as_i64() == Some(0)),
    ));
    out.push(("difficulty 0.001 reads as 1/10^3", decimal("0.001") == Some((1, 3))));
    out.push(("difficulty 8192.5 keeps its half", decimal("8192.5") == Some((81925, 1))));
    out.push(("a whole difficulty has no scale", decimal("8192") == Some((8192, 0))));
    out.push((
        "a difficulty too small to represent errs large, never to zero",
        matches!(decimal("0.0000000001"), Some((m, _)) if m > 0),
    ));
    out.push((
        "and a genuine zero is still zero, for target_for to refuse",
        decimal("0") == Some((0, 0)),
    ));
    out.push((
        "garbage and negatives are refused",
        decimal("-1").is_none() && decimal("abc").is_none() && decimal("").is_none(),
    ));

    // JSON numbers may carry an exponent and a pool that sends one must not be
    // ignored. This was a real miss: a stub whose difficulty Python serialised
    // as `1e-05` was refused, the previous difficulty stayed in force, and the
    // miner hashed against a target the pool had not set with nothing printed.
    out.push((
        "1e-05 is the same difficulty as 0.00001",
        decimal("1e-05") == decimal("0.00001") && decimal("1e-05") == Some((1, 5)),
    ));
    out.push((
        "a positive exponent multiplies out to a whole difficulty",
        decimal("1.5e3") == Some((1500, 0)) && decimal("2E2") == Some((200, 0)),
    ));
    out.push((
        "an exponent past the scale cap clamps large, never to zero",
        matches!(decimal("1e-10"), Some((m, s)) if m > 0 && s == stratum::MAX_SCALE),
    ));
    out.push((
        "a malformed exponent is refused rather than half-read",
        decimal("1e").is_none() && decimal("1e+").is_none() && decimal("1ex").is_none(),
    ));

    // --- hex ---
    out.push((
        "hex refuses an odd length and a non-hex byte",
        unhex("abc").is_none() && unhex("zz").is_none(),
    ));
    out.push((
        "and round-trips what it accepts",
        unhex("00ff10").as_deref() == Some(&[0x00, 0xff, 0x10][..]),
    ));

    // --- classification by shape ---
    let resp = classify("{\"id\":7,\"result\":true,\"error\":null}");
    out.push((
        "a message carrying result is a response, with its id",
        matches!(&resp, Ok(Message::Response { id: 7, ok: true, .. })),
    ));
    out.push((
        "an explicit error is a failed response, not a lost one",
        matches!(
            classify("{\"id\":8,\"result\":null,\"error\":[21,\"Job not found\",null]}"),
            Ok(Message::Response { id: 8, ok: false, .. })
        ),
    ));
    out.push((
        "a result of false is a rejection even with no error object",
        matches!(
            classify("{\"id\":9,\"result\":false,\"error\":null}"),
            Ok(Message::Response { ok: false, .. })
        ),
    ));
    out.push((
        "a method with a null id is a notification",
        matches!(
            classify("{\"id\":null,\"method\":\"mining.set_difficulty\",\"params\":[0.5]}"),
            Ok(Message::Notify { .. })
        ),
    ));

    // --- subscribe, whose result[0] differs between pools and is never read ---
    let a = Json::parse(
        "{\"id\":1,\"result\":[[[\"mining.set_difficulty\",\"b4b6\"],\
         [\"mining.notify\",\"ae6812\"]],\"08000002\",4],\"error\":null}",
    );
    let b = Json::parse("{\"id\":1,\"result\":[\"deadbeef\",\"1a2b3c4d\",8],\"error\":null}");
    out.push((
        "a nested subscribe result gives up extranonce1 and its size",
        matches!(a.as_ref().and_then(subscribe_result), Some((ref e, 4)) if e == &[0x08, 0x00, 0x00, 0x02]),
    ));
    out.push((
        "and so does a flat one, because result[0] is never read",
        matches!(b.as_ref().and_then(subscribe_result), Some((ref e, 8)) if e == &[0x1a, 0x2b, 0x3c, 0x4d]),
    ));

    // --- a job, with the real block's own fields ---
    let notify = alloc::format!(
        "{{\"id\":null,\"method\":\"mining.notify\",\"params\":[\"job1\",\"{}\",\
         \"01000000\",\"ffffffff\",[],\"00000001\",\"1a44b9f2\",\"4dd7f5c7\",true]}}",
        stratum::hex(&BTC_PREV_WIRE)
    );
    let job = match classify(&notify) {
        Ok(Message::Notify { params, .. }) => parse_job(&params),
        _ => None,
    };
    out.push((
        "a notify parses into a job with the fields in the right order",
        matches!(&job, Some(j)
            if j.id == "job1"
            && j.prev_wire == BTC_PREV_WIRE
            && j.version == BTC_VERSION
            && j.nbits == BTC_NBITS
            && j.ntime == BTC_NTIME
            && j.clean),
    ));
    out.push((
        "a notify one parameter short is refused rather than defaulted",
        matches!(
            classify("{\"id\":null,\"method\":\"mining.notify\",\"params\":[\"j\",\"00\"]}"),
            Ok(Message::Notify { ref params, .. }) if parse_job(params).is_none()
        ),
    ));

    // --- what we send ---
    let s = stratum::submit(3, "wk", "job1", &[0xaa, 0xbb], &[0x4d, 0xd7, 0xf5, 0xc7], &[0x95, 0x46, 0xa1, 0x42]);
    out.push((
        "a submit carries the job, the extranonce and both words as hex",
        s.contains("\"mining.submit\"")
            && s.contains("\"job1\"")
            && s.contains("\"aabb\"")
            && s.contains("\"4dd7f5c7\"")
            && s.contains("\"9546a142\"")
            && s.ends_with('\n'),
    ));
    out.push((
        "and every message we send is one line",
        stratum::subscribe(1).matches('\n').count() == 1
            && stratum::authorize(2, "u", "p").matches('\n').count() == 1,
    ));

    // --- BLAKE2s, against RFC 7693's own vectors and hashlib's ---
    //
    // Three messages spanning the three shapes the padding can take: empty
    // (one all-zero final block with t=0), short (one partial block), and 80
    // bytes (a full interior block plus a partial final one, which is the
    // only shape mining ever uses and the only one that exercises `last`
    // being false on a compression).
    out.push((
        "blake2s of the empty string matches RFC 7693",
        blake2s::hash(&[]) == hex32("69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"),
    ));
    out.push((
        "blake2s(\"abc\") matches RFC 7693",
        blake2s::hash(b"abc") == hex32("508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982"),
    ));
    let b2_hdr = hex32("753d9b626a850d23b32a05e6d531fbe985af7a300b21505d3b080611e9940aeb");
    out.push((
        "and an 80-byte header matches what hashlib computes for it",
        blake2s::hash(&BTC_HEADER) == b2_hdr,
    ));

    // The midstate, which is the whole reason this algorithm is cheap, and the
    // claim that `algo.rs` was wrong about it only having one for SHA-256d.
    let b2mid = blake2s::Midstate::new(&BTC_HEADER);
    out.push((
        "a blake2s midstate reproduces the whole-header hash",
        b2mid.hash_with(BTC_NONCE) == b2_hdr,
    ));
    // And moves with the nonce. A midstate that ignored its argument would
    // pass the line above and mine one nonce forever.
    out.push((
        "and a different nonce gives the digest hashlib gives for it",
        b2mid.hash_with(0x1234_5678)
            == hex32("d81f08a0b2790da2d71ddea1ea3f6a76688eb996b148aa7d4463f177506e4411"),
    ));
    // The counter is the message length and not the block index. Padding a
    // short final block without moving `t` makes two messages of different
    // lengths collide, which is the single misreading of section 3.2 that
    // produces a hash function that looks perfectly healthy.
    out.push((
        "blake2s tells a message from the same message zero-padded",
        blake2s::hash(b"a") != blake2s::hash(b"a "),
    ));
    // 64 bytes exactly: the loop must keep its last full block for the `last`
    // flag rather than compressing it as an interior one and then compressing
    // an all-zero final block.
    out.push((
        "a message of exactly one block is not compressed twice",
        blake2s::hash(&[0u8; 64]) != blake2s::hash(&[0u8; 128]),
    ));

    // The seam `algo.rs` claims: a third algorithm and nothing above `Hasher`
    // moved. Checked by driving it through the same `Hasher` the miner uses
    // rather than by calling `blake2s::hash`, since the point is the interface.
    out.push((
        "the Hasher seam takes a third algorithm unchanged",
        algo::Hasher::new(&algo::Algo::Blake2s, &BTC_HEADER)
            .map(|mut h| h.hash(&BTC_HEADER, BTC_NONCE) == b2_hdr)
            .unwrap_or(false),
    ));

    // NeoScrypt's own, which are two real Feathercoin blocks and the keyed
    // BLAKE2s underneath them. Extended here rather than duplicated, because a
    // second copy of a vector is a second thing to get wrong.
    out.extend(neoscrypt::checks());
    out.extend(work::checks());

    out
}
