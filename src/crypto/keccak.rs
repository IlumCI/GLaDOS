//! Keccak-256, for the one thing in this tree that needs it: EIP-55.
//!
//! **This is Keccak-256 and not SHA3-256.** They are the same permutation and
//! differ in one byte of padding -- `0x01` here against SHA3's `0x06` -- which
//! is a difference no test that only checks the plumbing can see. Ethereum
//! standardised on the original Keccak padding before SHA3 was finalised and
//! kept it, so a `sha3` crate's output is the wrong digest for every address,
//! signature and storage slot on every EVM chain. The two vectors below are
//! Keccak's own; SHA3-256 of the same inputs is different, which is the point
//! of having them.
//!
//! ### Why it is here at all
//!
//! `mine::addr` checks a payout address before a miner image spends a night
//! hashing to it, and an EVM address carries a checksum *only* in the mixed
//! case of its own hex digits, derived from the Keccak-256 of the lowercase
//! form. Without this there is no checksum to read, so a transposed pair of
//! characters in the address a wallet handed the operator is undetectable until
//! the money does not arrive.
//!
//! Nothing else uses it. There is no streaming form, because the longest thing
//! it will ever hash is forty bytes.

/// The 24 round constants of Keccak-f[1600].
const RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rho's rotation amounts, in the order Pi visits the lanes.
const ROT: [u32; 24] =
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

/// Pi's lane permutation, as the destination index of each step.
const PI: [usize; 24] =
    [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

/// The rate in bytes for a 256-bit digest: 200 - 2 * 32.
const RATE: usize = 136;

/// Keccak-f[1600] in place.
///
/// Rho and Pi are one loop rather than two, carrying the displaced lane in `t`,
/// which is the reference implementation's own arrangement: doing them
/// separately needs a second 25-lane array, and this permutation is a single
/// cycle so the chain never needs one.
fn f1600(st: &mut [u64; 25]) {
    for r in 0..24 {
        // Theta
        let mut bc = [0u64; 5];
        for i in 0..5 {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for i in 0..5 {
            let t = bc[(i + 4) % 5] ^ bc[(i + 1) % 5].rotate_left(1);
            let mut j = 0;
            while j < 25 {
                st[j + i] ^= t;
                j += 5;
            }
        }

        // Rho and Pi
        let mut t = st[1];
        for i in 0..24 {
            let j = PI[i];
            let held = st[j];
            st[j] = t.rotate_left(ROT[i]);
            t = held;
        }

        // Chi
        let mut j = 0;
        while j < 25 {
            let row = [st[j], st[j + 1], st[j + 2], st[j + 3], st[j + 4]];
            for i in 0..5 {
                st[j + i] = row[i] ^ (!row[(i + 1) % 5] & row[(i + 2) % 5]);
            }
            j += 5;
        }

        // Iota
        st[0] ^= RC[r];
    }
}

/// XOR `block` into the state's first bytes, little-endian per lane.
fn absorb(st: &mut [u64; 25], block: &[u8]) {
    for (i, chunk) in block.chunks(8).enumerate() {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        st[i] ^= u64::from_le_bytes(w);
    }
}

/// Keccak-256 of a message.
pub fn keccak256(msg: &[u8]) -> [u8; 32] {
    let mut st = [0u64; 25];

    let full = msg.len() / RATE;
    for b in 0..full {
        absorb(&mut st, &msg[b * RATE..(b + 1) * RATE]);
        f1600(&mut st);
    }

    // The final block, padded. `0x01` at the end of the message and `0x80` at
    // the end of the rate, which for a message that exactly fills the rate are
    // in the same block as each other and in none of the ones above -- the case
    // a padding written as "append then maybe permute" gets wrong.
    let mut tail = [0u8; RATE];
    let rest = &msg[full * RATE..];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] ^= 0x01;
    tail[RATE - 1] ^= 0x80;
    absorb(&mut st, &tail);
    f1600(&mut st);

    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&st[i].to_le_bytes());
    }
    out
}

/// Published Keccak-256 vectors, plus the two properties a wrong padding or a
/// wrong rate break.
pub fn checks() -> alloc::vec::Vec<(bool, alloc::string::String)> {
    use alloc::string::String;
    use alloc::vec::Vec;
    let mut out: Vec<(bool, String)> = Vec::new();
    let mut ok = |c: bool, w: &str| out.push((c, String::from(w)));

    fn hex(d: &[u8; 32]) -> alloc::string::String {
        let mut s = alloc::string::String::new();
        for b in d {
            s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            s.push(char::from_digit((b & 15) as u32, 16).unwrap());
        }
        s
    }

    ok(
        hex(&keccak256(b"")) == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
        "keccak256 of the empty string is its published value",
    );
    ok(
        hex(&keccak256(b"abc")) == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45",
        "and of 'abc'",
    );
    // The one that separates this from SHA3-256, which differs only in the
    // padding byte and would pass anything that merely checks self-consistency.
    ok(
        hex(&keccak256(b"")) != "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
        "and it is not SHA3-256, which differs only in one padding byte",
    );

    // A message that exactly fills the rate: the padding has to land in a block
    // of its own, and an implementation that appends before deciding whether to
    // permute produces a digest for a 136-byte input identical to some other
    // length's.
    let exact = [0x61u8; RATE];
    let over = [0x61u8; RATE + 1];
    ok(keccak256(&exact) != keccak256(&over), "a message filling the rate is padded into its own block");
    ok(keccak256(&exact) != keccak256(&exact[..RATE - 1]), "and is not the same as one byte fewer");

    // Absorbing more than one block at all.
    ok(keccak256(&[0u8; 300]) != keccak256(&[0u8; 200]), "several blocks absorb distinctly");

    out
}

/// The boot form. Prints nothing on success, for the reason every other
/// `selftest` in this module does not: the section prints one line per entry and
/// a primitive that narrated its own vectors would bury the ones that failed.
pub fn selftest() -> bool {
    let mut ok = true;
    for (good, what) in checks() {
        if !good {
            crate::kprintln!("    FAIL: {}", what);
            ok = false;
        }
    }
    ok
}
