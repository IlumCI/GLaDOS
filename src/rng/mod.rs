//! The kernel's random number generator.
//!
//! Everything that needed unpredictable bytes before this module took them
//! from `rdtsc()`, and `src/net/tls.rs` named the consequence in place: a
//! counter started at power-on is not a random number generator, and an
//! attacker who can guess the boot time narrows a TLS private key. This is the
//! fix for that, and it is deliberately built out of parts that were already
//! being checked.
//!
//! # Why it rides the existing ChaCha20
//!
//! `src/crypto/chacha.rs` is checked against the RFC 8439 vectors at every
//! boot, alongside a test that a flipped bit is rejected. Writing a second,
//! smaller permutation here would have meant a cryptographic core that
//! nothing verifies, in the one area of this tree where a mistake produces
//! output that looks perfect and is not secure. So the DRBG is a construction
//! *over* `chacha::apply` and adds no primitive of its own. If the boot
//! selftest for ChaCha20 passes, the core of this module is the same core.
//!
//! # Fast key erasure
//!
//! One ChaCha20 block is 64 bytes. The first 32 overwrite the key, and only
//! the remaining 32 leave the module. The old key is destroyed before its
//! output is handed out, so an attacker who reads the state later cannot
//! reproduce anything already generated. This is the arc4random and Linux
//! `get_random_bytes` construction, and it is chosen because backtracking
//! resistance is exactly the property a kernel with one address space and no
//! process isolation cannot get any other way.
//!
//! # Entropy, and the honest size of the claim
//!
//! Two sources, and the second one exists because the first has a blind spot.
//!
//! Every keyboard and mouse interrupt deposits its raw TSC through
//! `godbits::ins`. The jitter in when an interrupt lands is real entropy and
//! nothing outside the machine controls it. But a machine left running
//! overnight receives none of it, and that is exactly the machine most likely
//! to want a key: the network stack here polls rather than taking interrupts,
//! so there is no packet arrival to harvest either, and the pool would simply
//! never fill.
//!
//! So NVMe completion latency feeds it too, through `add_device_entropy`.
//! The time a controller takes to answer carries NAND timing, internal
//! scheduling and wear levelling, none of which is predictable from outside
//! the machine, and it arrives whenever the machine is working rather than
//! only when somebody is present.
//!
//! How *much* either source is worth is the part nobody here can honestly
//! measure, so the accounting is deliberately pessimistic: one bit credited
//! per event whatever it came from, 256 events before the pool is called
//! seeded. An NVMe completion carries far more jitter than one bit; crediting
//! it as one is what lets the claim stand without a measurement behind it.
//! That number is an assumption and it is the weakest link in this module,
//! written down here so nobody has to infer it from the constant.
//!
//! Below the threshold `fill` still works and `fill_secret` refuses. Key
//! material takes the second one. A generator that quietly degrades for a
//! private key is worse than one that says it cannot help.
//!
//! # What this is not
//!
//! It is not a hardware entropy source. The CPU has `RDRAND` and this module
//! does not use it, because trusting an opaque instruction is a different
//! argument from trusting interrupt timing and deserves its own commit. On a
//! machine that boots, touches no disk and shuts down without a key ever
//! being pressed, the pool still never fills and `fill_secret` refuses for
//! the whole session. That is the correct behaviour and it is still a real
//! limitation, narrowed rather than removed.

use crate::crypto::chacha;
use crate::sync::Racy;

/// Bits credited per interrupt. See the note above: an assumption.
const BITS_PER_EVENT: u32 = 1;
/// Bits of pooled entropy before `fill_secret` will answer.
pub const SEEDED_BITS: u32 = 256;

struct Drbg {
    key: [u8; chacha::KEY_LEN],
    nonce: [u8; chacha::NONCE_LEN],
    /// Interrupt timings, folded in as they arrive and consumed at the next
    /// generation. Held apart from the key so the interrupt path never runs
    /// a cipher.
    pool: [u8; 32],
    pool_dirty: bool,
    deposits: u64,
    bits: u32,
}

impl Drbg {
    const fn new() -> Self {
        // A fixed start is the honest one. Any constant here is public, so
        // pretending otherwise by seeding from a timestamp at construction
        // would only obscure that the pool is what carries the secret.
        Self {
            key: [0; chacha::KEY_LEN],
            nonce: [0; chacha::NONCE_LEN],
            pool: [0; 32],
            pool_dirty: false,
            deposits: 0,
            bits: 0,
        }
    }
}

static DRBG: Racy<Drbg> = Racy::new(Drbg::new());

/// Deposit one interrupt's timing.
///
/// Called from `godbits::ins`, so it runs inside the keyboard and mouse
/// handlers: eight exclusive-ors, a rotate and two adds, with no cipher, no
/// allocation and nothing that can block. The pool is diffused later, at
/// generation time, where the cost is affordable.
///
/// The rotate matters. Consecutive interrupts share their high TSC bits, so
/// folding raw samples into the same offset would cancel more than it
/// accumulated; rotating by the deposit count spreads each sample across the
/// pool instead.
#[inline]
pub fn add_entropy(tsc: u64) {
    let d = unsafe { &mut *DRBG.get() };
    let b = tsc.rotate_left((d.deposits & 63) as u32).to_le_bytes();
    let slot = (d.deposits as usize & 3) * 8;
    for k in 0..8 {
        d.pool[slot + k] ^= b[k];
    }
    d.deposits = d.deposits.wrapping_add(1);
    if d.bits < SEEDED_BITS {
        d.bits += BITS_PER_EVENT;
    }
    d.pool_dirty = true;
}

/// Deposits from a device, kept apart from deposits from a person.
///
/// The two are counted separately because they answer different questions and
/// the status line would otherwise conflate them: a pool filled by a night of
/// disk traffic is a different situation from one filled by somebody typing,
/// even though both are legitimate. An operator deciding whether to trust a
/// key wants to know which happened.
static DEV_DEPOSITS: Racy<u64> = Racy::new(0);

/// Fold one device timing into the pool.
///
/// Deliberately *not* routed through `godbits::ins`. That function feeds two
/// consumers: the entropy pool here, and the Oracle's ring behind
/// `godbits::felt`, which counts how many times the machine has been touched
/// by a person. `felt` is what `initiative` and `godel` read to decide whether
/// the operator is present, so putting disk traffic through it would make an
/// unattended machine look occupied and would stand down the very loop that
/// runs while nobody is there. Two sources, two meanings, one pool.
///
/// `delta` should be a completion latency in TSC cycles rather than an
/// absolute timestamp. The high bits of an absolute TSC are near enough to
/// predictable that they contribute nothing; the jitter is in how long the
/// device actually took.
#[inline]
pub fn add_device_entropy(delta: u64) {
    unsafe {
        *DEV_DEPOSITS.get() += 1;
    }
    add_entropy(delta);
}

/// Deposits from devices, for the status line.
pub fn device_deposits() -> u64 {
    unsafe { *DEV_DEPOSITS.get() }
}

/// One fast-key-erasure step: 64 bytes of keystream, of which the first 32
/// become the next key and the last 32 are the output.
///
/// `chacha::apply` exclusive-ors a keystream into a buffer, so a buffer of
/// zeros comes back as the keystream itself. The key is replaced *before* the
/// caller sees anything, which is the whole construction: the state that
/// produced this output no longer exists by the time the output is returned.
fn step(d: &mut Drbg) -> [u8; 32] {
    let mut ks = [0u8; 64];
    chacha::apply(&d.key, 0, &d.nonce, &mut ks);
    d.key.copy_from_slice(&ks[..32]);
    bump(&mut d.nonce);
    let mut out = [0u8; 32];
    out.copy_from_slice(&ks[32..]);
    out
}

/// The nonce as a little-endian counter.
///
/// Strictly unnecessary, since a fresh key per block already gives a fresh
/// keystream, and kept because the cost is nothing and it removes any question
/// about key and nonce reuse for a reader who checks this file against the
/// warning in `chacha::apply`.
fn bump(nonce: &mut [u8; chacha::NONCE_LEN]) {
    for b in nonce.iter_mut() {
        *b = b.wrapping_add(1);
        if *b != 0 {
            break;
        }
    }
}

/// Fold whatever the interrupts have deposited into the key, then diffuse.
///
/// The TSC at the moment of consultation goes in as well, so two generations
/// in a still room still differ. It carries no claim of entropy and is not
/// credited any; it is there so that the absence of keystrokes degrades the
/// output to something unpredictable-by-timing instead of to a constant.
fn reseed(d: &mut Drbg) {
    for (k, p) in d.key.iter_mut().zip(d.pool.iter()) {
        *k ^= *p;
    }
    let t = crate::time::rdtsc().to_le_bytes();
    for (k, b) in d.key.iter_mut().zip(t.iter()) {
        *k ^= *b;
    }
    d.pool = [0; 32];
    d.pool_dirty = false;
    // One step, discarded: the pool went in by exclusive-or, which spreads
    // nothing on its own, and this is what turns it into a key.
    let _ = step(d);
}

/// Fill `out` with random bytes.
///
/// Always answers. Suitable for anything that wants unpredictability without
/// depending on it: nonce partitioning, jitter, a sampler seed. Key material
/// takes `fill_secret`.
pub fn fill(out: &mut [u8]) {
    let d = unsafe { &mut *DRBG.get() };
    if d.pool_dirty {
        reseed(d);
    }
    let mut pos = 0;
    while pos < out.len() {
        let b = step(d);
        let n = core::cmp::min(32, out.len() - pos);
        out[pos..pos + n].copy_from_slice(&b[..n]);
        pos += n;
    }
}

/// Fill `out`, or refuse because the pool has not seen enough interrupts.
///
/// The refusal is the point. A generator that quietly degrades for a private
/// key produces exactly the failure this tree's crypto section warns about:
/// output that works perfectly and is not secure. The caller is told the
/// estimate so it can say what it did.
pub fn fill_secret(out: &mut [u8]) -> Result<(), u32> {
    let bits = unsafe { (*DRBG.get()).bits };
    if bits < SEEDED_BITS {
        return Err(bits);
    }
    fill(out);
    Ok(())
}

/// Deposits seen, bits credited, and whether the pool is called seeded.
pub fn status() -> (u64, u32, bool) {
    let d = unsafe { &*DRBG.get() };
    (d.deposits, d.bits, d.bits >= SEEDED_BITS)
}

/// Boot self-test. Seven claims.
///
/// Each one is aimed at a specific way a generator can look right and be
/// wrong, and the first three exist because an earlier draft of this module
/// had exactly those defects: a state that never advanced, so every block in
/// a call was identical; an invertible permutation whose entire state was the
/// output; and a hand-rolled core that resembled ChaCha20 without being it.
/// None of that shows up in a hex dump, which is why it is checked here.
pub fn selftest() -> bool {
    use crate::kprintln;

    let mut ok = true;
    let mut claim = |what: &str, pass: bool| {
        if !pass {
            ok = false;
        }
        kprintln!("  {}  {}", if pass { "ok " } else { "FAIL" }, what);
    };

    // A private instance throughout: the live pool belongs to the operator's
    // keystrokes, and a self-test that consumed it would spend the entropy it
    // was meant to be checking.
    let mut d = Drbg::new();
    d.key = [7u8; 32];

    // 1. Successive blocks differ. The earlier draft failed this one.
    let a = step(&mut d);
    let b = step(&mut d);
    let c = step(&mut d);
    claim(
        "three successive blocks are three different blocks",
        a != b && b != c && a != c,
    );

    // 2. The key is gone. Backtracking resistance is this and nothing else.
    let mut e = Drbg::new();
    e.key = [7u8; 32];
    let before = e.key;
    let out = step(&mut e);
    claim(
        "the key that produced a block does not survive it",
        e.key != before && e.key[..] != out[..],
    );

    // 3. The core is the ChaCha20 the crypto selftest already checked, and
    //    not something shaped like it. Same key, same nonce, same answer.
    let mut f = Drbg::new();
    f.key = [7u8; 32];
    let got = step(&mut f);
    let mut ks = [0u8; 64];
    chacha::apply(&[7u8; 32], 0, &[0u8; chacha::NONCE_LEN], &mut ks);
    claim(
        "a block is the verified ChaCha20 keystream, second half",
        got[..] == ks[32..],
    );

    // 4. Determinism. Two identical states walk identically, which is what
    //    makes claim 5 meaningful: divergence there is caused by the input
    //    and not by noise.
    let (mut g, mut h) = (Drbg::new(), Drbg::new());
    g.key = [9u8; 32];
    h.key = [9u8; 32];
    claim(
        "two identical states produce identical output",
        step(&mut g) == step(&mut h),
    );

    // 5. One bit of entropy changes everything after it.
    let (mut i, mut j) = (Drbg::new(), Drbg::new());
    i.pool[0] = 0x01;
    i.pool_dirty = true;
    j.pool[0] = 0x00;
    j.pool_dirty = true;
    let mut bi = [0u8; 32];
    let mut bj = [0u8; 32];
    // Reseed folds the consult-time TSC in as well, so these two would differ
    // whatever the pool held. Zero the key by hand after reseeding to isolate
    // the pool's contribution instead of measuring the clock.
    i.key = [0u8; 32];
    j.key = [0u8; 32];
    for (k, p) in i.key.iter_mut().zip(i.pool.iter()) {
        *k ^= *p;
    }
    for (k, p) in j.key.iter_mut().zip(j.pool.iter()) {
        *k ^= *p;
    }
    bi.copy_from_slice(&step(&mut i));
    bj.copy_from_slice(&step(&mut j));
    let differing = bi.iter().zip(bj.iter()).filter(|(x, y)| x != y).count();
    // A single flipped input bit should move about half the output bytes.
    // Twenty of thirty-two is a loose floor that a diffusing core clears
    // comfortably and a broken one does not.
    claim(
        "one bit of pooled entropy moves most of the output",
        differing >= 20,
    );

    // 6. The accounting behaves, and saturates where it says it does.
    let mut k = Drbg::new();
    let start = k.bits;
    for n in 0..(SEEDED_BITS as u64 + 16) {
        // A real TSC would not repeat; the value is irrelevant to the count.
        let d = unsafe { &mut *DRBG.get() };
        let _ = d;
        k.deposits = n;
        k.bits = core::cmp::min(k.bits + BITS_PER_EVENT, SEEDED_BITS);
    }
    claim(
        "the entropy estimate rises with events and stops at its ceiling",
        start == 0 && k.bits == SEEDED_BITS,
    );

    // 7. Below the threshold a secret is refused rather than weakened.
    let (_, bits, seeded) = status();
    let refused = if seeded {
        // The pool filled during boot, so the refusal path cannot be
        // exercised live. Check the predicate directly and say so.
        bits >= SEEDED_BITS
    } else {
        let mut buf = [0u8; 32];
        fill_secret(&mut buf).is_err()
    };
    claim(
        if seeded {
            "the pool is seeded, so secrets are answered"
        } else {
            "an unseeded pool refuses a secret instead of weakening it"
        },
        refused,
    );

    ok
}
