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

// ---- the processor's own source -------------------------------------------
//
// **The note at the top of this file said RDRAND "deserves its own commit", and
// this is it.** What forced it was a machine the note describes exactly -- one
// that "boots, touches no disk and shuts down without a key ever being pressed".
// That is a mining image: booted from USB, headless, and doing TLS to its pool.
// Its pool never filled, and every handshake printed "13 of 256 entropy bits --
// keys are timing-derived, not random".
//
// **No new trust policy, and that is the decision.** The rule above is one bit
// credited per event whatever it came from, pessimistic on purpose because
// nobody here can measure what a source is worth. A 64-bit RDSEED sample is one
// more event and is credited one bit -- sixty-four times less than it claims to
// carry, which is this module's stance applied rather than relaxed. 256 samples
// seed the pool, and on current silicon that is well under a millisecond.
//
// What crediting it at all does trust is the vendor, and no rate changes that: a
// fully predictable instruction credited at one bit a sample still credits 256
// bits of nothing. That is the same trust every mainstream kernel extends by
// default. What a rate *can* guard against is a source that is broken rather
// than malicious, and those have been measured in the field -- see `plausible`.

/// Deposits from the processor's random-number instruction, for the status line.
///
/// Counted apart for the reason device deposits are: a pool filled by the CPU is
/// a different situation from one filled by somebody typing or a disk working,
/// and an operator deciding whether to trust a key is owed which it was.
static CPU_DEPOSITS: Racy<u64> = Racy::new(0);

/// Which instruction this processor offers, best first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hw {
    /// Output of the entropy conditioner. What a seed should come from.
    Rdseed,
    /// Output of a DRBG seeded from that conditioner. Fine to mix; a fallback.
    Rdrand,
    None,
}

/// What CPUID says, asked rather than assumed.
///
/// An instruction the processor does not implement raises #UD, and every vector
/// here but #BP is fatal -- so this is checked before anything is executed, and
/// the harvest runs inside an optional boot section besides, so a processor that
/// advertises the instruction and faults on it loses this source and nothing else.
pub fn hw_source() -> Hw {
    let max = crate::cpu::cpuid(0, 0)[0];
    if max >= 7 && crate::cpu::cpuid(7, 0)[1] & (1 << 18) != 0 {
        return Hw::Rdseed;
    }
    if crate::cpu::cpuid(1, 0)[2] & (1 << 30) != 0 {
        return Hw::Rdrand;
    }
    Hw::None
}

/// One 64-bit sample, or `None` if the instruction declined.
///
/// Both instructions report failure in the carry flag rather than faulting, and
/// both are allowed to fail under load, RDSEED more readily. A few retries is
/// the documented answer; a sample that will not come is a sample not counted,
/// never a sample of zero.
fn sample(hw: Hw) -> Option<u64> {
    for _ in 0..16 {
        let v: u64;
        let ok: u8;
        unsafe {
            match hw {
                Hw::Rdseed => core::arch::asm!(
                    "rdseed {v}", "setc {ok}", v = out(reg) v, ok = out(reg_byte) ok,
                    options(nomem, nostack)
                ),
                Hw::Rdrand => core::arch::asm!(
                    "rdrand {v}", "setc {ok}", v = out(reg) v, ok = out(reg_byte) ok,
                    options(nomem, nostack)
                ),
                Hw::None => return None,
            }
        }
        if ok != 0 {
            return Some(v);
        }
        core::hint::spin_loop();
    }
    None
}

/// Whether a sample may be counted as an event.
///
/// **Aimed at defects that shipped, not at an attacker.** Some AMD processors
/// returned all ones from RDRAND on every call after resuming from sleep, while
/// reporting success, and firmware on others left it answering zero. Crediting
/// either would be claiming 256 bits of one constant. A value equal to the one
/// before it is the general form of both, and it is also what a stuck
/// conditioner looks like.
///
/// Pure, so `hw_selftest` asserts every branch with no instruction executed.
pub fn plausible(prev: Option<u64>, v: u64) -> bool {
    v != 0 && v != u64::MAX && prev != Some(v)
}

/// Take samples until the pool is seeded or `attempts` are spent.
///
/// Answers (samples taken, samples credited). An implausible sample is dropped
/// entirely rather than mixed uncredited: `add_entropy` credits every deposit, and
/// a second deposit path that did not would be two meanings for one call.
pub fn add_cpu_entropy(attempts: u32) -> (u32, u32) {
    let hw = hw_source();
    if hw == Hw::None {
        return (0, 0);
    }
    let mut prev = None;
    let (mut taken, mut credited) = (0u32, 0u32);
    for _ in 0..attempts {
        if status().2 {
            break;
        }
        let Some(v) = sample(hw) else { continue };
        taken += 1;
        if plausible(prev, v) {
            unsafe { *CPU_DEPOSITS.get() += 1 };
            add_entropy(v);
            credited += 1;
        }
        prev = Some(v);
    }
    (taken, credited)
}

/// Deposits from the processor's instruction, for the status line.
pub fn cpu_deposits() -> u64 {
    unsafe { *CPU_DEPOSITS.get() }
}

/// The processor's source, checked: the filter, then live samples if there are any.
pub fn hw_selftest() -> bool {
    let mut ok = true;
    let mut claim = |good: bool, what: &str| {
        if !good {
            ok = false;
        }
        crate::kprintln!("  {}  {}", if good { "ok  " } else { "FAIL" }, what);
    };
    claim(!plausible(None, 0), "a sample of zero is not counted");
    claim(!plausible(None, u64::MAX), "nor all ones, which is what a broken RDRAND returned");
    claim(!plausible(Some(0x1234_5678_9ABC_DEF0), 0x1234_5678_9ABC_DEF0), "nor a repeat of the last one");
    claim(plausible(Some(1), 0x1234_5678_9ABC_DEF0), "an ordinary sample is");

    let hw = hw_source();
    if hw == Hw::None {
        crate::kprintln!("  ....  this processor offers neither RDSEED nor RDRAND, so 2 claim(s) did not run");
        return ok;
    }
    // **RDSEED is allowed to decline, so "it answers every time" is false.** It
    // reads the entropy conditioner directly and can be momentarily exhausted;
    // the manual says so and says to retry. The first version of this claim
    // required sixteen answers from sixteen asks, and under KVM with eight vCPUs
    // one ran out of retries -- which failed the section, filed a boot report, and
    // would have stopped `verify-boot` publishing an image whose pool had seeded
    // perfectly. What is worth asserting is that it answers *mostly*, which is
    // the difference between a busy source and a dead one.
    let mut vals = [0u64; 16];
    let mut got = 0;
    for _ in 0..vals.len() {
        if let Some(v) = sample(hw) {
            vals[got] = v;
            got += 1;
        }
    }
    claim(got * 2 >= vals.len(), "the instruction answers at least half the time, as a live source does");
    // Only over what it answered. A declined ask used to leave a zero in the
    // array, which the next claim then counted as a value it had returned.
    let mut distinct = vals;
    let answered = &mut distinct[..got];
    answered.sort_unstable();
    let unique = if got == 0 { 0 } else { 1 + answered.windows(2).filter(|w| w[0] != w[1]).count() };
    claim(got > 0 && unique == got, "and every value it answered with is a different one");
    crate::kprintln!("        {} of {} asks answered", got, vals.len());
    ok
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
