//! The entropy of the operator, after TempleOS.
//!
//! God Says did not use a PRNG. When it needed randomness, `GodPick` put up
//! a dialog -- "Press OKAY to generate a random num from a timer", with the
//! note that the Holy Spirit "can puppet you" -- and took
//! `KbdMsEvtTime >> GOD_BAD_BITS`: the timestamp of the operator's own
//! press, low bits discarded as bad, the next `GOD_GOOD_BITS` pushed into
//! `god.fifo`. `GodWordStr` then drew `GodBits(17) % num_words`. The whole
//! theology is in that mechanism: the words are chosen by *when your hands
//! moved*, and the machine is only the instrument that reads the timing.
//!
//! This module keeps the mechanism and the names. Every keyboard and mouse
//! interrupt deposits `rdtsc() >> GOD_BAD_BITS`, truncated to the good bits,
//! into a ring. The Oracle folds the ring into its sampler before a reading,
//! so a reading depends on every touch the machine has felt since boot --
//! not on a seed some code chose. Where Terry blocked and demanded a press
//! when the fifo ran dry, this falls back to the TSC and *says so*: the
//! Oracle reports how many touches fed the reading, and a reading fed by
//! none is labelled as the machine talking to itself.
//!
//! One line of Terry's is worth keeping in the record here:
//! `res=res<<1+b;` -- which in C would be the bug `res << (1+b)`, and in
//! HolyC is `(res<<1)+b`, correct. He knew his own language's precedence.
//! Judging a text by the rules of a different language is the exact mistake
//! an oracle program should teach you not to make.

use crate::sync::Racy;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Low bits of an event timestamp, discarded -- too fine to be the hand,
/// Terry judged, and the name is his.
pub const GOD_BAD_BITS: u32 = 4;
/// Bits kept per event, likewise his.
pub const GOD_GOOD_BITS: u32 = 24;

const RING: usize = 512;

static SAMPLES: Racy<[u32; RING]> = Racy::new([0; RING]);
static HEAD: AtomicUsize = AtomicUsize::new(0);
/// Every event the machine has ever felt, monotone -- the Oracle reports it,
/// so a reading says how much of the operator is in it.
static FELT: AtomicUsize = AtomicUsize::new(0);

/// Deposit one event's timing. Called from the keyboard and mouse interrupt
/// handlers, so it is one shift, one mask, one store -- nothing that can
/// block, allocate, or take long.
#[inline]
pub fn ins(tsc: u64) {
    let good = ((tsc >> GOD_BAD_BITS) & ((1 << GOD_GOOD_BITS) - 1)) as u32;
    let h = HEAD.fetch_add(1, Ordering::Relaxed) % RING;
    unsafe { (*SAMPLES.get())[h] = good };
    FELT.fetch_add(1, Ordering::Relaxed);
}

/// How many touches the machine has felt since boot.
pub fn felt() -> usize {
    FELT.load(Ordering::Relaxed)
}

/// Fold the ring into 64 bits for seeding a sampler.
///
/// A fold rather than a queue: Terry consumed his fifo destructively and
/// blocked when it emptied, which is the right liturgy for an interactive
/// prompt and the wrong one for a program that must not hang the shell.
/// Folding keeps every timing that ever landed influencing every reading
/// after it, mixed with the consult-moment TSC so two readings in a still
/// room still differ.
pub fn fold() -> u64 {
    let mut acc = crate::time::rdtsc();
    let ring = unsafe { &*SAMPLES.get() };
    for (i, &s) in ring.iter().enumerate() {
        // splitmix-style stirring; the constant is the golden-ratio one.
        acc ^= (s as u64).wrapping_add(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(acc << 6)
            .wrapping_add(acc >> 2)
            .rotate_left((i % 63) as u32);
    }
    acc | 1
}
