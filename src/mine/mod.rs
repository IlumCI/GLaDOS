//! The lottery wing: proof-of-work hashing inside a ring-0 kernel.
//!
//! Economics first, because everything else in this module is engineering
//! sport. A single modern core does roughly 10-20 million sha256d hashes
//! per second; Bitcoin's network does about 7e20. Winning a block solo at
//! that ratio is a once-per-two-million-years event -- which makes this a
//! lottery ticket with excellent engineering and terrible expected value.
//! The number is printed by the commands here, on every run, so nobody
//! involved is allowed to forget it.
//!
//! What the kernel actually contributes:
//! - Midstate caching: an 80-byte header's first 64 bytes never change per
//!   template, so they are absorbed once and cloned; each nonce costs one
//!   compression plus one short finalisation instead of two full hashes.
//! - Zero scheduling noise for timing: `lapic::ticks` measures work without
//!   asking an operating system what time it thinks it is.
//! - A path to real speed later: SHA-NI (`sha256rnds2` et al.) on this
//!   i7-12650H has never been exercised anywhere in this tree, AVX2
//!   multi-buffer hashing would run eight lanes at once, and SMP bring-up
//!   would multiply both by every core that currently sits parked from
//!   boot. Those are upgrades to this file, not rewrites of it.

use crate::store::sha256::{hash, Sha256};
use crate::kprintln;
use alloc::format;
use alloc::string::String;

/// Bitcoin mainnet hash rate, hashes per second. Order-of-magnitude figure
/// as of early 2026; only ever used for the odds line, never for anything
/// that pretends to be precise.
const BTC_NETWORK_HPS: f64 = 7.0e20;

/// One synthetic 80-byte block header: version, previous hash, merkle root,
/// time, bits, nonce. There is no node feed yet -- stratum arrives later --
/// so the template is deterministic filler with a timestamp from the timer,
/// which keeps consecutive runs sweeping different nonce spaces even though
/// nothing here is a valid network block. That distinction is the whole
/// point of the word "lottery".
fn template() -> ([u8; 80], [u32; 8], u64) {
    let mut h = [0u8; 80];
    h[0..4].copy_from_slice(&0x2000_0000u32.to_le_bytes());
    let prev = hash(b"glados prev block placeholder");
    h[4..36].copy_from_slice(&prev);
    let merkle = hash(b"glados merkle placeholder");
    h[36..68].copy_from_slice(&merkle);
    let t = (crate::dev::lapic::ticks() / crate::TIMER_HZ as u64) as u32;
    h[68..72].copy_from_slice(&t.to_le_bytes());
    h[72..76].copy_from_slice(&0x1702_9EB3u32.to_be_bytes()); // bits, mainnet-ish
    // Midstate over everything but the nonce field: absorbed once, cloned
    // per attempt.
    let mut mid = Sha256::new();
    mid.update(&h[..64]);
    let (state, bits) = mid.snapshot();
    (h, state, bits)
}

fn leading_zero_bits(d: &[u8; 32]) -> u32 {
    let mut n = 0u32;
    for &b in d {
        if b == 0 {
            n += 8;
        } else {
            n += b.leading_zeros();
            break;
        }
    }
    n
}

#[inline]
fn sha256d_tail(state: ([u32; 8], u64), tail: &[u8]) -> [u8; 32] {
    let mut st = Sha256::from_snapshot(state.0, state.1);
    st.update(tail);
    let d1 = st.finish();
    hash(&d1)
}

fn bench_batch(header: &mut [u8; 80], state: ([u32; 8], u64), n: u32) -> u32 {
    let mut best = 0u32;
    for i in 0..n {
        header[76..80].copy_from_slice(&i.to_le_bytes());
        let d2 = sha256d_tail(state, &header[64..80]);
        let z = leading_zero_bits(&d2);
        if z > best {
            best = z;
        }
    }
    best
}

/// Hash-rate measurement over `seconds`, printing rate and the luckiest
/// hash found. The honest scoreboard for every future optimisation:
/// SHA-NI, multi-buffer AVX2 and SMP all answer to this one number.
pub fn bench(seconds: u64) -> String {
    let (mut header, state, bits) = template();
    let t0 = crate::dev::lapic::ticks();
    let mut done = 0u64;
    let mut best = 0u32;
    loop {
        best = best.max(bench_batch(&mut header, (state, bits), 65536));
        done += 65536;
        let elapsed =
            (crate::dev::lapic::ticks() - t0) as f64 / crate::TIMER_HZ as f64;
        if elapsed >= seconds as f64 {
            let rate = done as f64 / elapsed;
            return format!(
                "{} hashes in {:.1}s = {:.2} MH/s (best run: {} zero bits)",
                done,
                elapsed,
                rate / 1e6,
                best
            );
        }
    }
}

/// Solo sweep against a difficulty expressed as required leading zero bits.
/// Returns the nonce when a qualifying hash appears; prints the odds line
/// against mainnet either way, because a lottery ticket that hides its
/// odds is just a tax with extra steps.
pub fn lotto(seconds: u64, difficulty_bits: u32) -> Option<(u32, String)> {
    let (mut header, state, bits) = template();
    let t0 = crate::dev::lapic::ticks();
    let mut done = 0u64;
    let mut i = 0u32;
    loop {
        for _ in 0..4096 {
            header[76..80].copy_from_slice(&i.to_le_bytes());
            let d2 = sha256d_tail((state, bits), &header[64..80]);
            if leading_zero_bits(&d2) >= difficulty_bits {
                let hex: String = d2.iter().map(|b| format!("{:02x}", b)).collect();
                kprintln!(
                    "[mine] FOUND after {} hashes: nonce {} -> {}",
                    done + 1,
                    i,
                    hex
                );
                return Some((i, hex));
            }
            i = i.wrapping_add(1);
            done += 1;
        }
        let elapsed =
            (crate::dev::lapic::ticks() - t0) as f64 / crate::TIMER_HZ as f64;
        if elapsed >= seconds as f64 {
            break;
        }
    }
    let per_day = done as f64 * (86400.0 / seconds.max(1) as f64);
    let odds = BTC_NETWORK_HPS / per_day.max(1.0);
    kprintln!(
        "[mine] {} hashes in {}s -- vs bitcoin mainnet these are 1-in-{:.3e} daily odds",
        done,
        seconds,
        odds
    );
    None
}
