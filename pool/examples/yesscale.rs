//! How yespower actually scales across this machine's cores.
//!
//! `--bench` times one hash on one core, which is the right question for share
//! *validation* and the wrong one for mining: a miner runs every core at once
//! and yespower is memory-hard, so the working sets contend. 2 MiB a thread
//! against 24 MB of L3 means sixteen threads want 32 MiB and cannot have it.
//!
//! `design/mining.md`'s revenue figure comes from a rate measured under
//! emulation on a quarter of the cores running reference code, and the tool
//! that quotes it flags it as a floor. This is the same quantity measured
//! natively, on every core, so the floor can be replaced with a number.
use glados_pool::mine::algo::{Algo, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn run(algo: &Algo, threads: usize, secs: f64) -> f64 {
    let total = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicU64::new(0));
    let mut hs = Vec::new();
    for t in 0..threads {
        let algo = algo.clone();
        let total = Arc::clone(&total);
        let stop = Arc::clone(&stop);
        hs.push(std::thread::spawn(move || {
            let header = [0u8; 80];
            // Built inside the thread: the working set is per-thread and
            // allocating it on one core then using it on another is what
            // makes a scaling curve measure the allocator.
            let mut h = match Hasher::new(&algo, &header) {
                Some(h) => h,
                None => return,
            };
            let mut n = t as u32;
            let mut done = 0u64;
            while stop.load(Ordering::Relaxed) == 0 {
                for _ in 0..8 {
                    std::hint::black_box(h.hash(&header, n));
                    n = n.wrapping_add(1);
                    done += 1;
                }
            }
            total.fetch_add(done, Ordering::Relaxed);
        }));
    }
    std::thread::sleep(Duration::from_secs_f64(secs));
    let t0 = Instant::now();
    stop.store(1, Ordering::Relaxed);
    for h in hs {
        let _ = h.join();
    }
    let elapsed = secs + t0.elapsed().as_secs_f64();
    total.load(Ordering::Relaxed) as f64 / elapsed
}

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(6.0);
    let algo = Algo::Yespower { v10: true, n: 2048, r: 8, pers: None };
    println!("yespower 1.0 N=2048 r=8 (2 MiB a thread), {secs}s a point\n");
    println!("{:>8}  {:>12}  {:>10}  {:>9}", "threads", "H/s", "per thread", "scaling");
    let mut one = 0.0;
    for t in [1usize, 2, 4, 8, 12, 16] {
        let r = run(&algo, t, secs);
        if t == 1 {
            one = r;
        }
        println!("{:>8}  {:>12.1}  {:>10.1}  {:>8.2}x", t, r, r / t as f64, r / one);
    }
}
