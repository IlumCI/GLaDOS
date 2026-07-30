//! Machine learning primitives.

pub mod tensor;

use crate::gfx::console::{self, LTGREEN, LTRED, WHITE, YELLOW};
use crate::kprintln;
use alloc::vec;
use alloc::vec::Vec;

fn close(a: f32, b: f32, tol: f32) -> bool {
    let d = if a > b { a - b } else { b - a };
    d <= tol * (1.0 + if a > 0.0 { a } else { -a })
}

fn check(name: &str, ok: bool, detail: &str) -> bool {
    if ok {
        console::set_color(LTGREEN);
        kprintln!("  ok    {:<22} {}", name, detail);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL  {:<22} {}", name, detail);
    }
    console::set_color(WHITE);
    ok
}

/// Check the primitives against hand-computed values, and the SIMD kernel
/// against the scalar one.
///
/// The scalar path is the oracle. A hand-written AVX kernel that is subtly
/// wrong -- a misaligned tail, a bad horizontal sum -- produces numbers that
/// look entirely plausible, so "it ran without faulting" proves nothing.
pub fn selftest() -> bool {
    let mut all = true;
    console::set_color(YELLOW);
    kprintln!("[tensor]");
    console::set_color(WHITE);

    // --- math ---
    all &= check(
        "sqrtf(2)",
        close(tensor::sqrtf(2.0), 1.414_213_6, 1e-6),
        "expect 1.4142136",
    );
    all &= check("expf(0)", close(tensor::expf(0.0), 1.0, 1e-6), "expect 1");
    all &= check(
        "expf(1)",
        close(tensor::expf(1.0), 2.718_281_7, 1e-5),
        "expect 2.7182817",
    );
    all &= check(
        "expf(-5)",
        close(tensor::expf(-5.0), 0.006_737_947, 1e-4),
        "expect 0.006737947",
    );
    all &= check(
        "lnf(e)",
        close(tensor::lnf(2.718_281_7), 1.0, 1e-5),
        "expect 1",
    );
    all &= check(
        "sinf(pi/2)",
        close(tensor::sinf(core::f32::consts::FRAC_PI_2), 1.0, 1e-4),
        "expect 1",
    );
    all &= check(
        "cosf(0)",
        close(tensor::cosf(0.0), 1.0, 1e-4),
        "expect 1",
    );

    // --- matmul against a hand-computed case ---
    // w = [[1,2,3],[4,5,6]], x = [1,2,3] -> [14, 32]
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = [1.0f32, 2.0, 3.0];
    let mut out = [0.0f32; 2];
    tensor::matmul_scalar(&mut out, &x, &w, 3, 2);
    all &= check(
        "matmul scalar",
        close(out[0], 14.0, 1e-6) && close(out[1], 32.0, 1e-6),
        "expect [14, 32]",
    );

    // --- softmax ---
    let mut s = [1.0f32, 2.0, 3.0];
    tensor::softmax(&mut s);
    let sum = s[0] + s[1] + s[2];
    all &= check(
        "softmax sums to 1",
        close(sum, 1.0, 1e-5) && s[2] > s[1] && s[1] > s[0],
        "and is monotonic",
    );

    // --- rmsnorm ---
    let xs = [3.0f32, 4.0];
    let ws = [1.0f32, 1.0];
    let mut rn = [0.0f32; 2];
    tensor::rmsnorm(&mut rn, &xs, &ws);
    // rms = sqrt((9+16)/2) = 3.5355; 3/3.5355 = 0.8485
    all &= check(
        "rmsnorm",
        close(rn[0], 0.848_5, 1e-3) && close(rn[1], 1.131_4, 1e-3),
        "expect [0.8485, 1.1314]",
    );

    // --- SIMD against the scalar oracle, on a size with a ragged tail ---
    let f = crate::cpu::detected();
    if f.avx_enabled && f.fma {
        let n = 173usize; // deliberately not a multiple of 8 or 32
        let d = 7usize;
        let mut wv: Vec<f32> = Vec::with_capacity(n * d);
        let mut xv: Vec<f32> = Vec::with_capacity(n);
        // A deterministic, non-trivial pattern; no RNG needed.
        for i in 0..n * d {
            wv.push(((i * 37 % 101) as f32 - 50.0) * 0.01);
        }
        for i in 0..n {
            xv.push(((i * 53 % 97) as f32 - 48.0) * 0.02);
        }
        let mut a = vec![0.0f32; d];
        let mut b = vec![0.0f32; d];
        tensor::matmul_scalar(&mut a, &xv, &wv, n, d);
        unsafe { tensor::matmul_avx2(&mut b, &xv, &wv, n, d) };

        let mut worst = 0.0f32;
        for i in 0..d {
            let e = if a[i] > b[i] { a[i] - b[i] } else { b[i] - a[i] };
            if e > worst {
                worst = e;
            }
        }
        // FMA does not round the intermediate product, so results differ from
        // the scalar path in the last bits. That is expected and is not error.
        all &= check(
            "matmul avx2 == scalar",
            worst < 1e-3,
            "n=173 (ragged tail), d=7",
        );
    } else {
        console::set_color(YELLOW);
        kprintln!("  skip  matmul avx2             AVX not enabled on this CPU");
        console::set_color(WHITE);
    }

    all
}

/// Measure sustained matmul throughput at a size typical of a small model.
pub fn bench() {
    let n = 512usize;
    let d = 512usize;
    let mut wv: Vec<f32> = Vec::with_capacity(n * d);
    for i in 0..n * d {
        wv.push(((i % 251) as f32 - 125.0) * 0.001);
    }
    let xv: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
    let mut out = vec![0.0f32; d];

    let hz = crate::TIMER_HZ as u64;
    let target = hz / 2; // half a second
    let start = crate::dev::lapic::ticks();
    // Wait for a tick edge so the interval is not short by up to one period.
    while crate::dev::lapic::ticks() == start {
        core::hint::spin_loop();
    }
    let t0 = crate::dev::lapic::ticks();

    let mut iters = 0u64;
    while crate::dev::lapic::ticks() - t0 < target {
        tensor::matmul(&mut out, &xv, &wv, n, d);
        iters += 1;
    }
    let elapsed = crate::dev::lapic::ticks() - t0;

    // Two flops per element: one multiply, one add.
    let flops = 2.0 * (n * d) as f32 * iters as f32;
    let seconds = elapsed as f32 / hz as f32;
    let gflops = flops / seconds / 1.0e9;

    console::set_color(YELLOW);
    kprintln!("[bench]");
    console::set_color(WHITE);
    kprintln!("  {}x{} matmul, {} iterations in {} ticks", d, n, iters, elapsed);
    let path = if crate::cpu::detected().avx_enabled { "avx2+fma" } else { "scalar" };
    // Integer-formatted: printing floats needs a formatter we have not written.
    kprintln!(
        "  {}.{:02} GFLOP/s  ({})",
        gflops as u64,
        ((gflops * 100.0) as u64) % 100,
        path
    );
    // Guard against the compiler deciding the whole loop is dead.
    kprintln!("  checksum {}", (out[0] * 1000.0) as i64);
}
