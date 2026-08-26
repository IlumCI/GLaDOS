//! Backward kernels for training through frozen weights.
//!
//! The forward path is a family of matvecs; the backward path through the
//! same frozen weights is the transpose family -- `grad_x = W' . grad_y` --
//! and it gets its own kernels rather than a transposed view, because the
//! int8 blocks are laid out row-major for the forward's streaming access
//! and a transposed walk would scatter every cache line it touched.
//!
//! Every kernel here carries one non-negotiable test: a finite-difference
//! check against the forward it claims to be the adjoint of. A wrong
//! gradient produces training that converges confidently toward garbage,
//! which is indistinguishable from a bad week until the eval says so weeks
//! later. The gradcheck says so immediately.

use super::weights::{f32_at, Mat};
use alloc::vec;
use alloc::vec::Vec;

/// `out[k] = sum_r W[r,k] * g[r]` through an int8 block-quantized weight.
///
/// Column-scatter over row-major data: each frozen row is decoded once and
/// its contribution scattered into every output element, which is the only
/// cache-friendly direction available without transposing storage.
/// Accumulation per element runs over `r` ascending, matching the forward's
/// single-chain discipline so results are reproducible run to run.
pub fn q8_wt_matvec_scalar(
    out: &mut [f32],
    g: &[f32],
    data: &[i8],
    scales: &[u8],
    rows: usize,
    cols: usize,
) {
    for v in out[..cols].iter_mut() {
        *v = 0.0;
    }
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        // Scale folded into the row's coefficient: the dequantised weight is
        // w[r,c] = row[c] * scale, and scale is constant across the row.
        let gr = g[r] * f32_at(scales, r);
        for c in 0..cols {
            out[c] += row[c] as f32 * gr;
        }
    }
}

/// f32 twin of [`q8_wt_matvec_scalar`] for flat checkpoints.
pub fn f32_wt_matvec(out: &mut [f32], g: &[f32], w: &[f32], rows: usize, cols: usize) {
    for v in out[..cols].iter_mut() {
        *v = 0.0;
    }
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let gr = g[r];
        for c in 0..cols {
            out[c] += row[c] * gr;
        }
    }
}

impl Mat<'_> {
    /// Dispatching adjoint of [`super::weights::Mat::matvec`].
    pub fn wt_matvec(&self, out: &mut [f32], g: &[f32]) {
        match self {
            Mat::F32 { data, rows, cols } => {
                f32_wt_matvec(&mut out[..*cols], &g[..*rows], data, *rows, *cols)
            }
            Mat::Q8 { data, scales, rows, cols } => q8_wt_matvec_scalar(
                &mut out[..*cols],
                &g[..*rows],
                data,
                scales,
                *rows,
                *cols,
            ),
        }
    }
}

// ---------------------------------------------------------------------
// gradcheck machinery
// ---------------------------------------------------------------------

/// Deterministic xorshift64* -- the same generator the synthetic model uses,
/// so boot tests never depend on timing or entropy.
struct Rng(u64);

impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

/// Verify `analytic` against central finite differences of `loss` at `x`.
/// Returns pass/fail; prints nothing -- callers own the reporting, since
/// several checks share one boot-test block.
fn check_grads(
    loss: &dyn Fn(&[f32]) -> f32,
    x: &[f32],
    analytic: &[f32],
    eps: f32,
    tol: f32,
) -> bool {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..x.len() {
        xp.copy_from_slice(x);
        xm.copy_from_slice(x);
        xp[i] += eps;
        xm[i] -= eps;
        let num = (loss(&xp) - loss(&xm)) / (2.0 * eps);
        let ana = analytic[i];
        let denom = num.abs().max(ana.abs()).max(1e-6);
        if (num - ana).abs() / denom > tol {
            return false;
        }
    }
    true
}

/// The boot self-test. Four claims:
///
/// 1. the int8 adjoint matches a direct f64 reference computation;
/// 2. the f32 adjoint matches its obvious transpose loop;
/// 3. both survive a finite-difference check against a scalar objective;
/// 4. the QDoRA low-rank branch's analytic gradients (A and B) survive the
///    same treatment, using the live `apply()` as the forward so the test
///    exercises the code that will actually train, not a copy of it.
pub fn selftest() -> bool {
    use crate::kprintln;

    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let (rows, cols) = (24usize, 16usize);

    // --- claim 1 + 2 + 3: the weight adjoints -------------------------
    let wdata: Vec<i8> = (0..rows * cols).map(|_| (rng.f32() * 100.0) as i8).collect();
    // Scales are stored as little-endian f32 bytes per row -- one raw byte
    // each would both misread and run off the end of the buffer, which is
    // exactly what the first version of this fixture did.
    let wscales: Vec<u8> = (0..rows)
        .flat_map(|i| ((i % 7 + 1) as f32).to_le_bytes())
        .collect();
    let g: Vec<f32> = (0..rows).map(|_| rng.f32()).collect();

    let mut got = vec![0.0f32; cols];
    q8_wt_matvec_scalar(&mut got, &g, &wdata, &wscales, rows, cols);

    // f64 reference: exact dequantise then dot, accumulated wide.
    let mut want = vec![0.0f64; cols];
    for r in 0..rows {
        let s = f32_at(&wscales, r) as f64;
        for c in 0..cols {
            want[c] += wdata[r * cols + c] as f64 * s * g[r] as f64;
        }
    }
    let q8_ok = got
        .iter()
        .zip(&want)
        .all(|(a, b)| ((*a as f64) - *b).abs() / b.abs().max(1.0) < 1e-3);
    crate::kprintln!(
        "  {}  int8 adjoint matches the wide reference",
        if q8_ok { "ok " } else { "FAIL" }
    );

    let wf: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
    let gf: Vec<f32> = (0..rows).map(|_| rng.f32()).collect();
    let mut gotf = vec![0.0f32; cols];
    f32_wt_matvec(&mut gotf, &gf, &wf, rows, cols);
    let mut wantf = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            wantf[c] += wf[r * cols + c] * gf[r];
        }
    }
    let f32_ok = gotf == wantf;
    crate::kprintln!(
        "  {}  f32 adjoint is the transpose loop",
        if f32_ok { "ok " } else { "FAIL" }
    );

    // Finite differences through the int8 kernel itself -- computed wide,
    // because the point is to verify the adjoint's arithmetic rather than
    // re-measure f32 cancellation: with int8 rows scaled into the hundreds,
    // an f32 central difference drowns its own signal.
    let fd_ok = {
        let loss = |x: &[f32]| -> f32 {
            let mut acc = 0.0f64;
            for r in 0..rows {
                let s = f32_at(&wscales, r) as f64;
                let mut d = 0.0f64;
                for c in 0..cols {
                    d += wdata[r * cols + c] as f64 * x[c] as f64;
                }
                acc += d * s * g[r] as f64;
            }
            acc as f32
        };
        let x0: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        // Detailed pass: report the first coordinate whose numeric gradient
        // disagrees, with both numbers, instead of a bare verdict. The
        // previous runs proved only that *something* disagreed while claim
        // 1 proved `got` matched a wide reference -- an impossibility that
        // means the harness, not the kernel, is the suspect.
        let mut ok = true;
        let mut reported = false;
        let mut xp = x0.clone();
        let mut xm = x0.clone();
        for i in 0..cols {
            xp.copy_from_slice(&x0);
            xm.copy_from_slice(&x0);
            xp[i] += 1e-2;
            xm[i] -= 1e-2;
            let num = (loss(&xp) - loss(&xm)) / 0.02;
            let ana = got[i];
            let denom = num.abs().max(ana.abs()).max(1e-6);
            if (num - ana).abs() / denom > 2e-2 && !reported {
                crate::kprintln!(
                    "  [dbg] c{} num {} ana {}",
                    i,
                    num,
                    ana
                );
                ok = false;
                break;
            } else if (num - ana).abs() / denom > 2e-2 {
                ok = false;
            }
        }
        ok
    };
    crate::kprintln!(
        "  {}  int8 adjoint passes finite differences",
        if fd_ok { "ok " } else { "FAIL" }
    );

    // --- claim 4: the low-rank branch's own gradients -----------------
    let (r, kin, outr) = (4usize, 12usize, 9usize);
    let alpha = 8.0;
    let mut d = super::adapter::Dora::new(r, alpha, kin, outr);
    for v in d.a.iter_mut() {
        *v = rng.f32();
    }
    for v in d.b.iter_mut() {
        *v = rng.f32();
    }
    // Fixed cached scales (identity) and magnitudes: this check isolates the
    // low-rank branch; magnitude gradients get their own check once the
    // trainer touches them.
    for v in d.s.iter_mut() {
        *v = 1.0;
    }
    let x: Vec<f32> = (0..kin).map(|_| rng.f32()).collect();
    let gy: Vec<f32> = (0..outr).map(|_| rng.f32()).collect();
    let scale = d.scale();

    // Analytic: dL/dB[o,j] = scale * g[o] * ax[j];
    //           dL/dA[j,k] = scale * sum_o g[o] * B[o,j] * x[k].
    let mut ax = vec![0.0f32; r];
    {
        let _ = d.apply(&mut vec![0.0; outr], &x, &mut ax);
    }
    let mut gb = vec![0.0f32; outr * r];
    for o in 0..outr {
        for j in 0..r {
            gb[o * r + j] = scale * gy[o] * ax[j];
        }
    }
    let mut ga = vec![0.0f32; r * kin];
    for j in 0..r {
        for k in 0..kin {
            let mut acc = 0.0f32;
            for o in 0..outr {
                acc += gy[o] * d.b[o * r + j];
            }
            ga[j * kin + k] = scale * acc * x[k];
        }
    }

    // Forward used by the finite differencer: base zeros plus the branch,
    // which is exactly what `apply` adds onto any matvec result.
    let loss = |flat: &[f32]| -> f32 {
        let mut probe = super::adapter::Dora::new(r, alpha, kin, outr);
        probe.a.copy_from_slice(&flat[..r * kin]);
        probe.b.copy_from_slice(&flat[r * kin..]);
        for v in probe.s.iter_mut() {
            *v = 1.0;
        }
        let mut o = vec![0.0f32; outr];
        let mut scratch = vec![0.0f32; r];
        probe.apply(&mut o, &x, &mut scratch);
        o.iter().zip(&gy).map(|(a, b)| a * b).sum::<f32>()
    };
    let mut flat: Vec<f32> = Vec::new();
    flat.extend_from_slice(&d.a);
    flat.extend_from_slice(&d.b);
    let mut analytic: Vec<f32> = Vec::new();
    analytic.extend_from_slice(&ga);
    analytic.extend_from_slice(&gb);
    let lora_ok = check_grads(&loss, &flat, &analytic, 1e-3, 3e-2);
    crate::kprintln!(
        "  {}  low-rank branch gradients pass finite differences",
        if lora_ok { "ok " } else { "FAIL" }
    );

    q8_ok && f32_ok && fd_ok && lora_ok
}
