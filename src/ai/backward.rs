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

use super::tensor;
use super::weights::{f32_at, Mat};
use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------
// activation adjoints
// ---------------------------------------------------------------------

/// Adjoint of [`super::tensor::rmsnorm_eps`]: `y = w . x . s`, where
/// `s = (mean(x^2) + eps)^(-1/2)`.
///
/// `dL/dx_j = s.(g_j.w_j) - (s^3/n).x_j.(sum_i g_i.w_i.x_i)` -- the direct
/// term through each element's own scale, minus the coupling term through
/// every element's contribution to the norm.
pub fn rmsnorm_backward(gx: &mut [f32], gy: &[f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x.iter().take(n) {
        ss += v * v;
    }
    let s = 1.0 / tensor::sqrtf(ss / n as f32 + eps);
    let mut dot = 0.0f32;
    for i in 0..n {
        dot += gy[i] * w[i] * x[i];
    }
    for j in 0..n {
        gx[j] = s * gy[j] * w[j] - (s * s * s / n as f32) * x[j] * dot;
    }
}

/// Adjoints of [`super::tensor::swiglu`], which computes `h = silu(u) . v`.
/// `gu` receives `dL/du`, `gv` receives `dL/dv`.
pub fn swiglu_backward(gu: &mut [f32], gv: &mut [f32], gh: &[f32], u: &[f32], v: &[f32]) {
    for i in 0..gh.len() {
        let sig = tensor::sigmoid(u[i]);
        // d silu / du = sig + u.sig.(1-sig).
        let dsilu = sig + u[i] * sig * (1.0 - sig);
        gu[i] = gh[i] * v[i] * dsilu;
        gv[i] = gh[i] * tensor::silu(u[i]);
    }
}

/// Backward through one head of scaled causal attention,
/// `o_i = sum_{j<=i} p_ij . v_j`, `p = softmax(q.k' . scale)`.
///
/// Single-head over plain contiguous arrays: the live path's cache
/// indirection (`krot`, slot mapping, GQA fan-out) composes around this
/// core, and the core is what has to be provably correct first.
/// `dq`, `dk`, `dv` are accumulated into, so callers can chain heads
/// without a temporary.
pub fn attention_backward(
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    dy: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    scale: f32,
) {
    let mut p = vec![0.0f32; n];
    for i in 0..n {
        // Scores and softmax, recomputed exactly as the forward did.
        for j in 0..=i {
            let mut s = 0.0f32;
            for t in 0..d {
                s += q[i * d + t] * k[j * d + t];
            }
            p[j] = s * scale;
        }
        tensor::softmax(&mut p[..=i]);

        // Softmax Jacobian needs the whole prefix's dot before any ds is
        // final: c = sum_j p_j . dp_j.
        let mut c = 0.0f32;
        for j in 0..=i {
            let dp = dy[i * d..(i + 1) * d]
                .iter()
                .zip(v[j * d..(j + 1) * d].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            c += p[j] * dp;
        }
        for j in 0..=i {
            let dp = dy[i * d..(i + 1) * d]
                .iter()
                .zip(v[j * d..(j + 1) * d].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            let ds = p[j] * (dp - c);
            for t in 0..d {
                dq[i * d + t] += scale * ds * k[j * d + t];
                dk[j * d + t] += scale * ds * q[i * d + t];
            }
            dv[j * d..(j + 1) * d]
                .iter_mut()
                .zip(dy[i * d..(i + 1) * d].iter())
                .for_each(|(dvj, dyi)| *dvj += p[j] * dyi);
        }
    }
}

/// Transpose of one RoPE 2D rotation. Rotations are orthogonal, so the
/// adjoint is rotation by the opposite angle: same pairing, sine sign
/// flipped.
pub fn rope_pair_backward(a: f32, b: f32, c: f32, s: f32) -> (f32, f32) {
    (a * c + b * s, -a * s + b * c)
}

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

    // --- rmsnorm adjoint ----------------------------------------------
    let (n2, eps) = (20usize, 1e-5f32);
    let xr: Vec<f32> = (0..n2).map(|_| rng.f32()).collect();
    let wr: Vec<f32> = (0..n2).map(|_| rng.f32()).collect();
    let gyr: Vec<f32> = (0..n2).map(|_| rng.f32()).collect();
    let mut gxm = vec![0.0f32; n2];
    rmsnorm_backward(&mut gxm, &gyr, &xr, &wr, eps);
    let loss_rn = |x: &[f32]| -> f32 {
        let mut o = vec![0.0f32; n2];
        tensor::rmsnorm_eps(&mut o, x, &wr, eps);
        o.iter().zip(&gyr).map(|(a, b)| a * b).sum::<f32>()
    };
    let rn_ok = check_grads(&loss_rn, &xr, &gxm, 1e-3, 3e-2);
    crate::kprintln!(
        "  {}  rmsnorm adjoint passes finite differences",
        if rn_ok { "ok " } else { "FAIL" }
    );

    // --- swiglu adjoint -------------------------------------------------
    let nh = 14usize;
    let u: Vec<f32> = (0..nh).map(|_| rng.f32() * 2.0).collect();
    let v: Vec<f32> = (0..nh).map(|_| rng.f32()).collect();
    let gh: Vec<f32> = (0..nh).map(|_| rng.f32()).collect();
    let mut gu = vec![0.0f32; nh];
    let mut gv = vec![0.0f32; nh];
    swiglu_backward(&mut gu, &mut gv, &gh, &u, &v);
    let flat_uv: Vec<f32> = u.iter().chain(v.iter()).copied().collect();
    let flat_g: Vec<f32> = gu.iter().chain(gv.iter()).copied().collect();
    let loss_sw = |flat: &[f32]| -> f32 {
        let uu = &flat[..nh];
        let vv = &flat[nh..];
        uu.iter()
            .zip(vv)
            .zip(&gh)
            .map(|((uu, vv), gh)| tensor::silu(*uu) * vv * gh)
            .sum::<f32>()
    };
    let sw_ok = check_grads(&loss_sw, &flat_uv, &flat_g, 1e-3, 3e-2);
    crate::kprintln!(
        "  {}  swiglu adjoint passes finite differences",
        if sw_ok { "ok " } else { "FAIL" }
    );

    // --- causal attention adjoint ---------------------------------------
    let (an, ad) = (6usize, 4usize);
    let asc = 1.0 / tensor::sqrtf(ad as f32);
    let q: Vec<f32> = (0..an * ad).map(|_| rng.f32()).collect();
    let k: Vec<f32> = (0..an * ad).map(|_| rng.f32()).collect();
    let v: Vec<f32> = (0..an * ad).map(|_| rng.f32()).collect();
    let dy: Vec<f32> = (0..an * ad).map(|_| rng.f32()).collect();
    let mut dqa = vec![0.0f32; an * ad];
    let mut dka = vec![0.0f32; an * ad];
    let mut dva = vec![0.0f32; an * ad];
    attention_backward(&mut dqa, &mut dka, &mut dva, &dy, &q, &k, &v, an, ad, asc);
    let flat_qkv: Vec<f32> = q.iter().chain(&k).chain(&v).copied().collect();
    let flat_gqkv: Vec<f32> =
        dqa.iter().chain(&dka).chain(&dva).copied().collect();

    fn attend_fwd(
        flat: &[f32],
        dy: &[f32],
        n: usize,
        d: usize,
        scale: f32,
    ) -> f32 {
        let q = &flat[..n * d];
        let k = &flat[n * d..2 * n * d];
        let v = &flat[2 * n * d..];
        let mut total = 0.0f32;
        let mut p = vec![0.0f32; n];
        let mut o = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0f32;
                for t in 0..d {
                    s += q[i * d + t] * k[j * d + t];
                }
                p[j] = s * scale;
            }
            tensor::softmax(&mut p[..=i]);
            for t in 0..d {
                o[t] = 0.0;
                for j in 0..=i {
                    o[t] += p[j] * v[j * d + t];
                }
                total += o[t] * dy[i * d + t];
            }
        }
        total
    }
    let att_ok = check_grads(
        &|flat: &[f32]| attend_fwd(flat, &dy, an, ad, asc),
        &flat_qkv,
        &flat_gqkv,
        1e-3,
        3e-2,
    );
    crate::kprintln!(
        "  {}  causal attention adjoint passes finite differences",
        if att_ok { "ok " } else { "FAIL" }
    );

    // --- RoPE transpose --------------------------------------------------
    // The claim has two halves: algebraic (applying the backward after the
    // forward returns the original pair exactly) and numeric (finite
    // differences through the rotation agree with the transposed grads).
    let half = 4usize;
    let z: Vec<f32> = (0..2 * half).map(|_| rng.f32()).collect();
    let cs: Vec<f32> = (0..half).map(|i| tensor::cosf(i as f32 * 0.7)).collect();
    let sn: Vec<f32> = (0..half).map(|i| tensor::sinf(i as f32 * 0.7)).collect();
    let rot = |zz: &[f32]| -> Vec<f32> {
        let mut out = zz.to_vec();
        for p in 0..half {
            let (a, b) = (zz[p], zz[p + half]);
            out[p] = a * cs[p] - b * sn[p];
            out[p + half] = a * sn[p] + b * cs[p];
        }
        out
    };
    let rr = rot(&z);
    let rz: Vec<f32> = (0..2 * half).map(|i| rng.f32() + 0.5).collect();
    let mut ana_z = vec![0.0f32; 2 * half];
    for p in 0..half {
        let (ga, gb) =
            rope_pair_backward(rz[p], rz[p + half], cs[p], sn[p]);
        ana_z[p] = ga;
        ana_z[p + half] = gb;
    }
    let loss_ro = |zz: &[f32]| -> f32 {
        rot(zz)
            .iter()
            .zip(&rz)
            .map(|(a, b)| a * b)
            .sum::<f32>()
    };
    // Defining adjoint property rather than a rotation-twice claim (which
    // would be rotation by twice the angle -- the first version of this
    // check asserted exactly that falsehood).
    let identity_ok = {
        let lhs: f32 = rot(&z)
            .iter()
            .zip(&rz)
            .map(|(a, b)| a * b)
            .sum();
        let rhs: f32 = z
            .iter()
            .zip(&ana_z)
            .map(|(a, b)| a * b)
            .sum();
        (lhs - rhs).abs() < 1e-4
    };
    let rope_ok = identity_ok && check_grads(&loss_ro, &z, &ana_z, 1e-3, 3e-2);
    crate::kprintln!(
        "  {}  rope transpose is its adjoint and passes differences",
        if rope_ok { "ok " } else { "FAIL" }
    );

    // --- full QDoRA site gradient: norm terms included ------------------
    // The forward under test is the composition training actually uses:
    // set parameters, refresh cached scales against the frozen weight,
    // apply onto the fixed base matvec. Analytic gradients come from
    // Dora::backward, which carries the route through s that LoRA-only
    // checks never see.
    {
        let (r3, k3, o3) = (3usize, 8usize, 6usize);
        let wf: Vec<f32> = (0..o3 * k3).map(|_| rng.f32()).collect();
        let mat = Mat::F32 { data: &wf, rows: o3, cols: k3 };
        let mut dd = super::adapter::Dora::new(r3, 8.0, k3, o3);
        for v in dd.a.iter_mut() {
            *v = rng.f32();
        }
        for v in dd.b.iter_mut() {
            *v = rng.f32();
        }
        dd.refresh(&mat, true);
        let x3: Vec<f32> = (0..k3).map(|_| rng.f32()).collect();
        let gy3: Vec<f32> = (0..o3).map(|_| rng.f32()).collect();
        let mut base = vec![0.0f32; o3];
        mat.matvec(&mut base, &x3);
        let mut ax0 = vec![0.0f32; r3];
        {
            let mut outp = base.clone();
            dd.apply(&mut outp, &x3, &mut ax0);
        }

        let mut ga = vec![0.0f32; r3 * k3];
        let mut gb = vec![0.0f32; o3 * r3];
        let mut dm = vec![0.0f32; o3];
        dd.backward(&mat, &x3, &ax0, &base, &gy3, &mut ga, &mut gb, &mut dm);

        let flat0: Vec<f32> = dd
            .a
            .iter()
            .chain(&dd.b)
            .chain(&dd.m)
            .copied()
            .collect();

        let loss_d = |flat: &[f32]| -> f32 {
            let mut probe = super::adapter::Dora::new(r3, 8.0, k3, o3);
            probe.a.copy_from_slice(&flat[..r3 * k3]);
            probe.b.copy_from_slice(&flat[r3 * k3..r3 * k3 + o3 * r3]);
            probe.m.copy_from_slice(&flat[r3 * k3 + o3 * r3..]);
            probe.refresh(&mat, false);
            let mut out = base.clone();
            let mut scratch = vec![0.0f32; r3];
            probe.apply(&mut out, &x3, &mut scratch);
            out.iter().zip(&gy3).map(|(a, b)| a * b).sum::<f32>()
        };
        let analytic: Vec<f32> = ga.iter().chain(&gb).chain(&dm).copied().collect();
        let dora_ok = check_grads(&loss_d, &flat0, &analytic, 1e-3, 4e-2);
        crate::kprintln!(
            "  {}  full qdora site gradients pass finite differences",
            if dora_ok { "ok " } else { "FAIL" }
        );
        q8_ok
            && f32_ok
            && fd_ok
            && lora_ok
            && rn_ok
            && sw_ok
            && att_ok
            && rope_ok
            && dora_ok
    }
}
