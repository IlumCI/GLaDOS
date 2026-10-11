//! The operators Kokoro is made of, over time-major activations.
//!
//! An activation is `(t, c)`: `t` frames of `c` channels, a frame
//! contiguous. PyTorch keeps `(c, t)`; this is the transpose, chosen
//! because every heavy operator here reduces to "one weight matrix against
//! many frames", and that is exactly `Mat::matvec_batch` -- the int8 and
//! f32 kernels the language model already runs in AVX2. A convolution is
//! an im2col over time and that matmul; a transposed convolution is one
//! matmul per tap and a scatter; an LSTM's input projection is that matmul
//! over the whole sequence before the recurrence runs. Nothing in this file
//! multiplies two matrices by itself.
//!
//! im2col is chunked over time (`CHUNK` frames), because the vocoder's
//! last stage runs at 4800 frames a second with 1408 columns and a whole
//! sentence's column matrix would be eighty megabytes of heap for a
//! temporary.
//!
//! Every operator has a claim against an answer worked by hand, because a
//! convolution with its taps mirrored or its padding one short produces
//! perfectly finite audio that is merely wrong.

use crate::ai::tensor;
use crate::ai::weights::Mat;
use alloc::vec;
use alloc::vec::Vec;

const CHUNK: usize = 128;

pub fn sigmoid(x: f32) -> f32 {
    tensor::sigmoid(x)
}
pub fn tanh(x: f32) -> f32 {
    2.0 * tensor::sigmoid(2.0 * x) - 1.0
}
/// ALBERT's `gelu_new`: the tanh approximation, which is what the
/// checkpoint was trained through.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + tanh(0.797_884_6 * (x + 0.044_715 * x * x * x)))
}
pub fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}
/// Snake, per channel: `x + sin^2(a x) / a`.
pub fn snake(x: &mut [f32], c: usize, alpha: &[f32]) {
    for frame in x.chunks_mut(c) {
        for (v, &a) in frame.iter_mut().zip(alpha) {
            let s = tensor::sinf(a * *v);
            *v += s * s / a;
        }
    }
}

/// `y = x W^T + b` over every frame.
pub fn linear(w: &Mat, b: Option<&[f32]>, x: &[f32], t: usize, cin: usize, cout: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * cin);
    let mut out = vec![0f32; t * cout];
    let mut done = 0;
    while done < t {
        let n = (t - done).min(CHUNK);
        w.matvec_batch(&mut out[done * cout..(done + n) * cout], &x[done * cin..(done + n) * cin], n);
        done += n;
    }
    if let Some(b) = b {
        for frame in out.chunks_mut(cout) {
            tensor::add_into(frame, b);
        }
    }
    out
}

pub struct Conv1d {
    pub w: Mat<'static>,
    pub b: Option<&'static [f32]>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
    pub dil: usize,
    pub stride: usize,
    pub pad: usize,
}

impl Conv1d {
    pub fn out_len(&self, t: usize) -> usize {
        (t + 2 * self.pad).saturating_sub(self.dil * (self.k - 1) + 1) / self.stride + 1
    }

    /// `(t, cin)` in, `(out_len, cout)` out. Zero padding, PyTorch's
    /// cross-correlation (tap `j` reads `t*stride - pad + j*dil`).
    pub fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), t * self.cin);
        let to = self.out_len(t);
        let cols = self.cin * self.k;
        let mut out = vec![0f32; to * self.cout];
        let mut col = vec![0f32; CHUNK * cols];
        let mut done = 0;
        while done < to {
            let n = (to - done).min(CHUNK);
            for i in 0..n {
                let row = &mut col[i * cols..(i + 1) * cols];
                let base = ((done + i) * self.stride) as isize - self.pad as isize;
                for j in 0..self.k {
                    let src = base + (j * self.dil) as isize;
                    if src >= 0 && (src as usize) < t {
                        let s = src as usize;
                        // column order is (channel, tap), matching the
                        // weight's (out, in, k) flattened to (out, in*k)
                        for c in 0..self.cin {
                            row[c * self.k + j] = x[s * self.cin + c];
                        }
                    } else {
                        for c in 0..self.cin {
                            row[c * self.k + j] = 0.0;
                        }
                    }
                }
            }
            self.w.matvec_batch(&mut out[done * self.cout..(done + n) * self.cout], &col[..n * cols], n);
            done += n;
        }
        if let Some(b) = self.b {
            for frame in out.chunks_mut(self.cout) {
                tensor::add_into(frame, b);
            }
        }
        out
    }
}

/// ConvTranspose1d, one `(cout, cin)` matrix per tap: output position
/// `i*stride - pad + j` collects tap `j` of input frame `i`.
pub struct ConvT1d {
    pub taps: Vec<Mat<'static>>,
    pub b: Option<&'static [f32]>,
    pub cin: usize,
    pub cout: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvT1d {
    pub fn out_len(&self, t: usize) -> usize {
        (t - 1) * self.stride + self.taps.len() - 2 * self.pad
    }

    pub fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        let to = self.out_len(t);
        let mut out = vec![0f32; to * self.cout];
        if let Some(b) = self.b {
            for frame in out.chunks_mut(self.cout) {
                frame.copy_from_slice(b);
            }
        }
        for (j, w) in self.taps.iter().enumerate() {
            let y = linear(w, None, x, t, self.cin, self.cout);
            for i in 0..t {
                let pos = (i * self.stride + j) as isize - self.pad as isize;
                if pos >= 0 && (pos as usize) < to {
                    let p = pos as usize;
                    tensor::add_into(&mut out[p * self.cout..(p + 1) * self.cout], &y[i * self.cout..(i + 1) * self.cout]);
                }
            }
        }
        out
    }
}

/// The depthwise transposed convolution `AdainResBlk1d` upsamples with:
/// kernel 3, stride 2, padding 1, output padding 1, one `(3,)` tap row per
/// channel. Doubles the length exactly.
pub fn pool_x2(w: &[f32], b: &[f32], x: &[f32], t: usize, c: usize) -> Vec<f32> {
    let to = 2 * t;
    let mut out = vec![0f32; to * c];
    for frame in out.chunks_mut(c) {
        frame.copy_from_slice(b);
    }
    for i in 0..t {
        for j in 0..3 {
            let pos = (2 * i + j) as isize - 1;
            if pos >= 0 && (pos as usize) < to {
                let p = pos as usize;
                for ch in 0..c {
                    out[p * c + ch] += w[ch * 3 + j] * x[i * c + ch];
                }
            }
        }
    }
    out
}

/// Nearest-neighbour upsample by two over time.
pub fn upsample_x2(x: &[f32], t: usize, c: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * t * c);
    for i in 0..t {
        out.extend_from_slice(&x[i * c..(i + 1) * c]);
        out.extend_from_slice(&x[i * c..(i + 1) * c]);
    }
    out
}

/// LayerNorm over the channels of each frame; `affine` is (gamma, beta).
pub fn layer_norm(x: &mut [f32], c: usize, affine: Option<(&[f32], &[f32])>, eps: f32) {
    for frame in x.chunks_mut(c) {
        let mean = frame.iter().sum::<f32>() / c as f32;
        let var = frame.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / tensor::sqrtf(var + eps);
        match affine {
            Some((g, b)) => {
                for (i, v) in frame.iter_mut().enumerate() {
                    *v = (*v - mean) * inv * g[i] + b[i];
                }
            }
            None => {
                for v in frame.iter_mut() {
                    *v = (*v - mean) * inv;
                }
            }
        }
    }
}

/// InstanceNorm1d without affine: each channel normalised over time.
pub fn instance_norm(x: &mut [f32], t: usize, c: usize, eps: f32) {
    for ch in 0..c {
        let mut sum = 0f32;
        for i in 0..t {
            sum += x[i * c + ch];
        }
        let mean = sum / t as f32;
        let mut var = 0f32;
        for i in 0..t {
            let d = x[i * c + ch] - mean;
            var += d * d;
        }
        let inv = 1.0 / tensor::sqrtf(var / t as f32 + eps);
        for i in 0..t {
            x[i * c + ch] = (x[i * c + ch] - mean) * inv;
        }
    }
}

/// Adaptive normalisation: `(1 + gamma) * norm(x) + beta`, gamma and beta
/// from one linear layer over the 128-dim style vector. Kokoro uses it
/// two ways and the only difference is the norm underneath.
pub struct Ada {
    pub fc: Mat<'static>,
    pub b: &'static [f32],
    pub c: usize,
}

#[derive(Clone, Copy)]
pub enum Norm {
    Instance,
    Layer,
}

impl Ada {
    pub fn apply(&self, x: &mut [f32], t: usize, style: &[f32], norm: Norm) {
        let h = linear(&self.fc, Some(self.b), style, 1, style.len(), 2 * self.c);
        let (gamma, beta) = h.split_at(self.c);
        match norm {
            Norm::Instance => instance_norm(x, t, self.c, 1e-5),
            Norm::Layer => layer_norm(x, self.c, None, 1e-5),
        }
        for frame in x.chunks_mut(self.c) {
            for i in 0..self.c {
                frame[i] = (1.0 + gamma[i]) * frame[i] + beta[i];
            }
        }
    }
}

/// One direction of an LSTM, PyTorch's gate order (i, f, g, o), the two
/// biases already summed by the converter.
pub struct Lstm {
    pub wih: Mat<'static>,
    pub whh: Mat<'static>,
    pub b: &'static [f32],
    pub cin: usize,
    pub h: usize,
}

impl Lstm {
    fn run(&self, x: &[f32], t: usize, reverse: bool, out: &mut [f32], stride: usize, off: usize) {
        let h = self.h;
        let pre = linear(&self.wih, Some(self.b), x, t, self.cin, 4 * h);
        let mut hs = vec![0f32; h];
        let mut cs = vec![0f32; h];
        let mut g = vec![0f32; 4 * h];
        for step in 0..t {
            let i = if reverse { t - 1 - step } else { step };
            self.whh.matvec(&mut g, &hs);
            tensor::add_into(&mut g, &pre[i * 4 * h..(i + 1) * 4 * h]);
            for j in 0..h {
                let ig = sigmoid(g[j]);
                let fg = sigmoid(g[h + j]);
                let gg = tanh(g[2 * h + j]);
                let og = sigmoid(g[3 * h + j]);
                cs[j] = fg * cs[j] + ig * gg;
                hs[j] = og * tanh(cs[j]);
            }
            out[i * stride + off..i * stride + off + h].copy_from_slice(&hs);
        }
    }
}

/// Bidirectional: forward in the first `h` of each frame, backward in the
/// second, which is how PyTorch lays a bidirectional output out.
pub fn lstm_bi(f: &Lstm, r: &Lstm, x: &[f32], t: usize) -> Vec<f32> {
    let h = f.h;
    let mut out = vec![0f32; t * 2 * h];
    f.run(x, t, false, &mut out, 2 * h, 0);
    r.run(x, t, true, &mut out, 2 * h, h);
    out
}

pub fn softmax(x: &mut [f32]) {
    tensor::softmax(x)
}

pub fn checks() -> Vec<(&'static str, bool)> {
    fn leak(v: Vec<f32>) -> &'static [f32] {
        Vec::leak(v)
    }
    fn near(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }
    // conv: cin 2, cout 1, k 3, pad 1, dil 1 over 4 frames
    // w[out0] = [[1,0,-1] for ch0, [0,2,0] for ch1]
    let w = leak(vec![1.0, 0.0, -1.0, 0.0, 2.0, 0.0]);
    let conv = Conv1d { w: Mat::F32 { data: w, rows: 1, cols: 6 }, b: Some(leak(vec![0.5])), cin: 2, cout: 1, k: 3, dil: 1, stride: 1, pad: 1 };
    // x frames (ch0, ch1): (1,10) (2,20) (3,30) (4,40)
    let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
    // out[t] = x0[t-1] - x0[t+1] + 2*x1[t] + 0.5
    let want = [0.0 - 2.0 + 20.0 + 0.5, 1.0 - 3.0 + 40.0 + 0.5, 2.0 - 4.0 + 60.0 + 0.5, 3.0 - 0.0 + 80.0 + 0.5];
    let got = conv.apply(&x, 4);
    // dilation 2, same taps: out[t] = x0[t-2] - x0[t+2] + 2 x1[t] + 0.5, pad 2
    let conv2 = Conv1d { w: Mat::F32 { data: w, rows: 1, cols: 6 }, b: Some(leak(vec![0.5])), cin: 2, cout: 1, k: 3, dil: 2, stride: 1, pad: 2 };
    let got2 = conv2.apply(&x, 4);
    let want2 = [0.0 - 3.0 + 20.0 + 0.5, 0.0 - 4.0 + 40.0 + 0.5, 1.0 - 0.0 + 60.0 + 0.5, 2.0 - 0.0 + 80.0 + 0.5];
    // stride 2 k3 pad 1: out[o] = x0[2o-1] - x0[2o+1] + 2 x1[2o] + .5 -> o=0: 0-2+20+.5, o=1: 2-4+60+.5
    let conv3 = Conv1d { w: Mat::F32 { data: w, rows: 1, cols: 6 }, b: Some(leak(vec![0.5])), cin: 2, cout: 1, k: 3, dil: 1, stride: 2, pad: 1 };
    let got3 = conv3.apply(&x, 4);
    // transposed: cin 1, cout 1, k 4, stride 2, pad 1; taps [1,2,3,4]; x = [1, 10]
    // out len = (2-1)*2 + 4 - 2 = 4; pos = i*2 - 1 + j
    // i=0: pos -1,0,1,2 <- 1,2,3,4 ; i=1: pos 1,2,3,4 <- 10,20,30,40
    // out = [2, 3+10, 4+20, 30]
    let taps: Vec<Mat<'static>> = [1.0f32, 2.0, 3.0, 4.0].iter().map(|&v| Mat::F32 { data: leak(vec![v]), rows: 1, cols: 1 }).collect();
    let ct = ConvT1d { taps, b: None, cin: 1, cout: 1, stride: 2, pad: 1 };
    let gott = ct.apply(&[1.0, 10.0], 2);
    // depthwise pool: one channel, taps [1,2,3], x=[1,10] -> len 4, pos=2i-1+j
    // i=0: pos -1,0,1 <- 1,2,3; i=1: pos 1,2,3 <- 10,20,30 -> [2, 3+10, 20, 30]
    let gotp = pool_x2(&[1.0, 2.0, 3.0], &[0.0], &[1.0, 10.0], 2, 1);
    // lstm, h=1, cin=1: all weights 0 except bias -> gates = b
    // b = [i, f, g, o] = [0, 0, large, 0]: c = 0.5*1 = .5 ; h = 0.5*tanh(.5)
    let wz = Mat::F32 { data: leak(vec![0.0, 0.0, 0.0, 0.0]), rows: 4, cols: 1 };
    let l = Lstm { wih: wz.clone(), whh: wz.clone(), b: leak(vec![0.0, 0.0, 30.0, 0.0]), cin: 1, h: 1 };
    let got_l = lstm_bi(&l, &l, &[0.0, 0.0], 2);
    let h1 = 0.5 * tanh(0.5);
    let h2 = 0.5 * tanh(0.75);                     // c2 = 0.5 c1 + 0.5 = 0.75
    // forward: frame0 h1, frame1 h2; backward: frame1 h1, frame0 h2
    let want_l = [h1, h2, h2, h1];
    let mut ln = [1.0f32, 2.0, 3.0, 4.0];
    layer_norm(&mut ln, 2, None, 0.0);
    let mut inn = [1.0f32, 0.0, 3.0, 0.0];
    instance_norm(&mut inn, 2, 2, 1e-5);           // a flat channel divides by eps, not by zero
    let mut sn = [0.0f32, 1.0];
    snake(&mut sn, 1, &[1.0]);
    vec![
        ("conv1d k3 pad1 matches the hand-worked answer", near(&got, &want)),
        ("conv1d with dilation 2 reads taps two frames apart", near(&got2, &want2)),
        ("conv1d with stride 2 halves the length and reads even frames", near(&got3, &[18.5, 58.5])),
        ("conv_transpose1d scatters taps by stride and clips the padding", near(&gott, &[2.0, 13.0, 24.0, 30.0])),
        ("the depthwise x2 pool doubles the length with the right taps", near(&gotp, &[2.0, 13.0, 20.0, 30.0])),
        ("the LSTM cell state carries and the backward half runs backward", near(&got_l, &want_l)),
        ("layer_norm normalises each frame over its channels", near(&ln, &[-1.0, 1.0, -1.0, 1.0])),
        ("instance_norm normalises each channel over time", near(&inn, &[-1.0, 0.0, 1.0, 0.0])),
        ("snake at alpha 1: 0 stays 0, 1 becomes 1 + sin^2(1)", near(&sn, &[0.0, 1.0 + 0.708_073_4])),
        ("gelu(0) is 0 and gelu(5) is 5", gelu(0.0) == 0.0 && (gelu(5.0) - 5.0).abs() < 1e-3),
    ]
}
