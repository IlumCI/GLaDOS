//! Kokoro-82M, phonemes to a waveform, in the kernel.
//!
//! The graph is StyleTTS2's: a small ALBERT reads the phoneme string, a
//! duration predictor says how many 25 ms frames each phoneme takes, two
//! LSTM-and-convolution stacks predict pitch and noisiness per 12.5 ms,
//! and an iSTFTNet vocoder with a harmonic-plus-noise source turns the
//! expanded text features into 24 kHz audio. `tools/kokoro.py` lays the
//! weights out and this file walks them by name; a tensor with the wrong
//! shape is refused at load with its name.
//!
//! It is split where the house style needs to reach in. `front` runs
//! everything up to the model's own durations and `synth` runs the rest
//! from durations the caller chose -- the voice replaces the model's timing
//! with the rules in `style.rs`, so the alignment is a parameter and not a
//! step. The pitch and noise curves come back beside the audio because the
//! post stage reads them for voicing.
//!
//! Randomness: the source module seeds every harmonic above the first with
//! a random phase and adds noise, as the reference does. The kernel's draws
//! are its own, so the waveform is not bit-comparable to the host's; every
//! stage before it is, and `verify` compares exactly up to the audio and on
//! the spectral envelope after.

use super::ops::{self, Ada, Conv1d, ConvT1d, Lstm, Norm};
use super::vfile::VoiceFile;
use crate::ai::tensor;
use crate::ai::weights::Mat;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

pub const SR: usize = 24_000;
/// Samples per duration frame.
pub const FRAME: usize = 600;
const STYLE: usize = 128;
const HID: usize = 512;
const BERT: usize = 768;
const HEADS: usize = 12;
const NFFT: usize = 20;
const HOP: usize = 5;

type R<T> = Result<T, String>;

fn e(name: &str, why: &str) -> String {
    alloc::format!("{}: {}", name, why)
}

struct Lin {
    w: Mat<'static>,
    b: Option<&'static [f32]>,
    cin: usize,
    cout: usize,
}
impl Lin {
    fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        ops::linear(&self.w, self.b, x, t, self.cin, self.cout)
    }
}

fn lin(f: &VoiceFile, name: &str, cin: usize, cout: usize, bias: bool) -> R<Lin> {
    let w = f.mat(&alloc::format!("{name}.weight"), cout, cin).map_err(|w| e(name, w))?;
    let b = if bias { Some(f.f32_shaped(&alloc::format!("{name}.bias"), &[cout]).map_err(|w| e(name, w))?) } else { None };
    Ok(Lin { w, b, cin, cout })
}

fn conv(f: &VoiceFile, name: &str, cin: usize, cout: usize, k: usize, dil: usize, stride: usize, pad: usize, bias: bool) -> R<Conv1d> {
    let w = f.mat(&alloc::format!("{name}.weight"), cout, cin * k).map_err(|w| e(name, w))?;
    let b = if bias { Some(f.f32_shaped(&alloc::format!("{name}.bias"), &[cout]).map_err(|w| e(name, w))?) } else { None };
    Ok(Conv1d { w, b, cin, cout, k, dil, stride, pad })
}

fn convt(f: &VoiceFile, name: &str, cin: usize, cout: usize, k: usize, stride: usize, pad: usize) -> R<ConvT1d> {
    let w = f.f32_shaped(&alloc::format!("{name}.weight"), &[k, cout, cin]).map_err(|w| e(name, w))?;
    let b = f.f32_shaped(&alloc::format!("{name}.bias"), &[cout]).map_err(|w| e(name, w))?;
    let taps = (0..k).map(|j| Mat::F32 { data: &w[j * cout * cin..(j + 1) * cout * cin], rows: cout, cols: cin }).collect();
    Ok(ConvT1d { taps, b: Some(b), cin, cout, stride, pad })
}

fn ada(f: &VoiceFile, name: &str, c: usize) -> R<Ada> {
    let fc = f.mat(&alloc::format!("{name}.fc.weight"), 2 * c, STYLE).map_err(|w| e(name, w))?;
    let b = f.f32_shaped(&alloc::format!("{name}.fc.bias"), &[2 * c]).map_err(|w| e(name, w))?;
    Ok(Ada { fc, b, c })
}

fn lstm_pair(f: &VoiceFile, name: &str, cin: usize, h: usize) -> R<(Lstm, Lstm)> {
    let one = |suffix: &str| -> R<Lstm> {
        let wih = f.mat(&alloc::format!("{name}.weight_ih_l0{suffix}"), 4 * h, cin).map_err(|w| e(name, w))?;
        let whh = f.mat(&alloc::format!("{name}.weight_hh_l0{suffix}"), 4 * h, h).map_err(|w| e(name, w))?;
        let b = f.f32_shaped(&alloc::format!("{name}.bias_l0{suffix}"), &[4 * h]).map_err(|w| e(name, w))?;
        Ok(Lstm { wih, whh, b, cin, h })
    };
    Ok((one("")?, one("_reverse")?))
}

/// `AdainResBlk1d`: the prosody and decoder residual block.
struct Rb {
    norm1: Ada,
    norm2: Ada,
    conv1: Conv1d,
    conv2: Conv1d,
    sc: Option<Conv1d>,
    pool: Option<(&'static [f32], &'static [f32])>,
    cin: usize,
    cout: usize,
}

impl Rb {
    fn load(f: &VoiceFile, name: &str, cin: usize, cout: usize, up: bool) -> R<Rb> {
        Ok(Rb {
            norm1: ada(f, &alloc::format!("{name}.norm1"), cin)?,
            norm2: ada(f, &alloc::format!("{name}.norm2"), cout)?,
            conv1: conv(f, &alloc::format!("{name}.conv1"), cin, cout, 3, 1, 1, 1, true)?,
            conv2: conv(f, &alloc::format!("{name}.conv2"), cout, cout, 3, 1, 1, 1, true)?,
            sc: if cin != cout { Some(conv(f, &alloc::format!("{name}.conv1x1"), cin, cout, 1, 1, 1, 0, false)?) } else { None },
            pool: if up {
                Some((
                    f.f32_shaped(&alloc::format!("{name}.pool.weight"), &[cin, 3]).map_err(|w| e(name, w))?,
                    f.f32_shaped(&alloc::format!("{name}.pool.bias"), &[cin]).map_err(|w| e(name, w))?,
                ))
            } else {
                None
            },
            cin,
            cout,
        })
    }

    /// `(t, cin)` -> `(t or 2t, cout)`.
    fn apply(&self, x: &[f32], t: usize, s: &[f32]) -> (Vec<f32>, usize) {
        let mut r = x.to_vec();
        self.norm1.apply(&mut r, t, s, Norm::Instance);
        ops::leaky_relu(&mut r, 0.2);
        let (mut r, t2) = match self.pool {
            Some((w, b)) => (ops::pool_x2(w, b, &r, t, self.cin), 2 * t),
            None => (r, t),
        };
        r = self.conv1.apply(&r, t2);
        self.norm2.apply(&mut r, t2, s, Norm::Instance);
        ops::leaky_relu(&mut r, 0.2);
        r = self.conv2.apply(&r, t2);
        let sc = if self.pool.is_some() { ops::upsample_x2(x, t, self.cin) } else { x.to_vec() };
        let sc = match &self.sc {
            Some(c) => c.apply(&sc, t2),
            None => sc,
        };
        for (a, b) in r.iter_mut().zip(&sc) {
            *a = (*a + b) * core::f32::consts::FRAC_1_SQRT_2;
        }
        (r, t2)
    }
}

/// `AdaINResBlock1`: the vocoder's residual block, Snake activations and
/// three dilations.
struct RbG {
    c1: [Conv1d; 3],
    c2: [Conv1d; 3],
    n1: [Ada; 3],
    n2: [Ada; 3],
    a1: [&'static [f32]; 3],
    a2: [&'static [f32]; 3],
    c: usize,
}

impl RbG {
    fn load(f: &VoiceFile, name: &str, c: usize, k: usize) -> R<RbG> {
        let dil = [1usize, 3, 5];
        let pad = |d: usize| (k * d - d) / 2;
        let mk = |i: usize| -> R<(Conv1d, Conv1d, Ada, Ada, &'static [f32], &'static [f32])> {
            Ok((
                conv(f, &alloc::format!("{name}.convs1.{i}"), c, c, k, dil[i], 1, pad(dil[i]), true)?,
                conv(f, &alloc::format!("{name}.convs2.{i}"), c, c, k, 1, 1, pad(1), true)?,
                ada(f, &alloc::format!("{name}.adain1.{i}"), c)?,
                ada(f, &alloc::format!("{name}.adain2.{i}"), c)?,
                f.f32_shaped(&alloc::format!("{name}.alpha1.{i}"), &[1, c, 1]).map_err(|w| e(name, w))?,
                f.f32_shaped(&alloc::format!("{name}.alpha2.{i}"), &[1, c, 1]).map_err(|w| e(name, w))?,
            ))
        };
        let (a, b, cc) = (mk(0)?, mk(1)?, mk(2)?);
        Ok(RbG {
            c1: [a.0, b.0, cc.0],
            c2: [a.1, b.1, cc.1],
            n1: [a.2, b.2, cc.2],
            n2: [a.3, b.3, cc.3],
            a1: [a.4, b.4, cc.4],
            a2: [a.5, b.5, cc.5],
            c,
        })
    }

    fn apply(&self, x: &mut Vec<f32>, t: usize, s: &[f32]) {
        for i in 0..3 {
            let mut xt = x.clone();
            self.n1[i].apply(&mut xt, t, s, Norm::Instance);
            ops::snake(&mut xt, self.c, self.a1[i]);
            xt = self.c1[i].apply(&xt, t);
            self.n2[i].apply(&mut xt, t, s, Norm::Instance);
            ops::snake(&mut xt, self.c, self.a2[i]);
            xt = self.c2[i].apply(&xt, t);
            tensor::add_into(x, &xt);
        }
    }
}

struct Albert {
    word: &'static [f32],
    pos: &'static [f32],
    ttype: &'static [f32],
    ln_g: &'static [f32],
    ln_b: &'static [f32],
    map_in: Lin,
    q: Lin,
    k: Lin,
    v: Lin,
    dense: Lin,
    attn_ln: (&'static [f32], &'static [f32]),
    ffn: Lin,
    ffn_out: Lin,
    full_ln: (&'static [f32], &'static [f32]),
    layers: usize,
}

impl Albert {
    fn load(f: &VoiceFile) -> R<Albert> {
        let l = "bert.encoder.albert_layer_groups.0.albert_layers.0";
        let g = |n: &str, d: &[usize]| f.f32_shaped(n, d).map_err(|w| e(n, w));
        Ok(Albert {
            word: g("bert.embeddings.word_embeddings.weight", &[178, 128])?,
            pos: g("bert.embeddings.position_embeddings.weight", &[512, 128])?,
            ttype: g("bert.embeddings.token_type_embeddings.weight", &[2, 128])?,
            ln_g: g("bert.embeddings.LayerNorm.weight", &[128])?,
            ln_b: g("bert.embeddings.LayerNorm.bias", &[128])?,
            map_in: lin(f, "bert.encoder.embedding_hidden_mapping_in", 128, BERT, true)?,
            q: lin(f, &alloc::format!("{l}.attention.query"), BERT, BERT, true)?,
            k: lin(f, &alloc::format!("{l}.attention.key"), BERT, BERT, true)?,
            v: lin(f, &alloc::format!("{l}.attention.value"), BERT, BERT, true)?,
            dense: lin(f, &alloc::format!("{l}.attention.dense"), BERT, BERT, true)?,
            attn_ln: (g(&alloc::format!("{l}.attention.LayerNorm.weight"), &[BERT])?, g(&alloc::format!("{l}.attention.LayerNorm.bias"), &[BERT])?),
            ffn: lin(f, &alloc::format!("{l}.ffn"), BERT, 2048, true)?,
            ffn_out: lin(f, &alloc::format!("{l}.ffn_output"), 2048, BERT, true)?,
            full_ln: (g(&alloc::format!("{l}.full_layer_layer_norm.weight"), &[BERT])?, g(&alloc::format!("{l}.full_layer_layer_norm.bias"), &[BERT])?),
            layers: 12,
        })
    }

    /// `(L, 768)` for the ids.
    fn apply(&self, ids: &[u8]) -> Vec<f32> {
        let l = ids.len();
        let mut x = vec![0f32; l * 128];
        for (i, &id) in ids.iter().enumerate() {
            let fr = &mut x[i * 128..(i + 1) * 128];
            for j in 0..128 {
                fr[j] = self.word[id as usize * 128 + j] + self.pos[i * 128 + j] + self.ttype[j];
            }
        }
        ops::layer_norm(&mut x, 128, Some((self.ln_g, self.ln_b)), 1e-12);
        let mut x = self.map_in.apply(&x, l);
        let hd = BERT / HEADS;
        let scale = 1.0 / tensor::sqrtf(hd as f32);
        for _ in 0..self.layers {
            let q = self.q.apply(&x, l);
            let k = self.k.apply(&x, l);
            let v = self.v.apply(&x, l);
            let mut ctx = vec![0f32; l * BERT];
            let mut sc = vec![0f32; l];
            for h in 0..HEADS {
                for i in 0..l {
                    let qi = &q[i * BERT + h * hd..i * BERT + (h + 1) * hd];
                    for j in 0..l {
                        let kj = &k[j * BERT + h * hd..j * BERT + (h + 1) * hd];
                        sc[j] = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    ops::softmax(&mut sc);
                    let out = &mut ctx[i * BERT + h * hd..i * BERT + (h + 1) * hd];
                    for j in 0..l {
                        let vj = &v[j * BERT + h * hd..j * BERT + (h + 1) * hd];
                        let p = sc[j];
                        for d in 0..hd {
                            out[d] += p * vj[d];
                        }
                    }
                }
            }
            let mut a = self.dense.apply(&ctx, l);
            tensor::add_into(&mut a, &x);
            ops::layer_norm(&mut a, BERT, Some(self.attn_ln), 1e-12);
            let mut h = self.ffn.apply(&a, l);
            for v in h.iter_mut() {
                *v = ops::gelu(*v);
            }
            let mut o = self.ffn_out.apply(&h, l);
            tensor::add_into(&mut o, &a);
            ops::layer_norm(&mut o, BERT, Some(self.full_ln), 1e-12);
            x = o;
        }
        x
    }
}

pub struct Model {
    bert: Albert,
    bert_enc: Lin,
    dur_lstms: [(Lstm, Lstm); 3],
    dur_ln: [Ada; 3],
    lstm: (Lstm, Lstm),
    dur_proj: Lin,
    shared: (Lstm, Lstm),
    f0: [Rb; 3],
    n: [Rb; 3],
    f0_proj: Lin,
    n_proj: Lin,
    emb: &'static [f32],
    cnn: [(Conv1d, &'static [f32], &'static [f32]); 3],
    te_lstm: (Lstm, Lstm),
    f0_conv: (&'static [f32], f32),
    n_conv: (&'static [f32], f32),
    encode: Rb,
    decode: [Rb; 4],
    asr_res: Conv1d,
    src_lin: (&'static [f32], f32),
    noise_convs: [Conv1d; 2],
    noise_res: [RbG; 2],
    ups: [ConvT1d; 2],
    res: Vec<RbG>,
    post: Conv1d,
    voice: &'static [f32],
    vocab: Vec<(u32, u8)>,
}

/// What `front` answers: everything the alignment does not change.
pub struct Front {
    pub ids: Vec<u8>,
    /// `(L, 640)`: the duration encoder's output, the thing expanded by the
    /// alignment into the prosody input.
    pub d: Vec<f32>,
    /// `(L, 512)`: the text encoder's output, expanded into the decoder's.
    pub t_en: Vec<f32>,
    /// The model's own duration per token, in frames, before rounding.
    pub dur: Vec<f32>,
}

/// For `verify`: a harmonic source to use instead of the kernel's own
/// random one, and every generator intermediate on the way out, so the
/// vocoder can be compared exactly against the host rather than on an
/// envelope.
#[derive(Default)]
pub struct Trace {
    pub har: Option<Vec<f32>>,
    pub taps: Vec<(&'static str, Vec<f32>)>,
}

pub struct Spoken {
    pub audio: Vec<f32>,
    /// Pitch and noise curves, one per 12.5 ms (two per duration frame).
    pub f0: Vec<f32>,
    pub n: Vec<f32>,
}

impl Model {
    pub fn load(f: &VoiceFile) -> R<Model> {
        let g = |n: &str, d: &[usize]| f.f32_shaped(n, d).map_err(|w| e(n, w));
        let p = "predictor";
        let cnn = |i: usize| -> R<(Conv1d, &'static [f32], &'static [f32])> {
            Ok((
                conv(f, &alloc::format!("text_encoder.cnn.{i}.0"), HID, HID, 5, 1, 1, 2, true)?,
                g(&alloc::format!("text_encoder.cnn.{i}.1.gamma"), &[HID])?,
                g(&alloc::format!("text_encoder.cnn.{i}.1.beta"), &[HID])?,
            ))
        };
        let mut res = Vec::new();
        for (i, c) in [256usize, 128].iter().enumerate() {
            for (j, k) in [3usize, 7, 11].iter().enumerate() {
                res.push(RbG::load(f, &alloc::format!("decoder.generator.resblocks.{}", i * 3 + j), *c, *k)?);
            }
        }
        let cps = f.f32("vocab.codepoints").map_err(|w| e("vocab", w))?;
        let ids = f.f32("vocab.ids").map_err(|w| e("vocab", w))?;
        let vocab: Vec<(u32, u8)> = cps.iter().zip(ids).map(|(c, i)| (*c as u32, *i as u8)).collect();
        let small = |n: &str| -> R<(&'static [f32], f32)> { Ok((g(&alloc::format!("{n}.weight"), &[1, 3])?, g(&alloc::format!("{n}.bias"), &[1])?[0])) };
        Ok(Model {
            bert: Albert::load(f)?,
            bert_enc: lin(f, "bert_encoder", BERT, HID, true)?,
            dur_lstms: [
                lstm_pair(f, &alloc::format!("{p}.text_encoder.lstms.0"), HID + STYLE, HID / 2)?,
                lstm_pair(f, &alloc::format!("{p}.text_encoder.lstms.2"), HID + STYLE, HID / 2)?,
                lstm_pair(f, &alloc::format!("{p}.text_encoder.lstms.4"), HID + STYLE, HID / 2)?,
            ],
            dur_ln: [
                ada(f, &alloc::format!("{p}.text_encoder.lstms.1"), HID)?,
                ada(f, &alloc::format!("{p}.text_encoder.lstms.3"), HID)?,
                ada(f, &alloc::format!("{p}.text_encoder.lstms.5"), HID)?,
            ],
            lstm: lstm_pair(f, &alloc::format!("{p}.lstm"), HID + STYLE, HID / 2)?,
            dur_proj: lin(f, &alloc::format!("{p}.duration_proj.linear_layer"), HID, 50, true)?,
            shared: lstm_pair(f, &alloc::format!("{p}.shared"), HID + STYLE, HID / 2)?,
            f0: [Rb::load(f, &alloc::format!("{p}.F0.0"), HID, HID, false)?, Rb::load(f, &alloc::format!("{p}.F0.1"), HID, HID / 2, true)?, Rb::load(f, &alloc::format!("{p}.F0.2"), HID / 2, HID / 2, false)?],
            n: [Rb::load(f, &alloc::format!("{p}.N.0"), HID, HID, false)?, Rb::load(f, &alloc::format!("{p}.N.1"), HID, HID / 2, true)?, Rb::load(f, &alloc::format!("{p}.N.2"), HID / 2, HID / 2, false)?],
            f0_proj: lin(f, &alloc::format!("{p}.F0_proj"), HID / 2, 1, true)?,
            n_proj: lin(f, &alloc::format!("{p}.N_proj"), HID / 2, 1, true)?,
            emb: g("text_encoder.embedding.weight", &[178, HID])?,
            cnn: [cnn(0)?, cnn(1)?, cnn(2)?],
            te_lstm: lstm_pair(f, "text_encoder.lstm", HID, HID / 2)?,
            f0_conv: small("decoder.F0_conv")?,
            n_conv: small("decoder.N_conv")?,
            encode: Rb::load(f, "decoder.encode", HID + 2, 1024, false)?,
            decode: [
                Rb::load(f, "decoder.decode.0", 1024 + 2 + 64, 1024, false)?,
                Rb::load(f, "decoder.decode.1", 1024 + 2 + 64, 1024, false)?,
                Rb::load(f, "decoder.decode.2", 1024 + 2 + 64, 1024, false)?,
                Rb::load(f, "decoder.decode.3", 1024 + 2 + 64, 512, true)?,
            ],
            asr_res: conv(f, "decoder.asr_res.0", HID, 64, 1, 1, 1, 0, true)?,
            src_lin: (g("decoder.generator.m_source.l_linear.weight", &[1, 9])?, g("decoder.generator.m_source.l_linear.bias", &[1])?[0]),
            noise_convs: [
                conv(f, "decoder.generator.noise_convs.0", NFFT + 2, 256, 12, 1, 6, 3, true)?,
                conv(f, "decoder.generator.noise_convs.1", NFFT + 2, 128, 1, 1, 1, 0, true)?,
            ],
            noise_res: [RbG::load(f, "decoder.generator.noise_res.0", 256, 7)?, RbG::load(f, "decoder.generator.noise_res.1", 128, 11)?],
            ups: [convt(f, "decoder.generator.ups.0", 512, 256, 20, 10, 5)?, convt(f, "decoder.generator.ups.1", 256, 128, 12, 6, 3)?],
            res,
            post: conv(f, "decoder.generator.conv_post", 128, NFFT + 2, 7, 1, 1, 3, true)?,
            voice: g("voice.af_bella", &[510, 256])?,
            vocab,
        })
    }

    /// Phoneme string to ids, with the 0 the model expects at both ends.
    /// A character the vocabulary lacks is an error: a missing symbol is
    /// a misspelt phoneme, not a thing to skip.
    pub fn ids(&self, phonemes: &str) -> Result<Vec<u8>, char> {
        let mut out = vec![0u8];
        for ch in phonemes.chars() {
            match self.vocab.iter().find(|(c, _)| *c == ch as u32) {
                Some((_, id)) => out.push(*id),
                None => return Err(ch),
            }
        }
        out.push(0);
        Ok(out)
    }

    /// The style for a sequence: row `ids.len() - 3` of the voice pack.
    fn style(&self, l: usize) -> (&[f32], &[f32]) {
        let row = (l.saturating_sub(3)).min(509);
        let s = &self.voice[row * 256..(row + 1) * 256];
        (&s[..STYLE], &s[STYLE..])
    }

    pub fn front(&self, ids: &[u8]) -> Front {
        let l = ids.len();
        let (_, sp) = self.style(l);
        let bert = self.bert.apply(ids);
        let d_en = self.bert_enc.apply(&bert, l);
        // the duration encoder: style concatenated onto every frame, three
        // bidirectional LSTMs each followed by an adaptive layer norm
        let cat = |x: &[f32], c: usize| -> Vec<f32> {
            let mut o = Vec::with_capacity(l * (c + STYLE));
            for i in 0..l {
                o.extend_from_slice(&x[i * c..(i + 1) * c]);
                o.extend_from_slice(sp);
            }
            o
        };
        let mut x = cat(&d_en, HID);
        for i in 0..3 {
            let mut h = ops::lstm_bi(&self.dur_lstms[i].0, &self.dur_lstms[i].1, &x, l);
            self.dur_ln[i].apply(&mut h, l, sp, Norm::Layer);
            x = cat(&h, HID);
        }
        let d = x;
        let h = ops::lstm_bi(&self.lstm.0, &self.lstm.1, &d, l);
        let logits = self.dur_proj.apply(&h, l);
        let dur: Vec<f32> = logits.chunks(50).map(|r| r.iter().map(|v| ops::sigmoid(*v)).sum::<f32>()).collect();
        // the text encoder
        let mut t = vec![0f32; l * HID];
        for (i, &id) in ids.iter().enumerate() {
            t[i * HID..(i + 1) * HID].copy_from_slice(&self.emb[id as usize * HID..(id as usize + 1) * HID]);
        }
        for (c, gamma, beta) in &self.cnn {
            t = c.apply(&t, l);
            ops::layer_norm(&mut t, HID, Some((gamma, beta)), 1e-5);
            ops::leaky_relu(&mut t, 0.2);
        }
        let t_en = ops::lstm_bi(&self.te_lstm.0, &self.te_lstm.1, &t, l);
        Front { ids: ids.to_vec(), d, t_en, dur }
    }

    /// From a front and a duration per token (frames of 25 ms, at least
    /// one each) to audio.
    pub fn synth(&self, fr: &Front, dur: &[usize], rng: &mut Rng) -> Spoken {
        self.synth_traced(fr, dur, rng, None)
    }

    pub fn synth_traced(&self, fr: &Front, dur: &[usize], rng: &mut Rng, trace: Option<&mut Trace>) -> Spoken {
        let l = fr.ids.len();
        let (sd, sp) = self.style(l);
        let t: usize = dur.iter().sum();
        let expand = |x: &[f32], c: usize| -> Vec<f32> {
            let mut o = Vec::with_capacity(t * c);
            for (i, &n) in dur.iter().enumerate() {
                for _ in 0..n {
                    o.extend_from_slice(&x[i * c..(i + 1) * c]);
                }
            }
            o
        };
        let en = expand(&fr.d, HID + STYLE);
        let shared = ops::lstm_bi(&self.shared.0, &self.shared.1, &en, t);
        let branch = |blocks: &[Rb; 3], proj: &Lin| -> Vec<f32> {
            let (mut x, mut tt) = (shared.clone(), t);
            for b in blocks {
                let (y, t2) = b.apply(&x, tt, sp);
                x = y;
                tt = t2;
            }
            proj.apply(&x, tt)
        };
        let f0 = branch(&self.f0, &self.f0_proj);
        let n = branch(&self.n, &self.n_proj);
        let audio = self.decode(&expand(&fr.t_en, HID), t, &f0, &n, sd, rng, trace);
        Spoken { audio, f0, n }
    }

    fn decode(&self, asr: &[f32], t: usize, f0: &[f32], n: &[f32], s: &[f32], rng: &mut Rng, trace: Option<&mut Trace>) -> Vec<f32> {
        // F0_conv / N_conv: a (1,1,3) stride-2 conv over the 2t curve -> t
        let half = |curve: &[f32], (w, b): (&[f32], f32)| -> Vec<f32> {
            (0..t)
                .map(|i| {
                    let at = |j: isize| -> f32 {
                        let p = 2 * i as isize - 1 + j;
                        if p >= 0 && (p as usize) < curve.len() { curve[p as usize] } else { 0.0 }
                    };
                    w[0] * at(0) + w[1] * at(1) + w[2] * at(2) + b
                })
                .collect()
        };
        let f0h = half(f0, self.f0_conv);
        let nh = half(n, self.n_conv);
        let mut x = Vec::with_capacity(t * (HID + 2));
        for i in 0..t {
            x.extend_from_slice(&asr[i * HID..(i + 1) * HID]);
            x.push(f0h[i]);
            x.push(nh[i]);
        }
        let (mut x, _) = self.encode.apply(&x, t, s);
        let res = self.asr_res.apply(asr, t);
        let mut tt = t;
        for (bi, b) in self.decode.iter().enumerate() {
            let mut cat = Vec::with_capacity(tt * 1090);
            for i in 0..tt {
                cat.extend_from_slice(&x[i * 1024..(i + 1) * 1024]);
                cat.extend_from_slice(&res[i * 64..(i + 1) * 64]);
                cat.push(f0h[i]);
                cat.push(nh[i]);
            }
            let (y, t2) = b.apply(&cat, tt, s);
            x = y;
            tt = t2;
            let _ = bi;
        }
        // x is (2t, 512); the generator
        self.generator(x, tt, f0, s, rng, trace)
    }

    fn generator(&self, mut x: Vec<f32>, t2: usize, f0: &[f32], s: &[f32], rng: &mut Rng, mut trace: Option<&mut Trace>) -> Vec<f32> {
        let samples = t2 * 300;
        let injected = trace.as_deref_mut().and_then(|tr| tr.har.take());
        let mut tap = |name: &'static str, v: &Vec<f32>| {
            if let Some(tr) = trace.as_deref_mut() {
                tr.taps.push((name, v.clone()));
            }
        };
        tap("x_dec", &x);
        let har = match injected {
            Some(h) => h,
            None => self.source(f0, samples, rng),
        };
        let (spec, frames) = stft(&har);
        // (frames, 22): magnitude then phase, as the model concatenates them
        let mut harf = Vec::with_capacity(frames * (NFFT + 2));
        for i in 0..frames {
            for k in 0..=NFFT / 2 {
                harf.push(spec[i * (NFFT / 2 + 1) + k].0);
            }
            for k in 0..=NFFT / 2 {
                harf.push(spec[i * (NFFT / 2 + 1) + k].1);
            }
        }
        tap("harf", &harf);
        let mut tt = t2;
        for i in 0..2 {
            ops::leaky_relu(&mut x, 0.1);
            let mut src = self.noise_convs[i].apply(&harf, frames);
            if i == 0 {
                tap("nc0", &src);
            }
            let ts = self.noise_convs[i].out_len(frames);
            self.noise_res[i].apply(&mut src, ts, s);
            tap(if i == 0 { "nr0" } else { "nr1" }, &src);
            x = self.ups[i].apply(&x, tt);
            tt = self.ups[i].out_len(tt);
            tap(if i == 0 { "up0" } else { "up1" }, &x);
            if i == 1 {
                // ReflectionPad1d((1, 0)): one frame on the left, mirrored
                let c = self.ups[i].cout;
                let mut p = Vec::with_capacity((tt + 1) * c);
                p.extend_from_slice(&x[c..2 * c]);
                p.extend_from_slice(&x);
                x = p;
                tt += 1;
            }
            debug_assert_eq!(tt, ts);
            let n = x.len();
            tensor::add_into(&mut x, &src[..n]);
            let mut acc = vec![0f32; x.len()];
            for j in 0..3 {
                let mut y = x.clone();
                self.res[i * 3 + j].apply(&mut y, tt, s);
                tensor::add_into(&mut acc, &y);
            }
            for v in acc.iter_mut() {
                *v /= 3.0;
            }
            x = acc;
        }
        ops::leaky_relu(&mut x, 0.01);
        tap("postin", &x);
        let out = self.post.apply(&x, tt);
        tap("post", &out);
        let mut spec: Vec<(f32, f32)> = Vec::with_capacity(tt * (NFFT / 2 + 1));
        for i in 0..tt {
            let fr = &out[i * (NFFT + 2)..(i + 1) * (NFFT + 2)];
            for k in 0..=NFFT / 2 {
                spec.push((tensor::expf(fr[k]), tensor::sinf(fr[NFFT / 2 + 1 + k])));
            }
        }
        istft(&spec, tt)
    }

    /// The harmonic source: nine sines at multiples of f0 with a random
    /// phase above the first, noise where unvoiced, merged by one linear
    /// layer and a tanh.
    fn source(&self, f0: &[f32], samples: usize, rng: &mut Rng) -> Vec<f32> {
        let frames = f0.len();
        let (w, b) = self.src_lin;
        let mut out = vec![0f32; samples];
        let mut rad_ini = [0f32; 9];
        for r in rad_ini.iter_mut().skip(1) {
            *r = rng.uniform();
        }
        // per harmonic: the phase at each frame boundary is the cumulative
        // sum of the per-frame increment, and within a frame it runs
        // linearly between boundaries (the reference's linear upsample)
        let mut phase_at = vec![0f32; frames];
        for h in 0..9 {
            let mult = (h + 1) as f32;
            let mut acc = 0f32;
            for i in 0..frames {
                let mut rad = (f0[i] * mult / SR as f32) % 1.0;
                if i == 0 {
                    rad += rad_ini[h];
                }
                acc += rad;
                phase_at[i] = acc * core::f32::consts::TAU;
            }
            for n in 0..samples {
                let p = (n as f32 + 0.5) / 300.0 - 0.5;
                let i0 = if p < 0.0 { 0 } else { (p as usize).min(frames - 1) };
                let i1 = (i0 + 1).min(frames - 1);
                let frac = if p < 0.0 { 0.0 } else { (p - i0 as f32).min(1.0) };
                // the reference multiplies the frame phase by the upsample
                // factor before interpolating: a frame's increment was summed
                // once per frame and stands for 300 samples of it
                let ph = 300.0 * (phase_at[i0] * (1.0 - frac) + phase_at[i1] * frac);
                let fi = (n / 300).min(frames - 1);
                let voiced = f0[fi] > 10.0;
                let sine = if voiced { 0.1 * tensor::sinf(ph) } else { 0.0 };
                let noise = rng.normal() * if voiced { 0.003 } else { 0.1 / 3.0 };
                out[n] += w[h] * (sine + noise);
            }
        }
        for v in out.iter_mut() {
            *v = ops::tanh(*v + b);
        }
        out
    }
}

/// A 20-point STFT, hop 5, periodic Hann, centred with reflection
/// padding: `(magnitude, phase)` per bin per frame, `1 + len/hop` frames.
fn stft(x: &[f32]) -> (Vec<(f32, f32)>, usize) {
    let half = NFFT / 2;
    let frames = 1 + x.len() / HOP;
    let win: Vec<f32> = (0..NFFT).map(|n| 0.5 - 0.5 * tensor::cosf(core::f32::consts::TAU * n as f32 / NFFT as f32)).collect();
    let at = |i: isize| -> f32 {
        // reflect padding by `half` on both sides
        let n = x.len() as isize;
        let j = if i < 0 { -i } else if i >= n { 2 * (n - 1) - i } else { i };
        x[j.clamp(0, n - 1) as usize]
    };
    let mut out = Vec::with_capacity(frames * (half + 1));
    for f in 0..frames {
        let base = (f * HOP) as isize - half as isize;
        for k in 0..=half {
            let (mut re, mut im) = (0f32, 0f32);
            for n in 0..NFFT {
                let v = at(base + n as isize) * win[n];
                let a = core::f32::consts::TAU * (k * n) as f32 / NFFT as f32;
                re += v * tensor::cosf(a);
                im -= v * tensor::sinf(a);
            }
            // DC and Nyquist are real by construction; the rounding left
            // in `im` by sin(pi n) would otherwise pick -pi or +pi at random
            // where the reference, whose FFT gets an exact zero, says +pi.
            // The convolution after this reads the phase as a number.
            if k == 0 || k == half {
                im = 0.0;
            }
            out.push((tensor::sqrtf(re * re + im * im), atan2(im, re)));
        }
    }
    (out, frames)
}

/// The inverse: overlap-add of windowed inverse DFTs, normalised by the
/// window's own overlap, the centre padding trimmed. `hop * (frames - 1)`
/// samples.
fn istft(spec: &[(f32, f32)], frames: usize) -> Vec<f32> {
    let half = NFFT / 2;
    let len = HOP * (frames - 1);
    let win: Vec<f32> = (0..NFFT).map(|n| 0.5 - 0.5 * tensor::cosf(core::f32::consts::TAU * n as f32 / NFFT as f32)).collect();
    let total = len + NFFT;
    let mut y = vec![0f32; total];
    let mut wsum = vec![0f32; total];
    for f in 0..frames {
        let fr = &spec[f * (half + 1)..(f + 1) * (half + 1)];
        for n in 0..NFFT {
            let mut v = 0f32;
            for k in 0..=half {
                let (m, p) = fr[k];
                let a = core::f32::consts::TAU * (k * n) as f32 / NFFT as f32;
                let term = m * tensor::cosf(a + p);
                v += if k == 0 || k == half { term } else { 2.0 * term };
            }
            v /= NFFT as f32;
            y[f * HOP + n] += v * win[n];
            wsum[f * HOP + n] += win[n] * win[n];
        }
    }
    (0..len).map(|i| if wsum[i + half] > 1e-8 { y[i + half] / wsum[i + half] } else { 0.0 }).collect()
}

fn atan2(y: f32, x: f32) -> f32 {
    // good to ~1e-4, which is what a phase fed to a convolution needs
    let ax = x.abs();
    let ay = y.abs();
    let a = ax.min(ay) / ax.max(ay).max(1e-20);
    let s = a * a;
    let mut r = ((-0.046_496_47 * s + 0.159_314_22) * s - 0.327_622_76) * s * a + a;
    if ay > ax {
        r = core::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = core::f32::consts::PI - r;
    }
    if y < 0.0 {
        -r
    } else {
        r
    }
}

/// splitmix64 with Box-Muller: the noise the source module needs, seeded
/// per utterance so a sentence rendered twice is the same sentence.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        tensor::sqrtf(-2.0 * tensor::lnf(u1)) * tensor::cosf(core::f32::consts::TAU * u2)
    }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    // stft/istft round trip on a tone: the pair must be each other's inverse
    let x: Vec<f32> = (0..400).map(|n| tensor::sinf(0.3 * n as f32)).collect();
    let (s, frames) = stft(&x);
    let y = istft(&s, frames);
    let err = x.iter().zip(&y).skip(20).take(360).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mut r = Rng::new(7);
    let mean: f32 = (0..4000).map(|_| r.normal()).sum::<f32>() / 4000.0;
    vec![
        ("the 20-point istft inverts the stft to 1e-3", y.len() == 400 && err < 1e-3),
        ("atan2 agrees with the quadrants", (atan2(1.0, 1.0) - 0.785_398).abs() < 1e-3 && (atan2(1.0, -1.0) - 2.356_194).abs() < 1e-3 && (atan2(-1.0, -1.0) + 2.356_194).abs() < 1e-3),
        ("the noise source is zero-mean", mean.abs() < 0.1),
    ]
}
