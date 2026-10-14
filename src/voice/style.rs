//! The house style: Valve's published recipe, replicated over Kokoro.
//!
//! The original was performed after a rule-based speech synthesiser and
//! then processed -- pitch held to notes, formants moved, breath removed.
//! Here the rules come first: `plan` writes a line's timing and melody the
//! way such an engine did (even durations, a hat-pattern accent on each
//! content word, a declining baseline, a clipped ending), Kokoro is made
//! to perform exactly that timing by having its predicted durations
//! replaced, and then its *voice* is taken away. What survives of it is
//! an envelope: the spectrum of each 5 ms frame, smoothed to a few dozen
//! cepstral coefficients. Under that envelope goes a source of this
//! file's own -- a pulse train at the planned note, noise where the model
//! predicted noise -- and the formant warp and the 1.2 kHz colour. The
//! pitch tracker, the aperiodicity estimate and the vocoder the host
//! prototype borrowed from WORLD turned out to be unnecessary once the
//! voicing came from Kokoro's own pitch prediction; this is the shape the
//! prototype `glados_kshape.py` settled on, and the one the user heard.
//!
//! **The mask.** One number, 1.0 composed and 0.0 gone, drives every knob.
//! Composed is a machine: one held note with a step on stress, the
//! envelope liftered to 28 coefficients so no shimmer of a throat is
//! left, no breath, loudness evened, timing relaxed enough that short
//! words survive (`mech` 0.3 -- the metronomic 0.65 squeezed "you are"
//! out of "you are awake"). As the mask slips the melody widens, the
//! endings fall or lift, the pitch drops and breath returns; the machine
//! fades out between 1.0 and 0.7, so a slipping or unmasked voice is the
//! K recipe untouched.
//!
//! Numbers here are the prototype's and were chosen by ear over about
//! forty renders; none of them is derived.

use super::fft::{self, C};
use super::ipa::{self, Lexicon};
use super::kokoro::{Model, Rng, FRAME, SR};
use crate::ai::tensor;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

const NFFT: usize = 1024;
const HOP: usize = 120;
/// Noise grains against pulse responses, equal rms on a flat envelope:
/// measured on the host, not derived.
const NOISE_K: f32 = 1.62;

const VOWELS: &str = "AIOWYQaeiouæɑɒɔəɚɛɜɪʊʌᵊɐᵻ";
const FUNCTION: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "and", "or", "but", "is", "are", "was", "were", "be", "been", "am", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those", "as", "from", "will", "would", "can", "could", "do", "does", "did", "not", "so", "if", "than", "then", "there", "here", "have", "has", "had",
];

/// What the brain hands the voice.
#[derive(Clone, Copy)]
pub struct Dial {
    pub mask: f32,
    pub valence: f32,
    pub arousal: f32,
}

/// Every knob, derived from the dial.
struct Knobs {
    base: f32,
    formant: f32,
    accent: f32,
    mech: f32,
    decl: f32,
    fall: f32,
    lift: f32,
    mach: f32,
}

impl Knobs {
    fn from(d: Dial) -> Knobs {
        let mk = d.mask.clamp(0.0, 1.0);
        let slip = 1.0 - mk;
        let mach = ((mk - 0.7) / 0.3).clamp(0.0, 1.0);
        let mech_k = 0.15 + 0.5 * mk;
        Knobs {
            base: 4.0 + 5.5 * mk,
            formant: -2.5 + 1.5 * mk,
            accent: (0.4 + 4.2 * slip) * (1.0 + 0.3 * d.arousal),
            mech: mech_k * (1.0 - mach) + 0.3 * mach,
            decl: 0.6 + 2.4 * slip,
            fall: 0.5 + 3.5 * slip,
            lift: 0.5 + 4.0 * slip,
            mach,
        }
    }
}

struct Tok {
    ch: char,
    content: bool,
    emph: bool,
}

/// A phrase's tokens, with the rule duration (frames) and pitch target
/// (semitones above A2) each one gets.
struct Plan {
    ids: Vec<u8>,
    dur: Vec<usize>,
    pitch: Vec<Option<f32>>,
}

fn plan(m: &Model, lex: Option<&Lexicon>, phrase: &str, final_up: bool, k: &Knobs, val: f32) -> Plan {
    let mut toks: Vec<Tok> = Vec::new();
    let mut word = String::new();
    let mut flush = |word: &mut String, toks: &mut Vec<Tok>| {
        if word.is_empty() {
            return;
        }
        let emph = word.starts_with('*') && word.ends_with('*') && word.len() > 2;
        let w = word.trim_matches('*');
        if !w.is_empty() {
            let (p, _) = ipa::word(lex, w);
            if !toks.is_empty() {
                toks.push(Tok { ch: ' ', content: false, emph: false });
            }
            let content = !FUNCTION.contains(&w.to_lowercase().as_str());
            for ch in p.chars() {
                toks.push(Tok { ch, content, emph });
            }
        }
        word.clear();
    };
    for c in phrase.chars() {
        if c.is_alphabetic() || c == '\'' || c == '\u{2019}' || c == '*' {
            word.push(c);
        } else {
            flush(&mut word, &mut toks);
            if matches!(c, '.' | ',' | '!' | '?' | ';' | ':') {
                toks.push(Tok { ch: c, content: false, emph: false });
            }
        }
    }
    flush(&mut word, &mut toks);
    let n = toks.len().max(1);
    let last_vowel = toks.iter().rposition(|t| VOWELS.contains(t.ch));
    let mut dur = Vec::with_capacity(n);
    let mut pitch = Vec::with_capacity(n);
    let mut stressed = 0u8;
    for (i, t) in toks.iter().enumerate() {
        if t.ch == 'ˈ' || t.ch == 'ˌ' {
            stressed = if t.ch == 'ˈ' { 2 } else { 1 };
            dur.push(0);
            pitch.push(None);
            continue;
        }
        let progress = i as f32 / (n - 1).max(1) as f32;
        let decl = k.decl * (0.66 - progress);
        if VOWELS.contains(t.ch) {
            let mut d = match stressed {
                2 => 6,
                1 => 5,
                _ => 4,
            };
            let mut p = k.base + decl;
            if t.content && stressed == 2 {
                p += k.accent;
            } else if t.content && stressed == 1 {
                p += k.accent / 3.0;
            }
            if t.emph && stressed > 0 {
                p += 2.0;
                d = d * 3 / 2;
            }
            if Some(i) == last_vowel {
                p = k.base + decl + if final_up { k.lift } else { -k.fall - 1.0 * (-val).max(0.0) };
                d = 6;
            }
            stressed = 0;
            dur.push(d);
            pitch.push(Some(p));
        } else if t.ch == ' ' {
            dur.push(1);
            pitch.push(None);
        } else {
            dur.push(2);
            pitch.push(None);
        }
    }
    // only characters the model has a token for survive, in step
    let mut ids = vec![0u8];
    let mut kd = Vec::new();
    let mut kp = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if let Some(id) = m.id_of(t.ch) {
            ids.push(id);
            kd.push(dur[i]);
            kp.push(pitch[i]);
        }
    }
    ids.push(0);
    Plan { ids, dur: kd, pitch: kp }
}

/// The smoothed power spectrum per 5 ms frame: a Hann-windowed STFT, its
/// log liftered to `q` cepstral coefficients.
fn envelope(x: &[f32], q: usize) -> Vec<Vec<f32>> {
    if x.len() < NFFT {
        return Vec::new();
    }
    let n = (x.len() - NFFT) / HOP + 1;
    let win: Vec<f32> = (0..NFFT).map(|i| 0.5 - 0.5 * tensor::cosf(core::f32::consts::TAU * i as f32 / (NFFT - 1) as f32)).collect();
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0f32; NFFT];
    for f in 0..n {
        for i in 0..NFFT {
            buf[i] = x[f * HOP + i] * win[i];
        }
        let bins = fft::rfft(&buf);
        let lg: Vec<C> = bins.iter().map(|c| C::new(tensor::lnf(c.abs2() + 1e-12), 0.0)).collect();
        let mut cep = fft::irfft(&lg, NFFT);
        for i in q..NFFT - q + 1 {
            cep[i] = 0.0;
        }
        let back = fft::rfft(&cep);
        out.push(back.iter().map(|c| tensor::expf(c.re)).collect());
    }
    out
}

/// Pulses at the note through the frame's envelope, and noise grains,
/// level-matched frame by frame to the source audio.
fn resynth(env: &[Vec<f32>], f0: &[f32], ap: &[f32], ref_audio: &[f32], env_gain: &[f32], rng: &mut Rng) -> Vec<f32> {
    let n = env.len();
    let total = n * HOP + NFFT;
    let mut pul = vec![0f32; total];
    let mut nz = vec![0f32; total];
    let win: Vec<f32> = (0..NFFT).map(|i| 0.5 - 0.5 * tensor::cosf(core::f32::consts::TAU * i as f32 / (NFFT - 1) as f32)).collect();
    let mags: Vec<Vec<f32>> = env.iter().map(|e| e.iter().map(|v| tensor::sqrtf(*v)).collect()).collect();
    let mut phase = 0f32;
    let mut bins = vec![C::default(); NFFT / 2 + 1];
    for t in 0..n * HOP {
        let i = (t / HOP).min(n - 1);
        if f0[i] > 0.0 {
            let per = SR as f32 / f0[i];
            if phase >= per {
                phase -= per;
                let frac = -phase;
                for (k, b) in bins.iter_mut().enumerate() {
                    let a = -core::f32::consts::TAU * k as f32 * frac / NFFT as f32;
                    *b = C::new(mags[i][k] * tensor::cosf(a), mags[i][k] * tensor::sinf(a));
                }
                let h = fft::irfft(&bins, NFFT);
                let g = tensor::sqrtf(1.0 - ap[i]);
                let a0 = t as isize - (NFFT / 2) as isize;
                for j in 0..NFFT {
                    let p = a0 + j as isize;
                    if p >= 0 && (p as usize) < total {
                        // rolled by NFFT/2: sample j of the output is h[(j + NFFT/2) % NFFT]
                        pul[p as usize] += h[(j + NFFT / 2) % NFFT] * win[j] * g;
                    }
                }
            }
            phase += 1.0;
        } else {
            phase = (phase + 1.0).min(1e6);
        }
    }
    let mut noise = vec![0f32; NFFT];
    for i in 0..n {
        let share = if f0[i] > 0.0 { ap[i] } else { 1.0 };
        if share <= 0.0 {
            continue;
        }
        for v in noise.iter_mut() {
            *v = rng.normal();
        }
        let mut nb = fft::rfft(&noise);
        let s = 1.0 / tensor::sqrtf(NFFT as f32);
        for (k, b) in nb.iter_mut().enumerate() {
            b.re *= s * mags[i][k];
            b.im *= s * mags[i][k];
        }
        let g = fft::irfft(&nb, NFFT);
        let scale = tensor::sqrtf(share) * NOISE_K;
        for j in 0..NFFT {
            nz[i * HOP + j] += g[j] * win[j] * scale;
        }
    }
    let mut out: Vec<f32> = pul.iter().zip(&nz).map(|(a, b)| a + b).collect();
    // level: per frame, the source's rms times the envelope edit's gain,
    // against the output's own; smoothed over three frames
    let mut gain = vec![1f32; n];
    for i in 0..n {
        let o = &out[i * HOP..i * HOP + NFFT];
        let r = &ref_audio[(i * HOP).min(ref_audio.len())..(i * HOP + NFFT).min(ref_audio.len())];
        let want = tensor::sqrtf(r.iter().map(|v| v * v).sum::<f32>() / r.len().max(1) as f32 + 1e-12) * env_gain[i];
        let have = tensor::sqrtf(o.iter().map(|v| v * v).sum::<f32>() / NFFT as f32 + 1e-12);
        gain[i] = if have > 1e-5 { want / have } else { 0.0 };
    }
    let sm: Vec<f32> = (0..n).map(|i| (gain[i.saturating_sub(1)] + gain[i] + gain[(i + 1).min(n - 1)]) / 3.0).collect();
    let mut y = vec![0f32; total];
    for i in 0..n {
        for j in 0..NFFT {
            y[i * HOP + j] += out[i * HOP + j] * win[j] * sm[i];
        }
    }
    let norm = 1.0 / (NFFT as f32 * 0.5 / HOP as f32);
    for v in y.iter_mut() {
        *v *= norm;
    }
    y
}

/// One phrase: plan, perform, take the envelope, put the source under it.
fn phrase(m: &Model, lex: Option<&Lexicon>, text: &str, final_up: bool, k: &Knobs, d: Dial, rng: &mut Rng) -> Vec<f32> {
    let p = plan(m, lex, text, final_up, k, d.valence);
    if p.ids.len() < 3 {
        return Vec::new();
    }
    let fr = m.front(&p.ids);
    // the model's durations blended toward the rules; a stress mark's own
    // token keeps a frame or the alignment loses it
    let mut dur: Vec<usize> = Vec::with_capacity(p.ids.len());
    for (i, md) in fr.dur.iter().enumerate() {
        let rule = if i == 0 || i + 1 == fr.dur.len() { 2.0 } else { p.dur[i - 1] as f32 };
        let mixed = tensor::roundf((1.0 - k.mech) * md + k.mech * rule);
        dur.push((mixed.max(1.0)) as usize);
    }
    let sp = m.synth(&fr, &dur, rng);
    let audio = sp.audio;
    // the machine's discrete choices switch together, half way along
    // the fade, while every continuous knob fades
    let machine = k.mach > 0.5;
    let q = if machine { 28 } else { 60 };
    let env = envelope(&audio, q);
    let n = env.len();
    if n == 0 {
        return Vec::new();
    }
    // the rule contour per 5 ms frame, and which frames are a vowel
    let mut target = vec![f32::NAN; n + 400];
    let mut inv = vec![false; n + 400];
    let mut pos = 0usize;
    let mut note = None;
    for (ti, &dfr) in dur.iter().enumerate() {
        let pt = if ti == 0 || ti + 1 == dur.len() { None } else { p.pitch[ti - 1] };
        if let Some(v) = pt {
            note = Some(tensor::roundf(v));
        }
        let a0 = pos * FRAME / SR * 200;
        let a1 = (pos + dfr) * FRAME / SR * 200;
        if let Some(nn) = note {
            for v in target.iter_mut().take(a1.min(n + 400)).skip(a0) {
                *v = nn;
            }
        }
        if pt.is_some() {
            for v in inv.iter_mut().take(a1.saturating_sub(2).min(n + 400)).skip(a0 + 2) {
                *v = true;
            }
        }
        pos += dfr;
    }
    let fallback = note.unwrap_or(k.base);
    let target: Vec<f32> = target[..n].iter().map(|v| if v.is_nan() { fallback } else { *v }).collect();
    // voicing from Kokoro's own pitch (12.5 ms frames read at 5 ms) or a
    // vowel's position, settled by a five-frame majority: a flicker
    // through a consonant is a pulse train restarting, which is a chirp
    let f0k: Vec<f32> = (0..n).map(|i| sp.f0[(i * sp.f0.len() / n).min(sp.f0.len() - 1)]).collect();
    let nk: Vec<f32> = (0..n).map(|i| sp.n[(i * sp.n.len() / n).min(sp.n.len() - 1)]).collect();
    let raw: Vec<bool> = (0..n).map(|i| f0k[i] > 80.0 || inv[i]).collect();
    let v: Vec<bool> = (0..n)
        .map(|i| {
            let lo = i.saturating_sub(2);
            let hi = (i + 3).min(n);
            raw[lo..hi].iter().filter(|b| **b).count() * 2 > 5
        })
        .collect();
    let (f0, ap): (Vec<f32>, Vec<f32>) = if machine {
        let f0 = (0..n)
            .map(|i| {
                if !v[i] {
                    return 0.0;
                }
                let step = if target[i] > k.base + 1.5 { 1.0 } else if target[i] < k.base - 1.5 { -1.0 } else { 0.0 };
                110.0 * tensor::powf(2.0, tensor::roundf(k.base + step) / 12.0)
            })
            .collect();
        (f0, vec![0.0; n])
    } else {
        let f0 = (0..n).map(|i| if v[i] { 110.0 * tensor::powf(2.0, target[i] / 12.0) } else { 0.0 }).collect();
        let ap = (0..n)
            .map(|i| {
                let sgm = tensor::sigmoid((nk[i] - 4.0) / 2.0);
                let a = sgm.clamp(0.0, 1.0) * 0.35;
                a * a
            })
            .collect();
        (f0, ap)
    };
    // the formant warp and the 1.2 kHz colour; the machine evens loudness,
    // the person's consonants are brightened
    let bins = NFFT / 2 + 1;
    let alpha = tensor::powf(2.0, k.formant / 12.0);
    let shape: Vec<f32> = (0..bins)
        .map(|b| {
            let hz = b as f32 * SR as f32 / NFFT as f32;
            let g = (hz - 1200.0) / 180.0;
            let s = 1.0 + 0.8 * tensor::expf(-g * g) + 0.18 * tensor::cosf(core::f32::consts::TAU * hz / 900.0);
            s * s
        })
        .collect();
    let mut env2: Vec<Vec<f32>> = env
        .iter()
        .map(|row| {
            (0..bins)
                .map(|b| {
                    let src = b as f32 / alpha;
                    let lo = (src as usize).min(bins - 1);
                    let hi = (lo + 1).min(bins - 1);
                    let fr = src - lo as f32;
                    (row[lo] * (1.0 - fr) + row[hi] * fr) * shape[b]
                })
                .collect()
        })
        .collect();
    if machine {
        let mut e: Vec<f32> = env2.iter().zip(&v).filter(|(_, vv)| **vv).map(|(r, _)| r.iter().sum::<f32>()).collect();
        if !e.is_empty() {
            e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            let med = e[e.len() / 2];
            for (i, row) in env2.iter_mut().enumerate() {
                if v[i] {
                    let s: f32 = row.iter().sum();
                    let g = tensor::sqrtf(med / s.max(1e-16)).clamp(0.3, 3.0);
                    for x in row.iter_mut() {
                        *x *= g;
                    }
                }
            }
        }
    } else {
        for (i, row) in env2.iter_mut().enumerate() {
            if !v[i] {
                for (b, x) in row.iter_mut().enumerate() {
                    if b as f32 * SR as f32 / NFFT as f32 > 3000.0 {
                        *x *= 2.0;
                    }
                }
            }
        }
    }
    let env_gain: Vec<f32> = env.iter().zip(&env2).map(|(a, b)| tensor::sqrtf(b.iter().sum::<f32>() / a.iter().sum::<f32>().max(1e-16))).collect();
    let y = resynth(&env2, &f0, &ap, &audio, &env_gain, rng);
    // trim to where there is sound, with a little before and after
    let w = 240;
    let envl: Vec<f32> = (0..y.len()).map(|i| y[i.saturating_sub(w / 2)..(i + w / 2).min(y.len())].iter().map(|v| v.abs()).sum::<f32>() / w as f32).collect();
    let mx = envl.iter().cloned().fold(0f32, f32::max);
    let on: Vec<usize> = (0..y.len()).filter(|&i| envl[i] > 0.02 * mx).collect();
    match (on.first(), on.last()) {
        (Some(&a), Some(&b)) => y[a.saturating_sub(240)..(b + 480).min(y.len())].to_vec(),
        _ => y,
    }
}

fn biquad(x: &mut [f32], b: [f32; 3], a: [f32; 3]) {
    let (mut x1, mut x2, mut y1, mut y2) = (0f32, 0f32, 0f32, 0f32);
    for v in x.iter_mut() {
        let xi = *v;
        let yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2;
        x2 = x1;
        x1 = xi;
        y2 = y1;
        y1 = yi;
        *v = yi;
    }
}

/// The finishing chain of the prototype: a slow levelling, a 180 Hz
/// high-pass, a 3 kHz presence shelf, a 1.3 ms comb, and a short room.
fn finish(mut y: Vec<f32>, room: f32, presence_db: f32, rng: &mut Rng) -> Vec<f32> {
    let n = y.len();
    if n == 0 {
        return y;
    }
    let win = (0.02 * SR as f32) as usize;
    let sq: Vec<f32> = y.iter().map(|v| v * v).collect();
    let mut env = vec![0f32; n];
    let mut acc = 0f32;
    for i in 0..n + win / 2 {
        if i < n {
            acc += sq[i];
        }
        if i >= win {
            acc -= sq[i - win];
        }
        if i >= win / 2 && i - win / 2 < n {
            env[i - win / 2] = tensor::sqrtf(acc.max(0.0) / win as f32) + 1e-4;
        }
    }
    let mut sorted = env.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let target = sorted[sorted.len() * 8 / 10];
    let g: Vec<f32> = env.iter().map(|e| (target / e).clamp(0.3, 4.0)).collect();
    let mut gs = vec![0f32; n];
    acc = 0.0;
    for i in 0..n + win / 2 {
        if i < n {
            acc += g[i];
        }
        if i >= win {
            acc -= g[i - win];
        }
        if i >= win / 2 && i - win / 2 < n {
            gs[i - win / 2] = acc / win as f32;
        }
    }
    for i in 0..n {
        if env[i] > target * 0.05 {
            y[i] *= gs[i];
        }
    }
    let sr = SR as f32;
    let w = core::f32::consts::TAU * 180.0 / sr;
    let al = tensor::sinf(w) / (2.0 * 0.707);
    let c = tensor::cosf(w);
    let a0 = 1.0 + al;
    biquad(&mut y, [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0], [1.0, -2.0 * c / a0, (1.0 - al) / a0]);
    let aa = tensor::powf(10.0, presence_db / 40.0);
    let w = core::f32::consts::TAU * 3000.0 / sr;
    let al = tensor::sinf(w) / 2.0;
    let c = tensor::cosf(w);
    let a0 = 1.0 + al / aa;
    biquad(&mut y, [(1.0 + al * aa) / a0, -2.0 * c / a0, (1.0 - al * aa) / a0], [1.0, -2.0 * c / a0, (1.0 - al / aa) / a0]);
    let d = (0.0013 * sr) as usize;
    for i in d..n {
        y[i] += 0.3 * y[i - d];
    }
    if room > 0.0 {
        // a 350 ms exponentially decaying noise room, by FFT convolution
        let ir_len = (0.35 * sr) as usize;
        let mut ir: Vec<f32> = (0..ir_len).map(|i| rng.normal() * tensor::expf(-(i as f32 / sr) / 0.09)).collect();
        let e = tensor::sqrtf(ir.iter().map(|v| v * v).sum::<f32>());
        for v in ir.iter_mut() {
            *v /= e;
        }
        let wet = convolve(&y, &ir);
        for i in 0..n {
            y[i] = (1.0 - room) * y[i] + room * wet[i];
        }
    }
    y
}

/// Overlap-add FFT convolution, 16384-point blocks.
fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    const N: usize = 16384;
    let block = N - h.len();
    let mut hb = vec![C::default(); N];
    for (i, v) in h.iter().enumerate() {
        hb[i] = C::new(*v, 0.0);
    }
    fft::fft(&mut hb, false);
    let mut out = vec![0f32; x.len() + h.len()];
    let mut pos = 0;
    while pos < x.len() {
        let n = block.min(x.len() - pos);
        let mut xb = vec![C::default(); N];
        for i in 0..n {
            xb[i] = C::new(x[pos + i], 0.0);
        }
        fft::fft(&mut xb, false);
        for i in 0..N {
            xb[i] = xb[i].mul(hb[i]);
        }
        fft::fft(&mut xb, true);
        for i in 0..N {
            if pos + i < out.len() {
                out[pos + i] += xb[i].re;
            }
        }
        pos += n;
    }
    out.truncate(x.len());
    out
}

/// A sentence, as the machine says it: phrases split at punctuation, each
/// planned and performed, gaps between them by the punctuation and the
/// arousal, the whole finished and peak-normalised.
pub fn render(m: &Model, lex: Option<&Lexicon>, text: &str, d: Dial) -> Vec<f32> {
    let k = Knobs::from(d);
    let mut rng = Rng::new(hash(text));
    let lift_every = if d.valence >= 0.3 { 4 } else if d.valence > -0.3 { 6 } else { 0 };
    let mut out: Vec<f32> = Vec::new();
    let mut count = 0;
    let mut cur = String::new();
    let mut parts: Vec<String> = Vec::new();
    for c in text.chars() {
        cur.push(c);
        if matches!(c, '.' | ',' | '!' | '?' | ';' | ':') {
            parts.push(core::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    for p in parts {
        let p = p.trim();
        if p.is_empty() || !p.chars().any(|c| c.is_alphabetic()) {
            continue;
        }
        count += 1;
        let end = p.chars().last().unwrap_or('.');
        let final_up = end == '?' || (lift_every > 0 && count % lift_every == 0);
        let seg = phrase(m, lex, p, final_up, &k, d, &mut rng);
        out.extend_from_slice(&seg);
        let gap = match end {
            ',' => 0.45,
            ';' | ':' => 0.6,
            _ => 0.85,
        } * (1.0 - 0.3 * d.arousal);
        out.extend(core::iter::repeat(0f32).take((gap * SR as f32) as usize));
    }
    let mut y = finish(out, 0.16, 1.0, &mut rng);
    let peak = y.iter().map(|v| v.abs()).fold(0f32, f32::max) + 1e-9;
    for v in y.iter_mut() {
        *v *= 0.9 / peak;
    }
    y
}

fn hash(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let on = Knobs::from(Dial { mask: 1.0, valence: 0.0, arousal: 0.0 });
    let slip = Knobs::from(Dial { mask: 0.5, valence: 0.0, arousal: 0.0 });
    let off = Knobs::from(Dial { mask: 0.0, valence: 0.0, arousal: 0.0 });
    let mut imp = vec![0f32; 100];
    imp[10] = 1.0;
    let h = [0.5f32, 0.25, 0.125];
    let c = convolve(&imp, &h);
    vec![
        ("mask on is the machine: one note, timing 0.3", on.mach == 1.0 && (on.mech - 0.3).abs() < 1e-6 && on.base == 9.5),
        ("half slipped is not a machine at all", slip.mach == 0.0 && (slip.mech - 0.4).abs() < 1e-6),
        ("the mask gone is the lowest, widest voice", off.base == 4.0 && off.accent > on.accent && off.fall > on.fall),
        ("FFT convolution of an impulse is the kernel, placed", (c[10] - 0.5).abs() < 1e-4 && (c[12] - 0.125).abs() < 1e-4 && c[9].abs() < 1e-4),
    ]
}
