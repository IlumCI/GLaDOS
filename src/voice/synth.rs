//! A formant synthesiser, in the line of Klatt (1980).
//!
//! The voice is a source and a filter. The source is a glottal pulse train
//! at the pitch, plus noise for breath and for frication; the filter is five
//! two-pole resonators in cascade, one per formant, and one more in parallel
//! that shapes noise into an s or a sh. Everything is set per 5 ms frame and
//! interpolated, because a formant that jumps clicks.
//!
//! **The GLaDOS treatment is pitch, not distortion.** What makes that voice
//! hers is a contour quantised to semitones -- the autotuned, stepwise,
//! almost-singing line -- over a slow fall, with a short comb filter for the
//! metallic edge. All three are deterministic functions of the text and of
//! the affect handed in. Built entirely from this file's own arithmetic: no
//! recording of anybody went into it, and none ever will.

use super::phone::{self, Manner, Phone};
use super::text::Token;
use crate::ai::tensor::{cosf, expf, powf};
use alloc::vec;
use alloc::vec::Vec;

pub const RATE: u32 = 48_000;
/// Frication against voicing. A bandpass of noise at unit gain is about this
/// much quieter than a vowel through five resonators; set by ear off the
/// spectrogram, so an s sits under a vowel and not over it.
const FRIC_GAIN: f32 = 6.0;
const FRAME: usize = (RATE / 200) as usize; // 5 ms

#[derive(Clone, Copy, Default)]
struct Frame {
    f: [f32; 3],
    /// Voicing, aspiration and frication amplitudes, 0..1.
    av: f32,
    ah: f32,
    af: f32,
    fric: (f32, f32),
    /// Pitch multiplier before quantisation: stress lifts it.
    lift: f32,
    nasal: bool,
}

/// How the affect bends the voice. Bounded so a mood can make it livelier
/// or flatter and never unintelligible.
#[derive(Clone, Copy)]
pub struct Style {
    /// Base pitch, Hz.
    pub f0: f32,
    /// Speaking rate: 1.0 is normal, higher is faster.
    pub rate: f32,
    /// How far stress lifts pitch: the "range".
    pub range: f32,
    /// Comb depth, the metal: 0 none.
    pub metal: f32,
}

impl Style {
    pub const GLADOS: Style = Style { f0: 196.0, rate: 1.0, range: 0.12, metal: 0.35 };

    /// The house style, bent by valence and arousal (thousandths).
    pub fn from_affect(valence: i32, arousal: i32) -> Style {
        let a = arousal.clamp(0, 1000) as f32 / 1000.0;
        let v = valence.clamp(-1000, 1000) as f32 / 1000.0;
        let mut s = Style::GLADOS;
        s.rate = 1.0 + 0.18 * a;
        s.f0 = 196.0 * (1.0 + 0.06 * a + 0.03 * v);
        s.range = 0.12 * (1.0 + 0.6 * v.max(0.0)) * (1.0 - 0.5 * (-v).max(0.0));
        s
    }
}

fn frames_for(tokens: &[Token], style: Style) -> Vec<Frame> {
    let mut out: Vec<Frame> = Vec::new();
    let push = |out: &mut Vec<Frame>, ms: f32, f: Frame| {
        let n = ((ms / 5.0) as usize).max(1);
        for _ in 0..n {
            out.push(f);
        }
    };
    for t in tokens {
        match t {
            Token::Pause(ms) => push(&mut out, *ms as f32 / style.rate, Frame::default()),
            Token::Word(w) | Token::Letters(w) => {
                let phones: Vec<Phone> = match t {
                    Token::Letters(_) => w.chars().flat_map(super::g2p::letter).collect(),
                    _ => super::g2p::word(w).0,
                };
                for ph in phones {
                    let tg = phone::target(ph.p);
                    let mut ms = tg.ms / style.rate;
                    let stressed = ph.stress == 1 || ph.stress == 2;
                    if tg.manner == Manner::Vowel && !stressed {
                        ms *= 0.6;
                    }
                    let lift = if tg.manner == Manner::Vowel && ph.stress == 1 { 1.0 + style.range } else { 1.0 };
                    let base = Frame { f: tg.f, lift, ..Frame::default() };
                    match tg.manner {
                        Manner::Vowel => {
                            // A diphthong glides from start to end target.
                            let n = ((ms / 5.0) as usize).max(1);
                            for k in 0..n {
                                let x = k as f32 / n as f32;
                                let mut f = base;
                                for j in 0..3 {
                                    f.f[j] = tg.f[j] + (tg.f_end[j] - tg.f[j]) * x;
                                }
                                f.av = 1.0;
                                out.push(f);
                            }
                        }
                        Manner::Approximant => push(&mut out, ms, Frame { av: 0.7, ..base }),
                        Manner::Nasal => push(&mut out, ms, Frame { av: 0.45, nasal: true, ..base }),
                        Manner::Fricative => push(
                            &mut out,
                            ms,
                            Frame { av: if tg.voiced { 0.35 } else { 0.0 }, af: tg.fric.2, fric: (tg.fric.0, tg.fric.1), ..base },
                        ),
                        Manner::Aspirate => push(&mut out, ms, Frame { ah: 0.5, ..base }),
                        Manner::Stop | Manner::Affricate => {
                            // Closure, then release.
                            let closed = if tg.manner == Manner::Stop { 0.7 } else { 0.4 };
                            let bar = if tg.voiced { 0.12 } else { 0.0 };
                            push(&mut out, ms * closed, Frame { av: bar, ..base });
                            let burst = Frame {
                                af: tg.fric.2,
                                fric: (tg.fric.0, tg.fric.1),
                                av: if tg.voiced { 0.3 } else { 0.0 },
                                ah: if tg.voiced { 0.0 } else { 0.25 },
                                ..base
                            };
                            push(&mut out, ms * (1.0 - closed), burst);
                        }
                    }
                }
            }
        }
    }
    // Coarticulation: formants glide across about 25 ms rather than jumping;
    // amplitudes across 15 ms, which is enough to stop a click.
    let raw = out.clone();
    for i in 0..out.len() {
        let mut acc = [0.0f32; 3];
        let mut n = 0.0;
        for j in i.saturating_sub(2)..(i + 3).min(raw.len()) {
            for k in 0..3 {
                acc[k] += raw[j].f[k];
            }
            n += 1.0;
        }
        for k in 0..3 {
            out[i].f[k] = acc[k] / n;
        }
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(raw.len());
        let span = (hi - lo) as f32;
        out[i].av = raw[lo..hi].iter().map(|f| f.av).sum::<f32>() / span;
        out[i].ah = raw[lo..hi].iter().map(|f| f.ah).sum::<f32>() / span;
    }
    out
}

/// A two-pole resonator, Klatt's: y = a x + b y1 + c y2.
#[derive(Clone, Copy, Default)]
struct Res {
    a: f32,
    b: f32,
    c: f32,
    y1: f32,
    y2: f32,
}

impl Res {
    fn set(&mut self, f: f32, bw: f32) {
        let t = 1.0 / RATE as f32;
        let pi = core::f32::consts::PI;
        self.c = -expf(-2.0 * pi * bw * t);
        self.b = 2.0 * expf(-pi * bw * t) * cosf(2.0 * pi * f * t);
        self.a = 1.0 - self.b - self.c;
    }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let y = self.a * x + self.b * self.y1 + self.c * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// A biquad, for the filters a Klatt resonator is the wrong shape for: its
/// gain is one at DC by construction, so used as a "bandpass" on noise it
/// passes everything below the band as well -- which on the first recording
/// was every s and t arriving as full-band hiss louder than the vowels.
#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    /// RBJ's band-pass, 0 dB at the centre.
    fn bandpass(&mut self, f: f32, bw: f32) {
        let w = 2.0 * core::f32::consts::PI * f / RATE as f32;
        let q = (f / bw).max(0.3);
        let alpha = crate::ai::tensor::sinf(w) / (2.0 * q);
        let a0 = 1.0 + alpha;
        self.b0 = alpha / a0;
        self.b1 = 0.0;
        self.b2 = -alpha / a0;
        self.a1 = -2.0 * cosf(w) / a0;
        self.a2 = (1.0 - alpha) / a0;
    }
    /// RBJ's low-pass, Butterworth Q.
    fn lowpass(&mut self, f: f32) {
        let w = 2.0 * core::f32::consts::PI * f / RATE as f32;
        let alpha = crate::ai::tensor::sinf(w) / (2.0 * core::f32::consts::FRAC_1_SQRT_2);
        let c = cosf(w);
        let a0 = 1.0 + alpha;
        self.b0 = (1.0 - c) / 2.0 / a0;
        self.b1 = (1.0 - c) / a0;
        self.b2 = (1.0 - c) / 2.0 / a0;
        self.a1 = -2.0 * c / a0;
        self.a2 = (1.0 - alpha) / a0;
    }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Snap a pitch to the nearest semitone above A2 (110 Hz). The step from
/// one note to the next is the sound.
fn quantise(hz: f32) -> f32 {
    let st = 12.0 * crate::ai::tensor::lnf(hz / 110.0) / core::f32::consts::LN_2;
    let n = (st + 0.5) as i32 as f32;
    110.0 * powf(2.0, n / 12.0)
}

/// Say a run of tokens. Mono, `RATE` Hz, signed 16-bit.
pub fn render(tokens: &[Token], style: Style) -> Vec<i16> {
    let frames = frames_for(tokens, style);
    let total = frames.len() * FRAME;
    let mut out = vec![0.0f32; total];
    let mut casc = [Res::default(); 5];
    let mut fricr = Biquad::default();
    let mut phase = 0.0f32;
    let mut seed: u32 = 0x1234_5678;
    let mut noise = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    };
    let fs = RATE as f32;
    let n = frames.len().max(1) as f32;
    for (fi, fr) in frames.iter().enumerate() {
        let scale = phone::FEMALE;
        let bw = if fr.nasal { [150.0, 200.0, 250.0] } else { [70.0, 100.0, 140.0] };
        for k in 0..3 {
            casc[k].set((fr.f[k] * scale).min(fs * 0.45), bw[k]);
        }
        casc[3].set(3800.0, 250.0);
        casc[4].set(4600.0, 300.0);
        if fr.af > 0.0 {
            fricr.bandpass(fr.fric.0.min(fs * 0.4), fr.fric.1);
        }
        // A slow fall across the utterance, the lift of a stressed vowel,
        // and then the snap to a semitone.
        let f0 = quantise(style.f0 * (1.0 - 0.12 * fi as f32 / n) * fr.lift);
        for s in 0..FRAME {
            // Rosenberg's glottal pulse: open over 40% of the period, close
            // over 16%, shut for the rest.
            phase += f0 / fs;
            if phase >= 1.0 {
                phase -= 1.0;
            }
            let g = if phase < 0.4 {
                0.5 * (1.0 - cosf(core::f32::consts::PI * phase / 0.4))
            } else if phase < 0.56 {
                cosf(core::f32::consts::PI * (phase - 0.4) / 0.32)
            } else {
                0.0
            };
            let src = fr.av * g + fr.ah * noise() * 0.3;
            let mut y = src;
            for r in casc.iter_mut() {
                y = r.run(y);
            }
            let fric = if fr.af > 0.0 { fricr.run(noise()) * fr.af * FRIC_GAIN } else { 0.0 };
            out[fi * FRAME + s] = y + fric;
        }
    }
    // Radiation at the lips: a first difference, which is the tilt that makes
    // a cascade sound like a mouth rather than a pipe.
    let mut prev = 0.0;
    for x in out.iter_mut() {
        let d = *x - 0.97 * prev;
        prev = *x;
        *x = d;
    }
    // The metal: a 1 ms comb.
    let delay = (RATE / 1000) as usize;
    if style.metal > 0.0 {
        for i in (delay..out.len()).rev() {
            out[i] += style.metal * out[i - delay];
        }
    }
    // Nothing above 7 kHz: a formant voice has nothing to say up there but
    // noise, and the radiation filter tilts the noise upward.
    let mut lp = Biquad::default();
    lp.lowpass(7000.0);
    let mut lp2 = lp;
    for x in out.iter_mut() {
        *x = lp2.run(lp.run(*x));
    }
    // Loudness, set by the voiced sound and not by the loudest burst: the
    // 99.5th percentile of magnitude, so a stop's click cannot hold the
    // vowels down. What rises past that is clipped softly.
    let mut mags: Vec<f32> = out.iter().map(|x| x.abs()).collect();
    mags.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let level = mags.get(mags.len() * 995 / 1000).copied().unwrap_or(1.0).max(1e-6);
    let g = 0.45 / level;
    for x in out.iter_mut() {
        let y = *x * g;
        // A cubic soft clip: linear-ish below, flat at +-1 above.
        let c = y.clamp(-1.0, 1.0);
        *x = 1.5 * (c - c * c * c / 3.0);
    }
    let g = 32767.0 * 0.9;
    out.iter().map(|x| (x * g) as i16).collect()
}

pub fn checks() -> Vec<(&'static str, bool)> {
    use super::text::tokens;
    let a = render(&tokens("hello"), Style::GLADOS);
    let b = render(&tokens("hello"), Style::GLADOS);
    let quiet = render(&tokens("."), Style::GLADOS);
    let peak = a.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0);
    let longer = render(&tokens("hello there machine"), Style::GLADOS).len() > a.len();
    let fast = render(&tokens("hello there machine"), Style { rate: 1.3, ..Style::GLADOS }).len()
        < render(&tokens("hello there machine"), Style::GLADOS).len();
    alloc::vec![
        ("a word renders to sound", a.len() > (RATE / 10) as usize && peak > 5000),
        ("the same text renders to the same samples", a == b),
        ("a lone full stop is silence, not a sound", quiet.iter().all(|&x| x == 0) || quiet.is_empty()),
        ("more words take longer", longer),
        ("a faster rate takes less time", fast),
        ("pitch snaps to semitones", (quantise(196.0) - 196.0).abs() < 0.5 && (quantise(200.0) - 196.0).abs() < 0.5),
    ]
}
