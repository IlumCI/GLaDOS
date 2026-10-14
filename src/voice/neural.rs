//! The neural voice's slot, and the check that it is the same voice the
//! host renders.
//!
//! `init` parses the staged `GLADOSV1` and builds the model once; a file
//! that does not parse, or a tensor with the wrong shape, is reported by
//! name and the formant voice carries on. `verify` is the differential
//! check in `diag differ`'s shape: the host's reference dump holds the
//! input ids and every intermediate of one forward pass, and the kernel
//! runs the same ids through its own graph and prints the largest
//! disagreement per stage -- exact up to the waveform, and on the spectral
//! envelope after, because the source module's noise is each side's own.

use super::ipa::Lexicon;
use super::kokoro::{Model, Rng, Trace, FRAME};
use super::vfile::VoiceFile;
use crate::ai::tensor;
use crate::sync::Spin;
use alloc::string::String;
use alloc::vec::Vec;

static MODEL: Spin<Option<&'static Model>> = Spin::new(None);
static LEXICON: Spin<Option<&'static Lexicon>> = Spin::new(None);
static WHY: Spin<Option<String>> = Spin::new(None);

pub fn init(staged: Option<&'static [u8]>) {
    let Some(b) = staged else {
        return;
    };
    use crate::gfx::console::{self, LTGRAY, LTRED, YELLOW};
    console::set_color(YELLOW);
    crate::kprintln!("\n[voice]");
    console::set_color(LTGRAY);
    match VoiceFile::parse(b).map_err(String::from).and_then(|f| Model::load(&f).map(|m| (m, f.bytes("lexicon.us").and_then(Lexicon::parse)))) {
        Ok((m, lex)) => {
            crate::kprintln!(
                "  {} loaded: Kokoro-82M, {} MB in place, {}",
                crate::VOICE_PATH,
                b.len() / 1_000_000,
                match &lex {
                    Some(l) => alloc::format!("misaki lexicon of {} words", l.len()),
                    None => String::from("no lexicon (CMUdict will do)"),
                }
            );
            *MODEL.lock_irq() = Some(alloc::boxed::Box::leak(alloc::boxed::Box::new(m)));
            *LEXICON.lock_irq() = lex.map(|l| &*alloc::boxed::Box::leak(alloc::boxed::Box::new(l)));
        }
        Err(why) => {
            console::set_color(LTRED);
            crate::kprintln!("  {} refused: {} -- the formant voice speaks instead", crate::VOICE_PATH, why);
            console::set_color(LTGRAY);
            *WHY.lock_irq() = Some(why);
        }
    }
}

pub fn model() -> Option<&'static Model> {
    *MODEL.lock_irq()
}

pub fn lexicon() -> Option<&'static Lexicon> {
    *LEXICON.lock_irq()
}

/// A sentence through the neural voice and the house style, 24 kHz mono.
pub fn render(text: &str, d: super::style::Dial) -> Option<Vec<f32>> {
    let m = model()?;
    Some(super::style::render(m, lexicon(), text, d))
}

pub fn status() -> String {
    match (model(), WHY.lock_irq().clone()) {
        (Some(_), _) => String::from("neural voice: Kokoro-82M loaded"),
        (None, Some(why)) => alloc::format!("neural voice: refused at boot: {}", why),
        (None, None) => String::from("neural voice: no voice.bin was staged; the formant voice speaks"),
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = alloc::vec![0f32; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            o[c * rows + r] = x[r * cols + c];
        }
    }
    o
}

fn maxdiff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let mut m = 0f32;
    let mut scale = 0f32;
    for i in 0..n {
        m = m.max((a[i] - b[i]).abs());
        scale = scale.max(b[i].abs());
    }
    (m, scale)
}

/// Smoothed log power per 20 ms frame, for comparing two waveforms that
/// share an envelope and not a phase.
fn envelope_db(x: &[f32]) -> Vec<f32> {
    x.chunks(480).map(|c| 10.0 * log10(c.iter().map(|v| v * v).sum::<f32>() / c.len().max(1) as f32 + 1e-9)).collect()
}

fn now_us() -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        return crate::dev::lapic::ticks() * (1_000_000 / crate::TIMER_HZ as u64);
    }
    crate::time::rdtsc() / mhz
}

fn log10(x: f32) -> f32 {
    tensor::lnf(x) / core::f32::consts::LN_10
}

pub fn verify(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Some(m) = model() else {
        out.push(String::from("no neural voice is loaded"));
        return out;
    };
    let Some(blob) = crate::sysbox::read_blob(path) else {
        out.push(alloc::format!("{}: not found", path));
        return out;
    };
    // the parser views f32 in place and wants an 8-aligned base
    let bytes = blob.as_slice();
    let mut aligned: Vec<u64> = alloc::vec![0; (bytes.len() + 7) / 8];
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), aligned.as_mut_ptr() as *mut u8, bytes.len()) };
    let len = bytes.len();
    let leaked: &'static [u64] = Vec::leak(aligned);
    let b: &'static [u8] = unsafe { core::slice::from_raw_parts(leaked.as_ptr() as *const u8, len) };
    let f = match VoiceFile::parse(b) {
        Ok(f) => f,
        Err(why) => {
            out.push(alloc::format!("{}: {}", path, why));
            return out;
        }
    };
    let get = |n: &str| f.f32(n).ok();
    let (Some(ids), Some(r_bert), Some(r_d), Some(r_dur), Some(r_f0), Some(r_n), Some(r_ten), Some(r_audio)) =
        (get("ref.ids"), get("ref.bert"), get("ref.d"), get("ref.dur"), get("ref.f0"), get("ref.n"), get("ref.t_en"), get("ref.audio"))
    else {
        out.push(String::from("the dump is missing a stage"));
        return out;
    };
    let ids: Vec<u8> = ids.iter().map(|v| *v as u8).collect();
    let l = ids.len();
    let t0 = now_us();
    let fr = m.front(&ids);
    let t1 = now_us();
    let mut line = |what: &str, got: &[f32], want: &[f32]| {
        let (d, s) = maxdiff(got, want);
        out.push(alloc::format!("  {:8} max |diff| {:.3e} against values up to {:.3e}  {}", what, d, s, if d <= 1e-3 * s.max(1.0) { "ok" } else { "DIFFERS" }));
    };
    // the dump stores torch's (c, L) for d_en and t_en; d is already (L, 640)
    line("d", &fr.d, r_d);
    line("dur", &fr.dur, r_dur);
    line("t_en", &fr.t_en, &transpose(r_ten, 512, l));
    let _ = r_bert;
    let dur: Vec<usize> = r_dur.iter().map(|v| (tensor::roundf(*v) as usize).max(1)).collect();
    let mut rng = Rng::new(1);
    let mut trace = Trace { har: get("ref.har").map(|h| h.to_vec()), taps: Vec::new() };
    let exact = trace.har.is_some();
    let sp = m.synth_traced(&fr, &dur, &mut rng, Some(&mut trace));
    let t2 = now_us();
    line("f0", &sp.f0, r_f0);
    line("n", &sp.n, r_n);
    // the generator's intermediates, each stored by the host as (c, t)
    for (name, got) in &trace.taps {
        let key = alloc::format!("ref.{}", name);
        if let Some(want) = get(&key) {
            let c = match *name { "x_dec" => 512, "nr0" | "up0" | "nc0" => 256, "nr1" | "up1" | "postin" => 128, _ => 22 };
            let t = want.len() / c;
            line(name, got, &transpose(want, c, t));
        }
    }
    if exact {
        line("audio", &sp.audio, r_audio);
    }
    let (ea, eb) = (envelope_db(&sp.audio), envelope_db(r_audio));
    let n = ea.len().min(eb.len());
    let loud: Vec<usize> = (0..n).filter(|&i| eb[i] > eb.iter().cloned().fold(f32::MIN, f32::max) - 40.0).collect();
    let rms = tensor::sqrtf(loud.iter().map(|&i| (ea[i] - eb[i]) * (ea[i] - eb[i])).sum::<f32>() / loud.len().max(1) as f32);
    let worst = loud.iter().map(|&i| (ea[i] - eb[i]).abs()).fold(0f32, f32::max);
    out.push(alloc::format!(
        "  audio    {} samples against {}; envelope rms {:.2} dB, worst {:.1} dB over {} loud frames  {}",
        sp.audio.len(),
        r_audio.len(),
        rms,
        worst,
        loud.len(),
        if sp.audio.len() == r_audio.len() && rms < 2.0 { "ok" } else { "DIFFERS" }
    ));
    out.push(alloc::format!("  front {} ms, synth {} ms for {:.2} s of audio", (t1 - t0) / 1000, (t2 - t1) / 1000, sp.audio.len() as f32 / 24000.0));
    let _ = FRAME;
    out
}
