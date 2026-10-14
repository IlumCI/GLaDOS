//! The voice: text in, speech out of the HD Audio controller.
//!
//! `text` takes what the model wrote and removes everything a voice must not
//! read aloud; `g2p` turns words into phones, CMUdict first and rules after;
//! `synth` is a formant synthesiser with the house style -- semitone-stepped
//! pitch, a slow fall, a metallic comb -- bent by the brain's affect; and
//! this file is the queue and the task that plays sentences one after
//! another.
//!
//! **Built from nothing recorded.** The style is arithmetic in `synth.rs`.
//! No sample of any actor, and in particular none of the game this machine
//! is named after, went into it -- that is somebody's performance, under the
//! same rule that keeps game data out of this tree.
//!
//! A formant voice first, a neural one later (Kokoro, Apache-2.0, behind the
//! same `say`), as the 1.4.0 plan ordered: this one is small, deterministic,
//! and needs nothing from the boot volume.

pub mod fft;
pub mod g2p;
pub mod ipa;
pub mod kokoro;
pub mod neural;
pub mod ops;
pub mod phone;
pub mod style;
pub mod synth;
pub mod text;
pub mod vfile;

use crate::sync::Spin;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

static QUEUE: Spin<VecDeque<String>> = Spin::new(VecDeque::new());
/// On by default: the machine talks unless asked not to, and `voice off`
/// is kept in `/etc/voice` so it stays asked.
static ON: AtomicBool = AtomicBool::new(true);
static ECAM: AtomicU64 = AtomicU64::new(0);
static SPOKEN: AtomicUsize = AtomicUsize::new(0);
static SPEAKING: AtomicBool = AtomicBool::new(false);
/// The mask the last sentence was rendered with, in thousandths, for the
/// status line: what was heard, not what the brain reads now.
static LAST_MASK: AtomicUsize = AtomicUsize::new(1000);
/// How loud, 0 to 100. Forty by default rather than full scale: the first
/// sentence this machine ever spoke through the GF63's speakers was at 0 dB on
/// the codec and full scale in the samples, and the person in front of it
/// asked for volume control before anything else. Applied to the samples, so
/// it is the same on every codec and needs nothing the HDA path may lack.
static VOLUME: AtomicUsize = AtomicUsize::new(40);
/// Set from the keyboard's interrupt, written to `/etc/volume` by the voice
/// task: an interrupt handler may not write the namespace.
static VOLUME_DIRTY: AtomicBool = AtomicBool::new(false);

pub fn volume() -> usize {
    VOLUME.load(Ordering::Acquire)
}

/// Safe from an interrupt handler: two atomics, nothing else.
pub fn set_volume(v: usize) {
    VOLUME.store(v.min(100), Ordering::Release);
    VOLUME_DIRTY.store(true, Ordering::Release);
}

/// What mute put down, so pressing it again brings it back.
static BEFORE_MUTE: AtomicUsize = AtomicUsize::new(0);

/// The mute key: to silence, and back to where it was. Interrupt-safe.
pub fn toggle_mute() {
    let v = volume();
    if v > 0 {
        BEFORE_MUTE.store(v, Ordering::Release);
        set_volume(0);
    } else {
        let back = BEFORE_MUTE.load(Ordering::Acquire);
        set_volume(if back > 0 { back } else { 40 });
    }
}

/// Up or down by `step`, clamped, for the volume keys.
pub fn nudge_volume(step: isize) {
    let v = (volume() as isize + step).clamp(0, 100) as usize;
    set_volume(v);
}

/// The sample gain for a volume: squared, because loudness is heard on
/// something close to a log scale and a linear knob spends its top half on
/// differences nobody hears. 40 is 0.16, about -16 dB.
pub fn gain_of(v: usize) -> f32 {
    let x = v.min(100) as f32 / 100.0;
    x * x
}

/// Sentences waiting, at most. A reply longer than this is cut rather than
/// queued for minutes of speech nobody can stop.
const QUEUE_MAX: usize = 24;

pub fn init(ecam: Option<u64>, staged: Option<&'static [u8]>) {
    neural::init(staged);
    if let Some(b) = crate::sysbox::read_blob("/etc/voice") {
        ON.store(b.as_slice() != b"off", Ordering::Release);
    }
    if let Some(v) = crate::sysbox::read_blob("/etc/volume")
        .and_then(|b| core::str::from_utf8(&b).ok().and_then(|t| t.trim().parse::<usize>().ok()))
    {
        VOLUME.store(v.min(100), Ordering::Release);
    }
    if let Some(e) = ecam {
        ECAM.store(e, Ordering::Release);
    }
    let _ = crate::task::spawn("voice", voice_task);
}

pub fn on() -> bool {
    ON.load(Ordering::Acquire)
}

pub fn set_on(v: bool) {
    ON.store(v, Ordering::Release);
    crate::sysbox::write_blob("/etc/voice", if v { b"on".to_vec() } else { b"off".to_vec() });
    if !v {
        QUEUE.lock_irq().clear();
    }
}

/// Speak, whatever `voice on` says: the operator asked for this one.
pub fn say(text: &str) -> usize {
    let mut q = QUEUE.lock_irq();
    let mut n = 0;
    for s in text::sentences(text) {
        if q.len() >= QUEUE_MAX {
            break;
        }
        q.push_back(s);
        n += 1;
    }
    n
}

/// Speak what the machine said, if the voice is on.
pub fn speak_reply(text: &str) {
    if on() {
        say(text);
    }
}

/// Stop talking: drop what is queued and cut what is playing.
pub fn hush() {
    QUEUE.lock_irq().clear();
    let _ = crate::dev::hda::with(|h| h.stop());
}

pub fn status() -> String {
    let d = dial(crate::brain::now());
    alloc::format!(
        "voice {}, volume {}, {} sentence(s) waiting, {} spoken this boot{}; {}, mask {:.2} (valence {:+.2}, arousal {:.2})",
        if on() { "on -- replies are spoken" } else { "off" },
        volume(),
        QUEUE.lock_irq().len(),
        SPOKEN.load(Ordering::Acquire),
        if SPEAKING.load(Ordering::Acquire) { ", speaking now" } else { "" },
        if neural::model().is_some() { "neural" } else { "formant" },
        d.mask,
        d.valence,
        d.arousal
    ) + &alloc::format!("; the last sentence wore mask {:.2}", LAST_MASK.load(Ordering::Acquire) as f32 / 1000.0)
}

/// Render one sentence to playable PCM: 48 kHz stereo 16-bit, with a tail of
/// silence so the controller's loop back to the start lands in quiet.
pub fn render(sentence: &str) -> Vec<u8> {
    let a = crate::brain::now();
    // The neural voice when one is staged, the formant voice otherwise.
    // The mask is the brain's composure: whole while things go well, and
    // worn down by what goes badly. Rest restores it.
    let d = dial(a);
    LAST_MASK.store((d.mask * 1000.0) as usize, Ordering::Release);
    let mono: Vec<i16> = match neural::render(&text::strip(sentence), d) {
        Some(y) => to_48k(&y),
        None => {
            let mut style = synth::Style::from_affect(a.valence, a.arousal);
            if a.asleep {
                // Talking in its sleep, if anybody asks it to: slower and lower.
                style.rate *= 0.85;
                style.f0 *= 0.92;
            }
            synth::render(&text::tokens(sentence), style)
        }
    };
    let tail = (synth::RATE / 4) as usize;
    let mut pcm = Vec::with_capacity((mono.len() + tail) * 4);
    let g = gain_of(volume());
    for s in mono.iter().map(|&s| (s as f32 * g) as i16).chain(core::iter::repeat(0i16).take(tail)) {
        pcm.extend_from_slice(&s.to_le_bytes());
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    pcm
}

/// The mask from the affect: composed at a neutral or good valence,
/// slipping as it falls, gone at the bottom; asleep, it is whole.
pub fn dial(a: crate::brain::Affect) -> style::Dial {
    let valence = a.valence.clamp(-1000, 1000) as f32 / 1000.0;
    let arousal = a.arousal.clamp(0, 1000) as f32 / 1000.0;
    // whole at any valence that is not negative; a quarter below neutral
    // and it is slipping, most of the way down and it is gone
    let mask = if a.asleep { 1.0 } else { (1.0 + 1.2 * valence).clamp(0.0, 1.0) };
    style::Dial { mask, valence, arousal }
}

/// 24 kHz f32 to 48 kHz i16: a 2x half-band interpolation with a short
/// windowed sinc, which keeps the top of the band where linear
/// interpolation would dull it.
fn to_48k(y: &[f32]) -> Vec<i16> {
    const TAPS: [f32; 8] = [0.0049, -0.0217, 0.0689, -0.1766, 0.6245, 0.6245, -0.1766, 0.0689];
    let mut out = Vec::with_capacity(y.len() * 2);
    for i in 0..y.len() {
        out.push((y[i] * 32767.0).clamp(-32768.0, 32767.0) as i16);
        let mut v = 0f32;
        for (k, t) in TAPS.iter().enumerate() {
            let j = i as isize + k as isize - 3;
            if j >= 0 && (j as usize) < y.len() {
                v += t * y[j as usize];
            }
        }
        out.push((v * 32767.0).clamp(-32768.0, 32767.0) as i16);
    }
    out
}

fn voice_task() {
    loop {
        if VOLUME_DIRTY.swap(false, Ordering::AcqRel) {
            crate::sysbox::write_blob("/etc/volume", alloc::format!("{}", volume()).into_bytes());
        }
        let next = QUEUE.lock_irq().pop_front();
        let Some(s) = next else {
            crate::task::sleep_until(crate::dev::lapic::ticks() + 5);
            continue;
        };
        if crate::dev::hda::with(|_| ()).is_none() {
            let e = ECAM.load(Ordering::Acquire);
            if e == 0 || !crate::dev::hda::bring_up(e, None, false) {
                // No audio to speak through. Dropped, not retried forever.
                QUEUE.lock_irq().clear();
                continue;
            }
        }
        let pcm = render(&s);
        let fmt = crate::dev::hda::verb::format(synth::RATE, 16, 2).unwrap_or(0x11);
        let ms = (pcm.len() as u64 / 4) * 1000 / synth::RATE as u64;
        let started = crate::dev::hda::with(|h| h.play(&pcm, fmt).is_ok()).unwrap_or(false);
        if started {
            SPEAKING.store(true, Ordering::Release);
            // Stop a little before the loop would come round: the tail is a
            // quarter second of silence, so stopping inside it is clean.
            // In slices, so muting cuts the sentence that is playing rather
            // than waiting out what is left of it.
            let until = crate::dev::lapic::ticks() + (ms.saturating_sub(100)) / 10;
            while crate::dev::lapic::ticks() < until && volume() > 0 {
                crate::task::sleep_until((crate::dev::lapic::ticks() + 5).min(until));
            }
            let _ = crate::dev::hda::with(|h| h.stop());
            SPEAKING.store(false, Ordering::Release);
            SPOKEN.fetch_add(1, Ordering::AcqRel);
        }
    }
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut rows = Vec::new();
    rows.extend(text::checks());
    rows.extend(g2p::checks());
    rows.extend(synth::checks());
    rows.extend(vfile::checks());
    rows.extend(ops::checks());
    rows.extend(kokoro::checks());
    rows.extend(fft::checks());
    rows.extend(ipa::checks(neural::lexicon()));
    rows.extend(style::checks());
    let mut ok = true;
    for (what, good) in &rows {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
