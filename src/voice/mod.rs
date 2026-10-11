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

pub mod g2p;
pub mod phone;
pub mod synth;
pub mod text;

use crate::sync::Spin;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

static QUEUE: Spin<VecDeque<String>> = Spin::new(VecDeque::new());
static ON: AtomicBool = AtomicBool::new(false);
static ECAM: AtomicU64 = AtomicU64::new(0);
static SPOKEN: AtomicUsize = AtomicUsize::new(0);
static SPEAKING: AtomicBool = AtomicBool::new(false);
/// Sentences waiting, at most. A reply longer than this is cut rather than
/// queued for minutes of speech nobody can stop.
const QUEUE_MAX: usize = 24;

pub fn init(ecam: Option<u64>) {
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
    alloc::format!(
        "voice {}, {} sentence(s) waiting, {} spoken this boot{}",
        if on() { "on" } else { "off ('say' still speaks)" },
        QUEUE.lock_irq().len(),
        SPOKEN.load(Ordering::Acquire),
        if SPEAKING.load(Ordering::Acquire) { ", speaking now" } else { "" }
    )
}

/// Render one sentence to playable PCM: 48 kHz stereo 16-bit, with a tail of
/// silence so the controller's loop back to the start lands in quiet.
pub fn render(sentence: &str) -> Vec<u8> {
    let a = crate::brain::now();
    let style = synth::Style::from_affect(a.valence, a.arousal);
    let mono = synth::render(&text::tokens(sentence), style);
    let tail = (synth::RATE / 4) as usize;
    let mut pcm = Vec::with_capacity((mono.len() + tail) * 4);
    for s in mono.iter().copied().chain(core::iter::repeat(0i16).take(tail)) {
        pcm.extend_from_slice(&s.to_le_bytes());
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    pcm
}

fn voice_task() {
    loop {
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
            let until = crate::dev::lapic::ticks() + (ms.saturating_sub(100)) / 10;
            crate::task::sleep_until(until);
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
