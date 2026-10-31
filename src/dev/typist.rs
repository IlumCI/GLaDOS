//! The typist: text in, keystrokes out, into whatever window has focus.
//!
//! **Through the keyboard and nothing else**, which is the reason it works for
//! any program rather than for one. Each character becomes the set-1 scancodes
//! a real keyboard would send -- Shift down, the key, its release, Shift up --
//! and goes in through `kbd::inject_scancode`, the path `linux feed` already
//! drives. From there it is a keystroke like any other: the decoder records
//! it, the evdev ring carries it, and Skywalker hands it to the focused
//! Wayland client as `wl_keyboard.key`. LibreOffice cannot tell it from a
//! person typing, and neither can foot, a GTK text field or a Qt editor.
//!
//! The alternative was an office-specific route -- LibreOffice's UNO bridge,
//! a script inside the guest -- and it would have bought formatting at the
//! price of working for exactly one program. Typing is what every program
//! with a text field already accepts.
//!
//! **One character per timer tick**, so a hundred a second: faster than the
//! model writes, slow enough that a client reading its socket between frames
//! never sees a burst it has to coalesce. Paced on a task of its own, so the
//! model and the typist overlap -- the model is several tokens ahead and the
//! text appears as it is written.
//!
//! The map is the decoder's own tables read backwards (`kbd::scancode_for`),
//! so this can only type what the decoder would have produced from the same
//! keys: US layout, the same layout the keymap Skywalker sends describes.
//! Anything outside ASCII is folded to its nearest ASCII spelling first (curly
//! quotes, dashes, an ellipsis) and dropped if it has none, which is said in
//! the count `office` prints rather than discovered as a missing word.

use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::sync::Spin;

/// Characters waiting to be typed. Bounded so a runaway producer costs a
/// refusal rather than the heap.
static QUEUE: Spin<VecDeque<u8>> = Spin::new(VecDeque::new());
const QUEUE_CAP: usize = 64 * 1024;

static STARTED: AtomicBool = AtomicBool::new(false);
/// Characters typed since boot, and ones that could not be typed at all.
static TYPED: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// Set by `stop`, which empties the queue and lets a half-typed word end.
static STOP: AtomicBool = AtomicBool::new(false);
static TASK: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Queue text to be typed. Answers how many characters were accepted.
pub fn push(text: &str) -> usize {
    ensure_started();
    let mut q = QUEUE.lock_irq();
    let mut n = 0;
    for c in text.chars() {
        for b in fold(c).bytes() {
            if q.len() >= QUEUE_CAP {
                DROPPED.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            q.push_back(b);
            n += 1;
        }
        if fold(c).is_empty() && !c.is_control() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
    drop(q);
    let t = TASK.load(Ordering::Relaxed);
    if t != usize::MAX {
        crate::task::wake(t);
    }
    n
}

/// Whether anything is still waiting to be typed.
pub fn busy() -> bool {
    !QUEUE.lock_irq().is_empty()
}

/// Characters still waiting.
pub fn pending() -> usize {
    QUEUE.lock_irq().len()
}

/// Throw away whatever has not been typed yet.
pub fn stop() {
    STOP.store(true, Ordering::Release);
    QUEUE.lock_irq().clear();
}

/// (typed, dropped) since boot.
pub fn counts() -> (u64, u64) {
    (TYPED.load(Ordering::Relaxed), DROPPED.load(Ordering::Relaxed))
}

/// The nearest ASCII spelling of a character, or nothing.
///
/// A model writes curly quotes and dashes out of habit, and a keyboard has no
/// key for them; typing nothing would join two words, typing a box would be
/// worse. Carriage returns go, because `\r\n` would be two Enters.
pub fn fold(c: char) -> &'static str {
    match c {
        '\n' => "\n",
        '\t' => "\t",
        '\r' => "",
        ' '..='~' => {
            // Every printable ASCII character maps to itself; this table is
            // how a `char` becomes a `&'static str` without allocating.
            const ASCII: &str = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
            let i = c as usize - 0x20;
            &ASCII[i..i + 1]
        }
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{2032}' => "'",
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => "\"",
        '\u{2013}' | '\u{2014}' | '\u{2212}' => "-",
        '\u{2026}' => "...",
        '\u{2022}' | '\u{00B7}' => "-",
        // Currency a US keyboard has no key for, spelt as its code.
        '\u{20AC}' => "EUR ",
        '\u{00A3}' => "GBP ",
        '\u{00A0}' | '\u{2009}' | '\u{202F}' => " ",
        _ => "",
    }
}

fn ensure_started() {
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    // Core 0 and nowhere else: the keyboard interrupt is delivered there, and
    // with interrupts off around each injection that makes a person typing at
    // the same moment and this task strictly one after the other.
    match crate::task::spawn_on("typist", run, 0) {
        Some(t) => TASK.store(t, Ordering::Relaxed),
        None => STARTED.store(false, Ordering::Release),
    }
}

fn run() {
    loop {
        let next = QUEUE.lock_irq().pop_front();
        let Some(b) = next else {
            STOP.store(false, Ordering::Release);
            // Woken by `push`; the deadline is only a backstop.
            let until = crate::dev::lapic::ticks() + crate::TIMER_HZ as u64;
            crate::task::sleep_until(until);
            continue;
        };
        keep_pace();
        if type_byte(b) {
            TYPED.fetch_add(1, Ordering::Relaxed);
            // A program reads its keys when it next enters the kernel, and
            // one asleep in `poll` would otherwise not look until its nap
            // ends. Woken now, it takes the key while it is fresh.
            crate::linux::syscall::wake_nappers();
        } else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
        // One character a tick. See the module doc for why not a burst.
        let until = crate::dev::lapic::ticks() + 1;
        crate::task::sleep_until(until);
    }
}

/// Wait while the program being typed into is behind.
///
/// One character a tick is fast enough to outrun a program that is busy: it
/// takes its keys only when it next enters the kernel, and Calc on a loaded
/// machine went whole seconds without doing so. The keyboard ring holds about
/// two hundred keys and drops the oldest past that, and they were lost from
/// the front of what was typed.
/// The typist waits instead, while the backlog is more than a few keys and
/// somebody is reading at all: nobody having read for ten seconds means
/// nothing is going to, and the wait would only stop the typing.
fn keep_pace() {
    const BEHIND: u64 = 64;
    let hz = crate::TIMER_HZ as u64;
    let start = crate::dev::lapic::ticks();
    loop {
        let (unread, at) = crate::linux::input::unread();
        let now = crate::dev::lapic::ticks();
        if unread <= BEHIND || now.saturating_sub(at) > 10 * hz || now - start > 30 * hz || STOP.load(Ordering::Acquire) {
            return;
        }
        crate::linux::syscall::wake_nappers();
        crate::task::sleep_until(now + hz / 20);
    }
}

/// One character as a real keyboard would send it.
fn type_byte(b: u8) -> bool {
    let Some((code, shift)) = crate::dev::kbd::scancode_for(b) else { return false };
    crate::cpu::without_interrupts(|| {
        if shift {
            crate::dev::kbd::inject_scancode(crate::dev::kbd::SC_LSHIFT);
        }
        crate::dev::kbd::inject_scancode(code);
        crate::dev::kbd::inject_scancode(code | 0x80);
        if shift {
            crate::dev::kbd::inject_scancode(crate::dev::kbd::SC_LSHIFT | 0x80);
        }
    });
    true
}

/// What `diag typist` asks, with no window and nothing typed.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    use crate::dev::kbd::scancode_for;
    let mut out = alloc::vec::Vec::new();
    out.push(("a lowercase letter is its key, unshifted", scancode_for(b'a') == Some((0x1E, false))));
    out.push(("a capital is the same key with Shift", scancode_for(b'A') == Some((0x1E, true))));
    out.push(("Enter is the Enter key", scancode_for(b'\n') == Some((0x1C, false))));
    out.push(("a space is the space bar", scancode_for(b' ') == Some((0x39, false))));
    out.push(("punctuation that needs Shift says so", scancode_for(b'!') == Some((0x02, true))));
    out.push((
        "every printable ASCII character has a key",
        (0x20u8..0x7F).all(|b| scancode_for(b).is_some()),
    ));
    out.push(("a curly quote folds to a straight one", fold('\u{2019}') == "'"));
    out.push(("an em dash folds to a hyphen", fold('\u{2014}') == "-"));
    out.push(("an ellipsis folds to three dots", fold('\u{2026}') == "..."));
    out.push(("a carriage return types nothing, so CRLF is one Enter", fold('\r').is_empty()));
    out.push(("a character with no spelling is dropped, not boxed", fold('\u{4E2D}').is_empty()));
    out
}
