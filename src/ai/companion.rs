//! A conversation that survives the turn, and the reboot.
//!
//! `ask` was always a cold start. It framed one question, generated an answer,
//! and threw the state away; the next question knew nothing about the last, so
//! "what about the second one?" was unanswerable and always had been. That is
//! not a small gap for a machine meant to be lived with -- it is the whole
//! difference between a command and a companion.
//!
//! Everything needed was already here and simply never joined up:
//!
//! * `GenOpts::resume` continues from the live KV cache rather than rebuilding
//!   one, so the conversation so far is *already in the machine*.
//! * `ctx_save`/`ctx_load` write that cache to the namespace and read it back.
//! * `set_window` gives attention sinks plus a recent window, which is what
//!   lets a conversation outlive the trained context instead of hitting a wall.
//!
//! ### Why the cache, and not a transcript
//!
//! Every chat program re-sends its whole history on every turn, because the
//! model is behind an API and the cache belongs to somebody else. Here the
//! cache is ours and it stays. A turn costs the tokens of that turn, not the
//! tokens of the conversation -- so the tenth exchange is as cheap as the
//! first, and at 1.7 s/token that is the difference between usable and not.
//!
//! What it costs is that the conversation is only as durable as the cache. The
//! shape of the machine decides the shape of the feature, which is the whole
//! argument for owning the machine.

use alloc::string::String;
use alloc::vec::Vec;

/// Where the live conversation is parked so a reboot does not end it.
const LIVE: &str = "live";

/// Plain text the operator writes about themselves, injected into the system
/// turn. A file rather than a command because it is theirs to edit, and
/// because a companion that cannot be told it is wrong about you is worse
/// than one that knows nothing.
pub const ABOUT: &str = "/ai/about";

/// Turns exchanged since this conversation opened.
static TURNS: crate::sync::Racy<usize> = crate::sync::Racy::new(0);

/// What was said, as text, for anything that has to *show* the conversation.
///
/// The cache is the conversation and this is not a second copy of it -- the
/// model is never fed from here, and a turn still costs the tokens of that
/// turn. It exists because a window cannot read a KV cache: `slot_of` gives
/// positions and attention keys, not sentences, and there is no route from an
/// int8 cache back to the words that filled it.
///
/// So it is written where the words already exist, on the way past. The two
/// have exactly one point of contact and it is `reset`, which clears both --
/// a transcript outliving the cache it describes would show a conversation the
/// model has already forgotten, which is worse than showing none.
///
/// `true` is the operator's turn.
static LOG: crate::sync::Racy<Vec<(bool, String)>> = crate::sync::Racy::new(Vec::new());

/// How many turns are kept. Twenty-four is about six screens of the Ask
/// window at its densest, and the cache holds fewer than that at 512 slots
/// anyway.
const LOG_KEEP: usize = 24;

/// The conversation as text, oldest first. A clone of a small static, so it is
/// a free read for a paint pass -- see `glance`'s tiers.
pub fn log_snapshot() -> Vec<(bool, String)> {
    unsafe { (*LOG.get()).clone() }
}

fn log_push(mine: bool, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let log = unsafe { &mut *LOG.get() };
    log.push((mine, String::from(text)));
    if log.len() > LOG_KEEP {
        log.remove(0);
    }
}

pub fn turns() -> usize {
    unsafe { *TURNS.get() }
}

/// Forget the conversation and start over.
pub fn reset() {
    unsafe { *TURNS.get() = 0 };
    unsafe { (*LOG.get()).clear() };
    unsafe { *SYS_LEN.get() = 0 };
    super::with_engine(|e| {
        e.pos = 0;
        e.last_token = 0;
    });
}

/// The system turn: who this is, where it lives, and what it can reach.
///
/// The applet list is read from the live table rather than written out here,
/// for the same reason the decoding grammar is: a hardcoded list goes stale
/// silently, and the failure is a model confidently offering a tool that does
/// not exist.
fn system_turn() -> String {
    let mut s = String::from("<|im_start|>system\n");
    s.push_str(
        "You are the resident model of GLaDOS, a kernel written from scratch in \
         Rust that you run inside. You are not a service being called over a \
         network; you are part of this machine and you persist between its \
         reboots.\n\n",
    );

    s.push_str("Tools you can ask the system to run:\n");
    let mut n = 0;
    for a in crate::sysbox::APPLETS.iter() {
        if n > 0 {
            s.push_str(", ");
        }
        s.push_str(a.name);
        n += 1;
    }
    s.push_str(".\n");

    // What the operator has said about themselves. Absent is normal and says
    // nothing; an empty file is the same as no file.
    if let Some(about) = crate::sysbox::read_blob(ABOUT) {
        let text = String::from_utf8_lossy(&about);
        let text = text.trim();
        if !text.is_empty() {
            s.push_str("\nAbout the person you are talking to:\n");
            s.push_str(text);
            s.push('\n');
        }
    }

    s.push_str("<|im_end|>\n");
    s
}

/// Sinks to keep when nothing better is known.
///
/// Four, from the StreamingLLM result: a handful of early tokens absorb a
/// disproportionate share of attention, and dropping them is what makes a
/// naively-windowed model fall apart rather than merely forget.
const MIN_SINKS: usize = 4;

/// How many positions the open conversation's system turn occupies.
///
/// Zero when there is no conversation, or when one was revived and this was
/// not alongside it.
static SYS_LEN: crate::sync::Racy<usize> = crate::sync::Racy::new(0);

/// Where the length is parked, beside the cache it describes.
const LIVE_SYS: &str = "live.sys";

/// Sinks are *pinned slots*, so pinning the system turn is a matter of
/// counting it.
///
/// `slot_of` returns `j` unchanged for `j < n_sinks` and wraps everything
/// after into the ring. The sinks are therefore not merely privileged, they
/// are never written over -- so if the sink count is the system turn's length
/// rather than four, the system turn is what survives eviction, in its
/// original positions and with its original RoPE angles.
///
/// Four sinks buy stability and nothing else: the model keeps generating
/// coherently while the text that told it who it is has scrolled away. Pinning
/// the whole turn keeps the instructions, the applet list and what it knows
/// about the operator for as long as the conversation runs.
///
/// The cost is honest and worth stating: a pinned slot never recycles, so the
/// recent window is shorter by exactly the system turn. At 512 with a ~120
/// token system turn that is a fifth of the cache; at 8192 it is noise.
/// Clamped to a third of the trained length, because a system turn large
/// enough to crowd out the conversation is worse than one that scrolls.
fn sink_count(trained: usize) -> usize {
    let want = unsafe { *SYS_LEN.get() };
    let ceiling = (trained / 3).max(MIN_SINKS);
    want.clamp(MIN_SINKS, ceiling)
}

/// Start evicting this many positions before the wall.
const MARGIN: usize = 64;

/// Turn the cache into a ring before it fills, so the conversation does not
/// simply stop.
///
/// **Why this is safe to do partway through, and only here.** Unwindowed, a
/// position lives at the slot with its own number. Windowed, it lives at
/// `sinks + (abs - sinks) % ring`. Those two agree exactly while
/// `abs - sinks < ring` -- so every entry already written is already in the
/// slot the windowed scheme will look for, provided the ring has not yet had
/// cause to wrap. Enabling it *before* the cache fills is precisely that case.
///
/// Do it later and every existing entry is at the wrong address: the model
/// would read a real key from the wrong position and stay perfectly fluent
/// while doing it, which is this codebase's worst failure shape.
fn widen_if_near_the_wall() {
    let Some((pos, trained, already)) = super::with_engine(|e| {
        (e.pos, e.model.cfg.seq_len, e.model.cfg.streams())
    }) else {
        return;
    };
    if already || pos + MARGIN < trained {
        return;
    }
    // The largest ring the trained length allows, so eviction starts as late
    // as possible and the conversation keeps as much of itself as it can.
    let sinks = sink_count(trained);
    let ring = trained.saturating_sub(sinks + 1);
    if ring == 0 || pos > sinks + ring {
        return;
    }
    super::set_window(sinks, ring);
    crate::kprintln!(
        "  (reached {} of {} -- oldest turns now scroll; the first {} positions are pinned)",
        pos,
        trained,
        sinks
    );
    if sinks <= MIN_SINKS {
        // Said out loud, because the difference is what the model still knows
        // about itself twenty turns from now.
        crate::kprintln!("  (the system turn is not pinned -- it will scroll out like any other)");
    }
}

/// One turn of conversation. The first opens it; the rest continue it.
/// One turn of conversation. The first opens it; the rest continue it.
///
/// Returns the position reached, so a caller can tell an opening turn from a
/// continuing one without asking twice.
pub fn turn(message: &str, opts: &super::GenOpts) -> usize {
    let opening = turns() == 0 || super::with_engine(|e| e.pos).unwrap_or(0) == 0;
    if !opening {
        widen_if_near_the_wall();
    }

    let mut prompt = String::new();
    if opening {
        let sys = system_turn();
        // Counted the way `generate` will encode it -- same BOS, same
        // tokenizer -- so the sink count is the span the system turn actually
        // occupies and not an estimate of it. An estimate that ran short would
        // pin part of a turn and leave the rest to scroll, which is worse than
        // pinning none of it.
        let n = super::with_engine(|e| e.tok.encode(&sys, true, false).len()).unwrap_or(0);
        unsafe { *SYS_LEN.get() = n };
        prompt.push_str(&sys);
    } else {
        // Close the assistant's previous turn before opening the user's. The
        // cache ends mid-turn -- the model stopped generating, it did not emit
        // a terminator -- so without this the next user message reads as a
        // continuation of the assistant's own sentence.
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(message);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");

    // Same reasoning as `chat`: a Qwen3 opens `<think>` unprompted and a
    // truncated reasoning block reads as a broken model.
    if !opts.think && super::has_think_token() {
        prompt.push_str("<think>\n\n</think>\n\n");
    }

    let framed = super::GenOpts {
        bos: opening,
        echo_prompt: false,
        // The point of the whole file: continue from what is already cached.
        resume: !opening,
        ..*opts
    };
    // Recorded around the generation rather than after it, so a turn that
    // runs out of budget still leaves the question in the transcript. A
    // window showing an answer with nothing above it is the worse failure.
    log_push(true, message);
    super::echo_begin();
    super::generate(&prompt, &framed);
    if let Some(said) = super::echo_end() {
        log_push(false, &said);
    }

    unsafe { *TURNS.get() += 1 };
    super::with_engine(|e| e.pos).unwrap_or(0)
}

/// How far a parked conversation will actually reach.
pub enum Parked {
    /// Written, and it will survive a reboot once the tree is snapshotted.
    Durable,
    /// Written, but only into memory: there is no store mounted, so it ends
    /// when the machine is switched off.
    Volatile,
    Failed,
}

/// Frame something the machine thought of by itself as a turn in the
/// conversation.
///
/// The resident mind already generated into this KV cache -- it passes
/// `resume: true` and always has. What it did not do is *say whose turn it
/// was*: raw text resumed mid-assistant-turn, so a thought the machine had on
/// its own spliced into the middle of its last sentence to the operator, and
/// the next question read as a continuation of that. Fluent, and wrong about
/// who said what.
///
/// This closes the open turn and opens one that is labelled. The operator sees
/// where a thought came from, and so does the model on every later turn, which
/// is the part that actually matters: a conversation where the machine cannot
/// tell its own words from yours is not a conversation.
pub fn interject_frame(kind: &str) -> String {
    let mut s = String::new();
    if turns() > 0 {
        s.push_str("<|im_end|>
");
    }
    s.push_str("<|im_start|>system
");
    s.push_str(kind);
    s.push_str("<|im_end|>
<|im_start|>assistant
");
    s
}

/// Note that the machine has taken a turn, so the next operator turn closes
/// this one properly.
pub fn interjected() {
    unsafe { *TURNS.get() += 1 };
}

/// Park the conversation.
///
/// The distinction matters and the first version of this did not make it.
/// `sysbox::write_blob` puts a blob in the working tree, which is *memory* --
/// durability comes from a snapshot into the store, and there may be no store
/// mounted at all. So writing succeeded, this answered true, and the
/// conversation was gone at the next boot having reported that it was saved.
///
/// A companion that says it will remember and then does not is worse than one
/// that admits it cannot, so the two cases are now different answers.
pub fn park() -> Parked {
    if super::ctx_save(LIVE).is_none() {
        return Parked::Failed;
    }
    // Beside the cache rather than inside it: the context format has no field
    // for this and adding one would make every context written before today
    // unreadable, for eight bytes.
    let n = unsafe { *SYS_LEN.get() };
    let mut path = String::from(super::CTX_DIR);
    path.push('/');
    path.push_str(LIVE_SYS);
    let _ = crate::sysbox::write_blob(&path, Vec::from(&n.to_le_bytes()[..]));

    if crate::store::mounted() {
        Parked::Durable
    } else {
        Parked::Volatile
    }
}

/// Pick it back up. Answers the position, or `None` if there was nothing to
/// resume or it does not fit this model.
///
/// Called at boot, where a missing context is the ordinary case and not worth
/// a word: a machine that has never been spoken to should not announce it.
pub fn revive() -> Option<usize> {
    let p = super::ctx_load(LIVE)?;
    if p == 0 {
        return None;
    }
    unsafe { *TURNS.get() = 1 };

    let mut path = String::from(super::CTX_DIR);
    path.push('/');
    path.push_str(LIVE_SYS);
    let span = crate::sysbox::read_blob(&path)
        .filter(|b| b.len() == 8)
        .map(|b| {
            let mut a = [0u8; 8];
            a.copy_from_slice(&b);
            usize::from_le_bytes(a)
        })
        .unwrap_or(0);
    // A revived conversation whose span was not parked alongside it pins the
    // default four. Silent, because it is only worse than the alternative once
    // the cache fills, and `widen_if_near_the_wall` says so when it does.
    unsafe { *SYS_LEN.get() = span };

    Some(p)
}

/// Positions the open conversation's system turn occupies, and how many of
/// them would be pinned if the cache filled now.
///
/// Reportable so the pin can be checked without driving a conversation into
/// eviction to find out. The first version of this was verified by reading
/// fifteen turns of a 135M model's output, which is a slow way to learn a
/// number the machine already knows.
pub fn pinning(trained: usize) -> (usize, usize) {
    (unsafe { *SYS_LEN.get() }, sink_count(trained))
}
