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

pub fn turns() -> usize {
    unsafe { *TURNS.get() }
}

/// Forget the conversation and start over.
pub fn reset() {
    unsafe { *TURNS.get() = 0 };
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

/// One turn of conversation. The first opens it; the rest continue it.
///
/// Returns the position reached, so a caller can tell an opening turn from a
/// continuing one without asking twice.
pub fn turn(message: &str, opts: &super::GenOpts) -> usize {
    let opening = turns() == 0 || super::with_engine(|e| e.pos).unwrap_or(0) == 0;

    let mut prompt = String::new();
    if opening {
        prompt.push_str(&system_turn());
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
    super::generate(&prompt, &framed);

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
    Some(p)
}
