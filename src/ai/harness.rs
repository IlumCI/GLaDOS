//! The harness: how the system asks the model to act.
//!
//! A conventional agent harness is a loop around a text API -- render the tool
//! list into a prompt, get a string back, parse it, dispatch, append the result
//! and go round again. Every step of that exists because the model is a guest
//! and text is the only channel through the wall.
//!
//! There is no wall here. The pieces this needs -- the applet table, the
//! sampler, the KV cache, the scheduler -- are all in the same address space at
//! ring 0, so the harness can reach into any of them:
//!
//!   * The action space is not described to the model and then hoped for. It
//!     is compiled into a grammar from the live `sysbox::APPLETS` table and
//!     enforced inside the sampler, so an invalid choice cannot be produced.
//!   * Permission is enforced at the same place. Excluding mutating applets
//!     from the grammar is not a check performed on the model's answer; it
//!     removes those answers from the space of reachable outputs.
//!   * Nothing is parsed. `Cursor::finished` returns the index of the applet
//!     that was decoded, because the decode was structured by construction.
//!
//! What is deliberately *not* here yet is the loop. Feeding results back and
//! iterating is only worth building against a model that can follow an
//! instruction; stories260K cannot, and never will. What can be established
//! now -- and is, by `selftest` -- is that the mechanism is sound: that the
//! grammar admits exactly the applet names, that permission cannot leak, and
//! that a decode always terminates on a real applet. Those properties are
//! independent of which model is loaded, which is the point of testing them
//! against one that has no idea what it is doing.

use super::constrain::{step_bound, Alphabet, Cursor, Grammar, MAX_LEADING_SPACES};
use super::{sample, tokenizer, with_engine};
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::sysbox;
use crate::{kprintln, sync::Racy};
use alloc::string::String;
use alloc::vec::Vec;

/// Decoded token bytes, built once from the loaded tokenizer.
///
/// Cached because it is derived from the vocabulary and never changes, and
/// rebuilding it per decode would mean re-deriving every token's text on every
/// step of every choice.
static ALPHABET: Racy<Option<Alphabet>> = Racy::new(None);

fn with_alphabet<R>(f: impl FnOnce(&Alphabet) -> R) -> Option<R> {
    unsafe {
        if ALPHABET.get().is_none() {
            let built = with_engine(|e| Alphabet::new(&e.tok))?;
            *ALPHABET.get() = Some(built);
        }
        ALPHABET.get().as_ref().map(f)
    }
}

/// Which applets the model is allowed to reach for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    /// Read-only applets only. The mutating ones are absent from the grammar,
    /// so they are not merely refused -- there is no token sequence that names
    /// them.
    ReadOnly,
    Full,
}

impl Trust {
    fn admits(&self, a: &sysbox::Applet) -> bool {
        match self {
            Trust::ReadOnly => !a.mutates,
            Trust::Full => true,
        }
    }
}

fn grammar_for(trust: Trust) -> (Grammar, Vec<&'static str>) {
    let names: Vec<&'static str> = sysbox::APPLETS
        .iter()
        .filter(|a| trust.admits(a))
        .map(|a| a.name)
        .collect();
    (Grammar::new(names.iter().copied()), names)
}

/// Render the action space as the model sees it.
///
/// Still text, and still the weakest link: this is the one place the model has
/// to *understand* rather than merely be constrained. It is what a competent
/// model would read to make a sensible choice, and what stories260K will
/// ignore entirely.
fn prompt_for(task: &str, names: &[&'static str]) -> String {
    let mut s = String::from("Tools:");
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push(' ');
        s.push_str(n);
    }
    s.push_str(". Task: ");
    s.push_str(task);
    s.push_str(". Tool:");
    s
}

pub struct Choice {
    pub applet: &'static str,
    /// Whether it can change persistent content -- carried through so the
    /// caller never has to look it up again and get it wrong.
    pub mutates: bool,
    pub steps: usize,
}

/// Have the model pick one applet.
///
/// Returns `None` if there is no engine. It cannot return an invalid name:
/// the only way out of the decode loop is `Cursor::finished`, which yields an
/// index into the very list the grammar was built from.
pub fn choose(task: &str, trust: Trust, temperature: f32) -> Option<Choice> {
    let (grammar, names) = grammar_for(trust);
    if grammar.is_empty() {
        return None;
    }
    let bound = step_bound(&grammar);

    with_alphabet(|alphabet| {
        with_engine(|e| {
            let prompt = prompt_for(task, &names);
            let tokens = e.tok.encode(&prompt, true, false);

            // Prefill: run the prompt through so the KV cache holds it. The
            // logits from all but the last are discarded -- we are not
            // predicting the prompt, only conditioning on it.
            let mut pos = 0usize;
            let limit = e.model.cfg.seq_len;
            for &t in tokens.iter() {
                if pos >= limit {
                    break;
                }
                e.model.forward(&mut e.state, t, pos);
                pos += 1;
            }

            let mut cursor = Cursor::new(&grammar);
            let mut steps = 0usize;
            let mut idle = 0usize;

            while steps < bound && idle <= MAX_LEADING_SPACES && pos < limit {
                let candidates = cursor.candidates(alphabet);
                // Empty means no token in the vocabulary can extend the string
                // toward any alternative. With a byte-fallback vocabulary this
                // should be impossible, so it is a bug rather than a refusal.
                let next =
                    sample::sample_among(&e.state.logits, &candidates, temperature, 0.0, &mut e.rng)?;

                if cursor.push(alphabet, next) {
                    steps += 1;
                } else {
                    idle += 1;
                }

                if let Some(idx) = cursor.finished() {
                    let name = names[idx];
                    let mutates = sysbox::APPLETS
                        .iter()
                        .find(|a| a.name == name)
                        .map(|a| a.mutates)
                        .unwrap_or(false);
                    return Some(Choice { applet: name, mutates, steps });
                }

                // Advance the model by the token it just committed to.
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
            }
            None
        })?
    })?
}

pub fn report(task: &str, trust: Trust, temperature: f32) {
    let (_, names) = grammar_for(trust);
    console::set_color(YELLOW);
    kprintln!("[harness]");
    console::set_color(LTGRAY);
    kprintln!(
        "  {} of {} applets reachable ({})",
        names.len(),
        sysbox::APPLETS.len(),
        match trust {
            Trust::ReadOnly => "read-only",
            Trust::Full => "full",
        }
    );

    match choose(task, trust, temperature) {
        None => {
            console::set_color(LTRED);
            // Two different failures used to print the same line, which made a
            // decode that ran out of budget look like a missing model.
            if with_engine(|_| ()).is_none() {
                kprintln!("  no model loaded");
            } else {
                kprintln!("  decode did not settle on an applet");
            }
            console::set_color(LTGRAY);
        }
        Some(c) => {
            console::set_color(LTCYAN);
            kprintln!("  chose '{}' in {} constrained steps", c.applet, c.steps);
            console::set_color(LTGRAY);
            if c.mutates {
                console::set_color(YELLOW);
                kprintln!("  that applet mutates content");
                console::set_color(LTGRAY);
            }
        }
    }
}

// --- selftest -----------------------------------------------------------

fn check(what: &str, pass: bool) -> bool {
    if pass {
        console::set_color(LTGREEN);
        kprintln!("  ok   {}", what);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL {}", what);
    }
    console::set_color(LTGRAY);
    pass
}

/// Verify the constraint mechanism itself.
///
/// The model's *judgement* cannot be tested here -- stories260K has none to
/// test. What can be tested is that no sequence of sampling outcomes escapes
/// the grammar, and that is the property the whole design leans on.
pub fn selftest() -> bool {
    let mut ok = true;

    let (grammar, names) = grammar_for(Trust::ReadOnly);
    let mutating: Vec<&str> = sysbox::APPLETS
        .iter()
        .filter(|a| a.mutates)
        .map(|a| a.name)
        .collect();

    ok &= check(
        "read-only grammar excludes every mutating applet",
        mutating.iter().all(|m| !names.contains(m)),
    );
    ok &= check(
        "read-only grammar keeps every read-only applet",
        sysbox::APPLETS
            .iter()
            .filter(|a| !a.mutates)
            .all(|a| names.contains(&a.name)),
    );

    let Some(engine_present) = with_engine(|_| true) else {
        console::set_color(YELLOW);
        kprintln!("  (no model -- decode checks skipped)");
        console::set_color(LTGRAY);
        return ok;
    };
    let _ = engine_present;

    // Drive the decoder from random logits rather than from the model. If a
    // valid name still comes out every time under arbitrary preferences, then
    // validity is a property of the grammar and not of the model happening to
    // behave. This is the check that would catch a prefix bug like `snap`
    // shadowing `snaps`.
    let escaped = with_alphabet(|alphabet| {
        with_engine(|e| {
            let bound = step_bound(&grammar);
            let mut bad = 0u32;
            let mut reached_snaps = false;
            for _ in 0..200 {
                let mut cursor = Cursor::new(&grammar);
                let mut steps = 0;
                let mut idle = 0;
                let mut settled = None;
                while steps < bound && idle <= MAX_LEADING_SPACES {
                    let candidates = cursor.candidates(alphabet);
                    if candidates.is_empty() {
                        break;
                    }
                    // Uniformly random preference over the whole vocabulary.
                    let mut logits = alloc::vec![0.0f32; alphabet.len()];
                    for l in logits.iter_mut() {
                        *l = e.rng.next_f32();
                    }
                    let Some(next) =
                        sample::sample_among(&logits, &candidates, 1.0, 0.0, &mut e.rng)
                    else {
                        break;
                    };
                    if cursor.push(alphabet, next) {
                        steps += 1;
                    } else {
                        idle += 1;
                    }
                    if let Some(i) = cursor.finished() {
                        settled = Some(i);
                        break;
                    }
                }
                match settled {
                    Some(i) if i < names.len() => {
                        if names[i] == "snaps" {
                            reached_snaps = true;
                        }
                    }
                    _ => bad += 1,
                }
            }
            (bad, reached_snaps)
        })
    })
    .flatten();

    match escaped {
        None => ok &= check("constrained decode always lands on a real applet", false),
        Some((bad, reached_snaps)) => {
            ok &= check(
                "200 random constrained decodes all produced a real applet",
                bad == 0,
            );
            // `snap` is a proper prefix of `snaps`. Without the terminator in
            // the grammar, `snaps` is unreachable and this never fires.
            ok &= check(
                "a name that extends another is still reachable ('snaps')",
                reached_snaps,
            );
        }
    }

    let _ = tokenizer::BOS;
    let _ = WHITE;
    ok
}
