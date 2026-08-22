//! The agent loop: goal -> constrained applet choice -> validated arguments
//! -> executed tool -> observation fed back -> repeat.
//!
//! The harness module deliberately stops short of this: it proves that the
//! grammar admits exactly the applet names and that permission cannot leak,
//! against models too small to follow an instruction. What was missing is the
//! iteration -- feed the result back and go round again. This module adds
//! that edge, under three constraints inherited from the harness:
//!
//!   * The action space is compiled into a grammar from the live applet
//!     table, plus one pseudo-applet, `done`, which ends the episode. An
//!     invalid choice is unreachable rather than refused.
//!   * Under `Trust::ReadOnly` the mutating applets are absent from the
//!     grammar entirely. Permission and output validity stay the same piece
//!     of code.
//!   * Nothing is parsed on the way in. The decode ends on a real name; only
//!     the *arguments* are free text, and they are shape-checked against the
//!     applet's declared usage before dispatch. A bad argument becomes an
//!     observation the model can react to, never a fault.
//!
//! What this deliberately is not yet: asynchronous. `run` executes on the
//! calling task with a step budget, which keeps a scripted episode
//! deterministic -- there is no second task whose interleaving with the shell
//! has to be reasoned about while the loop itself is still being judged.

use super::constrain::{step_bound, Cursor, Grammar, MAX_LEADING_SPACES};
use super::harness::{self, Trust};
use super::{sample, with_engine};
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use crate::sysbox;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// The pseudo-applet that ends an episode. It travels in the same grammar as
/// the real names, so stopping is a choice the model makes under the same
/// constraints as acting -- not a special token smuggled past the sampler.
const DONE: &str = "done";

/// Longest observation fed back per step. The transcript stays bounded no
/// matter how chatty an applet is; `tree` on a deep namespace would eat the
/// context window otherwise.
const OBS_CLIP: usize = 240;

/// Longest argument string accepted from free generation.
const ARGS_CLIP_BYTES: usize = 96;

/// Tokens the model may spend on arguments before being cut off.
const ARGS_TOKEN_BUDGET: usize = 16;

struct Step {
    /// Rendered action, e.g. `ls /sys` -- what the transcript and the next
    /// prompt both show.
    action: String,
    ok: bool,
    observation: String,
}

fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return String::from(s);
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::from(&s[..end]);
    out.push_str("...");
    out
}

/// The names the grammar is built from live in `harness::admitted`; the only
/// thing this module adds is the sentinel below.

/// What the model sees each step. Kept short on purpose: history is the last
/// few steps with clipped observations, because a prompt that grows with the
/// episode would spend the context window retelling it and leave nothing for
/// thinking.
fn prompt_for(goal: &str, steps: &[Step], names: &[&str]) -> String {
    let mut s = String::from("Goal: ");
    s.push_str(goal);
    s.push('\n');
    if steps.is_empty() {
        s.push_str("(nothing done yet)\n");
    } else {
        for st in steps.iter().rev().take(3).rev() {
            s.push_str("- ");
            s.push_str(&st.action);
            s.push_str(" -> ");
            let mark = if st.ok { "" } else { "(rejected) " };
            s.push_str(mark);
            s.push_str(&clip(&st.observation, 100));
            s.push('\n');
        }
    }
    s.push_str("Tools:");
    for (i, n) in names.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push(' ');
        s.push_str(n);
    }
    s.push_str(". Next tool:");
    s
}

/// One decision: prefill the step prompt, decode an applet name under the
/// grammar, then continue generating its arguments greedily until a newline,
/// end-of-sequence, or budget. Returns None when there is no engine, no
/// alphabet, or the decode never settles -- all reported by the caller.
///
/// The name tokens are advanced into the cache as they are committed, which
/// is what lets the arguments be a genuine continuation of the chosen tool
/// rather than a fresh guess.
fn propose(goal: &str, steps: &[Step], trust: Trust) -> Option<(String, String)> {
    let mut names = harness::admitted(trust);
    names.push(DONE);
    let grammar = Grammar::new(names.iter().copied());
    let bound = step_bound(&grammar);

    // The closure returns Option; with_engine and with_alphabet each add one
    // layer around it. The match at the bottom removes exactly one -- the
    // alphabet/engine layer -- leaving the decode's own Option as the result.
    let wrapped = harness::with_alphabet(|alphabet| {
        with_engine(|e| {
            let prompt = prompt_for(goal, steps, &names);
            let tokens = e.tok.encode(&prompt, true, false);

            let mut pos = 0usize;
            let limit = e.model.cfg.seq_len;
            for &t in tokens.iter() {
                if pos >= limit {
                    break;
                }
                e.model.forward(&mut e.state, t, pos);
                pos += 1;
            }

            // Constrained choice, mirroring the harness decode exactly.
            let mut cursor = Cursor::new(&grammar);
            let mut used = 0usize;
            let mut idle = 0usize;
            let picked = loop {
                if used >= bound || idle > MAX_LEADING_SPACES || pos >= limit {
                    break None;
                }
                let candidates = cursor.candidates(alphabet);
                let next =
                    sample::sample_among(&e.state.logits, &candidates, 0.0, 0.0, &mut e.rng)?;
                if cursor.push(alphabet, next) {
                    used += 1;
                } else {
                    idle += 1;
                }
                if let Some(idx) = cursor.finished() {
                    break Some(idx);
                }
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
            };

            let idx = picked?;
            let name = names[idx];
            if name == DONE {
                harness::invalidate_conversation(e);
                return Some((String::from(name), String::new()));
            }

            // Free-text arguments, greedy, continuing the cache the decode
            // left behind so they are a continuation of the chosen tool.
            let mut raw: Vec<u8> = Vec::new();
            let eos = e.tok.eos();
            for _ in 0..ARGS_TOKEN_BUDGET {
                if pos >= limit || raw.len() >= ARGS_CLIP_BYTES {
                    break;
                }
                let vocab = e.tok.vocab_size();
                let all: Vec<u32> = (0..vocab as u32).collect();
                let next =
                    sample::sample_among(&e.state.logits, &all, 0.0, 0.0, &mut e.rng)?;
                if next == eos {
                    break;
                }
                let piece = e.tok.token_bytes(next).to_vec();
                let nl = piece.iter().position(|&b| b == b'\n');
                raw.extend_from_slice(&piece[..nl.unwrap_or(piece.len())]);
                let done = nl.is_some() || raw.len() >= ARGS_CLIP_BYTES;
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
                if done {
                    break;
                }
            }

            harness::invalidate_conversation(e);
            Some((String::from(name), String::from_utf8_lossy(&raw).into_owned()))
        })
    });
    // Two wrappers, two layers: with_engine and with_alphabet each add one
    // around the closure's own Option. The inner layer is the decode failing
    // to settle; the outer ones mean no engine or no alphabet, which run()
    // has already excluded before calling here. Either way the caller gets
    // one flat Option.
    match wrapped {
        Some(Some(decode)) => decode,
        _ => None,
    }
}

/// Run one episode to completion: bounded steps, live printing, transcript
/// written to /ai/episodes/. Synchronous by design -- see the module header.
pub fn run(goal: &str, trust: Trust, max_steps: usize) {
    console::set_color(YELLOW);
    kprintln!("[agent]");
    console::set_color(LTGRAY);
    kprintln!(
        "  goal: {}   trust: {}   budget: {} steps",
        goal,
        match trust {
            Trust::ReadOnly => "read-only",
            Trust::Full => "full",
        },
        max_steps
    );

    if with_engine(|_| ()).is_none() {
        console::set_color(LTRED);
        kprintln!("  no model loaded");
        console::set_color(LTGRAY);
        return;
    }

    let mut steps: Vec<Step> = Vec::new();
    let mut outcome = "step budget reached";

    for i in 0..max_steps {
        let Some((name, args)) = propose(goal, &steps, trust) else {
            outcome = "decode did not settle";
            break;
        };

        if name == DONE {
            outcome = "model called done";
            console::set_color(LTGREEN);
            kprintln!("  {}. done", i + 1);
            console::set_color(LTGRAY);
            break;
        }

        // Shape-check before dispatch. Rejection is an observation, not an
        // error: the model gets to read why and choose differently.
        let checked = sysbox::check_args(&name, &args);
        let (ok, observation) = match checked {
            Err(why) => (false, format!("invalid arguments: {}", why)),
            Ok(()) => {
                console::begin_capture();
                let ran = sysbox::dispatch(&name, &args);
                let mut obs = console::end_capture().unwrap_or_default();
                if !ran && obs.is_empty() {
                    obs = String::from("(applet did not run)");
                }
                (true, obs)
            }
        };

        let action = if args.is_empty() {
            String::from(name)
        } else {
            format!("{} {}", name, args)
        };
        console::set_color(if ok { LTCYAN } else { LTRED });
        kprintln!("  {}. {}", i + 1, action);
        console::set_color(LTGRAY);
        for line in observation.lines().take(4) {
            kprintln!("     | {}", line);
        }

        steps.push(Step {
            action,
            ok,
            observation: clip(&observation, OBS_CLIP),
        });
    }

    console::set_color(YELLOW);
    kprintln!("  [agent] {} after {} step(s)", outcome, steps.len());
    console::set_color(LTGRAY);

    // Transcript into the namespace. Content-addressed like everything else:
    // an episode can be hashed, diffed against its siblings, and snapshotted
    // without any of it being special-cased.
    let report = render(goal, outcome, &steps);
    let idx = sysbox::children("/ai/episodes").len() + 1;
    let path = format!("/ai/episodes/{:04}.txt", idx);
    if sysbox::write_text(&path, &report) {
        kprintln!("  transcript at {}", path);
    }
}

fn render(goal: &str, outcome: &str, steps: &[Step]) -> String {
    let mut s = format!("goal: {}\noutcome: {}\nsteps: {}\n\n", goal, outcome, steps.len());
    for (i, st) in steps.iter().enumerate() {
        s.push_str(&format!("[{}] {}\n", i + 1, st.action));
        s.push_str(&format!("    ok: {}\n", st.ok));
        for line in st.observation.lines() {
            s.push_str(&format!("    | {}\n", line));
        }
    }
    s
}
