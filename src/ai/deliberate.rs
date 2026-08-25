//! Three-tier deliberation: reflex, pulse, deliberation.
//!
//! Fast inference separated from expensive reasoning, with confidence-based
//! exit at every tier. The frozen backbone cannot grow an adaptive-computation
//! head -- that would be training -- but adaptive computation is a property of
//! the *architecture around* a model, and this is that architecture. The
//! measured basis for each tier:
//!
//!   * **Reflex** -- the router with all three cores agreeing: 90.3% right,
//!     microseconds. Answer and exit.
//!   * **Pulse** -- one constrained decode: ~33-40% on this corpus, one
//!     prefill plus a short walk. The default.
//!   * **Deliberation** -- fork the state K ways, sample K candidate choices
//!     at temperature, rank every candidate (the pulse's own included) by the
//!     probe's class scores, keep the survivor. The globular post-mortem
//!     measured the oracle here at +3.3pp over greedy and showed logprob
//!     cannot spend it; the probe -- 54.7% alone, and the same head the
//!     reflex tier acts on -- is the selector that can.
//!
//! The state contract with the caller: on return, the engine holds the
//! *chosen* branch's context -- prompt plus the chosen name's tokens -- so
//! argument generation continues exactly where the choice left off. A
//! reflex exit leaves the prompt context untouched.

use super::constrain::{step_bound, Alphabet, Cursor, Grammar, MAX_LEADING_SPACES};
use super::harness::{self, Trust};
use super::sample;
use super::with_engine;
use alloc::string::String;
use alloc::vec::Vec;

/// How the choice was made. Printed into the episode so the operator can see
/// how much of the budget a request spent.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Router, three cores agreeing.
    Reflex,
    /// One constrained decode.
    Pulse,
    /// Forked candidates ranked by the probe.
    Deliberate,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::Reflex => "reflex",
            Tier::Pulse => "pulse",
            Tier::Deliberate => "deliberate",
        }
    }
}

pub struct Decision {
    pub applet: String,
    pub tier: Tier,
}

/// One fork of the mind: the state plus the position that gives its caches
/// meaning. `Engine::pos` lives outside `State`, so a fork carries both or
/// restores neither.
struct Fork {
    state: super::model::State,
    pos: usize,
    last_token: usize,
}

fn capture(e: &super::Engine) -> Fork {
    Fork {
        state: e.state.fork(),
        pos: e.pos,
        last_token: e.last_token,
    }
}

fn restore(e: &mut super::Engine, f: &Fork) {
    e.state = f.state.clone();
    e.pos = f.pos;
    e.last_token = f.last_token;
}

/// Shared constrained walk. `temperature == 0` is the greedy pulse; above
/// zero it is a sampled exploration used by the forked candidates. Advances
/// the engine state as it commits tokens, exactly like the harness decode it
/// descends from.
fn decode(
    e: &mut super::Engine,
    alphabet: &Alphabet,
    grammar: &Grammar,
    rng: &mut sample::Rng,
    temperature: f32,
) -> Option<(usize, usize)> {
    let bound = step_bound(grammar);
    let mut cursor = Cursor::new(grammar);
    let mut used = 0usize;
    let mut idle = 0usize;
    let limit = e.model.cfg.seq_len;
    loop {
        if used >= bound || idle > MAX_LEADING_SPACES || e.pos >= limit {
            return None;
        }
        let candidates = cursor.candidates(alphabet);
        let next = sample::sample_among(
            &e.state.logits,
            &candidates,
            temperature,
            if temperature > 0.0 { 1.0 } else { 0.0 },
            rng,
        )?;
        if cursor.push(alphabet, next) {
            used += 1;
        } else {
            idle += 1;
        }
        if let Some(idx) = cursor.finished() {
            return Some((idx, used));
        }
        e.model.forward(&mut e.state, next, e.pos);
        e.pos += 1;
    }
}

/// Probe scores for one goal, as (name, score), best first. The measured
/// selector: 54.7% alone, and the same head the reflex tier acts on. Without
/// a fitted router this returns None -- an unfitted head's ranking is a
/// guess with a number attached, and the deliberation tier would rather be
/// honest than random.
fn probe_rank(goal: &str, trust: Trust) -> Option<Vec<(String, f32)>> {
    if !harness::ensure_router() {
        return None;
    }
    with_engine(|e| {
        let feats = harness::feature_for(e, goal)?;
        let scores = e.probe.as_ref()?.scores(&feats);
        let admitted = harness::admitted(trust);
        let mut out: Vec<(String, f32)> = (0..e.head.len())
            .filter_map(|i| {
                let name = e.head.name(i);
                admitted
                    .contains(&name)
                    .then(|| (String::from(name), scores[i]))
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        Some(out)
    })
    .flatten()
}

/// Decide how to act on one goal. See the module header for the tiers.
/// `prompt_of` rebuilds the step prompt -- the episode owns that context;
/// the tiers own only the choice.
pub fn decide(
    goal: &str,
    prompt_of: &dyn Fn() -> String,
    trust: Trust,
    forks: usize,
) -> Option<Decision> {
    // ---- reflex ----
    if harness::ensure_router() {
        if let Some((choice, verdict)) = harness::route_verdict(goal, trust) {
            if verdict.agreement >= 3 && harness::admitted(trust).contains(&choice.applet) {
                return Some(Decision {
                    applet: String::from(choice.applet),
                    tier: Tier::Reflex,
                });
            }
        }
    }

    let mut names = harness::admitted(trust);
    names.push(super::agent::DONE);
    let grammar = Grammar::new(names.iter().copied());

    let wrapped = harness::with_alphabet(|alphabet| {
        with_engine(|e| {
            let prompt = prompt_of();
            let tokens = e.tok.encode(&prompt, true, false);
            // `e.pos` must track the cache or the decode below would start
            // writing at the position from before the prompt, overwriting
            // keys it is about to attend to. It also carries into argument
            // generation, which continues from the chosen branch's context.
            e.pos = e.model.prefill(&mut e.state, &tokens, e.pos);
            let mut rng = e.rng.clone();

            // ---- pulse ----
            let Some((idx, _)) = decode(e, alphabet, &grammar, &mut rng, 0.0) else {
                harness::invalidate_conversation(e);
                return None;
            };
            let pulse = String::from(names[idx]);
            if pulse == super::agent::DONE {
                harness::invalidate_conversation(e);
                return Some(Decision { applet: pulse, tier: Tier::Pulse });
            }
            if forks == 0 {
                harness::invalidate_conversation(e);
                return Some(Decision { applet: pulse, tier: Tier::Pulse });
            }

            // ---- deliberation ----
            let Some(rank) = probe_rank(goal, trust) else {
                harness::invalidate_conversation(e);
                return Some(Decision { applet: pulse, tier: Tier::Pulse });
            };
            let score_of = |name: &str| -> f32 {
                rank.iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, s)| *s)
                    .unwrap_or(f32::NEG_INFINITY)
            };

            let snap = capture(e);
            let mut best_name = pulse.clone();
            let mut best_tier = Tier::Pulse;
            let mut best_score = score_of(&pulse);
            for _ in 0..forks {
                restore(e, &snap);
                if let Some((cand_idx, _)) = decode(e, alphabet, &grammar, &mut rng, 0.9) {
                    let cand = String::from(names[cand_idx]);
                    if cand == super::agent::DONE {
                        continue;
                    }
                    let s = score_of(&cand);
                    if s > best_score {
                        best_name = cand;
                        best_score = s;
                        best_tier = Tier::Deliberate;
                        // Keep the winning branch's context: the caller's
                        // argument generation continues from here.
                        continue;
                    }
                }
                restore(e, &snap);
            }
            if best_tier == Tier::Pulse {
                // The pulse's own branch is the winner; its context is the
                // one the greedy walk left behind.
                restore(e, &snap);
            }
            // Note: the chosen branch's KV carries its name tokens, which is
            // what argument generation wants. Invalidate only on the paths
            // that leave nothing behind.
            if best_name == super::agent::DONE {
                harness::invalidate_conversation(e);
            }
            Some(Decision { applet: best_name, tier: best_tier })
        })
    });
    // Two wrappers, two layers: with_engine and with_alphabet each add one
    // around the closure's own Option. The inner layer is the decode failing
    // to settle; the outer ones mean no engine or no alphabet. Either way the
    // caller gets one flat Option.
    match wrapped {
        // Three layers: the closure's own Option, plus one each from
        // with_engine and with_alphabet. The innermost is the decode
        // settling; the outer two mean no engine or no alphabet.
        Some(Some(decision)) => decision,
        _ => None,
    }
}


