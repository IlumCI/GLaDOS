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
//! instruction. This was written when the loaded model was stories260K, which
//! could not and never would; SmolLM2-135M can to a degree, and `act` does
//! reach the right applet on clear tasks. It is still not good enough to
//! iterate on -- `route` beats it -- so the loop stays unbuilt.
//!
//! What can be established regardless -- and is, by `selftest` -- is that the
//! mechanism is sound: that the grammar admits exactly the applet names, that
//! permission cannot leak, and that a decode always terminates on a real
//! applet. Those properties are independent of which model is loaded, which is
//! why they were worth testing against one that had no idea what it was doing.

use super::constrain::{step_bound, Alphabet, Cursor, Grammar, MAX_LEADING_SPACES};
use super::{sample, tensor, tokenizer, with_engine};
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::sysbox;
use crate::{kprintln, sync::Racy};
use alloc::string::String;
use alloc::format;
use alloc::vec::Vec;

/// Decoded token bytes, built once from the loaded tokenizer.
///
/// Cached because it is derived from the vocabulary and never changes, and
/// rebuilding it per decode would mean re-deriving every token's text on every
/// step of every choice.
static ALPHABET: Racy<Option<Alphabet>> = Racy::new(None);

pub(crate) fn with_alphabet<R>(f: impl FnOnce(&Alphabet) -> R) -> Option<R> {
    unsafe {
        if ALPHABET.get().is_none() {
            let built = with_engine(|e| Alphabet::new(&e.tok))?;
            *ALPHABET.get() = Some(built);
        }
        ALPHABET.get().as_ref().map(f)
    }
}

/// The cached alphabet, as a borrow that outlives the engine borrow.
///
/// A constrained decode needs the alphabet *and* `&mut Engine` at the same
/// time, and a closure holding `&e.tok` cannot also hand out `&mut e`. So the
/// reference is detached from the static instead. `Racy` is single-core
/// interior mutability, the alphabet is written once and never replaced, and
/// nothing takes `&mut` to it after construction -- which is the same footing
/// everything else in this kernel stands on, named here rather than left
/// implicit because this one hands out a lifetime it did not get honestly.
pub(crate) fn alphabet_for(tok: &tokenizer::Tokenizer) -> &'static Alphabet {
    unsafe {
        if ALPHABET.get().is_none() {
            *ALPHABET.get() = Some(Alphabet::new(tok));
        }
        &*(ALPHABET.get().as_ref().unwrap() as *const Alphabet)
    }
}

/// The same cache, for a caller that already holds the engine.
///
/// `with_alphabet` reaches for the engine to build the alphabet the first
/// time, which a caller already holding `&mut Engine` cannot do -- a second
/// `&mut Engine` is undefined behaviour rather than a race, and this module
/// has the scar tissue to prove it. Handing the tokenizer in instead lets the
/// trainer share the decoder's alphabet rather than deriving 151,936 token
/// texts of its own, which matters for more than time: the claim that
/// training moves the decision the decoder makes is only true if both are
/// looking at the same vocabulary.
pub(crate) fn with_alphabet_of<R>(
    tok: &tokenizer::Tokenizer,
    f: impl FnOnce(&Alphabet) -> R,
) -> R {
    unsafe {
        if ALPHABET.get().is_none() {
            *ALPHABET.get() = Some(Alphabet::new(tok));
        }
        // Just built if it was absent, so the unwrap cannot fire.
        f(ALPHABET.get().as_ref().unwrap())
    }
}

/// The applet names one trust level reaches. The agent loop builds its own
/// grammar from this -- its list carries the `done` sentinel besides -- but
/// the admission rule stays here so there is exactly one definition of what
/// each trust level may name.
pub(crate) fn admitted(trust: Trust) -> Vec<&'static str> {
    sysbox::APPLETS
        .iter()
        .filter(|a| trust.admits(a))
        .map(|a| a.name)
        .collect()
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
/// to *understand* rather than merely be constrained. Everything else in this
/// module is guaranteed by construction; this is the part that is merely
/// hoped for. It is measurably the weak link -- the probe in `probe.rs`, which
/// never reads this prompt at all, routes better than decoding against it.
pub(crate) fn prompt_for(task: &str, names: &[&'static str]) -> String {
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
                    // On the way out of the success path too, not just the
                    // failure one -- the cache is equally overwritten either
                    // way, and only invalidating on failure would leave the
                    // corruption in place exactly when it went well.
                    invalidate_conversation(e);
                    return Some(Choice { applet: name, mutates, steps });
                }

                // Advance the model by the token it just committed to.
                e.model.forward(&mut e.state, next, pos);
                pos += 1;
            }
            invalidate_conversation(e);
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
/// The model's *judgement* is deliberately not what is tested here; it varies
/// with whichever checkpoint is loaded, and `fit`/`gate` measure it properly.
/// What is tested is that no sequence of sampling outcomes escapes the
/// grammar, and that is the property the whole design leans on.
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
            let mut escaped_grammar = 0u32;
            let mut unsettled = 0u32;
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
                // Two very different failures, counted apart. Lumping them
                // together made an intermittent FAIL impossible to read: one
                // means the grammar let a decode out of bounds, which would be
                // a genuine hole in the mechanism, and the other means this
                // uniformly-random walk spent its whole budget on tokens that
                // do not advance -- which says nothing about the grammar and
                // everything about sampling with no model behind it.
                match settled {
                    Some(i) if i < names.len() => {
                        if names[i] == "snaps" {
                            reached_snaps = true;
                        }
                    }
                    Some(_) => escaped_grammar += 1,
                    None => unsettled += 1,
                }
            }
            (escaped_grammar, unsettled, reached_snaps)
        })
    })
    .flatten();

    match escaped {
        None => ok &= check("constrained decode always lands on a real applet", false),
        Some((escaped_grammar, unsettled, reached_snaps)) => {
            // The property the mechanism actually promises: a decode that
            // settles has settled on a real applet. This must never fail.
            ok &= check(
                "no random decode ever escaped the grammar",
                escaped_grammar == 0,
            );
            // Not a correctness property, so it does not gate `ok`. A uniform
            // sampler over the whole vocabulary will sometimes spend its
            // entire idle budget on tokens that do not advance the cursor;
            // that is the random walk wandering, not the grammar leaking, and
            // a real decode has a model steering it. Reported because a *rise*
            // here would be worth noticing.
            if unsettled > 0 {
                console::set_color(YELLOW);
                kprintln!(
                    "  note {}/200 random decodes ran out of budget without settling",
                    unsettled
                );
                console::set_color(LTGRAY);
            }
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

// --- native tool selection ----------------------------------------------
//
// The grammar path above spells an applet name out character by character.
// This one does not spell anything: the applet *is* a token, scored directly
// against the model's final hidden state. No grammar, no cursor, no
// terminator, one step instead of four to seven.
//
// Permission is enforced identically and for the same reason -- a forbidden
// applet is simply not among the rows scored, so it has no logit and cannot be
// sampled. The guarantee does not weaken by dropping the grammar.

use super::vocab;

/// Which representation of a task the head classifies.
///
/// `Hidden` is the obvious choice and the one that failed: `probe` measures
/// a separation gap of zero, because a 260K story model fed text far outside
/// its training distribution collapses every prompt onto nearly the same
/// direction. `Pooled` skips the transformer entirely and averages the
/// *embeddings* of the task's words, which the zero-shot result showed does
/// carry signal -- a row built from descriptions alone ranks its own applet
/// 4th of 21 rather than 11th.
///
/// Keeping both is the point. The failure is a measured property of one
/// feature, not a fact about the machine, and it is worth being able to
/// re-measure when a competent model arrives -- at which case `Hidden` should
/// win decisively and `Pooled` should look crude.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    Hidden,
    Pooled,
}

static FEATURE: Racy<Feature> = Racy::new(Feature::Pooled);

pub fn feature_mode() -> Feature {
    unsafe { *FEATURE.get() }
}

pub fn set_feature_mode(f: Feature) {
    unsafe { *FEATURE.get() = f };
}

fn feature(e: &mut super::Engine, task: &str) -> Option<Vec<f32>> {
    match feature_mode() {
        Feature::Hidden => feature_hidden(e, task),
        Feature::Pooled => {
            let ids = e.tok.encode(task, false, false);
            if ids.is_empty() {
                return None;
            }
            let dim = e.model.cfg.dim;
            let mut v = alloc::vec![0.0f32; dim];
            let mut row = alloc::vec![0.0f32; dim];
            for id in &ids {
                e.model.embed_into(*id, &mut row);
                for (a, x) in v.iter_mut().zip(row.iter()) {
                    *a += *x;
                }
            }
            let k = 1.0 / ids.len() as f32;
            for a in v.iter_mut() {
                *a *= k;
            }
            Some(v)
        }
    }
}

/// Run a prompt through the model and return the final hidden state.
fn feature_hidden(e: &mut super::Engine, task: &str) -> Option<Vec<f32>> {
    let prompt = prompt_for_task(task);
    let tokens = e.tok.encode(&prompt, true, false);
    if tokens.is_empty() {
        return None;
    }
    let limit = e.model.cfg.seq_len;
    let mut pos = 0usize;
    for &t in tokens.iter() {
        if pos >= limit {
            break;
        }
        e.model.forward(&mut e.state, t, pos);
        pos += 1;
    }
    let out = e.state.hidden().to_vec();
    invalidate_conversation(e);
    Some(out)
}

/// Mark the live conversation as gone.
///
/// Anything that runs a prompt from position zero overwrites the KV cache for
/// those positions, and `e.pos` is then a promise the cache cannot keep: a
/// `cont` would resume from entries belonging to somebody else's prompt and
/// produce fluent continuations of a sentence that was never written. Resetting
/// makes the next generation start clean, which is the honest outcome, and is
/// cheap next to snapshotting a cache purely to restore something the operator
/// was probably not still using.
pub(crate) fn invalidate_conversation(e: &mut super::Engine) {
    e.pos = 0;
    e.last_token = super::tokenizer::BOS;
}

/// The probe's feature vector for a task. The deliberation tier ranks
/// candidate choices with the same features the router was fitted on;
/// exposing the private helper to in-crate callers is the whole change.
pub(crate) fn feature_for(e: &mut super::Engine, task: &str) -> Option<Vec<f32>> {
    feature(e, task)
}

/// Shorter than the grammar prompt: with applet tokens there is no need to
/// list the tools, because the candidate set is imposed by the scoring rather
/// than described in words.
pub(crate) fn prompt_for_task(task: &str) -> String {
    let mut s = String::from("Task: ");
    s.push_str(task);
    s.push_str(". Tool:");
    s
}

// --- training -----------------------------------------------------------

pub struct TrainReport {
    pub examples: usize,
    pub held_out: usize,
    pub before_train: f32,
    pub after_train: f32,
    pub before_test: f32,
    pub after_test: f32,
}

/// The tail of the corpus is held out.
///
/// Not every fourth example, which is what this used to do. The corpus is
/// generated from template families and the generator emits whole families
/// into the held-out tail, so an interleaved split would put paraphrases of
/// training items into the test set and report memorisation as generalisation.
/// Anything appended later by `teach` lands past the seed and trains.
fn is_held_out(i: usize) -> bool {
    // Boundaries rather than constants: `vocab::splits` returns the compiled
    // ones until a bundle is imported over the corpus, and the imported ones
    // after. Reading the constants directly here would have measured a new
    // corpus against the old corpus's positions.
    let (train, _, seed) = vocab::splits();
    i >= train && i < seed
}

pub fn train(epochs: usize, lr: f32) -> Option<TrainReport> {
    let examples = vocab::examples();
    if examples.is_empty() {
        return None;
    }
    with_engine(|e| {
        let all: Vec<usize> = (0..e.head.len()).collect();
        let held_out = examples.iter().enumerate().filter(|(i, _)| is_held_out(*i)).count();

        // The base model is frozen, so a feature vector is a constant. The
        // first version recomputed it every epoch, which meant 600 forward
        // passes to do 600 outer products and made a 20-epoch run take two
        // minutes. Caching is not an optimisation detail here -- it is the
        // reason adapter training is cheap at all, and skipping it gave away
        // the entire advantage of freezing the base.
        let mut cached: Vec<(usize, Vec<f32>, bool)> = Vec::new();
        for (i, ex) in examples.iter().enumerate() {
            let Some(target) = e.head.index_of(&ex.applet) else { continue };
            let Some(x) = feature(e, &ex.task) else { continue };
            cached.push((target, x, is_held_out(i)));
        }

        let score = |e: &super::Engine, held: bool, cached: &[(usize, Vec<f32>, bool)]| -> f32 {
            let (mut right, mut total) = (0usize, 0usize);
            for (target, x, h) in cached.iter() {
                if *h != held {
                    continue;
                }
                let logits = e.head.logits(x, &all);
                let mut best = 0usize;
                for (j, l) in logits.iter().enumerate() {
                    if *l > logits[best] {
                        best = j;
                    }
                }
                total += 1;
                if best == *target {
                    right += 1;
                }
            }
            if total == 0 { 0.0 } else { right as f32 / total as f32 }
        };

        let before_train = score(e, false, &cached);
        let before_test = score(e, true, &cached);

        for _ in 0..epochs {
            for (target, x, held) in cached.iter() {
                if *held {
                    continue;
                }
                let mut probs = e.head.logits(x, &all);
                tensor::softmax(&mut probs);
                e.head.learn(x, &all, &probs, *target, lr);
            }
        }

        let after_train = score(e, false, &cached);
        let after_test = score(e, true, &cached);

        TrainReport {
            examples: examples.len(),
            held_out,
            before_train,
            after_train,
            before_test,
            after_test,
        }
    })
}

/// `train adapter` -- the QDoRA decision-layer run against the loaded model.
///
/// Separate from `train_report` above, which fits the linear probe's head and
/// has nothing to do with the model's own weights. Two different things are
/// called training here and confusing them would make every number ambiguous:
/// this one moves an adapter over the classifier, and `fit`/`train` move a
/// closed-form router that never touches the checkpoint at all.
pub fn adapter_train_report(b: &super::train::Budget) {
    use super::train::{RunError, RunReport};

    console::set_color(YELLOW);
    kprintln!("[train adapter]");
    console::set_color(LTGRAY);

    let outcome = super::with_engine(|e| super::train::run(e, b));
    let r: RunReport = match outcome {
        None => {
            kprintln!("  no engine, or another task holds it");
            return;
        }
        Some(Err(RunError::Hardware)) => {
            console::set_color(LTRED);
            kprintln!("  refused: this machine has no AVX2/FMA path");
            console::set_color(LTGRAY);
            // Said in full rather than as a flag, because the refusal is the
            // interesting part: the scalar kernels are correct and would
            // produce the same adapter, just slowly enough that every
            // judgement made from the run would be about the clock.
            kprintln!("  scalar emulation would make each step minutes; nothing was trained");
            return;
        }
        Some(Err(RunError::Hybrid)) => {
            kprintln!("  refused: hybrid checkpoints have no verified backward yet");
            return;
        }
        Some(Err(RunError::NoCorpus)) => {
            kprintln!("  no corpus at {}", super::vocab::CORPUS);
            return;
        }
        Some(Err(RunError::NoDecisions)) => {
            kprintln!("  the grammar could not spell a single applet from this vocabulary");
            return;
        }
        Some(Ok(r)) => r,
    };

    kprintln!(
        "  {} examples -> {} decisions ({} held out), {} reachable classifier rows",
        r.examples, r.decisions, r.held, r.rows
    );
    kprintln!("  grammar + rows {} ms, features {} ms (per example)", r.chains_ms, r.prep_ms);
    kprintln!("  trained {} epochs in {} ms", r.epochs_run, r.train_ms);
    if r.stopped {
        console::set_color(YELLOW);
        kprintln!("  stopped on the wall-clock budget, not on the epoch count");
        console::set_color(LTGRAY);
    }
    kprintln!("  loss      {:.3} -> {:.3}", r.first_loss, r.last_loss);
    kprintln!(
        "  seen      {}% -> {}%",
        (r.before_train * 100.0) as u32,
        (r.after_train * 100.0) as u32
    );
    // Same rule as everywhere else in this file: training accuracy is what
    // memorisation looks like, and only the held-out number can move for a
    // reason worth having.
    let up = r.after_held > r.before_held;
    console::set_color(if up { LTGREEN } else { YELLOW });
    kprintln!(
        "  held out  {}% -> {}%   <- the one that counts",
        (r.before_held * 100.0) as u32,
        (r.after_held * 100.0) as u32
    );
    console::set_color(LTGRAY);
    kprintln!("  adapter attached; 'act <task>' now decodes through it");
}

/// Score a description that belongs to no applet in the table.
///
/// This is the hypernetwork's reason to exist: if `h` learned the general map
/// from a description to a row, then a tool written after training still gets a
/// usable row. Returns the rank of the true applet among all of them, where 0
/// means the generated row scored it first.
pub fn zero_shot_rank(held: &str) -> Option<(usize, usize)> {
    with_engine(|e| {
        let idx = e.head.index_of(held)?;
        let applet = sysbox::APPLETS.iter().find(|a| a.name == held)?;
        let mut text = String::from(applet.name);
        text.push(' ');
        text.push_str(applet.help);

        // A row built only from the description, with no per-applet delta.
        let generated = e.head.row_for_text(&e.model, &e.tok, &text);

        // Rank the real rows by similarity to it.
        let mut better = 0usize;
        let target = vocab::cosine(&generated, &e.head.row(idx));
        for i in 0..e.head.len() {
            if i != idx && vocab::cosine(&generated, &e.head.row(i)) > target {
                better += 1;
            }
        }
        Some((better, e.head.len()))
    })?
}

pub fn train_report(epochs: usize) {
    console::set_color(YELLOW);
    kprintln!("[train]");
    console::set_color(LTGRAY);

    let t0 = crate::time::rdtsc();
    let Some(r) = train(epochs, 0.05) else {
        console::set_color(LTRED);
        kprintln!("  no corpus at {} (or no model loaded)", vocab::CORPUS);
        console::set_color(LTGRAY);
        return;
    };
    let elapsed = crate::time::rdtsc() - t0;

    let held = r.held_out;
    kprintln!(
        "  {} examples, {} held out, {} epochs",
        r.examples - held,
        held,
        epochs
    );
    kprintln!(
        "  seen      {}% -> {}%",
        (r.before_train * 100.0) as u32,
        (r.after_train * 100.0) as u32
    );
    // The number that means anything. Training accuracy can be driven up by
    // the per-applet delta memorising each example; held-out accuracy can only
    // improve if something general was learned.
    let up = r.after_test > r.before_test;
    console::set_color(if up { LTGREEN } else { YELLOW });
    kprintln!(
        "  held out  {}% -> {}%   <- the one that counts",
        (r.before_test * 100.0) as u32,
        (r.after_test * 100.0) as u32
    );
    console::set_color(LTGRAY);

    let mhz = crate::time::tsc_mhz();
    if mhz > 0 {
        let ms = elapsed / mhz / 1000;
        kprintln!("  {} ms, base model never touched", ms);
    }

    // Chance level, so the accuracies above can be read against something.
    with_engine(|e| {
        kprintln!("  (chance is {}%)", 100 / e.head.len().max(1));
    });
}

/// Report how well a description alone locates an applet.
pub fn zero_shot_report(name: &str) {
    console::set_color(YELLOW);
    kprintln!("[zero-shot]");
    console::set_color(LTGRAY);
    match zero_shot_rank(name) {
        None => kprintln!("  '{}' is not an applet, or no model is loaded", name),
        Some((rank, total)) => {
            console::set_color(if rank == 0 { LTGREEN } else { YELLOW });
            kprintln!(
                "  a row generated from '{}' descriptions alone ranks it {} of {}",
                name,
                rank + 1,
                total
            );
            console::set_color(LTGRAY);
            kprintln!("  (no per-applet delta used -- this is what a brand new applet would get)");
        }
    }
}

/// Ask whether the features can possibly work, before blaming the head.
///
/// The head is a linear classifier on the model's last hidden state. If that
/// state is nearly the same vector for every prompt -- which is entirely
/// plausible, since a 260K story model is being fed text far outside anything
/// it was trained on, through a template that barely varies -- then no linear
/// head can separate the classes and no amount of training or architecture
/// will help. That is a property of the features, not of the classifier, and
/// it is worth measuring before tuning anything.
///
/// The number that matters is the gap: mean cosine between features of
/// examples sharing an applet, against those with different applets. A gap of
/// nearly zero means the features carry no task information.
pub fn probe_features() {
    console::set_color(YELLOW);
    kprintln!("[probe]");
    console::set_color(LTGRAY);

    let examples = vocab::examples();
    if examples.is_empty() {
        kprintln!("  no corpus");
        return;
    }

    let got = with_engine(|e| {
        let mut feats: Vec<(usize, Vec<f32>)> = Vec::new();
        for ex in examples.iter() {
            let Some(t) = e.head.index_of(&ex.applet) else { continue };
            let Some(x) = feature(e, &ex.task) else { continue };
            feats.push((t, x));
        }

        let (mut same, mut same_n) = (0.0f32, 0usize);
        let (mut diff, mut diff_n) = (0.0f32, 0usize);
        let (mut lo, mut hi) = (1.0f32, -1.0f32);
        for i in 0..feats.len() {
            for j in (i + 1)..feats.len() {
                let c = vocab::cosine(&feats[i].1, &feats[j].1);
                if c < lo { lo = c; }
                if c > hi { hi = c; }
                if feats[i].0 == feats[j].0 {
                    same += c;
                    same_n += 1;
                } else {
                    diff += c;
                    diff_n += 1;
                }
            }
        }
        (
            feats.len(),
            if same_n > 0 { same / same_n as f32 } else { 0.0 },
            if diff_n > 0 { diff / diff_n as f32 } else { 0.0 },
            lo,
            hi,
        )
    });

    let Some((n, same, diff, lo, hi)) = got else {
        kprintln!("  no model loaded");
        return;
    };

    let pct = |v: f32| (v * 1000.0) as i32;
    kprintln!("  {} features, pairwise cosine {}..{} (x1000)", n, pct(lo), pct(hi));
    kprintln!("  same applet      {}", pct(same));
    kprintln!("  different applet {}", pct(diff));

    let gap = same - diff;
    let spread = hi - lo;
    console::set_color(if gap > 0.005 { LTGREEN } else { LTRED });
    kprintln!("  separation gap   {} (x1000)", pct(gap));
    console::set_color(LTGRAY);

    // Two different diagnoses, and conflating them sent me tuning a learning
    // rate when the features were the problem. Collapse means every prompt maps
    // to nearly one direction, and nothing downstream can recover; a small gap
    // with a wide spread means the features vary but not along task lines,
    // which more or better-labelled data could still exploit.
    if spread < 0.1 {
        console::set_color(LTRED);
        kprintln!("  features are collapsed -- no head of any kind can separate them");
        console::set_color(LTGRAY);
    } else if gap <= 0.005 {
        kprintln!("  features vary, but not along task lines");
    }
}

// --- the closed-form router ---------------------------------------------

/// Fit a ridge probe from the corpus, holding out a quarter to score it.
///
/// Replaces `train`. That one ran SGD on the vocabulary head and made held-out
/// accuracy fall monotonically -- 30% untrained, 0% after eight epochs -- which
/// is what fitting 21 classes from 40 examples by descent gets you. There is no
/// equivalent knob to get wrong here: the solution is closed form, so the only
/// choice is the regularisation, and the measured curve across two orders of
/// magnitude of it is 51%, 55%, 49%.
pub fn fit_probe(lambda: f32) {
    console::set_color(YELLOW);
    kprintln!("[probe]");
    console::set_color(LTGRAY);

    let examples = vocab::examples();
    if examples.is_empty() {
        kprintln!("  no corpus at {}", vocab::CORPUS);
        return;
    }

    let t0 = crate::time::rdtsc();
    let built = with_engine(|e| {
        let classes = e.head.len();
        let mut train_x: Vec<Vec<f32>> = Vec::new();
        let mut train_y: Vec<usize> = Vec::new();
        let mut test: Vec<(Vec<f32>, usize)> = Vec::new();

        for (i, ex) in examples.iter().enumerate() {
            let Some(y) = e.head.index_of(&ex.applet) else { continue };
            let Some(x) = feature(e, &ex.task) else { continue };
            if is_held_out(i) {
                test.push((x, y));
            } else {
                train_x.push(x);
                train_y.push(y);
            }
        }
        if train_x.is_empty() {
            return None;
        }

        let p = super::probe::Probe::fit(&train_x, &train_y, classes, lambda)?;

        let hit = |set: &[(Vec<f32>, usize)]| -> (usize, usize) {
            let mut right = 0;
            for (x, y) in set {
                if p.predict(x) == *y {
                    right += 1;
                }
            }
            (right, set.len())
        };
        let seen: Vec<(Vec<f32>, usize)> =
            train_x.iter().cloned().zip(train_y.iter().copied()).collect();
        let (tr_ok, tr_n) = hit(&seen);
        let (te_ok, te_n) = hit(&test);

        let params = p.params();
        e.probe = Some(p);

        // The corroborating cores train on the same examples the probe did --
        // never on the held-out tail, or agreement would be measured against
        // items all three had memorised and would look far more informative
        // than it is.
        let texts: Vec<&str> = examples
            .iter()
            .enumerate()
            .filter(|(i, _)| !is_held_out(*i))
            .map(|(_, ex)| ex.task.as_str())
            .collect();
        let labels: Vec<usize> = examples
            .iter()
            .enumerate()
            .filter(|(i, _)| !is_held_out(*i))
            .filter_map(|(_, ex)| e.head.index_of(&ex.applet))
            .collect();
        let council_params = if texts.len() == labels.len() {
            match super::council::Council::fit(&texts, &labels, classes, &e.tok) {
                Some(c) => {
                    let n = c.params();
                    e.council = Some(c);
                    n
                }
                None => 0,
            }
        } else {
            0
        };

        // Persist the fitted router beside its corpus hash, so the next boot
        // loads it instead of paying a forward pass per example again. A
        // stale blob -- corpus grown or changed since the fit -- is detected
        // by the hash and refused at load, never silently trusted.
        if let (Some(p), Some(c)) = (e.probe.as_ref(), e.council.as_ref()) {
            if let Some(hash) = crate::sysbox::hash_of(super::vocab::CORPUS) {
                let mut blob = Vec::with_capacity(8 + 64 + p.to_bytes().len() + c.to_bytes().len());
                blob.extend_from_slice(b"GLADOSRT");
                for b in hash {
                    blob.extend_from_slice(&format!("{:02x}", b).into_bytes());
                }
                blob.extend_from_slice(&p.to_bytes());
                blob.extend_from_slice(&c.to_bytes());
                crate::sysbox::write_blob(ROUTER_PATH, blob);
            }
        }

        Some((tr_ok, tr_n, te_ok, te_n, params, classes, council_params))
    })
    .flatten();

    let Some((tr_ok, tr_n, te_ok, te_n, params, classes, council_params)) = built else {
        console::set_color(LTRED);
        kprintln!("  could not fit (no model, or the features are degenerate)");
        console::set_color(LTGRAY);
        return;
    };
    let elapsed = crate::time::rdtsc() - t0;

    kprintln!("  {} train, {} held out, {} classes", tr_n, te_n, classes);
    kprintln!("  seen      {}%", pct(tr_ok, tr_n));
    let te = pct(te_ok, te_n);
    console::set_color(if te * classes as u32 > 200 { LTGREEN } else { YELLOW });
    kprintln!("  held out  {}%   <- the one that counts", te);
    console::set_color(LTGRAY);
    kprintln!("  chance is {}%", 100 / classes.max(1));
    kprintln!("  {} parameters, closed form -- no epochs, nothing to overfit", params);
    if council_params > 0 {
        kprintln!(
            "  council {} more, counted not solved -- they judge confidence, not the answer",
            council_params
        );
    }
    let mhz = crate::time::tsc_mhz();
    if mhz > 0 {
        kprintln!("  fitted in {} ms", elapsed / mhz / 1000);
    }
}

fn pct(a: usize, b: usize) -> u32 {
    if b == 0 {
        0
    } else {
        (a * 100 / b) as u32
    }
}

/// Route a task with the probe. Microseconds: one matrix-vector product.
pub fn route(task: &str, trust: Trust) -> Option<Choice> {
    with_engine(|e| {
        let x = feature(e, task)?;
        let p = e.probe.as_ref()?;
        let scores = p.scores(&x);

        // Permission is still enforced by omission -- a forbidden applet is
        // not considered, so it cannot be returned however high it scored.
        let mut best: Option<(usize, f32)> = None;
        for (i, s) in scores.iter().enumerate() {
            let name = e.head.name(i);
            let Some(a) = sysbox::APPLETS.iter().find(|a| a.name == name) else { continue };
            if !trust.admits(a) {
                continue;
            }
            if best.map(|(_, b)| *s > b).unwrap_or(true) {
                best = Some((i, *s));
            }
        }
        let (i, _) = best?;
        let name = e.head.name(i);
        let mutates = sysbox::APPLETS
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.mutates)
            .unwrap_or(false);
        Some(Choice { applet: name, mutates, steps: 1 })
    })?
}

/// Route, and say how much the answer should be trusted.
///
/// The probe answers alone. Combining the three cores was measured at 76.9%
/// against the probe's own 77.8%, so a product would make the answer slightly
/// worse; what the cores are for is the other number from that measurement --
/// where all three agree the answer is right 90.3% of the time, and where they
/// split, 50%.
///
/// Where the fitted router is kept. Content-addressed by the corpus hash it
/// was fitted against, stored as an ordinary namespace object.
const ROUTER_PATH: &str = "/ai/router.bin";

/// Load the persisted router if it is present and its corpus hash still
/// matches. Idempotent: a live probe is left alone. Returns whether the
/// engine now has a fitted router -- the gate-first path in the agent loop
/// refuses to act without one, because an unfitted head's answer is not a
/// measurement, it is a guess with a number attached.
pub fn ensure_router() -> bool {
    let fitted = with_engine(|e| e.probe.is_some());
    if fitted == Some(true) {
        return true;
    }
    let Some(blob) = crate::sysbox::read_blob(ROUTER_PATH) else {
        return false;
    };
    if blob.len() < 8 + 64 || &blob[0..8] != b"GLADOSRT" {
        return false;
    }
    let Some(hash) = crate::sysbox::hash_of(super::vocab::CORPUS) else {
        return false;
    };
    let hex: Vec<u8> = hash.iter().flat_map(|b| format!("{:02x}", b).into_bytes()).collect();
    if blob[8..72] != hex[..] {
        return false;
    }
    let rest = &blob[72..];
    // The probe's length is not stored; from_bytes is self-delimiting by
    // dims, and the council follows it. Parse the probe first, then the
    // council from the remainder.
    let p = match super::probe::Probe::from_bytes(rest) {
        Some(p) => p,
        None => return false,
    };
    let plen = p.byte_len();
    let c = match super::council::Council::from_bytes(&rest[plen..]) {
        Some(c) => c,
        None => return false,
    };
    with_engine(|e| {
        e.probe = Some(p);
        e.council = Some(c);
    })
    .is_some()
}

/// So the cores are consulted for corroboration and never for content. Their
/// verdict changes what is *said* about the answer, not the answer.
pub fn route_verdict(task: &str, trust: Trust) -> Option<(Choice, super::council::Verdict)> {
    let probe_choice = route(task, trust)?;
    with_engine(|e| {
        let probe_idx = e.head.index_of(probe_choice.applet)?;
        let allowed = allowed_classes(e, trust);

        let Some(c) = e.council.as_ref() else {
            return Some((
                probe_choice,
                super::council::Verdict {
                    applet: probe_idx,
                    agreement: 1,
                    lexical: probe_idx,
                    character: probe_idx,
                },
            ));
        };
        let (lexical, character) = c.corroborate(task, &e.tok, &allowed)?;

        // Whatever `search` last adopted, defaulting to majority with ties to
        // the probe.
        //
        // It was that majority, hardcoded -- so the search swept three rules,
        // spent part of the test budget choosing between them, announced an
        // adoption, and the router went on doing the same thing regardless.
        // The chosen rule had never reached a single routing decision. Going
        // through `decide` means the search and the router cannot disagree
        // about what a rule *is*, which is the second half of the same fault.
        //
        // The default is the old behaviour, so a machine that has never
        // searched routes exactly as it did.
        //
        // On the rule itself: the first version had the probe answer alone and
        // the cores only corroborate, on the grounds that a *product* of the
        // three scored 76.9% against the probe's 77.8%. That reasoning did not
        // transfer -- a majority vote is a different rule, and it scores
        // 80.6%. The difference lives in the eight items where the probe
        // stands alone against the other two, where the probe is right 25% of
        // the time and the pair 62.5%, so being outvoted is genuine evidence
        // and the measurement that dismissed it was of something else.
        let rule = rule_in_force();
        let winner = decide(
            rule,
            probe_idx,
            Some(c),
            &e.tok,
            task,
            &allowed,
            core_vote(rule, task, &allowed),
        );

        let agreement = usize::from(winner == probe_idx)
            + usize::from(winner == lexical)
            + usize::from(winner == character);

        let name = e.head.name(winner);
        let mutates = sysbox::APPLETS
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.mutates)
            .unwrap_or(false);

        Some((
            Choice { applet: name, mutates, steps: 1 },
            super::council::Verdict { applet: winner, agreement, lexical, character },
        ))
    })?
}

/// Class indices the trust level permits, in head order.
fn allowed_classes(e: &super::Engine, trust: Trust) -> Vec<usize> {
    (0..e.head.len())
        .filter(|i| {
            sysbox::APPLETS
                .iter()
                .find(|a| a.name == e.head.name(*i))
                .map(|a| trust.admits(a))
                .unwrap_or(false)
        })
        .collect()
}

pub fn route_report(task: &str, trust: Trust) {
    console::set_color(YELLOW);
    kprintln!("[route]");
    console::set_color(LTGRAY);
    let t0 = crate::time::rdtsc();
    let got = route_verdict(task, trust);
    let elapsed = crate::time::rdtsc() - t0;

    let Some((c, v)) = got else {
        kprintln!("  no probe fitted -- run 'fit' first");
        return;
    };

    // Which rule answered. Worth a line because it is now variable: `search`
    // adopts one and the router honours it, where before the router did
    // majority regardless of what the search had chosen.
    kprintln!("  rule {} ({})", rule_in_force().name(), match load_config() {
        Some(_) => "adopted",
        None => "default",
    });

    if v.confident() {
        console::set_color(LTGREEN);
        kprintln!("  {}", c.applet);
        console::set_color(LTGRAY);
        kprintln!("  all 3 cores agree (measured 90% right when they do)");
    } else {
        console::set_color(YELLOW);
        kprintln!("  {}  -- not confident", c.applet);
        console::set_color(LTGRAY);
        with_engine(|e| {
            // Naming the dissenters is the useful part. "Uncertain" alone
            // tells the operator nothing they can act on; two plausible
            // alternatives is a question they can answer.
            kprintln!("    lexical    {}", e.head.name(v.lexical));
            kprintln!("    character  {}", e.head.name(v.character));
        });
        kprintln!("  {} of 3 agree (measured 50% right when split)", v.agreement);
        kprintln!("  'ask <question>' to put it to the model instead");
    }

    let mhz = crate::time::tsc_mhz();
    if mhz > 0 {
        kprintln!("  {} us, no transformer involved", elapsed / mhz);
    }
    if c.mutates {
        console::set_color(YELLOW);
        kprintln!("  that applet mutates content");
        console::set_color(LTGRAY);
    }
}

/// Measure the gate on the held-out corpus.
///
/// Prints the two numbers the whole design rests on: how often the cores agree,
/// and how much more often they are right when they do. If that gap ever
/// closes, the gate is worthless and should be deleted rather than trusted.
pub fn gate_report() {
    console::set_color(YELLOW);
    kprintln!("[gate]");
    console::set_color(LTGRAY);

    let examples = vocab::examples();
    let out = with_engine(|e| {
        if e.probe.is_none() || e.council.is_none() {
            return None;
        }
        let (mut agree_n, mut agree_ok) = (0usize, 0usize);
        let (mut split_n, mut split_ok) = (0usize, 0usize);

        for (i, ex) in examples.iter().enumerate() {
            if !is_held_out(i) {
                continue;
            }
            let Some(want) = e.head.index_of(&ex.applet) else { continue };
            let Some(x) = feature(e, &ex.task) else { continue };
            let probe_pick = e.probe.as_ref()?.predict(&x);
            let all: Vec<usize> = (0..e.head.len()).collect();
            let (lex, chr) = e.council.as_ref()?.corroborate(&ex.task, &e.tok, &all)?;
            // Scored against the rule actually used, majority with ties to the
            // probe -- measuring the gate against a rule the router does not
            // follow would report a number nobody experiences.
            let got = if lex == chr && lex != probe_pick { lex } else { probe_pick };
            if lex == got && chr == got && probe_pick == got {
                agree_n += 1;
                agree_ok += usize::from(got == want);
            } else {
                split_n += 1;
                split_ok += usize::from(got == want);
            }
        }
        Some((agree_n, agree_ok, split_n, split_ok))
    })
    .flatten();

    let Some((an, aok, sn, sok)) = out else {
        kprintln!("  fit the probe and council first");
        return;
    };
    let total = an + sn;
    if total == 0 {
        kprintln!("  nothing held out to measure against");
        return;
    }
    console::set_color(LTGREEN);
    kprintln!("  all agree   {:>3}/{:<3}  {}% right", an, total, pct(aok, an));
    console::set_color(YELLOW);
    kprintln!("  they split  {:>3}/{:<3}  {}% right", sn, total, pct(sok, sn));
    console::set_color(LTGRAY);
    kprintln!("  overall     {}%", pct(aok + sok, total));
    kprintln!("  the gap is the whole point -- agreement is worth acting on");
}

// --- searching the space of routing functions ---------------------------
//
// Everything above fixes one pipeline and fits its parameters. This searches
// the pipeline itself: which cores participate, how they combine, how much
// each is smoothed or regularised. In the meta-learning ladder that is the
// step from f_theta(x) to P(f) -- the object being chosen is the reasoning
// function, not its coefficients.
//
// The Godel-machine condition is what keeps that from being decoration: a
// self-modification is adopted only when it is *verified* to improve a
// measured objective. Here the verification is held-out accuracy, the
// objective is routing, and the modification is a configuration. Nothing is
// adopted on the strength of an argument.
//
// Three splits, because selecting among candidates by score is fitting the set
// you select on. Validation is spent freely during the search; the test slice
// is read once, afterwards, and its number is the one that means anything.

#[derive(Clone, Copy, PartialEq)]
pub struct Config {
    pub lambda: f32,
    /// How the three cores decide. See `Rule`.
    pub rule: Rule,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The ridge probe alone.
    ProbeOnly,
    /// Majority of three, ties to the probe.
    Majority,
    /// Lexical core alone -- included so the search can discover that the
    /// expensive core is not pulling its weight, rather than being told.
    LexicalOnly,
    /// Majority of the three corroborators, ties to the probe, where the third
    /// is a core the machine wrote.
    ///
    /// Kept distinct from `Majority` rather than folded into it, so the search
    /// compares "with the written core" against "without" as two points. With
    /// no core installed the predicate reduces to `Majority` exactly -- two
    /// voters, and "at least two agree" is "they agree" -- so the fallback
    /// costs no special case.
    WithCore,
}

impl Rule {
    pub fn name(&self) -> &'static str {
        match self {
            Rule::ProbeOnly => "probe",
            Rule::Majority => "majority",
            Rule::LexicalOnly => "lexical",
            Rule::WithCore => "withcore",
        }
    }

    /// Whether answering under this rule needs the written core's vote.
    ///
    /// Asked before voting rather than inside `decide`, because a vote runs an
    /// interpreter and the rules that ignore it should not pay for it.
    pub fn needs_core(&self) -> bool {
        matches!(self, Rule::WithCore)
    }
}

/// Which slice example `i` belongs to: 0 train, 1 validation, 2 test.
///
/// Public because anything that mines the corpus has to agree with the judge
/// about where the boundary is. A second copy of this rule would drift, and
/// the drift would be a producer taking its cues from the slice its verdict is
/// measured on -- which is not a bug that announces itself, it is a number
/// that is too good.
pub fn split_of(i: usize) -> u8 {
    let (train, val_end, seed) = vocab::splits();
    if i < train {
        0 // train
    } else if i < val_end {
        1 // validation
    } else if i < seed {
        2 // test
    } else {
        0 // anything `teach` appended trains
    }
}

/// The corpus, featurised once.
///
/// `evaluate` used to do all of this per configuration, and `feature` is a
/// forward pass. Nine configurations therefore cost nine passes over the whole
/// corpus to compute values that cannot differ between them: the base is
/// frozen, so a hidden state is a function of the text alone. That is the same
/// observation `train.rs` is built on -- "the hidden state at every decision is
/// a constant and is cached once per example" -- applied to the one search that
/// was not using it.
///
/// The council is cached for a stronger reason: `Council::fit` does not take a
/// `Config` at all, so refitting it per configuration recomputed a value that
/// was provably identical each time.
struct Featurised {
    xs: Vec<Vec<f32>>,
    ys: Vec<usize>,
    texts: Vec<String>,
    /// `(split, feature, wanted class, text)` for everything not in train.
    ev: Vec<(u8, Vec<f32>, usize, String)>,
    classes: usize,
}

fn featurise(e: &mut super::Engine) -> Option<Featurised> {
    let examples = vocab::examples();
    let classes = e.head.len();
    let mut out = Featurised {
        xs: Vec::new(),
        ys: Vec::new(),
        texts: Vec::new(),
        ev: Vec::new(),
        classes,
    };
    for (i, ex) in examples.iter().enumerate() {
        let Some(y) = e.head.index_of(&ex.applet) else { continue };
        let Some(x) = feature(e, &ex.task) else { continue };
        let split = split_of(i);
        if split == 0 {
            out.xs.push(x);
            out.ys.push(y);
            out.texts.push(ex.task.clone());
        } else {
            out.ev.push((split, x, y, ex.task.clone()));
        }
    }
    if out.xs.is_empty() {
        return None;
    }
    Some(out)
}

/// Everything a configuration needs that a configuration does not change.
struct Fitted {
    council: Option<super::council::Council>,
    /// One probe per distinct lambda, in the order they were asked for.
    probes: Vec<(f32, super::probe::Probe)>,
}

fn fit_all(e: &super::Engine, f: &Featurised, lambdas: &[f32]) -> Fitted {
    let texts: Vec<&str> = f.texts.iter().map(|t| t.as_str()).collect();
    let mut probes = Vec::new();
    for &lambda in lambdas {
        if let Some(p) = super::probe::Probe::fit(&f.xs, &f.ys, f.classes, lambda) {
            probes.push((lambda, p));
        }
    }
    Fitted {
        council: super::council::Council::fit(&texts, &f.ys, f.classes, &e.tok),
        probes,
    }
}

/// What one configuration answers for one decision.
///
/// Lifted out of the scoring loop so the rule has exactly one implementation.
/// It was written inline in `evaluate`, which is fine while there is one
/// caller and is how a second caller ends up with a rule that has quietly
/// drifted from the first.
fn decide(
    rule: Rule,
    probe_says: usize,
    council: Option<&super::council::Council>,
    tok: &super::tokenizer::Tokenizer,
    text: &str,
    all: &[usize],
    core_says: Option<usize>,
) -> usize {
    if matches!(rule, Rule::ProbeOnly) {
        return probe_says;
    }
    let pair = match council {
        None => None,
        Some(c) => c.corroborate(text, tok, all),
    };
    decide_with(rule, probe_says, pair, core_says)
}

/// The rule itself, once the corroborators have already spoken.
///
/// Split out because corroborating is the expensive half -- a tokenize and two
/// Bayes predictions -- and every caller that needs more than one verdict for
/// the same item was paying for it again each time. `core_bench_core` asks
/// three questions per item (with the core, without it, and what the counters
/// said) and so corroborated three times for one answer.
///
/// One definition, reached two ways, for the reason the census shares
/// `core_bench_core` rather than counting separately: a second copy of this
/// arithmetic would be a second opinion about what the council decides, and
/// the two would eventually disagree about a verdict already in the ledger.
pub fn decide_with(
    rule: Rule,
    probe_says: usize,
    pair: Option<(usize, usize)>,
    core_says: Option<usize>,
) -> usize {
    let Some((l, ch)) = pair else { return probe_says };
    match rule {
        Rule::ProbeOnly => probe_says,
        Rule::Majority => {
            if l == ch && l != probe_says {
                l
            } else {
                probe_says
            }
        }
        Rule::LexicalOnly => l,
        // At least two of the corroborators agreeing on something the probe
        // did not say carries it. The probe keeps ties and keeps everything
        // else, which is what makes this an extension of `Majority` rather
        // than a different system: with no core there are two corroborators
        // and "at least two agree" is exactly "they agree".
        Rule::WithCore => {
            let mut votes = alloc::vec![l, ch];
            if let Some(k) = core_says {
                votes.push(k);
            }
            let mut best = probe_says;
            let mut best_n = 0;
            for v in &votes {
                if *v == probe_says {
                    continue;
                }
                let n = votes.iter().filter(|x| *x == v).count();
                if n >= 2 && n > best_n {
                    best = *v;
                    best_n = n;
                }
            }
            best
        }
    }
}

/// The written core's opinion, or `None` when the rule does not want one, no
/// core is installed, or the core declined.
fn core_vote(rule: Rule, text: &str, all: &[usize]) -> Option<usize> {
    if !rule.needs_core() {
        return None;
    }
    super::voter::installed()?.vote(text, all).0
}

/// Score one configuration on a split, against the cached fit.
fn score_cfg(
    e: &super::Engine,
    f: &Featurised,
    fitted: &Fitted,
    cfg: Config,
    split: u8,
) -> Option<(usize, usize)> {
    let probe = fitted
        .probes
        .iter()
        .find(|(l, _)| *l == cfg.lambda)
        .map(|(_, p)| p)?;
    let all: Vec<usize> = (0..f.classes).collect();
    let (mut right, mut total) = (0usize, 0usize);
    for (sp, x, want, text) in &f.ev {
        if *sp != split {
            continue;
        }
        let got = decide(
            cfg.rule,
            probe.predict(x),
            fitted.council.as_ref(),
            &e.tok,
            text,
            &all,
            core_vote(cfg.rule, text, &all),
        );
        total += 1;
        right += usize::from(got == *want);
    }
    Some((right, total))
}

/// Where the adopted routing configuration lives.
///
/// It lived nowhere. `search_report` fitted `e.probe` and `e.council` in memory
/// and wrote nothing, so a search that spent part of the test budget to choose
/// a configuration forgot the choice at the next boot -- the machine improved
/// itself and lost it by morning. Persisting it is what makes the search a
/// self-modification rather than a report.
pub const CONFIG: &str = "/ai/config";

pub fn save_config(cfg: Config) -> bool {
    let mut s = String::from("config 1\nlambda ");
    // Tenths, matching how every message about lambda already prints it.
    push_u32_local(&mut s, (cfg.lambda * 10.0) as u32);
    s.push_str("\nrule ");
    s.push_str(cfg.rule.name());
    s.push('\n');
    unsafe { *IN_FORCE.get() = Some(cfg) };
    crate::sysbox::write_text(CONFIG, &s)
}

/// The configuration in force, cached.
///
/// Read on the routing path, which runs for every decision, so the blob is
/// read once and remembered rather than parsed per route. `save_config`
/// refreshes it, so adopting a configuration takes effect immediately rather
/// than at the next boot.
///
/// A failed load leaves the cache empty rather than caching the failure, so a
/// configuration written after the first miss is still picked up. Editing the
/// blob by hand *after* a successful load is the one case that goes unseen
/// until the next boot; `search` is the supported way to change it.
static IN_FORCE: crate::sync::Racy<Option<Config>> = crate::sync::Racy::new(None);

/// What the router should do, defaulting to what it did before it could be
/// told: majority of three, ties to the probe.
///
/// The default matters. A machine that has never run `search` must behave
/// exactly as it did, or this change would silently alter routing on every
/// system that never asked for anything.
pub fn rule_in_force() -> Rule {
    if let Some(cfg) = unsafe { *IN_FORCE.get() } {
        return cfg.rule;
    }
    match load_config() {
        Some(cfg) => cfg.rule,
        None => Rule::Majority,
    }
}

/// The lambda a bare `fit` should use: whatever was last adopted, or 1.0.
pub fn default_lambda() -> f32 {
    if let Some(cfg) = unsafe { *IN_FORCE.get() } {
        return cfg.lambda;
    }
    load_config().map(|c| c.lambda).unwrap_or(1.0)
}

pub fn load_config() -> Option<Config> {
    let bytes = crate::sysbox::read_blob(CONFIG)?;
    let text = core::str::from_utf8(&bytes).ok()?;
    let mut lambda = None;
    let mut rule = None;
    for line in text.lines() {
        let mut w = line.split_whitespace();
        match (w.next(), w.next()) {
            (Some("lambda"), Some(v)) => {
                lambda = v.parse::<u32>().ok().map(|t| t as f32 / 10.0)
            }
            (Some("rule"), Some("probe")) => rule = Some(Rule::ProbeOnly),
            (Some("rule"), Some("majority")) => rule = Some(Rule::Majority),
            (Some("rule"), Some("lexical")) => rule = Some(Rule::LexicalOnly),
            (Some("rule"), Some("withcore")) => rule = Some(Rule::WithCore),
            _ => {}
        }
    }
    let cfg = Config { lambda: lambda?, rule: rule? };
    unsafe { *IN_FORCE.get() = Some(cfg) };
    Some(cfg)
}

fn push_u32_local(s: &mut String, v: u32) {
    if v >= 10 {
        push_u32_local(s, v / 10);
    }
    s.push((b'0' + (v % 10) as u8) as char);
}

// There is deliberately no `apply_stored_config`. One was written and taken
// out: refitting from the stored configuration needs features, and a feature
// is a forward pass, so honouring the configuration at boot would have added
// several hundred of them to every start. The probe is fitted on demand
// anyway -- `probe: None` at construction, `fit` or `search` fills it -- so
// persistence is carried by `default_lambda` and `rule_in_force` instead,
// which cost a blob read once and nothing thereafter.

/// The configuration round-trip, and the default that protects every machine
/// that has never searched.
///
/// Checked without a namespace, which the boot selftests run before: the
/// parse is exercised against text rather than against a file, and the default
/// is exercised by asking for it with no cache and no blob.
pub fn config_selftest() -> bool {
    // The default is the old hardcoded behaviour. If this ever stops being
    // `Majority`, every machine that never ran `search` silently starts
    // routing differently.
    if rule_in_force() != Rule::Majority {
        return false;
    }
    if default_lambda() != 1.0 {
        return false;
    }

    // Every rule name survives being written and read. A name that renders but
    // does not parse would make an adopted configuration fall back to the
    // default at the next boot, silently -- which is the failure this whole
    // change is about, arriving one layer down.
    for rule in [Rule::ProbeOnly, Rule::Majority, Rule::LexicalOnly, Rule::WithCore] {
        let cfg = Config { lambda: 1.0, rule };
        let mut text = String::from("config 1\nlambda 10\nrule ");
        text.push_str(cfg.rule.name());
        text.push('\n');
        let mut got = None;
        for line in text.lines() {
            let mut w = line.split_whitespace();
            if let (Some("rule"), Some(v)) = (w.next(), w.next()) {
                got = match v {
                    "probe" => Some(Rule::ProbeOnly),
                    "majority" => Some(Rule::Majority),
                    "lexical" => Some(Rule::LexicalOnly),
                    "withcore" => Some(Rule::WithCore),
                    _ => None,
                };
            }
        }
        if got != Some(rule) {
            return false;
        }
    }

    // Every rule name is distinct, or two configurations would be one.
    let names = [
        Rule::ProbeOnly.name(),
        Rule::Majority.name(),
        Rule::LexicalOnly.name(),
        Rule::WithCore.name(),
    ];
    for (i, a) in names.iter().enumerate() {
        if names.iter().skip(i + 1).any(|b| b == a) {
            return false;
        }
    }
    true
}

/// What a candidate core earned.
pub struct CoreVerdict {
    pub n: usize,
    /// How many of the `n` items the council *with* the core answered right.
    ///
    /// The paired counts say whether the core is an improvement; this says
    /// what the thing actually scores. They answer different questions and a
    /// certificate needs both -- "repairs four more than it breaks" is the
    /// selection criterion, "gets 71% right" is the number a reader wants.
    pub correct: usize,
    pub fixed: usize,
    pub broke: usize,
    pub chi: f32,
    pub declined: usize,
    pub disagreed: usize,
    pub worst_steps: u64,
    pub total_steps: u64,
    pub j1: bool,
    pub j5: bool,
    pub j6: bool,

    /// How much room a core has on this slice, measured while judging one.
    ///
    /// These three say nothing about the candidate and everything about
    /// whether *any* candidate could win, which turns out to be the question
    /// worth asking first. `decide` under `Rule::WithCore` carries a class
    /// only when two of `[lexical, character, core]` agree on it against the
    /// probe -- so when the two counters already agree, they carry it
    /// themselves and the core is arithmetically inert. A core's entire
    /// influence is the items where they *split*.
    ///
    ///   contested    the two counters disagree. Everywhere else is inert.
    ///   recoverable  contested, and one of them is actually right. A core
    ///                cannot introduce a third class, so this is the hard
    ///                ceiling on what any core could ever repair.
    ///   prize        recoverable, and the probe is wrong. This is `fixed`
    ///                for a core that gets every one of them right, which is
    ///                the honest upper bound on the judge's own statistic.
    ///
    /// Counted here because the loop already computes every term. Measuring it
    /// separately would be a second pass over the corpus to learn something
    /// this one knew.
    pub contested: usize,
    pub recoverable: usize,
    pub prize: usize,
}

impl CoreVerdict {
    pub fn passed(&self) -> bool {
        self.j1 && self.j5 && self.j6
    }
}

/// How much of the vote budget a core may actually use.
///
/// The budget stops a runaway; this is the far lower bar a core has to clear
/// to be worth having in the decision path at all. A core that needs most of
/// its ceiling for one short string will need all of it on a bad day, and the
/// bad day is a routing decision that stalls.
const CORE_STEP_CEILING: u64 = super::voter::VOTE_BUDGET / 4;

/// Measure a candidate core against the validation slice.
///
/// Paired against the same council without it, on the same items, which is the
/// only comparison that means anything -- two accuracy percentages over
/// different runs would differ by noise and by the core, with no way to tell
/// which. The statistic is `godel::mcnemar`, shared so a second judge cannot
/// grow a second idea of significance.
///
/// Validation, never test. A core is *selected* here, and a slice you select
/// on is a slice you have fitted.
pub fn core_bench(hash: &[u8; 32]) -> Option<Result<CoreVerdict, String>> {
    with_engine(|e| core_bench_in(e, hash, VALIDATION))
}

/// The validation slice, and the test slice, by their numbers in `split_of`.
pub const VALIDATION: u8 = 1;
pub const TEST: u8 = 2;

/// The same measurement, on an engine the caller is already holding.
///
/// Split out because `core_bench` claimed the engine itself, and
/// `godel::trial_core` -- which is called from inside `with_engine` -- then
/// re-entered it. `with_engine` hands the same task a second `&mut Engine`
/// rather than refusing, so two live mutable references to one engine existed
/// at once. That is undefined behaviour and not merely a race: `&mut` carries
/// `noalias`, so the compiler is entitled to keep reads of `*e` across this
/// call in registers, and `ensure_head` right after it reads the adapter
/// tensors to decide what to record. The lineage could name a state the
/// machine was not in.
///
/// Taking the engine as an argument makes the borrow the caller's, which is
/// what it always was in fact.
pub fn core_bench_in(
    e: &mut super::Engine,
    hash: &[u8; 32],
    split_wanted: u8,
) -> Result<CoreVerdict, String> {
    let core = super::voter::load(hash)?;
    core_bench_core(e, &core, split_wanted)
}

/// How much room a core has on a slice, without needing a core to ask with.
///
/// Measured by judging one that never answers. A core that always declines
/// changes no decision, so every paired count comes back zero and what is left
/// is the census: how often the two counters split, how often one of them is
/// right when they do, and how often the probe is wrong there. That is the
/// ceiling on `fixed` for any candidate whatsoever.
///
/// Written this way rather than as a second loop because a second loop is a
/// second definition of "contested", and the two would eventually disagree
/// about the number the whole question turns on.
pub fn core_census(e: &mut super::Engine, split_wanted: u8) -> Result<CoreVerdict, String> {
    let src = "fn vote(text: str, allowed: list): int { return -1 }\n";
    let h = crate::store::sha256::hash(src.as_bytes());
    let core = super::voter::parse(&h, src)?;
    core_bench_core(e, &core, split_wanted)
}

/// The last census any bench measured, or `None` if none has run this boot.
///
/// Kept so the question "could a core win here at all" can be answered without
/// paying for another pass over the corpus. Deliberately not persisted: it is
/// a measurement of a fitted probe and a corpus that both change, and a stale
/// answer to this question is worse than no answer.
static LAST_PRIZE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);

/// What that number was measured against.
///
/// Not persisting it across boots was never the hazard -- the corpus and the
/// split both change *while the machine is up*. `teach` appends an example,
/// `teach bundle` rewrites `/ai/train.split`, and `vocab::splits` re-reads the
/// file on every call, so a prize measured an hour ago can describe a slice
/// that no longer exists. `author_core` refuses to compose when the prize is
/// below the bar, so a stale low number is not a stale report: it is the
/// nightly loop switched off for the rest of the boot, silently, on evidence
/// about a corpus the operator has since grown. The number now carries its own
/// provenance and goes back to "unknown" when that stops matching, which sends
/// the loop back to measuring rather than to a confident wrong answer.
static LAST_KEY: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// A fingerprint of the corpus and the split the census would be measured on.
///
/// Zero means "cannot tell", which is treated as never matching: with no
/// corpus to hash there is nothing a remembered prize could still be true of.
fn census_key() -> u64 {
    let Some(h) = crate::sysbox::hash_of(super::vocab::CORPUS) else { return 0 };
    let mut k = u64::from_le_bytes([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]]);
    let (train, val_end, n) = super::vocab::splits();
    // The boundaries are part of the question, not just the content: the same
    // examples partitioned differently give a different census.
    for part in [train, val_end, n] {
        k = k.rotate_left(17) ^ (part as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    // Never collide with the "nothing measured" sentinel.
    if k == 0 {
        1
    } else {
        k
    }
}

pub fn last_prize() -> Option<usize> {
    let n = match LAST_PRIZE.load(core::sync::atomic::Ordering::Relaxed) {
        u32::MAX => return None,
        n => n as usize,
    };
    let key = LAST_KEY.load(core::sync::atomic::Ordering::Relaxed);
    if key == 0 || key != census_key() {
        return None;
    }
    Some(n)
}

/// What the cue pool could do, if the machine chose from it perfectly.
///
/// The producer's ceiling, the way `core_prize` is the judge's. Counting cues
/// says how much the filter admits; this says how much of what it admits is
/// worth anything -- and those are different questions, because a filter can
/// double the table with words that never fire on a held-out item.
pub struct CueVerdict {
    pub cues: usize,
    /// Would repair at least one validation item and break none.
    pub usable: usize,
    /// Would break at least one, whatever else it does.
    pub harmful: usize,
    /// Changes no decision either way. The common case, and not a failure:
    /// under `WithCore` a core that answers where the counters agree is
    /// arithmetically inert.
    pub inert: usize,
    /// The best single rule in the pool, and what it would score alone.
    pub best_fixed: usize,
    pub best_broke: usize,
    pub best: String,
    /// Items repaired by *some* rule in the pool -- the ceiling a multi-rule
    /// core could approach, as opposed to what one rule reaches.
    pub reach: usize,
}

/// Measure the cue pool against the validation slice.
///
/// **This reads validation, and that is a real cost.** It exists to answer
/// whether loosening the producer's filter admits cues that can do anything,
/// which cannot be answered from the training slice: a cue's value is whether
/// it fires where the counters split and the probe is wrong, and that is a
/// property of the slice being judged. So it is a diagnostic an operator runs,
/// never something the producer or the night loop consults -- choosing the
/// filter by what scores here would be fitting the slice J1 is computed on,
/// and the difference between measuring a thing and selecting on it is the
/// whole discipline of this module.
///
/// Corroborating is cached per item rather than per cue: the counters do not
/// depend on the candidate, so a hundred cues cost one pass over the slice and
/// then a substring test each.
pub fn cue_oracle(
    e: &mut super::Engine,
    names: &[String],
    purity: u32,
    min_uses: u32,
) -> Result<CueVerdict, String> {
    let table = super::voter::cue_table_at(names, purity, min_uses);
    let Some(f) = featurise(e) else {
        return Err(String::from("no corpus to judge against"));
    };
    let lambda = default_lambda();
    let fitted = fit_all(e, &f, &[lambda]);
    let Some((_, probe)) = fitted.probes.first() else {
        return Err(String::from("the probe would not fit"));
    };
    let all: Vec<usize> = (0..f.classes).collect();

    // (lowered text, wanted class, what the probe said, what the counters said)
    let mut items: Vec<(String, usize, usize, Option<(usize, usize)>)> = Vec::new();
    for (split, x, want, text) in &f.ev {
        if *split != VALIDATION {
            continue;
        }
        let p = probe.predict(x);
        let pair = fitted.council.as_ref().and_then(|c| c.corroborate(text, &e.tok, &all));
        let lowered: String = text.chars().map(|c| c.to_ascii_lowercase()).collect();
        items.push((lowered, *want, p, pair));
    }
    if items.is_empty() {
        return Err(String::from("nothing in validation to judge against"));
    }

    let mut v = CueVerdict {
        cues: table.len(),
        usable: 0,
        harmful: 0,
        inert: 0,
        best_fixed: 0,
        best_broke: 0,
        best: String::new(),
        reach: 0,
    };
    // Which items *any* rule repairs, so the pool's ceiling is distinguishable
    // from its best single rule.
    let mut reached = alloc::vec![false; items.len()];

    for (word, class, _) in &table {
        let (mut fixed, mut broke) = (0usize, 0usize);
        for (i, (text, want, p, pair)) in items.iter().enumerate() {
            let says = if text.contains(word.as_str()) { Some(*class) } else { None };
            if says.is_none() {
                continue;
            }
            let without = decide_with(Rule::Majority, *p, *pair, None);
            let with = decide_with(Rule::WithCore, *p, *pair, says);
            match (without == *want, with == *want) {
                (false, true) => {
                    fixed += 1;
                    reached[i] = true;
                }
                (true, false) => broke += 1,
                _ => {}
            }
        }
        if broke > 0 {
            v.harmful += 1;
        } else if fixed > 0 {
            v.usable += 1;
        } else {
            v.inert += 1;
        }
        // Best by net repair, so a rule that fixes six and breaks four does
        // not outrank one that fixes three cleanly.
        let net = fixed as i32 - broke as i32;
        let best_net = v.best_fixed as i32 - v.best_broke as i32;
        if net > best_net {
            v.best_fixed = fixed;
            v.best_broke = broke;
            v.best = word.clone();
        }
    }
    v.reach = reached.iter().filter(|r| **r).count();
    Ok(v)
}

pub fn core_bench_core(
    e: &mut super::Engine,
    core: &super::voter::Core,
    split_wanted: u8,
) -> Result<CoreVerdict, String> {
    {
        let Some(f) = featurise(e) else {
            return Err(String::from("no corpus to judge against"));
        };
        let lambda = default_lambda();
        let fitted = fit_all(e, &f, &[lambda]);
        let Some((_, probe)) = fitted.probes.first() else {
            return Err(String::from("the probe would not fit"));
        };
        let all: Vec<usize> = (0..f.classes).collect();

        let mut v = CoreVerdict {
            n: 0,
            correct: 0,
            fixed: 0,
            broke: 0,
            chi: 0.0,
            declined: 0,
            disagreed: 0,
            worst_steps: 0,
            total_steps: 0,
            j1: false,
            j5: false,
            j6: false,
            contested: 0,
            recoverable: 0,
            prize: 0,
        };

        for (split, x, want, text) in &f.ev {
            if *split != split_wanted {
                continue;
            }
            let p = probe.predict(x);
            let (says, steps) = core.vote(text, &all);
            v.total_steps += steps;
            v.worst_steps = v.worst_steps.max(steps);
            if says.is_none() {
                v.declined += 1;
            }

            // What the counters said, asked once and used three times: for the
            // paired comparison, for the census, and for J6.
            let pair = fitted.council.as_ref().and_then(|c| c.corroborate(text, &e.tok, &all));
            let without = decide_with(Rule::Majority, p, pair, None);
            let with = decide_with(Rule::WithCore, p, pair, says);

            if let Some((l, ch)) = pair {
                if l != ch {
                    v.contested += 1;
                    if l == *want || ch == *want {
                        v.recoverable += 1;
                        if p != *want {
                            v.prize += 1;
                        }
                    }
                }
            }

            // Independence is measured against the lexical core, which is the
            // one a written core is most likely to reinvent: both read the
            // words. A voter that only ever repeats an existing one adds a
            // vote and no information, and inflates agreement -- which is the
            // signal this council is built on -- without earning it.
            //
            // J6 catches redundancy, not uselessness, and the two are not the
            // same. A core that answers the same class every time differs from
            // lexical almost always and sails through here; it is J1 that
            // vetoes it, having found it repairs nothing. Neither judge is
            // sufficient alone and neither is trying to be -- the first
            // measured core to reach this gate passed J5 and J6 and was
            // refused by J1, which is the pair working as intended.
            if let (Some(k), Some((l, _))) = (says, pair) {
                if k != l {
                    v.disagreed += 1;
                }
            }

            v.n += 1;
            if with == *want {
                v.correct += 1;
            }
            match (without == *want, with == *want) {
                (false, true) => v.fixed += 1,
                (true, false) => v.broke += 1,
                _ => {}
            }
        }

        if v.n == 0 {
            return Err(String::from("nothing in that slice to judge against"));
        }
        v.chi = super::godel::mcnemar(v.broke, v.fixed);
        // J1, the same shape the adapter judge uses: a net repair above the
        // floor and beyond the noise.
        v.j1 = v.fixed > v.broke
            && v.fixed - v.broke >= super::godel::MIN_FIXED
            && v.chi >= super::godel::MCNEMAR_95;
        // J5, cost.
        v.j5 = v.worst_steps <= CORE_STEP_CEILING;
        // J6, independence. A core that never differs from lexical is lexical.
        v.j6 = v.disagreed > 0 && v.declined < v.n;
        // Only the validation census answers "could a core win", because that
        // is the slice J1 is computed on. A test-slice census is a different
        // number about a different question.
        if split_wanted == VALIDATION {
            // The key first, so a reader that sees a prize always sees one
            // that has been stamped -- the reverse order leaves a window in
            // which the new number is checked against the old corpus.
            LAST_KEY.store(census_key(), core::sync::atomic::Ordering::Relaxed);
            LAST_PRIZE.store(v.prize as u32, core::sync::atomic::Ordering::Relaxed);
        }
        Ok(v)
    }
}

/// Judge a candidate and wire it in if every judge passes.
pub fn core_report(hash: &[u8; 32], install: bool) {
    console::set_color(YELLOW);
    kprintln!("[core] {}", &super::voter::hex(hash)[..8]);
    console::set_color(LTGRAY);

    let out = core_bench(hash);
    let v = match out {
        None => {
            kprintln!("  {}", super::engine_refusal());
            return;
        }
        Some(Err(e)) => {
            console::set_color(LTRED);
            kprintln!("  {}", e);
            console::set_color(LTGRAY);
            return;
        }
        Some(Ok(v)) => v,
    };

    let mark = |p: bool| if p { "pass" } else { "VETO" };
    kprintln!("  judged on {} validation items", v.n);
    // The room, before the verdict. A reader who sees a veto without this
    // concludes the candidate was poor; more often the slice had nothing in
    // it for any candidate to win.
    let need = super::godel::clean_fixes_needed();
    kprintln!(
        "  room         {} contested, {} recoverable, {} within reach ({} needed)",
        v.contested,
        v.recoverable,
        v.prize,
        need
    );
    if v.prize < need {
        console::set_color(YELLOW);
        kprintln!("  no core of any construction can clear J1 on this slice");
        console::set_color(LTGRAY);
    }
    kprintln!(
        "  J1 margin    {}  {} repaired, {} broken, chi {}",
        mark(v.j1),
        v.fixed,
        v.broke,
        (v.chi * 100.0) as u32 as f32 / 100.0
    );
    kprintln!(
        "  J5 cost      {}  worst {} steps of {} allowed, {} mean",
        mark(v.j5),
        v.worst_steps,
        CORE_STEP_CEILING,
        v.total_steps / v.n as u64
    );
    kprintln!(
        "  J6 apart     {}  differs from lexical on {}, declined {}",
        mark(v.j6),
        v.disagreed,
        v.declined
    );

    if !v.passed() {
        console::set_color(YELLOW);
        kprintln!("  not installed -- a core joins the council on evidence or not at all");
        console::set_color(LTGRAY);
        return;
    }
    if !install {
        console::set_color(LTGREEN);
        kprintln!("  every judge passes -- 'core install {}' to wire it in", &super::voter::hex(hash)[..8]);
        console::set_color(LTGRAY);
        return;
    }
    if super::voter::install(hash) {
        console::set_color(LTGREEN);
        kprintln!("  installed. 'search' will now compare withcore against the rest");
        console::set_color(LTGRAY);
    } else {
        console::set_color(LTRED);
        kprintln!("  could not write the pointer -- nothing changed");
        console::set_color(LTGRAY);
    }
}

/// Search the configuration space, adopt the winner, then report on test.
pub fn search_report() {
    console::set_color(YELLOW);
    kprintln!("[search]");
    console::set_color(LTGRAY);

    let lambdas = [0.1f32, 1.0, 10.0];
    let rules = [Rule::ProbeOnly, Rule::Majority, Rule::LexicalOnly, Rule::WithCore];

    let t0 = crate::time::rdtsc();
    let outcome = with_engine(|e| {
        // Once, not once per configuration. See `Featurised`.
        let f = featurise(e)?;
        let fitted = fit_all(e, &f, &lambdas);
        let mut best: Option<(Config, usize, usize)> = None;
        let mut tried = 0usize;

        for &lambda in lambdas.iter() {
            for &rule in rules.iter() {
                let cfg = Config { lambda, rule };
                let Some((ok, n)) = score_cfg(e, &f, &fitted, cfg, 1) else { continue };
                tried += 1;
                console::set_color(LTGRAY);
                kprintln!(
                    "    lambda {:>4}  {:8}  val {}%",
                    (lambda * 10.0) as u32,
                    rule.name(),
                    pct(ok, n)
                );
                let better = best
                    .map(|(_, bok, bn)| ok * bn > bok * n)
                    .unwrap_or(true);
                if better {
                    best = Some((cfg, ok, n));
                }
            }
        }

        let (cfg, vok, vn) = best?;
        // Read once, after the choice is already made -- and spent against the
        // same budget the self-modification loop spends.
        //
        // It was not. `evaluate(e, cfg, 2)` read the test slice directly, so
        // every `search` burned a read that nobody counted, while `godel`
        // carefully budgeted its own three. A loop that improves itself reads
        // the held-out set forever unless somebody counts, and counting in one
        // of the two places that read it is not counting.
        let reads = super::godel::spend_test_read();
        let (tok_, tn) = score_cfg(e, &f, &fitted, cfg, 2)?;

        // Adopt: install the winner and write it down, so it survives a boot.
        let texts: Vec<&str> = f.texts.iter().map(|t| t.as_str()).collect();
        e.probe = super::probe::Probe::fit(&f.xs, &f.ys, f.classes, cfg.lambda);
        e.council = super::council::Council::fit(&texts, &f.ys, f.classes, &e.tok);
        let saved = save_config(cfg);

        Some((cfg, vok, vn, tok_, tn, tried, reads, saved))
    })
    .flatten();

    let Some((cfg, vok, vn, tok_, tn, tried, reads, saved)) = outcome else {
        console::set_color(LTRED);
        kprintln!("  no model, or no corpus to search against");
        console::set_color(LTGRAY);
        return;
    };
    let elapsed = crate::time::rdtsc() - t0;

    console::set_color(LTGREEN);
    kprintln!(
        "\n  adopted: lambda {}/10, rule {}",
        (cfg.lambda * 10.0) as u32,
        cfg.rule.name()
    );
    console::set_color(LTGRAY);
    if !saved {
        console::set_color(LTRED);
        kprintln!("  COULD NOT WRITE {} -- this choice dies at the next boot", CONFIG);
        console::set_color(LTGRAY);
    }
    kprintln!("  {} configurations tried, chosen on {} validation items", tried, vn);
    kprintln!("  validation {}%  (spent -- selected on, so optimistic)", pct(vok, vn));
    let (_, cap, _) = super::godel::test_status();
    if reads <= cap {
        console::set_color(LTCYAN);
        kprintln!(
            "  test       {}%  ({} items, read {} of {})",
            pct(tok_, tn),
            tn,
            reads,
            cap
        );
    } else {
        console::set_color(YELLOW);
        kprintln!(
            "  test       {}%  ({} items) -- STALE, read {} of {}, do not quote",
            pct(tok_, tn),
            tn,
            reads,
            cap
        );
    }
    console::set_color(LTGRAY);
    let mhz = crate::time::tsc_mhz();
    if mhz > 0 {
        kprintln!("  searched in {} ms", elapsed / mhz / 1000);
    }
    kprintln!("  a configuration is adopted only when measured better, never argued better");
}


