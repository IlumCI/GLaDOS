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
/// to *understand* rather than merely be constrained. Everything else in this
/// module is guaranteed by construction; this is the part that is merely
/// hoped for. It is measurably the weak link -- the probe in `probe.rs`, which
/// never reads this prompt at all, routes better than decoding against it.
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
///
/// Only reachable with `feature hidden`; the default pooled features never
/// touch the model at all.
fn invalidate_conversation(e: &mut super::Engine) {
    e.pos = 0;
    e.last_token = super::tokenizer::BOS;
}

/// Shorter than the grammar prompt: with applet tokens there is no need to
/// list the tools, because the candidate set is imposed by the scoring rather
/// than described in words.
fn prompt_for_task(task: &str) -> String {
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
    let seed = super::corpus::SEED.len();
    i >= super::corpus::SEED_TRAIN && i < seed
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

        // Majority, ties to the probe.
        //
        // The first version had the probe answer alone and the cores only
        // corroborate, on the grounds that a *product* of the three scored
        // 76.9% against the probe's 77.8%. That reasoning did not transfer:
        // a majority vote is a different rule, and it scores 80.6%. The
        // difference lives in the eight items where the probe stands alone
        // against the other two, where the probe is right 25% of the time and
        // the pair 62.5% -- so being outvoted is genuine evidence, and the
        // measurement that dismissed it was of something else.
        let winner = if lexical == character && lexical != probe_idx {
            lexical
        } else {
            probe_idx
        };

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
}

impl Rule {
    pub fn name(&self) -> &'static str {
        match self {
            Rule::ProbeOnly => "probe",
            Rule::Majority => "majority",
            Rule::LexicalOnly => "lexical",
        }
    }
}

fn split_of(i: usize) -> u8 {
    let seed = super::corpus::SEED.len();
    if i < super::corpus::SEED_TRAIN {
        0 // train
    } else if i < super::corpus::SEED_VAL_END {
        1 // validation
    } else if i < seed {
        2 // test
    } else {
        0 // anything `teach` appended trains
    }
}

/// Score one configuration on a split, having fitted it on train.
fn evaluate(e: &mut super::Engine, cfg: Config, split: u8) -> Option<(usize, usize)> {
    let examples = vocab::examples();
    let classes = e.head.len();

    let mut xs: Vec<Vec<f32>> = Vec::new();
    let mut ys: Vec<usize> = Vec::new();
    let mut texts: Vec<&str> = Vec::new();
    for (i, ex) in examples.iter().enumerate() {
        if split_of(i) != 0 {
            continue;
        }
        let Some(y) = e.head.index_of(&ex.applet) else { continue };
        let Some(x) = feature(e, &ex.task) else { continue };
        xs.push(x);
        ys.push(y);
        texts.push(&ex.task);
    }
    if xs.is_empty() {
        return None;
    }

    let probe = super::probe::Probe::fit(&xs, &ys, classes, cfg.lambda)?;
    let council = super::council::Council::fit(&texts, &ys, classes, &e.tok);
    let all: Vec<usize> = (0..classes).collect();

    let (mut right, mut total) = (0usize, 0usize);
    for (i, ex) in examples.iter().enumerate() {
        if split_of(i) != split {
            continue;
        }
        let Some(want) = e.head.index_of(&ex.applet) else { continue };
        let Some(x) = feature(e, &ex.task) else { continue };
        let p = probe.predict(&x);

        let got = match (cfg.rule, council.as_ref()) {
            (Rule::ProbeOnly, _) | (_, None) => p,
            (Rule::Majority, Some(c)) => match c.corroborate(&ex.task, &e.tok, &all) {
                Some((l, ch)) if l == ch && l != p => l,
                _ => p,
            },
            (Rule::LexicalOnly, Some(c)) => match c.corroborate(&ex.task, &e.tok, &all) {
                Some((l, _)) => l,
                None => p,
            },
        };
        total += 1;
        right += usize::from(got == want);
    }
    Some((right, total))
}

/// Search the configuration space, adopt the winner, then report on test.
pub fn search_report() {
    console::set_color(YELLOW);
    kprintln!("[search]");
    console::set_color(LTGRAY);

    let lambdas = [0.1f32, 1.0, 10.0];
    let rules = [Rule::ProbeOnly, Rule::Majority, Rule::LexicalOnly];

    let t0 = crate::time::rdtsc();
    let outcome = with_engine(|e| {
        let mut best: Option<(Config, usize, usize)> = None;
        let mut tried = 0usize;

        for &lambda in lambdas.iter() {
            for &rule in rules.iter() {
                let cfg = Config { lambda, rule };
                let Some((ok, n)) = evaluate(e, cfg, 1) else { continue };
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
        // Read once, after the choice is already made.
        let (tok_, tn) = evaluate(e, cfg, 2)?;

        // Adopt: refit on train and install, so the engine is left holding the
        // configuration that won rather than whichever was tried last.
        let examples = vocab::examples();
        let classes = e.head.len();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut texts: Vec<&str> = Vec::new();
        for (i, ex) in examples.iter().enumerate() {
            if split_of(i) != 0 {
                continue;
            }
            let Some(y) = e.head.index_of(&ex.applet) else { continue };
            let Some(x) = feature(e, &ex.task) else { continue };
            xs.push(x);
            ys.push(y);
            texts.push(&ex.task);
        }
        e.probe = super::probe::Probe::fit(&xs, &ys, classes, cfg.lambda);
        e.council = super::council::Council::fit(&texts, &ys, classes, &e.tok);

        Some((cfg, vok, vn, tok_, tn, tried))
    })
    .flatten();

    let Some((cfg, vok, vn, tok_, tn, tried)) = outcome else {
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
    kprintln!("  {} configurations tried, chosen on {} validation items", tried, vn);
    kprintln!("  validation {}%  (spent -- selected on, so optimistic)", pct(vok, vn));
    console::set_color(LTCYAN);
    kprintln!("  test       {}%  ({} items, read once)", pct(tok_, tn), tn);
    console::set_color(LTGRAY);
    let mhz = crate::time::tsc_mhz();
    if mhz > 0 {
        kprintln!("  searched in {} ms", elapsed / mhz / 1000);
    }
    kprintln!("  a configuration is adopted only when measured better, never argued better");
}
