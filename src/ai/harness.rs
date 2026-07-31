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

use super::vocab::{self, Head};

fn allowed_indices(head: &Head, trust: Trust) -> Vec<usize> {
    sysbox::APPLETS
        .iter()
        .enumerate()
        .filter(|(_, a)| trust.admits(a))
        .filter_map(|(i, _)| head.index_of(sysbox::APPLETS[i].name))
        .collect()
}

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
    Some(e.state.hidden().to_vec())
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

pub fn choose_native(task: &str, trust: Trust, temperature: f32) -> Option<Choice> {
    with_engine(|e| {
        let allowed = allowed_indices(&e.head, trust);
        if allowed.is_empty() {
            return None;
        }
        let x = feature(e, task)?;
        let logits = e.head.logits(&x, &allowed);
        let slot = sample::sample_among(
            &logits,
            &(0..logits.len() as u32).collect::<Vec<u32>>(),
            temperature,
            0.0,
            &mut e.rng,
        )?;
        let name = e.head.name(allowed[slot]);
        let mutates = sysbox::APPLETS
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.mutates)
            .unwrap_or(false);
        Some(Choice { applet: name, mutates, steps: 1 })
    })?
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

/// Every fourth example is held out. Reporting only training accuracy would
/// measure memorisation, which a per-applet delta can always achieve and which
/// says nothing about whether anything was learned.
fn is_held_out(i: usize) -> bool {
    i % 4 == 3
}

fn accuracy(e: &mut super::Engine, examples: &[vocab::Example], held: bool) -> f32 {
    let all: Vec<usize> = (0..e.head.len()).collect();
    let mut right = 0usize;
    let mut total = 0usize;
    for (i, ex) in examples.iter().enumerate() {
        if is_held_out(i) != held {
            continue;
        }
        let Some(target) = e.head.index_of(&ex.applet) else { continue };
        let Some(x) = feature(e, &ex.task) else { continue };
        let logits = e.head.logits(&x, &all);
        let mut best = 0usize;
        for (j, l) in logits.iter().enumerate() {
            if *l > logits[best] {
                best = j;
            }
        }
        total += 1;
        if all[best] == target {
            right += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        right as f32 / total as f32
    }
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
