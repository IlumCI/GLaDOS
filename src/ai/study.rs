//! Learning one field, then another, and checking what the second cost the
//! first.
//!
//! The machine can already be trained and can already be scored. What it has
//! never been asked is the question that separates learning from memorising:
//! after studying a second field, is it still any good at the first?
//!
//! So the corpus is partitioned into domains -- families of applets that form
//! a coherent thing to be good at -- and the probe is fitted on a growing
//! prefix of them while every domain is scored after every stage. The result
//! is a matrix, and each part of it answers a different question:
//!
//! ```text
//!   the diagonal      did studying this field teach it this field
//!   above the diagonal  what a field it has not studied scores anyway
//!   down a column     what later fields cost an earlier one
//! ```
//!
//! **Two honest limits, because the second one changes what the matrix
//! means.**
//!
//! The probe is a closed-form ridge fit, refitted from scratch at every stage
//! rather than updated. So it *cannot* catastrophically forget: nothing is
//! overwritten, and a column that drops is reporting **interference** --
//! domains competing for capacity in one joint solution -- rather than
//! forgetting. That is a real effect and worth measuring, and it is not the
//! same effect. True sequential forgetting needs a path that updates in place,
//! which here is the QDoRA adapter, and that is a second experiment rather
//! than a flag on this one.
//!
//! And "field" here means a family of *applets*, not a body of knowledge. This
//! measures whether the routing layer generalises across capability families.
//! It says nothing about whether the model knows any bioinformatics, and a
//! 0.6B checkpoint's parametric knowledge is capacity-bound whatever this
//! prints -- `design/benchmarks.md` has the numbers.
//!
//! **Why the features are cached.** The base model is frozen, so a hidden
//! state is a constant. Computing them once and refitting per stage makes an
//! N-stage curriculum cost one pass over the corpus instead of N. That is the
//! same property the whole judging apparatus rests on, spent here.

use super::vocab;
use alloc::string::String;
use alloc::vec::Vec;

/// A field: a group of applets that is a coherent thing to be good at.
///
/// Grouped by what the operator is *doing* rather than by what the code
/// touches, because the question is whether phrasings generalise, and a person
/// asking to move a file and a person asking to delete one are speaking the
/// same dialect.
pub struct Domain {
    pub name: &'static str,
    pub applets: &'static [&'static str],
}

pub const DOMAINS: &[Domain] = &[
    Domain { name: "navigate", applets: &["ls", "cd", "pwd", "tree", "find"] },
    Domain { name: "inspect", applets: &["cat", "stat", "hash", "same", "diff", "du", "fsck"] },
    Domain { name: "mutate", applets: &["mkdir", "write", "rm", "mv", "cp"] },
    Domain { name: "history", applets: &["snap", "back", "snaps"] },
    Domain { name: "meta", applets: &["sysbox", "run", "remember"] },
    // The field that is unlike the others, which is exactly what the
    // forgetting experiment was short of. Every domain above moves bytes
    // already on the machine and shares most of its vocabulary with the
    // rest -- "file", "path", "directory" -- so "learn A then B and see what
    // A cost" was being asked of two fields a probe can hardly tell apart.
    // Reading off the network shares almost no vocabulary with any of them.
    Domain { name: "network", applets: &["fetch", "save"] },
];

/// Which domain an applet belongs to, if any.
pub fn domain_of(applet: &str) -> Option<usize> {
    DOMAINS.iter().position(|d| d.applets.contains(&applet))
}

/// One stage of the curriculum: what was trained on, and what everything
/// scored afterwards.
pub struct Row {
    /// Domains trained on, in order.
    pub trained: usize,
    /// Per domain, (right, total) over its held-out examples.
    pub scores: Vec<(usize, usize)>,
    /// How many examples the fit actually saw.
    pub fitted_on: usize,
}

/// A cached example: its features, its label, its domain, and whether it is
/// held out.
struct Cached {
    x: Vec<f32>,
    y: usize,
    domain: usize,
    held: bool,
}

/// Run the curriculum and return one row per stage.
///
/// Stages are cumulative prefixes of `DOMAINS`: first alone, then the first
/// two, and so on. Cumulative rather than sequential because a closed-form fit
/// has no memory of the previous one -- fitting on B alone and calling the
/// result "after A then B" would be measuring a model that never saw A and
/// describing it as one that forgot.
pub fn curriculum(lambda: f32, stride: usize) -> Option<Vec<Row>> {
    let examples = vocab::examples();
    if examples.is_empty() {
        return None;
    }
    // Stride, never a prefix. The splits are positional, so the first N
    // examples are all training examples and a prefix would score held-out
    // accuracy over an empty set -- the same trap `train -n` documents.
    let stride = stride.max(1);
    let (train_end, _, seed) = vocab::splits();

    // One pass over the corpus, because the base is frozen.
    let cache: Vec<Cached> = super::with_engine(|e| {
        let mut out = Vec::new();
        for (i, ex) in examples.iter().enumerate() {
            if i % stride != 0 {
                continue;
            }
            let Some(y) = e.head.index_of(&ex.applet) else { continue };
            let Some(domain) = domain_of(&ex.applet) else { continue };
            let Some(x) = super::harness::feature_for(e, &ex.task) else { continue };
            out.push(Cached { x, y, domain, held: i >= train_end && i < seed });
        }
        out
    })?;
    if cache.is_empty() {
        return None;
    }
    let classes = super::with_engine(|e| e.head.len())?;

    let mut rows = Vec::new();
    for stage in 1..=DOMAINS.len() {
        let mut tx: Vec<Vec<f32>> = Vec::new();
        let mut ty: Vec<usize> = Vec::new();
        for c in &cache {
            if !c.held && c.domain < stage {
                tx.push(c.x.clone());
                ty.push(c.y);
            }
        }
        if tx.is_empty() {
            continue;
        }
        let Some(p) = super::probe::Probe::fit(&tx, &ty, classes, lambda) else { continue };

        // Score every domain, including the ones this stage has never seen.
        // Scoring only what was trained on is how a curriculum reports itself
        // as a success while the earlier fields quietly rot.
        let mut scores = alloc::vec![(0usize, 0usize); DOMAINS.len()];
        for c in &cache {
            if !c.held {
                continue;
            }
            scores[c.domain].1 += 1;
            if p.predict(&c.x) == c.y {
                scores[c.domain].0 += 1;
            }
        }
        rows.push(Row { trained: stage, scores, fitted_on: tx.len() });
    }
    if rows.is_empty() { None } else { Some(rows) }
}

/// How many held-out examples each domain has, without touching the model.
///
/// Worth its own function because a domain with two held-out examples produces
/// a percentage that looks like a measurement and is not one, and the caller
/// should be able to say so beside the number.
pub fn held_out_counts() -> Vec<usize> {
    let examples = vocab::examples();
    let (train_end, _, seed) = vocab::splits();
    let mut out = alloc::vec![0usize; DOMAINS.len()];
    for (i, ex) in examples.iter().enumerate() {
        if i >= train_end && i < seed {
            if let Some(d) = domain_of(&ex.applet) {
                out[d] += 1;
            }
        }
    }
    out
}

/// The partition, checked against the live applet table.
///
/// Model-free and namespace-free on purpose. What can go wrong here without
/// anything failing is that somebody adds an applet and does not classify it:
/// every example of it is then silently dropped from both training and
/// scoring, and the matrix reports a smaller world with perfect confidence.
/// That is exactly the failure this project keeps finding, so it gets a check.
pub fn selftest() -> bool {
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
            crate::kprintln!("    FAIL {}", what);
        }
    };

    // Every applet the machine can reach must belong to exactly one domain.
    let mut unclassified = String::new();
    let mut duplicated = String::new();
    for a in crate::sysbox::APPLETS.iter() {
        let n = DOMAINS.iter().filter(|d| d.applets.contains(&a.name)).count();
        if n == 0 {
            unclassified.push(' ');
            unclassified.push_str(a.name);
        } else if n > 1 {
            duplicated.push(' ');
            duplicated.push_str(a.name);
        }
    }
    if !unclassified.is_empty() {
        crate::kprintln!("    unclassified applets:{}", unclassified);
    }
    if !duplicated.is_empty() {
        crate::kprintln!("    applets in two domains:{}", duplicated);
    }
    claim("every applet has a domain", unclassified.is_empty());
    claim("no applet has two", duplicated.is_empty());

    // And no domain may name an applet that does not exist, which is the same
    // mistake from the other side: it would make a domain look smaller than it
    // is with nothing to show for it.
    let mut phantom = String::new();
    for d in DOMAINS {
        for name in d.applets {
            if !crate::sysbox::APPLETS.iter().any(|a| a.name == *name) {
                phantom.push(' ');
                phantom.push_str(name);
            }
        }
    }
    if !phantom.is_empty() {
        crate::kprintln!("    domains name applets that do not exist:{}", phantom);
    }
    claim("no domain names a missing applet", phantom.is_empty());
    claim("there is more than one domain to compare", DOMAINS.len() >= 2);

    // --- rehearsal selection, driven without a model ---------------------
    //
    // Twelve decisions over three applets in three domains, two of them held
    // out, laid out so the answers are countable by hand.
    let domain_of = alloc::vec![Some(0), Some(1), Some(2)];
    let mut meta: Vec<(usize, bool)> = Vec::new();
    for k in 0..12 {
        meta.push((k % 3, k >= 10));
    }

    // Off means exactly this stage's field, which is what the run without
    // rehearsal has to be if the two are to be compared at all.
    let (keep, n) = rehearsal_keep(&meta, &domain_of, 1, 0);
    claim("stride 0 replays nothing", n == 0);
    claim(
        "and selects exactly the stage's own field",
        keep.iter().enumerate().all(|(j, k)| {
            *k == (meta[j].0 == 1 && !meta[j].1)
        }),
    );

    // The first stage has nothing behind it, whatever the stride says.
    let (_, n) = rehearsal_keep(&meta, &domain_of, 0, 4);
    claim("the first field rehearses nothing", n == 0);

    // A stride of 1 is every earlier decision -- cumulative training, and the
    // upper bound on what rehearsal can replay.
    let (all_keep, all_n) = rehearsal_keep(&meta, &domain_of, 2, 1);
    let earlier = meta.iter().filter(|(a, h)| !h && *a < 2).count();
    claim("stride 1 replays every earlier decision", all_n == earlier);

    // And a wider stride replays strictly fewer, never more.
    let (_, few) = rehearsal_keep(&meta, &domain_of, 2, 4);
    claim("a wider stride replays fewer", few > 0 && few < all_n);

    // The one that would be silent: a held-out decision reaching the fit.
    claim(
        "no held-out decision is ever selected, at any stride",
        [0usize, 1, 2, 3, 4].iter().all(|st| {
            let (k, _) = rehearsal_keep(&meta, &domain_of, 2, *st);
            k.iter().enumerate().all(|(j, sel)| !(*sel && meta[j].1))
        }),
    );

    // The stage's own field is never thinned by the stride: rehearsal adds to
    // a stage, it does not sample it.
    let own = meta.iter().enumerate().filter(|(_, (a, h))| !h && *a == 2).count();
    let got = all_keep
        .iter()
        .enumerate()
        .filter(|(j, k)| **k && meta[*j].0 == 2)
        .count();
    claim("the stage's own field is never sampled away", got == own);

    ok
}

// --- the adapter version: real sequential forgetting ---------------------

/// One stage of a sequential curriculum.
pub struct SeqRow {
    /// Which domain was just studied. `None` is the frozen base, before any
    /// study at all, which is the reference every later row is read against.
    pub studied: Option<usize>,
    /// Per domain, (right, total) over its held-out decisions.
    pub scores: Vec<(usize, usize)>,
    pub first_loss: f32,
    pub last_loss: f32,
    pub trained_on: usize,
    /// Earlier-field decisions replayed into this stage, 0 without rehearsal.
    ///
    /// Reported rather than derived from the stride, because what actually
    /// went into the fit is the number worth reading: a stride of four over
    /// eleven earlier decisions is three, not "a quarter", and a stage that
    /// rehearsed nothing because nothing came before it should say 0 rather
    /// than leave a reader to work it out.
    pub rehearsed: usize,
}

/// Study each field in turn, carrying one adapter through all of them.
///
/// This is the experiment the probe curriculum could not run. There the fit is
/// closed-form and refit per stage, so nothing is ever overwritten and a
/// falling column means interference. Here a single adapter is carried from
/// stage to stage and each field's gradients land on the previous field's
/// weights, so a falling column means the machine has actually forgotten
/// something it could previously do.
///
/// The row before any studying is the frozen base, and it has to be there.
/// Without it a drop from stage one to stage five is unreadable: it could be
/// forgetting, or it could be an adapter that never helped that field to begin
/// with.
/// Which decisions a stage trains on, and how many of them are replays.
///
/// Pure in its inputs, for the reason `update::decide` and
/// `initiative::sized_budget` are: this is the part of rehearsal that can be
/// got wrong quietly, and asserting it must not cost a prepared `Trial` and
/// four minutes of forward passes.
///
/// `stride` of 0 is rehearsal off, and must select exactly the stage's own
/// field -- that is what makes the two runs comparable, so it is a claim
/// rather than an assumption.
///
/// **Held-out decisions are excluded here as well as in `train_selected`.**
/// Twice on purpose. Training on the validation slice does not announce
/// itself; it just makes every later figure optimistic, and a check that
/// exists in one place is a check somebody removes when refactoring the other.
///
/// The counter advances only over *eligible* earlier decisions, so the sample
/// is an even fraction of the earlier material rather than of the positions --
/// striding over positions would take almost nothing from a field whose
/// decisions happen to sit close together in the corpus, and would take it
/// silently.
pub fn rehearsal_keep(
    meta: &[(usize, bool)],
    domain_of: &[Option<usize>],
    stage: usize,
    stride: usize,
) -> (Vec<bool>, usize) {
    let mut keep = alloc::vec![false; meta.len()];
    let mut seen_earlier = 0usize;
    let mut rehearsed = 0usize;
    for (j, (applet, held)) in meta.iter().enumerate() {
        if *held {
            continue;
        }
        match domain_of.get(*applet).copied().flatten() {
            Some(d) if d == stage => keep[j] = true,
            Some(d) if d < stage && stride > 0 => {
                if seen_earlier % stride == 0 {
                    keep[j] = true;
                    rehearsed += 1;
                }
                seen_earlier += 1;
            }
            _ => {}
        }
    }
    (keep, rehearsed)
}

/// Replay one earlier decision in every `stride`.
///
/// A stride and not a coin, for the reason `godel::frontier` walks a declared
/// grid: a run somebody cannot re-derive is a run nobody can check. Four is a
/// quarter of the earlier material, which is large enough to matter and small
/// enough that the point of studying one field at a time survives -- rehearsing
/// everything is not rehearsal, it is cumulative training, and `curriculum`
/// already measures that and unsurprisingly finds no forgetting in it.
pub const REHEARSAL_STRIDE: usize = 4;

/// Learn each field in turn, carrying one adapter through, optionally
/// replaying a sample of what came before.
///
/// `rehearse` is the stride: 0 is off and reproduces exactly what this
/// function did before rehearsal existed, which is what makes the two
/// comparable. The comparison is the whole point -- rehearsal is a claim about
/// forgetting, and a claim about forgetting that is not measured against the
/// same run without it is an assertion.
pub fn sequential(b: &super::train::Budget, rehearse: usize) -> Option<Vec<SeqRow>> {
    use super::train::Slice;

    // Masks are indexed by position in APPLETS, which is what
    // `Decision::applet` carries. Scoring still works this way; only the
    // *training* selection had to become per-decision.
    let masks: Vec<Vec<bool>> = DOMAINS
        .iter()
        .map(|d| {
            crate::sysbox::APPLETS
                .iter()
                .map(|a| d.applets.contains(&a.name))
                .collect()
        })
        .collect();

    let trial = super::with_engine(|e| super::train::prepare(e, b))?.ok()?;
    let meta = trial.decisions_meta();

    // Which domain each applet belongs to, once, so the inner loop is a lookup
    // rather than a search through every domain's name list per decision.
    let domain_of: Vec<Option<usize>> = crate::sysbox::APPLETS
        .iter()
        .map(|a| DOMAINS.iter().position(|d| d.applets.contains(&a.name)))
        .collect();

    let mut rows = Vec::new();
    let score_all = |dora: Option<&super::adapter::Dora>| -> Vec<(usize, usize)> {
        masks.iter().map(|m| trial.score_masked(dora, Slice::Held, m)).collect()
    };

    rows.push(SeqRow {
        studied: None,
        scores: score_all(None),
        first_loss: 0.0,
        last_loss: 0.0,
        trained_on: 0,
        rehearsed: 0,
    });

    let mut carried: Option<super::adapter::Dora> = None;
    for i in 0..masks.len() {
        let (keep, rehearsed) = rehearsal_keep(&meta, &domain_of, i, rehearse);
        let fit = trial.train_selected(b, carried.as_ref(), &keep);
        rows.push(SeqRow {
            studied: Some(i),
            scores: score_all(Some(&fit.dora)),
            first_loss: fit.first_loss,
            last_loss: fit.last_loss,
            trained_on: fit.epochs,
            rehearsed,
        });
        carried = Some(fit.dora);
    }
    Some(rows)
}

pub fn control(b: &super::train::Budget) -> Option<(f32, f32, f32, f32, usize)> {
    use super::train::Slice;

    let all: Vec<bool> = crate::sysbox::APPLETS.iter().map(|_| true).collect();
    let trial = super::with_engine(|e| super::train::prepare(e, b))?.ok()?;

    let before = trial.score(None, Slice::Held);
    let fit = trial.train_masked(b, None, &all);
    let after = trial.score(Some(&fit.dora), Slice::Held);
    let (right, total) = trial.score_masked(Some(&fit.dora), Slice::Held, &all);
    let _ = right;
    Some((before, after, fit.first_loss, fit.last_loss, total))
}
