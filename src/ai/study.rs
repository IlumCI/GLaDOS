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

    ok
}
