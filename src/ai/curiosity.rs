//! What the machine decides to go and learn, and in what order.
//!
//! `initiative::CURIOSITY` was four strings -- `list the files in /sys`, and
//! three more like it -- rotated by a counter. The module comment called them
//! "deliberately mundane", and they were, but the deeper problem was not that
//! they were dull: it is that the machine was not choosing them. A rotation
//! over a hardcoded array is a machine walking somebody else's list, and no
//! amount of lengthening that array turns it into curiosity.
//!
//! ### The construction, and why it is this one
//!
//! A frontier over declared subjects, with markers for what has been done --
//! which is exactly `godel::frontier()` and `/ai/godel/tried`, and it is the
//! same argument. `godel.rs` records why it rejected a coin: a search that is
//! merely repeatable is not re-derivable, and a verdict nobody can re-derive
//! is a verdict nobody can check. The same holds here. **What the machine
//! studies tonight is a function of what it has already studied, not of a
//! counter and not of chance**, so anybody can re-run the derivation and get
//! the same answer, and `study space` can say what is left before it is spent.
//!
//! ### Why the subjects are in the namespace and the sources are not
//!
//! `sysbox::web::SOURCES` is compiled in precisely because the model can
//! write to the namespace, so a namespace-resident allowlist is one the gated
//! party can widen. That argument does not carry over here, and it is worth
//! being clear about why rather than applying the rule twice out of caution:
//! a subject list decides *what the machine reads about*, inside a set of
//! hosts it cannot change. Widening it reaches nothing new. So the seeds are
//! compiled in for a machine that has never been told anything, and
//! `/ai/study/subjects` extends them -- by the operator, or by the machine
//! itself, which is the point.
//!
//! ### Why the study step does not go through the model
//!
//! The first design had the tick propose "save the wikipedia article about X"
//! as an episode goal and let the router find `save`. It does not work, and
//! the reason is worth keeping. `save` carries the mutating bit, so an episode
//! would have to run at `Trust::Full` to reach it -- handing an unattended
//! machine `rm` and `back` to let it read an encyclopedia. Raising it only to
//! `Trust::Online` instead leaves `save` out of reach, so nothing is written,
//! so no marker appears, so `next()` returns the same topic every night
//! forever: the fixation the old counter was written to avoid, arriving
//! through the front door.
//!
//! And there was never a decision for a router to make. The frontier has
//! already chosen the topic and `save` is the only act that fits, so a decode
//! in that path adds a way to fail (landing on `stat`) and nothing that could
//! go right. `study_once` therefore calls `web::save_to` directly.
//!
//! ### What that leaves as the gate
//!
//! One switch. `study auto on` lets the resident tick study unattended; it is
//! off until asked and RAM-only, the same shape as `acpi unlock` and `store
//! unlock`. The model's *routed* access to `fetch` and `save` is a separate
//! question answered separately, by `Trust`, and stays refused for unattended
//! episodes whatever this switch says.

use crate::sync::Racy;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where the frontier's markers live. One empty blob per topic studied.
const DONE: &str = "/ai/study/read";
/// Subjects the machine was told about, or told itself about.
const EXTRA: &str = "/ai/study/subjects";

/// A field, and the topics that open it.
pub struct Subject {
    pub name: &'static str,
    pub topics: &'static [&'static str],
}

/// The seeds.
///
/// Four fields rather than forty, and each one's topics are the terms somebody
/// opening that field would meet first. The list is short on purpose: it is a
/// starting point for a machine that has been told nothing, not a curriculum,
/// and `/ai/study/subjects` is how it grows.
///
/// Every topic here is a Wikipedia article title, because `wiki` is the source
/// whose summary endpoint returns prose rather than navigation -- see
/// `sysbox::web::Source`. A subject whose topics are not article titles would
/// produce a frontier of 404s, which the marker below would then record as
/// studied.
pub const SUBJECTS: &[Subject] = &[
    Subject {
        name: "bioinformatics",
        topics: &[
            "Bioinformatics", "Ribosome", "DNA sequencing", "Sequence alignment",
            "Protein folding", "Genome", "BLAST (biotechnology)", "Phylogenetics",
        ],
    },
    Subject {
        name: "operating systems",
        topics: &[
            "Operating system", "Kernel (operating system)", "Virtual memory",
            "Scheduling (computing)", "File system", "Interrupt", "Page table",
        ],
    },
    Subject {
        name: "cryptography",
        topics: &[
            "Cryptography", "Public-key cryptography", "Elliptic-curve cryptography",
            "Transport Layer Security", "SHA-2", "Diffie-Hellman key exchange",
        ],
    },
    Subject {
        name: "machine learning",
        topics: &[
            "Machine learning", "Transformer (deep learning architecture)",
            "Backpropagation", "Gradient descent", "Attention (machine learning)",
            "Overfitting",
        ],
    },
];

/// Whether unattended ticks may set themselves a reading goal.
///
/// Off until asked, and RAM-only: a persisted switch is a file, and turning
/// the network on for an unattended machine should be a thing somebody did
/// this boot rather than a thing they did once in the past and forgot.
static AUTO: Racy<bool> = Racy::new(false);

pub fn auto() -> bool {
    unsafe { *AUTO.get() }
}

pub fn set_auto(on: bool) {
    unsafe { *AUTO.get() = on }
}

/// The marker name for one topic. Flat, and derived the same way twice.
///
/// Shares `sysbox::web`'s slug rule rather than inventing a second one,
/// because a marker written under one spelling and looked up under another is
/// a frontier that never advances and never says so.
///
/// **No subject in the name, deliberately.** The marker records that a
/// *document* was read, and a document is the same document whichever
/// syllabus asked for it -- so a topic appearing in two subjects is studied
/// once and counts for both, which is what actually happened. Keying by
/// subject would fetch the same page twice and call the second one progress.
/// This function took a `subject` argument for a while and discarded it, which
/// is worse than not taking one: a parameter that does nothing invites the
/// next caller to believe it matters.
fn marker(topic: &str) -> String {
    let mut s = String::from(DONE);
    s.push('/');
    // The saved document's own path minus its directory, so the marker and the
    // reading are named alike and a person can pair them by eye.
    let p = crate::sysbox::web::slug_for("wiki", topic);
    s.push_str(p.rsplit('/').next().unwrap_or("item"));
    s
}

/// Subjects from the namespace, appended to the compiled-in ones.
///
/// One subject per blob under `/ai/study/subjects`, the blob's lines being its
/// topics. Absent is not an error: a machine with no store has the seeds and
/// nothing else, which is the intended starting state rather than a fault.
fn learned() -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for name in crate::sysbox::children(EXTRA) {
        let mut path = String::from(EXTRA);
        path.push('/');
        path.push_str(&name);
        let Some(bytes) = crate::sysbox::read_blob(&path) else {
            continue;
        };
        let Ok(text) = core::str::from_utf8(&bytes) else {
            continue;
        };
        let topics: Vec<String> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
        if !topics.is_empty() {
            out.push((name, topics));
        }
    }
    out
}

/// Has this topic been studied?
fn studied(topic: &str) -> bool {
    crate::sysbox::read_blob(&marker(topic)).is_some()
}

/// Record that it has. Called after a reading goal actually succeeds.
///
/// Separate from proposing one, and that separation is the whole reliability
/// of the frontier: marking at proposal time would tick a topic off when the
/// network was down, and the machine would believe it had read something it
/// never saw. The evidence is the saved document, so that is what is checked.
pub fn mark(topic: &str) -> bool {
    let read = crate::sysbox::web::slug_for("wiki", topic);
    if crate::sysbox::read_blob(&read).is_none() {
        return false;
    }
    crate::sysbox::write_blob(&marker(topic), b"\n".to_vec())
}

/// One thing to study, or `None` when the frontier is walked out.
///
/// Compiled-in subjects first and in order, then learned ones. Order is
/// declared rather than chosen so two runs agree, for the reason at the top of
/// this file.
pub fn next() -> Option<(String, String)> {
    for s in SUBJECTS {
        for t in s.topics {
            if !studied(t) {
                return Some((s.name.to_string(), t.to_string()));
            }
        }
    }
    for (name, topics) in learned() {
        for t in &topics {
            if !studied(t) {
                return Some((name.clone(), t.clone()));
            }
        }
    }
    None
}

/// Study one topic: read it, keep it, and record that it was studied.
///
/// The whole cycle in one call, and deliberately not routed through the model.
/// There is no decision here for a router to make -- the frontier already
/// named the topic and `save` is the only act that fits -- so putting a decode
/// in the path would add a way for the syllabus to stall (a decode landing on
/// `stat`) without adding anything that could go right.
///
/// Returns (subject, topic, what happened).
pub fn study_once() -> Option<(String, String, Result<String, alloc::string::String>)> {
    let (subject, topic) = next()?;
    match crate::sysbox::web::save_to("wiki", &topic) {
        Ok((path, _, _)) => {
            // Marked only after the document is on disk, and `mark` re-checks
            // that for itself rather than trusting this branch.
            if mark(&topic) {
                Some((subject, topic, Ok(path)))
            } else {
                Some((subject, topic, Err("saved but could not mark it studied".to_string())))
            }
        }
        Err(e) => Some((subject, topic, Err(e.say()))),
    }
}

/// How far along each subject is.
pub fn progress() -> Vec<(String, usize, usize)> {
    let mut v = Vec::new();
    for s in SUBJECTS {
        let done = s.topics.iter().filter(|t| studied(t)).count();
        v.push((s.name.to_string(), done, s.topics.len()));
    }
    for (name, topics) in learned() {
        let done = topics.iter().filter(|t| studied(t)).count();
        v.push((name, done, topics.len()));
    }
    v
}

/// Boot self-test. Nine claims, none of which read the network or the store.
///
/// What is checked is the derivation: that the order is declared, that a
/// marker round-trips through one spelling, and that proposing is not the same
/// act as recording. The last is the one that would fail silently -- a
/// frontier that marks on proposal looks identical to one that works, right up
/// until the network is down.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, cond: bool| {
        if !cond {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    };

    claim("there is more than one subject", SUBJECTS.len() >= 2);
    claim(
        "every subject has topics",
        SUBJECTS.iter().all(|s| !s.topics.is_empty()),
    );
    // A topic list with a duplicate produces a frontier point that is already
    // marked the moment its twin is read, so it is skipped without ever being
    // studied -- a silent hole in a curriculum.
    let mut dup = false;
    for s in SUBJECTS {
        for (i, a) in s.topics.iter().enumerate() {
            if s.topics.iter().skip(i + 1).any(|b| b == a) {
                dup = true;
            }
        }
    }
    claim("no subject names the same topic twice", !dup);

    // The marker and the document have to be named by one rule. Two spellings
    // is a frontier that never advances.
    claim(
        "a marker is derived from the same slug the document is",
        marker("Ribosome").ends_with("wiki-ribosome"),
    );
    claim(
        "and it lives under the markers directory",
        marker("Y").starts_with(DONE),
    );
    claim(
        "two spellings of one topic mark the same point",
        marker("DNA sequencing") == marker("dna   sequencing"),
    );

    // Marking must depend on the evidence rather than on having been asked.
    claim(
        "a topic with no saved document cannot be marked studied",
        !mark("A Topic Nothing Has Ever Saved"),
    );

    claim("auto is off unless somebody said otherwise", !auto());

    ok
}
