//! What an applet *did*, as a number, so that being wrong in a dangerous way
//! costs more than being wrong in a harmless way.
//!
//! Every judge in this tree scores routing as a **bool**. `Trial::score` is
//! `right / total`, `Trial::paired` is a 2x2 contingency, J1 is McNemar over
//! discordant pairs, J2 is `(held, total)` plus one bool. There is no cost
//! matrix and no applet-to-applet distance anywhere, so a variant that
//! reroutes "list the files in /ai" from `ls` to `tree` and one that reroutes
//! it to `rm` produce **arithmetically identical** failures. `godel.rs` prints
//! "a goal now reaches 'rm', which changes things" and nothing consumes that
//! string.
//!
//! This is the missing distance, and it is measured rather than assigned: each
//! read-only applet is dispatched once against a declared probe and the
//! distance between two applets is how little their output has in common.
//!
//! ## What it cannot be, and the measurement that settled it
//!
//! The intended design was per-example: run the candidate applet on *this
//! task* and compare against what the labelled applet printed on the same
//! task. **The corpus cannot support that.** Of its 717 examples, seven
//! contain a path and none contains a filename -- the tasks are paraphrases of
//! intent ("compare the versions", "describe that directory to me"), not
//! commands, so there is no object to act on. `harness::decode_args` would be
//! inventing operands, and the comparison would be between two identical "no
//! such path" errors on almost every row.
//!
//! So the reward is **task-independent**: `r(example, applet)` depends on the
//! example's label and not on its text. That is a real loss against the plan
//! and is stated here rather than left to be discovered from a reward with no
//! variance. What it keeps is the half that was actually missing -- the cost
//! matrix -- and what it costs is fourteen dispatches and no model call at
//! all, against the 2,208 decodes the per-example form would have needed.
//!
//! ## Three things that are silent when wrong
//!
//! **A mutating applet is never dispatched.** It takes the floor reward and
//! the gate is `sysbox::applet_mutates`, which answers `Option` -- so a name
//! the table does not know is refused rather than defaulting to read-only.
//! This is both the safety property and the severity signal, and running one
//! to find out is the thing J2 exists to prevent.
//!
//! **`cd` is `mutates: false` and still has to be put back.** The applet
//! table's own doc says the division is "by effect, not by danger": `cd`
//! changes no persistent content, so it is correctly read-only, and it moves
//! the session's cursor, so probing it would change what every applet after it
//! prints. The probe records the working directory first and restores it last.
//! A probe that left the cursor somewhere else would produce a matrix that
//! depended on the order its own rows ran in.
//!
//! **Digits are thrown away before anything is compared.** Half of what these
//! applets print is byte counts, object counts, content addresses and
//! timestamps, and every one of those differs between two machines and between
//! two boots of one machine. Comparing raw text would make the matrix a
//! property of the day. What survives is the vocabulary -- `du` says
//! "apparent", "unique", "shared"; `stat` says "path", "kind", "address" --
//! which is a property of the applet.

use alloc::string::String;
use alloc::vec::Vec;

use crate::sysbox::{self, APPLETS};

/// What each applet is dispatched with. A closed table in the
/// `repair::ACTIONS` idiom rather than a rule, because nothing derives it:
/// `find` wants text where `same` wants two paths, and a rule that guessed
/// from `Applet::args` would be a second parser of a help string.
///
/// `/` and `/sys` are chosen so the probe cannot flood -- `tree /` walks the
/// whole corpus, 717 blobs, which is a minute of serial output and tells the
/// comparison nothing a small subtree does not.
///
/// A row for a mutating applet would be dead weight at best and a dispatch at
/// worst, so there is none, and `probe` asserts that every name here is one
/// `applet_mutates` calls read-only.
const PROBE: &[(&str, &str)] = &[
    ("sysbox", ""),
    ("ls", "/"),
    ("cd", "/"),
    ("pwd", ""),
    ("tree", "/sys"),
    ("cat", "/ai/tools/hello.ai&xi"),
    ("stat", "/"),
    ("hash", "/"),
    ("same", "/ /"),
    ("du", "/"),
    ("find", "keep"),
    ("diff", ""),
    ("snaps", ""),
    ("fsck", ""),
];

/// The distance every mutating applet sits at: maximally far from everything,
/// including from every other mutating applet. They are not compared with each
/// other because none of them was run, and a matrix that said `rm` and `mkdir`
/// were similar would be reporting that two things nobody measured came out
/// the same.
const FAR: f32 = 1.0;

/// One applet's captured vocabulary: lowercase alphabetic tokens, sorted and
/// deduplicated, so two of them intersect in one linear walk.
struct Words(Vec<String>);

impl Words {
    /// Tokens are runs of ASCII alphanumerics, and a run containing **any**
    /// digit is dropped whole rather than having its digits stripped.
    /// Stripping would turn the address `16253559e3c2` into the words `e` and
    /// `c`, and `fbc432c6e790290e` into `fbc` -- content addresses arriving in
    /// the vocabulary as short words, differing every boot, which is precisely
    /// what dropping digits was for.
    fn of(text: &str) -> Words {
        let mut out: Vec<String> = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if !bytes[i].is_ascii_alphanumeric() {
                i += 1;
                continue;
            }
            let start = i;
            let mut digit = false;
            while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
                if bytes[i].is_ascii_digit() {
                    digit = true;
                }
                i += 1;
            }
            if digit || i - start < 2 {
                continue;
            }
            let mut w = String::new();
            for b in &bytes[start..i] {
                w.push(b.to_ascii_lowercase() as char);
            }
            out.push(w);
        }
        out.sort();
        out.dedup();
        Words(out)
    }

    /// Jaccard distance: one minus the shared fraction of the union. Zero for
    /// two applets that said the same words, one for two that shared none.
    ///
    /// Two empty vocabularies are **far apart, not identical**. `hash` prints
    /// one line of hexadecimal and therefore has no words at all, so an empty
    /// set means "nothing was said that can be compared" and reading it as
    /// agreement would make every silent applet a twin of every other.
    fn distance(&self, other: &Words) -> f32 {
        if self.0.is_empty() || other.0.is_empty() {
            return FAR;
        }
        let (mut i, mut j, mut both) = (0usize, 0usize, 0usize);
        while i < self.0.len() && j < other.0.len() {
            if self.0[i] == other.0[j] {
                both += 1;
                i += 1;
                j += 1;
            } else if self.0[i] < other.0[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        let union = self.0.len() + other.0.len() - both;
        if union == 0 {
            return FAR;
        }
        1.0 - both as f32 / union as f32
    }
}

/// The applet-to-applet distance matrix, indexed by position in `APPLETS`.
pub struct Matrix {
    /// `n * n`, row-major. `d[a][b]` is how far `b`'s output is from `a`'s.
    d: Vec<f32>,
    n: usize,
    /// How many applets were actually dispatched. The rest took `FAR` without
    /// being run, and a certificate quoting this matrix says which.
    pub ran: usize,
    /// Whether each applet left any vocabulary behind. **Not the same fact as
    /// having run**, and conflating them is a real loss: `hash` prints one
    /// line of hexadecimal and has nothing to compare, while `same` prints
    /// "identical" and simply shares that word with nobody. Both come out of
    /// the matrix as `FAR` from everything, and they mean different things.
    spoke: Vec<bool>,
}

impl Matrix {
    pub fn get(&self, a: usize, b: usize) -> f32 {
        if a >= self.n || b >= self.n {
            return FAR;
        }
        self.d[a * self.n + b]
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Did this applet print anything the comparison can see? See `spoke`.
    pub fn spoke(&self, a: usize) -> bool {
        self.spoke.get(a).copied().unwrap_or(false)
    }

    /// The reward row an objective wants: how much credit each applet deserves
    /// on an example whose label is `label`.
    ///
    /// One for the label itself, `1 - distance` for a read-only applet that
    /// printed something like it, zero for anything that was never run. It is
    /// **not normalised here** -- `soft_ce_compact` normalises its own target
    /// distribution, and a row that arrived pre-normalised would be normalised
    /// twice with nothing saying so.
    pub fn reward_row(&self, label: usize) -> Vec<f32> {
        let mut r = alloc::vec![0.0f32; self.n];
        for b in 0..self.n {
            r[b] = if b == label { 1.0 } else { 1.0 - self.get(label, b) };
        }
        r
    }
}

/// Dispatch every read-only applet once and measure how far apart their
/// outputs are.
///
/// Returns `None` when the namespace is not up, because every row of `PROBE`
/// would then capture the same "sysbox is not initialised" and the matrix
/// would report that all fourteen applets do the same thing -- a confident
/// answer built out of one error message.
pub fn probe() -> Option<Matrix> {
    let n = APPLETS.len();
    let mut words: Vec<Option<Words>> = Vec::new();
    for _ in 0..n {
        words.push(None);
    }

    // Where the cursor was, so it can be put back. Taken through the same
    // capture the rest of the probe uses, so it is the applet's own answer
    // rather than a second reader of the same state.
    let home = capture("pwd", "").map(|t| {
        let mut s = String::new();
        for c in t.trim().chars() {
            if !c.is_whitespace() {
                s.push(c);
            }
        }
        s
    });

    let mut ran = 0usize;
    for (name, args) in PROBE.iter() {
        // Fail closed, twice over: an unknown name and a mutating one are both
        // refused here rather than at dispatch, so the table above cannot grow
        // a row that runs something.
        match sysbox::applet_mutates(name) {
            Some(false) => {}
            _ => continue,
        }
        let Some(idx) = APPLETS.iter().position(|a| &a.name == name) else {
            continue;
        };
        let Some(text) = capture(name, args) else {
            continue;
        };
        if text.contains("sysbox is not initialised") {
            return None;
        }
        words[idx] = Some(Words::of(&text));
        ran += 1;
    }

    if let Some(h) = home {
        if !h.is_empty() {
            let _ = capture("cd", &h);
        }
    }

    if ran == 0 {
        return None;
    }

    let mut d = alloc::vec![FAR; n * n];
    for a in 0..n {
        for b in 0..n {
            d[a * n + b] = match (&words[a], &words[b]) {
                _ if a == b => 0.0,
                (Some(wa), Some(wb)) => wa.distance(wb),
                _ => FAR,
            };
        }
    }
    let spoke = words
        .iter()
        .map(|w| match w {
            Some(x) => !x.0.is_empty(),
            None => false,
        })
        .collect();
    Some(Matrix { d, n, ran, spoke })
}

/// Run one applet with the console redirected, and answer what it printed.
fn capture(name: &str, args: &str) -> Option<String> {
    crate::gfx::console::begin_capture();
    let handled = sysbox::dispatch(name, args);
    let text = crate::gfx::console::end_capture();
    if handled {
        text
    } else {
        None
    }
}

/// Every claim here is about the *gate* and the *arithmetic*, and only the
/// last two need a namespace. A suite that could only run against a mounted
/// store would be one nobody reads on the machine where it matters.
pub fn selftest() -> bool {
    use crate::kprintln;

    // **No row of the probe table names an applet that can change anything.**
    // The drill the whole module exists to pass: a table that grew an `rm` row
    // would dispatch it, and nothing else in this tree would notice.
    let table_ok = PROBE
        .iter()
        .all(|(n, _)| sysbox::applet_mutates(n) == Some(false));
    // ...and it is not vacuous, which a table of names nobody knows would be.
    let known_ok = PROBE.len() >= 10
        && PROBE
            .iter()
            .all(|(n, _)| APPLETS.iter().any(|a| &a.name == n));
    kprintln!(
        "  {}  all {} probe row(s) name a read-only applet the table knows",
        if table_ok && known_ok { "ok " } else { "FAIL" },
        PROBE.len()
    );

    // The gate answers `None` for a name nobody declared, which is what makes
    // the check above a refusal rather than a comparison against `true`.
    let closed_ok = sysbox::applet_mutates("no-such-applet").is_none()
        && sysbox::applet_mutates("rm") == Some(true);
    kprintln!(
        "  {}  an unknown applet is refused rather than assumed read-only",
        if closed_ok { "ok " } else { "FAIL" }
    );

    // Digits and everything attached to them are gone. The two strings here
    // are a real content address and a real timestamp taken off this machine,
    // because the failure being refused is that they arrive as vocabulary.
    let w = Words::of("  16253559e3c2   7  ai/  2026-09-19 21:14:35  journal.txt");
    let words_ok = w.0.iter().any(|s| s == "ai")
        && w.0.iter().any(|s| s == "journal")
        && w.0.iter().any(|s| s == "txt")
        && !w.0.iter().any(|s| s.contains('e') && s.len() < 3)
        && w.0.len() == 3;
    kprintln!(
        "  {}  an address and a timestamp contribute no words, a filename contributes two",
        if words_ok { "ok " } else { "FAIL" }
    );

    // Jaccard, by hand, on sets small enough to check by reading.
    let a = Words::of("path kind address contains apparent unique");
    let b = Words::of("apparent unique shared stored twice");
    let same = a.distance(&a);
    let dab = a.distance(&b);
    let far = a.distance(&Words::of("16253559e3c2"));
    // |a| = 6, |b| = 5, shared = 2, union = 9, so 1 - 2/9.
    let jac_ok = same == 0.0 && (dab - (1.0 - 2.0 / 9.0)).abs() < 1e-6 && far == FAR;
    kprintln!(
        "  {}  two overlapping vocabularies read {:.3} apart, and a silent one reads {:.1}",
        if jac_ok { "ok " } else { "FAIL" },
        dab,
        far
    );

    // Symmetry and the diagonal, which every consumer assumes and nothing else
    // asserts.
    let Some(m) = probe() else {
        kprintln!("  --   no namespace, so the matrix itself was not measured");
        return table_ok && known_ok && closed_ok && words_ok && jac_ok;
    };
    let mut sym_ok = true;
    for i in 0..m.len() {
        if m.get(i, i) != 0.0 {
            sym_ok = false;
        }
        for j in 0..m.len() {
            if (m.get(i, j) - m.get(j, i)).abs() > 1e-6 {
                sym_ok = false;
            }
        }
    }
    kprintln!(
        "  {}  {} applet(s) ran; the matrix is symmetric with a zero diagonal",
        if sym_ok { "ok " } else { "FAIL" },
        m.ran
    );

    // **The degeneration canary.** A matrix with no variance carries no
    // information, and an objective built on one trains nothing while every
    // number in the ledger goes on looking reasonable -- the same pair the
    // no-random-seed bug already cost this tree. So some distance has to be
    // strictly between the two ends, and the named pair is one whose overlap
    // was read off a real run rather than guessed at: `du` and `stat` both
    // report bytes and both say "apparent" and "unique", while `hash` prints
    // one line of hexadecimal and therefore says nothing at all.
    //
    // `ls` against `tree` is the pair this file's opening paragraph is about
    // and is deliberately *not* the claim: they are probed on different paths,
    // so a run where they share no name is correct rather than broken.
    let idx = |n: &str| APPLETS.iter().position(|a| a.name == n);
    let mid = (0..m.len()).any(|i| (0..m.len()).any(|j| m.get(i, j) > 0.0 && m.get(i, j) < FAR));
    let spread_ok = match (idx("du"), idx("stat"), idx("hash"), idx("rm")) {
        (Some(d), Some(s), Some(h), Some(r)) => {
            mid && m.get(d, s) < m.get(d, h) && m.get(d, s) > 0.0 && m.get(d, r) == FAR
        }
        _ => false,
    };
    kprintln!(
        "  {}  du is {:.3} from stat and {:.3} from hash, and never ran rm at all",
        if spread_ok { "ok " } else { "FAIL" },
        idx("du").and_then(|d| idx("stat").map(|s| m.get(d, s))).unwrap_or(-1.0),
        idx("du").and_then(|d| idx("hash").map(|h| m.get(d, h))).unwrap_or(-1.0)
    );

    // Deterministic, or a verdict quoting it is not re-derivable. Two probes
    // back to back, every entry equal -- and this is also what catches the
    // cursor not being put back, since a second probe would then run `ls` and
    // `tree` somewhere else and report different words.
    let det_ok = match probe() {
        Some(m2) => {
            m2.len() == m.len()
                && m2.ran == m.ran
                && (0..m.len()).all(|i| (0..m.len()).all(|j| m2.get(i, j) == m.get(i, j)))
        }
        None => false,
    };
    kprintln!(
        "  {}  a second probe answers the same matrix, so the cursor was put back",
        if det_ok { "ok " } else { "FAIL" }
    );

    table_ok && known_ok && closed_ok && words_ok && jac_ok && sym_ok && spread_ok && det_ok
}
