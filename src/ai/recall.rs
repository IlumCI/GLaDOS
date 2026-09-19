//! Budgeted retrieval: what fits in the turn, and nothing more.
//!
//! Phase 4. The router says which subject a question belongs in; this picks the
//! nodes inside it and renders as many as a token budget admits.
//!
//! ### The budget is counted, never estimated
//!
//! `companion::turn` measures the system turn by encoding it the way `generate`
//! will -- same BOS, same tokenizer -- and says why: an estimate that ran short
//! would pin part of a turn and leave the rest to scroll. The same reasoning
//! applies with more force here, because this text is chosen *to* fill a
//! budget. So `fill` takes the counter as a closure and asks it after every
//! candidate, and the answer it returns is the real encoded length of the
//! string it produced rather than a sum of parts.
//!
//! Tokenisation is not additive at a boundary -- two pieces that encode to 10
//! and 12 tokens do not reliably encode to 22 when concatenated -- so summing
//! per-node counts would drift, always in the direction of admitting one node
//! too many. Re-encoding the accumulated block is exact and costs a handful of
//! encodes of a string that is by construction no longer than the budget.
//!
//! ### Per-node vectors, and what the router is actually for
//!
//! Scoring needs a vector per node, and `route`'s table holds only per-subject
//! sums. `Nodes` is that second table: one pooled head vector per node, written
//! by `forest embed` beside the first.
//!
//! **With it resident, routing buys nothing at this corpus size**, and saying so
//! is more useful than pretending otherwise: nine thousand cosines is five
//! million operations, which is nothing. The router earns its place when the
//! vectors do not fit, and until then its cost is measurable -- `recall` reports
//! how much of the unrouted answer the routed one found, which is exactly the
//! price of not looking everywhere.

use alloc::string::String;
use alloc::vec::Vec;

use crate::sysbox;

/// Where the per-node vectors live.
pub const NODES: &str = "/ai/route/nodes";

/// And the inverted index beside them.
pub const LEX: &str = "/ai/route/lex";

const MAGIC: &[u8; 8] = b"GLADOSNV";

/// One pooled vector per node, keyed by path.
pub struct Nodes {
    pub dim: usize,
    pub paths: Vec<String>,
    /// `paths.len() * dim`, row-major.
    pub vecs: Vec<f32>,
}

impl Nodes {
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.vecs[i * self.dim..(i + 1) * self.dim]
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.paths.len() * (self.dim * 4 + 32));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.paths.len() as u32).to_le_bytes());
        for i in 0..self.paths.len() {
            let p = self.paths[i].as_bytes();
            out.extend_from_slice(&(p.len() as u16).to_le_bytes());
            out.extend_from_slice(p);
            for v in self.row(i) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Walks and never seeks, and lands on the last byte -- `v4.py`'s bargain,
    /// for its reason: a body of unnamed floats leaves a reader that disagreed
    /// about one dimension holding perfectly valid garbage.
    pub fn decode(b: &[u8]) -> Option<Nodes> {
        if b.len() < 16 || &b[..8] != MAGIC {
            return None;
        }
        let dim = u32::from_le_bytes(b[8..12].try_into().ok()?) as usize;
        let rows = u32::from_le_bytes(b[12..16].try_into().ok()?) as usize;
        if dim == 0 || dim > 1 << 16 {
            return None;
        }
        let mut n = Nodes {
            dim,
            paths: Vec::with_capacity(rows),
            vecs: Vec::with_capacity(rows * dim),
        };
        let mut at = 16usize;
        for _ in 0..rows {
            if at + 2 > b.len() {
                return None;
            }
            let pl = u16::from_le_bytes(b[at..at + 2].try_into().ok()?) as usize;
            at += 2;
            if at + pl + dim * 4 > b.len() {
                return None;
            }
            n.paths.push(String::from_utf8(b[at..at + pl].to_vec()).ok()?);
            at += pl;
            for _ in 0..dim {
                n.vecs.push(f32::from_le_bytes(b[at..at + 4].try_into().ok()?));
                at += 4;
            }
        }
        if at != b.len() {
            return None;
        }
        Some(n)
    }
}

static CACHE: crate::sync::Racy<Option<Nodes>> = crate::sync::Racy::new(None);

/// The node table, loaded once and kept.
///
/// Twenty megabytes at dim 576 over nine thousand nodes -- the price of not
/// reading and re-hashing the disk once per question. `forget` drops it, and
/// `forest embed` calls that when it writes a new one, so the cache cannot
/// outlive the table it came from.
///
/// **Not centred.** Rows are IDF-weighted and unit length by construction, so a
/// cosine against another such row is already a plain dot product. The routing
/// table's centroid belongs to the *subject* comparison and was measured there;
/// applying it here would take a constant out of one side of a different
/// comparison, which is exactly the both-sides-or-neither trap `route::centre`
/// is written to warn about.
pub fn with_nodes<R>(f: impl FnOnce(&Nodes) -> R) -> Option<R> {
    unsafe {
        let slot = CACHE.get();
        if slot.is_none() {
            *slot = Some(load()?);
        }
        slot.as_ref().map(f)
    }
}

pub fn forget() {
    unsafe { *CACHE.get() = None };
}

pub fn load() -> Option<Nodes> {
    Nodes::decode(&sysbox::read_blob(NODES)?)
}

pub fn store(n: &Nodes) -> bool {
    sysbox::write_blob(NODES, n.encode())
}

pub fn load_lex() -> Option<crate::ai::lex::Lex> {
    crate::ai::lex::Lex::decode(&sysbox::read_blob(LEX)?)
}

pub fn store_lex(l: &crate::ai::lex::Lex) -> bool {
    sysbox::write_blob(LEX, l.encode())
}

/// How much of the score comes from the embedding rather than the terms.
///
/// **Zero, and that is a measurement rather than a dismissal.** `forest bench`
/// swept it over a known-item task with 8,913 candidates and no string shared
/// between query and index:
///
///                    long query      short query
///     mean pool      r@1  0.5%       r@1  8.0%
///     idf pool       r@1  5.5%       r@1 30.8%
///     terms          r@1 87.8%       r@1 47.9%
///
/// and every weight above zero came out worse -- 81.8%/47.4% at a=0.10, down to
/// 44.9%/46.9% at a=0.50. The embedding is far stronger on a short query than a
/// long one, which is the interesting half of that table: it closes most of the
/// gap and still never opens one. So the
/// embedding channel is switched off for node retrieval, and the constant is
/// here rather than inlined so the shipped behaviour and the measurement cannot
/// drift apart in silence.
///
/// What this is *not* is a claim that embeddings are useless. It is one
/// checkpoint's embedding table, mean- and IDF-pooled with no forward pass,
/// against exact term matching on a corpus of questions -- where names and
/// numbers either appear or do not, and that is most of the signal. A different
/// model, or a fusion on ranks rather than on raw scores, may well move it. The
/// sweep is one command, and it prints every rung.
pub const MIX: f32 = 0.0;

/// A candidate to render: where it came from and how good it looked.
pub struct Cand {
    pub path: String,
    pub score: f32,
    pub body: String,
}

/// What a fill produced.
pub struct Filled {
    pub text: String,
    /// Indices into the candidate list, in the order they were taken.
    pub taken: Vec<usize>,
    /// The encoded length of `text`, from the counter that was passed in.
    pub tokens: usize,
    /// Candidates that were skipped because they did not fit, but a later
    /// smaller one did.
    pub skipped: usize,
    /// Candidates refused for being the question rather than material for
    /// it. Counted separately from `skipped` because "three were the answer"
    /// and "three did not fit" are different facts about a corpus, and only
    /// one of them is a reason to distrust the block.
    pub leaked: usize,
}

// ---------------------------------------------------------------- redaction

/// Every key a node file may carry, for telling a node from plain text.
const KEYS: [&str; 8] = [
    "head", "kind", "source", "concept", "method", "check", "answer", "text",
];

/// What of a node may reach a context window: its `text` and nothing else.
///
/// **`answer` is the obvious field and it is not the dangerous one.** A
/// `gsm8k` node's `method` and `check` lines are the worked arithmetic --
/// `60+50 = 110` -- so a redaction that dropped only `answer` would hand
/// over the sum and withhold the total. `head`, `kind`, `source` and
/// `concept` go too, for a duller reason: they are index machinery and a
/// restatement of the body's own first sentence, so they spend budget on
/// nothing.
///
/// **Keep-one rather than drop-many, and that is a safety property.** The
/// first version of this walked every line, flipped a flag on each key it
/// recognised, and kept what followed `text`. Two things were wrong with
/// it, and the second is the one that matters:
///
///   - A body line whose first word happened to be `answer` flipped the
///     flag and silently truncated the node. `forest.py::parse` has the
///     same ambiguity by design, but a redaction inheriting it fails in a
///     direction a parser does not.
///   - A node with no `text` field at all fell through to `return body`,
///     which is the **unredacted** node. A safety function that fails open
///     on malformed input is one that ships an answer key exactly when
///     something else has already gone wrong.
///
/// `render` always emits `text` last, so everything after that line is the
/// body. Finding it and keeping the remainder needs no state, cannot be
/// confused by the body's own words, and has nothing to fail open into:
/// no `text`, no answer, `None`.
///
/// It lives in `render_one` rather than at the call sites, which is the
/// other half. A node is retrieved in three places today and the next
/// caller is the one who forgets; a redaction somebody has to remember is a
/// redaction that ships an answer key the first time somebody is in a
/// hurry. One door.
pub fn redact(body: &str) -> Option<String> {
    // **"Not a node" and "a node missing its text" are different facts, and
    // only one of them is dangerous.** A string with no field line anywhere
    // carries no `answer` to strip, so passing it whole leaks nothing; one
    // that carries `answer D` and no `text` is a malformed node, which is
    // exactly what a redaction must not wave through. Asking only for
    // `text` conflated the two and silently dropped every candidate not in
    // node format -- five claims in this suite, and most of what a caller
    // outside the forest can hand this.
    let structured = body
        .lines()
        .any(|l| KEYS.contains(&l.split_whitespace().next().unwrap_or("")));
    if !structured {
        let t = body.trim();
        return if t.is_empty() { None } else { Some(String::from(t)) };
    }
    let mut lines = body.lines();
    let first = lines.find(|l| {
        let mut w = l.split_whitespace();
        w.next() == Some("text")
    })?;
    let mut out = String::from(first.strip_prefix("text").unwrap_or("").trim_start());
    for l in lines {
        out.push('\n');
        out.push_str(l);
    }
    let out = String::from(out.trim());
    if out.is_empty() {
        return None;
    }
    Some(out)
}

// --------------------------------------------------------------- leak check

/// Containment above which a candidate is the question rather than material
/// for it. Mirrors `tools/forest_retrieve.py`, deliberately: a host
/// retriever exists to predict what this kernel would retrieve, and two
/// thresholds would make it predict a different machine.
pub const LEAK: f32 = 0.8;

/// Below this many terms a question cannot be judged by overlap at all.
/// `forest_retrieve.py` carries the worked case -- four-term algebra stems
/// where one differing term is 0.75 and two is 0.5, so no threshold
/// separates a duplicate from a sibling. Short questions fall back to the
/// substring test, which is exact.
pub const MIN_LEAK_TERMS: usize = 8;

/// Words too common to carry a question's identity. The host's list, to the
/// word, for the reason `LEAK` is.
const LEAK_STOP: [&str; 34] = [
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were", "for", "on", "at",
    "by", "with", "that", "this", "it", "as", "be", "from", "how", "what", "which", "if", "then",
    "each", "many", "much", "does", "do", "not",
];

/// Lowercased, with everything that is not a letter or digit removed.
fn flatten(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Content words of three letters or more, and every run of digits.
///
/// **The numbers are the load-bearing half.** MMLU is templated, so a
/// sibling question shares every content word and differs only in its
/// figures: without digits, containment reads 1.00 over a stem four words
/// long and the guard eats the single most useful thing a forest can hand a
/// model, which is a worked example of the same kind.
fn leak_terms(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut digits = false;
    let push = |cur: &mut String, digits: bool, out: &mut Vec<String>| {
        if cur.is_empty() {
            return;
        }
        let keep = if digits { true } else { cur.len() >= 3 };
        if keep && !LEAK_STOP.contains(&cur.as_str()) && !out.iter().any(|x| x == cur) {
            out.push(cur.clone());
        }
        cur.clear();
    };
    for c in s.chars() {
        let lc = c.to_ascii_lowercase();
        if lc.is_ascii_digit() {
            if !digits {
                push(&mut cur, digits, &mut out);
                digits = true;
            }
            cur.push(lc);
        } else if lc.is_ascii_lowercase() {
            if digits {
                push(&mut cur, digits, &mut out);
                digits = false;
            }
            cur.push(lc);
        } else {
            push(&mut cur, digits, &mut out);
            digits = false;
        }
    }
    push(&mut cur, digits, &mut out);
    out
}

/// Is this candidate the question being asked rather than material for it?
///
/// Two tests, and the first is exact. A retrieved node whose body contains
/// the question verbatim -- or is contained by it -- is the answer key
/// however the overlap arithmetic reads. The second catches the reworded
/// duplicate that substring matching cannot, and is skipped for questions
/// too short to be judged by overlap.
///
/// **Deliberately generous.** A node wrongly dropped costs one retrieval and
/// is counted out loud; a node wrongly kept is an answer key and costs the
/// whole answer.
pub fn leaks(query: &str, body: &str) -> bool {
    let fq = flatten(query);
    if !fq.is_empty() {
        let fb = flatten(body);
        if fb.contains(&fq) || fq.contains(&fb) {
            return true;
        }
    }
    let qt = leak_terms(query);
    if qt.len() < MIN_LEAK_TERMS {
        return false;
    }
    let bt = leak_terms(body);
    let shared = qt.iter().filter(|t| bt.iter().any(|b| b == *t)).count();
    shared as f32 / qt.len() as f32 >= LEAK
}

/// Render one node into the block.
fn render_one(n: usize, c: &Cand) -> String {
    let mut s = String::new();
    s.push_str("--- ");
    // One-based, because this is read by a model and by a person and neither
    // counts from zero.
    s.push_str(&alloc::format!("{}", n + 1));
    s.push('\n');
    s.push_str(redact(&c.body).unwrap_or_default().trim_end());
    s.push_str("\n\n");
    s
}

pub const PREAMBLE: &str = "Entries from the library that may bear on this:\n\n";

/// Greedily take candidates until the budget will not admit another.
///
/// **Best-first, and it keeps going after a miss.** A candidate that does not
/// fit is skipped rather than ending the fill, because the list is sorted by
/// score and not by size -- stopping at the first overflow would throw away
/// every smaller, slightly-worse entry behind a single large one, and the
/// budget would go unspent for no reason anybody could see. The count of those
/// is reported, since "three were skipped" and "there were only two" are
/// different facts about a corpus.
///
/// The counter is a closure so the budget arithmetic can be checked with no
/// model: a claim passes one that counts words and gets exact, predictable
/// answers out of the same code the real path runs.
pub fn fill<F: Fn(&str) -> usize>(
    query: &str,
    cands: &[Cand],
    budget: usize,
    count: F,
) -> Filled {
    let mut out = Filled {
        text: String::new(),
        taken: Vec::new(),
        tokens: 0,
        skipped: 0,
        leaked: 0,
    };
    // An empty block is not the preamble on its own: a heading promising
    // entries with nothing under it is worse than nothing at all, so the
    // preamble is only paid for once something fits beneath it.
    for (i, c) in cands.iter().enumerate() {
        // **Before the budget, not after.** A leaking candidate that did not
        // fit would otherwise be counted as skipped, and the one number that
        // says whether this block can be trusted would read clean.
        if leaks(query, &c.body) {
            out.leaked += 1;
            continue;
        }
        // Nothing a model may see. A numbered entry with an empty body is
        // the "heading promising entries with nothing under it" this
        // function already refuses to produce, arriving from redaction
        // rather than from an empty candidate list.
        if redact(&c.body).is_none() {
            out.skipped += 1;
            continue;
        }
        let mut next = if out.taken.is_empty() {
            String::from(PREAMBLE)
        } else {
            out.text.clone()
        };
        next.push_str(&render_one(out.taken.len(), c));
        let n = count(&next);
        if n > budget {
            out.skipped += 1;
            continue;
        }
        out.text = next;
        out.tokens = n;
        out.taken.push(i);
    }
    out
}

// ------------------------------------------------------- the automatic path

/// Whether `ask` consults the forest at all.
///
/// **Off by default, and the reason is a number rather than caution.**
/// `tools/retrieval.py` measures r@1 at 44.4% on the corpus this ships
/// against: the top-ranked node is the wrong one more often than it is the
/// right one. On by default would put a confidently-ranked wrong passage in
/// front of the majority of answers, and there is no rail anywhere in this
/// tree that would say so -- every judge here scores routing, and none of
/// them reads what `ask` replied.
///
/// So this is a thing an operator turns on, having read that sentence.
static ON: crate::sync::Racy<bool> = crate::sync::Racy::new(false);

pub fn enabled() -> bool {
    unsafe { *ON.get() }
}

pub fn set_enabled(v: bool) {
    unsafe { *ON.get() = v };
}

/// How much of a turn retrieval may spend.
///
/// Clamped against the trained length for `sink_count`'s reason one level
/// on: the system turn is pinned and the recent window is what is left, so
/// a block large enough to fill it would evict the conversation to make
/// room for a guess about it.
pub const ASK_BUDGET: usize = 192;

pub fn budget_for(seq_len: usize) -> usize {
    ASK_BUDGET.min(seq_len / 8)
}

/// Top nodes for a question, redacted and leak-checked, or `None`.
///
/// **One engine behind the verb and the turn.** `forest recall` and `ask`
/// must not grow two accounts of one retrieval: the verb is how an operator
/// checks what the turn will do, and a verb that answered a different
/// question would make that check worthless. The verb keeps its own
/// reporting on top -- it measures the price of routing, which a turn does
/// not care about -- but the selection, the redaction and the leak refusal
/// are these lines for both.
pub fn pick(q: &str, budget: usize, k: usize) -> Option<Filled> {
    if budget == 0 {
        return None;
    }
    let lex = load_lex()?;
    let ids = crate::ai::with_engine(|e| crate::ai::lex::tokens(&e.tok, q))?;
    if ids.is_empty() {
        return None;
    }
    let mut ranked = with_nodes(|n| {
        let mut sl = alloc::vec![0.0f32; n.len()];
        lex.score(&ids, &mut sl);
        let mut all: Vec<(usize, f32)> = (0..n.len()).map(|i| (i, sl[i])).collect();
        // Descending by score, and by index where scores tie, so two runs
        // over one corpus pick the same nodes -- the determinism every
        // re-derivable verdict in this tree rests on.
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        all.truncate(k);
        all.iter()
            .filter(|(_, s)| *s > 0.0)
            .filter_map(|(i, s)| n.paths.get(*i).map(|p| (p.clone(), *s)))
            .collect::<Vec<_>>()
    })?;
    ranked.retain(|(p, _)| !p.is_empty());
    if ranked.is_empty() {
        return None;
    }
    let cands: Vec<Cand> = ranked
        .iter()
        .filter_map(|(path, score)| {
            let b = sysbox::read_blob(path)?;
            Some(Cand {
                path: path.clone(),
                score: *score,
                body: String::from_utf8_lossy(&b).into_owned(),
            })
        })
        .collect();
    let count = |t: &str| {
        crate::ai::with_engine(|e| e.tok.encode(t, false, false).len()).unwrap_or(usize::MAX)
    };
    let f = fill(q, &cands, budget, count);
    if f.taken.is_empty() {
        return None;
    }
    Some(f)
}

/// How many candidates a turn considers before the budget decides.
pub const ASK_K: usize = 8;

/// What the reader and the budget arithmetic claim, with no model and no forest.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    // A counter that is easy to reason about: one token per whitespace-
    // separated word. The real one is the tokenizer; what is being checked
    // here is the arithmetic around it, which is the part with edges.
    let words = |s: &str| s.split_whitespace().count();

    let cand = |p: &str, score: f32, body: &str| Cand {
        path: String::from(p),
        score,
        body: String::from(body),
    };

    let one = alloc::vec![cand("/a", 1.0, "alpha beta gamma")];
    let f = fill("", &one, 1000, words);
    check(
        "a block that fits is rendered whole, and counted by the counter given",
        f.taken == alloc::vec![0] && f.tokens == words(&f.text) && f.text.contains("alpha"),
    );
    // The claim the whole phase is for. Not "about the budget" -- never over it.
    check(
        "and its own length is what the budget was compared against",
        f.tokens <= 1000 && f.tokens > 0,
    );

    let f0 = fill("", &one, 0, words);
    check(
        "a budget of zero renders nothing at all, not a bare heading",
        f0.text.is_empty() && f0.taken.is_empty() && f0.tokens == 0 && f0.skipped == 1,
    );
    // A heading with nothing under it promises entries and delivers none,
    // which is worse than silence -- so the preamble is only paid for once
    // something fits beneath it.
    let tight = fill("", &one, words(PREAMBLE) + 1, words);
    check(
        "a budget that admits only the heading still renders nothing",
        tight.text.is_empty() && tight.taken.is_empty(),
    );

    let many = alloc::vec![
        cand("/a", 0.9, "aaa aaa aaa aaa aaa aaa aaa aaa"),
        cand("/b", 0.8, "bbb"),
        cand("/c", 0.7, "ccc"),
    ];
    // Best-first, so the big one is offered first; the budget refuses it and
    // the two small ones behind it are still taken. Stopping at the first
    // miss would leave the budget unspent with nothing saying why.
    let small = fill("", &many, words(PREAMBLE) + 8, words);
    check(
        "a candidate that does not fit is skipped, not an end to the fill",
        small.taken == alloc::vec![1, 2] && small.skipped == 1,
    );
    check(
        "and the rendering is never over budget",
        small.tokens <= words(PREAMBLE) + 8 && small.tokens == words(&small.text),
    );
    // Numbering follows what was taken rather than the candidate list, or the
    // block would read "--- 1" then "--- 3" and a model would reasonably
    // conclude something had been withheld.
    check(
        "entries are numbered by what was kept, with no gaps",
        small.text.contains("--- 1") && small.text.contains("--- 2")
            && !small.text.contains("--- 3"),
    );

    let all = fill("", &many, 10_000, words);
    check(
        "a budget that admits everything takes everything, in score order",
        all.taken == alloc::vec![0, 1, 2] && all.skipped == 0,
    );
    check(
        "an empty candidate list is an empty block rather than a heading",
        fill("", &[], 10_000, words).text.is_empty(),
    );

    // --- what a model may be handed -----------------------------------
    let mmlu = "head life-sciences/anatomy | What is the embryological origin of the hyoid bone? | x\nkind mmlu\nsource mmlu/anatomy/dev\nconcept What is the embryological origin of the hyoid bone?\nanswer D\ntext What is the embryological origin of the hyoid bone?\nA. The first pharyngeal arch\nB. The second and third pharyngeal arches\n";
    let red = redact(mmlu).unwrap_or_default();
    check(
        "an mmlu node loses its answer letter",
        !red.contains("answer D") && red.contains("embryological"),
    );
    check(
        "and its index machinery, which is budget spent on nothing",
        !red.contains("head ") && !red.contains("kind mmlu") && !red.contains("source mmlu"),
    );
    let gsm = "head x | y | z\nkind gsm8k\nsource gsm8k/train\nconcept Ralph practises tennis.\nmethod 60+50 = 110\ncheck 60+50 = 110\nanswer 110\ntext Ralph hits some balls. How many did he miss?\nHe missed 60+50 = 110 of them.\n";
    let redg = redact(gsm).unwrap_or_default();
    // The field that matters is not `answer`. `method` and `check` are the
    // worked arithmetic, so dropping only `answer` hands over the sum and
    // withholds the total.
    check(
        "a gsm8k node loses the worked arithmetic, not only the total",
        !redg.contains("method") && !redg.contains("check 60") && !redg.contains("answer 110"),
    );
    check(
        "and keeps the body, which is what it was retrieved for",
        redg.contains("How many did he miss?"),
    );
    check(
        "a body line beginning with a field name survives",
        redact("text one\nanswer me this\ntwo")
            .unwrap_or_default()
            .contains("answer me this"),
    );
    // Fails closed. A node with no `text` is unrenderable, not unredacted:
    // the first version returned the whole node here, answer included.
    check(
        "a node with no text field yields nothing rather than everything",
        redact("head x\nkind mmlu\nanswer D").is_none(),
    );
    check(
        "but a plain string with no fields at all passes through whole",
        redact("just some prose").as_deref() == Some("just some prose"),
    );

    // --- the leak check -----------------------------------------------
    let q = "What is the embryological origin of the hyoid bone?";
    check(
        "the question itself is refused, however it is dressed",
        leaks(q, mmlu) && leaks(q, "TEXT:  what is the EMBRYOLOGICAL origin of the hyoid bone"),
    );
    check(
        "an unrelated node is not",
        !leaks(q, "text The Ise-class battleships were a pair of dreadnoughts."),
    );
    // The numbers are the load-bearing half: without them a templated
    // sibling shares every content word and reads 1.00, and the guard eats
    // the worked example that is the most useful thing a forest has.
    let a = "Find the order of the factor group Z_11 x Z_15 modulo 1 1 and 4 more words";
    let b = "text Find the order of the factor group Z_4 x Z_12 modulo 2 2 and 4 more words";
    check(
        "a templated sibling with different numbers is kept",
        !leaks(a, b),
    );
    check(
        "a question too short to judge by overlap falls back to substring",
        !leaks("what is two", "text something entirely other")
            && leaks("what is two", "text what is two"),
    );
    check(
        "fill counts what it refused as the question, apart from what did not fit",
        {
            let c = Cand {
                path: String::from("/p"),
                score: 1.0,
                body: String::from("text What is the embryological origin of the hyoid bone?"),
            };
            let f = fill(q, &[c], 10_000, words);
            f.taken.is_empty() && f.leaked == 1 && f.skipped == 0
        },
    );



    // --- the table -------------------------------------------------------
    let n = Nodes {
        dim: 2,
        paths: alloc::vec![String::from("/x/1"), String::from("/x/2")],
        vecs: alloc::vec![1.0, 0.0, 0.0, 1.0],
    };
    let enc = n.encode();
    check(
        "the node table round-trips through its own bytes",
        match Nodes::decode(&enc) {
            Some(d) => d.dim == 2 && d.paths == n.paths && d.vecs == n.vecs,
            None => false,
        },
    );
    let mut extra = enc.clone();
    extra.push(0);
    check(
        "and is refused when truncated, over-long, or not a table",
        Nodes::decode(&enc[..enc.len() - 1]).is_none()
            && Nodes::decode(&extra).is_none()
            && Nodes::decode(b"GLADOSXX________").is_none(),
    );

    ok
}
