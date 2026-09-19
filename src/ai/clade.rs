//! Which node to grow from, chosen by what grew from it before.
//!
//! **The loop has never made this choice.** Every trial extends `head`, which
//! is the most recently adopted variant, so the machine is a hill climber that
//! cannot go back: a lineage that walked into a dead end spends every
//! subsequent night proposing children of the dead end, and the only way out
//! is an operator typing `godel rollback`.
//!
//! The Huxley-Gödel Machine (arXiv 2510.21614) is about exactly that choice,
//! and its finding is the useful part rather than its mechanism. Asked which
//! self-modifying agent to expand, the obvious answer is the one with the best
//! score -- and measured, an agent's own score **badly predicts its
//! descendants'**. What predicts them is what its descendants already did. So
//! the quantity to select on is the *clade*: every trial at or below a node,
//! and how many of them were adopted.
//!
//! `godel::axis_uncertainty` is already a Laplace-smoothed Beta posterior and
//! it is pointed at axes. This points one at nodes, which is where the paper
//! puts it.
//!
//! ### An ancestor's clade contains its child's, and that is not a bug
//!
//! The spine is adopted nodes; a refused trial is a leaf hanging off whichever
//! node it was proposed from. So a node's clade counts its own trial plus
//! every trial made anywhere below it, and the root's clade is the whole
//! ledger. The head starts cheap and confident -- one trial, one adoption --
//! and gets worse as refusals accumulate under it, while an ancestor carries
//! those same refusals *plus* the productive stretch that came before. That
//! crossover is when backtracking happens, and it happens on evidence rather
//! than on a patience counter.
//!
//! ### Sampling, and why it is still re-derivable
//!
//! Thompson sampling draws a rate from each arm's posterior and takes the
//! best draw, which is a coin -- and this module's neighbours are built on the
//! opposite property. `frontier` walks a declared grid precisely so that "the
//! next point is a function of the markers, not a coin", and `godel.rs` opens
//! by saying a certificate has to be re-derivable bit for bit by any later
//! run.
//!
//! So the draw is seeded **from the record it is about**: the ledger's length
//! and the head's own hash. A later reader with the same ledger draws the same
//! numbers and reaches the same node, which is the whole of what
//! re-derivability asks. It is exploration that a reader can reconstruct
//! rather than exploration nobody can account for -- the same bargain
//! `Dora::new` starting at zeros makes, arrived at from the other side.
//!
//! ### What this does not do
//!
//! It chooses among the head's **ancestors** and not among every node in the
//! DAG, because the only mechanism for moving the machine is `godel::rollback`
//! and that walks one step to a parent. Reaching a sibling branch means
//! restoring an arbitrary node -- its adapter, its core, its rule and its bar,
//! each validated before anything moves -- which is `rollback` generalised
//! rather than `rollback` repeated, and is a separate change with its own
//! risks. Said here rather than left to be discovered from a report that
//! never offers a sibling.

use alloc::string::String;
use alloc::vec::Vec;

/// A node's short name: the first four bytes of its hash, as the ledger
/// renders them.
///
/// Four bytes, because that is what `godel::short` writes and the ledger is
/// the only record there is. Two nodes colliding would merge two clades, and
/// at the few dozen nodes a lineage holds that is not a risk worth a wider
/// field -- but it is a property of the *rendering* rather than of the DAG,
/// so it is written down here where somebody widening the ledger would look.
pub type Name = u32;

/// The root, which no line names as a variant.
pub const ROOT: Name = 0;

/// One ledger line, reduced to what a clade needs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Step {
    /// The node this trial was proposed from. `ROOT` for `parent=root....`.
    pub parent: Name,
    /// The node it produced, adopted or not.
    pub variant: Name,
    pub adopted: bool,
}

/// A node, and what happened at or below it.
#[derive(Clone, Copy)]
pub struct Arm {
    pub node: Name,
    /// Trials at or below it, its own included.
    pub trials: u32,
    /// How many of those were adopted.
    pub adoptions: u32,
    /// Steps from the head. 0 is the head itself.
    pub back: usize,
}

fn hex4(s: &str) -> Option<Name> {
    if s.len() < 8 {
        return None;
    }
    let mut v: u32 = 0;
    for c in s.bytes().take(8) {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        v = (v << 4) | d as u32;
    }
    Some(v)
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let at = line.find(key)? + key.len();
    let rest = &line[at..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Some(&rest[..end])
}

/// One ledger line, or `None` if it is not one this can read.
///
/// **A line it cannot read is skipped rather than guessed at**, which is the
/// rule `axis_counts` already follows: lines written before a field existed
/// count for nothing, and an unreadable line that was counted as a refusal
/// would be evidence the machine invented about itself.
pub fn parse(line: &str) -> Option<Step> {
    let p = field(line, " parent=")?;
    let parent = if p.starts_with("root") { ROOT } else { hex4(p)? };
    let variant = hex4(field(line, " variant=")?)?;
    // `ADOPT` is written with a trailing `test=` and `reject` stands alone, so
    // the space in front is what stops a rail or a reason containing the word
    // from being read as a verdict.
    let adopted = line.contains(" ADOPT");
    Some(Step { parent, variant, adopted })
}

/// Every readable line of a ledger, oldest first.
pub fn steps(lines: &[String]) -> Vec<Step> {
    lines.iter().filter_map(|l| parse(l)).collect()
}

/// The path from a node back to the root, nearest first.
///
/// **Bounded, and the bound is not caution.** The ledger is a text file on a
/// volume anything can write, so a hand-edited one can say a node is its own
/// ancestor. Walking that is a kernel that stops with no message, in a module
/// whose whole subject is what to do when something has gone wrong.
pub fn spine(steps: &[Step], head: Name) -> Vec<Name> {
    let mut out = Vec::new();
    let mut at = head;
    let mut seen = 0usize;
    while seen <= steps.len() {
        out.push(at);
        if at == ROOT {
            break;
        }
        // The line that *produced* this node says where it came from.
        let Some(s) = steps.iter().rev().find(|s| s.variant == at) else { break };
        if out.contains(&s.parent) {
            // A cycle, or a node naming itself. Stop where the record stops
            // making sense rather than reporting a spine that is not one.
            break;
        }
        at = s.parent;
        seen += 1;
    }
    out
}

/// Trials and adoptions at or below one node.
///
/// Breadth-first over the child edges rather than recursion, because there is
/// no guard page here and a ledger is a file: a chain a thousand long is a
/// legal thing to find on disk and a thousand frames is not a legal thing to
/// spend.
pub fn clade(steps: &[Step], node: Name) -> (u32, u32) {
    let mut frontier: Vec<Name> = Vec::new();
    let mut seen: Vec<Name> = Vec::new();
    let (mut trials, mut adoptions) = (0u32, 0u32);

    // The trial that produced this node counts as its own evidence: it is the
    // one measurement made *of* it, and a clade that excluded it would score a
    // freshly adopted node identically to one that had just been refused.
    if node != ROOT {
        if let Some(s) = steps.iter().find(|s| s.variant == node) {
            trials += 1;
            if s.adopted {
                adoptions += 1;
            }
        }
    }
    frontier.push(node);
    seen.push(node);
    while let Some(at) = frontier.pop() {
        for s in steps.iter().filter(|s| s.parent == at) {
            if seen.contains(&s.variant) {
                continue;
            }
            trials += 1;
            if s.adopted {
                adoptions += 1;
            }
            seen.push(s.variant);
            frontier.push(s.variant);
        }
    }
    (trials, adoptions)
}

/// The head and every ancestor, each with its clade.
pub fn arms(steps: &[Step], head: Name) -> Vec<Arm> {
    spine(steps, head)
        .into_iter()
        .enumerate()
        .map(|(back, node)| {
            let (trials, adoptions) = clade(steps, node);
            Arm { node, trials, adoptions, back }
        })
        .collect()
}

// --- the draw ------------------------------------------------------------

/// splitmix64, which is a hash and a generator at once.
///
/// Written out rather than reached for: `rng` here is a fast-key-erasure
/// ChaCha20 DRBG whose whole property is that its state is *gone* after a
/// draw, and what this needs is the opposite -- the same seed producing the
/// same stream, every time, forever.
fn mix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform in (0, 1), never 0, because the caller takes its logarithm.
fn unit(x: &mut u64) -> f32 {
    let bits = (mix(x) >> 40) as u32; // 24 bits
    (bits as f32 + 1.0) / 16_777_217.0
}

/// How many trials one arm's draw will look at.
///
/// A Gamma of integer shape is a sum of that many exponentials, so the cost is
/// linear in the count. A ledger of a few hundred lines is nothing; a ledger
/// somebody concatenated to itself a few times should still answer tonight.
const DRAW_CAP: u32 = 4096;

/// One draw from `Beta(a+1, b+1)`.
///
/// Gamma-of-integer-shape as a sum of exponentials, and the ratio of two of
/// them is a Beta. Exact for integer parameters rather than an approximation,
/// which matters here because the counts *are* integers and are small: an
/// approximation tuned for large shapes would be at its worst on the arm with
/// one trial, which is the arm whose uncertainty is the whole point.
pub fn beta(seed: u64, a: u32, b: u32) -> f32 {
    let mut s = seed;
    let gamma = |k: u32, s: &mut u64| -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..k.min(DRAW_CAP) {
            acc -= super::tensor::lnf(unit(s));
        }
        acc
    };
    let x = gamma(a + 1, &mut s);
    let y = gamma(b + 1, &mut s);
    let t = x + y;
    if t <= 0.0 {
        0.5
    } else {
        x / t
    }
}

/// The seed a ledger implies.
///
/// **From the record and from nothing else.** A clock or a counter would make
/// the choice unaccountable, and `godel.rs` rests on a later run being able to
/// re-derive what this one did. Two different heads, or one more line in the
/// ledger, give a different stream; the same ledger gives the same stream on
/// any machine.
pub fn seed_of(ledger_len: usize, head: Name) -> u64 {
    let mut s = (ledger_len as u64).wrapping_mul(0x517C_C1B7_2722_0A95) ^ (head as u64) << 17;
    mix(&mut s);
    mix(&mut s)
}

/// Which arm to grow from. The index into `arms`, never empty-handed.
///
/// Each arm draws from its own stream, keyed by its node rather than by its
/// position, so inserting a node between two others does not reshuffle
/// everybody else's draw. That is what makes two reports of one ledger
/// comparable line by line.
pub fn pick(seed: u64, arms: &[Arm]) -> usize {
    let mut best = 0usize;
    let mut best_draw = -1.0f32;
    for (i, a) in arms.iter().enumerate() {
        let d = draw_for(seed, a);
        // Strictly greater, so a tie leaves the earlier arm -- and the head is
        // arm zero, so a tie is a decision to stay where the machine is.
        if d > best_draw {
            best_draw = d;
            best = i;
        }
    }
    best
}

/// One arm's draw, exposed so a report can print the number the choice was
/// made on rather than a number computed a second way.
pub fn draw_for(seed: u64, a: &Arm) -> f32 {
    let s = seed ^ (a.node as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    beta(s, a.adoptions, a.trials.saturating_sub(a.adoptions))
}

/// What a night should do about where it is standing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    /// Grow from the head, which is what every night has ever done.
    Stay,
    /// Go back this many parents first. Never zero.
    Back(usize),
}

/// How much has to have happened below the head before it can be left.
///
/// A Beta with a handful of observations is dominated by its prior, so a draw
/// from it is a draw from the prior -- and backtracking on that is
/// backtracking on nothing. Six, which is the count `clean_fixes_needed`
/// arrives at from the other direction: below it, the arithmetic cannot
/// distinguish a result from a coin.
pub const MIN_EVIDENCE: u32 = 6;

/// How far one night may unwind.
///
/// Not a property of the posterior, which would happily send a lineage back
/// twenty adoptions on one draw. It is a bound on what an unattended machine
/// may do between two chances for anybody to look, and four is chosen to be
/// reviewable rather than derived: the decision is re-made every night with
/// the evidence that has accumulated since, so a genuinely bad branch is still
/// left, just not all at once.
pub const MAX_BACK: usize = 4;

/// The whole decision, from a ledger and a head.
///
/// **One arm means stay, and that is not a special case worth removing.** A
/// fresh machine has a spine of one node and no evidence at all; drawing for
/// it would spend a Beta on a choice with one option and report a rate nobody
/// can act on.
pub fn decide(lines: &[String], head: Name) -> (Move, Vec<Arm>) {
    let st = steps(lines);
    let arms = arms(&st, head);
    if arms.len() < 2 || arms[0].trials < MIN_EVIDENCE {
        return (Move::Stay, arms);
    }
    let seed = seed_of(lines.len(), head);
    let at = pick(seed, &arms);
    let mv = if at == 0 { Move::Stay } else { Move::Back(arms[at].back.min(MAX_BACK)) };
    (mv, arms)
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |good: bool, what: &str| {
        kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        ok &= good;
    };

    // A real line, in the shape `render_certificate` writes.
    let line = "7 h3 parent=1a2b3c4d variant=deadbeef axis=source rail=host.retrieval \
                moved=same corpus=f330c22c cell=0 n=96 reject";
    match parse(line) {
        Some(s) => {
            claim(s.parent == 0x1a2b_3c4d, "a ledger line gives up its parent");
            claim(s.variant == 0xdead_beef, "and the variant it produced");
            claim(!s.adopted, "and that it was refused");
        }
        None => claim(false, "a ledger line parses at all"),
    }
    let adopted = "8 h3 parent=deadbeef variant=00ff00ff axis=adapter ADOPT test=41.0%";
    claim(
        parse(adopted).map(|s| s.adopted) == Some(true),
        "and an adoption is read as one",
    );
    claim(
        parse("1 h3 parent=root.... variant=aaaabbbb axis=adapter reject")
            .map(|s| s.parent)
            == Some(ROOT),
        "a trial from the frozen model names the root",
    );
    claim(parse("this is not a ledger line").is_none(), "and anything else is skipped");

    // A lineage: root -> a (adopted) -> b (adopted), with refusals hung off
    // each. This is the shape every real ledger has, because the head only
    // moves on an adoption and a refusal is therefore always a leaf.
    let st = [
        Step { parent: ROOT, variant: 0xa, adopted: true },
        Step { parent: 0xa, variant: 0x11, adopted: false },
        Step { parent: 0xa, variant: 0xb, adopted: true },
        Step { parent: 0xb, variant: 0x21, adopted: false },
        Step { parent: 0xb, variant: 0x22, adopted: false },
    ];
    claim(clade(&st, 0x22) == (1, 0), "a leaf's clade is the one trial that made it");
    claim(clade(&st, 0xb) == (3, 1), "a node's clade counts itself and what came after");
    claim(clade(&st, 0xa) == (5, 2), "an ancestor's clade contains its child's");
    claim(clade(&st, ROOT) == (5, 2), "and the root's is the whole ledger");

    let sp = spine(&st, 0xb);
    claim(sp == [0xb, 0xa, ROOT], "the spine walks parents back to the root");
    let a = arms(&st, 0xb);
    claim(a.len() == 3 && a[0].back == 0 && a[2].back == 2, "and each arm knows how far back it is");

    // **A ledger that says a node is its own ancestor must stop.** It is a
    // text file on a volume anything can write, and this kernel has no guard
    // page: a walk that followed it would be a machine that went quiet.
    let loopy = [
        Step { parent: 0x2, variant: 0x1, adopted: true },
        Step { parent: 0x1, variant: 0x2, adopted: true },
    ];
    claim(spine(&loopy, 0x1).len() <= 3, "a cyclic lineage is walked once and stopped");
    claim(clade(&loopy, 0x1).0 <= 2, "and a cyclic clade is counted once");

    // The draw. Deterministic first, because everything else here rests on it.
    let arm_good = Arm { node: 1, trials: 20, adoptions: 18, back: 0 };
    let arm_bad = Arm { node: 2, trials: 20, adoptions: 1, back: 1 };
    claim(
        draw_for(99, &arm_good) == draw_for(99, &arm_good),
        "one seed and one arm give one draw, every time",
    );
    claim(
        seed_of(10, 0xabcd) == seed_of(10, 0xabcd) && seed_of(10, 0xabcd) != seed_of(11, 0xabcd),
        "a seed is a function of the ledger, and a longer ledger is a different one",
    );

    // It has to actually *sample*, or it is `axis_uncertainty` with more
    // arithmetic. Different seeds, different draws.
    let mut spread = false;
    for s in 0..8u64 {
        if draw_for(s, &arm_good) != draw_for(s + 1, &arm_good) {
            spread = true;
        }
    }
    claim(spread, "and different seeds give different draws, or it is not sampling");

    // And it has to be *ordered* by the evidence despite being random. 18 of
    // 20 beats 1 of 20 nearly always; "nearly" is the whole point, so this
    // counts rather than asserting one draw.
    let mut wins = 0;
    for s in 0..200u64 {
        if pick(s, &[arm_good, arm_bad]) == 0 {
            wins += 1;
        }
    }
    claim(wins > 180, "a clade that worked is picked far more often than one that did not");

    // The other direction, which is what separates this from taking the max.
    // An arm with no evidence must win sometimes, or the loop can never leave
    // the first thing that happened to work.
    let proven = Arm { node: 3, trials: 8, adoptions: 4, back: 0 };
    let unknown = Arm { node: 4, trials: 0, adoptions: 0, back: 1 };
    let mut explored = 0;
    for s in 0..200u64 {
        if pick(s, &[proven, unknown]) == 1 {
            explored += 1;
        }
    }
    claim(
        explored > 20 && explored < 180,
        "and an arm with no record is neither starved nor preferred",
    );

    // The crossover this exists for: a head that has produced nothing but
    // refusals loses to the ancestor that carries its productive stretch.
    let stuck = Arm { node: 5, trials: 12, adoptions: 1, back: 0 };
    let older = Arm { node: 6, trials: 40, adoptions: 12, back: 3 };
    let mut back = 0;
    for s in 0..200u64 {
        if pick(s, &[stuck, older]) == 1 {
            back += 1;
        }
    }
    claim(back > 120, "a head that has stopped producing loses to a better ancestor");

    // And the case that must not move: one arm is one option.
    let (mv, _) = decide(&[String::from("1 h0 parent=root.... variant=000000aa axis=adapter ADOPT test=1")], 0xaa);
    claim(mv == Move::Stay, "a lineage with nowhere to go back to stays");

    // A young head is left alone however the draw falls, because a posterior
    // with two observations is its own prior wearing a result's clothes.
    let young = [
        String::from("1 h0 parent=root.... variant=000000aa axis=adapter ADOPT test=1"),
        String::from("2 h0 parent=000000aa variant=000000bb axis=adapter ADOPT test=1"),
        String::from("3 h0 parent=000000bb variant=000000cc axis=adapter reject"),
    ];
    let (mv, a) = decide(&young, 0xbb);
    claim(
        mv == Move::Stay && a.len() == 3 && a[0].trials < MIN_EVIDENCE,
        "a head with less evidence than the prior is not left on a draw",
    );

    // And a long lineage cannot be unwound in one night, however the draw
    // falls: the decision is re-made tomorrow with what happened tonight.
    let mut deep: Vec<String> = Vec::new();
    let mut prev = String::from("root....");
    for i in 1..14u32 {
        deep.push(alloc::format!(
            "{} h0 parent={} variant={:08x} axis=adapter ADOPT test=1",
            i, prev, 0x1000 + i
        ));
        prev = alloc::format!("{:08x}", 0x1000 + i);
        deep.push(alloc::format!(
            "{} h0 parent={} variant={:08x} axis=adapter reject",
            i, prev, 0x9000 + i
        ));
    }
    let (mv, a) = decide(&deep, 0x1000 + 13);
    claim(a.len() == 14, "a long lineage gives an arm per ancestor");
    claim(
        match mv {
            Move::Stay => true,
            Move::Back(n) => n >= 1 && n <= MAX_BACK,
        },
        "and one night may unwind at most the declared number of steps",
    );

    ok
}
