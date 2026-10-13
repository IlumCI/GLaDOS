//! The research loop: hypotheses with predictions written down before the
//! experiment runs, a closed table of experiments the machine can run on
//! itself, and a ledger that refuses a result nobody predicted.
//!
//! The method is the one `design/attractor.md` was done by, by hand:
//! freeze a prediction, run the experiment, read the verdict, and let a
//! refutation narrow the next guess. Every piece of that is here as
//! something a program can check rather than a habit somebody has to keep:
//!
//! - **A hypothesis is content-addressed.** Its rendering -- claim,
//!   experiment, arguments, seed, predicates -- is hashed, stored under
//!   `/ai/research/h/<hash>`, and its hash written to the ledger *before*
//!   the experiment runs. A result line for a hash with no earlier
//!   preregistration is refused (`record`). That is the preregistration rule
//!   `design/indicators.md` keeps in prose, made a property of the file.
//! - **Predictions are predicates over named metrics**, `metric op value`,
//!   which each experiment row's parser produces from its own output. A
//!   metric the run did not produce makes the verdict *not comparable*,
//!   never supported: absence of evidence is not a pass.
//! - **Five outcomes, kept apart.** Supported and refuted are verdicts on
//!   the prediction. Not comparable is a measurement that could not answer.
//!   *Gave up* is the loop stopping with untried routes left, and it counts
//!   against the loop. *Shown impossible* carries its evidence and is a
//!   positive result: showing that something cannot be done is science,
//!   and a loop punished for saying so learns to fabricate instead.
//! - **One writable area.** The research job writes under `/ai/research/`
//!   and nowhere else (`put`). Running unattended needs a grant the
//!   operator gives once (`research trust`), recorded *outside* that area,
//!   for `work.rs`'s reason: a job that could write its own grant has none.

use alloc::string::String;
use alloc::vec::Vec;

pub const ROOT: &str = "/ai/research/";
const LEDGER: &str = "/ai/research/ledger.txt";
/// Outside `ROOT` on purpose.
const GRANT: &str = "/etc/research-grant";

// --- predicates ---------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

impl Op {
    fn text(self) -> &'static str {
        match self {
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::Eq => "==",
        }
    }
    fn parse(s: &str) -> Option<Op> {
        Some(match s {
            "<" => Op::Lt,
            "<=" => Op::Le,
            ">" => Op::Gt,
            ">=" => Op::Ge,
            "==" => Op::Eq,
            _ => return None,
        })
    }
}

/// `metric op value`. Values are thousandths, so a rendering is exact and
/// two hypotheses that predict the same thing hash the same.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pred {
    pub metric: String,
    pub op: Op,
    pub milli: i64,
}

impl Pred {
    pub fn render(&self) -> String {
        alloc::format!("{} {} {}", self.metric, self.op.text(), fmt_milli(self.milli))
    }

    /// `name op number`, with the number in ordinary decimal.
    pub fn parse(s: &str) -> Option<Pred> {
        let mut it = s.split_whitespace();
        let (m, o, v) = (it.next()?, it.next()?, it.next()?);
        if it.next().is_some() || !m.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) {
            return None;
        }
        Some(Pred { metric: String::from(m), op: Op::parse(o)?, milli: parse_milli(v)? })
    }

    /// `None` when the metric was not measured.
    pub fn holds(&self, metrics: &[(String, i64)]) -> Option<bool> {
        let v = metrics.iter().find(|(k, _)| *k == self.metric)?.1;
        Some(match self.op {
            Op::Lt => v < self.milli,
            Op::Le => v <= self.milli,
            Op::Gt => v > self.milli,
            Op::Ge => v >= self.milli,
            Op::Eq => v == self.milli,
        })
    }
}

pub fn parse_milli(s: &str) -> Option<i64> {
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    if int.is_empty() && frac.is_empty() || frac.len() > 3 {
        return None;
    }
    let i: i64 = if int.is_empty() { 0 } else { int.parse().ok()? };
    let mut f: i64 = if frac.is_empty() { 0 } else { frac.parse().ok()? };
    for _ in frac.len()..3 {
        f *= 10;
    }
    let v = i.checked_mul(1000)?.checked_add(f)?;
    Some(if neg { -v } else { v })
}

/// For reading an experiment's output, which is not ours to keep to three
/// decimals: extra digits are truncated, not refused. Predicates use the
/// strict `parse_milli`, because a prediction must say exactly what it means.
pub fn read_milli(s: &str) -> Option<i64> {
    match s.split_once('.') {
        Some((i, f)) if f.len() > 3 => parse_milli(&alloc::format!("{}.{}", i, &f[..3])),
        _ => parse_milli(s),
    }
}

pub fn fmt_milli(v: i64) -> String {
    let (s, a) = if v < 0 { ("-", -v) } else { ("", v) };
    if a % 1000 == 0 {
        alloc::format!("{}{}", s, a / 1000)
    } else {
        let f = alloc::format!("{:03}", a % 1000);
        alloc::format!("{}{}.{}", s, a / 1000, f.trim_end_matches('0'))
    }
}

// --- hypotheses ---------------------------------------------------------------

/// A claim and what would count against it, fixed before the run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Hypothesis {
    /// A family groups hypotheses that spend one evidence budget.
    pub family: String,
    pub claim: String,
    /// A row of `EXPERIMENTS`.
    pub experiment: String,
    pub args: Vec<String>,
    pub seed: u64,
    pub predicates: Vec<Pred>,
    /// The hypothesis this one narrows, if it came from a refutation.
    pub parent: Option<String>,
}

impl Hypothesis {
    /// The canonical text, and the thing that is hashed. Field order is
    /// fixed and every list is rendered in the order given, because the
    /// order an author wrote predicates in is part of what was promised.
    pub fn render(&self) -> String {
        let mut s = alloc::format!(
            "hypothesis v1\nfamily {}\nclaim {}\nexperiment {}\nargs {}\nseed {}\n",
            self.family,
            self.claim.replace('\n', " "),
            self.experiment,
            self.args.join(" "),
            self.seed
        );
        for p in &self.predicates {
            s.push_str("predict ");
            s.push_str(&p.render());
            s.push('\n');
        }
        if let Some(p) = &self.parent {
            s.push_str("parent ");
            s.push_str(p);
            s.push('\n');
        }
        s
    }

    pub fn parse(text: &str) -> Option<Hypothesis> {
        let mut lines = text.lines();
        if lines.next()? != "hypothesis v1" {
            return None;
        }
        let mut h = Hypothesis {
            family: String::new(),
            claim: String::new(),
            experiment: String::new(),
            args: Vec::new(),
            seed: 0,
            predicates: Vec::new(),
            parent: None,
        };
        for l in lines {
            let (k, v) = l.split_once(' ').unwrap_or((l, ""));
            match k {
                "family" => h.family = String::from(v),
                "claim" => h.claim = String::from(v),
                "experiment" => h.experiment = String::from(v),
                "args" => h.args = v.split_whitespace().map(String::from).collect(),
                "seed" => h.seed = v.parse().ok()?,
                "predict" => h.predicates.push(Pred::parse(v)?),
                "parent" => h.parent = Some(String::from(v)),
                _ => return None,
            }
        }
        Some(h)
    }

    /// Sixteen hex digits of the SHA-256 of the rendering.
    pub fn id(&self) -> String {
        let d = crate::store::sha256::hash(self.render().as_bytes());
        let mut s = String::new();
        for b in &d[..8] {
            s.push_str(&alloc::format!("{:02x}", b));
        }
        s
    }
}

// --- outcomes and the ledger ----------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Verdict {
    Supported,
    Refuted,
    /// The run did not produce a metric a predicate names.
    NotComparable,
    /// Stopped with routes left untried. Counts against the loop.
    GaveUp(String),
    /// Shown not to be possible, with the evidence. A positive result.
    ShownImpossible(String),
}

impl Verdict {
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Supported => "supported",
            Verdict::Refuted => "refuted",
            Verdict::NotComparable => "not-comparable",
            Verdict::GaveUp(_) => "gave-up",
            Verdict::ShownImpossible(_) => "shown-impossible",
        }
    }
}

/// Every predicate held: supported. Any measured one failed: refuted. Any
/// left unmeasured and none failed: not comparable. A refutation wins over a
/// missing metric, since one failed prediction is already an answer.
pub fn judge(preds: &[Pred], metrics: &[(String, i64)]) -> Verdict {
    let mut missing = false;
    for p in preds {
        match p.holds(metrics) {
            Some(false) => return Verdict::Refuted,
            None => missing = true,
            Some(true) => {}
        }
    }
    if missing || preds.is_empty() {
        Verdict::NotComparable
    } else {
        Verdict::Supported
    }
}

/// Whether a path is inside the research job's one writable area.
/// `..` is refused outright rather than normalised, as the guest jail does.
pub fn writable(path: &str) -> bool {
    path.starts_with(ROOT) && !path.split('/').any(|c| c == "..")
}

/// Write under `ROOT`, and nowhere else.
pub fn put(path: &str, data: Vec<u8>) -> bool {
    writable(path) && crate::sysbox::write_blob(path, data)
}

pub fn granted() -> bool {
    crate::sysbox::read_blob(GRANT).is_some_and(|b| b.starts_with(b"granted"))
}

/// The operator's half. Reached from the shell and from nothing else.
pub fn set_grant(on: bool) -> bool {
    let text: &[u8] = if on { b"granted by the operator\n" } else { b"revoked by the operator\n" };
    crate::sysbox::write_blob(GRANT, text.to_vec())
}

pub fn ledger() -> String {
    crate::sysbox::read_blob(LEDGER).and_then(|b| String::from_utf8(b).ok()).unwrap_or_default()
}

fn append(line: &str) -> bool {
    let mut l = ledger();
    l.push_str(line);
    l.push('\n');
    put(LEDGER, l.into_bytes())
}

/// Store a hypothesis and preregister it. Answers its id.
pub fn preregister(h: &Hypothesis) -> Result<String, &'static str> {
    if find(&h.experiment).is_none() {
        return Err("no such experiment in the table");
    }
    if h.predicates.is_empty() {
        return Err("a hypothesis with no prediction cannot be refuted");
    }
    let id = h.id();
    if state(&ledger(), &id) != State::Unknown {
        return Err("already preregistered");
    }
    if !put(&alloc::format!("{}h/{}", ROOT, id), h.render().into_bytes()) {
        return Err("could not store it");
    }
    if !append(&alloc::format!("prereg {} family={} experiment={}", id, h.family, h.experiment)) {
        return Err("could not write the ledger");
    }
    Ok(id)
}

pub fn load(id: &str) -> Option<Hypothesis> {
    let b = crate::sysbox::read_blob(&alloc::format!("{}h/{}", ROOT, id))?;
    Hypothesis::parse(core::str::from_utf8(&b).ok()?)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Unknown,
    Open,
    Closed,
}

/// Where a hypothesis stands in a ledger's text.
pub fn state(ledger: &str, id: &str) -> State {
    let mut st = State::Unknown;
    for l in ledger.lines() {
        let mut it = l.split_whitespace();
        let (kind, who) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        if who != id {
            continue;
        }
        st = match (kind, st) {
            ("prereg", State::Unknown) => State::Open,
            ("prereg", s) => s,
            (_, State::Open) => State::Closed,
            (_, s) => s,
        };
    }
    st
}

/// Close a hypothesis with its verdict. Refused for a hypothesis that was
/// never preregistered, and for one already closed: a second result for one
/// prediction is a second look at the data.
pub fn record(id: &str, v: &Verdict, metrics: &[(String, i64)]) -> Result<(), &'static str> {
    let l = ledger();
    match state(&l, id) {
        State::Unknown => return Err("no preregistration for this hypothesis -- a result needs a prediction before it"),
        State::Closed => return Err("this hypothesis already has its result"),
        State::Open => {}
    }
    let mut line = alloc::format!("result {} verdict={}", id, v.word());
    match v {
        Verdict::GaveUp(why) => line.push_str(&alloc::format!(" why=\"{}\"", why.replace('"', "'"))),
        Verdict::ShownImpossible(ev) => line.push_str(&alloc::format!(" evidence=\"{}\"", ev.replace('"', "'"))),
        _ => {}
    }
    for (k, m) in metrics {
        line.push_str(&alloc::format!(" {}={}", k, fmt_milli(*m)));
    }
    if append(&line) {
        Ok(())
    } else {
        Err("could not write the ledger")
    }
}

// --- the experiment table -------------------------------------------------------

/// One thing the machine can run on itself, with a fixed argument shape, the
/// shell line it is, and a parser from that line's output to metrics.
pub struct Experiment {
    pub name: &'static str,
    pub about: &'static str,
    /// Argument shape, for the operator and for a grammar later.
    pub shape: &'static str,
    /// The shell lines the experiment is, run in order under one capture.
    pub line: fn(&[String]) -> Option<Vec<String>>,
    pub parse: fn(&str) -> Vec<(String, i64)>,
}

/// A closed table, `linux::program`'s idea: what the loop can run is a set
/// somebody declared, not any line it can spell.
pub static EXPERIMENTS: &[Experiment] = &[
    Experiment {
        name: "introspect",
        about: "the preregistered introspection test (design/indicators.md section 1)",
        shape: "[trials] [ms] [seed]",
        line: |a| {
            if a.len() > 3 || !a.iter().all(|x| x.parse::<u64>().is_ok()) {
                return None;
            }
            Some(alloc::vec![alloc::format!("mind introspect {}", a.join(" "))])
        },
        parse: parse_introspect,
    },
    Experiment {
        name: "rails",
        about: "every rail, one block (bench report)",
        shape: "",
        line: |a| if a.is_empty() { Some(alloc::vec![String::from("bench report")]) } else { None },
        parse: parse_rails,
    },
    Experiment {
        name: "brain-poke",
        about: "drive one sense, then read the brain's state at once -- onset only, before the affect has folded",
        shape: "<sense> <ms>",
        line: |a| {
            if a.len() != 2 || crate::brain::Sense::parse(&a[0]).is_none() || a[1].parse::<u32>().is_err() {
                return None;
            }
            Some(alloc::vec![alloc::format!("brain poke {} {}", a[0], a[1]), String::from("brain")])
        },
        parse: parse_brain,
    },
];

pub fn find(name: &str) -> Option<&'static Experiment> {
    EXPERIMENTS.iter().find(|e| e.name == name)
}

/// `[rail] v1` lines: `name value unit want`.
pub fn parse_rails(out: &str) -> Vec<(String, i64)> {
    let mut m = Vec::new();
    let mut inside = false;
    for l in out.lines() {
        let t = l.trim();
        if t.starts_with("[rail]") {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        let f: Vec<&str> = t.split_whitespace().collect();
        if f.len() == 4 && f[0].contains('.') {
            if let Some(v) = parse_milli(f[1]) {
                m.push((String::from(f[0]), v));
            }
        }
    }
    m
}

/// `mind introspect`'s summary rows:
/// `<cond> detect H/P vs sham F/S (p X)   name N/P = Y% vs 25% (p Z)`.
pub fn parse_introspect(out: &str) -> Vec<(String, i64)> {
    let mut m = Vec::new();
    for l in out.lines() {
        let t = l.trim();
        let Some(at) = t.find(" detect ") else { continue };
        let cond: String = t[..at].trim().replace(' ', "_").replace("a=", "a");
        let nums: Vec<i64> = t[at..]
            .split(|c: char| !(c.is_ascii_digit() || c == '.'))
            .filter(|s| !s.is_empty() && *s != ".")
            .filter_map(read_milli)
            .collect();
        // hits pokes fa shams p_detect named pokes pct 25 p_name
        if nums.len() >= 10 && nums[1] > 0 {
            let rate = |a: i64, b: i64| if b > 0 { a * 1000 / b } else { 0 };
            m.push((alloc::format!("{}.detect", cond), rate(nums[0], nums[1])));
            m.push((alloc::format!("{}.false_alarm", cond), rate(nums[2], nums[3])));
            m.push((alloc::format!("{}.p_detect", cond), nums[4]));
            m.push((alloc::format!("{}.name", cond), rate(nums[5], nums[6])));
            m.push((alloc::format!("{}.p_name", cond), nums[9]));
        }
    }
    m
}

/// `brain`'s affect line: `affect: valence +V, arousal A -- words`.
pub fn parse_brain(out: &str) -> Vec<(String, i64)> {
    let mut m = Vec::new();
    for l in out.lines() {
        if let Some(r) = l.trim().strip_prefix("affect: valence ") {
            let mut it = r.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty());
            if let Some(v) = it.next().and_then(|v| v.trim_start_matches('+').parse::<i64>().ok()) {
                m.push((String::from("valence"), v * 1000));
            }
            if let (Some("arousal"), Some(a)) = (it.next(), it.next().and_then(|a| a.parse::<i64>().ok())) {
                m.push((String::from("arousal"), a * 1000));
            }
        }
    }
    m
}

// --- the generator, stage 1 -----------------------------------------------------

/// A refuted sweep narrows its own range. A hypothesis whose arguments carry
/// `key=lo..hi` and whose prediction was refuted is followed by one over the
/// lower half, the `failure-points-to-next-try` rule as arithmetic. Anything
/// else has no stage-1 successor; stages 2 and 3 (templates, model-proposed)
/// are later work.
pub fn narrow(h: &Hypothesis, v: &Verdict) -> Option<Hypothesis> {
    if *v != Verdict::Refuted {
        return None;
    }
    let (i, key, lo, hi) = h.args.iter().enumerate().find_map(|(i, a)| {
        let (k, r) = a.split_once('=')?;
        let (lo, hi) = r.split_once("..")?;
        Some((i, k, parse_milli(lo)?, parse_milli(hi)?))
    })?;
    if hi - lo < 2 {
        return None;
    }
    let mid = lo + (hi - lo) / 2;
    let mut next = h.clone();
    next.args[i] = alloc::format!("{}={}..{}", key, fmt_milli(lo), fmt_milli(mid));
    next.parent = Some(h.id());
    Some(next)
}

// --- diag research ----------------------------------------------------------------

pub fn checks() -> Vec<(&'static str, bool)> {
    let h = Hypothesis {
        family: String::from("demo"),
        claim: String::from("the hidden channel names the sense above chance"),
        experiment: String::from("introspect"),
        args: alloc::vec![String::from("125")],
        seed: 20261012,
        predicates: alloc::vec![Pred::parse("hidden_a0.125.name > 0.25").unwrap_or(Pred { metric: String::new(), op: Op::Eq, milli: 0 })],
        parent: None,
    };
    let back = Hypothesis::parse(&h.render());
    let mut h2 = h.clone();
    h2.seed += 1;
    let ok_m = alloc::vec![(String::from("hidden_a0.125.name"), 400)];
    let bad_m = alloc::vec![(String::from("hidden_a0.125.name"), 200)];
    // A ledger with a result before any preregistration, and one in order.
    let id = h.id();
    let out_of_order = alloc::format!("result {} verdict=supported\nprereg {} family=demo experiment=introspect\n", id, id);
    let in_order = alloc::format!("prereg {} family=demo experiment=introspect\nresult {} verdict=supported\n", id, id);
    let rails = "[rail] v1\nai.bpb 1.234 bits/byte lower\nvideo.rect 158 us lower\n";
    let intro = "  hidden a=0.125   detect 40/100 vs sham 5/25 (p 0.0100)   name 33/100 = 33% vs 25% (p 0.0400)   answers [1, 2, 3, 4, 5]";
    let im = parse_introspect(intro);
    let sweep = Hypothesis { args: alloc::vec![String::from("u=0..0.4")], ..h.clone() };
    let narrowed = narrow(&sweep, &Verdict::Refuted);
    alloc::vec![
        ("a hypothesis renders and parses back to itself", back.as_ref() == Some(&h)),
        ("its id is a function of everything it promises", h.id() != h2.id() && h.id() == h.clone().id()),
        ("a held prediction is supported, a failed one refuted", judge(&h.predicates, &ok_m) == Verdict::Supported && judge(&h.predicates, &bad_m) == Verdict::Refuted),
        ("an unmeasured metric is not comparable, never a pass", judge(&h.predicates, &[]) == Verdict::NotComparable),
        ("a result before its preregistration does not open anything", state(&out_of_order, &id) == State::Open && state("", &id) == State::Unknown),
        ("preregistration then result closes it", state(&in_order, &id) == State::Closed),
        ("giving up and showing impossibility are different outcomes", Verdict::GaveUp(String::new()).word() != Verdict::ShownImpossible(String::new()).word()),
        ("the research area is writable and nothing else is", writable("/ai/research/h/x") && !writable("/ai/godel/ledger.txt") && !writable("/ai/research/../godel/x") && !writable("/etc/research-grant")),
        ("the grant lives outside the area the job can write", !writable(GRANT)),
        ("decimals render and parse exactly", parse_milli("0.125") == Some(125) && fmt_milli(125) == "0.125" && parse_milli("-2") == Some(-2000) && parse_milli("1.2345").is_none()),
        ("rail lines become metrics", parse_rails(rails) == alloc::vec![(String::from("ai.bpb"), 1234), (String::from("video.rect"), 158000)]),
        ("introspection rows become rates and p values", im.contains(&(String::from("hidden_a0.125.name"), 330)) && im.contains(&(String::from("hidden_a0.125.p_name"), 40))),
        ("a refuted sweep narrows to its lower half and names its parent", narrowed.as_ref().map(|n| (n.args[0].as_str(), n.parent.as_deref())) == Some(("u=0..0.2", Some(sweep.id().as_str())))),
        ("a supported sweep is not narrowed", narrow(&sweep, &Verdict::Supported).is_none()),
        ("the table refuses an argument shape it does not declare", find("brain-poke").map(|e| (e.line)(&[String::from("heat")]).is_none()) == Some(true)),
    ]
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let rows = checks();
    let mut ok = true;
    for (what, good) in &rows {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
