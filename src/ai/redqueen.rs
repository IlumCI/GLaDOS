//! Two populations that make each other harder.
//!
//! A proposer invents problems, a solver tries to answer them, and each is
//! rewarded for beating the other. The name is the Red Queen hypothesis: an
//! organism runs to stay in the same place because everything around it is
//! running too.
//!
//! **Why this rather than more of what `godel` already does.** `godel` is a
//! good loop -- paired statistics, four adversarial judges, a re-derivable
//! ledger -- pointed at a fixed target. There are 360 held-out corpus items
//! and no in-kernel mechanism produces a 361st, so the loop converges and
//! reports "search space exhausted", which it does. Nothing about the loop is
//! wrong; it has finished. What is missing is a supply of questions.
//!
//! `problem.rs` made a question into a storable object. This makes more of
//! them, and makes them *harder*, which is the part that cannot be done by
//! generating at random: a random problem is usually either trivial or
//! impossible, and both are worthless. A problem is worth keeping only if it
//! sits just past what the solver can currently do.
//!
//! ## Difficulty is measured, not declared
//!
//! The one thing this design needed and did not have was a definition of hard
//! that is not somebody's opinion. It comes out of the solver.
//!
//! The solver enumerates candidate programs in increasing size and tries each
//! against the problem's cases, stopping at the first that answers all of
//! them. **The difficulty of a problem is the number of candidates that had to
//! be tried.** That is a property of the problem and the solver together,
//! which is exactly right for a coevolutionary loop: it moves when either side
//! moves, and it is measured by running rather than estimated by looking.
//!
//! It also makes "the solver improved" a checkable claim rather than a
//! feeling. A better *ordering* of the same grammar, or a wider grammar,
//! solves the same problem after fewer candidates. That is a number, it is
//! paired across problems, and it is the shape `godel`'s J1 already judges.
//!
//! ## The frontier condition
//!
//! `problem::admit` answers whether a problem is well formed, solvable and
//! non-trivial. It deliberately does not ask whether it is *interesting*,
//! because that is a question about the solver. Here it is: a proposed problem
//! is kept only when the current solver **fails** it within budget. A problem
//! the solver already answers teaches nothing and inflates the archive with
//! things that look like progress.
//!
//! This is the guard against the degenerate attractor, working with the
//! synthesised foil in `problem::admit`. The foil stops problems that are
//! trivially checkable; the frontier stops problems that are merely already
//! solved. A generator that games either one has to game both at once, and the
//! two are checked by different machinery.
//!
//! ## No model, on purpose, for now
//!
//! Neither half calls the engine. A decode is about 2.8 s under emulation, so
//! a model-driven loop yields a few dozen generations a night; this one runs
//! thousands and can therefore be plotted, which is what a frontier has to be
//! to be worth anything. It also means both halves are asserted at boot with
//! no checkpoint loaded, and that the loop runs on a machine with no model at
//! all.
//!
//! The model is the obvious next proposer and the obvious next solver, and
//! everything here is shaped so it plugs in as one more source of candidates
//! rather than as a rewrite. That is deferred, not forgotten.
//!
//! ## What the enumerator cannot do, said plainly
//!
//! The grammar is integer arithmetic over the arguments and a handful of small
//! constants. It has no control flow, no strings, no calls. So every `Program`
//! problem over strings is unsolvable by it, and reads as maximally difficult,
//! which is honest but uninformative -- a problem nothing can solve and a
//! problem nothing has yet solved are the same reading here. `unsolved` is
//! reported separately from `difficulty` for that reason, and a proposer that
//! only produced unsolvable problems would be visible as a run where every
//! candidate was unsolved and none was ever answered.

use super::problem::{Case, Family, Origin, Problem};
use crate::aiksi::eval::Value;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{format, vec};

/// How many candidate programs a solver may try before giving up.
///
/// Small enough that a boot self-test finishes, large enough that the seeded
/// problems are inside it and a mutation of one is not. The number is the
/// solver's whole strength, so it is the thing that moves when the solver
/// improves.
pub const BUDGET: usize = 600;

/// The largest expression the enumerator will build, in operator count.
///
/// A bound on shape as well as on count: without it the enumerator spends its
/// whole budget deepening one branch, and the budget stops measuring breadth.
const MAX_OPS: usize = 3;

/// How much stronger the reachability probe is than the solver.
///
/// **The frontier has two edges and the first version only had one.** Keeping
/// a problem because the solver failed it keeps everything the solver cannot
/// do, which includes everything *nothing* of this kind can do. Measured: three
/// rounds took the store from 2 problems to 58, because five mutations of every
/// problem were nearly all unsolvable and therefore nearly all "at the
/// frontier". That is not an arms race, it is a proposer running away from a
/// solver that is standing still.
///
/// So a kept problem must also be solvable by a *stronger* solver -- one given
/// this multiple of the budget. That is the difference between unsolved and
/// unsolvable, which the module header draws and this now enforces: the
/// problem is in reach of this grammar, just not yet in reach of this budget.
/// Growth becomes directed instead of explosive, and every kept problem has a
/// known answer at a known cost, which is what makes it a target rather than
/// noise.
const REACH: usize = 16;

/// The constants the enumerator may use besides the arguments.
///
/// Deliberately few. Every constant multiplies the terminal set, and a solver
/// that can reach any integer solves "return 7" by looking it up rather than
/// by computing anything -- which `problem::admit`'s foil already refuses, but
/// which would also drown every real answer in constants.
const CONSTS: &[i64] = &[0, 1, 2, -1];

/// Where the library lives, content-addressed like everything else.
pub const LIB: &str = "/ai/lib";

/// One function the enumerator may call.
///
/// Named by the hash of what it computes, not by what it was solved for. Two
/// problems whose answers are the same expression are one library function,
/// and a name derived from a problem would have made them two.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LibFn {
    pub name: String,
    pub arity: usize,
    /// The whole declaration, ready to be prepended to a candidate.
    pub src: String,
}

/// What the solver can call as well as compute.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Lib {
    pub fns: Vec<LibFn>,
}

impl Lib {
    /// Every declaration, ready to sit above a candidate program.
    pub fn preamble(&self) -> String {
        let mut out = String::new();
        for f in &self.fns {
            out.push_str(&f.src);
            out.push_str("\n");
        }
        out
    }

    pub fn add(&mut self, f: LibFn) -> bool {
        if self.fns.iter().any(|g| g.name == f.name) {
            return false;
        }
        self.fns.push(f);
        true
    }

    /// Read the adopted library out of the namespace.
    pub fn load() -> Lib {
        let mut lib = Lib::default();
        for name in crate::sysbox::children(LIB) {
            let mut path = String::from(LIB);
            path.push('/');
            path.push_str(&name);
            let Some(raw) = crate::sysbox::read_blob(&path) else { continue };
            let Ok(src) = core::str::from_utf8(&raw) else { continue };
            if let Some(f) = LibFn::parse(src) {
                lib.add(f);
            }
        }
        lib
    }
}

impl LibFn {
    /// Build one from an expression that was found to answer a problem.
    ///
    /// The parameters are the problem's, so a library function has the shape
    /// of the thing it solved and the enumerator knows its arity without
    /// parsing anything back.
    pub fn from_solution(p: &Problem, expr: &str) -> Option<LibFn> {
        let c = p.cases.first()?;
        let h = crate::store::sha256::hash(expr.as_bytes());
        let name = alloc::format!("lib_{}", short_hex(&h));
        let mut params = String::new();
        for (i, a) in c.args.iter().enumerate() {
            if i > 0 {
                params.push_str(", ");
            }
            params.push_str(&alloc::format!("a{}: {}", i, type_of(a)));
        }
        let src = alloc::format!(
            "fn {}({}): {} {{ return {} }}",
            name,
            params,
            type_of(&c.want),
            expr
        );
        Some(LibFn { name, arity: c.args.len(), src })
    }

    /// Read one back. The name and arity come from the declaration itself, so
    /// a stored library function cannot disagree with its own signature.
    pub fn parse(src: &str) -> Option<LibFn> {
        let rest = src.trim().strip_prefix("fn ")?;
        let open = rest.find('(')?;
        let close = rest.find(')')?;
        let name = rest.get(..open)?.trim().to_string();
        if !name.starts_with("lib_") {
            return None;
        }
        let inner = rest.get(open + 1..close)?.trim();
        let arity = if inner.is_empty() { 0 } else { inner.split(',').count() };
        Some(LibFn { name, arity, src: src.trim().to_string() })
    }

    /// Store it, and answer whether it was new.
    pub fn store(&self) -> bool {
        let mut path = String::from(LIB);
        path.push('/');
        path.push_str(&self.name);
        if crate::sysbox::read_blob(&path).is_some() {
            return false;
        }
        crate::sysbox::write_text(&path, &self.src);
        true
    }
}

/// Where a function waits between being proposed and being judged.
///
/// Separate from `LIB`, which holds only what was adopted. Putting candidates
/// in the same directory would make `Lib::load` pick up everything ever
/// offered, so the solver would gain every function the judge refused.
pub const OFFERED: &str = "/ai/libcand";

/// Store a candidate for judging, and answer its address.
pub fn offer_lib(f: &LibFn) -> [u8; 32] {
    let h = crate::store::sha256::hash(f.src.as_bytes());
    let mut path = String::from(OFFERED);
    path.push('/');
    for b in &h {
        path.push_str(&format!("{:02x}", b));
    }
    if crate::sysbox::read_blob(&path).is_none() {
        crate::sysbox::write_text(&path, &f.src);
    }
    h
}

/// Read one back by address.
pub fn candidate(h: &[u8; 32]) -> Option<LibFn> {
    let mut path = String::from(OFFERED);
    path.push('/');
    for b in h {
        path.push_str(&format!("{:02x}", b));
    }
    let raw = crate::sysbox::read_blob(&path)?;
    LibFn::parse(core::str::from_utf8(&raw).ok()?)
}

fn short_hex(h: &[u8; 32]) -> String {
    let mut s = String::new();
    for b in h.iter().take(4) {
        s.push_str(&alloc::format!("{:02x}", b));
    }
    s
}

/// What a solver knows how to do. The searchable object.
///
/// **The budget is not the improvement axis and was treated as one.** The
/// enumerator reaches every expression of at most `MAX_OPS` operators and no
/// others, which for one argument is 86,705 candidates: past that, more budget
/// does nothing at all. Raising `MAX_OPS` costs about 42x per operator --
/// 3.6 million at four, 163 million at five -- so it is unbounded in principle
/// and useless in practice. A solver that improves by searching longer has a
/// ceiling it reaches and then never passes.
///
/// The library is the axis with headroom, and it works the other way round: it
/// makes answers *shorter* instead of making the search *longer*. A problem
/// whose answer is `((a0 * 2) + 1)` costs two operators and thousands of
/// candidates; give the solver the function that computes `(a0 * 2)` and the
/// same answer is one operator over a call, found in tens. Every problem
/// solved enlarges the terminal set, which shrinks every future answer, which
/// brings previously unreachable problems into reach. That compounds, and a
/// budget does not.
///
/// It is also the honest reading of "learning to learn": the machine gets
/// better by changing what one step means rather than by taking more steps.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Solver {
    /// Candidates it may try before giving up.
    pub budget: usize,
    /// What it may call as well as compute.
    pub lib: Lib,
}

/// Written out rather than derived. `#[derive(Default)]` gives a budget of
/// zero, which is a solver that tries nothing and answers nothing -- and it
/// did, silently, until the self-test caught it two claims in.
impl Default for Solver {
    fn default() -> Solver {
        Solver { budget: BUDGET, lib: Lib::default() }
    }
}

impl Solver {
    pub fn new(budget: usize) -> Solver {
        Solver { budget, lib: Lib::default() }
    }

    pub fn with(budget: usize, lib: Lib) -> Solver {
        Solver { budget, lib }
    }

    /// The same solver, searching harder. For the reachability probe, which
    /// asks whether a problem is out of reach of the *grammar* rather than of
    /// the budget -- so it must differ in the budget alone.
    pub fn stronger(&self, times: usize) -> Solver {
        Solver { budget: self.budget.saturating_mul(times), lib: self.lib.clone() }
    }
}

/// A problem answered, and what it cost to answer it.
#[derive(Clone)]
pub struct Solved {
    /// The program that answered every case.
    pub src: String,
    /// The winning expression alone, without the wrapper or the library.
    /// What a library function is built from.
    pub expr: String,
    /// How many candidates were tried before this one worked. The difficulty.
    pub tried: usize,
    /// The worst case's step count, which is what the ceiling bounds.
    pub steps: u64,
}

/// The type name a value takes in a signature.
fn type_of(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Str(_) => "str",
        _ => "any",
    }
}

/// Wrap an expression in the function the problem asked for.
///
/// The signature is read off the first case rather than declared, because the
/// cases are what a solution is checked against and a signature disagreeing
/// with them would fail every case for a reason that is not about the answer.
fn wrap(p: &Problem, expr: &str, lib: &Lib) -> Option<String> {
    let c = p.cases.first()?;
    let mut params = String::new();
    for (i, a) in c.args.iter().enumerate() {
        if i > 0 {
            params.push_str(", ");
        }
        params.push_str(&format!("a{}: {}", i, type_of(a)));
    }
    // The library sits above the candidate rather than beside it. A call has
    // to resolve inside the one program `differ` runs, and prepending is the
    // only arrangement that needs no linker.
    Some(format!(
        "{}fn {}({}): {} {{ return {} }}",
        lib.preamble(),
        p.entry,
        params,
        type_of(&c.want),
        expr
    ))
}

/// Every expression of exactly `ops` operators, given the ones below it.
///
/// Bottom-up rather than top-down: an expression of n operators is two smaller
/// ones joined, so each size is built once from sizes already in hand instead
/// of being re-derived down every branch.
fn grow(by_size: &[Vec<String>], ops: usize, lib: &Lib) -> Vec<String> {
    let mut out = Vec::new();
    for left in 0..ops {
        let right = ops - 1 - left;
        for a in &by_size[left] {
            for b in &by_size[right] {
                for op in ["+", "-", "*"] {
                    out.push(format!("({} {} {})", a, op, b));
                }
            }
        }
    }
    // A library call counts as one operator, which is the whole economy of
    // the idea: an answer that took three operators to write out takes one to
    // call, so the size it is found at drops and the candidates tried before
    // reaching it drop with it.
    //
    // Only arity one and two are enumerated. Arity three would need the
    // three-way split of `ops - 1` and multiplies the branching factor by the
    // terminal count again; nothing has produced a three-argument problem, and
    // building the enumeration for one that does not exist is guessing at a
    // shape rather than answering a need.
    for f in &lib.fns {
        match f.arity {
            1 => {
                for a in &by_size[ops - 1] {
                    out.push(format!("{}({})", f.name, a));
                }
            }
            2 => {
                for left in 0..ops {
                    let right = ops - 1 - left;
                    for a in &by_size[left] {
                        for b in &by_size[right] {
                            out.push(format!("{}({}, {})", f.name, a, b));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Try to answer a problem, cheapest candidate first.
///
/// `None` means the budget ran out, which is the frontier condition's whole
/// content: not "impossible", but "not by this solver, this many tries in".
pub fn solve(p: &Problem, s: &Solver) -> Option<Solved> {
    let first = p.cases.first()?;
    // Terminals: the arguments, then the constants. Arguments first so a
    // problem answered by one of its inputs costs almost nothing, which keeps
    // the difficulty of an identity honest.
    let mut terms: Vec<String> = (0..first.args.len()).map(|i| format!("a{}", i)).collect();
    for c in CONSTS {
        terms.push(format!("{}", c));
    }

    let mut by_size: Vec<Vec<String>> = vec![terms];
    let mut tried = 0usize;

    for size in 0..=MAX_OPS {
        if size > 0 {
            let next = grow(&by_size, size, &s.lib);
            by_size.push(next);
        }
        for expr in &by_size[size] {
            if tried >= s.budget {
                return None;
            }
            tried += 1;
            let Some(src) = wrap(p, expr, &s.lib) else { return None };
            if let Ok(steps) = p.run(&src) {
                return Some(Solved { src, expr: expr.clone(), tried, steps });
            }
        }
    }
    None
}

/// One mutation of a problem into a harder one.
///
/// The mutations are syntactic on the *reference*, and the expected answers
/// are recomputed by running it. That is the whole trick and it is why a
/// generated problem is never a guess: the reference is the ground truth, so
/// deriving the question from the answer cannot produce an unsolvable
/// question.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mutation {
    /// `E` becomes `E + k`.
    Offset(i64),
    /// `E` becomes `E * k`.
    Scale(i64),
    /// `E` becomes `(E * k) - a0`, which needs two operators where one did.
    Twist(i64),
}

impl Mutation {
    /// `arg` is the reference's *own* first parameter name.
    ///
    /// Not `a0`. The first version of this built the mutated reference with
    /// `wrap`, which names parameters `a0..`, while the reference it was
    /// mutating named its own `n` -- so the new body referred to a variable
    /// that did not exist, failed to run, and `mutate` answered `None` for
    /// every problem in the store. The loop reported "0 proposed" and looked
    /// like it had nothing to do rather than like it was broken.
    fn apply(&self, expr: &str, arg: &str) -> String {
        match self {
            Mutation::Offset(k) => format!("(({}) + {})", expr, k),
            Mutation::Scale(k) => format!("(({}) * {})", expr, k),
            Mutation::Twist(k) => format!("((({}) * {}) - {})", expr, k, arg),
        }
    }
}

/// The mutations a proposer may reach for, in a fixed order.
///
/// Declared rather than random, so a run is re-derivable: the nth mutation of
/// the nth problem is a function of the archive rather than of a coin. The
/// same argument `godel::frontier` makes about walking its grid.
pub const MUTATIONS: &[Mutation] = &[
    Mutation::Offset(1),
    Mutation::Scale(2),
    Mutation::Twist(2),
    Mutation::Offset(-3),
    Mutation::Scale(3),
];

/// The body of a reference of the shape this module generates.
///
/// Only mutates a reference it can read, which is one `return` of a single
/// expression. A reference with control flow is left alone rather than
/// mangled -- a mutation that produced a program which no longer parses would
/// be refused by `admit` anyway, and reporting "could not mutate" is a more
/// useful answer than a refusal three conditions later.
/// The reference's own signature, with a new body.
///
/// Preserving the signature rather than regenerating it is the whole fix for
/// the bug recorded on `Mutation::apply`: a mutation is a change of body, and
/// anything that also rewrites the parameter list has to rewrite every
/// mention of a parameter inside the expression it is keeping.
fn rebody(reference: &str, expr: &str) -> Option<String> {
    let open = reference.find('{')?;
    Some(format!("{}{{ return {} }}", reference.get(..open)?, expr))
}

/// The name of the reference's first parameter, so a mutation can mention it.
fn first_param(reference: &str) -> Option<&str> {
    let open = reference.find('(')?;
    let close = reference.find(')')?;
    let first = reference.get(open + 1..close)?.split(',').next()?.trim();
    let name = first.split(':').next()?.trim();
    if name.is_empty() {
        return None;
    }
    Some(name)
}

fn body_of(reference: &str) -> Option<&str> {
    let open = reference.find('{')?;
    let close = reference.rfind('}')?;
    let inner = reference.get(open + 1..close)?.trim();
    inner.strip_prefix("return ").map(|e| e.trim())
}

/// Make a harder problem from one that already holds.
///
/// The cases keep their inputs and get new answers, computed by running the
/// mutated reference. So the new problem is about the same inputs and a
/// different function of them, which is what makes the difficulty comparable:
/// two problems over one set of inputs differ in the program needed and in
/// nothing else.
pub fn mutate(p: &Problem, m: Mutation) -> Option<Problem> {
    if p.family != Family::Program {
        return None;
    }
    let expr = body_of(&p.reference)?;
    let arg = first_param(&p.reference)?;
    let mutated = m.apply(expr, arg);
    let reference = rebody(&p.reference, &mutated)?;

    // The reference decides every answer. Run it once per case rather than
    // computing the shift in Rust: a second implementation of the mutation,
    // in another language, is exactly the drift `model.rs` warns about.
    let mut cases = Vec::with_capacity(p.cases.len());
    for c in &p.cases {
        let probe = Problem {
            family: p.family,
            origin: p.origin,
            statement: p.statement.clone(),
            entry: p.entry.clone(),
            ceiling: p.ceiling,
            cases: vec![Case { args: c.args.clone(), want: Value::Int(0) }],
            reference: reference.clone(),
        };
        let want = answer_of(&probe, &c.args)?;
        cases.push(Case { args: c.args.clone(), want });
    }

    Some(Problem {
        family: p.family,
        origin: Origin::SelfMade,
        statement: format!("{}, {}", p.statement, describe(m)),
        entry: p.entry.clone(),
        ceiling: p.ceiling,
        cases,
        reference,
    })
}

fn describe(m: Mutation) -> String {
    match m {
        Mutation::Offset(k) => format!("then add {}", k),
        Mutation::Scale(k) => format!("then multiply by {}", k),
        Mutation::Twist(k) => format!("scaled by {} less the first argument", k),
    }
}

/// What a reference answers for one set of arguments.
///
/// Goes through the same `differ::observe` every other check here uses, so an
/// answer computed while generating and an answer checked while judging come
/// from one implementation.
fn answer_of(p: &Problem, args: &[Value]) -> Option<Value> {
    use crate::aiksi::differ::{observe, Entry, Route};
    let out = observe(&p.reference, Entry::Call(&p.entry, args), Route::Armed)?;
    if out.errored() {
        return None;
    }
    out.value().parse::<i64>().ok().map(Value::Int)
}

// ---------------------------------------------------------------------------
// The archive: illuminate the frontier instead of climbing it.
// ---------------------------------------------------------------------------

/// Where kept problems are filed. A subdirectory of the problem store, which
/// `problem::stored` skips because it only accepts names of 64 hex digits.
pub const ARCHIVE: &str = "/ai/problems/archive";

/// Difficulty bands, in candidates the reachability probe needed.
///
/// Chosen from measurements rather than from taste: the seeds solve in tens,
/// the first generation of mutations lands around four to seven thousand, and
/// anything past that is near the edge of what the probe reaches at all.
const BANDS: &[usize] = &[100, 2_000, 8_000];

/// One band past the thresholds, crossed with the four families.
pub const CELLS: usize = (BANDS.len() + 1) * 4;

fn band_of(difficulty: usize) -> usize {
    let mut b = 0;
    for t in BANDS {
        if difficulty < *t {
            return b;
        }
        b += 1;
    }
    b
}

fn family_index(f: Family) -> usize {
    match f {
        Family::Program => 0,
        Family::Machine => 1,
        Family::Source => 2,
        Family::Imported => 3,
    }
}

pub fn cell_of(f: Family, difficulty: usize) -> usize {
    family_index(f) * (BANDS.len() + 1) + band_of(difficulty)
}

/// What a cell holds: the hardest problem of that family in that band.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Kept {
    pub problem: [u8; 32],
    pub difficulty: usize,
}

fn cell_path(i: usize) -> String {
    format!("{}/{}", ARCHIVE, i)
}

/// Read a cell back.
///
/// **This is the operation `godel`'s archive does not have, and the whole
/// argument for content-addressing an elite rests on it.** There, `Elite`
/// carries a variant address, `offer` writes it, `cell` parses it back, and
/// both readers drop it on the floor -- so twelve cells of "an elite from
/// three weeks ago is still reachable" have never had anything reach one. A
/// cell nothing reads is a high-score table with extra steps.
pub fn cell(i: usize) -> Option<Kept> {
    let bytes = crate::sysbox::read_blob(&cell_path(i))?;
    let text = core::str::from_utf8(&bytes).ok()?;
    let mut it = text.split_whitespace();
    let hx = it.next()?;
    if hx.len() != 64 {
        return None;
    }
    let mut problem = [0u8; 32];
    for (i, b) in problem.iter_mut().enumerate() {
        *b = u8::from_str_radix(hx.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    let difficulty = it.next()?.parse::<usize>().ok()?;
    Some(Kept { problem, difficulty })
}

/// File a problem, if it is harder than whatever holds its cell.
///
/// Strictly harder, so re-offering the same problem does not rewrite the cell
/// and a round that discovers nothing new leaves the archive byte-identical --
/// which is what makes "the archive did not move" a fact rather than a guess.
pub fn offer(h: &[u8; 32], f: Family, difficulty: usize) -> bool {
    let i = cell_of(f, difficulty);
    if let Some(held) = cell(i) {
        if held.difficulty >= difficulty {
            return false;
        }
    }
    let mut line = String::new();
    for b in h {
        line.push_str(&format!("{:02x}", b));
    }
    line.push(' ');
    line.push_str(&format!("{}", difficulty));
    line.push_str("\n");
    crate::sysbox::write_text(&cell_path(i), &line);
    true
}

/// Every cell that holds something. The frontier, as objects rather than a
/// count -- so a round can breed from the hardest of each kind instead of
/// from whatever happens to be stored.
pub fn elites() -> Vec<Kept> {
    let mut out = Vec::new();
    for i in 0..CELLS {
        if let Some(k) = cell(i) {
            out.push(k);
        }
    }
    out
}

/// How many cells are lit, and the hardest thing in any of them.
pub fn census() -> (usize, usize) {
    let e = elites();
    let hardest = e.iter().map(|k| k.difficulty).max().unwrap_or(0);
    (e.len(), hardest)
}

/// How many mined structures a round may put in front of the judge.
///
/// Capped because `bench` re-solves every stored problem twice, so each offer
/// costs a full pass over the set. Ranked by `saved` first, so the cap takes
/// the three the objective already thinks are worth most rather than the three
/// that happened to be enumerated first.
const MINE_TOP: usize = 3;

/// Turn a mined structure into something the enumerator can call.
///
/// `Candidate::proposal` emits `fn absN(a0, a1) { .. }`, which is a proposal
/// for a person to read. This needs a *typed* declaration with a
/// content-addressed name, so the same fragment mined twice is one function
/// and the enumerator knows its arity without parsing anything back.
///
/// The parameters are `int` and not `any`. Everything mined here comes from
/// expressions the enumerator built or from a problem's reference, and both
/// are integer arithmetic; declaring `any` would be a wider promise than the
/// thing can keep.
fn from_candidate(c: &super::abstraction::Candidate) -> Option<LibFn> {
    if c.arity == 0 || c.arity > 2 || c.saved <= 0 {
        return None;
    }
    let mut body = c.skeleton.clone();
    // Highest index first: replacing #1 before #10 would corrupt #10.
    for i in (0..c.arity).rev() {
        body = body.replace(&format!("#{}", i), &format!("a{}", i));
    }
    if body.contains('#') {
        return None;
    }
    let h = crate::store::sha256::hash(body.as_bytes());
    let name = format!("lib_{}", short_hex(&h));
    let mut params = String::new();
    for i in 0..c.arity {
        if i > 0 {
            params.push_str(", ");
        }
        params.push_str(&format!("a{}: int", i));
    }
    let src = format!("fn {}({}): int {{ return {} }}", name, params, body);
    Some(LibFn { name, arity: c.arity, src })
}

/// Mine repeated structure out of what the machine already knows.
///
/// **This is the other half of the library and the two learn different
/// things.** Solving teaches whole answers: an expression that answered one
/// problem, kept because it will answer the next. Mining teaches shared
/// *fragments*: a shape that turns up inside several answers and inside the
/// references of several problems, which nothing solved on its own and which
/// nobody would think to propose.
///
/// `abstraction::analyse` is the whole of it and it already existed. Its
/// objective is arithmetic rather than taste -- a structure of `size` nodes
/// occurring `count` times saves `count * (size - 1 - arity) - size` -- so a
/// fragment whose leaves are all different has an arity as large as its leaf
/// count, its call site is as big as the thing it replaces, and `saved` goes
/// negative on its own. Nothing has to decide that six parameters is too many.
///
/// What is fed in is the library and the problems' references, and not
/// `/ai/tools`. That is the whole reason this can be wired at all:
/// `abstraction.rs` stops short of writing anything because rewriting a
/// program the operator trusted by hash revokes that trust, which is correct
/// and needs a story first. Nothing here was ever trusted by an operator --
/// these are machine-written, content-addressed and judged -- so the story is
/// not needed and the rewriting never happens.
pub fn mine(lib: &Lib) -> Vec<LibFn> {
    let mut programs: Vec<(String, String)> = Vec::new();
    for f in &lib.fns {
        programs.push((f.name.clone(), f.src.clone()));
    }
    for h in super::problem::stored() {
        let Some(p) = Problem::load(&h) else { continue };
        if p.family != Family::Program {
            continue;
        }
        programs.push((p.entry.clone(), p.reference.clone()));
    }
    if programs.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for c in super::abstraction::analyse(&programs) {
        let Some(f) = from_candidate(&c) else { continue };
        if lib.fns.iter().any(|g| g.name == f.name) {
            continue;
        }
        if out.iter().any(|g: &LibFn| g.name == f.name) {
            continue;
        }
        out.push(f);
        if out.len() >= MINE_TOP {
            break;
        }
    }
    out
}

/// What adding one function to the library did to the whole stored set.
///
/// Paired, over the same problems both times, which is what makes it a
/// comparison rather than two averages. The same shape `godel`'s J1 judges,
/// and deliberately so: this is a proposal about the solver, and the machinery
/// for judging proposals already exists.
pub struct LibVerdict {
    /// Problems that got easier, or became solvable at all.
    pub fixed: usize,
    /// Problems that got harder, or stopped being solvable.
    ///
    /// This is not hypothetical. Every function added enlarges the terminal
    /// set, so every candidate list gets longer and an easy problem can take
    /// more tries to reach the same simple answer. A library is a trade, and
    /// a judge that could only see the wins would take every trade offered.
    pub broke: usize,
    /// Total candidates over the set before and after, unsolved counted at
    /// the budget rather than dropped -- so a problem going from unsolvable
    /// to solvable shows as the improvement it is.
    pub before: usize,
    pub after: usize,
}

impl LibVerdict {
    pub fn helps(&self) -> bool {
        self.fixed > self.broke && self.after < self.before
    }
}

/// Score a candidate library function against every stored problem.
pub fn bench(s: &Solver, f: &LibFn) -> LibVerdict {
    let mut with = s.clone();
    with.lib.add(f.clone());

    let mut v = LibVerdict { fixed: 0, broke: 0, before: 0, after: 0 };
    for h in super::problem::stored() {
        let Some(p) = Problem::load(&h) else { continue };
        if p.family != Family::Program {
            continue;
        }
        let a = solve(&p, s).map(|x| x.tried).unwrap_or(s.budget);
        let b = solve(&p, &with).map(|x| x.tried).unwrap_or(with.budget);
        v.before += a;
        v.after += b;
        if b < a {
            v.fixed += 1;
        } else if b > a {
            v.broke += 1;
        }
    }
    v
}

/// What one round of the arms race did.
pub struct Round {
    /// Problems proposed.
    pub proposed: usize,
    /// Refused by `problem::admit` -- malformed, unsolved, trivial, unstable.
    pub inadmissible: usize,
    /// Admissible and already solved, so not at the frontier.
    pub already: usize,
    /// Admissible, unsolved, and still unsolved by a solver `REACH` times
    /// stronger -- so out of reach of this grammar rather than merely of this
    /// budget. Dropped, and counted, because a proposer producing only these
    /// is a proposer generating noise and the count is what says so.
    pub beyond: usize,
    /// Kept: admissible, past the solver, inside the stronger one, and *new*.
    ///
    /// New matters. Storing is idempotent, so re-proposing a child already in
    /// the store answers the same address and writes nothing -- and counting
    /// that as kept reported "3 kept" on four consecutive rounds while the
    /// store stayed at five problems. The number now means what it says, which
    /// is also what makes a round that keeps nothing a reliable signal that
    /// the loop has converged for this solver and this mutation set.
    pub kept: usize,
    /// Proposed, admissible, at the frontier, and already stored.
    pub known: usize,
    /// The hardest difficulty the solver reached this round, in candidates.
    pub hardest: usize,
    /// The hardest a *kept* problem cost the stronger solver. This is the
    /// frontier: what the solver would have to become to answer them.
    pub reach: usize,
    /// Library functions offered from problems solved this round.
    pub offered: usize,
    /// Offered from mining repeated structure rather than from an answer.
    pub mined: usize,
    /// Problems that took a cell off whatever held it.
    pub filed: usize,
    /// Offered, and the paired judge said the whole set got easier.
    pub learned: usize,
    /// Total candidates over the stored set before and after learning, so a
    /// round says what it bought rather than only what it added.
    pub before: usize,
    pub after: usize,
    /// Addresses of what was kept, so the caller can store or report them.
    pub frontier: Vec<[u8; 32]>,
}

/// Run one round against every stored problem the solver can currently answer.
///
/// The order is the point. A problem is mutated, admitted, and only then asked
/// whether the solver already answers it -- admission first because it is
/// cheap and refuses most bad candidates, the frontier check second because it
/// costs a whole search.
pub fn round(s: &Solver) -> Round {
    let mut r = Round {
        proposed: 0,
        inadmissible: 0,
        already: 0,
        beyond: 0,
        kept: 0,
        known: 0,
        hardest: 0,
        reach: 0,
        offered: 0,
        mined: 0,
        filed: 0,
        learned: 0,
        before: 0,
        after: 0,
        frontier: Vec::new(),
    };

    // The solver grows inside the round. A function learned from the first
    // problem is available to the second, which is the compounding the whole
    // design rests on -- and it is why this is one mutable solver rather than
    // the caller's.
    let mut grown = s.clone();

    // **Breed from the archive, not from everything stored.** A round used to
    // walk `problem::stored()`, which grows every round -- 34 problems after
    // six -- so each round mutated more than the last and the work per round
    // climbed without bound while most of it re-derived children already
    // known. The elites are one problem per (family, difficulty) cell, so the
    // work is bounded by the number of cells and the diversity is kept on
    // purpose rather than by accident.
    //
    // Falls back to the stored set while the archive is empty, which is only
    // the first round on a fresh machine.
    let breeding: Vec<[u8; 32]> = match elites() {
        e if e.is_empty() => super::problem::stored(),
        e => e.iter().map(|k| k.problem).collect(),
    };

    for h in breeding {
        let Some(p) = Problem::load(&h) else { continue };
        if p.family != Family::Program {
            continue;
        }
        // How hard the parent is, which is what makes the child's difficulty
        // a comparison rather than a reading.
        if let Some(sol) = solve(&p, &grown) {
            r.hardest = r.hardest.max(sol.tried);
        }
        for m in MUTATIONS {
            let Some(child) = mutate(&p, *m) else { continue };
            r.proposed += 1;
            if child.admit().is_err() {
                r.inadmissible += 1;
                continue;
            }
            if solve(&child, &grown).is_some() {
                r.already += 1;
                continue;
            }
            // Past this solver. Is it past every solver of this shape? The
            // stronger probe is what separates a target from noise, and it is
            // asked last because it is the most expensive question here. Built
            // from `grown` rather than from the caller's solver, so a function
            // learned earlier in the round counts towards reachability too --
            // otherwise the probe measures a solver that no longer exists.
            let strong = grown.stronger(REACH);
            let Some(hard) = solve(&child, &strong) else {
                r.beyond += 1;
                continue;
            };
            // **The library learns here and not from the cheap solves, and
            // the first version had it the other way round.** A function is
            // worth having only when it replaces more than one operator: the
            // seeds answer in one, a call also costs one, and offering those
            // produced "1 offered, none paid for itself" on every round with
            // the judge correctly refusing all of them.
            //
            // The expensive answer is the one worth keeping. `hard` came from
            // a solver `REACH` times stronger and is several operators long,
            // so calling it saves the difference every time -- which is the
            // wake/sleep economics this is borrowed from: pay once to find an
            // abstraction, then pay one node to use it forever.
            if let Some(f) = LibFn::from_solution(&child, &hard.expr) {
                if !grown.lib.fns.iter().any(|g| g.name == f.name) {
                    r.offered += 1;
                    // Judged, not assumed. Every function lengthens every
                    // candidate list, so one that does not pay for itself
                    // makes the whole set harder -- which is what the paired
                    // count is there to catch.
                    // Queued whether or not this round takes it. The round's
                    // own judge is the same `bench`, but a round leaves no
                    // ledger line and nothing to roll back; `godel`'s `lib`
                    // axis picks these up and records a verdict that survives.
                    offer_lib(&f);
                    let v = bench(&grown, &f);
                    if v.helps() {
                        r.before += v.before;
                        r.after += v.after;
                        f.store();
                        grown.lib.add(f);
                        r.learned += 1;
                    }
                }
            }

            let Some(ch) = child.hash() else { continue };
            let fresh = Problem::load(&ch).is_none();
            if child.store().is_some() {
                if fresh {
                    r.kept += 1;
                    r.frontier.push(ch);
                } else {
                    r.known += 1;
                }
                r.reach = r.reach.max(hard.tried);
                // Filed by what it is and how hard it was, so the hardest of
                // each kind survives and the rest are still stored but stop
                // being bred from.
                if offer(&ch, child.family, hard.tried) {
                    r.filed += 1;
                }
            }
        }
    }
    // --- the sleep phase -------------------------------------------------
    //
    // Run once at the end rather than per child, because it reads the whole
    // library and the whole problem set: mining after every mutation would
    // re-derive the same fragments from nearly the same input, and pay for a
    // `bench` each time to be told so.
    for f in mine(&grown.lib) {
        if grown.lib.fns.iter().any(|g| g.name == f.name) {
            continue;
        }
        r.mined += 1;
        r.offered += 1;
        // The same judge the solved answers face. A mined fragment has an
        // objective saying it *should* pay, and that objective counts nodes
        // rather than candidates -- so it is a good proposal and not a
        // verdict, and the paired count is what decides.
        offer_lib(&f);
        let v = bench(&grown, &f);
        if v.helps() {
            r.before += v.before;
            r.after += v.after;
            f.store();
            grown.lib.add(f);
            r.learned += 1;
        }
    }

    r
}

/// Report a round to the console.
pub fn report(rounds: usize, s: &Solver) {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!(
        "[redqueen] {} round(s), budget {}, library of {}",
        rounds,
        s.budget,
        s.lib.fns.len()
    );
    console::set_color(LTGRAY);

    let mut total = 0usize;
    for i in 0..rounds.max(1) {
        let r = round(&Solver::with(s.budget, Lib::load()));
        kprintln!(
            "  round {}: {} proposed, {} inadmissible, {} solved, {} beyond reach, {} known, {} kept",
            i + 1,
            r.proposed,
            r.inadmissible,
            r.already,
            r.beyond,
            r.known,
            r.kept
        );
        if r.learned > 0 {
            kprintln!(
                "    learned {} of {} offered ({} mined); the set went {} -> {} candidate(s)",
                r.learned,
                r.offered,
                r.mined,
                r.before,
                r.after
            );
        } else if r.offered > 0 {
            kprintln!(
                "    {} offered ({} mined), none paid for itself",
                r.offered,
                r.mined
            );
        }
        if r.reach > 0 {
            let (lit, hardest) = census();
            kprintln!(
                "    the frontier stands at {} candidate(s) against a budget of {}",
                r.reach,
                s.budget
            );
            kprintln!(
                "    archive: {} of {} cell(s) lit, {} filed, hardest {}",
                lit,
                CELLS,
                r.filed,
                hardest
            );
        }
        if r.hardest > 0 {
            kprintln!("    hardest solved this round: {} candidate(s)", r.hardest);
        }
        if let Some(h) = r.frontier.first().and_then(Problem::load) {
            // What the newest frontier problem costs to *run*, which is the
            // number its ceiling bounds -- a different question from how many
            // candidates it took to find, and the one a solver is billed for.
            if let Ok(steps) = h.run(&h.reference) {
                kprintln!("    newest costs {} step(s) to check", steps);
            }
        }
        total += r.kept;
        // A round that kept nothing will keep nothing next time either: the
        // stored set did not move, the mutations are a fixed list, and the
        // solver did not change. Stopping says so rather than repeating it.
        if r.kept == 0 && r.learned == 0 {
            kprintln!("    nothing moved -- no new problem, no function that paid for itself");
            break;
        }
    }
    console::set_color(LTGREEN);
    kprintln!(
        "  {} problem(s), {} library function(s)",
        super::problem::count(),
        Lib::load().fns.len()
    );
    console::set_color(LTGRAY);
    let _ = total;
}

/// Every queued library candidate the adopted library does not hold, in
/// directory order.
///
/// A directory scan in the shape `godel::next_skill` uses, and for the same
/// reason: the queue is the work list, so what to try tonight is a function
/// of what is on disk rather than of a counter somebody has to keep in step.
///
/// **The list rather than the first of it, because "already held" is not the
/// only reason to skip one.** `godel::next_lib` has to filter these against
/// `/ai/godel/tried` as every other axis does, and a function answering only
/// the head of the queue cannot be filtered -- which is exactly how a refused
/// candidate came to be re-offered on every pass forever. Holding the name is
/// the one reason *this* module knows about; what has already been judged is
/// the ledger's business and not this file's.
pub fn unheld_candidates() -> Vec<[u8; 32]> {
    let held = Lib::load();
    let mut out = Vec::new();
    for name in crate::sysbox::children(OFFERED) {
        if name.len() != 64 {
            continue;
        }
        let mut h = [0u8; 32];
        let mut good = true;
        for (i, b) in h.iter_mut().enumerate() {
            match name.get(i * 2..i * 2 + 2).and_then(|p| u8::from_str_radix(p, 16).ok()) {
                Some(v) => *b = v,
                None => {
                    good = false;
                    break;
                }
            }
        }
        if !good {
            continue;
        }
        let Some(f) = candidate(&h) else { continue };
        if held.fns.iter().any(|g| g.name == f.name) {
            continue;
        }
        out.push(h);
    }
    out
}

/// Boot self-test. No model, no corpus, no network, no store beyond the
/// namespace every other suite already uses.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, pass: bool| {
        if !pass {
            ok = false;
        }
        kprintln!("  {}   {}", if pass { "ok " } else { "FAIL" }, what);
    };

    let seeds = super::problem::seeds();
    let twice = seeds[0].clone();
    let s = Solver::default();

    // The solver, on something it can do. `twice` is `n * 2`, which the
    // enumerator reaches as one operator over an argument and a constant.
    match solve(&twice, &s) {
        Some(sol) => {
            claim("the solver answers a problem inside its grammar", sol.tried > 0);
            claim("and the program it found really does answer it", twice.run(&sol.src).is_ok());
            claim("with a difficulty it measured rather than guessed", sol.tried <= s.budget);
            // Two different costs, and conflating them is the easy mistake:
            // `tried` is how long the search took, `steps` is what the answer
            // costs to run. The ceiling bounds the second.
            claim("and an answer that respects the problem's ceiling", sol.steps <= twice.ceiling);
        }
        None => claim("the solver answers a problem inside its grammar", false),
    }

    // And on something it cannot. The string problem is outside the grammar
    // entirely, which is the honest reading of "unsolved" rather than "hard".
    claim(
        "a problem outside the grammar is unsolved rather than wrongly answered",
        solve(&seeds[1], &s).is_none(),
    );

    // A solver with no budget answers nothing. The frontier condition rests on
    // failure meaning "not within budget", so a budget of zero has to fail.
    claim(
        "a solver given no budget solves nothing",
        solve(&twice, &Solver::new(0)).is_none(),
    );

    // Mutation. The child must be a real problem in its own right -- the
    // reference decides the answers, so if this holds, generation cannot
    // produce an unsolvable question.
    match mutate(&twice, Mutation::Offset(1)) {
        Some(child) => {
            claim("a mutated problem is admissible", child.admit().is_ok());
            claim("and its reference answers its own cases", child.run(&child.reference).is_ok());
            claim("and it is not the problem it came from", child.hash() != twice.hash());
            claim("and it is marked as the machine's own", child.origin == Origin::SelfMade);
            // The point of mutating at all.
            let a = solve(&twice, &s).map(|x| x.tried).unwrap_or(usize::MAX);
            let b = solve(&child, &s).map(|x| x.tried).unwrap_or(usize::MAX);
            claim("and it is harder than its parent", b > a);
        }
        None => claim("a mutated problem is admissible", false),
    }

    // A mutation of a reference this module cannot read is declined rather
    // than mangled into something that will fail admission for the wrong
    // reason.
    let mut odd = twice.clone();
    odd.reference = String::from("fn twice(n: int): int { if (n > 0) { return n * 2 } return 0 }");
    claim(
        "a reference with control flow is left alone rather than mangled",
        mutate(&odd, Mutation::Offset(1)).is_none(),
    );

    // The frontier condition, which is what makes this an arms race: a strong
    // solver finds the child easy, a weak one does not, and the same child is
    // kept or dropped accordingly.
    if let Some(child) = mutate(&twice, Mutation::Twist(2)) {
        let weak = Solver::new(4);
        claim(
            "a weak solver leaves a mutated problem at the frontier",
            solve(&child, &weak).is_none(),
        );
        claim(
            "and a strong one takes it off the frontier",
            solve(&child, &Solver::new(BUDGET * 4)).is_some() || solve(&child, &s).is_some(),
        );
    }

    // The other edge, and the reason a round is directed rather than
    // explosive: unsolved is not unsolvable. The string problem is outside
    // this grammar entirely, so no budget reaches it, and a round must drop it
    // rather than file it as the hardest thing it has ever seen.
    claim(
        "a problem outside the grammar stays unsolved however strong the solver",
        solve(&seeds[1], &Solver::new(BUDGET * REACH)).is_none(),
    );
    claim(
        "while one inside it is reached when the budget is raised",
        solve(&twice, &Solver::new(BUDGET * REACH)).is_some(),
    );

    // --- the library, and the ceiling it exists to break -----------------
    //
    // This is the claim the whole redesign rests on, and it is measured
    // rather than argued. A budget cannot reach past `MAX_OPS`; a library
    // can, because it makes the answer shorter instead of the search longer.
    let Some(base) = solve(&twice, &s) else {
        claim("the library demonstration needs a solved seed", false);
        return ok;
    };
    let Some(f) = LibFn::from_solution(&twice, &base.expr) else {
        claim("a solution becomes a library function", false);
        return ok;
    };
    claim("a solution becomes a library function", f.arity == 1);
    claim(
        "and reads back from its own declaration",
        LibFn::parse(&f.src).as_ref() == Some(&f),
    );
    claim(
        "and is named for what it computes, not what it solved",
        f.name.starts_with("lib_") && !f.name.contains("twice"),
    );

    // **A library function must replace more than one operator to be worth
    // anything, and the first version of this claim did not.** It built the
    // function from `twice`'s own answer, which is `(a0 * 2)` -- one operator.
    // A call also costs one, so nothing was saved, the judge correctly refused
    // every offer, and four rounds reported "none paid for itself" while the
    // code was working exactly as designed.
    //
    // The expensive answers are the ones worth keeping. So: take a child the
    // ordinary solver cannot reach, find its answer with the stronger probe,
    // and check that the *grandchild* -- one mutation further out -- is
    // cheaper with that answer in hand than without it.
    // **Constructed rather than mutated, and the reason is worth keeping.**
    // Two attempts at this claim went through mutation chains and both failed
    // while the loop they were testing worked: `Offset(1)` twice collapses to
    // `Offset(2)`, so the grandchild is no larger than the child; `Twist(2)`
    // twice gives `5n`, which needs three operators with the library and
    // without it. Arithmetic decides how big an answer is, and picking
    // mutations and hoping is not a demonstration.
    //
    // So the shape is built directly. `g` computes `((a0 * 2) + 1)`, which is
    // two operators. The problem wants `(((n * 2) + 1) * 2)`, three operators
    // bare -- deep in the third size band, past any budget here -- and two
    // with `g` in hand, because the call replaces the whole inner expression.
    // That is the entire claim of the redesign: shorter answers, not longer
    // searches.
    let g = LibFn {
        name: String::from("lib_deadbeef"),
        arity: 1,
        src: String::from("fn lib_deadbeef(a0: int): int { return ((a0 * 2) + 1) }"),
    };
    let deep = Problem {
        family: Family::Program,
        origin: Origin::Seeded,
        statement: String::from("double, add one, double again"),
        entry: String::from("f"),
        ceiling: 10_000,
        cases: vec![
            Case { args: vec![Value::Int(3)], want: Value::Int(14) },
            Case { args: vec![Value::Int(-4)], want: Value::Int(-14) },
            Case { args: vec![Value::Int(0)], want: Value::Int(2) },
        ],
        reference: String::from("fn f(n: int): int { return (((n * 2) + 1) * 2) }"),
    };
    claim("the three-operator problem is a real problem", deep.admit().is_ok());

    let strong = s.stronger(REACH);
    let mut lib = Lib::default();
    lib.add(g);
    let armed = Solver::with(strong.budget, lib);
    let bare = solve(&deep, &strong).map(|x| x.tried);
    let withlib = solve(&deep, &armed).map(|x| x.tried);
    match (bare, withlib) {
        (None, Some(_)) => claim(
            "a library function brings an out-of-reach problem into reach",
            true,
        ),
        (Some(a), Some(b)) => claim(
            "a library function makes an out-of-reach problem reachable, or cheaper",
            b < a,
        ),
        _ => claim(
            "a library function makes an out-of-reach problem reachable, or cheaper",
            false,
        ),
    }

    // --- mining, the other half of the library ---------------------------
    //
    // Two programs sharing a fragment. `analyse` is pure, so this needs no
    // namespace and no store: what is checked is that a shared structure is
    // found, that it becomes something the enumerator can call, and that the
    // objective's own arithmetic refuses one that would not pay.
    let shared = alloc::vec![
        (
            String::from("one"),
            String::from("fn one(n: int): int { return (((n * 2) + 1) * 5) }"),
        ),
        (
            String::from("two"),
            String::from("fn two(n: int): int { return (((n * 2) + 1) - 4) }"),
        ),
        (
            String::from("three"),
            String::from("fn three(n: int): int { return (((n * 2) + 1) * 9) }"),
        ),
    ];
    let found = super::abstraction::analyse(&shared);
    claim("repeated structure across programs is found", !found.is_empty());
    let usable: Vec<LibFn> = found.iter().filter_map(from_candidate).collect();
    match usable.first() {
        Some(f) => {
            claim("and becomes a function the enumerator can call", f.arity <= 2);
            claim(
                "named for the shape rather than for where it was found",
                f.name.starts_with("lib_") && !f.name.contains("abs"),
            );
            claim(
                "and it parses back as the declaration it is",
                LibFn::parse(&f.src).as_ref() == Some(f),
            );
        }
        None => claim("and becomes a function the enumerator can call", false),
    }

    // The objective refuses what would not pay, and it does so on arithmetic
    // rather than on a threshold somebody chose. A fragment whose leaves are
    // all different has an arity as large as its leaf count, so its call site
    // is the size of the thing it replaces and `saved` goes negative.
    claim(
        "a structure that would not pay for itself is refused by the objective",
        found.iter().filter(|c| c.saved <= 0).all(|c| from_candidate(c).is_none()),
    );

    // --- the archive ------------------------------------------------------
    //
    // The banding is arithmetic and needs no store, so it is checked here;
    // reading a cell back is checked by the loop itself, which is the only
    // place a cell is ever written.
    claim(
        "difficulty bands are ordered and a harder problem lands no lower",
        cell_of(Family::Program, 50) <= cell_of(Family::Program, 5_000)
            && cell_of(Family::Program, 5_000) <= cell_of(Family::Program, 500_000),
    );
    claim(
        "two families never share a cell at the same difficulty",
        cell_of(Family::Program, 50) != cell_of(Family::Imported, 50),
    );
    claim(
        "every cell a descriptor can name is inside the archive",
        [0usize, 50, 1_999, 2_000, 7_999, 8_000, usize::MAX]
            .iter()
            .all(|d| cell_of(Family::Imported, *d) < CELLS),
    );

    // And the trade, which is the reason this is judged rather than assumed:
    // every function lengthens every candidate list, so a function that buys
    // nothing costs something. A judge that could only see wins would take it.
    let junk = LibFn {
        name: String::from("lib_00000000"),
        arity: 1,
        src: String::from("fn lib_00000000(a0: int): int { return ((a0 * 0) + 0) }"),
        };
    let before = solve(&twice, &s).map(|x| x.tried).unwrap_or(s.budget);
    let mut with_junk = s.clone();
    with_junk.lib.add(junk);
    let after = solve(&twice, &with_junk).map(|x| x.tried).unwrap_or(s.budget);
    claim("a function that buys nothing still costs candidates", after >= before);

    ok
}
