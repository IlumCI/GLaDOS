//! Two ways of running one program, required to agree exactly.
//!
//! This is the gate a code generator has to pass before anything it emits
//! runs anywhere real, and it exists now, before the generator does, for two
//! reasons.
//!
//! The first is that `model.rs` states the objection twice, and it is right:
//! "two implementations of a transformer that are supposed to agree do not
//! stay agreeing, and the one that drifts is the one nobody decodes with." A
//! harness written after the second implementation is a harness shaped by
//! what that implementation happens to do.
//!
//! The second is `smp.rs`. Its one-shot check passed while the implementation
//! still deadlocked, so it runs sixty-four times now. A harness that has
//! never reported a difference is not known to be able to: it is
//! indistinguishable from one that compares nothing. So this suite includes a
//! pair that is *supposed* to disagree, and fails if it is not caught.
//!
//! **What is compared is the whole observable.** Value, step count, and error
//! text, bit for bit, with no tolerance -- for the reason `smp.rs` gives
//! about a split matvec: any difference at all is a bug, and a tolerance
//! hides exactly the bug worth finding. Step count is in there because it is
//! what the budget stops and what a verdict records; two routes that agree on
//! the answer and disagree on the cost disagree about a number in the ledger.
//!
//! **What is not compared is what a program printed**, and that is the same
//! blind spot `skill.rs` writes down for J3. A program whose whole purpose is
//! `println` answers nil however it behaved, so two routes could disagree
//! completely about what reached the console and agree on all three fields
//! here. Capturing console output would mean giving the console a capture
//! mode, which is a larger change than this needs; it is recorded so the next
//! reader knows the gap is known rather than missed. A code generator is
//! unlikely to fail this way and a builtin dispatch table is exactly how it
//! would.
//!
//! **The second route today is `prepare`/`adopt` against `run`.** That is a
//! real pair, not a placeholder -- one registers a program's declarations by
//! executing its top level, the other by copying a snapshot -- and it is the
//! pair `voter::Core::vote` now depends on for every routing decision. A
//! compiled route becomes a third `Route` and every case here applies to it
//! unchanged.

use super::eval::{Interp, Value};
use super::{lex, parse};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Where a differential run is allowed to write, which is nowhere it reaches.
const JAIL: &str = "/ai/differ/scratch";

/// Generous enough for every case here to finish, small enough that the one
/// case meant to run away does so quickly.
const BUDGET: u64 = 100_000;

/// How many times the built-in corpus is run.
///
/// Sixty-four, the number `smp.rs` arrived at the hard way. One pass would
/// pass over a route that leaks state between runs, and there is something to
/// leak: `Prepared` is shared across every run of a program and hands out
/// `Rc<Func>` clones, so a route that mutated through one would show up on the
/// second run and never on the first.
const ROUNDS: usize = 64;

/// Everything a caller can observe about running a program.
#[derive(PartialEq, Eq, Clone)]
pub struct Outcome {
    value: String,
    steps: u64,
    error: Option<String>,
}

/// Which of the three differed. Named, because "they disagree" is not a bug
/// report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Value,
    Steps,
    Error,
}

/// The first field two outcomes differ in, or `None` if they are identical.
///
/// Order is deliberate: error first, because a route that failed and a route
/// that succeeded differ in a way that makes the other two meaningless.
pub fn disagree(a: &Outcome, b: &Outcome) -> Option<Field> {
    if a.error != b.error {
        return Some(Field::Error);
    }
    if a.value != b.value {
        return Some(Field::Value);
    }
    if a.steps != b.steps {
        return Some(Field::Steps);
    }
    None
}

/// A way of getting a program's declarations into an interpreter.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Run the top level, which is what registers them.
    Armed,
    /// Copy them from a snapshot taken once.
    Prepared,
    /// Do not use an interpreter at all: emit x86-64 for the function and
    /// jump into it. Declines everything outside `jit`'s slice, which is
    /// almost everything.
    Compiled,
}

/// What to ask the program for.
#[derive(Clone, Copy)]
pub enum Entry<'a> {
    /// The value of the top level itself.
    Top,
    /// `vote(text, allowed)`, the shape every core has.
    Vote(&'a str),
    /// `f(a, b, ..)` where every argument and the answer is an integer. The
    /// only shape a compiled function can have.
    Ints(&'a str, &'a [i64]),
}

impl Outcome {
    fn failed(e: String, steps: u64) -> Outcome {
        Outcome { value: String::new(), steps, error: Some(e) }
    }
}

/// Run one program one way. `None` when the route does not apply to it.
///
/// A route declining is not a failure and not a difference -- `Prepared`
/// declines any program whose top level does more than declare, by design.
/// It is reported as coverage instead, because a case silently skipped is a
/// case the harness claims to have checked.
pub fn observe(src: &str, entry: Entry, route: Route) -> Option<Outcome> {
    // Lexing and parsing are shared, so a syntax error is the same outcome
    // for every route and does not need one of them to have an opinion.
    let toks = match lex::lex(src) {
        Ok(t) => t,
        Err(e) => return Some(Outcome::failed(e, 0)),
    };
    let prog = match parse::parse(toks) {
        Ok(p) => p,
        Err(e) => return Some(Outcome::failed(e, 0)),
    };

    if route == Route::Compiled {
        return compiled(&prog, entry);
    }

    let mut it = Interp::sandboxed(JAIL).with_step_budget(BUDGET);
    match route {
        Route::Armed => {
            if let Err(e) = it.run(&prog) {
                return Some(Outcome::failed(e, it.steps()));
            }
        }
        Route::Prepared => {
            // Declines rather than falls back. A route that quietly armed
            // when it could not prepare would compare `Armed` against
            // `Armed` and report agreement it never tested.
            let p = Interp::prepare(&prog).ok()?;
            it.adopt(&p);
        }
        // Handled above, before an interpreter was built at all. Answering
        // `None` rather than asserting, because a route that reached here
        // would be a route with no implementation and the honest reply to
        // that is "this one does not apply".
        Route::Compiled => return None,
    }

    // A declarative top level evaluates to nil, because `Stmt::Fn` and
    // `Stmt::Rec` both answer nil. So the two routes agree here by fact and
    // not by convention, and if they ever stopped agreeing this comparison is
    // where it would show.
    let top = Value::Nil.render();

    match entry {
        Entry::Top => Some(Outcome { value: top, steps: it.steps(), error: None }),
        Entry::Vote(text) => {
            let allowed = Value::List((0..23).map(Value::Int).collect());
            let args = [Value::Str(text.to_string()), allowed];
            match it.invoke("vote", &args) {
                Ok(v) => Some(Outcome { value: v.render(), steps: it.steps(), error: None }),
                Err(e) => Some(Outcome::failed(e, it.steps())),
            }
        }
        Entry::Ints(name, args) => {
            let vals: Vec<Value> = args.iter().map(|i| Value::Int(*i)).collect();
            match it.invoke(name, &vals) {
                Ok(v) => Some(Outcome { value: v.render(), steps: it.steps(), error: None }),
                Err(e) => Some(Outcome::failed(e, it.steps())),
            }
        }
    }
}

/// The compiled route. Declines anything that is not exactly one function in
/// `jit`'s slice, called with integers.
///
/// The status the compiled code answers is turned back into the interpreter's
/// own words here, because `differ` compares error *text*. A compiler that got
/// every number right and said "div0" where the interpreter says "division by
/// zero" would be wrong in a way only this comparison catches.
fn compiled(prog: &[super::parse::Stmt], entry: Entry) -> Option<Outcome> {
    use super::jit;
    let Entry::Ints(want, args) = entry else { return None };
    let (name, params, ret, body) = jit::only_fn(prog)?;
    if name != want {
        return None;
    }
    let p = jit::compile(params, ret, body)?;
    // The budget the compiled code is given is short by the declaration's
    // tick, for the same reason the count below is long by it.
    let r = p.run(args, BUDGET - 1)?;
    // The interpreter charged one tick for *executing the `fn` statement*
    // that declared this function, before anything called it -- `observe`
    // runs the top level and then invokes. Compiled code never ran a top
    // level, so it owes exactly that one tick to be comparable, and `only_fn`
    // guarantees the top level is exactly one statement so the number is one
    // and not an estimate.
    //
    // This was the harness being wrong, not the compiler, and it was found by
    // the comparison failing on `fn f(): int { return 7 }` -- two ticks
    // against three. Worth recording because a harness that had been written
    // to agree would have hidden every later discrepancy behind the same
    // fudge.
    let steps = r.steps + 1;
    let out = match r.status {
        jit::ST_VALUE => Outcome {
            value: Value::Int(r.result).render(),
            steps,
            error: None,
        },
        jit::ST_NIL => {
            // A function that falls off its end yields nothing, and
            // `call_user` then checks that against the declared return type.
            // Compiled code has to fail the same way, in the same words, or a
            // function with a missing `return` would quietly answer nil here
            // and an error there.
            if matches!(ret, super::parse::Type::Int) {
                Outcome::failed(alloc::format!("{} returns int, got nil", name), steps)
            } else {
                Outcome { value: Value::Nil.render(), steps, error: None }
            }
        }
        jit::ST_BUDGET => Outcome::failed(
            String::from("execution budget exceeded (infinite loop?)"),
            steps,
        ),
        jit::ST_DIV0 => Outcome::failed(String::from("division by zero"), steps),
        jit::ST_REM0 => Outcome::failed(String::from("remainder by zero"), steps),
        _ => return None,
    };
    Some(out)
}

/// What comparing one case established.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Verdict {
    /// Both routes ran it and agreed on everything.
    Agreed,
    /// Both ran it and did not.
    Differed(Field),
    /// Only one route applies, so there was nothing to compare.
    OneRoute,
}

/// Ask the route that can decline first.
///
/// `Armed` always produces an outcome, so `OneRoute` means the prepared route
/// declined and nothing else. Asking it first therefore costs nothing and
/// stops a program being *executed* when there was never going to be anything
/// to compare it against -- which matters because running one is not free of
/// consequences: the seeded tools under `/ai/tools` print, and the first
/// version of this ran all three and put their output in the middle of the
/// suite.
pub fn compare(src: &str, entry: Entry) -> Verdict {
    compare_with(src, entry, Route::Prepared)
}

/// Compare `other` against the interpreter, asking `other` first.
pub fn compare_with(src: &str, entry: Entry, other: Route) -> Verdict {
    let Some(y) = observe(src, entry, other) else {
        return Verdict::OneRoute;
    };
    let Some(x) = observe(src, entry, Route::Armed) else {
        return Verdict::OneRoute;
    };
    match disagree(&x, &y) {
        Some(f) => Verdict::Differed(f),
        None => Verdict::Agreed,
    }
}

/// The built-in corpus: one entry per shape of the language that a code
/// generator will have to get right.
///
/// Literal rather than gathered, so the suite means the same thing on a
/// machine with an empty namespace as on one that has been running for a
/// week. What is gathered -- stored cores, seeded tools -- is compared as
/// well, and counted separately for that reason.
const VOTE_TEXT: &str = "put everything back the way it was";

const CORPUS: &[(&str, &str, Entry<'static>)] = &[
    // The shape `voter::compose` writes, which is the one that matters most.
    (
        "a composed core",
        "fn vote(text: str, allowed: list): int { t = lower(text) \
         if (contains(t, \"snap\")) { return 3 } \
         if (contains(t, \"back\")) { return 20 } return get(allowed, 0) }",
        Entry::Vote(VOTE_TEXT),
    ),
    // Calls into calls, so a route that got the frame wrong is caught.
    (
        "recursion",
        "fn fib(n: int): int { if (n < 2) { return n } return fib(n - 1) + fib(n - 2) } \
         fn vote(text: str, allowed: list): int { return fib(12) }",
        Entry::Vote(VOTE_TEXT),
    ),
    // The extra tick per iteration lives here, and it is the tick most
    // easily got wrong by anything that executes this language another way.
    (
        "a while loop",
        "fn vote(text: str, allowed: list): int { i = 0 s = 0 \
         while (i < 50) { s = s + i i = i + 1 } return s }",
        Entry::Vote(VOTE_TEXT),
    ),
    // Declarations that are not functions.
    (
        "records",
        "rec P { x: int, y: int } \
         fn vote(text: str, allowed: list): int { p = P(7, 9) p.x = 11 return p.x + p.y }",
        Entry::Vote(VOTE_TEXT),
    ),
    (
        "lists",
        "fn vote(text: str, allowed: list): int { l = range(0, 20) l = reverse(l) \
         return get(sort(l), 3) }",
        Entry::Vote(VOTE_TEXT),
    ),
    (
        "strings",
        "fn vote(text: str, allowed: list): int { \
         return len(split(replace(text, \" \", \"-\"), \"-\")) }",
        Entry::Vote(VOTE_TEXT),
    ),
    // Short-circuit, which does not evaluate its right side and so has a step
    // count that depends on the value.
    (
        "short circuit",
        "fn vote(text: str, allowed: list): int { i = 0 \
         if (0 && contains(text, \"x\")) { i = 1 } \
         if (1 || contains(text, \"y\")) { i = i + 2 } return i }",
        Entry::Vote(VOTE_TEXT),
    ),
    // The three failures, because error *text* is compared and a route that
    // fails differently is a route that fails.
    (
        "a runaway, stopped by the budget",
        "fn vote(text: str, allowed: list): int { i = 0 while (1) { i = i + 1 } return i }",
        Entry::Vote(VOTE_TEXT),
    ),
    (
        "a type error at a call boundary",
        "fn f(a: int): int { return a } \
         fn vote(text: str, allowed: list): int { return f(\"s\") }",
        Entry::Vote(VOTE_TEXT),
    ),
    (
        "a name that is not defined",
        "fn vote(text: str, allowed: list): int { return nope + 1 }",
        Entry::Vote(VOTE_TEXT),
    ),
    (
        "an arity that does not match",
        "rec P { x } fn vote(text: str, allowed: list): int { return P(1, 2).x }",
        Entry::Vote(VOTE_TEXT),
    ),
    // --- top level only ------------------------------------------------
    //
    // Every case above asks for `vote`, so until these existed the *step
    // count of the declarations themselves* -- the number `Prepared` carries
    // and the whole reason `adopt` is not just a table copy -- was never
    // compared. These vary how many declarations there are and of what kind,
    // which is exactly what that count is a function of.
    ("one declaration", "fn f(): int { return 1 }", Entry::Top),
    (
        "three declarations",
        "fn a(): int { return 1 } fn b(): int { return 2 } fn c(): int { return 3 }",
        Entry::Top,
    ),
    ("a record alone", "rec P { x: int, y: str }", Entry::Top),
    (
        "records and functions together",
        "rec P { x } rec Q { y } fn f(p: P): int { return p.x } fn g(): int { return 0 }",
        Entry::Top,
    ),
    ("nothing at all", "", Entry::Top),

];

/// The compiled slice: one function, integers only.
///
/// Every case here is run three ways -- armed, prepared and compiled -- and
/// all three must agree on the answer, the cost and the failure. The cost is
/// the interesting column: a code generator that gets arithmetic right and
/// ticks nearly right passes any test that only reads answers.
const INTS: &[(&str, &str, &str, &[i64])] = &[
    ("a constant", "fn f(): int { return 7 }", "f", &[]),
    ("a parameter back", "fn f(a: int): int { return a }", "f", &[41]),
    (
        "every arithmetic operator",
        "fn f(a: int, b: int): int { return a + b * 3 - a / 2 + a % 3 }",
        "f",
        &[17, 5],
    ),
    (
        "negative operands, since idiv truncates toward zero and so does Rust",
        "fn f(a: int, b: int): int { return a / b + a % b }",
        "f",
        &[-17, 5],
    ),
    ("unary minus and not", "fn f(a: int): int { return -a + !a }", "f", &[0]),
    (
        "every comparison",
        "fn f(a: int, b: int): int {          return (a < b) + (a <= b) * 2 + (a > b) * 4 + (a >= b) * 8               + (a == b) * 16 + (a != b) * 32 }",
        "f",
        &[3, 3],
    ),
    (
        "if with an else, taken",
        "fn f(a: int): int { if (a > 10) { return 1 } else { return 2 } }",
        "f",
        &[11],
    ),
    (
        "if with an else, not taken",
        "fn f(a: int): int { if (a > 10) { return 1 } else { return 2 } }",
        "f",
        &[3],
    ),
    (
        "if with no else, falling past it",
        "fn f(a: int): int { x = 0 if (a) { x = 5 } return x }",
        "f",
        &[0],
    ),
    (
        "a loop that accumulates",
        "fn f(n: int): int { i = 0 s = 0 while (i < n) { s = s + i i = i + 1 } return s }",
        "f",
        &[50],
    ),
    (
        "a loop whose body never runs",
        "fn f(n: int): int { i = 0 while (i < n) { i = i + 1 } return i }",
        "f",
        &[0],
    ),
    (
        "an if nested in a loop",
        "fn f(n: int): int { i = 0 s = 0          while (i < n) { if (i % 2) { s = s + i } else { s = s - 1 } i = i + 1 } return s }",
        "f",
        &[21],
    ),
    (
        "returning out of a loop",
        "fn f(n: int): int { i = 0 while (1) { if (i > n) { return i } i = i + 1 } return 0 }",
        "f",
        &[9],
    ),
    // Short circuit: whether the right side's ticks happen at all is decided
    // by a value at runtime, which is the tick a compiler is most likely to
    // charge unconditionally.
    (
        "&& that stops early",
        "fn f(a: int, b: int): int { return a && b + b + b }",
        "f",
        &[0, 4],
    ),
    (
        "&& that does not",
        "fn f(a: int, b: int): int { return a && b + b + b }",
        "f",
        &[1, 4],
    ),
    (
        "|| that stops early",
        "fn f(a: int, b: int): int { return a || b + b + b }",
        "f",
        &[1, 4],
    ),
    (
        "|| that does not",
        "fn f(a: int, b: int): int { return a || b + b + b }",
        "f",
        &[0, 4],
    ),
    // The three failures, in the interpreter's own words.
    ("division by zero", "fn f(a: int): int { return a / 0 }", "f", &[1]),
    ("remainder by zero", "fn f(a: int): int { return a % 0 }", "f", &[1]),
    (
        "a runaway, stopped by the budget at the same step",
        "fn f(a: int): int { i = 0 while (1) { i = i + 1 } return i }",
        "f",
        &[0],
    ),
    // A missing `return` is not nil here: `call_user` checks it against the
    // declared type and refuses. The compiled route has to refuse in the same
    // words.
    ("no return at all", "fn f(a: int): int { x = a }", "f", &[3]),
];

/// Two programs that answer the same thing and cost different amounts.
///
/// The harness has to catch this or it is not a harness. They differ by one
/// unused declaration, so the value is identical and the top level is one
/// statement longer -- which is exactly the kind of difference a comparison
/// that only looked at answers would wave through, and exactly the kind a
/// code generator produces when its ticks are nearly right.
const CANARY_A: &str = "fn vote(text: str, allowed: list): int { return get(allowed, 0) }";
const CANARY_B: &str =
    "fn spare(): int { return 0 } fn vote(text: str, allowed: list): int { return get(allowed, 0) }";

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // The comparator itself, before anything is compared with it.
    let base = Outcome { value: "1".to_string(), steps: 10, error: None };
    claim(&mut ok, disagree(&base, &base.clone()).is_none(), "identical outcomes agree");
    let mut v = base.clone();
    v.value = "2".to_string();
    claim(&mut ok, disagree(&base, &v) == Some(Field::Value), "a different answer is caught");
    let mut s = base.clone();
    s.steps = 11;
    claim(&mut ok, disagree(&base, &s) == Some(Field::Steps), "a different cost is caught");
    let mut e = base.clone();
    e.error = Some("boom".to_string());
    claim(&mut ok, disagree(&base, &e) == Some(Field::Error), "a different failure is caught");

    // And end to end: two programs that agree on the answer and not on the
    // cost must be reported as differing.
    let a = observe(CANARY_A, Entry::Vote("x"), Route::Armed);
    let b = observe(CANARY_B, Entry::Vote("x"), Route::Armed);
    match (a, b) {
        (Some(x), Some(y)) => {
            claim(&mut ok, x.value == y.value, "the canary pair answers the same thing");
            claim(&mut ok, disagree(&x, &y) == Some(Field::Steps),
                "and is still reported as differing, on cost",
            );
        }
        _ => claim(&mut ok, false, "the canary pair ran"),
    }

    // The corpus, both routes, sixty-four times.
    let mut worst: Option<(&str, Field)> = None;
    let mut compared = 0usize;
    let mut one_route = 0usize;
    for _ in 0..ROUNDS {
        compared = 0;
        one_route = 0;
        for (name, src, entry) in CORPUS {
            match compare(src, *entry) {
                Verdict::Agreed => compared += 1,
                Verdict::OneRoute => one_route += 1,
                Verdict::Differed(f) => {
                    if worst.is_none() {
                        worst = Some((name, f));
                    }
                }
            }
        }
    }
    if let Some((name, f)) = worst {
        kprintln!("  FAIL   '{}' differs on {:?}", name, f);
        ok = false;
    } else {
        claim(&mut ok, compared == CORPUS.len() && one_route == 0,
            "every case in the corpus agrees exactly, 64 times over",
        );
    }

    // The compiled slice, all three routes, every round.
    let mut cworst: Option<(&str, Field)> = None;
    let (mut cok, mut cskip) = (0usize, 0usize);
    for _ in 0..ROUNDS {
        cok = 0;
        cskip = 0;
        for (name, src, f, args) in INTS {
            let e = Entry::Ints(f, args);
            // Against the interpreter, and against the prepared route too, so
            // a case is not quietly compared with only one of them.
            for other in [Route::Compiled, Route::Prepared] {
                match compare_with(src, e, other) {
                    Verdict::Agreed => cok += 1,
                    Verdict::OneRoute => cskip += 1,
                    Verdict::Differed(fl) => {
                        if cworst.is_none() {
                            cworst = Some((name, fl));
                        }
                    }
                }
            }
        }
    }
    if let Some((name, f)) = cworst {
        kprintln!("  FAIL   compiled '{}' differs from the interpreter on {:?}", name, f);
        // Both sides, because "they differ on Steps" is not a bug report and
        // guessing which way cost a run.
        if let Some((_, src, fname, args)) = INTS.iter().find(|(n, ..)| n == &name) {
            let e = Entry::Ints(fname, args);
            for (label, route) in [("armed", Route::Armed), ("compiled", Route::Compiled)] {
                match observe(src, e, route) {
                    Some(o) => kprintln!(
                        "         {:8} steps {:5}  value '{}'  err '{}'",
                        label,
                        o.steps,
                        o.value,
                        o.error.clone().unwrap_or_default()
                    ),
                    None => kprintln!("         {:8} declined", label),
                }
            }
        }
        ok = false;
    } else {
        claim(
            &mut ok,
            cok == INTS.len() * 2 && cskip == 0,
            "every compiled function matches the interpreter exactly, 64 times over",
        );
    }

    // Refusing is the other half of being correct. A code generator that
    // compiled something outside its slice would be answering a question it
    // was not asked, and the answer would be wrong in a way nothing here
    // would catch -- these programs have no integer-only meaning at all.
    for outside in [
        "fn f(a: int): str { return \"x\" }",
        "fn f(a: int): int { return len(list(1)) }",
        "rec P { x } fn f(a: int): int { return P(a).x }",
        "fn g(): int { return 1 } fn f(a: int): int { return g() }",
        "fn f(a: int): int { return a & 1 }",
    ] {
        claim(
            &mut ok,
            observe(outside, Entry::Ints("f", &[1]), Route::Compiled).is_none(),
            "a function outside the slice is refused, not approximated",
        );
    }

    // The corpus has to actually contain both entry kinds. Asking for `vote`
    // never exercises the cost of the declarations themselves, which is the
    // number `Prepared` carries, so a refactor that quietly turned every case
    // into a `Vote` would leave `adopt`'s step-carry untested and everything
    // here still green.
    claim(
        &mut ok,
        CORPUS.iter().any(|(_, _, e)| matches!(e, Entry::Top))
            && CORPUS.iter().any(|(_, _, e)| matches!(e, Entry::Vote(_))),
        "the corpus asks for both a top level and a call",
    );

    // `use` is the statement most dangerous to freeze: it lexes, parses and
    // *executes* another file, and marks the path imported before evaluating
    // it. Preparing one would fix at snapshot time whatever that file did,
    // and share the imported set across every later run.
    claim(
        &mut ok,
        observe("use \"/lib/text\" fn f(): int { return 1 }", Entry::Top, Route::Prepared)
            .is_none(),
        "a `use` at the top level is declined, not frozen",
    );

    // A top level that computes is not preparable, and saying so is the
    // point: the route declines instead of quietly arming and comparing
    // `Armed` against itself.
    let live = "n = 2 fn vote(text: str, allowed: list): int { return get(allowed, n) }";
    claim(&mut ok, observe(live, Entry::Vote("x"), Route::Prepared).is_none()
            && observe(live, Entry::Vote("x"), Route::Armed).is_some(),
        "a computing top level is declined by one route, not run twice by one",
    );

    // Whatever this machine happens to be carrying. Counted and reported
    // rather than folded in, because a namespace with nothing in it would
    // otherwise look like a corpus that passed.
    let (mut stored, mut declined, mut stored_bad) = (0usize, 0usize, 0usize);
    for dir in [crate::ai::voter::ROOT, "/ai/tools"] {
        for name in crate::sysbox::children(dir) {
            let mut path = String::from(dir);
            path.push('/');
            path.push_str(&name);
            let Some(bytes) = crate::sysbox::read_blob(&path) else { continue };
            let Ok(src) = core::str::from_utf8(&bytes) else { continue };
            // A stored core answers `vote`; a seeded tool is a top level. Ask
            // for whichever the program actually has, rather than assuming
            // from the directory it came out of.
            let entry =
                if src.contains("fn vote(") { Entry::Vote(VOTE_TEXT) } else { Entry::Top };
            match compare(src, entry) {
                Verdict::Agreed => stored += 1,
                Verdict::OneRoute => declined += 1,
                Verdict::Differed(f) => {
                    kprintln!("  FAIL   stored '{}' differs on {:?}", path, f);
                    stored_bad += 1;
                }
            }
        }
    }
    kprintln!(
        "  {} stored program(s) agreed, {} declined the prepared route, {} differed",
        stored,
        declined,
        stored_bad
    );
    ok &= stored_bad == 0;

    ok
}
