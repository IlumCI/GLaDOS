//! The language.
//!
//! Source -> tokens -> AST -> evaluation. In TempleOS the shell *was* the
//! compiler: what you typed at the prompt was compiled to machine code and
//! executed, with no separation between "using the system" and "programming
//! it". This is the first half of that. The second half replaces `eval` with a
//! single-pass code generator emitting x86-64 into the heap; the front end
//! here does not change when that happens.

pub mod eval;
pub mod lex;
pub mod parse;

pub use eval::{Interp, Value};

use alloc::string::String;

/// Lex, parse and evaluate one line, returning the value of its last expression.
pub fn eval_line(interp: &mut Interp, src: &str) -> Result<Value, String> {
    let toks = lex::lex(src)?;
    let ast = parse::parse(toks)?;
    interp.run(&ast)
}

/// Programs run end to end, and compared against what they should produce.
///
/// The point of the language is that the machine will write in it, so what
/// matters is not that each piece works but that a whole small program does:
/// define a function, hold a list, loop over it, return early. Each case here
/// is a program, and each is one an application would actually contain.
///
/// Silent, returning only a verdict, because the registry that calls it prints
/// a line per check already.
pub fn selftest() -> bool {
    fn run(src: &str) -> Option<eval::Value> {
        let mut it = eval::Interp::new();
        eval_line(&mut it, src).ok()
    }
    fn int(src: &str, want: i64) -> bool {
        matches!(run(src), Some(eval::Value::Int(v)) if v == want)
    }
    fn text(src: &str, want: &str) -> bool {
        match run(src) {
            Some(v) => v.render() == want,
            None => false,
        }
    }

    // Functions: definition, call, arguments, and a value coming back out.
    if !int("fn add(a, b) { return a + b } add(2, 3)", 5) {
        return false;
    }
    // Recursion, which is the shape of anything that walks a structure.
    if !int("fn f(n) { if (n < 2) { return n } return f(n - 1) + f(n - 2) } f(10)", 55) {
        return false;
    }
    // A return inside a loop leaves the function, not just the loop. This is
    // the one that a sentinel-based return gets wrong if a block forgets to
    // check it.
    if !int("fn first_big(n) { i = 0 while (i < 100) { if (i > n) { return i } i = i + 1 } return -1 } first_big(7)", 8) {
        return false;
    }
    // Falling off the end yields nothing rather than the last expression,
    // which is what a procedure called for its effect should say.
    if !text("fn noisy() { 42 } noisy()", "") {
        return false;
    }

    // Lists: build, measure, index, extend, replace.
    if !text("list(1, 2, 3)", "[1, 2, 3]") {
        return false;
    }
    if !int("len(list(1, 2, 3))", 3) {
        return false;
    }
    if !int("get(list(4, 5, 6), 1)", 5) {
        return false;
    }
    if !text("push(list(1), 2)", "[1, 2]") {
        return false;
    }
    if !text("set(list(1, 2), 0, 9)", "[9, 2]") {
        return false;
    }
    // Past the end is nothing, not a fault: a program walking a list it did
    // not build should be able to ask.
    if !text("get(list(1), 5)", "") {
        return false;
    }
    // Values and not references. If `push` mutated, `xs` would have grown too,
    // and two names would disagree about one list -- the whole reason this
    // language does not have references.
    if !text("xs = list(1) ys = push(xs, 2) xs", "[1]") {
        return false;
    }

    // Scope: a parameter shadows a global without destroying it, and an
    // assignment to a name that already exists outside updates it there.
    if !int("n = 1 fn shadow(n) { n = 99 return n } shadow(5) n", 1) {
        return false;
    }
    if !int("g = 1 fn bump() { g = g + 1 } bump() bump() g", 3) {
        return false;
    }

    // A whole program of the shape an application has: state in a list, a
    // function over it, a loop driving them.
    if !int(
        "fn total(xs) { s = 0 i = 0 while (i < len(xs)) { s = s + get(xs, i) i = i + 1 } return s } \
         items = list() items = push(items, 10) items = push(items, 20) total(items)",
        30,
    ) {
        return false;
    }

    // Runaway recursion is refused rather than being allowed to eat the
    // kernel stack, which has no guard page and would triple fault.
    if run("fn boom(n) { return boom(n + 1) } boom(0)").is_some() {
        return false;
    }
    // Malformed input is an error, not a panic. This is also the gate that
    // makes generated programs safe to run at all.
    run("fn (").is_none() && run("return return").is_none()
}
