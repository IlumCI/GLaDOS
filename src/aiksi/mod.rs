//! The language.
//!
//! Source -> tokens -> AST -> evaluation. In TempleOS the shell *was* the
//! compiler: what you typed at the prompt was compiled to machine code and
//! executed, with no separation between "using the system" and "programming
//! it". This is the first half of that. The second half replaces `eval` with a
//! single-pass code generator emitting x86-64 into the heap; the front end
//! here does not change when that happens.

pub mod eval;
pub mod kernel;
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

    // --- the gate ---------------------------------------------------------
    //
    // Checked by name and never by calling. Half this table pokes memory,
    // writes I/O ports or paints over the screen, and a test suite that
    // exercises every row would be a test suite that scribbles on the machine
    // to prove it can.
    {
        use eval::{Caps, Touch, BUILTINS};

        // No two rows may claim the same name. A duplicate means the first
        // silently decides the class, and if they disagree the stricter one is
        // the one that never runs.
        for (i, (n, ..)) in BUILTINS.iter().enumerate() {
            if BUILTINS.iter().skip(i + 1).any(|(m, ..)| m == n) {
                return false;
            }
        }

        // A name absent from the table is refused before dispatch. This is the
        // property the whole inversion rests on: reaching the match at all
        // requires a row, so an arm added without one is dead code rather than
        // an ungated builtin.
        let mut op = eval::Interp::new();
        if eval_line(&mut op, "nosuchbuiltin(1)").is_ok() {
            return false;
        }

        // Everything a stored program may not do is refused for a stored
        // program, and every row is covered rather than a chosen few.
        let mut jailed = eval::Interp::sandboxed("/app/t");
        for (name, touch, lo, _) in BUILTINS {
            if *touch == Touch::Pure || *touch == Touch::Read || *touch == Touch::Write {
                continue;
            }
            let args = alloc::vec!["0"; *lo].join(", ");
            let src = alloc::format!("{}({})", name, args);
            match eval_line(&mut jailed, &src) {
                Err(e) if e.contains("may not") => {}
                // Anything else means it ran, or failed for the wrong reason.
                _ => return false,
            }
        }

        // ...and the ones it may do are not refused by the gate. `read` of a
        // path that does not exist fails on the path, which is the arm
        // answering rather than the gate.
        if eval::available(Caps::Sandbox).len() >= BUILTINS.len() {
            return false;
        }
        if !eval::available(Caps::Operator).contains(&"poke8") {
            return false;
        }
        if eval::available(Caps::Sandbox).contains(&"poke8") {
            return false;
        }

        // Every row that can be exercised without damaging anything has an
        // implementation behind it. This is the other half of the inversion:
        // the gate makes an arm without a row unreachable, and this makes a row
        // without an arm a boot failure rather than something a program
        // discovers at three in the morning.
        //
        // Pure and Read only. The rest write files, poke memory, drive I/O
        // ports or paint over the screen, and a suite that called them to prove
        // they exist would be scribbling on the machine to do it. Those rows
        // are covered by the sandbox check above, which reaches every one.
        //
        // Numbers as arguments, because a builtin wanting a string accepts one
        // (`render` never fails) while a builtin wanting a number rejects a
        // string -- so `0` is the argument that gets furthest into the arm.
        for (name, touch, lo, _) in BUILTINS {
            if *touch != Touch::Pure && *touch != Touch::Read {
                continue;
            }
            let args = alloc::vec!["0"; *lo].join(", ");
            if let Err(e) = eval_line(&mut op, &alloc::format!("{}({})", name, args)) {
                if e.contains("no implementation") {
                    return false;
                }
            }
        }

        // Arity is enforced from the table, before the arm sees the arguments.
        if eval_line(&mut op, "hex()").is_ok() || eval_line(&mut op, "hex(1, 2)").is_ok() {
            return false;
        }
        // A variadic row accepts none and many.
        if eval_line(&mut op, "list()").is_err() || eval_line(&mut op, "list(1,2,3,4,5)").is_err() {
            return false;
        }
    }

    // The text builtins, on the operations a program actually strings
    // together: split a line, take a field, test it, put it back.
    if !text("join(split(\"a:b:c\", \":\"), \"-\")", "a-b-c") {
        return false;
    }
    if !int("find(\"hello\", \"ll\")", 2) {
        return false;
    }
    if !int("find(\"hello\", \"z\")", -1) {
        return false;
    }
    if !text("substr(\"hello\", 1, 3)", "ell") {
        return false;
    }
    if !text("upper(trim(\"  ok  \"))", "OK") {
        return false;
    }
    if !text("replace(\"a.b.c\", \".\", \"/\")", "a/b/c") {
        return false;
    }
    // An empty separator splits into characters rather than looping forever.
    if !int("len(split(\"abc\", \"\"))", 3) {
        return false;
    }
    // Hex round-trips, which is what a program reading a stored hash needs.
    if !text("hexdec(hexenc(\"hi\"))", "hi") {
        return false;
    }
    if !text("hexenc(\"A\")", "41") {
        return false;
    }
    // Lists: sort numerically rather than by spelling, which is the one a
    // string sort gets wrong and nobody notices until there are ten rows.
    if !text("join(sort(list(10, 9, 100)), \",\")", "9,10,100") {
        return false;
    }
    if !int("index(list(\"a\", \"b\"), \"b\")", 1) {
        return false;
    }
    if !int("len(range(0, 5))", 5) {
        return false;
    }
    // Bounded, so a generated program cannot ask for a billion-element list
    // in a repaint path and take the heap with it.
    if !int("len(range(0, 999999999))", 65_536) {
        return false;
    }
    if !int("len(repeat(\"x\", 999999999))", 65_536) {
        return false;
    }
    if !int("sqrt(144)", 12) {
        return false;
    }

    // --- records ----------------------------------------------------------
    //
    // A declaration, a constructor, a field read, a field write, and the
    // rendering all of those are read through.
    if !int("rec Host { name, port } h = Host(\"a\", 80) h.port", 80) {
        return false;
    }
    if !text("rec Host { name, port } h = Host(\"a\", 80) h.name", "a") {
        return false;
    }
    if run("rec Host { name, port } h = Host(\"a\", 80) h.nope = 1").is_some() {
        return false;
    }
    if !int("rec Host { name, port } h = Host(\"a\", 80) h.port = 443 h.port", 443) {
        return false;
    }
    if !text("rec P { x, y } P(1, 2)", "P{x: 1, y: 2}") {
        return false;
    }
    // Values, not references: a copy taken before the write keeps the old
    // field. This is the property that means nothing has to be said about
    // aliasing, so it is worth a test rather than a comment.
    if !int("rec P { x } a = P(1) b = a a.x = 9 b.x", 1) {
        return false;
    }
    // Records nest, and reading through two levels works.
    if !int("rec In { v } rec Out { i } o = Out(In(7)) o.i.v", 7) {
        return false;
    }
    // Constructor arity comes from the declaration.
    if run("rec P { x, y } P(1)").is_some() || run("rec P { x, y } P(1, 2, 3)").is_some() {
        return false;
    }
    // A field that was never declared is not readable.
    if run("rec P { x } P(1).y").is_some() {
        return false;
    }
    // Reading a field off something that is not a record is a mistake in the
    // program, not a Nil to carry forward.
    if run("(1).x").is_some() || run("\"s\".x").is_some() {
        return false;
    }
    // A record may not take a builtin's name, because it would install a
    // constructor and leave `len(x)` ambiguous to a reader.
    if run("rec len { a }").is_some() {
        return false;
    }
    // Two fields with one name is a declaration nobody can index.
    if run("rec P { x, x }").is_some() {
        return false;
    }

    // --- types --------------------------------------------------------------
    //
    // Optional. A function that says nothing behaves as it always did, which
    // is what every application already written depends on.
    if !int("fn f(a) { return a } f(1)", 1) {
        return false;
    }
    if !int("fn f(a: int): int { return a + 1 } f(1)", 2) {
        return false;
    }
    // The mistake is caught at the call, where the caller is, and not
    // wherever the value happened to be used.
    if run("fn f(a: int) { return a } f(\"x\")").is_some() {
        return false;
    }
    // A return type is checked too, which is the half that catches a function
    // falling off its end when it was supposed to answer something.
    if run("fn f(): int { return \"x\" } f()").is_some() {
        return false;
    }
    if run("fn f(): int { } f()").is_some() {
        return false;
    }
    // `any` is explicit as well as default.
    if !text("fn f(a: any): any { return a } f(\"x\")", "x") {
        return false;
    }
    // Record types are usable as annotations, and are checked by name.
    if !int("rec P { x } fn f(p: P): int { return p.x } f(P(4))", 4) {
        return false;
    }
    if run("rec P { x } rec Q { x } fn f(p: P) { return 1 } f(Q(1))").is_some() {
        return false;
    }
    // Field annotations hold at construction and at assignment, or the
    // annotation would be a comment.
    if run("rec P { x: int } P(\"s\")").is_some() {
        return false;
    }
    if run("rec P { x: int } p = P(1) p.x = \"s\"").is_some() {
        return false;
    }
    // A list is a type like any other.
    if !int("fn f(l: list): int { return len(l) } f(list(1, 2))", 2) {
        return false;
    }

    // --- use ----------------------------------------------------------------
    //
    // The file half needs a namespace and is driven under QEMU. What is
    // checkable here is the refusal, which is the part that matters: a
    // sandboxed program's imports are confined so that what it depends on can
    // be found in one place.
    {
        let mut jailed = eval::Interp::sandboxed("/app/t");
        match eval_line(&mut jailed, "use \"/ai/godel/HEAD\"") {
            Err(e) if e.contains("may only use") => {}
            _ => return false,
        }
        // Its own subtree is allowed to get as far as looking for the file.
        match eval_line(&mut jailed, "use \"/app/t/helper\"") {
            Err(e) if e.contains("no such program") => {}
            _ => return false,
        }
        // An operator is not confined, and gets the same honest miss.
        let mut op2 = eval::Interp::new();
        match eval_line(&mut op2, "use \"/nowhere/at/all\"") {
            Err(e) if e.contains("no such program") => {}
            _ => return false,
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

    // A program spread over real lines, which is what a stored one is.
    //
    // Every case above is a single-line string, and that is exactly how the
    // language shipped for months looking fine while `run` could not execute
    // any file containing a function: `sysbox::cmd_run` fed the blob to
    // `eval_line` one line at a time, and `fn total(xs) {` on its own is not a
    // statement. The lexer has always treated a newline as whitespace, so the
    // defect was never here -- but nothing here could have caught it either.
    // It can now.
    if !int(
        "fn total(xs) {\n    \
             s = 0\n    \
             i = 0\n    \
             while (i < len(xs)) {\n        \
                 s = s + get(xs, i)\n        \
                 i = i + 1\n    \
             }\n    \
             return s\n\
         }\n\
         total(list(1, 2, 3, 4))",
        10,
    ) {
        return false;
    }
    // Comments and blank lines between statements, which any authored file has.
    if !int("// a program\n\nfn one() {\n  return 1\n}\n\n// call it\none()", 1) {
        return false;
    }

    // --- capabilities ---------------------------------------------------
    //
    // The operator keeps everything and a stored program does not. Each of
    // these is a way an application that was written by something other than a
    // person could reach past its own subtree, and each is refused rather than
    // merely discouraged: a check that produces a warning is a check that gets
    // ignored by whatever is generating the program.
    fn boxed(src: &str) -> Option<eval::Value> {
        let mut it = eval::Interp::sandboxed("/app/demo");
        eval_line(&mut it, src).ok()
    }
    // Raw memory and the I/O ports. A hallucinated port write on real hardware
    // is not something the machine recovers from.
    if boxed("poke64(4096, 1)").is_some() || boxed("peek32(0)").is_some() {
        return false;
    }
    if boxed("outb(112, 0)").is_some() || boxed("inb(112)").is_some() {
        return false;
    }
    // Drawing, which goes straight at the framebuffer with no window in the
    // way, so a sandboxed program could paint over the whole desktop.
    if boxed("rect(0, 0, 10, 10, 1)").is_some() || boxed("pixel(0, 0, 1)").is_some() {
        return false;
    }
    // ...and the operator still has all of it.
    if run("hex(255)").is_none() {
        return false;
    }
    // Writes outside the program's own subtree, including by the route every
    // jail is defeated through. The check runs on the *resolved* path, so
    // `..` is spent before the comparison rather than after it.
    if boxed("write(\"/ai/godel/HEAD\", \"x\")").is_some() {
        return false;
    }
    if boxed("write(\"/app/demo/../../ai/godel/HEAD\", \"x\")").is_some() {
        return false;
    }
    // A neighbour whose name merely begins the same way is outside the jail:
    // `/app/demo-evil` is not under `/app/demo`, and a prefix test alone would
    // have said it was.
    if boxed("write(\"/app/demo-evil/x\", \"y\")").is_some() {
        return false;
    }
    // Applets that change something. The flag consulted is the one
    // `harness::Trust::ReadOnly` filters the model's grammar with, so "safe to
    // call" has a single definition in this tree rather than two that drift.
    if boxed("applet(\"write /ai/godel/HEAD x\")").is_some() {
        return false;
    }
    if boxed("applet(\"rm /ai/tools/hello.l\")").is_some() {
        return false;
    }
    // `run` is classified mutating whatever the program text says, so a
    // sandboxed program cannot launder its way out through another one.
    if boxed("applet(\"run /ai/tools/hello.l\")").is_some() {
        return false;
    }
    if boxed("applet(\"nosuchapplet\")").is_some() {
        return false;
    }
    // Everything a sandbox is *allowed* to do needs a namespace, and this runs
    // before `sysbox::init`. Those are proved by the seeded application
    // instead, which writes its items under its own subtree through this same
    // gate every time a row is added -- an app that could not write would not
    // work at all, which is a louder failure than an assertion here.

    // A lowered budget stops a loop the full one would let run.
    //
    // The desktop calls an application's row function on every repaint, so a
    // generated loop that runs for a second is a window manager that feels
    // broken for reasons nobody can attribute. Twenty million steps is the
    // operator's budget, where a long loop is visible and can be stopped.
    {
        let mut it = eval::Interp::new().with_step_budget(1000);
        if eval_line(&mut it, "i = 0 while (i < 100000) { i = i + 1 } i").is_ok() {
            return false;
        }
        // ...and something short still finishes inside it.
        let mut it = eval::Interp::new().with_step_budget(1000);
        if !matches!(eval_line(&mut it, "i = 0 while (i < 10) { i = i + 1 } i"),
                     Ok(eval::Value::Int(10)))
        {
            return false;
        }
        // The budget cannot be raised past the default by asking.
        let mut it = eval::Interp::new().with_step_budget(u64::MAX);
        if eval_line(&mut it, "i = 0 while (i < 30000000) { i = i + 1 } i").is_ok() {
            return false;
        }
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
