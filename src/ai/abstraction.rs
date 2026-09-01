//! Finding the abstractions hiding in the skills the machine has written.
//!
//! `agent learn` compiles a successful episode into a program under
//! `/ai/tools`, `skill judge` puts it through four judges, and adoption copies
//! it in. That is skill accumulation and it works. What has never happened is
//! anybody looking *across* the accumulated skills.
//!
//! So the library never gets smaller. Twenty skills that each walk a directory
//! contain twenty copies of the walk, and the twenty-first is written from
//! scratch because there is nothing to call. Accumulation without compression
//! is hoarding.
//!
//! This is DreamCoder's sleep-abstraction phase, which is the half of that
//! design this tree does not have: enumerate the subtrees of every stored
//! program, canonicalise them so that structurally identical code with
//! different names collides, and rank what repeats by how much naming it would
//! save. The wake phase already exists here under another name.
//!
//! **The objective is node count, and it is arithmetic rather than taste.**
//! An abstraction of `size` nodes occurring `count` times costs one definition
//! and turns each occurrence into a call:
//!
//! ```text
//!   before   count * size
//!   after    count * (1 + arity) + size
//!   saved    count * (size - 1 - arity) - size
//! ```
//!
//! That formula is why arity is not a separate tunable. A subtree whose leaves
//! are all different has an arity equal to its leaf count, the call site is as
//! big as the thing it replaces, and `saved` goes negative on its own. Nothing
//! has to decide that six parameters is too many; the objective already knows.
//!
//! **Two of the three gaps this file used to list are closed.** It runs the
//! judges (`bench`, below) and it writes `/lib` on adoption, as
//! `ProposalKind::Abstract` -- so an abstraction is now something the nightly
//! loop can find, judge and keep, with a ledger line and a rollback like every
//! other axis.
//!
//! **It still does not rewrite the callers**, and that one is deliberate
//! rather than pending. Rewriting a program the operator trusted by hash
//! changes its hash, which revokes the trust -- correct behaviour, and it
//! needs a story before it happens rather than after. So adoption is additive:
//! `/lib` gains a function that new programs can `use`, and every existing
//! skill goes on saying exactly what it said. The compression is therefore
//! *available* rather than *realised*, and the objective below should be read
//! as what naming this would save if the callers were rewritten, not as bytes
//! already reclaimed.

// --- judging, and adopting -----------------------------------------------

use crate::store::sha256;

/// Candidates under judgement, by content address. Same shape as
/// `skill::store`, and for the same reason: a proposal has to be nameable
/// before it can be marked tried, and its own hash is the only name that
/// cannot drift from its contents.
const STORE: &str = "/ai/abstractions";

/// Where an adopted abstraction lands.
///
/// `/lib` because that is the one directory `use` may reach from a sandboxed
/// program -- `aiksi`'s jail lets a stored program import its own files or
/// `/lib` and nothing else. An abstraction adopted anywhere else would be one
/// no sandboxed skill could call, which is every skill the machine writes.
const LIB: &str = "/lib";

/// How many occurrences make a repetition worth naming.
///
/// Three, and across at least two programs. The module comment argues the
/// second half: a structure repeated inside one program is a loop somebody did
/// not write, and naming it is refactoring that one program rather than
/// building a library. Both floors are here rather than in the objective
/// because `saved` already goes positive at two occurrences of a large enough
/// subtree, and that is exactly the case where the "abstraction" is one
/// program's own duplication.
const MIN_COUNT: usize = 3;
const MIN_PROGRAMS: usize = 2;

/// The largest definition worth keeping.
///
/// A ceiling on `size`, not on `saved`. A 400-node abstraction that occurs
/// three times saves a great deal and is not a library function, it is a
/// program somebody pasted twice -- and `/lib` is read by whoever debugs the
/// skills that call it.
const MAX_SIZE: usize = 40;

/// Steps one judged abstraction may take.
///
/// Small, because an abstraction's body is a single expression by
/// construction: `canon` builds it out of one subtree, so anything that takes
/// real time in here is a loop that got in, and a loop is not what this is
/// supposed to be finding.
const JUDGE_BUDGET: u64 = 100_000;

const NEWLINE: char = '\n';

fn path_of(h: &[u8; 32]) -> String {
    let mut p = String::from(STORE);
    p.push('/');
    for b in &h[..8] {
        p.push_str(&format!("{:02x}", b));
    }
    p
}

/// The short name an adopted abstraction gets.
///
/// Derived from the hash rather than counted, because a counter is state that
/// two machines would disagree about, and this tree names things by content
/// everywhere else it can.
pub fn name_for(h: &[u8; 32]) -> String {
    let mut n = String::from("abs");
    for b in &h[..4] {
        n.push_str(&format!("{:02x}", b));
    }
    n
}

pub fn adopted_path(h: &[u8; 32]) -> String {
    format!("{}/{}.ai&xi", LIB, name_for(h))
}

/// Store a skeleton and return its address.
///
/// The *skeleton* is stored, not the rendered function, and that is the
/// decision worth stating: the skeleton is the identity of the abstraction,
/// while the rendered function carries a generated name and would change
/// address every time the naming scheme did.
pub fn store(skeleton: &str) -> [u8; 32] {
    let h = sha256::hash(skeleton.as_bytes());
    let p = path_of(&h);
    if crate::sysbox::read_blob(&p).is_none() {
        crate::sysbox::write_text(&p, skeleton);
    }
    h
}

pub fn source(h: &[u8; 32]) -> Option<String> {
    crate::sysbox::read_blob(&path_of(h)).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// Re-derive a candidate from the library as it stands now.
///
/// Deliberately a fresh scan rather than metrics stored alongside the
/// skeleton. The question a judge is asking is "is this worth naming *now*",
/// and the library moves: a candidate that was worth naming when it was
/// proposed may occur once by the time it is judged, because the skill it came
/// from was replaced. Storing `count` would answer the older question and look
/// identical doing it.
pub fn find(h: &[u8; 32]) -> Option<Candidate> {
    let skel = source(h)?;
    scan().into_iter().find(|c| c.skeleton == skel)
}

/// What the four judges said.
pub struct Verdict {
    pub j1: bool,
    pub j1_why: &'static str,
    pub j2: bool,
    pub j2_why: &'static str,
    pub j3: bool,
    pub j3_why: &'static str,
    pub j4: bool,
    /// Nodes the naming would save, re-derived at judging time.
    pub saved: isize,
    pub count: usize,
    pub programs: usize,
    pub size: usize,
    pub arity: usize,
    /// The function source, when there was one to build.
    pub src: String,
}

/// Judge one abstraction.
///
/// Admission judges, not improvement judges, and the constraint is the same
/// one `skill::bench` records: the thing being judged is a library entry, and
/// "does it make the machine better at its job" cannot be asked of one without
/// a synthesiser that would use it. A judge that could never fail is worse
/// than an absent judge, so what is asked is whether this is a function at
/// all, whether it computes, whether it is genuinely shared, and whether it
/// pays for itself.
pub fn bench(h: &[u8; 32]) -> Verdict {
    let mut v = Verdict {
        j1: false,
        j1_why: "no such candidate",
        j2: false,
        j2_why: "not run",
        j3: false,
        j3_why: "not run",
        j4: false,
        saved: 0,
        count: 0,
        programs: 0,
        size: 0,
        arity: 0,
        src: String::new(),
    };
    let Some(c) = find(h) else {
        v.j1_why = "it no longer occurs in the library";
        return v;
    };
    v.saved = c.saved;
    v.count = c.count;
    v.programs = c.programs.len();
    v.size = c.size;
    v.arity = c.arity;
    v.src = c.proposal_named(&name_for(h));

    // --- J1: is it a function? ------------------------------------------
    match crate::aiksi::lex::lex(&v.src).and_then(crate::aiksi::parse::parse) {
        Err(_) => {
            v.j1_why = "the function it would become does not parse";
            return v;
        }
        Ok(prog) if prog.is_empty() => {
            v.j1_why = "it parses to nothing";
            return v;
        }
        Ok(_) => {
            v.j1 = true;
            v.j1_why = "a function";
        }
    }

    // --- J2: does it compute? --------------------------------------------
    //
    // Declared and then called with zeros. The point is not the value -- an
    // abstraction over `#0 + 1` answers 1 and that means nothing -- it is that
    // the body evaluates at all. `canon` builds skeletons by replacing leaves,
    // and a replacement that lands somewhere a leaf may not go produces text
    // that parses and cannot run.
    //
    // Zeros because they are the one argument every arity accepts and no
    // abstraction can be tuned to. Division by zero is a real error and a real
    // refusal here: an abstraction whose body divides by a parameter fails
    // this judge, which is stricter than necessary and is the safe direction
    // for something about to be written into `/lib`.
    let mut call = v.src.clone();
    call.push(NEWLINE);
    call.push_str(&name_for(h));
    call.push('(');
    for i in 0..c.arity {
        if i > 0 {
            call.push_str(", ");
        }
        call.push('0');
    }
    call.push(')');
    // Sandboxed, and bounded. Judging is not a reason to run unknown text
    // with operator powers -- `skill::cmd_run` learned that the hard way, and
    // this is the same class of text arriving by a different route.
    let mut interp = crate::aiksi::Interp::sandboxed(STORE).with_step_budget(JUDGE_BUDGET);
    match crate::aiksi::lex::lex(&call)
        .and_then(crate::aiksi::parse::parse)
        .and_then(|prog| interp.run(&prog))
    {
        Err(_) => {
            v.j2_why = "the body does not evaluate";
            return v;
        }
        Ok(_) => {
            v.j2 = true;
            v.j2_why = "it computes";
        }
    }

    // --- J3: is it actually shared? --------------------------------------
    v.j3 = c.count >= MIN_COUNT && c.programs.len() >= MIN_PROGRAMS;
    v.j3_why = if v.j3 {
        "repeated across programs"
    } else if c.programs.len() < MIN_PROGRAMS {
        "it repeats inside one program -- that is a loop, not a library function"
    } else {
        "too few occurrences to be worth a name"
    };

    // --- J4: does it pay, and can it be read? ----------------------------
    v.j4 = c.saved > 0 && c.size <= MAX_SIZE;

    v
}

/// Every candidate that has not been judged yet, cheapest first.
pub fn unjudged() -> Vec<Candidate> {
    let mut v = scan();
    // Largest saving first, so a night that only gets one trial spends it on
    // the best candidate rather than the alphabetically first.
    v.sort_by(|a, b| b.saved.cmp(&a.saved));
    v
}



use crate::aiksi::parse::{BinOp, Expr, Stmt, UnOp};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Smallest subtree worth naming.
///
/// Two nodes is an operator and an operand. Naming that produces a function
/// whose body is shorter than its own call, which the objective would reject
/// anyway -- this only saves the work of enumerating them.
const MIN_SIZE: usize = 3;

/// A repeated structure, and what naming it would buy.
#[derive(Clone)]
pub struct Candidate {
    /// The shape, with leaves written as `#0`, `#1` and so on.
    pub skeleton: String,
    /// How many distinct leaves it abstracts over, so the arity of the
    /// function this would become.
    pub arity: usize,
    /// Nodes in one occurrence.
    pub size: usize,
    /// Occurrences across every program scanned.
    pub count: usize,
    /// Which programs it appears in. A structure repeated inside one program
    /// is a loop somebody did not write; repeated across several is a library
    /// function nobody has written yet, and that is the interesting one.
    pub programs: Vec<String>,
    /// Nodes saved by naming it. See the module comment.
    pub saved: isize,
}

impl Candidate {
    /// The function this would become, as source.
    pub fn proposal(&self, n: usize) -> String {
        let mut params = String::new();
        for i in 0..self.arity {
            if i > 0 {
                params.push_str(", ");
            }
            params.push_str(&format!("a{}", i));
        }
        let mut body = self.skeleton.clone();
        // Highest index first: replacing #1 before #10 would corrupt #10.
        for i in (0..self.arity).rev() {
            body = body.replace(&format!("#{}", i), &format!("a{}", i));
        }
        format!("fn abs{}({}) {{ return {} }}", n, params, body)
    }

    /// The same, under a name the caller chose.
    ///
    /// Adoption names by content address rather than by position, because a
    /// positional name is one two machines would disagree about and one that
    /// changes when an earlier candidate is dropped -- renaming a function
    /// every skill in `/lib` might already call.
    pub fn proposal_named(&self, name: &str) -> String {
        let rendered = self.proposal(0);
        rendered.replacen("fn abs0", &format!("fn {}", name), 1)
    }
}

fn bin_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::And => "&",
        BinOp::Or => "|",
        BinOp::Xor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::LogAnd => "&&",
        BinOp::LogOr => "||",
    }
}

fn un_str(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "-",
        UnOp::Not => "!",
        UnOp::BitNot => "~",
    }
}

/// Nodes in an expression.
pub fn size(e: &Expr) -> usize {
    match e {
        Expr::Int(_) | Expr::Str(_) | Expr::Var(_) => 1,
        Expr::Unary(_, a) => 1 + size(a),
        Expr::Bin(_, a, b) => 1 + size(a) + size(b),
        Expr::Call(_, args) => 1 + args.iter().map(size).sum::<usize>(),
        Expr::Assign(_, v) => 1 + size(v),
        Expr::Field(a, _) => 1 + size(a),
        Expr::SetField(a, _, v) => 1 + size(a) + size(v),
    }
}

/// Render an expression with its leaves replaced by numbered holes.
///
/// The same leaf twice becomes the same hole, which is the whole point: it is
/// what makes `x + x` a one-parameter abstraction and `x + y` a two-parameter
/// one. Leaves are compared by their rendered text, so the variable `n` in one
/// program and the variable `n` in another collide deliberately -- an
/// abstraction is about shape, and two programs that both count with `n` have
/// the same shape.
///
/// Call names are *not* holes. `len(x)` and `ls(x)` are different structures,
/// and abstracting over the callee would propose a function that takes a
/// function, which this language has no way to express.
fn canon(e: &Expr, out: &mut String, holes: &mut Vec<(String, bool)>) {
    let leaf = |text: String, lit: bool, out: &mut String, holes: &mut Vec<(String, bool)>| {
        let idx = match holes.iter().position(|(h, _)| *h == text) {
            Some(i) => i,
            None => {
                holes.push((text, lit));
                holes.len() - 1
            }
        };
        out.push_str(&format!("#{}", idx));
    };
    match e {
        Expr::Int(v) => leaf(format!("{}", v), true, out, holes),
        Expr::Str(s) => leaf(format!("{:?}", s), true, out, holes),
        Expr::Var(n) => leaf(n.clone(), false, out, holes),
        Expr::Unary(op, a) => {
            out.push_str(un_str(*op));
            out.push('(');
            canon(a, out, holes);
            out.push(')');
        }
        Expr::Bin(op, a, b) => {
            out.push('(');
            canon(a, out, holes);
            out.push(' ');
            out.push_str(bin_str(*op));
            out.push(' ');
            canon(b, out, holes);
            out.push(')');
        }
        Expr::Call(name, args) => {
            out.push_str(name);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                canon(a, out, holes);
            }
            out.push(')');
        }
        Expr::Assign(n, v) => {
            leaf(n.clone(), false, out, holes);
            out.push_str(" = ");
            canon(v, out, holes);
        }
        Expr::Field(a, f) => {
            canon(a, out, holes);
            out.push('.');
            out.push_str(f);
        }
        Expr::SetField(a, f, v) => {
            canon(a, out, holes);
            out.push('.');
            out.push_str(f);
            out.push_str(" = ");
            canon(v, out, holes);
        }
    }
}

/// Every expression in a statement tree, outermost first.
fn exprs_of(stmts: &[Stmt], out: &mut Vec<Expr>) {
    for s in stmts {
        match s {
            Stmt::Expr(e) => out.push(e.clone()),
            Stmt::Return(Some(e)) => out.push(e.clone()),
            Stmt::Return(None) => {}
            Stmt::If(c, then, els) => {
                out.push(c.clone());
                exprs_of(then, out);
                if let Some(e) = els {
                    exprs_of(e, out);
                }
            }
            Stmt::While(c, body) => {
                out.push(c.clone());
                exprs_of(body, out);
            }
            Stmt::Fn(_, _, _, body) => exprs_of(body, out),
            Stmt::Rec(..) | Stmt::Use(_) => {}
        }
    }
}

/// Every subtree of an expression, including itself.
fn subtrees(e: &Expr, out: &mut Vec<Expr>) {
    out.push(e.clone());
    match e {
        Expr::Int(_) | Expr::Str(_) | Expr::Var(_) => {}
        Expr::Unary(_, a) | Expr::Field(a, _) | Expr::Assign(_, a) => subtrees(a, out),
        Expr::Bin(_, a, b) | Expr::SetField(a, _, b) => {
            subtrees(a, out);
            subtrees(b, out);
        }
        Expr::Call(_, args) => {
            for a in args {
                subtrees(a, out);
            }
        }
    }
}

/// Fold holes that never vary back into the body.
///
/// A hole is only worth a parameter if the occurrences disagree about it. The
/// separator in `path + "/" + name` is `"/"` every single time, so making it
/// an argument costs a node at every call site and buys nothing -- that one
/// hole was the difference between this pass finding a structure worth 1 node
/// and one worth 4.
///
/// Only *literals* fold. A variable that happens to be called `n` in every
/// occurrence is still a variable: substituting it into the body would leave
/// the abstraction referring to a name it does not bind, which is a capture
/// bug rather than a compression.
fn fold(skeleton: &str, binds: &[Vec<(String, bool)>]) -> (String, usize) {
    let n = binds.first().map(|b| b.len()).unwrap_or(0);
    let mut fixed: Vec<Option<String>> = Vec::new();
    for i in 0..n {
        let first = &binds[0][i];
        let same = first.1 && binds.iter().all(|b| b.get(i).map(|h| h.0 == first.0).unwrap_or(false));
        fixed.push(if same { Some(first.0.clone()) } else { None });
    }
    // Surviving holes are renumbered from zero, so the proposal's parameters
    // are a0..aN with no gaps.
    let mut renum: Vec<Option<usize>> = Vec::new();
    let mut next = 0;
    for f in &fixed {
        if f.is_some() {
            renum.push(None);
        } else {
            renum.push(Some(next));
            next += 1;
        }
    }

    let mut out = String::new();
    let bytes: Vec<char> = skeleton.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != '#' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut idx = 0usize;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            idx = idx * 10 + bytes[j] as usize - '0' as usize;
            j += 1;
        }
        match fixed.get(idx) {
            Some(Some(lit)) => out.push_str(lit),
            _ => out.push_str(&format!("#{}", renum.get(idx).copied().flatten().unwrap_or(idx))),
        }
        i = j;
    }
    (out, next)
}

/// Rank the repeated structures in a set of programs.
///
/// Pure, and separated from `scan` for the reason every judge in this tree is:
/// a function that reads the namespace can only be tested on a machine with
/// the right namespace, and then it is not really tested.
pub fn analyse(programs: &[(String, String)]) -> Vec<Candidate> {
    // skeleton -> (size, program names, one binding vector per occurrence)
    let mut seen: Vec<(String, usize, Vec<String>, Vec<Vec<(String, bool)>>)> = Vec::new();

    for (name, src) in programs {
        let stmts = match crate::aiksi::lex::lex(src).and_then(crate::aiksi::parse::parse) {
            Ok(s) => s,
            // A stored program that no longer parses is a real thing to find,
            // and it is not this pass's job to report it. Skip it rather than
            // failing the whole scan over one bad file.
            Err(_) => continue,
        };
        let mut top = Vec::new();
        exprs_of(&stmts, &mut top);
        let mut all = Vec::new();
        for e in &top {
            subtrees(e, &mut all);
        }
        for e in &all {
            let n = size(e);
            if n < MIN_SIZE {
                continue;
            }
            let mut sk = String::new();
            let mut holes = Vec::new();
            canon(e, &mut sk, &mut holes);
            match seen.iter_mut().find(|(s, ..)| *s == sk) {
                Some((_, _, progs, binds)) => {
                    binds.push(holes);
                    if !progs.contains(name) {
                        progs.push(name.clone());
                    }
                }
                None => seen.push((sk, n, alloc::vec![name.clone()], alloc::vec![holes])),
            }
        }
    }

    let mut out = Vec::new();
    for (skeleton, size, programs, binds) in seen {
        let count = binds.len();
        if count < 2 {
            continue;
        }
        let (skeleton, arity) = fold(&skeleton, &binds);
        let saved = count as isize * (size as isize - 1 - arity as isize) - size as isize;
        if saved <= 0 {
            continue;
        }
        out.push(Candidate { skeleton, arity, size, count, programs, saved });
    }
    // Most saved first; ties broken by breadth, since a structure in three
    // programs is a library function and the same structure three times in one
    // program is a loop that was written out.
    out.sort_by(|a, b| {
        b.saved
            .cmp(&a.saved)
            .then(b.programs.len().cmp(&a.programs.len()))
            .then(b.size.cmp(&a.size))
    });
    out
}

/// Every stored skill, as (name, source).
///
/// Same walk `godel::next_skill` does, and the extension filter matters for
/// the same reason: `/ai/tools` also holds `.trusted`, which is a list of
/// hashes and not a program.
pub fn stored() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in crate::sysbox::children("/ai/tools") {
        if !name.ends_with(".ai&xi") {
            continue;
        }
        let mut path = String::from("/ai/tools/");
        path.push_str(&name);
        let Some(bytes) = crate::sysbox::read_blob(&path) else { continue };
        let Ok(text) = String::from_utf8(bytes) else { continue };
        out.push((name, text));
    }
    out
}

/// Read every stored skill and rank what repeats.
pub fn scan() -> Vec<Candidate> {
    analyse(&stored())
}

/// The canonicaliser and the objective, on programs whose answer is known.
///
/// Hardware-free and namespace-free on purpose. What has to be right here is
/// that two structures collide when and only when they should, and that the
/// objective refuses an abstraction that would not pay -- both are properties
/// of the arithmetic rather than of the machine.
pub fn selftest() -> bool {
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
            crate::kprintln!("    FAIL {}", what);
        }
    };

    // Same shape, different names: must collide, and abstract over one leaf
    // because the two occurrences of the variable are the same leaf.
    let (mut a, mut ha): (String, Vec<(String, bool)>) = (String::new(), Vec::new());
    let (mut b, mut hb): (String, Vec<(String, bool)>) = (String::new(), Vec::new());
    let pa = crate::aiksi::lex::lex("x = len(ls(p)) + len(ls(p))")
        .and_then(crate::aiksi::parse::parse);
    let pb = crate::aiksi::lex::lex("y = len(ls(q)) + len(ls(q))")
        .and_then(crate::aiksi::parse::parse);
    if let (Ok(sa), Ok(sb)) = (pa, pb) {
        let (mut ea, mut eb) = (Vec::new(), Vec::new());
        exprs_of(&sa, &mut ea);
        exprs_of(&sb, &mut eb);
        if let (Some(x), Some(y)) = (ea.first(), eb.first()) {
            canon(x, &mut a, &mut ha);
            canon(y, &mut b, &mut hb);
        }
    }
    claim("renaming does not change the shape", !a.is_empty() && a == b);
    claim("a repeated leaf is one parameter", ha.len() == 2);

    // Distinct leaves must NOT collapse: `p + q` is two parameters, and a
    // canonicaliser that lost that would propose abstractions that are wrong
    // rather than merely useless.
    let (mut c, mut hc): (String, Vec<(String, bool)>) = (String::new(), Vec::new());
    if let Ok(s) = crate::aiksi::lex::lex("z = len(ls(p)) + len(ls(q))")
        .and_then(crate::aiksi::parse::parse)
    {
        let mut e = Vec::new();
        exprs_of(&s, &mut e);
        if let Some(x) = e.first() {
            canon(x, &mut c, &mut hc);
        }
    }
    claim("distinct leaves stay distinct", !c.is_empty() && c != a);
    claim("two leaves are two parameters", hc.len() == 3);

    // Three programs sharing one structure. The shared call should rank, and
    // it should be credited to all three.
    let shared = "stat(path + \"/\" + get(names, n))";
    let progs = [
        ("a".to_string(), format!("s = {}\nprintln(s)", shared)),
        ("b".to_string(), format!("t = {}\nprintln(t)", shared)),
        ("c".to_string(), format!("u = {}\nprintln(u)", shared)),
    ];
    let found = analyse(&progs);
    claim("a shared structure is found", !found.is_empty());
    let breadth = found.iter().any(|c| c.programs.len() == 3);
    claim("it is credited to every program", breadth);

    // The objective must refuse what does not pay. Two occurrences of a
    // structure whose leaves are all different saves nothing, because the call
    // is as big as the body.
    let nothing = [
        ("a".to_string(), "x = f(p, q, r)".to_string()),
        ("b".to_string(), "y = f(s, t, u)".to_string()),
    ];
    let none = analyse(&nothing);
    claim("an all-holes structure is refused", none.is_empty());

    // A single occurrence is not an abstraction however large.
    let once = [("a".to_string(), "x = a + b + c + d + e + f".to_string())];
    claim("one occurrence is refused", analyse(&once).is_empty());

    // The proposal must be syntactically real, since the whole point is that
    // it could be written to /lib. Parsing it is the cheapest possible check
    // and it caught a missing `return` the first time.
    if let Some(top) = found.first() {
        let src = top.proposal(0);
        let parses = crate::aiksi::lex::lex(&src)
            .and_then(crate::aiksi::parse::parse)
            .is_ok();
        claim("the proposed function parses", parses);
        claim("the proposal has no holes left", !src.contains('#'));
    } else {
        claim("there was a proposal to check", false);
    }

    ok
}
