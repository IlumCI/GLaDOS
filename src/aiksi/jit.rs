//! A code generator for the part of Aiksi that is only integers.
//!
//! The smallest slice that proves the approach: one function, Int arithmetic,
//! `if`, `while`, `return`, emitted as x86-64 into a page-aligned heap buffer
//! and called through the `sysv64` pointer `cpu::code` declares. No builtins,
//! no strings, no records, no `use`, no calls of any kind. Anything outside
//! that is **refused at compile time**, not approximated: `compile` answers
//! `None` and the interpreter remains the only thing that ran it.
//!
//! It is reached only from `differ`, never from a live path. Nothing routes
//! through this, nothing in `voter` knows it exists, and the one thing it is
//! for is to be disagreed with.
//!
//! **The step count is the hard part, not the arithmetic.** `Interp::tick`
//! states the rule -- once entering `stmt`, once entering `expr`, one extra
//! per `while` iteration, and nothing else -- and compiled code has to charge
//! at exactly those points. Not approximately: the budget is a safety bound,
//! so a compiled program given more room than an interpreted one is a runaway
//! that one path stops and the other does not, and `differ` compares step
//! counts bit for bit precisely because that is the number most easily got
//! nearly right.
//!
//! Short-circuit is where "nearly right" lives. `&&` evaluates its right side
//! only when the left is true, so the ticks of the right subtree happen or do
//! not depending on a value. Compiled code jumps over them the same way.
//!
//! **Everything the interpreter can answer, this must answer identically**,
//! including how it fails. Division by zero is `"division by zero"` there, so
//! it is a status here that `differ` turns back into that exact string. A
//! compiler that got the arithmetic right and the error text wrong would pass
//! a comparison that only looked at answers.

use super::parse::{BinOp, Expr, Stmt, Type, UnOp};
use crate::cpu::code::{Compiled, Exec};
use alloc::vec::Vec;

/// How many named locals a compiled function may have, params included.
///
/// Small, because this is a slice and not a compiler. A function with more is
/// refused rather than spilled: spilling is a real allocator's job and the
/// point here is the pipeline, not the register allocation.
const MAX_SLOTS: usize = 16;

/// What the compiled function is handed, and what it writes back.
///
/// One pointer in `rdi` rather than arguments in registers, because the
/// function has to return four things -- a value, a step count, a status, and
/// which of three ways it failed -- and `sysv64` returns one. It also keeps
/// `Compiled` as the single declared signature the substrate already pins.
#[repr(C)]
pub struct Ctx {
    /// Charged as the compiled code runs, at exactly the interpreter's points.
    pub steps: u64,
    /// Compared against `steps` after every tick. Exceeding it is `Budget`.
    pub budget: u64,
    pub result: i64,
    pub status: u64,
    pub slots: [i64; MAX_SLOTS],
}

const OFF_STEPS: i32 = 0;
const OFF_BUDGET: i32 = 8;
const OFF_RESULT: i32 = 16;
const OFF_STATUS: i32 = 24;
const OFF_SLOTS: i32 = 32;

pub const ST_VALUE: u64 = 0;
pub const ST_NIL: u64 = 1;
pub const ST_BUDGET: u64 = 2;
pub const ST_DIV0: u64 = 3;
pub const ST_REM0: u64 = 4;

/// What running a compiled function produced.
pub struct Run {
    pub status: u64,
    pub result: i64,
    pub steps: u64,
}

/// A compiled function, and the buffer it lives in.
pub struct Program {
    buf: Exec,
    params: usize,
}

impl Program {
    pub fn params(&self) -> usize {
        self.params
    }

    /// Call it. `None` if the argument count is wrong, which the interpreter
    /// would have refused before running anything.
    pub fn run(&self, args: &[i64], budget: u64) -> Option<Run> {
        if args.len() != self.params {
            return None;
        }
        let mut ctx = Ctx {
            steps: 0,
            budget,
            result: 0,
            status: ST_NIL,
            slots: [0; MAX_SLOTS],
        };
        for (i, a) in args.iter().enumerate() {
            ctx.slots[i] = *a;
        }
        let f: Compiled = unsafe { self.buf.entry()? };
        // The one place in this kernel that jumps into memory it wrote.
        unsafe { f(&mut ctx as *mut Ctx as u64) };
        Some(Run { status: ctx.status, result: ctx.result, steps: ctx.steps })
    }
}

// --- the assembler ------------------------------------------------------
//
// Hand-encoded, because bringing in an assembler for eleven instructions
// would be more code than the eleven instructions. Every encoding is written
// out beside its mnemonic so a reader can check it against a manual rather
// than against faith.

struct Asm {
    code: Vec<u8>,
    /// Jumps to the epilogues, patched once their addresses are known.
    budget_sites: Vec<usize>,
    div_sites: Vec<usize>,
    rem_sites: Vec<usize>,
}

impl Asm {
    fn new() -> Asm {
        Asm { code: Vec::new(), budget_sites: Vec::new(), div_sites: Vec::new(), rem_sites: Vec::new() }
    }

    fn put(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    fn put_i32(&mut self, v: i32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    fn here(&self) -> usize {
        self.code.len()
    }

    /// `mov rax, [rdi+d]`  48 8b 87 d32
    fn load_ctx(&mut self, d: i32) {
        self.put(&[0x48, 0x8b, 0x87]);
        self.put_i32(d);
    }

    /// `mov [rdi+d], rax`  48 89 87 d32
    fn store_ctx(&mut self, d: i32) {
        self.put(&[0x48, 0x89, 0x87]);
        self.put_i32(d);
    }

    /// `mov qword [rdi+d], imm32`  48 c7 87 d32 imm32
    fn store_imm(&mut self, d: i32, v: i32) {
        self.put(&[0x48, 0xc7, 0x87]);
        self.put_i32(d);
        self.put_i32(v);
    }

    /// `mov rax, imm64`  48 b8 imm64
    fn mov_rax_imm(&mut self, v: i64) {
        self.put(&[0x48, 0xb8]);
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    /// A `jmp`/`jcc` with a hole where the displacement goes. Answers the
    /// offset of that hole.
    fn jmp_hole(&mut self) -> usize {
        self.put(&[0xe9]);
        let at = self.here();
        self.put_i32(0);
        at
    }

    /// `jz rel32`  0f 84
    fn jz_hole(&mut self) -> usize {
        self.put(&[0x0f, 0x84]);
        let at = self.here();
        self.put_i32(0);
        at
    }

    /// `ja rel32`  0f 87 -- unsigned above, because steps and budget are u64
    /// and the interpreter's test is `self.steps > self.budget`.
    fn ja_hole(&mut self) -> usize {
        self.put(&[0x0f, 0x87]);
        let at = self.here();
        self.put_i32(0);
        at
    }

    fn patch_to(&mut self, hole: usize, target: usize) {
        let rel = target as i64 - (hole as i64 + 4);
        let b = (rel as i32).to_le_bytes();
        self.code[hole..hole + 4].copy_from_slice(&b);
    }

    /// One step, charged and checked.
    ///
    /// `inc qword [rdi+steps]` then compare against the budget. This is the
    /// whole of budget enforcement in compiled code, and it is emitted at the
    /// same places `Interp::tick` is called and nowhere else.
    fn tick(&mut self) {
        // inc qword [rdi+0]   48 ff 87 d32
        self.put(&[0x48, 0xff, 0x87]);
        self.put_i32(OFF_STEPS);
        self.load_ctx(OFF_STEPS);
        // cmp rax, [rdi+budget]   48 3b 87 d32
        self.put(&[0x48, 0x3b, 0x87]);
        self.put_i32(OFF_BUDGET);
        let h = self.ja_hole();
        self.budget_sites.push(h);
    }
}

/// Which names a function may touch, and where each lives.
///
/// Assignments inside an `if` or a `while` do not count as definitely made,
/// because whether they ran depends on a value. A read of a name that might
/// not have been assigned is refused rather than compiled as a zero: the
/// interpreter answers `undefined variable 'x'` there, and answering 0
/// instead is the kind of divergence that looks like a working program.
struct Slots {
    names: Vec<alloc::string::String>,
    defined: Vec<bool>,
}

impl Slots {
    fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    fn readable(&self, name: &str) -> Option<usize> {
        let i = self.index(name)?;
        if self.defined[i] {
            Some(i)
        } else {
            None
        }
    }

    fn declare(&mut self, name: &str) -> Option<usize> {
        if let Some(i) = self.index(name) {
            self.defined[i] = true;
            return Some(i);
        }
        if self.names.len() >= MAX_SLOTS {
            return None;
        }
        self.names.push(alloc::string::String::from(name));
        self.defined.push(true);
        Some(self.names.len() - 1)
    }
}

fn slot_off(i: usize) -> i32 {
    OFF_SLOTS + (i as i32) * 8
}

/// Compile one function, or decline it.
///
/// Every parameter and the return type must be `int` or unannotated. An
/// annotation this cannot honour is a reason to refuse, not to ignore: the
/// interpreter checks types at the call boundary and a compiled function that
/// skipped the check would accept what the interpreter rejects.
pub fn compile(params: &[(alloc::string::String, Type)], ret: &Type, body: &[Stmt]) -> Option<Program> {
    if !matches!(ret, Type::Int | Type::Any) || params.len() > MAX_SLOTS {
        return None;
    }
    let mut slots = Slots { names: Vec::new(), defined: Vec::new() };
    for (p, t) in params {
        if !matches!(t, Type::Int | Type::Any) {
            return None;
        }
        slots.declare(p)?;
    }

    let mut a = Asm::new();
    stmts(&mut a, body, &mut slots)?;

    // Falling off the end is `Value::Nil` in the interpreter. Compiled the
    // same way rather than refused, so a function that only returns down some
    // paths still compares.
    a.store_imm(OFF_STATUS, ST_NIL as i32);
    a.put(&[0xc3]);

    // The three failure epilogues. Emitted last so every jump to them is
    // forwards and patched from one place.
    let at_budget = a.here();
    a.store_imm(OFF_STATUS, ST_BUDGET as i32);
    a.put(&[0xc3]);
    let at_div = a.here();
    a.store_imm(OFF_STATUS, ST_DIV0 as i32);
    a.put(&[0xc3]);
    let at_rem = a.here();
    a.store_imm(OFF_STATUS, ST_REM0 as i32);
    a.put(&[0xc3]);

    for h in core::mem::take(&mut a.budget_sites) {
        a.patch_to(h, at_budget);
    }
    for h in core::mem::take(&mut a.div_sites) {
        a.patch_to(h, at_div);
    }
    for h in core::mem::take(&mut a.rem_sites) {
        a.patch_to(h, at_rem);
    }

    let mut buf = Exec::new(a.code.len())?;
    if !buf.push(&a.code) {
        return None;
    }
    // Serialise and register before anything can be entered. The tag is not a
    // content hash here because nothing stored this program; what a fault in
    // it needs to say is that it came from the compiler.
    if !buf.arm(0x71C0_0000_0000_0000) {
        return None;
    }
    Some(Program { buf, params: params.len() })
}

fn stmts(a: &mut Asm, body: &[Stmt], slots: &mut Slots) -> Option<()> {
    for s in body {
        stmt(a, s, slots)?;
    }
    Some(())
}

fn stmt(a: &mut Asm, s: &Stmt, slots: &mut Slots) -> Option<()> {
    // `Interp::stmt` ticks once on entry, whatever the statement is.
    a.tick();
    match s {
        Stmt::Expr(e) => expr(a, e, slots),
        Stmt::Return(Some(e)) => {
            expr(a, e, slots)?;
            a.store_ctx(OFF_RESULT);
            a.store_imm(OFF_STATUS, ST_VALUE as i32);
            a.put(&[0xc3]);
            Some(())
        }
        Stmt::Return(None) => {
            a.store_imm(OFF_STATUS, ST_NIL as i32);
            a.put(&[0xc3]);
            Some(())
        }
        Stmt::If(cond, then, otherwise) => {
            expr(a, cond, slots)?;
            // test rax, rax   48 85 c0
            a.put(&[0x48, 0x85, 0xc0]);
            let to_else = a.jz_hole();
            // Assignments inside a branch do not become definitely-defined
            // outside it, so the branch compiles against a copy.
            let mut inner = Slots { names: slots.names.clone(), defined: slots.defined.clone() };
            stmts(a, then, &mut inner)?;
            // Names *declared* in a branch still need slots, or two branches
            // would reuse one index for different variables.
            adopt_names(slots, &inner);
            let to_end = a.jmp_hole();
            let else_at = a.here();
            a.patch_to(to_else, else_at);
            if let Some(els) = otherwise {
                let mut inner = Slots { names: slots.names.clone(), defined: slots.defined.clone() };
                stmts(a, els, &mut inner)?;
                adopt_names(slots, &inner);
            }
            let end = a.here();
            a.patch_to(to_end, end);
            Some(())
        }
        Stmt::While(cond, body) => {
            let top = a.here();
            expr(a, cond, slots)?;
            a.put(&[0x48, 0x85, 0xc0]);
            let to_end = a.jz_hole();
            // The extra tick per iteration, charged after the condition has
            // answered true and before the body -- exactly where the
            // interpreter's `self.tick()?` sits inside its `while`.
            a.tick();
            let mut inner = Slots { names: slots.names.clone(), defined: slots.defined.clone() };
            stmts(a, body, &mut inner)?;
            adopt_names(slots, &inner);
            let back = a.jmp_hole();
            a.patch_to(back, top);
            let end = a.here();
            a.patch_to(to_end, end);
            Some(())
        }
        // Everything else is out of the slice, and being out of it is not a
        // thing to paper over.
        _ => None,
    }
}

/// Carry slot *indices* out of a branch without carrying definedness.
fn adopt_names(outer: &mut Slots, inner: &Slots) {
    for n in &inner.names {
        if outer.index(n).is_none() {
            outer.names.push(n.clone());
            outer.defined.push(false);
        }
    }
}

fn expr(a: &mut Asm, e: &Expr, slots: &mut Slots) -> Option<()> {
    // `Interp::expr` ticks once on entry, for every node.
    a.tick();
    match e {
        Expr::Int(v) => {
            a.mov_rax_imm(*v);
            Some(())
        }
        Expr::Var(name) => {
            let i = slots.readable(name)?;
            a.load_ctx(slot_off(i));
            Some(())
        }
        Expr::Assign(name, rhs) => {
            expr(a, rhs, slots)?;
            let i = slots.declare(name)?;
            a.store_ctx(slot_off(i));
            Some(())
        }
        Expr::Unary(op, inner) => {
            expr(a, inner, slots)?;
            match op {
                // neg rax   48 f7 d8
                UnOp::Neg => a.put(&[0x48, 0xf7, 0xd8]),
                // test rax,rax; sete al; movzx rax, al
                UnOp::Not => a.put(&[0x48, 0x85, 0xc0, 0x0f, 0x94, 0xc0, 0x48, 0x0f, 0xb6, 0xc0]),
                _ => return None,
            }
            Some(())
        }
        Expr::Bin(op, l, r) => binary(a, *op, l, r, slots),
        _ => None,
    }
}

fn binary(a: &mut Asm, op: BinOp, l: &Expr, r: &Expr, slots: &mut Slots) -> Option<()> {
    // Short-circuit first and before the right side is touched, because the
    // interpreter does and because the ticks of the right subtree are part of
    // what has to match.
    if op == BinOp::LogAnd || op == BinOp::LogOr {
        expr(a, l, slots)?;
        a.put(&[0x48, 0x85, 0xc0]); // test rax, rax
        let short = if op == BinOp::LogAnd {
            a.jz_hole()
        } else {
            // jnz rel32   0f 85
            a.put(&[0x0f, 0x85]);
            let at = a.here();
            a.put_i32(0);
            at
        };
        expr(a, r, slots)?;
        // The right side decides: truthy becomes 1, falsy 0.
        a.put(&[0x48, 0x85, 0xc0, 0x0f, 0x95, 0xc0, 0x48, 0x0f, 0xb6, 0xc0]);
        let done = a.jmp_hole();
        let at_short = a.here();
        a.patch_to(short, at_short);
        // `&&` that stopped early is 0; `||` that stopped early is 1.
        a.mov_rax_imm(if op == BinOp::LogAnd { 0 } else { 1 });
        let end = a.here();
        a.patch_to(done, end);
        return Some(());
    }

    expr(a, l, slots)?;
    a.put(&[0x50]); // push rax
    expr(a, r, slots)?;
    a.put(&[0x48, 0x89, 0xc1]); // mov rcx, rax
    a.put(&[0x58]); // pop rax   -- rax is now the left, rcx the right

    match op {
        BinOp::Add => a.put(&[0x48, 0x01, 0xc8]),
        BinOp::Sub => a.put(&[0x48, 0x29, 0xc8]),
        BinOp::Mul => a.put(&[0x48, 0x0f, 0xaf, 0xc1]),
        BinOp::Div | BinOp::Rem => {
            // cmp rcx, 0   48 83 f9 00
            a.put(&[0x48, 0x83, 0xf9, 0x00]);
            let zero = a.jz_hole();
            if op == BinOp::Div {
                a.div_sites.push(zero);
            } else {
                a.rem_sites.push(zero);
            }
            a.put(&[0x48, 0x99]); // cqo
            a.put(&[0x48, 0xf7, 0xf9]); // idiv rcx
            if op == BinOp::Rem {
                a.put(&[0x48, 0x89, 0xd0]); // mov rax, rdx
            }
        }
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
            a.put(&[0x48, 0x39, 0xc8]); // cmp rax, rcx
            let cc = match op {
                BinOp::Lt => 0x9c,
                BinOp::Le => 0x9e,
                BinOp::Gt => 0x9f,
                BinOp::Ge => 0x9d,
                BinOp::Eq => 0x94,
                _ => 0x95,
            };
            a.put(&[0x0f, cc, 0xc0]); // setcc al
            a.put(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
        }
        // Bit operations are out of the slice. They are easy to emit and
        // their shift semantics were not read, and emitting what was not
        // checked is how a compiler passes a test it should fail.
        _ => return None,
    }
    Some(())
}

/// The one `fn` in a program, if that is all it is.
pub fn only_fn(prog: &[Stmt]) -> Option<(&alloc::string::String, &[(alloc::string::String, Type)], &Type, &[Stmt])> {
    if prog.len() != 1 {
        return None;
    }
    match &prog[0] {
        Stmt::Fn(name, params, ret, body) => Some((name, params, ret, body)),
        _ => None,
    }
}
