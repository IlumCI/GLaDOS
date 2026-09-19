//! A code generator for the part of Aiksi that is only integers.
//!
//! One slice, widened once. It started as a single function of Int
//! arithmetic, `if`, `while` and `return`, emitted as x86-64 into a
//! page-aligned heap buffer and called through the `sysv64` pointer
//! `cpu::code` declares. It now compiles **a program of functions that call
//! each other**, recursion included. Still no builtins, no strings, no
//! records, no `use`. Anything outside that is **refused at compile time**,
//! not approximated: `compile` answers `None` and the interpreter remains the
//! only thing that ran it.
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
//! **A call costs nothing of its own**, and that is a fact about
//! `Interp::call_user` rather than a convenience: it pushes a frame, checks
//! the depth and runs the body, and ticks at none of those points. So the
//! whole cost of `f(g(1))` is the two `Expr::Call` nodes, the argument
//! subtrees, and the statements of each body. Charging a tick for the call
//! itself would be the obvious thing and would be wrong by one per call.
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
//!
//! ### Locals moved to the machine stack, and they had to
//!
//! They were a flat array inside `Ctx`, which is correct for exactly as long
//! as one frame exists. A recursive call would have written its parameters
//! over its caller's, and the tell would have been a `fib` that answers
//! confidently and wrongly. So a function gets a real frame: `rbp` walks the
//! machine stack, locals live under it, arguments are pushed by the caller
//! and copied in by the callee's prologue.
//!
//! That trades one hazard for another and the trade is only sound because of
//! `eval::MAX_DEPTH`. This kernel has no guard page, so running off the stack
//! is a triple fault rather than an error message -- which is the same reason
//! the interpreter carries a depth cap, and the cap is *read from there*
//! rather than copied, because two numbers that have to agree and are written
//! down twice are two numbers that will not.
//!
//! ### Nil is a value the interpreter has and this does not
//!
//! A function that falls off its end yields `Value::Nil`, and `call_user`
//! then checks that against the declared return type. At the top level that is
//! a status `differ` can turn back into words. Inside an expression it is a
//! *value* flowing into arithmetic, and reproducing what the interpreter says
//! about `nil + 1` is a much larger promise than this slice makes.
//!
//! So **a called function must declare `: int`**, checked at the call site,
//! which is exactly where the Nil could escape. The entry may still be
//! unannotated and fall off its end, because nothing consumes what it
//! answers. A function that is Any-returning and also called -- including one
//! that calls itself -- is refused rather than compiled optimistically.

use super::eval::MAX_DEPTH;
use super::parse::{BinOp, Expr, Stmt, Type, UnOp};
use crate::cpu::code::{Compiled, Exec};
use alloc::vec::Vec;

/// How many named locals a compiled function may have, params included.
///
/// Small, because this is a slice and not a compiler. A function with more is
/// refused rather than spilled: spilling is a real allocator's job and the
/// point here is the pipeline, not the register allocation.
const MAX_SLOTS: usize = 16;

/// How many functions one compiled program may hold.
const MAX_FNS: usize = 32;

/// What the compiled program is handed, and what it writes back.
///
/// One pointer in `rdi` rather than arguments in registers, because the code
/// has to return several things -- a value, a step count, a status, and which
/// function is to blame for it -- and `sysv64` returns one. It also keeps
/// `Compiled` as the single declared signature the substrate already pins.
///
/// `rdi` is therefore a global register for the whole of a run. Nothing
/// generated here writes it, and nothing generated here calls anything that
/// was not generated here, so it survives every call without being saved.
#[repr(C)]
pub struct Ctx {
    /// Charged as the compiled code runs, at exactly the interpreter's points.
    pub steps: u64,
    /// Compared against `steps` after every tick. Exceeding it is `Budget`.
    pub budget: u64,
    pub result: i64,
    pub status: u64,
    /// How many frames are live. Checked against `MAX_DEPTH` in each
    /// prologue, before the frame exists, which is where `call_user` checks.
    pub depth: u64,
    /// Which function set the status, for the statuses that name one.
    pub blame: u64,
    /// The entry frame's `rbp`, so an abort at any depth can get back to Rust
    /// without unwinding: there is no unwinder here and nothing to run on the
    /// way out, so the whole of returning is restoring one register.
    pub exit: u64,
    /// The entry call's arguments. Locals live on the machine stack; this is
    /// only how the first frame's parameters cross from Rust.
    pub args: [i64; MAX_SLOTS],
}

const OFF_STEPS: i32 = 0;
const OFF_BUDGET: i32 = 8;
const OFF_RESULT: i32 = 16;
const OFF_STATUS: i32 = 24;
const OFF_DEPTH: i32 = 32;
const OFF_BLAME: i32 = 40;
const OFF_EXIT: i32 = 48;
const OFF_ARGS: i32 = 56;

pub const ST_VALUE: u64 = 0;
pub const ST_NIL: u64 = 1;
pub const ST_BUDGET: u64 = 2;
pub const ST_DIV0: u64 = 3;
pub const ST_REM0: u64 = 4;
/// The compiled code faulted and was caught. Distinct from every status the
/// code itself can set, because it is the one nothing inside it chose.
pub const ST_FAULT: u64 = 5;
/// Nesting reached `MAX_DEPTH`. `blame` is not set: the interpreter's message
/// names no function either.
pub const ST_DEPTH: u64 = 6;
/// A function declared `: int` fell off its end or returned bare. `blame` is
/// its index, because the interpreter's message names it.
pub const ST_RETNIL: u64 = 7;

/// What running a compiled program produced.
pub struct Run {
    pub status: u64,
    pub result: i64,
    pub steps: u64,
    pub blame: u64,
}

/// A compiled program, and the buffer it lives in.
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
            depth: 0,
            blame: 0,
            exit: 0,
            args: [0; MAX_SLOTS],
        };
        for (i, a) in args.iter().enumerate() {
            ctx.args[i] = *a;
        }
        let f: Compiled = unsafe { self.buf.entry()? };
        // The one place in this kernel that jumps into memory it wrote, and
        // now the one place where doing so is survivable.
        //
        // The plan for this back end said a code generation bug in a ring-0
        // image gets exactly one mistake, and that was true when every vector
        // was fatal. Inside a guard a bad jump is an error the caller reads
        // instead of a machine that stopped, which is what makes it reasonable
        // to run generated code at all.
        let ptr = &mut ctx as *mut Ctx as u64;
        let caught = crate::cpu::recover::guard(|| {
            unsafe { f(ptr) };
        });
        if caught.is_err() {
            return Some(Run { status: ST_FAULT, result: 0, steps: ctx.steps, blame: 0 });
        }
        Some(Run { status: ctx.status, result: ctx.result, steps: ctx.steps, blame: ctx.blame })
    }
}

// --- the assembler ------------------------------------------------------
//
// Hand-encoded, because bringing in an assembler for two dozen instructions
// would be more code than the two dozen instructions. Every encoding is
// written out beside its mnemonic so a reader can check it against a manual
// rather than against faith.

struct Asm {
    code: Vec<u8>,
    /// Jumps to the shared epilogues, patched once their addresses are known.
    budget_sites: Vec<usize>,
    div_sites: Vec<usize>,
    rem_sites: Vec<usize>,
    depth_sites: Vec<usize>,
    unwind_sites: Vec<usize>,
    /// `(hole, which function)`. A call is emitted before its target exists,
    /// which is what makes mutual recursion fall out rather than need a pass.
    call_sites: Vec<(usize, usize)>,
}

impl Asm {
    fn new() -> Asm {
        Asm {
            code: Vec::new(),
            budget_sites: Vec::new(),
            div_sites: Vec::new(),
            rem_sites: Vec::new(),
            depth_sites: Vec::new(),
            unwind_sites: Vec::new(),
            call_sites: Vec::new(),
        }
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

    /// `mov rax, [rbp+d]`  48 8b 85 d32 -- a local, or an incoming argument.
    fn load_frame(&mut self, d: i32) {
        self.put(&[0x48, 0x8b, 0x85]);
        self.put_i32(d);
    }

    /// `mov [rbp+d], rax`  48 89 85 d32
    fn store_frame(&mut self, d: i32) {
        self.put(&[0x48, 0x89, 0x85]);
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

    /// `jae rel32`  0f 83 -- unsigned, matching `self.depth >= MAX_DEPTH`.
    fn jae_hole(&mut self) -> usize {
        self.put(&[0x0f, 0x83]);
        let at = self.here();
        self.put_i32(0);
        at
    }

    /// `call rel32`  e8
    fn call_hole(&mut self) -> usize {
        self.put(&[0xe8]);
        let at = self.here();
        self.put_i32(0);
        at
    }

    fn patch_to(&mut self, hole: usize, target: usize) {
        let rel = target as i64 - (hole as i64 + 4);
        let b = (rel as i32).to_le_bytes();
        self.code[hole..hole + 4].copy_from_slice(&b);
    }

    /// Write an already-emitted `imm32` in place. Used for a frame size that
    /// is not known until the body it belongs to has been compiled.
    fn patch_imm(&mut self, at: usize, v: i32) {
        self.code[at..at + 4].copy_from_slice(&v.to_le_bytes());
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

    /// `add rsp, imm32`  48 81 c4 imm32 -- the caller taking its arguments
    /// back off the stack, which is the whole of the calling convention's
    /// clean-up half.
    fn add_rsp(&mut self, v: i32) {
        if v == 0 {
            return;
        }
        self.put(&[0x48, 0x81, 0xc4]);
        self.put_i32(v);
    }

    /// Restore the frame and return to the caller. `ret` pops only the return
    /// address; the arguments above it belong to the caller to remove.
    fn leave_ret(&mut self) {
        self.put(&[0x48, 0x89, 0xec]); // mov rsp, rbp
        self.put(&[0x5d]); // pop rbp
        self.put(&[0xc3]); // ret
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

/// Local `i` lives at `[rbp - 8(i+1)]`, under the saved frame pointer.
fn slot_off(i: usize) -> i32 {
    -8 * (i as i32 + 1)
}

/// Incoming argument `i` of `n`, at `[rbp + 16 + 8(n-1-i)]`.
///
/// The caller pushes arguments left to right, so the last one is nearest the
/// return address and the first is furthest from it. `+8` is that return
/// address and another `+8` is the saved `rbp` the prologue pushed.
fn arg_off(i: usize, n: usize) -> i32 {
    16 + 8 * (n - 1 - i) as i32
}

/// One function of a program, as `only_fns` found it.
pub struct Decl<'a> {
    pub name: &'a str,
    pub params: &'a [(alloc::string::String, Type)],
    pub ret: &'a Type,
    pub body: &'a [Stmt],
}

/// What a body is being compiled against: every function it may call, and
/// which one it is.
struct Unit<'a> {
    fns: &'a [Decl<'a>],
    me: usize,
}

/// Compile a program of functions, entering at one of them, or decline it.
///
/// Every parameter and every return type must be `int` or unannotated. An
/// annotation this cannot honour is a reason to refuse, not to ignore: the
/// interpreter checks types at the call boundary and a compiled function that
/// skipped the check would accept what the interpreter rejects.
pub fn compile(fns: &[Decl], entry: usize) -> Option<Program> {
    if fns.is_empty() || fns.len() > MAX_FNS || entry >= fns.len() {
        return None;
    }
    for f in fns {
        if !matches!(f.ret, Type::Int | Type::Any) || f.params.len() > MAX_SLOTS {
            return None;
        }
        for (_, t) in f.params {
            if !matches!(t, Type::Int | Type::Any) {
                return None;
            }
        }
    }
    let nargs = fns[entry].params.len();

    let mut a = Asm::new();

    // --- the trampoline, at offset 0 because `Exec::entry` is the buffer ---
    a.put(&[0x55]); // push rbp
    a.put(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
    a.put(&[0x48, 0x89, 0xaf]); // mov [rdi+exit], rbp
    a.put_i32(OFF_EXIT);
    for i in 0..nargs {
        a.load_ctx(OFF_ARGS + 8 * i as i32);
        a.put(&[0x50]); // push rax
    }
    let entry_call = a.call_hole();
    a.call_sites.push((entry_call, entry));
    a.add_rsp(8 * nargs as i32);
    a.store_ctx(OFF_RESULT);
    a.store_imm(OFF_STATUS, ST_VALUE as i32);
    a.leave_ret();

    // --- every function, in declaration order ---
    let mut starts = Vec::with_capacity(fns.len());
    for (k, f) in fns.iter().enumerate() {
        starts.push(a.here());
        let mut slots = Slots { names: Vec::new(), defined: Vec::new() };
        for (p, _) in f.params {
            slots.declare(p)?;
        }

        // The depth check comes first and before the frame exists, because
        // that is where `call_user` makes it: after the arguments have been
        // evaluated and charged, before the body runs and charges anything.
        a.load_ctx(OFF_DEPTH);
        a.put(&[0x48, 0x3d]); // cmp rax, imm32
        a.put_i32(MAX_DEPTH as i32);
        let deep = a.jae_hole();
        a.depth_sites.push(deep);
        a.put(&[0x48, 0xff, 0x87]); // inc qword [rdi+depth]
        a.put_i32(OFF_DEPTH);

        a.put(&[0x55]); // push rbp
        a.put(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
        a.put(&[0x48, 0x81, 0xec]); // sub rsp, imm32 -- size patched below
        let frame_hole = a.here();
        a.put_i32(0);

        let n = f.params.len();
        for i in 0..n {
            a.load_frame(arg_off(i, n));
            a.store_frame(slot_off(i));
        }

        let unit = Unit { fns, me: k };
        stmts(&mut a, f.body, &mut slots, &unit)?;

        // Falling off the end is `Value::Nil`, which `call_user` then checks
        // against the declared return type. Both halves of that are emitted,
        // rather than the fall-through being refused, so a function that only
        // returns down some paths still compares.
        fall_off(&mut a, f.ret, k);

        // Rounded to sixteen. Nothing generated here calls anything that
        // cares, and long mode realigns the stack itself before pushing an
        // interrupt frame, so this is tidiness rather than a requirement --
        // said here because the opposite belief costs a debugging session.
        let frame = ((slots.names.len() * 8) as i32 + 15) & !15;
        a.patch_imm(frame_hole, frame);
    }

    // --- the shared exits ---
    //
    // Emitted last so every jump to them is resolved from one place, and
    // `unwind` before the aborts so the aborts can jump backwards to it.
    let at_unwind = a.here();
    a.put(&[0x48, 0x8b, 0xaf]); // mov rbp, [rdi+exit]
    a.put_i32(OFF_EXIT);
    a.leave_ret();

    let mut abort = |a: &mut Asm, st: u64| -> usize {
        let at = a.here();
        a.store_imm(OFF_STATUS, st as i32);
        let h = a.jmp_hole();
        a.unwind_sites.push(h);
        at
    };
    let at_budget = abort(&mut a, ST_BUDGET);
    let at_div = abort(&mut a, ST_DIV0);
    let at_rem = abort(&mut a, ST_REM0);
    let at_depth = abort(&mut a, ST_DEPTH);

    for (h, k) in core::mem::take(&mut a.call_sites) {
        let target = starts[k];
        a.patch_to(h, target);
    }
    for (holes, target) in [
        (core::mem::take(&mut a.budget_sites), at_budget),
        (core::mem::take(&mut a.div_sites), at_div),
        (core::mem::take(&mut a.rem_sites), at_rem),
        (core::mem::take(&mut a.depth_sites), at_depth),
        (core::mem::take(&mut a.unwind_sites), at_unwind),
    ] {
        for h in holes {
            a.patch_to(h, target);
        }
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
    Some(Program { buf, params: nargs })
}

/// What a function does when control reaches its end without a `return`.
///
/// Two shapes and the difference is whose problem the Nil is. A function
/// declared `: int` has one that `call_user` refuses by name, so the index
/// travels with it. An unannotated one can only be the entry -- a call to an
/// Any-returning function is refused at the call site -- so its Nil is the
/// program's answer and stops there.
fn fall_off(a: &mut Asm, ret: &Type, k: usize) {
    if matches!(ret, Type::Int) {
        a.store_imm(OFF_BLAME, k as i32);
        a.store_imm(OFF_STATUS, ST_RETNIL as i32);
    } else {
        a.store_imm(OFF_STATUS, ST_NIL as i32);
    }
    let h = a.jmp_hole();
    a.unwind_sites.push(h);
}

fn stmts(a: &mut Asm, body: &[Stmt], slots: &mut Slots, u: &Unit) -> Option<()> {
    for s in body {
        stmt(a, s, slots, u)?;
    }
    Some(())
}

fn stmt(a: &mut Asm, s: &Stmt, slots: &mut Slots, u: &Unit) -> Option<()> {
    // `Interp::stmt` ticks once on entry, whatever the statement is.
    a.tick();
    match s {
        Stmt::Expr(e) => expr(a, e, slots, u),
        Stmt::Return(Some(e)) => {
            expr(a, e, slots, u)?;
            // The frame goes away and the depth with it. Decrementing here
            // rather than in the caller keeps the two in step down every path
            // out of a function, which is the property an abort relies on not
            // needing: an abort never returns, so it never has to unwind one.
            a.put(&[0x48, 0xff, 0x8f]); // dec qword [rdi+depth]
            a.put_i32(OFF_DEPTH);
            a.leave_ret();
            Some(())
        }
        Stmt::Return(None) => {
            // A bare `return` yields Nil, which is the same value falling off
            // the end yields and is refused in the same words.
            fall_off(a, u.fns[u.me].ret, u.me);
            Some(())
        }
        Stmt::If(cond, then, otherwise) => {
            expr(a, cond, slots, u)?;
            // test rax, rax   48 85 c0
            a.put(&[0x48, 0x85, 0xc0]);
            let to_else = a.jz_hole();
            // Assignments inside a branch do not become definitely-defined
            // outside it, so the branch compiles against a copy.
            let mut inner = Slots { names: slots.names.clone(), defined: slots.defined.clone() };
            stmts(a, then, &mut inner, u)?;
            // Names *declared* in a branch still need slots, or two branches
            // would reuse one index for different variables.
            adopt_names(slots, &inner);
            let to_end = a.jmp_hole();
            let else_at = a.here();
            a.patch_to(to_else, else_at);
            if let Some(els) = otherwise {
                let mut inner = Slots { names: slots.names.clone(), defined: slots.defined.clone() };
                stmts(a, els, &mut inner, u)?;
                adopt_names(slots, &inner);
            }
            let end = a.here();
            a.patch_to(to_end, end);
            Some(())
        }
        Stmt::While(cond, body) => {
            let top = a.here();
            expr(a, cond, slots, u)?;
            a.put(&[0x48, 0x85, 0xc0]);
            let to_end = a.jz_hole();
            // The extra tick per iteration, charged after the condition has
            // answered true and before the body -- exactly where the
            // interpreter's `self.tick()?` sits inside its `while`.
            a.tick();
            let mut inner = Slots { names: slots.names.clone(), defined: slots.defined.clone() };
            stmts(a, body, &mut inner, u)?;
            adopt_names(slots, &inner);
            let back = a.jmp_hole();
            a.patch_to(back, top);
            let end = a.here();
            a.patch_to(to_end, end);
            Some(())
        }
        // Everything else is out of the slice, and being out of it is not a
        // thing to paper over. A nested `fn` is here rather than anywhere
        // else: the interpreter would declare it at the moment the statement
        // runs, so hoisting it to the program's function table would make it
        // callable before the interpreter has it.
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

fn expr(a: &mut Asm, e: &Expr, slots: &mut Slots, u: &Unit) -> Option<()> {
    // `Interp::expr` ticks once on entry, for every node.
    a.tick();
    match e {
        Expr::Int(v) => {
            a.mov_rax_imm(*v);
            Some(())
        }
        Expr::Var(name) => {
            let i = slots.readable(name)?;
            a.load_frame(slot_off(i));
            Some(())
        }
        Expr::Assign(name, rhs) => {
            expr(a, rhs, slots, u)?;
            let i = slots.declare(name)?;
            a.store_frame(slot_off(i));
            Some(())
        }
        Expr::Unary(op, inner) => {
            expr(a, inner, slots, u)?;
            match op {
                // neg rax   48 f7 d8
                UnOp::Neg => a.put(&[0x48, 0xf7, 0xd8]),
                // test rax,rax; sete al; movzx rax, al
                UnOp::Not => a.put(&[0x48, 0x85, 0xc0, 0x0f, 0x94, 0xc0, 0x48, 0x0f, 0xb6, 0xc0]),
                _ => return None,
            }
            Some(())
        }
        Expr::Bin(op, l, r) => binary(a, *op, l, r, slots, u),
        Expr::Call(name, args) => call(a, name, args, slots, u),
        _ => None,
    }
}

/// A call to another function of this program.
///
/// Four refusals, and each is a place the compiled route could otherwise
/// answer where the interpreter does not:
///
/// - **A name this program does not define.** The interpreter would look past
///   its functions to records and then to builtins, and the slice admits
///   neither, so there is nothing to emit.
/// - **The wrong number of arguments.** That is a static mistake with a
///   static message, and emitting code whose only purpose is to produce an
///   error is emitting code nobody wanted run.
/// - **A callee that is not `: int`.** See the module header: Nil inside an
///   expression is a promise this slice does not make.
/// - **Too many arguments to push.** `MAX_SLOTS` bounds a frame and bounds
///   this with it, so a call cannot outgrow what the callee can hold.
fn call(a: &mut Asm, name: &str, args: &[Expr], slots: &mut Slots, u: &Unit) -> Option<()> {
    let k = u.fns.iter().position(|f| f.name == name)?;
    let callee = &u.fns[k];
    if args.len() != callee.params.len() || args.len() > MAX_SLOTS {
        return None;
    }
    if !matches!(callee.ret, Type::Int) {
        return None;
    }
    // Left to right, which is the order the interpreter evaluates them in and
    // therefore the order their ticks are charged in. A right-to-left push
    // would give the same answer and the wrong step count on any argument
    // whose evaluation can fail.
    for arg in args {
        expr(a, arg, slots, u)?;
        a.put(&[0x50]); // push rax
    }
    let h = a.call_hole();
    a.call_sites.push((h, k));
    a.add_rsp(8 * args.len() as i32);
    Some(())
}

fn binary(a: &mut Asm, op: BinOp, l: &Expr, r: &Expr, slots: &mut Slots, u: &Unit) -> Option<()> {
    // Short-circuit first and before the right side is touched, because the
    // interpreter does and because the ticks of the right subtree are part of
    // what has to match.
    if op == BinOp::LogAnd || op == BinOp::LogOr {
        expr(a, l, slots, u)?;
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
        expr(a, r, slots, u)?;
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

    expr(a, l, slots, u)?;
    a.put(&[0x50]); // push rax
    expr(a, r, slots, u)?;
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

/// A program that is nothing but function declarations, if that is what it is.
///
/// Every top-level statement must be one, because anything else runs at the
/// top level and compiled code has no top level to run it in. That also fixes
/// the tick the harness owes: the interpreter charges one per declaration
/// before anything is called, so the count is the length of this and not an
/// estimate.
pub fn only_fns(prog: &[Stmt]) -> Option<Vec<Decl<'_>>> {
    if prog.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(prog.len());
    for s in prog {
        match s {
            Stmt::Fn(name, params, ret, body) => out.push(Decl {
                name: name.as_str(),
                params: params.as_slice(),
                ret,
                body: body.as_slice(),
            }),
            _ => return None,
        }
    }
    // Two functions of one name is the interpreter keeping the later one, and
    // a call resolved to the earlier one here would be a different program.
    for i in 0..out.len() {
        for j in i + 1..out.len() {
            if out[i].name == out[j].name {
                return None;
            }
        }
    }
    Some(out)
}
