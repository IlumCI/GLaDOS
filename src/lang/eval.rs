//! Tree-walking interpreter.
//!
//! Working beats fast. This evaluates the AST directly; the single-pass JIT
//! that replaces it can be written against exactly the same AST and checked
//! against exactly the same results, which is much easier than debugging a
//! code generator with nothing to compare it to.
//!
//! Builtins reach straight into the kernel. That is the point of a ring-0
//! single-address-space OS: `peek`, `poke`, `inb` and `outb` at the prompt are
//! a hardware debugger with no driver, no ioctl and no permission model in the
//! way. It is also exactly how you shoot yourself -- a bad `peek` faults, and
//! the M2 reporter prints the address. That is a feature.

use super::parse::{BinOp, Expr, Stmt, UnOp};
use crate::gfx::console::{self, PALETTE};
use crate::gfx::{self};
use crate::{kprint, kprintln};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Str(String),
    /// A sequence, and the only compound value there is.
    ///
    /// Applications hold collections -- rows in a list, cells on a board,
    /// lines in a document -- and a language with no way to hold one can only
    /// write calculators. Values, not references: `push` returns a new list
    /// rather than mutating a shared one, so there is no aliasing to reason
    /// about and no question of what two names pointing at the same list
    /// means. It copies, and a copy of a list that fits on a screen is
    /// nothing.
    List(Vec<Value>),
    Nil,
}

impl Value {
    pub fn truthy(&self) -> bool {
        match self {
            Value::Int(v) => *v != 0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            Value::Nil => false,
        }
    }

    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Value::Int(v) => Ok(*v),
            Value::Str(_) => Err("expected a number, found a string".to_string()),
            Value::List(_) => Err("expected a number, found a list".to_string()),
            Value::Nil => Err("expected a number, found nothing".to_string()),
        }
    }

    pub fn render(&self) -> String {
        match self {
            Value::Int(v) => format!("{}", v),
            Value::Str(s) => s.clone(),
            Value::List(items) => {
                let mut out = String::from("[");
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&v.render());
                }
                out.push(']');
                out
            }
            Value::Nil => String::new(),
        }
    }
}

/// Without this, `while (1) {}` at the prompt wedges the shell task forever.
/// There is no Ctrl-C to rescue you: the clock task keeps running because of
/// preemption, but nothing would ever schedule the shell back into a state
/// where it could read a key.
const STEP_BUDGET: u64 = 20_000_000;

/// What a program gets when something other than a person is waiting for it.
///
/// The full budget is sized for the prompt, where the operator can see a long
/// loop running and stop it. It is the wrong size for a program the desktop
/// calls: `app::document` runs an application's `rows()` on **every repaint**,
/// so a generated loop that takes a second makes the window manager feel
/// broken, and the symptom points at the compositor rather than at the
/// application. Small enough to bound a repaint, large enough that no
/// reasonable list-building loop reaches it.
pub const DRAW_BUDGET: u64 = 200_000;

/// A named procedure: its parameters and its body.
#[derive(Clone)]
struct Func {
    params: Vec<String>,
    body: Vec<Stmt>,
}

/// How deep calls may nest before it is called a runaway.
///
/// Recursion is not the reason. The step budget already stops a program that
/// loops forever, but it does not stop one that recurses forever, because
/// every frame is a fresh allocation and the kernel runs out of stack long
/// before twenty million steps -- and running out of stack in ring 0 with no
/// guard page is a triple fault, not an error message.
const MAX_DEPTH: usize = 64;

/// What a program is allowed to reach.
///
/// The gate lives here and not in `sysbox` or the shell for one reason: the
/// raw builtins are reachable from a bare expression at the prompt and from
/// any stored program `run` executes, so a check anywhere else has a hole
/// shaped like the other path. This is the only place both go through.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Caps {
    /// The operator's own interpreter. Everything, including `poke` and the
    /// I/O ports, which is the point of a ring-0 system with one address
    /// space: the prompt is a hardware debugger with nothing in the way.
    Operator,
    /// A stored application. No raw memory, no ports, no drawing outside its
    /// window, applets limited to those that do not mutate, and writes
    /// confined to its own subtree.
    Sandbox,
}

/// Builtins that reach the machine directly.
///
/// Checked as a list before the dispatch rather than one guard per arm,
/// because a per-arm check is a list that gets forgotten the next time a
/// builtin lands -- and the one that gets forgotten is the one that matters.
const RAW: &[&str] = &[
    "peek8", "peek16", "peek32", "peek64", "poke8", "poke32", "poke64", "inb", "outb", "inl",
    "outl",
];

/// Drawing builtins, which paint straight at `gfx::primary()` with no idea
/// which window is asking. A sandboxed application using one would paint over
/// the whole desktop, so they are the operator's until there is a windowed way
/// to offer them. A panel application has no need for them.
const DRAWS: &[&str] = &["pixel", "rect", "text"];

pub struct Interp {
    /// Innermost scope last. A name is looked up from the top down and
    /// assigned wherever it is already bound, so a function can read and
    /// update a global without ceremony, and a parameter shadows one without
    /// destroying it.
    scopes: Vec<BTreeMap<String, Value>>,
    funcs: BTreeMap<String, Func>,
    /// Set by `return`, and checked after every statement in a block. A
    /// sentinel rather than a control-flow type threaded through every
    /// signature, which for a tree-walker this size is the same thing with
    /// less to read.
    returning: Option<Value>,
    depth: usize,
    steps: u64,
    budget: u64,
    caps: Caps,
    /// The subtree a sandboxed program may write into. Absolute, with no
    /// trailing slash.
    jail: Option<String>,
}

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}

impl Interp {
    pub fn new() -> Self {
        Self {
            scopes: alloc::vec![BTreeMap::new()],
            funcs: BTreeMap::new(),
            returning: None,
            depth: 0,
            steps: 0,
            budget: STEP_BUDGET,
            caps: Caps::Operator,
            jail: None,
        }
    }

    /// An interpreter for a stored program, confined to one subtree.
    ///
    /// Every existing caller keeps `new` and keeps everything, which is what
    /// makes this safe to add: the prompt, the shell's session and the model's
    /// own tools are unchanged, and only code that opts in is confined.
    pub fn sandboxed(jail: &str) -> Self {
        let mut it = Self::new();
        it.caps = Caps::Sandbox;
        it.jail = Some(String::from(jail.trim_end_matches('/')));
        it
    }

    /// Lower the step budget. Cannot be raised above the default.
    pub fn with_step_budget(mut self, n: u64) -> Self {
        self.budget = n.min(STEP_BUDGET);
        self
    }

    pub fn caps(&self) -> Caps {
        self.caps
    }

    /// True if a sandboxed program may write here.
    ///
    /// The path is resolved first. A jail compared against what was typed is
    /// defeated by `../..`, which is the entire history of this kind of check.
    fn may_write(&self, path: &str) -> bool {
        let Some(jail) = &self.jail else {
            return true;
        };
        let full = crate::sysbox::resolve_path(path);
        // The subtree, and not merely the prefix: `/app/todo-evil` must not
        // pass a jail of `/app/todo`.
        full == *jail || full.starts_with(&alloc::format!("{}/", jail))
    }

    /// The global scope, which is what `vars` at the prompt means: a function's
    /// locals exist only while it is running and there is nothing to show.
    pub fn var_count(&self) -> usize {
        self.scopes[0].len()
    }

    pub fn vars(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.scopes[0].iter()
    }

    pub fn fn_names(&self) -> impl Iterator<Item = &String> {
        self.funcs.keys()
    }

    fn lookup(&self, name: &str) -> Option<&Value> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    /// Bind a name where it already lives, or in the innermost scope if it is
    /// new. So a function updating a global updates the global, and one
    /// introducing a name keeps it to itself.
    fn assign(&mut self, name: &str, v: Value) {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                *slot = v;
                return;
            }
        }
        if let Some(top) = self.scopes.last_mut() {
            top.insert(String::from(name), v);
        }
    }

    /// Run a block, stopping early if something inside it returned.
    fn body(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for st in stmts {
            self.stmt(st)?;
            if self.returning.is_some() {
                break;
            }
        }
        Ok(())
    }

    fn call_user(&mut self, f: &Func, args: &[Value]) -> Result<Value, String> {
        if args.len() != f.params.len() {
            return Err(format!(
                "expected {} argument(s), got {}",
                f.params.len(),
                args.len()
            ));
        }
        if self.depth >= MAX_DEPTH {
            return Err("call nesting too deep (runaway recursion?)".to_string());
        }
        let mut frame = BTreeMap::new();
        for (p, a) in f.params.iter().zip(args.iter()) {
            frame.insert(p.clone(), a.clone());
        }
        self.scopes.push(frame);
        self.depth += 1;
        let r = self.body(&f.body);
        self.scopes.pop();
        self.depth -= 1;
        r?;
        // A function that falls off its end yields nothing, which is what a
        // procedure called for its effect should say.
        Ok(self.returning.take().unwrap_or(Value::Nil))
    }

    /// Run a program, returning the value of the last expression statement.
    pub fn run(&mut self, prog: &[Stmt]) -> Result<Value, String> {
        self.steps = 0;
        // A `return` typed at the prompt has nothing to return from. Cleared
        // here so it cannot sit armed and silently truncate the next block
        // that runs.
        self.returning = None;
        let mut last = Value::Nil;
        for s in prog {
            last = self.stmt(s)?;
        }
        Ok(last)
    }

    fn tick(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > self.budget {
            return Err("execution budget exceeded (infinite loop?)".to_string());
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<Value, String> {
        self.tick()?;
        match s {
            Stmt::Expr(e) => self.expr(e),
            Stmt::Fn(name, params, body) => {
                self.funcs.insert(
                    name.clone(),
                    Func { params: params.clone(), body: body.clone() },
                );
                Ok(Value::Nil)
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(x) => self.expr(x)?,
                    None => Value::Nil,
                };
                self.returning = Some(v);
                Ok(Value::Nil)
            }
            Stmt::If(cond, then, otherwise) => {
                if self.expr(cond)?.truthy() {
                    self.body(then)?;
                } else if let Some(els) = otherwise {
                    self.body(els)?;
                }
                Ok(Value::Nil)
            }
            Stmt::While(cond, body) => {
                while self.expr(cond)?.truthy() {
                    self.tick()?;
                    self.body(body)?;
                    // A `return` inside the loop leaves the function, not
                    // just this iteration. Running the body through `body`
                    // stops the iteration; this stops the loop. Missing this
                    // is silent: the loop simply runs to completion and
                    // whatever it returned is overwritten by whatever comes
                    // after it, so the function answers the wrong thing
                    // rather than failing.
                    if self.returning.is_some() {
                        break;
                    }
                }
                Ok(Value::Nil)
            }
        }
    }

    fn expr(&mut self, e: &Expr) -> Result<Value, String> {
        self.tick()?;
        match e {
            Expr::Int(v) => Ok(Value::Int(*v)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Var(name) => self
                .lookup(name)
                .cloned()
                .ok_or_else(|| format!("undefined variable '{}'", name)),
            Expr::Assign(name, rhs) => {
                let v = self.expr(rhs)?;
                self.assign(name, v.clone());
                Ok(v)
            }
            Expr::Unary(op, inner) => {
                let v = self.expr(inner)?;
                match op {
                    UnOp::Neg => Ok(Value::Int(v.as_int()?.wrapping_neg())),
                    UnOp::Not => Ok(Value::Int(if v.truthy() { 0 } else { 1 })),
                    UnOp::BitNot => Ok(Value::Int(!v.as_int()?)),
                }
            }
            Expr::Bin(op, l, r) => self.binary(*op, l, r),
            Expr::Call(name, args) => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.expr(a)?);
                }
                // User functions shadow builtins deliberately. A program that
                // defines `rect` means its own `rect`, and finding out that a
                // name was reserved is worse than losing access to a builtin
                // the program chose to replace.
                if let Some(f) = self.funcs.get(name).cloned() {
                    return self.call_user(&f, &vals);
                }
                self.builtin(name, &vals)
            }
        }
    }

    fn binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<Value, String> {
        // Short-circuit before evaluating the right side.
        if op == BinOp::LogAnd {
            let a = self.expr(l)?;
            if !a.truthy() {
                return Ok(Value::Int(0));
            }
            return Ok(Value::Int(if self.expr(r)?.truthy() { 1 } else { 0 }));
        }
        if op == BinOp::LogOr {
            let a = self.expr(l)?;
            if a.truthy() {
                return Ok(Value::Int(1));
            }
            return Ok(Value::Int(if self.expr(r)?.truthy() { 1 } else { 0 }));
        }

        let a = self.expr(l)?;
        let b = self.expr(r)?;

        // String concatenation is the one non-numeric case.
        if op == BinOp::Add {
            if let (Value::Str(x), y) = (&a, &b) {
                return Ok(Value::Str(format!("{}{}", x, y.render())));
            }
            if let (x, Value::Str(y)) = (&a, &b) {
                return Ok(Value::Str(format!("{}{}", x.render(), y)));
            }
        }
        if op == BinOp::Eq {
            return Ok(Value::Int(if a == b { 1 } else { 0 }));
        }
        if op == BinOp::Ne {
            return Ok(Value::Int(if a != b { 1 } else { 0 }));
        }

        let x = a.as_int()?;
        let y = b.as_int()?;
        let v = match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => {
                if y == 0 {
                    return Err("division by zero".to_string());
                }
                x.wrapping_div(y)
            }
            BinOp::Rem => {
                if y == 0 {
                    return Err("remainder by zero".to_string());
                }
                x.wrapping_rem(y)
            }
            BinOp::Lt => (x < y) as i64,
            BinOp::Le => (x <= y) as i64,
            BinOp::Gt => (x > y) as i64,
            BinOp::Ge => (x >= y) as i64,
            BinOp::And => x & y,
            BinOp::Or => x | y,
            BinOp::Xor => x ^ y,
            // Shifts are masked to 63 so a silly count cannot panic.
            BinOp::Shl => x.wrapping_shl((y & 63) as u32),
            BinOp::Shr => x.wrapping_shr((y & 63) as u32),
            BinOp::Eq | BinOp::Ne | BinOp::LogAnd | BinOp::LogOr => unreachable!(),
        };
        Ok(Value::Int(v))
    }

    fn arg<'v>(args: &'v [Value], n: usize) -> Result<&'v Value, String> {
        args.get(n).ok_or_else(|| format!("missing argument {}", n + 1))
    }

    fn builtin(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        // One gate, before anything is dispatched.
        if self.caps == Caps::Sandbox {
            if RAW.contains(&name) {
                return Err(format!(
                    "'{}' reaches the machine directly and a stored program may not",
                    name
                ));
            }
            if DRAWS.contains(&name) {
                return Err(format!(
                    "'{}' draws outside any window and a stored program may not",
                    name
                ));
            }
            if name == "write" {
                // Checked here rather than inside the arm, so the arm cannot
                // be rewritten later without noticing the check.
                let path = args.first().map(|v| v.render()).unwrap_or_default();
                if !self.may_write(&path) {
                    return Err(format!("a stored program may not write to {}", path));
                }
            }
            if name == "applet" {
                let line = args.first().map(|v| v.render()).unwrap_or_default();
                let cmd = line.split(' ').next().unwrap_or("");
                match crate::sysbox::applet_mutates(cmd) {
                    None => return Err(format!("no applet '{}'", cmd)),
                    Some(true) => {
                        return Err(format!(
                            "'{}' changes the system and a stored program may not call it",
                            cmd
                        ))
                    }
                    Some(false) => {}
                }
            }
        }

        fn need(args: &[Value], n: usize, name: &str) -> Result<(), String> {
            if args.len() != n {
                Err(format!("{} takes {} argument(s), got {}", name, n, args.len()))
            } else {
                Ok(())
            }
        }
        fn int(args: &[Value], i: usize) -> Result<i64, String> {
            args[i].as_int()
        }
        fn colour(v: i64) -> crate::gfx::Color {
            PALETTE[(v & 0x0F) as usize]
        }

        match name {
            // --- lists ---------------------------------------------------
            //
            // Values, not references. `push` returns a new list rather than
            // changing one in place, so two names can never disagree about
            // what a list contains and nothing has to explain aliasing to
            // whoever -- or whatever -- is writing the program.
            "list" => Ok(Value::List(args.to_vec())),
            "len" => match args.first() {
                Some(Value::List(v)) => Ok(Value::Int(v.len() as i64)),
                Some(Value::Str(t)) => Ok(Value::Int(t.chars().count() as i64)),
                _ => Err("len wants a list or a string".to_string()),
            },
            "get" => {
                let (l, i) = (Self::arg(args, 0)?, Self::arg(args, 1)?.as_int()?);
                match l {
                    Value::List(v) => {
                        // Out of range is nothing, not a fault. A program
                        // walking a list it did not write should be able to
                        // ask past the end without dying.
                        Ok(v.get(i as usize).cloned().unwrap_or(Value::Nil))
                    }
                    Value::Str(t) => Ok(t
                        .chars()
                        .nth(i as usize)
                        .map(|c| Value::Str(c.to_string()))
                        .unwrap_or(Value::Nil)),
                    _ => Err("get wants a list or a string".to_string()),
                }
            }
            "push" => {
                let l = Self::arg(args, 0)?;
                let v = Self::arg(args, 1)?;
                match l {
                    Value::List(items) => {
                        let mut out = items.clone();
                        out.push(v.clone());
                        Ok(Value::List(out))
                    }
                    _ => Err("push wants a list".to_string()),
                }
            }
            "set" => {
                let l = Self::arg(args, 0)?;
                let i = Self::arg(args, 1)?.as_int()?;
                let v = Self::arg(args, 2)?;
                match l {
                    Value::List(items) => {
                        let mut out = items.clone();
                        let i = i as usize;
                        if i >= out.len() {
                            return Err("set past the end of the list".to_string());
                        }
                        out[i] = v.clone();
                        Ok(Value::List(out))
                    }
                    _ => Err("set wants a list".to_string()),
                }
            }
            "print" => {
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        kprint!(" ");
                    }
                    kprint!("{}", a.render());
                }
                Ok(Value::Nil)
            }
            "println" => {
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        kprint!(" ");
                    }
                    kprint!("{}", a.render());
                }
                kprintln!();
                Ok(Value::Nil)
            }
            "hex" => {
                need(args, 1, "hex")?;
                Ok(Value::Str(format!("{:#x}", int(args, 0)?)))
            }
            "cls" => {
                console::with(|c| c.clear());
                Ok(Value::Nil)
            }
            "color" => {
                need(args, 1, "color")?;
                console::set_color((int(args, 0)? & 0x0F) as u8);
                Ok(Value::Nil)
            }
            "ticks" => Ok(Value::Int(crate::dev::lapic::ticks() as i64)),
            "hz" => Ok(Value::Int(crate::dev::lapic::timer_hz() as i64)),
            "tasks" => Ok(Value::Int(crate::task::count() as i64)),
            "heap" => {
                let (used, total) = crate::mem::heap::HEAP.stats();
                kprintln!("  {} B used of {} B", used, total);
                Ok(Value::Int(used as i64))
            }

            // --- namespace ---
            //
            // These are what make a program in /ai/tools a skill rather than
            // a calculator: the ability to look at the namespace and change
            // it. The exposure is exactly the `cat`/`ls`/`write` applets',
            // reached through one more indirection; `run` is classified as
            // mutating, so the read-only grammar never reaches any of this.
            "read" => {
                need(args, 1, "read")?;
                let path = args[0].render();
                match crate::sysbox::read_blob(&path) {
                    Some(bytes) => Ok(Value::Str(String::from_utf8_lossy(&bytes).into_owned())),
                    None => Err(format!("read: no such file '{}'", path)),
                }
            }
            "exists" => {
                need(args, 1, "exists")?;
                let path = args[0].render();
                let yes =
                    crate::sysbox::is_dir(&path) || crate::sysbox::read_blob(&path).is_some();
                Ok(Value::Int(yes as i64))
            }
            "ls" => {
                need(args, 1, "ls")?;
                let path = args[0].render();
                if !crate::sysbox::is_dir(&path) {
                    return Err(format!("ls: '{}' is not a directory", path));
                }
                let names = crate::sysbox::children(&path);
                Ok(Value::Str(names.join("\n")))
            }
            "write" => {
                need(args, 2, "write")?;
                let path = args[0].render();
                let text = args[1].render();
                if crate::sysbox::write_text(&path, &text) {
                    Ok(Value::Int(text.len() as i64))
                } else {
                    Err(format!("write: could not write '{}'", path))
                }
            }
            "applet" => {
                // The program calls the OS. One string in -- "name args" --
                // the applet's captured output out as a string. This is what
                // turns a skill from a calculation into a script: a program
                // that can ls, cat, write and snap its way through the
                // namespace, compose applets, and hand the composed result
                // back to whoever ran it. Trust travels through `run`, which
                // is classified mutating regardless of program text, so the
                // read-only grammar never reaches this.
                need(args, 1, "applet")?;
                let line = args[0].render();
                let (cmd, rest) = match line.split_once(' ') {
                    Some((c, r)) => (c, r),
                    None => (line.as_str(), ""),
                };
                if !crate::sysbox::is_ready() {
                    return Err("applet: namespace not initialised".into());
                }
                if !crate::sysbox::is_applet(cmd) {
                    return Err(format!("applet: '{}' is not an applet", cmd));
                }
                console::begin_capture();
                let ran = crate::sysbox::dispatch(cmd, rest);
                let out = console::end_capture().unwrap_or_default();
                if !ran {
                    return Err(format!("applet: '{}' did not run", cmd));
                }
                Ok(Value::Str(out.trim_end().to_string()))
            }

            // --- graphics ---
            "width" => Ok(Value::Int(gfx::primary().map(|f| f.width()).unwrap_or(0) as i64)),
            "height" => Ok(Value::Int(gfx::primary().map(|f| f.height()).unwrap_or(0) as i64)),
            "pixel" => {
                need(args, 3, "pixel")?;
                if let Some(fb) = gfx::primary() {
                    let raw = fb.encode(colour(int(args, 2)?));
                    fb.put(int(args, 0)? as u32, int(args, 1)? as u32, raw);
                }
                Ok(Value::Nil)
            }
            "rect" => {
                need(args, 5, "rect")?;
                if let Some(fb) = gfx::primary() {
                    fb.rect(
                        int(args, 0)? as u32,
                        int(args, 1)? as u32,
                        int(args, 2)? as u32,
                        int(args, 3)? as u32,
                        colour(int(args, 4)?),
                    );
                }
                Ok(Value::Nil)
            }
            "text" => {
                need(args, 4, "text")?;
                if let Some(fb) = gfx::primary() {
                    fb.draw_text(
                        int(args, 0)? as u32,
                        int(args, 1)? as u32,
                        &args[2].render(),
                        colour(int(args, 3)?),
                        PALETTE[0],
                        2,
                    );
                }
                Ok(Value::Nil)
            }

            // --- raw memory. Ring 0 means these are simply loads and stores. ---
            "peek8" => {
                need(args, 1, "peek8")?;
                let a = int(args, 0)? as u64 as *const u8;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "peek16" => {
                need(args, 1, "peek16")?;
                let a = int(args, 0)? as u64 as *const u16;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "peek32" => {
                need(args, 1, "peek32")?;
                let a = int(args, 0)? as u64 as *const u32;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "peek64" => {
                need(args, 1, "peek64")?;
                let a = int(args, 0)? as u64 as *const u64;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "poke8" => {
                need(args, 2, "poke8")?;
                let a = int(args, 0)? as u64 as *mut u8;
                unsafe { core::ptr::write_volatile(a, int(args, 1)? as u8) };
                Ok(Value::Nil)
            }
            "poke32" => {
                need(args, 2, "poke32")?;
                let a = int(args, 0)? as u64 as *mut u32;
                unsafe { core::ptr::write_volatile(a, int(args, 1)? as u32) };
                Ok(Value::Nil)
            }
            "poke64" => {
                need(args, 2, "poke64")?;
                let a = int(args, 0)? as u64 as *mut u64;
                unsafe { core::ptr::write_volatile(a, int(args, 1)? as u64) };
                Ok(Value::Nil)
            }

            // --- port I/O. The EC lives behind these. ---
            "inb" => {
                need(args, 1, "inb")?;
                Ok(Value::Int(unsafe {
                    crate::cpu::port::inb(int(args, 0)? as u16)
                } as i64))
            }
            "outb" => {
                need(args, 2, "outb")?;
                unsafe { crate::cpu::port::outb(int(args, 0)? as u16, int(args, 1)? as u8) };
                Ok(Value::Nil)
            }
            "inl" => {
                need(args, 1, "inl")?;
                Ok(Value::Int(unsafe {
                    crate::cpu::port::inl(int(args, 0)? as u16)
                } as i64))
            }
            "outl" => {
                need(args, 2, "outl")?;
                unsafe { crate::cpu::port::outl(int(args, 0)? as u16, int(args, 1)? as u32) };
                Ok(Value::Nil)
            }

            other => Err(format!("unknown function '{}'", other)),
        }
    }
}

/// Names the shell offers in `words`.
pub const BUILTINS: &[&str] = &[
    "print", "println", "hex", "cls", "color", "ticks", "hz", "tasks", "heap",
    "read", "exists", "ls", "write", "applet",
    "list", "len", "get", "set", "push",
    "width", "height", "pixel", "rect", "text",
    "peek8", "peek16", "peek32", "peek64", "poke8", "poke32", "poke64",
    "inb", "outb", "inl", "outl",
];
