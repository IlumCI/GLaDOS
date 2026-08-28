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

/// What a builtin touches.
///
/// Every builtin declares one, and `BUILTINS` is the only way to reach the
/// dispatch, so a builtin that is added without saying what it touches is not
/// callable at all. That inversion is the whole point.
///
/// It replaced two denylists. Those were correct for eleven raw builtins and
/// three drawing ones, and they stopped being correct the moment the language
/// was wired to the rest of the kernel: a denylist grants by default, so every
/// builtin anyone forgot to list -- sockets included -- would have been
/// reachable from a program the machine wrote for itself. The old comment said
/// as much about per-arm checks and the same argument finished the list off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Touch {
    /// Values in, values out. Nothing outside the interpreter is read.
    Pure,
    /// Reads state that is not secret: the clock, the heap figure, the
    /// namespace, which interfaces exist.
    Read,
    /// Writes, and a sandboxed program may only write inside its own subtree.
    /// `may_write` is what enforces that; this only says a write happens.
    Write,
    /// Talks to the network.
    Net,
    /// Runs the model.
    Model,
    /// Paints outside any window.
    Draw,
    /// Reaches the machine directly, or changes the system.
    Raw,
}

impl Touch {
    /// Whether a stored program may do this unasked.
    ///
    /// Three classes and not a grant matrix. `manifest::Manifest` carries one
    /// bit for the same reason it gives: an operator approving a request has to
    /// hold the whole of it in their head, and "may write outside itself but
    /// not open sockets" is a sentence nobody can check against a program.
    /// Sandboxed or trusted is a question with an answer.
    fn sandboxable(self) -> bool {
        matches!(self, Touch::Pure | Touch::Read | Touch::Write)
    }

    fn why(self) -> &'static str {
        match self {
            Touch::Net => "talks to the network",
            Touch::Model => "runs the model",
            Touch::Draw => "draws outside any window",
            Touch::Raw => "reaches the machine directly",
            _ => "changes the system",
        }
    }
}

/// Every builtin, what it touches, and how many arguments it takes.
///
/// The single source of truth. `builtin` refuses anything absent from here
/// before it dispatches, so an arm added to the match without a row is dead
/// code and a row without an arm is a boot selftest failure. Neither can be
/// half-done, which is the property the old denylists could not offer.
///
/// Arity is `(min, max)`; `usize::MAX` for max means variadic.
pub const BUILTINS: &[(&str, Touch, usize, usize)] = &[
    // --- values -----------------------------------------------------------
    ("here", Touch::Pure, 0, 0),
    ("int", Touch::Pure, 1, 1),
    ("hex", Touch::Pure, 1, 1),
    ("list", Touch::Pure, 0, usize::MAX),
    ("len", Touch::Pure, 1, 1),
    ("get", Touch::Pure, 2, 2),
    ("push", Touch::Pure, 2, 2),
    ("set", Touch::Pure, 3, 3),
    // Transient: the console scrolls and nothing outlives the call.
    ("print", Touch::Pure, 0, usize::MAX),
    ("println", Touch::Pure, 0, usize::MAX),

    // --- reading the machine ----------------------------------------------
    ("ticks", Touch::Read, 0, 0),
    ("hz", Touch::Read, 0, 0),
    ("tasks", Touch::Read, 0, 0),
    ("heap", Touch::Read, 0, 0),
    ("width", Touch::Read, 0, 0),
    ("height", Touch::Read, 0, 0),
    ("read", Touch::Read, 1, 1),
    ("exists", Touch::Read, 1, 1),
    ("ls", Touch::Read, 1, 1),
    // Read, and then narrowed again per call by `applet_mutates`: the applet
    // table already answers "does this change anything", and asking it is
    // exact where a second list here would drift from it.
    ("applet", Touch::Read, 1, 1),

    // --- writing ----------------------------------------------------------
    // Sandboxed writes are confined by `may_write`, which resolves the path
    // first. The class says a write happens; the jail says where.
    ("write", Touch::Write, 2, 2),

    // --- the operator's terminal ------------------------------------------
    // Not sandboxable, and this is a tightening. The colour outlives the call
    // and the clear takes the operator's scrollback, so an application
    // repainting its rows could wipe the terminal underneath it. Nothing
    // stored uses either; the prompt runs with Operator caps and keeps both.
    ("cls", Touch::Raw, 0, 0),
    ("color", Touch::Raw, 1, 1),

    // --- drawing ----------------------------------------------------------
    ("pixel", Touch::Draw, 3, 3),
    ("rect", Touch::Draw, 5, 5),
    ("text", Touch::Draw, 4, 4),

    // --- text -------------------------------------------------------------
    //
    // A systems language whose only string operation is `+` cannot parse a
    // header, split a path or read a config file, so every program that needed
    // one grew a bad version of it in Aiksi. These are `str` and `char` in the
    // Rust underneath, named for what they do rather than for the method.
    ("upper", Touch::Pure, 1, 1),
    ("lower", Touch::Pure, 1, 1),
    ("trim", Touch::Pure, 1, 1),
    ("split", Touch::Pure, 2, 2),
    ("join", Touch::Pure, 2, 2),
    ("substr", Touch::Pure, 3, 3),
    ("find", Touch::Pure, 2, 2),
    ("replace", Touch::Pure, 3, 3),
    ("starts", Touch::Pure, 2, 2),
    ("ends", Touch::Pure, 2, 2),
    ("contains", Touch::Pure, 2, 2),
    ("chr", Touch::Pure, 1, 1),
    ("ord", Touch::Pure, 1, 1),
    ("repeat", Touch::Pure, 2, 2),
    ("pad", Touch::Pure, 2, 2),
    ("hexenc", Touch::Pure, 1, 1),
    ("hexdec", Touch::Pure, 1, 1),

    // --- arithmetic --------------------------------------------------------
    ("abs", Touch::Pure, 1, 1),
    ("min", Touch::Pure, 2, 2),
    ("max", Touch::Pure, 2, 2),
    ("clamp", Touch::Pure, 3, 3),
    ("sqrt", Touch::Pure, 1, 1),
    ("pow", Touch::Pure, 2, 2),

    // --- lists, beyond building one ----------------------------------------
    ("sort", Touch::Pure, 1, 1),
    ("reverse", Touch::Pure, 1, 1),
    ("slice", Touch::Pure, 3, 3),
    ("index", Touch::Pure, 2, 2),
    ("remove", Touch::Pure, 2, 2),
    ("range", Touch::Pure, 2, 2),

    // --- crate::dev::rtc, crate::time, crate::dev::lapic --------------------
    ("rtc_now", Touch::Read, 0, 0),
    ("rtc_unix", Touch::Read, 0, 0),
    ("uptime", Touch::Read, 0, 0),
    ("tsc", Touch::Read, 0, 0),
    ("tsc_mhz", Touch::Read, 0, 0),

    // --- crate::task -------------------------------------------------------
    ("task_count", Touch::Read, 0, 0),
    ("task_current", Touch::Read, 0, 0),
    ("task_switches", Touch::Read, 0, 0),
    // Yielding is not a read, but it is not a way to reach anything either:
    // the scheduler preempts at 100 Hz regardless, so this only gives up a
    // slice early. A long loop in a repaint path is bounded by the step
    // budget, not by whether it was polite.
    ("task_yield", Touch::Read, 0, 0),

    // --- crate::mem --------------------------------------------------------
    ("mem_used", Touch::Read, 0, 0),
    ("mem_total", Touch::Read, 0, 0),

    // --- crate::dev::pci ---------------------------------------------------
    ("pci_list", Touch::Read, 0, 0),

    // --- crate::net --------------------------------------------------------
    //
    // Status only, so far. Reading which interfaces exist and what address
    // this machine has tells a program about itself; it does not put a packet
    // on the wire, which is why these are Read and the sockets are not.
    ("net_ready", Touch::Read, 0, 0),
    ("net_ifaces", Touch::Read, 0, 0),
    ("net_ip", Touch::Read, 0, 0),
    ("net_gateway", Touch::Read, 0, 0),
    ("net_dns", Touch::Read, 0, 0),

    // --- crate::sysbox, beyond read and write -------------------------------
    ("hash_of", Touch::Read, 1, 1),
    ("size", Touch::Read, 1, 1),
    ("is_dir", Touch::Read, 1, 1),
    // Removal is a write, and confined by the same jail: `may_write` resolves
    // the path before comparing, so `../..` is not a way out of it.
    ("rm", Touch::Write, 1, 1),

    // --- the machine itself -----------------------------------------------
    ("peek8", Touch::Raw, 1, 1),
    ("peek16", Touch::Raw, 1, 1),
    ("peek32", Touch::Raw, 1, 1),
    ("peek64", Touch::Raw, 1, 1),
    ("poke8", Touch::Raw, 2, 2),
    ("poke32", Touch::Raw, 2, 2),
    ("poke64", Touch::Raw, 2, 2),
    ("inb", Touch::Raw, 1, 1),
    ("outb", Touch::Raw, 2, 2),
    ("inl", Touch::Raw, 1, 1),
    ("outl", Touch::Raw, 2, 2),
];

/// What a builtin touches, or `None` if there is no such builtin.
pub fn touch_of(name: &str) -> Option<Touch> {
    BUILTINS.iter().find(|(n, ..)| *n == name).map(|(_, t, ..)| *t)
}

/// Is this a builtin at all?
pub fn is_builtin(name: &str) -> bool {
    touch_of(name).is_some()
}

/// Every builtin a program with these capabilities may call.
pub fn available(caps: Caps) -> Vec<&'static str> {
    BUILTINS
        .iter()
        .filter(|(_, t, ..)| caps == Caps::Operator || t.sandboxable())
        .map(|(n, ..)| *n)
        .collect()
}

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
    /// `may_write` for the kernel arms, which live in another module and must
    /// not each grow their own idea of where the jail is.
    pub fn may_write_pub(&self, path: &str) -> bool {
        self.may_write(path)
    }

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
        // One gate, before anything is dispatched, and it opens only for a
        // builtin the table names. An unknown name cannot fall through to the
        // match: the match is unreachable without a row, so adding an arm and
        // forgetting the row produces dead code rather than an ungated builtin.
        let Some(touch) = touch_of(name) else {
            return Err(format!("no builtin called '{}'", name));
        };
        let (lo, hi) = BUILTINS
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, _, lo, hi)| (*lo, *hi))
            .unwrap_or((0, usize::MAX));
        if args.len() < lo || args.len() > hi {
            return Err(if lo == hi {
                format!("{} takes {} argument(s), got {}", name, lo, args.len())
            } else if hi == usize::MAX {
                format!("{} takes at least {}, got {}", name, lo, args.len())
            } else {
                format!("{} takes {} to {} arguments, got {}", name, lo, hi, args.len())
            });
        }
        if self.caps == Caps::Sandbox {
            if !touch.sandboxable() {
                return Err(format!(
                    "'{}' {} and a stored program may not -- 'app trust {}' if that is what you want",
                    name,
                    touch.why(),
                    self.jail
                        .as_deref()
                        .and_then(|j| j.rsplit('/').next())
                        .unwrap_or("<app>")
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
            // Where this program is allowed to keep things.
            //
            // A stored program cannot hardcode its own path: the same files
            // live at `/draft/<name>` while being written and `/app/<name>`
            // once adopted, and a literal path would be outside the jail on
            // one side of that move. Asking makes a program location
            // independent, and the jail is the authority on the answer rather
            // than a convention the program has to be told.
            "here" => Ok(Value::Str(self.jail.clone().unwrap_or_default())),
            // Text to number.
            //
            // `read` answers with what is in the file, which is text, and `+`
            // on text concatenates. A program keeping a count in a file and
            // adding one to it gets "01" and then "011", with nothing failing
            // anywhere -- the first skeleton written here did exactly that.
            // Anything unparseable is zero rather than an error: a counter
            // whose file has been emptied should start again, not refuse to
            // draw.
            "int" => {
                let t = Self::arg(args, 0)?.render();
                let t = t.trim();
                let (neg, digits) = match t.strip_prefix('-') {
                    Some(rest) => (true, rest),
                    None => (false, t),
                };
                let mut n: i64 = 0;
                let mut any = false;
                for c in digits.chars() {
                    let Some(d) = c.to_digit(10) else { break };
                    n = n.saturating_mul(10).saturating_add(d as i64);
                    any = true;
                }
                Ok(Value::Int(if any && neg { -n } else if any { n } else { 0 }))
            }
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

            // Everything that reaches a kernel subsystem. The gate and the
            // arity check have already run, so this cannot be a way around
            // either: a name only gets here by being in the table.
            other => super::kernel::call(self, other, args),
        }
    }
}

// The list the shell offers used to live here as a second array of names,
// hand-kept beside the match. It is gone: `BUILTINS` above is the one list,
// and `words` reads it. A name that appears in one place cannot disagree with
// itself.
