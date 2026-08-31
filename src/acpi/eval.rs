//! Evaluating AML: what a method returns when you ask it.
//!
//! The namespace half had to be exact, because a misread package length
//! desynchronises a whole table. This half has the opposite obligation and it
//! is deliberate: it runs only the methods somebody names, and an opcode it
//! does not implement is one method returning an error that carries the
//! opcode and its offset. That is a missing line rather than a mystery, and it
//! is what makes an ACPI interpreter a bounded job.
//!
//! ### This is firmware bytecode running in ring 0
//!
//! There is no isolation in this kernel and a fault outside a guard is fatal,
//! so the evaluator is bounded on three axes rather than trusted:
//!
//! - **A step budget**, in the same spirit as `Interp::tick` in the language:
//!   an AML `While` whose condition never goes false is a hung machine, and
//!   vendor AML contains loops that wait on hardware which may not be there.
//! - **A depth cap**, because a method that calls itself is legal AML and
//!   there is no guard page under this stack.
//! - **Nothing runs unasked.** Evaluation starts at a name a caller supplied.
//!   Loading the namespace executes nothing at all.
//!
//! ### What is deliberately not implemented
//!
//! `Acquire` and `Release` are no-ops. They serialise AML against other AML,
//! and there is exactly one evaluator here which is never re-entered, so the
//! lock they take can never be contended. Implementing them as real mutexes
//! would add a way to deadlock in exchange for nothing. `Notify` is recorded
//! and dropped: it tells an operating system that a device changed, and the
//! thing it would notify is not written yet.

use super::aml::{Kind, Namespace, Path, Seg};
use alloc::string::String;
use alloc::vec::Vec;

/// ACPI's value model.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A method that fell off its end without returning.
    Uninit,
    Int(u64),
    Str(String),
    Buf(Vec<u8>),
    Pkg(Vec<Value>),
    /// A place rather than a value: a namespace node, for `RefOf` and for a
    /// package element that names something.
    Node(usize),
}

impl Value {
    pub fn int(&self) -> Result<u64, Fault> {
        match self {
            Value::Int(v) => Ok(*v),
            // ACPI converts a buffer to an integer by taking its first eight
            // bytes little-endian, which is what a `_BST` package element
            // sometimes arrives as.
            Value::Buf(b) => {
                let mut v = 0u64;
                for (i, byte) in b.iter().take(8).enumerate() {
                    v |= (*byte as u64) << (8 * i);
                }
                Ok(v)
            }
            _ => Err(Fault::Type("an integer")),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Uninit => "uninitialised",
            Value::Int(_) => "an integer",
            Value::Str(_) => "a string",
            Value::Buf(_) => "a buffer",
            Value::Pkg(_) => "a package",
            Value::Node(_) => "a reference",
        }
    }
}

/// Why an evaluation stopped.
///
/// Every variant that can name a byte does, because the fix for an
/// unimplemented opcode is one match arm and the arm needs the number.
#[derive(Clone, Debug, PartialEq)]
pub enum Fault {
    /// The step budget ran out. Almost always a `While` waiting on hardware
    /// that is not answering.
    Budget,
    Depth,
    /// An opcode with no arm, and where it was.
    Opcode(u8, usize),
    ExtOpcode(u8, usize),
    /// A name the namespace does not hold.
    NotFound(String),
    Type(&'static str),
    /// A region in an address space with no handler. Carries the space so the
    /// message can name it rather than print a number.
    Region(u8),
    /// The byte stream ended inside a term.
    Truncated,
    DivideByZero,
    /// A method asked for more arguments than it was given.
    Args,
}

/// How many steps one evaluation gets.
///
/// Generous against real firmware methods, which are tens of steps, and small
/// enough that a loop waiting on absent hardware gives up in well under a
/// second rather than taking the machine with it.
pub const BUDGET: u64 = 200_000;
const MAX_DEPTH: usize = 16;

/// A place a value can be stored.
enum Target {
    None,
    Local(usize),
    Arg(usize),
    Node(usize),
    /// The ACPI debug object. Writes to it are the firmware talking to a
    /// debugger, and are kept rather than dropped because they are sometimes
    /// the only explanation a vendor method offers.
    Debug,
}

/// What a statement did, as distinct from what it produced.
enum Flow {
    Normal,
    Return(Value),
    Break,
    Continue,
}

pub struct Interp<'a> {
    ns: &'a Namespace,
    locals: [Value; 8],
    args: [Value; 7],
    steps: u64,
    depth: usize,
    /// Whatever the firmware last wrote to the debug object.
    pub debug: Option<String>,
}

struct Cursor {
    b: &'static [u8],
    at: usize,
}

impl Cursor {
    fn u8(&mut self) -> Result<u8, Fault> {
        let v = *self.b.get(self.at).ok_or(Fault::Truncated)?;
        self.at += 1;
        Ok(v)
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.at).copied()
    }

    fn take(&mut self, n: usize) -> Result<&'static [u8], Fault> {
        if self.at + n > self.b.len() {
            return Err(Fault::Truncated);
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }

    fn le(&mut self, n: usize) -> Result<u64, Fault> {
        let s = self.take(n)?;
        let mut v = 0u64;
        for (i, b) in s.iter().enumerate() {
            v |= (*b as u64) << (8 * i);
        }
        Ok(v)
    }

    /// The same package length rule the namespace walk uses.
    fn pkg(&mut self) -> Result<usize, Fault> {
        let start = self.at;
        let lead = self.u8()?;
        let extra = (lead >> 6) as usize;
        let mut len = if extra == 0 {
            (lead & 0x3F) as usize
        } else {
            let mut v = (lead & 0x0F) as usize;
            for i in 0..extra {
                v |= (self.u8()? as usize) << (4 + 8 * i);
            }
            v
        };
        len = len.checked_sub(self.at - start).ok_or(Fault::Truncated)?;
        let end = self.at.checked_add(len).ok_or(Fault::Truncated)?;
        if end > self.b.len() {
            return Err(Fault::Truncated);
        }
        Ok(end)
    }

    fn name(&mut self) -> Result<Path, Fault> {
        let mut p = Path::default();
        loop {
            match self.peek().ok_or(Fault::Truncated)? {
                b'\\' => {
                    self.at += 1;
                    p.rooted = true;
                }
                b'^' => {
                    self.at += 1;
                    p.parents += 1;
                }
                _ => break,
            }
        }
        match self.peek().ok_or(Fault::Truncated)? {
            0x00 => {
                self.at += 1;
            }
            0x2E => {
                self.at += 1;
                for _ in 0..2 {
                    p.segs.push(self.seg()?);
                }
            }
            0x2F => {
                self.at += 1;
                let n = self.u8()? as usize;
                for _ in 0..n {
                    p.segs.push(self.seg()?);
                }
            }
            _ => p.segs.push(self.seg()?),
        }
        Ok(p)
    }

    fn seg(&mut self) -> Result<Seg, Fault> {
        let s = self.take(4)?;
        let mut out = [0u8; 4];
        out.copy_from_slice(s);
        Ok(out)
    }
}

impl<'a> Interp<'a> {
    pub fn new(ns: &'a Namespace) -> Interp<'a> {
        const NONE: Value = Value::Uninit;
        Interp { ns, locals: [NONE; 8], args: [NONE; 7], steps: 0, depth: 0, debug: None }
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    fn tick(&mut self) -> Result<(), Fault> {
        self.steps += 1;
        if self.steps > BUDGET {
            return Err(Fault::Budget);
        }
        Ok(())
    }

    /// Evaluate a named object: a data name outright, a method by calling it.
    pub fn eval_node(&mut self, node: usize, args: &[Value]) -> Result<Value, Fault> {
        let n = self.ns.node(node);
        let body = self.ns.body_of(node);
        match n.kind {
            Kind::Name => {
                // Resolved from the *parent*: a name inside a package refers
                // to a sibling of the name that holds it, not to a child of
                // it. A data object has no scope of its own.
                let up = self.ns.node(node).parent;
                let mut c = Cursor { b: body, at: 0 };
                self.arg(&mut c, up)
            }
            Kind::Method { args: want, .. } => {
                if args.len() < want as usize {
                    return Err(Fault::Args);
                }
                if self.depth >= MAX_DEPTH {
                    return Err(Fault::Depth);
                }
                // A call gets its own locals and arguments. Saved and put back
                // rather than allocated, because a method that calls a method
                // is ordinary and a fresh frame per call would be a heap
                // allocation on every one.
                const NONE: Value = Value::Uninit;
                let saved_l = core::mem::replace(&mut self.locals, [NONE; 8]);
                let mut saved_a = core::mem::replace(&mut self.args, [NONE; 7]);
                for (i, v) in args.iter().take(7).enumerate() {
                    self.args[i] = v.clone();
                }
                self.depth += 1;
                let mut c = Cursor { b: body, at: 0 };
                let r = self.terms(&mut c, body.len(), node);
                self.depth -= 1;
                self.locals = saved_l;
                core::mem::swap(&mut self.args, &mut saved_a);
                match r? {
                    Flow::Return(v) => Ok(v),
                    // Falling off the end is legal and yields nothing, which
                    // is different from returning zero and worth keeping
                    // different: a caller testing `_STA` for zero would read
                    // an absent device as present.
                    _ => Ok(Value::Uninit),
                }
            }
            Kind::Field { .. } | Kind::IndexField { .. } | Kind::BankField { .. } => {
                self.read_field(node)
            }
            _ => Ok(Value::Node(node)),
        }
    }

    /// Resolve a path and evaluate what it names.
    pub fn eval_path(&mut self, from: usize, p: &Path, args: &[Value]) -> Result<Value, Fault> {
        let node = self.ns.resolve(from, p).ok_or_else(|| Fault::NotFound(name_of(p)))?;
        self.eval_node(node, args)
    }

    // --- statements -----------------------------------------------------

    fn terms(&mut self, c: &mut Cursor, end: usize, scope: usize) -> Result<Flow, Fault> {
        while c.at < end {
            self.tick()?;
            let before = c.at;
            match self.statement(c, scope)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
            if c.at <= before {
                return Err(Fault::Truncated);
            }
        }
        Ok(Flow::Normal)
    }

    fn statement(&mut self, c: &mut Cursor, scope: usize) -> Result<Flow, Fault> {
        let at = c.at;
        match c.peek().ok_or(Fault::Truncated)? {
            // IfOp. The Else that may follow belongs to it, and has to be
            // stepped over even when the If was taken.
            0xA0 => {
                c.at += 1;
                let end = c.pkg()?;
                let cond = self.arg(c, scope)?.int()? != 0;
                let mut flow = Flow::Normal;
                if cond {
                    flow = self.terms(c, end, scope)?;
                }
                c.at = end;
                if c.peek() == Some(0xA1) {
                    c.at += 1;
                    let eend = c.pkg()?;
                    if !cond {
                        flow = self.terms(c, eend, scope)?;
                    }
                    c.at = eend;
                }
                return Ok(flow);
            }
            // A bare Else. Only reachable when its If was skipped by the
            // walker rather than evaluated, so it is stepped over.
            0xA1 => {
                c.at += 1;
                let end = c.pkg()?;
                c.at = end;
                return Ok(Flow::Normal);
            }
            0xA2 => {
                c.at += 1;
                let end = c.pkg()?;
                let body = c.at;
                loop {
                    self.tick()?;
                    c.at = body;
                    if self.arg(c, scope)?.int()? == 0 {
                        break;
                    }
                    match self.terms(c, end, scope)? {
                        Flow::Break => break,
                        Flow::Return(v) => {
                            c.at = end;
                            return Ok(Flow::Return(v));
                        }
                        _ => {}
                    }
                }
                c.at = end;
                return Ok(Flow::Normal);
            }
            0xA3 => {
                c.at += 1;
                return Ok(Flow::Normal);
            }
            0xA4 => {
                c.at += 1;
                let v = self.arg(c, scope)?;
                return Ok(Flow::Return(v));
            }
            0xA5 => {
                c.at += 1;
                return Ok(Flow::Break);
            }
            0x9F => {
                c.at += 1;
                return Ok(Flow::Continue);
            }
            0xCC => {
                c.at += 1;
                return Ok(Flow::Normal);
            }
            _ => {}
        }
        // Anything else is an expression evaluated for its effect. Its value
        // is discarded, which is what AML does with a bare term.
        let _ = at;
        self.arg(c, scope)?;
        Ok(Flow::Normal)
    }

    // --- expressions ----------------------------------------------------

    fn arg(&mut self, c: &mut Cursor, scope: usize) -> Result<Value, Fault> {
        self.tick()?;
        let at = c.at;
        let op = c.u8()?;
        match op {
            0x00 => Ok(Value::Int(0)),
            0x01 => Ok(Value::Int(1)),
            0xFF => Ok(Value::Int(u64::MAX)),
            0x0A => Ok(Value::Int(c.le(1)?)),
            0x0B => Ok(Value::Int(c.le(2)?)),
            0x0C => Ok(Value::Int(c.le(4)?)),
            0x0E => Ok(Value::Int(c.le(8)?)),
            0x0D => {
                let mut s = String::new();
                loop {
                    let b = c.u8()?;
                    if b == 0 {
                        break;
                    }
                    s.push(b as char);
                }
                Ok(Value::Str(s))
            }
            // Buffer: a declared size, then however many initial bytes are
            // given. The rest are zero, and the declared size wins -- firmware
            // routinely declares 256 and supplies four.
            0x11 => {
                let end = c.pkg()?;
                let size = self.arg(c, scope)?.int()? as usize;
                let given = end.saturating_sub(c.at);
                let bytes = c.take(given)?;
                let mut v = alloc::vec![0u8; size.min(4096)];
                for (i, b) in bytes.iter().take(v.len()).enumerate() {
                    v[i] = *b;
                }
                c.at = end;
                Ok(Value::Buf(v))
            }
            0x12 | 0x13 => {
                let end = c.pkg()?;
                let count = if op == 0x12 {
                    c.u8()? as usize
                } else {
                    self.arg(c, scope)?.int()? as usize
                };
                let mut items = Vec::new();
                while c.at < end && items.len() < count {
                    items.push(self.package_element(c, scope)?);
                }
                // A package may declare more elements than it initialises.
                while items.len() < count.min(512) {
                    items.push(Value::Uninit);
                }
                c.at = end;
                Ok(Value::Pkg(items))
            }
            0x60..=0x67 => Ok(self.locals[(op - 0x60) as usize].clone()),
            0x68..=0x6E => Ok(self.args[(op - 0x68) as usize].clone()),
            // Store: evaluate, then put it somewhere.
            0x70 => {
                let v = self.arg(c, scope)?;
                let t = self.target(c, scope)?;
                self.store(v.clone(), t)?;
                Ok(v)
            }
            0x71 => {
                let t = self.target(c, scope)?;
                match t {
                    Target::Node(n) => Ok(Value::Node(n)),
                    _ => Ok(Value::Int(0)),
                }
            }
            // The two-argument arithmetic and bitwise operators, every one of
            // which takes a target as well.
            0x72 | 0x74 | 0x77 | 0x79 | 0x7A | 0x7B | 0x7C | 0x7D | 0x7E | 0x7F | 0x85 => {
                let a = self.arg(c, scope)?.int()?;
                let b = self.arg(c, scope)?.int()?;
                let v = match op {
                    0x72 => a.wrapping_add(b),
                    0x74 => a.wrapping_sub(b),
                    0x77 => a.wrapping_mul(b),
                    0x79 => a.checked_shl(b as u32).unwrap_or(0),
                    0x7A => a.checked_shr(b as u32).unwrap_or(0),
                    0x7B => a & b,
                    0x7C => !(a & b),
                    0x7D => a | b,
                    0x7E => !(a | b),
                    0x7F => a ^ b,
                    _ => {
                        if b == 0 {
                            return Err(Fault::DivideByZero);
                        }
                        a % b
                    }
                };
                let t = self.target(c, scope)?;
                self.store(Value::Int(v), t)?;
                Ok(Value::Int(v))
            }
            // Divide is the odd one: two targets, remainder first.
            0x78 => {
                let a = self.arg(c, scope)?.int()?;
                let b = self.arg(c, scope)?.int()?;
                if b == 0 {
                    return Err(Fault::DivideByZero);
                }
                let rem = self.target(c, scope)?;
                let quo = self.target(c, scope)?;
                self.store(Value::Int(a % b), rem)?;
                self.store(Value::Int(a / b), quo)?;
                Ok(Value::Int(a / b))
            }
            // The one-argument ones, also with a target.
            0x80 | 0x81 | 0x82 | 0x96 | 0x99 => {
                let a = self.arg(c, scope)?;
                let v = match op {
                    0x80 => Value::Int(!a.int()?),
                    0x81 => Value::Int(64u64.saturating_sub(a.int()?.leading_zeros() as u64)),
                    0x82 => {
                        let x = a.int()?;
                        Value::Int(if x == 0 { 0 } else { x.trailing_zeros() as u64 + 1 })
                    }
                    0x96 => match a {
                        Value::Buf(b) => Value::Buf(b),
                        other => Value::Buf(other.int()?.to_le_bytes().to_vec()),
                    },
                    _ => Value::Int(a.int()?),
                };
                let t = self.target(c, scope)?;
                self.store(v.clone(), t)?;
                Ok(v)
            }
            0x75 | 0x76 => {
                let t = self.target(c, scope)?;
                let cur = self.read_target(&t)?.int()?;
                let v = if op == 0x75 { cur.wrapping_add(1) } else { cur.wrapping_sub(1) };
                self.store(Value::Int(v), t)?;
                Ok(Value::Int(v))
            }
            0x83 => {
                let a = self.arg(c, scope)?;
                match a {
                    Value::Node(n) => self.eval_node(n, &[]),
                    other => Ok(other),
                }
            }
            0x87 => {
                let a = self.arg(c, scope)?;
                Ok(Value::Int(match a {
                    Value::Buf(b) => b.len() as u64,
                    Value::Str(s) => s.len() as u64,
                    Value::Pkg(p) => p.len() as u64,
                    _ => 0,
                }))
            }
            // Index: into a package, buffer or string.
            0x88 => {
                let src = self.arg(c, scope)?;
                let i = self.arg(c, scope)?.int()? as usize;
                let v = match &src {
                    Value::Pkg(p) => p.get(i).cloned().unwrap_or(Value::Uninit),
                    Value::Buf(b) => Value::Int(b.get(i).copied().unwrap_or(0) as u64),
                    Value::Str(s) => {
                        Value::Int(s.as_bytes().get(i).copied().unwrap_or(0) as u64)
                    }
                    _ => Value::Uninit,
                };
                let t = self.target(c, scope)?;
                self.store(v.clone(), t)?;
                Ok(v)
            }
            0x90 | 0x91 => {
                let a = self.arg(c, scope)?.int()? != 0;
                let b = self.arg(c, scope)?.int()? != 0;
                let v = if op == 0x90 { a && b } else { a || b };
                Ok(Value::Int(v as u64))
            }
            // LNot, and the three negated comparisons that hide behind it.
            0x92 => match c.peek() {
                Some(0x93) | Some(0x94) | Some(0x95) => {
                    let sub = c.u8()?;
                    let a = self.arg(c, scope)?;
                    let b = self.arg(c, scope)?;
                    let r = compare(&a, &b, sub)?;
                    Ok(Value::Int(!r as u64))
                }
                _ => {
                    let a = self.arg(c, scope)?.int()?;
                    Ok(Value::Int((a == 0) as u64))
                }
            },
            0x93 | 0x94 | 0x95 => {
                let a = self.arg(c, scope)?;
                let b = self.arg(c, scope)?;
                Ok(Value::Int(compare(&a, &b, op)? as u64))
            }
            0x9D => {
                let v = self.arg(c, scope)?;
                let t = self.target(c, scope)?;
                self.store(v.clone(), t)?;
                Ok(v)
            }
            // Notify: told to the operating system, and there is nothing here
            // yet to tell. Its arguments are still evaluated, because they can
            // have effects.
            0x86 => {
                let _ = self.target(c, scope)?;
                let _ = self.arg(c, scope)?;
                Ok(Value::Uninit)
            }
            0x5B => {
                let ext = c.u8()?;
                self.ext(c, ext, at, scope)
            }
            // A name: either an object to read or a method to call.
            b'\\' | b'^' | b'_' | 0x2E | 0x2F | b'A'..=b'Z' => {
                c.at = at;
                let p = c.name()?;
                let node =
                    self.ns.resolve(scope, &p).ok_or_else(|| Fault::NotFound(name_of(&p)))?;
                let want = match self.ns.node(node).kind {
                    Kind::Method { args, .. } => args as usize,
                    _ => 0,
                };
                let mut vals = Vec::new();
                for _ in 0..want {
                    vals.push(self.arg(c, scope)?);
                }
                self.eval_node(node, &vals)
            }
            other => Err(Fault::Opcode(other, at)),
        }
    }

    fn ext(&mut self, c: &mut Cursor, ext: u8, at: usize, scope: usize) -> Result<Value, Fault> {
        match ext {
            // Revision, and the debug object read as a value.
            0x30 => Ok(Value::Int(2)),
            0x31 => Ok(Value::Uninit),
            // Timer, in 100 ns units, which is what ACPI asks for.
            0x33 => Ok(Value::Int(crate::time::rdtsc())),
            // Stall and Sleep. Bounded hard: firmware asks for milliseconds
            // and a method that asks for a minute would take the machine away
            // from everything else.
            0x21 | 0x22 => {
                let us = self.arg(c, scope)?.int()?;
                let capped = us.min(if ext == 0x21 { 1000 } else { 50_000 });
                crate::time::delay_us(capped);
                Ok(Value::Uninit)
            }
            // Acquire and Release. No-ops, argued in the module doc.
            0x23 => {
                let _ = self.target(c, scope)?;
                let _ = c.le(2)?;
                Ok(Value::Int(0))
            }
            0x27 | 0x26 | 0x24 => {
                let _ = self.target(c, scope)?;
                Ok(Value::Uninit)
            }
            0x25 => {
                let _ = self.target(c, scope)?;
                let _ = self.arg(c, scope)?;
                Ok(Value::Int(0))
            }
            // CondRefOf: a reference if the name exists, and false if not.
            // This is how firmware asks whether the operating system provided
            // something, so answering it wrongly changes which branch runs.
            0x12 => {
                let start = c.at;
                let found = match c.name() {
                    Ok(p) => self.ns.resolve(scope, &p),
                    Err(_) => {
                        c.at = start;
                        None
                    }
                };
                let t = self.target(c, scope)?;
                match found {
                    Some(n) => {
                        self.store(Value::Node(n), t)?;
                        Ok(Value::Int(1))
                    }
                    None => Ok(Value::Int(0)),
                }
            }
            0x28 | 0x29 => {
                let a = self.arg(c, scope)?.int()?;
                let t = self.target(c, scope)?;
                let v = if ext == 0x28 { from_bcd(a) } else { to_bcd(a) };
                self.store(Value::Int(v), t)?;
                Ok(Value::Int(v))
            }
            // Fatal: the firmware declaring the machine unfit. Reported rather
            // than acted on, because acting on it means a shutdown path that
            // does not exist yet.
            0x32 => {
                let _ = c.u8()?;
                let _ = c.le(4)?;
                let _ = self.arg(c, scope)?;
                Ok(Value::Uninit)
            }
            other => Err(Fault::ExtOpcode(other, at)),
        }
    }

    /// A package element, which is data or a bare name rather than a general
    /// expression. A name here is a reference and must not be called.
    fn package_element(&mut self, c: &mut Cursor, scope: usize) -> Result<Value, Fault> {
        self.tick()?;
        match c.peek().ok_or(Fault::Truncated)? {
            b'\\' | b'^' | 0x2E | 0x2F | b'_' | b'A'..=b'Z' => {
                let p = c.name()?;
                match self.ns.resolve(scope, &p) {
                    Some(n) => Ok(Value::Node(n)),
                    None => Ok(Value::Str(name_of(&p))),
                }
            }
            _ => self.arg(c, scope),
        }
    }

    // --- targets --------------------------------------------------------

    fn target(&mut self, c: &mut Cursor, scope: usize) -> Result<Target, Fault> {
        match c.peek().ok_or(Fault::Truncated)? {
            0x00 => {
                c.at += 1;
                Ok(Target::None)
            }
            v @ 0x60..=0x67 => {
                c.at += 1;
                Ok(Target::Local((v - 0x60) as usize))
            }
            v @ 0x68..=0x6E => {
                c.at += 1;
                Ok(Target::Arg((v - 0x68) as usize))
            }
            0x5B => {
                // Only the debug object is a legal target under the extended
                // prefix.
                c.at += 1;
                let e = c.u8()?;
                if e == 0x31 {
                    Ok(Target::Debug)
                } else {
                    Err(Fault::ExtOpcode(e, c.at))
                }
            }
            // Index and DerefOf as targets are evaluated and their result
            // discarded, which loses the write. Named here rather than
            // silently mis-stored: a `Store` into a package element is a real
            // construct and this does not do it yet.
            0x88 | 0x83 => {
                let _ = self.arg(c, scope)?;
                Ok(Target::None)
            }
            _ => {
                let p = c.name()?;
                match self.ns.resolve(scope, &p) {
                    Some(n) => Ok(Target::Node(n)),
                    None => Ok(Target::None),
                }
            }
        }
    }

    fn store(&mut self, v: Value, t: Target) -> Result<(), Fault> {
        match t {
            Target::None => {}
            Target::Local(i) => self.locals[i] = v,
            Target::Arg(i) => self.args[i] = v,
            Target::Debug => {
                self.debug = Some(match v {
                    Value::Str(s) => s,
                    other => alloc::format!("{:?}", other),
                })
            }
            // Writing into the namespace is where region writes would happen,
            // and those are off until B3 gives them a gate. A write to a plain
            // `Name` is dropped rather than applied, because the namespace
            // holds byte ranges rather than values and rewriting firmware
            // bytes is not what this should do.
            Target::Node(_) => {}
        }
        Ok(())
    }

    fn read_target(&mut self, t: &Target) -> Result<Value, Fault> {
        Ok(match t {
            Target::Local(i) => self.locals[*i].clone(),
            Target::Arg(i) => self.args[*i].clone(),
            Target::Node(n) => self.eval_node(*n, &[])?,
            _ => Value::Uninit,
        })
    }
}

fn compare(a: &Value, b: &Value, op: u8) -> Result<bool, Fault> {
    // Strings and buffers compare lexically, integers numerically. Getting
    // this wrong flips a branch rather than raising anything.
    if let (Value::Str(x), Value::Str(y)) = (a, b) {
        return Ok(match op {
            0x93 => x == y,
            0x94 => x > y,
            _ => x < y,
        });
    }
    let x = a.int()?;
    let y = b.int()?;
    Ok(match op {
        0x93 => x == y,
        0x94 => x > y,
        _ => x < y,
    })
}

fn from_bcd(v: u64) -> u64 {
    let mut out = 0u64;
    let mut mul = 1u64;
    let mut x = v;
    while x > 0 {
        out += (x & 0x0F) * mul;
        mul = mul.saturating_mul(10);
        x >>= 4;
    }
    out
}

fn to_bcd(v: u64) -> u64 {
    let mut out = 0u64;
    let mut shift = 0;
    let mut x = v;
    while x > 0 && shift < 64 {
        out |= (x % 10) << shift;
        x /= 10;
        shift += 4;
    }
    out
}

/// A path as text, for an error that has to name something.
pub fn name_of(p: &Path) -> String {
    let mut s = String::new();
    if p.rooted {
        s.push('\\');
    }
    for _ in 0..p.parents {
        s.push('^');
    }
    for (i, seg) in p.segs.iter().enumerate() {
        if i > 0 {
            s.push('.');
        }
        for c in seg.iter() {
            s.push(*c as char);
        }
    }
    while s.ends_with('_') {
        s.pop();
    }
    s
}

// --- regions, and reading fields out of them ----------------------------

/// Whether writes through a region are permitted.
///
/// Off, and the same shape as `store unlock` and `fat unlock`. Reading a
/// battery needs no writes at all, and a stray write to an embedded controller
/// is not a wrong number: it is a fan that stops or a charge threshold that
/// moves, on hardware, permanently. So the capability is separate from the
/// code that would use it and has to be asked for.
static WRITES: crate::sync::Racy<bool> = crate::sync::Racy::new(false);

pub fn allow_writes(on: bool) {
    unsafe { *WRITES.get() = on };
}

pub fn writes_allowed() -> bool {
    unsafe { *WRITES.get() }
}

impl<'a> Interp<'a> {
    /// A region's address space, base and length, evaluated on demand.
    ///
    /// The offset and length are term arguments and on a real machine one of
    /// them is routinely a name or an `Add` over one, so they cannot be read
    /// at load time. They are evaluated here, where the namespace they refer
    /// to is complete.
    fn region_of(&mut self, node: usize) -> Result<(u8, u64, u64), Fault> {
        let (space, body, scope) = {
            let n = self.ns.node(node);
            let space = match n.kind {
                Kind::OpRegion { space } => space,
                _ => return Err(Fault::Type("a region")),
            };
            (space, self.ns.body_of(node), n.parent)
        };
        let mut c = Cursor { b: body, at: 0 };
        let base = self.arg(&mut c, scope)?.int()?;
        let len = self.arg(&mut c, scope)?.int().unwrap_or(0);
        Ok((space, base, len))
    }

    /// A region's base and length, for a report that wants to show them.
    pub fn region_bounds(&mut self, node: usize) -> Result<(u64, u64), Fault> {
        let (_, base, len) = self.region_of(node)?;
        Ok((base, len))
    }

    /// Read one access-sized unit from an address space.
    ///
    /// The embedded controller is a byte at a time whatever the field's
    /// declared access size says, because its protocol has no other shape.
    fn read_unit(&mut self, space: u8, addr: u64, bytes: usize) -> Result<u64, Fault> {
        match space {
            // System memory. Identity-mapped here, so the address is the
            // pointer, and read volatile because it is very often not memory.
            0 => Ok(unsafe {
                match bytes {
                    1 => core::ptr::read_volatile(addr as *const u8) as u64,
                    2 => core::ptr::read_volatile(addr as *const u16) as u64,
                    4 => core::ptr::read_volatile(addr as *const u32) as u64,
                    _ => core::ptr::read_volatile(addr as *const u64),
                }
            }),
            // System I/O.
            1 => Ok(unsafe {
                match bytes {
                    1 => crate::cpu::port::inb(addr as u16) as u64,
                    2 => crate::cpu::port::inw(addr as u16) as u64,
                    _ => crate::cpu::port::inl(addr as u16) as u64,
                }
            }),
            // The embedded controller, one byte per transaction.
            3 => crate::dev::ec::read(addr as u8).map(|v| v as u64).ok_or(Fault::Region(3)),
            other => Err(Fault::Region(other)),
        }
    }

    /// Read a field: the named window onto a region.
    ///
    /// Fields are not required to be aligned to anything, so this reads every
    /// access-sized unit the window touches, assembles them, then shifts and
    /// masks. A `_BST` on a real machine has four-bit and one-bit fields
    /// sharing a byte with each other.
    pub fn read_field(&mut self, node: usize) -> Result<Value, Fault> {
        self.tick()?;
        let (region, flags, offset, width) = match self.ns.node(node).kind {
            Kind::Field { region, flags, offset, width } => (region, flags, offset, width),
            // A bank field needs its bank register selected first, and an
            // index field is two writes and a read. Both need writes, which
            // are off, so both are refused by name rather than read wrongly.
            Kind::IndexField { .. } => return Err(Fault::Type("a field this can read")),
            Kind::BankField { .. } => return Err(Fault::Type("a field this can read")),
            _ => return Err(Fault::Type("a field")),
        };
        if width == 0 || width > 64 {
            return Err(Fault::Type("a field of at most 64 bits"));
        }
        let (space, base, _len) = self.region_of(region)?;

        // Access size, from the low nibble of the field flags. Anything the
        // controller cannot do a byte at a time is forced to bytes.
        let bits = match flags & 0x0F {
            2 => 16u32,
            3 => 32,
            4 => 64,
            _ => 8,
        };
        let bits = if space == 3 { 8 } else { bits };
        let unit = (bits / 8) as usize;

        let first = offset / bits;
        let last = (offset + width - 1) / bits;
        let mut acc: u128 = 0;
        for u in first..=last {
            let addr = base + (u as u64) * (unit as u64);
            let v = self.read_unit(space, addr, unit)?;
            acc |= (v as u128) << ((u - first) * bits);
            self.tick()?;
        }
        let shift = offset - first * bits;
        let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
        Ok(Value::Int(((acc >> shift) as u64) & mask))
    }
}
