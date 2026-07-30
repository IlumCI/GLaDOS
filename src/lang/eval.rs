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
    Nil,
}

impl Value {
    pub fn truthy(&self) -> bool {
        match self {
            Value::Int(v) => *v != 0,
            Value::Str(s) => !s.is_empty(),
            Value::Nil => false,
        }
    }

    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Value::Int(v) => Ok(*v),
            Value::Str(_) => Err("expected a number, found a string".to_string()),
            Value::Nil => Err("expected a number, found nothing".to_string()),
        }
    }

    pub fn render(&self) -> String {
        match self {
            Value::Int(v) => format!("{}", v),
            Value::Str(s) => s.clone(),
            Value::Nil => String::new(),
        }
    }
}

/// Without this, `while (1) {}` at the prompt wedges the shell task forever.
/// There is no Ctrl-C to rescue you: the clock task keeps running because of
/// preemption, but nothing would ever schedule the shell back into a state
/// where it could read a key.
const STEP_BUDGET: u64 = 20_000_000;

pub struct Interp {
    vars: BTreeMap<String, Value>,
    steps: u64,
}

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}

impl Interp {
    pub fn new() -> Self {
        Self { vars: BTreeMap::new(), steps: 0 }
    }

    pub fn var_count(&self) -> usize {
        self.vars.len()
    }

    pub fn vars(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.vars.iter()
    }

    /// Run a program, returning the value of the last expression statement.
    pub fn run(&mut self, prog: &[Stmt]) -> Result<Value, String> {
        self.steps = 0;
        let mut last = Value::Nil;
        for s in prog {
            last = self.stmt(s)?;
        }
        Ok(last)
    }

    fn tick(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > STEP_BUDGET {
            return Err("execution budget exceeded (infinite loop?)".to_string());
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<Value, String> {
        self.tick()?;
        match s {
            Stmt::Expr(e) => self.expr(e),
            Stmt::If(cond, then, otherwise) => {
                if self.expr(cond)?.truthy() {
                    for st in then {
                        self.stmt(st)?;
                    }
                } else if let Some(els) = otherwise {
                    for st in els {
                        self.stmt(st)?;
                    }
                }
                Ok(Value::Nil)
            }
            Stmt::While(cond, body) => {
                while self.expr(cond)?.truthy() {
                    self.tick()?;
                    for st in body {
                        self.stmt(st)?;
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
                .vars
                .get(name)
                .cloned()
                .ok_or_else(|| format!("undefined variable '{}'", name)),
            Expr::Assign(name, rhs) => {
                let v = self.expr(rhs)?;
                self.vars.insert(name.clone(), v.clone());
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

    fn builtin(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
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
    "width", "height", "pixel", "rect", "text",
    "peek8", "peek16", "peek32", "peek64", "poke8", "poke32", "poke64",
    "inb", "outb", "inl", "outl",
];
