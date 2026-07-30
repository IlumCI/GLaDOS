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
