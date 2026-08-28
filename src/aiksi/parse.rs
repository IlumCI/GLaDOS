//! Recursive-descent parser.
//!
//! One function per precedence level, each calling the next tighter one. That
//! is more code than a precedence-climbing table but it makes the grammar
//! readable top to bottom, and precedence bugs become obvious rather than
//! being buried in a table of numbers.
//!
//! Precedence, loosest first:
//!   =  ||  &&  |  ^  &  == !=  < <= > >=  << >>  + -  * / %  unary  primary

use super::lex::Tok;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    LogAnd,
    LogOr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64),
    Str(String),
    Var(String),
    Unary(UnOp, Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    Assign(String, Box<Expr>),
    /// `h.port`
    Field(Box<Expr>, String),
    /// `h.port = 443`
    ///
    /// The target is an expression rather than a name so that `a.b.c = 1`
    /// parses, and the evaluator rebuilds the chain outward -- records are
    /// values, so there is no shared object to mutate and the only way to
    /// change a nested field is to rebuild every record above it. That is the
    /// same bargain `push` already makes for lists, and it is why neither
    /// needs anything said about aliasing.
    SetField(Box<Expr>, String, Box<Expr>),
}

/// What a value is allowed to be, where somebody has said.
///
/// Optional everywhere. A program that says nothing gets `Any` and behaves
/// exactly as it did before types existed, which matters because every
/// application already written is such a program.
///
/// Checked when a value crosses a boundary somebody annotated -- a call, a
/// return, a record field -- and never inferred. Inference would mean a type
/// system with a solver in it, and the thing worth having here is much
/// smaller: a model that passes a string where a number belongs should get a
/// sentence naming the function and the parameter, instead of `int()` quietly
/// answering 0 four calls later.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Any,
    Int,
    Str,
    List,
    Nil,
    /// A declared record, by name. Resolved when it is checked rather than
    /// when it is parsed, so a function may take a record declared further
    /// down the file.
    Rec(String),
}

impl Type {
    pub fn parse(name: &str) -> Type {
        match name {
            "any" => Type::Any,
            "int" => Type::Int,
            "str" => Type::Str,
            "list" => Type::List,
            "nil" => Type::Nil,
            other => Type::Rec(String::from(other)),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Type::Any => "any",
            Type::Int => "int",
            Type::Str => "str",
            Type::List => "list",
            Type::Nil => "nil",
            Type::Rec(n) => n,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Expr(Expr),
    If(Expr, Vec<Stmt>, Option<Vec<Stmt>>),
    While(Expr, Vec<Stmt>),
    /// `fn name(a, b) { ... }`
    ///
    /// A statement rather than an expression because there are no first-class
    /// functions here and inventing them would mean closures, captured
    /// environments and a garbage collector's worth of questions. A program
    /// that can name a procedure and call it is enough to write an
    /// application, and that is what this is for.
    Fn(String, Vec<(String, Type)>, Type, Vec<Stmt>),
    Return(Option<Expr>),
    /// `rec Host { name: str, port: int }`
    ///
    /// Declaration and constructor in one: the name becomes callable with the
    /// fields in order. A separate `new` keyword would be a second thing to
    /// learn for no gain, and this way the arity of the constructor is the
    /// arity of the declaration by construction rather than by agreement.
    Rec(String, Vec<(String, Type)>),
    /// `use "/lib/text"`
    ///
    /// Evaluates another program into this interpreter, so its functions and
    /// records become available here. Not a namespace: there are no modules to
    /// qualify against, and a prefix would have to be invented, spelled and
    /// then explained. What it is instead is textual inclusion that happens
    /// once -- see `Interp::import`.
    Use(String),
}

pub struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

pub fn parse(toks: Vec<Tok>) -> Result<Vec<Stmt>, String> {
    let mut p = Parser { toks, pos: 0 };
    let mut out = Vec::new();
    while !p.at(&Tok::Eof) {
        out.push(p.statement()?);
    }
    Ok(out)
}

impl Parser {
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::Eof)
    }

    fn at(&self, t: &Tok) -> bool {
        self.peek() == t
    }

    fn bump(&mut self) -> Tok {
        let t = self.peek().clone();
        if self.pos < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.at(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok, what: &str) -> Result<(), String> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(format!("expected {}, found {:?}", what, self.peek()))
        }
    }

    fn keyword(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Ident(s) if s == kw)
    }

    // --- statements ---

    fn statement(&mut self) -> Result<Stmt, String> {
        if self.keyword("if") {
            self.bump();
            self.expect(&Tok::LParen, "'(' after if")?;
            let cond = self.expression()?;
            self.expect(&Tok::RParen, "')'")?;
            let then = self.block()?;
            let mut otherwise = None;
            if self.keyword("else") {
                self.bump();
                otherwise = Some(self.block()?);
            }
            return Ok(Stmt::If(cond, then, otherwise));
        }

        if self.keyword("fn") {
            self.bump();
            let name = match self.peek().clone() {
                Tok::Ident(n) => {
                    self.bump();
                    n
                }
                _ => return Err("a name after 'fn'".to_string()),
            };
            self.expect(&Tok::LParen, "'(' after the function name")?;
            let params = self.fields(&Tok::RParen)?;
            self.expect(&Tok::RParen, "')' after the parameters")?;
            // `fn f(): int { ... }`. Absent means `any`, so every function
            // written before types existed still parses and still means what
            // it meant.
            let ret = if self.eat(&Tok::Colon) {
                self.type_name()?
            } else {
                Type::Any
            };
            let body = self.block()?;
            return Ok(Stmt::Fn(name, params, ret, body));
        }

        if self.keyword("rec") {
            self.bump();
            let name = match self.peek().clone() {
                Tok::Ident(n) => {
                    self.bump();
                    n
                }
                _ => return Err("a name after 'rec'".to_string()),
            };
            self.expect(&Tok::LBrace, "'{' after the record name")?;
            let fields = self.fields(&Tok::RBrace)?;
            self.expect(&Tok::RBrace, "'}' after the fields")?;
            if fields.is_empty() {
                // A record with no fields is a constructor that returns
                // nothing useful and a type nothing can fail to be. Refusing
                // it here means the evaluator never has to describe one.
                return Err(format!("record '{}' has no fields", name));
            }
            return Ok(Stmt::Rec(name, fields));
        }

        if self.keyword("use") {
            self.bump();
            let path = match self.peek().clone() {
                Tok::Str(p) => {
                    self.bump();
                    p
                }
                // A bare word would have to be resolved against a search path
                // that does not exist, and a variable would make what a
                // program imports depend on what it computed -- which is a
                // thing no checker could read off the source.
                _ => return Err("a quoted path after 'use'".to_string()),
            };
            let _ = self.eat(&Tok::Semi);
            return Ok(Stmt::Use(path));
        }

        if self.keyword("return") {
            self.bump();
            // A bare `return` is legal and yields nothing, which is what a
            // procedure called for its effect wants to say.
            if self.eat(&Tok::Semi) || self.at(&Tok::RBrace) || self.at(&Tok::Eof) {
                return Ok(Stmt::Return(None));
            }
            let e = self.expression()?;
            let _ = self.eat(&Tok::Semi);
            return Ok(Stmt::Return(Some(e)));
        }

        if self.keyword("while") {
            self.bump();
            self.expect(&Tok::LParen, "'(' after while")?;
            let cond = self.expression()?;
            self.expect(&Tok::RParen, "')'")?;
            let body = self.block()?;
            return Ok(Stmt::While(cond, body));
        }

        let e = self.expression()?;
        // Semicolons are optional, so a bare expression at the prompt works.
        let _ = self.eat(&Tok::Semi);
        Ok(Stmt::Expr(e))
    }

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(&Tok::LBrace, "'{'")?;
        let mut out = Vec::new();
        while !self.at(&Tok::RBrace) {
            if self.at(&Tok::Eof) {
                return Err("unterminated block, expected '}'".to_string());
            }
            out.push(self.statement()?);
        }
        self.expect(&Tok::RBrace, "'}'")?;
        Ok(out)
    }

    // --- expressions ---

    /// `a, b: int, c: Host` -- used by both `fn` parameters and `rec` fields.
    ///
    /// One function because the two are the same grammar, and because they
    /// drifting apart is how a language ends up with parameters that accept a
    /// record type and fields that do not.
    fn fields(&mut self, end: &Tok) -> Result<Vec<(String, Type)>, String> {
        let mut out = Vec::new();
        if self.at(end) {
            return Ok(out);
        }
        loop {
            let name = match self.peek().clone() {
                Tok::Ident(n) => {
                    self.bump();
                    n
                }
                _ => return Err("a name".to_string()),
            };
            let ty = if self.eat(&Tok::Colon) {
                self.type_name()?
            } else {
                Type::Any
            };
            if out.iter().any(|(n, _): &(String, Type)| n == &name) {
                return Err(format!("'{}' is named twice", name));
            }
            out.push((name, ty));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(out)
    }

    fn type_name(&mut self) -> Result<Type, String> {
        match self.peek().clone() {
            Tok::Ident(n) => {
                self.bump();
                Ok(Type::parse(&n))
            }
            _ => Err("a type after ':'".to_string()),
        }
    }

    fn expression(&mut self) -> Result<Expr, String> {
        self.assignment()
    }

    fn assignment(&mut self) -> Result<Expr, String> {
        let lhs = self.log_or()?;
        if self.at(&Tok::Assign) {
            self.bump();
            // Right associative, so recurse into assignment rather than log_or.
            let rhs = self.assignment()?;
            return match lhs {
                Expr::Var(name) => Ok(Expr::Assign(name, Box::new(rhs))),
                Expr::Field(target, field) => Ok(Expr::SetField(target, field, Box::new(rhs))),
                _ => Err("left of '=' must be a variable or a field".to_string()),
            };
        }
        Ok(lhs)
    }

    fn log_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.log_and()?;
        while self.eat(&Tok::OrOr) {
            let rhs = self.log_and()?;
            lhs = Expr::Bin(BinOp::LogOr, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn log_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.bit_or()?;
        while self.eat(&Tok::AndAnd) {
            let rhs = self.bit_or()?;
            lhs = Expr::Bin(BinOp::LogAnd, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn bit_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.bit_xor()?;
        while self.eat(&Tok::Pipe) {
            let rhs = self.bit_xor()?;
            lhs = Expr::Bin(BinOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn bit_xor(&mut self) -> Result<Expr, String> {
        let mut lhs = self.bit_and()?;
        while self.eat(&Tok::Caret) {
            let rhs = self.bit_and()?;
            lhs = Expr::Bin(BinOp::Xor, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn bit_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.equality()?;
        while self.eat(&Tok::Amp) {
            let rhs = self.equality()?;
            lhs = Expr::Bin(BinOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn equality(&mut self) -> Result<Expr, String> {
        let mut lhs = self.comparison()?;
        loop {
            let op = if self.eat(&Tok::EqEq) {
                BinOp::Eq
            } else if self.eat(&Tok::NotEq) {
                BinOp::Ne
            } else {
                break;
            };
            let rhs = self.comparison()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let mut lhs = self.shift()?;
        loop {
            let op = if self.eat(&Tok::Lt) {
                BinOp::Lt
            } else if self.eat(&Tok::Le) {
                BinOp::Le
            } else if self.eat(&Tok::Gt) {
                BinOp::Gt
            } else if self.eat(&Tok::Ge) {
                BinOp::Ge
            } else {
                break;
            };
            let rhs = self.shift()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn shift(&mut self) -> Result<Expr, String> {
        let mut lhs = self.term()?;
        loop {
            let op = if self.eat(&Tok::Shl) {
                BinOp::Shl
            } else if self.eat(&Tok::Shr) {
                BinOp::Shr
            } else {
                break;
            };
            let rhs = self.term()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut lhs = self.factor()?;
        loop {
            let op = if self.eat(&Tok::Plus) {
                BinOp::Add
            } else if self.eat(&Tok::Minus) {
                BinOp::Sub
            } else {
                break;
            };
            let rhs = self.factor()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn factor(&mut self) -> Result<Expr, String> {
        let mut lhs = self.unary()?;
        loop {
            let op = if self.eat(&Tok::Star) {
                BinOp::Mul
            } else if self.eat(&Tok::Slash) {
                BinOp::Div
            } else if self.eat(&Tok::Percent) {
                BinOp::Rem
            } else {
                break;
            };
            let rhs = self.unary()?;
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat(&Tok::Minus) {
            return Ok(Expr::Unary(UnOp::Neg, Box::new(self.unary()?)));
        }
        if self.eat(&Tok::Not) {
            return Ok(Expr::Unary(UnOp::Not, Box::new(self.unary()?)));
        }
        if self.eat(&Tok::Tilde) {
            return Ok(Expr::Unary(UnOp::BitNot, Box::new(self.unary()?)));
        }
        self.postfix()
    }

    /// `expr.field`, repeated.
    ///
    /// A loop rather than recursion into `primary`, because `a.b.c` is one
    /// expression with two accesses and not an access whose target happens to
    /// be another. Binds tighter than any operator, so `-h.port` negates the
    /// field and `h.port + 1` adds to it.
    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.primary()?;
        while self.eat(&Tok::Dot) {
            match self.peek().clone() {
                Tok::Ident(field) => {
                    self.bump();
                    e = Expr::Field(Box::new(e), field);
                }
                _ => return Err("a field name after '.'".to_string()),
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        match self.bump() {
            Tok::Int(v) => Ok(Expr::Int(v)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::LParen => {
                let e = self.expression()?;
                self.expect(&Tok::RParen, "')'")?;
                Ok(e)
            }
            Tok::Ident(name) => {
                if self.eat(&Tok::LParen) {
                    let mut args = Vec::new();
                    if !self.at(&Tok::RParen) {
                        loop {
                            args.push(self.expression()?);
                            if !self.eat(&Tok::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen, "')' after arguments")?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(format!("unexpected {:?}", other)),
        }
    }
}
