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
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Expr(Expr),
    If(Expr, Vec<Stmt>, Option<Vec<Stmt>>),
    While(Expr, Vec<Stmt>),
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
                _ => Err("left of '=' must be a variable".to_string()),
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
        self.primary()
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
