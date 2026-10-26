//! Pseudo-terminals: `/dev/ptmx` and `/dev/pts/N`.
//!
//! **A terminal emulator is a program holding the master end, and a shell is
//! a program reading the slave end as if it were a keyboard and a screen.**
//! Between them sits the line discipline, which is the kernel's part and the
//! reason this is not just a pipe: in canonical mode it collects a line,
//! echoes it and applies erase and kill before the reader sees anything; it
//! turns Ctrl-C into SIGINT for the foreground process group; it turns `\n`
//! into `\r\n` on the way out. A shell that edits its own line (fish, bash
//! with readline) switches canonical mode and echo off and does all of it
//! itself, so both have to work.
//!
//! What is kept is what programs read back: the 36-byte kernel `termios`
//! (`TCGETS`), the window size (`TIOCGWINSZ`), the foreground process group.
//! Baud rates and parity are stored and mean nothing, since no wire exists.

use alloc::collections::VecDeque;
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::sync::Racy;

/// `struct termios` as the kernel's TCGETS fills it: four flag words, the
/// line discipline, and nineteen control characters.
pub const TERMIOS_LEN: usize = 36;

// c_iflag
const ICRNL: u32 = 0o400;
const IXON: u32 = 0o2000;
// c_oflag
const OPOST: u32 = 0o1;
const ONLCR: u32 = 0o4;
// c_lflag
const ISIG: u32 = 0o1;
const ICANON: u32 = 0o2;
const ECHO: u32 = 0o10;
const ECHOE: u32 = 0o20;
const ECHOK: u32 = 0o40;
const ECHOCTL: u32 = 0o1000;
const ECHOKE: u32 = 0o4000;
const IEXTEN: u32 = 0o100000;
// c_cc indices
const VINTR: usize = 0;
const VQUIT: usize = 1;
const VERASE: usize = 2;
const VKILL: usize = 3;
const VEOF: usize = 4;
const VMIN: usize = 6;
const VSUSP: usize = 10;
const VEOL: usize = 11;
const VWERASE: usize = 14;

/// A terminal won't hold more than this either way; past it a writer waits.
const CAP: usize = 64 * 1024;

pub struct Pty {
    pub num: u32,
    /// Typed at the terminal, for the shell: what the line discipline let
    /// through.
    to_slave: VecDeque<u8>,
    /// Written by the shell, for the terminal to draw.
    to_master: VecDeque<u8>,
    /// The line being edited in canonical mode, not yet readable.
    line: Vec<u8>,
    /// End-of-file pending for a canonical reader (Ctrl-D on an empty line).
    eof: bool,
    pub iflag: u32,
    pub oflag: u32,
    pub cflag: u32,
    pub lflag: u32,
    pub cc: [u8; 19],
    /// rows, cols, xpixel, ypixel.
    pub winsize: [u16; 4],
    /// `TIOCSPTLCK`: a slave cannot be opened until the master unlocks it.
    pub locked: bool,
    /// The foreground process group, as `TIOCSPGRP` set it.
    pub pgrp: u32,
    masters: Vec<Weak<RefCell<End>>>,
    slaves: Vec<Weak<RefCell<End>>>,
}

/// One open end. Shared across `dup` and `fork` like any open file.
pub struct End {
    pub pty: Rc<RefCell<Pty>>,
    pub master: bool,
    pub nonblock: bool,
}

/// Every terminal alive, by number. Weak, so a terminal both of whose ends
/// are closed is gone and its number free.
static PTYS: Racy<Vec<Weak<RefCell<Pty>>>> = Racy::new(Vec::new());

impl Pty {
    fn new(num: u32) -> Pty {
        let mut cc = [0u8; 19];
        cc[VINTR] = 3;
        cc[VQUIT] = 0x1c;
        cc[VERASE] = 0x7f;
        cc[VKILL] = 0x15;
        cc[VEOF] = 4;
        cc[VMIN] = 1;
        cc[8] = 0x11; // VSTART
        cc[9] = 0x13; // VSTOP
        cc[VSUSP] = 0x1a;
        cc[12] = 0x12; // VREPRINT
        cc[13] = 0x0f; // VDISCARD
        cc[VWERASE] = 0x17;
        cc[15] = 0x16; // VLNEXT
        Pty {
            num,
            to_slave: VecDeque::new(),
            to_master: VecDeque::new(),
            line: Vec::new(),
            eof: false,
            iflag: ICRNL | IXON,
            oflag: OPOST | ONLCR,
            cflag: 0o17 | 0o60 | 0o200 | 0o2000, // B38400 CS8 CREAD HUPCL
            lflag: ISIG | ICANON | ECHO | ECHOE | ECHOK | ECHOCTL | ECHOKE | IEXTEN,
            cc,
            winsize: [24, 80, 0, 0],
            locked: true,
            pgrp: 0,
            masters: Vec::new(),
            slaves: Vec::new(),
        }
    }

    pub fn master_open(&self) -> bool {
        self.masters.iter().any(|w| w.strong_count() > 0)
    }

    pub fn slave_open(&self) -> bool {
        self.slaves.iter().any(|w| w.strong_count() > 0)
    }

    pub fn termios(&self) -> [u8; TERMIOS_LEN] {
        let mut t = [0u8; TERMIOS_LEN];
        t[0..4].copy_from_slice(&self.iflag.to_le_bytes());
        t[4..8].copy_from_slice(&self.oflag.to_le_bytes());
        t[8..12].copy_from_slice(&self.cflag.to_le_bytes());
        t[12..16].copy_from_slice(&self.lflag.to_le_bytes());
        t[16] = 0;
        t[17..36].copy_from_slice(&self.cc);
        t
    }

    pub fn set_termios(&mut self, t: &[u8; TERMIOS_LEN]) {
        let w = |o: usize| u32::from_le_bytes([t[o], t[o + 1], t[o + 2], t[o + 3]]);
        let was_canon = self.lflag & ICANON != 0;
        self.iflag = w(0);
        self.oflag = w(4);
        self.cflag = w(8);
        self.lflag = w(12);
        self.cc.copy_from_slice(&t[17..36]);
        // Leaving canonical mode hands over the half-typed line as it is.
        if was_canon && self.lflag & ICANON == 0 {
            let l = core::mem::take(&mut self.line);
            self.to_slave.extend(l);
        }
    }

    /// The shell's output on its way to the terminal: `\n` becomes `\r\n`
    /// when the terminal is set to (OPOST|ONLCR, the default).
    fn out(&mut self, bytes: &[u8]) {
        let onlcr = self.oflag & OPOST != 0 && self.oflag & ONLCR != 0;
        for &b in bytes {
            if onlcr && b == b'\n' {
                self.to_master.push_back(b'\r');
            }
            self.to_master.push_back(b);
        }
    }

    fn echo(&mut self, b: u8) {
        if self.lflag & ECHO == 0 {
            return;
        }
        if self.lflag & ECHOCTL != 0 && b < 0x20 && b != b'\n' && b != b'\t' && b != b'\r' {
            self.out(&[b'^', b + 0x40]);
        } else {
            self.out(&[b]);
        }
    }

    /// Bytes typed at the terminal (the master writing). Answers the signal
    /// to deliver to the foreground group, if a key asked for one.
    pub fn input(&mut self, bytes: &[u8]) -> Option<u64> {
        let mut signal = None;
        for &raw in bytes {
            let mut c = raw;
            if self.lflag & ISIG != 0 {
                let sig = if c == self.cc[VINTR] && c != 0 {
                    Some(2) // SIGINT
                } else if c == self.cc[VQUIT] && c != 0 {
                    Some(3) // SIGQUIT
                } else if c == self.cc[VSUSP] && c != 0 {
                    Some(20) // SIGTSTP
                } else {
                    None
                };
                if let Some(s) = sig {
                    self.echo(c);
                    self.line.clear();
                    signal = Some(s);
                    continue;
                }
            }
            if self.iflag & ICRNL != 0 && c == b'\r' {
                c = b'\n';
            }
            if self.lflag & ICANON == 0 {
                self.to_slave.push_back(c);
                self.echo(c);
                continue;
            }
            if c == self.cc[VERASE] || c == 0x08 {
                if self.line.pop().is_some() && self.lflag & ECHO != 0 && self.lflag & ECHOE != 0 {
                    self.out(b"\x08 \x08");
                }
            } else if c == self.cc[VWERASE] && self.lflag & IEXTEN != 0 {
                while self.line.last() == Some(&b' ') {
                    self.line.pop();
                    if self.lflag & ECHO != 0 { self.out(b"\x08 \x08"); }
                }
                while self.line.last().is_some_and(|b| *b != b' ') {
                    self.line.pop();
                    if self.lflag & ECHO != 0 { self.out(b"\x08 \x08"); }
                }
            } else if c == self.cc[VKILL] {
                let n = core::mem::take(&mut self.line).len();
                if self.lflag & ECHO != 0 && self.lflag & ECHOK != 0 {
                    for _ in 0..n {
                        self.out(b"\x08 \x08");
                    }
                }
            } else if c == self.cc[VEOF] {
                if self.line.is_empty() {
                    self.eof = true;
                } else {
                    let l = core::mem::take(&mut self.line);
                    self.to_slave.extend(l);
                }
            } else if c == b'\n' || (c == self.cc[VEOL] && c != 0) {
                self.line.push(c);
                let l = core::mem::take(&mut self.line);
                self.to_slave.extend(l);
                self.echo(b'\n');
            } else {
                self.line.push(c);
                self.echo(c);
            }
        }
        signal
    }

    /// What a slave read may take now: `Some(0)` is end of file.
    pub fn slave_read(&mut self, out: &mut [u8]) -> Option<usize> {
        if self.to_slave.is_empty() {
            if self.eof {
                self.eof = false;
                return Some(0);
            }
            if !self.master_open() {
                return Some(0);
            }
            return None;
        }
        let canon = self.lflag & ICANON != 0;
        let mut n = 0;
        while n < out.len() {
            let Some(b) = self.to_slave.pop_front() else { break };
            out[n] = b;
            n += 1;
            // A canonical read returns at most one line.
            if canon && b == b'\n' {
                break;
            }
        }
        Some(n)
    }

    pub fn slave_write(&mut self, bytes: &[u8]) -> Option<usize> {
        if self.to_master.len() >= CAP {
            return None;
        }
        let take = bytes.len().min(CAP - self.to_master.len());
        self.out(&bytes[..take]);
        Some(take)
    }

    /// `None` waits; `Some(Err)` is EIO, which is what a master reads once
    /// every slave has closed and the output is drained -- how a terminal
    /// emulator learns its shell exited.
    pub fn master_read(&mut self, out: &mut [u8]) -> Option<Result<usize, ()>> {
        if self.to_master.is_empty() {
            return if self.slave_open() || self.slaves.is_empty() { None } else { Some(Err(())) };
        }
        let mut n = 0;
        while n < out.len() {
            let Some(b) = self.to_master.pop_front() else { break };
            out[n] = b;
            n += 1;
        }
        Some(Ok(n))
    }

    pub fn slave_readable(&self) -> bool {
        !self.to_slave.is_empty() || self.eof || !self.master_open()
    }

    pub fn master_readable(&self) -> bool {
        !self.to_master.is_empty() || (!self.slaves.is_empty() && !self.slave_open())
    }

    pub fn readable_bytes(&self, master: bool) -> usize {
        if master { self.to_master.len() } else { self.to_slave.len() }
    }

    pub fn flush(&mut self, input: bool, output: bool) {
        if input {
            self.to_slave.clear();
            self.line.clear();
        }
        if output {
            self.to_master.clear();
        }
    }
}

/// `/dev/ptmx`: a new terminal, and its master end.
pub fn open_master(nonblock: bool) -> Rc<RefCell<End>> {
    let all = unsafe { PTYS.get() };
    all.retain(|w| w.strong_count() > 0);
    let mut num = 0u32;
    while all.iter().any(|w| w.upgrade().is_some_and(|p| p.borrow().num == num)) {
        num += 1;
    }
    let pty = Rc::new(RefCell::new(Pty::new(num)));
    all.push(Rc::downgrade(&pty));
    let end = Rc::new(RefCell::new(End { pty: pty.clone(), master: true, nonblock }));
    pty.borrow_mut().masters.push(Rc::downgrade(&end));
    end
}

/// `/dev/pts/N`: the slave end of terminal N, once its master has unlocked it.
pub fn open_slave(num: u32, nonblock: bool) -> Result<Rc<RefCell<End>>, i64> {
    const ENOENT: i64 = -2;
    const EIO: i64 = -5;
    let pty = unsafe { PTYS.get() }
        .iter()
        .filter_map(|w| w.upgrade())
        .find(|p| p.borrow().num == num)
        .ok_or(ENOENT)?;
    if pty.borrow().locked {
        return Err(EIO);
    }
    let end = Rc::new(RefCell::new(End { pty: pty.clone(), master: false, nonblock }));
    pty.borrow_mut().slaves.push(Rc::downgrade(&end));
    Ok(end)
}

/// The numbers alive, for listing `/dev/pts`.
pub fn numbers() -> Vec<u32> {
    let mut v: Vec<u32> = unsafe { PTYS.get() }.iter().filter_map(|w| w.upgrade()).map(|p| p.borrow().num).collect();
    v.sort();
    v
}

/// `/dev/pts/N` as a path, if it names a terminal that exists.
pub fn slave_number(path: &str) -> Option<u32> {
    let n: u32 = path.strip_prefix("/dev/pts/")?.parse().ok()?;
    numbers().contains(&n).then_some(n)
}

/// What `diag linux` asks of the line discipline, with no guest.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let m = open_master(true);
    let num = m.borrow().pty.borrow().num;
    out.push(("a locked terminal refuses its slave", open_slave(num, true).is_err()));
    m.borrow().pty.borrow_mut().locked = false;
    let s = open_slave(num, true);
    out.push(("and opens it once unlocked", s.is_ok()));
    let pty = m.borrow().pty.clone();
    let mut p = pty.borrow_mut();
    let mut buf = [0u8; 64];
    p.input(b"ls");
    out.push(("a half-typed canonical line is not readable yet", p.slave_read(&mut buf).is_none()));
    p.input(b"x\x7f\r");
    let n = p.slave_read(&mut buf).unwrap_or(0);
    out.push(("erase and Enter make the line, with CR read as NL", &buf[..n] == b"ls\n"));
    let mut echo = [0u8; 64];
    let e = p.master_read(&mut echo).and_then(|r| r.ok()).unwrap_or(0);
    out.push(("and the typing was echoed, erase as backspace-space-backspace", &echo[..e] == b"lsx\x08 \x08\r\n"));
    p.slave_write(b"hi\n");
    let e = p.master_read(&mut echo).and_then(|r| r.ok()).unwrap_or(0);
    out.push(("output gets CR before NL", &echo[..e] == b"hi\r\n"));
    out.push(("Ctrl-C asks for SIGINT and does not reach the reader", p.input(&[3]) == Some(2) && p.slave_read(&mut buf).is_none()));
    let mut raw = p.termios();
    raw[12..16].copy_from_slice(&0u32.to_le_bytes());
    p.set_termios(&raw);
    p.master_read(&mut echo);
    p.input(b"q");
    let n = p.slave_read(&mut buf).unwrap_or(0);
    let mut none = [0u8; 8];
    out.push(("raw mode passes a key through at once, unechoed", &buf[..n] == b"q" && p.master_read(&mut none).is_none()));
    drop(p);
    drop(pty);
    drop(s);
    drop(m);
    out.push(("a terminal with both ends closed is gone", !numbers().contains(&num)));
    out
}
