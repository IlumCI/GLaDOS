//! A modal text editor.
//!
//! Until now the only way to author anything was `write <path> <one line>`,
//! which makes the namespace a place to keep text produced elsewhere rather
//! than somewhere text can be written. This is the missing half.
//!
//! Modal in the vi lineage, because that is the design that suits a machine
//! with no mouse, no chords worth speaking of, and a keyboard driver that
//! reports plain bytes. Normal mode gives every letter a meaning; insert mode
//! gives them all back.
//!
//! # Drawing
//!
//! The editor owns the screen rather than scrolling through the console, so it
//! draws to the framebuffer directly and asks the console to repaint on the
//! way out.
//!
//! It keeps a shadow of what is currently on screen and redraws only the cells
//! that differ. That is not premature: a full repaint at 1920x1080 is a little
//! over two million volatile stores -- the same cost that made console
//! scrolling visibly slow before it was fixed -- and paying it per keystroke
//! would put a tenth of a second between pressing a key and seeing it. Typing
//! changes a handful of cells, so typing costs a handful of writes.

use crate::dev::kbd;
use crate::gfx::{self, font, palette, Color, Framebuffer};
use crate::sysbox;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

const SCALE: u32 = 2;
const ESC: u8 = 27;

const FG: Color = palette::LTGRAY;
const BG: Color = palette::BLACK;
const GUTTER: Color = palette::DKGRAY;
const STATUS_BG: Color = palette::BLUE;
const STATUS_FG: Color = palette::WHITE;
const CURSOR: Color = palette::LTGREEN;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    Insert,
    Command,
}

impl Mode {
    fn label(&self) -> &'static str {
        match self {
            Mode::Normal => "NORMAL",
            Mode::Insert => "INSERT",
            Mode::Command => "COMMAND",
        }
    }
}

/// One drawn cell: the character and its colour index into `PALETTE`-ish use.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell {
    ch: u8,
    fg: u8,
    bg: u8,
}

const C_FG: u8 = 0;
const C_GUTTER: u8 = 1;
const C_STATUS: u8 = 2;
const C_CURSOR: u8 = 3;

fn colour(i: u8) -> (Color, Color) {
    match i {
        C_GUTTER => (GUTTER, BG),
        C_STATUS => (STATUS_FG, STATUS_BG),
        C_CURSOR => (BG, CURSOR),
        _ => (FG, BG),
    }
}

const BLANK: Cell = Cell { ch: b' ', fg: C_FG, bg: C_FG };

/// How far back `u` can go. Bounded because the buffer is cloned each time and
/// an editor that grows without limit on a machine with no swap is a way to
/// lose the file it is holding.
const UNDO_MAX: usize = 200;

#[derive(Clone)]
struct Snapshot {
    lines: Vec<Vec<char>>,
    cx: usize,
    cy: usize,
    dirty: bool,
}

pub struct Editor {
    /// Lines as characters rather than bytes, so cursor arithmetic is index
    /// arithmetic and cannot land in the middle of a multi-byte sequence.
    lines: Vec<Vec<char>>,
    cx: usize,
    cy: usize,
    top: usize,
    mode: Mode,
    path: String,
    dirty: bool,
    cmd: String,
    status: String,
    quit: bool,
    /// A single pending operator, for the two-key sequences: `dd`, `gg`.
    pending: Option<u8>,

    /// Whole-buffer snapshots.
    ///
    /// Files here are small enough that cloning the buffer costs less than the
    /// bookkeeping a diff-based history would need, and it cannot be subtly
    /// wrong the way a partial record can. Pushed once per normal-mode change
    /// and once per insert *session*, so `u` undoes a typed word rather than
    /// one letter -- which is what vi does and what muscle memory expects.
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,

    cols: usize,
    rows: usize,
    shadow: Vec<Cell>,
    frame: Vec<Cell>,
}

impl Editor {
    pub fn open(path: &str, fb: &Framebuffer) -> Self {
        let cell_w = font::GLYPH_W * SCALE;
        let cell_h = font::GLYPH_H * SCALE;
        let cols = (fb.width() / cell_w) as usize;
        let rows = (fb.height() / cell_h) as usize;

        let (lines, status) = match sysbox::read_blob(path) {
            Some(bytes) => {
                let text = String::from_utf8_lossy_ascii(&bytes);
                let mut v: Vec<Vec<char>> = text.split('\n').map(|l| l.chars().collect()).collect();
                // A trailing newline produces an empty final element; keeping
                // it would add a blank line on every save.
                if v.len() > 1 && v.last().map(|l| l.is_empty()).unwrap_or(false) {
                    v.pop();
                }
                if v.is_empty() {
                    v.push(Vec::new());
                }
                let n = v.len();
                (v, alloc::format!("\"{}\" {} lines", path, n))
            }
            None => (vec![Vec::new()], alloc::format!("\"{}\" [new]", path)),
        };

        Self {
            lines,
            cx: 0,
            cy: 0,
            top: 0,
            mode: Mode::Normal,
            path: String::from(path),
            dirty: false,
            cmd: String::new(),
            status,
            quit: false,
            pending: None,
            undo: Vec::new(),
            redo: Vec::new(),
            cols,
            rows,
            // Deliberately not equal to any real cell, so the first frame
            // repaints everything and nothing is left over from the shell.
            shadow: vec![BLANK; cols * rows],
            frame: vec![BLANK; cols * rows],
        }
    }

    fn text_rows(&self) -> usize {
        self.rows.saturating_sub(2)
    }

    fn gutter(&self) -> usize {
        // Wide enough for the largest line number plus a space.
        let mut w = 2;
        let mut n = self.lines.len();
        while n >= 10 {
            n /= 10;
            w += 1;
        }
        w + 1
    }

    fn line_len(&self, y: usize) -> usize {
        self.lines.get(y).map(|l| l.len()).unwrap_or(0)
    }

    fn clamp(&mut self) {
        if self.cy >= self.lines.len() {
            self.cy = self.lines.len().saturating_sub(1);
        }
        let len = self.line_len(self.cy);
        // Normal mode sits *on* a character; insert mode sits after the last.
        let max = if self.mode == Mode::Insert { len } else { len.saturating_sub(1) };
        if self.cx > max {
            self.cx = max;
        }
        let vis = self.text_rows();
        if self.cy < self.top {
            self.top = self.cy;
        } else if vis > 0 && self.cy >= self.top + vis {
            self.top = self.cy + 1 - vis;
        }
    }

    // --- rendering ------------------------------------------------------

    fn put(&mut self, x: usize, y: usize, ch: u8, fg: u8, bg: u8) {
        if x < self.cols && y < self.rows {
            self.frame[y * self.cols + x] = Cell { ch, fg, bg };
        }
    }

    fn puts(&mut self, x: usize, y: usize, s: &str, fg: u8, bg: u8) {
        for (i, b) in s.bytes().enumerate() {
            self.put(x + i, y, b, fg, bg);
        }
    }

    fn compose(&mut self) {
        for c in self.frame.iter_mut() {
            *c = BLANK;
        }
        let gut = self.gutter();
        let vis = self.text_rows();

        for row in 0..vis {
            let ln = self.top + row;
            if ln >= self.lines.len() {
                self.put(0, row, b'~', C_GUTTER, C_FG);
                continue;
            }
            // Line numbers, right-aligned in the gutter.
            let mut num = [b' '; 8];
            let mut n = ln + 1;
            let mut k = gut - 1;
            loop {
                num[k - 1] = b'0' + (n % 10) as u8;
                n /= 10;
                if n == 0 || k == 1 {
                    break;
                }
                k -= 1;
            }
            for i in 0..gut - 1 {
                self.put(i, row, num[i], C_GUTTER, C_FG);
            }

            let line = self.lines[ln].clone();
            for (i, ch) in line.iter().enumerate() {
                let x = gut + i;
                if x >= self.cols {
                    break;
                }
                // Non-ASCII is drawn as a placeholder rather than mangled: the
                // font has 256 glyphs and no notion of anything wider.
                let b = if (*ch as u32) < 128 { *ch as u8 } else { b'?' };
                self.put(x, row, b, C_FG, C_FG);
            }
        }

        // Status line.
        let dirty = if self.dirty { " [+]" } else { "" };
        let bar = alloc::format!(
            " {}  {}{}  {}:{} ",
            self.mode.label(),
            self.path,
            dirty,
            self.cy + 1,
            self.cx + 1
        );
        for x in 0..self.cols {
            self.put(x, self.rows - 2, b' ', C_STATUS, C_STATUS);
        }
        self.puts(0, self.rows - 2, &bar, C_STATUS, C_STATUS);

        // Message or command line.
        let last = self.rows - 1;
        if self.mode == Mode::Command {
            let line = alloc::format!(":{}", self.cmd);
            self.puts(0, last, &line, C_FG, C_FG);
            self.put(line.len(), last, b' ', C_CURSOR, C_CURSOR);
        } else {
            let msg = self.status.clone();
            self.puts(0, last, &msg, C_FG, C_FG);
        }

        // Cursor, drawn last so it wins.
        if self.mode != Mode::Command {
            let row = self.cy - self.top;
            if row < vis {
                let x = gut + self.cx;
                let ch = self
                    .lines
                    .get(self.cy)
                    .and_then(|l| l.get(self.cx))
                    .map(|c| if (*c as u32) < 128 { *c as u8 } else { b'?' })
                    .unwrap_or(b' ');
                self.put(x, row, ch, C_CURSOR, C_CURSOR);
            }
        }
    }

    fn flush(&mut self, fb: &Framebuffer) {
        let cell_w = font::GLYPH_W * SCALE;
        let cell_h = font::GLYPH_H * SCALE;
        for y in 0..self.rows {
            for x in 0..self.cols {
                let i = y * self.cols + x;
                if self.frame[i] == self.shadow[i] {
                    continue;
                }
                let c = self.frame[i];
                let (fg, bg) = colour(c.fg.max(c.bg));
                let (fg, bg) = if c.fg == C_CURSOR || c.bg == C_CURSOR {
                    colour(C_CURSOR)
                } else if c.fg == C_STATUS {
                    colour(C_STATUS)
                } else if c.fg == C_GUTTER {
                    (GUTTER, BG)
                } else {
                    (fg, bg)
                };
                let s = [c.ch];
                let text = core::str::from_utf8(&s).unwrap_or(" ");
                fb.draw_text(x as u32 * cell_w, y as u32 * cell_h, text, fg, bg, SCALE);
                self.shadow[i] = c;
            }
        }
    }

    // --- history --------------------------------------------------------

    fn snapshot(&self) -> Snapshot {
        Snapshot { lines: self.lines.clone(), cx: self.cx, cy: self.cy, dirty: self.dirty }
    }

    /// Record the state *before* a change.
    ///
    /// Any new edit invalidates the redo stack: once history has branched,
    /// replaying the abandoned branch would apply changes to a buffer they
    /// were never computed against.
    fn checkpoint(&mut self) {
        let s = self.snapshot();
        self.undo.push(s);
        if self.undo.len() > UNDO_MAX {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn restore(&mut self, s: Snapshot) {
        self.lines = s.lines;
        self.cx = s.cx;
        self.cy = s.cy;
        self.dirty = s.dirty;
    }

    fn undo_one(&mut self) {
        match self.undo.pop() {
            Some(s) => {
                let now = self.snapshot();
                self.redo.push(now);
                self.restore(s);
                self.status = alloc::format!("undo ({} left)", self.undo.len());
            }
            None => self.status = String::from("nothing to undo"),
        }
    }

    fn redo_one(&mut self) {
        match self.redo.pop() {
            Some(s) => {
                let now = self.snapshot();
                self.undo.push(now);
                self.restore(s);
                self.status = alloc::format!("redo ({} left)", self.redo.len());
            }
            None => self.status = String::from("nothing to redo"),
        }
    }

    // --- editing --------------------------------------------------------

    fn insert_char(&mut self, ch: char) {
        let cx = self.cx;
        if let Some(line) = self.lines.get_mut(self.cy) {
            let at = cx.min(line.len());
            line.insert(at, ch);
            self.cx = at + 1;
            self.dirty = true;
        }
    }

    fn split_line(&mut self) {
        let cx = self.cx;
        let rest: Vec<char> = {
            let line = &mut self.lines[self.cy];
            let at = cx.min(line.len());
            line.split_off(at)
        };
        self.lines.insert(self.cy + 1, rest);
        self.cy += 1;
        self.cx = 0;
        self.dirty = true;
    }

    fn backspace(&mut self) {
        if self.cx > 0 {
            let cx = self.cx;
            self.lines[self.cy].remove(cx - 1);
            self.cx -= 1;
            self.dirty = true;
        } else if self.cy > 0 {
            // Joining is the only case that changes the line count, so the
            // cursor has to land where the seam is rather than at column zero.
            let cur = self.lines.remove(self.cy);
            self.cy -= 1;
            self.cx = self.lines[self.cy].len();
            self.lines[self.cy].extend(cur);
            self.dirty = true;
        }
    }

    fn delete_line(&mut self) {
        if self.lines.len() == 1 {
            self.lines[0].clear();
        } else {
            self.lines.remove(self.cy);
            if self.cy >= self.lines.len() {
                self.cy = self.lines.len() - 1;
            }
        }
        self.cx = 0;
        self.dirty = true;
    }

    fn word_forward(&mut self) {
        let line = &self.lines[self.cy];
        let mut i = self.cx;
        while i < line.len() && !line[i].is_whitespace() {
            i += 1;
        }
        while i < line.len() && line[i].is_whitespace() {
            i += 1;
        }
        if i >= line.len() && self.cy + 1 < self.lines.len() {
            self.cy += 1;
            self.cx = 0;
        } else {
            self.cx = i;
        }
    }

    fn word_back(&mut self) {
        if self.cx == 0 {
            if self.cy > 0 {
                self.cy -= 1;
                self.cx = self.line_len(self.cy);
            }
            return;
        }
        let line = &self.lines[self.cy];
        let mut i = self.cx - 1;
        while i > 0 && line[i].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !line[i - 1].is_whitespace() {
            i -= 1;
        }
        self.cx = i;
    }

    fn save(&mut self, to: Option<&str>) {
        let path = match to {
            Some(p) if !p.is_empty() => String::from(p),
            _ => self.path.clone(),
        };
        let mut out = String::new();
        for (i, line) in self.lines.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            for c in line {
                out.push(*c);
            }
        }
        out.push('\n');
        let bytes = out.len();
        if sysbox::write_text(&path, &out) {
            self.dirty = false;
            self.path = path.clone();
            self.status = alloc::format!("\"{}\" {}L {}B written", path, self.lines.len(), bytes);
        } else {
            self.status = alloc::format!("could not write \"{}\"", path);
        }
    }

    fn run_command(&mut self) {
        let cmd = self.cmd.trim().to_string();
        self.cmd.clear();
        self.mode = Mode::Normal;

        let mut it = cmd.splitn(2, ' ');
        let verb = it.next().unwrap_or("");
        let arg = it.next().unwrap_or("").trim();

        match verb {
            "w" => self.save(Some(arg)),
            "wq" | "x" => {
                self.save(Some(arg));
                if !self.dirty {
                    self.quit = true;
                }
            }
            "q" => {
                if self.dirty {
                    // Refusing is the whole point: the alternative is silently
                    // discarding work on a typo.
                    self.status = String::from("unsaved changes -- :w to write, :q! to discard");
                } else {
                    self.quit = true;
                }
            }
            "q!" => self.quit = true,
            "" => {}
            other => self.status = alloc::format!("not a command: {}", other),
        }
    }

    fn normal_key(&mut self, k: u8) {
        // Two-key sequences first, so `dd` and `gg` do not fall through to the
        // single-key meanings of their second character.
        if let Some(p) = self.pending.take() {
            match (p, k) {
                (b'd', b'd') => {
                    self.checkpoint();
                    self.delete_line();
                }
                (b'g', b'g') => {
                    self.cy = 0;
                    self.cx = 0;
                }
                _ => {}
            }
            return;
        }

        match k {
            b'h' | kbd::KEY_LEFT => self.cx = self.cx.saturating_sub(1),
            b'l' | kbd::KEY_RIGHT => self.cx += 1,
            b'k' | kbd::KEY_UP => self.cy = self.cy.saturating_sub(1),
            b'j' | kbd::KEY_DOWN => self.cy += 1,
            b'0' | kbd::KEY_HOME => self.cx = 0,
            b'$' | kbd::KEY_END => self.cx = self.line_len(self.cy).saturating_sub(1),
            b'G' => self.cy = self.lines.len() - 1,
            b'w' => self.word_forward(),
            b'b' => self.word_back(),
            b'd' | b'g' => self.pending = Some(k),
            b'x' => {
                let cx = self.cx;
                if cx < self.line_len(self.cy) {
                    self.checkpoint();
                    self.lines[self.cy].remove(cx);
                    self.dirty = true;
                }
            }
            b'u' => self.undo_one(),
            // Ctrl-R, as in vi.
            0x12 => self.redo_one(),
            // Entering insert mode checkpoints once, so `u` undoes the whole
            // typed run rather than one character at a time.
            b'i' => {
                self.checkpoint();
                self.mode = Mode::Insert;
            }
            b'a' => {
                self.checkpoint();
                self.mode = Mode::Insert;
                self.cx += 1;
            }
            b'A' => {
                self.checkpoint();
                self.mode = Mode::Insert;
                self.cx = self.line_len(self.cy);
            }
            b'I' => {
                self.checkpoint();
                self.mode = Mode::Insert;
                self.cx = 0;
            }
            b'o' => {
                self.checkpoint();
                self.lines.insert(self.cy + 1, Vec::new());
                self.cy += 1;
                self.cx = 0;
                self.mode = Mode::Insert;
                self.dirty = true;
            }
            b'O' => {
                self.checkpoint();
                self.lines.insert(self.cy, Vec::new());
                self.cx = 0;
                self.mode = Mode::Insert;
                self.dirty = true;
            }
            b':' => {
                self.mode = Mode::Command;
                self.cmd.clear();
            }
            _ => {}
        }
    }

    fn insert_key(&mut self, k: u8) {
        match k {
            ESC => {
                self.mode = Mode::Normal;
                // vi leaves the cursor on the character before where insert
                // ended, and muscle memory expects it.
                self.cx = self.cx.saturating_sub(1);
            }
            b'\n' => self.split_line(),
            8 | 0x7F => self.backspace(),
            kbd::KEY_LEFT => self.cx = self.cx.saturating_sub(1),
            kbd::KEY_RIGHT => self.cx += 1,
            kbd::KEY_UP => self.cy = self.cy.saturating_sub(1),
            kbd::KEY_DOWN => self.cy += 1,
            kbd::KEY_HOME => self.cx = 0,
            kbd::KEY_END => self.cx = self.line_len(self.cy),
            kbd::KEY_DELETE => {
                let cx = self.cx;
                if cx < self.line_len(self.cy) {
                    self.lines[self.cy].remove(cx);
                    self.dirty = true;
                }
            }
            c if (0x20..0x7F).contains(&c) => self.insert_char(c as char),
            _ => {}
        }
    }

    fn command_key(&mut self, k: u8) {
        match k {
            ESC => {
                self.mode = Mode::Normal;
                self.cmd.clear();
            }
            b'\n' => self.run_command(),
            8 | 0x7F => {
                self.cmd.pop();
            }
            c if (0x20..0x7F).contains(&c) => self.cmd.push(c as char),
            _ => {}
        }
    }
}

/// Trait-free helper: the input may be arbitrary bytes from the store, and
/// `String::from_utf8` would reject a file that is perfectly editable.
trait Lossy {
    fn from_utf8_lossy_ascii(b: &[u8]) -> String;
}

impl Lossy for String {
    fn from_utf8_lossy_ascii(b: &[u8]) -> String {
        match core::str::from_utf8(b) {
            Ok(s) => String::from(s),
            Err(_) => b
                .iter()
                .map(|c| if (0x20..0x7F).contains(c) || *c == b'\n' { *c as char } else { '?' })
                .collect(),
        }
    }
}

/// Run the editor until it quits. Blocks the shell, deliberately.
pub fn run(path: &str) {
    let Some(fb) = gfx::primary() else {
        crate::kprintln!("  no framebuffer");
        return;
    };

    let mut ed = Editor::open(path, &fb);
    fb.fill(BG);

    loop {
        ed.clamp();
        ed.compose();
        ed.flush(&fb);
        if ed.quit {
            break;
        }

        let Some(k) = kbd::pop_any() else {
            // Idle until the next interrupt rather than spinning a core to
            // wait for a keystroke.
            unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
            continue;
        };

        match ed.mode {
            Mode::Normal => ed.normal_key(k),
            Mode::Insert => ed.insert_key(k),
            Mode::Command => ed.command_key(k),
        }
    }

    // Hand the screen back.
    fb.fill(palette::BLACK);
    crate::gfx::console::redraw();
}
