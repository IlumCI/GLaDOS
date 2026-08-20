//! Minesweeper. 9x9, ten mines, the 3.1 rules exactly.
//!
//! Here for the same reason Windows shipped it: it is the smallest program
//! that exercises everything a desktop claims to do -- both mouse buttons
//! doing different things on the same pixel, per-cell repaint under the
//! compositor, a timer, and a window that is pure content with no widget
//! stack. If Minesweeper plays right, the window manager works.
//!
//! Two departures from a plain grid, both the classic ones:
//!
//! * Mines are placed on the *first reveal*, never before, and never on the
//!   revealed cell. A game that can kill on click one is a coin toss with
//!   ceremony, and every implementation since 1990 has dealt with it this
//!   way.
//! * Revealing a zero floods outward. Done with an explicit stack rather
//!   than recursion -- kernel stacks are small and a 9x9 board can flood 70
//!   cells deep.

use super::theme::{self, Rect};
use super::{Color, DeskApp, Framebuffer};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

const W: usize = 9;
const H: usize = 9;
const MINES: usize = 10;
/// Pixel size of one cell. Chosen so a fingerless pointer can still hit one.
const CELL: u32 = 30;
/// The header strip: counter, face, timer.
const HEAD: u32 = 44;
const PAD: u32 = 6;

#[derive(Clone, Copy, Default)]
struct Cell {
    mine: bool,
    open: bool,
    flag: bool,
    /// Neighbouring mines, filled when the board is laid.
    n: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Playing,
    Won,
    Lost,
}

pub struct Mines {
    cells: Vec<Cell>,
    state: State,
    /// Keyboard selection, so the game is playable over `win keys`.
    sel: (usize, usize),
    /// Mines are laid on the first reveal, excluding it.
    laid: bool,
    /// TSC milliseconds at the first reveal and at the end, for the timer.
    t0: u64,
    t_end: u64,
    seed: u64,
}

fn now_ms() -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        return 0;
    }
    crate::time::rdtsc() / (mhz * 1000)
}

impl Mines {
    pub fn new() -> Self {
        Self {
            cells: vec![Cell::default(); W * H],
            state: State::Playing,
            sel: (W / 2, H / 2),
            laid: false,
            t0: 0,
            t_end: 0,
            seed: crate::time::rdtsc() | 1,
        }
    }

    /// The window size that fits the board exactly.
    pub fn preferred() -> (u32, u32) {
        (
            W as u32 * CELL + PAD * 2 + theme::FRAME * 2 + 8,
            H as u32 * CELL + HEAD + PAD * 3 + theme::TITLE_H + theme::FRAME * 2 + 10,
        )
    }

    fn reset(&mut self) {
        self.cells.fill(Cell::default());
        self.state = State::Playing;
        self.laid = false;
        self.t0 = 0;
        self.t_end = 0;
    }

    fn rand(&mut self) -> u64 {
        // xorshift64*, the same generator the synthetic model uses. Seeded
        // from the TSC at construction, which is as random as a kernel with
        // no RDRAND policy gets and exactly random enough for a game.
        let mut s = self.seed;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.seed = s;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Lay the mines, everywhere but `safe` -- the cell of the first reveal.
    fn lay(&mut self, safe: usize) {
        let mut placed = 0;
        while placed < MINES {
            let at = (self.rand() % (W * H) as u64) as usize;
            if at == safe || self.cells[at].mine {
                continue;
            }
            self.cells[at].mine = true;
            placed += 1;
        }
        for y in 0..H {
            for x in 0..W {
                let mut n = 0;
                for (nx, ny) in neighbours(x, y) {
                    if self.cells[ny * W + nx].mine {
                        n += 1;
                    }
                }
                self.cells[y * W + x].n = n;
            }
        }
        self.laid = true;
        self.t0 = now_ms();
    }

    fn reveal(&mut self, x: usize, y: usize) {
        if self.state != State::Playing || self.cells[y * W + x].flag {
            return;
        }
        if !self.laid {
            self.lay(y * W + x);
        }
        let at = y * W + x;
        if self.cells[at].open {
            return;
        }
        if self.cells[at].mine {
            // Every mine shows on a loss; a board that keeps its secret
            // teaches nothing.
            for c in self.cells.iter_mut() {
                if c.mine {
                    c.open = true;
                }
            }
            self.state = State::Lost;
            self.t_end = now_ms();
            return;
        }
        let mut stack = vec![(x, y)];
        while let Some((cx, cy)) = stack.pop() {
            let i = cy * W + cx;
            if self.cells[i].open || self.cells[i].flag {
                continue;
            }
            self.cells[i].open = true;
            if self.cells[i].n == 0 && !self.cells[i].mine {
                for (nx, ny) in neighbours(cx, cy) {
                    if !self.cells[ny * W + nx].open {
                        stack.push((nx, ny));
                    }
                }
            }
        }
        if self
            .cells
            .iter()
            .all(|c| c.open || c.mine)
        {
            self.state = State::Won;
            self.t_end = now_ms();
            // Winning flags the rest, as the original did.
            for c in self.cells.iter_mut() {
                if c.mine {
                    c.flag = true;
                }
            }
        }
    }

    fn toggle_flag(&mut self, x: usize, y: usize) {
        if self.state != State::Playing {
            return;
        }
        let c = &mut self.cells[y * W + x];
        if !c.open {
            c.flag = !c.flag;
        }
    }

    fn flags(&self) -> usize {
        self.cells.iter().filter(|c| c.flag).count()
    }

    /// Board origin inside a client rectangle.
    fn board_at(client: Rect) -> (u32, u32) {
        (client.x + PAD + 2, client.y + HEAD + PAD * 2)
    }

    fn face_rect(client: Rect) -> Rect {
        let s = HEAD - 10;
        Rect::new(client.x + (client.w.saturating_sub(s)) / 2, client.y + PAD + 2, s, s)
    }

    fn cell_at(client: Rect, px: i32, py: i32) -> Option<(usize, usize)> {
        let (bx, by) = Self::board_at(client);
        let (dx, dy) = (px - bx as i32, py - by as i32);
        if dx < 0 || dy < 0 {
            return None;
        }
        let (x, y) = ((dx as u32 / CELL) as usize, (dy as u32 / CELL) as usize);
        (x < W && y < H).then_some((x, y))
    }
}

fn neighbours(x: usize, y: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(8);
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < W && (ny as usize) < H {
                out.push((nx as usize, ny as usize));
            }
        }
    }
    out
}

/// The classic number colours, one per count. Index 0 is unused.
fn count_color(n: u8) -> Color {
    match n {
        1 => Color::new(0x00, 0x00, 0xC0),
        2 => Color::new(0x00, 0x80, 0x00),
        3 => Color::new(0xC0, 0x00, 0x00),
        4 => Color::new(0x00, 0x00, 0x80),
        5 => Color::new(0x80, 0x00, 0x00),
        6 => Color::new(0x00, 0x80, 0x80),
        7 => theme::TEXT,
        _ => theme::SHADOW,
    }
}

impl DeskApp for Mines {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);

        // Header: mines left, the face, elapsed seconds. Both readouts are
        // sunken wells, exactly where 3.1 put them.
        let digits = |v: i64| -> String {
            let mut s = String::new();
            let v = v.clamp(-99, 999);
            if v < 0 {
                s.push('-');
            }
            let a = if v < 0 { -v } else { v } as u64;
            if a >= 100 {
                s.push((b'0' + (a / 100 % 10) as u8) as char);
            }
            if a >= 10 {
                s.push((b'0' + (a / 10 % 10) as u8) as char);
            }
            s.push((b'0' + (a % 10) as u8) as char);
            while s.len() < 3 {
                s.insert(0, '0');
            }
            s
        };

        let left = MINES as i64 - self.flags() as i64;
        let lw = theme::text_w(3) + 8;
        let well_h = HEAD - 16;
        let ly = client.y + PAD + 5;
        let lr = Rect::new(client.x + PAD + 2, ly, lw, well_h);
        theme::well(fb, lr, theme::SCREEN);
        theme::text(fb, lr.x + 4, lr.y + (well_h - theme::text_h()) / 2, &digits(left), theme::APERTURE, theme::SCREEN);

        let secs = match self.state {
            _ if !self.laid => 0,
            State::Playing => (now_ms().saturating_sub(self.t0) / 1000) as i64,
            _ => (self.t_end.saturating_sub(self.t0) / 1000) as i64,
        };
        let tr = Rect::new(client.x + client.w - PAD - 2 - lw, ly, lw, well_h);
        theme::well(fb, tr, theme::SCREEN);
        theme::text(fb, tr.x + 4, tr.y + (well_h - theme::text_h()) / 2, &digits(secs), theme::APERTURE, theme::SCREEN);

        // The face is the reset button. The mark stands in for the smiley --
        // orange while playing, red when it went wrong, green when it did not.
        let f = Self::face_rect(client);
        theme::button(fb, f, "", false, false);
        let mood = match self.state {
            State::Playing => theme::APERTURE_DEEP,
            State::Lost => Color::new(0xC0, 0x20, 0x10),
            State::Won => Color::new(0x20, 0x90, 0x20),
        };
        super::splash::aperture(
            fb,
            (f.x + f.w / 2) as i32,
            (f.y + f.h / 2) as i32,
            (f.w as i32 / 2) - 5,
            mood,
            theme::FACE,
        );

        // The board.
        let (bx, by) = Self::board_at(client);
        for y in 0..H {
            for x in 0..W {
                let c = self.cells[y * W + x];
                let r = Rect::new(bx + x as u32 * CELL, by + y as u32 * CELL, CELL, CELL);
                if c.open {
                    // Sunken and flat, with a 1px grid line.
                    fb.rect(r.x, r.y, r.w, r.h, theme::FACE);
                    fb.rect(r.x, r.y, r.w, 1, theme::SHADOW);
                    fb.rect(r.x, r.y, 1, r.h, theme::SHADOW);
                    if c.mine {
                        let cx = r.x + CELL / 2;
                        let cy = r.y + CELL / 2;
                        // A mine the board lost on gets a red bed.
                        if self.state == State::Lost {
                            fb.rect(r.x + 1, r.y + 1, r.w - 1, r.h - 1, Color::new(0xD8, 0x50, 0x40));
                        }
                        fb.rect(cx - 6, cy - 6, 12, 12, theme::TEXT);
                        fb.rect(cx - 8, cy - 1, 16, 3, theme::TEXT);
                        fb.rect(cx - 1, cy - 8, 3, 16, theme::TEXT);
                        fb.rect(cx - 4, cy - 4, 3, 3, theme::HILIGHT);
                    } else if c.n > 0 {
                        let d = [(b'0' + c.n) as char];
                        let mut s = String::new();
                        s.push(d[0]);
                        let tx = r.x + (CELL - theme::text_w(1)) / 2 + 1;
                        let ty = r.y + (CELL - theme::text_h()) / 2 + 1;
                        theme::text_over(fb, tx, ty, &s, count_color(c.n));
                    }
                } else {
                    // Closed: a raised button face.
                    fb.rect(r.x, r.y, r.w, r.h, theme::FACE);
                    theme::bevel(fb, r, true);
                    if c.flag {
                        let cx = r.x + CELL / 2;
                        let cy = r.y + CELL / 2;
                        fb.rect(cx - 1, cy - 7, 2, 14, theme::TEXT);
                        fb.rect(cx - 8, cy - 7, 8, 6, theme::APERTURE);
                        fb.rect(cx - 5, cy + 5, 10, 2, theme::TEXT);
                    }
                }
            }
        }

        // Keyboard selection, only while the window has the keyboard --
        // exactly the rule every widget follows.
        if focused && self.state == State::Playing {
            let r = Rect::new(
                bx + self.sel.0 as u32 * CELL,
                by + self.sel.1 as u32 * CELL,
                CELL,
                CELL,
            );
            fb.frame(r.x + 1, r.y + 1, r.w - 2, r.h - 2, theme::APERTURE);
        }

        let msg = match self.state {
            State::Playing => "arrows + Enter reveal, f flags, n new",
            State::Won => "clear. n deals again",
            State::Lost => "that was a mine. n deals again",
        };
        let my = by + H as u32 * CELL + 4;
        if my + theme::text_h() < client.y + client.h {
            theme::text(fb, bx, my, msg, theme::TEXT, theme::FACE);
        }
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd;
        match k {
            kbd::KEY_LEFT => self.sel.0 = self.sel.0.saturating_sub(1),
            kbd::KEY_RIGHT => self.sel.0 = (self.sel.0 + 1).min(W - 1),
            kbd::KEY_UP => self.sel.1 = self.sel.1.saturating_sub(1),
            kbd::KEY_DOWN => self.sel.1 = (self.sel.1 + 1).min(H - 1),
            b'\n' | b'\r' | b' ' => self.reveal(self.sel.0, self.sel.1),
            b'f' | b'F' => self.toggle_flag(self.sel.0, self.sel.1),
            b'n' | b'N' => self.reset(),
            _ => return false,
        }
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let f = Self::face_rect(client);
        if x >= f.x as i32 && y >= f.y as i32 && x < (f.x + f.w) as i32 && y < (f.y + f.h) as i32 {
            self.reset();
            return true;
        }
        if let Some((cx, cy)) = Self::cell_at(client, x, y) {
            self.sel = (cx, cy);
            self.reveal(cx, cy);
            return true;
        }
        false
    }

    fn right_press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        if let Some((cx, cy)) = Self::cell_at(client, x, y) {
            self.sel = (cx, cy);
            self.toggle_flag(cx, cy);
            return true;
        }
        false
    }
}
