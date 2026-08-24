//! Scrolling text console on the linear framebuffer.
//!
//! Keeps a shadow character grid in RAM, which is what makes redrawing a cell
//! possible without reading back what is on screen.
//!
//! Scrolling used to re-render every cell from that grid, on the reasoning that
//! blitting the framebuffer up a row means *reading* video memory, which is
//! ruinous across an uncached MMIO aperture. That reasoning was sound when it
//! was written and stopped being true two milestones later: making all non-RAM
//! uncacheable (to fix the IOAPIC reporting 120 redirection entries) forced an
//! explicit write-back carve-out for the framebuffer in `build_identity_map`,
//! and nothing came back here to say so. The cost was roughly two million
//! serialised volatile stores per newline -- visible, at about ten lines a
//! second on a 1920x1080 panel. `Framebuffer::scroll_up` now does it as one
//! memmove.
//!
//! Pacing is therefore now deliberate rather than accidental: see `set_pace`.

use super::font;
use super::{Color, Framebuffer};
use crate::sync::Racy;
use core::fmt;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Console colours, indexed by the 4-bit attribute stored per cell.
pub const PALETTE: [Color; 16] = [
    super::palette::BLACK,
    super::palette::BLUE,
    super::palette::GREEN,
    super::palette::CYAN,
    super::palette::RED,
    super::palette::MAGENTA,
    super::palette::BROWN,
    super::palette::LTGRAY,
    super::palette::DKGRAY,
    super::palette::LTBLUE,
    super::palette::LTGREEN,
    super::palette::LTCYAN,
    super::palette::LTRED,
    super::palette::LTMAGENTA,
    super::palette::YELLOW,
    super::palette::WHITE,
];

pub const WHITE: u8 = 15;
pub const LTGRAY: u8 = 7;
pub const YELLOW: u8 = 14;
pub const LTRED: u8 = 12;
pub const LTGREEN: u8 = 10;
pub const LTCYAN: u8 = 11;

// 128x72 cells covers 2048x1152 at 2x scale, so it is enough for this panel's
// native 1920x1080 with room to spare. Sized as a fixed array because there is
// no heap until M3.
const MAX_COLS: usize = 128;
const MAX_ROWS: usize = 72;

#[derive(Clone, Copy)]
struct Cell {
    ch: u8,
    fg: u8,
}

const BLANK: Cell = Cell { ch: b' ', fg: LTGRAY };

pub struct Console {
    fb: Framebuffer,
    cells: [[Cell; MAX_COLS]; MAX_ROWS],
    cols: usize,
    rows: usize,
    col: usize,
    row: usize,
    fg: u8,
    bg: Color,
    scale: u32,
    /// Pixel origin of the character grid.
    ///
    /// The console no longer owns the panel: it sits in the client area of a
    /// window, and everything that used to be an absolute coordinate is now
    /// relative to here. Zero reproduces the old behaviour exactly, which is
    /// what boot uses before the chrome is drawn.
    ox: u32,
    oy: u32,
    /// False while the boot screen owns the framebuffer.
    visible: bool,
    /// Whether painted cells are pushed straight to the screen.
    ///
    /// Set when `fb` is the compositor's back buffer: the console prints on
    /// the shell's schedule, between desktop draws, and a cell that waited for
    /// the next `present` would make typing invisible. False before the
    /// compositor exists, when `fb` *is* the screen and there is nothing to
    /// push to.
    flush: bool,
}

impl Console {
    pub fn new(fb: Framebuffer, scale: u32, bg: Color) -> Self {
        let (w, h) = (fb.width(), fb.height());
        Self::new_in(fb, scale, bg, 0, 0, w, h)
    }

    /// A console occupying one rectangle of the framebuffer.
    pub fn new_in(fb: Framebuffer, scale: u32, bg: Color, x: u32, y: u32, w: u32, h: u32) -> Self {
        let scale = scale.max(1);
        let cols = ((w / (font::GLYPH_W * scale)) as usize).min(MAX_COLS);
        let rows = ((h / (font::GLYPH_H * scale)) as usize).min(MAX_ROWS);
        Self {
            fb,
            cells: [[BLANK; MAX_COLS]; MAX_ROWS],
            cols,
            rows,
            col: 0,
            row: 0,
            fg: LTGRAY,
            bg,
            scale,
            ox: x,
            oy: y,
            visible: true,
            flush: false,
        }
    }

    /// Repoint the console at a different buffer.
    ///
    /// Called once, when the compositor comes up: from then on the console
    /// paints into the back buffer like everything else, and pushes each cell
    /// through itself. The shadow grid is untouched -- the text survives, and
    /// the next full draw repaints it wherever the terminal window is.
    pub fn retarget(&mut self, fb: Framebuffer, flush: bool) {
        self.fb = fb;
        self.flush = flush;
    }

    /// Move the grid into a new rectangle, keeping its contents.
    ///
    /// Used once, when the boot screen hands over and the terminal gains its
    /// window. Text written during boot is in the shadow grid and is repainted
    /// at the new origin, so the log survives the move.
    pub fn reflow(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let cols = ((w / (font::GLYPH_W * self.scale)) as usize).min(MAX_COLS);
        let rows = ((h / (font::GLYPH_H * self.scale)) as usize).min(MAX_ROWS);
        // Keep the tail rather than the head: the interesting part of a boot
        // log is the end of it, and a shrinking grid would otherwise scroll the
        // most recent lines off the bottom.
        if self.row >= rows {
            let drop = self.row + 1 - rows;
            for r in 0..rows {
                self.cells[r] = self.cells[r + drop];
            }
            for r in rows..MAX_ROWS {
                self.cells[r] = [BLANK; MAX_COLS];
            }
            self.row -= drop;
        }
        self.cols = cols;
        self.rows = rows;
        self.col = self.col.min(cols.saturating_sub(1));
        self.ox = x;
        self.oy = y;
    }

    /// Pixel size of the grid as currently laid out.
    pub fn pixel_size(&self) -> (u32, u32) {
        (
            self.cols as u32 * font::GLYPH_W * self.scale,
            self.rows as u32 * font::GLYPH_H * self.scale,
        )
    }

    pub fn set_color(&mut self, fg: u8) {
        self.fg = fg & 0x0F;
    }

    pub fn clear(&mut self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.cells[r][c] = BLANK;
            }
        }
        self.col = 0;
        self.row = 0;
        if self.visible {
            // The grid's own rectangle, not the panel: clearing the screen must
            // not erase the window around it.
            let (w, h) = self.pixel_size();
            self.fb.rect(self.ox, self.oy, w, h, self.bg);
            if self.flush {
                super::compose::flush_rect(self.ox, self.oy, w, h);
            }
        }
    }

    fn draw_cell(&self, r: usize, c: usize) {
        // While the boot screen owns the framebuffer, text still updates the
        // shadow grid and simply is not painted. Nothing is lost: `redraw_all`
        // brings the whole log back the moment the splash hands over, so the
        // boot output is there to read exactly as it always was.
        if !self.visible {
            return;
        }
        let cell = self.cells[r][c];
        let rows = font::glyph(cell.ch);
        let fg = self.fb.encode(PALETTE[(cell.fg & 0x0F) as usize]);
        let bg = self.fb.encode(self.bg);
        let s = self.scale;
        let ox = self.ox + c as u32 * font::GLYPH_W * s;
        let oy = self.oy + r as u32 * font::GLYPH_H * s;

        for (gy, bits) in rows.iter().enumerate() {
            for gx in 0..font::GLYPH_W {
                // Bit 7 is the leftmost pixel.
                let on = bits & (0x80 >> gx) != 0;
                let raw = if on { fg } else { bg };
                let px = ox + gx * s;
                let py = oy + gy as u32 * s;
                for dy in 0..s {
                    for dx in 0..s {
                        self.fb.put(px + dx, py + dy, raw);
                    }
                }
            }
        }
        if self.flush {
            super::compose::flush_rect(ox, oy, font::GLYPH_W * s, font::GLYPH_H * s);
        }
    }

    /// Hand the framebuffer to something else, or take it back.
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v;
    }

    /// Repaint every cell from the shadow grid.
    ///
    /// The way back from anything that drew over the console -- the boot
    /// screen, or the language`s `rect` builtin, which writes straight to the
    /// framebuffer and will happily scribble over text.
    pub fn redraw_all(&self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                self.draw_cell(r, c);
            }
        }
    }

    fn scroll(&mut self) {
        for r in 1..self.rows {
            self.cells[r - 1] = self.cells[r];
        }
        for c in 0..self.cols {
            self.cells[self.rows - 1][c] = BLANK;
        }
        self.row = self.rows - 1;
        self.col = 0;

        // Shift pixels instead of re-rendering the grid, over the console's own
        // rectangle rather than the whole panel -- the panel height is rarely a
        // whole number of rows, and the console now has a window frame beside
        // it that must not be dragged upwards.
        let cell_h = font::GLYPH_H * self.scale;
        if self.visible {
            let (w, h) = self.pixel_size();
            self.fb.scroll_rect(self.ox, self.oy, w, h, cell_h, self.bg);
            if self.flush {
                super::compose::flush_rect(self.ox, self.oy, w, h);
            }
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        self.row += 1;
        if self.row >= self.rows {
            self.scroll();
        }
    }

    pub fn put_char(&mut self, ch: u8) {
        match ch {
            b'\n' => {
                self.newline();
                return;
            }
            b'\r' => {
                self.col = 0;
                return;
            }
            b'\t' => {
                let next = (self.col + 4) & !3;
                while self.col < next && self.col < self.cols {
                    self.put_char(b' ');
                }
                return;
            }
            8 => {
                // Backspace: erase in place so the shell can edit a line.
                if self.col > 0 {
                    self.col -= 1;
                    self.cells[self.row][self.col] = Cell { ch: b' ', fg: self.fg };
                    self.draw_cell(self.row, self.col);
                }
                return;
            }
            _ => {}
        }

        if self.col >= self.cols {
            self.newline();
        }
        self.cells[self.row][self.col] = Cell { ch, fg: self.fg };
        self.draw_cell(self.row, self.col);
        self.col += 1;
    }

    /// Move the cursor within the current row without disturbing what is drawn.
    ///
    /// Backspace erases, which is right for typing but wrong for a line editor
    /// that needs to reposition and redraw. This is the primitive that makes
    /// arrow keys possible.
    pub fn set_col(&mut self, col: usize) {
        self.col = col.min(self.cols.saturating_sub(1));
    }

    pub fn col(&self) -> usize {
        self.col
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bulk output. This is the path `kprintln!` takes, and the only one that
    /// is paced -- `put_char` stays immediate because the shell's line editor
    /// uses it to echo keystrokes and reposition the cursor, and pacing those
    /// would just feel like input lag.
    pub fn write_bytes(&mut self, s: &[u8]) {
        let pace = pace_us();
        for &b in s {
            self.put_char(b);
            if pace != 0 && !skip_requested() {
                crate::time::delay_us(pace);
            }
        }
    }
}

impl fmt::Write for Console {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_bytes(s.as_bytes());
        Ok(())
    }
}

static CONSOLE: Racy<Option<Console>> = Racy::new(None);

/// Install the global console. Call once, after `ExitBootServices`.
pub fn init(fb: Framebuffer, scale: u32, bg: Color) {
    let mut console = Console::new(fb, scale, bg);
    console.clear();
    unsafe { *CONSOLE.get() = Some(console) }
}

pub fn is_ready() -> bool {
    unsafe { CONSOLE.get().is_some() }
}

/// Run `f` against the global console if it exists.
pub fn with<F: FnOnce(&mut Console)>(f: F) {
    if let Some(c) = unsafe { CONSOLE.get().as_mut() } {
        f(c);
    }
}

// --- pacing -------------------------------------------------------------
//
// With the scroll fixed, output is far faster than anyone can read. The old
// pace was an artefact of a bug; this makes the same feel a choice, and a
// tunable one. Kept as a global rather than a `Console` field so that
// `write_bytes` can read it without the caller having to thread it through.

/// Microseconds per character. Roughly reproduces the pace the broken scroll
/// used to impose, which is the point -- it looked right.
const DEFAULT_PACE_US: u64 = 1200;

static PACE_US: AtomicU64 = AtomicU64::new(DEFAULT_PACE_US);
static SKIP: AtomicBool = AtomicBool::new(false);

pub fn pace_us() -> u64 {
    PACE_US.load(Ordering::Relaxed)
}

/// 0 disables pacing entirely.
pub fn set_pace(us: u64) {
    PACE_US.store(us, Ordering::Relaxed);
}

/// Re-arm pacing. The shell calls this before each command, so a skip only
/// ever applies to the output it was asked to skip.
pub fn resume_pacing() {
    SKIP.store(false, Ordering::Relaxed);
}

/// Has the operator asked to stop waiting?
///
/// A paced `tree` of a few hundred entries would otherwise hold the console
/// hostage for a minute with no way out, which is the failure mode that makes
/// deliberately slow output intolerable rather than charming. Any keystroke
/// drops the pacing for the remainder of the current command; the keystroke
/// itself stays in the buffer and is read as normal input afterwards.
fn skip_requested() -> bool {
    if SKIP.load(Ordering::Relaxed) {
        return true;
    }
    if crate::dev::kbd::has_input() {
        SKIP.store(true, Ordering::Relaxed);
        return true;
    }
    false
}

pub fn redraw() {
    with(|c| c.redraw_all());
}

pub fn set_color(fg: u8) {
    with(|c| c.set_color(fg));
}

pub fn set_col(col: usize) {
    with(|c| c.set_col(col));
}

pub fn cols() -> usize {
    let mut n = 80;
    with(|c| n = c.cols());
    n
}

// --- capture ------------------------------------------------------------
//
// Applets print rather than return, which is fine for a person reading them
// and useless for feeding one command into another. Rather than rewriting
// twenty applets to produce values, the console can be told to collect what it
// is given instead of drawing it -- so `tree | grep ai` and `snaps > /log`
// work without any applet knowing it is being redirected.
//
// Serial still receives everything. A capture is about what the operator sees,
// not about hiding output from the debug channel.

static CAPTURE: Racy<Option<alloc::string::String>> = Racy::new(None);

pub fn begin_capture() {
    unsafe { *CAPTURE.get() = Some(alloc::string::String::new()) };
}

/// Stop capturing and return what was collected.
pub fn end_capture() -> Option<alloc::string::String> {
    unsafe { CAPTURE.get().take() }
}

pub fn capturing() -> bool {
    unsafe { CAPTURE.get().is_some() }
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use fmt::Write;
    unsafe {
        if let Some(buf) = CAPTURE.get().as_mut() {
            // Capped. A capture wraps arbitrary applet output, and an applet
            // that prints forever -- a tool with a bad loop, a tree of a deep
            // namespace -- would otherwise grow memory without bound while
            // its caller is waiting on the capture to end. The cap is larger
            // than any legitimate observation and smaller than a heap.
            const CAPTURE_MAX: usize = 64 * 1024;
            if buf.len() < CAPTURE_MAX {
                let _ = buf.write_fmt(args);
            }
            return;
        }
    }
    with(|c| {
        let _ = c.write_fmt(args);
    });
}

/// Write to both the framebuffer console and COM1.
///
/// On the GF63 the serial half goes nowhere -- there is no UART -- but under
/// QEMU it gives us scrollback and a copyable transcript, which the framebuffer
/// cannot.
#[macro_export]
macro_rules! kprint {
    ($($arg:tt)*) => {{
        $crate::gfx::console::_print(format_args!($($arg)*));
        $crate::serial::_print(format_args!($($arg)*));
    }};
}

#[macro_export]
macro_rules! kprintln {
    () => { $crate::kprint!("\n") };
    ($($arg:tt)*) => { $crate::kprint!("{}\n", format_args!($($arg)*)) };
}
