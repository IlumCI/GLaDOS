//! Scrolling text console on the linear framebuffer.
//!
//! Keeps a shadow character grid in RAM and re-renders on scroll. The
//! alternative -- blitting the framebuffer up a row -- means *reading* video
//! memory, which is painfully slow over the uncached MMIO aperture on real
//! hardware. Re-rendering from RAM only touches the framebuffer with writes.

use super::font;
use super::{Color, Framebuffer};
use crate::sync::Racy;
use core::fmt;

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
}

impl Console {
    pub fn new(fb: Framebuffer, scale: u32, bg: Color) -> Self {
        let scale = scale.max(1);
        let cols = ((fb.width() / (font::GLYPH_W * scale)) as usize).min(MAX_COLS);
        let rows = ((fb.height() / (font::GLYPH_H * scale)) as usize).min(MAX_ROWS);
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
        }
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
        self.fb.fill(self.bg);
    }

    fn draw_cell(&self, r: usize, c: usize) {
        let cell = self.cells[r][c];
        let rows = font::glyph(cell.ch);
        let fg = self.fb.encode(PALETTE[(cell.fg & 0x0F) as usize]);
        let bg = self.fb.encode(self.bg);
        let s = self.scale;
        let ox = c as u32 * font::GLYPH_W * s;
        let oy = r as u32 * font::GLYPH_H * s;

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
    }

    fn redraw_all(&self) {
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
        self.redraw_all();
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

    pub fn write_bytes(&mut self, s: &[u8]) {
        for &b in s {
            self.put_char(b);
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

pub fn set_color(fg: u8) {
    with(|c| c.set_color(fg));
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use fmt::Write;
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
