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

/// One cell: a glyph index and a palette colour, packed into two bytes.
///
/// Packed rather than a `char` beside a `u8`, and the reason is *where* this
/// grid is built. `console::init` runs twelve lines before `cpu::idt::init`,
/// so it is constructed at a point in boot where a fault is a triple fault:
/// no message, no register dump, an instant reboot. Two consoles of 128x72
/// cells cost 36 KB today, and a `char` would take that to 147 KB and put an
/// extra 18 KB temporary on the boot stack in exactly the window where
/// running out of stack explains nothing at all. Twelve bits address four
/// thousand glyphs against the three hundred and twenty-five this font draws,
/// so nothing that was going to be used is being given up.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell(u16);

impl Cell {
    const fn new(glyph: u16, fg: u8) -> Self {
        Cell((glyph << 4) | (fg & 0x0F) as u16)
    }

    #[inline]
    fn glyph(self) -> u16 {
        self.0 >> 4
    }

    #[inline]
    fn fg(self) -> u8 {
        (self.0 & 0x0F) as u8
    }
}

const BLANK: Cell = Cell::new(font::SPACE, LTGRAY);

/// Twelve bits of index and four of colour. Asserted rather than remembered:
/// adding the three-hundredth glyph is safe and adding the four-thousandth is
/// not, and the failure is silent corruption of every cell rather than an
/// error anybody would see.
const _: () = assert!(font::MAX_INDEX < 0x1000);

/// What one byte did to the decoder.
enum Step {
    /// Part of a sequence. Nothing to draw yet.
    More,
    Emit(char),
    /// Malformed. The caller draws one replacement box.
    Bad,
    /// Malformed, and this byte begins something valid. Both are drawn.
    BadThen(char),
}

/// Incremental UTF-8, because this console is fed one byte at a time.
///
/// `write_bytes` is handed the bytes of a `&str` and could decode the whole
/// thing, but `put_char` is also the keyboard's path and the recovery
/// console's, and a decoder that only worked on complete strings would leave
/// those two able to print half a character. So the state lives here and
/// every byte goes through the same door.
///
/// Overlong forms are refused rather than decoded and then judged: an
/// overlong sequence decodes to a perfectly ordinary codepoint, so a check
/// after the fact has to remember to make it, and the one place it is easy to
/// forget is the one place it matters. `min` is the smallest codepoint a
/// sequence of that length is allowed to carry.
#[derive(Clone, Copy)]
struct Utf8 {
    acc: u32,
    need: u8,
    min: u32,
}

impl Utf8 {
    const IDLE: Utf8 = Utf8 { acc: 0, need: 0, min: 0 };

    fn start(&mut self, b: u8) -> Step {
        match b {
            0x00..=0x7F => Step::Emit(b as char),
            0xC2..=0xDF => {
                *self = Utf8 { acc: (b & 0x1F) as u32, need: 1, min: 0x80 };
                Step::More
            }
            0xE0..=0xEF => {
                *self = Utf8 { acc: (b & 0x0F) as u32, need: 2, min: 0x800 };
                Step::More
            }
            0xF0..=0xF4 => {
                *self = Utf8 { acc: (b & 0x07) as u32, need: 3, min: 0x10000 };
                Step::More
            }
            // 0xC0 and 0xC1 can only ever open an overlong encoding of an
            // ASCII character, and 0xF5 upwards is past the last codepoint
            // there is. Neither has a valid continuation, so neither is
            // worth carrying state for.
            _ => Step::Bad,
        }
    }

    fn feed(&mut self, b: u8) -> Step {
        if self.need == 0 {
            return self.start(b);
        }
        if b & 0xC0 == 0x80 {
            self.acc = (self.acc << 6) | (b & 0x3F) as u32;
            self.need -= 1;
            if self.need > 0 {
                return Step::More;
            }
            let cp = self.acc;
            let min = self.min;
            *self = Utf8::IDLE;
            return match char::from_u32(cp) {
                Some(c) if cp >= min => Step::Emit(c),
                _ => Step::Bad,
            };
        }
        // The sequence was cut short. Report it and then start again *on this
        // byte* rather than consuming it: swallowing it would eat the newline
        // that ends a truncated line, and with it the line after that.
        *self = Utf8::IDLE;
        match self.start(b) {
            Step::Emit(c) => Step::BadThen(c),
            _ => Step::Bad,
        }
    }
}

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
    /// Half-decoded multi-byte character, if the last byte was inside one.
    utf8: Utf8,
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
            utf8: Utf8::IDLE,
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

    /// Paint one cell, and tell the compositor about it.
    fn draw_cell(&self, r: usize, c: usize) {
        if !self.visible {
            return;
        }
        self.paint_cell(r, c);
        if self.flush {
            let s = self.scale;
            let ox = self.ox + c as u32 * font::GLYPH_W * s;
            let oy = self.oy + r as u32 * font::GLYPH_H * s;
            super::compose::flush_rect(ox, oy, font::GLYPH_W * s, font::GLYPH_H * s);
        }
    }

    /// The pixels alone. Split out because `redraw_all` flushes the whole
    /// console once at the end, and a flush per cell inside that is a
    /// per-row diff repeated a thousand times to say what one call already
    /// said.
    fn paint_cell(&self, r: usize, c: usize) {
        // While the boot screen owns the framebuffer, text still updates the
        // shadow grid and simply is not painted. Nothing is lost: `redraw_all`
        // brings the whole log back the moment the splash hands over, so the
        // boot output is there to read exactly as it always was.
        if !self.visible {
            return;
        }
        let cell = self.cells[r][c];
        let rows = font::rows(cell.glyph());
        let fg = self.fb.encode(PALETTE[cell.fg() as usize]);
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
    /// Repaint every cell.
    ///
    /// **Measured at 1,672 us of a 2,376 us frame -- seventy per cent of a
    /// repaint, for a terminal that is mostly empty.** `draw_cell` renders an
    /// 8x8 glyph whatever the character is, so a blank cell writes 64 scaled
    /// pixels of background one `put` at a time, and `desk::draw` had just
    /// filled that same background with a bulk span fill immediately before.
    /// About 1.2 million per-pixel stores per frame, nearly all of them
    /// writing the colour that was already there.
    ///
    /// So the background is painted once, as spans, and only cells that have
    /// something in them are rendered. The output is identical -- a blank cell
    /// *is* the background -- and the survey that ranked this work missed it
    /// because it looked for per-cell dirty tracking, which `desk::draw`
    /// defeats by erasing the client area every frame. The redundancy was
    /// never in repainting cells that had not changed. It was in painting
    /// nothing, pixel by pixel.
    ///
    /// Painting its own background also makes this correct standalone, where
    /// before it depended on whoever called it having filled the area first.
    pub fn redraw_all(&self) {
        if !self.visible {
            return;
        }
        // The same rectangle `clear` uses, from the same helper, so the two
        // cannot drift about what "the console" means.
        let (w, h) = self.pixel_size();
        self.fb.rect(self.ox, self.oy, w, h, self.bg);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.cells[r][c].glyph() == font::SPACE {
                    continue;
                }
                self.paint_cell(r, c);
            }
        }
        // One flush for the whole console rather than one per cell. The
        // per-cell path stays for `draw_cell` on its own, which is what a
        // single typed character takes.
        if self.flush {
            super::compose::flush_rect(self.ox, self.oy, w, h);
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

    /// One byte of a UTF-8 stream.
    ///
    /// Still bytes and not `char`, because the two callers outside this file
    /// are a keyboard interrupt and the recovery console, and neither has a
    /// decoded character to hand. A byte that completes nothing draws
    /// nothing, which is the only visible difference from before: a two-byte
    /// character used to draw two boxes and now draws one letter.
    pub fn put_char(&mut self, ch: u8) {
        match self.utf8.feed(ch) {
            Step::More => {}
            Step::Emit(c) => self.put_cp(c),
            Step::Bad => self.put_glyph(font::UNKNOWN),
            Step::BadThen(c) => {
                self.put_glyph(font::UNKNOWN);
                self.put_cp(c);
            }
        }
    }

    /// One decoded character.
    pub fn put_cp(&mut self, c: char) {
        match c {
            '\n' => {
                self.newline();
                return;
            }
            '\r' => {
                self.col = 0;
                return;
            }
            '\t' => {
                let next = (self.col + 4) & !3;
                while self.col < next && self.col < self.cols {
                    self.put_cp(' ');
                }
                return;
            }
            '\u{8}' => {
                // Backspace: erase in place so the shell can edit a line.
                if self.col > 0 {
                    self.col -= 1;
                    self.cells[self.row][self.col] = Cell::new(font::SPACE, self.fg);
                    self.draw_cell(self.row, self.col);
                }
                return;
            }
            _ => {}
        }
        self.put_glyph(font::index_of(c));
    }

    fn put_glyph(&mut self, glyph: u16) {
        if self.col >= self.cols {
            self.newline();
        }
        self.cells[self.row][self.col] = Cell::new(glyph, self.fg);
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

/// The operator's console: the prompt, and what commands print.
pub const USER: usize = 0;
/// The machine's console: boot, background tasks, and episodes it decided to
/// run on its own.
///
/// Split because they were one, and being one is what "the desktop freezes"
/// mostly turned out to mean. `initiative` wakes on its own schedule and
/// everything its episode printed arrived between a prompt and a half-typed
/// command, scrolling the answer to the last question off the top. Nothing was
/// frozen; two things were talking into the same grid.
///
/// Two grids and not two terminals-with-shells: this one is output. What would
/// it accept? Every command already runs as the operator, there being no user
/// or kernel to be root of, so an "executive" prompt would be the same prompt
/// against the same namespace with a different frame around it. It is named
/// for what it carries rather than for a privilege this system does not have.
pub const EXEC: usize = 1;
pub const NCONSOLE: usize = 2;

/// Every core prints through this, so it is a lock rather than a promise.
///
/// `lock_irq`, because the clock task prints from a timer tick: a handler that
/// spun for a lock the code it interrupted was holding would never get it.
/// Masking for the length of a line makes that impossible rather than rare.
///
/// The paint path underneath must never print, or it would take this lock
/// twice on one core. That is a real constraint and it is checkable: the
/// deadlock panic names the waiter and the lock instead of hanging, so a
/// violation announces itself at the first line of boot.
static CONSOLES: crate::sync::Spin<[Option<Console>; NCONSOLE]> =
    crate::sync::Spin::new([None, None]);

/// Where `kprint!` lands when nothing more specific applies.
///
/// Starts on the executive console, because everything before the shell exists
/// -- the memory map, the selftests, the model load -- is the machine
/// reporting on itself, and that is exactly what the executive console is for.
/// `shell::run` moves it to the operator's.
static CURRENT: Racy<usize> = Racy::new(EXEC);

/// Install both consoles. Call once, after `ExitBootServices`.
pub fn init(fb: Framebuffer, scale: u32, bg: Color) {
    let all = &mut *CONSOLES.lock_irq();
    for slot in all.iter_mut() {
        let mut console = Console::new(fb.clone(), scale, bg);
        console.clear();
        *slot = Some(console);
    }
}

pub fn is_ready() -> bool {
    CONSOLES.lock_irq()[USER].is_some()
}

/// Which console `kprint!` is writing to right now.
///
/// Asked per line rather than latched, because the answer depends on which
/// task is running and that changes under preemption. An episode the machine
/// chose to run is the whole reason this function is not simply `CURRENT`.
pub fn channel() -> usize {
    if let Some((task, ch)) = unsafe { *FORCE.get() } {
        if crate::task::current() == task {
            return ch;
        }
    }
    if crate::ai::agent::printing_in_background() {
        return EXEC;
    }
    unsafe { *CURRENT.get() }
}

/// A channel override for the duration of `f`.
///
/// For code that is the machine talking to itself but does not run on the
/// agent task -- `initiative::tick` is the one that matters, since it
/// announces every decision it makes and those announcements are what an
/// operator was reading between their own commands.
///
/// Records the task that asked as well as the channel, and applies only to
/// that task.
///
/// It was global first, on the reasoning that the window was one `kprint!`
/// wide. It is not: `initiative::tick` holds the override across everything it
/// decides and queues, and the shell task preempted into the middle of that
/// sent an operator's command output to the executive grid, where it looked
/// exactly like the command having silently done nothing. Scoping it to the
/// task costs one comparison and makes preemption a non-event.
static FORCE: Racy<Option<(usize, usize)>> = Racy::new(None);

pub fn on_channel<R>(ch: usize, f: impl FnOnce() -> R) -> R {
    let prev = unsafe { *FORCE.get() };
    unsafe { *FORCE.get() = Some((crate::task::current(), ch)) };
    let r = f();
    unsafe { *FORCE.get() = prev };
    r
}

/// Move the default channel. Called once, when the shell takes over.
pub fn set_default_channel(ch: usize) {
    unsafe { *CURRENT.get() = ch.min(NCONSOLE - 1) };
}

pub fn with_ch<F: FnOnce(&mut Console)>(ch: usize, f: F) {
    if let Some(c) = CONSOLES.lock_irq().get_mut(ch).and_then(|s| s.as_mut()) {
        f(c);
    }
}

/// Run `f` against the global console if it exists.
pub fn with<F: FnOnce(&mut Console)>(f: F) {
    with_ch(channel(), f)
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

pub fn redraw_ch(ch: usize) {
    with_ch(ch, |c| c.redraw_all());
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
//
// The capture is a stack, not a flag. A lang program run by the agent loop
// can itself call `applet(...)`, which dispatches and captures in turn --
// with a single flag the inner begin would discard the outer episode's
// collection mid-flight. Nesting is the difference between a tool that can
// call the OS and one that cannot.

/// Capture nests per pipeline and is reached from whichever core is running
/// the shell, so it is behind the same kind of lock for the same reason.
static CAPTURE: crate::sync::Spin<alloc::vec::Vec<alloc::string::String>> =
    crate::sync::Spin::new(alloc::vec::Vec::new());

pub fn begin_capture() {
    CAPTURE.lock_irq().push(alloc::string::String::new());
}

/// Stop the innermost capture and return what it collected.
pub fn end_capture() -> Option<alloc::string::String> {
    CAPTURE.lock_irq().pop()
}

pub fn capturing() -> bool {
    !CAPTURE.lock_irq().is_empty()
}

#[doc(hidden)]
pub fn _print(args: fmt::Arguments) {
    use fmt::Write;
    {
        let mut stack = CAPTURE.lock_irq();
        if let Some(buf) = stack.last_mut() {
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
        // The third sink. The console keeps one screen and the serial port
        // needs somebody listening on the other end; neither survives a boot
        // on the laptop, which is the machine whose boot lines matter.
        $crate::log::_record(format_args!($($arg)*));
    }};
}

#[macro_export]
macro_rules! kprintln {
    () => { $crate::kprint!("\n") };
    ($($arg:tt)*) => { $crate::kprint!("{}\n", format_args!($($arg)*)) };
}

/// The decoder, checked apart from any framebuffer.
///
/// Deliberately against `Utf8` and not against a `Console`: a console needs a
/// framebuffer, and what is being claimed here is arithmetic on bytes. The
/// cases that earn their place are the malformed ones, because a decoder that
/// only handles good input is a decoder nobody has tested.
pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    /// Feed a byte string and collect what came out, with '\u{FFFD}' standing
    /// for each replacement the console would have drawn.
    fn decode(bytes: &[u8]) -> alloc::string::String {
        let mut d = Utf8::IDLE;
        let mut out = alloc::string::String::new();
        for &b in bytes {
            match d.feed(b) {
                Step::More => {}
                Step::Emit(c) => out.push(c),
                Step::Bad => out.push('\u{FFFD}'),
                Step::BadThen(c) => {
                    out.push('\u{FFFD}');
                    out.push(c);
                }
            }
        }
        out
    }

    claim(&mut ok, decode(b"plain ascii") == "plain ascii", "ASCII goes through untouched");
    claim(&mut ok, decode("café".as_bytes()) == "café", "a two-byte character is one character");
    claim(&mut ok, decode("┌─┐".as_bytes()) == "┌─┐", "and a three-byte one is too");
    claim(&mut ok, decode("𝄞".as_bytes()) == "𝄞", "and a four-byte one, which nothing here can draw");

    // The case that matters most, and the one a decoder written in a hurry
    // gets wrong: a sequence cut off mid-way must not swallow what follows
    // it. Consuming the newline here loses the rest of the line as well.
    claim(&mut ok, decode(b"a\xC3\nb") == "a\u{FFFD}\nb",
        "a truncated sequence reports itself and gives the next byte back");

    claim(&mut ok, decode(b"\xC0\xAF") == "\u{FFFD}\u{FFFD}",
        "an overlong '/' is refused, so it cannot become a path separator");
    claim(&mut ok, decode(b"\xE0\x80\xAF") == "\u{FFFD}",
        "and so is the three-byte spelling of the same trick");
    claim(&mut ok, decode(b"\xED\xA0\x80") == "\u{FFFD}", "a surrogate is not a character");
    claim(&mut ok, decode(b"\xF5\x80\x80\x80") == "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}",
        "nor is anything above the last codepoint");
    claim(&mut ok, decode(b"\x80\x80") == "\u{FFFD}\u{FFFD}", "a stray continuation byte is not a character");

    // A cell is two bytes and the grid is the largest static structure here,
    // so this is checked rather than assumed: it is the reason the cell packs
    // at all, and `console::init` runs before there is an interrupt table to
    // report having run out of stack.
    claim(&mut ok, core::mem::size_of::<Cell>() == 2, "a cell is still two bytes");
    let c = Cell::new(font::index_of('é'), YELLOW);
    claim(&mut ok, c.glyph() == font::index_of('é') && c.fg() == YELLOW,
        "and packing a glyph beside a colour loses neither");
    ok
}
