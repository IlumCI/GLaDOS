//! Linear framebuffer.
//!
//! UEFI hands us a physical address, a resolution, a pixel format and a stride.
//! There is no VGA text mode to fall back on -- this machine boots UEFI with no
//! CSM, so there is no INT 10h, no 0xB8000 text buffer, and no VGA BIOS. Pixels
//! are the only output device that exists.

pub mod browse;
pub mod compose;
pub mod console;
pub mod mines;
pub mod oracle;
pub mod paint;
pub mod write;
pub mod splash;
pub mod font;
pub mod theme;
pub mod todo;
pub mod ui;
pub mod desk;
pub mod agentwin;

/// A program that owns a window's client area.
///
/// The third kind of window content, after the terminal and the widget
/// stack. A `Panel` is a form -- labels, lists, buttons in a column -- and
/// three programs in a row (a canvas, a document, a minefield) turned out
/// not to be forms. Rather than grow the widget enum until it could express
/// a canvas, a window can hold anything that answers these six questions.
/// `Browser` predates the trait and stays a named variant; these are the
/// same idea generalised.
///
/// `draw_in` takes `&self` because the desktop paints from a shared borrow;
/// layout facts discovered while drawing go in `Cell`s, the way the Browser
/// keeps its row count. Every handler returns whether it consumed the event,
/// so unclaimed keys keep falling through to the window manager -- Alt-Tab
/// must work inside a game.
pub trait DeskApp {
    fn draw_in(&self, fb: &Framebuffer, client: theme::Rect, focused: bool);
    fn key(&mut self, k: u8) -> bool;
    fn press(&mut self, client: theme::Rect, x: i32, y: i32) -> bool;
    /// The second button. Minesweeper flags with it; most programs decline
    /// it and the desktop opens the system menu instead.
    fn right_press(&mut self, _client: theme::Rect, _x: i32, _y: i32) -> bool {
        false
    }
    /// Pointer motion while the left button is held on this window --
    /// what a brush stroke is made of.
    fn drag(&mut self, _client: theme::Rect, _x: i32, _y: i32) -> bool {
        false
    }
    fn release(&mut self) -> bool {
        false
    }
    fn wheel(&mut self, _notches: i32) -> bool {
        false
    }

    /// The smallest client area this program stays usable in.
    ///
    /// The window manager asks before letting a resize continue, because the
    /// program is the only thing that knows. A board that does not reflow and
    /// a canvas that does not shrink both have a floor, and a frame dragged
    /// under it hides their content with no way back except maximising.
    ///
    /// The default is small enough not to obstruct anything that genuinely
    /// reflows, and large enough that a window cannot be lost.
    fn min_size(&self) -> (u32, u32) {
        (240, 120)
    }
}

use crate::sync::Racy;

/// The framebuffer, reachable from anywhere -- background tasks draw straight
/// to it rather than going through the console.
static PRIMARY: Racy<Option<Framebuffer>> = Racy::new(None);

pub fn set_primary(fb: Framebuffer) {
    unsafe { *PRIMARY.get() = Some(fb) };
}

pub fn primary() -> Option<Framebuffer> {
    unsafe { *PRIMARY.get() }
}

use core::ptr::write_volatile;

/// Integer square root, Newton. Used per scan line by `fill_circle`.
pub fn isqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// The 16-colour palette, in the spirit of the 640x480x4bpp mode TempleOS used.
/// We have 32-bit colour available and are choosing not to use most of it.
pub mod palette {
    use super::Color;
    pub const BLACK: Color = Color::new(0x00, 0x00, 0x00);
    pub const BLUE: Color = Color::new(0x00, 0x00, 0xAA);
    pub const GREEN: Color = Color::new(0x00, 0xAA, 0x00);
    pub const CYAN: Color = Color::new(0x00, 0xAA, 0xAA);
    pub const RED: Color = Color::new(0xAA, 0x00, 0x00);
    pub const MAGENTA: Color = Color::new(0xAA, 0x00, 0xAA);
    pub const BROWN: Color = Color::new(0xAA, 0x55, 0x00);
    pub const LTGRAY: Color = Color::new(0xAA, 0xAA, 0xAA);
    pub const DKGRAY: Color = Color::new(0x55, 0x55, 0x55);
    pub const LTBLUE: Color = Color::new(0x55, 0x55, 0xFF);
    pub const LTGREEN: Color = Color::new(0x55, 0xFF, 0x55);
    pub const LTCYAN: Color = Color::new(0x55, 0xFF, 0xFF);
    pub const LTRED: Color = Color::new(0xFF, 0x55, 0x55);
    pub const LTMAGENTA: Color = Color::new(0xFF, 0x55, 0xFF);
    pub const YELLOW: Color = Color::new(0xFF, 0xFF, 0x55);
    pub const WHITE: Color = Color::new(0xFF, 0xFF, 0xFF);

    /// The one colour outside the sixteen, and a deliberate exception.
    ///
    /// The palette above is a choice, not a limit -- the framebuffer is 32-bit
    /// and we are declining to use it. A brand mark is the one thing that has
    /// to be a specific colour rather than the nearest of sixteen: BROWN
    /// (0xAA5500) is the closest available and reads as mud.
    pub const AMBER: Color = Color::new(0xDD, 0xA3, 0x3C);
}

/// Pixel encodings we can drive. `BltOnly` is not here on purpose: it means
/// there is no linear framebuffer at all, and the Blt() call that would be the
/// only way to draw lives in boot services, which we are about to leave.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    /// Byte order in memory: R, G, B, unused.
    Rgbx,
    /// Byte order in memory: B, G, R, unused. OVMF and Intel iGPUs report this.
    Bgrx,
}

/// A window silhouette, as one horizontal span per row.
///
/// Kolibri gives every pixel on the screen a byte naming the window that owns
/// it, which makes an arbitrary outline free at draw time and costs a
/// screen-sized array rebuilt whenever anything moves. That trade is right for
/// a compositor that writes straight to the screen and has to clip every pixel
/// anyway. It is the wrong one here, where drawing is already a back-to-front
/// repaint into a buffer: a shaped window only has to *not write* outside its
/// outline, and whatever was painted before it shows through by itself.
///
/// One span per row instead of one byte per pixel makes the check per span
/// rather than per pixel, which matters because the span fills are what the
/// last round of work made fast. It costs the ability to describe a window
/// with a hole in it, which nothing wants, and describes every rounded
/// corner, circle and ellipse anybody does.
///
/// Spans are `[x0, x1)` in window-local coordinates. An empty span is a row
/// the window does not occupy at all.
pub struct Shape {
    rows: alloc::vec::Vec<(u32, u32)>,
    w: u32,
}

impl Shape {
    /// A rectangle with its corners rounded off.
    ///
    /// Computed once when the window is sized rather than per frame: the
    /// arithmetic is a square root per row of the corner arc, which is
    /// nothing once and noticeable sixty times a second.
    pub fn round_rect(w: u32, h: u32, r: u32) -> Shape {
        let r = r.min(w / 2).min(h / 2);
        let mut rows = alloc::vec::Vec::with_capacity(h as usize);
        for y in 0..h {
            // Distance from the nearer horizontal edge, inside the corner
            // band. Outside it the row is the full width.
            let dy = if y < r {
                Some(r - 1 - y)
            } else if y >= h - r {
                Some(y - (h - r))
            } else {
                None
            };
            match dy {
                None => rows.push((0, w)),
                Some(d) => {
                    // How far in the arc has come at this row: r - sqrt(r^2 - d^2).
                    let rr = (r * r) as i32;
                    let dd = (d * d) as i32;
                    let inset = r - isqrt((rr - dd).max(0) as u32);
                    let x0 = inset.min(w / 2);
                    rows.push((x0, w.saturating_sub(x0)));
                }
            }
        }
        Shape { rows, w }
    }

    /// The span this row occupies, or `None` for a row outside the shape.
    #[inline]
    pub fn span(&self, y: u32) -> Option<(u32, u32)> {
        let (a, b) = *self.rows.get(y as usize)?;
        if a >= b {
            return None;
        }
        Some((a, b))
    }

    pub fn width(&self) -> u32 {
        self.w
    }

    pub fn height(&self) -> u32 {
        self.rows.len() as u32
    }
}

/// The silhouette drawing is currently confined to, and where its origin sits
/// on screen.
///
/// A static rather than a field on `Framebuffer`, because every widget, panel
/// and application in this tree takes `&Framebuffer` and a field would have to
/// be threaded through all of them to reach the one place it is read. Single
/// core, set around one window's paint and cleared after, which is the same
/// bargain `Racy` is everywhere else here.
///
/// The pointer is borrowed for the duration of that paint and never outlives
/// it; `with_shape` is the only way to set it and restores the previous value,
/// so a nested paint cannot leave someone else's outline installed.
static CLIP: Racy<Option<(*const Shape, i32, i32)>> = Racy::new(None);

/// Draw `f` confined to `shape`, positioned with its top-left at `(x, y)`.
pub fn with_shape<R>(shape: &Shape, x: i32, y: i32, f: impl FnOnce() -> R) -> R {
    let prev = unsafe { *CLIP.get() };
    unsafe { *CLIP.get() = Some((shape as *const Shape, x, y)) };
    let r = f();
    unsafe { *CLIP.get() = prev };
    r
}

/// Intersect a horizontal run with the active silhouette.
///
/// Returns the part of `[x, x+w)` on row `y` that may be written, or `None`
/// when the row is outside the shape entirely. With no shape installed this
/// is a load and a branch, which is what keeps it off the cost of the span
/// fills everything else depends on.
#[inline]
fn clip_run(y: u32, x: u32, w: u32) -> Option<(u32, u32)> {
    let Some((sp, ox, oy)) = (unsafe { *CLIP.get() }) else {
        return Some((x, w));
    };
    let ly = (y as i64) - (oy as i64);
    if ly < 0 {
        return None;
    }
    let (a, b) = unsafe { (*sp).span(ly as u32) }?;
    let lo = (a as i64 + ox as i64).max(x as i64);
    let hi = (b as i64 + ox as i64).min(x as i64 + w as i64);
    if hi <= lo {
        return None;
    }
    Some((lo as u32, (hi - lo) as u32))
}

#[derive(Clone, Copy)]
pub struct Framebuffer {
    base: *mut u32,
    width: u32,
    height: u32,
    /// Pixels per scan line. Frequently larger than `width`; using `width` as
    /// the stride produces a picture that shears diagonally, which is a useful
    /// thing to be able to recognise on sight.
    stride: u32,
    format: Format,
    /// True when `base` is the firmware's aperture, false when it is heap the
    /// compositor owns.
    ///
    /// Worth a whole field because it is the difference between a fill the
    /// compiler may vectorise and a million stores it may not. Writes to the
    /// aperture have to be volatile: nothing in this program reads them, and
    /// the optimiser will delete a screen-fill loop as dead stores if allowed
    /// to. Writes to the back buffer have no such need -- it is ordinary
    /// memory that `present` reads straight afterwards -- and paying the
    /// volatile price there made every rectangle in the interface cost one
    /// un-mergeable store per pixel.
    mmio: bool,
}

// Single core, ring 0, one owner. There is no other CPU to race with yet.
unsafe impl Send for Framebuffer {}
unsafe impl Sync for Framebuffer {}

impl Framebuffer {
    /// # Safety
    /// `base` must be the physical address of a linear framebuffer of at least
    /// `stride * height * 4` bytes, and identity-mapped and writable.
    pub const unsafe fn new(
        base: u64,
        width: u32,
        height: u32,
        stride: u32,
        format: Format,
    ) -> Self {
        Self { base: base as *mut u32, width, height, stride, format, mmio: true }
    }

    /// A framebuffer over memory this program owns.
    ///
    /// # Safety
    /// Same as `new`, and additionally `base` must be an allocation this
    /// kernel owns for as long as the returned value lives, not a device
    /// aperture. Getting that wrong loses writes rather than corrupting
    /// memory, which is why the two constructors are separate rather than a
    /// boolean argument at the call site.
    pub const unsafe fn over_ram(
        base: u64,
        width: u32,
        height: u32,
        stride: u32,
        format: Format,
    ) -> Self {
        Self { base: base as *mut u32, width, height, stride, format, mmio: false }
    }

    /// One row as a slice, clipped to the surface.
    ///
    /// Only for RAM surfaces. The aperture is kept write-only by discipline
    /// everywhere else here, and handing out a slice of it would be an
    /// invitation to read it back.
    #[inline]
    fn row_mut(&self, y: u32, x: u32, w: u32) -> Option<&mut [u32]> {
        if self.mmio || y >= self.height || x >= self.width {
            return None;
        }
        let w = w.min(self.width - x) as usize;
        if w == 0 {
            return None;
        }
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        Some(unsafe { core::slice::from_raw_parts_mut(self.base.add(off), w) })
    }

    /// Copy a run of pixels into one row.
    ///
    /// The fast path under `present`, which used to write the changed span of
    /// every row a pixel at a time through `put` -- two bounds checks and a
    /// volatile store each, for something that is a memcpy.
    pub fn blit_span(&self, x: u32, y: u32, src: &[u32]) {
        // A shaped blit has to drop the same pixels from the source as from
        // the destination, so the offset into `src` moves with the clip.
        let (x, src) = match clip_run(y, x, src.len() as u32) {
            None => return,
            Some((cx, cw)) => {
                let skip = (cx - x) as usize;
                (cx, &src[skip..(skip + cw as usize).min(src.len())])
            }
        };
        if let Some(dst) = self.row_mut(y, x, src.len() as u32) {
            let n = dst.len();
            dst.copy_from_slice(&src[..n]);
            return;
        }
        if y >= self.height || x >= self.width {
            return;
        }
        let n = (src.len() as u32).min(self.width - x) as usize;
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        for (i, v) in src[..n].iter().enumerate() {
            unsafe { write_volatile(self.base.add(off + i), *v) }
        }
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Pixels per scan line, which is often greater than `width()`.
    #[inline]
    pub const fn stride(&self) -> u32 {
        self.stride
    }

    #[inline]
    pub const fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub const fn encode(&self, c: Color) -> u32 {
        // Both layouts put the unused byte in the high position on a
        // little-endian read of the 32-bit word.
        match self.format {
            Format::Rgbx => {
                (c.r as u32) | ((c.g as u32) << 8) | ((c.b as u32) << 16)
            }
            Format::Bgrx => {
                (c.b as u32) | ((c.g as u32) << 8) | ((c.r as u32) << 16)
            }
        }
    }

    #[inline]
    pub fn put(&self, x: u32, y: u32, raw: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        if clip_run(y, x, 1).is_none() {
            return;
        }
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        // Volatile: the compiler cannot see that anyone reads this memory, and
        // would happily delete a whole screen-fill loop as dead stores.
        unsafe { write_volatile(self.base.add(off), raw) }
    }

    /// Read a pixel back, raw.
    ///
    /// The framebuffer is normally write-only here, and this exists for one
    /// caller: the mouse pointer saves what it covers so it can be lifted
    /// without repainting the desktop underneath it.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> u32 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        unsafe { core::ptr::read_volatile(self.base.add(off)) }
    }

    /// Encode a colour for `put`.
    #[inline]
    pub fn raw(&self, c: Color) -> u32 {
        self.encode(c)
    }

    pub fn fill(&self, c: Color) {
        let raw = self.encode(c);
        if raw == 0 {
            // The overwhelmingly common case, since the console background is
            // black. `write_bytes` lowers to memset, which is an opaque call
            // the optimiser cannot discard -- so it needs none of the volatile
            // treatment below, and moves the whole screen in one pass.
            let span = (self.height as usize) * (self.stride as usize);
            unsafe { core::ptr::write_bytes(self.base, 0, span) };
            return;
        }
        self.fill_rows(0, self.height, raw);
    }

    /// Paint `count` scan lines, starting at `y`, in an already-encoded colour.
    fn fill_rows(&self, y: u32, count: u32, raw: u32) {
        let end = y.saturating_add(count).min(self.height);
        for row in y..end {
            self.fill_span(0, row, self.width, raw);
        }
    }

    /// Shift the top `region_h` scan lines up by `pixels`, clearing what is
    /// exposed at the bottom.
    ///
    /// The bulk move is `copy_within`, which lowers to memmove. That matters
    /// for more than speed: memmove is an opaque call, so the optimiser will
    /// not delete it the way it would delete a plain loop of non-volatile
    /// stores into memory nothing appears to read. Only the newly exposed
    /// strip is painted pixel by pixel, and that is one text row rather than a
    /// screenful.
    ///
    /// This *reads* video memory, which was once the reason the console
    /// re-rendered from its character grid instead of scrolling -- reads across
    /// an uncached MMIO aperture are ruinous. That is no longer the situation:
    /// `build_identity_map` gives the framebuffer aperture a write-back
    /// mapping, so these are ordinary cached reads of stolen DRAM.
    ///
    /// `region_h` exists because the panel height is rarely a whole number of
    /// text rows -- 1080 is not divisible by 16 -- and scrolling the leftover
    /// strip at the bottom would drag it up into the last row.
    pub fn scroll_up(&self, region_h: u32, pixels: u32, bg: Color) {
        let region_h = region_h.min(self.height);
        if pixels == 0 || region_h == 0 {
            return;
        }
        let raw = self.encode(bg);
        if pixels >= region_h {
            self.fill_rows(0, region_h, raw);
            return;
        }

        let stride = self.stride as usize;
        let shift = (pixels as usize) * stride;
        let span = (region_h as usize) * stride;
        unsafe {
            let buf = core::slice::from_raw_parts_mut(self.base, span);
            buf.copy_within(shift..span, 0);
        }
        self.fill_rows(region_h - pixels, pixels, raw);
    }

    /// Scroll a sub-rectangle up by `pixels`, filling the exposed strip.
    ///
    /// `scroll_up` moves whole scan lines and cannot be used once the console
    /// is inset into a window: it would drag the frame and anything beside it
    /// upwards too. This copies row by row within the rectangle, which is more
    /// work per line and the only thing that is correct.
    pub fn scroll_rect(&self, x: u32, y: u32, w: u32, h: u32, pixels: u32, bg: Color) {
        if pixels == 0 || w == 0 || h == 0 || x >= self.width || y >= self.height {
            return;
        }
        let w = w.min(self.width - x);
        let h = h.min(self.height - y);
        let raw = self.encode(bg);
        if pixels >= h {
            for dy in 0..h {
                self.fill_span(x, y + dy, w, raw);
            }
            return;
        }

        let stride = self.stride as usize;
        for dy in 0..(h - pixels) {
            let src = ((y + dy + pixels) as usize) * stride + x as usize;
            let dst = ((y + dy) as usize) * stride + x as usize;
            unsafe {
                let buf = core::slice::from_raw_parts_mut(self.base, (self.height as usize) * stride);
                buf.copy_within(src..src + w as usize, dst);
            }
        }
        for dy in (h - pixels)..h {
            self.fill_span(x, y + dy, w, raw);
        }
    }

    fn fill_span(&self, x: u32, y: u32, w: u32, raw: u32) {
        let Some((x, w)) = clip_run(y, x, w) else { return };
        // Clip once for the whole run rather than once per pixel, which is
        // what going through `put` used to cost.
        if let Some(dst) = self.row_mut(y, x, w) {
            dst.fill(raw);
            return;
        }
        if y >= self.height || x >= self.width {
            return;
        }
        let n = w.min(self.width - x) as usize;
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        for i in 0..n {
            unsafe { write_volatile(self.base.add(off + i), raw) }
        }
    }

    pub fn rect(&self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        let raw = self.encode(c);
        let end = y.saturating_add(h).min(self.height);
        for row in y..end {
            self.fill_span(x, row, w, raw);
        }
    }

    /// Draw a string at a pixel position, bypassing the console entirely.
    ///
    /// Used by background tasks that want a fixed spot on screen without
    /// fighting the shell for the cursor.
    pub fn draw_text(&self, x: u32, y: u32, s: &str, fg: Color, bg: Color, scale: u32) {
        let fg_raw = self.encode(fg);
        let bg_raw = self.encode(bg);
        let scale = scale.max(1);

        for (i, ch) in s.bytes().enumerate() {
            let glyph = font::glyph(ch);
            let ox = x + i as u32 * font::GLYPH_W * scale;
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in 0..font::GLYPH_W {
                    let raw = if bits & (0x80 >> gx) != 0 { fg_raw } else { bg_raw };
                    for dy in 0..scale {
                        for dx in 0..scale {
                            self.put(ox + gx * scale + dx, y + gy as u32 * scale + dy, raw);
                        }
                    }
                }
            }
        }
    }

    /// One-pixel outline. Handy as a stride check: if `stride` is wrong the
    /// right-hand edge walks across the screen instead of staying vertical.
    /// Filled disc, by horizontal runs.
    ///
    /// One integer square root per scan line rather than a distance test per
    /// pixel: the same picture for a fraction of the work, and `rect` already
    /// knows how to lay down a run.
    pub fn fill_circle(&self, cx: i32, cy: i32, r: i32, c: Color) {
        if r <= 0 {
            return;
        }
        for dy in -r..=r {
            let half = isqrt((r * r - dy * dy) as u32) as i32;
            let x0 = cx - half;
            let y = cy + dy;
            if y < 0 || half <= 0 {
                continue;
            }
            self.rect(x0.max(0) as u32, y as u32, (half * 2) as u32, 1, c);
        }
    }

    /// Filled triangle, scanline. The workhorse for anything that is not a
    /// rectangle -- which on this display is very nearly nothing, and then
    /// suddenly a logo.
    pub fn fill_triangle(&self, a: (i32, i32), b: (i32, i32), c: (i32, i32), col: Color) {
        let mut v = [a, b, c];
        v.sort_by_key(|p| p.1);
        let (top, mid, bot) = (v[0], v[1], v[2]);
        if bot.1 == top.1 {
            return;
        }

        // x along an edge at height y, in halves to keep the rounding honest
        // without reaching for floating point.
        let edge = |p: (i32, i32), q: (i32, i32), y: i32| -> i32 {
            if q.1 == p.1 {
                return p.0;
            }
            p.0 + (q.0 - p.0) * (y - p.1) / (q.1 - p.1)
        };

        for y in top.1.max(0)..=bot.1.max(0) {
            if y < 0 {
                continue;
            }
            let xa = edge(top, bot, y);
            let xb = if y < mid.1 {
                edge(top, mid, y)
            } else {
                edge(mid, bot, y)
            };
            let (x0, x1) = if xa <= xb { (xa, xb) } else { (xb, xa) };
            if x1 < 0 {
                continue;
            }
            let x0 = x0.max(0);
            self.rect(x0 as u32, y as u32, (x1 - x0 + 1) as u32, 1, col);
        }
    }

    /// Bresenham. Integer only -- there is no floating point in early boot and
    /// no reason to want any here.
    pub fn line(&self, x0: i32, y0: i32, x1: i32, y1: i32, c: Color) {
        let raw = self.encode(c);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let (mut x, mut y) = (x0, y0);
        let mut err = dx + dy;
        loop {
            if x >= 0 && y >= 0 {
                self.put(x as u32, y as u32, raw);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// A thick line, drawn as `w` parallel offsets.
    ///
    /// Crude next to a proper polygon fill, and exactly right for a 16-colour
    /// display with no antialiasing: the result is the same hard-edged stroke
    /// either way.
    pub fn line_thick(&self, x0: i32, y0: i32, x1: i32, y1: i32, w: i32, c: Color) {
        let steep = (y1 - y0).abs() > (x1 - x0).abs();
        for i in 0..w {
            let o = i - w / 2;
            if steep {
                self.line(x0 + o, y0, x1 + o, y1, c);
            } else {
                self.line(x0, y0 + o, x1, y1 + o, c);
            }
        }
    }

    /// Midpoint circle, outline only.
    pub fn circle(&self, cx: i32, cy: i32, r: i32, c: Color) {
        let raw = self.encode(c);
        let (mut x, mut y) = (r, 0);
        let mut err = 1 - r;
        let plot = |px: i32, py: i32| {
            if px >= 0 && py >= 0 {
                self.put(px as u32, py as u32, raw);
            }
        };
        while x >= y {
            for (a, b) in [
                (x, y), (y, x), (-y, x), (-x, y),
                (-x, -y), (-y, -x), (y, -x), (x, -y),
            ] {
                plot(cx + a, cy + b);
            }
            y += 1;
            if err < 0 {
                err += 2 * y + 1;
            } else {
                x -= 1;
                err += 2 * (y - x) + 1;
            }
        }
    }

    pub fn circle_thick(&self, cx: i32, cy: i32, r: i32, w: i32, c: Color) {
        for i in 0..w {
            self.circle(cx, cy, r - i, c);
        }
    }

    /// The two-pixel 3D edge that every control in a Windows 3.x interface is
    /// made of.
    ///
    /// This is the whole aesthetic, and it is four line draws: a light edge on
    /// the top and left, a dark edge on the bottom and right, and the reverse
    /// for a sunken one. Raised means a button or a panel; sunken means a well
    /// something sits *in* -- a text field, a progress trough, a list box.
    ///
    /// It lives here rather than in the boot screen because it is the single
    /// primitive a window manager needs most, and there should be exactly one
    /// of it.
    pub fn bevel(&self, x: u32, y: u32, w: u32, h: u32, raised: bool) {
        if w < 2 || h < 2 {
            return;
        }
        let (tl, br) = if raised {
            (palette::WHITE, palette::DKGRAY)
        } else {
            (palette::DKGRAY, palette::WHITE)
        };
        // Top and left.
        self.rect(x, y, w, 1, tl);
        self.rect(x, y, 1, h, tl);
        // Bottom and right. Drawn second so the corners belong to the shadow,
        // which is what makes the corner read as a mitre rather than a stair.
        self.rect(x, y + h - 1, w, 1, br);
        self.rect(x + w - 1, y, 1, h, br);
    }

    pub fn frame(&self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        let raw = self.encode(c);
        for dx in 0..w {
            self.put(x + dx, y, raw);
            self.put(x + dx, y + h - 1, raw);
        }
        for dy in 0..h {
            self.put(x, y + dy, raw);
            self.put(x + w - 1, y + dy, raw);
        }
    }
}

/// Where the drawing time actually goes.
///
/// Written because "the desktop feels slow" is not a number and cannot be
/// optimised against. Each figure is microseconds for one operation at the
/// real screen size, taken from the TSC, and the point is the ratio between
/// them rather than any absolute: a fill that costs more than the diff that
/// follows it says the compositor is not the problem.
pub fn bench() {
    use crate::kprintln;
    let Some(fb) = primary() else {
        kprintln!("  no framebuffer");
        return;
    };
    let mhz = crate::time::tsc_mhz().max(1);
    let us = |t: u64| t / mhz;

    let target = compose::target().unwrap_or_else(|| fb.clone());
    let (w, h) = (target.width(), target.height());
    kprintln!("  {}x{}  {} pixels", w, h, w as u64 * h as u64);

    let t = crate::time::rdtsc();
    target.rect(0, 0, w, h, theme::DESKTOP);
    let fill = crate::time::rdtsc() - t;
    kprintln!("  full-screen rect      {:>8} us", us(fill));

    let t = crate::time::rdtsc();
    desk::draw();
    let full = crate::time::rdtsc() - t;
    kprintln!("  desk::draw + present  {:>8} us", us(full));

    // A present with nothing changed: the diff alone, which is the floor for
    // any repaint that turns out to be a no-op.
    let t = crate::time::rdtsc();
    compose::present();
    let clean = crate::time::rdtsc() - t;
    kprintln!("  present, no change    {:>8} us", us(clean));

    // ...and one where every row differs, which is the worst case.
    let t = crate::time::rdtsc();
    target.rect(0, 0, w, h, theme::APERTURE);
    compose::present();
    let dirty = crate::time::rdtsc() - t;
    kprintln!("  fill + present, all   {:>8} us", us(dirty));
    desk::draw();
}
