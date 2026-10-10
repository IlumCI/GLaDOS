//! Linear framebuffer.
//!
//! UEFI hands us a physical address, a resolution, a pixel format and a stride.
//! There is no VGA text mode to fall back on -- this machine boots UEFI with no
//! CSM, so there is no INT 10h, no 0xB8000 text buffer, and no VGA BIOS. Pixels
//! are the only output device that exists.

pub mod browse;
pub mod lease;
pub mod region;
pub mod render;
pub mod blend;
pub mod scene;
pub mod shape;
pub mod grad;
pub mod shadow;
pub mod glass;
pub mod scale;
pub mod wall;
pub mod icons;
pub mod text;

/// The time a frame is about, in milliseconds since boot (TSC), taken once
/// when the frame's scene is built; outside a frame, the time now.
///
/// For programs that draw from the clock -- the Mines timer, Oracle's
/// animation. Their stamp, their paint and `render verify`'s reference paint
/// happen at three different moments, and a program reading the live clock in
/// each could see a second turn over between them: the stamp said one digit,
/// the paint drew another, and the verifier reported a difference that was the
/// clock's. One reading per frame makes all three agree.
pub fn now_ms() -> u64 {
    match FRAME_MS.load(core::sync::atomic::Ordering::Relaxed) {
        0 => live_ms(),
        t => t,
    }
}

fn live_ms() -> u64 {
    let mhz = crate::time::tsc_mhz() as u64;
    if mhz == 0 {
        return 0;
    }
    crate::time::rdtsc() / (mhz * 1000)
}

static FRAME_MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Called by the desktop as it builds a frame's scene, and with `false` when
/// the frame is done.
pub(crate) fn frame_clock(on: bool) {
    FRAME_MS.store(if on { live_ms().max(1) } else { 0 }, core::sync::atomic::Ordering::Relaxed);
}
pub mod compose;
pub mod console;
pub mod mines;
pub mod netman;
pub mod oracle;
pub mod paint;
pub mod write;
pub mod splash;
pub mod intro;
pub mod font;
pub mod theme;
pub mod todo;
pub mod ui;
pub mod uidoc;
pub mod desk;
pub mod agentwin;
pub mod mindwin;

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

    /// A number naming this window to whatever opened it, so it can be found
    /// again to retitle or close. Zero for the built-in programs, which are
    /// opened and closed by the desktop alone; a Wayland client's windows are
    /// opened and closed by the client.
    fn tag(&self) -> u64 {
        0
    }

    /// A fingerprint of everything this program draws from *other than input*
    /// -- its log's length, its data's generation, a running clock's second.
    ///
    /// The desktop keeps each program's pixels between frames and redraws them
    /// only when this, the focus, the size or an input event changes. `None`,
    /// the default, means "redraw on every composed frame", which is what
    /// every program got before there were stamps, so a program that does not
    /// answer loses nothing; a program that answers wrongly shows stale pixels
    /// until its next input, which `render verify` reports.
    ///
    /// Programs draw and are handed input in their own coordinates: `client`
    /// is `(0, 0, w, h)` in `draw_in`, `press`, `drag` and `right_press`, and
    /// the pointer is relative to it. Layout cached while drawing therefore
    /// means the same thing when a press reads it back.
    fn stamp(&self) -> Option<u64> {
        None
    }

    /// How often a program with a stamp wants its stamp looked at when
    /// nothing else is happening -- a timer, a clock, an animation. `None`,
    /// the default, means only on input or when something else composes.
    /// The desktop wakes the compositor at the shortest period any open
    /// program asks for; a frame whose stamps did not move repaints nothing.
    fn wake_ms(&self) -> Option<u32> {
        None
    }
}

use crate::sync::Racy;

/// The framebuffer, reachable from anywhere -- background tasks draw straight
/// to it rather than going through the console.
static PRIMARY: Racy<Option<Framebuffer>> = Racy::new(None);

pub fn set_primary(fb: Framebuffer) {
    unsafe { *PRIMARY.get() = Some(fb) };
}

/// Whether something other than the desktop owns the screen. The question
/// `gfx::lease` answers now, with an owner attached; this name stays for the
/// many places that only need yes or no.
pub fn exclusive() -> bool {
    lease::held()
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
        Self::rounded(w, h, r, r)
    }

    /// The top two corners rounded, the bottom two square.
    ///
    /// The window silhouette. The reason it is the top only is structural
    /// rather than decorative: the bottom edge of a window is something you
    /// drag, and a rounded one reads as a control that has come loose from
    /// its frame.
    pub fn round_top(w: u32, h: u32, r: u32) -> Shape {
        Self::rounded(w, h, r, 0)
    }

    /// Independent top and bottom radii.
    ///
    /// One arc, so a corner cannot be computed two ways. Both public forms
    /// are one call into here, which is what makes `round_top` free rather
    /// than a second copy of a square root nobody would diff.
    fn rounded(w: u32, h: u32, top: u32, bottom: u32) -> Shape {
        let cap = (w / 2).min(h / 2);
        let (top, bottom) = (top.min(cap), bottom.min(cap));
        let mut rows = alloc::vec::Vec::with_capacity(h as usize);
        for y in 0..h {
            // The radius in force at this row and the distance from the
            // nearer horizontal edge. Outside both corner bands the row is
            // the full width -- and a radius of zero puts every row there,
            // since `y < 0` and `y >= h` are both empty.
            let band = if y < top {
                Some((top, top - 1 - y))
            } else if y >= h - bottom {
                Some((bottom, y - (h - bottom)))
            } else {
                None
            };
            match band {
                None => rows.push((0, w)),
                Some((r, d)) => {
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
///
/// **It applies to one surface, the one it was installed for**, keyed by that
/// surface's base address. It was global, and two things showed why it cannot
/// be: `diag paint`, running on the shell task, drew its 16x16 test surface
/// through whatever window outline the compositor had installed when it was
/// preempted, and failed a dozen claims at random; and an application drawing
/// into its own surface inside a window's paint was clipped by that window's
/// outline in *screen* coordinates against its own *local* ones. A clipped
/// surface keeps its base, so a damage-limited repaint of the back buffer is
/// still shaped; a view or another buffer is not.
static CLIP: Racy<Option<(*const Shape, i32, i32, usize)>> = Racy::new(None);

/// Draw `f` with writes to `on` confined to `shape`, positioned with its
/// top-left at `(x, y)` in `on`'s coordinates.
pub fn with_shape<R>(on: &Framebuffer, shape: &Shape, x: i32, y: i32, f: impl FnOnce() -> R) -> R {
    let prev = unsafe { *CLIP.get() };
    unsafe { *CLIP.get() = Some((shape as *const Shape, x, y, on.base as usize)) };
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
fn clip_run(base: usize, y: u32, x: u32, w: u32) -> Option<(u32, u32)> {
    let Some((sp, ox, oy, on)) = (unsafe { *CLIP.get() }) else {
        return Some((x, w));
    };
    if on != base {
        return Some((x, w));
    }
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

/// The colour a stop table gives at row `dy` of `h`.
///
/// Pulled out of `vgrad` so a shape cut into something drawn over a ramp can
/// ask the ramp what it would have put there, rather than approximating it
/// with one colour that belongs to no part of it.
pub fn ramp_at(stops: &[(u8, Color)], dy: u32, h: u32) -> Color {
    if stops.is_empty() {
        return Color::new(0, 0, 0);
    }
    if stops.len() == 1 || h <= 1 {
        return stops[0].1;
    }
    let t = (dy.min(h - 1) * 255 / (h - 1)) as u8;
    let mut k = 0usize;
    while k + 2 < stops.len() && stops[k + 1].0 <= t {
        k += 1;
    }
    let (p0, c0) = stops[k];
    let (p1, c1) = stops[k + 1];
    let span = (p1.saturating_sub(p0)).max(1) as i32;
    let f = (t.saturating_sub(p0) as i32).min(span);
    let mix = |a: u8, b: u8| (a as i32 + (b as i32 - a as i32) * f / span) as u8;
    Color::new(mix(c0.r, c1.r), mix(c0.g, c1.g), mix(c0.b, c1.b))
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
    /// The part of the surface writes may land in: `[sx0, sx1) x [sy0, sy1)`.
    ///
    /// The whole surface unless a caller narrowed it with `clipped`. This is
    /// what lets a damage-limited compositor repaint one rectangle by calling
    /// the *same* painters that draw a whole frame -- every write path goes
    /// through `clip`, so a painter that draws a window, a gradient or a line
    /// of text simply loses whatever falls outside, with no painter knowing
    /// there was a rectangle at all.
    sx0: u32,
    sy0: u32,
    sx1: u32,
    sy1: u32,
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
        Self { base: base as *mut u32, width, height, stride, format, mmio: true,
               sx0: 0, sy0: 0, sx1: width, sy1: height }
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
        Self { base: base as *mut u32, width, height, stride, format, mmio: false,
               sx0: 0, sy0: 0, sx1: width, sy1: height }
    }

    /// A buffer of `w` x `h` that answers to the screen rectangle at
    /// `(x, y)`: a painter writes in screen coordinates and the pixel lands
    /// in `buf`, at its offset into the rectangle. The scissor is exactly that
    /// rectangle, so nothing outside it is ever addressed.
    ///
    /// For a window's image (`desk`): every painter in this tree works in
    /// screen coordinates, and glass in particular samples the wallpaper by
    /// screen position, so drawing a window off screen means lying about
    /// where the buffer is rather than teaching forty painters an origin.
    /// `base` points `y * w + x` pixels *before* `buf`, which is never read
    /// or written through -- every access is clipped to the scissor first,
    /// `get` and `blend` included.
    ///
    /// # Safety
    /// `buf` must hold `w * h` pixels and outlive the returned value.
    pub unsafe fn displaced(buf: *mut u32, x: u32, y: u32, w: u32, h: u32, format: Format) -> Self {
        let back = y as usize * w as usize + x as usize;
        Self {
            base: buf.wrapping_sub(back),
            width: x + w,
            height: y + h,
            stride: w,
            format,
            mmio: false,
            sx0: x,
            sy0: y,
            sx1: x + w,
            sy1: y + h,
        }
    }

    /// The same surface with writes confined to a rectangle, in the same
    /// coordinates. Narrowing an already narrowed one intersects the two, so a
    /// scissor can only ever shrink -- a painter handed a clipped surface
    /// cannot widen its way back out.
    pub fn clipped(&self, x: u32, y: u32, w: u32, h: u32) -> Framebuffer {
        let mut f = *self;
        f.sx0 = self.sx0.max(x);
        f.sy0 = self.sy0.max(y);
        f.sx1 = self.sx1.min(x.saturating_add(w));
        f.sy1 = self.sy1.min(y.saturating_add(h));
        if f.sx1 < f.sx0 {
            f.sx1 = f.sx0;
        }
        if f.sy1 < f.sy0 {
            f.sy1 = f.sy0;
        }
        f
    }

    /// A rectangle of this surface as a surface of its own, with `(0, 0)` at
    /// the rectangle's corner -- for something that draws in its own
    /// coordinates, an application's client area most of all. The base
    /// pointer moves and the stride stays, so every primitive works unchanged
    /// and clips to the rectangle by clipping to its own size. The parent's
    /// scissor carries over, translated. `None` when the rectangle is off the
    /// surface.
    ///
    /// A shape installed with `with_shape` is in absolute coordinates of
    /// whatever surface it is used on; inside a view, pass it local ones.
    pub fn view(&self, x: u32, y: u32, w: u32, h: u32) -> Option<Framebuffer> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let w = w.min(self.width - x);
        let h = h.min(self.height - y);
        let off = (y as usize) * (self.stride as usize) + x as usize;
        Some(Framebuffer {
            base: unsafe { self.base.add(off) },
            width: w,
            height: h,
            stride: self.stride,
            format: self.format,
            mmio: self.mmio,
            sx0: self.sx0.saturating_sub(x).min(w),
            sy0: self.sy0.saturating_sub(y).min(h),
            sx1: self.sx1.saturating_sub(x).min(w),
            sy1: self.sy1.saturating_sub(y).min(h),
        })
    }

    /// Whether anything drawn in this rectangle could land: it meets the
    /// scissor. For skipping whole layers outside the area being repainted.
    pub fn touches(&self, x: u32, y: u32, w: u32, h: u32) -> bool {
        x < self.sx1 && y < self.sy1
            && x.saturating_add(w) > self.sx0 && y.saturating_add(h) > self.sy0
    }

    /// The writable rectangle as `(x0, y0, x1, y1)`, exclusive at the far edge.
    pub fn scissor(&self) -> (u32, u32, u32, u32) {
        (self.sx0, self.sy0, self.sx1, self.sy1)
    }

    /// Whether writes reach every pixel, i.e. nothing has narrowed the scissor.
    fn unclipped(&self) -> bool {
        self.sx0 == 0 && self.sy0 == 0 && self.sx1 == self.width && self.sy1 == self.height
    }

    /// The part of a run on row `y` that may be written: the scissor, then any
    /// shape installed with `with_shape`. Every write path asks this.
    #[inline]
    fn clip(&self, y: u32, x: u32, w: u32) -> Option<(u32, u32)> {
        if y < self.sy0 || y >= self.sy1 {
            return None;
        }
        let lo = x.max(self.sx0);
        let hi = (x as u64 + w as u64).min(self.sx1 as u64) as u32;
        if hi <= lo {
            return None;
        }
        clip_run(self.base as usize, y, lo, hi - lo)
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

    /// Darken a run of pixels in place: every channel scaled by `num`/256.
    ///
    /// The one read-modify-write primitive here, and it exists so a window can
    /// cast a shadow. It sits beside `row_mut` rather than that going public:
    /// `row_mut` gives a reason for handing out no slice of the aperture, and
    /// spending it to serve one caller in the same `impl` would be paying for
    /// nothing.
    ///
    /// **RAM surfaces only, and silently so.** Reading the aperture back to
    /// blend is the discipline `row_mut` declines to break, and it is not
    /// worth breaking here: without a compositor there are no shadows and
    /// everything else looks the same, which is the bargain `compose::target`
    /// already makes about flicker.
    ///
    /// Format-independent by construction. Scaling every channel by the same
    /// factor commutes with any byte permutation, so `Rgbx` and `Bgrx` need no
    /// separate arms and cannot come to disagree about one.
    pub fn shade_span(&self, x: u32, y: u32, w: u32, num: u32) {
        // Clipped first, for symmetry with `fill_span` and `blit_span`: a
        // shadow drawn under a shaped window has the same right to be cut.
        let Some((x, w)) = self.clip(y, x, w) else { return };
        let Some(dst) = self.row_mut(y, x, w) else { return };
        let num = num.min(256);
        for px in dst.iter_mut() {
            let v = *px;
            // Two channels at a time. They are eight bits apart, so masking
            // alternate ones leaves the multiply room in the gaps and nothing
            // has to be unpacked.
            //
            // `0x00FF00FF * 256` is `0xFF00FF00` exactly, so `num <= 256`
            // cannot carry out of a channel into its neighbour. That is the
            // only reason this is safe without a wider type, and it is why the
            // clamp above is not merely defensive.
            //
            // Not a shift. `>>1` keeps half and `>>2` a quarter; a drop shadow
            // wants around 84%, and a power of two cannot say that.
            let lo = (((v & 0x00FF_00FF) * num) >> 8) & 0x00FF_00FF;
            let hi = (((v & 0x0000_FF00) * num) >> 8) & 0x0000_FF00;
            *px = lo | hi;
        }
    }

    /// `shade_span` per row, the way `rect` is `fill_span` per row.
    pub fn shade_rect(&self, x: u32, y: u32, w: u32, h: u32, num: u32) {
        let end = y.saturating_add(h).min(self.height);
        for row in y..end {
            self.shade_span(x, row, w, num);
        }
    }

    /// Blend a run toward `c` by `num`/256, leaving `256 - num` of what was
    /// there. Zero changes nothing and 256 is a solid fill.
    ///
    /// What `shade_span` is to a shadow, this is to glass. A translucent
    /// surface is not a darker one: it takes the colour of what it is made of
    /// and keeps some of what is behind it, and a taskbar that only darkened
    /// the desktop would read as a smoked strip rather than as something with
    /// a tint of its own.
    ///
    /// **No carry, for a reason worth stating.** Each half is floored
    /// independently, and `floor(a*(256-n)/256) + floor(b*n/256)` is at most
    /// the exact sum, which is at most `max(a, b)` and so at most 255. So the
    /// two-lane trick holds here as well and neither addition can spill into
    /// the neighbouring channel.
    ///
    /// RAM surfaces only, exactly as `shade_span`.
    pub fn tint_span(&self, x: u32, y: u32, w: u32, c: Color, num: u32) {
        let Some((x, w)) = self.clip(y, x, w) else { return };
        let num = num.min(256);
        let keep = 256 - num;
        let src = self.encode(c);
        let (src_lo, src_hi) = (
            (((src & 0x00FF_00FF) * num) >> 8) & 0x00FF_00FF,
            (((src & 0x0000_FF00) * num) >> 8) & 0x0000_FF00,
        );
        let Some(dst) = self.row_mut(y, x, w) else { return };
        for px in dst.iter_mut() {
            let v = *px;
            let lo = ((((v & 0x00FF_00FF) * keep) >> 8) & 0x00FF_00FF) + src_lo;
            let hi = ((((v & 0x0000_FF00) * keep) >> 8) & 0x0000_FF00) + src_hi;
            *px = lo | hi;
        }
    }

    /// `tint_span` per row.
    pub fn tint_rect(&self, x: u32, y: u32, w: u32, h: u32, c: Color, num: u32) {
        let end = y.saturating_add(h).min(self.height);
        for row in y..end {
            self.tint_span(x, row, w, c, num);
        }
    }

    /// A pane of glass: a vertical ramp of *opacity* over one colour.
    ///
    /// Aero's surfaces are not one translucency all the way down. They are
    /// most solid where the light catches them and thinnest at the foot, which
    /// is what makes a strip of it read as a pane rather than as a wash. So
    /// this is `vgrad`'s shape applied to `num` instead of to the colour, and
    /// it costs the same: one blended span per row.
    pub fn glass(&self, x: u32, y: u32, w: u32, h: u32, c: Color, stops: &[(u8, u32)]) {
        if h == 0 || stops.is_empty() {
            return;
        }
        let end = y.saturating_add(h).min(self.height);
        for row in y..end {
            let t = if h > 1 { ((row - y) * 255 / (h - 1)) as u8 } else { 0 };
            let mut k = 0usize;
            while k + 2 < stops.len() && stops[k + 1].0 <= t {
                k += 1;
            }
            let (p0, n0) = stops[k];
            let (p1, n1) = stops[(k + 1).min(stops.len() - 1)];
            let span = (p1.saturating_sub(p0)).max(1) as i32;
            let f = (t.saturating_sub(p0) as i32).min(span);
            let num = (n0 as i32 + (n1 as i32 - n0 as i32) * f / span).clamp(0, 256) as u32;
            self.tint_span(x, row, w, c, num);
        }
    }

    /// Copy a run of pixels into one row.
    ///
    /// The fast path under `present`, which used to write the changed span of
    /// every row a pixel at a time through `put` -- two bounds checks and a
    /// volatile store each, for something that is a memcpy.
    /// Note a write to the screen itself made while a program holds it and
    /// by a task other than that program's: `render`'s "wrote a leased
    /// screen". A lease promises the program the aperture to itself, and the
    /// only way to find out who breaks that is to catch them at the write.
    #[track_caller]
    #[inline]
    fn intrusion(&self) {
        if !self.mmio || lease::compositor_may_write() {
            return;
        }
        if lease::holder_task() == Some(crate::task::current()) {
            return;
        }
        render::intruded(core::panic::Location::caller());
    }

    /// The writable part of row `y` from `x` for `w` pixels, through the
    /// scissor and any installed shape, and how far into the request it starts
    /// -- so a caller can take the matching part of its own mask or source.
    /// RAM surfaces only (see `row_mut`).
    fn blend_target(&self, x: u32, y: u32, w: u32) -> Option<(&mut [u32], usize)> {
        let (cx, cw) = self.clip(y, x, w)?;
        let row = self.row_mut(y, cx, cw)?;
        Some((row, (cx - x) as usize))
    }

    /// Lay premultiplied `src` over row `y` from `x` (`gfx::blend::over`).
    pub fn over_row(&self, x: u32, y: u32, src: &[u32]) {
        if let Some((row, skip)) = self.blend_target(x, y, src.len() as u32) {
            let n = row.len();
            blend::over(row, &src[skip..skip + n]);
        } else if self.mmio {
            // The screen itself is never read; an opaque source still lands.
            for (i, &p) in src.iter().enumerate() {
                if p >> 24 == 0xFF {
                    self.put(x + i as u32, y, p & 0x00FF_FFFF);
                }
            }
        }
    }

    /// Lay a premultiplied `color` over row `y` through a coverage `mask`.
    pub fn over_mask_row(&self, x: u32, y: u32, mask: &[u8], color: u32) {
        if let Some((row, skip)) = self.blend_target(x, y, mask.len() as u32) {
            let n = row.len();
            blend::over_mask(row, &mask[skip..skip + n], color);
        } else if self.mmio {
            // No back buffer -- early boot, a fault report -- so no reading:
            // a mask is thresholded rather than blended.
            for (i, &m) in mask.iter().enumerate() {
                if m >= 128 {
                    self.put(x + i as u32, y, color & 0x00FF_FFFF);
                }
            }
        }
    }

    /// A translucent fill of `w` pixels of row `y` (`gfx::blend::over_solid`).
    pub fn over_span(&self, x: u32, y: u32, w: u32, color: u32) {
        if let Some((row, _)) = self.blend_target(x, y, w) {
            blend::over_solid(row, color);
        } else if self.mmio && color >> 24 >= 128 {
            self.fill_span(x, y, w, color & 0x00FF_FFFF);
        }
    }

    /// Every channel of `w` pixels of row `y` times `factor`, over 255: a dim.
    /// Nothing happens on the screen itself, which cannot be read.
    pub fn multiply_span(&self, x: u32, y: u32, w: u32, factor: u32) {
        if let Some((row, _)) = self.blend_target(x, y, w) {
            blend::multiply(row, factor);
        }
    }

    #[track_caller]
    pub fn blit_span(&self, x: u32, y: u32, src: &[u32]) {
        // A shaped blit has to drop the same pixels from the source as from
        // the destination, so the offset into `src` moves with the clip.
        let (x, src) = match self.clip(y, x, src.len() as u32) {
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
        self.intrusion();
        let n = (src.len() as u32).min(self.width - x) as usize;
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        // One bulk copy rather than `n` serialised volatile stores. This is
        // the write half of `present`, so at 1920x1080 the old form was up to
        // two million un-mergeable stores for a frame.
        //
        // The aperture is write-combining now (`mem::paging::set_cache`), and
        // the copy uses non-temporal stores when `STREAM` is on: they go
        // straight to the combining buffers in whole 64-byte lines and never
        // allocate a cache line for memory nobody will read. The `sfence`
        // after is what makes the row visible to the display engine before
        // anything that follows -- combining buffers drain in no promised
        // order otherwise. `video stream off` falls back to a plain copy, so
        // the two can be compared within one boot on the GF63.
        unsafe {
            let dst = self.base.add(off);
            if STREAM.load(core::sync::atomic::Ordering::Relaxed) {
                stream_copy(dst, src.as_ptr(), n);
            } else {
                core::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
            }
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
    /// Where the aperture is.
    ///
    /// Exposed for exactly one caller: `/dev/fb0` hands a Linux guest the real
    /// framebuffer rather than a shadow, and `smem_start` in
    /// `fb_fix_screeninfo` is that address. Nothing above the compositor
    /// should want this -- every drawing path in this tree goes through the
    /// back buffer, and a second writer to the aperture is the bug
    /// `paint_clock` was fixed for.
    pub fn addr(&self) -> u64 {
        self.base as u64
    }

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
    #[track_caller]
    pub fn put(&self, x: u32, y: u32, raw: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        if self.clip(y, x, 1).is_none() {
            return;
        }
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        // The same split `fill_span` and `blit_span` already make, and for the
        // same reason -- this was the one primitive still paying the aperture's
        // price on every surface. Volatile is required for the aperture, where
        // nothing in this program reads the memory and the optimiser would
        // delete a screen-fill as dead stores. The back buffer is ordinary heap
        // that `present` reads immediately afterwards, so an ordinary store is
        // both correct and mergeable.
        //
        // It matters because `put` is the glyph path: `console::draw_cell`
        // writes 8*8*scale^2 = 256 pixels per character through here, and a
        // full terminal redraw is around 1.2 million of them. Every one was an
        // un-mergeable volatile store into RAM.
        if self.mmio {
            self.intrusion();
            unsafe { write_volatile(self.base.add(off), raw) }
        } else {
            unsafe { *self.base.add(off) = raw }
        }
    }

    /// Read a pixel back, raw.
    ///
    /// The framebuffer is normally write-only here, and this exists for one
    /// caller: the mouse pointer saves what it covers so it can be lifted
    /// without repainting the desktop underneath it.
    #[inline]
    #[track_caller]
    pub fn get(&self, x: u32, y: u32) -> u32 {
        // Outside the scissor answers nothing: a displaced surface has no
        // memory there at all.
        if x >= self.width || y >= self.height || x < self.sx0 || y < self.sy0 || x >= self.sx1 || y >= self.sy1 {
            return 0;
        }
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        if self.mmio {
            render::mmio_read(1);
        }
        unsafe { core::ptr::read_volatile(self.base.add(off)) }
    }

    /// Alpha-blend a colour onto what is already on screen.
    ///
    /// `a` is 0..=256, where 0 leaves the pixel untouched and 256 replaces it.
    /// It reads the destination back through `get`, so it is for the few
    /// painters whose source *is* whatever was already there: the boot
    /// splash's glow halo and its reflection, where the pixel behind is the
    /// only thing a mirror or a bloom has to work from. Ordinary opaque
    /// drawing stays on `put`/`rect`, which never read.
    #[inline]
    pub fn blend(&self, x: u32, y: u32, c: Color, a: u16) {
        if a == 0 || x < self.sx0 || y < self.sy0 || x >= self.sx1 || y >= self.sy1 {
            return;
        }
        if a >= 256 {
            self.put(x, y, self.encode(c));
            return;
        }
        let raw = self.get(x, y);
        let (dr, dg, db) = match self.format {
            Format::Rgbx => (
                (raw & 0xff) as u16,
                ((raw >> 8) & 0xff) as u16,
                ((raw >> 16) & 0xff) as u16,
            ),
            Format::Bgrx => (
                ((raw >> 16) & 0xff) as u16,
                ((raw >> 8) & 0xff) as u16,
                (raw & 0xff) as u16,
            ),
        };
        let b = 256 - a;
        let mix = |d: u16, s: u8| ((d * b + s as u16 * a) >> 8) as u8;
        let out = Color::new(mix(dr, c.r), mix(dg, c.g), mix(db, c.b));
        self.put(x, y, self.encode(out));
    }

    /// Encode a colour for `put`.
    #[inline]
    pub fn raw(&self, c: Color) -> u32 {
        self.encode(c)
    }

    #[track_caller]
    pub fn fill(&self, c: Color) {
        self.intrusion();
        let raw = self.encode(c);
        // The memset covers `height * stride`, which on a view or a clipped
        // surface is pixels that are not this surface's to write.
        if raw == 0 && self.unclipped() && self.width == self.stride {
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
    #[track_caller]
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
    /// an uncached MMIO aperture are ruinous. They are again: the aperture is
    /// write-combining since 1.4.1 and `render` counts every read. What
    /// follows describes the write-back mapping that preceded that, when
    /// `build_identity_map` gave the framebuffer aperture a write-back
    /// mapping, so these are ordinary cached reads of stolen DRAM.
    ///
    /// `region_h` exists because the panel height is rarely a whole number of
    /// text rows -- 1080 is not divisible by 16 -- and scrolling the leftover
    /// strip at the bottom would drag it up into the last row.
    #[track_caller]
    pub fn scroll_up(&self, region_h: u32, pixels: u32, bg: Color) {
        self.intrusion();
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
        if self.mmio {
            render::mmio_read((span - shift) as u64);
        }
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
    #[track_caller]
    pub fn scroll_rect(&self, x: u32, y: u32, w: u32, h: u32, pixels: u32, bg: Color) {
        self.intrusion();
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
        if self.mmio {
            render::mmio_read((h - pixels) as u64 * w as u64);
        }
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

    #[track_caller]
    fn fill_span(&self, x: u32, y: u32, w: u32, raw: u32) {
        let Some((x, w)) = self.clip(y, x, w) else { return };
        // Clip once for the whole run rather than once per pixel, which is
        // what going through `put` used to cost.
        if let Some(dst) = self.row_mut(y, x, w) {
            dst.fill(raw);
            return;
        }
        if y >= self.height || x >= self.width {
            return;
        }
        self.intrusion();
        let n = w.min(self.width - x) as usize;
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        for i in 0..n {
            unsafe { write_volatile(self.base.add(off + i), raw) }
        }
    }

    #[track_caller]
    pub fn rect(&self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        let raw = self.encode(c);
        let end = y.saturating_add(h).min(self.height);
        for row in y..end {
            self.fill_span(x, row, w, raw);
        }
    }

    /// A vertical ramp through `stops`, one span fill per row.
    ///
    /// `rect` with a colour that varies by row, which is why it lives here
    /// beside the other loop that memsets rows rather than in `theme` with the
    /// tables that describe a look.
    ///
    /// **It allocates nothing, and that is why the ramp is vertical.** A row
    /// of a vertical ramp is one colour; a row of a horizontal one is a
    /// pattern, which is why the title bar's old ramp had to build a `Vec` of
    /// its width every window every frame and blit it down. Same picture, one
    /// less allocation inside `desk::draw`.
    ///
    /// Stops are `(position, colour)` with position 0..=255 across `h`,
    /// ascending, and are read as proportions rather than pixel offsets -- so
    /// one table serves a 30-pixel title bar, a 38-pixel taskbar and a
    /// 21-pixel caption button instead of three tables that drift apart.
    ///
    /// **Two stops one position apart is a hard step**, and that is deliberate
    /// rather than tolerated: the break across the middle of a Luna caption is
    /// a step and not a ramp, and a table that could not express one would
    /// turn every gradient in the interface into a plain fade.
    pub fn vgrad(&self, x: u32, y: u32, w: u32, h: u32, stops: &[(u8, Color)]) {
        if h == 0 || stops.is_empty() {
            return;
        }
        if stops.len() == 1 {
            self.rect(x, y, w, h, stops[0].1);
            return;
        }
        let end = y.saturating_add(h).min(self.height);
        for row in y..end {
            // Taken against the height that was asked for rather than the one
            // that survived clipping, so a ramp running off the bottom of the
            // screen is cut rather than squashed.
            let raw = self.encode(ramp_at(stops, row - y, h));
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

        for (i, ch) in s.chars().enumerate() {
            let ox = x + i as u32 * font::GLYPH_W * scale;
            self.blit_glyph(ox, y, font::index_of(ch), fg_raw, bg_raw, scale);
        }
    }

    /// A string with no background: only the lit pixels are written.
    ///
    /// `draw_text` paints each cell's background too, which is right on a flat
    /// fill and wrong over a gradient or a pane of glass, where it leaves a
    /// solid box behind every word. The miner's screen is nothing but text over
    /// a sky, so it draws with this. A lit run within a glyph row goes down as
    /// one `rect` rather than a `put` per pixel.
    pub fn draw_text_over(&self, x: u32, y: u32, s: &str, fg: Color, scale: u32) {
        let scale = scale.max(1);
        for (i, ch) in s.chars().enumerate() {
            let ox = x + i as u32 * font::GLYPH_W * scale;
            for (gy, bits) in font::rows(font::index_of(ch)).iter().enumerate() {
                let mut gx = 0;
                while gx < font::GLYPH_W {
                    if bits & (0x80 >> gx) == 0 {
                        gx += 1;
                        continue;
                    }
                    let start = gx;
                    while gx < font::GLYPH_W && bits & (0x80 >> gx) != 0 {
                        gx += 1;
                    }
                    self.rect(ox + start * scale, y + gy as u32 * scale, (gx - start) * scale, scale, fg);
                }
            }
        }
    }

    /// One glyph by index, for a caller that already resolved it.
    ///
    /// `edit.rs` keeps a cell grid of indices, and rebuilding a `&str` per
    /// cell to hand back to `draw_text` was how it drew before -- which is
    /// also why it could not draw anything a single byte could not spell.
    pub fn draw_glyph(&self, x: u32, y: u32, glyph: u16, fg: Color, bg: Color, scale: u32) {
        let fg_raw = self.encode(fg);
        let bg_raw = self.encode(bg);
        self.blit_glyph(x, y, glyph, fg_raw, bg_raw, scale.max(1));
    }

    fn blit_glyph(&self, x: u32, y: u32, glyph: u16, fg_raw: u32, bg_raw: u32, scale: u32) {
        let rows = font::rows(glyph);
        for (gy, bits) in rows.iter().enumerate() {
            for gx in 0..font::GLYPH_W {
                let raw = if bits & (0x80 >> gx) != 0 { fg_raw } else { bg_raw };
                for dy in 0..scale {
                    for dx in 0..scale {
                        self.put(x + gx * scale + dx, y + gy as u32 * scale + dy, raw);
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
    /// Walk a triangle's rows, handing each span to `row`.
    ///
    /// Split out so the two fillers cannot come to disagree about which pixels
    /// a triangle covers. One of them cuts holes in a logo and the other fills
    /// it in; a shape and its hole rasterising differently would show as a
    /// fringe nobody could account for.
    fn tri_spans(
        &self,
        a: (i32, i32),
        b: (i32, i32),
        c: (i32, i32),
        mut row: impl FnMut(u32, u32, u32),
    ) {
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
            row(y as u32, x0 as u32, (x1 - x0 + 1) as u32);
        }
    }

    /// A disc filled from a vertical ramp, top to bottom of the disc itself.
    ///
    /// `fill_circle` with a colour that varies by row. The ramp is measured
    /// across the disc rather than across the screen, so the same table gives
    /// the same sphere at any size -- which is what separates a lit body from
    /// a circle cut out of a background gradient.
    pub fn fill_circle_ramp(&self, cx: i32, cy: i32, r: i32, stops: &[(u8, Color)]) {
        if r <= 0 {
            return;
        }
        let span = (2 * r + 1) as u32;
        for dy in -r..=r {
            let y = cy + dy;
            if y < 0 || y as u32 >= self.height {
                continue;
            }
            let half = isqrt((r * r - dy * dy).max(0) as u32) as i32;
            let x0 = (cx - half).max(0);
            let w = (cx + half - x0).max(0) as u32;
            let raw = self.encode(ramp_at(stops, (dy + r) as u32, span));
            self.fill_span(x0 as u32, y as u32, w, raw);
        }
    }

    pub fn fill_triangle(&self, a: (i32, i32), b: (i32, i32), c: (i32, i32), col: Color) {
        let raw = self.encode(col);
        self.tri_spans(a, b, c, |y, x, w| self.fill_span(x, y, w, raw));
    }

    /// Fill a triangle with the colour a vertical ramp would have put there.
    ///
    /// For cutting a shape out of something drawn over a gradient: the hole
    /// gets exactly what the gradient would have given that row, which is the
    /// same thing as never having drawn over it. A logo whose blades are gaps
    /// needs this, because one background colour belongs to no part of a sky
    /// and shows as a rectangle of the wrong shade wherever it is put.
    ///
    /// `top` and `height` are the ramp's own, not the triangle's, so a hole
    /// anywhere in a full-screen gradient lines up with it.
    pub fn fill_triangle_over(
        &self,
        a: (i32, i32),
        b: (i32, i32),
        c: (i32, i32),
        stops: &[(u8, Color)],
        top: u32,
        height: u32,
    ) {
        self.tri_spans(a, b, c, |y, x, w| {
            let raw = self.encode(ramp_at(stops, y.saturating_sub(top), height));
            self.fill_span(x, y, w, raw);
        });
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
/// Time the renderer, robustly enough to act on.
///
/// **Best of several, not one run.** The first version timed each operation
/// once and the numbers were unusable: two consecutive `desk::draw` calls
/// measured 2,679 us and 24,318 us on an idle machine. Under emulation the
/// host's scheduler decides when the guest runs, so a single sample measures
/// the host. Noise only ever *adds* time, so the minimum over a handful of
/// runs is the honest estimate of what the code costs; the maximum is printed
/// beside it so the spread is visible rather than hidden.
/// What one frame costs, in microseconds, best of nine.
///
/// A struct rather than five printed lines, because `bench report` needs the
/// numbers and a person needs the table -- and two measurements of the same
/// thing is how two answers to "did this regress" come to disagree. One
/// `measure`, two consumers.
/// Whether writes to the aperture use non-temporal stores. On by default;
/// `video stream on|off` switches it for an A/B within one boot.
pub static STREAM: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Copy `n` pixels to write-combining memory with non-temporal stores.
///
/// The head is copied plainly up to a 16-byte boundary, because `movntdq`
/// faults on an unaligned destination; the body goes four pixels a store; the
/// tail plainly again. SSE2 is part of the x86-64 baseline and the kernel is
/// built with it on, so this needs no feature gate.
///
/// # Safety
/// `dst` and `src` must be valid for `n` pixels and must not overlap.
unsafe fn stream_copy(dst: *mut u32, src: *const u32, n: usize) {
    use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
    let mut i = 0;
    unsafe {
        while i < n && (dst.add(i) as usize) & 15 != 0 {
            dst.add(i).write(*src.add(i));
            i += 1;
        }
        while i + 4 <= n {
            let v = _mm_loadu_si128(src.add(i) as *const __m128i);
            _mm_stream_si128(dst.add(i) as *mut __m128i, v);
            i += 4;
        }
        while i < n {
            dst.add(i).write(*src.add(i));
            i += 1;
        }
        _mm_sfence();
    }
}

#[derive(Clone, Copy)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// One span-filled rectangle over the whole target. The control: nothing
    /// above the framebuffer can touch it, so what it moves by between two
    /// builds is the noise floor.
    pub rect_us: (u64, u64),
    pub draw_us: (u64, u64),
    /// A present with nothing changed, which is the diff alone and the floor
    /// for any repaint that turns out to be a no-op.
    pub present_us: (u64, u64),
    /// ...and one where every row differs, which is the worst case.
    pub fill_present_us: (u64, u64),
    /// The terminal's own cost, which is most of what a full `draw` is.
    pub console_us: (u64, u64),
    /// One pointer move, through the path a real packet takes.
    pub cursor_us: (u64, u64),
    /// A frame where nothing changed: building and diffing the scene, and the
    /// parts that repaint whenever a frame is composed at all.
    pub idle_us: (u64, u64),
    /// A 64x64 change -- a hover, a caret -- composed and presented.
    pub small_us: (u64, u64),
    /// An 800x600 window moved ten pixels: where it was and where it is.
    pub move_us: (u64, u64),
}

/// Time the graphics path. `None` when there is no framebuffer to time.
///
/// **Only comparable against a run with the same text on screen**, because
/// `console::redraw_all` skips blank cells and so scales with how much output
/// is sitting in the terminal. The same build measured 497 us after a bare
/// boot and 823 us after `diag all` had filled the console. Take the before
/// and after with the identical command prefix, and read `rect_us` as the
/// control.
pub fn measure() -> Option<Frame> {
    let fb = primary()?;
    let mhz = crate::time::tsc_mhz().max(1);
    const RUNS: usize = 9;

    let mut best_of = |f: &mut dyn FnMut()| -> (u64, u64) {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for _ in 0..RUNS {
            let t = crate::time::rdtsc();
            f();
            let d = (crate::time::rdtsc() - t) / mhz;
            lo = lo.min(d);
            hi = hi.max(d);
        }
        (lo, hi)
    };

    // The compositor stands down for the length of the measurement. It runs
    // on its own task, and every `damage` below wakes it: left running it
    // composed the forced whole frame itself, and the `draw` being timed
    // either found nothing left to do or lost the paint claim and returned at
    // once -- `video.draw` read 0 us. Turning it off and waiting for the claim
    // to be free means the frame timed is the frame asked for.
    let was = render::enabled();
    render::set_enabled(false);
    while desk::claim_holder().is_some() {
        crate::task::yield_now();
    }
    let target = compose::target().unwrap_or_else(|| fb.clone());
    let (w, h) = (target.width(), target.height());
    let out = Frame {
        width: w,
        height: h,
        rect_us: best_of(&mut || target.rect(0, 0, w, h, theme::DESKTOP)),
        // A whole frame, forced. Since frames became damage-limited a bare
        // `draw` composes only what changed -- nothing, here -- and this rail
        // kept its name while measuring that instead, which made every
        // comparison across that change a comparison of two different things.
        draw_us: best_of(&mut || {
            render::damage_all();
            desk::draw()
        }),
        present_us: best_of(&mut || compose::present()),
        fill_present_us: best_of(&mut || {
            target.rect(0, 0, w, h, theme::APERTURE);
            compose::present();
        }),
        console_us: best_of(&mut || console::redraw()),
        idle_us: best_of(&mut || desk::draw()),
        small_us: best_of(&mut || {
            render::damage(theme::Rect::new(w / 2, h / 2, 64, 64));
            desk::draw()
        }),
        move_us: best_of(&mut || {
            let (mw, mh) = (800.min(w), 600.min(h));
            render::damage(theme::Rect::new(w / 4, h / 4, mw, mh));
            render::damage(theme::Rect::new(w / 4 + 10, h / 4, mw, mh));
            desk::draw()
        }),
        cursor_us: {
            let mut step = 0u32;
            best_of(&mut || {
                step += 1;
                desk::bench_cursor(step);
            })
        },
    };
    desk::cursor_home();
    render::damage_all();
    desk::draw();
    render::set_enabled(was);
    Some(out)
}

pub fn bench() {
    use crate::kprintln;
    let Some(m) = measure() else {
        kprintln!("  no framebuffer");
        return;
    };
    kprintln!(
        "  {}x{}  {} pixels, best of 9",
        m.width,
        m.height,
        m.width as u64 * m.height as u64
    );
    for (what, (lo, hi)) in [
        ("full-screen rect     ", m.rect_us),
        ("desk::draw + present ", m.draw_us),
        ("present, no change   ", m.present_us),
        ("fill + present, all  ", m.fill_present_us),
        ("console redraw_all   ", m.console_us),
        ("pointer move         ", m.cursor_us),
        ("frame, nothing moved ", m.idle_us),
        ("frame, 64x64 changed ", m.small_us),
        ("frame, window moved  ", m.move_us),
    ] {
        kprintln!("  {} {:>7} us  (max {})", what, lo, hi);
    }
}


/// Everything about turning characters into pixels.
///
/// One suite rather than two, because the halves are only correct together:
/// a decoder that produces the right codepoints and a table that draws the
/// wrong glyph for them is a console that is wrong, and either half on its
/// own would report itself fine.
pub fn text_selftest() -> bool {
    let a = font::selftest();
    let b = console::selftest();
    a && b
}

/// The three primitives the chrome is drawn out of.
///
/// Pure arithmetic over a scratch surface, so it needs no framebuffer, no
/// compositor and no screen. That is the point of testing these here rather
/// than by looking: a gradient that is one shade wrong is invisible in a
/// screenshot and a shadow that bleeds one channel into the next is a colour
/// nobody would think to check.
/// The memory-type decoding behind `video cache`, against synthetic registers.
pub fn pat_selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    crate::cpu::mtrr::checks(&mut claim);
    // And the live machine: every core that came up holds the same PAT, and it
    // is the one the framebuffer's mapping was chosen against.
    if crate::cpu::mtrr::programmed() {
        let seen = crate::cpu::mtrr::per_core();
        claim(
            "every online core holds the PAT the bootstrap processor programmed",
            seen.iter().all(|v| *v == crate::cpu::mtrr::GLADOS_PAT),
        );
        if let Some(fb) = primary() {
            let slot = crate::mem::paging::pat_slot(fb.addr());
            claim("the framebuffer's leaf selects the write-combining slot", slot == Some(1));
        }
    } else {
        kprintln!("  ....  the PAT was not programmed on this machine, so 2 claim(s) about it did not run");
    }
    ok
}

/// What the processor really does with a store to the framebuffer.
///
/// Walks the aperture a 2 MiB step at a time, because the page tables map it
/// in 2 MiB entries and the MTRRs may split it anywhere, and prints each run
/// of identical answers once. Under a hypervisor the registers are virtual and
/// the emulator does not model caches, so the answer is about the guest's
/// configuration and says nothing about speed -- which is printed rather than
/// left to be remembered.
pub fn cache_report() {
    use crate::cpu::mtrr;
    use crate::kprintln;
    let Some(fb) = primary() else {
        kprintln!("  no framebuffer");
        return;
    };
    let Some(snap) = mtrr::read() else {
        kprintln!("  this processor does not report a PAT and MTRRs; nothing to decode");
        return;
    };
    let base = fb.addr();
    let len = fb.stride() as u64 * fb.height() as u64 * 4;
    kprintln!("  aperture  {:#x} .. {:#x}  ({} KiB)", base, base + len, len / 1024);
    kprintln!("  PAT       {:#018x}{}", snap.pat,
              if snap.pat == mtrr::PAT_POWER_ON { "  (power-on layout: no WC slot)" } else { "" });
    kprintln!("  MTRRs     {}, default {}, {} variable range(s), WC {}",
              if snap.def_type & (1 << 11) != 0 { "enabled" } else { "DISABLED" },
              mtrr::Kind::from_mtrr((snap.def_type & 0xFF) as u8).short(),
              snap.ranges.iter().filter(|r| r.valid()).count(),
              if snap.wc_supported { "supported" } else { "not supported" });

    let step = 2 * 1024 * 1024u64;
    let mut at = base & !(step - 1);
    let mut run: Option<(u64, u8, mtrr::Kind, mtrr::Kind)> = None;
    let flush = |r: (u64, u8, mtrr::Kind, mtrr::Kind), end: u64| {
        let (from, slot, p, m) = r;
        kprintln!("    {:#x}..{:#x}  page slot {} = {:<3}  mtrr {:<3}  -> {}",
                  from, end, slot, p.short(), m.short(), mtrr::effective(p, m).name());
    };
    while at < base + len {
        let probe = at.max(base);
        let slot = crate::mem::paging::pat_slot(probe).unwrap_or(0xFF);
        let p = if slot == 0xFF { mtrr::Kind::Reserved(0xFF) } else { mtrr::pat_kind(snap.pat, slot) };
        let m = mtrr::mtrr_kind(snap.def_type, &snap.ranges, probe).unwrap_or(mtrr::Kind::Reserved(0xFE));
        match run {
            Some(r) if r.1 == slot && r.2 == p && r.3 == m => {}
            Some(r) => {
                flush(r, probe);
                run = Some((probe, slot, p, m));
            }
            None => run = Some((probe, slot, p, m)),
        }
        at += step;
    }
    if let Some(r) = run {
        flush(r, base + len);
    }
    if let Some(name) = crate::cpu::hypervisor_name() {
        kprintln!("  running under {}: these are its registers, and it does not model caches", name);
    }
}

pub fn paint_selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    const W: u32 = 16;
    const H: u32 = 16;
    let mut buf = alloc::vec![0u32; (W * H) as usize];
    // The same construction `compose::init` makes, at a size that fits on the
    // boot stack's conscience.
    let fb = unsafe { Framebuffer::over_ram(buf.as_mut_ptr() as u64, W, H, W, Format::Bgrx) };

    // --- shade ------------------------------------------------------------
    let white = fb.encode(Color::new(0xFF, 0xFF, 0xFF));
    fb.rect(0, 0, W, H, Color::new(0xFF, 0xFF, 0xFF));
    fb.shade_span(0, 0, W, 256);
    claim("shading by 256 leaves a pixel alone", buf[0] == white);

    fb.rect(0, 0, W, H, Color::new(0xFF, 0xFF, 0xFF));
    fb.shade_span(0, 0, W, 0);
    claim("and by 0 takes it to black", buf[0] == 0);

    // --- clipping and views -------------------------------------------------
    // The property damage-limited repaint rests on: a painter handed a
    // clipped surface draws whatever it likes and only the rectangle changes.
    let black = fb.encode(Color::new(0, 0, 0));
    let red = Color::new(0xFF, 0, 0);
    let red_raw = fb.encode(red);
    fb.rect(0, 0, W, H, Color::new(0, 0, 0));
    let c = fb.clipped(4, 4, 6, 5);
    c.rect(0, 0, W, H, red);
    let inside = (4..10).all(|x| (4..9).all(|y| buf[(y * W + x) as usize] == red_raw));
    let outside = (0..W).all(|x| (0..H).all(|y| {
        (4..10).contains(&x) && (4..9).contains(&y) || buf[(y * W + x) as usize] == black
    }));
    claim("a clipped surface fills exactly its rectangle", inside);
    claim("and leaves every pixel outside it alone", outside);
    claim("clipping twice intersects rather than widening",
          !c.clipped(0, 0, W, H).touches(0, 0, 4, 4) && c.clipped(0, 0, W, H).touches(5, 5, 1, 1));
    fb.rect(0, 0, W, H, Color::new(0, 0, 0));
    if let Some(v) = fb.view(3, 2, 5, 4) {
        v.put(0, 0, red_raw);
        v.rect(0, 0, 100, 100, red);
        claim("a view's (0,0) is its corner on the parent", buf[(2 * W + 3) as usize] == red_raw);
        let only = (0..W).all(|x| (0..H).all(|y| {
            (3..8).contains(&x) && (2..6).contains(&y) || buf[(y * W + x) as usize] == black
        }));
        claim("and a view cannot draw past its own edge, whatever it asks for", only);
        claim("a view knows its own size", v.width() == 5 && v.height() == 4);
        // A red neighbour first, so a fill that ran past the view would show.
        fb.put(2, 2, red_raw);
        fb.put(8, 2, red_raw);
        v.fill(Color::new(0, 0, 0));
        claim("filling a view black clears the view and nothing beside it",
              buf[(2 * W + 3) as usize] == black
                  && buf[(2 * W + 2) as usize] == red_raw
                  && buf[(2 * W + 8) as usize] == red_raw);
    } else {
        claim("a view of an on-surface rectangle exists", false);
    }
    claim("a view off the surface is refused", fb.view(W, 0, 1, 1).is_none());

    fb.rect(0, 0, W, H, Color::new(0xFF, 0xFF, 0xFF));
    fb.shade_span(0, 0, W, 128);
    claim(
        "half of white is grey in every channel",
        buf[0] == fb.encode(Color::new(0x7F, 0x7F, 0x7F)),
    );

    // The claim the two-halves trick exists to earn. A channel at zero
    // between two at full is where a carry would land, and it would read as a
    // plausible colour rather than as an error.
    fb.rect(0, 0, W, H, Color::new(0xFF, 0x00, 0xFF));
    fb.shade_span(0, 0, W, 128);
    claim(
        "no channel carries into its neighbour",
        buf[0] == fb.encode(Color::new(0x7F, 0x00, 0x7F)),
    );

    fb.rect(0, 0, W, H, Color::new(0xFF, 0xFF, 0xFF));
    fb.shade_span(0, 0, W, 255);
    claim("and the top byte stays clear", buf[0] & 0xFF00_0000 == 0);

    // A run stops where it was told to. A shadow band that ran on would
    // darken the window it belongs to.
    fb.rect(0, 0, W, H, Color::new(0xFF, 0xFF, 0xFF));
    fb.shade_span(0, 0, 4, 0);
    claim(
        "a shaded run ends where it was asked to",
        buf[3] == 0 && buf[4] == white,
    );

    // --- tint -------------------------------------------------------------
    let red = Color::new(0xFF, 0x00, 0x00);
    fb.rect(0, 0, W, H, Color::new(0x00, 0x00, 0xFF));
    fb.tint_span(0, 0, W, red, 0);
    claim(
        "tinting by 0 leaves the backdrop alone",
        buf[0] == fb.encode(Color::new(0x00, 0x00, 0xFF)),
    );

    fb.rect(0, 0, W, H, Color::new(0x00, 0x00, 0xFF));
    fb.tint_span(0, 0, W, red, 256);
    claim("and by 256 replaces it outright", buf[0] == fb.encode(red));

    // The claim the no-carry argument earns. Blending a full channel against
    // an empty one is where a spill would land, and it would read as a
    // plausible colour rather than as an error.
    fb.rect(0, 0, W, H, Color::new(0x00, 0xFF, 0x00));
    fb.tint_span(0, 0, W, Color::new(0xFF, 0x00, 0xFF), 128);
    claim(
        "half and half carries into no neighbour",
        buf[0] == fb.encode(Color::new(0x7F, 0x7F, 0x7F)),
    );

    // Glass is a ramp of opacity rather than of colour, so its ends are the
    // backdrop and the pane, not two shades of one thing.
    let sky = Color::new(0x00, 0x00, 0x00);
    fb.rect(0, 0, W, H, Color::new(0xFF, 0xFF, 0xFF));
    fb.glass(0, 0, W, H, sky, &[(0, 0), (255, 256)]);
    claim(
        "glass keeps everything at one end and nothing at the other",
        buf[0] == fb.encode(Color::new(0xFF, 0xFF, 0xFF))
            && buf[((H - 1) * W) as usize] == fb.encode(sky),
    );

    // --- vgrad ------------------------------------------------------------
    let a = Color::new(0x10, 0x20, 0x30);
    let b = Color::new(0xF0, 0xE0, 0xD0);
    fb.rect(0, 0, W, H, Color::new(0, 0, 0));
    fb.vgrad(0, 0, W, H, &[(0, a), (255, b)]);
    claim("a ramp starts on its first stop", buf[0] == fb.encode(a));
    claim(
        "and ends exactly on its last, rather than near it",
        buf[((H - 1) * W) as usize] == fb.encode(b),
    );
    claim(
        "and moves in between",
        buf[(8 * W) as usize] != fb.encode(a) && buf[(8 * W) as usize] != fb.encode(b),
    );

    // The load-bearing one. Luna's caption breaks across the middle, and a
    // table that smoothed the break would turn every gradient in the
    // interface into a plain fade without anything reporting it.
    fb.rect(0, 0, W, H, Color::new(0, 0, 0));
    fb.vgrad(0, 0, W, H, &[(0, a), (128, a), (129, b), (255, b)]);
    let mid = (H / 2) as usize;
    claim(
        "two stops one apart are a step and not a ramp",
        buf[(mid - 1) * W as usize] == fb.encode(a) && buf[mid * W as usize] == fb.encode(b),
    );

    fb.rect(0, 0, W, H, Color::new(0, 0, 0));
    fb.vgrad(0, 0, W, H, &[(0, a)]);
    claim("one stop fills flat", buf[0] == fb.encode(a) && buf[(15 * W) as usize] == fb.encode(a));

    // --- shape ------------------------------------------------------------
    let top = Shape::round_top(40, 40, 8);
    let both = Shape::round_rect(40, 40, 8);
    claim(
        "a rounded top insets its first row",
        top.span(0).map(|(x0, _)| x0 > 0) == Some(true),
    );
    claim(
        "and leaves the last row square",
        top.span(39) == Some((0, 40)),
    );
    // The canary: the two forms must actually differ, or `round_top` is a
    // name for `round_rect` and every window would be rounded at the bottom
    // with nothing saying so.
    claim(
        "where a fully rounded one does not",
        both.span(39) != top.span(39),
    );
    claim(
        "a radius of zero is a plain rectangle",
        Shape::round_top(40, 40, 0).span(0) == Some((0, 40)),
    );

    ok
}
