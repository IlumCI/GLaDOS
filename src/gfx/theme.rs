//! Windows 3.1 chrome, in Aperture colours.
//!
//! One place that owns what the system looks like, so a widget is a few calls
//! rather than a pile of rectangles, and so changing the look is changing this
//! file rather than every caller.
//!
//! ### Why the bevel is two pixels
//!
//! The 3.1 look is not "a border" -- it is a specific claim about where the
//! light is. Every raised surface is lit from the top left: an outer white
//! edge, an inner light-grey edge, an inner dark-grey shadow, an outer black
//! shadow. Sunken surfaces state the reverse. One-pixel edges read as a line
//! drawn around a box; two-pixel edges read as a solid object with a thickness,
//! and that is the entire difference between "themed" and "chrome".
//!
//! The face colour matters for the same reason. 0xC0C0C0 is bright enough that
//! white reads as a highlight against it and dark grey reads as shadow. Against
//! the console palette's 0xAAAAAA the highlight is too close and the surface
//! goes flat, which is why these colours are their own set rather than indices
//! into the sixteen.
//!
//! ### Why Aperture rather than navy
//!
//! Everything structural is 3.1. The one thing that is not is the title bar,
//! which is the only surface a user looks at to know *whose* machine this is.

use super::font;
use super::{Color, Framebuffer};

// --- surfaces ------------------------------------------------------------

/// The face of every raised control. The canonical 3.1 grey.
pub const FACE: Color = Color::new(0xC0, 0xC0, 0xC0);
/// Outer highlight of a raised edge, inner shadow of a sunken one.
pub const HILIGHT: Color = Color::new(0xFF, 0xFF, 0xFF);
/// Inner shadow of a raised edge.
pub const SHADOW: Color = Color::new(0x80, 0x80, 0x80);
/// Outer shadow. Black, not dark grey -- 3.1 frames really do end in black.
pub const DARKEDGE: Color = Color::new(0x00, 0x00, 0x00);
/// The desktop behind everything.
pub const DESKTOP: Color = Color::new(0x0A, 0x0C, 0x10);
/// A sparse dot grid over it. Barely there on purpose: a wall should read
/// as a surface rather than compete with the windows on it.
pub const DESKTOP_GRID: Color = Color::new(0x1C, 0x22, 0x2C);
/// The Aperture mark on the wall. A muted orange rather than the title bar's:
/// at a fifth of the screen it would otherwise be the brightest thing on the
/// desktop, and a wallpaper that outshines the windows is one nobody can work
/// in front of.
pub const WALL_MARK: Color = Color::new(0x4A, 0x2E, 0x12);
pub const TEXT: Color = Color::new(0x00, 0x00, 0x00);
pub const TEXT_DIM: Color = Color::new(0x80, 0x80, 0x80);

// --- Aperture ------------------------------------------------------------

pub const APERTURE: Color = Color::new(0xF2, 0x8C, 0x1E);
pub const APERTURE_DEEP: Color = Color::new(0x9A, 0x50, 0x0C);
pub const TITLE_TEXT: Color = Color::new(0xFF, 0xFF, 0xFF);
/// An unfocused title bar. Grey rather than a dimmer orange, so "which window
/// has the keyboard" is answered by hue and not by brightness.
pub const TITLE_IDLE: Color = Color::new(0x80, 0x80, 0x80);

/// The console's own background, inside the terminal window's well.
pub const SCREEN: Color = Color::new(0x0A, 0x0C, 0x10);

/// Selection bar in a list, and the fill of a focused default button.
pub const SELECT: Color = APERTURE;
pub const SELECT_TEXT: Color = Color::new(0x00, 0x00, 0x00);

/// Reading colours for a page drawn in a screen well. Body is not pure white:
/// a wall of #FFF on near-black glares, and a browser is the one program here
/// somebody reads for minutes at a time.
pub const SCREEN_TEXT: Color = Color::new(0xC8, 0xC8, 0xC8);
pub const LINK: Color = APERTURE;
pub const HEADING: Color = HILIGHT;

pub const CHROME_SCALE: u32 = 2;
/// Height of a title bar at `CHROME_SCALE`, with room above and below the text.
pub const TITLE_H: u32 = font::GLYPH_H * CHROME_SCALE + 8;
/// Thickness of a window's outer frame.
pub const FRAME: u32 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
    /// Shrink by `n` on every side. Saturating, so an inset larger than the
    /// rect gives an empty one rather than wrapping into a huge one.
    pub fn shrink(&self, n: u32) -> Self {
        Self {
            x: self.x + n,
            y: self.y + n,
            w: self.w.saturating_sub(2 * n),
            h: self.h.saturating_sub(2 * n),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// Two-pixel raised or sunken edge. See the module note on why it is two.
pub fn bevel(fb: &Framebuffer, r: Rect, raised: bool) {
    if r.w < 4 || r.h < 4 {
        return;
    }
    let (outer_tl, inner_tl, inner_br, outer_br) = if raised {
        (HILIGHT, FACE, SHADOW, DARKEDGE)
    } else {
        (SHADOW, DARKEDGE, FACE, HILIGHT)
    };
    for (i, (tl, br)) in [(outer_tl, outer_br), (inner_tl, inner_br)].iter().enumerate() {
        let i = i as u32;
        let (x, y) = (r.x + i, r.y + i);
        let (w, h) = (r.w - 2 * i, r.h - 2 * i);
        // Top and left first, then bottom and right, so the corners belong to
        // the shadow. That is what makes a corner read as a mitre.
        fb.rect(x, y, w, 1, *tl);
        fb.rect(x, y, 1, h, *tl);
        fb.rect(x, y + h - 1, w, 1, *br);
        fb.rect(x + w - 1, y, 1, h, *br);
    }
}

/// A raised surface: face plus a raised edge. Buttons, panels, menu bars.
pub fn panel(fb: &Framebuffer, r: Rect) {
    fb.rect(r.x, r.y, r.w, r.h, FACE);
    bevel(fb, r, true);
}

/// A sunken surface. Text fields, list boxes, anything content sits *in*.
pub fn well(fb: &Framebuffer, r: Rect, fill: Color) {
    fb.rect(r.x, r.y, r.w, r.h, fill);
    bevel(fb, r, false);
}

pub fn text(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color, bg: Color) {
    fb.draw_text(x, y, s, fg, bg, CHROME_SCALE);
}

/// Height of one line of chrome text, the companion to `text_w`.
pub fn text_h() -> u32 {
    font::GLYPH_H * CHROME_SCALE
}

pub fn text_w(len: usize) -> u32 {
    len as u32 * font::GLYPH_W * CHROME_SCALE
}

/// The Aperture mark, small enough for a title bar.
///
/// Not the full boot logo: at this size the six-slash aperture turns to mush,
/// so this is the ring and the hexagonal opening only, which is what stays
/// legible when it is sixteen pixels across.
fn mark(fb: &Framebuffer, cx: i32, cy: i32, r: i32, fg: Color) {
    if r < 3 {
        return;
    }
    fb.circle_thick(cx, cy, r, 2, fg);
    let inner = (r * 45 / 100).max(2);
    fb.fill_circle(cx, cy, inner, fg);
}

/// Draw a window: frame, title bar, and a raised body. Returns the client area.
///
/// The caller gets a rectangle and no further obligations -- everything that
/// makes it look like a window has already happened.
pub fn window(fb: &Framebuffer, r: Rect, title: &str, active: bool) -> Rect {
    // Outer frame: a raised slab, then a groove, which is how 3.1 gets a border
    // thick enough to grab without looking like a picture frame.
    panel(fb, r);
    let inner = r.shrink(FACE_INSET);
    if inner.is_empty() {
        return inner;
    }

    let bar = Rect::new(inner.x, inner.y, inner.w, TITLE_H);
    title_bar(fb, bar, title, active);

    let client = Rect::new(
        inner.x,
        inner.y + TITLE_H + 2,
        inner.w,
        inner.h.saturating_sub(TITLE_H + 2),
    );
    client
}

const FACE_INSET: u32 = FRAME;

pub fn title_bar(fb: &Framebuffer, r: Rect, title: &str, active: bool) {
    let base = if active { APERTURE } else { TITLE_IDLE };
    fb.rect(r.x, r.y, r.w, r.h, base);

    // A one-step gradient rather than a smooth one: 3.1 had no gradient at all,
    // and a full ramp reads as XP. This is just enough to stop a large bar
    // looking like a flat sticker.
    if active {
        let band = r.h / 3;
        fb.rect(r.x, r.y + r.h - band, r.w, band, APERTURE_DEEP);
    }

    let pad = 6;
    let cy = r.y + r.h / 2;
    mark(fb, (r.x + pad + 8) as i32, cy as i32, (r.h as i32 / 2) - 4, TITLE_TEXT);

    let tx = r.x + pad + 22;
    let ty = r.y + (r.h - font::GLYPH_H * CHROME_SCALE) / 2;
    // Clip by characters rather than pixels: the text is drawn glyph by glyph
    // anyway, and a half-drawn glyph at the edge looks like a rendering fault.
    let room = (r.w.saturating_sub(tx - r.x + pad) / (font::GLYPH_W * CHROME_SCALE)) as usize;
    let shown = if title.len() > room { &title[..room] } else { title };
    // The title bar is its own background, so glyphs are drawn over the bar
    // colour rather than over the face.
    let bg = if active && ty + font::GLYPH_H * CHROME_SCALE > r.y + r.h - r.h / 3 {
        APERTURE_DEEP
    } else {
        base
    };
    text(fb, tx, ty, shown, TITLE_TEXT, bg);
}

/// A button. `focused` draws the keyboard focus; `default` marks the one Enter
/// would press if focus were elsewhere.
pub fn button(fb: &Framebuffer, r: Rect, label: &str, focused: bool, pressed: bool) {
    fb.rect(r.x, r.y, r.w, r.h, FACE);
    bevel(fb, r, !pressed);

    // 3.1 marked the default button with an extra black rectangle just inside
    // the edge. Reused here for keyboard focus, which is the thing that
    // actually needs to be visible on a machine with no pointer.
    if focused {
        let f = r.shrink(3);
        if !f.is_empty() {
            fb.frame(f.x, f.y, f.w, f.h, DARKEDGE);
        }
    }

    let tw = text_w(label.len());
    let tx = r.x + (r.w.saturating_sub(tw)) / 2 + u32::from(pressed);
    let ty = r.y + (r.h.saturating_sub(font::GLYPH_H * CHROME_SCALE)) / 2 + u32::from(pressed);
    text(fb, tx, ty, label, TEXT, FACE);
}

/// One row of a list box. Selected rows invert to the Aperture bar.
pub fn list_row(fb: &Framebuffer, r: Rect, label: &str, selected: bool, focused: bool) {
    let (fg, bg) = if selected && focused {
        (SELECT_TEXT, SELECT)
    } else if selected {
        // Selected but the list does not have focus: keep the bar, drop the
        // colour, so a form with several lists still says which one is live.
        (TEXT, Color::new(0xA8, 0xA8, 0xA8))
    } else {
        (TEXT, FACE)
    };
    fb.rect(r.x, r.y, r.w, r.h, bg);
    let ty = r.y + (r.h.saturating_sub(font::GLYPH_H * CHROME_SCALE)) / 2;
    let room = (r.w / (font::GLYPH_W * CHROME_SCALE)).saturating_sub(1) as usize;
    let shown = if label.len() > room { &label[..room] } else { label };
    text(fb, r.x + 6, ty, shown, fg, bg);
}

/// A vertical groove, for dividing a bar into sections.
pub fn separator_v(fb: &Framebuffer, x: u32, y: u32, h: u32) {
    fb.rect(x, y, 1, h, SHADOW);
    fb.rect(x + 1, y, 1, h, HILIGHT);
}

/// The Aperture mark at button size, on a raised face.
///
/// Small enough that the six blades are mush, so this is the ring and the
/// opening -- the same reduction `mark` makes for a title bar, exposed because
/// the app bar wants it too.
pub fn aperture_dot(fb: &Framebuffer, cx: u32, cy: u32, r: i32) {
    mark(fb, cx as i32, cy as i32, r.max(3), APERTURE_DEEP);
}

/// A horizontal rule, drawn as a groove. The 3.1 separator.
pub fn separator(fb: &Framebuffer, x: u32, y: u32, w: u32) {
    fb.rect(x, y, w, 1, SHADOW);
    fb.rect(x, y + 1, w, 1, HILIGHT);
}
