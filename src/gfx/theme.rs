//! Luna chrome, in Aperture colours.
//!
//! One place that owns what the system looks like, so a widget is a few calls
//! rather than a pile of rectangles, and so changing the look is changing this
//! file rather than every caller. That sentence is also the reason there is no
//! theme struct here and should not be: a scheme selectable at runtime would
//! put an indirection in front of every chrome draw to serve a switch nobody
//! has asked for twice.
//!
//! ### Why Aperture rather than Luna's blue
//!
//! XP shipped three schemes and two of them were not blue. Olive Green and
//! Silver are the same geometry, the same gradients and the same gloss with
//! one hue substituted, so keeping this machine's orange is what XP itself
//! did rather than a departure from it. The Start button stays green, as it
//! was in every one of the three.
//!
//! ### Why the bevel stopped being two pixels
//!
//! It was two, and the argument was sound while it held: the 3.1 look is a
//! claim about where the light is, an outer white edge and an inner light
//! grey against an outer black and an inner dark grey, and two pixels read as
//! an object with a thickness where one reads as a line drawn round a box.
//!
//! Two pixels needs two distinguishable lights, and against XP's warmer face
//! the inner one has nowhere to be. `EDGE_LIGHT` -- XP's own ButtonLight,
//! 0xF1EFE2 -- differs from the 0xECE9D8 face by **5, 6 and 10** across the
//! three channels, which is not a highlight, it is the same colour. So the
//! second pixel had nothing left to say, and one edge says what remains.
//!
//! **The highlight is still white, and a screenshot is what settled that.**
//! The first cut of this used `EDGE_LIGHT` for the raised edge on the
//! reasoning that white had stopped reading as a highlight too. It has not:
//! white against this face is 19, 22 and 39, and the difference between those
//! two numbers is the difference between a Minesweeper board of buttons and a
//! Minesweeper board of thin dark boxes, which is exactly what the first
//! screenshot showed. XP raises with ButtonHighlight and not with ButtonLight
//! for the same reason.
//!
//! `EDGE_LIGHT` keeps its place on the separators, which is where XP uses
//! ButtonLight and where it sits against a line rather than against a field.
//!
//! ### Why `SHADOW` and `DARKEDGE` kept their old values
//!
//! They are not only the halves of a bevel. Fourteen places outside this file
//! draw *with* them -- pictogram detail, the icon label shadow on the wall,
//! Minesweeper's counts, the Oracle's plot ink, ToDo's field labels, Paint's
//! pen-up mark. Retinting them to XP's greys would wash out ten drawings that
//! have nothing to do with edges. Keeping a name only helps when the name
//! means one thing, so the bevel got new constants instead.

use super::font;
use super::{Color, Framebuffer};

// --- surfaces ------------------------------------------------------------

/// The face of every raised control. XP's ButtonFace, and warm, which is the
/// same family the orange is in.
pub const FACE: Color = Color::new(0xEC, 0xE9, 0xD8);
/// A white field. Not the light half of a bevel -- that is `EDGE_LIGHT` --
/// but the fill behind text somebody types into, which is what the ten
/// callers outside this file mean by it.
pub const HILIGHT: Color = Color::new(0xFF, 0xFF, 0xFF);
/// A mid grey, used as *ink* by ten drawings across five files. See the
/// module note; it is not the bevel's dark half any more.
pub const SHADOW: Color = Color::new(0x80, 0x80, 0x80);
/// Black. Still black: the icon labels on the wall are shadowed with it and
/// every pictogram is drawn in it.
pub const DARKEDGE: Color = Color::new(0x00, 0x00, 0x00);

/// The dark half of a one-pixel edge. XP's ButtonShadow. The light half is
/// `HILIGHT`, which is white -- see the module note on what a screenshot said
/// about trying to use the softer one here.
pub const EDGE: Color = Color::new(0xAC, 0xA8, 0x99);
/// XP's ButtonLight. Only the separators, where it sits against a line and
/// not against the face.
pub const EDGE_LIGHT: Color = Color::new(0xF1, 0xEF, 0xE2);
/// The border round a field. One colour all the way round rather than a
/// light and dark pair: a well is a hole in the surface, and XP says so by
/// outlining it in a desaturated accent instead of by lighting it.
pub const WELL_EDGE: Color = Color::new(0xB0, 0x8A, 0x5E);
/// A selected row in a list that does not have focus. Was an anonymous
/// 0xA8A8A8 written out three times in two files.
pub const LIST_SEL_IDLE: Color = Color::new(0xD8, 0xD4, 0xC8);
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
pub const TEXT_DIM: Color = Color::new(0x8A, 0x86, 0x7A);
/// The close button under the pointer.
///
/// The one control on a window that cannot be undone, and the one an operator
/// coming from Windows identifies by colour before they read the glyph. Every
/// other button here answers hover with a ring; this one answers with red,
/// because "this is the destructive one" is worth saying twice.
pub const CLOSE_HOT: Color = Color::new(0xE0, 0x43, 0x43);

/// Status text on a panel face: working, needs attention, will not work.
///
/// Three colours and no more. A settings page that reports state in prose
/// makes the operator read every line to find the one that is wrong; a colour
/// lets them find it first and read it second. Chosen dark enough to stay
/// legible on FACE rather than picked for vividness.
pub const OK_TEXT: Color = Color::new(0x0A, 0x64, 0x0A);
pub const WARN_TEXT: Color = Color::new(0x8A, 0x55, 0x00);
pub const BAD_TEXT: Color = Color::new(0xA0, 0x1C, 0x1C);

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

/// One-pixel raised or sunken edge. See the module note on why it stopped
/// being two.
pub fn bevel(fb: &Framebuffer, r: Rect, raised: bool) {
    if r.w < 2 || r.h < 2 {
        return;
    }
    let (tl, br) = if raised { (HILIGHT, EDGE) } else { (EDGE, HILIGHT) };
    // Top and left first, then bottom and right, so the corners belong to the
    // shadow. That is what makes a corner read as a mitre, and it is the one
    // line of the two-pixel version that needed no rethinking.
    fb.rect(r.x, r.y, r.w, 1, tl);
    fb.rect(r.x, r.y, 1, r.h, tl);
    fb.rect(r.x, r.y + r.h - 1, r.w, 1, br);
    fb.rect(r.x + r.w - 1, r.y, 1, r.h, br);
}

/// A one-pixel outline in a single colour.
///
/// `Framebuffer::frame` draws the same thing through `put`, a pixel at a
/// time. This is four span fills, and it is on the path every window takes
/// every frame, so the difference is worth a second function rather than a
/// comment about it.
pub fn outline(fb: &Framebuffer, r: Rect, c: Color) {
    if r.w == 0 || r.h == 0 {
        return;
    }
    fb.rect(r.x, r.y, r.w, 1, c);
    fb.rect(r.x, r.y + r.h - 1, r.w, 1, c);
    fb.rect(r.x, r.y, 1, r.h, c);
    fb.rect(r.x + r.w - 1, r.y, 1, r.h, c);
}

/// A raised surface: face plus a raised edge. Buttons, panels, menu bars.
pub fn panel(fb: &Framebuffer, r: Rect) {
    fb.rect(r.x, r.y, r.w, r.h, FACE);
    bevel(fb, r, true);
}

/// A sunken surface. Text fields, list boxes, anything content sits *in*.
///
/// One border colour rather than a lit pair. A well under Luna is not a
/// surface pushed in, it is a hole with an edge, and lighting it from the top
/// left would be saying something about it that is not true.
pub fn well(fb: &Framebuffer, r: Rect, fill: Color) {
    fb.rect(r.x, r.y, r.w, r.h, fill);
    outline(fb, r, WELL_EDGE);
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

/// How wide a particular string draws.
///
/// Every glyph in this font is one cell wide, so width is a character count.
/// It is not a *byte* count, and the two stopped being the same thing the
/// moment the console learned to decode UTF-8: a label with three accents in
/// it measured three cells too wide, which centres it off-centre and lets a
/// menu clip a name it had room for. Call this for anything a person or the
/// model supplies; `text_w` still takes a count, for the callers that are
/// asking about columns rather than about a string.
pub fn text_w_of(s: &str) -> u32 {
    text_w(s.chars().count())
}

/// The first `n` characters, for a label that has to fit.
///
/// `&s[..n]` is the obvious spelling and it is a panic: `n` is a column count
/// and the slice wants a byte offset, and the two stopped agreeing when the
/// console learned to decode UTF-8. Truncating a path is a cosmetic
/// operation, so it must not be able to stop the machine.
pub fn head_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// The last `n` characters, for showing the end of a long path.
pub fn tail_chars(s: &str, n: usize) -> &str {
    let len = s.chars().count();
    if len <= n {
        return s;
    }
    match s.char_indices().nth(len - n) {
        Some((i, _)) => &s[i..],
        None => s,
    }
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
pub fn window(
    fb: &Framebuffer,
    r: Rect,
    title: &str,
    active: bool,
    maximised: bool,
    hot_caption: Option<usize>,
) -> Rect {
    // Outer frame: a raised slab, then a groove, which is how 3.1 gets a border
    // thick enough to grab without looking like a picture frame.
    panel(fb, r);
    let inner = r.shrink(FACE_INSET);
    if inner.is_empty() {
        return inner;
    }

    let bar = Rect::new(inner.x, inner.y, inner.w, TITLE_H);
    title_bar(fb, bar, title, active, maximised, hot_caption);

    let client = Rect::new(
        inner.x,
        inner.y + TITLE_H + 2,
        inner.w,
        inner.h.saturating_sub(TITLE_H + 2),
    );
    client
}

const FACE_INSET: u32 = FRAME;

/// The three caption buttons in a title bar, left to right:
/// minimise, maximise/restore, close.
///
/// One function for the geometry, used by the paint pass and the pointer's
/// hit-test alike -- the same rule as every other control here, because a
/// button that highlights in one place and presses in another is the bug
/// that duplicated layout always becomes.
pub fn caption_buttons(bar: Rect) -> [Rect; 3] {
    let s = bar.h.saturating_sub(8);
    let y = bar.y + 4;
    let close_x = bar.x + bar.w.saturating_sub(s + 6);
    // The close button sits apart from the pair, as it has since 95 -- the
    // one you reach for blind should not share an edge with the one that
    // merely tidies.
    let max_x = close_x.saturating_sub(s + 6);
    let min_x = max_x.saturating_sub(s + 2);
    [
        Rect::new(min_x, y, s, s),
        Rect::new(max_x, y, s, s),
        Rect::new(close_x, y, s, s),
    ]
}

pub fn title_bar(
    fb: &Framebuffer,
    r: Rect,
    title: &str,
    active: bool,
    maximised: bool,
    hot_caption: Option<usize>,
) {
    // The 98 half of the ancestry: a smooth left-to-right ramp, deep to
    // bright, in Aperture's colours instead of Redmond's blues. Inactive bars
    // ramp in greys, which is what makes the focused window findable at a
    // glance on a desktop of several.
    let (from, to) = if active { (APERTURE_DEEP, APERTURE) } else { (Color::new(0x50, 0x50, 0x50), TITLE_IDLE) };
    let w = r.w.max(1);
    // The ramp is horizontal, so every row of it is the same row. Building it
    // once and blitting it down costs `h` memcpys; the column-major form this
    // replaces cost `w * h` single-pixel spans -- 30,000 of them for a
    // 1250-pixel title bar, per window, per frame, in the worst possible order
    // for a row-major buffer.
    let mut row: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(w as usize);
    for i in 0..w {
        let lerp = |a: u8, b: u8| (a as u32 + (b as u32).abs_diff(a as u32) * i / w) as u8;
        let c = Color::new(
            if to.r >= from.r { lerp(from.r, to.r) } else { (from.r as u32 - (from.r - to.r) as u32 * i / w) as u8 },
            if to.g >= from.g { lerp(from.g, to.g) } else { (from.g as u32 - (from.g - to.g) as u32 * i / w) as u8 },
            if to.b >= from.b { lerp(from.b, to.b) } else { (from.b as u32 - (from.b - to.b) as u32 * i / w) as u8 },
        );
        row.push(fb.encode(c));
    }
    let bottom = r.y.saturating_add(r.h);
    for y in r.y..bottom {
        fb.blit_span(r.x, y, &row);
    }

    let pad = 6;
    let cy = r.y + r.h / 2;
    mark(fb, (r.x + pad + 8) as i32, cy as i32, (r.h as i32 / 2) - 4, TITLE_TEXT);

    let tx = r.x + pad + 22;
    let ty = r.y + (r.h - font::GLYPH_H * CHROME_SCALE) / 2;
    let btns = caption_buttons(r);
    // Clip by characters rather than pixels, and stop short of the buttons:
    // a title that runs under the close box reads as a title bar with no
    // close box.
    let text_end = btns[0].x.saturating_sub(6);
    let room = (text_end.saturating_sub(tx) / (font::GLYPH_W * CHROME_SCALE)) as usize;
    let shown = head_chars(title, room);
    // Over a gradient there is no one background colour, so the glyphs carry
    // their own shadow instead of a box.
    text_over(fb, tx + 1, ty + 1, shown, DARKEDGE);
    text_over(fb, tx, ty, shown, TITLE_TEXT);

    // The caption buttons: raised 3.1 faces on the 98 gradient, glyphs drawn
    // by hand because the font has no box-drawing worth the name. Hover gets
    // the same inner ring every other button here uses for "the next press
    // lands here".
    for (i, b) in btns.iter().enumerate() {
        let hot = hot_caption == Some(i);
        let danger = hot && i == 2;
        fb.rect(b.x, b.y, b.w, b.h, if danger { CLOSE_HOT } else { FACE });
        bevel(fb, *b, true);
        if hot && !danger {
            let f = b.shrink(2);
            if !f.is_empty() {
                fb.frame(f.x, f.y, f.w, f.h, DARKEDGE);
            }
        }
        let ink = if danger { HILIGHT } else { TEXT };
        let g = b.shrink(4);
        if g.is_empty() {
            continue;
        }
        match i {
            // Minimise: the bar along the bottom.
            0 => fb.rect(g.x, g.y + g.h.saturating_sub(3), g.w, 3, ink),
            // Maximise, or restore when already maximised: one frame with a
            // thick lid, or two overlapping ones.
            1 => {
                if maximised {
                    let s = g.w.saturating_sub(4);
                    fb.frame(g.x + 4, g.y, s, s, ink);
                    fb.rect(g.x + 4, g.y, s, 2, ink);
                    fb.rect(g.x, g.y + 4, s, s, FACE);
                    fb.frame(g.x, g.y + 4, s, s, ink);
                    fb.rect(g.x, g.y + 4, s, 2, ink);
                } else {
                    fb.frame(g.x, g.y, g.w, g.h, ink);
                    fb.rect(g.x, g.y, g.w, 2, ink);
                }
            }
            // Close: the cross, two pixels wide so it reads at this size.
            _ => {
                let (x0, y0) = (g.x as i32, g.y as i32);
                let (x1, y1) = ((g.x + g.w) as i32 - 1, (g.y + g.h) as i32 - 1);
                fb.line(x0, y0, x1, y1, ink);
                fb.line(x0 + 1, y0, x1, y1 - 1, ink);
                fb.line(x0, y1, x1, y0, ink);
                fb.line(x0 + 1, y1, x1, y0 + 1, ink);
            }
        }
    }
}

/// Text with no background: only the glyph pixels are painted.
///
/// For surfaces that are not one colour -- the title gradient, the wallpaper
/// under an icon label. `text` fills the cell behind each glyph, which on a
/// gradient stamps a rectangle of the wrong colour around every letter.
pub fn text_over(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color) {
    let mut cx = x;
    for b in s.chars() {
        let rows = font::rows(font::index_of(b));
        for (gy, bits) in rows.iter().enumerate() {
            for gx in 0..font::GLYPH_W {
                if bits & (0x80 >> gx) != 0 {
                    for dy in 0..CHROME_SCALE {
                        for dx in 0..CHROME_SCALE {
                            fb.put(
                                cx + gx * CHROME_SCALE + dx,
                                y + gy as u32 * CHROME_SCALE + dy,
                                fb.raw(fg),
                            );
                        }
                    }
                }
            }
        }
        cx += font::GLYPH_W * CHROME_SCALE;
    }
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

    let tw = text_w_of(label);
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
        (TEXT, LIST_SEL_IDLE)
    } else {
        (TEXT, FACE)
    };
    fb.rect(r.x, r.y, r.w, r.h, bg);
    let ty = r.y + (r.h.saturating_sub(font::GLYPH_H * CHROME_SCALE)) / 2;
    let room = (r.w / (font::GLYPH_W * CHROME_SCALE)).saturating_sub(1) as usize;
    let shown = head_chars(label, room);
    text(fb, r.x + 6, ty, shown, fg, bg);
}

/// A vertical groove, for dividing a bar into sections.
pub fn separator_v(fb: &Framebuffer, x: u32, y: u32, h: u32) {
    fb.rect(x, y, 1, h, EDGE);
    fb.rect(x + 1, y, 1, h, EDGE_LIGHT);
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
    fb.rect(x, y, w, 1, EDGE);
    fb.rect(x, y + 1, w, 1, EDGE_LIGHT);
}
