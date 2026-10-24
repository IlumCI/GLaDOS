//! Frutiger Aero chrome, in Aperture colours.
//!
//! One place that owns what the system looks like, so a widget is a few calls
//! rather than a pile of rectangles, and so changing the look is changing this
//! file rather than every caller. That sentence is also the reason there is no
//! theme struct here and should not be: a scheme selectable at runtime would
//! put an indirection in front of every chrome draw to serve a switch nobody
//! has asked for twice.
//!
//! ### Warm for the machine, cool for the environment
//!
//! Frutiger Aero is aqua, glass and saturation, and this machine's colour is
//! orange. Rather than pick one, the rule is where each belongs: **the
//! surfaces the machine speaks through are warm and the room it sits in is
//! cool.** Captions, the selection bar, links and focus are Aperture amber,
//! because those are the machine addressing an operator. The wallpaper, the
//! taskbar glass and the fields are Aero's blue-green, because those are the
//! space the windows are in.
//!
//! The wall is where they meet, and it meets honestly: a horizon, deep teal
//! at the top falling through cyan to warm gold at the foot. A sunset over
//! water is one of the style's own subjects, so this holds both colours in
//! one surface rather than averaging them into mud.
//!
//! The Start button stays green. It was green in Luna, it was a green orb in
//! Aero, and it is the one control here that has never followed an accent.
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
use super::text;
use alloc::string::String;
use super::{Color, Framebuffer};

// --- surfaces ------------------------------------------------------------

/// The face of every raised control. XP's ButtonFace, and warm, which is the
/// same family the orange is in.
pub const FACE: Color = Color::new(0xEF, 0xF3, 0xF5);
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
pub const EDGE: Color = Color::new(0xA3, 0xB4, 0xBC);
/// The soft light half. Only the separators, where it sits against a line and
/// not against the face.
pub const EDGE_LIGHT: Color = Color::new(0xF6, 0xFA, 0xFC);
/// The border round a field. One colour all the way round rather than a
/// light and dark pair: a well is a hole in the surface, and XP says so by
/// outlining it in a desaturated accent instead of by lighting it.
pub const WELL_EDGE: Color = Color::new(0x6E, 0x9E, 0xB4);
/// A selected row in a list that does not have focus. Was an anonymous
/// 0xA8A8A8 written out three times in two files.
pub const LIST_SEL_IDLE: Color = Color::new(0xD3, 0xE2, 0xE9);
/// The desktop behind everything.
/// A representative mid-tone of the sky, for the few pictograms that fill a
/// panel of their own against it. The wall itself is `gfx::wall`.
pub const DESKTOP: Color = Color::new(0x0E, 0x4A, 0x66);
/// The Aperture mark on the wall. A muted orange rather than the title bar's:
/// at a fifth of the screen it would otherwise be the brightest thing on the
/// desktop, and a wallpaper that outshines the windows is one nobody can work
/// in front of.
/// The mark on the wall, lit: a sun over the horizon.
///
/// The mark there is a disc at the centre of a sky that already has water at
/// the top and gold at the foot, which is a composition with a sun-shaped hole
/// in it. Filling that hole with the accent rather than with another blue is
/// also the one place the warm half of this scheme gets to be the largest
/// thing on the screen.
pub const SUN: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xF4, 0xC8)),
    (40, Color::new(0xFF, 0xDD, 0x74)),
    (132, Color::new(0xFF, 0xB1, 0x3A)),
    (200, Color::new(0xF2, 0x8A, 0x1E)),
    (255, Color::new(0xD9, 0x64, 0x10)),
];

pub const TEXT: Color = Color::new(0x00, 0x00, 0x00);
pub const TEXT_DIM: Color = Color::new(0x76, 0x88, 0x90);

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
/// Caption text on an unfocused bar. Near-white rather than grey: Luna dims
/// the *bar* and leaves its title readable, so "which window has the
/// keyboard" is answered by the surface behind the words instead of by the
/// words going faint.
pub const TITLE_IDLE: Color = Color::new(0x2E, 0x44, 0x50);

// --- ramps ---------------------------------------------------------------
//
// Positions are 0..=255 across whatever height the ramp is asked for, so one
// table serves a 30-pixel caption and a 21-pixel caption button. Two stops one
// apart is a hard step: the break across the middle of a Luna surface is a
// step and not a fade, and every one of these tables uses it.

/// The focused caption: a gloss over the top third, the step, then a lift at
/// the foot so the bar does not read as ending in mud.
pub const TITLE_ON: [(u8, Color); 7] = [
    (0, Color::new(0xFF, 0xDE, 0xAC)),
    (8, Color::new(0xFF, 0xC5, 0x76)),
    (120, Color::new(0xF5, 0x9C, 0x2B)),
    (128, Color::new(0xE2, 0x81, 0x14)),
    (129, Color::new(0xB0, 0x5D, 0x08)),
    (246, Color::new(0xC9, 0x74, 0x14)),
    (255, Color::new(0xF4, 0xA5, 0x36)),
];
/// The same shape in cool greys. Aero dims an unfocused caption toward the
/// room rather than toward black, which is why these are blue-grey and not
/// neutral.
pub const TITLE_OFF: [(u8, Color); 6] = [
    (0, Color::new(0xE8, 0xEF, 0xF3)),
    (8, Color::new(0xD2, 0xDF, 0xE6)),
    (128, Color::new(0xB4, 0xC6, 0xD0)),
    (129, Color::new(0x94, 0xAB, 0xB8)),
    (246, Color::new(0xA6, 0xBB, 0xC6)),
    (255, Color::new(0xC8, 0xD8, 0xE0)),
];

/// Close. Red whether or not the pointer is on it.
pub const CAP_CLOSE: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xA8, 0xA8)),
    (30, Color::new(0xE8, 0x5A, 0x5A)),
    (128, Color::new(0xD2, 0x3A, 0x3A)),
    (129, Color::new(0xBE, 0x2E, 0x2E)),
    (255, Color::new(0xA8, 0x24, 0x24)),
];
/// The pale outline round a caption button, and the catchlight along the top
/// row of the bar itself.
pub const CAP_EDGE: Color = Color::new(0xFF, 0xE0, 0xBC);
/// The glyph on a caption button. White, because these sit on colour now.
pub const CAP_INK: Color = Color::new(0xFF, 0xFF, 0xFF);

/// A push button. The ramp is there to catch the top edge rather than to be
/// looked at, which is why it is three stops and nearly flat by the foot.
pub const BTN: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xFF, 0xFF)),
    (6, Color::new(0xF4, 0xFC, 0xFF)),
    (128, Color::new(0xDF, 0xF1, 0xF9)),
    (129, Color::new(0xBF, 0xE1, 0xEF)),
    (255, Color::new(0xEA, 0xF7, 0xFC)),
];
pub const BTN_DOWN: [(u8, Color); 5] = [
    (0, Color::new(0xB4, 0xCE, 0xDA)),
    (6, Color::new(0xC6, 0xDE, 0xE8)),
    (128, Color::new(0xD6, 0xEA, 0xF2)),
    (129, Color::new(0xE4, 0xF3, 0xF9)),
    (255, Color::new(0xF2, 0xFA, 0xFD)),
];
/// The one-pixel outline XP puts round every button, and its lit form.
pub const BTN_EDGE: Color = Color::new(0x6E, 0x9E, 0xB4);
pub const BTN_EDGE_HOT: Color = APERTURE;



/// The terminal's status strip, which is the tray's colour given a ramp.
///
/// Dark rather than the pale window face, because it is read against the
/// console above it and a light strip there reads as a second window rather
/// than as the foot of this one.
pub const STATUS_BAR: [(u8, Color); 3] = [
    (0, Color::new(0x0D, 0x39, 0x4E)),
    (128, Color::new(0x07, 0x2A, 0x3C)),
    (255, Color::new(0x05, 0x1F, 0x2D)),
];

/// The green of an OK button: XP's Start green, kept for the one control that
/// says yes since the Start button became an orb.
pub const START: [(u8, Color); 7] = [
    (0, Color::new(0xD6, 0xF0, 0xA8)),
    (10, Color::new(0x9A, 0xDB, 0x62)),
    (120, Color::new(0x62, 0xB0, 0x33)),
    (128, Color::new(0x4E, 0x9A, 0x24)),
    (129, Color::new(0x2F, 0x70, 0x14)),
    (240, Color::new(0x3C, 0x84, 0x18)),
    (255, Color::new(0x6F, 0xBE, 0x3A)),
];
pub const START_EDGE: Color = Color::new(0x8E, 0xD4, 0x6A);

/// A popup menu. White with a face-coloured gutter down its left and one
/// border, which is what every Windows menu has been since XP.
pub const MENU_BG: Color = Color::new(0xFF, 0xFF, 0xFF);
pub const MENU_GUTTER: Color = Color::new(0xE2, 0xEE, 0xF4);
pub const MENU_EDGE: Color = Color::new(0x8F, 0xB2, 0xC2);
/// How wide the gutter is. Nothing is drawn in it yet -- XP puts an icon or a
/// tick there -- and it is here because a menu without one reads as a list
/// box, which is a different control that does a different thing.
pub fn menu_gutter_w() -> u32 {
    px(20)
}

/// How far a window's top corners are rounded.
///
/// Six pixels against a thirty-pixel caption is XP's proportion at 96 dpi. The
/// bottom corners stay square: the bottom edge is something you drag, and a
/// rounded one reads as a control that has come loose from the frame.
pub fn win_round() -> u32 {
    px(ROUND.load(core::sync::atomic::Ordering::Relaxed))
}

/// How round a window's head is, logical (Settings): 8 by default, 0 square.
static ROUND: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(8);

pub fn set_round(logical: u32) {
    ROUND.store(logical.min(16), core::sync::atomic::Ordering::Relaxed);
}

pub fn round_logical() -> u32 {
    ROUND.load(core::sync::atomic::Ordering::Relaxed)
}

// A window's shadow is `gfx::shadow` now, set in `desk::win_shadow`. It was a
// hard five-pixel band at 84%, under a comment that softness "needs the
// backdrop at the moment the corner is drawn and the back-to-front repaint has
// already covered it" -- true of a renderer that painted straight to the
// screen, and not of one that composes into a back buffer it can read.

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

/// The interface's scale, in thousandths: 1250 is 125%, Windows' choice for
/// this panel's 141 pixels to the inch, and the size every metric in this
/// file was drawn at -- so `px(n)` is `n` at the default, and the look moves
/// only when somebody moves the scale. Every metric goes through it, which
/// is the whole of "logical pixels": a number written here is a size at
/// 125%, and the screen gets that size at whatever the scale is.
static SCALE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1250);

/// The scale, in thousandths (1250 is 125%).
pub fn scale() -> u32 {
    SCALE.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set it, between 75% and 250%. Everything sized from it follows on the
/// next frame; the caller asks for that frame.
pub fn set_scale(permille: u32) {
    SCALE.store(permille.clamp(750, 2500), core::sync::atomic::Ordering::Relaxed);
}

/// A size drawn at 125%, at the scale in force.
pub fn px(n: u32) -> u32 {
    ((n as u64 * scale() as u64 + 625) / 1250) as u32
}

/// The interface face's size at the scale in force.
pub fn ui_size() -> f32 {
    UI_PX * scale() as f32 / 1250.0
}
/// Height of a title bar at `CHROME_SCALE`, with room above and below the text.
///
/// Fourteen rather than eight above a 16-pixel line. A caption is the surface
/// somebody looks at to know which window has the keyboard, and the 3.1
/// proportion crowds it: XP put air around the same text and that air is most
/// of why one reads as modern and the other does not.
pub fn title_h() -> u32 {
    px(font::GLYPH_H * CHROME_SCALE + 14)
}
/// Height of a menu bar and of one row in a dropdown.
///
/// **Its own constant, and that is the point of it.** It was
/// `= theme::TITLE_H` over in `desk`, which is a coincidence and not a fact:
/// the moment the caption grew, every menu row and every dropdown row would
/// have grown with it, silently, to thirty pixels around sixteen of text.
pub fn menu_h() -> u32 {
    px(font::GLYPH_H * CHROME_SCALE + 6)
}
/// Thickness of a window's outer frame.
///
/// Three, and it is a coloured border now rather than a grey slab. XP's frame
/// says which window is active along its whole length; 3.1's said only that
/// there was an edge here to grab.
pub fn frame_w() -> u32 {
    px(3)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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

/// A control face: a vertical ramp inside a one-pixel outline, with the four
/// corner pixels left unwritten.
///
/// The shape every XP control is -- push button, caption button, task button,
/// Start. The rounding is those four omitted pixels and nothing else, which is
/// enough at this size and costs nothing: what shows through them is whatever
/// the parent painted, the same argument `Shape` makes about a window, applied
/// where it happens to be free.
///
/// Deliberately not a `Shape`. A row table and a clip on every span inside it
/// is the right trade for a window and an absurd one for a twenty-one pixel
/// box.
pub fn control(fb: &Framebuffer, r: Rect, stops: &[(u8, Color)], edge: Color) {
    if r.w < 3 || r.h < 3 {
        return;
    }
    fb.vgrad(r.x + 1, r.y + 1, r.w - 2, r.h - 2, stops);
    fb.rect(r.x + 1, r.y, r.w - 2, 1, edge);
    fb.rect(r.x + 1, r.y + r.h - 1, r.w - 2, 1, edge);
    fb.rect(r.x, r.y + 1, 1, r.h - 2, edge);
    fb.rect(r.x + r.w - 1, r.y + 1, 1, r.h - 2, edge);
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

// Interface text is Noto Sans and editable text is Cascadia Code, through
// `gfx::text` (Q4c). The *layout* is still the bitmap's for now: `text_h` is
// the 16-pixel row every window was laid out against, and `text_w(n)` is
// still 16 pixels a character -- which stays a safe budget, since no glyph of
// Noto Sans at 15 px is wider than 16, so a label sized by it can only come
// out shorter than its room, never overlap what is beside it. The metrics
// move to logical pixels with the restyle (Q8); this commit is the faces.
//
// Editable text is monospaced on purpose: a field and the Write window place
// the caret at a *column*, and a column is a pixel offset only in a face
// whose every advance is the same. `mono_w` is that advance.

/// The interface face's size at `CHROME_SCALE`, and what a scale maps to.
pub const UI_PX: f32 = 15.0;

fn ui_px(scale: u32) -> f32 {
    match scale {
        0 | 1 => 10.5,
        2 => ui_size(),
        s => ui_size() * s as f32 / 2.0,
    }
}

/// Set `s` in `face` with the top of its row at `y`, the cap height centred in
/// a row of `row_h` pixels -- the vertical rule a 16-pixel bitmap cell used to
/// supply for free. Fills `bg` behind the measured run when there is one.
fn set(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color, bg: Option<Color>, face: text::Face, px: f32, row_h: u32) {
    if let Some(bg) = bg {
        let w = (text::measure(face, px, s) + 0.999) as u32;
        fb.rect(x, y, w, row_h, bg);
    }
    // Cap height is about 0.71 em in both faces; centre that, and let
    // descenders hang into the row's spare pixels below.
    let cap = px * 0.714;
    let base = y as f32 + (row_h as f32 + cap) / 2.0;
    text::draw(fb, x as f32, (base + 0.5) as i32, face, px, s, fg);
}

pub fn text(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color, bg: Color) {
    set(fb, x, y, s, fg, Some(bg), text::Face::Ui, ui_size(), text_h());
}

/// Interface text in the semibold weight: captions and headings.
pub fn text_strong(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color) {
    set(fb, x, y, s, fg, None, text::Face::UiStrong, ui_size(), text_h());
}

/// Editable or column-aligned text, monospaced, on a `bg` field.
pub fn mono_text(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color, bg: Color) {
    set(fb, x, y, s, fg, Some(bg), text::Face::Mono, ui_size(), text_h());
}

/// The width of `n` columns of `mono_text`.
pub fn mono_w(n: usize) -> u32 {
    let adv = text::measure(text::Face::Mono, ui_size(), "M");
    (n as f32 * adv + 0.5) as u32
}

/// The width of one column of `mono_text`, for hit-testing a press to a column.
pub fn mono_col() -> u32 {
    mono_w(1).max(1)
}

/// The measured width of `s` as `text_over_at` sets it at `scale`.
pub fn text_w_of_at(s: &str, scale: u32) -> u32 {
    (text::measure(text::Face::Ui, ui_px(scale), s) + 0.999) as u32
}

/// `fit` at a chosen scale.
pub fn fit_at(s: &str, max: u32, scale: u32) -> String {
    text::ellipsize(text::Face::Ui, ui_px(scale), s, max as f32)
}

/// `s` cut to fit `max` pixels of interface text, with an ellipsis if it was.
pub fn fit(s: &str, max: u32) -> String {
    text::ellipsize(text::Face::Ui, ui_size(), s, max as f32)
}

/// Height of one line of chrome text, the companion to `text_w`.
pub fn text_h() -> u32 {
    px(font::GLYPH_H * CHROME_SCALE)
}

pub fn text_w(len: usize) -> u32 {
    px(len as u32 * font::GLYPH_W * CHROME_SCALE)
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
    // Measured now, kerning and all: the face is proportional, so a count of
    // anything -- bytes or characters -- is not a width.
    (text::measure(text::Face::Ui, ui_size(), s) + 0.999) as u32
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

/// Where every part of a window is.
///
/// One formula, and it lives here rather than in `desk` because
/// `theme::window` paints from it and `desk::press_at` hit-tests from it, and
/// those two are exactly the pair that must never disagree -- a title bar you
/// can see and cannot grab is the shape that bug takes.
///
/// It was two copies of the same arithmetic in two files, held in agreement by
/// hand. That held right up until `TITLE_H` moved, which is the moment it
/// would have stopped holding silently. `caption_buttons` has been arranged
/// this way since it existed; this is the rest of it.
#[derive(Clone, Copy)]
pub struct Chrome {
    pub frame: Rect,
    pub title: Rect,
    /// Everything below the caption, before any menu bar is taken out of it.
    /// The ground the body is painted on.
    pub body: Rect,
    pub menubar: Option<Rect>,
    pub client: Rect,
}

/// Whether a frame is the whole screen, which is what a full-screen window's
/// frame is and nothing else's: those have no border and no caption. Decided
/// by the rectangle so every caller of `chrome` -- painting, hit-testing,
/// input -- agrees without being told a window's state.
pub fn is_bare(frame: Rect) -> bool {
    frame.x == 0
        && frame.y == 0
        && super::primary().is_some_and(|fb| frame.w == fb.width() && frame.h == fb.height())
}

pub fn chrome(frame: Rect, has_menus: bool) -> Chrome {
    if is_bare(frame) {
        let title = Rect::new(frame.x, frame.y, frame.w, 0);
        return Chrome { frame, title, body: frame, menubar: None, client: frame };
    }
    let inner = frame.shrink(frame_w());
    let title = Rect::new(inner.x, inner.y, inner.w, title_h());
    // Two pixels of border show between the caption and the body, which under
    // Luna is a thin accent line rather than the gap it used to be.
    let body = Rect::new(
        inner.x,
        inner.y + title_h() + 2,
        inner.w,
        inner.h.saturating_sub(title_h() + 2),
    );
    let (menubar, client) = if has_menus {
        (
            Some(Rect::new(body.x, body.y, body.w, menu_h())),
            Rect::new(body.x, body.y + menu_h(), body.w, body.h.saturating_sub(menu_h())),
        )
    } else {
        (None, body)
    };
    Chrome { frame, title, body, menubar, client }
}

/// Draw a window: border, title bar, and the ground its body sits on.
///
/// Answers where everything is, so a caller that wants to hit-test what it
/// just painted asks nobody a second time.
pub fn window(
    fb: &Framebuffer,
    r: Rect,
    title: &str,
    active: bool,
    maximised: bool,
    has_menus: bool,
    hot_caption: Option<usize>,
) -> Chrome {
    let c = chrome(r, has_menus);
    if is_bare(r) {
        fb.rect(r.x, r.y, r.w, r.h, Color::new(0, 0, 0));
        return c;
    }
    // The frame and caption are glass (1.4.1): the blurred wallpaper behind
    // them, colourised. Four strips -- the top one taking in the caption and
    // the line under it -- rather than the whole window, because the body is
    // solid by decision and glass under it would be computed and covered.
    let g = caption_glass(active);
    let top_h = c.body.y.saturating_sub(r.y);
    super::glass::fill(fb, r.x as i32, r.y as i32, r.w, top_h, 0, g);
    super::glass::fill(fb, r.x as i32, c.body.y as i32, frame_w(), r.h.saturating_sub(top_h), 0, g);
    super::glass::fill(fb, (r.x + r.w - frame_w()) as i32, c.body.y as i32, frame_w(), r.h.saturating_sub(top_h), 0, g);
    super::glass::fill(fb, r.x as i32, (r.y + r.h - frame_w()) as i32, r.w, frame_w(), 0, g);
    if c.body.is_empty() {
        return c;
    }
    title_bar(fb, c.title, title, active, maximised, hot_caption);
    // The body's ground, whether or not a menu bar will be drawn on part of
    // it. A window with menus and one without must stand on the same face.
    fb.rect(c.body.x, c.body.y, c.body.w, c.body.h, FACE);
    c
}

/// A popup menu's geometry.
///
/// One object where there were five copies of `text_w(longest) + 24`,
/// `n * MENU_H + 8` and a four-pixel inset: `dropdown` painted from one,
/// `dropdown_rows` answered a second, `dropdown_item_at` asked a third,
/// `start_menu_rect` built a fourth with an extra row, and the Start menu
/// painted from a fifth. They agreed by hand, and this commit is what would
/// have broken the agreement -- a Luna popup gains a border and a gutter, so
/// the inset stops being four and the width stops being the labels alone.
pub struct Popup {
    pub panel: Rect,
    pub rows: usize,
    /// Inset from the panel's edge to the first row: a menu's three pixels,
    /// or the Start menu's glass frame.
    pub pad: u32,
}

/// Inset from the panel edge to the first row.
fn popup_pad() -> u32 {
    px(3)
}

impl Popup {
    /// Sized from its widest label in columns and its row count, clamped to
    /// whatever room is left on the right.
    pub fn sized(x: u32, y: u32, cols: usize, rows: usize, max_w: u32) -> Popup {
        Self::padded(x, y, cols, rows, max_w, popup_pad())
    }

    /// `sized`, with a frame of `pad` round the rows.
    pub fn padded(x: u32, y: u32, cols: usize, rows: usize, max_w: u32, pad: u32) -> Popup {
        let w = (text_w(cols) + 24 + menu_gutter_w() + 2 * (pad - popup_pad().min(pad))).min(max_w);
        let h = rows as u32 * menu_h() + pad * 2;
        Popup { panel: Rect::new(x, y, w, h), rows, pad }
    }

    /// Where row `i` is drawn: to the right of the gutter, which is why a
    /// selection bar stops at it rather than running under it.
    pub fn row(&self, i: usize) -> Rect {
        let inset = self.pad + menu_gutter_w();
        Rect::new(
            self.panel.x + inset,
            self.panel.y + self.pad + i as u32 * menu_h(),
            self.panel.w.saturating_sub(inset + self.pad),
            menu_h(),
        )
    }

    /// Which row a point is on.
    ///
    /// The whole panel width and not just the row: a click in the gutter is a
    /// click on that line, which is what every menu anybody has used does.
    pub fn item_at(&self, x: i32, y: i32) -> Option<usize> {
        let p = self.panel;
        if x < p.x as i32 || x >= (p.x + p.w) as i32 {
            return None;
        }
        if y < p.y as i32 || y >= (p.y + p.h) as i32 {
            return None;
        }
        let row = (y - (p.y + self.pad) as i32) / menu_h() as i32;
        if row >= 0 && (row as usize) < self.rows {
            Some(row as usize)
        } else {
            None
        }
    }

    /// Whether there is room to draw anything at all.
    ///
    /// A menu opened near the right edge can be clipped to nothing, and the
    /// width arithmetic on a narrow one used to wrap to four billion -- not a
    /// crash but a fill loop long enough to look like a hang.
    pub fn wide_enough(&self) -> bool {
        self.panel.w >= menu_gutter_w() + 16
    }
}

/// The Start menu's frame: Vista's, a pane of the focused caption's glass
/// round a white body, so the menu reads as part of the bar it came out of.
pub fn start_popup(fb: &Framebuffer, p: &Popup) {
    let r = p.panel;
    super::glass::fill(fb, r.x as i32, r.y as i32, r.w, r.h, 8, caption_glass(true));
    let m = p.pad.saturating_sub(2);
    let body = Rect::new(r.x + m, r.y + m, r.w.saturating_sub(2 * m), r.h.saturating_sub(2 * m));
    fb.rect(body.x, body.y, body.w, body.h, MENU_BG);
    fb.rect(body.x + 1, body.y + 1, menu_gutter_w() + 2, body.h.saturating_sub(2), MENU_GUTTER);
    outline(fb, body, MENU_EDGE);
}

/// The light edge along the top of a strip that sits on the bar.
pub const TRAY_EDGE: Color = Color::new(0x9F, 0xCF, 0xE2);

/// The taskbar's glass: Vista's black, darker than any caption, so the bar
/// reads as the floor the windows stand on.
pub const TASKBAR_GLASS: super::glass::Glass = super::glass::Glass {
    tint: Color::new(18, 24, 34),
    glow: Color::new(80, 100, 130),
    blur: 46,
    afterglow: 14,
    color: 40,
    gloss: 34,
    gloss_end: 0.5,
};

/// The ground a popup's rows sit on: white, a gutter, and one border.
pub fn popup(fb: &Framebuffer, p: &Popup) {
    let r = p.panel;
    fb.rect(r.x, r.y, r.w, r.h, MENU_BG);
    fb.rect(r.x + 1, r.y + 1, menu_gutter_w(), r.h.saturating_sub(2), MENU_GUTTER);
    outline(fb, r, MENU_EDGE);
}

/// The rectangle of each menu-bar label, in order.
///
/// One formula for the paint loop, the hit-test, and the two places a dropdown
/// is positioned under one. There were four copies of it and the fourth was in
/// a different file from the first.
pub fn menu_labels<'a>(
    bar: Rect,
    labels: impl Iterator<Item = &'a str> + 'a,
) -> impl Iterator<Item = Rect> + 'a {
    let (y, h) = (bar.y, bar.h);
    labels.scan(bar.x + 6, move |x, label| {
        let w = text_w_of(label) + 12;
        let r = Rect::new(*x, y, w, h);
        *x += w;
        Some(r)
    })
}

/// The three caption buttons in a title bar, left to right:
/// minimise, maximise/restore, close.
///
/// One function for the geometry, used by the paint pass and the pointer's
/// hit-test alike -- the same rule as every other control here, because a
/// button that highlights in one place and presses in another is the bug
/// that duplicated layout always becomes.
pub fn caption_buttons(bar: Rect) -> [Rect; 3] {
    // Vista's proportions: a short glass pair and a close button half as wide
    // again as both, so the one reached for blind is the biggest target on the
    // bar. The gap before close stays, for the argument it has always made.
    let h = bar.h.saturating_sub(10);
    let y = bar.y + (bar.h.saturating_sub(h)) / 2;
    let small = h * 7 / 5;
    let wide = h * 11 / 5;
    let close_x = bar.x + bar.w.saturating_sub(wide + 6);
    let max_x = close_x.saturating_sub(small + 4);
    let min_x = max_x.saturating_sub(small + 2);
    [
        Rect::new(min_x, y, small, h),
        Rect::new(max_x, y, small, h),
        Rect::new(close_x, y, wide, h),
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
    // The caption is glass, laid by `window` with the frame; drawn again here
    // only so `title_bar` stays callable on its own, as `mindwin` does.
    super::glass::fill(fb, r.x as i32, r.y as i32, r.w, r.h, 0, caption_glass(active));

    let pad = 6;
    let cy = r.y + r.h / 2;
    // The Aperture mark itself, where this used to be a ring round a dot --
    // which is what the mark became at sixteen pixels with no anti-aliasing,
    // and stopped needing to be once the icons could draw it.
    let n = 22u32.min(r.h.saturating_sub(4));
    let kind = if active { super::icons::MARK } else { super::icons::MARK_IDLE };
    super::icons::lay(fb, kind, (r.x + pad + 8).saturating_sub(n / 2), cy.saturating_sub(n / 2), n);

    let tx = r.x + pad + 22;
    let ty = r.y + (r.h - font::GLYPH_H * CHROME_SCALE) / 2;
    let btns = caption_buttons(r);
    // Fitted by measuring, and stopped short of the buttons: a title that
    // runs under the close box reads as a title bar with no close box. The
    // one-pixel black shadow the bitmap needed is gone -- anti-aliased text
    // reads on the ramp without it, and a hard offset copy is the most
    // 1998 thing a caption can wear.
    let text_end = btns[0].x.saturating_sub(6);
    let shown = fit(title, text_end.saturating_sub(tx));
    text_strong(fb, tx, ty, &shown, if active { TITLE_TEXT } else { TITLE_IDLE });

    // The caption buttons, Vista's: glass minimise and maximise with dark
    // glyphs, and a wider close in red with a soft glow -- red whether or not
    // the pointer is on it, since under Vista as under Luna the colour is the
    // control's identity and hover only brightens it.
    for (i, b) in btns.iter().enumerate() {
        caption_button(fb, r, *b, i, hot_caption == Some(i), maximised);
    }
}

/// What tints the caption and frame glass, chosen at run time (`win glass`)
/// until the Settings window owns it.
///
/// Sky by default, chosen over Aperture orange against the three side by
/// side: the accent still marks focus, selection and links, and a caption
/// that is the colour of the room rather than of the machine lets it.
static CAPTION_TINT: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(1);

pub const TINTS: [&str; 3] = ["warm", "sky", "graphite"];

pub fn set_caption_tint(name: &str) -> bool {
    match TINTS.iter().position(|t| *t == name) {
        Some(i) => {
            CAPTION_TINT.store(i as u8, core::sync::atomic::Ordering::Relaxed);
            true
        }
        None => false,
    }
}

pub fn caption_tint() -> &'static str {
    TINTS[CAPTION_TINT.load(core::sync::atomic::Ordering::Relaxed) as usize % TINTS.len()]
}

/// The glass a caption wears. The focused window's carries the tint; the
/// rest are a neutral glass, darker, so focus reads at a glance -- the cue a
/// title bar's colour has always been.
pub fn caption_glass(active: bool) -> super::glass::Glass {
    use super::glass::Glass;
    // The tint takes a third of the colour here where Vista's default took
    // eight per cent. Measured on the night wallpaper: at Vista's balance all
    // three tints came out the same dark grey, because a blurred dark sky is
    // what the other two terms are made of. Vista's walls were bright.
    if !active {
        // Unfocused: a pale, cool, plainly transparent glass, so the focused
        // caption is the one with colour in it -- and dark text reads on it.
        return Glass { tint: Color::new(206, 214, 226), glow: Color::new(226, 232, 242), blur: 26, afterglow: 14, color: 60, gloss: 40, gloss_end: 0.5 };
    }
    match CAPTION_TINT.load(core::sync::atomic::Ordering::Relaxed) {
        // Aperture orange: the machine's own colour on the surface it speaks
        // through. Offered, not the default.
        0 => Glass { tint: Color::new(242, 128, 32), glow: Color::new(255, 176, 96), blur: 34, afterglow: 26, color: 40, gloss: 56, gloss_end: 0.5 },
        // Vista's own "Sky", at this balance.
        1 => Glass { tint: Color::new(0x74, 0xb8, 0xfc), glow: Color::new(0x74, 0xb8, 0xfc), blur: 34, afterglow: 26, color: 40, gloss: 56, gloss_end: 0.5 },
        // Smoke.
        _ => Glass { tint: Color::new(52, 58, 72), glow: Color::new(150, 160, 180), blur: 40, afterglow: 30, color: 30, gloss: 46, gloss_end: 0.5 },
    }
}

/// One caption button, drawn per pixel from a rounded box's distance: a
/// vertical ramp, a gloss on the upper half, a pale rim, and for close a red
/// glow outside it, clipped to the caption so it never leaves the window.
fn caption_button(fb: &Framebuffer, bar: Rect, b: Rect, kind: usize, hot: bool, maximised: bool) {
    use super::blend::{opaque, over_px, premul};
    let close = kind == 2;
    let (top, bot, rim) = match (close, hot) {
        (true, false) => ([236u8, 120, 100, 255], [170u8, 30, 20, 255], [255u8, 220, 210, 170]),
        (true, true) => ([255, 150, 126, 255], [206, 44, 30, 255], [255, 236, 228, 200]),
        (false, false) => ([214, 232, 250, 205], [120, 160, 200, 205], [255, 255, 255, 160]),
        (false, true) => ([236, 246, 255, 230], [150, 190, 230, 230], [255, 255, 255, 200]),
    };
    let glow = if close { if hot { 150.0 } else { 110.0 } } else { 0.0 };
    let reach = if close { 5i32 } else { 0 };
    let clip = fb.clipped(bar.x, bar.y, bar.w, bar.h);
    let (bx, by, bw, bh) = (b.x as f32, b.y as f32, b.w as f32, b.h as f32);
    let rad = 4.0f32;
    let enc = |c: [u8; 4]| premul(fb.encode(Color::new(c[0], c[1], c[2])) & 0x00FF_FFFF, c[3]);
    let x0 = (b.x as i32 - reach).max(bar.x as i32);
    let x1 = ((b.x + b.w) as i32 + reach).min((bar.x + bar.w) as i32);
    let y0 = (b.y as i32 - reach).max(bar.y as i32);
    let y1 = ((b.y + b.h) as i32 + reach).min((bar.y + bar.h) as i32);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let mut row: alloc::vec::Vec<u32> = alloc::vec![0; (x1 - x0) as usize];
    for y in y0..y1 {
        for x in x0..x1 {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let d = super::shape::sd_rbox(px, py, bx + bw / 2.0, by + bh / 2.0, bw / 2.0, bh / 2.0, rad);
            let mut p = 0u32;
            if glow > 0.0 && d > 0.0 {
                let a = glow * crate::ai::tensor::expf(-(d * d) / 8.0);
                p = premul(fb.encode(Color::new(255, 70, 40)) & 0x00FF_FFFF, a as u8);
            }
            let cov = (0.5 - d).clamp(0.0, 1.0);
            if cov > 0.0 {
                let t = ((py - by) / bh).clamp(0.0, 1.0);
                let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t) as u8;
                let body = [lerp(top[0], bot[0]), lerp(top[1], bot[1]), lerp(top[2], bot[2]), lerp(top[3], bot[3])];
                let a = (body[3] as f32 * cov) as u8;
                p = over_px(premul(fb.encode(Color::new(body[0], body[1], body[2])) & 0x00FF_FFFF, a), p);
                if py < by + bh * 0.48 {
                    p = over_px(premul(opaque(fb.encode(Color::new(255, 255, 255))) & 0x00FF_FFFF, (60.0 * cov) as u8), p);
                }
                let edge = (0.5 - (d + 0.6).abs()).clamp(0.0, 1.0);
                if edge > 0.0 {
                    p = over_px(super::blend::scale_px(enc(rim), (edge * 255.0) as u32), p);
                }
            }
            row[(x - x0) as usize] = p;
        }
        clip.over_row(x0 as u32, y as u32, &row);
    }
    // The glyph, anti-aliased, white on red and dark on glass.
    let ink = if close { premul(fb.encode(Color::new(255, 255, 255)) & 0x00FF_FFFF, 255) }
              else { premul(fb.encode(Color::new(30, 45, 70)) & 0x00FF_FFFF, 255) };
    let (cx, cy) = (bx + bw / 2.0, by + bh / 2.0);
    let s = (bh * 0.22).max(3.0);
    let lw = (bh * 0.09).max(1.4);
    match kind {
        0 => super::shape::line(&clip, cx - s, cy + s * 0.55, cx + s, cy + s * 0.55, lw * 1.1, ink),
        1 => {
            let bxx = |x: f32, y: f32, w: f32| super::shape::Box::new(x as i32, y as i32, w as u32, w as u32);
            if maximised {
                super::shape::ring(&clip, bxx(cx - s * 0.6, cy - s * 1.0, s * 1.6), 1, lw as u32, ink);
                super::shape::ring(&clip, bxx(cx - s * 1.0, cy - s * 0.6, s * 1.6), 1, lw as u32, ink);
            } else {
                super::shape::ring(&clip, bxx(cx - s, cy - s, s * 2.0), 1, lw as u32, ink);
            }
        }
        _ => {
            super::shape::line(&clip, cx - s, cy - s, cx + s, cy + s, lw * 1.2, ink);
            super::shape::line(&clip, cx - s, cy + s, cx + s, cy - s, lw * 1.2, ink);
        }
    }
}

/// Text with no background: only the glyph pixels are painted.
///
/// For surfaces that are not one colour -- the title gradient, the wallpaper
/// under an icon label. `text` fills the cell behind each glyph, which on a
/// gradient stamps a rectangle of the wrong colour around every letter.
pub fn text_over(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color) {
    text_over_at(fb, x, y, s, fg, CHROME_SCALE);
}

/// The same, at a chosen scale.
///
/// Chrome is drawn at `CHROME_SCALE` because it is read at arm's length and a
/// title bar has room to spare. A window that reports on the machine does not:
/// it is a dense list of facts in a quarter of a screen, and at scale two a
/// pane fits nine rows. At scale one it fits nineteen, and the glyphs are a
/// crisp bitmap rather than a shrunk one, because the font *is* eight pixels
/// and two was always a doubling.
pub fn text_over_at(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color, scale: u32) {
    set(fb, x, y, s, fg, None, text::Face::Ui, ui_px(scale), text_h_at(scale));
}

/// Row height and column width at a chosen scale, the companions to
/// `text_h` and `text_w`.
pub fn text_h_at(scale: u32) -> u32 {
    font::GLYPH_H * scale.max(1)
}

pub fn text_w_at(len: usize, scale: u32) -> u32 {
    len as u32 * font::GLYPH_W * scale.max(1)
}

/// A button. `focused` draws the keyboard focus; `default` marks the one Enter
/// would press if focus were elsewhere.
pub fn button(fb: &Framebuffer, r: Rect, label: &str, focused: bool, pressed: bool) {
    button_at(fb, r, label, focused, pressed, CHROME_SCALE);
}

/// The same, at a chosen text scale, and with the label truncated to fit.
///
/// Both halves of that matter for a dense window. `button` centres its label
/// and clips nothing, which is correct where a button is laid out to suit its
/// text; in a rail a quarter of the screen wide the row is laid out first and
/// the labels have to live inside it. Two full-size labels in a 300-pixel rail
/// ran into each other and read as one word.
pub fn button_at(
    fb: &Framebuffer,
    r: Rect,
    label: &str,
    focused: bool,
    pressed: bool,
    scale: u32,
) {
    control(
        fb,
        r,
        if pressed { &BTN_DOWN } else { &BTN },
        if focused { BTN_EDGE_HOT } else { BTN_EDGE },
    );

    // 3.1 marked the default button with an extra black rectangle just inside
    // the edge. Reused here for keyboard focus, which is the thing that
    // actually needs to be visible on a machine with no pointer.
    if focused {
        let f = r.shrink(3);
        if !f.is_empty() {
            fb.frame(f.x, f.y, f.w, f.h, DARKEDGE);
        }
    }

    let label = fit_at(label, r.w.saturating_sub(6), scale);
    let label = label.as_str();
    let tw = text_w_of_at(label, scale);
    let tx = r.x + (r.w.saturating_sub(tw)) / 2 + u32::from(pressed);
    let ty = r.y + (r.h.saturating_sub(text_h_at(scale))) / 2 + u32::from(pressed);
    // `text_over` rather than `text`: the face is a ramp now, so stamping one
    // background colour behind every glyph would print a flat block across it.
    text_over_at(fb, tx, ty, label, TEXT, scale);
}

/// One row of a list box. Selected rows invert to the Aperture bar.
pub fn list_row(fb: &Framebuffer, r: Rect, label: &str, selected: bool, focused: bool) {
    list_row_at(fb, r, label, selected, focused, CHROME_SCALE);
}

/// The same, at a chosen text scale.
pub fn list_row_at(
    fb: &Framebuffer,
    r: Rect,
    label: &str,
    selected: bool,
    focused: bool,
    scale: u32,
) {
    let (fg, bg) = if selected && focused {
        (SELECT_TEXT, SELECT)
    } else if selected {
        // Selected but the list does not have focus: keep the bar, drop the
        // colour, so a form with several lists still says which one is live.
        (TEXT, LIST_SEL_IDLE)
    } else {
        (TEXT, MENU_BG)
    };
    fb.rect(r.x, r.y, r.w, r.h, bg);
    let ty = r.y + (r.h.saturating_sub(text_h_at(scale))) / 2;
    let shown = fit_at(label, r.w.saturating_sub(8), scale);
    text_over_at(fb, r.x + 4, ty, &shown, fg, scale);
}

/// A vertical groove, for dividing a bar into sections.
pub fn separator_v(fb: &Framebuffer, x: u32, y: u32, h: u32) {
    fb.rect(x, y, 1, h, EDGE);
    fb.rect(x + 1, y, 1, h, EDGE_LIGHT);
}

// The icons are `gfx::icons` since 1.4.1: lit objects drawn by signed
// distance, each carrying its own ramps. The shared icon ramps, edge, gloss
// and inks that were here served the rectangle-and-frame drawings they
// replaced.


// --- dashboard painters ---------------------------------------------------
//
// A window that reports on the machine wants shapes `ui::Panel` has no variant
// for: a filled proportion, a row of chips, a name against a value. These live
// here for the same reason `button` and `list_row` do -- the look is this
// file's business and the layout is the caller's -- and they are span-based,
// so a window full of them costs what a window full of rectangles costs.

/// A filled proportion, `frac` out of 256, in a sunken track.
///
/// The ramp is clipped to the filled part rather than being drawn across the
/// whole track and covered, so an empty gauge costs one `rect` and a full one
/// costs a `vgrad`. Neither costs a per-pixel loop.
pub fn gauge(fb: &Framebuffer, r: Rect, frac: u32, stops: &[(u8, Color)]) {
    if r.w < 3 || r.h < 3 {
        return;
    }
    fb.rect(r.x, r.y, r.w, r.h, SCREEN);
    let inner = r.shrink(1);
    if !inner.is_empty() {
        let w = inner.w * frac.min(256) / 256;
        if w > 0 {
            fb.vgrad(inner.x, inner.y, w, inner.h, stops);
        }
    }
    outline(fb, r, WELL_EDGE);
}

/// The same shape in one colour, for a bar that is a quantity rather than a
/// surface -- a memory figure, a step count.
pub fn bar(fb: &Framebuffer, r: Rect, frac: u32, fill: Color) {
    if r.w < 3 || r.h < 3 {
        return;
    }
    fb.rect(r.x, r.y, r.w, r.h, SCREEN);
    let inner = r.shrink(1);
    if !inner.is_empty() {
        let w = inner.w * frac.min(256) / 256;
        if w > 0 {
            fb.rect(inner.x, inner.y, w, inner.h, fill);
        }
    }
    outline(fb, r, WELL_EDGE);
}

/// A strip of tabs across `r`, and **where each one landed**.
///
/// Returning the rectangles is the whole point. A tab strip drawn in one place
/// and hit-tested in another is the bug this desktop forbids, and every other
/// control here that has two halves -- `caption_buttons`, `Popup::row`,
/// `menu_labels` -- is arranged the same way for the same reason.
pub fn tabs(fb: &Framebuffer, r: Rect, labels: &[&str], sel: usize) -> alloc::vec::Vec<Rect> {
    let mut out = alloc::vec::Vec::with_capacity(labels.len());
    if labels.is_empty() || r.w == 0 {
        return out;
    }
    let n = labels.len() as u32;
    for (i, label) in labels.iter().enumerate() {
        // Divided by position rather than by accumulating a width, so rounding
        // cannot leave a gap or an overlap between two tabs.
        let x0 = r.x + r.w * i as u32 / n;
        let x1 = r.x + r.w * (i as u32 + 1) / n;
        let t = Rect::new(x0, r.y, x1.saturating_sub(x0), r.h);
        let on = i == sel;
        control(
            fb,
            t,
            if on { &TITLE_ON } else { &BTN },
            if on { BTN_EDGE_HOT } else { BTN_EDGE },
        );
        let shown = fit(label, t.w.saturating_sub(8));
        let tx = t.x + (t.w.saturating_sub(text_w_of(&shown))) / 2;
        let ty = t.y + (t.h.saturating_sub(text_h())) / 2;
        text_over(fb, tx, ty, &shown, if on { TITLE_TEXT } else { TEXT });
        out.push(t);
    }
    out
}

/// A small chip that is either good or not. J1 through J4, pass or fail.
pub fn pill(fb: &Framebuffer, r: Rect, label: &str, ok: bool) {
    if r.w < 3 || r.h < 3 {
        return;
    }
    control(
        fb,
        r,
        if ok { &START } else { &CAP_CLOSE },
        if ok { START_EDGE } else { CAP_EDGE },
    );
    let shown = fit(label, r.w.saturating_sub(8));
    let tx = r.x + (r.w.saturating_sub(text_w_of(&shown))) / 2;
    let ty = r.y + (r.h.saturating_sub(text_h())) / 2;
    text_over(fb, tx, ty, &shown, CAP_INK);
}

/// A name against a value, on one line.
///
/// `ui::Status` is this widget and cannot be used here, because a `Panel` is a
/// vertical stack rebuilt after a command and these windows are neither. Same
/// fourteen-column gutter, so a dashboard row and a settings row line up.
pub fn kv(fb: &Framebuffer, r: Rect, name: &str, value: &str, tone: Color) {
    if r.h < text_h() {
        return;
    }
    let gutter = text_w(14);
    text_over(fb, r.x, r.y, &fit(name, r.w.min(gutter).saturating_sub(8)), TEXT_DIM);
    if r.w > gutter {
        text_over(fb, r.x + gutter, r.y, &fit(value, r.w - gutter), tone);
    }
}

/// One turn of a conversation.
///
/// The machine's turns and the operator's are told apart by ground and by
/// which edge they sit against, which is how every chat since has done it and
/// costs nothing here: two fills and an outline.
pub fn bubble(fb: &Framebuffer, r: Rect, lines: &[alloc::string::String], from_machine: bool) {
    if r.w < 8 || r.h < 8 {
        return;
    }
    if from_machine {
        control(fb, r, &TITLE_ON, CAP_EDGE);
    } else {
        control(fb, r, &BTN, BTN_EDGE);
    }
    let ink = if from_machine { TITLE_TEXT } else { TEXT };
    let inner = r.shrink(4);
    let room = (inner.w / text_w(1)) as usize;
    for (i, line) in lines.iter().enumerate() {
        let y = inner.y + i as u32 * text_h();
        if y + text_h() > inner.y + inner.h {
            break;
        }
        text_over(fb, inner.x, y, head_chars(line, room), ink);
    }
}

/// How many lines `bubble` will show `text` as, wrapped to `cols`.
///
/// Wrapped by the caller and not by the painter, because a bubble's height has
/// to be known before it is placed -- the same order `ui::note` works in and
/// for the same reason.
pub fn wrap(text: &str, cols: usize) -> alloc::vec::Vec<alloc::string::String> {
    let mut out = alloc::vec::Vec::new();
    if cols == 0 {
        return out;
    }
    let mut line = alloc::string::String::new();
    for word in text.split_whitespace() {
        let need = if line.is_empty() { word.chars().count() } else { line.chars().count() + 1 + word.chars().count() };
        if need > cols && !line.is_empty() {
            out.push(core::mem::take(&mut line));
        }
        if word.chars().count() > cols {
            // A word longer than the line is cut rather than allowed to push
            // everything after it off the edge.
            let mut rest = word;
            while rest.chars().count() > cols {
                let head = head_chars(rest, cols);
                out.push(alloc::string::String::from(head));
                rest = &rest[head.len()..];
            }
            line.push_str(rest);
            continue;
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// A horizontal rule, drawn as a groove. The 3.1 separator.
pub fn separator(fb: &Framebuffer, x: u32, y: u32, w: u32) {
    fb.rect(x, y, w, 1, EDGE);
    fb.rect(x, y + 1, w, 1, EDGE_LIGHT);
}
