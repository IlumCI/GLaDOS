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
/// Caption text on an unfocused bar. Near-white rather than grey: Luna dims
/// the *bar* and leaves its title readable, so "which window has the
/// keyboard" is answered by the surface behind the words instead of by the
/// words going faint.
pub const TITLE_IDLE: Color = Color::new(0xE4, 0xE2, 0xDA);

// --- ramps ---------------------------------------------------------------
//
// Positions are 0..=255 across whatever height the ramp is asked for, so one
// table serves a 30-pixel caption and a 21-pixel caption button. Two stops one
// apart is a hard step: the break across the middle of a Luna surface is a
// step and not a fade, and every one of these tables uses it.

/// The focused caption: a gloss over the top third, the step, then a lift at
/// the foot so the bar does not read as ending in mud.
pub const TITLE_ON: [(u8, Color); 7] = [
    (0, Color::new(0xC9, 0x72, 0x14)),
    (18, Color::new(0xFF, 0xC0, 0x78)),
    (70, Color::new(0xF5, 0x9A, 0x2E)),
    (128, Color::new(0xEE, 0x8A, 0x18)),
    (129, Color::new(0xD3, 0x77, 0x10)),
    (225, Color::new(0xB0, 0x60, 0x0D)),
    (255, Color::new(0xC4, 0x6C, 0x11)),
];
/// The same shape in greys. Shape rather than brightness, so an unfocused bar
/// still looks like a bar rather than like a flat strip.
pub const TITLE_OFF: [(u8, Color); 5] = [
    (0, Color::new(0x9A, 0x9A, 0x96)),
    (18, Color::new(0xD8, 0xD8, 0xD2)),
    (128, Color::new(0xB4, 0xB4, 0xAE)),
    (129, Color::new(0xA2, 0xA2, 0x9C)),
    (255, Color::new(0x8C, 0x8C, 0x86)),
];
/// The window border, which is the caption's colour carried round the outside.
pub const BORDER_ON: Color = Color::new(0xB4, 0x63, 0x0E);
pub const BORDER_OFF: Color = Color::new(0x9A, 0x9A, 0x94);

/// Minimise and maximise.
pub const CAP_BTN: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xD9, 0xA6)),
    (30, Color::new(0xF6, 0xA9, 0x3E)),
    (128, Color::new(0xE6, 0x8D, 0x1C)),
    (129, Color::new(0xCE, 0x7A, 0x14)),
    (255, Color::new(0xB4, 0x65, 0x0F)),
];
pub const CAP_BTN_HOT: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xF0, 0xD8)),
    (30, Color::new(0xFF, 0xC8, 0x78)),
    (128, Color::new(0xF6, 0xA9, 0x3E)),
    (129, Color::new(0xE6, 0x8D, 0x1C)),
    (255, Color::new(0xCE, 0x7A, 0x14)),
];
/// Close. Red whether or not the pointer is on it -- see `CLOSE_HOT`.
pub const CAP_CLOSE: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xA8, 0xA8)),
    (30, Color::new(0xE8, 0x5A, 0x5A)),
    (128, Color::new(0xD2, 0x3A, 0x3A)),
    (129, Color::new(0xBE, 0x2E, 0x2E)),
    (255, Color::new(0xA8, 0x24, 0x24)),
];
/// Its hovered form, built round `CLOSE_HOT` so the two cannot drift apart.
pub const CAP_CLOSE_HOT: [(u8, Color); 5] = [
    (0, Color::new(0xFF, 0xD0, 0xD0)),
    (30, Color::new(0xF0, 0x80, 0x80)),
    (128, CLOSE_HOT),
    (129, Color::new(0xCE, 0x38, 0x38)),
    (255, Color::new(0xB8, 0x2C, 0x2C)),
];
/// The pale outline round a caption button, and the catchlight along the top
/// row of the bar itself.
pub const CAP_EDGE: Color = Color::new(0xFF, 0xE0, 0xBC);
/// The glyph on a caption button. White, because these sit on colour now.
pub const CAP_INK: Color = Color::new(0xFF, 0xFF, 0xFF);

/// A push button. The ramp is there to catch the top edge rather than to be
/// looked at, which is why it is three stops and nearly flat by the foot.
pub const BTN: [(u8, Color); 3] = [
    (0, Color::new(0xFF, 0xFF, 0xFF)),
    (24, Color::new(0xF6, 0xF4, 0xEA)),
    (255, FACE),
];
pub const BTN_DOWN: [(u8, Color); 3] = [
    (0, Color::new(0xC7, 0xC4, 0xB4)),
    (32, Color::new(0xDD, 0xDA, 0xCA)),
    (255, FACE),
];
/// Under the pointer. Luna's focused button glows; this is the cheap version
/// of the glow, which is a warmer face and a brighter outline.
pub const BTN_HOT: [(u8, Color); 3] = [
    (0, Color::new(0xFF, 0xFB, 0xE8)),
    (24, Color::new(0xFF, 0xEB, 0xB8)),
    (255, Color::new(0xFF, 0xD1, 0x74)),
];
/// The one-pixel outline XP puts round every button, and its lit form.
pub const BTN_EDGE: Color = APERTURE_DEEP;
pub const BTN_EDGE_HOT: Color = APERTURE;

/// The taskbar. The accent at full width, and deeper than the caption
/// deliberately: thirty-eight pixels of `APERTURE` across the whole screen
/// would be the brightest thing on it. XP made the same choice, and its
/// taskbar is a notably darker blue than its captions are.
pub const TASKBAR: [(u8, Color); 6] = [
    (0, Color::new(0xE8, 0x9F, 0x3E)),
    (14, Color::new(0xC4, 0x76, 0x18)),
    (34, Color::new(0x9C, 0x54, 0x0C)),
    (128, Color::new(0x86, 0x47, 0x0B)),
    (129, Color::new(0x7A, 0x40, 0x0A)),
    (255, Color::new(0x62, 0x33, 0x08)),
];
/// One bright line where the bar meets the desktop.
pub const TASK_EDGE: Color = Color::new(0xFF, 0xC8, 0x84);
/// The clock and battery recess. Darker than the bar, because a well on a
/// coloured surface reads by being darker and not by being outlined.
pub const TRAY: Color = Color::new(0x6E, 0x3A, 0x09);
pub const TRAY_EDGE: Color = Color::new(0x4A, 0x26, 0x06);

/// Start. Green in every real XP scheme, including the blue one, which is why
/// it stays green here rather than following the accent.
pub const START: [(u8, Color); 6] = [
    (0, Color::new(0x7A, 0xC0, 0x4E)),
    (16, Color::new(0x5C, 0xA5, 0x33)),
    (48, Color::new(0x45, 0x8C, 0x24)),
    (128, Color::new(0x39, 0x7C, 0x1C)),
    (129, Color::new(0x2F, 0x6C, 0x16)),
    (255, Color::new(0x24, 0x58, 0x10)),
];
pub const START_HOT: [(u8, Color); 6] = [
    (0, Color::new(0x96, 0xD6, 0x6A)),
    (16, Color::new(0x74, 0xBF, 0x46)),
    (48, Color::new(0x5A, 0xA7, 0x32)),
    (128, Color::new(0x4C, 0x95, 0x26)),
    (129, Color::new(0x42, 0x85, 0x1E)),
    (255, Color::new(0x35, 0x70, 0x16)),
];
pub const START_EDGE: Color = Color::new(0x8E, 0xD4, 0x6A);
pub const START_TEXT: Color = Color::new(0xFF, 0xFF, 0xFF);

/// A popup menu. White with a face-coloured gutter down its left and one
/// border, which is what every Windows menu has been since XP.
pub const MENU_BG: Color = Color::new(0xFF, 0xFF, 0xFF);
pub const MENU_GUTTER: Color = FACE;
pub const MENU_EDGE: Color = Color::new(0xAC, 0xA8, 0x99);
/// How wide the gutter is. Nothing is drawn in it yet -- XP puts an icon or a
/// tick there -- and it is here because a menu without one reads as a list
/// box, which is a different control that does a different thing.
pub const MENU_GUTTER_W: u32 = 20;

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
///
/// Fourteen rather than eight above a 16-pixel line. A caption is the surface
/// somebody looks at to know which window has the keyboard, and the 3.1
/// proportion crowds it: XP put air around the same text and that air is most
/// of why one reads as modern and the other does not.
pub const TITLE_H: u32 = font::GLYPH_H * CHROME_SCALE + 14;
/// Height of a menu bar and of one row in a dropdown.
///
/// **Its own constant, and that is the point of it.** It was
/// `= theme::TITLE_H` over in `desk`, which is a coincidence and not a fact:
/// the moment the caption grew, every menu row and every dropdown row would
/// have grown with it, silently, to thirty pixels around sixteen of text.
pub const MENU_H: u32 = font::GLYPH_H * CHROME_SCALE + 6;
/// Thickness of a window's outer frame.
///
/// Three, and it is a coloured border now rather than a grey slab. XP's frame
/// says which window is active along its whole length; 3.1's said only that
/// there was an edge here to grab.
pub const FRAME: u32 = 3;

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

pub fn chrome(frame: Rect, has_menus: bool) -> Chrome {
    let inner = frame.shrink(FRAME);
    let title = Rect::new(inner.x, inner.y, inner.w, TITLE_H);
    // Two pixels of border show between the caption and the body, which under
    // Luna is a thin accent line rather than the gap it used to be.
    let body = Rect::new(
        inner.x,
        inner.y + TITLE_H + 2,
        inner.w,
        inner.h.saturating_sub(TITLE_H + 2),
    );
    let (menubar, client) = if has_menus {
        (
            Some(Rect::new(body.x, body.y, body.w, MENU_H)),
            Rect::new(body.x, body.y + MENU_H, body.w, body.h.saturating_sub(MENU_H)),
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
    // The border is the caption's colour carried round the outside, painted as
    // the whole window and then covered. Two fills rather than four strips,
    // and the strip between the caption and the body comes out of it for
    // free.
    fb.rect(r.x, r.y, r.w, r.h, if active { BORDER_ON } else { BORDER_OFF });
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
}

/// Inset from the panel edge to the first row.
const POPUP_PAD: u32 = 3;

impl Popup {
    /// Sized from its widest label in columns and its row count, clamped to
    /// whatever room is left on the right.
    pub fn sized(x: u32, y: u32, cols: usize, rows: usize, max_w: u32) -> Popup {
        let w = (text_w(cols) + 24 + MENU_GUTTER_W).min(max_w);
        let h = rows as u32 * MENU_H + POPUP_PAD * 2;
        Popup { panel: Rect::new(x, y, w, h), rows }
    }

    /// Where row `i` is drawn: to the right of the gutter, which is why a
    /// selection bar stops at it rather than running under it.
    pub fn row(&self, i: usize) -> Rect {
        let inset = POPUP_PAD + MENU_GUTTER_W;
        Rect::new(
            self.panel.x + inset,
            self.panel.y + POPUP_PAD + i as u32 * MENU_H,
            self.panel.w.saturating_sub(inset + POPUP_PAD),
            MENU_H,
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
        let row = (y - (p.y + POPUP_PAD) as i32) / MENU_H as i32;
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
        self.panel.w >= MENU_GUTTER_W + 16
    }
}

/// The ground a popup's rows sit on: white, a gutter, and one border.
pub fn popup(fb: &Framebuffer, p: &Popup) {
    let r = p.panel;
    fb.rect(r.x, r.y, r.w, r.h, MENU_BG);
    fb.rect(r.x + 1, r.y + 1, MENU_GUTTER_W, r.h.saturating_sub(2), MENU_GUTTER);
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
    let s = bar.h.saturating_sub(9);
    let y = bar.y + (bar.h.saturating_sub(s)) / 2;
    let close_x = bar.x + bar.w.saturating_sub(s + 5);
    // The close button sits apart from the pair, as it has since 95 -- the
    // one you reach for blind should not share an edge with the one that
    // merely tidies. XP itself puts all three flush and this keeps the gap:
    // the argument for it did not stop being true because Redmond stopped
    // making it.
    let max_x = close_x.saturating_sub(s + 4);
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
    fb.vgrad(r.x, r.y, r.w, r.h, if active { &TITLE_ON } else { &TITLE_OFF });
    // A one-pixel catchlight along the very top. The stop table is
    // proportional and so cannot express an absolute one-pixel edge at any
    // height, which is the intended division of labour between the ramp and
    // the line drawn on top of it.
    fb.rect(r.x, r.y, r.w, 1, if active { CAP_EDGE } else { EDGE_LIGHT });

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
    text_over(fb, tx, ty, shown, if active { TITLE_TEXT } else { TITLE_IDLE });

    // The caption buttons: rounded ramps on the caption's own ramp, glyphs
    // drawn by hand because the font has no box-drawing worth the name.
    //
    // **Close is red whether or not the pointer is on it.** That overturns the
    // note on `CLOSE_HOT`, which argued the red was the hover feedback and
    // that every other control here answers hover with a ring. Under Luna the
    // red is the control's identity rather than its response, and an operator
    // coming from any Windows since XP finds it by colour before they read the
    // glyph. Hover still says something -- it brightens.
    for (i, b) in btns.iter().enumerate() {
        let hot = hot_caption == Some(i);
        let close = i == 2;
        let stops: &[(u8, Color)] = match (close, hot) {
            (true, true) => &CAP_CLOSE_HOT,
            (true, false) => &CAP_CLOSE,
            (false, true) => &CAP_BTN_HOT,
            (false, false) => &CAP_BTN,
        };
        control(fb, *b, stops, CAP_EDGE);
        let ink = CAP_INK;
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
                    fb.rect(g.x, g.y + 4, s, s, stops[2].1);
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

    let tw = text_w_of(label);
    let tx = r.x + (r.w.saturating_sub(tw)) / 2 + u32::from(pressed);
    let ty = r.y + (r.h.saturating_sub(font::GLYPH_H * CHROME_SCALE)) / 2 + u32::from(pressed);
    // `text_over` rather than `text`: the face is a ramp now, so stamping one
    // background colour behind every glyph would print a flat block across it.
    text_over(fb, tx, ty, label, TEXT);
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
        (TEXT, MENU_BG)
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
/// White, not `APERTURE_DEEP`. Its one caller is the Start button, which is
/// green now, and a deep orange on that green reads as mud.
pub fn aperture_dot(fb: &Framebuffer, cx: u32, cy: u32, r: i32) {
    mark(fb, cx as i32, cy as i32, r.max(3), HILIGHT);
}

/// A horizontal rule, drawn as a groove. The 3.1 separator.
pub fn separator(fb: &Framebuffer, x: u32, y: u32, w: u32) {
    fb.rect(x, y, w, 1, EDGE);
    fb.rect(x, y + 1, w, 1, EDGE_LIGHT);
}
