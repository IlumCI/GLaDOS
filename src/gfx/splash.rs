//! The boot screen.
//!
//! Not decoration. Boot takes a while -- 129 MB of weights come off a USB
//! stick, then eleven sets of test vectors run -- and on the GF63 there is no
//! serial port, so a machine that appears to be doing nothing is
//! indistinguishable from a machine that has hung. A progress bar is the
//! difference between "wait" and "reboot and give up".
//!
//! ### Nothing is hidden
//!
//! The console keeps a shadow grid in RAM, so while this owns the framebuffer
//! the boot log is still being written -- just not painted. `finish` gives the
//! screen back and repaints the lot, so the familiar text is there to read a
//! moment later, exactly as before. That matters more than the splash does:
//! the framebuffer is the only diagnostic channel this machine has.
//!
//! ### Why it looks like this
//!
//! Raised panel, sunken trough, two-pixel bevels, one bitmap font, four
//! colours out of sixteen. That is a Windows 3.x dialog, and it is also the
//! cheapest thing a framebuffer can draw: no blending, no antialiasing, no
//! gradients, no scaling. The aesthetic and the constraint are the same
//! choice, which is why it will still look right when there is a window
//! manager behind it.

use super::palette::{AMBER, BLACK, BLUE, DKGRAY, LTGRAY, WHITE};
use super::{font, primary};
use crate::sync::Racy;

/// How many `stage` calls make a full bar.
///
/// Kept in step with the call sites in `main` by hand. If it drifts the bar
/// finishes early or stops short -- ugly, never wrong, and never fatal, which
/// is the right failure mode for a progress indicator.
const STAGES: u32 = 9;

static STEP: Racy<u32> = Racy::new(0);
static ACTIVE: Racy<bool> = Racy::new(false);

pub fn active() -> bool {
    unsafe { *ACTIVE.get() }
}

/// Panel geometry, centred on whatever panel the firmware gave us.
///
/// Everything is stacked in order and the panel height falls out of the sum,
/// rather than each element being placed at some fraction of the whole. The
/// fractional version put the trough on top of the subtitle at one particular
/// resolution, which is the failure mode that arrangement always has.
struct Layout {
    px: u32,
    py: u32,
    pw: u32,
    ph: u32,
    scale: u32,
    logo_cy: u32,
    logo_r: u32,
    title_y: u32,
    sub_y: u32,
    bar_y: u32,
    bar_h: u32,
    label_y: u32,
}

fn layout(w: u32, h: u32) -> Layout {
    // Scale the whole thing off the panel width so a 1920x1080 laptop and an
    // 800x600 QEMU window both look deliberate rather than one being a
    // postage stamp.
    let scale = if w >= 1600 {
        3
    } else if w >= 1024 {
        2
    } else {
        1
    };
    let pw = (w * 3 / 5).max(320);
    let gh = font::GLYPH_H * scale;
    let pad = gh;
    let logo_r = (gh * 5).min(pw / 5);

    // Stack downward from the top of the panel.
    let mut y = pad;
    let logo_cy = y + logo_r;
    y = logo_cy + logo_r + pad;
    let title_y = y;
    y += font::GLYPH_H * (scale + 1) + pad / 2;
    let sub_y = y;
    y += gh + pad;
    let bar_y = y;
    let bar_h = gh;
    y += bar_h + pad / 2;
    let label_y = y;
    let ph = label_y + gh + pad;

    Layout {
        px: (w - pw) / 2,
        py: (h.saturating_sub(ph)) / 2,
        pw,
        ph,
        scale,
        logo_cy,
        logo_r,
        title_y,
        sub_y,
        bar_y,
        bar_h,
        label_y,
    }
}

/// Unit vectors every 15 degrees, scaled by 1000.
///
/// A whole trig implementation for a logo would be silly, and there is no
/// floating point this early anyway. Fifteen degrees is the coarsest step that
/// still divides the six blade positions and their offsets exactly.
const DIRS: [(i32, i32); 24] = [
    (1000, 0), (966, 259), (866, 500), (707, 707), (500, 866), (259, 966),
    (0, 1000), (-259, 966), (-500, 866), (-707, 707), (-866, 500), (-966, 259),
    (-1000, 0), (-966, -259), (-866, -500), (-707, -707), (-500, -866), (-259, -966),
    (0, -1000), (259, -966), (500, -866), (707, -707), (866, -500), (966, -259),
];

const BLADES: usize = 6;
/// 60 degrees, in 15-degree units.
const BLADE_STEP: usize = 4;

/// The iris on the wall at Aperture Science.
///
/// Drawn parametrically rather than stored as a bitmap: it costs no bytes on
/// the ESP, scales to whatever panel the firmware reports, and is a geometric
/// figure -- a camera aperture -- rather than a traced copy of anybody's
/// artwork.
///
/// Cut, not drawn. The first two attempts stroked the blade edges as lines,
/// and that is wrong twice over: it looks like wireframe, and full chords
/// between evenly spaced points always make a star -- at a quarter-circle span
/// it is literally two overlapping squares. What the mark actually is: a solid
/// disc with wedges taken *out* of it, each wedge narrow at the centre and
/// wide at the rim, swept round so the remaining blades appear to spiral.
///
/// So the gaps are painted in the background colour and the blades are simply
/// whatever disc is left over, which is also how a real iris works.
fn aperture(fb: &super::Framebuffer, cx: i32, cy: i32, r: i32, fg: super::Color, bg: super::Color) {
    let at = |k: usize, rad: i32| -> (i32, i32) {
        let (dx, dy) = DIRS[k % 24];
        (cx + dx * rad / 1000, cy + dy * rad / 1000)
    };

    fb.fill_circle(cx, cy, r, fg);

    // The opening. This is what every earlier attempt missed: the blades are a
    // *ring*, and the middle is open. A solid hub is what made the last version
    // read as a flower -- six petals around a centre, rather than six blades
    // around a hole.
    //
    // Its corners are where consecutive blade edges meet, so the opening is a
    // hexagon and not a circle, and the slashes below start from those corners.
    let open_r = r * 46 / 100;
    let centre = (cx, cy);
    for b in 0..BLADES {
        let k = b * BLADE_STEP;
        fb.fill_triangle(centre, at(k, open_r), at(k + BLADE_STEP, open_r), bg);
    }

    // One slash per corner, running outward and swept well off radial.
    //
    // A slash is a constant-width quadrilateral, not a triangle fanning out to
    // a span of the rim. A triangle wide enough to read at the edge is far too
    // wide where it meets the opening, and it takes the blades with it -- the
    // previous attempt cut them down to stubs. Width is set in pixels and the
    // ends are found by stepping perpendicular to the slash's own direction.
    let rim = r + 2;
    let half = (r / 14).max(1);
    for b in 0..BLADES {
        let k = b * BLADE_STEP;
        let (ax, ay) = at(k, open_r);
        // Out at the rim, swept 45 degrees ahead of the corner it starts from.
        let (ox, oy) = at(k + 3, rim);

        let (dx, dy) = (ox - ax, oy - ay);
        let len = (super::isqrt((dx * dx + dy * dy) as u32) as i32).max(1);
        let (px, py) = (-dy * half / len, dx * half / len);

        let a0 = (ax + px, ay + py);
        let a1 = (ax - px, ay - py);
        let o0 = (ox + px, oy + py);
        let o1 = (ox - px, oy - py);
        fb.fill_triangle(a0, a1, o1, bg);
        fb.fill_triangle(a0, o1, o0, bg);
    }
}

fn centred_text(fb: &super::Framebuffer, cx: u32, y: u32, s: &str, scale: u32, fg: super::Color) {
    let width = s.len() as u32 * font::GLYPH_W * scale;
    let x = cx.saturating_sub(width / 2);
    fb.draw_text(x, y, s, fg, LTGRAY, scale);
}

/// Take the screen and draw the frame.
pub fn begin() {
    let Some(fb) = primary() else { return };
    unsafe {
        *STEP.get() = 0;
        *ACTIVE.get() = true;
    }
    crate::gfx::console::with(|c| c.set_visible(false));

    let (w, h) = (fb.width(), fb.height());
    let l = layout(w, h);

    fb.fill(BLUE);

    // The panel: face, raised bevel, then a black outer line so it separates
    // from the background the way a dialog does.
    fb.rect(l.px, l.py, l.pw, l.ph, LTGRAY);
    fb.bevel(l.px, l.py, l.pw, l.ph, true);
    fb.frame(l.px - 1, l.py - 1, l.pw + 2, l.ph + 2, BLACK);

    let cx = l.px + l.pw / 2;

    aperture(
        &fb,
        cx as i32,
        (l.py + l.logo_cy) as i32,
        l.logo_r as i32,
        AMBER,
        LTGRAY,
    );

    centred_text(&fb, cx, l.py + l.title_y, "GLaDOS", l.scale + 1, BLACK);
    centred_text(
        &fb,
        cx,
        l.py + l.sub_y,
        "a model in the kernel",
        l.scale.max(1),
        DKGRAY,
    );

    // The trough. Sunken, because the bar sits inside it.
    let (bx, by, bw, bh) = bar_rect(&l);
    fb.rect(bx, by, bw, bh, WHITE);
    fb.bevel(bx, by, bw, bh, false);
    render(0, "starting");
}

fn bar_rect(l: &Layout) -> (u32, u32, u32, u32) {
    let bw = l.pw * 4 / 5;
    (l.px + (l.pw - bw) / 2, l.py + l.bar_y, bw, l.bar_h)
}

fn render(step: u32, label: &str) {
    let Some(fb) = primary() else { return };
    let l = layout(fb.width(), fb.height());
    let (bx, by, bw, bh) = bar_rect(&l);

    // Fill proportionally, inside the bevel so the trough edge stays visible.
    let inner_w = bw.saturating_sub(4);
    let filled = (inner_w * step.min(STAGES)) / STAGES;
    if filled > 0 {
        fb.rect(bx + 2, by + 2, filled, bh.saturating_sub(4), BLUE);
    }

    let _ = (by, bh);
    // The label sits under the trough, on a repainted strip so a shorter name
    // does not leave the tail of a longer one behind it.
    let ly = l.py + l.label_y;
    fb.rect(l.px + 2, ly, l.pw - 4, font::GLYPH_H * l.scale, LTGRAY);
    centred_text(&fb, l.px + l.pw / 2, ly, label, l.scale.max(1), BLACK);
}

/// Advance one step and name what is happening.
pub fn stage(label: &str) {
    if !active() {
        return;
    }
    let step = unsafe {
        *STEP.get() += 1;
        *STEP.get()
    };
    render(step, label);
}

/// Change the label without advancing the bar.
///
/// For the moments boot pauses to ask the operator something -- the recovery
/// prompt is the only one so far. The question is printed to a console nobody
/// can currently see, so it has to be said here too.
pub fn note(label: &str) {
    if !active() {
        return;
    }
    let step = unsafe { *STEP.get() };
    render(step, label);
}

/// Give the framebuffer back and repaint the boot log over the top.
pub fn finish() {
    if !active() {
        return;
    }
    render(STAGES, "ready");
    unsafe { *ACTIVE.get() = false };
    crate::gfx::console::with(|c| {
        c.set_visible(true);
    });
    // The console has been writing to its shadow grid the whole time, so
    // reflowing it into the window's client area and repainting brings the
    // boot log back at the new origin -- nothing logged during boot is lost by
    // gaining a frame around it.
    crate::gfx::desk::init();
}

/// Abandon the splash immediately, keeping whatever is on screen.
///
/// For a fault: the reporter draws straight to the framebuffer, and a panic
/// behind a progress bar helps nobody.
pub fn abandon() {
    if !active() {
        return;
    }
    unsafe { *ACTIVE.get() = false };
    crate::gfx::console::with(|c| c.set_visible(true));
    // Deliberately *not* `ui::chrome()`. This is the fault path: the reporter
    // is about to draw and the console is the only diagnostic channel there
    // is, so the cheapest thing that makes text visible is the right thing.
    // Painting a window frame first would be decoration on the way to a halt.
    crate::gfx::console::redraw();
}
