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

/// Blade directions and opening vertices, as unit vectors scaled by 1000.
///
/// Seven blades, so the angles are multiples of 360/7 and no table of round
/// degrees divides them. A whole trig implementation for a logo would be silly
/// and there is no floating point this early, so the fourteen vectors the mark
/// needs are simply written down.
///
/// `BLADE_DIR[i]` points at the tangent point of cut `i`. `OPEN_DIR[i]` points
/// at a vertex of the opening, offset half a step, which is where two blade
/// edges meet.
const BLADE_DIR: [(i32, i32); BLADES] = [
    (1000, 0), (623, 782), (-222, 975), (-901, 434),
    (-901, -434), (-222, -975), (623, -782),
];
const OPEN_DIR: [(i32, i32); BLADES] = [
    (901, 434), (222, 975), (-623, 782), (-1000, 0),
    (-623, -782), (222, -975), (901, -434),
];

const BLADES: usize = 7;
/// Opening radius, as hundredths of the disc radius.
const OPEN_PCT: i32 = 46;
/// Circumradius of the opening is its inradius over cos(pi/7) = 0.9010.
const OPEN_CIRCUM_NUM: i32 = 1110;
/// Half-width of a cut, as thousandths of the disc radius.
const CUT_PCT: i32 = 35;

/// The iris on the wall at Aperture Science.
///
/// Drawn parametrically rather than stored as a bitmap: it costs no bytes on
/// the ESP, scales to whatever panel the firmware reports, and is a geometric
/// figure -- a camera aperture -- rather than a traced copy of anybody's
/// artwork.
///
/// Cut, not drawn: a solid disc with wedges taken *out* of it. Three ways this
/// has been got wrong here, each of which looked plausible until put beside a
/// real aperture:
///
///   * Stroking the blade edges as lines gives wireframe, and full chords
///     between evenly spaced points always make a star.
///   * Leaving the middle solid gives a flower -- petals around a hub, rather
///     than blades around a hole. The middle is *open*, and largely so.
///   * **Cutting radially.** This was the long-lived one. A cut aimed out from
///     the centre only notches the disc, and six of them read as a wheel. The
///     cuts are **tangent to the opening**, so each blade's inner edge is a
///     straight chord and the leftover blades appear to spiral. That tangency
///     is the entire mark; without it the number of blades hardly matters.
///
/// Public because the desktop wall draws the same mark. One definition of what
/// the logo *is*, so the wall and the boot screen cannot drift apart --
/// `tools/mklogo.py` is a port of this and must be re-run if it changes.
pub fn aperture(fb: &super::Framebuffer, cx: i32, cy: i32, r: i32, fg: super::Color, bg: super::Color) {
    let scaled = |(dx, dy): (i32, i32), rad: i32| (cx + dx * rad / 1000, cy + dy * rad / 1000);

    fb.fill_circle(cx, cy, r, fg);

    // The opening: the polygon bounded by the same lines the cuts run along.
    // A circular hole leaves a nub where each straight cut meets the curve;
    // the polygon is what gives the blades their points.
    let rin = r * OPEN_PCT / 100;
    let circum = rin * OPEN_CIRCUM_NUM / 1000;
    let centre = (cx, cy);
    for b in 0..BLADES {
        let v0 = scaled(OPEN_DIR[b], circum);
        let v1 = scaled(OPEN_DIR[(b + 1) % BLADES], circum);
        fb.fill_triangle(centre, v0, v1, bg);
    }

    // One cut per blade, tangent to the opening, running out past the rim. The
    // chord from a tangent point to the rim is sqrt(r^2 - rin^2); overshooting
    // it slightly means the cut leaves the disc cleanly instead of stopping a
    // pixel short and leaving a bridge.
    let reach = (super::isqrt((r * r - rin * rin) as u32) as i32) * 106 / 100;
    let half = (r * CUT_PCT / 1000).max(1);
    for b in 0..BLADES {
        let (ux, uy) = BLADE_DIR[b];
        let (ax, ay) = scaled(BLADE_DIR[b], rin);
        // Tangent is the radial direction turned a quarter turn.
        let (ex, ey) = (ax - uy * reach / 1000, ay + ux * reach / 1000);
        // Width is measured along the radius, which is normal to the cut.
        let (ox, oy) = (ux * half / 1000, uy * half / 1000);
        let a0 = (ax + ox, ay + oy);
        let a1 = (ax - ox, ay - oy);
        let e0 = (ex + ox, ey + oy);
        let e1 = (ex - ox, ey - oy);
        fb.fill_triangle(a0, a1, e1, bg);
        fb.fill_triangle(a0, e1, e0, bg);
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
