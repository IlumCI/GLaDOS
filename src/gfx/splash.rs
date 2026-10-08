//! The boot screen.
//!
//! Not decoration. Boot takes a while -- 129 MB of weights come off a USB
//! stick, then the selftests run -- and on the GF63 there is no serial port,
//! so a machine that appears to be doing nothing is indistinguishable from a
//! machine that has hung. The old progress bar answered "wait, do not reboot";
//! this answers the same question a different way.
//!
//! ### What it shows
//!
//! The Aperture iris, forming one blade at a time, clockwise from the top. By
//! the last boot stage it is whole. A blade that is not yet there is the
//! progress the trough used to carry, read off the shape of the mark instead
//! of a filling rectangle -- so the thing that says "still working" is the
//! same thing that says what this machine *is*.
//!
//! ### Why it looks like this
//!
//! A near-black field with the mark lit as the one light source on it, placed
//! on the exact pixels the desktop wallpaper uses for its own iris -- centre of
//! the screen, `r = min(h/5, w/5)`, filled with `theme::SUN`. So the hand-off
//! at `finish` does not move the logo: the field dissolves from black to the
//! Aero sky around a mark that stays put. Each effect is a span-based or
//! single-read primitive the framebuffer already had, so none of it costs what
//! blending a photograph would:
//!
//!   * **Depth** is `Face::Ramp` -- the disc is filled with a top-bright,
//!     bottom-dark amber ramp, which reads as a sphere catching light from
//!     above rather than a sticker.
//!   * **Glow** is a radial amber halo blended into the dark field behind the
//!     disc, gated to the sectors of blades that have formed, so it grows with
//!     the reveal. `Framebuffer::blend` reads the field back and lightens it
//!     toward amber, so the halo is of the field rather than painted over it.
//!   * **Reflection** is the lower hemisphere mirrored downward onto the
//!     field, fading with distance -- a wet-floor copy, built by reading the
//!     disc back through `get` and blending it under.
//!   * **Finale** is `flare`: a warm bloom expands from the finished mark and
//!     washes the screen, then the desktop takes over with the iris in place.
//!
//! The continuous clockwise light-sweep of the design, the wide ambient cast
//! and the soft beam bloom want a back buffer to blend against cheaply and a
//! frame driver to animate between the boot stages; they are a separate change
//! on top of this one. What is here reveals the iris a blade per stage, which
//! is the progress the trough used to carry, read off the shape of the mark.
//!
//! ### Nothing is hidden
//!
//! The console keeps a shadow grid in RAM, so while this owns the framebuffer
//! the boot log is still being written -- just not painted. `finish` gives the
//! screen back and repaints the lot, so the familiar text is there to read a
//! moment later. That matters more than the splash does: the framebuffer is
//! the only diagnostic channel this machine has.

use super::palette::{BLACK, WHITE};
use super::{font, primary, Color, Format, Framebuffer};
use crate::sync::Racy;

/// How many `stage` calls make a whole iris.
///
/// Kept in step with the call sites in `main` by hand -- there are ten, so the
/// eighth-and-final blade lands on "ready". If it drifts the iris completes
/// early or stops a blade short: ugly, never wrong, and never fatal, which is
/// the right failure mode for a progress indicator.
const STAGES: u32 = 10;

static STEP: Racy<u32> = Racy::new(0);
static ACTIVE: Racy<bool> = Racy::new(false);

pub fn active() -> bool {
    unsafe { *ACTIVE.get() }
}

/// The field behind everything: a deep-blue vertical ramp, top to bottom.
///
/// Absolute screen rows, so a hole cut anywhere in it lines up with it -- which
/// is exactly what `Cut::Sky` needs when the iris's gaps show the field
/// through.
const FIELD: &[(u8, Color)] = &[
    (0, Color::new(2, 4, 10)),
    (150, Color::new(4, 8, 18)),
    (255, Color::new(9, 15, 30)),
];

/// The disc, lit from above. **Exactly `theme::SUN`**, the ramp the desktop
/// wallpaper fills its own iris with, so when the splash hands off the mark is
/// identical on both sides and only the background changes underneath it. One
/// value, so the two cannot drift; `diag`-style agreement is not needed because
/// there is one source.
const SUN: &[(u8, Color)] = &super::theme::SUN;

/// What the tight rim halo lightens the field toward.
const GLOW_C: Color = Color::new(255, 196, 104);
/// What the sun-flare beam and bloom lighten toward at the finale.
const BEAM_C: Color = Color::new(255, 236, 200);

/// Geometry, vertically centred, scaled off the smaller of width and height so
/// the mark is the hero at any resolution.
///
/// Everything is stacked in order and the block's height falls out of the sum,
/// rather than each element being placed at some fraction of the whole -- the
/// fractional version put one thing on top of another at one particular
/// resolution, which is the failure mode that arrangement always has.
struct Layout {
    scale: u32,
    cx: u32,
    cy: u32,
    r: u32,
    ro: u32,
    gap: u32,
    refl_h: u32,
    title_y: u32,
    sub_y: u32,
    label_y: u32,
    h: u32,
}

fn layout(w: u32, h: u32) -> Layout {
    let scale = if w >= 1600 {
        3
    } else if w >= 1024 {
        2
    } else {
        1
    };
    let gh = font::GLYPH_H * scale;
    // The iris is placed on the EXACT pixels the desktop wallpaper uses for its
    // own mark -- `theme::wallpaper` centres it at (w/2, h/2) with
    // r = min(h/5, w/5) -- so the hand-off at `finish` never moves it: the
    // background dissolves from near-black to the Aero sky around a logo that
    // stays put.
    let r = (h / 5).min(w / 5).max(40);
    let ro = r * 3 / 2; // glow reaches half a radius past the rim
    let gap = (r / 10).max(2);
    let refl_h = r * 2 / 3;
    let title_h = font::GLYPH_H * (scale + 1);

    let cx = w / 2;
    let cy = h / 2;
    // Title and subtitle sit above the glow; the stage label below the
    // reflection. Everything that is not the mark fades out at the flare, so
    // its exact placement matters only while the iris is forming.
    let glow_top = cy.saturating_sub(ro);
    let sub_y = glow_top.saturating_sub(gap + gh);
    let title_y = sub_y.saturating_sub(gap + title_h);
    let label_y = cy + r + gap + refl_h + gap * 2;

    Layout {
        scale,
        cx,
        cy,
        r,
        ro,
        gap,
        refl_h,
        title_y,
        sub_y,
        label_y,
        h,
    }
}

/// Blade directions and opening vertices, as unit vectors scaled by 1000.
///
/// There is no floating point this early and a trig implementation for a logo
/// would be silly, so the vectors the mark needs are written down. Generated
/// rather than derived by hand -- `tools/mklogo.py` holds the same
/// construction in floating point and is where the geometry changes first.
///
/// `BLADE_DIR[i]` points at the tangent point of cut `i`. `OPEN_DIR[i]` points
/// at a vertex of the opening, offset half a step, which is where two blade
/// edges meet.
const BLADE_DIR: [(i32, i32); BLADES] = [
    (1000, 0), (707, 707), (0, 1000), (-707, 707),
    (-1000, 0), (-707, -707), (0, -1000), (707, -707),
];
const OPEN_DIR: [(i32, i32); BLADES] = [
    (924, 383), (383, 924), (-383, 924), (-924, 383),
    (-924, -383), (-383, -924), (383, -924), (924, -383),
];

/// Eight, and it went from seven to get here.
///
/// Seven made each blade a seventh of the disc, which at the size the wall
/// draws it is a wedge big enough to read as a triangle rather than as a
/// blade. A real iris has more and finer ones. Eight also puts every angle on
/// a multiple of 45 degrees, so the table above is exact rather than a rounded
/// approximation of a seventh of a turn.
const BLADES: usize = 8;
/// Opening radius, as hundredths of the disc radius.
const OPEN_PCT: i32 = 46;
/// Circumradius of the opening is its inradius over cos(pi/8) = 0.9239.
const OPEN_CIRCUM_NUM: i32 = 1082;
/// Half-width of a cut, as thousandths of the disc radius. Narrower with more
/// blades, or the cuts take more of the disc than the blades do -- and the cut
/// is a gap between blades rather than a spoke, so it wants to be thin enough
/// to read as one.
const CUT_PCT: i32 = 22;

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
    aperture_with(fb, cx, cy, r, Face::Flat(fg), Cut::Solid(bg));
}

/// What the blades themselves are made of.
///
/// Flat is the mark as a mark: a boot screen, a favicon, an icon on a bar.
/// `Ramp` is the mark as an object with light on it, which is what it becomes
/// on a wall that already has a sky and a horizon -- at that size and in that
/// company a flat disc reads as a sticker and a lit one reads as a sun. The
/// boot screen now asks for `Ramp` too, for the same reason.
#[derive(Clone, Copy)]
pub enum Face<'a> {
    Flat(super::Color),
    Ramp(&'a [(u8, super::Color)]),
}

/// What a cut through the disc is filled with.
///
/// The blades are solid and the cuts between them are gaps, so what a gap is
/// filled with has to be whatever was behind the mark. On a boot screen and on
/// a twenty-pixel icon that is one colour and `Solid` is exact. On a wall that
/// is a gradient it is not: a single colour there belongs to no part of the
/// sky and reads as a rectangle of the wrong shade laid across the disc, which
/// is what a cut is least allowed to look like.
#[derive(Clone, Copy)]
pub enum Cut<'a> {
    Solid(super::Color),
    Sky {
        stops: &'a [(u8, super::Color)],
        top: u32,
        height: u32,
    },
}

/// The mark, with the caller saying what shows through its blades.
///
/// One geometry and one set of constants for both. Two marks that agreed about
/// where a blade goes only while somebody kept them agreeing would be the
/// duplicated-layout bug wearing a logo.
pub fn aperture_with(
    fb: &super::Framebuffer,
    cx: i32,
    cy: i32,
    r: i32,
    face: Face<'_>,
    cut: Cut<'_>,
) {
    let scaled = |(dx, dy): (i32, i32), rad: i32| (cx + dx * rad / 1000, cy + dy * rad / 1000);
    let carve = |a: (i32, i32), b: (i32, i32), c: (i32, i32)| match cut {
        Cut::Solid(bg) => fb.fill_triangle(a, b, c, bg),
        Cut::Sky { stops, top, height } => fb.fill_triangle_over(a, b, c, stops, top, height),
    };

    match face {
        Face::Flat(fg) => fb.fill_circle(cx, cy, r, fg),
        Face::Ramp(stops) => fb.fill_circle_ramp(cx, cy, r, stops),
    }

    // The opening: the polygon bounded by the same lines the cuts run along.
    // A circular hole leaves a nub where each straight cut meets the curve;
    // the polygon is what gives the blades their points.
    let rin = r * OPEN_PCT / 100;
    let circum = rin * OPEN_CIRCUM_NUM / 1000;
    let centre = (cx, cy);
    for b in 0..BLADES {
        let v0 = scaled(OPEN_DIR[b], circum);
        let v1 = scaled(OPEN_DIR[(b + 1) % BLADES], circum);
        carve(centre, v0, v1);
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
        carve(a0, a1, e1);
        carve(a0, e1, e0);
    }
}

/// The order blades appear in: clockwise from just right of twelve o'clock.
///
/// `OPEN_DIR` is indexed from three o'clock going clockwise, so this is that
/// list rotated to start at the top -- blade 6 (~1 o'clock) first, round to
/// blade 5 (~11 o'clock) last.
const REVEAL: [usize; BLADES] = [6, 7, 0, 1, 2, 3, 4, 5];

/// How many blades are showing at a given step. Rounds to nearest so the eight
/// blades spread evenly across ten stages rather than bunching at the end.
fn blades_for(step: u32) -> usize {
    let step = step.min(STAGES);
    (((step * BLADES as u32 + STAGES / 2) / STAGES) as usize).min(BLADES)
}

fn decode(fb: &Framebuffer, raw: u32) -> Color {
    match fb.format() {
        Format::Rgbx => Color::new(
            (raw & 0xff) as u8,
            ((raw >> 8) & 0xff) as u8,
            ((raw >> 16) & 0xff) as u8,
        ),
        Format::Bgrx => Color::new(
            ((raw >> 16) & 0xff) as u8,
            ((raw >> 8) & 0xff) as u8,
            (raw & 0xff) as u8,
        ),
    }
}

/// Repaint a band with the field gradient, so a redraw starts from clean sky.
fn field_band(fb: &Framebuffer, y0: u32, y1: u32) {
    let w = fb.width();
    let h = fb.height();
    let mut y = y0;
    while y < y1.min(h) {
        fb.rect(0, y, w, 1, super::ramp_at(FIELD, y, h));
        y += 1;
    }
}

/// Which blade's 45-degree sector a direction falls in, 0..8, with no trig.
///
/// The octants line up with `BLADE_DIR`: sector `k` runs from `BLADE_DIR[k]`
/// to `BLADE_DIR[k+1]`, so a pixel in sector `k` sits behind blade `k`. Screen
/// coordinates, `+y` down.
fn sector(dx: i32, dy: i32) -> usize {
    let (ax, ay) = (dx.abs(), dy.abs());
    if dx >= 0 && dy >= 0 {
        if ax >= ay { 0 } else { 1 }
    } else if dx < 0 && dy >= 0 {
        if ay >= ax { 2 } else { 3 }
    } else if dx < 0 && dy < 0 {
        if ax >= ay { 4 } else { 5 }
    } else if ay >= ax {
        6
    } else {
        7
    }
}

/// A radial halo blended into the field, growing with the iris.
///
/// Only the ring outside the disc is touched -- the interior is about to be
/// covered by the disc -- and only the angular sectors of blades that have
/// already formed, so the glow follows the reveal instead of ringing the empty
/// space a half-formed iris would otherwise sit in. Quadratic falloff, so the
/// halo fades into the sky rather than ending at a hard circle.
fn glow(fb: &Framebuffer, l: &Layout, blades: usize) {
    if blades == 0 {
        return;
    }
    let cx = l.cx as i32;
    let cy = l.cy as i32;
    let r = l.r as i32;
    let ro = l.ro as i32;
    let peak: i32 = 200;
    let ro2 = ro * ro;
    let formed = &REVEAL[..blades];
    for dy in -ro..=ro {
        for dx in -ro..=ro {
            let d2 = dx * dx + dy * dy;
            if d2 > ro2 {
                continue;
            }
            let dist = super::isqrt(d2 as u32) as i32;
            if dist < r {
                continue; // the disc will cover this
            }
            if !formed.contains(&sector(dx, dy)) {
                continue; // no blade here yet
            }
            let t = ro - dist;
            let a = peak * t * t / (ro * ro);
            if a > 0 {
                fb.blend((cx + dx) as u32, (cy + dy) as u32, GLOW_C, a as u16);
            }
        }
    }
}

/// Occlude the blades that have not formed yet by painting their wedges back
/// to field. A wedge is the sector between two adjacent cuts, taken out past
/// the rim so no sliver of disc survives at the edge.
fn occlude_unformed(fb: &Framebuffer, l: &Layout, blades: usize) {
    let cx = l.cx as i32;
    let cy = l.cy as i32;
    let r = l.r as i32;
    let h = l.h;
    let reach = r * 115 / 100;
    let scaled = |(dx, dy): (i32, i32)| (cx + dx * reach / 1000, cy + dy * reach / 1000);
    for &k in REVEAL[blades..].iter() {
        let pk = scaled(BLADE_DIR[k]);
        let pm = scaled(OPEN_DIR[k]);
        let pk1 = scaled(BLADE_DIR[(k + 1) % BLADES]);
        fb.fill_triangle_over((cx, cy), pk, pm, FIELD, 0, h);
        fb.fill_triangle_over((cx, cy), pm, pk1, FIELD, 0, h);
    }
}

/// Mirror the lower hemisphere downward onto the field, fading with distance.
///
/// Reads the disc back as it was drawn -- including the gaps and any not-yet
/// -formed blades, which read as field -- so the reflection always matches
/// what is above it.
fn reflection(fb: &Framebuffer, l: &Layout) {
    let cx = l.cx as i32;
    let cy = l.cy as i32;
    let r = l.r as i32;
    let floor = (l.cy + l.r + l.gap) as i32;
    let refl_h = l.refl_h as i32;
    for ry in 0..refl_h {
        let src_y = cy + r - 1 - ry;
        let dst_y = floor + ry;
        if src_y < 0 || dst_y < 0 {
            continue;
        }
        // Darken as it goes down, like a reflection in a dim wet floor.
        let a = (130 * (refl_h - ry) / refl_h) as u16;
        if a == 0 {
            continue;
        }
        for dx in -r..=r {
            let x = cx + dx;
            if x < 0 {
                continue;
            }
            let col = decode(fb, fb.get(x as u32, src_y as u32));
            fb.blend(x as u32, dst_y as u32, col, a);
        }
    }
}

/// Text centred on `cx`, with a soft drop shadow so it reads over the field
/// without a box behind it. Characters, not bytes -- a byte count centres
/// anything with an accent too far left, by exactly the number of accents.
fn centred_text(fb: &Framebuffer, cx: u32, y: u32, s: &str, scale: u32, fg: Color) {
    let width = s.chars().count() as u32 * font::GLYPH_W * scale;
    let x = cx.saturating_sub(width / 2);
    let off = scale.max(1);
    fb.draw_text_over(x + off, y + off, s, BLACK, scale);
    fb.draw_text_over(x, y, s, fg, scale);
}

/// Draw the whole logo for a given number of formed blades, from clean field.
fn draw_logo(fb: &Framebuffer, l: &Layout, blades: usize) {
    // Repaint the band the logo and its reflection live in, so each step
    // starts from sky. The static title and subtitle sit below this band and
    // are left alone.
    let top = l.cy.saturating_sub(l.ro);
    let bottom = l.cy + l.r + l.gap + l.refl_h;
    field_band(fb, top, bottom + 1);

    glow(fb, l, blades);
    aperture_with(
        fb,
        l.cx as i32,
        l.cy as i32,
        l.r as i32,
        Face::Ramp(SUN),
        Cut::Sky {
            stops: FIELD,
            top: 0,
            height: l.h,
        },
    );
    occlude_unformed(fb, l, blades);
    reflection(fb, l);
}

/// Take the screen and draw the field, the names, and the first (empty) iris.
pub fn begin() {
    let Some(fb) = primary() else { return };
    unsafe {
        *STEP.get() = 0;
        *ACTIVE.get() = true;
    }
    crate::gfx::console::with(|c| c.set_visible(false));

    let (w, h) = (fb.width(), fb.height());
    let l = layout(w, h);

    // The field, once. Everything else repaints over it.
    fb.vgrad(0, 0, w, h, FIELD);

    centred_text(&fb, l.cx, l.title_y, "GLaDOS", l.scale + 1, WHITE);
    centred_text(
        &fb,
        l.cx,
        l.sub_y,
        "a model in the kernel",
        l.scale.max(1),
        Color::new(190, 214, 240),
    );

    render(0, "starting");
}

fn render(step: u32, label: &str) {
    let Some(fb) = primary() else { return };
    let l = layout(fb.width(), fb.height());

    draw_logo(&fb, &l, blades_for(step));

    // The label sits under the names, on a repainted strip so a shorter name
    // does not leave the tail of a longer one behind it.
    field_band(&fb, l.label_y, l.label_y + font::GLYPH_H * l.scale);
    centred_text(
        &fb,
        l.cx,
        l.label_y,
        label,
        l.scale.max(1),
        Color::new(210, 228, 248),
    );
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

/// Change the label without advancing the iris.
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

/// The sun-flare finale: a warm bloom expands from the finished mark to wash
/// the screen, then the desktop takes over. Synchronous and bounded -- it runs
/// once, at the very end of boot, so blocking the boot task for its third of a
/// second costs nothing, and it needs no back buffer because it only ever
/// overwrites toward white.
///
/// This is the cheap form: an expanding `fill_circle`, span-based, no per-pixel
/// reads. The soft beam-and-bloom of the design wants a back buffer to blend
/// against and lands with the continuous-sweep work.
fn flare(fb: &Framebuffer, l: &Layout) {
    let (w, h) = (fb.width(), fb.height());
    let cx = l.cx as i32;
    let cy = l.cy as i32;
    let reach = (w + h) as i32; // past every corner
    let frames = 12i32;
    for f in 1..=frames {
        let rr = l.r as i32 + (reach - l.r as i32) * f / frames;
        fb.fill_circle(cx, cy, rr, BEAM_C);
        crate::time::delay_us(24_000);
    }
}

/// Give the framebuffer back and repaint the boot log over the top.
pub fn finish() {
    if !active() {
        return;
    }
    render(STAGES, "ready");
    // The bloom, then the desktop. The iris is on the wallpaper's own pixels,
    // so once the bloom clears, the mark is still there -- the desktop draws it
    // in place and only the field behind it has changed.
    if let Some(fb) = primary() {
        let l = layout(fb.width(), fb.height());
        flare(&fb, &l);
    }
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

/// Draw one frame of the splash at a given step, to the live framebuffer, for
/// the `splash` shell demo. Not part of boot -- it exists so the animation can
/// be screenshotted under QEMU without timing the real boot sequence.
pub fn demo_frame(step: u32, label: &str) {
    let Some(fb) = primary() else { return };
    let (w, h) = (fb.width(), fb.height());
    let l = layout(w, h);
    fb.vgrad(0, 0, w, h, FIELD);
    centred_text(&fb, l.cx, l.title_y, "GLaDOS", l.scale + 1, WHITE);
    centred_text(
        &fb,
        l.cx,
        l.sub_y,
        "a model in the kernel",
        l.scale.max(1),
        Color::new(190, 214, 240),
    );
    render(step, label);
}

/// The number of steps a full reveal takes, for the demo driver.
pub fn stages() -> u32 {
    STAGES
}
