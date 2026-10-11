//! Composition: draw the whole frame in RAM, show only what changed.
//!
//! The desktop's repaint has always been total -- wallpaper, then every window
//! back to front -- and that stays. Total repaint is what makes the window
//! manager obviously correct: there is no damage arithmetic to get wrong, and
//! the class of "stale pixels from a window that used to be there" cannot
//! exist. What changes is *where* the repaint lands. It used to land in the
//! framebuffer, which meant every update flashed the wallpaper through
//! whatever was on top of it while the frame was being rebuilt. Now it lands
//! in an ordinary heap buffer, and `present` copies to the screen only the
//! spans of pixels that differ from what is already there.
//!
//! Two buffers, not one. `back` is what the desktop composes into. `shadow`
//! is what the screen currently shows. `present` diffs them row by row --
//! plain reads of cached RAM, a memcmp in spirit -- and writes each changed
//! span once. The screen is never read: reads of the aperture are no longer
//! ruinous (it is mapped write-back), but a diff against RAM is faster still
//! and keeps the aperture write-only, which is the discipline everything else
//! here already follows.
//!
//! The mouse made this necessary rather than nice. A pointer produces events
//! at a rate keystrokes never did, and hover feedback means repainting on
//! *motion*. A full repaint per hover change was visible as a blink; a diff
//! per hover change is a few hundred pixels, because that is how much of the
//! frame a highlighted button actually is.
//!
//! The console bypasses `present` through `flush_rect`: its output arrives a
//! cell at a time on the shell's schedule, and waiting for the next desktop
//! draw would make typing invisible. A flushed rectangle is copied to both
//! the screen and `shadow`, so the next `present` sees it as already shown
//! and skips it -- the two paths cannot disagree about what is on screen.
//!
//! If the two 4-8 MB allocations fail, `target()` returns `None` and the
//! desktop draws straight to the framebuffer exactly as it always did. A
//! machine short on heap gets flicker, not a blank screen.

use super::Framebuffer;
use crate::sync::Racy;
use alloc::vec::Vec;

struct Compositor {
    back: Vec<u32>,
    shadow: Vec<u32>,
    w: u32,
    h: u32,
    /// One row of scratch, where a span crossing the pointer is assembled
    /// before it is compared and written. Owned here so a present allocates
    /// nothing.
    row: Vec<u32>,
}

/// The pointer, as a plane over the back buffer rather than pixels in it.
///
/// **This is Haiku's arrangement** (`HWInterface::_DrawCursor`): the back
/// buffer never contains the pointer, and the pointer is blended into each
/// row on its way to the screen. It replaced the cursor being painted straight
/// onto the aperture and erased by copying the back buffer back over it --
/// every move an erase and a redraw on the surface a person is looking at,
/// with the erase reading back a buffer another task might be halfway
/// through composing. That was the strobing arrow. Now a move re-presents
/// two eleven-by-seventeen rectangles, and nothing composes.
///
/// The shadow includes the pointer, because the shadow describes the screen;
/// so a frame that repaints the area under the pointer still compares equal
/// wherever only the pointer was, and writes nothing there.
/// The size of the bitmap pointer the no-compositor path still draws.
pub const CUR_W: u32 = 11;
pub const CUR_H: u32 = 17;

/// The pointer plane's image: premultiplied ARGB, `PLANE` square, laid over
/// each row with OVER rather than copied in. 1.4.1's arrow is anti-aliased
/// with a soft shadow, and both of those are partial alpha -- the old plane
/// held opaque-or-nothing pixels and could carry neither.
pub const PLANE: u32 = 32;
/// Where in the image the pointer's position falls: the arrow's tip sits a
/// little in from the corner so its shadow has room above and to the left.
pub const HOT_X: u32 = 3;
pub const HOT_Y: u32 = 3;
pub type Plane = [u32; (PLANE * PLANE) as usize];

struct Cursor {
    /// The image's top-left on screen, which may be off its left or top edge.
    at: Option<(i32, i32)>,
    image: Plane,
}
static CURSOR: Racy<Cursor> = Racy::new(Cursor { at: None, image: [0; (PLANE * PLANE) as usize] });

/// The screen rectangle the image covers when its top-left is at `(x, y)`.
fn plane_rect(x: i32, y: i32) -> (u32, u32, u32, u32) {
    let (x0, y0) = (x.max(0) as u32, y.max(0) as u32);
    let (x1, y1) = ((x + PLANE as i32).max(0) as u32, (y + PLANE as i32).max(0) as u32);
    (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

/// The pointer is at `at` and looks like `image`. Re-presents where it was
/// and where it is now; the back buffer is untouched. `None` hides it.
pub fn set_cursor(at: Option<(u32, u32)>, image: &Plane) {
    let cur = unsafe { &mut *CURSOR.get() };
    let was = cur.at;
    let now = at.map(|(x, y)| (x as i32 - HOT_X as i32, y as i32 - HOT_Y as i32));
    let changed_image = cur.image != *image;
    if was == now && !changed_image {
        return;
    }
    cur.at = now;
    cur.image = *image;
    if let Some((x, y)) = was {
        let (rx, ry, rw, rh) = plane_rect(x, y);
        flush_counted(rx, ry, rw, rh);
    }
    if let Some((x, y)) = now {
        let (rx, ry, rw, rh) = plane_rect(x, y);
        flush_counted(rx, ry, rw, rh);
    }
}

/// Whether the pointer plane is in use, so the desktop knows not to draw an
/// arrow of its own.
pub fn cursor_plane() -> bool {
    active()
}

static COMP: Racy<Option<Compositor>> = Racy::new(None);

/// Build the buffers. Called once the heap exists; safe to call again.
pub fn init() {
    let Some(fb) = super::primary() else { return };
    if unsafe { (*COMP.get()).is_some() } {
        return;
    }
    let n = (fb.width() as usize) * (fb.height() as usize);
    let mut back = Vec::new();
    let mut shadow = Vec::new();
    if back.try_reserve_exact(n).is_err() || shadow.try_reserve_exact(n).is_err() {
        return;
    }
    back.resize(n, 0);
    // A value `Framebuffer::encode` can never produce -- both formats leave
    // the top byte zero -- so the first present matches nothing and writes
    // every row. Zero would silently skip genuinely black rows and leave
    // whatever the firmware had there.
    shadow.resize(n, 0xFFFF_FFFF);
    unsafe {
        *COMP.get() = Some(Compositor {
            back,
            shadow,
            w: fb.width(),
            h: fb.height(),
            row: alloc::vec![0; fb.width() as usize],
        })
    };
}

/// The buffer the desktop should compose into, dressed as a `Framebuffer`.
///
/// Everything that draws takes `&Framebuffer` and cannot tell the difference,
/// which is the entire trick: the compositor costs no changes to any widget,
/// window or text path. Stride equals width -- heap rows are packed.
pub fn target() -> Option<Framebuffer> {
    unsafe {
        (*COMP.get()).as_mut().map(|c| {
            // Safety: the pointer is a live heap allocation of exactly
            // `w * h` u32s that lives for the kernel's lifetime (the
            // compositor is never dropped), and stride == width.
            Framebuffer::over_ram(
                c.back.as_mut_ptr() as u64,
                c.w,
                c.h,
                c.w,
                super::primary().map(|f| f.format()).unwrap_or(super::Format::Bgrx),
            )
        })
    }
}

/// Copy every changed span from the back buffer to the screen.
pub fn present() {
    // Through the same row path as everything else, so the pointer plane is
    // laid over every row it crosses here too. The comparison against the
    // shadow is per row and cheap where nothing changed.
    let Some((w, h)) = (unsafe { (*COMP.get()).as_ref().map(|c| (c.w, c.h)) }) else { return };
    let wrote = flush_counted(0, 0, w, h);
    super::render::presented(wrote);
}

/// Whether there is a back buffer at all.
/// Forget what is on screen, so the next `present` writes every row.
///
/// For anything that wrote to the framebuffer without going through here --
/// which is exactly what a full-screen program does, since it owns the screen
/// and has no desktop to compose with. The shadow still describes the desktop
/// that was there before, so `present` finds every row unchanged and repaints
/// nothing, and the desktop never comes back.
///
/// This is the same failure the clock had before it was moved onto the
/// compositor, recorded in CLAUDE.md: writing the aperture directly leaves the
/// shadow describing pixels that are no longer there.
///
/// The fill is `0xFFFF_FFFF` for the reason `init` uses it -- `encode` can
/// never produce it, because both pixel formats leave the top byte zero, so
/// nothing can accidentally match.
pub fn invalidate() {
    let Some(c) = (unsafe { (*COMP.get()).as_mut() }) else { return };
    for p in c.shadow.iter_mut() {
        *p = 0xFFFF_FFFF;
    }
}

pub fn active() -> bool {
    unsafe { (*COMP.get()).is_some() }
}

/// Repaint one rectangle from the back buffer whatever the shadow believes.
///
/// `flush_rect` skips rows it thinks are already correct, which is right for
/// everything that paints *through* the compositor and wrong for the one thing
/// that does not: the mouse cursor writes the arrow straight to the aperture,
/// so those pixels differ from the shadow without the shadow knowing. Asking
/// for them back is how the cursor stops being a hole in the compositor's
/// model of the screen.
///
/// Answers `false` when there is no compositor, which is the caller's signal
/// to fall back to its own saved pixels.
/// Write `rows` rows of what is already on screen back to it, from the
/// shadow, through the path `present` uses. Nothing on screen changes; what
/// it is for is timing the aperture (`gfx::perf_report`). Answers the bytes
/// written, nothing when a lease holds the screen.
pub fn rewrite_rows(y0: u32, rows: u32) -> u64 {
    if !super::lease::compositor_may_write() {
        return 0;
    }
    let Some(fb) = super::primary() else { return 0 };
    let Some(c) = (unsafe { (*COMP.get()).as_mut() }) else { return 0 };
    let w = c.w as usize;
    let mut n = 0u64;
    for row in y0..(y0 + rows).min(c.h) {
        let base = row as usize * w;
        fb.blit_span(0, row, &c.shadow[base..base + w]);
        n += w as u64 * 4;
    }
    n
}

pub fn repaint_rect(x: u32, y: u32, rw: u32, rh: u32) -> bool {
    if !super::lease::compositor_may_write() {
        return true;
    }
    let Some(fb) = super::primary() else { return false };
    let Some(c) = (unsafe { (*COMP.get()).as_mut() }) else { return false };
    let w = c.w as usize;
    let x0 = (x as usize).min(w);
    let x1 = ((x + rw) as usize).min(w);
    let y0 = (y as usize).min(c.h as usize);
    let y1 = ((y + rh) as usize).min(c.h as usize);
    if x0 >= x1 {
        return true;
    }
    for row in y0..y1 {
        let base = row * w;
        let back = &c.back[base + x0..base + x1];
        fb.blit_span(x0 as u32, row as u32, back);
        c.shadow[base + x0..base + x1].copy_from_slice(back);
    }
    true
}

/// Copy one rectangle straight through: back buffer to screen and shadow.
///
/// The console's path. Its cells are painted into the back buffer as the
/// shell prints, and this makes each one visible immediately instead of at
/// the next desktop draw.
fn flush_counted(x: u32, y: u32, rw: u32, rh: u32) -> u64 {
    // A program that draws on the aperture itself owns it outright; anything
    // written here would land on its picture. The shadow is left alone too, so
    // it does not claim pixels that never reached the screen -- the lease
    // forgets it on release anyway.
    if !super::lease::compositor_may_write() {
        return 0;
    }
    let Some(fb) = super::primary() else { return 0 };
    let Some(c) = (unsafe { (*COMP.get()).as_mut() }) else { return 0 };
    let mut wrote = 0;
    let w = c.w as usize;
    let x0 = (x as usize).min(w);
    let x1 = ((x + rw) as usize).min(w);
    let y0 = (y as usize).min(c.h as usize);
    let y1 = ((y + rh) as usize).min(c.h as usize);
    let cur = unsafe { &*CURSOR.get() };
    for row in y0..y1 {
        let base = row * w;
        let mut back = &c.back[base + x0..base + x1];
        // A row the pointer crosses is assembled in scratch first: the back
        // buffer's pixels with the pointer's laid over them. That is what the
        // screen should show, so that is what is compared and written.
        if let Some((cx, cy)) = cur.at {
            let r = row as i32 - cy;
            if r >= 0 && r < PLANE as i32 {
                let (cx0, cx1) = (cx, cx + PLANE as i32);
                if cx1 > x0 as i32 && cx0 < x1 as i32 {
                    let scratch = &mut c.row[x0..x1];
                    scratch.copy_from_slice(back);
                    let line = r as usize * PLANE as usize;
                    for px in (cx0.max(x0 as i32) as usize)..(cx1.min(x1 as i32) as usize) {
                        let v = cur.image[line + (px as i32 - cx0) as usize];
                        if v != 0 {
                            scratch[px - x0] = super::blend::over_px(v, scratch[px - x0]);
                        }
                    }
                    back = &c.row[x0..x1];
                }
            }
        }
        let seen = &mut c.shadow[base + x0..base + x1];
        if back == seen {
            continue;
        }
        // The same one-span-per-row rule `present` uses, and for the same
        // reason: a line of console output changes one run of pixels, and
        // writing that run once beats writing each differing pixel on its
        // own. This is the path every typed character takes, so the constant
        // factor here is what typing feels like.
        let a = back.iter().zip(seen.iter()).position(|(b, s)| b != s).unwrap_or(0);
        let b = back.len()
            - back
                .iter()
                .rev()
                .zip(seen.iter().rev())
                .position(|(b, s)| b != s)
                .unwrap_or(0);
        fb.blit_span((x0 + a) as u32, row as u32, &back[a..b]);
        seen[a..b].copy_from_slice(&back[a..b]);
        wrote += 1;
    }
    wrote
}


/// Present only the damaged area: the rows of each rectangle, compared
/// against the shadow and written as one span per row where they differ.
///
/// `present` compares the whole screen, 8 MB at 1080p, to find the handful of
/// rows a hover changed; this compares only where the frame painted. The
/// shadow comparison stays even so, because damage is allowed to be
/// conservative -- a rectangle collapsed to its bounding box, or a window
/// repainted identically -- and the shadow turns over-reporting into no
/// writes rather than into traffic to write-combining memory.
pub fn present_region(r: &super::region::Region) {
    let mut wrote = 0;
    for rect in r.iter() {
        wrote += flush_counted(rect.x, rect.y, rect.w, rect.h);
    }
    super::render::presented(wrote);
}

/// The pointer plane's promises, against the live compositor: moving the
/// pointer writes the screen and the shadow and never the back buffer, and
/// hiding it puts the shadow back to exactly what the back buffer holds.
pub fn cursor_checks(claim: &mut dyn FnMut(&str, bool)) {
    let Some(c) = (unsafe { (*COMP.get()).as_ref() }) else {
        crate::kprintln!("  ....  no compositor, so 4 claim(s) about the pointer plane did not run");
        return;
    };
    if !super::lease::compositor_may_write() {
        crate::kprintln!("  ....  the screen is leased, so 4 claim(s) about the pointer plane did not run");
        return;
    }
    let (w, h) = (c.w as usize, c.h as usize);
    let (x, y) = (w / 3, h / 3);
    let n = PLANE as usize;
    let span = |v: &Vec<u32>| -> Vec<u32> {
        (0..n).flat_map(|r| v[(y + r) * w + x..(y + r) * w + x + n].to_vec()).collect()
    };
    let saved = unsafe { (*CURSOR.get()).at };
    let image_saved = unsafe { (*CURSOR.get()).image };
    let back_before = span(&c.back);
    let mut image = [0u32; (PLANE * PLANE) as usize];
    // One opaque pixel at the image's corner, one half-transparent one beside
    // it: the second is what the plane could not carry before 1.4.1.
    image[0] = 0xFFAB_CDEF;
    image[1] = 0x8000_0000;
    let under = c.back[y * w + x + 1];
    set_cursor(Some(((x + HOT_X as usize) as u32, (y + HOT_Y as usize) as u32)), &image);
    let c = unsafe { (*COMP.get()).as_ref().unwrap() };
    claim("moving the pointer leaves the back buffer exactly as it was", span(&c.back) == back_before);
    claim("and the shadow describes the screen, pointer included", c.shadow[y * w + x] == 0xFFAB_CDEF);
    claim("a half-transparent pointer pixel is blended, not copied",
        c.shadow[y * w + x + 1] == super::blend::over_px(0x8000_0000, under));
    set_cursor(None, &image);
    let c = unsafe { (*COMP.get()).as_ref().unwrap() };
    claim("hiding it puts the shadow back to the back buffer", span(&c.shadow) == span(&c.back));
    // `at` is the image's top-left; the pointer was the hotspot inside it.
    let saved = saved.map(|(x, y)| ((x + HOT_X as i32).max(0) as u32, (y + HOT_Y as i32).max(0) as u32));
    set_cursor(saved, &image_saved);
}
