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
        *COMP.get() = Some(Compositor { back, shadow, w: fb.width(), h: fb.height() })
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
    let Some(fb) = super::primary() else { return };
    let Some(c) = (unsafe { (*COMP.get()).as_mut() }) else { return };
    let w = c.w as usize;
    for y in 0..c.h as usize {
        let row = &c.back[y * w..(y + 1) * w];
        let seen = &mut c.shadow[y * w..(y + 1) * w];
        if row == seen {
            continue;
        }
        // One contiguous span per row, from the first difference to the last.
        // A row usually changes in one place -- a button, a run of text -- and
        // finding multiple spans would spend more time comparing than the
        // extra writes cost.
        let a = row.iter().zip(seen.iter()).position(|(b, s)| b != s).unwrap_or(0);
        let b = w - row
            .iter()
            .rev()
            .zip(seen.iter().rev())
            .position(|(b, s)| b != s)
            .unwrap_or(0);
        fb.blit_span(a as u32, y as u32, &row[a..b]);
        seen[a..b].copy_from_slice(&row[a..b]);
    }
}

/// Copy one rectangle straight through: back buffer to screen and shadow.
///
/// The console's path. Its cells are painted into the back buffer as the
/// shell prints, and this makes each one visible immediately instead of at
/// the next desktop draw.
pub fn flush_rect(x: u32, y: u32, rw: u32, rh: u32) {
    let Some(fb) = super::primary() else { return };
    let Some(c) = (unsafe { (*COMP.get()).as_mut() }) else { return };
    let w = c.w as usize;
    let x0 = (x as usize).min(w);
    let x1 = ((x + rw) as usize).min(w);
    let y0 = (y as usize).min(c.h as usize);
    let y1 = ((y + rh) as usize).min(c.h as usize);
    for row in y0..y1 {
        let base = row * w;
        let back = &c.back[base + x0..base + x1];
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
    }
}
