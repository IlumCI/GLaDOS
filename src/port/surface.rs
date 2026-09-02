//! An 8-bit indexed frame, put on a 32-bit screen.
//!
//! Every game of DOOM's generation draws into a byte per pixel and a palette
//! of 256 colours, because that is what the hardware was. This machine's
//! framebuffer is 32bpp and has no palettized mode and no mode-setting -- the
//! resolution and format are whatever UEFI handed over before
//! `ExitBootServices` -- so the palette has to be applied in software, once
//! per pixel, every frame.
//!
//! Which sounds expensive and is not, because of where the work goes:
//!
//!   * The palette is **pre-encoded once** into the framebuffer's own word
//!     order (`Rgbx` or `Bgrx`) when it is set. Per pixel it is one array
//!     index, not a colour conversion.
//!   * A row is expanded into a `u32` scratch and then **blitted whole**.
//!     `Framebuffer::blit_span` is a `copy_from_slice` on a RAM surface and a
//!     single `copy_nonoverlapping` on the aperture -- the fastest path in the
//!     tree, and the one the compositor itself uses.
//!   * Integer scaling repeats that row rather than rebuilding it. At 320x200
//!     on a 1920x1080 screen the scale is 5, so one built row is blitted five
//!     times.
//!
//! `src/gfx/paint.rs` has the other approach -- scan for a run of equal bytes
//! and emit a `rect` per run -- and it is the slow shape. It is right for a
//! drawing program, where runs are long and frames are rare. It is wrong for a
//! rendered scene, where every pixel differs from its neighbour.

use alloc::vec::Vec;

use crate::gfx::Framebuffer;

/// An indexed frame and the palette it is read through.
pub struct Surface {
    w: usize,
    h: usize,
    /// One byte per pixel, `w * h`. This is what the program draws into.
    pix: Vec<u8>,
    /// The palette, already in the screen's word order.
    pal: [u32; 256],
    /// One expanded output row, kept between frames.
    row: Vec<u32>,
    /// Integer magnification, and where the result sits.
    scale: u32,
    ox: u32,
    oy: u32,
}

impl Surface {
    /// A frame of `w` by `h` indexed pixels, laid out for the current screen.
    ///
    /// Answers `None` when there is no framebuffer or when the screen cannot
    /// hold even one pixel per pixel -- which is not a case that arises on any
    /// machine this runs on, and is still not a reason to divide by zero.
    pub fn new(w: usize, h: usize) -> Option<Surface> {
        let fb = crate::gfx::primary()?;
        if w == 0 || h == 0 || fb.width() < w as u32 || fb.height() < h as u32 {
            return None;
        }
        let mut s = Surface {
            w,
            h,
            pix: Vec::new(),
            pal: [0; 256],
            row: Vec::new(),
            scale: 1,
            ox: 0,
            oy: 0,
        };
        super::sized(&mut s.pix, w * h);
        s.fit(&fb);
        Some(s)
    }

    /// The largest whole magnification that fits, centred.
    ///
    /// Whole numbers only. A 320-wide frame stretched to 1920 by a
    /// non-integer factor is a frame where some source pixels are two output
    /// pixels wide and some are three, which on art drawn pixel by pixel reads
    /// as a texture that ripples -- and this is a machine whose entire
    /// interface is an 8x8 bitmap font for that reason.
    fn fit(&mut self, fb: &Framebuffer) {
        let sx = fb.width() / self.w as u32;
        let sy = fb.height() / self.h as u32;
        self.scale = sx.min(sy).max(1);
        let (dw, dh) = (self.w as u32 * self.scale, self.h as u32 * self.scale);
        self.ox = (fb.width().saturating_sub(dw)) / 2;
        self.oy = (fb.height().saturating_sub(dh)) / 2;
        super::sized(&mut self.row, dw as usize);
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    /// Where the frame lands on screen: origin and magnification. For anything
    /// that needs to map a pointer back into frame coordinates.
    pub fn placement(&self) -> (u32, u32, u32) {
        (self.ox, self.oy, self.scale)
    }

    /// The pixels, to draw into. `w * h` bytes, row-major, top row first.
    pub fn pixels(&mut self) -> &mut [u8] {
        &mut self.pix
    }

    /// Set the palette from 768 bytes of RGB, which is the form every WAD and
    /// every PCX of the era stores it in.
    ///
    /// Short input is honoured as far as it goes rather than refused: a
    /// truncated palette gives wrong colours, and refusing gives no picture at
    /// all, which is harder to diagnose from a screenshot.
    pub fn set_palette_rgb(&mut self, rgb: &[u8]) {
        let Some(fb) = crate::gfx::primary() else { return };
        for i in 0..256 {
            let j = i * 3;
            let (r, g, b) = match (rgb.get(j), rgb.get(j + 1), rgb.get(j + 2)) {
                (Some(r), Some(g), Some(b)) => (*r, *g, *b),
                _ => (0, 0, 0),
            };
            self.pal[i] = fb.encode(crate::gfx::Color::new(r, g, b));
        }
    }

    /// Set one entry, for anything that builds a palette rather than loading
    /// one.
    pub fn set_colour(&mut self, i: u8, c: crate::gfx::Color) {
        if let Some(fb) = crate::gfx::primary() {
            self.pal[i as usize] = fb.encode(c);
        }
    }

    /// Put the frame on the screen.
    ///
    /// Straight to the framebuffer, not through the compositor. A full-screen
    /// program owns the screen for its duration -- that is what
    /// `port::with_screen` establishes -- so there is nothing to compose it
    /// with and nothing to diff it against. `desk::draw()` on the way out is
    /// what puts the desktop back, and it recomposes from scratch.
    pub fn present(&mut self) {
        let Some(fb) = crate::gfx::primary() else { return };
        // The screen can change under a long-running program -- not today,
        // since there is no mode-setting, but the cost of asking is one
        // comparison and the cost of being wrong is writing past the end.
        if fb.width() < self.ox + self.w as u32 * self.scale
            || fb.height() < self.oy + self.h as u32 * self.scale
        {
            self.fit(&fb);
        }
        let s = self.scale as usize;
        let dw = self.w * s;
        for y in 0..self.h {
            let src = &self.pix[y * self.w..(y + 1) * self.w];
            // Build the row once...
            for (x, p) in src.iter().enumerate() {
                let c = self.pal[*p as usize];
                for k in 0..s {
                    self.row[x * s + k] = c;
                }
            }
            // ...and lay it down `scale` times. The alternative -- rebuilding
            // it per output row -- is `scale` times the expansion work for
            // identical bytes.
            let base = self.oy + (y * s) as u32;
            for k in 0..s {
                fb.blit_span(self.ox, base + k as u32, &self.row[..dw]);
            }
        }
    }

    /// Fill the whole frame with one index. Cheaper than writing it per pixel
    /// from outside, and the thing a renderer does between levels.
    pub fn clear(&mut self, index: u8) {
        for p in self.pix.iter_mut() {
            *p = index;
        }
    }
}
