//! Linear framebuffer.
//!
//! UEFI hands us a physical address, a resolution, a pixel format and a stride.
//! There is no VGA text mode to fall back on -- this machine boots UEFI with no
//! CSM, so there is no INT 10h, no 0xB8000 text buffer, and no VGA BIOS. Pixels
//! are the only output device that exists.

pub mod console;
pub mod font;

use core::ptr::write_volatile;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// The 16-colour palette, in the spirit of the 640x480x4bpp mode TempleOS used.
/// We have 32-bit colour available and are choosing not to use most of it.
pub mod palette {
    use super::Color;
    pub const BLACK: Color = Color::new(0x00, 0x00, 0x00);
    pub const BLUE: Color = Color::new(0x00, 0x00, 0xAA);
    pub const GREEN: Color = Color::new(0x00, 0xAA, 0x00);
    pub const CYAN: Color = Color::new(0x00, 0xAA, 0xAA);
    pub const RED: Color = Color::new(0xAA, 0x00, 0x00);
    pub const MAGENTA: Color = Color::new(0xAA, 0x00, 0xAA);
    pub const BROWN: Color = Color::new(0xAA, 0x55, 0x00);
    pub const LTGRAY: Color = Color::new(0xAA, 0xAA, 0xAA);
    pub const DKGRAY: Color = Color::new(0x55, 0x55, 0x55);
    pub const LTBLUE: Color = Color::new(0x55, 0x55, 0xFF);
    pub const LTGREEN: Color = Color::new(0x55, 0xFF, 0x55);
    pub const LTCYAN: Color = Color::new(0x55, 0xFF, 0xFF);
    pub const LTRED: Color = Color::new(0xFF, 0x55, 0x55);
    pub const LTMAGENTA: Color = Color::new(0xFF, 0x55, 0xFF);
    pub const YELLOW: Color = Color::new(0xFF, 0xFF, 0x55);
    pub const WHITE: Color = Color::new(0xFF, 0xFF, 0xFF);
}

/// Pixel encodings we can drive. `BltOnly` is not here on purpose: it means
/// there is no linear framebuffer at all, and the Blt() call that would be the
/// only way to draw lives in boot services, which we are about to leave.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    /// Byte order in memory: R, G, B, unused.
    Rgbx,
    /// Byte order in memory: B, G, R, unused. OVMF and Intel iGPUs report this.
    Bgrx,
}

#[derive(Clone, Copy)]
pub struct Framebuffer {
    base: *mut u32,
    width: u32,
    height: u32,
    /// Pixels per scan line. Frequently larger than `width`; using `width` as
    /// the stride produces a picture that shears diagonally, which is a useful
    /// thing to be able to recognise on sight.
    stride: u32,
    format: Format,
}

// Single core, ring 0, one owner. There is no other CPU to race with yet.
unsafe impl Send for Framebuffer {}
unsafe impl Sync for Framebuffer {}

impl Framebuffer {
    /// # Safety
    /// `base` must be the physical address of a linear framebuffer of at least
    /// `stride * height * 4` bytes, and identity-mapped and writable.
    pub const unsafe fn new(
        base: u64,
        width: u32,
        height: u32,
        stride: u32,
        format: Format,
    ) -> Self {
        Self { base: base as *mut u32, width, height, stride, format }
    }

    #[inline]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Pixels per scan line, which is often greater than `width()`.
    #[inline]
    pub const fn stride(&self) -> u32 {
        self.stride
    }

    #[inline]
    pub const fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub const fn encode(&self, c: Color) -> u32 {
        // Both layouts put the unused byte in the high position on a
        // little-endian read of the 32-bit word.
        match self.format {
            Format::Rgbx => {
                (c.r as u32) | ((c.g as u32) << 8) | ((c.b as u32) << 16)
            }
            Format::Bgrx => {
                (c.b as u32) | ((c.g as u32) << 8) | ((c.r as u32) << 16)
            }
        }
    }

    #[inline]
    pub fn put(&self, x: u32, y: u32, raw: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let off = (y as usize) * (self.stride as usize) + (x as usize);
        // Volatile: the compiler cannot see that anyone reads this memory, and
        // would happily delete a whole screen-fill loop as dead stores.
        unsafe { write_volatile(self.base.add(off), raw) }
    }

    pub fn fill(&self, c: Color) {
        let raw = self.encode(c);
        for y in 0..self.height {
            let row = (y as usize) * (self.stride as usize);
            for x in 0..self.width {
                unsafe { write_volatile(self.base.add(row + x as usize), raw) }
            }
        }
    }

    pub fn rect(&self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        let raw = self.encode(c);
        for dy in 0..h {
            for dx in 0..w {
                self.put(x + dx, y + dy, raw);
            }
        }
    }

    /// One-pixel outline. Handy as a stride check: if `stride` is wrong the
    /// right-hand edge walks across the screen instead of staying vertical.
    pub fn frame(&self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        let raw = self.encode(c);
        for dx in 0..w {
            self.put(x + dx, y, raw);
            self.put(x + dx, y + h - 1, raw);
        }
        for dy in 0..h {
            self.put(x, y + dy, raw);
            self.put(x + w - 1, y + dy, raw);
        }
    }
}
