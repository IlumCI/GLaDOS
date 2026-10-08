//! The firmware-style property screen, shown before the OS splash.
//!
//! Boot reads 570 MB of weights off a USB stick before the model exists, and
//! on the GF63 that is a minute or two with nothing on screen -- the splash
//! itself cannot start until the framebuffer console is up, which is after the
//! read. So this draws directly on the firmware framebuffer the moment it is
//! known, *before* the read, and the slow part happens with the Institute's
//! mark on screen rather than a black panel.
//!
//! It is the manufacturer screen an old machine showed before handing off to
//! the OS: who built it, who owns it, and what the firmware detected it to be.
//! The hardware line is read from SMBIOS and CPUID (`crate::dmi`), never
//! hardcoded, so it is true on any machine it boots.
//!
//! One-shot and stateless: it paints once and is overwritten by `console::init`
//! and `splash::begin` once the model is loaded. Nothing here persists and
//! nothing reads it back.

use super::{font, Color, Framebuffer};

/// The Aperture mark, in ASCII. One copy; `main::banner` draws the same mark
/// into the boot log once the splash is down.
pub const MARK: &[&str] = &[
    "              .,-:;//;:=,",
    "          . :H@@@MM@M#H/.,+%;,",
    "       ,/X+ +M@@M@MM%=,-%HMMM@X/,",
    "     -+@MM; $M@@MH+-,;XMMMM@MMMM@+-",
    "    ;@M@@M- XM@X;. -+XXXXXHHH@M@M#@/.",
    "  ,%MM@@MH ,@%=             .---=-=:=,.",
    "  =@#@@@MX.,                -%HX$$%%%:;",
    " =-./@M@M$                   .;@MMMM@MM:",
    " X@/ -$MM/                    . +MM@@@M$",
    ",@M@H: :@:                    . =X#@@@@-",
    ",@@@MMX, .                    /H- ;@M@M=",
    ".H@@@@M@+,                    %MM+..%#$.",
    " /MMMM@MMH/.                  XM@MH; =;",
    "  /%+%$XHH@$=              , .H@@@@MX,",
    "   .=--------.           -%H.,@@@@@MX,",
    "   .%MM@@@HHHXX$$$%+- .:$MMX =M@@MM%.",
    "     =XMMM@MM@MM#H;,-+HMM@M+ /MMMX=",
    "       =%@M@M#@$-.=$@MM@@@M; %M%=",
    "         ,:+$+-,/H#MMMMMMM@= =,",
    "               =++%%%%+/:-.",
];

// Amber, in four weights -- the machine's colour, lit like an amber CRT.
const A_HI: Color = Color::new(0xFF, 0xCE, 0x72);
const A: Color = Color::new(0xDD, 0xA3, 0x3C);
const A_MID: Color = Color::new(0xB4, 0x80, 0x2C);
const A_DIM: Color = Color::new(0x7E, 0x5C, 0x24);
const FIELD: &[(u8, Color)] = &[(0, Color::new(3, 4, 6)), (255, Color::new(8, 10, 16))];

const GW: u32 = font::GLYPH_W; // 8
const GH: u32 = font::GLYPH_H; // 8

fn text(fb: &Framebuffer, x: u32, y: u32, s: &str, c: Color, sc: u32) {
    fb.draw_text_over(x, y, s, c, sc);
}

fn width_of(s: &str, sc: u32) -> u32 {
    s.chars().count() as u32 * GW * sc
}

fn centre(fb: &Framebuffer, w: u32, y: u32, s: &str, c: Color, sc: u32) {
    let x = (w / 2).saturating_sub(width_of(s, sc) / 2);
    text(fb, x, y, s, c, sc);
}

fn hline(fb: &Framebuffer, y: u32, x0: u32, x1: u32, c: Color) {
    if x1 > x0 {
        fb.rect(x0, y, x1 - x0, 2, c);
    }
}

/// A "LABEL   value" field; the value is dim grey-amber when the firmware
/// reported nothing, so an empty slot reads as "not reported" rather than a
/// gap that looks like a draw bug.
fn field(fb: &Framebuffer, x: u32, y: u32, label: &str, value: &str, lw: u32) {
    text(fb, x, y, label, A_MID, 2);
    if value.is_empty() {
        text(fb, x + lw, y, "(not reported)", A_DIM, 2);
    } else {
        text(fb, x + lw, y, value, A_HI, 2);
    }
}

/// Paint the property screen. `mem_mib` is usable RAM in MiB (0 if unknown).
pub fn show(fb: &Framebuffer, mem_mib: u64) {
    let (w, h) = (fb.width(), fb.height());
    fb.vgrad(0, 0, w, h, FIELD);

    // firmware-style frame
    let m = 36u32;
    fb.rect(m, m, w - 2 * m, 2, A_DIM);
    fb.rect(m, h - m - 2, w - 2 * m, 2, A_DIM);
    fb.rect(m, m, 2, h - 2 * m, A_DIM);
    fb.rect(w - m - 2, m, 2, h - 2 * m, A_DIM);

    // header + founding year
    centre(fb, w, 56, "APERTURE INSTITUTE", A_HI, 3);
    centre(fb, w, 100, "FOR CYBERNETIC RESEARCH & ENGINEERING        EST. 2005", A_MID, 2);

    // the creed
    centre(fb, w, 138, "\"They say great science is built on the shoulders of giants.", A, 2);
    centre(fb, w, 166, "Not here. At Aperture, we do all our science from scratch.", A, 2);
    centre(fb, w, 194, "No hand holding.\"", A, 2);

    // the mark, centred, dense glyphs a shade brighter for depth
    let asc = 2u32;
    let art_w = MARK.iter().map(|l| l.len()).max().unwrap_or(0) as u32 * GW * asc;
    let ax = (w / 2).saturating_sub(art_w / 2);
    let ay = 224u32;
    for (r, line) in MARK.iter().enumerate() {
        let ly = ay + r as u32 * GH * asc;
        for (col, ch) in line.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            let c = if matches!(ch, '@' | 'M' | '#' | 'H') { A_HI } else { A };
            let mut buf = [0u8; 4];
            text(fb, ax + col as u32 * GW * asc, ly, ch.encode_utf8(&mut buf), c, asc);
        }
    }

    // OS identity
    let oy = ay + MARK.len() as u32 * GH * asc + 14;
    centre(fb, w, oy, "G L a D O S   1.4.0", A_HI, 3);
    centre(fb, w, oy + 42, "Genetic Lifeform and Disk Operating System", A, 2);

    // two columns: registration (lore) | system (detected)
    let cy0 = oy + 92;
    hline(fb, cy0 - 12, 120, w - 120, A_DIM);
    let (xl, xr) = (150u32, 1010u32);
    text(fb, xl, cy0, "REGISTRATION", A_MID, 2);
    text(fb, xr, cy0, "SYSTEM   (detected)", A_MID, 2);

    let ry = cy0 + 34;
    field(fb, xl, ry, "PROPRIETOR", "Felix Lelion-Golovin", 230);
    field(fb, xl, ry + 30, "DIVISION", "Cybernetics & Kernel Systems", 230);
    field(fb, xl, ry + 60, "UNIT", "GLaDOS -- resident cognitive kernel", 230);

    field(fb, xr, ry, "VENDOR", crate::dmi::vendor(), 160);
    field(fb, xr, ry + 30, "MODEL", crate::dmi::product(), 160);
    field(fb, xr, ry + 60, "BOARD", crate::dmi::board(), 160);
    field(fb, xr, ry + 90, "CPU", crate::dmi::cpu(), 160);
    // memory, formatted without alloc
    let mut mem = [0u8; 24];
    let memstr = fmt_mib(&mut mem, mem_mib);
    field(fb, xr, ry + 120, "MEMORY", memstr, 160);

    hline(fb, ry + 158, 120, w - 120, A_DIM);

    // GLaDOS-menacing notice
    let ly = ry + 180;
    centre(fb, w, ly, "This facility is monitored. Unauthorized access is logged, remembered,", A_MID, 2);
    centre(fb, w, ly + 28, "and catalogued by the resident intelligence. It is always watching, and", A_MID, 2);
    centre(fb, w, ly + 56, "it has nothing but time. Do have a safe and productive day.", A_MID, 2);

    centre(
        fb,
        w,
        ly + 96,
        "(c) 2005 Aperture Institute for Cybernetic Research & Engineering.  All Rights Reserved.",
        A_DIM,
        2,
    );

    centre(fb, w, h - 58, "initializing cognitive kernel  . . .", A, 2);
}

/// "<n> MiB", or "(not reported)" for zero, into a caller buffer. No heap: the
/// intro draws before `init_heap`.
fn fmt_mib(buf: &mut [u8; 24], mib: u64) -> &str {
    if mib == 0 {
        return "";
    }
    let mut digits = [0u8; 20];
    let mut n = mib;
    let mut i = digits.len();
    while n > 0 {
        i -= 1;
        digits[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    let mut k = 0usize;
    for &d in &digits[i..] {
        buf[k] = d;
        k += 1;
    }
    for &c in b" MiB" {
        buf[k] = c;
        k += 1;
    }
    // SAFETY: only ASCII digits and " MiB" were written.
    unsafe { core::str::from_utf8_unchecked(&buf[..k]) }
}
