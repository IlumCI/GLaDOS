//! Paintbrush. A fixed canvas, five tools, the console's sixteen colours.
//!
//! The canvas is a `Vec<u8>` of palette indices rather than raw pixels:
//! sixteen colours is the 3.1 register this desktop draws in, an index byte
//! is a quarter the memory, and "which colour is this pixel" stays a lookup
//! instead of a reverse search when the fill tool needs to know.
//!
//! Strokes interpolate. The mouse reports positions, not paths -- a quick
//! gesture arrives as points forty pixels apart, and stamping only the
//! points draws a dotted line. Every implementation since the original
//! Paintbrush has drawn the segment between consecutive reports; this one
//! does too, with the classic integer line walk.
//!
//! Everything works from the keyboard as well: arrows move a pen, Enter is
//! the press, tools and colours are letters. Not a courtesy -- serial cannot
//! inject PS/2 packets, so a paint program that only painted by mouse could
//! never be tested by `drive.py`.

use super::console::PALETTE;
use super::theme::{self, Rect};
use super::{DeskApp, Framebuffer};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Canvas size in pixels. Fixed rather than window-sized: a drawing that
/// reflows when the window resizes is not a drawing.
const CW: usize = 440;
const CH: usize = 260;
/// Background palette index: white.
const BG: u8 = 15;

const TOOLS: [(char, &str); 5] = [
    ('p', "Pen"),
    ('e', "Eraser"),
    ('l', "Line"),
    ('r', "Rect"),
    ('f', "Fill"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Pen,
    Eraser,
    Line,
    Rect,
    Fill,
}

pub struct Paint {
    pix: Vec<u8>,
    color: u8,
    tool: Tool,
    /// Brush width in canvas pixels.
    brush: u32,
    /// The keyboard pen. Shares every code path with the mouse.
    pen: (i32, i32),
    pen_down: bool,
    /// Where a line or rectangle started; live preview until release.
    anchor: Option<(i32, i32)>,
    /// The moving end of the pending shape, or the last stamped point.
    last: Option<(i32, i32)>,
    status: String,
}

impl Paint {
    pub fn new() -> Self {
        Self {
            pix: vec![BG; CW * CH],
            color: 0,
            tool: Tool::Pen,
            brush: 2,
            pen: (CW as i32 / 2, CH as i32 / 2),
            pen_down: false,
            anchor: None,
            last: None,
            status: String::from("pen. p e l r f tools, , . colour, s saves"),
        }
    }

    pub fn preferred() -> (u32, u32) {
        (
            CW as u32 + 24 + theme::FRAME * 2,
            CH as u32 + 96 + theme::TITLE_H + theme::FRAME * 2,
        )
    }

    fn canvas_at(client: Rect) -> (i32, i32) {
        ((client.x + 8) as i32, (client.y + 66) as i32)
    }

    fn stamp(&mut self, x: i32, y: i32) {
        let c = if self.tool == Tool::Eraser { BG } else { self.color };
        let r = if self.tool == Tool::Eraser { self.brush + 3 } else { self.brush } as i32;
        for dy in -r / 2..=r / 2 {
            for dx in -r / 2..=r / 2 {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && py >= 0 && (px as usize) < CW && (py as usize) < CH {
                    self.pix[py as usize * CW + px as usize] = c;
                }
            }
        }
    }

    /// Stamp every point of the segment, the integer walk Bresenham described.
    fn stroke(&mut self, a: (i32, i32), b: (i32, i32)) {
        let (mut x, mut y) = a;
        let dx = (b.0 - a.0).abs();
        let dy = -(b.1 - a.1).abs();
        let sx = if a.0 < b.0 { 1 } else { -1 };
        let sy = if a.1 < b.1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.stamp(x, y);
            if x == b.0 && y == b.1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    fn rect_outline(&mut self, a: (i32, i32), b: (i32, i32)) {
        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
        self.stroke((x0, y0), (x1, y0));
        self.stroke((x1, y0), (x1, y1));
        self.stroke((x1, y1), (x0, y1));
        self.stroke((x0, y1), (x0, y0));
    }

    fn fill(&mut self, x: i32, y: i32) {
        if x < 0 || y < 0 || x as usize >= CW || y as usize >= CH {
            return;
        }
        let from = self.pix[y as usize * CW + x as usize];
        if from == self.color {
            return;
        }
        // An explicit stack: the kernel stack is small and a fill of the
        // whole canvas is 114k cells deep if done by recursion.
        let mut stack = vec![(x as usize, y as usize)];
        while let Some((cx, cy)) = stack.pop() {
            if self.pix[cy * CW + cx] != from {
                continue;
            }
            // Walk left and right to the run's ends, then seed above and
            // below -- span filling, which keeps the stack thousands of
            // entries smaller than pushing every neighbour.
            let mut lo = cx;
            while lo > 0 && self.pix[cy * CW + lo - 1] == from {
                lo -= 1;
            }
            let mut hi = cx;
            while hi + 1 < CW && self.pix[cy * CW + hi + 1] == from {
                hi += 1;
            }
            for i in lo..=hi {
                self.pix[cy * CW + i] = self.color;
                for ny in [cy.wrapping_sub(1), cy + 1] {
                    if ny < CH && self.pix[ny * CW + i] == from {
                        stack.push((i, ny));
                    }
                }
            }
        }
    }

    /// One press at canvas coordinates -- the single entry the mouse press,
    /// the drag start, and the keyboard Enter all route through.
    fn press_canvas(&mut self, x: i32, y: i32) {
        match self.tool {
            Tool::Pen | Tool::Eraser => {
                self.stamp(x, y);
                self.last = Some((x, y));
            }
            Tool::Fill => self.fill(x, y),
            Tool::Line | Tool::Rect => match self.anchor {
                // Second press commits -- this is the keyboard path; the
                // mouse commits on release instead.
                Some(a) => {
                    if self.tool == Tool::Line {
                        self.stroke(a, (x, y));
                    } else {
                        self.rect_outline(a, (x, y));
                    }
                    self.anchor = None;
                    self.last = None;
                }
                None => {
                    self.anchor = Some((x, y));
                    self.last = Some((x, y));
                }
            },
        }
    }

    fn tool_char(&self) -> char {
        match self.tool {
            Tool::Pen => 'p',
            Tool::Eraser => 'e',
            Tool::Line => 'l',
            Tool::Rect => 'r',
            Tool::Fill => 'f',
        }
    }

    fn set_tool(&mut self, ch: char) -> bool {
        self.tool = match ch {
            'p' => Tool::Pen,
            'e' => Tool::Eraser,
            'l' => Tool::Line,
            'r' => Tool::Rect,
            'f' => Tool::Fill,
            _ => return false,
        };
        self.anchor = None;
        self.status = format!("{}", TOOLS.iter().find(|(c, _)| *c == ch).map(|(_, n)| *n).unwrap_or("?"));
        true
    }

    /// Write the canvas as a P6 PPM into the namespace.
    ///
    /// PPM because it is the format that needs no library on either side:
    /// a text header and raw RGB. The namespace because that is where files
    /// go here -- content-addressed, snapshotted with everything else.
    fn save(&mut self) {
        let mut out = Vec::with_capacity(CW * CH * 3 + 32);
        out.extend_from_slice(b"P6\n");
        out.extend_from_slice(format!("{} {}\n255\n", CW, CH).as_bytes());
        for &i in &self.pix {
            let c = PALETTE[(i & 0x0F) as usize];
            out.push(c.r);
            out.push(c.g);
            out.push(c.b);
        }
        let n = out.len();
        // tree::put creates missing parents, so /draw needs no ceremony.
        let ok = crate::sysbox::write_blob("/draw/painting.ppm", out);
        self.status = if ok {
            format!("saved /draw/painting.ppm ({} B)", n)
        } else {
            String::from("save failed -- namespace refused the write")
        };
    }
}

/// The toolbar's button rectangles, shared by paint and hit-test.
fn tool_rects(client: Rect) -> Vec<(Rect, char)> {
    let mut x = client.x + 8;
    let y = client.y + 6;
    TOOLS
        .iter()
        .map(|(c, name)| {
            let w = theme::text_w(name.len()) + 14;
            let r = Rect::new(x, y, w, theme::text_h() + 10);
            x += w + 4;
            (r, *c)
        })
        .collect()
}

/// The sixteen swatches, likewise shared.
fn swatch_rects(client: Rect) -> Vec<(Rect, u8)> {
    let s = 22u32;
    let y = client.y + 38;
    (0..16u8)
        .map(|i| (Rect::new(client.x + 8 + i as u32 * (s + 2), y, s, s), i))
        .collect()
}

impl DeskApp for Paint {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);

        for (r, c) in tool_rects(client) {
            let name = TOOLS.iter().find(|(tc, _)| *tc == c).map(|(_, n)| *n).unwrap_or("?");
            theme::button(fb, r, name, false, c == self.tool_char());
        }
        // Brush width readout beside the tools.
        let tr = tool_rects(client);
        if let Some((last, _)) = tr.last() {
            let bx = last.x + last.w + 12;
            let label = format!("brush {}  ([ ] resize)", self.brush);
            theme::text(fb, bx, client.y + 10, &label, theme::TEXT, theme::FACE);
        }

        for (r, i) in swatch_rects(client) {
            fb.rect(r.x, r.y, r.w, r.h, PALETTE[i as usize]);
            theme::bevel(fb, r, i != self.color);
            if i == self.color {
                fb.frame(r.x, r.y, r.w, r.h, theme::TEXT);
            }
        }

        // The canvas, in a sunken well like every editable surface here.
        let (cx, cy) = Self::canvas_at(client);
        let well = Rect::new(cx as u32 - 3, cy as u32 - 3, CW as u32 + 6, CH as u32 + 6);
        theme::well(fb, well, theme::HILIGHT);
        // Rows of same-colour pixels become single rect calls; a canvas is
        // mostly runs, and per-pixel put would repaint 114k cells per frame.
        for y in 0..CH {
            let mut x = 0;
            while x < CW {
                let c = self.pix[y * CW + x];
                let mut run = 1;
                while x + run < CW && self.pix[y * CW + x + run] == c {
                    run += 1;
                }
                fb.rect(
                    (cx as usize + x) as u32,
                    (cy as usize + y) as u32,
                    run as u32,
                    1,
                    PALETTE[(c & 0x0F) as usize],
                );
                x += run;
            }
        }

        // Pending line or rectangle, previewed over the canvas without
        // touching it -- committing is what writes pixels.
        if let (Some(a), Some(b)) = (self.anchor, self.last) {
            let col = PALETTE[(self.color & 0x0F) as usize];
            let clip = |p: (i32, i32)| {
                (
                    (cx + p.0.clamp(0, CW as i32 - 1)) as u32,
                    (cy + p.1.clamp(0, CH as i32 - 1)) as u32,
                )
            };
            let (pa, pb) = (clip(a), clip(b));
            match self.tool {
                Tool::Line => fb.line(pa.0 as i32, pa.1 as i32, pb.0 as i32, pb.1 as i32, col),
                Tool::Rect => {
                    let (x0, x1) = (pa.0.min(pb.0), pa.0.max(pb.0));
                    let (y0, y1) = (pa.1.min(pb.1), pa.1.max(pb.1));
                    fb.frame(x0, y0, x1 - x0 + 1, y1 - y0 + 1, col);
                }
                _ => {}
            }
        }

        // The keyboard pen, a crosshair. Only while the window has the
        // keyboard, so a mouse-driven session is not haunted by it.
        if focused {
            let px = cx + self.pen.0;
            let py = cy + self.pen.1;
            let col = if self.pen_down { theme::APERTURE } else { theme::SHADOW };
            fb.rect((px - 5).max(0) as u32, py as u32, 11, 1, col);
            fb.rect(px as u32, (py - 5).max(0) as u32, 1, 11, col);
        }

        let sy = cy as u32 + CH as u32 + 8;
        if sy + theme::text_h() < client.y + client.h {
            theme::text(fb, client.x + 8, sy, &self.status, theme::TEXT, theme::FACE);
        }
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd;
        let step = 3;
        match k {
            kbd::KEY_LEFT => self.pen.0 = (self.pen.0 - step).max(0),
            kbd::KEY_RIGHT => self.pen.0 = (self.pen.0 + step).min(CW as i32 - 1),
            kbd::KEY_UP => self.pen.1 = (self.pen.1 - step).max(0),
            kbd::KEY_DOWN => self.pen.1 = (self.pen.1 + step).min(CH as i32 - 1),
            b'\n' | b'\r' => {
                // For the pen, Enter toggles drawing-while-moving; for every
                // other tool it is the press itself.
                if self.tool == Tool::Pen || self.tool == Tool::Eraser {
                    self.pen_down = !self.pen_down;
                    self.status = String::from(if self.pen_down { "pen down" } else { "pen up" });
                    if self.pen_down {
                        let p = self.pen;
                        self.stamp(p.0, p.1);
                    }
                } else {
                    let p = self.pen;
                    self.press_canvas(p.0, p.1);
                }
            }
            b',' => {
                self.color = (self.color + 15) & 0x0F;
                self.status = format!("colour {}", self.color);
            }
            b'.' => {
                self.color = (self.color + 1) & 0x0F;
                self.status = format!("colour {}", self.color);
            }
            b'[' => self.brush = self.brush.saturating_sub(1).max(1),
            b']' => self.brush = (self.brush + 1).min(9),
            b'x' | b'X' => {
                self.pix.fill(BG);
                self.status = String::from("cleared");
            }
            b's' | b'S' => self.save(),
            c if self.set_tool((c as char).to_ascii_lowercase()) => {}
            _ => return false,
        }
        // Keyboard pen draws on the move while down, matching a held button.
        if self.pen_down && matches!(k, kbd::KEY_LEFT | kbd::KEY_RIGHT | kbd::KEY_UP | kbd::KEY_DOWN) {
            let p = self.pen;
            match self.last {
                Some(l) => self.stroke(l, p),
                None => self.stamp(p.0, p.1),
            }
            self.last = Some(p);
        } else if !self.pen_down && matches!(k, kbd::KEY_LEFT | kbd::KEY_RIGHT | kbd::KEY_UP | kbd::KEY_DOWN) {
            self.last = None;
        }
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        for (r, c) in tool_rects(client) {
            if x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32 {
                self.set_tool(c);
                return true;
            }
        }
        for (r, i) in swatch_rects(client) {
            if x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32 {
                self.color = i;
                self.status = format!("colour {}", i);
                return true;
            }
        }
        let (cx, cy) = Self::canvas_at(client);
        let (px, py) = (x - cx, y - cy);
        if px >= 0 && py >= 0 && (px as usize) < CW && (py as usize) < CH {
            self.pen = (px, py);
            match self.tool {
                // The mouse's line and rect anchor on press and commit on
                // release; the keyboard's commit on the second Enter. Same
                // press_canvas, different second half.
                Tool::Line | Tool::Rect => {
                    self.anchor = Some((px, py));
                    self.last = Some((px, py));
                }
                _ => self.press_canvas(px, py),
            }
            return true;
        }
        false
    }

    fn drag(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let (cx, cy) = Self::canvas_at(client);
        let p = (
            (x - cx).clamp(0, CW as i32 - 1),
            (y - cy).clamp(0, CH as i32 - 1),
        );
        self.pen = p;
        match self.tool {
            Tool::Pen | Tool::Eraser => {
                match self.last {
                    Some(l) => self.stroke(l, p),
                    None => self.stamp(p.0, p.1),
                }
                self.last = Some(p);
                true
            }
            Tool::Line | Tool::Rect if self.anchor.is_some() => {
                self.last = Some(p);
                true
            }
            _ => false,
        }
    }

    fn release(&mut self) -> bool {
        match (self.tool, self.anchor, self.last) {
            (Tool::Line, Some(a), Some(b)) => {
                self.stroke(a, b);
                self.anchor = None;
                self.last = None;
                true
            }
            (Tool::Rect, Some(a), Some(b)) => {
                self.rect_outline(a, b);
                self.anchor = None;
                self.last = None;
                true
            }
            _ => {
                self.last = None;
                false
            }
        }
    }
}
