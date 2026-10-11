//! Which parts of the screen changed: a small set of disjoint rectangles.
//!
//! Damage is what lets a frame cost what changed rather than what exists. A
//! hover that recolours one caption button should repaint that button's
//! neighbourhood and not 2 million pixels of desktop, and the only way to know
//! the difference is to carry the changed area from where it is discovered to
//! where pixels are written.
//!
//! **Disjoint, inline and capped**, after looking at what other compositors
//! keep. pixman's banded region (X11, Weston, wlroots) is exact for any shape
//! and costs a heap allocation and a careful sweep; Linux's fbdev helper keeps
//! one bounding box and repaints everything between a change in one corner and
//! a change in the other. This sits between them, as SerenityOS's
//! `DisjointRectSet` does: up to `CAP` non-overlapping rectangles in a fixed
//! array, and past that one bounding box -- wlroots makes the same collapse at
//! 20 (`WLR_DAMAGE_RING_MAX_RECTS`). Collapsing can only ever repaint *more*
//! than changed, never less, which is the safe direction to be wrong in.
//!
//! Disjoint because each rectangle is repainted on its own: two that overlap
//! would paint their intersection twice, which is wasted work and, for a
//! translucent layer blended over what is already there, a wrong answer.

use super::theme::Rect;

/// How many rectangles before the region becomes its bounding box.
pub const CAP: usize = 16;

#[derive(Clone, Copy)]
pub struct Region {
    n: usize,
    r: [Rect; CAP],
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

/// `a` minus `b`, as at most four rectangles: the bands above and below `b`,
/// and the parts left and right of it in between.
fn subtract(a: Rect, b: Rect, out: &mut [Rect; 4]) -> usize {
    let (ax1, ay1) = (a.x + a.w, a.y + a.h);
    let (bx1, by1) = (b.x + b.w, b.y + b.h);
    if b.x >= ax1 || b.y >= ay1 || bx1 <= a.x || by1 <= a.y {
        out[0] = a;
        return 1;
    }
    let mut n = 0;
    let mut push = |r: Rect| {
        if !r.is_empty() {
            out[n] = r;
            n += 1;
        }
    };
    let top = b.y.max(a.y);
    let bot = by1.min(ay1);
    push(Rect::new(a.x, a.y, a.w, top - a.y));
    push(Rect::new(a.x, bot, a.w, ay1 - bot));
    let left = b.x.max(a.x);
    let right = bx1.min(ax1);
    push(Rect::new(a.x, top, left - a.x, bot - top));
    push(Rect::new(right, top, ax1 - right, bot - top));
    n
}

fn union_box(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// The overlap of two rectangles, empty when there is none.
pub fn intersect(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    if x1 <= x0 || y1 <= y0 {
        return Rect::new(0, 0, 0, 0);
    }
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

impl Region {
    pub const fn new() -> Self {
        Region { n: 0, r: [Rect::new(0, 0, 0, 0); CAP] }
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn iter(&self) -> impl Iterator<Item = Rect> + '_ {
        self.r[..self.n].iter().copied()
    }

    pub fn clear(&mut self) {
        self.n = 0;
    }

    /// The smallest rectangle containing all of it.
    pub fn bbox(&self) -> Option<Rect> {
        self.iter().reduce(union_box)
    }

    /// Total pixels covered. Exact, because the rectangles never overlap.
    pub fn area(&self) -> u64 {
        self.iter().map(|r| r.w as u64 * r.h as u64).sum()
    }

    /// Add a rectangle. Only the part not already covered is stored, so the
    /// set stays disjoint; past `CAP` it becomes its bounding box.
    pub fn add(&mut self, r: Rect) {
        if r.is_empty() {
            return;
        }
        // The pieces of `r` not yet covered, refined against each existing
        // rectangle in turn. Each subtraction yields at most four pieces, so
        // this is bounded; if it ever needs more room than the cap the region
        // simply collapses.
        let mut pieces = [Rect::new(0, 0, 0, 0); 4 * CAP];
        let mut np = 1;
        pieces[0] = r;
        for k in 0..self.n {
            let have = self.r[k];
            let mut next = [Rect::new(0, 0, 0, 0); 4 * CAP];
            let mut nn = 0;
            for p in &pieces[..np] {
                let mut out = [Rect::new(0, 0, 0, 0); 4];
                let m = subtract(*p, have, &mut out);
                for q in &out[..m] {
                    if nn == next.len() {
                        self.collapse_with(r);
                        return;
                    }
                    next[nn] = *q;
                    nn += 1;
                }
            }
            pieces = next;
            np = nn;
            if np == 0 {
                return;
            }
        }
        if self.n + np > CAP {
            self.collapse_with(r);
            return;
        }
        for p in &pieces[..np] {
            self.r[self.n] = *p;
            self.n += 1;
        }
    }

    /// Everything so far, and `r`, as one bounding box.
    fn collapse_with(&mut self, r: Rect) {
        let b = self.bbox().map_or(r, |b| union_box(b, r));
        self.n = 1;
        self.r[0] = b;
    }

    /// Keep only the part inside `bounds` -- the screen, usually, since a
    /// window dragged half off it damages pixels that do not exist.
    pub fn clip_to(&mut self, bounds: Rect) {
        let mut k = 0;
        for i in 0..self.n {
            let c = intersect(self.r[i], bounds);
            if !c.is_empty() {
                self.r[k] = c;
                k += 1;
            }
        }
        self.n = k;
    }
}

/// `diag region`: the properties every repaint rests on, on synthetic input.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let disjoint = |g: &Region| {
        let v: alloc::vec::Vec<Rect> = g.iter().collect();
        (0..v.len()).all(|i| (i + 1..v.len()).all(|j| intersect(v[i], v[j]).is_empty()))
    };

    let mut g = Region::new();
    claim("a new region is empty", g.is_empty() && g.area() == 0);
    g.add(Rect::new(10, 10, 20, 20));
    claim("one rectangle is held as itself", g.len() == 1 && g.area() == 400);
    g.add(Rect::new(10, 10, 20, 20));
    claim("adding it again changes nothing", g.len() == 1 && g.area() == 400);
    g.add(Rect::new(15, 15, 5, 5));
    claim("a rectangle already covered adds nothing", g.area() == 400);
    g.add(Rect::new(20, 20, 20, 20));
    claim("an overlapping one adds only what was not covered",
          g.area() == 400 + 400 - 100 && disjoint(&g));
    let mut h = Region::new();
    h.add(Rect::new(0, 0, 100, 100));
    h.add(Rect::new(40, 40, 10, 10));
    claim("a rectangle inside a larger one is absorbed", h.len() == 1 && h.area() == 10_000);
    claim("empty rectangles are ignored", {
        let mut e = Region::new();
        e.add(Rect::new(5, 5, 0, 9));
        e.is_empty()
    });

    let mut many = Region::new();
    for i in 0..CAP as u32 {
        many.add(Rect::new(i * 10, 0, 5, 5));
    }
    claim("the cap holds exactly CAP separate rectangles", many.len() == CAP && disjoint(&many));
    many.add(Rect::new(500, 500, 5, 5));
    claim("one more collapses to the bounding box", many.len() == 1
          && many.bbox() == Some(Rect::new(0, 0, 505, 505)));

    let mut c = Region::new();
    c.add(Rect::new(1900, 1000, 100, 100));
    c.clip_to(Rect::new(0, 0, 1920, 1080));
    claim("clipping to the screen keeps only the visible part",
          c.bbox() == Some(Rect::new(1900, 1000, 20, 80)));
    let mut cross = Region::new();
    cross.add(Rect::new(0, 10, 100, 10));
    cross.add(Rect::new(45, 0, 10, 100));
    claim("a cross stays exact and disjoint",
          cross.area() == 1000 + 1000 - 100 && disjoint(&cross));
    ok
}
