//! Dwindle tiling: a binary tree of splits whose leaves are windows.
//!
//! Hyprland's default layout, and the reason it can be pure. A new window
//! takes half of the window it was opened from -- split along that window's
//! longer side, so a wide screen fills left to right and a tall slot top to
//! bottom -- and closing a window gives its half back to its sibling. The
//! tree is the whole of the layout's state; where every window goes is a
//! function of it and the work area, recomputed whenever either changes,
//! so there is no stored rectangle anywhere that could disagree with it.
//!
//! Windows are named by serial, which is how the scene and the window images
//! already name them: an index shifts whenever a window below it closes.
//!
//! **The split direction is decided at layout, not at insertion**, which is
//! dwindle's own rule (`preserve_split` off): a split made side by side in a
//! wide slot turns top-to-bottom if its slot becomes tall. `toggle` pins one
//! instead, which is the only way a split ever stops following its slot.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::theme::Rect;

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Leaf(u64),
    Split {
        /// Pinned direction: `Some(true)` side by side, `Some(false)` one
        /// over the other, `None` by the slot's shape.
        pin: Option<bool>,
        /// How much of the slot the first child takes, in thousandths.
        ratio: u16,
        a: Box<Node>,
        b: Box<Node>,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tree {
    root: Option<Node>,
}

/// Hyprland's default gaps: between windows and around the edge.
pub const GAPS_IN: u32 = 5;
pub const GAPS_OUT: u32 = 20;

impl Tree {
    pub const fn new() -> Tree {
        Tree { root: None }
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Every window in the tree, in order: first child before second.
    pub fn leaves(&self) -> Vec<u64> {
        let mut out = Vec::new();
        fn walk(n: &Node, out: &mut Vec<u64>) {
            match n {
                Node::Leaf(s) => out.push(*s),
                Node::Split { a, b, .. } => {
                    walk(a, out);
                    walk(b, out);
                }
            }
        }
        if let Some(r) = &self.root {
            walk(r, &mut out);
        }
        out
    }

    pub fn contains(&self, serial: u64) -> bool {
        self.leaves().contains(&serial)
    }

    /// Put `serial` in, sharing the slot of `beside` -- the window it was
    /// opened from -- or of the last window when that is not in the tree.
    /// The new window takes the second half.
    pub fn insert(&mut self, serial: u64, beside: Option<u64>) {
        if self.contains(serial) {
            return;
        }
        let Some(root) = self.root.take() else {
            self.root = Some(Node::Leaf(serial));
            return;
        };
        let target = beside.filter(|b| contains(&root, *b)).or_else(|| last(&root));
        self.root = Some(match target {
            Some(t) => split_at(root, t, serial),
            None => root,
        });
    }

    /// Take `serial` out; its sibling inherits the slot they shared.
    pub fn remove(&mut self, serial: u64) {
        if let Some(r) = self.root.take() {
            self.root = without(r, serial);
        }
    }

    /// Exchange two windows' places.
    pub fn swap(&mut self, x: u64, y: u64) {
        fn walk(n: &mut Node, x: u64, y: u64) {
            match n {
                Node::Leaf(s) if *s == x => *s = y,
                Node::Leaf(s) if *s == y => *s = x,
                Node::Leaf(_) => {}
                Node::Split { a, b, .. } => {
                    walk(a, x, y);
                    walk(b, x, y);
                }
            }
        }
        if self.contains(x) && self.contains(y) {
            if let Some(r) = &mut self.root {
                walk(r, x, y);
            }
        }
    }

    /// Pin the direction of the split `serial` sits in, the other way from
    /// how it is showing in `area`.
    pub fn toggle(&mut self, serial: u64, area: Rect) {
        fn walk(n: &mut Node, serial: u64, slot: Rect) -> bool {
            let Node::Split { pin, ratio, a, b } = n else { return false };
            let side = pin.unwrap_or(slot.w >= slot.h);
            let (sa, sb) = halves(slot, side, *ratio, 0);
            let here = matches!(**a, Node::Leaf(s) if s == serial) || matches!(**b, Node::Leaf(s) if s == serial);
            if here {
                *pin = Some(!side);
                return true;
            }
            walk(a, serial, sa) || walk(b, serial, sb)
        }
        if let Some(r) = &mut self.root {
            walk(r, serial, area);
        }
    }

    /// Where every window goes in `area`: gaps of `GAPS_OUT` round the edge
    /// and `GAPS_IN` either side of every split, so two neighbours stand
    /// twice that apart -- Hyprland's arithmetic.
    pub fn layout(&self, area: Rect) -> Vec<(u64, Rect)> {
        let mut out = Vec::new();
        let Some(r) = &self.root else { return out };
        let inner = Rect::new(
            area.x + GAPS_OUT,
            area.y + GAPS_OUT,
            area.w.saturating_sub(2 * GAPS_OUT),
            area.h.saturating_sub(2 * GAPS_OUT),
        );
        fn walk(n: &Node, slot: Rect, out: &mut Vec<(u64, Rect)>) {
            match n {
                Node::Leaf(s) => out.push((*s, slot)),
                Node::Split { pin, ratio, a, b } => {
                    let side = pin.unwrap_or(slot.w >= slot.h);
                    let (sa, sb) = halves(slot, side, *ratio, GAPS_IN);
                    walk(a, sa, out);
                    walk(b, sb, out);
                }
            }
        }
        walk(r, inner, &mut out);
        out
    }
}

fn contains(n: &Node, serial: u64) -> bool {
    match n {
        Node::Leaf(s) => *s == serial,
        Node::Split { a, b, .. } => contains(a, serial) || contains(b, serial),
    }
}

fn last(n: &Node) -> Option<u64> {
    match n {
        Node::Leaf(s) => Some(*s),
        Node::Split { b, .. } => last(b),
    }
}

fn split_at(n: Node, target: u64, serial: u64) -> Node {
    match n {
        Node::Leaf(s) if s == target => Node::Split {
            pin: None,
            ratio: 500,
            a: Box::new(Node::Leaf(s)),
            b: Box::new(Node::Leaf(serial)),
        },
        Node::Leaf(s) => Node::Leaf(s),
        Node::Split { pin, ratio, a, b } => Node::Split {
            pin,
            ratio,
            a: Box::new(split_at(*a, target, serial)),
            b: Box::new(split_at(*b, target, serial)),
        },
    }
}

fn without(n: Node, serial: u64) -> Option<Node> {
    match n {
        Node::Leaf(s) if s == serial => None,
        Node::Leaf(s) => Some(Node::Leaf(s)),
        Node::Split { pin, ratio, a, b } => match (without(*a, serial), without(*b, serial)) {
            (Some(a), Some(b)) => Some(Node::Split { pin, ratio, a: Box::new(a), b: Box::new(b) }),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        },
    }
}

/// A slot cut in two along `side`, the first part `ratio` thousandths of it,
/// with `gap` taken from each side of the cut.
fn halves(slot: Rect, side: bool, ratio: u16, gap: u32) -> (Rect, Rect) {
    let ratio = ratio.clamp(100, 900) as u32;
    if side {
        let cut = slot.w * ratio / 1000;
        let aw = cut.saturating_sub(gap);
        let bx = slot.x + (cut + gap).min(slot.w);
        (Rect::new(slot.x, slot.y, aw, slot.h), Rect::new(bx, slot.y, (slot.x + slot.w).saturating_sub(bx), slot.h))
    } else {
        let cut = slot.h * ratio / 1000;
        let ah = cut.saturating_sub(gap);
        let by = slot.y + (cut + gap).min(slot.h);
        (Rect::new(slot.x, slot.y, slot.w, ah), Rect::new(slot.x, by, slot.w, (slot.y + slot.h).saturating_sub(by)))
    }
}

/// The window whose centre is nearest `from` in direction `dir` (0 left, 1
/// right, 2 up, 3 down), among `rects`: what Super and an arrow focuses, and
/// what Super-Shift and an arrow swaps with. Nearest along the direction,
/// with sideways distance counted double, so the window straight across wins
/// over one further round the corner.
pub fn neighbour(rects: &[(u64, Rect)], from: u64, dir: u8) -> Option<u64> {
    let (_, r) = rects.iter().find(|(s, _)| *s == from)?;
    let c = |r: &Rect| ((r.x + r.w / 2) as i64, (r.y + r.h / 2) as i64);
    let (fx, fy) = c(r);
    rects
        .iter()
        .filter(|(s, _)| *s != from)
        .filter_map(|(s, r)| {
            let (x, y) = c(r);
            let (ahead, side) = match dir {
                0 => (fx - x, (y - fy).abs()),
                1 => (x - fx, (y - fy).abs()),
                2 => (fy - y, (x - fx).abs()),
                _ => (y - fy, (x - fx).abs()),
            };
            (ahead > 0).then_some((ahead + 2 * side, *s))
        })
        .min()
        .map(|(_, s)| s)
}

/// `diag tile`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let area = Rect::new(0, 0, 1280, 760);
    let inner = Rect::new(GAPS_OUT, GAPS_OUT, area.w - 2 * GAPS_OUT, area.h - 2 * GAPS_OUT);
    let overlap = |a: &Rect, b: &Rect| a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    let inside = |r: &Rect| r.x >= inner.x && r.y >= inner.y && r.x + r.w <= inner.x + inner.w && r.y + r.h <= inner.y + inner.h;

    let mut t = Tree::new();
    claim("an empty tree lays out nothing", t.layout(area).is_empty());
    t.insert(1, None);
    claim("one window fills the area inside the outer gaps", t.layout(area) == alloc::vec![(1, inner)]);
    t.insert(2, Some(1));
    let l = t.layout(area);
    claim(
        "a second splits the first along its longer side",
        l.len() == 2 && l[0].1.y == l[1].1.y && l[0].1.h == inner.h && l[1].1.x > l[0].1.x,
    );
    claim("two neighbours stand twice the inner gap apart", l[1].1.x - (l[0].1.x + l[0].1.w) == 2 * GAPS_IN);
    claim("and the two reach both edges", l[0].1.x == inner.x && l[1].1.x + l[1].1.w == inner.x + inner.w);
    t.insert(3, Some(2));
    let l = t.layout(area);
    let r2 = l.iter().find(|(s, _)| *s == 2).unwrap().1;
    let r3 = l.iter().find(|(s, _)| *s == 3).unwrap().1;
    claim("a third halves the tall slot top to bottom", r2.x == r3.x && r3.y > r2.y && r2.w == r3.w);

    // Many windows, opened from wherever: the layout always tiles the area.
    let mut t = Tree::new();
    let mut tiles_exactly = true;
    for n in 1..=9u64 {
        t.insert(n, Some((n * 7) % n.max(1)));
        let l = t.layout(area);
        tiles_exactly &= l.len() == n as usize && l.iter().all(|(_, r)| inside(r) && r.w > 0 && r.h > 0);
        for (i, (_, a)) in l.iter().enumerate() {
            for (_, b) in l.iter().skip(i + 1) {
                tiles_exactly &= !overlap(a, b);
            }
        }
        // Covered: the area of the windows plus the gaps between them is
        // the whole inner area, so nothing is left unclaimed.
        let used: u64 = l.iter().map(|(_, r)| r.w as u64 * r.h as u64).sum();
        tiles_exactly &= used <= inner.w as u64 * inner.h as u64;
        let x1 = l.iter().map(|(_, r)| r.x + r.w).max().unwrap_or(0);
        let y1 = l.iter().map(|(_, r)| r.y + r.h).max().unwrap_or(0);
        tiles_exactly &= x1 == inner.x + inner.w && y1 == inner.y + inner.h;
    }
    claim("up to nine windows: inside the area, reaching its far edges, none overlapping", tiles_exactly);

    let before = {
        let mut t = Tree::new();
        t.insert(1, None);
        t.insert(2, Some(1));
        t.layout(area)
    };
    let mut t = Tree::new();
    t.insert(1, None);
    t.insert(2, Some(1));
    t.insert(3, Some(2));
    t.remove(3);
    claim("closing a window gives its half back exactly", t.layout(area) == before);
    t.remove(1);
    claim("and the last one left fills the area again", t.layout(area) == alloc::vec![(2, inner)]);
    t.remove(2);
    claim("removing everything empties the tree", t.is_empty());

    let mut t = Tree::new();
    t.insert(1, None);
    t.insert(2, Some(1));
    let l = t.layout(area);
    t.swap(1, 2);
    let s = t.layout(area);
    claim("a swap exchanges two windows' places", s[0] == (2, l[0].1) && s[1] == (1, l[1].1));
    t.toggle(1, area);
    let v = t.layout(area);
    claim("a pinned split turns the other way", v[0].1.x == v[1].1.x && v[0].1.w == inner.w);
    claim("inserting a window twice is a no-op", {
        let mut u = t.clone();
        u.insert(1, None);
        u == t
    });

    let rects = [(1u64, Rect::new(0, 0, 100, 100)), (2, Rect::new(200, 0, 100, 100)), (3, Rect::new(0, 200, 100, 100)), (4, Rect::new(200, 200, 100, 100))];
    claim(
        "a neighbour is the window that way, nearest straight across",
        neighbour(&rects, 1, 1) == Some(2) && neighbour(&rects, 1, 3) == Some(3) && neighbour(&rects, 4, 0) == Some(3) && neighbour(&rects, 4, 2) == Some(2),
    );
    claim("and there is none past the edge", neighbour(&rects, 1, 0).is_none() && neighbour(&rects, 1, 2).is_none());
    ok
}
