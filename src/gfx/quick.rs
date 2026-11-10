//! Quick settings: the few settings worth one click, in a panel that opens
//! out of the taskbar beside the clock.
//!
//! What is on it is not chosen here. A setting is quick when its row in
//! `prefs::ITEMS` says so, so the panel and the Settings window cannot
//! disagree about what a switch does or what it is called; this module is a
//! layout and a hit-test over those rows, and `layout` is the one answer both
//! the paint pass and the press handler read.
//!
//! It is a desktop mode rather than a window, the way the Start menu is: it
//! closes when anything else is pressed, and it is drawn on the overlay layer
//! whose stamp carries every value it shows.

use super::prefs::{self, Ctl, Item};
use super::setapp::{circle, fit, glyph, line_h, measure, px, ring, rrect, size, text, CARD, EDGE, OFF, RULE};
use super::text::Face;
use super::theme::{self, Rect};
use super::{Color, Framebuffer};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use crate::sync::Spin;

#[derive(Clone, Copy, PartialEq)]
pub enum Hit {
    Toggle(usize),
    Slider(usize, Rect),
    Choice(usize, usize),
    All,
}

const TILE_OFF: Color = Color::new(0xE6, 0xEC, 0xF0);

fn items() -> impl Iterator<Item = (usize, &'static Item)> {
    prefs::ITEMS.iter().enumerate().filter(|(_, it)| it.quick && it.when.map(|f| f()).unwrap_or(true))
}

/// Values set from the panel and not yet read back, as `setapp` keeps them:
/// the command is queued for the shell, and a switch that flicked back for
/// the moment in between would look like it had refused.
static PENDING: Spin<Vec<(usize, i32)>> = Spin::new(Vec::new());

fn live(it: &Item) -> i32 {
    match it.ctl {
        Ctl::Toggle { get, .. } => get() as i32,
        Ctl::Slider { get, .. } => get(),
        Ctl::Choice { get, .. } => get() as i32,
        _ => 0,
    }
}

fn shown(i: usize) -> i32 {
    let v = live(&prefs::ITEMS[i]);
    let mut p = PENDING.lock();
    match p.iter().position(|(k, _)| *k == i) {
        Some(at) if p[at].1 == v => {
            p.remove(at);
            v
        }
        Some(at) => p[at].1,
        None => v,
    }
}

fn set(i: usize, v: i32, cmd: &str) {
    {
        let mut p = PENDING.lock();
        p.retain(|(k, _)| *k != i);
        p.push((i, v));
    }
    super::desk::queue_command(cmd);
}

/// The tray button that opens the panel, left of the readouts.
pub fn button(bar: Rect, right: u32) -> Rect {
    let h = bar.h.saturating_sub(8);
    Rect::new(right.saturating_sub(h + 10), bar.y + 4, h + 4, h)
}

/// The panel, its right edge over the button's, standing on the taskbar.
pub fn panel(fb: &Framebuffer, button: Rect, bar: Rect) -> Rect {
    let w = px(392).min(fb.width().saturating_sub(16));
    let h = height(w);
    let right = (button.x + button.w + px(8)).min(fb.width().saturating_sub(8));
    Rect::new(right.saturating_sub(w), bar.y.saturating_sub(h + px(8)), w, h)
}

fn tile_h() -> u32 {
    px(62)
}
fn row_h() -> u32 {
    px(48)
}

fn height(_w: u32) -> u32 {
    let pad = px(14);
    let gap = px(8);
    let toggles = items().filter(|(_, it)| matches!(it.ctl, Ctl::Toggle { .. })).count() as u32;
    let others = items().filter(|(_, it)| !matches!(it.ctl, Ctl::Toggle { .. })).count() as u32;
    let rows = (toggles + 1) / 2;
    pad + rows * (tile_h() + gap) + others * row_h() + px(8) + px(44) + pad / 2
}

/// Every pressable thing on the panel, in keyboard order.
pub fn layout(p: Rect) -> Vec<(Rect, Hit)> {
    let pad = px(14);
    let gap = px(8);
    let mut out = Vec::new();
    let x0 = p.x + pad;
    let w = p.w - pad * 2;
    let tw = (w - gap) / 2;
    let mut y = p.y + pad;
    let mut col = 0;
    for (i, it) in items().filter(|(_, it)| matches!(it.ctl, Ctl::Toggle { .. })) {
        out.push((Rect::new(x0 + col * (tw + gap), y, tw, tile_h()), Hit::Toggle(i)));
        col += 1;
        if col == 2 {
            col = 0;
            y += tile_h() + gap;
        }
    }
    if col != 0 {
        y += tile_h() + gap;
    }
    for (i, it) in items().filter(|(_, it)| !matches!(it.ctl, Ctl::Toggle { .. })) {
        match it.ctl {
            Ctl::Slider { .. } => {
                let lx = x0 + px(36);
                let vw = measure("100%", size(), Face::Ui) + px(10);
                let track = Rect::new(lx, y + row_h() / 2 - px(10), w.saturating_sub(px(36) + vw), px(20));
                out.push((track, Hit::Slider(i, track)));
            }
            Ctl::Choice { opts, .. } => {
                let bh = px(30);
                let by = y + (row_h() - bh) / 2;
                let total: u32 = opts.iter().map(|(l, _)| measure(l, size() * 0.93, Face::Ui) + px(22)).sum();
                let mut bx = (x0 + w).saturating_sub(total + px(4)) + px(2);
                for (o, (l, _)) in opts.iter().enumerate() {
                    let bw = measure(l, size() * 0.93, Face::Ui) + px(22);
                    out.push((Rect::new(bx, by + px(2), bw, bh - px(4)), Hit::Choice(i, o)));
                    bx += bw;
                }
            }
            _ => {}
        }
        y += row_h();
    }
    y += px(8);
    let label = "All settings";
    let bw = measure(label, size(), Face::Ui) + px(40);
    out.push((Rect::new((x0 + w).saturating_sub(bw), y + px(6), bw, px(32)), Hit::All));
    out
}

fn inside(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && x < (r.x + r.w) as i32 && y >= r.y as i32 && y < (r.y + r.h) as i32
}

pub fn paint(fb: &Framebuffer, p: Rect, sel: usize) {
    super::shadow::cast(fb, p.x as i32, p.y as i32, p.w, p.h, super::shadow::Spec { radius: 12, sigma: 10.0 }, 0, 4, 90);
    rrect(fb, p.x as i32, p.y as i32, p.w, p.h, px(12), Color::new(0xF3, 0xF6, 0xF8));
    ring(fb, p.x as i32, p.y as i32, p.w, p.h, px(12), 1, EDGE);
    let hits = layout(p);
    let pad = px(14);
    let x0 = p.x + pad;
    let w = p.w - pad * 2;
    let mut drawn_rows: Vec<usize> = Vec::new();
    for (k, (r, h)) in hits.iter().enumerate() {
        let kb = k == sel;
        match *h {
            Hit::Toggle(i) => {
                let it = &prefs::ITEMS[i];
                let on = shown(i) != 0;
                let (bg, fg, sub) = if on { (theme::APERTURE, CARD, Color::new(0xFF, 0xF1, 0xE2)) } else { (TILE_OFF, theme::TEXT, theme::TEXT_DIM) };
                rrect(fb, r.x as i32, r.y as i32, r.w, r.h, px(10), bg);
                if kb {
                    ring(fb, r.x as i32 - 2, r.y as i32 - 2, r.w + 4, r.h + 4, px(12), 2, theme::APERTURE_DEEP);
                }
                let g = prefs::page(it.page).map(|p| p.glyph).unwrap_or(prefs::Glyph::System);
                let gx = r.x + px(22);
                glyph(fb, g, gx as f32, (r.y + r.h / 2) as f32, px(10) as f32, fg);
                let tx = (gx + px(20)) as i32;
                let room = (r.x + r.w).saturating_sub(tx as u32 + px(8));
                let ty = r.y as i32 + (r.h as i32 - line_h(size() * 0.93) as i32 - line_h(size() * 0.8) as i32) / 2;
                text(fb, tx, ty, &fit(it.label, size() * 0.93, Face::UiStrong, room), size() * 0.93, Face::UiStrong, fg);
                text(fb, tx, ty + line_h(size() * 0.93) as i32, if on { "On" } else { "Off" }, size() * 0.8, Face::Ui, sub);
            }
            Hit::Slider(i, track) => {
                let it = &prefs::ITEMS[i];
                let Ctl::Slider { min, max, unit, .. } = it.ctl else { continue };
                let v = shown(i);
                let cy = (track.y + track.h / 2) as i32;
                let g = prefs::page(it.page).map(|p| p.glyph).unwrap_or(prefs::Glyph::Sound);
                glyph(fb, g, (x0 + px(12)) as f32, cy as f32, px(9) as f32, theme::TEXT);
                let frac = (v - min) as f32 / (max - min).max(1) as f32;
                rrect(fb, track.x as i32, cy - px(2) as i32, track.w, px(4), px(2), OFF);
                let fill = (track.w as f32 * frac) as u32;
                rrect(fb, track.x as i32, cy - px(2) as i32, fill.max(px(4)), px(4), px(2), theme::APERTURE);
                let kx = track.x as f32 + fill as f32;
                circle(fb, kx, cy as f32, px(9) as f32, if kb { theme::APERTURE_DEEP } else { theme::APERTURE });
                circle(fb, kx, cy as f32, px(5) as f32, CARD);
                text(fb, (track.x + track.w + px(10)) as i32, cy - line_h(size()) as i32 / 2, &format!("{}{}", v, unit.trim()), size(), Face::Ui, theme::TEXT);
            }
            Hit::Choice(i, o) => {
                let it = &prefs::ITEMS[i];
                let Ctl::Choice { opts, .. } = it.ctl else { continue };
                if !drawn_rows.contains(&i) {
                    drawn_rows.push(i);
                    // The track behind the choices, and the setting's name.
                    let first = hits.iter().find(|(_, h)| matches!(h, Hit::Choice(j, 0) if *j == i)).map(|(r, _)| *r).unwrap_or(*r);
                    let total: u32 = hits.iter().filter(|(_, h)| matches!(h, Hit::Choice(j, _) if *j == i)).map(|(r, _)| r.w).sum();
                    rrect(fb, first.x as i32 - px(2) as i32, first.y as i32 - px(2) as i32, total + px(4), first.h + px(4), (first.h + px(4)) / 2, TILE_OFF);
                    let g = prefs::page(it.page).map(|p| p.glyph).unwrap_or(prefs::Glyph::Power);
                    let cy = first.y + first.h / 2;
                    glyph(fb, g, (x0 + px(12)) as f32, cy as f32, px(9) as f32, theme::TEXT);
                    // The name only where it fits whole: the glyph says it
                    // otherwise, and "..." says nothing.
                    let room = first.x.saturating_sub(x0 + px(36) + px(6));
                    if measure(it.label, size() * 0.93, Face::Ui) <= room {
                        text(fb, (x0 + px(32)) as i32, cy as i32 - line_h(size() * 0.93) as i32 / 2, it.label, size() * 0.93, Face::Ui, theme::TEXT);
                    }
                }
                let on = shown(i) as usize == o;
                if on {
                    rrect(fb, r.x as i32, r.y as i32, r.w, r.h, r.h / 2, theme::APERTURE);
                }
                if kb {
                    ring(fb, r.x as i32 - 1, r.y as i32 - 1, r.w + 2, r.h + 2, r.h / 2 + 1, 2, theme::APERTURE_DEEP);
                }
                let l = opts[o].0;
                text(fb, (r.x + px(11)) as i32, r.y as i32 + (r.h as i32 - line_h(size() * 0.93) as i32) / 2, l, size() * 0.93, if on { Face::UiStrong } else { Face::Ui }, if on { CARD } else { theme::TEXT });
            }
            Hit::All => {
                fb.rect(x0, r.y - px(8), w, 1, RULE);
                rrect(fb, r.x as i32, r.y as i32, r.w, r.h, r.h / 2, if kb { theme::APERTURE } else { CARD });
                ring(fb, r.x as i32, r.y as i32, r.w, r.h, r.h / 2, 1, if kb { theme::APERTURE_DEEP } else { EDGE });
                let fg = if kb { CARD } else { theme::TEXT };
                glyph(fb, prefs::Glyph::System, (r.x + px(16)) as f32, (r.y + r.h / 2) as f32, px(7) as f32, fg);
                text(fb, (r.x + px(28)) as i32, r.y as i32 + (r.h as i32 - line_h(size()) as i32) / 2, "All settings", size(), Face::Ui, fg);
                // What the machine is doing, at the panel's foot.
                let note = fit(&(prefs::PAGES.iter().find(|p| p.id == "system").map(|p| (p.summary)()).unwrap_or_default()), size() * 0.86, Face::Ui, r.x.saturating_sub(x0 + px(8)));
                text(fb, x0 as i32, r.y as i32 + (r.h as i32 - line_h(size() * 0.86) as i32) / 2, &note, size() * 0.86, Face::Ui, theme::TEXT_DIM);
            }
        }
    }
}

/// What a value the panel shows reads now, for the overlay layer's stamp.
pub fn stamp() -> u64 {
    let vals: Vec<(usize, i32)> = items().map(|(i, _)| (i, shown(i))).collect();
    super::scene::Stamp::of(vals)
}

fn slider_at(i: usize, track: Rect, x: i32) -> Option<(i32, String)> {
    let Ctl::Slider { min, max, step, cmd, .. } = prefs::ITEMS[i].ctl else { return None };
    let frac = ((x - track.x as i32) as f32 / track.w.max(1) as f32).clamp(0.0, 1.0);
    let raw = min as f32 + frac * (max - min) as f32;
    let v = ((((raw - min as f32) / step as f32) + 0.5) as i32 * step + min).clamp(min, max);
    Some((v, cmd(v)))
}

/// What pressing `hits[k]` does. Answers whether the panel should close.
fn fire(hits: &[(Rect, Hit)], k: usize, x: Option<i32>) -> bool {
    let Some((_, h)) = hits.get(k) else { return false };
    match *h {
        Hit::Toggle(i) => {
            if let Ctl::Toggle { on, off, .. } = prefs::ITEMS[i].ctl {
                let now = shown(i) != 0;
                set(i, (!now) as i32, if now { off } else { on });
            }
            false
        }
        Hit::Slider(i, track) => {
            if let Some(x) = x {
                if let Some((v, cmd)) = slider_at(i, track, x) {
                    set(i, v, &cmd);
                }
            }
            false
        }
        Hit::Choice(i, o) => {
            if let Ctl::Choice { opts, .. } = prefs::ITEMS[i].ctl {
                set(i, o as i32, opts[o].1);
            }
            false
        }
        Hit::All => {
            super::desk::queue_command("settings");
            true
        }
    }
}

/// A press inside the panel. Answers (consumed, close, new selection).
pub fn press(p: Rect, x: i32, y: i32) -> (bool, bool, Option<usize>) {
    if !inside(p, x, y) {
        return (false, true, None);
    }
    let hits = layout(p);
    let Some(k) = hits.iter().position(|(r, _)| inside(Rect::new(r.x, r.y.saturating_sub(px(6)), r.w, r.h + px(12)), x, y)) else {
        return (true, false, None);
    };
    let close = fire(&hits, k, Some(x));
    (true, close, Some(k))
}

/// A key while the panel is open. Answers (new selection, close).
pub fn key(p: Rect, sel: usize, k: u8) -> (usize, bool) {
    use crate::dev::kbd::{KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP};
    let hits = layout(p);
    let n = hits.len().max(1);
    match k {
        27 => (sel, true),
        KEY_DOWN | b'\t' => (step_row(&hits, sel, 1), false),
        KEY_UP | crate::dev::kbd::KEY_BACKTAB => (step_row(&hits, sel, -1), false),
        KEY_LEFT | KEY_RIGHT => {
            let d: i32 = if k == KEY_RIGHT { 1 } else { -1 };
            match hits.get(sel).map(|h| h.1) {
                Some(Hit::Slider(i, _)) => {
                    if let Ctl::Slider { min, max, step, cmd, .. } = prefs::ITEMS[i].ctl {
                        let v = (shown(i) + d * step).clamp(min, max);
                        set(i, v, &cmd(v));
                    }
                    (sel, false)
                }
                Some(Hit::Choice(i, _)) => {
                    // Left and Right walk the choices and pick as they go.
                    let next = (sel as i32 + d).clamp(0, n as i32 - 1) as usize;
                    match hits[next].1 {
                        Hit::Choice(j, _) if j == i => {
                            fire(&hits, next, None);
                            (next, false)
                        }
                        _ => (sel, false),
                    }
                }
                _ => ((sel as i32 + d).rem_euclid(n as i32) as usize, false),
            }
        }
        b'\n' | b'\r' | b' ' => {
            let close = fire(&hits, sel, None);
            (sel, close)
        }
        _ => (sel, false),
    }
}

/// Down and Up treat a row of choices as one stop, landing on the chosen one.
fn step_row(hits: &[(Rect, Hit)], sel: usize, d: i32) -> usize {
    let n = hits.len() as i32;
    let row_of = |h: Hit| match h {
        Hit::Choice(i, _) => Some(i),
        _ => None,
    };
    let here = hits.get(sel).and_then(|h| row_of(h.1));
    let mut k = sel as i32;
    for _ in 0..n {
        k = (k + d).rem_euclid(n);
        let h = hits[k as usize].1;
        match (row_of(h), here) {
            (Some(i), Some(j)) if i == j => continue,
            (Some(i), _) => {
                // The chosen one of that row.
                let cur = shown(i) as usize;
                return hits.iter().position(|(_, x)| *x == Hit::Choice(i, cur)).unwrap_or(k as usize);
            }
            _ => return k as usize,
        }
    }
    sel
}
