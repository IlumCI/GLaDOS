//! The installer, as a window: which disk, which way, exactly what will
//! happen, a typed confirmation, then the install itself.
//!
//! A front for `install`, and only a front, in `officewin`'s shape. The plan
//! it shows is `install::apply::describe` -- the same lines `install plan`
//! prints, from the same struct that is later written -- and the button that
//! installs queues `install erase <code>` or `install alongside <code>`
//! through `desk::queue_command`, so the work runs on the shell's task
//! exactly as if it had been typed. Nothing destructive happens on the
//! desktop's task, and the window cannot disagree with the verb.
//!
//! The confirmation is the plan's own eight-character code, typed back. A
//! click is not consent to erasing a disk; typing what is on the screen is
//! an act a person notices themselves doing.

use super::theme::{self, Rect};
use super::{DeskApp, Framebuffer};
use crate::install::{apply, firstboot, plan};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::sync::atomic::Ordering::Relaxed;

#[derive(Clone, Copy, PartialEq)]
enum Page {
    /// The disk, and the two ways to use it.
    Disk,
    /// Package sets and the model (shown for either image; the model only
    /// when this medium does not carry it).
    Software,
    /// Exactly what will happen, and the code to type.
    Confirm,
    /// Installing, then finished or stopped.
    Working,
}

struct Survey {
    lines: Vec<String>,
    erase: Result<(Vec<String>, String, Vec<String>), String>,
    alongside: Result<(Vec<String>, String, Vec<String>), String>,
}

/// Describe a plan for one mode: its lines, its code, and what blocks it.
fn mode_plan(disk: &plan::Disk, m: plan::Mode) -> Result<(Vec<String>, String, Vec<String>), String> {
    let p = plan::plan(disk, m, &plan::Want::DEFAULT).map_err(|r| String::from(r.say()))?;
    let keep: Vec<(u64, u64)> = p.kept.iter().map(|k| (k.start_lba, k.block_count)).collect();
    let why = apply::blockers(
        disk,
        crate::store::erofs::ram().map(|r| r.len()),
        crate::linux::disk::uses_disk_outside(&keep),
        crate::install::carried().iter().any(|(p, _)| p == "EFI/BOOT/BOOTX64.EFI"),
        &p,
    );
    Ok((apply::describe(disk, &p), apply::code(disk, &p), why))
}

fn survey() -> Survey {
    match apply::survey() {
        Err(e) => Survey { lines: alloc::vec![e.clone()], erase: Err(e.clone()), alongside: Err(e) },
        Ok(d) => {
            let lines = apply::describe(&d, &plan::Plan { mode: plan::Mode::Alongside, kept: d.partitions.clone(), new: Vec::new() });
            Survey {
                lines,
                erase: mode_plan(&d, plan::Mode::Erase),
                alongside: mode_plan(&d, plan::Mode::Alongside),
            }
        }
    }
}

pub struct Installer {
    page: Page,
    survey: Survey,
    mode: plan::Mode,
    sets: [bool; 4],
    model: bool,
    typed: String,
    btns: [Cell<Rect>; 6],
}

const BTN_N: usize = 6;

impl Installer {
    pub fn new() -> Installer {
        let z = || Cell::new(Rect::new(0, 0, 0, 0));
        Installer {
            page: Page::Disk,
            survey: survey(),
            mode: plan::Mode::Erase,
            sets: [true, true, false, false],
            model: !firstboot::medium_has_model(),
            typed: String::new(),
            btns: [z(), z(), z(), z(), z(), z()],
        }
    }

    fn chosen(&self) -> &Result<(Vec<String>, String, Vec<String>), String> {
        if self.mode == plan::Mode::Erase { &self.survey.erase } else { &self.survey.alongside }
    }

    fn usable(&self, m: plan::Mode) -> bool {
        let r = if m == plan::Mode::Erase { &self.survey.erase } else { &self.survey.alongside };
        r.as_ref().is_ok_and(|(_, _, why)| why.is_empty())
    }

    /// The buttons this page shows, by label; a press answers its index.
    fn buttons(&self) -> Vec<(&'static str, bool)> {
        match self.page {
            Page::Disk => alloc::vec![
                ("Erase this disk", self.usable(plan::Mode::Erase)),
                ("Install alongside", self.usable(plan::Mode::Alongside)),
                ("Look again", true),
            ],
            Page::Software => alloc::vec![
                ("Desktop", true),
                ("Office", true),
                ("Media", true),
                ("Model", !firstboot::medium_has_model()),
                ("Back", true),
                ("Next", true),
            ],
            Page::Confirm => alloc::vec![("Back", true), ("Install", self.code_matches())],
            Page::Working => match crate::install::STATE.load(Relaxed) {
                2 => alloc::vec![("Restart now", true)],
                3 => alloc::vec![("Back", true)],
                _ => Vec::new(),
            },
        }
    }

    fn code_matches(&self) -> bool {
        self.chosen().as_ref().is_ok_and(|(_, c, why)| why.is_empty() && self.typed.trim() == c)
    }

    fn fire(&mut self, i: usize) -> bool {
        let labels: Vec<&str> = self.buttons().iter().map(|b| b.0).collect();
        let Some(&label) = labels.get(i) else { return false };
        if !self.buttons()[i].1 {
            return true;
        }
        match (self.page, label) {
            (Page::Disk, "Erase this disk") => {
                self.mode = plan::Mode::Erase;
                self.page = Page::Software;
            }
            (Page::Disk, "Install alongside") => {
                self.mode = plan::Mode::Alongside;
                self.page = Page::Software;
            }
            (Page::Disk, "Look again") => self.survey = survey(),
            (Page::Software, "Desktop") => self.sets[1] = !self.sets[1],
            (Page::Software, "Office") => self.sets[2] = !self.sets[2],
            (Page::Software, "Media") => self.sets[3] = !self.sets[3],
            (Page::Software, "Model") => self.model = !self.model,
            (Page::Software, "Back") => self.page = Page::Disk,
            (Page::Software, "Next") => {
                // Office and media stand on the desktop set; choosing either
                // brings it.
                if self.sets[2] || self.sets[3] {
                    self.sets[1] = true;
                }
                let names: Vec<String> = firstboot::SETS
                    .iter()
                    .zip(self.sets.iter())
                    .filter(|(_, on)| **on)
                    .map(|((n, _), _)| String::from(*n))
                    .collect();
                let fetch_anything = names.iter().any(|n| n != "minimal") || self.model;
                firstboot::set_choice(fetch_anything.then(|| firstboot::Plan { sets: names, model: self.model }));
                self.survey = survey();
                self.typed.clear();
                self.page = Page::Confirm;
            }
            (Page::Confirm, "Back") => self.page = Page::Software,
            (Page::Confirm, "Install") => {
                if let Ok((_, code, _)) = self.chosen() {
                    let verb = if self.mode == plan::Mode::Erase { "erase" } else { "alongside" };
                    crate::install::STATE.store(1, Relaxed);
                    super::desk::queue_command(&format!("install {} {}", verb, code));
                    self.page = Page::Working;
                }
            }
            (Page::Working, "Restart now") => super::desk::queue_command("reboot"),
            (Page::Working, "Back") => {
                crate::install::STATE.store(0, Relaxed);
                self.survey = survey();
                self.page = Page::Disk;
            }
            _ => {}
        }
        true
    }

    /// What the body of the page says, line by line, with whether a line is
    /// a warning.
    fn body(&self) -> Vec<(String, bool)> {
        let mut out: Vec<(String, bool)> = Vec::new();
        let mut push = |s: String, warn: bool| out.push((s, warn));
        match self.page {
            Page::Disk => {
                push(String::from("Install GLaDOS on this computer's disk."), false);
                for l in &self.survey.lines {
                    push(format!("  {}", l), false);
                }
                for (name, r) in [("Erase", &self.survey.erase), ("Alongside", &self.survey.alongside)] {
                    match r {
                        Ok((_, _, why)) if why.is_empty() => push(format!("{}: possible", name), false),
                        Ok((_, _, why)) => push(format!("{}: {}", name, why[0]), true),
                        Err(e) => push(format!("{}: {}", name, e), true),
                    }
                }
                push(String::from("Alongside keeps everything that is there and uses free space; make room from Windows' Disk Management."), false);
            }
            Page::Software => {
                push(String::from("What the installed system should have. Sets come from the GLaDOS repository on its first start."), false);
                let names = ["the shell, a terminal and pacman", "Desktop -- GTK and Qt programs, fonts, editor, viewer", "Office -- LibreOffice, which the model writes into", "Media -- mpv for video and sound"];
                for (i, n) in names.iter().enumerate() {
                    push(format!("  [{}] {}", if self.sets[i] { 'x' } else { ' ' }, n), false);
                }
                if firstboot::medium_has_model() {
                    push(String::from("  [x] The model -- this medium carries it, so it is installed now"), false);
                } else {
                    push(format!("  [{}] The model -- about 600 MB, fetched on the first start", if self.model { 'x' } else { ' ' }), false);
                }
            }
            Page::Confirm => match self.chosen() {
                Err(e) => push(e.clone(), true),
                Ok((lines, code, why)) => {
                    if self.mode == plan::Mode::Erase {
                        push(String::from("Everything on this disk will be erased. Nothing on it can be recovered afterwards."), true);
                    } else {
                        push(String::from("Nothing on this disk moves. GLaDOS goes in the free space shown."), false);
                    }
                    for l in lines {
                        push(format!("  {}", l), false);
                    }
                    for w in why {
                        push(format!("cannot install: {}", w), true);
                    }
                    if let Some(p) = firstboot::choice() {
                        push(format!("On its first start it fetches: {}", p.packages().join(" ")), false);
                    }
                    push(format!("To go ahead, type {} below.", code), false);
                }
            },
            Page::Working => {
                let state = crate::install::STATE.load(Relaxed);
                push(String::from(match state {
                    1 => "Installing -- this takes a few minutes. Do not turn the computer off.",
                    2 => "Installed. Remove the install medium, then restart.",
                    3 => "The install stopped. Nothing after the line that failed was written.",
                    _ => "Waiting for the install to start...",
                }), state == 3);
                for l in crate::install::log() {
                    push(format!("  {}", l), l.starts_with("stopped"));
                }
            }
        }
        out
    }
}

fn dense() -> u32 {
    super::mindwin::dense()
}

fn row_h() -> u32 {
    theme::text_h_at(dense()) + 4
}

fn inside(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && x < (r.x + r.w) as i32 && y >= r.y as i32 && y < (r.y + r.h) as i32
}

impl DeskApp for Installer {
    fn stamp(&self) -> Option<u64> {
        Some(super::scene::Stamp::of((
            self.page as u8,
            self.typed.as_str(),
            self.sets,
            self.model,
            self.mode == plan::Mode::Erase,
            crate::install::STATE.load(Relaxed),
            crate::install::log().len(),
        )))
    }

    fn wake_ms(&self) -> Option<u32> {
        (self.page == Page::Working).then_some(300)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let pad = client.shrink(10);
        let rh = row_h();
        let cw = theme::text_w_at(1, dense()).max(1);
        let field_h = if self.page == Page::Confirm { rh + 10 } else { 0 };
        let bar = Rect::new(pad.x, pad.y + pad.h - rh - 8, pad.w, rh + 8);
        let body = Rect::new(pad.x, pad.y, pad.w, (bar.y.saturating_sub(pad.y)).saturating_sub(field_h + 8));
        theme::well(fb, body, theme::HILIGHT);
        let b = body.shrink(6);
        let room = (b.w / cw) as usize;
        let rows = (b.h / rh) as usize;
        let lines = self.body();
        // On the progress page the newest lines matter; elsewhere the first.
        let skip = if self.page == Page::Working { lines.len().saturating_sub(rows) } else { 0 };
        for (i, (l, warn)) in lines.iter().skip(skip).take(rows).enumerate() {
            let fg = if *warn { theme::APERTURE } else if i == 0 && skip == 0 { theme::TEXT } else { theme::TEXT_DIM };
            theme::text_over_at(fb, b.x, b.y + i as u32 * rh, theme::head_chars(l, room), fg, dense());
        }
        if self.page == Page::Confirm {
            let f = Rect::new(pad.x, body.y + body.h + 8, pad.w.min(cw * 24), rh + 10);
            theme::well(fb, f, theme::HILIGHT);
            let fr = f.shrink(5);
            let shown = if self.typed.is_empty() { "the code" } else { self.typed.as_str() };
            theme::text_over_at(fb, fr.x, fr.y, shown, if self.typed.is_empty() { theme::TEXT_DIM } else { theme::TEXT }, dense());
            if focused {
                let x = fr.x + if self.typed.is_empty() { 0 } else { theme::text_w_of_at(&self.typed, dense()) };
                fb.rect(x, fr.y, 2, theme::text_h_at(dense()), theme::APERTURE);
            }
        }
        let btns = self.buttons();
        let n = btns.len().max(1) as u32;
        for (i, cell) in self.btns.iter().enumerate() {
            if i >= btns.len() {
                cell.set(Rect::new(0, 0, 0, 0));
                continue;
            }
            let x0 = bar.x + bar.w * i as u32 / n;
            let x1 = bar.x + bar.w * (i as u32 + 1) / n;
            let r = Rect::new(x0 + 2, bar.y, x1.saturating_sub(x0 + 4), bar.h);
            cell.set(r);
            let (label, on) = btns[i];
            let lit = on && matches!(label, "Next" | "Install" | "Restart now") || (self.page == Page::Software && i < 4 && self.toggle(i));
            theme::button_at(fb, r, label, lit, !on, dense());
        }
    }

    /// Every button is a digit, in order, except while the code is being
    /// typed; Esc goes back. Serial cannot inject a click, and a wizard that
    /// only a pointer can drive is one nothing tests.
    fn key(&mut self, k: u8) -> bool {
        if k == 27 {
            return match self.page {
                Page::Software => self.fire(4),
                Page::Confirm => self.fire(0),
                _ => false,
            };
        }
        if self.page != Page::Confirm {
            return match k {
                b'1'..=b'6' => self.fire((k - b'1') as usize),
                _ => false,
            };
        }
        match k {
            b'\n' | b'\r' => {
                if self.code_matches() {
                    return self.fire(1);
                }
                true
            }
            8 => {
                self.typed.pop();
                true
            }
            b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => {
                if self.typed.len() < 16 {
                    self.typed.push((k as char).to_ascii_lowercase());
                }
                true
            }
            _ => false,
        }
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let ax = client.x as i32 + x;
        let ay = client.y as i32 + y;
        match self.btns.iter().position(|b| inside(b.get(), ax, ay)) {
            Some(i) => self.fire(i),
            None => false,
        }
    }

    fn min_size(&self) -> (u32, u32) {
        (520, 380)
    }
}

impl Installer {
    fn toggle(&self, i: usize) -> bool {
        match i {
            0 => self.sets[1],
            1 => self.sets[2],
            2 => self.sets[3],
            3 => self.model,
            _ => false,
        }
    }
}

const _: () = assert!(BTN_N >= 6);

/// Open the window, or bring it forward if it is open.
pub fn open() {
    if super::desk::has_window("Install") {
        super::desk::focus_titled("Install");
        return;
    }
    super::desk::post(super::desk::Op::OpenApp {
        title: "Install",
        icon: super::desk::ICO_SET,
        app: alloc::boxed::Box::new(Installer::new()),
        w: 680,
        h: 520,
    });
}
