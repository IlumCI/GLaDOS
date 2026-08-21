//! The hardware runbook.
//!
//! The machine that builds GLaDOS is not the machine that runs it, and the
//! person carrying a change to the GF63 needs to verify it there without
//! coming back to ask what "logits 7 11 3" was supposed to mean. So this is
//! not a checklist of reminders -- it is a runbook. Each step is selectable,
//! and the pane below it shows the exact command, where to run it (in GLaDOS,
//! on the dev box, or physically at the laptop), what a pass looks like, and
//! what a failure means. A step you can tick without knowing whether it
//! passed is a step that verifies nothing.
//!
//! The tick state is shared with the `todo` shell command, so a step ticked
//! in the window shows ticked in the terminal and back. One list, because a
//! checklist that exists twice is two checklists that disagree.

use super::theme::{self, Rect};
use super::{Color, DeskApp, Framebuffer};
use crate::sync::Racy;
use alloc::format;
use core::cell::Cell;
use alloc::string::String;

/// Where a step is carried out.
#[derive(Clone, Copy)]
pub enum Place {
    /// Typed at the GLaDOS shell, or read from its screen.
    Glados,
    /// On the development machine, in PowerShell.
    Host,
    /// At the laptop itself -- a reboot, a cable, a camera.
    Physical,
}

impl Place {
    pub fn tag(self) -> &'static str {
        match self {
            Place::Glados => "GLaDOS",
            Place::Host => "host",
            Place::Physical => "phys",
        }
    }
    fn color(self) -> Color {
        match self {
            Place::Glados => theme::APERTURE,
            Place::Host => Color::new(0x5A, 0x9B, 0xD5),
            Place::Physical => Color::new(0x30, 0xA0, 0x40),
        }
    }
}

pub struct Step {
    pub title: &'static str,
    pub place: Place,
    /// The exact command, or empty for a step that is an action rather than a
    /// command (a reboot, reading the boot log).
    pub cmd: &'static str,
    pub expect: &'static str,
    pub fail: &'static str,
}

/// The runbook, in the order a hardware visit should run it.
pub const STEPS: &[Step] = &[
    Step {
        title: "Deploy the kernel to the stick",
        place: Place::Host,
        cmd: ".\\scripts\\deploy.ps1 -EspDrive S: -Release",
        expect: "builds, then 'deployed ... BOOTX64.EFI' and the payload lines \
                 (model/tokenizer/roots, unchanged or copied).",
        fail: "'cargo build failed' -> fix the build first. Wrong drive -> \
               confirm S: is the ESP stick, not the Windows disk.",
    },
    Step {
        title: "Boot it: reboot, hold F11, pick USB",
        place: Place::Physical,
        cmd: "",
        expect: "the aperture splash, then the desktop with the icon taskbar.",
        fail: "firmware drops to the UEFI shell -> a stale NVRAM boot entry; \
               re-running deploy resets it. Black screen -> GOP framebuffer \
               issue, photograph anything on screen.",
    },
    Step {
        title: "Record the [boot] phys line",
        place: Place::Glados,
        cmd: "read the boot log (it scrolls past; the terminal keeps it)",
        expect: "'[boot] phys N MiB free, largest contiguous region M MiB'. \
                 Write down M -- it has never been captured on this machine.",
        fail: "no such line -> the heap ladder came down a rung; note which \
               size it reports getting. M is what decides whether 4B fits.",
    },
    Step {
        title: "Boot selftests: every line ok",
        place: Place::Glados,
        cmd: "read the [selftest] lines (crypto, heap, timer, probe, ...)",
        expect: "every line says ok. Crypto is 11 published vector sets.",
        fail: "any FAIL -> photograph the exact line and send it. An ECDSA \
               break once hid in [selftest] crypto for a whole debug cycle.",
    },
    Step {
        title: "Swap Qwen3.5 in for Qwen3",
        place: Place::Glados,
        cmd: "on S:\\GLADOS: rename model.bin -> q3.bin, then q35.bin -> model.bin",
        expect: "next boot's [ai] line shows dim 1024, 24 layers, the hybrid. \
                 Keep both files -- do not overwrite.",
        fail: "[ai] still shows dim 576 -> the rename did not take; check you \
               edited the ESP (S:) and not a stale copy.",
    },
    Step {
        title: "Verify Qwen3.5 logits vs the oracle",
        place: Place::Glados,
        cmd: "logits 7 11 3",
        expect: "the same top-5 ids as the oracle. On the dev box the oracle is: \
                 tools\\venv\\Scripts\\python.exe tools\\ref35.py --converted \
                 out\\q35-0.8b.bin. PASS when the ids match (the last logit \
                 digits may differ -- the KV cache is int8, the oracle is not).",
        fail: "ids differ -> the kernel port regressed. Capture the five lines \
               and send them; that is a real bug, not a rounding wobble.",
    },
    Step {
        title: "q35 is LOGITS ONLY, not text",
        place: Place::Glados,
        cmd: "(nothing to run -- a fact to remember)",
        expect: "trust `logits`. Do NOT trust `gen`/`ask` text with q35 loaded: \
                 its tokenizer's split pattern is not implemented, so decoding \
                 its 248k vocab is undefined and the words are garbage.",
        fail: "if the text looks wrong, that is expected here, not a fault. The \
               tokenizer is the next upgrade on the dev side.",
    },
    Step {
        title: "Time generation, both models",
        place: Place::Glados,
        cmd: "gen -n 32 the",
        expect: "note the tok/s it prints. Run once with Qwen3 (model.bin = q3) \
                 and once with q35 loaded. First real-hardware speed numbers; \
                 with q35 time it, do not read it.",
        fail: "far slower than QEMU is normal for the first run (cold caches). \
               A hang -> note the last line printed.",
    },
    Step {
        title: "Confirm the memory win",
        place: Place::Glados,
        cmd: "window",
        expect: "with q35: ~12 MiB KV over 6 of 24 layers, plus 19.3 MiB of \
                 recurrent state. Against Qwen3-0.6b's 112 MiB KV at seq 512. \
                 That gap is the entire reason for the hybrid.",
        fail: "numbers far off -> the sparse KV allocation miscounted the \
               full-attention layers; capture `window` and `ctx`.",
    },
    Step {
        title: "Mouse on real hardware",
        place: Place::Physical,
        cmd: "move, click, right-click, drag a title bar, spin the wheel, Start",
        expect: "hover highlights, drag moves a window, the second button opens \
                 a menu, the wheel scrolls. The wheel is the one thing QEMU \
                 could never test.",
        fail: "wheel dead -> the sample-rate knock (200/100/80) did not unlock \
               it on this touchpad; tell me and I will trace it.",
    },
    Step {
        title: "Network on the real card",
        place: Place::Glados,
        cmd: "dhcp   then   dns example.com   then   https example.com /",
        expect: "an address from DHCP, an A record, a TLS 1.3 fetch. The \
                 rtl8168 has never run -- QEMU emulates the 8139.",
        fail: "no link -> the rtl8168 driver's first real outing; capture `net` \
               and `pci`. TLS says 'encrypts but does not verify' unless \
               roots.der is present, which is expected.",
    },
    Step {
        title: "Photograph it for the site",
        place: Place::Physical,
        cmd: "a camera: the desktop, the apps, Enternet, the Oracle",
        expect: "real-hardware shots to sit beside the QEMU captures on the \
                 site -- proof it boots on metal, not just an emulator.",
        fail: "(nothing fails here; it is the reward)",
    },
];

/// Tick state, one bit per step, shared with the shell command.
static DONE: Racy<[bool; 32]> = Racy::new([false; 32]);

pub fn toggle(i: usize) -> bool {
    if i >= STEPS.len() {
        return false;
    }
    let d = unsafe { &mut *DONE.get() };
    d[i] = !d[i];
    true
}

pub fn reset() {
    unsafe { *DONE.get() = [false; 32] };
}

pub fn is_done(i: usize) -> bool {
    unsafe { (*DONE.get())[i.min(31)] }
}

pub fn n_done() -> usize {
    (0..STEPS.len()).filter(|&i| is_done(i)).count()
}

pub struct Todo {
    sel: usize,
    /// First visible row. Owned by the draw pass, which alone knows the list
    /// height, so it is the one place "keep the selection on screen" can be
    /// decided -- the same arrangement Write and the Browser use.
    scroll: Cell<usize>,
}

impl Todo {
    pub fn new() -> Self {
        Self { sel: 0, scroll: Cell::new(0) }
    }

    pub fn preferred() -> (u32, u32) {
        (600, 500)
    }

    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let lh = theme::text_h();
        let head = Rect::new(client.x + 8, client.y + 6, client.w.saturating_sub(16), lh + 4);
        // The list gets a little over half; the detail pane the rest.
        let list_h = (client.h.saturating_sub(head.h + 20)) * 9 / 16;
        let list = Rect::new(client.x + 8, head.y + head.h + 4, client.w.saturating_sub(16), list_h);
        let detail = Rect::new(
            client.x + 8,
            list.y + list.h + 6,
            client.w.saturating_sub(16),
            client.h.saturating_sub(head.h + list_h + 34),
        );
        (head, list, detail)
    }

    fn row_h() -> u32 {
        theme::text_h() + 8
    }

    fn rows_shown(list: Rect) -> usize {
        ((list.h.saturating_sub(4)) / Self::row_h()).max(1) as usize
    }

    /// Wrap `text` into `r`, in `fg`, starting at row `line`. Returns the next
    /// free row. Breaks at spaces the way a reader would.
    fn wrap(fb: &Framebuffer, r: Rect, mut line: u32, label: &str, text: &str, fg: Color) -> u32 {
        let cw = theme::text_w(1).max(1);
        let lh = theme::text_h() + 2;
        let cols = (r.w / cw).max(1) as usize;
        let rows = (r.h / lh) as u32;

        // The label sits inline before the first wrapped line.
        let mut buf = String::from(label);
        buf.push_str(text);
        let bytes = buf.as_bytes();
        let mut at = 0;
        while at < bytes.len() && line < rows {
            let end = (at + cols).min(bytes.len());
            let cut = if end < bytes.len() {
                bytes[at..end].iter().rposition(|&b| b == b' ').map(|i| at + i).unwrap_or(end)
            } else {
                end
            };
            if let Ok(s) = core::str::from_utf8(&bytes[at..cut]) {
                theme::text_over(fb, r.x, r.y + line * lh, s.trim_start(), fg);
            }
            at = if cut == at { end } else { cut + 1 };
            line += 1;
        }
        line
    }
}

impl DeskApp for Todo {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (head, list, detail) = Self::layout(client);
        let lh = theme::text_h();

        let title = format!("Hardware runbook -- {} of {} done", n_done(), STEPS.len());
        theme::text(fb, head.x, head.y, &title, theme::TEXT, theme::FACE);

        // The checklist.
        theme::well(fb, list, theme::FACE);
        let inner = list.shrink(2);
        let rows = Self::rows_shown(list);
        let row_h = Self::row_h();
        // Follow the selection: the draw pass owns the scroll because it is
        // the only place the row count is known.
        let mut scroll = self.scroll.get().min(STEPS.len().saturating_sub(1));
        if self.sel < scroll {
            scroll = self.sel;
        } else if self.sel >= scroll + rows {
            scroll = self.sel + 1 - rows;
        }
        self.scroll.set(scroll);
        for k in 0..rows {
            let i = scroll + k;
            if i >= STEPS.len() {
                break;
            }
            let s = &STEPS[i];
            let ry = inner.y + k as u32 * row_h;
            let r = Rect::new(inner.x, ry, inner.w, row_h);
            let selected = i == self.sel;
            if selected {
                fb.rect(r.x, r.y, r.w, r.h, if focused { theme::SELECT } else { Color::new(0xA8, 0xA8, 0xA8) });
            }
            let fg = if selected && focused { theme::SELECT_TEXT } else { theme::TEXT };
            // Checkbox.
            let box_r = Rect::new(r.x + 4, r.y + (row_h - 14) / 2, 14, 14);
            fb.rect(box_r.x, box_r.y, box_r.w, box_r.h, theme::HILIGHT);
            theme::bevel(fb, box_r, false);
            if is_done(i) {
                // A tick, drawn by hand.
                fb.rect(box_r.x + 3, box_r.y + 7, 3, 3, theme::APERTURE_DEEP);
                fb.rect(box_r.x + 5, box_r.y + 9, 2, 2, theme::APERTURE_DEEP);
                fb.rect(box_r.x + 7, box_r.y + 4, 4, 5, theme::APERTURE_DEEP);
            }
            // Place tag, coloured.
            let tag = s.place.tag();
            let tx = box_r.x + box_r.w + 6;
            if !selected {
                theme::text(fb, tx, r.y + (row_h - lh) / 2, tag, s.place.color(), theme::FACE);
            } else {
                theme::text(fb, tx, r.y + (row_h - lh) / 2, tag, fg, theme::SELECT);
            }
            // Title, clipped to the row.
            let title_x = tx + theme::text_w(7);
            let room = ((r.x + r.w).saturating_sub(title_x) / theme::text_w(1).max(1)) as usize;
            let shown = if s.title.len() > room { &s.title[..room] } else { s.title };
            let bg = if selected { theme::SELECT } else { theme::FACE };
            let _ = bg;
            if selected && focused {
                theme::text(fb, title_x, r.y + (row_h - lh) / 2, shown, theme::SELECT_TEXT, theme::SELECT);
            } else {
                theme::text(fb, title_x, r.y + (row_h - lh) / 2, shown, theme::TEXT, if selected { Color::new(0xA8, 0xA8, 0xA8) } else { theme::FACE });
            }
        }

        // The detail pane for the selected step.
        theme::well(fb, detail, theme::SCREEN);
        let d = detail.shrink(5);
        let s = &STEPS[self.sel.min(STEPS.len() - 1)];
        let mut line = 0u32;
        // Where.
        theme::text_over(fb, d.x, d.y, "where: ", theme::SHADOW);
        theme::text_over(fb, d.x + theme::text_w(7), d.y, s.place.tag(), s.place.color());
        line += 1;
        if !s.cmd.is_empty() {
            line = Self::wrap(fb, Rect::new(d.x, d.y, d.w, d.h), line, "run:    ", s.cmd, theme::APERTURE);
        }
        line = Self::wrap(fb, Rect::new(d.x, d.y, d.w, d.h), line + 0, "pass:   ", s.expect, theme::SCREEN_TEXT);
        line = Self::wrap(fb, Rect::new(d.x, d.y, d.w, d.h), line, "if not: ", s.fail, Color::new(0xD5, 0x8A, 0x6A));
        let _ = line;

        // Footer hint.
        let sy = detail.y + detail.h + 4;
        if sy + lh < client.y + client.h {
            theme::text(
                fb,
                client.x + 8,
                sy,
                "up/down select   space ticks   also 'todo' in the shell",
                theme::TEXT,
                theme::FACE,
            );
        }
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd;
        match k {
            kbd::KEY_UP => self.sel = self.sel.saturating_sub(1),
            kbd::KEY_DOWN => self.sel = (self.sel + 1).min(STEPS.len() - 1),
            kbd::KEY_HOME => self.sel = 0,
            kbd::KEY_END => self.sel = STEPS.len() - 1,
            b' ' | b'\n' | b'\r' | b'x' | b'X' => {
                toggle(self.sel);
            }
            _ => return false,
        }
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let (_, list, _) = Self::layout(client);
        if x >= list.x as i32 && y >= list.y as i32 && x < (list.x + list.w) as i32 && y < (list.y + list.h) as i32 {
            let inner = list.shrink(2);
            let row = ((y - inner.y as i32).max(0) as u32 / Self::row_h()) as usize;
            let i = self.scroll.get() + row;
            if i < STEPS.len() {
                // A press on the checkbox toggles; anywhere else on the row
                // just selects, so reading a step does not tick it.
                if self.sel == i && x < (inner.x + 22) as i32 {
                    toggle(i);
                } else {
                    self.sel = i;
                }
                return true;
            }
        }
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let was = self.sel;
        self.sel = if notches > 0 {
            (self.sel + 1).min(STEPS.len() - 1)
        } else {
            self.sel.saturating_sub(1)
        };
        self.sel != was
    }
}
