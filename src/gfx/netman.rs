//! The network manager: what is in the air, and what we are on.
//!
//! One table, one passphrase field, three buttons. The columns are the five
//! facts that decide anything: what the network is called, how well it is
//! heard, which access point is offering it, what its security actually is,
//! and what address this machine has on it.
//!
//! ### Three of those columns are usually got wrong, so they are the point
//!
//! **SSID and ESSID are one field.** ESSID is the older name for a network's
//! name, from when independent and infrastructure networks were still being
//! distinguished by it; every tool that shows both is showing one thing twice.
//! So this shows it once, under the name people actually say.
//!
//! **The access point is the BSSID, and it is not the network.** Two access
//! points carrying one network have one SSID and two BSSIDs, and the column is
//! here because that is the only thing on screen that tells them apart. It is
//! what to look at when a laptop keeps joining the far one -- which the
//! rehearsal room reproduces on purpose, with `glados` on two channels at
//! -42 and -71 dBm.
//!
//! **"Secured" is not one state.** Open, WEP and WPA2 are three, and WEP has
//! been broken since 2001 -- a list that folds it in with WPA2 under one
//! padlock tells the operator the opposite of what is true. `Network::security`
//! answers all three by name and this prints the name.
//!
//! ### The address column, and why most rows are empty
//!
//! A network you have not joined has no address for you: an IP comes from
//! DHCP after association, and belongs to `wlan0` rather than to the network.
//! So exactly one row can ever show one, and the rest show a dash. Filling
//! them in would be inventing a number, and a made-up address in a network
//! manager is the one thing an operator would act on without checking.

use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;

use super::theme::{self, Rect};
use super::{DeskApp, Framebuffer};
use crate::net::wifi::{self, Network};

pub struct NetMan {
    sel: Cell<usize>,
    scroll: Cell<usize>,
    pass: String,
    /// What the last action said, shown along the foot. A window that does
    /// something and reports nothing is one you press twice.
    said: String,
    /// Which button is under the pointer's last press, for the pressed look.
    held: Cell<usize>,
}

impl NetMan {
    pub fn new() -> NetMan {
        NetMan {
            sel: Cell::new(0),
            scroll: Cell::new(0),
            pass: String::new(),
            said: String::new(),
            held: Cell::new(usize::MAX),
        }
    }

    pub fn preferred() -> (u32, u32) {
        // Wide enough for all seven columns at once. Narrower is allowed and
        // degrades from the right, which is the order to lose them in: the
        // address and the access point matter least when you are looking for
        // a network at all.
        (790, 380)
    }

    fn row_h() -> u32 {
        theme::text_h_at(1) + 6
    }

    /// Header, table, and the strip along the foot. One function, because a
    /// control that highlights in one place and presses in another is the bug
    /// this split exists to forbid -- `desk.rs` says so about its own chrome.
    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let area = client.shrink(8);
        // Three lines: what we are, what we are doing, and -- when the radio
        // is a rehearsal -- that it is not real. Its own line rather than
        // squeezed alongside, because at 640 pixels the two ran into each
        // other and printed over one another's letters.
        let head_h = theme::text_h_at(1) * 3 + 12;
        let foot_h = theme::text_h_at(1) + 18;
        let head = Rect::new(area.x, area.y, area.w, head_h.min(area.h));
        let foot_y = area.y + area.h.saturating_sub(foot_h);
        let foot = Rect::new(area.x, foot_y, area.w, foot_h.min(area.h));
        let table = Rect::new(
            area.x,
            head.y + head.h + 4,
            area.w,
            foot_y.saturating_sub(head.y + head.h + 8),
        );
        (head, table, foot)
    }

    /// The three buttons along the foot, right to left.
    fn buttons(foot: Rect) -> [(Rect, &'static str); 3] {
        // Room for the longest label at the window's own text scale. The
        // first version used `theme::button`, which draws at `CHROME_SCALE`
        // -- twice this -- so "Rescan" was truncated to "Resc" and "Leave" to
        // "Leav" by the button's own fitting code, silently and correctly.
        let w = theme::text_w_at(8, 1) + 10;
        let h = theme::text_h_at(1) + 10;
        let y = foot.y + 4;
        let r = foot.x + foot.w;
        [
            (Rect::new(r.saturating_sub(w), y, w, h), "Rescan"),
            (Rect::new(r.saturating_sub(w * 2 + 6), y, w, h), "Leave"),
            (Rect::new(r.saturating_sub(w * 3 + 12), y, w, h), "Join"),
        ]
    }

    fn nets() -> Vec<Network> {
        wifi::scan().unwrap_or_default()
    }

    /// The address this machine holds on `wlan0`, once it has one.
    fn own_ip() -> Option<String> {
        let w = &crate::net::ifaces()[crate::net::WLAN0];
        if w.ip == crate::net::UNSPECIFIED {
            return None;
        }
        Some(alloc::format!("{}.{}.{}.{}", w.ip[0], w.ip[1], w.ip[2], w.ip[3]))
    }

    /// The access point we are actually on, if any.
    ///
    /// **The access point and not the network.** The rehearsal room carries
    /// `glados` on two of them on purpose, and matching by name marked both
    /// rows as joined and printed this machine's address against each -- which
    /// is precisely the confusion the BSSID column exists to end, committed by
    /// the window that shows the column.
    fn joined() -> Option<crate::net::Mac> {
        crate::net::wlan_ap()
    }

    fn act(&mut self, which: &str) {
        let _claim = crate::net::claim_wifi();
        let now = crate::net::now_ms();
        match which {
            "Rescan" => match crate::net::wlan() {
                None => self.said = String::from("no radio -- 'wifi rehearse' to try this out"),
                Some(w) => {
                    // A scan is a join with no network named, which is how
                    // `mlme` has it: two ways to sweep the channels would be
                    // two things to keep agreeing about dwell and DFS.
                    w.join("", "", now);
                    self.said = String::from("scanning every channel -- a few seconds");
                }
            },
            "Leave" => match crate::net::wlan() {
                None => self.said = String::from("no radio"),
                Some(w) => {
                    w.leave_net();
                    self.said = String::from("left, and the access point was told");
                }
            },
            "Join" => {
                let nets = Self::nets();
                let Some(n) = nets.get(self.sel.get()) else {
                    self.said = String::from("nothing selected");
                    return;
                };
                if n.secured && self.pass.is_empty() {
                    self.said = String::from("that network is encrypted: type a passphrase first");
                    return;
                }
                if n.secured && !n.rsn {
                    self.said = String::from("that network is WEP, which is not security");
                    return;
                }
                let ssid = n.ssid.clone();
                let pass = self.pass.clone();
                match crate::net::wlan() {
                    None => self.said = String::from("no radio"),
                    Some(w) => {
                        w.join(&ssid, &pass, now);
                        self.said = alloc::format!("joining {}", ssid);
                        // Dropped the moment it has been handed over. It goes
                        // into 4096 rounds of PBKDF2 and is never written
                        // anywhere, which is the whole reason there is no
                        // saved-network list.
                        self.pass.clear();
                    }
                }
            }
            _ => {}
        }
    }
}

impl DeskApp for NetMan {
    fn min_size(&self) -> (u32, u32) {
        // A table of six columns plus a row of buttons has a floor, and under
        // it the access point column is the first thing to go -- which is the
        // column hardest to do without.
        (460, 240)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, _focused: bool) {
        theme::panel(fb, client);
        let (head, table, foot) = Self::layout(client);
        let lh = theme::text_h_at(1);
        let rh = Self::row_h();

        // The station is the clock task's too; held for the whole paint.
        let _claim = crate::net::claim_wifi();
        // --- what we are, and what we are on --------------------------
        let (adapter, state, secure) = match crate::net::wlan() {
            Some(w) => {
                let (s, sec) = w.status();
                (crate::net::wlan_name().unwrap_or("wireless"), s, sec)
            }
            None => ("none", "no radio", false),
        };
        let mac = crate::net::ifaces()[crate::net::WLAN0]
            .nic
            .as_ref()
            .map(|n| {
                let m = n.mac();
                alloc::format!(
                    "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                    m[0], m[1], m[2], m[3], m[4], m[5]
                )
            })
            .unwrap_or_else(|| String::from("--"));
        theme::text_over_at(
            fb,
            head.x,
            head.y,
            &alloc::format!("wlan0   {}   {}", adapter, mac),
            theme::TEXT,
            1,
        );
        let tone = if secure {
            theme::OK_TEXT
        } else if state == "running" {
            theme::WARN_TEXT
        } else {
            theme::TEXT_DIM
        };
        let line2 = match (crate::net::wlan_ssid(), Self::own_ip()) {
            (Some(ssid), Some(ip)) => alloc::format!(
                "{} on {}   {}   {}",
                state,
                ssid,
                ip,
                if secure { "encrypted" } else { "NOT ENCRYPTED" }
            ),
            (Some(ssid), None) => alloc::format!(
                "{} on {}   no address yet ('dhcp')   {}",
                state,
                ssid,
                if secure { "encrypted" } else { "NOT ENCRYPTED" }
            ),
            _ => String::from(state),
        };
        theme::text_over_at(fb, head.x, head.y + lh + 4, &line2, tone, 1);

        // A rehearsal radio is said so on the window, every time it is drawn.
        // A list of networks that do not exist, shown the way a real list is
        // shown, is the one thing this window must not do.
        if adapter == "rehearsal" {
            theme::text_over_at(
                fb,
                head.x,
                head.y + (lh + 4) * 2,
                "REHEARSAL RADIO -- there is no wireless part in this machine and these networks are not real",
                theme::BAD_TEXT,
                1,
            );
        }

        // --- the table -------------------------------------------------
        if table.is_empty() {
            return;
        }
        theme::panel(fb, table);
        let inner = table.shrink(3);
        let joined = Self::joined();
        let ip = Self::own_ip();

        // Columns, measured in characters and converted once, so the header
        // and the rows cannot disagree about where a column starts.
        let cw = theme::text_w_at(1, 1).max(1);
        // Character columns with their own widths, so nothing can spill into
        // the column beside it. **Both halves had to be here.** Clipping to
        // the window instead of to the column let "Enrichment Center" run
        // under SIGNAL, and a header longer than its column printed
        // "SIGNALCH" -- two spellings of the same mistake, one in the values
        // and one in the labels. The widths are the longest value each holds:
        // "WEP (broken)" is twelve and a BSSID is seventeen.
        let cols: [(u32, u32, &str); 7] = [
            (0, 20, "SSID / ESSID"),
            (21, 6, "SIGNAL"),
            (28, 3, "CH"),
            (32, 4, "BAND"),
            (37, 13, "SECURITY"),
            (51, 18, "ACCESS POINT"),
            (70, 15, "ADDRESS"),
        ];
        let put = |c: u32, w: u32, y: u32, s: &str, fg: super::Color| {
            let x = inner.x + c * cw;
            if x >= inner.x + inner.w {
                return;
            }
            // Whichever runs out first: the column, or the window.
            let room = (w as usize).min((((inner.x + inner.w) - x) / cw.max(1)) as usize);
            if room == 0 {
                return;
            }
            theme::text_over_at(fb, x, y, theme::head_chars(s, room), fg, 1);
        };
        for (c, w, name) in cols.iter() {
            put(*c, *w, inner.y, name, theme::TEXT_DIM);
        }
        let body = Rect::new(inner.x, inner.y + lh + 3, inner.w, inner.h.saturating_sub(lh + 3));
        let rows = (body.h / rh.max(1)) as usize;

        let nets = Self::nets();
        if nets.is_empty() {
            let why = match crate::net::wlan() {
                None => "no wireless part in this machine. 'wifi' says what is fitted; 'wifi rehearse' puts a synthetic room behind this window.",
                Some(_) if state == "scanning" => "scanning every channel -- this takes a few seconds",
                Some(_) => "nothing heard. Rescan, or the room is empty.",
            };
            for (k, line) in wrap(why, (body.w / cw.max(1)) as usize).iter().enumerate() {
                theme::text_over_at(fb, body.x, body.y + (k as u32) * (lh + 2), line, theme::TEXT_DIM, 1);
            }
            draw_foot(fb, foot, &self.pass, &self.said, self.held.get());
            return;
        }

        let sel = self.sel.get().min(nets.len() - 1);
        let max_scroll = nets.len().saturating_sub(rows);
        let mut scroll = self.scroll.get().min(max_scroll);
        if sel < scroll {
            scroll = sel;
        } else if rows > 0 && sel >= scroll + rows {
            scroll = sel + 1 - rows;
        }
        self.scroll.set(scroll);
        self.sel.set(sel);

        for k in 0..rows {
            let i = scroll + k;
            let Some(n) = nets.get(i) else { break };
            let y = body.y + (k as u32) * rh;
            let on = joined == Some(n.bssid);
            if i == sel {
                fb.rect(body.x, y, body.w, rh, theme::SELECT);
            }
            let fg = if i == sel {
                theme::SELECT_TEXT
            } else if on {
                theme::OK_TEXT
            } else {
                theme::TEXT
            };
            let cell = |c: u32, w: u32, s: &str| put(c, w, y + 3, s, fg);
            // A joined network is marked in the name column rather than by
            // colour alone, because the selected row overrides the colour and
            // the two states have to be readable at once.
            let name = if on {
                alloc::format!("* {}", n.ssid)
            } else {
                alloc::format!("  {}", n.ssid)
            };
            cell(cols[0].0, cols[0].1, &name);
            cell(cols[1].0, cols[1].1, &bars_of(n.rssi));
            cell(cols[2].0, cols[2].1, &alloc::format!("{}", n.channel));
            cell(cols[3].0, cols[3].1, n.band());
            cell(cols[4].0, cols[4].1, n.security());
            cell(cols[5].0, cols[5].1, &n.ap());
            cell(
                cols[6].0,
                cols[6].1,
                if on { ip.as_deref().unwrap_or("--") } else { "--" },
            );
        }

        draw_foot(fb, foot, &self.pass, &self.said, self.held.get());
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd;
        let n = Self::nets().len();
        match k {
            kbd::KEY_UP => {
                self.sel.set(self.sel.get().saturating_sub(1));
            }
            kbd::KEY_DOWN => {
                if n > 0 {
                    self.sel.set((self.sel.get() + 1).min(n - 1));
                }
            }
            kbd::KEY_HOME => self.sel.set(0),
            kbd::KEY_END => self.sel.set(n.saturating_sub(1)),
            b'\n' | b'\r' => self.act("Join"),
            0x08 | 0x7F => {
                self.pass.pop();
            }
            0x1B => {
                self.pass.clear();
                self.said.clear();
            }
            // Everything printable is passphrase. There is no mode and no
            // focus ring to lose: a window whose typing goes somewhere
            // different depending on an invisible state is one where the
            // passphrase ends up in the wrong place.
            c if (0x20..0x7F).contains(&c) => self.pass.push(c as char),
            _ => return false,
        }
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let (_, table, foot) = Self::layout(client);
        for (i, (r, label)) in Self::buttons(foot).iter().enumerate() {
            if inside(*r, x, y) {
                self.held.set(i);
                let label = *label;
                self.act(label);
                return true;
            }
        }
        self.held.set(usize::MAX);
        if inside(table, x, y) {
            let inner = table.shrink(3);
            let lh = theme::text_h_at(1);
            let body_y = inner.y + lh + 3;
            if y >= body_y as i32 {
                let row = ((y - body_y as i32).max(0) as u32 / Self::row_h()) as usize;
                let i = self.scroll.get() + row;
                if i < Self::nets().len() {
                    self.sel.set(i);
                }
            }
            return true;
        }
        false
    }

    fn release(&mut self) -> bool {
        if self.held.get() == usize::MAX {
            return false;
        }
        self.held.set(usize::MAX);
        true
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let s = self.scroll.get() as i32 - notches;
        self.scroll.set(s.max(0) as usize);
        true
    }
}

fn inside(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32
}

/// Four bars, the way everybody already reads a signal.
///
/// The thresholds are `wifi::bars`, shared rather than repeated: a window and
/// a shell disagreeing about how many bars -75 dBm is would be two answers to
/// one question.
fn bars_of(rssi: i16) -> String {
    let n = wifi::bars(rssi);
    let mut s = String::with_capacity(8);
    for b in 0..4 {
        s.push(if b < n { '#' } else { '.' });
    }
    s.push(' ');
    s
}

fn wrap(s: &str, cols: usize) -> Vec<&str> {
    let mut out = Vec::new();
    if cols == 0 {
        return out;
    }
    let mut rest = s;
    while !rest.is_empty() {
        if rest.chars().count() <= cols {
            out.push(rest);
            break;
        }
        // Break at the last space that fits, so a line does not end mid-word.
        let mut cut = 0;
        let mut last = 0;
        for (i, c) in rest.char_indices() {
            if i > 0 && rest[..i].chars().count() > cols {
                break;
            }
            cut = i;
            if c == ' ' {
                last = i;
            }
        }
        let at = if last > 0 { last } else { cut.max(1) };
        out.push(rest[..at].trim_end());
        rest = rest[at..].trim_start();
        if out.len() > 6 {
            break;
        }
    }
    out
}

fn draw_foot(fb: &Framebuffer, foot: Rect, pass: &str, said: &str, held: usize) {
    if foot.is_empty() {
        return;
    }
    let lh = theme::text_h_at(1);
    let btns = NetMan::buttons(foot);
    for (i, (r, label)) in btns.iter().enumerate() {
        theme::button_at(fb, *r, label, false, held == i, 1);
    }
    // The field runs from the left edge to the leftmost button.
    let stop = btns[2].0.x.saturating_sub(10);
    let field = Rect::new(foot.x, foot.y + 4, stop.saturating_sub(foot.x), lh + 10);
    if !field.is_empty() {
        fb.rect(field.x, field.y, field.w, field.h, theme::SELECT_TEXT);
        fb.rect(field.x + 1, field.y + 1, field.w - 2, field.h - 2, theme::HILIGHT);
        // Dots, never the passphrase. Somebody is standing behind you, and a
        // manager that echoes it is the reason they know it now.
        let mut shown = String::from("passphrase  ");
        for _ in pass.chars() {
            shown.push('*');
        }
        let room = (field.w / theme::text_w_at(1, 1).max(1)) as usize;
        theme::text_over_at(
            fb,
            field.x + 4,
            field.y + 5,
            theme::tail_chars(&shown, room.saturating_sub(1)),
            if pass.is_empty() { theme::TEXT_DIM } else { theme::TEXT },
            1,
        );
    }
    if !said.is_empty() {
        let y = foot.y + foot.h.saturating_sub(lh);
        let room = (foot.w / theme::text_w_at(1, 1).max(1)) as usize;
        theme::text_over_at(fb, foot.x, y, theme::head_chars(said, room), theme::TEXT_DIM, 1);
    }
}
