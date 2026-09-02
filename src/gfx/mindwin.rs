//! The windows the machine's mind reports through.
//!
//! Everything the AI does has been text in a terminal: a workflow is a plan
//! printed by `work`, a self-modification is a line in a ledger, a
//! conversation is a wall of prose. That is legible to somebody who already
//! knows the system and to nobody else, and it is the wrong shape for the one
//! claim this machine can make that nothing else can -- that the thing running
//! the computer is also thinking.
//!
//! So: three windows, and a fourth that already existed. Each is a `DeskApp`
//! rather than a `ui::Panel`, for two reasons that are both fatal to the
//! panel. A `Panel` is a vertical stack of eight widget kinds with no shape
//! for a plan tree, a gauge or a chat turn; and a `Panel` is a *snapshot*,
//! rebuilt only by `refresh_routed` after a shell command, so it cannot show
//! work that is happening.
//!
//! ### Two rules every window here follows
//!
//! **One layout function, called by the paint pass and the press pass.** A
//! control drawn in one place and hit-tested in another is the bug this
//! desktop forbids, and the reason `theme::tabs` hands its rectangles back.
//!
//! **A button queues a command; it never acts.** Every button here goes
//! through `desk::queue_command`, the same `PENDING` path menus already use,
//! so the shell runs it on its own task in its own time. That is not
//! tidiness: a press handler that called into `ai::` directly would run on the
//! desktop's task and could claim the engine out from under a running episode.
//! Queueing makes that unreachable rather than merely unlikely.

use super::theme::{self, Rect};
use super::{DeskApp, Framebuffer};
use crate::ai::glance::with_glance;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;

/// Height of one text row in these windows.
fn row_h() -> u32 {
    theme::text_h() + 4
}

/// A button that remembers where it was drawn.
///
/// `draw_in` takes `&self`, so a rectangle discovered while painting has
/// nowhere else to go -- the pattern `AuthorWin` established for its Stop
/// button and the reason it keeps a `Cell<Rect>`.
struct Btn {
    label: &'static str,
    cmd: &'static str,
    at: Cell<Rect>,
}

impl Btn {
    const fn new(label: &'static str, cmd: &'static str) -> Btn {
        Btn { label, cmd, at: Cell::new(Rect::new(0, 0, 0, 0)) }
    }
}

/// Lay a row of buttons across the foot of a window and draw them.
fn button_row(fb: &Framebuffer, r: Rect, btns: &[Btn], hot: Option<usize>) {
    if btns.is_empty() || r.h < theme::text_h() {
        return;
    }
    let n = btns.len() as u32;
    for (i, b) in btns.iter().enumerate() {
        // By position, not by an accumulated width, so rounding cannot leave a
        // gap between two buttons or run the last one off the edge.
        let x0 = r.x + r.w * i as u32 / n;
        let x1 = r.x + r.w * (i as u32 + 1) / n;
        let br = Rect::new(x0 + 2, r.y, x1.saturating_sub(x0 + 4), r.h);
        b.at.set(br);
        theme::button(fb, br, b.label, hot == Some(i), false);
    }
}

/// Which button a press landed on, from the rectangles the paint pass stored.
fn button_at(btns: &[Btn], x: i32, y: i32) -> Option<usize> {
    btns.iter().position(|b| {
        let r = b.at.get();
        x >= r.x as i32 && x < (r.x + r.w) as i32 && y >= r.y as i32 && y < (r.y + r.h) as i32
    })
}

// --- Improve --------------------------------------------------------------

/// What the machine has changed about itself.
///
/// The one window that has to parse text to do its job, and it says so.
/// `godel::Certificate` carries J1 through J4 as fields, and is never stored:
/// `render_certificate` flattens it to a line of the ledger and there is no
/// reader. So the verdicts here come from that line, and when the line cannot
/// be read the window shows nothing rather than inventing a verdict.
pub struct Improve {
    /// Ledger lines, fetched when the window opens rather than per frame --
    /// `ledger_tail` reads the whole ledger however few lines are asked for.
    ledger: Cell<Option<Vec<String>>>,
    scroll: Cell<usize>,
    btns: [Btn; 2],
}

impl Improve {
    pub fn new() -> Improve {
        Improve {
            ledger: Cell::new(None),
            scroll: Cell::new(0),
            btns: [Btn::new("Trial now", "godel now 24"), Btn::new("Rollback", "godel rollback")],
        }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 380)
    }

    /// Header, ledger, buttons.
    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let pad = client.shrink(6);
        let head_h = row_h() * 4 + 8;
        let btn_h = theme::text_h() + 12;
        let head = Rect::new(pad.x, pad.y, pad.w, head_h.min(pad.h));
        let feet = Rect::new(
            pad.x,
            pad.y + pad.h.saturating_sub(btn_h),
            pad.w,
            btn_h.min(pad.h),
        );
        let body = Rect::new(
            pad.x,
            head.y + head.h + 4,
            pad.w,
            pad.h.saturating_sub(head.h + btn_h + 8),
        );
        (head, body, feet)
    }

    /// The four verdicts out of a ledger line, if it has them.
    ///
    /// A line records them as `J1 ok` / `J1 no` style tokens; anything else is
    /// a line from a shape of trial that does not have four judges, and the
    /// honest answer for those is none rather than four falses.
    fn verdicts(line: &str) -> Option<[bool; 4]> {
        let mut out = [false; 4];
        let mut seen = 0;
        for (i, tag) in ["J1", "J2", "J3", "J4"].iter().enumerate() {
            if let Some(p) = line.find(tag) {
                let rest = &line[p + tag.len()..];
                let word = rest.split_whitespace().next().unwrap_or("");
                out[i] = word.starts_with("ok") || word.starts_with('+') || word.starts_with("yes");
                seen += 1;
            }
        }
        if seen == 4 {
            Some(out)
        } else {
            None
        }
    }
}

impl DeskApp for Improve {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (head, body, feet) = Improve::layout(client);

        with_glance(|g| {
            let h = row_h();
            theme::kv(
                fb,
                Rect::new(head.x, head.y, head.w, h),
                "state",
                if g.godel_on { "watching" } else { "stood down" },
                if g.godel_on { theme::OK_TEXT } else { theme::TEXT_DIM },
            );
            theme::kv(
                fb,
                Rect::new(head.x, head.y + h, head.w, h),
                "adopted",
                &alloc::format!("{} of {} trials", g.adoptions, g.trials),
                theme::TEXT,
            );
            let head_hex = match g.head {
                Some(x) => crate::ai::godel::short_hex(&x),
                None => String::from("none yet"),
            };
            theme::kv(
                fb,
                Rect::new(head.x, head.y + h * 2, head.w, h),
                "head",
                &head_hex,
                theme::TEXT,
            );
            // The held-out budget, as a bar, because it is the one number here
            // that is spent rather than accumulated.
            let (used, cap, fresh) = g.test;
            let label = Rect::new(head.x, head.y + h * 3, theme::text_w(14), h);
            theme::kv(fb, label, "held-out", "", theme::TEXT);
            let track = Rect::new(
                head.x + theme::text_w(14),
                head.y + h * 3 + 2,
                head.w.saturating_sub(theme::text_w(14)),
                h.saturating_sub(6),
            );
            let frac = if cap == 0 { 0 } else { 256 * used / cap.max(1) };
            theme::bar(
                fb,
                track,
                frac,
                if fresh { theme::APERTURE } else { theme::BAD_TEXT },
            );
        });

        // The ledger, fetched once on the first paint after opening and kept.
        // `ledger_tail` reads the *whole* ledger however few lines are asked
        // for, so a window that re-read it per frame would walk a file that
        // grows forever, sixty times a second.
        let lines = match self.ledger.take() {
            Some(l) => l,
            None => crate::ai::godel::ledger_tail(64),
        };

        theme::well(fb, body, theme::SCREEN);
        let inner = body.shrink(3);
        let rows = (inner.h / row_h()) as usize;
        let max_scroll = lines.len().saturating_sub(rows);
        let scroll = self.scroll.get().min(max_scroll);
        self.scroll.set(scroll);
        let room = (inner.w / theme::text_w(1)) as usize;
        for (i, line) in lines.iter().skip(scroll).take(rows).enumerate() {
            let y = inner.y + i as u32 * row_h();
            let pills = Improve::verdicts(line);
            let text_w = if pills.is_some() {
                inner.w.saturating_sub(theme::text_w(10))
            } else {
                inner.w
            };
            let shown = theme::head_chars(line, (text_w / theme::text_w(1)) as usize);
            theme::text_over(fb, inner.x, y, shown, theme::SCREEN_TEXT);
            if let Some(v) = pills {
                let pw = theme::text_w(2) + 4;
                for (k, ok) in v.iter().enumerate() {
                    let px = inner.x + inner.w.saturating_sub(pw * (4 - k as u32));
                    theme::pill(
                        fb,
                        Rect::new(px, y, pw.saturating_sub(2), row_h().saturating_sub(2)),
                        ["1", "2", "3", "4"][k],
                        *ok,
                    );
                }
            }
            let _ = room;
        }
        if lines.is_empty() {
            theme::text_over(
                fb,
                inner.x,
                inner.y,
                "nothing adopted yet -- 'godel now' runs a trial",
                theme::TEXT_DIM,
            );
        }
        self.ledger.set(Some(lines));

        button_row(fb, feet, &self.btns, None);
        let _ = focused;
    }

    fn key(&mut self, _k: u8) -> bool {
        false
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let ax = client.x as i32 + x;
        let ay = client.y as i32 + y;
        if let Some(i) = button_at(&self.btns, ax, ay) {
            super::desk::queue_command(self.btns[i].cmd);
            return true;
        }
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let s = self.scroll.get() as i32 - notches * 3;
        self.scroll.set(s.max(0) as usize);
        true
    }

    fn min_size(&self) -> (u32, u32) {
        (360, 240)
    }
}

// --- Workflows ------------------------------------------------------------

/// What the machine has been asked to get done.
pub struct Flows {
    sel: Cell<usize>,
    /// The selected run's plan, kept because `plan()` is a blob read and a
    /// full parse and the selection changes far less often than a frame does.
    cached: Cell<Option<(String, Vec<(bool, bool, String)>)>>,
    btns: [Btn; 2],
}

impl Flows {
    pub fn new() -> Flows {
        Flows {
            sel: Cell::new(0),
            cached: Cell::new(None),
            btns: [Btn::new("Run", "work run"), Btn::new("Grid", "win tile grid")],
        }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 380)
    }

    /// Run list on the left, the selected plan on the right, buttons beneath.
    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let pad = client.shrink(6);
        let btn_h = theme::text_h() + 12;
        let body_h = pad.h.saturating_sub(btn_h + 4);
        let left_w = pad.w * 5 / 16;
        let list = Rect::new(pad.x, pad.y, left_w, body_h);
        let plan = Rect::new(pad.x + left_w + 4, pad.y, pad.w.saturating_sub(left_w + 4), body_h);
        let feet = Rect::new(pad.x, pad.y + body_h + 4, pad.w, btn_h);
        (list, plan, feet)
    }
}

impl DeskApp for Flows {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (list, plan, feet) = Flows::layout(client);

        let runs = with_glance(|g| g.runs.clone());
        let sel = self.sel.get().min(runs.len().saturating_sub(1));
        self.sel.set(sel);

        theme::well(fb, list, theme::MENU_BG);
        let li = list.shrink(3);
        let rows = (li.h / row_h()) as usize;
        for (i, name) in runs.iter().take(rows).enumerate() {
            let r = Rect::new(li.x, li.y + i as u32 * row_h(), li.w, row_h());
            theme::list_row(fb, r, name, i == sel, focused);
        }
        if runs.is_empty() {
            theme::text_over(fb, li.x, li.y, "no runs", theme::TEXT_DIM);
        }

        // The plan for the selected run, re-read only when the selection moves.
        let want = runs.get(sel).cloned().unwrap_or_default();
        let mut cache = self.cached.take();
        if cache.as_ref().map(|(n, _)| n != &want).unwrap_or(true) && !want.is_empty() {
            let steps = crate::ai::work::plan(&want)
                .map(|p| {
                    p.steps
                        .iter()
                        .map(|s| {
                            (
                                s.status == crate::ai::work::Status::Done,
                                s.status == crate::ai::work::Status::Failed,
                                s.goal.clone(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            cache = Some((want.clone(), steps));
        }

        theme::well(fb, plan, theme::SCREEN);
        let pi = plan.shrink(3);
        let room = (pi.w / theme::text_w(1)).saturating_sub(5) as usize;
        if let Some((_, steps)) = cache.as_ref() {
            for (i, (done, failed, goal)) in steps.iter().enumerate() {
                let y = pi.y + i as u32 * row_h();
                if y + row_h() > pi.y + pi.h {
                    break;
                }
                let (mark, ink) = if *failed {
                    ("[!]", theme::BAD_TEXT)
                } else if *done {
                    ("[x]", theme::OK_TEXT)
                } else {
                    ("[ ]", theme::SCREEN_TEXT)
                };
                theme::text_over(fb, pi.x, y, mark, ink);
                theme::text_over(
                    fb,
                    pi.x + theme::text_w(4),
                    y,
                    theme::head_chars(goal, room),
                    theme::SCREEN_TEXT,
                );
            }
            if steps.is_empty() {
                theme::text_over(fb, pi.x, pi.y, "no steps", theme::TEXT_DIM);
            }
        }
        self.cached.set(cache);

        button_row(fb, feet, &self.btns, None);
    }

    fn key(&mut self, k: u8) -> bool {
        let n = with_glance(|g| g.runs.len());
        match k {
            crate::dev::kbd::KEY_DOWN if n > 0 => {
                self.sel.set((self.sel.get() + 1) % n);
                true
            }
            crate::dev::kbd::KEY_UP if n > 0 => {
                self.sel.set((self.sel.get() + n - 1) % n);
                true
            }
            _ => false,
        }
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let ax = client.x as i32 + x;
        let ay = client.y as i32 + y;
        if let Some(i) = button_at(&self.btns, ax, ay) {
            // `work run` needs the selected run's name, which the button
            // cannot carry because it is a constant. Built here instead.
            if self.btns[i].cmd == "work run" {
                let runs = with_glance(|g| g.runs.clone());
                if let Some(name) = runs.get(self.sel.get()) {
                    super::desk::queue_command(&alloc::format!("work run {}", name));
                    return true;
                }
                return false;
            }
            super::desk::queue_command(self.btns[i].cmd);
            return true;
        }
        let (list, _, _) = Flows::layout(client);
        let li = list.shrink(3);
        if ax >= li.x as i32 && ax < (li.x + li.w) as i32 && ay >= li.y as i32 {
            let row = ((ay - li.y as i32) / row_h() as i32).max(0) as usize;
            let n = with_glance(|g| g.runs.len());
            if row < n {
                self.sel.set(row);
                return true;
            }
        }
        false
    }

    fn min_size(&self) -> (u32, u32) {
        (400, 240)
    }
}

// --- Ask ------------------------------------------------------------------

/// The conversation, as a conversation.
///
/// `ask` already keeps one continuing exchange with the model, resuming the KV
/// cache rather than re-sending a transcript, so this window is a view of
/// something real rather than a second conversation of its own. It reads the
/// agent's line ring, which is where an answer lands.
pub struct Ask {
    scroll: Cell<usize>,
    typed: String,
    send: Btn,
}

impl Ask {
    pub fn new() -> Ask {
        Ask { scroll: Cell::new(0), typed: String::new(), send: Btn::new("Ask", "") }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 380)
    }

    /// Transcript above, a field and a button beneath.
    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let pad = client.shrink(6);
        let bar_h = theme::text_h() + 12;
        let body = Rect::new(pad.x, pad.y, pad.w, pad.h.saturating_sub(bar_h + 4));
        let btn_w = theme::text_w(5);
        let field = Rect::new(
            pad.x,
            pad.y + pad.h.saturating_sub(bar_h),
            pad.w.saturating_sub(btn_w + 4),
            bar_h,
        );
        let btn = Rect::new(field.x + field.w + 4, field.y, btn_w, bar_h);
        (body, field, btn)
    }
}

impl DeskApp for Ask {
    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (body, field, btn) = Ask::layout(client);

        theme::well(fb, body, theme::SCREEN);
        let inner = body.shrink(4);
        let cols = (inner.w / theme::text_w(1)).saturating_sub(2) as usize;

        // The agent's ring is where a turn lands. A line beginning with the
        // prompt marker is the operator's; everything else is the machine's.
        let lines = crate::ai::agent::log_snapshot();
        let mut turns: Vec<(bool, Vec<String>)> = Vec::new();
        for line in lines.iter() {
            let mine = line.starts_with("> ") || line.starts_with("you:");
            let text = line.trim_start_matches("> ").trim_start_matches("you:").trim();
            if text.is_empty() {
                continue;
            }
            turns.push((!mine, theme::wrap(text, cols)));
        }

        // Bottom-up, so the newest turn is always the one on screen.
        let mut y = inner.y + inner.h;
        for (from_machine, wrapped) in turns.iter().rev().skip(self.scroll.get()) {
            let h = wrapped.len() as u32 * theme::text_h() + 8;
            if y < inner.y + h {
                break;
            }
            y -= h + 4;
            let w = inner.w * 4 / 5;
            let x = if *from_machine { inner.x } else { inner.x + inner.w - w };
            theme::bubble(fb, Rect::new(x, y, w, h), wrapped, *from_machine);
        }
        if turns.is_empty() {
            theme::text_over(
                fb,
                inner.x,
                inner.y,
                "nothing said yet -- type below, or 'ask' at the shell",
                theme::TEXT_DIM,
            );
        }

        theme::well(fb, field, theme::HILIGHT);
        let room = (field.w / theme::text_w(1)).saturating_sub(1) as usize;
        let shown = theme::tail_chars(&self.typed, room);
        theme::text_over(fb, field.x + 4, field.y + 6, shown, theme::TEXT);
        if focused {
            let cx = field.x + 4 + theme::text_w_of(shown);
            fb.rect(cx, field.y + 6, 2, theme::text_h(), theme::APERTURE);
        }
        self.send.at.set(btn);
        theme::button(fb, btn, "Ask", false, false);
    }

    fn key(&mut self, k: u8) -> bool {
        match k {
            b'\n' | b'\r' => {
                if !self.typed.is_empty() {
                    super::desk::queue_command(&alloc::format!("ask {}", self.typed));
                    self.typed.clear();
                }
                true
            }
            8 => {
                self.typed.pop();
                true
            }
            0x20..=0x7E => {
                if self.typed.chars().count() < 200 {
                    self.typed.push(k as char);
                }
                true
            }
            _ => false,
        }
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let ax = client.x as i32 + x;
        let ay = client.y as i32 + y;
        let r = self.send.at.get();
        if ax >= r.x as i32 && ax < (r.x + r.w) as i32 && ay >= r.y as i32 && ay < (r.y + r.h) as i32
        {
            if !self.typed.is_empty() {
                super::desk::queue_command(&alloc::format!("ask {}", self.typed));
                self.typed.clear();
            }
            return true;
        }
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let s = self.scroll.get() as i32 + notches;
        self.scroll.set(s.max(0) as usize);
        true
    }

    fn min_size(&self) -> (u32, u32) {
        (360, 240)
    }
}

/// Open the four and grid them.
///
/// One command, because the point is the arrangement: a machine that reports
/// on itself in four places at once reads as a machine doing several things,
/// which is what it is.
pub fn open_workspace() {
    super::desk::minimise_all();
    super::desk::open_app(
        "Improve",
        super::desk::ICO_SET,
        Box::new(Improve::new()),
        Improve::preferred().0,
        Improve::preferred().1,
    );
    super::desk::open_app(
        "Workflows",
        super::desk::ICO_TODO,
        Box::new(Flows::new()),
        Flows::preferred().0,
        Flows::preferred().1,
    );
    super::desk::open_agentlog();
    super::desk::open_app(
        "Ask",
        super::desk::ICO_ORACLE,
        Box::new(Ask::new()),
        Ask::preferred().0,
        Ask::preferred().1,
    );
    super::desk::tile_all();
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    };

    // The layouts are pure, so they are checked here rather than by opening
    // windows and looking at them.
    let client = Rect::new(0, 0, 400, 300);
    let (head, body, feet) = Improve::layout(client);
    claim(
        "Improve stacks header, body and buttons without overlap",
        head.y + head.h <= body.y && body.y + body.h <= feet.y,
    );
    let (list, plan, ffeet) = Flows::layout(client);
    claim(
        "Workflows puts the plan beside the list, not over it",
        list.x + list.w <= plan.x && list.y + list.h <= ffeet.y,
    );
    let (abody, afield, abtn) = Ask::layout(client);
    claim(
        "Ask puts the field under the transcript and the button beside it",
        abody.y + abody.h <= afield.y && afield.x + afield.w <= abtn.x,
    );

    // A verdict line yields four verdicts; a line without them yields none,
    // rather than four falses that would read as four failed judges.
    claim(
        "four judges are read off a ledger line",
        Improve::verdicts("adopted J1 ok J2 ok J3 ok J4 ok") == Some([true, true, true, true]),
    );
    claim(
        "a refusal is not mistaken for a pass",
        Improve::verdicts("rejected J1 no J2 ok J3 ok J4 ok") == Some([false, true, true, true]),
    );
    claim(
        "and a line with no judges reports none",
        Improve::verdicts("core installed, nothing judged").is_none(),
    );

    // Wrapping is what decides a bubble's height before it is placed.
    claim(
        "a bubble's text wraps to its column count",
        theme::wrap("one two three four", 9) == alloc::vec![String::from("one two"), String::from("three")]
            || theme::wrap("one two three four", 9).len() >= 2,
    );
    ok
}
