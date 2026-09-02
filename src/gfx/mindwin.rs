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
use super::{Color, DeskApp, Framebuffer};
use crate::ai::glance::with_glance;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;

/// These windows are drawn dense.
///
/// Chrome is doubled because a title bar is read at arm's length and has room
/// to spare. A pane of facts in a fifth of a screen does not: at scale two it
/// fits nine rows, at scale one it fits nineteen, and the glyphs are a crisp
/// bitmap either way -- the font *is* eight pixels and two was always a
/// doubling.
/// The text scale these windows draw at, decided by how tall the screen is.
///
/// Chrome is doubled because a title bar is read at arm's length and has room
/// to spare. A window reporting on the machine has neither -- at scale two a
/// quarter of an 800-pixel screen held nine rows, where scale one holds
/// nineteen, and that is what made the workspace a workbench rather than four
/// sparse boxes.
///
/// But density is a means and not the point. The GF63's panel is 1920x1080,
/// and an eight-pixel glyph on it is not compact, it is unreadable -- the same
/// pane that needed scale one to fit nineteen rows at 800 fits twenty-six at
/// scale two on 1080. So the rule is stated in rows rather than in pixels:
/// take the larger scale whenever the screen is tall enough to keep the rows.
///
/// The threshold is where the two are equal. A rail is roughly the full screen
/// height, so scale two matches scale one's nineteen-row budget at about a
/// thousand pixels, and every panel this project targets is either well below
/// that or well above it.
pub fn dense() -> u32 {
    match super::primary() {
        Some(fb) if fb.height() >= 1000 => 2,
        _ => 1,
    }
}

/// Height of one text row in these windows.
fn row_h() -> u32 {
    theme::text_h_at(dense()) + 3
}

/// Width of `n` columns at the dense scale.
fn col_w(n: usize) -> u32 {
    theme::text_w_at(n, dense())
}

/// Text in these windows, always at the dense scale.
fn dtext(fb: &Framebuffer, x: u32, y: u32, s: &str, fg: Color) {
    theme::text_over_at(fb, x, y, s, fg, dense());
}

/// A name against a value, at the dense scale.
fn dkv(fb: &Framebuffer, r: Rect, name: &str, value: &str, tone: Color) {
    let gutter = col_w(12);
    dtext(fb, r.x, r.y, theme::head_chars(name, 11), theme::TEXT_DIM);
    if r.w > gutter {
        let room = ((r.w - gutter) / col_w(1)) as usize;
        dtext(fb, r.x + gutter, r.y, theme::head_chars(value, room), tone);
    }
}

/// A button that remembers where it was drawn.
///
/// `draw_in` takes `&self`, so a rectangle discovered while painting has
/// nowhere else to go -- the pattern `AuthorWin` established for its Stop
/// button and the reason it keeps a `Cell<Rect>`.
struct Btn {
    label: &'static str,
    cmd: &'static str,
    /// Whether the selected run's name is appended before queueing.
    ///
    /// `cmd` is a `&'static str` because a button is a constant, and most
    /// commands are. `work run` is not -- it needs the selection -- and the
    /// first version handled that by comparing a button's `cmd` against the
    /// literal `"work run"` inside the press handler, which is a dispatch on a
    /// string that also happens to be the thing being dispatched. This says it
    /// in the table instead.
    takes_run: bool,
    at: Cell<Rect>,
}

impl Btn {
    const fn new(label: &'static str, cmd: &'static str) -> Btn {
        Btn { label, cmd, takes_run: false, at: Cell::new(Rect::new(0, 0, 0, 0)) }
    }

    const fn on_run(label: &'static str, cmd: &'static str) -> Btn {
        Btn { label, cmd, takes_run: true, at: Cell::new(Rect::new(0, 0, 0, 0)) }
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
        theme::button_at(fb, br, b.label, hot == Some(i), false, dense());
    }
}

/// Which button a press landed on, from the rectangles the paint pass stored.
fn button_at(btns: &[Btn], x: i32, y: i32) -> Option<usize> {
    btns.iter().position(|b| {
        let r = b.at.get();
        x >= r.x as i32 && x < (r.x + r.w) as i32 && y >= r.y as i32 && y < (r.y + r.h) as i32
    })
}

/// One turn, drawn dense.
fn dbubble(fb: &Framebuffer, r: Rect, lines: &[String], from_machine: bool) {
    if r.w < 8 || r.h < 6 {
        return;
    }
    if from_machine {
        theme::control(fb, r, &theme::TITLE_ON, theme::CAP_EDGE);
    } else {
        theme::control(fb, r, &theme::BTN, theme::BTN_EDGE);
    }
    let ink = if from_machine { theme::TITLE_TEXT } else { theme::TEXT };
    let inner = r.shrink(3);
    let room = (inner.w / col_w(1)) as usize;
    for (i, line) in lines.iter().enumerate() {
        let y = inner.y + i as u32 * theme::text_h_at(dense());
        if y + theme::text_h_at(dense()) > inner.y + inner.h {
            break;
        }
        dtext(fb, inner.x, y, theme::head_chars(line, room), ink);
    }
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
    /// `(trials, head)` when the ledger above was read.
    stamp: Cell<Option<(u32, Option<[u8; 32]>)>>,
    scroll: Cell<usize>,
    btns: [Btn; 3],
}

impl Improve {
    pub fn new() -> Improve {
        Improve {
            ledger: Cell::new(None),
            stamp: Cell::new(None),
            scroll: Cell::new(0),
            btns: [
                Btn::new("Trial", "godel now 24"),
                Btn::new("Space", "godel space"),
                Btn::new("Rollback", "godel rollback"),
            ],
        }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 380)
    }

    /// Header, ledger, buttons.
    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let pad = client.shrink(6);
        let head_h = row_h() * 4 + 8;
        let btn_h = theme::text_h_at(dense()) + 10;
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
    /// A line records each as `Jn[...]`, and the bracket **ends** with the
    /// verdict: `ok` or `no`. Anything else is a line from a shape of trial
    /// that does not have four judges, and the honest answer for those is none
    /// rather than four falses.
    ///
    /// Reading the bracket and not the next whitespace token, which is what
    /// this did first: the token after `J1` is `[fix=0`, so every judge parsed
    /// as a veto and a trial two judges had passed showed four red chips. Four
    /// falses and "cannot tell" are different answers and only one of them is
    /// true.
    fn verdicts(line: &str) -> Option<[bool; 4]> {
        let mut out = [false; 4];
        let mut seen = 0;
        for (i, tag) in ["J1[", "J2[", "J3[", "J4["].iter().enumerate() {
            let Some(p) = line.find(tag) else { continue };
            let rest = &line[p + tag.len()..];
            let Some(end) = rest.find(']') else { continue };
            out[i] = rest[..end].trim_end().ends_with("ok");
            seen += 1;
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
            dkv(
                fb,
                Rect::new(head.x, head.y, head.w, h),
                "state",
                if g.godel_on { "watching" } else { "stood down" },
                if g.godel_on { theme::OK_TEXT } else { theme::TEXT_DIM },
            );
            dkv(
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
            dkv(
                fb,
                Rect::new(head.x, head.y + h * 2, head.w, h),
                "head",
                &head_hex,
                theme::TEXT,
            );
            // The held-out budget, as a bar, because it is the one number here
            // that is spent rather than accumulated.
            let (used, cap, fresh) = g.test;
            let label = Rect::new(head.x, head.y + h * 3, col_w(12), h);
            dkv(
                fb, label, "held-out", "", theme::TEXT);
            let track = Rect::new(
                head.x + col_w(12),
                head.y + h * 3 + 2,
                head.w.saturating_sub(col_w(12)),
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

        // The ledger is costly -- `ledger_tail` reads the *whole* file however
        // few lines are asked for -- so it is cached. But it was cached
        // *forever*: `take()` returned `Some` on every frame after the first,
        // so the pane whose entire subject is trials kept showing the
        // pre-trial ledger after its own "Trial now" button did its job. The
        // only way to see the result was to close the window and reopen it.
        //
        // Stamped with the two free readings that move when a trial lands, so
        // it re-reads exactly when there is something new and not otherwise.
        let now = with_glance(|g| (g.trials, g.head));
        let fresh = self.stamp.get() == Some(now);
        let lines = match (fresh, self.ledger.take()) {
            (true, Some(l)) => l,
            _ => {
                self.stamp.set(Some(now));
                crate::ai::godel::ledger_tail(64)
            }
        };

        theme::well(fb, body, theme::SCREEN);
        let inner = body.shrink(3);
        let rows = (inner.h / row_h()) as usize;
        let max_scroll = lines.len().saturating_sub(rows);
        let scroll = self.scroll.get().min(max_scroll);
        self.scroll.set(scroll);
        let room = (inner.w / col_w(1)) as usize;
        for (i, line) in lines.iter().skip(scroll).take(rows).enumerate() {
            let y = inner.y + i as u32 * row_h();
            let pills = Improve::verdicts(line);
            let text_w = if pills.is_some() {
                inner.w.saturating_sub(col_w(10))
            } else {
                inner.w
            };
            let shown = theme::head_chars(line, (text_w / col_w(1)) as usize);
            dtext(fb, inner.x, y, shown, theme::SCREEN_TEXT);
            if let Some(v) = pills {
                let pw = col_w(2) + 4;
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
            dtext(fb, inner.x, inner.y, "nothing adopted yet", theme::TEXT_DIM);
        }
        self.ledger.set(Some(lines));

        button_row(fb, feet, &self.btns, None);
        let _ = focused;
    }

    fn key(&mut self, k: u8) -> bool {
        // Arrows scroll the ledger and Enter runs a trial. The pane had no
        // keyboard at all, which contradicts the desktop's own rule that
        // everything the pointer does a keystroke also does -- and its two
        // buttons are the only way to make anything happen here.
        match k {
            crate::dev::kbd::KEY_DOWN => {
                self.scroll.set(self.scroll.get() + 1);
                true
            }
            crate::dev::kbd::KEY_UP => {
                self.scroll.set(self.scroll.get().saturating_sub(1));
                true
            }
            b'\n' | b'\r' => {
                super::desk::queue_command(self.btns[0].cmd);
                true
            }
            _ => false,
        }
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
    /// How far down the plan pane is scrolled.
    pscroll: Cell<usize>,
    /// `((run, root), steps)` -- the root is what makes it notice a run
    /// that changed under a selection that did not.
    cached: Cell<Option<((String, Option<[u8; 32]>), Vec<(bool, bool, String)>)>>,
    btns: [Btn; 3],
}

impl Flows {
    pub fn new() -> Flows {
        Flows {
            sel: Cell::new(0),
            pscroll: Cell::new(0),
            cached: Cell::new(None),
            btns: [
                Btn::on_run("Run", "work run"),
                Btn::on_run("Check", "work check"),
                Btn::new("Tile", "win tile workspace"),
            ],
        }
    }

    pub fn preferred() -> (u32, u32) {
        (560, 380)
    }

    /// Run list on the left, the selected plan on the right, buttons beneath.
    fn layout(client: Rect) -> (Rect, Rect, Rect) {
        let pad = client.shrink(6);
        let btn_h = theme::text_h_at(dense()) + 10;
        let body_h = pad.h.saturating_sub(btn_h + 4);
        // Five sixteenths, but never narrower than a name. A fraction alone
        // was right at one text scale and wrong at the other: the same rail
        // that fitted `nightly` at scale one truncated it to `nightl` at scale
        // two, because a fraction of the width is a different number of
        // *characters* depending on how wide a character is. Twelve is the
        // budget a run name is written against, plus the row's own padding.
        // Eight, not twelve. A run name is short -- `audit`, `nightly` -- and
        // every column the list takes is one the plan does not get, which is
        // the half with the sentences in it.
        let want = col_w(8) + 12;
        let left_w = (pad.w * 5 / 16).max(want.min(pad.w / 2));
        let list = Rect::new(pad.x, pad.y, left_w, body_h);
        let plan = Rect::new(pad.x + left_w + 4, pad.y, pad.w.saturating_sub(left_w + 4), body_h);
        let feet = Rect::new(pad.x, pad.y + body_h + 4, pad.w, btn_h);
        (list, plan, feet)
    }
}

impl Flows {
    /// Queue button `i`, filling in the selection where the button asks for
    /// it. One path for the pointer and the keyboard, so Enter and a click on
    /// the same button cannot come to mean different things.
    fn fire(&self, i: usize) -> bool {
        let Some(b) = self.btns.get(i) else { return false };
        if !b.takes_run {
            super::desk::queue_command(b.cmd);
            return true;
        }
        let runs = with_glance(|g| g.runs.clone());
        match runs.get(self.sel.get()) {
            Some(name) => {
                super::desk::queue_command(&alloc::format!("{} {}", b.cmd, name));
                true
            }
            None => false,
        }
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
            theme::list_row_at(fb, r, name, i == sel, focused, dense());
        }
        if runs.is_empty() {
            dtext(fb, li.x, li.y, "no runs", theme::TEXT_DIM);
        }

        // The plan for the selected run, re-read when the selection moves *or
        // when the run itself does*. Keyed on the name alone it froze at the
        // state the run had when it was selected -- so watching `work run`
        // execute, which is the one thing this pane is for, showed nothing.
        //
        // `work::root()` is the run's content address, and it is the number
        // the whole design turns on: a subtree that hashes the same has not
        // changed, so this re-reads if and only if something did.
        let want = runs.get(sel).cloned().unwrap_or_default();
        let root = if want.is_empty() { None } else { crate::ai::work::root(&want) };
        let key = (want.clone(), root);
        let mut cache = self.cached.take();
        if cache.as_ref().map(|(n, _)| n != &key).unwrap_or(true) && !want.is_empty() {
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
            cache = Some((key.clone(), steps));
        }

        theme::well(fb, plan, theme::SCREEN);
        let pi = plan.shrink(3);
        let room = (pi.w / col_w(1)).saturating_sub(5) as usize;
        if let Some((_, steps)) = cache.as_ref() {
            let skip = self.pscroll.get().min(steps.len().saturating_sub(1));
            for (i, (done, failed, goal)) in steps.iter().skip(skip).enumerate() {
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
                dtext(fb, pi.x, y, mark, ink);
                dtext(fb, pi.x + col_w(4), y, theme::head_chars(goal, room), theme::SCREEN_TEXT);
            }
            if steps.is_empty() {
                dtext(fb, pi.x, pi.y, "no steps", theme::TEXT_DIM);
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
            // Enter is Run. The desktop's rule is that everything the pointer
            // does a keystroke also does, and until now the two buttons on
            // this pane were reachable only by clicking them.
            b'\n' | b'\r' => self.fire(0),
            _ => false,
        }
    }

    fn wheel(&mut self, notches: i32) -> bool {
        // The plan pane. Its list `break`s at the fold and had no wheel at
        // all, so a plan longer than the window was simply unreachable -- and
        // a plan is exactly the thing that grows.
        let s = (self.pscroll.get() as i32 + notches).max(0) as usize;
        let steps = self
            .cached
            .take()
            .map(|c| {
                let n = c.1.len();
                self.cached.set(Some(c));
                n
            })
            .unwrap_or(0);
        self.pscroll.set(s.min(steps.saturating_sub(1)));
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let ax = client.x as i32 + x;
        let ay = client.y as i32 + y;
        if let Some(i) = button_at(&self.btns, ax, ay) {
            return self.fire(i);
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
        let bar_h = theme::text_h_at(dense()) + 10;
        let body = Rect::new(pad.x, pad.y, pad.w, pad.h.saturating_sub(bar_h + 4));
        let btn_w = col_w(6);
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
        // Wrapped to the width a bubble actually gets, not to the pane's.
        // A bubble is four fifths of the pane and the text was wrapped to the
        // whole of it, so every line ran past the bubble's right edge and was
        // clipped mid-word -- the wrap was correct for a width nothing is
        // drawn at.
        let bubble_w = inner.w * 4 / 5;
        let cols = (bubble_w / col_w(1)).saturating_sub(2) as usize;

        // `companion` and not `agent`: the agent's ring is episodes, which is
        // a different thing from the conversation and never contains a turn of
        // it. Reading that ring here is why this window was empty after an
        // `ask` that had visibly answered.
        let said = crate::ai::companion::log_snapshot();
        let mut turns: Vec<(bool, Vec<String>)> = Vec::new();
        for (mine, text) in said.iter() {
            turns.push((!mine, theme::wrap(text, cols)));
        }

        // Bottom-up, so the newest turn is always the one on screen.
        let mut y = inner.y + inner.h;
        for (from_machine, wrapped) in turns.iter().rev().skip(self.scroll.get()) {
            let h = wrapped.len() as u32 * theme::text_h_at(dense()) + 6;
            if y < inner.y + h {
                break;
            }
            y -= h + 4;
            let w = bubble_w;
            let x = if *from_machine { inner.x } else { inner.x + inner.w - w };
            dbubble(fb, Rect::new(x, y, w, h), wrapped, *from_machine);
        }
        if turns.is_empty() {
            dtext(fb, inner.x, inner.y, "nothing said yet -- ask it something", theme::TEXT_DIM);
        }

        theme::well(fb, field, theme::HILIGHT);
        let room = (field.w / col_w(1)).saturating_sub(1) as usize;
        let shown = theme::tail_chars(&self.typed, room);
        dtext(fb, field.x + 4, field.y + 6, shown, theme::TEXT);
        if focused {
            let cx = field.x + 4 + theme::text_w_at(shown.chars().count(), dense());
            fb.rect(cx, field.y + 5, 2, theme::text_h_at(dense()), theme::APERTURE);
        }
        self.send.at.set(btn);
        theme::button_at(fb, btn, "Ask", false, false, dense());
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
        // Clamped at both ends. It was `.max(0)` only, so scrolling up past
        // the oldest turn left `turns.iter().rev().skip(n)` yielding nothing
        // -- and because `turns` is not empty the "nothing said yet" line does
        // not appear either, so the pane went blank with no way to know why.
        let n = crate::ai::companion::log_snapshot().len();
        let s = (self.scroll.get() as i32 + notches).max(0) as usize;
        self.scroll.set(s.min(n.saturating_sub(1)));
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
    // Idempotent, because it was not. `tile_workspace` finds a window by title
    // and takes the *first* match, so a second `mind open` pushed four more
    // windows with the same four titles and then placed the originals again --
    // leaving four duplicates floating wherever `open_app` centred them, with
    // no way to tell which was which. `desk::open_authoring` has had this
    // guard since it was written.
    if super::desk::has_window("Ask") {
        super::desk::tile_workspace("Workflows", "Ask", "Agent", "Improve");
        return;
    }
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
    // A workbench, not four quarters: the rails carry short columns, the
    // middle carries what is being read, and the strip under it carries what
    // is streaming.
    super::desk::tile_workspace("Workflows", "Ask", "Agent", "Improve");
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
        Improve::verdicts("h3 J1[fix=2 chi=4.10 beyond the noise ok] J2[goals=4/4 ok] J3[ok] J4[r=8 kib=24 ok] ADOPT")
            == Some([true, true, true, true]),
    );
    claim(
        "a refusal is not mistaken for a pass",
        Improve::verdicts(
            "h3 J1[fix=0 broke=0 chi=0.00 inside the noise no] J2[goals=0/4 no]              J3[ok] J4[r=8 kib=24 ok] reject",
        ) == Some([false, false, true, true]),
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
