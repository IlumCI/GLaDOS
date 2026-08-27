//! Write. A windowed editor over the namespace.
//!
//! The terminal already has `edit`, a modal editor in the console grid. This
//! is the other kind: a document in a window, cursor keys and typing, no
//! modes. Both edit the same namespace through the same two calls
//! (`sysbox::read_blob` / `write_blob`), so a file started in one opens in
//! the other -- there is one filesystem and these are two views of it.
//!
//! Layout is lines wrapped at the window's width. The wrap is display-only:
//! the document is bytes with newlines, and resizing the window rewraps
//! without touching it. The cursor is a byte offset into the text -- the
//! same representation the shell's line editor and the panel fields use --
//! and every movement key is defined in terms of the wrapped rows the user
//! is actually looking at.
//!
//! Shortcuts are control bytes, because that is what the keyboard driver
//! produces for Ctrl-letter: ^S saves, ^O edits the path, ^N starts a new
//! document. They are printed in the status line permanently; a shortcut
//! only the source code knows is a feature that does not exist.

use super::theme::{self, Rect};
use super::{DeskApp, Framebuffer};
use alloc::format;
use core::cell::Cell;
use alloc::string::String;
use alloc::vec::Vec;

const CTRL_S: u8 = 0x13;
const CTRL_O: u8 = 0x0F;
const CTRL_N: u8 = 0x0E;

pub struct Writer {
    path: String,
    text: String,
    /// Byte offset into `text`. Everything ASCII here, so byte == column.
    cursor: usize,
    /// First wrapped row on screen. A `Cell` because the draw pass owns the
    /// scroll: only it knows the real width and height, so it is the one
    /// place "keep the cursor on screen" can be decided. The Browser keeps
    /// its layout facts the same way.
    scroll: Cell<usize>,
    /// Wrapped width of the last draw, so cursor keys move in the columns
    /// the user is actually looking at rather than a guess.
    cols: Cell<usize>,
    /// Whether the next draw should bring the cursor into view. Edits set
    /// it; wheel scrolling clears it, or reading would be impossible while
    /// the caret is off screen.
    follow: Cell<bool>,
    /// Typing goes to the path well instead of the document.
    editing_path: bool,
    dirty: bool,
    status: String,
}

impl Writer {
    pub fn new(path: &str) -> Self {
        let mut w = Self {
            path: String::from(if path.is_empty() { "/doc/note.txt" } else { path }),
            text: String::new(),
            cursor: 0,
            scroll: Cell::new(0),
            cols: Cell::new(64),
            follow: Cell::new(true),
            editing_path: false,
            dirty: false,
            status: String::new(),
        };
        w.load();
        w
    }

    pub fn preferred() -> (u32, u32) {
        (560, 430)
    }

    fn load(&mut self) {
        match crate::sysbox::read_blob(&self.path) {
            Some(bytes) => {
                // Lossy on purpose: a file with stray bytes should open as a
                // damaged document, not refuse to open at all.
                self.text = String::from_utf8_lossy(&bytes).into_owned();
                self.status = format!("{} B read", self.text.len());
            }
            None => {
                self.text = String::new();
                self.status = String::from("new document");
            }
        }
        self.cursor = 0;
        self.scroll.set(0);
        self.follow.set(true);
        self.dirty = false;
    }

    fn save(&mut self) {
        let n = self.text.len();
        if crate::sysbox::write_blob(&self.path, self.text.as_bytes().to_vec()) {
            self.dirty = false;
            self.status = format!("saved {} B", n);
        } else {
            self.status = String::from("save refused -- is the path a directory?");
        }
    }

    /// The document as wrapped rows: byte range per row.
    ///
    /// Recomputed on demand rather than cached. The document is human-typed
    /// text -- kilobytes -- and a cache is a second copy of the truth that
    /// every edit would have to keep honest.
    fn rows(&self, cols: usize) -> Vec<(usize, usize)> {
        let cols = cols.max(1);
        let mut out = Vec::new();
        let bytes = self.text.as_bytes();
        let mut start = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\n' {
                out.push((start, i));
                start = i + 1;
            } else if i - start >= cols {
                out.push((start, i));
                start = i;
            }
            i += 1;
        }
        out.push((start, bytes.len()));
        out
    }

    /// Which row the cursor is on, and its column.
    fn cursor_pos(&self, cols: usize) -> (usize, usize) {
        for (r, (a, b)) in self.rows(cols).iter().enumerate() {
            // `<=` so a cursor at the very end of a row (including the end of
            // the document) belongs to that row rather than to nowhere.
            if self.cursor >= *a && self.cursor <= *b {
                return (r, self.cursor - a);
            }
        }
        (0, 0)
    }

    fn client_cols(client: Rect) -> usize {
        ((client.w.saturating_sub(20)) / theme::text_w(1).max(1)) as usize
    }

    fn view_rows(client: Rect) -> usize {
        ((client.h.saturating_sub(64)) / (theme::text_h() + 2).max(1)) as usize
    }

    fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += 1;
        self.dirty = true;
        self.follow.set(true);
    }

    /// Path well and text area, shared by paint and hit-test.
    fn metrics(client: Rect) -> (Rect, Rect) {
        let lh = theme::text_h();
        let bar = Rect::new(client.x + 8, client.y + 6, client.w.saturating_sub(16), lh + 8);
        let cap = theme::text_w(5);
        let well = Rect::new(bar.x + cap, bar.y, bar.w.saturating_sub(cap), bar.h);
        let body = Rect::new(
            client.x + 8,
            bar.y + bar.h + 6,
            client.w.saturating_sub(16),
            client.h.saturating_sub(bar.h + lh + 28),
        );
        (well, body)
    }
}

impl DeskApp for Writer {
    /// Text reflows, so this is only a floor against losing the window.
    fn min_size(&self) -> (u32, u32) {
        (280, 160)
    }

    fn draw_in(&self, fb: &Framebuffer, client: Rect, focused: bool) {
        theme::panel(fb, client);
        let (well, body) = Self::metrics(client);
        let lh = theme::text_h();

        theme::text(fb, client.x + 8, well.y + 4, "path", theme::TEXT, theme::FACE);
        theme::well(fb, well, if self.editing_path { theme::HILIGHT } else { theme::FACE });
        let inner = well.shrink(3);
        let mut shown = self.path.clone();
        if self.editing_path && focused {
            shown.push('_');
        }
        let room = (inner.w / theme::text_w(1).max(1)) as usize;
        let tail = if shown.len() > room { &shown[shown.len() - room..] } else { &shown };
        theme::text(fb, inner.x, inner.y, tail, theme::TEXT, if self.editing_path { theme::HILIGHT } else { theme::FACE });

        theme::well(fb, body, theme::HILIGHT);
        let text_area = body.shrink(4);
        let cols = Self::client_cols(client);
        self.cols.set(cols);
        let view = Self::view_rows(client);
        let rows = self.rows(cols);
        let (crow, ccol) = self.cursor_pos(cols);

        // The draw pass owns the scroll: clamp it to the document, and when
        // an edit asked for it, bring the cursor's row into view.
        let mut scroll = self.scroll.get().min(rows.len().saturating_sub(1));
        if self.follow.replace(false) {
            let h = view.max(1);
            if crow < scroll {
                scroll = crow;
            } else if crow >= scroll + h {
                scroll = crow + 1 - h;
            }
        }
        self.scroll.set(scroll);

        for (i, (a, b)) in rows.iter().enumerate().skip(scroll).take(view) {
            let y = text_area.y + (i - scroll) as u32 * (lh + 2);
            let line = &self.text[*a..*b];
            theme::text(fb, text_area.x, y, line, theme::TEXT, theme::HILIGHT);
            // The caret: a block at the cursor cell, only where the document
            // has the keyboard.
            if focused && !self.editing_path && i == crow {
                let cx = text_area.x + ccol as u32 * theme::text_w(1);
                fb.rect(cx, y, 2, lh, theme::APERTURE_DEEP);
            }
        }

        let sy = body.y + body.h + 4;
        let flag = if self.dirty { "*" } else { "" };
        let line = format!(
            "{}{}  ^S save  ^O path  ^N new   {}",
            self.path, flag, self.status
        );
        let room = (client.w.saturating_sub(16) / theme::text_w(1).max(1)) as usize;
        let shown = if line.len() > room { &line[..room] } else { &line };
        theme::text(fb, client.x + 8, sy, shown, theme::TEXT, theme::FACE);
    }

    fn key(&mut self, k: u8) -> bool {
        use crate::dev::kbd;

        if self.editing_path {
            match k {
                b'\n' | b'\r' => {
                    self.editing_path = false;
                    self.load();
                }
                27 => self.editing_path = false,
                8 => {
                    self.path.pop();
                }
                c if (32..127).contains(&c) => self.path.push(c as char),
                _ => return false,
            }
            return true;
        }

        match k {
            CTRL_S => self.save(),
            CTRL_O => {
                self.editing_path = true;
                self.status = String::from("Enter loads the path, Esc cancels");
            }
            CTRL_N => {
                self.text = String::new();
                self.cursor = 0;
                self.scroll.set(0);
                self.dirty = true;
                self.status = String::from("empty document (unsaved)");
            }
            kbd::KEY_LEFT => self.cursor = self.cursor.saturating_sub(1),
            kbd::KEY_RIGHT => self.cursor = (self.cursor + 1).min(self.text.len()),
            kbd::KEY_UP | kbd::KEY_DOWN => {
                // Vertical movement in the columns of the last draw: same
                // column one wrapped row over, clamped to that row's length.
                let cols = self.cols.get();
                let rows = self.rows(cols);
                let (r, c) = self.cursor_pos(cols);
                let nr = if k == kbd::KEY_UP {
                    r.saturating_sub(1)
                } else {
                    (r + 1).min(rows.len() - 1)
                };
                let (a, b) = rows[nr];
                self.cursor = (a + c).min(b);
            }
            kbd::KEY_HOME => {
                let rows = self.rows(self.cols.get());
                let (r, _) = self.cursor_pos(self.cols.get());
                self.cursor = rows[r].0;
            }
            kbd::KEY_END => {
                let rows = self.rows(self.cols.get());
                let (r, _) = self.cursor_pos(self.cols.get());
                self.cursor = rows[r].1;
            }
            8 => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.text.remove(self.cursor);
                    self.dirty = true;
                }
            }
            kbd::KEY_DELETE => {
                if self.cursor < self.text.len() {
                    self.text.remove(self.cursor);
                    self.dirty = true;
                }
            }
            b'\n' | b'\r' => self.insert('\n'),
            b'\t' => {
                for _ in 0..4 {
                    self.insert(' ');
                }
            }
            c if (32..127).contains(&c) => self.insert(c as char),
            _ => return false,
        }
        // Any handled key that moves the caret wants it visible; a save does
        // not move it and must not yank the view.
        if k != CTRL_S {
            self.follow.set(true);
        }
        true
    }

    fn press(&mut self, client: Rect, x: i32, y: i32) -> bool {
        let (well, body) = Self::metrics(client);
        let inside = |r: Rect| {
            x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32
        };
        if inside(well) {
            self.editing_path = true;
            return true;
        }
        if inside(body) {
            self.editing_path = false;
            let text_area = body.shrink(4);
            let lh = theme::text_h() + 2;
            let cols = Self::client_cols(client);
            let rows = self.rows(cols);
            let row = self.scroll.get() + ((y - text_area.y as i32).max(0) as u32 / lh) as usize;
            let col = ((x - text_area.x as i32).max(0) as u32 / theme::text_w(1).max(1)) as usize;
            let (a, b) = *rows.get(row.min(rows.len() - 1)).unwrap_or(&(0, 0));
            self.cursor = (a + col).min(b);
            return true;
        }
        false
    }

    fn wheel(&mut self, notches: i32) -> bool {
        let before = self.scroll.get();
        let next = if notches > 0 {
            before.saturating_add(notches as usize * 3)
        } else {
            before.saturating_sub((-notches) as usize * 3)
        };
        self.scroll.set(next);
        // Reading, not editing: the next draw must not yank the view back to
        // the caret.
        self.follow.set(false);
        next != before
    }
}
