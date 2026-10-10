//! The desktop and its window manager.
//!
//! The terminal is a window on this, not the other way round. That distinction
//! is the whole design: a shell that draws a dialog over itself is a program
//! with a pop-up, while a desktop that hosts a terminal alongside other windows
//! is an environment. The console already knows how to live inside a rectangle
//! (`console::reflow`), so the terminal needs no special case -- it is a window
//! whose content happens to be the character grid.
//!
//! ### Repaint is always total; what reaches the screen is not
//!
//! Windows overlap, and the usual price for that is damage tracking: work out
//! which rectangles a change dirtied and repaint only those. Not here. Every
//! change repaints the wall and then every window back to front -- into the
//! compositor's back buffer, and `compose::present` copies to the screen only
//! the rows that differ from what is already there. Total repaint keeps the
//! window manager obviously correct (the entire class of "stale pixels from a
//! window that used to be there" cannot occur when there is no partial
//! repaint); the diffed present is what stopped every keystroke flashing the
//! wallpaper through the frame while it was rebuilt, and it is what makes
//! repainting on pointer *motion* affordable at all.
//!
//! Back-to-front is also what makes the terminal work with no special case.
//! `console::redraw` paints the whole grid; a window in front of it is simply
//! drawn afterwards.
//!
//! ### The pointer
//!
//! Everything the pointer does, a keystroke can also do -- serial cannot
//! inject PS/2 packets, so an operation that existed only as a gesture would
//! be an operation `drive.py` could never test. The pointer's vocabulary:
//! press to focus and raise, title bar to drag, the corner grip to resize, a
//! double press on a title to maximise, the bar and the Start menu and the
//! wall icons to launch, the wheel to scroll, hover to see where a press
//! would land. The second button opens the menu for whatever is under it.
//!
//! The ancestry is deliberate: 98's furniture (icons, Start, a bar of window
//! buttons, gradient titles), 3.1's construction (bevels, the grey face,
//! dialogs that hug their content), Aperture's colours over both.

use super::browse::Browser;
use super::theme::{self, Rect};
use super::ui::{self, Action, Panel};
use super::{Color, DeskApp, Framebuffer};
use crate::dev::kbd;
use crate::sync::Racy;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// What is inside a window.
pub enum Content {
    /// The character grid. Exactly one window holds it -- there is one console
    /// -- and that is true by construction rather than checked.
    /// A console grid. The payload names which one.
    Terminal(usize),
    Panel(Panel),
    /// Enternet. Its own variant rather than a Panel because a page is not a
    /// stack of widgets: it scrolls, it wraps to the window, and its links are
    /// a selection model the widget enum has no shape for.
    Browser(Browser),
    /// A program that owns its client area: Paintbrush, Write, Minesweeper.
    App(Box<dyn DeskApp>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WinState {
    Normal,
    Maximised,
    Minimised,
}

pub struct MenuItem {
    pub label: String,
    pub action: Action,
}

pub struct Menu {
    pub label: String,
    pub items: Vec<MenuItem>,
}

/// Indices into the pictogram set, named so a window can say what it is
/// without string-matching its own title.
pub const ICO_TERM: usize = 0;
pub const ICO_PROGRAMS: usize = 1;
pub const ICO_FILES: usize = 2;
pub const ICO_TODO: usize = 3;
pub const ICO_NET: usize = 4;
pub const ICO_PAINT: usize = 5;
pub const ICO_WRITE: usize = 6;
pub const ICO_MINES: usize = 7;
pub const ICO_ORACLE: usize = 8;
pub const ICO_SET: usize = 9;

/// The wall draws `ICONS[k]`'s label under `pictogram(k)`, so the two tables
/// are one table with the index as its only key. Asserted, because it was
/// wrong: `ICO_SET` was 8 against `ICONS[8] == "Oracle"`, so the sliders wore
/// the Oracle's name and the eye wore Settings' -- for as long as those two
/// icons have existed. The comment on the `ICO_PAINT` arm says the named
/// constants made that mistake unwriteable, and it was half true. It made it
/// unwriteable for the four icons somebody checked.
const _: () = assert!(ICONS.len() == 10);

/// The icon for a named panel -- the names `win open` and the Browse routes
/// use. Anything unrecognised gets the mark, because everything here is
/// Aperture something.
fn panel_icon(name: &str) -> usize {
    match name {
        "files" => ICO_FILES,
        "todo" => ICO_TODO,
        "set" | "settings" => ICO_SET,
        _ => ICO_PROGRAMS,
    }
}

pub struct Window {
    pub title: String,
    /// Which pictogram stands for this window on the taskbar.
    pub icon: usize,
    /// Geometry when not maximised. Kept across a maximise so restoring is
    /// exact rather than approximate.
    pub rect: Rect,
    pub state: WinState,
    /// Geometry from before the window was snapped to an edge, if it was.
    ///
    /// A snapped window has lost the size the operator chose for it, and the
    /// only record of that size is here. Dragging back off the edge restores
    /// it, which is the half of snapping that makes the other half safe to
    /// use: an edge that swallows a window's proportions permanently is a
    /// trap, not a shortcut.
    pub snap_back: Option<Rect>,
    /// The route this window's contents came from, when they came from one.
    ///
    /// A panel is a snapshot: it was built from something that has since
    /// changed. Nothing rebuilt it, so pressing a button in an application ran
    /// the command and left the window showing what was true before. Files has
    /// the same defect and always has -- `write x` never updated an open
    /// browser. Keeping the route is what makes rebuilding possible at all.
    pub route: Option<String>,
    /// Corner radius, when this window is not a plain rectangle.
    ///
    /// Stored as the radius rather than as the outline so that a resize does
    /// not have to remember how the outline was asked for. The outline itself
    /// is rebuilt from it during the paint, and cached against the size it was
    /// built for, because the arithmetic is a square root per row of the arc
    /// and doing it sixty times a second for a shape that has not changed is
    /// the sort of waste that turns into "the desktop feels slow".
    pub round: u32,
    pub content: Content,
    pub menus: Vec<Menu>,
    pub closable: bool,
    /// Who this window is, for as long as it is open. The scene pairs a
    /// window across frames by this rather than by its index, which shifts
    /// whenever a window below it closes or one is raised.
    pub serial: u64,
    /// An application's own pixels, kept between frames.
    ///
    /// An application used to redraw itself into the desktop on every frame
    /// that touched its window -- including frames where nothing about it had
    /// changed, because another window moved across it or the clock ticked.
    /// Kept, it is redrawn only when its key moves (`app_key`), and a frame
    /// that merely exposes or moves it is a row copy. Interior mutability
    /// because the painter holds the desktop shared.
    pub app_surface: core::cell::RefCell<Option<AppSurface>>,
    /// The whole window as it was last painted, chrome and client, so a frame
    /// that only exposes it copies rows. See `WinImage`.
    pub image: core::cell::RefCell<Option<WinImage>>,
    /// A window on its way in: the pop-in's scale and the fade, each on its
    /// own curve and duration, as Hyprland's `windowsIn` and `fadeIn` are.
    pub arrive: Option<(super::anim::Tween, super::anim::Tween)>,
    /// Where this window is drawn this frame and at what opacity, while it
    /// is arriving or sliding -- fixed once per frame by `layers`, so the
    /// frame and `render verify`'s reference of it draw the same instant
    /// rather than two readings of the clock.
    pub shown: core::cell::Cell<Option<(Rect, f32)>>,
    /// In the tiling layout (`gfx::tile`), or floating over it. Every window
    /// tiles unless it is put afloat -- Super-V, or by taking hold of it.
    pub tiled: bool,
    /// On its way from one place to another: where it set out from and the
    /// tween (Hyprland's `windows`). The window is already at its new place
    /// as far as input and its image go; it is drawn on the way there.
    pub slide: Option<(Rect, super::anim::Tween)>,
}

/// A window, painted off screen and kept (1.4.1, phase 11).
///
/// Keyed by exactly what the scene stamps the window's layer with
/// (`window_key`): the layer's stamp moving re-draws all of it, a part's
/// stamp moving re-draws that part, and anything else -- a window dragged
/// across it, a menu closing over it, the pointer's plane -- is a copy out of
/// here. The terminal's grid is not in it: the console paints its own
/// surface and goes over the image, since text arrives at a rate chrome
/// never does.
///
/// Pixels outside the window's outline are kept and never shown: the copy
/// goes through the same shape the direct paint is clipped by, so the two
/// agree pixel for pixel, which is what `render verify` holds them to -- its
/// reference frame paints straight onto the frame (`REFERENCE`), so a stale
/// image is a mismatch rather than two frames agreeing about the same stale
/// copy.
pub struct WinImage {
    px: Vec<u32>,
    frame: Rect,
    key: u64,
    parts: Vec<(Rect, u64)>,
}

/// What `Window::app_surface` holds: the pixels, their size, and the key they
/// were drawn under.
pub struct AppSurface {
    px: Vec<u32>,
    w: u32,
    h: u32,
    key: u64,
}

/// How many events each application has been handed -- a key, a press, a
/// drag, a release, a wheel notch -- keyed by the address of the program
/// (its `Box` does not move while the window lives). Part of that program's
/// key, so input always redraws the program that received it, without it
/// having to say what the input changed, and redraws nothing else.
///
/// By address rather than by window because every caller holds the program
/// borrowed out of `d.windows[i].content`, and the window beside it cannot be
/// reached through the same index while that borrow lives. A closed window's
/// slot is taken by whichever program lands at its address next, which costs
/// that program nothing: its surface is new either way.
static APP_INPUT: crate::sync::Spin<[(usize, u64); 16]> = crate::sync::Spin::new([(0, 0); 16]);

fn app_addr(a: &dyn DeskApp) -> usize {
    a as *const dyn DeskApp as *const u8 as usize
}

fn app_input(a: &dyn DeskApp) {
    let at = app_addr(a);
    let mut t = APP_INPUT.lock_irq();
    match t.iter().position(|e| e.0 == at) {
        Some(k) => t[k].1 = t[k].1.wrapping_add(1),
        None => {
            // Evict the least-touched row; a program that loses its row gets
            // a count of zero, which differs from what it had, so it redraws.
            let k = (0..t.len()).min_by_key(|&k| t[k].1).unwrap_or(0);
            t[k] = (at, 1);
        }
    }
}

fn inputs_to(a: &dyn DeskApp) -> u64 {
    let at = app_addr(a);
    APP_INPUT.lock_irq().iter().find(|e| e.0 == at).map_or(0, |e| e.1)
}

/// The key an application's surface is drawn under: the program's own stamp
/// if it offers one, plus focus, size and input. A program with no stamp is
/// redrawn once per composed frame, which is what every program got before
/// there were stamps.
fn app_key(a: &dyn DeskApp, active: bool, client: Rect, seq: u64) -> u64 {
    match a.stamp() {
        Some(st) => super::scene::Stamp::of((
            st,
            active,
            client.w,
            client.h,
            inputs_to(a),
        )),
        None => super::scene::Stamp::of((seq, 0xA99u32)),
    }
}

/// Bring a window's application surface up to date and lay it into `fb`.
fn paint_app(win: &Window, a: &dyn DeskApp, fb: &Framebuffer, client: Rect, active: bool) {
    let seq = FRAME_SEQ.load(core::sync::atomic::Ordering::Relaxed);
    let key = app_key(a, active, client, seq);
    let mut slot = win.app_surface.borrow_mut();
    let stale = match slot.as_ref() {
        Some(sf) => sf.w != client.w || sf.h != client.h || sf.key != key,
        None => true,
    };
    if stale {
        let n = client.w as usize * client.h as usize;
        let mut px = match slot.take() {
            Some(sf) if sf.px.len() == n => sf.px,
            _ => {
                let mut v = Vec::new();
                if v.try_reserve_exact(n).is_err() {
                    // No room for a surface: draw straight into the frame, as
                    // every application did before there were surfaces.
                    drop(slot);
                    if let Some(v) = fb.view(client.x, client.y, client.w, client.h) {
                        a.draw_in(&v, Rect::new(0, 0, client.w, client.h), active);
                    }
                    return;
                }
                v.resize(n, 0);
                v
            }
        };
        let view = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, client.w, client.h, client.w, fb.format()) };
        // The window's face first: programs paint over the ground
        // `theme::window` laid, and do not all fill their whole client area.
        view.rect(0, 0, client.w, client.h, theme::FACE);
        let t0 = crate::time::rdtsc();
        a.draw_in(&view, Rect::new(0, 0, client.w, client.h), active);
        super::render::cost(super::render::Cost::AppRender, crate::time::rdtsc().saturating_sub(t0));
        *slot = Some(AppSurface { px, w: client.w, h: client.h, key });
    }
    let Some(sf) = slot.as_ref() else { return };
    let (_, y0, _, y1) = fb.scissor();
    for y in y0.max(client.y)..y1.min(client.y + client.h) {
        let r = (y - client.y) as usize * sf.w as usize;
        fb.blit_span(client.x, y, &sf.px[r..r + sf.w as usize]);
    }
}

static NEXT_SERIAL: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

fn next_serial() -> u64 {
    NEXT_SERIAL.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
}

impl Window {
    /// Where the window actually is right now.
    fn frame(&self, screen: Rect) -> Rect {
        match self.state {
            WinState::Maximised => screen,
            _ => self.rect,
        }
    }
}

/// What the keyboard is doing.
///
/// A mode rather than a pile of booleans: the states are mutually exclusive by
/// nature -- a window cannot be both being moved and being resized -- and an
/// enum makes that unrepresentable instead of merely unlikely.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    Normal,
    /// A menu-bar menu is open.
    Menu { menu: usize, item: usize },
    /// The system menu (Alt-Space) is open.
    Sys { item: usize },
    /// Arrows move the focused window; Enter commits, Esc restores.
    Move { from: Rect },
    /// Arrows resize it.
    Size { from: Rect },
    /// The keyboard is on the taskbar. `item` indexes the app buttons first,
    /// then the task buttons -- one flat list, because Left and Right should
    /// walk the bar the way it looks rather than the way it is built.
    Taskbar { item: usize },
    /// The pointer is moving the focused window by its title bar. `dx`, `dy`
    /// are where inside the bar it was grabbed, so the window does not snap
    /// its corner to the pointer on the first packet.
    Drag { dx: u32, dy: u32, from: Rect },
    /// The pointer is resizing it by the bottom-right corner.
    DragSize { from: Rect, edges: u8 },
    /// The Start menu is open above the taskbar.
    Start { item: usize },
}

/// What the pointer is over, for the paint pass.
///
/// Only things that draw differently when pointed at. Kept on the desktop and
/// compared per packet: a repaint happens when the *target* changes, not when
/// the pointer moves, so sliding across one button costs one redraw.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hover {
    None,
    /// A taskbar slot, apps and windows in one flat index.
    Task(usize),
    /// A menu-bar label: window `win`, label `menu`.
    MenuLabel { win: usize, menu: usize },
    /// A focusable widget in window `win`'s panel.
    Widget { win: usize, idx: usize },
    /// A desktop icon.
    Icon(usize),
    /// The Start button.
    Start,
    /// A caption button on window `win`: 0 minimise, 1 maximise, 2 close.
    Caption { win: usize, which: usize },
}

pub struct Desktop {
    /// Back to front. The last visible window is the focused one, so z-order
    /// and focus are the same fact stored once -- two fields would be two
    /// things to keep in agreement, and they would drift.
    pub windows: Vec<Window>,
    pub mode: Mode,
    pub hover: Hover,
    /// What has been typed into the Start menu's query row.
    ///
    /// Here rather than inside `Mode::Start`, because `Mode` is `Copy` and is
    /// assigned by value all over this file -- a `String` in it would turn
    /// every one of those into a move and the enum into something that has to
    /// be cloned to be read. It lives as long as the menu is open and is
    /// cleared when it closes, which is the same lifetime either way.
    pub query: String,
    /// The tiling layout: which windows tile, and how their slots split.
    pub tiles: super::tile::Tree,
}

static DESK: Racy<Option<Desktop>> = Racy::new(None);
/// Locked: the shell drains it while the compositor (a key, a click) and
/// other tasks push. It was a `Racy`, which was a promise that only one task
/// ever touched it, and the compositor routing keys made that untrue.
static PENDING: crate::sync::Spin<Vec<String>> = crate::sync::Spin::new(Vec::new());

/// How many queued commands are kept.
///
/// The shell drains one per pass of its idle loop, and a pass is fast, so this
/// only has to cover a burst -- somebody clicking three buttons before the
/// loop gets a turn. Eight is well past that and small enough that a runaway
/// press handler cannot grow the heap.
const PENDING_CAP: usize = 8;

/// Ask for a command to be run as though it had been typed.
///
/// The one way in from outside this module, so there stays exactly one path
/// from "a command should run" to "a command ran" -- the property the panel
/// dispatch was built around, and one a second setter would quietly end.
///
/// **A queue and not a slot.** It was `Option<String>`, so a second press
/// before the shell's idle loop took a turn overwrote the first and the first
/// was gone with nothing said. That was survivable while the AI windows had
/// six buttons between them and is not now.
///
/// Full drops the *newest* and says so. The alternative -- pushing the oldest
/// out -- is the bug this replaces, arriving one command later.
pub fn queue_command(cmd: &str) {
    let full = {
        let mut q = PENDING.lock_irq();
        let full = q.len() >= PENDING_CAP;
        if !full {
            q.push(String::from(cmd));
        }
        full
    };
    // Said outside the lock: printing takes the console's.
    if full {
        crate::kprintln!("  (queue full -- '{}' dropped)", cmd);
    }
}

/// Desktop changes asked for by a task that does not own the desktop.
///
/// **One writer, and it is the compositor.** `Desktop` lives behind a `Racy`,
/// which is single-core interior mutability and not a lock, so two tasks
/// holding `&mut Desktop` is undefined behaviour rather than a race. That was
/// safe by accident while the shell both changed the windows and drew them. It
/// stopped being safe when the compositor task started reading the window list
/// for the two milliseconds a frame takes: `open_authoring` runs on the *agent*
/// task and calls `open_app`, which is a `Vec::push` that can reallocate
/// `windows` out from under a walk.
///
/// So a task that is not the compositor posts an operation and the compositor
/// applies it between frames, where nothing else is looking at the desktop.
/// Same discipline as `autosnap_tick`/`autosnap_poll`, and the same reason:
/// the timer flags and the owner acts.
///
/// **A `Vec` and not the ring in `linux/input.rs`.** That ring is lock-free and
/// better in every way except the one that matters here -- its `const fn new`
/// needs `T: Copy`, and `Op::OpenAuthoring` carries a `Box<dyn DeskApp>`.
/// Forcing one mechanism would mean boxing every keystroke to serve the rare
/// case. Input gets the ring; owned payloads get this.
pub enum Op {
    /// Open the authoring window, if it is not already open.
    ///
    /// The `Box` is built by the poster rather than in here, because
    /// `AuthorWin::new` is the agent's business and this queue's job is to
    /// carry it, not to know about it.
    OpenApp { title: &'static str, icon: usize, app: alloc::boxed::Box<dyn super::DeskApp>, w: u32, h: u32 },
    /// A program's window by tag, the shape Skywalker's clients need: a
    /// title the client chose, and no "already open" check, since two
    /// windows of one client are two windows.
    OpenTagged { title: String, icon: usize, app: alloc::boxed::Box<dyn super::DeskApp>, w: u32, h: u32 },
    Retitle { tag: u64, title: String },
    Close { tag: u64 },
}

/// Locked, because posting from any task is the point of it: a `Racy` here
/// made "safe from any task" true only of the task that drains it.
static OPS: crate::sync::Spin<Vec<Posted>> = crate::sync::Spin::new(Vec::new());

/// An op in transit. `Send` by assertion: a `Box<dyn DeskApp>` is not, because
/// programs keep `Cell`s, but posting *moves* the box -- the poster keeps no
/// reference to it, so it is only ever touched by one task at a time, which
/// is all `Send` asks. A program that kept an `Rc` shared with its poster
/// would break this, and none does; Skywalker's window shares an `Arc<Spin>`.
struct Posted(Op);
unsafe impl Send for Posted {}

/// Small for the same reason `PENDING_CAP` is: this covers a burst, not a
/// backlog. Nothing posts more than one of these per episode.
const OPS_CAP: usize = 8;

/// Ask the compositor to change the desktop.
///
/// Safe from any task. Full drops the newest and says so, exactly as
/// `queue_command` does -- pushing the oldest out is the bug that replaces.
pub fn post(op: Op) {
    let full = {
        let mut q = OPS.lock_irq();
        let full = q.len() >= OPS_CAP;
        if !full {
            q.push(Posted(op));
        }
        full
    };
    if full {
        crate::kprintln!("  (desktop queue full -- an op was dropped)");
        return;
    }
    super::render::invalidate();
}

/// Apply everything posted. The compositor calls this, and nothing else may.
///
/// Drains to empty rather than one per frame: an op is cheap, and a window
/// that opened two frames after it was asked for would read as the lag this
/// whole change exists to remove.
pub fn drain_ops() {
    loop {
        let op = {
            let mut q = OPS.lock_irq();
            if q.is_empty() {
                return;
            }
            q.remove(0).0
        };
        match op {
            Op::OpenApp { title, icon, app, w, h } => {
                if has_window(title) {
                    continue;
                }
                open_app(title, icon, app, w, h);
                clear_of_terminal();
                focus_terminal();
            }
            Op::OpenTagged { title, icon, app, w, h } => open_app(&title, icon, app, w, h),
            Op::Retitle { tag, title } => retitle_tagged(tag, &title),
            Op::Close { tag } => {
                close_tagged(tag);
            }
        }
    }
}

/// How many ops are waiting. For the selftest.
pub fn ops_len() -> usize {
    OPS.lock_irq().len()
}

const MARGIN: u32 = 8;
const NUDGE: u32 = 16;
/// A menu bar's height, which is the theme's and no longer the caption's.
///
/// It was `= theme::TITLE_H`, which was a coincidence rather than a fact and
/// became a trap the moment the caption grew: every menu row and every
/// dropdown row would have gone to thirty pixels around sixteen of text,
/// silently, because nothing here says a menu row is as tall as a title.
const MENU_H: u32 = theme::MENU_H;
/// The bar along the bottom.
///
/// A number rather than `TITLE_H + 10` for the same reason. A taskbar is its
/// own surface and XP's is not a caption plus a constant.
const TASK_H: u32 = 38;
/// Gap between task buttons.
const TASK_GAP: u32 = 4;

/// The icons on the wall, top to bottom, and what opening one runs.
///
/// `term` is not a shell command: the terminal is not a panel to open but a
/// window to bring back, and the special case lives in `launch` rather than in
/// the shell so the icon works even while the shell is busy printing.
const ICONS: [(&str, &str); 10] = [
    ("Terminal", "term"),
    ("Programs", "win open programs"),
    ("Files", "win open files"),
    ("ToDo", "todo"),
    ("Enternet", "enternet"),
    ("Paint", "paint"),
    ("Write", "write"),
    ("Mines", "mines"),
    ("Oracle", "oracle"),
    ("Settings", "win open settings"),
];

/// The Start menu, bottom of the bar upward -- the 98 half of the ancestry.
/// Same entries as the icons plus the one thing that belongs behind a second
/// look, exactly where 98 kept it.
const START_ITEMS: [(&str, &str); 13] = [
    // "Search..." used to lead this list, opening a panel with one text field
    // in it. The query row at the foot of this menu does the same job in the
    // place a person already is, and dispatches through the same `open`, so
    // the item became a second door to one room. The panel itself is still
    // there -- `win open search`, and `open` still raises it to offer to write
    // something that does not exist.
    ("Terminal", "term"),
    ("Programs", "win open programs"),
    // The AI workspace. It was reachable only by typing `mind open`, which
    // appeared in no menu, no `help` and no Program Manager row -- so the four
    // windows this machine's whole argument rests on were undiscoverable from
    // inside the machine. Not on the wall: `ICONS` is index-locked to the
    // `ICO_*` constants and adding a row there renumbers every pictogram.
    ("Mind", "mind open"),
    ("Files", "win open files"),
    ("ToDo", "todo"),
    ("Enternet", "enternet"),
    ("Paint", "paint"),
    ("Write", "write"),
    ("Mines", "mines"),
    ("Oracle", "oracle"),
    ("Settings", "win open settings"),
    ("Reboot", "reboot"),
    ("Shut down", "shutdown"),
];

/// Run what an icon or Start entry names.
fn launch(cmd: &str) {
    if cmd == "term" {
        // Open the terminal, creating it if the desktop has none -- which a
        // clean startup does. The icon is also the way back for a minimised
        // one, which `open_terminal` restores and raises.
        open_terminal();
        return;
    }
    // Queue the command and let it run. A command that opens a window leaves
    // that window in front (see `open`/`open_app`); one that only prints does
    // so in the terminal wherever it sits. The shell processes PENDING every
    // loop regardless of focus, so nothing here needs to touch focus at all.
    queue_command(cmd);
}

pub fn ready() -> bool {
    unsafe { (*DESK.get()).is_some() }
}

/// Reach the desktop, under the painter's claim.
///
/// **This hands out `&mut Desktop` and used to do it with no exclusion at
/// all.** That was safe by accident for as long as one task both changed the
/// windows and drew them: the shell opened a window and the shell painted it,
/// and a preemption between the two could not produce a reader. A compositor
/// task ends that -- it reads this state for the two milliseconds a frame
/// takes, and a shell preempted halfway through moving a window leaves exactly
/// the half-written `Vec` the reader would walk.
///
/// So the claim covers the window list as well as the screen, and `with`
/// waits for it rather than refusing: a caller here is usually *changing*
/// something, and a window that declines to open because a frame was in
/// flight is a worse failure than the one being prevented.
///
/// `draw` holds the claim across its whole frame and calls this ninety-one
/// times underneath, which is why the claim is reentrant per task.
/// Read the desktop without being able to change it.
///
/// **`draw` is a reader and this is what makes that a fact rather than a
/// comment.** It composes a whole frame -- wallpaper, taskbar, every window,
/// popups -- and writes nothing to the model; `taskbar`, `draw_icons` and the
/// rest already take `&Desktop`. Until now it said so in prose while holding a
/// `&mut`, so nothing stopped a later edit from quietly mutating inside the
/// frame, which is the one place a mutation is hardest to reason about.
///
/// It still takes the claim, because a reader and a writer must not interleave
/// -- a frame walking `windows` while somebody pushes to it is the hazard, and
/// which side holds the `&mut` does not change that. What it does not do is
/// mark the screen dirty: reading changed nothing, and a reader that invited a
/// repaint would have the compositor composing forever off its own frames.
pub fn with_ref<R>(f: impl FnOnce(&Desktop) -> R) -> Option<R> {
    let _claim = Claim::wait();
    unsafe { (*DESK.get()).as_ref().map(f) }
}

#[track_caller]
pub fn with<R>(f: impl FnOnce(&mut Desktop) -> R) -> Option<R> {
    let claim = Claim::wait();
    // **Reaching the desktop from outside a frame marks the screen dirty**,
    // and that is what stops this becoming the treadmill it replaced. The old
    // arrangement asked every caller to remember a repaint, sixteen of them
    // did, and a task inside a long command remembered nothing -- so the rule
    // is now structural: if somebody took `&mut Desktop`, assume they changed
    // it, because the cost of being wrong is one frame and the cost of
    // forgetting is a frozen screen.
    //
    // Only a *top-level* claim, or `draw`'s own ninety-one nested calls would
    // re-dirty the flag from inside the frame that is servicing it and the
    // compositor would repaint forever at full tilt. `took` is exactly that
    // distinction: true for a fresh claim, false for a re-entry.
    if claim.took {
        super::render::invalidate();
    }
    unsafe { (*DESK.get()).as_mut().map(f) }
}

/// The system menu, identical for every window. Items a particular window
/// cannot do are still listed and simply refuse, which is what 3.1 did -- a
/// menu whose contents move around is a menu you cannot learn.
const SYS_ITEMS: [&str; 4] = ["Move", "Size", "Maximise / Restore", "Close"];

/// The area windows may occupy: everything above the taskbar.
///
/// Windows are laid out and maximised against this rather than the panel, so
/// the taskbar is never covered. A bar that a maximised window hides is a bar
/// that is not there when it is most wanted.
/// The shadow a window casts: a blurred rounded box under it (`gfx::shadow`),
/// deeper for the focused window, which reads as the one nearest the eye --
/// the cue that took over from a title bar's colour in every desktop since
/// Vista. None when it fills the work area, where there is nothing to fall on.
fn win_shadow(win: &Window, active: bool) -> Option<(super::shadow::Spec, i32, i32, u8)> {
    if win.state == WinState::Maximised {
        return None;
    }
    let radius = win.round;
    Some(if active {
        (super::shadow::Spec { radius, sigma: 12.0 }, 0, 5, 102)
    } else {
        (super::shadow::Spec { radius, sigma: 7.0 }, 0, 3, 61)
    })
}

/// A window's frame grown by everything its shadow reaches, clipped to the
/// screen: the layer's bounds, and so its damage when it moves.
fn shadowed(win: &Window, frame: Rect, active: bool, fb: &Framebuffer) -> Rect {
    let Some((spec, dx, dy, _)) = win_shadow(win, active) else { return frame };
    let (l, t, r, b) = super::shadow::reach(spec, dx, dy);
    let x0 = frame.x.saturating_sub(l);
    let y0 = frame.y.saturating_sub(t);
    let x1 = (frame.x + frame.w + r).min(fb.width());
    let y1 = (frame.y + frame.h + b).min(fb.height());
    Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
}

fn screen_rect(fb: &Framebuffer) -> Rect {
    Rect::new(
        MARGIN,
        MARGIN,
        fb.width().saturating_sub(MARGIN * 2),
        fb.height().saturating_sub(MARGIN * 2 + TASK_H),
    )
}

fn taskbar_rect(fb: &Framebuffer) -> Rect {
    Rect::new(0, fb.height().saturating_sub(TASK_H), fb.width(), TASK_H)
}

/// The Start button, left end of the bar.
fn start_rect(fb: &Framebuffer) -> Rect {
    let bar = taskbar_rect(fb);
    // Bleeding to the left edge and nearly the bar's full height, as XP's
    // does. A button inset from the corner gives away the pixels Fitts' law
    // makes the most valuable on the screen: the corner is infinitely large to
    // a pointer and a three-pixel margin throws that away.
    // The orb's own diameter plus a margin, still from the very corner.
    let n = bar.h.saturating_sub(2);
    Rect::new(bar.x, bar.y + 1, n + 12, n)
}

/// Where the Start menu pops, directly above its button. The width formula is
/// `dropdown`'s own, so the paint and the hit-test cannot disagree.
fn start_menu_rect(fb: &Framebuffer) -> theme::Popup {
    let n = start_rows();
    // Wide enough for the longest label, and for a query worth typing. A menu
    // sized only to its labels gives the search row about eleven characters,
    // which is narrower than the thing being searched for.
    let cols = START_ITEMS
        .iter()
        .map(|(l, _)| l.chars().count())
        .max()
        .unwrap_or(4)
        .max(QUERY_COLS);
    let bar = taskbar_rect(fb);
    let probe = theme::Popup::sized(bar.x + 2, 0, cols, n, fb.width());
    theme::Popup::sized(
        bar.x + 2,
        bar.y.saturating_sub(probe.panel.h),
        cols,
        n,
        fb.width(),
    )
}

/// How many characters wide the query row is sized for.
const QUERY_COLS: usize = 22;

/// Rows in the Start menu: every item, then the query.
///
/// The query is **last**, which is to say nearest the Start button, because
/// this menu opens upwards out of the taskbar. Windows 7 put its search box in
/// the same place for the same reason -- it is where the pointer already is
/// after clicking Start, and where the eye goes. It also leaves every item
/// index exactly as it was, so nothing that indexes `START_ITEMS` had to learn
/// about an offset.
fn start_rows() -> usize {
    START_ITEMS.len() + 1
}

/// True when this row is the query rather than an item.
fn is_query_row(item: usize) -> bool {
    item >= START_ITEMS.len()
}

// --- desktop icons ---------------------------------------------------------

const ICON_W: u32 = 132;
/// The wall's icons, in pixels: 38 logical at the 1.25 scale.
const ICON_PX: u32 = 48;
const ICON_H: u32 = 84;

/// The icon column, top-left of the wall, downwards -- the 98 half of the
/// ancestry. Stops above the taskbar rather than flowing into it.
fn icon_rects(fb: &Framebuffer) -> Vec<(Rect, usize)> {
    let mut out = Vec::new();
    let mut x = 10u32;
    let mut y = 14u32;
    for (k, _) in ICONS.iter().enumerate() {
        if y + ICON_H > fb.height().saturating_sub(TASK_H) {
            // Wrap to a second column, as the 98 desktop did. The terminal
            // reserves only the first column, so overflow icons sit behind
            // windows -- reachable, and always also in the Start menu.
            x += ICON_W + 8;
            y = 14;
        }
        out.push((Rect::new(x, y, ICON_W, ICON_H), k));
        y += ICON_H + 10;
    }
    out
}

fn icon_at(fb: &Framebuffer, x: i32, y: i32) -> Option<usize> {
    icon_rects(fb)
        .into_iter()
        .find(|(r, _)| contains(*r, x, y))
        .map(|(_, k)| k)
}

/// Paint the icon column. Hovering highlights the label, 98-style; the
/// pictograms are rect art in the theme's own palette, because a 40-pixel
/// bitmap format is more machinery than six drawings justify.
fn draw_icons(fb: &Framebuffer, hover: Hover) {
    for (r, k) in icon_rects(fb) {
        let px = r.x + (r.w - ICON_PX) / 2;
        pictogram(fb, k, px, r.y, ICON_PX);
        let (label, _) = ICONS[k];
        let tw = theme::text_w_of(label);
        let tx = r.x + (r.w.saturating_sub(tw)) / 2;
        let ty = r.y + ICON_PX + 6;
        if hover == Hover::Icon(k) {
            fb.rect(
                tx.saturating_sub(3),
                ty - 2,
                tw + 6,
                theme::text_h() + 4,
                theme::SELECT,
            );
            theme::text_over(fb, tx, ty, label, theme::SELECT_TEXT);
        } else {
            // A shadow under white text is what keeps a label readable over
            // any wallpaper without boxing it.
            theme::text_over(fb, tx + 1, ty + 1, label, theme::DARKEDGE);
            theme::text_over(fb, tx, ty, label, theme::TITLE_TEXT);
        }
    }
}

/// One pictogram at `(x, y)`, `s` pixels square, by icon index: `gfx::icons`,
/// the lit objects the operator chose for 1.4.1, anti-aliased at any size and
/// transparent around the object -- which is why there is no longer a `cut`
/// to say what shows through the mark: whatever is behind it simply does.
///
/// They replaced drawings built from rectangles, frames and one-pixel
/// glosses in a 40-unit space, with a `Detail` level to drop the flourishes
/// that turned to noise below 32 pixels; a signed-distance drawing has no such
/// floor, and the same one serves the wall, the taskbar and a title bar.
fn pictogram(fb: &Framebuffer, k: usize, x: u32, y: u32, s: u32) {
    super::icons::lay(fb, k, x, y, s);
}

impl Desktop {
    /// Index of the focused window: the topmost that is not minimised.
    pub fn focus(&self) -> Option<usize> {
        self.windows
            .iter()
            .rposition(|w| w.state != WinState::Minimised)
    }

    /// Bring a window to the front. Focus follows, because they are the same.
    fn raise(&mut self, i: usize) {
        if i >= self.windows.len() {
            return;
        }
        let w = self.windows.remove(i);
        self.windows.push(w);
    }
}

pub fn init() {
    if super::primary().is_none() {
        return;
    }

    // Paint the wallpaper and blur its glass copy now, during boot, rather
    // than in the first frame the compositor composes: together they are about
    // half a second, which as a frame is the desktop appearing and then
    // standing still. Through a surface clipped to nothing, so this builds the
    // caches and writes no pixel.
    //
    // **Before the desktop exists, and that order is the fix rather than
    // tidiness.** The compositor task is already running and composes the
    // moment `DESK` is set; warmed any later, its first frame arrived
    // mid-build, found the cache still empty and built a second copy at the
    // same time -- two `&mut` into one `Racy` from two tasks, and half a
    // second of frame anyway. Measured: two rebuilds, a 771 ms first frame.
    if let Some(fb) = super::primary() {
        wallpaper_cached(&fb.clipped(0, 0, 0, 0));
    }

    // A clean desktop: nothing open. The wallpaper, the icon column and the
    // taskbar are the whole of it, and the operator opens what they want from
    // the sidebar -- the machine covers its own wall with nothing. This is the
    // "manufacturer hands you a bare desktop" startup, and it is what lets the
    // boot splash settle onto the wallpaper's mark with nothing in front of it.
    //
    // The consoles are not given windows here, but they still exist: the shell
    // runs on the serial line regardless, `USER` is shown by a Terminal opened
    // from the sidebar, and `EXEC` carries the boot log for when its window is
    // opened. Both are retargeted into the back buffer below so their output is
    // ready the instant a window asks for them.
    unsafe {
        *DESK.get() = Some(Desktop {
            windows: Vec::new(),
            mode: Mode::Normal,
            hover: Hover::None,
            query: String::new(),
            tiles: super::tile::Tree::new(),
        })
    };
    // The compositor needs the heap, which exists by now; each console then
    // paints into a surface of its own, reports what changed as damage, and
    // is laid into the frame under whatever covers it. Typing stays immediate
    // because the compositor composes the first frame after a quiet spell at
    // once (`render::pace`).
    super::compose::init();
    if super::compose::active() {
        for ch in [super::console::USER, super::console::EXEC] {
            super::console::with_ch(ch, |c| c.compose_mode());
        }
    }
    super::render::invalidate();
}

fn item(label: &str, cmd: &str) -> MenuItem {
    MenuItem {
        label: String::from(label),
        action: Action::Run(String::from(cmd)),
    }
}

/// The user terminal window, with its menus. Built on demand now rather than at
/// `init`, because a clean desktop opens with nothing -- so this is the one
/// place the terminal's shape is described, and `open_terminal` is the one
/// caller. `closable` stays false: closing the terminal minimises it, the way
/// it always has, because the shell behind it keeps running.
fn make_terminal(screen: Rect) -> Window {
    let icons_w = 10 + ICON_W + 10;
    let term_x = screen.x.max(icons_w);
    let term_w = (screen.x + screen.w).saturating_sub(term_x + MARGIN);
    Window {
        serial: next_serial(),
        app_surface: Default::default(),
        image: Default::default(),
        arrive: None,
        shown: Default::default(),
        tiled: true,
        slide: None,
        title: String::from("GLaDOS Terminal"),
        icon: ICO_TERM,
        rect: Rect::new(term_x, screen.y, term_w, screen.h),
        state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
        content: Content::Terminal(super::console::USER),
        menus: alloc::vec![
            Menu {
                label: String::from("File"),
                items: alloc::vec![
                    item("New window", "win open status"),
                    item("Snapshot now", "snap"),
                    item("Reboot", "reboot"),
                ],
            },
            Menu {
                label: String::from("View"),
                items: alloc::vec![
                    item("Clear", "clear"),
                    item("Redraw", "refresh"),
                    item("Help", "help"),
                ],
            },
            Menu {
                label: String::from("System"),
                items: alloc::vec![
                    item("Status", "status"),
                    item("Memory", "mem"),
                    item("Windows", "win"),
                ],
            },
        ],
        closable: false,
    }
}

/// Open the user terminal, creating it if the desktop has none. The sidebar
/// icon and the `term` action land here, so a bare clean desktop is one click
/// from a shell. Distinct from `focus_terminal`, which the shell calls after
/// every command and must *not* create a window -- a shell that conjured its
/// own window on every line could never be closed.
pub fn open_terminal() {
    let Some(fb) = super::primary() else {
        return;
    };
    let screen = screen_rect(&fb);
    let existing = with(|d| {
        d.windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
    })
    .flatten();
    with(|d| match existing {
        Some(i) => {
            d.windows[i].state = WinState::Normal;
            d.raise(i);
        }
        None => {
            let term = make_terminal(screen);
            d.windows.push(term);
            let n = d.windows.len() - 1;
            d.raise(n);
        }
    });
    super::render::invalidate();
}

/// Open a new window holding a panel.
/// Open a panel, remembering the route it came from so it can be rebuilt.
pub fn open_routed(title: &str, panel: Panel, route: &str) {
    open(title, panel);
    let route = String::from(route);
    with(|d| {
        if let Some(w) = d.windows.last_mut() {
            w.route = Some(route);
        }
    });
}

/// Rebuild every window whose contents came from a route.
///
/// Called after a command runs, because a command is the only thing that
/// changes what a route would produce. Cheap when nothing is open: a window
/// without a route is skipped, and there is no route to resolve.
pub fn refresh_routed() {
    let routes: Vec<(usize, String)> = with(|d| {
        d.windows
            .iter()
            .enumerate()
            .filter_map(|(i, w)| w.route.clone().map(|r| (i, r)))
            .collect()
    })
    .unwrap_or_default();
    if routes.is_empty() {
        return;
    }
    let mut any = false;
    for (i, route) in routes {
        // Built outside `with`, because building an application's panel runs
        // its program, which reads the namespace -- and the desktop borrow is
        // held for the whole closure.
        let Some((_, panel)) = ui::panel_for_route(&route) else {
            continue;
        };
        with(|d| {
            if let Some(w) = d.windows.get_mut(i) {
                w.content = Content::Panel(panel);
                any = true;
            }
        });
    }
    if any {
        super::render::invalidate();
    }
}

/// Show a route in the window already showing that route's family, if there
/// is one.
///
/// A search box is one surface whose contents change, not a pile of windows.
/// Without this, asking twice left two windows with the same title and the
/// operator no way to tell which one was live -- and the third, the offer, made
/// three. Matching is on the route's verb, so `search` and `search:calc` are
/// the same surface while `app:todo` is not.
///
/// Returns whether a window was reused, because the caller's focus decision
/// differs: a reused window is already where the eye is.
pub fn show_routed(title: &str, panel: Panel, route: &str) -> bool {
    let verb = route.split(':').next().unwrap_or(route);
    let found = with(|d| {
        d.windows.iter().position(|w| {
            w.route.as_deref().map(|r| r.split(':').next().unwrap_or(r)) == Some(verb)
        })
    })
    .flatten();
    let Some(i) = found else {
        open_routed(title, panel, route);
        return false;
    };
    // The two panels are different heights, so keeping the old rectangle would
    // clip whichever is taller. Position stays: the window has not moved, only
    // what is in it.
    let Some(fb) = super::primary() else {
        return false;
    };
    let screen = screen_rect(&fb);
    let (pw, ph) = panel.preferred();
    // Clamped and pulled back on screen. `open` does this for a new window and
    // growing one in place has to do it too: the offer is wider than the query
    // box, and at the cascade position the extra width went off the right edge
    // -- taking the caption buttons and the right half of every line with it.
    let (pw, ph) = (pw.min(screen.w), ph.min(screen.h));
    let title = String::from(title);
    let route = String::from(route);
    with(|d| {
        let Some(mut w) = (i < d.windows.len()).then(|| d.windows.remove(i)) else {
            return;
        };
        w.title = title;
        w.route = Some(route);
        w.content = Content::Panel(panel);
        if w.state == WinState::Normal {
            w.rect.w = pw;
            w.rect.h = ph;
            w.rect.x = w.rect.x.min(screen.x + screen.w.saturating_sub(pw)).max(screen.x);
            w.rect.y = w.rect.y.min(screen.y + screen.h.saturating_sub(ph)).max(screen.y);
        }
        // Last is front, and front is focused.
        d.windows.push(w);
    });
    open_flourish();
    true
}

pub fn open(title: &str, panel: Panel) {
    let Some(fb) = super::primary() else {
        return;
    };
    let screen = screen_rect(&fb);
    let (w, h) = panel.preferred();
    let (w, h) = (w.min(screen.w), h.min(screen.h));
    with(|d| {
        // Cascade down the right-hand side rather than from the top-left
        // corner. The terminal is the widest window and starts at the left, so
        // a window cascading from the origin lands entirely behind it: opening
        // one looked like nothing had happened.
        let n = d.windows.len() as u32;
        let off = (n % 6) * 28;
        // Clear of the terminal's right edge where possible.
        //
        // Not tidiness. The console draws straight to the framebuffer on its
        // own schedule, with no idea which windows are above it, so a window
        // overlapping the terminal gets its overlapping strip repainted with
        // console output the moment anything is printed. Windows may still be
        // moved over it -- `redraw_over_terminal` repairs that between
        // commands -- but nothing should start out overlapping.
        let clear_of_terminal = d
            .windows
            .iter()
            .find(|win| matches!(win.content, Content::Terminal(c) if c == super::console::USER))
            .map(|win| win.rect.x + win.rect.w + MARGIN)
            .unwrap_or(screen.x);
        // Narrow it rather than slide it under the terminal.
        //
        // A panel is as wide as its widest widget, so a page with one long
        // line in it becomes a window wider than the gap beside the terminal
        // -- and the placement below then puts it as far right as it fits,
        // which is *underneath*, with its left half hidden behind console
        // text that repaints on its own schedule. Three separate panels hit
        // that while being written, each time looking like the window had
        // failed to draw rather than like it had been placed badly.
        //
        // Clamping the width keeps the whole window reachable: the content
        // clips at the frame, which is visible and fixable by dragging the
        // edge, where a window two thirds behind another is neither.
        // `MIN_CLEAR` is a floor, because a terminal wide enough to leave no
        // usable gap should get an overlapping window rather than a sliver.
        const MIN_CLEAR: u32 = 320;
        let gap = (screen.x + screen.w).saturating_sub(clear_of_terminal);
        let w = if gap >= MIN_CLEAR { w.min(gap) } else { w };
        let x = screen
            .x
            .max(screen.x + screen.w.saturating_sub(w))
            .max(clear_of_terminal.min(screen.x + screen.w.saturating_sub(w)));
        let y = (screen.y + screen.h / 3 + off)
            .min(screen.y + screen.h.saturating_sub(h));
        d.windows.push(Window {
            serial: next_serial(),
            app_surface: Default::default(),
        image: Default::default(),
        arrive: None,
        shown: Default::default(),
        tiled: true,
        slide: None,
            title: String::from(title),
            // `open` is reached through `win open <name>` with the panel's
            // name as the title, so the name is what there is to go by.
            icon: panel_icon(title),
            rect: Rect::new(x, y, w, h),
            state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
            content: Content::Panel(panel),
            menus: Vec::new(),
            closable: true,
        });
    });
    // The new window is what was opened, so it stays in front and focused --
    // window priority a person expects, and what every launch path relies on.
    // The keystroke that launched it (a shell Enter, a menu Enter, a click) is
    // consumed before the window exists, so there is nothing left to leak into
    // it; the earlier worry conflated "raise the window" with "steal the next
    // command", which is only a problem for a headless driver typing blind.
    open_flourish();
}

/// Open a window around a program.
///
/// The keyboard goes back to the terminal, exactly as `open` and
/// `open_browser` do -- and the first version of this function is why that
/// rule exists in triplicate. It kept focus on the new window, on the
/// reasoning that a game is opened to be played; but the desktop takes
/// *every* key while a non-terminal window has focus, so the next command
/// line -- typed or serial -- was fed to the program a byte at a time.
/// Minesweeper ate "echo after-mines" as e,c,h,o... and flagged a cell on
/// the f. The shell froze from the outside, which is exactly how the
/// browser presented when it made the same mistake. Click the window or
/// Alt-Tab to play; with a pointer that is one gesture.
pub fn open_app(title: &str, icon: usize, app: Box<dyn DeskApp>, w: u32, h: u32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);

    // Open no smaller than the program says it needs.
    //
    // The caller passes a preferred size and those were being clamped to the
    // screen and nothing else, so a program whose preferred size was stale or
    // whose chrome had grown since opened cropped, with the missing part
    // unreachable: the only resize was a 14-pixel corner grip nobody found.
    // Asking the program for its floor and honouring it here means a window
    // is born usable, and the edge dragging added alongside means it stays
    // that way.
    let inner = app.min_size();
    let need_w = inner.0 + theme::FRAME * 2;
    let need_h = inner.1 + theme::TITLE_H + theme::FRAME * 2;
    let (w, h) = (
        w.max(need_w).min(screen.w),
        h.max(need_h).min(screen.h),
    );
    with(|d| {
        let n = d.windows.len() as u32;
        let off = (n % 5) * 32;
        let x = (screen.x + (screen.w.saturating_sub(w)) / 2 + off)
            .min(screen.x + screen.w.saturating_sub(w));
        let y = (screen.y + (screen.h.saturating_sub(h)) / 3 + off)
            .min(screen.y + screen.h.saturating_sub(h));
        d.windows.push(Window {
            serial: next_serial(),
            app_surface: Default::default(),
        image: Default::default(),
        arrive: None,
        shown: Default::default(),
        tiled: true,
        slide: None,
            title: String::from(title),
            icon,
            rect: Rect::new(x, y, w, h),
            state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
            content: Content::App(app),
            menus: Vec::new(),
            closable: true,
        });
    });
    // Opened to be used -> in front and focused. To type at the shell again,
    // click the terminal, Alt-Tab, or its taskbar button, as with any window.
    open_flourish();
}

pub fn open_paint() {
    let (w, h) = super::paint::Paint::preferred();
    open_app("Paintbrush", ICO_PAINT, Box::new(super::paint::Paint::new()), w, h);
}

pub fn open_mines() {
    let (w, h) = super::mines::Mines::preferred();
    open_app("Minesweeper", ICO_MINES, Box::new(super::mines::Mines::new()), w, h);
}

pub fn open_write(path: &str) {
    let (w, h) = super::write::Writer::preferred();
    open_app("Write", ICO_WRITE, Box::new(super::write::Writer::new(path)), w, h);
}

/// The network manager. Its own icon, because it is the one window an
/// operator opens when the network is the thing that is wrong.
pub fn open_netman() {
    let (w, h) = super::netman::NetMan::preferred();
    open_app("Network", ICO_NET, Box::new(super::netman::NetMan::new()), w, h);
}

pub fn open_todo() {
    let (w, h) = super::todo::Todo::preferred();
    open_app("ToDo", ICO_TODO, Box::new(super::todo::Todo::new()), w, h);
}

pub fn open_oracle(premise: &str) {
    let (w, h) = super::oracle::Oracle::preferred();
    open_app("Oracle", ICO_ORACLE, Box::new(super::oracle::Oracle::new(premise)), w, h);
}

/// The agent transcript, live. Shares the Oracle's icon because both are
/// instruments pointed at what the machine is thinking.
///
/// Focus returns to the terminal immediately: this window is a view, it
/// consumes no keys, and a viewer that held the keyboard would eat every
/// serial command typed after opening it -- the desktop answers declined
/// keys with "handled" all the same, which is exactly how Minesweeper once
/// swallowed a whole test script.
/// Move the most recently opened window clear of the terminal.
///
/// For windows that hand focus back. `open_app` centres, which is right for a
/// program opened by a click -- it keeps focus and sits on top -- and wrong for
/// one opened on the machine's own initiative, because handing the keyboard
/// back also raises the terminal *over* it. The progress window opened
/// underneath and showed a 225-pixel strip of its right edge: no title, no
/// fields, and nothing to say it was working.
///
/// `desk::open` has carried this rule for panels since the same thing happened
/// to them. There is a second reason beyond tidiness, given there: the console
/// draws straight to the framebuffer on its own schedule with no idea which
/// windows are above it, so an overlapping window gets its overlapping strip
/// repainted with console output the moment anything is printed.
///
/// Best effort, and "best" means as far right as the screen allows rather than
/// giving up. A window wider than the gap cannot clear the terminal entirely,
/// and the first version of this declined to move one at all -- which left the
/// progress window exactly where the bug had put it, hidden, because 520 did
/// not fit in a 504-pixel gap. Sixteen pixels overlapped beats three hundred.
fn clear_of_terminal() {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    with(|d| {
        let Some(edge) = d
            .windows
            .iter()
            .find(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
            .map(|w| w.rect.x + w.rect.w + MARGIN)
        else {
            return;
        };
        if let Some(w) = d.windows.last_mut() {
            let room = screen.x + screen.w.saturating_sub(w.rect.w);
            w.rect.x = edge.min(room).max(screen.x);
        }
    });
}

/// Show the authoring loop, without taking the keyboard.
///
/// Opened by the run rather than by somebody asking, which is exactly the case
/// where focus must not move: the desktop takes every key while a non-terminal
/// window has focus, so a window appearing mid-command eats the rest of the
/// line. Minesweeper consumed `echo after-mines` a byte at a time for this
/// reason and flagged a cell on the `f`.
///
/// Idempotent. A second run must not stack a second window on the first.
/// Ask for the authoring window.
///
/// **Posts rather than opens, because its caller is the agent task.**
/// `ai/author.rs` runs on the resident agent task, and this used to reach
/// straight into the window list from there: `open_app` is a `Vec::push` that
/// can reallocate `windows` while the compositor task is walking them, which is
/// two `&mut Desktop` and undefined behaviour rather than a race. It was the
/// only genuine cross-task writer in the tree.
///
/// It also called `draw()` on the early-out, painting a frame from the agent
/// task. That is `invalidate()` now -- asking the owner to paint rather than
/// painting on its behalf.
pub fn open_authoring() {
    if has_window("Writing") {
        super::render::invalidate();
        return;
    }
    let (w, h) = super::agentwin::AuthorWin::preferred();
    post(Op::OpenApp {
        title: "Writing",
        icon: ICO_ORACLE,
        app: Box::new(super::agentwin::AuthorWin::new()),
        w,
        h,
    });
}

/// Is a window with this title already open?
pub fn has_window(title: &str) -> bool {
    with(|d| d.windows.iter().any(|w| w.title == title)).unwrap_or(false)
}

/// Put every window away, so what opens next is what is on screen.
///
/// Minimised rather than closed: the terminal and the Program Manager are
/// `closable: false` for a reason, and a workspace that destroyed the shell to
/// tidy the screen would be a workspace nobody could get out of.
pub fn minimise_all() {
    with(|d| {
        for w in d.windows.iter_mut() {
            w.state = WinState::Minimised;
        }
    });
}

/// Put every window away except the focused one.
///
/// The companion to `minimise_all`, and the reason it exists separately is
/// that "clear the screen" and "clear the screen around *this*" are the two
/// things anybody actually wants and neither is the other with an argument.
/// Minimised, not closed, for the reason above: the terminal and the Program
/// Manager are `closable: false`.
pub fn minimise_others() {
    with(|d| {
        let f = d.focus();
        for (i, w) in d.windows.iter_mut().enumerate() {
            if Some(i) != f {
                w.state = WinState::Minimised;
            }
        }
    });
    super::render::invalidate();
}

/// Put the focused window away.
pub fn minimise_focused() -> bool {
    let ok = with(|d| {
        let Some(f) = d.focus() else { return false };
        d.windows[f].state = WinState::Minimised;
        true
    })
    .unwrap_or(false);
    if ok {
        super::render::invalidate();
    }
    ok
}

pub fn open_agentlog() {
    let (w, h) = super::agentwin::AgentLog::preferred();
    open_app("Agent", ICO_ORACLE, Box::new(super::agentwin::AgentLog::new()), w, h);
    // The same latent fault: it hands focus back, so it opened under the
    // terminal too. Nobody noticed because a transcript is read after the
    // fact, by which point the window has usually been moved.
    clear_of_terminal();
    focus_terminal();
}

/// Open Enternet, optionally at a URL.
pub fn open_browser(url: &str) {
    let mut b = Browser::new();
    if !url.is_empty() {
        b.load(url);
    }
    let title = String::from("Enternet");
    with(|d| {
        let Some(fb) = super::primary() else { return };
        let screen = screen_rect(&fb);
        let w = (screen.w * 3 / 4).min(screen.w);
        let h = (screen.h * 3 / 4).min(screen.h);
        let x = screen.x + (screen.w.saturating_sub(w)) / 2;
        let y = screen.y + (screen.h.saturating_sub(h)) / 3;
        d.windows.push(Window {
            serial: next_serial(),
            app_surface: Default::default(),
        image: Default::default(),
        arrive: None,
        shown: Default::default(),
        tiled: true,
        slide: None,
            title,
            icon: ICO_NET,
            rect: Rect::new(x, y, w, h),
            state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
            content: Content::Browser(b),
            menus: Vec::new(),
            closable: true,
        });
    });
    // The keyboard goes back to the shell, exactly as `open` does.
    //
    // In front and focused, like any opened window. A browser is opened to be
    // driven, so it should have the keyboard; to return to the shell, focus
    // the terminal. (The headless driver must Alt-Tab back before typing the
    // next command -- it types blind over serial and cannot see what has the
    // keyboard.)
    open_flourish();
}

// --- the pointer ----------------------------------------------------------

/// The arrow, as a bitmap. `X` is outline, `.` is fill, space is transparent.
const CURSOR: [&str; 17] = [
    "X          ",
    "XX         ",
    "X.X        ",
    "X..X       ",
    "X...X      ",
    "X....X     ",
    "X.....X    ",
    "X......X   ",
    "X.......X  ",
    "X........X ",
    "X.....XXXXX",
    "X..X..X    ",
    "X.X X..X   ",
    "XX  X..X   ",
    "X    X..X  ",
    "     X..X  ",
    "      XX   ",
];
const CUR_W: u32 = super::compose::CUR_W;
const CUR_H: u32 = super::compose::CUR_H;

/// What the pointer is currently saying it will do.
///
/// A frame edge is six pixels of nothing until the pointer changes shape over
/// it. That change is the entire discoverability of resizing: there is no
/// visible grip to aim at, and an operator who has only ever used Windows
/// looks for the double arrow and nothing else. Held in a static because the
/// cursor is repainted on every mouse packet anyway, so switching bitmaps
/// costs a branch and no redraw.
pub const SHAPE_ARROW: u8 = 0;
const SHAPE_H: u8 = 1;
const SHAPE_V: u8 = 2;
const SHAPE_NWSE: u8 = 3;
const SHAPE_NESW: u8 = 4;
static SHAPE: Racy<u8> = Racy::new(SHAPE_ARROW);

/// The resize glyphs, on the same eleven by seventeen canvas as the arrow so
/// that saving and restoring the pixels underneath does not change with the
/// shape. Drawn as solid figures and outlined at paint time, because a thin
/// dark glyph vanishes on a dark title bar and a thin light one vanishes on
/// the wall; the halo makes each one legible on both.
const CUR_H_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "   X   X   ",
    "  XX   XX  ",
    " XXXXXXXXX ",
    "  XX   XX  ",
    "   X   X   ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
];
const CUR_V_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    "     X     ",
    "    XXX    ",
    "   XXXXX   ",
    "     X     ",
    "     X     ",
    "     X     ",
    "     X     ",
    "     X     ",
    "   XXXXX   ",
    "    XXX    ",
    "     X     ",
    "           ",
    "           ",
    "           ",
];
const CUR_NWSE_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    " XXXXXX    ",
    " XXXXX     ",
    " XXXX      ",
    " XXX       ",
    " XX        ",
    " X         ",
    "         X ",
    "        XX ",
    "       XXX ",
    "      XXXX ",
    "     XXXXX ",
    "    XXXXXX ",
    "           ",
    "           ",
];
const CUR_NESW_BITS: [&str; 17] = [
    "           ",
    "           ",
    "           ",
    "    XXXXXX ",
    "     XXXXX ",
    "      XXXX ",
    "       XXX ",
    "        XX ",
    "         X ",
    " X         ",
    " XX        ",
    " XXX       ",
    " XXXX      ",
    " XXXXX     ",
    " XXXXXX    ",
    "           ",
    "           ",
];

fn bitmap(shape: u8) -> &'static [&'static str; 17] {
    match shape {
        SHAPE_H => &CUR_H_BITS,
        SHAPE_V => &CUR_V_BITS,
        SHAPE_NWSE => &CUR_NWSE_BITS,
        SHAPE_NESW => &CUR_NESW_BITS,
        _ => &CURSOR,
    }
}

/// Which glyph a set of held edges asks for. Two adjacent edges is a corner,
/// and a corner drags along one of the two diagonals.
fn shape_for(e: u8) -> u8 {
    let l = e & edge::LEFT != 0;
    let r = e & edge::RIGHT != 0;
    let t = e & edge::TOP != 0;
    let b = e & edge::BOTTOM != 0;
    match (l, r, t, b) {
        (true, _, true, _) | (_, true, _, true) => SHAPE_NWSE,
        (_, true, true, _) | (true, _, _, true) => SHAPE_NESW,
        (true, _, _, _) | (_, true, _, _) => SHAPE_H,
        (_, _, true, _) | (_, _, _, true) => SHAPE_V,
        _ => SHAPE_ARROW,
    }
}

/// The glyph the pointer should wear at a point, given no drag is running.
///
/// Deliberately asks the same two questions the press path asks, in the same
/// order, so that the shape shown and the action taken cannot disagree: a
/// pointer that promises a resize where a click would move the window is
/// worse than no pointer feedback at all.
fn resize_shape_at(x: i32, y: i32) -> u8 {
    let Some(i) = window_at(x, y) else {
        return SHAPE_ARROW;
    };
    with(|d| {
        let w = &d.windows[i];
        if w.state == WinState::Maximised {
            return SHAPE_ARROW;
        }
        shape_for(edges_at(w.rect, x, y))
    })
    .unwrap_or(SHAPE_ARROW)
}

/// Pixels the cursor is currently covering, and where.
///
/// Saved and restored rather than repainted. The desktop redraws everything
/// back to front, which is a few million stores; doing that per mouse packet
/// at a hundred reports a second is a system that cannot be used. Only the
/// eleven by seventeen rectangle under the arrow is touched.
static SAVED: Racy<[u32; (CUR_W * CUR_H) as usize]> = Racy::new([0; (CUR_W * CUR_H) as usize]);
static SHOWN: Racy<Option<(u32, u32)>> = Racy::new(None);
/// Where the pointer is, whether or not it is currently painted. `draw` uses
/// this to put the arrow back after a repaint.
static POS: Racy<Option<(u32, u32)>> = Racy::new(None);

/// Held for as long as anything is touching `SAVED`, `SHOWN`, or the screen.
///
/// The cursor used to be safe by accident: `poll_mouse` was the only caller
/// and it ran on the shell task, so nothing could interleave. `pump_cursor`
/// broke that -- it runs from inside `generate`, which the agent task also
/// runs, so a cursor move could land in the middle of the shell's `draw`.
/// The interleaving is not a flicker: the arrow is painted after `present`,
/// the pixels under it are saved from the *pre*-present frame, and the next
/// hide stamps that stale block onto the screen and then saves it again as
/// the new background -- so the damage is copied forward every frame, and in
/// the worst ordering what gets saved is the arrow itself, leaving one
/// permanent ghost per occurrence.
/// Who holds it, rather than merely that somebody does.
///
/// It was an `AtomicBool`, which is the right shape while the only holders are
/// two small paints that can shrug and try again. It stopped being the right
/// shape once a compositor task began holding it for a whole frame and the
/// window state started needing the same protection: `draw` takes the claim at
/// the top and then calls `with` ninety-one times underneath, so a claim that
/// could not recognise its own holder would refuse itself instantly.
///
/// The owner is a task id, and re-entry from the same task is free -- which is
/// exactly the bargain `ai::with_engine` already makes, for the same reason
/// and with the same hazard: nesting stays forbidden between *tasks* and
/// costless within one.
const NOBODY: usize = usize::MAX;
static PAINTER: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(NOBODY);

/// Releases the claim however the holder leaves, and only if it took it.
struct Claim {
    /// False for a re-entry: the outer claim owns the release, and storing
    /// `NOBODY` here would free the screen in the middle of somebody's frame.
    took: bool,
}

impl Claim {
    /// `None` when **another task** holds it. Never waits.
    ///
    /// This is what the clock task and the cursor use, and they must never
    /// block: they run on a schedule with no event behind them, and a painter
    /// that waited would be waiting on the task it preempted.
    fn take() -> Option<Claim> {
        Self::take_as(crate::task::current())
    }

    /// Split out so the exclusion can be asked about a task that is not this
    /// one. A second `take()` from the same task is now *supposed* to succeed,
    /// so it can no longer answer "is a second holder refused" -- see
    /// `selftest`.
    fn take_as(me: usize) -> Option<Claim> {
        use core::sync::atomic::Ordering::{AcqRel, Acquire};
        match PAINTER.compare_exchange(NOBODY, me, AcqRel, Acquire) {
            Ok(_) => Some(Claim { took: true }),
            // Ours already: a frame in progress, or a nested `with`.
            Err(h) if h == me => Some(Claim { took: false }),
            Err(_) => None,
        }
    }

    /// Wait for it, yielding.
    ///
    /// For changing the desktop, which cannot sensibly refuse: a window that
    /// declined to open because a frame happened to be in flight would be a
    /// worse bug than the race this prevents. Yielding rather than spinning,
    /// because the holder only runs when this task gives the core up -- a spin
    /// here is a hang, which is the objection the non-blocking version was
    /// written around.
    fn wait() -> Claim {
        loop {
            if let Some(c) = Self::take() {
                return c;
            }
            crate::task::yield_now();
        }
    }
}

/// Who holds the paint claim, if anybody.
///
/// Exposed so a stuck compositor can name what is stopping it. `Claim` is
/// reentrant for its own holder, so a leaked claim is invisible from the task
/// that leaked it and fatal to every other one -- which is the shape that made
/// this take four runs to find.
pub fn claim_holder() -> Option<usize> {
    match PAINTER.load(core::sync::atomic::Ordering::Relaxed) {
        NOBODY => None,
        h => Some(h),
    }
}

/// Open the paint claim this task was holding. **Recovery path only.**
///
/// **A caught fault leaves the claim held, and nothing was putting it back.**
/// `recover::land` restores a stack pointer and jumps, so no destructor runs
/// and `Claim::drop` never happens -- exactly the reason stated one screen up
/// for why the compositor's frame is not wrapped in a guard. It is also true
/// of every *other* guarded scope that touches the desktop, and `diag all`
/// runs several that fault on purpose.
///
/// What it looks like is the interesting part, because it does not look like a
/// lock. The claim is reentrant for its holder, so the task that leaked it
/// goes on working perfectly -- `take_as(me)` matches and hands back a
/// non-owning claim forever. Every *other* task blocks in `Claim::wait`, which
/// is a `yield_now` loop, so it is scheduled constantly and progresses never.
/// Measured: the compositor resumed 2,194 times across one `diag all` and
/// completed essentially zero turns of its loop, while the shell that had
/// leaked the claim finished all sixty-eight suites without noticing.
///
/// Only this task's, by compare-exchange. A blanket store would open a claim
/// another task is legitimately holding, and two painters in one back buffer
/// is the thing the claim exists to prevent -- a recovery that caused it would
/// be worse than the leak.
///
/// **It assumes no `guard` is nested inside a held claim**, and that is true of
/// all twenty guarded scopes today: they are boot sections, diag suites, the
/// JIT, the paging check, hostile parsing and repair, and not one of them runs
/// underneath a `with`. A future one that did would have its outer claim
/// opened underneath it by a fault in its inner scope, which is the two-painter
/// race rather than a leak. The honest fix then is to snapshot the holder at
/// `guard` entry and restore it, not to widen this.
///
/// # Safety
/// Only after a `recover::guard` has caught a fault, when the code that held
/// this cannot resume.
pub unsafe fn release_claim() {
    use core::sync::atomic::Ordering::{AcqRel, Acquire};
    let me = crate::task::current();
    let _ = PAINTER.compare_exchange(me, NOBODY, AcqRel, Acquire);
}

impl Drop for Claim {
    fn drop(&mut self) {
        if self.took {
            PAINTER.store(NOBODY, core::sync::atomic::Ordering::Release);
        }
    }
}

/// The one thing about the desktop that can be checked without a screen.
///
/// Everything else here is pixels, and a claim about pixels needs eyes. This
/// is arithmetic: the cursor statics are shared between the shell task and
/// whichever task is generating, and the whole defence is that two holders
/// cannot exist at once. That property was argued for and never checked --
/// the race it prevents needs a concurrent generation and a moving mouse to
/// reproduce, which no boot test can arrange -- so at least the exclusion
/// itself is asserted rather than assumed.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    // **A second take from this task is now supposed to succeed**, so the old
    // form of this check -- take twice, expect a refusal -- stopped asking
    // about exclusion the day the claim became reentrant, and would have gone
    // on passing for the wrong reason. The property that matters was always
    // "another *task* is refused", which a same-task call cannot answer.
    // `take_as` exists so it can be asked without arranging a second task.
    let me = crate::task::current();
    let other = me.wrapping_add(1);
    // The compositor holds the claim for the whole of each frame, so "free"
    // has to be waited for: with `render verify on` a frame is slow enough
    // that taking it cold lost the race often enough to fail the suite. A
    // second's patience, then the claim is taken as before.
    let deadline = crate::time::rdtsc() + crate::time::tsc_mhz().max(1) as u64 * 1_000_000;
    let mut first = Claim::take();
    while first.is_none() && crate::time::rdtsc() < deadline {
        crate::task::yield_now();
        first = Claim::take();
    }
    claim("a free claim can be taken", first.is_some());
    claim("and another task is refused", Claim::take_as(other).is_none());
    // Reentrancy, asserted rather than assumed: `draw` holds this across a
    // whole frame and calls `with` ninety-one times underneath, so a claim
    // that refused its own holder would deadlock the compositor on its first
    // frame.
    claim("while this task may re-enter", Claim::take().is_some());
    drop(first);
    // The compositor may take it in between, which is the claim working;
    // what is asked is that it can be had again at all.
    let deadline = crate::time::rdtsc() + crate::time::tsc_mhz().max(1) as u64 * 1_000_000;
    let mut again = Claim::take();
    while again.is_none() && crate::time::rdtsc() < deadline {
        crate::task::yield_now();
        again = Claim::take();
    }
    claim("dropping it lets the next one in", again.is_some());
    drop(again);
    // Leaving it held would deadlock every later repaint, so the release path
    // is worth one more claim of its own -- and a re-entry must *not* release
    // it, which is what `Claim::took` is for. Asked of this task: the
    // compositor holding it a moment later is not this task leaking it.
    claim(
        "and this task no longer holds it",
        PAINTER.load(core::sync::atomic::Ordering::Relaxed) != me,
    );

    // --- tiling -----------------------------------------------------------
    //
    // Pure geometry, checked here rather than by dragging windows into corners
    // and looking. The odd width is the case that matters: two halves computed
    // as a width applied twice leave a column of wallpaper showing between
    // windows that are supposed to meet.
    {
        let screen = Rect::new(10, 20, 101, 51);
        let l = tile_rect(Tile::Left, screen);
        let r = tile_rect(Tile::Right, screen);
        claim("halves start at the screen and end at it", l.x == 10 && r.x + r.w == 111);
        claim("and meet exactly, on an odd width", l.x + l.w == r.x);
        let tl = tile_rect(Tile::TopLeft, screen);
        let bl = tile_rect(Tile::BottomLeft, screen);
        claim("quadrants meet vertically too", tl.y + tl.h == bl.y);
        claim("and the bottom row reaches the foot", bl.y + bl.h == 71);
        claim("full is the screen", tile_rect(Tile::Full, screen) == screen);
        // The four quadrants must cover the screen exactly: no overlap and no
        // gap, which is the property a grid of windows is judged on.
        let area: u32 = [Tile::TopLeft, Tile::TopRight, Tile::BottomLeft, Tile::BottomRight]
            .iter()
            .map(|t| {
                let q = tile_rect(*t, screen);
                q.w * q.h
            })
            .sum();
        claim("and they tile it with no gap or overlap", area == screen.w * screen.h);
    }

    // --- the command queue keeps the oldest ---------------------------------
    //
    // It was a single slot, so a burst of presses kept only the last one. The
    // claim that matters is which end is dropped when it fills: the whole
    // point of the change is that an early press is not silently lost, so
    // overflowing must cost the *newest*.
    {
        while take_pending().is_some() {}
        for i in 0..PENDING_CAP + 1 {
            let mut c = String::from("q");
            c.push_str(&alloc::format!("{}", i));
            queue_command(&c);
        }
        claim("the queue fills to its cap and no further", pending_len() == PENDING_CAP);
        let first = take_pending();
        claim("and the oldest press is the one that survives", first.as_deref() == Some("q0"));
        let mut last = first;
        while let Some(n) = take_pending() {
            last = Some(n);
        }
        claim(
            "while the one past the cap was refused",
            last.as_deref() == Some(&alloc::format!("q{}", PENDING_CAP - 1)[..]),
        );
        claim("and draining leaves it empty", pending_len() == 0);
    }

    // --- the icon tables agree ---------------------------------------------
    //
    // `draw_icons` draws `pictogram(k)` and writes `ICONS[k].0` under it, so a
    // constant that does not match its row puts one icon's name under another
    // icon's drawing. That is not a hypothetical: `ICO_SET` was 8 while
    // `ICONS[8]` was "Oracle", so the sliders were labelled Oracle and the eye
    // was labelled Settings, on every screenshot this project ever took.
    //
    // Checked by name rather than by index, because the index is the thing
    // that was wrong.
    {
        let named: [(usize, &str); 10] = [
            (ICO_TERM, "Terminal"),
            (ICO_PROGRAMS, "Programs"),
            (ICO_FILES, "Files"),
            (ICO_TODO, "ToDo"),
            (ICO_NET, "Enternet"),
            (ICO_PAINT, "Paint"),
            (ICO_WRITE, "Write"),
            (ICO_MINES, "Mines"),
            (ICO_ORACLE, "Oracle"),
            (ICO_SET, "Settings"),
        ];
        let mut agree = true;
        for (k, label) in named.iter() {
            agree &= ICONS[*k].0 == *label;
        }
        claim("every icon wears its own label", agree);
        // And no two share an index, which is the other way to break it.
        let mut distinct = true;
        for (i, (a, _)) in named.iter().enumerate() {
            for (b, _) in named.iter().skip(i + 1) {
                distinct &= a != b;
            }
        }
        claim("and no two icons share an index", distinct);
    }

    ok
}

/// The pointer's pixels for a shape, encoded, `None` where it is clear. The
/// arrow carries its outline in the bitmap; the resize glyphs are solid
/// figures given a dark halo here, so each reads on both a dark title bar and
/// the light wall.
/// The pointer plane's image for `shape`, premultiplied ARGB with the
/// pointer's position at `(HOT_X, HOT_Y)`.
///
/// The arrow is drawn, not looked up: an anti-aliased outline with a dark rim
/// and a soft shadow below and to the right, at the 1.25 scale this release
/// lays everything out for -- the old one was an eleven-by-seventeen bitmap in
/// two colours. It is built once and kept, because `cursor_show` runs on every
/// pointer packet and the drawing is a few thousand distance evaluations. The
/// resize glyphs are still bitmaps, converted, with their top-left where the
/// pointer is, as it always was.
fn cursor_image(fb: &Framebuffer, shape: u8) -> super::compose::Plane {
    use super::compose::{HOT_X, HOT_Y, PLANE};
    if shape == SHAPE_ARROW {
        static ARROW: Racy<Option<super::compose::Plane>> = Racy::new(None);
        let cache = unsafe { &mut *ARROW.get() };
        if cache.is_none() {
            *cache = Some(arrow_plane(fb));
        }
        return cache.unwrap();
    }
    let bits = bitmap(shape);
    let set = |r: i32, c: i32| -> bool {
        if r < 0 || c < 0 || r >= CUR_H as i32 || c >= CUR_W as i32 {
            return false;
        }
        matches!(bits[r as usize].as_bytes().get(c as usize), Some(b'X') | Some(b'.'))
    };
    let mut out = [0u32; (PLANE * PLANE) as usize];
    for row in 0..CUR_H {
        let line = bits[row as usize].as_bytes();
        for col in 0..CUR_W {
            let v = match line.get(col as usize) {
                Some(b'.') | Some(b'X') => Some(fb.raw(theme::HILIGHT)),
                _ => {
                    let (r, c) = (row as i32, col as i32);
                    let near = (-1..=1).any(|dr| (-1..=1).any(|dc| (dr, dc) != (0, 0) && set(r + dr, c + dc)));
                    near.then(|| fb.raw(theme::TEXT))
                }
            };
            if let Some(v) = v {
                out[((row + HOT_Y) * PLANE + col + HOT_X) as usize] = super::blend::opaque(v);
            }
        }
    }
    out
}

/// The arrow's outline, in logical pixels from its tip: down the left edge,
/// in to the notch, out along the tail, back to the notch, and across to the
/// wing. The proportions are the familiar ones; only the drawing is new.
const ARROW_PTS: [(f32, f32); 7] = [(0.0, 0.0), (0.0, 17.0), (4.0, 13.0), (7.0, 20.0), (10.0, 19.0), (7.0, 12.0), (12.0, 12.0)];

/// Signed distance from `(x, y)` to the arrow at `scale`: negative inside.
fn arrow_sd(x: f32, y: f32, scale: f32) -> f32 {
    let n = ARROW_PTS.len();
    let mut d = f32::MAX;
    let mut inside = false;
    for i in 0..n {
        let (ax, ay) = (ARROW_PTS[i].0 * scale, ARROW_PTS[i].1 * scale);
        let (bx, by) = (ARROW_PTS[(i + 1) % n].0 * scale, ARROW_PTS[(i + 1) % n].1 * scale);
        let (ex, ey) = (bx - ax, by - ay);
        let (px, py) = (x - ax, y - ay);
        let t = ((px * ex + py * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
        let (qx, qy) = (px - ex * t, py - ey * t);
        d = d.min(qx * qx + qy * qy);
        // Crossing number, for the sign.
        if (ay > y) != (by > y) && x < ax + (y - ay) * ex / ey {
            inside = !inside;
        }
    }
    let d = super::shape::libm_sqrt(d);
    if inside { -d } else { d }
}

fn arrow_plane(fb: &Framebuffer) -> super::compose::Plane {
    use super::compose::{HOT_X, HOT_Y, PLANE};
    let scale = 1.25f32;
    let rim = 1.25f32;
    let (sx, sy, sigma) = (1.0f32, 2.0f32, 1.6f32);
    let fill = super::blend::opaque(fb.raw(Color::new(255, 255, 255)));
    let ink = super::blend::opaque(fb.raw(Color::new(20, 24, 32)));
    let mut out = [0u32; (PLANE * PLANE) as usize];
    for py in 0..PLANE {
        for px in 0..PLANE {
            // Pixel centre, relative to the tip.
            let (x, y) = (px as f32 + 0.5 - HOT_X as f32, py as f32 + 0.5 - HOT_Y as f32);
            let d = arrow_sd(x, y, scale);
            // The shadow: the outline, offset and softened by its own distance
            // field rather than blurred -- a smoothstep across a sigma either
            // side of the edge, at about half black.
            let ds = arrow_sd(x - sx, y - sy, scale);
            let t = ((ds + sigma) / (3.0 * sigma)).clamp(0.0, 1.0);
            let shade = (1.0 - t * t * (3.0 - 2.0 * t)) * 0.55;
            let mut p = super::blend::premul(0, (shade * 255.0 + 0.5) as u8);
            // The rim, then the white body inside it.
            let rim_cov = (0.5 - d).clamp(0.0, 1.0);
            if rim_cov > 0.0 {
                p = super::blend::over_px(super::blend::scale_px(ink, (rim_cov * 255.0 + 0.5) as u32), p);
            }
            let body = (0.5 - (d + rim)).clamp(0.0, 1.0);
            if body > 0.0 {
                p = super::blend::over_px(super::blend::scale_px(fill, (body * 255.0 + 0.5) as u32), p);
            }
            out[(py * PLANE + px) as usize] = p;
        }
    }
    out
}

pub fn cursor_hide(fb: &Framebuffer) {
    // With a compositor the pointer is a plane over the back buffer and is
    // never painted into anything, so there is nothing to lift: a frame
    // repaints under it and the present lays it back over whatever changed.
    if super::compose::cursor_plane() {
        return;
    }
    let Some((x, y)) = (unsafe { *SHOWN.get() }) else { return };
    // Ask the compositor for those pixels back, rather than trusting a copy
    // taken earlier.
    //
    // The back buffer is by definition what the screen is supposed to show, so
    // restoring from it cannot be stale -- and staleness was the whole bug
    // class here. `pump_cursor` runs between tokens while the console is
    // streaming into the same region; every flushed line made `SAVED` describe
    // a frame that no longer existed, and putting it back punched a 17x11
    // block of old pixels into the answer text that nothing ever repaired,
    // because the compositor believed those pixels were already correct.
    if !super::compose::repaint_rect(x, y, CUR_W, CUR_H) {
        let saved = unsafe { &*SAVED.get() };
        for row in 0..CUR_H {
            for col in 0..CUR_W {
                if x + col < fb.width() && y + row < fb.height() {
                    fb.put(x + col, y + row, saved[(row * CUR_W + col) as usize]);
                }
            }
        }
    }
    unsafe { *SHOWN.get() = None };
}

pub fn cursor_show(fb: &Framebuffer, x: u32, y: u32) {
    let shape = unsafe { *SHAPE.get() };
    if super::compose::cursor_plane() {
        super::compose::set_cursor(Some((x, y)), &cursor_image(fb, shape));
        return;
    }
    cursor_hide(fb);
    let bits = bitmap(shape);
    // Only needed when there is no compositor to ask. With one, this loop is
    // 187 uncached reads off the aperture per mouse move -- on the path that
    // runs between every generated token -- to fill a buffer nothing reads.
    if !super::compose::active() {
        let saved = unsafe { &mut *SAVED.get() };
        // Save the whole box before painting anything. The halo pass writes
        // pixels the figure never covers, and restoring is only correct if
        // every pixel that might be written was read first.
        for row in 0..CUR_H {
            for col in 0..CUR_W {
                let (px, py) = (x + col, y + row);
                if px < fb.width() && py < fb.height() {
                    saved[(row * CUR_W + col) as usize] = fb.get(px, py);
                }
            }
        }
    }
    let set = |r: i32, c: i32| -> bool {
        if r < 0 || c < 0 || r >= CUR_H as i32 || c >= CUR_W as i32 {
            return false;
        }
        matches!(bits[r as usize].as_bytes().get(c as usize), Some(b'X') | Some(b'.'))
    };
    for row in 0..CUR_H {
        let line = bits[row as usize].as_bytes();
        for col in 0..CUR_W {
            let (px, py) = (x + col, y + row);
            if px >= fb.width() || py >= fb.height() {
                continue;
            }
            match line.get(col as usize) {
                // The arrow carries its own outline in the bitmap.
                Some(b'X') if shape == SHAPE_ARROW => fb.put(px, py, fb.raw(theme::TEXT)),
                Some(b'.') => fb.put(px, py, fb.raw(theme::HILIGHT)),
                Some(b'X') => fb.put(px, py, fb.raw(theme::HILIGHT)),
                _ if shape != SHAPE_ARROW => {
                    // Blank, but touching the figure: this is the halo.
                    let (r, c) = (row as i32, col as i32);
                    let near = set(r - 1, c)
                        || set(r + 1, c)
                        || set(r, c - 1)
                        || set(r, c + 1)
                        || set(r - 1, c - 1)
                        || set(r - 1, c + 1)
                        || set(r + 1, c - 1)
                        || set(r + 1, c + 1);
                    if near {
                        fb.put(px, py, fb.raw(theme::TEXT));
                    }
                }
                _ => {}
            }
        }
    }
    unsafe { *SHOWN.get() = Some((x, y)) };
}

/// When the tray readouts were last painted, in tenths of a second.
///
/// `Racy` and not an atomic because it is touched by one task -- which is the
/// whole point of this section and is now true by construction rather than by
/// a claim. It was two tasks until the readouts moved here.
static TRAY_AT: Racy<Option<u64>> = Racy::new(None);

/// The taskbar's uptime and charge, painted straight rather than composed.
///
/// **This is the last painter that was not the compositor, and moving it is
/// what retires the paint claim's reason for existing.** It ran on the clock
/// task, which wakes on its own quantum whatever the shell is doing, so it and
/// `draw` were two tasks writing one back buffer through a `&mut` neither knew
/// the other held. The claim is what stood between them. On this task there is
/// nobody to exclude.
///
/// It also ends the symptom the whole rearrangement started from: a frozen
/// desktop with an uptime still ticking in the corner of it. The readouts are
/// the compositor's now, so a compositor that stops takes them with it, and a
/// stopped screen looks stopped. What still reports liveness is the watchdog,
/// which says so in words on a channel the screen cannot take away.
///
/// **A part of the taskbar layer, stamped by the text it shows.** This only
/// notices that a tenth of a second has passed and asks for a frame; the
/// scene diff then finds the clock part's stamp moved and repaints that well
/// alone. It is asked once per tenth and never "after a frame": the tray is
/// in the frame, so a frame cannot erase it, and asking unconditionally after
/// every frame made each frame request the next.
pub fn paint_tray() {
    // The boot screen owns the framebuffer while it is up, and an uptime in
    // the corner of a splash is the tell that something is drawing behind the
    // curtain.
    if super::splash::active() {
        return;
    }
    let Some(fb) = super::primary() else { return };

    // The rate check comes before the exclusive check on purpose. This is
    // called every frame and the tray changes ten times a second, so counting
    // a refusal per *call* would report a few thousand of them for a minute of
    // DOOM and swamp the counter that exists to say a full-screen program was
    // running. One per tenth is the same fact at the tray's own rate.
    let tenths = crate::dev::lapic::ticks() * 10 / crate::TIMER_HZ as u64;
    if unsafe { *TRAY_AT.get() } == Some(tenths) {
        return;
    }
    // A full-screen program owns the screen outright, and the tray is not on
    // it. `TRAY_AT` is deliberately left alone here: recording this tenth
    // would make the tray believe it was up to date, and it would stay blank
    // when the program gave the screen back until the clock happened to tick.
    if super::exclusive() {
        super::render::refused();
        return;
    }
    unsafe { *TRAY_AT.get() = Some(tenths) };
    super::render::tray_painted();
    // A new tenth: the taskbar layer's clock part has a new stamp, so the
    // next frame repaints that well and nothing else.
    let _ = fb;
    super::render::invalidate();
}

/// The two readouts' text, right now. One function for the scene's stamp and
/// the painter, so they cannot disagree about what is shown.
fn tray_strings() -> (alloc::string::String, Option<alloc::string::String>) {
    let tenths = crate::dev::lapic::ticks() * 10 / crate::TIMER_HZ as u64;
    // Short, because the taskbar reserves a fixed well for it and every
    // character of that well is a character the task buttons do not get.
    let clock = alloc::format!(" up {}.{}s ", tenths / 10, tenths % 10);
    // The charge, in its own well beside the clock. The reading behind it is
    // cached, so asking ten times a second costs a comparison rather than a
    // run of the firmware's bytecode.
    (clock, battery_text())
}

/// Paint both readouts into the frame.
fn paint_tray_text(fb: &Framebuffer) {
    if super::splash::active() {
        return;
    }
    let cs = theme::CHROME_SCALE;
    let (clock, battery) = tray_strings();
    tray_text(fb, clock_rect(fb), &clock, cs);
    if let (Some(b), Some(t)) = (battery_rect(fb), battery) {
        tray_text(fb, b, &t, cs);
    }
}

/// One readout, centred in its well.
///
/// Characters and not bytes for the width. Both wells hold ASCII today and
/// `text.len()` was what the clock used, but a byte count is not a column
/// count -- this tree has paid for that distinction in three separate places,
/// and the charge string is the one here most likely to grow a symbol.
fn tray_text(fb: &Framebuffer, well: Rect, text: &str, scale: u32) {
    // Measured in the interface face; a readout that does not fit is not
    // drawn rather than cut, since half a clock is a wrong clock.
    let width = theme::text_w_of_at(text, scale);
    if well.w <= width {
        return;
    }
    let x = well.x + (well.w - width) / 2;
    let y = well.y + (well.h.saturating_sub(theme::text_h_at(scale))) / 2;
    // Into the compositor's back buffer like everything else, on the tray's
    // own ground: the inner well is laid down first, so a reading shorter
    // than the last one leaves none of the last one's digits behind.
    fb.rect(well.x + 1, well.y + 1, well.w.saturating_sub(2), well.h.saturating_sub(2), theme::TRAY);
    theme::text_over_at(fb, x, y, text, theme::TITLE_TEXT, scale);
}

/// Put the arrow somewhere, without letting anything else paint while it does.
///
/// Interrupts off for the whole of it, so the timer cannot switch tasks
/// between the save and the paint -- the body is a few hundred pixels, and the
/// alternative is a repaint landing inside it and saving the arrow's own
/// pixels as the background it will later restore.
///
/// `false` when a repaint already owns the screen. Losing costs one frame of
/// pointer lag and nothing else, so the caller leaves `POS` alone and tries
/// again rather than recording a move it never drew.
fn move_cursor(x: u32, y: u32) -> bool {
    // As with the clock: while something owns the screen, the pointer is not
    // on it. Answering `true` rather than `false` so the caller records the
    // new position and does not spend the rest of the session retrying a move
    // it can never make.
    if super::exclusive() {
        return true;
    }
    crate::cpu::without_interrupts(|| {
        let Some(_claim) = Claim::take() else { return false };
        let Some(fb) = super::primary() else { return false };
        cursor_show(&fb, x, y);
        super::render::cursor_painted();
        true
    })
}

/// One pointer move, for `video bench` and the `video.cursor_move` rail.
///
/// Alternates between two spots a few pixels apart so every call really moves
/// the arrow, and puts it back where the hand left it afterwards. Measured
/// through `move_cursor`, the same path a real packet takes, so the number is
/// what a person dragging the mouse pays per update -- today a hide and a show
/// straight on the aperture, and after phase 6 a re-present of two small
/// rectangles.
pub fn bench_cursor(step: u32) {
    let Some(fb) = super::primary() else { return };
    let (x, y) = (fb.width() / 2 + (step & 1) * 7, fb.height() / 2 + (step & 1) * 5);
    move_cursor(x, y);
}

/// Put the arrow back where the pointer says it is, after `bench_cursor`.
pub fn cursor_home() {
    if let Some((x, y)) = unsafe { *POS.get() } {
        move_cursor(x, y);
    }
}

pub fn poll_mouse() {
    use crate::dev::mouse;
    if !mouse::present() || !ready() {
        return;
    }
    let s = mouse::take();
    if !s.moved {
        // Nothing new, but a paint may be owed from a turn the pacing refused.
        if unsafe { *CURSOR_PENDING.get() } {
            if let Some((x, y)) = unsafe { *POS.get() } {
                paint_cursor_paced(x, y);
            }
        }
        return;
    }
    // Still a precondition even though the framebuffer is no longer touched
    // from here: `move_cursor` fetches its own, under the claim.
    if super::primary().is_none() {
        return;
    }
    let (x, y) = (s.x.max(0), s.y.max(0));
    unsafe { *POS.get() = Some((x as u32, y as u32)) };

    // A press is an edge, not a level: the mouse reports the button held down
    // in every packet while it is down, and acting on the level would fire a
    // click for every packet of a drag.
    let was = unsafe { *BUTTONS.get() };
    unsafe { *BUTTONS.get() = (s.left, s.right) };
    let pressed_left = s.left && !was.0;
    let released_left = !s.left && was.0;
    let pressed_right = s.right && !was.1;

    if s.wheel != 0 {
        wheel_at(x, y, s.wheel);
    }

    let dragging = matches!(
        with(|d| d.mode),
        Some(Mode::Drag { .. }) | Some(Mode::DragSize { .. })
    );
    let in_app = unsafe { *APP_PRESS.get() };
    if pressed_left {
        press_at(x, y);
    } else if s.left && dragging {
        drag_to(x, y);
    } else if s.left && in_app {
        // The stroke: motion with the button held, inside the program that
        // took the press.
        app_drag_to(x, y);
    }
    if released_left && dragging {
        let was_move = matches!(with(|d| d.mode), Some(Mode::Drag { .. }));
        with(|d| d.mode = Mode::Normal);
        if was_move {
            snap_release(x, y);
        }
        // The resulting rectangle, not just the fact that a drag ended. A
        // resize that silently does nothing and a resize that ran and was
        // clamped back look identical from outside; the numbers separate them.
        if let Some(Some(r)) = with(|d| d.focus().map(|f| d.windows[f].rect)) {
            crate::serial_println!("[desk] drag end {}x{}+{}+{}", r.w, r.h, r.x, r.y);
        }
        trace("drag end");
        super::render::invalidate();
    }
    if released_left && in_app {
        unsafe { *APP_PRESS.get() = false };
        app_release();
    }
    if pressed_right {
        right_click_at(x, y);
    }
    // Hover feedback only while no button is down: mid-drag the pointer
    // crosses half the screen, and highlighting everything on the way would
    // be noise.
    if !s.left {
        update_hover(x, y);
    }
    // Pick the glyph after the event has been acted on, so a press that
    // started a resize is already reflected in the mode.
    let want = match with(|d| d.mode) {
        Some(Mode::DragSize { edges, .. }) => shape_for(edges),
        Some(Mode::Drag { .. }) => SHAPE_ARROW,
        _ => resize_shape_at(x, y),
    };
    unsafe { *SHAPE.get() = want };
    // Same claim the pump takes. This runs on the shell task and the pump runs
    // on whichever task is generating, so without it the two interleave in
    // exactly the way the cursor statics cannot survive. Paced, not per event
    // -- see `CURSOR_PAINTED_AT`.
    paint_cursor_paced(x as u32, y as u32);
}

static BUTTONS: Racy<(bool, bool)> = Racy::new((false, false));
/// When the arrow was last painted, in TSC ticks, and whether a move has been
/// recorded since that is still owed a paint.
///
/// **The cursor is coalesced to one paint per `CURSOR_MIN_US`, and it had to
/// be.** `cursor_show` writes the arrow straight into the aperture and
/// `cursor_hide` repaints what was under it from the back buffer, so every
/// paint is an erase and a redraw on the one surface a person is looking at.
/// While the pointer was read from the shell's idle loop that happened at most
/// once a tick; read from the compositor's loop it happened once per HID
/// report, several hundred times a second, and the arrow strobed -- and each
/// erase copied whatever the back buffer held at that instant, which during a
/// `draw()` on another task is half a frame, so what the pointer crossed
/// flickered with it. Every position is still recorded and every press still
/// acts at once; only the painting waits, and the last position is always
/// painted on a later turn so the arrow never stops short of the hand.
static CURSOR_PAINTED_AT: Racy<u64> = Racy::new(0);
static CURSOR_PENDING: Racy<bool> = Racy::new(false);
/// Four milliseconds: 250 paints a second, which no hand can tell from
/// unlimited, and a tenth of what a 1 kHz mouse was asking for.
const CURSOR_MIN_US: u64 = 4_000;

/// Paint the arrow at `POS` if enough time has passed since the last paint;
/// otherwise remember that one is owed.
fn paint_cursor_paced(x: u32, y: u32) {
    let now = crate::time::rdtsc();
    let min = crate::time::tsc_mhz().max(1) * CURSOR_MIN_US;
    let last = unsafe { *CURSOR_PAINTED_AT.get() };
    if now.wrapping_sub(last) < min {
        unsafe { *CURSOR_PENDING.get() = true };
        return;
    }
    if move_cursor(x, y) {
        unsafe {
            *CURSOR_PAINTED_AT.get() = now;
            *CURSOR_PENDING.get() = false;
        }
    }
}
/// The previous press, for double-click detection: milliseconds and place.
static LAST_CLICK: Racy<(u64, i32, i32)> = Racy::new((0, -100, -100));

fn now_ms() -> u64 {
    let mhz = crate::time::tsc_mhz();
    if mhz == 0 {
        return 0;
    }
    crate::time::rdtsc() / (mhz * 1000)
}

/// True when this press pairs with the previous one as a double-click.
fn is_double(x: i32, y: i32) -> bool {
    let now = now_ms();
    let (t, lx, ly) = unsafe { *LAST_CLICK.get() };
    unsafe { *LAST_CLICK.get() = (now, x, y) };
    now.saturating_sub(t) < 400 && (x - lx).abs() < 5 && (y - ly).abs() < 5
}

/// Where a window's title bar, menu bar and client area are.
///
/// `theme::chrome` owns the formula now, because the painter and the
/// hit-tester both need it and they used to hold two copies in two files. This
/// is the shim that keeps the four callers reading as they did.
fn chrome(frame: Rect, has_menus: bool) -> (Rect, Option<Rect>, Rect) {
    let c = theme::chrome(frame, has_menus);
    (c.title, c.menubar, c.client)
}

/// Which menu-bar label sits under a point, from the same iterator the paint
/// loop walks.
fn menu_label_at(menus: &[Menu], bar: Rect, x: i32, y: i32) -> Option<usize> {
    if y < bar.y as i32 || y >= (bar.y + bar.h) as i32 {
        return None;
    }
    theme::menu_labels(bar, menus.iter().map(|m| m.label.as_str()))
        .position(|r| x >= r.x as i32 && x < (r.x + r.w) as i32)
}

/// Which sides of a frame a resize drag has hold of.
///
/// A bitmask rather than an enum of nine cases, because a corner is genuinely
/// two edges at once and the arithmetic that moves them is the same code
/// twice.
pub mod edge {
    pub const NONE: u8 = 0;
    pub const LEFT: u8 = 1 << 0;
    pub const RIGHT: u8 = 1 << 1;
    pub const TOP: u8 = 1 << 2;
    pub const BOTTOM: u8 = 1 << 3;
}

/// How close to a border counts as grabbing it.
///
/// Six pixels is what the desktops this imitates used, and it is the number a
/// hand trained on them expects to find. The corner zones are wider because a
/// corner is the only way to change both dimensions at once, and hunting for
/// a 6x6 square is the difference between a window manager that resizes and
/// one that appears not to.
const BORDER: i32 = 6;
const CORNER: i32 = 16;

/// Raise the executive console, restoring it if it is minimised.
///
/// Only ever called because somebody asked. Nothing in the system raises this
/// window on its own: a window that appears while an operator is typing is
/// the problem the split was made to solve, arriving by another route.
pub fn show_executive() {
    let Some(fb) = super::primary() else {
        return;
    };
    let screen = screen_rect(&fb);
    let existing = with(|d| {
        d.windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::EXEC))
    })
    .flatten();
    // Created on demand like the terminal: a clean desktop opens with no
    // Executive window, but its console has carried the boot log the whole
    // time, so opening one here shows everything printed before the shell.
    with(|d| match existing {
        Some(i) => {
            d.windows[i].state = WinState::Normal;
            d.raise(i);
        }
        None => {
            let e = make_executive(screen);
            d.windows.push(e);
            let n = d.windows.len() - 1;
            d.raise(n);
        }
    });
    super::render::invalidate();
}

/// The Executive (boot-log) window, built on demand. One place its shape is
/// described; `show_executive` is the one caller.
fn make_executive(screen: Rect) -> Window {
    let icons_w = 10 + ICON_W + 10;
    let term_x = screen.x.max(icons_w);
    let term_w = (screen.x + screen.w).saturating_sub(term_x + MARGIN);
    Window {
        serial: next_serial(),
        app_surface: Default::default(),
        image: Default::default(),
        arrive: None,
        shown: Default::default(),
        tiled: true,
        slide: None,
        title: String::from("Executive"),
        icon: ICO_TERM,
        rect: Rect::new(
            term_x + MARGIN,
            screen.y + MARGIN,
            term_w.saturating_sub(MARGIN * 2).max(320),
            screen.h.saturating_sub(MARGIN * 4).max(200),
        ),
        state: WinState::Normal,
        snap_back: None,
        route: None,
        round: theme::WIN_ROUND,
        content: Content::Terminal(super::console::EXEC),
        menus: alloc::vec![Menu {
            label: String::from("View"),
            items: alloc::vec![
                item("Clear", "exec clear"),
                item("Save log", "log save /tmp/boot.txt"),
                item("Redraw", "refresh"),
            ],
        }],
        closable: true,
    }
}

/// Round the focused window's corners, or square them again with zero.
pub fn set_round(r: u32) -> bool {
    let ok = with(|d| {
        let Some(f) = d.focus() else { return false };
        d.windows[f].round = r;
        true
    })
    .unwrap_or(false);
    if ok {
        super::render::invalidate();
    }
    ok
}

/// The outline for the window being painted, kept between frames.
///
/// One entry, not one per window: `draw` paints windows one at a time and
/// asks for each outline in turn, so a single slot serves as long as it
/// records what it was built for. Two windows with different radii alternate
/// and rebuild each time, which is two square roots per row of an arc and
/// still nothing next to the paint it is wrapping.
static SHAPE_CACHE: Racy<Option<(u32, u32, u32)>> = Racy::new(None);
static OUTLINE: Racy<Option<super::Shape>> = Racy::new(None);

/// Paint `f` clipped to this window's outline, or straight through when the
/// window is an ordinary rectangle.
fn with_window_shape<R>(fb: &Framebuffer, w: &Window, frame: Rect, f: impl FnOnce() -> R) -> R {
    // A maximised window has square corners and nothing behind it to show
    // through them. XP squares them too.
    if w.round == 0 || w.state == WinState::Maximised {
        return f();
    }
    let key = (frame.w, frame.h, w.round);
    let stale = unsafe { *SHAPE_CACHE.get() } != Some(key);
    if stale {
        unsafe {
            // Only ever `round_top` through this slot, which is why the key
            // does not name the kind. Somebody adding a second shape here has
            // to add a fourth field, and the failure if they forget is a
            // window with the wrong corners -- visible, and not looking like a
            // cache bug.
            *OUTLINE.get() = Some(super::Shape::round_top(frame.w, frame.h, w.round));
            *SHAPE_CACHE.get() = Some(key);
        }
    }
    let Some(shape) = (unsafe { (*OUTLINE.get()).as_ref() }) else {
        return f();
    };
    super::with_shape(fb, shape, frame.x as i32, frame.y as i32, f)
}

/// Which borders the pointer is over, if any.
///
/// The whole frame edge, not one corner grip. A 14x14 square in the
/// bottom-right was the only way to resize anything, which is discoverable if
/// you already know it is there and invisible otherwise -- the operator's
/// report was that windows could not be resized at all, only maximised.
fn edges_at(frame: Rect, x: i32, y: i32) -> u8 {
    let (l, t) = (frame.x as i32, frame.y as i32);
    let (r, b) = (l + frame.w as i32, t + frame.h as i32);
    // Outside the frame, or far enough inside that this is content.
    if x < l - 1 || x > r + 1 || y < t - 1 || y > b + 1 {
        return edge::NONE;
    }

    let mut e = edge::NONE;
    // Corners first: within CORNER of two borders claims both, so the
    // diagonal drag is available along a usable stretch of each side.
    let near_l = (x - l).abs() <= CORNER;
    let near_r = (r - x).abs() <= CORNER;
    let near_t = (y - t).abs() <= CORNER;
    let near_b = (b - y).abs() <= CORNER;
    let corner_h = (x - l).abs() <= BORDER || (r - x).abs() <= BORDER;
    let corner_v = (y - t).abs() <= BORDER || (b - y).abs() <= BORDER;

    if corner_h && near_t && near_b {
        // A window shorter than two corner zones: prefer the nearer edge.
        if y - t < b - y {
            e |= edge::TOP;
        } else {
            e |= edge::BOTTOM;
        }
    } else if corner_h && near_t {
        e |= edge::TOP;
    } else if corner_h && near_b {
        e |= edge::BOTTOM;
    }
    if corner_v && near_l && near_r {
        if x - l < r - x {
            e |= edge::LEFT;
        } else {
            e |= edge::RIGHT;
        }
    } else if corner_v && near_l {
        e |= edge::LEFT;
    } else if corner_v && near_r {
        e |= edge::RIGHT;
    }

    // Plain edges, for the long stretch between the corners.
    if (x - l).abs() <= BORDER {
        e |= edge::LEFT;
    }
    if (r - x).abs() <= BORDER {
        e |= edge::RIGHT;
    }
    if (y - t).abs() <= BORDER {
        e |= edge::TOP;
    }
    if (b - y).abs() <= BORDER {
        e |= edge::BOTTOM;
    }
    e
}

/// The smallest a window may be dragged to.
///
/// Asked of the program inside it, because the program is the only thing that
/// knows. Minesweeper's board does not reflow and Paint's canvas does not
/// shrink; a window manager that let either be dragged to 160 pixels wide
/// would be hiding their content behind a frame the operator cannot undo
/// without maximising.
fn min_size_of(w: &Window) -> (u32, u32) {
    let inner = match &w.content {
        Content::App(a) => a.min_size(),
        _ => (240, 120),
    };
    (
        inner.0 + theme::FRAME * 2,
        inner.1 + theme::TITLE_H + theme::FRAME * 2,
    )
}

/// The bottom-right corner that resizes, generous enough to hit.
fn size_grip(frame: Rect) -> Rect {
    let g = 14u32;
    Rect::new(
        frame.x + frame.w.saturating_sub(g),
        frame.y + frame.h.saturating_sub(g),
        g,
        g,
    )
}

fn contains(r: Rect, x: i32, y: i32) -> bool {
    x >= r.x as i32 && y >= r.y as i32 && x < (r.x + r.w) as i32 && y < (r.y + r.h) as i32
}

/// Topmost visible window containing a point.
fn window_at(x: i32, y: i32) -> Option<usize> {
    with(|d| {
        d.windows.iter().rposition(|w| {
            w.state != WinState::Minimised
                && x >= w.rect.x as i32
                && y >= w.rect.y as i32
                && x < (w.rect.x + w.rect.w) as i32
                && y < (w.rect.y + w.rect.h) as i32
        })
    })
    .flatten()
}

/// Route one left press: dropdowns, the Start menu, the taskbar, window
/// chrome, and finally the content under the point. This is the pointer's
/// whole vocabulary.
fn press_at(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let double = is_double(x, y);
    // A new press starts a new gesture; whatever the last one was is over.
    unsafe { *APP_PRESS.get() = false };
    // The one trace on this path: presses are the pointer's whole vocabulary,
    // and a click that silently does nothing is undebuggable from a
    // screenshot. Serial only, like every other trace here.
    crate::serial_println!("[desk] press {},{}{}", x, y, if double { " double" } else { "" });

    // An open menu eats the press: on an item it acts, anywhere else it
    // closes. Both must come before window routing or the press falls through
    // the menu onto whatever is behind it.
    if menu_press(&fb, x, y, screen) {
        super::render::invalidate();
        return;
    }

    if contains(taskbar_rect(&fb), x, y) {
        task_press(&fb, x, y);
        super::render::invalidate();
        return;
    }

    let Some(i) = window_at(x, y) else {
        // The wall, or an icon on it. Icons open on a double press, as they
        // have since 95 -- a single press is selection, which here is the
        // hover highlight already showing.
        with(|d| d.mode = Mode::Normal);
        if double {
            if let Some(k) = icon_at(&fb, x, y) {
                launch(ICONS[k].1);
            }
        }
        super::render::invalidate();
        return;
    };

    // Raising is focusing here, which is the one fact the window manager
    // keeps. Everything below acts on the window at its new index.
    with(|d| {
        d.raise(i);
        d.mode = Mode::Normal;
    });
    let f = with(|d| d.windows.len() - 1).unwrap_or(0);

    let got = with(|d| {
        let w = &d.windows[f];
        (w.frame(screen), !w.menus.is_empty(), w.state)
    });
    let Some((frame, has_menus, state)) = got else { return };
    let (title, menubar, client) = chrome(frame, has_menus);

    if contains(title, x, y) {
        // The caption buttons eat their presses before the bar means "drag".
        let hit = theme::caption_buttons(title)
            .iter()
            .position(|b| contains(*b, x, y));
        if let Some(which) = hit {
            caption_action(f, which);
            super::render::invalidate();
            return;
        }
        if double {
            with(|d| {
                d.windows[f].state = match d.windows[f].state {
                    WinState::Maximised => WinState::Normal,
                    _ => WinState::Maximised,
                };
            });
        } else if state != WinState::Maximised {
            // Remember where inside the frame the pointer took hold, so the
            // window follows the grab instead of snapping a corner to it.
            // Taking hold of a tiled window lifts it out of the layout, as
            // Hyprland's drag does: a window under the pointer has to be
            // free to follow it, and its slot closes up behind it.
            with(|d| {
                d.windows[f].tiled = false;
                d.mode = Mode::Drag {
                    dx: (x - frame.x as i32).max(0) as u32,
                    dy: (y - frame.y as i32).max(0) as u32,
                    from: frame,
                }
            });
        }
        super::render::invalidate();
        return;
    }

    if state != WinState::Maximised {
        let e = edges_at(frame, x, y);
        if e != edge::NONE {
            with(|d| {
                d.windows[f].tiled = false;
                d.mode = Mode::DragSize { from: frame, edges: e }
            });
            super::render::invalidate();
            return;
        }
    }

    if let Some(bar) = menubar {
        let hit = with(|d| menu_label_at(&d.windows[f].menus, bar, x, y)).flatten();
        if let Some(mi) = hit {
            with(|d| d.mode = Mode::Menu { menu: mi, item: 0 });
            super::render::invalidate();
            return;
        }
    }

    // The content. Each kind answers the press in its own terms.
    let step = with(|d| match &mut d.windows[f].content {
        Content::Panel(p) => p.mouse(client, x, y, double),
        Content::Browser(b) => {
            if b.click(client, x, y) {
                ui::Step::Redraw
            } else {
                ui::Step::Idle
            }
        }
        Content::App(a) => {
            app_input(a.as_ref());
            if a.press(Rect::new(0, 0, client.w, client.h), x - client.x as i32, y - client.y as i32) {
                // The press begins a gesture: until the button lifts, motion
                // belongs to this program. That is what a brush stroke is.
                unsafe { *APP_PRESS.get() = true };
                ui::Step::Redraw
            } else {
                ui::Step::Idle
            }
        }
        // A press in the terminal is focus, which raising already did.
        Content::Terminal(_) => ui::Step::Idle,
    })
    .unwrap_or(ui::Step::Idle);
    act_on(step);
    super::render::invalidate();
}

/// Whether the held left button is a gesture inside an app's client area,
/// as opposed to a window drag or nothing.
static APP_PRESS: Racy<bool> = Racy::new(false);

/// Forward held-button motion to the focused program.
fn app_drag_to(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        let frame = d.windows[f].frame(screen);
        let has_menus = !d.windows[f].menus.is_empty();
        let (_, _, client) = chrome(frame, has_menus);
        match &mut d.windows[f].content {
            Content::App(a) => {
                app_input(a.as_ref());
                a.drag(Rect::new(0, 0, client.w, client.h), x - client.x as i32, y - client.y as i32)
            }
            _ => false,
        }
    })
    .unwrap_or(false);
    if changed {
        super::render::invalidate();
    }
}

fn app_release() {
    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        match &mut d.windows[f].content {
            Content::App(a) => {
                app_input(a.as_ref());
                a.release()
            }
            _ => false,
        }
    })
    .unwrap_or(false);
    if changed {
        super::render::invalidate();
    }
}

/// One caption button, pressed. The same three verbs the system menu
/// offers, bound to the three squares 98 put them on.
fn caption_action(f: usize, which: usize) {
    crate::serial_println!("[desk] caption {} on window {}", which, f);
    with(|d| {
        if f >= d.windows.len() {
            return;
        }
        match which {
            0 => d.windows[f].state = WinState::Minimised,
            1 => {
                d.windows[f].state = match d.windows[f].state {
                    WinState::Maximised => WinState::Normal,
                    _ => WinState::Maximised,
                };
            }
            _ => {
                if d.windows[f].closable {
                    close_at(d, f);
                } else {
                    // The terminal is the shell: it keeps running with no
                    // window, so its X means "put it away", which is the
                    // honest version of close for a program that cannot die.
                    d.windows[f].state = WinState::Minimised;
                }
            }
        }
    });
}

/// Carry out what a panel handed back. The same arms the keyboard path runs,
/// in one place, so a pointer activation and an Enter cannot diverge.
fn act_on(step: ui::Step) {
    match step {
        ui::Step::Do(Action::Run(cmd)) => {
            focus_terminal();
            queue_command(&cmd);
        }
        ui::Step::Do(Action::Browse(route)) => {
            if let Some((title, panel)) = ui::panel_for_route(&route) {
                let kind = route.split(':').next().unwrap_or("");
                with(|d| {
                    if let Some(f) = d.focus() {
                        d.windows[f].title = title;
                        // Navigation changes what the window *is*, and the
                        // taskbar shows windows by what they are.
                        d.windows[f].icon = panel_icon(kind);
                        d.windows[f].content = Content::Panel(panel);
                    }
                });
            }
        }
        ui::Step::Close => close_focused(),
        _ => {}
    }
}

/// Close the focused window, or minimise it when it cannot be closed.
///
/// One function because there are two ways to press a Close button and they
/// used to do different things: clicking it removed the window, while Enter on
/// it (and Esc) fell to `cycle`, which alt-tabs -- so Cancel on a dialog left
/// the dialog open, one place further back. The same control has to mean the
/// same thing however it is pressed; that rule is why the pointer and the paint
/// pass share their layout functions, and it applies to the keyboard too.
/// Close every window whose program carries this tag. Answers how many.
pub fn close_tagged(tag: u64) -> usize {
    if tag == 0 {
        return 0;
    }
    let n = with(|d| {
        let before = d.windows.len();
        d.windows.retain(|w| !matches!(&w.content, Content::App(a) if a.tag() == tag));
        before - d.windows.len()
    })
    .unwrap_or(0);
    if n > 0 {
        super::render::invalidate();
    }
    n
}

/// Retitle the window whose program carries this tag.
pub fn retitle_tagged(tag: u64, title: &str) {
    if tag == 0 {
        return;
    }
    with(|d| {
        for w in d.windows.iter_mut() {
            if matches!(&w.content, Content::App(a) if a.tag() == tag) {
                w.title = String::from(title);
            }
        }
    });
    super::render::invalidate();
}

/// What a Super chord does.
///
/// Hyprland's binds where they do not collide with this desktop's: Enter a
/// terminal, Q close, V float, J turn the split, F full screen, the arrows
/// focus the window that way and with Shift swap places with it, D the
/// desktop. Super-Tab becomes the Flip view (phase 15); until then it
/// switches windows. A chord with no binding is still the desktop's and is
/// dropped, so a reader of the plan does not find Super-V typing a 'v'.
fn super_key(k: u8) {
    let Some((key, shift)) = kbd::chord_of(k) else { return };
    match (key, shift) {
        (b'\n', _) => open_terminal(),
        (b'q', _) => close_focused(),
        (b'd', _) => show_desktop(),
        (b'\t', _) => {
            with(|d| d.mode = Mode::Normal);
            cycle(false);
        }
        (b'v', _) => toggle_float(),
        (b'j', _) => toggle_split(),
        (b'f', _) => {
            with(|d| {
                if let Some(f) = d.focus() {
                    let w = &mut d.windows[f];
                    w.state = if w.state == WinState::Maximised { WinState::Normal } else { WinState::Maximised };
                }
            });
        }
        (k, shift) if k == kbd::KEY_LEFT => tile_move(0, shift),
        (k, shift) if k == kbd::KEY_RIGHT => tile_move(1, shift),
        (k, shift) if k == kbd::KEY_UP => tile_move(2, shift),
        (k, shift) if k == kbd::KEY_DOWN => tile_move(3, shift),
        _ => {}
    }
    super::render::invalidate();
}

/// Super-D: put every window away, and the second press brings back exactly
/// those that were showing -- not every window, which would also restore the
/// ones somebody had minimised on purpose.
fn show_desktop() {
    let away = unsafe { &mut *SHOWN_AWAY.get() };
    with(|d| {
        let showing: Vec<(u64, WinState)> = d
            .windows
            .iter()
            .filter(|w| w.state != WinState::Minimised)
            .map(|w| (w.serial, w.state))
            .collect();
        if showing.is_empty() && !away.is_empty() {
            for w in d.windows.iter_mut() {
                if let Some((_, was)) = away.iter().find(|(s, _)| *s == w.serial) {
                    w.state = *was;
                }
            }
            away.clear();
        } else {
            for w in d.windows.iter_mut() {
                w.state = WinState::Minimised;
            }
            *away = showing;
        }
    });
}

/// Which windows Super-D put away, and how each was showing -- a maximised
/// window comes back maximised. Only touched under the desktop's claim.
static SHOWN_AWAY: Racy<Vec<(u64, WinState)>> = Racy::new(Vec::new());

pub fn close_focused() {
    with(|d| {
        if let Some(f) = d.focus() {
            if d.windows[f].closable {
                close_at(d, f);
            } else {
                d.windows[f].state = WinState::Minimised;
            }
        }
    });
    super::render::invalidate();
}

/// A press on the taskbar: the Start button, an app, or a window button.
/// The focused window's own button minimises it, which is the bar's one
/// toggle and the reason it can replace stowed icons.
fn task_press(fb: &Framebuffer, x: i32, y: i32) {
    if contains(start_rect(fb), x, y) {
        with(|d| {
            d.mode = match d.mode {
                Mode::Start { .. } => Mode::Normal,
                // Opens on the query row, because pressing Start and typing is
                // the common case and arrowing up into the items is one key
                // either way.
                _ => Mode::Start { item: START_ITEMS.len() },
            }
        });
        return;
    }
    let hit = with(|d| {
        task_layout(fb, d)
            .into_iter()
            .enumerate()
            .find(|(_, (r, ..))| contains(*r, x, y))
            .map(|(i, _)| i)
    })
    .flatten();
    with(|d| d.mode = Mode::Normal);
    let Some(w) = hit else { return };
    with(|d| {
        if w >= d.windows.len() {
            return;
        }
        if d.windows[w].state == WinState::Minimised {
            d.windows[w].state = WinState::Normal;
            d.raise(w);
        } else if d.focus() == Some(w) {
            d.windows[w].state = WinState::Minimised;
        } else {
            d.raise(w);
        }
    });
}

/// Where the open dropdown is -- window menu, system menu, or Start -- and how
/// many rows it holds. Mirrors the paint pass; they must not disagree.
fn dropdown_rows(fb: &Framebuffer, d: &Desktop, screen: Rect) -> Option<theme::Popup> {
    if let Mode::Start { .. } = d.mode {
        return Some(start_menu_rect(fb));
    }
    let f = d.focus()?;
    let frame = d.windows[f].frame(screen);
    let inner = frame.shrink(theme::FRAME);
    match d.mode {
        Mode::Menu { menu, .. } => {
            let m = d.windows[f].menus.get(menu)?;
            let bar = Rect::new(inner.x, inner.y + theme::TITLE_H + 2, inner.w, MENU_H);
            let x = theme::menu_labels(bar, d.windows[f].menus.iter().map(|p| p.label.as_str()))
                .nth(menu)?
                .x;
            let cols = m.items.iter().map(|i| i.label.chars().count()).max().unwrap_or(4);
            let y = inner.y + theme::TITLE_H + 2 + MENU_H;
            Some(theme::Popup::sized(x, y, cols, m.items.len(), fb.width().saturating_sub(x)))
        }
        Mode::Sys { .. } => {
            let cols = SYS_ITEMS.iter().map(|s| s.chars().count()).max().unwrap_or(4);
            let y = inner.y + theme::TITLE_H;
            Some(theme::Popup::sized(
                inner.x,
                y,
                cols,
                SYS_ITEMS.len(),
                fb.width().saturating_sub(inner.x),
            ))
        }
        _ => None,
    }
}

/// Which dropdown row a point is in, mirroring `dropdown`'s row layout.


/// A press while any menu is open. Returns true when consumed.
fn menu_press(fb: &Framebuffer, x: i32, y: i32, screen: Rect) -> bool {
    let open = with(|d| {
        matches!(d.mode, Mode::Menu { .. } | Mode::Sys { .. } | Mode::Start { .. })
    })
    .unwrap_or(false);
    if !open {
        return false;
    }
    let mut run: Option<String> = None;
    with(|d| {
        let Some(p) = dropdown_rows(fb, d, screen) else {
            d.mode = Mode::Normal;
            return;
        };
        let Some(item) = p.item_at(x, y) else {
            d.mode = Mode::Normal;
            return;
        };
        match d.mode {
            Mode::Start { .. } => {
                if is_query_row(item) {
                    // Clicking the box puts the keyboard in it and leaves the
                    // menu open. Closing on a click into a text field would be
                    // the one interaction nobody expects.
                    d.mode = Mode::Start { item };
                } else {
                    d.mode = Mode::Normal;
                    d.query.clear();
                    run = Some(String::from(START_ITEMS[item].1));
                }
            }
            Mode::Menu { menu, .. } => {
                d.mode = Mode::Normal;
                if let Some(f) = d.focus() {
                    if let Some(mi) = d.windows[f].menus[menu].items.get(item) {
                        if let Action::Run(cmd) = &mi.action {
                            queue_command(cmd);
                        }
                    }
                }
            }
            Mode::Sys { .. } => {
                d.mode = Mode::Normal;
                if let Some(f) = d.focus() {
                    sys_action(d, f, item, screen);
                }
            }
            _ => {}
        }
    });
    if let Some(cmd) = run {
        launch(&cmd);
    }
    true
}

/// Where a tile goes.
///
/// A window manager that can only maximise makes an operator measure two
/// windows by hand to put them side by side. `snap_release` has answered half
/// of that since it was written -- drag to an edge, take half the screen --
/// and this is the rest: the corners, and a way to ask without a pointer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Full,
}

/// The rectangle a tile occupies on `screen`.
///
/// Pure, so the geometry is checkable at boot with no framebuffer and no
/// desktop -- the alternative being to check it by dragging windows into
/// corners and looking at them.
///
/// Halves are computed as *positions* rather than as a width applied twice:
/// `screen.w / 2` used twice leaves a column unpainted on an odd width, and
/// the gap shows as a stripe of wallpaper between two windows that are
/// supposed to meet.
fn tile_rect(t: Tile, screen: Rect) -> Rect {
    let midx = screen.x + screen.w / 2;
    let midy = screen.y + screen.h / 2;
    let right_w = (screen.x + screen.w).saturating_sub(midx);
    let bottom_h = (screen.y + screen.h).saturating_sub(midy);
    match t {
        Tile::Full => screen,
        Tile::Left => Rect::new(screen.x, screen.y, screen.w / 2, screen.h),
        Tile::Right => Rect::new(midx, screen.y, right_w, screen.h),
        Tile::TopLeft => Rect::new(screen.x, screen.y, screen.w / 2, screen.h / 2),
        Tile::TopRight => Rect::new(midx, screen.y, right_w, screen.h / 2),
        Tile::BottomLeft => Rect::new(screen.x, midy, screen.w / 2, bottom_h),
        Tile::BottomRight => Rect::new(midx, midy, right_w, bottom_h),
    }
}

/// Put the focused window in a tile.
pub fn tile_focused(t: Tile) -> bool {
    let Some(fb) = super::primary() else { return false };
    let screen = screen_rect(&fb);
    let ok = with(|d| {
        let Some(f) = d.focus() else { return false };
        let w = &mut d.windows[f];
        // Remembered the way a drag-snap remembers, so a tiled window dragged
        // off an edge afterwards comes back to the size it chose for itself
        // rather than to a half screen.
        w.snap_back = w.snap_back.or(Some(w.rect));
        w.state = WinState::Normal;
        w.rect = tile_rect(t, screen);
        true
    })
    .unwrap_or(false);
    if ok {
        super::render::invalidate();
    }
    ok
}

/// The four quadrants, filled from the front of the stack back.
///
/// Answers how many it placed. Capped at four deliberately: a fifth window has
/// no quadrant, and shrinking the grid to fit an arbitrary count produces
/// panes too small to read any of these windows in.
pub fn tile_all() -> usize {
    let Some(fb) = super::primary() else { return 0 };
    let screen = screen_rect(&fb);
    const ORDER: [Tile; 4] = [Tile::TopLeft, Tile::TopRight, Tile::BottomLeft, Tile::BottomRight];
    let n = with(|d| {
        // Front to back, so the window last used takes the first quadrant
        // rather than whichever happens to be deepest in the stack.
        let mut idx: Vec<usize> = (0..d.windows.len())
            .filter(|i| d.windows[*i].state != WinState::Minimised)
            .collect();
        idx.reverse();
        idx.truncate(ORDER.len());
        for (slot, i) in idx.iter().enumerate() {
            let w = &mut d.windows[*i];
            w.snap_back = w.snap_back.or(Some(w.rect));
            w.state = WinState::Normal;
            w.rect = tile_rect(ORDER[slot], screen);
        }
        idx.len()
    })
    .unwrap_or(0);
    if n > 0 {
        super::render::invalidate();
    }
    n
}

/// Four windows in the shape of a workbench rather than four equal quarters.
///
/// Quarters are the wrong arrangement for these windows and the reason is not
/// taste: they carry different amounts. A run list is a column of short names,
/// a conversation is the thing being read, a log is a tail, and a ledger is a
/// column of facts. Giving each a quarter gives the list three times the room
/// it needs and the conversation half of what it wants.
///
/// So: a narrow rail either side, the wide pane in the middle for what is
/// being read, and a short strip along the foot of it for what is streaming.
/// That is the layout every workbench has converged on, and it converged for
/// this reason.
///
/// Answers how many it placed.
pub fn tile_workspace(rail_l: &str, main: &str, foot: &str, rail_r: &str) -> usize {
    let Some(fb) = super::primary() else { return 0 };
    let screen = screen_rect(&fb);
    let lw = screen.w * 22 / 100;
    let rw = screen.w * 24 / 100;
    let midx = screen.x + lw;
    let midw = screen.w.saturating_sub(lw + rw);
    let foot_h = screen.h * 30 / 100;
    let mainh = screen.h.saturating_sub(foot_h);
    let places = [
        (rail_l, Rect::new(screen.x, screen.y, lw, screen.h)),
        (main, Rect::new(midx, screen.y, midw, mainh)),
        (foot, Rect::new(midx, screen.y + mainh, midw, foot_h)),
        (rail_r, Rect::new(screen.x + lw + midw, screen.y, rw, screen.h)),
    ];
    let n = with(|d| {
        let mut placed = 0;
        for (title, r) in places.iter() {
            if let Some(i) = d.windows.iter().position(|w| w.title == *title) {
                let w = &mut d.windows[i];
                w.snap_back = w.snap_back.or(Some(w.rect));
                w.state = WinState::Normal;
                w.rect = *r;
                placed += 1;
            }
        }
        placed
    })
    .unwrap_or(0);
    if n > 0 {
        super::render::invalidate();
    }
    n
}

/// Where the terminal's status strip goes inside its well, or `None` when the
/// window is too short to give a row to anything but the console.
///
/// One formula, because two would be the bug this tree keeps paying for: the
/// draw pass reflows the grid into what this leaves, and `refresh_status`
/// repaints into what it returns. A disagreement between them is a strip
/// painted over the last line of output.
fn terminal_strip(well: Rect) -> Option<Rect> {
    let h = theme::text_h_at(1) + 7;
    let inner = well.shrink(3);
    if inner.h <= h * 4 {
        return None;
    }
    Some(Rect::new(well.x, well.y + well.h - h, well.w, h))
}

/// Repaint the strip alone, without a full desktop draw.
///
/// The strip reports state a command changes -- `initiative off` is the
/// obvious one -- and `desk::draw` runs on window operations and pointer
/// motion, neither of which a serial session or a typed command produces. So
/// the strip was showing what was true when the desktop was last laid out,
/// which for `mind on` is not a stale field, it is a false statement.
///
/// Its own flush, because nothing else is going to present this rectangle:
/// the console flushes only its grid.
pub fn refresh_status() {
    // A full-screen program owns every pixel, the clock and the pointer already
    // stand down for it, and so does this: on a miner image it painted the
    // terminal's gutter and status strip ("no model | mind on") over the
    // address screen after every line typed.
    if super::exclusive() {
        return;
    }
    // The gutter and the status strip are parts of the terminal's layer,
    // repainted by any composed frame. Asking for one is all this needs to do;
    // it used to paint them itself, into the aperture and then the back
    // buffer, with geometry of its own.
    super::render::invalidate();
}

/// A tick in the margin beside every row that starts a command.
///
/// A terminal is an undifferentiated wall of text and always has been, and the
/// thing an operator actually looks for in one is where the last command
/// began. Marking it costs one rect per prompt and needs nothing from the
/// shell: `rows_starting` asks the grid, and a row that begins with the prompt
/// is the start of a run by construction.
///
/// In the three pixels between the well and the grid, which were margin. So
/// this adds no chrome and takes no space from the console.
/// Which rows of the main terminal begin with a prompt, the row height and
/// the grid's top: everything the gutter reads, for its painter and its stamp.
fn gutter_marks() -> ([bool; 96], u32, u32) {
    let mut starts = [false; 96];
    let mut row_h = 0u32;
    let mut oy = 0u32;
    super::console::with_ch(super::console::USER, |c| {
        row_h = c.rows_starting(crate::shell::PROMPT, &mut starts);
        oy = c.origin().1;
    });
    (starts, row_h, oy)
}

/// The gutter's and the strip's inputs as this frame's scene saw them.
///
/// Taken once, in `layers`, and read by the painters, so the stamp, the
/// incremental paint and `render verify`'s reference paint all draw the same
/// readings. Read live, a prompt the shell printed between the paint and the
/// reference made the two disagree about one gutter tick, and the verifier
/// reported a difference that was its own race.
static FRAME_INPUTS: Racy<Option<(([bool; 96], u32, u32), (String, String, bool))>> = Racy::new(None);

fn frame_marks() -> ([bool; 96], u32, u32) {
    unsafe { (*FRAME_INPUTS.get()).as_ref().map(|f| f.0) }.unwrap_or_else(gutter_marks)
}

fn frame_status() -> (String, String, bool) {
    unsafe { (*FRAME_INPUTS.get()).as_ref().map(|f| f.1.clone()) }.unwrap_or_else(status_text)
}

fn prompt_gutter(fb: &super::Framebuffer, well: Rect, grid: Rect) {
    let (starts, row_h, oy) = frame_marks();
    if row_h == 0 {
        return;
    }
    let x = well.x + 1;
    // The strip is cleared first, every time. It lives in the well's margin
    // and not in the console's own rectangle, so nothing the console does
    // repaints it -- `clear` fills `pixel_size()` and leaves the margin
    // exactly as it was, which left a column of ticks beside an empty screen
    // pointing at prompts that had been erased.
    fb.rect(x, grid.y, 2, grid.h, theme::SCREEN);
    for (r, hit) in starts.iter().enumerate() {
        if !hit {
            continue;
        }
        let y = oy + r as u32 * row_h;
        if y < grid.y || y + row_h > grid.y + grid.h {
            continue;
        }
        fb.rect(x, y + 1, 2, row_h.saturating_sub(2), theme::APERTURE);
    }
}

/// What the terminal is sitting on top of, along the foot of its window.
///
/// Every reading here is in `glance`'s free tier -- an atomic or a clone of a
/// small static -- and that is not an efficiency note, it is the correctness
/// argument. This runs inside `desk::draw`, which repaints on pointer motion;
/// anything that went through `with_engine` from here would *claim* the engine
/// out from under a running episode between two of its own calls, and the
/// failure is an interleaved decode and a corrupted cache rather than an
/// error. The model line is the one costly reading and `glance` takes it once,
/// while nothing holds the engine, and keeps it.
///
/// The strip is where the model facts moved to. They were a window before, and
/// a window is the wrong shape for four constants that never change: they are
/// the answer to "what am I talking to", which is a question asked at a glance
/// and never twice.
fn terminal_status(fb: &super::Framebuffer, r: Rect) {
    if r.w < 120 {
        return;
    }
    fb.vgrad(r.x, r.y, r.w, r.h, &theme::STATUS_BAR);
    fb.rect(r.x, r.y, r.w, 1, theme::TRAY_EDGE);
    let ty = r.y + (r.h.saturating_sub(theme::text_h_at(1))) / 2;

    let (left, right, engine) = frame_status();
    theme::text_over_at(fb, r.x + 6, ty, &left, theme::TITLE_TEXT, 1);
    // Right-aligned, and laid out from the right edge inwards so a long
    // model line runs out of room before it can overwrite the state --
    // which is the half that changes and therefore the half worth seeing.
    let w = theme::text_w_of_at(&right, 1);
    if r.w > w + 24 {
        let x = r.x + r.w - w - 6;
        let tone = if engine { theme::APERTURE } else { theme::TITLE_TEXT };
        theme::text_over_at(fb, x, ty, &right, tone, 1);
    }
}

/// The strip's two lines and whether the engine is held: everything its
/// painter reads, so the scene can stamp the strip with it.
fn status_text() -> (String, String, bool) {
    crate::ai::glance::with_glance(|g| {
        let left = crate::ai::glance::model_line(g);
        let engine = crate::ai::glance::engine_line(g);
        let mind = if g.mind_on { "mind on" } else { "mind off" };
        // A terminal showing something other than its last line has to say so.
        // Without it the only evidence is that nothing new appears, which reads
        // as a hung machine rather than as a scrolled one.
        let back = super::console::view_of(super::console::USER);
        let right = if back > 0 {
            alloc::format!("^ {} back  |  {}  |  {}", back, engine, mind)
        } else {
            alloc::format!("{}  |  {}", engine, mind)
        };
        (left, right, g.engine.is_some())
    })
}

/// The shortest period, in milliseconds, at which an open program asked for
/// its stamp to be looked at (`DeskApp::wake_ms`), as of the last frame; 0 for
/// none. Read by the compositor's pacing.
static APP_WAKE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

pub fn app_wake_ms() -> Option<u32> {
    match APP_WAKE.load(core::sync::atomic::Ordering::Relaxed) {
        0 => None,
        ms => Some(ms),
    }
}

/// Whether the compositor has work due within the next few milliseconds that
/// no `invalidate` will announce: something part-way through an animation,
/// or a pointer paint the 4 ms pacing deferred. While either is true it must
/// not sleep a whole timer tick, or the motion stutters and the arrow lags.
pub fn busy() -> bool {
    ANIMATING.load(core::sync::atomic::Ordering::Relaxed) || unsafe { *CURSOR_PENDING.get() }
}

/// Whether anything is moving: a window arriving or one leaving. Kept by
/// `anim_step`, read by the compositor's pacing without the desktop.
static ANIMATING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// A window that has closed and is still fading out: its last image, where
/// it was, and its two tweens. It outlives the window by the length of the
/// fade, which is the whole of Hyprland's `windowsOut` -- a window that
/// vanished in the frame after the click would have nothing to fade.
struct Ghost {
    serial: u64,
    frame: Rect,
    img: super::scale::Image,
    out: (super::anim::Tween, super::anim::Tween),
    shown: core::cell::Cell<(f32, f32)>,
}

static GHOSTS: Racy<Vec<Ghost>> = Racy::new(Vec::new());

/// A window's image with its outline as alpha: opaque inside the shape the
/// direct paint is clipped to, clear outside it -- what a scaled or faded
/// window has to be, since its corners now fall over something else.
fn premultiplied(win: &Window, im: &WinImage) -> super::scale::Image {
    let (w, h) = (im.frame.w, im.frame.h);
    let shape = (win.round != 0 && win.state != WinState::Maximised)
        .then(|| super::Shape::round_top(w, h, win.round));
    let mut px = Vec::with_capacity(w as usize * h as usize);
    for y in 0..h {
        let (a, b) = match &shape {
            Some(sh) => sh.span(y).unwrap_or((0, 0)),
            None => (0, w),
        };
        let row = &im.px[(y * w) as usize..((y + 1) * w) as usize];
        for (x, &p) in row.iter().enumerate() {
            let x = x as u32;
            px.push(if x >= a && x < b { p | 0xFF00_0000 } else { 0 });
        }
    }
    super::scale::Image { w, h, px }
}

/// A window has closed: keep its image fading where it stood.
fn depart(w: Window) {
    if !super::render::enabled() || super::exclusive() {
        return;
    }
    let img = {
        let slot = w.image.borrow();
        match slot.as_ref() {
            Some(im) => Some((premultiplied(&w, im), im.frame)),
            None => None,
        }
    };
    let Some((img, frame)) = img else { return };
    use super::anim::{preset, Tween};
    let out = (Tween::new(preset::WINDOWS_OUT.0, preset::WINDOWS_OUT.1), Tween::new(preset::FADE_OUT.0, preset::FADE_OUT.1));
    unsafe { (*GHOSTS.get()).push(Ghost { serial: w.serial, frame, img, out, shown: core::cell::Cell::new((1.0, 1.0)) }) };
    ANIMATING.store(true, core::sync::atomic::Ordering::Relaxed);
    super::render::invalidate();
}

/// Close window `i`, letting it fade.
fn close_at(d: &mut Desktop, i: usize) {
    if i < d.windows.len() {
        let w = d.windows.remove(i);
        depart(w);
    }
}

/// Start the newest window's arrival: it pops in from `POPIN` of its size
/// and fades up, on the clock (`anim`).
///
/// It was six frames of chrome only, a rectangle growing from the window's
/// centre with the window itself withheld until the last -- because a
/// window could not be drawn scaled until it had an image (phase 11). The
/// window is in the scene throughout now, takes input throughout, and is
/// drawn out of its image at this frame's scale and opacity.
pub fn open_flourish() {
    // With the compositor stood down there is nobody to animate it, and a
    // tween nothing composes is a window that sits invisible until something
    // else happens to repaint it. Arriving at once is what `render off` means.
    if !super::render::enabled() {
        return;
    }
    use super::anim::{preset, Tween};
    with(|d| {
        // The newest by serial, not the last in the stack: the stack's end is
        // whatever has focus, and opening a window hands focus back to the
        // terminal, so the last window is the shell and not the new one.
        let w = d.windows.iter_mut().max_by_key(|w| w.serial)?;
        if w.state == WinState::Minimised {
            return None;
        }
        w.arrive = Some((Tween::new(preset::WINDOWS_IN.0, preset::WINDOWS_IN.1), Tween::new(preset::FADE_IN.0, preset::FADE_IN.1)));
        Some(())
    });
    ANIMATING.store(true, core::sync::atomic::Ordering::Relaxed);
    super::render::invalidate();
}

/// The area the tiling layout fills: the screen above the taskbar. Its own
/// rectangle rather than `screen_rect`, whose margin is a floating window's
/// and not the layout's -- the layout's edge is `tile::GAPS_OUT`.
fn tile_area(fb: &Framebuffer) -> Rect {
    Rect::new(0, 0, fb.width(), fb.height().saturating_sub(TASK_H))
}

/// Bring the tiling tree in line with the windows, and every tiled window to
/// its slot.
///
/// **Reconciled, not hooked.** Windows are opened, closed, minimised and
/// restored from a dozen places in this file and from other tasks through
/// `post`; teaching each of them to tell the layout would be a dozen places
/// to forget one. So before each frame the tree is made to hold exactly the
/// tiled windows that are showing -- new ones split the window nearest the
/// top that already tiles, which is the one they were opened from -- and the
/// layout is applied. A window whose slot moved slides there.
fn tile_sync(d: &mut Desktop, area: Rect) {
    let want = |w: &Window| w.tiled && w.state != WinState::Minimised;
    for s in d.tiles.leaves() {
        if !d.windows.iter().any(|w| w.serial == s && want(w)) {
            d.tiles.remove(s);
        }
    }
    let mut missing: Vec<u64> = d.windows.iter().filter(|w| want(w) && !d.tiles.contains(w.serial)).map(|w| w.serial).collect();
    missing.sort_unstable();
    for s in missing {
        let beside = d.windows.iter().rev().map(|w| w.serial).find(|x| d.tiles.contains(*x));
        d.tiles.insert(s, beside);
    }
    let now = super::anim::now_us();
    let mut moved = false;
    for (s, r) in d.tiles.layout(area) {
        let Some(w) = d.windows.iter_mut().find(|w| w.serial == s) else { continue };
        if w.rect == r {
            continue;
        }
        // Slide from wherever it is drawn now, which mid-slide is part of the
        // way along the last one. A window that has never been drawn -- one
        // just opened, which arrives instead -- simply takes its place.
        if w.image.borrow().is_some() && w.arrive.is_none() && super::render::enabled() {
            let from = match w.slide {
                Some((f, t)) => lerp_rect(f, w.rect, t.at(now)),
                None => w.rect,
            };
            w.slide = Some((from, super::anim::Tween::new(super::anim::preset::WINDOWS_MOVE.0, super::anim::preset::WINDOWS_MOVE.1)));
            moved = true;
        }
        w.rect = r;
    }
    if moved {
        ANIMATING.store(true, core::sync::atomic::Ordering::Relaxed);
    }
}

/// A progress fixed to a 4096th, which is what every stamp of a moving
/// layer records.
///
/// **The stamp and the painter must read the same number.** They read the
/// unrounded one at first and `render verify` found the gap: a ring fading
/// up by less than a 4096th left the stamp where it was, so nothing was
/// repainted -- and still moved one glow pixel's rounding by one, which the
/// reference frame, painting at the new value, drew differently. Rounded
/// once, here, a change too small to damage anything is too small to draw.
fn q4096(x: f32) -> f32 {
    ((x.clamp(0.0, 1.0) * 4096.0 + 0.5) as u32) as f32 / 4096.0
}

/// `a` toward `b` by `t`, exactly `b` at 1.
fn lerp_rect(a: Rect, b: Rect, t: f32) -> Rect {
    if t >= 1.0 {
        return b;
    }
    let l = |x: u32, y: u32| (x as f32 + (y as f32 - x as f32) * t + 0.5) as u32;
    Rect::new(l(a.x, b.x), l(a.y, b.y), l(a.w, b.w).max(1), l(a.h, b.h).max(1))
}

/// `r` scaled by `k` about its centre.
fn scaled(r: Rect, k: f32) -> Rect {
    let w = ((r.w as f32 * k) + 0.5) as u32;
    let h = ((r.h as f32 * k) + 0.5) as u32;
    Rect::new(r.x + (r.w - w.min(r.w)) / 2, r.y + (r.h - h.min(r.h)) / 2, w.max(1), h.max(1))
}

/// Super-V: float a tiled window, or tile a floating one. Afloat it keeps a
/// size of its own -- at most two thirds of the area, centred -- and slides
/// there from its slot, as it slides back into one.
fn toggle_float() {
    let Some(fb) = super::primary() else { return };
    let area = tile_area(&fb);
    with(|d| {
        let Some(f) = d.focus() else { return };
        let w = &mut d.windows[f];
        if w.state == WinState::Maximised {
            w.state = WinState::Normal;
        }
        w.tiled = !w.tiled;
        if !w.tiled {
            let nw = w.rect.w.min(area.w * 2 / 3);
            let nh = w.rect.h.min(area.h * 2 / 3);
            let to = Rect::new(area.x + (area.w - nw) / 2, area.y + (area.h - nh) / 2, nw, nh);
            if w.image.borrow().is_some() {
                w.slide = Some((w.rect, super::anim::Tween::new(super::anim::preset::WINDOWS_MOVE.0, super::anim::preset::WINDOWS_MOVE.1)));
                ANIMATING.store(true, core::sync::atomic::Ordering::Relaxed);
            }
            w.rect = to;
        }
    });
}

/// Super and an arrow: focus the window that way. With Shift: swap places
/// with it, when both tile.
fn tile_move(dir: u8, swap: bool) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    with(|d| {
        let Some(f) = d.focus() else { return };
        let me = d.windows[f].serial;
        let rects: Vec<(u64, Rect)> = d
            .windows
            .iter()
            .filter(|w| w.state != WinState::Minimised)
            .map(|w| (w.serial, w.frame(screen)))
            .collect();
        let Some(to) = super::tile::neighbour(&rects, me, dir) else { return };
        if swap {
            d.tiles.swap(me, to);
        } else if let Some(i) = d.windows.iter().position(|w| w.serial == to) {
            d.raise(i);
        }
    });
}

/// Super-J: turn the split the focused window sits in the other way.
fn toggle_split() {
    let Some(fb) = super::primary() else { return };
    let area = tile_area(&fb);
    with(|d| {
        let Some(f) = d.focus() else { return };
        let s = d.windows[f].serial;
        let inner = Rect::new(
            area.x + super::tile::GAPS_OUT,
            area.y + super::tile::GAPS_OUT,
            area.w.saturating_sub(2 * super::tile::GAPS_OUT),
            area.h.saturating_sub(2 * super::tile::GAPS_OUT),
        );
        d.tiles.toggle(s, inner);
    });
}

/// Retire what has finished moving and keep frames coming while anything
/// has not. The compositor's, after each frame.
pub fn anim_step() {
    if !ANIMATING.load(core::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let now = super::anim::now_us();
    // A full-screen program owns the screen: everything lands at once rather
    // than holding windows half-drawn across a program that may run for an
    // hour.
    let all = super::exclusive();
    let mut moving = false;
    with(|d| {
        for w in d.windows.iter_mut() {
            if let Some((a, b)) = w.arrive {
                if all || (a.done(now) && b.done(now)) {
                    w.arrive = None;
                } else {
                    moving = true;
                }
            }
            if let Some((_, t)) = w.slide {
                if all || t.done(now) {
                    w.slide = None;
                } else {
                    moving = true;
                }
            }
        }
    });
    let g = unsafe { &mut *GHOSTS.get() };
    g.retain(|gh| !all && !(gh.out.0.done(now) && gh.out.1.done(now)));
    moving |= !g.is_empty();
    let ring = unsafe { &mut *FOCUS_RING.get() };
    if let Some(t) = ring.1 {
        if all || t.done(now) {
            ring.1 = None;
        } else {
            moving = true;
        }
    }
    ANIMATING.store(moving, core::sync::atomic::Ordering::Relaxed);
    // One more frame after the last tween ends, which is the one that draws
    // the window plainly or clears the last of a ghost.
    super::render::invalidate();
}

/// An arriving window's scale and opacity at `now`.
fn arrival_at(t: (super::anim::Tween, super::anim::Tween), now: u64) -> (f32, f32) {
    let p = super::anim::preset::POPIN;
    (p + (1.0 - p) * t.0.at(now), t.1.at(now))
}

/// A departing one's.
fn departure_at(t: (super::anim::Tween, super::anim::Tween), now: u64) -> (f32, f32) {
    let p = super::anim::preset::POPIN;
    (1.0 - (1.0 - p) * t.0.at(now), 1.0 - t.1.at(now))
}

/// A window, or a ghost, at scale `k` about its centre and at `opacity`,
/// with its shadow faded along with it.
fn paint_transformed(fb: &Framebuffer, img: &super::scale::Image, to: Rect, opacity: f32, shadow: Option<(super::shadow::Spec, i32, i32, u8)>) {
    let (x, y, w, h) = (to.x as i32, to.y as i32, to.w, to.h);
    let o = (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    if let Some((spec, dx, dy, alpha)) = shadow {
        super::shadow::cast(fb, x, y, w, h, spec, dx, dy, (alpha as u32 * o / 255) as u8);
    }
    super::scale::draw(fb, img, x, y, w, h, o as u8);
}

fn paint_ghosts(fb: &Framebuffer) {
    for gh in unsafe { (*GHOSTS.get()).iter() } {
        let r = gh.frame;
        if !fb.touches(r.x, r.y, r.w, r.h) {
            continue;
        }
        let (k, o) = gh.shown.get();
        paint_transformed(fb, &gh.img, scaled(r, k), o, None);
    }
}

/// How close to a screen edge a move has to end to count as a snap.
const SNAP: i32 = 8;

/// Finish a window move: snap to an edge, or come back off one.
///
/// The gesture an operator arriving from Windows tries first, and the reason
/// half-screen tiling is worth having at all -- two windows side by side
/// without measuring either. Left and right edges take half the screen, the
/// top maximises.
///
/// Deliberately decided at release rather than while the pointer is moving.
/// A window that resizes itself under a drag has to re-anchor the grab
/// mid-gesture, and getting that wrong makes the window jump away from the
/// pointer. Releasing is the moment the operator has committed.
fn snap_release(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let (l, t) = (screen.x as i32, screen.y as i32);
    let r = l + screen.w as i32;
    let half_w = screen.w / 2;

    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        let w = &mut d.windows[f];
        let here = w.rect;

        let b = screen.y as i32 + screen.h as i32;
        let near_l = x <= l + SNAP;
        let near_r = x >= r - SNAP;
        let near_t = y <= t + SNAP;
        let near_b = y >= b - SNAP;

        // A corner beats an edge, and is tested first for that reason. A drag
        // ending within the snap distance of two edges is at a corner and
        // meant one; checking the top or the side first would give it half the
        // screen and swallow the gesture.
        let corner = match (near_l, near_r, near_t, near_b) {
            (true, _, true, _) => Some(Tile::TopLeft),
            (_, true, true, _) => Some(Tile::TopRight),
            (true, _, _, true) => Some(Tile::BottomLeft),
            (_, true, _, true) => Some(Tile::BottomRight),
            _ => None,
        };
        if let Some(tl) = corner {
            w.snap_back = w.snap_back.or(Some(here));
            w.state = WinState::Normal;
            w.rect = tile_rect(tl, screen);
            return true;
        }

        if near_t {
            // The top edge maximises, using the state the caption button and
            // the system menu already use, so there is one maximised window
            // and not two ideas of one.
            if w.state != WinState::Maximised {
                w.snap_back = w.snap_back.or(Some(here));
                w.state = WinState::Maximised;
                return true;
            }
            return false;
        }

        let side = if near_l {
            Some(screen.x)
        } else if near_r {
            Some(screen.x + screen.w - half_w)
        } else {
            None
        };

        if let Some(nx) = side {
            w.snap_back = w.snap_back.or(Some(here));
            w.state = WinState::Normal;
            w.rect = Rect::new(nx, screen.y, half_w, screen.h);
            return true;
        }

        // Off the edge, and the window is carrying a size it did not choose:
        // give it back, around the place it was dropped rather than the place
        // it came from.
        if let Some(back) = w.snap_back.take() {
            if w.state != WinState::Maximised {
                let cx = here.x + here.w / 2;
                let nx = cx
                    .saturating_sub(back.w / 2)
                    .min(screen.x + screen.w.saturating_sub(back.w))
                    .max(screen.x);
                w.rect = Rect::new(nx, here.y, back.w, back.h);
                return true;
            }
        }
        false
    })
    .unwrap_or(false);
    if changed {
        super::render::invalidate();
    }
}

/// Pointer motion while the title bar or the size grip is held.
fn drag_to(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    let changed = with(|d| {
        let Some(f) = d.focus() else { return false };
        let r = d.windows[f].rect;
        let new = match d.mode {
            Mode::Drag { dx, dy, .. } => {
                let nx = (x - dx as i32).max(screen.x as i32) as u32;
                let ny = (y - dy as i32).max(screen.y as i32) as u32;
                Rect::new(
                    nx.min(screen.x + screen.w.saturating_sub(r.w)),
                    ny.min(screen.y + screen.h.saturating_sub(r.h)),
                    r.w,
                    r.h,
                )
            }
            Mode::DragSize { from, edges } => {
                // Each held edge moves to the pointer; the opposite side stays
                // where it was. Left and top change the origin as well as the
                // extent, which is why the arithmetic is not symmetric.
                let (min_w, min_h) = min_size_of(&d.windows[f]);
                let (mut nx, mut ny) = (from.x, from.y);
                let (mut nw, mut nh) = (from.w, from.h);
                let right = from.x + from.w;
                let bottom = from.y + from.h;

                if edges & edge::LEFT != 0 {
                    let want = x.max(screen.x as i32) as u32;
                    nx = want.min(right.saturating_sub(min_w));
                    nw = right - nx;
                }
                if edges & edge::RIGHT != 0 {
                    let lim = screen.x + screen.w;
                    let want = (x.max(0) as u32).min(lim);
                    nw = want.saturating_sub(from.x).max(min_w);
                    nw = nw.min(lim.saturating_sub(from.x));
                }
                if edges & edge::TOP != 0 {
                    let want = y.max(screen.y as i32) as u32;
                    ny = want.min(bottom.saturating_sub(min_h));
                    nh = bottom - ny;
                }
                if edges & edge::BOTTOM != 0 {
                    let lim = screen.y + screen.h;
                    let want = (y.max(0) as u32).min(lim);
                    nh = want.saturating_sub(from.y).max(min_h);
                    nh = nh.min(lim.saturating_sub(from.y));
                }
                Rect::new(nx, ny, nw.max(min_w), nh.max(min_h))
            }
            _ => return false,
        };
        let moved = new.x != r.x || new.y != r.y || new.w != r.w || new.h != r.h;
        if moved {
            d.windows[f].state = WinState::Normal;
            d.windows[f].rect = new;
        }
        moved
    })
    .unwrap_or(false);
    if changed {
        super::render::invalidate();
    }
}

/// What the pointer is over right now.
fn hover_of(fb: &Framebuffer, x: i32, y: i32) -> Hover {
    let screen = screen_rect(fb);
    if contains(taskbar_rect(fb), x, y) {
        if contains(start_rect(fb), x, y) {
            return Hover::Start;
        }
        let hit = with(|d| {
            task_layout(fb, d)
                .into_iter()
                .enumerate()
                .find(|(_, (r, ..))| contains(*r, x, y))
                .map(|(i, _)| i)
        })
        .flatten();
        return match hit {
            Some(i) => Hover::Task(i),
            None => Hover::None,
        };
    }
    if let Some(i) = window_at(x, y) {
        return with(|d| {
            let w = &d.windows[i];
            let (title, menubar, client) = chrome(w.frame(screen), !w.menus.is_empty());
            if contains(title, x, y) {
                if let Some(which) = theme::caption_buttons(title)
                    .iter()
                    .position(|b| contains(*b, x, y))
                {
                    return Hover::Caption { win: i, which };
                }
            }
            if let Some(bar) = menubar {
                if let Some(mi) = menu_label_at(&w.menus, bar, x, y) {
                    return Hover::MenuLabel { win: i, menu: mi };
                }
            }
            if let Content::Panel(p) = &w.content {
                if let Some(idx) = p.hover_at(client, x, y) {
                    return Hover::Widget { win: i, idx };
                }
            }
            Hover::None
        })
        .unwrap_or(Hover::None);
    }
    match icon_at(fb, x, y) {
        Some(k) => Hover::Icon(k),
        None => Hover::None,
    }
}

/// Repaint only when what the pointer indicates has changed.
fn update_hover(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);

    // An open dropdown tracks the pointer with its selection, exactly as the
    // arrows move it.
    let tracked = with(|d| {
        let Some(p) = dropdown_rows(&fb, d, screen) else {
            return None;
        };
        let Some(item) = p.item_at(x, y) else {
            // Off the menu: nothing tracks, and whatever was hot before the
            // menu opened must not stay lit underneath it.
            let stale = d.hover != Hover::None;
            d.hover = Hover::None;
            return Some(stale);
        };
        let moved = match d.mode {
            Mode::Menu { menu, item: cur } if cur != item => {
                d.mode = Mode::Menu { menu, item };
                true
            }
            Mode::Sys { item: cur } if cur != item => {
                d.mode = Mode::Sys { item };
                true
            }
            Mode::Start { item: cur } if cur != item => {
                d.mode = Mode::Start { item };
                true
            }
            _ => false,
        };
        Some(moved)
    })
    .flatten();
    if let Some(moved) = tracked {
        if moved {
            super::render::invalidate();
        }
        return;
    }

    let h = hover_of(&fb, x, y);
    let changed = with(|d| {
        if d.hover != h {
            d.hover = h;
            true
        } else {
            false
        }
    })
    .unwrap_or(false);
    if changed {
        super::render::invalidate();
    }
}

/// The second button opens the menu for whatever is under it.
///
/// Reuses the two menus that already exist rather than inventing a third: over
/// a window it is that window's system menu, which is the same one Alt-Space
/// opens, and over the wall it is the application list the taskbar shows. Both
/// are already driven by the arrow keys and already tested, so the button is a
/// second way in rather than a second implementation.
fn right_click_at(x: i32, y: i32) {
    let Some(fb) = super::primary() else { return };
    let screen = screen_rect(&fb);
    match window_at(x, y) {
        Some(i) => {
            // A program gets first refusal on the second button -- that is
            // how Minesweeper flags. Only if it declines does the button
            // mean the system menu, the meaning it has everywhere else.
            let consumed = with(|d| {
                d.raise(i);
                let f = d.windows.len() - 1;
                let frame = d.windows[f].frame(screen);
                let has_menus = !d.windows[f].menus.is_empty();
                let (_, _, client) = chrome(frame, has_menus);
                match &mut d.windows[f].content {
                    Content::App(a) if contains(client, x, y) => {
                        app_input(a.as_ref());
                        a.right_press(Rect::new(0, 0, client.w, client.h), x - client.x as i32, y - client.y as i32)
                    }
                    _ => false,
                }
            })
            .unwrap_or(false);
            if !consumed {
                with(|d| d.mode = Mode::Sys { item: 0 });
            }
        }
        None => {
            with(|d| d.mode = Mode::Taskbar { item: 0 });
        }
    };
    super::render::invalidate();
}

fn wheel_at(x: i32, y: i32, notches: i32) {
    let Some(i) = window_at(x, y) else { return };
    let hit = with(|d| match &mut d.windows[i].content {
        Content::Browser(b) => {
            b.scroll_by(notches * 3);
            true
        }
        // Over a panel the wheel walks the first list, which is what a wheel
        // over a launcher or a checklist means.
        Content::Panel(p) => p.wheel(notches),
        Content::App(a) => {
            app_input(a.as_ref());
            a.wheel(notches)
        }
        // Up is back, which is the direction every terminal has agreed on.
        // Three rows a notch, matching the log windows.
        Content::Terminal(ch) => super::console::scroll_view(*ch, (notches * 3) as isize),
        _ => false,
    })
    .unwrap_or(false);
    if hit {
        super::render::invalidate();
    }
}

/// The wallpaper: `gfx::wall`, an Aperture sun over rim-lit hills.
///
/// It replaced a gradient horizon ruled with a graticule, a dial, three
/// Bresenham ellipses and seven bubbles around the mark -- the busy wall the
/// 1.4.1 plan names among the reasons the desktop looked 1998.
pub(crate) fn paint_wallpaper(fb: &Framebuffer) {
    wallpaper(fb);
}

fn wallpaper(fb: &Framebuffer) {
    super::wall::paint(fb);
}

/// Buttons on the bar, one per window, in stacking order: the window's
/// pictogram and whether it is the focused one.
///
/// No launcher buttons: launching lives in the Start menu and on the wall,
/// so the bar never runs out of room for the windows it exists to hold.
fn task_slots(d: &Desktop) -> Vec<(usize, bool)> {
    let focus = d.focus();
    d.windows
        .iter()
        .enumerate()
        .map(|(i, w)| (w.icon, Some(i) == focus))
        .collect()
}

/// Every taskbar button's rectangle, in slot order.
///
/// The single source for where the buttons are: the paint pass draws these
/// rectangles and the pointer hit-tests them, so a button cannot highlight in
/// one place and press in another.
fn task_layout(fb: &Framebuffer, d: &Desktop) -> Vec<(Rect, usize, bool)> {
    let bar = taskbar_rect(fb);
    let slots = task_slots(d);
    let btn_h = bar.h - 8;
    let y = bar.y + 4;
    let start = start_rect(fb);
    let mut x = start.x + start.w + 10;
    let mut out = Vec::new();

    for (icon, pressed) in slots {
        // A pictogram and its air: every button the same width, which is what
        // lets a bar of nine windows still read as a row rather than a ransom
        // note. The title lives on the window; the bar says what *kind*.
        let w = btn_h + 10;
        // Stop before whatever is on the right rather than drawing under it.
        // The charge well sits left of the clock when there is one, so the
        // buttons have to break against it and not against the clock.
        let right = battery_rect(fb).map(|r| r.x).unwrap_or_else(|| clock_rect(fb).x);
        if x + w > right {
            break;
        }
        out.push((Rect::new(x, y, w, btn_h), icon, pressed));
        x += w + TASK_GAP;
    }
    out
}

/// The bar along the bottom: apps, then open windows, then the clock.
///
/// A minimised window is a task button that is not pressed, which is why the
/// icons that used to sit on the wall are gone. An icon on the wallpaper is
/// findable only if no window covers it, and something you stow is exactly
/// what you then cannot find.
fn taskbar(fb: &Framebuffer, d: &Desktop, sel: Option<usize>) {
    let bar = taskbar_rect(fb);
    // Glass, so the sky shows through it. The wallpaper is painted before this
    // and the back-to-front repaint means what is underneath is already
    // composed -- which is the whole reason a translucent bar costs one blended
    // span per row and no second pass over anything.
    fb.glass(bar.x, bar.y, bar.w, bar.h, theme::GLASS, &theme::GLASS_STOPS);
    // The specular sheen over the top, and then one hard line where the pane
    // meets the sky. The opacity table is proportional and cannot say "one
    // pixel" at any height, which is the division of labour between a ramp and
    // the line drawn on top of it.
    fb.glass(bar.x, bar.y, bar.w, bar.h / 2, theme::GLOSS, &theme::GLOSS_STOPS);
    fb.rect(bar.x, bar.y, bar.w, 1, theme::TASK_EDGE);

    // The Start orb: Vista's, with the Aperture mark lit in it where Vista
    // had its flag. It replaced a green Luna pill with the name on it and a
    // ring round a dot for a mark, which is what the mark had to be when
    // nothing here could draw a curve. Lit while its menu is open, which is
    // the one place the bar states a mode rather than a focus.
    let s = start_rect(fb);
    let start_open = matches!(d.mode, Mode::Start { .. });
    let lit = start_open || d.hover == Hover::Start;
    let n = s.h;
    super::icons::lay(
        fb,
        if lit { super::icons::ORB_HOT } else { super::icons::ORB },
        s.x + (s.w - n) / 2,
        s.y,
        n,
    );

    for (i, (r, icon, pressed)) in task_layout(fb, d).into_iter().enumerate() {
        // Keyboard selection and pointer hover draw the same way: both are "the
        // next click or Enter lands here", and two different highlights would
        // claim two different things.
        let hot = sel == Some(i) || d.hover == Hover::Task(i);
        // `control` rather than `button`: a task button wants the hover wash,
        // and `button` reads its one boolean as keyboard focus. The two mean
        // different things and only here do they need telling apart.
        let stops: &[(u8, Color)] = if pressed {
            &theme::BTN_DOWN
        } else if hot {
            &theme::BTN_HOT
        } else {
            &theme::BTN
        };
        theme::control(fb, r, stops, if hot { theme::BTN_EDGE_HOT } else { theme::BTN_EDGE });
        let s = r.h.saturating_sub(6);
        // Nudged a pixel when pressed, the same lie about depth the label
        // used to tell.
        let off = u32::from(pressed);
        // The button face is the ramp chosen just above, not `FACE` -- which
        // is no row of any of them.
        pictogram(fb, icon, r.x + (r.w.saturating_sub(s)) / 2 + off, r.y + 3 + off, s);
    }

    // What the button is for, which the bar itself cannot say.
    //
    // A task button is a pictogram and nothing else, and that is the right
    // trade -- every button the same width is what lets nine of them read as a
    // row rather than a ransom note, and a bar of truncated titles says
    // nothing three times over. But it leaves two windows of the same kind
    // indistinguishable, and there is no third state between "no label" and
    // "a label on every button": there is hover, which labels the one being
    // asked about.
    //
    // Painted at the end of the whole frame and not here, which is the part
    // that had to be found by looking: the bar is drawn *before* the windows,
    // and nothing about that mattered while everything the bar drew stayed
    // inside the bar. A tip stands above it, so drawn from here it went under
    // the terminal -- visible only in the strip of wallpaper to its left.
    taskbar_tail(fb, d);
}

/// The hovered task button's window title, standing above the bar.

/// Where the taskbar's hover tip goes, if one is showing. One formula for the
/// painter and the scene.
fn tip_rect(fb: &Framebuffer, d: &Desktop) -> Option<Rect> {
    let Hover::Task(i) = d.hover else { return None };
    let (r, _, _) = task_layout(fb, d).into_iter().nth(i)?;
    let title = d.windows.get(i).map(|w| w.title.as_str())?;
    let tw = theme::text_w_of(title);
    let box_w = tw + 12;
    let box_h = theme::text_h() + 8;
    // Clamped to the screen rather than centred blindly: the last button on a
    // full bar sits against the clock, and a tip centred on it would hang off
    // the edge.
    let want = r.x + r.w / 2 - (box_w / 2).min(r.x + r.w / 2);
    let x = want.min(fb.width().saturating_sub(box_w + 2)).max(2);
    let y = r.y.saturating_sub(box_h + 4);
    Some(Rect::new(x, y, box_w, box_h))
}

fn taskbar_tip(fb: &Framebuffer, d: &Desktop) {
    let Hover::Task(i) = d.hover else { return };
    let Some(tr) = tip_rect(fb, d) else { return };
    let Some(title) = d.windows.get(i).map(|w| w.title.as_str()) else { return };
    fb.shade_rect(tr.x + 2, tr.y + 2, tr.w, tr.h, 96);
    fb.vgrad(tr.x, tr.y, tr.w, tr.h, &theme::TITLE_OFF);
    theme::outline(fb, tr, theme::EDGE);
    theme::text_over(fb, tr.x + 6, tr.y + 4, title, theme::TEXT);
}

/// The Move/Size hint, when the keyboard is moving or sizing a window.
fn hint_rect(fb: &Framebuffer, d: &Desktop) -> Option<Rect> {
    d.focus()?;
    let msg = match d.mode {
        Mode::Move { .. } => MOVE_HINT,
        Mode::Size { .. } => SIZE_HINT,
        _ => return None,
    };
    let w = theme::text_w_of(msg);
    Some(Rect::new((fb.width().saturating_sub(w)) / 2, MARGIN, w, MENU_H))
}

const MOVE_HINT: &str = " Move: arrows, Enter to place, Esc to cancel ";
const SIZE_HINT: &str = " Size: arrows, Enter to keep, Esc to cancel ";

/// The rest of the bar: the recesses, and what did not fit.
fn taskbar_tail(fb: &Framebuffer, d: &Desktop) {
    // Windows the bar had no room for. `task_layout` breaks when it runs out,
    // which is right -- shrinking every button to fit an arbitrary count ends
    // with a row of slivers -- but it did so in silence, so a window could be
    // open, unreachable from the bar, and nothing said why.
    let shown = task_layout(fb, d).len();
    let total = task_slots(d).len();
    if total > shown {
        let label = alloc::format!("+{}", total - shown);
        let w = theme::text_w_of(&label) + 8;
        let right = battery_rect(fb).map(|r| r.x).unwrap_or_else(|| clock_rect(fb).x);
        let bar = taskbar_rect(fb);
        if right > w + 6 {
            let r = Rect::new(right - w - 6, bar.y + 4, w, bar.h - 8);
            theme::control(fb, r, &theme::BTN, theme::BTN_EDGE);
            let ty = r.y + (r.h.saturating_sub(theme::text_h())) / 2;
            theme::text_over(fb, r.x + 4, ty, &label, theme::TEXT_DIM);
        }
    }

    // The tray recesses. Not `theme::well`: a well outlines a hole in a face,
    // and on a coloured bar the thing that reads as a recess is being darker
    // than what surrounds it.
    let recess = |r: Rect| {
        fb.rect(r.x, r.y, r.w, r.h, theme::TRAY);
        theme::outline(fb, r, theme::TRAY_EDGE);
    };
    recess(clock_rect(fb));
    if let Some(b) = battery_rect(fb) {
        recess(b);
    }
}

/// Where the charge readout goes, when there is one.
///
/// Its own well rather than sharing the clock's. The clock's is exactly
/// thirteen columns because every column it takes is one the task buttons do
/// not get, and a string one character too long is dropped in silence rather
/// than truncated -- which is how it once came to be blank. Widening it to fit
/// a percentage would repeat that, so this is six columns of its own, and it
/// only exists on a machine that has a battery to put in it.
pub fn battery_rect(fb: &Framebuffer) -> Option<Rect> {
    let c = crate::dev::battery::status()?;
    if !c.present {
        return None;
    }
    let bar = taskbar_rect(fb);
    let w = theme::text_w(6);
    let clock = clock_rect(fb);
    Some(Rect::new(clock.x.saturating_sub(w + 4), bar.y + 4, w, bar.h - 8))
}

/// What the charge well says: a percentage, and a mark for the source.
///
/// Six columns, so "+100%" fits with a space either side. The plus is mains
/// and the minus is battery, which is one character to say the thing an
/// operator most wants at a glance.
pub fn battery_text() -> Option<alloc::string::String> {
    let c = crate::dev::battery::status()?;
    if !c.present {
        return None;
    }
    let mark = match c.on_ac {
        Some(true) => '+',
        Some(false) => '-',
        None => ' ',
    };
    Some(alloc::format!("{}{}% ", mark, c.percent))
}

/// Where the uptime readout goes. Right-hand end of the bar.
pub fn clock_rect(fb: &Framebuffer) -> Rect {
    let bar = taskbar_rect(fb);
    // Wide enough for " up 1234.5s " and no wider. Reserving more silently
    // steals room from the task buttons -- which is how the terminal's own
    // button came to be missing from the bar, and how the clock came to be
    // blank: the string was one character longer than its well and the draw
    // refused rather than overflowing.
    let w = theme::text_w(13);
    Rect::new(
        bar.x + bar.w.saturating_sub(w + 6),
        bar.y + 4,
        w,
        bar.h - 8,
    )
}

/// Compose a whole frame. **The compositor calls this. Nothing else should.**
///
/// It had fifty direct callers inside this file until they were converted, and
/// the flag beside it was written up as having replaced them when it had only
/// been added alongside. So every event handler here composed a frame on
/// whichever task had reached it -- the shell for a keystroke, the compositor
/// for a click -- and the property the whole design rests on held only on the
/// paths nobody had looked at. Measured before the conversion: `composers`
/// read 1 after a `render probe` and 3 after opening one window, because
/// `focus_terminal` ends in a repaint and the shell calls it after every
/// command.
///
/// Anything here that changes the screen calls `render::invalidate` instead,
/// which is free, idempotent, and safe from any task. The cost of being wrong
/// about it is one frame of staleness rather than two writers in one back
/// buffer.
///
/// **The one named exception is `video bench`**, which times this directly and
/// has to: a measurement of what a frame costs cannot be taken by asking for
/// one and waiting to see when it lands. It moves `composers`, and a reading
/// taken after one is expected to say so.
pub fn draw() {
    let Some(real) = super::primary() else {
        return;
    };
    // One painter at a time, for the whole frame.
    //
    // The claim covers `cursor_hide` at the top through `cursor_show` at the
    // bottom, because the window between them is the race: a cursor move that
    // lands after the hide and before the present saves pixels from a frame
    // that is about to be replaced. Holding it only around each cursor call
    // would leave that gap wide open.
    //
    // It also makes `draw` single-writer, which it has never been -- the agent
    // and author tasks both call it while the shell may be mid-frame, sharing
    // the back buffer, the shadow and the cursor statics with no lock at all.
    // Returning is the right answer to losing: the holder is painting the same
    // desktop this call would have painted.
    let Some(_claim) = Claim::take() else {
        return;
    };
    let started = crate::time::rdtsc();
    let screen = screen_rect(&real);
    // A no-op with a compositor, where the pointer is a plane and never in
    // the back buffer; without one it lifts the arrow drawn on the aperture.
    cursor_hide(&real);
    // Compose into the back buffer when there is one; the screen then gets
    // only the rows that changed. Without one (no heap yet, or its
    // allocation failed) this is the direct draw it always was.
    let fb = super::compose::target().unwrap_or(real);
    let (reported, full) = super::render::take_damage();
    let whole = Rect::new(0, 0, real.width(), real.height());
    let mut damage = super::region::Region::new();
    super::frame_clock(true);
    with(|d| {
        tile_sync(d, tile_area(&real));
        track_focus(d);
    });
    with_ref(|d| {
        let next = layers(d, &real, screen);
        let prev = unsafe { &mut *PREV.get() };
        // Everything when asked for, when there is no previous frame to
        // compare with, and when there is no back buffer -- drawing straight
        // to the aperture has no shadow to make a partial repaint safe.
        match prev.as_ref() {
            Some(p) if !full && super::compose::active() => {
                damage = reported;
                super::scene::diff(p, &next, &mut damage);
            }
            _ => damage.add(whole),
        }
        damage.clip_to(whole);
        // `render flash`: what this frame repaints because something changed,
        // before the last frame's outlines are added to it -- those are
        // repainted to erase them and are not themselves outlined, or every
        // frame would outline the frame before it forever.
        let shown = damage;
        let flashing = FLASH.load(core::sync::atomic::Ordering::Relaxed);
        let lit = unsafe { &mut *FLASHED.get() };
        for r in lit.iter() {
            damage.add(r);
        }
        damage.clip_to(whole);
        *lit = super::region::Region::new();
        place_consoles(d, screen);
        for r in damage.iter() {
            paint_scene(&fb.clipped(r.x, r.y, r.w, r.h), d, screen);
        }
        if verifying() {
            verify_frame(&fb, d, screen);
        }
        if flashing {
            for r in shown.iter() {
                outline(&fb, r, FLASH_COLOUR);
                lit.add(r);
            }
        }
        *prev = Some(next);
    });
    super::frame_clock(false);
    super::render::drew();
    let presenting = crate::time::rdtsc();
    super::render::damaged(damage.len() as u64, damage.area());
    super::compose::present_region(&damage);
    // Put the pointer back where it was. Without this every keystroke that
    // repaints would blink the arrow out until the mouse next moved.
    if let Some((px, py)) = unsafe { *POS.get() } {
        cursor_show(&real, px, py);
    }
    super::render::frame_timed(started, presenting, crate::time::rdtsc());
}

/// `render flash`: outline every rectangle a frame repaints, so damage can
/// be watched rather than inferred. The outlines are drawn into the back
/// buffer after the verifier has looked, and erased by the next frame, which
/// the tray guarantees within a tenth of a second -- long enough to see.
static FLASH: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// What the last frame outlined, to repaint away. Under the paint claim.
static FLASHED: Racy<super::region::Region> = Racy::new(super::region::Region::new());
const FLASH_COLOUR: Color = Color::new(0xFF, 0x20, 0xC0);

pub fn set_flash(on: bool) {
    FLASH.store(on, core::sync::atomic::Ordering::Relaxed);
    super::render::invalidate();
}

pub fn flashing() -> bool {
    FLASH.load(core::sync::atomic::Ordering::Relaxed)
}

/// A one-pixel frame just inside `r`.
fn outline(fb: &Framebuffer, r: Rect, c: Color) {
    if r.w == 0 || r.h == 0 {
        return;
    }
    fb.rect(r.x, r.y, r.w, 1, c);
    fb.rect(r.x, r.y + r.h - 1, r.w, 1, c);
    fb.rect(r.x, r.y, 1, r.h, c);
    fb.rect(r.x + r.w - 1, r.y, 1, r.h, c);
}

/// The last frame's layers, for `scene::diff`. Only the compositor's frame
/// reads or writes it, under the paint claim.
static PREV: Racy<Option<Vec<super::scene::Layer>>> = Racy::new(None);

/// Counts composed frames, for the parts of the scene whose content changes
/// without anything the scene can fingerprint -- an application drawing from
/// its own state, a status strip reading the machine. Stamping them with this
/// repaints them whenever a frame is composed at all, which is what happened to
/// the whole screen before damage existed; phase 8 gives applications stamps
/// of their own.
static FRAME_SEQ: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Where a terminal's well, grid and strip are inside its client area. One
/// formula for the painter, the scene and `place_consoles`, which must agree
/// about where the text goes or it is painted in one place and hit-tested,
/// reflowed or damaged in another.
fn terminal_geometry(client: Rect, ch: usize) -> (Rect, Rect, Option<Rect>) {
    let well = client.shrink(2);
    let inner = well.shrink(3);
    let strip = if ch == super::console::USER { terminal_strip(well) } else { None };
    let grid = match strip {
        Some(s) => Rect::new(inner.x, inner.y, inner.w, inner.h - s.h),
        None => inner,
    };
    (well, grid, strip)
}

/// Put each console where its window is, and hide the ones nobody can see,
/// before any painting.
///
/// This was part of the paint pass: every console was hidden at the top of a
/// frame and made visible again when its window was drawn. Once a frame only
/// repaints what changed, a frame that does not touch the terminal would leave
/// it hidden -- and a hidden console's output reaches nothing -- so where each
/// console is and whether it shows is decided here, from the window list,
/// whatever the frame repaints.
///
/// A minimised window's console was never reflowed *and* never told it was
/// hidden: it kept its old origin and visibility, and the next line written to
/// it went straight to the framebuffer at that stale position -- over the
/// desktop, outside any window. The Executive does this on its own schedule,
/// so minimising it and waiting produced `[mind t16] disabled` painted on the
/// wallpaper. Hiding every console first and showing only placed ones is what
/// prevents that, and nothing is lost: an invisible console still updates its
/// shadow grid, and its window repaints the whole log when it comes back.
fn place_consoles(d: &Desktop, screen: Rect) {
    // Decided first and applied once per console. It hid every console and
    // then showed the placed ones, every frame -- and a composed console only
    // reports damage while visible, so a line printed by another task in the
    // gap between the two reported nothing, and its cell stayed stale until
    // something else repainted that part of the terminal.
    let mut placed: [Option<Rect>; super::console::NCONSOLE] = [None; super::console::NCONSOLE];
    for win in d.windows.iter() {
        if win.state == WinState::Minimised {
            continue;
        }
        let Content::Terminal(ch) = win.content else { continue };
        let client = theme::chrome(win.frame(screen), !win.menus.is_empty()).client;
        if client.is_empty() || ch >= placed.len() {
            continue;
        }
        placed[ch] = Some(terminal_geometry(client, ch).1);
    }
    for (ch, grid) in placed.iter().enumerate() {
        super::console::with_ch(ch, |c| match grid {
            Some(g) => {
                c.set_visible(true);
                c.reflow(g.x, g.y, g.w, g.h);
            }
            None => c.set_visible(false),
        });
    }
}

/// What this frame contains, as the scene sees it. Pure over the desktop: it
/// paints nothing, and two calls over an unchanged desktop answer the same
/// list -- apart from the parts stamped with `FRAME_SEQ`, on purpose.
///
/// **Each stamp has to cover everything its painter reads.** That is the whole
/// contract, and the way it fails is a pixel that does not update when it
/// should. `render verify on` composes a whole reference frame every frame and
/// compares, which is how a missing input is found rather than guessed at.
/// What a window's painter reads, as the scene stamps it: the whole layer's
/// stamp, and the parts that repaint on their own. One function for `layers`,
/// which turns it into damage, and for the window's image, which is re-drawn
/// exactly where that damage says -- two copies of this would be two answers
/// to "has this window changed", and a stale image is what the second would
/// cost.
fn window_key(
    d: &Desktop,
    i: usize,
    win: &Window,
    active: bool,
    frame: Rect,
    inputs: &(([bool; 96], u32, u32), (String, String, bool)),
    seq: u64,
    wake: &mut u32,
) -> (u64, Vec<(Rect, u64)>) {
    use super::scene::Stamp;
    use core::hash::Hash;
    let hot = match d.hover {
        Hover::Caption { win, which } if win == i => Some(which),
        _ => None,
    };
    let menu_hover = match d.hover {
        Hover::MenuLabel { win, menu } if win == i => Some(menu),
        _ => None,
    };
    let open_menu = match d.mode {
        Mode::Menu { menu, .. } if active => Some(menu),
        _ => None,
    };
    let mut st = Stamp::new();
    (frame, win.title.as_str(), active, win.state == WinState::Maximised).hash(&mut st);
    (hot, menu_hover, open_menu, win.round).hash(&mut st);
    // The caption glass's tint is chrome the painter reads.
    theme::caption_tint().hash(&mut st);
    for m in &win.menus {
        m.label.hash(&mut st);
    }
    let chrome = theme::chrome(frame, !win.menus.is_empty());
    let client = chrome.client;
    let mut parts = Vec::new();
    if !client.is_empty() {
        match &win.content {
            Content::Terminal(ch) => {
                let (well, grid, strip) = terminal_geometry(client, *ch);
                // The well's border and the grid's own background are
                // fixed by the geometry; the console repaints its text
                // itself as it prints.
                core::hash::Hasher::write_u64(&mut st, Stamp::of((well, grid)));
                // The gutter marks where prompts start and the strip reads
                // the machine's state, each stamped with exactly what its
                // painter reads. The gutter belongs to the main terminal
                // alone, as its painter does.
                if *ch == super::console::USER {
                    parts.push((Rect::new(well.x + 1, grid.y, 2, grid.h), Stamp::of(inputs.0)));
                }
                if let Some(strip) = strip {
                    parts.push((strip, Stamp::of(&inputs.1)));
                }
            }
            Content::Panel(p) => {
                let hov = match d.hover {
                    Hover::Widget { win, idx } if win == i => Some(idx),
                    _ => None,
                };
                parts.push((client, Stamp::of((p.stamp(), active, hov))));
            }
            Content::Browser(b) => parts.push((client, Stamp::of((b.stamp(), active)))),
            Content::App(a) => {
                if let Some(ms) = a.wake_ms() {
                    *wake = (*wake).min(ms.max(1));
                }
                parts.push((client, app_key(a.as_ref(), active, client, seq)))
            }
        }
    }
    (core::hash::Hasher::finish(&st), parts)
}

fn layers(d: &Desktop, fb: &Framebuffer, screen: Rect) -> Vec<super::scene::Layer> {
    use super::scene::{Layer, LayerId, Stamp};
    use core::hash::Hash;
    let seq = FRAME_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let now = super::anim::now_us();
    let (w, h) = (fb.width(), fb.height());
    let mut out = Vec::new();
    let mut wake = u32::MAX;
    let inputs = (gutter_marks(), status_text());
    out.push(Layer { id: LayerId::Wallpaper, bounds: Rect::new(0, 0, w, h), stamp: Stamp::of((w, h)), parts: Vec::new() });

    let icons = icon_rects(fb);
    if let Some(b) = icons.iter().map(|(r, _)| *r).reduce(|a, b| {
        let x0 = a.x.min(b.x);
        let y0 = a.y.min(b.y);
        Rect::new(x0, y0, (a.x + a.w).max(b.x + b.w) - x0, (a.y + a.h).max(b.y + b.h) - y0)
    }) {
        // Hover moves the highlight between icons, and only the icons read it
        // among the things under the windows.
        //
        // Padded, because the painter draws outside the cells: a hovered
        // label's highlight is six pixels wider than the label and a long
        // label is as wide as its cell, so the highlight spilled a pixel past
        // each side and a repaint clipped to the cells left those columns
        // stale. `render verify` found it on the first hover over Terminal.
        let pad = 8;
        let b = Rect::new(b.x.saturating_sub(pad), b.y.saturating_sub(pad), b.w + 2 * pad, b.h + 2 * pad);
        out.push(Layer { id: LayerId::Icons, bounds: b, stamp: Stamp::of(d.hover), parts: Vec::new() });
    }

    let focus = d.focus();
    {
        let mut st = Stamp::new();
        for win in &d.windows {
            win.title.hash(&mut st);
            win.icon.hash(&mut st);
            (win.state == WinState::Minimised).hash(&mut st);
            (win.state == WinState::Maximised).hash(&mut st);
        }
        focus.hash(&mut st);
        d.mode.hash(&mut st);
        d.hover.hash(&mut st);
        // The readouts are parts of the bar: a new tenth of a second repaints
        // the clock's well and nothing else.
        let (clock, battery) = tray_strings();
        let mut parts = alloc::vec![(clock_rect(fb), Stamp::of(clock.as_str()))];
        if let Some(b) = battery_rect(fb) {
            parts.push((b, Stamp::of(battery.as_deref())));
        }
        out.push(Layer {
            id: LayerId::Taskbar,
            bounds: taskbar_rect(fb),
            stamp: core::hash::Hasher::finish(&st),
            parts,
        });
    }

    for (i, win) in d.windows.iter().enumerate() {
        if win.state == WinState::Minimised {
            continue;
        }
        let active = Some(i) == focus;
        let frame = win.frame(screen);
        let bounds = shadowed(win, frame, active, fb);
        let (mut stamp, parts) = window_key(d, i, win, active, frame, &inputs, seq, &mut wake);
        // An arriving or sliding window is a different picture every frame.
        // Its whole layer moves with the clock, fixed here for the frame, and
        // a slide's layer reaches back to where it set out from.
        let mut bounds = bounds;
        win.shown.set(
            win.arrive
                .map(|t| {
                    let (k, o) = arrival_at(t, now);
                    (scaled(frame, k), q4096(o))
                })
                .or_else(|| win.slide.map(|(from, t)| (lerp_rect(from, frame, t.at(now)), 1.0))),
        );
        if let Some((r, o)) = win.shown.get() {
            stamp = Stamp::of((stamp, r, (o * 4096.0) as u32));
        }
        if active {
            let ring = unsafe { &mut *FOCUS_RING.get() };
            ring.2 = q4096(ring.1.map_or(1.0, |t| t.at(now)));
            stamp = Stamp::of((stamp, (ring.2 * 4096.0) as u32));
        }
        if let Some((from, _)) = win.slide {
            let b = shadowed(win, from, active, fb);
            let x0 = bounds.x.min(b.x);
            let y0 = bounds.y.min(b.y);
            bounds = Rect::new(x0, y0, (bounds.x + bounds.w).max(b.x + b.w) - x0, (bounds.y + bounds.h).max(b.y + b.h) - y0);
        }
        out.push(Layer {
            id: LayerId::Window(win.serial),
            bounds,
            stamp,
            parts,
        });
    }

    // Everything drawn over the windows: the taskbar tip, menus, the move and
    // size hint, the Start menu. One layer, because they come and go together
    // with the mode and the hover, and its bounds are their union.
    // Windows on their way out, over the windows and under the popups.
    for gh in unsafe { (*GHOSTS.get()).iter() } {
        let (k, o) = departure_at(gh.out, now);
        let (k, o) = (q4096(k), q4096(o));
        gh.shown.set((k, o));
        out.push(Layer {
            id: LayerId::Ghost(gh.serial),
            bounds: gh.frame,
            stamp: Stamp::of(((k * 4096.0) as u32, (o * 4096.0) as u32)),
            parts: Vec::new(),
        });
    }

    let mut over: Option<Rect> = None;
    let mut grow = |r: Rect| {
        over = Some(match over {
            None => r,
            Some(a) => {
                let x0 = a.x.min(r.x);
                let y0 = a.y.min(r.y);
                Rect::new(x0, y0, (a.x + a.w).max(r.x + r.w) - x0, (a.y + a.h).max(r.y + r.h) - y0)
            }
        });
    };
    if let Some(t) = tip_rect(fb, d) {
        // The tip's shadow is offset two pixels.
        grow(Rect::new(t.x, t.y, t.w + 2, t.h + 2));
    }
    if let Some(p) = dropdown_rows(fb, d, screen) {
        grow(p.panel);
    }
    if let Some(r) = hint_rect(fb, d) {
        grow(r);
    }
    if let Some(b) = over {
        let mut st = Stamp::new();
        (d.mode, d.hover, d.query.as_str(), focus).hash(&mut st);
        if let Hover::Task(t) = d.hover {
            d.windows.get(t).map(|w| w.title.as_str()).hash(&mut st);
        }
        out.push(Layer { id: LayerId::Overlay, bounds: b, stamp: core::hash::Hasher::finish(&st), parts: Vec::new() });
    }
    unsafe { *FRAME_INPUTS.get() = Some(inputs) };
    APP_WAKE.store(if wake == u32::MAX { 0 } else { wake }, core::sync::atomic::Ordering::Relaxed);
    out
}

/// The wallpaper, painted once per screen size into memory of its own.
///
/// `wallpaper` draws a gradient sky, a lattice, a dial, three ellipses of
/// Bresenham chords and seven bubbles -- several thousand primitives -- and it
/// was doing so sixty times a second whenever anything moved. With damage it
/// would do so once per damaged rectangle, mostly to throw the result away.
/// Painted once and kept, a repaint of any part of it is a row copy.
///
/// 8 MB at 1920x1080. When that allocation fails the procedural painter runs
/// as it always did, which is slower and identical.
static WALL: Racy<Option<(Vec<u32>, u32, u32)>> = Racy::new(None);

/// How hard the glass blurs the wallpaper, in pixels at the screen's own
/// resolution. Vista blurred a downsampled copy at a sigma of three, which is
/// about twelve here; a little more reads as frosted rather than smeared.
const GLASS_SIGMA: f32 = 14.0;

/// Keep a wallpaper somebody else already painted -- the boot screen's glide
/// ends on it -- rather than painting it again, and take the glass blur from
/// it as `wallpaper_cached` would. Before the desktop exists, so nothing is
/// composing from the cache while it is set.
pub(crate) fn adopt_wallpaper(px: Vec<u32>, w: u32, h: u32) {
    super::glass::set_backdrop(&px, w, h, GLASS_SIGMA);
    unsafe { *WALL.get() = Some((px, w, h)) };
}

fn wallpaper_cached(fb: &Framebuffer) {
    let (w, h) = (fb.width(), fb.height());
    let cache = unsafe { &mut *WALL.get() };
    if cache.as_ref().map(|c| (c.1, c.2)) != Some((w, h)) {
        *cache = None;
        let n = w as usize * h as usize;
        let mut px = Vec::new();
        if px.try_reserve_exact(n).is_ok() {
            px.resize(n, 0u32);
            let surf = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, fb.format()) };
            wallpaper(&surf);
            // The blurred copy every pane of glass samples, made here because
            // this is the one moment the wallpaper changes.
            super::glass::set_backdrop(&px, w, h, GLASS_SIGMA);
            *cache = Some((px, w, h));
        }
    }
    match cache.as_ref() {
        Some((px, w, _)) => {
            let (x0, y0, x1, y1) = fb.scissor();
            for y in y0..y1 {
                let row = &px[(y * w) as usize..((y + 1) * w) as usize];
                fb.blit_span(x0, y, &row[x0 as usize..x1 as usize]);
            }
        }
        None => wallpaper(fb),
    }
}

/// Compose a whole reference frame every frame and compare it with what the
/// damage-limited one produced. A debugging aid, off by default and costly
/// when on; `render verify on|off`.
static VERIFY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static VERIFY_BAD: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
/// Frames that differed only inside a terminal's grid. Counted apart because
/// until phase 7 the console paints its own cells, caret included, straight
/// into the back buffer between frames, and a full repaint reproduces the text
/// but not every trace of the caret -- a known difference with a known owner,
/// which must not drown a scene bug in the same count.
static VERIFY_CONSOLE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static VERIFY_RUNS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static VERIFY_FIRST: Racy<Option<Rect>> = Racy::new(None);
/// The first differing pixel of the first bad frame: where, what the frame
/// had and what the reference had. A bounding box says where to look; two
/// values say what is wrong -- a missing blend, a stale frame, a dim applied
/// twice -- which a box never can.
static VERIFY_PIXEL: Racy<Option<(u32, u32, u32, u32)>> = Racy::new(None);

pub fn verify_pixel() -> Option<(u32, u32, u32, u32)> {
    unsafe { *VERIFY_PIXEL.get() }
}

/// The verifier's reference frame. Written only under the paint claim.
static VERIFY_BUF: Racy<Vec<u32>> = Racy::new(Vec::new());

pub fn set_verify(on: bool) {
    if on {
        // Sized here, on the shell's task and under the paint claim, so no
        // frame ever allocates it and none can be reading it while it moves.
        if let Some(fb) = super::primary() {
            let n = fb.width() as usize * fb.height() as usize;
            let deadline = crate::time::rdtsc() + crate::time::tsc_mhz().max(1) as u64 * 1_000_000;
            let mut held = Claim::take();
            while held.is_none() && crate::time::rdtsc() < deadline {
                crate::task::yield_now();
                held = Claim::take();
            }
            if held.is_some() {
                let buf = unsafe { &mut *VERIFY_BUF.get() };
                if buf.len() != n {
                    buf.clear();
                    if buf.try_reserve_exact(n).is_ok() {
                        buf.resize(n, 0);
                    }
                }
            }
        }
    }
    VERIFY.store(on, core::sync::atomic::Ordering::Relaxed);
    if on {
        VERIFY_BAD.store(0, core::sync::atomic::Ordering::Relaxed);
        VERIFY_CONSOLE.store(0, core::sync::atomic::Ordering::Relaxed);
        VERIFY_RUNS.store(0, core::sync::atomic::Ordering::Relaxed);
        unsafe { *VERIFY_FIRST.get() = None };
        unsafe { *VERIFY_PIXEL.get() = None };
    }
}

fn verifying() -> bool {
    VERIFY.load(core::sync::atomic::Ordering::Relaxed)
}

/// Frames checked, frames that differed outside any terminal grid, frames
/// that differed only inside one, and the first scene difference seen.
pub fn verify_report() -> (u64, u64, u64, Option<Rect>) {
    (
        VERIFY_RUNS.load(core::sync::atomic::Ordering::Relaxed),
        VERIFY_BAD.load(core::sync::atomic::Ordering::Relaxed),
        VERIFY_CONSOLE.load(core::sync::atomic::Ordering::Relaxed),
        unsafe { *VERIFY_FIRST.get() },
    )
}

/// Paint the whole frame into scratch memory and compare it with `back`.
///
/// The tray readouts are left out of the comparison: their text is read at
/// paint time, so a tenth of a second turning over between the frame and the
/// reference is a difference in the clock rather than in the renderer.
///
/// The reference lives in `VERIFY_BUF`, allocated once when verification is
/// switched on. It was a fresh eight megabytes per frame, and `diag mt` --
/// which checks the heap has not grown across a run -- caught a frame that
/// happened to be in flight at its closing sample and reported it as a leak
/// of exactly one 1920x1080 frame.
fn verify_frame(back: &Framebuffer, d: &Desktop, screen: Rect) {
    let (w, h) = (back.width(), back.height());
    let n = w as usize * h as usize;
    let px = unsafe { &mut *VERIFY_BUF.get() };
    if px.len() != n {
        return;
    }
    let scratch = unsafe { Framebuffer::over_ram(px.as_mut_ptr() as u64, w, h, w, back.format()) };
    REFERENCE.store(true, core::sync::atomic::Ordering::Relaxed);
    paint_scene(&scratch, d, screen);
    REFERENCE.store(false, core::sync::atomic::Ordering::Relaxed);
    VERIFY_RUNS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let skip = [clock_rect(back), battery_rect(back).unwrap_or(Rect::new(0, 0, 0, 0))];
    let grids: Vec<Rect> = d.windows.iter().enumerate()
        .filter(|(_, w)| w.state != WinState::Minimised && w.arrive.is_none() && w.slide.is_none())
        .filter_map(|(_, w)| match w.content {
            Content::Terminal(ch) => {
                let client = theme::chrome(w.frame(screen), !w.menus.is_empty()).client;
                Some(terminal_geometry(client, ch).1)
            }
            _ => None,
        })
        .collect();
    let inside = |rs: &[Rect], x: u32, y: u32| rs.iter().any(|r| x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h);
    let base = back.addr() as *const u32;
    let mut bad: Option<Rect> = None;
    let mut console_only = false;
    for y in 0..h {
        for x in 0..w {
            if inside(&skip, x, y) {
                continue;
            }
            let a = unsafe { *base.add((y * back.stride() + x) as usize) };
            if a != px[(y * w + x) as usize] && inside(&grids, x, y) {
                console_only = true;
                continue;
            }
            if a != px[(y * w + x) as usize] {
                let pix = unsafe { &mut *VERIFY_PIXEL.get() };
                if pix.is_none() {
                    *pix = Some((x, y, a, px[(y * w + x) as usize]));
                }
                bad = Some(match bad {
                    None => Rect::new(x, y, 1, 1),
                    Some(r) => {
                        let x0 = r.x.min(x);
                        let y0 = r.y.min(y);
                        Rect::new(x0, y0, (r.x + r.w).max(x + 1) - x0, (r.y + r.h).max(y + 1) - y0)
                    }
                });
            }
        }
    }
    if bad.is_none() && console_only {
        VERIFY_CONSOLE.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
    if let Some(r) = bad {
        VERIFY_BAD.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let first = unsafe { &mut *VERIFY_FIRST.get() };
        // Recorded, not printed: a line from the compositor lands in the
        // middle of whatever the shell is printing, and the first one landed
        // on the prompt, which hid it from `drive.py` and looked exactly like
        // a hung shell. `render verify` reports it.
        if first.is_none() {
            *first = Some(r);
        }
    }
}

/// Paint the desktop onto `target`: everything, in order, back to front.
///
/// This is what `draw` used to do inline, unchanged except that it is now run
/// once per damaged rectangle with `target` clipped to it -- so every painter
/// below draws exactly what it always drew and loses whatever falls outside
/// the rectangle. Windows wholly outside it are skipped. The order is the
/// order, so a translucent shadow or glass band always blends over pixels
/// repainted underneath it in the same pass.
/// Set while `render verify` composes its reference frame, which paints
/// every window straight onto the frame rather than out of its image.
static REFERENCE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Window images drawn whole and in part since boot, for `render`.
static IMAGE_WHOLE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static IMAGE_PART: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// `(whole, part, bytes held)`: how often window images were drawn, and what
/// they hold now.
pub fn image_stats() -> (u64, u64, usize) {
    let held = with_ref(|d| {
        d.windows.iter().map(|w| w.image.borrow().as_ref().map_or(0, |im| im.px.len() * 4)).sum()
    })
    .unwrap_or(0);
    (
        IMAGE_WHOLE.load(core::sync::atomic::Ordering::Relaxed),
        IMAGE_PART.load(core::sync::atomic::Ordering::Relaxed),
        held,
    )
}

/// How much an unfocused window is dimmed, of 256: Hyprland's
/// `dim_inactive`, kept light -- the cue is which window is nearest the eye,
/// and the others still have to be read.
const DIM: u32 = 228;

/// The focused window's ring: how wide it is, and how far its glow reaches.
const RING_W: f32 = 2.0;
const GLOW: f32 = 10.0;

/// Which window the ring is on and how far it has faded up (Hyprland's
/// `border`, 539 ms easeOutQuint), and this frame's opacity of it, fixed by
/// `layers` as a window's position is.
static FOCUS_RING: Racy<(u64, Option<super::anim::Tween>, f32)> = Racy::new((0, None, 1.0));

/// Notice focus moving to a different window and fade the ring up on it.
fn track_focus(d: &Desktop) {
    let now = d.focus().map_or(0, |f| d.windows[f].serial);
    let ring = unsafe { &mut *FOCUS_RING.get() };
    if ring.0 != now {
        ring.0 = now;
        if super::render::enabled() {
            use super::anim::{Curve, Tween};
            ring.1 = Some(Tween::new(539, Curve::EaseOutQuint));
            ANIMATING.store(true, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Signed distance from a point to a window's outline -- square at the
/// foot, corners of radius `r` at the head, as `Shape::round_top` cuts it.
fn sd_frame(px: f32, py: f32, f: Rect, r: f32) -> f32 {
    let (x0, y0, x1, y1) = (f.x as f32, f.y as f32, (f.x + f.w) as f32, (f.y + f.h) as f32);
    if r > 0.0 && py < y0 + r {
        let corner = |cx: f32| {
            let (dx, dy) = (px - cx, py - (y0 + r));
            super::shape::libm_sqrt(dx * dx + dy * dy) - r
        };
        if px < x0 + r {
            return corner(x0 + r);
        }
        if px > x1 - r {
            return corner(x1 - r);
        }
    }
    (x0 - px).max(px - x1).max(y0 - py).max(py - y1)
}

/// The focused window's border: a thin ring just inside its outline, in a
/// 45-degree ramp from Aperture orange to gold -- Hyprland's gradient
/// border, in the accent the Look keeps for focus -- with a soft warm glow
/// outside it. `glow` is false where there is nothing outside to glow on.
fn focus_ring(fb: &Framebuffer, frame: Rect, round: u32, alpha: f32, glow: bool) {
    if alpha <= 0.0 || frame.w < 8 || frame.h < 8 {
        return;
    }
    let r = round as f32;
    let reach = if glow { GLOW as u32 } else { 0 };
    let x0 = frame.x.saturating_sub(reach);
    let x1 = frame.x + frame.w + reach;
    let (sx0, sy0, sx1, sy1) = fb.scissor();
    let y0 = frame.y.saturating_sub(reach).max(sy0);
    let y1 = (frame.y + frame.h + reach).min(sy1);
    let band = round + RING_W as u32 + 2;
    let (a, b) = (theme::APERTURE, Color::new(0xFF, 0xC8, 0x5A));
    let span = (frame.w + frame.h).max(1) as f32;
    let mut row: Vec<u32> = Vec::new();
    let cov = |t: f32| (0.5 - t).clamp(0.0, 1.0);
    let mut paint = |y: u32, xa: u32, xb: u32, row: &mut Vec<u32>| {
        let (xa, xb) = (xa.max(sx0), xb.min(sx1));
        if xa >= xb {
            return;
        }
        row.clear();
        let py = y as f32 + 0.5;
        for x in xa..xb {
            let d = sd_frame(x as f32 + 0.5, py, frame, r);
            let ring = cov(d) - cov(d + RING_W);
            let g = if glow && d > 0.0 { let k = (1.0 - d / GLOW).max(0.0); 0.30 * k * k } else { 0.0 };
            let k = ring.max(g) * alpha;
            if k <= 0.0 {
                row.push(0);
                continue;
            }
            let t = (((x as f32 - frame.x as f32) + (py - frame.y as f32)) / span).clamp(0.0, 1.0);
            let mix = |p: u8, q: u8| (p as f32 + (q as f32 - p as f32) * t + 0.5) as u8;
            let raw = fb.encode(Color::new(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b)));
            row.push(super::blend::premul(raw, (k * 255.0 + 0.5) as u8));
        }
        fb.over_row(xa, y, row);
    };
    for y in y0..y1 {
        let near_edge = y < frame.y + band || y + RING_W as u32 + 2 >= frame.y + frame.h;
        if near_edge {
            paint(y, x0, x1, &mut row);
        } else {
            // Between the head and the foot only the two sides carry a ring.
            paint(y, x0, frame.x + RING_W as u32 + 2, &mut row);
            paint(y, (frame.x + frame.w).saturating_sub(RING_W as u32 + 2), x1, &mut row);
        }
    }
}

/// An unfocused window, dimmed where it shows.
fn dim_window(fb: &Framebuffer, win: &Window, frame: Rect) {
    with_window_shape(fb, win, frame, || {
        let (x0, y0, x1, y1) = fb.scissor();
        let xa = x0.max(frame.x);
        let xb = x1.min(frame.x + frame.w);
        if xa >= xb {
            return;
        }
        for y in y0.max(frame.y)..y1.min(frame.y + frame.h) {
            fb.shade_span(xa, y, xb - xa, DIM);
        }
    });
}

/// A terminal's grid, over whatever the window under it was painted from.
fn console_over(fb: &Framebuffer, win: &Window) {
    if let Content::Terminal(ch) = &win.content {
        super::console::blit_ch(*ch, fb);
    }
}

/// Bring a window's image up to date and copy the part of it the frame is
/// repainting. False when there is no image to be had, and the caller paints
/// directly.
fn from_image(fb: &Framebuffer, d: &Desktop, i: usize, win: &Window, frame: Rect, active: bool, hot: Option<usize>) -> bool {
    if !ensure_image(fb.format(), d, i, win, frame, active, hot) {
        return false;
    }
    let slot = win.image.borrow();
    let Some(im) = slot.as_ref() else { return false };
    with_window_shape(fb, win, frame, || {
        let (x0, y0, x1, y1) = fb.scissor();
        let xa = x0.max(frame.x);
        let xb = x1.min(frame.x + frame.w);
        if xa >= xb {
            return;
        }
        for y in y0.max(frame.y)..y1.min(frame.y + frame.h) {
            let r = (y - frame.y) as usize * frame.w as usize;
            fb.blit_span(xa, y, &im.px[r + (xa - frame.x) as usize..r + (xb - frame.x) as usize]);
        }
    });
    true
}

/// Bring a window's image up to date. False when there is none to be had.
fn ensure_image(format: super::Format, d: &Desktop, i: usize, win: &Window, frame: Rect, active: bool, hot: Option<usize>) -> bool {
    use core::sync::atomic::Ordering::Relaxed;
    if frame.is_empty() {
        return false;
    }
    let inputs = unsafe { (*FRAME_INPUTS.get()).clone() }.unwrap_or_else(|| (gutter_marks(), status_text()));
    let mut wake = u32::MAX;
    let (key, parts) = window_key(d, i, win, active, frame, &inputs, FRAME_SEQ.load(Relaxed), &mut wake);
    let mut slot = win.image.borrow_mut();
    let n = frame.w as usize * frame.h as usize;
    let whole = match slot.as_ref() {
        Some(im) => im.frame != frame || im.key != key || im.parts.len() != parts.len(),
        None => true,
    };
    let t0 = crate::time::rdtsc();
    if whole {
        let mut px = match slot.take() {
            Some(im) if im.px.len() == n => im.px,
            _ => {
                let mut v = Vec::new();
                if v.try_reserve_exact(n).is_err() {
                    return false;
                }
                v.resize(n, 0);
                v
            }
        };
        let img = unsafe { Framebuffer::displaced(px.as_mut_ptr(), frame.x, frame.y, frame.w, frame.h, format) };
        paint_window_body(&img, d, i, win, frame, active, hot);
        IMAGE_WHOLE.fetch_add(1, Relaxed);
        *slot = Some(WinImage { px, frame, key, parts });
    } else if let Some(im) = slot.as_mut() {
        // Only the parts whose stamps moved, which is exactly the damage the
        // scene gave this window: the rest of the image is still what it
        // would draw.
        for (k, part) in parts.iter().enumerate() {
            if im.parts[k] == *part {
                continue;
            }
            let (r, _) = *part;
            let img = unsafe { Framebuffer::displaced(im.px.as_mut_ptr(), frame.x, frame.y, frame.w, frame.h, format) };
            paint_window_body(&img.clipped(r.x, r.y, r.w, r.h), d, i, win, frame, active, hot);
            im.parts[k] = *part;
            IMAGE_PART.fetch_add(1, Relaxed);
        }
    }
    super::render::cost(super::render::Cost::WinImage, crate::time::rdtsc().saturating_sub(t0));
    slot.is_some()
}

/// Everything a window paints inside its own outline but the terminal
/// grid: the frame, the caption, the menu bar and the client. Into the
/// window's image normally, and straight onto the frame when there is no
/// image -- `render verify`'s reference frame, or no memory for one.
fn paint_window_body(fb: &Framebuffer, d: &Desktop, i: usize, win: &Window, frame: Rect, active: bool, hot: Option<usize>) {
    let fb = *fb;
    let c = theme::window(
        &fb,
        frame,
        &win.title,
        active,
        win.state == WinState::Maximised,
        !win.menus.is_empty(),
        hot,
    );
    let client = c.client;
    if client.is_empty() {
        // Nothing left to draw inside the frame; the chrome is
        // already painted.
        return;
    }

    if let Some(bar) = c.menubar {
        theme::panel(&fb, bar);
        let open = match d.mode {
            Mode::Menu { menu, .. } if active => Some(menu),
            _ => None,
        };
        let labels = theme::menu_labels(bar, win.menus.iter().map(|m| m.label.as_str()));
        for (mi, (m, lr)) in win.menus.iter().zip(labels).enumerate() {
            let hot = Some(mi) == open
                || d.hover == Hover::MenuLabel { win: i, menu: mi };
            let (fg, bg) = if hot {
                (theme::SELECT_TEXT, theme::SELECT)
            } else {
                (theme::TEXT, theme::FACE)
            };
            let ty = lr.y + (lr.h.saturating_sub(theme::text_h())) / 2;
            fb.rect(lr.x, lr.y + 1, lr.w, lr.h.saturating_sub(2), bg);
            theme::text(&fb, lr.x + 6, ty, &m.label, fg, bg);
        }
    }

    match &win.content {
        Content::Terminal(ch) => {
            let well = client.shrink(2);
            theme::well(&fb, well, theme::SCREEN);
            let inner = well.shrink(3);
            // A status strip along the foot of the *window*, with
            // the grid reflowed into what is left. It goes on the
            // main terminal only: the exec console is a transcript
            // of one command and has no state of its own to
            // report, and a second strip would be two rows of
            // chrome saying the same thing.
            let strip = if *ch == super::console::USER {
                terminal_strip(well)
            } else {
                None
            };
            let grid = match strip {
                Some(s) => Rect::new(inner.x, inner.y, inner.w, inner.h - s.h),
                None => inner,
            };
            // The grid itself is the console's own surface, laid over
            // this by `console_over` -- outside the window's image,
            // because text arrives far more often than chrome changes.
            let _ = grid;
            if *ch == super::console::USER {
                prompt_gutter(&fb, well, grid);
            }
            if let Some(strip) = strip {
                terminal_status(&fb, strip);
            }
        }
        Content::Panel(p) => {
            theme::panel(&fb, client);
            let hov = match d.hover {
                Hover::Widget { win, idx } if win == i => Some(idx),
                _ => None,
            };
            p.draw_in(&fb, client, active, hov);
        }
        Content::Browser(b) => b.draw_in(&fb, client, active),
        Content::App(a) => paint_app(win, a.as_ref(), &fb, client, active),
    }
}

fn paint_scene(target: &Framebuffer, d: &Desktop, screen: Rect) {
    let fb = *target;
    use super::render::{cost, Cost};
    let mut t = crate::time::rdtsc();
    let lap = |what: Cost, t: &mut u64| {
        let now = crate::time::rdtsc();
        cost(what, now.saturating_sub(*t));
        *t = now;
    };
    {
        wallpaper_cached(&fb);
        lap(Cost::Wallpaper, &mut t);
        draw_icons(&fb, d.hover);
        lap(Cost::Icons, &mut t);
        let focus = d.focus();
        let sel = match d.mode {
            Mode::Taskbar { item } => Some(item),
            _ => None,
        };
        taskbar(&fb, d, sel);
        paint_tray_text(&fb);
        lap(Cost::Taskbar, &mut t);

        for (i, win) in d.windows.iter().enumerate() {
            if win.state == WinState::Minimised {
                continue;
            }
            let active = Some(i) == focus;
            let frame = win.frame(screen);
            // Nothing of this window falls in the area being repainted -- its
            // shadow counted, which is the same rectangle its layer claims.
            let reach = match win.shown.get() {
                Some((r, _)) => {
                    let a = shadowed(win, frame, active, &fb);
                    let b = shadowed(win, r, active, &fb);
                    let x0 = a.x.min(b.x);
                    let y0 = a.y.min(b.y);
                    Rect::new(x0, y0, (a.x + a.w).max(b.x + b.w) - x0, (a.y + a.h).max(b.y + b.h) - y0)
                }
                None => shadowed(win, frame, active, &fb),
            };
            if !fb.touches(reach.x, reach.y, reach.w, reach.h) {
                continue;
            }
            let hot = match d.hover {
                Hover::Caption { win, which } if win == i => Some(which),
                _ => None,
            };
            // The shadow, before the window and after everything under it.
            //
            // Back to front is what makes this correct rather than careful:
            // whatever is already composed here -- wall, icons, an earlier
            // window -- is what gets darkened, and the only thing that can
            // cover it afterwards is a window in front, which is exactly what
            // should. No compositing, no second pass, no order to get right
            // beyond the one the loop already has.
            // On its way in: out of its image, scaled and faded, shadow and
            // all. The terminal's grid joins it when it lands -- the grid is
            // not in the image, and a scaled console is not worth a second
            // copy of it for four hundred milliseconds.
            if let Some((to, o)) = win.shown.get() {
                if ensure_image(fb.format(), d, i, win, frame, active, hot) {
                    if let Some(im) = win.image.borrow().as_ref() {
                        paint_transformed(&fb, &premultiplied(win, im), to, o, win_shadow(win, active));
                    }
                    continue;
                }
            }
            if let Some((spec, dx, dy, alpha)) = win_shadow(win, active) {
                super::shadow::cast(&fb, frame.x as i32, frame.y as i32, frame.w, frame.h, spec, dx, dy, alpha);
            }

            // Confine this window's paint to its own outline. A rectangular
            // window costs a load and a branch for the whole frame; a shaped
            // one simply does not write outside itself, and whatever the
            // back-to-front repaint already put there shows through.
            //
            // The window itself comes out of its image (`WinImage`) when it
            // has one, which turns a window that is merely exposed -- by
            // something moving over it, or a part of it changing -- into row
            // copies. The terminal's grid goes over either way.
            if REFERENCE.load(core::sync::atomic::Ordering::Relaxed) || !from_image(&fb, d, i, win, frame, active, hot) {
                with_window_shape(&fb, win, frame, || paint_window_body(&fb, d, i, win, frame, active, hot));
            }
            with_window_shape(&fb, win, frame, || console_over(&fb, win));
            if active {
                let maxed = win.state == WinState::Maximised;
                let alpha = unsafe { (*FOCUS_RING.get()).2 };
                focus_ring(&fb, frame, if maxed { 0 } else { win.round }, alpha, !maxed);
            } else {
                dim_window(&fb, win, frame);
            }
        }

        // Over the windows, because the bar is drawn under them and a tip
        // stands above the bar.
        lap(Cost::Windows, &mut t);
        // Windows on their way out, above the windows and under the popups.
        // (Where the arrival used to be painted -- which sat at the end of
        // this function until a search for that end matched `dropdown`'s
        // instead and put it inside that loop, once per menu row.)
        paint_ghosts(&fb);
        taskbar_tip(&fb, d);

        // Popups last, over everything, because that is what a popup is.
        if let Some(f) = focus {
            let frame = d.windows[f].frame(screen);
            match d.mode {
                Mode::Menu { menu, item } => {
                    if let Some(m) = d.windows[f].menus.get(menu) {
                        let inner = frame.shrink(theme::FRAME);
                        let bar = Rect::new(inner.x, inner.y + theme::TITLE_H + 2, inner.w, MENU_H);
                        // A label that cannot be placed skips its dropdown and
                        // nothing else. This was a `return` out of the whole
                        // painter, which ended the frame's overlays and its
                        // cost accounting along with it.
                        if let Some(lr) =
                            theme::menu_labels(bar, d.windows[f].menus.iter().map(|p| p.label.as_str()))
                                .nth(menu)
                        {
                            dropdown(
                                &fb,
                                lr.x,
                                inner.y + theme::TITLE_H + 2 + MENU_H,
                                m.items.iter().map(|i| i.label.as_str()),
                                item,
                            );
                        }
                    }
                }
                Mode::Sys { item } => {
                    let inner = frame.shrink(theme::FRAME);
                    dropdown(
                        &fb,
                        inner.x,
                        inner.y + theme::TITLE_H,
                        SYS_ITEMS.iter().copied(),
                        item,
                    );
                }
                Mode::Move { .. } | Mode::Size { .. } => {
                    // A hint, because a mode with no indication that it is on is
                    // a keyboard that has stopped working.
                    let msg = if matches!(d.mode, Mode::Move { .. }) { MOVE_HINT } else { SIZE_HINT };
                    if let Some(r) = hint_rect(&fb, d) {
                        theme::panel(&fb, r);
                        theme::text(&fb, r.x, r.y + 5, msg, theme::TEXT, theme::FACE);
                    }
                }
                // The taskbar draws its own selection, and it is not a popup
                // over the focused window -- it is a fixture with the keyboard.
                // A pointer drag needs no hint: the window moving under the
                // pointer is the indication. Start draws below, unconditional
                // on focus.
                Mode::Normal
                | Mode::Taskbar { .. }
                | Mode::Drag { .. }
                | Mode::DragSize { .. }
                | Mode::Start { .. } => {}
            }
        }
        if let Mode::Start { item } = d.mode {
            let p = start_menu_rect(&fb);
            // The panel is drawn to the full height including the query row,
            // then the items over it, then the query row last. It carries one
            // more row than it has items, which is why it builds its own
            // `Popup` rather than going through `dropdown`.
            theme::popup(&fb, &p);
            if p.wide_enough() {
                for (i, (label, _)) in START_ITEMS.iter().enumerate() {
                    theme::list_row(&fb, p.row(i), label, i == item, true);
                }
                let row = p.row(START_ITEMS.len());
                // A well, not a list row: it is a place to type, and it should
                // not look like something that runs when pressed.
                theme::well(&fb, row, theme::HILIGHT);
                let shown = if d.query.is_empty() {
                    alloc::string::String::from("Type to search")
                } else {
                    // The tail, so the end being typed stays visible rather
                    // than the beginning that has already been read.
                    let n = d.query.chars().count();
                    d.query.chars().skip(n.saturating_sub(QUERY_COLS - 1)).collect()
                };
                let fg = if d.query.is_empty() { theme::TEXT_DIM } else { theme::TEXT };
                theme::text(&fb, row.x + 4, row.y + 4, &shown, fg, theme::HILIGHT);
                // A caret, so a selected empty box does not read as inert.
                if is_query_row(item) {
                    let cx = row.x + 4 + theme::text_w_of(&shown);
                    fb.rect(cx, row.y + 4, 2, theme::text_h(), theme::TEXT);
                }
            }
        }
        lap(Cost::Overlay, &mut t);
    }
}

fn dropdown<'a>(
    fb: &Framebuffer,
    x: u32,
    y: u32,
    labels: impl Iterator<Item = &'a str> + Clone,
    sel: usize,
) {
    let items: Vec<&str> = labels.collect();
    if items.is_empty() {
        return;
    }
    let cols = items.iter().map(|s| s.chars().count()).max().unwrap_or(4);
    // The same construction `dropdown_rows` answers with, so the paint and the
    // hit-test cannot come to different conclusions about the same menu.
    let p = theme::Popup::sized(x, y, cols, items.len(), fb.width().saturating_sub(x));
    theme::popup(fb, &p);
    if !p.wide_enough() {
        return;
    }
    for (i, label) in items.iter().enumerate() {
        theme::list_row(fb, p.row(i), label, i == sel, true);
    }
}

pub fn focus_is_terminal() -> bool {
    with(|d| match d.focus() {
        Some(f) => matches!(d.windows[f].content, Content::Terminal(c) if c == super::console::USER),
        // No window at all: the shell has the keyboard. The terminal is an
        // application over a shell that never stops, so with every window
        // minimised, typing types at the prompt -- invisibly, but a command
        // and Enter still run, and the output is waiting when the terminal
        // comes back. `false` here made an all-minimised desktop swallow
        // the keyboard entirely, which read as a hang.
        None => true,
    })
    .unwrap_or(true)
}

/// Alt-Tab: swap the top two visible windows.
///
/// A swap, not a rotation. This used to send the top window to the back,
/// which walks all windows eventually but makes a second Alt-Tab land on a
/// *third* window -- pressed twice, it went somewhere new instead of back.
/// Windows has toggled the top pair on a single press since 3.1, for the
/// reason that became obvious here the hard way: a headless script (and a
/// person) needs "over and back" to be two presses, deterministically. It
/// cost a test run in which the second Alt-Tab focused the Program Manager
/// and the next command line ran one of its rows.
/// Alt-Tab, and Shift-Alt-Tab.
///
/// Plain Alt-Tab **swaps the top two and deliberately does not rotate**. That
/// is a decision this tree already made and paid for: rotating made a second
/// press land on a third window, and both scripts and habit need over-and-back
/// to be two presses. Nothing here changes it.
///
/// What `back` gets is the other half of the same idea rather than the other
/// direction of a rotation, because a swap has no other direction -- it is its
/// own inverse, and honouring `back` as "swap the top two, backwards" would be
/// the same operation under a second name. So it raises the *deepest* visible
/// window: the one furthest from the top, which in a stack where z-order is
/// recency is the one least recently used. Two keys, two reachable windows,
/// and plain Alt-Tab still returns to where it was.
pub fn cycle(back: bool) {
    with(|d| {
        let visible: Vec<usize> = d
            .windows
            .iter()
            .enumerate()
            .filter(|(_, w)| w.state != WinState::Minimised)
            .map(|(i, _)| i)
            .collect();
        if visible.len() < 2 {
            return;
        }
        // The two topmost visible windows; raising the lower one swaps them.
        // Backwards, the bottom of the stack instead, which for two windows is
        // the same window and for three or more is the one Alt-Tab alone can
        // never reach.
        let pick = if back { visible[0] } else { visible[visible.len() - 2] };
        d.raise(pick);
    });
    trace(if back { "alt-tab back" } else { "alt-tab" });
    super::render::invalidate();
}

pub fn trace(event: &str) {
    with(|d| {
        let focus = d.focus();
        crate::serial_println!("[desk] {}", event);
        for (i, w) in d.windows.iter().enumerate() {
            let state = match w.state {
                WinState::Normal => "normal",
                WinState::Maximised => "max",
                WinState::Minimised => "min",
            };
            crate::serial_println!(
                "[desk] {} {} {} {}x{}+{}+{}",
                if Some(i) == focus { '>' } else { ' ' },
                w.title,
                state,
                w.rect.w,
                w.rect.h,
                w.rect.x,
                w.rect.y
            );
        }
    });
}

pub enum Route {
    Handled,
    Shell(u8),
}

/// Act on a system-menu choice.
fn sys_action(d: &mut Desktop, f: usize, item: usize, screen: Rect) {
    match item {
        0 => d.mode = Mode::Move { from: d.windows[f].rect },
        1 => d.mode = Mode::Size { from: d.windows[f].rect },
        2 => {
            d.windows[f].state = match d.windows[f].state {
                WinState::Maximised => WinState::Normal,
                _ => WinState::Maximised,
            };
            let _ = screen;
        }
        3 => {
            if d.windows[f].closable {
                close_at(d, f);
            } else {
                // The terminal is the shell. Minimising is the honest thing to
                // offer instead of pretending a close happened.
                d.windows[f].state = WinState::Minimised;
            }
        }
        _ => {}
    }
}

/// Put the keyboard back where a person would expect it.
///
/// The desktop takes *every* key while it is in a menu mode, which is correct
/// when somebody opened the menu and catastrophic when nobody did. QEMU's
/// i8042 probe emits scancodes at boot -- the entropy ring already counts one
/// of them as a phantom touch -- and under the hypervisor accelerator that
/// probe was leaving the machine sitting in the Start menu before the shell
/// had printed its prompt. Every byte arriving over the serial line then fed
/// the menu instead of the command line, so a driven session looked like a
/// guest that had stopped reading its UART.
///
/// Called when the shell announces it is interactive. Nothing a controller
/// does while nobody is at the keyboard should decide where the next
/// keystroke goes.
pub fn dismiss_menus() {
    if !ready() {
        return;
    }
    let changed = with(|d| {
        let was = !matches!(d.mode, Mode::Normal);
        d.mode = Mode::Normal;
        was
    })
    .unwrap_or(false);
    if changed {
        super::render::invalidate();
    }
}

pub fn key(k: u8) -> Route {
    if !ready() {
        return Route::Shell(k);
    }
    let Some(fb) = super::primary() else {
        return Route::Shell(k);
    };
    let screen = screen_rect(&fb);

    // Window management first, in every mode, so there is no state the keyboard
    // can get stuck in.
    if k == kbd::KEY_ALTTAB || k == kbd::KEY_ALTTAB_BACK {
        with(|d| d.mode = Mode::Normal);
        cycle(k == kbd::KEY_ALTTAB_BACK);
        return Route::Handled;
    }
    // Before the mode dispatch, like Alt-Tab, so there is no state the
    // keyboard can get stuck in: pressing it again closes the menu.
    if k == kbd::KEY_STARTMENU {
        with(|d| {
            d.mode = match d.mode {
                Mode::Start { .. } => Mode::Normal,
                _ => Mode::Start { item: START_ITEMS.len() },
            };
            d.query.clear();
        });
        super::render::invalidate();
        return Route::Handled;
    }
    if k == kbd::KEY_TASKBAR {
        with(|d| {
            d.mode = match d.mode {
                // A second Ctrl-Esc puts the keyboard back, so the bar cannot
                // become somewhere you get stuck.
                Mode::Taskbar { .. } => Mode::Normal,
                _ => Mode::Taskbar { item: 0 },
            }
        });
        super::render::invalidate();
        return Route::Handled;
    }

    // Super chords are the desktop's in every mode, like Alt-Tab.
    if kbd::chord_of(k).is_some() {
        super_key(k);
        return Route::Handled;
    }

    let mode = with(|d| d.mode).unwrap_or(Mode::Normal);

    match mode {
        Mode::Normal => {
            if k == kbd::KEY_SYSMENU {
                with(|d| d.mode = Mode::Sys { item: 0 });
                super::render::invalidate();
                return Route::Handled;
            }
            if k == kbd::KEY_MENU {
                let opened = with(|d| match d.focus() {
                    Some(f) if !d.windows[f].menus.is_empty() => {
                        d.mode = Mode::Menu { menu: 0, item: 0 };
                        true
                    }
                    _ => false,
                })
                .unwrap_or(false);
                if opened {
                    super::render::invalidate();
                    return Route::Handled;
                }
                return Route::Handled;
            }
            if focus_is_terminal() {
                return Route::Shell(k);
            }
            // A panel has the keyboard.
            let step = with(|d| match d.focus() {
                Some(f) => match &mut d.windows[f].content {
                    Content::Panel(p) => p.key(k),
                    // The browser answers "did I use it", which is all the
                    // desktop needs: anything it declines falls through to the
                    // window manager, so Alt-Tab keeps working inside a page.
                    Content::Browser(b) => {
                        if b.key(k) { ui::Step::Redraw } else { ui::Step::Idle }
                    }
                    Content::App(a) => {
                        app_input(a.as_ref());
                        if a.key(k) { ui::Step::Redraw } else { ui::Step::Idle }
                    }
                    _ => ui::Step::Idle,
                },
                None => ui::Step::Idle,
            });
            match step {
                Some(ui::Step::Redraw) => super::render::invalidate(),
                Some(ui::Step::Close) => close_focused(),
                Some(ui::Step::Do(Action::Run(cmd))) => {
                    focus_terminal();
                    queue_command(&cmd);
                }
                // Navigation replaces the panel where it stands. The window
                // keeps its geometry, its place in the z-order and the
                // keyboard -- browsing is not opening something, and a browser
                // that jumped to the front of the stack on every keystroke
                // would be unusable.
                Some(ui::Step::Do(Action::Browse(route))) => {
                    // The desktop resolves a route without knowing what kind
                    // of app is on the other end of it.
                    if let Some((title, panel)) = ui::panel_for_route(&route) {
                        let kind = route.split(':').next().unwrap_or("");
                        with(|d| {
                            if let Some(f) = d.focus() {
                                d.windows[f].title = title;
                                d.windows[f].icon = panel_icon(kind);
                                d.windows[f].content = Content::Panel(panel);
                            }
                        });
                        super::render::invalidate();
                    }
                }
                _ => {}
            }
            Route::Handled
        }

        Mode::Sys { item } => {
            with(|d| {
                let Some(f) = d.focus() else { return };
                match k {
                    kbd::KEY_DOWN => {
                        d.mode = Mode::Sys { item: (item + 1) % SYS_ITEMS.len() }
                    }
                    kbd::KEY_UP => {
                        d.mode = Mode::Sys {
                            item: (item + SYS_ITEMS.len() - 1) % SYS_ITEMS.len(),
                        }
                    }
                    b'\n' | b'\r' => {
                        d.mode = Mode::Normal;
                        sys_action(d, f, item, screen);
                    }
                    27 => d.mode = Mode::Normal,
                    _ => {}
                }
            });
            super::render::invalidate();
            Route::Handled
        }

        Mode::Menu { menu, item } => {
            with(|d| {
                let Some(f) = d.focus() else { return };
                let n_menus = d.windows[f].menus.len();
                let n_items = d.windows[f].menus[menu].items.len();
                match k {
                    kbd::KEY_RIGHT => d.mode = Mode::Menu { menu: (menu + 1) % n_menus, item: 0 },
                    kbd::KEY_LEFT => {
                        d.mode = Mode::Menu { menu: (menu + n_menus - 1) % n_menus, item: 0 }
                    }
                    kbd::KEY_DOWN => {
                        d.mode = Mode::Menu { menu, item: (item + 1) % n_items.max(1) }
                    }
                    kbd::KEY_UP => {
                        d.mode = Mode::Menu {
                            menu,
                            item: (item + n_items.saturating_sub(1)) % n_items.max(1),
                        }
                    }
                    b'\n' | b'\r' => {
                        d.mode = Mode::Normal;
                        if let Some(mi) = d.windows[f].menus[menu].items.get(item) {
                            if let Action::Run(cmd) = &mi.action {
                                queue_command(cmd);
                            }
                        }
                    }
                    27 => d.mode = Mode::Normal,
                    _ => {}
                }
            });
            super::render::invalidate();
            Route::Handled
        }

        Mode::Taskbar { item } => {
            let n = with(|d| task_slots(d).len()).unwrap_or(0);
            if n == 0 {
                return Route::Handled;
            }
            match k {
                kbd::KEY_RIGHT => {
                    with(|d| d.mode = Mode::Taskbar { item: (item + 1) % n });
                    super::render::invalidate();
                }
                kbd::KEY_LEFT => {
                    with(|d| d.mode = Mode::Taskbar { item: (item + n - 1) % n });
                    super::render::invalidate();
                }
                b'\n' | b'\r' => {
                    with(|d| d.mode = Mode::Normal);
                    with(|d| {
                        if item < d.windows.len() {
                            // A task button restores a minimised window, which
                            // is the only way back for one.
                            if d.windows[item].state == WinState::Minimised {
                                d.windows[item].state = WinState::Normal;
                            }
                            d.raise(item);
                        }
                    });
                    super::render::invalidate();
                }
                27 => {
                    with(|d| d.mode = Mode::Normal);
                    super::render::invalidate();
                }
                _ => {}
            }
            Route::Handled
        }

        Mode::Start { item } => {
            let n = start_rows();
            match k {
                kbd::KEY_DOWN => {
                    with(|d| d.mode = Mode::Start { item: (item + 1) % n });
                    super::render::invalidate();
                }
                kbd::KEY_UP => {
                    with(|d| d.mode = Mode::Start { item: (item + n - 1) % n });
                    super::render::invalidate();
                }
                b'\n' | b'\r' => {
                    // The query row runs what was typed; every other row runs
                    // its own command. `open` is the same dispatcher the search
                    // panel uses, so a name means the same thing typed here as
                    // typed there -- an application opens, a command runs, and
                    // anything else offers to be written.
                    let typed = with(|d| {
                        let q = alloc::string::String::from(d.query.trim());
                        d.mode = Mode::Normal;
                        d.query.clear();
                        q
                    })
                    .unwrap_or_default();
                    if is_query_row(item) {
                        if !typed.is_empty() {
                            launch(&alloc::format!("open {}", typed));
                        }
                    } else {
                        launch(START_ITEMS[item].1);
                    }
                    super::render::invalidate();
                }
                27 => {
                    with(|d| {
                        d.mode = Mode::Normal;
                        d.query.clear();
                    });
                    super::render::invalidate();
                }
                8 => {
                    // Backspace edits the query wherever the selection is, and
                    // moves to it. Typing is the reason the row exists; making
                    // it reachable only by arrowing to it first would be a
                    // search box that has to be found before it can be used.
                    let moved = with(|d| {
                        d.query.pop();
                        d.mode = Mode::Start { item: START_ITEMS.len() };
                        !d.query.is_empty()
                    })
                    .unwrap_or(false);
                    let _ = moved;
                    super::render::invalidate();
                }
                // Printable ASCII. Typing anywhere in the menu goes to the
                // query and selects it, which is what Windows has done since
                // Vista and is the only behaviour that makes the box worth
                // having -- press Start, type, press Enter.
                c if (0x20..0x7F).contains(&c) => {
                    with(|d| {
                        if d.query.chars().count() < 64 {
                            d.query.push(c as char);
                        }
                        d.mode = Mode::Start { item: START_ITEMS.len() };
                    });
                    super::render::invalidate();
                }
                _ => {}
            }
            Route::Handled
        }

        // A pointer drag owns the screen until the button lifts; the keyboard
        // can only abandon it. Anything else mid-drag is swallowed, because a
        // window that responds to typing while it is being dragged is two
        // interfaces fighting over one object.
        Mode::Drag { from, .. } | Mode::DragSize { from, .. } => {
            if k == 27 {
                with(|d| {
                    if let Some(f) = d.focus() {
                        d.windows[f].rect = from;
                    }
                    d.mode = Mode::Normal;
                });
                super::render::invalidate();
            }
            Route::Handled
        }

        Mode::Move { from } | Mode::Size { from } => {
            let moving = matches!(mode, Mode::Move { .. });
            with(|d| {
                let Some(f) = d.focus() else { return };
                // Maximised windows have nowhere to go; the geometry being
                // edited is the restored one, so drop out of maximised first
                // rather than editing a rectangle nobody can see.
                d.windows[f].state = WinState::Normal;
                // Taken before the rectangle is borrowed mutably: it is a
                // fact about the window's content, not about its geometry.
                let floor_h = min_size_of(&d.windows[f]).1;
                let r = &mut d.windows[f].rect;
                match k {
                    kbd::KEY_LEFT if moving => r.x = r.x.saturating_sub(NUDGE),
                    kbd::KEY_RIGHT if moving => {
                        r.x = (r.x + NUDGE).min(screen.x + screen.w.saturating_sub(r.w))
                    }
                    kbd::KEY_UP if moving => r.y = r.y.saturating_sub(NUDGE),
                    kbd::KEY_DOWN if moving => {
                        r.y = (r.y + NUDGE).min(screen.y + screen.h.saturating_sub(r.h))
                    }
                    kbd::KEY_LEFT => r.w = r.w.saturating_sub(NUDGE).max(160),
                    kbd::KEY_RIGHT => {
                        r.w = (r.w + NUDGE).min(screen.x + screen.w - r.x)
                    }
                    // `TITLE_H * 3` was a stand-in for "a window has to keep
                    // some body", and it moved to 90 the moment the caption
                    // grew. `min_size_of` is what it meant.
                    kbd::KEY_UP => r.h = r.h.saturating_sub(NUDGE).max(floor_h),
                    kbd::KEY_DOWN => r.h = (r.h + NUDGE).min(screen.y + screen.h - r.y),
                    b'\n' | b'\r' => d.mode = Mode::Normal,
                    27 => {
                        d.windows[f].rect = from;
                        d.mode = Mode::Normal;
                    }
                    _ => {}
                }
            });
            super::render::invalidate();
            Route::Handled
        }
    }
}

/// Put the keyboard back on the terminal, raising it if need be.
pub fn focus_terminal() {
    with(|d| {
        if let Some(i) = d
            .windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
        {
            // Deliberately does *not* un-minimise. The terminal is an
            // application: it keeps running while closed, and a shell that
            // reopens its own window every time it prints is not something the
            // user can close. Restoring it is the taskbar's job, which is the
            // one place the user actually asked for it back.
            if d.windows[i].state != WinState::Minimised {
                d.raise(i);
            }
        }
    });
    super::render::invalidate();
}

/// Repaint if any window overlaps the terminal.
///
/// The console owns its rectangle and paints into it whenever the system
/// prints, which is correct until a window is in front of it -- then the
/// overlapping strip becomes console output on top of a window. Rather than
/// teach the console about occlusion (it would have to be consulted per
/// character, and it is the one path that has to stay fast) the desktop
/// repairs the damage between commands, which is the only moment the shell is
/// not printing.
///
/// Cheap because it does nothing in the common case: with no overlap there is
/// nothing to repair.
pub fn redraw_over_terminal() {
    let overlapped = with(|d| {
        let Some(ti) = d
            .windows
            .iter()
            .position(|w| matches!(w.content, Content::Terminal(c) if c == super::console::USER))
        else {
            return false;
        };
        let t = d.windows[ti].rect;
        d.windows.iter().enumerate().any(|(i, w)| {
            i > ti
                && w.state != WinState::Minimised
                && w.rect.x < t.x + t.w
                && t.x < w.rect.x + w.rect.w
                && w.rect.y < t.y + t.h
                && t.y < w.rect.y + w.rect.h
        })
    })
    .unwrap_or(false);
    // Nothing to repair any more. The console painted its text straight into
    // the back buffer over whatever covered it, and scrolled those pixels --
    // window included -- up a row at a time, so this repainted the terminal
    // between commands. It paints into a surface of its own now, which cannot
    // reach anything else, so the overlap is drawn correctly by every frame.
    let _ = overlapped;
}

pub fn take_pending() -> Option<String> {
    let mut q = PENDING.lock_irq();
    if q.is_empty() {
        return None;
    }
    Some(q.remove(0))
}

/// How many commands are waiting. For the selftest, and for anything that
/// wants to know whether a press landed.
pub fn pending_len() -> usize {
    PENDING.lock_irq().len()
}
