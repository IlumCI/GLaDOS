//! `wl_output` and `wl_seat`: the screen a client is on, and the keyboard and
//! pointer it is driven by.
//!
//! **Both are what a toolkit asks for before it will open a window.** SDL's
//! Wayland backend counts outputs to decide what displays exist and refuses
//! to start with none, and it binds the seat for input; a server offering
//! neither got a client that drew nothing. So the output is the one screen
//! there is, at its real size, and the seat has the two devices this machine
//! has.
//!
//! **Keys arrive as evdev codes, which is what `wl_keyboard.key` carries.**
//! `linux::input` already keeps every key press and release as an evdev event
//! (it is `/dev/input/event0`), so the keyboard reads that ring while its
//! window has the focus -- a press is a press and a release a release, where
//! the desktop's own key path hands an application decoded characters, which
//! is right for a text field and useless for a game.
//!
//! **The keymap is real** (`assets/xkb/us.xkb`, compiled from xkeyboard-config
//! with `xkbcli compile-keymap --rules evdev --model pc105 --layout us`), sent
//! the way every compositor sends one: text in a `memfd`, by descriptor.
//!
//! Pointer input comes from the desktop window's own press, drag and release,
//! in window-local coordinates. Motion with no button held is not delivered
//! there, so a client sees the pointer where it last pressed -- enough to
//! click a menu, and not yet enough to aim.

use super::client::{done, Client};
use super::wire::{Error, Fixed, Message, Writer};
use alloc::vec::Vec;

pub const WL_OUTPUT: &str = "wl_output";
pub const WL_SEAT: &str = "wl_seat";
pub const WL_POINTER: &str = "wl_pointer";
pub const WL_KEYBOARD: &str = "wl_keyboard";
pub const WL_TOUCH: &str = "wl_touch";
pub const RELATIVE_MANAGER: &str = "zwp_relative_pointer_manager_v1";
pub const RELATIVE: &str = "zwp_relative_pointer_v1";
pub const CONSTRAINTS: &str = "zwp_pointer_constraints_v1";
pub const LOCKED: &str = "zwp_locked_pointer_v1";
pub const CONFINED: &str = "zwp_confined_pointer_v1";
pub const DATA_MANAGER: &str = "wl_data_device_manager";
pub const DATA_SOURCE: &str = "wl_data_source";
pub const DATA_DEVICE: &str = "wl_data_device";
pub const KDE_DECORATION_MANAGER: &str = "org_kde_kwin_server_decoration_manager";
pub const KDE_DECORATION: &str = "org_kde_kwin_server_decoration";
pub const XDG_DECORATION_MANAGER: &str = "zxdg_decoration_manager_v1";
pub const XDG_DECORATION: &str = "zxdg_toplevel_decoration_v1";
/// `org_kde_kwin_server_decoration_manager`'s mode for "the server draws it".
const DECORATION_SERVER: u32 = 2;

const KEYMAP: &[u8] = include_bytes!("../../assets/xkb/us.xkb");

pub fn handles(iface: &str) -> bool {
    matches!(
        iface,
        WL_OUTPUT | WL_SEAT | WL_POINTER | WL_KEYBOARD | WL_TOUCH | RELATIVE_MANAGER | RELATIVE | CONSTRAINTS | LOCKED | CONFINED
            | DATA_MANAGER | DATA_SOURCE | DATA_DEVICE | KDE_DECORATION_MANAGER | KDE_DECORATION
            | XDG_DECORATION_MANAGER | XDG_DECORATION
    )
}

/// Input for one window, as the desktop saw it.
#[derive(Clone, Copy, Debug)]
pub enum Input {
    /// The pointer at (x, y) in the window, with a button changing or not.
    Pointer { x: i32, y: i32, button: Option<(u32, bool)> },
    /// The window gained or lost the keyboard.
    Focus(bool),
    /// An evdev key code going down or up.
    Key { code: u32, down: bool },
    /// The wheel, in notches; positive scrolls down, as Wayland counts.
    Wheel(i32),
}

/// What the seat side of a connection remembers.
#[derive(Default)]
pub struct Seat {
    pointers: Vec<u32>,
    keyboards: Vec<u32>,
    /// The surface the pointer is over, once told.
    over: Option<u32>,
    /// The surface holding the keyboard, once told.
    focus: Option<u32>,
    /// `zwp_relative_pointer_v1` objects: where unaccelerated motion goes.
    relative: Vec<u32>,
    /// A `zwp_locked_pointer_v1` asked for: (object, surface, oneshot), and
    /// whether it is in force now.
    pub(super) lock: Option<(u32, u32, bool)>,
    pub(super) locked: bool,
    /// Modifier masks in the keymap's own bits: held, and toggled on.
    mods_down: u32,
    mods_locked: u32,
}

/// Which modifier an evdev code is, as the bit `assets/xkb/us.xkb` gives it:
/// Shift 1, Control 4, Mod1 (Alt) 8, Mod4 (Super) 64. Caps Lock (Lock, 2) is
/// a toggle and handled apart.
fn modifier_bit(code: u32) -> Option<u32> {
    match code {
        42 | 54 => Some(1),
        29 | 97 => Some(4),
        56 | 100 => Some(8),
        125 | 126 => Some(64),
        _ => None,
    }
}

impl Client {
    /// What must be said when one of these globals is bound.
    ///
    /// **Only events the bound version has.** A client binding `wl_seat` at
    /// version 1 has no handler for `name`, which arrived in version 2, and
    /// libwayland aborts a client sent an event it cannot dispatch:
    /// weston-simple-shm died of SIGABRT on its first roundtrip for this.
    pub(super) fn seat_bound(&mut self, id: u32, iface: &str, version: u32) {
        match iface {
            WL_OUTPUT => {
                let (w, h) = crate::gfx::primary().map(|f| (f.width() as i32, f.height() as i32)).unwrap_or((1280, 800));
                // geometry: x, y, physical mm, subpixel, make, model, transform.
                let mut g = Writer::new(id, 0);
                g.int(0).int(0).int(w * 254 / 960).int(h * 254 / 960).int(0);
                g.string("GLaDOS").string("desktop").int(0);
                self.post(g);
                // mode: current and preferred, at 60 Hz.
                let mut m = Writer::new(id, 1);
                m.uint(3).int(w).int(h).int(60_000);
                self.post(m);
                // scale and done are version 2.
                if version >= 2 {
                    let mut s = Writer::new(id, 3);
                    s.int(1);
                    self.post(s);
                    self.post(Writer::new(id, 2)); // done
                }
            }
            // Every window here already has the desktop's frame round it, so
            // a toolkit drawing its own title bar inside that frame draws two.
            // GTK 3 has no xdg-decoration and does read KDE's older protocol,
            // so this is the one that tells it the server decorates.
            KDE_DECORATION_MANAGER => {
                let mut m = Writer::new(id, 0); // default_mode
                m.uint(DECORATION_SERVER);
                self.post(m);
            }
            WL_SEAT => {
                let mut c = Writer::new(id, 0);
                c.uint(3); // pointer | keyboard
                self.post(c);
                if version >= 2 {
                    let mut n = Writer::new(id, 1);
                    n.string("seat0");
                    self.post(n);
                }
            }
            _ => {}
        }
    }

    pub(super) fn seat_request(&mut self, iface: &'static str, version: u32, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match (iface, msg.opcode) {
            (WL_SEAT, 0) => {
                let id = r.new_id()?;
                self.objects.create(id, WL_POINTER, version)?;
                self.seat.pointers.push(id);
                Ok(())
            }
            (WL_SEAT, 1) => {
                let id = r.new_id()?;
                self.objects.create(id, WL_KEYBOARD, version)?;
                self.seat.keyboards.push(id);
                self.send_keymap(id, version);
                // A keyboard made after its window took the focus is told so
                // now. Focus is announced to the keyboards that exist at the
                // time, and GTK asks for its keyboard after the window is up
                // -- so it never saw an `enter`, dropped every key as aimed
                // at no surface, and a document could not be typed into.
                if let Some(surface) = self.seat.focus {
                    let s = self.next_serial();
                    let mut w = Writer::new(id, 1);
                    w.uint(s).object(surface).array(&[]);
                    self.post(w);
                    let mut m = Writer::new(id, 4);
                    m.uint(s).uint(0).uint(0).uint(0).uint(0);
                    self.post(m);
                }
                Ok(())
            }
            (WL_SEAT, 2) => {
                let id = r.new_id()?;
                self.objects.create(id, WL_TOUCH, version)?;
                Ok(())
            }
            // release, on every one of them.
            (WL_SEAT, 3) | (WL_POINTER, 1) | (WL_KEYBOARD, 0) | (WL_TOUCH, 0) | (WL_OUTPUT, 0) => {
                self.seat.pointers.retain(|p| *p != msg.object);
                self.seat.keyboards.retain(|k| *k != msg.object);
                self.objects.destroy(msg.object)?;
                let mut w = Writer::new(super::object::DISPLAY, 1);
                w.uint(msg.object);
                self.post(w);
                Ok(())
            }
            // set_cursor: the desktop draws its own pointer.
            (WL_POINTER, 0) => Ok(()),
            // Relative motion and pointer lock: what a game turns with. Both
            // managers' destroy, and the objects' own.
            (RELATIVE_MANAGER, 0) | (CONSTRAINTS, 0) | (RELATIVE, 0) | (LOCKED, 0) | (CONFINED, 0) => {
                self.seat.relative.retain(|p| *p != msg.object);
                if self.seat.lock.is_some_and(|(o, _, _)| o == msg.object) {
                    self.seat.lock = None;
                    self.seat.locked = false;
                }
                self.objects.destroy(msg.object)?;
                let mut w = Writer::new(super::object::DISPLAY, 1);
                w.uint(msg.object);
                self.post(w);
                Ok(())
            }
            (RELATIVE_MANAGER, 1) => {
                let id = r.new_id()?;
                let _pointer = r.object()?;
                self.objects.create(id, RELATIVE, version)?;
                self.seat.relative.push(id);
                Ok(())
            }
            // lock_pointer(id, surface, pointer, region, lifetime), and
            // confine_pointer with the same arguments: a confined pointer is
            // taken as a lock, which is stricter and is what a game confining
            // its cursor wants anyway.
            (CONSTRAINTS, 1) | (CONSTRAINTS, 2) => {
                let id = r.new_id()?;
                let surface = r.object()?;
                let _pointer = r.object()?;
                let _region = r.object()?;
                let lifetime = r.uint()?;
                self.objects.create(id, if msg.opcode == 1 { LOCKED } else { CONFINED }, version)?;
                self.seat.lock = Some((id, surface, lifetime == 1));
                self.seat.locked = false;
                Ok(())
            }
            // set_cursor_position_hint, set_region: nothing to do with either.
            (LOCKED, 1) | (LOCKED, 2) | (CONFINED, 1) => Ok(()),
            // create(id, surface): the decoration of one surface, and its mode
            // said at once. request_mode is honoured by answering what is
            // true, which is server-side whatever was asked.
            (KDE_DECORATION_MANAGER, 0) => {
                let id = r.new_id()?;
                self.objects.create(id, KDE_DECORATION, version)?;
                let mut m = Writer::new(id, 0);
                m.uint(DECORATION_SERVER);
                self.post(m);
                Ok(())
            }
            (KDE_DECORATION, 1) => {
                let mut m = Writer::new(msg.object, 0);
                m.uint(DECORATION_SERVER);
                self.post(m);
                Ok(())
            }
            // The standard decoration protocol, which Qt, GTK 4 and SDL use:
            // every toplevel is decorated by the server, and a request for
            // either mode is answered with that one.
            (XDG_DECORATION_MANAGER, 0) => {
                done(&r)?;
                self.objects.destroy(msg.object)?;
                let mut w = Writer::new(super::object::DISPLAY, 1);
                w.uint(msg.object);
                self.post(w);
                Ok(())
            }
            (XDG_DECORATION_MANAGER, 1) => {
                let id = r.new_id()?;
                let _toplevel = r.object()?;
                done(&r)?;
                self.objects.create(id, XDG_DECORATION, version)?;
                let mut m = Writer::new(id, 0);
                m.uint(2); // server_side
                self.post(m);
                Ok(())
            }
            (XDG_DECORATION, 1) | (XDG_DECORATION, 2) => {
                let mut m = Writer::new(msg.object, 0);
                m.uint(2);
                self.post(m);
                Ok(())
            }
            (XDG_DECORATION, 0) => {
                done(&r)?;
                self.objects.destroy(msg.object)?;
                let mut w = Writer::new(super::object::DISPLAY, 1);
                w.uint(msg.object);
                self.post(w);
                Ok(())
            }
            (KDE_DECORATION, 0) => {
                self.objects.destroy(msg.object)?;
                let mut w = Writer::new(super::object::DISPLAY, 1);
                w.uint(msg.object);
                self.post(w);
                Ok(())
            }
            // The clipboard and drag-and-drop, accepted and carrying nothing.
            // **GTK 3 will not create its seat without this global**: the seat
            // waits on `wl_data_device_manager`, so with no manager there was
            // no keyboard or pointer at all and LibreOffice's first key lookup
            // asserted. Sources are taken and never offered to anybody, and a
            // device never announces an offer, which a client reads as an
            // empty clipboard -- true, since nothing here holds one yet.
            (DATA_MANAGER, 0) => {
                let id = r.new_id()?;
                self.objects.create(id, DATA_SOURCE, version)?;
                Ok(())
            }
            (DATA_MANAGER, 1) => {
                let id = r.new_id()?;
                self.objects.create(id, DATA_DEVICE, version)?;
                Ok(())
            }
            // offer, set_actions; start_drag, set_selection.
            (DATA_SOURCE, 0) | (DATA_SOURCE, 2) | (DATA_DEVICE, 0) | (DATA_DEVICE, 1) => Ok(()),
            // wl_data_source.destroy and wl_data_device.release.
            (DATA_SOURCE, 1) | (DATA_DEVICE, 2) => {
                self.objects.destroy(msg.object)?;
                let mut w = Writer::new(super::object::DISPLAY, 1);
                w.uint(msg.object);
                self.post(w);
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }

    /// Whether the lock asked for should be in force now, given which
    /// window has the keyboard; sends `locked` or `unlocked` on a change and
    /// answers the window key holding the pointer.
    pub fn lock_turn(&mut self, focused: Option<u64>) -> Option<u64> {
        let (obj, surface, oneshot) = self.seat.lock?;
        let key = self.key_of(surface);
        let want = key.is_some() && key == focused;
        if want && !self.seat.locked {
            self.seat.locked = true;
            // Motion from before the lock is not this window's.
            let _ = crate::dev::mouse::take_relative();
            self.post(Writer::new(obj, 0));
        } else if !want && self.seat.locked {
            self.seat.locked = false;
            self.post(Writer::new(obj, 1));
            if oneshot {
                // A one-shot lock is spent once lifted; the client asks again.
                self.seat.lock = None;
            }
        }
        if self.seat.locked { key } else { None }
    }

    /// Relative motion while the pointer is locked, in mouse counts with y
    /// downward, which is the sense `zwp_relative_pointer_v1` reports.
    pub fn relative_motion(&mut self, dx: i32, dy: i32) {
        let us = crate::net::now_ms().wrapping_mul(1000);
        for p in self.seat.relative.clone() {
            let mut w = Writer::new(p, 0);
            w.uint((us >> 32) as u32).uint(us as u32);
            w.fixed(Fixed::from_int(dx)).fixed(Fixed::from_int(dy));
            w.fixed(Fixed::from_int(dx)).fixed(Fixed::from_int(dy));
            self.post(w);
        }
        for p in self.seat.pointers.clone() {
            if self.objects.get(p).is_some_and(|e| e.version >= 5) {
                self.post(Writer::new(p, 5));
            }
        }
    }

    fn send_keymap(&mut self, id: u32, version: u32) {
        let len = KEYMAP.len() + 1;
        let Some(at) = crate::linux::syscall::alloc_pages(len) else { return };
        unsafe {
            core::ptr::copy_nonoverlapping(KEYMAP.as_ptr(), at as *mut u8, KEYMAP.len());
            *((at + KEYMAP.len() as u64) as *mut u8) = 0;
        }
        let fd = crate::linux::fs::Fd::Memfd(alloc::rc::Rc::new(core::cell::RefCell::new(
            crate::linux::fs::Memfd { name: alloc::string::String::from("keymap"), at, len, cursor: 0, maps: 0, seals: 1 },
        )));
        self.out_fds.push(fd);
        let mut w = Writer::new(id, 0);
        w.uint(1).uint(len as u32); // XKB_V1, then the fd travels beside the bytes
        self.post(w);
        if version >= 4 {
            let mut ri = Writer::new(id, 5);
            ri.int(25).int(600);
            self.post(ri);
        }
    }

    fn next_serial(&mut self) -> u32 {
        self.serial = self.serial.wrapping_add(1);
        self.serial
    }

    /// Deliver one input event to `surface`.
    pub fn input(&mut self, surface: u32, ev: Input) {
        let time = crate::net::now_ms() as u32;
        match ev {
            Input::Pointer { x, y, button } => {
                // A press outside an open menu closes it and goes no further.
                if let Some((_, true)) = button {
                    if self.dismiss_outside(surface, x, y) {
                        return;
                    }
                }
                // The topmost surface under the pointer -- a popup, a
                // subsurface, or the window itself -- in its own coordinates.
                let (surface, x, y) = self.pick(surface, x, y);
                let ptrs = self.seat.pointers.clone();
                if self.seat.over != Some(surface) {
                    // `leave` before `enter`: a toolkit tracks hover per
                    // surface, and one never told the pointer left keeps its
                    // last item highlighted.
                    if let Some(old) = self.seat.over {
                        let s = self.next_serial();
                        for p in &ptrs {
                            let mut w = Writer::new(*p, 1);
                            w.uint(s).object(old);
                            self.post(w);
                        }
                    }
                    let s = self.next_serial();
                    for p in &ptrs {
                        let mut w = Writer::new(*p, 0);
                        w.uint(s).object(surface).fixed(Fixed::from_int(x)).fixed(Fixed::from_int(y));
                        self.post(w);
                    }
                    self.seat.over = Some(surface);
                }
                for p in &ptrs {
                    let mut w = Writer::new(*p, 2);
                    w.uint(time).fixed(Fixed::from_int(x)).fixed(Fixed::from_int(y));
                    self.post(w);
                }
                if let Some((code, down)) = button {
                    let s = self.next_serial();
                    for p in &ptrs {
                        let mut w = Writer::new(*p, 3);
                        w.uint(s).uint(time).uint(code).uint(down as u32);
                        self.post(w);
                    }
                }
                for p in &ptrs {
                    if self.objects.get(*p).is_some_and(|e| e.version >= 5) {
                        self.post(Writer::new(*p, 5)); // frame
                    }
                }
            }
            // axis_source wheel, axis_discrete, and the value: fifteen units a
            // notch, which is what libinput reports and what toolkits divide
            // by. Version-5 pointers get the discrete count; older ones only
            // the value.
            Input::Wheel(notches) => {
                let ptrs = self.seat.pointers.clone();
                for p in &ptrs {
                    let v5 = self.objects.get(*p).is_some_and(|e| e.version >= 5);
                    if v5 {
                        let mut src = Writer::new(*p, 6);
                        src.uint(0);
                        self.post(src);
                        let mut d = Writer::new(*p, 8);
                        d.uint(0).int(notches);
                        self.post(d);
                    }
                    let mut a = Writer::new(*p, 4);
                    a.uint(time).uint(0).fixed(Fixed::from_int(notches * 15));
                    self.post(a);
                    if v5 {
                        self.post(Writer::new(*p, 5));
                    }
                }
            }
            Input::Focus(on) => {
                let kbds = self.seat.keyboards.clone();
                let s = self.next_serial();
                for k in &kbds {
                    if on {
                        let mut w = Writer::new(*k, 1);
                        w.uint(s).object(surface).array(&[]);
                        self.post(w);
                        let mut m = Writer::new(*k, 4);
                        m.uint(s).uint(self.seat.mods_down).uint(0).uint(self.seat.mods_locked).uint(0);
                        self.post(m);
                    } else {
                        let mut w = Writer::new(*k, 2);
                        w.uint(s).object(surface);
                        self.post(w);
                    }
                }
                self.seat.focus = if on { Some(surface) } else { None };
                self.activate(surface, on);
            }
            Input::Key { code, down } => {
                if self.seat.focus != Some(surface) {
                    self.input(surface, Input::Focus(true));
                }
                let kbds = self.seat.keyboards.clone();
                let s = self.next_serial();
                for k in &kbds {
                    let mut w = Writer::new(*k, 3);
                    w.uint(s).uint(time).uint(code).uint(down as u32);
                    self.post(w);
                }
                // **The modifier state is the server's to say.** A client
                // does not work out Shift from the Shift key: it reads
                // `modifiers`, and this sent that once, as zero, at focus --
                // so a held Shift typed lowercase in every toolkit there is.
                let (was_down, was_locked) = (self.seat.mods_down, self.seat.mods_locked);
                if let Some(bit) = modifier_bit(code) {
                    if down {
                        self.seat.mods_down |= bit;
                    } else {
                        self.seat.mods_down &= !bit;
                    }
                } else if code == 58 && down {
                    self.seat.mods_locked ^= 2;
                }
                if (self.seat.mods_down, self.seat.mods_locked) != (was_down, was_locked) {
                    let s = self.next_serial();
                    for k in &kbds {
                        let mut m = Writer::new(*k, 4);
                        m.uint(s).uint(self.seat.mods_down).uint(0).uint(self.seat.mods_locked).uint(0);
                        self.post(m);
                    }
                }
            }
        }
    }
}
