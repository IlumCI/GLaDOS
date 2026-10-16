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

use super::client::Client;
use super::wire::{Error, Fixed, Message, Writer};
use alloc::vec::Vec;

pub const WL_OUTPUT: &str = "wl_output";
pub const WL_SEAT: &str = "wl_seat";
pub const WL_POINTER: &str = "wl_pointer";
pub const WL_KEYBOARD: &str = "wl_keyboard";
pub const WL_TOUCH: &str = "wl_touch";

const KEYMAP: &[u8] = include_bytes!("../../assets/xkb/us.xkb");

pub fn handles(iface: &str) -> bool {
    matches!(iface, WL_OUTPUT | WL_SEAT | WL_POINTER | WL_KEYBOARD | WL_TOUCH)
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
}

impl Client {
    /// What must be said when one of these globals is bound.
    pub(super) fn seat_bound(&mut self, id: u32, iface: &str) {
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
                let mut s = Writer::new(id, 3);
                s.int(1);
                self.post(s);
                self.post(Writer::new(id, 2)); // done
            }
            WL_SEAT => {
                let mut c = Writer::new(id, 0);
                c.uint(3); // pointer | keyboard
                self.post(c);
                let mut n = Writer::new(id, 1);
                n.string("seat0");
                self.post(n);
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
            _ => Err(Error::NoMethod),
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
            crate::linux::fs::Memfd { name: alloc::string::String::from("keymap"), at, len, cursor: 0, maps: 0 },
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
                let ptrs = self.seat.pointers.clone();
                if self.seat.over != Some(surface) {
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
            Input::Focus(on) => {
                let kbds = self.seat.keyboards.clone();
                let s = self.next_serial();
                for k in &kbds {
                    if on {
                        let mut w = Writer::new(*k, 1);
                        w.uint(s).object(surface).array(&[]);
                        self.post(w);
                        let mut m = Writer::new(*k, 4);
                        m.uint(s).uint(0).uint(0).uint(0).uint(0);
                        self.post(m);
                    } else {
                        let mut w = Writer::new(*k, 2);
                        w.uint(s).object(surface);
                        self.post(w);
                    }
                }
                self.seat.focus = if on { Some(surface) } else { None };
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
            }
        }
    }
}
