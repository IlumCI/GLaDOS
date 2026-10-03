//! Skywalker, the display server. It speaks Wayland.
//!
//! A Linux program that wants to draw does not write to a framebuffer. It
//! connects to a display server and asks. So every toolkit, every browser, and
//! Wine through `winewayland.drv` sits behind one, and a machine that runs
//! Linux applications has to be one.
//!
//! ### Why Wayland, and why only Wayland
//!
//! X11 is the other answer and it is far more expensive to write from scratch.
//! Its core protocol is 120 requests of drawing primitives that nothing has
//! used since compositing arrived, and everything a modern program actually
//! needs lives in extensions layered on top of that. Wayland's core is 22
//! requests across seven interfaces, and the hard half of a Wayland
//! compositor is already here: `gfx::compose` owns the screen, stacks the
//! windows and composites them, which is the job description.
//!
//! X11 clients are still reachable later, through `Xwayland`, which is itself
//! an ordinary Wayland client. That is the bargain every Wayland compositor
//! makes, and leaving it open costs nothing today.
//!
//! ### The one thing Wayland cannot do without
//!
//! Every buffer a client shows is passed as a **file descriptor** over the
//! connection, so `SCM_RIGHTS` is a hard dependency rather than a convenience.
//! `linux::unix` grew it first for exactly this reason, and the ordering rule
//! it established -- a descriptor is attached to a position in the byte
//! stream -- is what makes the `fd` argument below sound. When a message's
//! bytes are readable, its descriptors are already in hand.
//!
//! ### What is here so far
//!
//! The wire format, the object space, and the bootstrap (`wl_display`,
//! `wl_registry`); then `wl_compositor`, `wl_surface`, `wl_region`, `wl_shm`
//! with its pools and buffers, frame callbacks, and `xdg_wm_base` with
//! `xdg_surface` and `xdg_toplevel` (`surface`). `server` listens at
//! `/run/glados/wayland-0` and takes its turns inside the client's own
//! syscalls. A committed buffer is copied into a desktop window.
//!
//! **A real client draws.** `tools/sky.py` builds an ordinary
//! libwayland-client program and stages it with the host's own glibc; it
//! connects, makes a window, and animates ninety frames on the desktop.
//!
//! Next: `wl_seat`, so the window takes the pointer and the keyboard (the
//! keyboard wants an xkb keymap passed as a descriptor), `wl_output`, and a
//! client that keeps running beside the shell rather than holding it.

pub mod client;
pub mod object;
pub mod server;
pub mod surface;
pub mod wire;

/// What `diag sky` asks of everything here.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = wire::checks();
    out.extend(object::checks());
    out.extend(client::checks());
    out.extend(surface::checks());
    out.extend(server::checks());
    out
}
