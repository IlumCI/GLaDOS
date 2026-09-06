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
//! The wire format, the object space, and the two interfaces a connection
//! bootstraps through: `wl_display` and `wl_registry`. That is everything a
//! client does before it has asked for anything, and it is worth having on its
//! own because it can be checked completely with no display, no client and no
//! socket -- the same bargain `unix.rs` took, transport first and protocol
//! after.
//!
//! Nothing draws yet. `wl_compositor`, `wl_surface` and `wl_shm` come next,
//! which is where `gfx::compose` gets reached and there is a picture, and
//! after those `xdg_shell`, which is what gives a window a title and a place
//! to sit.

pub mod client;
pub mod object;
pub mod wire;

/// What `diag sky` asks of everything here.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = wire::checks();
    out.extend(object::checks());
    out.extend(client::checks());
    out
}
