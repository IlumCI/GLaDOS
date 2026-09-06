//! `/dev`, which is four nodes and one of them is the screen.
//!
//! Synthetic paths consulted before the store, exactly as `/proc` is and for
//! the same reason: a device resolves to a *function* rather than to a
//! content-addressed blob, so it has no hash, no size that stays still, and no
//! place in a snapshot. It follows the same rule, too --
//!
//! > **A field this machine does not know means the file does not exist.**
//!
//! -- which is why there is no `/dev/tty`, no `/dev/dri`, and no `/dev/snd`.
//! Each is a real interface with real semantics this machine cannot supply,
//! and a node that opens and then does nothing is worse than one that is
//! absent, because an absent one sends a program down a path it already has.
//!
//! ### The screen is the point and the rest are cheap
//!
//! `/dev/fb0` is the smallest well-specified way for a program written
//! somewhere else to put pixels on this machine. It is four ioctls and a
//! mapping, the layouts are fixed, and every graphical thing that follows
//! stands on it. `null`, `zero` and `urandom` come along because a `/dev`
//! holding one node is a `/dev` that gets rebuilt the first time anything real
//! runs, and because all three have exactly one correct behaviour each.
//!
//! ### What a guest gets is the actual framebuffer
//!
//! Not a shadow blitted later. This kernel is identity-mapped, so virtual is
//! physical and `smem_start` can be the framebuffer's own address rather than
//! a lie about a buffer somewhere else -- and a guest writing a frame writes
//! it where the display controller is already looking. The cost is that the
//! desktop has to stand down while that is true, which is `gfx::exclusive`,
//! the same flag `port::with_screen` sets for DOOM and the editor.
//!
//! The alternative was a heap shadow with a blit on `FBIOPAN_DISPLAY`. It buys
//! one thing -- the desktop could keep drawing -- and that one thing is
//! precisely what a full-screen program must not allow, so it buys nothing and
//! costs a copy of the whole screen per frame.

use crate::gfx::Format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// What a `/dev` path resolves to.
///
/// `random` and `urandom` are one variant, which is a deviation with a date on
/// it: they were different devices until Linux 5.6, and since then `random`
/// blocks only until the pool is initialised and behaves as `urandom`
/// afterwards. Blocking is not available here -- there is no guest scheduler
/// to block against, so a wait would be the whole machine spinning -- and a
/// `/dev/random` that returned an error would be one no program on earth
/// expects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Node {
    Null,
    Zero,
    Random,
    Fb,
}

pub fn is_dir(path: &str) -> bool {
    path == "/dev"
}

/// Which node a path names, as a pure function of the path.
///
/// Structural on purpose. `/proc` learned this the expensive way: a `claims`
/// that consulted the running guest made the table change shape depending on
/// whether anything was running, and a listing then offered names that
/// `openat` routed to the store.
pub fn node(path: &str) -> Option<Node> {
    match path {
        "/dev/null" => Some(Node::Null),
        "/dev/zero" => Some(Node::Zero),
        "/dev/random" | "/dev/urandom" => Some(Node::Random),
        "/dev/fb0" => Some(Node::Fb),
        _ => None,
    }
}

pub fn claims(path: &str) -> bool {
    is_dir(path) || node(path).is_some()
}

/// What a listing of `/dev` answers, in the shape `Dir` wants.
pub fn entries(dir: &str) -> Vec<(String, bool, usize)> {
    if dir != "/dev" {
        return Vec::new();
    }
    ["fb0", "null", "random", "urandom", "zero"]
        .iter()
        .map(|n| (n.to_string(), false, 0usize))
        .collect()
}

/// What `stat` reports as the size.
///
/// The framebuffer has one, and it is the same number `smem_len` carries,
/// because a program that seeks to the end of `/dev/fb0` to find the screen
/// size is doing something legal. Everything else is zero, which is what Linux
/// reports for a character device and is true here in the sense that matters:
/// there are no bytes sitting anywhere waiting to be read.
pub fn size(n: Node) -> usize {
    match n {
        Node::Fb => fb().map(|f| f.3).unwrap_or(0),
        _ => 0,
    }
}

/// The framebuffer, as (address, pixels wide, pixels high, byte length, bytes
/// per scan line, format), or nothing when there is no display at all.
///
/// **`stride` is not `width`.** Frequently larger, and using one for the other
/// gives a picture that shears diagonally -- `gfx` says so on the field itself,
/// having paid for it, and a guest handed the wrong `line_length` produces the
/// identical picture from the other side of the seam.
#[allow(clippy::type_complexity)]
pub fn fb() -> Option<(u64, u32, u32, usize, u32, Format)> {
    let f = crate::gfx::primary()?;
    let pitch = f.stride() * 4;
    let len = pitch as usize * f.height() as usize;
    Some((f.addr(), f.width(), f.height(), len, pitch, f.format()))
}

// The four ioctls a framebuffer program uses. `FBIOGET_CON2FBMAP` and the
// panning pair are absent rather than stubbed: this display cannot pan, and
// answering a pan with success would leave a program drawing into a page
// nothing scans out.
pub const FBIOGET_VSCREENINFO: u64 = 0x4600;
pub const FBIOPUT_VSCREENINFO: u64 = 0x4601;
pub const FBIOGET_FSCREENINFO: u64 = 0x4602;
pub const FBIOBLANK: u64 = 0x4611;

/// `struct fb_var_screeninfo`, 160 bytes on x86-64.
pub const VAR_LEN: usize = 160;
/// `struct fb_fix_screeninfo`, 80 bytes on x86-64.
///
/// The size is not obvious from the field list and is worth stating: three
/// `__u16` panning fields end at offset 46, `line_length` is a `__u32` so it
/// aligns to 48, and `mmio_start` is an `unsigned long` so it aligns to 56
/// rather than sitting at 52. A reader that packed it tightly would find every
/// field after `ywrapstep` in the wrong place -- and `line_length` in the
/// wrong place is a sheared picture rather than an error.
pub const FIX_LEN: usize = 80;

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

/// Where each channel sits inside a pixel, as (offset, length) triples.
///
/// **This is the field that is silently wrong.** `Format` names the order of
/// the *bytes in memory*; `fb_bitfield` names bit positions inside a
/// little-endian `u32`, so the two run in opposite directions and getting it
/// backwards swaps red and blue. Nothing errors, nothing faults, and the
/// picture is merely a strange colour -- which is why the fixture draws a
/// known red rather than a gradient.
///
/// `Bgrx` is bytes B, G, R, x, so as a word that is `0x00RRGGBB`: blue at bit
/// 0, red at bit 16. `Rgbx` is the mirror. Transparency is length **zero** in
/// both, because the fourth byte is genuinely unused here rather than an alpha
/// channel anybody honours, and claiming eight bits of alpha would invite a
/// program to blend against a value the display ignores.
pub fn channels(f: Format) -> [(u32, u32); 4] {
    match f {
        Format::Bgrx => [(16, 8), (8, 8), (0, 8), (24, 0)],
        Format::Rgbx => [(0, 8), (8, 8), (16, 8), (24, 0)],
    }
}

/// The current mode, in the shape `FBIOGET_VSCREENINFO` answers.
///
/// Everything about timing is zero -- `pixclock`, the four margins, the two
/// sync lengths -- and that is the rule this module opens with rather than an
/// omission. There is no mode-setting here: UEFI handed over a display already
/// running, and this kernel never learned what timings it is running at. A
/// program that reads them gets zero, which is what every framebuffer driver
/// over a pre-set mode reports and what `fbset` prints as an unknown mode.
pub fn var_screeninfo() -> Option<[u8; VAR_LEN]> {
    let (_, w, h, _, pitch, fmt) = fb()?;
    let mut b = [0u8; VAR_LEN];
    put32(&mut b, 0, w);
    put32(&mut b, 4, h);
    // The virtual resolution is the visible one. A larger virtual buffer is
    // how panning works, and this cannot pan.
    put32(&mut b, 8, pitch / 4);
    put32(&mut b, 12, h);
    put32(&mut b, 24, 32);
    let ch = channels(fmt);
    for (i, (off, len)) in ch.iter().enumerate() {
        put32(&mut b, 32 + i * 12, *off);
        put32(&mut b, 36 + i * 12, *len);
    }
    // `activate` is FB_ACTIVATE_NOW, which is zero, so it is already right.
    // `vmode` is FB_VMODE_NONINTERLACED, also zero. Both are named here
    // because a reader checking this against the header will look for them.
    Some(b)
}

/// The fixed properties, in the shape `FBIOGET_FSCREENINFO` answers.
pub fn fix_screeninfo() -> Option<[u8; FIX_LEN]> {
    let (at, _, _, len, pitch, _) = fb()?;
    let mut b = [0u8; FIX_LEN];
    b[..6].copy_from_slice(b"glados");
    b[16..24].copy_from_slice(&at.to_le_bytes());
    put32(&mut b, 24, len as u32);
    // FB_TYPE_PACKED_PIXELS is 0 and FB_VISUAL_TRUECOLOR is 2. Truecolor and
    // not DIRECTCOLOR, which is the one with a writable palette -- this has no
    // palette at all, so a program told DIRECTCOLOR would go looking for
    // `FBIOPUTCMAP` and find nothing.
    put32(&mut b, 36, 2);
    // Panning of every kind is zero steps, which is how a driver says it
    // cannot. `xpanstep` at 40, `ypanstep` at 42, `ywrapstep` at 44.
    put16(&mut b, 40, 0);
    put16(&mut b, 42, 0);
    put16(&mut b, 44, 0);
    put32(&mut b, 48, pitch);
    // `mmio_start`/`mmio_len` at 56 and 64 stay zero: those describe a
    // register aperture a driver would poke, and there is no such thing to
    // hand out. `accel` at 68 is FB_ACCEL_NONE, which is zero and is true.
    Some(b)
}

/// Whether a proposed mode is the one already running.
///
/// **The only honest answer to `FBIOPUT_VSCREENINFO` is yes or `EINVAL`.**
/// There is no mode-setting on this machine: the resolution, stride and pixel
/// format are whatever the firmware left. Accepting a mode change and not
/// performing it is the worst of the three available answers, because the
/// program then draws at a geometry the hardware does not have and every frame
/// after that is garbage with no error anywhere.
///
/// Only the four fields that would change what a frame looks like are
/// compared. A program is free to ask for different timings or a different
/// `activate`, and refusing over those would turn every well-behaved
/// mode-setting call into a failure for no reason.
pub fn mode_matches(want: &[u8]) -> bool {
    let Some(have) = var_screeninfo() else { return false };
    if want.len() < VAR_LEN {
        return false;
    }
    let at = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    // xres, yres, bits_per_pixel, and the virtual height. The virtual *width*
    // is deliberately not compared: a program that asks for a virtual width
    // equal to its visible one is asking for less than it has, which costs it
    // nothing here since it is told the real `line_length` regardless.
    at(want, 0) == at(&have, 0)
        && at(want, 4) == at(&have, 4)
        && at(want, 24) == at(&have, 24)
        && at(want, 12) <= at(&have, 12)
}

/// Read from a node at a cursor, answering how many bytes landed.
pub fn read(n: Node, at: usize, buf: &mut [u8]) -> usize {
    match n {
        // A permanent end of file, which is the whole of `/dev/null`.
        Node::Null => 0,
        Node::Zero => {
            buf.fill(0);
            buf.len()
        }
        Node::Random => {
            // `fill` and not `fill_secret`. The secret form refuses below the
            // entropy threshold, and a `/dev/urandom` that returns an error is
            // one nothing copes with -- glibc's own fallback for a failing
            // `getrandom` is to open this and read it, so a refusal here
            // leaves a program with no third option.
            crate::rng::fill(buf);
            buf.len()
        }
        Node::Fb => {
            let Some((base, _, _, len, _, _)) = fb() else { return 0 };
            if at >= len {
                return 0;
            }
            let n = buf.len().min(len - at);
            // Legal, and `fbgrab` is built entirely out of it. Safe to read
            // because `build_identity_map` gives the aperture a write-back
            // mapping rather than an uncached one, which `gfx` records as the
            // reason its own fast paths are allowed to exist.
            unsafe { core::ptr::copy_nonoverlapping((base + at as u64) as *const u8, buf.as_mut_ptr(), n) };
            n
        }
    }
}

/// Write to a node at a cursor, answering how many bytes were taken.
pub fn write(n: Node, at: usize, buf: &[u8]) -> usize {
    match n {
        // Both accept everything and keep none of it, which is what they are
        // for. A short write here would make `> /dev/null` an error.
        Node::Null | Node::Zero => buf.len(),
        // Linux mixes a write into the entropy pool without crediting it.
        // Accepted and discarded instead, because letting a guest choose what
        // goes into the kernel's generator is a capability nothing has asked
        // for and the answer a program checks is the byte count.
        Node::Random => buf.len(),
        Node::Fb => {
            let Some((base, _, _, len, _, _)) = fb() else { return 0 };
            if at >= len {
                // Past the end of video memory is `ENOSPC` on Linux, and the
                // caller turns a zero into that. Not a short write of nothing,
                // which a copy loop would spin on forever.
                return 0;
            }
            hold_screen(true);
            let n = buf.len().min(len - at);
            unsafe { core::ptr::copy_nonoverlapping(buf.as_ptr(), (base + at as u64) as *mut u8, n) };
            n
        }
    }
}

/// Whether a guest currently owns the display.
static HOLDING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Take the screen for a guest, or give it back.
///
/// The two halves of `port::with_screen`, split because a guest holds the
/// display across many syscalls rather than for the length of one call. Taken
/// on the first write or mapping rather than at `open`, since a program that
/// merely `stat`s `/dev/fb0` has not asked for the machine.
///
/// Giving it back has an order that is load-bearing and was found by looking
/// at a screenshot: the compositor's shadow has to be **forgotten first**, or
/// `present` compares the desktop it still believes is on screen against the
/// desktop it is about to draw, finds every row unchanged, repaints nothing,
/// and leaves the guest's last frame up with a terminal drawn over it.
pub fn hold_screen(on: bool) {
    use core::sync::atomic::Ordering;
    if HOLDING.swap(on, Ordering::AcqRel) == on {
        return;
    }
    crate::gfx::set_exclusive(on);
    if on {
        // Blanked on the way in, for the reason `port::with_screen` gives: a
        // frame smaller than the screen is letterboxed by whatever the desktop
        // last composed there, and against black a painter that failed to
        // stand down is obvious rather than indistinguishable.
        if let Some(f) = crate::gfx::primary() {
            f.fill(crate::gfx::Color::new(0, 0, 0));
        }
    } else {
        crate::gfx::compose::invalidate();
        crate::gfx::desk::draw();
    }
}

pub fn holding() -> bool {
    HOLDING.load(core::sync::atomic::Ordering::Acquire)
}

/// What `diag linux` asks of `/dev`.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    out.push((
        "every /dev node is named by the path alone, so the table cannot change shape",
        node("/dev/fb0") == Some(Node::Fb)
            && node("/dev/null") == Some(Node::Null)
            && node("/dev/urandom") == node("/dev/random")
            && node("/dev/fb1").is_none()
            && !claims("/dev/tty")
            && !claims("/dev/dri/card0"),
    ));
    out.push((
        "and a listing offers only names that answer",
        entries("/dev").iter().all(|(n, d, _)| {
            !d && claims(&alloc::format!("/dev/{}", n))
        }) && entries("/tmp").is_empty(),
    ));

    // The two structures are an ABI rather than a choice, the same argument
    // `struct stat` makes: a field short is not a smaller answer, it is a
    // different structure, and the caller reads past what was written.
    out.push((
        "fb_var_screeninfo is 160 bytes and fb_fix_screeninfo is 80",
        VAR_LEN == 160 && FIX_LEN == 80,
    ));

    // Both directions of the pixel layout, because the failure is a picture in
    // the wrong colour rather than an error. Blue at bit 0 for Bgrx is the
    // claim: bytes B,G,R,x read as a little-endian word put blue lowest.
    let bgr = channels(Format::Bgrx);
    let rgb = channels(Format::Rgbx);
    out.push((
        "a Bgrx screen puts blue at bit 0 and red at 16, and Rgbx is the mirror",
        bgr[0] == (16, 8) && bgr[2] == (0, 8) && rgb[0] == (0, 8) && rgb[2] == (16, 8),
    ));
    out.push((
        "and neither claims alpha bits the display does not honour",
        bgr[3].1 == 0 && rgb[3].1 == 0,
    ));

    if let (Some(v), Some(f)) = (var_screeninfo(), fix_screeninfo()) {
        let at32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let (base, w, h, len, pitch, _) = fb().unwrap_or((0, 0, 0, 0, 0, Format::Bgrx));
        out.push((
            "the mode reported is the one the firmware actually left running",
            at32(&v, 0) == w && at32(&v, 4) == h && at32(&v, 24) == 32,
        ));
        // The one that shears. `line_length` is bytes and `xres` is pixels, so
        // a driver reporting `xres * 4` on a screen whose stride is wider
        // gives a picture sliding a little further left on every row.
        out.push((
            "line_length is the real stride in bytes, which is not xres times four",
            at32(&f, 48) == pitch && pitch >= w * 4,
        ));
        out.push((
            "smem_start is the aperture itself, since virtual is physical here",
            u64::from_le_bytes(f[16..24].try_into().unwrap_or_default()) == base,
        ));
        out.push((
            "and smem_len covers every scan line rather than every visible pixel",
            at32(&f, 24) as usize == len && len == pitch as usize * h as usize,
        ));
        out.push((
            "the mode already running is accepted, since accepting it changes nothing",
            mode_matches(&v),
        ));
        // The refusal that matters. A program told "yes" about a mode this
        // machine cannot enter draws every frame afterwards at a geometry the
        // display does not have, and nothing reports it.
        let mut other = v;
        put32(&mut other, 0, w + 16);
        out.push((
            "and a different resolution is refused rather than accepted and ignored",
            !mode_matches(&other),
        ));
        let mut depth = v;
        put32(&mut depth, 24, 16);
        out.push((
            "as is a different depth, which is the same lie in a smaller field",
            !mode_matches(&depth),
        ));
        out.push((
            "a truncated structure is refused rather than read past its end",
            !mode_matches(&v[..VAR_LEN - 1]),
        ));
    }

    let mut buf = [0xAAu8; 8];
    out.push((
        "reading /dev/null is a permanent end of file and leaves the buffer alone",
        read(Node::Null, 0, &mut buf) == 0 && buf == [0xAA; 8],
    ));
    out.push((
        "reading /dev/zero fills what it was given",
        read(Node::Zero, 0, &mut buf) == 8 && buf == [0; 8],
    ));
    out.push((
        "and writing to either takes everything, or a redirect becomes an error",
        write(Node::Null, 0, &[1, 2, 3]) == 3 && write(Node::Zero, 0, &[1, 2, 3]) == 3,
    ));
    // Not a randomness test, which nothing at boot can be. It checks the one
    // failure that would matter: `fill` refusing and leaving the buffer as it
    // found it, which is what `fill_secret` does below the entropy threshold
    // and is exactly why this node does not call it.
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    read(Node::Random, 0, &mut a);
    read(Node::Random, 0, &mut b);
    out.push((
        "/dev/urandom answers, and twice in a row does not answer the same thing",
        a != [0u8; 32] && a != b,
    ));
    out
}
