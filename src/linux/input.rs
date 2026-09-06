//! `/dev/input/event0` and `event1`, which is how a Linux program reads a
//! keyboard and a mouse.
//!
//! A screen with no input is a picture. This is the other half of `/dev/fb0`,
//! and it is the same shape: a small, exactly-specified interface that a large
//! amount of software already speaks, sitting on drivers this kernel already
//! has.
//!
//! ### What an evdev device is
//!
//! A stream of 24-byte records and a set of ioctls describing what the device
//! can produce. The records are the easy half. The ioctls are where a device
//! is *classified*, and that is the part worth getting right: SDL and every
//! other input stack decide what a device is by reading its capability
//! bitmaps, so a mouse that fails to advertise `REL_X` is a device nothing
//! opens. The name is decoration; the bitmaps are the interface.
//!
//! ### Three details that are silent when wrong
//!
//! **`SYN_REPORT` is load-bearing.** A reader batches events until it sees
//! one and treats the batch as a single state change, so a device that never
//! synchronises delivers nothing at all while appearing to work perfectly at
//! the `read` level. Every packet here ends with one.
//!
//! **evdev's `REL_Y` grows downward.** `dev::mouse::apply` takes a `dy` that
//! grows *upward* -- it computes `s.y - dy` -- so the sign is flipped on the
//! way in. Getting this wrong gives a game whose mouse look is inverted, which
//! reads as a preference somebody forgot to expose rather than as a bug.
//!
//! **A Linux keycode is a set-1 scancode, for the main block only.** That is
//! historical rather than lucky: the keycodes were assigned to match the AT
//! scancodes, so `KEY_ESC` is 1 and `KEY_A` is 30 exactly as the wire does.
//! It stops holding at the `E0`-prefixed keys, which have no scancode small
//! enough to be their own keycode, so those need the table below and only
//! those do.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, Ordering};

pub const EV_SYN: u16 = 0;
pub const EV_KEY: u16 = 1;
pub const EV_REL: u16 = 2;

pub const SYN_REPORT: u16 = 0;
/// What a reader is told when it fell behind and the ring overwrote events it
/// had not taken. Linux emits this rather than silently skipping, because a
/// program holding a shift key whose release it never saw is stuck.
pub const SYN_DROPPED: u16 = 3;

pub const REL_X: u16 = 0;
pub const REL_Y: u16 = 1;
pub const REL_WHEEL: u16 = 8;

pub const BTN_LEFT: u16 = 0x110;
pub const BTN_RIGHT: u16 = 0x111;
pub const BTN_MIDDLE: u16 = 0x112;

/// `struct input_event` on x86-64: two 8-byte fields of `timeval`, then two
/// `__u16` and one `__s32`.
///
/// An ABI rather than a choice, the same argument `struct stat` makes. A
/// reader walks the stream by this stride, so a structure one field short
/// does not deliver less, it delivers garbage from the second record on.
pub const EVENT_LEN: usize = 24;

/// Which device a descriptor is reading.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dev {
    Keyboard,
    Pointer,
}

#[derive(Clone, Copy)]
pub struct Ev {
    pub kind: u16,
    pub code: u16,
    pub value: i32,
    pub us: u64,
}

const CAP: usize = 256;

/// One device's events.
///
/// Written from an interrupt and read from a syscall, on one core, which is
/// the single-producer case the keyboard's own ring already relies on: the
/// entry is filled first and `head` is bumped afterwards with a release, so a
/// reader that sees the new head sees the whole entry behind it.
struct Ring {
    buf: UnsafeCell<[Ev; CAP]>,
    head: AtomicU64,
}

unsafe impl Sync for Ring {}

impl Ring {
    const fn new() -> Self {
        Ring {
            buf: UnsafeCell::new(
                [Ev { kind: 0, code: 0, value: 0, us: 0 }; CAP],
            ),
            head: AtomicU64::new(0),
        }
    }

    fn push(&self, kind: u16, code: u16, value: i32, us: u64) {
        let h = self.head.load(Ordering::Relaxed);
        unsafe { (*self.buf.get())[(h as usize) % CAP] = Ev { kind, code, value, us } };
        self.head.store(h + 1, Ordering::Release);
    }

    fn at(&self, seq: u64) -> Ev {
        unsafe { (*self.buf.get())[(seq as usize) % CAP] }
    }
}

static KEYS: Ring = Ring::new();
static POINTER: Ring = Ring::new();

fn ring(d: Dev) -> &'static Ring {
    match d {
        Dev::Keyboard => &KEYS,
        Dev::Pointer => &POINTER,
    }
}

/// Microseconds since boot, or zero when the TSC was never calibrated.
///
/// Zero rather than a guess, which is the choice `port::clock` makes and gives
/// the reason for. It costs double-click detection, which compares two
/// readings and would see no time pass; a fabricated clock costs the same
/// thing while looking like it works.
fn now_us() -> u64 {
    let per = crate::time::tsc_mhz();
    if per == 0 {
        return 0;
    }
    crate::time::rdtsc() / per
}

/// Where a freshly opened descriptor starts.
///
/// The current head, so a program opening the keyboard does not immediately
/// receive every key pressed since boot. That is what Linux does and it is not
/// a nicety: a game replaying an hour of keystrokes at startup would look like
/// a possessed machine rather than a bug in this file.
pub fn now_at(d: Dev) -> u64 {
    ring(d).head.load(Ordering::Acquire)
}

pub fn pending(d: Dev, cursor: u64) -> bool {
    ring(d).head.load(Ordering::Acquire) > cursor
}

/// A key transition, called from the keyboard's decoder for every scancode.
///
/// `raw` carries the `E0` prefix in its high byte, because a set-1 make code
/// alone cannot tell the left control key from the right one and a game that
/// binds them separately would see the same key twice.
pub fn key(raw: u16, down: bool) {
    let Some(code) = keycode(raw) else { return };
    let us = now_us();
    // No key repeat. Linux's `atkbd` synthesises repeats from the hardware's
    // typematic, and a game reading raw events wants held keys rather than a
    // stutter of presses; anything that does want repeat asks for `EV_REP`,
    // which this device does not advertise.
    KEYS.push(EV_KEY, code, if down { 1 } else { 0 }, us);
    KEYS.push(EV_SYN, SYN_REPORT, 0, us);
}

/// Pointer motion and buttons, called from the one place PS/2 and USB HID
/// converge.
///
/// Takes the buttons both ways round because evdev reports *transitions* and
/// `apply` is handed state: a packet arrives for every movement whether or not
/// a button changed, and emitting a press per packet would give a program a
/// hundred clicks a second.
pub fn pointer(dx: i32, dy: i32, wheel: i32, left: bool, right: bool, was_left: bool, was_right: bool) {
    let mut any = false;
    let us = now_us();
    if dx != 0 {
        POINTER.push(EV_REL, REL_X, dx, us);
        any = true;
    }
    if dy != 0 {
        // **Flipped, and this is the one to get right.** `apply` computes
        // `s.y - dy`, so its `dy` grows upward; evdev's `REL_Y` grows down.
        POINTER.push(EV_REL, REL_Y, -dy, us);
        any = true;
    }
    if wheel != 0 {
        POINTER.push(EV_REL, REL_WHEEL, wheel, us);
        any = true;
    }
    if left != was_left {
        POINTER.push(EV_KEY, BTN_LEFT, if left { 1 } else { 0 }, us);
        any = true;
    }
    if right != was_right {
        POINTER.push(EV_KEY, BTN_RIGHT, if right { 1 } else { 0 }, us);
        any = true;
    }
    if any {
        POINTER.push(EV_SYN, SYN_REPORT, 0, us);
    }
}

/// Set-1 scancode to Linux keycode.
///
/// The main block is the identity, which is a fact about how the keycodes were
/// assigned rather than a coincidence worth hiding behind a table: `KEY_ESC`
/// is 1 and the escape key sends 0x01. Only the `E0` block needs mapping,
/// because those keys arrive as a prefix plus a code that already means
/// something else.
fn keycode(raw: u16) -> Option<u16> {
    if raw & 0xFF00 == 0 {
        let c = raw & 0x7F;
        // 0x59 is the last code in the identity range (KEY_KPEQUAL); above it
        // the two numbering schemes diverge and this has no table.
        return (c >= 1 && c <= 0x59).then_some(c);
    }
    let c = (raw & 0x7F) as u8;
    Some(match c {
        0x1C => 96,  // KEY_KPENTER
        0x1D => 97,  // KEY_RIGHTCTRL
        0x35 => 98,  // KEY_KPSLASH
        0x38 => 100, // KEY_RIGHTALT
        0x47 => 102, // KEY_HOME
        0x48 => 103, // KEY_UP
        0x49 => 104, // KEY_PAGEUP
        0x4B => 105, // KEY_LEFT
        0x4D => 106, // KEY_RIGHT
        0x4F => 107, // KEY_END
        0x50 => 108, // KEY_DOWN
        0x51 => 109, // KEY_PAGEDOWN
        0x52 => 110, // KEY_INSERT
        0x53 => 111, // KEY_DELETE
        0x5B => 125, // KEY_LEFTMETA
        0x5C => 126, // KEY_RIGHTMETA
        0x5D => 127, // KEY_COMPOSE
        _ => return None,
    })
}

/// Fill `out` with whole events from `cursor`, advancing it.
///
/// **Whole ones only.** A reader walks the stream by a fixed stride, so half a
/// record at the end of a buffer desynchronises everything after it; Linux
/// answers `EINVAL` for a buffer too small to hold one, which the caller does.
pub fn read(d: Dev, cursor: &mut u64, out: &mut [u8]) -> usize {
    let r = ring(d);
    let head = r.head.load(Ordering::Acquire);
    if *cursor >= head {
        return 0;
    }
    // Fell behind far enough that the ring overwrote what was owed. Jump to
    // the oldest surviving event and say so, rather than delivering a
    // plausible stream missing the middle of it: a release nobody saw leaves a
    // key held forever.
    let mut dropped = false;
    if head - *cursor > CAP as u64 {
        *cursor = head - CAP as u64;
        dropped = true;
    }
    let mut n = 0;
    if dropped && out.len() >= EVENT_LEN {
        let us = now_us();
        put(out, &mut n, &Ev { kind: EV_SYN, code: SYN_DROPPED, value: 0, us });
    }
    while *cursor < head && n + EVENT_LEN <= out.len() {
        let e = r.at(*cursor);
        put(out, &mut n, &e);
        *cursor += 1;
    }
    n
}

/// One record, in the byte order a `struct input_event` has.
///
/// The `timeval` is two 8-byte fields rather than a 16-byte one, because that
/// is what it is: seconds and microseconds, each a `long`. Packing it as a
/// single number would put the microseconds where nothing reads them.
fn put(out: &mut [u8], n: &mut usize, e: &Ev) {
    let b = &mut out[*n..*n + EVENT_LEN];
    b[0..8].copy_from_slice(&(e.us / 1_000_000).to_le_bytes());
    b[8..16].copy_from_slice(&(e.us % 1_000_000).to_le_bytes());
    b[16..18].copy_from_slice(&e.kind.to_le_bytes());
    b[18..20].copy_from_slice(&e.code.to_le_bytes());
    b[20..24].copy_from_slice(&e.value.to_le_bytes());
    *n += EVENT_LEN;
}

/// One record back out of a buffer, for the claims below.
fn at_of(buf: &[u8], i: usize) -> (u16, u16, i32) {
    (
        u16::from_le_bytes([buf[i * 24 + 16], buf[i * 24 + 17]]),
        u16::from_le_bytes([buf[i * 24 + 18], buf[i * 24 + 19]]),
        i32::from_le_bytes([
            buf[i * 24 + 20],
            buf[i * 24 + 21],
            buf[i * 24 + 22],
            buf[i * 24 + 23],
        ]),
    )
}

// ---------------------------------------------------------------- the ioctls

/// `EVIOCGVERSION`, and the value is `EV_VERSION`.
pub const EV_VERSION: u32 = 0x01_0001;

/// The bottom sixteen bits of an evdev request, which is type and number with
/// the direction and the size masked away.
///
/// Masked rather than matched whole because the length-carrying requests
/// encode it in bits 16..30, so `EVIOCGNAME(64)` and `EVIOCGNAME(256)` are
/// different numbers naming one thing.
pub fn request(req: u64) -> u16 {
    (req & 0xFFFF) as u16
}

pub fn request_len(req: u64) -> usize {
    ((req >> 16) & 0x3FFF) as usize
}

pub const EVIOC_VERSION: u16 = 0x4501;
pub const EVIOC_ID: u16 = 0x4502;
pub const EVIOC_NAME: u16 = 0x4506;
pub const EVIOC_PHYS: u16 = 0x4507;
pub const EVIOC_UNIQ: u16 = 0x4508;
pub const EVIOC_PROP: u16 = 0x4509;
pub const EVIOC_KEYSTATE: u16 = 0x4518;
/// `EVIOCGBIT(ev, len)` is this plus the event type, so the whole family is
/// one contiguous run of sixteen numbers.
pub const EVIOC_BIT: u16 = 0x4520;
pub const EVIOC_GRAB: u16 = 0x4590;
pub const EVIOC_SETCLOCK: u16 = 0x45A0;

pub fn name(d: Dev) -> &'static str {
    match d {
        Dev::Keyboard => "GLaDOS AT keyboard",
        Dev::Pointer => "GLaDOS pointer",
    }
}

/// `struct input_id`: bus, vendor, product, version.
///
/// `BUS_I8042` for both, which is true of what this machine boots with and is
/// a small lie once a USB device is the one feeding them. It is the bus a
/// program is least likely to branch on, and the alternative -- reporting
/// `BUS_VIRTUAL` -- makes libinput treat the device as a software emulation
/// and apply none of its pointer acceleration.
pub fn ident(d: Dev) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0..2].copy_from_slice(&0x11u16.to_le_bytes()); // BUS_I8042
    b[2..4].copy_from_slice(&0x0001u16.to_le_bytes());
    b[4..6].copy_from_slice(&(match d {
        Dev::Keyboard => 0x0001u16,
        Dev::Pointer => 0x0002u16,
    })
    .to_le_bytes());
    b[6..8].copy_from_slice(&0x0001u16.to_le_bytes());
    b
}

fn set(bits: &mut [u8], n: usize) {
    if n / 8 < bits.len() {
        bits[n / 8] |= 1 << (n % 8);
    }
}

/// The capability bitmap for one event type, or for the types themselves.
///
/// **This is the interface.** A program classifies a device by reading these
/// and nothing else: a device advertising `EV_REL` with `REL_X` and `REL_Y`
/// plus `BTN_LEFT` is a mouse, and one advertising a spread of `EV_KEY` codes
/// is a keyboard. Advertise the wrong set and the device is opened, read
/// successfully, and ignored -- which looks exactly like input that does not
/// arrive.
pub fn bits(d: Dev, ev: u16, out: &mut [u8]) {
    for b in out.iter_mut() {
        *b = 0;
    }
    match (d, ev) {
        // Which event types exist at all.
        (Dev::Keyboard, 0) => {
            set(out, EV_SYN as usize);
            set(out, EV_KEY as usize);
        }
        (Dev::Pointer, 0) => {
            set(out, EV_SYN as usize);
            set(out, EV_KEY as usize);
            set(out, EV_REL as usize);
        }
        // Every keycode this decoder can produce, which is the identity block
        // plus the seventeen extended ones. Derived from `keycode` rather than
        // listed again, so a key added there cannot go unadvertised.
        (Dev::Keyboard, EV_KEY) => {
            for raw in 1u16..=0x59 {
                if let Some(c) = keycode(raw) {
                    set(out, c as usize);
                }
            }
            for raw in 0u16..=0x7F {
                if let Some(c) = keycode(0xE000 | raw) {
                    set(out, c as usize);
                }
            }
        }
        (Dev::Pointer, EV_KEY) => {
            set(out, BTN_LEFT as usize);
            set(out, BTN_RIGHT as usize);
            // Advertised and never sent: `dev::mouse` tracks two buttons, and
            // a program that lays a binding on the middle one gets a control
            // nothing can reach. Named rather than hidden, because the
            // alternative is a device whose button count silently differs from
            // every other mouse in the world.
            set(out, BTN_MIDDLE as usize);
        }
        (Dev::Pointer, EV_REL) => {
            set(out, REL_X as usize);
            set(out, REL_Y as usize);
            set(out, REL_WHEEL as usize);
        }
        _ => {}
    }
}

/// Which keys are held right now, in the shape `EVIOCGKEY` answers.
///
/// Read from the keyboard's own held-key state rather than replayed from the
/// ring, because a program asks this exactly once, at startup, to find out
/// what is already down -- and the ring by then starts at the moment the
/// device was opened.
pub fn key_state(d: Dev, out: &mut [u8]) {
    for b in out.iter_mut() {
        *b = 0;
    }
    if d != Dev::Keyboard {
        return;
    }
    for raw in 1u16..=0x59 {
        if crate::dev::kbd::is_down(raw as u8) {
            if let Some(c) = keycode(raw) {
                set(out, c as usize);
            }
        }
    }
}

// ------------------------------------------------------- driving it headlessly

/// A scheduled input event, because nothing else can deliver one to a running
/// guest.
///
/// **A guest holds the machine while it runs.** `drive.py` sends the next
/// command when it sees a prompt, so anything a script types arrives before
/// the guest starts or after it has gone -- and a device opens at the present,
/// so events pushed beforehand are events the guest was never meant to see.
/// Without this, evdev is a subsystem that can be built and never exercised,
/// which by this tree's standard is a subsystem that does not work.
///
/// It is the same answer `win keys` and `doom play`'s timed script are, for
/// the same reason, so it borrows their spelling: `shift@200 -shift@400`.
///
/// Delivery is from the **timer interrupt**, which is the only thing that gets
/// a turn while a guest is running, and it is also where a real keyboard
/// interrupt would arrive from as far as everything downstream is concerned.
const FEED_MAX: usize = 64;

/// offset in ticks (32) | raw code (16) | down (1). Zero means empty.
///
/// An **offset from when the guest starts**, not from when the script was
/// armed, and the difference is the whole reason this works. The first version
/// stored absolute deadlines and every step of a two-step script fired before
/// the guest existed: `drive.py` waits for a prompt between commands, so more
/// than a second passes between `linux feed` and `linux run`, and a device
/// opens at the present so the events were delivered to nobody. The guest then
/// blocked forever on a keyboard whose keystrokes were already in the past.
static FEED: [AtomicU64; FEED_MAX] = [const { AtomicU64::new(0) }; FEED_MAX];
static ARMED: AtomicU64 = AtomicU64::new(0);
/// The tick a guest started at, or zero when none has.
static BASE: AtomicU64 = AtomicU64::new(0);
/// How many ticks `service` has been given and how many events it delivered.
///
/// Counters rather than a print, because `service` runs inside an interrupt
/// gate and printing from one takes a #GP on this machine -- which is a real
/// bug the console has and the reason the fault reporter writes serial first.
static TICKED: AtomicU64 = AtomicU64::new(0);
static FIRED: AtomicU64 = AtomicU64::new(0);

/// Scancodes that put nothing in the shell's own character ring.
///
/// The modifiers return early in `kbd::decode` before it pushes a character,
/// so feeding them exercises the real decoder without leaving a stray line for
/// the shell to execute as the next command. The arrows do push, and are here
/// because a program that only ever sees modifiers is not being tested; a
/// script using them costs one junk line at the prompt afterwards.
fn named(k: &str) -> Option<u16> {
    Some(match k {
        "shift" => 0x2A,
        "rshift" => 0x36,
        "ctrl" => 0x1D,
        "rctrl" => 0xE01D,
        "up" => 0xE048,
        "down" => 0xE050,
        "left" => 0xE04B,
        "right" => 0xE04D,
        _ => return None,
    })
}

/// The moment a script is measured from, called when a guest is installed.
///
/// Clearing it at teardown is deliberate: a script belongs to one run, and one
/// left armed would fire into whatever ran next, which is the kind of spooky
/// action that makes a harness untrustworthy.
pub fn start(now: u64) {
    BASE.store(now.max(1), Ordering::Release);
}

/// Stop measuring, without discarding what has not fired.
///
/// **Only the clock, and that distinction cost a run to find.** The first
/// version cleared the script too, on the reasoning that a script belongs to
/// one run -- and it was called from `teardown`, which `install` calls *at
/// the head of itself* to abandon any previous space. So the sequence was
/// arm, then install, then teardown eating the script, then start, and the
/// guest blocked forever on a keyboard whose script had been swept a
/// microsecond before it began.
///
/// "At teardown" is not "at the end of a run" in this module, and anything
/// else that hangs work off it has the same trap waiting.
///
/// What is left over stays armed for the next guest, which is what the
/// command says it does; `arm` clears before it arms, so two scripts still
/// cannot interleave.
pub fn stop() {
    BASE.store(0, Ordering::Release);
}

/// Arm one script. Answers how many events were scheduled, or why not.
pub fn arm(spec: &str) -> Result<usize, &'static str> {
    for slot in FEED.iter() {
        slot.store(0, Ordering::Relaxed);
    }
    ARMED.store(0, Ordering::Release);
    BASE.store(0, Ordering::Release);
    let hz = crate::TIMER_HZ as u64;
    let mut n = 0;
    for word in spec.split_whitespace() {
        let (name, ms) = word.split_once('@').ok_or("each step is name@ms")?;
        let ms: u64 = ms.parse().map_err(|_| "the time after @ is milliseconds")?;
        let (name, down) = match name.strip_prefix('-') {
            Some(rest) => (rest, false),
            None => (name, true),
        };
        let raw = named(name).ok_or("no such key: try shift, ctrl, or an arrow")?;
        if n >= FEED_MAX {
            return Err("too many steps");
        }
        // Rounded up rather than down, so a step at 10 ms on a 100 Hz clock
        // lands on the next tick instead of on the current one, which would
        // fire every early step at once.
        let at = ms.div_ceil(1000 / hz.max(1)).max(1);
        FEED[n].store((at << 32) | ((raw as u64) << 16) | u64::from(down), Ordering::Relaxed);
        n += 1;
    }
    ARMED.store(n as u64, Ordering::Release);
    Ok(n)
}

/// Deliver whatever is due. Called from the timer interrupt every tick.
///
/// Through `kbd::inject_scancode` rather than straight into the ring, which is
/// the whole point: it exercises the real decoder, the extended-code state
/// machine and the two call sites in `kbd.rs`, so what is driven here is the
/// path a real key takes and not a private back door that happens to agree.
pub fn service(now: u64) {
    TICKED.fetch_add(1, Ordering::Relaxed);
    let base = BASE.load(Ordering::Acquire);
    if base == 0 || ARMED.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut left = 0;
    for slot in FEED.iter() {
        let v = slot.load(Ordering::Relaxed);
        if v == 0 {
            continue;
        }
        if base + (v >> 32) > now {
            left += 1;
            continue;
        }
        slot.store(0, Ordering::Relaxed);
        FIRED.fetch_add(1, Ordering::Relaxed);
        let raw = ((v >> 16) & 0xFFFF) as u16;
        let down = v & 1 != 0;
        let release = if down { 0 } else { 0x80 };
        if raw & 0xFF00 != 0 {
            crate::dev::kbd::inject_scancode(0xE0);
        }
        crate::dev::kbd::inject_scancode((raw & 0xFF) as u8 | release);
    }
    ARMED.store(left, Ordering::Release);
}

pub fn armed() -> u64 {
    ARMED.load(Ordering::Acquire)
}

/// Whether a guest is being measured against right now.
pub fn fired_yet() -> bool {
    BASE.load(Ordering::Acquire) != 0
}

/// (ticks seen, events delivered, the tick a guest started at).
pub fn feed_stats() -> (u64, u64, u64) {
    (
        TICKED.load(Ordering::Relaxed),
        FIRED.load(Ordering::Relaxed),
        BASE.load(Ordering::Acquire),
    )
}

/// What `diag linux` asks of the input devices.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out = alloc::vec::Vec::new();

    out.push((
        "a struct input_event is 24 bytes, which is the stride a reader walks by",
        EVENT_LEN == 24,
    ));
    // The historical identity, asserted rather than assumed. If it were false
    // every letter would arrive as a different letter.
    out.push((
        "the main scancode block is the keycode block: esc is 1, a is 30, space is 57",
        keycode(0x01) == Some(1) && keycode(0x1E) == Some(30) && keycode(0x39) == Some(57),
    ));
    out.push((
        "and the extended block is not, so the arrows need the table they have",
        keycode(0xE048) == Some(103)
            && keycode(0xE04D) == Some(106)
            && keycode(0x48) == Some(0x48),
    ));
    out.push((
        "the two control keys are told apart, which one scancode alone cannot do",
        keycode(0x1D) == Some(29) && keycode(0xE01D) == Some(97),
    ));
    out.push((
        "a code outside both blocks is dropped rather than mapped to something near it",
        keycode(0x00).is_none() && keycode(0x7A).is_none() && keycode(0xE07A).is_none(),
    ));

    // The request decoder, against numbers taken from the header. Getting the
    // mask wrong makes every length-carrying request unrecognised, which is a
    // device that answers its version and nothing else.
    out.push((
        "a request is matched on type and number, with direction and size masked away",
        request(0x8004_4501) == EVIOC_VERSION
            && request(0x8100_4506) == EVIOC_NAME
            && request(0x8020_4520) == EVIOC_BIT
            && request_len(0x8100_4506) == 0x100,
    ));

    let mut b = [0u8; 8];
    bits(Dev::Pointer, 0, &mut b);
    let pointer_types = b[0];
    bits(Dev::Keyboard, 0, &mut b);
    out.push((
        "a pointer advertises EV_REL and a keyboard does not, which is how each is known",
        pointer_types & (1 << EV_REL) != 0 && b[0] & (1 << EV_REL) == 0 && b[0] & (1 << EV_KEY) != 0,
    ));

    let mut rel = [0u8; 2];
    bits(Dev::Pointer, EV_REL, &mut rel);
    out.push((
        "and it advertises both axes and a wheel, or nothing opens it as a mouse",
        rel[0] & (1 << REL_X) != 0 && rel[0] & (1 << REL_Y) != 0 && rel[1] & 1 != 0,
    ));

    let mut keys = [0u8; 96];
    bits(Dev::Keyboard, EV_KEY, &mut keys);
    let held = keys.iter().map(|b| b.count_ones()).sum::<u32>();
    out.push((
        "the keyboard advertises every code its decoder can produce, and no others",
        held == 106 && keys[30 / 8] & (1 << (30 % 8)) != 0 && keys[103 / 8] & (1 << (103 % 8)) != 0,
    ));
    let mut btn = [0u8; 96];
    bits(Dev::Pointer, EV_KEY, &mut btn);
    out.push((
        "a pointer advertises its buttons where a keyboard advertises none of them",
        btn[BTN_LEFT as usize / 8] & (1 << (BTN_LEFT % 8)) != 0
            && keys[BTN_LEFT as usize / 8] == 0,
    ));

    // The ring, driven through the same entry the interrupt uses. A read
    // starting where the device was opened is what stops a program receiving
    // an hour of keystrokes the moment it starts.
    let start = now_at(Dev::Keyboard);
    key(0x1E, true);
    key(0x1E, false);
    let mut cur = start;
    let mut buf = [0u8; EVENT_LEN * 8];
    let n = read(Dev::Keyboard, &mut cur, &mut buf);
    out.push((
        "a press and a release arrive as four events, each pair closed by a SYN",
        n == EVENT_LEN * 4
            && at_of(&buf, 0) == (EV_KEY, 30, 1)
            && at_of(&buf, 1) == (EV_SYN, SYN_REPORT, 0)
            && at_of(&buf, 2) == (EV_KEY, 30, 0)
            && at_of(&buf, 3) == (EV_SYN, SYN_REPORT, 0),
    ));
    out.push((
        "and the cursor is left past them, so a second read answers nothing",
        cur == start + 4 && read(Dev::Keyboard, &mut cur, &mut buf) == 0,
    ));

    // Whole events only. A buffer that fits two and a half is a reader that
    // would resynchronise on garbage.
    let mut cur = start;
    let n = read(Dev::Keyboard, &mut cur, &mut buf[..EVENT_LEN * 2 + 7]);
    out.push((
        "a buffer that fits two and a half events is given two",
        n == EVENT_LEN * 2 && cur == start + 2,
    ));

    // The sign, which is the one thing a claim can settle before a mouse is in
    // anybody's hand: evdev grows downward and `apply` grows upward.
    let p0 = now_at(Dev::Pointer);
    pointer(3, 5, 0, false, false, false, false);
    let mut pc = p0;
    let n = read(Dev::Pointer, &mut pc, &mut buf);
    out.push((
        "pointer motion is x then y then a SYN, with y negated for evdev's axis",
        n == EVENT_LEN * 3
            && at_of(&buf, 0) == (EV_REL, REL_X, 3)
            && at_of(&buf, 1) == (EV_REL, REL_Y, -5)
            && at_of(&buf, 2) == (EV_SYN, SYN_REPORT, 0),
    ));
    // A packet arrives for every movement, and buttons are reported as
    // transitions, so unchanged buttons must produce nothing at all.
    let p1 = now_at(Dev::Pointer);
    pointer(0, 0, 0, true, false, true, false);
    out.push((
        "a packet whose buttons did not change produces no event and no SYN",
        now_at(Dev::Pointer) == p1,
    ));
    let mut pc = p1;
    pointer(0, 0, 0, true, false, false, false);
    let n = read(Dev::Pointer, &mut pc, &mut buf);
    out.push((
        "and a button that did change produces one press, closed by a SYN",
        n == EVENT_LEN * 2 && at_of(&buf, 0) == (EV_KEY, BTN_LEFT, 1),
    ));

    // Falling behind. The dropped marker is what stops a lost release leaving
    // a key held forever, which is the failure that outlives the program.
    let p2 = now_at(Dev::Keyboard);
    for _ in 0..(CAP + 8) {
        key(0x1E, true);
    }
    let mut cur = p2;
    let n = read(Dev::Keyboard, &mut cur, &mut buf);
    out.push((
        "a reader that fell behind the ring is told so rather than handed a gap",
        n >= EVENT_LEN && at_of(&buf, 0) == (EV_SYN, SYN_DROPPED, 0),
    ));

    // The scheduler that makes any of this drivable. Nothing is delivered
    // here, only scheduled and then swept, because delivery goes through the
    // real decoder and would leave the machine's held-key state changed.
    out.push((
        "a feed script is parsed as name@ms, with a leading minus for a release",
        arm("shift@10 -shift@20 ctrl@30") == Ok(3) && arm("") == Ok(0),
    ));
    out.push((
        "and a step it cannot spell is refused rather than silently dropped",
        arm("nosuchkey@10").is_err()
            && arm("shift").is_err()
            && arm("shift@soon").is_err(),
    ));
    // The bug this arrangement exists to prevent, asserted rather than
    // remembered: a script armed at the prompt must not fire until a guest is
    // there to receive it, because a device opens at the present and an event
    // delivered early is one nobody sees.
    let _ = arm("shift@1000 -shift@2000");
    let before = armed();
    service(1_000_000);
    out.push((
        "nothing fires until a guest has started, however much time has passed",
        before == 2 && armed() == 2,
    ));
    self::start(0);
    service(0);
    out.push((
        "and nothing fires before its own offset once one has",
        armed() == 2,
    ));
    // The bug that cost a run. `teardown` calls this, and `install` calls
    // `teardown` at the head of itself, so a `stop` that discarded the script
    // would discard it between the arming and the guest that was armed for.
    stop();
    out.push((
        "stopping stops the clock and keeps the script, since install tears down first",
        armed() == 2 && !fired_yet(),
    ));
    // Cleared explicitly rather than delivered: firing them would put two
    // scancodes through the real decoder and leave shift held on the live
    // machine for whoever types next.
    let _ = arm("");
    out.push((
        "and arming again clears what was armed, so two scripts cannot interleave",
        armed() == 0,
    ));
    out
}
