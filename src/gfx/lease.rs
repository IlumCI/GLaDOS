//! Who owns the screen, when somebody other than the desktop does.
//!
//! **This replaces a bare `AtomicBool`, and the bool is why the screen
//! sometimes broke.** `gfx::exclusive` said "somebody owns the screen" and
//! nothing else: not who, not since when, and not whether they had finished.
//! So it could not be reentered safely (any holder clearing it cleared it for
//! everyone), it could not be forgotten safely (a guest that died holding
//! `/dev/fb0` left it to teardown to remember), and the three programs that
//! drew on the aperture without it -- the editor, the Aiksi `pixel`/`rect`/
//! `text` builtins and `video bars` -- never set it at all. Each left the
//! compositor's shadow describing a desktop that was no longer on screen, and
//! `present` then repainted nothing over the program's leftovers.
//!
//! A lease is an RAII guard. Taking it records the owner; dropping it gives
//! the screen back *and* forgets the shadow, so the next frame rewrites every
//! row -- the one step every caller used to have to remember and three of
//! them did not. It is reentrant by owner rather than by task, because a guest
//! holds the display across many syscalls on whichever of its threads asked,
//! and the splash demo holds it across shell commands.
//!
//! Two kinds of owner, and the difference is whether the compositor's own
//! writes are still wanted. The splash and the miner *draw through* the
//! compositor (`compose::present`), so their presents must go through; every
//! other owner draws on the aperture itself, so the compositor and the console
//! must not write there at all while it holds the screen (`Owner::composes`).
//!
//! The state is atomics rather than a lock, because `force_release` runs on
//! the fault path, where a lock held by the code that faulted is a lock
//! nobody will ever release.

use core::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Owner {
    /// The boot splash, and the `splash hold` demo.
    Splash,
    /// A ported full-screen program, through `port::with_screen`.
    Port,
    /// A Linux guest writing or mapping `/dev/fb0`.
    Guest,
    /// The full-screen text editor.
    Editor,
    /// An Aiksi program drawing with `pixel`, `rect` or `text`.
    Aiksi,
    /// `video bars`, the pixel-format check.
    VideoTest,
    /// The miner image, for the life of the machine.
    Miner,
}

impl Owner {
    fn code(self) -> u8 {
        match self {
            Owner::Splash => 1,
            Owner::Port => 2,
            Owner::Guest => 3,
            Owner::Editor => 4,
            Owner::Aiksi => 5,
            Owner::VideoTest => 6,
            Owner::Miner => 7,
        }
    }

    fn of(code: u8) -> Option<Owner> {
        Some(match code {
            1 => Owner::Splash,
            2 => Owner::Port,
            3 => Owner::Guest,
            4 => Owner::Editor,
            5 => Owner::Aiksi,
            6 => Owner::VideoTest,
            7 => Owner::Miner,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Owner::Splash => "the boot splash",
            Owner::Port => "a full-screen program",
            Owner::Guest => "a Linux guest (/dev/fb0)",
            Owner::Editor => "the editor",
            Owner::Aiksi => "an Aiksi program",
            Owner::VideoTest => "video bars",
            Owner::Miner => "the miner",
        }
    }

    /// Whether this owner draws through the compositor, so its presents (and
    /// the console's flushes) must still reach the screen. Everyone else draws
    /// on the aperture directly and the compositor must keep off it.
    pub fn composes(self) -> bool {
        matches!(self, Owner::Splash | Owner::Miner)
    }
}

/// The holder's code, 0 for nobody.
static HOLDER: AtomicU8 = AtomicU8::new(0);
/// How many leases the holder has outstanding.
static DEPTH: AtomicU32 = AtomicU32::new(0);
/// The task that first took it, for `render` to name. Informational only:
/// reentry is by owner.
static TASK: AtomicUsize = AtomicUsize::new(usize::MAX);
/// Bumped each time the screen goes from free to held, and by
/// `force_release`. A lease remembers the generation it belongs to, so one
/// that was forced away is inert when it is finally dropped instead of
/// releasing whoever holds the screen by then.
static GEN: AtomicU32 = AtomicU32::new(0);

/// The screen, held. Dropping it gives the screen back.
#[must_use = "the screen is released the moment the lease is dropped"]
pub struct Lease {
    gen: u32,
}

/// Take the screen for `owner`. `None` when a different owner holds it --
/// two programs cannot both draw full-screen, and the second is told rather
/// than allowed to paint over the first.
pub fn take(owner: Owner) -> Option<Lease> {
    let code = owner.code();
    let gen = match HOLDER.compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            DEPTH.store(1, Ordering::Release);
            TASK.store(crate::task::current(), Ordering::Release);
            GEN.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
        }
        Err(c) if c == code => {
            DEPTH.fetch_add(1, Ordering::AcqRel);
            GEN.load(Ordering::Acquire)
        }
        Err(_) => return None,
    };
    Some(Lease { gen })
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.gen != GEN.load(Ordering::Acquire) {
            return;
        }
        let prev = DEPTH
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |d| Some(d.saturating_sub(1)))
            .unwrap_or(0);
        if prev <= 1 && HOLDER.swap(0, Ordering::AcqRel) != 0 {
            given_back();
        }
    }
}

/// Whatever was on screen belongs to the departed owner: forget the shadow so
/// the next frame rewrites every row, and ask for that frame.
fn given_back() {
    TASK.store(usize::MAX, Ordering::Release);
    super::compose::invalidate();
    super::render::damage_all();
}

/// Who holds the screen, if anybody.
pub fn holder() -> Option<Owner> {
    Owner::of(HOLDER.load(Ordering::Acquire))
}

/// Whether anybody does. The desktop's periodic painters stand down on this.
pub fn held() -> bool {
    HOLDER.load(Ordering::Acquire) != 0
}

/// Whether the compositor and the console may write the aperture right now:
/// nobody holds the screen, or the holder draws through them.
pub fn compositor_may_write() -> bool {
    holder().map_or(true, Owner::composes)
}

/// The task that took the screen, for diagnostics.
pub fn holder_task() -> Option<usize> {
    let t = TASK.load(Ordering::Acquire);
    (held() && t != usize::MAX).then_some(t)
}

/// Take the screen away from whoever holds it, for a panic or a fault: the
/// console is the only diagnostic left and must be able to reach the screen.
/// Lock-free on purpose (see the module doc). Outstanding `Lease`s become
/// inert, through the generation.
pub fn force_release() {
    GEN.fetch_add(1, Ordering::AcqRel);
    DEPTH.store(0, Ordering::Release);
    if HOLDER.swap(0, Ordering::AcqRel) != 0 {
        given_back();
    }
}

/// `diag lease`: the rules, against the live lease (which must be free).
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    if held() {
        kprintln!("  the screen is held by {} -- run this with nothing full-screen open",
                  holder().map_or("?", Owner::name));
        return false;
    }
    {
        let a = take(Owner::Editor);
        claim("a free screen can be taken", a.is_some());
        claim("the holder is recorded", holder() == Some(Owner::Editor));
        claim("a raw owner keeps the compositor off the aperture", !compositor_may_write());
        claim("a different owner is refused", take(Owner::Aiksi).is_none());
        let b = take(Owner::Editor);
        claim("the same owner may take it again", b.is_some());
        drop(b);
        claim("dropping the inner lease keeps the screen held", held());
        drop(a);
        claim("dropping the outer one gives it back", !held());
    }
    {
        let s = take(Owner::Splash);
        claim("an owner that composes leaves the compositor writing", compositor_may_write());
        drop(s);
    }
    {
        let g = take(Owner::Guest);
        force_release();
        claim("force_release frees a held screen", !held());
        let other = take(Owner::Port);
        claim("and it can be taken again at once", other.is_some());
        drop(g);
        claim("a lease forced away does not release its successor", holder() == Some(Owner::Port));
        drop(other);
        claim("and the successor's own drop does", !held());
    }
    ok
}
