//! A stream somebody keeps writing to, where `Hda::play` takes a whole sound.
//!
//! The controller loops one buffer forever, so a stream is that buffer used
//! as a ring: the writer stays ahead of the controller's read position
//! (`LPIB`) and is held back from lapping it. Two things are silent when
//! wrong and both are handled here rather than by whoever writes:
//!
//! - **What was played is zeroed behind the controller.** The engine does not
//!   stop at the writer -- it has no idea where the writer is -- so a writer
//!   that stalls for a moment has the ring's old contents played again, which
//!   is heard as a buzz the length of the stall. Zeroed, a stall is silence.
//! - **A writer that fell behind starts again just ahead of the controller**
//!   rather than writing into what is being played, which would be heard as
//!   the newest sound arriving a whole ring late.
//!
//! The ring is 64 KiB: about 370 ms at 44.1 kHz stereo 16-bit, so a writer
//! may run late by a third of a second before anything is heard.

use crate::sync::Spin;

const RING: usize = 64 * 1024;
/// How far ahead of the controller a restart begins, in bytes: ten
/// milliseconds at 44.1 kHz stereo, so a restart is not itself an underrun.
const LEAD: u64 = 1792;
/// How close to lapping the controller a writer may come.
const GUARD: u64 = 1024;

struct Sink {
    /// Bytes written, ever.
    w: u64,
    /// Bytes the controller has read, ever, unwrapped from `LPIB`.
    hw: u64,
    last_pos: u32,
    last_tsc: u64,
    rate: u32,
    frame: u32,
}

static SINK: Spin<Option<Sink>> = Spin::new(None);

/// Start a stream of interleaved little-endian samples. `false` is no
/// controller, or a format it will not take.
pub fn open(rate: u32, channels: u8, bits: u8) -> bool {
    close();
    if super::with(|_| ()).is_none() {
        let e = crate::voice::ecam();
        if e == 0 || !super::bring_up(e, None, false) {
            return false;
        }
    }
    let Some(fmt) = super::verb::format(rate, bits, channels) else { return false };
    let silence = alloc::vec![0u8; RING];
    if super::with(|h| h.play(&silence, fmt).is_ok()) != Some(true) {
        return false;
    }
    let pos = super::with(|h| h.position()).flatten().unwrap_or(0);
    *SINK.lock_irq() = Some(Sink {
        w: pos as u64 + LEAD,
        hw: pos as u64,
        last_pos: pos,
        last_tsc: unsafe { core::arch::x86_64::_rdtsc() },
        rate,
        frame: channels as u32 * bits as u32 / 8,
    });
    true
}

pub fn is_open() -> bool {
    SINK.lock_irq().is_some()
}

pub fn close() {
    if SINK.lock_irq().take().is_some() {
        let _ = super::with(|h| h.stop());
    }
}

/// Bring the controller's position up to date: advance `hw`, zero what was
/// played, and restart a writer that fell behind. Answers the ring.
fn service<'a>(s: &mut Sink, h: &'a mut super::Hda) -> Option<&'a mut [u8]> {
    let pos = h.position()? as u64 % RING as u64;
    let now = unsafe { core::arch::x86_64::_rdtsc() };
    // The ring's period in TSC cycles. Asked less often than that, the
    // controller may have gone round any number of times, so the writer is
    // treated as having fallen behind.
    let lap = RING as u64 * 1_000_000 / (s.rate as u64 * s.frame as u64).max(1) * crate::time::tsc_mhz().max(1);
    let delta = (pos + RING as u64 - s.last_pos as u64) % RING as u64;
    let stale = now.wrapping_sub(s.last_tsc) > lap;
    let old = s.hw;
    s.hw += delta;
    s.last_pos = pos as u32;
    s.last_tsc = now;
    let ring = h.ring()?;
    // Zero what was played since last time.
    let mut z = old;
    while z < s.hw {
        let i = (z % RING as u64) as usize;
        let n = ((s.hw - z) as usize).min(RING - i);
        ring[i..i + n].fill(0);
        z += n as u64;
    }
    if stale || s.w < s.hw + 128 {
        if stale {
            ring.fill(0);
        }
        s.w = s.hw + LEAD;
    }
    Some(ring)
}

/// Take as much of `pcm` as there is room for, and answer how much that was.
/// Zero means the ring is full; the caller waits and tries again.
pub fn write(pcm: &[u8]) -> usize {
    let mut g = SINK.lock_irq();
    let Some(s) = g.as_mut() else { return 0 };
    let r = super::with(|h| {
        let ring = service(s, h)?;
        let ahead = s.w - s.hw;
        let room = (RING as u64).saturating_sub(GUARD + ahead) as usize;
        let take = room.min(pcm.len()) / s.frame as usize * s.frame as usize;
        let mut done = 0;
        while done < take {
            let i = ((s.w + done as u64) % RING as u64) as usize;
            let n = (take - done).min(RING - i);
            ring[i..i + n].copy_from_slice(&pcm[done..done + n]);
            done += n;
        }
        s.w += take as u64;
        Some(take)
    });
    r.flatten().unwrap_or(0)
}

/// Bytes written and not yet played -- what `SNDCTL_DSP_GETODELAY` and ALSA's
/// delay ask for. **Reads the controller**, and it has to: a caller waiting
/// for room asks this rather than writing, and a figure only `write` brought
/// up to date stayed full forever -- the first game to play through it wrote
/// two buffers and waited sixty thousand times for a third.
pub fn queued() -> u64 {
    let mut g = SINK.lock_irq();
    let Some(s) = g.as_mut() else { return 0 };
    let _ = super::with(|h| {
        service(s, h);
    });
    s.w.saturating_sub(s.hw)
}

/// The size of the ring, so a caller can say how much room it has.
pub const fn ring_len() -> usize {
    RING
}
