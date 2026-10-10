//! Motion: named curves, springs, and tweens driven by the clock.
//!
//! **Progress comes from time and never from a frame count.** The flourish
//! this replaces advanced one step per composed frame, so its length was a
//! property of how fast the machine composed: a tenth of a second on a quiet
//! desktop, and a second and a half on one that took 250 ms a frame, with the
//! gesture's shape the same either way. A tween here asks the TSC how far
//! along it is, so a slow frame shows the animation further along rather
//! than making it last longer -- the "every frame is the right frame" rule
//! Hyprland and every compositor since Quartz Extreme have kept.
//!
//! **Curves are Hyprland's, by name and number** (`hyprland.conf`'s example
//! config): cubic beziers through (0,0) and (1,1), baked once into 256
//! samples of y against x -- the arrangement `hyprutils` uses, since solving
//! the cubic for t at every sample of every frame is work that never changes.
//! Between samples the table is interpolated, and the ends are exact by
//! construction rather than by the bake's arithmetic: a last frame that lands
//! a hundredth of a pixel short of the target leaves a seam nothing repaints.
//!
//! Springs are closed-form (`spring`): the position of a damped oscillator
//! released from rest at 0 toward 1, as a function of time. No integrator
//! means no step size, and nothing to drift.

use crate::ai::tensor::{cosf, expf, sinf, sqrtf};
use crate::sync::Spin;

/// Microseconds on the TSC, the clock every duration here is measured on.
pub fn now_us() -> u64 {
    crate::time::rdtsc() / crate::time::tsc_mhz().max(1) as u64
}

/// A named easing curve.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Curve {
    Linear,
    /// `0.23, 1, 0.32, 1`: fast out of the gate and long to settle. Windows
    /// and the focus border.
    EaseOutQuint,
    /// `0.65, 0.05, 0.36, 1`.
    EaseInOutCubic,
    /// `0.5, 0.5, 0.75, 1`: nearly linear, eased only at the end. Fades.
    AlmostLinear,
    /// `0.15, 0, 0.1, 1`.
    Quick,
}

const CURVES: usize = 5;
const SAMPLES: usize = 256;

impl Curve {
    fn index(self) -> usize {
        self as usize
    }

    /// The two inner control points.
    fn control(self) -> (f32, f32, f32, f32) {
        match self {
            Curve::Linear => (0.0, 0.0, 1.0, 1.0),
            Curve::EaseOutQuint => (0.23, 1.0, 0.32, 1.0),
            Curve::EaseInOutCubic => (0.65, 0.05, 0.36, 1.0),
            Curve::AlmostLinear => (0.5, 0.5, 0.75, 1.0),
            Curve::Quick => (0.15, 0.0, 0.1, 1.0),
        }
    }
}

/// One coordinate of a cubic bezier from 0 to 1 with inner points `a`, `b`.
fn cubic(a: f32, b: f32, t: f32) -> f32 {
    let u = 1.0 - t;
    3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
}

/// y at each of 256 evenly spaced x. x(t) is monotone for every curve here
/// (both inner x in [0, 1]), so bisection finds t and needs no derivative.
fn bake(c: Curve) -> [f32; SAMPLES] {
    let (x1, y1, x2, y2) = c.control();
    let mut out = [0.0f32; SAMPLES];
    for (i, slot) in out.iter_mut().enumerate() {
        let x = i as f32 / (SAMPLES - 1) as f32;
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if cubic(x1, x2, mid) < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        *slot = cubic(y1, y2, 0.5 * (lo + hi));
    }
    out[0] = 0.0;
    out[SAMPLES - 1] = 1.0;
    out
}

static TABLES: Spin<[Option<[f32; SAMPLES]>; CURVES]> = Spin::new([None; CURVES]);

/// The curve's value at `x` in [0, 1]. Exactly 0 at or below 0 and exactly 1
/// at or above 1, whatever the table says near the ends.
pub fn ease(c: Curve, x: f32) -> f32 {
    if !(x > 0.0) {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if c == Curve::Linear {
        return x;
    }
    let mut t = TABLES.lock_irq();
    let table = t[c.index()].get_or_insert_with(|| bake(c));
    let f = x * (SAMPLES - 1) as f32;
    let i = (f as usize).min(SAMPLES - 2);
    let k = f - i as f32;
    table[i] + (table[i + 1] - table[i]) * k
}

/// A damped spring released from rest at 0, pulled toward 1: its position
/// `t` seconds later. Mass one; `stiffness` and `damping` as Hyprland's
/// spring curves take them. Under-damped springs overshoot 1 and come back,
/// which is what makes them read as springs.
pub fn spring(t: f32, stiffness: f32, damping: f32) -> f32 {
    if !(t > 0.0) {
        return 0.0;
    }
    let w0 = sqrtf(stiffness.max(1e-6));
    let zeta = damping / (2.0 * w0);
    if zeta < 1.0 {
        let wd = w0 * sqrtf(1.0 - zeta * zeta);
        1.0 - expf(-zeta * w0 * t) * (cosf(wd * t) + zeta * w0 / wd * sinf(wd * t))
    } else if zeta == 1.0 {
        1.0 - expf(-w0 * t) * (1.0 + w0 * t)
    } else {
        let s = w0 * sqrtf(zeta * zeta - 1.0);
        let (r1, r2) = (-zeta * w0 + s, -zeta * w0 - s);
        // x = 1 + A e^{r1 t} + B e^{r2 t}, x(0) = 0, x'(0) = 0.
        let a = r2 / (r1 - r2);
        let b = -r1 / (r1 - r2);
        1.0 + a * expf(r1 * t) + b * expf(r2 * t)
    }
}

/// An animation in flight: when it started, how long it lasts, its curve.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tween {
    pub start: u64,
    pub dur_us: u32,
    pub curve: Curve,
}

/// Every duration as a percentage of what it is configured to be: 100 as
/// configured, more for slower, 0 for no motion at all. The Settings window's
/// "animation speed" and the quality governor's last two rungs (half
/// duration, then none) are this one number.
static SPEED: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(100);

pub fn set_speed(percent: u32) {
    SPEED.store(percent.min(10_000), core::sync::atomic::Ordering::Relaxed);
}

/// The setting.
pub fn speed() -> u32 {
    SPEED.load(core::sync::atomic::Ordering::Relaxed)
}

/// What durations are actually scaled by: the setting, halved or zeroed by
/// the quality governor's last two rungs.
pub fn effective_speed() -> u32 {
    if super::quality::shed(4) {
        0
    } else if super::quality::shed(3) {
        speed() / 2
    } else {
        speed()
    }
}

impl Tween {
    /// A tween starting now, `ms` long at 100% speed.
    pub fn new(ms: u32, curve: Curve) -> Tween {
        let us = ms as u64 * 1000 * effective_speed() as u64 / 100;
        Tween { start: now_us(), dur_us: us.min(u32::MAX as u64) as u32, curve }
    }

    /// Raw progress through the duration at `now`, 0 to 1.
    pub fn linear(&self, now: u64) -> f32 {
        if self.dur_us == 0 {
            return 1.0;
        }
        (now.saturating_sub(self.start) as f32 / self.dur_us as f32).min(1.0)
    }

    /// Eased progress at `now`.
    pub fn at(&self, now: u64) -> f32 {
        ease(self.curve, self.linear(now))
    }

    pub fn done(&self, now: u64) -> bool {
        now.saturating_sub(self.start) >= self.dur_us as u64
    }
}

/// How motion is configured: Hyprland's example config, in its own units
/// (tenths of a second become milliseconds) and with its own curves.
pub mod preset {
    use super::Curve;
    /// `windowsIn, 1, 4.1, easeOutQuint, popin 87%`.
    pub const WINDOWS_IN: (u32, Curve) = (410, Curve::EaseOutQuint);
    /// `windowsOut, 1, 1.49, linear, popin 87%`.
    pub const WINDOWS_OUT: (u32, Curve) = (149, Curve::Linear);
    /// `fadeIn, 1, 1.73, almostLinear`.
    pub const FADE_IN: (u32, Curve) = (173, Curve::AlmostLinear);
    /// `fadeOut, 1, 1.46, almostLinear`.
    pub const FADE_OUT: (u32, Curve) = (146, Curve::AlmostLinear);
    /// `windows, 1, 4.79, easeOutQuint`: a window moving to a new place --
    /// the tiling layout's slides, which arrive with phase 13.
    #[allow(dead_code)]
    pub const WINDOWS_MOVE: (u32, Curve) = (479, Curve::EaseOutQuint);
    /// The scale a window pops in from and out to.
    pub const POPIN: f32 = 0.87;
}

/// `diag anim`.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    let all = [Curve::Linear, Curve::EaseOutQuint, Curve::EaseInOutCubic, Curve::AlmostLinear, Curve::Quick];
    claim(
        "every curve is exactly 0 at the start and exactly 1 at the end",
        all.iter().all(|&c| ease(c, 0.0) == 0.0 && ease(c, 1.0) == 1.0 && ease(c, -0.5) == 0.0 && ease(c, 2.0) == 1.0),
    );
    claim(
        "and never moves backwards between them",
        all.iter().all(|&c| (1..=1000).all(|i| ease(c, i as f32 / 1000.0) + 1e-6 >= ease(c, (i - 1) as f32 / 1000.0))),
    );
    // The bake against the cubic solved directly, mid-curve.
    let (x1, y1, x2, y2) = Curve::EaseInOutCubic.control();
    let t = 0.37f32;
    let (x, y) = (cubic(x1, x2, t), cubic(y1, y2, t));
    claim("a baked curve agrees with the bezier it was baked from", (ease(Curve::EaseInOutCubic, x) - y).abs() < 2e-3);
    claim(
        "easeOutQuint is most of the way there by a third of the time",
        ease(Curve::EaseOutQuint, 1.0 / 3.0) > 0.8,
    );
    claim("a spring starts at rest at 0", spring(0.0, 200.0, 20.0) == 0.0);
    claim(
        "an under-damped spring overshoots and settles",
        (1..400).any(|i| spring(i as f32 / 100.0, 200.0, 8.0) > 1.02) && (spring(8.0, 200.0, 8.0) - 1.0).abs() < 1e-3,
    );
    claim(
        "critically and over-damped springs reach 1 without passing it",
        (1..400).all(|i| spring(i as f32 / 100.0, 100.0, 20.0) <= 1.0 + 1e-4 && spring(i as f32 / 100.0, 100.0, 40.0) <= 1.0 + 1e-4)
            && (spring(6.0, 100.0, 20.0) - 1.0).abs() < 1e-3,
    );
    let tw = Tween { start: 1_000_000, dur_us: 400_000, curve: Curve::Linear };
    claim(
        "a tween's progress is a function of the clock and nothing else",
        tw.at(1_000_000) == 0.0 && tw.at(1_200_000) == 0.5 && tw.at(1_400_000) == 1.0 && tw.at(9_000_000) == 1.0,
    );
    claim("and it is done exactly when its duration is up", !tw.done(1_399_999) && tw.done(1_400_000));
    claim("a tween of no duration is already there", Tween { start: 5, dur_us: 0, curve: Curve::Quick }.at(5) == 1.0);
    ok
}
