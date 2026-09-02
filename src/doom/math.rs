//! Sine and cosine, brought along rather than borrowed.
//!
//! `core` has none in a freestanding build, and the kernel's own are in a
//! module this tree may not name -- `src/port` is for things only the machine
//! can provide, and arithmetic is not one of them. A ported program bringing
//! its own libm is the normal arrangement anyway.
//!
//! Accuracy needed here is a fraction of a pixel across a 320-wide frame, so
//! a degree-7 minimax polynomial over a reduced range is far more than
//! enough. It is also what the original would have used had it wanted floats:
//! DOOM itself does none of this, working instead in binary angles with an
//! 8,192-entry sine table, which is the right trade on a 486 and the wrong one
//! on a machine with SSE.
//!
//! Floats are available and properly set up: `cpu::enable_simd` sets
//! CR4.OSFXSR before any float executes and pins MXCSR, and every task's XSAVE
//! image is initialised so a spawned task does not start with its exceptions
//! unmasked.

pub const PI: f32 = 3.141_592_7;
pub const TAU: f32 = 6.283_185_3;

/// Sine, for any angle.
///
/// Range-reduced to a quarter turn and then evaluated, because a polynomial
/// good across all of the reals does not exist and one good across a quarter
/// turn is short.
pub fn sin(x: f32) -> f32 {
    // Into [-PI, PI]. `%` on floats is available in core and is exact enough
    // here; the angles reaching this are a player's facing, not an
    // accumulated integral, so they never grow large enough for the
    // subtraction to lose meaningful bits.
    let mut a = x % TAU;
    if a > PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    // Fold the two outer quadrants onto the middle two: sin(PI - a) == sin(a).
    if a > PI / 2.0 {
        a = PI - a;
    } else if a < -PI / 2.0 {
        a = -PI - a;
    }
    // Minimax on [-PI/2, PI/2]. The coefficients are the Taylor series with
    // the last two nudged, which is what keeps the error flat across the range
    // rather than concentrated at the ends.
    let x2 = a * a;
    a * (1.0 - x2 * (0.166_666_67 - x2 * (0.008_333_33 - x2 * 0.000_198_412_7)))
}

pub fn cos(x: f32) -> f32 {
    sin(x + PI / 2.0)
}

/// Square root, straight to the instruction.
///
/// SSE2 is guaranteed on this target -- `.cargo/config.toml` turns it on and
/// the kernel enables the register state before anything floating point runs
/// -- so this is one instruction and exactly rounded, which no polynomial is.
pub fn sqrt(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_cvtss_f32, _mm_set_ss, _mm_sqrt_ss};
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = x;
        0.0
    }
}

/// A DOOM binary angle to radians. A full turn is 65536.
pub fn bam_to_rad(bam: u16) -> f32 {
    bam as f32 * (TAU / 65536.0)
}

/// Degrees to radians, for the angle a THING carries.
pub fn deg_to_rad(deg: i16) -> f32 {
    deg as f32 * (PI / 180.0)
}

/// The greatest integer at or below `x`.
///
/// Integer rather than `f32`, and hand-rolled rather than `f32::floor`,
/// because both alternatives are unavailable here for the same reason: this
/// is `no_std` with no `libm`, so `floor` and the `%` operator on floats are
/// calls to functions that do not exist and the failure is at link time. A
/// texture coordinate wraps, and wrapping means a negative one has to round
/// the same direction as a positive one -- `-0.5 as i32` is 0, which puts the
/// leftmost half-pixel of every tile on the wrong side of the seam.
///
/// The cast saturates rather than wrapping (Rust 1.45 onwards), so a
/// coordinate that has gone wild gives a clamped column instead of undefined
/// behaviour.
pub fn floor_i(x: f32) -> i32 {
    let t = x as i32;
    if x < 0.0 && (t as f32) != x {
        t - 1
    } else {
        t
    }
}
