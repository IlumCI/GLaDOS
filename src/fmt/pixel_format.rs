//! a pixel format that represents a 4x4 block of pixels as a 4x4 block of bytes
//!
//! The items below were written by the loop's author against the
//! rung of that name. `selftest` is assembled rather than written:
//! its claim is what J1 counts, and a claim a model spells itself
//! is a claim that can be spelt wrong.

/// A 4x4 block of pixels is represented by a 4x4 block of bytes when encoded.
/// This ensures that the encoding operation maps each pixel directly to a byte,
/// preserving both structure and data integrity for a fixed-size block.
pub fn encode(block: &[u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = block[i];
        i += 1;
    }
    out
}

pub fn selftest() -> bool {
    let mut ok = true;
    let good = encode(&[7u8; 16])[0] == 7 && encode(&[7u8; 16])[15] == 7;
    crate::kprintln!("  {}   a pixel format that represents a 4x4 block of pixels as a 4x4 block of",
                     if good { "ok " } else { "FAIL" });
    ok &= good;
    ok
}
