//! What a miner can choose to be paid in.
//!
//! The pool pays every eligible wallet the same share of each payout, and the
//! wallet's owner chooses what that share is bought as: $GLaDOS, one tokenized
//! stock, or a category basket of them. The choice travels in `glados.hello`
//! and the pool's treasury does the buying (pool/edge/worker/rewards.js holds
//! the token addresses; this side only needs the names).
//!
//! **The codes here must match rewards.js exactly**, and
//! `pool/edge/worker/plan.test.mjs` reads this file to check that they do: a
//! code the kernel offers and the pool does not know would quietly pay $GLaDOS,
//! which is safe and also not what the miner asked for.
//!
//! Anything not on the menu is $GLaDOS rather than an error, on both sides, so
//! a typo in MINER.TXT never stops a machine mining.

/// `(code, what the screen calls it)`, in the order the screen lists them.
pub const MENU: &[(&str, &str)] = &[
    ("glados", "$GLaDOS"),
    ("nvda", "NVIDIA"),
    ("spcx", "SpaceX"),
    ("googl", "Alphabet"),
    ("amzn", "Amazon"),
    ("gme", "GameStop"),
    ("spy", "S&P 500"),
    ("chips", "Chips & hardware"),
    ("os", "Operating systems"),
    ("index", "Index"),
    ("metals", "Metals & commodities"),
];

pub const DEFAULT: &str = "glados";

/// The menu code for what somebody typed, or `None` if it is not on the menu.
/// Case-insensitive, and `$glados` is accepted for the default because that is
/// how the token is written everywhere else.
pub fn code(input: &str) -> Option<&'static str> {
    let t = input.trim().trim_start_matches('$');
    MENU.iter().find(|(c, _)| c.eq_ignore_ascii_case(t)).map(|(c, _)| *c)
}

/// What the screen calls a code; the default's name for anything unknown.
pub fn name(code: &str) -> &'static str {
    MENU.iter().find(|(c, _)| *c == code).map(|(_, n)| *n).unwrap_or("$GLaDOS")
}

/// The codes as one line, for a usage message.
pub fn list() -> alloc::string::String {
    let mut s = alloc::string::String::new();
    for (i, (c, _)) in MENU.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(c);
    }
    s
}

// ---- the choice in force ------------------------------------------------------

/// What this machine is mining for now. Set at boot from `MINER.TXT` or the
/// choice saved in the firmware, changed by typing a code on the mining
/// screen, and read by the greeting so the pool learns it on every connect.
static CURRENT: crate::sync::Spin<&'static str> = crate::sync::Spin::new(DEFAULT);

pub fn current() -> &'static str {
    *CURRENT.lock_irq()
}

pub fn set(code: &'static str) {
    *CURRENT.lock_irq() = code;
}

// Kept in the firmware beside the address (`boot.rs`'s `GladosPayout`), and for
// the same reason: the ISO is read-only and the variable store survives a power
// cycle. Fewer than twenty bytes, a menu code, nothing private.

/// `GladosReward`, NUL-terminated.
const SAVED_NAME: [u16; 13] = [
    b'G' as u16, b'l' as u16, b'a' as u16, b'd' as u16, b'o' as u16, b's' as u16,
    b'R' as u16, b'e' as u16, b'w' as u16, b'a' as u16, b'r' as u16, b'd' as u16, 0,
];
const SAVED_GUID: crate::uefi::Guid = crate::uefi::Guid {
    d1: 0x7a1d_0c3e,
    d2: 0x5b2f,
    d3: 0x4e8a,
    d4: [0x9c, 0x41, 0x6d, 0x3b, 0x2a, 0x90, 0xf1, 0xe7],
};

/// The choice saved on this PC, if any and still on the menu.
pub fn saved() -> Option<&'static str> {
    let mut buf = [0u8; 32];
    let n = crate::cpu::efi_get_variable(&SAVED_NAME, &SAVED_GUID, &mut buf)?;
    code(core::str::from_utf8(&buf[..n]).ok()?)
}

pub fn save(code: &str) -> bool {
    crate::cpu::efi_set_variable(&SAVED_NAME, &SAVED_GUID, code.as_bytes())
}

/// Claims, with no firmware and no pool; run by `boot::checks` at every boot.
pub fn checks() -> alloc::vec::Vec<(bool, alloc::string::String)> {
    use alloc::string::String;
    let mut out = alloc::vec::Vec::new();
    let mut ok = |c: bool, what: &str| out.push((c, String::from(what)));
    ok(code("CHIPS") == Some("chips") && code(" nvda ") == Some("nvda"), "a reward is case-insensitive and trimmed");
    ok(code("$GLaDOS") == Some("glados") && code("glados") == Some("glados"), "$GLaDOS is accepted as written");
    ok(code("tesla").is_none() && code("").is_none(), "anything off the menu is not a reward code");
    ok(name("chips") == "Chips & hardware" && name("nonsense") == "$GLaDOS", "the screen names a reward, and $GLaDOS for anything else");
    ok(MENU[0].0 == DEFAULT, "$GLaDOS is the first reward and the default");
    out
}
