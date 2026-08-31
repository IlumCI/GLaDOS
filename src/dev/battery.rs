//! The battery, read the only way a battery can be read: by running the
//! firmware's own bytecode.
//!
//! Everything under this is already built. The namespace says where `BAT1`
//! lives, the evaluator runs its methods, and the region handlers turn a field
//! into a number. This is the part that knows what the numbers mean.
//!
//! ### Two packages and a unit that changes between them
//!
//! `_BIF` describes the battery and `_BST` describes its state. The trap is
//! that **the units are not fixed**: element zero of `_BIF` says whether the
//! whole set is in milliwatts or milliamps, and different machines answer
//! differently. A design capacity in mAh compared against a rate in mW is a
//! number that looks like a time and is off by the battery's voltage.
//!
//! So everything is normalised to milliwatts at the boundary, once, here.
//!
//! A percentage is the exception and is computed *before* any conversion, on
//! purpose: remaining and last-full are always in the same unit as each other,
//! so their ratio is right whichever unit that is, and converting first would
//! introduce a rounding error to answer a question that did not need it.
//!
//! ### What "present" means
//!
//! `_STA` bit four says a battery is in the bay. That is separate from the
//! device existing, and both are separate from the battery answering: a bay
//! with no cell in it is a device that is present and reports nothing. Each is
//! a different field here rather than collapsed into one boolean, because a
//! machine with no battery and a machine whose battery cannot be read need to
//! be told apart.

use crate::acpi::{self, aml, eval};
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Charging,
    Discharging,
    /// Neither charging nor discharging, which on a plugged-in machine means
    /// full and on an unplugged one means the firmware is not sure.
    Idle,
    Unknown,
}

impl State {
    pub fn label(&self) -> &'static str {
        match self {
            State::Charging => "charging",
            State::Discharging => "discharging",
            State::Idle => "idle",
            State::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Charge {
    /// A cell is in the bay, from `_STA` bit four.
    pub present: bool,
    pub percent: u8,
    pub state: State,
    /// Critical, from `_BST` bit two. Separate from the percentage because the
    /// firmware may call a battery critical at a level the arithmetic does
    /// not.
    pub critical: bool,
    /// All in milliwatt-hours, or milliwatts for the rate, whatever unit the
    /// firmware reported them in.
    pub rate_mw: Option<u32>,
    pub remaining_mwh: Option<u32>,
    pub full_mwh: Option<u32>,
    pub design_mwh: Option<u32>,
    pub voltage_mv: Option<u32>,
    /// Time to empty, or to full when charging. `None` when the rate is zero,
    /// which is what a battery that is neither charging nor discharging
    /// reports, and dividing by it would answer infinity.
    pub minutes: Option<u32>,
    /// Last-full against design capacity: how much of its original capacity
    /// the cell still holds.
    pub health: Option<u8>,
    /// From the adapter's `_PSR`, and `None` when there is no adapter device.
    pub on_ac: Option<bool>,
}

/// Decode an EISA id into the form a person writes it.
///
/// `_HID` holds `PNP0C0A` as an integer: three five-bit letters packed into a
/// byte-swapped sixteen-bit half, then four hex digits. Comparing against the
/// packed constant would work and would be a magic number nobody could check,
/// so this decodes and compares the text instead.
pub fn eisa(id: u32) -> String {
    let b = id.to_le_bytes();
    let mfr = ((b[0] as u16) << 8) | b[1] as u16;
    let mut s = String::new();
    for shift in [10u16, 5, 0] {
        let c = ((mfr >> shift) & 0x1F) as u8 + 0x40;
        s.push(c as char);
    }
    for byte in [b[2], b[3]] {
        for nibble in [byte >> 4, byte & 0x0F] {
            s.push(char::from_digit(nibble as u32, 16).unwrap_or('?').to_ascii_uppercase());
        }
    }
    s
}

/// Whether a device's `_HID` or `_CID` names this hardware id.
fn is_a(it: &mut eval::Interp, ns: &aml::Namespace, dev: usize, want: &str) -> bool {
    for key in [b"_HID", b"_CID"] {
        let path = aml::Path { rooted: false, parents: 0, segs: alloc::vec![*key] };
        let Some(node) = ns.resolve(dev, &path) else { continue };
        // Only a child of this device counts. `resolve` searches upward for a
        // single segment, so without this a device with no `_HID` of its own
        // would inherit its parent's and every bay would look like a battery.
        if ns.node(node).parent != dev {
            continue;
        }
        match it.eval_node(node, &[]) {
            Ok(eval::Value::Int(v)) => {
                if eisa(v as u32) == want {
                    return true;
                }
            }
            Ok(eval::Value::Str(s)) => {
                if s == want {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn child(ns: &aml::Namespace, dev: usize, name: &[u8; 4]) -> Option<usize> {
    let path = aml::Path { rooted: false, parents: 0, segs: alloc::vec![*name] };
    let n = ns.resolve(dev, &path)?;
    if ns.node(n).parent == dev {
        Some(n)
    } else {
        None
    }
}

fn elem(p: &[eval::Value], i: usize) -> Option<u32> {
    let v = p.get(i)?.int().ok()?;
    // 0xFFFFFFFF is ACPI's "unknown", and it is not a capacity of four
    // billion. Taking it literally is how a battery reports as 4294967295 mWh.
    if v == 0xFFFF_FFFF || v == u64::MAX {
        None
    } else {
        Some(v as u32)
    }
}

/// Read the battery, running the firmware to do it.
pub fn read() -> Option<Charge> {
    let a = acpi::parsed()?;
    acpi::with_namespace(&a, |ns| {
        let mut it = eval::Interp::new(ns);

        // The adapter first, because it is one method and it tells the rest of
        // the reading how to be interpreted.
        let mut on_ac = None;
        for i in 0..ns.len() {
            if !matches!(ns.node(i).kind, aml::Kind::Device) {
                continue;
            }
            if is_a(&mut it, ns, i, "ACPI0003") {
                if let Some(psr) = child(ns, i, b"_PSR") {
                    if let Ok(v) = it.eval_node(psr, &[]) {
                        on_ac = v.int().ok().map(|x| x != 0);
                    }
                }
                break;
            }
        }

        for i in 0..ns.len() {
            if !matches!(ns.node(i).kind, aml::Kind::Device) {
                continue;
            }
            if !is_a(&mut it, ns, i, "PNP0C0A") {
                continue;
            }

            // `_STA` bit four is "a cell is in the bay". A device that exists
            // with an empty bay is a real state and answers zero here.
            let present = match child(ns, i, b"_STA") {
                Some(sta) => it.eval_node(sta, &[]).ok().and_then(|v| v.int().ok()).unwrap_or(0)
                    & 0x10
                    != 0,
                // No `_STA` means always present, which is what ACPI says.
                None => true,
            };

            // `_BIX` is the later, longer form and starts with a revision, so
            // its capacities sit one element further along. Preferred where it
            // exists, because a machine that offers both may only keep the new
            // one accurate.
            let (bif, shift) = match child(ns, i, b"_BIX") {
                Some(n) => (it.eval_node(n, &[]).ok(), 1usize),
                None => (child(ns, i, b"_BIF").and_then(|n| it.eval_node(n, &[]).ok()), 0),
            };
            let info: Vec<eval::Value> = match bif {
                Some(eval::Value::Pkg(p)) => p,
                _ => Vec::new(),
            };

            // Element zero: 0 is milliwatts, 1 is milliamps. Everything below
            // is converted once, here, using the design voltage.
            let in_ma = elem(&info, shift).unwrap_or(0) == 1;
            let design = elem(&info, shift + 1);
            let full = elem(&info, shift + 2);
            let design_mv = elem(&info, shift + 4);

            let state_pkg = child(ns, i, b"_BST").and_then(|n| it.eval_node(n, &[]).ok());
            let st: Vec<eval::Value> = match state_pkg {
                Some(eval::Value::Pkg(p)) => p,
                _ => Vec::new(),
            };
            let flags = elem(&st, 0).unwrap_or(0);
            let rate = elem(&st, 1);
            let remaining = elem(&st, 2);
            let voltage = elem(&st, 3).or(design_mv);

            // The percentage, before any conversion. Remaining and last-full
            // are always in the same unit as each other, so the ratio is right
            // whichever unit that is, and converting first would round twice
            // to answer a question that needed no conversion at all.
            let percent = match (remaining, full) {
                (Some(r), Some(f)) if f > 0 => ((r as u64 * 100 / f as u64).min(100)) as u8,
                _ => 0,
            };

            let mv = voltage.unwrap_or(0) as u64;
            let to_mw = |v: Option<u32>| -> Option<u32> {
                let v = v? as u64;
                if in_ma {
                    if mv == 0 {
                        return None;
                    }
                    Some((v * mv / 1000) as u32)
                } else {
                    Some(v as u32)
                }
            };

            // Time remaining, from the raw values rather than the converted
            // ones: they share a unit, so the division is exact either way and
            // the conversion could only lose precision.
            let minutes = match (remaining, rate) {
                (Some(r), Some(rt)) if rt > 0 => {
                    let target = if flags & 0x02 != 0 {
                        // Charging: the time that matters is to full.
                        full.unwrap_or(r).saturating_sub(r)
                    } else {
                        r
                    };
                    Some((target as u64 * 60 / rt as u64) as u32)
                }
                _ => None,
            };

            let state = if flags & 0x01 != 0 {
                State::Discharging
            } else if flags & 0x02 != 0 {
                State::Charging
            } else if remaining.is_some() {
                State::Idle
            } else {
                State::Unknown
            };

            let health = match (full, design) {
                (Some(f), Some(d)) if d > 0 => Some(((f as u64 * 100 / d as u64).min(100)) as u8),
                _ => None,
            };

            return Some(Charge {
                present,
                percent,
                state,
                critical: flags & 0x04 != 0,
                rate_mw: to_mw(rate),
                remaining_mwh: to_mw(remaining),
                full_mwh: to_mw(full),
                design_mwh: to_mw(design),
                voltage_mv: voltage,
                minutes,
                health,
                on_ac,
            });
        }

        // No battery device. A desktop, or a virtual machine.
        on_ac.map(|ac| Charge {
            present: false,
            percent: 0,
            state: State::Unknown,
            critical: false,
            rate_mw: None,
            remaining_mwh: None,
            full_mwh: None,
            design_mwh: None,
            voltage_mv: None,
            minutes: None,
            health: None,
            on_ac: Some(ac),
        })
    })
    .flatten()
}

/// How often a reading may be taken.
///
/// A `_BST` is hundreds of interpreter steps and a handful of embedded
/// controller transactions, each of which waits on hardware. A taskbar that
/// asked per frame would spend the machine on it, and a battery does not move
/// fast enough for that to buy anything.
const REFRESH_MS: u64 = 10_000;

static CACHED: crate::sync::Racy<Option<Charge>> = crate::sync::Racy::new(None);
static TAKEN_AT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// The last reading, taking a new one if the old one is stale.
pub fn status() -> Option<Charge> {
    let now = crate::dev::lapic::ticks() * 1000 / crate::TIMER_HZ as u64;
    let last = TAKEN_AT.load(core::sync::atomic::Ordering::Relaxed);
    let have = unsafe { *CACHED.get() };
    if have.is_some() && now.saturating_sub(last) < REFRESH_MS {
        return have;
    }
    let fresh = read();
    if fresh.is_some() {
        unsafe { *CACHED.get() = fresh };
        TAKEN_AT.store(now, core::sync::atomic::Ordering::Relaxed);
    }
    fresh.or(have)
}

/// Force a reading now.
pub fn refresh() -> Option<Charge> {
    TAKEN_AT.store(0, core::sync::atomic::Ordering::Relaxed);
    status()
}

pub fn report() {
    let Some(c) = refresh() else {
        crate::kprintln!("  no battery and no adapter in the namespace");
        crate::kprintln!("  ('acpi ns' lists what the firmware does declare)");
        return;
    };
    match c.on_ac {
        Some(true) => crate::kprintln!("  on mains"),
        Some(false) => crate::kprintln!("  on battery"),
        None => crate::kprintln!("  no adapter device, so mains state is unknown"),
    }
    if !c.present {
        crate::kprintln!("  no cell in the bay");
        return;
    }
    crate::kprintln!("  {}%, {}{}", c.percent, c.state.label(), if c.critical { ", CRITICAL" } else { "" });
    match c.minutes {
        Some(m) => crate::kprintln!(
            "  {}h {:02}m {}",
            m / 60,
            m % 60,
            if c.state == State::Charging { "until full" } else { "remaining" }
        ),
        None => crate::kprintln!("  no time estimate: the rate is zero, and dividing by it answers forever"),
    }
    if let (Some(r), Some(f)) = (c.remaining_mwh, c.full_mwh) {
        crate::kprintln!("  {} of {} mWh", r, f);
    }
    if let Some(rt) = c.rate_mw {
        crate::kprintln!("  drawing {} mW", rt);
    }
    if let (Some(h), Some(d)) = (c.health, c.design_mwh) {
        crate::kprintln!("  {}% of its design capacity of {} mWh", h, d);
    }
    if let Some(v) = c.voltage_mv {
        crate::kprintln!("  {} mV", v);
    }
}

pub fn selftest() -> bool {
    let mut ok = true;
    fn claim(ok: &mut bool, good: bool, what: &str) {
        crate::kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        *ok &= good;
    }

    // The hardware id decoder, which is what finds the device at all. Checked
    // against the two ids this module looks for and one it must not match.
    claim(&mut ok, eisa(0x0A0C_D041) == "PNP0C0A", "the battery's hardware id decodes");
    claim(&mut ok, eisa(0x0303_D041) == "PNP0303", "and so does an unrelated one");
    claim(&mut ok, eisa(0x0A0C_D041) != "PNP0C0B", "and it does not match a neighbouring id");

    // The reading, on whatever this machine is.
    match read() {
        None => {
            claim(&mut ok, acpi::parsed().is_some(), "ACPI was available to look in");
            crate::kprintln!("  no battery here, which is the right answer under emulation");
        }
        Some(c) => {
            claim(&mut ok, c.percent <= 100, "a percentage is a percentage");
            claim(
                &mut ok,
                c.health.map(|h| h <= 100).unwrap_or(true),
                "and so is the health figure",
            );
            // The one that catches a unit mix-up: a rate of zero must not
            // produce a time, because the division would answer forever.
            claim(
                &mut ok,
                c.rate_mw.unwrap_or(0) != 0 || c.minutes.is_none(),
                "a zero rate gives no time estimate rather than an infinite one",
            );
            claim(
                &mut ok,
                match (c.remaining_mwh, c.full_mwh) {
                    (Some(r), Some(f)) => r <= f.saturating_mul(2),
                    _ => true,
                },
                "remaining capacity is not wildly past full, which a unit mix-up gives",
            );
            report();
        }
    }
    ok
}
