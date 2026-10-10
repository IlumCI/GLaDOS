//! The desktop's settings: what there is, how each applies, where it is kept.
//!
//! One table, one verb (`desk <key> <value>`), one file (`/etc/desk`). The
//! Settings window's Desktop page is a set of buttons that run the verb, so
//! everything it can do has a typed form a headless driver can reach -- the
//! rule every control here has kept since `win keys`. Each value lives where
//! it is used (the scale in `theme`, the gaps in `tile`, the motion in
//! `anim`); this module only names them, applies them, and writes them down.
//!
//! **Kept in the namespace, durable by snapshot**, as everything else the
//! machine remembers is: a write lands in the working tree, and a snapshot
//! is what carries it across a reboot. Read once at the desktop's start.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

const PATH: &str = "/etc/desk";

/// Bumped by every setting applied, and part of every window's stamp.
///
/// A window's image is keyed by what its painter reads, and a setting is
/// something every painter reads -- blur, found by `render verify` on its
/// first run, was not in the key, so turning it off left every window's
/// image holding the glass it had been drawn with. One counter instead of
/// one more hash per setting, because the next setting would be the next
/// one missed.
static GENERATION: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn generation() -> u64 {
    GENERATION.load(core::sync::atomic::Ordering::Relaxed)
}

/// Every setting: its name, what it takes, and what it does.
pub const KEYS: &[(&str, &str, &str)] = &[
    ("scale", "75..250", "the interface's size, in per cent"),
    ("motion", "0..400", "how long motion takes, in per cent; 0 is none"),
    ("glass", "sky|warm|graphite", "the focused caption's glass"),
    ("ring", "aperture|sky|ice", "the focused window's border"),
    ("dim", "0..60", "how dark unfocused windows go, in per cent"),
    ("gaps", "<in> <out>", "the tiling gaps, between windows and round the edge"),
    ("round", "0..16", "how round a window's head is"),
    ("blur", "on|off", "whether glass shows the blurred wallpaper"),
    ("tile", "on|off", "whether a new window tiles or floats"),
    ("quality", "<floor> <ceiling>", "what the governor may shed, 0 everything .. 4 no motion"),
];

/// A setting's value now, as `desk` prints it and the file stores it.
pub fn get(key: &str) -> Option<String> {
    Some(match key {
        "scale" => format!("{}", super::theme::scale() / 10),
        "motion" => format!("{}", super::anim::speed()),
        "glass" => String::from(super::theme::caption_tint()),
        "ring" => String::from(super::desk::ring_name()),
        "dim" => format!("{}", super::desk::dim_percent()),
        "gaps" => {
            let (i, o) = super::tile::gaps_logical();
            format!("{} {}", i, o)
        }
        "round" => format!("{}", super::theme::round_logical()),
        "blur" => String::from(if super::glass::blur_setting() { "on" } else { "off" }),
        "quality" => {
            let (f, c) = super::quality::range();
            format!("{} {}", f, c)
        }
        "tile" => String::from(if super::desk::tile_new() { "on" } else { "off" }),
        _ => return None,
    })
}

fn onoff(v: &str) -> Result<bool, &'static str> {
    match v {
        "on" | "yes" | "1" => Ok(true),
        "off" | "no" | "0" => Ok(false),
        _ => Err("wants on or off"),
    }
}

fn num(v: &str, lo: u32, hi: u32) -> Result<u32, &'static str> {
    match v.trim().parse::<u32>() {
        Ok(n) if (lo..=hi).contains(&n) => Ok(n),
        _ => Err("out of range"),
    }
}

/// Apply one setting, without writing it down.
fn apply(key: &str, val: &str) -> Result<(), &'static str> {
    match key {
        "scale" => super::desk::set_scale(num(val, 75, 250)? * 10),
        "motion" => super::anim::set_speed(num(val, 0, 400)?),
        "glass" => {
            if !super::theme::set_caption_tint(val) {
                return Err("no such glass");
            }
        }
        "ring" => {
            if !super::desk::set_ring(val) {
                return Err("no such ring");
            }
        }
        "dim" => super::desk::set_dim(num(val, 0, 60)?),
        "gaps" => {
            let mut it = val.split_whitespace();
            let i = num(it.next().unwrap_or(""), 0, 64)?;
            let o = num(it.next().unwrap_or(""), 0, 128)?;
            super::tile::set_gaps(i, o);
        }
        "round" => super::theme::set_round(num(val, 0, 16)?),
        "blur" => super::glass::set_blur(onoff(val)?),
        "tile" => super::desk::set_tile_new(onoff(val)?),
        "quality" => {
            let mut it = val.split_whitespace();
            let f = num(it.next().unwrap_or(""), 0, 4)?;
            let c = num(it.next().unwrap_or(""), 0, 4)?;
            super::quality::set_range(f as u8, c as u8);
        }
        _ => return Err("no such setting"),
    }
    GENERATION.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    super::render::damage_all();
    Ok(())
}

/// Set a setting, and keep every setting as it now stands.
pub fn set(key: &str, val: &str) -> Result<(), &'static str> {
    apply(key, val)?;
    save();
    Ok(())
}

fn save() {
    let mut text = String::new();
    for (k, _, _) in KEYS {
        if let Some(v) = get(k) {
            text.push_str(k);
            text.push(' ');
            text.push_str(&v);
            text.push('\n');
        }
    }
    crate::sysbox::write_blob(PATH, text.into_bytes());
}

/// Read `/etc/desk` and apply what it says. A line that will not apply is
/// skipped rather than failing the rest: a file written by a later build may
/// carry a setting this one does not have.
pub fn load() -> usize {
    let Some(bytes) = crate::sysbox::read_blob(PATH) else { return 0 };
    let Ok(text) = core::str::from_utf8(&bytes) else { return 0 };
    let mut n = 0;
    for line in text.lines() {
        let line = line.trim();
        let Some((k, v)) = line.split_once(' ') else { continue };
        if apply(k, v.trim()).is_ok() {
            n += 1;
        }
    }
    n
}

/// The Desktop page: each setting, its value, and buttons that change it.
pub fn rows() -> Vec<(String, Vec<(String, String)>)> {
    let step = |key: &str, cur: u32, by: u32, lo: u32, hi: u32| -> Vec<(String, String)> {
        let mut v = Vec::new();
        if cur > lo {
            v.push((String::from("less"), format!("desk {} {}", key, cur.saturating_sub(by).max(lo))));
        }
        if cur < hi {
            v.push((String::from("more"), format!("desk {} {}", key, (cur + by).min(hi))));
        }
        v
    };
    let pick = |key: &str, names: &[&str]| -> Vec<(String, String)> {
        names.iter().map(|n| (String::from(*n), format!("desk {} {}", key, n))).collect()
    };
    let scale = super::theme::scale() / 10;
    let motion = super::anim::speed();
    let dim = super::desk::dim_percent();
    let round = super::theme::round_logical();
    let (gi, go) = super::tile::gaps_logical();
    alloc::vec![
        (format!("Size  {}%", scale), step("scale", scale, 25, 75, 250)),
        (format!("Motion  {}", if motion == 0 { String::from("off") } else { format!("{}%", motion) }), {
            let mut v = step("motion", motion, 50, 0, 400);
            v.push((String::from(if motion == 0 { "on" } else { "off" }), format!("desk motion {}", if motion == 0 { 100 } else { 0 })));
            v
        }),
        (format!("Caption glass  {}", super::theme::caption_tint()), pick("glass", &["sky", "warm", "graphite"])),
        (format!("Focus border  {}", super::desk::ring_name()), pick("ring", &["aperture", "sky", "ice"])),
        (format!("Dim others  {}%", dim), step("dim", dim, 5, 0, 60)),
        (
            format!("Gaps  {} / {}", gi, go),
            alloc::vec![
                (String::from("tighter"), format!("desk gaps {} {}", gi.saturating_sub(2), go.saturating_sub(5))),
                (String::from("wider"), format!("desk gaps {} {}", (gi + 2).min(64), (go + 5).min(128))),
            ],
        ),
        (format!("Corners  {}", round), step("round", round, 2, 0, 16)),
        (format!("Blur  {}", get("blur").unwrap_or_default()), pick("blur", &["on", "off"])),
        (format!("New windows  {}", if super::desk::tile_new() { "tile" } else { "float" }), pick("tile", &["on", "off"])),
        {
            let (f, c) = super::quality::range();
            let what = match (f, c) {
                (0, 4) => String::from("auto"),
                (0, 0) => String::from("always full"),
                (4, 4) => String::from("always plain"),
                _ => format!("{}..{}", f, c),
            };
            (
                format!("Quality  {} (now: {})", what, super::quality::name(super::quality::level())),
                alloc::vec![
                    (String::from("auto"), String::from("desk quality 0 4")),
                    (String::from("full"), String::from("desk quality 0 0")),
                    (String::from("plain"), String::from("desk quality 4 4")),
                ],
            )
        },
    ]
}
