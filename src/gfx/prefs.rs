//! Every setting the machine has, in one table: the Settings app, its search,
//! and the taskbar's quick settings all read from here.
//!
//! **A setting is a getter and a verb.** What a control shows is read from
//! the subsystem that owns the value, and what it does is a shell command --
//! the same command a person types and `drive.py` sends. So nothing here can
//! drift from what the machine actually does, everything stays scriptable,
//! and the app never has to reach into the desktop while the desktop is
//! drawing it (a `DeskApp` runs inside the desktop's borrow, which is why
//! panels queue commands rather than call).
//!
//! The old Settings window was a panel with less/more buttons for every
//! number and a page list that took half the window. The status those pages
//! gathered is still worth reading -- a battery's health, an adapter's address
//! -- so a page may carry a `Section`: the old page's rows, drawn in the new
//! style, beside the controls that replaced its buttons.

use super::ui::Widget;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A small drawing standing for a page on its tile and in the sidebar.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    Network,
    Wifi,
    Desktop,
    Brush,
    Sound,
    Power,
    Assistant,
    Model,
    Storage,
    Linux,
    Hardware,
    Updates,
    System,
}

pub struct Page {
    pub id: &'static str,
    pub title: &'static str,
    pub glyph: Glyph,
    /// One line under the title on the home grid: what this page is about.
    pub blurb: &'static str,
    /// What is true now, for the tile: "Connected - eth0", "125% - sky".
    pub summary: fn() -> String,
}

/// What a setting is and how it changes.
pub enum Ctl {
    /// On or off.
    Toggle { get: fn() -> bool, on: &'static str, off: &'static str },
    /// A number on a range. `cmd` makes the command for a value; it is only
    /// sent when a drag ends, so a slider is one command and not fifty.
    Slider { min: i32, max: i32, step: i32, unit: &'static str, get: fn() -> i32, cmd: fn(i32) -> String },
    /// One of several, each its own command.
    Choice { opts: &'static [(&'static str, &'static str)], get: fn() -> usize },
    /// A value to read and not to change.
    Info(fn() -> String),
    /// Something to do now.
    Button(&'static str),
    /// Another page, or a window: `page:<id>` or a command.
    Open(&'static str),
    /// A block of the old page's status rows, rebuilt when the page is shown.
    Section(fn() -> Vec<Widget>),
}

pub struct Item {
    pub page: &'static str,
    /// The group heading it sits under on its page.
    pub group: &'static str,
    pub label: &'static str,
    /// One sentence under the label, saying what it does.
    pub help: &'static str,
    /// Extra words search should find it by.
    pub words: &'static str,
    pub ctl: Ctl,
    /// Offered in the taskbar's quick settings too.
    pub quick: bool,
    /// Shown only when this answers true: a turbo switch on a part with no
    /// turbo control is a control that does nothing.
    pub when: Option<fn() -> bool>,
}

const fn item(page: &'static str, group: &'static str, label: &'static str, help: &'static str, words: &'static str, ctl: Ctl) -> Item {
    Item { page, group, label, help, words, ctl, quick: false, when: None }
}

const fn quick(page: &'static str, group: &'static str, label: &'static str, help: &'static str, words: &'static str, ctl: Ctl) -> Item {
    Item { page, group, label, help, words, ctl, quick: true, when: None }
}

// ---- the values ----------------------------------------------------------

fn scale() -> i32 {
    (super::theme::scale() / 10) as i32
}
fn motion() -> i32 {
    super::anim::speed() as i32
}
fn motion_on() -> bool {
    super::anim::speed() > 0
}
fn dim() -> i32 {
    super::desk::dim_percent() as i32
}
fn round() -> i32 {
    super::theme::round_logical() as i32
}
fn gap_in() -> i32 {
    super::tile::gaps_logical().0 as i32
}
fn gap_out() -> i32 {
    super::tile::gaps_logical().1 as i32
}
fn pointer() -> i32 {
    crate::dev::mouse::speed() as i32
}
fn blur() -> bool {
    super::glass::blur_setting()
}
fn tile() -> bool {
    super::desk::tile_new()
}
fn index_of(opts: &[(&str, &str)], cur: &str) -> usize {
    opts.iter().position(|(l, _)| l.eq_ignore_ascii_case(cur)).unwrap_or(0)
}
const GLASS: &[(&str, &str)] = &[("Sky", "desk glass sky"), ("Warm", "desk glass warm"), ("Graphite", "desk glass graphite")];
fn glass() -> usize {
    index_of(GLASS, super::theme::caption_tint())
}
const RING: &[(&str, &str)] = &[("Aperture", "desk ring aperture"), ("Sky", "desk ring sky"), ("Ice", "desk ring ice")];
fn ring() -> usize {
    index_of(RING, super::desk::ring_name())
}
const QUALITY: &[(&str, &str)] = &[("Automatic", "desk quality 0 4"), ("Always full", "desk quality 0 0"), ("Always plain", "desk quality 4 4")];
fn quality() -> usize {
    match super::quality::range() {
        (0, 0) => 1,
        (4, 4) => 2,
        _ => 0,
    }
}

fn volume() -> i32 {
    crate::voice::volume() as i32
}
fn voice() -> bool {
    crate::voice::on()
}

const GOVERNOR: &[(&str, &str)] = &[("Performance", "power performance"), ("Balanced", "power balanced"), ("Power saver", "power powersave")];
fn governor() -> usize {
    match crate::dev::power::governor() {
        crate::dev::power::Governor::Performance => 0,
        crate::dev::power::Governor::Balanced => 1,
        crate::dev::power::Governor::Powersave => 2,
    }
}
fn turbo() -> bool {
    crate::dev::power::turbo().unwrap_or(false)
}
fn turbo_known() -> bool {
    crate::dev::power::turbo().is_some()
}

fn initiative() -> bool {
    crate::ai::initiative::status().4
}
fn godel() -> bool {
    crate::ai::godel::enabled()
}
fn learn() -> bool {
    crate::ai::learn::enabled()
}

fn autosnap() -> bool {
    crate::sysbox::autosnap_enabled()
}
fn autosnap_every() -> i32 {
    crate::sysbox::autosnap_interval() as i32
}

const CORES: &[(&str, &str)] = &[("Every core", "linux cores all"), ("Not the first", "linux cores any"), ("One", "linux cores one")];
fn cores() -> usize {
    if !crate::linux::syscall::SPREAD.load(core::sync::atomic::Ordering::Relaxed) {
        2
    } else if crate::linux::syscall::SPARE_BSP.load(core::sync::atomic::Ordering::Relaxed) {
        1
    } else {
        0
    }
}

// ---- what each tile says ---------------------------------------------------

fn sum_network() -> String {
    let c = crate::net::config();
    if crate::net::ready() {
        format!("Connected - {}.{}.{}.{}", c.ip[0], c.ip[1], c.ip[2], c.ip[3])
    } else {
        String::from("Not connected")
    }
}
fn sum_wifi() -> String {
    match crate::net::wifi::hardware().first() {
        Some(_) => String::from("Adapter found"),
        None => String::from("No adapter"),
    }
}
fn sum_desktop() -> String {
    format!("{} - motion {}", if tile() { "Tiling" } else { "Floating" }, if motion_on() { "on" } else { "off" })
}
fn sum_custom() -> String {
    format!("{}% - {} glass", scale(), super::theme::caption_tint())
}
fn sum_sound() -> String {
    match volume() {
        0 => String::from("Muted"),
        v => format!("Volume {}%{}", v, if voice() { " - voice on" } else { "" }),
    }
}
fn sum_power() -> String {
    match crate::dev::battery::status() {
        Some(c) if c.present => format!("{}% - {}", c.percent, GOVERNOR[governor()].0),
        _ => String::from(GOVERNOR[governor()].0),
    }
}
fn sum_assistant() -> String {
    String::from(match (initiative(), godel()) {
        (true, true) => "Awake - improving at night",
        (true, false) => "Awake",
        (false, _) => "Resting",
    })
}
fn sum_model() -> String {
    String::from(if crate::ai::engine_ready() { "Loaded, in the kernel" } else { "No model loaded" })
}
fn sum_storage() -> String {
    let m = if crate::store::mounted() { "Store mounted" } else { "No store" };
    format!("{} - autosnap {}", m, if autosnap() { "on" } else { "off" })
}
fn sum_linux() -> String {
    String::from(if crate::linux::disk::claims("/usr") { "Root image mounted" } else { "No root image" })
}
fn sum_hardware() -> String {
    let d = crate::dev::manager::snapshot();
    use crate::dev::manager::Status;
    // A bridge with "nothing to drive" is not a device in trouble; counting it
    // made every machine read as five things wrong.
    let bad = d
        .iter()
        .filter(|x| match &x.status {
            Status::Working | Status::Ready => false,
            Status::NoDriver(why) => !why.contains("nothing to drive"),
            _ => true,
        })
        .count();
    if bad == 0 {
        format!("{} devices, all working", d.len())
    } else {
        format!("{} devices, {} need attention", d.len(), bad)
    }
}
fn sum_updates() -> String {
    format!("GLaDOS {}", crate::VERSION)
}
fn sum_system() -> String {
    let (used, total) = crate::mem::heap::HEAP.stats();
    format!("{} MiB of {} MiB in use", used >> 20, total >> 20)
}

pub const PAGES: &[Page] = &[
    Page { id: "network", title: "Network", glyph: Glyph::Network, blurb: "Wired, addresses, names", summary: sum_network },
    Page { id: "wifi", title: "Wi-Fi", glyph: Glyph::Wifi, blurb: "Wireless networks", summary: sum_wifi },
    Page { id: "custom", title: "Customization", glyph: Glyph::Brush, blurb: "Size, glass, colours, corners", summary: sum_custom },
    Page { id: "desktop", title: "Desktop", glyph: Glyph::Desktop, blurb: "Windows, tiling, motion, pointer", summary: sum_desktop },
    Page { id: "sound", title: "Sound", glyph: Glyph::Sound, blurb: "Volume and the voice", summary: sum_sound },
    Page { id: "power", title: "Power", glyph: Glyph::Power, blurb: "Battery, speed, temperature", summary: sum_power },
    Page { id: "assistant", title: "Assistant", glyph: Glyph::Assistant, blurb: "What it does on its own", summary: sum_assistant },
    Page { id: "model", title: "Model", glyph: Glyph::Model, blurb: "The language model in the kernel", summary: sum_model },
    Page { id: "storage", title: "Storage", glyph: Glyph::Storage, blurb: "Snapshots and the disk", summary: sum_storage },
    Page { id: "linux", title: "Linux programs", glyph: Glyph::Linux, blurb: "Guests, packages, services", summary: sum_linux },
    Page { id: "hardware", title: "Hardware", glyph: Glyph::Hardware, blurb: "Every device and its driver", summary: sum_hardware },
    Page { id: "updates", title: "Updates", glyph: Glyph::Updates, blurb: "New versions of GLaDOS", summary: sum_updates },
    Page { id: "system", title: "System", glyph: Glyph::System, blurb: "About, memory, restart", summary: sum_system },
];

pub fn page(id: &str) -> Option<&'static Page> {
    PAGES.iter().find(|p| p.id == id)
}

/// The old Settings pages' routes, so `win open set:desk` lands somewhere.
pub fn from_route(r: &str) -> &'static str {
    match r {
        "net" => "network",
        "desk" => "custom",
        "sys" => "system",
        "" => "",
        other => page(other).map(|p| p.id).unwrap_or(""),
    }
}

// ---- sections borrowed from the old pages ------------------------------------

fn sec_network() -> Vec<Widget> {
    super::ui::settings_body("net")
        .into_iter()
        .filter(|w| !matches!(w, Widget::Field { .. } | Widget::Button { action: super::ui::Action::Apply(_), .. }))
        .collect()
}
fn sec_wifi() -> Vec<Widget> {
    super::ui::settings_body("wifi")
}
fn sec_power() -> Vec<Widget> {
    super::ui::settings_body("power")
}
fn sec_updates() -> Vec<Widget> {
    super::ui::settings_body("updates")
}
fn sec_voice() -> Vec<Widget> {
    let mut v = Vec::new();
    v.push(super::ui::Widget::Status {
        name: String::from("Voice"),
        value: String::from(if crate::voice::neural::model().is_some() { "Kokoro-82M, in the kernel" } else { "formant synthesiser" }),
        tone: super::ui::Tone::Plain,
    });
    v
}
fn sec_hardware() -> Vec<Widget> {
    use crate::dev::manager::{Category, Status};
    let all = crate::dev::manager::snapshot();
    let mut v = Vec::new();
    for c in Category::ALL {
        let mine: Vec<_> = all.iter().filter(|d| d.category == c).collect();
        if mine.is_empty() {
            continue;
        }
        v.push(Widget::Heading(String::from(c.name())));
        for d in mine {
            let tone = match d.status {
                Status::Working | Status::Ready => super::ui::Tone::Ok,
                Status::Partial(_) => super::ui::Tone::Warn,
                Status::NoDriver(_) => super::ui::Tone::Plain,
                Status::Failing(_) => super::ui::Tone::Bad,
            };
            let mut value = String::from(d.status.word());
            if let Some(dr) = d.driver {
                value = format!("{} - {}", value, dr);
            }
            if !d.detail.is_empty() {
                value = format!("{} - {}", value, d.detail);
            }
            v.push(Widget::Status { name: d.name.clone(), value, tone });
            if let Some(r) = d.status.reason() {
                v.push(super::ui::note(r));
            }
        }
    }
    v
}
fn sec_system() -> Vec<Widget> {
    let (used, total) = crate::mem::heap::HEAP.stats();
    let up = crate::dev::lapic::ticks() / crate::TIMER_HZ as u64;
    alloc::vec![
        Widget::Status { name: String::from("Version"), value: format!("GLaDOS {}", crate::VERSION), tone: super::ui::Tone::Plain },
        Widget::Status { name: String::from("Up for"), value: format!("{}h {:02}m {:02}s", up / 3600, up / 60 % 60, up % 60), tone: super::ui::Tone::Plain },
        Widget::Status { name: String::from("Processors"), value: format!("{} cores", crate::smp::online()), tone: super::ui::Tone::Plain },
        Widget::Status { name: String::from("Memory"), value: format!("{} MiB of {} MiB in use", used >> 20, total >> 20), tone: super::ui::Tone::Plain },
    ]
}
fn sec_storage() -> Vec<Widget> {
    let mounted = crate::store::mounted();
    let unlocked = crate::dev::nvme::writes_unlocked();
    alloc::vec![
        Widget::Status {
            name: String::from("Store"),
            value: String::from(if mounted { "mounted" } else { "none -- nothing outlives a reboot" }),
            tone: if mounted { super::ui::Tone::Ok } else { super::ui::Tone::Warn },
        },
        Widget::Status {
            name: String::from("Writing"),
            value: String::from(if unlocked { "unlocked for this boot" } else { "locked -- type 'store unlock' to allow it" }),
            tone: if unlocked { super::ui::Tone::Ok } else { super::ui::Tone::Plain },
        },
    ]
}
fn sec_model() -> Vec<Widget> {
    let ready = crate::ai::engine_ready();
    alloc::vec![Widget::Status {
        name: String::from("Model"),
        value: String::from(if ready { "loaded and running in the kernel" } else { "none loaded -- stage model.bin on the boot volume" }),
        tone: if ready { super::ui::Tone::Ok } else { super::ui::Tone::Warn },
    }]
}
fn sec_linux() -> Vec<Widget> {
    let root = crate::linux::disk::claims("/usr");
    alloc::vec![Widget::Status {
        name: String::from("Root image"),
        value: String::from(if root { "mounted at /" } else { "not mounted -- 'linux mount / <partition>'" }),
        tone: if root { super::ui::Tone::Ok } else { super::ui::Tone::Warn },
    }]
}

// ---- the table -------------------------------------------------------------

pub static ITEMS: &[Item] = &[
    // Network
    item("network", "Connection", "Status", "", "ip address gateway dns dhcp ethernet", Ctl::Section(sec_network)),
    item("network", "Connection", "Network manager", "Connections, Wi-Fi and addresses in one window.", "wifi join", Ctl::Open("network")),
    // Wi-Fi
    item("wifi", "Wireless", "Wi-Fi", "", "wireless ssid scan join", Ctl::Section(sec_wifi)),
    item("wifi", "Wireless", "Join a network", "Scan, pick a network and type its passphrase.", "connect password", Ctl::Open("network")),
    // Customization
    item("custom", "Size", "Size of everything", "Text, windows and controls together.", "scale zoom dpi bigger smaller font",
        Ctl::Slider { min: 75, max: 250, step: 5, unit: "%", get: scale, cmd: |v| format!("desk scale {}", v) }),
    item("custom", "Glass", "Caption glass", "The tint of a focused window's title bar.", "colour color theme title",
        Ctl::Choice { opts: GLASS, get: glass }),
    item("custom", "Glass", "Blur behind glass", "Glass shows the wallpaper blurred through it.", "transparency effects",
        Ctl::Toggle { get: blur, on: "desk blur on", off: "desk blur off" }),
    item("custom", "Windows", "Focus border", "The colour round the window you are using.", "ring outline accent",
        Ctl::Choice { opts: RING, get: ring }),
    item("custom", "Windows", "Rounded corners", "How round a window's corners are.", "radius round",
        Ctl::Slider { min: 0, max: 16, step: 1, unit: "", get: round, cmd: |v| format!("desk round {}", v) }),
    item("custom", "Windows", "Dim other windows", "How far windows you are not using fade back.", "dim darken unfocused",
        Ctl::Slider { min: 0, max: 60, step: 5, unit: "%", get: dim, cmd: |v| format!("desk dim {}", v) }),
    // Desktop
    quick("desktop", "Windows", "Tile new windows", "New windows take a place in the layout instead of floating.", "tiling layout float",
        Ctl::Toggle { get: tile, on: "desk tile on", off: "desk tile off" }),
    item("desktop", "Windows", "Space between windows", "The gap between tiled windows.", "gaps inner padding",
        Ctl::Slider { min: 0, max: 40, step: 1, unit: " px", get: gap_in, cmd: |v| format!("desk gaps {} {}", v, gap_out()) }),
    item("desktop", "Windows", "Space round the edge", "The gap between tiled windows and the screen's edge.", "gaps outer margin",
        Ctl::Slider { min: 0, max: 80, step: 2, unit: " px", get: gap_out, cmd: |v| format!("desk gaps {} {}", gap_in(), v) }),
    quick("desktop", "Motion", "Animations", "Windows open, close and move with motion.", "animation motion effects reduce",
        Ctl::Toggle { get: motion_on, on: "desk motion 100", off: "desk motion 0" }),
    item("desktop", "Motion", "Animation speed", "How long motion takes: more is slower.", "duration slow fast",
        Ctl::Slider { min: 25, max: 400, step: 25, unit: "%", get: motion, cmd: |v| format!("desk motion {}", v) }),
    item("desktop", "Motion", "Effects when busy", "Whether blur and motion are shed when the machine is slow.", "quality performance governor",
        Ctl::Choice { opts: QUALITY, get: quality }),
    item("desktop", "Pointer", "Pointer speed", "How fast the pointer moves; quick movements go further.", "mouse touchpad sensitivity",
        Ctl::Slider { min: 25, max: 400, step: 25, unit: "%", get: pointer, cmd: |v| format!("desk pointer {}", v) }),
    // Sound
    quick("sound", "Volume", "Volume", "The laptop's volume keys set the same level.", "loud quiet mute speaker",
        Ctl::Slider { min: 0, max: 100, step: 5, unit: "%", get: volume, cmd: |v| format!("volume {}", v) }),
    quick("sound", "Voice", "Speak replies", "The assistant's answers are spoken aloud.", "voice speech tts talk",
        Ctl::Toggle { get: voice, on: "voice on", off: "voice off" }),
    item("sound", "Voice", "About the voice", "", "kokoro synthesiser", Ctl::Section(sec_voice)),
    item("sound", "Voice", "Say something", "Hear the voice at this volume.", "test", Ctl::Button("voice say The volume is set. This is how loud I am.")),
    item("sound", "Voice", "Stop talking", "Ends whatever it is saying now.", "silence", Ctl::Button("voice stop")),
    // Power
    quick("power", "Speed", "Power mode", "Performance runs hot and fast; Power saver is cool and quiet.", "governor cpu performance battery saver fan",
        Ctl::Choice { opts: GOVERNOR, get: governor }),
    Item { page: "power", group: "Speed", label: "Turbo boost", help: "Short bursts above the base clock.", words: "turbo boost clock",
        ctl: Ctl::Toggle { get: turbo, on: "power turbo on", off: "power turbo off" }, quick: false, when: Some(turbo_known) },
    item("power", "Battery", "Battery and temperature", "", "charge health temperature clock", Ctl::Section(sec_power)),
    // Assistant
    quick("assistant", "On its own", "Act on its own", "The resident mind sets itself small goals while you are away.", "initiative autonomy mind idle",
        Ctl::Toggle { get: initiative, on: "initiative on", off: "initiative off" }),
    item("assistant", "On its own", "Improve itself at night", "Between 02:00 and 06:00, when nobody is at the machine, it tries changes and keeps only those its judges accept.", "godel self improvement night",
        Ctl::Toggle { get: godel, on: "godel on", off: "godel off" }),
    item("assistant", "On its own", "Learn from corrections", "When you redo what it chose, the choice you made is remembered.", "learn correction feedback",
        Ctl::Toggle { get: learn, on: "learn on", off: "learn off" }),
    item("assistant", "On its own", "Stop what it is doing", "Ends a running episode or piece of writing.", "agent stop cancel", Ctl::Button("agent stop")),
    item("assistant", "History", "What it has done", "The ledger of its own changes.", "ledger history godel", Ctl::Button("godel ledger 10")),
    // Model
    item("model", "Model", "Model", "", "qwen llm language", Ctl::Section(sec_model)),
    item("model", "Model", "Model status", "What is loaded and how fast it runs.", "speed tokens", Ctl::Button("win open status")),
    item("model", "Model", "Retrain the router", "Fit the router again on everything it has been taught.", "fit probe routing", Ctl::Button("fit")),
    item("model", "Model", "Conversation memory", "The saved contexts.", "context ctx", Ctl::Button("ctx")),
    // Storage
    item("storage", "Store", "Store", "", "disk nvme unlock", Ctl::Section(sec_storage)),
    item("storage", "Snapshots", "Snapshot automatically", "Changes are kept on disk without being asked.", "autosnap backup save",
        Ctl::Toggle { get: autosnap, on: "autosnap on 60", off: "autosnap off" }),
    item("storage", "Snapshots", "How often", "Seconds between automatic snapshots.", "autosnap interval",
        Ctl::Slider { min: 15, max: 600, step: 15, unit: " s", get: autosnap_every, cmd: |v| format!("autosnap on {}", v) }),
    item("storage", "Snapshots", "Snapshot now", "Keep everything as it stands.", "save commit", Ctl::Button("snap")),
    item("storage", "Snapshots", "Snapshots", "Every snapshot kept.", "list history restore", Ctl::Button("snaps")),
    item("storage", "Disk", "Storage", "What is on the disk and how much.", "usage space", Ctl::Button("win open storage")),
    // Linux programs
    item("linux", "Root image", "Linux", "", "root image erofs mount", Ctl::Section(sec_linux)),
    item("linux", "Packages", "Check for package updates", "Fetch the package lists.", "orange pacman sync update", Ctl::Button("orange update")),
    item("linux", "Packages", "Installed packages", "Everything installed.", "orange pacman list", Ctl::Button("orange list")),
    item("linux", "Programs", "Cores for Linux programs", "Where Linux programs may run.", "cpu cores spread smp",
        Ctl::Choice { opts: CORES, get: cores }),
    item("linux", "Programs", "Programs on the desktop", "Installed applications.", "apps launch icons", Ctl::Button("apps")),
    item("linux", "Programs", "Background services", "What is running and its log.", "services dbus daemon", Ctl::Button("services")),
    // Hardware
    item("hardware", "Devices", "Devices", "", "device manager driver pci usb", Ctl::Section(sec_hardware)),
    // Updates
    item("updates", "Updates", "Updates", "", "version upgrade channel", Ctl::Section(sec_updates)),
    // System
    item("system", "About", "This machine", "", "version memory uptime about", Ctl::Section(sec_system)),
    item("system", "About", "Running tasks", "What the kernel is doing.", "tasks processes", Ctl::Button("win open tasks")),
    item("system", "Power", "Restart", "Start the machine again.", "reboot", Ctl::Button("reboot")),
    item("system", "Power", "Shut down", "Turn the machine off.", "power off halt", Ctl::Button("shutdown")),
];

/// Items on a page, in order, the conditional ones only when they apply.
pub fn on_page(id: &str) -> impl Iterator<Item = &'static Item> + '_ {
    ITEMS.iter().filter(move |i| i.page == id && i.when.map_or(true, |f| f()))
}

/// Settings matching a query: every word must appear in the label, the help,
/// the search words, or the page's title. Best first: a label match before a
/// help match.
pub fn search(q: &str) -> Vec<&'static Item> {
    let q = q.to_ascii_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<(u32, &'static Item)> = Vec::new();
    for it in ITEMS.iter().filter(|i| i.when.map_or(true, |f| f())) {
        let title = page(it.page).map(|p| p.title).unwrap_or("");
        let hay = format!("{} {} {} {}", it.label, it.help, it.words, title).to_ascii_lowercase();
        if words.iter().all(|w| hay.contains(w)) {
            let label = it.label.to_ascii_lowercase();
            let score = words.iter().filter(|w| label.contains(*w)).count() as u32;
            hits.push((score, it));
        }
    }
    hits.sort_by(|a, b| b.0.cmp(&a.0));
    hits.into_iter().map(|(_, i)| i).collect()
}

/// What `diag` can check without drawing anything.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    out.push(("every setting names a page that exists", ITEMS.iter().all(|i| page(i.page).is_some())));
    out.push(("every page has something on it", PAGES.iter().all(|p| ITEMS.iter().any(|i| i.page == p.id))));
    out.push((
        "every old route lands on a page",
        ["net", "wifi", "model", "sys", "power", "updates", "desk", "sound"].iter().all(|r| page(from_route(r)).is_some()),
    ));
    out.push(("search finds a setting by a word it does not show", search("dpi").iter().any(|i| i.label == "Size of everything")));
    out.push(("and every word of a query has to match", search("volume zebra").is_empty()));
    out.push((
        "a slider's range holds its step",
        ITEMS.iter().all(|i| match i.ctl {
            Ctl::Slider { min, max, step, .. } => step > 0 && max > min && (max - min) % step == 0,
            _ => true,
        }),
    ));
    out
}

/// `diag prefs`: the registry, and the quick panel's layout over it.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claims = checks();
    // The quick panel is laid out by one function for both paint and press,
    // so what is checked here is what a click lands on.
    let panel = super::theme::Rect::new(100, 100, 360, 600);
    let hits = super::quick::layout(panel);
    let inside = |r: &super::theme::Rect| r.x >= panel.x && r.y >= panel.y && r.x + r.w <= panel.x + panel.w;
    claims.push(("every quick control is inside its panel", hits.iter().all(|(r, _)| inside(r))));
    let mut apart = true;
    for (i, (a, _)) in hits.iter().enumerate() {
        for (b, _) in hits.iter().skip(i + 1) {
            apart &= a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
        }
    }
    claims.push(("no two quick controls overlap", apart));
    let quick = ITEMS.iter().enumerate().filter(|(_, i)| i.quick && i.when.map(|f| f()).unwrap_or(true));
    let mut all = true;
    for (k, it) in quick {
        all &= hits.iter().any(|(_, h)| match *h {
            super::quick::Hit::Toggle(i) | super::quick::Hit::Slider(i, _) | super::quick::Hit::Choice(i, _) => i == k,
            super::quick::Hit::All => false,
        });
        let _ = it;
    }
    claims.push(("every quick setting is on the panel", all));
    claims.push(("and the panel leads to the rest", hits.iter().any(|(_, h)| *h == super::quick::Hit::All)));
    for (what, good) in claims {
        if !good {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    }
    ok
}
