//! The device manager: every device this machine has, in one list, sorted
//! the way a person looks for one.
//!
//! The registry answers which row of the support table describes a device
//! and nothing else -- it never says a device works, only that a driver
//! claims it. Whether the touchpad is delivering packets, whether a USB mouse
//! is still on the bus, whether the disk answered, whether there is a battery
//! at all: each of those lived in its own driver behind its own verb. This is
//! the one place that asks all of them, so "is it working" has one answer
//! and one screen.
//!
//! **Built fresh on every ask, never kept.** A cached inventory is the thing
//! that goes stale -- the registry's USB list already outlived devices that
//! had been unplugged -- and every source here is cheap to read: config space
//! was swept at boot, the rest are counters and flags. `generation` moves
//! when something comes or goes, so a window can redraw without polling the
//! whole list.
//!
//! Two decisions are pure and asserted at boot (`diag devmgr`): which
//! category a registry role belongs in, and whether a device is inside the
//! machine or plugged into it.

use super::registry::{self, Bus, Role, Support, Where};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The groups, in the order they are shown -- what a person opens the list
/// to find comes first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Category {
    Display,
    Audio,
    Network,
    Storage,
    Input,
    Usb,
    Power,
    System,
    Other,
}

impl Category {
    pub const ALL: [Category; 9] = [
        Category::Display,
        Category::Audio,
        Category::Network,
        Category::Storage,
        Category::Input,
        Category::Usb,
        Category::Power,
        Category::System,
        Category::Other,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Category::Display => "Display",
            Category::Audio => "Audio",
            Category::Network => "Network",
            Category::Storage => "Storage",
            Category::Input => "Keyboards and pointers",
            Category::Usb => "USB controllers",
            Category::Power => "Power",
            Category::System => "System",
            Category::Other => "Other",
        }
    }
}

/// Inside the machine, or plugged into it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    Internal,
    External,
}

impl Place {
    pub fn name(self) -> &'static str {
        match self {
            Place::Internal => "internal",
            Place::External => "external",
        }
    }
}

/// How a device stands, worst last.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Status {
    /// A driver has it and it is doing its job.
    Working,
    /// A driver has it, and nothing has shown yet whether it works -- a
    /// keyboard nobody has pressed is not known to be broken.
    Ready,
    /// Driven in part; the reason says which part is missing.
    Partial(String),
    /// Recognised, with nothing behind it; the reason says why.
    NoDriver(String),
    /// A driver has it and it is failing.
    Failing(String),
}

impl Status {
    pub fn word(&self) -> &'static str {
        match self {
            Status::Working => "working",
            Status::Ready => "ready",
            Status::Partial(_) => "partial",
            Status::NoDriver(_) => "no driver",
            Status::Failing(_) => "failing",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Status::Partial(r) | Status::NoDriver(r) | Status::Failing(r) => Some(r),
            _ => None,
        }
    }
}

/// One device, as the manager shows it.
#[derive(Clone, Debug)]
pub struct Device {
    pub category: Category,
    pub place: Place,
    pub name: String,
    /// Where to find it: a PCI address, a USB port, "i8042".
    pub location: String,
    pub driver: Option<&'static str>,
    pub status: Status,
    /// One line of whatever the driver can say: a size, a charge, a count.
    pub detail: String,
}

static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Moves whenever a device arrives or leaves.
pub fn generation() -> u64 {
    GENERATION.load(Relaxed)
}

/// A driver saw a device come or go.
pub fn changed() {
    GENERATION.fetch_add(1, Relaxed);
}

/// Where a registry role is filed.
pub fn category_of(role: Role) -> Category {
    match role {
        Role::Display => Category::Display,
        Role::Audio => Category::Audio,
        Role::Ethernet | Role::Wireless | Role::Bluetooth => Category::Network,
        Role::Storage => Category::Storage,
        Role::Input => Category::Input,
        Role::UsbHost => Category::Usb,
        Role::Bridge => Category::System,
        Role::Other => Category::Other,
    }
}

/// Inside or outside, from the bus and what the device is.
///
/// Everything on PCI is soldered or slotted inside the case. USB is the
/// plug-in bus, with one exception that is the rule on laptops: the
/// Bluetooth radio is a USB device on an internal port, and calling it
/// external sends a person looking for a dongle that does not exist.
pub fn place_of(bus: Bus, role: Option<Role>) -> Place {
    match (bus, role) {
        (Bus::Pci, _) => Place::Internal,
        (Bus::Usb, Some(Role::Bluetooth)) => Place::Internal,
        (Bus::Usb, _) => Place::External,
    }
}

/// Everything, sorted by category, then internal before external, then name.
pub fn snapshot() -> Vec<Device> {
    let mut out = Vec::new();
    let hid_ports: Vec<u8> = super::usbhid::live().iter().map(|h| h.port).collect();

    // The registry: PCI from the boot sweep, USB from whoever enumerated.
    for n in registry::nodes() {
        // A USB device the input driver holds is shown from the driver's
        // live list below, which knows whether it is still there.
        if let Where::Usb { port, .. } = n.at {
            if hid_ports.contains(&port) {
                continue;
            }
            // And one the registry saw at enumeration and nobody took is
            // shown only if a row names it: an unnamed class code on a port
            // is a device this machine looked at once and let go.
            if n.entry.is_none() {
                continue;
            }
        }
        let role = n.entry.map(|e| e.role);
        let (driver, status) = match n.entry.map(|e| &e.support) {
            Some(Support::Driver(d)) => (Some(*d), live_status(d)),
            Some(Support::Partial(d, why)) => (Some(*d), Status::Partial(String::from(*why))),
            Some(Support::Known(why)) => (None, Status::NoDriver(String::from(*why))),
            None => (None, Status::NoDriver(String::from("nothing in the table recognises it"))),
        };
        let detail = match driver {
            Some("nvme") => nvme_detail(),
            _ => String::new(),
        };
        out.push(Device {
            category: role.map(category_of).unwrap_or(Category::Other),
            place: place_of(n.id.bus, role),
            name: n.what(),
            location: n.location(),
            driver,
            status,
            detail,
        });
    }

    // The screen, which is the firmware's framebuffer whatever GPU is fitted.
    if let Some(fb) = crate::gfx::primary() {
        out.push(Device {
            category: Category::Display,
            place: Place::Internal,
            name: String::from("Built-in display"),
            location: format!("framebuffer at {:#x}", fb.addr()),
            driver: Some("gop"),
            status: Status::Working,
            detail: format!("{}x{}, drawn by the processor; no GPU driver", fb.width(), fb.height()),
        });
    }

    // The i8042: the built-in keyboard and, on the GF63, the touchpad.
    let (kirqs, _, _, _) = super::kbd::stats();
    out.push(Device {
        category: Category::Input,
        place: Place::Internal,
        name: String::from("Built-in keyboard"),
        location: String::from("i8042 port 1"),
        driver: Some("kbd"),
        status: if kirqs > 0 { Status::Working } else { Status::Ready },
        detail: format!("{} interrupt(s)", kirqs),
    });
    if super::mouse::present() {
        let (irqs, packets, resyncs, stale, rescued) = super::mouse::stats();
        let status = if packets > 0 {
            Status::Working
        } else if irqs > 0 {
            Status::Failing(String::from("bytes arrive and never make a packet"))
        } else {
            Status::Ready
        };
        out.push(Device {
            category: Category::Input,
            place: Place::Internal,
            name: String::from("Built-in touchpad"),
            location: String::from("i8042 port 2"),
            driver: Some("mouse"),
            status,
            detail: format!(
                "{} packet(s), {} resync(s), {} abandoned, {} rescued",
                packets, resyncs, stale, rescued
            ),
        });
    }

    // USB input, from the driver's own list, one row per interface.
    for h in super::usbhid::live() {
        for i in &h.ifaces {
            let kind = if i.mouse { "mouse" } else { "keyboard" };
            let status = if i.failures > i.recovered {
                Status::Failing(format!("{} transfer(s) failed and stayed down", i.failures - i.recovered))
            } else if i.reports > 0 {
                Status::Working
            } else {
                Status::Ready
            };
            out.push(Device {
                category: Category::Input,
                place: Place::External,
                name: format!("USB {} {:04x}:{:04x}", kind, h.vid, h.pid),
                location: format!("usb port {}", h.port),
                driver: Some("usbhid"),
                status,
                detail: format!("{} report(s), {} failed, {} recovered", i.reports, i.failures, i.recovered),
            });
        }
        if h.other > 0 {
            out.push(Device {
                category: Category::Input,
                place: Place::External,
                name: format!("USB {:04x}:{:04x}, other interfaces", h.vid, h.pid),
                location: format!("usb port {}", h.port),
                driver: None,
                status: Status::NoDriver(format!(
                    "{} HID interface(s) with no boot form; a report-descriptor parser would drive them",
                    h.other
                )),
                detail: String::new(),
            });
        }
    }

    // The battery, if ACPI says there is one.
    if let Some(c) = super::battery::status() {
        if c.present {
            use super::battery::State;
            out.push(Device {
                category: Category::Power,
                place: Place::Internal,
                name: String::from("Battery"),
                location: String::from("ACPI"),
                driver: Some("battery"),
                status: Status::Working,
                detail: format!(
                    "{}%, {}",
                    c.percent,
                    match c.state {
                        State::Charging => "charging",
                        State::Discharging => "on battery",
                        State::Idle => "full or idle",
                        State::Unknown => "state unknown",
                    }
                ),
            });
        }
    }

    out.sort_by(|a, b| {
        (a.category, a.place == Place::External, &a.name).cmp(&(b.category, b.place == Place::External, &b.name))
    });
    out
}

/// Whether a device whose row names a driver is actually being driven, where
/// the driver can say. A driver named in the table and never started is
/// `Ready`, not `Working`: the table says what could drive it.
fn live_status(driver: &str) -> Status {
    match driver {
        "nvme" => {
            if super::nvme::present() {
                Status::Working
            } else {
                Status::Failing(String::from("the driver did not bring it up"))
            }
        }
        "xhci" => {
            if super::xhci::started() {
                Status::Working
            } else {
                Status::Ready
            }
        }
        "e1000" | "rtl8168" | "usb-ecm" | "rndis" => {
            let up = crate::net::ifaces().iter().any(|i| !i.wireless && i.nic.is_some() && i.name != "lo");
            if up { Status::Working } else { Status::Ready }
        }
        "iwx" => {
            if crate::net::wlan_name().is_some() {
                Status::Working
            } else {
                Status::Ready
            }
        }
        _ => Status::Ready,
    }
}

fn nvme_detail() -> String {
    super::nvme::with(|n| {
        let model = core::str::from_utf8(&n.model).unwrap_or("").trim();
        let bytes = n.block_count.saturating_mul(n.block_size as u64);
        // GB, as drives are sold; MB below ten of them, or a small disk
        // reads as "0 GB" and looks like a disk that is not there.
        if bytes >= 10_000_000_000 {
            format!("{}, {} GB", model, bytes / 1_000_000_000)
        } else {
            format!("{}, {} MB", model, bytes / 1_000_000)
        }
    })
    .unwrap_or_default()
}

/// `devices`: the list, grouped, with what is missing at the end.
pub fn report() {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
    use crate::kprintln;
    let list = snapshot();
    for cat in Category::ALL {
        let mut any = false;
        for d in list.iter().filter(|d| d.category == cat) {
            if !any {
                console::set_color(YELLOW);
                kprintln!("  {}", cat.name());
                any = true;
            }
            console::set_color(match d.status {
                Status::Working => LTGREEN,
                Status::Ready => WHITE,
                Status::Partial(_) | Status::NoDriver(_) => LTGRAY,
                Status::Failing(_) => LTRED,
            });
            kprintln!(
                "    {:<10} {:<8} {}  [{}]{}",
                d.status.word(),
                d.place.name(),
                d.name,
                d.location,
                match d.driver {
                    Some(n) => format!("  {}", n),
                    None => String::new(),
                }
            );
            console::set_color(LTGRAY);
            if !d.detail.is_empty() {
                kprintln!("               {}", d.detail);
            }
            if let Some(r) = d.status.reason() {
                kprintln!("               {}", r);
            }
        }
    }
    console::set_color(WHITE);
    let n = list.len();
    let working = list.iter().filter(|d| d.status == Status::Working).count();
    let missing = list.iter().filter(|d| matches!(d.status, Status::NoDriver(_))).count();
    kprintln!("  {} device(s): {} working, {} with no driver", n, working, missing);
}

/// `diag devmgr`: the two decisions, and that a snapshot is well formed.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        ok &= good;
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };
    claim("a PCI device is internal whatever it is", place_of(Bus::Pci, Some(Role::Input)) == Place::Internal);
    claim("a USB mouse is external", place_of(Bus::Usb, Some(Role::Input)) == Place::External);
    claim(
        "the Bluetooth radio is USB and internal, or a person goes looking for a dongle",
        place_of(Bus::Usb, Some(Role::Bluetooth)) == Place::Internal,
    );
    claim("an unrecognised USB device is external", place_of(Bus::Usb, None) == Place::External);
    claim("wireless and wired are both network", category_of(Role::Wireless) == category_of(Role::Ethernet));
    claim("Bluetooth is filed with the radios, not with input", category_of(Role::Bluetooth) == Category::Network);
    claim("a bridge is system plumbing", category_of(Role::Bridge) == Category::System);
    let mut seen = Vec::new();
    for c in Category::ALL {
        if !seen.contains(&c.name()) {
            seen.push(c.name());
        }
    }
    claim("every category has its own name", seen.len() == Category::ALL.len());
    let ordered = Category::ALL.windows(2).all(|w| w[0] < w[1]);
    claim("categories are listed in their sort order", ordered);
    let list = snapshot();
    claim("a snapshot has the keyboard in it", list.iter().any(|d| d.name == "Built-in keyboard"));
    let sorted = list.windows(2).all(|w| w[0].category <= w[1].category);
    claim("a snapshot is sorted by category", sorted);
    let named = list.iter().all(|d| !d.name.is_empty() && !d.location.is_empty());
    claim("every device has a name and a place to find it", named);
    ok
}
