//! Which wireless part is fitted, which driver wants it, and how far it gets.
//!
//! **Detection is the registry's job and this is the step after it.**
//! `dev::registry` answers which row describes a device and which driver that
//! row names; this asks each named driver to look at what it was handed and say
//! what it found, in a shape every driver shares. One table, closed, in the
//! idiom of the Ethernet dispatch in `net::init`: a driver is a row here and
//! nothing else in the boot path learns its name.
//!
//! ### Detection is not bring-up, and the boot does only the first
//!
//! Looking costs a guarded register read and grants nothing. Bringing a part up
//! grants it bus-master DMA and hands it a megabyte and a half of firmware to
//! run -- and on a part nothing here has driven to completion that is a boot
//! that may hang on hardware nobody can watch, for a radio that cannot yet carry
//! a frame. So the boot reads, decides, and says what it would do; `iwx boot`
//! does it, by name, until the driver can finish the job. When it can, a row
//! here gains the bring-up and `wlan0` attaches at boot.
//!
//! What the boot found is kept, so `wifi` can say it again on a machine with no
//! serial line, where the boot log scrolled away an hour ago.

use crate::sync::Racy;
use alloc::string::String;
use alloc::vec::Vec;

/// One part, as its driver saw it.
#[derive(Clone)]
pub struct Seen {
    pub driver: &'static str,
    /// `vvvv:dddd at bb:dd.f`.
    pub at: String,
    /// What the part says it is, from its own registers where it could be read.
    pub what: String,
    /// The firmware image it wants and whether one is present, or why it wants
    /// none this kernel can name.
    pub firmware: Result<(String, bool), &'static str>,
    /// How far this kernel can take it, in a sentence.
    pub reach: &'static str,
}

/// A driver's look at the bus. Reads; never grants DMA, never loads firmware.
type Detect = fn(u64) -> Vec<Seen>;

/// Every wireless driver this kernel has, by the name `dev::registry` uses.
const DRIVERS: &[(&str, Detect)] = &[("iwx", detect_iwx)];

static LAST: Racy<Vec<Seen>> = Racy::new(Vec::new());

/// Run every driver the registry names for a wireless part.
pub fn detect(ecam: u64) -> Vec<Seen> {
    use crate::dev::registry::{drivers_for, Role};
    let mut out = Vec::new();
    for name in drivers_for(Role::Wireless) {
        match DRIVERS.iter().find(|(n, _)| *n == name) {
            Some((_, f)) => out.extend(f(ecam)),
            // A row naming a driver this table does not have is a registry
            // edit that outran the code. Said, rather than skipped.
            None => out.push(Seen {
                driver: name,
                at: String::new(),
                what: String::from("a part the registry gives to a driver this kernel does not have"),
                firmware: Err("no driver"),
                reach: "nothing",
            }),
        }
    }
    unsafe { *LAST.get() = out.clone() };
    out
}

/// What the last `detect` found.
pub fn last() -> Vec<Seen> {
    unsafe { (*LAST.get()).clone() }
}

fn detect_iwx(ecam: u64) -> Vec<Seen> {
    use crate::dev::iwx::{self, Family};
    let radios = iwx::find(ecam);
    iwx::note_seen(radios.len());
    let mut out = Vec::new();
    for r in radios {
        let d = r.dev;
        let at = alloc::format!(
            "{:04x}:{:04x} at {:02x}:{:02x}.{}",
            d.vendor, d.device, d.bus, d.dev, d.func
        );
        let read = r.hw_rev(ecam);
        iwx::note_rev(read.map(|(rev, _)| rev));
        let (rev, rf) = match read {
            Ok(x) => x,
            Err(e) => {
                out.push(Seen {
                    driver: "iwx",
                    at,
                    what: String::from("Intel wireless that did not answer"),
                    firmware: Err(e.why()),
                    reach: "nothing until it answers",
                });
                continue;
            }
        };
        let rfid = iwx::rf_of(rf);
        let family = rev.mac.family();
        let what = alloc::format!(
            "Intel {} ({}/{}), {} family",
            iwx::product_name(rev.mac, rfid.rf).unwrap_or("Wi-Fi"),
            rev.mac.name(),
            rfid.rf.name(),
            family.map(|f| f.name()).unwrap_or("unknown")
        );
        let firmware = match iwx::firmware_base(d.device, rev, rfid) {
            None => Err("no firmware image is named for this controller and radio"),
            Some(base) => Ok(match iwx::firmware_for(&base) {
                Some((name, _)) => (name, true),
                None => (alloc::format!("iwlwifi-{}-<api>.ucode", base), false),
            }),
        };
        let reach = match family {
            Some(Family::Ax210) => "boots firmware and scans with `iwx boot`; cannot join yet",
            Some(Family::F22000) => "recognised; this family's boot path is written and not taken",
            Some(Family::Bz) => "recognised and refused: this family's reset is not written",
            None => "a controller type this kernel cannot name",
        };
        out.push(Seen { driver: "iwx", at, what, firmware, reach });
    }
    out
}

/// One line per part, for the boot and for `wifi`.
pub fn report(seen: &[Seen]) {
    use crate::kprintln;
    for s in seen {
        kprintln!("  wlan0  {}  {}", s.what, s.at);
        match &s.firmware {
            Ok((name, true)) => kprintln!("         firmware {} found", name),
            Ok((name, false)) => kprintln!(
                "         wants {} and there is none in {} -- tools/wifi_fw.py stages it",
                name,
                crate::dev::firmware::DIR
            ),
            Err(why) => kprintln!("         {}", why),
        }
        kprintln!("         {} ({})", s.reach, s.driver);
    }
}
