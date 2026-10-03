//! Wireless: which part is fitted, and what is left to do for it.
//!
//! **This file used to say "there is no 802.11 stack here" and list the stack
//! among the costs still to be paid.** That was true when it was written and
//! stopped being true three commits ago. It is recorded rather than quietly
//! edited because it is the exact failure this project keeps meeting: a note
//! describing what the code *was* reads identically to one describing what it
//! *is*, and the next reader is sent off to build something that exists.
//!
//! ### What exists, and it is everything above the part
//!
//! | | |
//! |---|---|
//! | `dev::radio` | the seam: 802.11 frames in and out, and the channel plan |
//! | `net::softmac` | Ethernet over 802.11, sequence numbers, CCMP in software |
//! | `net::mlme` | scan, authenticate, associate, four-way, install keys |
//! | `net::ccmp` | the link cipher: masks, packet numbers, replay |
//! | `crypto::ccm` | AES-CCM, against RFC 3610 |
//! | `net::wpa2` | the handshake, both halves of it |
//!
//! All of it is chip independent and all of it is asserted at boot against a
//! loopback radio and a fake access point -- `diag radio`, `diag softmac`,
//! `diag ccmp`, `diag ccm`, `diag mlme` -- with nothing plugged in. So what a
//! new part costs is one `impl Radio`: start it, tune it, move a frame, and
//! say in `Caps` whether the host or the firmware runs everything else.
//!
//! ### What is missing is per part, and the registry names it per part
//!
//! `dev::registry` is where to look rather than here, because it is one table
//! beside the ids it matches and this would be a second copy that drifts. The
//! shape of the answers:
//!
//!   * **Firmware.** Intel's AX-series will not initialise without a signed
//!     blob -- a megabyte and a half, loaded over a bootstrap protocol before
//!     the part does anything. Redistributable in binary form and not
//!     modifiable, and the sequence differs between families; `dev::iwx`
//!     takes the AX210 family's to ALIVE.
//!   * **A host command interface.** For a FullMAC part, not descriptor rings
//!     and registers but an asynchronous command and response protocol with
//!     the firmware, in its own versioned message formats. Such a part
//!     implements `Nic` directly and skips `softmac` entirely, which is what
//!     `Caps::softmac` exists to say.
//!   * **Not an undocumented bus**, which this said about CNVi for a while.
//!     The link between the chipset and the radio module is unpublished and
//!     the host never speaks it: from here a CNVi part is an ordinary PCIe
//!     function, and this machine's own is one.
//!
//! `ath9k`-class Atheros parts are the tractable case and the registry says so
//! on the row: no blob at all, and SoftMAC, so everything above them is
//! already written and already checked.


/// PCI class 0x02 is a network controller; subclass 0x80 is "other", which is
/// where essentially every wireless card lands. Ethernet is subclass 0x00.
const CLASS_NETWORK: u8 = 0x02;
const SUBCLASS_OTHER: u8 = 0x80;

pub enum Probe {
    /// No wireless-looking device on the bus. This is the QEMU answer.
    None,
    /// Something is there and nothing here can drive it.
    Unsupported {
        vendor: u16,
        device: u16,
        what: &'static str,
    },
}

pub fn probe(ecam: u64) -> Probe {
    use crate::dev::registry::{self, Role};
    if !registry::scanned() {
        registry::scan_pci(ecam);
    }
    // The first wireless part the registry knows about. It used to be the
    // first PCI function of class 02:80, with a `describe` beside it holding a
    // second copy of the same vendor table -- which is how a machine ends up
    // with two files that disagree about what is fitted.
    for n in registry::nodes() {
        let Some(e) = n.entry else { continue };
        if e.role != Role::Wireless {
            continue;
        }
        return Probe::Unsupported { vendor: n.id.vendor, device: n.id.device, what: e.what };
    }
    Probe::None
}

/// Every piece of networking hardware on the machine, and what drives it.
///
/// The probe below answers one question -- is there a wireless part -- and
/// stops at the first thing it finds. That was enough while wireless was the
/// only open question, and it is not enough to describe a machine: a laptop
/// with a wired card, a wireless card and a dongle plugged in has three, only
/// one of them is carrying traffic, and an operator asking why they have no
/// network needs to see all three and which is which.
///
/// PCI class 0x02 is the network controller class. Subclass 0x00 is ethernet
/// and 0x80 is "other", where essentially every wireless part lands, so both
/// are collected rather than filtering to the one this file used to care
/// about.
pub struct Hardware {
    pub vendor: u16,
    pub device: u16,
    /// Where it lives, for an operator who has to go and unplug it.
    pub bus: &'static str,
    pub what: alloc::string::String,
    /// The driver in this tree that claims this part, if one does. `None` is
    /// the honest answer for hardware we can see and cannot use, which is most
    /// of the wireless in this machine.
    pub driver: Option<&'static str>,
    /// Why there is no driver, or what the driver there is cannot do yet.
    ///
    /// A driver name alone was not enough once the registry started telling
    /// the two apart: `rtl8188eu` claims the dongle and cannot carry a frame,
    /// so a row showing a driver and nothing else reads as a part that works.
    pub gap: Option<&'static str>,
}

/// Ethernet parts this tree can actually drive, by id.
///
/// Duplicated from the two drivers on purpose. The alternative is each driver
/// exporting its id list, and a probe that says "supported" while the driver
/// that would claim it fails for another reason is worse than a short list
/// that is easy to check against the two `SUPPORTED` arrays it mirrors.
fn ethernet_driver(vendor: u16, device: u16) -> Option<&'static str> {
    match vendor {
        0x8086 if [0x100E, 0x1533, 0x10D3, 0x153A].contains(&device) => Some("e1000"),
        0x10EC if [0x8168, 0x8161, 0x8167, 0x8136].contains(&device) => Some("rtl8168"),
        _ => None,
    }
}

fn describe_ethernet(vendor: u16, device: u16) -> &'static str {
    match (vendor, device) {
        (0x8086, 0x100E) => "Intel 82540EM gigabit (QEMU's e1000)",
        (0x8086, _) => "Intel gigabit ethernet",
        // The GF63's wired port. 8168 and 8111 are the same silicon under two
        // marketing names, which is why the id list and not the name matches.
        (0x10EC, 0x8168) => "Realtek RTL8168/8111 gigabit",
        (0x10EC, _) => "Realtek ethernet",
        _ => "ethernet controller",
    }
}

/// Walk both buses and describe everything that carries packets.
pub fn hardware() -> alloc::vec::Vec<Hardware> {
    use crate::dev::registry::{self, Role, Support};
    let mut out = alloc::vec::Vec::new();
    for n in registry::nodes() {
        let role = n.entry.map(|e| e.role);
        // Network-ish by role where a row claims it, and by PCI class where
        // none does. The second half is what stops a card nobody has heard of
        // dropping out of the report entirely: token ring, FDDI and whatever
        // else class 0x02 covers are still facts about the machine, and hiding
        // one is how a report starts lying by omission.
        let networky = matches!(role, Some(Role::Ethernet) | Some(Role::Wireless))
            || (n.id.bus == registry::Bus::Pci && n.id.class == CLASS_NETWORK);
        if !networky {
            continue;
        }
        out.push(Hardware {
            vendor: n.id.vendor,
            device: n.id.device,
            bus: n.id.bus.name(),
            what: n.what(),
            driver: n.entry.and_then(|e| e.support.driver()),
            gap: n.entry.and_then(|e| match &e.support {
                Support::Driver(_) => None,
                Support::Partial(_, why) => Some(*why),
                Support::Known(why) => Some(*why),
            }),
        });
    }
    out
}

/// One network as a scan reports it.
///
/// The display shape of `mlme::Bss`, and `from_bss` is the one conversion so
/// the two cannot drift into disagreeing about what a network is called.
pub struct Network {
    /// The network's name. **SSID and ESSID are the same field**; ESSID is the
    /// older name for it, from when a distinction between independent and
    /// infrastructure networks was still being drawn. Showing both would be
    /// showing one thing twice under two labels, so this shows one.
    pub ssid: alloc::string::String,
    /// The access point's own address, which is what BSSID means. This is the
    /// one field that tells two access points carrying the same network apart,
    /// so it is the thing to look at when a laptop keeps joining the far one.
    pub bssid: crate::net::Mac,
    pub channel: u8,
    /// dBm, as the radio reports it. Negative, closer to zero is stronger.
    pub rssi: i16,
    /// False for an open network, which the UI has to say out loud.
    pub secured: bool,
    /// An RSN element was present, so WPA2 or later. Without it, `secured`
    /// means WEP, which is a different thing wearing the same word.
    pub rsn: bool,
}

impl Network {
    pub fn from_bss(b: &crate::net::mlme::Bss) -> Network {
        Network {
            ssid: b.ssid.clone(),
            bssid: b.bssid,
            channel: b.channel,
            rssi: b.rssi as i16,
            secured: b.secured,
            rsn: b.rsn,
        }
    }

    /// What the security actually is, in the three words that differ.
    ///
    /// **"Secured" is not one state.** An open network and a WEP network are
    /// both things this machine can join and neither is protected; WEP has
    /// been broken since 2001 and a list that calls it secured is telling the
    /// operator the opposite of what is true.
    pub fn security(&self) -> &'static str {
        match (self.secured, self.rsn) {
            (false, _) => "open",
            (true, false) => "WEP (broken)",
            (true, true) => "WPA2-CCMP",
        }
    }

    pub fn band(&self) -> &'static str {
        match crate::dev::radio::band_of(self.channel) {
            Some(crate::dev::radio::Band::G24) => "2.4",
            Some(crate::dev::radio::Band::G5) => "5",
            None => "?",
        }
    }

    /// `02:00:00:00:00:aa`, which is how everybody writes one.
    pub fn ap(&self) -> alloc::string::String {
        let b = &self.bssid;
        alloc::format!(
            "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5]
        )
    }
}

/// Signal as a count out of four, the way every operator already reads it.
pub fn bars(rssi: i16) -> u8 {
    match rssi {
        r if r >= -55 => 4,
        r if r >= -67 => 3,
        r if r >= -75 => 2,
        r if r >= -85 => 1,
        _ => 0,
    }
}

/// The wireless adapter this machine has, wherever it lives.
///
/// PCI and USB are separate questions with separate answers. A dongle is
/// behind the xHCI controller, which appears on PCI as a USB controller and
/// not as a network device, so a PCI walk looking for class 02:80 will never
/// see it. Asking only PCI and reporting "no wireless controller" is how a
/// machine with an adapter plugged into it gets told it has none.
pub enum Adapter {
    /// A USB device the id list recognises.
    Usb { vendor: u16, device: u16, what: &'static str },
    /// Something on PCI, which nothing here can drive.
    Pci { vendor: u16, device: u16, what: &'static str },
    /// Both buses have been looked at and neither has one.
    None,
    /// PCI has none, and USB has not been enumerated, so it is not known.
    /// Distinct from `None` on purpose: an unasked question is not a no.
    PciOnlyChecked,
}

pub fn adapter() -> Adapter {
    // USB first. It is the bus something usable could actually be on, and a
    // dongle is the answer to the PCI part being undriveable.
    if let Some(Some((vendor, device, what))) = crate::dev::xhci::usb_wireless() {
        return Adapter::Usb { vendor, device, what };
    }
    let pci = crate::net::ecam().map(probe);
    if let Some(Probe::Unsupported { vendor, device, what }) = pci {
        return Adapter::Pci { vendor, device, what };
    }
    match crate::dev::xhci::usb_wireless() {
        Some(None) => Adapter::None,
        None => Adapter::PciOnlyChecked,
        _ => Adapter::None,
    }
}

/// Ask the wireless hardware what it can hear.
///
/// The error arm is the whole point of this function today. An operator who
/// opens a wireless page and sees an empty list concludes their router is off;
/// one who sees why there is no list can act on it. Every arm is a real
/// answer, and none of them is an empty list standing in for a missing driver.
pub fn scan() -> Result<alloc::vec::Vec<Network>, &'static str> {
    // A radio that is actually installed answers for itself, and an empty list
    // from one is a real answer -- "nothing in range" -- rather than the
    // missing-driver case this function's other arms exist to name. The
    // difference matters because the two need opposite next steps, and a page
    // that showed one empty list for both is what this module opens by
    // refusing to do.
    {
        let _claim = crate::net::claim_wifi();
        if let Some(w) = crate::net::wlan() {
            return Ok(w.networks());
        }
    }
    match adapter() {
        // The RTL8188EU driver that made this arm say "its MAC can be brought
        // up" was removed with its hardware, so a recognised USB adapter is
        // now exactly as drivable as a recognised PCI one.
        Adapter::Usb { .. } => Err(
            "This USB adapter is recognised and nothing here drives it.",
        ),
        Adapter::Pci { .. } => Err(
            "The wireless part in this machine cannot be driven. A supported USB adapter is the way on to a wireless network here.",
        ),
        Adapter::None => Err(
            "No wireless adapter on either the PCI bus or USB. A wired connection or a supported USB adapter is the way on to a network.",
        ),
        Adapter::PciOnlyChecked => Err(
            "No wireless controller on the PCI bus. USB has not been enumerated yet, so a plugged-in adapter would not have been seen: run the USB scan below.",
        ),
    }
}

/// Print what is known, and what it would take.
pub fn report() {
    use crate::gfx::console::{self, LTGRAY, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("[wlan0]");
    console::set_color(LTGRAY);

    // What the boot's driver pass found, if a driver claimed anything. It has
    // read the part's own registers, which the registry row cannot have.
    let seen = crate::net::wireless::last();
    if !seen.is_empty() {
        crate::net::wireless::report(&seen);
        return;
    }

    match super::ecam().map(probe) {
        None => kprintln!("  no ECAM window, so the bus cannot be enumerated"),
        Some(Probe::None) => {
            kprintln!("  no wireless controller on the bus");
            kprintln!("  QEMU emulates none, so this is the expected answer here --");
            kprintln!("  the question can only be settled on the GF63.");
        }
        Some(Probe::Unsupported { vendor, device, what }) => {
            kprintln!("  {}", what);
            kprintln!("  pci {:04x}:{:04x}", vendor, device);
            // The reason comes off the registry row rather than out of a
            // sentence here, so there is one place that says why a part is
            // undriven and it is the place that matches the ids.
            for h in hardware() {
                if h.vendor == vendor && h.device == device {
                    if let Some(gap) = h.gap {
                        kprintln!("  no driver: {}", gap);
                    }
                }
            }
        }
    }

    // What is missing is the part, and saying so is the point of these lines.
    // An operator told only "no wireless" cannot tell a machine with no stack
    // from a machine with no radio, and those want completely different next
    // steps -- one is months of work and the other is a dongle.
    kprintln!("  everything above the radio is written and checked with no");
    kprintln!("  hardware at all: diag radio, softmac, ccmp, ccm, mlme.");
    kprintln!("  a new part is one impl of dev::radio::Radio.");
}
