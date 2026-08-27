//! Wireless: identification now, a driver later.
//!
//! This module deliberately does not pretend. There is no 802.11 stack here
//! and `wlan0` will not carry a packet. What it does is name the card, which
//! is the one thing standing between here and a driver -- and which cannot be
//! done from a QEMU guest at all, because QEMU emulates no wireless hardware.
//! The answer only exists on the GF63.
//!
//! ### What a wireless driver actually costs
//!
//! It is worth writing down, because "add WiFi" sounds like the same size of
//! job as "add Ethernet" and is not. The e1000 was ~350 lines: map a BAR, set
//! up two descriptor rings, poll them. For a modern wireless card:
//!
//!   * **Firmware.** Intel's AX-series will not initialise without a signed
//!     blob -- roughly a megabyte, loaded into the device over a bootstrap
//!     protocol before it does anything. It is not redistributable, it is not
//!     documented, and the loading sequence differs between families. That is
//!     the single largest obstacle, and no amount of writing code avoids it.
//!   * **A host command interface.** Not descriptor rings and registers but an
//!     asynchronous command/response protocol with the firmware, with its own
//!     versioned message formats.
//!   * **802.11 itself.** Scanning, authentication, association, and the fact
//!     that a wireless frame is not an Ethernet frame -- three or four address
//!     fields depending on direction, plus fragmentation and aggregation.
//!   * **WPA2/WPA3.** The four-way handshake, which needs PBKDF2-HMAC-SHA1 for
//!     the pairwise master key, AES key wrap, and CCMP for the data path.
//!     GLaDOS will have most of those primitives once TLS exists, which is the
//!     one part of this that gets cheaper by waiting.
//!
//! So the honest order is: identify the card, then decide whether its firmware
//! situation makes a driver possible at all. An Intel AX201 is a large project
//! with a blob problem. Some Realtek and Atheros parts are considerably more
//! tractable. Until the GF63 boots and prints a vendor and device id, every
//! sentence after this one would be a guess.

use crate::dev::pci;

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

/// Name the vendor, and say what is known about the driver situation.
///
/// The strings are chosen to be useful rather than decorative: whether a
/// driver is plausible depends almost entirely on whether the part needs
/// signed firmware.
fn describe(vendor: u16, device: u16) -> &'static str {
    match vendor {
        // Intel wireless is the likeliest thing in a 2022 MSI laptop, and the
        // hardest case: everything from Wireless-AC onward needs a signed blob.
        0x8086 => match device {
            // Discrete M.2 cards on the PCIe bus.
            0x2723 => "Intel Wi-Fi 6 AX200 (discrete)",
            // CNVi: the MAC and baseband live in the PCH and the M.2 module
            // carries only the radio. Confirmed present on the GF63 as
            // 8086:51f0 at 00:14.3. Worth calling out rather than filing under
            // "Intel wireless": a CNVi part is not a self-contained NIC, so
            // there is no card to drive on its own -- the driver talks to the
            // chipset over an interface Intel does not document, and the
            // firmware blob is still required on top of that.
            0x51f0 | 0x54f0 => "Intel Wi-Fi 6E, CNVi in the PCH (Alder Lake-P)",
            0x02f0 | 0x4df0 | 0xa0f0 => "Intel Wi-Fi 6 AX201, CNVi in the PCH",
            _ => "Intel wireless (firmware blob required)",
        },
        0x10ec => "Realtek wireless",
        0x14e4 => "Broadcom wireless",
        0x168c => "Qualcomm Atheros wireless",
        0x17cb => "Qualcomm wireless",
        0x1814 => "Ralink/MediaTek wireless",
        0x14c3 => "MediaTek wireless",
        _ => "unrecognised wireless controller",
    }
}

pub fn probe(ecam: u64) -> Probe {
    let mut found: Option<(u16, u16)> = None;
    // Every other caller walks all 255 buses, and this one must too. A
    // wireless card sits behind a PCIe root port, so its bus number is
    // assigned by the firmware and is routinely well above 8 on a laptop --
    // stopping early would report "no wireless controller" on a machine that
    // has one, which is the single question this module exists to answer.
    pci::scan(ecam, 255, |d| {
        if d.class == CLASS_NETWORK && d.subclass == SUBCLASS_OTHER && found.is_none() {
            found = Some((d.vendor, d.device));
        }
    });
    match found {
        None => Probe::None,
        Some((vendor, device)) => Probe::Unsupported {
            vendor,
            device,
            what: describe(vendor, device),
        },
    }
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
    let mut out = alloc::vec::Vec::new();
    if let Some(ecam) = crate::net::ecam() {
        pci::scan(ecam, 255, |d| {
            if d.class != CLASS_NETWORK {
                return;
            }
            let (bus, what, driver) = match d.subclass {
                0x00 => (
                    "PCI",
                    alloc::string::String::from(describe_ethernet(d.vendor, d.device)),
                    ethernet_driver(d.vendor, d.device),
                ),
                SUBCLASS_OTHER => (
                    "PCI",
                    alloc::string::String::from(describe(d.vendor, d.device)),
                    None,
                ),
                // Token ring, FDDI, ATM and friends. Named rather than hidden:
                // an unrecognised network device is still a fact about the
                // machine, and hiding it is how a report starts lying by
                // omission.
                _ => (
                    "PCI",
                    alloc::format!("network controller, subclass {:02x}", d.subclass),
                    None,
                ),
            };
            out.push(Hardware { vendor: d.vendor, device: d.device, bus, what, driver });
        });
    }
    // USB, from what the last enumeration recorded. Not enumerated here: doing
    // so resets the controller and drops whatever link is on it.
    if let Some((vendor, device)) = crate::dev::xhci::usb_ethernet() {
        out.push(Hardware {
            vendor,
            device,
            bus: "USB",
            what: alloc::string::String::from("CDC ethernet adapter"),
            driver: Some("usb-ecm"),
        });
    }
    if let Some(Some((vendor, device, what))) = crate::dev::xhci::usb_wireless() {
        out.push(Hardware {
            vendor,
            device,
            bus: "USB",
            what: alloc::string::String::from(what),
            // Recognised, brought up, and unable to carry a frame. Naming the
            // driver here would claim more than is true.
            driver: None,
        });
    }
    out
}

/// One network as a scan would report it.
///
/// Nothing constructs this yet. It is here so the settings page is written
/// against the shape a scan returns rather than against the absence of one,
/// and so the day a driver can associate, the UI above it already works.
pub struct Network {
    pub ssid: alloc::string::String,
    /// dBm, as the radio reports it. Negative, closer to zero is stronger.
    pub rssi: i16,
    /// False for an open network, which the UI has to say out loud.
    pub secured: bool,
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
    match adapter() {
        // The driver powers this chip on and loads its MAC registers, and the
        // PHY, AGC and radio tables are transcribed. What is missing is the
        // rest: the radio tables are not applied yet, no channel is set, and
        // there is no transmit or receive path at all -- not one frame goes
        // out or comes in. Scanning is sending probe requests and reading
        // beacons, so it needs exactly the part that is absent.
        Adapter::Usb { .. } => Err(
            "This adapter is recognised and its MAC can be brought up, but there is no transmit or receive path yet, so no probe request can be sent and no beacon can be read.",
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
            kprintln!("  no driver. see the note at the top of net/wifi.rs for");
            kprintln!("  what one costs -- firmware is the deciding factor.");
        }
    }
}
