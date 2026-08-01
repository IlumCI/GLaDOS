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
