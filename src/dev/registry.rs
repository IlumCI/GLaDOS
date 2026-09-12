//! What is in this machine, and what would drive it.
//!
//! Every driver in this tree used to answer that question for itself: sweep
//! all 256 PCI buses, compare against a private list of ids, take the first
//! hit. Nine drivers, nine sweeps, nine private lists, and **no single place
//! that could say what the machine contains**. That was survivable while there
//! was one machine. It stops being survivable the moment the answer has to be
//! right on a laptop nobody here has ever booted.
//!
//! So the matching rule becomes data. Adding hardware support is adding a row,
//! and -- the part that matters more -- *naming* hardware we cannot support is
//! also adding a row. A machine whose wireless card is an Intel CNVi part
//! deserves to be told that, by name, with the reason there is no driver.
//! Silence reads as "GLaDOS did not look".
//!
//! ### Three levels of support, because there really are three
//!
//! `Driver` means a driver claims it and works. `Known` means we recognise the
//! part and nothing here can drive it. `Partial` is the awkward middle that
//! this tree keeps landing in: the RTL8188EU dongle is identified, its
//! registers are readable, its firmware parses, and it cannot carry a frame.
//! Filing that under `Driver` would be a lie an operator only discovers when
//! the network does not work, and filing it under `Known` would throw away the
//! work. It gets its own level.
//!
//! ### What this deliberately does not do
//!
//! It does not bind. `lookup` answers which row describes a device; whether
//! that driver then initialises is the driver's business and it can still
//! fail for a dozen reasons a table cannot predict. The registry says "this is
//! an e1000 and `e1000` claims it", never "the network works".

use crate::dev::pci;
use crate::sync::Racy;
use alloc::vec::Vec;

/// Which bus a row is about. The same numbers mean different things on each,
/// so a PCI rule must never match a USB device: USB vendor 0x8086 is Intel
/// too, and PCI class 0x03 is a display controller while USB class 0x03 is a
/// keyboard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bus {
    Pci,
    Usb,
}

impl Bus {
    pub fn name(self) -> &'static str {
        match self {
            Bus::Pci => "PCI",
            Bus::Usb => "USB",
        }
    }
}

/// What the device is for, in the terms an operator thinks in.
///
/// Coarser than the PCI class codes on purpose. Someone asking "why is there
/// no network" wants ethernet and wireless in one answer, and does not care
/// that one is subclass 0x00 and the other 0x80.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Storage,
    Ethernet,
    Wireless,
    Bluetooth,
    Display,
    UsbHost,
    Input,
    Audio,
    Bridge,
    Other,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Storage => "storage",
            Role::Ethernet => "ethernet",
            Role::Wireless => "wireless",
            Role::Bluetooth => "bluetooth",
            Role::Display => "display",
            Role::UsbHost => "usb host",
            Role::Input => "input",
            Role::Audio => "audio",
            Role::Bridge => "bridge",
            Role::Other => "other",
        }
    }
}

/// How a row recognises a device.
///
/// Three kinds because hardware identifies itself three ways, and which one
/// applies is a property of the part rather than a choice. NVMe has a
/// programming interface every conforming controller reports, so it is matched
/// by class and nobody has to enumerate the world's SSDs. An RTL8188EU dongle
/// reports a vendor-specific interface and no useful class at all, so the id
/// list *is* the detection. Getting this backwards is how a driver either
/// misses hardware it could drive or claims hardware it cannot.
pub enum Match {
    /// One vendor, these device ids.
    Ids(u16, &'static [u16]),
    /// Any part of this class and subclass, whoever made it.
    Class(u8, u8),
    /// Class, subclass and programming interface: what a controller built to a
    /// standard is identified by.
    Interface(u8, u8, u8),
}

impl Match {
    fn matches(&self, id: &Ident) -> bool {
        match *self {
            Match::Ids(v, list) => id.vendor == v && list.contains(&id.device),
            Match::Class(c, s) => id.class == c && id.subclass == s,
            Match::Interface(c, s, p) => id.class == c && id.subclass == s && id.prog_if == p,
        }
    }

    /// How specific this rule is. A device can satisfy several rows -- an
    /// e1000 is both `Ids(0x8086, ...)` and `Class(0x02, 0x00)` -- and the more
    /// specific one has to win.
    ///
    /// Ranked rather than resolved by table order, which is the point. Order
    /// works right up until somebody inserts a row in the wrong place, and
    /// then a generic "ethernet controller, unrecognised" quietly shadows the
    /// driver that would have worked. This cannot be got wrong by editing.
    fn specificity(&self) -> u8 {
        match self {
            Match::Ids(..) => 3,
            Match::Interface(..) => 2,
            Match::Class(..) => 1,
        }
    }
}

/// What this tree can do with a part.
pub enum Support {
    /// A driver claims it, and that driver carries traffic or data.
    Driver(&'static str),
    /// A driver exists and does not finish the job yet. The second string says
    /// what is missing, because "partial" on its own tells nobody anything.
    Partial(&'static str, &'static str),
    /// Recognised, and nothing here can drive it. The string is the reason,
    /// and it is the most valuable field in this file: "needs signed firmware"
    /// and "nobody has written it yet" are completely different futures.
    Known(&'static str),
}

impl Support {
    /// The driver's name, for the two levels that have one.
    pub fn driver(&self) -> Option<&'static str> {
        match self {
            Support::Driver(n) => Some(n),
            Support::Partial(n, _) => Some(n),
            Support::Known(_) => None,
        }
    }

    /// One word for a column.
    pub fn tag(&self) -> &'static str {
        match self {
            Support::Driver(_) => "driven",
            Support::Partial(..) => "partial",
            Support::Known(_) => "known",
        }
    }
}

/// One row: a rule, what it recognises, and what we can do about it.
pub struct Entry {
    pub bus: Bus,
    pub rule: Match,
    pub role: Role,
    pub what: &'static str,
    pub support: Support,
}

/// A device reduced to the six numbers any rule can ask about.
#[derive(Clone, Copy)]
pub struct Ident {
    pub bus: Bus,
    pub vendor: u16,
    pub device: u16,
    pub class: u8,
    pub subclass: u8,
    pub prog_if: u8,
}

impl Ident {
    pub fn of_pci(d: &pci::Device) -> Ident {
        Ident {
            bus: Bus::Pci,
            vendor: d.vendor,
            device: d.device,
            class: d.class,
            subclass: d.subclass,
            prog_if: d.prog_if,
        }
    }

    /// A USB device whose class comes from its interface descriptor. Zero for
    /// the class fields is what a device that defers to its interfaces
    /// reports, and it matches no `Class` row, which is correct: a composite
    /// device is identified by what its interfaces claim.
    pub fn of_usb(vendor: u16, device: u16, class: u8, subclass: u8, protocol: u8) -> Ident {
        Ident { bus: Bus::Usb, vendor, device, class, subclass, prog_if: protocol }
    }
}

// ---------------------------------------------------------------- the table

/// Intel gigabit parts the `e1000` driver drives. 0x100E is QEMU's emulated
/// card, which is why development has ever worked.
const E1000_IDS: &[u16] = &[0x100E, 0x1533, 0x10D3, 0x153A];

/// Realtek gigabit parts the `rtl8168` driver drives. 8168 and 8111 are one
/// piece of silicon under two marketing names, which is why the id list and
/// not the model name is what matches. This is the GF63's wired port.
const RTL8168_IDS: &[u16] = &[0x8168, 0x8161, 0x8167, 0x8136];

/// Intel wireless that lives in the chipset rather than on the card.
///
/// CNVi splits a wireless NIC in half: the MAC and baseband are in the PCH and
/// the M.2 module carries only the radio. There is no self-contained card to
/// drive, the host-to-chipset interface is undocumented, and a signed firmware
/// blob is required on top of that. Naming these separately from Intel's
/// discrete parts is the difference between "hard" and "not possible from
/// public information".
const INTEL_CNVI_IDS: &[u16] = &[0x51f0, 0x54f0, 0x02f0, 0x4df0, 0xa0f0, 0x7af0, 0x7e40];

/// Intel's discrete M.2 wireless cards, which are whole NICs on the PCIe bus.
/// Still a signed blob, but at least the part is documented as a device.
const INTEL_WIFI_IDS: &[u16] = &[0x2723, 0x2725, 0x2726, 0x272b, 0x24fd, 0x9df0, 0x31dc];

/// RTL8188EU dongles, by every badge they ship under. There is no class code
/// to key off -- the interface is vendor-specific -- so the id list is the
/// entire detection mechanism.
const RTL8188EU_IDS_REALTEK: &[u16] = &[0x8179, 0x0179, 0x8176];
const RTL8188EU_IDS_TPLINK: &[u16] = &[0x010C, 0x0111];

/// Every rule, in no significant order.
///
/// Order is not significant because `lookup` ranks by specificity rather than
/// taking the first hit, so a row may be inserted anywhere. Grouped by role
/// for a reader, which is a different thing from being ordered for a machine.
pub static TABLE: &[Entry] = &[
    // ------------------------------------------------------------- storage
    Entry {
        bus: Bus::Pci,
        rule: Match::Interface(0x01, 0x08, 0x02),
        role: Role::Storage,
        what: "NVMe controller",
        support: Support::Driver("nvme"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Interface(0x01, 0x06, 0x01),
        role: Role::Storage,
        what: "SATA controller in AHCI mode",
        // The largest single hole in this table. A laptop with a SATA SSD and
        // no NVMe has no storage at all here, which means no store, no model,
        // and a boot that reaches a shell and can save nothing. It is also
        // entirely tractable: AHCI is a published specification with no
        // firmware requirement.
        support: Support::Known("no AHCI driver yet, and this is why some machines have no disk"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x01, 0x01),
        role: Role::Storage,
        what: "IDE controller",
        support: Support::Known("legacy ATA, not implemented"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x08, 0x05),
        role: Role::Storage,
        what: "SD/MMC host controller",
        support: Support::Known("not implemented"),
    },
    // ------------------------------------------------------------ ethernet
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x8086, E1000_IDS),
        role: Role::Ethernet,
        what: "Intel gigabit ethernet",
        support: Support::Driver("e1000"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x10EC, RTL8168_IDS),
        role: Role::Ethernet,
        what: "Realtek RTL8168/8111 gigabit",
        support: Support::Driver("rtl8168"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x02, 0x00),
        role: Role::Ethernet,
        what: "ethernet controller",
        support: Support::Known("recognised as ethernet, but not this model"),
    },
    // ------------------------------------------------------------ wireless
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x8086, INTEL_CNVI_IDS),
        role: Role::Wireless,
        what: "Intel Wi-Fi 6/6E, CNVi in the chipset",
        support: Support::Known("the radio is in the PCH over an undocumented interface, plus a signed blob"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x8086, INTEL_WIFI_IDS),
        role: Role::Wireless,
        what: "Intel discrete Wi-Fi card",
        support: Support::Known("iwlwifi, and a signed firmware blob that is not redistributable"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x168c, &[0x0030, 0x0032, 0x0033, 0x0034, 0x003e]),
        role: Role::Wireless,
        what: "Qualcomm Atheros Wi-Fi",
        // Named as the tractable case because it genuinely is: ath9k parts
        // need no firmware at all, which removes the obstacle that stops
        // every Intel part dead.
        support: Support::Known("ath9k-class, and the most tractable wireless here: no blob required"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x14e4, &[0x43a0, 0x43b1, 0x4331, 0x4353, 0x4727]),
        role: Role::Wireless,
        what: "Broadcom Wi-Fi",
        support: Support::Known("brcmfmac, firmware blob required"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x14c3, &[0x7961, 0x7922, 0x0616, 0x7902]),
        role: Role::Wireless,
        what: "MediaTek Wi-Fi",
        support: Support::Known("mt76, firmware blob required"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x10EC, &[0x8852, 0xb852, 0xc852, 0xc822, 0xb822, 0x8723]),
        role: Role::Wireless,
        what: "Realtek Wi-Fi",
        support: Support::Known("rtw88/rtw89, firmware blob required"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x02, 0x80),
        role: Role::Wireless,
        what: "wireless controller",
        support: Support::Known("recognised as wireless, but not this model"),
    },
    // ------------------------------------------------------------- display
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x03, 0x00),
        role: Role::Display,
        what: "VGA-compatible display controller",
        // Not "no driver": the firmware hands over a linear framebuffer and
        // the whole graphics stack draws into it. What is missing is
        // acceleration and mode setting, which is a different sentence.
        support: Support::Partial("gfx", "the firmware framebuffer only, with no mode setting"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x03, 0x02),
        role: Role::Display,
        what: "secondary display controller",
        support: Support::Known("the discrete half of a hybrid laptop, unused"),
    },
    // ------------------------------------------------------------ usb host
    Entry {
        bus: Bus::Pci,
        rule: Match::Interface(0x0C, 0x03, 0x30),
        role: Role::UsbHost,
        what: "xHCI USB 3 controller",
        support: Support::Driver("xhci"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Interface(0x0C, 0x03, 0x20),
        role: Role::UsbHost,
        what: "EHCI USB 2 controller",
        support: Support::Known("no EHCI driver, so USB on a pre-2010 machine is dark"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Interface(0x0C, 0x03, 0x00),
        role: Role::UsbHost,
        what: "UHCI USB 1 controller",
        support: Support::Known("not implemented"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Interface(0x0C, 0x03, 0x10),
        role: Role::UsbHost,
        what: "OHCI USB 1 controller",
        support: Support::Known("not implemented"),
    },
    // ------------------------------------------------- the rest of a laptop
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x04, 0x03),
        role: Role::Audio,
        what: "HD Audio controller",
        support: Support::Known("no audio stack"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x06, 0x04),
        role: Role::Bridge,
        what: "PCI-to-PCI bridge",
        // Listed rather than left unknown because on a laptop these are the
        // most numerous devices on the bus, and a report where half the rows
        // say "unrecognised" trains the reader to stop reading it.
        support: Support::Known("nothing to drive; it forwards a bus"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x06, 0x00),
        role: Role::Bridge,
        what: "host bridge",
        support: Support::Known("nothing to drive"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x06, 0x01),
        role: Role::Bridge,
        what: "ISA bridge",
        support: Support::Known("nothing to drive"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x0C, 0x05),
        role: Role::Other,
        what: "SMBus controller",
        support: Support::Known("not implemented"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Class(0x11, 0x80),
        role: Role::Other,
        what: "signal processing controller",
        support: Support::Known("thermal or sensor hub, not implemented"),
    },
    // ------------------------------------------------------- USB, by device
    Entry {
        bus: Bus::Usb,
        rule: Match::Ids(0x0BDA, RTL8188EU_IDS_REALTEK),
        role: Role::Wireless,
        what: "Realtek RTL8188EU wireless dongle",
        support: Support::Partial("rtl8188eu", "identified and readable; no association and no datapath"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Ids(0x2357, RTL8188EU_IDS_TPLINK),
        role: Role::Wireless,
        what: "TP-Link RTL8188EU wireless dongle",
        support: Support::Partial("rtl8188eu", "identified and readable; no association and no datapath"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0x02, 0x06, 0x00),
        role: Role::Ethernet,
        what: "CDC ethernet adapter",
        support: Support::Driver("usb-ecm"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0x03, 0x01, 0x01),
        role: Role::Input,
        what: "USB keyboard on the boot protocol",
        support: Support::Driver("usbhid"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0x03, 0x01, 0x02),
        role: Role::Input,
        what: "USB mouse on the boot protocol",
        support: Support::Driver("usbhid"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0x08, 0x06, 0x50),
        role: Role::Storage,
        what: "USB mass storage",
        // Worth its own row and its own reason: every part of this except the
        // SCSI command layer already exists, since bulk transfers work.
        support: Support::Known("bulk-only transport is not written, though bulk transfers work"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0xE0, 0x01, 0x01),
        role: Role::Bluetooth,
        what: "Bluetooth radio",
        support: Support::Known("no HCI, no stack"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Class(0x09, 0x00),
        role: Role::Other,
        what: "USB hub",
        support: Support::Known("enumeration does not descend through hubs yet"),
    },
];

/// The row that best describes a device, or nothing.
///
/// "Best" is the most specific rule that matches, so a table row may be added
/// anywhere without disturbing anything already there.
pub fn lookup(id: &Ident) -> Option<&'static Entry> {
    let mut best: Option<&'static Entry> = None;
    for e in TABLE {
        if e.bus != id.bus || !e.rule.matches(id) {
            continue;
        }
        let better = match best {
            None => true,
            Some(b) => e.rule.specificity() > b.rule.specificity(),
        };
        if better {
            best = Some(e);
        }
    }
    best
}

// ------------------------------------------------------------- the snapshot

/// Where a device physically is, for an operator who has to go and unplug it.
#[derive(Clone, Copy)]
pub enum Where {
    Pci(pci::Device),
    Usb { port: u8, address: u8 },
}

/// One device that is actually present.
pub struct Node {
    pub id: Ident,
    pub at: Where,
    pub entry: Option<&'static Entry>,
}

impl Node {
    /// A name for it, falling back to the PCI class table and then to the
    /// numbers. Something is always printable, because a device nobody can
    /// name is still a fact about the machine.
    pub fn what(&self) -> alloc::string::String {
        match self.entry {
            Some(e) => alloc::string::String::from(e.what),
            None if self.id.bus == Bus::Pci => alloc::string::String::from(pci::class_name(
                self.id.class,
                self.id.subclass,
            )),
            None => alloc::format!("USB class {:02x}.{:02x}", self.id.class, self.id.subclass),
        }
    }

    pub fn location(&self) -> alloc::string::String {
        match self.at {
            Where::Pci(d) => alloc::format!("{:02x}:{:02x}.{}", d.bus, d.dev, d.func),
            Where::Usb { port, address } => alloc::format!("usb{}.{}", port, address),
        }
    }
}

static NODES: Racy<Option<Vec<Node>>> = Racy::new(None);

fn nodes_mut() -> &'static mut Vec<Node> {
    let slot = unsafe { &mut *NODES.get() };
    if slot.is_none() {
        *slot = Some(Vec::new());
    }
    slot.as_mut().unwrap()
}

/// Everything the last sweep found.
pub fn nodes() -> &'static [Node] {
    nodes_mut().as_slice()
}

/// Whether a sweep has happened at all. `nodes()` answering empty is
/// ambiguous otherwise: a machine with no PCI bus and a machine nobody has
/// looked at read identically.
pub fn scanned() -> bool {
    unsafe { (*NODES.get()).is_some() }
}

/// Sweep the PCI bus and record what is there. Answers how many devices.
///
/// One sweep for the whole kernel. Nine drivers each walking 256 buses was
/// 65536 config-space reads apiece, and worse, it meant nothing could ask what
/// the machine had without doing it a tenth time.
pub fn scan_pci(ecam: u64) -> usize {
    let list = nodes_mut();
    list.retain(|n| !matches!(n.at, Where::Pci(_)));
    pci::scan(ecam, 255, |d| {
        let id = Ident::of_pci(&d);
        list.push(Node { id, at: Where::Pci(d), entry: lookup(&id) });
    });
    list.len()
}

/// Record a USB device the enumerator found.
///
/// Pushed rather than swept, because USB enumeration is not free the way a
/// config-space read is: bringing up the controller resets the bus and drops
/// whatever link is on it. So the registry learns about USB from whoever was
/// already enumerating, and holds no opinion about when that should happen.
pub fn note_usb(port: u8, address: u8, id: Ident) {
    let list = nodes_mut();
    list.retain(|n| !matches!(n.at, Where::Usb { port: p, address: a } if p == port && a == address));
    list.push(Node { id, at: Where::Usb { port, address }, entry: lookup(&id) });
}

/// The first PCI device a named driver claims, sweeping first if nobody has.
///
/// This is what a driver's `probe` calls instead of walking the bus itself.
/// The lazy sweep is what makes it safe to call from a probe that runs before
/// the boot sequence gets to the registry, which is the kind of ordering
/// dependency that is very easy to introduce and very annoying to find.
pub fn claimed_by(ecam: u64, driver: &str) -> Option<pci::Device> {
    if !scanned() {
        scan_pci(ecam);
    }
    nodes().iter().find_map(|n| match (n.at, n.entry) {
        (Where::Pci(d), Some(e)) if e.support.driver() == Some(driver) => Some(d),
        _ => None,
    })
}

/// Every present device in a role, most specific match first.
pub fn in_role(role: Role) -> Vec<&'static Node> {
    nodes().iter().filter(|n| n.entry.map(|e| e.role) == Some(role)).collect()
}

/// The drivers that should be tried for a role, in the order the machine
/// suggests rather than an order somebody hardcoded.
///
/// `net::init` used to be a nested match: try e1000, else rtl8168, else USB.
/// That is a preference list written for one laptop, and on a machine with
/// only a Realtek it paid for an Intel sweep to learn nothing. Asking the
/// registry instead means the order is "whatever is plugged in", and a machine
/// with no ethernet at all tries nothing and says so.
pub fn drivers_for(role: Role) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for n in nodes() {
        let Some(e) = n.entry else { continue };
        if e.role != role {
            continue;
        }
        if let Some(name) = e.support.driver() {
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// Every present PCI device in the wireless role, as the seam spells them.
///
/// `crate::radio` cannot name `crate::dev`, so it cannot hold a `pci::Device`
/// and cannot match on `Where`. This is the projection across that boundary,
/// and it lives here rather than in `radio::bus` for the reason every `impl
/// Nic` lives in `net::iface`: the dependency points one way, and the ported
/// tree is the end it points away from.
///
/// USB wireless parts are deliberately absent from this answer rather than
/// silently dropped -- they are not addressed by bus/device/function and a
/// driver for one wants a different seam entirely.
pub fn wireless_pci() -> Vec<crate::radio::bus::Pci> {
    nodes()
        .iter()
        .filter(|n| n.entry.map(|e| e.role) == Some(Role::Wireless))
        .filter_map(|n| match n.at {
            Where::Pci(d) => Some(crate::radio::bus::Pci {
                bus: d.bus,
                dev: d.dev,
                func: d.func,
                vendor: d.vendor,
                device: d.device,
            }),
            Where::Usb { .. } => None,
        })
        .collect()
}

/// Devices that are present, recognised, and unsupported.
///
/// The report a person actually wants when something does not work. It is
/// deliberately a separate function rather than a filter at the call site,
/// because this is the question the registry exists to answer.
pub fn gaps() -> Vec<(&'static Node, &'static str)> {
    nodes()
        .iter()
        .filter_map(|n| match n.entry {
            Some(Entry { support: Support::Known(why), .. }) => Some((n, *why)),
            Some(Entry { support: Support::Partial(_, why), .. }) => Some((n, *why)),
            _ => None,
        })
        .collect()
}

/// Present, driven, driven part of the way, and named with nothing behind it.
///
/// Four numbers rather than one, because "7 devices" says nothing anybody
/// wanted to know -- and four rather than three because folding `Partial` in
/// with the rest would undo the distinction the support levels exist for. The
/// first version of this line reported the framebuffer as having no driver.
pub fn tally() -> (usize, usize, usize, usize) {
    let mut driven = 0;
    let mut partial = 0;
    let mut known = 0;
    for n in nodes() {
        match n.entry.map(|e| &e.support) {
            Some(Support::Driver(_)) => driven += 1,
            Some(Support::Partial(..)) => partial += 1,
            _ => known += 1,
        }
    }
    (driven + partial + known, driven, partial, known)
}

// ---------------------------------------------------------------- reporting

pub fn report() {
    use crate::kprintln;
    if !scanned() {
        kprintln!("  nothing has swept the bus yet");
        return;
    }
    let all = nodes();
    kprintln!("  {} device(s)", all.len());
    for n in all {
        let (tag, driver) = match n.entry {
            Some(e) => (e.support.tag(), e.support.driver().unwrap_or("-")),
            None => ("unknown", "-"),
        };
        kprintln!(
            "  {:>3} {:<9} {:04x}:{:04x} {:<8} {:<10} {}",
            n.id.bus.name(),
            n.location(),
            n.id.vendor,
            n.id.device,
            tag,
            driver,
            n.what()
        );
    }
    let g = gaps();
    if !g.is_empty() {
        kprintln!("  what is missing:");
        for (n, why) in g {
            kprintln!("    {:<28} {}", n.what(), why);
        }
    }
}

// ------------------------------------------------------------------ claims

/// What `diag devices` asks of the table.
///
/// Against synthetic idents rather than the real bus, because a claim about
/// the machine under it would pass here and fail on the next laptop, which is
/// the exact failure this module exists to stop. The rules are arithmetic and
/// the arithmetic is the same everywhere.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();

    let e1000 = Ident { bus: Bus::Pci, vendor: 0x8086, device: 0x100E, class: 0x02, subclass: 0x00, prog_if: 0x00 };
    out.push((
        "an e1000 is matched by id and not by its class",
        matches!(lookup(&e1000).map(|e| &e.rule), Some(Match::Ids(..))),
    ));
    out.push((
        "and the driver named is the one that drives it",
        lookup(&e1000).and_then(|e| e.support.driver()) == Some("e1000"),
    ));

    // The shadowing case, which is the whole reason specificity is ranked.
    let unknown_eth = Ident { bus: Bus::Pci, vendor: 0x1af4, device: 0x1000, class: 0x02, subclass: 0x00, prog_if: 0x00 };
    out.push((
        "an unrecognised ethernet card is still named as ethernet",
        lookup(&unknown_eth).map(|e| e.role) == Some(Role::Ethernet),
    ));
    out.push((
        "and it is not offered a driver that would not drive it",
        lookup(&unknown_eth).and_then(|e| e.support.driver()).is_none(),
    ));

    // Class codes mean different things on the two buses, so a rule from one
    // must never answer for the other.
    let usb_kbd = Ident::of_usb(0x046d, 0xc31c, 0x03, 0x01, 0x01);
    out.push((
        "a USB keyboard is not mistaken for a PCI display controller",
        lookup(&usb_kbd).map(|e| e.role) == Some(Role::Input),
    ));
    let pci_vga = Ident { bus: Bus::Pci, vendor: 0x1234, device: 0x1111, class: 0x03, subclass: 0x00, prog_if: 0x00 };
    out.push((
        "and a display controller is not mistaken for a keyboard",
        lookup(&pci_vga).map(|e| e.role) == Some(Role::Display),
    ));

    // The dongle on this machine wears somebody else's vendor id, which is the
    // case that made an id list necessary in the first place.
    let tplink = Ident::of_usb(0x2357, 0x010C, 0xFF, 0xFF, 0xFF);
    out.push((
        "the TP-Link badge on a Realtek chip still finds the Realtek driver",
        lookup(&tplink).and_then(|e| e.support.driver()) == Some("rtl8188eu"),
    ));
    out.push((
        "and it is reported as partial, because it cannot carry a frame",
        matches!(lookup(&tplink).map(|e| &e.support), Some(Support::Partial(..))),
    ));

    // NVMe is matched by programming interface, so an SSD nobody has heard of
    // still works. This is the row that means most machines have storage.
    let ssd = Ident { bus: Bus::Pci, vendor: 0x1e0f, device: 0x0001, class: 0x01, subclass: 0x08, prog_if: 0x02 };
    out.push((
        "an unheard-of NVMe SSD is driven, because the interface is what matches",
        lookup(&ssd).and_then(|e| e.support.driver()) == Some("nvme"),
    ));
    let ahci = Ident { bus: Bus::Pci, vendor: 0x8086, device: 0x282a, class: 0x01, subclass: 0x06, prog_if: 0x01 };
    out.push((
        "and a SATA controller is named as the gap it is, rather than left unknown",
        matches!(lookup(&ahci).map(|e| &e.support), Some(Support::Known(_)))
            && lookup(&ahci).map(|e| e.role) == Some(Role::Storage),
    ));

    // A CNVi wireless part must not fall through to the generic wireless row,
    // because the generic row's reason is wrong for it.
    let cnvi = Ident { bus: Bus::Pci, vendor: 0x8086, device: 0x51f0, class: 0x02, subclass: 0x80, prog_if: 0x00 };
    out.push((
        "the wireless in this laptop is named exactly, not filed under 'wireless'",
        lookup(&cnvi).map(|e| e.what) == Some("Intel Wi-Fi 6/6E, CNVi in the chipset"),
    ));

    out.push((
        "a device matching nothing is matched by nothing, rather than by the last row",
        lookup(&Ident {
            bus: Bus::Pci,
            vendor: 0xDEAD,
            device: 0xBEEF,
            class: 0x13,
            subclass: 0x37,
            prog_if: 0x00,
        })
        .is_none(),
    ));

    // Every row has to be reachable, or it is documentation pretending to be
    // code. A row whose rule cannot win against the rest of the table would
    // never fire, and nothing else would ever say so.
    out.push((
        "every row in the table can be reached by some device",
        TABLE.iter().all(|e| {
            let id = match e.rule {
                Match::Ids(v, list) => Ident {
                    bus: e.bus,
                    vendor: v,
                    device: list[0],
                    class: 0xFF,
                    subclass: 0xFF,
                    prog_if: 0xFF,
                },
                Match::Class(c, s) => Ident {
                    bus: e.bus,
                    vendor: 0,
                    device: 0,
                    class: c,
                    subclass: s,
                    prog_if: 0xFF,
                },
                Match::Interface(c, s, p) => Ident {
                    bus: e.bus,
                    vendor: 0,
                    device: 0,
                    class: c,
                    subclass: s,
                    prog_if: p,
                },
            };
            lookup(&id).map(|f| core::ptr::eq(f, e)).unwrap_or(false)
        }),
    ));

    out.push((
        "no two rows claim the same device with the same specificity",
        TABLE.iter().enumerate().all(|(i, a)| {
            TABLE.iter().skip(i + 1).all(|b| {
                a.bus != b.bus
                    || a.rule.specificity() != b.rule.specificity()
                    || !overlap(&a.rule, &b.rule)
            })
        }),
    ));

    out
}

/// Whether two rules of equal specificity could both match one device.
fn overlap(a: &Match, b: &Match) -> bool {
    match (a, b) {
        (Match::Ids(v1, l1), Match::Ids(v2, l2)) => {
            v1 == v2 && l1.iter().any(|d| l2.contains(d))
        }
        (Match::Class(c1, s1), Match::Class(c2, s2)) => c1 == c2 && s1 == s2,
        (Match::Interface(c1, s1, p1), Match::Interface(c2, s2, p2)) => {
            c1 == c2 && s1 == s2 && p1 == p2
        }
        _ => false,
    }
}
