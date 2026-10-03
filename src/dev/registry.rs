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

/// Intel wireless that `dev::iwx` is written for, split by the family the
/// *controller* belongs to.
///
/// **These lists came from `pcidevs` and `iwx_attach` in OpenBSD, not from
/// memory, and the memory was wrong in three places.** The previous two lists
/// sorted by "CNVi or discrete", which is a fact about packaging and decides
/// nothing about the driver; what decides the bring-up is the generation, and
/// the old discrete list held two AC 9560 parts (`9df0`, `31dc`) and the 8265
/// (`24fd`), which are the generation *before* this one and are `iwm(4)`'s, plus
/// `272b`, which is a Bz part `iwx` refuses. A row claiming `iwx` for any of
/// those would have sent the driver at silicon it has no path for.
///
/// The family is still read off `CSR_HW_REV` at probe time rather than trusted
/// from this table -- `51f0` is an AX211 in `pcidevs` and an AX201 on the GF63,
/// because the name follows the radio module and the family follows the
/// controller -- so these lists say which *driver* to try, and the register says
/// which path inside it.
///
/// Family 22000: the AX200 and the AX201 as fitted to Ice, Comet, Tiger and
/// Jasper Lake. `02f0`, `06f0`, `34f0`, `3df0`, `43f0`, `4df0` and `a0f0` are
/// CNVi; `2723` is the discrete AX200.
const IWX_22000_IDS: &[u16] = &[0x2723, 0x02f0, 0x06f0, 0x34f0, 0x3df0, 0x43f0, 0x4df0, 0xa0f0];

/// The AX210 family: Typhoon Peak and Snow Owl. `2725` and `2726` are discrete;
/// `51f0` (the GF63's), `51f1`, `54f0`, `7a70`, `7af0`, `7e40` and `7f70` are
/// CNVi on Alder, Raptor, Meteor and Arrow Lake.
const IWX_AX210_IDS: &[u16] = &[0x2725, 0x2726, 0x51f0, 0x51f1, 0x54f0, 0x7a70, 0x7af0, 0x7e40, 0x7f70];

/// Bz and later. `iwx` names them only to refuse: the reset and the clock
/// handshake differ again, and nothing here has that path.
const INTEL_BZ_IDS: &[u16] = &[0x7740, 0x272b];

/// The generation before: 8265, 9260 and the AC 9560 in both its CNVi spins.
/// A different firmware API and `iwm(4)`'s parts, not `iwx(4)`'s.
const INTEL_MVM_IDS: &[u16] = &[0x24fd, 0x2526, 0x9df0, 0x31dc, 0xa370];

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
        rule: Match::Ids(0x8086, IWX_AX210_IDS),
        role: Role::Wireless,
        what: "Intel Wi-Fi 6/6E, AX210 family",
        // **"The radio is in the PCH over an undocumented interface" named the
        // wrong obstacle.** CNVi does put the MAC in the chipset and the radio
        // on the M.2 module, joined by CNVio, and CNVio is undocumented -- but
        // *the host never speaks it*. From here the part is an ordinary PCIe
        // function (00:14.3 on this laptop) with BARs and MSI-X, driven the way
        // a discrete card is. The cost is a firmware image in Intel's TLV
        // container and a host command protocol, which is big and not
        // impossible.
        //
        // Partial and not Known, because `dev::iwx` boots this family's
        // firmware to ALIVE and reads the NVM. The gap is everything after.
        support: Support::Partial("iwx", "boots firmware, reads the NVM and scans; joining and data are not written"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x8086, IWX_22000_IDS),
        role: Role::Wireless,
        what: "Intel Wi-Fi 6, 22000 family",
        // The descriptor this family boots out of is written and asserted
        // (`iwx::ctxt`); the boot path does not take it yet, and nobody here has
        // the part to try it on.
        support: Support::Partial("iwx", "the boot descriptor is written and never driven; no part here to test it"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x8086, INTEL_BZ_IDS),
        role: Role::Wireless,
        what: "Intel Wi-Fi 7, Bz family",
        support: Support::Known("iwlwifi-class, but the reset and clock handshake differ from the AX210's and are not written"),
    },
    Entry {
        bus: Bus::Pci,
        rule: Match::Ids(0x8086, INTEL_MVM_IDS),
        role: Role::Wireless,
        what: "Intel Wireless-AC 8265/9260/9560",
        // "not redistributable" was simply wrong. `LICENCE.iwlwifi_firmware`
        // permits redistribution and use in binary form without modification,
        // which is why Debian ships it at all -- in `non-free-firmware`, the
        // label for redistributable-and-not-free.
        support: Support::Known("the generation before iwx: an older firmware API and no driver for it here"),
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
        // Understated, and the understatement was doing harm: it read as a
        // driver nobody had started. `xhci` identifies the part, reads its
        // chip id and runs `bring_up`, which applies all four initialisation
        // tables including the radio over the register interface; the efuse
        // decoder, the LLT chain, the firmware container parser, the channel
        // plan and both descriptor formats are written and asserted at boot.
        // Bulk endpoints are not missing either -- `xhci` configures and
        // drives them for CDC and RNDIS already.
        //
        // What is left is the sequence that uses those pieces on the wire,
        // and then `impl Radio`, above which everything is done.
        support: Support::Known("the driver was removed when its test hardware died; it never carried a frame, and it was the tree's only GPL-2.0 code"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Ids(0x2357, RTL8188EU_IDS_TPLINK),
        role: Role::Wireless,
        what: "TP-Link RTL8188EU wireless dongle",
        support: Support::Known("the driver was removed when its test hardware died; it never carried a frame, and it was the tree's only GPL-2.0 code"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0x02, 0x06, 0x00),
        role: Role::Ethernet,
        what: "CDC ethernet adapter",
        support: Support::Driver("usb-ecm"),
    },
    // RNDIS, in both the encodings it is found in. Two rows rather than one
    // because they are two different interface triples and this table matches
    // triples; the driver behind them is the same.
    //
    // **This is the closest thing to a universal wireless driver there is.**
    // 802.11 hardware has no common register interface, so a radio needs a
    // driver per chip; a phone in USB tethering mode presents one of these and
    // shares the radio it already has.
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0xE0, 0x01, 0x03),
        role: Role::Ethernet,
        what: "RNDIS network device, as Android tethering presents one",
        support: Support::Driver("usb-rndis"),
    },
    Entry {
        bus: Bus::Usb,
        rule: Match::Interface(0x02, 0x02, 0xFF),
        role: Role::Ethernet,
        // Named for what it is rather than what it claims: the triple is
        // CDC/ACM/vendor, which a vendor-specific modem also uses. Nothing in
        // a descriptor separates them, and the driver finds out by trying --
        // a modem does not complete an RNDIS INITIALIZE.
        what: "RNDIS network device, or a vendor-specific modem",
        support: Support::Driver("usb-rndis"),
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

/// Devices that are present, recognised, and unsupported.
///
/// The report a person actually wants when something does not work. It is
/// deliberately a separate function rather than a filter at the call site,
/// because this is the question the registry exists to answer.
/// Which machine this is, as the *devices* say rather than as CPUID says.
///
/// **CPUID cannot answer this and the measurement is why.** Leaf `0x40000000`
/// carries a twelve-byte vendor string, and under QEMU accelerated by WHPX this
/// kernel read `56 4d 77 61 72 65 56 4d 77 61 72 65` -- `VMwareVMware`,
/// exactly. That is QEMU's `vmware-cpuid-freq`, on by default, which borrows
/// VMware's own CPUID convention for the leaf that reports TSC and APIC
/// frequency and takes the signature with it. So a kernel that named its
/// hypervisor from CPUID would tell every QEMU user they were running on
/// VMware, confidently and in writing.
///
/// Emulated hardware does not have that problem. A hypervisor has to put PCI
/// devices in front of a guest and those carry the vendor's own id, which is
/// not a courtesy string it chose to emit -- it is what the driver has to match
/// to work at all. `80EE` is VirtualBox's, `15AD` is VMware's, and QEMU's
/// devices come from Red Hat's `1B36` and `1AF4` plus the Bochs display at
/// `1234:1111`.
///
/// Still evidence rather than proof: a hypervisor can be configured to present
/// somebody else's devices, and one that presents none of these answers
/// `Unrecognised` rather than a guess.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Platform {
    /// No hypervisor bit. Real hardware, or one hiding well enough that
    /// nothing here can tell and nothing here should pretend to.
    Bare,
    Qemu,
    VMware,
    VirtualBox,
    HyperV,
    /// A hypervisor is present and none of its devices are ones we know.
    Unrecognised,
}

impl Platform {
    pub fn name(self) -> &'static str {
        match self {
            Platform::Bare => "bare metal",
            Platform::Qemu => "QEMU",
            Platform::VMware => "VMware",
            Platform::VirtualBox => "VirtualBox",
            Platform::HyperV => "Hyper-V",
            Platform::Unrecognised => "a hypervisor this kernel does not recognise",
        }
    }
}

pub fn platform() -> Platform {
    if crate::cpu::hypervisor().is_none() {
        return Platform::Bare;
    }
    platform_of(nodes())
}

/// The same answer over a list somebody hands in.
///
/// Split out so VirtualBox and VMware are testable on a machine that is
/// neither, which is the whole difficulty with this feature: the two guests
/// most people will use are the two nobody here can boot. `mem::fixed` asserts
/// its arithmetic against a synthetic memory map for the same reason, and
/// `diag devices` already checks every row against synthetic idents rather than
/// against the bus underneath.
pub fn platform_of(list: &[Node]) -> Platform {
    // Order is by how unambiguous the id is. VirtualBox and VMware own their
    // vendor numbers outright; QEMU's are Red Hat's and a Bochs display that
    // several emulators have borrowed, so they are checked last.
    let mut qemu = false;
    for n in list {
        match (n.id.vendor, n.id.device) {
            (0x80EE, _) => return Platform::VirtualBox,
            (0x15AD, _) => return Platform::VMware,
            (0x1414, _) => return Platform::HyperV,
            (0x1B36, _) | (0x1AF4, _) | (0x1234, 0x1111) => qemu = true,
            _ => {}
        }
    }
    if qemu {
        return Platform::Qemu;
    }
    Platform::Unrecognised
}

/// What this guest needs changed before GLaDOS is much use on it.
///
/// Derived from what was enumerated rather than from a table per hypervisor,
/// because the question is never "which VM is this" but "is the thing this
/// kernel requires actually present". A VirtualBox configured with an NVMe
/// controller needs nothing said; a QEMU without one needs the same sentence
/// VMware does.
pub fn vm_advice() -> Vec<&'static str> {
    if platform() == Platform::Bare {
        return Vec::new();
    }
    advice_for(nodes())
}

/// The same, over a list somebody hands in. See `platform_of`.
pub fn advice_for(list: &[Node]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    let has_nvme = list.iter().any(|n| n.id.class == 0x01 && n.id.subclass == 0x08);
    let has_ahci = list.iter().any(|n| n.id.class == 0x01 && n.id.subclass == 0x06);
    if !has_nvme {
        // The single most likely reason a guest boots to a shell with nothing
        // behind it. Both VirtualBox and VMware default to SATA.
        out.push(
            "no NVMe controller -- set the guest's disk controller to NVMe;              there is no AHCI driver here, so a SATA disk is a disk this kernel cannot see",
        );
        if has_ahci {
            out.push("(a SATA controller in AHCI mode is present and is exactly what cannot be driven)");
        }
    }
    // Asked of the same table a driver is chosen from, so a row added for a
    // new NIC silences this line without anybody remembering to.
    let has_eth = list
        .iter()
        .any(|n| n.entry.map(|e| e.support.driver().is_some() && e.role == Role::Ethernet).unwrap_or(false));
    if !has_eth {
        out.push(
            "no ethernet this kernel drives -- choose an Intel PRO/1000 (e1000) or e1000e adapter",
        );
    }
    out
}

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

    // Which machine this is, and what it needs changed. First, because a guest
    // whose disk controller is wrong has nothing else worth reading here -- and
    // because the install story for this kernel is about to be "boot it in a
    // VM", which makes this the line most readers are looking for.
    let p = platform();
    if p != Platform::Bare {
        crate::gfx::console::set_color(crate::gfx::console::YELLOW);
        // Both are printed when they disagree, and they do: QEMU reports
        // `VMwareVMware` at CPUID leaf 0x40000000. Naming only one of them
        // would be picking which of two true readings to hide.
        match crate::cpu::hypervisor_name() {
            Some(cpuid) if cpuid != p.name() => {
                kprintln!("  running on {} (its devices say so; CPUID says '{}')", p.name(), cpuid)
            }
            _ => kprintln!("  running on {}", p.name()),
        }
        for line in vm_advice() {
            crate::gfx::console::set_color(crate::gfx::console::LTRED);
            kprintln!("  {}", line);
        }
        crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
    }
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
/// A synthetic node for the VM claims, carrying only what they read.
///
/// Built here rather than taken off the bus, for the reason every other claim
/// in this file is: a check written against the machine underneath passes on
/// this one and says nothing about the next.
fn fake(vendor: u16, device: u16, class: u8, subclass: u8) -> Node {
    Node {
        id: Ident { bus: Bus::Pci, vendor, device, class, subclass, prog_if: 0 },
        at: Where::Pci(crate::dev::pci::Device {
            bus: 0,
            dev: 0,
            func: 0,
            vendor,
            device,
            class,
            subclass,
            prog_if: 0,
            header_type: 0,
        }),
        entry: None,
    }
}

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
        "the TP-Link badge on a Realtek chip still finds the Realtek row",
        lookup(&tplink).is_some(),
    ));
    // **It reported `Partial` and now reports `Known`**, which is the honest
    // change: `Partial` means identified with something behind it, and there is
    // nothing behind it any more. The row stays because the registry's job is to
    // name what is missing -- deleting it would make an unsupported dongle look
    // like an unknown one, and `what is missing` is the most useful half of
    // `devices`.
    out.push((
        "and it is reported as known rather than driven, because there is no driver",
        matches!(lookup(&tplink).map(|e| &e.support), Some(Support::Known(..))),
    ));
    out.push((
        "and it names no driver, so nothing can claim to drive it",
        lookup(&tplink).and_then(|e| e.support.driver()).is_none(),
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
    // because the generic row's reason is wrong for it. `0x51f0` is the part
    // actually fitted to the GF63 this is developed on, read off the machine
    // rather than remembered: Intel Wi-Fi 6 AX201 at 00:14.3.
    //
    // **Asserted as a distinction and not as a spelling.** This compared
    // `e.what` against a transcribed string, so renaming the row broke a claim
    // about something that had not changed -- the mistake this file makes a
    // point of avoiding two claims further down, arriving from the other
    // direction. What has to hold is that the specific row wins over the
    // generic one, and that survives either being reworded.
    let cnvi = Ident { bus: Bus::Pci, vendor: 0x8086, device: 0x51f0, class: 0x02, subclass: 0x80, prog_if: 0x00 };
    let nameless = Ident { bus: Bus::Pci, vendor: 0x1234, device: 0x5678, class: 0x02, subclass: 0x80, prog_if: 0x00 };
    out.push((
        "the wireless in this laptop is named exactly, not filed under the generic row",
        lookup(&cnvi).map(|e| e.role) == Some(Role::Wireless)
            && lookup(&nameless).map(|e| e.role) == Some(Role::Wireless)
            && lookup(&cnvi).map(|e| e.what) != lookup(&nameless).map(|e| e.what),
    ));
    // And it names the driver that will try it, which falling through to the
    // generic row ("recognised as wireless, but not this model") would not.
    out.push((
        "and it names iwx as the driver, where the generic row names none",
        lookup(&cnvi).and_then(|e| e.support.driver()) == Some("iwx")
            && lookup(&nameless).and_then(|e| e.support.driver()).is_none(),
    ));
    // The split the lists were rebuilt for: an AC 9560 is Intel wireless and is
    // *not* iwx's, and claiming it would send the driver at a generation it has
    // no path for.
    let ac9560 = Ident { bus: Bus::Pci, vendor: 0x8086, device: 0x9df0, class: 0x02, subclass: 0x80, prog_if: 0x00 };
    out.push((
        "an AC 9560 is recognised as Intel wireless and not claimed by iwx",
        lookup(&ac9560).map(|e| e.role) == Some(Role::Wireless)
            && lookup(&ac9560).and_then(|e| e.support.driver()).is_none(),
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

    // ---- which guest this is, and what it needs changed --------------------
    //
    // **VirtualBox and VMware are the two guests most people will use and the
    // two nobody here can boot**, so every one of these runs against a
    // synthetic device list. That is the same bargain `mem::fixed` makes with
    // its memory map: a claim about the machine underneath would pass here and
    // tell you nothing about the one you are trying to support.
    out.push((
        "VirtualBox is named by its own vendor id",
        platform_of(&[fake(0x80EE, 0xBEEF, 0x03, 0x00)]) == Platform::VirtualBox,
    ));
    out.push((
        "VMware is named by its own vendor id",
        platform_of(&[fake(0x15AD, 0x0405, 0x03, 0x00)]) == Platform::VMware,
    ));
    out.push((
        "and QEMU by the devices Red Hat ships it with",
        platform_of(&[fake(0x1B36, 0x0010, 0x01, 0x08)]) == Platform::Qemu
            && platform_of(&[fake(0x1234, 0x1111, 0x03, 0x00)]) == Platform::Qemu,
    ));
    // The one that matters, because CPUID says `VMwareVMware` under QEMU: a
    // guest carrying both signatures is named by the unambiguous one.
    out.push((
        "a guest showing VMware's id beside QEMU's is VMware, not QEMU",
        platform_of(&[fake(0x1234, 0x1111, 0x03, 0x00), fake(0x15AD, 0x0405, 0x03, 0x00)])
            == Platform::VMware,
    ));
    out.push((
        "a hypervisor whose devices are all strangers is not guessed at",
        platform_of(&[fake(0xDEAD, 0x0001, 0x03, 0x00)]) == Platform::Unrecognised,
    ));

    // The advice, which is the half a person actually acts on. Both of the
    // default configurations are wrong in the same way and it is worth saying
    // so in those words: VirtualBox and VMware both default to SATA.
    let sata_only = [fake(0x15AD, 0x07E0, 0x01, 0x06)];
    out.push((
        "a guest with only a SATA controller is told to switch it to NVMe",
        advice_for(&sata_only).iter().any(|l| l.contains("NVMe")),
    ));
    out.push((
        "and told that the SATA controller it has is the one that cannot be driven",
        advice_for(&sata_only).iter().any(|l| l.contains("AHCI")),
    ));
    out.push((
        "a guest that already has NVMe is told nothing about disks",
        !advice_for(&[fake(0x1B36, 0x0010, 0x01, 0x08)])
            .iter()
            .any(|l| l.contains("NVMe")),
    ));
    // Derived from the driver table rather than from a list of vendor ids, so
    // adding a NIC row silences this without anybody remembering to.
    out.push((
        "a guest with no drivable ethernet is told which adapter to pick",
        advice_for(&sata_only).iter().any(|l| l.contains("e1000")),
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
