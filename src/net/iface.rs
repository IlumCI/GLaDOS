//! Named interfaces, and the routing decision between them.
//!
//! Until now there was one NIC, one address, and one implied route: everything
//! either matched the subnet or went to the gateway. That is fine with a single
//! card and wrong the moment there are two, which is what wireless means.
//!
//! So: an interface is a name, a driver, an address, and a link state.
//! `eth0` is the wired card, `wlan0` is the wireless one when a driver exists
//! for it, and `lo` is the loopback that has no driver at all. Sending picks an
//! interface by looking at the destination, and every layer above asks for a
//! source address rather than assuming there is only one.
//!
//! ### Why the driver is a trait object
//!
//! `Box<dyn Nic>` costs an indirect call per frame, which is nothing next to
//! the DMA it wraps, and it means the e1000 and a future wireless driver are
//! interchangeable without the layers above knowing which is which. The
//! alternative -- an enum of every supported card -- puts every driver in one
//! file's match arms and makes adding one a change to shared code.

use super::{Ipv4, Mac};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// What a driver has to provide to be an interface.
///
/// Deliberately frame-level: the interface layer knows about Ethernet frames
/// and nothing about descriptor rings, and a wireless driver is expected to
/// present 802.11 as Ethernet the way every other OS does, because otherwise
/// every layer above would need a second version.
pub trait Nic {
    fn mac(&self) -> Mac;
    fn link_up(&mut self) -> bool;
    fn transmit(&mut self, frame: &[u8]) -> bool;
    fn receive(&mut self) -> Option<Vec<u8>>;
    /// For display, and for deciding whether wireless commands apply.
    fn kind(&self) -> Kind;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Ethernet,
    Wireless,
    Loopback,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Ethernet => "ethernet",
            Kind::Wireless => "wireless",
            Kind::Loopback => "loopback",
        }
    }
}

/// The loopback driver. Anything transmitted is immediately receivable.
///
/// Worth having for more than tidiness: it is the only way to exercise the IP
/// and ICMP paths with no hardware and no peer involved, so a broken checksum
/// shows up as `ping 127.0.0.1` failing rather than as a silent network.
pub struct Loopback {
    queue: Vec<Vec<u8>>,
}

impl Loopback {
    pub const fn new() -> Self {
        Loopback { queue: Vec::new() }
    }
}

impl Nic for Loopback {
    fn mac(&self) -> Mac {
        [0; 6]
    }
    fn link_up(&mut self) -> bool {
        true
    }
    fn transmit(&mut self, frame: &[u8]) -> bool {
        // Bounded: a loop that generates a reply to its own reply would
        // otherwise grow without limit.
        if self.queue.len() < 16 {
            self.queue.push(frame.to_vec());
        }
        true
    }
    fn receive(&mut self) -> Option<Vec<u8>> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        }
    }
    fn kind(&self) -> Kind {
        Kind::Loopback
    }
}

/// The wired card. Implemented here rather than in `dev::e1000` so that a
/// driver stays unaware of the network layer above it -- the dependency points
/// one way, and a driver can be tested without one.
impl Nic for crate::dev::e1000::E1000 {
    fn mac(&self) -> Mac {
        crate::dev::e1000::E1000::mac(self)
    }
    fn link_up(&mut self) -> bool {
        crate::dev::e1000::E1000::link_up(self)
    }
    fn transmit(&mut self, frame: &[u8]) -> bool {
        crate::dev::e1000::E1000::transmit(self, frame)
    }
    fn receive(&mut self) -> Option<Vec<u8>> {
        crate::dev::e1000::E1000::receive(self)
    }
    fn kind(&self) -> Kind {
        Kind::Ethernet
    }
}

#[derive(Clone, Copy, Default)]
pub struct Stats {
    pub rx_packets: u64,
    pub rx_bytes: u64,
    pub tx_packets: u64,
    pub tx_bytes: u64,
    pub tx_dropped: u64,
}

/// More than one entry, unlike the cache this replaces. A host that talks to a
/// gateway and a peer on the same subnet needs two, and the old single slot
/// meant the two evicted each other on every packet.
pub const ARP_ENTRIES: usize = 8;

pub struct Interface {
    pub name: &'static str,
    pub nic: Option<Box<dyn Nic>>,
    pub ip: Ipv4,
    pub netmask: Ipv4,
    pub gateway: Ipv4,
    pub dns: Ipv4,
    /// Administratively up. A link that is down in hardware is reported
    /// separately, because "unplugged" and "switched off" are different
    /// problems and conflating them hides the first.
    pub up: bool,
    pub arp: [Option<(Ipv4, Mac)>; ARP_ENTRIES],
    pub arp_next: usize,
    pub stats: Stats,
}

impl Interface {
    pub const fn empty(name: &'static str) -> Self {
        Interface {
            name,
            nic: None,
            ip: [0, 0, 0, 0],
            netmask: [0, 0, 0, 0],
            gateway: [0, 0, 0, 0],
            dns: [0, 0, 0, 0],
            up: false,
            arp: [None; ARP_ENTRIES],
            arp_next: 0,
            stats: Stats {
                rx_packets: 0,
                rx_bytes: 0,
                tx_packets: 0,
                tx_bytes: 0,
                tx_dropped: 0,
            },
        }
    }

    pub fn present(&self) -> bool {
        self.nic.is_some()
    }

    pub fn usable(&mut self) -> bool {
        self.up && self.nic.as_mut().map(|n| n.link_up()).unwrap_or(false)
    }

    pub fn on_subnet(&self, dst: Ipv4) -> bool {
        self.ip != [0, 0, 0, 0]
            && (0..4).all(|i| dst[i] & self.netmask[i] == self.ip[i] & self.netmask[i])
    }

    pub fn arp_lookup(&self, ip: Ipv4) -> Option<Mac> {
        self.arp
            .iter()
            .flatten()
            .find(|(a, _)| *a == ip)
            .map(|(_, m)| *m)
    }

    /// Insert or refresh. Replacement is round-robin rather than
    /// least-recently-used: with eight slots the difference is not measurable,
    /// and a counter per entry is state that can go stale on its own.
    pub fn arp_insert(&mut self, ip: Ipv4, mac: Mac) {
        for slot in self.arp.iter_mut() {
            if let Some((a, m)) = slot {
                if *a == ip {
                    *m = mac;
                    return;
                }
            }
        }
        for slot in self.arp.iter_mut() {
            if slot.is_none() {
                *slot = Some((ip, mac));
                return;
            }
        }
        self.arp[self.arp_next] = Some((ip, mac));
        self.arp_next = (self.arp_next + 1) % ARP_ENTRIES;
    }
}
