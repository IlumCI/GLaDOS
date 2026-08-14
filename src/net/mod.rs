//! Ethernet, ARP, IPv4 and ICMP, over named interfaces.
//!
//! `ping` was the first milestone because it exercises the whole path in one
//! visible result: PCI discovery, an MMIO mapping, DMA rings, frame parsing,
//! ARP resolution, an IPv4 header with a correct checksum, and a reply that
//! has to come back. Anything broken anywhere in that stack shows up as
//! silence, and silence is easy to bisect when the alternative is a TCP state
//! machine failing intermittently. That ordering paid for itself.
//!
//! Interfaces live in `iface`: `lo`, `eth0`, and `wlan0` when something can
//! drive it. Sending routes by destination rather than assuming one card, and
//! every address is a property of an interface rather than of the machine.
//!
//! ### Why TCP and UDP segments are queued rather than handled inline
//!
//! `poll` does not hand a segment straight to the transport. It pushes it onto
//! an inbox that the transport drains later. The reason is re-entrancy:
//! sending anything calls `send_ipv4` -> `resolve`, and `resolve` calls `poll`
//! while it waits for an ARP reply. If `poll` ran a state machine directly, a
//! connection could re-enter its own control block through that path while an
//! earlier borrow was still live. Queueing breaks the cycle at the one place
//! it can form.

use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use crate::sync::Racy;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

pub type Mac = [u8; 6];
pub type Ipv4 = [u8; 4];

const BROADCAST: Mac = [0xFF; 6];

const ETHERTYPE_IPV4: u16 = 0x0800;
const ETHERTYPE_ARP: u16 = 0x0806;

const ARP_REQUEST: u16 = 1;
const ARP_REPLY: u16 = 2;

const PROTO_ICMP: u8 = 1;
pub(crate) const PROTO_TCP: u8 = 6;
pub(crate) const PROTO_UDP: u8 = 17;
const ICMP_ECHO_REQUEST: u8 = 8;
const ICMP_ECHO_REPLY: u8 = 0;

pub mod css;
pub mod dhcp;
pub mod dns;
pub mod html;
pub mod iface;
pub mod tcp;
pub mod tls;
pub mod ws;
pub mod trust;
pub mod udp;
pub mod wifi;
pub mod wpa2;
pub mod x509;

use iface::{Interface, Kind, Loopback, Nic};

pub const UNSPECIFIED: Ipv4 = [0, 0, 0, 0];
pub const BROADCAST_IP: Ipv4 = [255, 255, 255, 255];

pub const LO: usize = 0;
pub const ETH0: usize = 1;
pub const WLAN0: usize = 2;

static IFACES: Racy<[Interface; 3]> = Racy::new([
    Interface::empty("lo"),
    Interface::empty("eth0"),
    Interface::empty("wlan0"),
]);

/// A snapshot of one interface's addressing, kept as a struct because DHCP
/// hands back all four at once and applying them one at a time would leave the
/// machine briefly half-configured.
#[derive(Clone, Copy)]
pub struct Config {
    pub ip: Ipv4,
    pub gateway: Ipv4,
    pub netmask: Ipv4,
    pub dns: Ipv4,
}

pub fn ifaces() -> &'static mut [Interface; 3] {
    unsafe { &mut *IFACES.get() }
}

/// Kept from boot so the bus can be re-enumerated later -- `wlan0` reports
/// what it finds on demand, and re-walking PCI is cheaper than caching a
/// device list that nothing else needs.
static ECAM: Racy<Option<u64>> = Racy::new(None);

pub fn ecam() -> Option<u64> {
    unsafe { *ECAM.get() }
}

pub fn index_of(name: &str) -> Option<usize> {
    ifaces().iter().position(|i| i.name == name)
}

// --- routing -------------------------------------------------------------

fn is_loopback(ip: Ipv4) -> bool {
    ip[0] == 127
}

/// Is this address one of ours, on any interface?
fn is_local(ip: Ipv4) -> bool {
    ifaces().iter().any(|i| i.present() && i.ip == ip && ip != UNSPECIFIED)
}

/// Choose the interface a packet to `dst` should leave by.
///
/// Three rules, in order: our own addresses and 127/8 go to loopback, a
/// destination on an interface's subnet goes out that interface, and anything
/// else goes to the first usable interface that has a gateway. That last one
/// is the default route, and having exactly one is why there is no routing
/// table -- with two uplinks it would need to become one.
pub fn route(dst: Ipv4) -> Option<usize> {
    if is_loopback(dst) || is_local(dst) {
        return ifaces()[LO].present().then_some(LO);
    }
    let ifs = ifaces();
    for (n, i) in ifs.iter_mut().enumerate() {
        if n != LO && i.usable() && i.on_subnet(dst) {
            return Some(n);
        }
    }
    for (n, i) in ifs.iter_mut().enumerate() {
        if n != LO && i.usable() && i.gateway != UNSPECIFIED {
            return Some(n);
        }
    }
    None
}

/// The interface commands act on when none is named.
pub fn primary() -> usize {
    let ifs = ifaces();
    for n in [ETH0, WLAN0] {
        if ifs[n].usable() {
            return n;
        }
    }
    for n in [ETH0, WLAN0] {
        if ifs[n].present() {
            return n;
        }
    }
    LO
}

/// The source address to put on a packet headed for `dst`.
pub(crate) fn local_addr_for(dst: Ipv4) -> Ipv4 {
    route(dst).map(|n| ifaces()[n].ip).unwrap_or(UNSPECIFIED)
}

pub fn ready() -> bool {
    let n = primary();
    n != LO && ifaces()[n].present()
}

pub fn config() -> Config {
    config_of(primary())
}

pub fn config_of(n: usize) -> Config {
    let i = &ifaces()[n];
    Config {
        ip: i.ip,
        gateway: i.gateway,
        netmask: i.netmask,
        dns: i.dns,
    }
}

pub fn set_config_of(n: usize, c: Config) {
    let i = &mut ifaces()[n];
    i.ip = c.ip;
    i.gateway = c.gateway;
    i.netmask = c.netmask;
    i.dns = c.dns;
}

// --- bring-up ------------------------------------------------------------

pub fn init(ecam: u64, roots: Option<&[u8]>) {
    console::set_color(YELLOW);
    kprintln!("\n[net]");
    console::set_color(LTGRAY);
    unsafe { *ECAM.get() = Some(ecam) };

    match roots {
        Some(data) => {
            let n = trust::load(data);
            kprintln!("  trust  {} root certificate(s)", n);
        }
        None => {
            // Said at boot rather than at the first failed connection, because
            // "https does not work" is a much worse symptom to debug than
            // "there are no roots".
            kprintln!("  trust  no roots -- https will encrypt but not authenticate");
        }
    }

    {
        let lo = &mut ifaces()[LO];
        lo.nic = Some(Box::new(Loopback::new()));
        lo.ip = [127, 0, 0, 1];
        lo.netmask = [255, 0, 0, 0];
        lo.up = true;
    }

    // Try each driver in turn and take the first that answers. The e1000 is
    // first only because it is what QEMU emulates, so the common development
    // case costs one probe; on the GF63 it misses and the Realtek answers.
    let driver: Option<(Box<dyn Nic>, &str)> = match crate::dev::e1000::probe(ecam) {
        Ok(n) => Some((Box::new(n), "e1000")),
        Err(e1000_err) => match crate::dev::rtl8168::probe(ecam) {
            Ok(n) => Some((Box::new(n), "rtl8168")),
            // USB last, and only when no PCI card answered. Bringing up the
            // xHCI controller resets it, so a machine with a working wired card
            // should not have its USB bus reset during boot for nothing -- and
            // `usb` on the shell resets it again, which would take the
            // interface out from under this driver.
            Err(rtl_err) => match crate::dev::xhci::probe_net(ecam) {
                Ok(n) => Some((Box::new(n), "usb-ecm")),
                Err(usb_err) => {
                    kprintln!("  eth0   no supported NIC");
                    kprintln!(
                        "         e1000 {:?}, rtl8168 {:?}, usb {}",
                        e1000_err, rtl_err, usb_err
                    );
                    None
                }
            },
        },
    };

    match driver {
        None => {}
        Some((nic, name)) => {
            let eth = &mut ifaces()[ETH0];
            let m = nic.mac();
            eth.nic = Some(nic);
            // QEMU's user-mode network puts the guest at 10.0.2.15, the
            // gateway at 10.0.2.2 and its resolver at 10.0.2.3. Defaulting to
            // that makes the first test work without configuring anything, and
            // `dhcp` replaces all of it with whatever the network says.
            eth.ip = [10, 0, 2, 15];
            eth.gateway = [10, 0, 2, 2];
            eth.netmask = [255, 255, 255, 0];
            eth.dns = [10, 0, 2, 3];
            eth.up = true;
            let up = eth.usable();
            kprintln!(
                "  eth0   {} {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}  link {}",
                name, m[0], m[1], m[2], m[3], m[4], m[5],
                if up { "up" } else { "down" }
            );
            kprintln!("         10.0.2.15 via 10.0.2.2  ('dhcp' to ask, 'if' to see)");
        }
    }

    match wifi::probe(ecam) {
        wifi::Probe::None => {}
        wifi::Probe::Unsupported { vendor, device, what } => {
            // Naming it is the point: the driver that is missing cannot be
            // written until the card is identified, and this is the only place
            // that identification happens.
            kprintln!("  wlan0  {} ({:04x}:{:04x}) -- no driver", what, vendor, device);
        }
    }
}

// --- checksums ----------------------------------------------------------

/// The one's-complement sum used by IPv4, ICMP, TCP and UDP.
///
/// Folding the carries in a loop rather than once: a single fold can leave a
/// carry behind, and the resulting checksum is wrong for about one packet in
/// sixty thousand -- which is exactly the kind of bug that looks like a flaky
/// network.
pub(crate) fn checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// The checksum TCP and UDP share, over the pseudo-header plus the segment.
///
/// The pseudo-header is never transmitted; it exists so that the checksum
/// covers the addresses and protocol, which is what catches a segment
/// delivered to the wrong host or handed to the wrong protocol.
pub(crate) fn transport_checksum(src: Ipv4, dst: Ipv4, proto: u8, segment: &[u8]) -> u16 {
    let mut buf = Vec::with_capacity(12 + segment.len());
    buf.extend_from_slice(&src);
    buf.extend_from_slice(&dst);
    buf.push(0);
    buf.push(proto);
    buf.extend_from_slice(&(segment.len() as u16).to_be_bytes());
    buf.extend_from_slice(segment);
    checksum(&buf)
}

// --- frames -------------------------------------------------------------

fn eth_frame(dst: Mac, src: Mac, ethertype: u16, payload: &[u8]) -> Vec<u8> {
    let mut f = Vec::with_capacity(14 + payload.len());
    f.extend_from_slice(&dst);
    f.extend_from_slice(&src);
    f.extend_from_slice(&ethertype.to_be_bytes());
    f.extend_from_slice(payload);
    // The card pads short frames itself (TCTL.PSP), so nothing is needed for
    // the 60-byte minimum.
    f
}

fn transmit_on(n: usize, frame: &[u8]) -> bool {
    let i = &mut ifaces()[n];
    let ok = i.nic.as_mut().map(|d| d.transmit(frame)).unwrap_or(false);
    if ok {
        i.stats.tx_packets += 1;
        i.stats.tx_bytes += frame.len() as u64;
    } else {
        i.stats.tx_dropped += 1;
    }
    ok
}

fn send_arp_request(n: usize, target: Ipv4) {
    let (mac, src_ip) = {
        let i = &ifaces()[n];
        match i.nic.as_ref() {
            None => return,
            Some(d) => (d.mac(), i.ip),
        }
    };

    let mut p = Vec::with_capacity(28);
    p.extend_from_slice(&1u16.to_be_bytes()); // Ethernet
    p.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
    p.push(6); // hardware length
    p.push(4); // protocol length
    p.extend_from_slice(&ARP_REQUEST.to_be_bytes());
    p.extend_from_slice(&mac);
    p.extend_from_slice(&src_ip);
    p.extend_from_slice(&[0u8; 6]);
    p.extend_from_slice(&target);

    let frame = eth_frame(BROADCAST, mac, ETHERTYPE_ARP, &p);
    transmit_on(n, &frame);
}

fn handle_arp(n: usize, payload: &[u8]) {
    if payload.len() < 28 {
        return;
    }
    let op = u16::from_be_bytes([payload[6], payload[7]]);
    let sender_mac: Mac = payload[8..14].try_into().unwrap_or([0; 6]);
    let sender_ip: Ipv4 = payload[14..18].try_into().unwrap_or([0; 4]);
    let target_ip: Ipv4 = payload[24..28].try_into().unwrap_or([0; 4]);

    // Learn from any ARP traffic, request or reply. A request tells us the
    // sender's mapping just as well as a reply does.
    ifaces()[n].arp_insert(sender_ip, sender_mac);

    let (mac, our_ip) = {
        let i = &ifaces()[n];
        match i.nic.as_ref() {
            None => return,
            Some(d) => (d.mac(), i.ip),
        }
    };
    if op == ARP_REQUEST && target_ip == our_ip {
        let mut p = Vec::with_capacity(28);
        p.extend_from_slice(&1u16.to_be_bytes());
        p.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        p.push(6);
        p.push(4);
        p.extend_from_slice(&ARP_REPLY.to_be_bytes());
        p.extend_from_slice(&mac);
        p.extend_from_slice(&our_ip);
        p.extend_from_slice(&sender_mac);
        p.extend_from_slice(&sender_ip);
        let frame = eth_frame(sender_mac, mac, ETHERTYPE_ARP, &p);
        transmit_on(n, &frame);
    }
}

fn ipv4_packet(src: Ipv4, dst: Ipv4, proto: u8, payload: &[u8], ident: u16) -> Vec<u8> {
    let total = 20 + payload.len();
    let mut h = Vec::with_capacity(total);
    h.push(0x45); // version 4, 5 words of header
    h.push(0); // DSCP/ECN
    h.extend_from_slice(&(total as u16).to_be_bytes());
    h.extend_from_slice(&ident.to_be_bytes());
    h.extend_from_slice(&0u16.to_be_bytes()); // no fragmentation
    h.push(64); // TTL
    h.push(proto);
    h.extend_from_slice(&[0, 0]); // checksum, filled below
    h.extend_from_slice(&src);
    h.extend_from_slice(&dst);
    // The header checksum covers the header only, and is computed with its own
    // field zeroed -- which is why it is written after the fact rather than
    // during construction.
    let c = checksum(&h);
    h[10..12].copy_from_slice(&c.to_be_bytes());
    h.extend_from_slice(payload);
    h
}

/// Outcome of one poll of the receive rings.
pub enum Event {
    None,
    Arp,
    EchoReply { from: Ipv4, seq: u16 },
    Tcp,
    Udp,
    Other,
}

/// Take one frame from whichever interface has one.
pub fn poll() -> Event {
    for n in 0..ifaces().len() {
        let frame = {
            let i = &mut ifaces()[n];
            if !i.up {
                continue;
            }
            match i.nic.as_mut().and_then(|d| d.receive()) {
                None => continue,
                Some(f) => {
                    i.stats.rx_packets += 1;
                    i.stats.rx_bytes += f.len() as u64;
                    f
                }
            }
        };
        return dispatch(n, &frame);
    }
    Event::None
}

fn dispatch(n: usize, frame: &[u8]) -> Event {
    if frame.len() < 14 {
        return Event::Other;
    }
    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    let payload = &frame[14..];

    match ethertype {
        ETHERTYPE_ARP => {
            handle_arp(n, payload);
            Event::Arp
        }
        ETHERTYPE_IPV4 => {
            if payload.len() < 20 {
                return Event::Other;
            }
            let ihl = ((payload[0] & 0x0F) as usize) * 4;
            let total = u16::from_be_bytes([payload[2], payload[3]]) as usize;
            // Trim to the length the IP header declares. Ethernet pads frames
            // to 60 bytes, so a bare 40-byte ACK arrives with 6 bytes of
            // trailing garbage -- and a TCP checksum computed over the padding
            // is wrong every time, which presents as a peer that answers
            // nothing. ICMP got away with ignoring this because its checksum
            // covers a length it carries itself.
            if ihl < 20 || total < ihl || payload.len() < total {
                return Event::Other;
            }
            let payload = &payload[..total];
            let src: Ipv4 = payload[12..16].try_into().unwrap_or([0; 4]);
            let dst: Ipv4 = payload[16..20].try_into().unwrap_or([0; 4]);
            if !addressed_to_us(n, dst) {
                return Event::Other;
            }

            if payload[9] == PROTO_TCP {
                tcp::deliver(src, dst, &payload[ihl..]);
                return Event::Tcp;
            }
            if payload[9] == PROTO_UDP {
                udp::deliver(src, dst, &payload[ihl..]);
                return Event::Udp;
            }
            if payload[9] != PROTO_ICMP {
                return Event::Other;
            }
            let icmp = &payload[ihl..];
            if icmp.len() < 8 {
                return Event::Other;
            }
            match icmp[0] {
                ICMP_ECHO_REPLY => Event::EchoReply {
                    from: src,
                    seq: u16::from_be_bytes([icmp[6], icmp[7]]),
                },
                ICMP_ECHO_REQUEST => {
                    // Answer it. A system that can be pinged is easier to
                    // diagnose from the other end than one that can only ping.
                    let mut reply = icmp.to_vec();
                    reply[0] = ICMP_ECHO_REPLY;
                    reply[2] = 0;
                    reply[3] = 0;
                    let c = checksum(&reply);
                    reply[2..4].copy_from_slice(&c.to_be_bytes());
                    send_ipv4(src, PROTO_ICMP, &reply);
                    Event::Other
                }
                _ => Event::Other,
            }
        }
        _ => Event::Other,
    }
}

/// Would a packet addressed here be ours?
///
/// Broadcast counts, and so does anything at all while the receiving
/// interface's address is still 0.0.0.0 -- during DHCP the reply is addressed
/// to an address we do not have yet, and refusing it would make the protocol
/// impossible to complete.
fn addressed_to_us(n: usize, dst: Ipv4) -> bool {
    if dst == BROADCAST_IP || is_local(dst) {
        return true;
    }
    let i = &ifaces()[n];
    if i.ip == UNSPECIFIED {
        return true;
    }
    // Directed broadcast for this interface's subnet, e.g. 10.0.2.255.
    i.on_subnet(dst) && (0..4).all(|k| dst[k] | i.netmask[k] == 255)
}

fn resolve(n: usize, target: Ipv4) -> Option<Mac> {
    // Loopback has no addressing of its own; the frame goes straight back.
    if ifaces()[n].nic.as_ref().map(|d| d.kind()) == Some(Kind::Loopback) {
        return Some([0; 6]);
    }
    // Broadcast is never resolved: there is nothing to ask, and DHCP has to
    // send this way before it owns an address to ask *from*.
    if target == BROADCAST_IP {
        return Some(BROADCAST);
    }

    let want = {
        let i = &ifaces()[n];
        // Anything off-link goes via this interface's gateway, which is the
        // whole of routing once the interface is chosen.
        if i.on_subnet(target) {
            target
        } else {
            i.gateway
        }
    };

    if let Some(mac) = ifaces()[n].arp_lookup(want) {
        return Some(mac);
    }

    send_arp_request(n, want);
    let deadline = crate::dev::lapic::ticks() + crate::TIMER_HZ as u64;
    while crate::dev::lapic::ticks() < deadline {
        poll();
        if let Some(mac) = ifaces()[n].arp_lookup(want) {
            return Some(mac);
        }
        core::hint::spin_loop();
    }
    None
}

pub(crate) fn send_ipv4(dst: Ipv4, proto: u8, payload: &[u8]) -> bool {
    let Some(n) = route(dst) else { return false };
    send_on(n, ifaces()[n].ip, dst, proto, payload)
}

/// Send with an explicit source address.
///
/// Exists for DHCP, which must send from 0.0.0.0 -- it is asking for the
/// address it would otherwise put in that field.
pub(crate) fn send_ipv4_from(src: Ipv4, dst: Ipv4, proto: u8, payload: &[u8]) -> bool {
    // With no address yet there is no subnet to match, so routing cannot pick
    // the interface; fall back to the primary one.
    let n = route(dst).unwrap_or_else(primary);
    send_on(n, src, dst, proto, payload)
}

fn send_on(n: usize, src: Ipv4, dst: Ipv4, proto: u8, payload: &[u8]) -> bool {
    let Some(mac) = ifaces()[n].nic.as_ref().map(|d| d.mac()) else { return false };
    let Some(dst_mac) = resolve(n, dst) else { return false };
    let packet = ipv4_packet(src, dst, proto, payload, 0x1234);
    let frame = eth_frame(dst_mac, mac, ETHERTYPE_IPV4, &packet);
    transmit_on(n, &frame)
}

pub fn ping(dst: Ipv4, count: u16) {
    console::set_color(YELLOW);
    kprintln!("[ping] {}.{}.{}.{}", dst[0], dst[1], dst[2], dst[3]);
    console::set_color(LTGRAY);
    if route(dst).is_none() {
        console::set_color(LTRED);
        kprintln!("  no route to that address");
        console::set_color(LTGRAY);
        return;
    }

    let mut sent = 0u16;
    let mut got = 0u16;
    for seq in 1..=count {
        let mut icmp = vec![0u8; 8 + 32];
        icmp[0] = ICMP_ECHO_REQUEST;
        icmp[4..6].copy_from_slice(&0xBEEFu16.to_be_bytes()); // identifier
        icmp[6..8].copy_from_slice(&seq.to_be_bytes());
        for (i, b) in icmp[8..].iter_mut().enumerate() {
            *b = b'a' + (i % 26) as u8;
        }
        let c = checksum(&icmp);
        icmp[2..4].copy_from_slice(&c.to_be_bytes());

        let t0 = crate::time::rdtsc();
        if !send_ipv4(dst, PROTO_ICMP, &icmp) {
            kprintln!("  seq {}: could not send (ARP failed?)", seq);
            continue;
        }
        sent += 1;

        let deadline = crate::dev::lapic::ticks() + crate::TIMER_HZ as u64;
        let mut answered = false;
        while crate::dev::lapic::ticks() < deadline {
            if let Event::EchoReply { from, seq: s } = poll() {
                if s == seq {
                    let us = crate::time::tsc_mhz();
                    let elapsed = crate::time::rdtsc() - t0;
                    console::set_color(LTGREEN);
                    if us > 0 {
                        kprintln!(
                            "  reply from {}.{}.{}.{}  seq {}  {} us",
                            from[0], from[1], from[2], from[3], s, elapsed / us
                        );
                    } else {
                        kprintln!("  reply from {}.{}.{}.{}  seq {}", from[0], from[1], from[2], from[3], s);
                    }
                    console::set_color(LTGRAY);
                    got += 1;
                    answered = true;
                    break;
                }
            }
            core::hint::spin_loop();
        }
        if !answered {
            console::set_color(YELLOW);
            kprintln!("  seq {}: timed out", seq);
            console::set_color(LTGRAY);
        }
    }

    console::set_color(if got == sent && sent > 0 { LTGREEN } else { YELLOW });
    kprintln!("  {} sent, {} received", sent, got);
    console::set_color(LTGRAY);
}

/// One line per interface, in the spirit of ifconfig.
pub fn report() {
    console::set_color(YELLOW);
    kprintln!("[interfaces]");
    console::set_color(LTGRAY);
    let def = route([8, 8, 8, 8]);
    for n in 0..ifaces().len() {
        let present = ifaces()[n].present();
        let usable = ifaces()[n].usable();
        let i = &ifaces()[n];
        if !present {
            console::set_color(LTGRAY);
            kprintln!("  {:<6} not present", i.name);
            continue;
        }
        let kind = i.nic.as_ref().map(|d| d.kind()).unwrap_or(Kind::Ethernet);
        console::set_color(if usable { LTGREEN } else { YELLOW });
        kprintln!(
            "  {:<6} {}  {}{}",
            i.name,
            kind.name(),
            if i.up { "up" } else { "down" },
            if def == Some(n) { "  (default route)" } else { "" }
        );
        console::set_color(LTGRAY);
        if kind != Kind::Loopback {
            let m = i.nic.as_ref().map(|d| d.mac()).unwrap_or([0; 6]);
            kprintln!(
                "         mac {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                m[0], m[1], m[2], m[3], m[4], m[5]
            );
        }
        kprintln!(
            "         inet {}.{}.{}.{}/{}",
            i.ip[0], i.ip[1], i.ip[2], i.ip[3],
            i.netmask.iter().map(|b| b.count_ones()).sum::<u32>()
        );
        if i.gateway != UNSPECIFIED {
            kprintln!(
                "         gw {}.{}.{}.{}   dns {}.{}.{}.{}",
                i.gateway[0], i.gateway[1], i.gateway[2], i.gateway[3],
                i.dns[0], i.dns[1], i.dns[2], i.dns[3]
            );
        }
        kprintln!(
            "         rx {} pkt / {} B     tx {} pkt / {} B{}",
            i.stats.rx_packets, i.stats.rx_bytes,
            i.stats.tx_packets, i.stats.tx_bytes,
            if i.stats.tx_dropped > 0 {
                " (drops)"
            } else {
                ""
            }
        );
        let learned = i.arp.iter().flatten().count();
        if learned > 0 {
            kprintln!("         arp {} entries", learned);
        }
    }
    if let Some((name, ip)) = dns::cached() {
        kprintln!("  resolver last saw {} at {}.{}.{}.{}", name, ip[0], ip[1], ip[2], ip[3]);
    }
    let _ = LTCYAN;
}

/// Parse "10.0.2.2" into four octets.
pub fn parse_ip(s: &str) -> Option<Ipv4> {
    let mut out = [0u8; 4];
    let mut n = 0;
    for part in s.trim().split('.') {
        if n >= 4 {
            return None;
        }
        out[n] = part.parse().ok()?;
        n += 1;
    }
    if n == 4 {
        Some(out)
    } else {
        None
    }
}

/// Turn a prefix length into a mask: 24 becomes 255.255.255.0.
pub fn mask_from_prefix(bits: u32) -> Ipv4 {
    let m: u32 = if bits >= 32 {
        u32::MAX
    } else {
        !(u32::MAX >> bits)
    };
    m.to_be_bytes()
}
