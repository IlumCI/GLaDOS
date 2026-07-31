//! Ethernet, ARP, IPv4 and ICMP, with TCP in `tcp`.
//!
//! `ping` was the first milestone because it exercises the whole path in one
//! visible result: PCI discovery, an MMIO mapping, DMA rings, frame parsing,
//! ARP resolution, an IPv4 header with a correct checksum, and a reply that
//! has to come back. Anything broken anywhere in that stack shows up as
//! silence, and silence is easy to bisect when the alternative is a TCP state
//! machine failing intermittently. That ordering paid for itself.
//!
//! Still no DHCP and no DNS: the address is configured by hand and peers are
//! named by number. Both need UDP, which does not exist yet.
//!
//! ### Why TCP segments are queued rather than handled inline
//!
//! `poll` does not hand a segment straight to the TCP state machine. It pushes
//! it onto an inbox that `tcp::pump` drains later. The reason is re-entrancy:
//! sending anything calls `send_ipv4` -> `resolve`, and `resolve` calls `poll`
//! while it waits for an ARP reply. If `poll` ran the state machine directly,
//! a connection could re-enter its own control block through that path while
//! an earlier borrow was still live. Queueing breaks the cycle at the one
//! place it can form.

use crate::dev::e1000::E1000;
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use crate::sync::Racy;
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

pub mod dhcp;
pub mod dns;
pub mod tcp;
pub mod udp;

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

static NIC: Racy<Option<E1000>> = Racy::new(None);
static CONFIG: Racy<Config> = Racy::new(Config {
    // QEMU's user-mode network puts the guest at 10.0.2.15, the gateway at
    // 10.0.2.2 and its DNS at 10.0.2.3. Defaulting to that makes the first
    // test work without configuring anything -- and `dhcp` replaces all of it
    // with whatever the network actually says.
    ip: [10, 0, 2, 15],
    gateway: [10, 0, 2, 2],
    netmask: [255, 255, 255, 0],
    dns: [10, 0, 2, 3],
});

#[derive(Clone, Copy)]
pub struct Config {
    pub ip: Ipv4,
    pub gateway: Ipv4,
    pub netmask: Ipv4,
    pub dns: Ipv4,
}

pub const UNSPECIFIED: Ipv4 = [0, 0, 0, 0];
pub const BROADCAST_IP: Ipv4 = [255, 255, 255, 255];

/// A very small ARP cache. One entry is enough to ping a gateway, and a table
/// that never evicts is a table that eventually holds something wrong.
static ARP_CACHE: Racy<Option<(Ipv4, Mac)>> = Racy::new(None);

pub fn config() -> Config {
    unsafe { *CONFIG.get() }
}

pub fn set_config(c: Config) {
    unsafe { *CONFIG.get() = c };
}

pub fn ready() -> bool {
    unsafe { NIC.get().is_some() }
}

pub fn mac() -> Option<Mac> {
    with_nic(|n| n.mac())
}

pub fn init(ecam: u64) {
    console::set_color(YELLOW);
    kprintln!("\n[net]");
    console::set_color(LTGRAY);
    match crate::dev::e1000::probe(ecam) {
        Err(e) => {
            kprintln!("  no supported NIC ({:?})", e);
        }
        Ok(nic) => {
            let m = nic.mac();
            kprintln!(
                "  e1000 {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}  link {}",
                m[0], m[1], m[2], m[3], m[4], m[5],
                if nic.link_up() { "up" } else { "down" }
            );
            let c = config();
            kprintln!(
                "  {}.{}.{}.{} via {}.{}.{}.{}  ('net' to change)",
                c.ip[0], c.ip[1], c.ip[2], c.ip[3],
                c.gateway[0], c.gateway[1], c.gateway[2], c.gateway[3]
            );
            unsafe { *NIC.get() = Some(nic) };
        }
    }
}

fn with_nic<R>(f: impl FnOnce(&mut E1000) -> R) -> Option<R> {
    unsafe { NIC.get().as_mut().map(f) }
}

// --- checksums ----------------------------------------------------------

/// The one's-complement sum used by IPv4 and ICMP.
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

fn send_arp_request(target: Ipv4) {
    let cfg = config();
    let Some(mac) = with_nic(|n| n.mac()) else { return };

    let mut p = Vec::with_capacity(28);
    p.extend_from_slice(&1u16.to_be_bytes()); // Ethernet
    p.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
    p.push(6); // hardware length
    p.push(4); // protocol length
    p.extend_from_slice(&ARP_REQUEST.to_be_bytes());
    p.extend_from_slice(&mac);
    p.extend_from_slice(&cfg.ip);
    p.extend_from_slice(&[0u8; 6]);
    p.extend_from_slice(&target);

    let frame = eth_frame(BROADCAST, mac, ETHERTYPE_ARP, &p);
    with_nic(|n| n.transmit(&frame));
}

fn handle_arp(payload: &[u8]) {
    if payload.len() < 28 {
        return;
    }
    let op = u16::from_be_bytes([payload[6], payload[7]]);
    let sender_mac: Mac = payload[8..14].try_into().unwrap_or([0; 6]);
    let sender_ip: Ipv4 = payload[14..18].try_into().unwrap_or([0; 4]);
    let target_ip: Ipv4 = payload[24..28].try_into().unwrap_or([0; 4]);
    let cfg = config();

    // Learn from any ARP traffic, request or reply. A request tells us the
    // sender's mapping just as well as a reply does.
    unsafe { *ARP_CACHE.get() = Some((sender_ip, sender_mac)) };

    if op == ARP_REQUEST && target_ip == cfg.ip {
        let Some(mac) = with_nic(|n| n.mac()) else { return };
        let mut p = Vec::with_capacity(28);
        p.extend_from_slice(&1u16.to_be_bytes());
        p.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        p.push(6);
        p.push(4);
        p.extend_from_slice(&ARP_REPLY.to_be_bytes());
        p.extend_from_slice(&mac);
        p.extend_from_slice(&cfg.ip);
        p.extend_from_slice(&sender_mac);
        p.extend_from_slice(&sender_ip);
        let frame = eth_frame(sender_mac, mac, ETHERTYPE_ARP, &p);
        with_nic(|n| n.transmit(&frame));
    }
}

fn ipv4_packet(src: Ipv4, dst: Ipv4, proto: u8, payload: &[u8], ident: u16) -> Vec<u8> {
    let total = 20 + payload.len();
    let mut h = Vec::with_capacity(total);
    h.push(0x45); // version 4, 5 words of header
    h.push(0);    // DSCP/ECN
    h.extend_from_slice(&(total as u16).to_be_bytes());
    h.extend_from_slice(&ident.to_be_bytes());
    h.extend_from_slice(&0u16.to_be_bytes()); // no fragmentation
    h.push(64);   // TTL
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

/// Outcome of one poll of the receive ring.
pub enum Event {
    None,
    Arp,
    EchoReply { from: Ipv4, seq: u16 },
    Tcp,
    Udp,
    Other,
}

pub fn poll() -> Event {
    let Some(Some(frame)) = with_nic(|n| n.receive()) else {
        return Event::None;
    };
    if frame.len() < 14 {
        return Event::Other;
    }
    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    let payload = &frame[14..];

    match ethertype {
        ETHERTYPE_ARP => {
            handle_arp(payload);
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
            if !addressed_to_us(dst) {
                return Event::Other;
            }

            if payload[9] == PROTO_TCP {
                tcp::deliver(src, &payload[ihl..]);
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

fn resolve(target: Ipv4) -> Option<Mac> {
    let cfg = config();
    // Broadcast is never resolved: there is nothing to ask, and DHCP has to
    // send this way before it owns an address to ask *from*.
    if target == BROADCAST_IP {
        return Some(BROADCAST);
    }
    // Anything off-link goes via the gateway, which is the whole of routing
    // for a host with one interface.
    let same_subnet = (0..4).all(|i| target[i] & cfg.netmask[i] == cfg.ip[i] & cfg.netmask[i]);
    let want = if same_subnet { target } else { cfg.gateway };

    if let Some((ip, mac)) = unsafe { *ARP_CACHE.get() } {
        if ip == want {
            return Some(mac);
        }
    }

    send_arp_request(want);
    let deadline = crate::dev::lapic::ticks() + crate::TIMER_HZ as u64;
    while crate::dev::lapic::ticks() < deadline {
        poll();
        if let Some((ip, mac)) = unsafe { *ARP_CACHE.get() } {
            if ip == want {
                return Some(mac);
            }
        }
        core::hint::spin_loop();
    }
    None
}

pub(crate) fn send_ipv4(dst: Ipv4, proto: u8, payload: &[u8]) -> bool {
    send_ipv4_from(config().ip, dst, proto, payload)
}

/// Send with an explicit source address.
///
/// Exists for DHCP, which must send from 0.0.0.0 -- it is asking for the
/// address it would otherwise put in that field.
pub(crate) fn send_ipv4_from(src: Ipv4, dst: Ipv4, proto: u8, payload: &[u8]) -> bool {
    let Some(mac) = with_nic(|n| n.mac()) else { return false };
    let Some(dst_mac) = resolve(dst) else { return false };
    let packet = ipv4_packet(src, dst, proto, payload, 0x1234);
    let frame = eth_frame(dst_mac, mac, ETHERTYPE_IPV4, &packet);
    with_nic(|n| n.transmit(&frame)).unwrap_or(false)
}

/// Would a packet addressed here be ours?
///
/// Broadcast counts, and so does anything at all while our address is still
/// 0.0.0.0 -- during DHCP the reply is addressed to an address we do not have
/// yet, and refusing it would make the protocol impossible to complete.
fn addressed_to_us(dst: Ipv4) -> bool {
    let cfg = config();
    if dst == BROADCAST_IP || dst == cfg.ip || cfg.ip == UNSPECIFIED {
        return true;
    }
    // Directed broadcast for our subnet, e.g. 10.0.2.255.
    (0..4).all(|i| dst[i] | cfg.netmask[i] == 255 || dst[i] == cfg.ip[i])
        && (0..4).all(|i| dst[i] & cfg.netmask[i] == cfg.ip[i] & cfg.netmask[i])
}

pub fn ping(dst: Ipv4, count: u16) {
    console::set_color(YELLOW);
    kprintln!("[ping] {}.{}.{}.{}", dst[0], dst[1], dst[2], dst[3]);
    console::set_color(LTGRAY);
    if !ready() {
        console::set_color(LTRED);
        kprintln!("  no NIC");
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
    let _ = LTCYAN;
}

pub fn report() {
    console::set_color(YELLOW);
    kprintln!("[net]");
    console::set_color(LTGRAY);
    match with_nic(|n| (n.mac(), n.link_up())) {
        None => kprintln!("  no NIC"),
        Some((m, up)) => {
            kprintln!(
                "  mac  {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}   link {}",
                m[0], m[1], m[2], m[3], m[4], m[5],
                if up { "up" } else { "down" }
            );
            let c = config();
            kprintln!("  ip   {}.{}.{}.{}", c.ip[0], c.ip[1], c.ip[2], c.ip[3]);
            kprintln!("  gw   {}.{}.{}.{}", c.gateway[0], c.gateway[1], c.gateway[2], c.gateway[3]);
            kprintln!("  dns  {}.{}.{}.{}", c.dns[0], c.dns[1], c.dns[2], c.dns[3]);
            if let Some((name, ip)) = dns::cached() {
                kprintln!("  last {} is {}.{}.{}.{}", name, ip[0], ip[1], ip[2], ip[3]);
            }
            match unsafe { *ARP_CACHE.get() } {
                Some((ip, mac)) => kprintln!(
                    "  arp  {}.{}.{}.{} is {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                    ip[0], ip[1], ip[2], ip[3], mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
                ),
                None => kprintln!("  arp  nothing learned yet"),
            }
        }
    }
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
