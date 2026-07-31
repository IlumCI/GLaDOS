//! DHCP: DISCOVER, OFFER, REQUEST, ACK.
//!
//! The four-message exchange of RFC 2131, which is what turns a machine that
//! has to be told its address into one that asks. Everything the rest of the
//! stack needs comes back in one reply: address, mask, router, and resolver.
//!
//! ### Sending before you have an address
//!
//! This is the protocol's one genuine oddity, and it shapes the code. The
//! client must transmit an IP packet in order to obtain an IP address, so
//! DISCOVER goes out from 0.0.0.0 to 255.255.255.255 -- which is why `net`
//! grew `send_ipv4_from` and why `resolve` short-circuits broadcast rather
//! than trying to ARP for it.
//!
//! The reply has the same problem in reverse: the server is answering a client
//! that cannot yet receive unicast, so we set the broadcast flag and accept
//! anything addressed to us while our own address is still unspecified.
//!
//! ### What is not here
//!
//! No lease renewal. The lease time is read and reported, and then nothing
//! watches it. A machine left running past its lease keeps using an address
//! the server believes is free -- fine for a session at a desk, wrong for
//! anything long-lived, and worth fixing before this is trusted on a network
//! it does not own.

use super::udp;
use super::{Config, Ipv4, BROADCAST_IP, UNSPECIFIED};
use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use alloc::vec::Vec;

const SERVER_PORT: u16 = 67;
const CLIENT_PORT: u16 = 68;

const OP_REQUEST: u8 = 1;
const OP_REPLY: u8 = 2;
const HTYPE_ETHERNET: u8 = 1;

const MAGIC: [u8; 4] = [99, 130, 83, 99];

const OPT_SUBNET_MASK: u8 = 1;
const OPT_ROUTER: u8 = 3;
const OPT_DNS: u8 = 6;
const OPT_REQUESTED_IP: u8 = 50;
const OPT_LEASE_TIME: u8 = 51;
const OPT_MESSAGE_TYPE: u8 = 53;
const OPT_SERVER_ID: u8 = 54;
const OPT_PARAM_LIST: u8 = 55;
const OPT_END: u8 = 255;

const DISCOVER: u8 = 1;
const OFFER: u8 = 2;
const REQUEST: u8 = 3;
const ACK: u8 = 5;
const NAK: u8 = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    NoNic,
    NoOffer,
    NoAck,
    Refused,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::NoNic => "no NIC",
            Error::NoOffer => "no server offered a lease",
            Error::NoAck => "the server never confirmed",
            Error::Refused => "the server refused the request",
        }
    }
}

#[derive(Default)]
struct Reply {
    kind: u8,
    yiaddr: Ipv4,
    mask: Option<Ipv4>,
    router: Option<Ipv4>,
    dns: Option<Ipv4>,
    server_id: Option<Ipv4>,
    lease: Option<u32>,
}

/// Build a BOOTP/DHCP message. The fixed part is 236 bytes whatever is in it.
fn message(kind: u8, xid: u32, mac: [u8; 6], requested: Option<Ipv4>, server: Option<Ipv4>) -> Vec<u8> {
    let mut m = Vec::with_capacity(300);
    m.push(OP_REQUEST);
    m.push(HTYPE_ETHERNET);
    m.push(6); // hardware address length
    m.push(0); // hops
    m.extend_from_slice(&xid.to_be_bytes());
    m.extend_from_slice(&0u16.to_be_bytes()); // seconds elapsed
    // Broadcast flag: we cannot receive a unicast reply until we have the
    // address the reply is carrying.
    m.extend_from_slice(&0x8000u16.to_be_bytes());
    m.extend_from_slice(&UNSPECIFIED); // ciaddr
    m.extend_from_slice(&UNSPECIFIED); // yiaddr
    m.extend_from_slice(&UNSPECIFIED); // siaddr
    m.extend_from_slice(&UNSPECIFIED); // giaddr
    m.extend_from_slice(&mac);
    m.extend_from_slice(&[0u8; 10]); // chaddr padding to 16
    m.extend_from_slice(&[0u8; 64]); // sname
    m.extend_from_slice(&[0u8; 128]); // file
    m.extend_from_slice(&MAGIC);

    m.push(OPT_MESSAGE_TYPE);
    m.push(1);
    m.push(kind);

    if let Some(ip) = requested {
        m.push(OPT_REQUESTED_IP);
        m.push(4);
        m.extend_from_slice(&ip);
    }
    if let Some(ip) = server {
        m.push(OPT_SERVER_ID);
        m.push(4);
        m.extend_from_slice(&ip);
    }

    m.push(OPT_PARAM_LIST);
    m.push(3);
    m.push(OPT_SUBNET_MASK);
    m.push(OPT_ROUTER);
    m.push(OPT_DNS);

    m.push(OPT_END);
    // Some servers ignore a BOOTP message shorter than the legacy 300 bytes.
    while m.len() < 300 {
        m.push(0);
    }
    m
}

fn parse(msg: &[u8], xid: u32) -> Option<Reply> {
    if msg.len() < 240 || msg[0] != OP_REPLY {
        return None;
    }
    if u32::from_be_bytes([msg[4], msg[5], msg[6], msg[7]]) != xid {
        // Another client's exchange on the same broadcast domain.
        return None;
    }
    if msg[236..240] != MAGIC {
        return None;
    }

    let mut r = Reply {
        yiaddr: [msg[16], msg[17], msg[18], msg[19]],
        ..Default::default()
    };

    let mut at = 240;
    while at < msg.len() {
        let code = msg[at];
        if code == OPT_END {
            break;
        }
        if code == 0 {
            at += 1; // pad
            continue;
        }
        if at + 1 >= msg.len() {
            break;
        }
        let len = msg[at + 1] as usize;
        let val = msg.get(at + 2..at + 2 + len)?;
        match code {
            OPT_MESSAGE_TYPE if len == 1 => r.kind = val[0],
            OPT_SUBNET_MASK if len == 4 => r.mask = Some([val[0], val[1], val[2], val[3]]),
            // Only the first router and the first resolver are kept; there is
            // one slot for each in Config.
            OPT_ROUTER if len >= 4 => r.router = Some([val[0], val[1], val[2], val[3]]),
            OPT_DNS if len >= 4 => r.dns = Some([val[0], val[1], val[2], val[3]]),
            OPT_SERVER_ID if len == 4 => r.server_id = Some([val[0], val[1], val[2], val[3]]),
            OPT_LEASE_TIME if len == 4 => {
                r.lease = Some(u32::from_be_bytes([val[0], val[1], val[2], val[3]]))
            }
            _ => {}
        }
        at += 2 + len;
    }
    Some(r)
}

/// Wait for a reply of the wanted type, ignoring anything else on the port.
fn await_reply(xid: u32, want: u8, ms: u64) -> Option<Reply> {
    let deadline = crate::dev::lapic::ticks() + (ms * crate::TIMER_HZ as u64) / 1000 + 1;
    loop {
        let remaining = deadline.saturating_sub(crate::dev::lapic::ticks());
        if remaining == 0 {
            return None;
        }
        let d = udp::recv(remaining * 1000 / crate::TIMER_HZ as u64)?;
        if d.src_port != SERVER_PORT {
            continue;
        }
        if let Some(r) = parse(&d.data, xid) {
            if r.kind == want {
                return Some(r);
            }
            if r.kind == NAK {
                return Some(r);
            }
        }
    }
}

/// Run the exchange and adopt whatever comes back.
pub fn configure() -> Result<Config, Error> {
    if !super::ready() {
        return Err(Error::NoNic);
    }
    let mac = super::mac().ok_or(Error::NoNic)?;
    let xid = crate::time::rdtsc() as u32;

    // Give up our address for the duration. It is not ours until the server
    // says so, and `addressed_to_us` lets everything through while it is
    // unspecified -- which is exactly what receiving the reply requires.
    let previous = super::config();
    let mut blank = previous;
    blank.ip = UNSPECIFIED;
    super::set_config(blank);
    udp::bind(CLIENT_PORT);

    let outcome = (|| {
        let discover = message(DISCOVER, xid, mac, None, None);
        if !udp::send_from(UNSPECIFIED, BROADCAST_IP, SERVER_PORT, CLIENT_PORT, &discover) {
            return Err(Error::NoOffer);
        }
        let offer = await_reply(xid, OFFER, 4000).ok_or(Error::NoOffer)?;
        if offer.kind == NAK {
            return Err(Error::Refused);
        }

        let request = message(REQUEST, xid, mac, Some(offer.yiaddr), offer.server_id);
        if !udp::send_from(UNSPECIFIED, BROADCAST_IP, SERVER_PORT, CLIENT_PORT, &request) {
            return Err(Error::NoAck);
        }
        let ack = await_reply(xid, ACK, 4000).ok_or(Error::NoAck)?;
        if ack.kind == NAK {
            return Err(Error::Refused);
        }

        // Anything the server did not supply keeps whatever was configured
        // before, rather than becoming zero.
        Ok((
            Config {
                ip: ack.yiaddr,
                netmask: ack.mask.unwrap_or(previous.netmask),
                gateway: ack.router.unwrap_or(previous.gateway),
                dns: ack.dns.unwrap_or(previous.dns),
            },
            ack.lease,
        ))
    })();

    udp::unbind();

    match outcome {
        Ok((cfg, lease)) => {
            super::set_config(cfg);
            unsafe { LAST_LEASE = lease };
            Ok(cfg)
        }
        Err(e) => {
            // Put back what was working before rather than leaving the machine
            // with no address because a server did not answer.
            super::set_config(previous);
            Err(e)
        }
    }
}

static mut LAST_LEASE: Option<u32> = None;

pub fn report() {
    console::set_color(YELLOW);
    kprintln!("[dhcp]");
    console::set_color(LTGRAY);
    match configure() {
        Err(e) => {
            console::set_color(LTRED);
            kprintln!("  {}", e.name());
            console::set_color(LTGRAY);
        }
        Ok(c) => {
            console::set_color(LTGREEN);
            kprintln!("  ip   {}.{}.{}.{}", c.ip[0], c.ip[1], c.ip[2], c.ip[3]);
            console::set_color(LTGRAY);
            kprintln!("  mask {}.{}.{}.{}", c.netmask[0], c.netmask[1], c.netmask[2], c.netmask[3]);
            kprintln!("  gw   {}.{}.{}.{}", c.gateway[0], c.gateway[1], c.gateway[2], c.gateway[3]);
            kprintln!("  dns  {}.{}.{}.{}", c.dns[0], c.dns[1], c.dns[2], c.dns[3]);
            match unsafe { LAST_LEASE } {
                Some(s) => kprintln!("  lease {} s  (not renewed -- see the module note)", s),
                None => kprintln!("  no lease time offered"),
            }
        }
    }
}
