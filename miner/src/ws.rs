//! The WebSocket client the miner needs, and nothing past it.
//!
//! **Why this is written here rather than taken from a crate.** The public pool
//! is a Cloudflare Worker, reachable only as `wss://.../mine`, so the miner has
//! to speak WebSocket. tungstenite was the obvious choice and brought some sixty
//! crates with it -- URL parsing with its Unicode tables, `rand`, proc macros --
//! for a protocol that, as the pool speaks it, is one text message per line. So
//! this file is the HTTP upgrade and the frame format (RFC 6455), and TLS, the
//! part not to hand-roll, comes from rustls.
//!
//! Client frames are masked and server frames are not, as the RFC requires.
//! Fragmented messages are reassembled; pings are answered; a close ends the
//! connection, and the caller reconnects as it does for any drop.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use glados_pool::json::Json;

/// How long the TCP connect, the TLS handshake and the HTTP upgrade may take
/// together. The caller shortens the read timeout to its poll interval after.
const HANDSHAKE: Duration = Duration::from_secs(15);

/// The largest single message accepted. The pool's biggest is a job, which is
/// a few hundred bytes; anything near this is not the pool.
const MAX_MESSAGE: u64 = 1 << 20;

/// Seconds the pool has asked this miner to wait before trying again: a 503's
/// Retry-After (maintenance, or the pool over its allowance), a close with
/// 1013, or a `glados.maintenance` notice. The reconnect loops take it in place
/// of their own backoff. Capped, so a miner told "back in an hour" still looks
/// every two minutes -- an attempt during maintenance is answered at the
/// pool's edge and costs it nothing -- and is back soon after the pool is.
static HOLD: AtomicU64 = AtomicU64::new(0);
pub const HOLD_MAX: u64 = 120;

pub fn hold(secs: u64) {
    HOLD.fetch_max(secs.min(HOLD_MAX), Ordering::Relaxed);
}

pub fn take_hold() -> u64 {
    HOLD.swap(0, Ordering::Relaxed)
}

/// The bytes under the WebSocket: plain TCP for `ws://`, TLS for `wss://`.
pub enum Wire {
    Plain(TcpStream),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
}

impl Wire {
    pub fn tcp(&self) -> &TcpStream {
        match self {
            Wire::Plain(s) => s,
            Wire::Tls(t) => &t.sock,
        }
    }
}

impl Read for Wire {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Wire::Plain(s) => s.read(b),
            Wire::Tls(t) => t.read(b),
        }
    }
}

impl Write for Wire {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            Wire::Plain(s) => s.write(b),
            Wire::Tls(t) => t.write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Wire::Plain(s) => s.flush(),
            Wire::Tls(t) => t.flush(),
        }
    }
}

pub fn idle(e: &std::io::Error) -> bool {
    matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)
}

struct Url<'a> {
    tls: bool,
    host: &'a str,
    port: u16,
    path: &'a str,
}

fn parse(url: &str) -> Result<Url<'_>, String> {
    let (tls, rest) = if let Some(r) = url.strip_prefix("wss://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("ws://") {
        (false, r)
    } else {
        return Err(format!("{url}: not a ws:// or wss:// address"));
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| format!("{url}: bad port {p}"))?),
        None => (authority, if tls { 443 } else { 80 }),
    };
    if host.is_empty() {
        return Err(format!("{url}: no host"));
    }
    Ok(Url { tls, host, port, path })
}

/// Unpredictable bytes for the handshake key and the frame masks, from the
/// standard library's per-process random hash seed and the clock. Neither use
/// is a secret -- the mask exists so a client cannot steer what a proxy's cache
/// sees -- so this needs to be unguessable, not cryptographic.
fn random() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish()
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { T[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    s
}

fn tls(host: &str, tcp: TcpStream) -> Result<Wire, String> {
    let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("tls: {e}"))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = rustls::pki_types::ServerName::try_from(host.to_string()).map_err(|e| format!("tls: {host}: {e}"))?;
    let conn = rustls::ClientConnection::new(Arc::new(config), name).map_err(|e| format!("tls: {e}"))?;
    Ok(Wire::Tls(Box::new(rustls::StreamOwned::new(conn, tcp))))
}

pub struct Ws {
    wire: Wire,
    /// Bytes read and not yet a whole frame.
    inbuf: Vec<u8>,
    /// A fragmented message being reassembled.
    frag: Vec<u8>,
}

impl Ws {
    pub fn connect(url: &str, agent: &str) -> Result<Ws, String> {
        let u = parse(url)?;
        let tcp = TcpStream::connect((u.host, u.port)).map_err(|e| format!("{}:{}: {e}", u.host, u.port))?;
        tcp.set_read_timeout(Some(HANDSHAKE)).map_err(|e| format!("{e}"))?;
        tcp.set_write_timeout(Some(HANDSHAKE)).map_err(|e| format!("{e}"))?;
        let _ = tcp.set_nodelay(true); // a share is one small write; do not hold it back
        let mut wire = if u.tls { tls(u.host, tcp)? } else { Wire::Plain(tcp) };

        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&random().to_le_bytes());
        key[8..].copy_from_slice(&random().to_le_bytes());
        let host = if (u.tls && u.port == 443) || (!u.tls && u.port == 80) { u.host.to_string() } else { format!("{}:{}", u.host, u.port) };
        let req = format!(
            "GET {} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Key: {}\r\nSec-WebSocket-Version: 13\r\nUser-Agent: {agent}\r\n\r\n",
            u.path,
            base64(&key)
        );
        // The TLS handshake happens inside this first write.
        wire.write_all(req.as_bytes()).and_then(|_| wire.flush()).map_err(|e| format!("{url}: {e}"))?;

        let mut head = Vec::new();
        let mut chunk = [0u8; 4096];
        let end = loop {
            if let Some(i) = head.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
            if head.len() > 16384 {
                return Err(format!("{url}: no end to the upgrade response"));
            }
            match wire.read(&mut chunk) {
                Ok(0) => return Err(format!("{url}: closed during the upgrade")),
                Ok(n) => head.extend_from_slice(&chunk[..n]),
                Err(e) => return Err(format!("{url}: {e}")),
            }
        };
        let text = String::from_utf8_lossy(&head[..end]).to_ascii_lowercase();
        let status = text.lines().next().unwrap_or("");
        if !status.contains(" 101 ") {
            // A refusal carries its reason as JSON, and how long to stay away.
            if let Some(s) = text.lines().find_map(|l| l.strip_prefix("retry-after:")).and_then(|v| v.trim().parse::<u64>().ok()) {
                hold(s);
            }
            let want = text
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0)
                .min(4096);
            while head.len() < end + want {
                match wire.read(&mut chunk) {
                    Ok(n) if n > 0 => head.extend_from_slice(&chunk[..n]),
                    _ => break,
                }
            }
            let body = String::from_utf8_lossy(&head[end..]).to_string();
            if let Some(why) = Json::parse(body.trim()).and_then(|v| v.get("why").and_then(|w| w.as_str()).map(String::from)) {
                return Err(format!("{}: {}", status.trim(), why));
            }
            return Err(format!("{url}: the pool answered {:?} rather than switching to WebSocket", status.trim()));
        }
        if !text.contains("upgrade: websocket") {
            return Err(format!("{url}: switched protocols, but not to WebSocket"));
        }
        let inbuf = head[end..].to_vec(); // a frame may have come with the response
        Ok(Ws { wire, inbuf, frag: Vec::new() })
    }

    pub fn tcp(&self) -> &TcpStream {
        self.wire.tcp()
    }

    /// One read (bounded by the socket's read timeout), then every complete
    /// message handed to `out`. `Ok(false)` when the pool closed.
    pub fn poll(&mut self, out: &mut dyn FnMut(&[u8])) -> Result<bool, String> {
        let mut chunk = [0u8; 16384];
        match self.wire.read(&mut chunk) {
            Ok(0) => return Ok(false),
            Ok(n) => self.inbuf.extend_from_slice(&chunk[..n]),
            Err(e) if idle(&e) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(e) => return Err(format!("read: {e}")),
        }
        while let Some((fin, op, payload, used)) = frame(&self.inbuf)? {
            self.inbuf.drain(..used);
            match op {
                0x0..=0x2 => {
                    self.frag.extend_from_slice(&payload);
                    if self.frag.len() as u64 > MAX_MESSAGE {
                        return Err(String::from("a message larger than any the pool sends"));
                    }
                    if fin {
                        out(&self.frag);
                        self.frag.clear();
                    }
                }
                0x8 => {
                    // 1013, "try again later": the pool is going down on purpose.
                    if payload.len() >= 2 && u16::from_be_bytes([payload[0], payload[1]]) == 1013 {
                        hold(HOLD_MAX);
                        eprintln!("[miner] the pool closed the connection: {}", String::from_utf8_lossy(&payload[2..]));
                    }
                    let _ = self.frame_out(0x8, &payload);
                    return Ok(false);
                }
                0x9 => self.frame_out(0xA, &payload)?,
                0xA => {}
                _ => return Err(format!("a frame with unknown opcode {op:#x}")),
            }
        }
        Ok(true)
    }

    pub fn send(&mut self, text: &str) -> Result<(), String> {
        self.frame_out(0x1, text.as_bytes())
    }

    fn frame_out(&mut self, op: u8, data: &[u8]) -> Result<(), String> {
        let mut f = Vec::with_capacity(data.len() + 14);
        f.push(0x80 | op);
        match data.len() {
            n if n < 126 => f.push(0x80 | n as u8),
            n if n <= 0xffff => {
                f.push(0x80 | 126);
                f.extend_from_slice(&(n as u16).to_be_bytes());
            }
            n => {
                f.push(0x80 | 127);
                f.extend_from_slice(&(n as u64).to_be_bytes());
            }
        }
        let mask = (random() as u32).to_be_bytes();
        f.extend_from_slice(&mask);
        f.extend(data.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.wire.write_all(&f).and_then(|_| self.wire.flush()).map_err(|e| format!("{e}"))
    }
}

/// One frame from the front of `b`: (fin, opcode, payload, bytes used), or
/// `None` while it is still incomplete.
fn frame(b: &[u8]) -> Result<Option<(bool, u8, Vec<u8>, usize)>, String> {
    if b.len() < 2 {
        return Ok(None);
    }
    let fin = b[0] & 0x80 != 0;
    let op = b[0] & 0x0f;
    let masked = b[1] & 0x80 != 0;
    let mut len = (b[1] & 0x7f) as u64;
    let mut p = 2;
    if len == 126 {
        if b.len() < 4 {
            return Ok(None);
        }
        len = u16::from_be_bytes([b[2], b[3]]) as u64;
        p = 4;
    } else if len == 127 {
        if b.len() < 10 {
            return Ok(None);
        }
        len = u64::from_be_bytes(b[2..10].try_into().unwrap());
        p = 10;
    }
    if len > MAX_MESSAGE {
        return Err(String::from("a frame larger than any the pool sends"));
    }
    let mask = if masked {
        if b.len() < p + 4 {
            return Ok(None);
        }
        let m = [b[p], b[p + 1], b[p + 2], b[p + 3]];
        p += 4;
        Some(m)
    } else {
        None
    };
    let end = p + len as usize;
    if b.len() < end {
        return Ok(None);
    }
    let mut payload = b[p..end].to_vec();
    if let Some(m) = mask {
        for (i, x) in payload.iter_mut().enumerate() {
            *x ^= m[i % 4];
        }
    }
    Ok(Some((fin, op, payload, end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_rfc_4648_vectors() {
        for (i, o) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(i.as_bytes()), o);
        }
    }

    #[test]
    fn urls_split_into_their_parts() {
        let u = parse("wss://pool.aperture.institute/mine/gpu").unwrap();
        assert!(u.tls && u.host == "pool.aperture.institute" && u.port == 443 && u.path == "/mine/gpu");
        let u = parse("ws://127.0.0.1:18787").unwrap();
        assert!(!u.tls && u.host == "127.0.0.1" && u.port == 18787 && u.path == "/");
        assert!(parse("http://x/").is_err());
    }

    #[test]
    fn frames_parse_short_extended_masked_and_partial() {
        // RFC 6455 5.7: an unmasked "Hello" and the masked one.
        assert_eq!(frame(&[0x81, 0x05, b'H', b'e', b'l', b'l', b'o']).unwrap(), Some((true, 1, b"Hello".to_vec(), 7)));
        assert_eq!(
            frame(&[0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58]).unwrap(),
            Some((true, 1, b"Hello".to_vec(), 11))
        );
        // A fragment ("Hel" without FIN), and a 256-byte payload behind a 16-bit length.
        assert_eq!(frame(&[0x01, 0x03, b'H', b'e', b'l']).unwrap(), Some((false, 1, b"Hel".to_vec(), 5)));
        let mut big = vec![0x82, 0x7e, 0x01, 0x00];
        big.extend(std::iter::repeat(7u8).take(256));
        assert_eq!(frame(&big).unwrap().map(|f| (f.2.len(), f.3)), Some((256, 260)));
        assert_eq!(frame(&big[..100]).unwrap(), None);
        assert_eq!(frame(&[0x81]).unwrap(), None);
    }
}
