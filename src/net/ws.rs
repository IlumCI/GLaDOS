//! WebSocket over TLS, client side, RFC 6455.
//!
//! Written for Discord's gateway, which is the only thing here that needs a
//! socket that stays open and talks in both directions. HTTP could poll; a chat
//! client that polls is a chat client that is always a little bit wrong.
//!
//! ### What is implemented, and what is not
//!
//! Text, binary, close, ping and pong. Continuation frames are reassembled.
//! Not implemented: extensions (`permessage-deflate` in particular), which is
//! why the gateway is opened without compression -- Discord sends plain JSON if
//! `compress` is not requested, and a deflate implementation to save bandwidth
//! this machine is not short of would be a large amount of code to get wrong.
//!
//! ### Masking is not optional
//!
//! Every client frame must be masked with a fresh 4-byte key, and a server is
//! required to close the connection on an unmasked one. It exists to stop a
//! hostile page poisoning intermediate caches, which is irrelevant here, but
//! the requirement is enforced by the peer rather than by taste -- an unmasked
//! frame gets a close and no explanation.
//!
//! The key comes from the TSC, like the rest of this system's randomness, and
//! is the same known weakness recorded against key material generally. For
//! masking specifically it costs nothing: the key is sent in the clear inside
//! the frame, so it is obfuscation with a defined purpose and no secrecy.

use super::tls;
use alloc::string::String;
use alloc::vec::Vec;

/// The fixed GUID from RFC 6455, appended to the client key before hashing.
const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

const OP_CONT: u8 = 0x0;
const OP_TEXT: u8 = 0x1;
const OP_BIN: u8 = 0x2;
const OP_CLOSE: u8 = 0x8;
const OP_PING: u8 = 0x9;
const OP_PONG: u8 = 0xA;

/// Refuse a frame larger than this.
///
/// A gateway READY for a large account is a few hundred kilobytes; a length
/// field claiming four gigabytes is a corrupt stream or a hostile one, and
/// `try_reserve` would fail far too late to be useful.
const MAX_FRAME: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub enum Error {
    Tls(tls::Error),
    Handshake,
    /// The server's `Sec-WebSocket-Accept` did not match. Either it is not a
    /// WebSocket endpoint or something is between us and it.
    BadAccept,
    Protocol,
    TooLarge,
    Closed,
}

impl From<tls::Error> for Error {
    fn from(e: tls::Error) -> Self {
        Error::Tls(e)
    }
}

pub fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

fn tsc_bytes(out: &mut [u8]) {
    let mut s = crate::time::rdtsc();
    for b in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s >> 24) as u8;
    }
}

/// A message reassembled from one or more frames.
pub enum Msg {
    Text(String),
    Binary(Vec<u8>),
    /// The peer closed. The code is the RFC 6455 status, when it sent one.
    Close(u16),
}

pub struct Socket {
    tls: tls::Session,
    /// Bytes read from TLS that did not form a whole frame yet.
    ///
    /// TLS records and WebSocket frames have nothing to do with one another --
    /// a record can split a frame or carry three -- so the framing layer needs
    /// its own buffer. Treating one `recv` as one frame is the mistake this
    /// exists to prevent.
    buf: Vec<u8>,
    /// Payload of a fragmented message in progress, and its opcode.
    partial: Vec<u8>,
    partial_op: u8,
}

impl Socket {
    /// Open a WebSocket to `host`, already resolved to `dst`.
    pub fn connect(dst: super::Ipv4, host: &str, port: u16, path: &str) -> Result<Socket, Error> {
        let mut s = tls::connect(dst, host, port)?;

        let mut nonce = [0u8; 16];
        tsc_bytes(&mut nonce);
        let key = base64(&nonce);

        let req = alloc::format!(
            "GET {} HTTP/1.1\r\n\
             Host: {}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {}\r\n\
             Sec-WebSocket-Version: 13\r\n\
             User-Agent: GLaDOS\r\n\
             \r\n",
            path, host, key
        );
        s.send(req.as_bytes())?;

        // Read until the header block ends. The 101 response is small, but it
        // can still arrive split across records.
        let mut head: Vec<u8> = Vec::new();
        loop {
            let chunk = s.recv(8000)?;
            if chunk.is_empty() {
                return Err(Error::Handshake);
            }
            head.extend_from_slice(&chunk);
            if let Some(end) = find(&head, b"\r\n\r\n") {
                let text = String::from_utf8_lossy(&head[..end]).into_owned();
                if !text.starts_with("HTTP/1.1 101") {
                    return Err(Error::Handshake);
                }
                let want = {
                    let mut h = crate::crypto::sha1::Sha1::new();
                    h.update(key.as_bytes());
                    h.update(GUID.as_bytes());
                    base64(&h.finish())
                };
                // Checked rather than assumed: a 101 from something that is not
                // a WebSocket endpoint would otherwise be followed by parsing
                // its HTML as frames.
                let got = header(&text, "sec-websocket-accept").unwrap_or_default();
                if got != want {
                    return Err(Error::BadAccept);
                }
                // Anything after the blank line is already frame data.
                let rest = head[end + 4..].to_vec();
                return Ok(Socket { tls: s, buf: rest, partial: Vec::new(), partial_op: 0 });
            }
            if head.len() > 16 * 1024 {
                return Err(Error::Handshake);
            }
        }
    }

    pub fn send_text(&mut self, text: &str) -> Result<(), Error> {
        let f = frame(OP_TEXT, text.as_bytes());
        self.tls.send(&f)?;
        Ok(())
    }

    fn send_raw(&mut self, op: u8, payload: &[u8]) -> Result<(), Error> {
        let f = frame(op, payload);
        self.tls.send(&f)?;
        Ok(())
    }

    pub fn close(&mut self) {
        // 1000 = normal. Best effort; the connection is going away regardless.
        let _ = self.send_raw(OP_CLOSE, &[0x03, 0xE8]);
        self.tls.close();
    }

    /// Next message, or `None` if none arrived within the timeout.
    ///
    /// Ping is answered here rather than handed up. A caller that forgot to
    /// pong would be disconnected minutes later for a reason with no visible
    /// connection to the omission.
    pub fn recv(&mut self, timeout_ms: u64) -> Result<Option<Msg>, Error> {
        loop {
            match take_frame(&mut self.buf)? {
                Some((fin, op, payload)) => {
                    match op {
                        OP_PING => {
                            self.send_raw(OP_PONG, &payload)?;
                            continue;
                        }
                        OP_PONG => continue,
                        OP_CLOSE => {
                            let code = if payload.len() >= 2 {
                                ((payload[0] as u16) << 8) | payload[1] as u16
                            } else {
                                1005
                            };
                            return Ok(Some(Msg::Close(code)));
                        }
                        OP_TEXT | OP_BIN => {
                            if fin {
                                return Ok(Some(finish(op, payload)));
                            }
                            self.partial_op = op;
                            self.partial = payload;
                            continue;
                        }
                        OP_CONT => {
                            self.partial.extend_from_slice(&payload);
                            if fin {
                                let op = self.partial_op;
                                let done = core::mem::take(&mut self.partial);
                                return Ok(Some(finish(op, done)));
                            }
                            continue;
                        }
                        _ => return Err(Error::Protocol),
                    }
                }
                None => {
                    let chunk = self.tls.recv(timeout_ms)?;
                    if chunk.is_empty() {
                        return Ok(None);
                    }
                    self.buf.extend_from_slice(&chunk);
                }
            }
        }
    }

}

/// Pull one whole frame out of a buffer, if there is a whole one in it.
///
/// A free function rather than a method so it can be tested without a TLS
/// session. The alternative was a `Socket` holding a zeroed `Session` -- a
/// struct full of null `Vec` pointers, constructed purely to reach a function
/// that never touches it. Splitting a frame across reads is the exact case
/// this has to get right, so it has to be reachable from a selftest.
fn take_frame(buf: &mut Vec<u8>) -> Result<Option<(bool, u8, Vec<u8>)>, Error> {
        let b = &buf;
        if b.len() < 2 {
            return Ok(None);
        }
        let fin = b[0] & 0x80 != 0;
        let op = b[0] & 0x0F;
        // A server frame is never masked; if this bit is set the stream is not
        // what it claims to be.
        if b[1] & 0x80 != 0 {
            return Err(Error::Protocol);
        }
        let short = (b[1] & 0x7F) as usize;
        let (len, head) = match short {
            126 => {
                if b.len() < 4 {
                    return Ok(None);
                }
                (((b[2] as usize) << 8) | b[3] as usize, 4)
            }
            127 => {
                if b.len() < 10 {
                    return Ok(None);
                }
                let mut n = 0usize;
                for i in 0..8 {
                    n = (n << 8) | b[2 + i] as usize;
                }
                (n, 10)
            }
            n => (n, 2),
        };
        if len > MAX_FRAME {
            return Err(Error::TooLarge);
        }
        if b.len() < head + len {
            return Ok(None);
        }
        let payload = b[head..head + len].to_vec();
        buf.drain(..head + len);
        Ok(Some((fin, op, payload)))
}

fn finish(op: u8, payload: Vec<u8>) -> Msg {
    if op == OP_TEXT {
        Msg::Text(String::from_utf8_lossy(&payload).into_owned())
    } else {
        Msg::Binary(payload)
    }
}

/// Encode one client frame: always final, always masked.
pub fn frame(op: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x80 | op);
    let n = payload.len();
    if n < 126 {
        out.push(0x80 | n as u8);
    } else if n <= 0xFFFF {
        out.push(0x80 | 126);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    } else {
        out.push(0x80 | 127);
        for i in (0..8).rev() {
            out.push((n >> (i * 8)) as u8);
        }
    }
    let mut mask = [0u8; 4];
    tsc_bytes(&mut mask);
    out.extend_from_slice(&mask);
    for (i, byte) in payload.iter().enumerate() {
        out.push(byte ^ mask[i % 4]);
    }
    out
}

/// Build an unmasked frame, the way a server does.
///
/// Only the selftest needs this -- a client never sends one, and a server that
/// received one would close the connection. It exists because `take_frame` is
/// the half that has to be right about splitting, and feeding it client frames
/// tests the wrong direction: it rejects them, correctly, for being masked.
pub fn server_frame(op: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x80 | op);
    let n = payload.len();
    if n < 126 {
        out.push(n as u8);
    } else if n <= 0xFFFF {
        out.push(126);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    } else {
        out.push(127);
        for i in (0..8).rev() {
            out.push((n >> (i * 8)) as u8);
        }
    }
    out.extend_from_slice(payload);
    out
}

/// Decode a client frame. Only used by the selftest -- a client never receives
/// masked frames -- but it is what makes `frame` checkable without a network.
pub fn unframe(f: &[u8]) -> Option<(u8, Vec<u8>)> {
    if f.len() < 2 {
        return None;
    }
    let op = f[0] & 0x0F;
    let masked = f[1] & 0x80 != 0;
    let short = (f[1] & 0x7F) as usize;
    let (len, mut i) = match short {
        126 => (((f[2] as usize) << 8) | f[3] as usize, 4),
        127 => {
            let mut n = 0usize;
            for k in 0..8 {
                n = (n << 8) | f[2 + k] as usize;
            }
            (n, 10)
        }
        n => (n, 2),
    };
    let mask = if masked {
        let m = [f[i], f[i + 1], f[i + 2], f[i + 3]];
        i += 4;
        Some(m)
    } else {
        None
    };
    if f.len() < i + len {
        return None;
    }
    let mut out = Vec::with_capacity(len);
    for k in 0..len {
        let b = f[i + k];
        out.push(match mask {
            Some(m) => b ^ m[k % 4],
            None => b,
        });
    }
    Some((op, out))
}

fn find(h: &[u8], needle: &[u8]) -> Option<usize> {
    h.windows(needle.len()).position(|w| w == needle)
}

/// Case-insensitive header lookup. HTTP header names are not case sensitive and
/// servers do vary -- matching `Sec-WebSocket-Accept` exactly would work until
/// it did not.
fn header(head: &str, name: &str) -> Option<String> {
    for line in head.split("\r\n").skip(1) {
        let (k, v) = line.split_once(':')?;
        if k.trim().eq_ignore_ascii_case(name) {
            return Some(String::from(v.trim()));
        }
    }
    None
}

pub fn selftest() -> bool {
    let mut ok = true;
    let mut check = |name: &str, pass: bool| {
        if !pass {
            crate::kprintln!("  FAIL  ws {}", name);
            ok = false;
        }
    };

    // RFC 4648 vectors.
    check("base64 empty", base64(b"") == "");
    check("base64 f", base64(b"f") == "Zg==");
    check("base64 fo", base64(b"fo") == "Zm8=");
    check("base64 foobar", base64(b"foobar") == "Zm9vYmFy");

    // RFC 6455 section 1.3: the one worked example in the specification.
    let want = {
        let mut h = crate::crypto::sha1::Sha1::new();
        h.update(b"dGhlIHNhbXBsZSBub25jZQ==");
        h.update(GUID.as_bytes());
        base64(&h.finish())
    };
    check("rfc6455 accept", want == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");

    // Framing round-trips, at each of the three length encodings.
    for n in [0usize, 5, 125, 126, 200, 65535, 65536, 70000] {
        let payload: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
        let f = frame(OP_TEXT, &payload);
        // Masked, always -- a server closes the connection on an unmasked
        // client frame and says nothing about why.
        let masked = f.len() > 1 && f[1] & 0x80 != 0;
        match unframe(&f) {
            Some((op, got)) => check(
                "frame round-trip",
                masked && op == OP_TEXT && got == payload,
            ),
            None => check("frame round-trip", false),
        }
    }

    // A frame split across reads must be reported as "not yet", never as a
    // frame -- TLS records and WebSocket frames have no relationship, so this
    // is the normal case rather than an edge one.
    let f = server_frame(OP_TEXT, b"hello world");
    let mut half = f[..5].to_vec();
    check("partial frame withheld", matches!(take_frame(&mut half), Ok(None)));
    // And the rest completes it, with the buffer consumed exactly.
    half.extend_from_slice(&f[5..]);
    let done = take_frame(&mut half);
    check(
        "frame completes on the rest",
        matches!(&done, Ok(Some((true, OP_TEXT, p))) if p == b"hello world") && half.is_empty(),
    );
    // Two frames in one buffer: the second must survive the first being taken.
    let mut two = server_frame(OP_TEXT, b"one");
    two.extend_from_slice(&server_frame(OP_TEXT, b"two"));
    let _ = take_frame(&mut two);
    check(
        "second frame survives",
        matches!(take_frame(&mut two), Ok(Some((_, _, p))) if p == b"two"),
    );

    // A masked frame from a server is a protocol violation, and this is the
    // check that first ran by accident -- the split tests were feeding client
    // frames and `take_frame` refused them, exactly as it should.
    let mut masked = frame(OP_TEXT, b"hi");
    check(
        "masked server frame rejected",
        matches!(take_frame(&mut masked), Err(Error::Protocol)),
    );

    check(
        "header lookup is case-insensitive",
        header("HTTP/1.1 101\r\nSec-WebSocket-Accept: abc\r\n", "sec-websocket-accept")
            .as_deref()
            == Some("abc"),
    );

    ok
}
