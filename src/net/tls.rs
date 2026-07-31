//! TLS 1.3 client: X25519, ChaCha20-Poly1305, SHA-256.
//!
//! # Read this before trusting it with anything
//!
//! **This does not authenticate the server.** The certificate is received,
//! parsed far enough to fingerprint, and then *not verified*: no signature
//! check, no chain building, no trust store, no name matching, no expiry.
//!
//! What that buys and what it does not:
//!
//!   * It stops a **passive** eavesdropper. Someone recording the wire sees
//!     ChaCha20-Poly1305 ciphertext and cannot read it.
//!   * It does **nothing** against an **active** attacker. Anyone able to
//!     answer in the server's place -- the network operator, anyone on the
//!     path, anyone who can redirect a route -- completes this handshake
//!     perfectly with their own key and reads everything.
//!
//! So it is real encryption with no identity behind it, which is the useful
//! half of TLS and the less important half. It is here because a byte stream
//! that survives the public internet is worth having, and because the
//! primitives underneath it are checked against their RFC vectors. It is not
//! here because it is safe. Do not put a password through it.
//!
//! Certificate validation is a larger job than everything above: X.509 DER
//! parsing, RSA-PSS and ECDSA P-256 verification (a bignum modexp and a second
//! curve), a trust store to ship and to update, and name matching with its own
//! long history of bugs. It is the honest next step and it is not a small one.
//!
//! ## What is implemented
//!
//! One cipher suite and one group, because TLS 1.3 permits exactly that and
//! every additional option is another path that is never exercised. The key
//! schedule follows RFC 8446 section 7.1 literally; the transcript is hashed
//! at the points the RFC names, because a transcript taken one message early
//! produces keys that are well-formed and simply wrong, and the failure
//! surfaces several messages later with nothing to point at.

use super::tcp;
use super::Ipv4;
use crate::crypto::{chacha, hkdf, x25519};
use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use crate::store::sha256;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

const REC_CHANGE_CIPHER_SPEC: u8 = 20;
const REC_ALERT: u8 = 21;
const REC_HANDSHAKE: u8 = 22;
const REC_APPLICATION: u8 = 23;

const HS_CLIENT_HELLO: u8 = 1;
const HS_SERVER_HELLO: u8 = 2;
const HS_NEW_SESSION_TICKET: u8 = 4;
const HS_ENCRYPTED_EXTENSIONS: u8 = 8;
const HS_CERTIFICATE: u8 = 11;
const HS_CERTIFICATE_VERIFY: u8 = 15;
const HS_FINISHED: u8 = 20;

/// TLS_CHACHA20_POLY1305_SHA256.
const CIPHER_SUITE: u16 = 0x1303;
const GROUP_X25519: u16 = 0x001D;

/// A record's payload may be 2^14 bytes, plus the AEAD tag and the inner
/// content type.
const MAX_RECORD: usize = 16384 + 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    Tcp,
    Timeout,
    Protocol,
    BadRecord,
    Decrypt,
    Alert(u8),
    HelloRetry,
    NoKeyShare,
    BadFinished,
    WrongVersion,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::Tcp => "could not connect",
            Error::Timeout => "the server stopped responding",
            Error::Protocol => "the server said something unexpected",
            Error::BadRecord => "malformed record",
            Error::Decrypt => "decryption failed -- keys do not match",
            Error::Alert(_) => "the server sent an alert",
            Error::HelloRetry => "the server asked to retry with another group",
            Error::NoKeyShare => "the server offered no usable key share",
            Error::BadFinished => "the server's Finished did not verify",
            Error::WrongVersion => "the server does not speak TLS 1.3",
        }
    }
}

/// One direction's keys. The sequence number is part of the key material's
/// contract: the nonce is the IV xored with it, so it must never repeat and
/// must reset to zero every time the key changes.
struct Keys {
    key: [u8; 32],
    iv: [u8; 12],
    seq: u64,
}

impl Keys {
    fn from_secret(secret: &[u8; 32]) -> Self {
        let k = hkdf::expand_label(secret, "key", &[], 32);
        let v = hkdf::expand_label(secret, "iv", &[], 12);
        let mut key = [0u8; 32];
        let mut iv = [0u8; 12];
        key.copy_from_slice(&k);
        iv.copy_from_slice(&v);
        Keys { key, iv, seq: 0 }
    }

    fn nonce(&self) -> [u8; 12] {
        let mut n = self.iv;
        let s = self.seq.to_be_bytes();
        // The sequence number is right-aligned in the 12-byte IV, so the
        // leading four bytes of the IV are never disturbed.
        for i in 0..8 {
            n[4 + i] ^= s[i];
        }
        n
    }
}

pub struct Session {
    client: Keys,
    server: Keys,
    /// Bytes received from TCP that do not yet make a whole record.
    inbuf: Vec<u8>,
    /// Decrypted application bytes not yet handed to the caller.
    plain: Vec<u8>,
    closed: bool,
    pub cert_fingerprint: Option<[u8; 32]>,
    pub cert_count: usize,
}

// --- wire helpers --------------------------------------------------------

fn u16v(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn push_vec16(out: &mut Vec<u8>, body: &[u8]) {
    out.extend_from_slice(&u16v(body.len() as u16));
    out.extend_from_slice(body);
}

fn extension(kind: u16, body: &[u8]) -> Vec<u8> {
    let mut e = Vec::with_capacity(4 + body.len());
    e.extend_from_slice(&u16v(kind));
    push_vec16(&mut e, body);
    e
}

fn build_client_hello(host: &str, random: &[u8; 32], session_id: &[u8; 32], pubkey: &[u8; 32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(512);
    // legacy_version is frozen at TLS 1.2 in TLS 1.3; the real version lives
    // in the supported_versions extension, because middleboxes reject
    // anything else in this field.
    b.extend_from_slice(&u16v(0x0303));
    b.extend_from_slice(random);
    b.push(32);
    b.extend_from_slice(session_id);
    b.extend_from_slice(&u16v(2));
    b.extend_from_slice(&u16v(CIPHER_SUITE));
    b.push(1);
    b.push(0); // no compression

    let mut ext = Vec::with_capacity(256);

    // server_name. The list-of-one shape is a historical artefact of a list
    // that never had a second member type.
    if !host.is_empty() && super::parse_ip(host).is_none() {
        let mut sni = Vec::with_capacity(host.len() + 5);
        sni.extend_from_slice(&u16v((host.len() + 3) as u16));
        sni.push(0); // host_name
        push_vec16(&mut sni, host.as_bytes());
        ext.extend_from_slice(&extension(0x0000, &sni));
    }

    // supported_groups
    let mut groups = Vec::new();
    groups.extend_from_slice(&u16v(2));
    groups.extend_from_slice(&u16v(GROUP_X25519));
    ext.extend_from_slice(&extension(0x000A, &groups));

    // signature_algorithms. Nothing here verifies a signature, but a server
    // will refuse a ClientHello without this extension, so the list has to be
    // present and plausible.
    let mut sigs = Vec::new();
    let algs: [u16; 5] = [0x0403, 0x0804, 0x0805, 0x0806, 0x0401];
    sigs.extend_from_slice(&u16v((algs.len() * 2) as u16));
    for a in algs {
        sigs.extend_from_slice(&u16v(a));
    }
    ext.extend_from_slice(&extension(0x000D, &sigs));

    // supported_versions: TLS 1.3 only.
    let mut vers = Vec::new();
    vers.push(2);
    vers.extend_from_slice(&u16v(0x0304));
    ext.extend_from_slice(&extension(0x002B, &vers));

    // key_share, sent up front so one round trip is enough.
    let mut ks = Vec::new();
    let mut entry = Vec::new();
    entry.extend_from_slice(&u16v(GROUP_X25519));
    push_vec16(&mut entry, pubkey);
    push_vec16(&mut ks, &entry);
    ext.extend_from_slice(&extension(0x0033, &ks));

    push_vec16(&mut b, &ext);

    // Wrap as a handshake message: type, 24-bit length.
    let mut m = Vec::with_capacity(b.len() + 4);
    m.push(HS_CLIENT_HELLO);
    m.push((b.len() >> 16) as u8);
    m.push((b.len() >> 8) as u8);
    m.push(b.len() as u8);
    m.extend_from_slice(&b);
    m
}

/// The magic random value that marks a HelloRetryRequest rather than a real
/// ServerHello. It is the SHA-256 of "HelloRetryRequest", frozen into the RFC.
const HELLO_RETRY: [u8; 32] = [
    0xCF, 0x21, 0xAD, 0x74, 0xE5, 0x9A, 0x61, 0x11, 0xBE, 0x1D, 0x8C, 0x02, 0x1E, 0x65, 0xB8, 0x91,
    0xC2, 0xA2, 0x11, 0x16, 0x7A, 0xBB, 0x8C, 0x5E, 0x07, 0x9E, 0x09, 0xE2, 0xC8, 0xA8, 0x33, 0x9C,
];

fn parse_server_hello(body: &[u8]) -> Result<[u8; 32], Error> {
    if body.len() < 38 {
        return Err(Error::BadRecord);
    }
    let random = &body[2..34];
    if random == HELLO_RETRY {
        // Only one group is offered, so a request to try another cannot be
        // satisfied. Saying so beats retrying forever.
        return Err(Error::HelloRetry);
    }
    let sid_len = body[34] as usize;
    let mut at = 35 + sid_len;
    if at + 3 > body.len() {
        return Err(Error::BadRecord);
    }
    let suite = u16::from_be_bytes([body[at], body[at + 1]]);
    if suite != CIPHER_SUITE {
        return Err(Error::Protocol);
    }
    at += 3; // suite and the legacy compression byte

    if at + 2 > body.len() {
        return Err(Error::BadRecord);
    }
    let ext_len = u16::from_be_bytes([body[at], body[at + 1]]) as usize;
    at += 2;
    let end = at + ext_len;
    if end > body.len() {
        return Err(Error::BadRecord);
    }

    let mut share: Option<[u8; 32]> = None;
    let mut saw_1_3 = false;
    while at + 4 <= end {
        let kind = u16::from_be_bytes([body[at], body[at + 1]]);
        let len = u16::from_be_bytes([body[at + 2], body[at + 3]]) as usize;
        at += 4;
        if at + len > end {
            return Err(Error::BadRecord);
        }
        let val = &body[at..at + len];
        match kind {
            0x002B => {
                if val.len() == 2 && u16::from_be_bytes([val[0], val[1]]) == 0x0304 {
                    saw_1_3 = true;
                }
            }
            0x0033 => {
                if val.len() >= 4 {
                    let group = u16::from_be_bytes([val[0], val[1]]);
                    let klen = u16::from_be_bytes([val[2], val[3]]) as usize;
                    if group == GROUP_X25519 && klen == 32 && val.len() >= 4 + 32 {
                        let mut k = [0u8; 32];
                        k.copy_from_slice(&val[4..36]);
                        share = Some(k);
                    }
                }
            }
            _ => {}
        }
        at += len;
    }

    if !saw_1_3 {
        return Err(Error::WrongVersion);
    }
    share.ok_or(Error::NoKeyShare)
}

// --- record plumbing -----------------------------------------------------

impl Session {
    /// Pull bytes from TCP until a whole record is buffered, then return it as
    /// (content_type, payload).
    fn read_record(&mut self, timeout_ms: u64) -> Result<(u8, Vec<u8>), Error> {
        let deadline =
            crate::dev::lapic::ticks() + (timeout_ms * crate::TIMER_HZ as u64) / 1000 + 1;
        loop {
            if self.inbuf.len() >= 5 {
                let len = u16::from_be_bytes([self.inbuf[3], self.inbuf[4]]) as usize;
                if len > MAX_RECORD {
                    return Err(Error::BadRecord);
                }
                if self.inbuf.len() >= 5 + len {
                    let kind = self.inbuf[0];
                    let body: Vec<u8> = self.inbuf[5..5 + len].to_vec();
                    self.inbuf.drain(..5 + len);
                    return Ok((kind, body));
                }
            }
            if crate::dev::lapic::ticks() >= deadline {
                return Err(Error::Timeout);
            }
            let chunk = tcp::recv(300);
            if chunk.is_empty() {
                if !matches!(tcp::state(), tcp::State::Established) {
                    self.closed = true;
                    return Err(Error::Timeout);
                }
                continue;
            }
            self.inbuf.extend_from_slice(&chunk);
        }
    }

    /// Read a record and decrypt it, returning the inner content type.
    ///
    /// A ChangeCipherSpec arriving mid-handshake is discarded rather than
    /// decrypted: TLS 1.3 does not use it, but clients and servers still emit
    /// one so that middleboxes built for TLS 1.2 see what they expect.
    fn read_encrypted(&mut self, timeout_ms: u64) -> Result<(u8, Vec<u8>), Error> {
        loop {
            let (kind, body) = self.read_record(timeout_ms)?;
            if kind == REC_CHANGE_CIPHER_SPEC {
                continue;
            }
            if kind == REC_ALERT && self.server.seq == 0 {
                // An alert before any key is in use is sent in the clear.
                return Err(Error::Alert(*body.get(1).unwrap_or(&0)));
            }
            if kind != REC_APPLICATION {
                return Err(Error::Protocol);
            }

            // The additional data is the record header exactly as it went on
            // the wire, which is why it is rebuilt rather than remembered.
            let mut aad = Vec::with_capacity(5);
            aad.push(REC_APPLICATION);
            aad.extend_from_slice(&u16v(0x0303));
            aad.extend_from_slice(&u16v(body.len() as u16));

            let nonce = self.server.nonce();
            let opened = chacha::open(&self.server.key, &nonce, &aad, &body)
                .ok_or(Error::Decrypt)?;
            self.server.seq += 1;

            // The real content type is the last non-zero byte; everything
            // after it is padding.
            let mut end = opened.len();
            while end > 0 && opened[end - 1] == 0 {
                end -= 1;
            }
            if end == 0 {
                return Err(Error::BadRecord);
            }
            let inner_type = opened[end - 1];
            return Ok((inner_type, opened[..end - 1].to_vec()));
        }
    }

    fn write_encrypted(&mut self, inner_type: u8, payload: &[u8]) -> Result<(), Error> {
        let mut inner = Vec::with_capacity(payload.len() + 1);
        inner.extend_from_slice(payload);
        inner.push(inner_type);

        let total = inner.len() + chacha::TAG_LEN;
        let mut aad = Vec::with_capacity(5);
        aad.push(REC_APPLICATION);
        aad.extend_from_slice(&u16v(0x0303));
        aad.extend_from_slice(&u16v(total as u16));

        let nonce = self.client.nonce();
        let sealed = chacha::seal(&self.client.key, &nonce, &aad, &inner);
        self.client.seq += 1;

        let mut rec = Vec::with_capacity(5 + sealed.len());
        rec.extend_from_slice(&aad);
        rec.extend_from_slice(&sealed);
        tcp::send(&rec, 5000).map_err(|_| Error::Tcp)
    }
}

/// Split a buffer of concatenated handshake messages into (type, body).
fn handshake_messages(buf: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 4 <= buf.len() {
        let kind = buf[at];
        let len = ((buf[at + 1] as usize) << 16) | ((buf[at + 2] as usize) << 8) | buf[at + 3] as usize;
        if at + 4 + len > buf.len() {
            break;
        }
        out.push((kind, buf[at + 4..at + 4 + len].to_vec()));
        at += 4 + len;
    }
    out
}

/// The leaf certificate, if the Certificate message carries one.
///
/// Parsed only far enough to find and fingerprint it. Nothing here inspects
/// the contents, and nothing here verifies anything -- see the module header.
fn leaf_certificate(body: &[u8]) -> (Option<[u8; 32]>, usize) {
    if body.is_empty() {
        return (None, 0);
    }
    let ctx_len = body[0] as usize;
    let mut at = 1 + ctx_len;
    if at + 3 > body.len() {
        return (None, 0);
    }
    let list_len = ((body[at] as usize) << 16) | ((body[at + 1] as usize) << 8) | body[at + 2] as usize;
    at += 3;
    let end = core::cmp::min(at + list_len, body.len());

    let mut first: Option<[u8; 32]> = None;
    let mut count = 0;
    while at + 3 <= end {
        let clen = ((body[at] as usize) << 16) | ((body[at + 1] as usize) << 8) | body[at + 2] as usize;
        at += 3;
        if at + clen > end {
            break;
        }
        if first.is_none() {
            first = Some(sha256::hash(&body[at..at + clen]));
        }
        count += 1;
        at += clen;
        if at + 2 > end {
            break;
        }
        // Each entry carries its own extensions, skipped here.
        let ext = u16::from_be_bytes([body[at], body[at + 1]]) as usize;
        at += 2 + ext;
    }
    (first, count)
}

/// Open a TLS 1.3 connection on top of a fresh TCP connection.
pub fn connect(dst: Ipv4, host: &str, port: u16) -> Result<Session, Error> {
    tcp::connect(dst, port, 5000).map_err(|_| Error::Tcp)?;

    // The private key and the two random values all come from the TSC. That is
    // the only entropy this machine has, and it is a genuine weakness worth
    // naming: a counter started at power-on is not a random number generator,
    // and an attacker who can guess the boot time narrows the key. Real
    // hardware entropy (RDRAND, which this CPU has) is the fix.
    let mut secret = [0u8; 32];
    let mut random = [0u8; 32];
    let mut session_id = [0u8; 32];
    for i in 0..4 {
        let t = crate::time::rdtsc();
        secret[i * 8..i * 8 + 8].copy_from_slice(&t.to_le_bytes());
        let t = crate::time::rdtsc().rotate_left(17) ^ 0x9E3779B97F4A7C15;
        random[i * 8..i * 8 + 8].copy_from_slice(&t.to_le_bytes());
        let t = crate::time::rdtsc().rotate_left(41) ^ 0xBF58476D1CE4E5B9;
        session_id[i * 8..i * 8 + 8].copy_from_slice(&t.to_le_bytes());
    }
    let pubkey = x25519::public_key(&secret);

    let ch = build_client_hello(host, &random, &session_id, &pubkey);
    let mut transcript: Vec<u8> = Vec::with_capacity(4096);
    transcript.extend_from_slice(&ch);

    let mut rec = Vec::with_capacity(5 + ch.len());
    rec.push(REC_HANDSHAKE);
    rec.extend_from_slice(&u16v(0x0301)); // the first record claims TLS 1.0
    rec.extend_from_slice(&u16v(ch.len() as u16));
    rec.extend_from_slice(&ch);
    tcp::send(&rec, 5000).map_err(|_| Error::Tcp)?;

    // A placeholder session so the record reader can be used before keys
    // exist. Nothing encrypted is read until the real keys replace these.
    let mut s = Session {
        client: Keys { key: [0; 32], iv: [0; 12], seq: 0 },
        server: Keys { key: [0; 32], iv: [0; 12], seq: 0 },
        inbuf: Vec::new(),
        plain: Vec::new(),
        closed: false,
        cert_fingerprint: None,
        cert_count: 0,
    };

    // --- ServerHello, in the clear ---
    let sh_body = loop {
        let (kind, body) = s.read_record(8000)?;
        match kind {
            REC_CHANGE_CIPHER_SPEC => continue,
            REC_ALERT => return Err(Error::Alert(*body.get(1).unwrap_or(&0))),
            REC_HANDSHAKE => break body,
            _ => return Err(Error::Protocol),
        }
    };
    let msgs = handshake_messages(&sh_body);
    let Some((HS_SERVER_HELLO, sh)) = msgs.first().map(|(k, v)| (*k, v.clone())) else {
        return Err(Error::Protocol);
    };
    let server_pub = parse_server_hello(&sh)?;
    transcript.extend_from_slice(&sh_body);

    // --- the key schedule, RFC 8446 section 7.1 ---
    let zeros = [0u8; 32];
    let early = hkdf::extract(&[], &zeros);
    let derived = hkdf::derive_secret(&early, "derived", &sha256::hash(&[]));
    let ecdh = x25519::shared_secret(&secret, &server_pub).ok_or(Error::Protocol)?;
    let handshake_secret = hkdf::extract(&derived, &ecdh);

    let th = sha256::hash(&transcript);
    let c_hs = hkdf::derive_secret(&handshake_secret, "c hs traffic", &th);
    let s_hs = hkdf::derive_secret(&handshake_secret, "s hs traffic", &th);
    s.client = Keys::from_secret(&c_hs);
    s.server = Keys::from_secret(&s_hs);

    // --- the encrypted half of the handshake ---
    let mut pending: Vec<u8> = Vec::new();
    let mut server_finished: Option<Vec<u8>> = None;
    let mut transcript_before_finished: Vec<u8> = Vec::new();

    while server_finished.is_none() {
        let (inner, body) = s.read_encrypted(8000)?;
        match inner {
            REC_ALERT => return Err(Error::Alert(*body.get(1).unwrap_or(&0))),
            REC_HANDSHAKE => pending.extend_from_slice(&body),
            _ => return Err(Error::Protocol),
        }
        for (kind, msg) in handshake_messages(&pending) {
            let mut framed = Vec::with_capacity(4 + msg.len());
            framed.push(kind);
            framed.push((msg.len() >> 16) as u8);
            framed.push((msg.len() >> 8) as u8);
            framed.push(msg.len() as u8);
            framed.extend_from_slice(&msg);

            match kind {
                HS_ENCRYPTED_EXTENSIONS | HS_CERTIFICATE_VERIFY => {}
                HS_CERTIFICATE => {
                    let (fp, n) = leaf_certificate(&msg);
                    s.cert_fingerprint = fp;
                    s.cert_count = n;
                }
                HS_FINISHED => {
                    // The transcript for verifying Finished stops *before*
                    // Finished itself. Getting this boundary wrong is the
                    // single easiest way to produce a handshake that fails
                    // with nothing to point at.
                    transcript_before_finished = transcript.clone();
                    server_finished = Some(msg.clone());
                }
                _ => {}
            }
            transcript.extend_from_slice(&framed);
            if server_finished.is_some() {
                break;
            }
        }
        pending.clear();
    }

    // --- verify the server's Finished ---
    let sf = server_finished.ok_or(Error::Protocol)?;
    let finished_key = hkdf::expand_label(&s_hs, "finished", &[], 32);
    let expect = hkdf::hmac(&finished_key, &sha256::hash(&transcript_before_finished));
    if sf.len() != 32 {
        return Err(Error::BadFinished);
    }
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= expect[i] ^ sf[i];
    }
    if diff != 0 {
        return Err(Error::BadFinished);
    }

    // --- our Finished, under the handshake keys ---
    let th_after = sha256::hash(&transcript);
    let c_fin_key = hkdf::expand_label(&c_hs, "finished", &[], 32);
    let verify = hkdf::hmac(&c_fin_key, &th_after);
    let mut fin = Vec::with_capacity(36);
    fin.push(HS_FINISHED);
    fin.push(0);
    fin.push(0);
    fin.push(32);
    fin.extend_from_slice(&verify);

    // The bare ChangeCipherSpec that middleboxes expect to see.
    let ccs = [REC_CHANGE_CIPHER_SPEC, 0x03, 0x03, 0x00, 0x01, 0x01];
    tcp::send(&ccs, 5000).map_err(|_| Error::Tcp)?;
    s.write_encrypted(REC_HANDSHAKE, &fin)?;

    // --- application keys ---
    let derived2 = hkdf::derive_secret(&handshake_secret, "derived", &sha256::hash(&[]));
    let master = hkdf::extract(&derived2, &zeros);
    let c_ap = hkdf::derive_secret(&master, "c ap traffic", &th_after);
    let s_ap = hkdf::derive_secret(&master, "s ap traffic", &th_after);
    // Both sequence numbers restart at zero with the new keys.
    s.client = Keys::from_secret(&c_ap);
    s.server = Keys::from_secret(&s_ap);

    Ok(s)
}

impl Session {
    pub fn send(&mut self, data: &[u8]) -> Result<(), Error> {
        for chunk in data.chunks(16384) {
            self.write_encrypted(REC_APPLICATION, chunk)?;
        }
        Ok(())
    }

    /// Read application bytes, skipping anything else the server sends.
    ///
    /// A server routinely sends NewSessionTicket messages after the handshake;
    /// they arrive as handshake content inside application records and are of
    /// no use to a client that never resumes.
    pub fn recv(&mut self, timeout_ms: u64) -> Result<Vec<u8>, Error> {
        if !self.plain.is_empty() {
            return Ok(core::mem::take(&mut self.plain));
        }
        loop {
            match self.read_encrypted(timeout_ms) {
                Err(Error::Timeout) => return Ok(Vec::new()),
                Err(e) => return Err(e),
                Ok((inner, body)) => match inner {
                    REC_APPLICATION => return Ok(body),
                    REC_HANDSHAKE => {
                        // Tickets and key-update requests. Ignored, but the
                        // records still advance the sequence number, which
                        // read_encrypted has already done.
                        let _ = HS_NEW_SESSION_TICKET;
                        continue;
                    }
                    REC_ALERT => {
                        // close_notify is the ordinary end of a stream.
                        if body.get(1) == Some(&0) {
                            self.closed = true;
                            return Ok(Vec::new());
                        }
                        return Err(Error::Alert(*body.get(1).unwrap_or(&0)));
                    }
                    _ => continue,
                },
            }
        }
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn close(&mut self) {
        // close_notify, then tear down the TCP underneath.
        let _ = self.write_encrypted(REC_ALERT, &[1, 0]);
        tcp::close(2000);
    }
}

/// Fetch one resource over HTTPS.
pub fn https_get(dst: Ipv4, host: &str, port: u16, path: &str) -> Result<(Vec<u8>, Option<[u8; 32]>, usize), Error> {
    let mut s = connect(dst, host, port)?;

    let mut req = String::new();
    req.push_str("GET ");
    req.push_str(if path.is_empty() { "/" } else { path });
    req.push_str(" HTTP/1.1\r\nHost: ");
    req.push_str(host);
    req.push_str("\r\nUser-Agent: glados/0.1\r\nConnection: close\r\nAccept: */*\r\n\r\n");
    s.send(req.as_bytes())?;

    let mut body = Vec::new();
    let deadline = crate::dev::lapic::ticks() + 15 * crate::TIMER_HZ as u64;
    while crate::dev::lapic::ticks() < deadline {
        let chunk = s.recv(2000)?;
        if chunk.is_empty() {
            if s.closed() || !matches!(tcp::state(), tcp::State::Established) {
                break;
            }
            continue;
        }
        body.extend_from_slice(&chunk);
    }
    let fp = s.cert_fingerprint;
    let n = s.cert_count;
    s.close();
    Ok((body, fp, n))
}

/// Undo chunked transfer encoding.
///
/// Needed because this asks for HTTP/1.1 -- a name-based virtual host will not
/// answer HTTP/1.0 reliably -- and 1.1 servers chunk even when told to close.
/// Without this the body arrives interleaved with its own framing: a hex
/// length, the data, then a zero-length chunk, all of which look like content.
fn dechunk(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len());
    let mut at = 0;
    loop {
        // A chunk header is a hex length, optionally followed by extensions
        // after a semicolon, terminated by CRLF.
        let Some(eol) = body[at..].windows(2).position(|w| w == b"\r\n") else {
            break;
        };
        let line = &body[at..at + eol];
        let hex = line.split(|b| *b == b';').next().unwrap_or(line);
        let mut len = 0usize;
        let mut any = false;
        for c in hex {
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => break,
            };
            len = len * 16 + d as usize;
            any = true;
        }
        if !any {
            break;
        }
        at += eol + 2;
        if len == 0 {
            break;
        }
        if at + len > body.len() {
            // Truncated: keep what arrived rather than discarding it.
            out.extend_from_slice(&body[at..]);
            break;
        }
        out.extend_from_slice(&body[at..at + len]);
        at += len + 2; // the CRLF that follows each chunk
    }
    out
}

pub fn report(dst: Ipv4, host: &str, port: u16, path: &str) {
    console::set_color(YELLOW);
    kprintln!("[https] {}:{}{}", host, port, if path.is_empty() { "/" } else { path });
    console::set_color(LTGRAY);
    kprintln!("  {}.{}.{}.{}", dst[0], dst[1], dst[2], dst[3]);

    let t0 = crate::time::rdtsc();
    match https_get(dst, host, port, path) {
        Err(e) => {
            console::set_color(LTRED);
            kprintln!("  {}", e.name());
            console::set_color(LTGRAY);
        }
        Ok((body, fp, ncerts)) => {
            let mhz = crate::time::tsc_mhz().max(1);
            let ms = (crate::time::rdtsc() - t0) / mhz / 1000;
            console::set_color(LTGREEN);
            kprintln!("  handshake ok -- TLS 1.3, x25519, chacha20-poly1305");
            console::set_color(LTGRAY);
            if let Some(f) = fp {
                let h = sha256::short_hex(&f);
                let s = core::str::from_utf8(&h).unwrap_or("?");
                kprintln!("  {} certificate(s); leaf sha256 {}..", ncerts, s);
            }
            console::set_color(YELLOW);
            kprintln!("  NOT VERIFIED -- encrypted, but the peer proved nothing");
            console::set_color(LTGRAY);

            let text = String::from_utf8_lossy(&body);
            if let Some(status) = text.lines().next() {
                kprintln!("  {}", status.trim());
            }
            match text.find("\r\n\r\n").map(|i| i + 4) {
                Some(i) => {
                    let chunked = text[..i].to_ascii_lowercase().contains("transfer-encoding: chunked");
                    let raw = &body[i..];
                    let decoded = if chunked { dechunk(raw) } else { raw.to_vec() };
                    let b = String::from_utf8_lossy(&decoded);
                    kprintln!(
                        "  {} B in {} ms, {} B of body{}",
                        body.len(), ms, decoded.len(),
                        if chunked { " (dechunked)" } else { "" }
                    );
                    for line in b.lines().take(8) {
                        kprintln!("  | {}", line);
                    }
                }
                None => kprintln!("  {} B in {} ms", body.len(), ms),
            }
        }
    }
    let _ = vec![0u8; 0];
}
