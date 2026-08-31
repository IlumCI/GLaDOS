//! TLS 1.3 client: X25519, ChaCha20-Poly1305, SHA-256.
//!
//! ## Authentication
//!
//! The server is now identified as well as encrypted to. Four things are
//! checked, in `check_identity` and `x509::validate`, and any one failing
//! leaves `Session::identity` as `Failed`:
//!
//!   1. **CertificateVerify** -- the server signs the handshake transcript
//!      with the key in its certificate. This is the step that proves the
//!      party at the other end of *this* connection holds the private key;
//!      without it an attacker could replay any certificate ever seen.
//!   2. **The chain** -- each certificate is verified against the next, up to
//!      one whose fingerprint is in the trust store.
//!   3. **Dates**, against the CMOS clock.
//!   4. **The name**, RFC 6125, from subjectAltName only.
//!
//! The result is *reported*, not enforced: `https` prints what was
//! established and shows the body either way. That is a deliberate choice for
//! a system whose purpose is inspection, and it is the opposite of what a
//! browser should do. A caller that cares must check `identity.ok()`.
//!
//! Two things still make this unsuitable for anything that matters. There is
//! **no revocation** -- no CRL, no OCSP -- so a withdrawn certificate is
//! accepted until it expires. And key material comes from the **TSC**, which
//! is a counter and not a random number generator.
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
use super::x509;
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

/// What, if anything, was established about the peer.
#[derive(Clone, PartialEq, Eq)]
pub enum Identity {
    /// Chain verified to a trusted root, signature over the transcript
    /// checked, dates in range, name matches.
    Verified { subject: String, roots: usize },
    /// The handshake completed but the peer's identity did not check out.
    Failed(x509::Error),
    /// No roots are loaded, so nothing could be checked.
    NoTrustStore,
}

impl Identity {
    pub fn ok(&self) -> bool {
        matches!(self, Identity::Verified { .. })
    }
}

/// Run every identity check, in the order that gives the most useful failure.
fn check_identity(
    chain: &[x509::Cert],
    cert_verify: Option<&[u8]>,
    transcript: &[u8],
    host: &str,
) -> Identity {
    if chain.is_empty() {
        return Identity::Failed(x509::Error::Malformed);
    }
    if super::trust::count() == 0 {
        return Identity::NoTrustStore;
    }
    // Proof of possession first: it is the cheapest check and the one whose
    // failure means the peer is not who the certificate says regardless of how
    // good the certificate is.
    let Some(cv) = cert_verify else {
        return Identity::Failed(x509::Error::BadSignature);
    };
    if let Err(e) = verify_certificate_verify(&chain[0], cv, transcript) {
        return Identity::Failed(e);
    }
    // The clock: zero means the RTC never answered, and a date comparison
    // against a wrong clock is worse than none.
    let now = crate::dev::rtc::now()
        .map(|dt| crate::dev::rtc::unix_seconds(&dt) as u64)
        .unwrap_or(0);
    match x509::validate(chain, host, now) {
        Ok(()) => Identity::Verified {
            subject: String::from_utf8_lossy(&chain[0].subject_cn).into_owned(),
            roots: super::trust::count(),
        },
        Err(e) => Identity::Failed(e),
    }
}

pub struct Session {
    client: Keys,
    server: Keys,
    pub identity: Identity,
    /// Bytes received from TCP that do not yet make a whole record.
    inbuf: Vec<u8>,
    /// Decrypted application bytes not yet handed to the caller.
    plain: Vec<u8>,
    closed: bool,
    pub cert_fingerprint: Option<[u8; 32]>,
    pub cert_count: usize,
    /// What the leaf certificate says it is for. Kept so a name mismatch can
    /// report the names it did find instead of only that it found none.
    pub leaf_cn: String,
    pub leaf_names: Vec<String>,
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

/// Split the Certificate message into its list of DER certificates.
fn certificate_list(body: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if body.is_empty() {
        return out;
    }
    let ctx_len = body[0] as usize;
    let mut at = 1 + ctx_len;
    if at + 3 > body.len() {
        return out;
    }
    let list_len =
        ((body[at] as usize) << 16) | ((body[at + 1] as usize) << 8) | body[at + 2] as usize;
    at += 3;
    let end = core::cmp::min(at + list_len, body.len());
    while at + 3 <= end {
        let clen =
            ((body[at] as usize) << 16) | ((body[at + 1] as usize) << 8) | body[at + 2] as usize;
        at += 3;
        if at + clen > end {
            break;
        }
        out.push(body[at..at + clen].to_vec());
        at += clen;
        if at + 2 > end {
            break;
        }
        // Each entry carries its own extensions, skipped here.
        let ext = u16::from_be_bytes([body[at], body[at + 1]]) as usize;
        at += 2 + ext;
    }
    out
}

fn parse_chain(body: &[u8]) -> Vec<x509::Cert> {
    certificate_list(body)
        .iter()
        .filter_map(|der| x509::parse(der).ok())
        .collect()
}

/// The signature scheme codes that can appear in CertificateVerify.
const SIG_RSA_PKCS1_SHA256: u16 = 0x0401;
const SIG_ECDSA_SECP256R1_SHA256: u16 = 0x0403;
const SIG_RSA_PSS_RSAE_SHA256: u16 = 0x0804;
const SIG_RSA_PSS_PSS_SHA256: u16 = 0x0809;

/// Verify CertificateVerify: the server's signature over the transcript.
///
/// This is the step that actually proves possession. Everything else in the
/// chain establishes that a certificate is genuine; only this establishes that
/// the party on the other end of *this* connection holds its private key.
/// Without it, an attacker could replay any certificate they had ever seen.
///
/// What is signed is not the transcript itself but a fixed context string --
/// 64 spaces, a label, a zero byte -- followed by the transcript hash. The
/// padding exists so that a signature made for one purpose cannot be presented
/// as one made for another.
fn verify_certificate_verify(
    cert: &x509::Cert,
    msg: &[u8],
    transcript: &[u8],
) -> Result<(), x509::Error> {
    if msg.len() < 4 {
        return Err(x509::Error::Malformed);
    }
    let scheme = u16::from_be_bytes([msg[0], msg[1]]);
    let siglen = u16::from_be_bytes([msg[2], msg[3]]) as usize;
    if msg.len() < 4 + siglen {
        return Err(x509::Error::Malformed);
    }
    let sig = &msg[4..4 + siglen];

    let mut signed = Vec::with_capacity(130);
    signed.extend_from_slice(&[0x20u8; 64]);
    signed.extend_from_slice(b"TLS 1.3, server CertificateVerify");
    signed.push(0);
    signed.extend_from_slice(&sha256::hash(transcript));
    let digest = sha256::hash(&signed);

    let ok = match (scheme, cert.key_kind) {
        (SIG_ECDSA_SECP256R1_SHA256, x509::KeyKind::EcP256) => {
            match crate::crypto::p256::parse_der_signature(sig, 32) {
                None => false,
                Some((r, s)) => crate::crypto::p256::verify(
                    crate::crypto::p256::Nist::P256, &cert.key, &digest, &r, &s),
            }
        }
        (SIG_RSA_PKCS1_SHA256, x509::KeyKind::Rsa) => {
            crate::crypto::rsa::verify_pkcs1_sha256(&cert.key, &cert.key_exp, &digest, sig)
        }
        (SIG_RSA_PSS_RSAE_SHA256, x509::KeyKind::Rsa)
        | (SIG_RSA_PSS_PSS_SHA256, x509::KeyKind::Rsa) => {
            crate::crypto::rsa::verify_pss_sha256(&cert.key, &cert.key_exp, &digest, sig)
        }
        _ => return Err(x509::Error::UnsupportedSignature),
    };
    if ok {
        Ok(())
    } else {
        Err(x509::Error::BadSignature)
    }
}

/// The leaf certificate, if the Certificate message carries one.
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

    // The private key and the two random values come from the kernel
    // generator, which is a fast-key-erasure ChaCha20 DRBG over interrupt
    // timing (`src/rng`). This used to be four `rdtsc()` reads, and the
    // comment here said what that meant: a counter started at power-on is not
    // a random number generator, and an attacker who can guess the boot time
    // narrows the key.
    //
    // `fill_secret` refuses below its threshold instead of degrading, and the
    // refusal is reported rather than swallowed. A machine that has booted,
    // run a script and never had a key pressed has no entropy, and a
    // handshake made from a timestamp should say so out loud while it happens
    // and not be discovered later in a log. The connection still proceeds,
    // because refusing to talk is not this layer's decision to make, and the
    // operator can see exactly what was used.
    let mut secret = [0u8; 32];
    let mut random = [0u8; 32];
    let mut session_id = [0u8; 32];
    let mut weak = None;
    for buf in [&mut secret, &mut random, &mut session_id] {
        if let Err(bits) = crate::rng::fill_secret(buf) {
            weak = Some(bits);
            // Fall back to the old construction so the handshake completes,
            // and mix the generator in anyway: it is no worse than the TSC
            // alone and it is what this path used to be.
            crate::rng::fill(buf);
            for (i, b) in buf.chunks_mut(8).enumerate() {
                let t = crate::time::rdtsc().rotate_left((17 * (i + 1)) as u32);
                for (x, y) in b.iter_mut().zip(t.to_le_bytes().iter()) {
                    *x ^= *y;
                }
            }
        }
    }
    if let Some(bits) = weak {
        crate::kprintln!(
            "  [tls] WARNING: {} of {} entropy bits -- keys for this",
            bits,
            crate::rng::SEEDED_BITS
        );
        crate::kprintln!("  handshake are timing-derived, not random.");
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
        identity: Identity::Failed(x509::Error::Malformed),
        inbuf: Vec::new(),
        plain: Vec::new(),
        closed: false,
        cert_fingerprint: None,
        cert_count: 0,
        leaf_cn: String::new(),
        leaf_names: Vec::new(),
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
    let mut chain: Vec<x509::Cert> = Vec::new();
    let mut cert_verify: Option<Vec<u8>> = None;
    let mut transcript_for_cv: Vec<u8> = Vec::new();

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
                HS_ENCRYPTED_EXTENSIONS => {}
                HS_CERTIFICATE => {
                    let (fp, n) = leaf_certificate(&msg);
                    s.cert_fingerprint = fp;
                    s.cert_count = n;
                    chain = parse_chain(&msg);
                    // The transcript up to and including Certificate is what
                    // CertificateVerify signs, so it is captured before the
                    // Certificate message is appended below... which means
                    // capturing it after.
                    transcript_for_cv = transcript.clone();
                    transcript_for_cv.extend_from_slice(&framed);
                }
                HS_CERTIFICATE_VERIFY => {
                    cert_verify = Some(msg.clone());
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

    // --- who is this? ---
    //
    // Done after Finished so that a failure is reported against a handshake
    // that was otherwise sound, and before any application data is sent: the
    // connection is cryptographically complete and the identity behind it is
    // still unestablished until this passes.
    if let Some(leaf) = chain.first() {
        s.leaf_cn = String::from_utf8_lossy(&leaf.subject_cn).into_owned();
        s.leaf_names = leaf
            .dns_names
            .iter()
            .map(|n| String::from_utf8_lossy(n).into_owned())
            .chain(leaf.ip_names.iter().map(|n| match n.len() {
                4 => alloc::format!("IP:{}.{}.{}.{}", n[0], n[1], n[2], n[3]),
                // IPv6 entries are listed but never match, since nothing here
                // speaks IPv6 -- saying so beats omitting them.
                16 => alloc::string::String::from("IP:<ipv6>"),
                _ => alloc::string::String::from("IP:<malformed>"),
            }))
            .collect();
    }
    s.identity = check_identity(&chain, cert_verify.as_deref(), &transcript_for_cv, host);

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

/// What one HTTPS fetch produced, and whether it is all of it.
///
/// The six-tuple this replaced grew out of a browser that only ever wanted the
/// bytes. An updater wants two things that tuple could not say: how long the
/// response claimed to be, and whether that many arrived. A short read
/// reported as a bad signature is the failure this exists to prevent.
pub struct Fetched {
    pub status: u16,
    /// The response head, lower-cased, terminator included.
    pub headers: String,
    /// The body, de-chunked if it arrived chunked. Never the framing.
    pub body: Vec<u8>,
    pub identity: Identity,
    pub cert_fingerprint: Option<[u8; 32]>,
    pub cert_count: usize,
    pub leaf_cn: String,
    pub leaf_names: Vec<String>,
    /// The body is as long as `Content-Length` said, or a chunked body ended
    /// with its terminator, or the peer closed a response that declared no
    /// length at all. False means the deadline ran out first.
    pub complete: bool,
    /// What `Content-Length` claimed, when it claimed anything.
    pub declared: Option<usize>,
}

/// Read `Content-Length` out of an already-lower-cased response head.
fn content_length(head: &str) -> Option<usize> {
    const KEY: &str = "content-length:";
    let rest = &head[head.find(KEY)? + KEY.len()..];
    let end = rest.find("\r\n").unwrap_or(rest.len());
    rest[..end].trim().parse::<usize>().ok()
}

/// Fetch one resource over HTTPS, stopping as soon as the response is whole.
///
/// `timeout_ms` bounds the whole transfer rather than one read. The fixed
/// fifteen seconds this replaced was ample for a page and a coin toss for a
/// three-megabyte image over a thirty-two kilobyte window -- and losing the
/// toss returned a truncated body with no error, leaving the caller to
/// misdiagnose it as something else.
///
/// Knowing the declared length also means not waiting for the close that
/// `Connection: close` promises: the body is done when it is done.
pub fn https_fetch(
    dst: Ipv4,
    host: &str,
    port: u16,
    path: &str,
    timeout_ms: u64,
) -> Result<Fetched, Error> {
    https_fetch_with(dst, host, port, path, timeout_ms, &[])
}

/// The same, with extra request headers.
///
/// Split out rather than added to every caller because exactly one thing
/// needs it: the gated update channel, which sends a bearer token. A token
/// in a query string is logged by every proxy it passes, and a bearer token
/// in a log is a bearer token somebody else has.
pub fn https_fetch_with(
    dst: Ipv4,
    host: &str,
    port: u16,
    path: &str,
    timeout_ms: u64,
    extra: &[(&str, &str)],
) -> Result<Fetched, Error> {
    let mut s = connect(dst, host, port)?;

    let mut req = String::new();
    req.push_str("GET ");
    req.push_str(if path.is_empty() { "/" } else { path });
    req.push_str(" HTTP/1.1\r\nHost: ");
    req.push_str(host);
    req.push_str("\r\nUser-Agent: glados/0.1\r\nConnection: close\r\nAccept: */*\r\n");
    for (name, value) in extra {
        req.push_str(name);
        req.push_str(": ");
        req.push_str(value);
        req.push_str("\r\n");
    }
    req.push_str("\r\n");
    s.send(req.as_bytes())?;

    let mut raw: Vec<u8> = Vec::new();
    let mut head_end: Option<usize> = None;
    let mut declared: Option<usize> = None;
    let mut chunked = false;
    let mut complete = false;

    let span = timeout_ms.saturating_mul(crate::TIMER_HZ as u64) / 1000;
    let deadline = crate::dev::lapic::ticks() + span.max(1);

    while crate::dev::lapic::ticks() < deadline {
        let part = s.recv(2000)?;
        if part.is_empty() {
            if s.closed() || !matches!(tcp::state(), tcp::State::Established) {
                // A response that declares no length at all ends when the peer
                // closes, so this is its terminator rather than a failure.
                complete = head_end.is_some() && declared.is_none() && !chunked;
                break;
            }
            continue;
        }
        raw.extend_from_slice(&part);

        if head_end.is_none() {
            if let Some(at) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                head_end = Some(at + 4);
                let head = String::from_utf8_lossy(&raw[..at + 4]).to_ascii_lowercase();
                // A chunked response has no length to trust: the framing wins,
                // and a server sending both is describing two different bodies.
                chunked = head.contains("transfer-encoding: chunked");
                declared = if chunked { None } else { content_length(&head) };
            }
        }

        if let Some(h) = head_end {
            match declared {
                Some(n) if raw.len() - h >= n => {
                    complete = true;
                    break;
                }
                None if chunked && raw.ends_with(b"0\r\n\r\n") => {
                    complete = true;
                    break;
                }
                _ => {}
            }
        }
    }

    // Split in place. The body is the larger part by orders of magnitude, and
    // copying it out to hand it over would mean holding two of it at once.
    let (status, headers, body) = match head_end {
        None => (0, String::new(), raw),
        Some(h) => {
            let headers = String::from_utf8_lossy(&raw[..h]).to_ascii_lowercase();
            let status = headers
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(0);
            let mut body = raw;
            body.drain(..h);
            let body = if chunked { dechunk(&body) } else { body };
            (status, headers, body)
        }
    };

    let out = Fetched {
        status,
        headers,
        body,
        identity: s.identity.clone(),
        cert_fingerprint: s.cert_fingerprint,
        cert_count: s.cert_count,
        leaf_cn: s.leaf_cn.clone(),
        leaf_names: s.leaf_names.clone(),
        complete,
        declared,
    };
    s.close();
    Ok(out)
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
    match https_fetch(dst, host, port, path, 30_000) {
        Err(e) => {
            console::set_color(LTRED);
            kprintln!("  {}", e.name());
            console::set_color(LTGRAY);
        }
        Ok(f) => {
            let mhz = crate::time::tsc_mhz().max(1);
            let ms = (crate::time::rdtsc() - t0) / mhz / 1000;
            console::set_color(LTGREEN);
            kprintln!("  handshake ok -- TLS 1.3, x25519, chacha20-poly1305");
            console::set_color(LTGRAY);
            if let Some(fp) = f.cert_fingerprint {
                let h = sha256::short_hex(&fp);
                let s = core::str::from_utf8(&h).unwrap_or("?");
                kprintln!("  {} certificate(s); leaf sha256 {}..", f.cert_count, s);
            }
            match &f.identity {
                Identity::Verified { subject, roots } => {
                    console::set_color(LTGREEN);
                    kprintln!("  verified: {} -- chain to 1 of {} trusted roots", subject, roots);
                    console::set_color(LTGRAY);
                }
                Identity::NoTrustStore => {
                    console::set_color(YELLOW);
                    kprintln!("  NOT VERIFIED -- no roots loaded, so nothing could be checked");
                    console::set_color(LTGRAY);
                }
                Identity::Failed(e) => {
                    console::set_color(LTRED);
                    kprintln!("  NOT VERIFIED -- {}", e.name());
                    console::set_color(LTGRAY);
                    // Say what the certificate claims, so a mismatch can be
                    // told apart from a certificate that was not parsed.
                    kprintln!("  leaf cn '{}', {} name(s):", f.leaf_cn, f.leaf_names.len());
                    for n in f.leaf_names.iter().take(6) {
                        kprintln!("    {}", n);
                    }
                }
            }

            if f.status > 0 {
                kprintln!("  HTTP {}", f.status);
            }
            kprintln!("  {} B of body in {} ms", f.body.len(), ms);

            // A truncated body is the one outcome that used to look like a
            // successful one, so it is said in red and it says which of the
            // two ways it ran out.
            if !f.complete {
                console::set_color(LTRED);
                match f.declared {
                    Some(n) => kprintln!("  TRUNCATED -- {} of {} B arrived", f.body.len(), n),
                    None => kprintln!("  TRUNCATED -- the deadline ran out first"),
                }
                console::set_color(LTGRAY);
            }

            let b = String::from_utf8_lossy(&f.body);
            for line in b.lines().take(8) {
                kprintln!("  | {}", line);
            }
        }
    }
    let _ = vec![0u8; 0];
}
