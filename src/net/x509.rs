//! DER parsing, X.509 certificates, and chain validation.
//!
//! ### Why the parser is strict
//!
//! This code reads bytes chosen by whoever we are talking to, before we know
//! who that is. It is the most hostile input in the system. So: every length
//! is checked against the buffer, indefinite-length encodings are refused
//! outright (DER forbids them, and accepting them is how a parser gets led
//! somewhere it should not go), nesting depth is bounded, and nothing is
//! copied on the strength of a length field alone.
//!
//! ### What validation actually checks
//!
//! Four things, and a certificate is rejected if any of them fails:
//!
//!   1. **Signatures chain.** Each certificate is signed by the next one up,
//!      verified with that one's public key, ending at a certificate in the
//!      trust store.
//!   2. **The root is trusted.** By SHA-256 of the full DER, compared against
//!      a built-in list. Fingerprint pinning rather than name lookup: it needs
//!      no name canonicalisation and there is nothing to confuse.
//!   3. **Dates.** Not before, not after, against the CMOS clock.
//!   4. **The name matches.** RFC 6125: subjectAltName dNSName entries, with
//!      wildcards permitted only in the leftmost label -- or iPAddress
//!      entries, when the host was given as an address literal. The two are
//!      kept strictly apart: a dNSName never matches an address.
//!
//! ### What it still does not check
//!
//! Revocation. There is no CRL fetching and no OCSP, so a certificate
//! withdrawn by its issuer is still accepted here until it expires. Path
//! length constraints and key usage bits are parsed but only basicConstraints
//! CA is enforced. Both are worth saying out loud rather than leaving to be
//! discovered.

use crate::crypto::{p256, rsa};
use crate::store::sha256;
use alloc::vec::Vec;

// --- DER ------------------------------------------------------------------

pub const TAG_INTEGER: u8 = 0x02;
pub const TAG_BITSTRING: u8 = 0x03;
pub const TAG_OCTETSTRING: u8 = 0x04;
pub const TAG_NULL: u8 = 0x05;
pub const TAG_OID: u8 = 0x06;
pub const TAG_UTF8: u8 = 0x0C;
pub const TAG_SEQUENCE: u8 = 0x30;
pub const TAG_SET: u8 = 0x31;
pub const TAG_PRINTABLE: u8 = 0x13;
pub const TAG_IA5: u8 = 0x16;
pub const TAG_UTCTIME: u8 = 0x17;
pub const TAG_GENERALTIME: u8 = 0x18;

/// A cursor over a DER buffer.
pub struct Der<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Der<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Der { buf, pos: 0 }
    }

    pub fn empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos.min(self.buf.len())..]
    }

    fn peek_tag(&self) -> Option<u8> {
        self.buf.get(self.pos).copied()
    }

    /// Read one tag-length-value, returning (tag, value) and consuming it.
    fn tlv(&mut self) -> Option<(u8, &'a [u8])> {
        let tag = *self.buf.get(self.pos)?;
        let first = *self.buf.get(self.pos + 1)?;
        let (len, header) = if first & 0x80 == 0 {
            (first as usize, 2)
        } else {
            let n = (first & 0x7F) as usize;
            // 0x80 is the indefinite-length form: legal in BER, forbidden in
            // DER, and a parser that accepts it has to guess where things end.
            if n == 0 || n > 4 {
                return None;
            }
            let mut len = 0usize;
            for i in 0..n {
                len = (len << 8) | *self.buf.get(self.pos + 2 + i)? as usize;
            }
            (len, 2 + n)
        };
        let start = self.pos + header;
        let end = start.checked_add(len)?;
        if end > self.buf.len() {
            return None;
        }
        self.pos = end;
        Some((tag, &self.buf[start..end]))
    }

    /// The full encoding of the next element, tag and length included.
    ///
    /// Needed because a signature is computed over the encoded bytes, not over
    /// the parsed contents -- re-encoding would have to reproduce them exactly
    /// and any difference silently breaks verification.
    pub fn raw_element(&mut self) -> Option<&'a [u8]> {
        let start = self.pos;
        self.tlv()?;
        Some(&self.buf[start..self.pos])
    }

    /// Read the next element only if it has the expected tag.
    ///
    /// The position is restored on a mismatch. That matters more than it
    /// looks: DER is full of optional fields -- the `critical` BOOLEAN in an
    /// extension, the `[0]` version of a certificate -- and a consuming
    /// `expect` turns "this optional field is absent" into "the next two
    /// fields have been eaten", which presents much later as a certificate
    /// with no subjectAltName.
    pub fn expect(&mut self, tag: u8) -> Option<&'a [u8]> {
        let save = self.pos;
        match self.tlv() {
            Some((t, v)) if t == tag => Some(v),
            _ => {
                self.pos = save;
                None
            }
        }
    }

    pub fn sequence(&mut self) -> Option<Der<'a>> {
        self.expect(TAG_SEQUENCE).map(Der::new)
    }

    pub fn integer(&mut self) -> Option<&'a [u8]> {
        self.expect(TAG_INTEGER)
    }

    pub fn skip(&mut self) -> Option<()> {
        self.tlv().map(|_| ())
    }

    /// A context-specific constructed tag, as used for the optional fields of
    /// a certificate.
    fn context(&mut self, n: u8) -> Option<Der<'a>> {
        let want = 0xA0 | n;
        if self.peek_tag()? != want {
            return None;
        }
        self.expect(want).map(Der::new)
    }
}

// --- object identifiers ---------------------------------------------------

const OID_RSA_ENCRYPTION: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x01];
const OID_SHA256_RSA: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B];
const OID_SHA384_RSA: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0C];
const OID_RSA_PSS: &[u8] = &[0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0A];
const OID_EC_PUBLIC_KEY: &[u8] = &[0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01];
const OID_ECDSA_SHA256: &[u8] = &[0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02];
const OID_ECDSA_SHA384: &[u8] = &[0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x03];
const OID_P256: &[u8] = &[0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07];
const OID_P384: &[u8] = &[0x2B, 0x81, 0x04, 0x00, 0x22];
const OID_BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1D, 0x13];
const OID_SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1D, 0x11];
const OID_COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SigAlg {
    RsaPkcs1Sha256,
    RsaPkcs1Sha384,
    RsaPss,
    EcdsaSha256,
    EcdsaSha384,
    Unsupported,
}

fn sig_alg(oid: &[u8]) -> SigAlg {
    match oid {
        x if x == OID_SHA256_RSA => SigAlg::RsaPkcs1Sha256,
        x if x == OID_SHA384_RSA => SigAlg::RsaPkcs1Sha384,
        x if x == OID_RSA_PSS => SigAlg::RsaPss,
        x if x == OID_ECDSA_SHA256 => SigAlg::EcdsaSha256,
        x if x == OID_ECDSA_SHA384 => SigAlg::EcdsaSha384,
        _ => SigAlg::Unsupported,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    Rsa,
    EcP256,
    EcP384,
    Other,
}

#[derive(Clone)]
pub struct Cert {
    /// The whole DER, kept for fingerprinting and trust-store comparison.
    pub der: Vec<u8>,
    /// The exact bytes the signature covers.
    tbs: Vec<u8>,
    pub sig_alg: SigAlg,
    pub signature: Vec<u8>,
    pub key_kind: KeyKind,
    /// RSA: modulus and exponent. EC: the uncompressed point.
    pub key: Vec<u8>,
    pub key_exp: Vec<u8>,
    pub subject_cn: Vec<u8>,
    pub issuer: Vec<u8>,
    pub subject: Vec<u8>,
    pub not_before: u64,
    pub not_after: u64,
    pub is_ca: bool,
    pub dns_names: Vec<Vec<u8>>,
    /// subjectAltName iPAddress entries, raw: four bytes for IPv4, sixteen
    /// for IPv6. Kept unparsed because the comparison is a byte comparison and
    /// converting to text only creates a way to get it wrong.
    pub ip_names: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    Malformed,
    UnsupportedKey,
    UnsupportedSignature,
    BadSignature,
    Expired,
    NotYetValid,
    NameMismatch,
    NoTrustAnchor,
    NotACa,
    ChainTooLong,
}

impl Error {
    pub fn name(self) -> &'static str {
        match self {
            Error::Malformed => "the certificate is malformed",
            Error::UnsupportedKey => "unsupported public key algorithm",
            Error::UnsupportedSignature => "unsupported signature algorithm",
            Error::BadSignature => "a signature in the chain does not verify",
            Error::Expired => "the certificate has expired",
            Error::NotYetValid => "the certificate is not valid yet",
            Error::NameMismatch => "the certificate is for a different host",
            Error::NoTrustAnchor => "the chain does not reach a trusted root",
            Error::NotACa => "a signer in the chain is not a CA",
            Error::ChainTooLong => "the chain is too long",
        }
    }
}

/// Convert YYMMDDHHMMSSZ or YYYYMMDDHHMMSSZ to seconds since 1970.
fn parse_time(tag: u8, v: &[u8]) -> Option<u64> {
    let d = |b: &[u8]| -> Option<u64> {
        let mut n = 0u64;
        for c in b {
            if !c.is_ascii_digit() {
                return None;
            }
            n = n * 10 + (c - b'0') as u64;
        }
        Some(n)
    };
    let (year, rest) = match tag {
        TAG_UTCTIME if v.len() >= 13 => {
            let yy = d(&v[0..2])?;
            // RFC 5280: 00-49 means 2000-2049, 50-99 means 1950-1999.
            (if yy < 50 { 2000 + yy } else { 1900 + yy }, &v[2..])
        }
        TAG_GENERALTIME if v.len() >= 15 => (d(&v[0..4])?, &v[4..]),
        _ => return None,
    };
    let month = d(&rest[0..2])?;
    let day = d(&rest[2..4])?;
    let hour = d(&rest[4..6])?;
    let min = d(&rest[6..8])?;
    let sec = d(&rest[8..10])?;

    // Days from the civil epoch. The shift to March makes the leap day the
    // last day of the year, which removes the special case entirely.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era as i64 * 146097 + doe as i64 - 719468;
    Some((days * 86400 + (hour * 3600 + min * 60 + sec) as i64) as u64)
}

fn parse_name(der: &[u8]) -> (Vec<u8>, Vec<u8>) {
    // The full encoding is kept for issuer/subject comparison; the common name
    // is pulled out for display only.
    let mut cn = Vec::new();
    let mut p = Der::new(der);
    while !p.empty() {
        let Some(set) = p.expect(TAG_SET) else { break };
        let mut s = Der::new(set);
        while !s.empty() {
            let Some(seq) = s.sequence() else { break };
            let mut a = Der::new(seq.rest());
            let Some(oid) = a.expect(TAG_OID) else { break };
            let Some((_, val)) = a.tlv() else { break };
            if oid == OID_COMMON_NAME {
                cn = val.to_vec();
            }
        }
    }
    (cn, der.to_vec())
}

fn parse_extensions(der: &[u8], cert: &mut Cert) {
    let mut p = Der::new(der);
    let Some(seq) = p.sequence() else { return };
    let mut list = Der::new(seq.rest());
    while !list.empty() {
        let Some(ext) = list.sequence() else { break };
        let mut e = Der::new(ext.rest());
        let Some(oid) = e.expect(TAG_OID) else { break };
        // `critical` is an optional BOOLEAN between the OID and the value.
        // Peek rather than try-and-see: `expect` consumes on a *match*, so
        // "try for the OCTET STRING, and skip if that failed" throws away the
        // value in exactly the common case where the extension is not marked
        // critical -- which is most of them, including subjectAltName.
        if e.peek_tag() == Some(0x01) {
            let _ = e.skip();
        }
        let Some(val) = e.expect(TAG_OCTETSTRING) else { continue };

        if oid == OID_BASIC_CONSTRAINTS {
            let mut b = Der::new(val);
            if let Some(inner) = b.sequence() {
                let mut i = Der::new(inner.rest());
                // An empty SEQUENCE means CA is false by default.
                if let Some((tag, v)) = i.tlv() {
                    if tag == 0x01 {
                        cert.is_ca = v.first().copied().unwrap_or(0) != 0;
                    }
                }
            }
        } else if oid == OID_SUBJECT_ALT_NAME {
            let mut s = Der::new(val);
            if let Some(inner) = s.sequence() {
                let mut i = Der::new(inner.rest());
                while !i.empty() {
                    match i.tlv() {
                        // [2] IMPLICIT IA5String is a dNSName.
                        Some((0x82, name)) => cert.dns_names.push(name.to_vec()),
                        // [7] IMPLICIT OCTET STRING is an iPAddress.
                        Some((0x87, ip)) => cert.ip_names.push(ip.to_vec()),
                        Some(_) => {}
                        None => break,
                    }
                }
            }
        }
    }
}

pub fn parse(der: &[u8]) -> Result<Cert, Error> {
    let mut top = Der::new(der);
    let outer = top.sequence().ok_or(Error::Malformed)?;
    let mut c = Der::new(outer.rest());

    let tbs = c.raw_element().ok_or(Error::Malformed)?;

    // signatureAlgorithm and signatureValue follow the tbsCertificate.
    let alg_seq = c.sequence().ok_or(Error::Malformed)?;
    let mut a = Der::new(alg_seq.rest());
    let alg_oid = a.expect(TAG_OID).ok_or(Error::Malformed)?;
    let sig_bits = c.expect(TAG_BITSTRING).ok_or(Error::Malformed)?;
    // A BIT STRING starts with a count of unused trailing bits, always zero
    // for a signature but present all the same.
    let signature = sig_bits.get(1..).ok_or(Error::Malformed)?.to_vec();

    let mut cert = Cert {
        der: der.to_vec(),
        tbs: tbs.to_vec(),
        sig_alg: sig_alg(alg_oid),
        signature,
        key_kind: KeyKind::Other,
        key: Vec::new(),
        key_exp: Vec::new(),
        subject_cn: Vec::new(),
        issuer: Vec::new(),
        subject: Vec::new(),
        not_before: 0,
        not_after: u64::MAX,
        is_ca: false,
        dns_names: Vec::new(),
        ip_names: Vec::new(),
    };

    // --- inside tbsCertificate ---
    let mut t = Der::new(tbs);
    let body = t.sequence().ok_or(Error::Malformed)?;
    let mut b = Der::new(body.rest());

    // [0] version, optional and almost always present.
    if b.context(0).is_some() {}
    b.integer().ok_or(Error::Malformed)?; // serialNumber
    b.skip().ok_or(Error::Malformed)?; // signature algorithm, repeated
    let issuer = b.expect(TAG_SEQUENCE).ok_or(Error::Malformed)?;
    let (_, issuer_raw) = parse_name(issuer);
    cert.issuer = issuer_raw;

    let validity = b.expect(TAG_SEQUENCE).ok_or(Error::Malformed)?;
    let mut v = Der::new(validity);
    let (t1, v1) = v.tlv().ok_or(Error::Malformed)?;
    let (t2, v2) = v.tlv().ok_or(Error::Malformed)?;
    cert.not_before = parse_time(t1, v1).ok_or(Error::Malformed)?;
    cert.not_after = parse_time(t2, v2).ok_or(Error::Malformed)?;

    let subject = b.expect(TAG_SEQUENCE).ok_or(Error::Malformed)?;
    let (cn, subject_raw) = parse_name(subject);
    cert.subject_cn = cn;
    cert.subject = subject_raw;

    // subjectPublicKeyInfo
    let spki = b.expect(TAG_SEQUENCE).ok_or(Error::Malformed)?;
    let mut s = Der::new(spki);
    let alg = s.sequence().ok_or(Error::Malformed)?;
    let mut al = Der::new(alg.rest());
    let key_oid = al.expect(TAG_OID).ok_or(Error::Malformed)?;
    let key_bits = s.expect(TAG_BITSTRING).ok_or(Error::Malformed)?;
    let key_bytes = key_bits.get(1..).ok_or(Error::Malformed)?;

    if key_oid == OID_RSA_ENCRYPTION {
        let mut k = Der::new(key_bytes);
        let seq = k.sequence().ok_or(Error::Malformed)?;
        let mut kk = Der::new(seq.rest());
        let n = kk.integer().ok_or(Error::Malformed)?;
        let e = kk.integer().ok_or(Error::Malformed)?;
        // Strip the sign byte DER adds to a value with the high bit set.
        cert.key = if !n.is_empty() && n[0] == 0 { n[1..].to_vec() } else { n.to_vec() };
        cert.key_exp = e.to_vec();
        cert.key_kind = KeyKind::Rsa;
    } else if key_oid == OID_EC_PUBLIC_KEY {
        let curve = al.expect(TAG_OID).unwrap_or(&[]);
        if curve == OID_P256 && key_bytes.len() == 65 {
            cert.key = key_bytes.to_vec();
            cert.key_kind = KeyKind::EcP256;
        } else if curve == OID_P384 && key_bytes.len() == 97 {
            cert.key = key_bytes.to_vec();
            cert.key_kind = KeyKind::EcP384;
        } else {
            // P-521 still appears occasionally; refusing is honest.
            cert.key_kind = KeyKind::Other;
        }
    }

    // [3] extensions.
    while !b.empty() {
        if let Some(ext) = b.context(3) {
            let e = ext;
            parse_extensions(e.rest(), &mut cert);
            break;
        }
        if b.skip().is_none() {
            break;
        }
    }

    Ok(cert)
}

impl Cert {
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256::hash(&self.der)
    }

    /// Verify that `self` was signed by `issuer`.
    pub fn verify_signed_by(&self, issuer: &Cert) -> Result<(), Error> {
        let digest: Vec<u8> = match self.sig_alg {
            SigAlg::RsaPkcs1Sha256 | SigAlg::EcdsaSha256 | SigAlg::RsaPss => {
                sha256::hash(&self.tbs).to_vec()
            }
            SigAlg::RsaPkcs1Sha384 | SigAlg::EcdsaSha384 => {
                crate::crypto::sha512::sha384(&self.tbs)
            }
            SigAlg::Unsupported => return Err(Error::UnsupportedSignature),
        };

        let ok = match (self.sig_alg, issuer.key_kind) {
            (SigAlg::RsaPkcs1Sha256, KeyKind::Rsa) => {
                rsa::verify_pkcs1_sha256(&issuer.key, &issuer.key_exp, &digest, &self.signature)
            }
            (SigAlg::RsaPkcs1Sha384, KeyKind::Rsa) => {
                rsa::verify_pkcs1_sha384(&issuer.key, &issuer.key_exp, &digest, &self.signature)
            }
            (SigAlg::RsaPss, KeyKind::Rsa) => {
                rsa::verify_pss_sha256(&issuer.key, &issuer.key_exp, &digest, &self.signature)
            }
            // The curve is the issuer's, and the hash is this certificate's;
            // they are independent, so a SHA-384 signature verified by a
            // P-256 key is a legal combination and does occur.
            (SigAlg::EcdsaSha256 | SigAlg::EcdsaSha384, KeyKind::EcP256) => {
                match p256::parse_der_signature(&self.signature, 32) {
                    None => false,
                    Some((r, s)) => p256::verify(p256::Nist::P256, &issuer.key, &digest, &r, &s),
                }
            }
            (SigAlg::EcdsaSha256 | SigAlg::EcdsaSha384, KeyKind::EcP384) => {
                match p256::parse_der_signature(&self.signature, 48) {
                    None => false,
                    Some((r, s)) => p256::verify(p256::Nist::P384, &issuer.key, &digest, &r, &s),
                }
            }
            _ => return Err(Error::UnsupportedSignature),
        };

        if ok {
            Ok(())
        } else {
            Err(Error::BadSignature)
        }
    }

    /// RFC 6125 name matching over subjectAltName.
    pub fn matches_host(&self, host: &str) -> bool {
        // An address literal is matched against iPAddress entries and against
        // nothing else. RFC 6125 is explicit that a dNSName never matches an
        // IP -- so a certificate for the *name* "1.1.1.1", which is a legal
        // thing to issue, must not authenticate the *address* 1.1.1.1. And no
        // wildcard applies: "*.1.1" is not a thing.
        if let Some(ip) = super::parse_ip(host) {
            return self.ip_names.iter().any(|n| n.as_slice() == ip);
        }

        let host = host.trim_end_matches('.').to_ascii_lowercase();
        for name in &self.dns_names {
            let Ok(n) = core::str::from_utf8(name) else { continue };
            let n = n.trim_end_matches('.').to_ascii_lowercase();
            if n == host {
                return true;
            }
            // A wildcard is permitted only as the entire leftmost label, and
            // matches exactly one label -- "*.a.com" covers "b.a.com" and not
            // "c.b.a.com".
            if let Some(suffix) = n.strip_prefix("*.") {
                let Some(dot) = host.find('.') else { continue };
                if &host[dot + 1..] == suffix && !host[..dot].is_empty() {
                    return true;
                }
            }
        }
        // Common Name is deliberately not consulted. It has been deprecated
        // for this purpose since RFC 2818 and accepting it is how a
        // certificate for one thing gets used for another.
        false
    }
}

/// Verify a chain as presented by a server: leaf first, then issuers.
/// What `validate_with` needs to know about the trust store.
///
/// Passed in rather than read from `trust`, so the decision is a pure function
/// of its arguments and every one of its answers can be asserted at boot with
/// no network and no roots file -- the discipline `update::decide` and
/// `ws::admit` follow, for the reason they give: the branch that matters here
/// is the one that almost never runs.
pub trait Anchors {
    /// Whether this exact certificate is a trusted root.
    fn trusts(&self, fingerprint: &[u8; 32]) -> bool;
    /// Every trusted CA root whose subject is `subject`, as DER.
    fn issued_by(&self, subject: &[u8]) -> Vec<Vec<u8>>;
}

/// The loaded trust store, which is what every real caller validates against.
struct Store;

impl Anchors for Store {
    fn trusts(&self, fingerprint: &[u8; 32]) -> bool {
        super::trust::is_trusted(fingerprint)
    }
    fn issued_by(&self, subject: &[u8]) -> Vec<Vec<u8>> {
        super::trust::issuers(subject)
    }
}

pub fn validate(chain: &[Cert], host: &str, now: u64) -> Result<(), Error> {
    validate_with(chain, host, now, &Store)
}

pub fn validate_with(chain: &[Cert], host: &str, now: u64, anchors: &dyn Anchors) -> Result<(), Error> {
    if chain.is_empty() {
        return Err(Error::Malformed);
    }
    if chain.len() > 10 {
        return Err(Error::ChainTooLong);
    }

    let leaf = &chain[0];
    if !leaf.matches_host(host) {
        return Err(Error::NameMismatch);
    }

    // A clock that has not been set makes date checking meaningless rather
    // than merely inconvenient, so it is skipped rather than failed -- and
    // said so at the call site.
    if now > 0 {
        for c in chain {
            if now < c.not_before {
                return Err(Error::NotYetValid);
            }
            if now > c.not_after {
                return Err(Error::Expired);
            }
        }
    }

    // Why a trusted root that *did* have the right name failed to sign, kept so
    // "a root by that name exists and this is not its signature" is not reported
    // as "no root by that name" -- they send somebody to different places.
    let mut near_miss: Option<Error> = None;

    for i in 0..chain.len() {
        let cert = &chain[i];

        // Stop as soon as a certificate is itself trusted.
        if anchors.trusts(&cert.fingerprint()) {
            return Ok(());
        }

        // **Ask the store who issued this before following the server.** The
        // walk used to consult trusted roots only once the served chain ran out,
        // and recognised a trusted certificate only by its fingerprint. A server
        // that sends a cross-signed copy of a root the store holds -- same key,
        // same subject, different issuer, different fingerprint -- therefore led
        // it past the root it trusted and on to whoever cross-signed it, and a
        // store that has since dropped that legacy root refused a chain one step
        // from an anchor it held. Cloudflare's GTS chains are exactly this shape.
        //
        // It stayed invisible because the store it was developed against still
        // carried the legacy root: the walk reached *a* trusted root, just not
        // the one the chain needed, so correctness depended on the store holding
        // a certificate nothing should have required.
        match anchored(cert, anchors) {
            Anchor::Yes => return Ok(()),
            Anchor::No => {}
            Anchor::NearMiss(e) => near_miss = Some(e),
        }

        let issuer = match chain.get(i + 1) {
            Some(next) => next,
            // The store was asked about this certificate's issuer just above, so
            // running out of served certificates here is running out of chain.
            None => return Err(near_miss.unwrap_or(Error::NoTrustAnchor)),
        };
        if !issuer.is_ca {
            return Err(Error::NotACa);
        }
        cert.verify_signed_by(issuer)?;
    }

    Err(near_miss.unwrap_or(Error::NoTrustAnchor))
}

/// What the store says about a certificate's issuer.
enum Anchor {
    /// A trusted CA with that name signed it.
    Yes,
    /// The store holds nothing by that name.
    No,
    /// It holds something by that name, and it did not sign this.
    NearMiss(Error),
}

/// Whether a trusted root issued `cert`, trying **every** root by that name.
///
/// A name is not an identity: a re-keyed root keeps its subject, and trying only
/// the first match -- which this used to, through `trust::find_issuer` -- makes
/// the answer depend on the order of `roots.der`. The signature decides; the
/// name only nominates candidates.
fn anchored(cert: &Cert, anchors: &dyn Anchors) -> Anchor {
    let mut miss = None;
    for der in anchors.issued_by(&cert.issuer) {
        let root = match parse(&der) {
            Ok(r) => r,
            Err(e) => {
                miss = Some(e);
                continue;
            }
        };
        // Both stores in this tree already filter on this. Checked again because
        // `Anchors` is a trait, and the next implementation of it is the one that
        // will not have read this comment.
        if !root.is_ca {
            miss = Some(Error::NotACa);
            continue;
        }
        match cert.verify_signed_by(&root) {
            Ok(()) => return Anchor::Yes,
            Err(e) => miss = Some(e),
        }
    }
    match miss {
        Some(e) => Anchor::NearMiss(e),
        None => Anchor::No,
    }
}

/// A trust store made of DER roots handed in, for the claims below.
struct Fixed<'a>(&'a [&'a [u8]]);

impl Anchors for Fixed<'_> {
    fn trusts(&self, fingerprint: &[u8; 32]) -> bool {
        self.0.iter().any(|der| sha256::hash(der) == *fingerprint)
    }
    fn issued_by(&self, subject: &[u8]) -> Vec<Vec<u8>> {
        self.0
            .iter()
            .filter(|der| parse(der).map(|c| c.is_ca && c.subject == subject).unwrap_or(false))
            .map(|der| der.to_vec())
            .collect()
    }
}

fn check(ok: &mut bool, what: &str, good: bool) {
    if !good {
        *ok = false;
    }
    crate::kprintln!("  {}  {}", if good { "ok  " } else { "FAIL" }, what);
}

/// Chain validation, against the chain that exposed it.
///
/// **`validate` had no claims at all**, which for the function deciding whether
/// a certificate is anybody's is the wrong amount. What found that was a miner
/// image fetching a Cloudflare-fronted URL with 121 roots loaded and being told
/// `NOT VERIFIED`: Cloudflare serves `leaf <- WE1 <- GTS Root R4`, where that
/// last certificate is the copy of GTS Root R4 **cross-signed by GlobalSign Root
/// CA**. The self-signed GTS Root R4 in the store has the same key and the same
/// subject and a different fingerprint, so a walk that only recognises roots by
/// fingerprint followed the served copy to GlobalSign's legacy root -- which
/// current stores have dropped -- and gave up one step short of a root it held.
///
/// The fixtures are those four certificates, byte for byte, and `NOW` is a
/// fixed instant inside all four validity windows, so these claims do not
/// expire when the leaf does.
pub fn selftest() -> bool {
    const LEAF: &[u8] = include_bytes!("fixtures/gts-leaf.der");
    const WE1: &[u8] = include_bytes!("fixtures/gts-we1.der");
    const CROSS: &[u8] = include_bytes!("fixtures/gts-r4-cross.der");
    const ROOT: &[u8] = include_bytes!("fixtures/gts-r4-root.der");
    // Matches the leaf's `*.supabase.co`. 2026-09-28 12:00 UTC, which is after
    // the leaf's notBefore (2026-08-26) and before its notAfter (2026-11-24).
    const HOST: &str = "vermcdgpqncfsralpesz.supabase.co";
    const NOW: u64 = 1_790_596_800;

    let mut ok = true;
    let parsed: Vec<Cert> = [LEAF, WE1, CROSS].iter().filter_map(|d| parse(d).ok()).collect();
    check(&mut ok, "the four fixture certificates parse", parsed.len() == 3 && parse(ROOT).is_ok());
    if parsed.len() != 3 {
        return false;
    }
    let served = &parsed[..];
    let root_only = Fixed(&[ROOT]);

    check(
        &mut ok,
        "the chain as Cloudflare serves it validates against the self-signed GTS Root R4",
        validate_with(served, HOST, NOW, &root_only) == Ok(()),
    );
    // The same answer with the cross-signed copy dropped, which is what a server
    // that sends a tidy chain looks like. It passed before and must still pass.
    check(
        &mut ok,
        "and so does the same chain without the cross-signed copy",
        validate_with(&served[..2], HOST, NOW, &root_only) == Ok(()),
    );
    check(
        &mut ok,
        "and a store trusting the served copy itself still anchors there",
        validate_with(served, HOST, NOW, &Fixed(&[CROSS])) == Ok(()),
    );

    // **The refusals, which are what makes the acceptance above worth anything.**
    // Anchoring on a trusted issuer is a new way to say yes, so the claims that
    // matter are the ones showing it still says no.
    check(
        &mut ok,
        "an empty store trusts nothing",
        validate_with(served, HOST, NOW, &Fixed(&[])) == Err(Error::NoTrustAnchor),
    );
    check(
        &mut ok,
        "a leaf alone is not anchored by a root that did not sign it",
        validate_with(&served[..1], HOST, NOW, &root_only) == Err(Error::NoTrustAnchor),
    );
    // One byte at the end of WE1 is the last byte of its ECDSA `s`, so the
    // structure still parses and only the signature is wrong. GTS Root R4 is
    // trusted and its subject matches WE1's issuer exactly -- the precise case
    // where a check that matched names and skipped the signature would say yes.
    let mut forged = WE1.to_vec();
    *forged.last_mut().unwrap() ^= 0x01;
    match parse(&forged) {
        Ok(bad) => {
            let chain = [served[0].clone(), bad];
            check(
                &mut ok,
                "a trusted root's name on a signature it did not make is refused",
                validate_with(&chain, HOST, NOW, &root_only) == Err(Error::BadSignature),
            );
        }
        Err(_) => check(&mut ok, "a forged WE1 still parses, so the forgery test ran", false),
    }
    check(
        &mut ok,
        "another host's name is refused before any signature is looked at",
        validate_with(served, "example.com", NOW, &root_only) == Err(Error::NameMismatch),
    );
    // 2026-12-01, a week past the leaf's notAfter and years inside everything
    // above it. The first figure written here was 2026-11-18, which is *before*
    // the leaf expires, and the claim would have failed for a reason that had
    // nothing to do with expiry -- so the instants are computed, not estimated.
    check(
        &mut ok,
        "a chain read after the leaf expires is refused as expired",
        validate_with(served, HOST, 1_796_083_200, &root_only) == Err(Error::Expired),
    );
    // 2026-07-25, a month before the leaf's notBefore.
    check(
        &mut ok,
        "and one read before it was issued as not yet valid",
        validate_with(served, HOST, 1_785_000_000, &root_only) == Err(Error::NotYetValid),
    );
    ok
}
