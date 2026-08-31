//! What a release says about itself, and whether to believe it.
//!
//! The image signature already proves the bytes came from the signer. It says
//! nothing about *which* signed image was offered, and that gap is real: a
//! host that chooses the answer can choose an old one with a known hole in it,
//! and every byte of that verifies. So the manifest is signed too, by the same
//! key through the same verifier, and the client refuses anything not newer
//! than what it is already running.
//!
//! Those are two defences against the same move and neither is sufficient
//! alone. The signature stops a stranger writing the manifest. The version
//! check stops the signer's own older work being replayed at us.
//!
//! The format is the one every other stored config here uses: a version line,
//! then `key value`. The parser is short enough to read in one sitting and
//! cannot be made to allocate without bound, and a manifest a person can read
//! is one they can check by hand on the day the machine disagrees with them.

use super::Verdict;
use crate::net::html::{parse_url, Url};
use crate::store::sha256;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The first line. A file not beginning with exactly this is refused before
/// any of its other lines are believed.
///
/// Bump the number for a change an older kernel would *misread*. Adding a
/// field does not need one: unknown keys are ignored, and the signature covers
/// them either way.
const HEADER: &str = "glados-update 1";

pub struct Manifest {
    pub channel: String,
    pub version: String,
    pub image: Url,
    pub sig: Url,
    pub size: usize,
    pub sha256: [u8; 32],
    pub notes: String,
}

/// Why a manifest was not accepted.
///
/// Distinct cases rather than one error, because "the host served something
/// that is not a manifest" and "the host served a manifest signed by somebody
/// else" deserve different reactions, and the second is worth being loud
/// about.
#[derive(Clone, Copy, PartialEq)]
pub enum Bad {
    /// The signature did not check out. Carries the verdict, so "no key is
    /// provisioned" does not get reported as "somebody forged this".
    Unsigned(Verdict),
    NotAManifest,
    Missing(&'static str),
    Malformed(&'static str),
}

impl Bad {
    pub fn why(self) -> String {
        match self {
            Bad::Unsigned(v) => format!("the manifest is not signed: {}", v.why()),
            Bad::NotAManifest => {
                format!("not a manifest -- expected '{}' on the first line", HEADER)
            }
            Bad::Missing(k) => format!("the manifest has no '{}' line", k),
            Bad::Malformed(k) => format!("the manifest's '{}' line does not parse", k),
        }
    }
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// A 64-character hex digest, or nothing. The length is checked first, so a
/// short field cannot be zero-extended into a different digest.
fn unhex32(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (nibble(b[2 * i])? << 4) | nibble(b[2 * i + 1])?;
    }
    Some(out)
}

/// An https URL, or nothing.
///
/// Plain http is refused rather than quietly upgraded. The bytes behind these
/// URLs are signed, so a downgrade could not substitute an image -- but it
/// would tell anyone on the path exactly which version this machine is about
/// to install, and there is no reason to accept that in a file we write
/// ourselves.
fn url(s: &str) -> Option<Url> {
    parse_url(s).filter(|u| u.https)
}

/// Parse a manifest. Says nothing about whether it was signed.
pub fn parse(text: &[u8]) -> Result<Manifest, Bad> {
    let text = core::str::from_utf8(text).map_err(|_| Bad::NotAManifest)?;
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some(HEADER) {
        return Err(Bad::NotAManifest);
    }

    let mut channel = None;
    let mut version = None;
    let mut image = None;
    let mut sig = None;
    let mut size = None;
    let mut digest = None;
    let mut notes = String::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = match line.split_once(char::is_whitespace) {
            Some((k, v)) => (k, v.trim()),
            None => (line, ""),
        };
        match key {
            "channel" => channel = Some(value.to_string()),
            "version" => version = Some(value.to_string()),
            "image" => image = Some(url(value).ok_or(Bad::Malformed("image"))?),
            "sig" => sig = Some(url(value).ok_or(Bad::Malformed("sig"))?),
            "size" => size = Some(value.parse::<usize>().map_err(|_| Bad::Malformed("size"))?),
            "sha256" => digest = Some(unhex32(value).ok_or(Bad::Malformed("sha256"))?),
            "notes" => notes = value.to_string(),
            // Ignored rather than refused, so a later format can add a line
            // without every older kernel rejecting the whole file. The
            // signature covers it regardless; HEADER guards the change that
            // would be misread rather than merely unread.
            _ => {}
        }
    }

    Ok(Manifest {
        channel: channel.ok_or(Bad::Missing("channel"))?,
        version: version.ok_or(Bad::Missing("version"))?,
        image: image.ok_or(Bad::Missing("image"))?,
        sig: sig.ok_or(Bad::Missing("sig"))?,
        size: size.ok_or(Bad::Missing("size"))?,
        sha256: digest.ok_or(Bad::Missing("sha256"))?,
        notes,
    })
}

/// Split a signed manifest into its text and the signature over it.
///
/// A signed manifest is its text followed by exactly `SIG_LEN` bytes. One
/// object rather than two, for two reasons: this stack has one connection and
/// no pipelining, so a second fetch is a second handshake; and two objects can
/// be served out of step with each other, which produces a signature failure
/// that is really a deployment race and gets diagnosed as an attack.
///
/// The length is fixed, so the split needs no framing and cannot be argued
/// with by the thing being split.
pub fn split(blob: &[u8]) -> Option<(&[u8], &[u8])> {
    if blob.len() <= super::SIG_LEN {
        return None;
    }
    Some(blob.split_at(blob.len() - super::SIG_LEN))
}

/// Parse a manifest only if the update key signed it.
///
/// Signature first, deliberately. Parsing attacker-chosen text is the larger
/// of the two surfaces, and there is no reason to enter it before knowing the
/// bytes came from the signer.
pub fn verified(text: &[u8], sig: &[u8]) -> Result<Manifest, Bad> {
    let v = super::verify(text, sig);
    if !v.ok() {
        return Err(Bad::Unsigned(v));
    }
    parse(text)
}

impl Manifest {
    /// Whether this is worth installing over what is running.
    ///
    /// The only anti-rollback that exists. A correctly signed older image
    /// verifies forever and always will, so refusing it has to happen here, in
    /// the client, against the version the running kernel was built with.
    pub fn is_upgrade(&self) -> bool {
        crate::version_newer(&self.version, crate::VERSION)
    }

    /// Whether these bytes are the image this manifest describes.
    ///
    /// Length first: it is the cheap half, and a short read has a different
    /// cause and a different fix from a wrong image. The digest collapses both
    /// into "no", and "no" is what sent somebody hunting a signing bug the
    /// last time a download was quietly cut short.
    pub fn matches(&self, image: &[u8]) -> Result<(), String> {
        if image.len() != self.size {
            return Err(format!(
                "{} B arrived, the manifest says {} -- a short read, not a wrong image",
                image.len(),
                self.size
            ));
        }
        if sha256::hash(image) != self.sha256 {
            return Err(String::from(
                "the right number of bytes, and not the ones the manifest names",
            ));
        }
        Ok(())
    }
}

/// What can be checked with no network and no key.
///
/// Every claim here is about the parser refusing something, which is the whole
/// job: a manifest arrives from a host nobody here controls, and the only
/// question that matters is whether a bad one can be made to look good.
pub fn selftest() -> bool {
    use crate::kprintln;

    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            ok = false;
        }
        kprintln!("  {}  {}", if good { "ok " } else { "FAIL" }, what);
    };

    // The digest is SHA-256 of the four bytes "test", so `size` and `sha256`
    // below describe a real payload rather than a plausible-looking one.
    let good: &[u8] = b"glados-update 1\n\
channel stable\n\
version 9.9.9\n\
image https://example.invalid/glados-9.9.9.efi\n\
sig https://example.invalid/glados-9.9.9.efi.sig\n\
size 4\n\
sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n\
notes a manifest that exists only in this test\n";

    let m = parse(good);
    claim("a well-formed manifest parses", m.is_ok());

    if let Ok(m) = &m {
        claim(
            "its fields survive the trip",
            m.channel == "stable"
                && m.version == "9.9.9"
                && m.size == 4
                && m.image.host == "example.invalid"
                && m.image.path == "/glados-9.9.9.efi",
        );
        claim("the bytes it describes are accepted", m.matches(b"test").is_ok());
        claim(
            "a short read is reported as a short read",
            m.matches(b"tes")
                .err()
                .map(|e| e.contains("short read"))
                .unwrap_or(false),
        );
        claim(
            "the right length of the wrong bytes is refused",
            m.matches(b"TEST").is_err(),
        );
        claim("and 9.9.9 is newer than what is running", m.is_upgrade());
    }

    claim(
        "a future format is refused rather than read as this one",
        matches!(
            parse(b"glados-update 2\nchannel stable\n"),
            Err(Bad::NotAManifest)
        ),
    );
    claim(
        "a manifest missing a field it needs is refused",
        matches!(
            parse(b"glados-update 1\nchannel stable\n"),
            Err(Bad::Missing(_))
        ),
    );
    claim(
        "a plain-http image URL is refused",
        matches!(
            parse(b"glados-update 1\nimage http://example.invalid/x.efi\n"),
            Err(Bad::Malformed("image"))
        ),
    );
    claim(
        "a truncated digest is refused rather than padded",
        matches!(
            parse(b"glados-update 1\nsha256 9f86d081\n"),
            Err(Bad::Malformed("sha256"))
        ),
    );

    let mut signed = good.to_vec();
    signed.extend_from_slice(&[0u8; super::SIG_LEN]);
    claim(
        "a signed manifest splits at a fixed offset from the end",
        split(&signed).map(|(t, s)| t == good && s.len() == super::SIG_LEN).unwrap_or(false),
    );
    claim(
        "and something shorter than a signature is not one",
        split(&[0u8; 8]).is_none(),
    );

    // The one that separates "checks the signature" from "checks the magic":
    // a perfectly well-formed manifest with nothing signing it.
    claim(
        "an unsigned manifest is refused however well formed it is",
        matches!(verified(good, &[]), Err(Bad::Unsigned(_))),
    );

    ok
}
