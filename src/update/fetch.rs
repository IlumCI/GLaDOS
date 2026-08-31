//! Getting an update off the network, and refusing to be lied to on the way.
//!
//! Every refusal here names itself. An updater that answers "failed" to a
//! missing trust store, a truncated transfer and a wrong image alike is an
//! updater whose failures all get diagnosed as the same thing, and the wrong
//! thing gets fixed.
//!
//! ### The check the `https` verb does not make
//!
//! `tls::https_fetch` reports what it established about the peer and hands
//! over the body either way. That is right for a person reading a page: the
//! verdict is printed, and they can decide. It is wrong for a machine
//! deciding what to boot, so this module treats anything short of `Verified`
//! as a failure -- including, and especially, "no roots are loaded", which is
//! the state a machine is in when `\GLADOS\roots.der` is missing and is
//! otherwise indistinguishable from success.

use super::manifest::{self, Manifest};
use crate::net::html::Url;
use crate::net::{dhcp, dns, tls};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A manifest is a few hundred bytes and should be quick or wrong.
const MANIFEST_MS: u64 = 20_000;

/// An image is a few megabytes through a thirty-two kilobyte window. At CDN
/// latency that is seconds; on a bad link it is minutes, and minutes of
/// waiting beats a download that gives up at a fixed fifteen seconds and
/// reports the truncation as a bad signature.
const IMAGE_MS: u64 = 300_000;

/// Make sure there is a route out, and say so if one had to be arranged.
///
/// `net::init` fills the primary interface in with QEMU's addresses and prints
/// "'dhcp' to ask", so a machine nobody has told to run DHCP has a gateway it
/// cannot reach. Failing there would be reporting a network problem this could
/// have fixed -- but it is reported rather than done silently, because
/// acquiring a lease changes the machine's address.
pub fn online() -> Result<(), String> {
    let n = crate::net::primary();
    if n == crate::net::LO {
        return Err(String::from("no network interface is up"));
    }
    if crate::net::config_of(n).gateway != crate::net::UNSPECIFIED {
        return Ok(());
    }
    let c = dhcp::configure_on(n).map_err(|e| format!("no address, and DHCP {}", e.name()))?;
    crate::kprintln!(
        "  dhcp: {}.{}.{}.{} via {}.{}.{}.{}",
        c.ip[0], c.ip[1], c.ip[2], c.ip[3],
        c.gateway[0], c.gateway[1], c.gateway[2], c.gateway[3]
    );
    Ok(())
}

/// Fetch one URL, or say precisely which way it went wrong.
///
/// `what` names the thing being fetched, so a failure reads as "the image: ..."
/// rather than as a bare error a caller has to guess the subject of.
pub fn get(url: &Url, what: &str, timeout_ms: u64) -> Result<Vec<u8>, String> {
    let ip = dns::lookup(&url.host).map_err(|e| format!("{}: {} -- {}", what, url.host, e.name()))?;

    let f = tls::https_fetch(ip, &url.host, url.port, &url.path, timeout_ms)
        .map_err(|e| format!("{}: {}", what, e.name()))?;
    inspect(f, what)
}

/// Everything that has to be true of a response before its body is believed.
///
/// Shared by both fetchers rather than written twice: the check that would get
/// forgotten in the second copy is the identity one, and forgetting it is
/// indistinguishable from success right up until it matters.
fn inspect(f: tls::Fetched, what: &str) -> Result<Vec<u8>, String> {
    match &f.identity {
        tls::Identity::Verified { .. } => {}
        tls::Identity::NoTrustStore => {
            return Err(String::from(
                "no roots are loaded, so the server could be anyone. Put roots.der on the ESP and reboot",
            ))
        }
        tls::Identity::Failed(e) => {
            return Err(format!("{}: the server did not verify -- {}", what, e.name()))
        }
    }

    if f.status != 200 {
        return Err(format!("{}: the server answered HTTP {}", what, f.status));
    }

    // A truncated body used to be indistinguishable from a whole one. It is
    // the failure most worth naming, because the next thing to notice it
    // would be the signature check, which would blame the signer.
    if !f.complete {
        return Err(match f.declared {
            Some(n) => format!(
                "{}: short read -- {} of {} B before the transfer stopped",
                what,
                f.body.len(),
                n
            ),
            None => format!("{}: the transfer did not finish", what),
        });
    }

    Ok(f.body)
}

/// Fetch one URL carrying a device code.
///
/// The code is checked for control characters before it goes anywhere near a
/// header. It arrives from an operator typing it in, and a value containing a
/// carriage return is not a code -- it is a second header somebody else wrote.
pub fn get_with(url: &Url, what: &str, timeout_ms: u64, code: &str) -> Result<Vec<u8>, String> {
    if code.is_empty() || code.chars().any(|c| c.is_control() || c == ':') {
        return Err(String::from("that device code has characters a header cannot carry"));
    }
    let bearer = format!("Bearer {}", code);

    let ip = dns::lookup(&url.host).map_err(|e| format!("{}: {} -- {}", what, url.host, e.name()))?;
    let f = tls::https_fetch_with(
        ip,
        &url.host,
        url.port,
        &url.path,
        timeout_ms,
        &[("Authorization", bearer.as_str())],
    )
    .map_err(|e| format!("{}: {}", what, e.name()))?;
    inspect(f, what)
}

/// Fetch and verify the signed manifest for a channel.
///
/// One object, one round trip. The gated channel needs the device code on the
/// way in, and it goes in a header rather than the URL: a query string is
/// logged by every proxy between here and there, and a bearer token in a log
/// is a bearer token somebody else has.
pub fn manifest_at(base: &Url, code: Option<&str>) -> Result<Manifest, String> {
    let blob = match code {
        Some(c) => get_with(base, "the manifest", MANIFEST_MS, c)?,
        None => get(base, "the manifest", MANIFEST_MS)?,
    };
    let (text, sig) = manifest::split(&blob).ok_or_else(|| {
        alloc::format!(
            "the source answered with {} B, which is too short to be a signed manifest",
            blob.len()
        )
    })?;
    manifest::verified(text, sig).map_err(|b| b.why())
}

/// Fetch the image a manifest names, and the signature over it.
///
/// The image is checked against the manifest's length and digest before the
/// signature is looked at, so a short read is reported as a short read. Only
/// then does `update::verify` get asked the question it is for.
pub fn image_for(m: &Manifest) -> Result<(Vec<u8>, Vec<u8>), String> {
    let image = get(&m.image, "the image", IMAGE_MS)?;
    m.matches(&image)?;

    let sig = get(&m.sig, "the image signature", MANIFEST_MS)?;
    if sig.len() != super::SIG_LEN {
        return Err(format!(
            "the signature is {} B, and a GLADOSIG is {}",
            sig.len(),
            super::SIG_LEN
        ));
    }

    let v = super::verify(&image, &sig);
    if !v.ok() {
        return Err(String::from(v.why()));
    }
    Ok((image, sig))
}
