//! Where updates come from, and what this machine may ask for.
//!
//! ### Why the source URL is allowed to be configured
//!
//! It looks like a trust anchor and it is not one. The manifest is signed by
//! the key compiled into this kernel, and so is the image it names, so a
//! source that is hostile or merely wrong can serve nothing this machine will
//! install. What pointing at a different host can do is deny service and
//! reveal which version is being asked for -- real, and a different order of
//! problem from installing somebody else's kernel.
//!
//! That is what makes it safe to ship with no source at all and let an
//! operator supply one, which is the state this is in until the bucket it will
//! default to exists.
//!
//! ### Two channels
//!
//! `stable` is a static object with no authentication in front of it: one GET,
//! no server-side compute, nothing to rate-limit and nothing whose failure can
//! take the free path down. Security fixes live there and always will.
//!
//! `experimental` is the gated one. The machine sends a device code and gets
//! back a manifest whose URLs are short-lived. The gate is the server
//! declining to answer -- there is no local check, and there could not be a
//! meaningful one in a kernel whose source is published.

use crate::net::html::{parse_url, Url};
use alloc::string::{String, ToString};

pub const SOURCE: &str = "/sys/update/source";
pub const CHANNEL: &str = "/sys/update/channel";
pub const CODE: &str = "/sys/update/code";
pub const SEEN: &str = "/sys/update/seen";

/// Where the staged pair is held between `update fetch` and `update stage`.
pub const IMAGE: &str = "/tmp/staged.efi";
pub const SIGNATURE: &str = "/tmp/staged.efi.sig";

/// Compiled-in default source, empty until there is a bucket to name.
///
/// Empty rather than a plausible-looking placeholder: a URL that resolves to
/// nothing produces a network error, and a network error is what a machine
/// with no configured source would report anyway -- indistinguishable from a
/// real outage, and it would send somebody debugging DNS.
pub const DEFAULT_SOURCE: &str = "https://vermcdgpqncfsralpesz.supabase.co";

fn read(path: &str) -> Option<String> {
    let raw = crate::sysbox::read_blob(path)?;
    let text = String::from_utf8(raw).ok()?;
    let text = text.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// The channel this machine follows. `stable` unless told otherwise, because
/// the default has to be the one that is free and always has the fixes in it.
pub fn channel() -> String {
    read(CHANNEL).unwrap_or_else(|| "stable".to_string())
}

pub fn set_channel(name: &str) -> Result<(), String> {
    match name {
        "stable" | "experimental" => {}
        _ => return Err(alloc::format!("no channel called '{}' -- stable or experimental", name)),
    }
    if !crate::sysbox::write_text(CHANNEL, name) {
        return Err(String::from("could not write the channel"));
    }
    Ok(())
}

/// The device code, if one has been linked. Opaque here on purpose: this
/// kernel has no idea what makes it valid and no way to check, which is the
/// correct amount for it to know.
pub fn code() -> Option<String> {
    read(CODE)
}

pub fn set_code(code: &str) -> bool {
    crate::sysbox::write_text(CODE, code.trim())
}

pub fn unlink() -> bool {
    crate::sysbox::detach(CODE)
}

pub fn source() -> Option<String> {
    read(SOURCE).or_else(|| {
        if DEFAULT_SOURCE.is_empty() {
            None
        } else {
            Some(DEFAULT_SOURCE.to_string())
        }
    })
}

pub fn set_source(url: &str) -> Result<(), String> {
    let u = parse_url(url).ok_or_else(|| String::from("that is not a URL"))?;
    if !u.https {
        return Err(String::from("the source has to be https"));
    }
    // An origin, because the channel paths are compiled in and appended. A
    // path here would be silently discarded, which looks like it was accepted.
    if u.path != "/" {
        return Err(alloc::format!(
            "give the origin only -- https://{}, without '{}'",
            u.host,
            u.path
        ));
    }
    if !crate::sysbox::write_text(SOURCE, url.trim().trim_end_matches('/')) {
        return Err(String::from("could not write the source"));
    }
    Ok(())
}

/// The path a channel's signed manifest lives at, relative to the origin.
///
/// Compiled in rather than configured, and both of them, so that switching
/// channel cannot be done by pointing at a different host. `stable` is a
/// static object -- one GET, no server-side compute, nothing to rate-limit.
/// `experimental` is a function that checks entitlement first and answers with
/// a manifest whose image URLs are short-lived.
fn path_for(name: &str) -> &'static str {
    match name {
        "experimental" => "/functions/v1/channel",
        _ => "/storage/v1/object/public/stable/manifest",
    }
}

/// The manifest URL for the channel in force.
///
/// The stored source is an *origin* and nothing more, so the two channel paths
/// are derived rather than edited. An earlier version stored the stable
/// manifest URL and rewrote its last path component for experimental, which
/// worked exactly until either path changed shape.
pub fn endpoint() -> Result<Url, String> {
    let origin = source().ok_or_else(|| {
        String::from("no update source is configured -- 'update source <url>' to set one")
    })?;
    let mut u = parse_url(&origin).ok_or_else(|| String::from("the stored source is not a URL"))?;
    u.path = String::from(path_for(&channel()));
    Ok(u)
}

/// What was last seen on the wire, so `update` with no arguments can answer
/// without going out to the network.
pub fn remember(version: &str, notes: &str) {
    let mut s = String::from(version);
    if !notes.is_empty() {
        s.push(' ');
        s.push_str(notes);
    }
    let _ = crate::sysbox::write_text(SEEN, &s);
}

pub fn seen() -> Option<String> {
    read(SEEN)
}
