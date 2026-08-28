//! What an application *is*, named by a hash, and what it is allowed to do.
//!
//! An application is two files. A manifest is the statement of which two, in
//! which versions, what it descends from, and whether it is asking for more
//! than the sandbox gives. It is text, its name is the SHA-256 of that text,
//! and it is stored beside its siblings under `/app/manifests`.
//!
//! This mirrors `ai::godel::Variant` deliberately rather than sharing code with
//! it: the shape is the same -- render to canonical text, hash the text, store
//! under the hash, name the parent by hash, adopt by writing a pointer -- and
//! `Variant`'s functions are bound to a struct that means something else.
//! Copying the shape and not the code keeps both readable; sharing it would
//! have meant a trait with one useful implementation each.
//!
//! ### Why the capability request is inside the hash
//!
//! `raw` is part of the rendered text, so an application that starts asking for
//! the operator's powers **becomes a different manifest**. A grant names a
//! hash. It therefore cannot survive the request it was given for changing, and
//! it cannot be inherited by a later version of the same application.
//!
//! ### Why the manifest is only a request
//!
//! Approval is not in it. If a manifest could carry its own approval then
//! anything that could write a manifest -- which is anything that can write two
//! files, which is the whole point of generated applications -- could approve
//! itself. So the manifest says what is wanted and `/app/grants` says what was
//! allowed, and only the operator writes the second.
//!
//! ### Identity is derived, not authored
//!
//! Nobody writes a manifest. `current` computes one from the files on disk, so
//! a manifest cannot describe an application that is not there, and editing a
//! byte of `code.ai&xi` renames the node whether or not anyone remembered to say
//! so. That is the property the grant depends on.

use crate::store::sha256;
use crate::sysbox;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::ROOT;

/// Where manifests, grants and the ledger live.
const NODES: &str = "/app/manifests";
const GRANTS: &str = "/app/grants";
const LEDGER: &str = "/app/ledger.txt";
/// Where a version's actual bytes live, addressed by their own hash.
const BLOBS: &str = "/app/blobs";

pub fn hex32(h: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    const D: &[u8; 16] = b"0123456789abcdef";
    for b in h.iter() {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 15) as usize] as char);
    }
    s
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

pub fn from_hex32(text: &str) -> Option<[u8; 32]> {
    let b = text.as_bytes();
    if b.len() < 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (hex_val(b[i * 2])? << 4) | hex_val(b[i * 2 + 1])?;
    }
    Some(out)
}

#[derive(Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The version this one replaced, by hash. `None` for the first.
    pub parent: Option<[u8; 32]>,
    pub panel: [u8; 32],
    pub code: [u8; 32],
    /// Asks to run with the operator's capabilities instead of the sandbox.
    ///
    /// One bit rather than a set of them. A matrix of partial powers reads as
    /// more careful and is harder to answer: an operator approving a request
    /// has to hold the whole of it in their head, and "this application may
    /// write outside itself but not touch ports" is a sentence nobody can
    /// check against the program. Sandboxed or trusted is a question with an
    /// answer.
    pub raw: bool,
}

impl Manifest {
    /// The canonical text. Everything that changes what the application is,
    /// and nothing else -- a timestamp in here would make the same application
    /// a different node every time it was looked at.
    pub fn render(&self) -> String {
        let mut s = String::from("app 1\n");
        s.push_str("parent ");
        s.push_str(&self.parent.map(|h| hex32(&h)).unwrap_or_else(|| "none".to_string()));
        s.push('\n');
        s.push_str("panel ");
        s.push_str(&hex32(&self.panel));
        s.push('\n');
        s.push_str("code ");
        s.push_str(&hex32(&self.code));
        s.push('\n');
        s.push_str("raw ");
        s.push_str(if self.raw { "yes" } else { "no" });
        s.push('\n');
        s
    }

    pub fn hash(&self) -> [u8; 32] {
        sha256::hash(self.render().as_bytes())
    }

    /// Write it under its own hash, if it is not already there.
    ///
    /// Write-if-absent, so rediscovering a version that already exists is the
    /// same node rather than a second one that looks identical.
    pub fn store(&self) -> [u8; 32] {
        let h = self.hash();
        let path = alloc::format!("{}/{}", NODES, hex32(&h));
        if sysbox::read_blob(&path).is_none() {
            sysbox::write_text(&path, &self.render());
        }
        h
    }

    pub fn load(h: &[u8; 32]) -> Option<Manifest> {
        let path = alloc::format!("{}/{}", NODES, hex32(h));
        let bytes = sysbox::read_blob(&path)?;
        Manifest::parse(&String::from_utf8_lossy(&bytes))
    }

    /// Read a manifest from its own text.
    ///
    /// Separate from `load` so the format can be checked without a namespace:
    /// the boot selftests run before `sysbox::init`, and a round trip that can
    /// only be exercised against storage is one that does not get exercised.
    pub fn parse(text: &str) -> Option<Manifest> {
        let mut parent = None;
        let mut panel = None;
        let mut code = None;
        let mut raw = None;
        for line in text.lines() {
            let (k, v) = line.split_once(' ').unwrap_or((line, ""));
            match k {
                "app" if v != "1" => return None,
                "parent" => parent = if v == "none" { Some(None) } else { from_hex32(v).map(Some) },
                "panel" => panel = from_hex32(v),
                "code" => code = from_hex32(v),
                "raw" => raw = Some(v == "yes"),
                _ => {}
            }
        }
        Some(Manifest { parent: parent?, panel: panel?, code: code?, raw: raw? })
    }
}

/// The format, checked against itself.
///
/// A manifest is an identity: if it does not survive being written and read
/// back, two runs disagree about which application they are looking at, and a
/// grant given to one stops applying to the other for no visible reason.
pub fn selftest() -> bool {
    let a = Manifest {
        parent: None,
        panel: [0x11; 32],
        code: [0x22; 32],
        raw: false,
    };
    match Manifest::parse(&a.render()) {
        Some(b) if b == a => {}
        _ => return false,
    }
    let child = Manifest { parent: Some(a.hash()), ..a.clone() };
    match Manifest::parse(&child.render()) {
        Some(b) if b == child => {}
        _ => return false,
    }
    // Lineage is real: naming a parent changes the node.
    if child.hash() == a.hash() {
        return false;
    }
    // The request is inside the identity, which is what stops a grant from
    // surviving an application that starts asking for more.
    let asking = Manifest { raw: true, ..a.clone() };
    if asking.hash() == a.hash() {
        return false;
    }
    // ...and so is every file it names.
    let other = Manifest { code: [0x23; 32], ..a.clone() };
    if other.hash() == a.hash() {
        return false;
    }
    // Hex survives the trip in both directions, which the hashes above depend
    // on being true.
    let h = a.hash();
    if from_hex32(&hex32(&h)) != Some(h) {
        return false;
    }
    if from_hex32("short").is_some() || from_hex32(&"z".repeat(64)).is_some() {
        return false;
    }
    // A version this reader does not know is refused rather than half-read.
    Manifest::parse("app 2\nparent none\npanel 00\ncode 00\nraw no\n").is_none()
}

fn file_hash(path: &str) -> Option<[u8; 32]> {
    sysbox::read_blob(path).map(|b| sha256::hash(&b))
}

/// Whether an application asks for the operator's capabilities.
///
/// Declared by a file rather than a line inside `code.ai&xi`, so asking is visible
/// in a directory listing and cannot be buried three hundred lines into a
/// program somebody skimmed.
fn asks_raw(name: &str) -> bool {
    sysbox::read_blob(&alloc::format!("{}/{}/raw", ROOT, name)).is_some()
}

/// The manifest the files on disk currently describe.
///
/// If they describe exactly what `HEAD` already names, this *is* `HEAD` --
/// parent and all. Otherwise it is a new node whose parent is `HEAD`. Without
/// that check every call would mint a fresh hash from an unchanged application
/// and no grant would ever match twice.
pub fn current(name: &str) -> Option<Manifest> {
    let panel = file_hash(&alloc::format!("{}/{}/panel.ui", ROOT, name))?;
    let code = file_hash(&alloc::format!("{}/{}/code.ai&xi", ROOT, name))?;
    let raw = asks_raw(name);
    let head = head(name);
    if let Some(h) = head {
        if let Some(m) = Manifest::load(&h) {
            if m.panel == panel && m.code == code && m.raw == raw {
                return Some(m);
            }
        }
    }
    Some(Manifest { parent: head, panel, code, raw })
}

fn head_path(name: &str) -> String {
    alloc::format!("{}/{}/HEAD", ROOT, name)
}

pub fn head(name: &str) -> Option<[u8; 32]> {
    let b = sysbox::read_blob(&head_path(name))?;
    from_hex32(String::from_utf8_lossy(&b).trim())
}

/// Record a version as the one in use. A pointer write, and nothing else moves.
pub fn adopt(name: &str, h: &[u8; 32]) -> bool {
    // Before the pointer moves, so the version being adopted can be returned to
    // later. Doing this at adoption rather than at rollback is what makes undo
    // possible at all: by the time somebody wants to go back, the bytes are
    // gone unless they were kept on the way past.
    preserve(name);
    let ok = sysbox::write_text(&head_path(name), &alloc::format!("{}\n", hex32(h)));
    if ok {
        ledger(name, "adopt", h);
    }
    ok
}

/// Go back to what this version replaced.
///
/// Costs a pointer write, because the parent node was never deleted -- that is
/// the whole return on naming things by hash.
pub fn rollback(name: &str) -> Option<[u8; 32]> {
    let h = head(name)?;
    let parent = Manifest::load(&h)?.parent?;
    let m = Manifest::load(&parent)?;
    // The files come back, not just the pointer.
    //
    // Moving `HEAD` alone was the first version of this, and it was worse than
    // doing nothing: the application would still have been the new one while
    // its recorded identity named the old, so every later hash comparison would
    // have been about a version that was not on disk. Anything claiming to undo
    // a change has to actually undo it.
    let (Some(panel), Some(code)) = (blob(&m.panel), blob(&m.code)) else {
        // Refused rather than half-done. A parent whose bytes were never kept
        // cannot be returned to, and saying so beats leaving content and
        // identity disagreeing.
        return None;
    };
    sysbox::write_blob(&alloc::format!("{}/{}/panel.ui", ROOT, name), panel);
    sysbox::write_blob(&alloc::format!("{}/{}/code.ai&xi", ROOT, name), code);
    sysbox::write_text(&head_path(name), &alloc::format!("{}\n", hex32(&parent)));
    ledger(name, "rollback", &parent);
    Some(parent)
}

/// Keep the bytes of what is on disk now, addressed by their own hash.
///
/// Called before adopting anything, so the version being replaced can still be
/// returned to. Write-if-absent, so a version adopted twice is stored once --
/// which is the whole reason the name is the hash.
pub fn preserve(name: &str) {
    for file in ["panel.ui", "code.ai&xi"] {
        let path = alloc::format!("{}/{}/{}", ROOT, name, file);
        if let Some(bytes) = sysbox::read_blob(&path) {
            let h = sha256::hash(&bytes);
            let at = alloc::format!("{}/{}", BLOBS, hex32(&h));
            if sysbox::read_blob(&at).is_none() {
                sysbox::write_blob(&at, bytes);
            }
        }
    }
}

fn blob(h: &[u8; 32]) -> Option<alloc::vec::Vec<u8>> {
    sysbox::read_blob(&alloc::format!("{}/{}", BLOBS, hex32(h)))
}

/// Has the operator approved this exact manifest?
///
/// By hash, so approval does not survive a byte changing anywhere in the
/// application or in what it asks for.
pub fn granted(h: &[u8; 32]) -> bool {
    let Some(b) = sysbox::read_blob(GRANTS) else {
        return false;
    };
    let want = hex32(h);
    String::from_utf8_lossy(&b).lines().any(|l| l.trim() == want)
}

/// Approve one manifest. Only the operator's command reaches this.
pub fn grant(name: &str, h: &[u8; 32]) -> bool {
    if granted(h) {
        return true;
    }
    let mut text = sysbox::read_blob(GRANTS)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    text.push_str(&hex32(h));
    text.push('\n');
    let ok = sysbox::write_text(GRANTS, &text);
    if ok {
        ledger(name, "grant", h);
    }
    ok
}

/// One line per decision, appended and never rewritten.
///
/// Text, and readable, for the reason `godel.rs` gives about its own: a history
/// nobody can read is a history nobody audits.
fn ledger(name: &str, what: &str, h: &[u8; 32]) {
    let mut text = sysbox::read_blob(LEDGER)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    text.push_str(what);
    text.push(' ');
    text.push_str(name);
    text.push(' ');
    text.push_str(&hex32(h));
    text.push('\n');
    sysbox::write_text(LEDGER, &text);
}

/// Every manifest in the store, newest last is not knowable -- they are
/// content-addressed and unordered, so this is for looking, not for history.
pub fn all() -> Vec<String> {
    sysbox::children(NODES)
}
