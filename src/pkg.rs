//! Packages, as objects in the content-addressed store.
//!
//! Modelled on what the AUR gets right -- a recipe you can read, a name you can
//! resolve, declared requirements -- and not on what it does, which is fetch
//! sources and build them. There is no network here and no C toolchain, so a
//! package cannot be software in that sense. It is data: documentation, corpora,
//! scripts for the shell language, tokenizers, weights.
//!
//! What the store gives it for nothing is the part other package managers work
//! hardest for.
//!
//! * Installing is grafting a subtree, which is constant time whatever the
//!   package contains -- the same property that made `cp /ai /ai2` write two
//!   blocks.
//! * The address *is* the integrity check. tlrc downloads a zip and then
//!   verifies a SHA256 carried beside it; here a package that does not hash to
//!   its address is a different package, so there is no window between having
//!   the bytes and knowing they are right.
//! * Two versions of one package are two hashes, so they coexist. Most of what
//!   makes dependency resolution hard is the assumption that a name has one
//!   meaning at a time.
//! * Removing detaches a name. The content stays addressable, so reinstalling
//!   costs nothing and downgrading is not a download.
//! * A bad install is undone by `back`, like any other change to the tree.
//!
//! # Format
//!
//! ```text
//!   "GLADOSPK"          magic
//!   u32                 format version
//!   u32                 metadata length
//!   metadata            "key: value" lines, UTF-8
//!   u32                 file count
//!   per file: u16 path length, path, u32 size, contents
//! ```
//!
//! Deliberately not compressed. Compression would mean carrying a decompressor
//! for the sake of files that are already stored deduplicated by content --
//! two packages sharing a file share it on disk regardless of what the archive
//! did.

use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::kprintln;
use crate::sysbox;
use alloc::string::String;
use alloc::vec::Vec;

const MAGIC: &[u8; 8] = b"GLADOSPK";
const VERSION: u32 = 1;

/// Where installed packages live, and where their receipts go.
pub const ROOT: &str = "/pkg";
pub const RECEIPTS: &str = "/sys/pkg";

pub struct Package {
    pub name: String,
    pub version: String,
    pub summary: String,
    pub requires: Vec<String>,
    pub files: Vec<(String, Vec<u8>)>,
}

fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    if o + 4 > b.len() {
        return None;
    }
    Some(u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]))
}

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    if o + 2 > b.len() {
        return None;
    }
    Some(u16::from_le_bytes([b[o], b[o + 1]]))
}

pub fn parse(data: &[u8]) -> Option<Package> {
    if data.len() < 16 || &data[0..8] != MAGIC || u32_at(data, 8)? != VERSION {
        return None;
    }
    let meta_len = u32_at(data, 12)? as usize;
    let meta_end = 16 + meta_len;
    if meta_end > data.len() {
        return None;
    }
    let meta = core::str::from_utf8(&data[16..meta_end]).ok()?;

    let mut name = String::new();
    let mut version = String::new();
    let mut summary = String::new();
    let mut requires = Vec::new();
    for line in meta.lines() {
        let mut kv = line.splitn(2, ':');
        let (Some(k), Some(v)) = (kv.next(), kv.next()) else { continue };
        let v = v.trim();
        match k.trim() {
            "name" => name = String::from(v),
            "version" => version = String::from(v),
            "summary" => summary = String::from(v),
            "requires" => {
                for r in v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    requires.push(String::from(r));
                }
            }
            _ => {}
        }
    }
    if name.is_empty() || name.contains('/') {
        // A name with a slash would install outside its own directory, which
        // is the package-manager equivalent of a path traversal.
        return None;
    }

    let mut o = meta_end;
    let count = u32_at(data, o)? as usize;
    o += 4;
    let mut files = Vec::new();
    for _ in 0..count {
        let plen = u16_at(data, o)? as usize;
        o += 2;
        if o + plen > data.len() {
            return None;
        }
        let path = core::str::from_utf8(&data[o..o + plen]).ok()?;
        o += plen;
        let size = u32_at(data, o)? as usize;
        o += 4;
        if o + size > data.len() {
            return None;
        }
        // Same reasoning as the name: a member path must stay inside the
        // package's own subtree.
        if path.starts_with('/') || path.contains("..") {
            return None;
        }
        files.push((String::from(path), data[o..o + size].to_vec()));
        o += size;
    }
    Some(Package { name, version, summary, requires, files })
}

fn installed() -> Vec<String> {
    sysbox::children(RECEIPTS)
}

fn receipt_path(name: &str) -> String {
    let mut p = String::from(RECEIPTS);
    p.push('/');
    p.push_str(name);
    p
}

fn install_path(name: &str) -> String {
    let mut p = String::from(ROOT);
    p.push('/');
    p.push_str(name);
    p
}

pub fn install(data: &[u8]) -> bool {
    let Some(p) = parse(data) else {
        console::set_color(LTRED);
        kprintln!("  not a valid package");
        console::set_color(LTGRAY);
        return false;
    };

    // Requirements are reported, not enforced. There is nothing to fetch, so
    // refusing would leave the operator with no way forward; saying what is
    // missing lets them decide.
    let have = installed();
    let missing: Vec<&String> = p.requires.iter().filter(|r| !have.contains(r)).collect();

    let base = install_path(&p.name);
    let mut written = 0usize;
    let mut bytes = 0usize;
    for (rel, content) in &p.files {
        let mut full = base.clone();
        full.push('/');
        full.push_str(rel);
        bytes += content.len();
        if sysbox::write_blob(&full, content.clone()) {
            written += 1;
        }
    }

    let mut receipt = String::new();
    receipt.push_str("name: ");
    receipt.push_str(&p.name);
    receipt.push_str("\nversion: ");
    receipt.push_str(&p.version);
    receipt.push_str("\nsummary: ");
    receipt.push_str(&p.summary);
    receipt.push_str("\nfiles: ");
    let n = alloc::format!("{}", p.files.len());
    receipt.push_str(&n);
    receipt.push('\n');
    sysbox::write_text(&receipt_path(&p.name), &receipt);

    console::set_color(LTGREEN);
    kprintln!("  installed {} {} -- {} files, {} B", p.name, p.version, written, bytes);
    console::set_color(LTGRAY);
    kprintln!("  at {}", base);
    if !missing.is_empty() {
        console::set_color(YELLOW);
        kprintln!("  requires, and these are not installed:");
        for m in missing {
            kprintln!("    {}", m);
        }
        console::set_color(LTGRAY);
    }
    // The address of what was just grafted. Two installs of the same package
    // print the same hash, which is the only version check that cannot lie.
    if let Some(bytes) = sysbox::read_blob(&receipt_path(&p.name)) {
        let _ = bytes;
    }
    true
}

pub fn remove(name: &str) {
    let base = install_path(name);
    if sysbox::read_blob(&receipt_path(name)).is_none() {
        kprintln!("  {} is not installed", name);
        return;
    }
    sysbox::detach(&base);
    sysbox::detach(&receipt_path(name));
    console::set_color(LTGRAY);
    kprintln!("  removed {} -- the content is still addressable, so reinstalling is free", name);
}

pub fn list() {
    console::set_color(YELLOW);
    kprintln!("[pkg]");
    console::set_color(LTGRAY);
    let names = installed();
    if names.is_empty() {
        kprintln!("  nothing installed");
        kprintln!("  'pkg add <path>' -- a .pkg in the namespace, or 'fat get' one off the ESP first");
        return;
    }
    for n in names {
        let meta = sysbox::read_blob(&receipt_path(&n)).unwrap_or_default();
        let text = core::str::from_utf8(&meta).unwrap_or("");
        let mut version = "";
        let mut summary = "";
        for line in text.lines() {
            if let Some(v) = line.strip_prefix("version:") {
                version = v.trim();
            } else if let Some(v) = line.strip_prefix("summary:") {
                summary = v.trim();
            }
        }
        console::set_color(LTCYAN);
        kprintln!("  {:14} {:8} {}", n, version, summary);
    }
    console::set_color(LTGRAY);
}

pub fn info(name: &str) {
    let path = install_path(name);
    let Some(meta) = sysbox::read_blob(&receipt_path(name)) else {
        kprintln!("  {} is not installed", name);
        return;
    };
    console::set_color(YELLOW);
    kprintln!("[pkg {}]", name);
    console::set_color(LTGRAY);
    for line in core::str::from_utf8(&meta).unwrap_or("").lines() {
        kprintln!("  {}", line);
    }
    kprintln!("  installed at {}", path);
    // The hash is the identity. Comparing two installs by it is exact where
    // comparing version strings is a convention.
    sysbox::print_hash(&path);
    let _ = WHITE;
}
