//! The user's packages, made to agree with a new system image.
//!
//! What `orange install` adds lives in the layer (`ns::UPPER`) over the
//! system image, and so does every package `orange upgrade` moved past the
//! image's version: pacman wrote the new files into the layer and **whited
//! out** the image's old ones (`ns::GONE`). A system update replaces the
//! image whole, and the new image may ship one of those packages too. Left
//! alone, either the layer's copy shadows the image's or -- worse -- the old
//! whiteouts hide the new image's files, and pacman finds a database that
//! names two versions of one package, or a library that is not there.
//!
//! So on the first boot of a new image, every package both hold is decided
//! by version, pacman's own rule (`vercmp`):
//!
//! - **the image's is as new or newer**: the layer's database entry and its
//!   files leave the layer, and the whiteouts over the image's copy of that
//!   package are lifted. The system's copy is used.
//! - **the layer's is newer** (the user upgraded past the image): the layer
//!   keeps it, and the image's database entry and any of its files the
//!   layer's version does not have are whited out, as pacman would have done
//!   upgrading it there.
//!
//! **`/etc` is never handed back**: a configuration file is the user's
//! whatever the package does, the rule pacman's `.pacnew` encodes.
//!
//! Nothing is deleted. What leaves the layer, and every whiteout lifted, is
//! moved to an **attic** (`/linux/attic`); every whiteout made is listed
//! there. If the trial fails and the old image comes back, the attic undoes
//! all three; once the new image is kept, it is dropped. Which of those a
//! boot does is a pure function of the slot record and the attic's note
//! (`act`), so a boot that dies anywhere in between is resolved by the next.
//!
//! The first version compared names only and handed every shared package to
//! the image. Driven on a disk whose layer had been `orange upgrade`d past an
//! older image, it moved 25 packages back and left pacman unable to start:
//! `liblz4.so.1` was whited out by the upgrade and nothing lifted it.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;

pub const ATTIC: &str = "/linux/attic";
/// Layer files moved aside, at their guest paths.
const ATTIC_UPPER: &str = "/linux/attic/upper";
/// Whiteouts lifted, at their guest paths.
const ATTIC_GONE: &str = "/linux/attic/gone";
/// Whiteouts made, one guest path per line.
const ATTIC_MADE: &str = "/linux/attic-made";
const NOTE: &str = "/linux/attic-for";
const LOCAL: &str = "/var/lib/pacman/local";

/// A package as its database entry describes it.
pub struct Pkg {
    pub name: String,
    pub version: String,
    /// The entry's directory name, `name-version`.
    pub entry: String,
    /// Paths relative to `/`, as pacman's `files` lists them; directories
    /// end in `/`.
    pub files: Vec<String>,
}

fn field(desc: &str, key: &str) -> Option<String> {
    let mut it = desc.lines();
    while let Some(l) = it.next() {
        if l.trim() == key {
            return it.next().map(|s| String::from(s.trim()));
        }
    }
    None
}

/// `%NAME%`, `%VERSION%` and `%FILES%` out of a database entry.
pub fn parse(entry: &str, desc: &str, files: &str) -> Option<Pkg> {
    let mut list = Vec::new();
    let mut on = false;
    for l in files.lines() {
        let l = l.trim();
        if l.starts_with('%') {
            on = l == "%FILES%";
            continue;
        }
        if on && !l.is_empty() {
            list.push(String::from(l));
        }
    }
    // The layer's database is written by guests, which run as root. A path
    // that climbs (`..`), is absolute, or an entry name that is not one
    // component would let a forged entry reach outside the layer -- the
    // namespace resolves `..` -- and move the kernel's own `/ai` or `/etc`
    // into the attic. Such an entry is not a package; it is refused whole.
    let bad = |f: &str| f.starts_with('/') || f.split('/').any(|c| c == ".." || c == ".");
    if entry.is_empty() || entry.contains('/') || entry == ".." || entry == "." || list.iter().any(|f| bad(f)) {
        return None;
    }
    Some(Pkg {
        name: field(desc, "%NAME%")?,
        version: field(desc, "%VERSION%").unwrap_or_default(),
        entry: String::from(entry),
        files: list,
    })
}

/// rpm's segment comparison, as libalpm has it: runs of digits compare as
/// numbers, runs of letters as strings, a number beats letters, and a
/// remaining letter run loses to nothing at all (so `1.0rc1` < `1.0`).
fn rpmvercmp(a: &[u8], b: &[u8]) -> Ordering {
    if a == b {
        return Ordering::Equal;
    }
    let (mut i, mut j) = (0usize, 0usize);
    loop {
        let (si, sj) = (i, j);
        while i < a.len() && !a[i].is_ascii_alphanumeric() {
            i += 1;
        }
        while j < b.len() && !b[j].is_ascii_alphanumeric() {
            j += 1;
        }
        if i >= a.len() || j >= b.len() {
            break;
        }
        // A longer separator is a later version (`1..2` against `1.2`).
        if i - si != j - sj {
            return (i - si).cmp(&(j - sj));
        }
        let num = a[i].is_ascii_digit();
        let (a0, b0) = (i, j);
        let class = |c: u8| if num { c.is_ascii_digit() } else { c.is_ascii_alphabetic() };
        while i < a.len() && class(a[i]) {
            i += 1;
        }
        while j < b.len() && class(b[j]) {
            j += 1;
        }
        let (x, y) = (&a[a0..i], &b[b0..j]);
        if y.is_empty() {
            return if num { Ordering::Greater } else { Ordering::Less };
        }
        let o = if num {
            let x = &x[x.iter().position(|&c| c != b'0').unwrap_or(x.len())..];
            let y = &y[y.iter().position(|&c| c != b'0').unwrap_or(y.len())..];
            x.len().cmp(&y.len()).then(x.cmp(y))
        } else {
            x.cmp(y)
        };
        if o != Ordering::Equal {
            return o;
        }
    }
    let (ra, rb) = (i >= a.len(), j >= b.len());
    if ra && rb {
        return Ordering::Equal;
    }
    if (ra && !b[j].is_ascii_alphabetic()) || (!ra && a[i].is_ascii_alphabetic()) {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// pacman's `vercmp`: epoch, then version, then release when both have one.
pub fn vercmp(a: &str, b: &str) -> Ordering {
    fn evr(s: &str) -> (&str, &str, Option<&str>) {
        let (e, rest) = match s.split_once(':') {
            Some((e, r)) if e.bytes().all(|c| c.is_ascii_digit()) => (e, r),
            _ => ("0", s),
        };
        match rest.rsplit_once('-') {
            Some((v, r)) => (e, v, Some(r)),
            None => (e, rest, None),
        }
    }
    let (ea, va, ra) = evr(a);
    let (eb, vb, rb) = evr(b);
    rpmvercmp(ea.as_bytes(), eb.as_bytes())
        .then_with(|| rpmvercmp(va.as_bytes(), vb.as_bytes()))
        .then_with(|| match (ra, rb) {
            (Some(x), Some(y)) => rpmvercmp(x.as_bytes(), y.as_bytes()),
            _ => Ordering::Equal,
        })
}

/// What to do about the packages both hold.
#[derive(Default, Debug, PartialEq)]
pub struct Plan {
    /// Guest paths to move out of the layer.
    pub take: Vec<String>,
    /// Guest paths whose whiteouts to lift.
    pub lift: Vec<String>,
    /// Guest paths to white out.
    pub hide: Vec<String>,
    /// The image's database entries to white out. Unlike a file, an entry
    /// may also have a copy in the layer -- pacman copies an entry up to
    /// touch it -- and that copy is moved aside as well.
    pub hide_entries: Vec<String>,
    /// Packages handed to the image, and kept in the layer.
    pub to_image: usize,
    pub kept: usize,
}

fn entry_paths(p: &Pkg) -> [String; 2] {
    let d = format!("{}/{}", LOCAL, p.entry);
    [format!("{}/desc", d), format!("{}/files", d)]
}

pub fn plan(layer: &[Pkg], image: &[Pkg]) -> Plan {
    let mut out = Plan::default();
    // Every path any layer package owns. A whiteout over one of them would
    // hide the layer's own copy too -- a lookup that finds a whiteout stops
    // there -- so nothing in this set is ever hidden, whichever package the
    // image thinks it belongs to.
    let owned: BTreeSet<&str> = layer.iter().flat_map(|p| p.files.iter().map(|s| s.as_str())).collect();
    for l in layer {
        let Some(i) = image.iter().find(|i| i.name == l.name) else { continue };
        if vercmp(&i.version, &l.version) != Ordering::Less {
            out.to_image += 1;
            out.take.push(format!("{}/{}", LOCAL, l.entry));
            out.take.extend(
                l.files.iter().filter(|f| !f.ends_with('/') && !f.starts_with("etc/")).map(|f| format!("/{}", f)),
            );
            // The image's own entry and files, in case an upgrade in the
            // layer whited them out.
            out.lift.push(format!("{}/{}", LOCAL, i.entry));
            out.lift.extend(entry_paths(i));
            out.lift.extend(i.files.iter().map(|f| format!("/{}", f.trim_end_matches('/'))));
        } else {
            out.kept += 1;
            if i.entry != l.entry {
                out.hide_entries.push(format!("{}/{}", LOCAL, i.entry));
            }
            out.hide.extend(
                i.files
                    .iter()
                    .filter(|f| !f.ends_with('/') && !f.starts_with("etc/") && !owned.contains(f.as_str()))
                    .map(|f| format!("/{}", f)),
            );
        }
    }
    out
}

/// What the boot should do with the attic, from the slot record and the note.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Act {
    Nothing,
    /// A new image is on trial and its layer has not been reconciled yet.
    Reconcile,
    /// The image the attic was made for was kept: forget the attic.
    Drop,
    /// The image the attic was made for is not the one running: undo it.
    Restore,
}

/// `note` is the digest the attic was made for; `digest` the record's
/// active image; `trial` whether that image is on trial.
pub fn act(note: Option<[u8; 32]>, digest: [u8; 32], trial: bool) -> Act {
    match (note, trial) {
        (None, true) => Act::Reconcile,
        (None, false) => Act::Nothing,
        (Some(n), _) if n != digest => Act::Restore,
        (Some(_), true) => Act::Nothing,
        (Some(_), false) => Act::Drop,
    }
}

// ---------------------------------------------------------------------------

fn upper(guest: &str) -> String {
    format!("{}{}", super::ns::UPPER, guest)
}

fn gone(guest: &str) -> String {
    format!("{}{}", super::ns::GONE, guest)
}

fn read_text(ns: &str) -> String {
    crate::sysbox::read_blob(ns).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default()
}

/// Every package entry in the layer.
fn layer_pkgs() -> Vec<Pkg> {
    let dir = upper(LOCAL);
    crate::sysbox::listing(&dir)
        .into_iter()
        .filter(|(_, is_dir, _)| *is_dir)
        .filter_map(|(e, _, _)| {
            parse(&e, &read_text(&format!("{}/{}/desc", dir, e)), &read_text(&format!("{}/{}/files", dir, e)))
        })
        .collect()
}

/// Every package the mounted image's own database holds, read off the image
/// alone -- not through the layer, which may hold an entry of the same name.
fn image_pkgs() -> Vec<Pkg> {
    super::disk::image_entries(LOCAL)
        .into_iter()
        .filter(|(_, d, _)| *d)
        .filter_map(|(e, _, _)| {
            let desc = super::disk::image_read(&format!("{}/{}/desc", LOCAL, e))?;
            let files = super::disk::image_read(&format!("{}/{}/files", LOCAL, e)).unwrap_or_default();
            parse(&e, &String::from_utf8_lossy(&desc), &String::from_utf8_lossy(&files))
        })
        .collect()
}

fn hex(d: &[u8; 32]) -> String {
    d.iter().map(|b| format!("{:02x}", b)).collect()
}

fn note() -> Option<[u8; 32]> {
    let t = read_text(NOTE);
    let t = t.trim();
    if t.len() != 64 || !t.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut o = [0u8; 32];
    for i in 0..32 {
        o[i] = u8::from_str_radix(&t[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(o)
}

/// Move a file, or a whole directory as one subtree, counting the files.
/// A directory moved file by file stays behind empty, and an empty package
/// entry is one pacman reads as a package with no files list.
fn move_path(from: &str, to: &str, n: &mut usize) {
    let files = if crate::sysbox::is_dir(from) { count(from) } else { 1 };
    if crate::sysbox::rename(from, to) {
        *n += files;
    }
}

fn count(dir: &str) -> usize {
    crate::sysbox::listing(dir)
        .into_iter()
        .map(|(name, d, _)| if d { count(&format!("{}/{}", dir, name)) } else { 1 })
        .sum()
}

/// Put every file under the attic directory `from` back under `to`, a file
/// at a time: `to` holds other files the attic never had.
fn move_back(from: &str, to: &str, n: &mut usize) {
    for (name, d, _) in crate::sysbox::listing(from) {
        let (f, t) = (format!("{}/{}", from, name), format!("{}/{}", to, name));
        if d && crate::sysbox::is_dir(&t) {
            move_back(&f, &t, n);
        } else {
            move_path(&f, &t, n);
        }
    }
}

fn forget() {
    crate::sysbox::detach(ATTIC);
    crate::sysbox::detach(ATTIC_MADE);
    crate::sysbox::detach(NOTE);
}

/// Settle the layer against the image this boot mounted. Answers a line for
/// the boot log when something moved.
pub fn at_boot() -> Option<String> {
    let r = crate::install::slot::current()?;
    super::disk::root_base()?;
    match act(note(), r.digest, r.trial.is_some()) {
        Act::Nothing => None,
        Act::Drop => {
            forget();
            let _ = crate::sysbox::commit_now();
            None
        }
        Act::Restore => {
            let mut n = 0;
            move_back(ATTIC_UPPER, super::ns::UPPER, &mut n);
            move_back(ATTIC_GONE, super::ns::GONE, &mut n);
            for p in read_text(ATTIC_MADE).lines().filter(|l| !l.is_empty()) {
                crate::sysbox::detach(&gone(p));
            }
            forget();
            let _ = crate::sysbox::commit_now();
            Some(format!("the previous system is back, and {} file(s) of what you installed with it", n))
        }
        Act::Reconcile => {
            // Only against the image on trial. When it would not mount, the
            // root is the old image and the trial is about to be undone;
            // reconciling the layer against the old image would hand it
            // packages for nothing and lean on the undo to put them back.
            let trial = crate::install::slot::slots().get(r.active as usize).map(|p| p.start_lba);
            if super::disk::root_base() != trial {
                return None;
            }
            let image = image_pkgs();
            let layer = layer_pkgs();
            let p = plan(&layer, &image);
            // A checkpoint first: the attic makes this reversible, and the
            // snapshot makes it reversible by hand too.
            crate::sysbox::snap_now();
            let mut moved = 0;
            for g in &p.take {
                move_path(&upper(g), &format!("{}{}", ATTIC_UPPER, g), &mut moved);
            }
            for g in &p.lift {
                if crate::sysbox::blob_len(&gone(g)).is_some()
                    && crate::sysbox::rename(&gone(g), &format!("{}{}", ATTIC_GONE, g))
                {
                    moved += 1;
                }
            }
            let mut made = String::new();
            let mut hidden = 0;
            for g in &p.hide_entries {
                if crate::sysbox::is_dir(&upper(g)) {
                    move_path(&upper(g), &format!("{}{}", ATTIC_UPPER, g), &mut moved);
                }
            }
            for g in p.hide_entries.iter().chain(p.hide.iter()) {
                if crate::sysbox::blob_len(&gone(g)).is_none() && crate::sysbox::write_blob(&gone(g), Vec::new()) {
                    made.push_str(g);
                    made.push('\n');
                    hidden += 1;
                }
            }
            crate::sysbox::write_blob(ATTIC_MADE, made.into_bytes());
            crate::sysbox::write_blob(NOTE, hex(&r.digest).into_bytes());
            // Committed before the shell can keep the trial: a kept record
            // on the ESP over a layer that was never saved reconciled is a
            // state no later boot repairs (the note is gone with it).
            let _ = crate::sysbox::commit_now();
            (p.to_image + p.kept > 0).then(|| {
                format!(
                    "{} package(s) you installed now come with the system and use its copy; {} you upgraded past it stay yours ({} file(s) set aside, {} of the system's hidden, until it is kept)",
                    p.to_image, p.kept, moved, hidden
                )
            })
        }
    }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    use Ordering::*;
    let pk = |name: &str, ver: &str, files: &[&str]| Pkg {
        name: name.into(),
        version: ver.into(),
        entry: format!("{}-{}", name, ver),
        files: files.iter().map(|s| String::from(*s)).collect(),
    };
    let layer = [
        pk("foo", "1.0-1", &["etc/", "etc/foo.conf", "usr/", "usr/bin/foo", "usr/lib/libfoo.so.1.0"]),
        pk("lz4", "1:1.10.0-2", &["usr/lib/liblz4.so.1.10.0"]),
        pk("bar", "2-1", &["usr/bin/bar"]),
    ];
    let image = [
        pk("foo", "1.1-1", &["usr/bin/foo", "usr/lib/libfoo.so.1.1"]),
        pk("lz4", "1:1.9.4-1", &["usr/lib/liblz4.so.1.9.4", "usr/lib/liblz4.so.1"]),
        pk("glibc", "2.40-1", &["usr/lib/libc.so.6"]),
    ];
    let p = plan(&layer, &image);
    let parsed = parse(
        "foo-1-1",
        "%NAME%\nfoo\n\n%VERSION%\n1-1\n",
        "%FILES%\netc/\netc/foo.conf\nusr/bin/foo\n\n%BACKUP%\netc/foo.conf\tabc\n",
    );
    let d = [7u8; 32];
    let other = [8u8; 32];
    alloc::vec![
        ("a forged entry whose files climb out of the layer is refused",
            parse("x-1-1", "%NAME%\nx\n", "%FILES%\n../../ai\n").is_none()
                && parse("x-1-1", "%NAME%\nx\n", "%FILES%\nusr/../../etc/x\n").is_none()
                && parse("../x", "%NAME%\nx\n", "%FILES%\nusr/bin/x\n").is_none()),
        ("vercmp orders numbers as numbers", vercmp("1.10-1", "1.9-1") == Greater && vercmp("1.0", "1.0") == Equal),
        ("vercmp puts an epoch first", vercmp("1:0.1-1", "9.9-1") == Greater),
        ("vercmp puts a release candidate before the release", vercmp("1.0rc1-1", "1.0-1") == Less),
        ("vercmp compares releases only when both have one", vercmp("1.0-2", "1.0-1") == Greater && vercmp("1.0", "1.0-5") == Equal),
        ("a database entry's name, version and files are read, and %BACKUP% is not a file", parsed.is_some_and(|q| q.name == "foo" && q.version == "1-1" && q.files.len() == 3)),
        ("a package the image ships at least as new leaves the layer", p.take.contains(&String::from("/var/lib/pacman/local/foo-1.0-1")) && p.take.contains(&String::from("/usr/bin/foo"))),
        ("and the whiteouts over the image's copy are lifted", p.lift.contains(&String::from("/usr/lib/libfoo.so.1.1")) && p.lift.contains(&String::from("/var/lib/pacman/local/foo-1.1-1"))),
        ("but its /etc stays, being the user's", !p.take.iter().any(|x| x.starts_with("/etc"))),
        ("a package upgraded past the image stays in the layer", !p.take.iter().any(|x| x.contains("lz4")) && p.kept == 1),
        ("and the image's older entry and files are hidden under it", p.hide_entries.contains(&String::from("/var/lib/pacman/local/lz4-1:1.9.4-1")) && p.hide.contains(&String::from("/usr/lib/liblz4.so.1"))),
        ("but never a path a layer package owns, which a whiteout would hide too", {
            let l2 = [pk("z", "2-1", &["usr/lib/libz.so"]), pk("y", "1-1", &["usr/include/z.h"])];
            let i2 = [pk("z", "1-1", &["usr/lib/libz.so", "usr/include/z.h", "usr/lib/libz.a"])];
            let q = plan(&l2, &i2);
            q.hide == [String::from("/usr/lib/libz.a")]
        }),
        ("a package only the user installed is not touched", !p.take.iter().chain(p.hide.iter()).any(|x| x.contains("bar"))),
        ("a fresh trial reconciles", act(None, d, true) == Act::Reconcile),
        ("a kept image with no attic does nothing", act(None, d, false) == Act::Nothing),
        ("an attic made for the image that was kept is dropped", act(Some(d), d, false) == Act::Drop),
        ("an attic made for an image that is not running is undone", act(Some(other), d, false) == Act::Restore && act(Some(other), d, true) == Act::Restore),
        ("and a trial already reconciled is not reconciled twice", act(Some(d), d, true) == Act::Nothing),
    ]
}
