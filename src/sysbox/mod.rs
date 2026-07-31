//! sysbox -- the system's action surface.
//!
//! Shaped like busybox: one table of small applets, each with a name, an
//! argument spec, a line of help, and a flag saying whether it can change
//! anything. The shape is borrowed because it is the right shape for something
//! that is not only typed at.
//!
//! GLaDOS is meant to be run by the model in `crate::ai`, and a model needs its
//! available actions described to it and its dangerous ones fenced off. A flat
//! enumerable table gives both: `APPLETS` renders directly into a prompt, and
//! `mutates` is the leash -- read-only applets can be handed over long before
//! the rest are. That is also why results are computed before they are
//! printed rather than being formatted on the fly; the printer is one consumer
//! of an applet's result and the model will be another.
//!
//! The namespace underneath is not a filesystem in any sense a POSIX program
//! would recognise -- see `tree`. The consequences show up in the applet list:
//! `cp` is constant time on any size, `same` compares whole subtrees in one
//! step, `rm` cannot destroy content, and a snapshot costs nothing. Those are
//! not features bolted on, they are what content addressing already implies.

pub mod tree;

use crate::gfx::console::{self, LTCYAN, LTGREEN, LTGRAY, LTRED, WHITE, YELLOW};
use crate::kprintln;
use crate::store::{self, cas};
use crate::sync::Racy;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use tree::{Node, Written};

pub struct Applet {
    pub name: &'static str,
    pub args: &'static str,
    pub help: &'static str,
    /// Whether this applet can change persistent content. The division is by
    /// effect, not by danger: `cd` moves the cursor around the namespace and
    /// counts as read-only, `rm` only detaches a name and still counts as
    /// mutating.
    pub mutates: bool,
}

pub const APPLETS: &[Applet] = &[
    Applet { name: "sysbox", args: "",             help: "this list", mutates: false },
    Applet { name: "ls",     args: "[path]",       help: "list a directory", mutates: false },
    Applet { name: "cd",     args: "[path|-]",     help: "change directory", mutates: false },
    Applet { name: "pwd",    args: "",             help: "print working directory", mutates: false },
    Applet { name: "tree",   args: "[path]",       help: "list recursively", mutates: false },
    Applet { name: "cat",    args: "<path>",       help: "print a file", mutates: false },
    Applet { name: "stat",   args: "<path>",       help: "address, kind, size", mutates: false },
    Applet { name: "hash",   args: "<path>",       help: "content address", mutates: false },
    Applet { name: "same",   args: "<a> <b>",      help: "compare two subtrees in one step", mutates: false },
    Applet { name: "du",     args: "[path]",       help: "apparent bytes vs bytes that exist", mutates: false },
    Applet { name: "find",   args: "<text>",       help: "search names and content", mutates: false },
    Applet { name: "diff",   args: "<seq> [seq]",  help: "compare snapshots, skipping equal subtrees", mutates: false },
    Applet { name: "snaps",  args: "",             help: "list snapshots", mutates: false },
    Applet { name: "fsck",   args: "",             help: "verify every stored object against its address", mutates: false },
    Applet { name: "mkdir",  args: "<path>",       help: "create a directory and its parents", mutates: true },
    Applet { name: "write",  args: "<path> <text>", help: "write a file", mutates: true },
    Applet { name: "rm",     args: "<path>",       help: "detach a name; content survives", mutates: true },
    Applet { name: "mv",     args: "<a> <b>",      help: "rename", mutates: true },
    Applet { name: "cp",     args: "<a> <b>",      help: "copy; constant time at any size", mutates: true },
    Applet { name: "snap",   args: "",             help: "commit the working tree as a snapshot", mutates: true },
    Applet { name: "back",   args: "<seq>",        help: "load a past snapshot into the working tree", mutates: true },
];

pub fn is_applet(name: &str) -> bool {
    APPLETS.iter().any(|a| a.name == name)
}

// --- state --------------------------------------------------------------

pub struct Sysbox {
    root: Node,
    cwd: Vec<String>,
    prev: Vec<String>,
    /// What is already on disk, keyed by content address. Persists across
    /// snapshots so an unchanged subtree is never written twice.
    written: Written,
}

static BOX: Racy<Option<Sysbox>> = Racy::new(None);

/// Build the initial namespace.
///
/// It exists in RAM whether or not there is a disk. A store does not create the
/// namespace, it only lets it outlive a reboot -- which is the right way round
/// for a system that intends to treat memory and storage as one thing.
pub fn init() {
    let mut sb = Sysbox {
        root: Node::empty_dir(),
        cwd: Vec::new(),
        prev: Vec::new(),
        written: Written::default(),
    };
    let _ = tree::put(&mut sb.root, &path_of("/sys/readme"), Node::Blob(
        b"sysbox: every object is its own address.\n\
          cp is free, rm keeps the bytes, snapshots cost nothing.\n\
          type 'sysbox' for the applet list.\n".to_vec()));
    let _ = tree::put(&mut sb.root, &path_of("/tmp/.keep"), Node::Blob(Vec::new()));
    let _ = tree::put(&mut sb.root, &path_of("/ai/.keep"), Node::Blob(Vec::new()));
    unsafe { *BOX.get() = Some(sb) };
}

/// Adopt the newest snapshot as the working tree, if there is one.
///
/// This is the small end of the wedge the whole store design is pointed at: a
/// reboot should not be an event the namespace can perceive. Today it restores
/// what was explicitly snapshotted; the intent is that it eventually restores
/// live memory and `snap` stops being something anyone types.
pub fn restore_latest() {
    let seq = match store::with(|st| st.sb.seq) {
        None => return,
        Some(s) => s,
    };
    let loaded = store::with(|st| root_of(st, seq)).flatten();
    match loaded {
        None => {}
        Some(node) => {
            let files = tree::stats(&node).files;
            with(|s| {
                s.root = node;
                // Everything already on disk is already written. Without
                // seeding this, the first snap after a boot would rewrite the
                // entire tree it just finished reading.
                let _ = store::with(|st| index_written(st, s));
            });
            console::set_color(LTGREEN);
            kprintln!("  restored snapshot {} ({} files)", seq, files);
            console::set_color(LTGRAY);
        }
    }
}

/// Rebuild the written-set by re-walking what is on disk, so a restored tree
/// knows its own contents are already stored.
fn index_written(st: &cas::Store, s: &mut Sysbox) -> Option<()> {
    let m = st.read_manifest(&st.sb.root).ok()?;
    let e = m.entries.first()?;
    index_walk(st, &e.chunk, tree::KIND_DIR, 0, &mut s.written);
    Some(())
}

fn index_walk(st: &cas::Store, r: &cas::ChunkRef, kind: u8, depth: usize, w: &mut Written) {
    if depth > tree::MAX_DEPTH {
        return;
    }
    let raw = match st.get(r) {
        Ok(v) => v,
        Err(_) => return,
    };
    if kind == tree::KIND_BLOB {
        // A blob's chunk address IS its content address, so the ChunkRef can be
        // memoised directly.
        w.insert(r.hash, *r);
        return;
    }
    let entries = match tree::decode_dir(&raw) {
        Some(v) => v,
        None => return,
    };
    let mut rebuilt = Vec::new();
    for (name, k, cr) in &entries {
        index_walk(st, cr, *k, depth + 1, w);
        rebuilt.push((name.clone(), *k, *cr));
    }
    // A directory's chunk address covers block locations, so it is not the
    // content address `content_hash` computes. Recover the content address the
    // only way available here: rebuild the node and hash it.
    if let Ok(node) = read_node(st, r, tree::KIND_DIR, depth) {
        w.insert(tree::content_hash(&node), *r);
    }
}

fn with<R>(f: impl FnOnce(&mut Sysbox) -> R) -> Option<R> {
    unsafe { BOX.get().as_mut().map(f) }
}

fn path_of(s: &str) -> Vec<String> {
    parse(&[], s)
}

/// Resolve a path against a working directory. Absolute if it starts with `/`.
fn parse(cwd: &[String], s: &str) -> Vec<String> {
    let mut out: Vec<String> = if s.starts_with('/') { Vec::new() } else { cwd.to_vec() };
    for part in s.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            p => out.push(p.to_string()),
        }
    }
    out
}

fn show(path: &[String]) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    let mut s = String::new();
    for p in path {
        s.push('/');
        s.push_str(p);
    }
    s
}

fn err(msg: &str) {
    console::set_color(LTRED);
    kprintln!("  {}", msg);
    console::set_color(LTGRAY);
}

// --- dispatch -----------------------------------------------------------

/// Returns false if `cmd` is not an applet, so the shell can fall through to
/// the language interpreter.
pub fn dispatch(cmd: &str, rest: &str) -> bool {
    if !is_applet(cmd) {
        return false;
    }
    if !is_ready() {
        err("sysbox is not initialised");
        return true;
    }
    let a1 = rest.split_whitespace().next().unwrap_or("");
    let a2 = rest.split_whitespace().nth(1).unwrap_or("");

    match cmd {
        "sysbox" => list_applets(),
        "pwd" => {
            with(|s| kprintln!("  {}", show(&s.cwd)));
        }
        "ls" => cmd_ls(a1),
        "cd" => cmd_cd(a1),
        "tree" => cmd_tree(a1),
        "cat" => cmd_cat(a1),
        "stat" => cmd_stat(a1),
        "hash" => cmd_hash(a1),
        "same" => cmd_same(a1, a2),
        "du" => cmd_du(a1),
        "find" => cmd_find(rest.trim()),
        "mkdir" => cmd_mkdir(a1),
        "write" => cmd_write(a1, rest),
        "rm" => cmd_rm(a1),
        "mv" => cmd_move(a1, a2, true),
        "cp" => cmd_move(a1, a2, false),
        "snap" => cmd_snap(),
        "snaps" => cmd_snaps(),
        "back" => cmd_back(a1),
        "diff" => cmd_diff(a1, a2),
        "fsck" => cmd_fsck(),
        _ => {}
    }
    true
}

pub fn is_ready() -> bool {
    unsafe { BOX.get().is_some() }
}

// --- programmatic access ------------------------------------------------
//
// The applets above exist to be typed at. These exist so the rest of the
// system can use the namespace as storage -- the training corpus lives in it,
// which is what makes a training set a snapshottable object rather than a file
// somebody has to remember to keep.

pub fn write_text(path: &str, text: &str) -> bool {
    write_blob(path, text.as_bytes().to_vec())
}

pub fn write_blob(path: &str, data: Vec<u8>) -> bool {
    with(|s| {
        let p = parse(&s.cwd, path);
        tree::put(&mut s.root, &p, Node::Blob(data)).is_ok()
    })
    .unwrap_or(false)
}

pub fn read_blob(path: &str) -> Option<Vec<u8>> {
    with(|s| {
        let p = parse(&s.cwd, path);
        match tree::resolve(&s.root, &p) {
            Some(Node::Blob(b)) => Some(b.clone()),
            _ => None,
        }
    })
    .flatten()
}

/// Entry names directly under `path`, in sorted order. Empty if it is missing
/// or is a file.
pub fn children(path: &str) -> Vec<String> {
    with(|s| {
        let p = parse(&s.cwd, path);
        match tree::resolve(&s.root, &p) {
            Some(Node::Dir(es)) => es.iter().map(|(k, _)| k.clone()).collect(),
            _ => Vec::new(),
        }
    })
    .unwrap_or_default()
}

// --- selftest -----------------------------------------------------------

fn check(what: &str, pass: bool) -> bool {
    if pass {
        console::set_color(LTGREEN);
        kprintln!("  ok   {}", what);
    } else {
        console::set_color(LTRED);
        kprintln!("  FAIL {}", what);
    }
    console::set_color(LTGRAY);
    pass
}

/// Assert the properties every applet above is quietly relying on.
///
/// These are cheap to check and expensive to get wrong: if addresses stop
/// being a faithful summary of content, `cp` starts aliasing unrelated files,
/// `diff` starts skipping real changes, and dedup starts discarding data. None
/// of those announce themselves.
pub fn selftest() -> bool {
    fn tree_with(path: &str, body: &[u8]) -> Node {
        let mut n = Node::empty_dir();
        let _ = tree::put(&mut n, &path_of(path), Node::Blob(body.to_vec()));
        n
    }

    let mut ok = true;

    let a = tree_with("/x/y/z", b"hello");
    let b = tree_with("/x/y/z", b"hello");
    let c = tree_with("/x/y/z", b"hellp");
    ok &= check(
        "trees built separately from equal content share an address",
        tree::content_hash(&a) == tree::content_hash(&b),
    );
    // Merkle propagation: a single byte three levels down has to reach the top,
    // or a snapshot can miss a change.
    ok &= check(
        "one changed byte three levels down changes the root",
        tree::content_hash(&a) != tree::content_hash(&c),
    );

    // Without length-prefixing the name before hashing it, names and the fields
    // that follow them can be re-cut into a different tree with the same bytes.
    let mut n1 = Node::empty_dir();
    let _ = tree::put(&mut n1, &path_of("/ab"), Node::Blob(Vec::new()));
    let mut n2 = Node::empty_dir();
    let _ = tree::put(&mut n2, &path_of("/a"), Node::Blob(Vec::new()));
    ok &= check(
        "distinct names cannot share an address",
        tree::content_hash(&n1) != tree::content_hash(&n2),
    );

    // The claim behind `cp` being constant time.
    let mut d = tree_with("/x/y/z", b"hello");
    let mut copy_ok = false;
    if let Some(sub) = tree::resolve(&d, &path_of("/x")) {
        let want = tree::content_hash(sub);
        let dup = tree::clone_node(sub);
        if tree::put(&mut d, &path_of("/copy"), dup).is_ok() {
            if let Some(made) = tree::resolve(&d, &path_of("/copy")) {
                copy_ok = tree::content_hash(made) == want;
            }
        }
    }
    ok &= check("a copied subtree has the original's address", copy_ok);

    // Detaching a name must leave no trace in the address, otherwise history
    // would record edits that did not happen.
    let mut e = tree_with("/x/y/z", b"hello");
    let before = tree::content_hash(&e);
    let _ = tree::put(&mut e, &path_of("/scratch"), Node::Blob(b"temp".to_vec()));
    let moved = tree::content_hash(&e) != before;
    tree::remove(&mut e, &path_of("/scratch"));
    ok &= check(
        "add then remove returns to the original address",
        moved && tree::content_hash(&e) == before,
    );

    ok
}

fn list_applets() {
    console::set_color(YELLOW);
    kprintln!("sysbox applets");
    console::set_color(WHITE);
    for a in APPLETS {
        let mark = if a.mutates { "*" } else { " " };
        let mut label = String::from(a.name);
        if !a.args.is_empty() {
            label.push(' ');
            label.push_str(a.args);
        }
        kprintln!("  {}{:22} {}", mark, label, a.help);
    }
    console::set_color(YELLOW);
    kprintln!("\n  * mutates content. read-only applets are safe to hand to the model first.");
    console::set_color(LTGRAY);
}

// --- read-only applets --------------------------------------------------

fn cmd_ls(arg: &str) {
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::resolve(&s.root, &p) {
            None => err("no such path"),
            Some(Node::Blob(b)) => {
                kprintln!("  {:>10}  {}", b.len(), show(&p));
            }
            Some(Node::Dir(es)) => {
                if es.is_empty() {
                    console::set_color(LTGRAY);
                    kprintln!("  (empty)");
                    return;
                }
                for (name, child) in es {
                    let h = tree::content_hash(child);
                    let hx = tree::short(&h);
                    let hx = core::str::from_utf8(&hx).unwrap_or("?");
                    match child {
                        Node::Dir(inner) => {
                            console::set_color(LTCYAN);
                            kprintln!("  {}  {:>8}  {}/", hx, inner.len(), name);
                        }
                        Node::Blob(b) => {
                            console::set_color(WHITE);
                            kprintln!("  {}  {:>8}  {}", hx, b.len(), name);
                        }
                    }
                }
                console::set_color(LTGRAY);
            }
        }
    });
}

fn cmd_cd(arg: &str) {
    with(|s| {
        let target = if arg == "-" {
            s.prev.clone()
        } else if arg.is_empty() {
            Vec::new()
        } else {
            parse(&s.cwd, arg)
        };
        match tree::resolve(&s.root, &target) {
            Some(Node::Dir(_)) => {
                s.prev = core::mem::replace(&mut s.cwd, target);
                kprintln!("  {}", show(&s.cwd));
            }
            Some(Node::Blob(_)) => err("not a directory"),
            None => err("no such path"),
        }
    });
}

fn cmd_tree(arg: &str) {
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::resolve(&s.root, &p) {
            None => err("no such path"),
            Some(n) => {
                kprintln!("  {}", show(&p));
                walk_tree(n, 1);
                console::set_color(LTGRAY);
            }
        }
    });
}

fn walk_tree(n: &Node, depth: usize) {
    if depth > tree::MAX_DEPTH {
        return;
    }
    if let Node::Dir(es) = n {
        for (name, child) in es {
            let mut pad = String::new();
            for _ in 0..depth {
                pad.push_str("  ");
            }
            match child {
                Node::Dir(_) => {
                    console::set_color(LTCYAN);
                    kprintln!("  {}{}/", pad, name);
                    walk_tree(child, depth + 1);
                }
                Node::Blob(b) => {
                    console::set_color(WHITE);
                    kprintln!("  {}{}  ({} B)", pad, name, b.len());
                }
            }
        }
    }
}

fn cmd_cat(arg: &str) {
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::resolve(&s.root, &p) {
            Some(Node::Blob(b)) => {
                for line in b.split(|&c| c == b'\n') {
                    kprintln!("  {}", core::str::from_utf8(line).unwrap_or("<binary>"));
                }
            }
            Some(Node::Dir(_)) => err("that is a directory"),
            None => err("no such path"),
        }
    });
}

fn cmd_stat(arg: &str) {
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::resolve(&s.root, &p) {
            None => err("no such path"),
            Some(n) => {
                let h = tree::content_hash(n);
                let hx = tree::short(&h);
                let st = tree::stats(n);
                kprintln!("  path     {}", show(&p));
                kprintln!("  kind     {}", if n.is_dir() { "directory" } else { "file" });
                kprintln!("  address  {}", core::str::from_utf8(&hx).unwrap_or("?"));
                if n.is_dir() {
                    kprintln!("  contains {} files, {} dirs", st.files, st.dirs - 1);
                }
                kprintln!("  apparent {} B", st.apparent);
                kprintln!("  unique   {} B", st.unique);
                let stored = with_store_lookup(&h);
                kprintln!("  on disk  {}", if stored { "yes" } else { "not since last snap" });
            }
        }
    });
}

fn with_store_lookup(h: &tree::Hash) -> bool {
    unsafe { BOX.get().as_ref().map(|s| s.written.get(h).is_some()).unwrap_or(false) }
}

fn cmd_hash(arg: &str) {
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::resolve(&s.root, &p) {
            None => err("no such path"),
            Some(n) => {
                let h = tree::content_hash(n);
                let mut line = String::new();
                for b in h.iter() {
                    line.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('?'));
                    line.push(char::from_digit((b & 0xF) as u32, 16).unwrap_or('?'));
                }
                kprintln!("  {}", line);
            }
        }
    });
}

fn cmd_same(a: &str, b: &str) {
    if a.is_empty() || b.is_empty() {
        err("usage: same <a> <b>");
        return;
    }
    with(|s| {
        let pa = parse(&s.cwd, a);
        let pb = parse(&s.cwd, b);
        match (tree::resolve(&s.root, &pa), tree::resolve(&s.root, &pb)) {
            (Some(na), Some(nb)) => {
                // One comparison, whatever is underneath. A thousand files
                // deep or a single byte, this costs the same.
                if tree::content_hash(na) == tree::content_hash(nb) {
                    console::set_color(LTGREEN);
                    kprintln!("  identical");
                } else {
                    console::set_color(YELLOW);
                    kprintln!("  different");
                }
                console::set_color(LTGRAY);
            }
            _ => err("no such path"),
        }
    });
}

fn cmd_du(arg: &str) {
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::resolve(&s.root, &p) {
            None => err("no such path"),
            Some(n) => {
                let st = tree::stats(n);
                kprintln!("  {} files in {} directories", st.files, st.dirs);
                kprintln!("  apparent  {} B", st.apparent);
                kprintln!("  unique    {} B", st.unique);
                let saved = st.apparent.saturating_sub(st.unique);
                if saved > 0 {
                    console::set_color(LTGREEN);
                    kprintln!("  shared    {} B never had to be stored twice", saved);
                    console::set_color(LTGRAY);
                }
            }
        }
    });
}

fn cmd_find(needle: &str) {
    if needle.is_empty() {
        err("usage: find <text>");
        return;
    }
    with(|s| {
        let mut hits = 0u32;
        find_walk(&s.root, &mut Vec::new(), needle, &mut hits);
        console::set_color(LTGRAY);
        kprintln!("  {} match(es)", hits);
    });
}

fn find_walk(n: &Node, at: &mut Vec<String>, needle: &str, hits: &mut u32) {
    match n {
        Node::Blob(b) => {
            let name_hit = at.last().map(|s| s.contains(needle)).unwrap_or(false);
            let body_hit = core::str::from_utf8(b).map(|t| t.contains(needle)).unwrap_or(false);
            if name_hit || body_hit {
                *hits += 1;
                console::set_color(WHITE);
                kprintln!("  {}  {}", show(at), if body_hit { "(content)" } else { "(name)" });
            }
        }
        Node::Dir(es) => {
            for (name, child) in es {
                at.push(name.clone());
                find_walk(child, at, needle, hits);
                at.pop();
            }
        }
    }
}

// --- mutating applets ---------------------------------------------------

fn cmd_mkdir(arg: &str) {
    if arg.is_empty() {
        err("usage: mkdir <path>");
        return;
    }
    with(|s| {
        let p = parse(&s.cwd, arg);
        if tree::resolve(&s.root, &p).is_some() {
            err("already exists");
            return;
        }
        match tree::put(&mut s.root, &p, Node::empty_dir()) {
            Ok(()) => kprintln!("  {}", show(&p)),
            Err(tree::PutError::TooDeep) => err("too deep"),
            Err(tree::PutError::NotADirectory) => err("a component of that path is a file"),
            Err(tree::PutError::Empty) => err("cannot create the root"),
        }
    });
}

fn cmd_write(arg: &str, rest: &str) {
    if arg.is_empty() {
        err("usage: write <path> <text>");
        return;
    }
    let text = rest[arg.len().min(rest.len())..].trim_start();
    with(|s| {
        let p = parse(&s.cwd, arg);
        let mut data = text.as_bytes().to_vec();
        data.push(b'\n');
        let n = data.len();
        match tree::put(&mut s.root, &p, Node::Blob(data)) {
            Ok(()) => kprintln!("  {}  {} B", show(&p), n),
            Err(tree::PutError::TooDeep) => err("too deep"),
            Err(tree::PutError::NotADirectory) => err("a component of that path is a file"),
            Err(tree::PutError::Empty) => err("cannot write the root"),
        }
    });
}

fn cmd_rm(arg: &str) {
    if arg.is_empty() {
        err("usage: rm <path>");
        return;
    }
    with(|s| {
        let p = parse(&s.cwd, arg);
        match tree::remove(&mut s.root, &p) {
            Some(n) => {
                let h = tree::content_hash(&n);
                let hx = tree::short(&h);
                console::set_color(LTGRAY);
                // Worth saying every time. This is the difference between this
                // and a filesystem, and it is easy to forget mid-session.
                kprintln!(
                    "  detached {}. content {} is unharmed.",
                    show(&p),
                    core::str::from_utf8(&hx).unwrap_or("?")
                );
            }
            None => err("no such path"),
        }
    });
}

/// `mv` and `cp` differ by one line. Both are address moves -- neither reads or
/// copies content, whatever the size of what is being moved.
fn cmd_move(a: &str, b: &str, detach: bool) {
    if a.is_empty() || b.is_empty() {
        err("usage: <a> <b>");
        return;
    }
    with(|s| {
        let pa = parse(&s.cwd, a);
        let pb = parse(&s.cwd, b);
        let node = match tree::resolve(&s.root, &pa) {
            None => {
                err("no such path");
                return;
            }
            // The in-RAM tree owns its nodes, so this is a real copy in memory.
            // On disk it is not: both names will resolve to one address, and
            // `snap` writes nothing for the second.
            Some(n) => tree::clone_node(n),
        };
        if detach {
            tree::remove(&mut s.root, &pa);
        }
        match tree::put(&mut s.root, &pb, node) {
            Ok(()) => kprintln!("  {} -> {}", show(&pa), show(&pb)),
            Err(_) => err("could not place that path"),
        }
    });
}

// --- persistence --------------------------------------------------------

fn cmd_snap() {
    let ok = with(|s| {
        let res = store::with(|st| {
            let before = st.sb.alloc_next;
            let root = write_node(st, &s.root, &mut s.written)?;
            let mut name = [0u8; cas::NAME_LEN];
            name[..4].copy_from_slice(b"root");
            let entry = cas::Entry { name, chunk: root };
            let m = st.commit(core::slice::from_ref(&entry))?;
            Ok::<_, cas::Error>((st.sb.seq, m, st.sb.alloc_next - before))
        });
        match res {
            None => {
                err("no store mounted -- 'store init' first");
                false
            }
            Some(Err(e)) => {
                report_store_error(e);
                false
            }
            Some(Ok((seq, m, blocks))) => {
                let hx = crate::store::sha256::short_hex(&m.hash);
                console::set_color(LTGREEN);
                kprintln!(
                    "  snapshot {}  root {}  {} block(s) written",
                    seq,
                    core::str::from_utf8(&hx).unwrap_or("?"),
                    blocks
                );
                console::set_color(LTGRAY);
                // The interesting number. A second snapshot of an unchanged
                // tree writes one block: the manifest. Everything else is
                // already stored under the same address.
                true
            }
        }
    });
    let _ = ok;
}

fn report_store_error(e: cas::Error) {
    match e {
        cas::Error::Unsafe => err("writes are locked -- 'store unlock' first"),
        cas::Error::Full => err("the store region is full"),
        cas::Error::NoDevice => err("no block device"),
        cas::Error::HashMismatch => err("an object failed to verify against its address"),
        other => {
            console::set_color(LTRED);
            kprintln!("  store error: {:?}", other);
            console::set_color(LTGRAY);
        }
    }
}

fn write_node(
    st: &mut cas::Store,
    n: &Node,
    w: &mut Written,
) -> Result<cas::ChunkRef, cas::Error> {
    let h = tree::content_hash(n);
    if let Some(r) = w.get(&h) {
        return Ok(r);
    }
    let r = match n {
        Node::Blob(b) => st.put(b)?,
        Node::Dir(es) => {
            let mut encoded = Vec::new();
            for (name, child) in es {
                let cr = write_node(st, child, w)?;
                encoded.push((name.clone(), child.kind(), cr));
            }
            st.put(&tree::encode_dir(&encoded))?
        }
    };
    w.insert(h, r);
    Ok(r)
}

fn read_node(
    st: &cas::Store,
    r: &cas::ChunkRef,
    kind: u8,
    depth: usize,
) -> Result<Node, cas::Error> {
    if depth > tree::MAX_DEPTH {
        return Err(cas::Error::Corrupt);
    }
    let raw = st.get(r)?;
    if kind == tree::KIND_BLOB {
        return Ok(Node::Blob(raw));
    }
    let entries = tree::decode_dir(&raw).ok_or(cas::Error::Corrupt)?;
    let mut out = Vec::new();
    for (name, k, cr) in entries {
        out.push((name, read_node(st, &cr, k, depth + 1)?));
    }
    Ok(Node::Dir(out))
}

fn cmd_snaps() {
    let res = store::with(|st| {
        let mut out = Vec::new();
        let mut cur = st.sb.root;
        let mut guard = 0;
        while !cur.is_none() && guard < 256 {
            match st.read_manifest(&cur) {
                Ok(m) => {
                    out.push((m.seq, cur.hash, m.entries.len()));
                    cur = m.prev;
                }
                Err(_) => break,
            }
            guard += 1;
        }
        out
    });
    match res {
        None => err("no store mounted"),
        Some(list) if list.is_empty() => {
            console::set_color(LTGRAY);
            kprintln!("  no snapshots yet -- 'snap' to take one");
        }
        Some(list) => {
            console::set_color(YELLOW);
            kprintln!("  {:>5}  {:16}  entries", "seq", "root");
            console::set_color(WHITE);
            for (seq, hash, n) in list {
                let hx = crate::store::sha256::short_hex(&hash);
                kprintln!("  {:>5}  {}  {}", seq, core::str::from_utf8(&hx).unwrap_or("?"), n);
            }
            console::set_color(LTGRAY);
        }
    }
}

/// Find the manifest with a given sequence number by walking back from the
/// current root.
fn manifest_at(st: &cas::Store, seq: u64) -> Option<cas::ChunkRef> {
    let mut cur = st.sb.root;
    let mut guard = 0;
    while !cur.is_none() && guard < 256 {
        let m = st.read_manifest(&cur).ok()?;
        if m.seq == seq {
            return Some(cur);
        }
        cur = m.prev;
        guard += 1;
    }
    None
}

fn root_of(st: &cas::Store, seq: u64) -> Option<Node> {
    let mref = manifest_at(st, seq)?;
    let m = st.read_manifest(&mref).ok()?;
    let e = m.entries.first()?;
    read_node(st, &e.chunk, tree::KIND_DIR, 0).ok()
}

fn cmd_back(arg: &str) {
    let seq: u64 = match arg.parse() {
        Ok(v) => v,
        Err(_) => {
            err("usage: back <seq>  (see 'snaps')");
            return;
        }
    };
    let loaded = store::with(|st| root_of(st, seq));
    match loaded {
        None => err("no store mounted"),
        Some(None) => err("no such snapshot, or it did not verify"),
        Some(Some(node)) => {
            with(|s| {
                s.root = node;
                s.cwd.clear();
                s.prev.clear();
            });
            console::set_color(LTGREEN);
            kprintln!("  working tree is now snapshot {}", seq);
            console::set_color(LTGRAY);
            // Deliberately does not move the store's root. Nothing has been
            // rewritten and nothing has been lost: the newer snapshots are
            // still there, and 'snap' from here appends rather than
            // overwrites, so history branches instead of being edited.
            kprintln!("  the store still points at its newest snapshot; 'snap' to branch from here");
        }
    }
}

fn cmd_diff(a: &str, b: &str) {
    let sa: u64 = match a.parse() {
        Ok(v) => v,
        Err(_) => {
            err("usage: diff <seq> [seq]   (omit the second to compare against the working tree)");
            return;
        }
    };
    let left = match store::with(|st| root_of(st, sa)) {
        None => {
            err("no store mounted");
            return;
        }
        Some(None) => {
            err("no such snapshot");
            return;
        }
        Some(Some(n)) => n,
    };

    let right = if b.is_empty() {
        with(|s| tree::clone_node(&s.root))
    } else {
        match b.parse::<u64>() {
            Err(_) => {
                err("second argument must be a sequence number");
                return;
            }
            Ok(sb) => match store::with(|st| root_of(st, sb)) {
                Some(Some(n)) => Some(n),
                _ => {
                    err("no such snapshot");
                    return;
                }
            },
        }
    };
    let right = match right {
        Some(n) => n,
        None => return,
    };

    let mut changes = 0u32;
    diff_walk(&left, &right, &mut Vec::new(), &mut changes);
    console::set_color(LTGRAY);
    if changes == 0 {
        kprintln!("  identical");
    } else {
        kprintln!("  {} change(s)", changes);
    }
}

/// The whole point of hashing the tree: if two directories have the same
/// address, everything below them is equal and there is nothing to walk. A
/// diff therefore costs time proportional to what changed rather than to how
/// much exists.
fn diff_walk(a: &Node, b: &Node, at: &mut Vec<String>, changes: &mut u32) {
    if tree::content_hash(a) == tree::content_hash(b) {
        return;
    }
    match (a, b) {
        (Node::Dir(ea), Node::Dir(eb)) => {
            let mut i = 0;
            let mut j = 0;
            while i < ea.len() || j < eb.len() {
                let ord = match (ea.get(i), eb.get(j)) {
                    (Some((ka, _)), Some((kb, _))) => ka.cmp(kb),
                    (Some(_), None) => core::cmp::Ordering::Less,
                    (None, Some(_)) => core::cmp::Ordering::Greater,
                    (None, None) => break,
                };
                match ord {
                    core::cmp::Ordering::Equal => {
                        at.push(ea[i].0.clone());
                        diff_walk(&ea[i].1, &eb[j].1, at, changes);
                        at.pop();
                        i += 1;
                        j += 1;
                    }
                    core::cmp::Ordering::Less => {
                        at.push(ea[i].0.clone());
                        mark("-", at, changes);
                        at.pop();
                        i += 1;
                    }
                    core::cmp::Ordering::Greater => {
                        at.push(eb[j].0.clone());
                        mark("+", at, changes);
                        at.pop();
                        j += 1;
                    }
                }
            }
        }
        _ => mark("~", at, changes),
    }
}

fn mark(sign: &str, at: &[String], changes: &mut u32) {
    *changes += 1;
    match sign {
        "-" => console::set_color(LTRED),
        "+" => console::set_color(LTGREEN),
        _ => console::set_color(YELLOW),
    }
    kprintln!("  {} {}", sign, show(at));
    console::set_color(LTGRAY);
}

fn cmd_fsck() {
    let res = store::with(|st| {
        if st.sb.root.is_none() {
            return Ok((0u64, 0u64));
        }
        let m = st.read_manifest(&st.sb.root)?;
        let e = match m.entries.first() {
            Some(e) => e,
            None => return Ok((0, 0)),
        };
        let mut objects = 0u64;
        let mut bytes = 0u64;
        verify(st, &e.chunk, tree::KIND_DIR, 0, &mut objects, &mut bytes)?;
        Ok::<_, cas::Error>((objects, bytes))
    });
    match res {
        None => err("no store mounted"),
        Some(Err(e)) => report_store_error(e),
        Some(Ok((objects, bytes))) => {
            console::set_color(LTGREEN);
            // `get` already rejects anything whose bytes do not hash to the
            // address it was fetched by, so reaching here IS the verification.
            kprintln!("  {} object(s), {} B -- every address verified", objects, bytes);
            console::set_color(LTGRAY);
        }
    }
}

fn verify(
    st: &cas::Store,
    r: &cas::ChunkRef,
    kind: u8,
    depth: usize,
    objects: &mut u64,
    bytes: &mut u64,
) -> Result<(), cas::Error> {
    if depth > tree::MAX_DEPTH {
        return Err(cas::Error::Corrupt);
    }
    let raw = st.get(r)?;
    *objects += 1;
    *bytes += raw.len() as u64;
    if kind == tree::KIND_DIR {
        let entries = tree::decode_dir(&raw).ok_or(cas::Error::Corrupt)?;
        for (_, k, cr) in entries {
            verify(st, &cr, k, depth + 1, objects, bytes)?;
        }
    }
    Ok(())
}
