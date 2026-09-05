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
use crate::aiksi;
use crate::store::{self, cas};
use crate::sync::Racy;
use alloc::format;
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
    // Always mutating, by construction rather than by inspection: a program
    // can reach `write`, and `write` mutates, so no static analysis of the
    // source may claim otherwise. The read-only grammar therefore never
    // carries it, whatever the tool's text says.
    Applet { name: "run",    args: "<path>",       help: "execute a lang program from the namespace", mutates: true },
    // The model's own way to keep something. It is in this table rather than
    // being a parse of the model's prose, because everything else it does goes
    // through this table: the decoding grammar is built from it, so `remember`
    // is reachable by the same route as `ls`, and an answer that merely *says*
    // it will remember cannot be mistaken for one that did.
    Applet { name: "remember", args: "<text>",     help: "keep a fact about the operator", mutates: true },
];

/// Resolve a path the way every applet does, against the working directory.
///
/// Public because a capability check has to run on the *resolved* path. A jail
/// tested against what was typed is defeated by `../..`, and the resolution is
/// already here -- duplicating it in the caller is how the two drift apart and
/// the check starts passing things the write then puts somewhere else.
pub fn resolve_path(path: &str) -> String {
    if let Some(r) = with(|s| show(&parse(&s.cwd, path))) {
        return r;
    }
    // No namespace yet -- the boot selftests run before `init`. An absolute
    // path does not need a working directory to resolve, and answering "/" for
    // one would make a capability check refuse something it should allow.
    show(&path_of(path))
}

/// Whether an applet can change persistent content, or `None` if there is no
/// such applet.
///
/// The same flag `harness::Trust::ReadOnly` filters the model's grammar with,
/// so "safe to call" has one definition in this tree rather than two that
/// drift.
pub fn applet_mutates(name: &str) -> Option<bool> {
    APPLETS.iter().find(|a| a.name == name).map(|a| a.mutates)
}

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

/// The interpreter tools run on.
///
/// Deliberately not the shell's REPL session: a skill must not reach the
/// variables the operator was playing with, and the operator's session must
/// not inherit a tool's leftovers. State persists across invocations within
/// a boot, so a tool can keep a workspace the way a REPL does -- and the
/// interpreter's own step budget is what stops a model-written `while (1)`
/// from wedging the agent task, since the loop's abort check only fires
/// between steps.
static TOOLS: Racy<Option<aiksi::Interp>> = Racy::new(None);

fn with_tools<R>(f: impl FnOnce(&mut aiksi::Interp) -> R) -> Option<R> {
    unsafe { TOOLS.get().as_mut().map(f) }
}

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
    seed_tools(&mut sb);
    seed_apps(&mut sb);
    unsafe {
        *BOX.get() = Some(sb);
        *TOOLS.get() = Some(aiksi::Interp::new());
    }
    // After the namespace exists and before anything looks for a program.
    // A restored snapshot can be older than the rename, so this is not a
    // one-time upgrade step that can be deleted next week -- it runs whenever
    // a store is mounted, and costs a directory listing when there is nothing
    // to do.
    let moved = crate::app::migrate_extension();
    if moved > 0 {
        kprintln!("  [app] carried {} program(s) to code.ai&xi", moved);
    }
}

/// The starter skills. Shipped rather than documented, so the convention is
/// demonstrated by working examples the first time anything lists /ai/tools.
/// Re-seeded after a snapshot restore, because a snapshot taken before these
/// existed would otherwise come back without them -- the same reasoning the
/// corpus seeding in ai::init follows.
/// The one application that ships with the system.
///
/// Hand-written, and that is the point of it: the format has to be shown to
/// work with a program a person wrote before anything is asked to generate
/// one. It is also the worked example -- a panel document with a live `rows`
/// line, a program that keeps its state in the namespace, and actions that go
/// back through `app` rather than through the shell's vocabulary.
fn seed_apps(sb: &mut Sysbox) {
    let _ = tree::put(&mut sb.root, &path_of("/app/todo/panel.ui"), Node::Blob(
        b"panel\t1\n\
title\tToDo\n\
field\tnew\tapply app todo add\t\n\
sep\n\
heading\tTasks\n\
rows\trows\n\
sep\n\
button\trun app todo clear\tClear all\n\
button\tclose\tClose\n".to_vec()));
    let _ = tree::put(&mut sb.root, &path_of("/app/todo/code.ai&xi"), Node::Blob(
        b"// a list you can add to and tick off\n\
fn file() { return \"/app/todo/items\" }\n\
fn all() {\n\
  if (exists(file())) { return read(file()) }\n\
  return \"\"\n\
}\n\
fn add(what) {\n\
  if (len(what) > 0) { write(file(), all() + what + \"\\n\") }\n\
  return \"\"\n\
}\n\
fn clear() {\n\
  write(file(), \"\")\n\
  return \"\"\n\
}\n\
// remove the first line equal to `what`\n\
fn drop(what) {\n\
  text = all()\n\
  out = \"\"\n\
  line = \"\"\n\
  gone = 0\n\
  i = 0\n\
  while (i < len(text)) {\n\
    c = get(text, i)\n\
    if (c == \"\\n\") {\n\
      if (line == what) {\n\
        if (gone == 1) { out = out + line + \"\\n\" }\n\
        gone = 1\n\
      } else { out = out + line + \"\\n\" }\n\
      line = \"\"\n\
    } else { line = line + c }\n\
    i = i + 1\n\
  }\n\
  write(file(), out)\n\
  return \"\"\n\
}\n\
// one panel line per item; pressing one drops it\n\
fn rows() {\n\
  text = all()\n\
  out = \"\"\n\
  line = \"\"\n\
  i = 0\n\
  while (i < len(text)) {\n\
    c = get(text, i)\n\
    if (c == \"\\n\") {\n\
      out = out + \"item\\trun app todo drop \" + line + \"\\t\" + line + \"\\n\"\n\
      line = \"\"\n\
    } else { line = line + c }\n\
    i = i + 1\n\
  }\n\
  return out\n\
}\n".to_vec()));
}

fn seed_tools(sb: &mut Sysbox) {
    let _ = tree::put(&mut sb.root, &path_of("/ai/tools/hello.ai&xi"), Node::Blob(
        b"// says hello; the smallest working skill\n\
          println(\"hello from a tool\")\n".to_vec()));
    let _ = tree::put(&mut sb.root, &path_of("/ai/tools/status.ai&xi"), Node::Blob(
        b"// one-line status card: ticks and task count\n\
          println(\"ticks\", ticks(), \"tasks\", tasks())\n".to_vec()));
    // A tool with a function in it, deliberately.
    //
    // Every seeded tool was one line, and that is how `run` shipped unable to
    // execute any file containing a function without anyone noticing: the
    // examples could not exercise the defect. One multi-line example makes the
    // whole path load-bearing at boot.
    let _ = tree::put(&mut sb.root, &path_of("/ai/tools/count.ai&xi"), Node::Blob(
        b"// what is in a directory, by kind: run /ai/tools/count.ai&xi\n\
          fn tally(path: str): str {\n\
            dirs = 0\n\
            files = 0\n\
            names = ls(path)\n\
            n = 0\n\
            while (n < len(names)) {\n\
              s = stat(path + \"/\" + get(names, n))\n\
              if (s.is_dir) { dirs = dirs + 1 } else { files = files + 1 }\n\
              n = n + 1\n\
            }\n\
            return files + \" file(s), \" + dirs + \" director(ies)\"\n\
          }\n\
          println(tally(\"/ai/tools\"))\n".to_vec()));
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
                // A snapshot older than the starter skills restores without
                // them; put them back so the convention survives reboots.
                if children("/ai/tools").is_empty() {
                    seed_tools(s);
                    seed_apps(s);
                }
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
        "run" => cmd_run(a1),
        "remember" => cmd_remember(rest),
        _ => {}
    }
    true
}

/// Execute a lang program from the namespace, line by line, on the tool
/// interpreter.
///
/// Lines, not the whole file at once: the REPL is line-oriented and skills
/// are written to the same shape, so a tool reads like the code it is. The
/// first failing line stops the tool and names itself -- an error an agent
/// can read and react to beats a half-explained silence. The value of each
/// expression line is printed, because a tool's intermediate results are
/// exactly what an episode's observation wants to contain.
/// True while `cmd_run` is executing a program.
///
/// A program can reach `applet("run other")` through the interpreter's own
/// `applet` builtin, and that path comes back here. `with_tools` would then
/// take a second `&mut` to the one interpreter through `Racy`, which is the
/// single-core interior mutability `src/sync.rs` grants on the understanding
/// that nothing re-enters. Refused rather than nested: a second borrow is
/// undefined behaviour, and the only honest fix at this depth is to say no.
static RUNNING: AtomicBool = AtomicBool::new(false);

/// Where the operator records the skills allowed to keep operator powers.
///
/// One hex hash per line, and the hash is of the file's *bytes*, so editing a
/// skill revokes its trust by construction. That is the same property
/// `app::manifest` gets by putting the `raw` bit inside the hash, and it is
/// the reason trust cannot be inherited by a later version of a program.
const TRUSTED: &str = "/ai/tools/.trusted";

/// Steps a sandboxed skill may take.
///
/// Generous -- a skill that walks a directory tree does real work -- and
/// bounded, because the loop's abort check only fires between steps and a
/// machine-written `while (1)` would otherwise wedge whichever task ran it.
/// `with_step_budget` clamps to `STEP_BUDGET`, so this can only ever narrow.
const SKILL_BUDGET: u64 = 5_000_000;

fn hex32_of(h: &[u8; 32]) -> String {
    let mut s = String::new();
    for b in h {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        s.push(char::from_digit((b & 0xF) as u32, 16).unwrap_or('0'));
    }
    s
}

/// Whether these exact bytes have been trusted by the operator.
fn skill_trusted(h: &[u8; 32]) -> bool {
    let want = hex32_of(h);
    let Some(b) = read_blob(TRUSTED) else { return false };
    let Ok(text) = core::str::from_utf8(&b) else { return false };
    text.lines().any(|l| l.trim() == want)
}

pub fn skill_trust(prefix: &str) -> Option<String> {
    // Resolve a prefix against what is actually in /ai/tools, so the operator
    // types eight characters like everywhere else -- and so trusting a hash
    // that names no file on disk is impossible.
    let mut found: Option<String> = None;
    for name in children("/ai/tools") {
        let mut path = String::from("/ai/tools/");
        path.push_str(&name);
        let Some(b) = read_blob(&path) else { continue };
        let h = hex32_of(&crate::store::sha256::hash(&b));
        if h.starts_with(prefix) {
            if found.is_some() {
                return None; // ambiguous: refuse rather than trust the first
            }
            found = Some(h);
        }
    }
    let h = found?;
    let mut buf = read_blob(TRUSTED).unwrap_or_default();
    if let Ok(t) = core::str::from_utf8(&buf) {
        if t.lines().any(|l| l.trim() == h) {
            return Some(h);
        }
    }
    if !buf.is_empty() && !buf.ends_with(b"\n") {
        buf.push(b'\n');
    }
    buf.extend_from_slice(h.as_bytes());
    buf.push(b'\n');
    if write_blob(TRUSTED, buf) { Some(h) } else { None }
}

pub fn skill_untrust_all() -> bool {
    write_blob(TRUSTED, Vec::new())
}

/// Every skill, with its hash and whether it is trusted.
pub fn skill_list() -> Vec<(String, String, bool)> {
    let mut out = Vec::new();
    for name in children("/ai/tools") {
        if name.starts_with('.') {
            continue;
        }
        let mut path = String::from("/ai/tools/");
        path.push_str(&name);
        let Some(b) = read_blob(&path) else { continue };
        let h = crate::store::sha256::hash(&b);
        out.push((name, hex32_of(&h), skill_trusted(&h)));
    }
    out
}

fn cmd_run(path: &str) {
    if path.is_empty() {
        kprintln!("  usage: run <path>");
        return;
    }
    let Some(bytes) = read_blob(path) else {
        kprintln!("  run: no such file '{}'", path);
        return;
    };
    if RUNNING.swap(true, Ordering::Acquire) {
        kprintln!("  run: already running a program -- a program cannot run another");
        return;
    }
    let text = String::from_utf8_lossy(&bytes).into_owned();
    // The whole file at once, not a line at a time.
    //
    // It was a line at a time, and that made a multi-line program impossible:
    // `fn add(a, b) {` is not a statement, so any file containing a function
    // failed on its first line. `lex` already treats a newline as whitespace
    // and `parse` already consumes statements until end of input, so a file
    // has always been a legal program -- the loop was the only thing insisting
    // otherwise. Nothing noticed because every seeded tool is one line and
    // every case in `aiksi::selftest` was a single-line string.
    //
    // The cost is the per-line `= value` echo, and the line number in a parse
    // error. Programs that want output call `println`, which the replay
    // programs `agent learn` writes already do.
    // **A skill is not automatically the operator.**
    //
    // Every program under /ai/tools used to run on `TOOLS`, which is
    // `Interp::new()` -- operator capabilities: raw memory, I/O ports, the
    // network, the model, the framebuffer. An *app* has been jailed since
    // `app::call` was written; a *tool* was not, and `agent learn` writes
    // tools, and a shared skill from a stranger is a tool. That was the widest
    // hole in the system and it was open by omission rather than by argument.
    //
    // The split follows `app::call` exactly: operator powers only for bytes
    // the operator has named, everything else fresh and sandboxed. Identity is
    // the hash of the file, so editing a trusted skill revokes its trust
    // without anyone having to remember to.
    let h = crate::store::sha256::hash(&bytes);
    let out = if skill_trusted(&h) {
        // The persistent session, deliberately: a trusted tool keeps a
        // workspace across invocations, which is the whole reason `TOOLS` is
        // not the shell's REPL.
        with_tools(|tools| aiksi::eval_line(tools, &text))
    } else {
        // Its own subtree, so two skills cannot reach each other's scratch,
        // and a fresh interpreter so nothing survives a run.
        //
        // The jail mirrors the program's own resolved path under
        // `/ai/tools/scratch`, so `/ai/tools/mk.ai&xi` writes under
        // `/ai/tools/scratch/ai/tools/mk.ai&xi`. A mirror rather than a
        // prettier name because the property being bought is that two
        // different programs can never share a jail, and a bijection is the
        // only way to have that for free.
        //
        // The name it replaces was the file's leaf with its extension
        // stripped, taken off the raw argument, and it failed both ways.
        // `run /ai/tools/evil.ai&xi/` still runs -- `parse` drops the trailing
        // slash -- but `rsplit('/')` then saw an empty component, so the jail
        // collapsed to `/ai/tools/scratch` itself, the parent of every other
        // skill's scratch, and an untrusted program could read and overwrite
        // the persisted state of every skill on the machine. Less
        // dramatically, `mk.ai&xi`, `mk.bak` and `mk.old` all stemmed to `mk`,
        // and `run` takes any namespace path, so `/app/report.ai&xi` and
        // `/ai/tools/report.ai&xi` shared one too. Resolving first fixes the
        // first; mirroring the whole path fixes the rest.
        let parts = with(|s| parse(&s.cwd, path)).unwrap_or_default();
        // A path that resolves to nothing is the root of the namespace, and
        // the root is not a jail. `read_blob` succeeded above so this should
        // be unreachable; refusing rather than trusting that is the difference
        // between a bug and a hole.
        if parts.is_empty() {
            RUNNING.store(false, Ordering::Release);
            kprintln!("  run: '{}' does not name a program", path);
            return;
        }
        let mut jail = String::from("/ai/tools/scratch");
        for part in &parts {
            jail.push('/');
            jail.push_str(part);
        }
        let mut sandbox = aiksi::Interp::sandboxed(&jail).with_step_budget(SKILL_BUDGET);
        Some(aiksi::eval_line(&mut sandbox, &text))
    };
    RUNNING.store(false, Ordering::Release);
    match out {
        None => kprintln!("  run: no interpreter"),
        Some(Ok(v)) => {
            if !matches!(v, aiksi::Value::Nil) {
                kprintln!("  = {}", v.render());
            }
        }
        Some(Err(e)) => kprintln!("  run: {}", e),
    }
}

/// The content address of a namespace path, for callers that want to compare
/// rather than print. `print_hash` is the operator's view; this is the
/// programmatic one, and it is what makes a fitted probe verifiable against
/// the corpus it was fitted on.
pub fn hash_of(path: &str) -> Option<[u8; 32]> {
    with(|s| {
        let p = parse(&s.cwd, path);
        tree::resolve(&s.root, &p).map(tree::content_hash)
    })
    .flatten()
}

/// The skills the namespace holds, with their first comment line as a
/// description. This is what episode prompts and `agent skills` render, so the
/// model can know what it (or the operator) has written.
///
/// Both extensions are accepted. A store mounted from before the rename holds
/// `.l` files until `migrate_extension` runs, and a filter that knew only the
/// new spelling would answer "no skills" for a namespace full of them --
/// silently, because an empty list is what a fresh system legitimately has.
pub fn skills() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in children("/ai/tools") {
        if !name.ends_with(".ai&xi") && !name.ends_with(".l") {
            continue;
        }
        let desc = read_blob(&format!("/ai/tools/{}", name))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .and_then(|t| t.lines().next().map(String::from))
            .unwrap_or_default();
        // Strip the comment marker: `// says hello` becomes `says hello`.
        let desc = desc.trim_start_matches('/').trim().to_string();
        out.push((name, desc));
    }
    out
}

pub fn is_ready() -> bool {
    unsafe { BOX.get().is_some() }
}

/// Shape-check arguments against the applet's declared usage, before dispatch.
///
/// This is the boundary the agent loop hands free generated text across. The
/// grammar guarantees the *name*; nothing yet has guaranteed the arguments,
/// and an applet that never runs is a poor answer compared to one that is
/// told its arguments were wrong before anything executed. What is checked
/// is shape only -- arity and the declared types. Whether a path exists is
/// deliberately left to the applet itself, because "no such path" is a
/// useful observation for a model to reason over, while "wrong number of
/// words" is merely noise to dispatch.
pub fn check_args(name: &str, rest: &str) -> Result<(), String> {
    let Some(a) = APPLETS.iter().find(|a| a.name == name) else {
        return Err(format!("'{}' is not an applet", name));
    };
    let spec: Vec<&str> = a.args.split_whitespace().collect();
    let words: Vec<&str> = rest.split_whitespace().collect();

    // A trailing <text> absorbs everything after it: `write <path> <text>`
    // takes two words minimum but any number beyond that.
    let text_tail = spec.last() == Some(&"<text>");
    let required = spec.iter().filter(|s| s.starts_with('<')).count();

    if text_tail {
        if words.len() < required {
            return Err(format!(
                "'{}' wants at least {} argument(s), got {}: usage {}",
                name,
                required,
                words.len(),
                a.args
            ));
        }
    } else if words.len() < required || words.len() > spec.len() {
        return Err(format!(
            "'{}' wants {} argument(s), got {}: usage {}",
            name,
            required,
            words.len(),
            a.args
        ));
    }

    // Typed checks where the usage names a type. Only exact-arity forms are
    // checked positionally; the optional-bearing specs (`[seq]`) are left to
    // the applet, which reports its own errors legibly enough.
    if !spec.contains(&"[seq]") && !spec.contains(&"[path]") && !spec.contains(&"[path|-]") {
        for (w, sp) in words.iter().zip(spec.iter()) {
            if *sp == "<seq>" && w.parse::<u64>().is_err() {
                return Err(format!("<seq> must be a number, got '{}'", w));
            }
        }
    }
    Ok(())
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

/// Append a line to what the model knows about the operator.
///
/// Appends rather than sets: facts accumulate, and a setter would make every
/// new one cost the old ones. Duplicates are refused because the file is read
/// into the system turn of every conversation, so a fact repeated forty times
/// is forty times the context for no more information.
fn cmd_remember(text: &str) {
    let text = text.trim();
    if text.is_empty() {
        crate::kprintln!("  usage: remember <text>");
        return;
    }
    let path = crate::ai::companion::ABOUT;
    let mut buf = read_blob(path).unwrap_or_default();
    if let Ok(existing) = core::str::from_utf8(&buf) {
        if existing.lines().any(|l| l.trim() == text) {
            crate::kprintln!("  already known");
            return;
        }
    }
    if !buf.is_empty() && !buf.ends_with(b"\n") {
        buf.push(b'\n');
    }
    buf.extend_from_slice(text.as_bytes());
    buf.push(b'\n');
    if write_blob(path, buf) {
        crate::kprintln!("  kept: {}", text);
        // Say which of the three it is. "'snap' to carry it past this boot"
        // was true before autosnap defaulted on and is now advice for a
        // problem the machine has already solved -- and it was silent about
        // the case that actually loses the fact, which is a store that is
        // mounted but not writable.
        if !crate::store::mounted() {
            crate::kprintln!("  but there is no store -- it ends with this boot ('store init')");
        } else if !crate::dev::nvme::writes_unlocked() {
            crate::kprintln!("  but writes are locked -- 'store unlock' to keep it past the reboot");
        } else if !autosnap_enabled() {
            crate::kprintln!("  autosnap is off -- 'snap' to carry it past this boot");
        } else {
            crate::kprintln!("  it will be written within {} s", autosnap_interval());
        }
    } else {
        crate::kprintln!("  could not write {}", path);
    }
}

/// One task's namespace writes, while a `shadow` is watching that task.
///
/// The identity is the task, not the tree, because the tree cannot tell who
/// wrote to it. Five tasks share this namespace and three of them write to it
/// unprompted; without a name on each write there is no way to undo a
/// sandboxed program's changes without also undoing the agent's.
struct Watch {
    task: usize,
    /// Each path the run touched, and what was there the *first* time it did.
    /// `None` means the path did not exist, so undoing it is a removal.
    seen: Vec<(Vec<String>, Option<Node>)>,
}

static WATCH: Racy<Option<Watch>> = Racy::new(None);

/// Record a mutation against the run that made it, if anyone is watching.
///
/// Called from every function that changes the live tree. A mutation that
/// reaches the tree without passing through here is one `shadow` cannot undo,
/// which is why the call sits beside the mutation rather than at the callers.
/// The pre-image is taken here, which is what removed the whole-tree copy.
///
/// `shadow` used to open by deep-copying the entire namespace so it had
/// something to restore from -- every `sandbox` run paid for a full clone of
/// every object in the tree, to undo a program that usually touches one file.
/// But this function already runs immediately *before* each mutation with the
/// path in hand, so the only bytes worth saving can be saved exactly here, and
/// the cost becomes proportional to what the run actually changed.
///
/// The full read-through overlay would go further and make the run *invisible*
/// to other tasks rather than merely undoable. It is not here, and the reason
/// is that it would have to be paid for in the type: `Node` owns its children,
/// so a persistent tree that shares unmodified subtrees needs reference
/// counting through every accessor in the kernel. What it would buy is
/// isolation the jail already provides -- a sandboxed skill can only write
/// under its own scratch subtree, so there is nothing for another task to see.
fn note(root: &Node, p: &[String]) {
    unsafe {
        let Some(w) = (*WATCH.get()).as_mut() else { return };
        if w.task != crate::task::current() {
            return;
        }
        // The shallowest path that does not exist yet, not the one asked for.
        //
        // `tree::put` creates intermediate directories, so writing `/a/b/c`
        // into an empty tree creates `/a` and `/a/b` as well. Recording only
        // `/a/b/c` and removing it on undo leaves those behind -- the run is
        // reported as undone and the namespace has two directories it did not
        // have before. Recording `/a` instead takes the whole thing back out.
        let mut at = p;
        if tree::resolve(root, p).is_none() {
            for n in 1..=p.len() {
                if tree::resolve(root, &p[..n]).is_none() {
                    at = &p[..n];
                    break;
                }
            }
        }
        if w.seen.iter().any(|(q, _)| q.as_slice() == at) {
            return;
        }
        let pre = tree::resolve(root, at).map(tree::clone_node);
        w.seen.push((at.to_vec(), pre));
    }
}

pub fn write_blob(path: &str, data: Vec<u8>) -> bool {
    with(|s| {
        let p = parse(&s.cwd, path);
        note(&s.root, &p);
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

/// Detach a name. The content stays addressable, as everywhere else.
pub fn detach(path: &str) -> bool {
    with(|s| {
        let p = parse(&s.cwd, path);
        note(&s.root, &p);
        tree::remove(&mut s.root, &p).is_some()
    })
    .unwrap_or(false)
}

/// Print the content address of a path, if it exists.
pub fn print_hash(path: &str) {
    let h = with(|s| {
        let p = parse(&s.cwd, path);
        tree::resolve(&s.root, &p).map(tree::content_hash)
    })
    .flatten();
    match h {
        Some(h) => {
            let hx = tree::short(&h);
            kprintln!("  address {}", core::str::from_utf8(&hx).unwrap_or("?"));
        }
        None => kprintln!("  (not present)"),
    }
}

/// Entries under `path` with what a browser needs to draw them: the name,
/// whether it is a directory, and how many children or bytes it holds.
///
/// Separate from `children` rather than replacing it because the callers that
/// only want names are the majority, and a browser is the only thing that has
/// to tell a directory from a file without opening it.
pub fn listing(path: &str) -> Vec<(String, bool, usize)> {
    with(|s| {
        let p = parse(&s.cwd, path);
        match tree::resolve(&s.root, &p) {
            Some(Node::Dir(es)) => es
                .iter()
                .map(|(k, v)| match v {
                    Node::Dir(inner) => (k.clone(), true, inner.len()),
                    Node::Blob(b) => (k.clone(), false, b.len()),
                })
                .collect(),
            _ => Vec::new(),
        }
    })
    .unwrap_or_default()
}

/// How many bytes a blob holds, without copying it out.
///
/// `read_blob` clones, which is right for a caller that wants the contents and
/// ruinous for one that only wants the length: `stat` on a file asks exactly
/// that question, and answering it through `read_blob` allocates the whole
/// file to look at a `usize`. A 600 MB checkpoint is a legal thing to `stat`
/// and not a legal thing to copy on a machine with one address space.
///
/// `None` for a directory as well as for a missing path, since a directory has
/// no byte count and answering zero would be indistinguishable from an empty
/// file.
pub fn blob_len(path: &str) -> Option<usize> {
    with(|s| {
        let p = parse(&s.cwd, path);
        match tree::resolve(&s.root, &p) {
            Some(Node::Blob(b)) => Some(b.len()),
            _ => None,
        }
    })
    .flatten()
}

/// Create an empty directory, failing if the name is taken.
///
/// `write_blob` already creates intermediate directories, so this exists for
/// the one case that has no blob at the end of it: `mkdir`. Refusing an
/// existing name rather than replacing it is what `EEXIST` is, and replacing a
/// directory that has things in it would be a silent recursive delete.
pub fn make_dir(path: &str) -> bool {
    with(|s| {
        let p = parse(&s.cwd, path);
        if tree::resolve(&s.root, &p).is_some() {
            return false;
        }
        note(&s.root, &p);
        tree::put(&mut s.root, &p, Node::Dir(Vec::new())).is_ok()
    })
    .unwrap_or(false)
}

/// Whether a path names a directory.
pub fn is_dir(path: &str) -> bool {
    with(|s| {
        let p = parse(&s.cwd, path);
        matches!(tree::resolve(&s.root, &p), Some(Node::Dir(_)))
    })
    .unwrap_or(false)
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

    // What `sandbox` promises: undo this run, and only this run.
    //
    // The foreign write below is not a contrivance. Three tasks write to this
    // namespace unprompted, a sandboxed skill has five million steps and no
    // yield point, so a run spans seconds during which the agent task is
    // writing episode transcripts and the initiative task its journal. The
    // first version of `shadow` restored the whole root and threw all of that
    // away silently, reporting "discarded" as though only the program had
    // been undone. A snapshot diff cannot tell the two apart -- both are
    // changes -- so the test has to be that a write made under another task's
    // name survives, which is exactly what is checked here.
    //
    // At boot the selftests run before a namespace is mounted, so there is
    // nothing to shadow. Said out loud rather than skipped quietly -- an
    // unrunnable claim that prints nothing is indistinguishable from one that
    // passed, and `diag sysbox` runs this same suite once there is a tree.
    if with(|_| ()).is_none() {
        kprintln!("  --   a shadow undoes its own run only (no namespace yet: `diag sysbox`)");
    } else if let Some(sh) = shadow(|| {
        write_text("/tmp/.selftest-mine", "mine\n");
        // A write by somebody else, which is precisely a write recorded
        // against a different task id.
        unsafe {
            if let Some(w) = (*WATCH.get()).as_mut() {
                w.task = w.task.wrapping_add(1);
            }
        }
        write_text("/tmp/.selftest-theirs", "theirs\n");
        unsafe {
            if let Some(w) = (*WATCH.get()).as_mut() {
                w.task = w.task.wrapping_sub(1);
            }
        }
    }) {
        ok &= check(
            "a shadow reports only the paths the run itself wrote",
            sh.changes == 1 && sh.touched.iter().any(|t| t.ends_with("/tmp/.selftest-mine")),
        );
        sh.discard();
        ok &= check(
            "discarding undoes the run's own write",
            read_blob("/tmp/.selftest-mine").is_none(),
        );
        ok &= check(
            "and leaves another task's write alone",
            read_blob("/tmp/.selftest-theirs").is_some(),
        );
        detach("/tmp/.selftest-theirs");
    }

    // Undoing a write takes back the directories the write created.
    //
    // `tree::put` makes intermediate directories, so a program writing one
    // file into a fresh path creates several objects and only names one of
    // them. Undoing just the named one reports the run as reverted and leaves
    // the tree with directories it did not have -- which is the same
    // "discarded, and something is different" failure the whole mechanism
    // exists to prevent, one level up. `note` records the shallowest path that
    // did not exist rather than the one asked for.
    if with(|_| ()).is_some() {
        detach("/tmp/.deep");
        match shadow(|| {
            write_text("/tmp/.deep/a/b/c", "nested\n");
        }) {
            Some(sh) => {
                let made = read_blob("/tmp/.deep/a/b/c").is_some();
                sh.discard();
                ok &= check(
                    "undoing a nested write removes the directories it created",
                    made && !is_dir("/tmp/.deep"),
                );
            }
            None => ok &= check("a shadow can be taken", false),
        }
    }

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
        note(&s.root, &p);
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
        note(&s.root, &p);
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
        note(&s.root, &p);
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
            note(&s.root, &pa);
            tree::remove(&mut s.root, &pa);
        }
        note(&s.root, &pb);
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
                // So the next automatic snapshot does not rewrite a tree that
                // was just committed by hand.
                note_committed();
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
                    out.push((m.seq, cur.hash, m.entries.len(), m.time));
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
            kprintln!("  {:>5}  {:16}  {:19}  entries", "seq", "root", "taken");
            console::set_color(WHITE);
            for (seq, hash, n, time) in list {
                let hx = crate::store::sha256::short_hex(&hash);
                // A snapshot written before the clock existed, or on a machine
                // with no working RTC, records zero. Saying so beats printing
                // 1970 as though it were a fact.
                if time == 0 {
                    kprintln!(
                        "  {:>5}  {}  {:19}  {}",
                        seq,
                        core::str::from_utf8(&hx).unwrap_or("?"),
                        "(no clock)",
                        n
                    );
                } else {
                    let d = crate::dev::rtc::from_unix(time);
                    kprintln!(
                        "  {:>5}  {}  {:04}-{:02}-{:02} {:02}:{:02}:{:02}  {}",
                        seq,
                        core::str::from_utf8(&hx).unwrap_or("?"),
                        d.year, d.month, d.day, d.hour, d.minute, d.second,
                        n
                    );
                }
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
    diff_walk(&left, &right, &mut Vec::new(), &mut changes, &mut |sign, at| mark(sign, at));
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
/// Walk two trees and report every difference to `sink`.
///
/// The sink exists so there is one walk rather than two. `diff` prints; the
/// shadow sandbox collects. A second copy of this traversal would be a second
/// place for the sorted-entry merge to be got wrong, and the two would drift.
fn diff_walk(
    a: &Node,
    b: &Node,
    at: &mut Vec<String>,
    changes: &mut u32,
    sink: &mut dyn FnMut(&str, &[String]),
) {
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
                        diff_walk(&ea[i].1, &eb[j].1, at, changes, sink);
                        at.pop();
                        i += 1;
                        j += 1;
                    }
                    core::cmp::Ordering::Less => {
                        at.push(ea[i].0.clone());
                        *changes += 1;
                        sink("-", at);
                        at.pop();
                        i += 1;
                    }
                    core::cmp::Ordering::Greater => {
                        at.push(eb[j].0.clone());
                        *changes += 1;
                        sink("+", at);
                        at.pop();
                        j += 1;
                    }
                }
            }
        }
        _ => {
            *changes += 1;
            sink("~", at);
        }
    }
}

/// The printing sink, for `diff`.
fn mark(sign: &str, at: &[String]) {
    match sign {
        "-" => console::set_color(LTRED),
        "+" => console::set_color(LTGREEN),
        _ => console::set_color(YELLOW),
    }
    kprintln!("  {} {}", sign, show(at));
    console::set_color(LTGRAY);
}

/// The namespace as it was, so a change can be put back.
pub struct Shadow {
    /// The sign, the path, and what was there before the run touched it.
    /// Sorted by path, which is what makes an ancestor undo before its
    /// descendants.
    undo: Vec<(char, Vec<String>, Option<Node>)>,
    /// Every path the change touched, signed: `+` added, `-` removed,
    /// `~` altered.
    pub touched: Vec<String>,
    pub changes: u32,
}

/// Run something and find out exactly what it did to the namespace.
///
/// This is the shape the self-improvement loop needs before it adopts anything
/// a model wrote: run it, see precisely which objects moved, then decide. The
/// pieces were all here and unassembled -- the tree is a Merkle tree, so
/// `content_hash` makes "did this subtree change" a single comparison, and
/// `diff_walk` already skips every subtree that did not.
///
/// **Reversible, not isolated, and the difference matters.** The change runs
/// against the live tree and is undone afterwards; it is not confined to a
/// copy. So a program that faults halfway leaves the tree as it was only
/// because `discard` puts it back, and anything reading the namespace *during*
/// the run sees the change.
///
/// **Concurrent writers do exist**, which was the flaw in the first version of
/// this. A sandboxed skill gets five million steps and Aiksi has no yield
/// point, so a run spans many seconds of wall clock, and the agent and
/// initiative tasks go on writing episode transcripts, outcome lines, journal
/// entries and freshly authored skills throughout it. `discard` restored the
/// whole root, so every one of those writes was silently thrown away while the
/// shell printed "discarded" as though only the sandboxed program had been
/// undone -- and `touched` blamed the sandboxed program for them into the
/// bargain, so the command answered its one question wrongly in both
/// directions.
///
/// **So the writes are attributed, not diffed.** Comparing two snapshots
/// cannot answer this: a change is a change whoever made it, so every
/// background write lands in the diff and gets reverted and reported exactly
/// like the run's own. What distinguishes them is *who*, and the tree does not
/// record that -- so `note` does, against `task::current()`, at each mutation.
/// A path this task did not write is a path this function neither reports nor
/// undoes.
///
/// The snapshot is still taken, because knowing which paths changed is not
/// knowing what they held before.
///
/// **Nothing is copied up front.** The first version opened by deep-copying the
/// whole namespace so it had something to restore from, which meant every
/// `sandbox` run paid for a clone of every object in the tree in order to undo
/// a program that usually touches one file. `note` already runs immediately
/// before each mutation with the path in hand, so each path's pre-image is
/// saved exactly there and the cost is proportional to the change. It also
/// closed the window this used to hold `&mut Sysbox` open for: a full
/// recursive walk with interrupts disabled, taken twice per run.
///
/// The read-through overlay -- a namespace handle threaded through `with`, so
/// the run is *invisible* to other tasks rather than merely undoable -- is
/// still not here, and the reason is worth stating rather than deferring
/// again. It has to be paid for in the type: `Node` owns its children, so a
/// persistent tree that shares unmodified subtrees needs reference counting
/// through every accessor in the kernel. What it would buy is isolation the
/// jail already provides, since a sandboxed skill can only write under its own
/// scratch subtree and there is nothing there for another task to see.
pub fn shadow<F: FnOnce()>(f: F) -> Option<Shadow> {
    // No snapshot. `note` captures each path's pre-image the first time the
    // run touches it, so there is nothing to copy up front and the cost is
    // proportional to the change rather than to the tree.
    let mine = crate::task::current();
    let prev = unsafe {
        core::mem::replace(&mut *WATCH.get(), Some(Watch { task: mine, seen: Vec::new() }))
    };
    f();
    let seen = unsafe { core::mem::replace(&mut *WATCH.get(), prev) };
    let mut seen = seen.map(|w| w.seen).unwrap_or_default();
    // Ancestors first, so restoring a directory happens before the entries
    // inside it are put back or taken away.
    seen.sort_by(|a, b| a.0.cmp(&b.0));

    let mut touched = Vec::new();
    let mut undo: Vec<(char, Vec<String>, Option<Node>)> = Vec::new();
    let mut changes = 0u32;
    with(|s| {
        for (at, pre) in seen {
            let was = pre.as_ref().map(tree::content_hash);
            let now = tree::resolve(&s.root, &at).map(tree::content_hash);
            // Written back exactly as it was. A write is not a change.
            if was == now {
                continue;
            }
            let sign = match (&was, &now) {
                (None, Some(_)) => '+',
                (Some(_), None) => '-',
                _ => '~',
            };
            changes += 1;
            let mut line = String::from(sign);
            line.push(' ');
            line.push_str(&show(&at));
            touched.push(line);
            undo.push((sign, at, pre));
        }
    })?;
    Some(Shadow { undo, touched, changes })
}

impl Shadow {
    /// Put back exactly what the run changed, and nothing else.
    ///
    /// Path by path rather than by restoring the root, because the root
    /// belongs to every task and this list belongs to one run. A file the
    /// agent task wrote while the program was thinking was never recorded
    /// against this run, so it is not in `undo`, so it survives.
    ///
    /// Sorted order matters in one case and is free: `/a` is undone before
    /// `/a/b`, so removing a directory the run created cannot be followed by
    /// re-creating a file inside it.
    pub fn discard(self) -> bool {
        crate::cpu::without_interrupts(|| {
            with(|s| {
                for (_, at, pre) in self.undo {
                    match pre {
                        // Whatever was there goes back, entry and all.
                        Some(node) => {
                            let _ = tree::put(&mut s.root, &at, node);
                        }
                        // It was not there before, so it should not be there
                        // now -- and `at` is the shallowest path the run
                        // created, so this takes any directories it made on
                        // the way with it.
                        None => {
                            tree::remove(&mut s.root, &at);
                        }
                    }
                }
            })
        })
        .is_some()
    }

    /// Leave the change in place.
    ///
    /// Named rather than implied by dropping, because "the default is to keep
    /// it" and "the default is to undo it" are opposite safety properties and
    /// a reader should not have to infer which one this is.
    pub fn keep(self) {}
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

// --- keeping itself ------------------------------------------------------
//
// The namespace has been persistent only when asked. `snap` writes it; forget
// to and a reboot loses everything since the last one. Orthogonal persistence
// was on the list from the beginning and this is the usable half of it: the
// system notices its own state has changed and commits it, without being told.
//
// The snapshot does *not* happen on the clock task, which is where a timer
// naturally belongs. Sysbox lives behind `Racy`, and a background writer would
// race the shell mid-command -- a `mkdir` half-applied while the tree is being
// serialised is a corrupt snapshot, and the single-core argument does not save
// it because preemption is real. So the timer only sets a flag and the shell
// acts on it between commands, where nothing else is touching the tree. Same
// discipline as the engine: one place, not a list of call sites.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// On by default.
///
/// Safe to default on only because the *write* gate is separate and stays
/// manual: `autosnap_poll` refuses unless `writes_unlocked()`, and mounting a
/// store deliberately does not unlock it. So this changes nothing at all until
/// the operator has asked for writes, and a machine that has not been asked
/// behaves exactly as it did before.
///
/// What it buys is that a fact the model was told -- `remember`, `about` -- is
/// kept without anybody remembering to type `snap` afterwards. A companion
/// whose memory depends on the operator running a command has not got one.
static AUTOSNAP_ON: AtomicBool = AtomicBool::new(true);
static AUTOSNAP_DUE: AtomicBool = AtomicBool::new(false);
static AUTOSNAP_EVERY: AtomicU64 = AtomicU64::new(60);
static LAST_SNAP_TICK: AtomicU64 = AtomicU64::new(0);
/// Content address of the tree as last committed, so an unchanged namespace
/// costs nothing.
static LAST_ROOT: Racy<Option<tree::Hash>> = Racy::new(None);

pub fn autosnap_configure(on: bool, seconds: u64) {
    AUTOSNAP_ON.store(on, Ordering::Relaxed);
    if seconds > 0 {
        AUTOSNAP_EVERY.store(seconds, Ordering::Relaxed);
    }
    LAST_SNAP_TICK.store(crate::dev::lapic::ticks(), Ordering::Relaxed);
}

pub fn autosnap_enabled() -> bool {
    AUTOSNAP_ON.load(Ordering::Relaxed)
}

pub fn autosnap_interval() -> u64 {
    AUTOSNAP_EVERY.load(Ordering::Relaxed)
}

/// Called from the clock task. Sets a flag and nothing more.
pub fn autosnap_tick() {
    if !AUTOSNAP_ON.load(Ordering::Relaxed) {
        return;
    }
    let hz = crate::TIMER_HZ as u64;
    let now = crate::dev::lapic::ticks();
    let last = LAST_SNAP_TICK.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= AUTOSNAP_EVERY.load(Ordering::Relaxed) * hz {
        LAST_SNAP_TICK.store(now, Ordering::Relaxed);
        AUTOSNAP_DUE.store(true, Ordering::Relaxed);
    }
}

/// Called by the shell between commands. Does the work, if any is due.
pub fn autosnap_poll() {
    if !AUTOSNAP_DUE.swap(false, Ordering::Relaxed) {
        return;
    }
    if !crate::store::mounted() || !crate::dev::nvme::writes_unlocked() {
        return;
    }

    // Hash first. An unchanged tree hashes the same, so a system sitting idle
    // writes nothing at all -- which is what makes a one-minute interval
    // reasonable rather than a way to fill the store with duplicates.
    let current = with(|s| tree::content_hash(&s.root));
    let Some(current) = current else { return };
    let unchanged = unsafe { LAST_ROOT.get().map(|h| h == current).unwrap_or(false) };
    if unchanged {
        return;
    }

    let saved = with(|s| {
        crate::store::with(|st| {
            let before = st.sb.alloc_next;
            let root = write_node(st, &s.root, &mut s.written).ok()?;
            let mut name = [0u8; cas::NAME_LEN];
            name[..4].copy_from_slice(b"root");
            let entry = cas::Entry { name, chunk: root };
            st.commit(core::slice::from_ref(&entry))
                .ok()
                .map(|_| (st.sb.seq, st.sb.alloc_next - before, st.free_blocks()))
        })
    })
    .flatten()
    .flatten();

    if let Some((seq, blocks, free)) = saved {
        unsafe { *LAST_ROOT.get() = Some(current) };
        console::set_color(LTGRAY);
        // The block count is not decoration. This store is append-only --
        // `alloc_next` only ever goes up and nothing reclaims -- so anything
        // large that changes often will fill it, and printing the cost each
        // time is the only way to find out which thing that is.
        kprintln!("\n[autosnap] snapshot {}, {} block(s), {} free", seq, blocks, free);
    }
}

/// Remember what was just committed, so a manual `snap` does not cause the
/// next automatic one to rewrite the same tree.
pub fn note_committed() {
    let current = with(|s| tree::content_hash(&s.root));
    if let Some(h) = current {
        unsafe { *LAST_ROOT.get() = Some(h) };
    }
}

pub fn autosnap_report() {
    console::set_color(YELLOW);
    kprintln!("[autosnap]");
    console::set_color(LTGRAY);
    if !autosnap_enabled() {
        kprintln!("  off -- 'autosnap on [seconds]' to have the system keep itself");
        return;
    }
    kprintln!("  every {} s, when the namespace has actually changed", autosnap_interval());
    if !crate::store::mounted() {
        console::set_color(YELLOW);
        kprintln!("  but no store is mounted, so nothing will be written");
        console::set_color(LTGRAY);
    } else if !crate::dev::nvme::writes_unlocked() {
        console::set_color(YELLOW);
        kprintln!("  but writes are locked -- 'store unlock'");
        console::set_color(LTGRAY);
    }
}
