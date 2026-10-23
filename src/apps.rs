//! Installed programs: what the desktop offers, and how each one starts.
//!
//! A program written for another system is a file plus a recipe -- which
//! volume to mount where, which interpreter to put where its headers look,
//! which environment, which directory to start in, what to run. Typing that
//! recipe is what running Portal meant until now, nine shell lines long. An
//! entry here is the recipe written down once, and the desktop turns it into
//! an icon.
//!
//! **Entries travel with the software.** Every FAT volume the machine can see
//! is looked at for an `apps/` directory of `*.app` files, so a game disk
//! carries its own entries and nothing about it lives in this tree. An entry
//! in `/etc/apps/` in the namespace counts too, for something installed by
//! hand.
//!
//! **The runtime is a field, not an assumption.** `runtime=linux` is the only
//! one today, and `launch` dispatches on it, so a Windows or macOS runtime is
//! a new arm here and a new `Runtime` rather than a second launcher.
//!
//! The format is `key=value` lines, `#` for comments, and keys that may
//! repeat (`env`, `copy`) simply repeat:
//!
//! ```text
//! name=Portal
//! runtime=linux
//! mount=/mnt                 # the volume this file was found on, mounted here
//! copy=/mnt/lib32/ld-linux.so.2 /lib/ld-linux.so.2
//! cwd=/mnt/portal
//! env=LD_LIBRARY_PATH=/mnt/portal/bin:/mnt/lib32
//! gpu=yes                    # Zink over Venus when a GPU is there
//! exec=/mnt/portal/hl2_linux -game portal -fullscreen
//! ```
//!
//! `$DISK` in any value is the volume's own mount point, for an entry that
//! does not want to pick one.

use crate::sync::Racy;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Runtime {
    Linux,
}

#[derive(Clone, Debug)]
pub struct App {
    pub name: String,
    pub runtime: Runtime,
    /// The FAT partition the entry came from, if it came from one.
    pub partition: Option<u32>,
    pub mount: Option<String>,
    pub copies: Vec<(String, String)>,
    pub cwd: Option<String>,
    pub env: Vec<String>,
    pub gpu: bool,
    pub exec: String,
}

static APPS: Racy<Vec<App>> = Racy::new(Vec::new());
/// Moves whenever the list changes, so the desktop knows to repaint icons.
pub static GENERATION: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Where every FAT volume is mounted for guests to read, so an entry's paths
/// have somewhere to point before anything has been launched.
fn disk_root(index: u32) -> String {
    alloc::format!("/disk/{}", index)
}

/// Read one entry. `None` for anything without a name and something to run,
/// or naming a runtime this machine does not have -- an entry that cannot
/// start is not offered.
pub fn parse(text: &str, partition: Option<u32>) -> Option<App> {
    let disk = partition.map(disk_root).unwrap_or_default();
    let mut a = App {
        name: String::new(),
        runtime: Runtime::Linux,
        partition,
        mount: None,
        copies: Vec::new(),
        cwd: None,
        env: Vec::new(),
        gpu: false,
        exec: String::new(),
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((k, v)) = line.split_once('=') else { continue };
        let v = v.trim().replace("$DISK", &disk);
        match k.trim() {
            "name" => a.name = v,
            "runtime" => match v.as_str() {
                "linux" => a.runtime = Runtime::Linux,
                _ => return None,
            },
            "mount" => a.mount = Some(v),
            "copy" => {
                let mut w = v.split_whitespace();
                if let (Some(from), Some(to)) = (w.next(), w.next()) {
                    a.copies.push((from.to_string(), to.to_string()));
                }
            }
            "cwd" => a.cwd = Some(v),
            "env" => a.env.push(v),
            "gpu" => a.gpu = matches!(v.as_str(), "yes" | "on" | "true" | "1"),
            "exec" => a.exec = v,
            _ => {}
        }
    }
    (!a.name.is_empty() && !a.exec.is_empty()).then_some(a)
}

/// Look again: every FAT volume's `apps/`, then `/etc/apps/`.
pub fn scan() -> usize {
    let mut found = Vec::new();
    if let Ok(layout) = crate::store::block::scan() {
        for p in layout.partitions.iter() {
            let root = disk_root(p.index);
            if crate::linux::disk::mount(&root, p.index).is_err() {
                continue;
            }
            let dir = alloc::format!("{}/apps", root);
            for (name, is_dir, _) in crate::linux::disk::entries(&dir) {
                if is_dir || !name.to_ascii_lowercase().ends_with(".app") {
                    continue;
                }
                let path = alloc::format!("{}/{}", dir, name);
                if let Some(bytes) = crate::linux::disk::read_all(&path) {
                    if let Some(a) = parse(&String::from_utf8_lossy(&bytes), Some(p.index)) {
                        found.push(a);
                    }
                }
            }
        }
    }
    for (name, is_dir, _) in crate::sysbox::listing("/etc/apps") {
        if is_dir {
            continue;
        }
        if let Some(bytes) = crate::sysbox::read_blob(&alloc::format!("/etc/apps/{}", name)) {
            if let Some(a) = parse(&String::from_utf8_lossy(&bytes), None) {
                found.push(a);
            }
        }
    }
    // One entry per name: a disk carrying two copies of a game offers it once.
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found.dedup_by(|a, b| a.name == b.name);
    let n = found.len();
    *unsafe { APPS.get() } = found;
    GENERATION.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    n
}

pub fn list() -> Vec<App> {
    unsafe { APPS.get() }.clone()
}

pub fn names() -> Vec<String> {
    unsafe { APPS.get() }.iter().map(|a| a.name.clone()).collect()
}

/// The environment that puts a GL program on the GPU: Zink over Venus,
/// presenting through Wayland shared memory. Only when a GPU is there --
/// without one the program gets software rendering, which is slow and works.
fn gpu_env(a: &App) -> Vec<String> {
    if !a.gpu || crate::linux::drm::present().is_none() {
        return Vec::new();
    }
    let icd = if a.exec_is_32bit() { "/lib32/virtio_icd32.json" } else { "/vk/virtio_icd.json" };
    let root = a.mount.clone().or_else(|| a.partition.map(disk_root)).unwrap_or_default();
    alloc::vec![
        String::from("MESA_LOADER_DRIVER_OVERRIDE=zink"),
        String::from("MESA_VK_WSI_DEBUG=sw,buffer"),
        alloc::format!("VK_ICD_FILENAMES={}{}", root, icd),
    ]
}

impl App {
    /// Whether the program is an i386 ELF, which decides whose Vulkan driver
    /// it needs. Read off the file rather than declared, so an entry cannot
    /// be wrong about it.
    fn exec_is_32bit(&self) -> bool {
        let path = self.exec.split_whitespace().next().unwrap_or("");
        crate::linux::disk::program_bytes(path).is_some_and(|b| b.len() > 4 && b[4] == 1)
    }
}

/// Start an entry by name, beside the shell. Answers the job number.
pub fn launch(name: &str) -> Result<u32, String> {
    let a = unsafe { APPS.get() }
        .iter()
        .find(|a| a.name.eq_ignore_ascii_case(name))
        .cloned()
        .ok_or_else(|| alloc::format!("no installed program called '{}' -- 'apps' lists them", name))?;
    match a.runtime {
        Runtime::Linux => launch_linux(&a),
    }
}

fn launch_linux(a: &App) -> Result<u32, String> {
    if let (Some(at), Some(p)) = (&a.mount, a.partition) {
        crate::linux::disk::mount(at, p).map_err(String::from)?;
    }
    for (from, to) in &a.copies {
        let bytes = crate::linux::disk::program_bytes(from).ok_or_else(|| alloc::format!("{} is not there", from))?;
        if !crate::sysbox::write_blob(to, bytes) {
            return Err(alloc::format!("could not put {} at {}", from, to));
        }
    }
    let mut env = a.env.clone();
    env.extend(gpu_env(a));
    let mut words = a.exec.split_whitespace();
    let path = words.next().ok_or("nothing to run")?;
    let args: Vec<&str> = words.collect();
    // The environment and the directory are what the loader reads when the
    // job starts, so they are set here, immediately before it. Two launches
    // in the same instant could see each other's; one click at a time cannot.
    for e in &env {
        crate::linux::syscall::set_env(e);
    }
    if let Some(c) = &a.cwd {
        crate::linux::syscall::set_start_cwd(c);
    }
    crate::linux::start_job(path, &args).map_err(String::from)
}

/// What `diag apps` checks, with no disk.
pub fn checks() -> Vec<(&'static str, bool)> {
    let text = "# a comment\nname=Game\nruntime=linux\nmount=/mnt\nenv=A=1\nenv=B=2\ncopy=/mnt/x /lib/x\ngpu=yes\nexec=$DISK/bin/game -w 1\n";
    let a = parse(text, Some(3));
    let mut out = Vec::new();
    out.push(("an entry with a name and something to run is read", a.is_some()));
    let a = a.unwrap_or_else(|| parse("name=x\nexec=y", None).unwrap());
    out.push(("repeated keys accumulate rather than replace", a.env.len() == 2));
    out.push(("$DISK becomes the volume's own mount point", a.exec == "/disk/3/bin/game -w 1"));
    out.push(("a copy is a source and a destination", a.copies.len() == 1 && a.copies[0].1 == "/lib/x"));
    out.push(("a runtime this machine lacks is not offered", parse("name=x\nruntime=nt\nexec=y", None).is_none()));
    out.push(("nor is an entry with nothing to run", parse("name=x", None).is_none()));
    out
}
