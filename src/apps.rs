//! Installed programs: what the desktop offers, and how each one starts.
//!
//! A program written for another system is a file plus a recipe -- which
//! volume to mount where, which interpreter to put where its headers look,
//! which environment, which directory to start in, what to run. Typing that
//! recipe is what running a 32-bit game meant until now, nine shell lines long. An
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
//! name=Game
//! runtime=linux
//! mount=/mnt                 # the volume this file was found on, mounted here
//! copy=/mnt/lib32/ld-linux.so.2 /lib/ld-linux.so.2
//! cwd=/mnt/game
//! env=LD_LIBRARY_PATH=/mnt/game/bin:/mnt/lib32
//! gpu=yes                    # Zink over Venus when a GPU is there
//! exec=/mnt/game/game -fullscreen
//! icon=$DISK/game/icon.bmp
//! ```
//!
//! `$DISK` in any value is the volume's own mount point, for an entry that
//! does not want to pick one.

use crate::gfx::scale::Image;
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
    /// A BMP on the volume, read at scan time. The picture stays on the disk
    /// it came with; nothing of a program's art is copied into this tree.
    pub icon: Option<String>,
}

static APPS: Racy<Vec<App>> = Racy::new(Vec::new());
/// One per entry in `APPS`, in the same order, decoded once at scan so the
/// desktop pays a scale and never a parse when it paints.
static PICTURES: Racy<Vec<Option<Image>>> = Racy::new(Vec::new());
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
        icon: None,
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
            "icon" => a.icon = Some(v),
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
    let pictures = found
        .iter()
        .map(|a| a.icon.as_deref().and_then(crate::linux::disk::read_all).and_then(|b| bmp(&b)))
        .collect();
    *unsafe { PICTURES.get() } = pictures;
    *unsafe { APPS.get() } = found;
    GENERATION.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    n
}

pub fn list() -> Vec<App> {
    unsafe { APPS.get() }.clone()
}

/// Hand the k-th program's picture, if it has one, to `f`.
pub fn with_picture<R>(k: usize, f: impl FnOnce(&Image) -> R) -> Option<R> {
    unsafe { PICTURES.get() }.get(k)?.as_ref().map(f)
}

/// Decode an uncompressed 24- or 32-bit Windows bitmap into premultiplied
/// ARGB. That is the whole of what games ship their icons as; anything else
/// -- palettes, RLE, a size no icon has -- answers `None` and the program
/// keeps the generic pictogram rather than a picture of the wrong bytes.
///
/// A 32-bit bitmap whose alpha is zero everywhere is opaque, not invisible:
/// the BI_RGB form leaves that byte unused, and taking it at its word would
/// draw nothing for every icon written by a tool that does not fill it.
pub fn bmp(b: &[u8]) -> Option<Image> {
    let u16_at = |o: usize| Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?));
    let u32_at = |o: usize| Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?));
    if b.get(0..2)? != b"BM" {
        return None;
    }
    let off = u32_at(10)? as usize;
    let w = u32_at(18)? as i32;
    let h = u32_at(22)? as i32;
    let bpp = u16_at(28)? as usize;
    let compression = u32_at(30)?;
    // 0 is BI_RGB, whose 32-bit layout is fixed as B, G, R and a spare
    // byte. 3 is BI_BITFIELDS, whose masks say where each channel is -- and
    // they are not the BI_RGB ones in practice: a real game's own icon is stored
    // A, B, G, R, and assuming the usual order read its alpha as blue.
    if !(compression == 0 || compression == 3) || !(bpp == 24 || bpp == 32) {
        return None;
    }
    let header = u32_at(14)? as usize;
    let (mut masks, mut has_alpha) = ([0x00ff_0000u32, 0x0000_ff00, 0x0000_00ff, 0xff00_0000], bpp == 32);
    if compression == 3 {
        if bpp != 32 {
            return None;
        }
        // The masks follow a 40-byte header, or sit inside a longer one at
        // the same offset; a header of 56 or more also carries alpha's.
        masks = [u32_at(54)?, u32_at(58)?, u32_at(62)?, if header >= 56 { u32_at(66)? } else { 0 }];
        has_alpha = masks[3] != 0;
        // Eight bits a channel is what an icon is; any other width would
        // need scaling this does not do.
        if masks[..if has_alpha { 4 } else { 3 }].iter().any(|m| m.count_ones() != 8 || (m >> m.trailing_zeros()) != 0xff) {
            return None;
        }
    }
    let chan = |v: u32, m: u32| if m == 0 { 0 } else { (v & m) >> m.trailing_zeros() };
    if w <= 0 || h == 0 || w > 1024 || h.unsigned_abs() > 1024 {
        return None;
    }
    let (w, top_down, h) = (w as usize, h < 0, h.unsigned_abs() as usize);
    let stride = (w * bpp / 8 + 3) & !3;
    if b.len() < off + stride * h {
        return None;
    }
    let mut px = alloc::vec![0u32; w * h];
    let mut any_alpha = false;
    for row in 0..h {
        let src = off + stride * if top_down { row } else { h - 1 - row };
        for x in 0..w {
            let p = &b[src + x * bpp / 8..];
            let v = if bpp == 32 { u32::from_le_bytes([p[0], p[1], p[2], p[3]]) } else { u32::from_le_bytes([p[0], p[1], p[2], 0]) };
            let a = if has_alpha { chan(v, masks[3]) } else { 255 };
            any_alpha |= a != 0;
            px[row * w + x] = a << 24 | chan(v, masks[0]) << 16 | chan(v, masks[1]) << 8 | chan(v, masks[2]);
        }
    }
    for v in px.iter_mut() {
        let a = if any_alpha { *v >> 24 } else { 255 };
        let m = |c: u32| (c * a + 127) / 255;
        *v = a << 24 | m((*v >> 16) & 0xff) << 16 | m((*v >> 8) & 0xff) << 8 | m(*v & 0xff);
    }
    Some(Image { w: w as u32, h: h as u32, px })
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
    // **Once at a time.** A double-click that lands twice started a second
    // copy, and the second launch's mount replaced the volume under the
    // first while it was reading from it -- two copies of a 32-bit GPU game
    // took the whole machine down. A program already running is refused by
    // name, before anything is mounted or copied.
    let cmd = a.exec.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(j) = crate::job::list().iter().find(|j| j.state != "done" && j.title == cmd) {
        return Err(alloc::format!("{} is already running as job {}", a.name, j.id));
    }
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
    // A 2x2 bottom-up 32-bit bitmap: the first stored row is the image's
    // bottom, which is the half of the format most often got backwards.
    let mut b = alloc::vec![0u8; 54];
    b[0..2].copy_from_slice(b"BM");
    b[10..14].copy_from_slice(&54u32.to_le_bytes());
    b[18..22].copy_from_slice(&2i32.to_le_bytes());
    b[22..26].copy_from_slice(&2i32.to_le_bytes());
    b[28..30].copy_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(&[0, 0, 255, 255, 0, 0, 255, 255]); // bottom: opaque red
    b.extend_from_slice(&[255, 0, 0, 0, 255, 0, 0, 128]); // top: clear, half blue
    let img = bmp(&b);
    out.push(("a bitmap's first row is its bottom", img.as_ref().is_some_and(|i| i.px[2] == 0xffff0000)));
    out.push(("colour is premultiplied by its alpha", img.as_ref().is_some_and(|i| i.px[0] == 0 && i.px[1] == 0x80000080)));
    out.push(("a bitmap with no alpha anywhere is opaque", {
        let mut o = b.clone();
        for i in [57, 61, 65, 69] { o[i] = 0; }
        bmp(&o).is_some_and(|i| i.px.iter().all(|p| p >> 24 == 255))
    }));
    out.push(("bitfield masks are honoured, in the A-B-G-R order a real game's icon uses", {
        // A 56-byte header, masks R ff000000 G 00ff0000 B 0000ff00 A 000000ff,
        // one pixel of opaque pure blue: stored bytes are A, B, G, R.
        let mut o = alloc::vec![0u8; 70];
        o[0..2].copy_from_slice(b"BM");
        o[10..14].copy_from_slice(&70u32.to_le_bytes());
        o[14..18].copy_from_slice(&56u32.to_le_bytes());
        o[18..22].copy_from_slice(&1i32.to_le_bytes());
        o[22..26].copy_from_slice(&1i32.to_le_bytes());
        o[28..30].copy_from_slice(&32u16.to_le_bytes());
        o[30..34].copy_from_slice(&3u32.to_le_bytes());
        for (i, m) in [0xff00_0000u32, 0x00ff_0000, 0x0000_ff00, 0x0000_00ff].iter().enumerate() {
            o[54 + 4 * i..58 + 4 * i].copy_from_slice(&m.to_le_bytes());
        }
        o.extend_from_slice(&[255, 255, 0, 0]);
        bmp(&o).is_some_and(|i| i.px[0] == 0xff0000ff)
    }));
    out.push(("a palette bitmap is refused rather than misread", {
        let mut o = b.clone();
        o[28] = 8;
        bmp(&o).is_none()
    }));
    out
}
