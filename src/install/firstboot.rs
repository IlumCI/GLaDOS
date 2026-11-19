//! What a Netinstall leaves for the installed system's first boot.
//!
//! The Netinstall image carries a kernel and a root that can run pacman, and
//! nothing else: no desktop toolkits, no office suite, no 600 MB of weights.
//! What the person chose in the installer -- package sets, and whether to
//! have the model (default yes) -- cannot be fetched by the install medium
//! into a disk it is still laying out, so it is written down instead:
//! `GLADOS\FIRSTBOOT.TXT` on the new ESP. The installed system's first boot
//! reads it and runs `install finish`, which goes online, installs the sets
//! and the model package at the image's pinned snapshot, copies the model
//! onto the ESP for the next boot, and removes the file. Anything that fails
//! leaves the file, and the shell says how to try again.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub const UEFI_PATH: &str = "\\GLADOS\\FIRSTBOOT.TXT";
pub const FAT_PATH: &str = "/GLADOS/FIRSTBOOT.TXT";
const HEADER: &str = "glados-firstboot 1";
pub const MODEL_PKG: &str = "glados-model-qwen3-0.6b";
pub const MODEL_DIR: &str = "/usr/share/glados/models/qwen3-0.6b";

/// The sets the installer offers, as the [glados] repository names them.
pub const SETS: [(&str, &str); 4] = [
    ("minimal", "the shell, a terminal and pacman -- nothing more"),
    ("desktop", "GTK and Qt programs, fonts, an editor, an image viewer"),
    ("office", "LibreOffice, which the model can write into"),
    ("media", "mpv, for video and sound"),
];

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub sets: Vec<String>,
    pub model: bool,
}

impl Plan {
    /// The packages `install finish` asks pacman for.
    pub fn packages(&self) -> Vec<String> {
        let mut v: Vec<String> =
            self.sets.iter().filter(|s| *s != "minimal").map(|s| format!("glados-set-{}", s)).collect();
        if self.model {
            v.push(String::from(MODEL_PKG));
        }
        v
    }

    pub fn render(&self) -> String {
        format!("{}\nsets {}\nmodel {}\n", HEADER, self.sets.join(" "), if self.model { "yes" } else { "no" })
    }
}

/// Parse a FIRSTBOOT file: the header, then `sets` and `model` lines. A set
/// the installer does not offer is refused rather than passed to pacman.
pub fn parse(text: &str) -> Option<Plan> {
    let mut lines = text.lines();
    if lines.next()?.trim() != HEADER {
        return None;
    }
    let mut plan = Plan { sets: Vec::new(), model: false };
    for l in lines {
        match l.split_once(' ') {
            Some(("sets", v)) => {
                for s in v.split_whitespace() {
                    if !SETS.iter().any(|(n, _)| *n == s) {
                        return None;
                    }
                    plan.sets.push(String::from(s));
                }
            }
            Some(("model", v)) => plan.model = v.trim() == "yes",
            _ => {}
        }
    }
    Some(plan)
}

// ---------------------------------------------------------------------------

use crate::sync::Spin;

/// What the installer will write, chosen in the wizard or by `install sets` /
/// `install model`. `None` until chosen: an install from Base media, which
/// carries everything, writes no file.
static CHOICE: Spin<Option<Plan>> = Spin::new(None);

/// The FIRSTBOOT this boot found on its own ESP, read before ExitBootServices.
static FOUND: crate::sync::Racy<Option<&'static [u8]>> = crate::sync::Racy::new(None);

pub fn found(b: &'static [u8]) {
    unsafe { *FOUND.get() = Some(b) };
}

pub fn pending() -> Option<Plan> {
    let b = unsafe { *FOUND.get() }?;
    parse(core::str::from_utf8(b).ok()?)
}

/// What will be written. Unchosen on a medium without the model, that is a
/// Netinstall, the default is the model and nothing else -- "ask, default
/// yes" -- so an install typed by hand gets what the wizard would offer.
pub fn choice() -> Option<Plan> {
    let c = CHOICE.lock().clone();
    c.or_else(|| (!medium_has_model()).then(|| Plan { sets: alloc::vec![String::from("minimal")], model: true }))
}

pub fn set_choice(p: Option<Plan>) {
    *CHOICE.lock() = p;
}

/// The file to write onto a new ESP, if anything was chosen.
pub fn text() -> Option<String> {
    choice().map(|p| p.render())
}

/// Whether this install medium lacks the model, so the wizard asks.
pub fn medium_has_model() -> bool {
    super::carried().iter().any(|(p, _)| p == "GLADOS/model.bin")
}

/// Run a shell line in the guest to its end, answering its exit status and
/// the last lines it printed.
fn guest(line: &str, secs: u64) -> Result<(i32, String), String> {
    if !crate::linux::disk::root_mounted() {
        return Err("no system image is mounted".into());
    }
    let _ = crate::linux::ns::detach("/tmp/install/rc");
    let _ = crate::linux::ns::detach("/tmp/install/out");
    crate::sysbox::make_dir("/tmp/install");
    let script = format!("#!/bin/bash\n( {}\n) > /tmp/install/out 2>&1 < /dev/null\necho $? > /tmp/install/rc\n", line);
    crate::sysbox::write_blob("/tmp/install/run.sh", script.into_bytes());
    let id = crate::linux::start_job_env_quiet("/usr/bin/bash", &["/tmp/install/run.sh"], &["HOME=/root", "LANG=C.UTF-8"])
        .map_err(String::from)?;
    let deadline = crate::dev::lapic::ticks() + secs * crate::TIMER_HZ as u64;
    loop {
        if crate::job::list().into_iter().any(|r| r.id == id && r.state == "done") {
            break;
        }
        if crate::dev::lapic::ticks() > deadline {
            let _ = crate::job::kill(id);
            return Err(format!("it did not finish in {} s", secs));
        }
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 5);
    }
    let read = |p: &str| crate::sysbox::read_blob(p).map(|b| String::from_utf8_lossy(&b).into_owned());
    let rc = read("/tmp/install/rc").and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(-1);
    let out = read("/tmp/install/out").unwrap_or_default();
    let tail: Vec<&str> = out.lines().rev().take(6).collect();
    Ok((rc, tail.into_iter().rev().collect::<Vec<_>>().join("\n")))
}

/// A guest file's bytes, from the layer or the image.
fn guest_file(path: &str) -> Option<Vec<u8>> {
    crate::linux::ns::read_blob(path).or_else(|| crate::linux::disk::read_all(path))
}

/// Copy a model package's files onto this machine's ESP, where the firmware
/// reads them at the next boot. One copy in memory at a time: the file is
/// written, dropped, then read back and hashed against the digest taken
/// before it was dropped.
pub fn model_install(dir: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let esp = crate::update::stage::find_esp()?;
    for (src, dst) in [("tokenizer.bin", "/GLADOS/tokenizer.bin"), ("model.bin", "/GLADOS/model.bin")] {
        let path = format!("{}/{}", dir, src);
        let data = guest_file(&path).ok_or_else(|| format!("{} is not installed", path))?;
        let want = crate::store::sha256::hash(&data);
        let len = data.len();
        say(&format!("  model   {} ({} MiB) onto the boot partition", src, len >> 20));
        if !crate::dev::nvme::claim(crate::dev::nvme::CONFIRM, crate::dev::nvme::Holder::Esp, esp.start_lba, esp.blocks) {
            return Err("the write gate refused the boot partition".into());
        }
        let r = crate::store::fatw::put(&esp.volume, dst, &data);
        crate::dev::nvme::release(crate::dev::nvme::Holder::Esp);
        r.map_err(|e| format!("{}: {}", dst, e))?;
        drop(data);
        let ent = esp.volume.find(dst).map_err(|_| format!("{} is not there after writing", dst))?;
        let back = esp.volume.read_file(&ent).map_err(|_| format!("{} does not read back", dst))?;
        if back.len() != len || crate::store::sha256::hash(&back) != want {
            return Err(format!("{} read back differently", dst));
        }
    }
    say("  model   copied and read back -- it is used from the next restart");
    Ok(())
}

/// The first boot's work: online, the packages, the model, then forget.
pub fn finish(say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let plan = pending().ok_or("there is nothing left from the install to finish")?;
    crate::update::fetch::online().map_err(|e| format!("not online yet ({}) -- connect, then 'install finish'", e))?;
    let pkgs = plan.packages();
    if !pkgs.is_empty() {
        say(&format!("  packages {} -- from the snapshot this system is pinned to", pkgs.join(" ")));
        let (rc, out) = guest(&format!("pacman -Sy --needed --noconfirm {}", pkgs.join(" ")), 3600)?;
        if rc != 0 {
            return Err(format!("pacman answered {}:\n{}", rc, out));
        }
    }
    if plan.model {
        // Room first: pacman's cached package is another 550 MB the layer
        // would hold, and a commit moves what was installed out of the heap
        // onto the store, so the copy below is the only large thing resident.
        let _ = guest("rm -f /var/cache/pacman/pkg/glados-model-*", 60);
        let _ = crate::sysbox::commit_now();
        model_install(MODEL_DIR, say)?;
    }
    let esp = crate::update::stage::find_esp()?;
    crate::update::stage::remove_some(&esp, &[FAT_PATH])?;
    unsafe { *FOUND.get() = None };
    let _ = crate::sysbox::commit_now();
    say("  finished -- restart to use the model");
    Ok(())
}

pub fn checks() -> Vec<(String, bool)> {
    let mut c: Vec<(String, bool)> = Vec::new();
    let mut p = |s: &str, ok: bool| c.push((String::from(s), ok));
    let plan = Plan { sets: alloc::vec![String::from("desktop"), String::from("office")], model: true };
    p("a first-boot plan round-trips through its file", parse(&plan.render()) == Some(plan.clone()));
    p("its packages are the sets' meta packages and the model's", plan.packages() == ["glados-set-desktop", "glados-set-office", MODEL_PKG]);
    p("minimal asks pacman for nothing", Plan { sets: alloc::vec![String::from("minimal")], model: false }.packages().is_empty());
    p("a set the installer does not offer is refused, not handed to pacman", parse("glados-firstboot 1\nsets desktop; rm -rf /\n").is_none());
    p("a file without the header is not a plan", parse("sets desktop\n").is_none());
    c
}
