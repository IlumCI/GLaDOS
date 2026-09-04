//! A command shell reading from the keyboard ring buffer.
//!
//! In TempleOS the shell and the compiler were the same thing: what you typed
//! was compiled and run. This is not that yet -- it dispatches on fixed command
//! names. M5 replaces `execute` with a tokeniser, a parser and a JIT, at which
//! point this file mostly disappears.

use crate::acpi::Acpi;
use crate::cpu::idt;
use crate::dev::{kbd, lapic};
use crate::gfx::console::{self, LTCYAN, LTGRAY, LTGREEN, LTRED, WHITE, YELLOW};
use crate::aiksi;
use crate::mem;
use crate::BootInfo;
use crate::{kprint, kprintln, serial_println};
use alloc::string::String;
use alloc::vec::Vec;

/// Public because the terminal window marks the rows that begin with it --
/// see `desk::prompt_gutter`. One spelling, so a changed prompt cannot leave
/// the gutter looking for a marker nothing prints.
pub const PROMPT: &str = "glados> ";
/// Derived, not written out. Hardcoding this was fine until the prompt changed
/// length during the rename, at which point every cursor position in the line
/// editor was off by one.
const PROMPT_LEN: usize = PROMPT.len();
const HISTORY_MAX: usize = 64;

fn prompt() {
    console::set_color(LTGREEN);
    kprint!("\n{}", PROMPT);
    console::set_color(WHITE);
}

/// Redraw the prompt after something else has printed over it.
///
/// A background task writing to the console leaves the cursor wherever its
/// output ended, and the line the operator was typing scrolled away with it.
/// This at least restores the prompt so the shell does not look dead.
pub fn reprompt() {
    prompt();
}

/// Say why a command did nothing, when the mind holds the engine.
///
/// The refusal itself is enforced inside `with_engine`, not here. This existed
/// as a guarded match arm sitting *after* the arms it guarded, so it was
/// unreachable and protected nothing; the compiler had been reporting it as an
/// unreachable pattern the whole time.
fn note_if_mind_busy() {
    if crate::ai::engine_held_by_mind() {
        console::set_color(YELLOW);
        kprintln!("  the mind is using the model -- try again when it finishes");
        console::set_color(WHITE);
    }
}

/// Repaint the edited line and place the cursor.
///
/// Deliberately console-only rather than `kprint!`: echoing every keystroke and
/// every redraw to the serial port would bury the log in partial lines. The
/// finished line is written to serial once, when Enter is pressed.
///
/// Scrolls horizontally rather than wrapping.
///
/// It used to assume the line fit on one row, which was true at 80 columns and
/// stopped being true when the terminal became a window 47 wide: the line
/// wrapped, `set_col` kept addressing the row the prompt started on, and the
/// echo went to pieces.
///
/// Scrolling rather than wrapping because the console can only address a
/// column, not a row -- there is no `set_row`. Following a wrapped line would
/// mean teaching the console to place a cursor in two dimensions for the
/// benefit of one caller. A window onto the line needs neither, and a shell
/// that scrolls its input is a shell every user has already met.
fn redraw(line: &str, cursor: usize) {
    console::with(|c| {
        let avail = c.cols().saturating_sub(PROMPT_LEN + 1);
        if avail == 0 {
            return;
        }
        // Keep the cursor on screen. Everything else follows from that.
        let off = cursor.saturating_sub(avail.saturating_sub(1));
        let end = (off + avail).min(line.len());
        let shown = &line[off..end];

        c.set_col(PROMPT_LEN);
        c.write_bytes(shown.as_bytes());
        // Blank the rest of the row unconditionally rather than tracking how
        // long the line used to be: it is 47 stores, and a stale tail is the
        // one artefact that makes an editor look broken.
        for _ in shown.len()..avail {
            c.put_char(b' ');
        }
        c.set_col(PROMPT_LEN + (cursor - off));
    });
}

/// Hold a full-screen picture until a key, or for `ms` if one is given.
///
/// The bounded form is the only one a test can use: `drive.py` sends the next
/// command when it sees a prompt, so a program that owns the screen until a
/// keypress never gives the prompt back, and the keystroke that would end it
/// is the one thing the harness cannot deliver.
fn wait_or(ms: u64) {
    // Throw away whatever is already queued before waiting for a key.
    //
    // Without this the keystroke that *starts* a full-screen program is the
    // one that ends it. The Enter closing `doom view` is still in the ring
    // when the first frame lands, so `pop_any` finds it immediately and the
    // picture is gone before anybody has seen it -- which reads as a program
    // that drew nothing rather than as one that exited. It showed up here as
    // three screenshot runs out of five coming back with the desktop on
    // screen and the prompt already returned, and it is a real defect on the
    // keyboard too, not an artefact of driving this over a serial line.
    while crate::dev::kbd::pop_any().is_some() {}
    let until = crate::port::clock::now_ms() + ms;
    loop {
        if crate::dev::kbd::pop_any().is_some() {
            break;
        }
        if ms != 0 && crate::port::clock::now_ms() >= until {
            break;
        }
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
    }
}

/// The first map marker in a WAD: an empty lump named like a map.
///
/// By name rather than by emptiness, because plenty of legitimate lumps are
/// empty and only these two spellings have ever marked a map.
fn first_map(w: &crate::doom::wad::Wad) -> alloc::string::String {
    for i in 0..w.len() {
        if let Some(e) = w.at(i) {
            let n = e.name.as_str();
            let looks = (n.len() == 4 && n.starts_with('E') && n.contains('M'))
                || (n.len() == 5 && n.starts_with("MAP"));
            if looks && e.is_empty() {
                return alloc::string::String::from(n);
            }
        }
    }
    alloc::string::String::new()
}

static INTERACTIVE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Whether the shell has reached its prompt at least once.
///
/// The resident mind waits on this rather than on a stopwatch. `uptime` at
/// the first prompt reads 21 s under the hypervisor accelerator against about
/// 150 s under TCG, so any fixed grace period is a guess that is wrong on one
/// of them by a factor of seven; what the mind actually needs to know is that
/// a person could have typed by now.
pub fn interactive() -> bool {
    INTERACTIVE.load(core::sync::atomic::Ordering::Acquire)
}

/// Shell commands `find` will run without being told they are commands.
///
/// Not every arm of `execute` -- only the ones somebody would plausibly type
/// into a search box expecting something to happen. Anything not here and not
/// an applet is treated as a request for an application that does not exist
/// yet, which is the case worth being generous about.
/// Resolve a core by hash or by an unambiguous prefix of one.
///
/// Eight characters is what every other address in this system is printed as,
/// so it has to be what one can be typed as. An ambiguous prefix resolves to
/// nothing rather than to the first match -- installing the wrong core because
/// two shared four bytes would be a very quiet mistake.
fn find_core(want: &str) -> Option<[u8; 32]> {
    if let Some(h) = crate::ai::voter::unhex(want) {
        return Some(h);
    }
    let mut hit = None;
    for name in crate::sysbox::children(crate::ai::voter::ROOT) {
        if name.len() == 64 && name.starts_with(want) {
            if hit.is_some() {
                return None;
            }
            hit = crate::ai::voter::unhex(&name);
        }
    }
    hit
}

const KNOWN_COMMANDS: &[&str] = &[
    "term", "todo", "paint", "write", "mines", "oracle", "enternet", "net", "dhcp", "mem",
    "uptime", "tasks", "status", "help", "app", "author", "video", "serial", "log", "snap",
    "update", "gpu", "abstract", "study", "work",
];

/// How many steps an authoring run gets.
///
/// Each step is a prefill plus a few short decodes, and the engine is taken
/// and released once per decode -- so the shell waits seconds at a time rather
/// than for the whole run. Small, because a run that cannot finish in this many
/// steps is one whose contract no skeleton serves, and saying so quickly is
/// more use than grinding.
const AUTHOR_STEPS: usize = 8;

pub fn run(boot: &BootInfo, acpi: &Option<Acpi>) -> ! {
    console::set_color(LTCYAN);
    INTERACTIVE.store(true, core::sync::atomic::Ordering::Release);
    crate::gfx::desk::dismiss_menus();
    kprintln!("\ninteractive. type 'help', or just type code.");
    console::set_color(WHITE);

    let mut interp = aiksi::Interp::new();
    let mut history: Vec<String> = Vec::new();
    let mut line = String::new();
    let mut cursor = 0usize;
    // Equal to history.len() means "editing a fresh line", not browsing.
    let mut hist = 0usize;
    let mut stash = String::new();

    prompt();

    loop {
        // A window may have asked for a command. Run it as though it had been
        // typed, so there is exactly one path from "a command should run" to
        // "a command ran", whether it came from the keyboard or from a panel.
        if let Some(cmd) = crate::gfx::desk::take_pending() {
            // `kprintln!` already mirrors to serial, so the explicit
            // `serial_println!` the typed path uses would echo this twice --
            // the typed path needs it precisely because the line editor draws
            // to the console only.
            console::with(|c| c.set_col(PROMPT_LEN + line.len()));
            kprintln!("{}", cmd);
            console::resume_pacing();
            if !run_pipeline(&cmd, boot, acpi, &mut interp) {
                execute(&cmd, boot, acpi, &mut interp);
            }
            crate::sysbox::autosnap_poll();
            note_if_mind_busy();
            line.clear();
            cursor = 0;
            prompt();
            // The cheap tier is a second stale by design, which is right
            // for a repaint and wrong immediately after a command that
            // changed exactly what it caches. `invalidate` was written for
            // this and its doc says "called after a command"; it had no
            // caller but its own selftest.
            crate::ai::glance::invalidate();
            crate::gfx::desk::refresh_routed();
            crate::gfx::desk::refresh_status();
            crate::gfx::desk::redraw_over_terminal();
            continue;
        }

        let raw = if let Some(k) = kbd::pop_any() {
            k
        } else {
            // Nothing queued: give the network a slice, then idle until the
            // next interrupt rather than spinning. The timer alone wakes us
            // 100 times a second, so this is also the stack's clock -- an
            // open connection only advances between keystrokes.
            crate::net::tcp::service();
            // USB is polled and not interrupt-driven in this kernel, so a
            // keyboard on it is only heard from when somebody asks. Here
            // rather than in the timer tick for the same reason the pointer
            // is: a keystroke can raise a window.
            crate::dev::usbhid::poll();
            // The pointer is read from the idle loop rather than acted on in
            // the interrupt: a click raises a window and repaints, and doing
            // that from an ISR would redraw the screen underneath whatever was
            // drawing when the mouse moved.
            crate::gfx::desk::poll_mouse();
            unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
            continue;
        };

        // The desktop gets first refusal. It takes Alt-Tab always, and every
        // key when a window other than the terminal has focus -- which is what
        // makes the terminal a window on the desktop rather than the desktop a
        // thing the terminal occasionally draws.
        // The desktop gets first refusal only for keys a person actually
        // pressed. A byte off the serial line is by definition addressed to
        // the shell, and letting the desktop swallow it means a driven session
        // that ends up in a menu can never leave one: the bytes that would
        // dismiss it are eaten by it. `win keys` remains the way to drive the
        // desktop headlessly, and it comes through here as a command.
        let key = if crate::dev::kbd::last_was_serial() {
            raw
        } else {
            match crate::gfx::desk::key(raw) {
                crate::gfx::desk::Route::Handled => continue,
                crate::gfx::desk::Route::Shell(k) => k,
            }
        };

        match key {
            b'\n' => {
                console::with(|c| {
                    let avail = c.cols().saturating_sub(PROMPT_LEN + 1);
                    c.set_col(PROMPT_LEN + line.len().min(avail));
                });
                kprintln!();
                // The one place the typed line reaches the serial log.
                serial_println!("{}{}", PROMPT, line);

                let trimmed = line.trim();
                if !trimmed.is_empty() && history.last().map(|h| h.as_str()) != Some(trimmed) {
                    history.push(String::from(trimmed));
                    if history.len() > HISTORY_MAX {
                        history.remove(0);
                    }
                }
                // Re-arm pacing per command, so a skip requested during one
                // command's output does not silently disable it forever.
                console::resume_pacing();
                // A pipeline runs `execute` itself, with the console captured.
                if !run_pipeline(trimmed, boot, acpi, &mut interp) {
                    execute(trimmed, boot, acpi, &mut interp);
                }
                // Between commands is the only place the namespace is
                // guaranteed to be whole, so this is where an automatic
                // snapshot is allowed to run.
                crate::sysbox::autosnap_poll();
                // Anything needing the model will have quietly done nothing if
                // the mind had it; say so once, here, rather than at each of a
                // dozen call sites that would drift out of step.
                note_if_mind_busy();

                line.clear();
                cursor = 0;
                hist = history.len();
                stash.clear();
                prompt();
                // A command is the only thing that changes what a routed
                // window would show, so this is where they are rebuilt. Both
                // paths need it: the typed one and the one a panel's own
                // button takes.
                crate::gfx::desk::refresh_routed();
                crate::gfx::desk::refresh_status();
                // *After* the prompt, not before. Everything the console prints
                // -- including the prompt itself -- lands in the terminal's
                // rectangle without regard for what is drawn on top of it, so
                // the repair has to be the last thing that touches the screen.
                crate::gfx::desk::redraw_over_terminal();
            }

            8 => {
                if cursor > 0 {
                    cursor -= 1;
                    line.remove(cursor);
                    redraw(&line, cursor);
                }
            }
            kbd::KEY_DELETE => {
                if cursor < line.len() {
                    line.remove(cursor);
                    redraw(&line, cursor);
                }
            }

            kbd::KEY_LEFT => {
                if cursor > 0 {
                    cursor -= 1;
                    console::set_col(PROMPT_LEN + cursor);
                }
            }
            kbd::KEY_RIGHT => {
                if cursor < line.len() {
                    cursor += 1;
                    console::set_col(PROMPT_LEN + cursor);
                }
            }
            // Ctrl-A / Ctrl-E, for the same reason every other shell has them.
            kbd::KEY_HOME | 0x01 => {
                cursor = 0;
                console::set_col(PROMPT_LEN);
            }
            kbd::KEY_END | 0x05 => {
                cursor = line.len();
                console::set_col(PROMPT_LEN + cursor);
            }
            // Ctrl-U: scrap the line.
            0x15 => {
                line.clear();
                cursor = 0;
                redraw(&line, cursor);
            }

            kbd::KEY_UP => {
                if hist > 0 {
                    if hist == history.len() {
                        stash = line.clone();
                    }
                    hist -= 1;
                    line = history[hist].clone();
                    cursor = line.len();
                    redraw(&line, cursor);
                }
            }
            kbd::KEY_DOWN => {
                if hist < history.len() {
                    hist += 1;
                    line = if hist == history.len() {
                        stash.clone()
                    } else {
                        history[hist].clone()
                    };
                    cursor = line.len();
                    redraw(&line, cursor);
                }
            }

            ch if (0x20..0x7F).contains(&ch) => {
                line.insert(cursor, ch as char);
                cursor += 1;
                redraw(&line, cursor);
            }
            _ => {}
        }
    }
}

/// The in-OS updater's operator surface.
///
/// Several verbs rather than one `update now`. Claiming a write range on the
/// boot partition is the most dangerous thing this system does, and `fat
/// unlock` is already a separate deliberate act for exactly that reason.
/// Checking, downloading and staging are three different decisions, and an
/// operator gets to make them one at a time -- with the last one naming what
/// it is about to overwrite and waiting to be told the digest back.
fn update_cmd(rest: &str) {
    use crate::store::sha256;
    use crate::update::{channel, fetch, stage};

    let (verb, arg) = match rest.trim().split_once(' ') {
        Some((v, a)) => (v, a.trim()),
        None => (rest.trim(), ""),
    };

    // Eight hex characters of a digest: enough that typing it back is an act
    // rather than a reflex, and short enough that somebody will.
    fn digest8(data: &[u8]) -> alloc::string::String {
        let h = sha256::short_hex(&sha256::hash(data));
        let s = core::str::from_utf8(&h).unwrap_or("????????");
        alloc::string::String::from(&s[..8])
    }

    /// Fetch and verify the manifest for the channel in force.
    fn manifest_now() -> Option<crate::update::manifest::Manifest> {
        if let Err(e) = fetch::online() {
            kprintln!("  {}", e);
            return None;
        }
        let url = match channel::endpoint() {
            Ok(u) => u,
            Err(e) => {
                kprintln!("  {}", e);
                return None;
            }
        };
        kprintln!("  asking {}{}", url.host, url.path);
        let code = channel::code();
        match fetch::manifest_at(&url, code.as_deref()) {
            Ok(m) => Some(m),
            Err(e) => {
                console::set_color(LTRED);
                kprintln!("  {}", e);
                console::set_color(LTGRAY);
                None
            }
        }
    }

    match verb {
        "" | "status" => {
            console::set_color(YELLOW);
            kprintln!("[update]");
            console::set_color(LTGRAY);
            kprintln!("  running   {}", crate::VERSION);
            kprintln!("  channel   {}", channel::channel());
            match channel::source() {
                Some(s) => kprintln!("  source    {}", s),
                None => kprintln!("  source    none -- 'update source <url>' to set one"),
            }
            kprintln!(
                "  linked    {}",
                if channel::code().is_some() { "yes" } else { "no" }
            );
            if let Some(s) = channel::seen() {
                kprintln!("  last seen {}", s);
            }
            if let Some(img) = crate::sysbox::read_blob(channel::IMAGE) {
                kprintln!(
                    "  fetched   {} B waiting -- 'update stage {}'",
                    img.len(),
                    digest8(&img)
                );
            }
            match stage::find_esp() {
                Ok(e) => kprintln!("  boot vol  partition {}, lba {}", e.index, e.start_lba),
                Err(e) => kprintln!("  boot vol  {}", e),
            }
            if !crate::update::have_key() {
                console::set_color(LTRED);
                kprintln!("  no update key is compiled in, so every image would be refused");
                console::set_color(LTGRAY);
            }
            kprintln!("  check | fetch | stage | unstage | source | channel | link | verify");
        }

        "check" => {
            let Some(m) = manifest_now() else { return };
            channel::remember(&m.version, &m.notes);
            kprintln!("  {} offers {}", m.channel, m.version);
            if !m.notes.is_empty() {
                kprintln!("  {}", m.notes);
            }
            let h = sha256::short_hex(&m.sha256);
            kprintln!(
                "  {} B, sha256 {}..",
                m.size,
                core::str::from_utf8(&h).unwrap_or("?")
            );
            if m.is_upgrade() {
                console::set_color(LTGREEN);
                kprintln!("  newer than {} -- 'update fetch' to download it", crate::VERSION);
            } else {
                kprintln!("  not newer than {}, so there is nothing to do", crate::VERSION);
            }
            console::set_color(LTGRAY);
        }

        "fetch" => {
            let Some(m) = manifest_now() else { return };
            channel::remember(&m.version, &m.notes);
            // An older image is signed just as well as a newer one and always
            // will be, so this is the only place a rollback gets refused.
            if !m.is_upgrade() && arg != "--force" {
                kprintln!(
                    "  {} is not newer than {} -- 'update fetch --force' to take it anyway",
                    m.version,
                    crate::VERSION
                );
                return;
            }
            kprintln!("  downloading {} B, which is slow through a 32 KB window", m.size);
            match fetch::image_for(&m) {
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(LTGRAY);
                }
                Ok((image, sig)) => {
                    let confirm = digest8(&image);
                    let n = image.len();
                    if !crate::sysbox::write_blob(channel::IMAGE, image)
                        || !crate::sysbox::write_blob(channel::SIGNATURE, sig)
                    {
                        kprintln!("  could not hold the image in the namespace");
                        return;
                    }
                    console::set_color(LTGREEN);
                    kprintln!("  {} B verified against the manifest and the update key", n);
                    console::set_color(LTGRAY);
                    kprintln!("  'update stage {}' to arm the next boot", confirm);
                }
            }
        }

        "stage" => {
            let (Some(image), Some(sig)) = (
                crate::sysbox::read_blob(channel::IMAGE),
                crate::sysbox::read_blob(channel::SIGNATURE),
            ) else {
                kprintln!("  nothing has been fetched -- 'update fetch' first");
                return;
            };

            // Checked again here rather than trusted from `fetch`. These blobs
            // live in a writable namespace between the two commands, and the
            // whole point of the key is that it is asked every time.
            let v = crate::update::verify(&image, &sig);
            if !v.ok() {
                console::set_color(LTRED);
                kprintln!("  {}", v.why());
                console::set_color(LTGRAY);
                return;
            }

            let confirm = digest8(&image);
            if arg != confirm {
                console::set_color(WHITE);
                kprintln!("  about to stage {} B, sha256 {}..", image.len(), confirm);
                console::set_color(LTGRAY);
                kprintln!("  this replaces the boot image at the next boot, keeping the");
                kprintln!("  current one as BOOTX64.OLD to fall back to.");
                kprintln!("  'update stage {}' to confirm", confirm);
                return;
            }

            match stage::stage(&image, &sig) {
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(LTGRAY);
                }
                Ok(msg) => {
                    console::set_color(LTGREEN);
                    kprintln!("  {}", msg);
                    console::set_color(LTGRAY);
                    // The held copy is a few megabytes and autosnap would park
                    // it in an append-only store on the next tick.
                    crate::sysbox::detach(channel::IMAGE);
                    crate::sysbox::detach(channel::SIGNATURE);
                    kprintln!("  'reboot' when ready; 'update unstage' to call it off");
                }
            }
        }

        "unstage" => match stage::unstage() {
            Ok(msg) => kprintln!("  {}", msg),
            Err(e) => kprintln!("  {}", e),
        },

        "source" => {
            if arg.is_empty() {
                match channel::source() {
                    Some(s) => kprintln!("  {}", s),
                    None => kprintln!("  none -- 'update source <url>' to set one"),
                }
                kprintln!("  the manifest is signed, so this is where to ask, not who to trust");
                return;
            }
            match channel::set_source(arg) {
                Ok(()) => kprintln!("  source is {}", arg),
                Err(e) => kprintln!("  {}", e),
            }
        }

        "channel" => {
            if arg.is_empty() {
                kprintln!("  {}", channel::channel());
                kprintln!("  stable is free; experimental needs a linked device code");
                return;
            }
            match channel::set_channel(arg) {
                Ok(()) => kprintln!("  channel is {}", arg),
                Err(e) => kprintln!("  {}", e),
            }
        }

        "link" => {
            if arg.is_empty() {
                kprintln!("  usage: update link <code>");
                kprintln!("  get one at aperture.institute; it gates the experimental channel");
                return;
            }
            if channel::set_code(arg) {
                kprintln!("  linked; 'update channel experimental' to use it");
            } else {
                kprintln!("  could not store the code");
            }
        }

        "unlink" => {
            if channel::unlink() {
                kprintln!("  unlinked");
            } else {
                kprintln!("  nothing was linked");
            }
        }

        // The offline path, and the only one that worked before there was a
        // server: verify a pair somebody brought in by hand with `fat get`.
        "verify" => {
            let mut w = Words::new(arg);
            let img = match w.next() {
                Some(a) if !a.is_empty() => alloc::string::String::from(a),
                _ => alloc::string::String::from(channel::IMAGE),
            };
            let sig = match w.next() {
                Some(a) if !a.is_empty() => alloc::string::String::from(a),
                _ => alloc::format!("{}.sig", img),
            };
            let (Some(image), Some(signature)) =
                (crate::sysbox::read_blob(&img), crate::sysbox::read_blob(&sig))
            else {
                kprintln!("  usage: update verify <image> [signature]");
                kprintln!("  reads both from the namespace ('fat get' brings them in)");
                kprintln!("  looked for {} and {}", img, sig);
                return;
            };
            let v = crate::update::verify(&image, &signature);
            console::set_color(if v.ok() { LTGREEN } else { LTRED });
            kprintln!("  {}", v.why());
            console::set_color(LTGRAY);
            kprintln!("  {} B image, {} B signature", image.len(), signature.len());
            if v.ok() {
                kprintln!("  'update stage {}' to arm the next boot", digest8(&image));
            }
        }

        _ => {
            kprintln!("  no 'update {}'", verb);
            kprintln!("  update              what is running and what was last seen");
            kprintln!("  update check        ask the channel what it offers");
            kprintln!("  update fetch        download it, verify it, hold it");
            kprintln!("  update stage <hex>  write it to the boot volume");
            kprintln!("  update unstage      call off a staged update");
            kprintln!("  update source <url> | channel <name> | link <code> | unlink");
            kprintln!("  update verify <image> [sig]   check a pair brought in by hand");
        }
    }
}

/// The workflow graph: what runs exist, and what one of them did.
///
/// Read-only. Nothing here starts a run, because there is nothing to start
/// yet: this is the graph half, built and tested before any worker exists.
fn work_cmd(rest: &str) {
    use crate::ai::work::{self, Status};

    let rest = rest.trim();
    if rest.is_empty() {
        let runs = work::runs();
        if runs.is_empty() {
            kprintln!("  no runs");
            kprintln!("  a run is a plan, its steps, and what it built, under /ai/work");
            return;
        }
        console::set_color(YELLOW);
        kprintln!("[work] {} run(s)", runs.len());
        console::set_color(WHITE);
        for r in runs {
            let n = work::steps(&r).len();
            // The address, because it is the whole point: two runs that did
            // the same thing show the same eight characters here.
            let root = work::root(&r)
                .map(|h| alloc::string::String::from(&crate::ai::voter::hex(&h)[..16]))
                .unwrap_or_else(|| String::from("--------"));
            let goal = work::plan(&r).map(|p| p.goal).unwrap_or_default();
            kprintln!("  {:20} {:>3} step(s)  {}  {}", r, n, root, goal);
        }
        return;
    }

    // `work new <run> <goal>` -- a run with one step, which is what there is
    // to make until a manager exists to decompose anything.
    if let Some(arg) = rest.strip_prefix("new ") {
        let (name, goal) = match arg.trim().split_once(' ') {
            Some((n, g)) => (n.trim(), g.trim()),
            None => {
                kprintln!("  usage: work new <run> <goal>");
                return;
            }
        };
        let plan = work::Plan {
            goal: alloc::string::String::from(goal),
            // Attended. Autonomy is declared afterwards, deliberately, so
            // that granting it is a second decision about a plan somebody has
            // read rather than a flag on the command that created it.
            autonomy: work::Autonomy::Attended,
            steps: alloc::vec![work::PlanStep {
                id: 1,
                parent: None,
                status: work::Status::Todo,
                role: alloc::string::String::from("worker"),
                // No pre-decided action: `work new` makes a step the worker
                // decides, which is the Stage 1 shape. `work plan` is the
                // manager, and fills this in.
                action: alloc::string::String::new(),
                goal: alloc::string::String::from(goal),
            }],
        };
        if work::set_plan(name, &plan) {
            kprintln!("  {}  one step, todo", name);
        } else {
            kprintln!("  could not write the plan");
        }
        return;
    }

    // `work plan <run> <goal>` -- the manager. One prefill, N decodes, and
    // every action written down so the workers that follow need no model.
    if let Some(arg) = rest.strip_prefix("plan ") {
        let Some((name, goal)) = arg.trim().split_once(' ') else {
            kprintln!("  usage: work plan <run> <goal>");
            return;
        };
        let (name, goal) = (name.trim(), goal.trim());
        if crate::ai::mind_busy() {
            kprintln!("  the mind is busy -- wait for it to finish first");
            return;
        }
        let trust = crate::ai::harness::Trust::ReadOnly;
        console::set_color(YELLOW);
        kprintln!("[work plan] {}  trust read-only", name);
        console::set_color(WHITE);
        match work::decompose(name, goal, trust, 4) {
            None => {
                console::set_color(LTRED);
                kprintln!("  the manager decided nothing");
                console::set_color(WHITE);
            }
            Some((n, calls)) => {
                kprintln!("  {} step(s) planned, {} decode call(s)", n, calls);
                kprintln!("  the workers that run them cost no model calls at all");
            }
        }
        return;
    }

    // `work autonomy <run> attended|unattended` -- the declaration half.
    if let Some(arg) = rest.strip_prefix("autonomy ") {
        let Some((name, word)) = arg.trim().split_once(' ') else {
            kprintln!("  usage: work autonomy <run> attended|unattended");
            return;
        };
        let (name, word) = (name.trim(), word.trim());
        let want = match word {
            "unattended" => work::Autonomy::Unattended,
            "attended" => work::Autonomy::Attended,
            _ => {
                kprintln!("  usage: work autonomy <run> attended|unattended");
                return;
            }
        };
        let Some(mut p) = work::plan(name) else {
            console::set_color(LTRED);
            kprintln!("  no such run: {}", name);
            console::set_color(WHITE);
            return;
        };
        p.autonomy = want;
        if !work::set_plan(name, &p) {
            kprintln!("  could not write the plan");
            return;
        }
        kprintln!("  {} is now {}", name, word);
        // Declaring is half of it, and the half a plan can do to itself. Say
        // the other half out loud rather than leaving an operator to find out
        // that nothing happened overnight.
        if want == work::Autonomy::Unattended {
            match work::intent_hash(name) {
                Some(h) => kprintln!(
                    "  it advances unattended once granted: work trust {} {}",
                    name,
                    &crate::ai::voter::hex(&h)[..8]
                ),
                None => kprintln!("  it advances unattended once granted"),
            }
        }
        return;
    }

    // `work check <run>` -- what a grant would be approving.
    if let Some(arg) = rest.strip_prefix("check ") {
        let name = arg.trim();
        let Some(f) = work::check(name) else {
            console::set_color(LTRED);
            kprintln!("  no such run: {}", name);
            console::set_color(WHITE);
            return;
        };
        console::set_color(YELLOW);
        kprintln!("[work check] {}", name);
        console::set_color(WHITE);
        let mark = |b: bool| if b { "pass" } else { "FAIL" };
        kprintln!("  declared    {}  (the plan says so)", mark(f.declared));
        match &f.refused {
            None => kprintln!("  admissible  pass  (every action a read-only worker can reach)"),
            Some(a) => kprintln!("  admissible  FAIL  ({})", a),
        }
        kprintln!("  acyclic     {}  (ids unique, every parent present)", mark(f.acyclic));
        kprintln!("  bounded     {}  ({} step(s), one per quiet tick)", mark(f.bounded), f.steps);
        if let Some(h) = work::intent_hash(name) {
            // The address a grant names, which is the plan without its
            // statuses. Printed so what was approved can be checked later.
            kprintln!("  intent      {}", crate::ai::voter::hex(&h));
        }
        if f.fit {
            console::set_color(LTGREEN);
            kprintln!("  fit to advance unattended");
        } else {
            console::set_color(LTRED);
            kprintln!("  not fit; a grant would be refused");
        }
        console::set_color(WHITE);
        return;
    }

    // `work trust <run> <hex8>` -- the operator's half.
    //
    // Eight characters of the intent typed back, the `update stage` idiom.
    // Shell-only and never an applet: `work` is absent from `sysbox::APPLETS`,
    // so no grammar can spell it and the model cannot reach this at all.
    if let Some(arg) = rest.strip_prefix("trust ") {
        let Some((name, typed)) = arg.trim().split_once(' ') else {
            kprintln!("  usage: work trust <run> <first 8 of the intent>");
            kprintln!("  `work check <run>` prints it");
            return;
        };
        let (name, typed) = (name.trim(), typed.trim());
        let Some(f) = work::check(name) else {
            console::set_color(LTRED);
            kprintln!("  no such run: {}", name);
            console::set_color(WHITE);
            return;
        };
        if !f.fit {
            console::set_color(LTRED);
            kprintln!("  refused: this plan is not fit to advance unattended");
            console::set_color(WHITE);
            kprintln!("  `work check {}` says which check failed", name);
            return;
        }
        let Some(h) = work::intent_hash(name) else {
            kprintln!("  the plan has no address");
            return;
        };
        let hex = crate::ai::voter::hex(&h);
        if typed.len() < 8 || !hex.starts_with(typed) {
            console::set_color(LTRED);
            kprintln!("  that is not this plan's intent");
            console::set_color(WHITE);
            kprintln!("  wanted the first 8 of {}", hex);
            return;
        }
        if work::grant(&h) {
            console::set_color(LTGREEN);
            kprintln!("  granted {}", hex);
            console::set_color(WHITE);
            // The property that makes the grant worth anything, said where
            // somebody will read it.
            kprintln!("  editing the plan revokes this, because the grant names its contents");
        } else {
            kprintln!("  could not write the grant");
        }
        return;
    }

    if let Some(arg) = rest.strip_prefix("untrust ") {
        let want = arg.trim();
        let all = work::grants();
        let hits: Vec<&String> = all.iter().filter(|g| g.starts_with(want)).collect();
        // Refused rather than resolved to the first match, the rule
        // `skill untrust` already follows.
        if hits.len() != 1 {
            console::set_color(LTRED);
            kprintln!("  {}", if hits.is_empty() { "no grant matches" } else { "ambiguous" });
            console::set_color(WHITE);
            return;
        }
        let mut h = [0u8; 32];
        let bytes = hits[0].as_bytes();
        for (i, b) in h.iter_mut().enumerate() {
            let hi = (bytes[i * 2] as char).to_digit(16).unwrap_or(0) as u8;
            let lo = (bytes[i * 2 + 1] as char).to_digit(16).unwrap_or(0) as u8;
            *b = (hi << 4) | lo;
        }
        if work::revoke(&h) {
            kprintln!("  revoked {}", hits[0]);
        } else {
            kprintln!("  could not write the grants");
        }
        return;
    }

    if rest == "trusted" {
        let all = work::grants();
        if all.is_empty() {
            kprintln!("  nothing is granted; no workflow advances unattended");
            return;
        }
        console::set_color(YELLOW);
        kprintln!("[work trusted] {}", all.len());
        console::set_color(WHITE);
        for g in all {
            // Which run still hashes to it, if any. A grant outliving its plan
            // is the ordinary case after an edit and is worth seeing.
            let owner = work::runs()
                .into_iter()
                .find(|r| work::intent_hash(r).map(|h| crate::ai::voter::hex(&h)) == Some(g.clone()));
            match owner {
                Some(r) => kprintln!("  {}  {}", &g[..16], r),
                None => kprintln!("  {}  (no run has this intent any more)", &g[..16]),
            }
        }
        return;
    }

    // `work night` -- take the step the quiet tick would take, now.
    //
    // The same call `initiative`'s sleep branch makes, so what an operator
    // sees here is what the night does. `godel next` exists for the same
    // reason and has to decline to act, because finding out would cost the
    // night; here a granted step is a dispatch and costs nothing, so this one
    // takes it.
    if rest == "night" {
        if crate::ai::mind_busy() {
            kprintln!("  the mind is busy -- wait for it to finish first");
            return;
        }
        console::set_color(YELLOW);
        kprintln!("[work night]");
        console::set_color(WHITE);
        match work::tick_unattended(true) {
            None => {
                kprintln!("  nothing advanced");
                kprintln!("  a run needs `work autonomy <run> unattended` and then `work trust`");
            }
            Some((name, decoded)) => {
                console::set_color(LTGREEN);
                kprintln!("  {} advanced one step, {}", name,
                          if decoded { "a decode" } else { "no model call" });
                console::set_color(WHITE);
            }
        }
        return;
    }

    // `work harvest` -- transcripts into role example sets.
    if rest == "harvest" {
        let runs = work::runs().len();
        let h = work::harvest();
        console::set_color(YELLOW);
        kprintln!("[work harvest] {} step(s) across {} run(s)", h.seen, runs);
        console::set_color(WHITE);
        // Every count, including what was thrown away. A harvest printing only
        // what it kept reads the same whether it found a rich set or dropped
        // four fifths of one.
        kprintln!("  {:>4} dropped, the manager decided it and not a worker", h.dispatched);
        kprintln!("  {:>4} dropped, the step failed", h.failed);
        kprintln!("  {:>4} dropped, a duplicate of one already kept", h.duplicate);
        if h.roles.is_empty() {
            console::set_color(LTRED);
            kprintln!("  nothing to train on");
            console::set_color(WHITE);
            kprintln!("  worker-decided steps come from `work new`; `work plan` pre-decides them");
            return;
        }
        for (name, n) in &h.roles {
            kprintln!("  {:16} {:>4} example(s)", name, n);
        }
        return;
    }

    // `work roles` -- what has been harvested, and what has been trained.
    if rest == "roles" {
        let names = work::role_names();
        if names.is_empty() {
            kprintln!("  no roles harvested; run `work harvest`");
            return;
        }
        console::set_color(YELLOW);
        kprintln!("[work roles] {}", names.len());
        console::set_color(WHITE);
        for name in names {
            let ex = work::role_examples(&name);
            let cut = work::role_split(&ex);
            let held = ex.len().saturating_sub(cut);
            let adapter = match work::role_adapter(&name) {
                Some(b) => alloc::format!("{} B", b.len()),
                None => String::from("none"),
            };
            kprintln!(
                "  {:16} {:>4} example(s)  {} run(s)  {} held  adapter {}",
                name, ex.len(), work::role_runs(&ex), held, adapter
            );
        }
        return;
    }

    // `work train <role>` -- the Stage 3 measurement, and the only one that
    // decides whether a role is a specialist or a word in a plan file.
    if let Some(arg) = rest.strip_prefix("train ") {
        let mut words = arg.split_whitespace();
        let Some(role) = words.next() else {
            kprintln!("  usage: work train <role> [-e epochs] [-r rank] [-lr rate]");
            return;
        };
        let mut b = crate::ai::train::Budget::default();
        let mut bad: Option<&str> = None;
        while let Some(flag) = words.next() {
            let value = words.next().unwrap_or("");
            match (flag, value.parse::<u64>()) {
                ("-e", Ok(v)) => b.epochs = v as usize,
                ("-ms", Ok(v)) => b.millis = v,
                ("-r", Ok(v)) => b.rank = v as usize,
                ("-lr", _) => match value.parse::<f32>() {
                    Ok(v) => b.lr = v,
                    Err(_) => bad = Some(value),
                },
                _ => bad = Some(flag),
            }
        }
        // No `-n`. It subsamples the corpus, and a role set is already the
        // small thing -- striding through forty examples to save time would
        // trade away the only evidence there is.
        if let Some(w) = bad {
            kprintln!("  unrecognised: {}", w);
            kprintln!("  usage: work train <role> [-e epochs] [-ms budget] [-r rank] [-lr rate]");
            return;
        }
        if crate::ai::mind_busy() {
            kprintln!("  the mind is busy -- wait for it to finish first");
            return;
        }
        console::set_color(YELLOW);
        kprintln!("[work train] {}", role);
        console::set_color(WHITE);
        match work::train_role(role, &b) {
            Err(work::RoleError::Unknown) => {
                console::set_color(LTRED);
                kprintln!("  no such role, or its set is empty");
                console::set_color(WHITE);
            }
            Err(work::RoleError::TooFew(n)) => {
                console::set_color(LTRED);
                kprintln!("  {} example(s): too few for a judge to work with", n);
                console::set_color(WHITE);
                // The arithmetic, said out loud, because the alternative is
                // printing a figure over a slice too small to carry one.
                kprintln!("  J1 wants a net repair of 4 on a quarter held out, so 16 is the floor");
            }
            Err(work::RoleError::OneRun) => {
                console::set_color(LTRED);
                kprintln!("  every example came from one run, so nothing can be held out");
                console::set_color(WHITE);
                kprintln!("  steps inside a run share a goal; splitting them measures memorisation");
            }
            Err(work::RoleError::OneClass) => {
                console::set_color(LTRED);
                kprintln!("  every example carries the same label");
                console::set_color(WHITE);
                kprintln!("  an adapter over one class learns a prior and scores 100% doing it");
            }
            Err(work::RoleError::NoEngine) => kprintln!("  {}", crate::ai::engine_refusal()),
            Err(work::RoleError::Train(e)) => {
                use crate::ai::train::RunError;
                console::set_color(LTRED);
                // Worded for a role set rather than for the corpus. "there is
                // no corpus" would be a true sentence about the wrong object.
                kprintln!("  {}", match e {
                    RunError::Hardware => "no AVX2/FMA, so this would measure the emulator",
                    RunError::Hybrid => "adapters are refused on a hybrid checkpoint",
                    RunError::NoCorpus => "the role set is empty",
                    RunError::NoDecisions => "the grammar cannot spell any applet in this set",
                });
                console::set_color(WHITE);
            }
            Ok(f) => {
                kprintln!(
                    "  {} example(s) from {} run(s), {} trained and {} held",
                    f.examples, f.runs, f.train_end, f.examples - f.train_end
                );
                kprintln!("  {} decision(s), {} of them held out, over {} applet(s)",
                          f.decisions, f.validation, f.classes);
                // Against what would otherwise run, not against a bare model.
                kprintln!("  trained    base {:.1}%", f.base_train * 100.0);
                kprintln!("  held-out   base {:.1}%  role {:.1}%", f.base * 100.0, f.adapted * 100.0);
                kprintln!("  paired     fixed {}  broke {}  chi {:.2}", f.fixed, f.broke, f.mcnemar);
                // The line that says whether the experiment was possible at
                // all. A harvested label is what the base already answered,
                // so a paired test with nothing in either cell is not a small
                // sample: there was no disagreement to find, and more
                // transcripts produce more examples the base already answers.
                if f.fixed == 0 && f.broke == 0 {
                    kprintln!("  the two agreed on every held-out decision, which is what a");
                    kprintln!("  harvested label is: the base's own argmax. More runs add more.");
                }
                let mark = |b: bool| if b { "pass" } else { "FAIL" };
                kprintln!("  J1 better  {}  ({})", mark(f.j1), f.j1_why);
                kprintln!("  J2 unasked {}  ({} of {} goals hold)", mark(f.j2), f.goals_held, f.goals_total);
                kprintln!("  J3 sane    {}  ({})", mark(f.j3), f.j3_why);
                kprintln!("  J4 cost    {}  ({} KiB resident)", mark(f.j4), f.resident_kib);
                if f.kept {
                    console::set_color(LTGREEN);
                    kprintln!("  kept -- workers of this role decode wearing it");
                } else {
                    console::set_color(LTRED);
                    kprintln!("  not kept -- on this evidence the role is a naming convention");
                }
                console::set_color(WHITE);
            }
        }
        return;
    }

    // `work cmp <a> <b>` -- the Stage 1 check, and it reports two answers
    // because they mean different things.
    if let Some(arg) = rest.strip_prefix("cmp ") {
        let Some((a, b)) = arg.trim().split_once(' ') else {
            kprintln!("  usage: work cmp <a> <b>");
            return;
        };
        let (a, b) = (a.trim(), b.trim());
        let c = work::compare(a, b);
        console::set_color(YELLOW);
        kprintln!("[work cmp] {} against {}", a, b);
        console::set_color(WHITE);
        kprintln!("  steps        {} and {}", c.steps_a, c.steps_b);
        if c.decisions_agree {
            console::set_color(LTGREEN);
            kprintln!("  decisions    agree -- the workflow is re-derivable over these inputs");
        } else {
            console::set_color(LTRED);
            match c.first_difference {
                Some(i) => kprintln!("  decisions    DIFFER, first at step {}", i),
                None => kprintln!("  decisions    DIFFER in count"),
            }
        }
        console::set_color(WHITE);
        if c.observations_agree {
            kprintln!("  observations agree");
        } else {
            // Not a failure. A run writes its own steps under /ai/work, which
            // is inside the /ai a worker lists, so the second run sees a
            // directory hash the first one changed.
            kprintln!("  observations differ, which is the world moving between runs");
        }
        return;
    }

    // `work run <run> [budget]`
    if let Some(arg) = rest.strip_prefix("run ") {
        let mut it = arg.split_whitespace();
        let Some(name) = it.next() else {
            kprintln!("  usage: work run <run> [budget]");
            return;
        };
        let budget = it.next().and_then(|w| w.parse::<usize>().ok()).unwrap_or(4);
        if work::plan(name).is_none() {
            console::set_color(LTRED);
            kprintln!("  no such run: {}", name);
            console::set_color(WHITE);
            return;
        }
        if crate::ai::mind_busy() {
            kprintln!("  the mind is busy -- wait for it to finish first");
            return;
        }
        // Read-only, and the operator types the word to widen it. Same shape
        // as `act` and for the same reason: the grammar is what enforces it,
        // so a worker at this level has no token sequence for a mutating
        // applet rather than being refused after choosing one.
        let trust = crate::ai::harness::Trust::ReadOnly;
        console::set_color(YELLOW);
        kprintln!("[work run] {}  budget {}  trust read-only", name, budget);
        console::set_color(WHITE);
        let (n, calls) = work::run(name, trust, budget);
        // The number Stage 2 is judged on. A manager that leaves nothing
        // decided costs one call per step and has bought structure only.
        kprintln!("  {} step(s) ran, {} model call(s)", n, calls);
        if let Some(h) = work::root(name) {
            // The number the whole design turns on. Two runs of one plan that
            // did the same thing print the same line here.
            kprintln!("  root {}", crate::ai::voter::hex(&h));
        }
        return;
    }

    let Some(plan) = work::plan(rest) else {
        console::set_color(LTRED);
        kprintln!("  no such run: {}", rest);
        console::set_color(WHITE);
        return;
    };

    console::set_color(YELLOW);
    kprintln!("[work] {}", rest);
    console::set_color(WHITE);
    kprintln!("  goal  {}", plan.goal);
    if let Some(h) = work::root(rest) {
        kprintln!("  root  {}", crate::ai::voter::hex(&h));
    }

    console::set_color(LTGRAY);
    kprintln!("  plan");
    console::set_color(WHITE);
    for st in &plan.steps {
        let mark = match st.status {
            Status::Done => "x",
            Status::Failed => "!",
            Status::Todo => " ",
        };
        // Indented by whether it has a parent. One level, because the render
        // is for reading and a deep tree printed as deep text is worse than a
        // flat list with parents named.
        let lead = if st.parent.is_some() { "    " } else { "  " };
        kprintln!("  [{}]{}{} {} {}", mark, lead, st.id,
                  if st.role.is_empty() { "-" } else { &st.role }, st.goal);
    }
    match plan.next() {
        Some(n) => kprintln!("  next  step {} ({})", n.id,
                             if n.role.is_empty() { "unassigned" } else { &n.role }),
        None if plan.done() => kprintln!("  next  nothing; every step is settled"),
        None => kprintln!("  next  nothing ready; every remaining step waits on a parent"),
    }

    let steps = work::steps(rest);
    if !steps.is_empty() {
        console::set_color(LTGRAY);
        kprintln!("  what happened");
        console::set_color(WHITE);
        for (i, st) in steps.iter().enumerate() {
            // The decision on its own line, because it is the half two runs
            // must agree on and the half a reader is checking.
            kprintln!("  {:>4} {} {} -- {}", i,
                      if st.ok { "ok  " } else { "FAIL" },
                      if st.role.is_empty() { "-" } else { &st.role },
                      st.goal);
            kprintln!("       > {}", st.action);
            for line in st.observation.lines().take(3) {
                kprintln!("       | {}", line);
            }
        }
    }
}

/// Learn one field, then another, and print what the second cost the first.
///
/// `study [stride]`. The stride subsamples the corpus, because the expensive
/// part is one forward pass per example and a full run is minutes; every stage
/// after that is a ridge fit over cached features and is nearly free.
fn study_cmd(rest: &str) {
    use crate::ai::study::{self, DOMAINS};

    let rest = rest.trim();
    if let Some(arg) = rest.strip_prefix("seq") {
        study_seq_cmd(arg.trim());
        return;
    }
    if let Some(arg) = rest.strip_prefix("check") {
        let n = arg.trim().parse::<usize>().unwrap_or(40).max(8);
        let b = crate::ai::train::Budget {
            epochs: 12, millis: 90_000, examples: n, lr: 0.02, rank: 8, alpha: 16.0,
        };
        if !crate::ai::train::hardware_ok() {
            console::set_color(LTRED);
            kprintln!("  the trainer needs AVX2 and FMA; add -cpu max under QEMU");
            console::set_color(WHITE);
            return;
        }
        match crate::ai::study::control(&b) {
            None => kprintln!("  could not prepare a trial"),
            Some((before, after, l0, l1, held)) => {
                kprintln!("[study check] train_masked with nothing masked, no adapter carried in");
                kprintln!("  loss      {} -> {}", l0 as i32, l1 as i32);
                kprintln!("  held out  {}% -> {}%  over {} decisions",
                          (before * 100.0) as i32, (after * 100.0) as i32, held);
                kprintln!("  compare against 'train adapter -n {} -e 12': they should agree", n);
            }
        }
        return;
    }

    let stride = rest.parse::<usize>().unwrap_or(1).max(1);
    let held = study::held_out_counts();

    console::set_color(YELLOW);
    kprintln!("[study] {} domains, stride {}", DOMAINS.len(), stride);
    console::set_color(WHITE);

    let Some(rows) = study::curriculum(1.0, stride) else {
        console::set_color(LTRED);
        kprintln!("  no corpus, or no model loaded");
        console::set_color(WHITE);
        return;
    };

    console::set_color(LTGRAY);
    let mut hdr = String::from("  trained on          fit ");
    for d in DOMAINS {
        hdr.push_str(&alloc::format!("{:>10}", d.name));
    }
    kprintln!("{}", hdr);
    console::set_color(WHITE);

    for r in &rows {
        let mut line = alloc::format!("  {:<18} {:>4} ", DOMAINS[r.trained - 1].name, r.fitted_on);
        for (i, (right, total)) in r.scores.iter().enumerate() {
            if i >= r.trained {
                // Not a zero, and printing one would be a lie worth catching.
                // The fit is given every applet as a class but only this
                // stage's labels, so a class with no positive example can
                // never be predicted -- the cell is zero by construction and
                // says nothing about transfer. Measuring zero-shot routing
                // needs the description-derived row that `zero_shot_rank`
                // builds, which is a different experiment.
                line.push_str("       n/a");
            } else if *total == 0 {
                line.push_str("         -");
            } else {
                // Marked when the slice is too small to mean anything. A
                // percentage over three examples is a number, not a
                // measurement, and it should not read like one.
                let pct = right * 100 / total;
                let thin = if *total < 8 { "?" } else { " " };
                line.push_str(&alloc::format!("{:>8}%{}", pct, thin));
            }
        }
        kprintln!("{}", line);
    }

    console::set_color(LTGRAY);
    let mut counts = String::from("  held out per domain:");
    for (i, n) in held.iter().enumerate() {
        counts.push_str(&alloc::format!(" {}={}", DOMAINS[i].name, n));
    }
    kprintln!("{}", counts);
    kprintln!("  ? marks a slice under 8 examples -- a number, not a measurement");
    kprintln!("  n/a is a field this stage never trained on: the label is not a class it");
    kprintln!("  can predict, so the cell would read 0% by construction and mean nothing");
    console::set_color(WHITE);

    // The reading, stated rather than left to the eye, because the interesting
    // cell is the one that went down and nobody looks for it.
    if rows.len() >= 2 {
        let first = &rows[0];
        let last = &rows[rows.len() - 1];
        let (a0, t0) = first.scores[0];
        let (a1, t1) = last.scores[0];
        if t0 > 0 && t1 > 0 {
            let before = a0 * 100 / t0;
            let after = a1 * 100 / t1;
            if after + 2 < before {
                console::set_color(LTRED);
                kprintln!(
                    "  {} fell {}% -> {}% as later fields were added: interference",
                    DOMAINS[0].name, before, after
                );
            } else {
                console::set_color(LTGREEN);
                kprintln!(
                    "  {} held at {}% -> {}% across every later field",
                    DOMAINS[0].name, before, after
                );
            }
            console::set_color(WHITE);
        }
    }
    kprintln!("  the probe is refit, not updated, so this is interference and not forgetting");
}

/// The adapter version: one adapter carried through every field in turn, so a
/// falling column is forgetting rather than interference.
///
/// `study seq [examples]`. Expensive -- the preparation is a forward pass per
/// example and every stage after it is a full training run, so the default
/// subsample is small and the wall-clock ceiling is real.
fn study_seq_cmd(arg: &str) {
    use crate::ai::study::{self, DOMAINS};
    use crate::ai::train::Budget;

    if !crate::ai::train::hardware_ok() {
        console::set_color(LTRED);
        kprintln!("  the trainer needs AVX2 and FMA, and this CPU reports neither");
        kprintln!("  under QEMU that is the default cpu model; add -cpu max");
        console::set_color(WHITE);
        return;
    }

    let examples = arg.parse::<usize>().unwrap_or(48).max(8);
    let b = Budget { epochs: 12, millis: 90_000, examples, lr: 0.02, rank: 8, alpha: 16.0 };

    console::set_color(YELLOW);
    kprintln!("[study seq] {} domains, ~{} examples, {} epochs a stage",
              DOMAINS.len(), examples, b.epochs);
    console::set_color(LTGRAY);
    kprintln!("  preparing: one forward pass per example, then {} training runs", DOMAINS.len());
    console::set_color(WHITE);

    let Some(rows) = study::sequential(&b) else {
        console::set_color(LTRED);
        kprintln!("  could not prepare a trial -- no corpus, no model, or nothing spellable");
        console::set_color(WHITE);
        return;
    };

    console::set_color(LTGRAY);
    let mut hdr = String::from("  after studying     loss       ");
    for d in DOMAINS {
        hdr.push_str(&alloc::format!("{:>10}", d.name));
    }
    kprintln!("{}", hdr);
    console::set_color(WHITE);

    let mut first: Vec<Option<usize>> = alloc::vec![None; DOMAINS.len()];
    for r in &rows {
        let label = match r.studied {
            None => "(frozen base)",
            Some(i) => DOMAINS[i].name,
        };
        // First and last, not just last. A single number cannot say whether
        // the run moved anything, and "the adapter changed nothing" is the
        // one outcome this table has to be able to show.
        let loss = if r.studied.is_none() {
            String::from("      -   ")
        } else {
            alloc::format!("{:>4}->{:<4}", r.first_loss as i32, r.last_loss as i32)
        };
        let mut line = alloc::format!("  {:<18} {} ", label, loss);
        for (i, (right, total)) in r.scores.iter().enumerate() {
            if *total == 0 {
                line.push_str("         -");
                continue;
            }
            let pct = right * 100 / total;
            // Mark the drop from this field's own best, which is the number
            // forgetting actually shows up in.
            if let Some(i2) = r.studied {
                if i2 == i && first[i].is_none() {
                    first[i] = Some(pct);
                }
            }
            let fell = match first[i] {
                Some(best) if pct + 5 < best => "v",
                _ => " ",
            };
            let thin = if *total < 8 { "?" } else { fell };
            line.push_str(&alloc::format!("{:>8}%{}", pct, thin));
        }
        kprintln!("{}", line);
    }

    console::set_color(LTGRAY);
    kprintln!("  v marks a field at least 5 points below its own best, after later study");
    kprintln!("  ? marks a slice under 8 held-out decisions");
    console::set_color(WHITE);

    // The reading. One adapter carried through, so this is the real thing.
    let mut forgot = String::new();
    if let Some(last) = rows.last() {
        for (i, (right, total)) in last.scores.iter().enumerate() {
            if *total < 8 {
                continue;
            }
            let pct = right * 100 / total;
            if let Some(best) = first[i] {
                if pct + 5 < best {
                    forgot.push_str(&alloc::format!(" {} {}%->{}%", DOMAINS[i].name, best, pct));
                }
            }
        }
    }
    if forgot.is_empty() {
        console::set_color(LTGREEN);
        kprintln!("  no field ended more than 5 points below its own best: no forgetting here");
    } else {
        console::set_color(LTRED);
        kprintln!("  forgotten by the end:{}", forgot);
    }
    console::set_color(WHITE);
    kprintln!("  one adapter carried through every stage, so this is forgetting and not interference");
}

/// What repeats across the skills this machine has written.
///
/// Reports and changes nothing. Rewriting a stored program would change its
/// content hash, which revokes the operator's trust in it -- correct
/// behaviour, and a thing that needs deciding before it happens rather than
/// discovered afterwards by a skill that stopped running.
fn abstract_cmd(rest: &str) {
    let progs = crate::ai::abstraction::stored();
    if progs.is_empty() {
        kprintln!("  no stored programs under /ai/tools");
        return;
    }

    console::set_color(YELLOW);
    kprintln!("[abstract] {} program(s)", progs.len());
    console::set_color(WHITE);

    let found = crate::ai::abstraction::analyse(&progs);
    if found.is_empty() {
        kprintln!("  nothing repeats that would pay to name");
        kprintln!("  (the objective refuses anything whose call is as big as its body)");
        return;
    }

    let show = if rest.trim() == "all" { found.len() } else { found.len().min(5) };
    console::set_color(LTGRAY);
    kprintln!("  saved  seen  in  arity  structure");
    console::set_color(WHITE);
    for c in found.iter().take(show) {
        kprintln!(
            "  {:>5}  {:>4}  {:>2}  {:>5}  {}",
            c.saved,
            c.count,
            c.programs.len(),
            c.arity,
            c.skeleton
        );
    }
    if show < found.len() {
        kprintln!("  ... {} more; 'abstract all' for every one", found.len() - show);
    }

    // The best candidate as something that could actually be written to /lib.
    if let Some(top) = found.first() {
        kprintln!("");
        console::set_color(LTGREEN);
        kprintln!("  {}", top.proposal(0));
        console::set_color(WHITE);
        kprintln!("  would save {} nodes across {} occurrence(s) in:", top.saved, top.count);
        for p in &top.programs {
            kprintln!("    {}", p);
        }
    }
}

/// What graphics hardware this machine has, and whether it will answer.
///
/// Read-only apart from setting the memory-space enable bit, which the
/// register read needs and which firmware has normally set already. Nothing
/// here powers a device on, resets one, or grants it DMA. Those are all
/// separate acts and they should stay separate ones.
fn gpu_cmd(acpi: &Option<Acpi>) {
    let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) else {
        console::set_color(LTRED);
        kprintln!("  no MCFG table, so no ECAM base and no PCIe enumeration at all");
        console::set_color(WHITE);
        return;
    };

    console::set_color(YELLOW);
    kprintln!("[gpu]");
    console::set_color(WHITE);

    let displays = crate::gpu::all(ecam);
    if displays.is_empty() {
        kprintln!("  no display controller answers on this bus");
    }
    for d in &displays {
        kprintln!(
            "  {:02x}:{:02x}.{}  {:04x}:{:04x}  {} {}",
            d.bus,
            d.dev,
            d.func,
            d.vendor,
            d.device,
            crate::dev::pci::class_name(d.class, d.subclass),
            crate::dev::pci::vendor_name(d.vendor)
        );
    }

    let Some(gpu) = crate::gpu::find(ecam) else {
        crate::gpu::record(None, None);
        report_absent(ecam);
        return;
    };

    kprintln!("  revision  {:#04x}", gpu.revision);
    kprintln!(
        "  subsys    {:04x}:{:04x}",
        gpu.subsystem & 0xffff,
        gpu.subsystem >> 16
    );
    for (n, bar) in [(0, gpu.bar0), (1, gpu.bar1), (3, gpu.bar3)] {
        match bar {
            Some(a) if a != 0 => kprintln!("  bar{}      {:#x}", n, a),
            Some(_) => kprintln!("  bar{}      unassigned", n),
            None => kprintln!("  bar{}      none", n),
        }
    }

    // NV_PMC_BOOT_0 is NVIDIA's register, and offset 0 of BAR0 on anyone
    // else's device is whatever they put there. Under QEMU it is the VGA
    // framebuffer, which reads back as 0x001c222c and is harmless; on a
    // device with a read-sensitive register at offset 0 it would not be.
    // Reading a vendor's register on another vendor's part is not a probe,
    // it is a poke.
    if !gpu.is_nvidia() {
        kprintln!("  boot0     not read: that register belongs to NVIDIA, and this part does not");
        report_absent(ecam);
        crate::gpu::record(Some(&gpu), None);
        return;
    }

    match gpu.boot0(ecam) {
        None => {
            console::set_color(LTRED);
            kprintln!("  boot0     not read: no usable BAR0, or the mapping failed");
            console::set_color(WHITE);
            crate::gpu::record(Some(&gpu), None);
        }
        Some(raw) => match crate::gpu::decode_boot0(raw) {
            Some((chipset, rev)) => {
                let name = crate::gpu::chip_name(chipset);
                let fam = crate::gpu::family(chipset);
                console::set_color(LTGREEN);
                if name.is_empty() {
                    kprintln!(
                        "  boot0     {:#010x}  chipset {:#05x} rev {:#04x}, {} family",
                        raw,
                        chipset,
                        rev,
                        if fam.is_empty() { "an unrecognised" } else { fam }
                    );
                } else {
                    kprintln!(
                        "  boot0     {:#010x}  {} rev {:#04x} ({})",
                        raw,
                        name,
                        rev,
                        fam
                    );
                }
                kprintln!("  the device is alive and addressable from ring 0");
                console::set_color(WHITE);
                crate::gpu::record(Some(&gpu), Some(if name.is_empty() { fam } else { name }));
            }
            None => {
                console::set_color(LTRED);
                kprintln!("  boot0     {:#010x}  not a chip id", raw);
                console::set_color(WHITE);
                if raw == 0xFFFF_FFFF {
                    kprintln!("  all ones: the aperture decodes, but nothing answers behind it");
                }
                crate::gpu::record(Some(&gpu), None);
            }
        },
    }
}

/// Say whether an absent GPU is missing or merely asleep.
///
/// Worth the extra scan. On a muxless Optimus laptop the discrete part is
/// parked in D3cold, where config space reads back as all ones and the device
/// is indistinguishable from one that was never fitted. Its root port does not
/// disappear, though, so a bridge forwarding a bus that nobody answers on is
/// the tell -- and it is the difference between giving up and reaching for the
/// ACPI interpreter this kernel already has.
fn report_absent(ecam: u64) {
    let empty = crate::gpu::empty_bridges(ecam);
    if empty.is_empty() {
        kprintln!("  every bridge forwards a bus that answers, so nothing is hiding");
        return;
    }
    console::set_color(YELLOW);
    for (b, secondary) in &empty {
        kprintln!(
            "  bridge {:02x}:{:02x}.{} forwards bus {:02x}, and nothing answers there",
            b.bus,
            b.dev,
            b.func,
            secondary
        );
    }
    kprintln!("  that is what a device parked in D3cold looks like");
    console::set_color(WHITE);
}

fn store_cmd(rest: &str) {
    use crate::store::{self, cas, sha256};

    match rest {
        "" | "status" => {
            console::set_color(YELLOW);
            kprintln!("[store]");
            console::set_color(WHITE);
            let sha_ok = sha256::selftest();
            console::set_color(if sha_ok { LTGREEN } else { LTRED });
            kprintln!("  sha-256 vectors: {}", if sha_ok { "pass" } else { "FAIL" });
            console::set_color(WHITE);
            if store::mounted() {
                let st = store::cas::stream_selftest();
                console::set_color(if st { LTGREEN } else { LTRED });
                kprintln!(
                    "  {}  a 300 KiB blob round-trips, and its middle reads without it",
                    if st { "ok " } else { "FAIL" }
                );
                console::set_color(WHITE);
            }
            if !store::mounted() {
                kprintln!("  not mounted. 'store init' to format free space.");
                return;
            }
            store::with(|s| {
                kprintln!(
                    "  region lba {}..{}  seq {}  checkpoints {}",
                    s.sb.region_start,
                    s.sb.region_start + s.sb.region_blocks,
                    s.sb.seq,
                    s.sb.checkpoints
                );
                kprintln!(
                    "  next free lba {}  ({} blocks left)",
                    s.sb.alloc_next,
                    s.free_blocks()
                );
                if s.sb.root.is_none() {
                    kprintln!("  no checkpoints yet");
                } else {
                    let h = sha256::short_hex(&s.sb.root.hash);
                    kprintln!(
                        "  root {}  at lba {} ({} B)",
                        core::str::from_utf8(&h).unwrap_or("?"),
                        s.sb.root.lba,
                        s.sb.root.len
                    );
                }
            });
        }

        "init" => match store::init() {
            Ok((start, blocks)) => {
                console::set_color(LTGREEN);
                kprintln!("  formatted lba {}..{} ({} blocks)", start, start + blocks, blocks);
                console::set_color(LTRED);
                kprintln!("  NVMe writes are now UNLOCKED for this session");
                console::set_color(WHITE);
            }
            Err(e) => {
                console::set_color(LTRED);
                match e {
                    store::InitError::NoRoom => {
                        kprintln!("  no unclaimed space on this disk -- refusing to write");
                        kprintln!("  (expected on a disk fully allocated to Windows)");
                    }
                    other => kprintln!("  init failed: {:?}", other),
                }
                console::set_color(WHITE);
            }
        },

        "unlock" => match store::unlock() {
            Ok((start, blocks)) => {
                console::set_color(LTRED);
                kprintln!("  writes UNLOCKED for lba {}..{}", start, start + blocks);
                console::set_color(WHITE);
                kprintln!("  region re-checked against the partition table first");
            }
            Err(e) => {
                console::set_color(LTRED);
                match e {
                    cas::Error::Unsafe => kprintln!("  region overlaps a partition that is not ours -- refusing"),
                    cas::Error::NotFormatted => kprintln!("  no store mounted"),
                    other => kprintln!("  {:?}", other),
                }
                console::set_color(WHITE);
            }
        },

        "test" => {
            if !store::mounted() {
                console::set_color(LTRED);
                kprintln!("  not mounted -- run 'store init' first");
                console::set_color(WHITE);
                return;
            }
            if !crate::dev::nvme::writes_unlocked() {
                console::set_color(LTRED);
                kprintln!("  writes are locked -- run 'store unlock' first");
                console::set_color(WHITE);
                return;
            }
            let ok = store::with(|s| {
                let mut entries: alloc::vec::Vec<cas::Entry> = alloc::vec::Vec::new();
                let payloads: [&[u8]; 3] = [
                    b"the first blob",
                    b"a second, longer blob that spans more of a block",
                    b"third",
                ];
                for (i, p) in payloads.iter().enumerate() {
                    match s.put(p) {
                        Ok(c) => {
                            let mut name = [0u8; cas::NAME_LEN];
                            name[0] = b'b';
                            name[1] = b'0' + i as u8;
                            entries.push(cas::Entry { name, chunk: c });
                        }
                        Err(e) => {
                            kprintln!("  put failed: {:?}", e);
                            return false;
                        }
                    }
                }
                // Read back and verify content addressing before committing.
                for (i, e) in entries.iter().enumerate() {
                    match s.get(&e.chunk) {
                        Ok(d) => {
                            if d.as_slice() != payloads[i] {
                                kprintln!("  blob {} round-trip MISMATCH", i);
                                return false;
                            }
                        }
                        Err(err) => {
                            kprintln!("  get failed: {:?}", err);
                            return false;
                        }
                    }
                }
                match s.commit(&entries) {
                    Ok(r) => {
                        let h = sha256::short_hex(&r.hash);
                        kprintln!(
                            "  committed {} entries, root {}",
                            entries.len(),
                            core::str::from_utf8(&h).unwrap_or("?")
                        );
                        true
                    }
                    Err(e) => {
                        kprintln!("  commit failed: {:?}", e);
                        false
                    }
                }
            })
            .unwrap_or(false);

            console::set_color(if ok { LTGREEN } else { LTRED });
            kprintln!("  {}", if ok { "put / get / commit verified" } else { "STORE TEST FAILED" });
            console::set_color(WHITE);
        }

        "log" => {
            if !store::mounted() {
                kprintln!("  not mounted");
                return;
            }
            store::with(|s| {
                let mut r = s.sb.root;
                let mut n = 0;
                console::set_color(YELLOW);
                kprintln!("  seq   root              entries");
                console::set_color(WHITE);
                while !r.is_none() && n < 32 {
                    match s.read_manifest(&r) {
                        Ok(m) => {
                            let h = sha256::short_hex(&r.hash);
                            kprintln!(
                                "  {:<5} {}  {}",
                                m.seq,
                                core::str::from_utf8(&h).unwrap_or("?"),
                                m.entries.len()
                            );
                            r = m.prev;
                        }
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("  chain broken: {:?}", e);
                            console::set_color(WHITE);
                            break;
                        }
                    }
                    n += 1;
                }
                if n == 0 {
                    kprintln!("  (no checkpoints)");
                }
            });
        }

        "rollback" => {
            if !store::mounted() {
                kprintln!("  not mounted");
                return;
            }
            store::with(|s| match s.read_manifest(&s.sb.root) {
                Ok(m) => {
                    if m.prev.is_none() {
                        kprintln!("  already at the first checkpoint");
                        return;
                    }
                    match s.rollback_to(m.prev) {
                        Ok(()) => {
                            console::set_color(LTGREEN);
                            kprintln!("  rolled back to seq {}", s.sb.seq);
                            console::set_color(WHITE);
                            kprintln!("  nothing was erased; roll forward is still possible");
                        }
                        Err(e) => kprintln!("  rollback failed: {:?}", e),
                    }
                }
                Err(e) => kprintln!("  cannot read current manifest: {:?}", e),
            });
        }

        other => {
            console::set_color(LTRED);
            kprintln!("  unknown: store {}", other);
            console::set_color(WHITE);
            kprintln!("  store [status|init|unlock|test|log|rollback]");
        }
    }
}

fn execute(line: &str, boot: &BootInfo, acpi: &Option<Acpi>, interp: &mut aiksi::Interp) {
    let mut parts = line.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    // One name, two meanings, told apart by shape: `write <path> <text>` is
    // the sysbox applet and has always been; `write [path]` with no text is
    // the editor window, because an editor is what "write" with nothing to
    // write means. Decided here, before sysbox, or the applet's usage error
    // would claim the bare form.
    if cmd == "write" && rest.splitn(2, char::is_whitespace).nth(1).is_none() {
        crate::gfx::desk::open_write(rest);
        return;
    }

    // sysbox first: it owns a whole vocabulary of short names, and claiming
    // them here keeps that list in one place instead of spreading twenty more
    // arms across this match.
    if crate::sysbox::dispatch(cmd, rest) {
        return;
    }

    match cmd {
        "" => {}
        "typewriter" => {
            match rest.parse::<u64>() {
                Ok(us) => {
                    console::set_pace(us);
                    if us == 0 {
                        kprintln!("  pacing off");
                    } else {
                        kprintln!("  {} us per character", us);
                    }
                }
                Err(_) => {
                    kprintln!("  {} us per character", console::pace_us());
                    kprintln!("  'typewriter <us>' to change, 'typewriter 0' for instant");
                    kprintln!("  any keypress skips pacing for the rest of a command");
                }
            }
        }
        // The whole screen, not just the character grid. `redraw` alone would
        // restore the text and leave whatever scribbled on the frame around it
        // still there -- which is exactly the state `refresh` exists to fix.
        "refresh" => crate::gfx::desk::draw(),
        "fat" => fat_cmd(rest),
        // `if` and `net` are the same command. `if` because that is what it
        // operates on; `net` because that is what it used to be called and
        // muscle memory is a real cost.
        "if" | "net" | "ifconfig" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                None => crate::net::report(),
                Some(name) => {
                    let Some(n) = crate::net::index_of(name) else {
                        kprintln!("  no such interface: {}  ('if' to list)", name);
                        return;
                    };
                    match it.next() {
                        None => crate::net::report(),
                        Some("up") => {
                            crate::net::ifaces()[n].up = true;
                            crate::net::report();
                        }
                        Some("down") => {
                            crate::net::ifaces()[n].up = false;
                            crate::net::report();
                        }
                        Some("dhcp") => crate::net::dhcp::report_on(n),
                        Some("ip") => {
                            // `if eth0 ip 10.0.2.15/24 10.0.2.2 10.0.2.3`
                            let mut c = crate::net::config_of(n);
                            if let Some(spec) = it.next() {
                                let (addr, prefix) = match spec.split_once('/') {
                                    Some((a, p)) => (a, p.parse().ok()),
                                    None => (spec, None),
                                };
                                match crate::net::parse_ip(addr) {
                                    None => {
                                        kprintln!("  not an address: {}", addr);
                                        return;
                                    }
                                    Some(ip) => c.ip = ip,
                                }
                                if let Some(bits) = prefix {
                                    c.netmask = crate::net::mask_from_prefix(bits);
                                }
                            }
                            if let Some(gw) = it.next().and_then(crate::net::parse_ip) {
                                c.gateway = gw;
                            }
                            if let Some(d) = it.next().and_then(crate::net::parse_ip) {
                                c.dns = d;
                            }
                            crate::net::set_config_of(n, c);
                            crate::net::report();
                        }
                        Some(other) => {
                            kprintln!("  unknown: if {} {}", name, other);
                            kprintln!("  try: up | down | dhcp | ip <a.b.c.d>[/bits] [gw] [dns]");
                        }
                    }
                }
            }
        }
        "wlan" | "wifi" => crate::net::wifi::report(),
        "trust" => match rest.trim() {
            "verify" => crate::net::trust::verify_roots(),
            _ => crate::net::trust::report(),
        },
        "wpa2" => crate::net::wpa2::report(),
        "https" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                None => kprintln!("  usage: https <host>[:port] [/path]"),
                Some(host) => {
                    let (h, port) = match host.split_once(':') {
                        Some((h, p)) => (h, p.parse().unwrap_or(443)),
                        None => (host, 443),
                    };
                    if let Some(ip) = host_to_ip(h) {
                        crate::net::tls::report(ip, h, port, it.next().unwrap_or("/"));
                    }
                }
            }
        }
        "crypto" => {
            let t0 = crate::time::rdtsc();
            let ok = crate::crypto::selftest();
            let mhz = crate::time::tsc_mhz().max(1);
            kprintln!(
                "  {} in {} ms",
                if ok { "all vectors pass" } else { "SOMETHING IS WRONG" },
                (crate::time::rdtsc() - t0) / mhz / 1000
            );
        }
        "ping" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                None => kprintln!("  usage: ping <host> [count]"),
                Some(host) => match host_to_ip(host) {
                    None => {}
                    Some(ip) => {
                        let n = it.next().and_then(|s| s.parse().ok()).unwrap_or(4);
                        crate::net::ping(ip, n);
                    }
                },
            }
        }
        "dhcp" => crate::net::dhcp::report(),
        "dns" => {
            if rest.is_empty() {
                kprintln!("  usage: dns <name>");
            } else {
                let t0 = crate::time::rdtsc();
                match crate::net::dns::resolve(rest.trim()) {
                    Ok(ip) => {
                        let mhz = crate::time::tsc_mhz().max(1);
                        kprintln!(
                            "  {} is {}.{}.{}.{}   ({} ms)",
                            rest.trim(), ip[0], ip[1], ip[2], ip[3],
                            (crate::time::rdtsc() - t0) / mhz / 1000
                        );
                    }
                    Err(e) => kprintln!("  {}", e.name()),
                }
            }
        }
        "tcp" => {
            use crate::net::tcp;
            let mut it = rest.splitn(2, ' ');
            match (it.next().unwrap_or(""), it.next().unwrap_or("").trim()) {
                ("connect", args) => {
                    let mut a = args.split_whitespace();
                    match (a.next(), a.next().and_then(|s| s.parse::<u16>().ok())) {
                        (Some(host), Some(port)) => {
                            if let Some(ip) = host_to_ip(host) {
                                match tcp::connect(ip, port, 5000) {
                                    Ok(()) => kprintln!("  connected to {}.{}.{}.{}:{}",
                                        ip[0], ip[1], ip[2], ip[3], port),
                                    Err(e) => kprintln!("  {}", e.name()),
                                }
                            }
                        }
                        _ => kprintln!("  usage: tcp connect <host> <port>"),
                    }
                }
                ("send", text) if !text.is_empty() => {
                    // A trailing CRLF is what a line-oriented peer is waiting
                    // for; typing one at this prompt is not possible.
                    let mut line = alloc::string::String::from(text);
                    line.push_str("\r\n");
                    match tcp::send(line.as_bytes(), 5000) {
                        Ok(()) => kprintln!("  sent {} B, acknowledged", line.len()),
                        Err(e) => kprintln!("  {}", e.name()),
                    }
                }
                ("recv", arg) => {
                    let ms = arg.parse().unwrap_or(2000);
                    let data = tcp::recv(ms);
                    if data.is_empty() {
                        kprintln!("  nothing");
                    } else {
                        kprintln!("  {} B", data.len());
                        let s = alloc::string::String::from_utf8_lossy(&data);
                        for line in s.lines().take(20) {
                            kprintln!("  | {}", line);
                        }
                    }
                }
                ("close", _) => {
                    tcp::close(2000);
                    kprintln!("  closed");
                }
                _ => tcp::report(),
            }
        }
        "http" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                None => kprintln!("  usage: http <host>[:port] [/path]"),
                Some(host) => {
                    let (h, port) = match host.split_once(':') {
                        Some((h, p)) => (h, p.parse().unwrap_or(80)),
                        None => (host, 80),
                    };
                    if let Some(ip) = host_to_ip(h) {
                        // The Host header carries the name when there is one:
                        // a virtual host answers on a shared address and would
                        // otherwise hand back the wrong site.
                        crate::net::tcp::http_report(ip, h, port, it.next().unwrap_or("/"));
                    }
                }
            }
        }
        "pkg" => {
            let mut it = rest.splitn(2, ' ');
            let verb = it.next().unwrap_or("");
            let arg = it.next().unwrap_or("").trim();
            match verb {
                "" | "list" => crate::pkg::list(),
                "info" => crate::pkg::info(arg),
                "remove" => crate::pkg::remove(arg),
                "add" => match crate::sysbox::read_blob(arg) {
                    Some(bytes) => {
                        crate::pkg::install(&bytes);
                    }
                    None => kprintln!("  no such file: {}", arg),
                },
                other => kprintln!("  not a pkg subcommand: {}", other),
            }
        }
        "edit" | "vi" => {
            if rest.is_empty() {
                kprintln!("  usage: edit <path>");
            } else {
                crate::edit::run(rest);
            }
        }
        "date" => match crate::dev::rtc::now() {
            Some(d) => {
                kprintln!(
                    "  {:04}-{:02}-{:02} {:02}:{:02}:{:02}",
                    d.year, d.month, d.day, d.hour, d.minute, d.second
                );
                kprintln!("  {} seconds since 1970", crate::dev::rtc::unix_seconds(&d));
            }
            None => kprintln!("  no usable RTC on this machine"),
        },
        "autosnap" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                Some("on") => {
                    let secs = it.next().and_then(|s| s.parse().ok()).unwrap_or(60);
                    crate::sysbox::autosnap_configure(true, secs);
                }
                Some("off") => crate::sysbox::autosnap_configure(false, 0),
                _ => {}
            }
            crate::sysbox::autosnap_report();
        }
        "act" => {
            // Read-only unless explicitly told otherwise, and "trusted" has to
            // be typed in full. The distinction is enforced in the grammar, not
            // after the fact: with ReadOnly the mutating applets are not
            // reachable outputs at all.
            let (trust, task) = match rest.strip_prefix("trusted ") {
                Some(t) => (crate::ai::harness::Trust::Full, t.trim()),
                None => (crate::ai::harness::Trust::ReadOnly, rest),
            };
            let task = if task.is_empty() { "look at the files" } else { task };
            crate::ai::harness::report(task, trust, 1.0);
        }
        // `teach file <path>` reads "applet task" one per line, so a batch of
        // examples can be written in the editor instead of one shell line at a
        // time. Whitespace-separated rather than tab-separated: a tab is
        // awkward to type in a modal editor, and the applet name never
        // contains a space.
        // `teach bundle <path>` replaces the whole corpus from one blob --
        // the transfer `fat get` leaves in the namespace. Replacing rather
        // than appending, because the bundle carries split *positions* and a
        // merge would leave them describing a corpus that no longer exists.
        "teach" if rest.starts_with("bundle ") => {
            let path = rest[7..].trim();
            match crate::sysbox::read_blob(path) {
                None => kprintln!("  no such file: {}", path),
                Some(bytes) => {
                    let n = bytes.len();
                    match crate::ai::vocab::import_bundle(crate::ai::vocab::CORPUS, &bytes) {
                        Err(e) => kprintln!("  {}: {:?} -- corpus untouched", path, e),
                        Ok(count) => {
                            let (train, val_end, len) = crate::ai::vocab::splits();
                            kprintln!(
                                "  {} examples from {} bytes -> {}",
                                count,
                                n,
                                crate::ai::vocab::CORPUS
                            );
                            kprintln!(
                                "  train [0,{})  validation [{},{})  test [{},{})",
                                train, train, val_end, val_end, len
                            );
                            kprintln!("  'fit' to rebuild the router from them");
                        }
                    }
                }
            }
        }
        "teach" if rest.starts_with("file ") => {
            let path = rest[5..].trim();
            match crate::sysbox::read_blob(path) {
                None => kprintln!("  no such file: {}", path),
                Some(bytes) => {
                    let text = core::str::from_utf8(&bytes).unwrap_or("");
                    let (mut ok, mut bad) = (0usize, 0usize);
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        let mut p = line.splitn(2, char::is_whitespace);
                        match (p.next(), p.next()) {
                            (Some(applet), Some(task)) if !task.trim().is_empty() => {
                                if crate::sysbox::APPLETS.iter().any(|a| a.name == applet)
                                    && crate::ai::vocab::record(applet, task.trim())
                                {
                                    ok += 1;
                                } else {
                                    bad += 1;
                                    kprintln!("  skipped: {}", line);
                                }
                            }
                            _ => {
                                bad += 1;
                                kprintln!("  skipped: {}", line);
                            }
                        }
                    }
                    kprintln!("  {} example(s) recorded, {} skipped", ok, bad);
                    if ok > 0 {
                        kprintln!("  'fit' to rebuild the router from them");
                    }
                }
            }
        }
        "teach" => {
            let mut it = rest.splitn(2, ' ');
            match (it.next(), it.next()) {
                (Some(applet), Some(task)) if !task.trim().is_empty() => {
                    if crate::sysbox::APPLETS.iter().any(|a| a.name == applet) {
                        if crate::ai::vocab::record(applet, task.trim()) {
                            kprintln!("  recorded: {} <- \"{}\"", applet, task.trim());
                        } else {
                            kprintln!("  could not write to {}", crate::ai::vocab::CORPUS);
                        }
                    } else {
                        kprintln!("  '{}' is not an applet", applet);
                    }
                }
                _ => kprintln!("  usage: teach <applet> <task description>"),
            }
        }
        "ctx" => {
            let mut it = rest.splitn(2, ' ');
            match (it.next().unwrap_or(""), it.next().unwrap_or("").trim()) {
                ("save", name) if !name.is_empty() => match crate::ai::ctx_save(name) {
                    Some(n) => kprintln!("  saved {} B to {}/{}", n, crate::ai::CTX_DIR, name),
                    None => kprintln!("  could not save"),
                },
                ("load", name) if !name.is_empty() => match crate::ai::ctx_load(name) {
                    Some(p) => kprintln!("  restored to position {}", p),
                    None => kprintln!("  no such context, or it does not fit this model"),
                },
                _ => crate::ai::ctx_report(),
            }
        }
        "cont" => {
            let mut opts = crate::ai::GenOpts { resume: true, ..Default::default() };
            opts.steps = 60;
            crate::ai::generate(rest, &opts);
        }
        "ask" => {
            let mut opts = crate::ai::GenOpts { steps: 64, temperature: 0.3, ..Default::default() };
            let mut q = rest;
            // Flags in any order, and -t before -n is the natural way to type
            // it, so this loops rather than testing a fixed sequence.
            loop {
                if let Some(t) = q.strip_prefix("-t ") {
                    opts.think = true;
                    q = t.trim_start();
                } else if let Some(t) = q.strip_prefix("-n ") {
                    let mut it = t.splitn(2, ' ');
                    match (it.next(), it.next()) {
                        (Some(n), Some(tail)) => {
                            opts.steps = n.parse().unwrap_or(opts.steps);
                            q = tail.trim_start();
                        }
                        _ => break,
                    }
                } else {
                    break;
                }
            }
            if opts.think && opts.steps < 256 {
                // Reasoning is not free and a truncated <think> block prints as
                // a monologue with no answer, which looks like a broken model.
                opts.steps = 256;
                kprintln!("  (thinking: raised to {} tokens)", opts.steps);
            }
            if q.is_empty() {
                kprintln!("  usage: ask [-t] [-n tokens] <question>");
                kprintln!("     -t  let the model reason first, if it can");
                kprintln!("     one conversation, continued -- 'ask new' to forget it");
            } else if q == "new" {
                crate::ai::companion::reset();
                kprintln!("  forgotten. the next question starts a new conversation");
            } else {
                // A turn of one conversation, not a question asked into the
                // void. The cache already holds what was said, so this costs
                // the tokens of this turn and not of the whole exchange.
                let n = crate::ai::companion::turns();
                crate::ai::companion::turn(q, &opts);
                // **Not** parked per turn, and the measurement is why.
                //
                // Parking writes the whole KV cache as a blob. The store is
                // append-only -- `alloc_next` only rises and nothing reclaims
                // -- so with autosnap running, two turns of a 512-slot cache
                // wrote 16,375 then 19,625 blocks and took half a 27 MiB
                // region. A 0.6B at 8k has a cache three orders of magnitude
                // larger. There is no cadence that makes this affordable, so
                // it is on request: `ctx save live`, which `revive` reads at
                // boot.
                //
                // What survives on its own is what is small: `/ai/about` is a
                // few hundred bytes and autosnap carries it, which is the part
                // that actually has to be automatic.
                if n == 0 {
                    kprintln!("  ('ctx save live' to carry this conversation past the reboot)");
                }
            }
        }
        "repeat" => {
            let mut it = rest.split_whitespace();
            let (p, w) = crate::ai::repeat_settings();
            let p = it.next().and_then(|s| s.parse().ok()).unwrap_or(p);
            let w = it.next().and_then(|s| s.parse().ok()).unwrap_or(w);
            crate::ai::set_repeat(p, w);
            let (p, w) = crate::ai::repeat_settings();
            kprintln!("  repetition penalty {} over the last {} tokens", p, w);
            if p <= 1.0 {
                kprintln!("  (1.0 is off -- a loop can reinforce itself unchecked)");
            }
        }
        "logits" => {
            let ids: Vec<usize> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            crate::ai::logits_for(&ids);
        }
        "fit" => {
            // Defaults to whatever `search` adopted, not to 1.0. Refitting
            // with the old default would silently undo the search's choice,
            // which is a thing an operator does by typing `fit` to see the
            // numbers.
            let lambda: f32 = rest.split_whitespace().next()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(crate::ai::harness::default_lambda);
            crate::ai::harness::fit_probe(lambda);
        }
        "route" => {
            let (trust, task) = match rest.strip_prefix("trusted ") {
                Some(t) => (crate::ai::harness::Trust::Full, t.trim()),
                None => (crate::ai::harness::Trust::ReadOnly, rest),
            };
            if task.is_empty() {
                kprintln!("  usage: route [trusted] <task>");
            } else {
                crate::ai::harness::route_report(task, trust);
            }
        }
        "window" => {
            let n: Vec<usize> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if n.len() == 2 {
                crate::ai::set_window(n[0], n[1]);
            }
            crate::ai::window_report();
        }
        "gate" => crate::ai::harness::gate_report(),
        "search" => crate::ai::harness::search_report(),
        "probe" => crate::ai::harness::probe_features(),
        "feature" => {
            use crate::ai::harness::{set_feature_mode, Feature};
            match rest {
                "hidden" => { set_feature_mode(Feature::Hidden); kprintln!("  feature = hidden state"); }
                "pooled" => { set_feature_mode(Feature::Pooled); kprintln!("  feature = pooled embedding"); }
                _ => kprintln!("  usage: feature hidden|pooled"),
            }
        }
        "zeroshot" => crate::ai::harness::zero_shot_report(if rest.is_empty() { "diff" } else { rest }),
        // `adapter` -- what is attached, and moving it in and out of the
        // namespace. The blob is the adapter alone; the frozen checkpoint is
        // never written, which is what makes saving one cheap enough to do
        // before every change rather than after the interesting ones.
        // `godel` -- the self-modification loop, its ledger, and the way back.
        // `log` -- everything the machine has printed since power-on.
        //
        // The console keeps one screen, so this is the only place a line that
        // has scrolled past still exists. `log save` puts it in the namespace,
        // where `snap` can commit it to a store if one is provisioned.
        "log" => {
            use crate::gfx::console::{self, LTGRAY, YELLOW};
            let mut words = rest.split_whitespace();
            let (held, total, lost) = crate::log::stats();
            match words.next().unwrap_or("") {
                "save" => {
                    let path = words.next().unwrap_or(crate::log::PATH);
                    match crate::log::save(path) {
                        Some(n) => {
                            kprintln!("  {} bytes -> {}", n, path);
                            if lost > 0 {
                                console::set_color(YELLOW);
                                kprintln!("  {} earlier bytes had already wrapped out", lost);
                                console::set_color(LTGRAY);
                            }
                            if crate::store::mounted() {
                                kprintln!("  'snap' commits it to the store");
                            } else {
                                console::set_color(YELLOW);
                                kprintln!("  no store mounted: this survives until power-off and no longer");
                                console::set_color(LTGRAY);
                            }
                        }
                        None => kprintln!("  could not write {}", path),
                    }
                }
                "" | "status" => {
                    console::set_color(YELLOW);
                    kprintln!("[log]");
                    console::set_color(LTGRAY);
                    kprintln!("  {} bytes held of {} printed, {} wrapped out", held, total, lost);
                    kprintln!("  'log all' to print it, 'log save [path]' to keep it");
                }
                "all" => {
                    // Straight to the console without going back through
                    // kprintln, which would record the log into itself.
                    let bytes = crate::log::contents();
                    console::with(|c| c.write_bytes(&bytes));
                }
                _ => kprintln!("  usage: log [status|all|save [path]]"),
            }
        }
        // A council core the machine wrote, and the judges that let one in.
        // Train more than the routing head.
        //
        // `train adapter` moves the classifier against cached features. This
        // moves every q/k/v site too, which means no cached features and a
        // forward pass per example per epoch -- a different cost, so a
        // different command rather than a flag that hides which one is being
        // paid.
        "deeptrain" => {
            let mut w = Words::new(rest);
            let mut n = 8usize;
            let mut ep = 4usize;
            let mut rank = 4usize;
            while let Some(a) = w.next() {
                let v: usize = w.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                match a {
                    "-n" => n = v.max(1),
                    "-e" => ep = v.max(1),
                    "-r" => rank = v.max(1),
                    _ => {}
                }
            }
            if !crate::ai::engine_ready() {
                kprintln!("  {}", crate::ai::engine_refusal());
                return;
            }
            console::set_color(YELLOW);
            kprintln!("[deeptrain]");
            console::set_color(LTGRAY);
            // Through the judges, like every other change the machine makes
            // to itself.
            //
            // This used to call `train_full` straight and keep whatever came
            // out: the live model mutated, no certificate, no ledger line, no
            // node, and nothing to roll back with -- the one self-modifying
            // path outside the discipline every other one obeys. It now goes
            // through `godel::run` as a `Deep` proposal, which means it is
            // reverted unless J1 through J4 all agree, and recorded either way.
            let p = crate::ai::godel::Proposal::deep(0.02, rank, 2.0 * rank as f32, ep);
            let b = p.budget(n, 0);
            let verdict = crate::ai::with_engine(|e| crate::ai::godel::run(e, &b, &p));
            match verdict {
                None => {
                    kprintln!("  {}", crate::ai::engine_refusal());
                    return;
                }
                Some(Ok(c)) => {
                    console::set_color(if c.adopted { LTGREEN } else { YELLOW });
                    kprintln!(
                        "  {} -- fixed {} broke {} chi {}",
                        if c.adopted { "adopted" } else { "rejected, and reverted" },
                        c.fixed,
                        c.broke,
                        c.mcnemar
                    );
                    console::set_color(LTGRAY);
                    kprintln!(
                        "  J1 {} | J2 goals {}/{} | J3 {} | J4 {} KiB rank {}",
                        if c.j1 { "pass" } else { c.j1_why },
                        c.goals_held,
                        c.goals_total,
                        if c.j3 { "pass" } else { c.j3_why },
                        c.resident_kib,
                        c.rank
                    );
                    kprintln!("  'godel ledger' for the line, 'godel rollback' to undo");
                    return;
                }
                Some(Err(why)) => {
                    console::set_color(LTRED);
                    kprintln!("  refused: {}", why.why());
                    console::set_color(LTGRAY);
                }
            }
            // Why it refused, not a list of candidates. The machine knows
            // perfectly well: a checkpoint's architecture and the CPU's
            // feature bits are both sitting right there.
            let f = crate::cpu::detected();
            let why = crate::ai::with_engine(|e| {
                if e.model.cfg.hybrid() {
                    "the checkpoint is a hybrid -- q/k/v adapters need a dense one"
                } else if e.model.cfg.streams() {
                    "the cache is windowed, so a live index is not a position"
                } else {
                    "no corpus, or nothing in the training slice"
                }
            })
            .unwrap_or("no model is loaded");
            console::set_color(LTRED);
            if !(f.avx_enabled && f.avx2 && f.fma) {
                kprintln!(
                    "  refused -- avx enabled={} avx2={} fma={}",
                    f.avx_enabled, f.avx2, f.fma
                );
            } else {
                kprintln!("  refused -- {}", why);
            }
            console::set_color(LTGRAY);
        }
        "core" => {
            use crate::ai::{harness, voter};
            let mut w = Words::new(rest);
            match w.next().unwrap_or("") {
                "" | "status" => {
                    match voter::installed() {
                        Some(c) => kprintln!("  installed: {}", &voter::hex(&c.hash)[..8]),
                        None => kprintln!("  none installed -- the council is the three it shipped with"),
                    }
                    let names = crate::sysbox::children(voter::ROOT);
                    let n = names.iter().filter(|n| n.len() == 64).count();
                    kprintln!("  {} candidate(s) in {}", n, voter::ROOT);
                    kprintln!("  core prize | core list | core author | core write <path>");
                    kprintln!("  core judge <hash> | core trial <hash> | core off");
                }
                "list" => {
                    let names = crate::sysbox::children(voter::ROOT);
                    let mut any = false;
                    for name in names {
                        if name.len() != 64 {
                            continue;
                        }
                        any = true;
                        kprintln!("  {}", &name[..8]);
                    }
                    if !any {
                        kprintln!("  none");
                    }
                }
                // Take an Aiksi program from the namespace and store it as a
                // candidate. The operator's way in; `core author` is the
                // machine's, and they meet at the same content address and the
                // same judges.
                "write" => match w.next() {
                    None => kprintln!("  usage: core write <path-to-aiksi-program>"),
                    Some(path) => match crate::sysbox::read_blob(path) {
                        None => kprintln!("  no such file: {}", path),
                        Some(bytes) => {
                            let src = alloc::string::String::from_utf8_lossy(&bytes).into_owned();
                            let h = voter::store(&src);
                            match voter::load(&h) {
                                Ok(_) => {
                                    kprintln!("  stored {}", &voter::hex(&h)[..8]);
                                    kprintln!("  'core judge {}' before it goes anywhere near a decision", &voter::hex(&h)[..8]);
                                }
                                Err(e) => kprintln!("  stored, but it is not a core: {}", e),
                            }
                        }
                    },
                },
                "judge" | "install" => {
                    let install = rest.starts_with("install");
                    // With one candidate and no hash, mean that one. Addresses
                    // here are sixty-four characters and printed as eight, so
                    // requiring one to be retyped for the overwhelmingly
                    // common case -- a machine that has written exactly one
                    // core -- is ceremony that only ever produces typos.
                    let picked = match w.next() {
                        Some(want) => find_core(want),
                        None => {
                            let all: alloc::vec::Vec<alloc::string::String> =
                                crate::sysbox::children(voter::ROOT)
                                    .into_iter()
                                    .filter(|n| n.len() == 64)
                                    .collect();
                            if all.len() == 1 { voter::unhex(&all[0]) } else { None }
                        }
                    };
                    match picked {
                        None => {
                            kprintln!("  usage: core judge|install <hash>");
                            kprintln!("  'core list' shows them; the hash may be shortened");
                        }
                        Some(h) => harness::core_report(&h, install),
                    }
                }
                // What the producer has to write with, at a stated filter.
                //
                // Takes the setting as an argument so every candidate can be
                // measured in one boot against one corpus. Recompiling per
                // setting would compare numbers from different machines and
                // call the difference an effect.
                //
                // Train slice only, and no verdict is computed here: choosing
                // the filter by what scores best on validation would be
                // fitting the slice the judges measure on, which is the one
                // thing this whole module is arranged to prevent.
                "cues" => {
                    let purity: u32 = w.next().and_then(|s| s.parse().ok()).unwrap_or(100);
                    let min_uses: u32 = w.next().and_then(|s| s.parse().ok()).unwrap_or(2);
                    let names: Option<alloc::vec::Vec<alloc::string::String>> =
                        crate::ai::with_engine(|e| {
                            (0..e.head.len())
                                .map(|i| alloc::string::String::from(e.head.name(i)))
                                .collect()
                        });
                    match names {
                        None => kprintln!("  {}", crate::ai::engine_refusal()),
                        Some(names) => {
                            let t = voter::cue_table_at(&names, purity, min_uses);
                            let mut classes: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
                            for (_, c, _) in &t {
                                if !classes.contains(c) {
                                    classes.push(*c);
                                }
                            }
                            kprintln!(
                                "  purity {}%, support {}: {} cue(s) over {} of {} class(es)",
                                purity,
                                min_uses,
                                t.len(),
                                classes.len(),
                                names.len()
                            );
                            // The cues themselves, because the number alone
                            // cannot say whether loosening bought signal or
                            // filler. A table full of "the" is a bigger table
                            // and a worse one.
                            for c in classes.iter().take(8) {
                                let mut line = alloc::string::String::new();
                                for (word, owner, n) in t.iter().filter(|(_, o, _)| o == c).take(6) {
                                    if !line.is_empty() {
                                        line.push(' ');
                                    }
                                    let _ = owner;
                                    line.push_str(word);
                                    line.push('/');
                                    line.push_str(&alloc::format!("{}", n));
                                }
                                kprintln!("    {:8} {}", names[*c], line);
                            }
                            if classes.len() > 8 {
                                kprintln!("    ... and {} more class(es)", classes.len() - 8);
                            }
                        }
                    }
                }
                // What the cue pool could achieve if the machine chose from it
                // perfectly. The producer's ceiling, beside the judge's.
                "oracle" => {
                    let first = w.next().unwrap_or("");
                    // `core oracle contested` measures the pool the producer
                    // actually draws on now; the numeric form measures the old
                    // class-exclusive pool, which is kept precisely so the two
                    // can be put side by side on one slice.
                    let contested = first == "contested";
                    let purity: u32 = first.parse().unwrap_or(100);
                    let min_uses: u32 = w.next().and_then(|s| s.parse().ok()).unwrap_or(2);
                    let names: Option<alloc::vec::Vec<alloc::string::String>> =
                        crate::ai::with_engine(|e| {
                            (0..e.head.len())
                                .map(|i| alloc::string::String::from(e.head.name(i)))
                                .collect()
                        });
                    let names_for_best = names.clone().unwrap_or_default();
                    let out = names.and_then(|names| {
                        crate::ai::with_engine(|e| {
                            if contested {
                                let t = harness::contested_cues(e, &names);
                                kprintln!("  contested pool: {} cue(s)", t.len());
                                harness::cue_oracle_on(e, &t)
                            } else {
                                harness::cue_oracle(e, &names, purity, min_uses)
                            }
                        })
                    });
                    match out {
                        None => kprintln!("  {}", crate::ai::engine_refusal()),
                        Some(Err(e)) => kprintln!("  {}", e),
                        Some(Ok(v)) => {
                            kprintln!(
                                "  purity {}%, support {}: {} cue(s)",
                                purity, min_uses, v.cues
                            );
                            kprintln!("  {} usable, {} harmful, {} inert", v.usable, v.harmful, v.inert);
                            kprintln!(
                                "  best single rule: '{}' -> {} fixes {} breaks {}",
                                v.best,
                                names_for_best
                                    .get(v.best_class)
                                    .map(|s| s.as_str())
                                    .unwrap_or("?"),
                                v.best_fixed,
                                v.best_broke
                            );
                            let need = crate::ai::godel::clean_fixes_needed();
                            console::set_color(if v.reach >= need { LTGREEN } else { YELLOW });
                            kprintln!(
                                "  {} item(s) repairable by some rule in the pool ({} needed)",
                                v.reach, need
                            );
                            console::set_color(LTGRAY);
                        }
                    }
                }
                // Where a routing decision's time actually goes.
                "bench" => {
                    console::set_color(YELLOW);
                    kprintln!("[aiksi]");
                    console::set_color(LTGRAY);
                    crate::aiksi::bench();
                    console::set_color(YELLOW);
                    kprintln!("[vote]");
                    console::set_color(LTGRAY);
                    voter::bench();
                }
                // How much room a core has here, asked without one.
                //
                // The first question, and until now an unaskable one. A core
                // only matters where the two counters disagree, so a slice
                // where they mostly agree has nothing for any core to win --
                // and that is a fact about the corpus and the judge, not about
                // anything the machine might write.
                "prize" | "room" => {
                    match crate::ai::with_engine(|e| {
                        harness::core_census(e, harness::VALIDATION)
                    }) {
                        None => kprintln!("  {}", crate::ai::engine_refusal()),
                        Some(Err(e)) => kprintln!("  {}", e),
                        Some(Ok(v)) => {
                            let need = crate::ai::godel::clean_fixes_needed();
                            kprintln!("  {} validation item(s) judged", v.n);
                            kprintln!("  {} contested -- the two counters disagree", v.contested);
                            kprintln!("  {} recoverable -- one of them is right", v.recoverable);
                            kprintln!("  {} within reach -- and the probe is wrong", v.prize);
                            console::set_color(if v.prize >= need { LTGREEN } else { YELLOW });
                            kprintln!("  J1 needs {} clean repair(s)", need);
                            if v.prize < need {
                                kprintln!("  so no core can pass on this slice, however well written");
                                console::set_color(LTGRAY);
                                kprintln!("  a core seconds one counter against the other; it cannot");
                                kprintln!("  answer where they already agree. more room comes from a");
                                kprintln!("  wider validation slice or a rule that lets a core speak");
                                kprintln!("  where they agree -- both are changes to the judged system");
                            }
                            console::set_color(LTGRAY);
                        }
                    }
                }
                // The machine writes one.
                //
                // Printed rather than judged, and deliberately two steps: the
                // thesis of this whole module is that a core is a rule a
                // person can argue with, and that is worth nothing if the
                // first time anybody sees it is in a ledger line saying it was
                // adopted. `core trial` is the next step and it is the
                // operator's to take -- the night loop takes it unattended,
                // which is what the judges are for.
                "author" => match crate::ai::godel::write_core() {
                    None => {
                        kprintln!("  nothing written");
                        kprintln!("  needs a model, a training corpus with cues to draw on,");
                        kprintln!("  and a decode that commits -- 'godel status' and 'model' say which is missing");
                    }
                    Some(h) => {
                        kprintln!("  wrote {}", &voter::hex(&h)[..8]);
                        if let Some(src) = crate::sysbox::read_blob(&{
                            let mut p = alloc::string::String::from(voter::ROOT);
                            p.push('/');
                            p.push_str(&voter::hex(&h));
                            p
                        }) {
                            console::set_color(LTGRAY);
                            for line in alloc::string::String::from_utf8_lossy(&src).lines() {
                                kprintln!("    {}", line);
                            }
                        }
                        console::set_color(YELLOW);
                        kprintln!("  'core trial {}' to judge it and record what happened", &voter::hex(&h)[..8]);
                        console::set_color(LTGRAY);
                    }
                },
                // Judge it *and* keep a record of what happened.
                //
                // `core judge` prints a verdict and forgets it; `core install`
                // changes the mind and leaves nothing to undo it with. This is
                // the same three judges through `godel`, so the core gets a
                // node in the lineage, a line in the ledger, and a
                // `godel rollback` that puts the previous one back.
                "trial" => {
                    let picked = match w.next() {
                        Some(want) => find_core(want),
                        None => {
                            let all: alloc::vec::Vec<alloc::string::String> =
                                crate::sysbox::children(voter::ROOT)
                                    .into_iter()
                                    .filter(|n| n.len() == 64)
                                    .collect();
                            if all.len() == 1 { voter::unhex(&all[0]) } else { None }
                        }
                    };
                    match picked {
                        None => kprintln!("  usage: core trial <hash>"),
                        Some(h) => {
                            // Through the dispatcher, so this leaves the same
                            // marker the night loop reads. Judging a core here
                            // and having the machine judge it again unprompted
                            // at 3am is the ledger recording one event twice.
                            let p = crate::ai::godel::Proposal::core(h);
                            let b = p.budget(0, 0);
                            match crate::ai::with_engine(|e| crate::ai::godel::run(e, &b, &p)) {
                                None => kprintln!("  {}", crate::ai::engine_refusal()),
                                Some(Err(why)) => kprintln!("  refused: {}", why.why()),
                                Some(Ok(c)) => {
                                    console::set_color(if c.adopted { LTGREEN } else { YELLOW });
                                    kprintln!(
                                        "  {} -- fixed {} broke {} chi {}",
                                        if c.adopted { "adopted" } else { "rejected" },
                                        c.fixed,
                                        c.broke,
                                        c.mcnemar
                                    );
                                    console::set_color(LTGRAY);
                                    kprintln!("  'godel ledger' for the line, 'godel rollback' to undo");
                                }
                            }
                        }
                    }
                }
                "off" => {
                    if voter::uninstall() {
                        kprintln!("  the written core is out of the decision path");
                    } else {
                        kprintln!("  none was installed");
                    }
                }
                other => kprintln!("  no such action: {}", other),
            }
        }
        "godel" => {
            use crate::ai::godel;
            let mut words = rest.split_whitespace();
            match words.next().unwrap_or("") {
                "" | "status" => {
                    let (trials, adoptions, reads) = godel::counts();
                    let (used, cap, fresh) = godel::test_status();
                    console::set_color(YELLOW);
                    kprintln!("[godel]");
                    console::set_color(LTGRAY);
                    // The RTC hour is printed beside the window because they
                    // are compared directly and nothing here knows the offset
                    // between that clock and the wall. Under QEMU it is UTC;
                    // on a machine that dual-boots Windows it is usually local.
                    let (from, until) = godel::window();
                    kprintln!(
                        "  {}, quiet window {:02}:00-{:02}:00 by the rtc",
                        if godel::enabled() { "armed" } else { "off" },
                        from,
                        until
                    );
                    match godel::rtc_hour() {
                        Some(h) => kprintln!("  the rtc says hour {:02}  ('godel window <from> <until>' to align)", h),
                        None => kprintln!("  no rtc, so no window and no trials"),
                    }
                    match godel::quiet_now() {
                        Ok(h) => kprintln!("  eligible now (hour {}, no hardware input)", h),
                        Err(why) => kprintln!("  not eligible: {}", why),
                    }
                    kprintln!("  {} trial(s), {} adopted", trials, adoptions);
                    // The test slice is the one resource a self-improving loop
                    // spends without noticing, so its balance is status, not
                    // a footnote.
                    kprintln!(
                        "  test slice read {}/{} times{}",
                        used,
                        cap,
                        if fresh { "" } else { "  -- any test figure is now stale" }
                    );
                    let _ = reads;
                    let line = godel::lineage(8);
                    match godel::head() {
                        None => kprintln!("  head: none (the frozen model is the variant)"),
                        Some(_) => {
                            kprintln!("  lineage, newest first:");
                            for (h, a) in line.iter() {
                                kprintln!(
                                    "    {}  adapter {}",
                                    godel::short_hex(h),
                                    a.map(|x| godel::short_hex(&x))
                                        .unwrap_or(alloc::string::String::from("none"))
                                );
                            }
                        }
                    }
                }
                "now" => {
                    // Forced, so the quiet window does not apply: the operator
                    // asking for a trial *is* the consent the window stands in
                    // for the rest of the time.
                    let mut b = crate::ai::train::Budget::default();
                    for w in words.clone() {
                        if let Ok(v) = w.parse::<usize>() {
                            b.examples = v;
                        }
                    }
                    godel::report_trial(&b);
                }
                // The whole apparatus in one command: a generation trained
                // against one set of cached features, bred, filed, and the
                // best of it put in front of the same judge the nightly loop
                // uses.
                "storm" => {
                    // `godel storm [points] [examples]`. Examples is bounded
                    // and not "all" by default, which the first run of this
                    // command found the hard way: `prepare` over the whole
                    // corpus is about twenty-two minutes under WHPX, so a
                    // storm that defaulted to it could not finish inside any
                    // harness timeout and printed nothing at all. A command
                    // somebody types has to answer.
                    let mut b = crate::ai::train::Budget::default();
                    let mut points = 8usize;
                    let mut nth = 0usize;
                    for w in words.clone() {
                        if let Ok(v) = w.parse::<usize>() {
                            if nth == 0 {
                                points = v;
                            } else {
                                b.examples = v;
                            }
                            nth += 1;
                        }
                    }
                    if b.examples == 0 {
                        b.examples = 64;
                    }
                    kprintln!(
                        "  {} point(s) over {} example(s) -- one prepare, then no forward passes",
                        points, b.examples
                    );
                    match crate::ai::with_engine(|e| godel::storm(e, &b, points)) {
                        Some(Ok(r)) => {
                            console::set_color(YELLOW);
                            kprintln!(
                                "[storm]  {} trained, {} descendant(s) of the incumbent, {} chimera(s) bred",
                                r.trained, r.descendants, r.chimeras
                            );
                            console::set_color(LTGRAY);
                            kprintln!(
                                "         archive: {} cell(s) lit; best validation {}%",
                                r.lit,
                                (r.best * 100.0) as u32
                            );
                            kprintln!(
                                "         tribunal: the best was {} -- {}",
                                if r.adopted { "accepted" } else { "rejected" },
                                r.verdict
                            );
                        }
                        Some(Err(why)) => kprintln!("  no storm: {}", why.why()),
                        // The mind is mid-thought and holds the engine. Not an
                        // error, and not a storm either.
                        None => kprintln!("  the engine is claimed -- try again when the mind is idle"),
                    }
                }
                // What the loop has left to try, which is the question the
                // status line could not answer before there was a search.
                "space" => {
                    let (seen, all) = godel::explored();
                    kprintln!("  {} of {} point(s) tried", seen, all);
                    match godel::frontier() {
                        Some(p) => kprintln!(
                            "  next: lr {}, rank {}, alpha {}, epochs {}, rule {}",
                            p.lr, p.rank, p.alpha, p.epochs, p.rule
                        ),
                        None => {
                            kprintln!("  the grid is spent -- every point has been trained and judged");
                            kprintln!("  from here the night loop composes a core instead, which is");
                            kprintln!("  a space it writes rather than one it was given");
                            kprintln!("  'core author' to see one now; 'godel forget' re-walks the grid");
                        }
                    }
                }
                // Judge a routing rule on calibration.
                //
                // The one axis where accuracy is the wrong judge: what a rule
                // moves is how much better the council's confident answers are
                // than its unconfident ones, and J1 as written would veto a
                // trade of a point of accuracy for a much sharper signal.
                "rule" => {
                    let want = words.next().unwrap_or("");
                    let picked = match want {
                        "probe" => Some(0u8),
                        "majority" => Some(1),
                        "lexical" => Some(2),
                        "withcore" => Some(3),
                        _ => None,
                    };
                    match picked {
                        None => {
                            kprintln!("  usage: godel rule probe|majority|lexical|withcore");
                            kprintln!("  in force: {}", crate::ai::harness::rule_in_force().name());
                        }
                        Some(r) => {
                            let p = godel::Proposal::config(r);
                            let b = p.budget(0, 0);
                            match crate::ai::with_engine(|e| godel::run(e, &b, &p)) {
                                None => kprintln!("  {}", crate::ai::engine_refusal()),
                                Some(Err(why)) => kprintln!("  refused: {}", why.why()),
                                Some(Ok(c)) => {
                                    console::set_color(if c.adopted { LTGREEN } else { YELLOW });
                                    kprintln!(
                                        "  {} -- {}",
                                        if c.adopted { "adopted" } else { "rejected" },
                                        c.j1_why
                                    );
                                    console::set_color(LTGRAY);
                                    kprintln!(
                                        "  accuracy: fixed {} broke {} chi {}",
                                        c.fixed, c.broke, c.mcnemar
                                    );
                                    kprintln!(
                                        "  confident on {} item(s), was {}",
                                        c.goals_held, c.goals_total
                                    );
                                    kprintln!("  J2 calibration {}", if c.j2 { "pass" } else { "VETO" });
                                }
                            }
                        }
                    }
                }
                // What the loop would try tonight, without trying it.
                "next" => {
                    let (boundary, slots) = godel::rotation();
                    kprintln!(
                        "  {} verdict(s) recorded, so the bar is {}",
                        godel::ledger_len(),
                        if boundary { "open to question tonight" } else { "frozen for this epoch" }
                    );
                    for (i, (name, unc, has)) in slots.iter().enumerate() {
                        let mark = if i == 0 { "->" } else { "  " };
                        kprintln!(
                            "  {} {:8} surprise {}  {}",
                            mark,
                            name,
                            (unc * 100.0) as u32,
                            if *has { "has work" } else { "spent" }
                        );
                    }
                    kprintln!("     core     composes on demand, and is last regardless");
                    kprintln!("  it takes the most surprising axis that still has work");
                }
                "forget" => {
                    let n = godel::forget();
                    kprintln!("  {} marker(s) cleared -- the grid will be walked again", n);
                    kprintln!("  the nodes and the ledger are untouched, so a rediscovered");
                    kprintln!("  point lands on the hash it landed on before");
                }
                "ledger" => {
                    let n = words.next().and_then(|w| w.parse().ok()).unwrap_or(12);
                    let tail = godel::ledger_tail(n);
                    if tail.is_empty() {
                        kprintln!("  nothing recorded yet");
                    }
                    for l in tail.iter() {
                        kprintln!("  {}", l);
                    }
                }
                "rollback" => match crate::ai::with_engine(|e| godel::rollback(e)) {
                    None => kprintln!("  no engine, or another task holds it"),
                    Some(Err(why)) => kprintln!("  cannot roll back: {}", why),
                    Some(Ok(None)) => kprintln!("  back to the frozen model"),
                    Some(Ok(Some(h))) => kprintln!("  head is now {}", godel::short_hex(&h)),
                },
                "window" => {
                    let f = words.next().and_then(|w| w.parse::<u8>().ok());
                    let u = words.next().and_then(|w| w.parse::<u8>().ok());
                    match (f, u) {
                        (Some(f), Some(u)) if godel::set_window(f, u) => {
                            kprintln!("  quiet window is now {:02}:00-{:02}:00 by the rtc", f, u)
                        }
                        _ => kprintln!("  usage: godel window <from-hour> <until-hour>   (0-23)"),
                    }
                }
                "on" => {
                    godel::set_enabled(true);
                    kprintln!("  armed -- trials may run inside the quiet window");
                }
                "off" => {
                    godel::set_enabled(false);
                    kprintln!("  off -- nothing will change itself");
                }
                _ => kprintln!("  usage: godel [status|now [n]|ledger [n]|window <f> <u>|rollback|on|off]"),
            }
        }
        "adapter" => {
            const DEFAULT: &str = "/ai/adapter.bin";
            let mut words = rest.split_whitespace();
            let verb = words.next().unwrap_or("");
            let path = words.next().unwrap_or(DEFAULT);
            match verb {
                "save" => match crate::ai::with_engine(|e| e.model.save_adapters(path)) {
                    None => kprintln!("  no engine, or another task holds it"),
                    Some(None) => kprintln!("  nothing attached to save"),
                    Some(Some(n)) => {
                        kprintln!("  {} bytes -> {}", n, path);
                        kprintln!("  'snap' versions it like anything else here");
                    }
                },
                "load" => {
                    let blob = crate::sysbox::read_blob(path);
                    match blob {
                        None => kprintln!("  no such file: {}", path),
                        Some(b) => match crate::ai::with_engine(|e| e.model.load_adapters(&b)) {
                            None => kprintln!("  no engine, or another task holds it"),
                            Some(Err(e)) => {
                                kprintln!("  {}: {:?} -- nothing attached", path, e)
                            }
                            Some(Ok(n)) => {
                                kprintln!("  {} site(s) attached from {}", n, path);
                                // Addressable, even though nobody judged it.
                                //
                                // `deeptrain` used to be the other way an
                                // unrecorded adapter reached the live model
                                // and it goes through the judges now; this is
                                // the one that is left, and it has the same
                                // failure if left alone -- the lineage's
                                // account of the mind is silently wrong until
                                // some later trial notices, and `rollback` has
                                // nothing to step back to. The node carries
                                // the honest outside-arrival zeros.
                                let noted =
                                    crate::ai::with_engine(crate::ai::godel::record_current);
                                match noted.flatten() {
                                    Some(h) => kprintln!(
                                        "  unjudged, recorded as {} -- 'godel rollback' steps back over it",
                                        &crate::ai::godel::short_hex(&h)[..8]
                                    ),
                                    None => {}
                                }
                            }
                        },
                    }
                }
                "off" | "detach" => match crate::ai::with_engine(|e| e.model.detach_adapters()) {
                    None => kprintln!("  no engine, or another task holds it"),
                    Some(None) => kprintln!("  nothing was attached"),
                    Some(Some(_)) => kprintln!("  detached -- the frozen model is what runs now"),
                },
                "" | "status" => match crate::ai::with_engine(|e| {
                    e.model.adapters.as_ref().map(|a| {
                        let sites = a.qkv.iter().flatten().filter(|s| s.is_some()).count()
                            + a.cls.is_some() as usize;
                        (a.r, a.alpha, sites, a.resident_bytes())
                    })
                }) {
                    None => kprintln!("  no engine, or another task holds it"),
                    Some(None) => kprintln!("  none attached ('train adapter', or 'adapter load')"),
                    Some(Some((r, alpha, sites, bytes))) => {
                        kprintln!("  rank {}, alpha {}, {} site(s), {} KiB resident", r, alpha as u32, sites, bytes / 1024);
                    }
                },
                _ => kprintln!("  usage: adapter [status|save|load|off] [path]"),
            }
        }
        "train" => {
            // Two trainers behind one verb, told apart by the first word.
            // `train [epochs]` is the linear probe's head, unchanged; `train
            // adapter ...` is the QDoRA run against the model's own
            // classifier. Dispatched inside the arm rather than as a second
            // guarded arm, because a guard placed after the arms it guards is
            // unreachable and this file has already paid for that once.
            let mut words = rest.split_whitespace();
            match words.next() {
                Some("adapter") => {
                    let mut b = crate::ai::train::Budget::default();
                    let mut bad: Option<&str> = None;
                    while let Some(flag) = words.next() {
                        let value = words.next().unwrap_or("");
                        match (flag, value.parse::<u64>()) {
                            ("-e", Ok(v)) => b.epochs = v as usize,
                            ("-n", Ok(v)) => b.examples = v as usize,
                            ("-ms", Ok(v)) => b.millis = v,
                            ("-r", Ok(v)) => b.rank = v as usize,
                            ("-lr", _) => match value.parse::<f32>() {
                                Ok(v) => b.lr = v,
                                Err(_) => bad = Some(value),
                            },
                            _ => bad = Some(flag),
                        }
                    }
                    match bad {
                        Some(w) => {
                            kprintln!("  unrecognised: {}", w);
                            kprintln!("  usage: train adapter [-e epochs] [-n examples] [-ms budget] [-r rank] [-lr rate]");
                        }
                        None => crate::ai::harness::adapter_train_report(&b),
                    }
                }
                _ => {
                    let epochs: usize = rest.split_whitespace().next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(20);
                    crate::ai::harness::train_report(epochs);
                }
            }
        }
        "think" => {
            if rest.is_empty() {
                kprintln!("  usage: think <prompt>   (runs in the background)");
            } else if crate::ai::agent_busy() {
                kprintln!("  an agent episode is running -- 'think' waits");
            } else if crate::ai::think(rest) {
                kprintln!("  queued -- the shell stays yours while it runs");
            } else {
                kprintln!("  a request is already pending");
            }
        }
        "agent" => {
            // `agent [-n steps] [--trust full] goal words...` -- flags first,
            // the goal verbatim after them. Read-only by default: the grammar
            // the loop decodes under simply does not contain the mutating
            // applets unless full trust is asked for by name.
            if rest.trim() == "stop" {
                kprintln!("  {}", crate::ai::agent::request_abort());
                return;
            }
            // What the machine watched its own episodes do. Validity and
            // progress, and never whether the goal was met, because nothing
            // here can observe that.
            if rest.trim_start().starts_with("outcomes") {
                use crate::gfx::console::{self, LTGRAY, YELLOW};
                let n = rest.split_whitespace().nth(1)
                    .and_then(|w| w.parse().ok())
                    .unwrap_or(16);
                let rows = crate::ai::agent::outcomes(n);
                console::set_color(YELLOW);
                kprintln!("[agent outcomes]");
                console::set_color(LTGRAY);
                if rows.is_empty() {
                    kprintln!("  none recorded this boot");
                }
                for r in rows.iter() {
                    kprintln!("  {}", r);
                }
                kprintln!("  score is a stated convention, and nothing acts on it yet");
                return;
            }
            if rest.trim() == "skills" {
                let skills = crate::sysbox::skills();
                if skills.is_empty() {
                    kprintln!("  no skills in /ai/tools -- write one and 'run' it");
                } else {
                    for (name, desc) in skills {
                        kprintln!("  {} -- {}", name, desc);
                    }
                }
                return;
            }
            if let Some(name) = rest.trim().strip_prefix("learn") {
                let n = name.trim();
                match crate::ai::agent::learn(if n.is_empty() { None } else { Some(n) }) {
                    Ok(path) => {
                        kprintln!("  learned -> {}", path);
                        kprintln!("  'run {}' replays the procedure", path);
                    }
                    Err(why) => kprintln!("  {}", why),
                }
                return;
            }
            let mut max_steps = 6usize;
            let mut trust = crate::ai::harness::Trust::ReadOnly;
            let mut parts = rest.trim();
            loop {
                let mut w = parts.splitn(3, ' ');
                let flag = w.next().unwrap_or("");
                match flag {
                    "-n" => {
                        let v = w.next().unwrap_or("");
                        let tail = w.next().unwrap_or("");
                        match v.parse::<usize>() {
                            Ok(n) if n > 0 && n <= 32 => max_steps = n,
                            _ => {
                                kprintln!("  -n wants 1..32");
                                return;
                            }
                        }
                        parts = tail;
                    }
                    "--trust" => {
                        let v = w.next().unwrap_or("");
                        let tail = w.next().unwrap_or("");
                        if v == "full" {
                            trust = crate::ai::harness::Trust::Full;
                        } else {
                            kprintln!("  --trust wants 'full' (default is read-only)");
                            return;
                        }
                        parts = tail;
                    }
                    _ => break,
                }
            }
            if crate::ai::mind_busy() {
                kprintln!("  the mind is busy -- wait for it to finish first");
                return;
            }
            if !crate::sysbox::is_ready() {
                kprintln!("  no namespace to act on");
                return;
            }
            if parts.is_empty() {
                kprintln!(
                    "  usage: agent [-n steps] [--trust full] <goal>   (read-only unless told otherwise)"
                );
                return;
            }
            crate::ai::agent_run(parts, trust, max_steps);
        }
        "gen" => {
            // `gen -t 0 once upon a time` -- flags first, everything after is
            // the prompt verbatim, so it can contain spaces without quoting.
            let mut opts = crate::ai::GenOpts::default();
            let mut prompt = rest;
            loop {
                let mut it = prompt.splitn(3, ' ');
                let flag = it.next().unwrap_or("");
                if !matches!(flag, "-t" | "-p" | "-n") {
                    break;
                }
                let Some(value) = it.next() else { break };
                let tail = it.next().unwrap_or("");
                match flag {
                    "-t" => opts.temperature = value.parse().unwrap_or(opts.temperature),
                    "-p" => opts.topp = value.parse().unwrap_or(opts.topp),
                    _ => opts.steps = value.parse().unwrap_or(opts.steps),
                }
                prompt = tail.trim_start();
            }
            if prompt.is_empty() {
                prompt = "Once upon a time";
            }
            crate::ai::generate(prompt, &opts);
        }
        // Grouped rather than alphabetical: the list is long enough now that
        // finding a command matters more than enumerating them. Each group is
        // one subsystem, in the order you would meet them.
        "help" => {
            console::set_color(YELLOW);
            kprintln!("machine");
            console::set_color(WHITE);
            kprintln!("  mem uptime tasks cpu acpi pci video date reboot shutdown");
            kprintln!("  fault         deliberately dereference null");
            kprintln!("  clear refresh echo <text>");
            kprintln!("  log [all|save]  everything printed since power-on; the console keeps one screen");
            kprintln!("  paint write [path] mines oracle agentlog   desktop programs; todo   the checklist");
            kprintln!("  typewriter    output pacing, in us per character");
            kprintln!("  font          every glyph this machine can draw");
            kprintln!("  battery       charge, state and time remaining");
            kprintln!("  power         temperature, frequency and the governor");
            kprintln!("  usb [hid]     the USB bus; 'usb hid' for keyboards and mice");
            kprintln!("  gpu           the display controllers, and whether one answers");
            kprintln!("  abstract      what repeats across the skills, and what naming it would save");
            kprintln!("  study [n]     learn one field then another; what the second cost the first");
            kprintln!("  study seq [n] the same with an adapter carried through: real forgetting");

            console::set_color(YELLOW);
            kprintln!("\nstorage");
            console::set_color(WHITE);
            kprintln!("  nvme disk     controller and namespace state");
            kprintln!("  store         init/unlock/test/log/rollback -- 'store' for the list");
            kprintln!("  autosnap      snapshot the namespace on every write");
            kprintln!("  fat           read a FAT16/32 volume: fat ls|cat <path>");
            kprintln!("  pkg           list/info/add/remove content-addressed packages");
            kprintln!("  edit <path>   modal editor ('vi' works too)");
            kprintln!("  update        check, download and stage a signed kernel image");

            console::set_color(YELLOW);
            kprintln!("\nnetwork");
            console::set_color(WHITE);
            kprintln!("  net           link, addresses, ARP and resolver cache");
            kprintln!("  dhcp          ask the network for all of it");
            kprintln!("  net ip <addr> [gw]    or set it by hand");
            kprintln!("  dns <name>    resolve a name");
            kprintln!("  ping <host> [count]");
            kprintln!("  tcp           connection state");
            kprintln!("  tcp connect <host> <port> | send <text> | recv | close");
            kprintln!("  http <host>[:port] [/path]   fetch over HTTP/1.0");
            kprintln!("  https <host>[:port] [/path]  the same, over TLS 1.3");
            kprintln!("  wlan          what wireless hardware is present");
            kprintln!("  crypto        re-run the cipher test vectors");
            kprintln!("  rng [n]       random bytes, and what the pool thinks of itself");
            console::set_color(YELLOW);
            kprintln!("  <host> is a name or an address, anywhere one is taken");
            kprintln!("  https verifies the server against the roots.der on the ESP;");
            kprintln!("  with no roots it encrypts and authenticates nothing, and says so");
            console::set_color(WHITE);

            console::set_color(YELLOW);
            kprintln!("\nthe model");
            console::set_color(WHITE);
            // First, because it is the door to everything under it and it was
            // the one command in this section that appeared nowhere -- not
            // here, not on the wall, not in the Start menu, not in the Program
            // Manager. Four windows nobody could find.
            kprintln!("  mind open     the workspace: Workflows, Ask, Agent, Improve");
            kprintln!("  work ...      multi-agent runs      godel ...  self-modification");
            kprintln!("  author <name> <what it should do>   write an application");
            kprintln!("  gen <prompt>  generate text     ask <prompt>  chat turn");
            kprintln!("  think <p>     run it in the background, off the shell");
            kprintln!("  agent [-n n] [--trust full] <goal>   act-observe-repeat; 'agent stop' cancels");
            kprintln!("  agent outcomes [n]      what episodes did: dispatched, refused, circling");
            kprintln!("  act <task>    choose an applet by constrained decoding");
            kprintln!("  route <task>  choose one with the probe -- no transformer");
            kprintln!("  teach <applet> <task>   add an example ('teach file <path>' for many)");
            kprintln!("  teach bundle <path>     replace the whole corpus from one blob");
            kprintln!("  train adapter [-e N]    train the model's own decision layer (needs AVX2)");
            kprintln!("  adapter [save|load|off] what is attached, and moving it in and out");
            kprintln!("  godel [now|space|ledger|rollback]  the machine changing itself, on evidence");
            kprintln!("  core [list|judge|install]    a council voter the machine wrote");
            kprintln!("  deeptrain [-n N -e E -r R]   train q/k/v as well as the head");
            kprintln!("  simd                         what the CPU has, and what is enabled");
            kprintln!("  smp [bench]                  the other cores, and what they are worth");
            kprintln!("  ask [-t] <question>          one continuing conversation; 'ask new' forgets");
            kprintln!("  about [text]                 what the model should know about you");
            kprintln!("  skill [trust <hash>]         which skills keep operator powers");
            kprintln!("  sandbox [keep] <path>        run a program, see what it touched, undo it");
            kprintln!("  version                      what this build is");
            kprintln!("  diag [all|<name>]            run the self-tests, remember the verdicts");
            kprintln!("  update <image> [sig]         is a staged image signed by the update key");
            kprintln!("  fit [lambda]  refit the probe and the council on what it knows");
            kprintln!("  gate search   how often agreement is right; the config search");
            kprintln!("  ctx cont window logits probe feature zeroshot train");
            console::set_color(YELLOW);
            kprintln!("  'act' and 'route' are read-only unless you type 'trusted' first");
            console::set_color(WHITE);

            console::set_color(YELLOW);
            kprintln!("\nsysbox owns the namespace -- type 'sysbox' for its applets");
            kprintln!("a command may be piped into a filter: <cmd> | grep|head|tail|sort|wc");
            console::set_color(WHITE);

            console::set_color(YELLOW);
            kprintln!("\nanything else is evaluated as code -- 'words' and 'vars' to look around");
            console::set_color(WHITE);
            kprintln!("  x = 6*7               println(\"hi\", x)");
            kprintln!("  i=0 while(i<8){{ rect(i*40,300,32,32,i+1) i=i+1 }}");
            kprintln!("  hex(peek32(0xfec00000))");
        }
        "mem" => {
            let (used, total) = mem::heap::HEAP.stats();
            kprintln!("  heap  {} B used / {} B total", used, total);
            kprintln!("  free  {} KiB", (total - used) / 1024);
        }
        "uptime" => {
            let t = lapic::ticks();
            let hz = crate::TIMER_HZ as u64;
            kprintln!("  {} ticks  ({}.{:02} s at {} Hz)", t, t / hz, (t % hz) * 100 / hz, hz);
            kprintln!("  apic timer calibrated at {} Hz", lapic::timer_hz());
        }
        "battery" | "batt" => crate::dev::battery::report(),
        "ec" => crate::dev::ec::report(),
        // The ACPI path on its own, so it can be exercised without the
        // firmware answering first and hiding it.
        "acpi" if rest.trim() == "off" => match acpi {
            Some(a) => {
                kprintln!("  powering off through ACPI, skipping the firmware");
                if let Err(e) = crate::acpi::power_off(a) {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(LTGRAY);
                }
            }
            None => kprintln!("  ACPI was not parsed"),
        },
        "acpi" if rest.trim() == "s5" => match acpi {
            Some(a) => crate::acpi::s5_report(a),
            None => kprintln!("  ACPI was not parsed"),
        },
        "acpi" if rest.trim() == "unlock" => {
            crate::acpi::eval::allow_writes(true);
            console::set_color(LTRED);
            kprintln!("  region writes are on. A stray write to an embedded controller");
            kprintln!("  is a fan that stops or a charge threshold that moves, on hardware.");
            console::set_color(LTGRAY);
        }
        "acpi" if rest.trim().starts_with("load") => {
            crate::acpi::load_report(rest.trim()[4..].trim());
        }
        "acpi" if rest.trim().starts_with("eval") => match acpi {
            Some(a) => crate::acpi::eval_report(a, rest.trim()[4..].trim()),
            None => kprintln!("  ACPI was not parsed"),
        },
        "acpi" if rest.trim().starts_with("ns") => match acpi {
            Some(a) => crate::acpi::ns_report(a, rest.trim()[2..].trim()),
            None => kprintln!("  ACPI was not parsed"),
        },
        "acpi" if rest.trim() == "tables" => match acpi {
            Some(a) => crate::acpi::report(a),
            None => {
                console::set_color(LTRED);
                kprintln!("  ACPI was not parsed");
                console::set_color(WHITE);
            }
        },
        "acpi" => match acpi {
            Some(a) => {
                kprintln!("  revision {}  cpus {}", a.revision, a.cpus);
                kprintln!("  lapic    {:#x}  (id {})", a.lapic_addr, lapic::id());
                for i in 0..a.ioapic_count {
                    let io = a.ioapics[i];
                    kprintln!("  ioapic {} {:#x}  gsi base {}", io.id, io.addr, io.gsi_base);
                }
                for i in 0..a.override_count {
                    let o = a.overrides[i];
                    kprintln!("  irq {} -> gsi {}  flags {:#06x}", o.source, o.gsi, o.flags);
                }
                kprintln!("  {} tables ('acpi tables' for the list)", a.table_count);
            }
            None => {
                console::set_color(LTRED);
                kprintln!("  ACPI was not parsed");
                console::set_color(WHITE);
            }
        },
        "tasks" => {
            kprintln!(
                "  {} tasks, {} switches total",
                crate::task::count(),
                crate::task::total_switches()
            );
            for i in 0..crate::task::count() {
                if let Some(t) = crate::task::snapshot(i) {
                    let marker = if i == crate::task::current() { '*' } else { ' ' };
                    kprintln!(
                        "  {}{} {:<8} rsp {:#018x}  resumed {}",
                        marker,
                        i,
                        t.name,
                        t.rsp,
                        t.switches
                    );
                }
            }
            // Only advances while the clock task is actually on the CPU, so a
            // rising number here is proof of preemption rather than of a timer.
            kprintln!("  clock task iterations: {}", crate::clock_iterations());
            kprintln!(
                "  fpu state per task: {} B ({})",
                crate::task::fpu_area_bytes(),
                if crate::cpu::detected().avx_enabled { "xsave" } else { "fxsave" }
            );
            let checks = crate::ai::fpu_checks();
            let errs = crate::ai::fpu_errors();
            console::set_color(if errs == 0 { LTGREEN } else { LTRED });
            kprintln!("  ymm survived {} preemption checks, {} corrupted", checks, errs);
            console::set_color(WHITE);
        }
        "usb" => match acpi.as_ref().and_then(|a| a.mcfg) {
            Some(ecam) if rest.trim() == "hid" => {
                // Says nothing when it found nothing new, because the usual
                // reason for that is that boot already took them, and "0
                // configured" over a list of two working devices reads as a
                // failure.
                match crate::dev::usbhid::probe(ecam) {
                    Ok(0) => {}
                    Ok(n) => kprintln!("  {} newly configured", n),
                    Err(e) => kprintln!("  {}", e),
                }
                crate::dev::usbhid::report();
            }
            Some(ecam) => crate::dev::xhci::report(ecam),
            None => {
                console::set_color(LTRED);
                kprintln!("  no MCFG -- cannot reach PCI configuration space");
                console::set_color(WHITE);
            }
        },
        "gpu" => gpu_cmd(acpi),
        "abstract" => abstract_cmd(rest),
        "study" => study_cmd(rest),
        "work" => work_cmd(rest),
        "pci" => match acpi.as_ref().and_then(|a| a.mcfg) {
            Some(ecam) => {
                console::set_color(YELLOW);
                kprintln!("  bb:dd.f  vendor device  class");
                console::set_color(WHITE);
                let mut n = 0usize;
                crate::dev::pci::scan(ecam, 255, |d| {
                    n += 1;
                    let vname = crate::dev::pci::vendor_name(d.vendor);
                    kprintln!(
                        "  {:02x}:{:02x}.{}  {:04x} {:04x}  {} {}",
                        d.bus,
                        d.dev,
                        d.func,
                        d.vendor,
                        d.device,
                        crate::dev::pci::class_name(d.class, d.subclass),
                        vname
                    );
                });
                kprintln!("  {} functions found via ecam at {:#x}", n, ecam);
            }
            None => {
                console::set_color(LTRED);
                kprintln!("  no MCFG table, so no ECAM base to scan");
                console::set_color(WHITE);
            }
        },
        "nvme" => {
            let Some(ecam) = acpi.as_ref().and_then(|a| a.mcfg) else {
                console::set_color(LTRED);
                kprintln!("  no MCFG, so no ECAM base");
                console::set_color(WHITE);
                return;
            };
            if !crate::dev::nvme::present() {
                kprintln!("  probing...");
                match crate::dev::nvme::init(ecam) {
                    Ok(()) => {}
                    Err(e) => {
                        console::set_color(LTRED);
                        kprintln!("  init failed: {:?}", e);
                        console::set_color(WHITE);
                        return;
                    }
                }
            }
            crate::dev::nvme::with(|n| {
                // A nested fn, not a closure: closure lifetime elision cannot
                // tie the returned &str back to the input slice.
                fn trim(b: &[u8]) -> &str {
                    let mut end = b.len();
                    while end > 0 && (b[end - 1] == b' ' || b[end - 1] == 0) {
                        end -= 1;
                    }
                    core::str::from_utf8(&b[..end]).unwrap_or("?")
                }
                console::set_color(YELLOW);
                kprintln!("[nvme]");
                console::set_color(WHITE);
                kprintln!("  model   {}", trim(&n.model));
                kprintln!("  serial  {}", trim(&n.serial));
                kprintln!(
                    "  ns {}  {} blocks x {} B = {} MiB",
                    n.nsid,
                    n.block_count,
                    n.block_size,
                    n.capacity_bytes() / (1024 * 1024)
                );
                console::set_color(if crate::dev::nvme::writes_unlocked() { LTRED } else { LTGREEN });
                kprintln!(
                    "  writes  {}",
                    if crate::dev::nvme::writes_unlocked() { "UNLOCKED" } else { "locked (read-only)" }
                );
                // Where, not just whether. "UNLOCKED" on its own reads as
                // "the disk is writable", and for a long time that was
                // exactly what it meant -- the gate held one bit and no
                // range. The window is the answer to the question an operator
                // is actually asking when they look at this line.
                if let Some((start, end)) = crate::dev::nvme::write_window() {
                    kprintln!(
                        "          only LBA {}..{} ({} block(s)); everything else is refused",
                        start,
                        end,
                        end.saturating_sub(start)
                    );
                }
                console::set_color(WHITE);

                // A page-aligned DMA buffer, not a stack array. An unaligned
                // buffer can straddle a page boundary, and the tail of the
                // transfer then never arrives.
                let Some(dma) = crate::dev::nvme::alloc_dma(4096) else {
                    kprintln!("  could not allocate a DMA buffer");
                    return;
                };
                match n.read(0, 1, dma) {
                    Ok(()) => {
                        let buf = unsafe { core::slice::from_raw_parts(dma, 512) };
                        kprint!("  lba0   ");
                        for b in buf.iter().take(16) {
                            kprint!("{:02x} ", b);
                        }
                        kprintln!();
                        let sig = u16::from_le_bytes([buf[510], buf[511]]);
                        if sig == 0xAA55 {
                            console::set_color(LTGREEN);
                            kprintln!("  boot signature 0xAA55 present -- a real partition table");
                        } else {
                            kprintln!("  no 0xAA55 signature (sig {:#06x})", sig);
                        }
                        console::set_color(WHITE);
                    }
                    Err(e) => {
                        console::set_color(LTRED);
                        kprintln!("  read failed: status {:#x}", e);
                        console::set_color(WHITE);
                    }
                }
            });
        }
        "disk" => {
            if !crate::dev::nvme::present() {
                console::set_color(LTRED);
                kprintln!("  no NVMe controller initialised -- run 'nvme' first");
                console::set_color(WHITE);
                return;
            }
            match crate::store::block::scan() {
                Ok(layout) => {
                    let bs = crate::store::block::block_size() as u64;
                    let total = crate::store::block::block_count();
                    console::set_color(YELLOW);
                    kprintln!("[disk]  scheme {:?}", layout.scheme);
                    console::set_color(WHITE);
                    kprintln!("  {} blocks x {} B = {} MiB total", total, bs, total * bs / (1024 * 1024));
                    if layout.partitions.is_empty() {
                        kprintln!("  no partitions");
                    }
                    for p in &layout.partitions {
                        kprintln!(
                            "  {}: lba {:>12}..{:<12} {:>8} MiB  {}",
                            p.index,
                            p.start_lba,
                            p.end_lba(),
                            p.block_count * bs / (1024 * 1024),
                            p.kind()
                        );
                    }
                    let used = layout.highest_used_lba();
                    let free = total.saturating_sub(used);
                    console::set_color(LTCYAN);
                    kprintln!(
                        "  highest claimed lba {}, {} MiB beyond it",
                        used,
                        free * bs / (1024 * 1024)
                    );
                    console::set_color(WHITE);
                    // The whole point of this command: say plainly what would
                    // be destroyed, before anything can write.
                    console::set_color(LTRED);
                    kprintln!("  every range above is IN USE -- writing to it destroys data");
                    console::set_color(WHITE);
                }
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  scan failed: {:?}", e);
                    console::set_color(WHITE);
                }
            }
        }
        "store" => store_cmd(rest),
        "recovery" => {
            // Available from the shell too, but the boot-time path is the one
            // that matters -- this is a convenience, not the design.
            let _ = crate::recovery::console();
        }
        "cpu" => {
            let max_ext = crate::cpu::cpuid(0x8000_0000, 0)[0];
            if max_ext >= 0x8000_0004 {
                // The brand string is 48 bytes spread across three leaves,
                // each returning it in eax/ebx/ecx/edx little-endian.
                let mut brand = [0u8; 48];
                for (i, leaf) in [0x8000_0002u32, 0x8000_0003, 0x8000_0004].iter().enumerate() {
                    let r = crate::cpu::cpuid(*leaf, 0);
                    for (j, v) in r.iter().enumerate() {
                        brand[i * 16 + j * 4..i * 16 + j * 4 + 4]
                            .copy_from_slice(&v.to_le_bytes());
                    }
                }
                let s = core::str::from_utf8(&brand).unwrap_or("?");
                kprintln!("  {}", s.trim_end_matches('\0').trim());
            }
            let v = crate::cpu::cpuid(0, 0);
            let mut vend = [0u8; 12];
            vend[0..4].copy_from_slice(&v[1].to_le_bytes());
            vend[4..8].copy_from_slice(&v[3].to_le_bytes());
            vend[8..12].copy_from_slice(&v[2].to_le_bytes());
            kprintln!("  vendor  {}", core::str::from_utf8(&vend).unwrap_or("?"));
            let f = crate::cpu::cpuid(1, 0);
            kprintln!("  family  {:#x}  apic id {}", f[0], f[1] >> 24);
            kprintln!("  lapic   {}", lapic::id());
            let s = crate::cpu::detected();
            kprintln!(
                "  simd    sse={} sse2={} sse4.1={} avx={} avx2={} fma={} f16c={} avx512f={}",
                s.sse as u8, s.sse2 as u8, s.sse41 as u8, s.avx as u8,
                s.avx2 as u8, s.fma as u8, s.f16c as u8, s.avx512f as u8
            );
            console::set_color(if s.avx_enabled { LTGREEN } else { LTRED });
            kprintln!("  avx state enabled by the OS: {}", s.avx_enabled);
            console::set_color(WHITE);
        }
        "reboot" => {
            console::set_color(YELLOW);
            kprintln!("  asking the firmware for a cold reset...");
            console::set_color(LTGRAY);
            crate::cpu::reboot();
        }
        "shutdown" | "poweroff" | "halt" => {
            console::set_color(YELLOW);
            kprintln!("  asking the firmware to power down...");
            console::set_color(LTGRAY);
            crate::cpu::shutdown();
        }
        "enternet" => {
            // Named for the dial-up software, and it browses about as much of
            // the web as that era did: no scripts, no images, no forms.
            crate::gfx::desk::open_browser(rest.trim());
        }
        "paint" => crate::gfx::desk::open_paint(),
        "oracle" | "godsays" => {
            // `-p` prints the machine's projected futures into the terminal,
            // which is how a serial run verifies it; bare, it opens the
            // timeline window. `godsays` is the ancestor's name and keeps
            // working.
            match rest.trim() {
                "-p" => crate::ai::futures_report(),
                _ => crate::gfx::desk::open_oracle(rest.trim()),
            }
        }
        "plan" => crate::ai::aixi::report(),
        // `rng` -- bytes from the kernel generator, and what it thinks of
        // itself. The status line is the interesting half: an estimate below
        // the threshold means `fill_secret` is refusing, which is what a TLS
        // handshake will report if one is attempted now.
        "rng" => {
            use crate::gfx::console::{self, LTGRAY, LTGREEN, YELLOW};
            let (deposits, bits, seeded) = crate::rng::status();
            match rest.trim().split_whitespace().next() {
                None | Some("") | Some("status") => {
                    console::set_color(YELLOW);
                    kprintln!("[rng]");
                    console::set_color(LTGRAY);
                    // Split by source, because a pool filled by a night of
                    // disk traffic is a different situation from one filled
                    // by somebody typing, and an operator deciding whether to
                    // trust a key wants to know which happened.
                    let dev = crate::rng::device_deposits();
                    kprintln!(
                        "  {} deposit(s): {} from input, {} from storage",
                        deposits,
                        deposits.saturating_sub(dev),
                        dev
                    );
                    kprintln!("  {} of {} bits credited", bits, crate::rng::SEEDED_BITS);
                    if seeded {
                        console::set_color(LTGREEN);
                        kprintln!("  seeded -- key material will be answered");
                    } else {
                        console::set_color(YELLOW);
                        kprintln!("  not seeded -- key material is refused, type at it");
                    }
                    console::set_color(LTGRAY);
                    kprintln!("  one bit credited per event, which is an assumption and not a measurement");
                }
                Some(w) => {
                    let n: usize = w.parse().unwrap_or(16).min(64);
                    let mut buf = [0u8; 64];
                    crate::rng::fill(&mut buf[..n]);
                    let mut hex = alloc::string::String::with_capacity(n * 2);
                    const D: &[u8; 16] = b"0123456789abcdef";
                    for b in buf.iter().take(n) {
                        hex.push(D[(b >> 4) as usize] as char);
                        hex.push(D[(b & 15) as usize] as char);
                    }
                    kprintln!("  {}", hex);
                    if !seeded {
                        console::set_color(YELLOW);
                        kprintln!("  unpredictable by timing, and below the bar for a secret");
                        console::set_color(LTGRAY);
                    }
                }
            }
        }
        "mind" if rest.trim() == "open" => {
            // The workspace. One command because the arrangement is the point:
            // a machine reporting on itself in four places at once reads as a
            // machine doing several things, which is what it is.
            crate::gfx::mindwin::open_workspace();
            kprintln!("  four windows, gridded -- Ask, Workflows, Agent, Improve");
        }
        "mind" => {
            use crate::gfx::console::{self, LTGRAY, YELLOW};
            let (ticks, acts, episodes, suppressed, enabled, seen, tenths) =
                crate::ai::initiative::status();
            console::set_color(YELLOW);
            kprintln!("[mind]");
            console::set_color(LTGRAY);
            kprintln!(
                "  initiative {} -- {} ticks: {} acted, {} episodes queued, {} stood down",
                if enabled { "on" } else { "off" },
                ticks,
                acts,
                episodes,
                suppressed
            );
            kprintln!(
                "  loop: {} iterations seen, timer at {}.{}s",
                seen,
                tenths / 10,
                tenths % 10
            );
            let tail = crate::ai::initiative::journal_tail(8);
            if tail.is_empty() {
                kprintln!("  the journal is empty; 'initiative now' forces a tick");
            }
            for line in tail {
                kprintln!("  {}", line);
            }
        }
        "initiative" => match rest.trim() {
            "off" | "quiet" => {
                crate::ai::initiative::set_enabled(false);
                kprintln!("[initiative] off -- the machine will wait to be asked");
            }
            "on" => {
                crate::ai::initiative::set_enabled(true);
                kprintln!("[initiative] on -- resident mind active");
            }
            "now" => {
                // The headless handle: one full perceive-decide-act cycle,
                // past silence and cooldown but never past busy or disabled.
                crate::ai::initiative::force_tick();
                kprintln!("[initiative] ticked -- see 'mind'");
            }
            other => kprintln!("  usage: initiative on|off|now (got '{}')", other),
        },
        "mines" | "minesweeper" => crate::gfx::desk::open_mines(),
        "agentlog" => crate::gfx::desk::open_agentlog(),
        "todo" => {
            // No args opens the runbook window -- what someone clicking the
            // icon wants. `-p` prints it to the terminal for a serial run;
            // `todo <n>` ticks a step; `todo reset` clears the ticks. One
            // list, shared with the window.
            use crate::gfx::todo;
            let arg = rest.trim();
            match arg {
                "" => {
                    crate::gfx::desk::open_todo();
                    return;
                }
                "reset" => todo::reset(),
                "-p" | "p" | "list" => {}
                n => match n.parse::<usize>() {
                    Ok(i) if todo::toggle(i) => {}
                    _ => {
                        kprintln!("  usage: todo [<n>|reset|-p]   (bare opens the window)");
                        return;
                    }
                },
            }
            console::set_color(YELLOW);
            kprintln!("[todo] hardware runbook");
            console::set_color(LTGRAY);
            for (i, s) in todo::STEPS.iter().enumerate() {
                kprintln!(
                    "  {:2}  {} [{:<6}] {}",
                    i,
                    if todo::is_done(i) { "x" } else { " " },
                    s.place.tag(),
                    s.title
                );
            }
            kprintln!(
                "  {} of {} done. 'todo <n>' details+ticks in the window; '-p' prints here.",
                todo::n_done(),
                todo::STEPS.len()
            );
        }
        // Hold a key down, or let it go.
        //
        // The typed equivalent of leaning on a key, and the only way anything
        // continuous is ever tested here: serial can send a keystroke but not
        // a *held* one -- there is no make without a break -- so a game that
        // reads held keys is a game no harness can drive. `win keys` sends
        // events; this sets state.
        //
        // Held keys survive across commands on purpose, and the point is that
        // where something ends up becomes arithmetic rather than an opinion
        // about a screenshot.
        //
        // **They do not survive into `doom play`, and this said they did.**
        // `play::run` clears the down-map on entry -- deliberately, because a
        // key the operator was leaning on is not a key the player pressed --
        // and then applies its own script. So `keys hold w` before it is
        // silently discarded; the keys go in the command instead,
        // `doom play 2000 w`, or with a time, `doom play 4000 w use@1500`.
        // The claim cost a run that reported a player who never moved.
        "keys" => {
            use crate::dev::kbd;
            let mut it = rest.split_whitespace();
            let verb = it.next().unwrap_or("");
            let named = |n: &str| -> Option<u8> {
                Some(match n {
                    "w" => kbd::SC_W,
                    "a" => kbd::SC_A,
                    "s" => kbd::SC_S,
                    "d" => kbd::SC_D,
                    "left" => kbd::SC_LEFT,
                    "right" => kbd::SC_RIGHT,
                    "up" => kbd::SC_UP,
                    "down" => kbd::SC_DOWN,
                    "shift" => kbd::SC_LSHIFT,
                    "ctrl" => kbd::SC_LCTRL,
                    "space" => kbd::SC_SPACE,
                    "esc" => kbd::SC_ESC,
                    _ => return None,
                })
            };
            match verb {
                "hold" | "release" => {
                    let down = verb == "hold";
                    let mut any = false;
                    for n in it {
                        match named(n) {
                            Some(c) => {
                                kbd::force_down(c, down);
                                any = true;
                                kprintln!("  {} {}", if down { "holding" } else { "released" }, n);
                            }
                            None => kprintln!("  no key called '{}'", n),
                        }
                    }
                    if !any {
                        kprintln!("  usage: keys hold|release w a s d left right up down shift ctrl space esc");
                    }
                }
                "clear" => {
                    kbd::clear_down();
                    kprintln!("  nothing held");
                }
                "" | "status" => {
                    let names = [
                        ("w", kbd::SC_W), ("a", kbd::SC_A), ("s", kbd::SC_S), ("d", kbd::SC_D),
                        ("left", kbd::SC_LEFT), ("right", kbd::SC_RIGHT),
                        ("up", kbd::SC_UP), ("down", kbd::SC_DOWN),
                        ("shift", kbd::SC_LSHIFT), ("ctrl", kbd::SC_LCTRL),
                        ("space", kbd::SC_SPACE), ("esc", kbd::SC_ESC),
                    ];
                    let mut held = alloc::string::String::new();
                    for (n, c) in names.iter() {
                        if kbd::is_down(*c) {
                            if !held.is_empty() {
                                held.push(' ');
                            }
                            held.push_str(n);
                        }
                    }
                    if held.is_empty() {
                        kprintln!("  nothing held");
                    } else {
                        kprintln!("  held: {}", held);
                    }
                }
                other => kprintln!("  no such action: {}  (try: hold, release, clear, status)", other),
            }
        }
        "win" => {
            use crate::gfx::desk;
            let mut it = rest.split_whitespace();
            match it.next().unwrap_or("") {
                "" | "list" => {}
                "next" => desk::cycle(false),
                // Every gesture needs a typed equivalent: serial cannot hold
                // Alt and Shift, and a control with no typed form never gets
                // tested.
                "prev" => desk::cycle(true),
                // The typed equivalent of the wheel over the terminal. Serial
                // cannot inject PS/2 packets, so a scrollback reachable only
                // by a wheel is a scrollback nothing ever checks -- the same
                // reason `win keys` exists at all.
                "scroll" => {
                    let arg = it.next().unwrap_or("");
                    let by: isize = match arg {
                        "end" | "" => -(crate::gfx::console::view_of(
                            crate::gfx::console::USER,
                        ) as isize),
                        n => n.parse().unwrap_or(0),
                    };
                    let moved = crate::gfx::console::scroll_view(
                        crate::gfx::console::USER,
                        by,
                    );
                    // Read *before* printing. `kprintln!` writes, and every
                    // write snaps the view back to the tail -- so a message
                    // that read the view while rendering itself would always
                    // report zero, which is exactly what it did.
                    let now = crate::gfx::console::view_of(crate::gfx::console::USER);
                    let kept = crate::gfx::console::history_of(crate::gfx::console::USER);
                    if moved {
                        desk::draw();
                    }
                    kprintln!("  {} row(s) back of {} kept", now, kept);
                }
                // Focus the terminal without reaching for the mouse.
                //
                // `open` leaves the new window focused on purpose, which is
                // right for a person and leaves a headless driver typing into
                // whatever it just opened: the next line goes to the panel and
                // its Enter presses whatever the panel had focused. That is
                // how the first test of an application deleted the item it had
                // just added.
                "term" => desk::focus_terminal(),
                // Clearing the screen, and clearing the screen around one
                // thing. Neither is the other with an argument, so they are
                // two verbs. Minimised and not closed -- the terminal and the
                // Program Manager are `closable: false`, and a command that
                // destroyed the shell to tidy the screen would be one nobody
                // could come back from.
                "only" | "solo" => desk::minimise_others(),
                "min" | "hide" => {
                    if !desk::minimise_focused() {
                        kprintln!("  nothing focused");
                    }
                }
                "clear" | "none" => {
                    desk::minimise_all();
                    desk::draw();
                }
                // `win round [n]` -- the focused window's corner radius, 0 for
                // a plain rectangle. Exposed on the shell rather than settled
                // in the theme because whether rounded corners belong on a
                // desktop that otherwise looks like 98 is a taste question,
                // and the mechanism should not depend on the answer.
                // Tiling. The gesture an operator arriving from any modern
                // desktop tries first, and the one `snap_release` could only
                // half answer because a drag has no keyboard.
                "tile" => {
                    use desk::Tile;
                    let which = it.next().unwrap_or("grid");
                    let t = match which {
                        "left" | "l" => Some(Tile::Left),
                        "right" | "r" => Some(Tile::Right),
                        "tl" | "topleft" => Some(Tile::TopLeft),
                        "tr" | "topright" => Some(Tile::TopRight),
                        "bl" | "bottomleft" => Some(Tile::BottomLeft),
                        "br" | "bottomright" => Some(Tile::BottomRight),
                        "full" | "max" => Some(Tile::Full),
                        _ => None,
                    };
                    match t {
                        Some(t) => match desk::tile_focused(t) {
                            true => kprintln!("  tiled {}", which),
                            false => kprintln!("  no focused window"),
                        },
                        None if which == "grid" || which == "all" => {
                            match desk::tile_all() {
                                0 => kprintln!("  nothing to tile"),
                                n => kprintln!("  {} window(s) into a grid", n),
                            }
                        }
                        // The workbench, by name. Without this the only way
                        // back to it was `mind open`, which re-opens the four
                        // windows rather than re-placing them -- so the
                        // workspace's own "Grid" button was a one-way door out
                        // of the layout `tile_workspace`'s doc argues for.
                        None if which == "workspace" || which == "work" => {
                            match desk::tile_workspace("Workflows", "Ask", "Agent", "Improve") {
                                0 => kprintln!("  the workspace is not open -- 'mind open'"),
                                n => kprintln!("  {} window(s) back to the workbench", n),
                            }
                        }
                        None => kprintln!(
                            "  usage: win tile left|right|tl|tr|bl|br|full|grid|workspace"
                        ),
                    }
                }
                "round" => {
                    let r: u32 = it.next().and_then(|w| w.parse().ok()).unwrap_or(12);
                    match desk::set_round(r) {
                        true => kprintln!("  corner radius {}", r),
                        false => kprintln!("  no focused window"),
                    }
                }
                "open" => {
                    let what = it.next().unwrap_or("status");
                    match crate::gfx::ui::panel_named(what) {
                        // The panel names its own window. `win open wifi`
                        // asked for a page, not for a window called "wifi",
                        // and a title bar that echoes the command word tells
                        // the operator nothing they did not just type.
                        Some(p) => {
                            let title = p.title.clone();
                            let typing = p.wants_typing();
                            // A name that is a route keeps it, so the window
                            // can be rebuilt when what it shows changes.
                            if what.contains(':') {
                                desk::open_routed(&title, p, what);
                            } else if what == "search" {
                                // Asking for the search box twice should give
                                // the search box, not a second one beside it.
                                desk::show_routed(&title, p, what);
                            } else {
                                desk::open(&title, p);
                            }
                            // Opened from the prompt, so the operator is at
                            // the keyboard typing commands: leave them there --
                            // unless the panel is asking to be typed into, in
                            // which case that *is* where they are.
                            //
                            // `open` keeps focus on the new window, which is
                            // right when it was opened by a click on an icon
                            // or a menu -- attention is already on the window.
                            // It is wrong here, and not only for a headless
                            // driver: the desktop takes *every* key while a
                            // non-terminal window has focus, so the next line
                            // typed goes into the window that just appeared
                            // and its Enter presses whatever that window had
                            // focused. Opening a to-do list and typing the
                            // next command added the command as an item, and
                            // an earlier arrangement of the same panel deleted
                            // one instead.
                            if !typing {
                                desk::focus_terminal();
                            }
                        }
                        None => {
                            kprintln!("  no such panel: {}", what);
                            kprintln!("  try: status, programs");
                            return;
                        }
                    }
                }
                "keys" => {
                    // Feed keystrokes to the desktop as if typed. Alt-Tab,
                    // Alt-Space and the arrows have no wire representation over
                    // serial, so without this the window manager could only be
                    // driven by a person sitting at the machine -- and an
                    // interface that cannot be driven headlessly does not get
                    // tested.
                    for k in crate::gfx::ui::parse_keys(rest[4..].trim()) {
                        if let desk::Route::Shell(_) = desk::key(k) {
                            // The terminal had focus and would have typed it.
                        }
                    }
                }
                other => {
                    kprintln!("  no such action: {}", other);
                    kprintln!(
                        "  usage: win [list|next|prev|term|only|min|clear|scroll <n>|open <panel>|keys <spec>]"
                    );
                    return;
                }
            }
            desk::trace("windows");
        }
        "tensor" => {
            let ok = crate::ai::selftest();
            if ok {
                console::set_color(LTGREEN);
                kprintln!("  all tensor selftests passed");
            } else {
                console::set_color(LTRED);
                kprintln!("  TENSOR SELFTESTS FAILED");
            }
            console::set_color(WHITE);
        }
        "bench" => crate::ai::bench(),
        "model" => crate::ai::model_demo(),
        // The two pictures that come before a renderer: the WAD's own
        // palette, and its level seen from above. Both go through
        // `port::Surface`, which is the surface the renderer will draw into --
        // so the path being proven is the path that will be used, rather than
        // a convenient stand-in for it.
        "doom" => {
            // Each subcommand parses its own arguments, because no rule
            // covers all three. `pal [ms]`, `map [name] [ms]` and
            // `view [deg] [ms]` disagree about what a leading number means,
            // and two general rules have already been wrong here: reading the
            // duration first ate `map E1M1`, and then treating a trailing
            // number as the duration ate `view 45`.
            let mut it = rest.split_whitespace();
            let what = it.next().unwrap_or("");
            let args: alloc::vec::Vec<&str> = it.collect();
            let num = |i: usize| -> Option<u64> { args.get(i).and_then(|t| t.parse().ok()) };

            let Some(parsed) = crate::doom::open() else {
                kprintln!("  no WAD on the boot volume -- see 'wad'");
                return;
            };
            let w = match parsed {
                Ok(w) => w,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(LTGRAY);
                    return;
                }
            };
            let Some(pal) = w.lump("PLAYPAL") else {
                kprintln!("  this WAD has no PLAYPAL, so nothing can be coloured");
                return;
            };
            if pal.len() < 768 {
                console::set_color(LTRED);
                kprintln!("  PLAYPAL is {} bytes, short of one 768-byte palette", pal.len());
                console::set_color(LTGRAY);
                return;
            }
            kprintln!(
                "  PLAYPAL: {} palette(s) of 256",
                crate::doom::draw::palette_count(pal)
            );
            // The picture directory, read once for whichever subcommand wants
            // it -- and only for those. Loading it decodes every patch in the
            // WAD and composes every texture, which for a real IWAD is a few
            // hundred pictures and a couple of megabytes; `doom pal` and `doom
            // map` want none of it and used to be instant. A WAD with no
            // TEXTURE1 is a legal WAD -- a map-only PWAD has none -- so its
            // absence is reported rather than treated as an error.
            let wants_pics =
                matches!(what, "tex" | "flat" | "sprite" | "mug" | "view" | "play");
            let pics = if wants_pics { crate::doom::pic::Pics::load(&w) } else { None };
            let flats = if wants_pics { Some(crate::doom::pic::Flats::load(&w)) } else { None };
            let sprites =
                if wants_pics { Some(crate::doom::sprite::Sprites::load(&w)) } else { None };
            let art = crate::doom::pic::Art {
                playpal: pal,
                colormap: w.lump("COLORMAP"),
                pics: pics.as_ref(),
                flats: flats.as_ref(),
            };
            if wants_pics {
                match &pics {
                    Some(p) => kprintln!(
                        "  TEXTURE1: {} texture(s) over {} patch name(s)",
                        p.textures.len(),
                        p.patch_names.len()
                    ),
                    None => kprintln!("  no TEXTURE1/PNAMES, so walls draw flat"),
                }
                match flats.as_ref() {
                    Some(f) if !f.is_empty() => kprintln!(
                        "  {} flat(s){}",
                        f.len(),
                        if f.marked { "" } else { " (no F_START marker)" }
                    ),
                    _ => kprintln!("  no flats, so floors and ceilings stay open sky"),
                }
                match sprites.as_ref() {
                    Some(sp) if !sp.is_empty() => kprintln!("  {} sprite lump(s)", sp.len()),
                    _ => kprintln!("  no sprites, so things do not appear"),
                }
            }

            match what {
                "pal" | "" => {
                    let hold = num(0).unwrap_or(0);
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        crate::doom::draw::palette(&mut surf, pal);
                        surf.present();
                        wait_or(hold);
                    });
                    kprintln!("  drew 256 colours");
                }
                // One texture, composed and blown up. This is where a
                // decoding mistake is caught, because on a wall every one of
                // them reads as "the art looks odd" and here each reads as a
                // different obviously wrong picture.
                "tex" => {
                    let Some(p) = pics.as_ref() else {
                        kprintln!("  this WAD has no texture directory");
                        return;
                    };
                    if p.textures.is_empty() {
                        kprintln!("  its texture directory is empty");
                        return;
                    }
                    let named = args.first().map(|t| t.parse::<u64>().is_err()).unwrap_or(false);
                    let idx = if named {
                        match p.index_of(args[0]) {
                            Some(i) => i,
                            None => {
                                console::set_color(YELLOW);
                                kprintln!("  no texture called '{}'", args[0]);
                                console::set_color(LTGRAY);
                                let mut shown = 0;
                                for d in p.textures.iter() {
                                    kprintln!("    {} {}x{}", d.name, d.width, d.height);
                                    shown += 1;
                                    if shown == 12 {
                                        kprintln!("    ... and {} more", p.textures.len() - shown);
                                        break;
                                    }
                                }
                                return;
                            }
                        }
                    } else {
                        0
                    };
                    let hold = num(if named { 1 } else { 0 }).unwrap_or(0);
                    let Some(d) = p.def(idx) else { return };
                    kprintln!(
                        "  {} -- {}x{} from {} patch(es)",
                        d.name,
                        d.width,
                        d.height,
                        d.patches.len()
                    );
                    let mut zoom = 0usize;
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        zoom = crate::doom::draw::texture(&mut surf, p, idx, pal).unwrap_or(0);
                        surf.present();
                        wait_or(hold);
                    });
                    if zoom == 0 {
                        kprintln!("  it has no pixels");
                    } else {
                        kprintln!("  drew it at {}x", zoom);
                    }
                }
                // One flat, blown up. The companion to `tex`, and it catches
                // two things `tex` cannot: a flat is addressed by world
                // position, so its coordinates can be transposed or one of
                // them mirrored, and neither shows on a symmetric picture.
                "flat" => {
                    let Some(f) = flats.as_ref().filter(|f| !f.is_empty()) else {
                        kprintln!("  this WAD has no flats");
                        return;
                    };
                    let named = args.first().map(|t| t.parse::<u64>().is_err()).unwrap_or(false);
                    let idx = if named {
                        match f.index_of(args[0]) {
                            Some(i) => i,
                            None => {
                                console::set_color(YELLOW);
                                kprintln!("  no flat called '{}'", args[0]);
                                console::set_color(LTGRAY);
                                for i in 0..f.len().min(12) {
                                    if let Some(n) = f.name_at(i) {
                                        kprintln!("    {}", n);
                                    }
                                }
                                if f.len() > 12 {
                                    kprintln!("    ... and {} more", f.len() - 12);
                                }
                                return;
                            }
                        }
                    } else {
                        0
                    };
                    let hold = num(if named { 1 } else { 0 }).unwrap_or(0);
                    let Some(px) = f.at(idx) else { return };
                    match f.name_at(idx) {
                        Some(n) => kprintln!("  {} -- 64x64, {} bytes", n, px.len()),
                        None => return,
                    }
                    let mut zoom = 0usize;
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        zoom = crate::doom::draw::flat(&mut surf, px, pal).unwrap_or(0);
                        surf.present();
                        wait_or(hold);
                    });
                    if zoom == 0 {
                        kprintln!("  it is not 64x64");
                    } else {
                        kprintln!("  drew it at {}x", zoom);
                    }
                }
                // One sprite lump, blown up. Same eye-check as `tex` and
                // `flat`, and it is the one that shows transparency: a sprite
                // is mostly hole, so a decoder that filled its bounding box
                // draws a rectangle where a barrel should be.
                "sprite" => {
                    let Some(sp) = sprites.as_ref().filter(|s| !s.is_empty()) else {
                        kprintln!("  this WAD has no sprites");
                        return;
                    };
                    let named = args.first().map(|t| t.parse::<u64>().is_err()).unwrap_or(false);
                    let idx = if named {
                        // A four-character name is a sprite; six is a lump.
                        let want = args[0];
                        let found =
                            if want.len() == 4 { sp.still(want, b'A') } else { sp.index_of(want) };
                        match found {
                            Some(i) => i,
                            None => {
                                console::set_color(YELLOW);
                                kprintln!("  no sprite lump called '{}'", want);
                                console::set_color(LTGRAY);
                                for i in 0..sp.len().min(12) {
                                    if let Some(n) = sp.name_at(i) {
                                        kprintln!("    {}", n);
                                    }
                                }
                                if sp.len() > 12 {
                                    kprintln!("    ... and {} more", sp.len() - 12);
                                }
                                return;
                            }
                        }
                    } else {
                        0
                    };
                    let hold = num(if named { 1 } else { 0 }).unwrap_or(0);
                    let Some(bytes) = sp.bytes(idx) else { return };
                    let Some(nm) = sp.name_at(idx).copied() else { return };
                    let Some(patch) = crate::doom::pic::Patch::parse(nm, bytes) else {
                        console::set_color(LTRED);
                        kprintln!("  {} will not decode as a patch", nm);
                        console::set_color(LTGRAY);
                        return;
                    };
                    kprintln!(
                        "  {} -- {}x{}, centre {} from the left, feet {} from the top",
                        nm,
                        patch.width,
                        patch.height,
                        patch.left,
                        patch.top
                    );
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        crate::doom::draw::sprite(&mut surf, &patch, pal);
                        surf.present();
                        wait_or(hold);
                    });
                    kprintln!("  drew it");
                }
                // All eight facings of one sprite, side by side.
                //
                // The check a census cannot make. Counting says every thing
                // found a picture; it cannot say the picture is the *right*
                // one, and one frame of a level shows each monster from
                // exactly one bearing -- so a facing taken from the wrong
                // bucket, or a mirrored lump drawn unmirrored, renders
                // perfectly and is wrong. FreeDoom is the ideal subject
                // because it stores both forms: POSS has eight separate
                // lumps, while SARG has five and gets its other three
                // facings by mirroring.
                "mug" => {
                    let Some(sp) = sprites.as_ref() else {
                        kprintln!("  this WAD has no sprites");
                        return;
                    };
                    let named = args.first().map(|t| t.parse::<u64>().is_err()).unwrap_or(false);
                    let name = if named { args[0] } else { "POSS" };
                    let Some(frame) = crate::doom::sprite::Frame::load(sp, name, b'A') else {
                        console::set_color(YELLOW);
                        kprintln!("  no sprite called '{}' with a frame A", name);
                        console::set_color(LTGRAY);
                        return;
                    };
                    let (have, mirrored) = frame.found();
                    kprintln!(
                        "  {} -- {}",
                        name,
                        if frame.turns() {
                            "turns to face the viewer"
                        } else {
                            "one picture for every angle"
                        }
                    );
                    if frame.turns() {
                        kprintln!("  {} of 8 facing(s), {} of them mirrored", have, mirrored);
                    }
                    let hold = num(if named { 1 } else { 0 }).unwrap_or(0);
                    let mut drew = 0usize;
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        drew = crate::doom::draw::mugshot(&mut surf, &frame, pal);
                        surf.present();
                        wait_or(hold);
                    });
                    kprintln!("  drew {}", drew);
                }
                // What a map asks of the specials, against what is implemented.
                // The honest measure of the environment work: not how many
                // specials exist here, but how many a real level uses that do
                // not.
                "specials" => {
                    let marker = first_map(&w);
                    if marker.is_empty() {
                        kprintln!("  no map marker in this WAD");
                        return;
                    }
                    let lv = match crate::doom::level::Level::load(&w, &marker) {
                        Ok(l) => l,
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("  {}: {}", marker, e);
                            console::set_color(LTGRAY);
                            return;
                        }
                    };
                    let mut seen: alloc::vec::Vec<(u16, usize)> = alloc::vec::Vec::new();
                    for l in lv.linedefs.iter() {
                        if l.special == 0 {
                            continue;
                        }
                        match seen.iter_mut().find(|(s, _)| *s == l.special) {
                            Some((_, n)) => *n += 1,
                            None => seen.push((l.special, 1)),
                        }
                    }
                    seen.sort_unstable_by_key(|(s, _)| *s);
                    let (mut ok, mut miss, mut ok_lines, mut miss_lines) = (0, 0, 0, 0);
                    for (sp, n) in seen.iter() {
                        if crate::doom::specials::handled(*sp) {
                            ok += 1;
                            ok_lines += n;
                        } else {
                            miss += 1;
                            miss_lines += n;
                        }
                    }
                    kprintln!(
                        "  {}: {} distinct special(s) on {} line(s)",
                        lv.name,
                        seen.len(),
                        ok_lines + miss_lines
                    );
                    kprintln!("  handled: {} kind(s), {} line(s)", ok, ok_lines);
                    kprintln!("  missing: {} kind(s), {} line(s)", miss, miss_lines);
                    if miss > 0 {
                        let mut shown = 0;
                        for (sp, n) in seen.iter() {
                            if crate::doom::specials::handled(*sp) {
                                continue;
                            }
                            kprintln!("    {} x{}", sp, n);
                            shown += 1;
                            if shown == 16 {
                                kprintln!("    ... and {} more kind(s)", miss - shown);
                                break;
                            }
                        }
                    }
                }
                "map" => {
                    let named = args.first().map(|t| t.parse::<u64>().is_err()).unwrap_or(false);
                    let name = if named { args[0] } else { "" };
                    let hold = num(if named { 1 } else { 0 }).unwrap_or(0);
                    let marker = if name.is_empty() {
                        first_map(&w)
                    } else {
                        alloc::string::String::from(name)
                    };
                    if marker.is_empty() {
                        kprintln!("  no map marker in this WAD");
                        return;
                    }
                    let lv = match crate::doom::level::Level::load(&w, &marker) {
                        Ok(l) => l,
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("  {}: {}", marker, e);
                            console::set_color(LTGRAY);
                            return;
                        }
                    };
                    kprintln!("  {} -- {} linedefs from above", lv.name, lv.linedefs.len());
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        crate::doom::draw::overhead(&mut surf, &lv, pal);
                        surf.present();
                        wait_or(hold);
                    });
                    kprintln!("  drew it");
                }
                // The first three-dimensional picture. `doom view [deg] [ms]`
                // stands at the player start facing `deg`, so a frame can be
                // asked for from a known place -- which is the only way to
                // check a projection without a person at the keyboard.
                "view" => {
                    let marker = first_map(&w);
                    if marker.is_empty() {
                        kprintln!("  no map marker in this WAD");
                        return;
                    }
                    let lv = match crate::doom::level::Level::load(&w, &marker) {
                        Ok(l) => l,
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("  {}: {}", marker, e);
                            console::set_color(LTGRAY);
                            return;
                        }
                    };
                    let Some(start) = lv.player_start().copied() else {
                        console::set_color(YELLOW);
                        kprintln!("  no player start, so there is nowhere to stand");
                        console::set_color(LTGRAY);
                        return;
                    };
                    // The floor under the start, so the eye sits at a
                    // sensible height rather than at zero -- which in a room
                    // with a raised floor is inside it.
                    let floor = lv
                        .sector_at(start.x as i32, start.y as i32)
                        .map(|s| s.floor as f32)
                        .unwrap_or(0.0);

                    let deg: f32 = args
                        .first()
                        .and_then(|t| t.parse::<i32>().ok())
                        .map(|d| d as f32)
                        .unwrap_or(start.angle as f32);
                    let hold = num(1).unwrap_or(0);
                    let view = crate::doom::render::View {
                        x: start.x as f32,
                        y: start.y as f32,
                        z: floor + crate::doom::render::EYE_HEIGHT,
                        angle: deg * (crate::doom::math::PI / 180.0),
                    };
                    kprintln!(
                        "  {} from {},{} at z {} facing {} deg",
                        lv.name,
                        start.x,
                        start.y,
                        view.z as i32,
                        deg as i32
                    );
                    let billboards = match sprites.as_ref() {
                        Some(sp) => crate::doom::sprite::collect(&lv, sp),
                        None => crate::doom::sprite::Things::none(),
                    };
                    if let Some(sp) = sprites.as_ref() {
                        let (drew, placed, no_kind, no_lump) =
                            crate::doom::sprite::census(&lv, sp);
                        kprintln!(
                            "  {} thing(s) drawn, {} start(s), {} of unknown kind, {} with no picture",
                            drew,
                            placed,
                            no_kind,
                            no_lump
                        );
                        kprintln!(
                            "  {} of {} kind(s) turn to face the viewer, {} animate over {} state(s)",
                            billboards.turning(),
                            billboards.kinds(),
                            billboards.animated(),
                            billboards.states()
                        );
                    }
                    let mut lit = false;
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        surf.set_palette_rgb(pal);
                        let sky = crate::doom::draw::nearest(pal, 0x10, 0x18, 0x60);
                        let mut r = crate::doom::render::Renderer::new(&surf, &art);
                        lit = r.lit_from_wad;
                        r.frame(&mut surf, &lv, &art, view, sky);
                        let vscale = surf.width() as f32 / 2.0;
                        r.things(&mut surf, &lv, &billboards, &view, vscale);
                        surf.present();
                        wait_or(hold);
                    });
                    kprintln!(
                        "  drew a frame, lit by {}",
                        if lit { "the WAD's COLORMAP" } else { "a ramp built from PLAYPAL" }
                    );
                }
                // Standing in it. `doom play [ms]` -- bounded because a
                // program that runs until a keypress cannot be driven over a
                // serial line, which is how everything here is tested.
                "play" => {
                    let marker = first_map(&w);
                    if marker.is_empty() {
                        kprintln!("  no map marker in this WAD");
                        return;
                    }
                    let mut lv = match crate::doom::level::Level::load(&w, &marker) {
                        Ok(l) => l,
                        Err(e) => {
                            console::set_color(LTRED);
                            kprintln!("  {}: {}", marker, e);
                            console::set_color(LTGRAY);
                            return;
                        }
                    };
                    let hold = num(0).unwrap_or(0);
                    // Mouse motion to inject before the run starts, so
                    // turning is checkable as a number: 400 counts is
                    // 400 * 0.0439 = 17.6 degrees, and the run reports the
                    // angle it finished at. A scheduled form would need the
                    // script to carry something other than a key, and nothing
                    // yet wants to turn *during* a run.
                    let mouse = args
                        .iter()
                        .find_map(|t| t.strip_prefix("mouse=").and_then(|v| v.parse::<i32>().ok()))
                        .unwrap_or(0);
                    // A sector to report the ceiling height of, so a door is
                    // checkable as a number. `doom play 3000 w watch=1`.
                    let watch = args
                        .iter()
                        .find_map(|a| a.strip_prefix("watch="))
                        .and_then(|n| n.parse::<usize>().ok())
                        .unwrap_or(0);
                    // Keys named after the duration are held for the session.
                    // `w` holds from the start; `use@1500` presses at 1500 ms;
                    // a leading `-` releases instead, so `fire@200 -fire@300
                    // fire@400` is two shots. Options carrying `=` are not
                    // keys and are skipped.
                    let mut script: alloc::vec::Vec<(u8, u64, bool)> = alloc::vec::Vec::new();
                    for n in args.iter().skip(1) {
                        if n.contains('=') {
                            continue;
                        }
                        let (n, down) = match n.strip_prefix('-') {
                            Some(rest) => (rest, false),
                            None => (*n, true),
                        };
                        let (n, at) = match n.split_once('@') {
                            Some((k, t)) => (k, t.parse::<u64>().unwrap_or(0)),
                            None => (n, 0),
                        };
                        let n = &n;
                        use crate::dev::kbd;
                        let c = match *n {
                            "w" => kbd::SC_W,
                            "a" => kbd::SC_A,
                            "s" => kbd::SC_S,
                            "d" => kbd::SC_D,
                            "left" => kbd::SC_LEFT,
                            "right" => kbd::SC_RIGHT,
                            "shift" => kbd::SC_LSHIFT,
                            "use" | "space" => kbd::SC_SPACE,
                            "fire" | "ctrl" => kbd::SC_LCTRL,
                            _ => {
                                kprintln!("  no key called '{}'", n);
                                continue;
                            }
                        };
                        script.push((c, at, down));
                    }
                    let mut billboards = match sprites.as_ref() {
                        Some(sp) => crate::doom::sprite::collect(&lv, sp),
                        None => crate::doom::sprite::Things::none(),
                    };
                    let guns = match sprites.as_ref() {
                        Some(sp) => crate::doom::weapon::Guns::load(sp),
                        None => crate::doom::weapon::Guns::none(),
                    };
                    kprintln!("  {}  --  W/S walk, A/D strafe, arrows turn, ctrl fires, Esc quits", lv.name);
                    kprintln!(
                        "  {} thing(s) to draw, {} of {} kind(s) turning, {} animating",
                        billboards.len(),
                        billboards.turning(),
                        billboards.kinds(),
                        billboards.animated()
                    );
                    if hold != 0 {
                        kprintln!("  (running for {} ms)", hold);
                    }
                    // Before the run, because the run removes what it kills
                    // and comparing the end against itself says nothing.
                    let began_with = billboards.len();
                    if mouse != 0 {
                        let m = crate::dev::mouse::peek();
                        crate::dev::mouse::apply(mouse, 0, m.left, m.right, 0);
                    }
                    let mut out: Option<crate::doom::play::Stats> = None;
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            return;
                        };
                        out = crate::doom::play::run(
                            &mut surf,
                            &mut lv,
                            &art,
                            &mut billboards,
                            &guns,
                            hold,
                            &script,
                            watch,
                        );
                    });
                    match out {
                        None => kprintln!("  no player start, so there is nowhere to stand"),
                        Some(st) => {
                            let fps = if st.ms > 0 { st.frames as u64 * 1000 / st.ms } else { 0 };
                            kprintln!(
                                "  {} frames, {} tics in {} ms  ({} fps)",
                                st.frames,
                                st.tics,
                                st.ms,
                                fps
                            );
                            kprintln!(
                                "  sector {} ceiling {}, {} special(s) fired, {} still moving{}",
                                watch,
                                st.watched,
                                st.spawned,
                                st.moving,
                                if st.exited { ", LEVEL EXITED" } else { "" }
                            );
                            kprintln!(
                                "  {} of {} kind(s) animate, {} frame change(s)",
                                billboards.animated(),
                                billboards.kinds(),
                                st.flips
                            );
                            kprintln!(
                                "  {} shot(s), {} killed, {} thing(s) left of {}, {} gun frame(s)",
                                st.shots,
                                st.killed,
                                st.remaining,
                                began_with,
                                guns.len()
                            );
                            kprintln!(
                                "  {} monster(s), {} awake, nearest {}, {} damage taken{}",
                                st.monsters,
                                st.awake,
                                if st.nearest.is_finite() { st.nearest as i32 } else { -1 },
                                st.hurt_by,
                                if st.died { ", PLAYER DIED" } else { "" }
                            );
                            kprintln!("  {} random draw(s) spent", st.rolls);
                            kprintln!(
                                "  health {}, armour {}, {} bullet(s), {} key(s), {} picked up{}",
                                st.health,
                                st.armour,
                                st.bullets,
                                st.keys,
                                st.picked,
                                match st.refused {
                                    Some(crate::doom::player::Colour::Blue) =>
                                        ", refused by a blue door",
                                    Some(crate::doom::player::Colour::Yellow) =>
                                        ", refused by a yellow door",
                                    Some(crate::doom::player::Colour::Red) =>
                                        ", refused by a red door",
                                    None => "",
                                }
                            );
                            kprintln!(
                            "  ended at {},{} facing {} deg, lit by {}",
                                st.x as i32,
                                st.y as i32,
                                st.deg as i32,
                                if st.lit_from_wad { "COLORMAP" } else { "PLAYPAL" }
                            );
                            // The world runs at 35 Hz whatever the renderer
                            // manages, so this is the number that says whether
                            // it did. Reported rather than assumed.
                            let want = st.ms * 35 / 1000;
                            if st.tics as u64 + 2 < want {
                                console::set_color(YELLOW);
                                kprintln!("  the world fell behind: {} tics where {} were due", st.tics, want);
                                console::set_color(LTGRAY);
                            }
                        }
                    }
                }
                other => kprintln!(
                    "  no such action: {}  (try: pal, tex, flat, sprite, map, view [deg], play)",
                    other
                ),
            }
        }
        // One map, decoded. `map` alone takes the first marker it can find;
        // `map E1M1` names one.
        "map" => {
            let Some(parsed) = crate::doom::open() else {
                kprintln!("  no WAD on the boot volume -- see 'wad'");
                return;
            };
            let w = match parsed {
                Ok(w) => w,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(LTGRAY);
                    return;
                }
            };
            // A marker is an empty lump whose name looks like a map. Guessing
            // by name rather than by emptiness because plenty of legitimate
            // lumps are empty, and only these two spellings have ever been
            // used for a marker.
            let want = rest.trim();
            let name = if !want.is_empty() {
                alloc::string::String::from(want)
            } else {
                let mut found = alloc::string::String::new();
                for i in 0..w.len() {
                    if let Some(e) = w.at(i) {
                        let n = e.name.as_str();
                        let looks = (n.len() == 4 && n.starts_with('E') && n.contains('M'))
                            || (n.len() == 5 && n.starts_with("MAP"));
                        if looks && e.is_empty() {
                            found = alloc::string::String::from(n);
                            break;
                        }
                    }
                }
                found
            };
            if name.is_empty() {
                kprintln!("  no map marker in this WAD");
                return;
            }

            let lv = match crate::doom::level::Level::load(&w, &name) {
                Ok(l) => l,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}: {}", name, e);
                    console::set_color(LTGRAY);
                    return;
                }
            };
            kprintln!("  {}", lv.name);
            kprintln!(
                "  {} vertexes  {} linedefs  {} sidedefs  {} sector(s)",
                lv.vertexes.len(),
                lv.linedefs.len(),
                lv.sidedefs.len(),
                lv.sectors.len()
            );
            kprintln!(
                "  {} segs  {} subsectors  {} node(s)  {} thing(s)",
                lv.segs.len(),
                lv.subsectors.len(),
                lv.nodes.len(),
                lv.things.len()
            );
            let two = lv.linedefs.iter().filter(|l| l.two_sided()).count();
            kprintln!("  {} of the linedefs are two-sided", two);

            match lv.player_start() {
                None => {
                    console::set_color(YELLOW);
                    kprintln!("  no player start -- nothing could spawn here");
                    console::set_color(LTGRAY);
                }
                Some(t) => {
                    kprintln!("  player 1 starts at {},{} facing {} deg", t.x, t.y, t.angle);
                    // The BSP, walked. This is the first thing in the port
                    // that *uses* the level rather than reporting it, and it
                    // is the cheapest possible check that the tree is
                    // navigable: if the side test is mirrored or a child index
                    // is wrong, the descent lands somewhere impossible or not
                    // at all.
                    match lv.subsector_at(t.x as i32, t.y as i32) {
                        Some(ss) => kprintln!(
                            "  which is in a subsector of {} seg(s) from {}",
                            ss.count,
                            ss.first
                        ),
                        None => {
                            console::set_color(LTRED);
                            kprintln!("  and the BSP does not lead anywhere from there");
                            console::set_color(LTGRAY);
                        }
                    }
                }
            }

            let mut floor = i16::MAX;
            let mut ceil = i16::MIN;
            for s in lv.sectors.iter() {
                floor = floor.min(s.floor);
                ceil = ceil.max(s.ceiling);
            }
            if !lv.sectors.is_empty() {
                kprintln!("  floors from {} up to a ceiling of {}", floor, ceil);
                let s = &lv.sectors[0];
                kprintln!("  sector 0: {} / {}, light {}", s.floor_pic, s.ceiling_pic, s.light);
            }
        }
        // What a WAD contains. The parser cannot print -- it lives under a
        // tree that may not name `crate::` outside `crate::port`, so
        // `kprintln!` is out of reach -- which is why it has an `Error` type
        // with a `Display` and no diagnostics of its own. This does the
        // talking.
        "wad" => {
            let Some(parsed) = crate::doom::open() else {
                kprintln!("  no WAD on the boot volume");
                kprintln!("  put one at {} and reboot", crate::WAD_PATH);
                kprintln!("  DOOM1.WAD is id's; freedoom.github.io is the free one");
                return;
            };
            let w = match parsed {
                Ok(w) => w,
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(LTGRAY);
                    return;
                }
            };
            let arg = rest.trim();
            kprintln!(
                "  {}  {} lump(s)  {} B",
                w.kind().as_str(),
                w.len(),
                w.size()
            );
            match arg {
                // A named lump: say where it is and how big, and show the
                // first bytes. A WAD's own structure is the fastest way to
                // tell a truncated download from a wrong file, and the head of
                // a lump is where a format announces itself.
                name if !name.is_empty() && !name.starts_with('-') => {
                    let Some(e) = w.find(name) else {
                        kprintln!("  no lump called '{}'", name);
                        return;
                    };
                    kprintln!("  {}  {} B", e.name, e.len());
                    let d = w.data(e);
                    let n = d.len().min(32);
                    let mut hex = alloc::string::String::new();
                    let mut txt = alloc::string::String::new();
                    for b in &d[..n] {
                        hex.push_str(&alloc::format!("{:02x} ", b));
                        txt.push(if (0x20..0x7F).contains(b) { *b as char } else { '.' });
                    }
                    kprintln!("  {}", hex);
                    kprintln!("  {}", txt);
                }
                _ => {
                    // The whole directory is thousands of lines and would
                    // scroll every one of them past the operator. A summary
                    // and the first few, which is what says "this parsed" --
                    // `wad <name>` is there for the rest.
                    let show = 12.min(w.len());
                    for i in 0..show {
                        if let Some(e) = w.at(i) {
                            kprintln!("    {:8}  {:>8} B", e.name.as_str(), e.len());
                        }
                    }
                    if w.len() > show {
                        kprintln!("    ... {} more -- 'wad <name>' for one", w.len() - show);
                    }
                    // Two lumps worth naming: every IWAD has a palette, and a
                    // map marker is how you know it is a game rather than a
                    // resource pack.
                    for probe in ["PLAYPAL", "E1M1", "MAP01"] {
                        if let Some(e) = w.find(probe) {
                            kprintln!("  has {} ({} B)", e.name, e.len());
                        }
                    }
                }
            }
        }
        // The pointer, typed.
        //
        // Serial cannot inject PS/2 packets, so a capability reachable only by
        // hand is one nothing ever checks -- the argument `win keys` makes for
        // the keyboard, and `port bars` makes for anything full-screen. It
        // goes through `dev::mouse::apply`, which is the single point the PS/2
        // and USB roads already converge on, so what this injects is
        // indistinguishable from a real movement by construction rather than
        // by resemblance.
        "mouse" => {
            let mut it = rest.split_whitespace();
            match it.next() {
                None | Some("status") => {
                    let s = crate::dev::mouse::peek();
                    kprintln!(
                        "  pointer {}, at {},{}  buttons{}{}",
                        if crate::dev::mouse::present() { "present" } else { "absent" },
                        s.x,
                        s.y,
                        if s.left { " left" } else { "" },
                        if s.right { " right" } else { "" }
                    );
                }
                Some("move") => {
                    let dx = it.next().and_then(|t| t.parse::<i32>().ok());
                    let dy = it.next().and_then(|t| t.parse::<i32>().ok()).unwrap_or(0);
                    match dx {
                        None => kprintln!("  usage: mouse move <dx> [dy]"),
                        Some(dx) => {
                            let s = crate::dev::mouse::peek();
                            crate::dev::mouse::apply(dx, dy, s.left, s.right, 0);
                            let now = crate::dev::mouse::peek();
                            kprintln!("  moved {},{} -- pointer now at {},{}", dx, dy, now.x, now.y);
                        }
                    }
                }
                Some(w @ ("down" | "up" | "click")) => {
                    let which = it.next().unwrap_or("left");
                    let right = which == "right";
                    let down = w != "up";
                    let s = crate::dev::mouse::peek();
                    let (l, r) = if right { (s.left, down) } else { (down, s.right) };
                    crate::dev::mouse::apply(0, 0, l, r, 0);
                    // A click is a press and a release, and the release has to
                    // happen: a button left down is a button every later
                    // command in the script is holding.
                    if w == "click" {
                        crate::dev::mouse::apply(0, 0, s.left, s.right, 0);
                    }
                    kprintln!("  {} {}", w, which);
                }
                Some(other) => kprintln!("  no idea what '{}' is (try: status, move, down, up, click)", other),
            }
        }
        // The seam a ported program reaches this machine through, and the
        // only way to look at it before there is a program.
        "port" => {
            match rest.trim() {
                "" | "status" => {
                    kprintln!("  files a ported program can reach:");
                    let names = crate::port::files::names();
                    if names.is_empty() {
                        kprintln!("    (none -- nothing was found on the boot volume)");
                    }
                    for n in names {
                        let len = crate::port::files::get(n).map(|b| b.len()).unwrap_or(0);
                        kprintln!("    {}  {} B", n, len);
                    }
                    kprintln!(
                        "  clock {}, screen {}",
                        if crate::port::clock::ready() { "calibrated" } else { "UNCALIBRATED" },
                        if crate::gfx::exclusive() { "held" } else { "free" }
                    );
                    kprintln!("  'port bars' draws a test frame through the surface");
                }
                // The whole indexed-frame path, with something a person can
                // check by looking. Every element of it is exercised -- the
                // 256-entry palette, the pre-encode into the screen's word
                // order, the row expansion, the integer scale and the blit --
                // and if any of them is wrong the picture is wrong in a way
                // that needs no debugger. A renderer put on top of an untested
                // present path is two unknowns at once.
                // `bars` waits for a key; `bars <ms>` returns on its own.
                //
                // The bounded form is not a convenience, it is the only way
                // this is ever tested. `drive.py` sends the next command when
                // it sees a prompt, and a program that owns the screen until
                // somebody presses a key never gives the prompt back -- so the
                // keystroke that would end it is the one command the harness
                // cannot deliver. The first run of this deadlocked exactly
                // there, with two commands unsent.
                //
                // Every full-screen program that follows needs the same thing.
                b if b == "bars" || b.starts_with("bars ") => {
                    let hold_ms: u64 = b
                        .strip_prefix("bars")
                        .map(|t| t.trim())
                        .filter(|t| !t.is_empty())
                        .and_then(|t| t.parse().ok())
                        .unwrap_or(0);
                    crate::port::with_screen(|| {
                        let Some(mut surf) = crate::port::Surface::new(320, 200) else {
                            kprintln!("  no framebuffer");
                            return;
                        };
                        // A palette that makes an index legible: sixteen hues
                        // by sixteen brightnesses, so a wrong byte lands in
                        // visibly the wrong place rather than merely the wrong
                        // shade.
                        for i in 0..256u32 {
                            let (hue, lev) = (i / 16, i % 16);
                            let v = (lev * 17) as u8;
                            let c = match hue {
                                0 => crate::gfx::Color::new(v, v, v),
                                1 => crate::gfx::Color::new(v, 0, 0),
                                2 => crate::gfx::Color::new(0, v, 0),
                                3 => crate::gfx::Color::new(0, 0, v),
                                4 => crate::gfx::Color::new(v, v, 0),
                                5 => crate::gfx::Color::new(0, v, v),
                                6 => crate::gfx::Color::new(v, 0, v),
                                7 => crate::gfx::Color::new(v, v / 2, 0),
                                _ => {
                                    let k = (hue - 8) as u8 * 32;
                                    crate::gfx::Color::new(v, k, 255u8.saturating_sub(v))
                                }
                            };
                            surf.set_colour(i as u8, c);
                        }
                        // Sixteen by sixteen blocks, one per palette entry, so
                        // the index at any point on screen is computable by
                        // eye from its position.
                        let (w, h) = (surf.width(), surf.height());
                        {
                            let px = surf.pixels();
                            for y in 0..h {
                                for x in 0..w {
                                    let cx = x * 16 / w;
                                    let cy = y * 16 / h;
                                    px[y * w + x] = (cy * 16 + cx) as u8;
                                }
                            }
                        }
                        surf.present();
                        let (ox, oy, sc) = surf.placement();
                        crate::serial_println!(
                            "[port] 320x200 at +{},{} scale {}",
                            ox,
                            oy,
                            sc
                        );
                        // Held until a key, so the clock has time to fail to
                        // paint over it -- ten seconds is a hundred chances at
                        // 10 Hz, and one digit would be visible.
                        crate::gfx::desk::trace("port bars");
                        let until = crate::port::clock::now_ms() + hold_ms;
                        loop {
                            if crate::dev::kbd::pop_any().is_some() {
                                break;
                            }
                            if hold_ms != 0 && crate::port::clock::now_ms() >= until {
                                break;
                            }
                            unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
                        }
                    });
                    kprintln!("  done");
                }
                other => kprintln!("  no such action: {}  (try: status, bars [ms])", other),
            }
        }
        "video" => {
            kprintln!(
                "  {}x{}  stride {}  {:?}",
                boot.fb.width(),
                boot.fb.height(),
                boot.fb.stride(),
                boot.fb.format()
            );
            if rest.trim() == "bench" {
                crate::gfx::bench();
                return;
            }
            if rest.trim() == "bars" {
                // Deferred from boot because the splash owned the screen.
                // Swapped red and blue mean the firmware's reported pixel
                // format is wrong, which is worth being able to re-check.
                let y = boot.fb.height().saturating_sub(80);
                let w = boot.fb.width() / 6;
                use crate::gfx::palette;
                boot.fb.rect(w, y, w, 40, palette::LTRED);
                boot.fb.rect(w * 2, y, w, 40, palette::LTGREEN);
                boot.fb.rect(w * 3, y, w, 40, palette::LTBLUE);
                kprintln!("  bars drawn: red green blue, left to right");
            } else {
                kprintln!("  'video bars' to check the pixel format");
            }
        }
        // `splash hold` leaves the boot screen up and returns to the prompt,
        // so a following `fault` lands while it owns the framebuffer. That is
        // the one path that cannot be checked by reading the code: the fault
        // reporter prints through the console, and if the console is still
        // hidden the diagnostic goes nowhere -- on the machine where the
        // framebuffer is the only output there is.
        "splash" if rest.trim() == "hold" => {
            crate::gfx::splash::begin();
            crate::gfx::splash::stage("held -- try 'fault'");
        }
        "splash" => {
            // Worth having beyond nostalgia: it is the only way to look at the
            // boot screen without rebooting, which is how it got laid out.
            crate::gfx::splash::begin();
            for s in ["a", "b", "c", "d", "e", "f", "g", "h"] {
                crate::gfx::splash::stage(s);
                crate::time::delay_us(120_000);
            }
            crate::gfx::splash::note("press a key");
            while crate::dev::kbd::pop_any().is_none() {
                unsafe { core::arch::asm!("hlt", options(nomem, nostack)) };
            }
            crate::gfx::splash::finish();
        }
        "echo" => kprintln!("  {}", rest),
        "clear" => console::with(|c| c.clear()),
        // `open`, not `find`: `find` is already a sysbox applet that searches
        // the namespace, and sysbox is dispatched before this match, so an arm
        // named `find` here is unreachable and says nothing about it. The same
        // trap `exec` fell into. Anything added to this match should be checked
        // against `sysbox::APPLETS` first.
        //
        // What a typed name means, in the order an operator expects.
        //
        // An application by that name opens. A command runs. Anything else is
        // not an error -- it is the interesting case, and the machine offers to
        // write it rather than doing so: writing holds the model for minutes,
        // and a keystroke that starts that silently is one that gets regretted.
        "open" => {
            let q = rest.trim();
            if q.is_empty() {
                kprintln!("  usage: open <name>");
                return;
            }
            let first = Words::new(q).word();
            if crate::app::exists(first) {
                kprintln!("  opening {}", first);
                let route = alloc::format!("app:{}", first);
                if let Some(p) = crate::gfx::ui::panel_named(&route) {
                    let title = p.title.clone();
                    crate::gfx::desk::open_routed(&title, p, &route);
                    crate::gfx::desk::focus_terminal();
                }
                return;
            }
            // A command, including every sysbox applet. Run verbatim, so
            // `find dhcp` does what typing `dhcp` does.
            if crate::sysbox::is_applet(first) || KNOWN_COMMANDS.contains(&first) {
                kprintln!("  running {}", q);
                crate::gfx::desk::queue_command(q);
                return;
            }
            // Nothing by that name. The offer goes into the same window the
            // query was typed into, which is what makes this a search box
            // rather than a window that spawns windows.
            let route = alloc::format!("search:{}", q);
            match crate::gfx::ui::panel_for_route(&route) {
                None => kprintln!("  nothing called '{}', and no way to offer one", q),
                Some((title, p)) => {
                    // Left in front, unlike everything else the shell opens.
                    //
                    // This one is a question, and its buttons are the answer.
                    // Handing focus back to the terminal raises the terminal
                    // over it -- the panel is wider than the gap to the right
                    // of the terminal, so the overlap covers exactly the left
                    // edge where the buttons are, and the operator is asked
                    // something they cannot see.
                    //
                    // The cost is the one this tree already knows: a headless
                    // driver typing blind after this sends its next line into
                    // the panel. `win term` is the way back.
                    crate::gfx::desk::show_routed(&title, p, &route);
                }
            }
        }
        // The machine writes an application.
        //
        // `author <name> <goal>` runs the loop and leaves a draft. Nothing is
        // adopted: `app try <name>` says what it managed and `app take <name>`
        // accepts it, which keeps a person between the machine's output and the
        // launcher.
        "author" => {
            let mut a = rest.splitn(2, ' ');
            let name = a.next().unwrap_or("").trim();
            let goal = a.next().unwrap_or("").trim();
            if name.is_empty() {
                kprintln!("  usage: author <name> <what it should do>");
                kprintln!("  author stop          ask a run to stop at its next step");
                kprintln!("  leaves a draft; 'app try' checks it, 'app take' adopts it");
                return;
            }
            // The keyboard half of the progress window's Stop button.
            //
            // Everything the pointer does here a keystroke must also do,
            // because serial cannot inject PS/2 packets and a control with no
            // typed equivalent never gets tested. `agent stop` is the same
            // thing and always was -- there is one queue and one abort flag --
            // so this is an alias that says so rather than a second mechanism.
            if name == "stop" {
                kprintln!("  {}", crate::ai::agent::request_abort());
                kprintln!("  ('agent stop' is the same thing)");
                return;
            }
            if !crate::ai::engine_ready() {
                kprintln!("  no model loaded, so there is nothing to ask");
                return;
            }
            let goal = if goal.is_empty() { name } else { goal };
            // Queued, not run here.
            //
            // It used to run inline, which held the shell for the minutes the
            // model took -- so the Stop button the run had just drawn could
            // not be reached from a keyboard that was busy waiting for the run
            // to end. Now the resident task does the work and the prompt comes
            // straight back, which is what `agent <goal>` has always done.
            if crate::ai::agent::queue_author(name, goal, AUTHOR_STEPS, false) {
                console::set_color(LTGREEN);
                kprintln!(
                    "  queued -- up to {} steps; the shell stays yours, 'agent stop' cancels",
                    AUTHOR_STEPS
                );
                console::set_color(LTGRAY);
                // Future tense: the draft does not exist yet. Printing the
                // path as though it did would send somebody to `app try` on
                // nothing, and read as the run having failed instantly.
                kprintln!("  it will leave /draft/{} -- then 'app try {}'", name, name);
            } else {
                console::set_color(YELLOW);
                kprintln!("  something is already pending or running");
                console::set_color(LTGRAY);
            }
        }
        // Applications. `app list`, `app show <name>`, and otherwise
        // `app <name> <fn> [args]`, which is the form a panel's own buttons
        // use -- so everything a stored panel can invoke is a function in its
        // own program and nothing in the shell's vocabulary.
        "app" => {
            let mut w = Words::new(rest);
            match w.next().unwrap_or("") {
                "" => {
                    kprintln!("  app list             applications on this machine");
                    kprintln!("  app show <name>      its panel document, rows filled in");
                    kprintln!("  app check <name>     what can be known without running it");
                    kprintln!("  app fix [name]       repair what can be repaired, or all of them");
                    kprintln!("  app draft <n> <kind> start one from a skeleton");
                    kprintln!("  app try <name>       check a draft, running it too");
                    kprintln!("  app take <name>      adopt a draft into /app");
                    kprintln!("  app drop <name>      throw a draft away");
                    kprintln!("  app info <name>      what it is, by hash, and what it may do");
                    kprintln!("  app trust <name> <h> approve exactly that version");
                    kprintln!("  app adopt <name>     record what is on disk as in use");
                    kprintln!("  app rollback <name>  back to what it replaced");
                    kprintln!("  app <name> <fn> [x]  call one of its functions");
                }
                "list" => {
                    let names = crate::app::names();
                    if names.is_empty() {
                        kprintln!("  none");
                    }
                    for n in names {
                        kprintln!("  {}", n);
                    }
                }
                // What this application is, by hash, and what it may do.
                "info" => match w.next() {
                    None => kprintln!("  usage: app info <name>"),
                    Some(name) => {
                        use crate::app::manifest as man;
                        match man::current(name) {
                            None => kprintln!("  no application '{}'", name),
                            Some(m) => {
                                let h = m.hash();
                                kprintln!("  manifest {}", man::hex32(&h));
                                match m.parent {
                                    None => kprintln!("  parent   none"),
                                    Some(p) => kprintln!("  parent   {}", man::hex32(&p)),
                                }
                                kprintln!("  panel    {}", man::hex32(&m.panel));
                                kprintln!("  code     {}", man::hex32(&m.code));
                                kprintln!(
                                    "  asks raw {}   granted {}",
                                    if m.raw { "yes" } else { "no" },
                                    if man::granted(&h) { "yes" } else { "no" }
                                );
                                match man::head(name) {
                                    None => kprintln!("  adopted  never"),
                                    Some(hd) => kprintln!(
                                        "  adopted  {}{}",
                                        man::hex32(&hd),
                                        if hd == h { "  (this one)" } else { "  (a different version)" }
                                    ),
                                }
                            }
                        }
                    }
                },
                // Approve one manifest to run with the operator's capabilities.
                //
                // The hash has to be typed back. An approval that could be
                // given without naming what is being approved would be given
                // to whatever happened to be on disk at the time, which is the
                // one thing a grant must not be -- the point is that it names
                // a specific artifact and stops applying the moment that
                // artifact changes.
                "trust" => {
                    use crate::app::manifest as man;
                    // `rest` was already split three ways, so the name and the
                    // hash are the next two pieces -- splitting again here
                    // would look at the name and find no hash in it.
                    let name = w.word();
                    let typed = w.word();
                    match man::current(name) {
                        None => kprintln!("  no application '{}'", name),
                        Some(m) => {
                            let h = m.hash();
                            let full = man::hex32(&h);
                            if typed.len() < 8 || !full.starts_with(typed) {
                                kprintln!("  this application is {}", full);
                                kprintln!("  asks for raw access: {}", if m.raw { "yes" } else { "no" });
                                kprintln!("  to approve exactly this version:");
                                kprintln!("    app trust {} {}", name, &full[..12]);
                                if !m.raw {
                                    kprintln!("  (it is not asking for anything, so this would do nothing)");
                                }
                            } else {
                                m.store();
                                let g = man::grant(name, &h);
                                let ad = man::adopt(name, &h);
                                kprintln!(
                                    "  {} trusted: grant {}, adopted {}",
                                    name,
                                    if g { "yes" } else { "failed" },
                                    if ad { "yes" } else { "failed" }
                                );
                            }
                        }
                    }
                }
                // Record what is on disk as the version in use, without
                // approving anything. This is how a change is accepted when it
                // asks for nothing.
                "adopt" => match w.next() {
                    None => kprintln!("  usage: app adopt <name>"),
                    Some(name) => {
                        use crate::app::manifest as man;
                        match man::current(name) {
                            None => kprintln!("  no application '{}'", name),
                            Some(m) => {
                                let h = m.store();
                                kprintln!(
                                    "  {} now at {}",
                                    name,
                                    if man::adopt(name, &h) { man::hex32(&h) } else { String::from("(failed)") }
                                );
                            }
                        }
                    }
                },
                "rollback" => match w.next() {
                    None => kprintln!("  usage: app rollback <name>"),
                    Some(name) => {
                        use crate::app::manifest as man;
                        match man::rollback(name) {
                            None => kprintln!("  nothing to roll back to"),
                            Some(p) => kprintln!("  {} back to {}", name, man::hex32(&p)),
                        }
                    }
                },
                // Drafts: an application being written, before anybody agrees
                // to run it. `app draft <name> <kind> [title]` starts one from
                // a skeleton, `app draft` alone lists the skeletons.
                "draft" => {
                    use crate::app::{draft, skel};
                    // `rest` was already split three ways, so the name is the
                    // next piece and everything after it is the third. Splitting
                    // the name again finds no kind in it.
                    let name = w.word();
                    let kind = w.word();
                    let title = w.rest();
                    if name.is_empty() {
                        kprintln!("  usage: app draft <name> <kind> [title]");
                        kprintln!("  skeletons:");
                        for sk in skel::SKELETONS {
                            kprintln!("    {:<9} {}", sk.kind, sk.what);
                        }
                        let d = draft::names();
                        if !d.is_empty() {
                            kprintln!("  drafts in progress: {}", d.join(", "));
                        }
                    } else if kind.is_empty() {
                        kprintln!("  usage: app draft {} <kind> [title]", name);
                    } else {
                        let title = if title.is_empty() { name } else { title };
                        match draft::create(name, kind, title, "Items") {
                            Err(e) => kprintln!("  {}", e),
                            Ok(()) => {
                                kprintln!("  drafted {}/{} from {}",
                                    draft::ROOT, name, kind);
                                kprintln!("  'app try {}' to check it, 'app take {}' to adopt",
                                    name, name);
                            }
                        }
                    }
                }
                // Everything that can be known about a draft, including the
                // parts that need running it. Safe here and nowhere else: smoke
                // testing writes state, and a draft's state is scratch.
                "try" => match w.next() {
                    None => kprintln!("  usage: app try <name>"),
                    Some(name) => {
                        use crate::gfx::console::{LTGREEN, LTRED};
                        let v = crate::app::draft::verdicts(name);
                        let mut bad = 0;
                        for r in &v {
                            console::set_color(if r.ok { LTGREEN } else { LTRED });
                            match r.line {
                                Some(l) => kprintln!("  {}  line {}: {}",
                                    if r.ok { "ok  " } else { "FAIL" }, l, r.why),
                                None => kprintln!("  {}  {}",
                                    if r.ok { "ok  " } else { "FAIL" }, r.why),
                            }
                            if !r.ok { bad += 1; }
                        }
                        console::set_color(LTGRAY);
                        kprintln!("  {} of {} checks failed", bad, v.len());
                    }
                },
                // Adopt a draft into /app. Refuses unless every check passes.
                "take" => match w.next() {
                    None => kprintln!("  usage: app take <name>"),
                    Some(name) => match crate::app::draft::adopt(name) {
                        Err(e) => kprintln!("  {}", e),
                        Ok(h) => {
                            kprintln!("  {} adopted as {}",
                                name, crate::app::manifest::hex32(&h));
                            kprintln!("  'win open app:{}' to use it", name);
                        }
                    },
                },
                "drop" => match w.next() {
                    None => kprintln!("  usage: app drop <name>"),
                    Some(name) => {
                        if crate::app::draft::abandon(name) {
                            kprintln!("  draft {} thrown away", name);
                            kprintln!("  anything ever adopted is addressed by hash and untouched");
                        } else {
                            kprintln!("  no draft called '{}'", name);
                        }
                    }
                },
                // Put right what can be put right, rather than describing it.
                //
                // `app fix` with no name sweeps everything, which is the shape
                // an operator actually wants: they know something is wrong and
                // not which application it is in.
                "fix" => {
                    use crate::gfx::console::{LTGREEN, LTRED, YELLOW};
                    let one = w.word();
                    let names: Vec<String> = if one.is_empty() {
                        crate::app::names()
                    } else {
                        alloc::vec![String::from(one)]
                    };
                    if names.is_empty() {
                        kprintln!("  no applications to look at");
                        return;
                    }
                    let mut touched = 0;
                    let mut left = 0;
                    for n in &names {
                        let o = crate::app::fix::fix(n);
                        if o.repairs.is_empty() && o.before == 0 {
                            continue;
                        }
                        kprintln!("  {}", n);
                        for r in &o.repairs {
                            console::set_color(if r.done { LTGREEN } else { YELLOW });
                            kprintln!("    {} {}", if r.done { "fixed " } else { "left  " }, r.what);
                        }
                        console::set_color(if o.after == 0 { LTGREEN } else { LTRED });
                        kprintln!("    {} fault(s) before, {} after", o.before, o.after);
                        console::set_color(LTGRAY);
                        if let Some(h) = o.adopted {
                            touched += 1;
                            kprintln!(
                                "    now {}  ('app rollback {}' undoes this)",
                                &crate::app::manifest::hex32(&h)[..12],
                                n
                            );
                        }
                        left += o.after;
                    }
                    if touched == 0 && left == 0 {
                        kprintln!("  nothing to repair in {} application(s)", names.len());
                    } else if left > 0 {
                        kprintln!("  {} fault(s) need a person: only removals are automatic,", left);
                        kprintln!("  because writing a function or repointing an action would");
                        kprintln!("  change what the application does rather than tidy it.");
                    }
                }
                // Everything that can be known about an application without
                // being told what it is for.
                "check" => match w.next() {
                    None => kprintln!("  usage: app check <name>"),
                    Some(name) => {
                        use crate::gfx::console::{LTGREEN, LTRED};
                        let v = crate::app::check::check_all("/app", name);
                        let mut bad = 0;
                        for r in &v {
                            console::set_color(if r.ok { LTGREEN } else { LTRED });
                            match r.line {
                                Some(l) => kprintln!("  {}  line {}: {}",
                                    if r.ok { "ok  " } else { "FAIL" }, l, r.why),
                                None => kprintln!("  {}  {}",
                                    if r.ok { "ok  " } else { "FAIL" }, r.why),
                            }
                            if !r.ok { bad += 1; }
                        }
                        console::set_color(LTGRAY);
                        if bad == 0 {
                            kprintln!("  nothing wrong with its form.");
                            kprintln!("  form is not function: these cannot tell whether it does");
                            kprintln!("  anything, only that what it does is well shaped.");
                        }
                    }
                },
                "show" => match w.next() {
                    None => kprintln!("  usage: app show <name>"),
                    Some(name) => match crate::app::document(name) {
                        None => kprintln!("  no application '{}'", name),
                        Some(d) => kprint!("{}", d),
                    },
                },
                name => {
                    let Some(func) = w.next() else {
                        kprintln!("  usage: app {} <fn> [args]", name);
                        return;
                    };
                    // The rest, not one word: `app todo add buy milk` has to
                    // keep its spaces, and the parent no longer glues them on.
                    let arg = w.rest();
                    match crate::app::call_fn(name, func, arg) {
                        Ok(v) => {
                            if !v.is_empty() {
                                kprintln!("  {}", v);
                            }
                        }
                        Err(e) => kprintln!("  app: {}", e),
                    }
                }
            }
        }
        // Whether the serial port is delivering by interrupt, and what it has
        // lost. Exists because the last attempt at interrupt-driven receive
        // could not be told apart from a port that was simply quiet, and a
        // guess was made instead of a measurement.
        "serial" => {
            let (held, irqs, overruns, spills) = crate::serial::rx_stats();
            kprintln!(
                "  {}  {} handler runs  {} waiting",
                if crate::serial::irq_live() { "interrupt-driven" } else { "polled" },
                irqs,
                held
            );
            kprintln!("  {} hardware overruns, {} ring spills", overruns, spills);
        }
        // The executive console, from the operator's. It is an output grid
        // with no prompt of its own, so the only way to act on it is from
        // here.
        "exec" => {
            let mut w = rest.split_whitespace();
            match w.next().unwrap_or("") {
                "clear" => {
                    console::with_ch(console::EXEC, |c| c.clear());
                    kprintln!("  executive console cleared");
                }
                "show" => {
                    crate::gfx::desk::show_executive();
                    kprintln!("  executive console raised");
                }
                _ => {
                    kprintln!("  exec clear   empty the executive console");
                    kprintln!("  exec show    raise its window");
                    kprintln!("  the boot log and anything the machine runs on");
                    kprintln!("  its own land there rather than here");
                }
            }
        }
        // Which single-core assumptions are actually shared.
        "racy" => {
            match rest.trim() {
                "on" => {
                    crate::sync::audit::start();
                    kprintln!("  watching. use the machine, then 'racy' to read it back");
                }
                "off" => {
                    crate::sync::audit::stop();
                    kprintln!("  stopped");
                }
                _ => crate::sync::audit::report(),
            }
        }
        // Who is using the memory.
        "census" => {
            if rest.trim() == "reset" {
                crate::mem::census::reset();
                kprintln!("  census cleared");
            } else {
                crate::mem::census::report();
            }
        }
        // Temperature, frequency, and the policy over both.
        "power" => {
            let mut w = rest.split_whitespace();
            match w.next().unwrap_or("") {
                "" | "status" => crate::dev::power::report(),
                "force" => {
                    let on = w.next() != Some("off");
                    crate::dev::power::force(on);
                    console::set_color(LTRED);
                    kprintln!("  forced {}. A register the part does not implement halts this machine.", on);
                    console::set_color(LTGRAY);
                    crate::dev::power::report();
                }
                "turbo" => match w.next() {
                    Some(v) => {
                        let want = v != "off";
                        if crate::dev::power::set_turbo(want) {
                            kprintln!("  turbo {}", if want { "on" } else { "off" });
                        } else {
                            kprintln!("  unavailable: {}", crate::dev::power::why());
                        }
                    }
                    None => match crate::dev::power::turbo() {
                        Some(t) => kprintln!("  turbo {}", if t { "on" } else { "off" }),
                        None => kprintln!("  unavailable: {}", crate::dev::power::why()),
                    },
                },
                g => match crate::dev::power::Governor::parse(g) {
                    Some(gov) => {
                        if crate::dev::power::set_governor(gov) {
                            kprintln!("  governor {}", gov.name());
                        } else {
                            kprintln!("  unavailable: {}", crate::dev::power::why());
                        }
                    }
                    None => {
                        kprintln!("  power                 what the part reports");
                        kprintln!("  power performance|balanced|powersave");
                        kprintln!("  power turbo [on|off]");
                        kprintln!("  power force [on|off]  permit MSRs under a hypervisor");
                    }
                },
            }
        }
        // Every glyph, to be looked at.
        "font" => {
            crate::gfx::font::report();
        }
        // What a file is, and what is in it.
        "file" => {
            let path = rest.trim();
            if path.is_empty() {
                kprintln!("  file <path>   what it is, and an outline of it");
            } else if let Some(bytes) = crate::sysbox::read_blob(path) {
                let name = path.rsplit('/').next().unwrap_or(path);
                let kind = crate::fmt::detect(name, &bytes);
                kprintln!("  {}  {} byte(s)", kind.name(), bytes.len());
                if kind.is_text() {
                    match core::str::from_utf8(&bytes) {
                        Ok(s) => {
                            let o = crate::fmt::outline::of(kind, s);
                            if o.is_empty() {
                                kprintln!("  no structure to report");
                            }
                            for e in o.iter().take(40) {
                                let pad = "  ".repeat(e.depth.min(6) as usize);
                                kprintln!("  {:>5}  {}{} {}", e.line, pad, e.kind, e.name);
                            }
                            if o.len() > 40 {
                                kprintln!("  ... {} more", o.len() - 40);
                            }
                        }
                        // The kind said text and the bytes disagree, which is
                        // worth saying rather than showing replacement
                        // characters and letting somebody wonder.
                        Err(e) => kprintln!("  not valid UTF-8 at byte {}", e.valid_up_to()),
                    }
                }
            } else {
                kprintln!("  no such file");
            }
        }
        "fault" => {
            console::set_color(LTRED);
            kprintln!("  this will halt the machine.");
            console::set_color(LTGRAY);
            // `fault code` takes the same fault from a heap buffer instead of
            // from the image, which is the only way to see the reporter name
            // a generated range. Both halt; they differ in what gets printed
            // on the way down.
            // Guarded: the same dereference, inside a guard, so it is caught
            // and the machine carries on. This is the live-system version of
            // what `diag recover` asserts, and it is worth having separately
            // because a suite runs with the desktop quiet and this does not.
            if rest.split_whitespace().next() == Some("guarded") {
                let r = crate::cpu::recover::guard(|| unsafe {
                    core::ptr::read_volatile(0x0 as *const u64);
                });
                console::set_color(LTGREEN);
                match r {
                    Err(what) => kprintln!("  caught: {}. still here.", what),
                    Ok(()) => kprintln!("  no fault happened, which is itself wrong"),
                }
                console::set_color(LTGRAY);
                kprintln!("  {} fault(s) caught since boot", crate::cpu::recover::caught());
                return;
            }
            if rest.split_whitespace().next() == Some("code") {
                crate::cpu::code::trigger_generated_fault();
            }
            idt::trigger_page_fault();
        }
        // What the CPU offers and what the kernel turned on.
        //
        // Printed at boot and then twenty selftest sections scroll it away,
        // which on hardware means it is gone -- and it is the line that decides
        // whether every kernel in the system is the vectorised one or the
        // scalar fallback. Worth being able to ask for.
        "simd" => {
            let f = crate::cpu::detected();
            kprintln!(
                "  sse2={} sse4.1={} avx={} avx2={} fma={} f16c={} avx512f={}",
                f.sse2, f.sse41, f.avx, f.avx2, f.fma, f.f16c, f.avx512f
            );
            console::set_color(if f.avx_enabled { LTGREEN } else { LTRED });
            kprintln!("  xsave={}  avx enabled={}", f.xsave, f.avx_enabled);
            console::set_color(LTGRAY);
            if !f.avx_enabled {
                kprintln!("  the OS has not enabled the state, so every avx kernel");
                kprintln!("  falls back to scalar -- this is a ~10x factor, not a rounding one");
            }
        }
        // The other cores: how many answered, and what they are worth.
        "smp" => {
            let n = crate::smp::online();
            kprintln!("  {} core(s) online", n);
            if rest.starts_with("bench") {
                // Runs on one core too, and reports the same number twice.
                // That is the measurement worth having on its own: a machine
                // with no helpers gives the honest serial figure, which is the
                // baseline the parallel one has to be judged against.
                crate::smp::bench();
            } else if n > 1 {
                kprintln!("  'smp bench' to time a 16 MiB matvec on one core against all of them");
            }
        }
        // What the model is told about the person it is talking to.
        //
        // A file rather than a setting, because it is prose and because the
        // operator has to be able to correct it. It is read at the start of
        // every conversation, so an edit takes effect on the next `ask new`.
        "about" => {
            let path = crate::ai::companion::ABOUT;
            if rest.is_empty() {
                match crate::sysbox::read_blob(path) {
                    Some(b) if !b.is_empty() => {
                        kprintln!("{}", alloc::string::String::from_utf8_lossy(&b))
                    }
                    _ => {
                        kprintln!("  nothing recorded. 'about <text>' to add a line.");
                        kprintln!("  it is read into the system turn of every new conversation.");
                    }
                }
            } else if rest == "clear" {
                if crate::sysbox::write_blob(path, alloc::vec::Vec::new()) {
                    kprintln!("  cleared");
                }
            } else {
                // Append. Facts accumulate; a setter that overwrote would make
                // every new fact cost the operator the old ones.
                let mut buf = crate::sysbox::read_blob(path).unwrap_or_default();
                if !buf.is_empty() && !buf.ends_with(b"\n") {
                    buf.push(b'\n');
                }
                buf.extend_from_slice(rest.as_bytes());
                buf.push(b'\n');
                let n = buf.len();
                if crate::sysbox::write_blob(path, buf) {
                    kprintln!("  {} B at {}", n, path);
                    kprintln!("  takes effect on the next new conversation ('ask new')");
                } else {
                    kprintln!("  could not write {}", path);
                }
            }
        }
        // Which skills may keep operator powers.
        //
        // Shell-only and never an applet, for the same reason `app trust` is:
        // a model that could grant itself trust would have defeated the gate
        // by using it. The hash has to be typed back, so approving one is a
        // deliberate act about specific bytes.
        "skill" => {
            let mut it = rest.splitn(2, ' ');
            match (it.next().unwrap_or(""), it.next().unwrap_or("").trim()) {
                ("trust", p) if !p.is_empty() => match crate::sysbox::skill_trust(p) {
                    Some(h) => {
                        kprintln!("  trusted {}", &h[..16]);
                        kprintln!("  it keeps operator powers until its bytes change");
                    }
                    None => kprintln!("  no single skill matches '{}' -- refusing", p),
                },
                ("untrust", _) => {
                    if crate::sysbox::skill_untrust_all() {
                        kprintln!("  every skill is sandboxed again");
                    }
                }
                // Judge a program and adopt it into the toolkit if it passes.
                //
                // `agent learn` used to be the whole of adoption: the file
                // appeared under /ai/tools and `run` would execute it, with
                // nothing having asked whether it was any good. This is the
                // same route every other change the machine makes to itself
                // takes -- judges, a node, a ledger line, and a rollback.
                ("judge", p) if !p.is_empty() => match crate::sysbox::read_blob(p) {
                    None => kprintln!("  no such file: {}", p),
                    Some(bytes) => {
                        let src = alloc::string::String::from_utf8_lossy(&bytes).into_owned();
                        let h = crate::ai::skill::store(&src);
                        let prop = crate::ai::godel::Proposal::skill(h);
                        let b = prop.budget(0, 0);
                        match crate::ai::with_engine(|e| crate::ai::godel::run(e, &b, &prop)) {
                            None => kprintln!("  {}", crate::ai::engine_refusal()),
                            Some(Err(why)) => kprintln!("  refused: {}", why.why()),
                            Some(Ok(c)) => {
                                let v = crate::ai::skill::bench(&h);
                                console::set_color(if c.adopted { LTGREEN } else { YELLOW });
                                kprintln!(
                                    "  {} {}",
                                    if c.adopted { "adopted" } else { "rejected" },
                                    &crate::ai::voter::hex(&h)[..8]
                                );
                                console::set_color(LTGRAY);
                                kprintln!("  J1 {} | J2 {} | J3 {} | J4 {} step(s)", v.j1_why, v.j2_why, v.j3_why, v.steps);
                                if c.adopted {
                                    kprintln!("  at {}", crate::ai::skill::adopted_path(&h));
                                    kprintln!("  'godel rollback' takes it back out");
                                }
                            }
                        }
                    }
                },
                _ => {
                    let all = crate::sysbox::skill_list();
                    if all.is_empty() {
                        kprintln!("  no skills in /ai/tools");
                    }
                    for (name, h, trusted) in all {
                        console::set_color(if trusted { YELLOW } else { LTGRAY });
                        kprintln!(
                            "  {}  {}  {}",
                            &h[..8],
                            if trusted { "operator " } else { "sandboxed" },
                            name
                        );
                    }
                    console::set_color(LTGRAY);
                    kprintln!("  'skill trust <hash>' to grant operator powers, 'skill untrust' to revoke all");
                }
            }
        }
        // Run a program and find out what it would do, without keeping it.
        //
        // The step the self-improvement loop needs before adopting anything a
        // model wrote: run it, see precisely which objects moved, decide. The
        // program runs sandboxed like any other stored program, and the
        // namespace is put back unless `keep` is asked for.
        "sandbox" => {
            let mut w = Words::new(rest);
            let (verb, path) = (w.next().unwrap_or(""), w.next().unwrap_or(""));
            let (keep, path) = match verb {
                "keep" => (true, path),
                _ => (false, if verb.is_empty() { "" } else { verb }),
            };
            if path.is_empty() {
                kprintln!("  usage: sandbox [keep] <path>");
                kprintln!("  runs it, reports every object it touched, then puts the tree back");
                return;
            }
            let owned = alloc::string::String::from(path);
            match crate::sysbox::shadow(|| { crate::sysbox::dispatch("run", &owned); }) {
                None => kprintln!("  no namespace"),
                Some(sh) => {
                    if sh.changes == 0 {
                        kprintln!("  touched nothing");
                    } else {
                        console::set_color(YELLOW);
                        kprintln!("  {} object(s) touched:", sh.changes);
                        console::set_color(LTGRAY);
                        for line in sh.touched.iter().take(24) {
                            kprintln!("    {}", line);
                        }
                        if sh.touched.len() > 24 {
                            kprintln!("    ... and {} more", sh.touched.len() - 24);
                        }
                    }
                    if keep {
                        sh.keep();
                        kprintln!("  kept");
                    } else {
                        sh.discard();
                        kprintln!("  discarded -- 'sandbox keep <path>' to let it stand");
                    }
                }
            }
        }
        "version" | "uname" => {
            console::set_color(LTCYAN);
            kprintln!("  glados {}", crate::VERSION);
            console::set_color(LTGRAY);
            kprintln!("  a ring-0 kernel for MSI MS-16R8, one address space, no syscalls");
            // The formats an update has to stay compatible with. Each is
            // refused rather than guessed at when its version does not match,
            // so this is the map of what a new image may not silently change.
            kprintln!("  formats: checkpoint v2/v3/v4, app 1, GLADOSPK 1, GLADOSTR, GLADOSA1, GLADOSC1");
        }
        // Run the self-tests on demand and remember what they said.
        //
        // They all ran at boot and scrolled away. The question afterwards is
        // whether the machine is still correct, which is a different question
        // and needs a different surface.
        "diag" => {
            use crate::diag;
            let want = rest.trim();
            let run = |i: usize| {
                let s = &diag::SUITES[i];
                console::set_color(YELLOW);
                kprintln!("[{}] {}", s.name, s.about);
                console::set_color(LTGRAY);
                match diag::run_one(i) {
                    Some(true) => {
                        console::set_color(LTGREEN);
                        kprintln!("  {} passed", s.name);
                    }
                    _ => {
                        console::set_color(LTRED);
                        kprintln!("  {} FAILED", s.name);
                    }
                }
                console::set_color(LTGRAY);
            };
            if want.is_empty() {
                let (p, f, u) = diag::tally();
                for (i, s) in diag::SUITES.iter().enumerate() {
                    let (mark, col) = match diag::verdict(i) {
                        diag::Verdict::Pass => ("pass", LTGREEN),
                        diag::Verdict::Fail => ("FAIL", LTRED),
                        diag::Verdict::Unknown => ("  - ", LTGRAY),
                    };
                    console::set_color(col);
                    kprintln!("  {}  {:<8} {}", mark, s.name, s.about);
                }
                console::set_color(LTGRAY);
                kprintln!("  {} passed, {} failed, {} not run", p, f, u);
                kprintln!("  'diag all' runs everything, 'diag <name>' runs one");
            } else if want == "all" {
                for i in 0..diag::SUITES.len() {
                    run(i);
                }
                let (p, f, _) = diag::tally();
                console::set_color(if f == 0 { LTGREEN } else { LTRED });
                kprintln!("  {} passed, {} failed", p, f);
                console::set_color(LTGRAY);
            } else {
                match diag::find(want) {
                    Some(i) => run(i),
                    None => kprintln!("  no suite called '{}' -- 'diag' lists them", want),
                }
            }
        }
        // Is the staged image one this machine will run?
        //
        // Verification only. Applying it is a pre-ExitBootServices step that
        // does not exist yet, and a command that verified and then did nothing
        // while sounding like it installed something would be worse than no
        // command.
        "update" => update_cmd(rest),
        "words" => {
            console::set_color(YELLOW);
            kprintln!("builtins");
            console::set_color(WHITE);
            // Grouped by what each one touches, because that is the fact an
            // operator needs before writing anything a stored program will
            // run: the three classes on top are the ones the sandbox allows,
            // and the rest need `app trust`.
            use crate::aiksi::eval::Touch;
            for (label, want) in [
                ("values", Touch::Pure),
                ("reads the machine", Touch::Read),
                ("writes", Touch::Write),
                ("network", Touch::Net),
                ("the model", Touch::Model),
                ("draws", Touch::Draw),
                ("the machine itself", Touch::Raw),
            ] {
                let names: alloc::vec::Vec<&str> = aiksi::eval::BUILTINS
                    .iter()
                    .filter(|(_, t, ..)| *t == want)
                    .map(|(n, ..)| *n)
                    .collect();
                if names.is_empty() {
                    continue;
                }
                console::set_color(YELLOW);
                kprintln!("  {}", label);
                console::set_color(WHITE);
                let mut col = 0;
                for w in names {
                    kprint!("  {:<11}", w);
                    col += 1;
                    if col % 5 == 0 {
                        kprintln!();
                    }
                }
                if col % 5 != 0 {
                    kprintln!();
                }
            }
        }
        "vars" => {
            if interp.var_count() == 0 {
                kprintln!("  none yet");
            } else {
                for (k, v) in interp.vars() {
                    kprintln!("  {} = {}", k, v.render());
                }
            }
        }
        _ => {
            // Not a command, so it is code. This is the point of the whole
            // exercise: there is no boundary between using the machine and
            // programming it.
            match aiksi::eval_line(interp, line) {
                Ok(aiksi::Value::Nil) => {}
                Ok(v) => {
                    console::set_color(LTCYAN);
                    kprintln!("  {}", v.render());
                    console::set_color(WHITE);
                }
                Err(e) => {
                    console::set_color(LTRED);
                    kprintln!("  {}", e);
                    console::set_color(WHITE);
                }
            }
        }
    }
}

// --- composition --------------------------------------------------------
//
// `cmd > /path` sends what a command printed into the namespace, and
// `cmd | filter` runs it through a text filter. Both work by capturing the
// console rather than by changing any applet, which is what makes them
// available to every command at once -- including the ones that existed long
// before this did.
//
// Filters take text and give text. They are not applets: an applet acts on the
// namespace, and these only ever look at a string that has already been
// produced.

fn strip_ansi_free(s: &str) -> alloc::string::String {
    // Console colour is set through a side channel rather than escape codes,
    // so captured text is already plain. This exists so that stays true by
    // intent rather than by accident.
    alloc::string::String::from(s)
}

fn apply_filter(text: &str, spec: &str) -> alloc::string::String {
    let mut it = spec.trim().splitn(2, ' ');
    let verb = it.next().unwrap_or("");
    let arg = it.next().unwrap_or("").trim();
    let lines: Vec<&str> = text.lines().collect();

    let mut out = alloc::string::String::new();
    match verb {
        "grep" => {
            for l in lines.iter().filter(|l| l.contains(arg)) {
                out.push_str(l);
                out.push('\n');
            }
        }
        "head" => {
            let n: usize = arg.parse().unwrap_or(10);
            for l in lines.iter().take(n) {
                out.push_str(l);
                out.push('\n');
            }
        }
        "tail" => {
            let n: usize = arg.parse().unwrap_or(10);
            let skip = lines.len().saturating_sub(n);
            for l in lines.iter().skip(skip) {
                out.push_str(l);
                out.push('\n');
            }
        }
        "sort" => {
            let mut v = lines.clone();
            v.sort_unstable();
            for l in v {
                out.push_str(l);
                out.push('\n');
            }
        }
        "count" | "wc" => {
            let words = text.split_whitespace().count();
            out = alloc::format!(
                "  {} lines, {} words, {} bytes\n",
                lines.len(),
                words,
                text.len()
            );
        }
        other => {
            out = alloc::format!("  not a filter: {}\n", other);
        }
    }
    out
}

/// Split a command line on the *last* unquoted `|` or `>`.
///
/// Last rather than first, so `grep >` reads as a filter argument rather than
/// as a redirection -- the trailing operator is the one that applies to
/// everything before it.
/// A command line taken one word at a time, with the remainder still available.
///
/// Exists because the obvious thing went wrong three times in a row. An arm
/// would do `rest.splitn(3, ' ')`, a sub-arm would take one of those pieces and
/// split it again, and the second split looked at a piece whose shape the first
/// had already fixed -- so `app trust <name> <hash>` found no hash and
/// `app draft <name> <kind>` found no kind. Both failed by doing nothing and
/// saying nothing, which is the worst way to fail an operator.
///
/// The fix is not to count more carefully, it is to stop counting. A sub-arm
/// asks for the next word, asks for the rest when it wants free text, and never
/// needs to know how many pieces anything above it made. There is no argument
/// count anywhere for a later arm to disagree with.
///
/// `next` keeps the `Option` shape so that arms which already read well as a
/// `match` do not have to be rewritten to gain the property.
pub struct Words<'a> {
    s: &'a str,
}

impl<'a> Words<'a> {
    pub fn new(s: &'a str) -> Words<'a> {
        Words { s: s.trim_start() }
    }

    /// The next word, or `None` when the line is spent.
    pub fn next(&mut self) -> Option<&'a str> {
        if self.s.is_empty() {
            return None;
        }
        Some(self.word())
    }

    /// The next word, or `""` when the line is spent.
    pub fn word(&mut self) -> &'a str {
        let s = self.s;
        match s.find(' ') {
            None => {
                self.s = "";
                s
            }
            Some(i) => {
                self.s = s[i + 1..].trim_start();
                &s[..i]
            }
        }
    }

    /// Everything not yet taken, verbatim. Free text keeps its spaces.
    pub fn rest(&self) -> &'a str {
        self.s
    }

    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }
}

pub fn words_selftest() -> bool {
    let mut w = Words::new("  trust todo abc123 ");
    if w.word() != "trust" || w.word() != "todo" || w.word() != "abc123" {
        return false;
    }
    if !w.is_empty() {
        return false;
    }
    // Running off the end answers empty for ever rather than repeating the last
    // word, which is what would let an arm act on a stale argument.
    let mut e = Words::new("");
    if e.next().is_some() || !e.word().is_empty() {
        return false;
    }
    // Free text keeps its spaces, so `app todo add buy milk` still adds two
    // words and not one.
    let mut g = Words::new("todo add buy milk");
    if g.word() != "todo" || g.word() != "add" || g.rest() != "buy milk" {
        return false;
    }
    // Runs of spaces do not produce empty words.
    let mut m = Words::new("a    b");
    if m.word() != "a" || m.word() != "b" || !m.is_empty() {
        return false;
    }
    // And the two shapes agree about where they are.
    let mut n = Words::new("one two");
    if !(n.next() == Some("one") && n.rest() == "two") {
        return false;
    }

    // Redirection, against the expressions that used to be eaten by it.
    // A comparison is not a redirect, at any depth or spacing.
    for src in [
        "if (len(x) > 0) { println(1) }",
        "uptime() > 5",
        "a >= b",
        "x <> y",
        "count() > 0",
        // Inside a string, where the old version happily split.
        "println(\"a > b\")",
        "split(t, \"|\")",
        // Aiksi's or, not a pipe.
        "a || b",
    ] {
        if split_pipeline(src).is_some() {
            return false;
        }
    }
    // ...and the real thing still works, including with an expression in front
    // of it that contains both characters.
    match split_pipeline("mem > /sys/boot.log") {
        Some((h, '>', t)) if h == "mem" && t == "/sys/boot.log" => {}
        _ => return false,
    }
    match split_pipeline("log | grep boot") {
        Some((h, '|', t)) if h == "log" && t == "grep boot" => {}
        _ => return false,
    }
    match split_pipeline("if (a > b) { println(2) } > /tmp/out") {
        Some((_, '>', t)) if t == "/tmp/out" => {}
        _ => return false,
    }
    true
}

/// Find the redirection or pipe in a command line, if there is one.
///
/// This used to take the last `|` or `>` anywhere in the string, which was
/// fine while the prompt was mostly commands and became untenable when Aiksi
/// grew into the language everything is written in. `>` is comparison and `||`
/// is or, so `if (len(x) > 0)` typed at the prompt was silently redirected --
/// the head ran, and 26 bytes went into a file named `0) { ... }`. Nothing
/// failed and nothing said so.
///
/// Three rules, and each one exists because the naive version broke on it:
///
/// * **Not inside a string or brackets.** `split(t, "|")` is an argument, and
///   `get(f, i) > 0` inside a call is a comparison.
/// * **Not part of a longer operator.** `>=`, `>>`, `->` and `||` are all
///   things Aiksi or the shell spells with these characters.
/// * **A redirect target is an absolute path.** This is what actually
///   separates `mem > /sys/boot.log` from `uptime() > 5`, because both are a
///   bare `>` at depth zero with a one-word tail. Every redirect this system
///   has ever documented writes into the namespace, so requiring the `/` costs
///   nothing and makes the comparison unambiguous.
///
/// A pipe needs no such rule: Aiksi has no single `|` operator, so a bare one
/// outside a string can only be a pipe.
fn split_pipeline(line: &str) -> Option<(&str, char, &str)> {
    let b = line.as_bytes();
    let mut found: Option<(usize, char)> = None;
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'|' if depth == 0 => {
                if b.get(i + 1) == Some(&b'|') {
                    i += 2;
                    continue;
                }
                found = Some((i, '|'));
            }
            b'>' if depth == 0 => {
                let prev = if i > 0 { b[i - 1] } else { 0 };
                let next = b.get(i + 1).copied().unwrap_or(0);
                let part_of_operator =
                    next == b'=' || next == b'>' || prev == b'-' || prev == b'>' || prev == b'<';
                if !part_of_operator && line[i + 1..].trim().starts_with('/') {
                    found = Some((i, '>'));
                }
            }
            _ => {}
        }
        i += 1;
    }
    let (i, c) = found?;
    Some((line[..i].trim(), c, line[i + 1..].trim()))
}

/// Run `cmd`, capturing everything it prints.
fn capture(
    cmd: &str,
    boot: &BootInfo,
    acpi: &Option<Acpi>,
    interp: &mut aiksi::Interp,
) -> alloc::string::String {
    console::begin_capture();
    execute(cmd, boot, acpi, interp);
    console::end_capture().unwrap_or_default()
}

/// Handle a line containing `|` or `>`. Returns false if there was neither.
pub fn run_pipeline(
    line: &str,
    boot: &BootInfo,
    acpi: &Option<Acpi>,
    interp: &mut aiksi::Interp,
) -> bool {
    let Some((head, op, tail)) = split_pipeline(line) else {
        return false;
    };
    if head.is_empty() || tail.is_empty() {
        return false;
    }

    let text = strip_ansi_free(&capture(head, boot, acpi, interp));

    match op {
        '|' => {
            let out = apply_filter(&text, tail);
            kprint!("{}", out);
        }
        '>' => {
            if crate::sysbox::write_text(tail, &text) {
                kprintln!("  {} bytes -> {}", text.len(), tail);
            } else {
                kprintln!("  could not write {}", tail);
            }
        }
        _ => return false,
    }
    true
}

/// Browse a real filesystem on the disk.
///
/// The firmware could read files and no longer exists; this is the kernel
/// doing it for itself. Read-only on purpose -- a tool that can inspect a
/// broken disk is useful, one that can half-write it is worse than nothing.
/// Turn a command-line host into an address, reporting why if it cannot.
///
/// Every network command routes through this so that a name and a dotted quad
/// are interchangeable everywhere -- there is no separate "resolve first" step
/// to forget.
fn host_to_ip(host: &str) -> Option<crate::net::Ipv4> {
    match crate::net::dns::lookup(host) {
        Ok(ip) => Some(ip),
        Err(e) => {
            kprintln!("  {}: {}", host, e.name());
            None
        }
    }
}

fn fat_cmd(rest: &str) {
    use crate::store::{block, fat};

    let mut it = rest.split_whitespace();
    let verb = it.next().unwrap_or("");
    let arg = it.next().unwrap_or("");
    let arg2 = rest.splitn(3, ' ').nth(2).unwrap_or("").trim();

    // Which partition. Named by index from `disk`, or "auto" for the first one
    // that actually parses as FAT -- which is the useful default, since the
    // partition type byte is a claim and mounting is a check.
    let layout = match block::scan() {
        Ok(l) => l,
        Err(e) => {
            kprintln!("  cannot read the partition table: {:?}", e);
            return;
        }
    };

    let pick_auto = || -> Option<(u32, fat::Volume)> {
        for p in layout.partitions.iter() {
            if let Ok(v) = fat::Volume::mount(p.start_lba) {
                return Some((p.index, v));
            }
        }
        None
    };

    match verb {
        "" | "list" if arg.is_empty() => {
            console::set_color(YELLOW);
            kprintln!("[fat]");
            console::set_color(LTGRAY);
            let mut any = false;
            for p in layout.partitions.iter() {
                if let Ok(v) = fat::Volume::mount(p.start_lba) {
                    any = true;
                    kprintln!(
                        "  partition {}  {:?}  {} clusters of {} B  ({})",
                        p.index,
                        v.kind(),
                        v.total_clusters(),
                        v.cluster_bytes(),
                        p.kind()
                    );
                }
            }
            if !any {
                kprintln!("  no FAT filesystem found on this disk");
            } else {
                kprintln!("  'fat ls <path>', 'fat cat <path>', 'fat get <path> <namespace-path>'");
            }
        }
        // Claim this partition's range so it can be written.
        //
        // Separate from `store unlock`, which claims the object store's
        // region: these are different ranges and conflating them would let a
        // claim for one authorise writes to the other, which is the whole
        // failure the ranged gate exists to prevent.
        "unlock" => {
            let Some(p) = layout.partitions.iter().find(|p| {
                arg.is_empty() || arg.parse::<u32>().ok() == Some(p.index)
            }) else {
                kprintln!("  no such partition");
                return;
            };
            if crate::dev::nvme::unlock_writes(0xD15EA5E, p.start_lba, p.block_count) {
                console::set_color(LTRED);
                kprintln!(
                    "  partition {} is writable: lba {}..{}",
                    p.index,
                    p.start_lba,
                    p.start_lba + p.block_count
                );
                console::set_color(WHITE);
                kprintln!("  nothing outside that range can be written, including the table");
                console::set_color(LTGRAY);
            } else {
                kprintln!("  refused");
            }
        }
        // Writing. Refused unless the operator has claimed a range covering
        // this partition, because that gate is the only thing standing between
        // a file write and the Windows volume next to it.
        "put" | "rm" => {
            let Some((idx, vol)) = pick_auto() else {
                kprintln!("  no FAT filesystem found");
                return;
            };
            if !crate::dev::nvme::writes_unlocked() {
                console::set_color(YELLOW);
                kprintln!("  writes are locked. 'store unlock' claims a range first.");
                console::set_color(LTGRAY);
                return;
            }
            if verb == "rm" {
                match crate::store::fatw::remove(&vol, arg) {
                    Ok(()) => kprintln!("  removed {} from partition {}", arg, idx),
                    Err(e) => kprintln!("  {}", e),
                }
                return;
            }
            // `fat put <path> <text>`. Text rather than bytes, because the
            // thing somebody wants to carry off this machine on a stick is
            // almost always something they can read at the other end.
            if arg2.is_empty() {
                kprintln!("  fat put <path> <text>");
                return;
            }
            match crate::store::fatw::put(&vol, arg, arg2.as_bytes()) {
                Ok(()) => kprintln!("  wrote {} byte(s) to {} on partition {}", arg2.len(), arg, idx),
                Err(e) => kprintln!("  {}", e),
            }
        }
        "ls" | "cat" | "get" => {
            let Some((idx, vol)) = pick_auto() else {
                kprintln!("  no FAT filesystem found");
                return;
            };
            let path = if verb == "get" { arg } else { rest[verb.len()..].trim() };
            match vol.find(path) {
                Err(e) => kprintln!("  {}: {:?}", path, e),
                Ok(entry) => match verb {
                    "ls" => {
                        if !entry.is_dir {
                            kprintln!("  {:>10}  {}", entry.size, entry.name);
                            return;
                        }
                        match vol.list(entry.cluster) {
                            Err(e) => kprintln!("  {:?}", e),
                            Ok(items) => {
                                kprintln!("  partition {}, {} entries", idx, items.len());
                                for e in items {
                                    if e.is_dir {
                                        console::set_color(LTCYAN);
                                        kprintln!("  {:>10}  {}/", "", e.name);
                                    } else {
                                        console::set_color(WHITE);
                                        kprintln!("  {:>10}  {}", e.size, e.name);
                                    }
                                }
                                console::set_color(LTGRAY);
                            }
                        }
                    }
                    "cat" => match vol.read_file(&entry) {
                        Err(e) => kprintln!("  {:?}", e),
                        Ok(bytes) => {
                            for line in bytes.split(|c| *c == b'\n') {
                                kprintln!(
                                    "  {}",
                                    core::str::from_utf8(line).unwrap_or("<binary>")
                                );
                            }
                        }
                    },
                    _ => match vol.read_file(&entry) {
                        Err(e) => kprintln!("  {:?}", e),
                        Ok(bytes) => {
                            let dest = if arg2.is_empty() { "/tmp/imported" } else { arg2 };
                            let n = bytes.len();
                            if crate::sysbox::write_blob(dest, bytes) {
                                kprintln!("  {} bytes -> {}", n, dest);
                            } else {
                                kprintln!("  could not write {}", dest);
                            }
                        }
                    },
                },
            }
        }
        other => kprintln!("  not a fat subcommand: {}", other),
    }
}
