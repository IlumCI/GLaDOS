//! `office`: the in-kernel model doing office work through LibreOffice.
//!
//! Two halves, and they meet in the document on the screen.
//!
//! **LibreOffice is a guest**, unmodified, from the root image: Writer on the
//! desktop through Skywalker, and a second, headless copy for the work no
//! window is needed for -- reading a document as text, converting one format
//! to another. The headless copy runs on a profile of its own, because
//! LibreOffice is single-instance per profile: a second launch on the GUI's
//! profile hands its arguments to the running window over a pipe and exits,
//! and the conversion it was asked for happens in somebody else's process or
//! not at all.
//!
//! **The model is the kernel's own**, and what it writes reaches the document
//! the way a person's writing would: as keystrokes (`dev::typist`). The draft
//! is typed as it is generated, so a sentence appears while the next is still
//! being decided. Nothing here knows LibreOffice's internals and nothing has
//! to, which is why the same verb types into any program with a text field.
//!
//! Every verb is the operator's: `office` is a shell verb and not an applet,
//! so the model cannot type into a window on its own initiative. Making that
//! an agent action is a separate decision, gated like every other mutating
//! applet, and it is not taken here.

mod ask;
mod table;
pub use ask::colleague;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{kprint, kprintln};

/// LibreOffice's real binary. `soffice` is a shell script that re-execs it
/// through `oosplash`, which is one more process for no gain here.
pub const SOFFICE: &str = "/usr/lib/libreoffice/program/soffice.bin";
/// The headless copy's own profile. See the module doc.
const HEADLESS: &str = "-env:UserInstallation=file:///tmp/office/profile";
/// Where conversions land. `/tmp` is the one tree a guest may write and the
/// kernel reads directly, so nothing has to be copied out afterwards.
pub const OUT: &str = "/tmp/office";
/// What the Writer window's title always contains.
const WRITER: &str = "LibreOffice Writer";

/// How long a headless LibreOffice may take. A first run builds its profile,
/// which is most of a minute under emulation.
const CONVERT_S: u64 = 240;
/// How long Writer may take to put up a window.
const WINDOW_S: u64 = 240;

/// The last things `office` did, for the Office window. Each is also
/// printed, so the terminal and the window tell one story.
static LOG: crate::sync::Spin<Vec<String>> = crate::sync::Spin::new(Vec::new());
const LOG_CAP: usize = 32;

fn note(line: String) {
    kprintln!("  {}", line);
    let mut l = LOG.lock_irq();
    if l.len() >= LOG_CAP {
        l.remove(0);
    }
    l.push(line);
}

pub fn log_snapshot() -> Vec<String> {
    LOG.lock_irq().clone()
}

pub fn command(rest: &str) {
    let rest = rest.trim();
    let (verb, arg) = match rest.split_once(char::is_whitespace) {
        Some((v, a)) => (v, a.trim()),
        None => (rest, ""),
    };
    let out = match verb {
        "" | "status" => {
            status();
            Ok(())
        }
        "start" | "writer" => {
            let r = match arg {
                "calc" => start_app(App::Calc, None),
                "" => start_app(App::Writer, None),
                f if f.ends_with(".ods") || f.ends_with(".xlsx") || f.ends_with(".csv") => start_app(App::Calc, Some(f)),
                f => start_app(App::Writer, Some(f)),
            };
            r.map(|t| note(format!("up: {}", t)))
        }
        "write" | "draft" if !arg.is_empty() => write(arg),
        "type" if !arg.is_empty() => type_into_writer(arg),
        "read" if !arg.is_empty() => extract(arg).map(|t| {
            kprintln!("{}", t.trim_end());
            kprintln!("  ({} characters)", t.chars().count());
        }),
        "summarize" | "summarise" if !arg.is_empty() => {
            let (path, into) = match arg.strip_suffix("--type") {
                Some(p) => (p.trim(), true),
                None => (arg, false),
            };
            summarize(path, into)
        }
        "convert" if !arg.is_empty() => match arg.rsplit_once(char::is_whitespace) {
            Some((path, to)) => convert(path.trim(), to.trim()).map(|p| note(format!("written: {}", p))),
            None => Err("usage: office convert <path> <format>   (pdf, docx, odt, txt, html, xlsx, ...)".into()),
        },
        "ui" | "window" => {
            crate::gfx::officewin::open();
            Ok(())
        }
        "calc" | "sheet" if !arg.is_empty() => calc(arg),
        "bench" => bench(arg),
        "wikisql" => table::wikisql(arg),
        // The workbook `colleague` works on: opened in Calc, and remembered.
        "open" if !arg.is_empty() => ask::open(arg),
        "ask" if !arg.is_empty() => match arg.split_once(char::is_whitespace) {
            Some((path, req)) => ask::ask(path, req.trim()),
            None => Err("usage: office ask <workbook> <request>".into()),
        },
        // Into whatever has focus, with \t and \n spelt out: how a table is
        // typed without the model, and how the typist is tested.
        "keys" if !arg.is_empty() => {
            let text = arg.replace("\\t", "\t").replace("\\n", "\n");
            let n = crate::dev::typist::push(&text);
            wait_typed();
            note(format!("typed {} characters into the focused window", n));
            Ok(())
        }
        "focus" if !arg.is_empty() => {
            if crate::gfx::desk::focus_titled(arg) {
                Ok(())
            } else {
                Err(format!("no window's title contains '{}'", arg))
            }
        }
        "stop" => {
            crate::dev::typist::stop();
            kprintln!("  typing stopped");
            Ok(())
        }
        _ => Err(String::from(USAGE)),
    };
    if let Err(why) = out {
        note(why);
    }
}

const USAGE: &str = "usage: office [status]
         office start [calc|path]       Writer (or Calc) on the desktop, with a document if given
         office write <what to write>   the model drafts it, typed into Writer as it goes
         office type <text>             type text into Writer
         office read <path>             a document as plain text, through LibreOffice
         office summarize <path> [--type]   the model summarises it; --type puts it in Writer
         office convert <path> <format> pdf, docx, odt, txt, html, xlsx, ...
         office calc <what to build>    the model builds a spreadsheet, opened in Calc, totals as =SUM formulas
         office bench [dir] [from] [count] [--oracle]   SpreadsheetBench cell-level tasks, scored
         office ui                      the Office window
         office stop                    stop typing";

fn status() {
    let root = crate::linux::disk::root_mounted();
    kprintln!("  root filesystem   {}", if root { "mounted" } else { "none -- no disk carries one" });
    let lo = root && exists(SOFFICE);
    kprintln!("  LibreOffice       {}", if lo { "installed" } else { "not found" });
    match crate::gfx::desk::window_titled(WRITER) {
        Some(t) => kprintln!("  Writer            open: {}", t),
        None => kprintln!("  Writer            not open -- 'office start'"),
    }
    let model = crate::ai::with_engine(|e| (e.model.cfg.seq_len, e.model.cfg.dim)).map(|(s, d)| (s, d));
    match model {
        Some((seq, dim)) => kprintln!("  model             loaded, {} positions of context, width {}", seq, dim),
        None => kprintln!("  model             {}", crate::ai::engine_refusal()),
    }
    let (typed, dropped) = crate::dev::typist::counts();
    kprintln!(
        "  typist            {} typed, {} could not be, {} waiting",
        typed,
        dropped,
        crate::dev::typist::pending()
    );
}

/// The root filesystem and its session, found if nobody has mounted them.
fn ready() -> Result<(), String> {
    if crate::linux::disk::root_mounted() {
        return Ok(());
    }
    match crate::linux::disk::auto_root() {
        Some(_) => {
            crate::linux::services::begin_session();
            Ok(())
        }
        None => Err("no disk carries a root filesystem, so there is no LibreOffice to run".into()),
    }
}

/// Start LibreOffice as a job and wait for it to end. Answers how it ended.
///
/// **On LibreOffice's own headless toolkit.** The session asks every program
/// for GTK (`SAL_USE_VCLPLUGIN=gtk3`, so Writer draws through Wayland), and
/// LibreOffice honours that even under `--headless`: each conversion brought
/// GTK up, and GTK started helpers through `sh` until the guest ran out of
/// processes -- two minutes a launch and a screen of `fork: retry`. `svp` is
/// the plugin `--headless` would have chosen, given to this job alone
/// (`start_job_env`): flipping the session's value around the launch raced
/// the loader, and some launches still came up on GTK.
fn run_to_end(args: &[&str], secs: u64) -> Result<String, String> {
    // Quiet: what a headless run prints is GTK and the soffice script talking
    // to nobody, and it went into the terminal the operator was reading.
    let id = crate::linux::start_job_env_quiet(SOFFICE, args, &["SAL_USE_VCLPLUGIN=svp"]).map_err(String::from)?;
    let deadline = crate::dev::lapic::ticks() + secs * crate::TIMER_HZ as u64;
    loop {
        if let Some(r) = crate::job::list().into_iter().find(|r| r.id == id) {
            if r.state == "done" {
                return Ok(r.summary);
            }
        }
        if crate::dev::lapic::ticks() > deadline {
            let _ = crate::job::kill(id);
            return Err(format!("LibreOffice did not finish in {} s, so it was stopped", secs));
        }
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 5);
    }
}

/// Run a headless conversion until the file it promises exists.
///
/// **Up to three times, and not out of hope.** A LibreOffice that has just
/// made its profile exits with status 81 and expects the launcher script to
/// start it again -- the `soffice` wrapper does exactly that and the binary
/// run directly does not. So the first conversion on a fresh profile always
/// produces nothing, and the second produces the file. A third is the margin
/// for a profile half-written by a run that was stopped.
fn headless(args: &[&str], want: &str) -> Result<(), String> {
    if !crate::linux::ns::is_dir(OUT) {
        crate::linux::ns::make_dir(OUT);
    }
    let _ = crate::linux::ns::detach(want);
    let mut last = String::new();
    for _ in 0..3 {
        let mut full: Vec<&str> = alloc::vec!["--headless", "--norestore", "--nologo", HEADLESS];
        full.extend_from_slice(args);
        last = run_to_end(&full, CONVERT_S)?;
        if crate::linux::ns::blob_len(want).is_some() {
            return Ok(());
        }
    }
    Err(format!("LibreOffice produced no {} ({})", want, last.trim()))
}

/// The file name a conversion of `path` to `ext` is written as.
fn out_name(path: &str, ext: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.rsplit_once('.').map(|(s, _)| s).unwrap_or(file);
    format!("{}/{}.{}", OUT, stem, ext)
}

/// A file as a guest sees it: the writable layer first, then the mounted
/// disks -- the order `openat` resolves in.
fn exists(path: &str) -> bool {
    crate::linux::ns::blob_len(path).is_some()
        || matches!(crate::linux::disk::lookup(path), Some(Some(n)) if !n.is_dir)
}


/// A document as plain text, read by LibreOffice itself -- so anything it can
/// open (Word, OpenDocument, RTF, HTML, a PDF it can import) can be read.
pub fn extract(path: &str) -> Result<String, String> {
    ready()?;
    if !exists(path) {
        return Err(format!("no such file: {}", path));
    }
    // A plain text file needs no LibreOffice at all.
    if path.ends_with(".txt") || path.ends_with(".md") {
        return read_any(path);
    }
    let want = out_name(path, "txt");
    headless(&["--convert-to", "txt:Text (encoded):UTF8", "--outdir", OUT, path], &want)?;
    read_any(&want)
}

/// A file's bytes from the namespace or from a mounted disk.
fn read_bytes(path: &str) -> Option<Vec<u8>> {
    crate::linux::ns::read_blob(path).or_else(|| crate::linux::disk::program_bytes(path))
}

fn read_any(path: &str) -> Result<String, String> {
    let bytes = crate::linux::ns::read_blob(path)
        .or_else(|| crate::linux::disk::program_bytes(path))
        .ok_or_else(|| format!("could not read {}", path))?;
    let s = String::from_utf8_lossy(&bytes);
    // LibreOffice's UTF-8 text export starts with a byte-order mark.
    Ok(s.trim_start_matches('\u{FEFF}').replace("\r\n", "\n"))
}

pub fn convert(path: &str, to: &str) -> Result<String, String> {
    ready()?;
    if !exists(path) {
        return Err(format!("no such file: {}", path));
    }
    let ext = to.trim_start_matches('.').to_ascii_lowercase();
    if ext.is_empty() || !ext.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(format!("'{}' is not a format", to));
    }
    let want = out_name(path, &ext);
    headless(&["--convert-to", &ext, "--outdir", OUT, path], &want)?;
    Ok(want)
}

/// The LibreOffice applications `office` drives, and how to tell their
/// windows apart: LibreOffice titles every window "<document> — LibreOffice
/// <application>".
#[derive(Clone, Copy, PartialEq)]
pub enum App {
    Writer,
    Calc,
}

impl App {
    fn name(self) -> &'static str {
        match self {
            App::Writer => "Writer",
            App::Calc => "Calc",
        }
    }
    fn needle(self) -> &'static str {
        match self {
            App::Writer => "LibreOffice Writer",
            App::Calc => "LibreOffice Calc",
        }
    }
    fn flag(self) -> &'static str {
        match self {
            App::Writer => "--writer",
            App::Calc => "--calc",
        }
    }
}

/// Writer on the desktop, started if it is not open. Answers its title.
pub fn start_writer(file: Option<&str>) -> Result<String, String> {
    start_app(App::Writer, file)
}

/// An application's window, started if none is open, raised and focused.
pub fn start_app(app: App, file: Option<&str>) -> Result<String, String> {
    if file.is_none() {
        if let Some(t) = crate::gfx::desk::window_titled(app.needle()) {
            crate::gfx::desk::focus_titled(app.needle());
            return Ok(t);
        }
    }
    ready()?;
    if let Some(f) = file {
        if !exists(f) {
            return Err(format!("no such file: {}", f));
        }
    }
    seed_profile();
    kprint!("  starting {}", app.name());
    let mut args = alloc::vec![app.flag(), "--norestore", "--nologo"];
    if let Some(f) = file {
        args.push(f);
    }
    // A LibreOffice already running is asked through its pipe: the new
    // process hands the request over and exits at once, and the window comes
    // from the old one. That exit is the request delivered, not a failure.
    let handoff = crate::gfx::desk::window_titled("LibreOffice").is_some();
    // Quiet: a desktop application's stderr is GTK's cursor-theme warnings
    // and the soffice script's retries, which buried the terminal it shares.
    let mut id = crate::linux::start_service_job(SOFFICE, &args).map_err(String::from)?;
    let deadline = crate::dev::lapic::ticks() + WINDOW_S * crate::TIMER_HZ as u64;
    let mut dots = 0;
    // A LibreOffice that has just finished setting its profile up exits with
    // status 81 and expects to be started again (see `headless`), so a launch
    // that ends without a window is retried -- twice, which is the margin.
    let mut retries = 2;
    let mut gone = false;
    loop {
        if let Some(t) = crate::gfx::desk::window_titled(app.needle()) {
            kprintln!();
            // A moment for the document to take the caret before anybody
            // types: the window is mapped a frame or two before the
            // application has put the cursor in the page.
            crate::task::sleep_until(crate::dev::lapic::ticks() + 2 * crate::TIMER_HZ as u64);
            crate::gfx::desk::focus_titled(app.needle());
            return Ok(t);
        }
        if !gone && crate::job::list().iter().any(|r| r.id == id && r.state == "done") {
            if handoff {
                gone = true;
            } else if retries > 0 {
                retries -= 1;
                kprint!(" again");
                id = crate::linux::start_service_job(SOFFICE, &args).map_err(String::from)?;
                continue;
            } else {
                kprintln!();
                return Err(format!("{} exited before it opened a window -- 'jobs' says how", app.name()));
            }
        }
        if crate::dev::lapic::ticks() > deadline {
            kprintln!();
            return Err(format!("{} has not opened a window after {} s", app.name(), WINDOW_S));
        }
        dots += 1;
        if dots % 25 == 0 {
            kprint!(".");
        }
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 5);
    }
}

/// Where LibreOffice keeps its settings, under the session's `HOME`.
const PROFILE: &str = "/root/.config/libreoffice/4/user";

/// LibreOffice's settings, written before its first start, for the two that
/// stand between the typist and a correct document.
///
/// **Calc's AutoInput is off**, and it has to be. It completes a cell from
/// what is already in the column as you type, and Tab accepts the completion
/// -- so typing "Flour" under "Flowers" can leave "Flowers", and the sheet is
/// wrong in a way nobody typed. A person sees the suggestion and keeps
/// typing; a typist cannot see anything.
///
/// **Writer's word completion is off** for the same reason: Enter accepts the
/// word it offers, and the typist ends a paragraph with Enter.
///
/// **Calc recalculates an OpenDocument file when it opens one.** Its default
/// is to trust the values cached in the file, and the sheets `office calc`
/// writes carry formulas with no cached values -- every total read zero.
///
/// Only when no settings exist, so anything somebody has since changed is
/// theirs. LibreOffice reads this file on its first start and keeps what it
/// finds.
fn seed_profile() {
    seed_profile_at(PROFILE)
}

/// The same settings for a profile at `dir` -- the GUI's, or the headless
/// one `office bench` judges with.
///
/// **And threaded formula groups off.** Calc computes a run of filled cells
/// as one group on a thread pool, and in the guest that path left the filled
/// cells empty while the first cell -- not in any group -- was right: task
/// 59196's reference formula passed H3 and failed H4, where the host passes
/// all three. A judge has to compute every formula the same way on every
/// machine, so its profile computes them on one thread. Why the pool fails
/// in the guest is a kernel question and is open.
fn seed_profile_at(dir: &str) {
    let file = format!("{}/registrymodifications.xcu", dir);
    if crate::linux::ns::blob_len(&file).is_some() {
        return;
    }
    let xcu = r#"<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<item oor:path="/org.openoffice.Office.Calc/Input"><prop oor:name="AutoInput" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Writer/AutoFunction/Completion"><prop oor:name="Enable" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="ShowTipOfTheDay" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Common/Misc"><prop oor:name="FirstRun" oor:op="fuse"><value>false</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Formula/Calculation"><prop oor:name="UseThreadedCalculationForFormulaGroups" oor:op="fuse"><value>false</value></prop></item>
</oor:items>
"#;
    let mut path = String::new();
    for part in dir.split('/').filter(|p| !p.is_empty()) {
        path.push('/');
        path.push_str(part);
        if !crate::linux::ns::is_dir(&path) {
            crate::linux::ns::make_dir(&path);
        }
    }
    crate::linux::ns::write_blob(&file, xcu.as_bytes().to_vec());
}

/// Whether the draft has typed anything yet. Whitespace before the first
/// word is not typed: a model opens its answer with a newline as often as
/// not, and Writer meets a document that begins with an empty line with an
/// autocorrect banner across the top of the page.
static BEGUN: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn to_typist(s: &str) {
    use core::sync::atomic::Ordering;
    if BEGUN.load(Ordering::Relaxed) {
        crate::dev::typist::push(s);
        return;
    }
    let t = s.trim_start();
    if !t.is_empty() {
        BEGUN.store(true, Ordering::Relaxed);
        crate::dev::typist::push(t);
    }
}

/// Wait for the typist to finish, so the next command does not land in the
/// document as keystrokes of its own.
fn wait_typed() {
    let deadline = crate::dev::lapic::ticks() + 600 * crate::TIMER_HZ as u64;
    while crate::dev::typist::busy() && crate::dev::lapic::ticks() < deadline {
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 10);
    }
}

pub fn type_into_writer(text: &str) -> Result<(), String> {
    start_writer(None)?;
    crate::gfx::desk::focus_titled(WRITER);
    let n = crate::dev::typist::push(text);
    wait_typed();
    note(format!("typed {} characters", n));
    Ok(())
}

/// What the model is told it is, for every office task.
///
/// Plain text, said three ways, because a small model writes markdown out of
/// habit and a keyboard types every asterisk of it. The console strips
/// markers it does see (`set_plain`), so this is belt and braces.
const SYSTEM: &str = "You are GLaDOS, the operating system of this computer, helping with office work. \
Write exactly what is asked for and nothing else: no preamble, no remarks about the task, no closing offer of more help. \
Write plain text only, with no markdown, no asterisks and no # headings. Separate paragraphs with one blank line.";

/// One chat turn, streamed: to the screen always, and to the typist when
/// `typing`. Answers the text, or why there is none.
/// Where a draft goes besides the screen.
#[derive(Clone, Copy, PartialEq)]
enum Sink {
    Screen,
    Writer,
    Calc,
}

fn draft(user: &str, steps: usize, typing: bool) -> Result<String, String> {
    draft_to(SYSTEM, user, steps, if typing { Sink::Writer } else { Sink::Screen })
}

/// Set while the model is writing for `office`, for the Office window, which
/// must not claim the engine to find out -- a claim from the compositor's task
/// would fail for exactly as long as the answer is "busy".
static DRAFTING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub fn drafting() -> bool {
    DRAFTING.load(core::sync::atomic::Ordering::Relaxed)
}

fn draft_to(system: &str, user: &str, steps: usize, sink: Sink) -> Result<String, String> {
    draft_at(system, user, steps, sink, 0.7)
}

fn draft_at(system: &str, user: &str, steps: usize, sink: Sink, temperature: f32) -> Result<String, String> {
    DRAFTING.store(true, core::sync::atomic::Ordering::Relaxed);
    let out = draft_inner(system, user, steps, sink, temperature);
    DRAFTING.store(false, core::sync::atomic::Ordering::Relaxed);
    out
}

/// One ChatML turn, ready for the assistant to answer.
fn chat_prompt(system: &str, user: &str) -> String {
    let mut prompt = String::from("<|im_start|>system\n");
    prompt.push_str(system);
    prompt.push_str("<|im_end|>\n<|im_start|>user\n");
    prompt.push_str(user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    if crate::ai::has_think_token() {
        // Qwen3 reasons first unless the block is closed for it, which is how
        // its authors document turning that off. A letter does not need it
        // and the reasoning would be typed.
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

fn draft_inner(system: &str, user: &str, steps: usize, sink: Sink, temperature: f32) -> Result<String, String> {
    let prompt = chat_prompt(system, user);
    let opts = crate::ai::GenOpts {
        steps,
        temperature,
        topp: 0.9,
        resume: false,
        // Not yielding between tokens: the timer preempts this task a
        // hundred times a second anyway, which is all the typist and the
        // compositor need, and a yield per token handed core 0 to whatever
        // else was runnable there before every single one.
        yielding: false,
        bos: false,
        echo_prompt: false,
        think: false,
        ..Default::default()
    };
    // Markdown is stripped for prose, and kept for a table: a model writes a
    // table as `| a | b |`, and the pipes are exactly what says where one
    // cell ends.
    crate::ai::set_plain(sink != Sink::Calc);
    crate::ai::echo_begin();
    match sink {
        Sink::Screen => {}
        Sink::Writer => {
            BEGUN.store(false, core::sync::atomic::Ordering::Relaxed);
            crate::ai::set_tap(Some(to_typist));
        }
        Sink::Calc => {
            *SHEET.lock_irq() = Sheet::new();
            crate::ai::set_tap(Some(to_sheet));
        }
    }
    // The connectome takes a whole core for itself, and the decode splits
    // every matrix across all of them: on a four-core guest that is a quarter
    // of the arithmetic the model would otherwise have. It is stepped down
    // while somebody waits on a draft and put back afterwards.
    let pace = crate::brain::pace();
    crate::brain::set_pace(pace.min(20));
    crate::ai::generate(&prompt, &opts);
    crate::brain::set_pace(pace);
    crate::ai::set_tap(None);
    crate::ai::set_plain(false);
    let text = crate::ai::echo_end().unwrap_or_default();
    if text.trim().is_empty() {
        return Err("the model wrote nothing".into());
    }
    Ok(text)
}

/// How many positions the loaded model has. `None` with no model.
fn context() -> Option<usize> {
    crate::ai::with_engine(|e| e.model.cfg.seq_len)
}

pub fn write(what: &str) -> Result<(), String> {
    let Some(_) = context() else { return Err(crate::ai::engine_refusal()) };
    start_writer(None)?;
    crate::gfx::desk::focus_titled(WRITER);
    note(format!("drafting: {}", what));
    let text = draft(what, 600, true)?;
    wait_typed();
    note(format!("{} words typed into the document", text.split_whitespace().count()));
    Ok(())
}

pub fn summarize(path: &str, into_writer: bool) -> Result<(), String> {
    let Some(seq) = context() else { return Err(crate::ai::engine_refusal()) };
    note(format!("reading {} through LibreOffice", path));
    let text = extract(path)?;
    // What fits: the context, less the answer and the framing, at about three
    // characters a token for English prose. A document that does not fit is
    // cut at a paragraph and said to be cut, because a summary of the first
    // half presented as a summary of the whole is the failure that matters.
    let answer = 300usize;
    let room = seq.saturating_sub(answer + 160) * 3;
    let (body, cut) = if text.len() > room {
        let mut end = room.min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let end = text[..end].rfind("\n\n").filter(|&i| i > room / 2).unwrap_or(end);
        (&text[..end], true)
    } else {
        (&text[..], false)
    };
    if cut {
        kprintln!(
            "  the document is {} characters and {} fit this model's context -- summarising the first part",
            text.len(),
            body.len()
        );
    }
    let ask = format!(
        "Summarise this document in a few short paragraphs: what it is, its main points, and anything that needs action.\n\n---\n{}\n---",
        body.trim()
    );
    if into_writer {
        start_writer(None)?;
        crate::gfx::desk::focus_titled(WRITER);
    }
    note(format!("summarising {} characters of {}", body.len(), path));
    let text = draft(&ask, answer, into_writer)?;
    if into_writer {
        wait_typed();
        note(format!("summary typed into Writer, {} words", text.split_whitespace().count()));
    } else {
        note(format!("summary written, {} words", text.split_whitespace().count()));
    }
    Ok(())
}

// --- Calc ------------------------------------------------------------------
//
// The model writes the table; the spreadsheet does the arithmetic. A 0.6B is
// a poor calculator and a good enough clerk, so it is asked for rows and for
// formulas where one cell follows from others, and every total is a `=SUM`
// the kernel adds -- live, so changing a number changes the total, and
// checkable, so a wrong figure is a visible formula rather than a confident
// number.
//
// **Written as a file and opened, not typed.** Typing reached Writer
// perfectly and Calc not at all past the first cell: Tab, which moves to the
// next cell when a person presses it, left LibreOffice's GTK frame with the
// cell still open and every key after it went nowhere. A flat OpenDocument
// spreadsheet is exact where keystrokes are fragile, and it carries what
// typing never could -- number formats, a styled header, a ruled total row.

/// What the model is told for a spreadsheet.
const SHEET_SYSTEM: &str = "You are GLaDOS, the operating system of this computer, building spreadsheets. \
Output only a table, one row per line, with the cells of a row separated by |. \
The first line is the header row. Write every row in full; never abbreviate with dots or a remark. \
Numbers are plain numbers: no currency symbols, no thousands separators, no units inside the cell. \
When a cell is calculated from other cells in its row, write it as a formula such as =B2*12, using spreadsheet references: column A is the first column, row 1 is the header, so the first data row is row 2. \
Do not write a total row; totals are added for you. Write no text before or after the table.\n\
Example:\n| Item | Monthly | Yearly |\n| Rent | 950 | =B2*12 |\n| Food | 420 | =B3*12 |";

/// Cells and rows beyond these are the model running away, not a table.
const MAX_COLS: usize = 12;
const MAX_ROWS: usize = 60;

/// The table as it streams in.
struct Sheet {
    /// The line being built, until its newline arrives.
    line: String,
    /// The rows so far, header first.
    rows: Vec<Vec<String>>,
    /// Per column: whether every data cell in it so far is a number or a
    /// formula. A column with any text in it gets no total.
    numeric: [bool; MAX_COLS],
    /// Whether the column has had any data cell at all.
    seen: [bool; MAX_COLS],
}

impl Sheet {
    const fn new() -> Sheet {
        Sheet { line: String::new(), rows: Vec::new(), numeric: [true; MAX_COLS], seen: [false; MAX_COLS] }
    }
    fn cols(&self) -> usize {
        self.rows.iter().map(|r| r.len()).max().unwrap_or(0)
    }
}

static SHEET: crate::sync::Spin<Sheet> = crate::sync::Spin::new(Sheet::new());

fn to_sheet(s: &str) {
    let mut lines = Vec::new();
    {
        let mut sh = SHEET.lock_irq();
        for c in s.chars() {
            if c == '\n' {
                lines.push(core::mem::take(&mut sh.line));
            } else {
                sh.line.push(c);
            }
        }
    }
    for l in lines {
        sheet_line(&l);
    }
}

/// The column letter for a zero-based index, A to Z then AA onwards.
fn col_name(i: usize) -> String {
    let mut n = i + 1;
    let mut out = Vec::new();
    while n > 0 {
        let r = (n - 1) % 26;
        out.push(b'A' + r as u8);
        n = (n - 1) / 26;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// A cell as it should be stored: markdown and currency taken off, and a
/// number's thousands separators removed so it is a number.
fn clean_cell(raw: &str) -> String {
    let mut c = raw.trim().replace("**", "").replace('`', "");
    c = String::from(c.trim());
    if c.starts_with('=') {
        return c.replace(' ', "");
    }
    let mut n = c.clone();
    for sym in ["€", "£", "$", "EUR", "USD", "GBP"] {
        n = n.replace(sym, "");
    }
    let n = n.trim().replace(',', "");
    let pct = n.ends_with('%');
    let body = n.trim_end_matches('%');
    if !body.is_empty() && body.parse::<f64>().is_ok() {
        return if pct { format!("{}%", body) } else { String::from(body) };
    }
    c
}

fn is_number(c: &str) -> bool {
    c.starts_with('=') || c.trim_end_matches('%').parse::<f64>().is_ok()
}

/// One line of the model's table, kept as one row of cells.
fn sheet_line(line: &str) {
    let t = line.trim();
    if t.is_empty() {
        return;
    }
    // A markdown separator row says nothing a cell needs.
    if t.chars().all(|c| matches!(c, '-' | ':' | '|' | ' ' | '+')) {
        return;
    }
    // A small model doubles its pipes now and then, `| a || b |`, and every
    // doubled pipe would otherwise be an empty cell that shifts the rest of
    // the row a column right of its formula.
    let mut joined = String::from(t);
    while joined.contains("||") {
        joined = joined.replace("||", "|");
    }
    let t = joined.as_str();
    let parts: Vec<&str> = if t.contains('|') {
        t.trim_matches('|').split('|').collect()
    } else if t.contains('\t') {
        t.split('\t').collect()
    } else {
        // Prose around the table, despite being told: not kept.
        return;
    };
    let mut cells: Vec<String> = parts.iter().take(MAX_COLS).map(|p| clean_cell(p)).collect();
    // Trailing empties are a model padding a row with pipes, not cells.
    while cells.last().is_some_and(|c| c.is_empty()) && cells.len() > 1 {
        cells.pop();
    }
    if cells.iter().all(|c| c.is_empty()) {
        return;
    }
    // A row the model abbreviated ("... (other rows similarly)") is not data.
    if cells.iter().any(|c| c.starts_with("...") || c.contains("similarly")) {
        return;
    }
    let mut sh = SHEET.lock_irq();
    if sh.rows.len() >= MAX_ROWS {
        return;
    }
    // **A row one cell short, with a name and its amount in one cell**, is
    // the model's commonest slip -- asked for "rent 950" it writes
    // `| Rent 950 | =B2*12 |`, and every formula after it then points one
    // column left of its number. Split back apart, it is the row it meant.
    // Only when the header says a cell is missing, and only a name followed
    // by a number, so a real "Floor 2" in a table of the right width stays.
    if let Some(want) = sh.rows.first().map(|h| h.len()) {
        if cells.len() + 1 == want {
            if let Some(i) = cells.iter().position(|c| {
                c.rsplit_once(' ').is_some_and(|(name, num)| !name.trim().is_empty() && is_number(num) && !is_number(name.trim()))
            }) {
                let (name, num) = cells[i].rsplit_once(' ').map(|(a, b)| (String::from(a.trim()), String::from(b))).unwrap();
                cells[i] = name;
                cells.insert(i + 1, num);
            }
        }
    }
    // Totals are the kernel's, so a total row the model wrote anyway is left
    // out rather than counted twice.
    if !sh.rows.is_empty() && cells[0].to_ascii_lowercase().starts_with("total") {
        return;
    }
    if !sh.rows.is_empty() {
        for (i, c) in cells.iter().enumerate() {
            if c.is_empty() {
                continue;
            }
            sh.seen[i] = true;
            if !is_number(c) {
                sh.numeric[i] = false;
            }
        }
    }
    sh.rows.push(cells);
}

/// The totals row, as formulas over every numeric column, or `None` when
/// there is nothing to total.
fn totals_row(sh: &Sheet) -> Option<Vec<String>> {
    if sh.rows.len() < 3 {
        return None;
    }
    let last = sh.rows.len();
    let mut cells = alloc::vec![String::from("Total")];
    let mut any = false;
    for i in 1..sh.cols() {
        if sh.seen[i] && sh.numeric[i] {
            let c = col_name(i);
            cells.push(format!("=SUM({}2:{}{})", c, c, last));
            any = true;
        } else {
            cells.push(String::new());
        }
    }
    any.then_some(cells)
}

/// A formula in the A1 syntax a person types, in the ODF syntax a file
/// stores: `=B2*12` is `of:=[.B2]*12` and `=SUM(B2:B9)` is
/// `of:=SUM([.B2:.B9])`. A reference is letters then digits not followed by
/// a parenthesis -- which is what tells `B2` from a function called `LOG10`.
fn odf_formula(f: &str) -> String {
    let body = f.trim_start_matches('=');
    let b = body.as_bytes();
    let mut out = String::from("of:=");
    let mut i = 0;
    let is_ref_at = |i: usize| -> Option<usize> {
        let mut j = i;
        while j < b.len() && b[j].is_ascii_uppercase() && j - i < 3 {
            j += 1;
        }
        if j == i || j >= b.len() || !b[j].is_ascii_digit() {
            return None;
        }
        // Not the tail of a longer name.
        if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
            return None;
        }
        let mut k = j;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        if k < b.len() && (b[k] == b'(' || b[k].is_ascii_alphabetic()) {
            return None;
        }
        Some(k)
    };
    while i < b.len() {
        if let Some(end) = is_ref_at(i) {
            // A range is two references around a colon, and is one bracket.
            if end < b.len() && b[end] == b':' {
                if let Some(end2) = is_ref_at(end + 1) {
                    out.push_str(&format!("[.{}:.{}]", &body[i..end], &body[end + 1..end2]));
                    i = end2;
                    continue;
                }
            }
            out.push_str(&format!("[.{}]", &body[i..end]));
            i = end;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// One cell as ODF: text, a number, a percentage or a formula.
fn odf_cell(c: &str, style: &str) -> String {
    let st = if style.is_empty() { String::new() } else { format!(" table:style-name=\"{}\"", style) };
    if c.is_empty() {
        return format!("<table:table-cell{}/>", st);
    }
    if c.starts_with('=') {
        return format!(
            "<table:table-cell{} table:formula=\"{}\" office:value-type=\"float\"/>",
            st,
            xml_escape(&odf_formula(c))
        );
    }
    if let Some(p) = c.strip_suffix('%') {
        if let Ok(v) = p.parse::<f64>() {
            return format!(
                "<table:table-cell table:style-name=\"pct\" office:value-type=\"percentage\" office:value=\"{}\"><text:p>{}</text:p></table:table-cell>",
                v / 100.0,
                xml_escape(c)
            );
        }
    }
    if c.parse::<f64>().is_ok() {
        return format!(
            "<table:table-cell{} office:value-type=\"float\" office:value=\"{}\"><text:p>{}</text:p></table:table-cell>",
            st, c, c
        );
    }
    format!("<table:table-cell{} office:value-type=\"string\"><text:p>{}</text:p></table:table-cell>", st, xml_escape(c))
}

/// The whole sheet as a flat OpenDocument spreadsheet.
fn fods(rows: &[Vec<String>], totals: Option<&Vec<String>>, title: &str) -> String {
    let cols = rows.iter().chain(totals).map(|r| r.len()).max().unwrap_or(1);
    let mut x = String::from(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:number="urn:oasis:names:tc:opendocument:xmlns:datastyle:1.0" xmlns:of="urn:oasis:names:tc:opendocument:xmlns:of:1.2" office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.spreadsheet">
<office:automatic-styles>
<number:number-style style:name="N2"><number:number number:decimal-places="2" number:min-decimal-places="0" number:min-integer-digits="1" number:grouping="true"/></number:number-style>
<number:percentage-style style:name="P1"><number:number number:decimal-places="1" number:min-integer-digits="1"/><number:text>%</number:text></number:percentage-style>
<style:style style:name="first" style:family="table-column"><style:table-column-properties style:column-width="2.2in"/></style:style>
<style:style style:name="col" style:family="table-column"><style:table-column-properties style:column-width="1.2in"/></style:style>
<style:style style:name="hdr" style:family="table-cell"><style:table-cell-properties fo:background-color="#f28c1e" fo:padding="0.03in"/><style:text-properties fo:font-weight="bold" fo:color="#ffffff"/></style:style>
<style:style style:name="num" style:family="table-cell" style:data-style-name="N2"/>
<style:style style:name="pct" style:family="table-cell" style:data-style-name="P1"/>
<style:style style:name="tot" style:family="table-cell" style:data-style-name="N2"><style:table-cell-properties fo:border-top="0.75pt solid #000000" fo:background-color="#fbe3c8"/><style:text-properties fo:font-weight="bold"/></style:style>
</office:automatic-styles>
<office:body><office:spreadsheet>
"##,
    );
    x.push_str(&format!("<table:table table:name=\"{}\">\n", xml_escape(title)));
    x.push_str("<table:table-column table:style-name=\"first\"/>\n");
    if cols > 1 {
        x.push_str(&format!("<table:table-column table:style-name=\"col\" table:number-columns-repeated=\"{}\"/>\n", cols - 1));
    }
    for (r, row) in rows.iter().enumerate() {
        x.push_str("<table:table-row>");
        for i in 0..cols {
            let c = row.get(i).map(|s| s.as_str()).unwrap_or("");
            let style = if r == 0 { "hdr" } else if i > 0 { "num" } else { "" };
            x.push_str(&odf_cell(c, style));
        }
        x.push_str("</table:table-row>\n");
    }
    if let Some(t) = totals {
        x.push_str("<table:table-row>");
        for i in 0..cols {
            let c = t.get(i).map(|s| s.as_str()).unwrap_or("");
            x.push_str(&odf_cell(c, "tot"));
        }
        x.push_str("</table:table-row>\n");
    }
    x.push_str("</table:table>\n</office:spreadsheet></office:body></office:document>\n");
    x
}

/// A file name from what was asked for: its first few words.
fn slug(what: &str) -> String {
    let words: Vec<String> = what
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty() && !matches!(w.to_ascii_lowercase().as_str(), "a" | "an" | "the" | "of" | "for" | "with"))
        .take(3)
        .map(|w| w.to_ascii_lowercase())
        .collect();
    if words.is_empty() { String::from("sheet") } else { words.join("-") }
}

pub fn calc(what: &str) -> Result<(), String> {
    let Some(_) = context() else { return Err(crate::ai::engine_refusal()) };
    ready()?;
    note(format!("building a sheet: {}", what));
    draft_to(SHEET_SYSTEM, what, 700, Sink::Calc)?;
    // A last row with no newline after it is still a row.
    let rest = core::mem::take(&mut SHEET.lock_irq().line);
    sheet_line(&rest);
    let (rows, totals) = {
        let sh = SHEET.lock_irq();
        (sh.rows.clone(), totals_row(&sh))
    };
    if rows.len() < 2 {
        return Err("the model wrote no table".into());
    }
    let name = slug(what);
    let path = format!("{}/{}.fods", OUT, name);
    if !crate::linux::ns::is_dir(OUT) {
        crate::linux::ns::make_dir(OUT);
    }
    let title: String = name.chars().take(31).collect();
    crate::linux::ns::write_blob(&path, fods(&rows, totals.as_ref(), &title).into_bytes());
    note(format!(
        "{} rows by {} columns{} -> {}",
        rows.len() - 1,
        rows.iter().map(|r| r.len()).max().unwrap_or(0),
        if totals.is_some() { ", totals as =SUM formulas" } else { "" },
        path
    ));
    start_app(App::Calc, Some(&path))?;
    note(format!("opened in Calc; 'office convert {} xlsx' makes an Excel copy", path));
    Ok(())
}

// --- bench -----------------------------------------------------------------
//
// SpreadsheetBench (Ma et al., NeurIPS 2024): questions people asked on Excel
// forums, each with a workbook, the cells the answer goes in, and two or
// three test cases -- input workbooks and the answer workbooks they should
// become. A task is solved only when the answer holds on every case, which
// is what stops a formula that hardcodes one sheet's numbers from scoring.
//
// The model is the kernel's and sees what a person on the forum saw: the
// sheet and the question. LibreOffice is the judge, through `bench.bas` in
// its own profile: it puts the formula in the first answer cell, fills it
// across the rest the way a person drags it, recalculates, and compares cell
// by cell. Only the cell-level tasks are attempted -- a sheet-level task
// wants a sequence of edits, and one formula is the honest unit for a model
// this size.

const BENCH_DIR: &str = "/mnt/bench/sb";
const BENCH_OUT: &str = "/tmp/office/bench";
const BENCH_BAS: &str = include_str!("office/bench.bas");

const BENCH_SYSTEM: &str = "You are an Excel expert. You are shown a spreadsheet as cells (address=value, with {formula} where a cell has one) and a question about it. \
Answer with exactly one Excel formula for the cell named, starting with =, which will be filled down and across the answer range the way a formula is dragged. \
Use relative references so it works when filled. Never refer to the answer cell itself. \
Write only the formula, with no explanation: for example =SUM(B2:B9) or =IF(C2>0,\"yes\",\"no\").";

/// The headless profile's user directory, where its macros live.
const HEADLESS_USER: &str = "/tmp/office/profile/user";

/// Put `Bench` into the headless profile's Standard library.
///
/// After the profile exists rather than before: LibreOffice builds a new
/// profile's Basic libraries from its presets, and a library written first
/// would be replaced by the default one.
fn seed_bench_macros() -> Result<(), String> {
    let base = format!("{}/basic", HEADLESS_USER);
    let std = format!("{}/Standard", base);
    for d in [HEADLESS_USER, base.as_str(), std.as_str()] {
        if !crate::linux::ns::is_dir(d) {
            crate::linux::ns::make_dir(d);
        }
    }
    let code = BENCH_BAS.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
    let module = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE script:module PUBLIC \"-//OpenOffice.org//DTD OfficeDocument 1.0//EN\" \"module.dtd\">\n<script:module xmlns:script=\"http://openoffice.org/2000/script\" script:name=\"Bench\" script:language=\"StarBasic\">{}</script:module>\n",
        code
    );
    let lib = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE library:library PUBLIC \"-//OpenOffice.org//DTD OfficeDocument 1.0//EN\" \"library.dtd\">\n<library:library xmlns:library=\"http://openoffice.org/2000/library\" library:name=\"Standard\" library:readonly=\"false\" library:passwordprotected=\"false\">\n <library:element library:name=\"Bench\"/>\n</library:library>\n";
    let index = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE library:libraries PUBLIC \"-//OpenOffice.org//DTD OfficeDocument 1.0//EN\" \"libraries.dtd\">\n<library:libraries xmlns:library=\"http://openoffice.org/2000/library\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <library:library library:name=\"Standard\" xlink:href=\"$(USER)/basic/Standard/script.xlb/\" xlink:type=\"simple\" library:link=\"false\"/>\n</library:libraries>\n";
    let ok = crate::linux::ns::write_blob(&format!("{}/Bench.xba", std), module.into_bytes())
        && crate::linux::ns::write_blob(&format!("{}/script.xlb", std), lib.as_bytes().to_vec())
        && crate::linux::ns::write_blob(&format!("{}/script.xlc", base), index.as_bytes().to_vec());
    if ok { Ok(()) } else { Err("could not write the bench macros into LibreOffice's profile".into()) }
}

/// Run one of `bench.bas`'s entry points over a whole batch: one record per
/// line, fields separated by tabs. Each record names its own output file,
/// so what comes back is read per task by the caller.
fn macro_batch(entry: &str, records: &[Vec<String>]) {
    let path = format!("{}/{}.job", BENCH_OUT, entry.to_ascii_lowercase());
    let mut text = String::new();
    for r in records {
        let _ = crate::linux::ns::detach(&r[0]);
        text.push_str(&r.join("\t"));
        text.push('\n');
    }
    crate::linux::ns::write_blob(&path, text.into_bytes());
    // No argument in the URL: `bench.bas` reads the job from the path above.
    let url = format!("macro:///Standard.Bench.{}", entry);
    // Twice at most: the first full start on a new profile can be the one
    // that restarts itself (status 81, see `headless`) and runs no macro.
    for _ in 0..2 {
        // A minute a record on top of the start: a big workbook's three
        // test cases in the guest take tens of seconds, and a batch cut off
        // by its own deadline leaves every later task without a verdict.
        let budget = CONVERT_S + 60 * records.len() as u64;
        let _ = run_to_end(&["--headless", "--norestore", "--nologo", HEADLESS, &url], budget);
        if records.iter().any(|r| read_any(&r[0]).is_ok()) {
            return;
        }
    }
}

/// The headless profile, with `bench.bas` in it, ready for `macro_batch`.
fn headless_macros() -> Result<(), String> {
    for d in [OUT, BENCH_OUT] {
        if !crate::linux::ns::is_dir(d) {
            crate::linux::ns::make_dir(d);
        }
    }
    // The headless profile first, then the macros into it -- see
    // `seed_bench_macros` for the order.
    if !crate::linux::ns::is_dir(&format!("{}/basic/Standard", HEADLESS_USER)) {
        seed_profile_at(HEADLESS_USER);
        note(String::from("making the headless profile (once)"));
        for _ in 0..2 {
            run_to_end(&["--headless", "--norestore", "--nologo", HEADLESS, "--terminate_after_init"], CONVERT_S)?;
        }
    }
    seed_bench_macros()
}

/// The answer position split into an optional sheet and a range.
fn split_position(p: &str) -> (String, String) {
    match p.rsplit_once('!') {
        Some((sheet, range)) => (String::from(sheet.trim_matches('\'')), String::from(range)),
        None => (String::new(), String::from(p)),
    }
}

/// The first formula in what the model wrote. An array formula keeps its
/// braces, `{=...}`, because it is entered differently: over the whole range
/// at once rather than in one cell and filled.
fn formula_in(text: &str) -> Option<String> {
    for l in text.lines() {
        let l = l.trim().trim_matches('`').trim();
        if let Some(i) = l.find("{=") {
            if let Some(j) = l[i..].find('}') {
                return Some(String::from(&l[i..i + j + 1]));
            }
        }
        if let Some(i) = l.find('=') {
            let f = l[i..].trim().trim_end_matches(|c| c == '`' || c == '.' || c == ';').trim();
            if f.len() > 1 {
                return Some(String::from(f));
            }
        }
    }
    None
}

/// What a small model wraps around a formula, taken off it.
///
/// Measured on the first 40 tasks, where Qwen3-0.6B solved none: about one
/// answer in three named the answer cell before the formula, as `=C1=LEFT(..)`
/// or `=B7:B10=SUM(..)`, which Calc reads as a comparison against the cell
/// itself, a circular reference. Others ended in prose (`=B5 on sheet
/// Overview`) or restated themselves (`=K27 formula: =C5$C$16`). Only the
/// target's own name is stripped: `=A1=B1` is a perfectly good formula
/// whenever A1 is not where it is going.
fn tidy(f: &str, position: &str) -> String {
    let mut f = String::from(f.trim());
    if let Some(i) = f.rfind("formula:") {
        if let Some(j) = f[i..].find('=') {
            f = String::from(f[i + j..].trim());
        }
    }
    if let Some(i) = f.find(" on sheet") {
        f.truncate(i);
    }
    let (_, range) = split_position(position);
    let upper = f.to_ascii_uppercase();
    let first = String::from(range.split(':').next().unwrap_or(&range)).to_ascii_uppercase();
    for name in [range.to_ascii_uppercase(), first] {
        let lead = format!("={}=", name);
        if upper.starts_with(&lead) && f.len() > lead.len() {
            f = format!("={}", &f[lead.len()..]);
            break;
        }
    }
    f
}

/// Which candidate to keep, from what each computed on the input.
///
/// A line is `OK` and the range's values, or `ERR` and where. A candidate is
/// usable when it computed without error, produced something in at least one
/// cell, and refers to a cell at all -- a constant can fill a range and say
/// nothing about the question. Among the usable the most common result wins,
/// the way independent attempts agreeing is evidence none of them is alone;
/// ties go to the earlier candidate, so the greedy answer leads. `None` when
/// nothing is usable, which leaves the greedy answer in place.
fn choose(candidates: &[String], probed: &[&str]) -> Option<usize> {
    let refers = |f: &str| {
        let b = f.as_bytes();
        (1..b.len()).any(|i| b[i].is_ascii_digit() && b[i - 1].is_ascii_alphabetic())
    };
    let usable: Vec<usize> = (0..candidates.len().min(probed.len()))
        .filter(|&i| {
            let l = probed[i];
            l.starts_with("OK")
                && l.split('\t').skip(1).any(|v| !v.trim().is_empty())
                && refers(&candidates[i])
        })
        .collect();
    let mut best: Option<(usize, usize)> = None;
    for &i in &usable {
        let n = usable.iter().filter(|&&j| probed[j] == probed[i]).count();
        if best.map_or(true, |(_, m)| n > m) {
            best = Some((i, n));
        }
    }
    best.map(|(i, _)| i)
}

pub fn bench(arg: &str) -> Result<(), String> {
    let Some(_) = context() else { return Err(crate::ai::engine_refusal()) };
    ready()?;
    // `--oracle` answers each task with the formula its own answer workbook
    // keeps, through the same checker: the ceiling this harness can measure,
    // and the proof that it can say "solved" at all.
    let oracle = arg.split_whitespace().any(|w| w == "--oracle");
    // `--k N`: N formulas a task, sampled from one prefill, tried on the
    // input workbook, and the one most of the working candidates agree on
    // kept. One is the old single greedy answer and probes nothing.
    let k: usize = arg
        .split_whitespace()
        .skip_while(|w| *w != "--k")
        .nth(1)
        .and_then(|w| w.parse().ok())
        .unwrap_or(1)
        .clamp(1, 32);
    // The directory is optional, so it is peeked rather than taken: `next`
    // then `filter` consumed the first number when no directory was given,
    // and `office bench 0 40` ran from task 40.
    let mut skip_next = false;
    let mut words = arg
        .split_whitespace()
        .filter(|w| {
            if skip_next {
                skip_next = false;
                return false;
            }
            if *w == "--k" {
                skip_next = true;
                return false;
            }
            *w != "--oracle"
        })
        .peekable();
    let dir = match words.peek() {
        Some(w) if w.starts_with('/') => words.next().unwrap(),
        _ => BENCH_DIR,
    };
    let from: usize = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
    let count: usize = words.next().and_then(|w| w.parse().ok()).unwrap_or(usize::MAX);
    let manifest = read_any(&format!("{}/tasks.tsv", dir))?;
    let all: Vec<Vec<String>> = manifest
        .lines()
        .map(|l| l.splitn(4, '\t').map(String::from).collect::<Vec<_>>())
        .filter(|t| t.len() == 4)
        .collect();
    let tasks: Vec<&Vec<String>> = all.iter().skip(from).take(count).collect();
    if tasks.is_empty() {
        return Err(format!("no tasks in {}/tasks.tsv", dir));
    }
    headless_macros()?;
    let input = |id: &str, k: usize| format!("{}/{}/{}_{}_input.xlsx", dir, id, k, id);
    let answer = |id: &str, k: usize| format!("{}/{}/{}_{}_answer.xlsx", dir, id, k, id);
    let first_cell = |range: &str| String::from(range.split(':').next().unwrap_or(range));

    // 1. What each task's reader saw, or its reference formula -- one launch.
    let records: Vec<Vec<String>> = tasks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let (sheet, range) = split_position(&t[1]);
            if oracle {
                alloc::vec![format!("{}/f{}.txt", BENCH_OUT, i), sheet, answer(&t[0], 1), first_cell(&range)]
            } else {
                alloc::vec![format!("{}/v{}.txt", BENCH_OUT, i), sheet, input(&t[0], 1)]
            }
        })
        .collect();
    note(format!("{} task(s): LibreOffice {}", tasks.len(), if oracle { "reads the reference formulas" } else { "renders the sheets" }));
    macro_batch(if oracle { "Formula" } else { "View" }, &records);

    // 2. A formula per task, with nothing else running.
    let seq = context().unwrap_or(2048);
    let mut formulas: Vec<Option<String>> = Vec::new();
    let mut candidates: Vec<Vec<String>> = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let (sheet, range) = split_position(&t[1]);
        let got = read_any(&records[i][0]).unwrap_or_default();
        let said = if oracle {
            got
        } else {
            let mut view = got;
            // Cut to fit: the context less the answer and the framing, at two
            // characters a token. Three is right for prose and wrong here: a
            // sheet dump is addresses and numbers, and at three some prompts
            // ran past the context, where prefill stops and the model answers
            // nothing.
            let room = (seq.saturating_sub(400) * 2).saturating_sub(t[3].len() * 2 / 3 + 300);
            if view.len() > room {
                let mut e = room;
                while !view.is_char_boundary(e) {
                    e -= 1;
                }
                view.truncate(e);
                view.push_str("\n(the sheet continues)\n");
            }
            let user = format!(
                "{}\nQuestion: {}\n\nThe answer goes in {}{}. Reply with the formula {} should contain.",
                view.trim_end(),
                t[3],
                range,
                if sheet.is_empty() { String::new() } else { format!(" on sheet {}", sheet) },
                first_cell(&range)
            );
            note(format!("{} ({} of {}): asking the model", t[0], i + 1, tasks.len()));
            if k > 1 {
                let drawn = crate::ai::samples(&chat_prompt(BENCH_SYSTEM, &user), k, 120, 0.8, 0.95).unwrap_or_default();
                let mut seen: Vec<String> = Vec::new();
                for d in &drawn {
                    if let Some(f) = formula_in(d).map(|f| tidy(&f, &t[1])) {
                        // A tab or a newline would split the job record.
                        let f = f.replace(['\t', '\n', '\r'], " ");
                        if !seen.contains(&f) {
                            seen.push(f);
                        }
                    }
                }
                note(format!("  {} distinct formula(s) from {} draw(s)", seen.len(), drawn.len()));
                let first = seen.first().cloned().unwrap_or_default();
                candidates.push(seen);
                first
            } else {
                draft_at(BENCH_SYSTEM, &user, 120, Sink::Screen, 0.0).unwrap_or_default()
            }
        };
        if k > 1 && !oracle {
            formulas.push(candidates.last().and_then(|c| c.first().cloned()));
        } else {
            formulas.push(formula_in(&said).map(|f| if oracle { f } else { tidy(&f, &t[1]) }));
        }
    }

    // 2b. Each task's candidates tried on its first input -- one launch. The
    // answer workbooks are not opened: what decides is only what trying a
    // formula shows, and the check below is still the only judge.
    if k > 1 && !oracle {
        let probes: Vec<Vec<String>> = tasks
            .iter()
            .enumerate()
            .filter(|(i, _)| candidates[*i].len() > 1)
            .map(|(i, t)| {
                let (sheet, range) = split_position(&t[1]);
                let mut r = alloc::vec![format!("{}/p{}.txt", BENCH_OUT, i), sheet, range, input(&t[0], 1)];
                r.extend(candidates[i].iter().cloned());
                r
            })
            .collect();
        note(format!("LibreOffice tries the candidates of {} task(s)", probes.len()));
        if !probes.is_empty() {
            macro_batch("Probe", &probes);
        }
        for (i, t) in tasks.iter().enumerate() {
            if candidates[i].len() < 2 {
                continue;
            }
            let lines = read_any(&format!("{}/p{}.txt", BENCH_OUT, i)).unwrap_or_default();
            let probed: Vec<&str> = lines.lines().collect();
            if let Some(c) = choose(&candidates[i], &probed) {
                note(format!("{} chose {} of {}: {}", t[0], c + 1, candidates[i].len(), candidates[i][c]));
                formulas[i] = Some(candidates[i][c].clone());
            }
        }
    }

    // 3. Every answer checked against every test case -- one launch.
    let mut checks: Vec<Vec<String>> = Vec::new();
    let mut which: Vec<usize> = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let Some(f) = &formulas[i] else { continue };
        let (sheet, range) = split_position(&t[1]);
        let n = t[2].parse::<usize>().unwrap_or(1);
        let mut r = alloc::vec![format!("{}/c{}.txt", BENCH_OUT, i), sheet, range, f.clone()];
        for k in 1..=n {
            r.push(input(&t[0], k));
            r.push(answer(&t[0], k));
        }
        checks.push(r);
        which.push(i);
    }
    note(format!("LibreOffice checks {} answer(s) against their test cases", checks.len()));
    if !checks.is_empty() {
        macro_batch("Check", &checks);
    }

    let (mut solved, mut cases, mut passed) = (0usize, 0usize, 0usize);
    let mut log = String::new();
    for (i, t) in tasks.iter().enumerate() {
        let n = t[2].parse::<usize>().unwrap_or(1);
        cases += n;
        let Some(f) = &formulas[i] else {
            note(format!("{} no formula", t[0]));
            log.push_str(&format!("{}\t0\t0/{}\t-\t\n", t[0], n));
            continue;
        };
        let c = which.iter().position(|&w| w == i).map(|j| checks[j][0].clone()).unwrap_or_default();
        let verdict = read_any(&c).unwrap_or_else(|_| String::from("ERROR no verdict"));
        let ok = verdict.lines().filter(|l| l.starts_with("PASS")).count().min(n);
        passed += ok;
        let all = ok == n;
        if all {
            solved += 1;
        }
        let why = verdict.lines().find(|l| !l.starts_with("PASS") && !l.trim().is_empty()).unwrap_or("").trim();
        note(format!("{} {} {}/{}  {}  {}", t[0], if all { "SOLVED" } else { "failed" }, ok, n, f, if all { "" } else { why }));
        log.push_str(&format!("{}\t{}\t{}/{}\t{}\t{}\n", t[0], all as u8, ok, n, f, why));
    }
    let name = if oracle { "oracle" } else { "model" };
    let _ = crate::linux::ns::write_blob(&format!("{}/results-{}.tsv", BENCH_OUT, name), log.into_bytes());
    note(format!(
        "SpreadsheetBench, cell-level{}: {} of {} tasks solved on every test case, {} of {} test cases passed",
        if oracle { ", answered by the reference formula" } else { ", answered by the model" },
        solved,
        tasks.len(),
        passed,
        cases
    ));
    Ok(())
}

/// What `diag office` asks, with no LibreOffice and no model.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    out.push(("column 0 is A and 25 is Z", col_name(0) == "A" && col_name(25) == "Z"));
    out.push(("column 26 is AA", col_name(26) == "AA"));
    out.push(("a currency amount loses its symbol and separators", clean_cell(" €84,000 ") == "84000"));
    out.push(("a percentage stays a percentage", clean_cell("12%") == "12%"));
    out.push(("a formula keeps its = and loses its spaces", clean_cell("= B2 * 12") == "=B2*12"));
    out.push(("bold markers come off", clean_cell("**Flour**") == "Flour"));
    out.push(("text stays text", clean_cell("Oven repair") == "Oven repair"));
    out.push(("a formula counts as a number", is_number("=SUM(B2:B4)")));
    out.push(("a word does not", !is_number("Flour")));
    out.push(("a cell reference becomes ODF's bracketed form", odf_formula("=B2*12") == "of:=[.B2]*12"));
    out.push(("and a range is one bracket", odf_formula("=SUM(B2:B9)") == "of:=SUM([.B2:.B9])"));
    out.push(("a function name with digits is not a reference", odf_formula("=LOG10(C3)") == "of:=LOG10([.C3])"));
    // A whole table through the parser: a separator row, an abbreviation, a
    // total the model added itself, and prose around it.
    *SHEET.lock_irq() = Sheet::new();
    for l in [
        "Here is your budget:",
        "| Item | Monthly | Yearly |",
        "|------|---------|--------|",
        "| Rent | €950 | =B2*12 |",
        "| Water 30 || =B4*12 ||",
        "| Food | 420 | =B3*12 |",
        "| ... (other rows similarly) | | |",
        "| Total | 1370 | |",
    ] {
        sheet_line(l);
    }
    let (rows, tot) = {
        let sh = SHEET.lock_irq();
        (sh.rows.clone(), totals_row(&sh))
    };
    *SHEET.lock_irq() = Sheet::new();
    out.push(("prose, separators, abbreviations and the model's own total are dropped", rows.len() == 4));
    out.push((
        "a name and its amount fused into one cell are split back apart",
        rows.get(2).is_some_and(|r| r.len() == 3 && r[0] == "Water" && r[1] == "30"),
    ));
    out.push(("a currency cell became a number", rows.get(1).and_then(|r| r.get(1)).map(|c| c.as_str()) == Some("950")));
    out.push((
        "the kernel's total sums every numeric column over the data rows",
        tot.as_ref().is_some_and(|t| t.get(1).map(|c| c.as_str()) == Some("=SUM(B2:B4)") && t.get(2).map(|c| c.as_str()) == Some("=SUM(C2:C4)")),
    ));
    let doc = fods(&rows, tot.as_ref(), "t");
    out.push(("the file carries the formulas in ODF syntax", doc.contains("of:=[.B2]*12") && doc.contains("of:=SUM([.B2:.B4])")));
    out.push(("and escapes what XML would misread", xml_escape("a<b&c") == "a&lt;b&amp;c"));
    out.push(("a sheet-qualified answer position splits, quotes off", split_position("'My Sheet'!B2:B9") == (String::from("My Sheet"), String::from("B2:B9"))));
    out.push(("a bare one has no sheet", split_position("H3:H5") == (String::new(), String::from("H3:H5"))));
    out.push(("the formula is found inside a code fence", formula_in("```\n=SUM(A1:A3)\n```").as_deref() == Some("=SUM(A1:A3)")));
    out.push(("a formula that names its own cell first loses the name", tidy("=C1=LEFT(A1,2)", "C1:C9") == "=LEFT(A1,2)"));
    out.push(("and so does one naming its whole range", tidy("=B7:B10=SUM(B3:B5)", "B7:B10") == "=SUM(B3:B5)"));
    out.push(("but a comparison of another cell is a formula and stays", tidy("=A1=B1", "C1:C9") == "=A1=B1"));
    out.push(("prose after a formula is cut", tidy("=B5 on sheet Overview", "'Overview'!C5") == "=B5"));
    out.push(("a formula restated after a label is the restatement", tidy("=K27 formula: =C5$C$16", "K27") == "=C5$C$16"));
    let c = |v: &[&str]| v.iter().map(|x| String::from(*x)).collect::<Vec<_>>();
    out.push(("an erroring candidate is passed over", choose(&c(&["=A1/0", "=A1+1"]), &["ERR B1 #DIV/0!", "OK\t2"]) == Some(1)));
    out.push(("the result most candidates agree on wins", choose(&c(&["=A1", "=B1*2", "=B1+B1"]), &["OK\t1", "OK\t4", "OK\t4"]) == Some(1)));
    out.push(("a tie goes to the earlier, so greedy leads", choose(&c(&["=A1", "=B1"]), &["OK\t1", "OK\t2"]) == Some(0)));
    out.push(("a constant says nothing and is not chosen", choose(&c(&["=5", "=A1"]), &["OK\t5", "OK\t3"]) == Some(1)));
    out.push(("an empty result is not an answer", choose(&c(&["=A1", "=B1"]), &["OK\t", "OK\t7"]) == Some(1)));
    out.push(("nothing usable leaves the choice alone", choose(&c(&["=A1"]), &["ERR A1 Err:522"]).is_none()));
    out.push(("an array formula keeps its braces", formula_in("{=A1:A3*B1:B3}").as_deref() == Some("{=A1:A3*B1:B3}")));
    out.push(("and after a preamble, without its full stop", formula_in("Use this: =INDEX(A1:C1,MATCH(MAX(A2:C2),A2:C2,0)).").as_deref() == Some("=INDEX(A1:C1,MATCH(MAX(A2:C2),A2:C2,0))")));
    out.extend(ask::checks());
    out.extend(table::checks());
    out
}
