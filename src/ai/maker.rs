//! Writing a real program from a sentence: a web page, a Python program, a C
//! program or an Aiksi one, checked by the tools that would run it.
//!
//! `author.rs` was the first answer to "write me an application", and it was
//! honest about being a prototype: the model chose among skeletons and
//! widgets, every operand an index into a list, so it could not write a line
//! of logic and could not write anything wrong. That was the right first
//! bargain for a 135M checkpoint and the wrong last one -- what came out was a
//! panel with buttons, never a program.
//!
//! This keeps the half of that bargain that was worth keeping and drops the
//! other. **The model writes free text now**, a whole file, in the language
//! the program is for. **What decides whether it worked is still a machine,
//! never the model's opinion of its own work** -- but the machine is the real
//! toolchain: Python's own parser and the program actually run, gcc, a
//! JavaScript engine compiling every script on the page, the Aiksi
//! interpreter. A failure comes back as the tool's own words with a line
//! number, and the model is handed exactly that and asked for the file again.
//! That loop -- write, check, read the error, rewrite -- is what turns a small
//! model's first draft into a program that runs, and it is the same loop a
//! person runs.
//!
//! ### Who writes
//!
//! `Brain::Local` is the model in this kernel and is the default: the machine
//! writes programs with nothing outside it. `Brain::Cloud` is optional and off
//! until somebody configures it (`maker cloud ...`), and it is reached the way
//! everything network-shaped and userspace-shaped is reached here -- a short
//! Python program in the guest, which has a TLS client with POST and the CA
//! bundle, where the kernel's own client is GET-only by design. A cloud brain
//! that fails is reported and the local one is not silently substituted:
//! "which model wrote this" is part of the result.
//!
//! ### What is checked, per target
//!
//! | target | file | checked by |
//! |---|---|---|
//! | `web` | `index.html` | the HTML has a body; every script compiles in a JS engine; every element the script looks up by id exists |
//! | `python` | `main.py` | `ast.parse`, then the program is run for a few seconds with no input |
//! | `c` | `main.c` | `gcc -Wall`, then the binary is run the same way |
//! | `aiksi` | `main.ai&xi` | the interpreter, sandboxed, under a step budget |
//!
//! "Run with no input" is a smoke test, not a test suite: it catches the
//! program that dies on its first line, which with a small model is most of
//! the programs that are wrong. A program that waits for input or serves
//! forever is not a failure for doing so -- running out of input or out of
//! time is reported as that, and only a crash is sent back to be fixed.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// What kind of program.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Web,
    Python,
    C,
    Aiksi,
}

impl Target {
    pub const ALL: [Target; 4] = [Target::Web, Target::Python, Target::C, Target::Aiksi];

    pub fn name(self) -> &'static str {
        match self {
            Target::Web => "web",
            Target::Python => "python",
            Target::C => "c",
            Target::Aiksi => "aiksi",
        }
    }

    pub fn parse(s: &str) -> Option<Target> {
        match s {
            "web" | "html" | "site" | "page" => Some(Target::Web),
            "python" | "py" => Some(Target::Python),
            "c" => Some(Target::C),
            "aiksi" | "native" | "glados" => Some(Target::Aiksi),
            _ => None,
        }
    }

    pub fn file(self) -> &'static str {
        match self {
            Target::Web => "index.html",
            Target::Python => "main.py",
            Target::C => "main.c",
            Target::Aiksi => "main.ai&xi",
        }
    }

    /// The fence a model writes the file inside.
    fn fence(self) -> &'static str {
        match self {
            Target::Web => "html",
            Target::Python => "python",
            Target::C => "c",
            // The model writes the Python-shaped surface, so it fences it as
            // the language it already knows.
            Target::Aiksi => "python",
        }
    }

    fn language(self) -> &'static str {
        match self {
            Target::Web => "a single-file web page (HTML with its CSS in a <style> and its JavaScript in a <script>)",
            Target::Python => "a Python 3 program",
            Target::C => "a C program",
            Target::Aiksi => "an Aiksi program (Aiksi is written exactly like simple Python)",
        }
    }

    /// What the program has to be, said to the model once, up front. Each
    /// rule is one the checker would otherwise have to send back.
    fn rules(self) -> &'static str {
        match self {
            Target::Web => {
                "- One file. No external libraries, no CDN links, no images from the internet.\n\
                 - Every element the script uses must exist in the HTML with that exact id.\n\
                 - Plain modern JavaScript; no frameworks, no build step.\n\
                 - It must look finished: a heading, sensible spacing, readable colours."
            }
            Target::Python => {
                "- Python 3 standard library only; nothing that needs pip.\n\
                 - No GUI toolkit: a command-line program that prints its results.\n\
                 - Put the program in functions and run it from `if __name__ == \"__main__\":`.\n\
                 - If it reads input, handle the end of input without crashing."
            }
            Target::C => {
                "- One file, standard C (C11) and the C standard library only.\n\
                 - A complete program with `int main(void)` that returns 0.\n\
                 - No warnings with -Wall."
            }
            Target::Aiksi => AIKSI_RULES,
        }
    }
}

/// Aiksi is not in any model's training data, so the model is told the
/// language rather than its name. Short, and only what a program needs.
const AIKSI_RULES: &str = "- Aiksi is written like Python: def, if/elif/else, while, for x in xs, \
return, break, continue, lists [1, 2], xs[i], xs.append(v), len, range, str, int, print, f-strings, \
True/False/None, and/or/not, `in`.\n\
- Numbers are whole numbers only: no floats; `/` divides and rounds down.\n\
- Not available: classes, dictionaries, try/except, slices, default arguments, lambda, imports. \
Use lists, and functions that take what they need.\n\
- Put the work in functions and finish with `main()` on the last line.";

/// Who writes the code.
#[derive(Clone, PartialEq, Eq)]
pub enum Brain {
    /// The model in this kernel.
    Local,
    /// An OpenAI-compatible chat endpoint, reached from the guest.
    Cloud { url: String, model: String },
}

impl Brain {
    pub fn describe(&self) -> String {
        match self {
            Brain::Local => String::from("the local model"),
            Brain::Cloud { model, .. } => format!("{} (cloud)", model),
        }
    }
}

/// Where the cloud settings live. The key is in a file of its own so that
/// `maker` can print the rest without printing it.
pub const CLOUD_CFG: &str = "/etc/maker/cloud";
pub const CLOUD_KEY: &str = "/etc/maker/cloud.key";

/// The brain in force: the cloud only when it is configured *and* switched
/// on, the local model otherwise.
pub fn brain() -> Brain {
    let cfg = crate::sysbox::read_blob(CLOUD_CFG).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let mut on = false;
    let mut url = String::new();
    let mut model = String::new();
    for line in cfg.lines() {
        match line.split_once('=') {
            Some(("use", v)) => on = v.trim() == "on",
            Some(("url", v)) => url = String::from(v.trim()),
            Some(("model", v)) => model = String::from(v.trim()),
            _ => {}
        }
    }
    if on && !url.is_empty() && !model.is_empty() {
        Brain::Cloud { url, model }
    } else {
        Brain::Local
    }
}

pub fn set_cloud(url: Option<&str>, model: Option<&str>, on: Option<bool>) {
    let (mut u, mut m, mut o) = match brain_cfg() {
        (u, m, o) => (u, m, o),
    };
    if let Some(x) = url {
        u = String::from(x);
    }
    if let Some(x) = model {
        m = String::from(x);
    }
    if let Some(x) = on {
        o = x;
    }
    crate::sysbox::make_dir("/etc/maker");
    crate::sysbox::write_text(CLOUD_CFG, &format!("use={}\nurl={}\nmodel={}\n", if o { "on" } else { "off" }, u, m));
}

/// (url, model, switched on) as written, whether or not it is usable.
pub fn brain_cfg() -> (String, String, bool) {
    let cfg = crate::sysbox::read_blob(CLOUD_CFG).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let (mut u, mut m, mut o) = (String::new(), String::new(), false);
    for line in cfg.lines() {
        match line.split_once('=') {
            Some(("use", v)) => o = v.trim() == "on",
            Some(("url", v)) => u = String::from(v.trim()),
            Some(("model", v)) => m = String::from(v.trim()),
            _ => {}
        }
    }
    (u, m, o)
}

pub fn set_key(key: &str) {
    crate::sysbox::make_dir("/etc/maker");
    crate::sysbox::write_text(CLOUD_KEY, key.trim());
}

pub fn has_key() -> bool {
    crate::sysbox::read_blob(CLOUD_KEY).map(|b| !b.is_empty()).unwrap_or(false)
}

// ---- projects ---------------------------------------------------------------

/// Where projects live, as a guest path. With a root image mounted the guest
/// has a writable layer of its own and the project is somewhere a person
/// would look; without one only `/tmp` is writable to a guest, and a program
/// that writes a file beside itself has to be able to.
pub fn projects_dir() -> &'static str {
    if crate::linux::disk::root_mounted() {
        "/projects"
    } else {
        "/tmp/projects"
    }
}

pub fn project_dir(name: &str) -> String {
    format!("{}/{}", projects_dir(), name)
}

/// A name a directory can carry and a shell line can hold unquoted.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 40
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        && !name.starts_with('-')
}

fn write(path: &str, text: &str) -> bool {
    if let Some((dir, _)) = path.rsplit_once('/') {
        mkdirs(dir);
    }
    crate::linux::ns::write_blob(path, Vec::from(text.as_bytes()))
}

fn mkdirs(dir: &str) {
    let mut at = String::new();
    for part in dir.split('/').filter(|p| !p.is_empty()) {
        at.push('/');
        at.push_str(part);
        if !crate::linux::ns::is_dir(&at) {
            crate::linux::ns::make_dir(&at);
        }
    }
}

fn read(path: &str) -> Option<String> {
    crate::linux::ns::read_blob(path)
        .or_else(|| crate::linux::disk::program_bytes(path))
        .map(|b| String::from_utf8_lossy(&b).into_owned())
}

// ---- the guest, as a tool runner -----------------------------------------------

/// Run a shell script in the guest to its end and answer its exit status and
/// what it printed. Every checker and the cloud brain go through here, so
/// "what does running a tool mean" has one answer.
fn guest(script: &str, secs: u64) -> Result<(i32, String), String> {
    if !crate::linux::disk::root_mounted() {
        return Err(String::from(
            "the checkers run in the Linux guest and no root image is mounted -- 'linux mount / 2'",
        ));
    }
    let dir = "/tmp/maker";
    mkdirs(dir);
    let _ = crate::linux::ns::detach("/tmp/maker/rc");
    let _ = crate::linux::ns::detach("/tmp/maker/out");
    let wrapped = format!(
        "#!/bin/bash\n( {}\n) > /tmp/maker/out 2>&1 < /dev/null\necho $? > /tmp/maker/rc\n",
        script
    );
    write("/tmp/maker/run.sh", &wrapped);
    let id = crate::linux::start_job_env_quiet("/usr/bin/bash", &["/tmp/maker/run.sh"], &["HOME=/tmp", "LANG=C.UTF-8"])
        .map_err(String::from)?;
    let deadline = crate::dev::lapic::ticks() + (secs + 10) * crate::TIMER_HZ as u64;
    loop {
        if let Some(r) = crate::job::list().into_iter().find(|r| r.id == id) {
            if r.state == "done" {
                break;
            }
        }
        if crate::dev::lapic::ticks() > deadline {
            let _ = crate::job::kill(id);
            return Err(format!("the check did not finish in {} s", secs + 10));
        }
        if crate::ai::agent::aborted() {
            let _ = crate::job::kill(id);
            return Err(String::from("stopped"));
        }
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 10);
    }
    let rc = read("/tmp/maker/rc").and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(-1);
    let out = read("/tmp/maker/out").unwrap_or_default();
    Ok((rc, out))
}

/// Is a program installed in the guest?
fn has_tool(path: &str) -> bool {
    crate::linux::ns::blob_len(path).is_some()
        || matches!(crate::linux::disk::lookup(path), Some(Some(n)) if !n.is_dir)
}

// ---- checking ------------------------------------------------------------------

/// What a check found. `fatal` problems go back to the model; the rest are
/// notes for the person.
#[derive(Clone)]
pub struct Verdict {
    pub ok: bool,
    /// The tool's own words, trimmed to what the model needs.
    pub problems: Vec<String>,
    /// What running it printed, when it was run.
    pub output: String,
    pub notes: Vec<String>,
}

impl Verdict {
    fn pass(output: String, notes: Vec<String>) -> Verdict {
        Verdict { ok: true, problems: Vec::new(), output, notes }
    }
    fn fail(problems: Vec<String>) -> Verdict {
        Verdict { ok: false, problems, output: String::new(), notes: Vec::new() }
    }
}

/// Keep the end of a long error, which is where Python and gcc put the line
/// that matters, and keep it short enough to fit beside the file.
fn clip_tail(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.len() <= max {
        return String::from(s);
    }
    let mut start = s.len() - max;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    format!("...{}", &s[start..])
}

fn clip_head(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.len() <= max {
        return String::from(s);
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}

pub fn check(t: Target, dir: &str, code: &str) -> Verdict {
    if code.trim().is_empty() {
        return Verdict::fail(alloc::vec![String::from("the file is empty")]);
    }
    match t {
        Target::Web => check_web(dir, code),
        Target::Python => check_python(dir),
        Target::C => check_c(dir),
        Target::Aiksi => check_aiksi(code),
    }
}

/// Seconds a program is given to run in a smoke test.
const RUN_S: u64 = 6;

fn check_python(dir: &str) -> Verdict {
    if !has_tool("/usr/bin/python3") {
        return Verdict::fail(alloc::vec![String::from("there is no python3 in the guest")]);
    }
    // Syntax first, by the parser Python itself uses, so a syntax error is
    // reported as one and not as whatever the run happened to print.
    let syntax = format!(
        "cd {d} && python3 -c 'import ast,sys; ast.parse(open(\"main.py\").read(), \"main.py\")'",
        d = dir
    );
    match guest(&syntax, 30) {
        Err(e) => return Verdict::fail(alloc::vec![e]),
        Ok((0, _)) => {}
        Ok((_, out)) => return Verdict::fail(alloc::vec![clip_tail(&out, 900)]),
    }
    let run = format!("cd {d} && timeout {s} python3 -X utf8 main.py", d = dir, s = RUN_S);
    match guest(&run, RUN_S + 20) {
        Err(e) => Verdict::fail(alloc::vec![e]),
        Ok((0, out)) => Verdict::pass(out, Vec::new()),
        Ok((124, out)) => Verdict::pass(out, alloc::vec![format!("still running after {} s, which is fine for a program that serves or loops", RUN_S)]),
        Ok((_, out)) => {
            // Running out of input is the person's to give, not a bug.
            if out.contains("EOFError") && !out.contains("Error:") {
                return Verdict::pass(out, alloc::vec![String::from("it wants input; run it yourself to type some")]);
            }
            if out.trim_end().ends_with("EOFError") || out.contains("EOFError: EOF when reading a line") {
                return Verdict::pass(out, alloc::vec![String::from("it wants input; run it yourself to type some")]);
            }
            if out.contains("Traceback") {
                Verdict::fail(alloc::vec![clip_tail(&out, 900)])
            } else {
                // A nonzero exit that is the program's own choice: usage
                // messages are the common case.
                Verdict::pass(out, alloc::vec![String::from("it exited with a nonzero status")])
            }
        }
    }
}

fn check_c(dir: &str) -> Verdict {
    let cc = ["/usr/bin/gcc", "/usr/bin/cc", "/usr/bin/tcc", "/usr/bin/clang"].into_iter().find(|p| has_tool(p));
    let Some(cc) = cc else {
        return Verdict::fail(alloc::vec![String::from(
            "there is no C compiler in the guest -- 'orange install gcc' puts one there",
        )]);
    };
    // The binary goes to /tmp: a project directory need not be writable by
    // the compiler's own idea of the umask, and a stale binary beside the
    // source is a binary nobody rebuilt.
    let build = format!("cd {d} && {cc} -std=c11 -Wall -O1 -o /tmp/maker/a.out main.c -lm", d = dir, cc = cc);
    match guest(&build, 60) {
        Err(e) => return Verdict::fail(alloc::vec![e]),
        Ok((0, out)) if out.contains("warning:") => return Verdict::fail(alloc::vec![clip_head(&out, 900)]),
        Ok((0, _)) => {}
        Ok((_, out)) => return Verdict::fail(alloc::vec![clip_head(&out, 900)]),
    }
    match guest(&format!("cd {} && timeout {} /tmp/maker/a.out", dir, RUN_S), RUN_S + 20) {
        Err(e) => Verdict::fail(alloc::vec![e]),
        Ok((0, out)) => Verdict::pass(out, Vec::new()),
        Ok((124, out)) => Verdict::pass(out, alloc::vec![format!("still running after {} s", RUN_S)]),
        Ok((rc, out)) if rc >= 128 => Verdict::fail(alloc::vec![format!(
            "the program crashed (signal {}). It printed:\n{}",
            rc - 128,
            clip_tail(&out, 600)
        )]),
        Ok((_, out)) => Verdict::pass(out, alloc::vec![String::from("it exited with a nonzero status")]),
    }
}

fn check_aiksi(code: &str) -> Verdict {
    let prog = match crate::aiksi::parse_source(code) {
        Ok(p) => p,
        Err(e) => return Verdict::fail(alloc::vec![e]),
    };
    let mut it = crate::aiksi::Interp::sandboxed("/tmp/maker").with_step_budget(2_000_000);
    crate::gfx::console::begin_capture();
    let r = it.run(&prog);
    let out = crate::gfx::console::end_capture().unwrap_or_default();
    match r {
        Ok(_) => Verdict::pass(out, Vec::new()),
        Err(e) => Verdict::fail(alloc::vec![e]),
    }
}

/// Ids a script looks up, from the forms a model writes: `getElementById`,
/// and a `querySelector` whose selector is a single `#id`.
fn ids_used(js: &str) -> Vec<String> {
    let mut out = Vec::new();
    for pat in ["getElementById(", "querySelector("] {
        let mut rest = js;
        while let Some(at) = rest.find(pat) {
            rest = &rest[at + pat.len()..];
            let r = rest.trim_start();
            let Some(q) = r.chars().next().filter(|c| *c == '\'' || *c == '"' || *c == '`') else { continue };
            let body = &r[1..];
            let Some(end) = body.find(q) else { continue };
            let lit = &body[..end];
            let id = if pat == "querySelector(" {
                match lit.strip_prefix('#') {
                    Some(id) if id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') => id,
                    _ => continue,
                }
            } else {
                lit
            };
            if !id.is_empty() && !id.contains('$') && !out.iter().any(|x: &String| x == id) {
                out.push(String::from(id));
            }
        }
    }
    out
}

/// Ids the page declares, from `id="x"` and `id='x'`.
fn ids_declared(html: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(at) = rest.find("id=") {
        let before = rest[..at].chars().last();
        rest = &rest[at + 3..];
        // `id=` inside a longer attribute name, `data-id=` or `grid=`.
        if before.map(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_').unwrap_or(false) {
            continue;
        }
        let Some(q) = rest.chars().next().filter(|c| *c == '"' || *c == '\'') else { continue };
        if let Some(end) = rest[1..].find(q) {
            out.push(String::from(&rest[1..1 + end]));
        }
    }
    out
}

/// Inline scripts, in order, with the line each starts on.
fn scripts(html: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let lower = html.to_ascii_lowercase();
    let mut from = 0;
    while let Some(open) = lower[from..].find("<script") {
        let open = from + open;
        let Some(gt) = lower[open..].find('>') else { break };
        let tag = &lower[open..open + gt];
        let body_at = open + gt + 1;
        let Some(close) = lower[body_at..].find("</script>") else { break };
        if !tag.contains("src=") {
            let line = html[..body_at].matches('\n').count() + 1;
            out.push((line, String::from(&html[body_at..body_at + close])));
        }
        from = body_at + close + 9;
    }
    out
}

fn check_web(dir: &str, html: &str) -> Verdict {
    let mut problems = Vec::new();
    let lower = html.to_ascii_lowercase();
    if !lower.contains("<body") && !lower.contains("<div") && !lower.contains("<main") {
        problems.push(String::from("the page has no body: there is nothing for the browser to show"));
    }
    if lower.contains("<script src=\"http") || lower.contains("<link rel=\"stylesheet\" href=\"http") {
        problems.push(String::from("the page loads something from the internet; everything must be in this one file"));
    }
    let js: Vec<(usize, String)> = scripts(html);
    let all_js: String = js.iter().map(|(_, s)| s.as_str()).collect::<Vec<_>>().join("\n");
    let declared = ids_declared(html);
    // A script may also create an element and give it an id; those are not
    // missing, they are made at run time.
    for id in ids_used(&all_js) {
        let made = all_js.contains(&format!(".id = \"{}\"", id)) || all_js.contains(&format!(".id = '{}'", id));
        if !declared.contains(&id) && !made {
            problems.push(format!(
                "the script looks up the element with id \"{}\" but no element in the HTML has id=\"{}\"",
                id, id
            ));
        }
    }
    // Every script must compile, in a real engine.
    let mut notes = Vec::new();
    let engine = ["/usr/bin/qjs", "/usr/local/bin/qjs", "/usr/bin/node", "/linux/bin/quickjs"].into_iter().find(|p| has_tool(p));
    if js.is_empty() {
        notes.push(String::from("the page has no script"));
    } else if let Some(engine) = engine {
        for (k, (line, src)) in js.iter().enumerate() {
            let path = format!("/tmp/maker/s{}.js", k);
            write(&path, src);
            // `new Function` compiles the text without running it: the page's
            // own DOM does not exist here, and running it would report
            // `document is not defined` about a perfectly good script.
            let script = if engine.ends_with("node") {
                format!(
                    "{e} -e \"const fs=require('fs');try{{new Function(fs.readFileSync('{p}','utf8'))}}catch(e){{console.log(e.name+': '+e.message);process.exit(1)}}\"",
                    e = engine,
                    p = path
                )
            } else {
                // quickjs-ng: with --std, `std` is a global of a classic script.
                write("/tmp/maker/probe.js", &format!(
                    "try {{ new Function(std.loadFile('{p}')); }} catch (e) {{ std.out.puts(e.name + ': ' + e.message + (e.lineNumber ? ' (line ' + (e.lineNumber + {base}) + ' of the page)' : '') + '\\n'); std.exit(1); }}\n",
                    p = path,
                    base = line.saturating_sub(1),
                ));
                format!("{} --std -C /tmp/maker/probe.js", engine)
            };
            match guest(&script, 20) {
                Err(e) => problems.push(e),
                Ok((0, _)) => {}
                Ok((_, out)) => {
                    let first = out.lines().find(|l| l.contains("Error")).unwrap_or(out.trim());
                    problems.push(format!("script {} does not compile: {}", k + 1, clip_head(first, 300)));
                }
            }
        }
    } else {
        notes.push(String::from("no JavaScript engine in the guest, so the script was not compiled -- 'orange install quickjs-ng'"));
    }
    if problems.is_empty() {
        Verdict::pass(String::new(), notes)
    } else {
        Verdict { ok: false, problems, output: String::new(), notes }
    }
}

// ---- writing -------------------------------------------------------------------

const SYSTEM: &str = "You are an expert programmer. You write complete, working programs that run on the \
first try. You reply with exactly one fenced code block containing the whole file, and nothing else.";

/// The code in a reply: the first fenced block, or the whole reply when the
/// model wrote none. A fence the model opened and never closed -- it ran out
/// of room -- is still the file, and the checker will say it is cut off.
pub fn extract(reply: &str) -> String {
    let Some(open) = reply.find("```") else { return String::from(reply.trim()) + "\n" };
    let after = &reply[open + 3..];
    // The rest of the opening line is the language tag.
    let body = match after.find('\n') {
        Some(nl) => &after[nl + 1..],
        None => "",
    };
    let body = match body.find("\n```") {
        Some(end) => &body[..end + 1],
        None => body.strip_suffix("```").unwrap_or(body),
    };
    String::from(body.trim_end()) + "\n"
}

/// The line number a tool's message names: `line 5:`, `line 5,` or
/// `main.c:5:`.
fn line_named(msg: &str) -> Option<usize> {
    let digits = |s: &str| s.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<usize>().ok();
    if let Some(at) = msg.rfind("line ") {
        if let Some(n) = digits(&msg[at + 5..]) {
            return Some(n);
        }
    }
    for marker in ["main.c:", "main.py\", line ", "main.py:"] {
        if let Some(at) = msg.find(marker) {
            if let Some(n) = digits(&msg[at + marker.len()..]) {
                return Some(n);
            }
        }
    }
    None
}

fn first_prompt(t: Target, name: &str, desc: &str) -> String {
    format!(
        "Write {lang} called \"{name}\".\n\nWhat it should do: {desc}\n\nRules:\n{rules}\n\nReply with the whole file `{file}` in one ```{fence} code block.",
        lang = t.language(),
        name = name,
        desc = desc,
        rules = t.rules(),
        file = t.file(),
        fence = t.fence(),
    )
}

fn repair_prompt(t: Target, name: &str, desc: &str, code: &str, problems: &[String]) -> String {
    let mut p = format!(
        "This is {lang} called \"{name}\". What it should do: {desc}\n\n```{fence}\n{code}```\n\nChecking it found:\n",
        lang = t.language(),
        name = name,
        desc = desc,
        fence = t.fence(),
        code = code,
    );
    for pr in problems {
        p.push_str("- ");
        p.push_str(pr);
        p.push('\n');
        // The line the tool named, quoted, so the model looks at it rather
        // than at the whole file.
        if let Some(n) = line_named(pr) {
            if let Some(text) = code.lines().nth(n.saturating_sub(1)) {
                p.push_str(&format!("  (line {} is: {})\n", n, text.trim()));
            }
        }
    }
    p.push_str(&format!(
        "\nFix every problem. Keep everything that already works.\nRules:\n{}\n\nReply with the whole corrected file `{}` in one ```{} code block.",
        t.rules(),
        t.file(),
        t.fence()
    ));
    p
}

/// One ChatML turn, thinking closed for a model that has it: a program
/// written after a page of reasoning is a program with half the room.
fn chat(system: &str, user: &str) -> String {
    let mut p = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", system, user);
    if crate::ai::has_think_token() {
        p.push_str("<think>\n\n</think>\n\n");
    }
    p
}

/// Ask the brain in force. Answers the reply text.
fn ask(brain: &Brain, user: &str) -> Result<String, String> {
    ask_at(brain, user, 0.2)
}

/// `temperature` is raised when the model is stuck: at 0.2 the same prompt
/// gives back the same file, and the first driven run sent one file back five
/// times and got it five times, byte for byte.
fn ask_at(brain: &Brain, user: &str, temperature: f32) -> Result<String, String> {
    match brain {
        Brain::Local => ask_local(user, temperature),
        Brain::Cloud { url, model } => ask_cloud(url, model, user),
    }
}

fn ask_local(user: &str, temperature: f32) -> Result<String, String> {
    let prompt = chat(SYSTEM, user);
    let room = crate::ai::with_engine(|e| {
        let used = e.tok.encode(&prompt, false, false).len();
        e.model.cfg.seq_len.saturating_sub(used + 8)
    })
    .ok_or_else(crate::ai::engine_refusal)?;
    if room < 200 {
        return Err(format!(
            "the model's context is full before it can write: {} tokens left. A checkpoint with a longer context ('--seq 4096') is what a program needs",
            room
        ));
    }
    let opts = crate::ai::GenOpts {
        steps: room.min(3000),
        // Low and not zero: zero repeats itself into a loop on a long file,
        // and code wants the likeliest token far more than prose does.
        temperature,
        topp: 0.9,
        resume: false,
        yielding: false,
        bos: false,
        echo_prompt: false,
        think: false,
        // Code repeats tokens on purpose -- every closing brace -- and a
        // penalty tuned for prose makes it avoid the ones it needs.
        repeat_penalty: 1.0,
        ..Default::default()
    };
    crate::ai::set_plain(false);
    crate::ai::echo_begin_quiet(64 * 1024);
    let pace = crate::brain::pace();
    crate::brain::set_pace(pace.min(20));
    crate::ai::generate(&prompt, &opts);
    crate::brain::set_pace(pace);
    let text = crate::ai::echo_end_quiet().unwrap_or_default();
    if text.trim().is_empty() {
        return Err(String::from("the model wrote nothing"));
    }
    Ok(text)
}

/// The cloud brain, as a guest program: Python has TLS with POST and the CA
/// bundle, and this kernel's own client deliberately does not POST.
const CLOUD_PY: &str = r#"import json, sys, urllib.request
url, model, keyf, sysf, userf, outf = sys.argv[1:7]
key = open(keyf).read().strip()
body = {"model": model, "temperature": 0.2, "messages": [
    {"role": "system", "content": open(sysf).read()},
    {"role": "user", "content": open(userf).read()}]}
req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
try:
    with urllib.request.urlopen(req, timeout=170) as r:
        reply = json.load(r)["choices"][0]["message"]["content"]
except urllib.error.HTTPError as e:
    print("the cloud answered", e.code, e.read()[:300].decode("utf-8", "replace")); sys.exit(2)
except Exception as e:
    print("could not reach the cloud:", e); sys.exit(2)
open(outf, "w").write(reply)
"#;

fn ask_cloud(url: &str, model: &str, user: &str) -> Result<String, String> {
    let key = read(CLOUD_KEY).unwrap_or_default();
    if key.trim().is_empty() {
        return Err(String::from("the cloud brain has no key -- 'maker cloud key <key>'"));
    }
    // The key travels in a file the guest reads, never on a command line a
    // trace would print.
    write("/tmp/maker/key", &key);
    write("/tmp/maker/cloud.py", CLOUD_PY);
    write("/tmp/maker/system.txt", SYSTEM);
    write("/tmp/maker/user.txt", user);
    let _ = crate::linux::ns::detach("/tmp/maker/reply.txt");
    let r = guest(
        &format!(
            "python3 /tmp/maker/cloud.py '{}' '{}' /tmp/maker/key /tmp/maker/system.txt /tmp/maker/user.txt /tmp/maker/reply.txt",
            url.replace('\'', ""),
            model.replace('\'', "")
        ),
        180,
    );
    let _ = crate::linux::ns::detach("/tmp/maker/key");
    match r? {
        (0, _) => read("/tmp/maker/reply.txt").ok_or_else(|| String::from("the cloud answered nothing")),
        (_, out) => Err(clip_head(&out, 400)),
    }
}

// ---- the contract ----------------------------------------------------------------

fn contract_prompt(t: Target, name: &str, desc: &str) -> String {
    format!(
        "Someone will write {lang} called \"{name}\" that does this: {desc}\n\n\
         It is run with no input. List two to four short, exact pieces of text that its output \
         MUST contain if it works correctly -- a specific number, word or line it has to print. \
         Only things that are certain from the description. One per line, each starting with \"- \". \
         Nothing else.",
        lang = t.language(),
        name = name,
        desc = desc,
    )
}

/// The promises in a reply: `- text` lines, quotes and backticks taken off,
/// at most four, each short. A promise the model hedged ("maybe", "e.g.")
/// is not certain and is dropped.
pub fn parse_contract(reply: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in reply.lines() {
        let l = line.trim();
        // `- x`, `* x`, `• x`, `1. x`, `1) x`: a model lists in whichever.
        let numbered = {
            let d = l.chars().take_while(|c| c.is_ascii_digit()).count();
            if d > 0 && d < 3 { l[d..].strip_prefix(". ").or_else(|| l[d..].strip_prefix(") ")) } else { None }
        };
        let Some(rest) = l.strip_prefix("- ").or_else(|| l.strip_prefix("* ")).or_else(|| l.strip_prefix("\u{2022} ")).or(numbered) else { continue };
        let s = rest.trim().trim_matches(|c| c == '`' || c == '"' || c == '\'').trim();
        let lower = s.to_ascii_lowercase();
        if s.is_empty() || s.len() > 40 || lower.contains("e.g") || lower.contains("maybe") || lower.contains("for example") {
            continue;
        }
        if !out.iter().any(|x| x == s) {
            out.push(String::from(s));
        }
        if out.len() == 4 {
            break;
        }
    }
    out
}

/// What a run that passed the tools still owes: something printed, and every
/// promise in it. A program that asked for input is excused -- what it prints
/// depends on what it is given.
fn held_to(v: &Verdict, promises: &[String]) -> Vec<String> {
    if v.notes.iter().any(|n| n.contains("wants input") || n.contains("still running")) {
        return Vec::new();
    }
    let out = v.output.trim();
    let lines = out.lines().filter(|l| !l.trim().is_empty()).count();
    let mut missing = Vec::new();
    // A heading and nothing under it is the same failure as nothing at all:
    // the results the heading announced never came.
    if lines == 1 && out.trim_end().ends_with(':') {
        missing.push(format!("the program printed only the heading {:?} and no results under it", out.trim()));
        return missing;
    }
    if lines == 0 {
        missing.push(String::from("the program ran and printed nothing; it must print its results"));
        return missing;
    }
    let shown = clip_head(out, 400);
    for s in promises {
        if !out.contains(s.as_str()) {
            missing.push(format!("the output should contain {:?} but does not. It printed:\n{}", s, shown));
        }
    }
    missing
}

// ---- the loop ----------------------------------------------------------------------

/// How many times a program is sent back. Four rewrites of a file that keeps
/// failing is the point where the problem is the request or the model, and
/// saying so beats a fifth.
pub const MAX_ROUNDS: usize = 5;

#[derive(Clone)]
pub struct Progress {
    pub name: String,
    pub target: Target,
    pub brain: String,
    pub round: usize,
    pub doing: String,
    pub last: Vec<String>,
    pub running: bool,
    pub ok: bool,
}

static PROGRESS: crate::sync::Spin<Option<Progress>> = crate::sync::Spin::new(None);

pub fn progress() -> Option<Progress> {
    PROGRESS.lock().clone()
}

fn say(p: &mut Progress, doing: String) {
    p.doing = doing;
    *PROGRESS.lock() = Some(p.clone());
    crate::gfx::render::invalidate();
}

pub struct Outcome {
    pub ok: bool,
    pub rounds: usize,
    pub path: String,
    pub summary: String,
}

/// Write a program and keep rewriting it until the tools accept it or the
/// rounds run out. Leaves the project directory either way, with the history
/// of every round in `log.txt`, because a failed attempt is also what somebody
/// reads to see what the model misunderstood.
pub fn make(t: Target, name: &str, desc: &str) -> Outcome {
    let dir = project_dir(name);
    let path = format!("{}/{}", dir, t.file());
    let brain = brain();
    let mut p = Progress {
        name: String::from(name),
        target: t,
        brain: brain.describe(),
        round: 0,
        doing: String::new(),
        last: Vec::new(),
        running: true,
        ok: false,
    };
    mkdirs(&dir);
    write(&format!("{}/README.md", dir), &format!("# {}\n\n{}\n\nWritten by {} as {}.\n", name, desc, brain.describe(), t.language()));
    let mut log = format!("{} -- {} -- {}\n{}\n", name, t.name(), brain.describe(), desc);
    let mut code = String::new();
    let mut problems: Vec<String> = Vec::new();
    let mut verdict: Option<Verdict> = None;
    // The contract: what the output must contain if the program does what
    // was asked, decided *before* any code exists. Without it every check here
    // is satisfied by a program that runs and prints a heading -- which is
    // exactly what the first driven runs produced and called a pass.
    let promises = if t == Target::Web {
        Vec::new()
    } else {
        say(&mut p, String::from("deciding what it must print"));
        match ask(&brain, &contract_prompt(t, name, desc)) {
            Ok(r) => {
                let got = parse_contract(&r);
                if got.is_empty() {
                    // Kept, because a contract that silently came out empty
                    // is a run held to nothing -- which is how the second
                    // driven run "passed" a program that printed a heading.
                    log.push_str(&format!("the contract reply held no promises:\n{}\n", clip_head(&r, 400)));
                }
                got
            }
            Err(e) => {
                log.push_str(&format!("no contract: {}\n", e));
                Vec::new()
            }
        }
    };
    if !promises.is_empty() {
        log.push_str("its output must contain:\n");
        for s in &promises {
            log.push_str(&format!("  {:?}\n", s));
        }
    }
    let mut stuck = false;
    for round in 1..=MAX_ROUNDS {
        if crate::ai::agent::aborted() {
            log.push_str("stopped\n");
            break;
        }
        p.round = round;
        say(&mut p, if round == 1 { String::from("writing") } else { format!("rewriting, round {}", round) });
        let mut prompt = if round == 1 { first_prompt(t, name, desc) } else { repair_prompt(t, name, desc, &code, &problems) };
        if stuck {
            prompt.push_str("\n\nYour last answer was exactly the same file, with the same problem. Do not repeat it: change the line the problem names, or write that part a different way.");
        }
        let reply = match ask_at(&brain, &prompt, if stuck { 0.7 } else { 0.2 }) {
            Ok(r) => r,
            Err(e) => {
                log.push_str(&format!("round {}: {}\n", round, e));
                p.last = alloc::vec![e];
                break;
            }
        };
        let next = extract(&reply);
        // A rewrite that came back empty or cut to a stub is worse than the
        // draft it replaces; keep the draft and say what happened.
        if round > 1 && next.trim().len() < code.trim().len() / 4 {
            log.push_str(&format!("round {}: the rewrite came back {} bytes against {}; kept the draft\n", round, next.len(), code.len()));
            continue;
        }
        if round > 1 && next.trim() == code.trim() {
            log.push_str(&format!("round {}: the same file again\n", round));
            stuck = true;
            continue;
        }
        stuck = false;
        code = next;
        write(&path, &code);
        say(&mut p, String::from("checking"));
        let mut v = check(t, &dir, &code);
        if v.ok && t != Target::Web {
            let mut missing = held_to(&v, &promises);
            // The commonest reason a program prints nothing here: it waited
            // on input that a run with none never gives it.
            if missing.iter().any(|m| m.contains("printed nothing") || m.contains("only the heading"))
                && (code.contains("stdin") || code.contains("input("))
            {
                missing = alloc::vec![String::from(
                    "the program reads its text from standard input, which is empty when it is run, so it printed nothing. \
                     Put the text it works on inside the program as a string, and print the results.",
                )];
            }
            if !missing.is_empty() {
                v.ok = false;
                v.problems = missing;
            }
        }
        log.push_str(&format!("round {}: {} bytes, {}\n", round, code.len(), if v.ok { "accepted" } else { "sent back" }));
        for pr in &v.problems {
            log.push_str(&format!("  - {}\n", pr));
        }
        p.last = v.problems.clone();
        problems = v.problems.clone();
        let ok = v.ok;
        verdict = Some(v);
        if ok {
            break;
        }
    }
    let ok = verdict.as_ref().map(|v| v.ok).unwrap_or(false);
    if let Some(v) = &verdict {
        if !v.output.trim().is_empty() {
            write(&format!("{}/run-output.txt", dir), &v.output);
        }
        for n in &v.notes {
            log.push_str(&format!("note: {}\n", n));
        }
    }
    write(&format!("{}/log.txt", dir), &log);
    p.running = false;
    p.ok = ok;
    say(&mut p, if ok { String::from("done") } else { String::from("gave up") });
    let summary = match (&verdict, ok) {
        (Some(v), true) => {
            let mut s = format!("{} passed its checks in {} round(s)", t.file(), p.round);
            for n in &v.notes {
                s.push_str("; ");
                s.push_str(n);
            }
            s
        }
        (Some(v), false) => format!(
            "not accepted after {} round(s): {}",
            p.round,
            v.problems.first().map(|s| clip_head(s, 200)).unwrap_or_default()
        ),
        (None, _) => format!("nothing written: {}", p.last.first().cloned().unwrap_or_default()),
    };
    Outcome { ok, rounds: p.round, path, summary }
}

/// Every project, with its target read from the file it holds.
pub fn list() -> Vec<(String, Option<Target>)> {
    let dir = projects_dir();
    crate::linux::ns::listing(dir)
        .into_iter()
        .filter(|(_, is_dir, _)| *is_dir)
        .map(|(n, _, _)| {
            let t = Target::ALL.into_iter().find(|t| crate::linux::ns::blob_len(&format!("{}/{}/{}", dir, n, t.file())).is_some());
            (n, t)
        })
        .collect()
}

/// How a person runs what was made.
pub fn how_to_run(t: Target, name: &str) -> String {
    let d = project_dir(name);
    match t {
        Target::Web => format!("open {}/index.html in a browser (it is one self-contained file)", d),
        Target::Python => format!("linux run /usr/bin/python3 {}/main.py", d),
        Target::C => format!("maker run {} (compiles and runs it)", name),
        Target::Aiksi => format!("run {}/main.ai&xi", d),
    }
}

// ---- checked without a model --------------------------------------------------------

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |what: &str, good: bool| {
        if !good {
            kprintln!("    FAIL: {}", what);
            ok = false;
        }
    };
    claim("a fenced reply yields its body", extract("Here:\n```python\nprint(1)\n```\nDone.") == "print(1)\n");
    claim("an unclosed fence is still the file", extract("```c\nint main(void){\n") == "int main(void){\n");
    claim("a reply with no fence is taken whole", extract("print(2)\n") == "print(2)\n");
    let page = "<div id=\"out\"></div><button id='go'>Go</button>\n<script>\nlet o = document.getElementById(\"out\");\ndocument.querySelector('#go').onclick = f;\ndocument.getElementById('missing');\n</script>";
    let s = scripts(page);
    claim("inline scripts are found with their line", s.len() == 1 && s[0].0 == 2);
    let used = ids_used(&s[0].1);
    claim("ids a script looks up are found, by both lookups", used.len() == 3 && used.iter().any(|u| u == "go"));
    let declared = ids_declared(page);
    claim("ids the page declares are found, either quote", declared.len() == 2);
    claim("data-id is not an id", ids_declared("<p data-id=\"x\">").is_empty());
    claim("a missing element is caught without running anything", {
        let v = check_web("/tmp", page);
        !v.ok && v.problems.iter().any(|p| p.contains("\"missing\""))
    });
    claim("a contract is read off '- ' lines, hedges dropped", parse_contract("Sure:\n- 97\n- `FizzBuzz`\n- e.g. 3\n- 2 3 5 7\nThat's all.") == alloc::vec![String::from("97"), String::from("FizzBuzz"), String::from("2 3 5 7")]);
    claim("a numbered contract is read too", parse_contract("1. 97\n2) 2 3 5 7") == alloc::vec![String::from("97"), String::from("2 3 5 7")]);
    claim("a heading with nothing under it is sent back", !held_to(&Verdict::pass(String::from("Top 5 words:\n"), Vec::new()), &[]).is_empty());
    claim("a run that printed nothing owes its results", !held_to(&Verdict::pass(String::new(), Vec::new()), &[]).is_empty());
    claim("and one missing a promise is sent back naming it", held_to(&Verdict::pass(String::from("Top words:\n"), Vec::new()), &[String::from("the: 3")]).iter().any(|m| m.contains("the: 3")));
    claim("the line a tool names is found", line_named("main.c:12:5: error: x") == Some(12) && line_named("File \"main.py\", line 7, in f") == Some(7) && line_named("line 3: expected ')'") == Some(3));
    claim("names that reach a shell line are refused", !valid_name("a b") && !valid_name("x;rm") && !valid_name("-x") && valid_name("todo-app_2"));
    claim("every target has a file and a fence the prompt names", Target::ALL.iter().all(|t| first_prompt(*t, "n", "d").contains(t.file())));
    claim("a repair prompt carries the tool's own words", repair_prompt(Target::Python, "n", "d", "x\n", &[String::from("NameError: q")]).contains("NameError: q"));
    claim("an Aiksi program that runs is accepted", check_aiksi("fn main() { println(1 + 1) }\nmain()").ok);
    claim("and so is one written like Python", check_aiksi("def main():\n    xs = []\n    for i in range(3):\n        xs.append(i * 2)\n    print(f\"{len(xs)} items\")\n\nmain()\n").ok);
    claim("and one that does not parse is sent back", !check_aiksi("fn main( {").ok);
    ok
}
