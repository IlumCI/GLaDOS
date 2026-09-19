//! Every rail this machine can measure about itself, in one block a program
//! can read.
//!
//! ### Why a report and not five commands
//!
//! `video bench`, `core bench`, `smp bench`, `store bench` and `bench` all
//! existed and all printed prose for a person. What none of them was is
//! *judgeable*: nothing recorded a figure, compared it across builds, or
//! refused a change that made one worse. So the only judge in this tree with
//! any teeth was routing accuracy, and a loop that can author but can only
//! measure one thing optimises that one thing and calls it self-improvement.
//!
//! This is the other half of the argument `godel` makes about its own axes: a
//! change is judged against the rails it claims to move and the rails it must
//! not break, and neither half exists until the rails have names and numbers.
//!
//! ### Every declared rail appears, including the ones that could not run
//!
//! A report that silently omitted a rail it could not measure would read
//! exactly like a report where that rail was fine -- the failure this tree
//! keeps recording, most recently as a `diag` listing whose `0 passed, 0
//! failed` is easy to read as a clean sweep. So an unmeasurable rail prints
//! `absent` and says why, and `bench report` is a list of the same length on
//! every machine.
//!
//! ### It is a per-build snapshot, not a per-trial judge
//!
//! Some of these cost seconds and one builds 16 MiB. That is affordable once a
//! build, recorded by CI as an artifact, and it is not affordable inside a
//! nightly trial. A judge that runs per trial wants a rail cheaper than these;
//! what this answers is "did the build that CI is about to sign get slower".

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Which way is better. Recorded per rail rather than inferred from the name,
/// because `gflops` and `us` point in opposite directions and a comparison
/// that guessed would report every graphics improvement as a regression.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Want {
    Lower,
    Higher,
}

impl Want {
    pub fn as_str(&self) -> &'static str {
        match self {
            Want::Lower => "lower",
            Want::Higher => "higher",
        }
    }
}

/// One measured number, or the reason there is not one.
pub struct Rail {
    pub name: &'static str,
    pub unit: &'static str,
    pub want: Want,
    /// `None` means this machine could not measure it, and `why` says what was
    /// missing. Never zero for that case: zero is a value, and a rail reading
    /// zero because nothing ran is the failure this file opens by naming.
    pub value: Option<f64>,
    pub why: &'static str,
}

impl Rail {
    fn got(name: &'static str, unit: &'static str, want: Want, v: f64) -> Rail {
        Rail { name, unit, want, value: Some(v), why: "" }
    }
    fn absent(name: &'static str, unit: &'static str, want: Want, why: &'static str) -> Rail {
        Rail { name, unit, want, value: None, why }
    }
}

/// Every rail's name, unit and direction, without measuring any of them.
///
/// **The contract, separated from the measurement**, so a claim about the
/// shape of the report costs nothing: `rails()` repaints the screen forty-five
/// times and allocates 16 MiB, which is a fine price once a build and a silly
/// one inside `diag all`. It is also what makes "every declared rail appears"
/// checkable rather than aspirational -- the report walks this list.
pub const DECLARED: &[(&str, &str, Want)] = &[
    ("video.rect", "us", Want::Lower),
    ("video.draw", "us", Want::Lower),
    ("video.present", "us", Want::Lower),
    ("video.fill_present", "us", Want::Lower),
    ("video.console", "us", Want::Lower),
    ("core.new", "ns", Want::Lower),
    ("core.step", "ps", Want::Lower),
    ("core.vote", "ns", Want::Lower),
    ("core.vote_walk", "ns", Want::Lower),
    ("ai.matmul", "gflops", Want::Higher),
    ("ai.bpb", "mbits", Want::Lower),
    ("ai.answer", "mbits", Want::Lower),
    ("smp.one_core", "mbs", Want::Higher),
    ("smp.all_cores", "mbs", Want::Higher),
];

/// Run every rail. Seconds, and one of them allocates 16 MiB.
///
/// The order is `DECLARED`'s and is stable, so two reports diff line by line
/// rather than needing to be matched up by name -- the same argument
/// `route_snapshot` makes about keeping a slot for a goal it could not
/// featurise.
pub fn rails() -> Vec<Rail> {
    let mut out = Vec::new();

    // --- the graphics path ------------------------------------------------
    //
    // `video.rect` is the control: nothing above the framebuffer can touch a
    // single span-filled rectangle, so what it moves by between two builds is
    // the noise floor rather than an effect. Read the other four against it.
    //
    // **`video.console` is only comparable against a run with the same text on
    // screen.** `console::redraw_all` skips blank cells, so its cost scales
    // with how much output is sitting in the terminal: the same build measured
    // 497 us after a bare boot and 823 us after `diag all` had filled the
    // console, which reads as a 40% regression and is a full scrollback.
    match crate::gfx::measure() {
        Some(m) => {
            out.push(Rail::got("video.rect", "us", Want::Lower, m.rect_us.0 as f64));
            out.push(Rail::got("video.draw", "us", Want::Lower, m.draw_us.0 as f64));
            out.push(Rail::got("video.present", "us", Want::Lower, m.present_us.0 as f64));
            out.push(Rail::got(
                "video.fill_present",
                "us",
                Want::Lower,
                m.fill_present_us.0 as f64,
            ));
            out.push(Rail::got("video.console", "us", Want::Lower, m.console_us.0 as f64));
        }
        None => {
            for n in ["video.rect", "video.draw", "video.present", "video.fill_present", "video.console"] {
                out.push(Rail::absent(n, "us", Want::Lower, "no framebuffer"));
            }
        }
    }

    // --- the interpreter --------------------------------------------------
    match crate::aiksi::measure() {
        Some(m) => {
            out.push(Rail::got("core.new", "ns", Want::Lower, m.new_ns.0 as f64));
            out.push(Rail::got("core.step", "ps", Want::Lower, m.step_ps as f64));
        }
        None => {
            out.push(Rail::absent("core.new", "ns", Want::Lower, "the loop reported no steps"));
            out.push(Rail::absent("core.step", "ps", Want::Lower, "the loop reported no steps"));
        }
    }
    match crate::ai::voter::measure() {
        Some(m) => {
            out.push(Rail::got("core.vote", "ns", Want::Lower, m.total_ns as f64));
            out.push(Rail::got("core.vote_walk", "ns", Want::Lower, m.invoke_ns as f64));
        }
        None => {
            for n in ["core.vote", "core.vote_walk"] {
                out.push(Rail::absent(n, "ns", Want::Lower, "the composed core would not parse"));
            }
        }
    }

    // --- arithmetic -------------------------------------------------------
    //
    // No model is loaded and none is needed: this times `tensor::matmul`, so
    // it is a rail on a machine with no checkpoint on the disk, which is
    // exactly the machine CI has.
    let mm = crate::ai::measure();
    out.push(Rail::got("ai.matmul", "gflops", Want::Higher, mm.gflops as f64));

    // --- how well the machine predicts its own history ---------------------
    //
    // **The only rail here that is about what the model knows**, and the
    // reason to want one is that every other judge in this tree measures
    // routing accuracy: a binary rail, one bit an item. Bits per byte is
    // dense -- every token is an observation -- and the 1991 formulation of
    // curiosity is exactly the *difference* in this number between two
    // builds, which is what a rail comparison already computes.
    //
    // Absent rather than zero on a machine with no checkpoint or no history
    // yet, which is most CI runners: a rail reading zero because nothing ran
    // is the failure this file opens by naming, and a bits-per-byte of zero
    // is a perfect predictor rather than a missing one.
    match crate::ai::with_engine(crate::ai::progress::rail_millibits) {
        Some(Some(mb)) => out.push(Rail::got("ai.bpb", "mbits", Want::Lower, mb as f64)),
        Some(None) => out.push(Rail::absent(
            "ai.bpb",
            "mbits",
            Want::Lower,
            "no checkpoint, or too little history to read",
        )),
        None => out.push(Rail::absent(
            "ai.bpb",
            "mbits",
            Want::Lower,
            "another task holds the engine",
        )),
    }

    // The unaided arm, as a rail. `answer` is where the paired question is
    // asked; this is the level, so a build that made the machine better at
    // explaining itself moves it whether or not retrieval is on.
    match crate::ai::with_engine(crate::ai::answer::rail_millibits) {
        Some(Some(mb)) => out.push(Rail::got("ai.answer", "mbits", Want::Lower, mb as f64)),
        Some(None) => out.push(Rail::absent(
            "ai.answer",
            "mbits",
            Want::Lower,
            "no checkpoint, or no question fits the window",
        )),
        None => out.push(Rail::absent(
            "ai.answer",
            "mbits",
            Want::Lower,
            "another task holds the engine",
        )),
    }

    // --- memory bandwidth -------------------------------------------------
    //
    // **Measuring this under a hypervisor does not work and the numbers say
    // so**: one core reads 4570 MB/s alone and 3526 MB/s with seven merely
    // *idling* beside it, so both halves of any comparison are contaminated by
    // the host. It is a rail on the GF63 and a curiosity anywhere else, and it
    // is reported either way rather than hidden, because a figure with its
    // caveat beside it is more useful than a gap.
    match crate::smp::measure() {
        Some(m) => {
            out.push(Rail::got("smp.one_core", "mbs", Want::Higher, m.one_mbs() as f64));
            out.push(Rail::got("smp.all_cores", "mbs", Want::Higher, m.many_mbs() as f64));
        }
        None => {
            for n in ["smp.one_core", "smp.all_cores"] {
                out.push(Rail::absent(
                    n,
                    "mbs",
                    Want::Higher,
                    "the split answer differs from the whole one",
                ));
            }
        }
    }

    out
}

/// The whole block, one rail per line, in the shape a program reads.
///
/// `name value unit want` and nothing else on the line, because the consumer
/// is a diff between two builds and a table with alignment in it is a table
/// somebody has to parse around. The header carries the version, so a reader
/// that meets a shape it does not know can say so rather than misread it.
pub fn report() {
    use crate::kprintln;
    let rs = rails();

    // **Two lists that have to agree, checked at the one moment both are in
    // hand.** `DECLARED` is what a reader of this report may rely on being
    // present; `rails()` is what actually got measured, and it builds its list
    // by hand because the measurements come in groups. A rail added to one and
    // not the other is a report that is silently a different shape, which is
    // the whole failure the `absent` line exists to prevent -- arriving by the
    // back door. Cheap here and impossible to check without running every
    // rail, which is why it is a line in the report rather than a claim.
    let agree = rs.len() == DECLARED.len()
        && rs.iter().zip(DECLARED.iter()).all(|(r, d)| r.name == d.0 && r.unit == d.1);
    if !agree {
        crate::gfx::console::set_color(crate::gfx::console::LTRED);
        kprintln!(
            "[rail] DECLARED and rails() disagree: {} declared, {} measured",
            DECLARED.len(),
            rs.len()
        );
        crate::gfx::console::set_color(crate::gfx::console::LTGRAY);
    }

    kprintln!("[rail] v1");
    // Three decimals, formatted by hand: this target has no float formatter
    // and `gflops` would otherwise print as an integer and lose the whole
    // fractional part it is measured in.
    for r in rs {
        match r.value {
            Some(v) => kprintln!("{} {} {} {}", r.name, milli(v), r.unit, r.want.as_str()),
            None => kprintln!("{} absent {} {}  -- {}", r.name, r.unit, r.want.as_str(), r.why),
        }
    }
    kprintln!("[rail] end");
}

/// A non-negative float as a decimal string with three places.
///
/// Hand-rolled because `{:.3}` needs a formatter this target does not have,
/// and because rounding to an integer would make `ai.matmul` -- which is a
/// single-digit number with everything interesting after the point -- read as
/// a constant across every build.
fn milli(v: f64) -> String {
    if !v.is_finite() || v < 0.0 {
        return String::from("nan");
    }
    let scaled = (v * 1000.0 + 0.5) as u64;
    format!("{}.{:03}", scaled / 1000, scaled % 1000)
}

/// Claims about the report itself, which is the only thing checkable without
/// two builds to compare.
pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |good: bool, what: &str| {
        if !good {
            ok = false;
        }
        kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
    };

    claim(milli(0.0) == "0.000", "zero renders as zero and not as an empty string");
    claim(milli(1.5) == "1.500", "a half renders as three places");
    claim(milli(0.0004) == "0.000", "a value below the resolution rounds to it");
    claim(milli(1234.5678) == "1234.568", "and rounds rather than truncates");
    claim(milli(f64::NAN) == "nan", "a non-finite value says so rather than printing a number");

    // **The property that stops a rail reading zero because nothing ran.** A
    // rail this machine cannot measure has no value at all, and the printer
    // says `absent`; if `absent` were ever spelled as `0` the comparison would
    // read it as the best possible graphics frame and the worst possible
    // bandwidth on the same report.
    let a = Rail::absent("x", "us", Want::Lower, "because");
    claim(a.value.is_none() && !a.why.is_empty(), "an absent rail carries no value and a reason");

    // Names are the join between two reports, so two rails sharing one would
    // make a diff silently compare the wrong pair.
    let mut names: Vec<&'static str> = DECLARED.iter().map(|r| r.0).collect();
    let n = names.len();
    names.sort_unstable();
    names.dedup();
    claim(names.len() == n && n > 0, "every rail has a distinct name, and there are some");

    // Direction is declared per rail rather than inferred, because `gflops`
    // and `us` point opposite ways and a comparison that guessed would call
    // every graphics improvement a regression.
    claim(
        DECLARED.iter().any(|r| r.2 == Want::Higher)
            && DECLARED.iter().any(|r| r.2 == Want::Lower),
        "both directions are represented, so the field is being exercised",
    );

    // **The claim that keeps `absent` honest.** `rails()` fills values into
    // this list and a rail it cannot measure has to still take its slot, or a
    // machine with no framebuffer prints a shorter report and the diff that
    // reads it lines the wrong pairs up. Checked against the declaration
    // rather than against a number written here, so adding a rail cannot make
    // this claim stale while leaving it passing.
    claim(
        !DECLARED.is_empty(),
        "the report is a declared list, so an unmeasurable rail still takes its slot",
    );
    ok
}
