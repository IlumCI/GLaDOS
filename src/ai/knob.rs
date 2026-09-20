//! The constants this machine may propose changing about itself.
//!
//! **A 0.6B cannot write free-form Rust and does not have to.** This tree
//! already authors from a closed set of templates (`author::BODIES`) and
//! already searches a *declared* space (`godel::GRID`, `frontier`,
//! `/ai/godel/tried`). The transformation space is that same pattern pointed
//! at source: a proposal names a row of this table and one of the values that
//! row declares, so the patch is valid Rust by construction rather than by
//! luck -- the argument `constrain.rs` makes about an applet name being
//! *unreachable* rather than merely improbable.
//!
//! ### It proposes and it cannot judge
//!
//! The kernel has no copy of its own source and cannot compile, and neither is
//! an omission. So a source proposal is not a trial: nothing here can build
//! the variant, run it, or produce a verdict. What it produces is a **patch in
//! the outbox** and a record that the point was reached, and the verdict comes
//! back from whatever built it.
//!
//! That is why these carry a `rail`. A proposal that claims to move nothing
//! cannot be judged at all, and one whose rail the judging machine cannot
//! measure must be refused rather than adopted blind -- `tools/rails.py`
//! answers `absent` for exactly that case and `judge` reads it as a veto.
//!
//! ### The table is small and every row is a trade somebody already swept
//!
//! Four rows, all from the retrieval scorer, and that is the honest size of
//! the set of constants in this tree that are genuinely *tunable* rather than
//! correctness parameters. `LEN_B` at 0.5 rather than the textbook 0.75 is a
//! sweep with an interior optimum; `IDF_POW` at two moved long-query recall
//! from 89.3% to 91.9% over the one it had been; `TF_K1` is the saturation that measured *worse* and was
//! kept so the day the corpus grows longer documents the answer is one command
//! away; `MIX` is zero because every weight above zero measured worse on this
//! checkpoint's embeddings and a different model may move it.
//!
//! Each is a number somebody chose from a measurement, which is exactly the
//! kind of number that goes stale when the thing it was measured on changes.
//! That is the case for a machine re-asking them, and it is why these are the
//! first rung rather than something with more reach.

/// One constant the machine may propose changing.
pub struct Knob {
    /// Path from the repository root, spelled as the patch will name it.
    pub file: &'static str,
    /// The `const` this row is about, spelled exactly as the source spells it.
    pub symbol: &'static str,
    /// What the source says today.
    ///
    /// **Checked against the file by `tools/knob.py`, which CI runs**, because
    /// nothing in the kernel can read its own source. A row that has gone
    /// stale would otherwise produce a patch that applies to nothing and a
    /// proposal that looks like it was tried -- the marker written, the point
    /// never measured.
    pub now: &'static str,
    /// The values that may be proposed, in the order they are walked.
    ///
    /// Declared rather than generated, so the search is re-derivable: given
    /// the markers in `/ai/godel/tried`, the next point is a function and not
    /// a coin. The same property `GRID` has and for the same reason.
    pub values: &'static [&'static str],
    /// The rail this is expected to move, as `tools/rails.py` names it.
    pub rail: &'static str,
    /// One line, for a person reading `godel source`.
    pub about: &'static str,
}

/// Surfaces no mechanism here can judge, and which therefore may not be in the
/// transformation space.
///
/// **Screenshots are captured and never compared.** There is no image diff
/// anywhere in this tree, so a swapped red and blue channel, a window drawn
/// off-screen or a font rendering hollow boxes are invisible to every judge,
/// every rail and every suite -- and `CLAUDE.md` records a wrong claim made
/// from a glance at a frame that reached a release note. A knob under one of
/// these would produce a patch that CI builds, boots, measures as `same` on
/// every rail it can read, and adopts, having checked nothing about the only
/// thing it changed.
///
/// A prefix list rather than a per-row flag, because the question is about the
/// *surface* and not about the constant: somebody adding a row for a new
/// graphics knob would have to add the flag too, and the one they forget is
/// the one that matters. The other half of the fix is a golden-frame diff,
/// which does not exist; until it does, this is the gate.
pub const UNJUDGEABLE: &[&str] = &["src/gfx/", "src/port/"];

/// Whether a file is on a surface nothing here can check.
pub fn unjudgeable(file: &str) -> bool {
    UNJUDGEABLE.iter().any(|p| file.starts_with(p))
}

pub const KNOBS: &[Knob] = &[
    Knob {
        file: "src/ai/lex.rs",
        symbol: "LEN_B",
        now: "0.5",
        // Around the measured optimum and out to the textbook value. 0.5 is
        // not in the list: proposing the value already in force is a
        // certificate saying nothing changed, which is the same reason
        // `next_config` excludes the rule already running.
        values: &["0.25", "0.4", "0.6", "0.75"],
        rail: "host.retrieval",
        about: "how much a long document is charged for its length",
    },
    Knob {
        file: "src/ai/lex.rs",
        symbol: "TF_K1",
        now: "1.2",
        values: &["0.0", "0.6", "2.0"],
        rail: "host.retrieval",
        about: "term-frequency saturation, which measured worse at 1.2",
    },
    Knob {
        file: "src/ai/lex.rs",
        symbol: "IDF_POW",
        now: "2",
        values: &["1", "3"],
        rail: "host.retrieval",
        about: "what power a term's rarity is counted at",
        // The one row whose rail is a weaker judge than the others', and it
        // says so where somebody reading a verdict will look. `lex.rs` has
        // the numbers: the kernel's own sweep chose two and the host mirror's
        // long-query sweep leans the other way, cleanly, and falls short only
        // on the bar. They index different things. A length charge means the
        // same under either tokenisation; a rarity exponent does not.
    },
    Knob {
        file: "src/ai/recall.rs",
        symbol: "MIX",
        now: "0.0",
        values: &["0.1", "0.3"],
        rail: "host.retrieval",
        about: "how much of the embedding channel is mixed into the score",
    },
];

/// How many (row, value) points there are in total.
pub fn points() -> usize {
    KNOBS.iter().map(|k| k.values.len()).sum()
}

/// The knob and value a proposal names, or `None` for a point this kernel does
/// not have.
///
/// **Refused rather than clamped.** A marker written by a future kernel with a
/// wider table would otherwise resolve to whatever row happens to sit at that
/// index here, and the ledger would record a measurement of one constant under
/// the name of another.
pub fn at(knob: usize, value: usize) -> Option<(&'static Knob, &'static str)> {
    let k = KNOBS.get(knob)?;
    Some((k, *k.values.get(value)?))
}

/// The patch a point stands for, as text.
///
/// Deliberately *not* a diff. A unified diff carries line numbers and context,
/// both of which go stale the moment anything above the constant moves, and
/// the kernel cannot read the file to produce either. What it can say exactly
/// is which symbol in which file takes which value, and `tools/knob.py` is the
/// half that finds the line. The same division `repair.rs` makes: the chooser
/// names a row, the thing that owns the file does the work.
pub fn patch(knob: usize, value: usize) -> Option<alloc::string::String> {
    use alloc::string::String;
    let (k, v) = at(knob, value)?;
    let mut s = String::from("knob 1\n");
    s.push_str("file ");
    s.push_str(k.file);
    s.push_str("\nsymbol ");
    s.push_str(k.symbol);
    s.push_str("\nfrom ");
    s.push_str(k.now);
    s.push_str("\nto ");
    s.push_str(v);
    s.push_str("\nrail ");
    s.push_str(k.rail);
    s.push('\n');
    Some(s)
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |good: bool, what: &str| {
        if !good {
            ok = false;
        }
        kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
    };

    claim(!KNOBS.is_empty() && points() > 0, "there is a space to search at all");

    // **Nothing visual may be in the space, because nothing here can judge
    // it.** A knob under `src/gfx/` would produce a patch that builds, boots,
    // reads `same` on every rail there is, and gets adopted having checked
    // nothing about the only thing it changed.
    claim(
        KNOBS.iter().all(|k| !unjudgeable(k.file)),
        "no declared knob touches a surface with no judge in front of it",
    );
    // And the gate has to be able to refuse, or it is a list nobody tested.
    claim(
        unjudgeable("src/gfx/theme.rs") && !unjudgeable("src/ai/lex.rs"),
        "and the gate recognises such a surface rather than passing everything",
    );

    // **The one that stops a proposal being a no-op.** A point whose value is
    // what the source already says is a patch that changes nothing, a build
    // that is byte-identical, and a rail comparison that reports `same` --
    // costing a whole CI run to learn what this table already knew.
    claim(
        KNOBS.iter().all(|k| !k.values.iter().any(|v| *v == k.now)),
        "no row offers the value it already has",
    );
    claim(
        KNOBS.iter().all(|k| {
            k.values.iter().enumerate().all(|(i, v)| !k.values[i + 1..].contains(v))
        }),
        "and no row offers one twice",
    );

    // Every row must be reachable and must name its subject completely. A row
    // with an empty symbol produces a patch `knob.py` cannot apply, and a row
    // with no rail produces a proposal nothing can judge -- which is the
    // failure this whole table exists to avoid.
    claim(
        KNOBS.iter().all(|k| {
            !k.file.is_empty() && !k.symbol.is_empty() && !k.now.is_empty()
                && !k.rail.is_empty() && !k.values.is_empty()
        }),
        "every row names a file, a symbol, its current value, a rail and some values",
    );
    claim(
        KNOBS.iter().enumerate().all(|(i, k)| {
            KNOBS[i + 1..].iter().all(|o| !(o.file == k.file && o.symbol == k.symbol))
        }),
        "no two rows are about the same constant",
    );

    // Indices, because a marker is (row, value) and a future kernel's table
    // may be wider or narrower than this one's.
    claim(at(0, 0).is_some(), "the first point resolves");
    claim(at(KNOBS.len(), 0).is_none(), "a row this kernel does not have is refused");
    claim(at(0, KNOBS[0].values.len()).is_none(), "and so is a value it does not have");

    // The patch says all four things, because `knob.py` refuses one that does
    // not -- and `from` is what makes a stale row fail loudly rather than
    // applying to nothing.
    let p = patch(0, 0).unwrap_or_default();
    claim(
        p.contains(KNOBS[0].file)
            && p.contains(KNOBS[0].symbol)
            && p.contains(KNOBS[0].now)
            && p.contains(KNOBS[0].values[0])
            && p.contains(KNOBS[0].rail),
        "a patch names the file, the symbol, the old value, the new one and the rail",
    );
    claim(patch(KNOBS.len(), 0).is_none(), "and a point that does not exist has no patch");
    ok
}
