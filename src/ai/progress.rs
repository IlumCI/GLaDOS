//! How well this machine predicts its own life, in bits.
//!
//! Schmidhuber's 1991 signal, and the reason to want it here is written on the
//! judges next door. Every one of J1's measurements is *routing accuracy* over
//! a corpus of task descriptions: a binary rail, one bit an item, needing a
//! thousand items before a paired test can see anything. `initiative`'s
//! curiosity goals are the same shape -- eight declared questions with a
//! declared right answer -- so the whole of what the loop can want is "pick a
//! better applet more often".
//!
//! Bits per byte is the dense alternative and the figure is measured rather
//! than argued: on the host, `lm_eval --task bpb` reaches t = 6.61 on 256
//! windows where GSM8K needed 1,319 questions to reach chi 3.86 against a 3.84
//! bar. Every token is an observation instead of every question.
//!
//! ### Learning progress is a *difference*, and the difference already has
//! machinery
//!
//! The 1991 formulation is `bits(history under the compressor at t) - bits(the
//! same history at t+1)`: not how well the model predicts, but how much better
//! it predicts than it did. Two builds, one corpus, one number each.
//!
//! That is exactly what `tools/rails.py` does, with a declared noise floor, a
//! control divided out and a paired test -- so this ships as a **rail** and
//! the differencing is somebody else's job that is already done. A rail that
//! computed its own progress would need to remember what it read last time,
//! which is a second record that can disagree with the first.
//!
//! ### The history is the machine's own, and that is the curiosity part
//!
//! Not a held-out slice of somebody's corpus. The journal the night writes,
//! the ledger the judges write: the record of what this machine has been
//! doing. A variant that predicts its own life better has learned something
//! about itself, which is the only reading of "curiosity" that a kernel with
//! one address space and no internet can honestly make.
//!
//! **It is not held out and it cannot be.** The journal is written by the same
//! machine being measured, so a variant that changed what the night *does* has
//! changed the text as well as the predictor, and the rail moves for two
//! reasons at once. `rails.py` cannot tell those apart and neither can this.
//! What keeps it honest is that the comparison is between two builds reading
//! **one** history file, taken once: the `before` arm and the `after` arm are
//! handed the same bytes, so a difference is about the model. Said here rather
//! than discovered from a rail that drifted.

use alloc::string::String;

/// Bits and the bytes they were spent on.
#[derive(Clone, Copy)]
pub struct Bits {
    pub bits: f32,
    /// Bytes actually predicted. The first token is spelt by the prompt and
    /// nothing predicted it, so it is not among these.
    pub bytes: usize,
    pub tokens: usize,
}

impl Bits {
    pub fn per_byte(&self) -> f32 {
        if self.bytes == 0 {
            0.0
        } else {
            self.bits / self.bytes as f32
        }
    }

    /// Bits per *token*, which is the figure a language modelling paper quotes
    /// and is the wrong one to compare across tokenizers. Kept because it is
    /// free and because seeing both is how a tokenizer change gets noticed:
    /// bits per byte is invariant to how the text was cut up and bits per
    /// token is not.
    pub fn per_token(&self) -> f32 {
        if self.tokens == 0 {
            0.0
        } else {
            self.bits / self.tokens as f32
        }
    }
}

/// How surprised a distribution was by what actually came next.
///
/// Log-sum-exp with the maximum subtracted, which is not tidiness: the
/// classifier's logits reach into the tens and `exp(40)` in f32 is infinity,
/// so the naive form answers `inf - inf` on exactly the confident predictions
/// a good model makes.
///
/// `None` rather than a number when the logits are not finite. A NaN here
/// would propagate into a rail, and a rail carrying NaN compares equal to
/// nothing and unequal to everything, which reads as a build that changed
/// every measurement.
pub fn bits_for(logits: &[f32], target: usize) -> Option<f32> {
    let z = *logits.get(target)?;
    if !z.is_finite() {
        return None;
    }
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if !l.is_finite() {
            return None;
        }
        if l > max {
            max = l;
        }
    }
    let mut sum = 0.0f32;
    for &l in logits {
        sum += super::tensor::expf(l - max);
    }
    if !(sum > 0.0) || !sum.is_finite() {
        return None;
    }
    let nats = (max + super::tensor::lnf(sum)) - z;
    // A probability cannot exceed one, so the surprise cannot be negative --
    // but f32 rounding in the sum can put it a hair below zero on a prediction
    // that was essentially certain, and a negative bit count accumulated over
    // a window is a rail that improves by being confidently right about
    // nothing. Clamped at the one place it can happen.
    Some((nats * core::f32::consts::LOG2_E).max(0.0))
}

/// Score a text under the model, one position at a time.
///
/// **A forward per token, and no way around it here.** `prefill` is
/// weight-stationary and materialises only the last position's logits, which
/// is what makes it worth having and exactly the wrong shape for this: bits
/// per byte needs a distribution at every position. So this costs what
/// generation costs, which on the 0.6B is tens of milliseconds a token -- a
/// nightly figure, never an interactive one.
///
/// **It runs on a scratch state, not on the live one**, and that is not
/// politeness. `harness` walks the corpus through `e.state` and costs whoever
/// was talking their conversation, which is affordable for a trial that also
/// costs twenty minutes. This is a rail: `bench report` runs it twice on a
/// CI boot and an operator may run it at the prompt, and a measurement that
/// silently ends the conversation to take a reading is a measurement nobody
/// will take twice.
///
/// The scratch is sized to the window rather than to the checkpoint, which is
/// what makes that affordable: `State::new` allocates by `live_cap`, so a full
/// one for the 0.6B is 112 MiB of KV cache to score 256 positions. A config
/// with `seq_len` cut to the window is the same state, three orders of
/// magnitude smaller, and every other field of the model untouched.
pub fn measure(e: &mut super::Engine, text: &str, cap_tokens: usize) -> Option<Bits> {
    if text.is_empty() {
        return None;
    }
    // With a beginning-of-sequence token, so the first real token is predicted
    // too and every byte of the text is covered. Without it the first token is
    // free and a short window's figure depends on how long that token was.
    let mut ids = e.tok.encode(text, true, false);
    let room = e.model.cfg.live_cap().saturating_sub(2).min(cap_tokens);
    if room < 2 {
        return None;
    }
    if ids.len() > room {
        ids.truncate(room);
    }
    if ids.len() < 2 {
        return None;
    }

    let mut cfg = e.model.cfg;
    cfg.seq_len = ids.len() + 1;
    let mut st = super::model::State::new(&cfg);
    let (mut bits, mut bytes, mut tokens) = (0.0f32, 0usize, 0usize);
    for i in 0..ids.len() - 1 {
        e.model.forward(&mut st, ids[i], i);
        let b = bits_for(&st.logits, ids[i + 1])?;
        bits += b;
        bytes += e.tok.token_bytes(ids[i + 1]).len();
        tokens += 1;
    }
    Some(Bits { bits, bytes, tokens })
}

/// Where this machine's own history is kept.
///
/// The night's journal and the judges' ledger, in that order, because the
/// journal is prose about what happened and the ledger is the arithmetic that
/// justified it -- a predictor that has learned the shape of one has not
/// necessarily learned the other.
pub const SOURCES: [&str; 2] = ["/ai/mind/journal.txt", "/ai/godel/ledger.txt"];

/// This machine's recent history, newest last, bounded.
///
/// **Taken once and handed to both arms of a comparison.** The machine writes
/// this file, so reading it separately per build would compare two predictors
/// on two texts and call the difference learning.
///
/// The *tail*, because the tail is what a machine that has been running has
/// most of and because a head would freeze the measurement on whatever
/// happened first. Cut at a line boundary, so a window never begins mid-word
/// and charges the model for a prefix it could not have seen.
pub fn own_history(max_bytes: usize) -> String {
    let mut out = String::new();
    for path in SOURCES {
        let Some(bytes) = crate::sysbox::read_blob(path) else { continue };
        let Ok(text) = String::from_utf8(bytes) else { continue };
        out.push_str(&text);
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    tail_lines(&out, max_bytes)
}

/// The last whole lines of a text, up to a byte budget.
///
/// **The budget is a ceiling and the cut is the earliest line start that fits
/// under it**, not the next one after the offset. Those differ by a whole line
/// exactly when the offset already lands on a boundary, which for a log of
/// similar-length entries is most of the time -- and a rail that silently read
/// one line less than it was given would be a rail comparing two builds on
/// almost the same corpus.
pub fn tail_lines(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return String::from(text);
    }
    let from = text.len() - max_bytes;
    if from == 0 || text.as_bytes()[from - 1] == b'\n' {
        return String::from(&text[from..]);
    }
    // Forward to the next line start. `char_indices` rather than slicing at
    // `from`, because a byte offset can land inside a multi-byte character and
    // slicing there panics -- the same "a byte count is not a column count"
    // family this tree keeps paying for.
    let at = text[..]
        .char_indices()
        .find(|(i, c)| *i >= from && *c == '\n')
        .map(|(i, _)| i + 1)
        .unwrap_or_else(|| {
            text.char_indices().find(|(i, _)| *i >= from).map(|(i, _)| i).unwrap_or(text.len())
        });
    String::from(&text[at..])
}

/// How much history a rail reads. Declared, because a rail whose corpus size
/// moved between two builds is a rail comparing two different questions.
pub const RAIL_BYTES: usize = 2048;

/// How many tokens a rail will spend. At tens of milliseconds a token on the
/// 0.6B this is the whole cost of the rail, and it is bounded here rather than
/// by whatever happened to be in the journal.
pub const RAIL_TOKENS: usize = 256;

/// The rail: bits per byte of this machine's own history, in millibits.
///
/// Integer, because `bench::Rail` carries integers and because a rail that
/// reported a float would invite a comparison at a precision the measurement
/// does not have. Milli, so a change of a thousandth of a bit per byte is
/// visible -- which is the scale a small model's improvement actually lands
/// on.
pub fn rail_millibits(e: &mut super::Engine) -> Option<u64> {
    let text = own_history(RAIL_BYTES);
    if text.len() < 64 {
        return None;
    }
    let b = measure(e, &text, RAIL_TOKENS)?;
    if b.bytes == 0 {
        return None;
    }
    Some((b.per_byte() * 1000.0) as u64)
}

pub fn selftest() -> bool {
    use crate::kprintln;
    let mut ok = true;
    let mut claim = |good: bool, what: &str| {
        kprintln!("  {}   {}", if good { "ok " } else { "FAIL" }, what);
        ok &= good;
    };

    // The arithmetic, with no model at all -- the `update::decide` discipline,
    // and it is what makes these checks mean the same thing on a machine with
    // no checkpoint as on one with the 0.6B resident.

    // A uniform distribution over V spends exactly log2(V) bits, whatever it
    // was asked about. Eight is chosen so the answer is 3 and a reader can
    // check it without a calculator.
    let flat = [0.0f32; 8];
    let b = bits_for(&flat, 3).unwrap_or(-1.0);
    claim((b - 3.0).abs() < 0.01, "a uniform choice of eight costs three bits");
    claim(
        bits_for(&flat, 0) == bits_for(&flat, 7),
        "and a uniform distribution is equally surprised by anything",
    );

    // Certainty is free, and being certain of the wrong thing is expensive.
    let mut sharp = [0.0f32; 8];
    sharp[2] = 20.0;
    let right = bits_for(&sharp, 2).unwrap_or(-1.0);
    let wrong = bits_for(&sharp, 5).unwrap_or(-1.0);
    claim(right < 0.001, "a confident prediction that lands costs almost nothing");
    claim(wrong > 25.0, "and one that misses costs a great deal");
    claim(right >= 0.0, "and no prediction ever costs less than nothing");

    // **The one that fails on a naive implementation.** `exp(300)` is infinity
    // in f32, so log-sum-exp without the maximum subtracted answers `inf -
    // inf`, which is NaN -- on exactly the confident predictions a model that
    // is working produces.
    let mut huge = [300.0f32; 4];
    huge[1] = 301.0;
    let h = bits_for(&huge, 1);
    claim(
        h.map(|v| v.is_finite()) == Some(true),
        "logits large enough to overflow exp are still scored",
    );

    claim(bits_for(&flat, 99).is_none(), "a target outside the vocabulary is refused");
    claim(
        bits_for(&[0.0, f32::NAN, 1.0], 0).is_none(),
        "and a distribution with a NaN in it answers nothing rather than a number",
    );

    // Bits per byte against bits per token: the same bits, divided two ways,
    // and the reason both are reported.
    let bb = Bits { bits: 120.0, bytes: 60, tokens: 20 };
    claim(
        (bb.per_byte() - 2.0).abs() < 0.001 && (bb.per_token() - 6.0).abs() < 0.001,
        "bits divide by bytes and by tokens to different figures",
    );
    claim(
        Bits { bits: 1.0, bytes: 0, tokens: 0 }.per_byte() == 0.0,
        "and an empty measurement answers zero rather than dividing by it",
    );

    // The window. A tail cut mid-character is a panic, not a bad measurement,
    // and this tree has paid for that family twice already.
    let text = "alpha\nbeta\ngamma\ndelta\n";
    claim(tail_lines(text, 100) == text, "a history shorter than the budget is taken whole");
    // Twelve is `gamma\ndelta\n` exactly, so the budget lands on a line start
    // and nothing should be dropped. The version that always advanced to the
    // *next* newline answered `delta\n` here -- a whole line short, silently,
    // which for a rail is two builds read on different corpora.
    let cut = tail_lines(text, 12);
    claim(cut == "gamma\ndelta\n", "a budget landing on a line start drops nothing");
    let cut = tail_lines(text, 13);
    claim(
        cut.len() <= 13 && cut == "gamma\ndelta\n",
        "and one landing mid-line comes forward rather than cutting a line in half",
    );
    let wide = "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\nxyz\n";
    let w = tail_lines(wide, 9);
    claim(!w.is_empty(), "a cut that lands inside a character does not take the machine");
    claim(
        core::str::from_utf8(w.as_bytes()).is_ok(),
        "and what comes back is still text",
    );

    ok
}
