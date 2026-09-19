//! Does retrieval make the answer better? The rail that was missing.
//!
//! Every judge in this tree scores *routing* -- J1 is McNemar over applet
//! choices, J2 replays curiosity goals, `core_bench` and `rule_bench` score the
//! same thing again -- and not one of them reads what `ask` replied. So when
//! `recall on` shipped there was no instrument that could say whether it helped,
//! and the one comparison anybody had taken by hand was a regression:
//!
//!     recall off   "The somatic motor processes are the ones that control
//!                   movement. They're responsible"
//!     recall on    "D. None of the above"
//!
//! The model had parroted the multiple-choice format of a retrieved MMLU node
//! instead of using it. That is the failure this module exists to catch, and
//! catching it decides what a corpus should contain.
//!
//! ### Bits of a reference answer, not a match against it
//!
//! The obvious instrument is exact match, and it is the wrong one twice over.
//! It needs questions whose answer is a short string, which is an exam and not
//! a conversation; and it is binary, so it throws away almost everything the
//! model did -- `progress.rs` has the measurement, `t = 6.61` on 256 windows
//! where GSM8K needed 1,319 questions to reach chi 3.86.
//!
//! So this scores the **bits the model spends on a written reference answer**,
//! teacher-forced, with the retrieved block in front of the question and
//! without it. Every token of the reference is an observation. Nothing is
//! generated, nothing is sampled, and the figure is deterministic.
//!
//! **And it catches the format failure where a match rail would not.** A block
//! that teaches the model to reply `D. None of the above` makes a prose
//! reference *less* probable, so the parroting shows up as more bits rather
//! than as an answer that merely looked odd to somebody reading it.
//!
//! ### Paired, and refused rather than truncated
//!
//! The arms differ only in the prefix; the answer is the same text, so the
//! byte count is identical and the difference is exact per item. What is not
//! automatic is that all three arms *fit*: the retrieved block makes the
//! prefix longer, so a question can fit unaided and overflow with a block in
//! front of it. An item that does not fit every arm is dropped from all of
//! them, because a pair where one side was truncated is two measurements of
//! different things with one name.
//!
//! ### The oracle arm is the canary
//!
//! A rail that has never reported an improvement is indistinguishable from one
//! that measures nothing -- `differ.rs` makes the argument and `smp.rs` paid
//! for it. So there is a third arm that shows the model a sentence genuinely
//! containing the fact, and **it must beat the unaided arm**. If putting the
//! answer's own source in the context does not reduce the bits spent on the
//! answer, this instrument is broken and its verdict about retrieval means
//! nothing. The bench says so in those words rather than printing a number.
//!
//! The oracle is deliberately *not* the reference answer restated. A block
//! containing the answer verbatim would drive the bits to nearly zero and
//! prove only that the model can copy.
//!
//! ### What the questions are about, and why that is the honest choice
//!
//! They are about this machine. An operator asking this machine questions asks
//! about its verbs, its namespace and its own state, not about mitosis -- and
//! the shipped corpus is MMLU, GSM8K and Wikipedia, which contains none of it.
//!
//! That makes the retrieved arm a **calibrated negative today**: the corpus
//! cannot answer these, so retrieval should not help, and if this rail ever
//! reports that it did, something is wrong with the rail rather than good
//! about the corpus. What it *can* honestly report is harm, which is the
//! measurement that was missing.

use alloc::string::String;
use alloc::vec::Vec;

use super::progress::{bits_for, Bits};

/// A question, the answer a good reply would be close to, and a sentence that
/// genuinely carries the fact.
pub struct Ask {
    pub q: &'static str,
    /// Scored. Prose rather than a token, because bits per byte over a written
    /// sentence is dense where a letter is one observation.
    pub a: &'static str,
    /// What the oracle arm is shown. Overlapping facts, different words: a
    /// restatement of `a` would measure copying.
    pub src: &'static str,
}

/// Twenty-four, about this machine, from its own documentation.
///
/// Small on purpose and stated rather than apologised for: every token of
/// every answer is an observation, so the evidence is the ~700 scored tokens
/// rather than the 24 items, and the paired statistic is over per-item
/// deltas in bits per byte.
pub const ASKS: &[Ask] = &[
    Ask {
        q: "What does 'store unlock' do?",
        a: "It permits writes to the NVMe store, and only inside the range the unlock claimed. Mounting a store does not unlock it, because the write gate is separate and stays manual.",
        src: "NVMe writes are locked by default; the unlock names an LBA window and nvme::write checks every write against it.",
    },
    Ask {
        q: "Why is there no cargo test in this project?",
        a: "Because this is a no_std UEFI binary with no host test runner. Verification is the boot selftest sections plus driving QEMU over a serial socket.",
        src: "A no_std UEFI target has nowhere for a host test runner to live, so the selftests run at boot and drive.py does the rest.",
    },
    Ask {
        q: "What is Aiksi?",
        a: "The system language everything above the kernel is written in. A program is a file ending .ai&xi, and it is lexed, parsed, and walked as a tree.",
        src: "src/aiksi holds a language whose relationship to GLaDOS is C's to Unix, with an extension deliberately unusual enough that nothing on the host claims it.",
    },
    Ask {
        q: "How does a new boot image reach this machine?",
        a: "Three files are written to the ESP: the new image, its detached signature, and a flag whose presence is the request. The swap takes effect at the next boot rather than the running one.",
        src: "update::hook runs before ExitBootServices because the firmware's FAT driver is the only writer of the ESP that exists while a boot image can still be swapped.",
    },
    Ask {
        q: "What is Racy and when should I not use it?",
        a: "Racy is single-core interior mutability and it is not a lock. Anything a second task can reach needs Spin instead, and every conversion is a claim that has to be verified.",
        src: "sync::Racy is the designated grep target for the day SMP arrives; sync::Spin is the real lock, and lock_irq is not optional on either.",
    },
    Ask {
        q: "Why does the kernel keep its KV cache as int8?",
        a: "To fit. The cache is sized from the trained context length, and holding it in float would cost several times the memory on a machine with one contiguous heap.",
        src: "KvLayer stores int8, which is why the loss is piecewise constant in anything upstream of a cached key and why the exact state exists for training.",
    },
    Ask {
        q: "What does 'diag all' do that a bare 'diag' does not?",
        a: "It runs every named suite. A bare diag only lists them, with a dash beside everything that has not run, which is easy to misread as a clean sweep.",
        src: "diag on its own prints a table and a tally reading zero passed, zero failed, and the rest not run.",
    },
    Ask {
        q: "Why must drive.py be given a release build?",
        a: "Because it stages the release artifact when one exists and falls back to debug otherwise, so a debug-only build leaves a stale binary staged and the change under test never boots.",
        src: "drive.py prefers target/x86_64-unknown-uefi/release/glados.efi, so building debug alone means the change never runs.",
    },
    Ask {
        q: "What is a forest node?",
        a: "A structured concept node: a head line the router pools over, and a body carrying the concept, a method, machine-checkable steps where they exist, and the original passage.",
        src: "The head is what stays resident and the body is what reaches the model's context when a node wins retrieval.",
    },
    Ask {
        q: "Why is the system turn pinned in the KV cache?",
        a: "Because a pinned slot never recycles. Setting the sink count to the system turn's token length keeps the instructions and the applet list in their original positions for as long as the conversation runs.",
        src: "slot_of returns j unchanged below the sink count, so a sink is a slot that never recycles rather than merely a privileged position.",
    },
    Ask {
        q: "What happens when a guest at ring 3 faults?",
        a: "The guest is ended and the machine carries on. At ring 3 the kernel is intact by construction, so ending the guest is the honest response rather than halting.",
        src: "Only a guest's own pages carry the U bit, so a fault there cannot have corrupted the kernel and the machine answers afterwards.",
    },
    Ask {
        q: "Why does this kernel refuse to run a mixture-of-experts model?",
        a: "Because the smallest published one is far larger than anything that reaches a UEFI pool on this laptop, so a forward pass for it could never be run and contradicted.",
        src: "MoE is refused at load rather than half-implemented, since nothing that size fits and an untestable path is worse than an absent one.",
    },
    Ask {
        q: "What is the quiet window?",
        a: "The hours when the machine may run unattended work. A trial runs only when the clock falls inside it and the entropy ring shows no hardware input.",
        src: "Trials run between two and six in the morning, and godbits::felt is what answers whether anybody is present.",
    },
    Ask {
        q: "Why are rails compared with a control divided out?",
        a: "Because a reading taken on a busy day moves for reasons that have nothing to do with the change. The control moves the same way, so dividing it out leaves what the build did.",
        src: "video.rect and core.new are the controls, and ai.* and smp.* have none, which is a hole the tool declares rather than leaving to be noticed.",
    },
    Ask {
        q: "What does a signed manifest look like?",
        a: "Its text followed by exactly eighty bytes of signature, as one object. Two objects can be served out of step and produce a signature failure that is really a deployment race.",
        src: "There is one connection and no pipelining, so the manifest and its signature travel together rather than as two fetches.",
    },
    Ask {
        q: "Why does the kernel need a second key for verdicts?",
        a: "Because the two private halves live in different places. The key that ships a kernel must not be reachable from a workflow any allowlisted machine in the field can start.",
        src: "release.yml signs images and is reached by pushing a tag; propose.yml signs verdicts and is reached by a dispatch from the proposal function.",
    },
    Ask {
        q: "What is the difference between fit and train adapter?",
        a: "Fit moves the linear probe and its head by closed-form ridge regression and never touches the checkpoint. Train adapter moves a low-rank adapter over the model's classifier.",
        src: "Two different things here are called training, and confusing them makes every number ambiguous: one is a probe, the other is an adapter.",
    },
    Ask {
        q: "Why is constrained decoding used for applet names?",
        a: "Because it makes an invalid name unreachable rather than merely improbable. The grammar is built from the live applet table, so a name outside it cannot be sampled at all.",
        src: "Read-only mode works by removing mutating applets from the reachable set before sampling, and never by checking afterwards.",
    },
    Ask {
        q: "What is a content-addressed store?",
        a: "One where an object is named by the hash of its contents. A copy costs nothing, a snapshot is a single root hash, and moving bytes on disk cannot rename anything.",
        src: "The content hash covers content only and never block locations, or relocating a block would give an object a new name.",
    },
    Ask {
        q: "Why does the loop spend alpha?",
        a: "Because every judged comparison is a test, and running one nightly for a year would adopt noise by construction. The budget makes the bar rise as evidence is spent.",
        src: "Each trial debits from a declared series that sums to a fixed total, and only a new body of evidence refills it.",
    },
    Ask {
        q: "What does 'ask new' do?",
        a: "It forgets the conversation, so the next question starts a fresh one. Without it the cache is resumed and the next turn continues what came before.",
        src: "ask is one continuing conversation rather than a question asked into the void, and the cache is what carries it.",
    },
    Ask {
        q: "Why does the fault reporter print twice?",
        a: "Because painting the console from inside an interrupt gate can fault, so the report goes to the serial port first, whole, and is then attempted on the screen regardless.",
        src: "Serial is a port write and cannot block or fault; on the laptop there is no serial port and the framebuffer is the only diagnostic there is.",
    },
    Ask {
        q: "What is the difference between a skill and an application here?",
        a: "A skill is a program under the tools directory that the agent compiled from an episode. An application is a stored program with a manifest and a grant behind it.",
        src: "Both run sandboxed unless an operator has named the exact bytes, and identity is the hash of the file, so editing one revokes its trust.",
    },
    Ask {
        q: "Why is the routing corpus compiled into the image?",
        a: "So the machine can route before anything is mounted. A corpus that lived only on disk would leave a freshly installed machine unable to decide anything.",
        src: "The seed examples are in the source, and teach appends to what is in the namespace afterwards.",
    },
];

/// Which prefix an arm puts in front of the question.
#[derive(Clone, Copy, PartialEq)]
pub enum Sees {
    /// The question alone. What the machine does today with `recall off`.
    Nothing,
    /// Whatever `recall::pick` returns for this question.
    Retrieved,
    /// A sentence that genuinely carries the fact. The canary.
    Oracle,
}

/// The turn as `companion::turn` frames it, minus the system turn.
///
/// **The system turn is left out and that is a stated bias, not an oversight.**
/// It is identical in both arms, so it cannot move the paired difference --
/// but it does shift the absolute level, so `ai.answer` is bits per byte of a
/// reference answer in an unprimed turn rather than in a conversation. The
/// framing that remains is exactly what `turn` writes, because a rail built on
/// a different prompt shape measures a different machine.
pub fn prompt(block: &str, q: &str) -> String {
    let mut s = String::from("<|im_start|>user\n");
    s.push_str(block);
    s.push_str(q);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    s
}

/// Bits the model spends on `answer` given `prefix`.
///
/// The prefix is walked without scoring, then every token of the answer is
/// scored including the first -- which `progress::measure` cannot do for its
/// own text, since there the first token is spelt by the prompt and nothing
/// predicted it. Here the prompt is the prefix, so the answer's opening token
/// is predicted like any other and every byte of it is covered.
pub fn conditional(e: &mut super::Engine, prefix: &str, answer: &str) -> Option<Bits> {
    let pre = e.tok.encode(prefix, true, false);
    let ans = e.tok.encode(answer, false, false);
    if pre.is_empty() || ans.is_empty() {
        return None;
    }
    // **Refused rather than truncated.** A pair whose arms scored different
    // numbers of the answer's tokens is two measurements with one name.
    let room = e.model.cfg.live_cap().saturating_sub(2);
    if pre.len() + ans.len() > room {
        return None;
    }

    let mut cfg = e.model.cfg;
    cfg.seq_len = pre.len() + ans.len() + 1;
    let mut st = super::model::State::new(&cfg);
    for (i, &t) in pre.iter().enumerate() {
        e.model.forward(&mut st, t, i);
    }
    let (mut bits, mut bytes, mut tokens) = (0.0f32, 0usize, 0usize);
    let mut want = ans[0];
    for k in 0..ans.len() {
        let b = bits_for(&st.logits, want)?;
        bits += b;
        bytes += e.tok.token_bytes(want).len();
        tokens += 1;
        if k + 1 == ans.len() {
            break;
        }
        e.model.forward(&mut st, ans[k], pre.len() + k);
        want = ans[k + 1];
    }
    Some(Bits { bits, bytes, tokens })
}

/// What one run of the bench found.
pub struct Verdict {
    /// Items where all three arms fitted and scored.
    pub n: usize,
    /// Items dropped because some arm did not fit the window.
    pub dropped: usize,
    /// Mean bits per byte, one per arm.
    pub off: f32,
    pub on: f32,
    pub oracle: f32,
    /// Paired t over per-item (off - on). Positive means retrieval helped.
    pub t_on: f32,
    /// Paired t over per-item (off - oracle). The canary: must be positive
    /// and large, or this instrument is not measuring what it claims.
    pub t_oracle: f32,
    /// Items retrieval helped, and items it hurt.
    pub helped: usize,
    pub hurt: usize,
    /// Items where `pick` returned nothing at all, so the retrieved arm and
    /// the unaided arm are the same measurement. Reported because a rail that
    /// compared a corpus against itself and called the difference zero would
    /// be right for the wrong reason.
    pub empty: usize,
}

/// Paired t over a list of differences. `None` when there is nothing to say.
///
/// Separate from the bench so it can be checked as arithmetic, with no model:
/// a statistic is exactly the kind of thing that is wrong by a square root and
/// gives no sign of it.
pub fn paired_t(d: &[f32]) -> Option<f32> {
    if d.len() < 2 {
        return None;
    }
    let n = d.len() as f32;
    let mean = d.iter().sum::<f32>() / n;
    let var = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (n - 1.0);
    if !(var > 0.0) {
        // Every difference identical. That is a real answer for a rail whose
        // two arms can be the same prompt, and it is not a t of infinity.
        return if mean == 0.0 { Some(0.0) } else { None };
    }
    Some(mean / super::tensor::sqrtf(var / n))
}

/// Run all three arms over `ASKS`.
pub fn bench(e: &mut super::Engine) -> Option<Verdict> {
    let mut d_on: Vec<f32> = Vec::new();
    let mut d_or: Vec<f32> = Vec::new();
    let (mut off, mut on, mut oracle) = (0.0f32, 0.0f32, 0.0f32);
    let (mut helped, mut hurt, mut dropped, mut empty) = (0, 0, 0, 0);

    for a in ASKS {
        // The retrieved block is taken once and used as the prefix, which is
        // what `companion::turn` does -- and it is taken whatever `recall on`
        // says, because this bench is the thing that decides whether that
        // switch should be on and must not read its own conclusion.
        let budget = super::recall::budget_for(e.model.cfg.seq_len);
        let block = match super::recall::pick(a.q, budget, super::recall::ASK_K) {
            Some(f) => f.text,
            None => {
                empty += 1;
                String::new()
            }
        };
        let mut orc = String::from(super::recall::PREAMBLE);
        orc.push_str("--- 1\n");
        orc.push_str(a.src);
        orc.push_str("\n\n");

        let b0 = conditional(e, &prompt("", a.q), a.a);
        let b1 = conditional(e, &prompt(&block, a.q), a.a);
        let b2 = conditional(e, &prompt(&orc, a.q), a.a);
        // All three or none: a pair with one arm missing is not a pair.
        let (b0, b1, b2) = match (b0, b1, b2) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => {
                dropped += 1;
                continue;
            }
        };
        let (p0, p1, p2) = (b0.per_byte(), b1.per_byte(), b2.per_byte());
        off += p0;
        on += p1;
        oracle += p2;
        d_on.push(p0 - p1);
        d_or.push(p0 - p2);
        if p1 < p0 {
            helped += 1;
        } else if p1 > p0 {
            hurt += 1;
        }
    }

    let n = d_on.len();
    if n == 0 {
        return None;
    }
    let f = n as f32;
    Some(Verdict {
        n,
        dropped,
        off: off / f,
        on: on / f,
        oracle: oracle / f,
        t_on: paired_t(&d_on).unwrap_or(0.0),
        t_oracle: paired_t(&d_or).unwrap_or(0.0),
        helped,
        hurt,
        empty,
    })
}

/// The unaided figure, as a rail. Millibits per byte, lower being better.
///
/// The *unaided* arm rather than the retrieved one, deliberately: a rail is a
/// number two builds are compared on, and retrieval is off by default, so the
/// figure that describes the shipped machine is the one without a block. What
/// retrieval does to it is a paired question and `bench` is where it is asked.
pub fn rail_millibits(e: &mut super::Engine) -> Option<u64> {
    let (mut bits, mut bytes) = (0.0f32, 0usize);
    for a in ASKS {
        let b = conditional(e, &prompt("", a.q), a.a)?;
        bits += b.bits;
        bytes += b.bytes;
    }
    if bytes == 0 {
        return None;
    }
    Some((bits / bytes as f32 * 1000.0) as u64)
}

/// What the arithmetic claims, with no model and no forest.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    check("there are questions to ask at all", ASKS.len() >= 16);
    // A reference answer is what gets scored, so a one-word one is a rail with
    // one observation in it wearing a dense rail's name.
    check(
        "every reference answer is prose rather than a token",
        ASKS.iter().all(|a| a.a.split_whitespace().count() >= 12),
    );
    check(
        "every question has a source sentence for the oracle arm",
        ASKS.iter().all(|a| a.src.split_whitespace().count() >= 8),
    );
    // **The oracle must not be the answer restated.** A block containing the
    // reference verbatim drives the bits to nearly nothing and would prove
    // only that the model can copy, so the canary would pass on an instrument
    // that measures copying rather than help.
    check(
        "and no source is the answer repeated back",
        ASKS.iter().all(|a| !a.src.contains(a.a) && !a.a.contains(a.src)),
    );
    check(
        "no question is asked twice",
        ASKS.iter().enumerate().all(|(i, a)| {
            ASKS.iter().take(i).all(|b| b.q != a.q)
        }),
    );

    // --- the statistic, as arithmetic ---------------------------------
    check("a single difference says nothing", paired_t(&[1.0]).is_none());
    check(
        "identical differences are a t of zero when they are zero",
        paired_t(&[0.0, 0.0, 0.0]) == Some(0.0),
    );
    // A constant non-zero difference has no spread, so the t is undefined
    // rather than infinite. Answering infinity here would make a rail that
    // moved every item by exactly the same hair look like certainty.
    check(
        "and are refused when they are not",
        paired_t(&[0.5, 0.5, 0.5]).is_none(),
    );
    let t = paired_t(&[1.0, 2.0, 3.0, 4.0]).unwrap_or(0.0);
    // mean 2.5, sd 1.29099, se 0.645497, t = 3.87298
    check(
        "a worked case matches the arithmetic to three places",
        (t - 3.87298).abs() < 0.001,
    );
    check(
        "the sign says which arm won",
        paired_t(&[-1.0, -2.0, -3.0, -4.0]).unwrap_or(0.0) < 0.0,
    );

    // --- the framing -------------------------------------------------
    let p = prompt("BLOCK\n", "Q?");
    check(
        "the block goes before the question, inside the user turn",
        p.find("BLOCK").unwrap_or(9) < p.find("Q?").unwrap_or(0)
            && p.starts_with("<|im_start|>user\n"),
    );
    check(
        "and the turn is handed to the assistant afterwards",
        p.ends_with("<|im_start|>assistant\n"),
    );
    check(
        "an empty block leaves the question alone with its framing",
        prompt("", "Q?") == "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n",
    );

    ok
}
