//! Tokens the system added to its own vocabulary, and how they learn.
//!
//! A hosted model has a fixed vocabulary, so its tool calls have to be *spelled
//! out* in tokens it already has -- as JSON, or as a name matched against a
//! grammar -- and something downstream has to turn those characters back into
//! an action. We own the embedding matrix, so none of that is needed: one row
//! per applet makes a tool call a single token, with nothing to spell and
//! nothing to parse.
//!
//! # Why this is not fine-tuning
//!
//! The 260K-parameter transformer underneath never moves. Only the rows here
//! do. That is the adapter regime rather than SFT (Houlsby et al. 2019;
//! AdapterHub, Pfeiffer et al. 2020), and it is chosen for the obvious reason:
//! nothing in the base can drift, so no prior capability can be lost, and a
//! training step costs an outer product rather than a backward pass through
//! five layers.
//!
//! # Why a hypernetwork on top of that
//!
//! Direct rows have one flaw, and it is specific to this system: `APPLETS` is
//! *data*, and it grows. A newly added applet would get a random row and be
//! useless until somebody had demonstrated it enough times.
//!
//! So rather than learning 21 independent rows, learn the general map from a
//! description to a row (Ha et al. 2016; Karimi Mahabadi et al. 2021 for the
//! adapter-generating form). A new applet is then competent the moment it is
//! written, from its help text alone, with no examples at all. That is the
//! property `selftest` holds an applet out to check.
//!
//! # Shape
//!
//! ```text
//!   row_i = desc_i + h(desc_i) + delta_i
//! ```
//!
//! `desc_i` is the mean of the embeddings of the words describing applet i --
//! frozen, and already a decent row on its own. `h` is a two-layer MLP shared
//! across every applet. `delta_i` is a per-applet correction for whatever `h`
//! cannot express.
//!
//! The residual form matters. If `h` produced the row outright, initialisation
//! would be random and the sensible starting point would be thrown away; as a
//! correction, `h` and `delta` both start near zero and the head begins exactly
//! where the pooled description put it.
//!
//! Gradients are derived by hand. For a two-layer MLP that is about forty lines
//! and no autodiff, which is the whole reason this is feasible on a machine
//! with no training framework on it.

use super::model::Model;
use super::sample::Rng;
use super::tensor;
use super::tokenizer::Tokenizer;
use crate::sysbox;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Width of the hypernetwork's hidden layer.
const HIDDEN: usize = 64;

pub struct Head {
    dim: usize,
    /// Frozen description embeddings, one row of `dim` per applet.
    desc: Vec<f32>,
    /// Per-applet adapter corrections, same shape. Starts at zero.
    delta: Vec<f32>,
    names: Vec<&'static str>,

    // Hypernetwork: dim -> HIDDEN -> dim, ReLU between.
    w1: Vec<f32>, // HIDDEN * dim
    b1: Vec<f32>, // HIDDEN
    w2: Vec<f32>, // dim * HIDDEN
    b2: Vec<f32>, // dim
}

/// Pool the embeddings of a piece of text into one vector.
fn pool(model: &Model, tok: &Tokenizer, text: &str, dim: usize) -> Vec<f32> {
    let ids = tok.encode(text, false, false);
    let mut v = vec![0.0f32; dim];
    // A quantised embedding table has no f32 row to borrow, so rows are
    // dequantised into a scratch buffer rather than returned by reference.
    let mut row = vec![0.0f32; dim];
    for id in &ids {
        model.embed_into(*id, &mut row);
        for (acc, e) in v.iter_mut().zip(row.iter()) {
            *acc += *e;
        }
    }
    if !ids.is_empty() {
        let k = 1.0 / ids.len() as f32;
        for a in v.iter_mut() {
            *a *= k;
        }
    }
    v
}

impl Head {
    pub fn for_applets(model: &Model, tok: &Tokenizer, rng: &mut Rng) -> Self {
        let dim = model.cfg.dim;
        let mut desc = Vec::new();
        let mut names = Vec::new();

        for a in sysbox::APPLETS {
            let mut text = String::from(a.name);
            text.push(' ');
            text.push_str(a.help);
            desc.extend_from_slice(&pool(model, tok, &text, dim));
            names.push(a.name);
        }

        let n = names.len();
        // Small symmetric init, scaled by fan-in. Not zero: identical rows
        // would receive identical gradients forever and the hidden layer would
        // never differentiate.
        let mut init = |count: usize, fan_in: usize| -> Vec<f32> {
            let scale = 1.0 / tensor::sqrtf(fan_in as f32);
            (0..count).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale * 0.1).collect()
        };

        Self {
            dim,
            delta: vec![0.0; n * dim],
            w1: init(HIDDEN * dim, dim),
            b1: vec![0.0; HIDDEN],
            w2: init(dim * HIDDEN, HIDDEN),
            b2: vec![0.0; dim],
            desc,
            names,
        }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn name(&self, i: usize) -> &'static str {
        self.names[i]
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| *n == name)
    }

    /// Number of trainable parameters. The base model's 260,032 are not here.
    pub fn params(&self) -> usize {
        self.delta.len() + self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    fn desc_of(&self, i: usize) -> &[f32] {
        &self.desc[i * self.dim..(i + 1) * self.dim]
    }

    /// Hypernetwork forward. Returns the correction and the hidden activation,
    /// which the backward pass needs.
    fn hyper(&self, d: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0f32; HIDDEN];
        for h in 0..HIDDEN {
            let mut z = self.b1[h];
            let w = &self.w1[h * self.dim..(h + 1) * self.dim];
            for j in 0..self.dim {
                z += w[j] * d[j];
            }
            a[h] = if z > 0.0 { z } else { 0.0 }; // ReLU
        }
        let mut r = vec![0.0f32; self.dim];
        for o in 0..self.dim {
            let mut v = self.b2[o];
            let w = &self.w2[o * HIDDEN..(o + 1) * HIDDEN];
            for h in 0..HIDDEN {
                v += w[h] * a[h];
            }
            r[o] = v;
        }
        (r, a)
    }

    /// The effective classifier row for applet `i`.
    pub fn row(&self, i: usize) -> Vec<f32> {
        let d = self.desc_of(i);
        let (r, _) = self.hyper(d);
        let base = i * self.dim;
        (0..self.dim).map(|j| d[j] + r[j] + self.delta[base + j]).collect()
    }

    /// A row for an applet that does not exist yet, from its description alone.
    ///
    /// This is the whole point of the hypernetwork. There is no `delta` to add
    /// because nothing has ever been learned about this applet specifically --
    /// only about descriptions in general.
    pub fn row_for_text(&self, model: &Model, tok: &Tokenizer, text: &str) -> Vec<f32> {
        let d = pool(model, tok, text, self.dim);
        let (r, _) = self.hyper(&d);
        (0..self.dim).map(|j| d[j] + r[j]).collect()
    }

    /// Score `hidden` against the applets named by `allowed`.
    ///
    /// Only those rows are scored, so permission is enforced by omission
    /// exactly as the decoding grammar enforces it: a forbidden applet has no
    /// logit at all, and no sampling outcome reaches it.
    pub fn logits(&self, hidden: &[f32], allowed: &[usize]) -> Vec<f32> {
        allowed
            .iter()
            .map(|&i| dot(&self.row(i), hidden))
            .collect()
    }

    /// One SGD step of cross-entropy.
    ///
    /// `probs` is the softmax over `allowed` in that order, and `target_slot`
    /// indexes into `allowed` rather than into the applet table -- the two
    /// coincide only when everything is permitted, and conflating them trains
    /// the wrong row whenever they do not.
    ///
    /// Both the shared hypernetwork and the per-applet delta are updated from
    /// the same gradient. dL/drow_i is `g_i * x`; delta takes it directly, and
    /// the rest is chain rule back through two linear layers and a ReLU.
    pub fn learn(
        &mut self,
        hidden: &[f32],
        allowed: &[usize],
        probs: &[f32],
        target_slot: usize,
        lr: f32,
    ) {
        let dim = self.dim;

        // Accumulate hypernetwork gradients across the whole step before
        // applying them: w1 and w2 are shared, so applying per-applet would
        // make the result depend on the order of `allowed`.
        let mut gw1 = vec![0.0f32; self.w1.len()];
        let mut gb1 = vec![0.0f32; HIDDEN];
        let mut gw2 = vec![0.0f32; self.w2.len()];
        let mut gb2 = vec![0.0f32; dim];

        for (slot, &i) in allowed.iter().enumerate() {
            let g = probs[slot] - if slot == target_slot { 1.0 } else { 0.0 };
            if g == 0.0 {
                continue;
            }

            // dL/drow_i, shared by every path below.
            let mut grow = vec![0.0f32; dim];
            for j in 0..dim {
                grow[j] = g * hidden[j];
            }

            // The adapter sits directly on the row.
            let base = i * dim;
            for j in 0..dim {
                self.delta[base + j] -= lr * grow[j];
            }

            // Back through the hypernetwork. `desc` is frozen, so the chain
            // stops at w1 and never reaches the model.
            let d = self.desc_of(i).to_vec();
            let (_, a) = self.hyper(&d);

            // r = w2 @ a + b2
            for o in 0..dim {
                gb2[o] += grow[o];
                for h in 0..HIDDEN {
                    gw2[o * HIDDEN + h] += grow[o] * a[h];
                }
            }

            // da = w2^T @ grow, then through the ReLU.
            for h in 0..HIDDEN {
                let mut da = 0.0f32;
                for o in 0..dim {
                    da += self.w2[o * HIDDEN + h] * grow[o];
                }
                // ReLU': a[h] > 0 exactly when the pre-activation was positive.
                if a[h] <= 0.0 {
                    continue;
                }
                gb1[h] += da;
                for j in 0..dim {
                    gw1[h * dim + j] += da * d[j];
                }
            }
        }

        // The hypernetwork is shared, so its gradient is a sum over every
        // applet scored -- roughly n times the magnitude of any single row's.
        // Applying the same rate to both makes the shared parameters diverge
        // while the deltas are still creeping, which shows up as training
        // accuracy *falling* during training.
        let hyper_lr = lr / allowed.len().max(1) as f32;

        for (p, g) in self.w1.iter_mut().zip(gw1.iter()) {
            *p -= hyper_lr * *g;
        }
        for (p, g) in self.b1.iter_mut().zip(gb1.iter()) {
            *p -= hyper_lr * *g;
        }
        for (p, g) in self.w2.iter_mut().zip(gw2.iter()) {
            *p -= hyper_lr * *g;
        }
        for (p, g) in self.b2.iter_mut().zip(gb2.iter()) {
            *p -= hyper_lr * *g;
        }
    }

}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = tensor::sqrtf(dot(a, a)) * tensor::sqrtf(dot(b, b));
    if d == 0.0 {
        0.0
    } else {
        dot(a, b) / d
    }
}

// --- the corpus ---------------------------------------------------------
//
// Examples live in the namespace, at /ai/train, as "applet<tab>task" blobs.
// That is not decoration. It makes the training set a content-addressed object
// like everything else: `snap` versions it, `back` restores it, and copying a
// corpus between snapshots costs nothing. The system's own filesystem is the
// library it learns from, and a bad training run is undone by `back` rather
// than by having kept a copy somewhere.

pub const CORPUS: &str = "/ai/train";

pub struct Example {
    pub applet: String,
    pub task: String,
}

pub fn record(applet: &str, task: &str) -> bool {
    let n = sysbox::children(CORPUS).len();
    let mut body = String::from(applet);
    body.push('\t');
    body.push_str(task);

    let mut path = String::from(CORPUS);
    path.push('/');
    // Zero-padded, so sorted order in the directory is insertion order.
    let mut digits = [b'0'; 4];
    let mut v = n;
    for d in digits.iter_mut().rev() {
        *d = b'0' + (v % 10) as u8;
        v /= 10;
    }
    path.push_str(core::str::from_utf8(&digits).unwrap_or("0000"));
    sysbox::write_text(&path, &body)
}

pub fn examples() -> Vec<Example> {
    let mut out = Vec::new();
    for name in sysbox::children(CORPUS) {
        let mut path = String::from(CORPUS);
        path.push('/');
        path.push_str(&name);
        let Some(bytes) = sysbox::read_blob(&path) else { continue };
        let Ok(text) = core::str::from_utf8(&bytes) else { continue };
        let text = text.trim_end_matches('\n');
        let mut parts = text.splitn(2, '\t');
        let (Some(applet), Some(task)) = (parts.next(), parts.next()) else {
            continue;
        };
        out.push(Example { applet: String::from(applet), task: String::from(task) });
    }
    out
}


// --- corpus bundles -----------------------------------------------------
//
// `teach` records one example at a time and `corpus.rs` compiles a fixed set
// in at build time. Neither lets a corpus be *replaced* on a running machine,
// which is what adapter training needs: the trainer's material has to be able
// to grow without a rebuild, and to be snapshotted and undone like everything
// else in the namespace.
//
// A bundle is that transfer, in one file. The kernel pulls host files in with
// `fat get <esp-path> <namespace-path>`, one command each -- four hundred
// examples one at a time is four hundred commands over a serial line -- so the
// examples travel as one blob and are unpacked here.
//
// Written by `tools/dataset.py --blobs`. The format is documented there in
// full; the invariants worth restating on this side are that every field is
// length-prefixed, that the reader *walks and never seeks*, and that it must
// land exactly on the last byte. There are no names or shapes in the body to
// disagree about, so one length read wrongly leaves everything after it as
// perfectly valid records of the wrong text -- the same bargain `v4.rs` makes
// about checkpoints, for the same reason.

/// Split boundaries for an imported corpus, as text: `train val_end count`.
///
/// A sibling of the corpus directory rather than a member of it, because
/// `children(CORPUS)` *is* the example list and a boundary file living inside
/// it would be read back as an example whose applet is "357".
pub const SPLITS: &str = "/ai/train.split";

const BUNDLE_MAGIC: &[u8; 8] = b"GLADOSC1";
const BUNDLE_HEADER: usize = 24;

pub struct Bundle {
    pub records: Vec<(String, Vec<u8>)>,
    /// Records `[0, train)` train, `[train, val_end)` validate, the rest test.
    pub train: usize,
    pub val_end: usize,
}

#[derive(Debug, PartialEq)]
pub enum BundleError {
    NotABundle,
    /// A length prefix ran off the end, at this record index.
    Truncated(usize),
    /// A field claimed more bytes than the blob has, at this record index.
    Overrun(usize),
    /// Bytes left over after the declared record count.
    Trailing(usize),
    /// `train <= val_end <= count` does not hold.
    Boundaries,
    /// A name that is not a single namespace leaf, at this record index.
    BadName(usize),
}

/// Parse a bundle without touching the namespace.
///
/// Kept separate from `import_bundle` so the boot self-test can drive every
/// rejection path without writing a corpus into the live tree to do it.
pub fn parse_bundle(blob: &[u8]) -> Result<Bundle, BundleError> {
    if blob.len() < BUNDLE_HEADER || &blob[..8] != BUNDLE_MAGIC {
        return Err(BundleError::NotABundle);
    }
    let u32_at = |o: usize| {
        u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) as usize
    };
    let (count, train, val_end) = (u32_at(8), u32_at(12), u32_at(16));
    if !(train <= val_end && val_end <= count) {
        return Err(BundleError::Boundaries);
    }

    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    let mut off = BUNDLE_HEADER;
    for i in 0..count {
        let mut field: [&[u8]; 2] = [&[], &[]];
        for f in field.iter_mut() {
            if off + 4 > blob.len() {
                return Err(BundleError::Truncated(i));
            }
            let n = u32_at(off);
            off += 4;
            if off + n > blob.len() {
                return Err(BundleError::Overrun(i));
            }
            *f = &blob[off..off + n];
            off += n;
        }
        let Ok(name) = core::str::from_utf8(field[0]) else {
            return Err(BundleError::BadName(i));
        };
        // A leaf name, never a path. The alternative -- letting a bundle name
        // its own destinations -- is a corpus file with write access to the
        // whole namespace, and no corpus has a reason to want one.
        if name.is_empty() || name.contains('/') || name == "." || name == ".." {
            return Err(BundleError::BadName(i));
        }
        records.push((String::from(name), field[1].to_vec()));
    }
    if off != blob.len() {
        return Err(BundleError::Trailing(blob.len() - off));
    }
    Ok(Bundle { records, train, val_end })
}

/// Unpack a bundle into `dir`, replacing whatever was there, and write the
/// split boundaries beside it.
///
/// Replacing rather than appending is the only coherent choice: the boundaries
/// are *positions*, so a bundle merged into an existing corpus would describe
/// a slice of a corpus that no longer exists, and the held-out number would be
/// computed over the wrong examples while still looking like a number.
pub fn import_bundle(dir: &str, blob: &[u8]) -> Result<usize, BundleError> {
    let b = parse_bundle(blob)?;

    for name in sysbox::children(dir) {
        let mut path = String::from(dir);
        path.push('/');
        path.push_str(&name);
        sysbox::detach(&path);
    }

    let mut n = 0usize;
    for (name, body) in b.records {
        let mut path = String::from(dir);
        path.push('/');
        path.push_str(&name);
        if sysbox::write_blob(&path, body) {
            n += 1;
        }
    }

    let mut splits = String::new();
    write_usize(&mut splits, b.train);
    splits.push(' ');
    write_usize(&mut splits, b.val_end);
    splits.push(' ');
    write_usize(&mut splits, n);
    splits.push('\n');
    let mut spath = String::from(dir);
    spath.push_str(".split");
    sysbox::write_text(&spath, &splits);

    Ok(n)
}

fn write_usize(out: &mut String, mut v: usize) {
    if v == 0 {
        out.push('0');
        return;
    }
    let mut digits = [0u8; 20];
    let mut n = 0;
    while v > 0 {
        digits[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    while n > 0 {
        n -= 1;
        out.push(digits[n] as char);
    }
}

/// The corpus split boundaries: `(train, val_end, len)`.
///
/// The compiled constants unless a bundle has been imported over the corpus,
/// in which case the boundaries that came with it. Everything past `len`
/// trains, which is what makes `teach` on a live system safe: an appended
/// example can only ever join the training slice, never silently enter the
/// one number that is supposed to be read once.
///
/// The stored boundaries are used only while they still describe the corpus
/// that is actually present. A recorded count *larger* than the directory
/// means the two have diverged -- somebody detached examples by hand -- and a
/// stale boundary is worse than a conservative one, so it is discarded.
pub fn splits() -> (usize, usize, usize) {
    let n = sysbox::children(CORPUS).len();
    if let Some(bytes) = sysbox::read_blob(SPLITS) {
        if let Ok(text) = core::str::from_utf8(&bytes) {
            let mut it = text.split_whitespace().filter_map(|w| w.parse::<usize>().ok());
            if let (Some(t), Some(v), Some(c)) = (it.next(), it.next(), it.next()) {
                if t <= v && v <= c && c <= n {
                    return (t, v, c);
                }
            }
        }
    }
    (
        super::corpus::SEED_TRAIN,
        super::corpus::SEED_VAL_END,
        super::corpus::SEED.len(),
    )
}

/// Boot self-test for the bundle reader. Five claims.
///
/// Claim 1 is the one that matters and it is deliberately not a round trip
/// through a writer on this side. The writer is `tools/dataset.py` and the
/// reader is here, so a Rust writer checked against the Rust reader would
/// prove only that two halves of the same misunderstanding agree. The fixture
/// below is bytes that generator actually produced, pasted in, which makes the
/// claim "these two tools agree about the format" rather than "this file is
/// self-consistent".
///
/// The rest are the rejections. Every one of them is a length or a name read
/// wrongly, and none of them would announce itself at training time: a corpus
/// shifted by four bytes is still a corpus of valid-looking strings.
pub fn bundle_selftest() -> bool {
    use crate::kprintln;

    // tools/dataset.py: bundle_bytes([ls/list the files, cat/show me /ai/notes],
    // [mv/rename a to b]) -- two training records, one test, val_end == train
    // because a single test record halves to zero.
    const FIXTURE: &[u8] = &[
        0x47, 0x4c, 0x41, 0x44, 0x4f, 0x53, 0x43, 0x31, 0x03, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00, 0x30, 0x30, 0x30, 0x30, 0x11, 0x00, 0x00, 0x00,
        0x6c, 0x73, 0x09, 0x6c, 0x69, 0x73, 0x74, 0x20, 0x74, 0x68, 0x65, 0x20,
        0x66, 0x69, 0x6c, 0x65, 0x73, 0x04, 0x00, 0x00, 0x00, 0x30, 0x30, 0x30,
        0x31, 0x15, 0x00, 0x00, 0x00, 0x63, 0x61, 0x74, 0x09, 0x73, 0x68, 0x6f,
        0x77, 0x20, 0x6d, 0x65, 0x20, 0x2f, 0x61, 0x69, 0x2f, 0x6e, 0x6f, 0x74,
        0x65, 0x73, 0x04, 0x00, 0x00, 0x00, 0x30, 0x30, 0x30, 0x32, 0x10, 0x00,
        0x00, 0x00, 0x6d, 0x76, 0x09, 0x72, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x20,
        0x61, 0x20, 0x74, 0x6f, 0x20, 0x62,
    ];

    let mut ok = true;
    let mut claim = |what: &str, pass: bool| {
        if !pass {
            ok = false;
        }
        kprintln!("  {}  {}", if pass { "ok " } else { "FAIL" }, what);
    };

    match parse_bundle(FIXTURE) {
        Ok(b) => {
            let shape = b.records.len() == 3 && b.train == 2 && b.val_end == 2;
            let names = b.records[0].0 == "0000"
                && b.records[1].0 == "0001"
                && b.records[2].0 == "0002";
            let bodies = b.records[0].1 == b"ls\tlist the files"
                && b.records[1].1 == b"cat\tshow me /ai/notes"
                && b.records[2].1 == b"mv\trename a to b";
            claim("a bundle from tools/dataset.py parses to what it was built from",
                  shape && names && bodies);
        }
        Err(e) => {
            claim("a bundle from tools/dataset.py parses", false);
            kprintln!("    {:?}", e);
        }
    }

    // One byte short: the last body claims more than remains.
    claim(
        "one byte short is caught rather than read as a shorter example",
        parse_bundle(&FIXTURE[..FIXTURE.len() - 1]).err() == Some(BundleError::Overrun(2)),
    );

    // One byte long: every record parsed and something is still left.
    let mut long = Vec::from(FIXTURE);
    long.push(0);
    claim(
        "a byte past the last record is caught",
        parse_bundle(&long).err() == Some(BundleError::Trailing(1)),
    );

    claim(
        "a blob that is not a bundle is refused",
        parse_bundle(b"GLADOSM3 not this one either").err() == Some(BundleError::NotABundle),
    );

    // A name that is a path rather than a leaf. Assembled here by hand from
    // the documented layout, because the generator will not write one.
    let mut escape: Vec<u8> = Vec::new();
    escape.extend_from_slice(BUNDLE_MAGIC);
    for v in [1u32, 1, 1, 0] {
        escape.extend_from_slice(&v.to_le_bytes());
    }
    let name = b"../agent/policy";
    escape.extend_from_slice(&(name.len() as u32).to_le_bytes());
    escape.extend_from_slice(name);
    escape.extend_from_slice(&3u32.to_le_bytes());
    escape.extend_from_slice(b"own");
    claim(
        "a record naming a path outside its directory is refused",
        parse_bundle(&escape).err() == Some(BundleError::BadName(0)),
    );

    // The write side, in a scratch directory so the live corpus is untouched.
    const SCRATCH: &str = "/tmp/bundle-selftest";
    let imported = import_bundle(SCRATCH, FIXTURE);
    let landed = sysbox::children(SCRATCH).len();
    let split = sysbox::read_blob("/tmp/bundle-selftest.split")
        .and_then(|b| core::str::from_utf8(&b).ok().map(String::from))
        .unwrap_or_default();
    claim(
        "importing writes every blob and the boundaries beside them",
        imported.ok() == Some(3) && landed == 3 && split.trim() == "2 2 3",
    );
    for name in sysbox::children(SCRATCH) {
        let mut p = String::from(SCRATCH);
        p.push('/');
        p.push_str(&name);
        sysbox::detach(&p);
    }
    sysbox::detach(SCRATCH);
    sysbox::detach("/tmp/bundle-selftest.split");

    ok
}
