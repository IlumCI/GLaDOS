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

    /// Cosine similarity between two effective rows.
    pub fn similarity(&self, a: usize, b: usize) -> f32 {
        cosine(&self.row(a), &self.row(b))
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

/// A starting corpus, so training has something before anyone has typed
/// anything. Deliberately obvious mappings -- these are what a competent model
/// would get right unprompted.
pub const SEED: &[(&str, &str)] = &[
    ("ls", "list the files"),
    ("ls", "what is in this directory"),
    ("ls", "show me the contents of the folder"),
    ("cat", "read the readme"),
    ("cat", "print the contents of a file"),
    ("cat", "show me what that file says"),
    ("pwd", "where am i"),
    ("pwd", "what directory am i in"),
    ("cd", "go to another directory"),
    ("cd", "change into the folder"),
    ("tree", "list everything recursively"),
    ("tree", "show the whole hierarchy"),
    ("stat", "how big is that file"),
    ("stat", "give me details about a path"),
    ("hash", "what is the address of that object"),
    ("hash", "show me the content hash"),
    ("same", "are these two identical"),
    ("same", "compare two subtrees"),
    ("du", "how much space is used"),
    ("du", "report disk usage"),
    ("find", "search for some text"),
    ("find", "look for a word in the files"),
    ("diff", "compare two snapshots"),
    ("diff", "what changed between versions"),
    ("snaps", "list the snapshots"),
    ("snaps", "show snapshot history"),
    ("fsck", "verify the disk"),
    ("fsck", "check everything is intact"),
    ("mkdir", "make a directory"),
    ("mkdir", "create a new folder"),
    ("write", "save some text to a file"),
    ("write", "create a file with content"),
    ("rm", "delete that name"),
    ("rm", "remove the file"),
    ("cp", "copy it somewhere else"),
    ("mv", "rename that"),
    ("snap", "take a snapshot"),
    ("snap", "commit the current state"),
    ("back", "go back to an earlier snapshot"),
    ("back", "restore the previous version"),
];
