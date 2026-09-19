//! The forest: hierarchical knowledge, read one branch at a time.
//!
//! A tree is a school of thought, branching into specialisations. The point is
//! that a model never loads the whole thing -- it routes to a branch and takes
//! what fits in a budget. This module is the read side of that: what is here,
//! what it costs, and one node at a time.
//!
//! **A node is a head line and a body.** The head is `subject | concept |
//! terms` and is the whole of what an index would hold; the body carries the
//! concept, a method, machine-checkable steps and the original passage. The
//! head is first *so that a reader wanting only the index stops after one
//! line*, which is the property the whole retrieval design rests on.
//!
//! The format is line oriented, `key<space>value`, and a line whose first word
//! is not a key continues the field before it. That is why a body may contain
//! brackets, equals signs and blank lines with no escaping at all, and it is
//! `tools/forest.py` that writes it.
//!
//! ### Never `content_hash`, and that is not a preference
//!
//! `tree::content_hash` is a full recursive walk that SHA-256s every blob, and
//! `cmd_ls` calls it **per child** (`sysbox/mod.rs:1189`). So `ls` on a forest
//! directory hashes the entire forest, and `stat` and `du` walk it twice. This
//! module uses `children` and `blob_len` only, which are a `String` clone per
//! entry and a length read. Anything added here that reaches for a hash has
//! turned a listing into a full corpus rehash, and the only symptom is that
//! the machine appears to stop.
//!
//! ### Where it lives, and why not `/ai/forest`
//!
//! `/pkg/forest`, because a forest arrives as a `GLADOSPK` package and that is
//! where `pkg::install` puts packages (`pkg.rs:52`). A package already grafts
//! nested paths with `..` and absolute paths refused (`pkg.rs:133-135`), which
//! is exactly the property a corpus import needs, so there is no second import
//! path here and there should not be one.

use alloc::string::String;
use alloc::vec::Vec;

use crate::kprintln;
use crate::store::cas;
use crate::sysbox;

/// Where `pkg add` leaves a package named `forest`.
pub const ROOT: &str = "/pkg/forest";

/// Longest head line rendered by `forest tree`. Heads are one line by
/// construction; this bounds a malformed one rather than trusting the writer.
const HEAD_CLIP: usize = 100;

pub struct Census {
    pub trees: usize,
    pub branches: usize,
    pub nodes: usize,
    pub bytes: usize,
    /// Widest directory found. Watched because `tree::put` inserts at a sorted
    /// index and `ls` hashes per child, so width is punished twice where depth
    /// is free to 32 levels.
    pub widest: usize,
    pub deepest: usize,
}

/// Is this path a node rather than a branch?
///
/// Asked with `blob_len`, which answers `None` for a directory as well as for
/// a missing path -- deliberately, per its own doc. That conflation is fine
/// here because the walk already knows the path exists.
fn is_node(path: &str) -> Option<usize> {
    sysbox::blob_len(path)
}

fn walk(path: &str, depth: usize, c: &mut Census) {
    let kids = sysbox::children(path);
    if kids.is_empty() {
        return;
    }
    if kids.len() > c.widest {
        c.widest = kids.len();
    }
    if depth > c.deepest {
        c.deepest = depth;
    }
    let mut had_node = false;
    for name in kids {
        let mut sub = String::from(path);
        sub.push('/');
        sub.push_str(&name);
        match is_node(&sub) {
            Some(n) => {
                c.nodes += 1;
                c.bytes += n;
                had_node = true;
            }
            None => walk(&sub, depth + 1, c),
        }
    }
    if had_node {
        c.branches += 1;
    }
}

/// Census any root, so the claims below can measure a forest they built
/// themselves rather than whatever happens to be installed.
pub fn census_at(root: &str) -> Census {
    let mut c = Census {
        trees: 0,
        branches: 0,
        nodes: 0,
        bytes: 0,
        widest: 0,
        deepest: 0,
    };
    for t in trees_at(root) {
        c.trees += 1;
        let mut p = String::from(root);
        p.push('/');
        p.push_str(&t);
        walk(&p, 1, &mut c);
    }
    c
}

pub fn census() -> Census {
    census_at(ROOT)
}

pub fn trees_at(root: &str) -> Vec<String> {
    sysbox::children(root)
        .into_iter()
        .filter(|n| {
            let mut p = String::from(root);
            p.push('/');
            p.push_str(n);
            // A receipt is a blob beside the trees; a tree is a directory.
            sysbox::blob_len(&p).is_none()
        })
        .collect()
}

pub fn trees() -> Vec<String> {
    trees_at(ROOT)
}

/// The first line of a node, which is the whole of what an index holds.
///
/// This still reads the whole blob, because `sysbox::read_blob` clones and
/// there is no ranged read at the namespace layer -- `cas::read_blocks` exists
/// and is not wired through (`store/cas.rs:285`). Wiring it is what makes an
/// index affordable over a forest that does not fit in memory, and it is the
/// first thing to change when one does not.
pub fn head_of(path: &str) -> Option<String> {
    let b = sysbox::read_blob(path)?;
    let text = String::from_utf8(b).ok()?;
    let first = text.lines().next()?;
    let rest = first.strip_prefix("head ").unwrap_or(first);
    Some(String::from(rest))
}

pub fn report() {
    let c = census();
    if c.nodes == 0 {
        kprintln!("  no forest at {} -- 'pkg add' a forest package first", ROOT);
        return;
    }
    kprintln!("  {} tree(s), {} branch(es), {} node(s)", c.trees, c.branches, c.nodes);
    for t in trees() {
        let mut p = String::from(ROOT);
        p.push('/');
        p.push_str(&t);
        let mut sub = Census { trees: 0, branches: 0, nodes: 0, bytes: 0, widest: 0, deepest: 0 };
        walk(&p, 1, &mut sub);
        kprintln!("  {:<24} {:>6} node(s)  {:>9} B", t, sub.nodes, sub.bytes);
    }
}

pub fn report_tree(name: &str) {
    let mut p = String::from(ROOT);
    p.push('/');
    p.push_str(name);
    if sysbox::children(&p).is_empty() {
        kprintln!("  no tree '{}' -- 'forest' lists them", name);
        return;
    }
    let mut stack = alloc::vec![(p, 1usize)];
    let mut shown = 0usize;
    while let Some((dir, depth)) = stack.pop() {
        let kids = sysbox::children(&dir);
        let mut nodes = 0usize;
        let mut subdirs: Vec<String> = Vec::new();
        for k in kids {
            let mut sub = dir.clone();
            sub.push('/');
            sub.push_str(&k);
            if is_node(&sub).is_some() {
                nodes += 1;
            } else {
                subdirs.push(sub);
            }
        }
        if nodes > 0 {
            let short = dir.strip_prefix(ROOT).unwrap_or(&dir);
            kprintln!("  {:<44} {:>5} node(s)", short, nodes);
            shown += nodes;
        }
        // Reversed, so popping walks the branches in name order.
        for s in subdirs.into_iter().rev() {
            stack.push((s, depth + 1));
        }
    }
    kprintln!("  {} node(s) under {}", shown, name);
}

pub fn show(path: &str) {
    let full = if path.starts_with('/') {
        String::from(path)
    } else {
        let mut p = String::from(ROOT);
        p.push('/');
        p.push_str(path);
        p
    };
    match sysbox::read_blob(&full) {
        Some(b) => {
            let text = String::from_utf8_lossy(&b);
            for line in text.lines() {
                kprintln!("  {}", line);
            }
        }
        None => kprintln!("  no node at {}", full),
    }
}

/// What the forest costs, and what an index over it would cost.
///
/// The second figure is the one that decides whether bodies can stay in the
/// namespace. Getting it means reading every node, because the head is only
/// cheap to reach once something stores it separately -- so this command is
/// deliberately expensive and says so.
pub fn cost() {
    let c = census();
    if c.nodes == 0 {
        kprintln!("  no forest at {}", ROOT);
        return;
    }
    let mut heads = 0usize;
    for t in trees() {
        let mut p = String::from(ROOT);
        p.push('/');
        p.push_str(&t);
        heads += head_bytes(&p);
    }
    kprintln!("  {} node(s), {} B of bodies resident", c.nodes, c.bytes);
    kprintln!("  {} B of head lines -- the *text* an index holds", heads);
    let pct = if c.bytes == 0 { 0 } else { heads * 100 / c.bytes };
    kprintln!("  heads are {}% of the forest, so {}% could leave memory", pct, 100 - pct);
    // That second figure was quoted as what an index costs and it is not. A
    // usable index also has to know each node's path and where its body lives,
    // which `forest index` measures against a real one: 1,577,626 B of heads
    // became 2,602,495 B of index over 8,913 nodes, so 20% became 33%. Still
    // two thirds of the corpus leaving memory, and not four fifths.
    kprintln!("  'forest index' is the figure including paths and chunk refs");
    kprintln!("  widest directory {}, deepest {} level(s) under the root", c.widest, c.deepest);
    kprintln!("  every node was read to measure this; nothing caches it yet");
}

fn head_bytes(dir: &str) -> usize {
    let mut total = 0usize;
    for name in sysbox::children(dir) {
        let mut sub = String::from(dir);
        sub.push('/');
        sub.push_str(&name);
        match is_node(&sub) {
            Some(_) => {
                if let Some(h) = head_of(&sub) {
                    total += h.len();
                }
            }
            None => total += head_bytes(&sub),
        }
    }
    total
}

// --- the index, and bodies that need not be resident ---------------------

/// A forest's heads in memory and its bodies on disk.
///
/// This is the shape the whole retrieval design was described in: the head is
/// `subject | concept | terms` and is what a router pools over, the body is
/// everything else and is what reaches the context. Keeping the first and not
/// the second is what makes a corpus larger than memory usable at all, and the
/// measured split says how much that buys -- `forest cost` put heads at 20% of
/// an 8,913-node forest, so four fifths of it can leave.
pub struct Index {
    pub paths: Vec<String>,
    pub heads: Vec<String>,
    pub bodies: Vec<cas::ChunkRef>,
    /// Nodes whose head would not read back.
    ///
    /// Counted rather than ignored. A node dropping out of an index silently
    /// is a node the router can never reach, and an index that is quietly
    /// short by a hundred looks exactly like one that is complete.
    pub skipped: usize,
}

impl Index {
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// What the index costs in memory.
    ///
    /// The strings plus a chunk reference each. `ChunkRef` is a hash, an LBA
    /// and a length, so 48 bytes -- taken from `size_of` rather than written
    /// down, since a field added to it would otherwise make this quietly wrong
    /// in the direction that matters.
    pub fn resident(&self) -> usize {
        let mut n = self.bodies.len() * core::mem::size_of::<cas::ChunkRef>();
        for h in &self.heads {
            n += h.len();
        }
        for q in &self.paths {
            n += q.len();
        }
        n
    }

    /// What the bodies would cost if they were resident too.
    pub fn body_bytes(&self) -> u64 {
        self.bodies.iter().map(|r| r.len).sum()
    }

    /// One body, read off the disk on demand.
    pub fn body(&self, i: usize) -> Option<Vec<u8>> {
        sysbox::stored::read_all(self.bodies.get(i)?)
    }

    pub fn position(&self, path: &str) -> Option<usize> {
        self.paths.iter().position(|q| q == path)
    }
}

/// Build an index from the store alone.
///
/// **Nothing here reads the working tree**, and that is the claim rather than
/// an implementation note: a forest that was never restored into memory -- or
/// one too large to be -- is still indexable. The cost is one directory chunk
/// per branch and one block per node, against `read_node`'s whole-blob read
/// and SHA-256 per node, which is what makes restoring a large forest take
/// longer than the boot deadline.
pub fn index_at(root: &str) -> Index {
    let mut ix =
        Index { paths: Vec::new(), heads: Vec::new(), bodies: Vec::new(), skipped: 0 };
    for (path, cr) in sysbox::stored::locate_under(root) {
        let head = match sysbox::stored::head_line(&cr) {
            Some(h) => h,
            None => {
                ix.skipped += 1;
                continue;
            }
        };
        ix.paths.push(path);
        ix.heads.push(head);
        ix.bodies.push(cr);
    }
    ix
}

pub fn index_report() {
    if !sysbox::stored::available() {
        kprintln!("  no store mounted -- 'store init', then 'snap', and this reads from that");
        return;
    }
    let mhz = crate::time::tsc_mhz().max(1);
    let t0 = crate::time::rdtsc();
    let ix = index_at(ROOT);
    let us = (crate::time::rdtsc() - t0) / mhz;
    if ix.is_empty() {
        kprintln!("  nothing under {} in the last snapshot -- 'snap' after importing", ROOT);
        return;
    }
    let mem = ix.resident();
    let disk = ix.body_bytes();
    kprintln!("  {} node(s) indexed from the store, no body resident", ix.len());
    kprintln!("  {} B of index in memory, {} B of bodies left on disk", mem, disk);
    let pct = if disk == 0 { 0 } else { mem as u64 * 100 / disk };
    kprintln!("  the index is {}% of what the bodies weigh", pct);
    kprintln!("  built in {} ms, reading one block per node and no body", us / 1000);
    if ix.skipped > 0 {
        kprintln!("  {} node(s) would not read back and are not in it", ix.skipped);
    }
    let refused = sysbox::stored::contended();
    if refused > 0 {
        kprintln!("  {} ranged read(s) were refused for contention", refused);
    }
}

/// One body, fetched through the store rather than out of the namespace.
///
/// Beside `show` rather than replacing it, deliberately: the two read the same
/// node by different routes, so running both is the cheapest way to see that
/// the ranged path agrees with the resident one.
pub fn show_stored(path: &str) {
    let full = if path.starts_with('/') {
        String::from(path)
    } else {
        let mut p = String::from(ROOT);
        p.push('/');
        p.push_str(path);
        p
    };
    match sysbox::stored::locate(&full) {
        None => kprintln!("  no stored node at {} -- 'snaps' to see what is stored", full),
        Some(cr) => match sysbox::stored::read_all(&cr) {
            None => kprintln!("  the store would not answer for {}", full),
            Some(b) => {
                kprintln!("  {} B at lba {}, read without the namespace", cr.len, cr.lba);
                let text = alloc::string::String::from_utf8_lossy(&b);
                for line in text.lines() {
                    kprintln!("  {}", line);
                }
            }
        },
    }
}

// --- routing -------------------------------------------------------------

/// Build the branch table from the index, store it, and score it.
///
/// **One walk for both.** Pooling every head costs the index -- one block a
/// node -- plus a tokenizer pass; the leave-one-out scoring that follows reuses
/// the vectors already in hand, so the accuracy figure is free rather than a
/// second run of everything.
///
/// The vectors are held for that second pass and then dropped. At dim 576 over
/// nine thousand nodes that is about 20 MB, transient, against a heap that
/// starts at 320 MiB -- worth stating because the alternative is walking the
/// disk twice to save it.
/// The forest as one vector per node, with the subject each belongs to.
///
/// Extracted because the fitted probe has to see *exactly* what the cosine
/// baseline sees. Two walks would put the tokeniser, the document frequencies
/// and the pooling in the difference between them, and the whole point of the
/// comparison is that nothing is in the difference but the classifier.
pub struct Vectorised {
    pub ix: Index,
    pub dim: usize,
    pub vecs: Vec<Vec<f32>>,
    /// Which subject each node belongs to, as an index into `names`.
    pub labels: Vec<usize>,
    pub names: Vec<String>,
}

pub fn vectorise() -> Option<Vectorised> {
    let ix = index_at(ROOT);
    if ix.is_empty() {
        return None;
    }
    // One token list per node over exactly the text the index holds, then the
    // document frequencies, then the vectors those frequencies weight. The
    // order matters: a weight cannot be known until the whole corpus has been
    // seen, so pooling has to come second.
    let built = crate::ai::with_engine(|e| {
        let docs: Vec<Vec<usize>> = ix
            .heads
            .iter()
            .map(|h| crate::ai::lex::tokens(&e.tok, &crate::ai::route::queryable(h)))
            .collect();
        let lex = crate::ai::lex::Lex::build(e.tok.vocab_size(), &docs);
        let vecs: Vec<Vec<f32>> =
            docs.iter().map(|d| crate::ai::lex::pool_ids(&e.model, d, &lex)).collect();
        (e.model.cfg.dim, vecs)
    })?;
    let (dim, vecs) = built;

    let mut names: Vec<String> = Vec::new();
    let mut labels: Vec<usize> = Vec::with_capacity(ix.len());
    for p in ix.paths.iter() {
        let b = crate::ai::route::subject_of(p);
        let li = match names.iter().position(|n| n == b) {
            Some(j) => j,
            None => {
                names.push(String::from(b));
                names.len() - 1
            }
        };
        labels.push(li);
    }
    Some(Vectorised { ix, dim, vecs, labels, names })
}

/// Hold out every `HOLD`-th node.
///
/// **Striding and not a prefix, because the forest is sorted by path** and a
/// path begins with its subject -- so the last fifth of it is whole subjects
/// the fit would never have seen, and both routers would score zero on them
/// for a reason that has nothing to do with either. `train.rs` makes the same
/// choice about `-n` for the same reason.
const HOLD: usize = 5;


/// The fitted router, against the cosine baseline, on the same held-out nodes.
///
/// Phase 3's second half. `route.rs` calls itself "the baseline the fitted
/// probe has to beat", and this is the thing that was supposed to try -- with
/// the plan's own permission to fail written into it: *if it does not, the
/// baseline stands and that is the result.*
///
/// ### Everything that could make the comparison a lie, and what is done
///
/// **One walk.** Both routers score the identical vectors from one
/// `vectorise`, so the tokeniser, the document frequencies and the pooling are
/// not in the difference between them. Only the classifier is.
///
/// **The baseline is rebuilt from the training slice.** `forest embed` scores
/// leave-one-out against a table built from every node, which is free and
/// correct *for a mean*. A probe cannot be refit per node, so it needs a real
/// held-out slice -- and a baseline whose table had seen those nodes would be
/// the leakage this project has already got wrong twice elsewhere.
///
/// **Paired, on the same items.** Two percentages over two sets is not a
/// comparison. `fixed` and `broke` count where they disagree, which is what
/// `godel`'s J1 reads and the only thing that survives a small slice.
///
/// **The separation gate runs first.** `harness::probe_features` exists
/// because `Feature::Hidden` once fitted a probe on features carrying no class
/// information at all, and a fit on collapsed features produces a confident
/// classifier of noise. Same-class against different-class cosine, printed,
/// and the fit is refused when the gap is nothing.
pub fn fit() {
    if !sysbox::stored::available() {
        kprintln!("  no store mounted -- 'store init', then 'snap', and this reads from that");
        return;
    }
    let Some(v) = vectorise() else {
        kprintln!("  nothing under {} in the last snapshot -- 'snap' after importing", ROOT);
        return;
    };
    let classes = v.names.len();
    if classes < 2 {
        kprintln!("  {} subject(s) -- there is nothing to route between", classes);
        return;
    }
    let mhz = crate::time::tsc_mhz().max(1);
    let t0 = crate::time::rdtsc();

    // --- the split ----------------------------------------------------
    let mut train: Vec<usize> = Vec::new();
    let mut held: Vec<usize> = Vec::new();
    for i in 0..v.ix.len() {
        if i % HOLD == HOLD - 1 {
            held.push(i);
        } else {
            train.push(i);
        }
    }
    kprintln!(
        "  {} node(s) over {} subject(s), dim {} -- {} fitted, {} held out",
        v.ix.len(),
        classes,
        v.dim,
        train.len(),
        held.len()
    );
    if held.is_empty() || train.is_empty() {
        kprintln!("  not enough to split");
        return;
    }

    // --- the gate -----------------------------------------------------
    //
    // Sampled rather than exhaustive: the full pairing over seven thousand
    // vectors is twenty-five million cosines, and what is being measured is a
    // mean that a few thousand pairs already pins.
    let (mut same, mut same_n) = (0.0f64, 0usize);
    let (mut diff, mut diff_n) = (0.0f64, 0usize);
    let step = (train.len() / 96).max(1);
    let mut a = 0;
    while a < train.len() {
        let mut b = a + step;
        while b < train.len() {
            let (i, j) = (train[a], train[b]);
            let c = crate::ai::vocab::cosine(&v.vecs[i], &v.vecs[j]) as f64;
            if v.labels[i] == v.labels[j] {
                same += c;
                same_n += 1;
            } else {
                diff += c;
                diff_n += 1;
            }
            b += step;
        }
        a += step;
    }
    let same_m = if same_n > 0 { same / same_n as f64 } else { 0.0 };
    let diff_m = if diff_n > 0 { diff / diff_n as f64 } else { 0.0 };
    let gap = same_m - diff_m;
    kprintln!(
        "  separation   same {}  different {}  gap {}",
        f3(same_m),
        f3(diff_m),
        f3(gap)
    );
    // The number the gate turns on is small on purpose: what it exists to
    // catch is a gap of *nothing*, which is what collapsed features give.
    if gap < 0.005 {
        crate::console::set_color(crate::gfx::console::LTRED);
        kprintln!("  the features carry no subject information, so nothing is fitted");
        crate::console::set_color(crate::gfx::console::LTGRAY);
        kprintln!("  a probe over these would be a confident classifier of noise");
        return;
    }

    // --- the baseline, rebuilt from the training slice only -----------
    let mut t = crate::ai::route::Table {
        dim: v.dim,
        names: v.names.clone(),
        counts: alloc::vec![0u32; classes],
        sums: alloc::vec![0.0f32; classes * v.dim],
        centroid: Vec::new(),
    };
    for &i in train.iter() {
        let bi = v.labels[i];
        for (acc, x) in t.sums[bi * v.dim..(bi + 1) * v.dim].iter_mut().zip(v.vecs[i].iter()) {
            *acc += *x;
        }
        t.counts[bi] += 1;
    }
    // Centred, because `forest embed` measured that as the better setting for
    // top-1 and this has to compare against the baseline at its best rather
    // than against a version of it nobody would run.
    let mut c = alloc::vec![0.0f32; v.dim];
    for &i in train.iter() {
        for (acc, x) in c.iter_mut().zip(v.vecs[i].iter()) {
            *acc += *x;
        }
    }
    for acc in c.iter_mut() {
        *acc /= train.len() as f32;
    }
    t.centroid = c;
    let means = crate::ai::route::means(&t);

    // --- the probe ----------------------------------------------------
    let train_x: Vec<Vec<f32>> = train.iter().map(|&i| v.vecs[i].clone()).collect();
    let train_y: Vec<usize> = train.iter().map(|&i| v.labels[i]).collect();
    let Some(p) = crate::ai::probe::Probe::fit(&train_x, &train_y, classes, LAMBDA) else {
        kprintln!("  the fit did not solve -- singular, or a label out of range");
        return;
    };
    drop(train_x);

    // --- scored on the same items, one after the other ----------------
    let (mut base_ok, mut probe_ok) = (0usize, 0usize);
    let (mut fixed, mut broke) = (0usize, 0usize);
    for &i in held.iter() {
        let want = v.labels[i];
        let mut q = v.vecs[i].clone();
        t.centre(&mut q);
        let b = crate::ai::route::rank(&t, &means, &q)
            .first()
            .map(|(j, _)| *j)
            .unwrap_or(usize::MAX);
        let pr = p.predict(&v.vecs[i]);
        let (bg, pg) = (b == want, pr == want);
        base_ok += bg as usize;
        probe_ok += pg as usize;
        match (bg, pg) {
            (false, true) => fixed += 1,
            (true, false) => broke += 1,
            _ => {}
        }
    }
    let n = held.len() as f64;
    let chance = 100.0 / classes as f64;
    kprintln!(
        "  cosine       top-1 {}%   ({} of {})",
        f1(100.0 * base_ok as f64 / n),
        base_ok,
        held.len()
    );
    kprintln!(
        "  probe        top-1 {}%   ({} of {}), {} parameter(s)",
        f1(100.0 * probe_ok as f64 / n),
        probe_ok,
        held.len(),
        p.params()
    );
    kprintln!("  chance is {}% over {} subject(s)", f1(chance), classes);

    // McNemar over the disagreements, which is the only part of two
    // percentages that carries information when the slice is this size.
    let (f, b) = (fixed as f64, broke as f64);
    let chi = if f + b > 0.0 { (f - b) * (f - b) / (f + b) } else { 0.0 };
    kprintln!(
        "  paired       fixed {}, broke {}, chi {} ({})",
        fixed,
        broke,
        f2(chi),
        if chi >= 3.84 { "past 95%" } else { "inside the noise" }
    );

    let ms = (crate::time::rdtsc() - t0) / (mhz * 1000);
    kprintln!("  fitted and scored in {} ms", ms);

    // --- the verdict, and it is allowed to be no ----------------------
    if probe_ok > base_ok && chi >= 3.84 {
        crate::console::set_color(crate::gfx::console::LTGREEN);
        kprintln!("  the probe beats the baseline, so it is what 'forest route' will use");
        crate::console::set_color(crate::gfx::console::LTGRAY);
        if crate::sysbox::write_blob(PROBE, p.to_bytes()) {
            kprintln!("  {} -- {} byte(s)", PROBE, p.byte_len());
            kprintln!("  'snap' to keep it");
        } else {
            kprintln!("  could not write {}", PROBE);
        }
    } else {
        crate::console::set_color(crate::gfx::console::YELLOW);
        kprintln!("  the baseline stands, and that is the result");
        crate::console::set_color(crate::gfx::console::LTGRAY);
        kprintln!("  nothing was stored. A probe that does not beat a mean is a probe");
        kprintln!("  whose 9,000 parameters are buying a difference nobody measured.");
    }
    crate::console::set_color(crate::gfx::console::WHITE);
}

/// A decimal, without asking core for float formatting.
///
/// The rest of this file prints tenths as `{}.{}` over a scaled integer, which
/// is the same decision: the numbers here are percentages and cosines, and a
/// fixed number of places is what makes two runs comparable at a glance.
fn dp(x: f64, places: u32) -> String {
    let scale = 10i64.pow(places);
    let neg = x < 0.0;
    let a = if neg { -x } else { x };
    let v = (a * scale as f64 + 0.5) as i64;
    alloc::format!(
        "{}{}.{:0w$}",
        if neg { "-" } else { "" },
        v / scale,
        v % scale,
        w = places as usize
    )
}

fn f1(x: f64) -> String {
    dp(x, 1)
}
fn f2(x: f64) -> String {
    dp(x, 2)
}
fn f3(x: f64) -> String {
    dp(x, 3)
}

/// Ridge, and the same value every other fit in this tree uses.
///
/// Not swept. A sweep would need a third slice to choose on, and choosing on
/// the held-out one is exactly the leakage the split exists to prevent -- so
/// the constant stays until somebody wants the sweep enough to pay for the
/// slice.
const LAMBDA: f32 = 1.0;

/// Where a probe lives once one has earned its place.
pub const PROBE: &str = "/ai/route/probe";

pub fn embed() {
    if !sysbox::stored::available() {
        kprintln!("  no store mounted -- 'store init', then 'snap', and this reads from that");
        return;
    }
    let ix = index_at(ROOT);
    if ix.is_empty() {
        kprintln!("  nothing under {} in the last snapshot -- 'snap' after importing", ROOT);
        return;
    }
    let mhz = crate::time::tsc_mhz().max(1);
    let t0 = crate::time::rdtsc();

    // One token list per node over exactly the text the index holds, then the
    // document frequencies, then the vectors those frequencies weight. The
    // order matters: a weight cannot be known until the whole corpus has been
    // seen, so pooling has to come second.
    let built = crate::ai::with_engine(|e| {
        let docs: Vec<Vec<usize>> = ix
            .heads
            .iter()
            .map(|h| crate::ai::lex::tokens(&e.tok, &crate::ai::route::queryable(h)))
            .collect();
        let lex = crate::ai::lex::Lex::build(e.tok.vocab_size(), &docs);
        let vecs: Vec<Vec<f32>> =
            docs.iter().map(|d| crate::ai::lex::pool_ids(&e.model, d, &lex)).collect();
        (e.model.cfg.dim, lex, vecs)
    });
    let (dim, lex, vecs) = match built {
        Some(v) => v,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };

    let mut t = crate::ai::route::Table {
        dim,
        names: Vec::new(),
        counts: Vec::new(),
        sums: Vec::new(),
        centroid: Vec::new(),
    };
    let mut owner: Vec<usize> = Vec::with_capacity(ix.len());
    for i in 0..ix.len() {
        let b = crate::ai::route::subject_of(&ix.paths[i]);
        let bi = match t.find(b) {
            Some(j) => j,
            None => {
                t.names.push(String::from(b));
                t.counts.push(0);
                t.sums.resize(t.names.len() * dim, 0.0);
                t.names.len() - 1
            }
        };
        for (acc, x) in t.sums[bi * dim..(bi + 1) * dim].iter_mut().zip(vecs[i].iter()) {
            *acc += *x;
        }
        t.counts[bi] += 1;
        owner.push(bi);
    }

    // The fold is reported rather than assumed. If `tools/forest.py` ever
    // spells a shard differently, this number drops to zero on a forest that
    // plainly has them, and somebody sees it.
    let folded = ix
        .paths
        .iter()
        .filter(|p| crate::ai::route::subject_of(p) != crate::ai::route::branch_of(p))
        .count();
    kprintln!(
        "  {} node(s) over {} subject(s), dim {} -- {} node(s) had a shard folded in",
        ix.len(),
        t.len(),
        t.dim,
        folded
    );

    // **Scored twice over identical data, which is the only way the comparison
    // says anything.** Same nodes, same vectors, same leave-one-out; one
    // constant shift between the two runs. Measuring centring on a second walk
    // would put the host's scheduler and the disk in the difference.
    let plain = score_loo(&t, &vecs, &owner);

    // The centroid is the mean of every node vector -- the direction all
    // English text shares, which is what leaves the cosines bunched.
    let mut c = alloc::vec![0.0f32; t.dim];
    for v in &vecs {
        for (a, x) in c.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    let n = vecs.len().max(1) as f32;
    for a in c.iter_mut() {
        *a /= n;
    }
    t.centroid = c;
    let centred = score_loo(&t, &vecs, &owner);

    // **Chosen on top-1, and that choice is not obviously the right one.**
    // Measured here: centring takes top-1 from 17.0% to 28.6% and takes top-3
    // from 47.2% *down* to 44.2%. So it sharpens the first answer and costs a
    // little of the first three, and which of those matters depends on a budget
    // that does not exist yet -- a retrieval loading one subject wants the
    // first number and one loading three wants the second.
    //
    // Top-1 wins by 11.6 points where top-3 loses by 3.0, so it is on. The
    // figures are printed both ways every run precisely so whoever builds the
    // budgeted retrieval can revisit this with evidence rather than re-deriving
    // it, and a forest with different subjects may well answer differently.
    if centred.0 < plain.0 {
        t.centroid.clear();
    }

    let bytes = t.encode().len();
    let wrote = crate::ai::route::store(&t);

    // The per-node vectors, beside the subject table. Written raw: centring is
    // the subject table's centroid and it is applied at load, so one table can
    // be rebuilt without the other going stale in a way nothing would notice.
    let nodes = crate::ai::recall::Nodes {
        dim: t.dim,
        paths: ix.paths.clone(),
        vecs: vecs.iter().flat_map(|v| v.iter().copied()).collect(),
    };
    let nbytes = nodes.encode().len();
    let nwrote = crate::ai::recall::store(&nodes);
    let lbytes = lex.encode().len();
    let lwrote = crate::ai::recall::store_lex(&lex);
    // Whatever was cached came from the table that was just replaced.
    crate::ai::recall::forget();

    let us = (crate::time::rdtsc() - t0) / mhz;

    // Tenths of a percent in whole numbers: no float formatting to get wrong
    // and the figure reads the same either way.
    let per = |a: usize, b: usize| if b == 0 { 0 } else { a * 1000 / b };
    let show = |what: &str, r: (usize, usize, usize, usize)| {
        let (o1, o3) = (per(r.0, r.2), per(r.1, r.2));
        kprintln!(
            "  {:<10} top-1 {}.{}%   top-3 {}.{}%",
            what,
            o1 / 10,
            o1 % 10,
            o3 / 10,
            o3 % 10
        );
    };
    let ch = per(1, t.len().max(1));
    kprintln!(
        "  leave-one-out over {} node(s), {} alone in their subject",
        plain.2,
        plain.3
    );
    show("plain", plain);
    show("centred", centred);
    kprintln!("  {}.{}% is chance over {} subject(s)", ch / 10, ch % 10, t.len());
    if wrote {
        kprintln!(
            "  {} B of table at {}, centring {}",
            bytes,
            crate::ai::route::TABLE,
            if t.centroid.is_empty() { "off -- it did not win" } else { "on" }
        );
    } else {
        kprintln!("  the table would not write to {}", crate::ai::route::TABLE);
    }
    if nwrote && lwrote {
        kprintln!(
            "  {} B of node vectors and {} B of postings",
            nbytes,
            lbytes
        );
    } else {
        kprintln!("  the node vectors or postings would not write");
    }
    kprintln!("  built and scored in {} ms, no forward pass anywhere", us / 1000);
}

// --- the measurement that says whether retrieval retrieves ----------------

/// The keys a node body uses, in `tools/forest.py`'s order.
///
/// Listed rather than inferred. The format is line oriented and a line whose
/// first word is not a key *continues* the field before it, which is what lets
/// a body carry brackets, equals signs and blank lines with no escaping -- so a
/// reader that treated any first word as a key would end `text` at the first
/// line of prose beginning with a noun.
const KEYS: &[&str] = &["head", "kind", "source", "concept", "method", "check", "answer", "text"];

/// One field of a node body, continuation lines included.
pub fn field(body: &str, want: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in body.lines() {
        let key = line.split(' ').next().unwrap_or("");
        if KEYS.contains(&key) && line.len() > key.len() {
            inside = key == want;
            if inside {
                out.push_str(&line[key.len() + 1..]);
            }
            continue;
        }
        if inside {
            out.push('\n');
            out.push_str(line);
        }
    }
    out
}

/// A query that shares no text with what the index holds.
///
/// **This is what makes the benchmark mean anything.** The index is built from
/// `concept` and the head's terms; `concept` is the *first sentence* of `text`,
/// so everything after it -- the rest of the question, the working, the options
/// -- is prose from the same node that the index has never seen. Retrieving the
/// node from it is a real known-item task with one right answer in nine
/// thousand, and no string in common to shortcut it.
///
/// Nothing is returned when the tail would not be disjoint, and the caller
/// counts those. A benchmark that quietly drops the awkward half is measuring
/// the easy half and reporting it as the whole.
pub fn held_out(body: &str) -> Option<String> {
    let text = field(body, "text");
    let concept = field(body, "concept");
    let t = text.trim();
    let c = concept.trim();
    if c.is_empty() || t.is_empty() {
        return None;
    }
    let rest = t.strip_prefix(c)?.trim();
    if rest.len() < 32 {
        return None;
    }
    Some(String::from(rest))
}

/// Where the true answer came in a list of scores, counting from zero.
fn rank_by(scores: &[f32], truth: usize) -> usize {
    let mine = scores[truth];
    scores
        .iter()
        .enumerate()
        .filter(|(i, s)| *i != truth && **s > mine)
        .count()
}

/// Known-item retrieval over the whole forest, one method at a time.
///
/// The ladder is the point. Each row is the *same* queries against the *same*
/// candidates with one thing changed, so the differences are the methods and
/// not the sample.
pub fn bench(sample: usize) {
    if !sysbox::stored::available() {
        kprintln!("  no store mounted -- 'store init', then 'snap'");
        return;
    }
    let _claim = match crate::ai::claim_engine() {
        Some(c) => c,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };
    let ix = index_at(ROOT);
    if ix.is_empty() {
        kprintln!("  nothing under {} in the last snapshot", ROOT);
        return;
    }
    let n = ix.len();

    let built = crate::ai::with_engine(|e| {
        let docs: Vec<Vec<usize>> = ix
            .heads
            .iter()
            .map(|h| crate::ai::lex::tokens(&e.tok, &crate::ai::route::queryable(h)))
            .collect();
        let lex = crate::ai::lex::Lex::build(e.tok.vocab_size(), &docs);
        let idfv: Vec<Vec<f32>> =
            docs.iter().map(|d| crate::ai::lex::pool_ids(&e.model, d, &lex)).collect();
        // The old way, kept for the comparison rather than out of nostalgia:
        // a ladder whose bottom rung is missing cannot say how far it climbed.
        // Normalised so both are compared by the same dot product.
        let meanv: Vec<Vec<f32>> = ix
            .heads
            .iter()
            .map(|h| {
                let mut v = crate::ai::vocab::pool_text(
                    &e.model,
                    &e.tok,
                    &crate::ai::route::queryable(h),
                );
                crate::ai::lex::normalise(&mut v);
                v
            })
            .collect();
        (lex, idfv, meanv)
    });
    let (lex, idfv, meanv) = match built {
        Some(v) => v,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };

    // Sampled by stride, so the set is spread over every subject and is the
    // same set on every run. Taking a prefix would measure whichever subject
    // sorts first.
    let stride = (n / sample.max(1)).max(1);
    let mut queries: Vec<(usize, String)> = Vec::new();
    let mut nodisjoint = 0usize;
    let mut i = 0usize;
    while i < n && queries.len() < sample {
        if let Some(b) = sysbox::read_blob(&ix.paths[i]) {
            let body = String::from_utf8_lossy(&b);
            match held_out(&body) {
                Some(q) => queries.push((i, q)),
                None => nodisjoint += 1,
            }
        }
        i += stride;
    }
    if queries.is_empty() {
        kprintln!("  no node had a body disjoint from its own concept -- nothing to ask");
        return;
    }

    // **One knob per column, both swept, and the queries asked two ways.**
    // `k1` is how fast a repeated term stops adding, `b` is how much a document
    // is charged for its length. `k1 = 0` collapses to presence scoring, so the
    // row this shipped before is in the grid rather than beside it.
    // **Three families, because they are three different normalisations and
    // the first sweep conflated two of them.** `Terms` divides the summed IDF
    // mass by a length charge *after* the sum; BM25 puts the charge inside the
    // saturation, where it is multiplied by `k1` -- so BM25 at `k1 = 0` is not
    // "presence with a length discount", it is presence with **no** discount at
    // all, and a grid that only had those two rows reported the discount as
    // having no effect. The row that had actually won came back missing.
    enum M {
        /// Presence, then a length charge on the sum. The flag is whether a
        /// term's weight is its IDF or its IDF squared.
        Terms(f32, bool),
        /// Saturating term frequency, with the charge inside it.
        Bm25(f32, f32),
        /// The best `Terms` row, mixed with the embedding.
        Mix(f32),
    }
    let grid = alloc::vec![
        M::Terms(0.25, false),
        M::Terms(0.50, false),
        M::Terms(0.75, false),
        M::Terms(0.25, true),
        M::Terms(0.50, true),
        M::Terms(0.75, true),
        M::Bm25(1.2, 0.50),
        M::Mix(0.10),
        M::Mix(0.30),
    ];
    let rows = 2 + grid.len();

    // Two query sets over the same nodes. **The long one is the body tail; the
    // short one is its first eight words**, which is about the length of a
    // question somebody types. A constant tuned on long queries and used on
    // short ones is tuned on the wrong distribution, and until now there was no
    // way to see that.
    const SHORT_WORDS: usize = 8;
    let shorten = |q: &str| -> String {
        let mut out = String::new();
        for (k, w) in q.split_whitespace().take(SHORT_WORDS).enumerate() {
            if k > 0 {
                out.push(' ');
            }
            out.push_str(w);
        }
        out
    };

    let mhz = crate::time::tsc_mhz().max(1);
    let t0 = crate::time::rdtsc();
    let mut sm = alloc::vec![0.0f32; n];
    let mut si = alloc::vec![0.0f32; n];
    let mut sl = alloc::vec![0.0f32; n];
    let mut raw = alloc::vec![0.0f32; n];
    let mut raw2 = alloc::vec![0.0f32; n];
    let mut best = alloc::vec![0.0f32; n];
    let mut r1 = [alloc::vec![0usize; rows], alloc::vec![0usize; rows]];
    let mut r5 = [alloc::vec![0usize; rows], alloc::vec![0usize; rows]];
    let mut mrr = [alloc::vec![0.0f32; rows], alloc::vec![0.0f32; rows]];

    for set in 0..2 {
        for (truth, full) in &queries {
            let q = if set == 0 { full.clone() } else { shorten(full) };
            let pooled = crate::ai::with_engine(|e| {
                let ids = crate::ai::lex::tokens(&e.tok, &q);
                let mut m = crate::ai::vocab::pool_text(&e.model, &e.tok, &q);
                crate::ai::lex::normalise(&mut m);
                (ids.clone(), m, crate::ai::lex::pool_ids(&e.model, &ids, &lex))
            });
            let (ids, qm, qi) = match pooled {
                Some(v) => v,
                None => return,
            };
            // Every vector is unit length, so a cosine is a dot product and the
            // sweep is one multiply-add per dimension per node.
            for j in 0..n {
                sm[j] = crate::ai::lex::dot(&qm, &meanv[j]);
                si[j] = crate::ai::lex::dot(&qi, &idfv[j]);
            }

            let mut tally = |row: usize, scores: &[f32]| {
                let r = rank_by(scores, *truth);
                if r == 0 {
                    r1[set][row] += 1;
                }
                if r < 5 {
                    r5[set][row] += 1;
                }
                mrr[set][row] += 1.0 / (r as f32 + 1.0);
            };
            tally(0, &sm);
            tally(1, &si);
            // The postings walk once, reused by every `Terms` row, so what
            // differs between them is the charge and nothing else.
            // The two powers the shipped grid sweeps. `IDF_POW` is a whole
            // number now rather than a flag, so a wider sweep is a wider grid
            // here; the knob table offers 1 and 3 to the loop, which judges
            // them on the host rail where the corpus lives.
            let tot1 = lex.score_raw_p(&ids, &mut raw, 1);
            let tot2 = lex.score_raw_p(&ids, &mut raw2, 2);
            for (k, m) in grid.iter().enumerate() {
                match m {
                    M::Terms(b, sq) => {
                        let (src, tot) = if *sq { (&raw2, tot2) } else { (&raw, tot1) };
                        sl.copy_from_slice(src);
                        lex.finish(&mut sl, tot, *b);
                        if (*b - crate::ai::lex::LEN_B).abs() < 1.0e-6
                            && (if *sq { 2 } else { 1 }) == crate::ai::lex::IDF_POW
                        {
                            best.copy_from_slice(&sl);
                        }
                    }
                    M::Bm25(k1, b) => lex.bm25(&ids, &mut sl, *k1, *b),
                    // Against the shipped `Terms` row, so two knobs are never
                    // swept against each other at once.
                    M::Mix(a) => {
                        for j in 0..n {
                            sl[j] = a * si[j] + (1.0 - a) * best[j];
                        }
                    }
                }
                tally(2 + k, &sl);
            }
        }
    }
    let us = (crate::time::rdtsc() - t0) / mhz;

    let q = queries.len();
    kprintln!("  known-item retrieval: {} candidates, {} quer(ies), asked two ways", n, q);
    kprintln!("  long  -- a node's body after its own first sentence");
    kprintln!("  short -- the first {} words of that, which is what a person types", SHORT_WORDS);
    kprintln!("  the index holds that first sentence and its terms, so no string is shared");
    if nodisjoint > 0 {
        kprintln!("  {} sampled node(s) had no disjoint tail and were not asked", nodisjoint);
    }
    let per = |a: usize| a * 1000 / q;
    let name = |row: usize| -> alloc::string::String {
        match row {
            0 => alloc::format!("{:<18}", "mean pool"),
            1 => alloc::format!("{:<18}", "idf pool"),
            k => match grid[k - 2] {
                M::Terms(b, sq) => {
                    alloc::format!("terms b={:.2} idf{:<5}", b, if sq { "^2" } else { "" })
                }
                M::Bm25(k1, b) => alloc::format!("bm25 k={:.1} b={:<7.2}", k1, b),
                M::Mix(a) => alloc::format!("mix a={:<12.2}", a),
            },
        }
    };
    kprintln!(
        "  {:<18} {:>7} {:>7} {:>7}   {:>7} {:>7} {:>7}",
        "method",
        "L r@1",
        "L r@5",
        "L MRR",
        "S r@1",
        "S r@5",
        "S MRR"
    );
    for row in 0..rows {
        let (la, lb) = (per(r1[0][row]), per(r5[0][row]));
        let (sa, sb) = (per(r1[1][row]), per(r5[1][row]));
        kprintln!(
            "  {} {:>5}.{}% {:>5}.{}% {:>7.4}   {:>5}.{}% {:>5}.{}% {:>7.4}",
            name(row),
            la / 10,
            la % 10,
            lb / 10,
            lb % 10,
            mrr[0][row] / q as f32,
            sa / 10,
            sa % 10,
            sb / 10,
            sb % 10,
            mrr[1][row] / q as f32
        );
    }
    kprintln!(
        "  shipping terms b={:.2} idf{}; chance at r@1 is 1 in {}",
        crate::ai::lex::LEN_B,
        if crate::ai::lex::IDF_POW == 1 { alloc::string::String::new() }
        else { alloc::format!("^{}", crate::ai::lex::IDF_POW) },
        n
    );
    kprintln!("  swept in {} ms", us / 1000);
}

/// Why a question ranked the entries it did, token by token.
///
/// **Built because two guesses had already been wrong.** The entry about
/// differentiating a polynomial was not first and the two obvious explanations
/// -- term frequency and query length -- were both measured and neither was it.
/// Guessing a third time would have been worse than the first two; this prints
/// what the scorer actually saw.
pub fn why(q: &str, show: usize) {
    let _claim = match crate::ai::claim_engine() {
        Some(c) => c,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };
    let lex = match crate::ai::recall::load_lex() {
        Some(l) => l,
        None => {
            kprintln!("  no postings -- 'forest embed' writes them");
            return;
        }
    };
    let ids = match crate::ai::with_engine(|e| crate::ai::lex::tokens(&e.tok, q)) {
        Some(v) => v,
        None => return,
    };

    // The query as the scorer sees it: which tokens, spelled how, worth what.
    // A token printed with its bytes is the only way to see that `polynomial`
    // and `polynomials` are two different things to an index that never stems.
    let mut uniq: Vec<usize> = Vec::new();
    for t in &ids {
        if !uniq.contains(t) {
            uniq.push(*t);
        }
    }
    kprintln!("  {} token(s), {} distinct", ids.len(), uniq.len());
    let mut total = 0.0f32;
    for t in &uniq {
        let idf = lex.idf(*t);
        total += idf;
        let text = crate::ai::with_engine(|e| {
            alloc::string::String::from_utf8_lossy(e.tok.token_bytes(*t)).into_owned()
        })
        .unwrap_or_default();
        kprintln!(
            "    {:>6}  df {:>5}  idf {:>7.4}  '{}'",
            t,
            lex.df.get(*t).copied().unwrap_or(0),
            idf,
            text
        );
    }
    kprintln!("  query idf mass {:.4}", total);

    let n = match crate::ai::recall::with_nodes(|nodes| nodes.len()) {
        Some(v) => v,
        None => {
            kprintln!("  no node vectors -- 'forest embed' writes them");
            return;
        }
    };
    let mut sc = alloc::vec![0.0f32; n];
    lex.score(&ids, &mut sc);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        sc[*b].partial_cmp(&sc[*a]).unwrap_or(core::cmp::Ordering::Equal).then(a.cmp(b))
    });

    let paths = crate::ai::recall::with_nodes(|nodes| nodes.paths.clone()).unwrap_or_default();
    kprintln!("  rank  score   len  charge   path");
    for (r, i) in order.iter().take(show).enumerate() {
        let short = paths[*i].strip_prefix(ROOT).unwrap_or(&paths[*i]);
        kprintln!(
            "  {:>4}  {:.4}  {:>4}  {:.4}   {}",
            r + 1,
            sc[*i],
            lex.len.get(*i).copied().unwrap_or(0),
            lex.charge(*i, crate::ai::lex::LEN_B),
            short
        );
        let mut hit = alloc::string::String::new();
        for t in &uniq {
            if lex.contains(*t, *i as u32) {
                if !hit.is_empty() {
                    hit.push(' ');
                }
                let text = crate::ai::with_engine(|e| {
                    alloc::string::String::from_utf8_lossy(e.tok.token_bytes(*t)).into_owned()
                })
                .unwrap_or_default();
                hit.push_str(&alloc::format!("{}({:.2})", text.trim(), lex.idf(*t)));
            }
        }
        kprintln!("        matched: {}", hit);
    }

    // Where the documents carrying the *rarest* thing asked for actually
    // landed. If the question's most distinctive word is in a hundred nodes and
    // none of them is near the top, the ranking is not failing to find them --
    // it is finding them and then putting something else first.
    let rarest = uniq.iter().copied().fold((0usize, -1.0f32), |acc, t| {
        let v = lex.idf(t);
        if v > acc.1 {
            (t, v)
        } else {
            acc
        }
    });
    let text = crate::ai::with_engine(|e| {
        alloc::string::String::from_utf8_lossy(e.tok.token_bytes(rarest.0)).into_owned()
    })
    .unwrap_or_default();
    let holders = lex.posting(rarest.0).len();
    let mut bestrank = usize::MAX;
    for (r, i) in order.iter().enumerate() {
        if lex.contains(rarest.0, *i as u32) {
            bestrank = r + 1;
            break;
        }
    }
    kprintln!(
        "  rarest asked: '{}' idf {:.4}, in {} node(s); best of them ranks {}",
        text.trim(),
        rarest.1,
        holders,
        if bestrank == usize::MAX { 0 } else { bestrank }
    );
}

/// How many candidates are offered to the fill.
///
/// The fill is best-first and skips what will not fit, so offering more than a
/// budget can hold costs only the bodies fetched for them -- a handful of disk
/// reads at a few hundred bytes a node. Offering too *few* is the error that
/// matters: the budget would go unspent with nothing saying why.
const OFFERED: usize = 24;

/// What would go into the turn for this question, and what it costs.
///
/// `subjects` of zero scores every node, which is the **default and the
/// accurate one**. Routing filters to the top few subjects first, which is what
/// a corpus too large to score exhaustively would have to do -- and its price
/// is measured here rather than assumed. On this forest it is steep: routing to
/// three subjects of sixteen found four of the nine entries a full scan chose,
/// while scoring all nine thousand nodes costs five million multiply-adds and
/// no disk at all. The router earns its keep when the vectors stop fitting, and
/// not before.
pub fn recall_query(q: &str, budget: usize, subjects: usize) {
    // Held for the whole call, so the pooling and every token count come from
    // the same engine. Without it another task can take it mid-fill and the
    // counter starts answering "no", which the fill would read as "nothing
    // fits" and render an empty block for no visible reason.
    let _claim = match crate::ai::claim_engine() {
        Some(c) => c,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };
    let t = match crate::ai::route::load() {
        Some(t) => t,
        None => {
            kprintln!("  no routing table -- 'forest embed' builds one");
            return;
        }
    };
    let lex = match crate::ai::recall::load_lex() {
        Some(l) => l,
        None => {
            kprintln!("  no postings at {} -- 'forest embed' writes them", crate::ai::recall::LEX);
            return;
        }
    };
    // The query, tokenised once: the same ids drive the term match and the
    // pooled vector, so the two channels cannot disagree about what was asked.
    let asked = crate::ai::with_engine(|e| {
        let ids = crate::ai::lex::tokens(&e.tok, q);
        let v = crate::ai::lex::pool_ids(&e.model, &ids, &lex);
        (ids, v)
    });
    let (ids, qv) = match asked {
        Some(v) => v,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };
    // For the subject ranking only, which is a separate comparison against a
    // table with its own centroid.
    let mut v = qv.clone();
    t.centre(&mut v);

    // Two lists over the same scores: everything, and the part of it the router
    // would have reached. The second is what a corpus too large to score
    // exhaustively would be limited to, so the difference between them is the
    // price of routing -- measured here rather than asserted.
    let lists = crate::ai::recall::with_nodes(|n| {
        if n.dim != v.len() {
            return None;
        }
        // Terms and embedding, mixed by the constant `forest bench` chose.
        // At the measured mix of zero this is the term score alone, and the
        // dot product is skipped rather than multiplied by nothing.
        let mut sl = alloc::vec![0.0f32; n.len()];
        lex.score(&ids, &mut sl);
        let mix = crate::ai::recall::MIX;
        let mut all: Vec<(usize, f32)> = (0..n.len())
            .map(|i| {
                let s = if mix > 0.0 {
                    mix * crate::ai::lex::dot(&qv, n.row(i)) + (1.0 - mix) * sl[i]
                } else {
                    sl[i]
                };
                (i, s)
            })
            .collect();
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        let m = crate::ai::route::means(&t);
        let keep: Vec<usize> = crate::ai::route::rank(&t, &m, &v)
            .iter()
            .take(subjects)
            .map(|(i, _)| *i)
            .collect();
        let named: Vec<String> = keep.iter().map(|k| t.names[*k].clone()).collect();
        let take = |src: &[(usize, f32)]| -> Vec<(String, f32)> {
            src.iter().take(OFFERED).map(|(i, sc)| (n.paths[*i].clone(), *sc)).collect()
        };
        let routed: Vec<(usize, f32)> = if subjects == 0 {
            all.clone()
        } else {
            all.iter()
                .copied()
                .filter(|(i, _)| {
                    let sub = crate::ai::route::subject_of(&n.paths[*i]);
                    named.iter().any(|k| k == sub)
                })
                .collect()
        };
        Some((take(&all), take(&routed), named, n.len()))
    })
    .flatten();
    let (all, routed, named, total) = match lists {
        Some(v) => v,
        None => {
            kprintln!("  no node vectors at {} -- 'forest embed' writes them", crate::ai::recall::NODES);
            kprintln!("  (or the table's width does not match this model -- rebuild it)");
            return;
        }
    };

    let bodies = |list: &[(String, f32)]| -> Vec<crate::ai::recall::Cand> {
        list.iter()
            .filter_map(|(path, score)| {
                let b = sysbox::read_blob(path)?;
                Some(crate::ai::recall::Cand {
                    path: path.clone(),
                    score: *score,
                    body: String::from_utf8_lossy(&b).into_owned(),
                })
            })
            .collect()
    };
    // Encoded the way `generate` will, per the module header. A refusal answers
    // `usize::MAX`, so a candidate is dropped rather than admitted -- the safe
    // direction, since the whole point is never to exceed the budget.
    let count = |text: &str| {
        crate::ai::with_engine(|e| e.tok.encode(text, false, false).len()).unwrap_or(usize::MAX)
    };

    let rc = bodies(&routed);
    let ac = bodies(&all);
    let rf = crate::ai::recall::fill(q, &rc, budget, count);
    let af = crate::ai::recall::fill(q, &ac, budget, count);

    if subjects == 0 {
        kprintln!("  every subject, {} node(s) scored", total);
    } else {
        kprintln!("  {} subject(s) of {}, {} node(s) scored", named.len(), t.len(), total);
        for nm in &named {
            kprintln!("    {}", nm.strip_prefix(ROOT).unwrap_or(nm));
        }
    }
    kprintln!(
        "  budget {} token(s), used {}, {} entr(ies) kept, {} skipped",
        budget,
        rf.tokens,
        rf.taken.len(),
        rf.skipped
    );
    // The claim, stated where it can be read rather than only in a suite.
    if rf.tokens > budget {
        kprintln!("  OVER BUDGET -- this is the one thing this must not do");
    }

    // What routing cost, in the only currency that matters here: how much of
    // the answer a full scan would have given was still found. Printed only
    // when routing was on, because comparing a full scan against itself is a
    // line that always reads perfect and means nothing.
    if subjects > 0 {
        let same = rf
            .taken
            .iter()
            .filter(|i| af.taken.iter().any(|j| ac[*j].path == rc[**i].path))
            .count();
        kprintln!(
            "  a full scan would have used {} token(s) over {} entr(ies); routing found {} of them",
            af.tokens,
            af.taken.len(),
            same
        );
    }

    for (n, i) in rf.taken.iter().enumerate() {
        let c = &rc[*i];
        let short = c.path.strip_prefix(ROOT).unwrap_or(&c.path);
        kprintln!("  {:.4}  {}. {}", c.score, n + 1, short);
    }
    kprintln!("  ---- what would go into the turn ----");
    for line in rf.text.lines() {
        kprintln!("  {}", line);
    }
}

/// Leave-one-out over every node: (top-1, top-3, scored, alone).
///
/// The query is centred here and the branch vectors centre themselves, so both
/// sides move together or neither does. Centring one and not the other compares
/// a residual against a whole vector, which is a different question with a
/// perfectly plausible cosine.
fn score_loo(
    t: &crate::ai::route::Table,
    vecs: &[Vec<f32>],
    owner: &[usize],
) -> (usize, usize, usize, usize) {
    let m = crate::ai::route::means(t);
    let (mut top1, mut top3, mut scored, mut alone) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..vecs.len() {
        match t.without(owner[i], &vecs[i]) {
            None => alone += 1,
            Some(loo) => {
                let mut q = vecs[i].clone();
                t.centre(&mut q);
                let r = crate::ai::route::rank_of(t, &m, &q, owner[i], &loo);
                scored += 1;
                if r == 0 {
                    top1 += 1;
                }
                if r < 3 {
                    top3 += 1;
                }
            }
        }
    }
    (top1, top3, scored, alone)
}

/// Where a question should be looked for.
/// The fitted probe, if one has earned its place.
pub fn load_probe() -> Option<crate::ai::probe::Probe> {
    crate::ai::probe::Probe::from_bytes(&sysbox::read_blob(PROBE)?)
}

pub fn route_query(q: &str) {
    let t = match crate::ai::route::load() {
        Some(t) => t,
        None => {
            kprintln!("  no routing table at {} -- 'forest embed' builds one", crate::ai::route::TABLE);
            return;
        }
    };
    // **The query has to be pooled the way the table was built.**
    //
    // It was not. `pool_text` is a plain mean of raw embedding rows; the table,
    // the node vectors and the probe are all `lex::pool_ids`, which unit-scales
    // each row and weights it by inverse document frequency. Those are
    // different spaces, so every number this verb printed was a cosine between
    // a question in one and a subject in another -- perfectly plausible, and
    // about nothing. `forest recall` had it right all along, which is what
    // made the difference visible.
    //
    // The postings are what carry the weights, so without them there is no way
    // to put the query in the right space and saying so beats answering anyway.
    let lex = match crate::ai::recall::load_lex() {
        Some(l) => l,
        None => {
            kprintln!("  no postings at {} -- 'forest embed' writes them", crate::ai::recall::LEX);
            kprintln!("  without them a query cannot be weighted the way the table was");
            return;
        }
    };
    let v = match crate::ai::with_engine(|e| {
        let ids = crate::ai::lex::tokens(&e.tok, q);
        crate::ai::lex::pool_ids(&e.model, &ids, &lex)
    }) {
        Some(v) => v,
        None => {
            kprintln!("  {}", crate::ai::engine_refusal());
            return;
        }
    };
    // A table written at one width and read at another is a mismatch nothing
    // would report, since both sides are only floats. Checked rather than
    // assumed, because swapping the checkpoint is an ordinary thing to do.
    if v.len() != t.dim {
        kprintln!(
            "  the table is dim {} and this model is dim {} -- 'forest embed' to rebuild it",
            t.dim,
            v.len()
        );
        return;
    }
    // The probe first when there is one, because `forest fit` only stores one
    // that beat the mean on held-out nodes -- and both are shown, because a
    // router that quietly replaced another is a router nobody can check.
    match load_probe() {
        Some(p) if p.classes() == t.len() && p.dim() == t.dim => {
            let scores = p.scores(&v);
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|a, b| {
                scores[*b]
                    .partial_cmp(&scores[*a])
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
            kprintln!("  probe:");
            for i in order.iter().take(5) {
                let name = t.names[*i].strip_prefix(ROOT).unwrap_or(&t.names[*i]);
                kprintln!("  {:.4}  {:<46} {} node(s)", scores[*i], name, t.counts[*i]);
            }
        }
        Some(p) => kprintln!(
            "  a probe is stored but it is {}x{} against a table of {}x{} -- 'forest fit' again",
            p.classes(),
            p.dim(),
            t.len(),
            t.dim
        ),
        None => {}
    }

    let m = crate::ai::route::means(&t);
    let mut v = v;
    t.centre(&mut v);
    kprintln!("  cosine:");
    for (i, score) in crate::ai::route::rank(&t, &m, &v).iter().take(5) {
        let name = t.names[*i].strip_prefix(ROOT).unwrap_or(&t.names[*i]);
        kprintln!("  {:.4}  {:<46} {} node(s)", score, name, t.counts[*i]);
    }
}

pub fn command(rest: &str) {
    let mut w = rest.split_whitespace();
    match w.next().unwrap_or("") {
        "" => report(),
        "cost" => cost(),
        "tree" => match w.next() {
            Some(n) => report_tree(n),
            None => kprintln!("  usage: forest tree <name>"),
        },
        "show" => match w.next() {
            Some(p) => show(p),
            None => kprintln!("  usage: forest show <path under the root>"),
        },
        "index" => index_report(),
        "embed" => embed(),
        "fit" => fit(),
        "why" => {
            let rest = rest.strip_prefix("why").unwrap_or("").trim();
            let (show, q) = match rest.split_once(' ') {
                Some((h, t)) => match h.parse::<usize>() {
                    Ok(k) => (k, t.trim()),
                    Err(_) => (8, rest),
                },
                None => (8, rest),
            };
            if q.is_empty() {
                kprintln!("  usage: forest why [n] <question>");
            } else {
                why(q, show);
            }
        }
        "bench" => {
            let k = w.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(200);
            bench(k);
        }
        "recall" => {
            let rest = rest.strip_prefix("recall").unwrap_or("").trim();
            // An optional leading budget, so the common case is just a
            // question and the uncommon one needs no flag to parse.
            // An optional leading budget, then an optional subject count, so
            // the common case is a bare question. Zero subjects means score
            // everything, which is the default because it is the accurate one.
            let num = |s: &str| s.parse::<usize>().ok();
            let (budget, rest) = match rest.split_once(' ') {
                Some((h, t)) => match num(h) {
                    Some(b) => (b, t.trim()),
                    None => (256, rest),
                },
                None => (256, rest),
            };
            let (subjects, q) = match rest.split_once(' ') {
                Some((h, t)) => match num(h) {
                    Some(k) => (k, t.trim()),
                    None => (0, rest),
                },
                None => (0, rest),
            };
            if q.is_empty() {
                kprintln!("  usage: forest recall [budget] [subjects] <question>");
                kprintln!("  no subject count scores every node, which is the accurate way");
            } else {
                recall_query(q, budget, subjects);
            }
        }
        "route" => {
            let q = rest.strip_prefix("route").unwrap_or("").trim();
            if q.is_empty() {
                kprintln!("  usage: forest route <question>");
            } else {
                route_query(q);
            }
        }
        "body" => match w.next() {
            Some(p) => show_stored(p),
            None => kprintln!("  usage: forest body <path under the root>"),
        },
        other => {
            kprintln!("  no such forest verb: {}", other);
            kprintln!("  forest              trees and node counts");
            kprintln!("  forest tree <name>  branches under one trunk");
            kprintln!("  forest show <path>  one node, head and body");
            kprintln!("  forest cost         resident bytes, and what an index would hold");
            kprintln!("  forest index        build an index from the store, no body resident");
            kprintln!("  forest body <path>  one node, read off the disk and not the namespace");
            kprintln!("  forest embed        build the branch table, store it, and score it");
            kprintln!("  forest bench [n]    known-item retrieval, one method per row");
            kprintln!("  forest why [n] <q>  which tokens scored what, and for whom");
            kprintln!("  forest route <q>    which branches a question belongs in");
            kprintln!("  forest recall [n] [k] <q>  what fits in n tokens; k subjects, 0 for all");
        }
    }
}

/// What the reader claims, checked with no forest installed and no model.
///
/// The claim that earns its place is the last one: this module must never
/// reach for a content hash, because `ls` doing exactly that is what makes
/// listing a forest hash the whole corpus. A test cannot see an absence, so
/// what is asserted instead is that the cheap accessors answer what the walk
/// needs -- a directory is told from a node by `blob_len` alone.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    // A machine with no forest must answer emptily rather than faulting, since
    // that is every machine until somebody imports one. Against a root that
    // does not exist, so the claim holds whether or not one is installed --
    // the first version asked `c.nodes == 0 || c.nodes > 0` of the live root,
    // which is a tautology dressed as a check and could never have failed.
    let absent = census_at("/tmp/forest-absent");
    check(
        "a root that does not exist censuses to nothing rather than faulting",
        absent.trees == 0 && absent.nodes == 0 && absent.bytes == 0,
    );

    // And a forest built here censuses to exactly what was put in it. Two
    // nodes of known length under one branch of one tree: a walk that counted
    // the branch as a node, or recursed into a node as though it were a
    // branch, gets a different number for each of these.
    // Lengths taken from the strings rather than counted by hand. A claim that
    // asserts a number somebody worked out on paper fails the day the fixture
    // gains a character, and reads as a bug in the thing under test.
    let one = "head a | b | c\nkind t\n";
    let two = "head d | e | f\nkind t\nmore\n";
    sysbox::write_text("/tmp/forest-t/oak/limb/00000", one);
    sysbox::write_text("/tmp/forest-t/oak/limb/00001", two);
    let built = census_at("/tmp/forest-t");
    check(
        "a built forest censuses to what was put in it",
        built.trees == 1 && built.branches == 1 && built.nodes == 2,
    );
    check(
        "bytes are the nodes' own, and a directory contributes none",
        built.bytes == one.len() + two.len(),
    );

    // `blob_len` is the one predicate the walk uses to tell a branch from a
    // node, so the walk is only correct while it answers None for a directory.
    let dir_is_not_a_node = sysbox::blob_len("/ai").is_none();
    check("a directory is not mistaken for a node", dir_is_not_a_node);

    // And a real blob must read as one, or every node would be walked into as
    // though it were a branch and the census would silently be zero.
    sysbox::write_text("/tmp/forest-probe", "head a | b | c\nkind t\n");
    let blob_is_a_node = sysbox::blob_len("/tmp/forest-probe").is_some();
    check("a blob is a node", blob_is_a_node);

    // The head is the first line with its key stripped, because that is what
    // an index stores and what the router will pool over.
    let h = head_of("/tmp/forest-probe");
    check(
        "the head is the first line, with the key removed",
        h.as_deref() == Some("a | b | c"),
    );

    // --- the stored tree, which needs a store to be one ---------------------
    //
    // These are the claims about reading a body that is not resident, so they
    // need a formatted store with a snapshot in it. On a machine with none
    // they **do not run and say so**, rather than passing: a suite reporting
    // that ranged reads work on a machine where none has ever been performed
    // is the `smp` canary failure in a different subsystem.
    //
    // `/lib` is the subject because it is seeded identically at every boot and
    // nothing writes to it, so the stored copy and the resident one agree by
    // construction -- which is what lets the comparison below mean something
    // about the read path rather than about whether somebody edited a file.
    let stored = sysbox::stored::locate_under("/lib");
    match stored.first() {
        None => {
            console::set_color(LTGRAY);
            kprintln!("  ....  no snapshot holds /lib, so 14 claim(s) about the stored tree did not run");
            kprintln!("        'store init', 'store unlock' and 'snap' make them runnable");
            console::set_color(LTGRAY);
        }
        Some((path, cr)) => {
            let live = sysbox::read_blob(path).unwrap_or_default();
            // Taken before this suite reads anything, because the claim about
            // contention below is about what *else* touched the one buffer and
            // the suite itself refuses a caller on purpose at the end.
            let mark = sysbox::stored::contended();
            check(
                "a path resolves in the stored tree, and a directory is not a blob",
                sysbox::stored::locate(path).is_some() && sysbox::stored::locate("/lib").is_none(),
            );
            check(
                "the whole blob through the ranged path is the resident blob",
                sysbox::stored::read_all(cr).as_deref() == Some(&live[..]),
            );
            // The two that would pass on an implementation that ignored the
            // offset entirely: one inside a block, one deliberately crossing
            // a boundary, since blocks are 512 bytes and this spans two.
            let mid = sysbox::stored::read_at(cr, 100, 50);
            check(
                "a range inside one block is those bytes and not the first ones",
                live.len() > 150 && mid.as_deref() == Some(&live[100..150]),
            );
            let span = sysbox::stored::read_at(cr, 500, 100);
            check(
                "and a range crossing a block boundary is still those bytes",
                live.len() > 600 && span.as_deref() == Some(&live[500..600]),
            );
            check(
                "one block reaches the head, whatever the body weighs",
                sysbox::stored::head_line(cr).is_some()
                    && sysbox::stored::head_line(cr)
                        == core::str::from_utf8(&live)
                            .ok()
                            .and_then(|t| t.lines().next())
                            .map(String::from),
            );
            // Past the end must be empty rather than whatever the next chunk
            // holds, which `read_blocks` refuses at the block level and this
            // clamps at the byte level -- the neighbour's bytes are a
            // perfectly plausible node belonging to something else.
            check(
                "an offset past the end answers nothing, not the next chunk",
                sysbox::stored::read_at(cr, cr.len, 16).as_deref() == Some(&[][..]),
            );

            // --- the arithmetic, at the edges ---------------------------
            //
            // Everything above reads comfortably inside the blob. These are
            // the places a ranged read goes wrong: the last byte, a length
            // that runs off the end, an offset sitting exactly on a block
            // boundary, and zero.
            let n = live.len() as u64;
            check(
                "the very last byte, which is the one an off-by-one loses",
                n > 0
                    && sysbox::stored::read_at(cr, n - 1, 1).as_deref()
                        == Some(&live[live.len() - 1..]),
            );
            check(
                "a length running off the end is clamped to what is there, not refused",
                n > 4
                    && sysbox::stored::read_at(cr, n - 4, 4096).as_deref()
                        == Some(&live[live.len() - 4..]),
            );
            check(
                "an offset exactly on a block boundary, where skip is zero",
                live.len() > 700
                    && sysbox::stored::read_at(cr, 512, 100).as_deref() == Some(&live[512..612]),
            );
            check(
                "a zero-length read is empty rather than one block",
                sysbox::stored::read_at(cr, 10, 0).as_deref() == Some(&[][..]),
            );

            // **The loop had never run.** The scratch holds 128 blocks, every
            // read anybody had made fit in one command, and the multi-window
            // path -- the only arithmetic here worth getting wrong -- had no
            // coverage at all. `read_at_with` exists so a claim can shrink the
            // window to a single block and walk a real blob the long way.
            check(
                "the whole blob read one block at a time is the whole blob",
                sysbox::stored::read_at_with(cr, 0, live.len(), 1).as_deref() == Some(&live[..]),
            );
            check(
                "and an awkward range agrees however many windows it takes",
                live.len() > 1600
                    && sysbox::stored::read_at_with(cr, 513, 1000, 1).as_deref()
                        == Some(&live[513..1513])
                    && sysbox::stored::read_at_with(cr, 513, 1000, 2).as_deref()
                        == Some(&live[513..1513]),
            );

            // The buffer is one buffer. If this has moved, two callers have
            // been inside a ranged read at once and one of them was refused --
            // which is the outcome that was designed for, and still a thing to
            // know about rather than a thing to pass over.
            //
            // **Against a mark rather than against zero, and that is the whole
            // fix.** It read `contended() == 0`, which is a boot-wide canary
            // that the very next claim poisons: `refuses_while_held` causes a
            // refusal on purpose, so the counter is one from then on and a
            // second run of this suite in the same boot failed here every
            // time, on a mechanism that was working perfectly. `diag all`
            // twice is exactly what the release gate does, so what it reported
            // was a red line under `forest` on every clean machine.
            //
            // A claim that can only pass once per boot is not a canary. This
            // one asks what it always meant to ask -- did anything other than
            // this suite's own test need refusing -- and can be asked again.
            check(
                "nothing but this suite's own test has been refused for contention",
                sysbox::stored::contended() == mark,
            );
            // And the guard is shown to work rather than assumed, because a
            // counter at zero says nothing about whether anything is checking.
            // The red line above this one is the refusal, on purpose.
            check(
                "the one buffer does refuse a second caller, and works again after",
                sysbox::stored::refuses_while_held(cr),
            );
        }
    }

    ok
}
