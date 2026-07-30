//! The namespace: a Merkle tree held in RAM, backed by the content-addressed
//! store.
//!
//! There are no inodes, no allocation bitmap, no directory blocks, and no
//! rename-vs-unlink races, because there is no mutable on-disk structure at
//! all. A directory is a list of (name, child address). Its own address is the
//! hash of that list. Change anything anywhere and every address from there to
//! the root changes; change nothing and every address is identical.
//!
//! Most of what looks like sleight of hand downstream is just that property
//! being read back:
//!
//!   * `cp` of a directory copies 32 bytes, whatever is inside it.
//!   * `same a b` compares two whole subtrees in one memcmp.
//!   * `diff` skips any subtree whose two addresses match, so comparing two
//!     snapshots costs time proportional to what changed, not to what exists.
//!   * a snapshot is just the root address, so taking one is free.
//!   * `du` can separate apparent bytes from unique bytes, because identical
//!     content is literally the same object.
//!
//! The hash is over content only -- names, kinds and child content hashes --
//! and deliberately not over where anything is stored. If it covered block
//! addresses then the same tree written twice would hash differently and every
//! property above would quietly stop holding.
//!
//! Every variable-length field is length-prefixed before it is hashed. Without
//! that, `("ab", x)` and `("a", "bx")` can serialise identically, and two
//! different trees sharing an address is the one failure this design cannot
//! survive.

use crate::store::cas::ChunkRef;
use crate::store::sha256::Sha256;
use alloc::string::String;
use alloc::vec::Vec;

pub type Hash = [u8; 32];

pub const KIND_BLOB: u8 = 0;
pub const KIND_DIR: u8 = 1;

/// Bounds the recursion in `content_hash` and friends. Nothing here uses an
/// explicit stack, and a kernel task stack is not large.
pub const MAX_DEPTH: usize = 32;

const DIR_MAGIC: &[u8; 8] = b"GLADOSTR";

pub enum Node {
    Blob(Vec<u8>),
    /// Kept sorted by name. Sorting is not for lookup speed -- these are tiny
    /// -- but so that serialisation is canonical: the same contents must
    /// produce the same bytes, or the hash means nothing.
    Dir(Vec<(String, Node)>),
}

impl Node {
    pub fn empty_dir() -> Node {
        Node::Dir(Vec::new())
    }

    pub fn kind(&self) -> u8 {
        match self {
            Node::Blob(_) => KIND_BLOB,
            Node::Dir(_) => KIND_DIR,
        }
    }

    pub fn is_dir(&self) -> bool {
        matches!(self, Node::Dir(_))
    }
}

/// A deep copy. Only ever needed because the in-RAM tree owns its nodes; the
/// *stored* form shares structure, which is why `cp` is cheap on disk even
/// though this is not.
pub fn clone_node(n: &Node) -> Node {
    match n {
        Node::Blob(b) => Node::Blob(b.clone()),
        Node::Dir(es) => Node::Dir(es.iter().map(|(k, v)| (k.clone(), clone_node(v))).collect()),
    }
}

/// The content address of a node.
///
/// For a blob this is exactly `sha256(bytes)`, which is also what `Store::put`
/// computes -- so the address `hash` prints for a file is the same address the
/// store knows it by, and `blob <that>` finds it.
///
/// Recomputed from scratch on every call rather than cached in the node. At the
/// sizes this holds that is not worth the invalidation bugs a cache would
/// introduce, but it does make repeated whole-tree hashing quadratic, so
/// callers that hash every child of a directory should expect it.
pub fn content_hash(n: &Node) -> Hash {
    match n {
        Node::Blob(b) => {
            let mut h = Sha256::new();
            h.update(b);
            h.finish()
        }
        Node::Dir(es) => {
            let mut h = Sha256::new();
            h.update(b"tree");
            h.update(&(es.len() as u64).to_le_bytes());
            for (name, child) in es {
                h.update(&(name.len() as u64).to_le_bytes());
                h.update(name.as_bytes());
                h.update(&[child.kind()]);
                h.update(&content_hash(child));
            }
            h.finish()
        }
    }
}

pub fn short(h: &Hash) -> [u8; 12] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 12];
    for i in 0..6 {
        out[i * 2] = HEX[(h[i] >> 4) as usize];
        out[i * 2 + 1] = HEX[(h[i] & 0x0F) as usize];
    }
    out
}

// --- lookup -------------------------------------------------------------

pub fn resolve<'a>(root: &'a Node, path: &[String]) -> Option<&'a Node> {
    let mut cur = root;
    for part in path {
        match cur {
            Node::Dir(es) => {
                let i = es.binary_search_by(|(k, _)| k.as_str().cmp(part.as_str())).ok()?;
                cur = &es[i].1;
            }
            Node::Blob(_) => return None,
        }
    }
    Some(cur)
}

pub fn resolve_mut<'a>(root: &'a mut Node, path: &[String]) -> Option<&'a mut Node> {
    let mut cur = root;
    for part in path {
        match cur {
            Node::Dir(es) => {
                let i = es.binary_search_by(|(k, _)| k.as_str().cmp(part.as_str())).ok()?;
                cur = &mut es[i].1;
            }
            Node::Blob(_) => return None,
        }
    }
    Some(cur)
}

pub enum PutError {
    TooDeep,
    NotADirectory,
    Empty,
}

/// Place `node` at `path`, creating intermediate directories.
///
/// Creating parents is the default rather than an opt-in flag: there is no
/// filesystem-level reason for `mkdir -p` to be a separate mode, it is a
/// historical artefact of directories being expensive.
pub fn put(root: &mut Node, path: &[String], node: Node) -> Result<(), PutError> {
    if path.is_empty() {
        return Err(PutError::Empty);
    }
    if path.len() > MAX_DEPTH {
        return Err(PutError::TooDeep);
    }
    let mut cur = root;
    for part in &path[..path.len() - 1] {
        let es = match cur {
            Node::Dir(es) => es,
            Node::Blob(_) => return Err(PutError::NotADirectory),
        };
        let i = match es.binary_search_by(|(k, _)| k.as_str().cmp(part.as_str())) {
            Ok(i) => i,
            Err(i) => {
                es.insert(i, (part.clone(), Node::empty_dir()));
                i
            }
        };
        cur = &mut es[i].1;
    }
    let leaf = &path[path.len() - 1];
    match cur {
        Node::Dir(es) => {
            match es.binary_search_by(|(k, _)| k.as_str().cmp(leaf.as_str())) {
                Ok(i) => es[i].1 = node,
                Err(i) => es.insert(i, (leaf.clone(), node)),
            }
            Ok(())
        }
        Node::Blob(_) => Err(PutError::NotADirectory),
    }
}

/// Detach a name. The object itself is untouched -- if it was ever committed
/// it is still on disk and still reachable by address. Names are the only
/// thing here that can be destroyed.
pub fn remove(root: &mut Node, path: &[String]) -> Option<Node> {
    if path.is_empty() {
        return None;
    }
    let parent = resolve_mut(root, &path[..path.len() - 1])?;
    let leaf = &path[path.len() - 1];
    match parent {
        Node::Dir(es) => {
            let i = es.binary_search_by(|(k, _)| k.as_str().cmp(leaf.as_str())).ok()?;
            Some(es.remove(i).1)
        }
        Node::Blob(_) => None,
    }
}

// --- accounting ---------------------------------------------------------

#[derive(Default)]
pub struct Stats {
    pub files: u64,
    pub dirs: u64,
    /// Bytes as the namespace presents them: the same content named twice
    /// counts twice.
    pub apparent: u64,
    /// Bytes that actually have to exist: distinct content counted once.
    pub unique: u64,
}

pub fn stats(n: &Node) -> Stats {
    let mut seen: Vec<Hash> = Vec::new();
    let mut s = Stats::default();
    walk_stats(n, &mut s, &mut seen);
    s
}

fn walk_stats(n: &Node, s: &mut Stats, seen: &mut Vec<Hash>) {
    match n {
        Node::Blob(b) => {
            s.files += 1;
            s.apparent += b.len() as u64;
            let h = content_hash(n);
            if let Err(i) = seen.binary_search(&h) {
                seen.insert(i, h);
                s.unique += b.len() as u64;
            }
        }
        Node::Dir(es) => {
            s.dirs += 1;
            for (_, c) in es {
                walk_stats(c, s, seen);
            }
        }
    }
}

// --- persistence --------------------------------------------------------
//
// A blob is stored as its raw bytes, so its chunk address is its content
// address. A directory is stored as a listing that also records where each
// child landed, because without that a load would need a global
// address-to-location index and we would be building a second database to
// find the first one.

/// Memo of what has already been written, keyed by content address. This is
/// what makes a second snapshot of a mostly-unchanged tree nearly free: an
/// unchanged subtree hashes the same, hits here, and no blocks are written.
#[derive(Default)]
pub struct Written {
    map: Vec<(Hash, ChunkRef)>,
}

impl Written {
    pub fn get(&self, h: &Hash) -> Option<ChunkRef> {
        self.map
            .binary_search_by(|(k, _)| k.cmp(h))
            .ok()
            .map(|i| self.map[i].1)
    }

    pub fn insert(&mut self, h: Hash, r: ChunkRef) {
        if let Err(i) = self.map.binary_search_by(|(k, _)| k.cmp(&h)) {
            self.map.insert(i, (h, r));
        }
    }
}

pub fn encode_dir(entries: &[(String, u8, ChunkRef)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(DIR_MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, kind, r) in entries {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(*kind);
        out.extend_from_slice(&r.hash);
        out.extend_from_slice(&r.lba.to_le_bytes());
        out.extend_from_slice(&r.len.to_le_bytes());
    }
    out
}

pub fn decode_dir(b: &[u8]) -> Option<Vec<(String, u8, ChunkRef)>> {
    if b.len() < 12 || &b[0..8] != DIR_MAGIC {
        return None;
    }
    let count = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
    let mut out = Vec::new();
    let mut o = 12;
    for _ in 0..count {
        if o + 2 > b.len() {
            return None;
        }
        let nlen = u16::from_le_bytes([b[o], b[o + 1]]) as usize;
        o += 2;
        if o + nlen + 1 + 32 + 16 > b.len() {
            return None;
        }
        let name = String::from_utf8(b[o..o + nlen].to_vec()).ok()?;
        o += nlen;
        let kind = b[o];
        o += 1;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&b[o..o + 32]);
        o += 32;
        let lba = u64::from_le_bytes(b[o..o + 8].try_into().ok()?);
        o += 8;
        let len = u64::from_le_bytes(b[o..o + 8].try_into().ok()?);
        o += 8;
        out.push((name, kind, ChunkRef { hash, lba, len }));
    }
    Some(out)
}
