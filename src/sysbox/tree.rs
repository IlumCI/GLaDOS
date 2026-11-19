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
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

pub type Hash = [u8; 32];

pub const KIND_BLOB: u8 = 0;
pub const KIND_DIR: u8 = 1;
/// The same two kinds, for an entry that carries attributes. **Only** then:
/// an entry with none hashes and encodes exactly as it always did, so every
/// tree written before attributes existed keeps its address.
pub const KIND_BLOB_A: u8 = 2;
pub const KIND_DIR_A: u8 = 3;
// 4 is reserved for a blob stored as a list of extents.

/// A stored kind without its attribute bit: `KIND_BLOB` or `KIND_DIR`.
pub fn base_kind(k: u8) -> u8 {
    k & 1
}

/// What a Linux program means by a file's metadata: mode (with its type bits),
/// owner, modification time, extended attributes.
///
/// **On the directory entry, never on the blob.** A blob is named by the hash
/// of its bytes, and two files with the same bytes and different owners must
/// still be one blob -- or a package's thousand identical license files stop
/// deduplicating the moment pacman stamps them. So the entry carries the
/// attributes and the content stays shared.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Attr {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime_ns: i64,
    /// Sorted by name, so the encoding is canonical.
    pub xattrs: Vec<(String, Vec<u8>)>,
}

impl Attr {
    /// The canonical bytes, which are both what is hashed and what is stored.
    pub fn encode(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(&self.mode.to_le_bytes());
        o.extend_from_slice(&self.uid.to_le_bytes());
        o.extend_from_slice(&self.gid.to_le_bytes());
        o.extend_from_slice(&self.mtime_ns.to_le_bytes());
        o.extend_from_slice(&(self.xattrs.len() as u16).to_le_bytes());
        for (k, v) in &self.xattrs {
            o.extend_from_slice(&(k.len() as u16).to_le_bytes());
            o.extend_from_slice(k.as_bytes());
            o.extend_from_slice(&(v.len() as u32).to_le_bytes());
            o.extend_from_slice(v);
        }
        o
    }

    pub fn decode(b: &[u8]) -> Option<Attr> {
        if b.len() < 22 {
            return None;
        }
        let u32_ = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let mut a = Attr {
            mode: u32_(0),
            uid: u32_(4),
            gid: u32_(8),
            mtime_ns: i64::from_le_bytes(b[12..20].try_into().unwrap()),
            xattrs: Vec::new(),
        };
        let n = u16::from_le_bytes([b[20], b[21]]) as usize;
        let mut o = 22;
        for _ in 0..n {
            let kl = u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]) as usize;
            o += 2;
            let k = String::from_utf8(b.get(o..o + kl)?.to_vec()).ok()?;
            o += kl;
            let vl = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize;
            o += 4;
            let v = b.get(o..o + vl)?.to_vec();
            o += vl;
            a.xattrs.push((k, v));
        }
        (o == b.len()).then_some(a)
    }

    pub fn xattr(&self, name: &str) -> Option<&[u8]> {
        self.xattrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_slice())
    }

    pub fn set_xattr(&mut self, name: &str, value: Vec<u8>) {
        match self.xattrs.binary_search_by(|(k, _)| k.as_str().cmp(name)) {
            Ok(i) => self.xattrs[i].1 = value,
            Err(i) => self.xattrs.insert(i, (String::from(name), value)),
        }
    }

    pub fn remove_xattr(&mut self, name: &str) -> bool {
        match self.xattrs.binary_search_by(|(k, _)| k.as_str().cmp(name)) {
            Ok(i) => {
                self.xattrs.remove(i);
                true
            }
            Err(_) => false,
        }
    }
}

/// One name in a directory: what it is called, what is stored there, and the
/// attributes, if anybody set any.
pub struct Ent {
    pub name: String,
    pub attr: Option<Box<Attr>>,
    pub node: Node,
}

impl Ent {
    pub fn new(name: String, node: Node) -> Ent {
        Ent { name, attr: None, node }
    }
    /// The kind byte this entry hashes and stores under.
    pub fn kind(&self) -> u8 {
        self.node.kind() + if self.attr.is_some() { 2 } else { 0 }
    }
}

/// Bounds the recursion in `content_hash` and friends. Nothing here uses an
/// explicit stack, and a kernel task stack is not large.
pub const MAX_DEPTH: usize = 32;

const DIR_MAGIC: &[u8; 8] = b"GLADOSTR";

pub enum Node {
    Blob(Vec<u8>),
    /// A blob that is on disk and not in memory.
    ///
    /// Its `ChunkRef` is the whole of what the tree needs to know about it.
    /// `hash` is the blob's **content address**, because `content_hash` of a
    /// blob is exactly `sha256(bytes)` and that is exactly what `Store::put`
    /// computes; `len` is its length. So an away node hashes, compares, counts
    /// and serialises identically to the blob it stands for, and every address
    /// from it to the root is bit-identical whether or not the bytes are
    /// resident.
    ///
    /// **That is not a coincidence, it is the property the store was designed
    /// around**, and it is why this variant costs nothing anywhere else: a
    /// representation that changed an address would have broken `same`, `diff`,
    /// every snapshot and every stored lineage at once.
    ///
    /// What it does change is that reading the tree can now fail. A resident
    /// tree was self-contained; this one depends on the store staying
    /// readable, which is stated here rather than discovered when a disk goes.
    Away(ChunkRef),
    /// Kept sorted by name. Sorting is not for lookup speed -- these are tiny
    /// -- but so that serialisation is canonical: the same contents must
    /// produce the same bytes, or the hash means nothing.
    Dir(Vec<Ent>),
}

impl Node {
    pub fn empty_dir() -> Node {
        Node::Dir(Vec::new())
    }

    /// A blob is a blob whether or not its bytes are here. Answering anything
    /// else would re-address every directory above an away node.
    pub fn kind(&self) -> u8 {
        match self {
            Node::Blob(_) | Node::Away(_) => KIND_BLOB,
            Node::Dir(_) => KIND_DIR,
        }
    }

    /// How long this blob is, without reading it. `None` for a directory.
    pub fn blob_len(&self) -> Option<u64> {
        match self {
            Node::Blob(b) => Some(b.len() as u64),
            Node::Away(r) => Some(r.len),
            Node::Dir(_) => None,
        }
    }

    /// Where this blob lives, if it is not here.
    pub fn away(&self) -> Option<ChunkRef> {
        match self {
            Node::Away(r) => Some(*r),
            _ => None,
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
        // Forty-eight bytes, whatever the blob weighs. `cp` of an away subtree
        // is therefore nearly as cheap in memory as it always was on disk.
        Node::Away(r) => Node::Away(*r),
        Node::Dir(es) => Node::Dir(
            es.iter().map(|e| Ent { name: e.name.clone(), attr: e.attr.clone(), node: clone_node(&e.node) }).collect(),
        ),
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
        // Already computed, by `Store::put`, over these exact bytes. Reading
        // the blob back to hash it again would answer the same thing and undo
        // the entire point of the variant.
        Node::Away(r) => r.hash,
        Node::Dir(es) => {
            let mut h = Sha256::new();
            h.update(b"tree");
            h.update(&(es.len() as u64).to_le_bytes());
            for e in es {
                h.update(&(e.name.len() as u64).to_le_bytes());
                h.update(e.name.as_bytes());
                h.update(&[e.kind()]);
                h.update(&content_hash(&e.node));
                // After the child, and only when there is one: an entry with
                // no attributes is byte-for-byte what it always was.
                if let Some(a) = &e.attr {
                    let b = a.encode();
                    h.update(&(b.len() as u64).to_le_bytes());
                    h.update(&b);
                }
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
                let i = es.binary_search_by(|e| e.name.as_str().cmp(part.as_str())).ok()?;
                cur = &es[i].node;
            }
            Node::Blob(_) | Node::Away(_) => return None,
        }
    }
    Some(cur)
}

pub fn resolve_mut<'a>(root: &'a mut Node, path: &[String]) -> Option<&'a mut Node> {
    let mut cur = root;
    for part in path {
        match cur {
            Node::Dir(es) => {
                let i = es.binary_search_by(|e| e.name.as_str().cmp(part.as_str())).ok()?;
                cur = &mut es[i].node;
            }
            Node::Blob(_) | Node::Away(_) => return None,
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
            Node::Blob(_) | Node::Away(_) => return Err(PutError::NotADirectory),
        };
        let i = match es.binary_search_by(|e| e.name.as_str().cmp(part.as_str())) {
            Ok(i) => i,
            Err(i) => {
                es.insert(i, Ent::new(part.clone(), Node::empty_dir()));
                i
            }
        };
        cur = &mut es[i].node;
    }
    let leaf = &path[path.len() - 1];
    match cur {
        Node::Dir(es) => {
            // Replacing what is stored keeps the attributes: a file written
            // anew keeps its mode and owner, as on any filesystem.
            match es.binary_search_by(|e| e.name.as_str().cmp(leaf.as_str())) {
                Ok(i) => es[i].node = node,
                Err(i) => es.insert(i, Ent::new(leaf.clone(), node)),
            }
            Ok(())
        }
        Node::Blob(_) | Node::Away(_) => Err(PutError::NotADirectory),
    }
}

/// Detach a name. The object itself is untouched -- if it was ever committed
/// it is still on disk and still reachable by address. Names are the only
/// thing here that can be destroyed.
pub fn remove(root: &mut Node, path: &[String]) -> Option<Node> {
    remove_ent(root, path).map(|e| e.node)
}

/// `remove`, keeping the attributes, for a move that must carry them.
pub fn remove_ent(root: &mut Node, path: &[String]) -> Option<Ent> {
    if path.is_empty() {
        return None;
    }
    let parent = resolve_mut(root, &path[..path.len() - 1])?;
    let leaf = &path[path.len() - 1];
    match parent {
        Node::Dir(es) => {
            let i = es.binary_search_by(|e| e.name.as_str().cmp(leaf.as_str())).ok()?;
            Some(es.remove(i))
        }
        Node::Blob(_) | Node::Away(_) => None,
    }
}

/// The entry at `path`, for its attributes. `None` for the root, which is
/// nobody's child.
pub fn entry<'a>(root: &'a Node, path: &[String]) -> Option<&'a Ent> {
    let (leaf, parent) = path.split_last()?;
    match resolve(root, parent)? {
        Node::Dir(es) => es.binary_search_by(|e| e.name.as_str().cmp(leaf.as_str())).ok().map(|i| &es[i]),
        _ => None,
    }
}

pub fn entry_mut<'a>(root: &'a mut Node, path: &[String]) -> Option<&'a mut Ent> {
    let (leaf, parent) = path.split_last()?;
    match resolve_mut(root, parent)? {
        Node::Dir(es) => es.binary_search_by(|e| e.name.as_str().cmp(leaf.as_str())).ok().map(move |i| &mut es[i]),
        _ => None,
    }
}

/// Place a whole entry, attributes and all, creating parents.
pub fn put_ent(root: &mut Node, path: &[String], ent: Ent) -> Result<(), PutError> {
    let attr = ent.attr;
    put(root, path, ent.node)?;
    if let Some(e) = entry_mut(root, path) {
        e.attr = attr;
    }
    Ok(())
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
    /// How many of those files are on disk rather than in memory.
    pub away: u64,
    /// And what they would weigh if they were here. This is the figure that
    /// says whether moving a corpus out of memory actually worked.
    pub away_bytes: u64,
}

pub fn stats(n: &Node) -> Stats {
    let mut seen: Vec<Hash> = Vec::new();
    let mut s = Stats::default();
    walk_stats(n, &mut s, &mut seen);
    s
}

fn walk_stats(n: &Node, s: &mut Stats, seen: &mut Vec<Hash>) {
    match n {
        Node::Blob(_) | Node::Away(_) => {
            let len = n.blob_len().unwrap_or(0);
            s.files += 1;
            s.apparent += len;
            if matches!(n, Node::Away(_)) {
                s.away += 1;
                s.away_bytes += len;
            }
            let h = content_hash(n);
            if let Err(i) = seen.binary_search(&h) {
                seen.insert(i, h);
                s.unique += len;
            }
        }
        Node::Dir(es) => {
            s.dirs += 1;
            for e in es {
                walk_stats(&e.node, s, seen);
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

/// One stored directory entry: name, kind (0-3), where the child landed, and
/// attributes when the kind says there are some.
pub type Stored = (String, u8, ChunkRef, Option<Attr>);

pub fn encode_dir(entries: &[Stored]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(DIR_MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, kind, r, attr) in entries {
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(*kind);
        out.extend_from_slice(&r.hash);
        out.extend_from_slice(&r.lba.to_le_bytes());
        out.extend_from_slice(&r.len.to_le_bytes());
        if let Some(a) = attr {
            let b = a.encode();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(&b);
        }
    }
    out
}

pub fn decode_dir(b: &[u8]) -> Option<Vec<Stored>> {
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
        let attr = match kind {
            KIND_BLOB | KIND_DIR => None,
            KIND_BLOB_A | KIND_DIR_A => {
                let al = u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?) as usize;
                o += 4;
                let a = Attr::decode(b.get(o..o + al)?)?;
                o += al;
                Some(a)
            }
            // A kind this kernel does not know is a directory it cannot read
            // correctly, so it reads none of it rather than part.
            _ => return None,
        };
        out.push((name, kind, ChunkRef { hash, lba, len }, attr));
    }
    Some(out)
}
