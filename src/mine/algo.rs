//! Which proof-of-work a job wants, and the thing that computes it.
//!
//! Every algorithm here answers one question -- given an 80-byte header and a
//! nonce, what is the digest -- and the header assembly, the merkle fold, the
//! target comparison and the whole Stratum client above are shared unchanged.
//! That is the payoff of picking Bitcoin-family coins: yespower chains inherit
//! Bitcoin's header, so only the hash differs.
//!
//! ### Why the hasher lives on the task and not in the template
//!
//! `Yespower` holds its working set -- up to 8 MiB at N=2048, r=32 -- and
//! reuses it across nonces, because allocating per hash would spend more time
//! in this kernel's locking allocator than in the algorithm. That makes it a
//! large, mutable, single-owner object, which is exactly what a `Spin<Template>`
//! shared with the socket task must not contain. So the template carries the
//! *parameters* and the miner task builds the hasher from them, rebuilding only
//! when they change.
//!
//! ### The batch size is a property of the algorithm
//!
//! SHA-256d runs about 4,000 nonces in a millisecond and yespower runs single
//! digits, so one constant cannot serve both: 4,096 yespower hashes is seconds
//! of a task holding its quantum, which starves the pool connection and makes
//! `mine off` take that long to be felt. `Algo::batch` is the per-algorithm
//! answer and it is sized to roughly a millisecond and a half of work.

use alloc::string::String;
use alloc::vec::Vec;

use super::blake2s;
use super::hash;
use super::neoscrypt::Neoscrypt;
use super::yespower::{Version, Yespower};

/// What a hash is waiting on.
///
/// Two classes and not six, deliberately. A machine has more resources than
/// this -- integer units, shared memory, VRAM bandwidth, three levels of CPU
/// cache -- but an `Algo` cannot know which device it landed on, and the split
/// that survives that ignorance is whether the work is arithmetic or whether
/// it is waiting for memory.
///
/// **There were two variants and the missing third cost about 1.7x of the hash
/// rate.** "Waiting for memory" was one word for two opposite prescriptions.
/// NeoScrypt waits on *capacity*: a 32 KiB working set per slice, and running
/// more slices than the cache holds makes every one of them slower, so the right
/// answer is to cap the count. yespower waits on *latency*: a dependent chain of
/// random reads, where a stalled slice leaves the execution units idle and
/// another slice fills them, so the right answer is the opposite -- oversubscribe.
/// Capping it by cache capacity, which is what `Bound::Memory` asks
/// `work::cache_budget` to do, held a sixteen-core machine to seven slices.
///
/// The doc here used to say "anything finer would be a claim about hardware made
/// in a file that has never seen any", which was the right instinct and is now
/// answerable. The claim is measured twice over. A cycle model of the pwxform
/// lane -- `pmuludq` 5 cycles, `paddq` 1, `pxor` 1, then an L1 load of 5 feeding
/// the next round's gather, so a ~12-cycle chain with four independent gather
/// lanes filling it -- predicts 3.0 cycles per lane, and the measured figure is
/// 3.05 against a pure-throughput floor of 1.4. And removing the cap took a
/// sixteen-core sweep from 693 H/s at seven slices to 1475 at fifteen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Arithmetic,
    /// Waiting on how much fits. More slices than the cache holds is slower.
    Memory,
    /// Waiting on a dependent chain. More slices than cores is *faster*, up to
    /// the point the execution units are full, because one slice's stall is
    /// another's turn.
    Latency,
}

#[derive(Clone, PartialEq)]
pub enum Algo {
    /// Bitcoin's own.
    ///
    /// This said "the only one with a usable midstate, because it is the only
    /// one whose first 64 header bytes can be absorbed once". The reasoning
    /// was right and the conclusion was about SHA-256 rather than about
    /// midstates -- BLAKE2s is also a 64-byte block function over the same
    /// header and gets one too. What actually distinguishes yespower is that
    /// it puts all eighty bytes through PBKDF2 before the expensive part
    /// begins, so there is no prefix to absorb.
    Sha256d,
    /// The scheme BitZeny, Yenten, Koto, WAVI, Veco and PRiVCY use.
    ///
    /// Parameters are explicit rather than named per coin. A preset table would
    /// be a set of numbers this tree asserts about somebody else's network
    /// without having read their source, and a wrong one hashes a different
    /// function perfectly correctly -- which is the failure `Yespower::new`
    /// refuses to clamp its way into.
    Yespower {
        v10: bool,
        n: u32,
        r: u32,
        pers: Option<Vec<u8>>,
    },
    /// Feathercoin's, and the one algorithm here with no parameters *and* no
    /// midstate.
    ///
    /// N, r and the round count are not configurable the way yespower's are:
    /// 128, 2 and 20 are the profile, and a chain that changed them would be
    /// running a different proof of work rather than this one differently
    /// configured. So there is nothing to carry on the wire, which is why this
    /// variant is a bare name where `Yespower` is a structure.
    Neoscrypt,
    /// RFC 7693 BLAKE2s-256 over the header. Verge's `blake2s` chain and
    /// others.
    ///
    /// No parameters, and that is the algorithm rather than a simplification:
    /// a chain using keyed or salted BLAKE2s would be a different `Algo`
    /// variant, not this one with a field bolted on, because a key is part of
    /// the parameter block and changes the initial state.
    Blake2s,
}

impl Algo {
    pub fn name(&self) -> &'static str {
        match self {
            Algo::Sha256d => "sha256d",
            Algo::Yespower { v10: true, .. } => "yespower-1.0",
            Algo::Yespower { v10: false, .. } => "yespower-0.5",
            Algo::Blake2s => "blake2s",
            Algo::Neoscrypt => "neoscrypt",
        }
    }

    /// Nonces per batch, sized to about a millisecond and a half.
    ///
    /// The batch is what bounds how long the miner holds its quantum, so it is
    /// also what bounds how promptly a new job or a `mine off` is felt.
    pub fn batch(&self) -> u32 {
        match self {
            Algo::Sha256d => 4096,
            // Two compressions per nonce against SHA-256d's four, so it runs
            // ahead of sha256d rather than behind it. The same batch is
            // therefore a shorter hold on the quantum, which is the direction
            // that is safe to be wrong in.
            Algo::Blake2s => 4096,
            // Between the two by three orders of magnitude at each end. A
            // NeoScrypt hash is two full SMix passes over 32 KiB, so it lands
            // nearer yespower than sha256d -- measured on the GPU at 190 kH/s
            // against sha256d's 630 MH/s, a factor of 3,300.
            Algo::Neoscrypt => 64,
            // Measured rather than guessed: see `mine bench`. Small because one
            // yespower hash is three orders of magnitude more work than one
            // sha256d, by design.
            Algo::Yespower { .. } => 8,
        }
    }

    /// Bytes of working memory one concurrent hash needs.
    ///
    /// **This is the number that decides how many fit**, and it is the whole
    /// reason a GPU is not automatically better at everything. A device runs
    /// as many hashes at once as its memory divides by this, so an algorithm
    /// at two megabytes puts a 4 GiB card at about two thousand concurrent
    /// hashes against the thousands of threads it wants -- while sha256d at a
    /// couple of hundred bytes puts no limit on it at all.
    ///
    /// The yespower figure is the formula rather than a measurement, and
    /// `checks` asserts it against `Yespower::footprint()` for real parameter
    /// sets. Two expressions of one quantity is a thing this tree normally
    /// refuses; here the alternative is allocating eight megabytes to answer a
    /// scheduling question, so the duplication is bought and then checked.
    pub fn working_set(&self) -> usize {
        match self {
            // A midstate, a header and a digest. Register and L1 territory,
            // which is why these two never bound a device on memory.
            Algo::Sha256d | Algo::Blake2s => 256,
            // Upstream's `(N + 3) * r * 2 * BLOCK_SIZE` plus the 608 bytes of
            // FastKDF buffers. Fixed, because the profile is fixed.
            Algo::Neoscrypt => (128 + 3) * 2 * 2 * 64 + 608,
            Algo::Yespower { v10, n, r, .. } => {
                let (swidth, sboxes) = if *v10 { (11usize, 3usize) } else { (8, 2) };
                // S-boxes, then V at 128*r*N, then X, B and the 128-byte
                // scratch `smix1` runs at r=1 in.
                let s = sboxes * (1usize << swidth) * 2 * 8;
                let b = 128 * *r as usize;
                s + b * *n as usize + b + b + 128
            }
        }
    }

    /// What a hash spends its time waiting for, and therefore what two of them
    /// on one device take from each other.
    ///
    /// **Two algorithms sharing a device contend only when they share this.**
    /// A sha256d kernel saturates integer units and barely touches memory
    /// bandwidth; yespower saturates a cache and leaves the arithmetic units
    /// idle. Run those two together on hardware that has both and the second
    /// is close to free -- run two arithmetic ones together and they simply
    /// halve each other.
    ///
    /// That is the honest form of "mine many coins at once". Concurrency over
    /// one bottleneck cannot beat picking the best thing that bottleneck can
    /// do, because the total is fixed and the split only averages the rates
    /// down. Concurrency over *different* bottlenecks is the case where the
    /// machine genuinely does more work.
    pub fn bound(&self) -> Bound {
        match self {
            // ARX and integer addition over a working set that fits in
            // registers.
            Algo::Sha256d | Algo::Blake2s => Bound::Arithmetic,
            // Two SMix passes of 256 dependent random reads each over 32 KiB.
            // Measured on the GPU by removing SMix: it is 78% of a hash, and
            // the FastKDF that remains is the other 22%.
            Algo::Neoscrypt => Bound::Memory,
            // Sequentially dependent random reads over megabytes. The limit is
            // a cache's *latency* and nothing about the ALUs -- which is what
            // makes this family CPU-only by construction rather than by
            // convention, and which this arm reported as `Memory` while the
            // comment said latency. `cache_budget` reads the variant, not the
            // comment, so it capped the slice count to what the cache holds and
            // denied the oversubscription that hides the chain. See `Bound`.
            Algo::Yespower { .. } => Bound::Latency,
        }
    }

    /// Whether running these two at once on **one device** divides it.
    ///
    /// The device is the caller's to know: two algorithms on a GPU and a CPU
    /// never contend however they are bound, and nothing in an `Algo` says
    /// which silicon it landed on. Keeping that out of here is what stops this
    /// predicate quietly becoming a scheduler.
    pub fn contends_with(&self, other: &Algo) -> bool {
        self.bound() == other.bound()
    }

    /// A human-readable parameter line for the report.
    pub fn detail(&self) -> String {
        match self {
            Algo::Sha256d => String::from("sha256d"),
            Algo::Blake2s => String::from("blake2s (RFC 7693)"),
            Algo::Neoscrypt => String::from("neoscrypt (N=128 r=2, ChaCha+Salsa)"),
            Algo::Yespower { v10, n, r, pers } => {
                let mut s = String::from(if *v10 { "yespower 1.0 N=" } else { "yespower 0.5 N=" });
                push_u32(&mut s, *n);
                s.push_str(" r=");
                push_u32(&mut s, *r);
                if let Some(p) = pers {
                    s.push_str(" pers=");
                    for b in p.iter().take(16) {
                        s.push(*b as char);
                    }
                }
                s
            }
        }
    }
}

fn push_u32(s: &mut String, mut v: u32) {
    if v == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    for &b in &buf[i..] {
        s.push(b as char);
    }
}

/// A prepared hasher, holding whatever working set its algorithm needs.
pub enum Hasher {
    Sha256d(hash::Midstate),
    Blake2s(blake2s::Midstate),
    Yespower(Yespower, Option<Vec<u8>>),
    Neoscrypt(Neoscrypt),
}

impl Hasher {
    /// `None` when the parameters are ones the algorithm refuses, or when the
    /// working set will not fit.
    pub fn new(algo: &Algo, header: &[u8; 80]) -> Option<Hasher> {
        match algo {
            Algo::Sha256d => Some(Hasher::Sha256d(hash::Midstate::new(header))),
            Algo::Blake2s => Some(Hasher::Blake2s(blake2s::Midstate::new(header))),
            Algo::Neoscrypt => Some(Hasher::Neoscrypt(Neoscrypt::new())),
            Algo::Yespower { v10, n, r, pers } => {
                let v = if *v10 { Version::V1_0 } else { Version::V0_5 };
                Some(Hasher::Yespower(Yespower::new(v, *n, *r)?, pers.clone()))
            }
        }
    }

    /// Bytes of working memory. What the slice budget in `design/mining.md` is
    /// actually spending, and the reason the supervisor will need to ask.
    pub fn footprint(&self) -> usize {
        match self {
            Hasher::Sha256d(_) => 0,
            Hasher::Blake2s(_) => 0,
            Hasher::Yespower(y, _) => y.footprint(),
            Hasher::Neoscrypt(n) => n.footprint(),
        }
    }

    /// Point an existing hasher at a new job's header.
    ///
    /// Separate from `new` because the two algorithms differ in what a new
    /// header costs. SHA-256d must re-absorb its constant 64 bytes, which is
    /// one compression; yespower does not depend on the header at all until
    /// `hash` is called, so rebuilding it per job would throw away and
    /// reallocate up to 8 MiB every time the pool sends work.
    pub fn retarget(&mut self, header: &[u8; 80]) {
        match self {
            Hasher::Sha256d(mid) => *mid = hash::Midstate::new(header),
            Hasher::Blake2s(mid) => *mid = blake2s::Midstate::new(header),
            // Neither depends on the header until `hash` is called, and
            // rebuilding either would throw away its working set -- 8 MiB for
            // yespower, 33 KiB for this one -- every time the pool sends work.
            Hasher::Yespower(..) | Hasher::Neoscrypt(..) => {}
        }
    }

    /// The digest for this header with `nonce` substituted.
    ///
    /// Little-endian at offset 76, because every multi-byte field in a Bitcoin
    /// header is and yespower chains inherited the header unchanged.
    pub fn hash(&mut self, header: &[u8; 80], nonce: u32) -> [u8; 32] {
        match self {
            // The midstate already holds the constant 64 bytes, so the header
            // argument is unused here -- and it must stay that way, because a
            // midstate built from a *different* header would hash a block that
            // never existed while looking perfectly healthy.
            Hasher::Sha256d(mid) => mid.hash_with(nonce),
            Hasher::Blake2s(mid) => mid.hash_with(nonce),
            Hasher::Yespower(y, pers) => {
                let mut h = *header;
                h[76..80].copy_from_slice(&nonce.to_le_bytes());
                y.hash(&h, pers.as_deref())
            }
            Hasher::Neoscrypt(n) => {
                let mut h = *header;
                h[76..80].copy_from_slice(&nonce.to_le_bytes());
                n.hash(&h)
            }
        }
    }
}
