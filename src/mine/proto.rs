//! The protocol GLaDOS speaks to its own pool.
//!
//! `design/pool.md` is the argument; this is the codec. The short version of
//! the argument: Stratum V1 has no field saying which proof-of-work a job
//! wants, because a pool serves one coin and the miner is assumed to know. A
//! kernel working four coins at once over one connection needs that field, and
//! adding it to Stratum would be a dialect nobody else speaks anyway.
//!
//! ### The kernel never learns what a coinbase is
//!
//! A job is an **assembled 80-byte header with a zero nonce**, an algorithm
//! and a target. The miner substitutes a nonce at offset 76, hashes, compares,
//! and returns the nonce. Bitcoin's `mining.notify` instead hands over two
//! coinbase halves and a merkle branch and has the *miner* assemble the header
//! -- which for a dozen chains means a dozen transaction formats in ring 0, to
//! compute a number the pool already has.
//!
//! What that costs is real and is not waved away by the pool being ours: a
//! miner handed a finished header cannot see which address the block would
//! pay. `Job::proof` carries the coinbase halves and the branch when the pool
//! chooses to send them, and `header::coinbase` and `header::merkle_root` are
//! still here to check that they produce the header that arrived. Verified
//! once per job against several billion hashes, so it costs nothing.
//!
//! ### This file is shared with the pool, and that is the point
//!
//! `pool/src/lib.rs` includes it by `#[path]`, so there is one encoder and one
//! decoder rather than two that agree until they do not. A protocol is the
//! place that failure is most expensive and hardest to see: a field read
//! differently at each end produces a share rejected for a reason neither side
//! can name.
//!
//! ### `echo` is a flat map of strings, and the restriction is deliberate
//!
//! Whatever an upstream dialect needs handed back at submit time -- an
//! extranonce2, an ntime, a job blob id -- travels in `echo`, which the miner
//! stores without reading and returns verbatim. Verbatim is the requirement,
//! and `Json` discards source text, so re-emitting arbitrary JSON would mean
//! teaching the parser to keep it. A flat `string -> string` map re-emits
//! exactly, and every field any dialect has wanted so far is hex.

use alloc::string::String;
use alloc::vec::Vec;

use super::algo::Algo;
use super::stratum::{hex, unhex};
use super::u256::U256;
use crate::json::{write_str, Json};

/// Bumped when a change would make an old miner misread a job.
///
/// Checked at `hello` and refused rather than negotiated. A pool and a miner
/// that disagree about the wire have nothing to talk about, and a negotiation
/// is a second code path that only runs against versions nobody has any more.
pub const VERSION: u64 = 1;

/// Opaque round-trip fields. Ordered, so an encoding is a function of its
/// contents rather than of a map's iteration order -- two encoders that
/// disagree about field order produce different bytes for the same job, which
/// is a problem the moment anything hashes or signs one.
pub type Echo = Vec<(String, String)>;

/// What a pool sends so a miner can check what it is mining for.
///
/// A miner handed a finished header cannot see which address the block would
/// pay: the coinbase is inside a merkle root, and a root is a hash. This is the
/// answer, and `proves` below is the whole of the check.
///
/// **The extranonce arrives already spliced.** Stratum splits it into a
/// per-connection half and a per-job half because the miner has to vary the
/// second; here the pool varies it and the miner only ever rebuilds, so where
/// the boundary falls is not information the miner can use. One field is one
/// fewer thing to get wrong.
pub struct Proof {
    pub coinb1: Vec<u8>,
    pub extranonce: Vec<u8>,
    pub coinb2: Vec<u8>,
    pub branch: Vec<[u8; 32]>,
}

/// Does this proof produce the header it came with.
///
/// Only the merkle root is checked, and that is sufficient rather than
/// partial: the root is the single field a coinbase reaches, and everything
/// else in the header -- version, previous hash, time, bits -- belongs to the
/// chain rather than to the pool's payout. A root that matches proves the
/// coinbase the miner was *shown* is the coinbase that was *committed to*, and
/// a pool cannot show one coinbase and mine another without breaking it.
///
/// What it does not prove is that the chain wants this header at all. A pool
/// free-running its own template passes this check perfectly, which is why
/// `Source` is reported beside every coin.
pub fn proves(p: &Proof, header: &[u8; 80]) -> bool {
    let coinbase = super::header::coinbase(&p.coinb1, &p.extranonce, &[], &p.coinb2);
    let root = super::header::merkle_root(&coinbase, &p.branch);
    root[..] == header[36..68]
}

/// The coinbase a proof describes, for reading its outputs.
pub fn coinbase_of(p: &Proof) -> Vec<u8> {
    super::header::coinbase(&p.coinb1, &p.extranonce, &[], &p.coinb2)
}

pub struct Job {
    /// Which of the pool's coins this belongs to. The pool's numbering, not
    /// the miner's: the miner puts it in whichever local slot it likes, and a
    /// miner-chosen number would mean the pool keeping a per-connection map
    /// that can disagree with the miner's.
    pub slot: u32,
    pub coin: String,
    pub job: String,
    pub algo: Algo,
    pub header: [u8; 80],
    pub target: U256,
    pub echo: Echo,
    /// Abandon work on this slot's previous job. Per slot and never global:
    /// one coin's chain moving on says nothing about another's.
    pub clean: bool,
    /// Present when the pool is willing to show its working. Optional because
    /// a pool with no chain behind a coin has no meaningful coinbase to show,
    /// and sending a fabricated one would be worse than sending none.
    pub proof: Option<Proof>,
}

pub struct Share {
    pub job: String,
    pub nonce: u32,
    pub echo: Echo,
}

pub struct Hello {
    pub v: u64,
    pub worker: String,
    pub agent: String,
}

pub struct Welcome {
    pub v: u64,
    /// How many coins this connection may be given at once. The miner may work
    /// fewer; it must not expect more.
    pub slots: u32,
    pub session: String,
}

// ---------------------------------------------------------------- encoding

fn push_u64(s: &mut String, mut v: u64) {
    if v == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 20];
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

fn push_echo(s: &mut String, echo: &Echo) {
    s.push('{');
    for (i, (k, v)) in echo.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write_str(s, k);
        s.push(':');
        write_str(s, v);
    }
    s.push('}');
}

/// The algorithm, as a structure and never as a name.
///
/// `"yespower"` alone is not an algorithm: BitZeny, Yenten and Koto all run it
/// with different parameters, so a name-only field is a job that hashes a
/// different function perfectly correctly and has every share rejected. This is
/// `Algo`'s own refusal of a preset table, moved onto the wire.
fn push_algo(s: &mut String, a: &Algo) {
    s.push_str("{\"name\":");
    match a {
        Algo::Sha256d => {
            write_str(s, "sha256d");
            s.push('}');
        }
        Algo::Blake2s => {
            write_str(s, "blake2s");
            s.push('}');
        }
        // A bare name for Neoscrypt's reason: the matrix comes from the header,
        // so there is nothing a pool could vary.
        Algo::HeavyHash => {
            write_str(s, "heavyhash");
            s.push('}');
        }
        // A bare name, and that is not the shortcut it looks like. N, r and
        // the round count *are* the NeoScrypt profile rather than settings, so
        // there is nothing a pool could vary and nothing a miner could get
        // wrong -- which is precisely why yespower cannot be sent this way.
        Algo::Neoscrypt => {
            write_str(s, "neoscrypt");
            s.push('}');
        }
        Algo::Yespower { v10, n, r, pers } => {
            write_str(s, "yespower");
            s.push_str(",\"v10\":");
            s.push_str(if *v10 { "true" } else { "false" });
            s.push_str(",\"n\":");
            push_u64(s, *n as u64);
            s.push_str(",\"r\":");
            push_u64(s, *r as u64);
            if let Some(p) = pers {
                // Hex, because personalisation is arbitrary bytes. Upstream's
                // own vectors use ASCII ("Client Key", "WaviBanana") and it
                // would be tempting to send it as text -- but nothing in the
                // algorithm requires that, and a chain using a byte outside
                // UTF-8 would produce a job no JSON string can carry.
                s.push_str(",\"pers\":");
                write_str(s, &hex(p));
            }
            s.push('}');
        }
    }
}

pub fn encode_job(j: &Job) -> String {
    let mut s = String::new();
    s.push_str("{\"method\":\"glados.job\",\"params\":{\"slot\":");
    push_u64(&mut s, j.slot as u64);
    s.push_str(",\"coin\":");
    write_str(&mut s, &j.coin);
    s.push_str(",\"job\":");
    write_str(&mut s, &j.job);
    s.push_str(",\"algo\":");
    push_algo(&mut s, &j.algo);
    s.push_str(",\"header\":");
    write_str(&mut s, &hex(&j.header));
    s.push_str(",\"target\":");
    write_str(&mut s, &hex(&j.target.to_be_bytes()));
    s.push_str(",\"echo\":");
    push_echo(&mut s, &j.echo);
    s.push_str(",\"clean\":");
    s.push_str(if j.clean { "true" } else { "false" });
    if let Some(p) = &j.proof {
        s.push_str(",\"proof\":{\"coinb1\":");
        write_str(&mut s, &hex(&p.coinb1));
        s.push_str(",\"extranonce\":");
        write_str(&mut s, &hex(&p.extranonce));
        s.push_str(",\"coinb2\":");
        write_str(&mut s, &hex(&p.coinb2));
        s.push_str(",\"branch\":[");
        for (i, b) in p.branch.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write_str(&mut s, &hex(b));
        }
        s.push_str("]}");
    }
    s.push_str("}}\n");
    s
}

pub fn encode_hello(id: u64, worker: &str, agent: &str) -> String {
    let mut s = String::new();
    s.push_str("{\"id\":");
    push_u64(&mut s, id);
    s.push_str(",\"method\":\"glados.hello\",\"params\":{\"v\":");
    push_u64(&mut s, VERSION);
    s.push_str(",\"worker\":");
    write_str(&mut s, worker);
    s.push_str(",\"agent\":");
    write_str(&mut s, agent);
    s.push_str("}}\n");
    s
}

pub fn encode_welcome(id: u64, w: &Welcome) -> String {
    let mut s = String::new();
    s.push_str("{\"id\":");
    push_u64(&mut s, id);
    s.push_str(",\"result\":{\"v\":");
    push_u64(&mut s, w.v);
    s.push_str(",\"slots\":");
    push_u64(&mut s, w.slots as u64);
    s.push_str(",\"session\":");
    write_str(&mut s, &w.session);
    s.push_str("},\"error\":null}\n");
    s
}

/// A miner saying it has spent a slot's nonce space and wants a fresh job.
///
/// **A job is a finite search and a fast device finishes it.** The header is
/// assembled by the pool with a zero nonce, so the whole of the work a job
/// carries is the 2^32 values at offset 76 -- which at half a gigahash a
/// second is eight and a half seconds. The pool re-issues on a thirty-second
/// idle timeout, so without this a device four times too fast for its work
/// simply rescans the space it has already searched and submits the same
/// nonces again. Measured on the RTX 3050 before this existed: 23 accepted
/// shares against 35 duplicates, every repeated nonce found exactly six times.
///
/// It names one slot rather than asking for everything, because the reason to
/// ask is always about one coin, and re-issuing all of them would churn the
/// pool's job ring for no one's benefit -- which is the shape of a bug this
/// pool has already had once.
pub fn encode_work(id: u64, slot: u32) -> String {
    let mut s = String::new();
    s.push_str("{\"id\":");
    push_u64(&mut s, id);
    s.push_str(",\"method\":\"glados.work\",\"params\":{\"slot\":");
    push_u64(&mut s, slot as u64);
    s.push_str("}}
");
    s
}

pub fn encode_submit(id: u64, sh: &Share) -> String {
    let mut s = String::new();
    s.push_str("{\"id\":");
    push_u64(&mut s, id);
    s.push_str(",\"method\":\"glados.submit\",\"params\":{\"job\":");
    write_str(&mut s, &sh.job);
    s.push_str(",\"nonce\":");
    // Big-endian hex, matching `header::submit_hex`, so a nonce that reaches an
    // upstream Stratum pool is spelled the one way that pool expects and the
    // conversion happens nowhere.
    write_str(&mut s, &hex(&sh.nonce.to_be_bytes()));
    s.push_str(",\"echo\":");
    push_echo(&mut s, &sh.echo);
    s.push_str("}}\n");
    s
}

// ---------------------------------------------------------------- decoding

fn take_echo(j: Option<&Json>) -> Echo {
    let mut out: Echo = Vec::new();
    // Absent is empty rather than an error. A dialect that needs nothing back
    // is the ordinary case for a solo-mined chain, and requiring the field
    // would make every such job carry `"echo":{}`.
    let Some(Json::Obj(pairs)) = j else {
        return out;
    };
    for (k, v) in pairs.iter() {
        // Non-string values are dropped rather than stringified. The contract
        // is verbatim round-trip, and a number re-emitted as a string is not
        // what arrived -- an upstream comparing it byte for byte would refuse
        // the share and the reason would be invisible at both ends.
        if let Some(s) = v.as_str() {
            out.push((k.clone(), String::from(s)));
        }
    }
    out
}

fn take_proof(j: Option<&Json>) -> Option<Proof> {
    let j = j?;
    let hexfield = |k: &str| -> Option<Vec<u8>> { unhex(j.get(k)?.as_str()?) };
    let mut branch: Vec<[u8; 32]> = Vec::new();
    match j.get("branch") {
        Some(Json::Arr(items)) => {
            for it in items.iter() {
                let b = unhex(it.as_str()?)?;
                // Exactly 32, and refused rather than padded. A short sibling
                // zero-extended folds to a root that is arithmetically fine and
                // describes nothing, which would turn a proof into a check that
                // always fails for a reason nobody could find.
                if b.len() != 32 {
                    return None;
                }
                let mut w = [0u8; 32];
                w.copy_from_slice(&b);
                branch.push(w);
            }
        }
        // An absent branch is a one-transaction block, which is legal and is
        // what a freshly-started chain looks like. `None` would be a missing
        // field; an empty list is a real answer.
        None => {}
        _ => return None,
    }
    Some(Proof {
        coinb1: hexfield("coinb1")?,
        extranonce: hexfield("extranonce")?,
        coinb2: hexfield("coinb2")?,
        branch,
    })
}

fn take_algo(j: &Json) -> Option<Algo> {
    match j.get("name")?.as_str()? {
        "sha256d" => Some(Algo::Sha256d),
        "blake2s" => Some(Algo::Blake2s),
        "neoscrypt" => Some(Algo::Neoscrypt),
        "heavyhash" => Some(Algo::HeavyHash),
        "yespower" => {
            let v10 = j.get("v10").and_then(|x| x.as_bool())?;
            let n = j.get("n").and_then(|x| x.as_i64())?;
            let r = j.get("r").and_then(|x| x.as_i64())?;
            // Refused rather than clamped, for the reason `Yespower::new`
            // refuses: a clamped parameter hashes a different function
            // perfectly correctly, and every share it finds is rejected by a
            // pool that will not say why.
            if !(0..=u32::MAX as i64).contains(&n) || !(0..=u32::MAX as i64).contains(&r) {
                return None;
            }
            let pers = match j.get("pers").and_then(|x| x.as_str()) {
                Some(h) => Some(unhex(h)?),
                None => None,
            };
            Some(Algo::Yespower {
                v10,
                n: n as u32,
                r: r as u32,
                pers,
            })
        }
        // An algorithm this build does not know is not an error in the pool
        // and must not read as one. The caller declines the job and says so.
        _ => None,
    }
}

pub fn parse_job(params: &Json) -> Option<Job> {
    let slot = params.get("slot")?.as_i64()?;
    if !(0..=u32::MAX as i64).contains(&slot) {
        return None;
    }
    let header_hex = params.get("header")?.as_str()?;
    let hb = unhex(header_hex)?;
    // Exactly eighty. A short header would be zero-padded into something that
    // hashes fine and belongs to no chain, which is the failure this whole
    // protocol is arranged to make impossible rather than merely unlikely.
    if hb.len() != 80 {
        return None;
    }
    let mut header = [0u8; 80];
    header.copy_from_slice(&hb);

    let tb = unhex(params.get("target")?.as_str()?)?;
    if tb.len() != 32 {
        return None;
    }
    let mut t = [0u8; 32];
    t.copy_from_slice(&tb);

    Some(Job {
        slot: slot as u32,
        coin: String::from(params.get("coin")?.as_str()?),
        job: String::from(params.get("job")?.as_str()?),
        algo: take_algo(params.get("algo")?)?,
        header,
        target: U256::from_be_bytes(&t),
        echo: take_echo(params.get("echo")),
        // Absent means false. A job that does not say is a job that adds work
        // rather than replacing it, which is the safe direction: mining a
        // stale template wastes a slice, and dropping a live one loses shares
        // nobody can tell were ever found.
        clean: params.get("clean").and_then(|x| x.as_bool()).unwrap_or(false),
        // A proof that will not parse is dropped rather than failing the job.
        // The job is still perfectly minable; what is lost is the ability to
        // check it, and `client` says so out loud rather than refusing work
        // over a field that is optional by design.
        proof: take_proof(params.get("proof")),
    })
}

/// The slot a `glados.work` asks about.
pub fn parse_work(params: &Json) -> Option<u32> {
    let n = params.get("slot")?.as_i64()?;
    if !(0..=u32::MAX as i64).contains(&n) {
        return None;
    }
    Some(n as u32)
}

pub fn parse_share(params: &Json) -> Option<Share> {
    let nb = unhex(params.get("nonce")?.as_str()?)?;
    if nb.len() != 4 {
        return None;
    }
    Some(Share {
        job: String::from(params.get("job")?.as_str()?),
        nonce: u32::from_be_bytes([nb[0], nb[1], nb[2], nb[3]]),
        echo: take_echo(params.get("echo")),
    })
}

pub fn parse_hello(params: &Json) -> Option<Hello> {
    let v = params.get("v")?.as_i64()?;
    if v < 0 {
        return None;
    }
    Some(Hello {
        v: v as u64,
        worker: String::from(params.get("worker")?.as_str()?),
        // An agent string is decoration and its absence is not a refusal.
        agent: String::from(
            params
                .get("agent")
                .and_then(|x| x.as_str())
                .unwrap_or("unknown"),
        ),
    })
}

pub fn parse_welcome(result: &Json) -> Option<Welcome> {
    let v = result.get("v")?.as_i64()?;
    let slots = result.get("slots")?.as_i64()?;
    if v < 0 || !(0..=u32::MAX as i64).contains(&slots) {
        return None;
    }
    Some(Welcome {
        v: v as u64,
        slots: slots as u32,
        session: String::from(result.get("session").and_then(|x| x.as_str()).unwrap_or("")),
    })
}
