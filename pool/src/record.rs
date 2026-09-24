//! A share, written down so somebody else can check it.
//!
//! `design/pool.md` says the share log is "the only thing standing in for
//! trust", because a non-custodial pool has no wallet to audit. But the log the
//! pool publishes is a *tally* -- work, accepted, stale, bad, duplicate per
//! worker -- and a tally cannot be re-verified. Nothing in it lets a third party
//! recompute a single hash. A miner's whole assurance is that the operator's own
//! program counted honestly.
//!
//! This is the missing half: one line per accepted share, carrying everything
//! needed to recompute it and nothing else. Anybody who has the file can check
//! every share in it, with code the operator did not run.
//!
//! ### The record is self-contained on purpose
//!
//! `Pool::submit` validates against the pool's own issued-job table, which is
//! live state that exists for about thirty seconds. A verifier running somewhere
//! else, later, has none of that -- so a record carries the assembled 80-byte
//! header rather than a job id, and the target rather than a difficulty. It is
//! four fields because those are exactly the four `Hasher` and `below_target`
//! need, and a fifth thing would be a field somebody has to trust.
//!
//! ### The algorithm is spelled in full, and that is the whole risk
//!
//! `Algo::name()` answers `"yespower-1.0"` and drops N, r and the
//! personalisation. A record carrying that would verify against *a* yespower and
//! not the one the miner ran, which is `design/equihash.md`'s first hazard
//! arriving on a different algorithm: the verifier is correct, the arithmetic is
//! correct, and every share fails for a reason nothing reports. So the spec is
//! the pool's own coin-spec form, `yespower-10-2048-8[-pershex]`, and
//! `parse_algo` is the *same function* the coin specs go through -- moved here
//! rather than copied, because two parsers for one format is the thing this
//! repository keeps writing differs to catch.
use crate::mine::algo::{Algo, Hasher};
use crate::mine::hash::below_target;
use crate::mine::stratum::unhex;
use crate::mine::u256::U256;

/// Parse an algorithm spec. The one parser; coin specs come through here too.
pub fn parse_algo(spec: &str) -> Result<Algo, String> {
    let mut parts = spec.split('-');
    match parts.next().unwrap_or("") {
        "sha256d" => Ok(Algo::Sha256d),
        "blake2s" => Ok(Algo::Blake2s),
        // No parameters after the name, unlike yespower: N, r and the round
        // count are the NeoScrypt profile rather than settings, so a chain
        // varying them would be a different proof of work.
        "neoscrypt" => Ok(Algo::Neoscrypt),
        "yespower" => {
            let v = parts.next().unwrap_or("");
            let n: u32 = parts
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| String::from("yespower needs N"))?;
            let r: u32 = parts
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| String::from("yespower needs r"))?;
            let pers = match parts.next() {
                Some(h) => Some(unhex(h).ok_or_else(|| String::from("pers is not hex"))?),
                None => None,
            };
            let v10 = match v {
                "10" => true,
                "05" => false,
                _ => return Err(String::from("yespower version is 10 or 05")),
            };
            Ok(Algo::Yespower { v10, n, r, pers })
        }
        other => Err(format!("no such algorithm '{other}'")),
    }
}

/// Render an algorithm back to the spec `parse_algo` reads.
///
/// **Round-tripped in `checks`, not assumed.** A renderer that disagreed with
/// the parser would write records that verify against a different hash, which is
/// the one failure in this module that produces confident wrong answers rather
/// than an error.
pub fn algo_spec(a: &Algo) -> String {
    match a {
        Algo::Sha256d => String::from("sha256d"),
        Algo::Blake2s => String::from("blake2s"),
        Algo::Neoscrypt => String::from("neoscrypt"),
        Algo::Yespower { v10, n, r, pers } => {
            let mut s = format!("yespower-{}-{}-{}", if *v10 { "10" } else { "05" }, n, r);
            if let Some(p) = pers {
                s.push('-');
                for b in p {
                    s.push_str(&format!("{b:02x}"));
                }
            }
            s
        }
    }
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

/// One accepted share, recomputable by anybody.
pub struct Record {
    pub algo: Algo,
    pub header: [u8; 80],
    pub nonce: u32,
    pub target: U256,
    pub worker: String,
}

impl Record {
    /// `algo header nonce target worker`, space separated, hex throughout.
    ///
    /// Text and one line each so a batch survives every channel it has to go
    /// through -- a git commit, a workflow input, an artifact -- without a
    /// framing format anybody has to agree about.
    pub fn render(&self) -> String {
        format!(
            "{} {} {:08x} {} {}",
            algo_spec(&self.algo),
            hex(&self.header),
            self.nonce,
            hex(&self.target.to_be_bytes()),
            if self.worker.is_empty() { "-" } else { self.worker.as_str() }
        )
    }

    /// `Ok(None)` for a blank line or a comment, so a file can be annotated.
    pub fn parse(line: &str) -> Result<Option<Record>, String> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }
        let mut f = line.split_whitespace();
        let algo_s = f.next().ok_or_else(|| String::from("no algorithm"))?;
        let hdr_s = f.next().ok_or_else(|| String::from("no header"))?;
        let nonce_s = f.next().ok_or_else(|| String::from("no nonce"))?;
        let tgt_s = f.next().ok_or_else(|| String::from("no target"))?;
        let worker = f.next().unwrap_or("-");

        let algo = parse_algo(algo_s)?;
        let hb = unhex(hdr_s).ok_or_else(|| String::from("header is not hex"))?;
        if hb.len() != 80 {
            return Err(format!("a header is 80 bytes, not {}", hb.len()));
        }
        let mut header = [0u8; 80];
        header.copy_from_slice(&hb);
        let nonce = u32::from_str_radix(nonce_s, 16)
            .map_err(|_| String::from("nonce is not hex"))?;
        let tb = unhex(tgt_s).ok_or_else(|| String::from("target is not hex"))?;
        if tb.len() != 32 {
            return Err(format!("a target is 32 bytes, not {}", tb.len()));
        }
        let mut t = [0u8; 32];
        t.copy_from_slice(&tb);
        Ok(Some(Record {
            algo,
            header,
            nonce,
            target: U256::from_be_bytes(&t),
            worker: String::from(worker),
        }))
    }

    /// Recompute the hash and answer whether it meets the target.
    ///
    /// `None` when the parameters will not build a hasher at all, which is a
    /// record the operator wrote wrongly rather than a share that failed.
    pub fn check(&self) -> Option<(bool, [u8; 32])> {
        let mut h = Hasher::new(&self.algo, &self.header)?;
        let d = h.hash(&self.header, self.nonce);
        Some((below_target(&d, &self.target), d))
    }
}

/// Claims about the format. No pool, no socket, no network.
pub fn checks() -> Vec<(bool, String)> {
    let mut out: Vec<(bool, String)> = Vec::new();
    let mut ok = |c: bool, w: &str| out.push((c, String::from(w)));

    // Every algorithm round-trips through the pair, including the one whose
    // parameters `Algo::name()` throws away.
    for a in [
        Algo::Sha256d,
        Algo::Blake2s,
        Algo::Neoscrypt,
        Algo::Yespower { v10: true, n: 2048, r: 8, pers: None },
        Algo::Yespower { v10: false, n: 4096, r: 16, pers: Some(vec![1, 2, 0xff]) },
    ] {
        let s = algo_spec(&a);
        let back = parse_algo(&s);
        ok(
            back.as_ref().map(|b| algo_spec(b)) == Ok(s.clone()),
            &format!("the spec '{s}' round-trips through parse and render"),
        );
    }
    ok(
        algo_spec(&Algo::Yespower { v10: true, n: 2048, r: 8, pers: None }) == "yespower-10-2048-8",
        "and yespower carries N and r, which Algo::name() drops",
    );

    // The block-125552 fixture the kernel's own suite uses, so a record that
    // verifies here is verifying against a header anybody can look up.
    let mut header = [0u8; 80];
    header[0] = 1;
    let target = crate::pool::target_with_leading_zeros(8);
    let r = Record {
        algo: Algo::Sha256d,
        header,
        nonce: 0x1234_5678,
        target,
        worker: String::from("rig1"),
    };
    let line = r.render();
    match Record::parse(&line) {
        Ok(Some(b)) => {
            ok(b.header == r.header, "a record's header survives the round trip");
            ok(b.nonce == r.nonce, "and its nonce");
            ok(b.target == r.target, "and its target");
            ok(b.worker == "rig1", "and the worker it belongs to");
            ok(b.check().map(|x| x.1) == r.check().map(|x| x.1),
               "and both compute the same digest");
        }
        other => ok(false, &format!("a rendered record parses ({other:?})", other = other.map(|_| ()))),
    }

    ok(Record::parse("").unwrap().is_none(), "a blank line is not a record");
    ok(Record::parse("# a note").unwrap().is_none(), "nor is a comment");
    ok(Record::parse("sha256d 00 0 00").is_err(), "a short header is refused");
    ok(Record::parse("nosuch 00 0 00").is_err(), "and an algorithm nobody has");
    // The one that matters: a record whose nonce was edited must fail, or this
    // file is a log that agrees with whatever it is handed.
    let mut tampered = Record::parse(&line).unwrap().unwrap();
    tampered.nonce ^= 1;
    ok(
        tampered.check().map(|x| x.1) != r.check().map(|x| x.1),
        "changing one bit of the nonce changes the digest",
    );
    out
}
