//! The kernel's hash core, checked on the host against published vectors.
//!
//! Every constant here came from outside this project: a block anyone can look
//! up, an RFC, and upstream yespower's own `TESTS-OK`. That they also appear in
//! `src/mine/mod.rs` and in `tools/algocheck.py` is the arrangement rather than
//! a duplication to be tidied away -- **a published vector is a fact about the
//! world and a third holder of it is a third checker.** What must never be
//! duplicated is the *implementation*, and it is not: these tests run the
//! kernel's own source, included by `#[path]`.
//!
//! What this catches that a boot selftest cannot is a change to the *host*
//! side. The pool validates shares by computing the miner's hash, so the day
//! those disagree the pool credits nothing and blames the miner.

use glados_pool::mine::algo::{Algo, Hasher};
use glados_pool::mine::stratum::unhex;
use glados_pool::mine::{blake2s, hash, header, proto, u256};

/// Block 125552, whose id is public and checkable against any explorer.
const BTC_HEADER_HEX: &str = concat!(
    "01000000",
    "81cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000",
    "e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122b",
    "c7f5d74d",
    "f2b9441a",
    "42a14695",
);
/// The nonce that header carries, little-endian at offset 76.
const BTC_NONCE: u32 = 0x9546_a142;

fn h32(s: &str) -> [u8; 32] {
    let v = unhex(s).expect("test constant is not hex");
    assert_eq!(v.len(), 32, "test constant is not 32 bytes");
    let mut o = [0u8; 32];
    o.copy_from_slice(&v);
    o
}

fn btc_header() -> [u8; 80] {
    let v = unhex(BTC_HEADER_HEX).unwrap();
    assert_eq!(v.len(), 80);
    let mut o = [0u8; 80];
    o.copy_from_slice(&v);
    o
}

#[test]
fn sha256d_reproduces_block_125552() {
    let h = btc_header();
    let d = hash::sha256d(&h);
    // Bitcoin displays a block id reversed, which is the trap this pins down.
    let mut shown = d;
    shown.reverse();
    assert_eq!(
        shown,
        h32("00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d")
    );
}

#[test]
fn the_sha256d_midstate_agrees_with_the_whole_header() {
    let h = btc_header();
    let mid = hash::Midstate::new(&h);
    assert_eq!(mid.hash_with(BTC_NONCE), hash::sha256d(&h));
    // And moves with the nonce. A midstate ignoring its argument passes the
    // line above and mines one nonce forever.
    assert_ne!(mid.hash_with(BTC_NONCE ^ 1), hash::sha256d(&h));
}

#[test]
fn blake2s_matches_rfc_7693() {
    assert_eq!(
        blake2s::hash(b""),
        h32("69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9")
    );
    assert_eq!(
        blake2s::hash(b"abc"),
        h32("508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982")
    );
}

#[test]
fn blake2s_counts_bytes_and_not_blocks() {
    // RFC 7693 section 3.2's counter is the message length. Padding a short
    // final block without moving it makes these two collide, which is the one
    // misreading that yields a hash function looking entirely healthy.
    assert_ne!(blake2s::hash(b"a"), blake2s::hash(b"a\0"));
    // And a message of exactly one block must keep that block for the `last`
    // flag rather than compressing it as an interior one.
    assert_ne!(blake2s::hash(&[0u8; 64]), blake2s::hash(&[0u8; 128]));
}

#[test]
fn the_blake2s_midstate_agrees_with_the_whole_header() {
    let h = btc_header();
    let whole = h32("753d9b626a850d23b32a05e6d531fbe985af7a300b21505d3b080611e9940aeb");
    assert_eq!(blake2s::hash(&h), whole);
    let mid = blake2s::Midstate::new(&h);
    assert_eq!(mid.hash_with(BTC_NONCE), whole);
    assert_eq!(
        mid.hash_with(0x1234_5678),
        h32("d81f08a0b2790da2d71ddea1ea3f6a76688eb996b148aa7d4463f177506e4411")
    );
}

/// Upstream yespower's own `TESTS-OK`, verbatim. Input is `src[i] = i * 3`
/// over 80 bytes, which is what `tests.c` hashes and is a header's length.
const YESPOWER: &[(bool, u32, u32, Option<&[u8]>, &str)] = &[
    (false, 2048, 8, Some(b"Client Key"), "a59fec4c4fdda16e3b1405adda66d525b68e7cadfcfe6ac066c7ad118cd80590"),
    (false, 4096, 16, Some(b"Client Key"), "927e72d0ded3d80475473f40f1743c67289d453d5242d4f55af4e325e06699c5"),
    (false, 4096, 24, Some(b"Jagaricoin"), "0e1366973211e7fea8ad9d81989c84a254d968c9d333dd8ff099324f38611e04"),
    (false, 4096, 32, Some(b"WaviBanana"), "3ae05abb3c5cf6f75415a92554c98d50e38ec9552cfa78373616f480b24e559f"),
    (false, 2048, 32, Some(b"Client Key"), "560a891b5ca2e1c636111a9ff7c894a5d0a2602f43fdcfa5949b95e22fe4461e"),
    (false, 1024, 32, Some(b"Client Key"), "2a79e53d1be6669bc556ccc417bce3d22a74a232f56b8e1d39b45792675de108"),
    (false, 2048, 8, None, "5ecbd8e8d7c90baed4bbf8916a1225dcc3c65f5c9165bae81cdde3cffad128e8"),
    (true, 2048, 8, None, "69e0e895b3df7aeeb837d71fe199e9d34f7ec46ecbca7a2c4308e51857ae9b46"),
    (true, 4096, 16, None, "33fb8f063824a4a020f63dca535f5ca66ab5576468c75d1ccaac7542f76495ac"),
    (true, 4096, 32, None, "771aeefda8fe79a0825bc7f2aee162ab5578574639ffc6ca3723cc18e5e3e285"),
    (true, 2048, 32, None, "d5efb813cd263e9b34540130233cbbc6a921fbff3431e5ec1a1abde2aea6ff4d"),
    (true, 1024, 32, None, "501b792db42e388f6e7d453c95d03a12a36016a5154a688390ddc609a40c6799"),
    (true, 1024, 32, Some(b"personality test"), "1f0269acf565c49adc0ef9b8f26ab3808cdc38394a254fddeedcc3aacff6ad9d"),
];

#[test]
fn yespower_matches_every_upstream_vector() {
    let src: [u8; 80] = core::array::from_fn(|i| (i as u32 * 3) as u8);
    for (v10, n, r, pers, want) in YESPOWER {
        let algo = Algo::Yespower {
            v10: *v10,
            n: *n,
            r: *r,
            pers: pers.map(|p| p.to_vec()),
        };
        // Through `Hasher` rather than through `Yespower` directly, because
        // `Hasher` is the path the miner actually takes: a correct algorithm
        // reached through a wrong seam is still a rejected share.
        let mut h = Hasher::new(&algo, &src).expect("upstream parameters refused");
        // The nonce lives at offset 76 of the input, so hashing `src` with the
        // nonce it already contains has to reproduce upstream's digest.
        let nonce = u32::from_le_bytes([src[76], src[77], src[78], src[79]]);
        assert_eq!(h.hash(&src, nonce), h32(want), "v10={v10} n={n} r={r}");
    }
}

/// The four names the world calls "yescrypt", checked against the table above.
///
/// zpool and cpuminer-opt both use these spellings, and every one of them is
/// `yespower` 0.5 with different parameters -- cpuminer's own
/// `register_yescrypt_algo` is commented "Legacy Yescrypt (yespower v0.5)". So
/// `parse_algo` expands them rather than implementing anything, and what has to
/// be true is that the expansion lands on parameters `YESPOWER` already pins.
///
/// **Checked against the vectors, not against itself.** The parameters are
/// typed into two files -- `shell::parse_algo` and `record::parse_algo` -- and a
/// digit wrong in either hashes a different function perfectly correctly and has
/// every share rejected, which is the exact failure `shell::parse_algo` refuses
/// presets to avoid. Looking the alias up in a second copy of the same table
/// would prove nothing; reproducing upstream's digest proves it.
#[test]
fn every_yescrypt_alias_expands_to_a_verified_vector() {
    let src: [u8; 80] = core::array::from_fn(|i| (i as u32 * 3) as u8);
    let nonce = u32::from_le_bytes([src[76], src[77], src[78], src[79]]);

    // (alias, the vector it must reproduce)
    let aliases: &[(&str, &str)] = &[
        ("yescrypt", "5ecbd8e8d7c90baed4bbf8916a1225dcc3c65f5c9165bae81cdde3cffad128e8"),
        ("yescryptr8", "a59fec4c4fdda16e3b1405adda66d525b68e7cadfcfe6ac066c7ad118cd80590"),
        ("yescryptr16", "927e72d0ded3d80475473f40f1743c67289d453d5242d4f55af4e325e06699c5"),
        ("yescryptr32", "3ae05abb3c5cf6f75415a92554c98d50e38ec9552cfa78373616f480b24e559f"),
    ];
    for (spec, want) in aliases {
        let algo = glados_pool::record::parse_algo(spec)
            .unwrap_or_else(|e| panic!("the alias '{spec}' does not parse: {e}"));
        // Every expected digest must also be a row of YESPOWER, or this test is
        // checking an alias against a constant nobody else believes.
        assert!(
            YESPOWER.iter().any(|(_, _, _, _, v)| v == want),
            "the alias '{spec}' expects a digest that is not an upstream vector"
        );
        let mut h = Hasher::new(&algo, &src).expect("alias expands to refused parameters");
        assert_eq!(h.hash(&src, nonce), h32(want), "alias {spec}");
    }
}

/// A digest whose *value* is the number this hex spells.
///
/// `below_target` reads a digest little-endian, because a block hash is a
/// 256-bit integer stored least-significant byte first -- which is why a
/// Bitcoin block id is displayed reversed. So a digest meant to equal a
/// big-endian target is that target's bytes reversed.
fn digest_worth(be_hex: &str) -> [u8; 32] {
    let mut d = h32(be_hex);
    d.reverse();
    d
}

#[test]
fn a_target_comparison_is_a_real_comparison() {
    // This test asserted the right three outcomes for the wrong reason first.
    // It built its digests big-endian, so "equal to the target" was in fact a
    // number vastly below it and "one over" was vastly above -- three passing
    // assertions, none of which touched the boundary they were named for. A
    // comparison that only ever sees values orders of magnitude apart would
    // pass with almost any implementation, including the leading-zero count
    // this is here to rule out.
    let t = u256::U256::from_be_bytes(&h32(
        "00000000ffff0000000000000000000000000000000000000000000000000000",
    ));
    let equal = digest_worth("00000000ffff0000000000000000000000000000000000000000000000000000");
    let under = digest_worth("00000000fffe0000000000000000000000000000000000000000000000000000");
    let over = digest_worth("00000000ffff0000000000000000000000000000000000000000000000000001");

    assert!(hash::below_target(&equal, &t), "equal to the target must pass");
    assert!(hash::below_target(&under, &t), "one below must pass");
    assert!(!hash::below_target(&over, &t), "one above must not");

    // And byte order is load-bearing rather than incidental. These two arrays
    // are each other reversed, and as little-endian integers they are 1 and
    // 2^248 -- so a comparison that ignored order would have to give them the
    // same answer, and no correct one can.
    let one = {
        let mut d = [0u8; 32];
        d[0] = 1;
        d
    };
    let vast = {
        let mut d = [0u8; 32];
        d[31] = 1;
        d
    };
    assert!(hash::below_target(&one, &t), "a digest worth 1 must pass");
    assert!(!hash::below_target(&vast, &t), "a digest worth 2^248 must not");
}

// ------------------------------------------------------------ the protocol

fn roundtrip_job(j: &proto::Job) -> proto::Job {
    let line = proto::encode_job(j);
    assert!(line.ends_with('\n'), "a frame must be one line");
    let v = glados_pool::json::Json::parse(line.trim_end()).expect("encoder emitted bad JSON");
    assert_eq!(v.get("method").unwrap().as_str(), Some("glados.job"));
    proto::parse_job(v.get("params").unwrap()).expect("decoder refused the encoder's output")
}

fn a_job(algo: Algo) -> proto::Job {
    proto::Job {
        slot: 2,
        coin: String::from("bitzeny"),
        job: String::from("a3f1"),
        algo,
        header: btc_header(),
        target: u256::U256::from_be_bytes(&h32(
            "00000000ffff0000000000000000000000000000000000000000000000000000",
        )),
        echo: vec![
            (String::from("extranonce2"), String::from("00000001")),
            (String::from("ntime"), String::from("4dd7f5c7")),
        ],
        clean: true,
        proof: None,
    }
}

#[test]
fn a_job_survives_the_round_trip_for_every_algorithm() {
    for algo in [
        Algo::Sha256d,
        Algo::Blake2s,
        Algo::Yespower { v10: true, n: 2048, r: 8, pers: None },
        // The parameterised case is the one that matters. A codec that dropped
        // `pers` would produce a job that hashes a different function
        // perfectly correctly and has every share rejected.
        Algo::Yespower {
            v10: false,
            n: 4096,
            r: 32,
            pers: Some(b"WaviBanana".to_vec()),
        },
    ] {
        let j = a_job(algo.clone());
        let back = roundtrip_job(&j);
        assert_eq!(back.slot, j.slot);
        assert_eq!(back.coin, j.coin);
        assert_eq!(back.job, j.job);
        assert_eq!(back.header, j.header);
        assert_eq!(back.target.to_be_bytes(), j.target.to_be_bytes());
        assert_eq!(back.clean, j.clean);
        assert_eq!(back.echo, j.echo);
        assert!(back.algo == algo, "the algorithm did not survive the wire");
    }
}

#[test]
fn the_algorithm_is_carried_and_not_assumed() {
    // The whole reason this protocol exists rather than Stratum V1: two jobs
    // identical but for the proof-of-work must decode differently.
    let a = roundtrip_job(&a_job(Algo::Sha256d));
    let b = roundtrip_job(&a_job(Algo::Blake2s));
    assert!(a.algo != b.algo);
    let mut ha = Hasher::new(&a.algo, &a.header).unwrap();
    let mut hb = Hasher::new(&b.algo, &b.header).unwrap();
    assert_ne!(ha.hash(&a.header, 7), hb.hash(&b.header, 7));
}

/// A proof survives the wire, and still proves the header afterwards.
///
/// The round-trip alone is not the claim -- fields could survive encoding and
/// still describe nothing. What matters is that  still answers yes on
/// the far side, since that is the only thing a miner actually does with it.
#[test]
fn a_proof_survives_the_round_trip_and_still_proves() {
    let c1 = unhex("0100000001").unwrap();
    let e = unhex("deadbeef00000001").unwrap();
    let c2 = unhex("ffffffff0100f2052a01000000434104").unwrap();
    let branch = [
        h32("aa00000000000000000000000000000000000000000000000000000000000001"),
        h32("bb00000000000000000000000000000000000000000000000000000000000002"),
    ];
    let coinbase = header::coinbase(&c1, &e, &[], &c2);
    let root = header::merkle_root(&coinbase, &branch);
    let prev = h32("81cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000");
    let hdr = header::assemble(1, &prev, &root, 0x4dd7_f5c7, 0x1a44_b9f2, 0);

    let mut j = a_job(Algo::Sha256d);
    j.header = hdr;
    j.proof = Some(proto::Proof {
        coinb1: c1,
        extranonce: e,
        coinb2: c2,
        branch: branch.to_vec(),
    });
    assert!(proto::proves(j.proof.as_ref().unwrap(), &j.header));

    let back = roundtrip_job(&j);
    let p = back.proof.expect("the proof did not survive the wire");
    assert!(
        proto::proves(&p, &back.header),
        "the proof arrived but no longer proves the header"
    );
}

#[test]
fn a_share_survives_the_round_trip() {
    let sh = proto::Share {
        job: String::from("a3f1"),
        nonce: 0x9546_a142,
        echo: vec![(String::from("extranonce2"), String::from("00000001"))],
    };
    let line = proto::encode_submit(9, &sh);
    let v = glados_pool::json::Json::parse(line.trim_end()).unwrap();
    assert_eq!(v.get("id").unwrap().as_i64(), Some(9));
    let back = proto::parse_share(v.get("params").unwrap()).unwrap();
    assert_eq!(back.nonce, sh.nonce);
    assert_eq!(back.job, sh.job);
    assert_eq!(back.echo, sh.echo);
}

#[test]
fn the_handshake_survives_the_round_trip() {
    let line = proto::encode_hello(1, "gl4d0s.rig1", "glados/1.3.5");
    let v = glados_pool::json::Json::parse(line.trim_end()).unwrap();
    let h = proto::parse_hello(v.get("params").unwrap()).unwrap();
    assert_eq!(h.v, proto::VERSION);
    assert_eq!(h.worker, "gl4d0s.rig1");

    let w = proto::Welcome { v: proto::VERSION, slots: 4, session: String::from("s1") };
    let line = proto::encode_welcome(1, &w);
    let v = glados_pool::json::Json::parse(line.trim_end()).unwrap();
    let back = proto::parse_welcome(v.get("result").unwrap()).unwrap();
    assert_eq!(back.slots, 4);
    assert_eq!(back.session, "s1");
}

#[test]
fn a_request_for_work_names_one_slot_and_survives_the_round_trip() {
    let line = proto::encode_work(7, 2);
    let v = glados_pool::json::Json::parse(line.trim_end()).unwrap();
    assert_eq!(v.get("id").unwrap().as_i64(), Some(7));
    assert_eq!(v.get("method").unwrap().as_str(), Some("glados.work"));
    assert_eq!(proto::parse_work(v.get("params").unwrap()), Some(2));

    // Slot zero is a real slot and must not read as absent -- the first coin
    // in the table is the one a single-algorithm device is most likely to be
    // working, so a falsy zero would break exactly the case this exists for.
    let line = proto::encode_work(8, 0);
    let v = glados_pool::json::Json::parse(line.trim_end()).unwrap();
    assert_eq!(proto::parse_work(v.get("params").unwrap()), Some(0));

    // No slot at all, and a slot outside the range a `u32` can hold, are
    // refused rather than defaulted. Defaulting to zero would send a fast
    // device the wrong coin's work and report nothing.
    let v = glados_pool::json::Json::parse("{\"nothing\":1}").unwrap();
    assert_eq!(proto::parse_work(&v), None);
    let v = glados_pool::json::Json::parse("{\"slot\":-1}").unwrap();
    assert_eq!(proto::parse_work(&v), None);
}

#[test]
fn a_malformed_job_is_refused_rather_than_padded() {
    let good = proto::encode_job(&a_job(Algo::Sha256d));
    let v = glados_pool::json::Json::parse(good.trim_end()).unwrap();
    assert!(proto::parse_job(v.get("params").unwrap()).is_some());

    // A short header is the case that must not be tolerated: zero-padding one
    // produces a header that hashes perfectly and belongs to no chain.
    let short = good.replace(&BTC_HEADER_HEX[..160], &BTC_HEADER_HEX[..158]);
    let v = glados_pool::json::Json::parse(short.trim_end()).unwrap();
    assert!(proto::parse_job(v.get("params").unwrap()).is_none());

    // An algorithm this build does not know is declined, not guessed at.
    let alien = good.replace("\"sha256d\"", "\"randomx\"");
    let v = glados_pool::json::Json::parse(alien.trim_end()).unwrap();
    assert!(proto::parse_job(v.get("params").unwrap()).is_none());
}

#[test]
fn the_optional_proof_reproduces_the_header_it_came_with() {
    // The trust cost of sending an assembled header is that a miner cannot see
    // which address the block pays. This is the check that answers it, and the
    // reason `header::coinbase` and `merkle_root` stay in the kernel: given the
    // coinbase halves and the branch, the miner rebuilds the header and
    // requires it to equal the one it was sent.
    let c1 = unhex("0100000001").unwrap();
    let e1 = unhex("deadbeef").unwrap();
    let e2 = unhex("00000001").unwrap();
    let c2 = unhex("ffffffff0100f2052a01000000434104").unwrap();
    let coinbase = header::coinbase(&c1, &e1, &e2, &c2);
    let branch: [[u8; 32]; 2] = [
        h32("aa00000000000000000000000000000000000000000000000000000000000001"),
        h32("bb00000000000000000000000000000000000000000000000000000000000002"),
    ];
    let root = header::merkle_root(&coinbase, &branch);

    let prev = h32("81cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000");
    let built = header::assemble(1, &prev, &root, 0x4dd7_f5c7, 0x1a44_b9f2, 0);

    // Same inputs, same header. What the miner would actually do is compare
    // this against the `header` field of the job, and refuse the job when they
    // differ -- which is a complete check, since a matching header proves the
    // coinbase it was shown is the coinbase that was committed to.
    let again = header::merkle_root(&header::coinbase(&c1, &e1, &e2, &c2), &branch);
    assert_eq!(root, again);
    assert_eq!(&built[36..68], &root[..]);

    // And a branch in the wrong order lands somewhere else, so the check has
    // content rather than passing on any branch at all.
    let flipped: [[u8; 32]; 2] = [branch[1], branch[0]];
    assert_ne!(header::merkle_root(&coinbase, &flipped), root);
}

/// **The declared working set against the one actually allocated.**
///
/// `Algo::working_set` is a formula and `Yespower::footprint` is the
/// allocation, written separately because asking the second costs eight
/// megabytes and a scheduler cannot pay that to decide what to run next. Two
/// expressions of one quantity is the arrangement this tree warns about, so it
/// is checked: a table claiming two megabytes where the allocator takes eight
/// would pack a device wrong and report nothing.
#[test]
fn the_declared_working_set_is_the_one_yespower_allocates() {
    use glados_pool::mine::yespower::{Version, Yespower};
    for (v10, n, r) in [
        (true, 1024u32, 8u32),
        (true, 2048, 8),
        (true, 1024, 32),
        (false, 1024, 8),
        (false, 2048, 16),
    ] {
        let a = Algo::Yespower { v10, n, r, pers: None };
        let ver = if v10 { Version::V1_0 } else { Version::V0_5 };
        let y = Yespower::new(ver, n, r).expect("constructible");
        assert_eq!(
            a.working_set(),
            y.footprint(),
            "v10={v10} N={n} r={r}"
        );
    }
}

/// The predicate the scheduler will rest on. Two algorithms sharing a device
/// contend only when they wait on the same thing -- two arithmetic ones halve
/// each other, while an arithmetic one beside a memory-bound one is close to
/// free. That difference is the only reason running several coins at once does
/// more work rather than the same work divided up.
#[test]
fn contention_follows_the_bound_and_not_the_algorithm() {
    use glados_pool::mine::algo::Bound;
    let yes = Algo::Yespower { v10: true, n: 2048, r: 8, pers: None };
    assert_eq!(Algo::Sha256d.bound(), Bound::Arithmetic);
    assert_eq!(Algo::Blake2s.bound(), Bound::Arithmetic);
    assert_eq!(yes.bound(), Bound::Memory);

    assert!(Algo::Sha256d.contends_with(&Algo::Blake2s));
    assert!(!Algo::Sha256d.contends_with(&yes));
    // Reflexive, or a slot would not contend with a second copy of itself --
    // which is the commonest case a scheduler actually meets.
    assert!(Algo::Sha256d.contends_with(&Algo::Sha256d));
    assert!(yes.contends_with(&yes));

    // And the size gap that justifies the split, asserted rather than told.
    assert!(Algo::Sha256d.working_set() <= 4096);
    assert!(yes.working_set() > 2 * 1024 * 1024);
}

// --------------------------------------------------------------- PPLNS

/// **The window is bounded by work, not by shares, and this is why.**
///
/// VarDiff means one worker's share is not another's: measured on one sweep,
/// 394 shares at 10 bits against 5 at 16, for the same effort. A window of
/// "the last 100 shares" would hand the low-difficulty miner almost all of it.
#[test]
fn the_payout_window_is_measured_in_work() {
    use glados_pool::pool::Window;
    let mut w = Window::default();
    // 8 shares of 16 units each into a window holding 64.
    for i in 0..8 {
        w.push(if i % 2 == 0 { "even" } else { "odd" }, 16, 64);
    }
    assert!(w.total() >= 64, "window holds at least the limit: {}", w.total());
    // The smallest suffix that does: 64 exactly, which is four shares.
    assert_eq!(w.total(), 64);
    assert_eq!(w.len(), 4);
    let by = w.shares_by_worker();
    assert_eq!(by.len(), 2);
    assert_eq!(by[0].1, 32);
    assert_eq!(by[1].1, 32);
}

/// One big share and many small ones. The big one must not be split, because a
/// share is one miner's one discovery and half of it is not a thing.
#[test]
fn a_share_is_never_divided_to_fit_the_window() {
    use glados_pool::pool::Window;
    let mut w = Window::default();
    w.push("whale", 1000, 100);
    assert_eq!(w.total(), 1000);
    assert_eq!(w.len(), 1);
    // And it leaves as soon as anything else can carry the window alone.
    w.push("minnow", 100, 100);
    assert_eq!(w.len(), 1);
    assert_eq!(w.shares_by_worker()[0].0, "minnow");
}

/// **The property that makes PPLNS resist pool-hopping.** A worker who stops
/// gets diluted out of the window by everybody who kept going -- their slice
/// falls to zero on its own, without anything having to notice they left.
#[test]
fn a_worker_who_stops_falls_out_of_the_window() {
    use glados_pool::pool::Window;
    let mut w = Window::default();
    for _ in 0..10 {
        w.push("hopper", 10, 100);
    }
    assert_eq!(w.shares_by_worker()[0].0, "hopper");
    // The hopper leaves; somebody else keeps hashing.
    for _ in 0..10 {
        w.push("steady", 10, 100);
    }
    let by = w.shares_by_worker();
    assert_eq!(by.len(), 1, "the hopper is gone entirely");
    assert_eq!(by[0].0, "steady");
}

/// Ordering is total, so a published document does not look edited every time
/// it is regenerated -- the property `ledger` already has.
#[test]
fn the_payout_order_is_the_same_on_every_run() {
    use glados_pool::pool::Window;
    let build = || {
        let mut w = Window::default();
        for name in ["c", "a", "b", "a", "b", "a"] {
            w.push(name, 10, u64::MAX);
        }
        w.shares_by_worker()
    };
    let first = build();
    assert_eq!(first, build());
    // Largest first, ties by name: a=30, b=20, c=10.
    assert_eq!(first[0], (String::from("a"), 30));
    assert_eq!(first[1], (String::from("b"), 20));
    assert_eq!(first[2], (String::from("c"), 10));
}

/// A window of nothing pays only whoever found the last share, which is a
/// lottery with one ticket rather than a payout scheme.
#[test]
fn a_window_of_zero_work_is_refused() {
    use glados_pool::pool::{Coin, Pool, Source, target_with_leading_zeros};
    use glados_pool::mine::algo::Algo;
    let mut p = Pool::new(vec![Coin {
        label: String::from("t"),
        asset: String::from("t"),
        algo: Algo::Sha256d,
        share_bits: 8,
        share_target: target_with_leading_zeros(8),
        network_target: None,
        source: Source::Local,
        work: None,
        e2: 0,
    }]);
    let before = p.window_work();
    assert!(!p.set_window(0));
    assert_eq!(p.window_work(), before);
    assert!(p.set_window(4096));
    assert_eq!(p.window_work(), 4096);
}

/// **NeoScrypt, in the tests CI actually runs.**
///
/// `src/mine/neoscrypt.rs` carries its own claims and the kernel runs them at
/// boot, which CI cannot do -- it is a `no_std` UEFI binary with no host test
/// runner. The pool compiles the same file by `#[path]`, so this is where those
/// vectors become something a push can check. Two real Feathercoin blocks: the
/// network accepted each because its digest beat the target its own `nbits`
/// declares, which is a coincidence at 1 in 6.9e7 and 1 in 4.1e9.
#[test]
fn neoscrypt_reproduces_blocks_the_chain_accepted() {
    use glados_pool::mine::hash::below_target;
    use glados_pool::mine::stratum::hex;
    use glados_pool::mine::neoscrypt::Neoscrypt;
    use glados_pool::mine::u256::U256;

    const BLOCKS: [(&str, &str, u32); 2] = [
        ("0200000054aa94a46a70931d29f2a2ed3ee4ab5832cd6446a090f6f63292d004dd306e96\
          bca6f3f22928ee4aa468ed5d5ff0f0a31137c2f9e780e0a70411a4a39b98d91c1b364d54\
          dcdd3d1d52660400",
         "aac8eaaea3c756f584d884f2de9e0e8c401f3ba0a640a14221550d2d06000000",
         0x1d3ddddc),
        ("040000200a9245b1198825ab30d6dbae2b185e31f342d1058cc36851629bf435fc682b55\
          4d100771e4a23d210cd3a1503e3b1ca524e8482913ee5b4036edf4bd20db6a2993c0a16a\
          dc09011d002c5f48",
         "ff87010b15afee123759c6cb14fe33078d1795ca6f39286c3566d52500000000",
         0x1d0109dc),
    ];

    let mut n = Neoscrypt::new();
    for (hdr_hex, want, nbits) in BLOCKS {
        let bytes = unhex(&hdr_hex.replace([' ', '\n'], "")).expect("header is hex");
        let hdr: [u8; 80] = bytes.as_slice().try_into().expect("a header is 80 bytes");
        let got = n.hash(&hdr);
        assert_eq!(hex(&got), want, "nbits {nbits:08x}");
        // The part that makes it a proof rather than a stored answer: the
        // network only accepted this block because the digest beat its target.
        let target = U256::from_nbits(nbits).expect("nbits decodes");
        assert!(below_target(&got, &target), "digest is not under its own target");
    }
    // The same instance twice, because it holds a 32 KiB working set across
    // calls and a hasher that did not reset would pass the first and fail here.
    let bytes = unhex(&BLOCKS[0].0.replace([' ', '\n'], "")).unwrap();
    let hdr: [u8; 80] = bytes.as_slice().try_into().unwrap();
    assert_eq!(hex(&n.hash(&hdr)), BLOCKS[0].1);
}
