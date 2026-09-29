//! Whether the coins this pool is about to serve can be sold.
//!
//! `tools/prices.py` writes the file; this reads it. Two parsers over one
//! format, the bargain `tokenizer.py --verify` makes and for the same reason:
//! the writer is Python and `json.dumps`, the reader is this tree's own
//! `Json`, and a field one of them gets wrong is a field the other disagrees
//! about rather than a field nobody checks.
//!
//! ### What it does not do, and why not yet
//!
//! It does not compute an expected value. The arithmetic is
//! `price x reward / (2^256 / network_target)` and the pool holds two of those
//! three -- `Coin::network_target` from a live `nbits`, `ev::coinbase_value`
//! from the block's own coinbase -- so it is a few lines away. It is not
//! written, because every coin configured today is `Source::Local`, a local
//! coin has no chain and therefore no network target, and code whose only
//! path is the one that answers "cannot say" is code nobody has run.
//!
//! There is a second reason and it is the more interesting one. A coinbase
//! output is in the chain's own base unit, and how many of those make a coin
//! is a per-chain constant this pool cannot read off the wire. Writing 1e8
//! because Bitcoin uses it is exactly the invented figure `ev.rs` refuses in
//! its own header. When there is a live upstream there is also a chain to read
//! it from, so the two blockers lift together.
//!
//! **And a third route makes both moot for the ranking question.** A multi-coin
//! auto-exchange pool publishes `estimate_current` per algorithm -- what a unit
//! of hashrate earned in a day, in BTC, after it sold what it mined. Price,
//! difficulty, reward and the decimals constant are all already inside that,
//! because somebody else did the selling and is quoting the proceeds.
//! `tools/payrate.py` reads it, and it is what actually settled which half of
//! this laptop is worth pointing at a pool.
//!
//! That answers "what should this machine mine". It does not answer "what is a
//! share on *our* pool worth", which is the question this file is eventually
//! for and which still wants the arithmetic above.
//!
//! ### The network target is not only an expected-value blocker
//!
//! Worth stating here because this is where its absence is recorded, and the
//! second consequence is larger than the first. `Window`'s doc explains it in
//! full: Rosenfeld proves simple PPLNS hopping-proof only while difficulty is
//! constant, and the hopping-proof variant stores each share's value of `p` --
//! share difficulty over *network* difficulty -- rather than raw work. So the
//! missing number is the difference between the scheme this pool runs and the
//! one it believes it runs, on chains that retarget often, which is precisely
//! the class `payrate.py` selects for.
//!
//! It is also what would let the window's own variance-against-maturity
//! tradeoff be printed at startup instead of chosen blind. One absent
//! quantity, three consequences: no expected value, no hopping proof, and an
//! operator dial with no units on it.
//!
//! ### The refusal is the product
//!
//! Naming a coin the price file cannot quote is the whole of what this is for.
//! `design/mining.md` spent a document ranking yespower first and the survey
//! that answered it took one run; a pool that prints "bitzeny: last priced
//! 1493 days ago" at startup is the same finding arriving before the machine
//! is pointed at it rather than after.

use crate::json::Json;

/// One asset, as `prices.py` decided it.
pub struct Price {
    pub coin: String,
    pub usd: f64,
    pub volume_24h_usd: f64,
    pub age_hours: f64,
    pub usable: bool,
    /// Why it cannot be quoted. Empty when it can.
    pub why: String,
    /// Something a reader of a quotable figure still has to know. Separate
    /// from `why` because a caveat is not a refusal -- collapsing the two
    /// either loses caveats or loses coins, and the coin it would have lost is
    /// Verge, whose two sources sit a couple of percent apart.
    pub note: String,
}

pub struct Market {
    pub fetched_at: u64,
    pub coins: Vec<Price>,
}

impl Market {
    pub fn get(&self, asset: &str) -> Option<&Price> {
        self.coins.iter().find(|p| p.coin == asset)
    }

    /// How old the whole file is, in hours, against a unix time the caller
    /// supplies. Taken as an argument rather than read here so the claim below
    /// is not a test about today's date.
    pub fn age_hours(&self, now: u64) -> f64 {
        now.saturating_sub(self.fetched_at) as f64 / 3600.0
    }
}

/// A number that may legally be absent, and is `f64::NAN` when it is.
///
/// `as_i64` is the wrong reader here for the reason `stratum::decimal` exists:
/// it splits at the `.` and answers the integer part, so every price under a
/// dollar reads as zero and every one of these coins is under a dollar. This
/// takes the token text.
fn num(v: Option<&Json>) -> f64 {
    match v {
        Some(Json::Num(t)) => t.parse::<f64>().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

fn text(v: Option<&Json>) -> String {
    v.and_then(|x| x.as_str()).unwrap_or("").to_string()
}

pub fn parse(doc: &str) -> Result<Market, String> {
    let v = Json::parse(doc.trim()).ok_or("not JSON")?;
    let version = v.get("version").and_then(|x| x.as_i64()).unwrap_or(0);
    // Refused rather than guessed at, the same way `proto::VERSION` is. A file
    // whose shape changed is one whose fields mean something else, and reading
    // it anyway produces prices that parse.
    if version != 1 {
        return Err(format!("version {version} is not 1"));
    }
    let fetched_at = v.get("fetched_at").and_then(|x| x.as_i64()).unwrap_or(0);
    if fetched_at <= 0 {
        return Err(String::from("no fetched_at, so nothing can say how old it is"));
    }
    let list = match v.get("coins") {
        Some(Json::Arr(a)) => a,
        _ => return Err(String::from("no coins array")),
    };

    let mut coins = Vec::new();
    for c in list {
        let coin = text(c.get("coin"));
        if coin.is_empty() {
            return Err(String::from("a row with no coin name"));
        }
        let usable = c.get("usable").and_then(|x| x.as_bool()).unwrap_or(false);
        let why = text(c.get("why"));
        // The one invariant the pool depends on and cannot recover if it is
        // broken: a coin that is not usable says why, and a coin that is
        // usable does not pretend otherwise. `prices.py` asserts both from its
        // own side; this is the other side of the same claim, and two ends
        // asserting one property is the whole point of two parsers.
        if !usable && why.is_empty() {
            return Err(format!("{coin} is unusable and gives no reason"));
        }
        if usable && !why.is_empty() {
            return Err(format!("{coin} is usable and also refused: {why}"));
        }
        coins.push(Price {
            coin,
            usd: num(c.get("usd")),
            volume_24h_usd: num(c.get("volume_24h_usd")),
            age_hours: num(c.get("age_hours")),
            usable,
            why,
            note: text(c.get("note")),
        });
    }
    Ok(Market { fetched_at: fetched_at as u64, coins })
}

/// What this pool would print about one configured coin.
///
/// A `String` rather than a `println!` so the wording is a value a claim can
/// read. Every branch of it is a different fact and they were one line each
/// until the difference between "we did not ask" and "we asked and it is dead"
/// turned out to be the whole finding.
pub fn verdict(market: Option<&Market>, label: &str, asset: &str) -> String {
    let Some(m) = market else {
        return format!("{label}: no price file, so nothing here knows what it is worth");
    };
    let Some(p) = m.get(asset) else {
        return format!(
            "{label}: '{asset}' is not in the price file, so nothing here knows what it is worth"
        );
    };
    if !p.usable {
        return format!("{label}: cannot be quoted -- {}", p.why);
    }
    // Two formats, because the table spans eight orders of magnitude. Ten
    // decimal places on Bitcoin prints $78141.2908071005, which reads as more
    // precision than two exchange averages have; two decimal places on Verge
    // prints $0.00 and loses the coin entirely.
    let mut s = if p.usd >= 1.0 {
        format!("{label}: ${:.2} at ${:.0} of 24h volume", p.usd, p.volume_24h_usd)
    } else {
        format!("{label}: ${:.8} at ${:.0} of 24h volume", p.usd, p.volume_24h_usd)
    };
    if !p.note.is_empty() {
        s.push_str(" (");
        s.push_str(&p.note);
        s.push(')');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trimmed from a real `prices.py --write`, keeping one row of each verdict.
    const DOC: &str = r#"{
      "version": 1,
      "fetched_at": 1788995410,
      "coins": [
        {"coin":"verge","algo":"blake2s","usd":0.00264568,"volume_24h_usd":6058540.0,
         "age_hours":1.1,"usable":true,"why":"",
         "note":"the sources are 2.5% apart; the lower is quoted",
         "sources":["coingecko","coinpaprika"]},
        {"coin":"bitcoin","algo":"sha256d","usd":78141.29,"volume_24h_usd":30017587394.0,
         "age_hours":1.1,"usable":true,"why":"","note":"",
         "sources":["coingecko","coinpaprika"]},
        {"coin":"bitzeny","algo":"yespower","usd":0.00023968,"volume_24h_usd":0.0,
         "age_hours":35836.8,"usable":false,"why":"last priced 1493 days ago",
         "note":"","sources":["coingecko"]}
      ]
    }"#;

    #[test]
    fn the_file_python_writes_is_the_file_this_reads() {
        let m = parse(DOC).expect("parses");
        assert_eq!(m.coins.len(), 3);
        assert_eq!(m.fetched_at, 1788995410);
        assert_eq!(m.age_hours(1788995410 + 7200), 2.0);
    }

    /// **The reason this reader exists at all.** `as_i64` splits at the `.`
    /// and answers the integer part, which is zero for every coin in the table
    /// except Bitcoin and Monero -- and a price of zero is not an error, it is
    /// a coin the pool would rank last forever. Same trap `stratum::decimal`
    /// was written for, one file over.
    #[test]
    fn a_price_under_a_dollar_is_not_zero() {
        let m = parse(DOC).unwrap();
        let v = m.get("verge").unwrap();
        assert!(v.usd > 0.0026 && v.usd < 0.0027, "{}", v.usd);
        assert!(m.get("bitcoin").unwrap().usd > 78_000.0);
    }

    #[test]
    fn a_dead_coin_is_present_and_carries_its_reason() {
        let m = parse(DOC).unwrap();
        let z = m.get("bitzeny").unwrap();
        assert!(!z.usable);
        assert!(z.why.contains("1493 days"));
        // Present rather than absent, so "we did not ask" and "we asked and it
        // is dead" are different answers.
        assert!(verdict(Some(&m), "zeny", "bitzeny").contains("1493 days"));
        assert!(verdict(Some(&m), "zeny", "nosuchcoin").contains("not in the price file"));
        assert!(verdict(None, "zeny", "bitzeny").contains("no price file"));
    }

    #[test]
    fn a_caveat_survives_to_the_operator() {
        let m = parse(DOC).unwrap();
        let line = verdict(Some(&m), "xvg", "verge");
        assert!(line.contains("2.5% apart"), "{line}");
        assert!(!verdict(Some(&m), "btc", "bitcoin").contains('('));
    }

    /// The invariant both ends assert. A file that said a coin was unusable
    /// and gave no reason would leave the pool printing a refusal with nothing
    /// after the dash, which reads as a bug in the pool.
    #[test]
    fn a_refusal_without_a_reason_is_refused() {
        let bad = DOC.replace(r#""why":"last priced 1493 days ago""#, r#""why":""#);
        assert!(parse(&bad).is_err());
        let worse = DOC.replace(
            r#"{"coin":"verge","algo":"blake2s","usd":0.00264568,"volume_24h_usd":6058540.0,
         "age_hours":1.1,"usable":true,"why":"","#,
            r#"{"coin":"verge","algo":"blake2s","usd":0.00264568,"volume_24h_usd":6058540.0,
         "age_hours":1.1,"usable":true,"why":"dead","#,
        );
        assert!(parse(&worse).is_err());
    }

    #[test]
    fn a_file_of_another_version_is_refused_rather_than_read() {
        assert!(parse(&DOC.replace("\"version\": 1", "\"version\": 2")).is_err());
        assert!(parse(&DOC.replace("\"fetched_at\": 1788995410", "\"fetched_at\": 0")).is_err());
    }
}

/// What zpool pays per hash, per algorithm, from its `/api/status` document.
///
/// In mBTC per MH/s per day, which is one unit across every algorithm and all
/// `Pool::choose` needs. Keys are zpool's algorithm names, lowercased, which is
/// what a switched pool's coin labels must be.
///
/// **The lower of the forward estimate and twice what was actually paid.**
/// `estimate_current` is what switching needs -- it moves with difficulty and
/// price now, where `actual_last24h` lags a day -- but an estimate can spike on
/// one lucky block and never pay. Capping it at double the realised figure keeps
/// a spike from dragging every miner onto an algorithm that has not yet paid
/// anyone, while still letting a genuine rise through. Both fields have the
/// same meaning here: `estimate_current` is BTC and `actual_last24h` is mBTC, per
/// `mbtc_mh_factor` MH/s, which was read off the API and not assumed.
pub fn zpool_rates(doc: &str) -> Result<alloc::collections::BTreeMap<String, f64>, String> {
    let j = Json::parse(doc.trim()).ok_or("zpool status is not JSON")?;
    let Json::Obj(algos) = j else {
        return Err(String::from("zpool status is not an object"));
    };
    let num = |v: Option<&Json>| -> Option<f64> {
        match v? {
            Json::Num(t) | Json::Str(t) => t.parse::<f64>().ok(),
            _ => None,
        }
    };
    let mut out = alloc::collections::BTreeMap::new();
    for (name, a) in algos.iter() {
        let (Some(est), Some(act), Some(factor)) = (
            num(a.get("estimate_current")),
            num(a.get("actual_last24h")),
            num(a.get("mbtc_mh_factor")),
        ) else {
            continue;
        };
        if factor <= 0.0 || !est.is_finite() || !act.is_finite() {
            continue;
        }
        let rate = (est * 1000.0).min(2.0 * act) / factor;
        if rate > 0.0 {
            out.insert(name.to_ascii_lowercase(), rate);
        }
    }
    Ok(out)
}
