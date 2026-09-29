//! A coin as a string: `label:algo:bits[:asset][@host:port,user[,pass]]`.
//!
//! **Moved out of `main.rs` so that more than one program can read it.** The
//! daemon parses coins from its command line; a Durable Object serving the same
//! pool parses them from its configuration. One parser for one format -- the
//! rule `record::parse_algo` already states about algorithm specs, one level up.

use crate::pool::{target_with_leading_zeros, Coin, Source};

pub fn parse_coin(spec: &str) -> Result<Coin, String> {
    // Split the upstream off first. `@` rather than another `:`, because a
    // `host:port` already contains one and a positional parser would have to
    // count colons from the right to tell them apart -- which breaks the first
    // time somebody omits the port.
    let (spec, source) = match spec.split_once('@') {
        None => (spec, Source::Local),
        Some((left, up)) => {
            let mut f = up.split(',');
            let hostport = f.next().unwrap_or("");
            let user = f.next().unwrap_or("");
            // Most pools ignore the password entirely and the convention is a
            // single `x`. Defaulted rather than required, since demanding a
            // field nobody reads is how a config gets copied wrong.
            let pass = f.next().unwrap_or("x");
            if user.is_empty() {
                return Err(format!("'{up}' has no worker name after the host"));
            }
            let (host, port) = match hostport.rsplit_once(':') {
                Some((h, p)) => match p.parse::<u16>() {
                    Ok(n) => (h.to_string(), n),
                    Err(_) => return Err(format!("'{p}' is not a port")),
                },
                None => return Err(format!("'{hostport}' needs a :port")),
            };
            (
                left,
                Source::Upstream {
                    host,
                    port,
                    user: String::from(user),
                    pass: String::from(pass),
                },
            )
        }
    };

    let mut it = spec.split(':');
    let label = it.next().unwrap_or("").trim();
    let algo_s = it.next().unwrap_or("");
    let bits_s = it.next().unwrap_or("");
    // A fourth field, optional, naming the traded asset. Positional and last
    // so every configuration written before it still parses -- and defaulting
    // to the label, so a pool whose coins are named as `prices.py` names them
    // needs nothing at all.
    let asset = it.next().unwrap_or(label).trim().to_string();
    if label.is_empty() || algo_s.is_empty() || bits_s.is_empty() {
        return Err(format!("'{spec}' is not label:algo:bits"));
    }
    let bits: u32 = bits_s
        .parse()
        .map_err(|_| format!("'{bits_s}' is not a number of bits"))?;
    // 256 leading zero bits is a target of zero, which nothing ever meets. A
    // pool configured that way accepts no share ever and looks exactly like a
    // pool with a broken hash, so it is refused at the argument instead.
    if bits >= 256 {
        return Err(format!("{bits} leading bits is a target nothing can meet"));
    }

    // One parser, in `record`, because a share record has to spell the
    // algorithm in exactly this form -- and two parsers for one format is what
    // `differ.rs` exists to catch elsewhere in this tree.
    let algo = crate::record::parse_algo(algo_s)?;

    Ok(Coin {
        label: String::from(label),
        asset,
        algo,
        share_bits: bits,
        share_target: target_with_leading_zeros(bits),
        // Filled from upstream's `set_difficulty` when there is an upstream,
        // and `None` otherwise -- rather than a plausible constant, because an
        // expected value derived from an invented difficulty is worse than one
        // that refuses to print.
        network_target: None,
        source,
        work: None,
        e2: 0,
    })
}
