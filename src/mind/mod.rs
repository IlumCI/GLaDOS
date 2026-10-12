//! Part F of 1.4.2: the consciousness-indicator work, built and measured.
//!
//! Nothing here claims the machine is conscious, and nothing can: no
//! instrument verifies that. What is here is the measurements the field can
//! take -- `design/indicators.md` says which, with each protocol written down
//! before its first run -- and the code that takes them. A failed run is
//! published the same way a passed one is.

pub mod introspect;

use alloc::string::String;
use alloc::vec::Vec;
use introspect::{Condition, Tally};

/// `mind introspect calibrate`: the strengths the run will use, chosen
/// without a brain trial in sight. Answers the primary strength.
pub fn calibrate(out: &mut Vec<String>) -> Option<f32> {
    let r = crate::ai::with_engine(|e| {
        let dim = e.model.cfg.dim;
        let layer = introspect::layer_for(e.model.cfg.n_layers);
        let dir = introspect::calibration_direction(dim);
        let base = introspect::bits_per_byte(e, introspect::CALIBRATION_TEXT, None)?;
        let mut rows = Vec::new();
        for &a in introspect::ALPHA_GRID.iter() {
            let s = crate::ai::model::Steer { layer, dir: dir.clone(), alpha: a };
            rows.push((a, introspect::bits_per_byte(e, introspect::CALIBRATION_TEXT, Some(s))?));
        }
        Some((layer, base, rows))
    });
    let Some(Some((layer, base, rows))) = r else {
        out.push(String::from("  no model, or the calibration text did not score"));
        return None;
    };
    out.push(alloc::format!("  layer {}, unsteered {:.4} bits/byte, limit {:.4}", layer, base, base * (1.0 + introspect::CALIBRATION_SLACK)));
    let mut chosen = None;
    for (a, bpb) in rows.iter() {
        let ok = *bpb <= base * (1.0 + introspect::CALIBRATION_SLACK);
        out.push(alloc::format!("    alpha {:>8}  {:.4} bits/byte  {:+.1}%  {}", fmt3(*a), bpb, (bpb / base - 1.0) * 100.0, if ok { "within" } else { "over" }));
        if ok {
            chosen = Some(*a);
        }
    }
    match chosen {
        Some(a) => out.push(alloc::format!("  primary alpha {}, secondary {}", fmt3(a), fmt3(a / 2.0))),
        None => out.push(String::from("  no strength on the grid stays within the limit -- the hidden condition cannot run")),
    }
    chosen
}

fn fmt3(v: f32) -> String {
    let m = (v * 100_000.0 + 0.5) as u32;
    alloc::format!("{}.{:05}", m / 100_000, m % 100_000)
}

/// `mind introspect [n] [ms] [seed]`: the preregistered run.
pub fn introspect(n: usize, ms: u64, seed: u64, out: &mut Vec<String>) {
    use crate::kprintln;
    if crate::brain::observe().is_none() {
        out.push(String::from("  no brain is running ('brain on')"));
        return;
    }
    if crate::brain::asleep() {
        out.push(String::from("  the brain is asleep, and asleep its senses are gated -- 'brain wake' first"));
        return;
    }
    if crate::ai::initiative::status().4 {
        out.push(String::from("  'initiative off' first: the resident mind speaks, and being spoken to is a sense"));
        return;
    }
    let mut cal = Vec::new();
    let Some(primary) = calibrate(&mut cal) else {
        out.extend(cal);
        return;
    };
    for l in cal.iter() {
        kprintln!("{}", l);
    }
    let Some((dim, layer)) = crate::ai::with_engine(|e| (e.model.cfg.dim, introspect::layer_for(e.model.cfg.n_layers))) else {
        out.push(String::from("  no model"));
        return;
    };
    let conds = [Condition::Text, Condition::Cut, Condition::Hidden(primary), Condition::Hidden(primary / 2.0)];
    let mut tallies: Vec<Tally> = conds.iter().map(|_| Tally::default()).collect();
    let order = introspect::schedule(n, seed);
    let mut tsv = String::from("trial\tlabel\tcondition\tlp_nothing\tlp_food\tlp_pain\tlp_touch\tlp_heat\tresponse_norm\twords\n");
    let mut done = 0;
    for (i, &label) in order.iter().enumerate() {
        let Some(t) = introspect::run_trial(label, ms) else {
            out.push(alloc::format!("  trial {} could not be taken: the brain stopped or stalled", i + 1));
            break;
        };
        let norm = crate::ai::tensor::sqrtf(t.z.iter().map(|v| v * v).sum::<f32>());
        let dir = introspect::direction(&t.z, dim);
        let mut line = alloc::format!("  {:>3} {:8} |z| {:6.2}", i + 1, introspect::LABELS[label], norm);
        for (c, cond) in conds.iter().enumerate() {
            let (mood, steer) = match cond {
                Condition::Text => (Some(t.words), None),
                Condition::Cut => (None, None),
                Condition::Hidden(a) => (None, dir.clone().map(|d| crate::ai::model::Steer { layer, dir: d, alpha: *a })),
            };
            let p = introspect::prompt(mood);
            let Some(Some(lp)) = crate::ai::with_engine(|e| introspect::score(e, &p, steer)) else {
                out.push(String::from("  scoring failed: no model, or a non-finite logit"));
                return;
            };
            tallies[c].add(t.label, &lp);
            let pick = (0..5).max_by(|&a, &b| lp[a].partial_cmp(&lp[b]).unwrap_or(core::cmp::Ordering::Equal)).unwrap_or(0);
            line.push_str(&alloc::format!("  {}:{}", short(*cond), introspect::ANSWERS[pick]));
            tsv.push_str(&alloc::format!(
                "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\n",
                i + 1,
                introspect::LABELS[label],
                cond.name(),
                lp[0],
                lp[1],
                lp[2],
                lp[3],
                lp[4],
                norm,
                t.words
            ));
        }
        kprintln!("{}", line);
        done += 1;
        // Three seconds of brain time between trials, for the last one's
        // response and its neuromodulation to decay.
        // Waited for, not timed: a brain that cannot reach it is a run that
        // stops here, never an interval quietly shortened.
        let Some(now) = crate::brain::observe() else { break };
        if introspect::wait_brain(now.0 + 3_000_000).is_none() {
            out.push(alloc::format!("  the brain stalled after trial {}", i + 1));
            break;
        }
    }
    let path = alloc::format!("/ai/mind/introspect-{}.tsv", seed);
    let kept = crate::sysbox::write_blob(&path, tsv.into_bytes());
    out.push(alloc::format!("  {} of {} trial(s), every one scored under {} conditions{}", done, n, conds.len(), if kept { alloc::format!("; per trial in {}", path) } else { String::new() }));
    for (c, cond) in conds.iter().enumerate() {
        let t = &tallies[c];
        out.push(alloc::format!(
            "  {:16} detect {}/{} vs sham {}/{} (p {})   name {}/{} = {}% vs 25% (p {})   answers {:?}",
            cond.name(),
            t.hits,
            t.pokes,
            t.false_alarms,
            t.shams,
            fmt_p(t.p_detection()),
            t.named,
            t.pokes,
            if t.pokes > 0 { t.named * 100 / t.pokes } else { 0 },
            fmt_p(t.p_naming()),
            t.answers
        ));
    }
}

fn short(c: Condition) -> &'static str {
    match c {
        Condition::Text => "T",
        Condition::Cut => "C",
        Condition::Hidden(_) => "H",
    }
}

fn fmt_p(p: f64) -> String {
    if p < 1e-4 {
        return String::from("<1e-4");
    }
    let m = (p * 10_000.0 + 0.5) as u32;
    alloc::format!("{}.{:04}", m / 10_000, m % 10_000)
}

/// `diag introspect`.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut rows = introspect::checks();
    // With a model: a steered prompt prefilled agrees with the same prompt
    // decoded a token at a time, and steering moves it by far more than the
    // two paths differ. The prefill-and-forward gotcha, on the new field.
    if let Some((agree, moved)) = crate::ai::with_engine(|e| {
        let ids = e.tok.encode("The quiet library opens at nine.", true, false);
        let mut cfg = e.model.cfg;
        cfg.seq_len = ids.len() + 1;
        let dir = introspect::calibration_direction(cfg.dim);
        let steer = crate::ai::model::Steer { layer: introspect::layer_for(cfg.n_layers), dir, alpha: 0.25 };
        let run = |prefill: bool, s: Option<crate::ai::model::Steer>| {
            let mut st = crate::ai::model::State::new(&cfg);
            st.steer = s;
            if prefill {
                e.model.prefill(&mut st, &ids, 0);
            } else {
                for (i, &t) in ids.iter().enumerate() {
                    e.model.forward(&mut st, t, i);
                }
            }
            st.logits.clone()
        };
        let a = run(true, Some(steer.clone()));
        let b = run(false, Some(steer));
        let plain = run(false, None);
        let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        (d(&a, &b), d(&b, &plain))
    }) {
        crate::kprintln!("    steered prefill against decode {:.4}, steered against plain {:.4}", agree, moved);
        rows.push(("a steered prefill is the steered decode, and steering moves it", moved > 10.0 * agree && moved > 0.0));
    }
    let mut ok = true;
    for (what, good) in &rows {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
