//! Questions over a table, WikiSQL's kind: "what position does the player
//! who played for Butler CC play?", "what is the lowest attendance for week
//! 2?". One column, at most one aggregate, and conditions on other columns.
//!
//! Solved the way `ask` solves its questions, by choosing rather than
//! writing. A condition is a cell value the question spells out, found by
//! looking for every value of every column in the question's words -- a
//! closed set read off the table, so a condition on a value that is not
//! there cannot be made. The column asked about is a header the question
//! names, the aggregate a word it uses. The model is asked only what those
//! leave open: which column, when no header is named or a value appears in
//! several. The kernel writes the formula and LibreOffice computes it.

use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::{headless_macros, macro_batch, note, read_any, ready, BENCH_OUT};

pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// What `Bench.Cells` wrote: a row a line, a tab between cells, the header
/// row first.
pub fn parse(text: &str) -> Option<Table> {
    let mut lines = text.lines().map(|l| l.split('\t').map(|c| String::from(c.trim())).collect::<Vec<_>>());
    let headers = lines.next()?;
    let width = headers.iter().rposition(|h| !h.is_empty())? + 1;
    let headers: Vec<String> = headers.into_iter().take(width).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|mut r| {
            r.resize(width, String::new());
            r
        })
        .filter(|r| r.iter().any(|c| !c.is_empty()))
        .collect();
    Some(Table { headers, rows })
}

pub fn col_name(i: usize) -> String {
    let mut n = i + 1;
    let mut s = Vec::new();
    while n > 0 {
        s.insert(0, b'A' + ((n - 1) % 26) as u8);
        n = (n - 1) / 26;
    }
    String::from_utf8(s).unwrap_or_default()
}

fn number(v: &str) -> Option<f64> {
    let t: String = v.chars().filter(|c| !matches!(c, ',' | '$' | '%' | ' ')).collect();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

/// A number as a formula wants it: "$500,000" is 500000.
fn plain(v: &str) -> String {
    v.chars().filter(|c| !matches!(c, ',' | '$' | '%' | ' ')).collect()
}

fn numeric(t: &Table, c: usize) -> bool {
    let vals: Vec<&String> = t.rows.iter().map(|r| &r[c]).filter(|v| !v.is_empty()).collect();
    !vals.is_empty() && vals.iter().filter(|v| number(v).is_some()).count() * 4 >= vals.len() * 3
}

/// Lowercase, with everything that is not a letter or digit a single space,
/// and a space at each end so a word match is a substring match.
fn norm(s: &str) -> String {
    let mut out = String::from(" ");
    let mut gap = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            for l in c.to_lowercase() {
                out.push(l);
            }
            gap = false;
        } else if !gap {
            out.push(' ');
            gap = true;
        }
    }
    if !gap {
        out.push(' ');
    }
    out
}

/// The question lowercased with its decimal points kept, for telling "4.9"
/// from "4 9" after `norm` has made them alike.
fn norm_keep_points(s: &str) -> String {
    s.to_lowercase()
}

fn with_decimals(raw: &str, n: &str) -> String {
    if raw.contains(n) {
        String::from(n)
    } else {
        String::from(n.split('.').next().unwrap_or(n))
    }
}

const STOP: &[&str] = &["the", "of", "a", "an", "in", "for", "to", "and", "no", "is", "at", "on", "by"];

/// How strongly the question names a header: its whole phrase counts most,
/// then each of its words that is not a stop word.
fn names(q: &str, header: &str) -> usize {
    let h = norm(header);
    if h.trim().is_empty() {
        return 0;
    }
    if q.contains(&h) {
        return 100 + h.len();
    }
    h.split(' ')
        .filter(|w| w.len() > 1 && !STOP.contains(w))
        .filter(|w| q.contains(&format!(" {} ", w)) || q.contains(&format!(" {}s ", w)))
        .count()
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Agg {
    None,
    Max,
    Min,
    Count,
    Sum,
    Avg,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Cond {
    pub col: usize,
    /// '=', '>' or '<'.
    pub op: char,
    pub value: String,
}

/// What was decided, and by what.
pub struct Plan {
    pub sel: usize,
    pub agg: Agg,
    pub conds: Vec<Cond>,
    pub said: Vec<&'static str>,
}

/// The conditions a question spells out: every value of every column found
/// among its words, longest first so "new york city" is not "york", and
/// no stretch of the question used twice. A value found in several columns
/// is a decision, answered by a header the question names, else by `pick`.
fn equalities(q: &str, t: &Table, pick: &mut dyn FnMut(&[usize]) -> Option<usize>, said: &mut Vec<&'static str>) -> Vec<Cond> {
    // (normalised value, original, columns holding it)
    let mut found: Vec<(String, String, Vec<usize>)> = Vec::new();
    for c in 0..t.headers.len() {
        for r in &t.rows {
            let v = &r[c];
            let n = norm(v);
            let core = n.trim();
            if core.is_empty() || (core.len() < 2 && !core.chars().all(|ch| ch.is_ascii_digit())) {
                continue;
            }
            if !q.contains(&n) {
                continue;
            }
            // Some tables repeat their header row among the data; a header's
            // own text is a column name in the question, never a value.
            if t.headers.iter().any(|h| norm(h) == n) {
                continue;
            }
            match found.iter_mut().find(|(k, _, _)| *k == n) {
                Some((_, _, cols)) => {
                    if !cols.contains(&c) {
                        cols.push(c)
                    }
                }
                None => found.push((n.clone(), v.clone(), alloc::vec![c])),
            }
        }
    }
    found.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    let mut used: Vec<(usize, usize)> = Vec::new();
    let mut out: Vec<Cond> = Vec::new();
    for (n, v, cols) in found {
        let Some(at) = q.find(&n) else { continue };
        let span = (at, at + n.len());
        // Spans share their boundary spaces, so overlap is strict.
        if used.iter().any(|&(a, b)| span.0 + 1 < b && a + 1 < span.1) {
            continue;
        }
        let col = if cols.len() == 1 {
            Some(cols[0])
        } else {
            // A header beside the value names its column: "week 2".
            let best = cols.iter().copied().max_by_key(|&c| names(q, &t.headers[c]));
            match best.filter(|&c| names(q, &t.headers[c]) > 0) {
                Some(c) => {
                    said.push("a shared value's column by its header");
                    Some(c)
                }
                None => {
                    said.push("a shared value's column by the model");
                    pick(&cols).map(|i| cols[i])
                }
            }
        };
        if let Some(col) = col {
            if out.iter().any(|c| c.col == col) {
                continue;
            }
            used.push(span);
            out.push(Cond { col, op: '=', value: v });
        }
    }
    out
}

/// Every "more than 5", "fewer than 20,000", "after 1990" in a question:
/// the operator, the number, and where in the question the phrase sits.
fn inequalities(q: &str) -> Vec<(char, String, usize, usize)> {
    const GT: &[&str] = &["more than", "greater than", "larger than", "higher than", "bigger than", "over", "above", "after", "later than", "exceeds"];
    const LT: &[&str] = &["less than", "fewer than", "smaller than", "lower than", "under", "below", "before", "earlier than"];
    let mut out = Vec::new();
    for (op, words) in [('>', GT), ('<', LT)] {
        for w in words {
            let key = format!(" {} ", w);
            let mut from = 0;
            while let Some(j) = q[from..].find(&key) {
                let i = from + j;
                let rest = &q[i + key.len()..];
                let tok = rest.split(' ').next().unwrap_or("");
                // "4 9" is how `norm` writes 4.9.
                let mut num = String::from(tok);
                let after = &rest[tok.len()..];
                if let Some(frac) = after.strip_prefix(' ').and_then(|r| r.split(' ').next()) {
                    if !tok.is_empty() && !frac.is_empty() && frac.chars().all(|c| c.is_ascii_digit()) && q.contains(&format!(" {} {} ", tok, frac)) {
                        // Only taken as a decimal when the original had one;
                        // see `with_decimals`.
                        num = format!("{}.{}", tok, frac);
                    }
                }
                if !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit()) {
                    if !out.iter().any(|x: &(char, String, usize, usize)| x.2 == i) {
                        out.push((op, num, i, i + key.len() + tok.len()));
                    }
                }
                from = i + key.len();
            }
        }
    }
    out.sort_by_key(|x| x.2);
    out
}

#[cfg(any())]
fn inequality(q: &str) -> Option<(char, String)> {
    inequalities(q).into_iter().next().map(|(op, n, _, _)| (op, n))
}

fn aggregate(q: &str, t: &Table, sel: usize) -> Agg {
    // Header phrases are not aggregate words: "total votes" names a column.
    let mut q = String::from(q);
    for h in &t.headers {
        let h = norm(h);
        if h.trim().len() > 2 {
            q = q.replace(&h, " ");
        }
    }
    let has = |ws: &[&str]| ws.iter().any(|w| q.contains(&format!(" {} ", w)));
    // WikiSQL's annotators mean a count by "total number of" and "how many
    // in total", whatever the column holds.
    if has(&["total number of", "how many total", "in total"]) {
        return Agg::Count;
    }
    if has(&["average", "mean"]) {
        Agg::Avg
    } else if has(&["total", "sum", "combined"]) {
        Agg::Sum
    } else if has(&["highest", "most", "maximum", "largest", "biggest", "greatest", "top", "max", "latest", "last"]) {
        Agg::Max
    } else if has(&["lowest", "least", "minimum", "smallest", "fewest", "min", "earliest", "first"]) {
        Agg::Min
    } else if has(&["how many", "number of", "count"]) {
        // "How many points did ..." asks for a number the table holds;
        // "how many schools did ..." asks for a count of rows.
        if numeric(t, sel) {
            Agg::None
        } else {
            Agg::Count
        }
    } else {
        Agg::None
    }
}

/// Decide a question against a table. `pick(prompt, options)` is the model.
pub fn plan(question: &str, t: &Table, pick: &mut dyn FnMut(&str, &[&str]) -> Option<usize>) -> Option<Plan> {
    let q = norm(question);
    let mut said = Vec::new();
    let heads: Vec<&str> = t.headers.iter().map(|h| h.as_str()).collect();
    let mut conds = {
        let mut ask_col = |cols: &[usize]| {
            let opts: Vec<&str> = cols.iter().map(|&c| heads[c]).collect();
            pick(&format!("Question: {} Which column holds the value it names?", question), &opts)
        };
        equalities(&q, t, &mut ask_col, &mut said)
    };
    let raw = norm_keep_points(question);
    for (op, n, at, end) in inequalities(&q) {
        let n = with_decimals(&raw, &n);
        let nums: Vec<usize> = (0..heads.len()).filter(|&c| numeric(t, c) && !conds.iter().any(|x| x.col == c && x.op != '=')).collect();
        // The column is named beside the phrase: "more than 4 tries",
        // "Total is greater than 283".
        let near = |c: usize| {
            let lo = at.saturating_sub(40);
            let hi = (end + 40).min(q.len());
            let lo = (0..=lo).rev().find(|&k| q.is_char_boundary(k)).unwrap_or(0);
            let hi = (hi..=q.len()).find(|&k| q.is_char_boundary(k)).unwrap_or(q.len());
            names(&q[lo..hi], heads[c])
        };
        let c = nums.iter().copied().filter(|&c| near(c) > 0).max_by_key(|&c| near(c));
        let c = match c {
            Some(c) => Some(c),
            None if !nums.is_empty() => {
                let opts: Vec<&str> = nums.iter().map(|&c| heads[c]).collect();
                said.push("a comparison's column by the model");
                pick(&format!("Question: {} Which column is compared with {}?", question, n), &opts).map(|i| nums[i])
            }
            None => None,
        };
        if let Some(col) = c {
            // The number itself was taken for an equality; it is not one.
            let whole = n.split('.').next().unwrap_or(&n).to_string();
            conds.retain(|x| {
                let v = norm(&x.value);
                let v = v.trim();
                !(x.op == '=' && (v == n || v == whole || v == n.replace('.', " ")))
            });
            conds.retain(|x| x.col != col);
            conds.push(Cond { col, op, value: n });
        }
    }
    let free: Vec<usize> = (0..heads.len()).filter(|c| !conds.iter().any(|x| x.col == *c)).collect();
    if free.is_empty() {
        return None;
    }
    let named = free.iter().copied().filter(|&c| names(&q, heads[c]) > 0).max_by(|&a, &b| {
        names(&q, heads[a]).cmp(&names(&q, heads[b])).then(b.cmp(&a))
    });
    let sel = match named {
        Some(c) => {
            said.push("column by its header");
            c
        }
        None => {
            said.push("column by the model");
            let opts: Vec<&str> = free.iter().map(|&c| heads[c]).collect();
            free[pick(&format!("Question: {} Which column holds the answer?", question), &opts)?]
        }
    };
    let agg = aggregate(&q, t, sel);
    Some(Plan { sel, agg, conds, said })
}

fn criterion(t: &Table, c: &Cond) -> String {
    let numeric_col = numeric(t, c.col);
    match c.op {
        '=' if numeric_col && number(&c.value).is_some() => plain(&c.value),
        '=' => {
            // A criterion string is a pattern in Calc: `*`, `?` and `~` are
            // escaped so a value holding one matches only itself.
            let v = c.value.replace('~', "~~").replace('*', "~*").replace('?', "~?").replace('"', "\"\"");
            format!("\"{}\"", v)
        }
        op => format!("\"{}{}\"", op, c.value),
    }
}

/// The formula a plan asks for, over data rows 2..=n.
pub fn formula(t: &Table, p: &Plan) -> String {
    let n = t.rows.len() + 1;
    let rng = |c: usize| format!("{}2:{}{}", col_name(c), col_name(c), n);
    let s = rng(p.sel);
    let crit: Vec<String> = p.conds.iter().map(|c| format!("{},{}", rng(c.col), criterion(t, c))).collect();
    let ifs = |f: &str| format!("={}({},{})", f, s, crit.join(","));
    match (p.agg, p.conds.is_empty()) {
        (Agg::None, true) => format!("=INDEX({},1)", s),
        (Agg::None, false) => {
            let tests: Vec<String> = p
                .conds
                .iter()
                .map(|c| {
                    let quoted = format!("\"{}\"", c.value.replace('"', "\"\""));
                    match c.op {
                        // A number may be stored as a number or as its text:
                        // WikiSQL types most columns as text, so "92.0" can
                        // be either, and testing one form misses the other.
                        '=' if number(&c.value).is_some() => {
                            format!("(({r}={n})+({r}={q})>0)", r = rng(c.col), n = plain(&c.value), q = quoted)
                        }
                        '=' => format!("({}={})", rng(c.col), quoted),
                        op => format!("({}{}{})", rng(c.col), op, c.value),
                    }
                })
                .collect();
            format!("{{=INDEX({},MATCH(1,{},0))}}", s, tests.join("*"))
        }
        (Agg::Count, true) => format!("=COUNTA({})", s),
        (Agg::Count, false) => format!("=COUNTIFS({})", crit.join(",")),
        (Agg::Sum, true) => format!("=SUM({})", s),
        (Agg::Sum, false) => ifs("SUMIFS"),
        (Agg::Avg, true) => format!("=AVERAGE({})", s),
        (Agg::Avg, false) => ifs("AVERAGEIFS"),
        (Agg::Max, true) => format!("=MAX({})", s),
        (Agg::Max, false) => ifs("MAXIFS"),
        (Agg::Min, true) => format!("=MIN({})", s),
        (Agg::Min, false) => ifs("MINIFS"),
    }
}

/// Whether a computed answer is one of the gold ones: numbers within a part
/// in a million, text the same once lowercased and trimmed.
pub fn agrees(got: &str, gold: &[&str]) -> bool {
    // Dashes of every width are one dash: the gold comes from SQLite and the
    // answer from Calc, and they disagree about U+2013 where nothing else does.
    let clean = |s: &str| -> String {
        let s: String = s
            .trim()
            .to_lowercase()
            .chars()
            .map(|c| if matches!(c, '\u{2010}'..='\u{2015}' | '\u{2212}') { '-' } else { c })
            .collect();
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    let g = clean(got);
    gold.iter().any(|w| {
        let w = clean(w);
        match (number(&g), number(&w)) {
            (Some(a), Some(b)) => {
                let tol = (b.abs() * 1e-6).max(1e-6);
                (a - b).abs() <= tol
            }
            _ => g == w,
        }
    })
}

/// `office wikisql [dir] [from] [count]`.
pub fn wikisql(arg: &str) -> Result<(), String> {
    ready()?;
    headless_macros()?;
    let mut words = arg.split_whitespace().peekable();
    let dir = match words.peek() {
        Some(w) if w.starts_with('/') => words.next().unwrap(),
        _ => "/mnt/wikisql/dev",
    };
    let from: usize = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
    let count: usize = words.next().and_then(|w| w.parse().ok()).unwrap_or(usize::MAX);
    let manifest = read_any(&format!("{}/tasks.tsv", dir))?;
    let tasks: Vec<Vec<&str>> = manifest
        .lines()
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|t| t.len() >= 4)
        .skip(from)
        .take(count)
        .collect();
    if tasks.is_empty() {
        return Err(format!("no tasks in {}/tasks.tsv", dir));
    }
    note(format!("{} WikiSQL question(s): LibreOffice reads the tables", tasks.len()));
    let cells: Vec<Vec<String>> = tasks
        .iter()
        .enumerate()
        .map(|(i, t)| alloc::vec![format!("{}/w{}.txt", BENCH_OUT, i), String::new(), format!("{}/{}", dir, t[1])])
        .collect();
    macro_batch("Cells", &cells);

    let mut plans: Vec<Option<(String, Vec<&'static str>)>> = Vec::new();
    let mut spare: Vec<String> = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let table = read_any(&cells[i][0]).ok().and_then(|s| parse(&s));
        let Some(table) = table else {
            plans.push(None);
            spare.push(String::new());
            continue;
        };
        let mut pick = |prompt: &str, opts: &[&str]| crate::ai::author::choose(prompt, opts);
        let p = plan(t[2], &table, &mut pick);
        spare.push(format!("{}1", col_name(table.headers.len() + 1)));
        plans.push(p.map(|p| (formula(&table, &p), p.said)));
    }

    note(String::from("LibreOffice computes the answers"));
    let probes: Vec<Vec<String>> = tasks
        .iter()
        .enumerate()
        .filter(|(i, _)| plans[*i].is_some())
        .map(|(i, t)| {
            alloc::vec![
                format!("{}/a{}.txt", BENCH_OUT, i),
                String::new(),
                spare[i].clone(),
                format!("{}/{}", dir, t[1]),
                plans[i].as_ref().unwrap().0.clone()
            ]
        })
        .collect();
    macro_batch("Probe", &probes);

    let (mut right, mut by_model, mut right_model) = (0usize, 0usize, 0usize);
    let mut log = String::new();
    for (i, t) in tasks.iter().enumerate() {
        let gold: Vec<&str> = t[3].split('\u{1f}').collect();
        let (f, said) = match &plans[i] {
            Some((f, s)) => (f.as_str(), s.clone()),
            None => ("", Vec::new()),
        };
        let line = read_any(&format!("{}/a{}.txt", BENCH_OUT, i)).unwrap_or_default();
        let got: String = line.lines().next().unwrap_or("").strip_prefix("OK\t").unwrap_or("").trim().into();
        let ok = !f.is_empty() && agrees(&got, &gold);
        let model = said.iter().any(|s| s.ends_with("the model"));
        right += ok as usize;
        by_model += model as usize;
        right_model += (ok && model) as usize;
        note(format!(
            "{} {:<5} {}  -> {}  (gold {})  {}",
            t[0],
            if ok { "RIGHT" } else { "wrong" },
            t[2],
            if got.is_empty() { "-" } else { &got },
            gold.join(" | "),
            f
        ));
        log.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\n", t[0], ok as u8, got, t[3].replace('\u{1f}', " | "), f, said.join("; ")));
    }
    let _ = crate::linux::ns::write_blob(&format!("{}/wikisql.tsv", BENCH_OUT), log.into_bytes());
    note(format!(
        "WikiSQL: {} of {} answered right by execution ({} needed the model, {} of those right)",
        right,
        tasks.len(),
        by_model,
        right_model
    ));
    Ok(())
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let t = parse("Player\tNo.\tPosition\tSchool\nAntonio Lang\t21\tGuard-Forward\tDuke\nMartin Lewis\t32\tGuard\tButler CC (KS)\nBrad Lohaus\t33\tForward-Center\tIowa\n").unwrap();
    let mut never = |_: &str, _: &[&str]| -> Option<usize> { None };
    let p = plan("What position does the player who played for butler cc (ks) play?", &t, &mut never);
    let mut out = Vec::new();
    out.push(("a table's first row is its headers", t.headers.len() == 4 && t.rows.len() == 3));
    out.push(("a value the question spells out is its condition", p.as_ref().is_some_and(|p| p.conds == [Cond { col: 3, op: '=', value: String::from("Butler CC (KS)") }])));
    out.push(("and the header it names is the column asked about", p.as_ref().is_some_and(|p| p.sel == 2 && p.agg == Agg::None)));
    out.push((
        "a lookup is an INDEX over a MATCH of every condition",
        p.as_ref().is_some_and(|p| formula(&t, p) == "{=INDEX(C2:C4,MATCH(1,(D2:D4=\"Butler CC (KS)\"),0))}"),
    ));
    let c = plan("How many players played at Duke?", &t, &mut never);
    out.push(("'how many' of a text column counts rows", c.as_ref().is_some_and(|p| p.agg == Agg::Count && formula(&t, p) == "=COUNTIFS(D2:D4,\"Duke\")")));
    let m = plan("What is the highest No. for a guard?", &t, &mut never);
    out.push(("'highest' is a MAX with its conditions", m.as_ref().is_some_and(|p| p.agg == Agg::Max && formula(&t, p).starts_with("=MAXIFS(B2:B4,"))));
    out.push(("'more than 25' is a condition on a number", inequalities(&norm("players with No. more than 25")).first().map(|x| (x.0, x.1.clone())) == Some(('>', String::from("25")))));
    out.push(("and every comparison in a question is found", inequalities(&norm("more than 4 tries, more than 32 points and fewer than 2 goals")).len() == 3));
    out.push(("an en dash agrees with a hyphen", agrees("1-4 1", &["1\u{2013}4 1"])));
    out.push(("an answer agrees with its gold as a number or as text", agrees("16", &["16.0"]) && agrees("Guard", &["guard"]) && !agrees("Duke", &["iowa"])));
    out.push(("a column past Z is AA", col_name(0) == "A" && col_name(25) == "Z" && col_name(26) == "AA"));
    out
}
