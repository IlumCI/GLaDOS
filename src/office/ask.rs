//! `office ask <workbook> <request>`: a question about a sheet, answered by
//! choosing rather than writing.
//!
//! The 0.6B model is bad at writing a formula: on SpreadsheetBench's first
//! forty tasks its single answer solved none, and most answers were not valid
//! Excel at all. It is good at picking from a list, which is how it routes to
//! an applet. So here it does only that: it picks an operation out of a closed
//! table, then the column it applies to, and the grouping column if there is
//! one, every choice decoded under a grammar of what exists. The columns are
//! the sheet's own header row, so a column that is not there cannot be
//! named. The kernel writes the formula from those choices, and LibreOffice
//! computes it -- the arithmetic is the spreadsheet's, as in `office calc`.
//!
//! A wrong pick is still possible. A made-up column, an invalid formula and a
//! wrong sum are not.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::{headless_macros, macro_batch, note, read_any, ready, BENCH_OUT};

/// What a request can ask of a column. The names are what the model picks
/// between, so they are short ordinary words: a name carries probability mass
/// of its own, which `repair.rs` measured, and an unusual one is a name the
/// model avoids.
struct Op {
    name: &'static str,
    /// The spreadsheet function over the whole column.
    whole: &'static str,
    /// Per group, as a `*IF`; `None` for an operation with no grouped form.
    grouped: Option<&'static str>,
    /// Whether it needs a column of numbers at all: a count does not.
    needs_value: bool,
    by: bool,
}

const OPS: &[Op] = &[
    Op { name: "total", whole: "SUM", grouped: None, needs_value: true, by: false },
    Op { name: "average", whole: "AVERAGE", grouped: None, needs_value: true, by: false },
    Op { name: "count", whole: "COUNTA", grouped: None, needs_value: false, by: false },
    Op { name: "largest", whole: "MAX", grouped: None, needs_value: true, by: false },
    Op { name: "smallest", whole: "MIN", grouped: None, needs_value: true, by: false },
    Op { name: "total per group", whole: "SUM", grouped: Some("SUMIF"), needs_value: true, by: true },
    Op { name: "average per group", whole: "AVERAGE", grouped: Some("AVERAGEIF"), needs_value: true, by: true },
    Op { name: "count per group", whole: "COUNTA", grouped: Some("COUNTIF"), needs_value: false, by: true },
];

/// A sheet as `Bench.View` rendered it: the header row, how far the data
/// goes, and the cells seen.
struct Sheet {
    /// (column letters, header text), left to right.
    headers: Vec<(String, String)>,
    last_row: usize,
    last_col: String,
    /// (address, value) for every cell the view carried.
    cells: Vec<(String, String)>,
}

fn split_addr(a: &str) -> Option<(&str, usize)> {
    let i = a.find(|c: char| c.is_ascii_digit())?;
    let (col, row) = a.split_at(i);
    (!col.is_empty() && col.bytes().all(|b| b.is_ascii_uppercase())).then_some(())?;
    Some((col, row.parse().ok()?))
}

/// Parse a view: a first line `Sheet <name>, used area A1:G41`, then one line
/// per row of `A1=value | B1=value ...`, a formula's text in braces after it.
fn parse(view: &str) -> Option<Sheet> {
    let mut lines = view.lines();
    let head = lines.next()?;
    let area = head.rsplit("used area ").next()?;
    let end = area.split(':').nth(1)?.trim();
    let (last_col, last_row) = split_addr(end)?;
    let mut cells = Vec::new();
    for l in lines {
        for part in l.split(" | ") {
            if let Some((addr, val)) = part.split_once('=') {
                if split_addr(addr).is_some() {
                    // A formula's own text is kept out of the value.
                    let v = val.split(" {").next().unwrap_or(val);
                    cells.push((String::from(addr), String::from(v.trim())));
                }
            }
        }
    }
    let headers = cells
        .iter()
        .filter(|(a, _)| split_addr(a).is_some_and(|(_, r)| r == 1))
        .filter(|(_, v)| !v.is_empty())
        .map(|(a, v)| (String::from(split_addr(a).unwrap().0), v.clone()))
        .collect::<Vec<_>>();
    (!headers.is_empty()).then_some(Sheet { headers, last_row, last_col: String::from(last_col), cells })
}

/// The column after `col`, for a cell nothing in the sheet is using.
fn next_col(col: &str) -> String {
    let mut b: Vec<u8> = col.bytes().collect();
    let mut i = b.len();
    loop {
        if i == 0 {
            b.insert(0, b'A');
            break;
        }
        i -= 1;
        if b[i] == b'Z' {
            b[i] = b'A';
        } else {
            b[i] += 1;
            break;
        }
    }
    String::from_utf8(b).unwrap_or_default()
}

/// The distinct values of a column below its header, in order of appearance.
fn groups(sheet: &Sheet, col: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (a, v) in &sheet.cells {
        if let Some((c, r)) = split_addr(a) {
            if c == col && r > 1 && !v.is_empty() && !out.contains(v) {
                out.push(v.clone());
            }
        }
    }
    out
}

/// The formulas that answer `op` over `value` (and `by`), with their labels.
fn formulas(sheet: &Sheet, op: &Op, value: &str, by: Option<&str>) -> Vec<(String, String)> {
    let n = sheet.last_row;
    let range = |c: &str| format!("{}2:{}{}", c, c, n);
    match (op.grouped, by) {
        (Some(f), Some(g)) => groups(sheet, g)
            .into_iter()
            .map(|label| {
                let quoted = label.replace('"', "\"\"");
                let formula = if f == "COUNTIF" {
                    format!("=COUNTIF({},\"{}\")", range(g), quoted)
                } else {
                    format!("={}({},\"{}\",{})", f, range(g), quoted, range(value))
                };
                (label, formula)
            })
            .collect(),
        _ => alloc::vec![(String::new(), format!("={}({})", op.whole, range(value)))],
    }
}

/// Whether a column's data cells are numbers: most of them parse once a
/// currency sign, thousands commas and a percent are taken off. Offered as
/// the column to add up only if so, and as the groups only if not -- "the
/// total of Date" and "grouped by Revenue" were both the model's picks on the
/// first run, and neither can be asked for now.
fn numeric(sheet: &Sheet, col: &str) -> bool {
    let vals: Vec<&str> = sheet
        .cells
        .iter()
        .filter(|(a, v)| split_addr(a).is_some_and(|(c, r)| c == col && r > 1) && !v.is_empty())
        .map(|(_, v)| v.as_str())
        .collect();
    let num = vals
        .iter()
        .filter(|v| {
            let t: String = v.chars().filter(|c| !matches!(c, '$' | ',' | '%' | ' ' | '\u{a3}' | '\u{20ac}')).collect();
            !t.is_empty() && t.parse::<f64>().is_ok()
        })
        .count();
    !vals.is_empty() && num * 4 >= vals.len() * 3
}

/// Whether a request wants one answer per group. Read off the words rather
/// than chosen: on the first run "what is our total revenue" was answered per
/// group, the grouped operations' names beginning with the plain ones'.
fn wants_groups(request: &str) -> bool {
    request
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| matches!(w.to_ascii_lowercase().as_str(), "each" | "per" | "by" | "every" | "breakdown"))
}

/// A computed value for reading: a long fraction to two places, as a person
/// would quote an average, and anything else as LibreOffice wrote it.
fn shown(v: &str) -> String {
    match (v.find('.'), v.parse::<f64>()) {
        (Some(d), Ok(x)) if v.len() - d > 3 => format!("{:.2}", x),
        _ => String::from(v),
    }
}

fn words(t: &str) -> Vec<String> {
    t.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| {
            let w = w.to_lowercase();
            // "regions" names Region, "sales" does not become "sale" usefully
            // but costs nothing to try.
            String::from(w.strip_suffix('s').filter(|r| r.len() > 2).unwrap_or(&w))
        })
        .collect()
}

/// The candidates the request names, by any word of their header, in the
/// order the request mentions them. Asked before the model: a header the
/// request spells out is not a decision, and a 0.6B asked "which column?"
/// picked by its own prior -- "revenue by region" came back as Units by
/// Product on the second run, with both named in the question.
fn named(request: &str, headers: &[&str], among: &[usize]) -> Vec<usize> {
    let req = words(request);
    let mut hits: Vec<(usize, usize)> = among
        .iter()
        .filter_map(|&i| {
            let hw = words(headers[i]);
            req.iter().position(|w| hw.contains(w)).map(|p| (p, i))
        })
        .collect();
    hits.sort();
    hits.into_iter().map(|(_, i)| i).collect()
}

/// The operation a request's own words ask for, by first match; `None` leaves
/// it to the model.
fn op_named(request: &str) -> Option<&'static str> {
    let t = request.to_lowercase();
    let has = |k: &[&str]| k.iter().any(|w| t.contains(w));
    if has(&["average", "mean", "typical"]) {
        Some("average")
    } else if has(&["biggest", "largest", "highest", "most expensive", "maximum", "max ", "top "]) {
        Some("largest")
    } else if has(&["smallest", "lowest", "least", "cheapest", "minimum", "min "]) {
        Some("smallest")
    } else if has(&["how many", "count", "number of"]) {
        Some("count")
    } else if has(&["total", "sum", "how much", "altogether", "overall", "bring in", "brought in"]) {
        Some("total")
    } else {
        None
    }
}

pub fn ask(path: &str, request: &str) -> Result<(), String> {
    ready()?;
    headless_macros()?;
    let out = format!("{}/ask-view.txt", BENCH_OUT);
    macro_batch("View", &[alloc::vec![out.clone(), String::new(), String::from(path)]]);
    let view = read_any(&out).map_err(|_| format!("LibreOffice could not read {}", path))?;
    let sheet = parse(&view).ok_or_else(|| format!("no header row found in {}", path))?;
    let names: Vec<&str> = sheet.headers.iter().map(|(_, h)| h.as_str()).collect();
    note(format!("{} column(s): {}", names.len(), names.join(", ")));
    let numbers: Vec<usize> = (0..sheet.headers.len()).filter(|&i| numeric(&sheet, &sheet.headers[i].0)).collect();
    let labels: Vec<usize> = (0..sheet.headers.len()).filter(|i| !numbers.contains(i)).collect();

    let lead = format!("A spreadsheet has the columns {}. Request: {}.", names.join(", "), request);
    let grouped = wants_groups(request) && !labels.is_empty();
    let ops: Vec<&Op> = OPS.iter().filter(|o| o.by == grouped).collect();
    let op_names: Vec<&str> = ops.iter().map(|o| o.name.split(' ').next().unwrap_or(o.name)).collect();
    let pick = |q: &str, from: &[&str]| crate::ai::author::choose(&format!("{} {}", lead, q), from);
    let from = |set: &[usize]| set.iter().map(|&i| names[i]).collect::<Vec<_>>();
    let mut said: Vec<&str> = Vec::new();
    let mut op_i = match op_named(request).and_then(|w| ops.iter().position(|o| o.name.starts_with(w))) {
        Some(i) => {
            said.push("operation by its word");
            i
        }
        None => {
            said.push("operation by the model");
            pick("Which operation answers it?", &op_names).ok_or("the model chose no operation")?
        }
    };
    // "How many units" is a total of a column of numbers the request names,
    // where "how many sales" is a count of rows.
    if ops[op_i].name.starts_with("count") && !named(request, &names, &numbers).is_empty() {
        if let Some(t) = ops.iter().position(|o| o.name.starts_with("total")) {
            op_i = t;
        }
    }
    let op = ops[op_i];
    let by = if op.by {
        let i = match named(request, &names, &labels).first() {
            Some(&i) => {
                said.push("groups by name");
                i
            }
            None => {
                said.push("groups by the model");
                labels[pick("Which column are the groups?", &from(&labels)).ok_or("the model chose no grouping column")?]
            }
        };
        Some(sheet.headers[i].0.clone())
    } else {
        None
    };
    let value = if op.needs_value {
        if numbers.is_empty() {
            return Err("no column of numbers to work on".into());
        }
        let i = match named(request, &names, &numbers).first() {
            Some(&i) => {
                said.push("column by name");
                i
            }
            None => {
                said.push("column by the model");
                let q = format!("Which column is the {} of?", op.name.split(' ').next().unwrap_or(op.name));
                numbers[pick(&q, &from(&numbers)).ok_or("the model chose no column")?]
            }
        };
        sheet.headers[i].0.clone()
    } else {
        // A count over the grouping column, or the first column.
        by.clone().unwrap_or_else(|| sheet.headers[0].0.clone())
    };
    let header_of = |c: &str| sheet.headers.iter().find(|(l, _)| l == c).map(|(_, h)| h.clone()).unwrap_or_else(|| String::from(c));
    note(format!(
        "chose {}{}{}  ({})",
        op.name,
        if op.needs_value { format!(" of {}", header_of(&value)) } else { String::new() },
        by.as_deref().map(|g| format!(", by {}", header_of(g))).unwrap_or_default(),
        said.join(", ")
    ));

    let asked = formulas(&sheet, op, &value, by.as_deref());
    if asked.is_empty() {
        return Err("no groups found in that column".into());
    }
    // Computed by LibreOffice in a cell nothing uses, on a copy: the workbook
    // itself is opened as a template and never saved.
    let cell = format!("{}1", next_col(&sheet.last_col));
    let probed = format!("{}/ask-probe.txt", BENCH_OUT);
    let mut r = alloc::vec![probed.clone(), String::new(), cell, String::from(path)];
    r.extend(asked.iter().map(|(_, f)| f.clone()));
    macro_batch("Probe", &[r]);
    let lines = read_any(&probed).unwrap_or_default();
    let got: Vec<&str> = lines.lines().collect();
    for (i, (label, f)) in asked.iter().enumerate() {
        let v = got
            .get(i)
            .map(|l| l.strip_prefix("OK\t").unwrap_or(l).trim())
            .map(shown)
            .unwrap_or_else(|| String::from("(not computed)"));
        if label.is_empty() {
            note(format!("{}    {}", v, f));
        } else {
            note(format!("{:<16} {:>12}    {}", label, v, f));
        }
    }
    Ok(())
}

/// What `diag office` asks of this, with no model and no LibreOffice.
pub fn checks() -> Vec<(&'static str, bool)> {
    let view = "Sheet Sales, used area A1:C4\nA1=Region | B1=Units | C1=Revenue\nA2=East | B2=3 | C2=30\nA3=West | B3=1 | C3=10 {=B3*10}\nA4=East | B4=2 | C4=20\n";
    let s = parse(view);
    let mut out = Vec::new();
    out.push(("a view's header row is its columns", s.as_ref().is_some_and(|s| s.headers.len() == 3 && s.headers[2].1 == "Revenue")));
    out.push(("and its used area is how far the data goes", s.as_ref().is_some_and(|s| s.last_row == 4 && s.last_col == "C")));
    out.push(("a formula's text is not taken for its value", s.as_ref().is_some_and(|s| s.cells.iter().any(|(a, v)| a == "C3" && v == "10"))));
    out.push(("groups are a column's distinct values, in order", s.as_ref().is_some_and(|s| groups(s, "A") == ["East", "West"])));
    let total = &OPS[0];
    out.push(("a total is one SUM over the data rows", s.as_ref().is_some_and(|s| formulas(s, total, "C", None)[0].1 == "=SUM(C2:C4)")));
    let per = &OPS[5];
    out.push((
        "a total per group is a SUMIF per distinct value",
        s.as_ref().is_some_and(|s| formulas(s, per, "C", Some("A")) == [
            (String::from("East"), String::from("=SUMIF(A2:A4,\"East\",C2:C4)")),
            (String::from("West"), String::from("=SUMIF(A2:A4,\"West\",C2:C4)")),
        ]),
    ));
    out.push(("the spare cell is the column after the data", next_col("C") == "D" && next_col("Z") == "AA" && next_col("AZ") == "BA"));
    out.push(("a column of numbers is numeric, one of names is not", s.as_ref().is_some_and(|s| numeric(s, "C") && !numeric(s, "A"))));
    out.push(("'each', 'per' and 'by' ask for groups", wants_groups("revenue by region") && wants_groups("units of each product") && !wants_groups("what is our total revenue")));
    let hs = ["Region", "Rep", "Unit Price", "Revenue"];
    out.push(("a header the request names is picked, in the request's order", named("revenue by region", &hs, &[0, 1, 2, 3]) == [3, 0]));
    out.push(("a plural names its header", named("how much did the regions make", &hs, &[0, 1]) == [0]));
    out.push(("'how many' is a count and 'biggest' the largest", op_named("how many sales were there") == Some("count") && op_named("the biggest single sale") == Some("largest")));
    out.push(("an average is shown to two places, a whole number as it is", shown("731.142857142857") == "731.14" && shown("32136") == "32136" && shown("East") == "East"));
    out.push(("a sheet with no header row is refused", parse("Sheet S, used area A1:A1\n").is_none()));
    out
}
