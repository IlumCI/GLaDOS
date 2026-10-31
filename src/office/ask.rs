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

use super::{headless_macros, macro_batch, note, read_any, ready, start_app, App, BENCH_OUT, OUT};

/// The workbook `colleague` asks about: the last one `office open` put on
/// the desktop.
pub static BOOK: crate::sync::Spin<Option<String>> = crate::sync::Spin::new(None);

/// Put a workbook on the desktop for `colleague` to work on.
///
/// A copy in /tmp is what opens: a workbook read off the read-only disk opens
/// in Calc with a read-only banner across it, which is not how anybody's own
/// spreadsheet looks.
pub fn open(path: &str) -> Result<(), String> {
    let bytes = super::read_bytes(path).ok_or_else(|| format!("no such file: {}", path))?;
    if !crate::linux::ns::is_dir(OUT) {
        crate::linux::ns::make_dir(OUT);
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    let copy = format!("{}/{}", OUT, name);
    crate::linux::ns::write_blob(&copy, bytes);
    // Questions are answered from the original: the copy is held open by the
    // desktop's Calc, and LibreOffice will not load a document another
    // instance has locked -- every `colleague` answered "no header row".
    *BOOK.lock_irq() = Some(String::from(path));
    NEXT_ROW.store(0, core::sync::atomic::Ordering::Relaxed);
    super::report::reset(path, 0);
    show(&copy)?;
    // `Bench` into the desktop's own profile, for `Bench.Live`: now that the
    // profile exists, since one written before the first start is replaced
    // by the presets. The Standard library is read when first used, so the
    // running Calc finds it.
    super::seed_bench_macros_at(super::PROFILE)?;
    // The headless profile made now rather than by the first question: it is
    // the slow part of the first launch, and run during a question it took
    // the machine from the Calc being typed into.
    headless_macros()
}

/// The row of the open workbook's sheet the next answer is typed at: under
/// the data, then under the answer before it. Zero until the first question
/// has read how far the data goes.
static NEXT_ROW: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Close whatever Calc has open and open `file` in a Calc of its own.
///
/// Not handed to the running one: `start_app` accepted any Calc window as the
/// one it opened, so with the original still up the copy was "opened" at once
/// and never appeared. A fresh instance after the old one has gone is slower
/// and is the one whose window is the copy's.
fn show(file: &str) -> Result<(), String> {
    if crate::gfx::desk::close_titled("LibreOffice Calc") > 0 {
        let deadline = crate::dev::lapic::ticks() + 30 * crate::TIMER_HZ as u64;
        while crate::dev::lapic::ticks() < deadline
            && (crate::gfx::desk::window_titled("LibreOffice").is_some() || crate::job::list().iter().any(|r| r.state == "running" && r.title.contains("--calc")))
        {
            crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 4);
        }
    }
    start_app(App::Calc, Some(file))?;
    let stem = file.rsplit('/').next().unwrap_or(file);
    crate::gfx::desk::focus_titled(stem);
    Ok(())
}

/// `colleague <request>`: a question about the open workbook, answered in
/// the workbook on screen -- the formulas typed into Calc under the data, so
/// the sheet computes them where it can be watched -- and in the terminal.
pub fn colleague(request: &str) {
    let book = BOOK.lock_irq().clone();
    let Some(book) = book else {
        crate::kprintln!("  no workbook is open -- 'office open <path>' first");
        return;
    };
    let done = if super::report::asks_for(request) {
        super::report::write()
    } else {
        ask(&book, &format!("{} --live", request.trim()))
    };
    if let Err(why) = done {
        crate::kprintln!("  {}", why);
    }
}

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
    Op { name: "largest per group", whole: "MAX", grouped: Some("MAXIFS"), needs_value: true, by: true },
    Op { name: "smallest per group", whole: "MIN", grouped: Some("MINIFS"), needs_value: true, by: true },
];

/// A sheet as `Bench.View` rendered it: the header row, how far the data
/// goes, and the cells seen.
struct Sheet {
    /// The sheet's own name, which the answer's formulas refer back to.
    name: String,
    /// (column letters, header text), left to right.
    headers: Vec<(String, String)>,
    last_row: usize,
    last_col: String,
    /// (address, value) for every cell the view carried.
    cells: Vec<(String, String)>,
    /// (column letters, every distinct value) for each column of names, over
    /// the whole sheet and not only the rows the view carried.
    distinct: Vec<(String, Vec<String>)>,
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
    let name = String::from(head.strip_prefix("Sheet ")?.rsplit_once(", used area")?.0);
    let area = head.rsplit("used area ").next()?;
    let end = area.split(':').nth(1)?.trim();
    let (last_col, last_row) = split_addr(end)?;
    let mut cells = Vec::new();
    let mut distinct = Vec::new();
    for l in lines {
        if let Some((col, vals)) = l.strip_prefix("Distinct ").and_then(|r| r.split_once(": ")) {
            distinct.push((String::from(col), vals.split('|').map(String::from).collect()));
            continue;
        }
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
    (!headers.is_empty()).then_some(Sheet { name, headers, last_row, last_col: String::from(last_col), cells, distinct })
}

/// A formula's ranges qualified with the sheet they are on, for a formula
/// written on another sheet: `B2:B41` becomes `$Sales.B2:B41`. Text in quotes
/// is left alone, so a group called "A1:A2" is not a reference.
fn qualify(formula: &str, sheet: &str) -> String {
    let prefix = if sheet.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        format!("${}.", sheet)
    } else {
        format!("$'{}'.", sheet.replace('\'', "''"))
    };
    // One cell reference at `i`: capitals then digits. Answers its end.
    let cell = |b: &[u8], i: usize| -> Option<usize> {
        let mut j = i;
        while j < b.len() && b[j].is_ascii_uppercase() {
            j += 1;
        }
        let k = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        (k > i && j > k).then_some(j)
    };
    let b = formula.as_bytes();
    let mut out = String::new();
    let (mut i, mut quoted) = (0, false);
    while i < b.len() {
        if b[i] == b'"' {
            quoted = !quoted;
        } else if !quoted && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            if let Some(e) = cell(b, i) {
                if e < b.len() && b[e] == b':' {
                    if let Some(e2) = cell(b, e + 1) {
                        out.push_str(&prefix);
                        out.push_str(&formula[i..e2]);
                        i = e2;
                        continue;
                    }
                }
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
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
    // The whole column's, sorted, when the view carried them: a ledger's
    // groups are not all in its first sixty rows.
    if let Some((_, vals)) = sheet.distinct.iter().find(|(c, _)| c == col) {
        let mut v = vals.clone();
        v.sort();
        return v;
    }
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
                } else if f.ends_with("IFS") {
                    // MAXIFS and MINIFS put the range they reduce first.
                    format!("={}({},{},\"{}\")", f, range(value), range(g), quoted)
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
    let w: Vec<String> = request.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_ascii_lowercase()).collect();
    // "Per sale" is per row, which is what an average already is: grouped by
    // it, "the average revenue per sale" became forty averages, one a date.
    let row = |i: usize| w.get(i + 1).is_some_and(|n| matches!(n.trim_end_matches('s'), "sale" | "row" | "order" | "transaction" | "entry" | "entrie" | "record" | "line" | "item"));
    w.iter().enumerate().any(|(i, x)| matches!(x.as_str(), "each" | "per" | "by" | "every" | "breakdown") && !row(i))
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

/// What an answer's column is headed: "Total Revenue", "Average Units",
/// "Count".
fn title(word: &str, of: Option<&str>) -> String {
    let mut w = String::from(word);
    if let Some(f) = w.get_mut(0..1) {
        f.make_ascii_uppercase();
    }
    match of {
        Some(c) => format!("{} {}", w, c),
        None => w,
    }
}

/// A number as a person reads one: thousands grouped, an average to two
/// places. Anything that is not a number is left as it came.
fn readable(v: &str) -> String {
    let v = shown(v);
    let Ok(_) = v.parse::<f64>() else { return v };
    let (sign, rest) = match v.strip_prefix('-') {
        Some(r) => ("-", r),
        None => ("", v.as_str()),
    };
    let (int, frac) = match rest.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (rest, None),
    };
    let mut out = String::from(sign);
    for (i, ch) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    if let Some(f) = frac {
        out.push('.');
        out.push_str(f);
    }
    out
}

/// Text cut into lines no wider than `width` columns, at spaces where there
/// are any. A word wider than the line is cut, since it has to go somewhere.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split(' ').filter(|w| !w.is_empty()) {
        let mut word = word;
        loop {
            let have = line.chars().count();
            let need = word.chars().count() + usize::from(have > 0);
            if have + need <= width {
                if have > 0 {
                    line.push(' ');
                }
                line.push_str(word);
                break;
            }
            if have > 0 {
                lines.push(core::mem::take(&mut line));
                continue;
            }
            let cut = word.char_indices().nth(width).map_or(word.len(), |(i, _)| i);
            lines.push(String::from(&word[..cut]));
            word = &word[cut..];
            if word.is_empty() {
                break;
            }
        }
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

/// The terminal's width as `note` writes into it: two columns of indent.
fn room() -> usize {
    crate::gfx::console::cols().saturating_sub(3).max(24)
}

/// One labelled line of the report, wrapped to the terminal as it is now
/// with the label's column kept clear: a line written for a full-width
/// terminal and read in a tiled one was cut mid-word and carried on at the
/// left edge, where it read as a line of its own.
fn say(label: &str, text: &str) {
    const LABEL: usize = 9;
    for (i, l) in wrap(text, room().saturating_sub(LABEL)).into_iter().enumerate() {
        let tag = if i == 0 { label } else { "" };
        note(format!("{:<w$}{}", tag, l, w = LABEL));
    }
}

/// A two-column table in box-drawing characters, as wide as it needs to be
/// and never wider than the terminal: labels are cut with an ellipsis before
/// a number is, since a number cut short is a different number.
fn table(heads: (&str, &str), rows: &[(String, String)]) -> Vec<String> {
    let len = |s: &str| s.chars().count();
    let mut b = len(heads.1).max(rows.iter().map(|(_, v)| len(v)).max().unwrap_or(0));
    let mut a = len(heads.0).max(rows.iter().map(|(l, _)| len(l)).max().unwrap_or(0));
    // Seven columns of rule and padding around the two cells.
    let room = room().saturating_sub(7);
    if a + b > room {
        b = b.min(room.saturating_sub(4).max(4));
        a = room.saturating_sub(b).max(3);
    }
    let fit = |s: &str, w: usize| -> String {
        if len(s) <= w {
            return String::from(s);
        }
        let mut t: String = s.chars().take(w.saturating_sub(1)).collect();
        t.push('…');
        t
    };
    let rule = |l: char, m: char, r: char| {
        let mut s = String::new();
        s.push(l);
        s.extend(core::iter::repeat('─').take(a + 2));
        s.push(m);
        s.extend(core::iter::repeat('─').take(b + 2));
        s.push(r);
        s
    };
    let row = |x: &str, y: &str| format!("│ {:<a$} │ {:>b$} │", fit(x, a), fit(y, b), a = a, b = b);
    let mut out = alloc::vec![rule('┌', '┬', '┐'), row(heads.0, heads.1), rule('├', '┼', '┤')];
    for (l, v) in rows {
        out.push(row(l, v));
    }
    out.push(rule('└', '┴', '┘'));
    out
}

/// The answer written into the workbook on screen, and read back as a table.
///
/// **Into the open sheet, not a copy.** The first version put the answer on a
/// sheet of its own in a copy and opened that copy, which reads -- correctly
/// -- as a picture of a result: Calc relaunched, and there it was. Here the
/// question, the headings and every formula arrive cell by cell under the
/// data, selected as they are written, and each number appears the moment its
/// formula is entered, computed by the Calc being watched.
///
/// **And the terminal's figures are read back from that sheet** -- the cells
/// Calc just computed, not the same formulas run again somewhere else.
fn answer_live(path: &str, sheet: &Sheet, request: &str, heads: (String, String), asked: &[(String, String)], source: Option<String>) -> Result<(), String> {
    use core::sync::atomic::Ordering;
    if crate::gfx::desk::window_titled("LibreOffice Calc").is_none() {
        return Err("no Calc window to answer in -- 'office open <path>' first".into());
    }
    // Under the data with a row to spare, or under the last answer.
    let top = match NEXT_ROW.load(Ordering::Relaxed) {
        0 => sheet.last_row + 2,
        r => r,
    };
    let grouped = asked.len() > 1 || !asked[0].0.is_empty();
    let first = top + if grouped { 2 } else { 1 };
    let last = first + asked.len() - 1;
    say("writing", &format!("into the sheet at A{}: the question, then {}", top, if grouped { format!("{} formulas in B{}:B{}", asked.len(), first, last) } else { format!("the formula in B{}", first) }));

    let mut q = String::from(request.trim());
    if let Some(f) = q.get_mut(0..1) {
        f.make_ascii_uppercase();
    }
    if !q.ends_with('?') {
        q.push('?');
    }
    let left = if grouped { heads.0.as_str() } else { heads.1.as_str() };
    // The cells, in the order they are written: the question, the headings,
    // then each label and its formula. A single answer has no heading row;
    // its label is its heading.
    let mut job: Vec<(String, String, &str)> = Vec::new();
    job.push((format!("A{}", top), q.clone(), "q"));
    if grouped {
        job.push((format!("A{}", top + 1), String::from(left), "h"));
        job.push((format!("B{}", top + 1), heads.1.clone(), "h"));
    }
    for (i, (label, f)) in asked.iter().enumerate() {
        let row = first + i;
        job.push((format!("A{}", row), if label.is_empty() { String::from(left) } else { label.clone() }, "h"));
        job.push((format!("B{}", row), f.clone(), "v"));
    }
    NEXT_ROW.store(last + 3, Ordering::Relaxed);

    // Written by the Calc on screen, which polls for the job (`Bench.Watch`).
    // The window is raised first so the answer is seen arriving.
    let stem = path.rsplit('/').next().unwrap_or(path);
    if !crate::gfx::desk::focus_titled(&format!("{} ", stem)) {
        crate::gfx::desk::focus_titled("LibreOffice Calc");
    }
    let out = format!("{}/live-out.txt", BENCH_OUT);
    let _ = crate::linux::ns::detach(&out);
    let mut text = format!("{}\n{}\n", out, stem);
    for (cell, what, style) in &job {
        // A value carries the column it summarises, for its number format.
        let src = if *style == "v" { source.as_deref().unwrap_or("") } else { "" };
        text.push_str(&format!("{}\t{}\t{}\t{}\n", cell, what, style, src));
    }
    // Picked up by `Bench.Watch` in the Calc on screen.
    crate::linux::ns::write_blob(&format!("{}/live.job", BENCH_OUT), text.into_bytes());
    let deadline = crate::dev::lapic::ticks() + 180 * crate::TIMER_HZ as u64;
    let lines = loop {
        if let Ok(t) = read_any(&out) {
            break t;
        }
        if crate::dev::lapic::ticks() > deadline {
            return Err("the Calc on screen did not answer within three minutes".into());
        }
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 4);
    };
    if let Some(e) = lines.strip_prefix("ERROR") {
        return Err(format!("Calc could not write the answer:{}", e.trim_end()));
    }
    // "OK", the value, and the cell's own text, per value cell in order. The
    // terminal shows the text, exactly as the sheet does: 976,802.90 there
    // and 976,802.9 here read as two different answers.
    let parts: Vec<Vec<&str>> = lines.lines().map(|l| l.split('\t').collect()).collect();
    let got: Vec<&str> = parts.iter().map(|p| p.get(1).copied().unwrap_or("")).collect();
    let text_of: Vec<Option<&str>> = parts.iter().map(|p| p.get(2).copied().filter(|t| !t.trim().is_empty())).collect();

    let rows: Vec<(String, String)> = asked
        .iter()
        .enumerate()
        .map(|(i, (label, _))| {
            let v = match text_of.get(i).copied().flatten() {
                Some(t) => String::from(t.trim()),
                None => got.get(i).map(|l| readable(l.trim())).unwrap_or_else(|| String::from("?")),
            };
            (if label.is_empty() { String::from(left) } else { label.clone() }, v)
        })
        .collect();
    note(String::new());
    for l in wrap(&q, room()) {
        note(l);
    }
    let shown_heads = if grouped { (heads.0.as_str(), heads.1.as_str()) } else { ("", "") };
    let t = table(shown_heads, &rows);
    // A single answer needs no heading row.
    let t: Vec<String> = if grouped { t } else { t.into_iter().enumerate().filter(|(i, _)| *i != 1 && *i != 2).map(|(_, l)| l).collect() };
    for l in t {
        note(l);
    }
    if let Some(b) = super::report::BOOK.lock_irq().as_mut() {
        b.1 = sheet.last_row.saturating_sub(1);
    }
    super::report::keep(super::report::Answer {
        question: q.clone(),
        heads: heads.clone(),
        rows: asked
            .iter()
            .zip(&rows)
            .enumerate()
            .map(|(i, (_, (l, shown)))| (l.clone(), got.get(i).and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(0.0), shown.clone()))
            .collect(),
        word: String::from(heads.1.split(' ').next().unwrap_or("").to_ascii_lowercase().as_str()),
        formula: asked[0].1.clone(),
        cells: if grouped { format!("B{}:B{}, one formula per {}", first, last, heads.0.to_lowercase()) } else { format!("B{}", first) },
    });
    let f = &asked[0].1;
    if grouped {
        say("formula", &format!("{} and {} like it, in B{}:B{}", f, asked.len() - 1, first, last));
    } else {
        say("formula", &format!("{} in B{}", f, first));
    }
    say("", "computed by the Calc on screen, and read back from it");
    Ok(())
}

pub fn ask(path: &str, request: &str) -> Result<(), String> {
    // `--open`: the answer written into a copy of the workbook, beside the
    // data, and that copy opened in Calc on the desktop.
    let (request, open) = match request.strip_suffix("--open") {
        Some(r) => (r.trim(), true),
        None => (request, false),
    };
    // `--live`: typed into the Calc on the desktop, the one `office open` put
    // there, and reported as a table.
    let (request, live) = match request.strip_suffix("--live") {
        Some(r) => (r.trim(), true),
        None => (request, false),
    };
    ready()?;
    headless_macros()?;
    let out = format!("{}/ask-view.txt", BENCH_OUT);
    macro_batch("View", &[alloc::vec![out.clone(), String::new(), String::from(path)]]);
    let view = read_any(&out).map_err(|_| format!("LibreOffice could not read {}", path))?;
    let sheet = parse(&view).ok_or_else(|| format!("no header row found in {}", path))?;
    let names: Vec<&str> = sheet.headers.iter().map(|(_, h)| h.as_str()).collect();
    if live {
        say("sheet", &format!("{}, {} rows: {}", sheet.name, sheet.last_row.saturating_sub(1), names.join(", ")));
    } else {
        note(format!("{} column(s): {}", names.len(), names.join(", ")));
    }
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
    if live {
        let word = op.name.split(' ').next().unwrap_or(op.name);
        say(
            "chose",
            &format!(
                "{}{}{}",
                word,
                if op.needs_value { format!(" of {}", header_of(&value)) } else { String::from(" of rows") },
                by.as_deref().map(|g| format!(", per {}", header_of(g))).unwrap_or_default(),
            ),
        );
        say("", &said.join(", "));
        let asked = formulas(&sheet, op, &value, by.as_deref());
        if asked.is_empty() {
            return Err("no groups found in that column".into());
        }
        let heads = (
            by.as_deref().map(|g| header_of(g)).unwrap_or_default(),
            title(word, op.needs_value.then(|| header_of(&value)).as_deref()),
        );
        let source = op.needs_value.then(|| format!("{}2", value));
        return answer_live(path, &sheet, request, heads, &asked, source);
    }
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
    if open {
        let stem = path.rsplit('/').next().unwrap_or(path).split('.').next().unwrap_or("answer");
        // A new name each time: the copy before may be open in Calc, and a
        // headless instance cannot save over a document another one holds.
        static N: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(1);
        let n = N.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let copy = format!("{}/{}-answered-{}.ods", OUT, stem, n);
        let mut r = alloc::vec![
            format!("{}/ask-answer.txt", BENCH_OUT),
            String::from(path),
            copy.clone(),
            String::from("Colleague"),
            String::from(request)
        ];
        for (label, f) in &asked {
            r.push(if label.is_empty() { String::from("Answer") } else { label.clone() });
            r.push(qualify(f, &sheet.name));
        }
        macro_batch("Answer", &[r]);
        if read_any(&format!("{}/ask-answer.txt", BENCH_OUT)).is_err() {
            return Err("LibreOffice did not write the answered copy".into());
        }
        show(&copy)?;
        note(format!("opened {} in Calc, the answer beside the data", copy));
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
    out.push(("a range written on another sheet names its own", qualify("=SUMIF(B2:B4,\"East\",C2:C4)", "Sales") == "=SUMIF($Sales.B2:B4,\"East\",$Sales.C2:C4)"));
    out.push(("but text in quotes is not a reference", qualify("=COUNTIF(A2:A4,\"A1:B2\")", "My Data") == "=COUNTIF($'My Data'.A2:A4,\"A1:B2\")"));
    out.push(("the spare cell is the column after the data", next_col("C") == "D" && next_col("Z") == "AA" && next_col("AZ") == "BA"));
    out.push(("a column of numbers is numeric, one of names is not", s.as_ref().is_some_and(|s| numeric(s, "C") && !numeric(s, "A"))));
    out.push(("'each', 'per' and 'by' ask for groups", wants_groups("revenue by region") && wants_groups("units of each product") && !wants_groups("what is our total revenue")));
    out.push(("but 'per sale' is per row, not a grouping", !wants_groups("the average revenue per sale") && !wants_groups("units for each order") && wants_groups("revenue per region")));
    let hs = ["Region", "Rep", "Unit Price", "Revenue"];
    out.push(("a header the request names is picked, in the request's order", named("revenue by region", &hs, &[0, 1, 2, 3]) == [3, 0]));
    out.push(("a plural names its header", named("how much did the regions make", &hs, &[0, 1]) == [0]));
    out.push(("'how many' is a count and 'biggest' the largest", op_named("how many sales were there") == Some("count") && op_named("the biggest single sale") == Some("largest")));
    out.push(("an average is shown to two places, a whole number as it is", shown("731.142857142857") == "731.14" && shown("32136") == "32136" && shown("East") == "East"));
    out.push(("a sheet with no header row is refused", parse("Sheet S, used area A1:A1\n").is_none()));
    out.push(("a number reads with its thousands grouped", readable("32136") == "32,136" && readable("-1234567.5") == "-1,234,567.5" && readable("731.142857142857") == "731.14" && readable("East") == "East"));
    out.push(("text wraps at spaces, inside the width", wrap("the quick brown fox jumps", 10) == ["the quick", "brown fox", "jumps"]));
    out.push(("a word wider than the line is cut, not lost", wrap("abcdefghijkl", 8) == ["abcdefgh", "ijkl"]));
    out.push(("an answer's column says what it is", title("total", Some("Revenue")) == "Total Revenue" && title("count", None) == "Count"));
    out
}
