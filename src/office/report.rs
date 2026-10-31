//! `colleague write a report`: the answers given so far, written up in Writer.
//!
//! **Every figure in it is one Calc computed.** The tables and the findings
//! under them -- which group came first, its share, which came last -- are
//! the kernel's arithmetic over the values read back from the sheet, so they
//! cannot be wrong in a way the sheet is not. The model writes one thing: the
//! summary at the top, in prose, from those findings. A 0.6B model asked to
//! restate numbers restates some of them wrongly, so a sentence carrying a
//! number that is not among the findings is dropped rather than kept, and the
//! report says how many were.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use super::{note, BENCH_OUT, OUT};

/// One answered question, as the sheet answered it.
pub struct Answer {
    pub question: String,
    /// (the groups' heading, the values' heading); the first is empty for a
    /// single number.
    pub heads: (String, String),
    /// (label, value, the value as the terminal showed it).
    pub rows: Vec<(String, f64, String)>,
    /// What the operation was: total, average, count, largest, smallest.
    pub word: String,
    /// The first formula, and where the answer is in the sheet.
    pub formula: String,
    pub cells: String,
}

/// The workbook the answers are about: its name and how many rows of data.
pub static BOOK: crate::sync::Spin<Option<(String, usize)>> = crate::sync::Spin::new(None);
pub static ANSWERS: crate::sync::Spin<Vec<Answer>> = crate::sync::Spin::new(Vec::new());

/// Forget the answers: a new workbook was opened.
pub fn reset(book: &str, rows: usize) {
    *BOOK.lock_irq() = Some((String::from(book), rows));
    ANSWERS.lock_irq().clear();
}

pub fn keep(a: Answer) {
    ANSWERS.lock_irq().push(a);
}

/// Whether a request asks for the report rather than for a number.
pub fn asks_for(request: &str) -> bool {
    let t = request.to_ascii_lowercase();
    t.contains("report") || t.contains("write it up") || t.contains("write up")
}

/// A number as the report prints it: thousands grouped, two places when it
/// has a fraction.
fn figure(v: f64) -> String {
    let whole = (v - (v as i64) as f64).abs() < 1e-9;
    let s = if whole { format!("{}", v as i64) } else { format!("{:.2}", v) };
    group(&s)
}

fn group(s: &str) -> String {
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => ("-", r),
        None => ("", s),
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

fn percent(part: f64, whole: f64) -> String {
    if whole == 0.0 {
        return String::from("0%");
    }
    format!("{:.1}%", 100.0 * part / whole)
}

/// What an answer says, in sentences built from its numbers alone.
fn findings(a: &Answer) -> Vec<String> {
    let mut out = Vec::new();
    if a.heads.0.is_empty() || a.rows.len() < 2 {
        if let Some(r) = a.rows.first() {
            out.push(format!("{}: {}.", r.0, if r.2.is_empty() { figure(r.1) } else { r.2.clone() }));
        }
        return out;
    }
    let mut by: Vec<&(String, f64, String)> = a.rows.iter().collect();
    by.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(core::cmp::Ordering::Equal));
    let (top, bottom) = (by[0], by[by.len() - 1]);
    // A group's own figure as the sheet showed it -- "6%", not 0.06.
    let shown = |r: &(String, f64, String)| if r.2.is_empty() { figure(r.1) } else { r.2.clone() };
    let what = a.heads.1.to_lowercase();
    let group = a.heads.0.to_lowercase();
    // A share means something only for a sum or a count: the parts add up
    // to a whole. An average or a maximum per group does not.
    let adds = a.word == "total" || a.word == "count";
    if adds {
        let whole: f64 = a.rows.iter().map(|r| r.1).sum();
        out.push(format!(
            "{} leads on {}, with {} of {} ({}).",
            top.0,
            what,
            figure(top.1),
            figure(whole),
            percent(top.1, whole)
        ));
        out.push(format!("{} is last, with {} ({}).", bottom.0, figure(bottom.1), percent(bottom.1, whole)));
        if by.len() > 2 {
            let gap = top.1 / bottom.1.max(f64::MIN_POSITIVE);
            out.push(format!("The first {} is {:.1} times the last.", group, gap));
        }
    } else {
        // What the column is, without the operation's word: "revenue" out
        // of "Largest Revenue". Said per operation, because "Sweden is
        // lowest" about a largest-sale question was restated by the model as
        // Sweden having the lowest revenue.
        let of = a.heads.1.split_once(' ').map(|(_, r)| r.to_lowercase()).unwrap_or_else(|| what.clone());
        match a.word.as_str() {
            "largest" => {
                out.push(format!("The largest single {} was in {}: {}.", of, top.0, shown(top)));
                out.push(format!("{}'s largest {} was the smallest of any {}, at {}.", bottom.0, of, group, shown(bottom)));
            }
            "smallest" => {
                out.push(format!("The smallest single {} was in {}: {}.", of, bottom.0, shown(bottom)));
                out.push(format!("{}'s smallest {} was the largest of any {}, at {}.", top.0, of, group, shown(top)));
            }
            _ => {
                out.push(format!("{} has the highest average {}, at {}.", top.0, of, shown(top)));
                out.push(format!("{} has the lowest, at {}.", bottom.0, shown(bottom)));
            }
        }
    }
    out
}

/// The numbers a sentence states, as digits alone: "9,542.50" is "954250"
/// and "29.7%" is "297". Compared that way, a figure written with or without
/// its commas is the same figure.
fn numbers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let b: Vec<char> = text.chars().collect();
    for (i, &c) in b.iter().enumerate() {
        let joins = (c == ',' || c == '.') && !cur.is_empty() && b.get(i + 1).is_some_and(|n| n.is_ascii_digit());
        if c.is_ascii_digit() {
            cur.push(c);
        } else if !joins && !cur.is_empty() {
            out.push(core::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The model's summary, kept a sentence at a time: one stating a number that
/// none of the findings states is dropped. Answers the sentences kept and how
/// many went.
fn vet(text: &str, facts: &[String]) -> (Vec<String>, usize) {
    let known: Vec<String> = facts.iter().flat_map(|f| numbers(f)).collect();
    let mut kept = Vec::new();
    let mut dropped = 0;
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut sentence = String::new();
    let mut flush = |s: &mut String, kept: &mut Vec<String>, dropped: &mut usize| {
        let t = s.trim();
        if !t.is_empty() {
            if numbers(t).iter().all(|n| known.contains(n)) {
                // A currency sign is the model's, never the sheet's.
                kept.push(t.replace('$', ""));
            } else {
                *dropped += 1;
            }
        }
        s.clear();
    };
    let chars: Vec<char> = flat.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        sentence.push(c);
        // A full stop ends a sentence; one between digits is a decimal point.
        let decimal = c == '.' && i > 0 && chars[i - 1].is_ascii_digit() && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
        if matches!(c, '.' | '!' | '?') && !decimal {
            flush(&mut sentence, &mut kept, &mut dropped);
        }
    }
    flush(&mut sentence, &mut kept, &mut dropped);
    (kept, dropped)
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The report as a flat ODF text document, which Writer opens as it is.
fn fodt(title: &str, sub: &str, summary: &[String], dropped: usize, answers: &[Answer]) -> String {
    let mut x = String::from(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" office:version="1.3" office:mimetype="application/vnd.oasis.opendocument.text">
<office:automatic-styles>
<style:style style:name="Title" style:family="paragraph"><style:paragraph-properties fo:margin-bottom="0.04in"/><style:text-properties fo:font-size="24pt" fo:font-weight="bold" fo:color="#1f2a44"/></style:style>
<style:style style:name="Sub" style:family="paragraph"><style:paragraph-properties fo:margin-bottom="0.2in" fo:border-bottom="1.5pt solid #f28c1e" fo:padding-bottom="0.06in"/><style:text-properties fo:font-size="10pt" fo:color="#5a6478"/></style:style>
<style:style style:name="H1" style:family="paragraph"><style:paragraph-properties fo:margin-top="0.22in" fo:margin-bottom="0.08in" fo:keep-with-next="always"/><style:text-properties fo:font-size="15pt" fo:font-weight="bold" fo:color="#c25e00"/></style:style>
<style:style style:name="Body" style:family="paragraph"><style:paragraph-properties fo:margin-bottom="0.08in" fo:line-height="125%"/><style:text-properties fo:font-size="11pt"/></style:style>
<style:style style:name="Find" style:family="paragraph"><style:paragraph-properties fo:margin-left="0.15in" fo:margin-bottom="0.04in"/><style:text-properties fo:font-size="11pt"/></style:style>
<style:style style:name="Note" style:family="paragraph"><style:paragraph-properties fo:margin-top="0.06in"/><style:text-properties fo:font-size="8.5pt" fo:color="#7a8396" style:font-name="Liberation Mono"/></style:style>
<style:style style:name="Tbl" style:family="table"><style:table-properties style:width="4.6in" table:align="left" fo:margin-top="0.06in" fo:margin-bottom="0.08in"/></style:style>
<style:style style:name="ColA" style:family="table-column"><style:table-column-properties style:column-width="2.6in"/></style:style>
<style:style style:name="ColB" style:family="table-column"><style:table-column-properties style:column-width="2.0in"/></style:style>
<style:style style:name="Hd" style:family="table-cell"><style:table-cell-properties fo:background-color="#f28c1e" fo:padding="0.05in" fo:border="0.5pt solid #d9770f"/></style:style>
<style:style style:name="Cell" style:family="table-cell"><style:table-cell-properties fo:padding="0.04in" fo:border="0.5pt solid #e3d7c8"/></style:style>
<style:style style:name="Alt" style:family="table-cell"><style:table-cell-properties fo:background-color="#fff6ea" fo:padding="0.04in" fo:border="0.5pt solid #e3d7c8"/></style:style>
<style:style style:name="HdT" style:family="paragraph"><style:text-properties fo:font-weight="bold" fo:color="#ffffff" fo:font-size="10.5pt"/></style:style>
<style:style style:name="HdN" style:family="paragraph"><style:paragraph-properties fo:text-align="end"/><style:text-properties fo:font-weight="bold" fo:color="#ffffff" fo:font-size="10.5pt"/></style:style>
<style:style style:name="CT" style:family="paragraph"><style:text-properties fo:font-size="10.5pt"/></style:style>
<style:style style:name="CN" style:family="paragraph"><style:paragraph-properties fo:text-align="end"/><style:text-properties fo:font-size="10.5pt"/></style:style>
</office:automatic-styles>
<office:body><office:text>
"##,
    );
    x.push_str(&format!("<text:p text:style-name=\"Title\">{}</text:p>\n", esc(title)));
    x.push_str(&format!("<text:p text:style-name=\"Sub\">{}</text:p>\n", esc(sub)));
    if !summary.is_empty() {
        x.push_str("<text:p text:style-name=\"H1\">Summary</text:p>\n");
        x.push_str(&format!("<text:p text:style-name=\"Body\">{}</text:p>\n", esc(&summary.join(" "))));
        if dropped > 0 {
            x.push_str(&format!(
                "<text:p text:style-name=\"Note\">{} sentence(s) of the drafted summary were left out for stating a figure the sheet did not compute.</text:p>\n",
                dropped
            ));
        }
    }
    for (n, a) in answers.iter().enumerate() {
        x.push_str(&format!("<text:p text:style-name=\"H1\">{}. {}</text:p>\n", n + 1, esc(&a.question)));
        x.push_str(&format!("<table:table table:name=\"Answer{}\" table:style-name=\"Tbl\">\n", n + 1));
        x.push_str("<table:table-column table:style-name=\"ColA\"/><table:table-column table:style-name=\"ColB\"/>\n");
        let (ha, hb) = if a.heads.0.is_empty() { ("", a.heads.1.as_str()) } else { (a.heads.0.as_str(), a.heads.1.as_str()) };
        x.push_str(&format!(
            "<table:table-row><table:table-cell table:style-name=\"Hd\"><text:p text:style-name=\"HdT\">{}</text:p></table:table-cell><table:table-cell table:style-name=\"Hd\"><text:p text:style-name=\"HdN\">{}</text:p></table:table-cell></table:table-row>\n",
            esc(ha),
            esc(hb)
        ));
        for (i, (l, _, shown)) in a.rows.iter().enumerate() {
            let st = if i % 2 == 1 { "Alt" } else { "Cell" };
            x.push_str(&format!(
                "<table:table-row><table:table-cell table:style-name=\"{st}\"><text:p text:style-name=\"CT\">{}</text:p></table:table-cell><table:table-cell table:style-name=\"{st}\"><text:p text:style-name=\"CN\">{}</text:p></table:table-cell></table:table-row>\n",
                esc(l),
                esc(shown)
            ));
        }
        x.push_str("</table:table>\n");
        for f in findings(a) {
            x.push_str(&format!("<text:p text:style-name=\"Find\">• {}</text:p>\n", esc(&f)));
        }
        x.push_str(&format!("<text:p text:style-name=\"Note\">Computed in Calc: {} ({})</text:p>\n", esc(&a.formula), esc(&a.cells)));
    }
    x.push_str("</office:text></office:body></office:document>\n");
    x
}

/// Write the report and open it in Writer.
pub fn write() -> Result<(), String> {
    let (book, rows) = BOOK.lock_irq().clone().ok_or("no workbook is open -- 'office open <path>' first")?;
    let n = ANSWERS.lock_irq().len();
    if n == 0 {
        return Err("nothing to report yet -- ask a question about the sheet first".into());
    }
    let name = String::from(book.rsplit('/').next().unwrap_or(&book));
    let stem = String::from(name.split('.').next().unwrap_or("workbook"));
    let facts: Vec<String> = ANSWERS.lock_irq().iter().flat_map(|a| findings(a)).collect();
    note(format!("report  {} answer(s), {} finding(s), from {}", n, facts.len(), name));
    note(String::from("drafting the summary..."));

    let mut ask = String::from("Findings from a company's sales ledger, each computed by the spreadsheet:\n");
    for f in &facts {
        ask.push_str("- ");
        ask.push_str(f);
        ask.push('\n');
    }
    ask.push_str(
        "\nWrite a short executive summary of these findings for a manager: three or four sentences, one paragraph. \
Use only the figures given above, written exactly as they are given. Do not invent any other numbers. \
The figures have no currency: do not add a currency sign.",
    );
    let (summary, dropped) = match super::draft_at(super::SYSTEM, &ask, 160, super::Sink::Screen, 0.1) {
        Ok(t) => vet(&t, &facts),
        Err(why) => {
            note(format!("no summary: {}", why));
            (Vec::new(), 0)
        }
    };
    if dropped > 0 {
        note(format!("dropped {} sentence(s) stating a figure the sheet did not compute", dropped));
    }

    let title = format!("{}: findings", stem[..1].to_uppercase() + &stem[1..]);
    let sub = format!(
        "{} rows of {}, prepared by GLaDOS Colleague. Every figure was computed by LibreOffice Calc from the workbook.",
        figure(rows as f64),
        name
    );
    let body = {
        let answers = ANSWERS.lock_irq();
        fodt(&title, &sub, &summary, dropped, &answers)
    };
    let file = format!("{}/{}-report.fodt", OUT, stem);
    crate::linux::ns::write_blob(&file, body.into_bytes());
    // Opened by the LibreOffice already running, through its watcher: a
    // second soffice cannot hand it a file in this guest (see `Bench.Watch`).
    crate::linux::ns::write_blob(&format!("{}/open.job", BENCH_OUT), format!("{}\n", file).into_bytes());
    let deadline = crate::dev::lapic::ticks() + 120 * crate::TIMER_HZ as u64;
    while crate::gfx::desk::window_titled("LibreOffice Writer").is_none() {
        if crate::dev::lapic::ticks() > deadline {
            return Err(format!("Writer did not open {} within two minutes", file));
        }
        crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64 / 4);
    }
    crate::task::sleep_until(crate::dev::lapic::ticks() + crate::TIMER_HZ as u64);
    crate::gfx::desk::focus_titled("LibreOffice Writer");
    note(format!("opened {} in Writer", file));
    Ok(())
}

/// What `diag office` asks of this, with no model and no LibreOffice.
pub fn checks() -> Vec<(&'static str, bool)> {
    let a = Answer {
        question: String::from("Revenue by region?"),
        heads: (String::from("Region"), String::from("Total Revenue")),
        rows: alloc::vec![
            (String::from("East"), 9542.0, String::from("9,542")),
            (String::from("North"), 6879.0, String::from("6,879")),
            (String::from("West"), 10080.0, String::from("10,080")),
        ],
        word: String::from("total"),
        formula: String::new(),
        cells: String::new(),
    };
    let f = findings(&a);
    let mut out = Vec::new();
    out.push(("a report leads with the largest group and its share", f.first().is_some_and(|s| s.starts_with("West leads") && s.contains("10,080") && s.contains("26,501") && s.contains("38.0%"))));
    out.push(("and names the smallest", f.get(1).is_some_and(|s| s.starts_with("North is last"))));
    out.push(("a figure reads the same with or without its commas", numbers("revenue of 10,080 and 38.0%") == ["10080", "380"]));
    let (kept, dropped) = vet("West leads with 10,080. Sales grew 12% this year. North trails.", &f);
    out.push(("a summary sentence with a figure nobody computed is dropped", kept.len() == 2 && dropped == 1));
    out.push(("a decimal point does not end a sentence", vet("East averaged 731.14 a sale.", &[String::from("731.14")]).0.len() == 1));
    out.push(("asking for a report is told from asking a question", asks_for("write a report on this") && !asks_for("total revenue by region")));
    out
}
