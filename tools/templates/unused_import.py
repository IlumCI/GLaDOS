"""Unused imports, as the compiler itself names them.

**The space is enumerated by rustc, not guessed at.** `cargo check
--message-format=json` reports every `unused_imports` diagnostic with an
exact file, line and column span, so a candidate here is a deletion of a
line the compiler has already declared dead. That is the same argument
`forest.py` makes about its arithmetic checker and `knob.py` about its
table: derived from the data, never hand-assigned.

**Why this family can win**, which is the bar for being in `templates` at
all: it removes a warning, so `cost.warnings` moves down; it removes a line
of code, so `cost.image_bytes` cannot move up; and it changes no behaviour,
so every other rail should read `same`. The `cleanup` kind asks for
deletions to dominate, and a pure deletion satisfies that by construction.

**Three refusals worth stating**, because each is a way a mechanical
deletion stops being safe:

- A span naming less than the whole import is left alone. `use a::{b, c}`
  with only `c` unused wants the brace list rewritten, and a template that
  rewrote one would be authoring rather than deleting. rustc emits those as
  a narrower span, so the check is the span against the path the statement
  declares -- see `_covers_whole_line`, which had this backwards and is the
  reason the family found nothing for two runs.

  It earns itself on this tree rather than in principle: of ten spans, four
  are one name out of `pub use futures::{project, Projection, Branch,
  snapshot}` and one is `Fired` out of `use super::thing::{Fired, Objs};`.
  Deleting that last line whole removes `Objs`, which IS used, so the
  refusal is the difference between a cleanup and a build failure.
- A line under `#[cfg(...)]` is left alone. The import is dead in this
  configuration and may be the only one keeping another alive elsewhere,
  and CI builds one target.
- Anything outside `src/` is left alone, and the evaluator paths are
  refused again at `godel.py admit` regardless.
"""

import io
import json
import os
import subprocess

#: rustc's own name for the lint, so a rename upstream is a family that
#: finds nothing rather than a family that deletes the wrong lines.
LINT = "unused_imports"

#: What this family claims to move, and the kind whose judge reads it.
#: Declared rather than written at each emit, so the package's selftest can
#: ask whether the rail is one `rails.py` actually counts -- a family
#: claiming a rail nobody reads spends nights on a comparison that cannot
#: come back better.
RAIL = "cost.warnings"
KIND = "cleanup"


def diagnostics(root):
    """Every unused-import span the compiler reports, as
    (path, line, col_start, col_end, text, message)."""
    cargo = os.environ.get("CARGO", "cargo")
    # **`check`, into a target directory of its own.** `check` because the
    # lint is the same and codegen buys nothing to produce it; a private
    # directory because `cost.image_bytes` is read off a release build, and
    # a lint run sharing that target dir churns the fingerprints the judged
    # build reads -- the enumerate would be perturbing the artifact the
    # judge measures.
    #
    # It is deliberately NOT here to defeat a warm cache, which is what the
    # first version of this comment claimed. That was a guess standing in
    # for a measurement, and it was wrong: cargo replays stored diagnostics
    # for a unit it did not rebuild, measured at 14 s cold and 0 s warm,
    # five candidates both times. The run that found nothing and started
    # the hunt was a wrong span check two functions down, not a cache --
    # and a comment blaming the cache would have sent the next reader at
    # the build system instead of at the bug.
    r = subprocess.run(
        [cargo, "check", "--locked", "--message-format=json",
         "--target-dir", "target/lint"],
        cwd=root, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") != "compiler-message":
            continue
        d = m.get("message") or {}
        if ((d.get("code") or {}).get("code")) != LINT:
            continue
        for sp in d.get("spans") or []:
            if not sp.get("is_primary"):
                continue
            texts = sp.get("text") or []
            if not texts or sp["line_start"] != sp["line_end"]:
                continue
            out.append((
                sp["file_name"].replace(os.sep, "/"),
                sp["line_start"],
                sp["column_start"],
                sp["column_end"],
                texts[0]["text"],
                d.get("message", "unused import"),
            ))
    # A fixed order, so two runs over one tree enumerate identically.
    out.sort()
    return out


def _covers_whole_line(text, col_start, col_end):
    """Is the span the whole import, leaving nothing behind to delete?

    A narrower span is rustc saying "part of this import is unused", which
    wants the brace list rewritten -- and rewriting is authoring.

    **The span is the path, never the statement**, and assuming otherwise
    is why this family found nothing on its first two runs. rustc reports
    `    use crate::kprintln;` at columns 9..24, which is `crate::kprintln`
    alone: no `use`, no semicolon, no indent. A check written against the
    line's own extent therefore refused every candidate there is, while
    looking exactly like a tree with no dead imports in it. So the
    comparison is against the path the statement declares -- equal means
    the whole import is dead and the line may go; shorter means one name
    out of a list, which is the case to leave alone.
    """
    stripped = text.strip()
    if not stripped.startswith("use ") or not stripped.endswith(";"):
        return False
    # `pub use` is a re-export and is deliberately not matched: what makes
    # it unused is that nothing INSIDE the crate reads it, which is not
    # the same question as whether it may be removed.
    body = stripped[len("use "):-1].strip()
    return text[col_start - 1:col_end - 1] == body


def _diff_deleting(path, lineno, lines):
    """A unified diff removing exactly `lineno` (1-based) from `path`."""
    i = lineno - 1
    before = lines[max(0, i - 3):i]
    after = lines[i + 1:i + 4]
    start = max(0, i - 3) + 1
    n_old = len(before) + 1 + len(after)
    n_new = len(before) + len(after)
    body = "".join(f" {l}\n" for l in before)
    body += f"-{lines[i]}\n"
    body += "".join(f" {l}\n" for l in after)
    return (f"--- a/{path}\n+++ b/{path}\n"
            f"@@ -{start},{n_old} +{start},{n_new} @@\n{body}")


def emit(root):
    """Candidates, newest-safe-first. Each is a dict the loop can file."""
    out = []
    for path, lineno, c0, c1, text, msg in diagnostics(root):
        if not path.startswith("src/"):
            continue
        full = os.path.join(root, path)
        if not os.path.isfile(full):
            continue
        if not _covers_whole_line(text, c0, c1):
            continue
        body = io.open(full, encoding="utf-8").read()
        lines = body.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if lineno > len(lines) or lines[lineno - 1] != text.rstrip("\n"):
            # The tree moved under the diagnostics. Refuse rather than
            # delete by line number, which is how a stale span eats a
            # line that means something else now.
            continue
        # A cfg-gated import is dead in THIS configuration only.
        window = "\n".join(lines[max(0, lineno - 3):lineno])
        if "#[cfg(" in window:
            continue
        out.append({
            "kind": KIND,
            "rail": RAIL,
            "why": f"{msg} at {path}:{lineno}",
            "patch": _diff_deleting(path, lineno, lines),
        })
    return out


#: Spans in exactly the shape rustc hands over, lifted from a real run on
#: this tree rather than invented -- (text, col_start, col_end, admit?).
#: The two `False` rows are the ones that would be build failures: deleting
#: `use super::thing::{Fired, Objs};` whole removes `Objs`, which is used.
SPANS = (
    ("    use crate::kprintln;", 9, 24, True),
    ("use alloc::vec::Vec;", 5, 20, True),
    ("    use alloc::vec;", 9, 19, True),
    ("use super::thing::{Fired, Objs};", 20, 25, False),
    ("pub use futures::{project, Projection, Branch, snapshot};", 19, 26, False),
    ("pub use futures::{project, Projection, Branch, snapshot};", 1, 57, False),
    ("    let x = 1;", 5, 14, False),
)


def selftest(claim):
    """Claims about what may be deleted. `claim(what, good)` is the caller's,
    in godel.py's own argument order so these land in its suite rather than
    in a gate of their own."""
    got = tuple(_covers_whole_line(t, a, b) for t, a, b, _w in SPANS)
    want = tuple(w for *_x, w in SPANS)
    claim("every span rustc really emits is judged as measured", got == want)

    # The canary. A check that accepted everything would pass every
    # positive claim above, and accepting everything is exactly the shape
    # this family failed in -- so the suite asserts the refusals are
    # load-bearing rather than incidental: three of seven, and which three.
    claim("and a name out of a brace list is refused, or a cleanup is a "
          "build failure", sum(want) == 3 and not want[3])

    # Whole-line `pub use` is refused even where the span DOES cover it,
    # the one row where the rule is about meaning rather than shape.
    claim("a re-export is left alone however wide its span",
          not _covers_whole_line(SPANS[5][0], 1, 57))

    lines = ["a", "b", "c", "use dead::thing;", "d", "e", "f"]
    d = _diff_deleting("src/x.rs", 4, lines)
    head_line = [l for l in d.split("\n") if l.startswith("@@")][0]
    body = d.split("@@\n", 1)[1].split("\n")
    ctx = sum(1 for l in body if l.startswith(" "))
    gone = sum(1 for l in body if l.startswith("-"))
    # The header read back by a second reader and required to agree with
    # the body it describes: a count one out is a patch `git apply`
    # refuses, which on a runner is a spent job and no verdict.
    span = head_line.split(" ")[1:3]
    n_old = int(span[0].split(",")[1])
    n_new = int(span[1].split(",")[1])
    claim("a deletion diff's hunk header agrees with the body it describes",
          n_old == ctx + gone and n_new == ctx and gone == 1)
    claim("and the line it removes is the dead one",
          "-use dead::thing;" in body)
