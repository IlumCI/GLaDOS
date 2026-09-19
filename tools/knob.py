#!/usr/bin/env python3
"""The half of a source proposal that owns the files.

The kernel has no copy of its own source and cannot compile, so what it can
say exactly is *which symbol in which file takes which value*. This finds the
line, checks it still says what the proposal thinks it says, and changes it.

The same division `repair.rs` makes and for the same reason: nothing the
proposal contains is executed, parsed as a command, or used as a path. Four
declared fields are read out of it and everything else on the line is ignored,
because the file it came from is a blob a machine wrote and not a script.

Usage:
    knob.py check                  every KNOBS row against the real source
    knob.py show PATCH             what a patch would do, without doing it
    knob.py apply PATCH            do it
    knob.py revert PATCH           put it back
    knob.py --selftest

`check` is the one CI should run. Nothing in the kernel can compare its table
against the source, so a row that has gone stale would produce a patch that
applies to nothing -- the marker written, the point never measured, and the
loop quietly spending nights on a constant that moved.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "src", "ai", "knob.rs")

REQUIRED = ("file", "symbol", "from", "to", "rail")


def read_patch(path):
    """The four fields, and a refusal for anything that is not all four.

    A patch missing `from` is the dangerous one: without it there is nothing
    to check the source against, so a row that has drifted would be silently
    overwritten with a value chosen for a number that is no longer there.
    """
    got = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
        if first != "knob 1":
            raise SystemExit(f"  {path} is not a knob patch (first line {first!r})")
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, _, v = line.partition(" ")
            if k in REQUIRED:
                got[k] = v
    missing = [k for k in REQUIRED if k not in got]
    if missing:
        raise SystemExit(f"  {path} is missing {', '.join(missing)}")
    # A path is the one field that could reach outside the tree, so it is
    # checked as a path rather than trusted as a string.
    p = os.path.normpath(os.path.join(ROOT, got["file"]))
    if not p.startswith(ROOT + os.sep):
        raise SystemExit(f"  {got['file']} is outside the repository")
    got["_abs"] = p
    return got


def find_const(text, symbol):
    """(line index, the value as written), or None.

    Matches a `const` declaration and nothing else. A `let` or a use of the
    same name is not the thing a knob is about, and a regex loose enough to
    catch one would be loose enough to rewrite the wrong line.
    """
    pat = re.compile(
        r"^(\s*(?:pub\s+)?const\s+" + re.escape(symbol) + r"\s*:\s*[^=]+=\s*)([^;]+)(;.*)$"
    )
    for i, line in enumerate(text.splitlines()):
        m = pat.match(line)
        if m:
            return i, m.group(2).strip(), m
    return None


def rewrite(path, symbol, want_from, want_to):
    """Answers (ok, message). Never writes unless the current value matches."""
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    found = find_const(text, symbol)
    if not found:
        return False, f"no `const {symbol}` in {os.path.relpath(path, ROOT)}"
    i, current, m = found
    if current != want_from:
        return False, (
            f"{symbol} says {current!r}, the patch expected {want_from!r} -- "
            "the table has gone stale, and applying this would change a "
            "constant somebody has already moved"
        )
    lines = text.splitlines(keepends=True)
    old = lines[i]
    end = ""
    while old.endswith("\n") or old.endswith("\r"):
        end = old[-1] + end
        old = old[:-1]
    m2 = re.match(
        r"^(\s*(?:pub\s+)?const\s+" + re.escape(symbol) + r"\s*:\s*[^=]+=\s*)([^;]+)(;.*)$",
        old,
    )
    lines[i] = m2.group(1) + want_to + m2.group(3) + end
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("".join(lines))
    return True, f"{os.path.relpath(path, ROOT)}: {symbol} {want_from} -> {want_to}"


def table_rows():
    """Every KNOBS row, read out of the Rust source that declares it.

    Parsed rather than duplicated here. A second copy of the table in Python
    would agree on the day it was written and then drift, and the one that
    drifts is the one nobody reads -- which is precisely the failure `check`
    exists to catch one level up.
    """
    with open(TABLE, encoding="utf-8") as f:
        text = f.read()
    start = text.index("pub const KNOBS:")
    body = text[start:]
    out = []
    for block in re.finditer(r"Knob\s*\{(.*?)\n    \},", body, re.S):
        b = block.group(1)

        def field(name):
            m = re.search(name + r':\s*"((?:[^"\\]|\\.)*)"', b)
            return m.group(1) if m else None

        vals = re.search(r"values:\s*&\[(.*?)\]", b, re.S)
        values = re.findall(r'"((?:[^"\\]|\\.)*)"', vals.group(1)) if vals else []
        out.append({
            "file": field("file"),
            "symbol": field("symbol"),
            "now": field("now"),
            "rail": field("rail"),
            "values": values,
        })
    return out


def check():
    """Every row against the file it names. This is what CI runs."""
    rows = table_rows()
    if not rows:
        print("  no rows parsed out of knob.rs -- the table's shape changed")
        return False
    ok = True
    for r in rows:
        path = os.path.join(ROOT, r["file"])
        if not os.path.exists(path):
            print(f"  FAIL  {r['file']} does not exist")
            ok = False
            continue
        with open(path, encoding="utf-8") as f:
            found = find_const(f.read(), r["symbol"])
        if not found:
            print(f"  FAIL  no `const {r['symbol']}` in {r['file']}")
            ok = False
            continue
        _, current, _ = found
        if current != r["now"]:
            print(
                f"  FAIL  {r['file']} {r['symbol']} says {current!r}, "
                f"the table says {r['now']!r}"
            )
            ok = False
            continue
        if r["now"] in r["values"]:
            print(f"  FAIL  {r['symbol']} offers the value it already has")
            ok = False
            continue
        print(f"  ok    {r['file']} {r['symbol']} = {current}  -> {', '.join(r['values'])}")
    return ok


def selftest():
    ok = True

    def claim(good, what):
        nonlocal ok
        if not good:
            ok = False
        print(f"  {'ok ' if good else 'FAIL'}  {what}")

    src = (
        "// a comment mentioning LEN_B = 9.9 which is not a declaration\n"
        "pub const LEN_B: f32 = 0.5;\n"
        "const OTHER: usize = 3;\n"
        "fn f() { let LEN_B = 7; }\n"
    )
    found = find_const(src, "LEN_B")
    claim(found is not None and found[1] == "0.5", "a const declaration is found and read")
    claim(found[0] == 1, "and it is the declaration rather than the comment above it")
    claim(find_const(src, "NOPE") is None, "a symbol that is not declared is not invented")

    # The one that matters most. A `let` of the same name is not the constant,
    # and a pattern loose enough to match it would rewrite a function body.
    only_let = "fn f() { let LEN_B = 7; }\n"
    claim(find_const(only_let, "LEN_B") is None, "a `let` of the same name is not a const")

    # Types with a `=` inside them, which a naive split on `=` would break on.
    arr = "pub const G: [u8; 2] = [1, 2];\n"
    claim(find_const(arr, "G")[1] == "[1, 2]", "a value that is not a scalar reads whole")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.rs")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        good, why = rewrite(p, "LEN_B", "0.5", "0.75")
        claim(good, "a value that matches is rewritten")
        with open(p, encoding="utf-8") as f:
            after = f.read()
        claim("pub const LEN_B: f32 = 0.75;" in after, "and the new value is what is there")
        claim(after.count("\n") == src.count("\n"), "and nothing else about the file moved")

        # **The refusal that keeps a stale table from silently overwriting.**
        good, why = rewrite(p, "LEN_B", "0.5", "0.25")
        claim(not good and "stale" in why,
              "a value that has moved since the table was written is refused, and says why")

        good, why = rewrite(p, "NOPE", "1", "2")
        claim(not good, "and a symbol that is not there is refused")

    return ok


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("--selftest", "selftest"):
        print("[knob] finding and changing a declared constant")
        return 0 if selftest() else 1

    cmd = argv[0]
    if cmd == "check":
        print("[knob] every declared row against the source it names")
        return 0 if check() else 1

    if cmd in ("show", "apply", "revert"):
        if len(argv) < 2:
            raise SystemExit(f"  usage: knob.py {cmd} PATCH")
        pat = read_patch(argv[1])
        a, b = (pat["from"], pat["to"]) if cmd != "revert" else (pat["to"], pat["from"])
        if cmd == "show":
            with open(pat["_abs"], encoding="utf-8") as f:
                found = find_const(f.read(), pat["symbol"])
            here = found[1] if found else "(not found)"
            print(f"  {pat['file']} {pat['symbol']}")
            print(f"    now      {here}")
            print(f"    expected {pat['from']}")
            print(f"    would be {pat['to']}")
            print(f"    claims to move {pat['rail']}")
            return 0 if found and found[1] == pat["from"] else 1
        good, why = rewrite(pat["_abs"], pat["symbol"], a, b)
        print(("  " if good else "  refused: ") + why)
        return 0 if good else 1

    raise SystemExit(f"  unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main())
