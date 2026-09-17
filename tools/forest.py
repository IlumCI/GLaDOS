#!/usr/bin/env python3
"""Build the first forest tree -- logic and mathematics -- from local data.

A forest node is a *structured concept node*: a head line the router pools
over, and a body carrying the concept, a method, machine-checkable steps, and
the original passage. The head is what stays resident; the body is what reaches
the model's context when a node wins retrieval.

**Mathematics first because a maths node can check itself.** GSM8K writes its
solutions with the arithmetic inline --

    Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.

-- so every `<<expr=value>>` is an executable expression carrying its own
expected result. This tool evaluates all of them and **refuses a node whose own
arithmetic does not check out**, which is `problem.rs`'s admission discipline
applied to knowledge rather than to programs. No model is involved and none can
be: a corpus that needed a model to decide what belongs in it would inherit
whatever that model was wrong about.

Output is a directory tree, because `tools/mkpkg.py` already walks one and
`GLADOSPK` already grafts nested paths into the namespace with `..` and
absolute paths refused (`src/pkg.rs:133-135`). There is no new bundle format
here and there should not be one.

Two shapes the storage layer wants, and this obeys both. `tree::put` inserts at
a sorted index, so names emitted in **ascending order** append instead of
memmoving O(N^2); and `cmd_ls` hashes every child of a directory, so **fanout is
capped** and depth is spent instead -- `MAX_DEPTH` is 32 and depth is free.

    python tools/forest.py out/forest --limit 400
    python tools/forest.py out/forest --verify

`--verify` re-reads the emitted tree with a parser written separately from the
writer, the bargain `dataset.py --blobs` and `v4.py` both make: a node file is
opaque text on both sides, and the first moment a field error would otherwise
surface is as a corpus quietly full of shifted content.
"""

import argparse
import re
import sys
from pathlib import Path

# These three are lifted from `tools/lm_eval.py`, which is where they belong
# and where they are maintained. They are not imported, because importing that
# module pulls in numpy, `ref35`, `v4`, `evaluate` and `reference` -- the whole
# model stack -- and this tool reads parquet and writes text. A corpus builder
# that cannot run without a checkpoint is a corpus builder nobody runs.
HF_CACHE = Path.home() / ".cache/huggingface/hub"


def snapshot_dir(name):
    base = HF_CACHE / name / "snapshots"
    if not base.is_dir():
        raise SystemExit(f"dataset not cached: {name}")
    snaps = sorted(p for p in base.iterdir() if p.is_dir())
    if not snaps:
        raise SystemExit(f"dataset not cached: {name}")
    return snaps[0]


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path)).to_pylist()


def find_file(dirpath, prefix):
    hits = sorted(Path(dirpath).rglob(f"{prefix}*.parquet"))
    if not hits:
        raise SystemExit(f"no {prefix} parquet under {dirpath}")
    return hits[0]


# --- the node format --------------------------------------------------------
#
# Line oriented, `key<space>value`. A line whose first word is a known key
# starts that field; anything else continues the field before it. That makes
# the kernel-side parser a match on the first word and nothing else, and it
# means a body containing a colon, a bracket or a blank line needs no escaping.
#
# `head` is first and is the whole of what the index holds, so a reader that
# wants only the index stops after one line.

KEYS = ("head", "kind", "source", "concept", "method", "check", "answer", "text")

# `check` lines are the machine-verifiable part: `<expr> = <value>`.
CHECK_RE = re.compile(r"<<([^<>=]+)=([^<>]*)>>")
FINAL_RE = re.compile(r"####\s*(.+?)\s*$", re.M)

# Fanout cap. The storage layer punishes width twice -- `Vec::insert` on `put`
# and a `content_hash` per child on `ls` -- and rewards depth, which is free to
# 32 levels. 200 is comfortably inside "low hundreds".
FANOUT = 200

# Five digits rather than four. `vocab::record` pads to four and then *wraps*
# at 10000, overwriting `0000` forever with no error; `dataset.py` refuses past
# 9999 for exactly that reason. Nothing here shares that naming, but the width
# is chosen deliberately rather than inherited.
NAME_WIDTH = 5
NAME_MAX = 10 ** NAME_WIDTH


class Refused(Exception):
    """A node that does not survive its own checks."""


# --- a safe arithmetic evaluator -------------------------------------------
#
# `eval` is not an option: these strings come out of a dataset, and a corpus
# builder that executes them is a corpus builder that runs whatever the dataset
# says. Recursive descent over + - * / ( ) and decimal literals, and nothing
# else -- no names, no calls, no attributes, so there is nothing to escape from.

# `//` before `.`, or it lexes as two divisions and `560//10` becomes
# `(560/10)/nothing`. GSM8K writes floor division that way and two nodes in the
# training split were refused for it before this was here.
TOKEN_RE = re.compile(r"\s*(?:(\d+\.?\d*|\.\d+)|(//)|(.))")


def tokenize(s):
    out, i = [], 0
    while i < len(s):
        m = TOKEN_RE.match(s, i)
        if not m or m.end() == i:
            raise Refused(f"cannot tokenise {s!r}")
        i = m.end()
        if m.group(1) is not None:
            out.append(("num", float(m.group(1))))
        elif m.group(2):
            out.append(("op", "//"))
        elif m.group(3).strip():
            out.append(("op", m.group(3)))
    return out


def arith(s):
    """Evaluate a arithmetic expression, or raise `Refused`."""
    toks = tokenize(s)
    pos = 0

    def peek():
        return toks[pos][1] if pos < len(toks) else None

    def expr():
        nonlocal pos
        v = term()
        while peek() in ("+", "-"):
            op = toks[pos][1]
            pos += 1
            r = term()
            v = v + r if op == "+" else v - r
        return v

    def term():
        nonlocal pos
        v = unary()
        while peek() in ("*", "/", "//"):
            op = toks[pos][1]
            pos += 1
            r = unary()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise Refused("division by zero")
                # Floor, genuinely. Treating `//` as `/` would pass `560//10`
                # by luck (it divides exactly) and silently accept a wrong
                # value for `7//2`, which is the shape of error this whole
                # checker exists to catch.
                v = v // r if op == "//" else v / r
        return v

    def unary():
        nonlocal pos
        if peek() == "-":
            pos += 1
            return -unary()
        if peek() == "+":
            pos += 1
            return unary()
        return atom()

    def atom():
        nonlocal pos
        if pos >= len(toks):
            raise Refused("expression ends early")
        kind, v = toks[pos]
        if kind == "num":
            pos += 1
            return v
        if v == "(":
            pos += 1
            inner = expr()
            if peek() != ")":
                raise Refused("unclosed bracket")
            pos += 1
            return inner
        raise Refused(f"unexpected {v!r}")

    value = expr()
    if pos != len(toks):
        raise Refused("trailing tokens")
    return value


def num(s):
    """A dataset number: commas as thousands separators, a stray currency mark."""
    t = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not t:
        raise Refused("empty number")
    try:
        return float(t)
    except ValueError as e:
        raise Refused(f"not a number: {s!r}") from e


def close(a, b):
    # The dataset rounds its own intermediates, so exact equality refuses
    # perfectly good nodes. A part in a million is far tighter than any
    # rounding in the source and far looser than float noise.
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


# --- building nodes ---------------------------------------------------------


def gsm8k_node(row):
    """One worked problem, with its arithmetic chain checked.

    Raises `Refused` when the solution's own annotations disagree with the
    arithmetic they claim, which is the whole reason this tree went first.
    """
    q, a = row["question"].strip(), row["answer"].strip()

    checks = CHECK_RE.findall(a)
    if not checks:
        raise Refused("no arithmetic to check")

    verified = []
    for expr, want in checks:
        got = arith(expr)
        if not close(got, num(want)):
            raise Refused(f"{expr} = {want}, but it is {got:g}")
        verified.append((expr.strip(), want.strip()))

    final = FINAL_RE.search(a)
    if not final:
        raise Refused("no final answer")

    ops = {c for expr, _ in verified for c in expr if c in "+-*/"}
    prose = FINAL_RE.sub("", a).strip()

    return {
        "ops": ops,
        "steps": len(verified),
        "kind": "gsm8k",
        "source": "gsm8k/train",
        # The *question*, not the solution. Taking it from the solution put a
        # fragment of the working in the head -- "3/5 x 100 = 60 tennis bal" --
        # which is the one line the router pools over, so it was steering
        # retrieval by whatever arithmetic happened to appear first.
        "concept": first_sentence(q),
        "method": " ; ".join(f"{e} = {w}" for e, w in verified),
        "checks": verified,
        "answer": final.group(1).strip(),
        "text": q + "\n" + prose,
        "terms": terms_of(q),
    }


# --- which splits may become nodes -----------------------------------------
#
# **A forest built from a test split is an answer key, and this one was.**
# The loop below walked `("dev", "validation", "test")`, so 1,353 MMLU *test*
# questions went into the forest carrying their `answer` field. The plan for
# retrieval says in as many words to measure `--task mmlu --forest`, and that
# experiment would have reported a large improvement produced entirely by
# handing the model the answer to the question it was being scored on.
#
# Nothing was wrong with the ingest when it was written: the forest was a
# knowledge base and MMLU was not yet a rail it would be measured against. It
# became wrong the moment both were true, which is the same shape as the test
# set that *moved* when the routing corpus was appended to -- recorded in
# `CLAUDE.md` as one of the three ways measurement was got wrong here before.
#
# So `test` is off by default and `--allow-test` is the deliberate exception,
# for a forest that is never going to be retrieved into during an MMLU run.
# `--report` prints the split composition of whatever was built, because the
# number that matters is one nobody thought to look at.
SPLITS = ("dev", "validation")


def mmlu_node(row, subject, split):
    letters = "ABCD"
    ch = list(row["choices"])
    idx = int(row["answer"])
    if not 0 <= idx < len(ch):
        raise Refused("answer index outside the choices")
    q = row["question"].strip()
    body = q + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ch[:4]))
    return {
        "kind": "mmlu",
        "source": f"mmlu/{subject}/{split}",
        "concept": first_sentence(q),
        "method": "",
        "checks": [],
        "answer": letters[idx],
        "text": body,
        "terms": terms_of(q),
    }


STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "that", "this", "it", "as", "be", "from",
    "how", "what", "which", "if", "then", "each", "many", "much", "does", "do",
}


def terms_of(text, n=8):
    """Key terms for the head line: the longest distinctive words, in order.

    Crude on purpose. The head is pooled into an embedding, so what matters is
    that it carries the subject's vocabulary -- not that it is a good summary.
    Anything cleverer would need a model, and see the module doc for why not.
    """
    seen, out = set(), []
    for w in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower()):
        if w in STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
    out.sort(key=len, reverse=True)
    return out[:n]


def first_sentence(text, cap=120):
    s = " ".join(text.split())
    m = re.search(r"^(.{20,%d}?[.?!])\s" % cap, s)
    if m:
        return m.group(1).strip()
    # No sentence end within the cap. Cut at a word boundary rather than mid
    # word: the head is pooled into an embedding, and half a word tokenises as
    # something the vocabulary never saw in training.
    if len(s) <= cap:
        return s.strip()
    cut = s[:cap].rsplit(" ", 1)[0]
    return (cut or s[:cap]).strip()


def bucket(node):
    """Where a node lives. Derived from the data, never hand-assigned.

    GSM8K branches by which operators the solution actually uses and by how
    many steps it takes -- both read off the verified chain, so the taxonomy is
    a fact about the node rather than a guess about it. Step count is a
    difficulty axis, the same shape `redqueen`'s archive bands use.
    """
    if node["kind"] == "mmlu":
        return ["logic-and-maths", node["source"].split("/")[1].replace("_", "-")]
    ops = node["ops"]
    if ops <= {"+", "-"}:
        family = "add-sub"
    elif ops <= {"*", "/"}:
        family = "mul-div"
    else:
        family = "mixed"
    steps = node["steps"]
    band = "1-step" if steps <= 1 else "2-step" if steps == 2 else "3-plus-step"
    return ["logic-and-maths", "arithmetic", family, band]


def render(node, path):
    """The node's text. `head` first, because the index stops after one line."""
    head = f"{'/'.join(path[:-1])} | {node['concept']} | {' '.join(node['terms'])}"
    lines = [f"head {one_line(head)}", f"kind {node['kind']}", f"source {node['source']}"]
    lines.append(f"concept {one_line(node['concept'])}")
    if node["method"]:
        lines.append(f"method {one_line(node['method'])}")
    for expr, want in node["checks"]:
        lines.append(f"check {expr} = {want}")
    lines.append(f"answer {one_line(node['answer'])}")
    lines.append("text " + node["text"].replace("\r", ""))
    return "\n".join(lines) + "\n"


def one_line(s):
    return " ".join(str(s).split())


def parse(text):
    """Read a node back. Deliberately not `render`'s inverse by construction.

    A line whose first word is a key starts that field; everything else
    continues the one before it. Written from the format description rather
    than from `render`, so a disagreement between them shows up here instead of
    as a corpus of shifted fields.
    """
    fields, order = {}, []
    cur = None
    for line in text.splitlines():
        first = line.split(" ", 1)[0] if line else ""
        if first in KEYS:
            cur = first
            rest = line[len(first):].lstrip(" ")
            if cur == "check":
                fields.setdefault("check", []).append(rest)
            else:
                fields[cur] = rest
                order.append(cur)
        elif cur is not None:
            if cur == "check":
                fields["check"][-1] += "\n" + line
            else:
                fields[cur] = fields.get(cur, "") + "\n" + line
    return fields, order


# --- emit -------------------------------------------------------------------


def emit(outdir, nodes, quiet=False):
    """Write the tree. Ascending name order, capped fanout.

    **A rebuild has to be able to make the tree smaller.** Names are ordinal
    per directory -- `00000`, `00001` -- so a build admitting fewer nodes than
    the last one writes over the low names and leaves the high ones exactly
    where they were. Dropping MMLU's test split takes 1,353 nodes out of the
    admitted set and would have left all 1,353 on disk: still readable, still
    carrying their answers, under a tree that now reports itself clean. That
    is the answer key this guard exists to remove, surviving its removal.

    So the sweep below deletes node files this build did not write, and says
    how many. It touches only files whose first word is `head`, because that
    is the one thing every node has and a stranger's file does not. Anything
    else under `outdir` is left alone and counted out loud.
    """
    groups = {}
    for n in nodes:
        groups.setdefault(tuple(bucket(n)), []).append(n)

    written, dirs, keep = 0, 0, set()
    for key in sorted(groups):
        items = groups[key]
        # Spill into numbered sub-buckets rather than widening a directory.
        chunks = [items[i:i + FANOUT] for i in range(0, len(items), FANOUT)]
        for ci, chunk in enumerate(chunks):
            parts = list(key) + ([f"part-{ci:02d}"] if len(chunks) > 1 else [])
            d = outdir.joinpath(*parts)
            d.mkdir(parents=True, exist_ok=True)
            dirs += 1
            if len(chunk) >= NAME_MAX:
                raise SystemExit(
                    f"  {len(chunk)} nodes exceeds the {NAME_WIDTH}-digit name width")
            for i, n in enumerate(chunk):
                name = str(i).zfill(NAME_WIDTH)
                path = parts + [name]
                (d / name).write_text(render(n, path), encoding="utf-8", newline="\n")
                keep.add((d / name).resolve())
                written += 1

    stale, foreign = 0, 0
    for p in outdir.rglob("*"):
        if not p.is_file() or p.resolve() in keep:
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                first = f.readline()
        except (OSError, UnicodeDecodeError):
            first = ""
        if first.startswith("head "):
            p.unlink()
            stale += 1
        else:
            foreign += 1
    for d in sorted((p for p in outdir.rglob("*") if p.is_dir()),
                    key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()

    if not quiet:
        print(f"  {written} node(s) in {dirs} director(ies), fanout <= {FANOUT}")
        if stale:
            print(f"  {stale} node(s) from an earlier build removed")
        if foreign:
            print(f"  {foreign} file(s) under {outdir} are not nodes, left alone")
    return written, dirs


def verify(outdir):
    """Re-read every node with `parse` and check the invariants hold."""
    files = sorted(p for p in outdir.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"  nothing under {outdir}")

    bad, checked, widest = 0, 0, 0
    for p in files:
        fields, order = parse(p.read_text(encoding="utf-8"))
        if order[:1] != ["head"]:
            print(f"  {p}: head is not the first field")
            bad += 1
            continue
        for k in ("kind", "source", "answer", "text"):
            if k not in fields:
                print(f"  {p}: no {k}")
                bad += 1
        # The self-check: every recorded step must still evaluate to its value.
        for c in fields.get("check", []):
            expr, _, want = c.partition("=")
            try:
                if not close(arith(expr), num(want)):
                    print(f"  {p}: {c.strip()} does not hold")
                    bad += 1
                checked += 1
            except Refused as e:
                print(f"  {p}: {e}")
                bad += 1

    for d in {p.parent for p in files}:
        widest = max(widest, sum(1 for _ in d.iterdir()))

    print(f"  {len(files)} node(s) read back, {checked} arithmetic step(s) re-checked")
    print(f"  widest directory {widest} entr(ies), cap {FANOUT}")
    if bad:
        raise SystemExit(f"  {bad} problem(s)")
    print("  verified")


def collect(limit, splits=SPLITS):
    nodes, refused = [], {}

    def refuse(e):
        key = str(e).split(",")[0][:40]
        refused[key] = refused.get(key, 0) + 1

    g = find_file(snapshot_dir("datasets--gsm8k"), "train")
    rows = parquet_rows(g)
    for row in rows[: limit or len(rows)]:
        try:
            nodes.append(gsm8k_node(row))
        except Refused as e:
            refuse(e)

    m = snapshot_dir("datasets--cais--mmlu")
    subs = sorted({p.parent.name for p in m.rglob("*.parquet")})
    want = ("math", "algebra", "logic", "statistic")
    for s in [x for x in subs if any(k in x for k in want)]:
        for split in splits:
            try:
                f = find_file(m / s, split)
            except SystemExit:
                continue
            rows = parquet_rows(f)
            for row in rows[: limit or len(rows)]:
                try:
                    nodes.append(mmlu_node(row, s, split))
                except Refused as e:
                    refuse(e)
    return nodes, refused


def selftest():
    """What the builder claims, checked where it can fail loudly.

    The one that earns its place is the canary. Building 1,100 nodes refused
    two, both for having no arithmetic at all -- so the arithmetic checker had
    never once rejected a *wrong* claim, and a checker that has never rejected
    anything is indistinguishable from one that checks nothing. That is
    `differ`'s argument about its own suite, and it applies here exactly.
    """
    ok = True

    def check(what, cond):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
        ok &= bool(cond)

    def refuses(f):
        try:
            f()
        except Refused:
            return True
        return False

    check("precedence, and brackets that override it",
          close(arith("2+3*4"), 14) and close(arith("(2+3)*4"), 20))
    check("unary minus, and division by a later term", close(arith("-6/2"), -3.0))
    check("decimal literals", close(arith("0.5*8"), 4.0) and close(arith(".25*4"), 1.0))
    check("division by zero is refused, not infinite", refuses(lambda: arith("1/0")))
    # The pair that matters: `//` has to floor, and it has to lex as one token.
    # Read as two divisions it is a parse error; read as one division it agrees
    # with the answer whenever the division happens to be exact, and disagrees
    # silently when it is not.
    check("floor division floors, and is one token",
          close(arith("560//10"), 56) and close(arith("7//2"), 3))

    # The corpus builder must not be a way to run what the dataset says. There
    # is no name, call or attribute in the grammar at all, so this is refused
    # at the tokenizer rather than by a denylist somebody has to maintain.
    check("an expression that is not arithmetic is refused, never run",
          refuses(lambda: arith("__import__('os').system('echo')")))

    wrong = {"question": "q", "answer": "two and two is <<2+2=5>>5.\n#### 5"}
    right = {"question": "q", "answer": "two and two is <<2+2=4>>4.\n#### 4"}
    check("a node whose own arithmetic is wrong is refused",
          refuses(lambda: gsm8k_node(wrong)))
    check("and one whose arithmetic holds is admitted",
          not refuses(lambda: gsm8k_node(right)))

    # A node that cannot say what its answer is has nothing to be checked
    # against later, which is worse than a node that is merely hard.
    check("a solution with no final answer is refused",
          refuses(lambda: gsm8k_node({"question": "q", "answer": "<<2+2=4>>4."})))

    n = gsm8k_node(right)
    path = bucket(n) + ["00000"]
    fields, order = parse(render(n, path))
    check("head is the first field, so an index reader can stop after one line",
          order[:1] == ["head"])
    check("a round trip keeps the answer and the checked step",
          fields.get("answer") == "4" and fields.get("check") == ["2+2 = 4"])
    # `text` carries newlines and the question mark, bracket and equals signs
    # that a key-value format usually has to escape. It does not, because a
    # continuation line is anything whose first word is not a key.
    multi = gsm8k_node({"question": "a? b = c (d)", "answer": "x <<1+1=2>>2\nyy\n#### 2"})
    f2, _ = parse(render(multi, path))
    check("a body may contain newlines, brackets and equals with no escaping",
          "yy" in f2.get("text", "") and "(d)" in f2.get("text", ""))

    print("  " + ("all claims hold" if ok else "SOMETHING IS WRONG"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="rows per source, 0 for everything")
    ap.add_argument("--verify", action="store_true",
                    help="re-read an emitted tree instead of building one")
    ap.add_argument("--allow-test", action="store_true", dest="allow_test",
                    help="ingest MMLU's test split too. Refused by default: a "
                         "node carries its answer, so a forest holding a test "
                         "split is an answer key for that rail.")
    ap.add_argument("--selftest", action="store_true",
                    help="check the builder's own claims, with no data")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.outdir:
        raise SystemExit("  usage: forest.py <outdir> [--limit N] [--verify]")
    out = Path(args.outdir)

    if args.verify:
        verify(out)
        return

    splits = SPLITS + ("test",) if args.allow_test else SPLITS
    nodes, refused = collect(args.limit, splits)
    # **The composition is printed, because it is the number nobody looked
    # at.** A forest is a pile of nodes until you ask which split each came
    # from, and the answer decided whether a planned experiment was valid.
    from collections import Counter
    comp = Counter(n["source"].rsplit("/", 1)[-1] if "/" in n["source"]
                   else n["source"] for n in nodes)
    print("  splits: " + ", ".join(f"{k} {v}" for k, v in sorted(comp.items())))
    if "test" in comp:
        print("  WARNING: this forest holds a test split and is an answer key "
              "for that rail -- do not retrieve into it while scoring one")
    print(f"[forest] {len(nodes)} node(s) admitted")
    if refused:
        # Printed unprompted, the way `traces.py` reports what it could not
        # produce: a builder that quietly drops a third of its input yields a
        # corpus whose gaps nobody knows about.
        total = sum(refused.values())
        print(f"  {total} refused:")
        for why, n in sorted(refused.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:5d}  {why}")
    if not nodes:
        raise SystemExit("  nothing to write")
    emit(out, nodes)
    print(f"  now: python tools/mkpkg.py {out} out/forest.pkg --max-bytes 8388608")


if __name__ == "__main__":
    sys.exit(main())
