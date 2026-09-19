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


def answer_key_nodes(nodes):
    """Nodes that make this forest an answer key: a test-split node that
    carries an answer.

    **The check used to be `"test" in composition`, and that is the name
    rather than the hazard.** What makes a forest an answer key is stated
    two paragraphs up -- test questions went in *carrying their `answer`
    field* -- so the predicate is the conjunction, and it has to be, now
    that a source exists whose own splits are called `test` and
    `validation` and which has no answers at all. wikitext partitions
    Wikipedia articles; there is nothing in one to leak into an MMLU score.

    This is strictly sharper and never weaker: every MMLU test node carries
    an answer, so every one of them still trips it. `selftest` holds both
    directions so a later loosening fails loudly rather than quietly.
    """
    return [n for n in nodes
            if n["source"].rsplit("/", 1)[-1] == "test" and n["answer"].strip()]


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


#: Where the wikitext parquet files live. Fetched rather than cached under
#: ~/.cache, because this is the one source that is not a HuggingFace
#: `datasets--*` snapshot and pretending it were would put a hand-made
#: directory where `snapshot_dir` promises a real one.
WIKI_DIR = Path(__file__).resolve().parent / "hf" / "wikitext"

#: ` = Title = `, ` = = Section = = `, ` = = = Sub = = = `. The equals signs
#: are SPACED in wikitext, so `=+` matches the first one and nothing else --
#: which collapsed every article to a single node and read as a corpus of
#: 122 very long documents rather than 1,063 sections.
WIKI_HEAD = re.compile(r"^ (=(?: =)*) (.+?) \1 $")


def detok(s):
    """Wikitext back into prose.

    **The corpus ships pre-tokenised and the index must not see that.**
    wikitext writes `gunpowder @-@ propelled`, ` , ` and ` . `, which are
    artefacts of how the set was built for language modelling. Indexed raw
    they become terms no query ever spells: a retrieval corpus whose terms
    are `@-@` is a corpus that scores its own preprocessing. The reverse is
    lossy in the other direction too -- `@.@` really is a decimal point and
    `@-@` really is a hyphen, so this restores rather than guesses.
    """
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    s = re.sub(r" ' s\b", "'s", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def slug(s, cap=48):
    """A directory name from an article title.

    `emit` uses what `bucket` returns as a path component verbatim, and an
    encyclopedia title carries `(`, `)`, `:` and `/` -- none of which are
    names this can write on Windows, and one of which would silently make a
    deeper tree than the bucket declared.
    """
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:cap].rstrip("-") or "untitled")


def wiki_sections(rows):
    """(article, heading path, body) per leaf section.

    Splits at every heading level rather than at articles, because an
    article is thousands of words and a section is a few hundred -- and a
    node is what retrieval returns into a context window with a budget.
    """
    art, path, buf = None, [], []
    for r in rows:
        line = r["text"].rstrip("\n")
        m = WIKI_HEAD.match(line)
        if m:
            if buf and art:
                yield art, " / ".join(path[1:]), " ".join(buf)
            buf = []
            lvl, title = m.group(1).count("="), detok(m.group(2))
            if lvl == 1:
                art, path = title, [title]
            else:
                path = (path[:lvl - 1] + [title]) if art else []
        elif line.strip() and art:
            buf.append(line.strip())
    if buf and art:
        yield art, " / ".join(path[1:]), " ".join(buf)


def wiki_node(article, heading, raw, split):
    """A section of a Wikipedia article as a node.

    **It carries no `check` and that is the honest shape**, not an omission.
    GSM8K admits a node by executing the arithmetic it states about itself;
    there is nothing in an encyclopedia paragraph a builder can execute, and
    inventing a check that always passes would be worse than having none --
    it would make the refusal counters read as though prose had been
    verified. What this source is admitted on is structure: long enough to
    be a document, prose enough to have a sentence, and not mostly `<unk>`.
    """
    body = detok(raw)
    if body.count("<unk>") > 2:
        raise Refused("too many unknown-word markers to be prose")
    if len(body.split()) < WIKI_MIN_WORDS:
        raise Refused("too short to be a document")
    if not re.search(r"[.?!]\s", body):
        raise Refused("no sentence end, so no concept to index")
    return {
        "kind": "wiki",
        "source": f"wikitext-103/{slug(article)}/{split}",
        "concept": first_sentence(body),
        "method": heading,
        "checks": [],
        "answer": "",
        "text": body,
        "terms": terms_of(body),
    }


#: What the CC BY-SA 3.0 / GFDL licence on the wikitext source obliges,
#: written into the tree rather than only into a commit message, because
#: the tree is what gets published.
ATTRIBUTION = """\
Bodies of nodes under encyclopedia/ are sections of Wikipedia articles,
taken from the wikitext-103-raw-v1 dataset (Salesforce/wikitext on
HuggingFace; Merity et al., "Pointer Sentinel Mixture Models", 2016).

Wikipedia text is licensed CC BY-SA 3.0 and GFDL. Reuse of this tree
carries those terms for those nodes, including attribution and
share-alike. https://creativecommons.org/licenses/by-sa/3.0/

Text has been detokenised (the dataset ships ` @-@ ` and spaced
punctuation) and split at heading boundaries. No other change was made.
Nodes of kind `mmlu` and `gsm8k` come from their own datasets and are not
covered by this notice.
"""


#: A section under this many words is a stub, a disambiguation line or a
#: table caption rather than a document. Declared because it is the one
#: number that decides how much of the source survives, and the whole point
#: of this source is document *length*.
WIKI_MIN_WORDS = 60


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



# Which tree an MMLU subject belongs to.
#
# **A declared table, because there is nothing to derive it from.** Every
# other bucket in this file is read off the node -- GSM8K branches by the
# operators its verified chain actually uses -- and that is the rule here
# too wherever it can be kept. It cannot be kept for a discipline: nothing
# in the string "professional_medicine" says life sciences, and a rule that
# guessed from substrings would put `machine_learning` under psychology on
# the strength of "learning". So it is a closed table in the `KNOBS` and
# `repair::ACTIONS` idiom -- one place to read, one place to be wrong in --
# and a subject absent from it becomes its OWN tree rather than being filed
# somewhere plausible. Misfiling is worse than a thin tree: the subject
# router pools a tree's nodes into one vector, so a wrong home is a wrong
# vector for every node that lands in it.
#
# **Why the forest stopped being one subject.** The filter here was
# `("math", "algebra", "logic", "statistic")` and every surviving subject
# was hardcoded to the `logic-and-maths` tree, so `host.retrieval` measured
# retrieval *within mathematics* and any constant tuned against it was
# tuned on one discipline -- the `run_mmlu` failure this project already
# recorded once, where 100 questions of abstract_algebra were reported as
# MMLU. Widening relaxes no admission rule, because `mmlu_node` never had
# the arithmetic check in the first place: that discipline is GSM8K's, and
# it is untouched.
MMLU_TREES = {
    "logic-and-maths": (
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "formal_logic", "high_school_mathematics", "high_school_statistics",
        "logical_fallacies", "econometrics",
    ),
    "physical-sciences": (
        "astronomy", "college_chemistry", "college_physics",
        "conceptual_physics", "high_school_chemistry", "high_school_physics",
    ),
    "life-sciences": (
        "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "high_school_biology", "human_aging", "medical_genetics", "nutrition",
        "professional_medicine", "virology", "human_sexuality",
    ),
    "computing": (
        "college_computer_science", "computer_security", "electrical_engineering",
        "high_school_computer_science", "machine_learning",
    ),
    "law-and-politics": (
        "international_law", "jurisprudence", "professional_law",
        "high_school_government_and_politics", "us_foreign_policy",
        "security_studies",
    ),
    "economics-and-business": (
        "business_ethics", "high_school_macroeconomics",
        "high_school_microeconomics", "management", "marketing",
        "professional_accounting", "public_relations",
    ),
    "history-and-geography": (
        "high_school_european_history", "high_school_geography",
        "high_school_us_history", "high_school_world_history", "prehistory",
        "global_facts",
    ),
    "mind-and-society": (
        "high_school_psychology", "professional_psychology", "sociology",
        "moral_disputes", "moral_scenarios", "philosophy", "world_religions",
        "miscellaneous",
    ),
}

#: subject -> tree, inverted once so `bucket` is a lookup rather than a scan.
MMLU_TREE_OF = {s: t for t, subs in MMLU_TREES.items() for s in subs}


def bucket(node):
    """Where a node lives. Derived from the data, never hand-assigned.

    GSM8K branches by which operators the solution actually uses and by how
    many steps it takes -- both read off the verified chain, so the taxonomy is
    a fact about the node rather than a guess about it. Step count is a
    difficulty axis, the same shape `redqueen`'s archive bands use.
    """
    if node["kind"] == "mmlu":
        subject = node["source"].split("/")[1]
        # Its own tree when the table does not name it, so a new MMLU
        # subject is visible as a thin tree rather than silently swelling
        # whichever one a substring rule happened to match.
        tree = MMLU_TREE_OF.get(subject, subject.replace("_", "-"))
        return [tree, subject.replace("_", "-")]
    if node["kind"] == "wiki":
        # **One declared tree, because this source carries no subject
        # labels at all.** MMLU ships a subject per row and GSM8K's family
        # is read off its own verified chain; an encyclopedia article says
        # only what it is about, in prose. Classifying 122 articles into
        # the eight MMLU disciplines would be exactly the substring guess
        # `MMLU_TREES` exists to refuse, so the article itself is the
        # second level -- derived from the data, and capping fanout for
        # free the way a subject does. `<kind>/<subject>/<split>` is the
        # layout the composition report reads its last component from, so
        # the article goes in the MIDDLE: putting it last made that report
        # list 122 article slugs where it means to list splits, which is
        # the one number its own comment says nobody looked at.
        return ["encyclopedia", node["source"].split("/")[1]]
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
    # **A node is a file whose first word is `head`, which is the rule
    # `emit`'s sweep already uses.** The two disagreed: emit left a foreign
    # file alone and counted it out loud, while this read every file as a
    # node and reported the difference as a corrupt one. Nothing exercised
    # the gap until the tree had to carry its own CC BY-SA notice, and then
    # a correct build failed its own verify. One rule, stated once.
    every = sorted(p for p in outdir.rglob("*") if p.is_file())
    files, foreign = [], 0
    for p in every:
        try:
            with p.open("r", encoding="utf-8") as f:
                head = f.readline()
        except (OSError, UnicodeDecodeError):
            head = ""
        (files.append(p) if head.startswith("head ") else None)
        foreign += 0 if head.startswith("head ") else 1
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
    if foreign:
        print(f"  {foreign} file(s) are not nodes, left alone (emit says the same)")
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
    # Every subject the snapshot carries. See `MMLU_TREES` for why the
    # maths-only filter that used to live here was a measurement problem.
    for s in subs:
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

    # **The third source, and it is here to change the SHAPE of the corpus
    # rather than to add to it.** MMLU and GSM8K are both exam items, so
    # every constant `host.retrieval` judges was fitted to one register --
    # question stems and their choice lists. Wikipedia sections are
    # encyclopedic prose: different concept sentences, different queries,
    # and 4,713 distinct terms the index had never seen (14,342 -> 19,055).
    #
    # **It does NOT give `LEN_B` longer documents, and the first version of
    # this comment said it did.** That was reasoned from body length --
    # wiki sections run 362 words against MMLU's 88 -- and the reasoning
    # never reached the index. `forest_retrieve.Node` says so in as many
    # words: the length charge is over the *indexed* document, which is the
    # head line, and a head is `subject | concept | terms` by construction.
    # Measured over both corpora: mean 20 terms, median 20, max 32. A
    # source of 362-word bodies moved that distribution not at all.
    #
    # What it did move is the sweep. Across LEN_B 0.0 to 1.0 the rail spans
    # 57.6-60.0% on the two-source forest and 41.6-46.8% on this one, so
    # the constant's measurable effect roughly doubled -- from vocabulary
    # and from 2,697 more nodes to be confused by, not from length.
    for split in ("test", "validation"):
        f = WIKI_DIR / f"{split}.parquet"
        if not f.is_file():
            continue
        seen = 0
        for article, heading, raw in wiki_sections(parquet_rows(str(f))):
            if limit and seen >= limit:
                break
            seen += 1
            try:
                nodes.append(wiki_node(article, heading, raw, split))
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

    # --- the third source ---------------------------------------------
    raw = " Robert Boulter is an English film , television and theatre actor ."
    check("wikitext detokenises back into prose",
          detok(" a @-@ b , c . ") == "a-b, c." and
          detok(raw).startswith("Robert Boulter is an English film,"))
    check("a title becomes a directory name a filesystem will take",
          slug("Kiss You ( One Direction song )") == "kiss-you-one-direction-song"
          and "/" not in slug("New Jersey Route 29 / 50"))

    lines = [{"text": " = A = "}, {"text": "one two three."},
             {"text": " = = S = = "}, {"text": "four five six."},
             {"text": " = = = T = = = "}, {"text": "seven eight nine."}]
    secs = list(wiki_sections(lines))
    # The regex that reads these counts SPACED equals. `=+` matches the
    # first one only, which collapsed every article into a single node and
    # read as a corpus of 122 very long documents -- plausible, and wrong.
    check("every heading level splits, not only the article title",
          [h for _a, h, _b in secs] == ["", "S", "S / T"])
    check("and the article carries through each of them",
          {a for a, _h, _b in secs} == {"A"})

    long_enough = "The subject is a thing. " * 40
    n = wiki_node("A Title", "S", long_enough, "test")
    check("a wiki node carries no check, having nothing to execute",
          n["checks"] == [] and n["answer"] == "")
    for bad, why in ((" ".join(["<unk>"] * 5), "unknown-word"),
                     ("too short", "short"),
                     ("nosentenceendhere " * 40, "sentence")):
        try:
            wiki_node("T", "", bad, "test")
            check(f"a {why} body is refused", False)
        except Refused:
            check(f"a {why} body is refused", True)

    # **The answer-key guard, both directions.** Sharpening it from "is
    # there a test split" to "is there a test-split node carrying an
    # answer" is only safe if the first direction still fires, so the
    # canary is an MMLU-shaped node and it must trip.
    keyed = {"source": "mmlu/x/test", "answer": "B"}
    check("a test-split node carrying an answer still makes an answer key",
          len(answer_key_nodes([keyed])) == 1)
    check("and one with nothing to leak does not",
          answer_key_nodes([{"source": "wikitext-103/a/test", "answer": ""}]) == [])

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
    risky = answer_key_nodes(nodes)
    if risky:
        print(f"  WARNING: {len(risky)} test-split node(s) carry an answer, so "
              "this forest is an answer key "
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
    if any(n["kind"] == "wiki" for n in nodes):
        # **CC BY-SA is a licence with an obligation, and the obligation is
        # attribution.** This tree gets published as a release asset, so the
        # notice travels with the bytes rather than living in a commit
        # message -- the same reason `rtl8188eu_tables.rs` says at the top
        # what it came from. `emit` leaves it alone and counts it out loud,
        # because its sweep only deletes files whose first word is `head`.
        (out / "ATTRIBUTION").write_text(ATTRIBUTION, encoding="utf-8",
                                         newline="\n")
        print(f"  wrote {out / 'ATTRIBUTION'} -- CC BY-SA 3.0 travels with this tree")
    # **The cap is computed from what was just written, not written down.**
    # A hardcoded 8388608 was right for a 9,194-node forest and refuses an
    # 10,257-node one, so the line this prints would have been a command
    # that does not work -- the same drift as a figure in a doc going stale,
    # arriving in the one place somebody copies and pastes.
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    cap = 1 << max(22, (size * 5 // 4 - 1).bit_length())
    print(f"  {size} byte(s) on disk")
    # `--name` is required and this line omitted it, so the command it
    # printed has never once run as printed.
    print(f"  now: python tools/mkpkg.py {out} out/forest.pkg"
          f" --name forest --max-bytes {cap}")


if __name__ == "__main__":
    sys.exit(main())
