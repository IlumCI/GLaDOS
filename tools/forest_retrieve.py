"""Read side of the forest, on the host.

`forest.py` builds the tree and `src/ai/forest.rs` reads it in the kernel.
This is the third reader, and it exists so a retrieval figure can be measured
before any of it is wired through the kernel: `lm_eval.py --forest out/forest`
retrieves into the prompt and the delta against the same run without it is the
measurement.

**It is `src/ai/lex.rs`'s scorer and not a new one**, which is the whole
reason it is worth anything. The kernel already ranks nodes, already swept the
families that could do it, and already has the numbers: 91.9% recall-at-1 over
198 known-item queries on this same forest, against 80.3% for BM25 and 0.5%
for mean-pooled embeddings. A host retriever that scored by some other formula
would answer a question about a machine nobody is building. See `LEN_B` below
for what was got wrong here first.

What this is *not* is the embedding router. `route.rs` cosines a pooled
question against pooled branch descriptors and `recall.rs` fills a token
budget from them, and neither runs here -- pooling needs the checkpoint's own
embedding table. So a delta measured through this file is a delta from term
matching, which is the floor, and is not evidence about the routed path.

**The guard, which matters more than the scoring.** A forest node carries its
own answer. A forest holding the split being scored is therefore an answer
key, and this project has already built one by accident: 1,353 MMLU test
questions went in with their `answer` fields because the ingest walked every
split it could find. `forest.py` refuses `test` now, but that is a guarantee
about the builder and this is a different program reading the tree it left
behind. So `refused` below re-checks it from the other side, whatever the
builder believed, and `lm_eval` prints the count.

**What that count means is a rate and not a bit.** The check is deliberately
loose enough to catch a reworded duplicate, so a handful of refusals across a
run is two datasets sharing a question, which happens. A large fraction of
questions refusing something is the forest holding the split being scored, and
no figure from that run is about retrieval.
"""

import argparse
import math
import re
import sys
from pathlib import Path

# Mirrors `forest.STOP`. Duplicated rather than imported: this reader is meant
# to be separable from the writer, which is the same bargain `forest.parse`
# makes with `forest.render`.
STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "that", "this", "it", "as", "be", "from",
    "how", "what", "which", "if", "then", "each", "many", "much", "does", "do",
}

WORD = re.compile(r"[a-zA-Z][a-zA-Z-]{2,}")
ALNUM = re.compile(r"[^a-z0-9]+")

# **The leak check counts numbers and the scorer does not, and the difference
# is the whole of what separates a duplicate from a sibling.**
#
# MMLU is templated. Scoring 100 abstract-algebra questions with retrieval on,
# twelve of them refused a node -- and every one of those nodes was a
# *validation* question from the same template with different numbers in it:
#
#     test     Find the order of the factor group (Z_11 x Z_15)/(<1, 1>)
#     refused  Find the order of the factor group (Z_4 x Z_12)/(<2> x <2>)
#
# Every content word is shared, so containment read 1.00 over a question whose
# alphabetic vocabulary is four words long. That is not the question with its
# answer attached; it is a worked example of the same kind, which is the single
# most useful thing a forest can hand a model, and the guard was eating it.
#
# Numbers are what the template varies, so including them is exactly the
# discrimination needed. The Natalia case still refuses -- 48, 24 and 72 are
# shared as well as the words -- and the algebra sibling no longer does.
#
# It is deliberately *only* the leak check. `lex.rs` scores over BPE ids and
# this scores over words, and widening the scorer's vocabulary here would move
# it further from the kernel's rather than closer.
LEAK_TERM = re.compile(r"[a-z]{3,}|\d+")

# `src/ai/lex.rs`'s scorer, and matching it is the whole point.
#
# **The first version of this was BM25 with k1 = 1.2, which the kernel measured
# and rejected.** `forest bench` swept the families over 198 known-item queries
# on this same forest: BM25 read 80.3% r@1 where presence-weighted idf-squared
# read 91.9%. These are short questions with almost no term repetition, so `tf`
# is nearly always 1 and BM25's saturation collapses to a constant -- what
# survives is its length normalisation, sitting *inside* the saturation where
# `k1` multiplies it, and that measured worse than charging the sum directly.
#
# A host retriever exists to predict what GLaDOS would retrieve. One scoring by
# a different formula measures a different machine, so the constants below are
# `lex.rs`'s and not a fresh choice: `IDF_SQUARED` because five stopwords
# outweighed the only `derivative` in nine thousand nodes under linear idf, and
# `LEN_B = 0.5` because the sweep has an interior optimum there rather than at
# the textbook 0.75.
#
# One deviation, stated rather than hidden: the kernel indexes BPE token ids
# and this indexes words, so `lex::prep`'s leading-space fix has no analogue
# here and the two will not rank identically. The formula is what is shared.
IDF_SQUARED = True
LEN_B = 0.5

# How much of a question's vocabulary a node may contain before the node is
# treated as that question rather than as material for it. Deliberately
# generous: a node wrongly dropped costs one retrieval and is counted out
# loud, where a node wrongly kept is an answer key and costs the whole run.
LEAK = 0.8

# **Below this many terms a question cannot be judged by overlap at all**, and
# adding numbers to the comparison was not enough to fix that on its own.
#
#     Find the characteristic of the ring Z_3 x 3Z.        <- test
#     Find the characteristic of the ring 2Z.              <- refused at 1.00
#
# Four terms, so one differing term is 0.75 and two is 0.5: there is no
# threshold that separates a duplicate from a sibling at that length, because
# at that length they are the same string with a symbol changed. MMLU is
# templated and its abstract-algebra questions run to four or five content
# terms, which is why this surfaced there and not on GSM8K.
#
# Short questions fall back to the substring test, which is exact. That is not
# a hole: the containment rule exists to catch a *reworded* duplicate, and a
# five-word question has nowhere to put a rewording. The Natalia case carries
# twelve terms and is still refused.
MIN_LEAK_TERMS = 8

# What the retrieved block is introduced by. It is part of the budget, so it
# has to be the same string when the block is counted and when it is rendered.
LABEL = "Related worked examples"


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def leak_words(text):
    """What the leak check compares: content words *and* numbers."""
    return {w for w in LEAK_TERM.findall(text.lower()) if w not in STOP}


def flatten(text):
    """Lowercase, letters and digits only. What the leak check compares."""
    return ALNUM.sub(" ", text.lower()).strip()


class Node:
    __slots__ = ("path", "head", "kind", "source", "text", "terms", "flat",
                 "body", "length")

    def __init__(self, path, fields):
        self.path = path
        self.head = fields.get("head", "")
        self.kind = fields.get("kind", "")
        self.source = fields.get("source", "")
        self.text = fields.get("text", "")
        # The head is what the index holds, so the head is what is scored. The
        # body is never read until a node has already been chosen.
        self.terms = words(self.head)
        # The length charge is over the indexed document, so it is the head's
        # own length and not the body's -- `lex.rs` keeps `len[n]` beside the
        # postings for the same reason.
        self.length = max(1, len(self.terms))
        self.flat = flatten(self.text)
        self.body = leak_words(self.text)


def read_head_and_text(path):
    """`forest.parse`, cut down to the two fields retrieval needs.

    Written from the format description rather than from `forest.render`, for
    the same reason `forest.parse` is: a disagreement should show up as a
    parse failure here and not as a corpus of shifted fields.
    """
    fields, key = {}, None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            first = line.split(" ", 1)[0]
            if first in ("head", "kind", "source", "concept", "method",
                         "check", "answer", "text"):
                key = first
                rest = line[len(first):].lstrip(" ")
                # `check` is the one key that legitimately repeats, so a
                # second occurrence extends rather than replaces. Every field
                # retrieval actually reads appears exactly once.
                fields[key] = (fields[key] + "\n" + rest) if key in fields else rest
            elif key:
                fields[key] = fields.get(key, "") + "\n" + line
    return fields


class Forest:
    def __init__(self, nodes):
        self.nodes = nodes
        self.df = {}
        for n in nodes:
            for t in set(n.terms):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = max(1, len(nodes))
        self.avg = sum(n.length for n in nodes) / self.N if nodes else 1.0
        self.post = {}
        for i, n in enumerate(nodes):
            for t in set(n.terms):
                self.post.setdefault(t, []).append(i)

    @classmethod
    def load(cls, root):
        root = Path(root)
        if not root.is_dir():
            raise SystemExit(f"  no forest at {root}")
        nodes = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            fields = read_head_and_text(p)
            if "head" not in fields:
                continue
            nodes.append(Node(p.relative_to(root).as_posix(), fields))
        if not nodes:
            raise SystemExit(f"  {root} holds no nodes")
        return cls(nodes)

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query):
        """`Lex::score`, in Python. Returns (score, index) descending.

        The share of the query's idf mass a node accounts for, each node
        charged for its length. A term counts once however often it occurs,
        which is `score_raw_p` walking `seen` rather than a term frequency.
        """
        acc = {}
        total = 0.0
        for t in dict.fromkeys(words(query)):
            w = self.idf(t)
            if IDF_SQUARED:
                w *= w
            total += w
            for i in self.post.get(t, ()):
                acc[i] = acc.get(i, 0.0) + w
        if total <= 0.0:
            return []
        for i in acc:
            dl = self.nodes[i].length
            norm = max(1e-6, 1.0 - LEN_B + LEN_B * dl / self.avg)
            acc[i] /= total * norm
        return sorted(((v, i) for i, v in acc.items()), reverse=True)

    def retrieve(self, query, k=4, budget=0, count=None, pool=64,
                 label=LABEL):
        """Top nodes for a question, under a token budget, leak-checked.

        Returns `(nodes, refused)`. `refused` counts nodes dropped for
        carrying the question itself -- see the module docstring for what a
        count there means.

        `budget` is in tokens and `count` is a callable answering how many the
        model's own encoder makes of a string, so the budget is measured the
        way the prompt will be rather than estimated. With none it counts
        characters, which is honest about being an approximation rather than
        quietly wrong about being exact.

        **The whole accumulated block is re-counted per candidate**, which
        `recall::fill` does for the reason it gives: tokenisation is not
        additive at a boundary, so summing per-node counts drifts, always in
        the direction of admitting one node too many.

        A candidate that does not fit is **skipped rather than ending the
        fill**, also `recall::fill`'s rule. The list is sorted by score and not
        by size, so stopping at the first overflow throws away every smaller
        entry behind one large one and leaves the budget unspent.
        """
        flat_q = flatten(query)
        qset = leak_words(query)
        out, refused = [], 0
        for _, i in self.score(query)[:pool]:
            n = self.nodes[i]
            # The leak check, from the reader's side. A node whose body
            # contains the question is that question with its answer attached.
            #
            # **Substring alone is not enough and was measured not to be.**
            # The first thing this retriever was ever asked returned a node
            # whose body was the question with three words moved, and the
            # substring test passed it: the query said "to 48 friends" and the
            # node said "to 48 of her friends". Duplicates across splits do not
            # arrive byte-identical. So containment of the question's content
            # words is checked as well, which catches the paraphrase and the
            # reformatting that substring matching cannot.
            if flat_q and (flat_q in n.flat or n.flat in flat_q):
                refused += 1
                continue
            if (len(qset) >= MIN_LEAK_TERMS
                    and len(qset & n.body) / len(qset) >= LEAK):
                refused += 1
                continue
            if budget:
                block = render_block(out + [n], label)
                cost = count(block) if count else len(block)
                if cost > budget:
                    continue
            out.append(n)
            if len(out) >= k:
                break
        return out, refused


def render_block(nodes, label=LABEL):
    """What goes into the prompt. Empty string for no nodes, so the caller
    can concatenate unconditionally and a forest that returns nothing is
    exactly the run without one."""
    if not nodes:
        return ""
    parts = [f"{label}:"]
    for n in nodes:
        parts.append(n.text.strip())
    return "\n\n".join(parts) + "\n\n"


# --- claims ----------------------------------------------------------------


def selftest():
    import tempfile

    ok = True

    def claim(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")

    d = Path(tempfile.mkdtemp())
    (d / "sub").mkdir()
    (d / "sub" / "00000").write_text(
        "head sub | adding apples | apples basket orchard\n"
        "kind gsm8k\nsource gsm8k/train\nanswer 12\n"
        "text Ann has 5 apples in a basket and picks 7 more.\n"
        "She has 12 apples.\n", encoding="utf-8")
    (d / "sub" / "00001").write_text(
        "head sub | dividing pears | pears crates warehouse\n"
        "kind gsm8k\nsource gsm8k/train\nanswer 4\n"
        "text Bob splits 12 pears into 3 crates for the Saturday market.\n"
        "Each crate holds 4 pears.\n",
        encoding="utf-8")
    (d / "notes.txt").write_text("not a node\n", encoding="utf-8")

    f = Forest.load(d)
    claim("a file with no head line is not a node", len(f.nodes) == 2)
    claim("a multi-line body survives the read",
          "She has 12 apples." in f.nodes[0].text)

    got, ref = f.retrieve("How many apples are in the basket?", k=1)
    claim("the apple question retrieves the apple node",
          len(got) == 1 and "apples" in got[0].text and ref == 0)

    got, ref = f.retrieve(
        "Bob splits 12 pears into 3 crates for the Saturday market.", k=2)
    claim("a node holding the question verbatim is refused", ref >= 1)
    claim("and refusing it does not refuse the others",
          all("pears into 3 crates" not in n.text for n in got))

    # The case substring matching passed and should not have.
    got, ref = f.retrieve("For the Saturday market Bob splits his 12 pears "
                          "into 3 crates", k=2)
    claim("a reworded question is refused too", ref >= 1)
    claim("and a question that merely shares a topic is not",
          f.retrieve("How do you divide fruit between containers?", k=2)[1] == 0)

    # A template sibling is the most useful thing a forest can hand a model,
    # and the guard ate twelve of them on MMLU before the floor existed.
    (d / "sub" / "00002").write_text(
        "head sub | ring characteristic | ring characteristic\n"
        "kind mmlu\nsource mmlu/abstract_algebra/val\nanswer A\n"
        "text Find the characteristic of the ring 2Z.\nA. 0 B. 3 C. 12 D. 30\n",
        encoding="utf-8")
    g = Forest.load(d)
    claim("a short question does not refuse its template sibling",
          g.retrieve("Find the characteristic of the ring Z_3 x 3Z.",
                     k=2)[1] == 0)
    claim("and an exact one is still refused, by the substring test",
          g.retrieve("Find the characteristic of the ring 2Z.", k=2)[1] >= 1)

    # The budget is over the **rendered block**, label and separators
    # included, because that is what reaches the prompt. Summing the nodes
    # would leave the preamble unpaid for and admit one node too many.
    got, _ = f.retrieve("apples", k=2, budget=10)
    claim("a block over the budget is not rendered at all",
          len(render_block(got)) <= 10)

    words_of = lambda s: len(s.split())
    tight, _ = f.retrieve("apples pears", k=2, budget=20, count=words_of)
    big, _ = f.retrieve("apples pears", k=2, budget=10_000, count=words_of)
    claim("a budget in the model's own units is honoured",
          words_of(render_block(tight)) <= 20)
    claim("and it is the budget that cut it, not the pool",
          len(big) == 2 and len(tight) == 1)

    claim("no nodes renders as the run without a forest",
          render_block([]) == "")
    claim("a block ends where the prompt resumes",
          render_block(f.nodes[:1]).endswith("\n\n"))

    print("  all claims hold" if ok else "  CLAIMS FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("forest", nargs="?", default="")
    ap.add_argument("--query", default="")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.forest:
        raise SystemExit("  usage: forest_retrieve.py <dir> --query '...'")

    f = Forest.load(a.forest)
    kinds = {}
    for n in f.nodes:
        kinds[n.source.rsplit("/", 1)[-1]] = kinds.get(
            n.source.rsplit("/", 1)[-1], 0) + 1
    print(f"[forest] {len(f.nodes)} node(s), {len(f.df)} distinct term(s)")
    print("  splits: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    if "test" in kinds:
        print("  WARNING: this forest holds a test split")
    if not a.query:
        return 0
    got, refused = f.retrieve(a.query, k=a.k)
    print(f"  {len(got)} node(s), {refused} refused for carrying the question")
    for n in got:
        print(f"    {n.path}  {n.head[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
