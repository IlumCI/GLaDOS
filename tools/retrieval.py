#!/usr/bin/env python3
"""The retrieval rail: known-item recall over the forest, per query.

**The rail every knob in `src/ai/knob.rs` claims to move, and until now the
only thing that could measure it was `forest bench` inside a booted kernel
with a store mounted.** So a source proposal claiming `host.retrieval` could
be built and could not be judged, which makes it a proposal nobody can answer.

The task is built so it cannot be gamed, and the construction is `lex.rs`'s
own: a node's *concept* is the first sentence of its text, so the index holds
the concept and its terms while the query is everything after that first
sentence. Prose from the same node that the index has never seen, one right
answer among thousands.

Per query and not one percentage, because the percentage is the weakest thing
this can produce. Two runs over the same queries either agree on an item or
they do not, and only the disagreements carry information -- which is
McNemar's test, needs per-item outcomes, and is exactly what
`tools/paired.py` and `tools/rails.py` already know how to read. A harness
that printed 91.9% and threw the items away would make the loop's judge a
comparison of two numbers.

Usage:
    retrieval.py out/forest --limit 200 --dump out/retrieval-a.tsv
    retrieval.py --selftest

The dump's format is `lm_eval.py`'s, so `rails.py compare` pairs it with no
special case: `# task`, then `qid want got hit ntok` per line.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import forest_retrieve as fr


def concept_of(node):
    """The sentence the index actually holds, out of `subject | concept | terms`.

    **Not `node.head`, and getting that wrong made this rail read 99.5%.** A
    forest head is three fields joined by ` | `, so stripping the whole head
    off the body strips nothing -- the body does not start with the subject --
    and the query kept the very sentence the index was being asked for. A
    lookup wearing a benchmark's costume, and the number it produced looked
    like a good result rather than like a broken one, which is the shape this
    project keeps recording.
    """
    head = (node.head or "").strip()
    parts = head.split(" | ")
    return parts[1].strip() if len(parts) >= 2 else head


def split_head(node):
    """(the concept the index holds, the prose used as a query), or None.

    The query is the body with its first sentence removed, because the first
    sentence *is* the concept and asking with it would be handing the index
    the key. A node with nothing after it has no query and is skipped --
    counted rather than dropped, since a corpus where most nodes are skipped
    is a corpus this rail cannot measure.
    """
    body = (node.text or "").strip()
    concept = concept_of(node)
    if not body or not concept:
        return None
    tail = body[len(concept):].strip() if body.startswith(concept) else body
    # It has to have actually been removed. A node whose body does not begin
    # with its concept is one this construction cannot make a fair query out
    # of, and guessing would put the answer back in the question.
    if concept in tail:
        return None
    if len(tail.split()) < 8:
        return None
    return concept, tail


# The word pool a fixture forest is built from.
#
# Ordinary English rather than nonsense, because the scorer weights by document
# frequency and a pool of unique tokens would give every query exactly one
# candidate: a benchmark that is 100% by construction and cannot move. These
# repeat at very different rates, which is what makes `IDF_POW` and `LEN_B`
# have anything to bite on.
_POOL = (
    "ring field group set axiom proof lemma matrix vector basis kernel image "
    "prime factor modulus residue graph edge vertex path cycle tree leaf root "
    "apple basket crate lorry driver mile hour minute wage shift ticket fare "
    "garden fence plank nail hammer paint litre metre gram flour sugar butter "
    "pupil class marker sheet folder shelf window ledger column entry balance"
).split()


def build_fixture(dst, nodes=400, seed=7):
    """A forest with no third-party data in it, for a runner that has none.

    **It proves the pipeline runs. It is not evidence about a constant**, and
    the difference is worth being exact about because the temptation is to
    treat it as both. Measured against the real forest: `LEN_B` at 0.75 moves
    thirteen items of six hundred there and one or two of three hundred here.
    A fixture engineered until it *was* that sensitive would be a fixture whose
    verdicts are about the fixture.

    So a runner with no corpus gets a rail that says the tooling works, and
    `rails.py` answering `absent` for the real one is the correct outcome
    rather than a gap to paper over -- a proposal whose rail the judging
    machine cannot measure must be refused, not adopted blind.

    **The absolute number this produces means nothing and is not meant to.**
    What a judge needs is two measurements of the same task under two builds,
    and for that the corpus only has to be fixed, shaped like the real one, and
    hard enough to have headroom. It is generated for the reason `mkwad.py`
    generates its art and `hybtest.py` its weights: the real corpus is somebody
    else's and is not in this repository.

    Each node is a concept sentence plus a body that shares *some* of its
    vocabulary and introduces more, which is the structure a GSM8K node has --
    the rest of a word problem names the people its first sentence introduced.
    Seeded, so two runs on two builds measure the same thing rather than two
    samples of it.
    """
    import random

    rng = random.Random(seed)
    root = Path(dst)
    for i in range(nodes):
        # A long tail of shared words and a short head of rarer ones, so
        # document frequency spans orders of magnitude the way it does in
        # prose. Without that spread, weighting by rarity changes nothing and
        # the knobs that exist to tune it cannot be measured at all.
        shared = [_POOL[rng.randrange(0, 20)] for _ in range(4)]
        rare = [_POOL[rng.randrange(20, len(_POOL))] for _ in range(3)]
        concept = "The " + " ".join(shared[:2] + rare[:1]) + " problem here."

        # **Lengths vary by an order of magnitude, and they have to.** The
        # first fixture gave every node about the same number of terms, and
        # `LEN_B` is a *length* discount: with nothing to discount between, a
        # value of 0.75 against 0.5 moved three items in four hundred and the
        # rail could not tell the knobs apart. A corpus with one length is a
        # corpus that cannot measure the constant that exists to normalise
        # length, which is the same shape as a benchmark with no headroom.
        n_terms = 4 + int(rng.random() ** 2 * 56)
        tail = " ".join(
            [rng.choice(shared + rare) for _ in range(max(2, n_terms // 4))]
            + [_POOL[rng.randrange(0, len(_POOL))] for _ in range(n_terms)]
        )
        d = root / f"part-{i // 100:02}"
        d.mkdir(parents=True, exist_ok=True)
        terms = " ".join(dict.fromkeys(shared + rare))
        eol = chr(10)
        body = eol.join([
            f"head fixture/part-{i // 100:02} | {concept} | {terms}",
            "kind fixture",
            "source fixture",
            f"concept {concept}",
            f"text {concept} {tail}",
            "",
        ])
        (d / f"{i:05}").write_text(body, encoding="utf-8", newline=eol)
    print(f"  fixture: {nodes} node(s) -> {dst}")


def qid(node):
    """A stable id for a query, keyed on the node it is asking for.

    The node's path and not its position: `--limit` and a different sample
    order both move a position and neither moves a path, so two dumps taken at
    different limits still pair on whatever they have in common. The same
    argument `lm_eval.qid` makes.
    """
    return hashlib.sha1(node.path.encode("utf-8")).hexdigest()[:10]


#: How many words of the tail a *short* query is.
#
# **The form with headroom, and the form a person actually types.** Asked with
# the whole body tail, this corpus reads 96.8% at r@1 -- it is 97.6% GSM8K,
# where the rest of a word problem names the same people and objects as its
# first sentence, so known-item recall is nearly free. Thirteen misses in four
# hundred is not something a judge can work with: `MIN_FIXED` wants four net
# repairs and there are thirteen to be had.
#
# Eight words is `lex.rs`'s own short form, and the two columns it reports
# there differ by forty points. A rail wants the discriminating one.
SHORT_WORDS = 8


def run(forest_dir, limit=0, short=True, quiet=False):
    """Answers (rows, skipped). A row is (qid, want, got, hit, nterms)."""
    f = fr.Forest.load(forest_dir)
    if not quiet:
        print(f"[retrieval] {len(f.nodes)} node(s), {len(f.df)} distinct term(s)")
        print(f"  LEN_B={fr.LEN_B} IDF_POW={fr.IDF_POW} TF_K1={fr.TF_K1}"
              "  (read from src/ai/lex.rs)")
        print(f"  {'short' if short else 'long'} queries"
              f"{f', first {SHORT_WORDS} words' if short else ', the whole tail'}")

    # Every Nth node rather than the first N. The forest is laid out by
    # subject, so a prefix is one subject and a run limited to 200 would
    # measure abstract algebra and print it as retrieval -- which is the
    # `run_mmlu` failure this project already recorded once.
    nodes = f.nodes
    if limit and limit < len(nodes):
        stride = max(1, len(nodes) // limit)
        nodes = [n for i, n in enumerate(nodes) if i % stride == 0][:limit]

    rows, skipped = [], 0
    for n in nodes:
        pair = split_head(n)
        if not pair:
            skipped += 1
            continue
        _, query = pair
        if short:
            query = " ".join(query.split()[:SHORT_WORDS])
        ranked = f.score(query)
        got = f.nodes[ranked[0][1]].path if ranked else ""
        rows.append((qid(n), n.path, got, 1 if got == n.path else 0,
                     len(query.split())))
    return rows, skipped


def write_dump(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# retrieval\tqid\twant\tgot\thit\tntok\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def selftest():
    ok = True

    def claim(good, what):
        nonlocal ok
        if not good:
            ok = False
        print(f"  {'ok ' if good else 'FAIL'}  {what}")

    class N:
        def __init__(self, path, head, text):
            self.path, self.head, self.text = path, head, text

    # **A node in the real forest's shape**, which is where this went wrong:
    # the head is `subject | concept | terms` and only the middle field is the
    # sentence the body begins with. A claim written against a synthetic node
    # whose head *was* its first sentence passed while the rail read 99.5%,
    # because 99.5% is what you get for asking the index for a string it
    # holds.
    real = N("a/b",
             "maths/rings | What is a ring. | ring set axioms",
             "What is a ring. A ring is a set with two operations that satisfy "
             "a list of axioms about them.")
    got = split_head(real)
    claim(got is not None, "a node in the forest's own head shape yields a query")
    claim(got[0] == "What is a ring.", "and the concept is the middle field, not the whole head")
    claim(got[1].startswith("A ring is"), "and the query begins after the concept")
    claim("What is a ring" not in got[1], "and the concept is not in the query")

    # The guard that would have caught the original bug on any corpus: if the
    # concept survives into the query, there is no fair question to ask.
    leaky = N("a/b", "s | Q. | t", "Preamble. Q. And more words after it here.")
    claim(split_head(leaky) is None,
          "a body that does not begin with its concept is refused rather than asked")

    claim(split_head(N("a", "s | Head. | t", "Head.")) is None,
          "a node whose body is only its concept has no query and is skipped")
    claim(split_head(N("a", "", "text")) is None, "and one with no head at all")
    claim(split_head(N("a", "s | H | t", "H and four more words")) is None,
          "and one whose tail is too short to be a question")

    # The short form is a prefix of the long one, so a rail that claimed to be
    # asking eight words and was asking all of them would still pair, still
    # look plausible, and be measuring the easy task.
    long_q = split_head(real)[1]
    short_q = " ".join(long_q.split()[:SHORT_WORDS])
    claim(len(short_q.split()) == SHORT_WORDS and long_q.startswith(short_q),
          "a short query is the first eight words of the long one and no more")

    # Ids key on the node, so two runs at different limits still pair.
    claim(qid(N("x/y", "", "")) == qid(N("x/y", "h", "t")),
          "an id is a property of the node and not of what was asked")
    claim(qid(N("x/y", "", "")) != qid(N("x/z", "", "")),
          "and two nodes are two ids")

    # The dump has to be the shape `paired.read` expects, or `rails.py` pairs
    # nothing and reports it as no overlap rather than as a broken format.
    import tempfile
    import paired

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.tsv")
        write_dump(p, [("aa", "w/1", "w/1", 1, 9), ("bb", "w/2", "w/3", 0, 7)])
        task, rows = paired.read(p)
        claim(task == "retrieval", "the dump names its task where paired.py looks")
        claim(len(rows) == 2 and rows["aa"][0] and not rows["bb"][0],
              "and its hits read back as hits")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("forest", nargs="?", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--long", action="store_true",
                    help="ask with the whole body tail rather than its first "
                         f"{SHORT_WORDS} words")
    ap.add_argument("--dump", default="")
    ap.add_argument("--fixture", default="",
                    help="write a generated forest here and measure that")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        print("[retrieval] the rail, without a forest to point it at")
        return 0 if selftest() else 1
    if a.fixture:
        build_fixture(a.fixture, nodes=a.nodes)
        a.forest = a.forest or a.fixture
    if not a.forest:
        raise SystemExit("  usage: retrieval.py <forest dir> [--limit N] [--dump FILE]")

    rows, skipped = run(a.forest, limit=a.limit, short=not a.long)
    if not rows:
        raise SystemExit("  no node in that forest has prose after its head")
    hits = sum(r[3] for r in rows)
    print(f"  {len(rows)} quer(y|ies), {skipped} node(s) with no query")
    print(f"  r@1 {hits}/{len(rows)} = {hits / len(rows):.1%}")
    if a.dump:
        write_dump(a.dump, rows)
        print(f"  per-query outcomes -> {a.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
