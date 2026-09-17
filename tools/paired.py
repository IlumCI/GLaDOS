#!/usr/bin/env python3
"""Compare two `lm_eval --dump` runs as pairs rather than as two percentages.

    python tools/paired.py out/eval/a.tsv out/eval/b.tsv

**Two accuracies cannot answer whether a change helped.** 39.2% and 37.4% on
1,319 questions differ by 24 items, and the 95% interval on either figure
alone is about plus or minus 2.6 points -- wide enough to swallow the whole
difference twice over. Taken as pairs it is not: the same question under both
configurations either agrees or it does not, and only the disagreements carry
any information at all. That is McNemar's test.

`godel::mcnemar` is the kernel's implementation and this is deliberately the
same arithmetic, Yates' correction included, against the same `MCNEMAR_95` of
3.84 -- two definitions of "beyond the noise" that drifted apart would let a
change be significant to the host and not to the machine, with nothing saying
so. `godel.rs` makes that argument about its own two judges and it applies
across the boundary for the same reason.

What this does **not** do is decide anything. It reports the counts, the
statistic and which side the difference falls on; whether that is worth
adopting is a question with a budget attached, and the budget lives in the
kernel.
"""

import argparse
import sys

# `godel::MCNEMAR_95`. Named rather than inlined, and imported from nowhere,
# because the kernel is Rust and this is Python -- so the one thing that can
# be done is to say which constant this is a copy of, and to say it here.
MCNEMAR_95 = 3.84


def mcnemar(broke, fixed):
    """`godel::mcnemar`, in Python.

    Only the discordant pairs are in it. Yates' correction subtracts one from
    the difference before squaring: without it small counts overstate the
    evidence, and small counts are the regime this lives in.
    """
    n = broke + fixed
    if n == 0:
        return 0.0
    d = abs(broke - fixed)
    num = max(0.0, d - 1.0)
    return num * num / n


def read(path):
    """qid -> (hit, got, want). The header carries the task name."""
    rows, task = {}, ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                task = line[1:].split("\t")[0].strip()
                continue
            p = line.split("\t")
            if len(p) < 4:
                continue
            rows[p[0]] = (p[3] == "1", p[2], p[1])
    if not rows:
        raise SystemExit(f"  {path} holds no rows")
    return task, rows


def report(a_path, b_path, show=0):
    ta, a = read(a_path)
    tb, b = read(b_path)
    if ta and tb and ta != tb:
        raise SystemExit(f"  {ta} against {tb} -- different tasks do not pair")

    # **Paired on the question and never on position.** A run at a different
    # --limit or --seed sees a different sample, and two files lined up row by
    # row would silently compare unrelated questions. The overlap is printed
    # because a small one is the thing that makes the rest meaningless.
    both = sorted(set(a) & set(b))
    if not both:
        raise SystemExit("  the two runs share no questions")

    aa = sum(1 for q in both if a[q][0])
    bb = sum(1 for q in both if b[q][0])
    fixed = [q for q in both if b[q][0] and not a[q][0]]
    broke = [q for q in both if a[q][0] and not b[q][0]]
    chi = mcnemar(len(broke), len(fixed))

    print(f"[paired] {ta or 'task'}, {len(both)} question(s) in common "
          f"({len(a)} and {len(b)} in the two runs)")
    print(f"  A {a_path}   {aa / len(both):6.1%}  ({aa}/{len(both)})")
    print(f"  B {b_path}   {bb / len(both):6.1%}  ({bb}/{len(both)})")
    print(f"  B fixed {len(fixed)}, B broke {len(broke)}, "
          f"agreed {len(both) - len(fixed) - len(broke)}")
    print(f"  mcnemar chi {chi:.2f} against {MCNEMAR_95} at 95%")
    if chi < MCNEMAR_95:
        print("  -> the difference is inside the noise; this run does not "
              "say B is better or worse than A")
    elif len(fixed) > len(broke):
        print("  -> B is better than A beyond the noise")
    else:
        print("  -> B is worse than A beyond the noise")

    for label, qs in (("B fixed", fixed), ("B broke", broke)):
        for q in qs[:show]:
            print(f"    {label} {q}  want {a[q][2]}  "
                  f"A said {a[q][1]}, B said {b[q][1]}")
    return chi


def selftest():
    ok = True

    def claim(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}")

    claim("no discordant pairs is no evidence", mcnemar(0, 0) == 0.0)
    claim("an equal split is no evidence", mcnemar(7, 7) == 0.0)
    # The case `clean_fixes_needed` exists for: Yates' correction means four
    # clean fixes score 2.25 and do not clear a bar of 3.84, and six do.
    claim("four clean fixes do not clear the bar", mcnemar(0, 4) < MCNEMAR_95)
    claim("and six do", mcnemar(0, 6) >= MCNEMAR_95)
    claim("four clean fixes score exactly 2.25",
          abs(mcnemar(0, 4) - 2.25) < 1e-6)
    claim("the statistic does not care which side won",
          mcnemar(3, 12) == mcnemar(12, 3))

    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    (d / "a.tsv").write_text(
        "# gsm8k\tqid\twant\tgot\thit\tntok\n"
        "aaa\t1\t1\t1\t10\nbbb\t2\t9\t0\t10\nccc\t3\t3\t1\t10\n",
        encoding="utf-8")
    (d / "b.tsv").write_text(
        "# gsm8k\tqid\twant\tgot\thit\tntok\n"
        "bbb\t2\t2\t1\t10\nccc\t3\t8\t0\t10\nzzz\t4\t4\t1\t10\n",
        encoding="utf-8")
    task, rows = read(d / "a.tsv")
    claim("the task comes off the header", task == "gsm8k")
    claim("a run is read as qid -> outcome", len(rows) == 3 and rows["aaa"][0])
    chi = report(str(d / "a.tsv"), str(d / "b.tsv"))
    claim("one fixed and one broken is no evidence", chi == 0.0)

    print("  all claims hold" if ok else "  CLAIMS FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?", default="")
    ap.add_argument("b", nargs="?", default="")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many of the questions that moved, each way")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.a or not args.b:
        raise SystemExit("  usage: paired.py <a.tsv> <b.tsv> [--show N]")
    report(args.a, args.b, args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
