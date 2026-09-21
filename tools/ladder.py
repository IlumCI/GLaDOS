#!/usr/bin/env python3
"""The ladder between a north star and a night's work.

### What was missing

The loop had a judge and no destination. `godel.py next` ranks axes by which
one's verdict the ledger can least predict, which is a good answer to "what
should I measure" and no answer at all to "what am I for". The kernel-side
machine already showed where that ends: "search space exhausted" was the end
of self-improvement, eight points and then nothing, every night forever.

So the operator writes **one line** -- `loop/goal.txt` -- and the machine
writes the rungs. A rung is a milestone with a witness: something that is not
true today, stated so that a check can fail on the baseline arm and pass on
the candidate.

### Three rules, and each exists because of a way this could be a lie

**A rung is met by the ledger, never by a field.** Nothing here records
"done". A rung's `point` is the sha256 of its own declaration -- which is how
a proposal has always been identified in this tree, so the two are the same
object seen twice -- and it is met when an *adopted* certificate carries that
point. Since a witnessed kind's J1 is the witness itself, "met" means a check
genuinely failed before the patch and passed after it. There is no field
anywhere the loop can write its own success into.

**A rung must name a witnessed kind.** `bugfix` and `test` carry
`witness=True`, so their J1 is fail-then-pass. `feature` does not: it claims
`rail: none`, and `rail none` refuses. A milestone filed under an unwitnessed
kind is a milestone with no judge, which is the arrangement `godel.rs` opens
by warning about, so it is refused at admission rather than discovered at
three in the morning.

**A rung names the goal it was written for.** The goal's sha256 travels in
every rung. Rewriting the north star does not re-aim a ladder built for
something else -- it orphans it, visibly, and `progress` says so. Same rule
the corpus identity has, and for the same reason: evidence gathered for one
question is not evidence for another.

### What this does not do

It does not write rungs. That is rung 4's job and it is the next thing to
build; this is the shape that has to exist first, so that when a model
proposes a milestone there is something to refuse it with.

    python3 tools/ladder.py goal
    python3 tools/ladder.py list
    python3 tools/ladder.py next --emit-env /tmp/rung.env
    python3 tools/ladder.py progress
    python3 tools/ladder.py admit some.rung
    python3 tools/ladder.py --selftest
"""

import argparse
import hashlib
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A rung's declaration. Closed, and ordered, because the point is a hash of
#: the rendering and a rendering with a free field order is several hashes.
RUNG_KEYS = ("seq", "goal", "kind", "title", "witness", "why")

#: The kinds a milestone may be filed under: exactly those whose J1 is the
#: witness. Read off `godel.KINDS` rather than written down here, so a kind
#: that stops being witnessed cannot leave a stale copy behind.
def witnessed_kinds():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    return tuple(sorted(
        n for n, k in godel.KINDS.items() if k.enabled and k.witness))


HEX64 = re.compile(r"^[0-9a-f]{64}$")
SLUG = re.compile(r"^[0-9]{4}-[a-z0-9-]+\.rung$")


class Bad(Exception):
    """A rung this ladder will not climb, with the reason."""


# ------------------------------------------------------------------ the goal


def goal_text(root):
    p = os.path.join(root, "loop", "goal.txt")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read()


def goal_hash(root):
    """The identity. `None` when there is no north star, which is a state the
    loop is allowed to be in -- it simply has no ladder."""
    t = goal_text(root)
    return hashlib.sha256(t.encode("utf-8")).hexdigest() if t is not None else None


def north_star(root):
    for line in (goal_text(root) or "").split("\n"):
        if line.startswith("north-star "):
            return line[len("north-star "):].strip()
    return None


# ----------------------------------------------------------------- the rungs


def render_rung(r):
    """The bytes a rung's point is taken over. Comments are not part of it."""
    out = ["looprung 1"]
    for k in RUNG_KEYS:
        if k not in r:
            raise Bad(f"the rung has no {k!r}")
        out.append(f"{k} {r[k]}")
    return "\n".join(out) + "\n"


def rung_point(r):
    return hashlib.sha256(render_rung(r).encode("utf-8")).hexdigest()


def parse_rung(text):
    lines = [l for l in text.split("\n") if l != "" and not l.startswith("#")]
    if not lines or lines[0] != "looprung 1":
        raise Bad("that is not a ladder rung")
    r = {}
    for line in lines[1:]:
        k, _, v = line.partition(" ")
        if k in r:
            raise Bad(f"{k!r} appears twice")
        if k not in RUNG_KEYS:
            raise Bad(f"{k!r} is not a rung field")
        r[k] = v.strip()
    for k in RUNG_KEYS:
        if k not in r:
            raise Bad(f"the rung has no {k!r}")
    if not r["seq"].isdigit() or int(r["seq"]) < 1:
        raise Bad("seq is not a positive number")
    if not HEX64.match(r["goal"]):
        raise Bad("goal is not a sha256")
    for k in ("title", "witness", "why"):
        if not r[k]:
            raise Bad(f"{k} is empty, and a rung nobody can read is not a rung")
    return r


def admit(r, root):
    """Everything that must hold before a rung is worth a night."""
    kinds = witnessed_kinds()
    if r["kind"] not in kinds:
        raise Bad(
            "kind %r has no witness, so this milestone would have no judge; "
            "the witnessed kinds are %s" % (r["kind"], ", ".join(kinds)))
    g = goal_hash(root)
    if g is None:
        raise Bad("there is no north star, so this rung is aimed at nothing")
    if r["goal"] != g:
        raise Bad(
            "this rung was written for goal %s and the north star is now %s"
            % (r["goal"][:12], g[:12]))
    return r


def rung_files(root):
    d = os.path.join(root, "loop", "ladder")
    if not os.path.isdir(d):
        return []
    out = []
    for n in sorted(os.listdir(d)):
        if n.endswith(".rung"):
            if not SLUG.match(n):
                raise Bad("%s is not named NNNN-slug.rung" % n)
            out.append(os.path.join(d, n))
    return out


def load(root, strict=True):
    """Every rung, in order, with its point. Refuses a ladder with holes."""
    rungs = []
    for p in rung_files(root):
        with open(p, encoding="utf-8") as f:
            r = parse_rung(f.read())
        if strict:
            admit(r, root)
        r["point"] = rung_point({k: r[k] for k in RUNG_KEYS})
        r["file"] = os.path.relpath(p, root).replace(os.sep, "/")
        rungs.append(r)
    seqs = [int(r["seq"]) for r in rungs]
    if seqs != list(range(1, len(seqs) + 1)):
        raise Bad(
            "the rungs are %s, which is not 1..%d -- a gap means one was "
            "removed, and a ladder that can lose a rung can lose the one it "
            "failed" % (seqs, len(seqs)))
    return rungs


# ----------------------------------------------- met, read out of the ledger


def adopted_points(root):
    """Points carried by adopted certificates. The only source of 'met'."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    d = os.path.join(root, "loop", "ledger", "entries")
    if not os.path.isdir(d):
        return set()
    out = set()
    for n in sorted(os.listdir(d)):
        if not n.endswith(".cert"):
            continue
        with open(os.path.join(d, n), encoding="utf-8") as f:
            c = godel.parse_cert(f.read())
        if c["verdict"] == "adopt":
            out.add(c["point"])
    return out


def state(root):
    rungs = load(root)
    met = adopted_points(root)
    for r in rungs:
        r["met"] = r["point"] in met
    return rungs


# ------------------------------------------------------------------ commands


def cmd_goal(root):
    g = goal_hash(root)
    if g is None:
        print("  there is no north star")
        return 1
    print("  north star: %s" % north_star(root))
    print("  identity:   %s" % g)
    return 0


def cmd_list(root):
    rungs = state(root)
    if not rungs:
        print("  the ladder is empty -- nothing has proposed a rung yet")
        return 0
    for r in rungs:
        print("  %s %s  %-9s %s" % (
            "met " if r["met"] else "open", r["seq"], r["kind"], r["title"]))
        print("       point %s  witness %s" % (r["point"][:16], r["witness"]))
    return 0


def cmd_progress(root):
    """Derived, every time. Nothing stores this."""
    rungs = state(root)
    met = sum(1 for r in rungs if r["met"])
    print("  %d of %d rung(s) met" % (met, len(rungs)))
    if not rungs:
        print("  and that is not 100%: a ladder with no rungs is not a "
              "goal reached, it is a goal nobody has decomposed yet")
        return 0
    nxt = next((r for r in rungs if not r["met"]), None)
    print("  next: %s" % (nxt["title"] if nxt else
                          "nothing open -- every declared rung is met"))
    return 0


def cmd_next(root, emit_env=None):
    rungs = state(root)
    nxt = next((r for r in rungs if not r["met"]), None)
    if nxt is None:
        print("  no open rung" if rungs else "  the ladder is empty")
        return 2
    print("  rung %s  %s" % (nxt["seq"], nxt["title"]))
    print("  kind %s  point %s" % (nxt["kind"], nxt["point"]))
    print("  witness %s" % nxt["witness"])
    if emit_env:
        with open(emit_env, "w", encoding="utf-8", newline="\n") as f:
            f.write("RUNG_SEQ=%s\n" % nxt["seq"])
            f.write("RUNG_POINT=%s\n" % nxt["point"])
            f.write("RUNG_KIND=%s\n" % nxt["kind"])
            f.write("RUNG_TITLE=%s\n" % nxt["title"])
            f.write("RUNG_WITNESS=%s\n" % nxt["witness"])
    return 0


def cmd_admit(root, path):
    with open(path, encoding="utf-8") as f:
        r = parse_rung(f.read())
    admit(r, root)
    print("  admitted: %s" % r["title"])
    print("  point %s" % rung_point({k: r[k] for k in RUNG_KEYS}))
    return 0


# ----------------------------------------------------------------- selftest


def _rung(goal, seq=1, kind="test", title="a thing", witness="w", why="y"):
    return "\n".join([
        "looprung 1", "seq %d" % seq, "goal %s" % goal, "kind %s" % kind,
        "title %s" % title, "witness %s" % witness, "why %s" % why]) + "\n"


def selftest():
    claims = []

    def claim(ok, what):
        claims.append(bool(ok))
        print("  %-4s %s" % ("ok" if ok else "FAIL", what))

    def refuses(fn, needle, what):
        try:
            fn()
            claim(False, what + " (it was accepted)")
        except Bad as e:
            claim(needle in str(e), what)

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "loop", "ladder"))
        with open(os.path.join(tmp, "loop", "goal.txt"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write("goal 1\nnorth-star a test star\n")
        g = goal_hash(tmp)
        claim(HEX64.match(g or ""), "the goal has a sha256 identity")
        claim(north_star(tmp) == "a test star", "and the north star reads back")

        # The point is the rendering's hash, which is what makes a rung and a
        # proposal the same object. Two rungs differing by one character are
        # two points, or the ladder could not tell them apart.
        a = parse_rung(_rung(g, title="alpha"))
        b = parse_rung(_rung(g, title="alphb"))
        claim(rung_point(a) != rung_point(b), "one character is a different point")
        claim(rung_point(a) == rung_point(parse_rung(render_rung(a))),
              "and a rung round-trips to the same point")

        ladder = os.path.join(tmp, "loop", "ladder")
        with open(os.path.join(ladder, "0001-first.rung"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(_rung(g, 1, title="first"))
        claim(len(load(tmp)) == 1, "a well-formed ladder loads")
        claim(state(tmp)[0]["met"] is False,
              "and a rung with no adopted certificate is open")

        # **The canary.** Every refusal below is a way this could have been a
        # lie, and a checker that has never refused anything is one that reads
        # nothing.
        refuses(lambda: admit(parse_rung(_rung(g, kind="feature")), tmp),
                "has no witness",
                "an unwitnessed kind is refused, so no milestone lacks a judge")
        refuses(lambda: admit(parse_rung(_rung("0" * 64)), tmp),
                "was written for goal",
                "a rung aimed at another north star is refused")
        refuses(lambda: parse_rung(_rung(g, title="")), "is empty",
                "a rung with no title is refused")
        refuses(lambda: parse_rung("looprung 1\nseq 1\n"), "has no",
                "a rung missing a field is refused")
        refuses(lambda: parse_rung(_rung(g) + "seq 2\n"), "appears twice",
                "a duplicated field is refused")

        with open(os.path.join(ladder, "0003-third.rung"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(_rung(g, 3, title="third"))
        refuses(lambda: load(tmp), "which is not 1..",
                "a ladder with a hole in it is refused")
        os.remove(os.path.join(ladder, "0003-third.rung"))

        # And the one that matters most: met comes from an adopted
        # certificate and from nowhere else, so the ladder cannot be
        # advanced by editing the ladder.
        entries = os.path.join(tmp, "loop", "ledger", "entries")
        os.makedirs(entries)
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import godel
        r = state(tmp)[0]
        c = {k: "-" for k in godel.CERT_KEYS}
        c.update({
            "seq": "1", "utc": "2026-01-01T00:00:00Z", "point": r["point"],
            "kind": "test", "rung": "4", "axis": "model",
            "parent-tree": "0" * 40, "candidate-tree": "1" * 40,
            "rail": "none", "corpus": "00000000", "alpha-k": "0",
            "chi-bar": "-", "minutes": "0", "half": "extend", "level": "0",
            "boots": "1", "queries": "0", "sections": "0/0", "suites": "0/0",
            "claims": "0/0", "witness": "fail-then-pass", "moved": "-",
            "why": "the witness failed then passed", "verdict": "refuse",
            "runner": "-",
        })
        with open(os.path.join(entries, "a.cert"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(godel.render_cert(c))
        claim(state(tmp)[0]["met"] is False,
              "a REFUSED certificate does not meet a rung")
        c["verdict"] = "adopt"
        with open(os.path.join(entries, "a.cert"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(godel.render_cert(c))
        claim(state(tmp)[0]["met"] is True,
              "and an adopted one with the rung's point does")

    print()
    if all(claims):
        print("  ladder passed (%d claims)" % len(claims))
        return 0
    print("  ladder FAILED")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", default="progress",
                    choices=("goal", "list", "next", "progress", "admit"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--emit-env")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    try:
        if a.cmd == "goal":
            return cmd_goal(a.root)
        if a.cmd == "list":
            return cmd_list(a.root)
        if a.cmd == "next":
            return cmd_next(a.root, a.emit_env)
        if a.cmd == "admit":
            if not a.path:
                print("  admit wants a rung file"); return 2
            return cmd_admit(a.root, a.path)
        return cmd_progress(a.root)
    except Bad as e:
        print("::error::%s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
