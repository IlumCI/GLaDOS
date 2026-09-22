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

**A rung must name a kind whose J1 can say yes without a prior reading.**
`bugfix` and `test` carry `witness=True`, so their J1 is fail-then-pass;
`feature`'s is the boot's own claim count. `tune`, `cleanup` and `rewrite`
are excluded not for lacking a judge but for needing a rail that already
reads something, and a rung creates what nothing was measuring. A milestone
filed under a kind whose J1 cannot reach it is a milestone with no judge,
which is the arrangement `godel.rs` opens by warning about, so it is refused
at admission rather than discovered at three in the morning.

A witnessed kind also needs a target that already exists, and that rule cost
three nights to learn: the witness arm builds against the PARENT tree, so a
file being created is absent from it and the arm dies of infrastructure.
Creating a thing is `feature`; the witnessed rungs come after it, once there
is something to be wrong about.

**A rung names the goal it was written for.** The goal's sha256 travels in
every rung. Rewriting the north star does not re-aim a ladder built for
something else -- it orphans it, visibly, and `progress` says so. Same rule
the corpus identity has, and for the same reason: evidence gathered for one
question is not evidence for another.

### Who writes the rungs

The machine does, and that is the point: the operator's whole input is the
one line in `loop/goal.txt`. `propose` asks a model **running inside the CI
job** -- a ternary Bonsai 4B served by `llama-server`, no credential of any
kind -- for the next milestone, under the same shape `godel.author` uses for
patches: a system prompt from `tools/prompts/`, a structured card as the user
turn, and exactly one fenced block back.

(It asked GitHub Models until that was retired mid-loop. The endpoint was
chosen for "the workflow's own token, no new credential"; a model in the job
keeps that property without depending on anybody's service staying up.)

The card carries the north star, the rungs so far, the kinds it may name
with their line budgets, the surfaces no milestone may target, and a *listing* of
the modules that exist. A listing rather than source, because a milestone is
chosen from what is there rather than written against whatever one file
happens to say -- and because a listing is a far smaller injection surface
than a file the loop may itself have written.

Everything the reply could do is refused somewhere. Prose outside the fence,
two fences, none; a kind with no reachable J1; a goal hash that is not the
one; a sequence number that skips ahead; a line trying to declare a field the
format does not have -- including `point`, which is the one a reply would
forge to mark itself met. None of that is trusted and then checked; it is
checked before it is a rung.

A refusal is the ordinary outcome, not an error. The contract held, the night
records it and moves on. Retrying with more context is how a fence contract
stops being one.

    python3 tools/ladder.py goal
    python3 tools/ladder.py list
    python3 tools/ladder.py next --emit-env /tmp/rung.env
    python3 tools/ladder.py progress
    python3 tools/ladder.py propose        # GITHUB_TOKEN, models: read
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
RUNG_KEYS = ("seq", "goal", "kind", "target", "title", "witness", "why")

#: The kinds a milestone may be filed under: exactly those whose J1 needs
#: no prior rail reading. Read off `godel.KINDS` rather than written down here, so a kind
#: whose J1 changes cannot leave a stale copy behind.
def judgeable_kinds():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    return tuple(sorted(
        n for n, k in godel.KINDS.items()
        if k.enabled and k.j1 in ("witness", "claims")))


def godel_kind(name):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    return godel.KINDS[name]


def goal_phrase(root):
    """The north star with its leading verb stripped, lowercased.

    `create a native video codec` names a thing to build; the thing is
    `a native video codec`. That noun phrase is what a rung must not simply
    repeat, so it is what `restates_the_goal` looks for.
    """
    star = (north_star(root) or "").strip().lower()
    for verb in ("create ", "build ", "write ", "make ", "implement ",
                 "add ", "design "):
        if star.startswith(verb):
            return star[len(verb):].strip()
    return star


def restates_the_goal(title, root):
    """True when a rung's title is the north star wearing a longer coat.

    **Measured, and it is the whole reason the ladder was not decomposing.**
    Every rung the 4B wrote for `create a native video codec` was titled
    `a native video codec that encodes a 4x4 pixel buffer to a 4x4 pixel
    buffer` -- the goal restated with a size bolted on, twice, under two
    different kinds and two different targets. The prompt already said "it
    must be small" and "split it and propose the first half", in prose,
    which this tree has now measured three separate models ignoring.

    So it is a refusal rather than advice: a title carrying the goal's own
    noun phrase is not a part of the goal, it is the goal. What a part
    looks like is a zigzag order, a quantisation table, a bit writer -- none
    of which contain the phrase, and all of which are one file of work.

    Deliberately a containment test and not an overlap fraction. A
    fraction needs a threshold nobody can defend, and the failure being
    caught is exact: the phrase appears whole.
    """
    phrase = goal_phrase(root)
    if not phrase:
        return False
    flat = " ".join(title.lower().split())
    return phrase in flat


def nothing_built(rungs):
    """True while no rung has been met.

    **The first rung toward a north star cannot be witnessed, and that is
    arithmetic rather than taste.** A witness must fail against the tree as
    it stands *because of the goal*; with nothing of the goal built, every
    file that exists belongs to something else, so the only witnessed rung
    available is one aimed at a file the goal has never touched. Such a rung
    is admissible, spends a runner, and is refused -- three times, before it
    retires, and then the next one is drawn from the same distribution.

    Measured rather than assumed: the 4B chose `test` against
    `src/fmt/outline.rs` five times out of five, deterministically, with a
    witness whose own text says "the struct does not exist". It is not
    confused. It is being offered a choice that cannot be right yet.

    So while nothing is built only the creating kind is offered, and the
    restriction lifts the moment one rung is met -- from then on there is
    something of the goal's to be wrong about, which is exactly when a
    witness starts being able to fail for the right reason.
    """
    return not any(r.get("met") for r in rungs)


def kinds_open(rungs):
    """The kinds a rung may name, given what the ladder has already met."""
    ks = judgeable_kinds()
    if rungs is not None and nothing_built(rungs):
        return tuple(k for k in ks if not godel_kind(k).witness)
    return ks


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


def admit(r, root, rungs=None):
    """Everything that must hold before a rung is worth a night.

    `rungs` is what the ladder has met so far, and it narrows the kinds --
    see `nothing_built`. It is optional because `load` re-admits rungs
    written under earlier rules, and a rung already on the ladder must not
    become stale for a restriction that did not exist when it was written.
    """
    kinds = kinds_open(rungs)
    if r["kind"] not in kinds:
        raise Bad(
            "kind %r cannot be judged here yet; the kinds you may name are "
            "%s" % (r["kind"], ", ".join(kinds)))
    g = goal_hash(root)
    if g is None:
        raise Bad("there is no north star, so this rung is aimed at nothing")
    if r["goal"] != g:
        raise Bad(
            "this rung was written for goal %s and the north star is now %s"
            % (r["goal"][:12], g[:12]))

    # The target, against the kind's own scope rather than a second copy of
    # it. `target` exists because the author slices source from a file and a
    # codec's module does not exist yet -- so the rung has to say where the
    # work lands, and saying it is what makes it checkable.
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    t = r["target"]
    if t.startswith("/") or ".." in t.split("/") or "\\" in t:
        raise Bad("the target %r reaches outside the tree" % t)
    masks = godel.KINDS[r["kind"]].masks
    if not any(t.startswith(pre) for pre in masks):
        raise Bad("the target %r is outside the %s kind's masks %s"
                  % (t, r["kind"], list(masks)))
    if any(t.startswith(pre) for pre in godel.EVALUATOR):
        raise Bad("the target %r is evaluator machinery, which the loop may "
                  "not aim at" % t)

    # **And it must be somewhere a verdict can be reached.** `src/gfx/` and
    # `src/port/` are unjudgeable: screenshots are captured and never
    # compared, so a change there builds, boots, reads `same` on every rail
    # there is, and would be adopted having checked nothing about the only
    # thing it altered. `knob.UNJUDGEABLE` is that list and `admit` in
    # `godel.py` already refuses a patch aimed at one.
    #
    # It was refusing them a step too late. The first rung the decomposer
    # wrote for a video codec targeted `src/gfx/video.rs`, which is exactly
    # the plausible-looking wrong answer -- a codec is data in and data out
    # and perfectly judgeable, but not from inside the graphics tree. The
    # refusal arrived after a model had written the file. Here it arrives
    # before a night is spent.
    import knob
    if any(t.startswith(pre) for pre in knob.UNJUDGEABLE):
        raise Bad("the target %r is an unjudgeable surface %s -- nothing "
                  "there can be compared, so no witness could settle it"
                  % (t, list(knob.UNJUDGEABLE)))

    # **A witnessed kind needs a target that already exists**, and this is
    # the rule whose absence cost three nights. The witness arm builds the
    # witness against the PARENT tree; a file being created is absent from
    # it, so the arm dies of infrastructure and `loop-judge.yml` refuses by
    # name. Worse, the refusal never arrives: `create_finish` carries no
    # witness, so `godel.admit` refuses locally, no certificate is filed,
    # and `retired` -- which counts refused certificates -- never moves. A
    # rung like that is not slow, it is stuck, and it blocks every rung
    # behind it for good.
    #
    # Greenfield work is what `feature` is for. Naming it here is what
    # keeps a milestone that creates something from being filed under a
    # judge that cannot reach it.
    # The other half of the same rule. A rung drives `godel.create`, which
    # builds a `--- /dev/null` diff, so a `feature` naming a file that is
    # already there produces a patch `git apply` refuses -- locally, with no
    # certificate filed and therefore nothing that would ever retire it.
    # Same stuck shape as the witnessed case, arriving from the other side.
    if restates_the_goal(r["title"], root):
        raise Bad(
            "the title repeats the north star (%r) rather than naming one "
            "part of it -- a rung is a component, and the ladder is how the "
            "parts add up" % goal_phrase(root))
    if not godel.KINDS[r["kind"]].witness and os.path.exists(
            os.path.join(root, t)):
        raise Bad(
            "%r creates a file and %r already exists -- to change what is "
            "there, name a witnessed kind whose check fails on it today"
            % (r["kind"], t))
    if godel.KINDS[r["kind"]].witness and not os.path.exists(
            os.path.join(root, t)):
        raise Bad(
            "%r is a witnessed kind and %r does not exist yet -- a witness "
            "runs against the parent tree, so nothing there could fail. "
            "Creating a file is 'feature'; witnessed kinds come after it"
            % (r["kind"], t))
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
        r["stale"] = ""
        if strict:
            # **A rung `admit` would no longer take is stale, not fatal.**
            # It was a raise, which killed the whole read -- so one rung
            # written before a rule tightened took the ladder down with it,
            # and `next` exited 1 where the night needed it to say "nothing
            # open" and go stock another. `propose` still calls `admit`
            # directly and still raises, which is where a refusal belongs:
            # before a rung is written, never after.
            try:
                admit(r, root)
            except Bad as why:
                r["stale"] = str(why)
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


#: How many refused certificates retire a rung.
#:
#: **Nothing abandoned a rung before this, and that is the loop's own oldest
#: failure in a new costume.** `godel.rs` records where an undirected search
#: ends -- "search space exhausted was the end of self-improvement, eight
#: points and then nothing, every night forever". A ladder whose first rung
#: cannot be built has the same shape: every night proposes it, every night
#: is refused, and the loop looks busy forever.
#:
#: Three, because a witness can fail for reasons that are not the rung's --
#: a flaky boot, a runner without KVM, a night that died in the judge. One
#: refusal is weather. Three is the rung.
RETIRE_AFTER = 3


def point_verdicts(root):
    """Every verdict each point has collected, adopted or not."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    d = os.path.join(root, "loop", "ledger", "entries")
    out = {}
    if not os.path.isdir(d):
        return out
    for n in sorted(os.listdir(d)):
        if not n.endswith(".cert"):
            continue
        with open(os.path.join(d, n), encoding="utf-8") as f:
            c = godel.parse_cert(f.read())
        out.setdefault(c["point"], []).append(c["verdict"])
    return out


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


def balked_dir(root):
    return os.path.join(root, "loop", "ladder", "balked")


def balked(root, point):
    """How many nights the author could not write this rung at all.

    **The third way a rung gets stuck, and it is the same shape as the
    other two.** `create` decodes at temperature 0, so its first attempt
    against an unchanged card is identical every night; if all its
    attempts fail the night files no certificate, `refused` never moves,
    and the rung blocks every rung behind it for good. The other two were
    a raise that killed the read, and a refusal earlier than the
    certificate; this one is a refusal earlier than the runner.

    Counted in files rather than derived, unlike `met` and `refused`, and
    the difference is honest: those are properties of the ledger, which
    the loop cannot forge, while this records something that happened here
    and nowhere else. It can only ever retire a rung and never mark one
    met, so the property that matters -- that nothing the loop writes can
    declare its own success -- is untouched.
    """
    d = os.path.join(balked_dir(root), point[:16])
    if not os.path.isdir(d):
        return 0
    return len([n for n in os.listdir(d) if not n.startswith(".")])


def note_balk(root, point, why):
    """Record that the author could not write this rung tonight."""
    d = os.path.join(balked_dir(root), point[:16])
    os.makedirs(d, exist_ok=True)
    n = "%04d.txt" % (balked(root, point) + 1)
    with open(os.path.join(d, n), "w", encoding="utf-8", newline="\n") as f:
        f.write(why.rstrip("\n") + "\n")
    return os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")


def state(root):
    """Each rung with `met` and `retired`, both derived from the ledger.

    Neither is stored, for the reason `met` was never stored: a field the
    loop can write is a field the loop can write its own success into. A
    rung is met when an adopted certificate carries its point and retired
    when `RETIRE_AFTER` refused ones do -- both are counts over the record,
    recomputed on every call, and a certificate cannot be forged without
    also passing `fsck`.
    """
    rungs = load(root)
    met = adopted_points(root)
    verdicts = point_verdicts(root)
    for r in rungs:
        r["met"] = r["point"] in met
        refused = sum(1 for v in verdicts.get(r["point"], []) if v == "refuse")
        r["refused"] = refused
        # Met wins over retired: a rung that was eventually built is built,
        # however many nights it cost to get there.
        # Stale retires too, and it has to: `refused` counts *certificates*,
        # so it only moves for a rung that reached a runner. A rung refused
        # earlier than that -- by `godel.admit`, before an envelope exists --
        # files nothing, never retires, and blocks every rung behind it for
        # good. Exactly one was in that state.
        r["balked"] = balked(root, r["point"])
        r["retired"] = (not r["met"]) and (
            refused >= RETIRE_AFTER or bool(r["stale"])
            or r["balked"] >= RETIRE_AFTER)
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
        mark = "met " if r["met"] else ("gone" if r["retired"] else "open")
        print("  %s %s  %-9s %s" % (mark, r["seq"], r["kind"], r["title"]))
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
    # Two ways to retire and they are different facts: a rung the judges
    # refused three times was tried and lost, a stale one was never tried at
    # all. One line for both would read as evidence that does not exist.
    spent = sum(1 for r in rungs if r["retired"] and not r["stale"]
                and r.get("balked", 0) < RETIRE_AFTER)
    stale = sum(1 for r in rungs if r["retired"] and r["stale"])
    if spent:
        print("  %d retired after %d refusals each" % (spent, RETIRE_AFTER))
    if stale:
        print("  %d retired without being tried, the rules having tightened"
              % stale)
    balk = sum(1 for r in rungs
               if r["retired"] and r.get("balked", 0) >= RETIRE_AFTER)
    if balk:
        print("  %d retired because the author could not write them, %d "
              "night(s) each" % (balk, RETIRE_AFTER))
    nxt = open_rung(rungs, root)
    for r in rungs:
        if r.get("stale"):
            print("  rung %s retired: %s" % (r["seq"], r["stale"]))
    print("  next: %s" % (nxt["title"] if nxt else
                          "nothing open -- every rung is met or retired"))
    return 0


def open_rung(rungs, root=None):
    """The rung a night should aim at: the first neither met nor retired.

    `root` is accepted and unused: `load` is what decides staleness now, so
    every caller of `state` already has it folded in. Kept so the callers
    that pass it keep reading as the question they are asking.

    **A rung `admit` would no longer take is retired too, and that is not
    tidiness.** `retired` counts refused *certificates*, so it only ever
    moves for a rung that reached a runner. A rung refused earlier than that
    -- by `godel.admit`, before an envelope is built -- files nothing, never
    retires, and blocks every rung behind it for good. Exactly one is in the
    ledger: a `test` rung aimed at a file that does not exist, written
    before the rule that now refuses it.

    So a rule tightened after a rung was written retires that rung instead
    of deadlocking on it, and the reason is printed rather than inferred
    from a ladder that silently stopped moving. `root` is optional because
    `state` is read in places with no tree to check against; passing it is
    what turns the check on.
    """
    return next((r for r in rungs if not r["met"] and not r["retired"]), None)


def cmd_next(root, emit_env=None):
    rungs = state(root)
    nxt = open_rung(rungs, root)
    for r in rungs:
        if r.get("stale"):
            print("  rung %s retired: %s" % (r["seq"], r["stale"]))
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
            f.write("RUNG_TARGET=%s\n" % nxt["target"])
            f.write("RUNG_TITLE=%s\n" % nxt["title"])
            f.write("RUNG_WITNESS=%s\n" % nxt["witness"])
            # **Which author entry point this rung wants.** A witnessed kind
            # changes a file that exists, which is a patch, so `author`; an
            # unwitnessed one makes a file that does not, so `create`, which
            # asks for contents and builds the diff itself. Decided here
            # rather than in the workflow because the rule is one line of
            # Python about the kind table and several of shell about a
            # string.
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            import godel
            f.write("RUNG_VERB=%s\n"
                    % ("author" if godel.KINDS[nxt["kind"]].witness
                       else "create"))
    return 0


def cmd_admit(root, path):
    with open(path, encoding="utf-8") as f:
        r = parse_rung(f.read())
    admit(r, root)
    print("  admitted: %s" % r["title"])
    print("  point %s" % rung_point({k: r[k] for k in RUNG_KEYS}))
    return 0


# ------------------------------------------------- the decomposer, rung 4b
#
# The operator writes one line. Everything below is how the machine turns it
# into rungs without anybody's help -- which is the whole point, and is also
# the part with the most ways to be a lie, so each one is refused here rather
# than discovered in a ledger.

FENCE = re.compile(r"```rung\n(.*?)```", re.S)


def judgeable_dirs(root):
    """Directories under `src/` a milestone may target.

    Shared by the card and the grammar, because those are two statements of
    one fact and the first version let them disagree: the card listed every
    module including `src/gfx/`, then forbade `src/gfx/` a few lines later,
    and the model duly chose a target there twice. Offering a thing and
    banning it is not a rule, it is a contradiction, and a 4B resolves it by
    following the association rather than the prohibition -- "video codec"
    reaches for the graphics tree.
    """
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import knob
    src = os.path.join(root, "src")
    if not os.path.isdir(src):
        return []
    out = []
    for n in sorted(os.listdir(src)):
        if not os.path.isdir(os.path.join(src, n)):
            continue
        if any(("src/%s/" % n).startswith(pre) for pre in knob.UNJUDGEABLE):
            continue
        out.append(n)
    return out


def rung_card(root, rungs):
    """Structured data only. The tree's shape is a listing, never a slice of
    source, because a milestone is chosen from what exists rather than
    written against what one file happens to say."""
    kinds = kinds_open(rungs)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    import knob
    lines = [
        "north star: %s" % north_star(root),
        "goal hash: %s" % goal_hash(root),
        "next seq: %d" % (len(rungs) + 1),
        "kinds you may choose from:",
    ]
    for n in kinds:
        k = godel.KINDS[n]
        # **The card says what each kind is FOR**, not only what it costs.
        # A budget alone left the model picking `test` for a file that does
        # not exist yet, every time -- a refusal `admit` makes correctly and
        # a rung nobody could have written correctly from what they were
        # told. The rule is one sentence and it belongs where the choice is
        # made rather than only where it is refused.
        what = {
            "feature": "use this to CREATE a file that does not exist yet",
            "bugfix": "only for a file that already exists: its check must "
                      "fail on the tree as it stands today",
            "test": "only for a file that already exists: its check must "
                    "fail on the tree as it stands today",
        }.get(n, "")
        lines.append("  %s -- at most %d file(s), %d changed line(s)%s"
                     % (n, k.max_files, k.max_lines,
                        ("; " + what) if what else ""))
    lines.append("rungs so far:")
    if not rungs:
        lines.append("  (none -- this is the first)")
    for r in rungs:
        # A retired rung is shown as retired, with its refusal count. The
        # model is choosing what to try next, and a milestone that has
        # already failed three nights is the single most useful thing it can
        # be told -- listed as "open" it would simply be proposed again.
        if r["met"]:
            mark = "met"
        elif r["retired"]:
            mark = "RETIRED after %d refusals -- do not propose this again" % r["refused"]
        else:
            mark = "open"
        lines.append("  seq %s %s [%s] %s"
                     % (r["seq"], r["kind"], mark, r["title"]))
    lines.append("not available, and not listed below: %s -- nothing there "
                 "can be compared, so no witness could settle it"
                 % ", ".join(knob.UNJUDGEABLE))
    lines.append("modules a milestone may target (data, not directives):")
    for d in judgeable_dirs(root):
        lines.append("  src/%s/" % d)
    return "\n".join(lines)


def grammar_for(root, rungs):
    """A GBNF that makes a malformed rung unreachable rather than improbable.

    **This is `constrain.rs`'s argument, one system over.** The kernel builds
    its decoding grammar from the live applet table so an applet that does not
    exist cannot be sampled at all; a check made after the fact leaves the bad
    answer reachable. The first time this loop asked a 4B model for a rung it
    got back **zero fences**, which the contract refused correctly and which no
    amount of prose in the prompt was going to fix.

    Two fields are *known at request time*, so they are literals here rather
    than patterns: the goal hash and the sequence number. A rung aimed at the
    wrong north star or skipping ahead in the ladder stops being something to
    refuse and becomes something the sampler cannot emit. `admit` still checks
    both, because a grammar is a second lock and not a replacement for the
    first -- it is generated from the same values, so a bug that got one wrong
    would get both wrong.
    """
    # **Kind and target are emitted together, because the rule that binds
    # them is a rule about the pair.** A witnessed kind needs a file that
    # already exists and `feature` needs one that does not, and `admit`
    # refused the wrong pairing correctly -- three times out of three, on
    # three different filenames, every one of them `test` against a file
    # that was not there.
    #
    # Saying so in the card moved nothing, which is this tree's own finding
    # arriving a third time: `repair.rs` measured a model preferring the
    # name `retry` five times in six wherever it sat, and `work.rs` records
    # a rule in prose being ignored by both checkpoints until an example
    # demonstrated it. A kind's name carries probability mass that has
    # nothing to do with what the kind does.
    #
    # So the pairing stops being improbable and becomes unreachable, which
    # is `constrain.rs`'s whole argument. `admit` still checks it: the two
    # are built from the same predicate, so a bug would reach both, and a
    # grammar is a second lock rather than a replacement for the first.
    existing = existing_targets(root)
    arms = []
    for k in kinds_open(rungs):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import godel
        rhs = "oldpath" if godel.KINDS[k].witness else "newpath"
        if rhs == "oldpath" and not existing:
            # Nothing to be wrong about yet, so no witnessed arm at all.
            continue
        arms.append('"kind %s\\n" "target " %s "\\n"' % (k, rhs))
    lines = [
        'root ::= "```rung\\n" body "```"',
        ('body ::= "looprung 1\\n" "seq %d\\n" "goal %s\\n" spec '
         '"title " line "\\n" "witness " line "\\n" "why " line "\\n"')
        % (len(rungs) + 1, goal_hash(root)),
        "spec ::= %s" % " | ".join(arms),
        # **The directory is an alternation over judgeable modules**, so a
        # target in `src/gfx/` stops being something to refuse and becomes
        # something the sampler cannot emit. `admit` still checks it, because
        # both are built from `knob.UNJUDGEABLE` and one bug would reach both.
        'newpath ::= "src/" dir "/" stem ".rs"',
        "dir ::= %s" % " | ".join('"%s"' % d for d in judgeable_dirs(root)),
        'stem ::= [a-z0-9_]+',
        # One line, and never empty: a rung nobody can read is not a rung.
        'line ::= [^\\n]+',
    ]
    if existing:
        lines.insert(4, "oldpath ::= %s"
                     % " | ".join('"%s"' % t for t in existing))
    return "\n".join(lines) + "\n"


def existing_targets(root):
    """Every judgeable source file that is there to be wrong about.

    Enumerated rather than patterned, for `grammar_for`'s reason: a witness
    must fail against the tree as it stands, so the set of files it can name
    is exactly the set that exists. Bounded by the same masks and the same
    unjudgeable list the rest of admission uses.
    """
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import knob
    out = []
    for d in judgeable_dirs(root):
        base = os.path.join(root, "src", d)
        for n in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if not n.endswith(".rs"):
                continue
            t = "src/%s/%s" % (d, n)
            if any(t.startswith(pre) for pre in knob.UNJUDGEABLE):
                continue
            out.append(t)
    return tuple(out)


def propose_finish(root, reply, rungs=None):
    """The offline half, so the drills need no network.

    Split for the reason `godel.author_finish` is: the contract, the parse
    and every refusal are the interesting part, and a check that can only run
    with a credential is a check that does not run.
    """
    fences = FENCE.findall(reply)
    if len(fences) != 1:
        # **The reply travels with the refusal.** The first real run answered
        # "0 rung fence(s)" and that named the rule broken without showing
        # what was written instead, which is the difference between a refusal
        # somebody can act on and one they can only count. Bounded, because a
        # model with no grammar can run on for a while.
        preview = " ".join(reply.split())[:240]
        return None, ("%d rung fence(s) where the contract says exactly one"
                      " -- it said: %s" % (len(fences), preview or "(nothing)"))
    before, _, rest = reply.partition("```rung")
    _, _, after = rest.partition("```")
    if before.strip() or after.strip():
        return None, "content outside the fence"
    # The ladder is read BEFORE admission, because what has been met is
    # what decides which kinds are open -- see `nothing_built`.
    if rungs is None:
        rungs = state(root)
    try:
        r = parse_rung(fences[0])
        admit(r, root, rungs)
    except Bad as e:
        return None, str(e)
    want = len(rungs) + 1
    if int(r["seq"]) != want:
        return None, ("the rung claims seq %s where the ladder's next is %d"
                      % (r["seq"], want))
    return r, None


#: How many times the decomposer may be asked before the night gives up.
#:
#: **A refusal it can act on deserves another turn.** The contract, the
#: goal hash and the sequence number are all things a reply gets wrong in
#: ways the refusal names exactly, and this model has been measured
#: producing the same first answer every time against an unchanged card --
#: so one attempt is one sample from one distribution, and the card growing
#: is the only thing that moves it. Same shape `godel.create` uses, and the
#: same reason: three attempts of a 400-token decode is under a minute
#: against a night.
PROPOSE_TRIES = 3


def propose(root, token):
    """Ask for the next rung. Answers (rung, why)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import godel
    rungs = state(root)
    meta, system = godel.read_prompt("decompose.md")
    card = rung_card(root, rungs)
    grammar = grammar_for(root, rungs)
    why = "no attempt was made"
    for attempt in range(PROPOSE_TRIES):
        this = card if attempt == 0 else card + "\n" + "\n".join([
            "",
            "your last answer was refused:",
            "  " + why,
            "",
            "write the rung again, fixed. name one PART of the north star,",
            "never the north star itself.",
        ])
        try:
            reply = godel.ask_model(system, this, meta, token,
                                    "glados-loop-decompose", grammar=grammar)
        except godel.NoInference as e:
            # Not a traceback, and not a verdict either. The ladder simply
            # does not grow tonight, and the reason is one line rather than
            # a stack -- the first night to reach this printed twenty lines
            # of urllib and buried `410 github_models_retirement_brownout`
            # in the middle. A transport failure ends the attempts: asking
            # again cannot fix a server that is not there.
            return None, str(e)
        r, why = propose_finish(root, reply, rungs)
        if r is not None:
            if attempt:
                print("  admitted on attempt %d" % (attempt + 1),
                      file=sys.stderr)
            return r, None
        print("  attempt %d refused: %s" % (attempt + 1, why),
              file=sys.stderr)
    return None, "%d attempt(s) and none admitted; the last was: %s" % (
        PROPOSE_TRIES, why)


def write_rung(root, r):
    """Land an admitted rung on the ladder, named so order is the filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", r["title"].lower()).strip("-")[:40]
    slug = slug or "rung"
    d = os.path.join(root, "loop", "ladder")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%04d-%s.rung" % (int(r["seq"]), slug))
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_rung({k: r[k] for k in RUNG_KEYS}))
    return os.path.relpath(p, root).replace(os.sep, "/")


# ----------------------------------------------------------------- selftest


#: `feature` and a file that does not exist, because that is the shape of
#: the first rung toward any north star -- and a witnessed kind aimed at the
#: same target is now a refusal, which is a claim below rather than a
#: fixture that quietly stopped being admissible.
def _rung(goal, seq=1, kind="feature", title="a thing", witness="w", why="y",
          target="src/codec/mod.rs"):
    return "\n".join([
        "looprung 1", "seq %d" % seq, "goal %s" % goal, "kind %s" % kind,
        "target %s" % target, "title %s" % title, "witness %s" % witness,
        "why %s" % why]) + "\n"


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
        refuses(lambda: admit(parse_rung(_rung(g, kind="cleanup")), tmp),
                "cannot be judged here yet",
                "a kind with no reachable J1 is refused, so no milestone "
                "lacks a judge")
        refuses(lambda: admit(parse_rung(_rung("0" * 64)), tmp),
                "was written for goal",
                "a rung aimed at another north star is refused")
        refuses(lambda: parse_rung(_rung(g, title="")), "is empty",
                "a rung with no title is refused")
        refuses(lambda: admit(parse_rung(_rung(g, target="docs/x.html")), tmp),
                "outside the",
                "a target outside the kind's scope is refused")
        refuses(lambda: admit(parse_rung(_rung(g, target="tools/godel.py")), tmp),
                "evaluator machinery",
                "and a target that IS the evaluator is refused by name")
        refuses(lambda: admit(parse_rung(_rung(g, target="src/gfx/video.rs")), tmp),
                "unjudgeable surface",
                "and an unjudgeable surface is refused before a night is spent")
        refuses(lambda: admit(parse_rung(_rung(g, target="../escape.rs")), tmp),
                "outside the tree",
                "and one reaching out of the tree is refused")
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

        # --- retirement, which is what stops a bad rung being forever ------
        def write_cert(name, verdict):
            c["verdict"] = verdict
            with open(os.path.join(entries, name), "w",
                      encoding="utf-8", newline="\n") as f:
                f.write(godel.render_cert(c))

        os.remove(os.path.join(entries, "a.cert"))
        for i in range(RETIRE_AFTER - 1):
            write_cert("r%d.cert" % i, "refuse")
        st = state(tmp)[0]
        claim(st["refused"] == RETIRE_AFTER - 1 and not st["retired"],
              "a rung short of the refusal count is still open")
        claim(open_rung(state(tmp)) is not None,
              "and a night would still aim at it")

        write_cert("r%d.cert" % (RETIRE_AFTER - 1), "refuse")
        st = state(tmp)[0]
        claim(st["retired"] is True, "at the count it retires")
        claim(open_rung(state(tmp)) is None,
              "and no night aims at it again, which is the whole point")

        # Met beats retired: a rung that was eventually built is built,
        # however many nights it cost.
        write_cert("won.cert", "adopt")
        st = state(tmp)[0]
        claim(st["met"] and not st["retired"],
              "but an adoption afterwards un-retires it, because it was built")
        for n in os.listdir(entries):
            os.remove(os.path.join(entries, n))

        # --- the grammar, which must agree with what `admit` would allow ---
        g_text = grammar_for(tmp, [])
        claim(('"seq 1\\n"' in g_text) and (g in g_text),
              "the grammar pins the goal hash and the next seq as literals")
        claim(all(('"kind %s' % k) in g_text for k in judgeable_kinds()
                  if k == "feature" or existing_targets(tmp))
              and '"kind cleanup' not in g_text and '"kind tune' not in g_text,
              "and offers exactly the judgeable kinds, so an unjudgeable "
              "milestone cannot be sampled")
        # The grammar and the card are two statements of one fact; if the
        # grammar could still spell an unjudgeable directory, `admit` would
        # be refusing what the sampler was invited to write.
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import knob
        real = grammar_for(ROOT, [])
        claim(all(('"%s"' % pre.split("/")[1]) not in real
                  for pre in knob.UNJUDGEABLE),
              "and no unjudgeable directory is spellable in it at all")

        # --- the decomposer's output contract -----------------------------
        # Everything a model could answer that must not become a rung. The
        # ladder now has one met rung, so the next seq is 2.
        def fenced(body):
            return "```rung\n" + body + "```"

        good = _rung(g, 2, title="second")
        r, why = propose_finish(tmp, fenced(good))
        claim(r is not None and r["title"] == "second",
              "a clean reply becomes a rung")

        r, why = propose_finish(tmp, "Sure! Here you go:\n" + fenced(good))
        claim(r is None and "outside the fence" in (why or ""),
              "prose outside the fence is refused, not trimmed")

        r, why = propose_finish(tmp, fenced(good) + "\n" + fenced(good))
        claim(r is None and "fence(s)" in (why or ""),
              "two fences are a refusal, not a choice")

        r, why = propose_finish(tmp, "no fence at all")
        claim(r is None and "fence(s)" in (why or ""),
              "no fence is a refusal, never a retry with more context")

        # `cleanup` and not `feature`: feature's J1 is the claim count and
        # it is admissible now, which is the whole point of the row. What
        # stays refused is a kind whose J1 needs a rail that already reads
        # something, since a rung creates what nothing was measuring.
        r, why = propose_finish(tmp, fenced(_rung(g, 2, kind="cleanup")))
        claim(r is None and "cannot be judged here yet" in (why or ""),
              "a milestone under a kind with no reachable J1 is refused by name")

        r, why = propose_finish(tmp, fenced(_rung(g, 2, kind="feature")))
        claim(r is not None,
              "a feature rung is admitted, so greenfield work has a lane")
        os.makedirs(os.path.join(tmp, "src", "ai"), exist_ok=True)
        open(os.path.join(tmp, "src", "ai", "here.rs"), "w").write("// x")
        r2, why2 = propose_finish(tmp, fenced(
            _rung(g, 2, kind="feature", target="src/ai/here.rs")))
        claim(r2 is None and "already exists" in (why2 or ""),
              "and a feature naming a file that is already there is refused")

        # **A rung must name a part, not the whole thing.** Every rung the
        # 4B wrote for `create a native video codec` was titled `a native
        # video codec that encodes a 4x4 pixel buffer...` -- the goal with a
        # size bolted on, twice, under two kinds and two targets. The prompt
        # said "split it" in prose and named that exact wrong answer as a
        # counter-example, and it was produced anyway.
        r3, why3 = propose_finish(tmp, fenced(_rung(
            g, 2, kind="feature",
            title="a test star that encodes a 4x4 block")))
        claim(r3 is None and "repeats the north star" in (why3 or ""),
              "a title that restates the north star is refused by name")
        r4, why4 = propose_finish(tmp, fenced(_rung(
            g, 2, kind="feature", title="a zigzag scan order over a block")))
        claim(r4 is not None,
              "and a title naming one part of it is admitted")

        # The rule whose absence cost three nights: a witnessed kind aimed
        # at a file that is not there yet is stuck, not slow -- it refuses
        # locally, files no certificate, and so never retires.
        # **A ladder with something met**, because the first-rung rule closes
        # the witnessed kinds until one is, and these two claims are about
        # the target check sitting behind it.
        built = [dict(parse_rung(_rung(g, 1)), met=True, retired=False,
                      stale="", point="x" * 64)]
        r, why = propose_finish(tmp, fenced(_rung(g, 2, kind="test")), built)
        claim(r is None and "does not exist yet" in (why or ""),
              "a witnessed kind aimed at a file that is not there is refused")
        os.makedirs(os.path.join(tmp, "src", "ai"), exist_ok=True)
        open(os.path.join(tmp, "src", "ai", "there.rs"), "w").write("// x")
        r, why = propose_finish(tmp, fenced(
            _rung(g, 2, kind="test", target="src/ai/there.rs")), built)
        claim(r is not None,
              "and the same kind aimed at a file that IS there is admitted")
        claim('"kind test' not in grammar_for(tmp, [])
              and '"kind feature' in grammar_for(tmp, []),
              "with nothing met, the grammar offers only the creating kind")
        claim('"kind test' in grammar_for(tmp, [dict(met=True)]),
              "and a witnessed kind opens as soon as one rung is met")

        # **A rung that stops admitting must not take the ladder down.**
        # `load` raised, so one rung written before a rule tightened made
        # `next` exit 1 where the night needed "nothing open" to go stock
        # another -- and `retired` counts certificates, which a rung refused
        # before an envelope exists never earns. It blocks for good.
        # Written to disk, because `propose_finish` deliberately does not
        # persist -- the staleness is a property of the ladder as it is read
        # back, which is where the raise was.
        with open(os.path.join(tmp, "loop", "ladder", "0002-there.rung"),
                  "w", encoding="utf-8", newline="\n") as f:
            f.write(render_rung({k: r[k] for k in RUNG_KEYS}))
        os.remove(os.path.join(tmp, "src", "ai", "there.rs"))
        rs = state(tmp)
        claim(len(rs) == 2 and rs[-1]["stale"] and rs[-1]["retired"],
              "a rung the rules no longer admit retires instead of raising")
        nxt = open_rung(rs)
        claim(nxt is not None and nxt["seq"] == rs[0]["seq"],
              "so the read survives it and the stale rung is never offered")

        # **The third stuck shape.** `create` is temperature 0, so a rung
        # it cannot write is a rung it cannot write every night, and a
        # local refusal files no certificate for `refused` to count.
        first = state(tmp)[0]
        claim(balked(tmp, first["point"]) == 0,
              "a rung starts with no balks against it")
        for _ in range(RETIRE_AFTER):
            note_balk(tmp, first["point"], "the author came back with nothing")
        claim(balked(tmp, first["point"]) == RETIRE_AFTER,
              "and each night the author balks is recorded")
        claim(state(tmp)[0]["retired"],
              "so a rung nobody can write retires instead of blocking")
        os.remove(os.path.join(tmp, "loop", "ladder", "0002-there.rung"))

        r, why = propose_finish(tmp, fenced(_rung(g, 7, title="leapfrog")))
        claim(r is None and "next is 2" in (why or ""),
              "a rung that skips ahead in the sequence is refused")

        # The injection drill. A value is whatever follows the first space on
        # one line, so a newline inside one is not expressible -- but an extra
        # *line* is, and that is the shape that would forge a field.
        r, why = propose_finish(tmp, fenced(_rung(g, 2) + "point deadbeef\n"))
        claim(r is None and "not a rung field" in (why or ""),
              "a reply that tries to declare its own point is refused")

    print()
    if all(claims):
        print("  ladder passed (%d claims)" % len(claims))
        return 0
    print("  ladder FAILED")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", default="progress",
                    choices=("goal", "list", "next", "progress", "admit",
                             "propose", "balk"))
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
        if a.cmd == "balk":
            # The night calls this when the author came back with nothing,
            # so a rung nobody can write retires instead of blocking.
            rungs = state(a.root)
            nxt = open_rung(rungs)
            if nxt is None:
                print("  no open rung to record a balk against")
                return 2
            at = note_balk(a.root, nxt["point"], a.path or "(unstated)")
            n = balked(a.root, nxt["point"])
            print("  rung %s balked %d time(s) of %d: %s"
                  % (nxt["seq"], n, RETIRE_AFTER, at))
            return 0
        if a.cmd == "propose":
            # **What `propose` needs is an endpoint, and it was demanding a
            # credential.** The gate predates the author moving into the job:
            # `ask_model` puts the token in an `Authorization` header that
            # `llama-server` ignores, so in CI this passed because a token
            # happened to be in the environment and was then read by nobody.
            # A machine with the model and no GitHub token could not stock
            # its own ladder, which is the opposite of what this file claims
            # about needing no credential of any kind.
            token = (os.environ.get("GITHUB_TOKEN")
                     or os.environ.get("GH_TOKEN") or "")
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            import godel
            if not token and not os.environ.get("GLADOS_INFERENCE_URL") \
                    and godel.INFERENCE_URL.startswith("https://"):
                print("::error::propose needs somewhere to ask: set "
                      "GLADOS_INFERENCE_URL, or GITHUB_TOKEN for the "
                      "hosted default")
                return 2
            r, why = propose(a.root, token)
            if r is None:
                # A refusal is the ordinary outcome and not an error: the
                # contract held. The night records it and moves on rather
                # than retrying with more context, which is how a fence
                # contract stops being one.
                print("  nothing proposed: %s" % why)
                return 3
            print("  %s" % write_rung(a.root, r))
            print("  rung %s [%s] %s" % (r["seq"], r["kind"], r["title"]))
            print("  witness %s" % r["witness"])
            return 0
        return cmd_progress(a.root)
    except Bad as e:
        print("::error::%s" % e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
