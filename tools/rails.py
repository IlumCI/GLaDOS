#!/usr/bin/env python3
"""Rails, and the judge that decides whether a change moved one.

`bench report` in the kernel emits a block of named numbers; `lm_eval.py`
writes per-item dumps for the model rails. This is the layer above both: one
format that carries either kind, and one comparison that says whether a rail
moved beyond its own noise.

**Why this exists.** `godel`'s J1 asks whether a variant repaired routing
decisions beyond the noise and J2 asks whether it broke the machine's own
goals. Both are about one subsystem, because routing was the only thing this
tree could measure. A change that speeds the renderer, fixes a driver or
shortens a decode step had no judge at all -- so a loop that can author
anything would optimise routing accuracy and call it self-improvement. The
generalisation is: **a change is judged against the rail it claims to move and
the rails it must not break.** J1 becomes the first, J2 the second, and this
file is what either one asks.

Usage:
    rails.py collect --kernel LOG [--rail NAME=DUMP ...] --out rails.txt
    rails.py compare BEFORE AFTER
    rails.py judge BEFORE AFTER --claims NAME
    rails.py --selftest

A line is `name value unit want [key=value ...]`, and `want` is `lower` or
`higher` because `gflops` and `us` point opposite ways and a comparison that
guessed would call every graphics improvement a regression. `dump=PATH` on a
line says per-item data exists, which is what turns a comparison of two
scalars into a paired test.
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BEGIN = "[rail] v1"
END = "[rail] end"

# How far a rail may move and still be noise, as a fraction.
#
# **Measured rather than chosen, and the ones with no measurement behind them
# say so.** A threshold nobody can find is a threshold nobody can argue with,
# which is the point `MCNEMAR_95` makes about itself.
#
# **These came from three boots of one binary and the first set was wrong by up
# to seven times.** The numbers first written here were within-boot figures
# lifted out of `CLAUDE.md` -- "about 10% between boots", "16% across the pair
# that judged the `Rc` change" -- and across boots under the hypervisor
# accelerator, `--no-payload`, `-smp 4`, the same command prefix, they are:
#
#   rail               1 vs 2    1 vs 3    2 vs 3 (post-control)
#   video.rect         -69.4%    -62.0%    +24.4%     (the control itself)
#   video.*            ...       ...       -20 to -30%
#   core.new           -73.7%    -72.7%     +3.9%     (the control itself)
#   core.*             ...       ...       +0.2 to +1.5%
#   ai.matmul           +2.5%    -32.5%    -34.1%
#   smp.*              -50%      -61%      -21.7%, -5.1%
#
# Two things fall out and both are protocol rather than tuning.
#
# **Discard the first reading after a build.** Run 1 was two to three times
# slower than runs 2 and 3 on every timing rail, which is the host's page cache
# meeting a freshly written 5 MB image. Comparing anything against it reports
# the build system.
#
# **The control works, where there is a real one.** After dividing `core.new`
# out, the three interpreter rails agree to within 1.5% across boots. That is
# the design doing exactly what it was for. `video.rect` only half works --
# the group still moves 20 to 30% once it is divided out, because
# `desk::draw` depends on what is on screen and a rectangle does not -- so the
# graphics floor is set from the residual rather than from the control.
NOISE = {
    "video.": 0.35,
    "core.": 0.16,
    "smp.": 0.29,
    "ai.matmul": 0.35,
    # **Deterministic given the same text, and the text is the risk.**
    # `ai.bpb` is a forward pass per token over a fixed window: the same model
    # on the same bytes answers the same figure to the last millibit, so the
    # instrument contributes no noise at all and a floor of zero would be
    # defensible about the arithmetic.
    #
    # It is not defensible about the corpus. The window is the machine's *own*
    # history -- the journal the night writes and the ledger the judges write
    # -- so two arms booted separately read two different texts unless
    # something hands them one. A comparison that got that wrong would move
    # this rail for a reason that has nothing to do with the model, and
    # nothing here can detect it. Two per cent is the bound on what a few
    # lines of drift in a 2 KB window is worth; it is not a measurement, and
    # it is written down as an assumption rather than left implicit.
    "ai.bpb": 0.02,
}

# Which rail is the control for a group, for a comparison that wants to divide
# out the day rather than trust a fixed percentage.
#
# **`ai.matmul` and `smp.*` have none, and that is a hole rather than a
# decision.** A control has to be something nothing under test can touch, and
# neither group has such a thing: `ai.matmul` *is* the arithmetic, and `smp`'s
# one-core arm shares every source of noise with its many-core arm. So on a day
# where the host was busier those two report a regression that is the day, with
# nothing to divide out and nothing to say so -- which is exactly what happened
# the first time this ran across two boots at different core counts. Until they
# have one, compare them only between runs on one machine with the same `-smp`
# and the same command prefix.
CONTROL = {
    "video.": "video.rect",
    "core.": "core.new",
}

#: Groups with no control, named so a reader is not left to notice the absence.
UNCONTROLLED = ("ai.", "smp.")

# The two-sided 95% bar on a standard normal, and the conventional chi-squared
# 95% line for one degree of freedom. Both named rather than inlined, and both
# the same numbers `godel` and `paired.py` use -- two definitions of "beyond
# the noise" free to drift is the thing `judge_one` exists to prevent.
Z95 = 1.96
MCNEMAR_95 = 3.84
# A net repair under this is not a repair. `godel::MIN_FIXED`.
MIN_FIXED = 4


def noise_for(name):
    """The fraction this rail may move and still be noise, and where it came
    from. Longest declared prefix wins, so a specific rail can override its
    group."""
    best, floor = "", None
    for k, v in NOISE.items():
        if name.startswith(k) and len(k) > len(best):
            best, floor = k, v
    return floor


def control_for(name):
    best, ctl = "", None
    for k, v in CONTROL.items():
        if name.startswith(k) and len(k) > len(best):
            best, ctl = k, v
    return ctl


class Rail:
    __slots__ = ("name", "value", "unit", "want", "extra")

    def __init__(self, name, value, unit, want, extra=None):
        self.name = name
        self.value = value          # None means absent
        self.unit = unit
        self.want = want            # "lower" or "higher"
        self.extra = extra or {}

    def render(self):
        v = "absent" if self.value is None else f"{self.value:.3f}"
        tail = "".join(f" {k}={x}" for k, x in sorted(self.extra.items()))
        return f"{self.name} {v} {self.unit} {self.want}{tail}"


def parse_block(text):
    """Every rail in a block, in order. Raises if the block never ends.

    **An unterminated block is refused rather than read short**, for the reason
    `v4.py` walks and never seeks: a truncated transcript would otherwise
    produce a shorter rail list that compares clean against nothing, and the
    comparison would silently be about a different set of rails.
    """
    rails, inside, closed = [], False, False
    for line in text.splitlines():
        line = line.strip()
        if line == BEGIN:
            inside, rails = True, []
            continue
        if line == END:
            closed = True
            break
        if not inside or not line:
            continue
        # The `absent` line carries a reason after `--`, which is prose for a
        # person and not part of the record.
        line = line.split("  --", 1)[0].strip()
        p = line.split()
        if len(p) < 4:
            continue
        name, val, unit, want = p[0], p[1], p[2], p[3]
        extra = {}
        for kv in p[4:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                extra[k] = v
        rails.append(Rail(name, None if val == "absent" else float(val), unit, want, extra))
    if not closed:
        raise SystemExit("  the block has no '[rail] end' -- the transcript is truncated")
    return rails


def read(path):
    with open(path, encoding="utf-8") as f:
        return parse_block(f.read())


def render(rails):
    return "\n".join([BEGIN] + [r.render() for r in rails] + [END, ""])


# --- the judge ------------------------------------------------------------

#: A rail moved for the better, for the worse, or not beyond its noise.
BETTER, WORSE, SAME, ABSENT = "better", "worse", "same", "absent"
#: ...or the two readings are not comparable at all, which is neither.
UNSTABLE = "unstable"

# How far a control may move before dividing it out stops being defensible, as
# a multiple of its own floor.
#
# **This is a decision and not a derivation**, and it is here because the first
# real comparison ran into it. Two boots of the *same* build under emulation,
# one at `-smp 1` and one at `-smp 4`, moved `video.rect` -- a single
# span-filled rectangle, the strictest control in the set -- by **129.6%**, and
# every other rail by 100 to 185%. Reported as eleven regressions, which is
# what a comparison with no control does. Normalising by the control turns
# that into a few per cent, but normalising assumes the day's effect is
# multiplicative and uniform, and past some point that assumption is doing
# more work than the measurement.
#
# **Fitted to one separation, which is worth saying rather than dressing up.**
# Across the three boots above, the pair that should be refused moved its
# controls by 69.4% and 73.7% -- 1.98x and 4.6x the floors below -- and the
# pair that should be admitted moved them by 24.4% and 3.9%, which is 0.70x and
# 0.24x. So the window is roughly 0.7 to 1.98 and 1.5 sits in it with margin
# both ways, where 2.0 would have let the first pair through by 0.017.
#
# Two data points is two data points. What makes an error here cheap in one
# direction and not the other is the asymmetry: a refusal says "take it again"
# and an admission produces a verdict, so this leans toward refusing.
CONTROL_LIMIT = 1.5


def relative(before, after):
    """The fractional change, or None when one side has no number."""
    if before is None or after is None:
        return None
    if before.value is None or after.value is None or before.value == 0:
        return None
    return (after.value - before.value) / abs(before.value)


def verdict(before, after, drift=None, floor=None):
    """Did this rail move, and which way, given the two readings?

    Answers (verdict, why). Unpaired: the only evidence is two scalars, so the
    test is whether the relative change clears a declared floor. That is a
    weaker instrument than the paired ones below, and the floor plus the
    control are what make it honest rather than a `!=`.

    `drift` is what the group's control did between the same two readings. It
    is *divided out* rather than subtracted, because the thing it stands in for
    -- a busier host, a different core count, a colder cache -- scales a
    duration rather than shifting it. Without it the first real comparison
    reported eleven regressions on two boots of one build.
    """
    if before.value is None or after.value is None:
        return ABSENT, "one side could not measure it"
    if before.value == 0:
        return SAME, "the baseline reads zero, so a ratio says nothing"
    rel = (after.value - before.value) / abs(before.value)
    floor = noise_for(after.name) if floor is None else floor
    if floor is None:
        return SAME, "no noise floor is declared for this rail"
    said = f"{rel:+.1%}"
    if drift is not None and drift != 0:
        # Both as ratios of the baseline, so the control cancels.
        rel = (1.0 + rel) / (1.0 + drift) - 1.0
        said = f"{rel:+.1%} once the control's {drift:+.1%} is divided out"
    if abs(rel) < floor:
        return SAME, f"{said}, inside the {floor:.0%} floor"
    improved = rel > 0 if after.want == "higher" else rel < 0
    return (BETTER if improved else WORSE), f"{said}, against a {floor:.0%} floor"


def paired_verdict(before, after):
    """The same question where per-item data exists on both sides.

    **Far more powerful than comparing two scalars, and that is the whole
    reason the dumps are kept.** GSM8K reduces a 107-token answer to one bit,
    so the int8 KV cache needed all 1,319 questions to reach chi 3.86 against a
    bar of 3.84; the same change reads t = 6.61 on 256 bits-per-byte windows.
    A rail with a dump on both sides is judged on the items, never on the
    percentages.
    """
    import paired

    a, b = before.extra.get("dump"), after.extra.get("dump")
    if not a or not b:
        return None
    kind = before.extra.get("paired", "binary")
    if kind == "bpb":
        _, rows_a = paired.read_bpb(a)
        _, rows_b = paired.read_bpb(b)
        keys = sorted(set(rows_a) & set(rows_b))
        if not keys:
            return SAME, "no window is in both runs, so nothing is paired"
        # Bits per byte per window, differenced. Lower is better, always.
        d = []
        for k in keys:
            nb_a, _, nats_a = rows_a[k]
            nb_b, _, nats_b = rows_b[k]
            if nb_a <= 0 or nb_b <= 0:
                continue
            d.append(nats_b / nb_b / math.log(2) - nats_a / nb_a / math.log(2))
        if len(d) < 2:
            return SAME, "fewer than two comparable windows"
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
        se = math.sqrt(var / len(d))
        if se > 0:
            t = mean / se
        elif mean == 0.0:
            t = 0.0
        else:
            # Zero variance is the *strongest* evidence, not the weakest: every
            # window moved by the same non-zero amount.
            t = math.copysign(math.inf, mean)
        if abs(t) < Z95:
            return SAME, f"t = {t:+.2f} over {len(d)} windows, inside {Z95}"
        # Fewer bits per byte is better, so a negative mean is an improvement.
        return (BETTER if mean < 0 else WORSE), f"t = {t:+.2f} over {len(d)} windows"

    _, rows_a = paired.read(a)
    _, rows_b = paired.read(b)
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        return SAME, "no item is in both runs, so nothing is paired"
    fixed = sum(1 for k in keys if not rows_a[k][0] and rows_b[k][0])
    broke = sum(1 for k in keys if rows_a[k][0] and not rows_b[k][0])
    chi = paired.mcnemar(broke, fixed)
    net = fixed - broke
    if abs(net) < MIN_FIXED:
        return SAME, f"fixed {fixed} broke {broke} of {len(keys)}, net under {MIN_FIXED}"
    if chi < MCNEMAR_95:
        return SAME, f"fixed {fixed} broke {broke}, chi {chi:.2f} inside {MCNEMAR_95}"
    return (BETTER if net > 0 else WORSE), f"fixed {fixed} broke {broke}, chi {chi:.2f}"


def compare(before, after):
    """Every rail present on either side, paired by name.

    By name and not by position, and every rail on either side appears. A rail
    that exists in one report and not the other is a change in the report's
    shape, which is a thing to say out loud rather than to drop.

    **A group whose control moved too far is not judged, it is refused.** That
    is the `Caught::Unguarded` idea on a rail: the closure ran and proved
    nothing, so treating it as a pass and treating it as a failure are both
    wrong. The first real comparison here was two boots of one build at
    different core counts, where the graphics control moved 129.6% and the
    honest report is "take it again", not eleven regressions.
    """
    ba = {r.name: r for r in before}
    aa = {r.name: r for r in after}

    # What each group's control did, and whether it did too much.
    drift, refuse = {}, {}
    for prefix, ctl in CONTROL.items():
        d = relative(ba.get(ctl), aa.get(ctl))
        floor = noise_for(ctl) or 0.0
        drift[prefix] = d
        refuse[prefix] = d is not None and floor > 0 and abs(d) > floor * CONTROL_LIMIT

    out = []
    for name in list(ba) + [n for n in aa if n not in ba]:
        b, a = ba.get(name), aa.get(name)
        if b is None or a is None:
            out.append((name, ABSENT, "only one report has this rail"))
            continue
        # The paired test needs no control: it compares the same items on both
        # sides and the day divides out by construction.
        p = paired_verdict(b, a)
        if p:
            out.append((name, *p))
            continue
        prefix = next((k for k in CONTROL if name.startswith(k)), None)
        if prefix is not None and refuse[prefix]:
            out.append((
                name,
                UNSTABLE,
                f"{CONTROL[prefix]} moved {drift[prefix]:+.1%}, so these two "
                f"readings do not compare",
            ))
            continue
        # A control judged against itself is always 0% and always `same`, which
        # is true and useless. Reported as what it is instead, so a reader can
        # see the number the rest of the group was divided by.
        if prefix is not None and name == CONTROL[prefix]:
            out.append((name, SAME, f"the control; it moved {drift[prefix]:+.1%}"))
            continue
        out.append((name, *verdict(b, a, drift.get(prefix))))
    return out


def judge(before, after, claims):
    """J1 and J2, generalised to rails.

    - **J1**: every rail the change *claims* to move got better.
    - **J2**: no other rail got worse.

    Unanimity, as `godel` requires it, and for the same reason: a change that
    improves one rail by breaking another has not been measured, it has been
    traded, and the trade is the operator's to make rather than the loop's.

    A claimed rail that reads `absent` or `same` fails J1. "It did not
    regress" is not what a proposal claiming to move something promised, and
    admitting it is how a search fills the ledger with changes that did
    nothing.
    """
    rows = compare(before, after)
    by = {n: (v, why) for n, v, why in rows}
    j1, j1_why = True, []
    for c in claims:
        v, why = by.get(c, (ABSENT, "no such rail in either report"))
        if v != BETTER:
            j1 = False
        j1_why.append(f"{c}: {v} ({why})")
    if not claims:
        j1, j1_why = False, ["nothing was claimed, so there is nothing to have improved"]
    regressed = [(n, why) for n, v, why in rows if v == WORSE and n not in claims]
    # An unstable rail is neither evidence of a regression nor evidence against
    # one, so it does not veto J2 -- it invalidates the whole comparison, which
    # is a different answer from either verdict and is reported as one.
    unstable = [(n, why) for n, v, why in rows if v == UNSTABLE]
    return {
        "rows": rows,
        "j1": j1,
        "j1_why": j1_why,
        "j2": not regressed,
        "j2_why": regressed,
        "comparable": not unstable,
        "unstable": unstable,
    }


# --- collecting -----------------------------------------------------------


def collect(kernel_log, dumps):
    """One block from a kernel transcript plus whatever host dumps exist.

    The host rails are appended rather than merged into the kernel's list: they
    come from a different machine with a different clock, and a block that hid
    that would invite somebody to read `ai.matmul` and `gsm8k` as two readings
    of one system.
    """
    rails = []
    if kernel_log:
        with open(kernel_log, encoding="utf-8") as f:
            rails.extend(parse_block(f.read()))
    import paired

    for name, path in dumps:
        if name == "bpb":
            _, rows = paired.read_bpb(path)
            nb = sum(r[0] for r in rows.values())
            nats = sum(r[2] for r in rows.values())
            v = (nats / nb / math.log(2)) if nb else None
            rails.append(Rail("host.bpb", v, "bits", "lower",
                              {"dump": path, "paired": "bpb", "n": len(rows)}))
        else:
            _, rows = paired.read(path)
            hits = sum(1 for r in rows.values() if r[0])
            v = hits / len(rows) if rows else None
            rails.append(Rail(f"host.{name}", v, "acc", "higher",
                              {"dump": path, "paired": "binary", "n": len(rows)}))
    return rails


# --- claims about the judge ------------------------------------------------


def selftest():
    ok = True

    def claim(good, what):
        nonlocal ok
        if not good:
            ok = False
        print(f"  {'ok ' if good else 'FAIL'}  {what}")

    block = f"""{BEGIN}
video.rect 100.000 us lower
video.draw 2000.000 us lower
ai.matmul 9.500 gflops higher
store.read absent us higher  -- no store mounted
{END}
"""
    rs = parse_block(block)
    claim(len(rs) == 4, "every line of a block is read, including the absent one")
    claim(rs[3].value is None and rs[3].unit == "us",
          "an absent rail carries no value and keeps its unit")
    claim(render(rs).count("absent") == 1, "and renders back as absent rather than as zero")

    # The failure that made this a function rather than a loop: a transcript
    # that stops mid-block would otherwise yield a short list that compares
    # clean against nothing.
    try:
        parse_block(f"{BEGIN}\nvideo.rect 100.000 us lower\n")
        claim(False, "a block with no end is refused")
    except SystemExit:
        claim(True, "a block with no end is refused")

    def one(name, v, unit, want):
        return Rail(name, v, unit, want)

    # Direction. The same relative move is an improvement on one rail and a
    # regression on another, which is the whole reason `want` is recorded.
    v, _ = verdict(one("video.draw", 100.0, "us", "lower"),
                   one("video.draw", 40.0, "us", "lower"))
    claim(v == BETTER, "a timing cut by 60% is better")
    v, _ = verdict(one("ai.matmul", 10.0, "gflops", "higher"),
                   one("ai.matmul", 5.0, "gflops", "higher"))
    claim(v == WORSE, "and a throughput that halved is worse")

    # The floor, which is what makes this more than a `!=`.
    v, why = verdict(one("video.draw", 100.0, "us", "lower"),
                     one("video.draw", 95.0, "us", "lower"))
    claim(v == SAME and "floor" in why, "a 5% move on a 35% floor is noise, and says so")
    v, _ = verdict(one("smp.one_core", 100.0, "mbs", "higher"),
                   one("smp.one_core", 120.0, "mbs", "higher"))
    claim(v == SAME, "and 20% on the smp rail is inside its measured 29%")
    claim(noise_for("smp.all_cores") == 0.29 and noise_for("video.console") == 0.35,
          "the floor comes from the longest declared prefix")
    # Named rather than left to be noticed. A group with no control is judged
    # on a fixed percentage alone, which is the weakest instrument here.
    claim(
        all(not any(u.startswith(k) for k in CONTROL) for u in UNCONTROLLED)
        and all(any(n.startswith(u) for u in UNCONTROLLED) or control_for(n)
                for n, _, _ in [(k + "x", "", "") for k in NOISE]),
        "every rail group either has a control or is declared as having none",
    )
    v, _ = verdict(one("nobody.declared", 1.0, "x", "higher"),
                   one("nobody.declared", 1000.0, "x", "higher"))
    claim(v == SAME, "a rail with no declared floor cannot be said to have moved")

    # A rail present on one side only. Dropping it silently is how a
    # comparison comes to be about a different set of rails than it says.
    rows = compare([one("a", 1.0, "u", "higher")], [one("b", 1.0, "u", "higher")])
    claim(len(rows) == 2 and all(v == ABSENT for _, v, _ in rows),
          "a rail in only one report is reported, not dropped")

    # J1 and J2.
    b = [one("video.draw", 100.0, "us", "lower"), one("ai.matmul", 10.0, "gflops", "higher")]
    a = [one("video.draw", 50.0, "us", "lower"), one("ai.matmul", 10.0, "gflops", "higher")]
    r = judge(b, a, ["video.draw"])
    claim(r["j1"] and r["j2"], "the claimed rail improved and nothing else moved")

    a2 = [one("video.draw", 50.0, "us", "lower"), one("ai.matmul", 5.0, "gflops", "higher")]
    r = judge(b, a2, ["video.draw"])
    claim(r["j1"] and not r["j2"], "and a rail it did not claim regressing is J2's veto")

    a3 = [one("video.draw", 100.0, "us", "lower"), one("ai.matmul", 20.0, "gflops", "higher")]
    r = judge(b, a3, ["video.draw"])
    claim(not r["j1"] and r["j2"],
          "a change that improved something else entirely still fails what it claimed")

    r = judge(b, a, [])
    claim(not r["j1"], "and a change claiming nothing cannot have improved what it claimed")

    # The one that keeps J1 strict. "It did not regress" is not what a
    # proposal claiming to move a rail promised.
    a4 = [one("video.draw", 99.0, "us", "lower"), one("ai.matmul", 10.0, "gflops", "higher")]
    r = judge(b, a4, ["video.draw"])
    claim(not r["j1"], "a claimed rail that merely held still fails J1")

    # --- the control ------------------------------------------------------
    #
    # A whole group slower by the same factor is a slower machine, not a worse
    # build. Without dividing the control out, the first real comparison in
    # this tree reported eleven regressions on two boots of one binary.
    slow_b = [one("video.rect", 100.0, "us", "lower"), one("video.draw", 1000.0, "us", "lower")]
    slow_a = [one("video.rect", 120.0, "us", "lower"), one("video.draw", 1200.0, "us", "lower")]
    rows = {n: (v, why) for n, v, why in compare(slow_b, slow_a)}
    claim(rows["video.draw"][0] == SAME,
          "a group slower by the same factor as its control has not regressed")
    claim("divided out" in rows["video.draw"][1],
          "and the line says the control was divided out")
    claim(rows["video.rect"][0] == SAME and "control" in rows["video.rect"][1],
          "the control reports what it did rather than comparing against itself")

    # ...and a rail that moved *against* its group still reads as moved.
    real_a = [one("video.rect", 120.0, "us", "lower"), one("video.draw", 600.0, "us", "lower")]
    rows = {n: (v, why) for n, v, why in compare(slow_b, real_a)}
    claim(rows["video.draw"][0] == BETTER,
          "a rail that beat its own control by half is better, on a slower day")

    # The refusal. Past `CONTROL_LIMIT` times its floor, dividing the control
    # out is doing more work than the measurement.
    wild_a = [one("video.rect", 230.0, "us", "lower"), one("video.draw", 2600.0, "us", "lower")]
    rows = {n: (v, why) for n, v, why in compare(slow_b, wild_a)}
    claim(rows["video.draw"][0] == UNSTABLE,
          "a control past its limit makes the group not comparable rather than worse")
    r = judge(slow_b, wild_a, ["video.draw"])
    claim(not r["comparable"] and not r["j1"],
          "and a claim cannot be granted from a measurement that does not compare")
    claim(r["j2"] and not r["j2_why"],
          "while J2 does not read an unstable rail as a regression either")

    # **The separation the limit was fitted to, locked in.** Three boots of one
    # binary: the pair with the cold first reading moved its controls by 69.4%
    # and 73.7%, the settled pair by 24.4% and 3.9%. If a later floor or limit
    # stops telling those apart, the instrument has stopped being able to say
    # "take it again" and will start producing verdicts from cold caches.
    for ctl, moved, want_refuse in [
        ("video.rect", 0.694, True),
        ("core.new", 0.737, True),
        ("video.rect", 0.244, False),
        ("core.new", 0.039, False),
    ]:
        floor = noise_for(ctl)
        refused = moved > floor * CONTROL_LIMIT
        claim(
            refused == want_refuse,
            f"a control moving {moved:.1%} is {'refused' if want_refuse else 'admitted'}"
            f" against {ctl}'s {floor:.0%} floor",
        )

    return ok


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("--selftest", "selftest"):
        print("[rails] the judge, without two builds to point it at")
        return 0 if selftest() else 1

    cmd = argv[0]
    if cmd == "collect":
        kernel, out, dumps = None, None, []
        i = 1
        while i < len(argv):
            if argv[i] == "--kernel":
                kernel = argv[i + 1]
                i += 2
            elif argv[i] == "--out":
                out = argv[i + 1]
                i += 2
            elif argv[i] == "--rail":
                name, path = argv[i + 1].split("=", 1)
                dumps.append((name, path))
                i += 2
            else:
                raise SystemExit(f"  unknown argument {argv[i]}")
        text = render(collect(kernel, dumps))
        if out:
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  {out}")
        else:
            sys.stdout.write(text)
        return 0

    if cmd in ("compare", "judge"):
        if len(argv) < 3:
            raise SystemExit(f"  usage: rails.py {cmd} BEFORE AFTER [--claims NAME ...]")
        before, after = read(argv[1]), read(argv[2])
        claims = []
        if "--claims" in argv:
            claims = [a for a in argv[argv.index("--claims") + 1:] if not a.startswith("-")]
        if cmd == "compare":
            for name, v, why in compare(before, after):
                print(f"  {name:<20} {v:<7} {why}")
            return 0
        r = judge(before, after, claims)
        for name, v, why in r["rows"]:
            print(f"  {name:<20} {v:<7} {why}")
        print()
        print(f"  J1 claimed  {'pass' if r['j1'] else 'VETO'}")
        for w in r["j1_why"]:
            print(f"    {w}")
        print(f"  J2 the rest {'pass' if r['j2'] else 'VETO'}")
        for n, why in r["j2_why"]:
            print(f"    {n} regressed: {why}")
        if not r["comparable"]:
            print()
            print(f"  NOT COMPARABLE -- {len(r['unstable'])} rail(s) had a control that drifted")
            for n, why in r["unstable"][:1]:
                print(f"    {why}")
            print("    take both readings again on one machine, same command prefix")
            # Two is neither adoption nor refusal, which is the whole point: a
            # verdict of "no" from an invalid measurement is as wrong as a yes.
            return 2
        return 0 if (r["j1"] and r["j2"]) else 1

    raise SystemExit(f"  unknown command {cmd}")


if __name__ == "__main__":
    sys.exit(main())
