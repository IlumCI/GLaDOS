#!/usr/bin/env python3
"""The workflow files, read the way GitHub reads them rather than the way a
permissive parser does.

### Why this exists

`experimental.yml` carried **two `env:` keys on one step** from `1f1744f`
until this was written. GitHub rejects a duplicate mapping key outright, and
what that produces is not an error message anybody reads -- it is a run named
after the *file path* instead of the workflow, holding zero jobs, failing in
under a second, attributed to whatever push happened to be nearby. Including
pushes to branches `on.push.branches` excludes, because the branch filter
lives inside the file that would not load.

So the workflow published nothing for that whole stretch and the only symptom
was a fast red X on unrelated branches, which reads as noise.

**The sweep that was supposed to catch it reported `ok`.** `yaml.safe_load`
resolves a duplicate by keeping the last one, so the file parsed cleanly in
Python while being unloadable on the runner -- and the value it kept dropped
`KEY` and `ORIGIN`, so even the survivor was wrong. A checker that disagrees
with the thing it is checking for is the failure this tree keeps recording:
`time::calibrate` derived its microsecond from the counter it was checking,
and `render`'s own suite reset the counters it measured.

### What it checks, and what it deliberately does not

This is **not** a schema validator for Actions, and writing one would be
signing up to track somebody else's format forever. It checks the two things
that produce a zero-job run, which is the failure mode that hides:

- **Duplicate keys**, anywhere in the document. The real defect.
- **A local `uses:` naming a path that is not in the tree.** The other way to
  get a workflow GitHub cannot load, and the one this repository is most
  exposed to: eleven call sites point at `./.github/actions/verify-boot`, and
  a rename would take every one of them out at once.

Everything else -- whether a step's `with:` matches the action's inputs,
whether an expression resolves -- is a runtime failure with a log attached,
which is to say a failure somebody can read. Those are not this tool's job.

    python3 tools/workflows.py             # check the tree
    python3 tools/workflows.py --selftest  # prove the check can fail
"""

import argparse
import os
import sys
import tempfile

try:
    import yaml
except ImportError:  # pragma: no cover
    print("  pyyaml is not installed; nothing can be checked", file=sys.stderr)
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Duplicate(Exception):
    """A key that appears twice in one mapping, with both line numbers."""


def _strict_loader():
    """A SafeLoader that refuses a duplicate key instead of keeping the last.

    Built per call rather than once at import, because `add_constructor` edits
    the class and a module-level one would leak the strictness into any other
    caller that happens to use `yaml.SafeLoader` afterwards.
    """

    class Strict(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        seen = {}
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            line = key_node.start_mark.line + 1
            if key in seen:
                raise Duplicate(
                    "duplicate key %r at line %d, first seen at line %d"
                    % (key, line, seen[key])
                )
            seen[key] = line
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return Strict


def load_strict(path):
    """Parse one file the strict way. Raises Duplicate or yaml.YAMLError."""
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, _strict_loader())


def local_uses(doc):
    """Every `uses:` in this document that names a path in the repository.

    Walked generically rather than by knowing where steps live, because a
    `uses:` appears at three different depths -- a job's step, a composite
    action's step, and a job calling a reusable workflow -- and a walk that
    knew the shapes would stop finding them the day a fourth is added.
    """
    found = []

    def walk(node):
        if isinstance(node, dict):
            u = node.get("uses")
            if isinstance(u, str) and u.startswith("./"):
                found.append(u)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return found


def folded_runs(path):
    """`run:` blocks that lost their `|` and therefore fold into one line.

    **This is what stopped the CI Godel machine reaching its first verdict.**
    `loop-judge.yml` had

        run: cp kernel-b2-within.txt kernel-b2.txt
          python3 tools/retrieval.py ...

    with no block scalar, so YAML folds the lines together with spaces and the
    runner is handed one command. It died on `syntax error near unexpected
    token '}'`, the `{ ... }` group having lost the newline before it. Its
    baseline twin thirty lines up was written `run: |` and passed, which made
    the failure read as something about the candidate arm.

    Nothing in the parsed document can show this -- both spellings produce a
    string -- so the *node* is inspected instead: a `run:` scalar spanning
    more than one source line must be literal. A folded or quoted one is a
    shell script with its newlines removed, which is never what anybody meant.
    """
    with open(path, encoding="utf-8") as f:
        root = yaml.compose(f, yaml.SafeLoader)
    bad = []

    def walk(n):
        if isinstance(n, yaml.MappingNode):
            for k, v in n.value:
                if (
                    isinstance(k, yaml.ScalarNode)
                    and k.value == "run"
                    and isinstance(v, yaml.ScalarNode)
                    and v.style != "|"
                    and v.start_mark.line != v.end_mark.line
                ):
                    bad.append(
                        "`run:` at line %d spans %d lines without `|`, so it "
                        "folds into one command"
                        % (k.start_mark.line + 1,
                           v.end_mark.line - v.start_mark.line + 1)
                    )
                walk(v)
        elif isinstance(n, yaml.SequenceNode):
            for v in n.value:
                walk(v)

    if root is not None:
        walk(root)
    return bad


def on_block(doc):
    """A workflow's `on:`, which YAML 1.1 resolves to the boolean `True`.

    Worth a function rather than a comment: `doc.get("on")` answers `None` on
    every workflow in this tree and looks like a workflow with no triggers.
    """
    if not isinstance(doc, dict):
        return {}
    got = doc.get("on")
    if got is None:
        got = doc.get(True)
    return got if isinstance(got, dict) else {}


def call_contract(root, doc):
    """Reusable-workflow calls against what the callee declares.

    `loop-night` calls `loop-judge` with **fifteen** inputs. An input the
    callee does not declare fails the run outright; a *required* one the
    caller omits does the same. Neither is a thing anybody notices while
    editing one of the two files, and the loop's whole night is that one call.
    """
    bad = []
    for name, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        u = job.get("uses")
        if not (isinstance(u, str) and u.startswith("./") and u.endswith((".yml", ".yaml"))):
            continue
        callee_path = os.path.join(root, u[2:])
        if not os.path.isfile(callee_path):
            continue  # already reported by the `uses:` check
        try:
            callee = load_strict(callee_path)
        except (Duplicate, yaml.YAMLError):
            continue  # already reported against that file
        declared = (on_block(callee).get("workflow_call") or {}).get("inputs") or {}
        supplied = job.get("with") or {}
        for k in supplied:
            if k not in declared:
                bad.append("job %s passes `%s` to %s, which does not declare it"
                           % (name, k, u))
        for k, spec in declared.items():
            if isinstance(spec, dict) and spec.get("required") and k not in supplied:
                bad.append("job %s omits `%s`, which %s requires" % (name, k, u))
    return bad


def declared_outputs(root, job):
    """What a job actually publishes.

    A normal job declares `outputs:`. **A job that is a reusable-workflow call
    declares none of its own** -- its outputs are the callee's
    `workflow_call.outputs`, so reading `outputs:` off the caller would report
    every one of them as undeclared. `loop-night.judge` is exactly that shape,
    which is the false positive this exists to avoid.
    """
    if not isinstance(job, dict):
        return {}
    u = job.get("uses")
    if isinstance(u, str) and u.startswith("./"):
        p = os.path.join(root, u[2:])
        if os.path.isfile(p):
            try:
                return (on_block(load_strict(p)).get("workflow_call") or {}).get(
                    "outputs"
                ) or {}
            except (Duplicate, yaml.YAMLError):
                return {}
        return {}
    if isinstance(u, str):
        return None  # a remote reusable workflow: not knowable from here
    return job.get("outputs") or {}


def output_refs(root, doc):
    """`needs.<job>.outputs.<name>` that the named job never declares.

    **This one fails quietly, which is why it is here.** An undeclared output
    resolves to the empty string with no error at all, so the judge is handed
    an empty `rail` or an empty `chi_bar` and goes on to measure something
    against nothing. A run that fails is cheaper than a verdict that is wrong.
    """
    import re

    jobs = doc.get("jobs") or {}
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str):
            for m in re.finditer(
                r"needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)", node
            ):
                found.add(m.groups())

    walk(jobs)
    bad = []
    for job, out in sorted(found):
        if job not in jobs:
            bad.append("`needs.%s.outputs.%s` names no such job" % (job, out))
            continue
        declared = declared_outputs(root, jobs[job])
        if declared is None:
            continue  # a remote reusable workflow; nothing here can say
        if out not in declared:
            bad.append("`needs.%s.outputs.%s` is never declared, so it is empty"
                       % (job, out))
    return bad


def resolves(root, ref):
    """Whether a local `uses:` names something that is there.

    An action may be a directory holding `action.yml` or `action.yaml`; a
    reusable workflow is the file itself.
    """
    base = os.path.join(root, ref[2:])
    if os.path.isfile(base):
        return True
    return any(
        os.path.isfile(os.path.join(base, n)) for n in ("action.yml", "action.yaml")
    )


def targets(root):
    """The files GitHub will try to load: workflows, and local actions."""
    out = []
    wf = os.path.join(root, ".github", "workflows")
    if os.path.isdir(wf):
        out += [
            os.path.join(wf, n)
            for n in sorted(os.listdir(wf))
            if n.endswith((".yml", ".yaml"))
        ]
    actions = os.path.join(root, ".github", "actions")
    if os.path.isdir(actions):
        for d in sorted(os.listdir(actions)):
            for n in ("action.yml", "action.yaml"):
                p = os.path.join(actions, d, n)
                if os.path.isfile(p):
                    out.append(p)
    return out


def check(root, quiet=False):
    """Returns a list of complaints. Empty means the tree is loadable."""
    bad = []
    for path in targets(root):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        try:
            doc = load_strict(path)
        except Duplicate as e:
            bad.append("%s: %s" % (rel, e))
            continue
        except yaml.YAMLError as e:
            bad.append("%s: will not parse: %s" % (rel, str(e).replace("\n", " ")))
            continue
        here = ["`uses: %s` names nothing in the tree" % u
                for u in local_uses(doc) if not resolves(root, u)]
        here += folded_runs(path)
        here += call_contract(root, doc)
        here += output_refs(root, doc)
        bad += ["%s: %s" % (rel, h) for h in here]
        if not quiet and not here:
            print("  ok   %s" % rel)
    return bad


# --- the selftest --------------------------------------------------------
#
# Written against files this makes rather than against the repository, for the
# reason `mem::fixed` gives about its own map: a claim about the tree passes
# here and fails on the next checkout, and what is under test is the *checker*.

CLEAN = """
name: fine
on:
  push:
    branches: ["main"]
jobs:
  one:
    runs-on: ubuntu-latest
    steps:
      - uses: ./.github/actions/thing
      - name: a step
        env:
          A: "1"
          B: "2"
        run: echo hi
"""

DUPED = CLEAN.replace('          A: "1"\n', '          A: "1"\n        env:\n')

CALLEE = """
name: callee
on:
  workflow_call:
    inputs:
      point:
        required: true
        type: string
      rail:
        required: false
        type: string
    outputs:
      verdict:
        value: "x"
jobs:
  only:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"""

# The shape `loop-night` has: a job that computes, a job that calls the judge
# with what it computed, and a job that reads the judge's answer.
CALLER = """
name: caller
on:
  push:
    branches: ["main"]
jobs:
  propose:
    runs-on: ubuntu-latest
    outputs:
      point: ${{ steps.s.outputs.point }}
    steps:
      - id: s
        run: echo hi
  judge:
    needs: propose
    uses: ./.github/workflows/callee.yml
    with:
      point: ${{ needs.propose.outputs.point }}
  after:
    needs: judge
    runs-on: ubuntu-latest
    steps:
      - run: echo ${{ needs.judge.outputs.verdict }}
"""


def selftest():
    claims = []

    def claim(ok, what):
        claims.append(ok)
        print("  %-4s %s" % ("ok" if ok else "FAIL", what))

    with tempfile.TemporaryDirectory() as tmp:
        wf = os.path.join(tmp, ".github", "workflows")
        act = os.path.join(tmp, ".github", "actions", "thing")
        os.makedirs(wf)
        os.makedirs(act)
        with open(os.path.join(act, "action.yml"), "w", encoding="utf-8") as f:
            f.write("name: thing\nruns:\n  using: composite\n  steps: []\n")

        good = os.path.join(wf, "good.yml")
        with open(good, "w", encoding="utf-8") as f:
            f.write(CLEAN)
        claim(not check(tmp, quiet=True), "a clean tree is accepted")

        # **The canary.** A checker that has never refused anything is
        # indistinguishable from one that reads nothing, which is the
        # objection `differ.rs` makes about its own suite and the one
        # `smp.rs` paid for. So the duplicate is introduced deliberately
        # and must be caught, by name and by line.
        with open(good, "w", encoding="utf-8") as f:
            f.write(DUPED)
        bad = check(tmp, quiet=True)
        claim(len(bad) == 1, "a second `env` on one step is refused")
        claim("duplicate key 'env'" in bad[0] if bad else False,
              "and the complaint names the key")

        # The shape the permissive reader takes, stated as a claim so the
        # difference between the two readers is asserted rather than
        # described in a comment.
        loose = yaml.safe_load(DUPED)
        kept = loose["jobs"]["one"]["steps"][1]["env"]
        claim(kept == {} or "A" not in kept,
              "and a permissive reader would have kept the wrong one")

        with open(good, "w", encoding="utf-8") as f:
            f.write(CLEAN.replace("./.github/actions/thing", "./.github/actions/gone"))
        bad = check(tmp, quiet=True)
        claim(len(bad) == 1 and "names nothing" in bad[0],
              "a local `uses:` pointing at nothing is refused")

        # The `run:` that folds. Written as the real one was, with the second
        # line indented under a plain scalar rather than under a `|`.
        with open(good, "w", encoding="utf-8") as f:
            f.write(CLEAN.replace(
                "      - name: a step\n"
                "        env:\n          A: \"1\"\n          B: \"2\"\n"
                "        run: echo hi\n",
                "      - name: a step\n"
                "        run: echo one\n          echo two\n"))
        bad = check(tmp, quiet=True)
        claim(any("folds into one command" in b for b in bad),
              "a `run:` that lost its `|` is refused")
        with open(good, "w", encoding="utf-8") as f:
            f.write(CLEAN)
        claim(not check(tmp, quiet=True),
              "and a one-line `run:` is not mistaken for one")

        with open(good, "w", encoding="utf-8") as f:
            f.write("name: x\non: push\njobs: [\n")
        claim(len(check(tmp, quiet=True)) == 1, "a file that will not parse is refused")
        os.remove(good)

        # --- the reusable-workflow contract, which is loop-night's whole night
        callee = os.path.join(wf, "callee.yml")
        caller = os.path.join(wf, "caller.yml")
        with open(callee, "w", encoding="utf-8") as f:
            f.write(CALLEE)

        def write_caller(text):
            with open(caller, "w", encoding="utf-8") as f:
                f.write(text)

        write_caller(CALLER)
        # This also pins the false positive worth having: `after` reads
        # `needs.judge.outputs.verdict`, and `judge` is a `uses:` job that
        # declares no outputs of its own -- they belong to the callee. A check
        # reading `outputs:` off the caller would refuse every one of them.
        claim(not check(tmp, quiet=True),
              "a correct call is accepted, callee-owned outputs included")

        write_caller(CALLER.replace(
            "      point: ${{ needs.propose.outputs.point }}\n",
            "      point: ${{ needs.propose.outputs.point }}\n      bogus: 1\n"))
        bad = check(tmp, quiet=True)
        claim(len(bad) == 1 and "does not declare it" in bad[0],
              "an input the callee never declared is refused")

        write_caller(CALLER.replace(
            "      point: ${{ needs.propose.outputs.point }}\n", ""))
        bad = check(tmp, quiet=True)
        claim(any("requires" in b for b in bad),
              "a required input the caller omits is refused")

        # The quiet one. An undeclared output is not an error on the runner --
        # it resolves to "" and the judge measures against nothing.
        write_caller(CALLER.replace("outputs.point }}", "outputs.nope }}"))
        bad = check(tmp, quiet=True)
        claim(any("is never declared, so it is empty" in b for b in bad),
              "an output nobody declared is refused rather than passed empty")

    print()
    if all(claims):
        print("  workflows passed (%d claims)" % len(claims))
        return 0
    print("  workflows FAILED")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="prove the check can fail, against fixtures it writes")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    bad = check(a.root)
    if bad:
        print()
        for b in bad:
            print("::error::%s" % b)
        print("\n  %d workflow file(s) GitHub would refuse to load" % len(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
