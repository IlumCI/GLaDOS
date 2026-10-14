#!/usr/bin/env python3
"""Check `ai::intent::VERBS` against the verbs `shell::execute` matches on.

Rust cannot enumerate the arms of a `match`, so the list the shell's
did-you-mean consults is written by hand -- and a hand-kept list is exactly
how `KNOWN_COMMANDS` went quietly partial. This reads both and fails on any
difference, the bargain `knob.py check` makes about its table.

    python3 tools/verbs.py            check
    python3 tools/verbs.py --selftest the extraction, on a fixture
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARM = re.compile(r'^        ("[a-z0-9_-]+"(?: \| "[a-z0-9_-]+")*) (?:if .*)?=>')
NAME = re.compile(r'"([a-z0-9_-]+)"')


def arms(src: str) -> set:
    start = src.index("fn execute(")
    end = src.index("fn run_decision(")
    out = set()
    for line in src[start:end].splitlines():
        m = ARM.match(line)
        if m:
            out.update(NAME.findall(m.group(1)))
    return out


def declared(src: str) -> set:
    body = src[src.index("pub const VERBS"):]
    body = body[: body.index("];")]
    return set(NAME.findall(body))


def selftest() -> bool:
    fx = 'fn execute(x) {\n    match cmd {\n        "a" | "b" => {}\n        "c" if x => {}\n            "nested" => {}\n    }\n}\nfn run_decision('
    got = arms(fx)
    ok = got == {"a", "b", "c"}
    print(("ok  " if ok else "FAIL"), "arms are read at the shell's indent and nowhere else:", sorted(got))
    return ok


def main() -> int:
    if "--selftest" in sys.argv:
        return 0 if selftest() else 1
    shell = arms((ROOT / "src/shell.rs").read_text())
    verbs = declared((ROOT / "src/ai/intent.rs").read_text())
    missing, extra = shell - verbs, verbs - shell
    for v in sorted(missing):
        print(f"missing from VERBS: {v}")
    for v in sorted(extra):
        print(f"in VERBS with no arm: {v}")
    if missing or extra:
        return 1
    print(f"{len(verbs)} verbs, in step with shell::execute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
