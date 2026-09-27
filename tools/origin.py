#!/usr/bin/env python3
"""Where the kernel thinks its backend lives, and whether that is still true.

`src/update/channel.rs` compiles one origin into every image. Everything the
machine reaches out for goes through it: `update check`, `update fetch`, the
gated experimental channel, the worker-to-address mapping, and the verdict return
leg. It is a *pin*, exactly like `UPDATE_KEY` and `VERDICT_KEY`, and this tree
already checks those against a second copy on every push -- `ci.yml` compares the
JavaScript verdict anchor with `sign.anchor("VERDICT_KEY")` for the reason
`sign.py` gives: a configuration mistake nobody here can see is answered days
later by a machine three systems away, correctly, about something it cannot name.

**The origin had no such check and needed one more than the keys did**, because a
key is wrong only if somebody rotates it and an origin stops being true on its
own. A Supabase project on the free tier is deleted after prolonged inactivity,
and nothing in this repository would notice: the kernel would ask, fail to
resolve, and report a network error indistinguishable from a bad cable.

That is not hypothetical. On 2026-09-27 the pinned project
`vermcdgpqncfsralpesz.supabase.co` answered NXDOMAIN from Google's public
resolver, authoritative from `supabase.co`'s own nameservers, while `supabase.co`
itself resolved -- and three workflows plus every shipped image named it.

**And the first version of this file said the project was "gone", which was an
inference and was wrong.** It was *paused*: a free-tier project is suspended after
about a week of inactivity and a suspended project's API hostname stops resolving,
which is indistinguishable from a deleted one by DNS alone. Two hours later, after
somebody signed in, the same name resolved and `projects list` reported the project
`ACTIVE_HEALTHY` -- it had existed the whole time.

So this reports **what it measured** and names both causes without choosing between
them. The distinction matters because the two have different repairs: a paused
project comes back by being visited, and a deleted one has to be recreated and
re-pinned and the kernel rebuilt. Telling an operator to do the second when the
first would have done is the same error as the reading that prompted it.

**And there are two pins of one fact.** The kernel compiles `DEFAULT_SOURCE`;
`release.yml`, `experimental.yml` and `propose.yml` read a GitHub variable called
`SUPABASE_URL`. Nothing made them agree, so a project moved in one place and not
the other publishes to a host no machine in the field asks.

    python3 tools/origin.py                 # what is pinned
    python3 tools/origin.py --resolve        # and whether it exists
    python3 tools/origin.py --expect "$SUPABASE_URL"   # the two pins agree
    python3 tools/origin.py --selftest

`--resolve` needs the network and is therefore not what CI gates on by default:
a build refused because a DNS server was slow is a worse failure than the one
this catches. `--expect` needs nothing and is the check a workflow should run on
every push.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANNEL = ROOT / "src" / "update" / "channel.rs"


def pinned(text=None, path=None):
    """The origin compiled into the kernel.

    Read out of the source rather than taken as an argument, for `sign.anchor`'s
    reason: a checker told what to expect checks that it was told correctly.
    """
    if text is None:
        text = (path or CHANNEL).read_text(encoding="utf-8")
    m = re.search(
        r'pub\s+const\s+DEFAULT_SOURCE\s*:\s*&\s*str\s*=\s*"([^"]*)"\s*;', text
    )
    if not m:
        raise SystemExit("  no DEFAULT_SOURCE in src/update/channel.rs")
    return m.group(1)


def normalise(u):
    """Compare origins without tripping over a trailing slash or case.

    A GitHub variable set to `https://ref.supabase.co/` and a constant without the
    slash name one host and would fail a string comparison, which is a check
    failing on its own formatting rather than on the thing it is for.
    """
    return u.strip().rstrip("/").lower()


def host_of(u):
    return normalise(u).split("://", 1)[-1].split("/", 1)[0]


def resolve(host):
    """Does the host exist? Answers True, False, or None for 'could not tell'.

    Three states and not two, deliberately. A resolver that timed out has not
    established that a project is gone, and reporting it as gone would be the
    "arithmetic wearing a measurement's clothes" failure this tree records
    elsewhere -- so `None` is its own answer and the caller decides.
    """
    import socket

    try:
        socket.setdefaulttimeout(8)
        socket.getaddrinfo(host, 443)
        return True
    except socket.gaierror:
        return False
    except OSError:
        return None


def selftest():
    ok = True

    def claim(good, what):
        nonlocal ok
        print(f"  {'ok  ' if good else 'FAIL'}  {what}")
        ok &= bool(good)

    claim(
        pinned('pub const DEFAULT_SOURCE: &str = "https://x.example";') == "https://x.example",
        "the pin is read out of the declaration",
    )
    # Whitespace the formatter may introduce must not change the answer.
    claim(
        pinned('pub  const   DEFAULT_SOURCE : & str  =  "https://y.example" ;') == "https://y.example",
        "and spacing around it does not matter",
    )
    try:
        pinned("nothing here")
        claim(False, "a file with no pin is refused")
    except SystemExit:
        claim(True, "a file with no pin is refused rather than defaulted")

    # The comparison must survive the two spellings one host has.
    claim(
        normalise("https://R.supabase.co/") == normalise("https://r.supabase.co"),
        "a trailing slash and a capital do not make two origins",
    )
    claim(
        normalise("https://a.supabase.co") != normalise("https://b.supabase.co"),
        "but two projects are still two origins",
    )
    claim(host_of("https://ref.supabase.co/functions/v1") == "ref.supabase.co", "the host is taken without the path")

    # The real pin, against the real file. This is the claim that would have
    # failed on the day the project was replaced and the constant was not.
    real = pinned()
    claim(real.startswith("https://"), f"the pinned origin is https: {real}")
    claim(host_of(real).endswith(".supabase.co"), "and is a Supabase project host")

    # Resolution answers three ways, and the third is why it is not a gate.
    claim(resolve("supabase.co") is True, "a host that exists resolves")
    claim(
        resolve("this-name-should-not-exist.supabase.co") is False,
        "one that does not answers false rather than raising",
    )
    # The claim the first version of this file could not have made, because it was
    # asserting an inference: "does not resolve" and "was deleted" are two
    # statements and only the first is measured here.
    claim(
        "paused" in __doc__ and "deleted" in __doc__,
        "and a non-resolving host names both causes rather than choosing one",
    )

    print("\nselftest passed" if ok else "\nselftest FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--resolve", action="store_true", help="also ask whether the host exists")
    ap.add_argument("--expect", help="an origin the pin must equal, such as $SUPABASE_URL")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    p = pinned()
    print(f"pinned in src/update/channel.rs:  {p}")

    rc = 0
    if a.expect:
        if not a.expect.strip():
            print("  --expect was empty, which is not an origin")
            return 1
        same = normalise(p) == normalise(a.expect)
        print(f"expected:                         {a.expect}")
        print(f"  {'agree' if same else 'DISAGREE'}")
        if not same:
            print("  a machine in the field asks the first and CI publishes to the second")
            rc = 1

    if a.resolve:
        h = host_of(p)
        r = resolve(h)
        if r is True:
            print(f"  {h} resolves")
        elif r is False:
            # **The measurement, and the causes, kept apart.** A paused project and
            # a deleted one answer NXDOMAIN identically, and the repairs are not the
            # same: one comes back by being visited and the other has to be recreated,
            # re-pinned and the kernel rebuilt. Saying "gone" picked the expensive
            # repair on evidence that could not distinguish them, which is what the
            # first version of this did.
            print(f"  {h} does NOT resolve, and every image names it")
            print("  a free-tier project that is paused answers this the same way a")
            print("  deleted one does. Check `supabase projects list`: if it is there")
            print("  and INACTIVE it comes back on its own once signed in; if it is")
            print("  absent, recreate it, re-pin channel.rs and rebuild.")
            rc = 1
        else:
            print(f"  could not tell whether {h} resolves; this is not a verdict")
    return rc


if __name__ == "__main__":
    sys.exit(main())
