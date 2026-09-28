#!/usr/bin/env python3
"""Check a published share log, the way a miner would before trusting one.

    ledgercheck.py PATH_OR_URL          verify the digest and the arithmetic
    ledgercheck.py PATH --worker NAME   and say what that worker is owed
    ledgercheck.py --selftest           check this file against itself

`tools/prooftest.py` earns its place twice -- CI uses it, and it is what a
person would actually run against a stranger's pool before pointing hardware
at it. This is the same thing one layer up: `prooftest` checks that a *job*
pays who the pool says, and this checks that the *record* says what the pool
published.

### Independent on purpose

`pool/src/pool.rs` writes the document, `Pool::load_ledger` reads it back, and
`pool/site/index.html` recomputes the digest in a browser. This is the fourth
reader and the only one CI can run headlessly. A format that four
implementations agree about is a format, and one that only its writer agrees
about is a habit.

### What is checked, and what a green result does not mean

Checked: the digest is the SHA-256 of the canonical rows; the payout
percentages are the shares divided by the window total; the window total is the
sum of its shares; every tally is non-negative.

**Not checked, because nothing here can:** whether the pool actually mined
those shares, whether the coins exist, or who wrote the file. The digest is not
a signature -- anybody who can edit the document can recompute it. What a match
proves is that the rows have not changed since the digest was written, which is
what lets two copies be compared and a later distributor be audited against
one.
"""

import argparse
import hashlib
import io
import json
import sys
import urllib.request

# Must match `Pool::ledger_json` byte for byte: tab-separated, newline
# terminated, and deliberately not JSON -- two JSON writers that agree about a
# document can still disagree about its bytes.
def canonical(doc):
    out = []
    for r in doc.get("shares", []):
        out.append(
            "%s\t%s\t%d\t%d\t%d\t%d\t%d\n"
            % (r["worker"], r["coin"], r["work"], r["accepted"], r["stale"],
               r["bad"], r["duplicate"])
        )
    # Window rows follow the tallies. The order of shares within a coin is part
    # of the record rather than presentation: it is what decides who falls out
    # of the window next.
    for w in doc.get("windows", []):
        for sh in w.get("shares", []):
            out.append("w\t%s\t%s\t%d\n" % (w["coin"], sh["worker"], sh["work"]))
    return "".join(out)


def load(where):
    if where.startswith("http://") or where.startswith("https://"):
        req = urllib.request.Request(where, headers={"User-Agent": "glados/ledgercheck"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    with io.open(where, encoding="utf-8") as f:
        return json.load(f)


def check(doc, worker=None):
    """Answers a list of `(ok, text)`. Empty `ok=False` means it verified."""
    out = []

    def claim(name, cond, detail=""):
        out.append((bool(cond), name + (("  [%s]" % detail) if detail else "")))

    v = doc.get("v")
    # Refused rather than read past: a document of another version has fields
    # that mean something else, and checking it against this canonicalisation
    # would report a mismatch about a format difference.
    claim("the document is a format this knows (v3)", v == 3, "v%s" % v)
    if v != 3:
        return out

    got = hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()
    want = doc.get("digest", "")
    claim("the rows hash to the published digest", got == want,
          got if got == want else "published %s, recomputed %s" % (want or "(none)", got))

    for w in doc.get("windows", []):
        coin = w.get("coin", "?")
        shares = w.get("shares", [])
        total = w.get("total", 0)
        summed = sum(s["work"] for s in shares)
        claim("%s: the window total is the sum of its shares" % coin,
              summed == total, "%d against %d" % (summed, total))

        # The payout is the only number anybody is paid on, so it is
        # recomputed rather than read.
        by = {}
        for s in shares:
            by[s["worker"]] = by.get(s["worker"], 0) + s["work"]
        published = {p["worker"]: p for p in w.get("payout", [])}
        claim("%s: the payout names every worker in the window" % coin,
              set(by) == set(published),
              "window %s, payout %s" % (sorted(by), sorted(published)))
        for name, work in sorted(by.items()):
            p = published.get(name)
            if not p:
                continue
            claim("%s: %s's work is what its shares sum to" % (coin, name),
                  p["work"] == work, "%d against %d" % (p["work"], work))
            if total:
                # Nine decimal places is what the writer emits, so the
                # tolerance is that and not a judgement about how close is
                # close enough.
                want_share = work / total
                claim("%s: %s's percentage is its work over the total" % (coin, name),
                      abs(p["share"] - want_share) < 1e-9,
                      "%.9f against %.9f" % (p["share"], want_share))

    for r in doc.get("shares", []):
        neg = [k for k in ("work", "accepted", "stale", "bad", "duplicate") if r.get(k, 0) < 0]
        claim("%s/%s has no negative counts" % (r.get("worker"), r.get("coin")),
              not neg, ",".join(neg))

    if worker:
        rows = [r for r in doc.get("shares", []) if r["worker"] == worker]
        claim("%s appears in the log" % worker, bool(rows))
    return out


def report(doc, worker):
    print("epoch %s, generated %s, format v%s"
          % (doc.get("epoch"), doc.get("generated_at"), doc.get("v")))
    for w in doc.get("windows", []):
        total = w.get("total", 0)
        window = doc.get("window_work", 0)
        print("\n%s: %d work over %d share(s), window %d"
              % (w.get("coin"), total, len(w.get("shares", [])), window))
        # **Reported, not refused, and the distinction is the point.** PPLNS keeps
        # the last `window_work` of work, so the total in a window should not exceed
        # it -- except that eviction cannot go below one share, so a single share
        # larger than the whole window leaves a total that legitimately overshoots.
        # That is a reachable state and a misconfigured one, which is why this is a
        # note rather than a claim: refusing it would call a real record invalid,
        # and staying silent hides the one thing the operator needs to know.
        #
        # The pool says this at *startup* ("far too small to be a payout window on
        # this chain") and a published ledger carries no startup log, so a reader
        # checking a record has no other way to learn it. Found on this repository's
        # own sample ledger, whose window was 2^28 against a single 2^30 share.
        if window and total > window:
            print("  note: the work in this window exceeds the window itself, by %.1fx."
                  % (total / window))
            print("        PPLNS cannot evict below one share, so a share bigger than")
            print("        the window overshoots it -- which means this window paid the")
            print("        most recent shares and not a window's worth of work.")
        for p in w.get("payout", []):
            mark = " <-- you" if worker and p["worker"] == worker else ""
            print("  %-16s %16d  %8.4f%%%s"
                  % (p["worker"], p["work"], p["share"] * 100.0, mark))


def selftest():
    ok = True

    def claim(name, cond):
        nonlocal ok
        print("%-4s  %s" % ("ok" if cond else "FAIL", name))
        ok = ok and cond

    rows = [{"worker": "a", "coin": "c", "work": 8, "accepted": 2,
             "stale": 0, "bad": 0, "duplicate": 0},
            {"worker": "b", "coin": "c", "work": 4, "accepted": 1,
             "stale": 0, "bad": 0, "duplicate": 0}]
    win = [{"coin": "c", "total": 12,
            "shares": [{"worker": "a", "work": 8}, {"worker": "b", "work": 4}],
            "payout": [{"worker": "a", "work": 8, "share": 8 / 12},
                       {"worker": "b", "work": 4, "share": 4 / 12}]}]
    doc = {"v": 3, "epoch": 1, "generated_at": 1, "window_work": 8,
           "coins": [], "windows": win, "shares": rows}
    doc["digest"] = hashlib.sha256(canonical(doc).encode()).hexdigest()

    claim("a well-formed document verifies", all(o for o, _ in check(doc)))

    # The edit somebody would actually make: move work between workers and
    # leave the digest alone.
    import copy
    t = copy.deepcopy(doc)
    t["shares"][0]["work"] += 1
    claim("one unit of work moved is caught", not all(o for o, _ in check(t)))

    # A payout that does not follow from the shares it claims to summarise.
    t = copy.deepcopy(doc)
    t["windows"][0]["payout"][0]["share"] = 0.99
    t["digest"] = hashlib.sha256(canonical(t).encode()).hexdigest()
    claim("a payout percentage that does not follow is caught, digest or not",
          not all(o for o, _ in check(t)))

    # A window total that is not the sum of its shares -- the number every
    # percentage is divided by.
    t = copy.deepcopy(doc)
    t["windows"][0]["total"] = 99
    t["digest"] = hashlib.sha256(canonical(t).encode()).hexdigest()
    claim("a window total that is not its shares' sum is caught",
          not all(o for o, _ in check(t)))

    # A worker in the payout who never appears in the window.
    t = copy.deepcopy(doc)
    t["windows"][0]["payout"].append({"worker": "ghost", "work": 1, "share": 0.0})
    t["digest"] = hashlib.sha256(canonical(t).encode()).hexdigest()
    claim("a worker paid who is in no share is caught",
          not all(o for o, _ in check(t)))

    t = copy.deepcopy(doc)
    t["v"] = 2
    claim("another format is refused rather than read past",
          not all(o for o, _ in check(t)))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger", nargs="?", help="path or URL to ledger.json")
    ap.add_argument("--worker", help="say what this worker is owed")
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest() else 1
    if not a.ledger:
        ap.print_help()
        return 2

    doc = load(a.ledger)
    results = check(doc, a.worker)
    bad = 0
    for ok, name in results:
        if not ok:
            bad += 1
        if not a.quiet or not ok:
            print("%-4s  %s" % ("ok" if ok else "FAIL", name))
    if not a.quiet:
        print()
        report(doc, a.worker)
    if bad:
        print("\n%d check(s) failed. The digest is not a signature, so this says the "
              "record changed\nsince it was written -- not who changed it." % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
