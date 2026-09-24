#!/usr/bin/env python3
"""The whole reward loop, on one machine, with nothing at risk.

    python tools/loop.py

Mine, be credited, have an epoch built from that credit, and claim the GLADOS
it entitles you to -- pool, ledger, Merkle tree, contract and all. The only
thing that is not real is the chain: the claim executes in an in-process EVM
rather than on Robinhood Chain, so it costs nothing and proves everything
except that somebody funded it.

### Why this exists

"Can I run the reward mechanism locally" is a question a description answers
badly and a command answers well. Every piece of this has been tested on its
own -- the pool has a selftest, the builder has one, the contract has sixty
claims -- and none of that tells you whether the *seam* between them holds.
This is the seam: a share the pool credited becomes work in a ledger, becomes
an amount in a tree, becomes a leaf a contract accepts, becomes tokens in a
wallet. Five formats, four programs, one number that has to survive all of it.

### What it does not prove

That anybody put GLADOS in the distributor. Funding is the operator's own money
and no simulation can stand in for it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def say(step, text):
    # Flushed, because every stage after this hands stdout to a subprocess, and
    # an unflushed parent prints its heading *after* the output it introduces.
    # The first run printed all five headings at the end, underneath the
    # results they were supposed to label.
    print("\n%s %s" % (step, text), flush=True)


def find_pool():
    """The pool binary, for whichever platform this is."""
    for triple, exe in (("x86_64-pc-windows-msvc", "glados-pool.exe"),
                        ("x86_64-unknown-linux-musl", "glados-pool"),
                        ("", "glados-pool")):
        p = os.path.join(ROOT, "pool", "target", triple, "release", exe) if triple else \
            os.path.join(ROOT, "pool", "target", "release", exe)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3399)
    ap.add_argument("--bits", type=int, default=12,
                    help="share target; low so a Python miner finds shares in seconds")
    ap.add_argument("--basis", choices=("tally", "window"), default="tally",
                    help="which payout basis to settle from; tally is what a "
                         "bounded run wants, and distribute.py refuses to guess")
    ap.add_argument("--seconds", type=float, default=75.0,
                    help="how long to mine; the pool writes its ledger once a minute")
    # **The unit changes with the mode and the default could not serve both.**
    # Without `--fork` the epoch is funded in a mock token this script mints,
    # so a round million is free and reads well. With it, the epoch is funded
    # in *real WETH* -- `fork.mjs` wraps `doc.total` through the deployed WETH
    # contract, deliberately, so the funding is real rather than a balance
    # written into storage by hand. A million there asks the operator to wrap a
    # million ETH against the 100 the fork grants, and every check after it
    # failed on a number that was never about the code.
    #
    # 0.01 WETH is about $24 at the ETH price `design/token.md` measured, which
    # is roughly 0.3% of the way down that file's slippage table -- small enough
    # that the buy tests the mechanism rather than the depth. The real pool holds
    # about 6.4 WETH in total, so anything larger is measuring what a whale does
    # to a thin pool.
    ap.add_argument("--reward", default=None,
                    help="the epoch's total: GLADOS locally, real WETH under "
                         "--fork. Defaults to 1000000e18 and 0.01e18.")
    ap.add_argument("--address", default="0x000000000000000000000000000000000000beef",
                    help="where the reward would go")
    ap.add_argument("--keep", action="store_true", help="leave the working directory behind")
    ap.add_argument("--fork", action="store_true",
                    help="claim against a fork of Robinhood Chain: the real GLADOS "
                         "contract, the real pool, the real tax, and nothing spent")
    a = ap.parse_args()
    if a.reward is None:
        a.reward = "0.01e18" if a.fork else "1000000e18"

    work = os.path.join(ROOT, "out", "loop")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    ledger = os.path.join(work, "ledger.json")
    poollog = open(os.path.join(work, "pool.log"), "w", encoding="utf-8")

    binary = find_pool()
    if not binary:
        print("no pool binary. Build one:", file=sys.stderr)
        print("  cd pool && cargo build --release --target x86_64-pc-windows-msvc", file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- 1
    say("[1/5]", "starting a pool on 127.0.0.1:%d" % a.port)
    pool = subprocess.Popen(
        [binary, "--listen", "127.0.0.1:%d" % a.port, "--ledger", ledger,
         "--cpu-percent", "50", "--share-seconds", "10",
         "loop:sha256d:%d" % a.bits],
        stdout=poollog, stderr=subprocess.STDOUT)
    time.sleep(2)
    if pool.poll() is not None:
        print("the pool exited immediately; see %s" % poollog.name, file=sys.stderr)
        return 1

    try:
        # ------------------------------------------------------------ 2
        # The worker name *is* the payout address, which is what every
        # no-account pool does and what lets this run with no identity server.
        say("[2/5]", "mining as %s for %.0fs" % (a.address, a.seconds))
        miner = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "tools", "poolsoak.py"),
             "--host", "127.0.0.1", "--port", str(a.port),
             "--worker", a.address, "--hours", str(a.seconds / 3600.0),
             "--cycle", str(a.seconds + 5), "--duty", "0.85",
             "--log", os.path.join(work, "miner.log")],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        deadline = time.time() + a.seconds + 20
        while time.time() < deadline and not os.path.exists(ledger):
            time.sleep(2)
        miner.terminate()
        try:
            miner.wait(timeout=10)
        except subprocess.TimeoutExpired:
            miner.kill()

        if not os.path.exists(ledger):
            print("the pool never wrote a ledger; see %s" % poollog.name, file=sys.stderr)
            return 1
    finally:
        pool.terminate()
        try:
            pool.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pool.kill()
        poollog.close()

    doc = json.load(open(ledger, encoding="utf-8"))
    rows = [r for r in doc.get("shares", []) if int(r.get("work", 0)) > 0]
    if not rows:
        print("no credited work. The share target may be too hard for a Python", file=sys.stderr)
        print("miner in the time given; try --bits 10 or --seconds 120.", file=sys.stderr)
        return 1
    say("[3/5]", "the pool credited:")
    for r in rows:
        print("        %s  %s  work=%s  accepted=%s" % (r["worker"], r["coin"], r["work"], r["accepted"]))

    # ---------------------------------------------------------------- 4
    epoch = os.path.join(work, "epoch.json")
    say("[4/5]", "building an epoch worth %s from that ledger" % a.reward)
    # **`--basis` has no default and this passed none**, so every run of this
    # script has failed at step 4 since `distribute.py` learned to refuse the
    # question. It refuses for a good reason -- a ledger carries a PPLNS window
    # and a lifetime tally, they differ by about 4x on a real one, and the tool
    # used to pick one silently -- but the refusal reached a caller that never
    # answered it, and nothing noticed because nothing ran this.
    #
    # `tally` rather than `window`, for the reason `design/runbook.md` step 5
    # gives: the window ages shares out, so a bounded run settles from every
    # share it credited. This script mines for seventy-five seconds and builds
    # one epoch from it, which is a bounded run by construction.
    rc = subprocess.call([sys.executable, os.path.join(ROOT, "tools", "distribute.py"),
                          ledger, "--basis", a.basis, "--total", a.reward,
                          "--out", epoch])
    if rc != 0:
        return rc

    # ---------------------------------------------------------------- 5
    if a.fork:
        say("[5/5]", "claiming against a fork of Robinhood Chain (real token, real pool)")
        script = "fork.mjs"
    else:
        say("[5/5]", "deploying the distributor and claiming, in a real EVM")
        script = "cross.mjs"
    rc = subprocess.call(["node", os.path.join("test", script), epoch],
                         cwd=os.path.join(ROOT, "contracts"),
                         shell=(os.name == "nt"))
    if rc != 0:
        return rc

    e = json.load(open(epoch, encoding="utf-8"))
    print("\n" + "-" * 68)
    print("A share this machine found became work in a ledger, became an amount")
    print("in a Merkle tree, became a leaf the contract accepted, and became")
    print("tokens in a wallet. Five formats, four programs, one number.")
    print()
    for addr, c in sorted(e["claims"].items()):
        print("  %s  %s GLADOS" % (addr, c["amount"]))
    print()
    if a.fork:
        print("Every contract in that last step except the distributor is the one")
        print("deployed on Robinhood Chain, read over RPC. The GLADOS is real, the")
        print("price is real, the tax is real. The fork died with the process.")
    print("What is not proven here is that anybody funded it. That is the")
    print("operator's own money and no simulation stands in for it.")
    print("-" * 68)
    if not a.keep:
        pass  # left in out/loop either way; --keep only documents the intent
    print("\nworking files: %s" % work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
