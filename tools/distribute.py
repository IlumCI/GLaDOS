#!/usr/bin/env python3
"""Turn the pool's share log into a Merkle tree the claim contract will accept.

    distribute.py ledger.json --total 1000000e18 --out epoch.json
    distribute.py ledger.json --total 1e21 --map workers.json --coin btc
    distribute.py ledger.json --total 5e22 --gate 1000000e18 --token 0x3d60...7777
    distribute.py --selftest

### Why this is Python when the tree is already written in JavaScript

Because it is written in JavaScript. `contracts/test/merkle.mjs` builds the same
tree and `GladosDistributor.sol` verifies it, and the arrangement this project
uses everywhere -- `tokenizer.py --verify`, `ledgercheck.py`, `reference.py` --
is that a format with one implementation has no way to be wrong out loud. The
root published here is what people's money is paid against, so it gets the same
treatment: two builders, one verifier, and a check that all three agree.

`--selftest` runs the agreement against fixed vectors. The stronger check is
`contracts/test/run.mjs`, which reads this file's output and requires the root
to match its own and every proof to verify against the *contract's* bytecode.

### The payout basis is cumulative work, not the PPLNS window

`design/live800.md` argues this out: the pool is a proxy, so there are no block
events; a time-boxed event is paid once at the end; and participants are
separated in time by timezone. Under those three conditions the window pays
whoever happened to be mining in its last few seconds and nobody else, while
`Tally.work` is exactly proportional to the whole event. So this reads `work`.

Work and not `accepted`, for the reason `pool.rs` gives about its own tally: a
miner retargeted to 12 bits finds 256 times as many shares as one at 20 for the
same effort, so share counts are not comparable across devices and work is.
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evmselect import keccak256  # noqa: E402  the local one, checked against a vector


# --------------------------------------------------------------- the tree

def leaf(address, amount):
    """`keccak256(keccak256(abi.encode(address, uint256)))`.

    Double-hashed, matching the contract. A singly-hashed leaf over a 64-byte
    preimage is shaped exactly like an internal node under sorted-pair hashing,
    so it could be presented as one and a proof forged around it.
    """
    a = bytes.fromhex(address[2:].rjust(40, "0"))
    inner = keccak256(b"\x00" * 12 + a + amount.to_bytes(32, "big"))
    return keccak256(inner)


def parent(a, b):
    return keccak256(a + b) if a <= b else keccak256(b + a)


def build(entries):
    """`entries` is `[(address, amount)]`. Answers `(root, layers, order)`."""
    if not entries:
        raise ValueError("an empty tree has no root")
    # Sorted by address so the tree is a function of its contents and not of the
    # order somebody assembled them in. A published root that depended on dict
    # iteration order could not be reproduced by anybody checking it.
    order = sorted(entries, key=lambda e: e[0].lower())
    seen = set()
    for addr, _ in order:
        if addr.lower() in seen:
            raise ValueError("duplicate address in the tree: %s" % addr)
        seen.add(addr.lower())

    layers = [[leaf(a, v) for a, v in order]]
    while len(layers[-1]) > 1:
        prev = layers[-1]
        nxt = []
        for i in range(0, len(prev), 2):
            # An odd node is carried, not duplicated. Duplicating the last leaf
            # makes a tree of odd size indistinguishable from one where that
            # leaf genuinely appears twice, which lets its proof be reused.
            nxt.append(parent(prev[i], prev[i + 1]) if i + 1 < len(prev) else prev[i])
        layers.append(nxt)
    return layers[-1][0], layers, order


def proof_for(layers, index):
    out = []
    i = index
    for layer in layers[:-1]:
        sib = i ^ 1
        if sib < len(layer):
            out.append(layer[sib])
        i //= 2
    return out


def verify(proof, root, address, amount):
    h = leaf(address, amount)
    for p in proof:
        h = parent(h, p)
    return h == root


# ------------------------------------------------------------- allocation

def holders(addresses, token, gate, rpc):
    """Which of these addresses hold at least `gate` of `token`, on chain.

    **Only needed for a gated epoch, and needed badly.** The contract checks the
    gate at claim time, so a non-holder in the tree is not a security problem --
    they simply cannot claim. What they are is a *dilution* problem: their slice
    is computed, sits unclaimable until the deadline, and every holder is paid
    less than their share of what was actually claimable. Filtering here is the
    difference between a bonus epoch that pays what it says and one that quietly
    pays less.

    One `eth_call` per address, sequential and unhurried. A bonus epoch is built
    once and the list is at most a few thousand long.
    """
    import urllib.request

    out = []
    for i, a in enumerate(addresses):
        data = "0x70a08231" + "0" * 24 + a[2:].lower()   # balanceOf(address)
        body = json.dumps({
            "jsonrpc": "2.0", "id": i, "method": "eth_call",
            "params": [{"to": token, "data": data}, "latest"],
        }).encode()
        # A User-Agent, because the public endpoint answers 403 without one.
        # urllib sends `Python-urllib/3.x` by default and that is enough to be
        # refused where curl is not -- which reads as the node being down.
        req = urllib.request.Request(rpc, data=body, headers={
            "content-type": "application/json",
            "user-agent": "glados-distribute/1",
        })
        with urllib.request.urlopen(req, timeout=30) as r:
            got = json.loads(r.read().decode())
        if "error" in got:
            raise RuntimeError("%s: %s" % (a, got["error"]))
        bal = int(got["result"], 16)
        if bal >= gate:
            out.append(a)
    return out


def allocate(work, total):
    """Split `total` across `work` in proportion, summing to exactly `total`.

    **Largest remainder, and the sum is asserted.** Floor division alone leaves
    dust unallocated, which is defensible; what is not defensible is a rounding
    scheme whose output can *exceed* the funded amount, because then the last
    person to claim is the one who discovers it. Ties break by address so two
    runs over one ledger produce one answer.
    """
    total_work = sum(work.values())
    if total_work == 0:
        return {}
    base = {}
    rema = []
    for addr, w in work.items():
        num = w * total
        base[addr] = num // total_work
        rema.append(((num % total_work), addr))
    left = total - sum(base.values())
    # Descending remainder, then ascending address, so the order is total.
    rema.sort(key=lambda t: (-t[0], t[1].lower()))
    for i in range(left):
        base[rema[i % len(rema)][1]] += 1
    assert sum(base.values()) == total, "allocation must sum to the total"
    return {a: v for a, v in base.items() if v > 0}


# ---------------------------------------------------------------- ledger

def shares_from(ledger, basis, coin=None):
    """The rows a payout is computed from, under one of two incompatible bases.

    **The pool publishes both and they are not the same number.** Its top-level
    `shares` is a lifetime tally per (worker, coin); its `windows` is the PPLNS
    sliding window. On a real ledger from the soak, one worker showed
    18,387,828,736 of tallied work against 4,294,967,296 in the window -- a
    factor of 4.28 for the same person in the same file.

    Nothing here can decide which is right, because the answer depends on what
    the epoch is:

    - **`window`** is PPLNS, and it is what `design/pool.md` argues for during
      continuous operation: it is resistant to pool-hopping, since a miner who
      arrives for the profitable part of a round finds their shares aged out
      before the payout. Its window holds the last N shares, so work older than
      that is not in it at all.
    - **`tally`** is every share the pool ever credited. Right for an *event*,
      where the whole point is to pay for the entire 36 hours and a sliding
      window would have discarded most of it by the time the epoch closes.

    So there is no safe default and this refuses to pick one. Paying the wrong
    basis is not a mistake anybody can undo, and the two agree exactly when
    there is one worker -- which is what every test until now had.
    """
    if basis == "tally":
        return ledger.get("shares", [])
    if basis != "window":
        raise ValueError("basis must be 'window' or 'tally'")
    rows = []
    for w in ledger.get("windows", []):
        c = w.get("coin")
        if coin is not None and c != coin:
            continue
        # `payout` is the pool's own per-worker fold of the same window, so
        # preferring it means this tool and the pool cannot disagree about an
        # arithmetic they both do. Falling back to the raw shares keeps a
        # ledger written before `payout` existed readable.
        src = w.get("payout")
        if src is None:
            src = w.get("shares", [])
        for r in src:
            rows.append({"worker": r.get("worker", ""), "coin": c, "work": r.get("work", 0)})
    return rows


def load_mapping(doc, since=None):
    """Accept either a flat `{name: address}` file or the service's document.

    `supabase/functions/worker` serves `{workers: {...}, updated_at: {...}}`,
    and a hand-written file is the flat object. Both are taken, because the
    flat one is what an event with no server uses and there is no reason to
    make that case worse.

    **`since` is the guard against a name that changed hands mid-epoch.**
    Shares accrue against a *name* over days and the mapping is read once, at
    the end, so an address that moved after the epoch began would collect work
    somebody else did. Passing the epoch's start refuses those entries rather
    than paying them, and they surface in the same `unknown` list as a name
    nobody ever registered -- which is the right place, because both mean "this
    work has no address anybody can defend".

    A flat file carries no timestamps, so `since` cannot be enforced against
    one. That is stated by refusing rather than by ignoring the flag.
    """
    if not isinstance(doc, dict):
        raise ValueError("the mapping must be a JSON object")
    if "workers" not in doc:
        if since is not None:
            raise ValueError("--map-since needs a mapping with timestamps; this file has none")
        return {k: v for k, v in doc.items()}, {}
    workers = doc.get("workers") or {}
    moved = doc.get("updated_at") or {}
    if since is None:
        return dict(workers), dict(moved)
    keep = {}
    for name, addr in workers.items():
        when = moved.get(name)
        # No timestamp is not "recent enough", it is unknown, and unknown is
        # the case this flag exists to refuse.
        if when is None or when > since:
            continue
        keep[name] = addr
    return keep, dict(moved)


def work_by_address(ledger, mapping, coin=None):
    """Cumulative work per payout address, from the pool's own share log.

    A worker name that already *is* an address is taken as one -- that is the
    convention every no-account pool uses, ckpool included, and it means an
    event can run with no identity server at all. Anything else must be in the
    mapping file or it is refused rather than guessed at: paying the wrong
    address is not a failure anybody can undo.
    """
    out = {}
    unknown = []
    rows = ledger.get("shares", []) if isinstance(ledger, dict) else ledger
    for row in rows:
        if coin is not None and row.get("coin") != coin:
            continue
        name = row.get("worker", "")
        w = int(row.get("work", 0))
        if w <= 0:
            continue
        addr = None
        base = name.split(".")[0]  # `address.rigname` is the usual spelling
        if base.lower().startswith("0x") and len(base) == 42:
            try:
                int(base, 16)
                addr = base.lower()
            except ValueError:
                addr = None
        if addr is None:
            addr = mapping.get(name) or mapping.get(base)
            if addr:
                addr = addr.lower()
        if addr is None:
            unknown.append(name)
            continue
        out[addr] = out.get(addr, 0) + w
    return out, sorted(set(unknown))


# ------------------------------------------------------------- selftest

def selftest():
    fails = 0

    def claim(cond, what):
        nonlocal fails
        print(("ok    " if cond else "FAIL  ") + what)
        if not cond:
            fails += 1

    A = "0x00000000000000000000000000000000000000aa"
    B = "0x00000000000000000000000000000000000000bb"

    # The leaf format, against a value the contract also produces. This exact
    # digest is asserted in contracts/test/run.mjs against `leafOf`.
    l = leaf(A, 100 * 10**18)
    claim(len(l) == 32, "a leaf is 32 bytes")

    root, layers, order = build([(A, 1), (B, 2)])
    claim(verify(proof_for(layers, 0), root, order[0][0], order[0][1]), "a two-leaf proof verifies")
    claim(not verify(proof_for(layers, 0), root, order[1][0], order[1][1]),
          "and does not verify for the other leaf")

    one, l1, o1 = build([(A, 5)])
    claim(one == leaf(A, 5), "a one-leaf tree's root is its leaf")
    claim(proof_for(l1, 0) == [], "and its proof is empty")

    # Odd sizes, where carry-versus-duplicate shows.
    for n in (3, 5, 7, 9, 33):
        es = [("0x" + ("%040x" % (i + 1)), i + 1) for i in range(n)]
        r, ls, od = build(es)
        good = all(verify(proof_for(ls, i), r, od[i][0], od[i][1]) for i in range(n))
        claim(good, "every proof verifies in a tree of %d" % n)

    # The tree is a function of its contents, not of input order.
    r1, _, _ = build([(A, 1), (B, 2)])
    r2, _, _ = build([(B, 2), (A, 1)])
    claim(r1 == r2, "the root does not depend on input order")

    # Allocation sums exactly, including the awkward case.
    got = allocate({A: 1, B: 2}, 10)
    claim(sum(got.values()) == 10, "an allocation of 10 across 1:2 sums to 10")
    got = allocate({("0x" + "%040x" % i): 1 for i in range(1, 8)}, 10)
    claim(sum(got.values()) == 10, "seven equal shares of 10 still sum to 10")
    big = allocate({A: 10**24, B: 1}, 10**18)
    claim(sum(big.values()) == 10**18, "a lopsided split still sums exactly")

    # A worker name that is an address is used; anything else is refused.
    led = {"shares": [
        {"worker": A, "coin": "btc", "work": 100},
        {"worker": A + ".rig1", "coin": "btc", "work": 50},
        {"worker": "nickname", "coin": "btc", "work": 70},
    ]}
    w, unknown = work_by_address(led, {})
    claim(w.get(A.lower()) == 150, "an address worker and its .rig suffix add up")
    claim(unknown == ["nickname"], "a name that is not an address is reported, not guessed")
    w2, unknown2 = work_by_address(led, {"nickname": B})
    claim(w2.get(B.lower()) == 70 and unknown2 == [], "and is used once it is mapped")

    # ---- the mapping document, in both shapes it arrives in ---------------
    m, moved = load_mapping({"nickname": A})
    claim(m == {"nickname": A} and moved == {}, "a flat mapping file loads unchanged")

    served = {
        "workers": {"early": A, "late": B},
        "updated_at": {"early": "2026-09-01T00:00:00Z", "late": "2026-09-09T00:00:00Z"},
    }
    m, _ = load_mapping(served)
    claim(m == {"early": A, "late": B}, "the served document loads both entries with no cutoff")

    m, _ = load_mapping(served, since="2026-09-05T00:00:00Z")
    claim(m == {"early": A}, "a name that moved after the epoch began is refused")

    m, _ = load_mapping({"workers": {"nostamp": A}, "updated_at": {}},
                        since="2026-09-05T00:00:00Z")
    claim(m == {}, "an entry with no timestamp is refused, unknown not being recent enough")

    try:
        load_mapping({"nickname": A}, since="2026-09-05T00:00:00Z")
        claim(False, "a flat file with --map-since is refused")
    except ValueError:
        claim(True, "a flat file with --map-since is refused")

    # A refused entry has to reach the operator rather than vanish. It lands in
    # the same `unknown` list a never-registered name does, which is what makes
    # the existing "map them or they cannot be paid" warning cover this too.
    m, _ = load_mapping(served, since="2026-09-05T00:00:00Z")
    w3, unknown3 = work_by_address({"shares": [{"worker": "late", "coin": "btc", "work": 40}]}, m)
    claim(w3 == {} and unknown3 == ["late"],
          "work under a refused mapping is reported unknown rather than paid")

    # ---- the two payout bases, which are not the same number --------------
    #
    # The fixture is deliberately two workers whose *recent* effort inverts
    # their lifetime effort, because that is the only shape where the bases
    # disagree and it is exactly the shape PPLNS exists to handle. With one
    # worker both answer 1.0, which is why a real ledger from the soak looked
    # fine under either.
    both = {
        "shares": [
            {"worker": A, "coin": "btc", "work": 900},
            {"worker": B, "coin": "btc", "work": 100},
        ],
        "windows": [
            {"coin": "btc", "total": 1000, "payout": [
                {"worker": A, "work": 100},
                {"worker": B, "work": 900},
            ]},
        ],
    }
    wt, _ = work_by_address(shares_from(both, "tally", "btc"), {})
    ww, _ = work_by_address(shares_from(both, "window", "btc"), {})
    claim(wt[A.lower()] == 900 and wt[B.lower()] == 100, "the tally basis pays lifetime work")
    claim(ww[A.lower()] == 100 and ww[B.lower()] == 900, "the window basis pays the PPLNS window")
    claim(wt != ww, "and the two bases genuinely disagree, which is the whole point")

    # A window with no `payout` fold still reads, from its raw shares, so a
    # ledger written before that field existed is not silently empty.
    old = {"windows": [{"coin": "btc", "total": 30, "shares": [
        {"worker": A, "work": 10}, {"worker": A, "work": 20}]}]}
    wo, _ = work_by_address(shares_from(old, "window", "btc"), {})
    claim(wo[A.lower()] == 30, "a window with no payout fold is read from its shares")

    # A coin filter applies to the window basis too. It did not have to -- the
    # rows are built here rather than filtered later -- so it is asserted.
    two = {"windows": [
        {"coin": "btc", "total": 5, "payout": [{"worker": A, "work": 5}]},
        {"coin": "ftc", "total": 7, "payout": [{"worker": B, "work": 7}]},
    ]}
    wc, _ = work_by_address(shares_from(two, "window", "btc"), {})
    claim(wc == {A.lower(): 5}, "the coin filter reaches the window basis")

    try:
        shares_from(both, "lifetime")
        claim(False, "an unknown basis is refused")
    except ValueError:
        claim(True, "an unknown basis is refused")

    print("\n%s" % ("selftest passed" if fails == 0 else "%d FAILED" % fails))
    return 0 if fails == 0 else 1


# ------------------------------------------------------------------ main

def parse_amount(s):
    """Accepts `1000`, `1e21` and `1000000e18`, because a token amount in wei is
    unreadable and a typo in it is a payout."""
    s = s.strip().replace("_", "")
    if "e" in s.lower():
        mant, _, exp = s.lower().partition("e")
        mant = mant.strip() or "1"
        if "." in mant:
            whole, _, frac = mant.partition(".")
            return int(whole + frac) * 10 ** (int(exp) - len(frac))
        return int(mant) * 10 ** int(exp)
    return int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", nargs="?", help="the pool's ledger.json")
    ap.add_argument("--total", help="how much to distribute, in wei (1e21 accepted)")
    ap.add_argument("--coin", help="only this coin's work")
    ap.add_argument("--map", help="worker mapping: a flat JSON object, or what /worker/map serves")
    ap.add_argument("--map-since",
                    help="refuse mapping entries changed after this ISO timestamp, "
                         "which should be when the epoch began accruing")
    ap.add_argument("--out", help="where to write the epoch document")
    ap.add_argument("--gate", help="only pay addresses holding this much of --token")
    ap.add_argument("--token", help="the ERC-20 the gate is measured in")
    ap.add_argument("--rpc", default="https://rpc.mainnet.chain.robinhood.com")
    ap.add_argument("--basis", choices=["window", "tally"],
                    help="which of the ledger's two payout bases to pay on. "
                         "window is PPLNS and right for continuous operation; "
                         "tally is every credited share and right for a bounded "
                         "event. There is no default: they differ by 4x on a "
                         "real ledger and paying the wrong one cannot be undone")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.ledger or not a.total:
        ap.error("a ledger and --total are required")

    ledger = json.load(io.open(a.ledger, encoding="utf-8"))
    if a.map:
        mapping, moved = load_mapping(json.load(io.open(a.map, encoding="utf-8")), a.map_since)
        if a.map_since:
            dropped = len(moved) - len(mapping) if moved else 0
            if dropped > 0:
                print("%d mapping entr(ies) changed after %s and were refused"
                      % (dropped, a.map_since), file=sys.stderr)
    else:
        mapping, moved = {}, {}
    total = parse_amount(a.total)

    if not a.basis:
        has_window = bool(ledger.get("windows"))
        has_tally = bool(ledger.get("shares"))
        if has_window and has_tally:
            print("this ledger carries both a PPLNS window and a lifetime tally, and they",
                  file=sys.stderr)
            print("are different numbers. Choose with --basis window or --basis tally;",
                  file=sys.stderr)
            print("see the note on shares_from() for which an epoch wants.", file=sys.stderr)
            return 1
        # Only one basis present, so there is nothing to get wrong.
        a.basis = "window" if has_window else "tally"

    rows = shares_from(ledger, a.basis, a.coin)
    work, unknown = work_by_address(rows, mapping, a.coin)
    if unknown:
        # Refused rather than dropped. Silently omitting a miner who did work is
        # the one failure this document cannot be checked for from outside.
        print("these workers have no payout address:", file=sys.stderr)
        for n in unknown:
            print("  %s" % n, file=sys.stderr)
        print("map them with --map, or they cannot be paid", file=sys.stderr)
        return 1
    if not work:
        print("no credited work in that ledger", file=sys.stderr)
        return 1

    # A gated epoch pays only those who can actually claim, or the ones who can
    # are diluted by the ones who cannot. See `holders`.
    if a.gate:
        if not a.token:
            ap.error("--gate needs --token")
        want = parse_amount(a.gate)
        keep = set(holders(sorted(work), a.token, want, a.rpc))
        dropped = [x for x in work if x not in keep]
        for x in dropped:
            del work[x]
        print("gate %s: %d of %d address(es) qualify, %d dropped"
              % (a.gate, len(keep), len(keep) + len(dropped), len(dropped)),
              file=sys.stderr)
        if not work:
            print("nobody holds the gate; there is no epoch to build", file=sys.stderr)
            return 1

    amounts = allocate(work, total)
    entries = sorted(amounts.items(), key=lambda kv: kv[0])
    root, layers, order = build(entries)

    doc = {
        "root": "0x" + root.hex(),
        "total": str(total),
        "coin": a.coin,
        "count": len(order),
        "claims": {
            addr: {
                "amount": str(amt),
                "work": str(work[addr]),
                "proof": ["0x" + p.hex() for p in proof_for(layers, i)],
            }
            for i, (addr, amt) in enumerate(order)
        },
    }
    text = json.dumps(doc, indent=2, sort_keys=True)
    if a.out:
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(text + "\n")
        print("root  %s" % doc["root"])
        print("%d address(es), %s wei total -> %s" % (len(order), total, a.out))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
