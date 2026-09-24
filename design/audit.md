# The distributor, read by somebody who did not write it

`contracts/README.md` has said "**Not reviewed by anybody.** Written and tested
in one sitting. A contract that holds tokens should be read by somebody who did
not write it" since the day it shipped, and `design/runbook.md` calls that the
largest open risk in the system. This is that reading. It is one reader over one
morning, which is not an engagement, and the last section says so again in case
this file is read out of order.

    subject     contracts/src/GladosDistributor.sol, 686 lines
    commit      66eaa37, 2026-09-23
    before      79 claims, 0 failures, 12,111 bytes deployed
    after       93 claims, 0 failures, source unchanged

**Nothing in the contract was changed.** Every fix below is a proposed diff and
not a commit. The reason is the same one `drive.py` gives about staging a stale
binary: the value of an audit is that the artefact audited and the artefact
deployed are the same bytes, and editing the source on audit day quietly makes
them two things. The fourteen new claims in `contracts/test/run.mjs` are the
evidence, they pass against the source as it stands, and each one will *invert*
when the corresponding fix lands — which is what makes them regression tests and
not decoration.

**Out of scope, named rather than implied.** `contracts/bridge.mjs`,
`tools/distribute.py`, `supabase/functions/worker` and the pool's own ledger
format were not audited. `contracts/deploy.mjs` was read only where a contract
finding lands in it, and one finding below is entirely its.

## How it was read

Three passes, because reading a contract for "smells" finds whatever the reader
already believes. Pass one wrote down what the contract claims to hold, before
looking for bugs. Pass two worked a threat list drawn from what the code
actually does. Pass three turned every surviving suspicion into a case in
`contracts/test/run.mjs`, which has a 97-line in-process EVM behind it, so the
cost of settling a hypothesis is twenty lines rather than an argument.

**The standing rule was that a finding needs a failing test.** Anything that
could not be expressed that way is in "Stated dependencies" or "What was
checked and came back clean" instead, labelled as reasoning rather than
evidence. Two items below are there for exactly that reason and are weaker than
the five findings above them.

## Invariants, and how each was established

| | invariant | how |
|---|---|---|
| 1 | An epoch's terms never change after `_epochs.push` | **mechanically.** The only writes to mutable state in the file are three `_epochs.push`, `_inFlight`, `hasClaimed`, `e.claimed +=` and `e.reclaimed = true`. Nothing assigns `root`, `gate`, `deadline`, `mode`, `pool` or `reward` anywhere. |
| 2 | `_admit`'s bookkeeping persists | **by reading the signature.** `returns (Epoch storage e)`, so `e.claimed += amount` writes through. Had it been `Epoch memory` the solvency bound would have been a no-op, which is why it was checked rather than assumed. |
| 3 | `hasClaimed` is set before any external call | **by reading.** Both claim paths go through `_admit`, which sets it and increments `claimed` before returning. Checks-effects-interactions by construction, and the existing reentrancy test at `run.mjs:281` already exercises it. |
| 4 | The claim and reclaim windows are disjoint and exhaustive | **by test, at the boundary.** Claim refuses `ts > deadline`; reclaim refuses `ts <= deadline`. Four new claims pin the instant itself, which the suite tested either side of and not at. |
| 5 | Per-epoch solvency holds under a taxing token | **by test.** See finding 4; the half of that worry that concerns solvency does not materialise. |

Invariant 1 is the one worth the two minutes it took: "the epoch is immutable"
is the sort of claim that is true when written and stops being true three
commits later, and a grep answers it permanently where a reading answers it
once.

## Findings

Five, none of them a theft vector, and the first is the one to act on before the
contract is deployed because it cannot be acted on afterwards.

### 1. The V2 pair is never checked against the pair it is meant to be

`openEpochOnV3` validates its pool twice, at lines 382–385: the factory must
name it, *and* its two tokens must be exactly `{quote, reward}`. The constructor
does neither for `pair`. `quoteIsToken0` is `quote_ < token_` (line 260) — a
property of two addresses, never compared against what the pair holds — and
nothing anywhere reads `pair.token0()`.

`claimOnMarket`'s own comment at line 499 says asking for the wrong output "asks
the pool to pay out what is being paid in. Hence `quoteIsToken0`, fixed at
construction." That is true of the flag and says nothing about the pair.

    ok  a distributor is constructed with a pair for two tokens it never mentions
    ok  and it takes funding for a market epoch against that pair
    ok  and then nobody can ever claim it (TooLittleOut(0,1))

Money in, nobody out. `pair` is `immutable`, so there is no setter, no upgrade
and no recovery but `reclaim` after the deadline — and the contract has **never
been deployed**, so this is the one finding whose window to fix is open now and
shut forever afterwards.

It is not a theft vector: a wrong pair bricks the market path rather than
draining it, and the operator is the deployer, so the hostile-pair case is the
operator attacking themselves. It is a permanent misconfiguration reachable by a
typo in one of five constructor arguments, and the same check exists 120 lines
away.

**Fix, two lines in the constructor**, guarded so a distributor with no market
still deploys:

```solidity
if (pair_ != address(0)) {
    address p0 = IUniswapV2Pair(pair_).token0();
    address p1 = IUniswapV2Pair(pair_).token1();
    require((p0 == quote_ && p1 == token_) || (p0 == token_ && p1 == quote_), "pair");
}
```

**And in `deploy.mjs deploy`, regardless**, because a constructor check that
reverts on chain costs a broadcast to discover. It already reads `token0`/
`token1` for the V3 pool during `open` simulation (lines 176–179); the same
comparison belongs in `deploy` for the pair.

### 2. `checkClaim` approves claims the contract refuses

`checkClaim` (line 590) answers for the deadline, the claim record, the gate,
the proof and solvency. It does not read `mode`. So for a `Market` or `MarketV3`
epoch it answers `(true, "")` and `claim()` then reverts `WrongMode`.

    ok  checkClaim calls a market epoch's claim good
    ok  and claim() then refuses the very claim it approved (WrongMode)

This is the miner-facing helper and `deploy.mjs` calls it at line 232 before
claiming. One line fixes it, and the reason it matters more than it looks is
finding 5.

### 3. One root opened twice pays the same leaf twice

The leaf is `keccak256(keccak256(abi.encode(account, amount)))` and carries no
epoch id, no chain id and no contract address; `hasClaimed` is keyed
`[epochId][account]`. So the same published root opened as two epochs is not a
duplicate to be rejected — it is a second entitlement to the same work.

    ok  one root opened as two epochs pays the same leaf twice

Nothing on chain can tell the difference between an operator republishing a root
by mistake and one deliberately funding a second distribution of the same
allocations, so **the check does not belong in the contract**: putting the epoch
id in the leaf would invalidate `tools/distribute.py`, `contracts/test/merkle.mjs`
and every proof either has ever produced, to defend against an operator error a
six-line read loop catches for nothing.

**Fix in `deploy.mjs open`:** during simulation, read `epochCount()` and each
existing epoch's `root`, and refuse a duplicate by name. It already simulates
everything else about the epoch there.

### 4. The Direct path publishes a number it did not send

The contract's header argues at length (lines 28–35) that amounts must be
measured rather than assumed, because a contract that assumes an untaxed
transfer "hands out claims it cannot pay, and the failure lands on whoever
claims last". Every inbound leg does measure. Both market claims measure the
claimant's balance either side. `claim()` does neither:

```solidity
_send(token, msg.sender, amount);
emit Claimed(epochId, msg.sender, amount, amount);
```

At a 3% tax:

    ok  claim() reports 2000000000000000000 received where 1940000000000000000 arrived

**The solvency half of the contract's own fear does not materialise, and that is
worth recording as much as the finding is.** With this tax model the cut comes
out of the amount transferred rather than in addition to it, so the contract's
balance falls by exactly what its bookkeeping says and the last claimant is not
short:

    ok  and it still holds at least what that epoch's bookkeeping owes

So what is wrong is the number in the event, which is what an indexer, a miner
and any published accounting read. `Claimed.received` is the field that exists
to be true.

**Fix:** measure it, the way `claimOnMarket` does at line 513. Note that a
`Direct` claim has no `minOut` and therefore no way for a claimant to bound what
they accept — which is defensible for an untaxed transfer and is the assumption
the header refuses to make everywhere else.

### 5. `deploy.mjs` can open an epoch it cannot claim from

Not a contract finding; found while checking finding 2's blast radius. `open`
creates a `MarketV3` epoch at line 212. `claim` calls `dist.claimOnMarket`
unconditionally at line 251. There is no `claimOnV3` path in the tool at all.

The file's own comment at lines 143–146 explains that the two modes are not
interchangeable and that the contract will not let them be. The claim path then
implements one of them. Composed with finding 2, the sequence a miner gets is:
`checkClaim` says the claim is good, then the tool calls the wrong function,
then the contract reverts `WrongMode` — three things agreeing to be unhelpful.

`design/runbook.md` step 7 documents opening a V3 epoch and is the path the RWA
work in `design/rwa.md` exists to reach, so this is on the route somebody will
actually take.

### 6. `_inFlight` is set and cleared, never saved and restored

A V3 pool pays the recipient *before* calling back — `MockV3.sol:71` models the
real ordering and says so. So a reward token with a transfer hook runs code
inside the outer swap. If that code enters `claimOnV3` again and **succeeds**,
the inner call sets `_inFlight` to its own pool and then clears it to zero on
the way out; the outer pool's callback then finds zero and is refused.

    ok  a V3 claim succeeds with no hook in the way
    ok  and fails once a nested claim has cleared _inFlight (BadCallback)

Both halves are asserted, because a test that only shows the second proves the
fixture is broken rather than the contract.

Funds do not move — the transaction unwinds whole — so this is denial and not
theft, and it needs a reward token the operator chose, which the operator names
per epoch. The authorisation itself is right, and the doc comment at 189–201
correctly identifies and refuses the standard way this callback gets drained
(authorising by `getPool`, which says yes to an attacker's own pool). What is
missing is one line:

```solidity
address prev = _inFlight;
_inFlight = pool_;
IUniswapV3Pool(pool_).swap(...);
_inFlight = prev;
```

## Two more, found in the tool while acting on the first five

### 7. `deploy.mjs deploy` could not deploy this contract at all

The constructor takes five arguments; `deploy.mjs` passed four. It never learned
about `v3Factory`, which was added with the V3 path. So the first command of
`design/runbook.md` -- the one that can only be run once, ever -- threw
`types/values length mismatch` before it reached the network.

Proven offline against the compiled ABI rather than by running it, because the
thing that was wrong needed neither a key nor a chain:

    4 args: REFUSED -- missing arguemnt: types/values length mismatch
    5 args: accepted

A distributor deployed with a zero factory could never open a `MarketV3` epoch,
and unlike `pair` and `quote` -- which may be zero *together*, so the absence is
symmetric and visible -- `v3Factory` has no paired argument to make its absence
obvious.

### 8. `Direct` mode had no operator path

`open` chose between `openEpochOnMarket` and `openEpochOnV3` on whether a pool
was named. `openEpoch` was unreachable, so the tool could claim from three modes
and open two. That matters at this scale rather than in principle:
`contracts/README.md`'s own table prices a `Market` claim's gas at **2,286%** of
what it pays out over a 36-hour event, and `Direct` is the mode that is not
absurd there.

## What is now enforced, and where

The contract is **still unchanged** -- the argument in the header holds, and the
artefact audited is the artefact that would be deployed. What changed is the
tool, which is where three of these findings always belonged:

| finding | now |
|---|---|
| 1, the unchecked `pair` | `deploy` reads `token0`/`token1` back and refuses a pair that is not `{quote, token}`, printing both. The constructor still does not check. |
| 2, `checkClaim` ignores `mode` | `claim` reads the epoch's mode itself and dispatches, so the view's blind spot cannot decide anything. The view is unchanged. |
| 3, one root twice | `open` walks every existing epoch's root and refuses a match, which is where it belongs: nothing on chain can tell that from a deliberate second distribution. |
| 5, no `claimOnV3` | `claim` handles all three modes, quoting V3 against `QuoterV2` on 4663 with `--min-out` as the override. |
| 7, constructor arity | five arguments, with the V3 factory `design/rwa.md` measured. |
| 8, `Direct` unreachable | `--direct` on `open`, funding in the reward token rather than the quote. |
| 4, `claim()` does not measure | **not fixed** -- it is in the contract. `claim --direct` reads the claimant's balance either side and prints the difference when it disagrees with what the event will say. |
| 6, `_inFlight` not restored | **not fixed** -- also the contract, and one line when it is. |

Four claims in `contracts/test/run.mjs` now hold the tool to the contract's
shape: that it passes as many constructor arguments as the ABI declares, and
that it can claim from every mode it can open. They read the tool's source
against the compiled ABI, which is why they exist at all -- a test that needed a
key and a chain would not have been written, and that is exactly how a four-
against-five mismatch shipped in the one command nobody can rehearse.

**One of those checks was wrong when first written**, and it is worth keeping as
the correction: `src.includes("dist.openEpoch")` matches `dist.openEpochOnMarket`,
so it reported `Direct` as supported by the very line that implements `Market`.
It claimed three open modes where there were two. A word boundary fixes it, and
a check that says yes for the wrong reason is the failure this tree keeps
naming.

## What was checked and came back clean

Kept because a suspicion that died is worth as much as one that held, and rather
more than one nobody wrote down.

- **Merkle second preimages.** `_leaf` double-hashes, and the comment at 630–636
  gives the right reason: a singly-hashed leaf over `abi.encode(address,uint256)`
  is 64 bytes, exactly the shape of an internal node under sorted-pair hashing,
  so it could be offered as one. Hashing twice makes the preimage 32 bytes. The
  tree builder carries an odd node up rather than duplicating it, which is the
  convention without the known proof-reuse flaw, and `cross.mjs` already makes
  the compiled contract the arbiter rather than the Python. This is the best-
  reasoned part of the file.
- **The deadline boundary**, all four ways. The instant belongs to the claimant
  and reclaim cannot reach it.
- **`_admit` writing through storage** rather than to a memory copy.
- **Epoch immutability**, mechanically.
- **`reclaim` sending the epoch's own asset** — `e.mode == Mode.Direct ? token :
  quote`, already asserted at `run.mjs:543`.
- **The no-code case in `_move`.** `asset.call(data)` against an address with no
  code returns `(true, "")` and is read as success, which is a real gap in that
  helper and is unreachable: every path that `_move`s `token` or `quote` is
  accompanied by a typed `IERC20(...).balanceOf(...)` on the same address in the
  same function, and a typed external call reverts on empty returndata. So a
  zero-code token address fails at `openEpoch`, not silently. This is reasoning
  and not a test, and it is therefore weaker than the findings above.
- **Cross-epoch drain.** `Direct` epochs hold `token`, market epochs hold
  `quote`, each epoch's spend is bounded by its own `funded - claimed`, and
  `reclaim` sends only that remainder. There is no path where one epoch's claim
  reaches another's funding. Checked by reading every `_send` and `_move` call
  site against the epoch it belongs to.

## Stated dependencies

Properties the contract relies on and does not check. None is a defect; all four
are now written down, which they were not.

- **The pair's fee is 0.3%.** `_amountOut` writes out `997/1000` (line 668) with
  a correct reason — a V2 pair does not expose its fee, the arithmetic lives in
  the router, and this contract is deliberately doing without one. The V3 path
  reads `fee()` off the pool; the V2 path cannot. A fork of V2 on 4663 with a
  different fee would make every `Market` claim either revert on the `k` check
  (higher real fee) or shortchange the claimant (lower). **Worth asserting once
  in `fork.mjs`**, which already has the real pair's address.
- **`quote` is not fee-on-transfer.** `claimOnMarket` computes `out` from the
  reserves and then sends `amount` to the pair; if the pair receives less, the
  `k` check fails and every claim reverts. WETH is not, and nothing says so.
- **`quote`'s total supply is below 2^255.** `int256(amount)` at line 434 is an
  unchecked cast, and a negative `amountSpecified` is an exact-output swap —
  the pool would send `amount` of reward and ask the callback for however much
  `quote` it takes, unbounded by the epoch. It is safe only because `_admit`
  bounds `amount` by `funded - claimed` and `funded` is a measured balance of
  `quote`. That is a correct argument and was nowhere in the file. WETH's supply
  is about 3e24 against 5.8e76, so the margin is enormous — but it is an
  argument about the quote token and not about this contract.
- **A factory-verified V3 pool asks the callback only for what it was offered.**
  The callback sends whichever delta is positive without bounding it against the
  epoch's `amount`. The bound is real and external: the pool is checked against
  `getPool` at open, and an honest exact-input pool asks for exactly
  `amountSpecified`.

## One thing that is a judgement rather than a bug

**The gate is a one-transaction balance requirement, not a holding
requirement.** `_admit` reads `balanceOf(msg.sender)` at claim time, which is
exactly what the design intends and is already tested to the wei at
`run.mjs:181`. The contract's header calls it "a holding requirement, not an
entry fee" (line 26), which is stronger than what the code can enforce: nothing
stops a claimant buying the gate, claiming, and selling in one transaction.

Priced rather than argued, because the arithmetic decides it. A 50,000 GLADOS
round trip pays the token's 1% buy and 3% sell — 4% of the gate's value — plus
slippage on a pool `design/token.md` measures at about $58.9k of depth. Against
`contracts/README.md`'s own table, a claim is worth $0.0021 on a 36-hour event
and $0.51 on a yearly epoch with 1,000 miners. The round trip is orders of
magnitude more expensive than the claim at every row in that table, so the gate
holds economically for as long as those numbers hold, and it is the numbers
rather than the code doing the holding.

That is worth stating in the header rather than fixing in the code: enforcing a
genuine holding period needs a snapshot or a lock, and both are much larger than
the thing they would protect.

## Still unreviewed

- **This is one reader, one morning.** `contracts/README.md`'s "Not reviewed by
  anybody" should become "read once, by an agent that did not write it, findings
  in `design/audit.md`" — which is a different sentence from "reviewed", and the
  difference is the point. A contract that holds tokens deserves more than one
  reader and this does not substitute for one.
- **No swap against a real pool has ever happened**, which `contracts/README.md`
  already says. Every swap in the suite is against a mock, and a passing mock is
  not a filled trade.
- **`fork.mjs` was not run for this audit.** Two items above want it: the real
  token's behaviour on a contract-to-wallet transfer (finding 4's premise, which
  is demonstrated here against a configurable mock rather than against GLADOS),
  and the real pair's fee (dependency 1). It needs no key.
- **The 93 claims run only when somebody remembers.** There is no
  `contracts.yml`; `grep -rn contracts .github/workflows/` finds nothing, while
  `pool.yml` exists precisely to stop a promise from going unenforced. Worth
  more now that there are fourteen more claims to protect.
