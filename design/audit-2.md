# The distributor, read a second time

The second independent reading of `contracts/src/GladosDistributor.sol`. The
reader did not write the contract or `design/audit.md`. That file's "clean"
verdicts were treated as claims to re-check, and two of them do not hold as
written: the gate's economic argument (F1) and "no path where one epoch's claim
reaches another's funding" (F3, F4). Both are covered below.

    subject     contracts/src/GladosDistributor.sol, 746 lines, 12,543 bytes deployed
    commit      df5ce9a (source unchanged since design/audit.md's fixes landed)
    before      106 claims, 0 failures (`npm test` ran test/run.mjs only)
    after       160 claims, 0 failures (`npm test` now also runs test/audit2.mjs: 54)
    patched     test/fixtures/audit2-fixes.diff -> 12,695 bytes; 106 + 54 pass,
                the 54 with AUDIT2_EXPECT_FIXED=1

**The contract source is unchanged.** Every proposed fix is in
`contracts/test/fixtures/audit2-fixes.diff`. Nothing was deployed, broadcast,
signed or pushed.

## How to reproduce

```bash
cd contracts && npm install && npm run build && npm test          # 106 + 54
# The same 54 asserted the other way. Fails 7 on the current source:
AUDIT2_EXPECT_FIXED=1 node test/audit2.mjs
# and passes all 54 on the patched copy:
cp src/GladosDistributor.sol /tmp/d.sol && patch /tmp/d.sol test/fixtures/audit2-fixes.diff
AUDIT2_DIST=/tmp/d.sol AUDIT2_EXPECT_FIXED=1 node test/audit2.mjs
```

`test/audit2.mjs` labels every line. `Fn present` means a finding that holds
because the defect exists. Each fixable finding inverts under
`AUDIT2_EXPECT_FIXED=1`, so on the current source it is a failing test: seven
fail there. `Cn clean` means a suspicion that was checked and settled the other
way. The fixtures (`OddToken`, `EvilV3Pool`, `TryClaimer`, `GateBorrower`) are
in `test/fixtures/Audit2Fixtures.sol`. `test/build.mjs` never compiles that
file, so the fixtures cannot reach a deployment.

`python tools/loop.py --fork` was run once, against block 75,714,719. It got
as far as "a market epoch opens with the published root" and then died in
`RPCStateManager` (`reading 'balance'`). `design/audit.md` already documents
that failure, and the rule is to read a partial fork run as no evidence. It
adds nothing here.

## Findings, by severity

No critical or high findings. No unprivileged caller can take funds from a
distributor configured with the canonical V3 factory, GLADOS and WETH.

| id | severity | title | test (in `test/audit2.mjs`) | fixed by diff |
|---|---|---|---|---|
| F1 | medium | The gate is one balance that can be passed around, not a holding | `F1 present: one gate-sized balance, passed wallet to wallet…`, `F1 present: a leaf holding nothing borrows the gate…` | no (design) |
| F2 | low | A V3 partial fill charges the whole leaf and strands the rest | `F2 present: a partial fill spends…`, `F2 present: and reclaim returns 0…` | yes |
| F3 | low | The callback pays whatever a vouched pool asks, as often as it asks | `F3 present: a vouched pool that over-asks…`, `F3 present: …calls back twice…` | yes |
| F4 | low | Outbound legs assume the contract is debited exactly `amount` | `F4 present: with a sender-side fee…` | yes |
| F6 | info | Tokens that arrive outside `open*` cannot be recovered | `F6 present: after every epoch is reclaimed…` | no (optional diff below) |
| F7 | info | `_move` edge cases: wrong error on a bad word; a silent no-op is paid as 0 | `F7 present: a 32-byte answer…`, `F7 present: a transfer that returns nothing…` | yes |
| F8 | info | Market-mode slippage is borne by the claimant, bounded only by their `minOut` | `F8 present: with minOut=1…` | no (design) |
| F9 | info | V3 cannot be used without a V2 pair, although the constructor comment says it can | `F9 present: a distributor with a quote and a V3 factory but no V2 pair…` | yes |
| F10 | info | The runbook's NVDA example names a USDG pool against a WETH quote | read over RPC; not in the suite | no (docs) |

### F1 (medium). The gate is one balance that can be passed around

`_admit` reads `balanceOf(msg.sender) >= e.gate` at the moment of the claim.
`design/audit.md` accepted this as "a one-transaction balance requirement" and
argued that it holds economically: to fake it you would buy the gate and sell
it again, paying the 1% buy and 3% sell tax, about 4%.

That argument prices the wrong route. You do not need to trade. GLADOS's
wallet-to-wallet transfers are untaxed; `design/token.md` records this, and
live `Transfer` logs over the last ~2M blocks show it again (for example tx
`0xbdbda274…`, where one amount is passed on three times with no tax leg).
So one gate's worth of GLADOS clears the gate for every leaf, one after another,
at about $0.03 of gas per hop. A contract leaf can also borrow the gate from
any holder who approves it, claim, and return it, all in one call.

    ok  F1 present: one gate-sized balance, passed wallet to wallet, lets every leaf through the gate
    ok  F1 present: a leaf holding nothing borrows the gate for one call and claims

No funds are at risk: every leaf that claims was in the tree anyway. What fails
is the control the header describes as "the whole reason this lives on-chain".
Enforcing a real holding requirement takes a balance snapshot (for example,
gating the leaves off-chain in `distribute.py` against balances at the epoch's
closing block) or a lock. Either way the fix belongs in the design, not in this
file. **The minimum fix:** correct the header and `design/audit.md` so they call
the gate what it is: a balance at claim time that anyone can borrow or relay.

### F2 (low). A V3 partial fill strands funds

`claimOnV3` swaps with a price limit of `MIN_SQRT_RATIO + 1` or
`MAX_SQRT_RATIO - 1`. An honest pool whose liquidity runs out before the input
does stops at that limit and asks the callback for less than `amountSpecified`.
`_admit` has already added the full `amount` to `e.claimed`. The quote the
pool did not take stays in the contract, the bookkeeping says it has been
spent, and `reclaim` returns `funded - claimed`, so nobody can ever recover it.

    ok  F2 present: a partial fill spends 1000000000000000000 but charges the epoch 2000000000000000000; the rest is stranded
    ok  F2 present: and reclaim returns 0 of it, because the bookkeeping says nothing is left

This needs a leaf larger than the pool's whole in-range liquidity. The claimant
would also have to set a `minOut` loose enough to accept the drained-pool fill,
since `deploy.mjs` quotes through QuoterV2. It is rare on the NVDA pools
`design/rwa.md` measured, and possible on a thin pool the operator names.

**Fix:** refuse anything other than an exact fill. The diff sets a one-shot
budget beside `_inFlight`, and the callback must be asked for exactly that
budget. A partial fill reverts, the claim stays unused, and the claimant can
retry later.

### F3 (low). The callback trusts a vouched pool's arithmetic

`uniswapV3SwapCallback` pays `owed` from the contract's single `quote` balance.
It checks nothing against the claim's `amount`, and it can be called more than
once while `_inFlight` is set. Every Market and MarketV3 epoch draws on that one
balance, so a pool that asks for too much, or calls back twice, is paid out of
other epochs' funds.

    ok  F3 present: a vouched pool that over-asks is paid 6000000000000000000 for a 2e18 claim,
        from other epochs' quote (holds 6000000000000000000, owes 10000000000000000000)
    ok  F3 present: and one that calls back twice in one swap is paid twice (4000000000000000000 for 2e18)

This is out of reach with the canonical Uniswap factory: its pools are the
reference bytecode, and they ask exactly once and never for more than the
input. It becomes reachable only if `v3Factory` is not that factory. That value
is an immutable constructor argument, `deploy.mjs` pins it, and it is one typo
or one `GLADOS_V3_FACTORY` environment variable away. `design/audit.md` listed
this as a "stated dependency". It is also the one place where epoch isolation
rests on a third party's code, and the change that removes the dependency costs
three storage writes.

**Fix:** the same budget as F2. The callback requires `owed == _budget`, then
zeroes both `_budget` and `_inFlight`, which makes it one-shot. `claimOnV3`
requires the budget to be spent before it restores the previous values. The
nested-claim case from `design/audit.md` finding 6 still passes: that is
`run.mjs`'s "and still succeeds with a nested claim inside it", run against the
patched tree.

### F4 (low). Outbound legs assume an exact debit

Every inbound leg measures what arrived. No outbound leg measures what left:
`claim`, `claimOnMarket`'s send to the pair, the callback and `reclaim` all
assume the contract's balance drops by exactly `amount`. Suppose a token charges
its fee to the sender, on top of the amount. Then each claim takes the fee out
of other epochs' funds, and the last claimant in the contract fails.

    ok  F4 present: with a sender-side fee, epoch 0's claim spends epoch 1's funds
        (99000000000000000000 held, 100000000000000000000 owed; B: TransferFailed)

GLADOS does not do this. Its tax applies only on legs to or from the pair, and it
comes out of the amount transferred (verified on-chain in `design/token.md`, and
consistent with the logs read for this review). WETH has no fee. So this is a
dependency made concrete, not a live bug. The contract's header argues
explicitly against relying on assumptions of this kind.

**Fix:** `_sendExact` measures the contract's own balance either side and
reverts unless it fell by exactly `amount`. It replaces `_send` in `claim`,
`claimOnMarket` and `reclaim`. One caveat: if a GLADOS dividend were credited
to the distributor *inside* the same transfer, the delta would come out short
and the claim would revert (the claimant could retry). On-chain, dividend
payouts are separate transactions sent by flap's bot (for example
`0x542955408a…`), so this has not been observed.

### F6 (info). Tokens that arrive outside `open*` are stranded

`funded` counts only what arrives inside `openEpoch*`, and `reclaim` returns
only `funded - claimed`. Anything else that lands in the contract stays there
for good: a mistaken transfer, or the GLADOS dividend that flap's dividend
contract pays to holders at or above `minimumShareBalance()` (1,000,000 GLADOS,
per `design/token.md`, a threshold a Direct distributor can reach).

    ok  F6 present: after every epoch is reclaimed the contract still holds 7000000000000000000 of the token, with no function to move it

Optional fix, not in the diff because it adds an operator power. Keep
`mapping(address => uint256) _owed` (incremented by `got` at open, decremented
by `amount` in `_admit` and in `reclaim`) and add
`sweep(asset) onlyOperator`, which sends `balanceOf(this) - _owed[asset]`. That
is only safe once F4's exact-debit check is in, because `_owed` must be the
truth.

### F7 (info). `_move` edge cases

- If a token returns a 32-byte word that is neither 0 nor 1, `abi.decode(ret,
  (bool))` reverts with no data. The caller sees an empty revert, not
  `TransferFailed`. This is the class of error message `design/audit.md`
  already fixed for short answers.
- If a token returns nothing and moves nothing, that counts as success. On the
  inbound side the balance measurement catches it (`NothingFunded`). On a
  `claim` it marks `hasClaimed`, emits `received = 0`, and the claimant is paid
  nothing.

      ok  F7 present: a 32-byte answer that is neither 0 nor 1 reverts with '(no data)' rather than TransferFailed
      ok  F7 present: a transfer that returns nothing and moves nothing marks the claim used and pays 0

**Fix:** decode the word as `uint256` and require it to equal 1. `_sendExact`
(F4) covers the silent no-op.

### F8 (info). Market slippage is the claimant's

A Market claim takes its output from spot reserves in the same transaction,
so a front-running buy moves the fill. `minOut` is the only bound, and the
claimant bears everything inside it. In the mock, a 2% bound refuses a claim
sandwiched behind a 20-quote buy, and `minOut = 1` fills at 69% of the fair
output.

    ok  C8.5 clean: a 2% bound stops a claim sandwiched behind a 20-quote buy
    ok  F8 present: with minOut=1 the same claim fills at 69% of the fair output -- the claimant bears it

This is inherent to the design and documented. On the GLADOS pair the 1% + 3%
round-trip tax is the attacker's cost too, so a sandwich loses money at the
claim sizes in `contracts/README.md`'s own table. That is reasoning, not a test.
Out of scope but adjacent: anyone can call `GladosBurner.claimAndBurn` with any
`minOut`, so the burner's fill price is whatever its caller accepts.

### F9 (info). V3 without V2 is refused

The constructor comment says a chain "can have a V3 deployment and no V2 pair
for this token". But `require((pair_ == 0) == (quote_ == 0))` ties `quote` to
`pair`, and `openEpochOnV3` requires a non-zero `quote`. The test deploys with
`pair = 0` and `quote != 0` and the constructor refuses it.

**Fix:** `require(pair_ == address(0) || quote_ != address(0))`. A pair still
needs a quote, but a quote no longer needs a pair. `openEpochOnMarket` already
refuses when `pair == 0`.

### F10 (info). The runbook's V3 example cannot open

`design/runbook.md` step 7 opens NVDA through `--v3-pool 0xd4eb2120…`. Read
over RPC for this review, that pool's `token0()` is `0x5fc5360d…`, whose
`symbol()` is `USDG`, and its `token1()` is NVDA. The distributor's `quote` is
WETH (`deploy.mjs`'s `QUOTE`), and the constructor requires it to be WETH
because the GLADOS pair is GLADOS/WETH. So `openEpochOnV3` answers `BadPool`.
`deploy.mjs open` checks the pool's tokens against `QUOTE` and refuses first,
so nothing is lost. But the documented path does not work. A USDG-quoted V3
distributor needs its own deployment with `pair = 0, quote = USDG`, which only
F9's fix makes possible. USDG has an issuer with freeze powers
(`design/rwa.md`), and a freeze on the distributor would lock every
USDG-funded epoch, including `reclaim`.

## Threat list, item by item

### 1. Callback authorisation

| check | result | test |
|---|---|---|
| `_inFlight` is zero at rest, after a successful claim, and after a claim that reverts inside the caller's `try/catch` (read directly from storage slot 0) | clean | `_inFlight is zero…` (×4), `C1.1`, `C1.2` |
| A factory-vouched pool calling the callback outside a swap | clean: `BadCallback`, nothing moves | `C1.3`, `C1.4` |
| A second vouched pool calling back during a genuine swap (via the reward token's transfer hook) | clean: refused, exactly `amount` leaves | `C1.5` |
| Double entry within one swap, or over-asking | **F3** | `F3 present…` |
| Partial fill | **F2** | `F2 present…` |
| Reentrancy through a nested claim | clean (`design/audit.md` 6's fix holds; re-run on the patched tree) | `run.mjs`: "and still succeeds with a nested claim inside it" |

Every exit path clears `_inFlight` because a revert rolls back the write. No
path returns normally without restoring it. The one path that could leave it
set is a caller that swallows the revert, and `C1.1` covers that.

### 2. Merkle encoding

The leaf is `keccak(keccak(abi.encode(account, amount)))`. An internal node is
`keccak(abi.encode(lo, hi))` over sorted children. Checked: an internal node
whose 64-byte preimage is exactly `abi.encode(A, amt)` cannot be claimed as
`(A, amt)` (`C2.1`), because the extra hash takes the leaf out of the
internal-node domain (`C2.2`). Also checked against the contract, not the JS
builder: truncated, extended and repeated-element proofs fail (`C2.4`–`C2.6`);
an empty proof against a multi-leaf root fails (`C2.7`); a sibling's proof does
not transfer (`C2.8`); an amount differing only above bit 160 is a different
leaf (`C2.9`). With sorted pairs, a proof carries no direction bits, and `h ==
p` is handled by `<=`. Clean. `design/audit.md` 3 (the leaf has no epoch id) was
re-read and stands as written.

### 3. Holding gate

**F1.** The gate can be passed between wallets or borrowed for a single call.
Flash-borrowing from the pair works too, but it is the expensive route (about
4.6% round trip) and nobody needs it.

### 4. Cross-epoch accounting

Each epoch's spend is capped by its own `funded - claimed`, checked before any
effect (`C4.1`: an over-promised epoch is refused while the contract holds 25e18
for others; `C4.2`). A per-asset solvency invariant (the contract's balance of
each asset is at least the sum of live epochs' `funded - claimed`) is
recomputed after mixed Direct, Market and V3 activity (`C4.3`) and after
reclaims (`C5.6`). It holds under the stated dependencies and fails under two
of them: **F3** (the pool's arithmetic) and **F4** (the token's debit). Rounding
dust: none on Direct or V2. On V3 the only drift is F2. So `design/audit.md`'s
"there is no path where one epoch's claim reaches another's funding" is true
only given those dependencies.

### 5. `reclaim`

It returns exactly the remainder (`C5.2`). It is refused before the deadline
even while a neighbouring epoch has expired (`C5.1`). The reclaimed epoch then
refuses claims (`C5.3`) and a second reclaim (`C5.5`). The neighbouring live
epoch still pays in full (`C5.4`). Deadline boundaries were already covered by
`run.mjs`. Clean, except for the F2 remainder, which `reclaim` cannot see.

### 6. Amount as received

The measurement is a balance delta across one external call. Anything moving
the contract's balance inside that call is attributed to the new epoch. With a
token hook that donates during the pull, the epoch records 15 where the operator
sent 10, and the contract stays solvent (`C6.1`). That is the safe direction,
and GLADOS has no transfer hooks. An outbound transfer inside the pull could
only under-record, which is also safe, or underflow and revert. Two epochs
opened back to back in one block record their own amounts (`C6.2`). Clean. The
separate issue of tokens arriving outside `open*` is **F6**.

### 7. `_call`/`_move`

Explicit `false`, a one-byte answer, and "nothing returned, nothing moved" on
inbound are all refused correctly (`C7.1`, `C7.2`, `C7.4`). 64 bytes whose first
word is `true` are accepted, as OpenZeppelin's rule accepts them (`C7.3`). The
two gaps are **F7**. The no-code case is unreachable, because a typed
`balanceOf` on the same address precedes every `_move`. `design/audit.md`
already reasoned this through and it was re-read here.

### 8. Modes, double claims, claims for others, front-running, slippage

All six wrong-function/mode pairings revert `WrongMode` (`C8.1`). A stranger
who holds the gate and replays A's proof and amount gets `BadProof`, because the
leaf binds `msg.sender` (`C8.2`), and so does the operator (`C8.3`). No claim
function takes a recipient (`C8.4`). A front-runner therefore cannot steal or
redirect a claim, only move its price (**F8**). Double claims were already
covered in `run.mjs`.

### 9. Owner and admin powers

The only state-changing entry points are the three `open*`, the three claims,
`reclaim` and the callback (`C9.1`). Nothing is payable (`C9.2`).
`operator` is immutable, with no transfer, no pause, no upgrade and no cancel.
Once an epoch is funded, the operator's only power over it is `reclaim` after
the deadline. The operator cannot drain a live epoch.

Consequences, stated rather than tested:

- A lost operator key freezes every epoch's unclaimed remainder permanently.
  Claims still work until each deadline.
- A wrong root cannot be withdrawn before its deadline.
- `deadline` has no upper bound, so an epoch can be made effectively
  unreclaimable.
- `gate = 0` is allowed.

These are properties of the trust model, not defects.

### 10. Everything else

- **Integer edges.** `int256(amount)` is safe because `amount <= funded`, a
  measured balance.
- **`_amountOut`.** Cannot overflow for any real supply.
- **`e.funded - e.claimed`.** Cannot underflow, because `claimed <= funded` is
  enforced before the increment.
- **Loops.** The only loop is `_verify` over a caller-supplied proof, so there
  is no gas griefing of anyone else.
- **ETH.** Forced ETH (`selfdestruct`) is inert.
- **Approvals.** The contract grants none. The existing test covers the pair,
  and every path was re-read.
- **Events.** `EpochOpened` does not carry `pool` or `reward` for V3, so an
  indexer must call `epochs(id)` to learn what a V3 epoch pays (info, no test).

## What this review does not establish

- **No swap against a real pool.** The fork run died before claiming. Every
  swap result here comes from a mock, and the V3 mocks are deliberately not
  Uniswap's arithmetic.
- **No test of the real GLADOS on a contract-to-wallet transfer.** That leg
  decides whether F4 is only hypothetical. On-chain logs and `design/token.md`
  both point that way, but no test here exercises the real token on that path.
- **Findings without a test.** F10 comes from RPC reads, not the suite. F8's
  economic claim about taxes is arithmetic, not a test. These are labelled as
  such above.
- **Coverage.** Two readers are better than one, but this is still not an
  engagement.
