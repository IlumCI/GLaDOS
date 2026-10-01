# Paying miners in tokenized stock, and why it does not work yet

Robinhood Chain (id 4663) is an Arbitrum Orbit L2 built for tokenized real-world
assets, and $GLADOS already lives on it. So the question is a fair one: if the
chain the token is on was built to carry tokenized equities, why is the payout
denominated in anything else?

This file records what was **measured on the chain** rather than what the
marketing says, because the two disagree in one important place.

## What is actually deployed

Robinhood's own contracts page lists exactly two addresses, **WETH and USDG**.
No stock token is documented anywhere the issuer publishes. They are there
regardless -- found by scanning `Transfer` logs and reading `symbol()` and
`name()` off every contract that emitted one:

| contract | transfers | symbol | name |
|---|---:|---|---|
| `0xd0601ce157db5bdc3162bbac2a2c8af5320d9eec` | 33,539 | NVDA | NVIDIA . Robinhood Token |
| `0x2e0847e8910a9732eb3fb1bb4b70a580adad4fe3` | 19,459 | GOOGL | Alphabet Class A . Robinhood Token |
| `0x1b0e319c6a659f002271b69db8a7df2f911c153e` | 16,123 | GME | GameStop . Robinhood Token |
| `0x4a0e65a3eccec6dbe60ae065f2e7bb85fae35eea` | 13,016 | SPCX | Space Exploration Technologies Corp. |
| `0x12f190a9f9d7d37a250758b26824b97ce941bf54` | 9,817 | AMZN | Amazon . Robinhood Token |
| `0x117cc2133c37b721f49de2a7a74833232b3b4c0c` | 7,090 | SPY | SPDR S&P 500 ETF Trust |

**The undocumented half is the interesting half.** SPCX is SpaceX, which is a
private company: there is no public market for it at all, and its presence says
the chain is carrying more than a mirror of a brokerage's listed inventory.

## They are permissionlessly transferable, which was not the expectation

Every tokenized-equity design published anywhere gates transfers -- ERC-3643,
an identity registry, a whitelist, a transfer agent. So the first question was
which of those NVDA carries. **The answer is none of them.**

    paused()            -> false
    owner()             reverts (absent)
    identityRegistry()  reverts (absent)
    compliance()        reverts (absent)
    transferAgent()     reverts (absent)
    isWhitelisted()     reverts (absent)
    isVerified()        reverts (absent)
    isFrozen()          reverts (absent)

An absent function is weak evidence on its own -- a gate can be spelled a
hundred ways and probing for seven names proves nothing about the eighth. So
the check that settles it is a simulated transfer rather than a survey of
selectors: `transfer` of 0.35 NVDA **from a real holder** (`0xd01d…2257`, whose
balance is 35.27) **to an address that has never been used** (`0x…beef`).

It succeeds.

That is the whole finding. A never-seen address can receive the token, so
nothing on the contract asks who the recipient is, and paying a stranger in
NVDA is mechanically possible today.

## Every liquidity figure this file gave before was wrong, by five orders of magnitude

Twice, and the second time with more confidence than the first. Both errors are
the same one: **read one venue, report a fact about the chain.**

There are at least nine AMM factories on 4663. The one this file measured,
`0x8bceaa40...`, is **Uniswap v2**, and for tokenized stock it holds roughly
**0.04%** of the liquidity. The market is on **Uniswap v3**
(`0x1f7d7550B1b028f7571E69A784071F0205FD2EfA`, 24,535 bytes, 6,578 pools) and
**Uniswap v4** (PoolManager singleton `0x8366a39CC670B4001A1121B8F6A443A643e40951`).
Nothing pointed at v2 except that it was the first factory found.

The same six tokens, quote-side depth, v3 pools only, summed over every fee
tier that exists:

| token | USDG side | WETH side | USD |
|---|---:|---:|---:|
| NVDA | 2,658,529.90 | 246.4444 | **3,265,004.82** |
| SPCX | 1,615,701.89 | 160.9038 | 2,011,669.97 |
| GME | 1,498,504.41 | 17.4786 | 1,541,517.54 |
| SPY | 84,870.78 | 250.8914 | 702,289.38 |
| AMZN | 552,446.23 | ~0 | 552,446.23 |
| GOOGL | 311,815.85 | 0.3610 | 312,704.11 |
| | | | **$8,385,632** |

    on Uniswap v3   $8,385,632
    on Uniswap v2       $61.11      <- what this file called "the chain"

**NVDA alone is $3.27M against the $9.70 reported here as its total depth**, a
factor of 337,000. And v3 is not the largest venue: v4 holds more stock-token
liquidity than v3 does, so the real total is roughly double the table above.

The flow figure died the same way. This file measured `Swap` events on seven v2
pairs, found $683.93 over fourteen days, and called it the basket's throughput.
Chain-wide stock-token volume is in the hundreds of millions of dollars a day.

## So the objection was never real, and neither was the reasoning under it

Both of the earlier verdicts fall:

- **"There is nowhere to buy it."** There is. $8.4M of v3 depth is reachable
  with USDG or WETH from any address, and a settlement of a few hundred dollars
  is a few basis points against a single pool. The custody objection -- that
  acquisition must route through Robinhood's app, an account and an identity
  check, putting the operator in the shape the whole design avoids -- **was
  built entirely on the missing liquidity and goes with it.**
- **"Depth is a stock, flow is what binds."** The reasoning is still correct and
  worth keeping; the arithmetic was applied to the wrong pools. Against real
  volume the ratio it computes is negligible.

**On liquidity grounds a tokenized-stock payout works.** That is the honest
finding and it reopens a route this file closed twice.

## What actually constrains it, measured on the right contracts this time

The "permissionlessly transferable" finding was also wrong, and wrong the same
way: **the controls are not on the token, so probing the token found nothing.**
Every one of the 194 stock tokens is a beacon proxy over
`0xe10b6f6b275de231345c20d14ab812db62151b00`, and that beacon is where the
control surface lives.

    NVDA    isBlocked(address)         reverts
    beacon  isBlocked(address)         answers -- false for a fresh address
    beacon  implementation()           0xb35490d6f9163de4f80d88dc75c3516eb64c5ae2
    NVDA    paused()                   false

So one beacon upgrades all 194 tokens at once and answers the blocklist for all
of them. The implementation carries `adminBurn(address,uint256)`, `pause()`,
`mint`/`burn`, and `updateMultiplier` -- a live rebasing hook, not decoration,
since fifteen tokens currently sit at a multiplier other than 1.0 (CRWD at
4.000000 is a 4:1 split).

The transfer test in this file was not wrong, it was **narrow**: a fresh address
is not a blocked address, so it proves transfers are open by default and says
nothing about whether they can be closed. They can.

**USDG is the same shape and its powers are hidden better.** It is a Paxos
proxy whose privileged functions are not in the main dispatch table at all --
they sit behind a facet router, `facets(bytes4)`:

    facets(pause())            -> 0x58cab81e3d8468a0e90df8cbfacb34535e1de942
    facets(0xdeadbeef)         -> 0x0000000000000000000000000000000000000000

The control returning zero is what makes the hit meaningful: the mapping
discriminates. That facet also carries freeze, unfreeze and
`wipeFrozenAddress`, which destroys a frozen balance. `paused()` reads false
today.

And the tokens are **chain-locked by construction**: `l1Address()` reverts on
them where it answers on WETH, the implementation has no `IArbToken` interface,
and the registry lists exactly one deployment per asset, all on 4663. They
cannot leave through the canonical bridge. For this design that is survivable,
since $GLADOS is on 4663 too -- but a miner paid in NVDA holds something whose
only exits are selling on 4663 or redeeming with Robinhood.

## The verdict, third time

Liquidity is not the blocker and this file said it was, twice, on measurements
of the wrong venue. What remains is a real and different objection: **paying
miners in an asset that a single beacon can pause, blocklist and `adminBurn`,
and that cannot leave the chain, is handing them an IOU with an off-switch.**
That is a judgement about counterparty risk rather than a mechanical
impossibility, and it is the kind of trade this project takes deliberately
elsewhere -- `payout.md` uses unMineable knowing exactly what it is, and says
so, because the exposure is bounded to a day.

### The swap was quoted, at every size that matters

That was the last mechanical unknown, so it was closed rather than left as a
recommendation. Uniswap's `QuoterV2` on this chain
(`0x33e885eD0Ec9bF04EcfB19341582aADCb4c8A9E7`, 8,273 bytes) simulates the swap
against live state, USDG into NVDA on the 500 pool:

    $1     ->   0.004578 NVDA    $218.43/NVDA    +0.000%
    $10    ->   0.045781 NVDA    $218.43         +0.000%
    $84    ->   0.384556 NVDA    $218.43         +0.000%
    $250   ->   1.144512 NVDA    $218.43         +0.000%
    $630   ->   2.884163 NVDA    $218.43         +0.000%
    $5,000 ->  22.889561 NVDA    $218.44         +0.003%

The last column is the effective price against a $1 clip, so it is slippage with
the fee already in both sides. **A quarterly settlement moves the price by
nothing that can be measured**, and the size at which this design would start to
care is four orders of magnitude above what it will ever convert.

What remains unrun is an actual filled transaction, which needs funds on the
chain and is the same caveat `payout.md` records against Across: a quote is not
a fill. But a quote off the real quoter against live reserves is a much stronger
position than the priced-route-never-observed one, and nothing about the
liquidity is in question any more.

**The distributor can do this now**, and what it took is worth recording since
the first version of this paragraph guessed at it.

It read `getReserves()` and called `swap()` on a V2 pair, which was right when
the target was the launchpad's own pair. None of that reaches a V3 pool, so
`Mode.MarketV3` was added along with a reward token the epoch names rather than
the one the contract gates on. **Those two changes had to travel together**: the
entire point is paying somebody in NVDA while still gating on GLADOS, so a mode
that changed venue without splitting reward from gate would have had no caller.

The swap goes direct to the pool rather than through `SwapRouter02`, keeping the
argument the contract already made about routers, and the cost of that is a
callback. V3 inverts V2's shape: a pair is paid and then told to send, while a
pool sends and then calls `uniswapV3SwapCallback` on the caller, which must pay
before the call returns. So the contract now exposes a function whose job is to
pay somebody, and **the authorisation on that function is the whole security
surface of this feature**.

It is a single-call `_inFlight` address, set immediately before `swap` and
cleared immediately after. Authorising by factory lookup instead is the standard
way this callback gets drained, and it is worth naming because it looks
correct: `getPool` says yes to every genuine pool, including one an attacker
deployed for two worthless tokens of their own, from which they can call the
callback and be paid in `quote` for nothing. A test does exactly that, with a
real pool the factory knows about, and asserts both the refusal and that the
contract's balance did not move.

79 tests, 0 failures, up from 59. Deployed size 12,868 bytes against the 24,576
limit. What has still never happened is a swap against a real pool, which needs
funds on the chain.

**A method note, since this is the third time.** Every wrong number here came
from finding one contract that answered and treating it as the population.
`getPair` on four pairs became "the chain"; seven v2 pairs became "the basket";
seven selectors reverting became "no permissioning". The fix each time was to
enumerate rather than sample -- `allPairsLength` was 41,741, the factory list
was nine, and the control surface was one level up at the beacon. **Ask what
the denominator is before quoting the numerator.**

## The legal question was being chased and it does not bind

It was, and the chase is called off. The argument against it is good and worth
recording, because it is a correction to how this file was framed rather than to
anything it measured.

**A tokenized RWA is not the RWA.** The thing on chain is an ERC-20 whose value
rests entirely on an issuer's promise to track something; the share itself is
somewhere else, in a custodian's book, and no amount of holding the token
reaches it. Every measurement above says the same thing from the other end: no
identity registry, no whitelist, no transfer agent, and a transfer to a
never-used address that simply works. **Mechanically, `NVDA` on 4663 is
indistinguishable from any other ERC-20 launched by a private entity**, which is
to say it is indistinguishable from a memecoin. Treating it as though the
securities weight of NVIDIA stock travels with the token is a category error,
and this file was drifting toward making it.

Where the reasoning stops, stated once so nobody has to re-derive it: the
distinction that would matter is not what the token *is* but what it is *sold
as*. A thing that names itself "NVIDIA . Robinhood Token" is representing a
relationship, which is a different position from a token representing nothing
and claiming nothing. That is a fact about the **issuer's** conduct, though, not
about a downstream holder's -- and this pool would be a downstream holder buying
on a market like anyone else.

**Either way it changes no decision here**, and that is the reason to stop
rather than resolve it. The route died on ten dollars of depth. A legal finding
in either direction leaves it dead, and one that came back permissive would
change nothing about a pair that cannot fill an $84 order. The one number to
watch is still depth, and the legal question is worth reopening exactly when
that number moves and not before.

## Built: miners choose what they are paid in (2026-10-01)

The route above is now the product. A miner chooses $GLaDOS, one stock, or a
category basket, and every eligible wallet still gets the same share of each
payout; the choice decides what the share is bought as.

**The menu** (`pool/edge/worker/rewards.js`, mirrored in `src/mine/reward.rs`):
`glados` (default), `nvda`, `spcx`, `googl`, `amzn`, `gme`, `spy`, and four
baskets split evenly: `chips` (NVDA AMD INTC MU AVGO TSM ASML MRVL SKHY DELL),
`os` (MSFT AAPL GOOGL IBM BB), `index` (SPY QQQ), `metals` (GLD SLV USO).

**How it was chosen, measured on 2026-10-01**, enumerating rather than sampling
this time: all 438,340 Uniswap V3 pools from the factory's `PoolCreated` log,
the 5,855 tokens with a USDG pool, USDG held per pool, then only tokens that
are genuine stock tokens -- an EIP-1967 beacon proxy over `0xe10b6f6b...` --
because 279 tokens *call* themselves "Robinhood Token" and a name is free. 50
genuine stock tokens held at least $20k of USDG depth; every menu token's
route ETH -> USDG -> stock was then quoted on QuoterV2 and filled. SK hynix
(`SKHY`, ~$13k) is under that bar and is in the chips basket anyway: payouts
are a few dollars a wallet. The *top six by depth* changed since the table at
the top of this file: SGOV, USO, GLD and QQQ now outrank GME, GOOGL, AMZN and
SPY. The single-stock menu kept the operator's six.

**One contract, one transaction a payout** (`contracts/src/GladosPayout2.sol`):
`payAll(groups)` buys every group's legs -- $GLADOS on its V2 pair, stocks in two
V3 hops through USDG -- and splits each leg equally across that group's wallets,
measuring every arrival. One short leg reverts the whole epoch, so nobody is
ever half-paid. The treasury had not deployed `GladosPayout` yet (`contract:
null` at /treasury), so this replaces it rather than sitting beside it.

**Every way a choice can fail lands on $GLADOS, never on a reverted payout**
(`pool/edge/worker/plan.js`): a wallet the stock beacon blocklists (read per
epoch; unreadable counts as blocked), a basket whose quote fails that epoch,
and any code off the menu.

**Gas is planned per leg.** A ten-stock basket wallet costs ten sends, so the
epoch admits wallets most-work-first until one transaction's budget (24M of
4663's 32M) is spent, and the rest carry. Measured in the simulator at 3,000
miners, 80% choosing stocks: 72-107 wallets a payout, the heaviest 16.8M gas.

**The choice travels in `glados.hello`** as a `reward` field beside the worker
name, is stored per address on the shard that address routes to, and reaches
the treasury in `/work.json`. On a miner it comes from `reward <code>` in
`GLADOS/MINER.TXT`, else the choice saved in the PC's firmware
(`GladosReward`), else $GLaDOS; typing a code on the mining screen saves it and
restarts, as a new address does.
