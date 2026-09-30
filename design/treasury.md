# The treasury: every way it can fail, and what each one costs

`pool/edge/worker/{flow,runner,epoch,treasury}.js`. The pool's own hot wallets
turn zpool's RVN into $GLADOS paid equally to every eligible miner, with nobody
at a keyboard:

    RVN accumulates -> ChangeNOW exchange -> RVN sent -> ETH arrives on 4663
      -> GladosPayout deployed (once) -> buyAndPay: wrap, buy, split, measure

`flow.js` decides and `runner.js` acts, one action per five-minute tick. Every
action that moves money is **written to storage before the network sees it**,
so an object evicted mid-broadcast wakes, finds the same bytes and sends them
again rather than making new ones.

## What was wrong after the second review, and is not now

| | failure | now |
|---|---|---|
| C1 | a miner naming the pair (or any contract) as their address made every payout revert, for everyone, forever | excluded at build (`eth_getCode`, plus the pair, token, WETH, the contract and the hot wallet) and refused at connect. EIP-7702 delegations are ordinary accounts and are paid |
| H2 | a wallet of many small payouts sized an exchange under ChangeNOW's minimum and stalled every tick; dust sent to a public address did the same | coins chosen largest first, anything under 0.01 RVN ignored, the amount computed from exactly those coins, and the exchange records which coins it will spend. Coins vanishing from under it drop the exchange before anything is sent |
| H3 | a transaction a node refused but another node mined was counted a failure, and the next tick paid the same epoch again | every EVM transaction is signed at the **confirmed** nonce, so a replacement and the thing it replaces can never both be mined; earlier attempts at that nonce travel with the new one and whichever the chain kept is recorded as what it was. A nonce used by something that is none of ours lets that transaction go, as an alerted incident |
| H4 | lost keys were silently regenerated, and every later payout went to addresses nobody watched | `TREASURY_RVN` / `TREASURY_EVM` pin the published addresses; missing or different keys refuse |
| M1 | a 429 counted as a revert, three of them halted | only a node's definite answer counts. 429, 5xx, timeouts and garbage are retried with backoff and then leave the step undone |
| M2/L3 | an unreadable balance, or a miss of the floor, consumed that miner's work | the work carries to the next epoch; the snapshot never moves backwards |
| M3 | a failed exchange was forgotten | written off with the id kept and watched (ChangeNOW wants a ticket), alerted, and the treasury carries on; a refund is let go and retried |
| M4 | the epoch list and snapshot were single values that outgrow 128 KiB | chunked, written atomically with the state that refers to them, and each payout's record is keyed by its own transaction hash |
| M5 | the buy tax was assumed to be 1% | read per payout and priced into the slippage bound; over 25% payouts wait and resume by themselves |
| L1 | a stale `TREASURY_RESUME` resumed every later halt | there are no halts: pauses end by themselves, and a *change* to `TREASURY_RESUME` ends one early |
| L2 | dry mode wrote state | dry mode writes nothing |

Two more from building the simulator: a buy waits while the pair's price is
more than 15% off the median of its recent readings (a pair shoved for one
block is the moment a buy gets the fewest tokens), and an EVM transaction
neither mined nor refused for two hours is replaced at the same nonce at
today's gas price.

## The simulator, and what it found that reading did not

`node pool/edge/worker/stress.test.mjs [seeds]` runs `runner.js` unmodified
against a simulated Ravencoin (inputs, fees, dust, coinbase maturity, mempool),
ChangeNOW (deposits, expiry, refund, failure, a stuck exchange, lying about
addresses) and chain 4663 (real signature recovery, nonces, replacement and
underpricing, reverts, receipts lagging the nonce count). Every seam can
refuse, rate-limit, time out, answer garbage, accept and then report an error,
or evict the object before or after a write. It checks the *world* after every
tick, never the treasury's account of itself: no coin spent twice, no exchange
funded twice or after it lapsed, no ETH anywhere but gas and the contract, at
most one contract, every mined payout recorded as exactly what it paid, at
most one unrecorded and that one in flight, nobody paid who must not be, no
address credited more work than it did, a snapshot that only rises, no key in
a log line, and dry mode touching nothing.

**The first run found a false halt, which is the finding worth keeping.** Under
receipt lag, the treasury saw its nonce used, found no receipt for its own
deploy, waited fifteen minutes and halted with "a transaction that is none of
ours" -- about a deploy that had been mined. Safe, and wrong: it would stop
payouts for a lag. Time alone cannot bound a lag measured in blocks, and blocks
alone cannot bound one on a chain whose blocks are 0.1 s apart (4663's,
measured: 1,000 blocks in 100 s). So both must pass: fifteen minutes **and**
3,000 blocks.

**The second run found the one that would have cost money.** 60 random storms,
four broken: three payouts mined and never recorded, and a second contract
deployed. One cause for all four: a transaction counted as failed -- refused by
one node, or stuck past two hours -- is not a transaction that cannot be mined.
Another mempool held it, and it landed after the treasury had moved on. The
next transaction was signed at the *next* nonce, so both were mined: a second
deploy, or an unrecorded payout whose work the following epoch would have paid
again.

The nonce reuse was meant to close exactly this, and did so only while the
earlier attempt was still unmined when its replacement was signed. Now the
earlier attempts are settled before anything new is signed: if their nonce is
still unused, the next transaction takes it and carries them; if it has been
used, the attempt that used it is recorded as what it was (its contract, its
epoch), exactly as if its receipt had arrived on time; and if none of them used
it, the same patience applies as for a pending transaction, then the same letting go.
A mined revert is final, and is never remembered as an attempt to settle again.

## Under load: what broke, and where the ceilings are now

`LOAD=<miners> DAYS=<n> node stress.test.mjs` runs a quiet world with that many
miners (income scaled with them) and prints epochs, recipients, the largest
stored value and gas per payout. Measured:

    miners   shards  epochs  per payout  paid (distinct)  requests/tick  keys/write
       100       2       6    54-70           79              11           5
     2,000       2      16      400        1,471              12           7
   100,000     100     148      400       37,705              12          63

(three simulated days at 100,000, two at the others; per-transaction gas at
400 recipients is 20.5M of 4663's 32M cap; the largest single value stays
under 107 KB of the 128 KiB limit.)

**What 100,000 broke, before it passed:**

- **The RPC.** Balances were read for every miner every five minutes, two
  JSON-RPC calls each: 200,000 calls a tick at 100,000 miners. And a live
  probe found the public 4663 RPC refuses even *one* batch of 100 calls
  (HTTP 429; batches of 40 pass) -- so at any size, live, every read would
  have been rate-limited, every miner's work carried forever, and nobody
  paid. `GladosReader` (creation code run as an `eth_call`, never deployed)
  now reads balance and code for 200 addresses per request; the treasury
  reads only the likeliest recipients, most work first, until 400 are
  eligible or 1,000 are read, a second apart; the floor comes from work
  alone so ranking needs no reads; and an epoch is built at most every 30
  minutes, only when the ETH on hand is worth paying out. The connect gate
  batches arrivals within 300 ms into one read.
- **Memory and storage.** Shards now answer `/work.json` (sums per address)
  instead of ledger rows, which at 100 shards were tens of megabytes; a
  payout stores only the snapshot entries it moves, not the whole snapshot;
  and its working record is deleted once copied into the epoch (9,577 keys
  after three days, otherwise, and growing forever).
- **The public endpoints.** `/status` and `/ledger.json` fanned out to every
  shard on every request, and `/ledger.json` merged every row in one Worker.
  Both are edge-cached (30 s / 60 s), `?shard=` reads one shard, and past
  eight shards the ledger is an index of per-shard links.

**Before the earlier fixes, 2,000 halted forever** (1,337 eligible, over the
400 one transaction carries), and **the pool's ledger could not be saved past
about 250 miners** (one 128 KiB value, silently refused). Both are fixed:
rotation by work waiting, and chunked storage (`chunks.js`).

The ceilings that remain, nearest first:

- **About 1,000 miners per shard**: the core's 4,096 (worker, coin) tallies.
  `/status` reports the fullest shard's `tallies` against `maxTallies`, the
  object logs at 90%, and the remedy is raising `SHARDS` -- 100,000 miners is
  about `SHARDS = 100`.
- **About 200,000 miners for the treasury**, where the snapshot (63 of 128
  keys at 100,000) outgrows one atomic write.
- **Durable Object memory at 100,000 is estimated, not measured**: about 40-60
  MB for the maps an epoch build holds, against 128 MB. The simulator's own
  4.8 GB heap is the simulated world, not the treasury.
- **Share validation, 2,200-6,300 miners per shard, by arithmetic**: 7-19 ms a
  yespower share on one thread at one share per 45 s. The tally cap binds
  first. No real connections have been put against a deployed shard.

## What the live probes found

About twenty requests in all, spaced 1.5-10 s apart, nothing that could move
money: the hot wallets (both empty), a 100-call batch (refused, 429), batches
of 5 to 40 (accepted), Multicall3 (deployed), a revert and an "insufficient
funds" refusal (both definite answers, correctly not classed as transient),
an unknown receipt (`null`), blockbook's unknown-tx and garbage-sendtx
answers (HTTP 400 with `{"error": ...}`), a real transaction's
`confirmations` (a number), and one GladosReader call against real
addresses (the pair: a contract holding 252,417,850 GLADOS).

Not probed, deliberately: creating a ChangeNOW exchange. The key lives only
in the Worker, and the one call that proves the create path is the first
live exchange itself.

## Operating it: nothing halts, nothing is silent

The operator's rule is that an error nobody can fix is handled automatically,
and no miner should ever bring the operator an error message. So there is no
halt left anywhere in the treasury. Every failure is one of three things, and
the operator hears about the ones that matter from the treasury itself:

| failure | handled by | alerted |
|---|---|---|
| a dependency not answering (429, 5xx, timeout, garbage) | retried with backoff; the step waits and resumes | after 2 h continuous, and on recovery |
| ChangeNOW refusing the API key | RVN is held, nothing is sent, resumes when the key works | at once: the one thing only the operator can fix |
| three definite failures in a row (reverts, refusals) | paused 1 h, doubling to at most 24 h, then tries again | each pause |
| an exchange ChangeNOW failed, or sat on for 48 h | written off, still watched hourly; the next RVN goes to a new exchange. Late ETH or a late refund simply arrives | when written off, and if it later finishes or refunds |
| an unsent exchange's coins spent elsewhere | dropped before anything is sent | yes |
| a nonce used by somebody else | the transaction on it can never be mined, so it is let go | yes |
| the buy tax over 25% | payouts wait, and resume when it drops (under 25% it is priced into the bound) | yes |
| a pot over `TREASURY_MAX_PAY_WEI` | paid a cap at a time | no: that is the cap working |
| more than 400 eligible miners | the 400 with most work waiting are paid, the rest carry | no |
| a mined payout's record missing | rebuilt from the chain's calldata; nobody paid twice | yes |
| the connection gate unable to read a balance | the miner is admitted; the payout gate re-checks | no |

What makes "carry on" safe rather than hopeful is the same two properties
throughout: a transaction given up on shares its nonce with its replacement,
so they can never both be mined; and an exchange written off is never funded
again.

**Alerting.** Set `ALERT_WEBHOOK` (a Discord or Slack incoming webhook, as a
Cloudflare secret) and every incident is posted there, at least once: the
last delivered incident is stored and advances only when the webhook accepts,
so an alert lost to an eviction or a webhook outage is sent on the next tick.
The stress test found the first version losing one to exactly that eviction.

**Monitoring.** `/treasury` carries `status` (`ok`, `degraded` with the names
of the dependencies down, or `paused`), per-dependency last success and
failure, the last ten incidents and the written-off exchanges. Point an uptime
check at it and alert on anything but `ok` for more than a few hours.

**Levers,** all a config change and a deploy, never a request:
`TREASURY=off|dry|live` is the kill switch; `TREASURY_MAX_PAY_WEI` bounds any
single payout (start low, raise it as epochs land cleanly); changing
`TREASURY_RESUME` ends a pause early; `TREASURY_RVN`/`TREASURY_EVM` pin the
addresses once published.

## What it cannot see

- **A sandwich inside the 3% slippage bound.** Bounded, not prevented; the
  median check narrows the moment and does not close it.
- **A key taken while no transaction is in flight.** The nonce check only
  catches a thief who races a pending transaction. The wallets hold one
  epoch's proceeds at a time, which is the limit on that loss.
- **ChangeNOW keeping the money.** A failed exchange is written off, watched
  and alerted; getting that RVN back is a support ticket, and nothing else
  waits for it.
- **A second exchange or RPC to fail over to.** Every dependency has exactly
  one verified endpoint. An outage delays payouts, it does not lose them, and
  the alert says which one; a second provider is the change that would turn
  the delay into nothing.
