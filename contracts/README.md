# The claim contract

`GladosDistributor.sol` is Layer 2 of the pool: Layer 1 never holds a miner's
coins, and this distributes the operator's *own* fee revenue, converted to
GLADOS, against a Merkle root published from the share log.

```bash
npm install          # solc, an EVM, and an ABI coder. No framework.
npm run build        # compile
npm test             # 60 claims, mostly about what it refuses
```

The end-to-end check, which is the one that matters:

```bash
python ../tools/distribute.py ledger.json --total 1e24 --coin btc \
    --map workers.json --out epoch.json
node test/cross.mjs epoch.json
```

Three implementations of one tree, required to agree: `tools/distribute.py`
builds it in Python from the pool's own ledger, `test/merkle.mjs` builds it
again in JavaScript, and the compiled contract verifies it. The first two
agreeing is worth something; the first two agreeing with the **third** is the
only thing that matters, because the third is what holds the tokens.

`cross.mjs` then funds a real epoch with that root in a real EVM and has every
address in the ledger claim, so a builder that got amounts or ordering wrong
shows up as a claim that reverts rather than as a root that looks fine.

## Two ways an epoch can pay, and the arithmetic picks

`Direct` holds the reward token and a claim transfers it: one market buy, made
by the operator when they convert, and every claimant gets the same rate.

`Market` holds the quote token and **each claim is the claimant's own buy on
the pool**. Every claim moves the price, pays the token's buy tax into whatever
the token does with it, and appears on-chain as a trade rather than as an
operator handing out tokens converted somewhere nobody watched. The operator
never converts anything, so there is no conversion rate to have to trust.

It is not a better mode, it is a different trade, and the numbers decide:

| | pot | per claim | gas as a share of it |
|---|---:|---:|---:|
| 36h event, 256 miners | $0.54 | $0.0021 | 2,286% |
| 36h event, 800 miners | $1.68 | $0.0021 | 2,286% |
| a year, 1000 miners, quarterly | $127.75 | $0.1278 | 38% |
| a year, 1000 miners, yearly | $511.00 | $0.5110 | 9% |

A swap is about $0.048 of gas on this chain and a transfer $0.029, so a claim
has to be worth more than that before either mode pays for itself. `Market` is
right for a large annual epoch and absurd for a weekend.

The cost `Market` carries beyond gas is stated in the contract and asserted in
the tests: **the leaf is denominated in the quote token, so the first claimant
gets a better price than the last.** That is a race, it is inherent to paying
through a market rather than around one, and it is why `Direct` still exists.

## Testing it for real, and the one leg that cannot be rushed

Three levels, each strictly more real than the last:

```bash
python ../tools/loop.py             # mine, credit, build, claim -- all local
python ../tools/loop.py --fork      # same, but claiming against a fork of the
                                    # real chain: real token, real pool, real tax
node deploy.mjs status              # what the real chain says right now
```

And on the chain itself, with your own key and your own money:

```bash
node deploy.mjs deploy                            --send
node deploy.mjs wrap  --amount 0.01               --send
node deploy.mjs open  epoch.json --amount 0.01    --send
node deploy.mjs claim epoch.json --epoch 0        --send   # from the miner's wallet
```

**Every command simulates first and refuses to broadcast without `--send`.**
That is the whole safety model, and it is there because the difference between
a correct epoch and one funded with the wrong number is a transaction that
succeeds either way -- the only moment to catch it is before signing.

The key comes from `GLADOS_KEY` in the environment, is used to sign, and is
never printed, written or put in an error message.

### What is real at each level

| | mining | ledger | tree | contract | pool | money |
|---|---|---|---|---|---|---|
| `loop.py` | real | real | real | real | mock | none |
| `loop.py --fork` | real | real | real | real | **real** | none |
| `deploy.mjs` | real | real | real | **real, deployed** | **real** | **yours** |

**The one leg that cannot be tested quickly** is actually receiving mining
proceeds from an upstream. At the measured $0.07 a day, a common 0.001 BTC
minimum payout is 3.7 years away and a generous 0.0001 BTC one is 136 days.
That is not a test, it is a waiting game, and no amount of code shortens it.

So a real end-to-end run funds the epoch from the operator's own wallet rather
than from accumulated mining revenue. That is not a cheat: it is what the
design says happens anyway -- Layer 2 distributes the *operator's* money -- and
at this scale the difference between "funded by mining" and "funded by you" is
ten cents against ten dollars.

## No hardhat, no foundry

What is needed is a compiler and an EVM, and each is one package. A framework
would bring a config file, a plugin system and a directory layout, none of which
this repository would then be able to explain the way it explains everything
else. `test/evm.mjs` is ninety lines and is the whole harness.

## What is not done

- **Never executed against a real pool.** Every swap in the suite is against a
  mock. The mock pays out before it asks and then checks it was paid, which is
  the shape that makes a broken callback fail here rather than on the chain, but
  a passing mock is not a filled trade. One small real swap comes before
  anything else uses this.
- **Not deployed.** No address, no verified source on any explorer.
- ~~**Not reviewed by anybody.**~~ Read once, by somebody who did not write it:
  `design/audit.md`. Six findings, none a theft vector, and the first has to be
  acted on *before* the first deployment because `pair` is immutable and is
  never checked against the pair it is supposed to be. The suite carries
  fourteen more claims for them and the source is deliberately unchanged, so
  each one inverts when its fix lands. That is a different sentence from
  "reviewed", and a contract holding tokens still deserves a second reader.
- **The gate size is not decided.** `design/live800.md` shows 1,000,000 is
  unreachable for 800 entrants -- the pool holds 130 gates -- and that 50,000
  is the number that lets an event happen. The contract takes it per epoch and
  does not care.
- ~~**No worker-to-address mapping service.**~~ Built:
  `supabase/functions/worker`, migration `0003_workers.sql`. A miner signs a
  message naming the address and the worker name, and `distribute.py --map`
  reads what it serves. The flat file still works, because an event with no
  server is worth keeping cheap. What is new beyond the plan is `--map-since`:
  shares accrue against a name for days while the mapping is read once at the
  end, so a name that changed hands in between would collect somebody else's
  work.
