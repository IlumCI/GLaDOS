# Running an event, from first miner to paid claim

Every other document here explains a decision. This one is the sequence, in
order, with the commands. It exists because the path from "the event ended" to
"an epoch is open on chain" runs through five tools, and until it was written
down it lived in one person's head.

**Nothing here has been run end to end with real money.** Each step has been
exercised on its own; the joins have not. That is the honest status and it is
what step 8 is for.

---

## 0. Before anything: decide two numbers

**The gate.** `design/live800.md` shows 1,000,000 is unreachable for 800
entrants -- the pool holds about 130 whole gates -- and that 50,000 is the
number at which an event can happen. The contract takes it per epoch and does
not care what you choose, so this is a decision nobody will make for you.

**The payout basis.** `window` or `tally`, and they are not the same number.
See step 5; choosing wrong cannot be undone.

---

## 1. Build and install the pool

```bash
cargo build --release --manifest-path pool/Cargo.toml --target x86_64-unknown-linux-musl
```

musl and `rust-lld` so a fully static binary comes out of a machine with no C
toolchain. `pool/.cargo/config.toml` carries the linker line and says why.

```bash
mkdir -p ~/.local/bin ~/.local/state/glados-pool
install -m 0755 target/x86_64-unknown-linux-musl/release/glados-pool ~/.local/bin/
```

Check it before it faces anybody:

```bash
~/.local/bin/glados-pool --selftest
python tools/poolroster.py ~/.local/bin/glados-pool
```

The first is 18 roster claims plus the socket exchange; the second drives the
refusal path over a real socket, which `--selftest` structurally cannot.

## 2. Deploy the worker mapping

```bash
supabase db push                      # 0003_workers.sql
supabase functions deploy worker
```

Then set `WORKER_NAME_CAP`, and confirm `TOKEN_CHAIN_ID` and `LINK_DOMAIN` are
already set from the `link` deployment -- they are shared, and the signed
message names the domain, so the two functions disagreeing about it makes every
signature fail to verify with no obvious cause.

Check it answers:

```bash
curl -s https://<project>.supabase.co/functions/v1/worker/map
```

## 3. Refresh the roster on the pool host

The pool reads a file and does not fetch, for the reason `pool/src/roster.rs`
gives at length: it has no dependencies and no TLS, and a hidden `curl` inside
the process fails differently on every host with nothing saying why.

```bash
# once, then in cron
curl -fsS https://<project>.supabase.co/functions/v1/worker/map \
  -o ~/.local/state/glados-pool/roster.json.tmp \
  && mv ~/.local/state/glados-pool/roster.json.tmp \
        ~/.local/state/glados-pool/roster.json
```

**The `.tmp` and the `mv` are the point.** `mv` within one filesystem is
atomic, so the pool never reads a half-written file. Writing directly would
give it a truncated document to parse, and although a parse failure leaves the
previous roster in place, that is a safety net rather than a plan.

```cron
*/5 * * * * curl -fsS <url> -o <state>/roster.json.tmp && mv <state>/roster.json.tmp <state>/roster.json
```

## 4. Run the pool

```bash
~/.local/bin/glados-pool \
  --listen 0.0.0.0:3334 \
  --ledger ~/.local/state/glados-pool/ledger.json \
  --roster ~/.local/state/glados-pool/roster.json \
  --roster-url https://glados.aperture.institute/pool/ \
  --require-roster
```

`pool/deploy/run-pool.sh` wraps this with `ulimit -n 65536` and log rotation.

**On `--require-roster`.** With it, a miner whose name is not registered is
refused at the greeting with a message pointing at `--roster-url`. Without it
they are admitted and a warning is logged, and they find out after the event
that their work cannot be paid. Leave it off only for an event where everybody
mines under their own `0x` address as the worker name, which needs no mapping
service at all and is a perfectly good way to run one.

**The pool fails open.** If the roster file is missing or unparseable, every
name is admitted regardless of `--require-roster`, and the startup log says so
in those words. Read that line.

## 5. Close the epoch and build the tree

```bash
python tools/distribute.py ~/.local/state/glados-pool/ledger.json \
  --basis tally \
  --total 1e21 \
  --coin btc \
  --map <(curl -fsS https://<project>.supabase.co/functions/v1/worker/map) \
  --map-since 2026-09-08T00:00:00Z \
  --gate 50000e18 --token 0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777 \
  --out epoch.json
```

**`--basis` has no default and the two answers differ by 4x** on a real ledger.
`tally` is every share the pool credited and is what a bounded event wants;
`window` is the PPLNS sliding window and is what continuous operation wants,
because a miner who arrives for the profitable part of a round finds their
shares aged out. A 36-hour event settled weeks later would have almost nothing
left in the window, so `tally` is the answer for an event. The tool refuses to
guess.

**`--map-since` should be when the epoch began accruing.** Shares accrue against
a *name* for days and the mapping is read once, at the end, so a name that
changed hands in between would collect work its previous holder did. Entries
that moved after this timestamp are refused and land in the "no payout address"
list, which is where they belong.

**If any name comes back unmapped, the command exits 1 and pays nobody.** That
is deliberate. Chase the names, or accept that they cannot be paid and rerun --
but decide it, do not discover it.

## 6. Check the tree with something that did not build it

```bash
node contracts/test/cross.mjs epoch.json
node contracts/test/fork.mjs  epoch.json          # against a fork of the live chain
```

`cross.mjs` verifies every proof against the contract's own `verifyProof`, run
in an EVM, rather than against the Python that produced them. Two
implementations that are supposed to agree do not stay agreeing on their own.

## 6a. Get the money onto 4663

The epoch is funded in USDG, so mining proceeds have to arrive there first.
`contracts/bridge.mjs` is the one leg of that which is a transaction somebody
has to build:

```
mine -> the venue credits a balance        the venue does this
     -> it pays out to your address        the venue does this, at a threshold
     -> [swap to a bridgeable token]       avoidable, see below
     -> bridge to 4663                     bridge.mjs
     -> fund an epoch                      deploy.mjs open
```

**Pick the venue that pays the token the bridge already takes.** unMineable
pays POL on Polygon, which has to be sold for USDC before it can cross;
Kryptex pays USDC on Polygon directly, and Across carries USDC from Polygon to
USDG on 4663 in one hop. That choice removes a DEX integration, a slippage
bound and a second approval from a path that moves real money, which is worth
more than automating the swap would be.

```bash
node contracts/bridge.mjs routes                          # what Across reaches 4663 from
node contracts/bridge.mjs status --address 0x...          # where the money is now
node contracts/bridge.mjs quote  --amount 100 --chain 137
node contracts/bridge.mjs send   --amount 100 --chain 137 --send
```

**Trip size is the only lever.** The fee is almost entirely flat relayer gas,
measured live on the Polygon route:

    $10     0.5270%        $250    0.0785%
    $25     0.2458%        $500    0.0692%
    $50     0.1529%      $1,000    0.0646%
    $100    0.1062%

So one $500 trip costs $0.35 where five $100 trips cost $0.53, and anything
under about $5 is refused outright because the flat part exceeds it. Fill time
is around a second.

**The recipient is always the sender and there is no flag for it.** This moves
the operator's own float; a recipient argument is one typo away from bridging
an event's funding to a stranger, irreversibly, with the transaction
succeeding.

## 7. Open the epoch

### Three things to check before the first `--send`, from `design/audit.md`

The commands below are unchanged. What is new is that an audit found two of them
carry a precondition nobody knew about, and one of the three cannot be satisfied
after the fact.

1. **Read back the `pair` argument before deploying.** It is an immutable
   constructor argument and the contract never checks it against the pair it is
   meant to be, while `openEpochOnV3` checks its pool twice. A typo produces a
   distributor that accepts funding for market epochs and refuses every claim
   forever -- driven, `TooLittleOut(0,1)` -- with no setter and no recovery but
   `reclaim` after the deadline. **This is the one thing on this page that a
   later step cannot fix.**
2. **Check the root is not already open.** The leaf carries no epoch id, so the
   same root opened twice is a second entitlement to the same work rather than a
   duplicate anything refuses. `epochCount()` and each epoch's `root` answer it.
3. **Do not open a `MarketV3` epoch with this tool yet.** `open` creates one;
   `claim` calls `claimOnMarket` unconditionally, so the tool cannot claim from
   what it just opened, and `checkClaim` does not read `mode` either -- it will
   say the claim is good first. The V3 invocation further down this page is
   therefore the one to leave alone until `deploy.mjs` grows a `claimOnV3` path.

```bash
export GLADOS_KEY=0x...            # never printed, never stored; see deploy.mjs
node contracts/deploy.mjs status
node contracts/deploy.mjs deploy --send                    # first time only
```

Then, paying in $GLADOS through the V2 pair:

```bash
node contracts/deploy.mjs open epoch.json --amount 0.005 --gate 50000
node contracts/deploy.mjs open epoch.json --amount 0.005 --gate 50000 --send
```

Or paying in a tokenized equity through a Uniswap V3 pool:

```bash
node contracts/deploy.mjs open epoch.json --amount 0.005 --gate 50000 \
  --v3-pool 0xd4eb21209c4d6093f80b5b84f5c45cc093ea14a3 \
  --reward  0xd0601ce157db5bdc3162bbac2a2c8af5320d9eec        # NVDA
```

**Run it once without `--send` and read the output.** Simulation is the default
and it is the whole safety model: the difference between a correct epoch and one
funded with the wrong number is a transaction that succeeds either way, and the
only moment anybody can catch it is before it is signed. The tree's sum is
checked against `--amount` here, and a V3 pool is read back and its pair
printed, because the contract's own `BadPool` check happens inside a
transaction you have already paid for.

**The two modes are not interchangeable.** `claimOnMarket` refuses a V3 epoch
and `claimOnV3` refuses a Market one, so opening in the wrong mode is an epoch
nobody can claim from until it expires and you `reclaim` it.

## 8. The step nobody has taken

Before an event with strangers in it, run the whole of the above with **one
address, your own, and a few dollars**. Mine to the pool for an hour, build a
one-leaf tree, open the epoch, and claim it.

Every piece has been exercised alone. The joins have not, and the two most
likely to bite are the ones this document cannot check for you: whether the
ledger the pool wrote is the ledger `distribute.py` reads on a host that is not
this one, and whether a V3 swap actually fills at the price the quoter
promised. A quote is not a fill.

## 9. Publish

The share log, every epoch root, and the transaction hash of every conversion.
`design/pool.md` argues why at length; the short version is that a
non-custodial pool has no wallet to audit, so the arithmetic being reproducible
from published documents is the whole of what it offers instead of trust.

---

## What can still go wrong that nothing above prevents

- ~~**The contract has not been read by anybody who did not write it.**~~ It has
  been, once: `design/audit.md`. **One finding there has to be acted on before
  step 7's `deploy --send` and cannot be acted on afterwards** -- `pair` is an
  immutable constructor argument and is never checked against the pair it is
  supposed to be, so a typo there is a distributor that accepts funding for
  market epochs nobody can ever claim. The other five are smaller and two of
  them are `deploy.mjs`'s rather than the contract's, including: **this tool
  cannot claim a `MarketV3` epoch it is perfectly able to open.** So the V3
  invocation below opens something whose only exit today is `reclaim` after the
  deadline. One reader is not a review; the contract still holds tokens.
- **`reclaim` is the operator's one power over committed funds**, bounded to
  the unclaimed remainder after a deadline fixed when the epoch opened. A
  miner should be told the deadline, because it is the date their allocation
  stops existing.
- **The bridge has never been run with real money.** `bridge.mjs` quotes
  against the live API and its refusals are exercised, but no deposit has been
  broadcast. The legs either side of it -- the venue paying out, and
  `deploy.mjs open` -- are still entirely operator-run, and the venue half
  cannot be automated at all: it is a threshold somebody else's server decides
  to cross.
