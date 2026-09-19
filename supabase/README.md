# The update service

What the in-OS updater talks to. Two channels, one signing key, and a gate that
is honest about what it is.

| | `stable` | `experimental` |
| --- | --- | --- |
| Bucket | public | private |
| Reached by | one GET at a static object | `channel` function, bearer device code |
| Auth | none, ever | entitlement |
| Built by | `.github/workflows/release.yml` on a `v*` tag | `experimental.yml` on `exp/**` or dispatch |

**Stable has no server-side compute.** It is an object in a public bucket, so
there is nothing to rate-limit, nothing to cold-start, and no way for the gated
path's failure to take the free path down. Security fixes live there and always
will.

## What the gate can and cannot do

The gate is the server declining to answer. An experimental image that leaks
installs on any machine, because it is signed by the same key as stable and the
kernel's only question is whether the signature is good.

There is no local check and there could not be a meaningful one: the kernel's
source is published, so any balance test in it is a test the machine's owner can
delete and rebuild without. Anything that claims otherwise on the token page
would be false.

## Setting it up

One project. Two buckets, `stable` (public) and `experimental` (private).

```sql
-- migrations/0001_updates.sql
```

Four tables, RLS enabled with **no policy on any of them**, which denies every
access through the anon and authenticated keys. The only door is an Edge
Function holding the service role.

### Secrets

| Where | Name | What |
| --- | --- | --- |
| GitHub → Secrets | `UPDATE_SIGNING_KEY` | the private half, hex |
| GitHub → Secrets | `SUPABASE_SERVICE_KEY` | service role key |
| GitHub → Variables | `SUPABASE_URL` | `https://<ref>.supabase.co` |
| Supabase → Function secrets | `UPDATE_SIGNING_KEY` | the same private half |

The signing key lives in two places, and that is a real cost rather than an
oversight. The `channel` function rewrites image URLs into short-lived signed
ones, which changes the manifest's bytes, so it has to re-sign what it
rewrites. The alternative — signing URLs at publish time — means a build stops
being installable an hour after CI ran.

### Generating the key

```bash
python tools/sign.py --keygen --out update.key
```

`--out` writes the private half to a 0600 file and **does not print it**. Use
it. A previous key's private half was printed to a terminal while the signer
was being written, which is why `UPDATE_KEY` in the kernel has been zeroed ever
since.

Paste the public rows into `UPDATE_KEY` in `src/update/mod.rs` and rebuild.
Adopting a signer is itself a kernel change, which is the point — and it means
**the first build carrying the key cannot be delivered by this system**, since
no kernel in the field trusts it yet. That one ships as an ISO.

### Granting access before there is a chain

```sql
insert into allowlist (code_hash, note)
values (encode(digest('the-code-you-issued', 'sha256'), 'hex'), 'who this is for');
```

Codes are stored hashed. The server only ever needs to *recognise* one, never
read one back, and a database leak that included them would be a leak of
working credentials.

`entitled()` in `functions/channel/index.ts` is one function with one job. Its
first implementation is that lookup. When custody is settled, a balance read
goes in it and nothing else changes.

## Checking the signer

The Edge Function's P-256 signer is plain JavaScript in `functions/_shared/`
precisely so it can be run outside Deno and checked:

```bash
node supabase/functions/_shared/crosscheck.mjs /tmp
python tools/manifest.py --verify /tmp/xcheck.manifest --key "$(cat /tmp/xcheck.pub)"
```

That checks the function's signer against the Python verifier, which mirrors the
kernel's, which is itself checked at every boot against published ECDSA vectors.
Two implementations that are supposed to agree do not stay agreeing on their
own.

The key in `crosscheck.mjs` is fixed and worthless — the whole point is that it
sits in a public repo.

## Publishing order

Both workflows put the image and its signature **before** the manifest, and
`experimental.yml` writes its `builds` row last. A manifest naming an object
that is not there yet is a window in which every machine that checks gets a 404
for an update it was just told about.

## The link function

Wallet linking, added once the token launched as an ERC-20 in self-custody.
Two POST steps, distinguished by the last path segment:

```
POST /functions/v1/link/nonce   {address}             -> {nonce, message, expires_at}
POST /functions/v1/link/verify  {address, signature}  -> {code, address, balance}
```

Run migration `0002_link.sql` first; it adds `nonces` and three columns to
`wallets`.

### Secrets it needs

| Where | Name | What |
|---|---|---|
| Supabase → Function secrets | `TOKEN_CONTRACT` | `0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777` |
| | `TOKEN_RPC` | `https://rpc.mainnet.chain.robinhood.com` |
| | `TOKEN_CHAIN_ID` | `4663` |
| | `TOKEN_MIN_BALANCE` | `1000000000000000000000000` (1e6 tokens at 18 decimals) |
| | `LINK_DOMAIN` | `glados.aperture.institute` |

All five have defaults in the source, so the function runs without them. They
exist so the threshold can move without a redeploy of anything the kernel
trusts, and `channel` reads the same three, from the same place, so the door
that issues a code and the door that honours one cannot disagree about what
counts as holding.

## The worker function

Which address a mining worker name's shares are owed to. Run migration
`0003_workers.sql` first.

```
POST /functions/v1/worker/nonce    {address, worker, verb?}    -> {nonce, message, ...}
POST /functions/v1/worker/claim    {address, signature, worker} -> {worker, address}
POST /functions/v1/worker/release  {address, signature, worker} -> {released}
GET  /functions/v1/worker/map                                   -> {workers, updated_at}
```

`tools/distribute.py --map` reads what `/worker/map` serves, and also still
reads a flat `{name: address}` file, because an event with no server at all is
a case worth keeping cheap.

**It is not gated on a balance and `link` is.** The difference is what each
hands out. `link` gives access to builds, so it checks the holding. This
records who a name's work belongs to, and gating it would mean buying before
mining -- which inverts the order the design depends on. The gate is the
contract's, checked at claim time against a balance held then; a miner who
never buys never claims, and the allocation returns through `reclaim`.

**A worker name shaped like an address is refused.** `distribute.py` already
takes an address-shaped name as the address itself, before consulting the
mapping, so a row saying `0xVICTIM -> attacker` would be dead weight. It is
refused anyway, because the only thing keeping it dead is the order of two
branches in one Python function.

**Names are unique case-folded and stored as spelled.** Two people holding
`Alice` and `alice` is not a collision a pool would notice and is the shape a
lookalike attack takes, so the database enforces one. What is served back is
the exact spelling, because that is the string the ledger will carry.

### The one hazard worth knowing before running an event

Shares accrue against a *name* over days and the mapping is read once, at the
end. A name that changes hands in between would collect work its previous
holder did. `updated_at` travels in the map for exactly this, and
`distribute.py --map-since <epoch start>` refuses entries that moved after the
epoch began. They land in the same "cannot be paid" list as a name nobody ever
registered, which is the honest place for them.

### Secrets it needs

| Where | Name | What |
|---|---|---|
| Supabase → Function secrets | `TOKEN_CHAIN_ID` | `4663`, shared with `link` |
| | `LINK_DOMAIN` | `glados.aperture.institute`, shared with `link` |
| | `WORKER_NAME_CAP` | `10`, how many names one address may hold |

Both shared names are read from the same place `link` reads them, so the
signed message cannot claim one domain while the other function checks
another.

### Both functions deploy with `--no-verify-jwt`

```
supabase functions deploy link --no-verify-jwt
supabase functions deploy channel --no-verify-jwt
```

Neither uses Supabase auth. `channel` authenticates with a device code it
hashes itself, and `link` authenticates with a wallet signature. Leaving JWT
verification on would put a second, unrelated credential in front of both, and
the kernel has no way to present one.

### What the signature does and does not prove

It proves the signer holds the key for the recovered address. Nothing else --
no chain, no balance, no token. The balance is a separate `eth_call`, which is
why a wallet that has never added Robinhood Chain can still link.

That balance is read server-side and re-read on every download rather than
cached. A holding is not a permanent fact about a person, and a remembered one
is a gate that stays open after the thing it gated on is gone.

### The check on the address recovery

There is no published test vector for it in this repository, and one this code
generated for itself would prove nothing. The real check happens on first use:
the wallet page displays the address the *wallet* reports beside the address
the *server* recovered. MetaMask and Phantom are the independent
implementations, and a wrong recovery shows up as two different addresses
rather than as silence.

## The proposal function

The door a machine's own proposals come in through, and the one that spends
somebody else's CI minutes.

`godel source` picks a point out of the declared table of tunable constants and
writes a patch; `godel push` sends it here as an *envelope* -- who is asking,
from which lineage, against which corpus, how many tests that corpus has
already paid for, and then the patch. The kernel cannot compile and cannot
judge a source change, so this starts `propose.yml`: two builds on one runner,
two boots, and the rail the proposal claimed.

```
supabase functions deploy proposal --no-verify-jwt
```

### The token is here and never there

Dispatching a workflow needs a GitHub credential, and it lives in this
function's environment. **The kernel never holds one**, which is the whole
reason the outward path goes through a server rather than the machine talking
to GitHub directly: a machine in the field carries a device code that can ask
for a build to be judged, and nothing that can write to a repository.

That is the same division `channel.rs` makes on the kernel side. What goes out
is the machine's own text, over an authenticated connection to the origin it
already talks to, and the origin decides what that means.

### What arrives is not what is forwarded

`_shared/envelope.js` reads the body into declared fields, checks every one
against a shape, and **renders a fresh envelope from what it parsed**. The
bytes handed to the workflow are bytes this side wrote.

A validator that checked its input and then passed the original through would
be one carriage return away from meaning something else, and the thing on the
other end compiles code. The refusals are the point rather than the parse:

```
node supabase/functions/_shared/envelope.test.mjs
```

Forty-odd claims, and what earns their place is the list of things that must be
refused -- a value that is a command, a value that is an expression, a path
that climbs out of the tree, a symbol that is not a constant's name, a body
carrying a control character. An Edge Function has no test runner of its own,
which is why the validator is plain JS that `node` can drive, the same
arrangement `gladosig.js` has with `crosscheck.mjs`.

**And it has been checked against the kernel rather than against itself.** The
recipe is two commands, and it is the same bargain `crosscheck.mjs` makes for
the signer:

```powershell
.\tools\venv\Scripts\python.exe tools\drive.py --qemu-extra "-accel whpx -cpu max" `
  "initiative off" "agent stop" "godel source" "godel push --dry" > out\push.log
node supabase\functions\_shared\envelope.check.mjs out\push.log
```

Driven: the envelope the kernel rendered is 259 bytes over twelve lines, it
parses, and `render` reproduces it **exactly**. Two implementations that are
supposed to agree do not stay agreeing, and this is the pair where a single
byte of disagreement is a proposal that silently means something else.

It reads the transcript rather than a bare file, because a transcript is what
somebody actually has -- and stripping the shell's `  | ` prefix belongs in the
checker rather than in the recipe, since a recipe with a `sed` in the middle is
one somebody gets wrong once and then stops running.

**That already earned itself.** A serial transcript carries `\r\r\n` -- the
guest's line ending and the harness's -- so a strip that took exactly one
carriage return left exactly one, and `parse` accepted it anyway because
`trim()` eats a trailing CR. Nothing but comparing the rendered bytes noticed,
which is the argument for doing that rather than asking whether it parsed.

It also refuses a patch under `src/gfx/`, `src/doom/` or `src/port/`, from the
same list the kernel and `tools/knob.py` carry. Screenshots are captured and
never compared, so such a patch would build, boot, read `same` on every rail
there is, and be adopted having checked nothing about the only thing it
changed. The kernel's copy stops the machine proposing one; this stops one
arriving from anywhere else, which is what the auth check exists for.

### The gate is the allowlist, not the balance

`channel` opens on a token balance because what it hands out is a build, which
is a product. This hands out CI minutes on somebody's behalf, and a balance is
not a promise to behave -- so the allowlist only, plus a per-device daily
count kept in `proposals` rather than inferred from GitHub. A run is two
`cargo build --release` and two QEMU boots; an entitled device that could
dispatch in a loop could spend an account's whole budget in an afternoon.

### Secrets it needs

| | |
|---|---|
| `GITHUB_DISPATCH_TOKEN` | fine-grained, `actions: write`, one repository, nothing else |
| `GITHUB_REPO` | `owner/name`; defaults to this one |
| `GITHUB_REF` | the branch to run on, default `main` |
| `GITHUB_WORKFLOW` | default `propose.yml` |
| `PROPOSAL_FOREST` | a path on the runner; blank means the workflow generates a fixture, **which it then refuses to adopt on** |
| `PROPOSAL_PER_DAY` | runs one device may ask for, default 6 |

With no token the door answers 503 and records the proposal as
`unconfigured`, which is a configuration rather than an error and says so.

### What it cannot do, said plainly

It cannot cause an adoption. All it starts is a measurement; the verdict comes
back **signed by the update key**, and `godel verdict` on the machine refuses
an unsigned one -- a verdict is the only thing that writes a ledger line the
machine did not derive itself, so an unsigned one is somebody else editing the
lineage. Signing the verdict is the step `propose.yml` deliberately leaves out,
and it is the one piece of this loop that is still not wired.
