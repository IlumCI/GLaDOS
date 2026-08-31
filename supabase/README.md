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
