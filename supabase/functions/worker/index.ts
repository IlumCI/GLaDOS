// Which address a worker name's shares are owed to.
//
// Four steps, distinguished by the last path segment:
//
//   POST /worker/nonce    {address}                      -> {nonce, message, ...}
//   POST /worker/claim    {address, signature, worker}    -> {worker, address}
//   POST /worker/release  {address, signature, worker}    -> {released}
//   GET  /worker/map                                      -> {workers: {...}}
//
// ### The one rule that is a security rule
//
// **A worker name shaped like an address may not be claimed.** `distribute.py`
// takes an address-shaped name as the address itself, before it ever consults
// this mapping, so a row saying `0xVICTIM -> attacker` is dead weight today.
// It is refused anyway, because the only thing keeping it dead is the order of
// two branches in one Python function, and a future consumer that checked the
// map first would turn a harmless row into a theft. Refusing at the point of
// entry does not depend on anybody remembering.
//
// ### Why the signer is the payout address
//
// A claim is signed by the address the work will be paid to, and there is no
// field for a third address. Allowing one means the message has to say which
// of two addresses is which, and a person approving a signature in a wallet
// popup reads the first address they see. Moving a name is therefore
// `release` by the current owner and then `claim` by the new one -- two
// signatures, each about one address.
//
// That leaves a window where a released name can be taken by somebody else.
// It is real and it is small, and the thing it could steal is answered on the
// other side: `updated_at` travels in the map, and a distribution refuses
// entries that moved after the epoch it is paying began.
//
// ### No balance gate here
//
// Deliberate, and the opposite of `link`. See `0003_workers.sql`.

import { createClient } from "jsr:@supabase/supabase-js@2";
import { recoverAddress, toChecksum } from "../_shared/evm.js";

const url = Deno.env.get("SUPABASE_URL")!;
const service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(url, service);

const CHAIN_ID = Number(Deno.env.get("TOKEN_CHAIN_ID") ?? 4663);
const DOMAIN = Deno.env.get("LINK_DOMAIN") ?? "glados.aperture.institute";
const NONCE_TTL_MS = 10 * 60 * 1000;

// How many names one address may hold. A rig fleet is usually one name with
// `.rig1`, `.rig2` suffixes the distributor already folds together, so ten is
// generous. The number matters more than its value: without one, a single
// address can squat every short name there is.
const NAME_CAP = Number(Deno.env.get("WORKER_NAME_CAP") ?? 10);

const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "content-type",
  "access-control-allow-methods": "GET, POST, OPTIONS",
};

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...cors },
  });
}
const refuse = (s: number, why: string) => json(s, { error: why });

const isAddress = (s: unknown): s is string =>
  typeof s === "string" && /^0x[0-9a-fA-F]{40}$/.test(s);

// What a stratum worker name may be here.
//
// Narrower than what a pool accepts, on purpose. The pool has to take whatever
// arrives on the wire; this decides what may be *registered*, and a name
// carrying whitespace, a dot or a non-ASCII lookalike is a name somebody else
// will one day be paid for. The dot is excluded because `distribute.py` splits
// on it to fold `name.rig1` into `name`, so a registered `a.b` could never be
// looked up.
const NAME_RE = /^[A-Za-z0-9_-]{3,32}$/;

function nameProblem(w: unknown): string | null {
  if (typeof w !== "string") return "worker must be a string";
  if (!NAME_RE.test(w)) return "worker must be 3-32 chars of A-Z a-z 0-9 _ -";
  if (/^0x[0-9a-fA-F]{40}$/.test(w))
    return "a worker name that is an address is already its own payout address";
  // Belt and braces: `NAME_RE` forbids `x` after `0`? It does not, and
  // `0x...` of the wrong length would pass the test above. Anything starting
  // `0x` is refused outright rather than reasoned about.
  if (/^0[xX]/.test(w)) return "a worker name may not begin with 0x";
  return null;
}

/// The EIP-4361 message, with the worker name inside it.
///
/// The name is in the signed text rather than only in the request body, which
/// is the whole point of signing: a body a proxy could edit is not what the
/// person approved. Without this line a captured signature for one name is a
/// signature for every name.
function siwe(address: string, worker: string, verb: string, nonce: string, issued: Date, expires: Date) {
  return [
    `${DOMAIN} wants you to sign in with your Ethereum account:`,
    toChecksum(address),
    "",
    `${verb} the mining worker name "${worker}" for this address.`,
    "This signature costs nothing, moves nothing, and approves no transaction.",
    "",
    `URI: https://${DOMAIN}/pool/`,
    "Version: 1",
    `Chain ID: ${CHAIN_ID}`,
    `Nonce: ${nonce}`,
    `Issued At: ${issued.toISOString()}`,
    `Expiration Time: ${expires.toISOString()}`,
  ].join("\n");
}

/// Find and spend the caller's nonce, answering the message it signed.
///
/// Spending happens here, in an update conditioned on the row still being
/// unspent, so two requests racing one nonce cannot both win. Copied in shape
/// from `link` rather than shared, because sharing it would mean a nonce
/// issued for one function could be spent by the other.
async function spendNonce(address: string): Promise<
  { ok: true; nonce: string; issued: Date; expires: Date } | { ok: false; status: number; why: string }
> {
  const { data: row } = await db.from("nonces")
    .select("nonce, address, issued_at, expires_at, used_at")
    .eq("address", address).is("used_at", null)
    .order("issued_at", { ascending: false }).limit(1).maybeSingle();
  if (!row) return { ok: false, status: 400, why: "no unused nonce for that address -- ask for one first" };
  if (new Date(row.expires_at) < new Date())
    return { ok: false, status: 400, why: "that nonce has expired" };
  return { ok: true, nonce: row.nonce, issued: new Date(row.issued_at), expires: new Date(row.expires_at) };
}

async function burn(nonce: string): Promise<boolean> {
  const { data } = await db.from("nonces")
    .update({ used_at: new Date().toISOString() })
    .eq("nonce", nonce).is("used_at", null).select("nonce").maybeSingle();
  return !!data;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: cors });

  const step = new URL(req.url).pathname.split("/").filter(Boolean).pop();

  // ---- the map, which is the only thing here anybody reads ---------------
  //
  // Public and unauthenticated, because it is public information: it is
  // published beside every epoch root so a miner can check that the tree paid
  // the address they registered. A mapping only the operator can see is a
  // mapping nobody can audit.
  if (step === "map") {
    if (req.method !== "GET") return refuse(405, "GET only");
    const { data, error } = await db.from("workers")
      .select("worker, address, updated_at").order("worker");
    if (error) return refuse(500, "could not read the mapping");
    const workers: Record<string, string> = {};
    const moved: Record<string, string> = {};
    for (const r of data ?? []) {
      workers[r.worker] = r.address;
      moved[r.worker] = r.updated_at;
    }
    return json(200, { workers, updated_at: moved, count: Object.keys(workers).length });
  }

  if (req.method !== "POST") return refuse(405, "POST only");

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return refuse(400, "body must be JSON");
  }
  const address = typeof body.address === "string" ? body.address.toLowerCase() : "";
  if (!isAddress(address)) return refuse(400, "address must be 0x and 40 hex digits");

  if (step === "nonce") {
    const problem = nameProblem(body.worker);
    if (problem) return refuse(400, problem);
    const worker = body.worker as string;
    const verb = body.verb === "release" ? "Release" : "Claim";
    const nonce = crypto.randomUUID().replace(/-/g, "");
    const issued = new Date();
    const expires = new Date(issued.getTime() + NONCE_TTL_MS);
    const { error } = await db.from("nonces").insert({
      nonce, address, issued_at: issued.toISOString(), expires_at: expires.toISOString(),
    });
    if (error) return refuse(500, "could not issue a nonce");
    return json(200, {
      nonce,
      message: siwe(address, worker, verb, nonce, issued, expires),
      // **`issued_at` travels because the signed text contains it.** Without it a
      // client cannot rebuild the message it is being asked to sign and has to
      // trust the `message` string -- which is most of what SIWE is for: the
      // fields are the thing a caller checks, and text it cannot derive is text
      // it can only accept. Deriving it from `expires_at` minus the TTL would put
      // a second copy of `NONCE_TTL_MS` in every client, which is the class of
      // duplicated constant this tree refuses everywhere else.
      // Found by `tools/workercheck.py` on its first run against the deployed
      // function, rebuilding the message and finding one line it could not.
      issued_at: issued.toISOString(),
      expires_at: expires.toISOString(),
    });
  }

  if (step !== "claim" && step !== "release")
    return refuse(404, "use /worker/nonce, /worker/claim, /worker/release or /worker/map");

  const problem = nameProblem(body.worker);
  if (problem) return refuse(400, problem);
  const worker = body.worker as string;
  const fold = worker.toLowerCase();

  const signature = typeof body.signature === "string" ? body.signature : "";
  if (!/^0x[0-9a-fA-F]{130}$/.test(signature))
    return refuse(400, "signature must be 0x and 130 hex digits");

  const got = await spendNonce(address);
  if (!got.ok) return refuse(got.status, got.why);

  const verb = step === "release" ? "Release" : "Claim";
  const message = siwe(address, worker, verb, got.nonce, got.issued, got.expires);
  const recovered = recoverAddress(message, signature);
  if (!recovered || recovered !== address)
    return refuse(401, "that signature does not belong to that address");
  if (!(await burn(got.nonce))) return refuse(409, "that nonce was already used");

  const { data: held } = await db.from("workers")
    .select("worker, address").eq("worker_fold", fold).maybeSingle();

  if (step === "release") {
    if (!held) return refuse(404, "that name is not claimed");
    if (held.address !== address) return refuse(403, "that name belongs to another address");
    const { error } = await db.from("workers").delete().eq("worker_fold", fold);
    if (error) return refuse(500, "could not release the name");
    return json(200, { released: held.worker });
  }

  if (held && held.address !== address)
    return refuse(409, "that name is already claimed by another address");
  if (held && held.worker !== worker)
    // Same name folded, different spelling. Refused rather than silently
    // rewritten: the ledger will carry whichever the rig actually sends, and
    // quietly changing the stored spelling would break the lookup for shares
    // already logged under the old one.
    return refuse(409, `you hold this name spelled "${held.worker}"; release it first to respell it`);

  if (!held) {
    const { count } = await db.from("workers")
      .select("worker", { count: "exact", head: true }).eq("address", address);
    if ((count ?? 0) >= NAME_CAP)
      return refuse(409, `this address already holds ${NAME_CAP} worker names; release one first`);
  }

  const { error } = await db.from("workers").upsert({
    worker, worker_fold: fold, address, updated_at: new Date().toISOString(),
  }, { onConflict: "worker" });
  if (error) return refuse(500, "could not record the name");

  return json(200, {
    worker,
    address: toChecksum(address),
    note: "Point your rig at this exact name. A different spelling accrues to nobody.",
  });
});
