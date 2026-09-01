// Linking a wallet to a device code.
//
// Two steps, both POST, distinguished by the last path segment:
//
//   /link/nonce   {address}                  -> {nonce, message, expires_at}
//   /link/verify  {address, signature}       -> {code, address, balance}
//
// The code comes back exactly once and is never retrievable again. Only its
// hash is stored, for the reason `devices` gives: a bearer token the server
// can read back is a credential a database leak hands over.
//
// ### What the signature proves, and what it does not
//
// It proves the signer holds the key for the recovered address. That is all,
// and it is enough -- the balance is a separate question answered by an
// `eth_call` against the token contract. Keeping them apart means a wallet
// that has never added Robinhood Chain can still link, because the proof does
// not touch a chain at all.
//
// ### Where the balance is read, and how often
//
// Here, server-side, and again in `channel` at every download. Never on the
// client, which can claim whatever it likes, and never cached as a permanent
// fact, because a holding is not one. `wallets.linked_balance` is recorded for
// support and is not consulted for entitlement.

import { createClient } from "jsr:@supabase/supabase-js@2";
import { recoverAddress, balanceOf, toChecksum } from "../_shared/evm.js";

const url = Deno.env.get("SUPABASE_URL")!;
const service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(url, service);

// The token, the chain and the threshold. Environment rather than constants
// so the threshold can move without a redeploy of anything the kernel trusts.
const TOKEN = (Deno.env.get("TOKEN_CONTRACT") ??
  "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777").toLowerCase();
const RPC = Deno.env.get("TOKEN_RPC") ?? "https://rpc.mainnet.chain.robinhood.com";
const CHAIN_ID = Number(Deno.env.get("TOKEN_CHAIN_ID") ?? 4663);
// 1,000,000 tokens at 18 decimals.
const MIN_BALANCE = BigInt(Deno.env.get("TOKEN_MIN_BALANCE") ?? "1000000000000000000000000");
const DOMAIN = Deno.env.get("LINK_DOMAIN") ?? "glados.aperture.institute";

const NONCE_TTL_MS = 10 * 60 * 1000;

const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "content-type",
  "access-control-allow-methods": "POST, OPTIONS",
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

async function sha256Hex(s: string): Promise<string> {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

/// The EIP-4361 message. Domain-bound and time-bound on purpose: a signature
/// harvested by another site cannot be replayed here, because the domain line
/// names this one.
function siwe(address: string, nonce: string, issued: Date, expires: Date) {
  return [
    `${DOMAIN} wants you to sign in with your Ethereum account:`,
    toChecksum(address),
    "",
    "Link this wallet to a GLaDOS device code for the experimental update channel.",
    "This signature costs nothing, moves nothing, and approves no transaction.",
    "",
    `URI: https://${DOMAIN}/wallet/`,
    "Version: 1",
    `Chain ID: ${CHAIN_ID}`,
    `Nonce: ${nonce}`,
    `Issued At: ${issued.toISOString()}`,
    `Expiration Time: ${expires.toISOString()}`,
  ].join("\n");
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: cors });
  if (req.method !== "POST") return refuse(405, "POST only");

  const step = new URL(req.url).pathname.split("/").filter(Boolean).pop();
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return refuse(400, "body must be JSON");
  }
  const address = typeof body.address === "string" ? body.address.toLowerCase() : "";
  if (!isAddress(address)) return refuse(400, "address must be 0x and 40 hex digits");

  // ---- step one: hand out a nonce ---------------------------------------
  if (step === "nonce") {
    const nonce = crypto.randomUUID().replace(/-/g, "");
    const issued = new Date();
    const expires = new Date(issued.getTime() + NONCE_TTL_MS);
    const { error } = await db.from("nonces").insert({
      nonce, address, issued_at: issued.toISOString(), expires_at: expires.toISOString(),
    });
    if (error) return refuse(500, "could not issue a nonce");
    return json(200, {
      nonce,
      message: siwe(address, nonce, issued, expires),
      expires_at: expires.toISOString(),
    });
  }

  if (step !== "verify") return refuse(404, "use /link/nonce or /link/verify");

  // ---- step two: check the signature, then the chain ---------------------
  const signature = typeof body.signature === "string" ? body.signature : "";
  if (!/^0x[0-9a-fA-F]{130}$/.test(signature))
    return refuse(400, "signature must be 0x and 130 hex digits");

  const { data: row } = await db.from("nonces")
    .select("nonce, address, issued_at, expires_at, used_at")
    .eq("address", address).is("used_at", null)
    .order("issued_at", { ascending: false }).limit(1).maybeSingle();
  if (!row) return refuse(400, "no unused nonce for that address -- ask for one first");
  if (new Date(row.expires_at) < new Date()) return refuse(400, "that nonce has expired");

  const message = siwe(address, row.nonce, new Date(row.issued_at), new Date(row.expires_at));
  const recovered = recoverAddress(message, signature);
  if (!recovered || recovered !== address)
    return refuse(401, "that signature does not belong to that address");

  // Spend the nonce before doing anything else, and only if it is still
  // unspent. Two requests racing one nonce: the second update matches no row.
  const { data: spent } = await db.from("nonces")
    .update({ used_at: new Date().toISOString() })
    .eq("nonce", row.nonce).is("used_at", null).select("nonce").maybeSingle();
  if (!spent) return refuse(409, "that nonce was already used");

  let balance: bigint;
  try {
    balance = await balanceOf(RPC, TOKEN, address);
  } catch {
    // An RPC that is down is not a holder who owns nothing, and answering 403
    // here would tell somebody their wallet failed when the server did.
    return refuse(503, "could not read the balance from the chain; try again shortly");
  }
  if (balance < MIN_BALANCE)
    return json(403, {
      error: "balance below the threshold",
      balance: balance.toString(),
      required: MIN_BALANCE.toString(),
    });

  await db.from("wallets").upsert({
    address, chain: `eip155:${CHAIN_ID}`,
    linked_balance: balance.toString(), last_checked: new Date().toISOString(),
  });

  const { data: wallet } = await db.from("wallets")
    .select("device_cap").eq("address", address).maybeSingle();
  const cap = wallet?.device_cap ?? 5;
  const { count } = await db.from("devices")
    .select("code_hash", { count: "exact", head: true })
    .eq("wallet", address).eq("revoked", false);
  if ((count ?? 0) >= cap)
    return refuse(409, `this wallet already holds ${cap} device codes; revoke one first`);

  // The code, once. Base32-ish over an alphabet without the characters people
  // mistype reading them aloud.
  const raw = crypto.getRandomValues(new Uint8Array(20));
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let i = 0; i < raw.length; i++) {
    if (i && i % 5 === 0) code += "-";
    code += alphabet[raw[i] % 32];
  }
  const { error: insErr } = await db.from("devices")
    .insert({ code_hash: await sha256Hex(code), wallet: address });
  if (insErr) return refuse(500, "could not store the device code");

  return json(200, {
    code,
    address: toChecksum(address),
    balance: balance.toString(),
    note: "This code is shown once. Store it now; the server keeps only its hash.",
  });
});
