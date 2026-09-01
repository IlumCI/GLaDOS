// The gated channel.
//
// A machine sends its device code as a bearer token and gets back a signed
// manifest for the newest experimental build, with the image URLs rewritten
// into short-lived signed ones. That rewrite is the whole of the gate: the
// bucket is private, so the URLs in the stored manifest are unreachable, and
// these are good for an hour.
//
// ### What this cannot do, said here so nobody builds on the belief that it can
//
// The image it hands over is signed by the same key as stable and installs on
// any machine. There is no per-machine binding and there could not be a
// meaningful one -- the kernel's source is published, so any local check is a
// check the machine's owner can delete. The gate is that the server declines
// to answer, and that is the only claim worth making for it.
//
// ### Why the manifest is re-signed rather than edited
//
// The kernel verifies a P-256 signature over the manifest's exact bytes.
// Rewriting a URL changes those bytes, so an edited manifest fails
// verification -- correctly. So this function holds the signing key and
// re-signs what it rewrites. That is a real cost: the private half of the
// update key lives in a second place. The alternative is signed URLs baked in
// at publish time, which would expire and could not be renewed without
// republishing, so a build would stop being installable an hour after CI ran.

import { createClient } from "jsr:@supabase/supabase-js@2";
// Plain JS, and imported rather than inlined, because it is the half
// that can be run by node and cross-checked against the Python verifier
// that mirrors the kernel's. See _shared/crosscheck.mjs.
import { sha256, signedManifest } from "../_shared/gladosig.js";
import { balanceOf } from "../_shared/evm.js";

const TTL_SECONDS = 60 * 60;

const url = Deno.env.get("SUPABASE_URL")!;
const service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const signingKey = Deno.env.get("UPDATE_SIGNING_KEY")!;

// The same three values the link function reads, from the same environment, so
// the threshold cannot drift between the door that issues a code and the door
// that honours one.
const TOKEN = (Deno.env.get("TOKEN_CONTRACT") ??
  "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777").toLowerCase();
const RPC = Deno.env.get("TOKEN_RPC") ?? "https://rpc.mainnet.chain.robinhood.com";
const MIN_BALANCE = BigInt(Deno.env.get("TOKEN_MIN_BALANCE") ?? "1000000000000000000000000");

const db = createClient(url, service);

// --- entitlement ---------------------------------------------------------

/// Whether this device may have an experimental build.
///
/// ONE function, one job, and the only thing that knows about entitlement.
/// Two ways in, and they answer different needs.
///
/// `allowlist` was the first implementation and stays: it hands access to a
/// tester or a contributor without a wallet, and it is the only route for
/// anyone holding custodially, who has no key to sign with. Rows go in by hand.
///
/// The balance read is the second, and it exists because the token launched as
/// an ERC-20 in self-custody rather than inside a custodial app -- so holders
/// do have keys, and `link` can prove an address. That was the open question
/// this comment used to describe, and it is closed.
///
/// Everything else in this file was written not to care which of the two said
/// yes, and still does not.
async function entitled(codeHash: string): Promise<boolean> {
  const now = new Date().toISOString();

  const { data: allow } = await db
    .from("allowlist")
    .select("code_hash")
    .eq("code_hash", codeHash)
    .or(`expires_at.is.null,expires_at.gt.${now}`)
    .maybeSingle();
  if (allow) return true;

  const { data: device } = await db
    .from("devices")
    .select("code_hash, wallet, revoked")
    .eq("code_hash", codeHash)
    .maybeSingle();
  if (!device || device.revoked) return false;
  if (!device.wallet) return false;

  // The chain half, written now that there is a chain to read.
  //
  // Re-read every time rather than trusting `wallets.linked_balance`. A
  // holding is not a permanent fact about a person, and a cached one is a gate
  // that stays open after the thing it was gating on is gone. It costs one
  // eth_call against a public RPC per download, which is nothing beside the
  // image that follows.
  let balance: bigint;
  try {
    balance = await balanceOf(RPC, TOKEN, device.wallet);
  } catch {
    // An RPC that is unreachable is not a holder who sold. Refusing here is
    // the safe direction -- the alternative is an outage that opens the gate
    // -- but it is a refusal for a reason the holder cannot act on, so it is
    // logged as itself rather than folded into "not entitled".
    console.error("entitled: rpc unreachable for", device.wallet);
    return false;
  }
  await db.from("wallets")
    .update({ linked_balance: balance.toString(), last_checked: new Date().toISOString() })
    .eq("address", device.wallet);
  return balance >= MIN_BALANCE;
}

// --- the handler ---------------------------------------------------------

function refuse(status: number, why: string): Response {
  // Plain text, because the client parsing this is a kernel with no JSON
  // reader. It only ever looks at the status; the body is for a person with
  // curl trying to work out what went wrong.
  return new Response(why + "\n", {
    status,
    headers: { "content-type": "text/plain" },
  });
}

Deno.serve(async (req) => {
  const auth = req.headers.get("authorization") ?? "";
  const code = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
  if (!code) return refuse(401, "no device code");

  const hashBytes = await sha256(new TextEncoder().encode(code));
  const codeHash = Array.from(hashBytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  if (!(await entitled(codeHash))) {
    // Deliberately the same answer for "no such code" and "a code that is not
    // entitled". Telling them apart turns this into an oracle for guessing
    // valid codes.
    return refuse(403, "this device is not entitled to the experimental channel");
  }

  await db.from("devices").update({ last_seen: new Date().toISOString() })
    .eq("code_hash", codeHash);

  const { data: build } = await db
    .from("builds")
    .select("slug, version, sha256, size, object_path, notes")
    .eq("channel", "experimental")
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (!build) return refuse(404, "no experimental build has been published");

  // Sign a URL for the image and its signature. The stored manifest names
  // paths in a private bucket; these are what a machine can actually fetch.
  const image = `${build.slug}/glados-${build.version}.efi`;
  const { data: signed, error } = await db.storage
    .from("experimental")
    .createSignedUrls([image, `${image}.sig`], TTL_SECONDS);
  if (error || !signed || signed.length !== 2) {
    return refuse(500, "could not sign the download URLs");
  }
  const origin = url.replace(/\/$/, "");
  const full = signed.map((s) =>
    s.signedUrl.startsWith("http") ? s.signedUrl : origin + "/" + s.signedUrl.replace(/^\//, "")
  );
  if (!full.every((u) => u.startsWith("https://"))) {
    // The kernel refuses a plain-http URL in a manifest, and it would do so
    // AFTER verifying the signature -- which reads as a signing problem.
    return refuse(500, "the signed URLs are not https");
  }

  // Field order and spelling match src/update/manifest.rs. Unknown keys are
  // ignored there, missing ones are refused, and `notes` takes the rest of
  // its line -- so it is flattened to one line here.
  const text = [
    "glados-update 1",
    "channel experimental",
    `version ${build.version}`,
    `image ${full[0]}`,
    `sig ${full[1]}`,
    `size ${build.size}`,
    `sha256 ${build.sha256}`,
    `notes ${String(build.notes ?? "").replace(/\s+/g, " ").trim()}`,
    "",
  ].join("\n");

  const out = await signedManifest(signingKey, text);

  return new Response(out, {
    headers: {
      "content-type": "application/octet-stream",
      "content-length": String(out.length),
      // Short-lived URLs inside, so this must not be cached anywhere.
      "cache-control": "no-store",
    },
  });
});
