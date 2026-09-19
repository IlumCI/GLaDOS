// The verdict's way home -- the missing return arrow in the loop diagram.
//
// POST (from CI): `Authorization: Bearer <VERDICT_INGEST_TOKEN>`, body the
// raw signed bin. The blob's GLADOSIG is verified against the pinned
// verdict point BEFORE anything is stored, so a leaked ingest token can
// insert only verdicts the verdict key actually signed -- the same class
// of statement propose.yml makes about its own signing step. The matching
// `proposals` row (same point, newest dispatched) turns `answered` and
// lends its device hash, which is who may fetch this.
//
// GET (from a machine): `Authorization: Bearer <device code>`, the same
// allowlist gate the proposal function uses -- deliberately NOT channel's
// balance gate. A device that may propose may collect its answers, and
// coupling retrieval to a wallet balance would let a balance dip strand a
// signed answer that was already paid for.
//
// The kernel side is `godel verdicts`: fetch, then `file_verdict`, which
// verifies again with the compiled-in anchor -- this service is a mailbox,
// never a authority, and a machine trusts nothing it did not check itself.

import { createClient } from "jsr:@supabase/supabase-js@2";
import { sha256 } from "../_shared/gladosig.js";
import { verifyVerdict } from "../_shared/verdict_key.js";

const url = Deno.env.get("SUPABASE_URL")!;
const service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(url, service);

// The shared secret CI posts with. Absent means the door is closed, which
// is a configuration rather than an error, and ingest answers 503 the way
// the proposal function does without its dispatch token.
const INGEST = Deno.env.get("VERDICT_INGEST_TOKEN") ?? "";

// A verdict is a few hundred bytes; the kernel's own envelope cap is 4096.
// Anything bigger is not a verdict, whatever it is.
const MAX_BIN = 16 * 1024;

function refuse(status: number, why: string): Response {
  return new Response(`${why}\n`, {
    status,
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
}

function bearer(req: Request): string {
  const auth = req.headers.get("authorization") ?? "";
  return auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
}

async function hashHex(s: string): Promise<string> {
  const bytes = await sha256(new TextEncoder().encode(s));
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** The proposal function's gate, verbatim in behaviour: allowlist only. */
async function admitted(codeHash: string): Promise<boolean> {
  const now = new Date().toISOString();
  const { data } = await db
    .from("allowlist")
    .select("code_hash")
    .eq("code_hash", codeHash)
    .or(`expires_at.is.null,expires_at.gt.${now}`)
    .maybeSingle();
  return Boolean(data);
}

async function ingest(req: Request): Promise<Response> {
  if (!INGEST) return refuse(503, "this service has no ingest token configured");
  const token = bearer(req);
  if (!token || token !== INGEST) {
    // One answer for absent and wrong, as everywhere else here.
    return refuse(403, "this is not the ingest token");
  }
  const buf = new Uint8Array(await req.arrayBuffer());
  if (buf.length > MAX_BIN) {
    return refuse(400, `${buf.length} B is not a verdict`);
  }

  // **Verified before stored.** The whole reason this function exists
  // rather than a storage bucket.
  const v = await verifyVerdict(buf);
  if (!v.ok) return refuse(400, `refused: ${v.why}`);

  // The point, out of the verified text. The kernel's parse_verdict is the
  // format authority; this reads the one line the row is keyed by, and a
  // text without it is refused rather than filed unfindable.
  const text = new TextDecoder().decode(v.text);
  const m = text.match(/^point ([0-9a-f]{64})$/m);
  if (!m) return refuse(400, "the verdict names no point");
  const point = m[1];

  const runId = req.headers.get("x-run-id") ?? "0";
  const runAttempt = req.headers.get("x-run-attempt") ?? "0";

  // The newest dispatched proposal with this point lends its device.
  const { data: prop } = await db
    .from("proposals")
    .select("id, code_hash")
    .eq("point", point)
    .eq("status", "dispatched")
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  const b64 = btoa(String.fromCharCode(...buf));
  const { error } = await db.from("verdicts").insert({
    proposal_id: prop?.id ?? null,
    point,
    code_hash: prop?.code_hash ?? "",
    body: b64,
    run_id: runId,
    run_attempt: runAttempt,
  });
  if (error) return refuse(500, "could not store the verdict");

  if (prop) {
    await db.from("proposals").update({ status: "answered" }).eq("id", prop.id);
  }
  return new Response(
    `verdict stored 1\npoint ${point}\nfor ${prop ? "its proposer" : "audit only"}\n`,
    { status: 202, headers: { "content-type": "text/plain; charset=utf-8" } },
  );
}

async function serve(req: Request): Promise<Response> {
  const code = bearer(req);
  if (!code) return refuse(401, "no device code");
  const codeHash = await hashHex(code);
  if (!(await admitted(codeHash))) {
    return refuse(403, "this device may not collect verdicts");
  }

  const again = new URL(req.url).searchParams.get("again") === "1";
  let q = db
    .from("verdicts")
    .select("id, body")
    .eq("code_hash", codeHash)
    .order("created_at", { ascending: true })
    .limit(1);
  if (!again) q = q.is("claimed_at", null);
  const { data: row } = await q.maybeSingle();
  if (!row) return new Response(null, { status: 204 });

  if (!again) {
    await db.from("verdicts")
      .update({ claimed_at: new Date().toISOString() })
      .eq("id", row.id);
  }
  const bin = Uint8Array.from(atob(row.body), (c) => c.charCodeAt(0));
  return new Response(bin, {
    status: 200,
    headers: {
      "content-type": "application/octet-stream",
      "cache-control": "no-store",
    },
  });
}

Deno.serve(async (req) => {
  if (req.method === "POST") return ingest(req);
  if (req.method === "GET") return serve(req);
  return refuse(405, "POST a signed verdict, or GET the oldest unclaimed one");
});
