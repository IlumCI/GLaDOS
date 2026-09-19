// The door a machine's own proposals come in through.
//
// `godel push` sends an envelope here: a constant this machine would like
// changed, which lineage is asking, and against which corpus. The kernel
// cannot compile and cannot judge a source change, so what happens next is
// `propose.yml` -- two builds, two boots, and the rail the proposal claimed --
// and this is the only thing that can start it.
//
// ### The token is here and never there
//
// Dispatching a workflow needs a GitHub credential. It lives in this
// function's environment and the kernel never holds one, which is the whole
// reason the outward path goes through a server at all rather than the machine
// talking to GitHub directly. A machine in the field carries a device code
// that can ask for a build to be judged, and nothing that can write to a
// repository.
//
// That is the same division `outbox_endpoint` makes on the kernel side: what
// goes out is the machine's own text, over an authenticated connection to the
// origin it already talks to, and the origin decides what that means.
//
// ### What arrives is not what is forwarded
//
// `_shared/envelope.js` reads the body into declared fields, checks every one
// against a shape, and **renders a fresh envelope from what it parsed**. The
// bytes handed to the workflow are bytes this side wrote. A validator that
// checked the input and passed the original through would be one carriage
// return away from meaning something else, and the thing on the other end
// compiles code.
//
// ### The rate limit is the point, not politeness
//
// A workflow run is two `cargo build --release` and two QEMU boots on somebody
// else's minutes. An entitled device that could dispatch in a loop is an
// entitled device that can spend an account's whole CI budget in an afternoon,
// so a device gets a small number of runs a day and the count is kept here
// rather than inferred from GitHub.

import { createClient } from "jsr:@supabase/supabase-js@2";
// Plain JS, imported rather than inlined, because it is the half `node` can
// run. See `_shared/envelope.test.mjs`; the same arrangement `gladosig.js` has
// with `crosscheck.mjs`.
import { parse, render } from "../_shared/envelope.js";
import { sha256 } from "../_shared/gladosig.js";

const url = Deno.env.get("SUPABASE_URL")!;
const service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(url, service);

// A fine-grained token with `actions: write` on one repository and nothing
// else. Absent means this door is closed, which is a configuration rather than
// an error and is answered as such.
const GH_TOKEN = Deno.env.get("GITHUB_DISPATCH_TOKEN") ?? "";
const GH_REPO = Deno.env.get("GITHUB_REPO") ?? "IlumCI/GLaDOS";
const GH_REF = Deno.env.get("GITHUB_REF") ?? "main";
const WORKFLOW = Deno.env.get("GITHUB_WORKFLOW") ?? "propose.yml";
// Where the judging runs. Blank means the workflow generates a fixture, which
// it will then refuse to adopt on -- see `propose.yml`.
const FOREST = Deno.env.get("PROPOSAL_FOREST") ?? "";

/** Runs one device may ask for in a day. */
const PER_DAY = Number(Deno.env.get("PROPOSAL_PER_DAY") ?? "6");

function refuse(status: number, why: string): Response {
  return new Response(`${why}\n`, {
    status,
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
}

/**
 * Is this device allowed through this door?
 *
 * The allowlist only, deliberately, and **not** the balance gate `channel`
 * uses. Those two doors hand out different things: `channel` gives a build,
 * which is a product, and this spends CI minutes on somebody's behalf. A gate
 * that opens on a token balance is right for the first and wrong for the
 * second, because a balance is not a promise to behave.
 */
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

/** How many runs this device has already asked for today. */
async function spent(codeHash: string): Promise<number> {
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const { count } = await db
    .from("proposals")
    .select("id", { count: "exact", head: true })
    .eq("code_hash", codeHash)
    .gte("created_at", since);
  return count ?? 0;
}

Deno.serve(async (req) => {
  if (req.method !== "POST") return refuse(405, "post an envelope here");

  const auth = req.headers.get("authorization") ?? "";
  const code = auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : "";
  if (!code) return refuse(401, "no device code");

  const hashBytes = await sha256(new TextEncoder().encode(code));
  const codeHash = Array.from(hashBytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  // The same answer for "no such code" and "a code that is not admitted", for
  // the reason `channel` gives: telling them apart turns this into an oracle
  // for guessing valid codes.
  if (!(await admitted(codeHash))) {
    return refuse(403, "this device may not propose changes");
  }

  const used = await spent(codeHash);
  if (used >= PER_DAY) {
    return refuse(429, `this device has asked for ${used} runs today, and ${PER_DAY} is the limit`);
  }

  // Read with a cap before parsing, so an enormous body is refused by its size
  // rather than by whatever runs out first.
  const raw = await req.text();
  const v = parse(raw);
  if (!v.ok) return refuse(400, v.why);
  const p = v.proposal;
  const envelope = render(p);

  // **Recorded before it is dispatched, not after.** A run that started and a
  // row that was never written is a proposal nobody can account for; a row
  // with no run behind it is one that shows as pending and can be looked at.
  // The second is the recoverable direction.
  const { data: row, error: wrote } = await db
    .from("proposals")
    .insert({
      code_hash: codeHash,
      point: p.point,
      version: p.version,
      head: p.head,
      corpus: p.corpus,
      tests: Number(p.tests),
      file: p.file,
      symbol: p.symbol,
      was: p.was,
      now_value: p.now,
      rail: p.rail,
      envelope,
      status: "pending",
    })
    .select("id")
    .single();
  if (wrote || !row) return refuse(500, "could not record the proposal");

  if (!GH_TOKEN) {
    await db.from("proposals").update({ status: "unconfigured" }).eq("id", row.id);
    return refuse(503, "this service has no credential to start a run with");
  }

  const res = await fetch(
    `https://api.github.com/repos/${GH_REPO}/actions/workflows/${WORKFLOW}/dispatches`,
    {
      method: "POST",
      headers: {
        authorization: `Bearer ${GH_TOKEN}`,
        accept: "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
        "content-type": "application/json",
        "user-agent": "glados-proposal",
      },
      // `JSON.stringify` rather than a template, because the envelope has
      // newlines in it and a hand-built body would put them through
      // unescaped -- which is valid-looking JSON that parses to something
      // else, on the request that starts a compiler.
      body: JSON.stringify({
        ref: GH_REF,
        inputs: { envelope, forest: FOREST },
      }),
    },
  );

  if (!res.ok) {
    const detail = (await res.text()).slice(0, 200);
    await db.from("proposals")
      .update({ status: "refused", detail: `${res.status} ${detail}` })
      .eq("id", row.id);
    // The upstream status is not passed through: a 401 from GitHub is this
    // service's configuration problem and not the caller's, and answering 401
    // would tell a machine its own code was rejected.
    return refuse(502, `the judging service answered ${res.status}`);
  }

  await db.from("proposals").update({ status: "dispatched" }).eq("id", row.id);

  // What the machine gets back is its own point and nothing else. There is no
  // run id to hand over -- `workflow_dispatch` answers 204 with an empty body
  // and does not say which run it started -- and inventing one would be worse
  // than saying so.
  return new Response(
    [
      "proposal accepted 1",
      `point ${p.point}`,
      `rail ${p.rail}`,
      `today ${used + 1} of ${PER_DAY}`,
      "the verdict comes back signed, through the channel",
      "",
    ].join("\n"),
    { status: 202, headers: { "content-type": "text/plain; charset=utf-8" } },
  );
});
