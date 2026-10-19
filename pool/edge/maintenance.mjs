#!/usr/bin/env node
// Declare the pool down for maintenance, or up again. The rules are in
// worker/maintenance.js; this only asks.
//
//   node pool/edge/maintenance.mjs on "moving the treasury" [--minutes 30] [--drain 20]
//   node pool/edge/maintenance.mjs off
//   node pool/edge/maintenance.mjs status
//   node pool/edge/maintenance.mjs token      # once: make the secret, then
//       npx wrangler secret put MAINT_TOKEN < ~/.config/glados/maint-token   (in pool/edge/worker)
//
// The token comes from $GLADOS_MAINT_TOKEN, else ~/.config/glados/maint-token.
// --pool (or $GLADOS_POOL) picks another pool, e.g. a `wrangler dev` one.
import { readFileSync, writeFileSync, mkdirSync, existsSync, chmodSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { randomBytes } from "node:crypto";

const TOKEN_FILE = join(homedir(), ".config", "glados", "maint-token");
const args = process.argv.slice(2);
const flag = (name) => { const i = args.indexOf(`--${name}`); if (i < 0) return undefined; const v = args[i + 1]; args.splice(i, 2); return v; };
const pool = (flag("pool") || process.env.GLADOS_POOL || "https://pool.aperture.institute").replace(/\/+$/, "");
const minutes = flag("minutes"), drain = flag("drain");
const [cmd, ...rest] = args;
const die = (msg) => { console.error(msg); process.exit(1); };
const when = (ms) => (ms ? new Date(ms).toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC") : "-");

function token() {
  if (process.env.GLADOS_MAINT_TOKEN) return process.env.GLADOS_MAINT_TOKEN.trim();
  if (existsSync(TOKEN_FILE)) return readFileSync(TOKEN_FILE, "utf8").trim();
  die(`no token: set GLADOS_MAINT_TOKEN, or run \`node ${process.argv[1]} token\` and put it on the pool`);
}

function show(m) {
  if (!m || !m.on) return console.log(`up${m && m.since ? ` (since ${when(m.since)})` : ""}`);
  console.log(`DOWN for maintenance since ${when(m.since)}: ${m.reason}`);
  if (m.until) console.log(`  expected back by ${when(m.until)}${m.until < Date.now() ? "  (overdue)" : ""}`);
}

async function declare(body) {
  let res;
  try {
    res = await fetch(`${pool}/maintenance`, { method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${token()}` }, body: JSON.stringify(body) });
  } catch (e) { die(`${pool} did not answer: ${e.message}`); }
  const out = await res.json().catch(() => ({}));
  if (!res.ok) die(`${pool} refused (${res.status}): ${out.error || out.why || "no reason given"}`);
  show(out.maintenance);
  let missed = 0;
  for (const s of out.shards || []) {
    if (!s.ok) missed++;
    console.log(`  ${s.name.padEnd(8)} ${s.ok ? `${s.connections ?? 0} connected${s.draining ? ", told and draining" : ""}${s.owedUpstream ? `, ${s.owedUpstream} share(s) owed upstream` : ""}` : `NOT REACHED: ${s.why}`}`);
  }
  if (body.on) console.log(`  miners keep being credited for ${out.maintenance.drainMs / 1000} s, then are closed; payouts pause until \`off\``);
  if (missed) die(`${missed} shard(s) not reached: they still refuse at the router, and will pick the state up from the pool on their next connection. Run it again to retry.`);
}

if (cmd === "on") {
  const reason = rest.join(" ").trim();
  if (!reason) die('say why: node maintenance.mjs on "what is happening" [--minutes N] [--drain S]');
  const body = { on: true, reason };
  if (minutes !== undefined) body.minutes = Number(minutes);
  if (drain !== undefined) body.drainSeconds = Number(drain);
  await declare(body);
} else if (cmd === "off") {
  await declare({ on: false });
} else if (cmd === "status") {
  const res = await fetch(`${pool}/maintenance?_=${Date.now()}`).catch((e) => die(`${pool} did not answer: ${e.message}`));
  if (!res.ok) die(`${pool} answered ${res.status}`);
  show(await res.json());
} else if (cmd === "token") {
  if (existsSync(TOKEN_FILE)) die(`${TOKEN_FILE} exists already; delete it first to make a new one (and put the new one on the pool)`);
  mkdirSync(dirname(TOKEN_FILE), { recursive: true, mode: 0o700 });
  writeFileSync(TOKEN_FILE, randomBytes(32).toString("base64url") + "\n", { mode: 0o600 });
  chmodSync(TOKEN_FILE, 0o600);
  console.log(`wrote ${TOKEN_FILE} (only you can read it). Put it on the pool, from pool/edge/worker:\n  npx wrangler secret put MAINT_TOKEN < ${TOKEN_FILE}`);
} else {
  die("usage: maintenance.mjs on \"reason\" [--minutes N] [--drain S] | off | status | token   [--pool URL]");
}
