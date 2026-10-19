// The Pearl shard's rules, with no sockets in them: what a miner is sent, what
// counts as a share, what is credited, and what is passed upstream. The
// Durable Object in pearl-pool.js only moves lines and keeps storage; this is
// the part a test can hold without Cloudflare.
//
// ### Upstream
//
// A PRL pool (2Miners) sends `mining.notify {job_id, header, target, height}`
// after `mining.authorize {wallet}`. The header is Pearl's 76-byte incomplete
// block header; the target is big-endian hex and is scaled by the tile's
// adjustment factor before a jackpot hash is held against it -- Pearl's own
// `penalized_target_bound`. Every miner here mines that same header: unlike a
// nonce search there is no space to split, because each miner commits its own
// random matrices and so its own noise.
//
// ### Miners
//
//   <- {"id":1,"method":"glados.hello","params":{"v":1,"worker":"0x...","agent":"...","reward":"..."}}
//   -> {"id":1,"result":{"pearl":{"k":2048,"rank":128,"tile":16}}}
//   -> {"method":"glados.pearl.job","params":{"job":"...","header":"<152 hex>","bits":26,"k":2048,"rank":128,"cert":3}}
//   <- {"id":n,"method":"glados.pearl.submit","params":{"job":"...","proof":"<base64 plain_proof>"}}
//   -> {"id":n,"result":{"ok":true,"verdict":"accepted"}}   (or ok:false with stale, duplicate, low, invalid)
//
// A share is a tile whose jackpot hash, read as Pearl reads it (a little-endian
// 256-bit integer), is below 2^(256 - bits). It is credited 2^bits -- the tiles
// expected per share -- which is how the core credits every other algorithm:
// work is expected hashes, never a share count. VarDiff moves `bits` so each
// miner sends a share every TARGET_S or so, whatever its speed.
//
// Every share is verified in full (pearlproof.js, checked against Pearl's own
// verifier) before it is credited or forwarded; a hash the miner claims is
// never trusted. A share that also meets upstream's bound is forwarded, which
// is the only thing that earns the pool anything.

import { verifyShare, adjustmentFactor, scaleTarget, fromBase64, toBase64, decodeProof, encodeProof } from "./pearlproof.js";

export const K = 2048;
export const RANK = 128;
export const TILE = 16;
const FACTOR = adjustmentFactor(K, RANK); // tile x (k / rank) x 128 = 2^19, what a target is scaled by
const TARGET_S = 20;      // a share this often per miner
const MIN_BITS = 16, START_BITS = 24; // defaults; PEARL_MIN_BITS / PEARL_START_BITS lower them for tests
const JOBS_KEPT = 4;      // upstream jobs a share may still be for
const MAX_SEEN = 50000;   // duplicate memory, across the jobs kept
// A share costs ~9 ms of verification, so submits are rationed before any of
// that is spent: a bucket of SUBMIT_BURST refilling at SUBMIT_RATE a second --
// far above one share per TARGET_S, far below what would starve the shard.
const SUBMIT_RATE = 2, SUBMIT_BURST = 10;
// A connection whose proofs keep failing verification is closed: past
// BAD_MIN failures, once they outnumber its accepted shares.
const BAD_MIN = 10;

const hexOk = (s, n) => typeof s === "string" && s.length === n && /^[0-9a-fA-F]+$/.test(s);
const toU8 = (h) => Uint8Array.from(h.match(/../g), (b) => parseInt(b, 16));
const reply = (id, result) => JSON.stringify({ id, result }) + "\n";
const verdict = (id, ok, v, why) => reply(id, why ? { ok, verdict: v, why } : { ok, verdict: v });

export class PearlRoom {
  constructor({ wallet, worker = "glados", now = () => Date.now(), minBits = MIN_BITS, startBits = START_BITS } = {}) {
    this.wallet = wallet;
    this.minBits = Math.max(0, minBits);
    this.startBits = Math.max(this.minBits, startBits);
    this.worker = worker;
    this.now = now;
    this.jobs = [];            // newest first: { id, header (Uint8Array), headerHex, upBound (BigInt), height }
    // Jackpot hashes already credited, over every job kept. Not per job: an
    // upstream that re-issues one header under a new job id (it does, on a
    // difficulty change) would otherwise credit the same tile twice. The hash
    // already depends on the header, so one set covers every job.
    this.seen = new Set();
    this.conns = new Map();    // connId -> { addr, bits, last, ema, sent }
    this.work = new Map();     // address -> BigInt, all-time credited tiles
    this.stats = { accepted: 0, forwarded: 0, upAccepted: 0, upRejected: 0, rejected: 0 };
    this.nextUpId = 2;
    this.pendingUp = new Map(); // upstream submit id -> { addr, at }
  }

  // ---------------------------------------------------------------- upstream
  authorize() {
    return JSON.stringify({ id: 1, method: "mining.authorize", params: { wallet: this.wallet, worker: this.worker, pass: "x" } }) + "\n";
  }

  /** One line from upstream. Answers { newJob, log: [] }. */
  upLine(line) {
    const out = { newJob: false, log: [] };
    let m;
    try { m = JSON.parse(line); } catch { m = null; }
    // JSON, and an object: `null`, `42` and `[]` parse but are not messages.
    if (!m || typeof m !== "object" || Array.isArray(m)) { out.log.push(`[pearl up] unreadable line: ${String(line).slice(0, 120)}`); return out; }
    if (m.method === "mining.notify" && m.params && typeof m.params === "object") {
      const p = m.params;
      if (!hexOk(p.header, 152) || !hexOk(p.target, 64) || typeof p.job_id !== "string") {
        out.log.push(`[pearl up] a job this pool cannot read: ${JSON.stringify(p).slice(0, 160)}`);
        return out;
      }
      if (this.jobs[0] && this.jobs[0].id === p.job_id) return out;
      const upBound = scaleTarget(BigInt("0x" + p.target), FACTOR);
      this.jobs.unshift({ id: p.job_id, header: toU8(p.header), headerHex: p.header.toLowerCase(), upBound, height: p.height ?? null });
      this.jobs.length = Math.min(this.jobs.length, JOBS_KEPT);
      out.newJob = true;
      return out;
    }
    if (m.id === 1) {
      if (m.result === true) out.log.push(`[pearl up] authorized as ${this.wallet.slice(0, 10)}...`);
      else out.log.push(`[pearl up] authorize refused: ${JSON.stringify(m.error)}`);
      return out;
    }
    if (this.pendingUp.has(m.id)) {
      const s = this.pendingUp.get(m.id);
      this.pendingUp.delete(m.id);
      if (m.result && !m.error) { this.stats.upAccepted++; out.log.push(`[pearl up] share from ${s.addr.slice(0, 10)}... accepted upstream`); }
      else { this.stats.upRejected++; out.log.push(`[pearl up] share from ${s.addr.slice(0, 10)}... rejected upstream: ${JSON.stringify(m.error ?? m.result)}`); }
    }
    return out;
  }

  // ------------------------------------------------------------------ miners
  open(id) { this.conns.set(id, { addr: null, bits: this.startBits, last: this.now(), ema: TARGET_S, sent: null, tokens: SUBMIT_BURST, filled: this.now(), bad: 0, good: 0 }); }
  close(id) { this.conns.delete(id); }
  addresses() { return new Set([...this.conns.values()].map((c) => c.addr).filter(Boolean)); }

  /** The current job for one connection, or null with none yet. */
  jobLine(id) {
    const c = this.conns.get(id), j = this.jobs[0];
    if (!c || !j || !c.addr) return null;
    c.sent = j.id;
    return JSON.stringify({ method: "glados.pearl.job", params: { job: j.id, header: j.headerHex, bits: c.bits, k: K, rank: RANK, cert: 3 } }) + "\n";
  }

  /** One line from a miner. Answers { send: [lines], up: [lines], credited, close, log: [] }. */
  line(id, text) {
    const out = { send: [], up: [], credited: false, close: false, log: [] };
    const c = this.conns.get(id);
    if (!c) return out;
    let m;
    try { m = JSON.parse(text); } catch { return out; }
    if (!m || typeof m !== "object" || Array.isArray(m)) return out; // `null` parses too
    const rid = m.id ?? null;
    if (m.method === "glados.hello") {
      const head = String((m.params && m.params.worker) || "").split(".")[0];
      if (!/^0x[0-9a-fA-F]{40}$/.test(head)) { out.send.push(JSON.stringify({ id: rid, result: null, error: "the worker must be a 0x payout address" }) + "\n"); return out; }
      // The first greeting names whom this connection's work pays; a later one
      // cannot move it. A different wallet is a different connection.
      if (c.addr && c.addr !== head.toLowerCase()) { out.send.push(JSON.stringify({ id: rid, result: null, error: "this connection already mines for another address" }) + "\n"); return out; }
      c.addr = head.toLowerCase();
      out.send.push(reply(rid, { pearl: { k: K, rank: RANK, tile: TILE } }));
      const j = this.jobLine(id);
      if (j) out.send.push(j);
      return out;
    }
    if (m.method !== "glados.pearl.submit") return out;
    if (!c.addr) { out.send.push(verdict(rid, false, "invalid", "say glados.hello first")); return out; }
    // The ration, before anything costs CPU.
    const t = this.now();
    c.tokens = Math.min(SUBMIT_BURST, c.tokens + ((t - c.filled) / 1000) * SUBMIT_RATE);
    c.filled = t;
    if (c.tokens < 1) { this.stats.rejected++; out.send.push(verdict(rid, false, "rate", "too many submits; slow down")); return out; }
    c.tokens -= 1;
    const p = m.params || {};
    const job = this.jobs.find((j) => j.id === p.job);
    if (!job) { this.stats.rejected++; out.send.push(verdict(rid, false, "stale")); return out; }
    if (typeof p.proof !== "string" || p.proof.length > 200_000) { this.stats.rejected++; out.send.push(verdict(rid, false, "invalid", "no proof")); return out; }
    let bytes;
    try { bytes = fromBase64(p.proof); } catch { this.stats.rejected++; out.send.push(verdict(rid, false, "invalid", "proof is not base64")); return out; }

    const bits = c.bits;
    const bound = (1n << BigInt(256 - bits)) - 1n;
    const v = verifyShare(bytes, job.header, { bound });
    if (!v.ok) {
      this.stats.rejected++;
      const low = /above the target/.test(v.why);
      out.send.push(verdict(rid, false, low ? "low" : "invalid", v.why));
      // "low" can be honest -- a share found just before a harder job arrived;
      // a proof that fails verification cannot.
      if (!low && ++c.bad >= BAD_MIN && c.bad > c.good) { out.close = true; out.log.push(`[pearl] closing ${c.addr.slice(0, 10)}...: ${c.bad} invalid proofs`); }
      return out;
    }
    if (v.k !== K || v.rank !== RANK) { this.stats.rejected++; out.send.push(verdict(rid, false, "invalid", `this pool mines k=${K} rank=${RANK}`)); return out; }
    if (this.seen.has(v.jackpotHash)) { this.stats.rejected++; out.send.push(verdict(rid, false, "duplicate")); return out; }
    if (this.seen.size >= MAX_SEEN) this.seen.delete(this.seen.values().next().value); // oldest first
    this.seen.add(v.jackpotHash);
    c.good++;

    this.work.set(c.addr, (this.work.get(c.addr) || 0n) + (1n << BigInt(bits)));
    this.stats.accepted++;
    out.credited = true;
    out.send.push(verdict(rid, true, "accepted"));

    // Upstream only wants what meets its own bound; that share is the pool's pay.
    const h = BigInt("0x" + v.jackpotHash.match(/../g).reverse().join(""));
    if (h <= job.upBound) {
      const uid = this.nextUpId++;
      this.pendingUp.set(uid, { addr: c.addr, at: this.now() });
      if (this.pendingUp.size > 256) this.pendingUp.delete(this.pendingUp.keys().next().value);
      // Re-encoded in Pearl's current format: a miner may send the legacy one
      // (Pearl's gateway reads both), and upstream need not.
      const canonical = toBase64(encodeProof(decodeProof(bytes)));
      out.up.push(JSON.stringify({ id: uid, method: "mining.submit", params: { job_id: job.id, plain_proof: canonical, jackpot: v.jackpotHash, adjustment_factor: v.adjustmentFactor } }) + "\n");
      this.stats.forwarded++;
      out.log.push(`[pearl] ${c.addr.slice(0, 10)}... found a share for upstream`);
    }

    // VarDiff: an exponential average of the interval, then whole bits up or
    // down -- one bit is a factor of two, so this settles in a few shares.
    const dt = Math.max(0.05, (t - c.last) / 1000);
    c.last = t;
    c.ema = 0.7 * c.ema + 0.3 * dt;
    const want = Math.round(Math.log2(TARGET_S / c.ema));
    const next = Math.max(this.minBits, Math.min(this.maxBits(), c.bits + Math.max(-2, Math.min(2, want))));
    if (next !== c.bits) {
      c.bits = next;
      c.ema = TARGET_S;
      const j = this.jobLine(id);
      if (j) out.send.push(j);
    }
    return out;
  }

  // Never harder than upstream: a share that meets upstream's bound must also
  // count as ours, or the pool would forward work it did not credit.
  maxBits() {
    const j = this.jobs[0];
    if (!j) return 40;
    let b = 0;
    while (b < 255 && (1n << BigInt(256 - (b + 1))) - 1n >= j.upBound) b++;
    return Math.max(this.minBits, b);
  }

  /** Called periodically: a miner silent past twice its target gets an easier job. */
  idle(id) {
    const c = this.conns.get(id);
    if (!c || !c.addr) return null;
    if ((this.now() - c.last) / 1000 < 3 * TARGET_S || c.bits <= this.minBits) return null;
    c.bits = Math.max(this.minBits, c.bits - 1);
    c.last = this.now();
    return this.jobLine(id);
  }

  // ----------------------------------------------------------------- record
  save() { return { work: Object.fromEntries([...this.work].map(([a, w]) => [a, w.toString()])), stats: this.stats }; }
  load(s) {
    if (!s) return 0;
    for (const [a, w] of Object.entries(s.work || {})) if (/^0x[0-9a-f]{40}$/.test(a) && /^\d+$/.test(w)) this.work.set(a, BigInt(w));
    if (s.stats) Object.assign(this.stats, s.stats);
    return this.work.size;
  }
}
