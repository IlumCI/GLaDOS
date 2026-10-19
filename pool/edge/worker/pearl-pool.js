// The Pearl shard: miners at /mine/pearl, one upstream PRL pool, and the
// record of who did the work. Every rule is in pearl-room.js; this file moves
// lines between sockets and the room, keeps the record in storage, and answers
// the two reads the rest of the pool makes of a shard: /work.json for the
// treasury's epochs and /status for the pool bar.
//
// Configured by two variables, and off without either:
//   PEARL_UPSTREAM  host:port of the PRL pool (prl.2miners.com:1818)
//   TREASURY_PRL    the PRL address upstream pays -- a wallet the operator
//                   holds; PRL is bridged and swapped by hand from there
//
// Pearl work is credited in tiles, as expected hashes, the way every other
// shard credits its algorithm, and paid through the same epochs: the payout
// asset is the one each wallet chose, never PRL.
//
// Like the other shards it holds its upstream only while a miner is here, and
// writes its record when a miner leaves as well as on the alarm, so an evicted
// object costs at most one period of credit.

import { connect } from "cloudflare:sockets";
import { PearlRoom } from "./pearl-room.js";
import { rewardOf, DEFAULT as DEFAULT_REWARD } from "./rewards.js";

const TICK_MS = 15_000;
const IDLE_MAX_MS = 3 * 60_000; // as worker.js: a silent socket is a dead one
const MAX_LINE = 262_144;       // a share is ~90 KB of base64
// Shares that met upstream's bound while upstream was down: kept, briefly,
// and sent after the next authorize. They are the pool's pay, and an outage --
// a restart of this object's upstream, a blip -- should not cost them. ~90 KB
// each, so few; and old ones are dropped, since upstream will have moved on.
// A share written to upstream stays held until upstream answers it: a write
// into a connection that has just died succeeds locally and arrives nowhere,
// so whatever is unanswered when the connection goes is queued again.
const UP_QUEUE = 64, UP_QUEUE_MS = 60_000;

// The id of a line this pool wrote upstream, without parsing ~90 KB of proof.
function upId(line) {
  const m = /^\{"id":(\d+),/.exec(line);
  return m ? Number(m[1]) : null;
}

export class PearlPool {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    const bits = (v) => (v === undefined || v === "" ? undefined : Number(v));
    this.room = new PearlRoom({ wallet: env.TREASURY_PRL || "", worker: env.PEARL_WORKER || "glados",
      ...(bits(env.PEARL_MIN_BITS) !== undefined ? { minBits: bits(env.PEARL_MIN_BITS) } : {}),
      ...(bits(env.PEARL_START_BITS) !== undefined ? { startBits: bits(env.PEARL_START_BITS) } : {}) });
    this.socks = new Map();    // WebSocket -> connection id in the room
    this.lastSeen = new Map(); // WebSocket -> last inbound
    this.rate = new Map();     // WebSocket -> the miner's reported MAC/s
    this.nextId = 1;
    this.up = { sock: null, writer: null, running: false, authorized: false };
    this.upQueue = []; // { line, at }
    this.inFlight = new Map(); // upstream id -> { line, at }, written and not yet answered
    this.upCount = { sent: 0, answered: 0, queued: 0, lost: 0 };
    this.dirty = false;
    ctx.blockConcurrencyWhile(async () => {
      const n = this.room.load(await ctx.storage.get("pearl.record"));
      console.log(`[pearl] ${n} address(es) of work restored`);
    });
  }

  configured() { return Boolean(this.env.PEARL_UPSTREAM && this.env.TREASURY_PRL); }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/work.json") {
      const m = await this.rewards(), rewards = {};
      for (const a of this.room.work.keys()) if (m.has(a)) rewards[a] = m.get(a);
      return Response.json({ work: Object.fromEntries([...this.room.work].map(([a, w]) => [a, w.toString()])), rewards });
    }
    if (url.pathname === "/status") {
      let mac = 0;
      for (const v of this.rate.values()) mac += v;
      return Response.json({
        slots: 1, connections: this.socks.size, addresses: this.room.addresses().size,
        hashrate: 0, // other shards' figure is hashes of their algorithm; Pearl's is MACs, so it is kept apart
        pearl: { hashrate: mac, ...this.room.stats, ...this.upCount, waiting: this.upQueue.length, inFlight: this.inFlight.size, height: this.room.jobs[0] ? this.room.jobs[0].height : null },
      });
    }
    if (request.headers.get("Upgrade") !== "websocket") return new Response("expected a WebSocket upgrade", { status: 426 });
    if (!this.configured()) return new Response("Pearl mining is not configured on this pool", { status: 503 });

    const [client, server] = Object.values(new WebSocketPair());
    server.accept();
    const id = this.nextId++;
    this.socks.set(server, id);
    this.lastSeen.set(server, Date.now());
    this.room.open(id);

    server.addEventListener("message", (ev) => {
      const text = typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data);
      this.lastSeen.set(server, Date.now());
      if (text.length > MAX_LINE) { server.close(1009, "line too long"); this.drop(server); return; }
      for (const line of text.split("\n")) {
        if (!line.trim()) continue;
        if (line.includes("glados.rate")) { this.noteRate(server, line); continue; }
        if (line.includes("glados.hello")) this.noteReward(line).catch((e) => console.log(`[reward] not stored: ${e && e.message}`));
        // The room is tested never to throw; this is so that, if it ever does,
        // one bad line costs one line and not the connection's handler.
        let out;
        try { out = this.room.line(id, line); } catch (e) { console.log(`[pearl] a line the room could not take: ${e && e.message}`); continue; }
        for (const l of out.log) console.log(l);
        for (const s of out.send) { try { server.send(s); } catch {} }
        for (const u of out.up) this.upSend(u);
        if (out.credited) this.dirty = true;
        if (out.close) { try { server.close(1008, "too many invalid proofs"); } catch {} this.drop(server); return; }
      }
    });
    server.addEventListener("close", () => this.drop(server));
    server.addEventListener("error", () => this.drop(server));

    if (!this.up.running) this.ctx.waitUntil(this.runUp());
    if (!(await this.ctx.storage.getAlarm())) await this.ctx.storage.setAlarm(Date.now() + TICK_MS);
    return new Response(null, { status: 101, webSocket: client });
  }

  noteRate(ws, line) {
    try {
      const hs = Number((JSON.parse(line).params || {}).hs);
      this.rate.set(ws, Number.isFinite(hs) && hs > 0 ? Math.min(hs, 1e15) : 0); // MAC/s; a browser is ~1e12
    } catch {}
  }

  // A wallet's payout choice, stored as the main shard stores it, and handed
  // to the treasury beside the work in /work.json.
  async rewards() {
    if (!this.rewardMap) {
      const m = new Map();
      for (const [k, v] of await this.ctx.storage.list({ prefix: "reward." })) m.set(k.slice(7), v);
      this.rewardMap = m;
    }
    return this.rewardMap;
  }

  async noteReward(hello) {
    let worker = "", reward;
    try { const p = JSON.parse(hello).params || {}; worker = p.worker || ""; reward = p.reward; } catch { return; }
    const head = worker.split(".")[0];
    if (!/^0x[0-9a-fA-F]{40}$/.test(head)) return;
    const addr = head.toLowerCase(), code = rewardOf(reward), m = await this.rewards();
    if ((m.get(addr) || DEFAULT_REWARD) === code) return;
    if (code === DEFAULT_REWARD) { m.delete(addr); await this.ctx.storage.delete(`reward.${addr}`); }
    else { m.set(addr, code); await this.ctx.storage.put(`reward.${addr}`, code); }
  }

  async save() {
    if (!this.dirty) return;
    this.dirty = false;
    await this.ctx.storage.put("pearl.record", this.room.save());
  }

  drop(ws) {
    const id = this.socks.get(ws);
    if (id === undefined) return;
    this.socks.delete(ws);
    this.lastSeen.delete(ws);
    this.rate.delete(ws);
    this.room.close(id);
    this.ctx.waitUntil(this.save().catch((e) => console.log(`[pearl] record not saved, the alarm tries again: ${e && e.message}`)));
    if (this.socks.size === 0 && !this.upHeld()) this.closeUp();
  }

  async alarm() {
    const now = Date.now();
    for (const [ws, id] of this.socks) {
      if (now - (this.lastSeen.get(ws) || now) > IDLE_MAX_MS) { try { ws.close(1001, "idle"); } catch {} this.drop(ws); continue; }
      const easier = this.room.idle(id);
      if (easier) { try { ws.send(easier); } catch {} }
    }
    try { await this.save(); } catch (e) { console.log(`[pearl] save failed: ${e && e.message}`); }
    if (this.socks.size > 0) await this.ctx.storage.setAlarm(Date.now() + TICK_MS);
  }

  // ----------------------------------------------------------------- upstream
  upSend(line) {
    if (this.up.writer && this.up.authorized) {
      const at = Date.now();
      this.up.writer.write(new TextEncoder().encode(line)).catch(() => {});
      const id = upId(line);
      if (id !== null) {
        this.inFlight.set(id, { line, at });
        if (this.inFlight.size > 256) this.inFlight.delete(this.inFlight.keys().next().value); // an upstream that answers nothing
      }
      this.upCount.sent++;
      return;
    }
    this.upQueue.push({ line, at: Date.now() });
    this.upCount.queued++;
    if (this.upQueue.length > UP_QUEUE) { this.upQueue.shift(); this.upCount.lost++; console.log("[pearl] upstream queue full; the oldest waiting share is lost"); }
  }

  // Shares still owed to upstream and young enough to be worth sending.
  upHeld() {
    const now = Date.now();
    return this.inFlight.size > 0 || this.upQueue.some((q) => now - q.at <= UP_QUEUE_MS);
  }

  // After an authorize: everything still fresh enough goes out, oldest first.
  flushUp() {
    const now = Date.now(), waiting = this.upQueue.splice(0);
    let sent = 0;
    for (const { line, at } of waiting) {
      if (now - at > UP_QUEUE_MS) { this.upCount.lost++; continue; }
      this.upSend(line);
      sent++;
    }
    if (waiting.length) console.log(`[pearl up] ${sent} share(s) that waited for upstream sent, ${waiting.length - sent} too old`);
  }

  async runUp() {
    this.up.running = true;
    let backoff = 2000;
    while (this.socks.size > 0 || this.upHeld()) {
      try {
        this.up.sock = connect(this.env.PEARL_UPSTREAM);
        this.up.writer = this.up.sock.writable.getWriter();
        await this.up.writer.write(new TextEncoder().encode(this.room.authorize()));
        const reader = this.up.sock.readable.getReader(), dec = new TextDecoder();
        let buf = "";
        read: for (;;) {
          const { value, done } = await reader.read();
          if (done) { console.log("[pearl up] upstream closed the connection"); break; }
          buf += dec.decode(value, { stream: true });
          if (buf.length > 1 << 20) { console.log("[pearl up] runaway line from upstream"); break; }
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (!line) continue;
            const out = this.room.upLine(line);
            for (const l of out.log) console.log(l);
            // A job counts as a yes too: an upstream that sends work has taken us.
            if (!this.up.authorized && (out.authorized || out.newJob)) { this.up.authorized = true; this.flushUp(); }
            if (out.answered !== undefined) { this.inFlight.delete(out.answered); this.upCount.answered++; }
            if (out.newJob) this.broadcast();
            if (this.socks.size === 0 && !this.upHeld()) break read; // owed nothing, and no one to mine for
          }
          backoff = 2000;
        }
      } catch (e) {
        if (this.socks.size > 0 || this.upHeld()) console.log(`[pearl up] ${e && e.message ? e.message : e}`);
      }
      this.closeUp();
      if (this.socks.size === 0 && !this.upHeld()) break;
      await new Promise((r) => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 60_000);
    }
    this.up.running = false;
  }

  broadcast() {
    for (const [ws, id] of this.socks) {
      const j = this.room.jobLine(id);
      if (j) { try { ws.send(j); } catch {} }
    }
  }

  closeUp() {
    this.up.authorized = false;
    if (this.inFlight.size) {
      // Oldest first, ahead of anything queued since; they are not counted twice.
      const back = [...this.inFlight.values()];
      this.inFlight.clear();
      this.upCount.sent -= back.length;
      this.upQueue.unshift(...back);
      while (this.upQueue.length > UP_QUEUE) { this.upQueue.shift(); this.upCount.lost++; }
      console.log(`[pearl up] ${back.length} share(s) upstream never answered, queued again`);
    }
    try { this.up.writer && this.up.writer.releaseLock(); } catch {}
    try { this.up.sock && this.up.sock.close(); } catch {}
    this.up.sock = null;
    this.up.writer = null;
  }
}
