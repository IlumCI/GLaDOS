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
    this.up = { sock: null, writer: null, running: false };
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
        pearl: { hashrate: mac, ...this.room.stats, height: this.room.jobs[0] ? this.room.jobs[0].height : null },
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
        const out = this.room.line(id, line);
        for (const l of out.log) console.log(l);
        for (const s of out.send) { try { server.send(s); } catch {} }
        for (const u of out.up) this.upSend(u);
        if (out.credited) this.dirty = true;
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
    if (this.socks.size === 0) this.closeUp();
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
    if (this.up.writer) this.up.writer.write(new TextEncoder().encode(line)).catch(() => {});
    else console.log("[pearl] a share for upstream arrived with upstream down; it is lost");
  }

  async runUp() {
    this.up.running = true;
    let backoff = 2000;
    while (this.socks.size > 0) {
      try {
        this.up.sock = connect(this.env.PEARL_UPSTREAM);
        this.up.writer = this.up.sock.writable.getWriter();
        await this.up.writer.write(new TextEncoder().encode(this.room.authorize()));
        const reader = this.up.sock.readable.getReader(), dec = new TextDecoder();
        let buf = "";
        for (;;) {
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
            if (out.newJob) this.broadcast();
          }
          backoff = 2000;
        }
      } catch (e) {
        if (this.socks.size > 0) console.log(`[pearl up] ${e && e.message ? e.message : e}`);
      }
      this.closeUp();
      if (this.socks.size === 0) break;
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
    try { this.up.writer && this.up.writer.releaseLock(); } catch {}
    try { this.up.sock && this.up.sock.close(); } catch {}
    this.up.sock = null;
    this.up.writer = null;
  }
}
