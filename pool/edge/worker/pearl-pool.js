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
import { MAINT_KEY, validMaint, notice, refusal, closeReason } from "./maintenance.js";

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
// **And they survive the object.** A restart -- a deploy rolling out, an
// eviction -- used to take every owed share with it: the first live test lost
// its one share for 2Miners exactly so. Each owed share is kept under its own
// key (one is ~90 KB, and a value holds 128 KiB), deleted once upstream answers
// it or it is too old to send, and restored oldest first when the object starts.
const OWED = "owed.";
// 2Miners sends a job every few seconds. An upstream that has said nothing for
// this long -- no answer to the authorize, no job -- is a dead line, open or not.
const UP_SILENCE_MS = 90_000;

// The id of a line this pool wrote upstream, without parsing ~90 KB of proof.
function upId(line) {
  const m = /^\{"id":(\d+),/.exec(line);
  return m ? Number(m[1]) : null;
}

// `p`, or `why` thrown once `ms` have passed without it.
function within(p, ms, why) {
  let t;
  return Promise.race([p, new Promise((_, no) => { t = setTimeout(() => no(new Error(why)), ms); })]).finally(() => clearTimeout(t));
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
    // `conn` is the one upstream connection the pool writes to: { sock, writer,
    // reader, authorized }. A loop reads only its own and stops once it is not
    // this one; `gen` retires a whole loop the alarm finds stuck.
    this.up = { conn: null, running: false, gen: 0, aliveAt: 0, lineAt: 0, reconnects: 0 };
    this.silenceMs = Number(env.PEARL_UP_SILENCE_MS) || UP_SILENCE_MS;
    this.upQueue = []; // { line, at }
    this.inFlight = new Map(); // upstream id -> { line, at }, written and not yet answered
    this.upCount = { sent: 0, answered: 0, queued: 0, lost: 0 };
    this.dirty = false;
    ctx.blockConcurrencyWhile(async () => {
      this.maint = validMaint(await ctx.storage.get(MAINT_KEY));
      this.maintAt = Date.now();
      const n = this.room.load(await ctx.storage.get("pearl.record"));
      console.log(`[pearl] ${n} address(es) of work restored`);
      const owed = await ctx.storage.list({ prefix: OWED });
      for (const [key, v] of owed) {
        if (!v || typeof v.line !== "string") continue;
        const id = upId(v.line);
        this.upQueue.push({ line: v.line, at: Number(v.at) || 0, key });
        // The room matches upstream's answers by id: it knows these again, and
        // never hands out one of their ids to a new share.
        if (id !== null) { this.room.pendingUp.set(id, { addr: "(restored)", at: Number(v.at) || 0 }); this.room.nextUpId = Math.max(this.room.nextUpId, id + 1); }
        this.owedSeq = Math.max(this.owedSeq || 0, Number(key.slice(OWED.length)) + 1);
      }
      if (this.upQueue.length) {
        console.log(`[pearl] ${this.upQueue.length} share(s) owed upstream restored; sending them`);
        await ctx.storage.setAlarm(Date.now() + 1000);
      }
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
    if (url.pathname === "/maintenance/apply" && request.method === "POST") {
      let s = null;
      try { s = validMaint(await request.json()); } catch {}
      if (!s) return new Response("not a maintenance state", { status: 400 });
      return Response.json(await this.applyMaint(s));
    }
    if (url.pathname === "/status") {
      let mac = 0;
      for (const v of this.rate.values()) mac += v;
      return Response.json({
        maintenance: this.maint && this.maint.on ? this.maint : undefined,
        slots: 1, connections: this.socks.size, addresses: this.room.addresses().size,
        hashrate: 0, // other shards' figure is hashes of their algorithm; Pearl's is MACs, so it is kept apart
        pearl: { hashrate: mac, ...this.room.stats, ...this.upCount, waiting: this.upQueue.length, inFlight: this.inFlight.size, height: this.room.jobs[0] ? this.room.jobs[0].height : null,
          upstream: this.upState(), upLineAgoS: this.up.lineAt ? Math.round((Date.now() - this.up.lineAt) / 1000) : null, reconnects: this.up.reconnects },
      });
    }
    if (request.headers.get("Upgrade") !== "websocket") return new Response("expected a WebSocket upgrade", { status: 426 });
    if (!this.configured()) return new Response("Pearl mining is not configured on this pool", { status: 503 });
    if (this.maint && this.maint.on && (await this.stillDown())) return refusal(this.maint, Date.now());

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
    // A loop that has not stepped in three silences is stuck somewhere no
    // timer of its own reaches: retired, and a fresh one started.
    if (this.up.running && now - this.up.aliveAt > 3 * this.silenceMs) {
      console.log(`[pearl up] the upstream loop has not moved in ${Math.round((now - this.up.aliveAt) / 1000)} s; starting it again`);
      this.up.gen++;
      this.up.running = false;
      this.closeUp();
    }
    // Shares owed with nobody connected (a restart, say): the upstream opens for them alone.
    if (!this.up.running && (this.socks.size > 0 || this.upHeld())) this.ctx.waitUntil(this.runUp());
    if (this.socks.size > 0 || this.upHeld()) await this.ctx.storage.setAlarm(Date.now() + TICK_MS);
  }

  // The storage side of the owed shares. Failures are logged and left: the
  // share is still held in memory and goes out regardless.
  owe(entry) {
    if (entry.key || entry.line.length > 120_000) return;
    entry.key = OWED + String(this.owedSeq = (this.owedSeq || 0) + 1).padStart(12, "0");
    this.ctx.waitUntil(this.ctx.storage.put(entry.key, { line: entry.line, at: entry.at }).catch((e) => console.log(`[pearl] an owed share was not stored: ${e && e.message}`)));
  }
  paid(entry) {
    if (!entry || !entry.key) return;
    this.ctx.waitUntil(this.ctx.storage.delete(entry.key).catch(() => {}));
  }

  // -------------------------------------------------------------- maintenance
  // As worker.js's: tell, drain, close; and shares owed upstream still go --
  // runUp keeps the upstream open after the last miner while upHeld() says so.
  async applyMaint(s) {
    const was = this.maint && this.maint.on;
    this.maint = s;
    this.maintAt = Date.now();
    await this.ctx.storage.put(MAINT_KEY, s);
    const connections = this.socks.size;
    if (!s.on) {
      if (was) console.log(`[pearl maint] off; ${connections} still connected`);
      return { connections, draining: false };
    }
    console.log(`[pearl maint] on: ${s.reason}; ${connections} miner(s) told, closing in ${s.drainMs / 1000} s`);
    const line = notice(s, Date.now());
    for (const [ws] of this.socks) { try { ws.send(line); } catch { this.drop(ws); } }
    const drainId = (this.drainId || 0) + 1;
    this.drainId = drainId;
    this.ctx.waitUntil(this.drain(drainId, s.drainMs));
    return { connections, draining: true, owedUpstream: this.upQueue.length + this.inFlight.size };
  }

  async drain(drainId, ms) {
    if (ms > 0) await new Promise((r) => setTimeout(r, ms));
    if (this.drainId !== drainId || !(this.maint && this.maint.on)) return;
    const why = closeReason(this.maint);
    let n = 0;
    for (const [ws] of this.socks) { try { ws.close(1013, why); } catch {} this.drop(ws); n++; }
    try { await this.save(); } catch (e) { console.log(`[pearl maint] record not saved on the way down: ${e && e.message}`); }
    console.log(`[pearl maint] drained: ${n} miner(s) closed; ${this.upQueue.length + this.inFlight.size} share(s) still owed upstream`);
  }

  async stillDown() {
    if (Date.now() - (this.maintAt || 0) < 30_000 || !this.env.POOL) return true;
    this.maintAt = Date.now();
    try {
      const s = validMaint(await (await this.env.POOL.get(this.env.POOL.idFromName("control")).fetch(new Request("https://control/maintenance"))).json());
      if (s && !s.on) { this.maint = s; await this.ctx.storage.put(MAINT_KEY, s); return false; }
    } catch (e) { console.log(`[pearl maint] control unreadable, staying down: ${e && e.message}`); }
    return true;
  }

  // ----------------------------------------------------------------- upstream
  // `entry` is a share already owed (its storage key carried along); a bare
  // line is a new one, stored before anything else happens to it.
  upSend(line, entry = null) {
    const e = entry || { line, at: Date.now() };
    this.owe(e);
    const conn = this.up.conn;
    if (conn && conn.authorized) {
      conn.writer.write(new TextEncoder().encode(e.line)).catch(() => {});
      const id = upId(e.line);
      if (id !== null) {
        // `at` stays when it was found: a share sent again is no younger.
        this.inFlight.set(id, { line: e.line, at: e.at, key: e.key });
        if (this.inFlight.size > 256) { const [, old] = this.inFlight.entries().next().value; this.inFlight.delete(this.inFlight.keys().next().value); this.paid(old); } // an upstream that answers nothing
      } else this.paid(e); // no id to be answered by: sent is all it can be
      this.upCount.sent++;
      return;
    }
    this.upQueue.push(e);
    if (!entry) this.upCount.queued++;
    if (this.upQueue.length > UP_QUEUE) { this.paid(this.upQueue.shift()); this.upCount.lost++; console.log("[pearl] upstream queue full; the oldest waiting share is lost"); }
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
    for (const e of waiting) {
      if (now - e.at > UP_QUEUE_MS) { this.upCount.lost++; this.paid(e); continue; }
      this.upSend(e.line, e);
      sent++;
    }
    if (waiting.length) console.log(`[pearl up] ${sent} share(s) that waited for upstream sent, ${waiting.length - sent} too old`);
  }

  upState() {
    const c = this.up.conn;
    return c ? (c.authorized ? "authorized" : "connecting") : this.up.running ? "reconnecting" : "closed";
  }

  async runUp() {
    const gen = ++this.up.gen;
    this.up.running = true;
    this.up.aliveAt = Date.now();
    let backoff = 2000;
    while (this.up.gen === gen && (this.socks.size > 0 || this.upHeld())) {
      const conn = { sock: null, writer: null, reader: null, authorized: false };
      this.up.conn = conn;
      try {
        conn.sock = connect(this.env.PEARL_UPSTREAM);
        conn.writer = conn.sock.writable.getWriter();
        conn.reader = conn.sock.readable.getReader();
        await within(conn.writer.write(new TextEncoder().encode(this.room.authorize())), this.silenceMs, "upstream did not take the authorize");
        const dec = new TextDecoder();
        let buf = "";
        read: for (;;) {
          const { value, done } = await within(conn.reader.read(), this.silenceMs, `upstream silent for ${Math.round(this.silenceMs / 1000)} s`);
          // Closed, or replaced, while this read waited: not this loop's to go on with.
          if (this.up.conn !== conn) break;
          if (done) { console.log("[pearl up] upstream closed the connection"); break; }
          this.up.aliveAt = this.up.lineAt = Date.now();
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
            if (!conn.authorized && (out.authorized || out.newJob)) { conn.authorized = true; this.flushUp(); }
            if (out.answered !== undefined) { this.paid(this.inFlight.get(out.answered)); this.inFlight.delete(out.answered); this.upCount.answered++; }
            if (out.newJob) this.broadcast();
            if (this.socks.size === 0 && !this.upHeld()) break read; // owed nothing, and no one to mine for
          }
          backoff = 2000;
        }
      } catch (e) {
        if (this.socks.size > 0 || this.upHeld()) console.log(`[pearl up] ${e && e.message ? e.message : e}`);
      }
      this.closeUp(conn);
      if (this.up.gen !== gen || (this.socks.size === 0 && !this.upHeld())) break;
      this.up.reconnects++;
      await new Promise((r) => setTimeout(r, backoff));
      this.up.aliveAt = Date.now();
      backoff = Math.min(backoff * 2, 60_000);
    }
    if (this.up.gen === gen) this.up.running = false;
  }

  broadcast() {
    for (const [ws, id] of this.socks) {
      const j = this.room.jobLine(id);
      if (j) { try { ws.send(j); } catch {} }
    }
  }

  // Close `conn` (the current connection when none is named). Only the
  // current one's unanswered shares go back in the queue: they were written
  // to it. Its reader is cancelled -- a close() ends our side only, and an
  // upstream that goes on talking must not keep a loop reading it.
  closeUp(conn = this.up.conn) {
    if (!conn) return;
    if (this.up.conn === conn) {
      this.up.conn = null;
      if (this.inFlight.size) {
        // Oldest first, ahead of anything queued since; they are not counted twice.
        const back = [...this.inFlight.values()];
        this.inFlight.clear();
        this.upCount.sent -= back.length;
        this.upQueue.unshift(...back);
        while (this.upQueue.length > UP_QUEUE) { this.paid(this.upQueue.shift()); this.upCount.lost++; }
        console.log(`[pearl up] ${back.length} share(s) upstream never answered, queued again`);
      }
    }
    conn.authorized = false;
    try { conn.reader && conn.reader.cancel().catch(() => {}); } catch {}
    try { conn.writer && conn.writer.releaseLock(); } catch {}
    try { conn.sock && conn.sock.close(); } catch {}
  }
}
