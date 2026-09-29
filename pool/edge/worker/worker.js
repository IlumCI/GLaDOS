// The GLaDOS pool, served from a Cloudflare Durable Object.
//
// **Nothing in this file decides anything about a miner.** Every rule -- one
// worker per connection, rate limits, the bad-share cutoff, VarDiff, validation
// itself -- runs in the WebAssembly build of `pool/src`, the same source the TCP
// daemon runs. This file is the other half of `server.rs`: it moves lines between
// a socket and the core, keeps the share log in storage, and wakes the core when a
// miner has gone quiet. `harness.mjs` does the same job over TCP, which is how
// this core was tested before it was ever deployed.
//
// ### One object, and why it stays awake
//
// Every miner reaches the same object (`idFromName("main")`), because a pool is
// one ledger and one job counter; splitting it would split the record. Sockets are
// accepted with the ordinary `accept()` rather than the hibernation API, and that
// is a decision with a price: a hibernated object loses its memory, and the core's
// memory is the pool -- the job ring that tells a fresh share from a stale one,
// each miner's difficulty, the duplicates already seen. Rebuilding all of that on
// every wake would be a second persistence layer to get wrong. So the object stays
// resident while anyone is connected, which Cloudflare bills as duration, and the
// ledger is written to storage every job period so an eviction costs at most one.
//
// ### The protocol
//
// ### Upstream
//
// Work comes from zpool over an outbound `connect()`, driven through the same
// `Upstream` state machine the native daemon runs from a thread. **It is open
// only while a miner is**: an open socket keeps the object resident and billed,
// and with nobody hashing there is nothing to forward and nobody to hand work
// to. The first miner opens it and the last one closes it, so a connection's
// first job can wait on an upstream handshake -- about a second -- and then
// arrives the moment upstream sends one rather than at the next idle period.
//
// Miners speak the native protocol (`mine::proto`), one JSON message per line.
// A WebSocket message already has edges, but a sender may batch several lines into
// one, so messages are still split on newlines -- the core is handed exactly one
// line at a time, as `server.rs` hands it one.

import { connect } from "cloudflare:sockets";
import wasmModule from "../target/wasm32-unknown-unknown/release/glados_edge.wasm";
import { bind, imports } from "../boundary.mjs";

// `server::JOB_PERIOD`. A quiet period is when a connection gets fresh work.
const JOB_PERIOD_MS = 30_000;
// `mine::stratum::MAX_LINE`. A longer line is a desynchronised peer.
const MAX_LINE = 131072;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const pool = env.POOL.get(env.POOL.idFromName("main"));
    if (url.pathname === "/mine" || url.pathname === "/ledger.json" || url.pathname === "/status") {
      return pool.fetch(request);
    }
    return new Response(
      "GLaDOS pool. Miners connect over WebSocket at /mine and speak the native " +
        "protocol; the share log is at /ledger.json.\n",
      { headers: { "content-type": "text/plain; charset=utf-8" } },
    );
  },
};

export class Pool {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    this.conns = new Map(); // WebSocket -> connection id in the core
    this.core = null;
    this.ups = [];          // { i, where, sock, writer, running }
    // Nothing is served until the core exists and the ledger is back, so a miner
    // arriving during a cold start cannot be credited into an empty record that
    // then overwrites the stored one.
    ctx.blockConcurrencyWhile(async () => {
      const instance = await WebAssembly.instantiate(wasmModule, imports);
      this.core = bind(instance);
      // Whitespace only: a coin with an upstream carries commas of its own
      // (`...@host:port,address,password`), and splitting on them tore one
      // coin into three that would not parse.
      const coins = (env.COINS || "yescrypt:yescrypt:12").split(/\s+/).filter(Boolean);
      this.slots = this.core.init(coins, Number(env.SHARE_SECONDS || 45), Number(env.WINDOW || 0));
      // **Restored, or the object refuses to serve.** A ledger that fails to load
      // and is then overwritten by the next job period is the share log erased
      // by a cold start, with nothing anywhere saying it happened. Throwing here
      // fails every request loudly instead, which is a problem somebody sees.
      const saved = await ctx.storage.get("ledger");
      const restored = saved ? this.core.loadLedger(saved) : 0;
      for (let i = 0; i < this.core.upCount(); i++) {
        this.ups.push({ i, where: this.core.upWhere(i), sock: null, writer: null, running: false });
      }
      // Coin specs carry the upstream address and password, so they are logged
      // by label only.
      const labels = coins.map((c) => c.split("@")[0] + (c.includes("@") ? " (upstream)" : " (local)"));
      this.log(`[edge] ${this.slots} slot(s): ${labels.join(" ")}; ${saved ? `${restored} record(s) restored` : "no stored ledger, starting from zero"}`);
    });
  }

  log(line) {
    console.log(line);
  }

  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/ledger.json") {
      return new Response(this.core.ledger(1, Math.floor(Date.now() / 1000)), {
        headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
      });
    }
    if (url.pathname === "/status") {
      return Response.json({ slots: this.slots, connections: this.conns.size });
    }
    if (request.headers.get("Upgrade") !== "websocket") {
      return new Response("expected a WebSocket upgrade", { status: 426 });
    }

    const [client, server] = Object.values(new WebSocketPair());
    server.accept();
    const peer = request.headers.get("CF-Connecting-IP") || "?";
    const id = this.core.open(peer);
    this.conns.set(server, id);

    const deliver = (r) => {
      for (const l of r.log) this.log(l);
      for (const m of r.send) server.send(m);
      if (r.close) {
        server.close(1000, "closed by the pool");
        this.drop(server);
      }
      return r.close;
    };
    server.addEventListener("message", (ev) => {
      const text = typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data);
      if (text.length > MAX_LINE) {
        server.close(1009, "line too long");
        this.drop(server);
        return;
      }
      for (const line of text.split("\n")) {
        if (line.trim() && deliver(this.core.line(id, line))) return;
      }
      // A share may have beaten upstream's target: forward it now, not at the
      // next alarm, since upstream's job may be replaced before then.
      this.tickUps();
    });
    server.addEventListener("close", () => this.drop(server));
    server.addEventListener("error", () => this.drop(server));

    for (const u of this.ups) if (!u.running) this.runUp(u);
    await this.arm();
    return new Response(null, { status: 101, webSocket: client });
  }

  // **The ledger is written when a miner leaves, not only on the alarm.** An
  // object with no connections is evicted after a few idle seconds, and its
  // memory with it; the alarm then wakes a *fresh* object that restores the
  // last save and writes that back. So every share since the previous period
  // was lost whenever a miner disconnected more than a few seconds before it --
  // measured: a four-minute run's shares absent from the tally while the next
  // run's, closing six seconds before its alarm, survived.
  drop(ws) {
    const id = this.conns.get(ws);
    if (id !== undefined) {
      this.core.close(id);
      this.conns.delete(ws);
      this.ctx.waitUntil(this.save());
      if (this.conns.size === 0) {
        for (const u of this.ups) this.closeUp(u);
      }
    }
  }

  // One upstream connection, reconnecting with backoff while anyone is mining.
  async runUp(u) {
    u.running = true;
    let backoff = 2000;
    while (this.conns.size > 0) {
      try {
        u.sock = connect(u.where);
        u.writer = u.sock.writable.getWriter();
        if (!this.upDeliver(u, this.core.upOpen(u.i))) {
          const reader = u.sock.readable.getReader();
          for (;;) {
            const { value, done } = await reader.read();
            if (done) {
              this.log(`[up ${u.i}] upstream closed the connection`);
              break;
            }
            if (this.upDeliver(u, this.core.upBytes(u.i, value))) break;
            backoff = 2000;
          }
        }
      } catch (e) {
        // The last miner leaving closes the socket under the read, which
        // throws; that is the ordinary way out, not a fault worth a line.
        if (this.conns.size > 0) this.log(`[up ${u.i}] ${e && e.message ? e.message : e}`);
      }
      this.closeUp(u);
      if (this.conns.size === 0) break;
      await new Promise((r) => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 60_000);
    }
    u.running = false;
  }

  // Act on what the core answered. `true` when the connection is finished.
  upDeliver(u, r) {
    for (const l of r.log) this.log(l);
    for (const m of r.send) {
      if (u.writer) u.writer.write(new TextEncoder().encode(m)).catch(() => {});
    }
    if (r.work) {
      for (const [ws, id] of this.conns) {
        const w = this.core.work(id);
        try {
          for (const m of w.send) ws.send(m);
        } catch {
          this.drop(ws);
        }
      }
    }
    if (r.close) this.log(`[up ${u.i}] ${r.why}`);
    return r.close;
  }

  tickUps() {
    for (const u of this.ups) {
      if (u.writer && this.upDeliver(u, this.core.upTick(u.i))) this.closeUp(u);
    }
  }

  closeUp(u) {
    try { u.writer && u.writer.releaseLock(); } catch {}
    try { u.sock && u.sock.close(); } catch {}
    u.sock = null;
    u.writer = null;
  }

  save() {
    return this.ctx.storage.put("ledger", this.core.ledger(1, Math.floor(Date.now() / 1000)));
  }

  async arm() {
    if ((await this.ctx.storage.getAlarm()) === null) {
      await this.ctx.storage.setAlarm(Date.now() + JOB_PERIOD_MS);
    }
  }

  // Every job period: fresh work for quiet miners, and the ledger to storage.
  async alarm() {
    this.tickUps();
    for (const [ws, id] of this.conns) {
      const r = this.core.idle(id);
      for (const l of r.log) this.log(l);
      try {
        for (const m of r.send) ws.send(m);
      } catch {
        this.drop(ws);
      }
    }
    await this.save();
    if (this.conns.size > 0) await this.ctx.storage.setAlarm(Date.now() + JOB_PERIOD_MS);
  }
}
