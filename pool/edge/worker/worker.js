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
import { newKey, rvnAddress, evmAddress } from "./treasury.js";
import { parseLedger, workByAddress } from "./epoch.js";
import { readerData, readerDecode, READ_MAX } from "./reader.js";
import { textChunks, loadText } from "./chunks.js";
import { Treasury } from "./runner.js";
import wasmModule from "../target/wasm32-unknown-unknown/release/glados_edge.wasm";
import { bind, imports } from "../boundary.mjs";

// `server::JOB_PERIOD`. A quiet period is when a connection gets fresh work.
const JOB_PERIOD_MS = 30_000;
// `mine::stratum::MAX_LINE`. A longer line is a desynchronised peer.
const MAX_LINE = 131072;
// The holding gate: a miner must hold GATE_MIN whole GLADOS on chain 4663 to
// connect. "0" (or unset) turns it off. See `Pool.gate`.
const GLADOS = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777";
const PAIR_ADDR = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
// How long the gate gathers connections before one read answers them all.
const GATE_BATCH_MS = 300;
const RPC_4663 = "https://rpc.mainnet.chain.robinhood.com";
const GATE_CACHE_MS = 10 * 60_000;

// How often zpool's rates are re-read. Its figures move over minutes, and a
// switch costs every miner its current job, so faster buys nothing.
const RATES_MS = 5 * 60_000;

// ### Shards
//
// **Many objects, one record.** A single Durable Object is single-threaded and
// validates every share itself, so it is the ceiling on how many miners one pool
// can hold. `SHARDS` objects split the miners between them by a hash of the
// connecting address; each is today's pool unchanged -- its own core, its own
// upstream sessions, and so its own extranonce from zpool, which is what keeps
// shards from handing out the same search space.
//
// Shard 0 is named "main", the object every miner used before sharding, so the
// record it holds carries on rather than starting again.
//
// `/ledger.json` sums the shards through `Pool::merge_ledger`: tallies are
// plain counters, a worker seen on two shards is simply the sum, and each
// shard's digest is checked before its rows are added.
// The pool core's per-shard cap on (worker, coin) tallies (pool.rs MAX_TALLIES).
const MAX_TALLIES = 4096;

// A public read that fans out to every shard, answered from Cloudflare's edge
// cache for `seconds`: however often anybody asks, the shards are asked at
// most once a period per location.
async function cached(request, seconds, make) {
  const cache = typeof caches !== "undefined" ? caches.default : null;
  const key = new Request(new URL(request.url).toString(), { method: "GET" });
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return hit;
  }
  const res = await make();
  if (cache && res.ok) {
    const copy = new Response(res.clone().body, res);
    copy.headers.set("cache-control", `public, max-age=${seconds}`);
    await cache.put(key, copy);
  }
  return res;
}

function shardName(i, gpu = false) {
  if (gpu) return `gpu-${i}`;
  return i === 0 ? "main" : `shard-${i}`;
}

// Keyed by `?w=` when a miner sends one -- its payout address, say, so the
// same address always lands on the same shard -- and by the connecting IP
// otherwise. Either is correct: tallies sum across shards, so which shard a
// miner lands on changes load, never the record.
function shardFor(request, n) {
  const ip = new URL(request.url).searchParams.get("w") || request.headers.get("CF-Connecting-IP") || "";
  let h = 2166136261;
  for (let i = 0; i < ip.length; i++) h = Math.imul(h ^ ip.charCodeAt(i), 16777619) >>> 0;
  return n > 1 ? h % n : 0;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const n = Math.max(1, Number(env.SHARDS || 1));
    // GPU miners get shards of their own, with their own coin list, so CPU and
    // GPU switching are independent -- a card and a CPU never compete for one
    // slot choice, and neither drags the other onto an algorithm it hashes badly.
    const gpu = env.GPU_COINS ? Math.max(1, Number(env.GPU_SHARDS || 1)) : 0;
    const stub = (i, g = false) => env.POOL.get(env.POOL.idFromName(shardName(i, g)));
    const everyShard = () => [
      ...[...Array(n).keys()].map((i) => stub(i)),
      ...[...Array(gpu).keys()].map((i) => stub(i, true)),
    ];
    if (url.pathname === "/mine") {
      return stub(shardFor(request, n)).fetch(request);
    }
    // The treasury lives in shard 0 -- "main", the one object that existed
    // before sharding -- so there is exactly one place the keys are.
    if (url.pathname === "/treasury" || url.pathname === "/payouts.json") {
      return stub(0).fetch(request);
    }
    if (url.pathname === "/mine/gpu") {
      if (!gpu) return new Response("no GPU coins on this pool", { status: 404 });
      return stub(shardFor(request, gpu), true).fetch(request);
    }
    if (url.pathname === "/status") {
      return cached(request, 30, async () => {
        const all = await Promise.all(everyShard().map((s) => s.fetch(new Request(`${url.origin}/status`)).then((r) => r.json())));
        return Response.json({
          shards: n, gpuShards: gpu, slots: all[0].slots,
          connections: all.reduce((a, s) => a + s.connections, 0),
          perShard: all.map((s) => s.connections),
          // The fullest shard's tallies against the core's cap: past ~90%,
          // new miners' shares go uncredited and SHARDS should be raised.
          tallies: Math.max(0, ...all.map((s) => s.tallies || 0)), maxTallies: all[0].maxTallies ?? null,
        });
      });
    }
    if (url.pathname === "/ledger.json") {
      if (n === 1 && !gpu) return stub(0).fetch(request);
      // One shard's ledger, for recomputing a payout at any scale.
      const one = url.searchParams.get("shard");
      if (one !== null) {
        const names = [...[...Array(n).keys()].map((i) => shardName(i)), ...[...Array(gpu).keys()].map((i) => shardName(i, true))];
        if (!names.includes(one)) return new Response(`no shard ${one}; they are ${names.join(", ")}`, { status: 404 });
        return env.POOL.get(env.POOL.idFromName(one)).fetch(new Request(`${url.origin}/ledger.json`));
      }
      // **Past eight shards the merged ledger is an index.** Merging means
      // every shard's rows in one Worker's memory -- tens of megabytes at
      // 100,000 miners, past what a Worker has -- on every request, for anyone
      // who asks. The same rows are all there, one link a shard.
      if (n + gpu > 8) {
        const names = [...[...Array(n).keys()].map((i) => shardName(i)), ...[...Array(gpu).keys()].map((i) => shardName(i, true))];
        return Response.json({ shards: names.map((s) => `${url.origin}/ledger.json?shard=${s}`) }, { headers: { "access-control-allow-origin": "*" } });
      }
      return cached(request, 60, async () => {
      const docs = await Promise.all(everyShard().map((s) => s.fetch(new Request(`${url.origin}/ledger.json`)).then((r) => r.text())));
      const instance = await WebAssembly.instantiate(wasmModule, imports);
      const core = bind(instance);
      const coins = `${env.COINS || "yescrypt:yescrypt:12"} ${env.GPU_COINS || ""}`;
      core.init(coins.split(/\s+/).filter(Boolean), Number(env.SHARE_SECONDS || 45));
      for (const d of docs) core.mergeLedger(d);
      return new Response(core.ledger(1, Math.floor(Date.now() / 1000)), {
        headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
      });
      });
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
    this.gateMin = BigInt(env.GATE_MIN || "0") * 10n ** 18n;
    // Only shard 0 ("main") holds the keys and runs the treasury.
    this.treasurer = new Treasury(this);
    this.balances = new Map(); // address -> { wei, at }
    // Nothing is served until the core exists and the ledger is back, so a miner
    // arriving during a cold start cannot be credited into an empty record that
    // then overwrites the stored one.
    ctx.blockConcurrencyWhile(async () => {
      const instance = await WebAssembly.instantiate(wasmModule, imports);
      this.core = bind(instance);
      // Whitespace only: a coin with an upstream carries commas of its own
      // (`...@host:port,address,password`), and splitting on them tore one
      // coin into three that would not parse.
      // A shard's name says what it serves: `gpu-N` takes GPU_COINS, anything
      // else COINS. `ctx.id.name` is the name it was reached by (Cloudflare,
      // 2026-03, available in the constructor and under `wrangler dev`).
      this.gpu = String(ctx.id.name || "").startsWith("gpu-");
      const spec = this.gpu ? env.GPU_COINS : env.COINS;
      const coins = (spec || "yescrypt:yescrypt:12").split(/\s+/).filter(Boolean);
      this.slots = this.core.init(coins, Number(env.SHARE_SECONDS || 45), Number(env.WINDOW || 0));
      // **Restored, or the object refuses to serve.** A ledger that fails to load
      // and is then overwritten by the next job period is the share log erased
      // by a cold start, with nothing anywhere saying it happened. Throwing here
      // fails every request loudly instead, which is a problem somebody sees.
      const saved = await loadText(ctx.storage, "ledger");
      const restored = saved ? this.core.loadLedger(saved) : 0;
      // SWITCH=1: miners are shown one slot, the one that pays best per unit of
      // their work, chosen from zpool's live rates. Coin labels must be zpool's
      // algorithm names for the rates to find them.
      this.switching = env.SWITCH === "1";
      if (this.switching) this.core.switchOn();
      this.ratesAt = 0;
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
    // This shard's work per payout address, summed over workers and coins: what
    // the treasury needs from a shard, a few dozen bytes an address instead of
    // a ledger's hundred and thirty a (worker, coin) row. Internal: the
    // top-level router never sends a request here.
    if (url.pathname === "/work.json") {
      const work = workByAddress(parseLedger(this.core.ledger(1, Math.floor(Date.now() / 1000))));
      return Response.json({ work: Object.fromEntries([...work].map(([a, w]) => [a, w.toString()])) });
    }
    if (url.pathname === "/status") {
      return Response.json({ slots: this.slots, connections: this.conns.size, tallies: this.tallies ?? null, maxTallies: MAX_TALLIES });
    }
    if (url.pathname === "/treasury") {
      const t = await this.treasury();
      await this.arm();
      return Response.json({ rvn: t.rvn, evm: t.evm, ...(await this.treasurer.summary()) },
        { headers: { "access-control-allow-origin": "*" } });
    }
    if (url.pathname === "/payouts.json") {
      // Every epoch paid: who, how much work each did since the last one, the
      // floor, who was left out and why, and the transaction -- recomputable
      // with tools/distribute.py --split equal against the ledgers.
      const epochs = await this.treasurer.epochs();
      return Response.json({ epochs }, { headers: { "access-control-allow-origin": "*" } });
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
    // Lines wait here until the connection has passed the gate, so nothing a
    // miner sends reaches the core before its greeting has been checked.
    const gate = { passed: this.gateMin === 0n, queue: [] };
    const feed = (line) => deliver(this.core.line(id, line));
    server.addEventListener("message", async (ev) => {
      const text = typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data);
      if (text.length > MAX_LINE) {
        server.close(1009, "line too long");
        this.drop(server);
        return;
      }
      const lines = text.split("\n").filter((l) => l.trim());
      if (!gate.passed) {
        gate.queue.push(...lines);
        if (gate.checking) return;
        const hello = gate.queue.find((l) => l.includes("glados.hello"));
        if (!hello) return;
        gate.checking = true;
        const refusal = await this.gate(hello);
        if (refusal) {
          this.log(`[gate] ${peer} refused: ${refusal}`);
          let rid = 1;
          try { rid = JSON.parse(hello).id ?? 1; } catch {}
          try { server.send(JSON.stringify({ id: rid, result: null, error: refusal }) + "\n"); } catch {}
          server.close(1008, "holding requirement not met");
          this.drop(server);
          return;
        }
        gate.passed = true;
        const held = gate.queue.splice(0);
        for (const line of held) if (feed(line)) return;
        this.tickUps();
        return;
      }
      for (const line of lines) {
        if (feed(line)) return;
      }
      // A share may have beaten upstream's target: forward it now, not at the
      // next alarm, since upstream's job may be replaced before then.
      this.tickUps();
    });
    server.addEventListener("close", () => this.drop(server));
    server.addEventListener("error", () => this.drop(server));

    for (const u of this.ups) if (!u.running) this.runUp(u);
    this.ctx.waitUntil(this.refreshRates());
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
  // The pool's hot wallets, created on first use and never regenerated.
  //
  // **Generated here, kept here, never shown.** The operator agreed to the
  // automation holding keys on the condition that they are nobody's personal
  // keys and never pass through a conversation. So they are made from
  // `crypto.getRandomValues` inside this object and live only in its storage;
  // the one thing that leaves is the pair of addresses.
  //
  // **Never overwritten.** A key that is regenerated orphans whatever was paid
  // to the old address, silently and forever. So a read that fails throws
  // rather than falling through to "no keys yet", and new keys are written
  // only when storage positively answers that there are none.
  async treasury() {
    if (this.keys) return this.keys;
    // One at a time: two first requests racing would each generate keys, and
    // the second write would replace keys whose addresses were already shown.
    await this.ctx.blockConcurrencyWhile(async () => {
      if (this.keys) return;
      let k = await this.ctx.storage.get("treasury.keys");
      // **Pinned once published.** TREASURY_RVN and TREASURY_EVM name the
      // addresses the operator has told zpool and ChangeNOW about. With them
      // set, storage answering "no keys" is not a fresh start, it is keys lost
      // -- and generating new ones would send every later payout to addresses
      // nobody watches. So it refuses, and so does a mismatch.
      const pinRvn = this.env.TREASURY_RVN || "", pinEvm = (this.env.TREASURY_EVM || "").toLowerCase();
      if (k === undefined) {
        if (pinRvn || pinEvm) throw new Error("the treasury's keys are missing from storage while its addresses are pinned; refusing to make new ones");
        k = { rvnKey: newKey(), evmKey: newKey(), created: Date.now() };
        await this.ctx.storage.put("treasury.keys", k);
        this.log("[treasury] hot wallets created");
      }
      const keys = { ...k, rvn: rvnAddress(k.rvnKey), evm: evmAddress(k.evmKey) };
      if ((pinRvn && pinRvn !== keys.rvn) || (pinEvm && pinEvm !== keys.evm.toLowerCase())) {
        throw new Error(`the stored keys are ${keys.rvn} / ${keys.evm}, not the pinned ${pinRvn || "-"} / ${pinEvm || "-"}; refusing to use them`);
      }
      this.keys = keys;
    });
    return this.keys;
  }

  // **The holding gate, enforced where a miner connects.** The distributor has
  // a gate of its own at claim time, and the second review (audit-2 F1) showed
  // it can be shared: GLADOS moves between wallets untaxed, so one gate-sized
  // balance can be handed from claimant to claimant. The operator's decision
  // is to refuse at the door instead -- a miner whose payout address does not
  // hold the minimum is not mining here at all.
  //
  // Answers a refusal to send, or null. An unreadable balance admits: the gate
  // that pays (epoch.js, at build time) is the one that must not be walked
  // through, and it refuses an unreadable balance by carrying the work.
  async gate(hello) {
    let worker = "";
    try { worker = JSON.parse(hello).params.worker || ""; } catch {}
    const head = worker.split(".")[0];
    if (!/^0x[0-9a-fA-F]{40}$/.test(head)) {
      return `mine under your 0x address as the worker name (address.rig): the pool pays $GLaDOS there, and checks that it holds ${this.gateMin / 10n ** 18n} GLADOS`;
    }
    const addr = head.toLowerCase();
    let hit = this.balances.get(addr);
    if (!(hit && Date.now() - hit.at < GATE_CACHE_MS)) {
      try {
        hit = await this.readGate(addr);
      } catch (e) {
        // **Admitted, not refused, when the chain cannot be read.** This gate
        // only spares a miner hashing for nothing; the one that decides who is
        // paid runs again at every epoch, on the balance and code as they are
        // then. So an RPC outage refusing every miner at the door would be an
        // error message in every miner's face that protects nobody.
        this.log(`[gate] ${head} admitted unchecked: balance unreadable (${e && e.message ? e.message : e}); the payout checks it`);
        return null;
      }
    }
    const wei = hit.wei;
    // A contract is never paid (the payout would revert for everyone if it
    // were the pair), so it is refused at the door rather than mined for nothing.
    if (hit.contract || addr === PAIR_ADDR) {
      return `${head} is a contract; the pool pays only ordinary accounts, so mine under the address of a wallet you hold`;
    }
    if (wei < this.gateMin) {
      const have = wei / 10n ** 18n;
      return `${head} holds ${have} GLADOS on chain 4663 and mining here needs ${this.gateMin / 10n ** 18n}`;
    }
    return null;
  }

  // **Connections arriving together are checked together.** Every address that
  // asks within GATE_BATCH_MS goes into one GladosReader eth_call (up to 200),
  // so a crowd reconnecting after a deploy is a handful of requests to the
  // public RPC instead of one each -- which is what would get the pool's
  // Cloudflare egress rate-limited, and then every miner admitted unchecked.
  readGate(addr) {
    return new Promise((resolve, reject) => {
      this.gateWait = this.gateWait || new Map();
      const list = this.gateWait.get(addr) || [];
      list.push({ resolve, reject });
      this.gateWait.set(addr, list);
      if (this.gateWait.size >= READ_MAX) this.flushGate();
      else if (!this.gateTimer) this.gateTimer = setTimeout(() => this.flushGate(), GATE_BATCH_MS);
    });
  }

  async flushGate() {
    clearTimeout(this.gateTimer);
    this.gateTimer = null;
    const batch = this.gateWait || new Map();
    this.gateWait = new Map();
    const who = [...batch.keys()];
    if (!who.length) return;
    try {
      const res = await fetch(RPC_4663, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_call", params: [{ data: readerData(GLADOS, who) }, "latest"] }),
      });
      const j = await res.json();
      if (!j || !j.result) throw new Error(j && j.error ? j.error.message : `HTTP ${res.status}`);
      const m = readerDecode(who, j.result);
      for (const [a, waiters] of batch) {
        const v = m.get(a);
        if (v.balance === undefined) { for (const w of waiters) w.reject(new Error("balanceOf failed")); continue; }
        const hit = { wei: v.balance, at: Date.now(), contract: v.code === "contract" };
        this.balances.set(a, hit);
        for (const w of waiters) w.resolve(hit);
      }
    } catch (e) {
      for (const waiters of batch.values()) for (const w of waiters) w.reject(e);
    }
  }

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
      // Upstream work arriving can make a slot choosable that was not when the
      // rates were read -- the first connection's rates routinely land before
      // any upstream has answered. Re-decide against the rates already held; no
      // fetch, and silence unless the choice actually moved.
      if (this.switching && this.ratesDoc) {
        const again = this.core.rates(this.ratesDoc);
        for (const l of again.log) if (!l.startsWith("[switch] staying")) this.log(l);
      }
      this.pushWork();
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

  // **Chunked, because one value is capped at 128 KiB and a ledger is not.**
  // A row is about 130 bytes, so a single value held some 1,000 tallies --
  // about 250 miners switching across four coins -- and past that every save
  // was refused while the object kept serving from memory, until the next
  // deploy or eviction dropped everything since the last save that fitted.
  // Found by the load test. The head and its chunks go in one atomic put.
  save() {
    const text = this.core.ledger(1, Math.floor(Date.now() / 1000));
    this.tallies = (text.match(/"worker":/g) || []).length;
    if (this.tallies >= MAX_TALLIES * 0.9 && !this.warnedFull) {
      this.warnedFull = true;
      this.log(`[edge] ${this.tallies} of ${MAX_TALLIES} tallies: past the cap, new miners' shares are counted but credited to nobody -- raise SHARDS`);
    }
    return this.ctx.storage.put(textChunks("ledger", text));
  }

  async arm() {
    if ((await this.ctx.storage.getAlarm()) === null) {
      await this.ctx.storage.setAlarm(Date.now() + JOB_PERIOD_MS);
    }
  }

  // The treasury runs in "main" only, and its clock must not depend on anybody
  // mining: a payout that waits for the next miner to connect is a payout that
  // never comes on a quiet day.
  get runsTreasury() {
    return String(this.ctx.id.name || "") === "main" && (this.env.TREASURY || "off") !== "off";
  }

  // Every job period: fresh work for quiet miners, and the ledger to storage.
  // Every RATES_MS at most, while anyone is mining: fetch zpool's rates and let
  // the core re-choose. A failed fetch keeps the current choice; it is not a
  // reason to move every miner.
  async refreshRates() {
    if (!this.switching || Date.now() - this.ratesAt < RATES_MS) return;
    this.ratesAt = Date.now();
    try {
      const res = await fetch("https://zpool.ca/api/status", { headers: { "user-agent": "glados-pool" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.ratesDoc = await res.text();
      const r = this.core.rates(this.ratesDoc);
      for (const l of r.log) this.log(l);
      if (r.work) this.pushWork();
    } catch (e) {
      this.log(`[switch] rates unavailable, staying: ${e && e.message ? e.message : e}`);
    }
  }

  pushWork() {
    for (const [ws, id] of this.conns) {
      const w = this.core.work(id);
      try {
        for (const m of w.send) ws.send(m);
      } catch {
        this.drop(ws);
      }
    }
  }

  async alarm() {
    await this.refreshRates();
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
    if (this.runsTreasury) await this.treasurer.tick();
    if (this.conns.size > 0) await this.ctx.storage.setAlarm(Date.now() + JOB_PERIOD_MS);
    else if (this.runsTreasury) await this.ctx.storage.setAlarm(Date.now() + 5 * 60_000);
  }
}
