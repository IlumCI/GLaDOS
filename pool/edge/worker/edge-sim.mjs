// A Durable Object runtime small enough to read, for testing worker.js and
// pearl-pool.js under node -- and for breaking them on purpose.
//
// What it models, because the pool's damage control depends on it:
//   - storage that survives an object and a value cap of 128 KiB, as Cloudflare's
//   - an object's memory, which eviction throws away; the next request builds a
//     fresh object from storage, running the constructor again
//   - blockConcurrencyWhile: no request reaches an object before its constructor's
//     block settles, and a block that throws resets the object and fails the call.
//     After construction too: Cloudflare terminates an object whose
//     blockConcurrencyWhile callback throws, and every socket it held drops
//   - alarms: one per object, run when the test says
//   - WebSocket pairs, both ends, with close codes and reasons
//   - faults: any storage operation, or any call into a named object, can be made
//     to throw whatever the test likes, once or until cleared
//
// What it does not model: time (the test drives alarms and passes its own
// clock where the code takes one), CPU limits, or the network.
import { register } from "node:module";

register("./edge-loader.mjs", import.meta.url);

export const VALUE_MAX = 128 * 1024;

// --------------------------------------------------------------- Response
// node refuses status 101; workerd uses it for every WebSocket upgrade.
const NodeResponse = globalThis.Response;
class EdgeResponse extends NodeResponse {
  constructor(body, init = {}) {
    if (init && init.status === 101) {
      super(null, { status: 200 });
      this.edgeStatus = 101;
      this.webSocket = init.webSocket;
    } else super(body, init);
  }
  get status() { return this.edgeStatus ?? super.status; }
  get ok() { return this.edgeStatus ? false : super.ok; }
  static json(data, init = {}) {
    const headers = new Headers(init.headers || {});
    if (!headers.has("content-type")) headers.set("content-type", "application/json");
    return new EdgeResponse(JSON.stringify(data), { ...init, headers });
  }
}
globalThis.Response = EdgeResponse;

// --------------------------------------------------------------- WebSockets
export class FakeSocket {
  constructor() {
    this.readyState = 1;
    this.listeners = { message: [], close: [], error: [] };
    this.received = []; // every message this end was sent, as sent
    this.closed = null; // { code, reason } once closed
    this.peer = null;
  }
  accept() {}
  addEventListener(kind, f) { (this.listeners[kind] = this.listeners[kind] || []).push(f); }
  send(data) {
    if (this.readyState !== 1) throw new Error("WebSocket is not open");
    const p = this.peer;
    p.received.push(data);
    for (const f of p.listeners.message) f({ data });
  }
  close(code = 1000, reason = "") {
    if (this.readyState === 3) return;
    this.readyState = 3;
    this.closed = { code, reason, by: "self" };
    const p = this.peer;
    if (p.readyState !== 3) {
      p.readyState = 3;
      p.closed = { code, reason, by: "peer" };
      for (const f of p.listeners.close) f({ code, reason });
    }
    for (const f of this.listeners.close) f({ code, reason });
  }
  // A drop with no close frame: what a sleeping laptop does. The pool's end is
  // left OPEN and hears nothing, which is exactly the case reap() exists for.
  vanish() { this.readyState = 3; this.closed = { code: 1006, reason: "vanished", by: "self" }; }
  // Every newline-separated JSON line received, parsed where it parses.
  lines() {
    return this.received.flatMap((m) => String(m).split("\n")).filter((l) => l.trim()).map((l) => { try { return JSON.parse(l); } catch { return { unparsed: l }; } });
  }
}
globalThis.WebSocketPair = class {
  constructor() {
    const a = new FakeSocket(), b = new FakeSocket();
    a.peer = b; b.peer = a;
    this[0] = a; this[1] = b;
  }
};

// --------------------------------------------------------------- storage
export class FakeStorage {
  constructor(name) {
    this.name = name;
    this.data = new Map();
    this.alarm = null;
    this.faults = []; // { op, match(key) -> bool, error, times }
    this.ops = [];    // a log of every operation, for tests that count
  }
  fail(op, error, { times = Infinity, match = () => true } = {}) { this.faults.push({ op, error, times, match }); }
  heal() { this.faults = []; }
  check(op, key) {
    this.ops.push([op, key]);
    for (const f of this.faults) {
      if ((f.op === op || f.op === "*") && f.times > 0 && f.match(key)) {
        f.times--;
        throw typeof f.error === "function" ? f.error(op, key) : f.error;
      }
    }
  }
  static size(v) {
    if (typeof v === "string") return v.length;
    try { return JSON.stringify(v, (_, x) => (typeof x === "bigint" ? x.toString() : x)).length; } catch { return 0; }
  }
  async get(key) {
    this.check("get", key);
    if (Array.isArray(key)) { const m = new Map(); for (const k of key) if (this.data.has(k)) m.set(k, structuredClone(this.data.get(k))); return m; }
    return this.data.has(key) ? structuredClone(this.data.get(key)) : undefined;
  }
  async put(key, value) {
    const entries = typeof key === "object" && key !== null ? Object.entries(key) : [[key, value]];
    this.check("put", entries.map(([k]) => k).join(","));
    for (const [k, v] of entries) if (FakeStorage.size(v) > VALUE_MAX) throw new RangeError(`Values cannot be larger than ${VALUE_MAX} bytes (${k})`);
    // One put is atomic: every key or none.
    for (const [k, v] of entries) this.data.set(k, structuredClone(v));
  }
  async delete(key) {
    this.check("delete", key);
    if (Array.isArray(key)) { let n = 0; for (const k of key) n += this.data.delete(k) ? 1 : 0; return n; }
    return this.data.delete(key);
  }
  async list({ prefix = "" } = {}) {
    this.check("list", prefix);
    const m = new Map();
    for (const k of [...this.data.keys()].sort()) if (k.startsWith(prefix)) m.set(k, structuredClone(this.data.get(k)));
    return m;
  }
  async getAlarm() { this.check("getAlarm", ""); return this.alarm; }
  async setAlarm(t) { this.check("setAlarm", ""); this.alarm = Number(t); }
  async deleteAlarm() { this.check("deleteAlarm", ""); this.alarm = null; }
}

// --------------------------------------------------------------- objects
// One namespace: objects by name, each with storage that outlives it.
export class Namespace {
  constructor(Class, env) {
    this.Class = Class;
    this.env = env;
    this.storages = new Map();
    this.live = new Map();   // name -> { obj, ready, waits }
    this.callFaults = [];    // { name, error, times }
    this.calls = [];         // names called, in order
    this.created = new Map(); // name -> how many times constructed
    this.resets = new Map();  // name -> how many times reset by a throw
  }
  // What Cloudflare does to an object that broke: memory gone, sockets dropped (1006).
  reset(name, why) {
    const o = this.live.get(name);
    if (!o) return;
    this.live.delete(name);
    this.resets.set(name, (this.resets.get(name) || 0) + 1);
    const held = [...((o.obj && o.obj.conns && o.obj.conns.keys()) || []), ...((o.obj && o.obj.socks && o.obj.socks.keys()) || [])];
    for (const ws of held) {
      if (ws.readyState === 3) continue;
      ws.readyState = 3;
      const p = ws.peer;
      if (p && p.readyState !== 3) { p.readyState = 3; p.closed = { code: 1006, reason: `object reset: ${why}`, by: "peer" }; for (const f of p.listeners.close) f({ code: 1006, reason: "" }); }
    }
  }
  idFromName(name) { return { name, toString: () => `id:${name}` }; }
  storage(name) {
    if (!this.storages.has(name)) this.storages.set(name, new FakeStorage(name));
    return this.storages.get(name);
  }
  instance(name) {
    let o = this.live.get(name);
    if (o) return o;
    const waits = [];
    const ctx = {
      id: { name, toString: () => `id:${name}` },
      storage: this.storage(name),
      blockConcurrencyWhile: (f) => {
        const p = Promise.resolve().then(f);
        o.ready = o.ready.then(() => p);
        o.ready.catch(() => {}); // seen by whoever awaits it; not an unhandled rejection meanwhile
        if (o.built) p.catch((e) => this.reset(name, (e && e.message) || e));
        return p;
      },
      waitUntil: (p) => { waits.push(Promise.resolve(p).catch(() => {})); },
    };
    o = { obj: null, ready: Promise.resolve(), waits, ctx };
    this.live.set(name, o);
    this.created.set(name, (this.created.get(name) || 0) + 1);
    o.obj = new this.Class(ctx, this.env);
    o.built = true;
    return o;
  }
  // Fails calls into `name` ("*" for any) with `error`, `times` times.
  failCalls(name, error, times = Infinity) { this.callFaults.push({ name, error, times }); }
  healCalls() { this.callFaults = []; }
  get(id) {
    const name = id.name;
    return {
      fetch: async (request) => {
        this.calls.push(name);
        for (const f of this.callFaults) {
          if ((f.name === name || f.name === "*") && f.times > 0) { f.times--; throw typeof f.error === "function" ? f.error() : f.error; }
        }
        const o = this.instance(name);
        try { await o.ready; }
        catch (e) { this.live.delete(name); throw e; } // a constructor that refused: the object is reset
        return o.obj.fetch(request);
      },
    };
  }
  // Throws the object's memory away; its storage stays.
  evict(name) { this.live.delete(name); }
  async settle(name) {
    const o = this.live.get(name);
    if (!o) return;
    for (let i = 0; i < 5; i++) { const w = o.waits.splice(0); if (!w.length) break; await Promise.all(w); }
  }
  // Runs the alarm, as Cloudflare would: on the live object, or a fresh one.
  async alarm(name) {
    const o = this.instance(name);
    await o.ready;
    const st = this.storage(name);
    st.alarm = null;
    await o.obj.alarm();
    await this.settle(name);
  }
}

export const tick = () => new Promise((r) => setImmediate(r));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
