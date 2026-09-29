// The one JavaScript implementation of the WebAssembly pool's boundary.
//
// Shared by `harness.mjs` (Node, over TCP) and `worker/worker.js` (a Cloudflare
// Durable Object, over WebSocket). **One, and not one each**: this is where bytes
// are copied in and tagged records decoded out, and two copies of that are two
// places a length could be read one byte off -- which would show up as a pool
// that works in the harness and garbles every third message in production.
//
// It takes an instance rather than a path, because a Worker has no filesystem:
// it is handed a compiled module by the bundler, and Node reads a file.

export const imports = { glados: { glados_now_ms: () => Date.now() } };

export function bind(instance) {
  const x = instance.exports;
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  // Copy a string into the module and call `f` with (ptr, len), then free it.
  const withBytes = (s, f) => {
    const b = enc.encode(s);
    const p = x.edge_alloc(b.length);
    new Uint8Array(x.memory.buffer, p, b.length).set(b);
    try { return f(p, b.length); } finally { x.edge_free(p, b.length); }
  };
  // The output buffer is re-read after every call: an allocation may grow
  // memory, and a view taken before that points at a detached buffer.
  const out = (n) => new Uint8Array(x.memory.buffer, x.edge_out_ptr(), n).slice();
  const records = (n) => {
    const b = out(n), r = { send: [], log: [], close: false, work: false, why: "" };
    for (let i = 0; i < b.length;) {
      const kind = b[i], len = b[i + 1] | (b[i + 2] << 8) | (b[i + 3] << 16) | (b[i + 4] << 24);
      const body = dec.decode(b.subarray(i + 5, i + 5 + len));
      if (kind === 0x53) r.send.push(body);
      else if (kind === 0x4c) r.log.push(body);
      else if (kind === 0x57) r.work = true;
      else if (kind === 0x43) { r.close = true; r.why = body; }
      i += 5 + len;
    }
    return r;
  };

  return {
    init(coins, shareSecs = 10, window = 0) {
      const slots = withBytes(coins.join("\n"), (p, l) => x.edge_init(p, l, shareSecs, window));
      if (!slots) throw new Error("edge_init: " + dec.decode(out(256)).replace(/\0.*$/, ""));
      return slots;
    },
    open: (peer) => withBytes(peer, (p, l) => x.edge_open(p, l)),
    line: (id, s) => records(withBytes(s, (p, l) => x.edge_line(id, p, l))),
    idle: (id) => records(x.edge_idle(id)),
    // Fresh jobs for one miner, after upstream installed new work.
    work: (id) => records(x.edge_work(id)),

    // Upstreams, addressed by index. The caller owns the socket and hands every
    // chunk it reads to `upBytes`, in whatever pieces the transport delivered.
    upCount: () => x.edge_up_count(),
    upWhere: (i) => {
      const [host, port] = dec.decode(out(x.edge_up_where(i))).split("\t");
      return { hostname: host, port: Number(port) };
    },
    upOpen: (i) => records(x.edge_up_open(i)),
    upTick: (i) => records(x.edge_up_tick(i)),
    upBytes: (i, bytes) => {
      const p = x.edge_alloc(bytes.length);
      new Uint8Array(x.memory.buffer, p, bytes.length).set(bytes);
      try { return records(x.edge_up_bytes(i, p, bytes.length)); } finally { x.edge_free(p, bytes.length); }
    },
    close: (id) => x.edge_close(id),
    ledger: (epoch, at) => dec.decode(out(x.edge_ledger(epoch, at))),
    // Records restored, or an Error carrying the core's reason. Throws rather than
    // answering -1, because the caller is restoring a payout record and a quiet
    // failure there is a pool that starts from zero and overwrites what it lost.
    loadLedger: (doc) => {
      const n = withBytes(doc, (p, l) => x.edge_load_ledger(p, l));
      if (n < 0) throw new Error("edge_load_ledger: " + dec.decode(out(256)).replace(/\0.*$/s, ""));
      return n;
    },
  };
}
