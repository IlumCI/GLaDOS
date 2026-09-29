// The WebAssembly pool, served over plain TCP, for testing it without Cloudflare.
//
//   node pool/edge/harness.mjs --port 3336 --coin yescrypt:yescrypt:12 [--coin ...]
//
// **This is the Durable Object's job done on a laptop, so the core can be driven
// by the same clients as the native daemon.** A stock `glados-miner` speaks TCP
// and so do the abuse drills in `tools/poolabuse.py`; pointing both at this and
// at `glados-pool` makes those drills a differential test of one core compiled
// twice. A disagreement is a bug in the boundary, since nothing on either side of
// it is allowed to decide anything.
//
// Framing is the transport's, as it is in `server.rs`: bytes are cut into lines
// here, and a line longer than the kernel's `MAX_LINE` closes the connection
// rather than being guessed at.
import net from "node:net";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { bind, imports } from "./boundary.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const MAX_LINE = 131072;          // `mine::stratum::MAX_LINE`
const JOB_PERIOD_MS = 30_000;     // `server::JOB_PERIOD`

export async function load(wasmPath = join(here, "target/wasm32-unknown-unknown/release/glados_edge.wasm")) {
  const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), imports);
  return bind(instance);
}

async function main() {
  const args = process.argv.slice(2);
  const port = Number(args[args.indexOf("--port") + 1] || 3336);
  const coins = args.flatMap((a, i) => (a === "--coin" ? [args[i + 1]] : []));
  const secs = Number(args.includes("--share-seconds") ? args[args.indexOf("--share-seconds") + 1] : 10);
  const pool = await load();
  const slots = pool.init(coins.length ? coins : ["yescrypt:yescrypt:12"], secs);
  console.log(`[edge] ${slots} slot(s), the pool core running as WebAssembly`);

  net.createServer((sock) => {
    const peer = `${sock.remoteAddress}:${sock.remotePort}`;
    const id = pool.open(peer);
    let buf = Buffer.alloc(0);
    const deliver = (r) => {
      for (const l of r.log) console.log(l);
      for (const m of r.send) sock.write(m);
      if (r.close) sock.end();
      return r.close;
    };
    const timer = setInterval(() => deliver(pool.idle(id)), JOB_PERIOD_MS);
    sock.on("data", (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      for (let nl; (nl = buf.indexOf(10)) >= 0;) {
        const line = buf.subarray(0, nl).toString("utf8");
        buf = buf.subarray(nl + 1);
        if (deliver(pool.line(id, line))) return;
      }
      if (buf.length > MAX_LINE) sock.destroy();
    });
    const done = () => { clearInterval(timer); pool.close(id); };
    sock.on("close", done);
    sock.on("error", done);
  }).listen(port, "127.0.0.1", () => console.log(`[edge] listening on 127.0.0.1:${port}`));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
