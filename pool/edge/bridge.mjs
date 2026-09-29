// A TCP listener that carries each connection to the pool over WebSocket.
//
//   node pool/edge/bridge.mjs --listen 3337 --to wss://glados-pool.example.workers.dev/mine
//
// **Why this exists: a Cloudflare Worker cannot accept a TCP connection.** The
// docs say so outright -- "it is not possible to make an inbound TCP connection
// to your Worker" -- so a pool served from a Durable Object is reachable only over
// WebSocket, and every miner that speaks plain TCP is shut out. That includes the
// Linux `glados-miner` build. This puts it back: run it beside the miner, point
// the miner at the local port, and each connection becomes one WebSocket to the
// pool with its lines carried both ways unchanged.
//
// It decides nothing and parses nothing past the newline. One line in is one
// WebSocket message out, and one message in is written back as it came, so the
// pool sees exactly what the miner sent and the miner exactly what the pool
// answered -- which is also what makes it a fair test harness.
import net from "node:net";

const args = process.argv.slice(2);
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const port = Number(opt("--listen", 3337));
const target = opt("--to", "");
if (!/^wss?:\/\//.test(target)) {
  console.error("usage: bridge.mjs --listen PORT --to wss://host/mine");
  process.exit(2);
}

net.createServer((sock) => {
  const who = `${sock.remoteAddress}:${sock.remotePort}`;
  const ws = new WebSocket(target);
  const early = [];            // lines that arrive before the socket opens
  let buf = "";
  ws.addEventListener("open", () => {
    for (const l of early.splice(0)) ws.send(l);
    console.log(`[bridge] ${who} -> ${target}`);
  });
  ws.addEventListener("message", (ev) => {
    sock.write(typeof ev.data === "string" ? ev.data : Buffer.from(ev.data));
  });
  ws.addEventListener("close", (ev) => {
    console.log(`[bridge] ${who} pool closed (${ev.code}${ev.reason ? " " + ev.reason : ""})`);
    sock.end();
  });
  ws.addEventListener("error", () => sock.destroy());

  sock.setEncoding("utf8");
  sock.on("data", (chunk) => {
    buf += chunk;
    for (let nl; (nl = buf.indexOf("\n")) >= 0;) {
      const line = buf.slice(0, nl + 1);
      buf = buf.slice(nl + 1);
      if (ws.readyState === WebSocket.OPEN) ws.send(line);
      else early.push(line);
    }
  });
  const done = () => { try { ws.close(); } catch {} };
  sock.on("close", done);
  sock.on("error", done);
}).listen(port, "127.0.0.1", () => console.log(`[bridge] 127.0.0.1:${port} -> ${target}`));
