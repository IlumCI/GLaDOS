// node pool/edge/worker/pearl-fake-upstream.mjs [port] [targetBits]
// A stand-in for 2Miners' Pearl port, for local tests: answers
// `mining.authorize {wallet}`, sends one job on a fixed header with target
// 2^(256 - 19 - targetBits) -- so a forwarded share is one in 2^targetBits
// tiles -- and logs every `mining.submit` it is sent.
import net from "node:net";

const port = Number(process.argv[2] || 18930), bits = Number(process.argv[3] || 29);
const header = Array.from({ length: 76 }, (_, i) => ((i * 53 + 7) & 255).toString(16).padStart(2, "0")).join("");
const target = (1n << BigInt(256 - 19 - bits)).toString(16).padStart(64, "0");
let submits = 0;
net.createServer((sock) => {
  let buf = "";
  sock.on("data", (d) => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      let m; try { m = JSON.parse(buf.slice(0, nl)); } catch { m = {}; }
      buf = buf.slice(nl + 1);
      if (m.method === "mining.authorize") {
        console.log(`authorize ${JSON.stringify(m.params)}`);
        sock.write(JSON.stringify({ id: m.id, result: true, error: null }) + "\n");
        sock.write(JSON.stringify({ id: null, method: "mining.notify", params: { job_id: "fake_1", header, target, height: 1 } }) + "\n");
      } else if (m.method === "mining.submit") {
        submits++;
        console.log(`submit #${submits}: job ${m.params.job_id}, jackpot ${String(m.params.jackpot).slice(0, 16)}..., factor ${m.params.adjustment_factor}, proof ${String(m.params.plain_proof).length} chars`);
        sock.write(JSON.stringify({ id: m.id, result: true, error: null }) + "\n");
      }
    }
  });
}).listen(port, "127.0.0.1", () => console.log(`fake upstream on ${port}, forwarding 1 in 2^${bits} tiles`));
