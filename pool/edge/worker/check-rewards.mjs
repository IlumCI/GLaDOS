// node pool/edge/worker/check-rewards.mjs [rpc url]
//
// rewards.js against the chain: every pool it names is asked, read-only, for
// its fee and its two tokens. A tier or a pair that is wrong in the table is a
// swap routed to a pool that does not exist, and every payout carrying that
// reward reverts -- nothing offline can catch it (mutation testing showed every
// tier could change unnoticed), so it is checked where the truth is.
import { CHAIN, TOKENS } from "./rewards.js";

const RPC = process.argv[2] || "https://rpc.mainnet.chain.robinhood.com";
const call = async (to, data) => {
  const r = await fetch(RPC, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_call", params: [{ to, data }, "latest"] }) });
  const j = await r.json();
  if (j.error || !j.result || j.result === "0x") throw new Error(j.error ? j.error.message : "no code or empty answer");
  return j.result;
};
const word = (hex) => BigInt(hex.slice(0, 66));
const addrOf = (hex) => "0x" + hex.slice(-40).toLowerCase();

async function pool(name, address, fee, pair) {
  try {
    const [f, t0, t1] = await Promise.all([call(address, "0xddca3f43"), call(address, "0x0dfe1681"), call(address, "0xd21220a7")]);
    const got = new Set([addrOf(t0), addrOf(t1)]);
    const want = pair.map((a) => a.toLowerCase());
    const bad = [];
    if (Number(word(f)) !== fee) bad.push(`fee is ${Number(word(f))}, the table says ${fee}`);
    if (!want.every((a) => got.has(a))) bad.push(`pairs ${[...got].join(" / ")}, not ${want.join(" / ")}`);
    console.log(`${bad.length ? "WRONG" : "ok   "}  ${name.padEnd(10)} ${address}${bad.length ? "  " + bad.join("; ") : ""}`);
    return bad.length === 0;
  } catch (e) {
    console.log(`ERROR  ${name.padEnd(10)} ${address}  ${e.message}`);
    return false;
  }
}

// One pool at a time: a public RPC answers a burst of seventy calls by refusing most of them.
const results = [await pool("WETH/USDG", CHAIN.wethUsdgPool, CHAIN.wethUsdgFee, [CHAIN.weth, CHAIN.usdg])];
for (const [sym, t] of Object.entries(TOKENS)) results.push(await pool(sym, t.pool, t.fee, [t.token, CHAIN.usdg]));
const bad = results.filter((x) => !x).length;
console.log(`\n${results.length - bad} of ${results.length} pools match rewards.js`);
process.exit(bad ? 1 : 0);
