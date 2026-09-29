// GladosPayout: buy and split in one call, holding nothing, shorting nobody.
//
//   node test/payout.mjs
//
// Against the in-process EVM, a MockPair holding a real constant-product `k`
// check (so a wrong output is refused as the chain would refuse it), MockWETH
// and TestToken. The claims that matter are about money that could go missing:
// nothing left in the contract, every miner exactly equal, and any shortfall
// reverting the whole payout rather than paying some miners less.
import { compile } from "./build.mjs";
import { makeVm, deploy, call, fund } from "./evm.mjs";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const POOL = "0x1111111111111111111111111111111111111111";
const SINK = "0x3333333333333333333333333333333333333333";
const ONE = 10n ** 18n;
const miners = Array.from({ length: 50 }, (_, i) => "0x" + (0xb000 + i).toString(16).padStart(40, "0"));

function amountOut(i, rIn, rOut) {
  const w = i * 997n;
  return (w * rOut) / (rIn * 1000n + w);
}

async function main() {
  const c = compile(["GladosPayout.sol", "MockWETH.sol", "MockPair.sol", "TestToken.sol"]);
  const A = (f, n) => ({ abi: c[f][n].abi, bytecode: c[f][n].evm.bytecode.object });
  const art = { payout: A("GladosPayout.sol", "GladosPayout"), weth: A("MockWETH.sol", "MockWETH"),
                pair: A("MockPair.sol", "MockPair"), token: A("TestToken.sol", "TestToken") };

  // The treasury deploys the bytecode committed beside the Worker; it must be
  // exactly what this compile produces, or the pool deploys an untested contract.
  const shipped = JSON.parse((await import("node:fs")).readFileSync(new URL("../../pool/edge/worker/GladosPayout.json", import.meta.url), "utf8"));
  ok(shipped.bytecode === art.payout.bytecode, "the bytecode the pool deploys is the bytecode tested here");

  const vm = await makeVm();
  await fund(vm, POOL, 10n ** 24n);
  const weth = await deploy(vm, POOL, art.weth, []);
  const tok = await deploy(vm, POOL, art.token, [10n ** 27n]);
  const pair = await deploy(vm, POOL, art.pair, [weth, tok]);

  // Seed the pool: 100 WETH against 250M tokens, roughly the real pair's shape.
  const W0 = 100n * ONE, T0 = 250_000_000n * ONE;
  await call(vm, POOL, weth, art.weth, "deposit", [], { value: W0 });
  await call(vm, POOL, weth, art.weth, "approve", [pair, W0]);
  await call(vm, POOL, tok, art.token, "approve", [pair, T0]);
  const wethFirst = weth.toLowerCase() < tok.toLowerCase();
  const seeded = await call(vm, POOL, pair, art.pair, "seed", wethFirst ? [W0, T0] : [T0, W0]);
  ok(seeded.ok, "the pair is seeded");

  const payout = await deploy(vm, POOL, art.payout, [weth, tok, pair]);
  const bal = async (a, who) => (await call(vm, POOL, a, art.token, "balanceOf", [who])).result;

  // An epoch: 1 ETH, fifty miners.
  const IN = ONE;
  const quote = amountOut(IN, W0, T0);
  const r = await call(vm, POOL, payout, art.payout, "buyAndPay", [miners, (quote * 97n) / 100n], { value: IN });
  ok(r.ok, `one call buys and pays fifty miners${r.ok ? "" : " (" + r.reason + ")"}`);
  const got = await Promise.all(miners.map((m) => bal(tok, m)));
  const each = quote / 50n;
  ok(got.every((g) => g === each), "every miner received exactly the same amount");
  ok(each > 0n, `and it is a real amount: ${(Number(each) / 1e18).toFixed(2)} tokens each`);
  ok((await bal(tok, payout)) === 0n, "the contract holds no tokens afterwards");
  ok((await call(vm, POOL, weth, art.weth, "balanceOf", [payout])).result === 0n, "and no WETH");
  console.log(`      gas: ${r.gas} for 50, ${(Number(r.gas) / 50).toFixed(0)} per recipient`);
  ok(Number(r.gas) / 50 < 40_000, "under 40k gas per recipient, buy included");

  // Refusals.
  const zero = await call(vm, POOL, payout, art.payout, "buyAndPay", [miners, 0n], { value: IN });
  ok(!zero.ok && /NoSlippageBound/.test(zero.reason || ""), "a payout with no slippage bound is refused");
  const none = await call(vm, POOL, payout, art.payout, "buyAndPay", [[], 1n], { value: IN });
  ok(!none.ok && /NoRecipients/.test(none.reason || ""), "a payout to nobody is refused");
  const greedy = await call(vm, POOL, payout, art.payout, "buyAndPay", [miners, quote * 2n], { value: IN });
  ok(!greedy.ok && /TooLittleOut/.test(greedy.reason || ""), "an output under the bound -- a sandwich, say -- reverts the whole payout");

  // A token that taxes wallet transfers too: miners would be shorted, so nobody is paid.
  await call(vm, POOL, tok, art.token, "setTax", [100n, SINK]);
  const snap = await Promise.all(miners.map((m) => bal(tok, m)));
  const taxed = await call(vm, POOL, payout, art.payout, "buyAndPay", [miners, 1n], { value: IN });
  ok(!taxed.ok && /ShortPaid/.test(taxed.reason || ""), "a taxed send to a miner reverts the payout rather than shorting them");
  ok((await Promise.all(miners.map((m) => bal(tok, m)))).every((g, i) => g === snap[i]), "and no miner was paid part of it");
  await call(vm, POOL, tok, art.token, "setTax", [0n, SINK]);

  // A contract built against the wrong pair does not exist.
  const other = await deploy(vm, POOL, art.token, [10n ** 27n]);
  let refused = false;
  try { await deploy(vm, POOL, art.payout, [weth, other, pair]); } catch { refused = true; }
  ok(refused, "a payout contract pointed at a pair of other tokens cannot be deployed");

  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
