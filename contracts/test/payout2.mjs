// GladosPayout2: one epoch, paid in whatever each miner chose, in one call.
//
//   node test/payout2.mjs
//
// Against the in-process EVM: MockPair for $GLADOS (a real constant-product
// `k` check), MockV3Pool for the stock legs (pays first, calls back, checks it
// was paid, as a real V3 pool does), MockV3Factory, MockWETH and TestToken.
// The claims are about money that could go missing: every wallet in a group
// equal on every leg, nothing left in the contract, any shortfall reverting the
// whole epoch, and the callback paying only the pool mid-swap.
import { compile } from "./build.mjs";
import { makeVm, deploy, call, fund } from "./evm.mjs";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const OP = "0x1111111111111111111111111111111111111111";
const ATTACKER = "0x4444444444444444444444444444444444444444";
const SINK = "0x3333333333333333333333333333333333333333";
const ONE = 10n ** 18n;
const wallets = (base, n) => Array.from({ length: n }, (_, i) => "0x" + (base + i).toString(16).padStart(40, "0"));

async function main() {
  const c = compile(["GladosPayout2.sol", "MockWETH.sol", "MockPair.sol", "MockV3.sol", "TestToken.sol"]);
  const A = (f, n) => ({ abi: c[f][n].abi, bytecode: c[f][n].evm.bytecode.object });
  const art = { pay: A("GladosPayout2.sol", "GladosPayout2"), weth: A("MockWETH.sol", "MockWETH"), pair: A("MockPair.sol", "MockPair"),
                pool: A("MockV3.sol", "MockV3Pool"), fac: A("MockV3.sol", "MockV3Factory"), tok: A("TestToken.sol", "TestToken") };

  const shipped = JSON.parse((await import("node:fs")).readFileSync(new URL("../../pool/edge/worker/GladosPayout2.json", import.meta.url), "utf8"));
  ok(shipped.bytecode === art.pay.bytecode, "the bytecode the pool deploys is the bytecode tested here");

  const vm = await makeVm();
  await fund(vm, OP, 10n ** 26n);
  await fund(vm, ATTACKER, 10n ** 20n);
  const weth = await deploy(vm, OP, art.weth, []);
  const glados = await deploy(vm, OP, art.tok, [10n ** 30n]);
  const usdg = await deploy(vm, OP, art.tok, [10n ** 30n]);
  const fac = await deploy(vm, OP, art.fac, []);
  const bal = async (t, who) => (await call(vm, OP, t, art.tok, "balanceOf", [who])).result;

  // The $GLADOS pair: 100 WETH against 250M, roughly the real pair's shape.
  const pair = await deploy(vm, OP, art.pair, [weth, glados]);
  await call(vm, OP, weth, art.weth, "deposit", [], { value: 100n * ONE });
  await call(vm, OP, weth, art.weth, "approve", [pair, 100n * ONE]);
  await call(vm, OP, glados, art.tok, "approve", [pair, 250_000_000n * ONE]);
  const wethFirst = weth.toLowerCase() < glados.toLowerCase();
  ok((await call(vm, OP, pair, art.pair, "seed", wethFirst ? [100n * ONE, 250_000_000n * ONE] : [250_000_000n * ONE, 100n * ONE])).ok, "the $GLADOS pair is seeded");

  // V3 pools seeded by direct transfer: the mock prices off its own balances.
  async function v3(a, b, ra, rb) {
    const p = await deploy(vm, OP, art.pool, [a, b, 3000]);
    await call(vm, OP, fac, art.fac, "record", [p, a, b, 3000]);
    for (const [t, r] of [[a, ra], [b, rb]]) {
      if (t === weth) { await call(vm, OP, weth, art.weth, "deposit", [], { value: r }); await call(vm, OP, weth, art.weth, "transfer", [p, r]); }
      else await call(vm, OP, t, art.tok, "transfer", [p, r]);
    }
    return p;
  }
  const wu = await v3(weth, usdg, 2000n * ONE, 6_000_000n * ONE);
  const stocks = [];
  for (const price of [200n, 150n, 90n, 400n]) {
    const s = await deploy(vm, OP, art.tok, [10n ** 30n]);
    stocks.push({ token: s, pool: await v3(usdg, s, 1_000_000n * ONE, (1_000_000n * ONE) / price) });
  }

  const pay = await deploy(vm, OP, art.pay, [weth, glados, pair, usdg, fac, wu]);
  const gLeg = (eth, minOut = 1n) => [pair, glados, eth, minOut];
  const sLeg = (s, eth, minOut = 1n) => [s.pool, s.token, eth, minOut];

  // One epoch, three groups: $GLADOS, one stock, a three-stock basket.
  const gW = wallets(0xa000, 20), sW = wallets(0xb000, 10), kW = wallets(0xc000, 10);
  const per = ONE / 40n; // every wallet's share of a 1 ETH pot
  const groups = [
    [[gLeg(per * 20n)], gW],
    [[sLeg(stocks[0], per * 10n)], sW],
    [stocks.slice(1, 4).map((s, i) => sLeg(s, i < 2 ? (per * 10n) / 3n : per * 10n - 2n * ((per * 10n) / 3n))), kW],
  ];
  const value = per * 40n;
  const r = await call(vm, OP, pay, art.pay, "payAll", [groups], { value });
  ok(r.ok, `one call pays three groups in three currencies${r.ok ? "" : " (" + r.reason + ")"}`);
  const eq = async (t, ws) => { const g = await Promise.all(ws.map((w) => bal(t, w))); return g.every((x) => x === g[0]) && g[0] > 0n; };
  ok(await eq(glados, gW), "every $GLADOS wallet got the same, and something");
  ok(await eq(stocks[0].token, sW), "every single-stock wallet got the same, and something");
  for (const [i, s] of stocks.slice(1, 4).entries()) ok(await eq(s.token, kW), `basket leg ${i + 1}: every basket wallet got the same, and something`);
  ok((await bal(glados, gW[0])) > 0n && (await bal(glados, sW[0])) === 0n, "a stock wallet received no $GLADOS: one currency per choice");
  const empty = (await Promise.all([glados, usdg, ...stocks.map((s) => s.token)].map((t) => bal(t, pay)))).every((x) => x === 0n);
  ok(empty && (await call(vm, OP, weth, art.weth, "balanceOf", [pay])).result === 0n, "the contract keeps nothing: no tokens, no USDG, no WETH");
  console.log(`      gas: ${r.gas} for 20 $GLADOS + 10 single-stock + 10 three-stock basket wallets`);

  // The cost model the treasury plans with: gas per wallet per leg, measured.
  const big = wallets(0xd000, 100);
  const g1 = await call(vm, OP, pay, art.pay, "payAll", [[[[gLeg(ONE)], big]]], { value: ONE });
  const s1 = await call(vm, OP, pay, art.pay, "payAll", [[[[sLeg(stocks[0], ONE)], big]]], { value: ONE });
  console.log(`      gas: ${g1.gas} for 100 $GLADOS wallets, ${s1.gas} for 100 single-stock wallets`);
  ok(g1.ok && s1.ok && Number(g1.gas) / 100 < 40_000 && Number(s1.gas) / 100 < 45_000, "under 45k gas per wallet per leg, buys included");

  // Refusals.
  const bad = async (gs, v, re, what) => { const x = await call(vm, OP, pay, art.pay, "payAll", [gs], { value: v }); ok(!x.ok && re.test(x.reason || ""), what); };
  await bad([], ONE, /NoGroups/, "an epoch with no groups is refused");
  await bad([[[], gW]], ONE, /NoLegs/, "a group buying nothing is refused");
  await bad([[[gLeg(ONE)], []]], ONE, /NoRecipients/, "a group paying nobody is refused");
  await bad([[[gLeg(ONE, 0n)], gW]], ONE, /NoSlippageBound/, "a leg with no slippage bound is refused");
  await bad([[[gLeg(ONE)], gW]], ONE * 2n, /ValueMismatch/, "ETH sent must equal what the legs spend");
  await bad([[[sLeg(stocks[0], ONE, 10n ** 30n)], sW]], ONE, /TooLittleOut/, "an output under the bound reverts the whole epoch");
  await bad([[[gLeg(ONE)], gW], [[sLeg(stocks[0], ONE, 10n ** 30n)], sW]], ONE * 2n, /TooLittleOut/, "one short leg reverts every group, so nobody is half-paid");
  await bad([[[[stocks[0].pool, glados, ONE, 1n]], gW]], ONE, /BadPair/, "a $GLADOS leg on anything but the pair is refused");
  const rogue = await deploy(vm, OP, art.pool, [usdg, stocks[1].token, 3000]);
  await bad([[[[rogue, stocks[1].token, ONE, 1n]], sW]], ONE, /BadPool/, "a pool the factory does not know is refused");
  await bad([[[[stocks[1].pool, stocks[2].token, ONE, 1n]], sW]], ONE, /BadPool/, "a pool that does not trade the leg's token is refused");

  // The callback pays nobody who is not mid-swap with this contract.
  await call(vm, OP, usdg, art.tok, "transfer", [pay, 1000n * ONE]);
  const steal = await call(vm, ATTACKER, pay, art.pay, "uniswapV3SwapCallback", [1000n * ONE, 0n, "0x"]);
  ok(!steal.ok && /BadCallback/.test(steal.reason || ""), "calling the callback directly is refused");
  ok((await bal(usdg, pay)) === 1000n * ONE, "and nothing left the contract");
  await call(vm, OP, stocks[2].pool, art.pool, "setAskForNothing", [true]);
  await bad([[[sLeg(stocks[2], ONE)], sW]], ONE, /BadCallback/, "a pool that never asks to be paid is refused");
  await call(vm, OP, stocks[2].pool, art.pool, "setAskForNothing", [false]);

  // A stock that taxes wallet transfers would short miners, so nobody is paid.
  await call(vm, OP, stocks[1].token, art.tok, "setTax", [100n, SINK]);
  const snap = await Promise.all(kW.map((w) => bal(stocks[1].token, w)));
  await bad([[[sLeg(stocks[1], ONE)], kW]], ONE, /ShortPaid/, "a taxed send to a miner reverts rather than shorting them");
  ok((await Promise.all(kW.map((w) => bal(stocks[1].token, w)))).every((g, i) => g === snap[i]), "and no miner was paid part of it");

  // Built on a pair or pool that is not what it claims: cannot exist.
  for (const [args, what] of [[[weth, stocks[0].token, pair, usdg, fac, wu], "a pair of other tokens"], [[weth, glados, pair, usdg, fac, rogue], "an unknown WETH/USDG pool"]]) {
    let refused = false;
    try { await deploy(vm, OP, art.pay, args); } catch { refused = true; }
    ok(refused, `a payout contract built on ${what} cannot be deployed`);
  }

  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
