// GladosBatch: one call pays a whole epoch, holds nothing, and never shorts anyone.
//
//   node test/batch.mjs
//
// Same in-process EVM and TestToken as run.mjs. The claims that earn their
// place are the refusals: a partial batch, a taxed transfer and a lying return
// value each have to revert the *whole* batch, because the failure this
// contract exists to prevent is a payout that looks delivered and is not.
import { compile } from "./build.mjs";
import { makeVm, deploy, call, fund } from "./evm.mjs";

let passed = 0;
let failed = 0;
function ok(cond, what) {
  if (cond) { passed++; console.log(`ok    ${what}`); }
  else { failed++; console.log(`FAIL  ${what}`); }
}

const POOL = "0x1111111111111111111111111111111111111111";
const SINK = "0x3333333333333333333333333333333333333333";
const ONE = 10n ** 18n;
const miners = Array.from({ length: 50 }, (_, i) => "0x" + (0xa000 + i).toString(16).padStart(40, "0"));

async function main() {
  const c = compile(["GladosBatch.sol", "TestToken.sol"]);
  const art = {
    batch: { abi: c["GladosBatch.sol"].GladosBatch.abi, bytecode: c["GladosBatch.sol"].GladosBatch.evm.bytecode.object },
    token: { abi: c["TestToken.sol"].TestToken.abi, bytecode: c["TestToken.sol"].TestToken.evm.bytecode.object },
  };
  const vm = await makeVm();
  await fund(vm, POOL);
  const batch = await deploy(vm, POOL, art.batch, []);
  const tok = await deploy(vm, POOL, art.token, [10n ** 27n]);
  const bal = async (who) => (await call(vm, POOL, tok, art.token, "balanceOf", [who])).result;

  // An equal split of 50 miners, one call.
  const each = 7n * ONE;
  const amounts = miners.map(() => each);
  const total = each * BigInt(miners.length);
  await call(vm, POOL, tok, art.token, "approve", [batch, total]);
  const before = await bal(POOL);
  const r = await call(vm, POOL, batch, art.batch, "send", [tok, miners, amounts]);
  ok(r.ok, "one call pays fifty miners");
  const got = await Promise.all(miners.map(bal));
  ok(got.every((g) => g === each), "every miner received exactly their equal share");
  ok(before - (await bal(POOL)) === total, "the pool paid exactly the epoch's total");
  ok((await bal(batch)) === 0n, "and the batch contract holds nothing afterwards");
  const per = Number(r.gas) / miners.length;
  console.log(`      gas: ${r.gas} for ${miners.length}, ${per.toFixed(0)} per recipient`);
  ok(per < 40_000, "under 40k gas per recipient");

  // All or nothing: approval short by one wei fails the batch, and nobody moves.
  await call(vm, POOL, tok, art.token, "approve", [batch, total - 1n]);
  const snap = await Promise.all(miners.map(bal));
  const short = await call(vm, POOL, batch, art.batch, "send", [tok, miners, amounts]);
  ok(!short.ok, "an allowance one wei short reverts the whole batch");
  ok((await Promise.all(miners.map(bal))).every((g, i) => g === snap[i]), "and not one miner was paid part of it");

  // Mismatched lists are refused before anything moves.
  await call(vm, POOL, tok, art.token, "approve", [batch, total]);
  const mm = await call(vm, POOL, batch, art.batch, "send", [tok, miners, amounts.slice(1)]);
  ok(!mm.ok && /LengthMismatch/.test(mm.reason || ""), "mismatched recipients and amounts are refused");

  // A token that taxes the transfer: a miner would be shorted, so the batch reverts.
  await call(vm, POOL, tok, art.token, "setTax", [100n, SINK]);
  const taxed = await call(vm, POOL, batch, art.batch, "send", [tok, miners.slice(0, 3), amounts.slice(0, 3)]);
  ok(!taxed.ok && /ShortPaid/.test(taxed.reason || ""), "a taxed transfer reverts the batch rather than shorting a miner");
  await call(vm, POOL, tok, art.token, "setTax", [0n, SINK]);

  // A token that returns nothing is fine; one that says `false` is not.
  await call(vm, POOL, tok, art.token, "setSilent", [true]);
  const silent = await call(vm, POOL, batch, art.batch, "send", [tok, miners.slice(0, 2), amounts.slice(0, 2)]);
  ok(silent.ok, "a token returning nothing on success is accepted");
  await call(vm, POOL, tok, art.token, "setSilent", [false]);

  // Somebody else's tokens cannot be spent: `from` is always msg.sender.
  const STRANGER = "0x2222222222222222222222222222222222222222";
  await fund(vm, STRANGER);
  const theft = await call(vm, STRANGER, batch, art.batch, "send", [tok, [STRANGER], [ONE]]);
  ok(!theft.ok, "a caller cannot pay out of the pool's allowance");

  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
