// What the distributor must do, and the much longer list of what it must refuse.
//
// A distributor that pays a valid claim is half checked, and it is the wrong
// half: the interesting behaviour of a contract holding somebody's tokens is
// every path where it says no. So most of what follows is refusals, and each
// one asserts *which* refusal -- getting the wrong one is a real failure and
// "it reverted" hides it.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ethers } from "ethers";
import { compile } from "./build.mjs";
import { makeVm, deploy, call, fund, decodeRevert } from "./evm.mjs";
import { build, leaf } from "./merkle.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));

let passed = 0;
let failed = 0;
function ok(cond, what) {
  if (cond) {
    passed++;
    console.log(`ok    ${what}`);
  } else {
    failed++;
    console.log(`FAIL  ${what}`);
  }
}
function eq(a, b, what) {
  ok(String(a) === String(b), `${what}${String(a) === String(b) ? "" : `  (got ${a}, wanted ${b})`}`);
}

const OPERATOR = "0x1111111111111111111111111111111111111111";
const STRANGER = "0x2222222222222222222222222222222222222222";
const A = "0x00000000000000000000000000000000000000aa";
const B = "0x00000000000000000000000000000000000000bb";
const C = "0x00000000000000000000000000000000000000cc";
const D = "0x00000000000000000000000000000000000000dd";

const ONE = 10n ** 18n;
const GATE = 1_000_000n * ONE;
const DAY = 86_400n;

function artifacts() {
  const c = compile(["GladosDistributor.sol", "TestToken.sol", "MockPair.sol", "MockV3.sol"]);
  return {
    dist: {
      abi: c["GladosDistributor.sol"].GladosDistributor.abi,
      bytecode: c["GladosDistributor.sol"].GladosDistributor.evm.bytecode.object,
    },
    token: {
      abi: c["TestToken.sol"].TestToken.abi,
      bytecode: c["TestToken.sol"].TestToken.evm.bytecode.object,
    },
    pair: {
      abi: c["MockPair.sol"].MockPair.abi,
      bytecode: c["MockPair.sol"].MockPair.evm.bytecode.object,
    },
    v3pool: {
      abi: c["MockV3.sol"].MockV3Pool.abi,
      bytecode: c["MockV3.sol"].MockV3Pool.evm.bytecode.object,
    },
    v3factory: {
      abi: c["MockV3.sol"].MockV3Factory.abi,
      bytecode: c["MockV3.sol"].MockV3Factory.evm.bytecode.object,
    },
  };
}

/// A block whose timestamp we control, because half of what this contract does
/// is about a deadline and a test that cannot move the clock cannot check it.
function at(ts) {
  return { header: { timestamp: ts, number: 1n, cliqueSigner: () => ({ toString: () => "0x0" }) } };
}

async function main() {
  const art = artifacts();
  const vm = await makeVm();
  for (const a of [OPERATOR, STRANGER, A, B, C, D]) await fund(vm, a);

  const token = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
  const quote = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
  const pair = await deploy(vm, OPERATOR, art.pair, [quote, token]);

  // The reward token for the V3 path is deliberately *not* `token`: the whole
  // point of that mode is paying something this contract does not gate on, and
  // a fixture where the two coincide could not tell the two apart.
  const reward = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
  const factory = await deploy(vm, OPERATOR, art.v3factory, []);
  const v3 = await deploy(vm, OPERATOR, art.v3pool, [quote, reward, 500]);
  await call(vm, OPERATOR, factory, art.v3factory, "record", [v3, quote, reward, 500]);
  await call(vm, OPERATOR, quote, art.token, "mint", [v3, 400_000n * ONE]);
  await call(vm, OPERATOR, reward, art.token, "mint", [v3, 2_000n * ONE]);

  const dist = await deploy(vm, OPERATOR, art.dist, [token, OPERATOR, pair, quote, factory]);

  // Everybody who will claim holds exactly the gate, except D, who holds one
  // short -- the boundary is where a `>=` becomes a `>` by accident.
  for (const [who, bal] of [[A, GATE], [B, GATE * 5n], [C, GATE], [D, GATE - 1n]]) {
    await call(vm, OPERATOR, token, art.token, "mint", [who, bal]);
  }

  const entries = [
    { account: A, amount: 100n * ONE },
    { account: B, amount: 250n * ONE },
    { account: C, amount: 650n * ONE },
  ];
  const tree = build(entries);
  const total = entries.reduce((s, e) => s + e.amount, 0n);

  // ---------------------------------------------------- the leaf format
  {
    const r = await call(vm, OPERATOR, dist, art.dist, "leafOf", [A, 100n * ONE]);
    eq(r.result, leaf(A, 100n * ONE), "the JS leaf matches the contract's leaf");
  }

  // ---------------------------------------------------- opening an epoch
  {
    const r = await call(vm, STRANGER, dist, art.dist, "openEpoch", [tree.root, total, GATE, 1_000_000n]);
    eq(r.reason, "NotOperator", "a stranger cannot open an epoch");
  }
  {
    const r = await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [ethers.ZeroHash, total, GATE, 1_000_000n], { block: at(1000n) });
    eq(r.reason, "NoRoot", "an empty root is refused");
  }
  {
    const r = await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [tree.root, total, GATE, 500n], { block: at(1000n) });
    eq(r.reason, "DeadlineInPast", "a deadline already past is refused");
  }
  {
    // No approval yet, so the pull must fail rather than create an epoch that
    // exists and cannot pay.
    const r = await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [tree.root, total, GATE, 1_000_000n], { block: at(1000n) });
    eq(r.reason, "TransferFailed", "an unfunded epoch is not created");
    const n = await call(vm, OPERATOR, dist, art.dist, "epochCount");
    eq(n.result, 0n, "and no epoch was recorded");
  }

  await call(vm, OPERATOR, token, art.token, "approve", [dist, 10n ** 27n]);
  const DEADLINE = 1000n + 30n * DAY;
  {
    const r = await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [tree.root, total, GATE, DEADLINE], { block: at(1000n) });
    ok(r.ok, "the operator opens epoch 0");
    const e = await call(vm, OPERATOR, dist, art.dist, "epochs", [0]);
    eq(e.result[1], total, "the epoch records what actually arrived");
  }

  // ---------------------------------------------------------- claiming
  {
    const r = await call(vm, A, dist, art.dist, "claim",
      [0, 100n * ONE, tree.proof(A)], { block: at(2000n) });
    ok(r.ok, "a holder with a valid proof claims");
    const bal = await call(vm, A, token, art.token, "balanceOf", [A]);
    eq(bal.result, GATE + 100n * ONE, "and receives exactly the leaf amount");
  }
  {
    const r = await call(vm, A, dist, art.dist, "claim",
      [0, 100n * ONE, tree.proof(A)], { block: at(2000n) });
    eq(r.reason, "AlreadyClaimed", "the same claim a second time is refused");
  }
  {
    // B's proof, but A's amount. The leaf is a pair and a proof is only valid
    // for the pair it was built from.
    const r = await call(vm, B, dist, art.dist, "claim",
      [0, 100n * ONE, tree.proof(B)], { block: at(2000n) });
    eq(r.reason, "BadProof", "claiming somebody else's amount is refused");
  }
  {
    const r = await call(vm, B, dist, art.dist, "claim",
      [0, 250n * ONE, tree.proof(A)], { block: at(2000n) });
    eq(r.reason, "BadProof", "claiming with somebody else's proof is refused");
  }
  {
    const r = await call(vm, STRANGER, dist, art.dist, "claim",
      [0, 100n * ONE, tree.proof(A)], { block: at(2000n) });
    // A stranger holds no tokens, so the gate stops them before the proof does.
    ok(r.reason.startsWith("BelowGate"), `an address not in the tree is refused (${r.reason})`);
  }

  // ----------------------------------------------------------- the gate
  {
    // D is in no tree at all but is one wei short of the gate, which is the
    // boundary this check exists for.
    const r = await call(vm, D, dist, art.dist, "claim",
      [0, 1n, []], { block: at(2000n) });
    ok(r.reason.startsWith("BelowGate"), `one wei below the gate is refused (${r.reason})`);
  }
  {
    // C sells down below the gate after mining, and forfeits. Intended: the
    // gate is a holding requirement rather than an entry fee.
    await call(vm, C, token, art.token, "transfer", [STRANGER, 1n]);
    const r = await call(vm, C, dist, art.dist, "claim",
      [0, 650n * ONE, tree.proof(C)], { block: at(2000n) });
    ok(r.reason.startsWith("BelowGate"), "selling below the gate before claiming forfeits");
    // And buying back restores it, so the check is on the balance now rather
    // than on a flag set once.
    await call(vm, STRANGER, token, art.token, "transfer", [C, 1n]);
    const back = await call(vm, C, dist, art.dist, "checkClaim", [0, C, 650n * ONE, tree.proof(C)],
      { block: at(2000n) });
    eq(back.result[0], true, "and buying back restores it");
  }

  // ------------------------------------------------------ checkClaim view
  {
    const r = await call(vm, A, dist, art.dist, "checkClaim", [0, A, 100n * ONE, tree.proof(A)],
      { block: at(2000n) });
    eq(r.result[1], "already claimed", "checkClaim explains an already-claimed claim");
    const s = await call(vm, A, dist, art.dist, "checkClaim", [0, STRANGER, 1n, []], { block: at(2000n) });
    eq(s.result[1], "below the gate", "checkClaim explains a gate failure");
    const bad = await call(vm, A, dist, art.dist, "checkClaim", [0, B, 999n * ONE, tree.proof(B)],
      { block: at(2000n) });
    eq(bad.result[1], "proof does not match the root", "checkClaim explains a bad proof");
  }

  // --------------------------------------------------------- the deadline
  {
    const r = await call(vm, B, dist, art.dist, "claim",
      [0, 250n * ONE, tree.proof(B)], { block: at(DEADLINE + 1n) });
    eq(r.reason, "EpochClosed", "a claim after the deadline is refused");
  }
  {
    const r = await call(vm, OPERATOR, dist, art.dist, "reclaim", [0], { block: at(2000n) });
    eq(r.reason, "EpochOpen", "the operator cannot reclaim before the deadline");
  }
  {
    const r = await call(vm, STRANGER, dist, art.dist, "reclaim", [0], { block: at(DEADLINE + 1n) });
    eq(r.reason, "NotOperator", "a stranger cannot reclaim");
  }
  {
    const before = await call(vm, OPERATOR, token, art.token, "balanceOf", [OPERATOR]);
    const r = await call(vm, OPERATOR, dist, art.dist, "reclaim", [0], { block: at(DEADLINE + 1n) });
    ok(r.ok, "the operator reclaims the remainder after the deadline");
    const after = await call(vm, OPERATOR, token, art.token, "balanceOf", [OPERATOR]);
    // A claimed 100; B and C never did. So 900 goes back.
    eq(after.result - before.result, 900n * ONE, "and gets back exactly what nobody claimed");
    const again = await call(vm, OPERATOR, dist, art.dist, "reclaim", [0], { block: at(DEADLINE + 2n) });
    eq(again.reason, "AlreadyReclaimed", "and cannot reclaim twice");
  }

  // -------------------------------------------- a token that takes a cut
  //
  // The real token does not tax wallet-to-wallet transfers -- read off the
  // chain rather than assumed. This is the test for being wrong about that:
  // the epoch must record what arrived, not what was asked for, so the
  // contract can never promise more than it holds.
  {
    await call(vm, OPERATOR, token, art.token, "setTax", [500, STRANGER]); // 5%
    const t2 = build([{ account: A, amount: 100n * ONE }]);
    const r = await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t2.root, 100n * ONE, 0n, DEADLINE * 2n], { block: at(1000n) });
    ok(r.ok, "an epoch opens even when the token takes a cut");
    const e = await call(vm, OPERATOR, dist, art.dist, "epochs", [1]);
    eq(e.result[1], 95n * ONE, "and records the 95 that arrived, not the 100 requested");
    // The claim for 100 is now more than the epoch holds, and it is refused
    // here rather than as a failed transfer for whoever claims last.
    const c = await call(vm, A, dist, art.dist, "claim", [1, 100n * ONE, t2.proof(A)],
      { block: at(2000n) });
    ok(c.reason.startsWith("Insolvent"), `an over-promised claim is refused as insolvent (${c.reason})`);
    await call(vm, OPERATOR, token, art.token, "setTax", [0, STRANGER]);
  }

  // ------------------------------------------------------- reentrancy
  //
  // The token calls back into the distributor during the transfer, trying the
  // same claim again. Effects land before the interaction, so the reentrant
  // call must find `hasClaimed` already true.
  {
    const t3 = build([{ account: A, amount: 10n * ONE }, { account: B, amount: 10n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t3.root, 20n * ONE, 0n, DEADLINE * 3n], { block: at(1000n) });
    const iface = new ethers.Interface(art.dist.abi);
    const reenter = iface.encodeFunctionData("claim", [2, 10n * ONE, t3.proof(A)]);
    await call(vm, OPERATOR, token, art.token, "setHook", [dist, reenter]);
    const before = await call(vm, A, token, art.token, "balanceOf", [A]);
    const r = await call(vm, A, dist, art.dist, "claim", [2, 10n * ONE, t3.proof(A)],
      { block: at(2000n) });
    ok(r.ok, "a claim succeeds while the token reenters");
    const after = await call(vm, A, token, art.token, "balanceOf", [A]);
    eq(after.result - before.result, 10n * ONE, "and the reentrant claim paid nothing extra");
    const e = await call(vm, OPERATOR, dist, art.dist, "epochs", [2]);
    eq(e.result[2], 10n * ONE, "the epoch counted one claim, not two");
    await call(vm, OPERATOR, token, art.token, "setHook", [ethers.ZeroAddress, "0x"]);
  }

  // ------------------------------------------- a token that returns nothing
  {
    await call(vm, OPERATOR, token, art.token, "setSilent", [true]);
    const t4 = build([{ account: A, amount: 7n * ONE }]);
    const r = await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t4.root, 7n * ONE, 0n, DEADLINE * 4n], { block: at(1000n) });
    ok(r.ok, "a token that returns no data still works");
    const c = await call(vm, A, dist, art.dist, "claim", [3, 7n * ONE, t4.proof(A)], { block: at(2000n) });
    ok(c.ok, "and the claim against it succeeds");
    await call(vm, OPERATOR, token, art.token, "setSilent", [false]);
  }


  // What the pool owes for a given input, computed from its own reserves
  // rather than asked of the thing under test. Declared out here because two
  // separate sections need it.
  let q0 = false;
  const outFor = async (amt) => {
    const r = await call(vm, OPERATOR, pair, art.pair, "getReserves");
    const [rq, rt] = q0 ? [r.result[0], r.result[1]] : [r.result[1], r.result[0]];
    const wf = amt * 997n;
    return (wf * rt) / (rq * 1000n + wf);
  };

  // ------------------------------------------------ paying through the market
  //
  // The other mode: the epoch holds `quote`, and each claim is the claimant's
  // own buy on the pool. What is being checked is not the AMM -- that is the
  // router's problem -- but that the distributor takes what the market gave
  // rather than what it asked for, bounds slippage, and cannot be used to
  // claim twice.
  {
    // Seed the pool. 100 quote against 1,000,000 token is a thin pool on
    // purpose: a thin one makes the price move visibly between claims, which is
    // the property that separates this mode from the other.
    await call(vm, OPERATOR, quote, art.token, "approve", [pair, 10n ** 27n]);
    await call(vm, OPERATOR, token, art.token, "approve", [pair, 10n ** 27n]);
    await call(vm, OPERATOR, token, art.token, "mint", [OPERATOR, 10n ** 24n]);
    // The pair sorts by address, so seed in its own order rather than ours.
    q0 = BigInt(quote) < BigInt(token);
    await call(vm, OPERATOR, pair, art.pair, "seed",
      q0 ? [100n * ONE, 1_000_000n * ONE] : [1_000_000n * ONE, 100n * ONE]);
    // The real token's 1% buy tax, applied to every transfer out of the pair.
    await call(vm, OPERATOR, token, art.token, "setTax", [100, STRANGER]);

    const t = build([{ account: A, amount: 4n * ONE }, { account: B, amount: 4n * ONE }]);
    await call(vm, OPERATOR, quote, art.token, "approve", [dist, 10n ** 27n]);
    const open = await call(vm, OPERATOR, dist, art.dist, "openEpochOnMarket",
      [t.root, 8n * ONE, 0n, 200_000n], { block: at(1000n) });
    ok(open.ok, `a market epoch opens${open.ok ? "" : "  (" + open.reason + ")"}`);
    const id = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;

    // A direct claim against a market epoch, and the reverse, must not work.
    {
      const r = await call(vm, A, dist, art.dist, "claim", [id, 4n * ONE, t.proof(A)], { block: at(2000n) });
      eq(r.reason, "WrongMode", "a direct claim against a market epoch is refused");
      const r2 = await call(vm, A, dist, art.dist, "claimOnMarket",
        [0, 100n * ONE, tree.proof(A), 1n], { block: at(2000n) });
      eq(r2.reason, "AlreadyClaimed", "and a market claim against a direct epoch stops at the earlier check");
    }
    {
      const r = await call(vm, A, dist, art.dist, "claimOnMarket",
        [id, 4n * ONE, t.proof(A), 0n], { block: at(2000n) });
      eq(r.reason, "NoSlippageBound", "a swap with no slippage bound is refused");
    }

    // What the pool would give, asked before claiming, so the assertion is
    // against the router's own arithmetic rather than a number written here.
    const expect = await outFor(4n * ONE);
    const afterTax = expect - expect / 100n;
    {
      const before = await call(vm, OPERATOR, token, art.token, "balanceOf", [A]);
      const r = await call(vm, A, dist, art.dist, "claimOnMarket",
        [id, 4n * ONE, t.proof(A), afterTax], { block: at(2000n) });
      ok(r.ok, `a market claim buys on the pool${r.ok ? "" : "  (" + r.reason + ")"}`);
      const after = await call(vm, OPERATOR, token, art.token, "balanceOf", [A]);
      eq(after.result - before.result, afterTax, "and the claimant receives what the market gave, after the buy tax");
    }
    {
      // The second claimant, same leaf amount, gets *less* -- the first claim
      // moved the price. That is the cost of paying through a market and it is
      // asserted rather than mentioned.
      const second = await outFor(4n * ONE);
      ok(second < expect, `the second claimant gets a worse price (${second} < ${expect})`);
      const before = await call(vm, OPERATOR, token, art.token, "balanceOf", [B]);
      const r = await call(vm, B, dist, art.dist, "claimOnMarket",
        [id, 4n * ONE, t.proof(B), 1n], { block: at(2000n) });
      ok(r.ok, "and can still claim");
      const after = await call(vm, OPERATOR, token, art.token, "balanceOf", [B]);
      ok(after.result - before.result < afterTax, "receiving less than the first did");
    }
    {
      const r = await call(vm, A, dist, art.dist, "claimOnMarket",
        [id, 4n * ONE, t.proof(A), 1n], { block: at(2000n) });
      eq(r.reason, "AlreadyClaimed", "a market claim cannot be made twice");
    }
    {
      // The distributor must hold no standing allowance to the router.
      const a = await call(vm, OPERATOR, quote, art.token, "allowance", [dist, pair]);
      eq(a.result, 0n, "and never grants the pair an allowance at all");
    }
  }

  // ------------------------------------ when the token takes more than expected
  //
  // Going direct to the pair means the pair's own `k` check bounds the swap and
  // the distributor's post-swap check is the only thing standing between a
  // claimant and a bad fill from the *token* side -- a tax rate that changed
  // since the claimant computed their bound. Without exercising it, that check
  // is unreachable code, and an unreachable check is one nobody has seen work.
  {
    const t = build([{ account: C, amount: 2n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpochOnMarket",
      [t.root, 2n * ONE, 0n, 300_000n], { block: at(1000n) });
    const id = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const want = await outFor(2n * ONE);

    await call(vm, OPERATOR, token, art.token, "setTax", [5000, STRANGER]); // 50%
    const r = await call(vm, C, dist, art.dist, "claimOnMarket",
      [id, 2n * ONE, t.proof(C), want], { block: at(2000n) });
    ok(r.reason.startsWith("TooLittleOut"),
       `a tax that moved under the claimant is refused (${r.reason})`);
    const still = await call(vm, OPERATOR, dist, art.dist, "hasClaimed", [id, C]);
    eq(still.result, false, "and the claim is not marked used, so nothing is lost");

    await call(vm, OPERATOR, token, art.token, "setTax", [100, STRANGER]);
    const ok2 = await call(vm, C, dist, art.dist, "claimOnMarket",
      [id, 2n * ONE, t.proof(C), 1n], { block: at(2000n) });
    ok(ok2.ok, "and the same claim works once the tax is back");
  }

  // ------------------------------------------------- the Uniswap V3 path
  //
  // The reward here is a token the distributor does not gate on, which is the
  // whole reason this mode exists: gate on GLADOS, pay in something else. Most
  // of what follows is the callback, because a function whose job is to pay
  // somebody is the only part of this carrying real risk.
  {
    const t = build([{ account: A, amount: 4n * ONE }, { account: B, amount: 4n * ONE }]);
    await call(vm, OPERATOR, quote, art.token, "approve", [dist, 100n * ONE]);
    const open = await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
      [t.root, 8n * ONE, GATE, 200_000n, v3, reward], { block: at(1000n) });
    ok(open.ok, `a V3 epoch opens${open.ok ? "" : "  (" + open.reason + ")"}`);
    const id = (await call(vm, OPERATOR, dist, art.dist, "epochCount")).result - 1n;

    {
      const gBefore = await call(vm, OPERATOR, token, art.token, "balanceOf", [A]);
      const before = await call(vm, OPERATOR, reward, art.token, "balanceOf", [A]);
      const r = await call(vm, A, dist, art.dist, "claimOnV3",
        [id, 4n * ONE, t.proof(A), 1n], { block: at(2000n) });
      ok(r.ok, `a V3 claim swaps and pays${r.ok ? "" : "  (" + r.reason + ")"}`);
      const after = await call(vm, OPERATOR, reward, art.token, "balanceOf", [A]);
      ok(after.result > before.result, "the claimant receives the reward token");
      const gAfter = await call(vm, OPERATOR, token, art.token, "balanceOf", [A]);
      eq(gAfter.result, gBefore.result, "and their gate-token balance is untouched");
    }

    // The gate is still the gate. D holds one under it and holds no reward
    // token at all, so a contract that had started gating on the reward would
    // refuse for the wrong reason and this would still pass -- hence the
    // explicit selector rather than merely asserting failure.
    {
      const t2 = build([{ account: D, amount: 1n * ONE }]);
      await call(vm, OPERATOR, quote, art.token, "approve", [dist, 10n * ONE]);
      const o2 = await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
        [t2.root, 1n * ONE, GATE, 200_000n, v3, reward], { block: at(1000n) });
      const id2 = (await call(vm, OPERATOR, dist, art.dist, "epochCount")).result - 1n;
      ok(o2.ok, "a second V3 epoch opens");
      const r = await call(vm, D, dist, art.dist, "claimOnV3",
        [id2, 1n * ONE, t2.proof(D), 1n], { block: at(2000n) });
      ok(r.reason.startsWith("BelowGate"), `the gate is still measured on the gate token (${r.reason})`);
    }

    eq((await call(vm, A, dist, art.dist, "claimOnV3",
        [id, 4n * ONE, t.proof(A), 1n], { block: at(2000n) })).reason,
      "AlreadyClaimed", "a V3 claim cannot be made twice");

    eq((await call(vm, B, dist, art.dist, "claimOnV3",
        [id, 4n * ONE, t.proof(B), 0n], { block: at(2000n) })).reason,
      "NoSlippageBound", "a V3 claim with no slippage bound is refused");

    eq((await call(vm, B, dist, art.dist, "claimOnMarket",
        [id, 4n * ONE, t.proof(B), 1n], { block: at(2000n) })).reason,
      "WrongMode", "claimOnMarket refuses a V3 epoch");

    // The slippage bound has to be reachable or the test asserting it fires is
    // asserting nothing, so the pool is told to underpay.
    {
      await call(vm, OPERATOR, v3, art.v3pool, "setShortfall", [5000n]);
      const r = await call(vm, B, dist, art.dist, "claimOnV3",
        [id, 4n * ONE, t.proof(B), 10n ** 18n], { block: at(2000n) });
      ok(r.reason.startsWith("TooLittleOut"), "a V3 claim under its slippage bound is refused");
      await call(vm, OPERATOR, v3, art.v3pool, "setShortfall", [0n]);
    }

    // ---- the callback, which is the part that can lose money
    //
    // Anybody may call it, so the check is that it refuses everybody. The
    // second case is the one that matters: a *genuine* pool, of the attacker's
    // own creation, which a factory lookup would happily authorise.
    eq((await call(vm, STRANGER, dist, art.dist, "uniswapV3SwapCallback",
        [1n * ONE, 0n, "0x"], { block: at(2000n) })).reason,
      "BadCallback", "a stranger cannot call the swap callback");

    eq((await call(vm, OPERATOR, dist, art.dist, "uniswapV3SwapCallback",
        [1n * ONE, 0n, "0x"], { block: at(2000n) })).reason,
      "BadCallback", "nor can the operator");

    {
      const evil = await deploy(vm, STRANGER, art.v3pool, [quote, reward, 3000]);
      await call(vm, STRANGER, factory, art.v3factory, "record", [evil, quote, reward, 3000]);
      const before = await call(vm, OPERATOR, quote, art.token, "balanceOf", [dist]);
      const r = await call(vm, STRANGER, dist, art.dist, "uniswapV3SwapCallback",
        [1n * ONE, 0n, "0x"], { block: at(2000n) });
      eq(r.reason, "BadCallback", "nor can a real pool the factory knows about");
      const after = await call(vm, OPERATOR, quote, art.token, "balanceOf", [dist]);
      eq(after.result, before.result, "and nothing left the contract while it was refusing");
    }

    // ---- the pool argument, which the operator could get wrong
    {
      const rogue = await deploy(vm, STRANGER, art.v3pool, [quote, reward, 10000]);
      const r = await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
        [t.root, 1n * ONE, GATE, 200_000n, rogue, reward], { block: at(1000n) });
      eq(r.reason, "BadPool", "a pool the factory does not name is refused");
    }
    {
      const other = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
      const wrong = await deploy(vm, OPERATOR, art.v3pool, [quote, other, 500]);
      await call(vm, OPERATOR, factory, art.v3factory, "record", [wrong, quote, other, 500]);
      const r = await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
        [t.root, 1n * ONE, GATE, 200_000n, wrong, reward], { block: at(1000n) });
      eq(r.reason, "BadPool", "a real pool for a different pair is refused");
    }
    eq((await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
        [t.root, 1n * ONE, GATE, 200_000n, v3, quote], { block: at(1000n) })).reason,
      "BadPool", "an epoch rewarding the quote token itself is refused");

    // A pool that asks for nothing back would, against a callback that paid
    // whatever it was told, be a way to take the output for free. Here it is
    // refused on the way in.
    {
      await call(vm, OPERATOR, v3, art.v3pool, "setAskForNothing", [true]);
      const r = await call(vm, B, dist, art.dist, "claimOnV3",
        [id, 4n * ONE, t.proof(B), 1n], { block: at(2000n) });
      eq(r.reason, "BadCallback", "a pool asking for nothing back is refused");
      await call(vm, OPERATOR, v3, art.v3pool, "setAskForNothing", [false]);
    }

    // Reclaim hands back the asset the epoch actually holds, which for a V3
    // epoch is the quote and never the reward.
    {
      const before = await call(vm, OPERATOR, quote, art.token, "balanceOf", [OPERATOR]);
      const r = await call(vm, OPERATOR, dist, art.dist, "reclaim", [id], { block: at(300_000n) });
      ok(r.ok, `a V3 epoch reclaims${r.ok ? "" : "  (" + r.reason + ")"}`);
      const after = await call(vm, OPERATOR, quote, art.token, "balanceOf", [OPERATOR]);
      ok(after.result > before.result, "and what comes back is the quote token");
    }
  }

  // ------------------------------- a distributor with no market configured
  {
    const lone = await deploy(vm, OPERATOR, art.dist,
      [token, OPERATOR, ethers.ZeroAddress, ethers.ZeroAddress, ethers.ZeroAddress]);
    const t = build([{ account: A, amount: 1n }]);
    const r = await call(vm, OPERATOR, lone, art.dist, "openEpochOnMarket",
      [t.root, 1n, 0n, 200_000n], { block: at(1000n) });
    eq(r.reason, "NoMarket", "a distributor with no pair refuses market epochs");
  }

  // ------------------------------------------------- the tree itself
  {
    // A single-leaf tree has an empty proof, and it is the shape most easily
    // got wrong -- the loop never runs and the root is the leaf.
    const one = build([{ account: A, amount: 1n }]);
    eq(one.root, leaf(A, 1n), "a one-leaf tree's root is its leaf");
    eq(one.proof(A).length, 0, "and its proof is empty");
    // Odd counts, which is where carry-versus-duplicate would show.
    for (const n of [2, 3, 5, 7, 8, 9, 33]) {
      const es = [];
      for (let i = 0; i < n; i++) {
        es.push({ account: ethers.getAddress("0x" + (i + 1).toString(16).padStart(40, "0")), amount: BigInt(i + 1) });
      }
      const t = build(es);
      // **Verified by the contract, not by the builder that made them.** A
      // proof folded with the same code that built the tree agrees with itself
      // whatever the rules are -- reversed handedness, a singly-hashed leaf, a
      // duplicated odd node all pass. The arbiter has to be the bytecode.
      let allOk = true;
      for (const e of es) {
        const v = await call(vm, OPERATOR, dist, art.dist, "verifyProof",
          [t.proof(e.account), t.root, e.account, e.amount]);
        if (!v.ok || v.result !== true) allOk = false;
      }
      // And a proof from the wrong tree must not verify, or the check above is
      // satisfied by a verifier that says yes to everything.
      const other = build(es.map((e) => ({ account: e.account, amount: e.amount + 1n })));
      const wrong = await call(vm, OPERATOR, dist, art.dist, "verifyProof",
        [other.proof(es[0].account), t.root, es[0].account, es[0].amount]);
      if (wrong.result === true) allOk = false;
      ok(allOk, `every proof verifies against the contract in a tree of ${n}, and a foreign one does not`);
    }
    let threw = false;
    try {
      build([{ account: A, amount: 1n }, { account: A, amount: 2n }]);
    } catch {
      threw = true;
    }
    ok(threw, "a duplicate account in one tree is refused at build time");
  }

  // ------------------------------------------------ what an audit went looking for
  //
  // Everything above was written by the author of the contract. What follows
  // was written by a reader who did not, against a threat list drawn up before
  // the code was read, and each case here exists because the reading raised a
  // suspicion that had to be settled one way or the other. Several of them are
  // settled *clean* and are kept for that reason -- a suspicion that died is
  // worth as much as one that held, and rather more than one nobody wrote down.
  //
  // `design/audit.md` carries the reasoning; this carries the evidence.
  {
    // ---- the V2 pair is never checked against the pair it is supposed to be
    //
    // `openEpochOnV3` validates its pool twice: the factory must name it, and
    // its two tokens must be exactly {quote, reward}. The constructor does
    // neither for `pair`, and `quoteIsToken0` is `quote < token` -- a property
    // of two addresses, never compared against what the pair actually holds.
    // The comment on `claimOnMarket` says asking for the wrong output "asks the
    // pool to pay out what is being paid in. Hence `quoteIsToken0`, fixed at
    // construction", which is true of the flag and not of the pair.
    //
    // `pair` is immutable, so this is not a setting that can be corrected after
    // the fact: it is decided once, by one of five constructor arguments, and
    // the contract has never been deployed.
    const alt1 = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
    const alt2 = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
    const wrongPair = await deploy(vm, OPERATOR, art.pair, [alt1, alt2]);
    const tok2 = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);
    const quo2 = await deploy(vm, OPERATOR, art.token, [10n ** 27n]);

    let badDist = null;
    try {
      badDist = await deploy(vm, OPERATOR, art.dist, [tok2, OPERATOR, wrongPair, quo2, factory]);
    } catch {
      badDist = null;
    }
    ok(badDist !== null, "a distributor is constructed with a pair for two tokens it never mentions");

    if (badDist) {
      await call(vm, OPERATOR, quo2, art.token, "approve", [badDist, 10n ** 27n]);
      const tw = build([{ account: A, amount: 4n * ONE }]);
      const openw = await call(vm, OPERATOR, badDist, art.dist, "openEpochOnMarket",
        [tw.root, 8n * ONE, 0n, 200_000n], { block: at(1000n) });
      ok(openw.ok, `and it takes funding for a market epoch against that pair${openw.ok ? "" : "  (" + openw.reason + ")"}`);
      const clw = await call(vm, A, badDist, art.dist, "claimOnMarket",
        [0, 4n * ONE, tw.proof(A), 1n], { block: at(1001n) });
      ok(!clw.ok, `and then nobody can ever claim it (${clw.reason})`);
    }

    // ---- checkClaim does not ask what mode the epoch is
    //
    // It is the miner-facing helper and `deploy.mjs` calls it. It answers for
    // the deadline, the claim record, the gate, the proof and solvency -- and
    // not for `mode`, so it says yes to a claim the contract will refuse.
    const t2 = build([{ account: C, amount: 2n * ONE }]);
    await call(vm, OPERATOR, quote, art.token, "approve", [dist, 10n ** 27n]);
    await call(vm, OPERATOR, dist, art.dist, "openEpochOnMarket",
      [t2.root, 4n * ONE, 0n, 300_000n], { block: at(2000n) });
    const id2 = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const chk = await call(vm, OPERATOR, dist, art.dist, "checkClaim",
      [id2, C, 2n * ONE, t2.proof(C)], { block: at(2001n) });
    ok(chk.ok && chk.result[0] === true, "checkClaim calls a market epoch's claim good");
    const wrongWay = await call(vm, C, dist, art.dist, "claim",
      [id2, 2n * ONE, t2.proof(C)], { block: at(2001n) });
    ok(!wrongWay.ok && wrongWay.reason === "WrongMode",
      `and claim() then refuses the very claim it approved (${wrongWay.reason})`);

    // ---- one root, two epochs, paid twice
    //
    // The leaf is `(account, amount)` and carries no epoch, no chain and no
    // contract; `hasClaimed` is keyed per epoch. So the same published root
    // opened twice is not a duplicate, it is a second entitlement to the same
    // work. Nothing on-chain can know the difference, which puts the check in
    // the tool: `deploy.mjs open` should read every existing root and refuse.
    await call(vm, OPERATOR, token, art.token, "setTax", [0, STRANGER]);
    await call(vm, OPERATOR, token, art.token, "mint", [OPERATOR, 100n * ONE]);
    await call(vm, OPERATOR, token, art.token, "approve", [dist, 10n ** 27n]);
    const t5 = build([{ account: D, amount: 1n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t5.root, 1n * ONE, 0n, 400_000n], { block: at(3000n) });
    const idA = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t5.root, 1n * ONE, 0n, 400_000n], { block: at(3000n) });
    const idB = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const d1 = await call(vm, D, dist, art.dist, "claim", [idA, 1n * ONE, t5.proof(D)], { block: at(3001n) });
    const d2 = await call(vm, D, dist, art.dist, "claim", [idB, 1n * ONE, t5.proof(D)], { block: at(3001n) });
    ok(d1.ok && d2.ok, "one root opened as two epochs pays the same leaf twice");

    // ---- the Direct path publishes a number it did not send
    //
    // Every inbound leg measures what arrived, and the header argues at length
    // for it: a contract that assumes an untaxed transfer "hands out claims it
    // cannot pay, and the failure lands on whoever claims last". Both market
    // claims measure the claimant's balance either side. `claim()` does not --
    // it emits `Claimed(..., amount, amount)`.
    //
    // The solvency half of that fear turns out **not** to materialise with this
    // tax model, and that is worth recording: the cut comes out of the amount
    // transferred rather than in addition to it, so the contract's balance
    // falls by exactly what its bookkeeping says. What is wrong is the number
    // in the event, which is what an indexer and a miner read.
    await call(vm, OPERATOR, token, art.token, "setTax", [300, STRANGER]);
    const t6 = build([{ account: C, amount: 2n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t6.root, 4n * ONE, 0n, 410_000n], { block: at(3100n) });
    const id6 = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const bal6a = await call(vm, OPERATOR, token, art.token, "balanceOf", [C]);
    const cl6 = await call(vm, C, dist, art.dist, "claim", [id6, 2n * ONE, t6.proof(C)], { block: at(3101n) });
    const bal6b = await call(vm, OPERATOR, token, art.token, "balanceOf", [C]);
    const arrived = BigInt(bal6b.result) - BigInt(bal6a.result);
    let reported = null;
    const di = new ethers.Interface(art.dist.abi);
    for (const lg of cl6.logs ?? []) {
      try {
        const p = di.parseLog({
          topics: lg[1].map((t) => ethers.hexlify(t)),
          data: ethers.hexlify(lg[2]),
        });
        if (p && p.name === "Claimed") reported = p.args.received;
      } catch {
        /* not ours */
      }
    }
    ok(cl6.ok && reported !== null && BigInt(reported) !== arrived,
      `claim() reports ${reported} received where ${arrived} arrived`);

    // And the contract is still solvent afterwards, which is the half of the
    // same worry that does not hold. Stated as a claim so it cannot rot.
    const held6 = await call(vm, OPERATOR, token, art.token, "balanceOf", [dist]);
    const ep6 = await call(vm, OPERATOR, dist, art.dist, "epochs", [id6]);
    ok(BigInt(held6.result) >= BigInt(ep6.result.funded) - BigInt(ep6.result.claimed),
      "and it still holds at least what that epoch's bookkeeping owes");
    await call(vm, OPERATOR, token, art.token, "setTax", [0, STRANGER]);

    // ---- the deadline, at exactly the deadline
    //
    // Claim requires `!(ts > deadline)` and reclaim `!(ts <= deadline)`, so the
    // two windows are disjoint and exhaustive and the boundary belongs to the
    // claimant. Off-by-one is the specific way a pair of checks like this goes
    // wrong, and the suite tested either side of the deadline but not the
    // instant itself.
    const t7 = build([{ account: B, amount: 1n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t7.root, 2n * ONE, 0n, 500_000n], { block: at(4000n) });
    const id7 = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const onTime = await call(vm, B, dist, art.dist, "claim",
      [id7, 1n * ONE, t7.proof(B)], { block: at(500_000n) });
    ok(onTime.ok, `a claim at exactly the deadline is admitted${onTime.ok ? "" : "  (" + onTime.reason + ")"}`);
    const rcSame = await call(vm, OPERATOR, dist, art.dist, "reclaim", [id7], { block: at(500_000n) });
    ok(!rcSame.ok && rcSame.reason === "EpochOpen",
      `and a reclaim at that same instant is refused (${rcSame.reason})`);
    const rcNext = await call(vm, OPERATOR, dist, art.dist, "reclaim", [id7], { block: at(500_001n) });
    ok(rcNext.ok, `and allowed one second later${rcNext.ok ? "" : "  (" + rcNext.reason + ")"}`);

    const t7b = build([{ account: C, amount: 1n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpoch",
      [t7b.root, 2n * ONE, 0n, 510_000n], { block: at(4000n) });
    const id7b = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const tooLate = await call(vm, C, dist, art.dist, "claim",
      [id7b, 1n * ONE, t7b.proof(C)], { block: at(510_001n) });
    ok(!tooLate.ok && tooLate.reason === "EpochClosed",
      `and a claim one second past it is refused (${tooLate.reason})`);

    // ---- `_inFlight` is set and cleared, but never saved and restored
    //
    // The V3 pool pays the recipient *before* it calls back, so a reward token
    // with a transfer hook runs code inside the outer swap. If that code enters
    // `claimOnV3` again and succeeds, the inner call clears `_inFlight` on its
    // way out and the outer pool's callback then finds a zero and is refused.
    //
    // Funds do not move -- the whole transaction unwinds -- so this is denial
    // rather than theft, and it needs a reward token the operator chose. It is
    // still worth one line to fix: keep the previous value and put it back.
    const tY = build([{ account: reward, amount: 1n * ONE }]);
    await call(vm, OPERATOR, quote, art.token, "approve", [dist, 10n ** 27n]);
    await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
      [tY.root, 2n * ONE, 0n, 600_000n, v3, reward], { block: at(5000n) });
    const idY = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const tX = build([{ account: A, amount: 1n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
      [tX.root, 2n * ONE, 0n, 600_000n, v3, reward], { block: at(5000n) });
    const idX = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;

    // Unhooked first, so the failure below is attributable to the hook and not
    // to the fixture. A comparison needs both halves.
    const plain = await call(vm, A, dist, art.dist, "claimOnV3",
      [idX, 1n * ONE, tX.proof(A), 1n], { block: at(5001n) });
    ok(plain.ok, `a V3 claim succeeds with no hook in the way${plain.ok ? "" : "  (" + plain.reason + ")"}`);

    const nested = di.encodeFunctionData("claimOnV3", [idY, 1n * ONE, tY.proof(reward), 1n]);
    await call(vm, OPERATOR, reward, art.token, "setHook", [dist, nested]);
    const tZ = build([{ account: B, amount: 1n * ONE }]);
    await call(vm, OPERATOR, dist, art.dist, "openEpochOnV3",
      [tZ.root, 2n * ONE, 0n, 600_000n, v3, reward], { block: at(5000n) });
    const idZ = Number((await call(vm, OPERATOR, dist, art.dist, "epochCount")).result) - 1;
    const hooked = await call(vm, B, dist, art.dist, "claimOnV3",
      [idZ, 1n * ONE, tZ.proof(B), 1n], { block: at(5002n) });
    ok(!hooked.ok && hooked.reason === "BadCallback",
      `and fails once a nested claim has cleared _inFlight (${hooked.reason})`);
    await call(vm, OPERATOR, reward, art.token, "setHook", [ethers.ZeroAddress, "0x"]);

    // ---- the operator's tool agrees with the contract about the constructor
    //
    // `deploy.mjs` passed four arguments where the constructor takes five --
    // it never learned about `v3Factory` -- so `node deploy.mjs deploy` threw
    // `types/values length mismatch` before it reached the network. That is the
    // first command of the whole runbook and the one that can only be run once.
    //
    // It is checked here rather than by running the tool because running it
    // needs a key and a chain, and the thing that was wrong needs neither: the
    // ABI says how many arguments there are and the source says how many are
    // passed. A test that needed the network would not have existed, and this
    // shipped for exactly that reason.
    const ctorInputs = art.dist.abi.find((x) => x.type === "constructor").inputs.length;
    const src = fs.readFileSync(path.join(here, "..", "deploy.mjs"), "utf8");
    const calls = [...src.matchAll(/factory\.(?:getDeployTransaction|deploy)\(([^)]*)\)/g)]
      .map((m) => m[1].split(",").map((x) => x.trim()).filter(Boolean).length);
    ok(calls.length > 0, "deploy.mjs constructs the distributor somewhere");
    ok(calls.every((n) => n === ctorInputs),
      `and passes ${ctorInputs} constructor argument(s) everywhere it does` +
      (calls.every((n) => n === ctorInputs) ? "" : `  (found ${calls.join(", ")})`));

    // And that it can claim from every mode it can open, which is the other
    // half of the same class: the tool opened `MarketV3` epochs and called
    // `claimOnMarket` unconditionally, so it could not claim from what it had
    // just opened.
    // A word boundary, not `includes`: "dist.openEpoch" is a prefix of
    // "dist.openEpochOnMarket", so a substring test reports Direct as supported
    // by the very line that implements Market. Caught by this check claiming
    // three modes when `open` could create two.
    const opens = ["openEpoch", "openEpochOnMarket", "openEpochOnV3"]
      .filter((f) => new RegExp("dist\\." + f + "\\(").test(src));
    const claims = ["claim", "claimOnMarket", "claimOnV3"]
      .filter((f) => new RegExp("dist\\." + f + "\\(").test(src));
    ok(claims.length === 3,
      `deploy.mjs can claim from all three modes (has ${claims.join(", ") || "none"})`);
    ok(opens.length >= 2,
      `and opens the modes it claims (${opens.join(", ") || "none"})`);
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  fs.writeFileSync(path.join(here, "..", "out", "root.txt"), tree.root + "\n");
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
