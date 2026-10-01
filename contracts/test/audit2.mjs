// The second, independent review of `GladosDistributor`: `design/audit-2.md`.
//
// Written by a reader who did not write the contract and did not write the
// first audit, against a threat list fixed before the code was read. Every
// item on that list ends here as either
//
//   FINDING  -- a claim that holds *because the defect is present*. Run with
//               AUDIT2_EXPECT_FIXED=1 and each fixable one is asserted the
//               other way round, so it fails on the current source and passes
//               on the patched copy the report proposes. That is what makes it
//               a regression test rather than a description.
//   CLEAN    -- a suspicion that was settled the other way, kept because a
//               negative nobody wrote down gets re-investigated forever.
//
// AUDIT2_DIST=<path> compiles a different distributor source (the scratch copy
// carrying the proposed fixes) in place of `src/GladosDistributor.sol`.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import solc from "solc";
import { ethers } from "ethers";
import { hexToBytes, bytesToHex, setLengthLeft } from "@ethereumjs/util";
import { makeVm, deploy, call, fund, addr } from "./evm.mjs";
import { build, leaf } from "./merkle.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const EXPECT_FIXED = process.env.AUDIT2_EXPECT_FIXED === "1";
const DIST_SRC = process.env.AUDIT2_DIST || path.join(root, "src", "GladosDistributor.sol");

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
/// A finding. `present` is whether the defect showed in this run. `fixable`
/// findings invert under AUDIT2_EXPECT_FIXED; the design ones (no code change
/// can remove them) are asserted present either way.
function finding(id, present, what, fixable = true) {
  if (EXPECT_FIXED && fixable) ok(!present, `${id} fixed: ${what}`);
  else ok(present, `${id} present: ${what}`);
}
function clean(id, cond, what) {
  ok(cond, `${id} clean: ${what}`);
}

function compileAll() {
  const read = (p) => fs.readFileSync(p, "utf8");
  const sources = {
    "GladosDistributor.sol": { content: read(DIST_SRC) },
    "TestToken.sol": { content: read(path.join(root, "src", "TestToken.sol")) },
    "MockPair.sol": { content: read(path.join(root, "src", "MockPair.sol")) },
    "MockV3.sol": { content: read(path.join(root, "src", "MockV3.sol")) },
    "Audit2Fixtures.sol": { content: read(path.join(here, "fixtures", "Audit2Fixtures.sol")) },
  };
  const input = {
    language: "Solidity",
    sources,
    settings: {
      optimizer: { enabled: true, runs: 200 },
      evmVersion: "paris",
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object", "storageLayout"] } },
    },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  const errs = (out.errors || []).filter((e) => e.severity === "error");
  if (errs.length) {
    for (const e of errs) console.error(e.formattedMessage);
    throw new Error("compile failed");
  }
  const a = (f, n) => ({ abi: out.contracts[f][n].abi, bytecode: out.contracts[f][n].evm.bytecode.object,
                         layout: out.contracts[f][n].storageLayout });
  return {
    dist: a("GladosDistributor.sol", "GladosDistributor"),
    token: a("TestToken.sol", "TestToken"),
    pair: a("MockPair.sol", "MockPair"),
    v3pool: a("MockV3.sol", "MockV3Pool"),
    v3factory: a("MockV3.sol", "MockV3Factory"),
    odd: a("Audit2Fixtures.sol", "OddToken"),
    evil: a("Audit2Fixtures.sol", "EvilV3Pool"),
    tryc: a("Audit2Fixtures.sol", "TryClaimer"),
    borrower: a("Audit2Fixtures.sol", "GateBorrower"),
  };
}

function at(ts) {
  return { header: { timestamp: ts, number: 1n, cliqueSigner: () => ({ toString: () => "0x0" }) } };
}

const OP = "0x1111111111111111111111111111111111111111";
const EVE = "0x2222222222222222222222222222222222222222";
const A = "0x00000000000000000000000000000000000000aa";
const B = "0x00000000000000000000000000000000000000bb";
const C = "0x00000000000000000000000000000000000000cc";
const X1 = "0x00000000000000000000000000000000000000e1";
const X2 = "0x00000000000000000000000000000000000000e2";
const X3 = "0x00000000000000000000000000000000000000e3";
const ONE = 10n ** 18n;
const GATE = 50_000n * ONE;
const T0 = 1000n;
const coder = ethers.AbiCoder.defaultAbiCoder();

async function main() {
  const art = compileAll();
  const vm = await makeVm();
  for (const a of [OP, EVE, A, B, C, X1, X2, X3]) await fund(vm, a);

  const bal = async (tok, who) => BigInt((await call(vm, OP, tok, art.token, "balanceOf", [who])).result);
  const count = async (d) => Number((await call(vm, OP, d, art.dist, "epochCount")).result);
  const epoch = async (d, id) => (await call(vm, OP, d, art.dist, "epochs", [id])).result;
  const slot0 = async (d) => {
    const v = await vm.stateManager.getContractStorage(addr(d), setLengthLeft(hexToBytes("0x00"), 32));
    return v.length === 0 ? 0n : BigInt(bytesToHex(v));
  };
  const inflightSlot = art.dist.layout.storage.find((s) => s.label === "_inFlight");
  ok(inflightSlot && inflightSlot.slot === "0", "_inFlight is storage slot 0, so it can be read directly");

  /// What every live epoch still owes, per asset, from the contract's own
  /// bookkeeping. The solvency invariant is that the contract holds at least
  /// this much of each asset; every section below re-checks it.
  async function owed(d, tokenAddr, quoteAddr) {
    const n = await count(d);
    const o = { [tokenAddr.toLowerCase()]: 0n };
    if (quoteAddr) o[quoteAddr.toLowerCase()] = 0n;
    for (let i = 0; i < n; i++) {
      const e = await epoch(d, i);
      if (e.reclaimed) continue;
      const asset = Number(e.mode) === 0 ? tokenAddr : quoteAddr;
      o[asset.toLowerCase()] += BigInt(e.funded) - BigInt(e.claimed);
    }
    return o;
  }
  async function solvent(d, tokenAddr, quoteAddr) {
    const o = await owed(d, tokenAddr, quoteAddr);
    for (const [asset, need] of Object.entries(o)) {
      const have = await bal(asset, d);
      if (have < need) return { ok: false, asset, have, need };
    }
    return { ok: true };
  }

  // ------------------------------------------------------------------ fixture
  const G = await deploy(vm, OP, art.token, [10n ** 30n]); // GLADOS stand-in: gate and Direct reward
  const Q = await deploy(vm, OP, art.token, [10n ** 30n]); // quote (WETH stand-in)
  const R = await deploy(vm, OP, art.token, [10n ** 30n]); // a V3 reward (NVDA stand-in)
  const pair = await deploy(vm, OP, art.pair, [Q, G]);
  const factory = await deploy(vm, OP, art.v3factory, []);
  const v3 = await deploy(vm, OP, art.v3pool, [Q, R, 500]);
  await call(vm, OP, factory, art.v3factory, "record", [v3, Q, R, 500]);
  await call(vm, OP, Q, art.token, "mint", [v3, 1_000n * ONE]);
  await call(vm, OP, R, art.token, "mint", [v3, 1_000n * ONE]);
  const evil = await deploy(vm, EVE, art.evil, [Q, R, 3000]);
  await call(vm, EVE, factory, art.v3factory, "record", [evil, Q, R, 3000]);
  await call(vm, OP, R, art.token, "mint", [evil, 1_000n * ONE]);

  await call(vm, OP, Q, art.token, "approve", [pair, 10n ** 30n]);
  await call(vm, OP, G, art.token, "approve", [pair, 10n ** 30n]);
  const q0 = BigInt(Q) < BigInt(G);
  await call(vm, OP, pair, art.pair, "seed", q0 ? [100n * ONE, 1_000_000n * ONE] : [1_000_000n * ONE, 100n * ONE]);

  const dist = await deploy(vm, OP, art.dist, [G, OP, pair, Q, factory]);
  await call(vm, OP, G, art.token, "approve", [dist, 10n ** 30n]);
  await call(vm, OP, Q, art.token, "approve", [dist, 10n ** 30n]);
  for (const w of [A, B, C]) await call(vm, OP, G, art.token, "mint", [w, GATE]);
  const di = new ethers.Interface(art.dist.abi);

  const open = async (fn, t, amount, gate, deadline, extra = []) => {
    const r = await call(vm, OP, dist, art.dist, fn, [t.root, amount, gate, deadline, ...extra], { block: at(T0) });
    if (!r.ok) throw new Error(`${fn} failed: ${r.reason}`);
    return (await count(dist)) - 1;
  };

  // ======================================================== 1. the V3 callback
  console.log("\n-- 1. uniswapV3SwapCallback authorisation");
  {
    eq0(await slot0(dist), "_inFlight is zero after construction");
    const t = build([{ account: A, amount: 2n * ONE }]);
    const id = await open("openEpochOnV3", t, 2n * ONE, 0n, 100_000n, [v3, R]);
    const r = await call(vm, A, dist, art.dist, "claimOnV3", [id, 2n * ONE, t.proof(A), 1n], { block: at(2000n) });
    ok(r.ok, "a genuine V3 claim succeeds");
    eq0(await slot0(dist), "and _inFlight is zero again after it returns");

    // A reverting claim inside a try/catch: the only arrangement in which a
    // leaked authorisation could outlive the call that set it.
    const tc = await deploy(vm, OP, art.tryc, []);
    const t2 = build([{ account: tc, amount: 2n * ONE }]);
    const id2 = await open("openEpochOnV3", t2, 2n * ONE, 0n, 100_000n, [v3, R]);
    await call(vm, OP, v3, art.v3pool, "setShortfall", [9000n]);
    const tr = await call(vm, EVE, tc, art.tryc, "tryClaimV3", [dist, id2, 2n * ONE, t2.proof(tc), 2n * ONE],
      { block: at(2000n) });
    await call(vm, OP, v3, art.v3pool, "setShortfall", [0n]);
    clean("C1.1", tr.ok && tr.result === false,
      "a claim that reverts inside a caller's try/catch leaves the caller's transaction standing");
    eq0(await slot0(dist), "and _inFlight is zero after it -- the revert undid the write");
    const hc = await call(vm, OP, dist, art.dist, "hasClaimed", [id2, tc]);
    clean("C1.2", hc.result === false, "and the reverted claim is not marked used");

    // The pool itself, which the factory vouches for, calling back outside any swap.
    const qBefore = await bal(Q, dist);
    const cb = di.encodeFunctionData("uniswapV3SwapCallback", [1n * ONE, 0n, "0x"]);
    const poke = await call(vm, EVE, evil, art.evil, "poke", [dist, cb], { block: at(2000n) });
    clean("C1.3", poke.ok && poke.result[0] === false,
      "a factory-vouched pool calling the callback outside a swap is refused");
    clean("C1.4", (await bal(Q, dist)) === qBefore, "and nothing left the contract");

    // During a genuine swap with `v3`, a *different* vouched pool calls back.
    // The reward's transfer hook fires when `v3` pays the claimant, which is
    // inside the window where `_inFlight` is non-zero.
    const t3 = build([{ account: B, amount: 2n * ONE }]);
    const id3 = await open("openEpochOnV3", t3, 2n * ONE, 0n, 100_000n, [v3, R]);
    const hookData = new ethers.Interface(art.evil.abi).encodeFunctionData("poke", [dist, cb]);
    await call(vm, OP, R, art.token, "setHook", [evil, hookData]);
    const qb = await bal(Q, dist);
    const r3 = await call(vm, B, dist, art.dist, "claimOnV3", [id3, 2n * ONE, t3.proof(B), 1n], { block: at(2000n) });
    await call(vm, OP, R, art.token, "setHook", [ethers.ZeroAddress, "0x"]);
    clean("C1.5", r3.ok && qb - (await bal(Q, dist)) === 2n * ONE && (await bal(Q, evil)) === 0n,
      "a second vouched pool calling back mid-swap is refused; exactly the claim's amount left");
    eq0(await slot0(dist), "and _inFlight is zero afterwards");
  }

  // Findings about what the callback trusts once a pool *is* in flight. On a
  // distributor of their own, because F3 leaves it insolvent by design and the
  // sections after this one check solvency on the main one.
  {
    const dist = await deploy(vm, OP, art.dist, [G, OP, pair, Q, factory]);
    await call(vm, OP, Q, art.token, "approve", [dist, 10n ** 30n]);
    const open = async (fn, t, amount, gate, deadline, extra = []) => {
      const r = await call(vm, OP, dist, art.dist, fn, [t.root, amount, gate, deadline, ...extra], { block: at(T0) });
      if (!r.ok) throw new Error(`${fn} failed: ${r.reason}`);
      return (await count(dist)) - 1;
    };
    // A second quote epoch, so there is somebody else's money to lose.
    const bystander = build([{ account: C, amount: 10n * ONE }]);
    await open("openEpochOnMarket", bystander, 10n * ONE, 0n, 100_000n);

    const t = build([{ account: A, amount: 2n * ONE }]);
    const id = await open("openEpochOnV3", t, 2n * ONE, 0n, 100_000n, [evil, R]);
    await call(vm, EVE, evil, art.evil, "set", [30_000n, 1n]); // asks three times the input
    const qb = await bal(Q, dist);
    const r = await call(vm, A, dist, art.dist, "claimOnV3", [id, 2n * ONE, t.proof(A), 1n], { block: at(2000n) });
    const took = qb - (await bal(Q, dist));
    const s = await solvent(dist, G, Q);
    finding("F3", r.ok && took === 6n * ONE && !s.ok,
      `a vouched pool that over-asks is paid ${took} for a 2e18 claim, from other epochs' quote` +
      (s.ok ? "" : ` (holds ${s.have}, owes ${s.need})`));

    const t2 = build([{ account: B, amount: 2n * ONE }]);
    const id2 = await open("openEpochOnV3", t2, 2n * ONE, 0n, 100_000n, [evil, R]);
    await call(vm, EVE, evil, art.evil, "set", [10_000n, 2n]); // honest amount, called back twice
    const qb2 = await bal(Q, dist);
    const r2 = await call(vm, B, dist, art.dist, "claimOnV3", [id2, 2n * ONE, t2.proof(B), 1n], { block: at(2000n) });
    const took2 = qb2 - (await bal(Q, dist));
    finding("F3", r2.ok && took2 === 4n * ONE,
      `and one that calls back twice in one swap is paid twice (${took2} for 2e18)`);

    // Partial fill: an honest pool whose liquidity runs out before the input
    // does asks for less than `amountSpecified`. The epoch is charged the leaf.
    const t3 = build([{ account: C, amount: 2n * ONE }]);
    const id3 = await open("openEpochOnV3", t3, 2n * ONE, 0n, 100_000n, [evil, R]);
    await call(vm, EVE, evil, art.evil, "set", [5_000n, 1n]);
    const qb3 = await bal(Q, dist);
    const r3 = await call(vm, C, dist, art.dist, "claimOnV3", [id3, 2n * ONE, t3.proof(C), 1n], { block: at(2000n) });
    const took3 = qb3 - (await bal(Q, dist));
    const e3 = await epoch(dist, id3);
    finding("F2", r3.ok && took3 === 1n * ONE && BigInt(e3.claimed) === 2n * ONE,
      `a partial fill spends ${took3} but charges the epoch ${e3.claimed}; the rest is stranded`);
    const opQ = await bal(Q, OP);
    const rc = await call(vm, OP, dist, art.dist, "reclaim", [id3], { block: at(100_001n) });
    const back = (await bal(Q, OP)) - opQ;
    finding("F2", r3.ok && rc.ok && back === 0n,
      `and reclaim returns ${back} of it, because the bookkeeping says nothing is left`);
    await call(vm, EVE, evil, art.evil, "set", [10_000n, 1n]);
  }

  // ============================================================ 2. the tree
  console.log("\n-- 2. Merkle leaf encoding and proofs");
  {
    // The second-preimage shape: a node whose 64-byte preimage is exactly
    // abi.encode(account, amount). Under a singly-hashed leaf the attacker
    // presents (account, amount) and the sibling; here the extra hash moves
    // the leaf out of the internal-node domain.
    const amt = 12345n;
    const inner = ethers.keccak256(coder.encode(["address", "uint256"], [A, amt]));
    const sib = ethers.keccak256("0x1234");
    const [lo, hi] = BigInt(inner) <= BigInt(sib) ? [inner, sib] : [sib, inner];
    const rootN = ethers.keccak256(coder.encode(["bytes32", "bytes32"], [lo, hi]));
    const v = await call(vm, OP, dist, art.dist, "verifyProof", [[sib], rootN, A, amt]);
    clean("C2.1", v.ok && v.result === false,
      "an internal node whose preimage is abi.encode(account, amount) cannot be claimed as a leaf");
    const lf = await call(vm, OP, dist, art.dist, "leafOf", [A, amt]);
    clean("C2.2", lf.result === ethers.keccak256(inner), "the leaf is keccak(keccak(abi.encode(a, amt))), 32-byte preimage");

    const es = [];
    for (let i = 0; i < 5; i++) es.push({ account: ethers.getAddress("0x" + (0x100 + i).toString(16).padStart(40, "0")), amount: BigInt(i + 1) });
    const t = build(es);
    const p = t.proof(es[0].account);
    const vf = async (proof, a = es[0].account, m = es[0].amount) =>
      (await call(vm, OP, dist, art.dist, "verifyProof", [proof, t.root, a, m])).result;
    clean("C2.3", (await vf(p)) === true, "the honest proof verifies (the control)");
    clean("C2.4", (await vf(p.slice(0, -1))) === false, "a truncated proof does not");
    clean("C2.5", (await vf([...p, ethers.ZeroHash])) === false, "an extended proof does not");
    clean("C2.6", (await vf([p[0], ...p])) === false, "a proof with an element repeated does not");
    clean("C2.7", (await vf([])) === false, "an empty proof against a many-leaf root does not");
    clean("C2.8", (await vf(p, es[1].account, es[1].amount)) === false, "a leaf's proof does not serve its sibling");
    clean("C2.9", (await vf(p, es[0].account, es[0].amount + (1n << 160n))) === false,
      "an amount differing only above bit 160 is a different leaf");
  }

  // ============================================================ 3. the gate
  console.log("\n-- 3. the holding gate");
  {
    // One gate's worth of GLADOS, three leaves. Wallet-to-wallet transfers of
    // the real token are untaxed (design/token.md, and live logs), so passing
    // it along costs gas and nothing else.
    const t = build([{ account: X1, amount: ONE }, { account: X2, amount: ONE }, { account: X3, amount: ONE }]);
    const id = await open("openEpoch", t, 3n * ONE, GATE, 100_000n);
    await call(vm, OP, G, art.token, "mint", [X1, GATE]);
    const c1 = await call(vm, X1, dist, art.dist, "claim", [id, ONE, t.proof(X1)], { block: at(2000n) });
    await call(vm, X1, G, art.token, "transfer", [X2, GATE]);
    const c2 = await call(vm, X2, dist, art.dist, "claim", [id, ONE, t.proof(X2)], { block: at(2000n) });
    await call(vm, X2, G, art.token, "transfer", [X3, GATE]);
    const c3 = await call(vm, X3, dist, art.dist, "claim", [id, ONE, t.proof(X3)], { block: at(2000n) });
    finding("F1", c1.ok && c2.ok && c3.ok,
      "one gate-sized balance, passed wallet to wallet, lets every leaf through the gate", false);

    // And atomically, with no trade at all: a contract leaf borrows the gate
    // for one call from anybody who approves it.
    const bor = await deploy(vm, OP, art.borrower, []);
    const t2 = build([{ account: bor, amount: ONE }]);
    const id2 = await open("openEpoch", t2, ONE, GATE, 100_000n);
    await call(vm, X3, G, art.token, "approve", [bor, GATE]);
    const r = await call(vm, EVE, bor, art.borrower, "borrowAndClaim", [G, X3, GATE, dist, id2, ONE, t2.proof(bor)],
      { block: at(2000n) });
    finding("F1", r.ok && (await bal(G, bor)) === ONE,
      `a leaf holding nothing borrows the gate for one call and claims${r.ok ? "" : " (" + r.reason + ")"}`, false);
  }

  // ===================================================== 4. cross-epoch money
  console.log("\n-- 4. cross-epoch accounting");
  {
    const ta = build([{ account: A, amount: 10n * ONE }]);
    const tb = build([{ account: B, amount: 10n * ONE }]);
    const tc = build([{ account: C, amount: 10n * ONE }]); // over-promises: funded 5
    const ia = await open("openEpoch", ta, 10n * ONE, 0n, 100_000n);
    const ib = await open("openEpoch", tb, 10n * ONE, 0n, 100_000n);
    const ic = await open("openEpoch", tc, 5n * ONE, 0n, 100_000n);
    const r = await call(vm, C, dist, art.dist, "claim", [ic, 10n * ONE, tc.proof(C)], { block: at(2000n) });
    const held = await bal(G, dist);
    clean("C4.1", !r.ok && r.reason.startsWith("Insolvent") && held >= 25n * ONE,
      `an over-promised epoch is refused (${r.reason}) while the contract holds ${held} for others`);
    const ra = await call(vm, A, dist, art.dist, "claim", [ia, 10n * ONE, ta.proof(A)], { block: at(2000n) });
    const rb = await call(vm, B, dist, art.dist, "claim", [ib, 10n * ONE, tb.proof(B)], { block: at(2000n) });
    clean("C4.2", ra.ok && rb.ok, "and the correctly funded epochs beside it pay in full");
    const s = await solvent(dist, G, Q);
    clean("C4.3", s.ok, "the per-asset solvency invariant holds after mixed Direct/Market/V3 activity");
  }
  {
    // A sender-side fee: the token charges the *sender* on top of the amount.
    // Every inbound leg is measured; no outbound leg is. Not GLADOS's model --
    // its tax is on pair legs and comes out of the amount -- so this prices a
    // dependency, not a live bug.
    const odd = await deploy(vm, OP, art.odd, [10n ** 30n]);
    const d2 = await deploy(vm, OP, art.dist, [odd, OP, ethers.ZeroAddress, ethers.ZeroAddress, ethers.ZeroAddress]);
    await call(vm, OP, odd, art.odd, "approve", [d2, 10n ** 30n]);
    const ta = build([{ account: A, amount: 100n * ONE }]);
    const tb = build([{ account: B, amount: 100n * ONE }]);
    await call(vm, OP, d2, art.dist, "openEpoch", [ta.root, 100n * ONE, 0n, 100_000n], { block: at(T0) });
    await call(vm, OP, d2, art.dist, "openEpoch", [tb.root, 100n * ONE, 0n, 100_000n], { block: at(T0) });
    await call(vm, OP, odd, art.odd, "setSenderFee", [100n]);
    await call(vm, A, d2, art.dist, "claim", [0, 100n * ONE, ta.proof(A)], { block: at(2000n) });
    const s = await solvent(d2, odd, null);
    const rb = await call(vm, B, d2, art.dist, "claim", [1, 100n * ONE, tb.proof(B)], { block: at(2000n) });
    finding("F4", !s.ok && !rb.ok,
      `with a sender-side fee, epoch 0's claim spends epoch 1's funds (${s.have ?? "-"} held, ${s.need ?? "-"} owed; B: ${rb.reason ?? "paid"})`);
  }

  // ============================================================== 5. reclaim
  console.log("\n-- 5. reclaim bounds");
  {
    const ta = build([{ account: A, amount: 4n * ONE }, { account: B, amount: 6n * ONE }]);
    const tb = build([{ account: C, amount: 7n * ONE }]);
    const ia = await open("openEpoch", ta, 10n * ONE, 0n, 50_000n);
    const ib = await open("openEpoch", tb, 7n * ONE, 0n, 90_000n);
    await call(vm, A, dist, art.dist, "claim", [ia, 4n * ONE, ta.proof(A)], { block: at(2000n) });
    const early = await call(vm, OP, dist, art.dist, "reclaim", [ib], { block: at(60_000n) });
    clean("C5.1", early.reason === "EpochOpen", "reclaim of a live epoch is refused even while a neighbour has expired");
    const before = await bal(G, OP);
    const rc = await call(vm, OP, dist, art.dist, "reclaim", [ia], { block: at(60_000n) });
    clean("C5.2", rc.ok && (await bal(G, OP)) - before === 6n * ONE, "reclaim returns exactly the unclaimed remainder");
    const late = await call(vm, B, dist, art.dist, "claim", [ia, 6n * ONE, ta.proof(B)], { block: at(60_000n) });
    clean("C5.3", late.reason === "EpochClosed", "and a claim on the reclaimed epoch is refused");
    const cb = await call(vm, C, dist, art.dist, "claim", [ib, 7n * ONE, tb.proof(C)], { block: at(60_000n) });
    clean("C5.4", cb.ok, "the neighbouring live epoch still pays in full after the reclaim");
    const again = await call(vm, OP, dist, art.dist, "reclaim", [ia], { block: at(60_001n) });
    clean("C5.5", again.reason === "AlreadyReclaimed", "and the reclaimed epoch cannot be reclaimed again");
    const s = await solvent(dist, G, Q);
    clean("C5.6", s.ok, "solvency holds after reclaims");
  }

  // ======================================================= 6. measurement
  console.log("\n-- 6. amount-as-received measurement");
  {
    // An inbound transfer landing inside the pull is attributed to the epoch.
    // Only reachable through a token hook, which GLADOS has not; and the
    // direction is safe (the epoch promises what really arrived).
    await call(vm, OP, G, art.token, "mint", [G, 5n * ONE]);
    const donate = new ethers.Interface(art.token.abi).encodeFunctionData("transfer", [dist, 5n * ONE]);
    await call(vm, OP, G, art.token, "setHook", [G, donate]);
    const t = build([{ account: A, amount: 15n * ONE }]);
    const id = await open("openEpoch", t, 10n * ONE, 0n, 100_000n);
    await call(vm, OP, G, art.token, "setHook", [ethers.ZeroAddress, "0x"]);
    const e = await epoch(dist, id);
    clean("C6.1", BigInt(e.funded) === 15n * ONE && (await solvent(dist, G, Q)).ok,
      "an inbound transfer inside the pull is counted as funding, and the contract stays solvent");

    // Two epochs opened in the same block each record their own amount.
    const t1 = build([{ account: B, amount: 3n * ONE }]);
    const t2 = build([{ account: C, amount: 4n * ONE }]);
    const i1 = await open("openEpoch", t1, 3n * ONE, 0n, 100_000n);
    const i2 = await open("openEpoch", t2, 4n * ONE, 0n, 100_000n);
    clean("C6.2", BigInt((await epoch(dist, i1)).funded) === 3n * ONE && BigInt((await epoch(dist, i2)).funded) === 4n * ONE,
      "two epochs opened back to back in one block record 3 and 4, not a shared delta");

    // Tokens that arrive any other way -- GLADOS's own dividend contract pays
    // holders in GLADOS, and a distributor holding >= 1M GLADOS is one -- are
    // counted by nothing and recoverable by nothing.
    await call(vm, OP, G, art.token, "mint", [dist, 7n * ONE]);
    const n = await count(dist);
    for (let i = 0; i < n; i++) await call(vm, OP, dist, art.dist, "reclaim", [i], { block: at(200_000n) });
    const left = await bal(G, dist);
    finding("F6", left >= 7n * ONE,
      `after every epoch is reclaimed the contract still holds ${left} of the token, with no function to move it`, false);
  }

  // ============================================================ 7. _move
  console.log("\n-- 7. _move and the shapes of a transfer's answer");
  {
    const odd = await deploy(vm, OP, art.odd, [10n ** 30n]);
    const d = await deploy(vm, OP, art.dist, [odd, OP, ethers.ZeroAddress, ethers.ZeroAddress, ethers.ZeroAddress]);
    await call(vm, OP, odd, art.odd, "approve", [d, 10n ** 30n]);
    const t = build([{ account: A, amount: ONE }]);
    const openOdd = async (mode) => {
      await call(vm, OP, odd, art.odd, "setMode", [mode]);
      const r = await call(vm, OP, d, art.dist, "openEpoch", [t.root, ONE, 0n, 100_000n], { block: at(T0) });
      await call(vm, OP, odd, art.odd, "setMode", [0]);
      return r;
    };
    clean("C7.1", (await openOdd(1)).reason === "TransferFailed", "an explicit false is refused as TransferFailed");
    clean("C7.2", (await openOdd(2)).reason === "TransferFailed", "a one-byte answer is refused as TransferFailed");
    const two = await openOdd(3);
    finding("F7", !two.ok && two.reason !== "TransferFailed",
      `a 32-byte answer that is neither 0 nor 1 reverts with '${two.reason}' rather than TransferFailed`);
    clean("C7.3", (await openOdd(4)).ok, "64 bytes whose first word is true is accepted (the usual safe-transfer rule)");
    clean("C7.4", (await openOdd(5)).reason === "NothingFunded",
      "a transferFrom that returns nothing and moves nothing opens no epoch (the measurement catches it)");

    // Outbound, nothing measures a silent no-op against what was asked.
    const o = await call(vm, OP, d, art.dist, "openEpoch", [t.root, ONE, 0n, 100_000n], { block: at(T0) });
    const id = (await count(d)) - 1;
    await call(vm, OP, odd, art.odd, "setMode", [5]);
    const c = await call(vm, A, d, art.dist, "claim", [id, ONE, t.proof(A)], { block: at(2000n) });
    await call(vm, OP, odd, art.odd, "setMode", [0]);
    const used = (await call(vm, OP, d, art.dist, "hasClaimed", [id, A])).result;
    finding("F7", o.ok && c.ok && used === true && (await bal(odd, A)) === 0n,
      "a transfer that returns nothing and moves nothing marks the claim used and pays 0");
  }

  // ================================================= 8. modes, others, MEV
  console.log("\n-- 8. mode confusion, claims for others, slippage");
  {
    const tD = build([{ account: A, amount: ONE }]);
    const tM = build([{ account: A, amount: ONE }]);
    const tV = build([{ account: A, amount: ONE }]);
    const iD = await open("openEpoch", tD, ONE, 0n, 300_000n);
    const iM = await open("openEpochOnMarket", tM, ONE, 0n, 300_000n);
    const iV = await open("openEpochOnV3", tV, ONE, 0n, 300_000n, [v3, R]);
    const tries = [
      ["claim", iM, []], ["claim", iV, []],
      ["claimOnMarket", iD, [1n]], ["claimOnMarket", iV, [1n]],
      ["claimOnV3", iD, [1n]], ["claimOnV3", iM, [1n]],
    ];
    let allWrong = true;
    for (const [fn, id, extra] of tries) {
      const r = await call(vm, A, dist, art.dist, fn, [id, ONE, tD.proof(A), ...extra], { block: at(2000n) });
      if (r.reason !== "WrongMode") allWrong = false;
    }
    clean("C8.1", allWrong, "all six wrong-function/mode pairings revert WrongMode");

    await call(vm, OP, G, art.token, "mint", [EVE, GATE]);
    const tg = build([{ account: A, amount: ONE }]);
    const ig = await open("openEpoch", tg, ONE, GATE, 300_000n);
    const steal = await call(vm, EVE, dist, art.dist, "claim", [ig, ONE, tg.proof(A)], { block: at(2000n) });
    clean("C8.2", steal.reason === "BadProof",
      "a stranger holding the gate and replaying A's proof and amount is refused BadProof (the leaf binds msg.sender)");
    const byOp = await call(vm, OP, dist, art.dist, "claim", [ig, ONE, tg.proof(A)], { block: at(2000n) });
    clean("C8.3", !byOp.ok, "and so is the operator");
    const recipientParams = art.dist.abi.filter((f) => f.type === "function" && /^claim/.test(f.name))
      .some((f) => f.inputs.some((i) => i.type === "address"));
    clean("C8.4", !recipientParams, "no claim function takes a recipient, so a claim cannot be redirected");

    // A sandwich on a Market claim: bounded by the claimant's own minOut and
    // borne by the claimant. Priced here against the mock pair.
    const tS = build([{ account: B, amount: ONE }]);
    const iS = await open("openEpochOnMarket", tS, ONE, 0n, 300_000n);
    const quoteOut = async (amt) => {
      const rr = (await call(vm, OP, pair, art.pair, "getReserves")).result;
      const [rq, rt] = q0 ? [rr[0], rr[1]] : [rr[1], rr[0]];
      return (amt * 997n * rt) / (rq * 1000n + amt * 997n);
    };
    const fair = await quoteOut(ONE);
    // Front-run: EVE buys with 20 quote.
    await call(vm, OP, Q, art.token, "mint", [EVE, 20n * ONE]);
    const eOut = await quoteOut(20n * ONE);
    await call(vm, EVE, Q, art.token, "transfer", [pair, 20n * ONE]);
    await call(vm, EVE, pair, art.pair, "swap", q0 ? [0n, eOut, EVE, "0x"] : [eOut, 0n, EVE, "0x"]);
    const tight = await call(vm, B, dist, art.dist, "claimOnMarket", [iS, ONE, tS.proof(B), (fair * 98n) / 100n],
      { block: at(2000n) });
    clean("C8.5", tight.reason.startsWith("TooLittleOut"), "a 2% bound stops a claim sandwiched behind a 20-quote buy");
    const gb = await bal(G, B);
    const loose = await call(vm, B, dist, art.dist, "claimOnMarket", [iS, ONE, tS.proof(B), 1n], { block: at(2000n) });
    const got = (await bal(G, B)) - gb;
    finding("F8", loose.ok && got < (fair * 80n) / 100n,
      `with minOut=1 the same claim fills at ${(got * 100n) / fair}% of the fair output -- the claimant bears it`, false);
  }

  // =============================================== 9. what the operator can do
  console.log("\n-- 9. operator powers and surface");
  {
    const mutating = art.dist.abi.filter((f) => f.type === "function" && !["view", "pure"].includes(f.stateMutability))
      .map((f) => f.name).sort();
    const expected = ["claim", "claimOnMarket", "claimOnV3", "openEpoch", "openEpochOnMarket", "openEpochOnV3",
      "reclaim", "uniswapV3SwapCallback"].sort();
    clean("C9.1", JSON.stringify(mutating) === JSON.stringify(expected),
      `the only state-changing entry points are ${mutating.join(", ")}`);
    const payable = art.dist.abi.some((f) => f.stateMutability === "payable" || f.type === "receive" || f.type === "fallback");
    clean("C9.2", !payable, "nothing is payable and there is no receive/fallback, so ETH cannot be sent in a call");

    // V3 without a V2 pair: the constructor's comment says a chain may have a
    // V3 deployment and no V2 pair for this token, but `quote` can only be set
    // together with `pair`, and `openEpochOnV3` requires `quote`.
    let lone = null;
    try {
      lone = await deploy(vm, OP, art.dist, [G, OP, ethers.ZeroAddress, Q, factory]);
    } catch {
      lone = null;
    }
    let refused = lone === null;
    if (lone) {
      await call(vm, OP, Q, art.token, "approve", [lone, 10n ** 30n]);
      const t = build([{ account: A, amount: ONE }]);
      const r = await call(vm, OP, lone, art.dist, "openEpochOnV3", [t.root, ONE, 0n, 100_000n, v3, R], { block: at(T0) });
      refused = !r.ok;
    }
    finding("F9", refused,
      "a distributor with a quote and a V3 factory but no V2 pair cannot be built, so V3 needs a V2 pair");
  }

  console.log(`\n${passed} passed, ${failed} failed${EXPECT_FIXED ? "  (AUDIT2_EXPECT_FIXED)" : ""}`);
  process.exit(failed === 0 ? 0 : 1);

  function eq0(v, what) {
    ok(v === 0n, `${what}${v === 0n ? "" : `  (slot 0 = ${v})`}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
