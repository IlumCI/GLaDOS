// The whole loop against a fork of Robinhood Chain.
//
//   node test/fork.mjs <epoch.json> [--address 0x...]
//
// The real GLADOS contract, the real WETH/GLADOS pair, the real reserves and
// the real tax -- fetched over RPC as the EVM asks for them, so this is the
// chain as it actually is at the current block rather than a model of it. The
// distributor is deployed *into* the fork. Nothing is spent, nothing is
// published, and no key is needed.
//
// ### Why this is worth more than the mock
//
// `MockPair` proves the distributor's arithmetic against a pair that behaves
// the way this file's author believes a pair behaves. That is exactly the class
// of assumption this repository distrusts -- and it has already been wrong once
// here, in the mock's own `amountIn`. A fork removes the author from the
// question: the pair is the deployed bytecode, the tax is whatever the token
// does today, and the reserves are whatever the market left.
//
// What it still cannot prove is that anybody funded an epoch, and that the
// operator holds the key they think they hold.
import fs from "node:fs";
import { VM } from "@ethereumjs/vm";
import { Chain, Common, Hardfork } from "@ethereumjs/common";
import { RPCStateManager } from "@ethereumjs/statemanager";
import { Account, Address, hexToBytes, bytesToHex } from "@ethereumjs/util";
import { ethers } from "ethers";
import { compile } from "./build.mjs";

const RPC = process.env.GLADOS_RPC || "https://rpc.mainnet.chain.robinhood.com";
const TOKEN = "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777";
const PAIR = "0x93f777932d98d15b351d1bce8c76b34381eede5b";
const WETH = "0x0bd7d308f8e1639fab988df18a8011f41eacad73";
/// Uniswap's V3 factory on 4663, verified by reading its code and its
/// PoolCreated log rather than by the address looking canonical -- the
/// canonical V3 factory address on every other chain *also* has code here and
/// is not a factory, which is a trap worth one line of comment.
const V3_FACTORY = "0x1f7d7550B1b028f7571E69A784071F0205FD2EfA";

const args = process.argv.slice(2);
const file = args.find((x) => !x.startsWith("--"));
const iAddr = args.indexOf("--address");
const MINER = iAddr >= 0 ? ethers.getAddress(args[iAddr + 1]) : null;
const OPERATOR = "0x1111111111111111111111111111111111111111";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };

const coder = ethers.AbiCoder.defaultAbiCoder();
const addr = (h) => new Address(hexToBytes(h.toLowerCase()));

async function rpcBlockNumber() {
  const r = await fetch(RPC, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_blockNumber", params: [] }),
  });
  return parseInt((await r.json()).result, 16);
}

// **A reverted call here leaves its writes behind, which is a property of the
// harness and not of the chain.**
//
// Wrapping this in `vm.stateManager.checkpoint()` and reverting on failure was
// tried and does not fix it: `RPCStateManager` does not restore storage it has
// already fetched and written, so the flag a reverting `swap` clears stays
// cleared. What it looks like is a pair answering `UniswapV2: LOCKED` to every
// call after the first refusal -- a reentrancy guard that was taken and, because
// the revert did not unwind, never given back.
//
// So the rule for anything written here is **never ask for a refusal and then
// carry on**. A check that expects a revert either comes last, or is the only
// one of its kind in the run, and a sweep that needs several must assert the
// *reason* rather than the failure -- `K` is the pair's own answer and `LOCKED`
// is this harness's.
//
// It also means `ok(false, ...)` followed by more checks is weaker than it looks,
// which is why a partial run is already documented as no evidence rather than as
// a set of failures.
async function call(vm, from, to, data, value = 0n) {
  const res = await vm.evm.runCall({
    caller: addr(from), origin: addr(from), to: addr(to),
    data: hexToBytes(data), gasLimit: 30_000_000n, value,
  });
  return {
    ok: !res.execResult.exceptionError,
    ret: bytesToHex(res.execResult.returnValue),
    err: res.execResult.exceptionError?.error,
  };
}

async function deploy(vm, from, artifact, ctorArgs) {
  const iface = new ethers.Interface(artifact.abi);
  const enc = iface.deploy.inputs.length
    ? coder.encode(iface.deploy.inputs.map((i) => i.type), ctorArgs)
    : "0x";
  const res = await vm.evm.runCall({
    caller: addr(from), origin: addr(from), to: undefined,
    data: hexToBytes("0x" + artifact.bytecode + enc.slice(2)),
    gasLimit: 30_000_000n, value: 0n,
  });
  if (res.execResult.exceptionError) throw new Error("deploy: " + res.execResult.exceptionError.error);
  return bytesToHex(res.createdAddress.bytes);
}

// ### This is unreliable against the public RPC, and the failure is not ours
//
// `RPCStateManager` fetches each account lazily through `eth_getProof` and, when
// the provider does not answer, throws `Cannot read properties of undefined
// (reading 'balance')` from inside the EVM's gas handler. Robinhood Chain's
// public endpoint stops answering after enough calls in one run, so this file
// fails part-way through at a point that **moves between runs** -- sometimes at
// the miner's claim, sometimes at the burn.
//
// It is not a fault in the contracts and not in this harness's logic: the same
// code completed the claim leg and reported 420,575 GLADOS one run and died
// before it the next, against the same block.
//
// Two things were tried and neither is the answer. Seeding the addresses this
// file names -- the burn address, the stranger -- with `putAccount` and no read
// removes those fetches, and the failing one is a *callee* inside the token's own
// tax distribution, which this file cannot enumerate. And retrying would hide a
// real RPC failure behind a loop.
//
// What would fix it is an endpoint that answers every `eth_getProof`, which is
// what `GLADOS_RPC` is for. Until there is one, read a completed run as evidence
// and an incomplete one as no evidence, rather than as a failure.
async function main() {
  if (!file) {
    console.error("usage: node test/fork.mjs <epoch.json> [--address 0x...]");
    process.exit(2);
  }
  const doc = JSON.parse(fs.readFileSync(file, "utf8"));

  const block = await rpcBlockNumber();
  console.log(`forking Robinhood Chain at block ${block}`);
  const common = new Common({ chain: Chain.Mainnet, hardfork: Hardfork.Shanghai });
  const stateManager = new RPCStateManager({ provider: RPC, blockTag: BigInt(block) });
  const vm = await VM.create({ common, stateManager });

  // Sanity: is this really the chain we think it is?
  //
  // **Numbers rather than the symbol, and that is a harness limitation stated
  // rather than hidden.** `symbol()` and `name()` revert under
  // `RPCStateManager` while every numeric view answers correctly -- a string
  // longer than 31 bytes lives across keccak-derived storage slots and the
  // lazy fetcher does not follow it. Nothing to do with the token: `decimals`,
  // `totalSupply` and `buyTaxRate` all return exactly what the live node
  // returns for them.
  //
  // These are the better check anyway. A symbol is a label anybody can choose;
  // a supply of exactly 1e27 and a buy tax of 100 basis points are the two
  // facts this project has independently read from the chain and published.
  const sup = await call(vm, OPERATOR, TOKEN, "0x18160ddd");
  const supply = sup.ok ? coder.decode(["uint256"], sup.ret)[0] : 0n;
  ok(supply === 10n ** 27n, `the forked token's supply is ${supply / 10n ** 18n} whole tokens`);
  const tax = await call(vm, OPERATOR, TOKEN, "0x691f224f");
  const buyTax = tax.ok ? coder.decode(["uint256"], tax.ret)[0] : 0n;
  ok(buyTax === 100n, `and its buy tax is ${Number(buyTax) / 100}%`);

  const res = await call(vm, OPERATOR, PAIR, "0x0902f1ac");
  const [r0, r1] = coder.decode(["uint112", "uint112", "uint32"], res.ret);
  const quoteIsToken0 = BigInt(WETH) < BigInt(TOKEN);
  const [rw, rg] = quoteIsToken0 ? [r0, r1] : [r1, r0];
  ok(rw > 0n && rg > 0n,
     `the real pool holds ${ethers.formatEther(rw)} WETH against ${(Number(rg) / 1e24).toFixed(1)}M GLADOS`);

  // Everyone in the epoch, plus the operator, gets spendable ETH. This is the
  // one thing invented here, and it stands in for "the operator funded this".
  const claimants = Object.keys(doc.claims).map(ethers.getAddress);
  const who = MINER ? [MINER] : claimants;
  for (const a of [OPERATOR, ...who]) {
    const acc = (await vm.stateManager.getAccount(addr(a))) ?? new Account();
    acc.balance = 10n ** 20n;
    await vm.stateManager.putAccount(addr(a), acc);
  }

  // WETH by depositing ETH, so the operator's funding is real WETH from the
  // real contract rather than a balance written into storage by hand.
  const want = BigInt(doc.total);
  const dep = await call(vm, OPERATOR, WETH, "0xd0e30db0", want);
  ok(dep.ok, `the operator wraps ${ethers.formatEther(want)} ETH into real WETH${dep.ok ? "" : "  (" + dep.err + ")"}`);

  const art = compile(["GladosDistributor.sol", "TestToken.sol", "MockPair.sol",
                       "GladosBurner.sol"]);
  const dist = {
    abi: art["GladosDistributor.sol"].GladosDistributor.abi,
    bytecode: art["GladosDistributor.sol"].GladosDistributor.evm.bytecode.object,
  };
  const at = await deploy(vm, OPERATOR, dist, [TOKEN, OPERATOR, PAIR, WETH, V3_FACTORY]);
  console.log(`distributor deployed into the fork at ${at}`);

  const iface = new ethers.Interface(dist.abi);
  const erc20 = new ethers.Interface([
    "function approve(address,uint256)",
    "function balanceOf(address) view returns (uint256)",
  ]);
  const ap = await call(vm, OPERATOR, WETH, erc20.encodeFunctionData("approve", [at, want]));
  ok(ap.ok, "and approves the distributor to pull it");

  const openData = iface.encodeFunctionData("openEpochOnMarket",
    [doc.root, want, 0n, BigInt(Math.floor(Date.now() / 1000) + 86400)]);
  const opened = await call(vm, OPERATOR, at, openData);
  ok(opened.ok, `a market epoch opens with the published root${opened.ok ? "" : "  (" + opened.err + ")"}`);

  // ------------------------------------------------------------- claiming
  let anyPaid = false;
  for (const a of who) {
    const c = doc.claims[a.toLowerCase()];
    if (!c) {
      console.log(`      ${a} is not in this epoch`);
      continue;
    }
    const balBefore = await call(vm, a, TOKEN, erc20.encodeFunctionData("balanceOf", [a]));
    const before = coder.decode(["uint256"], balBefore.ret)[0];

    const data = iface.encodeFunctionData("claimOnMarket", [0, BigInt(c.amount), c.proof, 1n]);
    const r = await call(vm, a, at, data);
    if (!r.ok) {
      ok(false, `${a} claimed  (${r.err})`);
      continue;
    }
    const balAfter = await call(vm, a, TOKEN, erc20.encodeFunctionData("balanceOf", [a]));
    const got = coder.decode(["uint256"], balAfter.ret)[0] - before;
    anyPaid = anyPaid || got > 0n;
    // **"would have bought", not "bought".** This said "bought N real GLADOS" and
    // was read as a purchase, which it is not: the token and the pool are the
    // deployed ones and the WETH is invented by this harness, so the figure is
    // what *would* happen and the event did not. A log nobody has to interpret
    // is worth more than a shorter line.
    ok(got > 0n, `${a} would have bought ${ethers.formatEther(got)} GLADOS with `
       + `${ethers.formatEther(BigInt(c.amount))} WETH -- simulated, nothing moved`);
  }
  ok(anyPaid, "the real pool would have paid somebody in GLADOS (simulated)");

  // ------------------------------------------------- the pair's fee, measured
  //
  // **`design/audit.md`'s first stated dependency, measured at last.** The
  // contract writes `997/1000` into `_amountOut` because a V2 pair does not
  // expose its fee and there is no router here to ask. A fork of V2 on 4663 with
  // a different fee makes every `Market` claim either revert on the `k` check (a
  // higher real fee) or quietly shortchange the claimant (a lower one), and the
  // audit asked for this to be asserted here, where the real pair is.
  //
  // **A successful claim is only half the answer.** The distributor computes
  // `out` at 997/1000 and asks the pair for exactly that, so the pair pays what
  // it was asked rather than its maximum -- the claims above prove the fee is no
  // *higher* than 0.3% and say nothing about it being lower. Asking for what a
  // 0.2% fee would give and watching `k` refuse it closes the other side.
  //
  // **Two probes, in this order, for the reason `call` records.** A reverted call
  // does not unwind here, so the refusal must come last or the pair spends the
  // rest of the run answering `LOCKED`. The first version swept four numerators
  // descending, got a real `K` on the first and a manufactured `LOCKED` on the
  // other three, and reported them all as the pair's own refusals -- which is
  // the canary failure `differ.rs` describes, arriving on a fork.
  //
  // So the refusal is asserted by its **reason**. `K` is the pair's arithmetic;
  // anything else means this measured the harness.
  {
    const FEE_IN = 10n ** 15n;                    // 0.001 WETH per probe
    const xfer = new ethers.Interface(["function transfer(address,uint256) returns (bool)"]);
    const swapAbi = new ethers.Interface(["function swap(uint256,uint256,address,bytes)"]);
    const ask = (out) => swapAbi.encodeFunctionData("swap",
      quoteIsToken0 ? [0n, out, OPERATOR, "0x"] : [out, 0n, OPERATOR, "0x"]);
    const reason = (r) => {
      try {
        if (r.ret && r.ret.length > 10 && r.ret.startsWith("0x08c379a0"))
          return coder.decode(["string"], "0x" + r.ret.slice(10))[0];
      } catch { /* not a string revert */ }
      return r.err ?? "no reason";
    };
    // A fresh donation per probe, because a V2 swap consumes what it was sent
    // and a second probe sharing one donation measures an empty pair.
    const probe = async (num) => {
      const w = await call(vm, OPERATOR, WETH, "0xd0e30db0", FEE_IN);
      const r = await call(vm, OPERATOR, PAIR, "0x0902f1ac");
      const [p0, p1] = coder.decode(["uint112", "uint112", "uint32"], r.ret);
      const [RW, RG] = quoteIsToken0 ? [p0, p1] : [p1, p0];
      const inWithFee = FEE_IN * num;
      const out = (inWithFee * RG) / (RW * 1000n + inWithFee);
      const d = await call(vm, OPERATOR, WETH, xfer.encodeFunctionData("transfer", [PAIR, FEE_IN]));
      if (!w.ok || !d.ok) return { out, setup: false, res: null };
      return { out, setup: true, res: await call(vm, OPERATOR, PAIR, ask(out)) };
    };

    const fair = await probe(997n);
    ok(fair.setup && fair.res.ok,
       `the real pair pays ${fair.out} for ${FEE_IN} in, which is what a 0.3% fee gives`
       + (fair.setup ? (fair.res.ok ? "" : `  (${reason(fair.res)})`) : "  (the donation did not land)"));

    const greedy = await probe(998n);
    const why = greedy.setup && greedy.res ? reason(greedy.res) : "the donation did not land";
    ok(greedy.setup && !greedy.res.ok && why === "UniswapV2: K",
       `and refuses ${greedy.out}, which a 0.2% fee would give -- on its own k check`
       + (why === "UniswapV2: K" ? "" : `, but the reason was "${why}"`));
  }

  // ------------------------------------------- a real buy that nobody owns
  //
  // The same thing again, except the leaf is a `GladosBurner` and what it buys
  // goes to an address with no key. This is the leg that cannot be checked
  // against a mock: whether the *real* GLADOS token permits a transfer to
  // `0xdEaD` at all, and what its tax does on the way.
  //
  // A second epoch rather than reusing the first, because a leaf is an address
  // and the burner is a different address from the miner.
  const BURN = "0x000000000000000000000000000000000000dEaD";
  // **The burn address has never been touched on 4663, and the fork cannot cope
  // with that.** `RPCStateManager` fetches an account lazily and throws
  // `Cannot read properties of undefined (reading 'balance')` inside the gas
  // handler when the node answers that it does not exist -- so the transfer that
  // burns the tokens dies in the fixture rather than in the contract.
  //
  // Seeded empty, which is what an unused address *is*. Worth recording as a
  // property of the chain rather than of this code: on the real chain the first
  // burn simply creates the account, and that costs the extra gas of a new
  // account rather than failing.
  //
  // `putAccount` with no `getAccount` first, deliberately. Reading it goes to
  // the provider, which is the call that throws -- and there is nothing to read:
  // an EOA's token balance lives in the *token's* storage, not in its account,
  // so a fresh `Account` here loses none of the 12.9M GLADOS that address
  // already holds on chain.
  await vm.stateManager.putAccount(addr(BURN), new Account());
  const burnArt = {
    abi: art["GladosBurner.sol"].GladosBurner.abi,
    bytecode: art["GladosBurner.sol"].GladosBurner.evm.bytecode.object,
  };
  const burner = await deploy(vm, OPERATOR, burnArt, [at, TOKEN]);
  ok(!!burner, `a burner is deployed into the fork at ${burner}`);

  // Fund and build a one-leaf tree for it, with the same machinery the epoch
  // above used so the proof is the contract's own format rather than a second
  // idea of it.
  {
    const { build } = await import("./merkle.mjs");
    const amount = want / 4n > 0n ? want / 4n : 1n;
    const t = build([{ account: burner, amount }]);

    const dep2 = await call(vm, OPERATOR, WETH, "0xd0e30db0", amount);
    ok(dep2.ok, `the operator wraps ${ethers.formatEther(amount)} more ETH for the burn epoch`);
    const ap2 = await call(vm, OPERATOR, WETH, erc20.encodeFunctionData("approve", [at, amount]));
    ok(ap2.ok, "and approves it");

    const open2 = iface.encodeFunctionData("openEpochOnMarket",
      [t.root, amount, 0n, BigInt(Math.floor(Date.now() / 1000) + 86400)]);
    const o2 = await call(vm, OPERATOR, at, open2);
    ok(o2.ok, `an epoch opens with the burner as its only leaf${o2.ok ? "" : "  (" + o2.err + ")"}`);

    const bBefore = await call(vm, OPERATOR, TOKEN, erc20.encodeFunctionData("balanceOf", [BURN]));
    const beforeBurn = coder.decode(["uint256"], bBefore.ret)[0];

    const bIface = new ethers.Interface(burnArt.abi);
    // Called by an address that is nobody in particular, to show the burn needs
    // no permission and enriches no one who triggers it.
    const stranger = "0x2222222222222222222222222222222222222222";
    // `putAccount` with no read first. **This was the crash**, and it is worth the
    // comment because the error names nothing useful: `RPCStateManager` fetches a
    // missing account lazily through `eth_getProof` and throws
    // `Cannot read properties of undefined (reading 'balance')` from inside the
    // gas handler when the provider does not answer -- which the public endpoint
    // stops doing after enough calls in one run.
    //
    // There is nothing to read anyway. This address has never been touched on
    // 4663, so an empty account is exactly what it is, and writing one directly
    // skips the fetch that fails. The accounts seeded earlier in this file were
    // read the same way and survived only by being early enough in the run.
    await vm.stateManager.putAccount(addr(stranger), new Account(0n, 10n ** 20n));
    const bd = bIface.encodeFunctionData("claimAndBurn", [1, amount, t.proof(burner), 1n]);
    const br = await call(vm, stranger, burner, bd);
    ok(br.ok, `a stranger triggers the claim and burn${br.ok ? "" : "  (" + br.err + ")"}`);

    const bAfter = await call(vm, OPERATOR, TOKEN, erc20.encodeFunctionData("balanceOf", [BURN]));
    const moved = coder.decode(["uint256"], bAfter.ret)[0] - beforeBurn;
    ok(moved > 0n,
       `and ${ethers.formatEther(moved)} GLADOS would reach 0xdEaD for `
       + `${ethers.formatEther(amount)} WETH -- simulated, nothing burnt`);

    const stranded = await call(vm, OPERATOR, burner, bIface.encodeFunctionData("stranded", []));
    ok(coder.decode(["uint256"], stranded.ret)[0] === 0n, "the burner holds nothing afterwards");

    const sBal = await call(vm, OPERATOR, TOKEN, erc20.encodeFunctionData("balanceOf", [stranger]));
    ok(coder.decode(["uint256"], sBal.ret)[0] === 0n,
       "and whoever paid the gas holds none of it");
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  console.log("\n*** NOTHING ABOVE HAPPENED. No transaction was broadcast, no token");
  console.log("*** moved, nobody was paid and nothing was burnt. Every figure is");
  console.log("*** what the deployed contracts would do, computed against their");
  console.log("*** real state. The money is invented by this harness.\n");
  console.log("Every contract above except the distributor is the deployed one,");
  console.log("read over RPC at block " + block + ". Nothing was spent and nothing");
  console.log("was published: the fork is in this process and dies with it.");
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
