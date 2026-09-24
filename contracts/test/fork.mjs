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
    ok(got > 0n, `${a} bought ${ethers.formatEther(got)} real GLADOS with ${ethers.formatEther(BigInt(c.amount))} WETH`);
  }
  ok(anyPaid, "somebody was actually paid in GLADOS from the real pool");

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
  {
    const acc = (await vm.stateManager.getAccount(addr(BURN))) ?? new Account();
    await vm.stateManager.putAccount(addr(BURN), acc);
  }
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
    {
      const acc = (await vm.stateManager.getAccount(addr(stranger))) ?? new Account();
      acc.balance = 10n ** 20n;
      await vm.stateManager.putAccount(addr(stranger), acc);
    }
    const bd = bIface.encodeFunctionData("claimAndBurn", [1, amount, t.proof(burner), 1n]);
    const br = await call(vm, stranger, burner, bd);
    ok(br.ok, `a stranger triggers the claim and burn${br.ok ? "" : "  (" + br.err + ")"}`);

    const bAfter = await call(vm, OPERATOR, TOKEN, erc20.encodeFunctionData("balanceOf", [BURN]));
    const moved = coder.decode(["uint256"], bAfter.ret)[0] - beforeBurn;
    ok(moved > 0n,
       `and ${ethers.formatEther(moved)} real GLADOS is now at 0xdEaD, bought with ${ethers.formatEther(amount)} WETH`);

    const stranded = await call(vm, OPERATOR, burner, bIface.encodeFunctionData("stranded", []));
    ok(coder.decode(["uint256"], stranded.ret)[0] === 0n, "the burner holds nothing afterwards");

    const sBal = await call(vm, OPERATOR, TOKEN, erc20.encodeFunctionData("balanceOf", [stranger]));
    ok(coder.decode(["uint256"], sBal.ret)[0] === 0n,
       "and whoever paid the gas holds none of it");
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  console.log("\nEvery contract above except the distributor is the deployed one,");
  console.log("read over RPC at block " + block + ". Nothing was spent and nothing");
  console.log("was published: the fork is in this process and dies with it.");
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
