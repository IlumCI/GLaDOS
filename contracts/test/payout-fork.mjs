// GladosPayout against a fork of Robinhood Chain: the real GLADOS, the real
// WETH/GLADOS pair, the real tax.
//
//   node test/payout-fork.mjs
//
// The question MockPair cannot answer: does the real token let this contract
// pass GLADOS on to miners untaxed? If it taxes contract-to-wallet transfers,
// every payout reverts with ShortPaid -- correct, and fatal to push payouts --
// so it is asked of the deployed bytecode rather than assumed. Nothing is
// spent or published and no key is needed. Kept short on purpose: the public
// RPC stops answering `eth_getProof` after enough calls (see fork.mjs), so a
// run that dies part-way is no evidence, not a failure.
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
const OPERATOR = "0x1111111111111111111111111111111111111111";
const MINERS = ["0x000000000000000000000000000000000000c0a1", "0x000000000000000000000000000000000000c0a2",
                "0x000000000000000000000000000000000000c0a3"];

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const coder = ethers.AbiCoder.defaultAbiCoder();
const addr = (h) => new Address(hexToBytes(h.toLowerCase()));

async function run(vm, from, to, data, value = 0n) {
  const res = await vm.evm.runCall({ caller: addr(from), origin: addr(from), to: to ? addr(to) : undefined,
                                     data: hexToBytes(data), gasLimit: 30_000_000n, value });
  return { ok: !res.execResult.exceptionError, ret: bytesToHex(res.execResult.returnValue),
           err: res.execResult.exceptionError?.error, created: res.createdAddress, gas: res.execResult.executionGasUsed };
}

async function main() {
  const r = await fetch(RPC, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_blockNumber", params: [] }) });
  const block = parseInt((await r.json()).result, 16);
  console.log(`forking Robinhood Chain at block ${block}`);
  const vm = await VM.create({ common: new Common({ chain: Chain.Mainnet, hardfork: Hardfork.Shanghai }),
                               stateManager: new RPCStateManager({ provider: RPC, blockTag: BigInt(block) }) });
  const acc = new Account(); acc.balance = 10n ** 20n;
  await vm.stateManager.putAccount(addr(OPERATOR), acc);

  const c = compile(["GladosPayout.sol"]);
  const art = c["GladosPayout.sol"].GladosPayout;
  const iface = new ethers.Interface(art.abi);
  const dep = await run(vm, OPERATOR, null, "0x" + art.evm.bytecode.object + coder.encode(["address", "address", "address"], [WETH, TOKEN, PAIR]).slice(2));
  ok(dep.ok, "GladosPayout deploys against the real pair, whose token0/token1 check passes");
  if (!dep.ok) return finish();
  const payout = bytesToHex(dep.created.bytes);

  const res = await run(vm, OPERATOR, PAIR, "0x0902f1ac");
  const [r0, r1] = coder.decode(["uint112", "uint112", "uint32"], res.ret);
  const [rw, rg] = BigInt(WETH) < BigInt(TOKEN) ? [r0, r1] : [r1, r0];
  const IN = 10n ** 15n; // 0.001 ETH
  const w = IN * 997n;
  const quote = (w * rg) / (rw * 1000n + w);
  // The buy tax comes out of the pair's transfer, so bound at 95% of the quote.
  const minOut = (quote * 95n) / 100n;
  console.log(`      pool ${ethers.formatEther(rw)} WETH / ${(Number(rg) / 1e24).toFixed(1)}M GLADOS; quote ${(Number(quote) / 1e18).toFixed(0)} GLADOS for 0.001 ETH`);

  const call = await run(vm, OPERATOR, payout, iface.encodeFunctionData("buyAndPay", [MINERS, minOut]), IN);
  if (!call.ok) {
    let reason = call.err;
    try { reason = iface.parseError(call.ret)?.name ?? reason; } catch {}
    ok(false, `buyAndPay against the real token: ${reason}`);
    return finish();
  }
  const each = iface.decodeFunctionResult("buyAndPay", call.ret)[0];
  const bals = [];
  for (const m of MINERS) {
    const b = await run(vm, OPERATOR, TOKEN, "0x70a08231" + m.slice(2).padStart(64, "0"));
    bals.push(coder.decode(["uint256"], b.ret)[0]);
  }
  ok(bals.every((b) => b === each), `each of three miners received exactly ${(Number(each) / 1e18).toFixed(2)} real GLADOS`);
  const got = each * 3n;
  ok(Number(got) / Number(quote) > 0.97, `the buy tax took ${(100 - (100 * Number(got)) / Number(quote)).toFixed(2)}% of the quote, and the send to miners took nothing`);
  const left = await run(vm, OPERATOR, TOKEN, "0x70a08231" + payout.slice(2).padStart(64, "0"));
  ok(coder.decode(["uint256"], left.ret)[0] === 0n, "the contract holds no GLADOS afterwards");
  console.log(`      gas ${call.gas} for three recipients, buy included`);
  finish();
}

function finish() {
  console.log(`${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error("no evidence (RPC or harness):", e.message); process.exit(3); });
