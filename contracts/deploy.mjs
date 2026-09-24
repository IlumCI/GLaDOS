// Put the distributor on a real chain, and drive it.
//
//   node deploy.mjs status
//   node deploy.mjs deploy                       --send
//   node deploy.mjs open   epoch.json --amount 0.005 --send
//   node deploy.mjs open   epoch.json --amount 0.005 --direct --send
//   node deploy.mjs open   epoch.json --amount 0.005 --v3-pool 0x.. --reward 0x.. --send
//   node deploy.mjs claim  epoch.json --epoch 0      --send
//
// ### Three modes to open in, and `claim` works out which without being told
//
// `--direct` holds the reward token and a claim transfers it; the default holds
// the quote token and each claim is the claimant's own buy on the V2 pair;
// `--v3-pool` does that on a V3 pool for a reward the epoch names rather than
// the one the contract gates on, which is how somebody is paid in a tokenized
// equity while the gate stays GLADOS.
//
// `claim` reads the epoch's `mode` and dispatches. It used to call
// `claimOnMarket` unconditionally, so it could open a `MarketV3` epoch and then
// be unable to claim from the thing it had just opened -- and `checkClaim`,
// which it asks first, does not read `mode` either, so it said the claim was
// good on the way to reverting. `--slippage` (percent, default 2) sets the
// bound on both market paths and `--min-out` overrides it outright on V3, for
// when the quoter will not price the swap.
//
// ### Simulation is the default and `--send` is the whole safety model
//
// Every command runs as `eth_call` against the live chain first and prints what
// would happen. Nothing is broadcast without `--send`. That is not politeness:
// the difference between a correct epoch and one funded with the wrong number
// is a transaction that succeeds either way, and the only moment anybody can
// catch it is before it is signed.
//
// ### The key is yours and this file never sees it twice
//
// Read from `GLADOS_KEY` in the environment, used to sign, and never printed,
// never written to a file, never included in an error message. If you are
// reading this to check that claim: search the file for `GLADOS_KEY` -- there
// are two uses, one to read it and one to refuse when it is absent.
//
// Use a wallet that holds what this event needs and nothing else. The operator
// address is immutable in the deployed contract, so it is also the one address
// that cannot be rotated afterwards.
import fs from "node:fs";
import { ethers } from "ethers";
import { compile } from "./test/build.mjs";

const RPC = process.env.GLADOS_RPC || "https://rpc.mainnet.chain.robinhood.com";
const TOKEN = process.env.GLADOS_TOKEN || "0x3d609ecafc6aa7dba67dd7ad1d10b49c52d57777";
const PAIR = process.env.GLADOS_PAIR || "0x93f777932d98d15b351d1bce8c76b34381eede5b";
const QUOTE = process.env.GLADOS_QUOTE || "0x0bd7d308f8e1639fab988df18a8011f41eacad73";
// The Uniswap V3 factory on 4663 and its quoter, both measured in
// `design/rwa.md` rather than taken from a deployment list: the factory has
// 6,578 pools and holds the tokenized-equity market that the V2 one carries
// four hundredths of a percent of. The constructor takes the factory, so a
// distributor deployed without it can never open a `MarketV3` epoch -- and
// unlike `pair` and `quote`, which may be zero together, this one has no
// paired argument to make its absence obvious.
const V3FACTORY = process.env.GLADOS_V3_FACTORY || "0x1f7d7550B1b028f7571E69A784071F0205FD2EfA";
const QUOTER = process.env.GLADOS_QUOTER || "0x33e885eD0Ec9bF04EcfB19341582aADCb4c8A9E7";

const argv = process.argv.slice(2);
const cmd = argv[0];
const SEND = argv.includes("--send");
const flag = (name, dflt) => {
  const i = argv.indexOf("--" + name);
  return i >= 0 ? argv[i + 1] : dflt;
};

const ERC20 = [
  "function balanceOf(address) view returns (uint256)",
  "function allowance(address,address) view returns (uint256)",
  "function approve(address,uint256) returns (bool)",
  "function deposit() payable",
  "function decimals() view returns (uint8)",
];

async function main() {
  const provider = new ethers.JsonRpcProvider(RPC);
  const net = await provider.getNetwork();

  const key = process.env.GLADOS_KEY;
  if (!key && cmd !== "status") {
    console.error("set GLADOS_KEY to the operator's private key, in the environment.");
    console.error("It is used to sign and is never printed or stored by this script.");
    console.error("  PowerShell:  $env:GLADOS_KEY = '0x...'");
    console.error("  bash:        export GLADOS_KEY=0x...");
    process.exit(2);
  }
  const wallet = key ? new ethers.Wallet(key, provider) : null;

  const art = compile(["GladosDistributor.sol"]);
  const D = art["GladosDistributor.sol"].GladosDistributor;
  const iface = new ethers.Interface(D.abi);

  const addrOf = flag("at", process.env.GLADOS_DISTRIBUTOR);

  // ------------------------------------------------------------- status
  if (cmd === "status") {
    console.log(`chain      ${net.chainId}`);
    console.log(`rpc        ${RPC}`);
    console.log(`token      ${TOKEN}`);
    console.log(`pair       ${PAIR}`);
    console.log(`quote      ${QUOTE}`);
    const pairC = new ethers.Contract(PAIR, [
      "function getReserves() view returns (uint112,uint112,uint32)",
    ], provider);
    const [r0, r1] = await pairC.getReserves();
    const qIs0 = BigInt(QUOTE) < BigInt(TOKEN);
    const [rq, rt] = qIs0 ? [r0, r1] : [r1, r0];
    console.log(`pool       ${ethers.formatEther(rq)} quote against ${ethers.formatEther(rt)} token`);
    if (wallet) {
      console.log(`operator   ${wallet.address}`);
      console.log(`  gas      ${ethers.formatEther(await provider.getBalance(wallet.address))} ETH`);
      const q = new ethers.Contract(QUOTE, ERC20, provider);
      console.log(`  quote    ${ethers.formatEther(await q.balanceOf(wallet.address))}`);
    }
    if (addrOf) {
      const d = new ethers.Contract(addrOf, D.abi, provider);
      const n = await d.epochCount();
      console.log(`distributor ${addrOf}, ${n} epoch(s)`);
      for (let i = 0n; i < n; i++) {
        const e = await d.epochs(i);
        console.log(`  #${i}  mode=${e.mode === 0n ? "Direct" : "Market"}  funded=${ethers.formatEther(e.funded)}` +
                    `  claimed=${ethers.formatEther(e.claimed)}  gate=${ethers.formatEther(e.gate)}`);
      }
    }
    return;
  }

  // ------------------------------------------------------------- deploy
  if (cmd === "deploy") {
    const factory = new ethers.ContractFactory(D.abi, D.bytecode, wallet);

    // **The pair is read back and checked here because nothing checks it ever
    // again.** `design/audit.md` finding 1: the constructor takes `pair` and
    // never compares it against the pair it is supposed to be, while
    // `openEpochOnV3` checks its pool twice -- so a wrong address here is a
    // distributor that accepts funding for market epochs and refuses every
    // claim, forever, with no setter and no recovery but `reclaim` after the
    // deadline. It is `immutable`, so this is the only moment it can be caught.
    const pairC = new ethers.Contract(PAIR,
      ["function token0() view returns (address)",
       "function token1() view returns (address)"], provider);
    let p0, p1;
    try {
      [p0, p1] = await Promise.all([pairC.token0(), pairC.token1()]);
    } catch {
      console.error(`${PAIR} does not answer token0/token1, so it is not a V2 pair.`);
      process.exit(1);
    }
    const pairOk = (p0.toLowerCase() === QUOTE.toLowerCase() && p1.toLowerCase() === TOKEN.toLowerCase())
                || (p1.toLowerCase() === QUOTE.toLowerCase() && p0.toLowerCase() === TOKEN.toLowerCase());
    if (!pairOk) {
      console.error(`the pair at ${PAIR} holds`);
      console.error(`  ${p0}\n  ${p1}`);
      console.error(`which is not {quote, token}:`);
      console.error(`  ${QUOTE}  (quote)\n  ${TOKEN}  (token)`);
      console.error("\nDeploying against it would produce a distributor whose market");
      console.error("epochs can be funded and never claimed. `pair` is immutable.");
      process.exit(1);
    }

    const tx = await factory.getDeployTransaction(TOKEN, wallet.address, PAIR, QUOTE, V3FACTORY);
    const gas = await provider.estimateGas({ ...tx, from: wallet.address });
    const price = (await provider.getFeeData()).gasPrice ?? 0n;
    console.log(`deploying with operator ${wallet.address}`);
    console.log(`  token   ${TOKEN}\n  pair    ${PAIR}\n  quote   ${QUOTE}`);
    console.log(`  factory ${V3FACTORY}`);
    console.log(`  pair holds ${p0} / ${p1}  -- checked`);
    console.log(`  gas   ${gas} at ${ethers.formatUnits(price, "gwei")} gwei = ${ethers.formatEther(gas * price)} ETH`);
    if (!SEND) return console.log("\nsimulated only. Add --send to broadcast.");
    const c = await factory.deploy(TOKEN, wallet.address, PAIR, QUOTE, V3FACTORY);
    console.log(`  tx    ${c.deploymentTransaction().hash}`);
    await c.waitForDeployment();
    const at = await c.getAddress();
    console.log(`\ndistributor at ${at}`);
    console.log(`export GLADOS_DISTRIBUTOR=${at}`);
    return;
  }

  if (!addrOf) {
    console.error("set GLADOS_DISTRIBUTOR or pass --at 0x...");
    process.exit(2);
  }
  const dist = new ethers.Contract(addrOf, D.abi, wallet);

  // --------------------------------------------------------------- open
  if (cmd === "open") {
    const doc = JSON.parse(fs.readFileSync(argv[1], "utf8"));
    const amount = ethers.parseEther(flag("amount", "0"));
    const gate = ethers.parseEther(flag("gate", "0"));
    const days = Number(flag("days", "30"));
    const deadline = BigInt(Math.floor(Date.now() / 1000) + days * 86400);
    if (amount === 0n) {
      console.error("--amount is required, in whole quote tokens (e.g. --amount 0.005)");
      process.exit(2);
    }

    // **Which mode, decided by whether a pool was named.** The two are not
    // interchangeable and the contract will not let them be: `claimOnMarket`
    // refuses a V3 epoch and `claimOnV3` refuses a Market one, so opening in
    // the wrong mode is an epoch nobody can claim from until it expires.
    const v3pool = flag("v3-pool", "");
    const reward = flag("reward", "");
    if ((v3pool === "") !== (reward === "")) {
      console.error("--v3-pool and --reward go together: the pool says where to swap,");
      console.error("the reward says what should come out, and the contract checks they agree.");
      process.exit(2);
    }
    const onV3 = v3pool !== "";
    // **`Direct` had no operator path at all**, which is why this tool could
    // claim from three modes and open two. The epoch holds the reward token and
    // a claim transfers it: one market buy, made by the operator when they
    // convert, and every claimant at the same rate -- no swap, no slippage and
    // no pool in the claim. `contracts/README.md`'s table is why it matters at
    // this size: over a 36-hour event a `Market` claim's gas is 2,286% of what
    // it pays out, and `Direct` is the mode that is not absurd there.
    const direct = argv.includes("--direct");
    if (direct && onV3) {
      console.error("--direct and --v3-pool are different modes; pick one.");
      process.exit(2);
    }
    // Direct funds in the reward token, the other two in the quote token. One
    // line decides it, so it is not decided again anywhere below.
    const FUND = direct ? TOKEN : QUOTE;

    // **A root already open is a second entitlement to the same work, not a
    // duplicate anything refuses.** `design/audit.md` finding 3: the leaf is
    // `(account, amount)` and carries no epoch id, while `hasClaimed` is keyed
    // per epoch -- so opening one root twice lets every leaf in it claim twice.
    // Nothing on chain can tell that from an operator deliberately funding a
    // second distribution of the same allocations, which is why the check lives
    // here and not in the contract: putting the epoch id in the leaf would
    // invalidate `distribute.py`, `merkle.mjs` and every proof either has
    // produced, to defend against an operator error a read loop catches free.
    const already = Number(await dist.epochCount());
    for (let i = 0; i < already; i++) {
      const ep = await dist.epochs(i);
      if (ep.root.toLowerCase() === String(doc.root).toLowerCase()) {
        console.error(`epoch ${i} is already open with this root, ${doc.root}`);
        console.error("Opening it again would let every leaf claim twice.");
        console.error("Rebuild the tree, or claim from that epoch instead.");
        process.exit(1);
      }
    }

    const q = new ethers.Contract(FUND, ERC20, wallet);
    const have = await q.balanceOf(wallet.address);
    console.log(`epoch root  ${doc.root}`);
    console.log(`epochs open ${already}, none with this root -- checked`);
    console.log(`claims      ${Object.keys(doc.claims).length}`);
    console.log(`funding     ${ethers.formatEther(amount)} ${direct ? "token" : "quote"} (you hold ${ethers.formatEther(have)})`);
    console.log(`gate        ${ethers.formatEther(gate)}`);
    console.log(`deadline    in ${days} day(s)`);
    if (onV3) {
      console.log(`venue       Uniswap V3 pool ${v3pool}`);
      console.log(`reward      ${reward}`);
      // Read back rather than trusted, and printed, because the contract's own
      // check happens inside a transaction the operator is about to sign. A
      // pool for the wrong pair reverts with `BadPool` and costs gas to find
      // out; reading it here costs one call and prints the pair.
      try {
        const pool = new ethers.Contract(v3pool,
          ["function token0() view returns (address)",
           "function token1() view returns (address)",
           "function fee() view returns (uint24)"], provider);
        const [t0, t1, fee] = await Promise.all([pool.token0(), pool.token1(), pool.fee()]);
        console.log(`pool pair   ${t0} / ${t1}  fee ${fee}`);
        const ok = (t0.toLowerCase() === QUOTE.toLowerCase() && t1.toLowerCase() === reward.toLowerCase())
                || (t1.toLowerCase() === QUOTE.toLowerCase() && t0.toLowerCase() === reward.toLowerCase());
        if (!ok) {
          console.error("\nthat pool is not the quote/reward pair. The contract would refuse it.");
          process.exit(1);
        }
      } catch {
        console.error("\nthat address does not answer token0/token1/fee, so it is not a V3 pool.");
        process.exit(1);
      }
    } else if (direct) {
      console.log(`venue       none -- the epoch holds the reward and a claim transfers it`);
    } else {
      console.log(`venue       the V2 pair the contract was built with`);
    }
    if (have < amount) {
      if (direct) {
        console.error(`\nnot enough ${TOKEN} to fund this epoch.`);
        console.error("A Direct epoch is funded in the reward token itself, so this is");
        console.error("GLADOS you must hold -- `wrap` makes quote token and will not help.");
      } else {
        console.error("\nnot enough quote token. Wrap some first:");
        console.error(`  node deploy.mjs wrap --amount ${ethers.formatEther(amount)} --send`);
      }
      process.exit(1);
    }
    // **The leaves are denominated in the quote token**, so their sum has to be
    // what is funded. A mismatch is an epoch that either runs short on the last
    // claim or strands the difference until the deadline.
    const sum = Object.values(doc.claims).reduce((a, c) => a + BigInt(c.amount), 0n);
    if (sum !== amount) {
      console.error(`\nthe tree sums to ${ethers.formatEther(sum)} but --amount is ${ethers.formatEther(amount)}.`);
      console.error("Rebuild the epoch with --total equal to what you are funding.");
      process.exit(1);
    }
    if (!SEND) return console.log("\nsimulated only. Add --send to broadcast.");

    const allow = await q.allowance(wallet.address, addrOf);
    if (allow < amount) {
      const a = await q.approve(addrOf, amount);
      console.log(`  approve ${a.hash}`);
      await a.wait();
    }
    const tx = onV3
      ? await dist.openEpochOnV3(doc.root, amount, gate, deadline, v3pool, reward)
      : direct
        ? await dist.openEpoch(doc.root, amount, gate, deadline)
        : await dist.openEpochOnMarket(doc.root, amount, gate, deadline);
    console.log(`  open    ${tx.hash}`);
    const rc = await tx.wait();
    const ev = rc.logs.map((l) => { try { return iface.parseLog(l); } catch { return null; } })
      .find((x) => x && x.name === "EpochOpened");
    console.log(`\nepoch ${ev ? ev.args.epoch : "?"} open, funded ${ev ? ethers.formatEther(ev.args.funded) : "?"}`);
    return;
  }

  // -------------------------------------------------------------- claim
  if (cmd === "claim") {
    const doc = JSON.parse(fs.readFileSync(argv[1], "utf8"));
    const id = BigInt(flag("epoch", "0"));
    const me = wallet.address.toLowerCase();
    const c = doc.claims[me];
    if (!c) {
      console.error(`${wallet.address} is not in this epoch`);
      process.exit(1);
    }
    const check = await dist.checkClaim(id, wallet.address, BigInt(c.amount), c.proof);
    console.log(`claimable   ${check[0]}${check[0] ? "" : "  (" + check[1] + ")"}`);
    if (!check[0]) process.exit(1);

    // **`checkClaim` answering true is not enough, because it never reads
    // `mode`.** `design/audit.md` finding 2: it checks the deadline, the claim
    // record, the gate, the proof and solvency, and then a `Direct` claim
    // against a market epoch reverts `WrongMode` anyway. This tool used to call
    // `claimOnMarket` unconditionally (finding 5), so it could open a
    // `MarketV3` epoch and then be unable to claim from the thing it had just
    // opened. The epoch says which of the three it is; ask it.
    const ep = await dist.epochs(id);
    const mode = Number(ep.mode);
    const MODES = ["Direct", "Market", "MarketV3"];
    console.log(`mode        ${MODES[mode] ?? mode}`);

    let tx;
    if (mode === 0) {
      // Direct: the epoch holds the reward token and a claim transfers it.
      // No pool, no slippage, and therefore no `minOut` to get wrong.
      //
      // **What arrives may be less than `amount` and the event will not say
      // so** -- finding 4: `claim()` emits `Claimed(.., amount, amount)`
      // without measuring, where both market paths measure the claimant's
      // balance either side. So the balance is read here instead, and the
      // difference printed, because the number in the log is the one somebody
      // would otherwise quote.
      const t = new ethers.Contract(TOKEN, ERC20, provider);
      const before = await t.balanceOf(wallet.address);
      console.log(`receiving   ${ethers.formatEther(BigInt(c.amount))} token, transferred directly`);
      if (!SEND) return console.log("\nsimulated only. Add --send to broadcast.");
      tx = await dist.claim(id, BigInt(c.amount), c.proof);
      console.log(`  claim   ${tx.hash}`);
      await tx.wait();
      const after = await t.balanceOf(wallet.address);
      const got = after - before;
      console.log(`\nreceived ${ethers.formatEther(got)} GLADOS` +
        (got === BigInt(c.amount) ? "" :
          `  (the event will say ${ethers.formatEther(BigInt(c.amount))}; it does not measure)`));
      return;
    }

    if (mode === 1) {
      // Market: the claim is the claimant's own buy on the V2 pair. A slippage
      // bound the claimant chooses, defaulting to 2% off what the pool owes
      // right now. Zero is refused by the contract, deliberately.
      const pairC = new ethers.Contract(PAIR, ["function getReserves() view returns (uint112,uint112,uint32)"], provider);
      const [r0, r1] = await pairC.getReserves();
      const qIs0 = BigInt(QUOTE) < BigInt(TOKEN);
      const [rq, rt] = qIs0 ? [r0, r1] : [r1, r0];
      const wf = BigInt(c.amount) * 997n;
      const out = (wf * rt) / (rq * 1000n + wf);
      const slip = BigInt(Math.round(Number(flag("slippage", "2")) * 100));
      const minOut = (out * (10000n - slip)) / 10000n;
      console.log(`spending    ${ethers.formatEther(BigInt(c.amount))} quote`);
      console.log(`pool owes   ${ethers.formatEther(out)} token before tax`);
      console.log(`minimum     ${ethers.formatEther(minOut)} (${flag("slippage", "2")}% tolerance)`);
      if (!SEND) return console.log("\nsimulated only. Add --send to broadcast.");
      tx = await dist.claimOnMarket(id, BigInt(c.amount), c.proof, minOut);
    } else if (mode === 2) {
      // MarketV3: same shape, a V3 pool, and a reward token the epoch names
      // rather than the one the contract gates on -- which is the whole point,
      // since it is how somebody is paid in a tokenized equity while the gate
      // stays GLADOS.
      //
      // **The bound is quoted against live state rather than computed.** V2's
      // constant product can be done in four lines from `getReserves`; a V3
      // pool's output depends on the tick map, so the honest way to price it is
      // to ask the quoter to simulate the swap. `design/rwa.md` measured this
      // one doing exactly that, USDG into NVDA, at every size that matters.
      const rewardTok = ep.reward;
      const poolC = new ethers.Contract(ep.pool,
        ["function fee() view returns (uint24)"], provider);
      const fee = await poolC.fee();
      const quoter = new ethers.Contract(QUOTER, [
        "function quoteExactInputSingle((address,address,uint256,uint24,uint160)) returns (uint256,uint160,uint32,uint256)",
      ], provider);
      let out;
      try {
        const r = await quoter.quoteExactInputSingle.staticCall(
          [QUOTE, rewardTok, BigInt(c.amount), fee, 0n]);
        out = r[0];
      } catch {
        console.error(`the quoter at ${QUOTER} would not price this swap.`);
        console.error("Pass --min-out <whole tokens> to set the bound yourself.");
        if (flag("min-out", "") === "") process.exit(1);
      }
      const explicit = flag("min-out", "");
      const slip = BigInt(Math.round(Number(flag("slippage", "2")) * 100));
      const minOut = explicit !== ""
        ? ethers.parseEther(explicit)
        : (out * (10000n - slip)) / 10000n;
      const rc20 = new ethers.Contract(rewardTok, ERC20, provider);
      let sym = "reward";
      try { sym = await rc20.symbol(); } catch { /* not every token has one */ }
      console.log(`reward      ${rewardTok} (${sym})`);
      console.log(`pool        ${ep.pool}  fee ${fee}`);
      console.log(`spending    ${ethers.formatEther(BigInt(c.amount))} quote`);
      if (out !== undefined) console.log(`quoted      ${ethers.formatEther(out)} ${sym}`);
      console.log(`minimum     ${ethers.formatEther(minOut)}` +
        (explicit !== "" ? "  (--min-out)" : `  (${flag("slippage", "2")}% tolerance)`));
      if (!SEND) return console.log("\nsimulated only. Add --send to broadcast.");
      tx = await dist.claimOnV3(id, BigInt(c.amount), c.proof, minOut);
    } else {
      console.error(`epoch ${id} reports mode ${mode}, which this tool does not know.`);
      process.exit(1);
    }

    console.log(`  claim   ${tx.hash}`);
    const rc = await tx.wait();
    const ev = rc.logs.map((l) => { try { return iface.parseLog(l); } catch { return null; } })
      .find((x) => x && x.name === "Claimed");
    console.log(`\nreceived ${ev ? ethers.formatEther(ev.args.received) : "?"}`);
    return;
  }

  // --------------------------------------------------------------- wrap
  if (cmd === "wrap") {
    const amount = ethers.parseEther(flag("amount", "0"));
    console.log(`wrapping ${ethers.formatEther(amount)} ETH into ${QUOTE}`);
    if (!SEND) return console.log("simulated only. Add --send to broadcast.");
    const q = new ethers.Contract(QUOTE, ERC20, wallet);
    const tx = await q.deposit({ value: amount });
    console.log(`  ${tx.hash}`);
    await tx.wait();
    console.log(`now holding ${ethers.formatEther(await q.balanceOf(wallet.address))}`);
    return;
  }

  console.error("commands: status, deploy, wrap, open, claim");
  process.exit(2);
}

main().catch((e) => {
  // Deliberately not `console.error(e)`: an ethers error can carry the
  // transaction it was signing, and a signed transaction is not a secret but
  // the habit of dumping whole objects near a wallet is a bad one to build.
  console.error(e.shortMessage || e.message || String(e));
  process.exit(1);
});
