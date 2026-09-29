// RVN -> ETH on Robinhood Chain (4663), through ChangeNOW. The leg between
// zpool's payout and `deploy.mjs open`.
//
//   node swap.mjs min
//   node swap.mjs quote  --rvn 500
//   node swap.mjs create --rvn 500 --to 0x...       registers an exchange; moves nothing
//   node swap.mjs status --id <exchange id>
//
// **This moves no money and holds no key.** `create` asks ChangeNOW for a
// deposit address and prints it with the exact amount; the RVN is sent from the
// operator's own wallet, by the operator, and that send is the step that cannot
// be undone. Semi-automatic by design for testing; the unattended form belongs
// to the pool's own treasury once it exists.
//
// **Why ChangeNOW.** zpool pays RVN, and of the no-account swappers that reach
// 4663 only ChangeNOW lists RVN at all -- SideShift reaches 4663 and has no
// RVN. Its minimum is ~153 RVN (checked live), and the second research pass
// measured ~9% lost on a $2.35 quote, falling with size: batch, don't dribble.
//
// Needs CHANGENOW_KEY in the environment (a free partner API key). `min` works
// without one.

const API = "https://api.changenow.io/v2";
const KEY = process.env.CHANGENOW_KEY || "";

const args = process.argv.slice(2);
const cmd = args[0];
const opt = (k) => (args.includes(k) ? args[args.indexOf(k) + 1] : undefined);

const PAIR = "fromCurrency=rvn&fromNetwork=rvn&toCurrency=eth&toNetwork=hood&flow=standard";

async function get(path, keyed = true) {
  const res = await fetch(`${API}${path}`, { headers: keyed ? { "x-changenow-api-key": KEY } : {} });
  const text = await res.text();
  if (!res.ok) throw new Error(`${res.status} ${text.slice(0, 200)}`);
  return JSON.parse(text);
}

function needKey() {
  if (!KEY) {
    console.error("swap: CHANGENOW_KEY is not set. It is a free ChangeNOW partner API key,");
    console.error("      kept in your own shell -- never in a file this repository tracks.");
    process.exit(2);
  }
}

// An EVM address, or refuse: the payout is irreversible, and a mistyped one is
// somebody else's.
function evm(a) {
  if (!/^0x[0-9a-fA-F]{40}$/.test(a || "")) {
    console.error(`swap: '${a}' is not a 0x address`);
    process.exit(2);
  }
  return a;
}

const n = (v) => {
  const x = Number(v);
  if (!(x > 0)) {
    console.error(`swap: '${v}' is not a positive amount`);
    process.exit(2);
  }
  return x;
};

async function main() {
  if (cmd === "min") {
    const m = await get(`/exchange/min-amount?${PAIR}`, false);
    console.log(`minimum    ${m.minAmount} RVN -> ETH on Robinhood Chain (4663)`);
    return;
  }
  if (cmd === "quote") {
    needKey();
    const rvn = n(opt("--rvn"));
    const q = await get(`/exchange/estimated-amount?${PAIR}&fromAmount=${rvn}`);
    console.log(`send       ${rvn} RVN`);
    console.log(`receive    ~${q.toAmount} ETH on 4663 (estimate; the rate floats until it fills)`);
    return;
  }
  if (cmd === "create") {
    needKey();
    const rvn = n(opt("--rvn"));
    const to = evm(opt("--to"));
    const m = await get(`/exchange/min-amount?${PAIR}`, false);
    if (rvn < m.minAmount) {
      console.error(`swap: ${rvn} RVN is under ChangeNOW's minimum of ${m.minAmount}; batch until it is not`);
      process.exit(2);
    }
    const res = await fetch(`${API}/exchange`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-changenow-api-key": KEY },
      body: JSON.stringify({
        fromCurrency: "rvn", fromNetwork: "rvn",
        toCurrency: "eth", toNetwork: "hood",
        fromAmount: String(rvn), address: to, flow: "standard", type: "direct",
      }),
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`${res.status} ${text.slice(0, 300)}`);
    const x = JSON.parse(text);
    console.log(`exchange   ${x.id}`);
    console.log(`send       exactly ${x.fromAmount ?? rvn} RVN`);
    console.log(`to         ${x.payinAddress}`);
    console.log(`pays       ETH on Robinhood Chain to ${x.payoutAddress ?? to}`);
    console.log(`then       node swap.mjs status --id ${x.id}`);
    return;
  }
  if (cmd === "status") {
    needKey();
    const id = opt("--id");
    if (!id) throw new Error("status needs --id");
    const s = await get(`/exchange/by-id?id=${encodeURIComponent(id)}`);
    console.log(`exchange   ${s.id}  ${s.status}`);
    console.log(`in         ${s.amountFrom ?? s.expectedAmountFrom ?? "?"} RVN  ${s.payinHash ?? ""}`);
    console.log(`out        ${s.amountTo ?? s.expectedAmountTo ?? "?"} ETH  ${s.payoutHash ?? ""}`);
    return;
  }
  console.log("usage: node swap.mjs min | quote --rvn N | create --rvn N --to 0x.. | status --id ID");
  process.exit(2);
}

main().catch((e) => {
  console.error(`swap: ${e.message}`);
  process.exit(1);
});
