// node pool/edge/worker/plan.test.mjs
//
// The claims that keep a payout fair when miners choose what they are paid in:
// every wallet the same share of the pot whatever it chose, the pot spent to
// the wei, one currency per wallet, and every choice that cannot be honoured
// falling back to $GLADOS rather than reverting the epoch for everybody.
// Also that rewards.js and the kernel's src/mine/reward.rs offer the same menu.
import { plan, planGas, legsOf, effective } from "./plan.js";
import { MENU, TOKENS, DEFAULT, rewardOf } from "./rewards.js";
import { readFileSync } from "node:fs";

let passed = 0, failed = 0;
const ok = (c, w) => { c ? passed++ : failed++; console.log(`${c ? "ok  " : "FAIL"}  ${w}`); };
const W = (i) => "0x" + i.toString(16).padStart(40, "0");
const sum = (p) => p.groups.reduce((s, g) => s + g.legs.reduce((t, l) => t + l.eth, 0n), 0n);
const shareOf = (p, a) => { const g = p.groups.find((x) => x.to.includes(a)); return g.legs.reduce((t, l) => t + l.eth, 0n) / BigInt(g.to.length); };

// The menu.
ok(rewardOf("CHIPS") === "chips" && rewardOf(" nvda ") === "nvda", "a choice is case-insensitive and trimmed");
ok(rewardOf("tesla") === DEFAULT && rewardOf("") === DEFAULT && rewardOf(undefined) === DEFAULT, "anything off the menu is $GLADOS, so a typo never costs a payout");
ok(Object.values(MENU).every((m) => m.tokens.every((s) => TOKENS[s])), "every basket names only tokens the table can buy");
ok(MENU.chips.tokens.includes("SKHY") && MENU.chips.tokens.includes("NVDA") && MENU.os.tokens.includes("MSFT"), "chips carries NVIDIA and SK hynix; operating systems carries Microsoft");
ok(legsOf("glados") === 1 && legsOf("nvda") === 1 && legsOf("chips") === 10, "gas is planned per leg: one for $GLADOS or a single stock, ten for chips");

// An epoch of mixed choices.
const rs = [...Array(10).keys()].map((i) => W(i + 1));
const choices = new Map([[rs[0], "nvda"], [rs[1], "nvda"], [rs[2], "chips"], [rs[3], "os"], [rs[4], "glados"], [rs[5], "nonsense"]]);
const value = 10n ** 18n + 7n;
const p = plan({ recipients: rs, choices, value });
ok(sum(p) === value, "the legs spend the pot to the wei, so payAll's value check holds");
const shares = rs.map((a) => shareOf(p, a));
const base = value / 10n;
ok(shares.every((s) => s - base <= 7n && s >= base), "every wallet's share is the same whatever it chose (to the remainder's wei)");
ok(p.groups.map((g) => g.code).join(",") === "glados,nvda,chips,os", "groups come $GLADOS first, then in menu order");
ok(p.groups.every((g) => g.to.every((a) => p.groups.filter((h) => h.to.includes(a)).length === 1)), "one currency per wallet: nobody is in two groups");
ok(p.groups[0].to.length === 6, "unchosen, $GLADOS and nonsense all land in the $GLADOS group (6 wallets)");
const chips = p.groups.find((g) => g.code === "chips");
ok(chips.legs.length === 10 && chips.legs.every((l) => l.eth > 0n), "a chips wallet's share is spread over all ten chip tokens");

// Fallbacks: blocked wallets and failed quotes are paid in $GLADOS, never dropped.
const blocked = new Set([rs[0]]);
const pb = plan({ recipients: rs, choices, blocked, value });
ok(pb.groups.find((g) => g.code === "glados").to.includes(rs[0]) && pb.groups.find((g) => g.code === "nvda").to.length === 1,
   "a blocklisted wallet that chose NVIDIA is paid in $GLADOS instead, so the epoch cannot revert on it");
const pf = plan({ recipients: rs, choices, fallback: new Set(["chips"]), value });
ok(!pf.groups.some((g) => g.code === "chips") && pf.groups[0].to.includes(rs[2]), "a basket whose quote failed this epoch pays its wallets in $GLADOS");
ok(effective(rs[2], choices, new Set(), new Set(["chips"])) === DEFAULT && effective(rs[1], choices) === "nvda", "effective() agrees with the plan");
ok(sum(pb) === value && sum(pf) === value, "and the pot is still spent to the wei after any fallback");

// Edges.
ok(plan({ recipients: [], value }).groups.length === 0 && plan({ recipients: rs, value: 0n }).groups.length === 0, "nobody, or nothing, plans nothing");
const solo = plan({ recipients: [rs[2]], choices, value: 10n });
ok(sum(solo) === 10n && solo.groups.length === 1 && solo.groups[0].code === "chips", "one wallet alone in a basket still spends exactly the pot");

// Gas grows with legs, so baskets cost what they should.
const cfg = { baseGas: 200_000, v2LegGas: 150_000, v3LegGas: 300_000, perRecipientGas: 60_000 };
ok(planGas(plan({ recipients: rs, value }), cfg) < planGas(p, cfg), "a plan with baskets is planned as more gas than the same wallets in $GLADOS");

// The kernel offers the same menu.
const rust = readFileSync(new URL("../../../src/mine/reward.rs", import.meta.url), "utf8");
const codes = [...rust.matchAll(/\(\s*"([a-z]+)"\s*,\s*"[^"]+"\s*\)/g)].map((m) => m[1]);
ok(codes.length > 0 && codes.join(",") === Object.keys(MENU).join(","), `src/mine/reward.rs offers exactly rewards.js's menu (${codes.length} choices)`);

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
