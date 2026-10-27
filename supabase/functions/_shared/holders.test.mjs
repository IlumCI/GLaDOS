// The holder gate's arithmetic, checked by node.
//
//     node supabase/functions/_shared/holders.test.mjs
//
// The positives are cheap. What earns its place here is the boundary: a holder
// exactly at 2% is in and one wei short is out, a 4% holder is in both chats,
// and a tier list with a typo refuses to load rather than loading with a chat
// missing from it.

import {
  barOf, chatsFor, encodeIsValidSignature, ERC6492_SUFFIX, holderMessage, isMagic, meets,
  parseTiers, percent, tierOf, tokens, unwrap6492,
} from "./holders.js";

let ok = true;
function claim(good, what) {
  console.log(`  ${good ? "ok " : "FAIL"}   ${what}`);
  if (!good) ok = false;
}
const throws = (f) => { try { f(); return false; } catch { return true; } };

const E18 = 10n ** 18n;
const SUPPLY = 1_000_000_000n * E18;
const TWO = SUPPLY / 50n;                       // exactly 2%

console.log("[holders] the bar");
claim(meets(TWO, SUPPLY, 200), "exactly 2% meets a 200 bps bar");
claim(!meets(TWO - 1n, SUPPLY, 200), "one wei under 2% does not");
claim(!meets(TWO, SUPPLY, 400), "2% does not meet a 400 bps bar");
claim(meets(2n * TWO, SUPPLY, 400), "exactly 4% meets a 400 bps bar");
claim(!meets(TWO, 0n, 200), "a supply of zero admits nobody, rather than dividing by it");
// A share that rounds to 2.0000% at four places and is still short.
claim(percent(TWO - 1n, SUPPLY) === "1.9999" && !meets(TWO - 1n, SUPPLY, 200),
  "the displayed share truncates, so it never shows 2% for a holder the gate refuses");

console.log("[holders] tiers");
const T = parseTiers("400=-1009876543210, 200=-1001234567890");
claim(T.length === 2 && T[0].bps === 200 && T[1].bps === 400, "tiers sort ascending whatever order they are written in");
claim(tierOf(TWO, SUPPLY, T) === 200, "a 2% holder reaches the 2% tier");
claim(tierOf(2n * TWO, SUPPLY, T) === 400, "a 4% holder reaches the 4% tier");
claim(tierOf(TWO - 1n, SUPPLY, T) === 0, "below every bar is tier 0");
claim(JSON.stringify(chatsFor(400, T)) === JSON.stringify(["-1001234567890", "-1009876543210"]),
  "a 4% holder may be in the 2% chat as well");
claim(JSON.stringify(chatsFor(200, T)) === JSON.stringify(["-1001234567890"]), "a 2% holder may not be in the 4% chat");
claim(barOf(-1009876543210, T) === 400, "a chat id arriving as a number finds its bar");
claim(barOf("-1", T) === null, "a chat no tier names has no bar, and is left alone");
const D = parseTiers("");
claim(D.length === 2 && D[0].bps === 200 && D[1].bps === 400 && D.every((t) => t.chat === null),
  "unset, the tiers are 2% and 4% with no chats to admit to");
claim(parseTiers("250=@glados_holders")[0].chat === "@glados_holders", "a public chat may be named by username");
claim(throws(() => parseTiers("200=-100,2O0=-200")), "a typo in a bar refuses to load");
claim(throws(() => parseTiers("200=-100,200=-200")), "one bar twice refuses to load");
claim(throws(() => parseTiers("200=-100,400=-100")), "one chat under two bars refuses to load");
claim(throws(() => parseTiers("0=-100")) && throws(() => parseTiers("10001=-100")), "a bar outside 1..10000 bps refuses to load");

console.log("[holders] display");
claim(tokens(20_000_000n * E18 + 5n) === "20,000,000", "token amounts truncate and group by thousands");
claim(percent(TWO, SUPPLY) === "2.0000", "2% displays as 2.0000");

console.log("[holders] the message");
const msg = holderMessage({
  domain: "glados.aperture.institute", address: "0x0000000000000000000000000000000000000001",
  tgUser: 123456789, chainId: 4663, nonce: "abc123def456",
  issued: new Date("2026-10-27T00:00:00Z"), expires: new Date("2026-10-27T00:10:00Z"),
});
const lines = msg.split("\n");
claim(lines[0] === "glados.aperture.institute wants you to sign in with your Ethereum account:", "the first line names this domain");
claim(lines[3].includes("Telegram user 123456789"), "the statement names the Telegram user");
claim(lines.includes("- tg://user?id=123456789"), "and so does a resource line");
claim(lines.includes("Chain ID: 4663"), "the chain is the token's");
claim(throws(() => holderMessage({ domain: "x", address: "0x1", tgUser: "12\n34", chainId: 1, nonce: "n", issued: new Date(), expires: new Date() })),
  "a Telegram id that is not a number cannot be written into the message");

console.log("[holders] contract wallets");
const H = "11".repeat(32);
const cd = encodeIsValidSignature("0x" + H, "0x" + "ab".repeat(65));
claim(cd.startsWith("0x1626ba7e" + H), "isValidSignature calldata leads with its selector and the hash");
claim(cd.length === 2 + 8 + 64 * 3 + 64 * 3, "and a 65-byte signature pads to three words");
claim(isMagic("0x1626ba7e" + "0".repeat(56)), "the magic value is recognised");
claim(!isMagic("0x1626ba7f" + "0".repeat(56)) && !isMagic("0x") && !isMagic("0x1626ba7e"),
  "anything else, including a short answer, is not");
claim(unwrap6492("0x" + "ab".repeat(65)) === null, "a plain signature is not a 6492 wrapper");
claim(throws(() => unwrap6492("0x" + "00".repeat(10) + ERC6492_SUFFIX)), "a truncated wrapper refuses rather than reading past its end");

console.log(ok ? "holders: all claims hold" : "holders: FAILED");
process.exit(ok ? 0 : 1);
