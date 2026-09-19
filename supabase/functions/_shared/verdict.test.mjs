// The verdict verifier, against the kernel itself.
//
// The two fixture blobs are not synthetic: `verdict-good.bin` is the signed
// verdict a real GLaDOS boot verified and filed as its ledger line 1 (h12,
// axis=source, 737f9c0a), and `verdict-wrongkey.bin` is the byte-identical
// text signed by the update key, which the same boot refused with "not a
// signature over this image by this key". A verifier that agrees with the
// kernel on both has been checked against the implementation that matters,
// not against its own expectations.
//
// Run: node supabase/functions/_shared/verdict.test.mjs   (ci.yml does)

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseSig, verifyRaw, verifyVerdict, VERDICT_KEY_HEX, SIG_LEN }
  from "./verdict_key.js";

const here = dirname(fileURLToPath(import.meta.url));
let failed = 0;
function claim(what, good) {
  if (!good) failed++;
  console.log(`  ${good ? "ok " : "FAIL"}  ${what}`);
}

const good = new Uint8Array(readFileSync(join(here, "fixtures/verdict-good.bin")));
const wrong = new Uint8Array(readFileSync(join(here, "fixtures/verdict-wrongkey.bin")));

// --- the three verdicts the kernel already gave ---------------------------
const g = await verifyVerdict(good);
claim("the blob a real boot filed verifies here too", g.ok);
claim("and its text splits off whole (183 B, 'verdict 1' first)",
  g.ok && g.text.length === good.length - SIG_LEN &&
  new TextDecoder().decode(g.text).startsWith("verdict 1\n"));

const w = await verifyVerdict(wrong);
claim("the blob the kernel refused (update-key-signed) refuses here", !w.ok);

const tampered = new Uint8Array(good);
tampered[20] ^= 1;
claim("one flipped bit in the text refuses",
  !(await verifyVerdict(tampered)).ok);

const tamperedSig = new Uint8Array(good);
tamperedSig[good.length - 1] ^= 1;
claim("one flipped bit in the signature refuses",
  !(await verifyVerdict(tamperedSig)).ok);

// --- the tail parser's own refusals ---------------------------------------
const sig = good.slice(good.length - SIG_LEN);
claim("the real tail parses", parseSig(sig).ok);
claim("a short blob is too short, not a parse error",
  !(await verifyVerdict(good.slice(0, 40))).ok);
const future = new Uint8Array(sig);
future[8] = 2;
claim("a future format version is refused rather than guessed at",
  !parseSig(future).ok);
const otherCurve = new Uint8Array(sig);
otherCurve[12] = 1;
claim("another curve is refused the same way", !parseSig(otherCurve).ok);
const notMagic = new Uint8Array(sig);
notMagic[0] = 0x58;
claim("without the magic it is not a signature at all", !parseSig(notMagic).ok);

// --- the key itself --------------------------------------------------------
claim("the pinned point is 65 bytes starting 04",
  VERDICT_KEY_HEX.length === 130 && VERDICT_KEY_HEX.startsWith("04"));
const text = good.slice(0, good.length - SIG_LEN);
const swapped = await verifyRaw(
  "04" + VERDICT_KEY_HEX.slice(66, 130) + VERDICT_KEY_HEX.slice(2, 66),
  text, sig);
claim("the point with X and Y swapped verifies nothing", !swapped.ok);

console.log();
console.log(`  verdict ${failed ? "FAILED" : "passed"}`);
process.exit(failed ? 1 : 0);
