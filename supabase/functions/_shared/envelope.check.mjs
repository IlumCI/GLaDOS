// The envelope the *kernel* renders, read by the code that will receive it.
//
//     drive.py ... "godel source" "godel push --dry" > out/push.log
//     node supabase/functions/_shared/envelope.check.mjs out/push.log
//
// `envelope.test.mjs` checks this reader against fixtures written beside it,
// which establishes that it is self-consistent and nothing more. This is the
// other half: two implementations that are supposed to agree do not stay
// agreeing, and here a single byte of disagreement is a proposal that silently
// means something else.
//
// It reads a `drive.py` transcript, because that is what somebody actually has
// -- `godel push --dry` prints the envelope with the shell's `  | ` in front
// of every line. Stripping that here rather than in the recipe is deliberate:
// a recipe with a `sed` in the middle is a recipe somebody gets wrong once and
// then stops running. A file that is already a bare envelope is read as one.

import { readFileSync } from "node:fs";
import { parse, render } from "./envelope.js";

const given = process.argv[2];
if (!given) {
  console.log("  usage: envelope.check.mjs <drive.py transcript, or a bare envelope>");
  process.exit(2);
}

// **Every carriage return, not one.** A `drive.py` transcript of a serial
// console carries `\r\r\n` -- the guest's own line ending and the harness's --
// so a strip that took exactly one left exactly one, and `parse` accepted it
// anyway because `trim()` eats a trailing CR. The reader was right, the round
// trip was the only thing that noticed, and that is the argument for comparing
// the rendered bytes rather than asking whether it parsed.
const CR = String.fromCharCode(13);
const onDisk = readFileSync(given, "utf8");
const lifted = onDisk
  .split("\n")
  .filter((l) => l.startsWith("  | "))
  .map((l) => l.slice(4).split(CR).join(""))
  .join("\n");
const text = lifted ? lifted + "\n" : onDisk.split(CR).join("");

const v = parse(text);
console.log(`  ${text.length} B over ${text.trimEnd().split("\n").length} line(s)`);
if (!v.ok) {
  console.log(`  FAIL   this reader would refuse what the kernel sent: ${v.why}`);
  process.exit(1);
}

const back = render(v.proposal);
const same = back === text;
console.log(`  ${same ? "ok " : "FAIL"}   the kernel's bytes round-trip through this reader`);
console.log(`         ${v.proposal.file} ${v.proposal.symbol} ${v.proposal.was} -> ${v.proposal.now}`);
console.log(`         claims ${v.proposal.rail}, from ${v.proposal.version}, point ${v.proposal.point.slice(0, 8)}`);
if (!same) {
  console.log(`  sent:  ${JSON.stringify(text)}`);
  console.log(`  built: ${JSON.stringify(back)}`);
}
process.exit(same ? 0 : 1);
