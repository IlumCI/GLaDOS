# GLaDOS 1.3.8 — every commit

Eighty-three commits since the 1.3.7 bump (`124e2bb`). One sentence each.

## Benchmarks, and the five defects that made them lie

- `175f8cd` — the KV cache gets its own dtype, because it is what scales with context rather than the weights.
- `fffd6cb` — `--batch` never reached the GSM8K task, so every "batching buys nothing" figure was two runs of batch 1.
- `693ea01` — GSM8K is **39.2%** over the whole 1,319-question test set, where the rail had read 0.0% on every checkpoint ever run.
- `e2ffa40` — needle-in-a-haystack reads 6/6 on the 0.6B, and the zero it used to report was an eight-token answer budget.
- `c3da79a` — the retrieval forest was built from MMLU's **test** split, which makes it an answer key rather than a knowledge base.
- `689c07f` — SDPA is 6.5x slower than two plain matmuls at decode shape, so the "obvious" optimisation was removed.
- `a06c17b` — a forest rebuild that admits fewer nodes has to delete the ones it no longer admits, or the old ones linger.
- `fa6a058` — retrieval goes into the evaluation prompt, and the answer key is checked from the other side so a leak is visible.
- `3cc593e` — the retrieval guard was discarding exactly the best matches a forest had.
- `7587949` — `--kv8` had no route into `--check`, and a figure had already been quoted from the path that did not run.
- `7a2fed9` — adds bits-per-byte, a rail dense enough to iterate against instead of a binary one needing thousands of items.
- `88e2f39` — the MMLU rail was scoring 100 questions of `abstract_algebra` and calling it MMLU, for the life of the rail.
- `f6c2b97` — re-runs the rails and records the fifth defect of the same family.
- `9e4ef93` — retrieval measures *harmful* across the whole of MMLU, and a decomposition offered to explain it disproved itself.
- `36bfcf9` — the paged KV cache is measured on the host before a line of it is written in the kernel.
- `cffd9f7` — benches the store read path such a cache would use, and finds the blob layout decides the answer.

## Boot, and reading the verdict a selftest was already returning

- `fda726d` — a selftest's `bool` is read, so a subsystem that runs to completion and answers wrongly now stops the machine.
- `b758a29` — wraps the remaining boot checks in `main::section`, which immediately found two real bugs.
- `31fc978` — skill judge J3 compares what a candidate **printed**, which is the only field that moves for a replay skill.

## The self-modification loop, and five defects in it

- `a4067d7` — five defects in the loop that changes this machine, four of them chained.
- `52f9c1b` — documents those five and the budget that made J1 arithmetically impossible to pass.
- `6a1ae7f` — J2 asks *where* a goal went, and asks it of the variant being judged rather than the incumbent.
- `bb42a6b` — CI green stops meaning "it compiled" and starts meaning "it booted and passed its own checks".
- `e6b4c8e` — `bench report` emits every rail this machine can measure as one machine-readable block.
- `11a7f09` — `rails.py` judges whether a change moved a rail, with a declared noise floor and a control divided out.
- `d90b392` — a source axis that can propose a constant change and explicitly cannot judge it.
- `7eaef3b` — the network path out, and the two holes found on the way there.
- `21bdb8a` — a family-wise alpha budget, plus the eleven selftest sections CI had silently not been running.
- `91133ab` — the loop's two ends, one credential short of closing.
- `1fb4d05` — the rail the tunable constants claim, run over the whole of their space.
- `f68b717` — `propose.yml`, the half of the loop that can actually compile, and the verdict's return leg.
- `126a3b0` — the Aiksi JIT grows calls and recursion, which forced locals onto the machine stack.
- `8d5d076` — `clade.rs` chooses which node to grow from, by what already grew from it rather than by its own score.
- `5604803` — `progress.rs` scores how well the machine predicts its own history, in bits.
- `36372f6` — documents the shipping loop, the node chooser and the bits rail.
- `f4c110e` — `oops.rs` decides how much a night may spend, and bounds the waste of not knowing at a factor of four.
- `ac8deab` — rails measure their own floor on the day, and a build whose two readings disagree is refused rather than judged.
- `24db88b` — the OOPS level is read out of the ledger, because the first rule converged only by coincidence.
- `bbcddc1` — what a rollback decides is lifted into pure functions and asserted, since `reconsider` calls it unattended.

## The verdict's way home, and the key that signs it

- `6ad9d39` — a second trust anchor, so a verdict cannot be signed by the key that ships kernels.
- `812643b` — commits the verdict key rotation, because the pinned point had no private half anywhere.
- `7e620ac` — `sign.py` proves the key matches what the tree pins *before* signing, and reads the signature back after.
- `32a0666` — documents that check and the orphaned key point it caught on its first run.
- `dbe619c` — re-drives the verdict transcript against the key the tree now actually pins.
- `9b05cb3` — the Supabase door a machine's own proposals come in through.
- `dcbc074` — `envelope.check`, so the kernel's own bytes are read by the thing that will receive them.

## The CI Gödel machine

- `df57ee9` — the core: `godel.py`, the closed job-kind table, and the night job.
- `e5460c0` — ledger directories are made where files land rather than tracked empty, because fsck was right to refuse the placeholders.
- `05e570b` — four evidence factories: boots, floors, fuzz and sweeps, which produce data and never verdicts.
- `5a02dc6` — the verdict's way home: migration 0005, the verdict function, and the POST.
- `3bad785` — the authoring rungs and the boundary lane.
- `fc9c375` — documents the CI machine's full suite and names what is still owed.
- `92fd197` — the kernel's counterpart to the verdict: the verb, and the night that polls for one.
- `e8cd05f` — hardens the gate with cost rails, a witness arm, and a definition of what an OOPS level buys.
- `cf72d2e` — documents the return leg and what the hardening pass changed.
- `14fa08b` — closes the three owed items: peek claims, the anchor harvest, and the extra boot per arm.
- `7eb66a7` — documents those three as closed.
- `59fc9d5` — rung 3, where the compiler enumerates the candidate space instead of a model guessing at it.
- `8f42d68` — makes the existing CI gates true before a loop is built on top of them.
- `b1cc2ac` — documents the night's poll driven end to end, and the three recipe traps it cost.
- `7b07f46` — the nightly path, driven end to end for the first time.
- `df33433` — documents the loop driven again with every piece of that night in it.

## Retrieval

- `f2de3bc` — all 57 MMLU subjects filed across eight declared trees, instead of one hardcoded maths tree.
- `bdc8ebe` — a third source, so the retrieval rail stops being exam-shaped.
- `5ed1a67` — the host-side rail was scoring by a formula the kernel does not use.
- `c7f2e3f` — the IDF flag becomes a whole-number power, and the host rail does not re-confirm the value of two.
- `134dd2c` — retrieved nodes are redacted of their answer line, and `ask` is wired to the forest.
- `a5f327b` — `answer.rs`, the rail that says whether retrieval helps, with an oracle arm as its canary.

## Learning from what an action did

- `0c563f4` — the objective the plan asked for, refuted by its own algebra and replaced with a target-distribution form.
- `05ac629` — `outcome.rs` measures how far apart two applets are by what they printed, rather than assigning severity by hand.
- `f0e97d4` — documents the objective that does not work and the outcome signal that does.
- `23dfd56` — wires the objective into both training loops and measures it refusing at every mix value.
- `7a4270e` — measures the spelling defect on real logits and shrinks the claim from "half of all steps" to one decision in 113.
- `08261c7` — documents the mix axis as measured refusing, and why no axis is registered.

## Everything else

- `21158b3` — gives the TSC a reference that does not derive from the counter it is used to check, which had hidden a core-count error in every network timeout.
- `facca61` — a console capture belongs to the task that opened it, closing a bug previously filed as an interleaving artefact.
- `7014e84` — the journal tail takes a count, because eight lines of output hid the thing being looked for.
- `1334a32` — repairs a README recipe a heredoc had eaten.
- `228f302` — records that the TCP stack grew multiple connections and never grew a listener.
- `a33c8db` — the README's "every task is pinned to core 0" line goes stale the honest way.
- `6c0f0fe` — regenerates the docs from the releases API.
- `6881976` — `Cargo.lock` for 1.3.7.
- `76fcfce` — 1.3.8, and removes an accessor that already existed under another name.
