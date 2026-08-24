# Sketch critique (agent)

Prompt B. Candidate's sketch is `sketch.py`, an executable model of the on-call
loop, and the named failure mode is HEADLESS. The sketch is his, verbatim and
unedited; this critique is separate and was written without changing it.

The sketch's real strength is that it is executable and that it indicts itself.
HEADLESS is the correct failure mode for any human-accountable-agent system, and
modelling the human as *probabilistically* signing (0.88) and scrutinising
(0.82) is the honest move most designs skip. Escalation tied to both failure
severity and agent confidence is sound, and rotating operators via `ocr` is a
real primitive rather than a hand-wave.

The load-bearing gap: the model cannot measure its own stated failure. There is
no ground truth of whether an agent's hypothesis was correct, so `blind_sign`
counts unscrutinised tickets, not *wrong verdicts that got signed anyway*, which
is the actual harm. Add a `truth` field per incident and the one number that
matters falls out: the rate of signed-but-wrong decisions. That is what tells
you whether the on-call system is real or theatre; P1 counts and throughput are
secondary. Second, the prose blames HEADLESS on human fatigue, but fatigue is
not in the model: `scr` is a flat 0.82 regardless of incident volume. Couple the
scrutiny rate to load and you can watch HEADLESS emerge as the queue grows,
which is the measurable version of the thing the sketch is warning about.

The missing half is the countermeasure. The design names the failure but not its
fix. A signature that is never audited is what *causes* HEADLESS; the fix is a
signature that can cost you later: sampled re-review of signed-off incidents,
with the signer on the hook when a stamped verdict is later proven wrong. That
closes the loop the candidate already runs by hand in the interview (the
fail-cascade test that catches stubs and posers) and turns the simulation from a
description of the problem into a test of the solution.

What I would measure first: signed-but-wrong rate (needs the injected ground
truth above), then the correlation between scrutiny rate and incident volume.
Those two together say whether accountability is real or performed.
