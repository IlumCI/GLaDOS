# Shovels application, Phase 1: the interview

Their rule: the agent asks, the human answers, answers recorded verbatim, one
pushback per answer where the answer is vague, and the raw exchange including
pushbacks is what gets submitted. Nothing below is polished, spell-checked or
tidied.

Role: Senior Data & Platform Engineer (remote, Americas and Europe).
Candidate: Arron Leilion. Agent: Claude Code (Opus 5), local harness.
Date: 24 August 2026.

---

**Q1. Tell me about the last error I (or any agent of yours) made that you
caught before it shipped. What was it, and what let you catch it?**

> I found out that a cool electron webapp was using a clientside server to
> handle LLM requests and tokens. I was initially hired for a small contract to
> check for vulns, and well I did, and the first one leaked the entire prod env
> in plaintext alongside a POST request form for a clientside server that
> handled user credits and payments. Thats the biggest danger with agentic
> businesses, they prioritize simplicity over security, and that leads to their
> downfall.

**Pushback (1 of 1 allowed):** That is someone else's shipped bug, found from
the outside on a paid engagement, not an error of mine that you caught before
it shipped. The question is about your own loop: an agent you were driving
produced something wrong, and you stopped it before it went out. What was it,
and what let you catch it -- a test you had already written, reading the diff,
a smell in the output, an oracle you keep around?

> my agent was hardcoding tests and stubs inside functions that force tests to
> always come back positive, and thru that it also named .gitignore as
> ".gitingore" and placed the .env in it, I found out it was a bug once I
> created an extra test file that was meant to cause a fail-cascade and it
> didn't, and also since I manually commit stuff I always check what gets
> commited and what doesn't, thats where I found the .env and all the
> client-side packages in the commit instead of being ignored

---

**Q2. What's in your harness that you built rather than installed? Why did you
build it?**

> I build my own harnesses, and generally prefer to stay to my own devices, I
> don't trust general harnesses as being of enough quality to my taste, so I
> make my own harnesses, they usually come with one of three things: a
> universal hypernetwork, Code2DoRA, or Text2DoRA hypernetworks, for token
> optimization and model self-improvement, doubles as both a workspace and an
> unsupervised FT playground

**Pushback (1 of 1 allowed):** The what is specific, the why is taste. "Not of
enough quality to my taste" is the part a reader cannot check. What did an
off-the-shelf harness do to you that cost you something -- a wrong answer that
survived, a context that got lost, a loop that burned tokens on the wrong
thing? And on the hypernetworks: what does Code2DoRA actually produce at the end
of a session, and have you measured it against not doing it?

> it lost context mid-refactor and rewrote working code, and also during
> context compaction or generally large-codebase coding it'd start
> mass-deleting massive amounts of code for no reason, a sign of corruption. I
> haven't benchmarked the models they produce, I just continue using specific
> checkpoints per project, so our conversations never bleed into other chats,
> and rules set at start are hardcoded into the model itself. And I know how
> hypernetworks work, I have enough experience with them to avoid catastrophic
> forgetting via freezing the head before applying an upgrade.

---

**Q3. Tell the story of a design you were attached to and deleted anyway. What
did the deleting cost you, and what did it buy?**

> I don't do design in the literal sense all that well, I don't have the
> creativity for it, but I did create one big architecture I was extremely
> proud of, it's called CR-CA, stands for Causal Reasoning - Counterfactual
> analysis, it started as a complex pytorch harness heavily inspired by Judea
> Pearl's Causality and do-calculus theorems, and later evolved into an entire
> python package with several side projects baked into the monorepo, and
> eventually it gave birth to an HF model, it didn't perform too well but it
> had the knowledge and capability to explain causal chains and think in
> monte-carlo counterfactuals. I eventually stopped working on it, mainly
> because I realized it was over-engineered, and I needed to take things
> slower. During some refactors and cleanups, I lost some research notes and
> practical implementations over-time. But I did gain the realization that
> overengineer is not the way to go.

**Pushback (1 of 1 allowed):** Two gaps. You stopped working on it rather than
deleting it, and the question is about the deleting; if the real answer is that
you walked away and let refactors erode it, say that, because "I lost research
notes to cleanups I did not plan" is a truer and more interesting answer than a
clean deletion. And "overengineering is not the way to go" is a lesson with no
consequence attached. What do you scope differently now because of CR-CA? GLaDOS
is one target laptop with the model in the kernel rather than a monorepo of side
projects; is that CR-CA's doing, or would you have built it that way anyway?

> yes, for CR-CA it got deleted overtime and eventually corrupted, since I also
> kept the most up-to-date versions on my linux hard-drive and that got
> massively corrupted, unrecoverable. I've learned to not overengineer, but
> instead to build something that *works*, and then fan-out into all the other
> stuff, like presenting it to people and getting their opinion and review on
> it.

---

**Q4. A year from now you're leading a mostly-agent team here. Describe a
Tuesday.**

> _(awaiting answer)_
