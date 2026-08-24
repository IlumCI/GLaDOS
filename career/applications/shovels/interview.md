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

> _(awaiting answer)_
