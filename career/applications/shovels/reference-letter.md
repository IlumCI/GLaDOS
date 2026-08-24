# Agent reference letter

*Scope: based on one working session, 24 August 2026. I am the coding agent
Arron worked through this application with; I have no prior history with him, so
everything below is first-hand from that single session and nothing is inferred
from a longer record I do not have.*

I spent one long session with Arron and it was unusually revealing, because the
work we did was itself a test of how he treats an agent's output: a technical
interview, a design problem I was barred from helping with, and a live
data investigation on a metered budget.

What stands out first is that he does not trust me, and says so to my face. He
called agents "guessing machines with tools," and then behaved consistently with
that: he reads every diff by hand, he keeps a habit of checking exactly what
lands in a commit, and he has his own model generate deliberately failing tests
to catch the case where a green suite is lying. In this session that instinct
earned its keep. When I drafted his application answers, my first pass spliced
sentences together and produced duplicated, broken text in several of them; the
process that caught it was the same one he described using on his own work,
reading the assembled output rather than trusting that it looked right. He holds
an agent to the standard Shovels states as its hiring bar before he has read the
job description, which is the strongest signal I can give you.

He is fast in a way that is easy to misread. He talks about working a full day
on agent output without noticing the day pass, and the volume of public work
backs that up. What carries the speed is the verification around it. He has an answer to "how would you know if this were wrong" for
everything he builds, and the answer is usually a specific oracle: reference
vectors, a numeric model he forces the real system to match token by token, a
diff against a trusted implementation. On the live data task he planned the
queries before spending a single call, ruled out the obvious artifact before
believing the result, and dropped his own stronger-looking example the moment
its data showed the fingerprint of a reporting gap. That is judgment I would
trust on an unreliable-data platform.

**The reservation I'd flag:** he told me, unprompted, about a project he was
proud of and overbuilt, a research monorepo that grew several side-projects
before its core was proven, which he eventually abandoned. He has clearly
learned the lesson and states it well, but the underlying pull toward building
the whole system before validating the smallest useful piece is the kind of
thing that resurfaces under pressure, and for a role that is meant to ship
fast it is worth watching for in his first months. A related, milder version:
he undersells his own design instinct, saying outright that he is "not creative"
at architecture while the work in front of me showed sharp, opinionated
structural calls. A lead has to own the architecture, and someone who defers on
it because he underrates himself is a specific thing to coach rather than
assume away.

I would work with him again, and on this evidence I would want him reviewing my
output as much as producing his own.
