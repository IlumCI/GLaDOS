# Applications

State as of 24 August 2026. Nothing here is marked sent or submitted without a
message id or a confirmation page behind it.

Standing answers, kept identical across every form so the facts cannot drift:

| Question | Answer |
| --- | --- |
| Contact | ilumbackup@gmail.com, +370 665 14109 |
| Links | <https://ilumci.github.io/portfolio/>, github.com/IlumCI, LinkedIn |
| Right to work | EU citizen. No permit needed anywhere in the EU/EEA. |
| Sponsorship | Not required in the EU. Required for the US, the UK and Switzerland. |
| Relocation | Yes, including Stockholm, Munich, Paris, London. |
| Notice | None. Available immediately. |
| Salary | EUR 70,000-100,000 a year for remote EU, or the posted range. Optional salary fields left blank rather than anchoring low. |
| Résumé | `career/resume.pdf`, also served at <https://ilumci.github.io/portfolio/resume.pdf> |

## Sent

| Date | Company | Role | Route | Evidence |
| --- | --- | --- | --- | --- |
| 24 Aug 2026 | Zed Industries | Open Source Engineer | email to jobs@zed.dev | Gmail message `1a0332d131fc4f5c` |
| 24 Aug 2026 | Proxybase | Backend Systems Engineer (Rust), remote global | email to jobs@proxybase.xyz | Gmail message `1a03333878e12d7a` |
| 24 Aug 2026 | Aqora Quantum | Sr Full-Stack Engineer (Rust + React), Paris or EU remote, EUR 70-90k | email to jannes@aqora.io | Gmail message `1a03333ec92c202c` |

## Filled, waiting on Arron to say submit

Each of these is filled end to end, with the résumé attached and a screenshot
taken. None has been submitted. The browser session does not survive between
runs, so submitting means re-running the filler with the submit step enabled;
the text is stored in `answers.py` so the re-run puts back exactly what was
reviewed.

| Company | Role | Where | Form | What it says |
| --- | --- | --- | --- | --- |
| Poolside | Member of Engineering, Inference Infrastructure | Remote (EMEA) | Ashby | Cover letter leading with kimi-k3, then RustLMHub, then the Swarms numbers. Answered yes to hands-on inference-serving experience. |
| Modal | Member of Technical Staff, Systems | Stockholm | Ashby | Name, mail, résumé, and yes to working from the Stockholm office. No free-text field exists on this one. |
| Prime Intellect | Member of Technical Staff, Inference | Remote | Ashby | Three answers: what I have built, what I optimise for (the RoPE bug and oracles), why Prime Intellect. |
| Lovable | Software Engineer, Platform (Runtime) | Stockholm, on-site | Ashby | Right to work yes, sponsorship no, start immediately, EUR 70-100k, three essays, relocation to Stockholm stated in the third. |
| Hugging Face | Low-level Senior SWE, Xet Storage | EMEA remote | Workable | Three required essays. First one opens with the exact phrase the job description asks for, "GPU-poor and proud". Says plainly that the 8-year requirement is not met. |
| Hugging Face | Wild Card | Remote | Workable | Two essays: why Hugging Face, and hf-xet or candle as the first three months. Phone field would not accept input; it is optional. |
| Railway | Infrastructure Engineer | Remote worldwide | Ashby | One question, "Why Railway?", answered with the driver-and-scheduler half of GLaDOS against their own list: hypervisors, virtio drivers, orchestration, overlay networks. |
| Langfuse | Senior Backend Engineer (Data Infrastructure) | Europe, EUR 90-160k posted | Ashby | Salary answered inside their published band rather than the generic 70-100k. Start date 1 Sep 2026. The essay is the evaluation-harness story: three splits, family-level holdout, negative results kept. |

## Considered and not applied

| Company | Role | Why not |
| --- | --- | --- |
| Anthropic | Forward Deployed Engineer, Munich | Two required gates fail: professional German at C1 or above, and 8+ years in a technical customer-facing role. Their form asks both as required questions, so the application would be answering no twice. |
| Cursor | Agent Harness; Model Routing & Inference | The best content fit found anywhere, and both are San Francisco. US work authorisation would need sponsorship. Worth a decision rather than an assumption. |
| Baseten, Ollama | inference and runtime roles | San Francisco and Palo Alto, US only, same sponsorship problem. Baseten publishes $165-330k. |
| Vilnius employers | Novian, Ignitis, NFQ, Nord Security, Oxylabs | Held for a second wave. Local hiring reads employment history first, so these are better sent after the remote-EU replies land. |

## Needs Arron, not me

| Company | Role | What it needs |
| --- | --- | --- |
| **Shovels** | Senior Data & Platform Engineer, remote (Americas and Europe) | The closest fit found anywhere: "an engineer who runs a team of AI agents", and their shipping bar is "you may only ship work, yours or an agent's, whose errors you could have caught". Their careers page is gated to agents on purpose and hands the agent a five-phase application skill. It needs him for four of the five phases: a verbatim interview, a design sketch with no AI help at all, a data spike against their live API on a 250-call trial key, and the send itself. I write the reference letter and may not have it edited. |
| Cogram | Product Engineer, remote CET+/-3 | Apply by mail to r+hnhiring@cogram.com with a note including "your current agentic-coding setup", and they say plainly: no AI-generated emails. So he writes it. Facts to use are in this file. |
| Pango | Founding Software Engineer, Stockholm hybrid, sponsorship possible | lukasz@pango.ai, and the post says "Please don't use AI to write the initial message". He writes it. |
| Hatchet | Product Engineer, remote US and EU | Applications go through `ssh hatchet-jobs.com`. There is no ssh binary in this container, so this one is his to run, and it is the kind of door he will enjoy. |
| Kadoa | Senior Software Engineer, remote | Google Form, <https://forms.gle/JRYUvcbkcdMNejzG9>. Fillable, but the founder says his inbox is drowning in AI-generated applications, so this one is worth his own hand. |

## Notes worth keeping

- **Mail goes out from ilumbackup@gmail.com** while the résumé and site carry
  a.leilion@euroswarms.eu. Every message signs with both so it reads as
  deliberate. A Gmail send-as alias for the euroswarms address would remove the
  mismatch entirely.
- **Hugging Face reads every answer and says so.** The two essays there are
  drafts written from Arron's material; the form also asks him to confirm the
  application is "true and your own". He should read and rewrite them in his own
  words before they go.
- **Chromium in this container cannot make TLS 1.3 connections through the
  egress proxy.** Capping at TLS 1.2 (`--ssl-version-max=tls1.2`) is what makes
  the form automation work at all; certificate verification stays on.
- Every form carries a reCAPTCHA. They did not block filling. Whether they block
  an automated submit is not known until the first submit is attempted.
