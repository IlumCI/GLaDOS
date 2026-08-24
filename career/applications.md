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
| 24 Aug 2026 | Shovels | Senior Data & Platform Engineer, remote (Americas/Europe) | email to luka@shovels.ai + attachments | Gmail message `1a033a626651a19c` |
| 24 Aug 2026 | Poolside | Member of Engineering, Inference Infrastructure, EMEA remote | Ashby form | "successfully submitted" confirmation |
| 24 Aug 2026 | Prime Intellect | Member of Technical Staff, Inference, remote | Ashby form | "successfully submitted" confirmation |
| 24 Aug 2026 | Railway | Infrastructure Engineer, worldwide remote | Ashby form | "successfully submitted" confirmation |
| 24 Aug 2026 | Lovable | Software Engineer, Platform (Runtime), Stockholm | Ashby form | "application was received" confirmation |

## Filled but blocked by anti-bot gates: needs Arron to finish

These four are filled end to end with the correct answers (all in `answers.py`
and `drafts.md`), but the final submit is gated by a human-verification or
bot-detection control that must not be evaded. Arron opens each URL, re-enters
or pastes from `drafts.md`, clears the gate, and submits.

| Company | Role | Gate | URL |
| --- | --- | --- | --- |
| Modal | MTS, Systems, Stockholm | Ashby flagged the automated submit as "possible spam" | jobs.ashbyhq.com/modal/3b3c6c42-326e-40c5-b78d-9f556739513b/application |
| Langfuse | Senior Backend Engineer (Data Infra), Europe | Ashby "possible spam" flag | jobs.ashbyhq.com/langfuse/1225fa3d-d590-41d2-b798-ef927320fb2e/application |
| Hugging Face | Low-level Senior SWE, Xet Storage, EMEA remote | Cloudflare "verify you are human" checkbox | apply.workable.com/huggingface/j/F4C096B22E/apply/ |
| Hugging Face | Wild Card, remote | Cloudflare "verify you are human" checkbox | apply.workable.com/huggingface/j/0BD8C06DB3/apply/ |

The two Hugging Face forms should carry Arron's own words anyway, since that
form asks him to confirm the application is true and his own.


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
- **Form automation, what worked and what did not.** Ashby fields are
  React-controlled: `fill()` sets the DOM value but validation reads React
  state, so text fields must be replayed right before submit and yes/no and
  radio controls must be committed by clicking a decoy option then the target
  (a single click leaves React desynced). The resume "autofill" also overwrites
  fields a few seconds after upload, so the upload runs first and a full replay
  of every field happens just before submit. With that, four Ashby forms
  submitted cleanly. The anti-bot gates (Ashby spam flag, Cloudflare Turnstile)
  are a hard stop and correctly so: they are left for Arron to clear by hand.
