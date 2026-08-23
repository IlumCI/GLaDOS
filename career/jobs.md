# Where to apply

Checked 23 August 2026. Every posting below was open on that date; boards move,
so re-check before writing anything long. Anything I could not verify myself is
marked as such rather than being listed as if I had.

Two things shape this list. First, the strongest evidence you have is a
repository, not a CV line, which means the places worth your time are the ones
where an engineer reads the repository. Second, you are applying without a
degree and with one year of formal employment, so seniority filters are the
thing that will actually stop you, not ability. Both point the same way: small
teams, open-source-facing companies, and anywhere the application asks for code.

## Tier 1: apply this week

These are the ones where your work maps onto what the team does, closely enough
that a hiring engineer will see it in ten seconds.

| Company | Role | Where | Why it fits |
| --- | --- | --- | --- |
| **Hugging Face** | [Low-level Senior Software Engineer, Xet Storage, EMEA remote](https://apply.workable.com/huggingface/jobs/view/F4C096B22E) | EMEA remote, hiring out of Paris | `xet-core` is content-addressed storage in Rust for 200+ PB of model weights. You wrote a content-addressed Merkle store in Rust inside a kernel, and RustLMHub streams weights off NVMe for a living. Asks for 8+ years, which you do not have; apply anyway, and lead with the two repositories rather than the years. |
| **Hugging Face** | [Open-Source ML Engineer, EMEA remote](https://apply.workable.com/huggingface/jobs/view/81B46579FE) · [Senior Python Engineer / OSS contributor, EMEA remote](https://apply.workable.com/huggingface/jobs/view/CB1DEFE6CE) | EMEA remote | They hire on public contribution history. 387 merged PRs is the exact currency here. |
| **Hugging Face** | [Wild Card](https://apply.workable.com/huggingface/jobs/view/0BD8C06DB3) | US remote | Their open application. It exists for people whose work does not fit a req, which is you. Cheap to send, no seniority filter to fail. |
| **Cursor (Anysphere)** | [Software Engineer, Agent Harness](https://cursor.com/careers) and Model Routing & Inference | SF / NY, a few roles remote | You built an agent harness (`harness-sdk`) and cut inference spend 89% with routing work. Two of their thirty teams are literally named after those. SF-weighted, so treat it as a relocation application. |
| **Lovable** | [Forward Deployed Engineer](https://lovable.dev/careers) (multiple locations), Software Engineer Platform (Runtime / Infrastructure) | Stockholm, London, multi-location | The vibecoding company, in your timezone, hiring hard. Culture line on their own page is "move fast, ship often, results over talk", which is a fair description of your GitHub. |
| **Zed Industries** | [Open Source Engineer](https://zed.dev/jobs) | Fully remote, "work from anywhere" | A Rust editor team of a dozen people who read code for a living. They also invite direct mail to jobs@zed.dev when nothing fits. |

## Tier 2: strong fit, longer odds or slower process

| Company | Role | Where | Note |
| --- | --- | --- | --- |
| **Anthropic** | Forward Deployed Engineer; Applied AI Engineer, Enterprise | Munich, Paris, London ([careers](https://www.anthropic.com/jobs)) | Munich FDE has been published in the €205-220k range. Bar is high and the process is long, but applied roles weigh shipping over credentials more than research roles do. |
| **Anthropic** | Staff Software Engineer, Inference | London, Dublin | Exactly your subject matter. "Staff" is the filter you will fail; worth one application anyway if you can point at RustLMHub's kernels and speculative decoding. |
| **Ollama** | [Careers board](https://jobs.ashbyhq.com/ollama) | Palo Alto / remote | Local inference, small team. Board renders in JavaScript so I could not read the listings; check it in a browser. |
| **Qdrant** | Rust engineering, Berlin | Berlin / EU remote | Rust vector database, EU-based, hires Rust people without ceremony. Their careers page 404s and the Lever board blocks fetching, so find the current link from their GitHub org readme. |
| **Poolside** | [Careers](https://jobs.ashbyhq.com/poolside) | Paris | Code models, EU-headquartered, Rust and systems-heavy. JS-rendered board, check by hand. |
| **Prime Intellect** | [Careers](https://jobs.ashbyhq.com/PrimeIntellect) | SF / remote | Distributed training and inference on hardware people said was not enough, which is the thesis of both your inference repos. JS-rendered board. |
| **Mistral** | Paris, inference and applied | Paris | Board refuses automated fetches; go through mistral.ai/careers. Closest large EU lab to your work. |
| **Glean** | [Founding Forward Deployed Engineer](https://job-boards.greenhouse.io/gleanwork/jobs/4651991005) | US-weighted | Listed because founding FDE roles are the fastest-growing category in this market. Check the location line before spending time. |

## Tier 3: Vilnius, on-site or hybrid

Lithuanian law requires salary ranges in ads, so unlike everything above you can
see the number before applying. Monthly gross, from cvbankas.lt on 23 Aug 2026:

| Company | Role | Salary (gross/month) |
| --- | --- | --- |
| Novian Pro | ML/AI Engineer, Vilnius | €5,000-8,000 |
| Ignitis Group | AI Engineer, Vilnius | €4,290-6,440 |
| NFQ Technologies | Senior AI Engineer (Python), Vilnius | €3,850-6,600 |
| Avion Express | Senior Data Scientist, Vilnius | €4,000-6,000 |
| Vilniaus vandenys | AI engineer, Vilnius | €4,000-5,000 |
| Eika Group | AI tooling specialist / coordinator, Vilnius | €2,500-4,000 |

Also worth approaching directly, no specific ad verified: **Nord Security**
(60 open engineering roles, has an AI tooling team), **Oxylabs** (software
engineer, AI products), **Vinted**, **Salesforge** (backend for AI agents),
**Whatagraph**, **Hostinger**.

The honest read on this tier: the ceiling is roughly a third of what remote
US work pays, and none of these companies does anything as interesting as what
you do on weekends. Its value is a salaried floor while you apply upward, and
a Lithuanian employer will not blink at your age the way a US startup's
recruiter might.

## Boards worth a standing weekly check

- [goodvibecode.com](https://www.goodvibecode.com/jobs/remote) and
  [remotevibecodingjobs.com](https://remotevibecodingjobs.com/whos-hiring):
  aggregate roles that explicitly want AI-assisted developers. ~1,000 open
  positions across ~400 companies as of this month. Filter hard, and verify the
  country restriction on anything labelled "remote".
- [vibehackers.io/jobs](https://vibehackers.io/jobs) for the same, smaller and
  more startup-weighted.
- [Y Combinator, remote software engineering](https://www.ycombinator.com/jobs/role/software-engineer/remote):
  the single best source for teams that will hire on a repository. Sort by
  recently funded and mail founders directly.
- [Wellfound Europe AI](https://wellfound.com/role/l/ai-engineer/europe) and
  [remoterocketship](https://www.remoterocketship.com/country/europe/jobs/ai-engineer/)
  for EU-remote listings with published ranges (€47-158k).
- Hacker News "Who is hiring", first working day of each month. Search the
  thread for `Rust`, `inference`, `agents`, `EU remote`.
- [startup.jobs Lithuania](https://startup.jobs/locations/lithuania) for the local end.

## What to lead with, per audience

Do not send the same first paragraph to all of them.

- **Rust and inference teams** (HF, Zed, Qdrant, Ollama, Poolside): open with
  kimi-k3-in-rust. "2.78 trillion parameters on 8 GB of RAM, byte-identical
  output from 8 GB to 224 GB" is a sentence that gets a repository opened.
  GLaDOS second, as evidence you go all the way down.
- **Agent and orchestration teams** (Cursor, Anthropic applied, Lovable): open
  with the Swarms numbers, because they are commercial and measured: 89% off
  inference spend, 118x on workflow execution at 21% of the cost. Then the
  harness work.
- **Forward-deployed roles**: they are buying judgement under someone else's
  constraints. Lead with GLaDOS as a story about correctness rather than
  novelty: RFC vectors at every boot, a NumPy oracle the kernel has to agree
  with token by token, the RoPE bug that produced fluent nonsense for weeks and
  the discipline that eventually caught it.
- **Vilnius employers**: lead with the year at Swarms and the fact that you
  have shipped in production, then the projects. Local hiring is more
  conservative and reads employment history first.

## Practical notes

- **Age and degree.** Neither belongs in a cover letter. Your resume no longer
  mentions university at all, and nobody will ask about a degree if the first
  link they open is a kernel. Where an application form insists on years of
  experience, put the real number and let the work argue.
- **Getting paid.** Most US "remote" postings mean US-authorised. The ones that
  genuinely hire in Lithuania either use an employer of record (Deel, Remote,
  Velocity Global) or contract with you as an individual. Ask which in the first
  call; it decides whether an offer can exist at all.
- **Volume.** Ten careful applications beat sixty templated ones, but ten is the
  floor, not the target. Expect most to go unanswered for reasons that have
  nothing to do with you.
- **The portfolio page.** Link it in every application. It says out loud that
  the work is agent-assisted and then shows the engineering judgement that makes
  that a strength rather than a confession. Anyone who is going to hold it
  against you will do so at some point anyway; better it happens before you have
  spent four interview rounds.
