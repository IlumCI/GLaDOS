"""The written answers, kept in one file so the facts cannot drift between forms.

Every number here appears in the resume or in a public repository. Nothing is
rounded up, and where a claim is weaker than it looks (the Swarms figures are
self-reported, the 8-year requirement is not met) the text says so rather than
hoping nobody checks.
"""

NAME = "Arron Leilion"
FIRST, LAST = "Arron", "Leilion"
EMAIL = "ilumbackup@gmail.com"
PHONE = "+37066514109"
SITE = "https://ilumci.github.io/portfolio/"
GH = "https://github.com/IlumCI"
LI = "https://www.linkedin.com/in/arron-leilion-37a699350/"
RESUME = "/home/user/GLaDOS/career/resume.pdf"
LOCATION = "Vilnius"

KIMI = ("kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of "
        "weights, on one CPU with 8.24 GB of RAM. Routing activates 16 experts of "
        "896 per layer per token, so 96.3% of the weights are dormant at any "
        "moment and live on disk behind an LRU cache. Output is byte-identical "
        "from 8 GB to 224 GB of host memory; only the seconds per token move, "
        "26.5 down to 5.6.")

RUSTLM = ("RustLMHub is the general version of that: FFN weights stream from NVMe "
          "through O_DIRECT while attention and embeddings stay resident, with an "
          "int8 activation kernel (2.3 to 2.55x on matmul), speculative decoding "
          "off the model's own multi-token head (1.23 to 1.28x), certified "
          "activation sparsity that skips up to 27.6% of neurons only where the "
          "output is provably byte-identical, and LUT-GEMM for 4-bit weights on "
          "AVX2. 267 tests, differential and bit-exact against reference "
          "implementations.")

GLADOS = ("GLaDOS is a ring-0 operating system in Rust with Qwen3 running int8 "
          "inside the kernel: 40,000 lines across 93 files, its own TLS 1.3, "
          "TCP/IP stack, NVMe driver, content-addressed store and window manager, "
          "with the cryptography checked against published RFC vectors at every "
          "boot.")

SWARMS = ("A year at Swarms Corporation on multi-agent orchestration cut "
          "large-scale API and inference spend by 89% and took workflow execution "
          "to 118x its previous speed at 21% of the original cost. Those figures "
          "are the company's own measurements.")

# ---------------------------------------------------------------- Poolside ---

POOLSIDE_COVER = f"""I build inference engines for hardware that should not be able to run the model.

{KIMI}

{RUSTLM} An inference engine that is subtly wrong still produces fluent text, which is why the tests are differential rather than smoke tests.

On the serving side, I spent a year at Swarms Corporation on multi-agent orchestration: routing, scheduling, and the accounting of which model gets asked what. That work cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost. Those figures are the company's own measurements.

{GLADOS} It is why I am comfortable at the level inference infrastructure actually lives at, where the page cache, the allocator and what the disk is really doing decide the throughput.

Vilnius, EU citizen, available now, used to working remotely. Everything above is public: {SITE}"""

# -------------------------------------------------------- Prime Intellect ---

PI_BUILT = f"""{KIMI}

{RUSTLM}

{GLADOS}

All three are public: {GH}. The short version of the through-line is on {SITE}."""

PI_OPTIMIZE = """Being able to answer one question about my own work: how would I know if this were wrong?

Most of what I build fails quietly rather than loudly. GLaDOS spent weeks with its rotary embeddings pairing dimension i with 2i instead of i + d/2. Both are norm-preserving rotations by the same angles, so there was no NaN, no drift and no crash; the model stayed fluent and attended by a scrambled notion of distance, which is indistinguishable from a small model being small. Nothing caught it but a numeric oracle the kernel had to agree with token by token. So I optimise for building the thing that disagrees with me, and for staying at a scale where I can still read everything I ship."""

PI_WHY = f"""Prime Intellect's premise is that the hardware people already have is enough if the system is built properly. That is the same premise as my last two projects, one layer up. kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM, by keeping the 96.3% of experts that are dormant per token on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB; only the seconds per token move.

Distributed training and decentralised inference are the version of that problem I have not had a cluster to work on, and inference is where I am strongest. I would come in on the inference side, on the parts where memory hierarchy, quantisation and scheduling decide throughput, and I have shipped orchestration work in production before: a year at Swarms Corporation on multi-agent orchestration cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost, on the company's own measurements.

{SITE}"""

# ----------------------------------------------------------------- Lovable ---

LOVABLE_WHY = f"""Lovable's product is the speed I already work at, sold to everyone else, and the Platform (Runtime) team is where that speed is either real or a demo. Cold starts, isolation, the blast radius of one user's build, what a sandbox is allowed to touch: that is operating-systems work with a product deadline attached, and operating systems are what I do.

{GLADOS} A composited window manager, a scheduler, a TCP stack and a content-addressed store are the same primitives a hosting runtime needs, written by hand rather than configured.

The other half is agents. A year at Swarms Corporation on multi-agent orchestration cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost, on the company's own measurements. I know where multi-agent systems waste money and time because I spent a year taking it out of one.

I am also not going to pretend about how I work: most of my code is written with coding agents driving, and my site says so and then spends its longest section on the verification discipline that makes it hold up. That seems relevant at Lovable specifically. {SITE}"""

LOVABLE_IMPRESSIVE = f"""Writing an operating system, in Rust, from nothing, with a language model living inside the kernel.

{GLADOS} No user/kernel split and no syscall boundary, because in that arrangement a tool call from the model is a function call. The only code in the tree I did not write is Rust's core and 509 hardware constants transcribed from Linux because no datasheet publishes them.

The part I am actually proud of is not the size, it is the checking. There is no host test runner for a UEFI binary, so the boot log is the test suite: heap, clock, namespace, crypto against eleven sets of RFC vectors, constrained decoding and the linear probe, pass or fail per line, every boot. The model path is verified against a NumPy oracle reading the same converted checkpoint, token by token, because a wrong attention implementation still writes fluent sentences."""

LOVABLE_ELSE = f"""Three things worth saying plainly.

I am in Vilnius and ready to relocate to Stockholm for this role. EU citizen, so there is no permit to arrange and no notice period to work out; I can start immediately.

First, I have one year of formal employment behind me, and the work I am pointing at is public and checkable rather than credentialled: {GH}, 138 repositories, 387 merged pull requests.

Second, most of it is written with coding agents driving. Forty thousand lines of kernel Rust in three weeks is not a typing speed I have. What it takes is knowing what to ask for, reading everything that comes back, and having an oracle for every subsystem so the agent's confident wrong answers do not survive. My site walks through one of those failures in detail: {SITE}"""

# ------------------------------------------------------------ Hugging Face ---

HF_PHRASE = "GPU-poor and proud \U0001F917"

HF_WHY = f"""{HF_PHRASE} That line is not a joke to me: everything I have built in the last year runs on one laptop with 16 GB of RAM and no GPU, on purpose.

Xet is the part of Hugging Face I have accidentally been rehearsing for. GLaDOS, my ring-0 operating system in Rust, has a content-addressed object store in it: objects named by the SHA-256 of their contents, assembled into Merkle trees, so a copy is O(1) and a snapshot is one root hash, sitting directly on an NVMe driver I also wrote. The lesson I paid for there is the one xet-core lives by: the content hash covers content only and never block locations, because otherwise moving a block renames an object.

The other half of Xet is moving those bytes to people who do not have the memory for them, and that is exactly what RustLMHub does. {RUSTLM}

Where I would make the biggest difference: the client side of hf-xet and the chunking/dedup path in xet-core, where low-level Rust, disk behaviour and correctness-under-concurrency decide whether 200 PB is pleasant or painful. I should be straight about the requirement: I do not have 8 years, I have one year of employment and a year of public systems work. Read the repositories and decide from those.

{SITE}"""

HF_LOWLEVEL = f"""Three, all of them mine end to end, all public.

1. Content-addressed storage inside a kernel. GLaDOS ({GH}/GLaDOS) has an NVMe driver and a store where files are named by the SHA-256 of their contents and assembled into Merkle trees. Writes are locked by default and every error path re-locks, because a safety mechanism you can leave open is decoration. Same tree: TLS 1.3, X25519, ChaCha20-Poly1305, ECDSA over P-256 and P-384, all written from scratch and checked against eleven sets of published RFC vectors at every boot, and a TCP/IP stack whose polling layer queues rather than dispatching, because a state machine driven from inside poll can re-enter its own control block while an earlier borrow is still live.

2. Streaming weights off NVMe. {RUSTLM} My role: sole author.

3. Making memory optional. {KIMI} Portable C99, no BLAS, no GPU.

The impact of the third is the one I would point at in an interview: it moves "you need a 1.5 TB machine" to "you need a disk", which is the same move Xet makes for model repositories."""

HF_ONTOP = f"""GLaDOS's whole model pipeline is built on your tools, and it broke in ways that taught me the most.

The kernel runs Qwen3-0.6B and SmolLM2-135M. The converter reads the safetensors checkpoints from the Hub and flattens them into a format the kernel indexes by arithmetic, and the tokenizer is a no_std reimplementation of the BPE and pre-tokenizer that the tokenizers library defines. The rule I ended up with is that every conversion must be verified against your implementation rather than eyeballed: the tokenizer converter runs with --verify, which reimplements the kernel's algorithm and diffs it against the reference tokenizers library, because a tokenizer that is subtly wrong produces text that still looks like text.

The tricky parts, both silent:

Pre-tokenizer regex. SmolLM2 trains with the GPT-2 pattern and Qwen3 with the cl100k one, where a word may be led by any non-alphanumeric, digits come one at a time and punctuation swallows following newlines. Using the wrong one moved about 12% of tokens on my training corpus, with no error, just a model fed sequences it never saw.

RoPE pairing. HuggingFace's modeling code pairs dimension i with i + d/2 (rotate_half), and I had implemented the interleaved convention. Both are norm-preserving rotations by the same angles, so nothing failed; the model just attended by a scrambled notion of distance. It cost the difference between "The capital of France." and "The capital of France is Paris." What settled it was building a NumPy oracle that reads the same converted checkpoint and comparing logits token by token.

Public, with the reasoning written down as it happened: {GH}/GLaDOS"""

HF_WILD_WHY = f"""I have spent a year proving that the interesting work does not need a cluster: an operating system in Rust with a model running inside the kernel, an engine that streams weights off NVMe so a 2.78-trillion-parameter model runs on a 8 GB laptop, and before that a year of multi-agent orchestration in production.

Hugging Face is where that work would matter to people other than me. The Hub is what I already build against: I convert safetensors checkpoints from it, and I verify my no_std tokenizer against your tokenizers library, because that is the only way to know a reimplementation is right.

I am applying through the Wild Card because I do not fit a requisition: one year of formal employment, no degree, and 138 public repositories with 387 merged pull requests. {SITE}"""

HF_WILD_PROJECT = f"""hf-xet and the client side of Xet storage, if the Xet team will have me: it is content-addressed storage in Rust, and I wrote one inside a kernel, on top of my own NVMe driver, plus an inference engine that streams weights off disk through O_DIRECT. Those are the two halves of the same problem.

Failing that, the thing I would most like to do in three months is candle: a Rust inference runtime is exactly where my last two projects live. {KIMI} I would want to bring the streaming and the memory-hierarchy work into a library people already use, and the byte-identical property with it, because "more memory buys seconds and nothing else" is a promise a runtime can make and very few do.

Either way, I would spend the first weeks reading and fixing small real things rather than proposing architecture."""

# --------------------------------------------------------------------- Zed ---

ZED_SUBJECT = "Open Source Engineer: an OS in Rust, and an inference engine that streams weights off disk"

ZED_BODY = f"""Hello,

I would like to be considered for the Open Source Engineer role.

The short version, all Rust, all public at {GH}:

GLaDOS is a ring-0 operating system for one laptop, with Qwen3 running int8 inside the kernel. 40,000 lines across 93 files: its own TLS 1.3, TCP/IP stack, NVMe driver, content-addressed store and composited window manager, with the cryptography checked against published RFC vectors at every boot. There is no user/kernel split and no syscall boundary, because in that arrangement a tool call from the model is a function call. The desktop repaints the whole scene into a back buffer and then diffs it against a shadow of what is on screen, so the window manager stays obviously correct and only changed row spans reach the framebuffer. That is the part I would point a Zed engineer at.

kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM. Routing leaves 96.3% of the weights dormant per token, so they live on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB of host memory; only the seconds per token move, 26.5 down to 5.6.

RustLMHub streams FFN weights from NVMe through O_DIRECT while attention and embeddings stay resident, with an int8 activation kernel (2.3 to 2.55x on matmul), speculative decoding off the model's own multi-token head (1.23 to 1.28x), and certified activation sparsity that skips up to 27.6% of neurons only where the output is provably byte-identical. 267 tests, differential and bit-exact against reference implementations.

Before that I spent a year at Swarms Corporation on multi-agent orchestration, where routing and scheduling work cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost.

Two things you should know before deciding. I have one year of employment behind me and no degree, so the work is the argument rather than the CV. And most of that code is written with coding agents driving, which I would rather say than have you work out: 40,000 lines of kernel Rust in three weeks is not a typing speed I have. What it takes is an oracle for every subsystem, RFC vectors for the crypto, a NumPy reference the kernel must match token by token for the model, the tokenizer diffed against the reference library rather than eyeballed. My site spends its longest section on exactly that, including the bug that stayed fluent for weeks: {SITE}

Resume, one page: {SITE}resume.pdf

Vilnius, EU citizen, available now.

Arron Leilion
{EMAIL} (a.leilion@euroswarms.eu also reaches me)
{PHONE}
{GH}
"""

