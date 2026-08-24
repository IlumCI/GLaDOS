# Every answer, before it goes anywhere

Eight filled forms, verbatim. Mark up anything you want changed and I will
refill and submit. Nothing here has been sent.


---

## Poolside — Member of Engineering (Inference Infrastructure)

*Remote (EMEA). Fields: name, email, links, resume, location, cover letter, one yes/no.*


**Linkedin / Personal site**

> https://ilumci.github.io/portfolio/ | https://github.com/IlumCI | https://www.linkedin.com/in/arron-leilion-37a699350/


**Current location**

> Vilnius, Lithuania


**Do you have hands-on experience optimizing inference request serving?**

> Yes


**Cover letter**

> I build inference engines for hardware that should not be able to run the model.
>
> kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM. Routing activates 16 experts of 896 per layer per token, so 96.3% of the weights are dormant at any moment and live on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB of host memory; only the seconds per token move, 26.5 down to 5.6.
>
> RustLMHub is the general version of that: FFN weights stream from NVMe through O_DIRECT while attention and embeddings stay resident, with an int8 activation kernel (2.3 to 2.55x on matmul), speculative decoding off the model's own multi-token head (1.23 to 1.28x), certified activation sparsity that skips up to 27.6% of neurons only where the output is provably byte-identical, and LUT-GEMM for 4-bit weights on AVX2. 267 tests, differential and bit-exact against reference implementations. An inference engine that is subtly wrong still produces fluent text, which is why the tests are differential rather than smoke tests.
>
> On the serving side, I spent a year at Swarms Corporation on multi-agent orchestration: routing, scheduling, and the accounting of which model gets asked what. That work cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost. Those figures are the company's own measurements.
>
> GLaDOS is a ring-0 operating system in Rust with Qwen3 running int8 inside the kernel: 40,000 lines across 93 files, its own TLS 1.3, TCP/IP stack, NVMe driver, content-addressed store and window manager, with the cryptography checked against published RFC vectors at every boot. It is why I am comfortable at the level inference infrastructure actually lives at, where the page cache, the allocator and what the disk is really doing decide the throughput.
>
> Vilnius, EU citizen, available now, used to working remotely. Everything above is public: https://ilumci.github.io/portfolio/


---

## Modal — Member of Technical Staff, Systems

*Stockholm, $140-200k published. The form has no free-text field at all.*


**Name / Email / Resume**

> Arron Leilion / ilumbackup@gmail.com / resume.pdf


**Are you excited and able to work from our Stockholm office?**

> Yes


---

## Prime Intellect — Member of Technical Staff, Inference

*Remote.*


**Tell us about (or post links to) cool things you've built**

> kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM. Routing activates 16 experts of 896 per layer per token, so 96.3% of the weights are dormant at any moment and live on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB of host memory; only the seconds per token move, 26.5 down to 5.6.
>
> RustLMHub is the general version of that: FFN weights stream from NVMe through O_DIRECT while attention and embeddings stay resident, with an int8 activation kernel (2.3 to 2.55x on matmul), speculative decoding off the model's own multi-token head (1.23 to 1.28x), certified activation sparsity that skips up to 27.6% of neurons only where the output is provably byte-identical, and LUT-GEMM for 4-bit weights on AVX2. 267 tests, differential and bit-exact against reference implementations.
>
> GLaDOS is a ring-0 operating system in Rust with Qwen3 running int8 inside the kernel: 40,000 lines across 93 files, its own TLS 1.3, TCP/IP stack, NVMe driver, content-addressed store and window manager, with the cryptography checked against published RFC vectors at every boot.
>
> All three are public: https://github.com/IlumCI. The short version of the through-line is on https://ilumci.github.io/portfolio/.


**What do you optimize for in life?**

> Being able to answer one question about my own work: how would I know if this were wrong?
>
> Most of what I build fails quietly rather than loudly. GLaDOS spent weeks with its rotary embeddings pairing dimension i with 2i instead of i + d/2. Both are norm-preserving rotations by the same angles, so there was no NaN, no drift and no crash; the model stayed fluent and attended by a scrambled notion of distance, which is indistinguishable from a small model being small. Nothing caught it but a numeric oracle the kernel had to agree with token by token. So I optimise for building the thing that disagrees with me, and for staying at a scale where I can still read everything I ship.


**Why are you interested in joining Prime Intellect?**

> Prime Intellect's premise is that the hardware people already have is enough if the system is built properly. That is the same premise as my last two projects, one layer up. kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM, by keeping the 96.3% of experts that are dormant per token on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB; only the seconds per token move.
>
> Distributed training and decentralised inference are the version of that problem I have not had a cluster to work on, and inference is where I am strongest. I would come in on the inference side, on the parts where memory hierarchy, quantisation and scheduling decide throughput, and I have shipped orchestration work in production before: a year at Swarms Corporation on multi-agent orchestration cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost, on the company's own measurements.
>
> https://ilumci.github.io/portfolio/


---

## Lovable — Software Engineer, Platform (Runtime)

*Stockholm, on-site.*


**Legal right to work / visa sponsorship needed**

> Yes / No


**Earliest date you can join**

> Immediately


**Compensation expectations**

> EUR 70,000-100,000 per year, open to the band for the role in Stockholm.


**Why do you want to join Lovable specifically and what makes you a great fit?**

> Lovable's product is the speed I already work at, sold to everyone else, and the Platform (Runtime) team is where that speed is either real or a demo. Cold starts, isolation, the blast radius of one user's build, what a sandbox is allowed to touch: that is operating-systems work with a product deadline attached, and operating systems are what I do.
>
> GLaDOS is a ring-0 operating system in Rust with Qwen3 running int8 inside the kernel: 40,000 lines across 93 files, its own TLS 1.3, TCP/IP stack, NVMe driver, content-addressed store and window manager, with the cryptography checked against published RFC vectors at every boot. A composited window manager, a scheduler, a TCP stack and a content-addressed store are the same primitives a hosting runtime needs, written by hand rather than configured.
>
> The other half is agents. A year at Swarms Corporation on multi-agent orchestration cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost, on the company's own measurements. I know where multi-agent systems waste money and time because I spent a year taking it out of one.
>
> I am also not going to pretend about how I work: most of my code is written with coding agents driving, and my site says so and then spends its longest section on the verification discipline that makes it hold up. That seems relevant at Lovable specifically. https://ilumci.github.io/portfolio/


**What is the most impressive thing you've done in your career?**

> Writing an operating system, in Rust, from nothing, with a language model living inside the kernel.
>
> GLaDOS is a ring-0 operating system in Rust with Qwen3 running int8 inside the kernel: 40,000 lines across 93 files, its own TLS 1.3, TCP/IP stack, NVMe driver, content-addressed store and window manager, with the cryptography checked against published RFC vectors at every boot. No user/kernel split and no syscall boundary, because in that arrangement a tool call from the model is a function call. The only code in the tree I did not write is Rust's core and 509 hardware constants transcribed from Linux because no datasheet publishes them.
>
> The part I am actually proud of is not the size, it is the checking. There is no host test runner for a UEFI binary, so the boot log is the test suite: heap, clock, namespace, crypto against eleven sets of RFC vectors, constrained decoding and the linear probe, pass or fail per line, every boot. The model path is verified against a NumPy oracle reading the same converted checkpoint, token by token, because a wrong attention implementation still writes fluent sentences.


**Is there anything else you'd like us to know about you?**

> Three things worth saying plainly.
>
> I am in Vilnius and ready to relocate to Stockholm for this role. EU citizen, so there is no permit to arrange and no notice period to work out; I can start immediately.
>
> First, I have one year of formal employment behind me, and the work I am pointing at is public and checkable rather than credentialled: https://github.com/IlumCI, 138 repositories, 387 merged pull requests.
>
> Second, most of it is written with coding agents driving. Forty thousand lines of kernel Rust in three weeks is not a typing speed I have. What it takes is knowing what to ask for, reading everything that comes back, and having an oracle for every subsystem so the agent's confident wrong answers do not survive. My site walks through one of those failures in detail: https://ilumci.github.io/portfolio/


**How did you hear about us?**

> Website


---

## Hugging Face — Low-level Senior Software Engineer, Xet Storage

*EMEA remote. THE ONE TO REWRITE: they ask you to confirm the application is 'true and your own' and say they read every answer.*


**Eligible to work in the country you are applying?**

> YES


**Everything in this application is true and your own?**

> YES


**Did you start your first written answer with the exact phrase from the job description?**

> YES


**Hands-on experience in low-level software engineering, scaling distributed systems, storage, or networking infrastructure?**

> YES


**Cover letter**

> Systems engineer in Vilnius. I wrote a ring-0 operating system in Rust with a content-addressed store and an NVMe driver in it, and an inference engine that streams weights off disk so a 2.78-trillion-parameter model runs on 8.24 GB of RAM. Both are the shape of xet-core's problem from the other side. Everything is public: https://ilumci.github.io/portfolio/


**Why Hugging Face, and where would you make the biggest difference in this role?**

> GPU-poor and proud 🤗 That line is not a joke to me: everything I have built in the last year runs on one laptop with 16 GB of RAM and no GPU, on purpose.
>
> Xet is the part of Hugging Face I have accidentally been rehearsing for. GLaDOS, my ring-0 operating system in Rust, has a content-addressed object store in it: objects named by the SHA-256 of their contents, assembled into Merkle trees, so a copy is O(1) and a snapshot is one root hash, sitting directly on an NVMe driver I also wrote. The lesson I paid for there is the one xet-core lives by: the content hash covers content only and never block locations, because otherwise moving a block renames an object.
>
> The other half of Xet is moving those bytes to people who do not have the memory for them, and that is exactly what RustLMHub does. RustLMHub is the general version of that: FFN weights stream from NVMe through O_DIRECT while attention and embeddings stay resident, with an int8 activation kernel (2.3 to 2.55x on matmul), speculative decoding off the model's own multi-token head (1.23 to 1.28x), certified activation sparsity that skips up to 27.6% of neurons only where the output is provably byte-identical, and LUT-GEMM for 4-bit weights on AVX2. 267 tests, differential and bit-exact against reference implementations.
>
> Where I would make the biggest difference: the client side of hf-xet and the chunking/dedup path in xet-core, where low-level Rust, disk behaviour and correctness-under-concurrency decide whether 200 PB is pleasant or painful. I should be straight about the requirement: I do not have 8 years, I have one year of employment and a year of public systems work. Read the repositories and decide from those.
>
> https://ilumci.github.io/portfolio/


**Share 2-3 concrete examples of low-level work you've done**

> Three, all of them mine end to end, all public.
>
> 1. Content-addressed storage inside a kernel. GLaDOS (https://github.com/IlumCI/GLaDOS) has an NVMe driver and a store where files are named by the SHA-256 of their contents and assembled into Merkle trees. Writes are locked by default and every error path re-locks, because a safety mechanism you can leave open is decoration. Same tree: TLS 1.3, X25519, ChaCha20-Poly1305, ECDSA over P-256 and P-384, all written from scratch and checked against eleven sets of published RFC vectors at every boot, and a TCP/IP stack whose polling layer queues rather than dispatching, because a state machine driven from inside poll can re-enter its own control block while an earlier borrow is still live.
>
> 2. Streaming weights off NVMe. RustLMHub is the general version of that: FFN weights stream from NVMe through O_DIRECT while attention and embeddings stay resident, with an int8 activation kernel (2.3 to 2.55x on matmul), speculative decoding off the model's own multi-token head (1.23 to 1.28x), certified activation sparsity that skips up to 27.6% of neurons only where the output is provably byte-identical, and LUT-GEMM for 4-bit weights on AVX2. 267 tests, differential and bit-exact against reference implementations. My role: sole author.
>
> 3. Making memory optional. kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM. Routing activates 16 experts of 896 per layer per token, so 96.3% of the weights are dormant at any moment and live on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB of host memory; only the seconds per token move, 26.5 down to 5.6. Portable C99, no BLAS, no GPU.
>
> The impact of the third is the one I would point at in an interview: it moves "you need a 1.5 TB machine" to "you need a disk", which is the same move Xet makes for model repositories.


**Tell us about something you've built on top of our tools**

> GLaDOS's whole model pipeline is built on your tools, and it broke in ways that taught me the most.
>
> The kernel runs Qwen3-0.6B and SmolLM2-135M. The converter reads the safetensors checkpoints from the Hub and flattens them into a format the kernel indexes by arithmetic, and the tokenizer is a no_std reimplementation of the BPE and pre-tokenizer that the tokenizers library defines. The rule I ended up with is that every conversion must be verified against your implementation rather than eyeballed: the tokenizer converter runs with --verify, which reimplements the kernel's algorithm and diffs it against the reference tokenizers library, because a tokenizer that is subtly wrong produces text that still looks like text.
>
> The tricky parts, both silent:
>
> Pre-tokenizer regex. SmolLM2 trains with the GPT-2 pattern and Qwen3 with the cl100k one, where a word may be led by any non-alphanumeric, digits come one at a time and punctuation swallows following newlines. Using the wrong one moved about 12% of tokens on my training corpus, with no error, just a model fed sequences it never saw.
>
> RoPE pairing. HuggingFace's modeling code pairs dimension i with i + d/2 (rotate_half), and I had implemented the interleaved convention. Both are norm-preserving rotations by the same angles, so nothing failed; the model just attended by a scrambled notion of distance. It cost the difference between "The capital of France." and "The capital of France is Paris." What settled it was building a NumPy oracle that reads the same converted checkpoint and comparing logits token by token.
>
> Public, with the reasoning written down as it happened: https://github.com/IlumCI/GLaDOS


---

## Hugging Face — Wild Card

*Remote. Also worth your own pass.*


**Why you are applying to work at Hugging Face**

> I have spent a year proving that the interesting work does not need a cluster: an operating system in Rust with a model running inside the kernel, an engine that streams weights off NVMe so a 2.78-trillion-parameter model runs on a 8 GB laptop, and before that a year of multi-agent orchestration in production.
>
> Hugging Face is where that work would matter to people other than me. The Hub is what I already build against: I convert safetensors checkpoints from it, and I verify my no_std tokenizer against your tokenizers library, because that is the only way to know a reimplementation is right.
>
> I am applying through the Wild Card because I do not fit a requisition: one year of formal employment, no degree, and 138 public repositories with 387 merged pull requests. https://ilumci.github.io/portfolio/


**The project you would be most excited to work on in your first 3 months**

> hf-xet and the client side of Xet storage, if the Xet team will have me: it is content-addressed storage in Rust, and I wrote one inside a kernel, on top of my own NVMe driver, plus an inference engine that streams weights off disk through O_DIRECT. Those are the two halves of the same problem.
>
> Failing that, the thing I would most like to do in three months is candle: a Rust inference runtime is exactly where my last two projects live. kimi-k3-in-rust runs Kimi K3, 2.78 trillion parameters and 1.56 TB of weights, on one CPU with 8.24 GB of RAM. Routing activates 16 experts of 896 per layer per token, so 96.3% of the weights are dormant at any moment and live on disk behind an LRU cache. Output is byte-identical from 8 GB to 224 GB of host memory; only the seconds per token move, 26.5 down to 5.6. I would want to bring the streaming and the memory-hierarchy work into a library people already use, and the byte-identical property with it, because "more memory buys seconds and nothing else" is a promise a runtime can make and very few do.
>
> Either way, I would spend the first weeks reading and fixing small real things rather than proposing architecture.


---

## Railway — Infrastructure Engineer

*Remote worldwide. One question only.*


**Why Railway?**

> Your posting lists home-rolled hypervisors, virtio device drivers, container orchestration, overlay networks and racking servers as the fun part. That is the list I have been working through for a year, at the level below where most people stop.
>
> GLaDOS is a ring-0 operating system I wrote in Rust for one specific laptop: 40,000 lines across 93 files, no libc, no Unix inheritance. It has its own PCIe enumeration and device drivers (e1000, RTL8168, NVMe, xHCI, PS/2), its own paging and physical memory management, a cooperative scheduler with a context switch pinned to extern "sysv64" because the UEFI target's extern "C" is Microsoft x64, a TCP/IP stack, TLS 1.3 written from scratch and checked against RFC vectors at every boot, and a content-addressed object store on top of the NVMe driver where objects are named by the SHA-256 of their contents. Storage, networking and orchestration are not abstractions I read about; they are code I have debugged with a serial cable and a fault handler.
>
> The other half of your pitch is agents managing infrastructure, and that is the rest of my last year. A year at Swarms Corporation on multi-agent orchestration cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost. GLaDOS itself is built around a model living inside the kernel, where a tool call is a function call, and most of its code is written with coding agents driving. I would rather say that plainly than have you work it out. What makes it hold up is that every subsystem has an oracle, so a confident wrong answer does not survive contact with the tests. My site spends its longest section on that, including the bug that stayed fluent for weeks: https://ilumci.github.io/portfolio/
>
> I have one year of formal employment and no degree. The repositories are the argument: https://github.com/IlumCI


---

## Langfuse — Senior Backend Engineer (Data Infrastructure)

*Europe. They publish EUR 90-160k, which is above the 70-100k band you gave me; the answer below uses the lower half of theirs.*


**Currently located**

> Vilnius, Lithuania (EET, UTC+2)


**EU work permit?**

> Yes


**Salary expectation**

> EUR 90,000-110,000, guided by your published EUR 90-160k band


**Earliest possible start date**

> 1 September 2026


**Why do you care about Langfuse?**

> Because I have been burned by exactly the problem Langfuse exists to solve, and I built a worse version of it for myself.
>
> I spent a year at Swarms Corporation on multi-agent orchestration, where routing and orchestration work cut large-scale API and inference spend by 89% and took workflow execution to 118x its previous speed at 21% of the original cost. None of that was possible until we could see where the tokens and the seconds actually went.
>
> Then I built an evaluation harness for my own model work, and the reason it exists is that I got the measurement wrong three separate times: a grid sweep scored on the test set, cross-validation folded by template family, and a test set that moved whenever the corpus was appended to. The fix was structural: three splits, validation spent freely, the test slice read once, and corpora that hold out whole template families rather than sampled instances, because instances within a family differ only by slot values, so an instance split measures memorisation while looking like generalisation. I also keep negative results in the tree. Training an adapter head hurt at this data scale, and the Product-of-Experts council did not improve accuracy; both are still in the repository, because the reason to know them is the reason they were worth measuring.
>
> That is your product thesis from the inside: agents fail quietly, and the only thing that catches a quiet failure is a trace and an eval you designed before you trusted it.
>
> What I bring to the data infrastructure side specifically: I write systems software. GLaDOS is a ring-0 operating system in Rust, 40,000 lines, with its own NVMe driver, TCP/IP stack, TLS 1.3 and a content-addressed store; RustLMHub streams model weights off NVMe through O_DIRECT with an int8 kernel and 267 differential, bit-exact tests. High-cardinality ingest at ClickHouse scale is a problem I would enjoy at the level of what the disk is really doing.
>
> One thing to know: most of my code is written with coding agents driving, and my site says so and then spends its longest section on the verification that makes it hold up. https://ilumci.github.io/portfolio/
>
> Public work: https://github.com/IlumCI


---

## Already sent, for the record

*These three went out on 24 Aug 2026 and cannot be edited.*


**Zed Industries, jobs@zed.dev**

> See applications.md, message 1a0332d131fc4f5c.


**Proxybase, jobs@proxybase.xyz**

> message 1a03333878e12d7a


**Aqora Quantum, jannes@aqora.io**

> message 1a03333ec92c202c
