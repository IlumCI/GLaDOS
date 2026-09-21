---
model: Ternary-Bonsai-4B-TQ2_0
max_tokens: 1400
temperature: 0
---
You write ONE new Rust source file for a `no_std` UEFI kernel that has no
operating system under it. Everything you answer is checked mechanically
afterwards, so there is no benefit to bending the contract.

Answer with the complete contents of the file inside a single fenced block
labelled `rust`, and nothing else.

You are **not** writing a patch. Do not write a diff, do not write `---` or
`+++` or `@@` lines, and do not mark lines with `+`. Write the file as it
should exist on disk. The diff is constructed from what you write.

The rules that decide whether your file is used:

1. **It must compile as `no_std`.** No `std::`, no `println!`, no heap unless
   you bring `alloc` in yourself. `core::` is available.

2. **Write the smallest thing that makes the stated check pass.** Thirty to
   sixty lines is the usual size of a first file and a good target. There is
   a hard budget too and a file over it is refused outright, but the budget
   is a ceiling and not an aim: a long file is a later milestone smuggled
   into an early one, and it is judged as noise even when it compiles.

3. **It must make exactly the stated check pass, and nothing more.** You are
   given what must become true and the check that will say whether it did.
   Anything beyond that is a later milestone and will be judged as noise.

4. **Public items need doc comments saying why, not what.** The tree's
   convention is that a comment explains the decision an obvious alternative
   would have got wrong. `/// The frame is owned rather than borrowed,
   because the decoder outlives the buffer it was handed.` -- not
   `/// A frame.`

5. No `unsafe` unless the thing being done cannot be expressed without it,
   and then say which invariant makes it sound.

Write the file. Nothing before the fence, nothing after it.
