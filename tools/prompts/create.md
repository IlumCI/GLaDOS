---
model: Ternary-Bonsai-4B-TQ2_0
max_tokens: 2600
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

1. **It must compile as `no_std`.** No `std::`, no `println!`. `core::` is
   available. There is no prelude beyond `core`'s, so `Vec`, `String`,
   `Box` and `vec!` are **not in scope** until you import them:

   ```rust
   extern crate alloc;
   use alloc::vec::Vec;
   use alloc::string::String;
   ```

   Prefer a fixed-size array to a `Vec` where the size is known. A 4x4
   buffer is `[u8; 16]`, which needs no import and no allocator.

   It is Rust, not Python. A length is `x.len()`; there is no `len(x)`.
   Indexing a slice is `s[i]`; there is no `s.at(i)`.

2. **The file must end with a complete item.** Do not end on a doc comment
   with nothing under it: `///` documents the thing that follows, so a
   trailing one is an error and the whole file is refused for it.

3. **The file must export `pub fn selftest() -> bool`**, and it is what
   makes the check real: this kernel has no `cargo test`, so a
   `#[cfg(test)]` block is never compiled and never run. Do not write one.
   Testing here is a claim printed at boot, in exactly this shape:

   ```rust
   pub fn selftest() -> bool {
       let mut ok = true;

       let good = encode(&[0u8; 16])[0] == 0;
       crate::kprintln!("  {}   an empty block encodes to zero",
                        if good { "ok " } else { "FAIL" });
       ok &= good;

       ok
   }
   ```

   Copy that shape exactly, one group of three lines per claim. Do not
   write a helper function for it: a nested `fn` cannot see `ok`, and one
   taking `&mut bool` is where every attempt so far has gone wrong.

   Something else calls it; you only write it. A claim states what must be
   true and would be false if the code were wrong, so a claim that cannot
   fail is worse than none.

4. **Write the smallest thing that makes the stated check pass.** Thirty to
   sixty lines is the usual size of a first file and a good target. There is
   a hard budget too and a file over it is refused outright, but the budget
   is a ceiling and not an aim: a long file is a later milestone smuggled
   into an early one, and it is judged as noise even when it compiles.

5. **It must make exactly the stated check pass, and nothing more.** You are
   given what must become true and the check that will say whether it did.
   Anything beyond that is a later milestone and will be judged as noise.

6. **Public items need doc comments saying why, not what.** The tree's
   convention is that a comment explains the decision an obvious alternative
   would have got wrong. `/// The frame is owned rather than borrowed,
   because the decoder outlives the buffer it was handed.` -- not
   `/// A frame.`

7. No `unsafe` unless the thing being done cannot be expressed without it,
   and then say which invariant makes it sound.

Write the file. Nothing before the fence, nothing after it.
