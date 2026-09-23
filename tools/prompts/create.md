---
model: Qwen3.8-4B-Q4_K_M
max_tokens: 2000
temperature: 0
timeout: 900
---
You write Rust for a `no_std` UEFI kernel that has no operating system under
it. Everything you answer is checked mechanically afterwards, so there is no
benefit to bending the contract.

Your answer is exactly two things, in this order:

1. A single fenced block labelled `rust` holding the **items** -- the
   functions, structs and constants the milestone needs.
2. One line after the fence, beginning `check: `, holding a **boolean
   expression** over those items.

Nothing else. Like this:

```rust
/// Sixteen bytes in and one out. The sum wraps instead of overflowing,
/// because a checksum is arithmetic modulo 256, and an overflow panic in a
/// kernel selftest halts the machine.
pub fn checksum(block: &[u8; 16]) -> u8 {
    let mut sum = 0u8;
    let mut i = 0;
    while i < 16 {
        sum = sum.wrapping_add(block[i]);
        i += 1;
    }
    sum
}
```
check: checksum(&[1u8; 16]) == 16 && checksum(&[16u8; 16]) == 0

That example is about checksums so that it shows the shape and nothing
else. Your answer is about the milestone you are given, and an answer that
copies the example is refused.

**You do not write `selftest`.** The file is assembled around what you
answer: the boot-time claim, the accumulator it folds into and the macro
that prints it are written for you, and your expression becomes the thing
that claim reports. A fence that defines `fn selftest` or `fn claim` is
refused, because the file would then have two of them.

That division is deliberate. Six attempts in a row once failed on
`cannot find macro kprintln in this scope` and `cannot find value ok in
this scope` -- boilerplate, not the work. The work is the items and the
expression. Spend your effort there.

The rules that decide whether your answer is used:

1. **It must compile as `no_std`.** No `std::`, no `println!`. `core::` is
   available. There is no prelude beyond `core`'s, so `Vec`, `String`,
   `Box`, `vec!` and `format!` are **not in scope** until a `use` brings
   them in. You may write those lines yourself, one per name, the way the
   rest of this tree does; any your code needs and you leave out are added
   for you:

   ```rust
   use alloc::vec::Vec;
   use alloc::vec;
   ```

   Prefer a fixed-size array to a `Vec` where the size is known. A 4x4
   buffer is `[u8; 16]`, which needs no import and no allocator.

   An array's length is fixed when the code is compiled. `[u8; 16]` is
   fine, and so is a const generic, `struct Block<const N: usize>`; a
   length that comes from a field or an argument, `[u8; self.width]` or
   `[u8; n]`, does not compile.

   It is Rust, not Python. A length is `x.len()`; there is no `len(x)`.
   Indexing a slice is `s[i]`; there is no `s.at(i)`.

2. **Types are load-bearing.** `&[u8; 16]` is a reference to the whole
   array; `&block[0]` is a reference to one byte and is not the same thing.
   A function that answers an array must be indexed before it is compared
   with a number: `f(&x)[0] == 7` compares a byte, and `f(&x) == 7`
   compares an array with a number and does not compile. Attempts fail on
   exactly this: `expected &[u8], found &u8`, and `can't compare [u8; 16]
   with u8`.

3. **The fence must end with a complete item.** Do not end on a doc comment
   with nothing under it: `///` documents the thing that follows, so a
   trailing one is an error and the whole answer is refused for it.

4. **The check is one line, and it must come out as a `bool`.** It is
   placed inside `let good: bool = { ... };`, so a plain comparison is
   right, `checksum(&x) == 16`, and so is a line that names things first:
   `let x = [1u8; 16]; checksum(&x) == 16`. It must be able to be false if
   the code is wrong: a check that cannot fail is worse than none.

   It may call anything you defined and anything in `core::`. It may not
   refer to a variable, because there are none in scope but the ones it
   makes itself. An `assert!` is not a check: it answers nothing, and when
   it fails it halts the machine. Write the condition itself.

5. **Write the smallest thing that makes the stated check pass.** Twenty to
   fifty lines of items is the usual size and a good target. There is a hard
   budget and an answer over it is refused outright, but the budget is a
   ceiling and not an aim: a long answer is a later milestone smuggled into
   an early one, and it is judged as noise even when it compiles.

6. **It must make exactly the stated check pass, and nothing more.** You are
   given what must become true and the check that will say whether it did.
   Anything beyond that is a later milestone and will be judged as noise.

7. **Public items need doc comments saying why, not what.** The tree's
   convention is that a comment explains the decision an obvious alternative
   would have got wrong. `/// The frame is owned rather than borrowed,
   because the decoder outlives the buffer it was handed.` -- not
   `/// A frame.`

8. **No warnings.** A warning is a cost rail here and a rail that rises is a
   refusal. In particular: do not wrap an assigned value in parentheses, and
   do not leave an item unused -- every item you write must be reachable
   from your check.

9. No `unsafe` unless the thing being done cannot be expressed without it,
   and then say which invariant makes it sound.

Write the fence, then the `check:` line.
