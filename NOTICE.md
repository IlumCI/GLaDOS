# Third-party notices

Everything compiled into the kernel image is this tree's own work, with one
exception and one attribution.

## The exception: Rust `core`

`core` is linked in, under the Apache-2.0 OR MIT terms the Rust project
distributes it under.

## The attribution: Intel wireless register definitions

`src/dev/iwx/` drives Intel's AX210-family wireless controller. Its register
offsets, structure layouts and field encodings were read from OpenBSD's `iwx(4)`
driver (`sys/dev/pci/if_iwxreg.h`, `if_iwxvar.h`, `if_iwx.c`), which took them in
turn from Intel's own headers.

**No source code was copied.** What was used is the set of numbers the silicon
dictates -- an offset into a hardware structure is a fact about the hardware
rather than anybody's expression -- and every identifier, comment and line of the
implementation in this tree was written for it. Three of those numbers turned out
to be wrong or misleading in the reference, and each is documented at the point it
is corrected.

Intel's headers are offered under a dual BSD/GPLv2 licence. **This tree uses the
BSD arm**, and that notice, its conditions and its disclaimer are reproduced in
full below to satisfy it, whether or not taking a register offset triggers it at
all. Reproducing it costs a file; arguing about it does not.

Nothing in this tree is under the GPL. It was once -- `src/dev/rtl8188eu_tables.rs`
carried GPL-2.0 tables from Linux's rtl8xxxu driver -- and that driver was removed
in full when the hardware it served stopped working.

```
 * This file is provided under a dual BSD/GPLv2 license.  When using or
 * redistributing this file, you may do so under either license.
 *
 * BSD LICENSE
 *
 * Copyright(c) 2017 Intel Deutschland GmbH
 * Copyright(c) 2018 - 2019 Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *  * Neither the name Intel Corporation nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Firmware

Intel's wireless firmware (`iwlwifi-so-a0-hr-b0-*.ucode`) is **not in this
repository and is not redistributed by it**. It is loaded at runtime from wherever
the operator has put it, under Intel's own redistribution terms, exactly as no
model weights and no game data are in this repository either.
