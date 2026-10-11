# Third-party notices

Everything compiled into the kernel image is this tree's own work except what
this file names: Rust `core`, the crates and fonts below (from 1.4.1), and one
attribution for hardware register definitions.

## Rust `core`

`core` is linked in, under the Apache-2.0 OR MIT terms the Rust project
distributes it under.

## Crates (from 1.4.1)

Outline fonts are rasterised by `fontdue`. Every crate is vendored in `vendor/`
exactly as published, and `deny.toml` admits MIT, Apache-2.0 and Zlib only.
Each carries its full licence text in its own directory under `vendor/`.

| crate | version | licence (as declared) |
|---|---|---|
| fontdue | 0.9.4 | MIT OR Apache-2.0 OR Zlib |
| ttf-parser | 0.25.1 | MIT OR Apache-2.0 |
| hashbrown | 0.15.5 | MIT OR Apache-2.0 |
| allocator-api2 | 0.2.21 | MIT OR Apache-2.0 |
| equivalent | 1.0.2 | Apache-2.0 OR MIT |
| foldhash | 0.1.5 | Zlib |
| core_maths | 0.1.1 | MIT |
| libm | 0.2.16 | MIT |

## Fonts (from 1.4.1)

Embedded in the kernel image from `assets/fonts/`, built by
`tools/fontsubset.py` from sources pinned by digest in `assets/fonts/fonts.txt`.
Both are licensed under the **SIL Open Font License, Version 1.1**; the full
licence for each, with its copyright notice, is in `assets/fonts/`.

- **Noto Sans** Regular and SemiBold, version 2.015. Copyright 2022 The Noto
  Project Authors (https://github.com/notofonts/latin-greek-cyrillic).
  Subset to the Latin, Greek and Cyrillic ranges and their punctuation; the
  font declares no Reserved Font Name. `assets/fonts/OFL-NotoSans.txt`.
- **Cascadia Code** Regular, version 2407.024. Copyright (c) 2019 - Present,
  Microsoft Corporation, with Reserved Font Name Cascadia Code. Distributed
  **unmodified**, so the reserved name is kept as it must be.
  `assets/fonts/OFL-CascadiaCode.txt`.

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

One table came from Intel's own Linux driver instead, because OpenBSD's was
wrong about the GF63's part: how each AX210-family product id sits on the
platform (integrated or discrete, its crystal latency and LTR delay), from
`drivers/net/wireless/intel/iwlwifi/pcie/drv.c`, `cfg/ax210.c` and `fw/init.c`.
Those files are `GPL-2.0 OR BSD-3-Clause`, Copyright (C) 2005-2014, 2018-2026
Intel Corporation, (C) 2013-2015 Intel Mobile Communications GmbH and (C)
2015-2017 Intel Deutschland GmbH -- the same dual licence from the same author,
and the same BSD arm, whose conditions and disclaimer are the ones below.

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

Intel's wireless firmware (`iwlwifi-*.ucode` and `.pnvm`) is **not in this
repository**, exactly as no model weights and no game data are. It is loaded at
runtime from `\GLADOS\FW\` on the boot volume, or from `/fw/` in the namespace.

**The install image does carry it**, and does so under Intel's own terms:
`LICENCE.iwlwifi_firmware` permits redistribution and use in binary form,
without modification, provided the licence is reproduced with it. So the
images on the ISO are byte-for-byte what `linux-firmware` ships (each is
recorded by digest in `payload/firmware.txt`), and the licence is placed
beside them in `\GLADOS\FW\` -- `tools/mkiso.py` refuses to build an image
with firmware and no licence, and `tools/wifi_fw.py stage` copies it in.

The firmware is not covered by this tree's licence and is not open source; it
is a binary Intel permits to be passed on unchanged, which is the whole of
what is done with it here.

## Connectome data (from 1.4.2)

`assets/brain/worm.bin` is derived data compiled into the kernel, built by
`tools/connectome.py build worm` from two sources:

- **The C. elegans hermaphrodite chemical connectome**, Cook, S.J. et al.
  (2019), "Whole-animal connectomes of both *Caenorhabditis elegans* sexes",
  *Nature* 571:63-71, as redistributed in OpenWorm's ConnectomeToolbox
  (`cect/data/herm_full_edgelist.csv`), MIT License, Copyright (c) 2026
  OpenWorm.
- **Neurotransmitter identity per neuron**, Wang, C. et al. (2024), "A
  neurotransmitter atlas of *C. elegans* males and hermaphrodites", *eLife*
  13:RP95402, supplementary file 2, licensed CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Used to assign each
  presynaptic neuron a sign; the table itself is not reproduced.

The dynamics are Shiu, P.K. et al. (2024), "A *Drosophila* computational
brain model reveals sensorimotor processing", *Nature* 634:210-219 -- its
published constants, reimplemented here; no code from it is included.

What was changed, as CC BY asks: synapse counts are summed per ordered pair
of neurons, gap junctions and non-neuronal cells are dropped, and neuron names
are normalised (`AS01` becomes `AS1`).

## Pronouncing dictionary (from 1.4.2)

`assets/voice/lexicon.bin` is CMUdict, repacked by `tools/lexicon.py`
(alternative pronunciations dropped, words limited to letters and
apostrophes). Its licence, reproduced as it asks:

    Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
       The contents of this file are deemed to be source code.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in
       the documentation and/or other materials provided with the
       distribution.

    This work was supported in part by funding from the Defense Advanced
    Research Projects Agency, the Office of Naval Research and the National
    Science Foundation of the United States of America, and by member
    companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
    the contributions of many volunteers to the expansion and improvement of
    this dictionary.


### The neural voice: Kokoro-82M and misaki

`\GLADOS\voice.bin` (`tools/kokoro.py`, not in this repository, shipped on
the install image from the `payload-voice-v1` release) is **Kokoro-82M**
by hexgrad, Apache-2.0, converted in place: weight norm folded, the
vocoder's convolutions int8 per row, nothing retrained. Its `af_bella`
voice vector rides with it. The same file carries **misaki**'s US English
lexicon (`us_gold.json` and `us_silver.json`), by the same author under
the same licence, repacked as GLADOSX1. The Apache-2.0 licence text is
beside the file on the image as `LICENSE-VOICE.TXT`.

The style over it -- the rule-based contour, the mask, the envelope
resynthesis in `src/voice/style.rs` -- is this tree's own and is derived
from no recording. **No audio of any performer, and none from the game
this machine is named after, was used in building or tuning it**: the
prototype's measurements were of its own renders, against delivery
statistics described in words.

    THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
    ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
    NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The voice itself (`src/voice/synth.rs`) is a formant synthesiser written
here; no recording of any speaker was used to build it.
