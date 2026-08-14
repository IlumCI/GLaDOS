#!/usr/bin/env python3
"""Generate the GLaDOS documentation site: static HTML, one file per topic.

Why generated rather than hand-written
--------------------------------------
The previous site was one page that routed on `location.hash`. That is fine for
a human clicking through a file listing and useless for everything else: a
crawler sees exactly one URL, so `#/iso` and `#/wiki/rope` are not pages, they
are scroll positions. Every page here is a real file at a real path with its
own title, description, canonical link and structured data, because that is the
only shape a search engine can index.

Generating them from one data structure is what keeps twenty pages consistent.
The alternative -- twenty hand-maintained files -- diverges the first time a
nav item changes, and the divergence is invisible until someone lands on the
one page with the stale header.

The output has no build step and no dependencies: HTML, one stylesheet, one
small script. It works from `file://`, from GitHub Pages, and from any static
host, which is the same reason the ISO builder writes its own filesystems.

Usage:
    mksite.py [--out docs]
"""

import argparse
import html
import re
from pathlib import Path

BASE = "https://ilumci.github.io/GLaDOS"
REPO = "https://github.com/IlumCI/GLaDOS"
REL = REPO + "/releases/latest/download/"
UPDATED = "2026-08-14"

# Trademark line. Present on every page, not buried on one, because the whole
# site leans on a fictional company's name and the honest version of that is
# saying so everywhere rather than once.
DISCLAIMER = (
    "GLaDOS, Aperture Science and Portal are properties of Valve Corporation. "
    "This is an independent, non-commercial homage and is not affiliated with, "
    "endorsed by, or connected to Valve in any way."
)


# --- content model --------------------------------------------------------
#
# A page is metadata plus a list of blocks. Blocks are tuples so the content
# reads as content rather than as markup, and so a rendering change happens in
# one function instead of in twenty files.

def h2(t): return ("h2", t)
def h3(t): return ("h3", t)
def p(t): return ("p", t)
def ul(*items): return ("ul", list(items))
def ol(*items): return ("ol", list(items))
def code(t, lang=""): return ("code", (t, lang))
def note(t): return ("note", t)
def table(head, rows): return ("table", (head, rows))
def infobox(title, pairs): return ("infobox", (title, pairs))
def faq(pairs): return ("faq", pairs)


PAGES = {}


def page(slug, title, desc, keywords, blocks, kind="article", nav=None, faqs=None):
    PAGES[slug] = dict(slug=slug, title=title, desc=desc, keywords=keywords,
                       blocks=blocks, kind=kind, nav=nav, faqs=faqs)


# --------------------------------------------------------------------------
# Landing page
# --------------------------------------------------------------------------

page(
    "index",
    "GLaDOS: an operating system in Rust with a language model in the kernel",
    "GLaDOS is a from-scratch ring-0 operating system written in Rust with a "
    "language model running inside the kernel. Free bootable ISO, full source, "
    "and documentation on how every part works.",
    ["GLaDOS", "GLaDOS OS", "AI operating system", "Rust OS", "Aperture Science OS",
     "Portal OS", "bare metal Rust", "ring 0 kernel", "LLM in kernel"],
    kind="home",
    blocks=[
        p("<strong>GLaDOS</strong> is an operating system written from scratch in "
          "Rust, for x86-64, running in ring 0 on bare metal. With a language "
          "model living <em>inside</em> the kernel rather than on top of it."),
        p("No user/kernel split. No syscalls. No process isolation. One address "
          "space. A tool call from the model is a function call: not an IPC round "
          "trip, not a sandbox boundary, just a call."),
        p("It is named after the AI from <em>Portal</em>, and it borrows Aperture "
          "Science's look, amber on black, Windows 3.1 chrome, an enrichment "
          "centre tone. The engineering underneath is entirely real: a TCP/IP "
          "stack, TLS 1.3, NVMe, USB 3, and a transformer, all written by hand."),
        ("cards", [
            ("Download the ISO", "download/",
             "Bootable UEFI images with the model baked in. 33 MB to 575 MB."),
            ("Read the wiki", "wiki/",
             "How every subsystem works, and what it cost to find out."),
            ("See it running", "screenshots/",
             "The desktop, the shell, and the model answering from ring 0."),
            ("Browse the archive", "archive/",
             "Mirror-style file index of images, sources and checksums."),
            ("Source on GitHub", REPO,
             "119 files of Rust. All rights reserved, published to be read."),
        ]),
        ("shots", [
            ("desktop.png", "Desktop with terminal and Program Manager",
             "GLaDOS OS desktop showing a Windows 3.1 styled terminal window with "
             "the boot log, a Program Manager window, and a taskbar",
             "The desktop as it comes up: the terminal showing its own boot log, "
             "Program Manager, and a taskbar. Every pixel is drawn by the kernel "
             "into a UEFI framebuffer. There is no graphics library under it.",
             True),
            ("desktop-clean.png", "Desktop wallpaper",
             "The GLaDOS desktop wallpaper showing the aperture mark drawn as "
             "vector geometry",
             "With the terminal minimised. The wallpaper mark is drawn as geometry "
             "at boot rather than decoded from an image. There is no image "
             "decoder in the kernel.", False),
            ("model.png", "The model answering",
             "GLaDOS running a language model in kernel space, answering a question "
             "about operating systems",
             "The resident model answering <em>what is an operating system</em> "
             "from inside the kernel. No process, no API, no server.", False),
        ]),
        h2("What actually works"),
        p("This is a research kernel, and the honest version of a feature list "
          "includes the parts that do not work. Both lists are below."),
        table(
            ["Subsystem", "State"],
            [["Boot", "UEFI application, own page tables, GDT/IDT, APIC timer, PS/2 keyboard"],
             ["Memory", "Frame allocator, identity paging to 4 GiB, coalescing heap"],
             ["Tasks", "Preemptive at 100 Hz, hand-written context switch"],
             ["Graphics", "GOP framebuffer, Windows 3.1 desktop, window manager, taskbar"],
             ["Storage", "NVMe driver, content-addressed store, Merkle trees, snapshots"],
             ["Network", "ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS, TLS 1.3 with chain validation"],
             ["Crypto", "SHA-1/256/384, HMAC, AES, ChaCha20-Poly1305, X25519, RSA, ECDSA"],
             ["Model", "Qwen3-0.6B or SmolLM2-135M, int8, inference in kernel"],
             ["Language", "Lexer, parser, interpreter with kernel builtins"]]),
        h3("And what does not"),
        ul("<strong>Wireless.</strong> The laptop's card is CNVi, so the MAC is in "
           "the chipset and the module is only a radio behind an undocumented "
           "signed-firmware protocol. The WPA2 supplicant is finished and passes "
           "IEEE test vectors at every boot, and has never had hardware to run on. "
           "See <a href=\"wiki/usb-wifi-driver.html\">the USB WiFi work</a>.",
           "<strong>SMP.</strong> One core. The interior-mutability type is named "
           "<code>Racy</code> precisely so it is greppable the day that changes.",
           "<strong>An autonomous agent loop.</strong> Deliberately absent. The "
           "model proposes; a keystroke adopts."),
        h2("Why it exists"),
        p("Most \"AI operating systems\" are a chat window on top of Linux. The "
          "question here is narrower and more interesting: if a model is a kernel "
          "primitive rather than a userspace process, what changes?"),
        p("Some answers so far. Tool calls stop being serialised. The model names "
          "a function and the kernel calls it. Invalid tool names become "
          "<a href=\"wiki/constrained-decoding.html\">unreachable rather than "
          "improbable</a>, because the grammar is built from the live table. And "
          "the best router turned out not to be the transformer at all, but "
          "<a href=\"wiki/routing.html\">a ridge regression from 1960</a> reading "
          "one hidden state, 12,672 parameters, no forward pass, better accuracy."),
        note("Lineage, not target: <a href=\"wiki/templeos.html\">TempleOS</a> is "
             "the obvious ancestor. One person, ring 0, no isolation, a "
             "deliberate aesthetic. GLaDOS is not trying to be it."),
    ],
)

# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

page(
    "download/index",
    "Download GLaDOS ISO. Bootable AI operating system images",
    "Download the GLaDOS OS ISO. Bootable UEFI images from 33 MB to 575 MB, with "
    "an in-kernel language model included. Flash with Rufus, balenaEtcher or dd.",
    ["GLaDOS ISO", "GLaDOS download", "AI OS ISO", "Rust OS ISO", "bootable ISO",
     "Aperture Science OS download"],
    kind="download",
    blocks=[
        p("Three images. All are bootable UEFI ISOs built by "
          "<a href=\"../wiki/iso-el-torito.html\">a from-scratch ISO writer</a>. "
          "a FAT32 EFI System Partition wrapped in ISO 9660 with an El Torito EFI "
          "boot entry."),
        ("downloads", [
            ("glados-qwen3-0.6b.iso", "575 MB", REL + "glados-qwen3-0.6b.iso",
             "The full system. Kernel plus Qwen3-0.6B quantised to int8, 512-token "
             "context. What you want unless you have a reason otherwise."),
            ("glados-smollm2-135m.iso", "257 MB", REL + "glados-smollm2-135m.iso",
             "Kernel plus SmolLM2-135M. Smaller and faster, and the one that fits "
             "inside QEMU's disk limits, so it is the image to test with."),
            ("glados-nomodel.iso", "33 MB", REL + "glados-nomodel.iso",
             "Kernel only. Boots to a shell and reports no model. Supply your own "
             "on the EFI System Partition."),
        ]),
        p("The same files are browsable in "
          "<a href=\"../archive/#/iso\">the archive index</a>, if you prefer a "
          "directory listing."),
        p("Checksums: <a href=\"" + REL + "SHA256SUMS\">SHA256SUMS</a>. Verify "
          "before flashing, a truncated download produces a disc that boots "
          "partway and fails somewhere confusing."),
        h2("Flashing it"),
        p("Write the image to a USB stick. Any of these work:"),
        code("# Linux / macOS. Check the device name first, this overwrites it\n"
             "sudo dd if=glados-qwen3-0.6b.iso of=/dev/sdX bs=4M status=progress oflag=sync",
             "bash"),
        ul("<strong>Windows:</strong> Rufus, in DD mode, or balenaEtcher.",
           "<strong>Anywhere:</strong> balenaEtcher handles the ISO directly."),
        note("<strong>Secure Boot must be off.</strong> The kernel is unsigned. "
             "there is no Microsoft-signed shim in front of it. So firmware with "
             "Secure Boot enabled refuses to load it and usually says nothing more "
             "useful than \"no bootable device\"."),
        h2("Booting it"),
        ol("Write the image and leave the stick in.",
           "Reboot and open the firmware boot menu. Commonly F11, F12, F8 or Esc.",
           "Pick the USB device listed under UEFI.",
           "Watch the boot log. It prints its memory map, every driver it finds, "
           "and a pass or fail line for each self-test."),
        h2("What you get"),
        p("A Windows 3.1-styled desktop with a window manager, taskbar and a "
          "terminal, and a shell where <code>gen</code> generates text, "
          "<code>ask</code> answers a question, <code>if</code> shows interfaces, "
          "<code>dhcp</code> gets a lease and <code>https example.com /</code> "
          "fetches a page over a TLS 1.3 connection this kernel negotiated itself."),
        h2("Hardware support, honestly"),
        p("Developed against one laptop: an MSI Thin GF63 12UC. That is the only "
          "machine it is meaningfully tested on."),
        p("It should boot on most x86-64 UEFI systems. The graphics path is plain "
          "GOP and nothing in the boot path is vendor-specific. Storage and "
          "networking are another matter: the drivers target particular chips "
          "(Intel e1000, Realtek RTL8168, NVMe, xHCI). Expect a working shell and "
          "a working model; do not expect your disk or your network card."),
        faq([
            ("Is the GLaDOS ISO free?",
             "Yes, and free to download and run. The source is published under "
             "all-rights-reserved, so you may read it but not redistribute or "
             "build derivatives from it without permission."),
            ("Will it damage my computer or my files?",
             "It runs entirely from the USB stick and does not install anything. "
             "It will not write to your disk unless it finds unallocated space "
             "that has been explicitly designated. On a machine whose disk is "
             "fully allocated to Windows, storage initialisation simply fails, "
             "which is the intended outcome."),
            ("Can I run it in a virtual machine?",
             "Yes, with UEFI firmware (OVMF). Use the SmolLM2 image: QEMU's "
             "built-in FAT support caps the whole disk at 516 MB, which the larger "
             "image exceeds."),
            ("Why does it need Secure Boot disabled?",
             "The kernel is not signed by a key your firmware trusts. Signing "
             "would mean a Microsoft-signed shim, which is a distribution problem "
             "rather than a technical one."),
        ]),
    ],
)

# --------------------------------------------------------------------------
# Wiki index
# --------------------------------------------------------------------------

WIKI = [
    ("glados-os", "GLaDOS OS", "What the system is, and how it is put together"),
    ("aperture-science", "Aperture Science", "The homage, the aesthetic, and where it came from"),
    ("portal-os", "Portal OS", "Portal-inspired computing and what a real one looks like"),
    ("rust-os", "Writing an OS in Rust", "no_std, ownership at ring 0, and what Rust buys you"),
    ("ring-0", "Ring 0", "Privilege levels, and what removing the boundary means"),
    ("uefi-kernel", "UEFI as the kernel", "No bootloader: the UEFI application is the OS"),
    ("llm-in-kernel", "A language model in the kernel", "Inference with no allocator, no OS, no libm"),
    ("qwen3", "Qwen3 in 570 MB", "Head width, QK-norm, and two bugs that never threw an error"),
    ("rope", "RoPE", "Rotary embeddings, and the convention that cost months"),
    ("kv-cache", "The int8 KV cache", "Quantising attention memory, and why keys hurt more"),
    ("tokenizer", "Tokenizers", "BPE, pre-tokenizer regexes, and 12% silent divergence"),
    ("constrained-decoding", "Constrained decoding", "Making invalid output unreachable"),
    ("routing", "Tool routing", "Why a 1960 regression beats the transformer"),
    ("gui", "The Windows 3.1 desktop", "Bevels, a window manager, and painting from scratch"),
    ("network-stack", "The network stack", "ARP to TLS 1.3, written by hand"),
    ("tls", "TLS 1.3 from scratch", "Certificate chains, and why this is the dangerous part"),
    ("storage", "Content-addressed storage", "NVMe, Merkle trees, and O(1) snapshots"),
    ("usb-xhci", "The USB stack", "xHCI rings, cycle bits, and enumeration"),
    ("usb-wifi-driver", "USB WiFi", "CNVi, a dongle, and 509 constants"),
    ("iso-el-torito", "How the ISO is built", "FAT32 and El Torito, written from scratch"),
    ("templeos", "TempleOS", "The lineage, and what it got right"),
    ("testing", "Testing without a test runner", "Boot self-tests and driving QEMU"),
]

page(
    "wiki/index",
    "GLaDOS Wiki. How an AI operating system is built, subsystem by subsystem",
    "Documentation for GLaDOS OS: the kernel, the in-kernel language model, the "
    "network stack, the USB drivers, and the bugs that cost the most time.",
    ["GLaDOS wiki", "OS development", "kernel documentation", "Rust OS tutorial",
     "operating system internals"],
    kind="index",
    blocks=[
        p("Every subsystem, how it works, and. Where it is interesting. What it "
          "cost to get right. This project's habit is to record the measurement "
          "that overturned an assumption rather than only the conclusion, so a "
          "fair number of these pages are about being wrong."),
        ("wikilist", WIKI),
    ],
)

# --------------------------------------------------------------------------
# Wiki pages
# --------------------------------------------------------------------------

page(
    "wiki/glados-os",
    "GLaDOS OS, a ring-0 Rust operating system with an in-kernel LLM",
    "GLaDOS OS explained: a from-scratch Rust kernel for x86-64 with a "
    "transformer running in ring 0, no syscalls, and one address space.",
    ["GLaDOS OS", "AI operating system", "Rust kernel", "in-kernel LLM",
     "operating system with AI"],
    blocks=[
        infobox("GLaDOS", [
            ("Type", "Research operating system"),
            ("Language", "Rust (<code>no_std</code>)"),
            ("Architecture", "x86-64, UEFI"),
            ("Privilege", "Ring 0 only"),
            ("Model", "Qwen3-0.6B, int8"),
            ("Lines of Rust", "~119 files"),
            ("Licence", "All rights reserved"),
        ]),
        p("GLaDOS is a single-address-space operating system whose distinguishing "
          "feature is that a transformer runs inside the kernel. There is no "
          "userspace to put it in. There is no userspace at all."),
        h2("The structural choice"),
        p("Conventional systems separate kernel and user code with a privilege "
          "boundary, crossed by syscalls. That boundary buys isolation and costs a "
          "context switch. GLaDOS removes it entirely: everything runs at "
          "<a href=\"ring-0.html\">ring 0</a>, in one address space, with no "
          "syscalls and no memory protection between components."),
        p("The consequence that matters is what a tool call becomes. In a normal "
          "AI agent, the model emits text, something parses it, a dispatcher looks "
          "up a handler, and the call crosses a process boundary. Here the model "
          "names a function under "
          "<a href=\"constrained-decoding.html\">a grammar built from the live "
          "function table</a> and the kernel calls it. There is nothing in between."),
        p("What this costs is equally plain: a bug anywhere can corrupt anything. "
          "There is no adversary model, and the system does not pretend to have "
          "one. It stops incompetence, not intent."),
        h2("Boot"),
        p("There is no bootloader. UEFI already delivers long mode, ring 0 and an "
          "identity map, so the UEFI application <em>is</em> the kernel. No ELF "
          "loading, no relocation, no handoff ABI. See "
          "<a href=\"uefi-kernel.html\">UEFI as the kernel</a>."),
        p("The model, tokenizer and root certificates are read before "
          "<code>ExitBootServices</code>, because that is the only moment a "
          "filesystem exists. After that the system is on its own page tables with "
          "no firmware services at all."),
        h2("Subsystems"),
        ul("<a href=\"llm-in-kernel.html\">Inference</a>, a transformer with no "
           "allocator behind it and no libm to call.",
           "<a href=\"network-stack.html\">Networking</a>, ARP through TCP, DHCP, "
           "DNS and <a href=\"tls.html\">TLS 1.3</a>, all hand-written.",
           "<a href=\"storage.html\">Storage</a>. NVMe under a content-addressed "
           "object store where a snapshot is one hash.",
           "<a href=\"usb-xhci.html\">USB</a>. xHCI rings, enumeration, and a "
           "CDC Ethernet driver that carries real traffic.",
           "<a href=\"gui.html\">Graphics</a>, a Windows 3.1 desktop drawn pixel "
           "by pixel into a GOP framebuffer."),
        h2("The interesting negative results"),
        p("Training an adapter head on top of the model <em>hurts</em> at this data "
          "scale. A Product-of-Experts council of three classifiers does not beat "
          "the single best one. Both stay in the source tree, because the reason to "
          "know them is the reason they were worth measuring."),
        ("seealso", ["rust-os", "ring-0", "templeos", "routing"]),
    ],
)

page(
    "wiki/aperture-science",
    "Aperture Science. The aesthetic behind an amber-on-black operating system",
    "Why GLaDOS OS looks like Aperture Science: amber on black, Windows 3.1 "
    "chrome, and a boot screen built from the Portal-era design language.",
    ["Aperture Science", "Aperture Science OS", "Portal aesthetic", "GLaDOS",
     "Aperture Laboratories", "Portal design"],
    blocks=[
        p("<strong>Aperture Science</strong> is the fictional research company from "
          "Valve's <em>Portal</em>, and GLaDOS is the artificial intelligence that "
          "runs its facility. This operating system takes the name and the look "
          "and applies both to something real."),
        note(DISCLAIMER),
        h2("What the aesthetic actually is"),
        p("Reduced to specifics, the Aperture look is a small set of decisions:"),
        ul("<strong>Amber on black.</strong> The accent here is "
           "<code>#F28C1E</code>, the colour of a CRT phosphor and of every warning "
           "label in the facility.",
           "<strong>Monospace everything.</strong> Not a font choice so much as an "
           "institutional one. The typography of test protocols and lab equipment.",
           "<strong>Cheerful text about dangerous things.</strong> The enrichment "
           "centre voice: procedural, encouraging, describing hazards in the tone "
           "of a safety video.",
           "<strong>Industrial chrome.</strong> Beveled panels, hard edges, visible "
           "structure. Which lands, conveniently, very close to Windows 3.1."),
        h2("Where it shows up"),
        p("The boot screen draws the Aperture logo as vector geometry. There is no "
          "image decoder in the kernel, so it is arcs and lines computed at "
          "startup. The desktop wallpaper uses the same mark. The window chrome is "
          "<a href=\"gui.html\">genuine Windows 3.1</a>: two-pixel bevels, light on "
          "the top-left, dark on the bottom-right, over a <code>#C0C0C0</code> face."),
        p("The result is a machine that looks like it was requisitioned by a "
          "1990s research facility with a generous budget and poor oversight."),
        h2("The joke, and the part that is not a joke"),
        p("Naming an operating system after a fictional AI that murders its "
          "researchers is a joke about AI safety, and it is meant as one. The "
          "system itself is deliberately unambitious about autonomy: there is no "
          "agent loop, the model proposes and a human keystroke adopts, and "
          "self-modification is gated behind measurement on a held-out split."),
        p("The engineering is not a joke at all. The "
          "<a href=\"network-stack.html\">TCP/IP stack</a>, the "
          "<a href=\"tls.html\">TLS 1.3 implementation</a>, the "
          "<a href=\"usb-xhci.html\">xHCI driver</a> and the "
          "<a href=\"llm-in-kernel.html\">transformer</a> are all real, all "
          "hand-written, and all tested."),
        ("seealso", ["portal-os", "gui", "glados-os", "templeos"]),
    ],
)

page(
    "wiki/portal-os",
    "Portal OS. What a real Portal-inspired operating system looks like",
    "Portal OS: not a theme pack, but a genuine ring-0 kernel named for GLaDOS, "
    "with Aperture Science styling and a language model in kernel space.",
    ["Portal OS", "Portal operating system", "GLaDOS OS", "Portal theme",
     "Aperture Science OS", "Portal 2"],
    blocks=[
        p("Search for a \"Portal OS\" and you mostly find themes: wallpapers, icon "
          "packs, a terminal colour scheme, occasionally a Linux distribution with "
          "the logo swapped. This is not that."),
        p("GLaDOS is an operating system in the literal sense. It boots on bare "
          "metal, brings up its own page tables, drives its own hardware, and has "
          "no Linux, no BSD and no kernel from anyone else underneath it. The "
          "Portal reference is in the name, the "
          "<a href=\"aperture-science.html\">aesthetic</a>, and one design joke. "
          "Everything else is an operating system."),
        h2("What makes it Portal-ish"),
        ul("Named for the facility AI, and the resident model answers as it.",
           "The <a href=\"aperture-science.html\">Aperture visual language</a>: "
           "amber on black, monospace, vector-drawn logo at boot.",
           "A tone that treats catastrophic failure as a procedural note."),
        h2("What makes it an operating system"),
        table(
            ["Layer", "What is there"],
            [["Boot", "Its own UEFI application; no GRUB, no shim, no Linux"],
             ["Memory", "Physical frame allocator and hand-built page tables"],
             ["Scheduling", "Preemptive multitasking on an APIC timer"],
             ["Drivers", "NVMe, Intel e1000, Realtek RTL8168, xHCI USB 3"],
             ["Network", "Its own ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS, TLS 1.3"],
             ["Crypto", "Its own SHA-2, AES, ChaCha20, X25519, RSA, ECDSA"],
             ["Graphics", "Its own compositor and window manager"],
             ["AI", "Its own transformer inference, in kernel space"]]),
        p("The only code in it that was not written for this project is Rust's "
          "<code>core</code> library, and "
          "<a href=\"usb-wifi-driver.html\">509 hardware constants</a> transcribed "
          "from Linux because there is no other source for them."),
        note(DISCLAIMER),
        ("seealso", ["aperture-science", "glados-os", "rust-os", "download"]),
    ],
)

page(
    "wiki/rust-os",
    "Writing an operating system in Rust. no_std, ring 0, and no runtime",
    "What building a Rust OS actually involves: no_std, a hand-written allocator, "
    "the UEFI target, calling-convention traps, and where Rust genuinely helps.",
    ["Rust OS", "Rust operating system", "no_std", "bare metal Rust",
     "Rust kernel development", "osdev Rust"],
    blocks=[
        p("GLaDOS is written in Rust targeting <code>x86_64-unknown-uefi</code>, "
          "with <code>no_std</code> and no runtime. This is what that means in "
          "practice."),
        h2("What you give up"),
        p("<code>no_std</code> removes the standard library, which removes rather "
          "more than people expect: no heap until you write an allocator, no "
          "<code>File</code>, no <code>Vec</code> until <code>alloc</code> is wired "
          "up, no threads, no <code>println!</code>, and no floating-point maths "
          "functions, <code>exp</code>, <code>sqrt</code> and <code>tanh</code> "
          "live in libm, which is C, which is not there."),
        p("For an OS with a transformer in it, that last one is not a detail. "
          "Softmax needs <code>exp</code>. So the "
          "<a href=\"llm-in-kernel.html\">inference code</a> carries its own."),
        h2("What you get"),
        p("Ownership does not stop caring at the hardware boundary. A driver that "
          "hands out a buffer it still holds, a ring the allocator might free "
          "underneath the controller, a device structure aliased by two code paths "
          ". These are compile errors rather than corruption discovered three "
          "subsystems later."),
        p("The honest caveat: everything touching hardware is <code>unsafe</code> "
          "anyway. Rust does not verify that a physical address is mapped, that a "
          "DMA buffer is where the device thinks, or that a volatile write hit the "
          "register you meant. It narrows the blast radius; it does not remove it."),
        h2("Traps that cost real time"),
        ul("<strong><code>extern \"C\"</code> on the UEFI target is Microsoft x64, "
           "not System V.</strong> The context switch has to say "
           "<code>extern \"sysv64\"</code> explicitly, or arguments arrive in the "
           "wrong registers and the failure looks like memory corruption.",
           "<strong>A guarded match arm placed after the arms it guards is "
           "unreachable.</strong> The compiler warns. The warning went unread for "
           "several commits.",
           "<strong>Interior mutability is not a lock.</strong> The type used here "
           "is called <code>Racy</code> so that the day SMP arrives, one grep finds "
           "every place that assumed one core."),
        h2("The allocator"),
        p("The kernel heap is a single physically contiguous allocation chosen from "
          "the UEFI memory map, sized by a ladder that steps down until one fits. "
          "That matters because the machine's largest contiguous region is not its "
          "total RAM, a map can report 1.9 GB free across 84 regions and not have "
          "320 MB in one piece."),
        note("A subtle bug lived here: the ladder never degraded, because the "
             "probing allocator advanced its cursor destructively, so once the "
             "largest rung failed every smaller rung failed too. The fix was a "
             "non-consuming <code>largest_span()</code> that looks forward without "
             "consuming. Rewinding would have handed out the page tables' own "
             "frames a second time."),
        ("seealso", ["ring-0", "uefi-kernel", "llm-in-kernel", "glados-os"]),
    ],
)

page(
    "wiki/ring-0",
    "Ring 0. What an operating system with no privilege boundary looks like",
    "x86 protection rings explained, and what changes when a kernel deletes the "
    "user/kernel split entirely: no syscalls, one address space, no isolation.",
    ["ring 0", "kernel mode", "protection rings", "x86 privilege levels",
     "single address space OS", "no syscalls"],
    blocks=[
        p("x86 defines four privilege levels. In practice systems use two: ring 0 "
          "for the kernel, ring 3 for everything else. The boundary between them is "
          "what makes a process crash instead of a machine crash."),
        p("GLaDOS runs entirely in ring 0. There is no ring 3, no syscall "
          "instruction, and no separate address space per program."),
        h2("What the boundary costs"),
        p("Crossing it is not free. A syscall saves registers, switches stacks, "
          "changes page tables on kernel-page-table-isolation systems, validates "
          "every pointer coming across, and reverses all of it on the way back. "
          "For a call that does real work this is noise. For millions of tiny "
          "calls it is the dominant cost."),
        p("An in-kernel language model makes millions of tiny calls. A tool "
          "invocation, a namespace read, a tensor operation dispatched by the "
          "router, all of it would cross the boundary in a conventional design."),
        h2("What removing it costs"),
        p("Everything the boundary was buying:"),
        ul("A null dereference is a triple fault and a reboot, not a segfault.",
           "A buffer overrun can overwrite the page tables.",
           "There is no <code>kill</code>, a runaway loop owns the machine.",
           "A script that exhausts the heap exhausts <em>the</em> heap."),
        p("The system says this plainly rather than claiming a security model it "
          "does not have. Capabilities exist for scripts, a tame mode that refuses "
          "raw port access by name before arguments are even examined. But this is "
          "one interpreter in one address space with no isolation. It stops "
          "incompetence, not intent."),
        h2("Why that is acceptable here"),
        p("Because the threat model is honest: this is a research kernel for one "
          "laptop, running one model that is not hostile, it is small. A system "
          "that claimed isolation it could not enforce would be worse than one that "
          "says there is none."),
        ("seealso", ["glados-os", "rust-os", "templeos", "uefi-kernel"]),
    ],
)

page(
    "wiki/uefi-kernel",
    "UEFI as the kernel. Booting an OS with no bootloader at all",
    "How GLaDOS boots: the UEFI application is the kernel. No GRUB, no ELF "
    "loading, no handoff ABI, and what ExitBootServices takes away permanently.",
    ["UEFI", "UEFI bootloader", "ExitBootServices", "EFI application", "OS boot",
     "bare metal boot", "EFI System Partition"],
    blocks=[
        p("Most kernels are loaded by a bootloader. GRUB, systemd-boot, limine. "
          "which reads an ELF file, sets up long mode, builds a memory map and "
          "jumps to an entry point with an agreed calling convention."),
        p("GLaDOS skips all of it. UEFI firmware already puts you in 64-bit long "
          "mode, at ring 0, with an identity-mapped address space and a working "
          "memory allocator. A UEFI application starting in that state is already "
          "a kernel. So this one simply is."),
        h2("What that removes"),
        ul("No ELF loader, no relocation, no linker script gymnastics.",
           "No handoff ABI to agree on and get subtly wrong.",
           "No second codebase to keep in sync with the first.",
           "No multiboot header, no protected-mode stub, no A20 line."),
        h2("The one-way door"),
        p("Boot services. The firmware's filesystem access, allocator, console and "
          "device protocols. Exist only until <code>ExitBootServices</code>. After "
          "that call they are gone, permanently, and every pointer into them is "
          "dangling."),
        p("So everything the kernel needs from disk must be read <em>before</em> "
          "that moment: the 574 MB model, the tokenizer, the root certificate "
          "bundle. They are loaded into a pool that survives the transition and "
          "referenced in place afterwards, never copied to the heap."),
        note("This is also why the firmware's perfectly good USB stack is thrown "
             "away and <a href=\"usb-xhci.html\">rewritten from scratch</a>. It "
             "lives in boot services. That is the price of being the kernel rather "
             "than a guest of the firmware, and it is not refundable."),
        h2("A memory-map trap"),
        p("Do not take the maximum over every UEFI memory descriptor to decide how "
          "much to map. OVMF describes reserved space out to 1 TiB; using that as a "
          "limit exceeds what one page-directory-pointer table can cover, the "
          "identity map silently fails to install, and the firmware's own tables "
          "stay active. Which map page zero. The null-dereference self-test then "
          "passes without faulting, for the worst possible reason."),
        ("seealso", ["rust-os", "ring-0", "usb-xhci", "iso-el-torito"]),
    ],
)

page(
    "wiki/llm-in-kernel",
    "Running a language model inside a kernel. Transformer inference at ring 0",
    "How a transformer runs in kernel space with no standard library, no libm and "
    "no allocator underneath it: int8 weights, hand-written exp, SIMD matmul.",
    ["LLM in kernel", "in-kernel inference", "transformer bare metal",
     "AI operating system", "no_std machine learning", "kernel LLM"],
    blocks=[
        p("The model is a decoder-only transformer. Qwen3-0.6B by default. "
          "quantised to int8 and executed inside the kernel address space. It is "
          "not a service, not a process, and not behind an API."),
        h2("What is missing at ring 0"),
        p("Inference code normally assumes a great deal: an allocator, threads, "
          "BLAS, and libm. None of it is present."),
        ul("<strong>No libm.</strong> Softmax needs <code>exp</code>, RMSNorm needs "
           "<code>sqrt</code>, GELU needs <code>tanh</code>. All hand-written.",
           "<strong>No BLAS.</strong> The matrix-vector product is the whole cost "
           "of inference, so it is written directly, with an AVX2 path.",
           "<strong>No memory to waste.</strong> The 574 MB of weights are "
           "referenced in place in the pool they were loaded into, never copied.",
           "<strong>No threads.</strong> One core, cooperative yields so the shell "
           "stays responsive during generation."),
        h2("Where the time goes"),
        p("Generation is memory-bandwidth bound, not compute bound. Bytes read per "
          "token is roughly the model size, so a 570 MB model takes about 4.4× as "
          "long per token as a 135 MB one, almost exactly the size ratio."),
        p("155 MB of that 570 is the output classifier. Since "
          "<a href=\"constrained-decoding.html\">constrained decoding</a> only ever "
          "needs logits for the reachable token set, restricting that final "
          "matrix-vector product is the obvious win whenever the grammar is narrow."),
        h2("A feature gate that tested the wrong feature"),
        p("The AVX2 kernel was gated on <code>avx_enabled && fma</code> and never "
          "on <code>avx2</code> itself. It ran on hardware that had FMA without "
          "AVX2 and produced wrong numbers, and it did not run on hardware that had "
          "AVX2 without the exact FMA reporting expected. A feature gate must test "
          "the feature the code actually uses."),
        h2("The general lesson"),
        note("<strong>A model can be wrong without being broken.</strong> "
             "<a href=\"rope.html\">RoPE pairing</a>, QK-Norm, head width, RMSNorm "
             "epsilon and <a href=\"tokenizer.html\">the pre-tokenizer regex</a> "
             "each produce a network that loads, runs, stays numerically "
             "well-behaved and writes fluent text. There is no error to catch. The "
             "only things that settle them are a reference implementation to diff "
             "against, or output that is supposed to contain a known fact."),
        ("seealso", ["qwen3", "rope", "kv-cache", "constrained-decoding", "routing"]),
    ],
)

page(
    "wiki/qwen3",
    "Qwen3-0.6B in a kernel. Head width, QK-norm, and silent failure modes",
    "Running Qwen3-0.6B inside an OS kernel: why its head dimension is stated "
    "rather than derived, what QK-Norm does, and why both fail without any error.",
    ["Qwen3", "Qwen3-0.6B", "QK-Norm", "head_dim", "quantised LLM", "int8 inference"],
    blocks=[
        p("Qwen3-0.6B is the default model: 28 layers, 1024-dimensional residual "
          "stream, 16 query heads and 8 key/value heads, quantised to int8 for "
          "about 570 MB on disk."),
        h2("It is not a Llama, and neither difference is loud"),
        h3("Head width is stated, not derived"),
        p("Most implementations compute head dimension as "
          "<code>dim / n_heads</code>. For Qwen3 that gives 1024/16 = 64, and the "
          "correct answer is <strong>128</strong>. The model states it "
          "independently, which means the query projection is 2048 wide against a "
          "1024-wide residual stream. The attention path is deliberately wider "
          "than the thing it reads from."),
        p("Derive it instead and every weight tensor is reshaped wrong. The model "
          "still loads. It still runs. It generates confident nonsense."),
        h3("QK-Norm"),
        p("Qwen3 applies RMSNorm to each head's queries and keys, per head, before "
          "rotary embeddings. Omit it and the model remains numerically "
          "well-behaved and produces fluent text. Just text from a network being "
          "fed activations it was never trained on."),
        h2("How these get caught"),
        p("Not by exceptions, because there are none. The checkpoint converter "
          "writes both facts into the file header so they travel with the weights "
          "rather than being inferred, and a NumPy reference implementation reads "
          "the <em>converted</em> file and produces logits to diff against. A "
          "converter bug shows up there too; only a kernel bug shows as a mismatch "
          "between the two."),
        p("The cheap end of the same check is reading the output. An "
          "instruction-tuned model whose attention path is wired correctly answers "
          "\"what is the capital of France\" with Paris."),
        h2("Thinking tokens"),
        p("Qwen3 reasons at length inside <code>&lt;think&gt;</code> blocks when "
          "left alone, which is the model working as designed and useless at a "
          "64-token budget. The shell closes the block itself unless told not to. "
          "Whether the model has thinking tokens at all is decided by asking the "
          "tokenizer whether it knows <code>&lt;think&gt;</code> as a single token "
          ", a property of the vocabulary, rather than a guess from the model's "
          "name."),
        ("seealso", ["llm-in-kernel", "rope", "tokenizer", "kv-cache"]),
    ],
)

page(
    "wiki/rope",
    "RoPE. Rotary position embeddings, and the convention that fails silently",
    "Rotary Position Embedding explained, and why pairing dimension i with "
    "i+head_dim/2 instead of 2i with 2i+1 produces fluent text and wrong attention.",
    ["RoPE", "rotary position embedding", "rotate_half", "transformer positions",
     "attention", "LLM implementation"],
    blocks=[
        p("Rotary Position Embedding encodes position by rotating pairs of "
          "dimensions in the query and key vectors by an angle proportional to "
          "position. Because a dot product between two rotated vectors depends only "
          "on the difference of their angles, attention scores become naturally "
          "relative."),
        h2("Two conventions, one right answer per checkpoint"),
        p("There are two ways to choose the pairs:"),
        ul("<strong>Interleaved:</strong> pair dimension <code>2i</code> with "
           "<code>2i+1</code>. This is what the original llama2.c does.",
           "<strong>rotate_half:</strong> pair dimension <code>i</code> with "
           "<code>i + head_dim/2</code>. This is what HuggingFace transformers "
           "does, and therefore what essentially every published checkpoint "
           "expects."),
        h2("Why getting it wrong produces no error"),
        p("This is the important part. Both conventions are norm-preserving "
          "rotations by the same set of angles. Pick the wrong one and there is no "
          "NaN, no overflow, no drift, no assertion. The vectors keep their "
          "lengths. The softmax stays well-behaved. The model stays fluent."),
        p("What actually happens is that the model attends by a scrambled notion of "
          "distance. Which is indistinguishable, from the outside, from a small "
          "model being small."),
        h2("What it cost, measured"),
        p("This kernel used the interleaved convention for a long time. Switching "
          "to rotate_half on the same checkpoint, same prompt, same seed:"),
        table(
            ["Convention", "Output for \"The capital of France\""],
            [["Interleaved (wrong)", "\"The capital of France.\" then blank lines"],
             ["rotate_half (correct)", "\"The capital of France is Paris. Paris is a "
              "city known for...\""]]),
        p("Token-level, the first tokens went from <code>'\\n'</code>, "
          "<code>'\\n\\n'</code>, <code>'ity'</code>, <code>' capital'</code> to "
          "<code>'The'</code>, <code>'Paris'</code>, <code>'France'</code>."),
        note("If you are debugging an implementation that produces plausible but "
             "slightly-wrong text, check the RoPE pairing before anything else. It "
             "is the highest-probability silent bug in a hand-written transformer, "
             "and it never announces itself."),
        ("seealso", ["qwen3", "llm-in-kernel", "kv-cache", "tokenizer"]),
    ],
)

page(
    "wiki/kv-cache",
    "The int8 KV cache. Quantising attention memory, and why keys hurt more",
    "How GLaDOS fits a long context in kernel memory: int8 KV cache with per-block "
    "scales, split per layer, plus attention sinks and a sliding window.",
    ["KV cache", "int8 quantisation", "attention sinks", "StreamingLLM",
     "long context", "LLM memory"],
    blocks=[
        p("The key-value cache is where an autoregressive model keeps everything it "
          "has already read. It is also, past a few thousand tokens, far larger "
          "than the model itself."),
        h2("The arithmetic"),
        p("For Qwen3-0.6B: 2 (keys and values) × 28 layers × 1024 kv-dimensions × "
          "4 bytes = <strong>224 KiB per token</strong>."),
        table(
            ["Scheme", "Per token", "At 32,768 tokens"],
            [["f32, one allocation", "224 KiB", "7.0 GiB contiguous. Unreachable"],
             ["int8, one allocation", "56 KiB", "1.75 GiB contiguous. Very unlikely"],
             ["int8, split per layer", "56 KiB", "28 × ~66 MiB, achievable"]]),
        p("The split is what makes long context possible at all. As a single "
          "allocation the cache needs one unbroken physical region; split per layer "
          "the largest single request drops 28-fold, and a fragmented memory map "
          "can satisfy it."),
        h2("Quantisation, and an asymmetry"),
        p("Values are stored int8 in blocks of 32 with one f32 scale per block. The "
          "measured error is not symmetric between keys and values: <strong>keys "
          "carry roughly 15× the error impact</strong>."),
        p("The reason is structural. A key participates in a dot product and then "
          "goes through softmax, which is exponential, a small perturbation in a "
          "score becomes a large change in attention weight. A value is only "
          "averaged, and averaging is forgiving."),
        h2("Attention sinks and a sliding window"),
        p("Past the allocated window, the cache runs as a ring: a few initial "
          "tokens are pinned as attention sinks and the rest slides. The sinks "
          "matter more than they look. Models place a large amount of attention "
          "mass on the first few positions regardless of content, and evicting them "
          "degrades output sharply. With them, generation continues past the "
          "nominal context length instead of stopping."),
        note("Speed is the real ceiling, not memory. Attention is linear in live "
             "positions, so at 32k tokens it is roughly 3.8 GMAC per token on top "
             "of the model's own 0.6 GMAC. Several seconds per token at full "
             "context. Excellent for reading a long document once; painful for chat."),
        ("seealso", ["llm-in-kernel", "qwen3", "rope"]),
    ],
)

page(
    "wiki/tokenizer",
    "Tokenizers. BPE, pre-tokenizer regexes, and 12% silent divergence",
    "Why a byte-pair encoding tokenizer needs the exact pre-tokenizer regex its "
    "model trained with, and how the wrong one moved 12% of tokens with no error.",
    ["tokenizer", "BPE", "byte pair encoding", "pre-tokenizer", "cl100k", "GPT-2 regex"],
    blocks=[
        p("A byte-pair encoding tokenizer has two halves. A merge table, which "
          "everyone remembers, and a <strong>pre-tokenizer regex</strong> that "
          "splits text into candidate pieces before any merging happens. Which "
          "people forget, because it is usually invisible."),
        h2("The regexes differ, and it matters"),
        p("GPT-2's pattern and cl100k's pattern disagree in ways that look "
          "cosmetic:"),
        ul("Under cl100k, a word may be led by any non-alphanumeric character, so "
           "<code>(x</code> is <em>one</em> piece rather than two.",
           "Digits are emitted one at a time rather than in runs.",
           "Punctuation swallows the newlines that follow it."),
        p("SmolLM2 trained with the GPT-2 pattern. Qwen3 spells out the cl100k one. "
          "Using the wrong pattern moved <strong>about 12% of tokens</strong> on "
          "this project's training corpus. Including on the ChatML structure that "
          "the instruction tuning depends on."),
        h2("Twelve percent of nothing visible"),
        p("There is no error. The tokenizer produces token ids, the model consumes "
          "them and emits fluent text. What is actually happening is that the model "
          "is being fed sequences it never saw during training, and it degrades "
          "gently rather than failing."),
        p("So the checkpoint carries which pre-tokenizer it needs as a flag in its "
          "header, alongside "
          "<a href=\"qwen3.html\">head width and QK-norm</a>. After the fix, "
          "divergence across 407 texts and 66,598 tokens was <strong>zero</strong>."),
        h2("Verify against the reference, always"),
        p("The converter has a <code>--verify</code> mode that reimplements the "
          "kernel's algorithm in Python and diffs it against the reference "
          "<code>tokenizers</code> library, token for token. It is not optional. "
          "A tokenizer that is subtly wrong produces text that still looks like "
          "text."),
        note("A related trap: sizing the vocabulary from the merge table. Qwen3's "
             "293 special tokens live <em>above</em> it, so the vocabulary was "
             "indexed past its end, and the end-of-turn token resolved to id 2. "
             "meaning generation would never stop. Size from the highest added "
             "token id, not from the BPE table."),
        ("seealso", ["qwen3", "llm-in-kernel", "constrained-decoding"]),
    ],
)

page(
    "wiki/constrained-decoding",
    "Constrained decoding. Making invalid model output unreachable, not unlikely",
    "How GLaDOS guarantees a model can never name a tool that does not exist: a "
    "grammar built from the live function table, applied before sampling.",
    ["constrained decoding", "grammar-constrained generation", "structured output",
     "LLM tool calling", "guaranteed valid output"],
    blocks=[
        p("When a language model picks a tool to run, the usual approach is to "
          "generate text, parse it, and handle the case where it named something "
          "that does not exist. GLaDOS makes that case impossible instead."),
        h2("The mechanism"),
        p("At each decoding step the sampler is given only the tokens that could "
          "extend the current prefix into a valid name from the live function "
          "table. Every other logit is removed <em>before</em> sampling, not "
          "checked afterwards."),
        p("The consequence is categorical rather than statistical: <strong>there is "
          "no sequence of sampling outcomes that produces an invalid name.</strong> "
          "Not unlikely. Unreachable. Temperature does not matter. A badly "
          "calibrated model does not matter."),
        h2("Read-only mode is the same mechanism"),
        p("When the system runs in a read-only trust level, mutating functions are "
          "removed from the reachable set before sampling. The model is not asked "
          "to behave and then audited; the tokens that spell <code>rm</code> simply "
          "have no path."),
        p("This is why a script's mutating-ness is <em>derived</em> from its code "
          "rather than declared in its header. A script that claims to be read-only "
          "and then calls a mutating function would put a mutating entry into the "
          "read-only grammar. Which is exactly the guarantee leaking. So the "
          "system walks the syntax tree, and treats any call whose target is not a "
          "literal string as mutating, because a computed name cannot be resolved "
          "and guessing would break the property."),
        h2("How it is tested"),
        p("The grammar's self-test runs 200 random decodes at maximum temperature "
          "and asserts that none escaped the reachable set. It also checks a subtle "
          "case: that a name which is a prefix of another name is still reachable. "
          "if <code>snap</code> and <code>snaps</code> both exist, a naive "
          "implementation makes the shorter one impossible to finish."),
        ("seealso", ["routing", "llm-in-kernel", "tokenizer", "glados-os"]),
    ],
)

page(
    "wiki/routing",
    "Tool routing. Why a 1960 regression beats the transformer",
    "GLaDOS routes tool calls with closed-form ridge regression on one hidden "
    "state: 12,672 parameters, 1.6 ms, no forward pass, and better accuracy.",
    ["tool routing", "LLM routing", "ridge regression", "Widrow-Hoff", "linear probe",
     "classifier", "AI tool selection"],
    blocks=[
        p("Given a task in natural language, something has to decide which function "
          "to call. This system has two implementations, and the interesting result "
          "is that the older idea wins."),
        h2("The two paths"),
        table(
            ["Approach", "Cost", "Accuracy"],
            [["Decode the name token-by-token under a grammar",
              "A full transformer forward pass per token", "Lower"],
             ["Read one hidden state into a ridge regression",
              "12,672 parameters, ~1.6 ms, <strong>no forward pass</strong>",
              "<strong>Higher</strong>"]]),
        p("The second is a linear probe: take the hidden state the model has "
          "already computed, and multiply it by a matrix solved in closed form by "
          "Cholesky decomposition. The Widrow-Hoff least-mean-squares idea from "
          "1960, fitted inside the kernel at boot."),
        p("It wins on held-out accuracy <em>and</em> costs nothing, because the "
          "hidden state is a byproduct of work already done."),
        h2("Agreement, not votes"),
        p("Three independent classifiers run: the linear probe, a hashed n-gram "
          "Bayes model, and a lexical matcher. The useful signal turns out not to "
          "be their majority vote but <strong>whether they agree</strong>:"),
        ul("All three agree: <strong>90%</strong> correct.",
           "They split: <strong>61%</strong> correct."),
        p("That 29-point gap is actionable in a way a vote is not. It is a "
          "confidence measure that costs nothing extra, and it is what the system "
          "gates on when deciding whether to act or ask."),
        h2("Measurement discipline"),
        p("This project got its evaluation wrong three separate times before "
          "settling: a grid sweep scored on the test set, cross-validation folded "
          "by template family, and a test set that moved whenever the corpus grew."),
        p("The current arrangement has three splits. Validation is spent freely; "
          "the test slice is read once. Corpora hold out <em>whole template "
          "families</em>, never sampled instances. Instances within a family "
          "differ only by slot values, so splitting by instance measures "
          "memorisation while looking like generalisation."),
        note("Negative results stay in the tree. Training an adapter head hurts at "
             "this data scale. The Product-of-Experts council does not improve "
             "accuracy. Both are kept, because the reason to know them is the "
             "reason they were worth measuring."),
        ("seealso", ["constrained-decoding", "llm-in-kernel", "glados-os"]),
    ],
)

page(
    "wiki/gui",
    "The Windows 3.1 desktop. Writing a window manager from scratch",
    "How GLaDOS draws its GUI: raw GOP framebuffer, two-pixel bevels, z-order as "
    "focus, total repaint, and a keyboard-only window manager in kernel space.",
    ["window manager", "Windows 3.1 UI", "framebuffer graphics", "GUI from scratch",
     "retro UI", "OS graphics"],
    blocks=[
        p("There is no graphics library. The kernel gets a linear framebuffer from "
          "UEFI's Graphics Output Protocol, a base address, a width, a height and "
          "a stride, and everything above that is written by hand."),
        h2("Windows 3.1 chrome, precisely"),
        p("The look is not approximate. It is the actual construction:"),
        ul("A <code>#C0C0C0</code> face colour on every panel.",
           "Two-pixel bevels: white on the top and left, dark grey on the bottom "
           "and right, for a raised surface. Reversed for a sunken one.",
           "Title bars that fill with the selection colour when focused and grey "
           "when not.",
           "Hard rectangles. No anti-aliasing, no shadows, no rounded corners."),
        p("Over that sits the <a href=\"aperture-science.html\">Aperture palette</a> "
          ", amber accents, and a wallpaper drawn as vector geometry because there "
          "is no image decoder in the kernel."),
        h2("The window manager, and its one idea"),
        p("Z-order <em>is</em> focus. There is no separate focused-window pointer to "
          "keep in agreement with the stacking order; the front window is the "
          "focused window, and raising is focusing. One fact instead of two that "
          "can disagree."),
        p("Repainting is total, back to front, on every change. No damage tracking, "
          "no dirty rectangles. At this resolution a full repaint is a few million "
          "stores and costs less than the bookkeeping would, and damage tracking "
          "bugs are the worst kind, because they leave the screen showing something "
          "that was true a moment ago."),
        h2("A bug that only a screenshot could find"),
        p("The console wrote directly to its own rectangle with no knowledge of "
          "windows on top of it, so a file browser opened over the terminal was "
          "gradually eaten by shell output. Nothing in the serial log showed it. "
          "the log was correct, the window manager was correct, and the screen was "
          "wrong."),
        note("It was found by capturing the framebuffer through QEMU's monitor and "
             "converting it to PNG. That capture also produced a false alarm: a "
             "screenshot taken mid-repaint looked like a broken compositor, and "
             "cost a bisect before the fix turned out to be waiting two seconds for "
             "the frame to settle. A GUI nobody has looked at is a GUI nobody has "
             "tested."),
        ("seealso", ["aperture-science", "glados-os", "testing"]),
    ],
)

page(
    "wiki/network-stack",
    "Writing a TCP/IP stack from scratch, ARP to TLS in a kernel",
    "GLaDOS implements its own ARP, IPv4, ICMP, UDP, TCP, DHCP and DNS, with no "
    "interrupt-driven receive and a deliberate rule against re-entrant dispatch.",
    ["TCP/IP stack", "network stack from scratch", "ARP", "DHCP", "DNS",
     "TCP implementation", "OS networking"],
    blocks=[
        p("Every layer is hand-written: ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS and "
          "<a href=\"tls.html\">TLS 1.3</a>. Drivers exist for Intel e1000, Realtek "
          "RTL8168 and <a href=\"usb-xhci.html\">USB CDC Ethernet</a>."),
        h2("The rule that shapes everything"),
        p("<strong>The poll function never dispatches into a transport state "
          "machine. It only queues.</strong>"),
        p("The reason is a re-entrancy trap that is easy to walk into. Sending an "
          "IPv4 packet calls address resolution; resolution has to poll for the ARP "
          "reply while it waits. If polling could run TCP's state machine, a "
          "connection could re-enter its own control block while an earlier mutable "
          "borrow was still live. So poll queues frames, and TCP and UDP drain "
          "their own inboxes from a context where nothing else is borrowed."),
        h2("No interrupt-driven receive"),
        p("TCP advances while the shell is idle, or inside a blocking call. This is "
          "a deliberate simplification, and it is visible in behaviour: a "
          "connection makes no progress while a long generation is running unless "
          "the generation yields."),
        h2("Bugs worth knowing"),
        ul("<strong>Ethernet pads frames to 60 bytes.</strong> A bare 40-byte TCP "
           "ACK therefore arrives with 20 bytes of garbage after it. IPv4 payloads "
           "must be trimmed to the length the header declares, not to the length "
           "the frame happens to be.",
           "<strong>A shared event ring needs demultiplexing.</strong> On the USB "
           "driver, waiting for \"a transfer event\" rather than \"a transfer event "
           "for <em>my</em> endpoint\" meant a send completed against an arriving "
           "receive. DHCP still worked, because it alternates send and receive; ARP "
           "did not, because it had no send to piggyback on. The asymmetry is what "
           "named the bug."),
        h2("What it can actually do"),
        p("From the shell: <code>dhcp</code> obtains a lease, <code>ping</code> "
          "round-trips, <code>dns example.com</code> resolves, and "
          "<code>https example.com /</code> completes a TLS 1.3 handshake, "
          "validates the certificate chain against a bundled root store, and "
          "returns the page."),
        ("seealso", ["tls", "usb-xhci", "usb-wifi-driver", "glados-os"]),
    ],
)

page(
    "wiki/tls",
    "TLS 1.3 from scratch, and why hand-written crypto is the dangerous part",
    "GLaDOS implements TLS 1.3, X25519, ChaCha20-Poly1305 and ECDSA by hand, "
    "validates certificate chains, and is honest about what is still not safe.",
    ["TLS 1.3", "cryptography from scratch", "X25519", "ChaCha20", "ECDSA",
     "certificate validation", "kernel TLS"],
    blocks=[
        p("The kernel speaks TLS 1.3 using its own implementations of SHA-256/384, "
          "HMAC, HKDF, AES, ChaCha20-Poly1305, X25519, RSA and ECDSA. It validates "
          "certificate chains against a root bundle exported from the host's store."),
        note("<strong>This is the one place where writing everything yourself is a "
             "liability rather than a virtue.</strong> A bug in a driver produces a "
             "device that does not work. A bug here produces a connection that "
             "works perfectly and is not secure."),
        h2("Primitives chosen for checkability"),
        p("The selections are deliberate:"),
        ul("<strong>ChaCha20-Poly1305 over AES-GCM</strong>. No key-dependent "
           "table lookups, so no cache-timing side channel to reason about.",
           "<strong>X25519 over a NIST curve</strong>. No point validation "
           "required, far fewer ways to be subtly wrong."),
        p("Every primitive is verified against published RFC and FIPS test vectors "
          "at every boot, and the boot log prints a pass or fail line for each."),
        h2("An ECDSA break that sat in plain sight"),
        p("The self-test output is easy to scroll past, and doing so cost an entire "
          "debugging cycle: a broken ECDSA implementation was printing "
          "<code>FAIL</code> in the crypto block the whole time, while the log was "
          "being sliced away to look at something else."),
        h2("Jacobian coordinates, and a trap"),
        p("ECDSA uses Jacobian coordinates because affine ones cost a modular "
          "inversion per point operation, a full modular exponentiation. For "
          "P-384 that meant roughly 460,000 allocating multiplications per "
          "signature, which exhausted the heap."),
        p("The trap: the inversion routine takes and returns <em>ordinary</em> "
          "values, and passing it something already in Montgomery form computes the "
          "wrong thing silently. There is a wrapper that does the right conversion, "
          "and using it is not optional."),
        h2("What is still not safe"),
        ul("Validation <strong>reports</strong> rather than enforces, a caller "
           "that cares must check the result. This is a known gap.",
           "There is no revocation checking of any kind.",
           "<strong>Key material still comes from the timestamp counter, not "
           "<code>RDRAND</code>.</strong> That is a real weakness and it is on the "
           "list."),
        p("Stating these is the point. A system that claimed to be secure here "
          "would be worse than one that says exactly where it is not."),
        ("seealso", ["network-stack", "storage", "glados-os"]),
    ],
)

page(
    "wiki/storage",
    "Content-addressed storage. Merkle trees, NVMe, and O(1) snapshots",
    "How GLaDOS stores data: objects named by SHA-256 of their contents, "
    "assembled into Merkle trees, so a copy is free and a snapshot is one hash.",
    ["content addressed storage", "Merkle tree", "NVMe driver", "snapshots",
     "SHA-256", "filesystem design"],
    blocks=[
        p("The filesystem is not a tree of blocks with names attached. Objects are "
          "named by the SHA-256 of their contents and assembled into Merkle trees, "
          "which makes several operations collapse to nothing."),
        ul("<strong>A copy is O(1)</strong>. Identical content already has the "
           "same name, so copying is adding a reference.",
           "<strong>A snapshot is one hash</strong>. The root of the tree names "
           "the entire state.",
           "<strong>Rolling back branches rather than rewrites</strong>. The old "
           "root is still valid and still points at everything it did."),
        h2("The rule that makes it work"),
        p("<strong>The content hash covers content only, and never block "
          "locations.</strong> This sounds obvious and is easy to violate by "
          "including a block pointer in the hashed structure, at which point "
          "moving a block to defragment renames the object, every reference breaks, "
          "and deduplication silently stops working."),
        h2("Write locking, and why it stays that way"),
        p("NVMe writes are locked by default. They unlock only after a target "
          "region has been explicitly identified, the formatter re-checks before "
          "touching anything, and every error path re-locks."),
        p("On a laptop whose disk is fully allocated to Windows there is no such "
          "region, and storage initialisation <em>fails</em>. That is the intended "
          "outcome, not a bug to work around. Leaving the lock open on failure is "
          "exactly how a safety mechanism becomes decorative."),
        note("The development machine's boot disk is counterfeit: it advertises "
             "976 GB and holds 14.67. That is why the partitioning tooling uses MBR "
             ", a GPT backup header would be written to flash that does not exist. "
             "and why it carries an explicit safe-capacity limit."),
        ("seealso", ["glados-os", "tls", "testing"]),
    ],
)

page(
    "wiki/usb-xhci",
    "The USB stack. xHCI rings, cycle bits, and enumeration from scratch",
    "Writing an xHCI driver in a kernel: transfer request blocks, the cycle bit "
    "handshake, device slots, endpoint contexts, and CDC Ethernet over bulk pipes.",
    ["xHCI", "USB driver", "USB 3", "TRB ring", "CDC ECM", "USB Ethernet",
     "kernel USB stack"],
    blocks=[
        p("UEFI has a working USB stack, and "
          "<a href=\"uefi-kernel.html\"><code>ExitBootServices</code> throws it "
          "away</a> permanently. So the kernel has its own."),
        h2("Rings are the whole design"),
        p("xHCI is not register-poking like an older NIC. Everything is a ring of "
          "16-byte Transfer Request Blocks shared with the controller, and each "
          "ring carries a <strong>cycle bit</strong> that flips every time the "
          "producer wraps around."),
        p("That single bit is the entire synchronisation protocol. The controller "
          "consumes entries whose cycle matches its own state and stops at the "
          "first that does not. Get it wrong and nothing is corrupted. The "
          "controller simply never sees the command, which presents as a device "
          "that enumerates into silence."),
        p("Three rings matter: a command ring the driver writes, an event ring the "
          "controller writes, and one transfer ring per endpoint. Everything "
          "completes asynchronously through the event ring, even the operations "
          "that read as synchronous."),
        h2("Three bugs, each found by the layer above"),
        ol("<strong>Version 0.0.</strong> The capability-length and version fields "
           "share one 32-bit register, and this block only answers dword reads, a "
           "16-bit read at offset 2 returns zero. The controller appeared to "
           "implement xHCI version 0.0.",
           "<strong>Setup packet byte order.</strong> Little-endian means byte zero "
           "is the request type, so <code>SET_CONFIGURATION</code> is "
           "<code>0x0900</code>, not <code>0x0009</code>. The descriptor read next "
           "to it was correct by luck.",
           "<strong>A stall halts the endpoint.</strong> Probing configuration "
           "descriptors past the end stalls, which halts endpoint zero until it is "
           "reset. So every later control transfer failed. The symptom appeared "
           "two steps downstream of the cause."),
        h2("The shared event ring"),
        p("The subtlest one. The event ring is shared by every endpoint, and code "
          "that waits for \"a transfer event\" rather than \"a transfer event for "
          "this endpoint\" is only correct while exactly one transfer is in flight."),
        p("It stops being correct the moment a receive is left permanently armed: a "
          "send then completes against whichever event arrives first, so a "
          "transmit reports success on a <em>receive</em>, and the frame that "
          "actually arrived is dropped while the endpoint still looks armed."),
        h2("CDC Ethernet"),
        p("With bulk endpoints working, a USB Ethernet adapter becomes a network "
          "interface. One trap: \"has a CDC data interface\" is not the same as "
          "\"is an Ethernet adapter\", a common emulated device offers two "
          "configurations that both qualify, and the first is RNDIS, which accepts "
          "bulk writes and silently passes nothing because it wants a control "
          "protocol first."),
        p("The correct test is the Ethernet Networking Functional Descriptor, which "
          "is also where the MAC address lives, as a string, in ASCII hex, because "
          "CDC has no binary field for it anywhere."),
        ("seealso", ["usb-wifi-driver", "network-stack", "uefi-kernel"]),
    ],
)

page(
    "wiki/usb-wifi-driver",
    "USB WiFi on bare metal. CNVi, a Realtek dongle, and 509 constants",
    "Why the laptop's built-in WiFi cannot work, how a USB dongle sidesteps it, "
    "and why the RTL8188EU init tables were transcribed rather than written.",
    ["USB WiFi driver", "RTL8188EU", "CNVi", "WPA2", "802.11", "wireless driver",
     "rtl8xxxu"],
    blocks=[
        p("Wireless is the one major subsystem that does not work, and the reason "
          "is worth explaining because it is not a matter of effort."),
        h2("CNVi: the card is not a card"),
        p("The laptop's built-in wireless is CNVi, Intel's split architecture. The "
          "MAC. The part that would need a driver. Lives <em>in the chipset</em>, "
          "and the M.2 module is only a radio. Talking to it requires an "
          "undocumented signed-firmware protocol that is not published anywhere."),
        p("So the kernel's wireless module identifies the hardware and refuses to "
          "pretend it can drive it. The "
          "<a href=\"tls.html\">WPA2 supplicant</a> is complete, implements the "
          "four-way handshake, and passes IEEE 802.11i test vectors at every boot. "
          "and has never had a radio to run on."),
        h2("A USB dongle sidesteps all of it"),
        p("A USB wireless adapter is a complete radio <em>and</em> MAC behind a bus "
          "the kernel <a href=\"usb-xhci.html\">now drives</a>. The target is a "
          "Realtek RTL8188EU."),
        p("It has no memory-mapped registers at all. Every register is a vendor "
          "control transfer, so a single register read is a full USB round trip of "
          "a few hundred microseconds, a fact that shapes the entire driver, since "
          "anything that polls a register in a loop is far slower than it looks."),
        h2("Why the tables were copied, and why that is stated"),
        p("Bringing up the chip needs a power-on sequence, PHY and radio "
          "initialisation tables, and an efuse layout: <strong>509 specific "
          "register/value pairs</strong> that exist in Realtek's vendor driver and "
          "in Linux's rtl8xxxu, and nowhere else."),
        p("They were transcribed from Linux rather than written, and mechanically "
          "rather than by hand. 509 hex pairs retyped by a human contain errors at "
          "some rate, and an error here does not fail loudly. A wrong AGC value "
          "costs sensitivity; a wrong PHY value skews a filter. Both present as "
          "\"wireless is a bit unreliable.\""),
        note("Those constants are GPL-2.0 and are kept in a single file, marked as "
             "such. They are the only code in this kernel that was not written for "
             "it, apart from Rust's <code>core</code>. Saying so is cheaper than a "
             "blanket claim that quietly covers someone else's work."),
        h2("Order is content"),
        p("The AGC table writes the <em>same register</em> 130 times, with the "
          "index encoded in the value. Sorting or deduplicating it produces a "
          "different table that still looks entirely reasonable."),
        ("seealso", ["usb-xhci", "network-stack", "tls"]),
    ],
)

page(
    "wiki/iso-el-torito",
    "How the GLaDOS ISO is built. FAT32 and El Torito, written from scratch",
    "Building a bootable UEFI ISO without xorriso: a hand-written FAT32 EFI "
    "System Partition wrapped in ISO 9660 with an El Torito EFI boot catalog.",
    ["El Torito", "bootable ISO", "FAT32", "ISO 9660", "UEFI boot", "EFI System Partition",
     "mkisofs alternative"],
    blocks=[
        p("The ISO builder writes both filesystems itself. xorriso and mkisofs are "
          "not present on most Windows machines, and the Windows ADK's tool is a "
          "1.5 GB install to produce one 600 MB file. While both formats are "
          "published and neither is large."),
        h2("How UEFI boots an optical disc"),
        p("The firmware reads the El Torito boot catalog, looks for an entry whose "
          "platform id is <code>0xEF</code>, and treats the sectors it points at as "
          "a FAT filesystem, an EFI System Partition that happens to live inside "
          "an ISO."),
        p("So the ISO 9660 structure wrapped around it is almost ceremonial. It "
          "exists so the disc mounts and shows its contents, not so it boots. It is "
          "built correctly anyway, because a disc that boots and appears empty when "
          "mounted looks broken."),
        h2("Long file names are not optional"),
        p("The kernel opens <code>\\GLADOS\\tokenizer.bin</code>. That base name is "
          "nine characters, which has no 8.3 representation, a short-name-only "
          "image presents it as <code>TOKENI~1.BIN</code>, and the kernel then "
          "fails to find its tokenizer at boot, on real hardware, with no "
          "filesystem available to debug from."),
        p("So the builder generates VFAT long-name entries, with the specified "
          "rotate-and-add checksum tying each set to its short entry."),
        h2("FAT32 has a floor"),
        p("FAT32 is <em>defined</em> as having at least 65,525 clusters. Below "
          "that the volume is FAT16, and firmware that trusts the count misparses "
          "everything. So the cluster size sets a minimum image size."),
        table(
            ["Cluster size", "Smallest possible volume"],
            [["4 KiB", "~256 MB"],
             ["512 B", "~33 MB"]]),
        p("The kernel-only image was 269 MB for a 1 MB kernel before this was "
          "noticed. 255 MB of zeroes. Choosing the cluster size from the payload "
          "brings it to 33 MB, and that is still the floor rather than slack."),
        ("seealso", ["uefi-kernel", "download", "testing"]),
    ],
)

page(
    "wiki/templeos",
    "TempleOS. The lineage behind a single-address-space ring-0 kernel",
    "What TempleOS got right about single-address-space computing, and how GLaDOS "
    "relates to it: lineage, not target.",
    ["TempleOS", "Terry Davis", "single address space", "hobby OS", "ring 0",
     "HolyC", "operating system design"],
    blocks=[
        p("TempleOS is the obvious ancestor of a project like this, and it is worth "
          "being precise about what it shares and what it does not."),
        h2("What is shared"),
        ul("<strong>Ring 0, always.</strong> No user/kernel split, no syscalls.",
           "<strong>One address space.</strong> Everything can reach everything.",
           "<strong>No isolation, as a design choice</strong> rather than an "
           "oversight. With the costs accepted rather than hidden.",
           "<strong>A deliberate aesthetic.</strong> TempleOS committed to 640×480 "
           "and 16 colours; this one commits to "
           "<a href=\"aperture-science.html\">amber on black and Windows 3.1 "
           "chrome</a>.",
           "<strong>One person, from scratch.</strong>"),
        h2("What is not"),
        p("TempleOS had its own language, HolyC, as the shell and the system "
          "language at once. GLaDOS has "
          "<a href=\"glados-os.html\">a small interpreted language</a>, but the "
          "system is Rust and stays Rust."),
        p("More fundamentally, the organising idea is different. TempleOS was built "
          "around a specific personal vision of what a computer should be. This is "
          "built around one technical question: what changes when a language model "
          "is a kernel primitive instead of a userspace process?"),
        h2("What it got right"),
        p("That the boundary between kernel and user is a <em>choice</em>, not a "
          "law, and that removing it makes some things genuinely simpler rather "
          "than merely more dangerous. A function call is a function call. There is "
          "no marshalling, no permission check, no context switch, and no ABI to "
          "keep stable."),
        p("For a system where the model calls kernel functions millions of times, "
          "that stops being a philosophical position and becomes an engineering one."),
        note("Lineage, not target. This is not an attempt to continue TempleOS, and "
             "it is not a tribute to it. It is a different project that happens to "
             "have reached some of the same conclusions."),
        ("seealso", ["ring-0", "glados-os", "aperture-science", "rust-os"]),
    ],
)

page(
    "wiki/testing",
    "Testing an OS with no test runner. Boot self-tests and driving QEMU",
    "There is no cargo test for a no_std UEFI binary. How GLaDOS is verified: "
    "self-tests at boot, a scripted QEMU serial harness, and a NumPy oracle.",
    ["OS testing", "kernel testing", "QEMU automation", "self-test", "no_std testing",
     "embedded testing"],
    blocks=[
        p("There is no <code>cargo test</code>. This is a <code>no_std</code> UEFI "
          "binary with no host test runner, so the test suite had to be somewhere "
          "else. It is in three places."),
        h2("1. Self-tests at boot"),
        p("At every boot the system exercises the heap, timer, clock, namespace, "
          "13 sets of published cryptographic vectors, "
          "<a href=\"constrained-decoding.html\">constrained decoding</a> and the "
          "linear probe, printing <code>ok</code> or <code>FAIL</code> per line. "
          "That output <em>is</em> the test suite."),
        note("It is easy to scroll past, and it does catch real bugs. An ECDSA break "
             "was visible in the crypto block for an entire debugging cycle while "
             "the output was being sliced away to look at something else."),
        h2("2. A scripted QEMU harness"),
        p("A driver script boots QEMU, stages the binary, resets the firmware's "
          "NVRAM to a pristine state, and drives the shell over a serial socket, "
          "sending commands and capturing everything."),
        p("Two details that were not obvious. QEMU's Windows stdio backend reads "
          "console handles directly, so piping a script into it does nothing at all "
          ". Silently; a TCP socket behaves like a socket everywhere. And a stale "
          "firmware boot entry sends the machine to the UEFI shell, which looks "
          "exactly like the system failing to boot, so the NVRAM is reset every run."),
        p("The harness also captures the framebuffer through QEMU's monitor and "
          "converts it to PNG, because <a href=\"gui.html\">a serial log says "
          "nothing about whether the screen is right</a>."),
        h2("3. A numeric oracle"),
        p("For the model, a NumPy reference implementation reads the "
          "<em>converted</em> checkpoint and produces logits to diff against. "
          "Reading the converted file rather than the original is the point: a "
          "converter bug shows up in both, so only a genuine kernel bug appears as "
          "a mismatch."),
        p("This is what caught <a href=\"rope.html\">the RoPE convention</a>, which "
          "no amount of reading the code would have. Both conventions are "
          "perfectly reasonable code."),
        h2("The habit underneath"),
        p("Measure, do not assume; and look at it. Repeatedly in this project a "
          "confident belief turned out to be false, and a cheap deterministic test "
          "found it: the oracle found the RoPE bug, screenshots found three layout "
          "bugs the serial log could not, and an A/B against a deliberately tiny "
          "context isolated an off-by-one to a single token."),
        ("seealso", ["gui", "rope", "tls", "glados-os"]),
    ],
)


page(
    "screenshots/index",
    "GLaDOS OS screenshots. The desktop, the shell and the model",
    "Screenshots of GLaDOS: a Windows 3.1 styled desktop drawn straight into a "
    "UEFI framebuffer, the shell, and a language model answering from inside "
    "the kernel.",
    ["GLaDOS screenshots", "GLaDOS OS screenshots", "AI operating system screenshots",
     "Rust OS screenshots", "Aperture Science OS", "retro desktop"],
    kind="index",
    blocks=[
        p("Captured from the running system under QEMU, through the emulator's "
          "monitor rather than a camera pointed at a screen. These are the "
          "framebuffer's actual contents."),
        ("shots", [
            ("desktop.png", "Desktop with terminal and Program Manager",
             "GLaDOS OS desktop showing a Windows 3.1 styled terminal window with "
             "the boot log, a Program Manager window, and a taskbar",
             "The desktop as it comes up: the terminal showing its own boot log, "
             "Program Manager, and a taskbar. Every pixel is drawn by the kernel "
             "into a UEFI framebuffer. There is no graphics library under it.",
             True),
            ("desktop-clean.png", "Desktop wallpaper",
             "The GLaDOS desktop wallpaper showing the aperture mark drawn as "
             "vector geometry",
             "With the terminal minimised. The wallpaper mark is drawn as geometry "
             "at boot rather than decoded from an image. There is no image "
             "decoder in the kernel.", False),
            ("model.png", "The model answering",
             "GLaDOS running a language model in kernel space, answering a question "
             "about operating systems",
             "The resident model answering <em>what is an operating system</em> "
             "from inside the kernel. No process, no API, no server.", False),
        ]),
        h2("What you are looking at"),
        p("The window chrome is genuine Windows 3.1 construction: a "
          "<code>#C0C0C0</code> face, two-pixel bevels light on the top-left and "
          "dark on the bottom-right, title bars that fill with the selection "
          "colour when focused. Over that sits the "
          "<a href=\"../wiki/aperture-science.html\">Aperture palette</a>."),
        p("There is no graphics library beneath any of it. The kernel gets a base "
          "address, a width, a height and a stride from UEFI's Graphics Output "
          "Protocol, and everything above that. Glyphs, bevels, the window "
          "manager, the wallpaper. Is written by hand. See "
          "<a href=\"../wiki/gui.html\">the GUI page</a> for how, and for the bug "
          "that only a screenshot could have found."),
        ("seealso", ["gui", "glados-os", "download", "aperture-science"]),
    ],
)

# --------------------------------------------------------------------------
# Archive (the file listing, now a real page)
# --------------------------------------------------------------------------

page(
    "archive/index",
    "GLaDOS archive. ISO images, source snapshots and checksums",
    "Mirror-style file index for GLaDOS OS: bootable ISO images, source "
    "snapshots, documentation and SHA-256 checksums.",
    ["GLaDOS archive", "GLaDOS mirror", "ISO download", "software archive",
     "file index"],
    kind="archive",
    blocks=[
        p("A file index in the shape of the mirrors this sort of thing used to be "
          "distributed from. Images are served from GitHub Releases. GitHub Pages "
          "caps at 100 MB per file and would reject every one of them."),
        p("If you only want the ISO, "
          "<a href=\"../download/\">the download page</a> has flashing "
          "instructions and hardware notes alongside it."),
    ],
)

# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

CSS = """/* Layout copied in proportion from linux.org as it stood in 2005: a fixed
   centre column, a coloured masthead bar, a narrow left nav of boxed sections,
   and a white content well. Aperture supplies the two colours and nothing
   else. */

body {
  background: #e8e8e8;
  color: #000;
  margin: 0;
  padding: 0;
  font-family: Verdana, Arial, sans-serif;
  font-size: 12px;
  line-height: 1.5;
}

a { color: #964b00; }
a:visited { color: #6b3600; }
a:hover { color: #f28c1e; }

#page {
  width: 900px;
  margin: 0 auto;
  background: #fff;
  border-left: 1px solid #999;
  border-right: 1px solid #999;
}

#masthead {
  background: #b35900;
  padding: 8px 10px;
  overflow: hidden;
}
#masthead img { float: left; margin-right: 10px; }
#masthead .name {
  color: #fff;
  font-size: 19px;
  font-weight: bold;
  text-decoration: none;
}
#masthead .sub { color: #f4e2c8; font-size: 11px; display: block; }

#bar {
  background: #f4e6d0;
  border-top: 1px solid #b35900;
  border-bottom: 1px solid #b35900;
  padding: 3px 10px;
  font-size: 11px;
}
#bar a { text-decoration: none; }
#bar a:hover { text-decoration: underline; }

#body { overflow: hidden; padding: 10px; }
#sidebar { float: left; width: 150px; }
#content { margin-left: 162px; }

.box { margin-bottom: 10px; }
.box .t {
  background: #b35900;
  color: #fff;
  font-size: 11px;
  font-weight: bold;
  padding: 2px 5px;
}
.box ul {
  margin: 0;
  padding: 4px 4px 5px 20px;
  background: #f4e6d0;
  border: 1px solid #d8bc94;
  border-top: 0;
  font-size: 11px;
  list-style: square;
}
.box li { margin: 2px 0; }
.box a { text-decoration: none; }
.box a:hover { text-decoration: underline; }

h1 {
  font-size: 18px;
  font-weight: bold;
  margin: 0 0 6px;
  padding-bottom: 3px;
  border-bottom: 1px solid #b35900;
}
h2 {
  font-size: 14px;
  font-weight: bold;
  margin: 18px 0 5px;
  padding-bottom: 2px;
  border-bottom: 1px solid #ccc;
}
h3 { font-size: 12px; font-weight: bold; margin: 14px 0 4px; }

p { margin: 8px 0; }
ul, ol { margin: 8px 0; padding-left: 22px; }
li { margin: 3px 0; }

code {
  font-family: "Courier New", monospace;
  background: #f4f4f4;
  border: 1px solid #ddd;
  padding: 0 2px;
  font-size: 11px;
}
pre {
  font-family: "Courier New", monospace;
  background: #f8f8f8;
  border: 1px solid #ccc;
  padding: 7px 9px;
  overflow-x: auto;
  margin: 10px 0;
  font-size: 11px;
  line-height: 1.4;
}
pre code { background: none; border: 0; padding: 0; }

.toc {
  display: table;
  background: #f9f9f9;
  border: 1px solid #ccc;
  padding: 5px 14px 5px 5px;
  margin: 10px 0;
  font-size: 11px;
}
.toc .h { font-weight: bold; text-align: center; margin-bottom: 3px; }
.toc ol { margin: 0; padding-left: 22px; }

.note {
  background: #f4e6d0;
  border: 1px solid #d8bc94;
  padding: 6px 9px;
  margin: 10px 0;
}

table.data { border-collapse: collapse; margin: 10px 0; font-size: 11px; }
table.data th, table.data td {
  border: 1px solid #ccc;
  padding: 3px 7px;
  text-align: left;
  vertical-align: top;
}
table.data th { background: #f4e6d0; }

.infobox {
  float: right;
  width: 230px;
  border: 1px solid #ccc;
  background: #f9f9f9;
  margin: 0 0 10px 12px;
  font-size: 11px;
}
.infobox .ib-t {
  background: #b35900;
  color: #fff;
  font-weight: bold;
  padding: 3px 6px;
  text-align: center;
}
.infobox table { border-collapse: collapse; width: 100%; }
.infobox td { padding: 2px 6px; border-bottom: 1px solid #e2e2e2; }
.infobox tr:last-child td { border-bottom: 0; }
.infobox td:first-child { color: #555; width: 42%; }

table.dl { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 11px; }
table.dl th, table.dl td {
  border: 1px solid #ccc;
  padding: 4px 7px;
  text-align: left;
  vertical-align: top;
}
table.dl th { background: #f4e6d0; }
table.dl td.size { white-space: nowrap; }

.shot { margin: 12px 0; }
.shot img { display: block; border: 1px solid #999; max-width: 100%; height: auto; }
.shot .cap { font-size: 11px; color: #444; margin-top: 3px; }

.wikilist { list-style: square; }
.wikilist span { color: #555; }

.seealso { margin-top: 18px; padding-top: 6px; border-top: 1px solid #ccc; }
dl.faq dt { font-weight: bold; margin-top: 10px; }
dl.faq dd { margin: 3px 0 0 20px; }

#footer {
  clear: both;
  background: #f4e6d0;
  border-top: 1px solid #b35900;
  padding: 6px 10px;
  font-size: 10px;
  color: #555;
}
#footer p { margin: 3px 0; }

.crumbs { font-size: 11px; color: #555; margin-bottom: 6px; }
.crumbs a { color: #555; }

.listing { border-collapse: collapse; width: 100%; font-size: 11px; }
.listing th {
  border-bottom: 1px solid #999;
  text-align: left;
  padding: 2px 8px 2px 0;
}
.listing td { padding: 2px 8px 2px 0; border-bottom: 1px solid #eee; }
.icon { color: #777; display: inline-block; width: 2.6em; }

.skip { position: absolute; left: -9999px; }

@media (max-width: 920px) {
  #page { width: auto; border: 0; }
  #sidebar { float: none; width: auto; }
  #content { margin-left: 0; }
  .infobox { float: none; width: 100%; margin-left: 0; }
}
"""

JS = """// Archive listing. The only script on the site, and only this page needs it.
(function(){
  var rows=document.getElementById('rows'); if(!rows) return;
  var REPO='%REPO%', REL='%REL%';
  var TREE={
    '':{kids:[
      {n:'iso/',t:'dir',d:'bootable images'},
      {n:'src/',t:'dir',d:'source tree'},
      {n:'docs/',t:'dir',d:'documentation'},
      {n:'checksums/',t:'dir',d:'digests and verification'}]},
    'iso':{kids:[
      {n:'glados-qwen3-0.6b.iso',t:'f',d:'kernel + Qwen3-0.6B, int8',
       href:REL+'glados-qwen3-0.6b.iso',size:'575M'},
      {n:'glados-smollm2-135m.iso',t:'f',d:'kernel + SmolLM2-135M',
       href:REL+'glados-smollm2-135m.iso',size:'257M'},
      {n:'glados-nomodel.iso',t:'f',d:'kernel only',
       href:REL+'glados-nomodel.iso',size:'33M'}]},
    'src':{kids:[
      {n:'glados-src.tar.gz',t:'f',d:'repository snapshot',
       href:REPO+'/archive/refs/heads/main.tar.gz',size:'700K'},
      {n:'browse/',t:'l',d:'read it on GitHub',href:REPO}]},
    'docs':{kids:[
      {n:'README.md',t:'f',d:'overview and build instructions',
       href:REPO+'/blob/main/README.md',size:'13K'},
      {n:'wiki/',t:'l',d:'the documentation wiki',href:'../wiki/'}]},
    'checksums':{kids:[
      {n:'SHA256SUMS',t:'f',d:'digests for every image',
       href:REL+'SHA256SUMS',size:'312'}]}
  };
  var ICON={dir:'[DIR]',f:'[   ]',l:'[LNK]',up:'[UP ]'};
  function td(cls,txt){var e=document.createElement('td');
    if(cls)e.className=cls; if(txt!==undefined)e.textContent=txt; return e;}
  // Path is a parameter with the hash as default, so it can be tested without
  // driving a browser at a real URL.
  function render(explicit){
    var src=explicit!==undefined?explicit:location.hash;
    var path=(String(src).replace(/^#\\/?/,'')||'').replace(/\\/$/,'');
    var node=TREE[path];
    rows.innerHTML='';
    var t=document.getElementById('atitle');
    if(!node){ t.textContent='Index of /'+path+'. Not found';
      var tr=document.createElement('tr'), c=td('d');
      c.colSpan=4; c.innerHTML='No such directory. <a href="#/">Return to root</a>.';
      tr.appendChild(c); rows.appendChild(tr); return; }
    t.textContent='Index of /'+(path?path+'/':'');
    if(path){ node={kids:[{n:'Parent Directory',t:'up',
      href:'#/'+path.split('/').slice(0,-1).join('/'),d:''}].concat(node.kids)}; }
    node.kids.forEach(function(k){
      var tr=document.createElement('tr');
      var n=td(); var i=document.createElement('span');
      i.className='icon'; i.textContent=ICON[k.t]||ICON.f; n.appendChild(i);
      var a=document.createElement('a');
      a.href=k.t==='dir'?'#/'+k.n.replace(/\\/$/,''):k.href;
      a.textContent=k.n; n.appendChild(a);
      tr.appendChild(n);
      tr.appendChild(td('s',k.size||'-'));
      tr.appendChild(td('d',k.d||''));
      rows.appendChild(tr);
    });
  }
  window.renderArchive=render;
  addEventListener('hashchange',function(){render();});
  render();
})();
"""


def esc(t):
    return html.escape(t, quote=False)


def rel_prefix(slug):
    """Path back to the site root from a page's own directory."""
    depth = slug.count("/")
    if slug.endswith("index"):
        depth = slug.count("/")
    return "../" * depth if depth else ""


def out_path(slug):
    return slug + ".html" if not slug.endswith("index") else slug + ".html"


def url_for(slug, from_slug):
    """A relative link between two pages, so the site works from any prefix."""
    target = out_path(slug)
    up = rel_prefix(from_slug)
    # Directory-style URLs for index pages: /wiki/ rather than /wiki/index.html.
    if target.endswith("/index.html"):
        target = target[: -len("index.html")]
    elif target == "index.html":
        target = ""
    return up + target


def canonical(slug):
    t = out_path(slug)
    if t.endswith("/index.html"):
        t = t[: -len("index.html")]
    elif t == "index.html":
        t = ""
    return BASE + "/" + t


def render_blocks(blocks, slug):
    out = []
    up = rel_prefix(slug)
    for b in blocks:
        kind, val = b
        if kind == "p":
            out.append(f"<p>{val}</p>")
        elif kind == "h2":
            out.append(f'<h2 id="{slugify(val)}">{esc(val)}</h2>')
        elif kind == "h3":
            out.append(f"<h3>{esc(val)}</h3>")
        elif kind == "ul":
            items = "".join(f"<li>{i}</li>" for i in val)
            out.append(f"<ul>{items}</ul>")
        elif kind == "ol":
            items = "".join(f"<li>{i}</li>" for i in val)
            out.append(f"<ol>{items}</ol>")
        elif kind == "code":
            text, lang = val
            out.append(f"<pre><code>{esc(text)}</code></pre>")
        elif kind == "note":
            out.append(f'<div class="note">{val}</div>')
        elif kind == "table":
            head, rows = val
            th = "".join(f"<th>{esc(h)}</th>" for h in head)
            tr = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
            )
            out.append(f'<table class="data"><thead><tr>{th}</tr></thead>'
                       f"<tbody>{tr}</tbody></table>")
        elif kind == "infobox":
            title, pairs = val
            tr = "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in pairs)
            out.append(f'<aside class="infobox"><div class="ib-t">{esc(title)}</div>'
                       f"<table><tbody>{tr}</tbody></table></aside>")
        elif kind == "cards":
            # A plain list. Boxed "cards" are a modern affectation and this
            # layout predates them by twenty years.
            items = "".join(f'<li><a href="{h}">{esc(t)}</a>: {esc(d)}</li>'
                            for t, h, d in val)
            out.append(f"<ul>{items}</ul>")
        elif kind == "downloads":
            rows = "".join(
                f'<tr><td><a href="{h}">{esc(n)}</a></td>'
                f'<td class="size">{esc(s)}</td><td>{esc(d)}</td></tr>'
                for n, s, h, d in val)
            out.append('<table class="dl"><thead><tr><th>File</th><th>Size</th>'
                       f"<th>Contents</th></tr></thead><tbody>{rows}</tbody></table>")
        elif kind == "shots":
            for f, t_, alt, c, wide in val:
                out.append(
                    f'<div class="shot"><img src="{up}img/{f}" '
                    f'alt="{html.escape(alt, quote=True)}" width="1280" '
                    f'height="800" loading="lazy" decoding="async">'
                    f'<div class="cap"><b>{esc(t_)}.</b> {c}</div></div>')
        elif kind == "wikilist":
            ls = "".join(
                f'<li><a href="{url_for("wiki/" + s, slug)}">{esc(t)}</a>'
                f"<span>{esc(d)}</span></li>" for s, t, d in val)
            out.append(f'<ul class="wikilist">{ls}</ul>')
        elif kind == "seealso":
            links = []
            for s in val:
                # Bare slugs mean wiki pages; anything with a path is literal.
                # The top-level pages must be checked *before* that default, or
                # "download" silently becomes "wiki/download" and links nowhere.
                if s in ("download", "archive", "wiki", "screenshots"):
                    target = s + "/index"
                elif "/" in s:
                    target = s
                else:
                    target = "wiki/" + s
                if target not in PAGES:
                    raise KeyError(f"see-also points at a page that does not exist: {s}")
                title = PAGES.get(target, {}).get("nav") or short_title(target)
                links.append(f'<a href="{url_for(target, slug)}">{esc(title)}</a>')
            out.append('<div class="seealso"><b>See also:</b> '
                       + ", ".join(links) + "</div>")
        elif kind == "faq":
            items = "".join(f"<dt>{esc(q)}</dt><dd>{esc(a)}</dd>" for q, a in val)
            out.append(f'<h2 id="faq">Frequently asked questions</h2>'
                       f'<dl class="faq">{items}</dl>')
    return "\n".join(out)


def slugify(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def short_title(slug):
    for s, t, _ in WIKI:
        if "wiki/" + s == slug:
            return t
    return {"download/index": "Download", "archive/index": "Archive",
            "wiki/index": "Wiki", "index": "Home",
            "screenshots/index": "Screenshots"}.get(slug, slug)


def jsonld(pg):
    """Structured data. Search engines read this; humans never see it."""
    slug = pg["slug"]
    url = canonical(slug)
    crumbs = [{"@type": "ListItem", "position": 1, "name": "GLaDOS",
               "item": BASE + "/"}]
    if slug.startswith("wiki/") and slug != "wiki/index":
        crumbs.append({"@type": "ListItem", "position": 2, "name": "Wiki",
                       "item": BASE + "/wiki/"})
        crumbs.append({"@type": "ListItem", "position": 3,
                       "name": short_title(slug), "item": url})
    elif slug != "index":
        crumbs.append({"@type": "ListItem", "position": 2,
                       "name": short_title(slug), "item": url})

    blocks = [{
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": crumbs,
    }]

    if pg["kind"] == "home":
        blocks.append({
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": "GLaDOS",
            "applicationCategory": "OperatingSystem",
            "operatingSystem": "x86-64 UEFI (bare metal)",
            "description": pg["desc"],
            "url": BASE + "/",
            "downloadUrl": BASE + "/download/",
            "softwareVersion": "0.1",
            "programmingLanguage": "Rust",
            "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
        })
    else:
        blocks.append({
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": pg["title"],
            "description": pg["desc"],
            "url": url,
            "dateModified": UPDATED,
            "keywords": ", ".join(pg["keywords"]),
            "inLanguage": "en",
            "isPartOf": {"@type": "WebSite", "name": "GLaDOS", "url": BASE + "/"},
        })

    fq = [b for b in pg["blocks"] if b[0] == "faq"]
    if fq:
        blocks.append({
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": q,
                 "acceptedAnswer": {"@type": "Answer", "text": a}}
                for q, a in fq[0][1]
            ],
        })

    import json
    return "\n".join(
        f'<script type="application/ld+json">{json.dumps(b, separators=(",", ":"))}</script>'
        for b in blocks)


NAVITEMS = [("index", "Home"), ("download/index", "Download"),
            ("wiki/index", "Wiki"), ("screenshots/index", "Screenshots"),
            ("archive/index", "Archive")]


def render_page(pg):
    slug = pg["slug"]
    up = rel_prefix(slug)
    bar = " | ".join(
        f'<a href="{url_for(s, slug)}">{esc(t)}</a>' for s, t in NAVITEMS)
    bar += f' | <a href="{REPO}" rel="noopener">Source code</a>'

    def sidebox(title, items):
        li = "".join(f'<li><a href="{h}">{esc(n)}</a></li>' for n, h in items)
        return (f'<div class="box"><div class="t">{esc(title)}</div>'
                f"<ul>{li}</ul></div>")

    side = sidebox("Navigation",
                   [(n, url_for(s, slug)) for s, n in NAVITEMS]
                   + [("Source code", REPO)])
    side += sidebox("Get it", [
        ("Download the ISO", url_for("download/index", slug)),
        ("Checksums", REL + "SHA256SUMS"),
    ])
    side += sidebox("The system",
                    [(ti, url_for("wiki/" + sl, slug)) for sl, ti, _ in WIKI[:8]])
    side += sidebox("Internals",
                    [(ti, url_for("wiki/" + sl, slug)) for sl, ti, _ in WIKI[8:16]])
    side += sidebox("Hardware",
                    [(ti, url_for("wiki/" + sl, slug)) for sl, ti, _ in WIKI[16:]])

    crumbs = ""
    if slug != "index":
        parts = [f'<a href="{url_for("index", slug)}">GLaDOS</a>']
        if slug.startswith("wiki/") and slug != "wiki/index":
            parts.append(f'<a href="{url_for("wiki/index", slug)}">Wiki</a>')
        parts.append(esc(short_title(slug)))
        crumbs = f'<div class="crumbs">{" / ".join(parts)}</div>'

    body = render_blocks(pg["blocks"], slug)
    if pg["kind"] == "archive":
        body += ARCHIVE_HTML.replace(
            "%ARCHIVE_JS%", JS.replace("%REPO%", REPO).replace("%REL%", REL))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(pg["title"])}</title>
<meta name="description" content="{html.escape(pg["desc"], quote=True)}">
<meta name="keywords" content="{html.escape(", ".join(pg["keywords"]), quote=True)}">
<link rel="canonical" href="{canonical(slug)}">
<meta name="robots" content="index,follow,max-image-preview:large">
<meta property="og:type" content="website">
<meta property="og:site_name" content="GLaDOS">
<meta property="og:title" content="{html.escape(pg["title"], quote=True)}">
<meta property="og:description" content="{html.escape(pg["desc"], quote=True)}">
<meta property="og:url" content="{canonical(slug)}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{html.escape(pg["title"], quote=True)}">
<meta name="twitter:description" content="{html.escape(pg["desc"], quote=True)}">
<meta property="og:image" content="{BASE}/img/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="The GLaDOS aperture mark">
<meta name="twitter:image" content="{BASE}/img/og.png">
<link rel="stylesheet" href="{up}style.css">
<link rel="icon" type="image/svg+xml" href="{up}img/logo.svg">
<link rel="icon" type="image/png" sizes="32x32" href="{up}img/icon-32.png">
<link rel="apple-touch-icon" sizes="180x180" href="{up}img/icon-180.png">
<link rel="manifest" href="{up}site.webmanifest">
{jsonld(pg)}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div id="page">

<div id="masthead">
  <img src="{up}img/logo.svg" alt="" width="34" height="34">
  <a class="name" href="{url_for("index", slug)}">GLaDOS</a>
  <span class="sub">An operating system in Rust, with a language model in the kernel</span>
</div>

<div id="bar">{bar}</div>

<div id="body">
<div id="sidebar">{side}</div>
<div id="content">
{crumbs}
<div id="main">
<h1>{esc(pg["title"].split(". ")[0].split(". ")[0])}</h1>
{body}
</div>
</div>
</div>

<div id="footer">
  <p>{esc(DISCLAIMER)}</p>
  <p>Copyright 2026. All rights reserved. Source is published to be read, not
  reused. One file (<code>src/dev/rtl8188eu_tables.rs</code>) is GPL-2.0 from
  the Linux kernel and carries its own terms.</p>
  <p>No cookies, no analytics. Last updated {UPDATED}.</p>
</div>

</div>
</body>
</html>
"""


ARCHIVE_HTML = """
<h2 id="atitle">Index of /</h2>
<table class="listing">
<thead><tr><th>Name</th><th>Size</th><th>Description</th></tr></thead>
<tbody id="rows"></tbody>
</table>
<script>%ARCHIVE_JS%</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "style.css").write_text(CSS, encoding="utf-8")

    written = []
    for slug, pg in PAGES.items():
        f = out / out_path(slug)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(render_page(pg), encoding="utf-8")
        written.append(slug)

    # A 404 that keeps the navigation, so a bad link is recoverable rather than
    # a dead end. GitHub Pages serves docs/404.html automatically.
    nf = dict(slug="404", title="404. Page not found",
              desc="That page does not exist. The wiki index lists everything.",
              keywords=["404"], kind="article", nav=None, faqs=None,
              blocks=[p("No page at that address."),
                      p('Try <a href="/GLaDOS/wiki/">the wiki index</a>, '
                        '<a href="/GLaDOS/download/">the downloads</a>, or '
                        '<a href="/GLaDOS/">the front page</a>.')])
    PAGES["404"] = nf
    (out / "404.html").write_text(render_page(nf), encoding="utf-8")

    # sitemap.xml, so every generated page is discoverable rather than relying
    # on the crawler finding its way through internal links.
    urls = "".join(
        f"<url><loc>{canonical(s)}</loc><lastmod>{UPDATED}</lastmod>"
        f"<changefreq>weekly</changefreq>"
        f"<priority>{'1.0' if s == 'index' else '0.8' if '/' not in s or s.endswith('/index') else '0.6'}</priority>"
        f"</url>" for s in written)
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + urls + "</urlset>\n", encoding="utf-8")

    import json as _j
    (out / "site.webmanifest").write_text(_j.dumps({
        "name": "GLaDOS", "short_name": "GLaDOS",
        "description": "An AI operating system in Rust, ring 0, bare metal.",
        "start_url": BASE + "/", "display": "standalone",
        "background_color": "#0b0b0c", "theme_color": "#f28c1e",
        "icons": [
            {"src": "img/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "img/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {"src": "img/logo.svg", "sizes": "any", "type": "image/svg+xml"},
        ],
    }, indent=1), encoding="utf-8")

    (out / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n\nSitemap: " + BASE + "/sitemap.xml\n",
        encoding="utf-8")

    # Stops GitHub Pages running the output through Jekyll, which would drop any
    # file beginning with an underscore and is pure overhead for static HTML.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    print(f"{len(written) + 1} pages -> {out}/")
    for s in written:
        print("  " + out_path(s))


if __name__ == "__main__":
    main()
