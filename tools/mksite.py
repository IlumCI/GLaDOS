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
import json
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


def fig(key, caption, alt, wide=False):
    """A borrowed diagram. `key` indexes MEDIA, which fetch-media.py wrote when
    it downloaded the file, so the filename, the dimensions and the credit line
    all come from the same record. Writing them out here by hand is how a
    caption ends up attributing the wrong picture."""
    return ("fig", (key, caption, alt, wide))


PAGES = {}

# Provenance for every borrowed image, written by tools/fetch-media.py. Loaded
# rather than inlined so that re-fetching a file updates the credit with it.
# Missing is not fatal: a checkout without the images should still build, it
# just builds without them.
MEDIA = {}
_mediafile = Path(__file__).with_name("media-credits.json")
if _mediafile.exists():
    MEDIA = json.loads(_mediafile.read_text(encoding="utf-8"))


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
        p("<strong>GLaDOS</strong> is an operating system for x86-64, written from "
          "scratch in Rust, which boots on bare metal and runs entirely in ring 0. "
          "Its distinguishing feature is that a transformer lives in kernel space. "
          "The model is compiled into the same binary as the page tables and the "
          "disk driver, and shares an address space with both."),
        p("That arrangement deletes most of the machinery an AI agent usually needs. "
          "There is no userspace to marshal arguments into, no wire format, no "
          "dispatcher and no process boundary to cross. When the model decides to "
          "run <code>ls</code>, the kernel calls <code>ls</code>, and the call costs "
          "what any other function call costs."),
        p("The name comes from <em>Portal</em>, and so does the look: amber on "
          "black, a boot screen drawn as vector geometry, Windows 3.1 chrome, and a "
          "certain institutional cheerfulness on the subject of hazards. What is "
          "underneath the paint is ordinary systems work. The TCP/IP stack, the "
          "TLS 1.3 client, the NVMe and xHCI drivers, the window manager and the "
          "inference code were all written for this project, which comes to 89 files "
          "of Rust and about 36,000 lines."),
        ("cards", [
            ("Download the ISO", "download/",
             "Bootable UEFI images with the model baked in. 33 MB to 575 MB."),
            ("Read the wiki", "wiki/",
             "Every subsystem, how it works, and what it cost to find out."),
            ("See it running", "screenshots/",
             "The desktop, the shell, and the model answering from ring 0."),
            ("Browse the archive", "archive/",
             "Mirror-style file index of images, sources and checksums."),
            ("Source on GitHub", REPO,
             "89 files of Rust. All rights reserved, published to be read."),
        ]),
        ("shots", [
            ("desktop.png", "Desktop with terminal and Program Manager",
             "GLaDOS OS desktop showing a Windows 3.1 styled terminal window with "
             "the boot log, a Program Manager window, and a taskbar",
             "The desktop as it comes up, with the terminal showing its own boot "
             "log, Program Manager beside it and a taskbar along the bottom. Every "
             "pixel here was drawn by the kernel into a UEFI framebuffer, with no "
             "graphics library underneath.",
             True),
            ("desktop-clean.png", "Desktop wallpaper",
             "The GLaDOS desktop wallpaper showing the aperture mark drawn as "
             "vector geometry",
             "The same desktop with the terminal minimised. The mark is computed as "
             "arcs and lines at boot, since the kernel has no image decoder and "
             "writing one in order to display a logo seemed like the wrong order to "
             "do things in.", False),
            ("model.png", "The model answering",
             "GLaDOS running a language model in kernel space, answering a question "
             "about operating systems",
             "The resident model answering <em>what is an operating system</em>. The "
             "forward pass is running in the same address space as the code that "
             "drew the window it is printing into.", False),
        ]),
        h2("What works"),
        p("This is a research kernel, so a feature list is only useful with the "
          "gaps marked. Both halves are below."),
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
        h3("Known limits"),
        ul("Wireless does not work. The laptop's card is CNVi, which means the MAC "
           "sits in the chipset and the M.2 module is only a radio, reachable through "
           "an undocumented signed-firmware protocol. The WPA2 supplicant is finished, "
           "passes the IEEE 802.11i test vectors at every single boot, and has never "
           "been allowed within reach of an antenna. See "
           "<a href=\"wiki/usb-wifi-driver.html\">the USB WiFi work</a>.",
           "One core. SMP is unimplemented, and the interior-mutability type is called "
           "<code>Racy</code> so that the day it matters, one grep finds every place "
           "that has to be revisited.",
           "There is no autonomous agent loop. The model proposes and a keystroke "
           "adopts. Given the name over the door, this is arguably the most important "
           "design decision in the project.",
           "TLS validates certificates and then reports the verdict without acting on "
           "it, there is no revocation checking, and key material comes from the "
           "timestamp counter. Treat the network stack as an instrument, not as "
           "armour."),
        h2("Why it exists"),
        p("An operating system advertised as AI-powered is usually a chat window with "
          "a Linux underneath, which answers a question about product design. The "
          "question here is narrower: if the model is a kernel primitive, what "
          "actually changes?"),
        p("Three answers so far, each of which took measuring. Tool calls stop being "
          "text: the model names a function and the kernel calls it, with nothing "
          "parsed in between. Invalid tool names become "
          "<a href=\"wiki/constrained-decoding.html\">unreachable</a>, because the "
          "grammar handed to the sampler is compiled from the live applet table, and "
          "permissions are enforced by removing applets from that grammar before "
          "sampling starts. And the best router in the system is not the transformer: "
          "it is <a href=\"wiki/routing.html\">a ridge regression published in 1960</a>, "
          "reading one hidden state through 12,672 parameters, which beats the 135M "
          "model at picking the right applet by a factor of nine and does it without "
          "a forward pass."),
        p("The last one is the reason the wiki is written the way it is. A project "
          "like this generates confident assumptions at a rate that only measurement "
          "keeps up with, and several of the pages here exist to record an assumption "
          "that lost."),
        note("<a href=\"wiki/templeos.html\">TempleOS</a> is the obvious ancestor: "
             "one person, ring 0, one address space, no isolation, and an aesthetic "
             "chosen on purpose. The identity map here exists for the reason Terry "
             "Davis gave for his, and it works for the same reason."),
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
        p("Three images, all bootable UEFI ISOs. Each one is a FAT32 EFI System "
          "Partition wrapped in an ISO 9660 filesystem with an El Torito boot "
          "entry pointing at it, and all three layers are written by "
          "<a href=\"../wiki/iso-el-torito.html\">this project's own ISO writer</a>, not by <code>mkisofs</code>. "
          "The difference is invisible to your "
          "firmware and was not invisible to write."),
        ("downloads", [
            ("glados-qwen3-0.6b.iso", "575 MB", REL + "glados-qwen3-0.6b.iso",
             "The full system: kernel plus Qwen3-0.6B quantised to int8, with a "
             "512-token context window. Take this one unless you have a reason not "
             "to. It needs real hardware; QEMU cannot serve a disk this size."),
            ("glados-smollm2-135m.iso", "257 MB", REL + "glados-smollm2-135m.iso",
             "Kernel plus SmolLM2-135M. Four times faster per token because "
             "generation is bound by how many bytes of weights get read, and small "
             "enough to fit QEMU's 516 MB disk ceiling, which makes it the image to "
             "develop against."),
            ("glados-nomodel.iso", "33 MB", REL + "glados-nomodel.iso",
             "Kernel only. Boots to a desktop, reports that it has no model, and "
             "everything except inference works. Drop a converted checkpoint on the "
             "EFI System Partition and it will find it."),
        ]),
        p("The same files are browsable in "
          "<a href=\"../archive/#/iso\">the archive index</a>, if you prefer a "
          "directory listing."),
        p("Checksums: <a href=\"" + REL + "SHA256SUMS\">SHA256SUMS</a>. Worth "
          "verifying. A truncated download still flashes, still boots, and then "
          "fails somewhere that has nothing to do with the truncation, usually "
          "while loading weights."),
        h2("Flashing it"),
        p("Write the image to a USB stick. It is a hybrid image, so any of these "
          "work:"),
        code("# Linux / macOS. Check the device name first, this overwrites it\n"
             "sudo dd if=glados-qwen3-0.6b.iso of=/dev/sdX bs=4M status=progress oflag=sync",
             "bash"),
        ul("On Windows, Rufus in DD mode.",
           "Anywhere, balenaEtcher handles the ISO directly."),
        note("<strong>Secure Boot has to be off.</strong> The kernel is unsigned and "
             "there is no Microsoft-signed shim in front of it, so firmware with "
             "Secure Boot enabled will refuse to load it. What it tells you about "
             "that refusal is usually \"no bootable device\", which is true in the "
             "same way that a locked door is not a wall."),
        h2("Booting it"),
        ol("Write the image and leave the stick in.",
           "Reboot and open the firmware boot menu. Commonly F11, F12, F8 or Esc.",
           "Pick the USB device listed under UEFI.",
           "Watch the boot log. It prints the memory map it was handed, the size of "
           "heap it managed to get, every device it finds on the PCI bus, and a pass "
           "or fail line for each self-test."),
        p("That log rewards a read. The self-tests run eleven "
          "sets of RFC test vectors through the crypto primitives, exercise the heap "
          "and the timer, and check that the grammar the sampler is given admits "
          "exactly the applet names it should. An ECDSA bug once sat visible in that "
          "output for an entire debugging session while the person reading the screen "
          "was scrolling past it."),
        h2("What you get"),
        p("A Windows 3.1-styled desktop with a window manager, a taskbar and a "
          "terminal, and behind that a shell. <code>gen</code> generates text and "
          "<code>ask</code> answers a question, both from the model in kernel space. "
          "<code>if</code> lists interfaces, <code>dhcp</code> takes a lease, "
          "<code>dns</code> resolves a name and <code>https example.com /</code> "
          "fetches a page over a TLS 1.3 connection that this kernel negotiated "
          "itself, down to the X25519 key exchange. <code>crypto</code>, "
          "<code>tensor</code> and <code>model</code> re-run the self-tests on "
          "demand, and <code>help</code> lists the rest."),
        h2("Hardware support, honestly"),
        p("Development happens against one laptop, an MSI Thin GF63 12UC, and that "
          "is the only machine any of this is meaningfully tested on."),
        p("It should boot on most x86-64 UEFI systems. Nothing in the boot path is "
          "vendor-specific and the graphics path is plain GOP, which every UEFI "
          "implementation provides. Storage and networking are a different story, "
          "because a driver has to match a chip: there is one for Intel e1000, one "
          "for Realtek RTL8168, one for NVMe and one for xHCI, and that is the whole "
          "list. So expect a working desktop and a working model, and treat your disk "
          "and your network card as an open question."),
        faq([
            ("Is the GLaDOS ISO free?",
             "Free to download and free to run. The source is published under "
             "all-rights-reserved, which means you are welcome to read it and not "
             "to redistribute it or build derivatives from it without asking."),
            ("Will it damage my computer or my files?",
             "It runs from the USB stick and installs nothing. NVMe writes are "
             "locked at boot and only unlock if the driver finds a disk region that "
             "was explicitly set aside for it, and every error path locks them again. "
             "On a laptop whose disk is entirely allocated to Windows there is no "
             "such region, so storage initialisation fails and says so, which is the "
             "intended outcome."),
            ("Can I run it in a virtual machine?",
             "Yes, with UEFI firmware, which means OVMF. Use the SmolLM2 image: "
             "QEMU's built-in FAT support caps the whole emulated disk at 516 MB and "
             "the Qwen3 image is larger than that. Give the guest at least 2 GB of "
             "RAM as well, because the weights are read into memory before "
             "ExitBootServices and the guest has to have somewhere to put them."),
            ("Why does it need Secure Boot disabled?",
             "Because the kernel is not signed by a key your firmware trusts. Getting "
             "one would mean going through a Microsoft-signed shim, which is a distribution and paperwork problem, not a technical "
             "one."),
            ("How fast is the model?",
             "Generation is bound by memory bandwidth, so the useful rule is that "
             "each token costs roughly one pass over the weights. That makes the "
             "570 MB Qwen3 checkpoint about 4.4 times slower per token than the "
             "135 MB SmolLM2 one. Around 155 MB of Qwen3 is the output classifier "
             "alone, which is why restricting that final matrix multiply to the "
             "tokens a grammar can actually reach is such an effective optimisation."),
            ("Does it send anything anywhere?",
             "No. There is no telemetry, no update check and no network activity at "
             "all unless you type a command that causes some. The model runs locally "
             "because there is nowhere else for it to run."),
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
        p("Every subsystem, how it works, and where it turned out to be more "
          "interesting than expected. The habit in this project is to write down the "
          "measurement that overturned an assumption alongside the conclusion it "
          "produced, which means a good number of these pages are accounts of being "
          "wrong for a while. Those tend to be the useful ones: a transformer with "
          "its position encoding wired the wrong way round still writes fluent "
          "English, and nothing in the system objects."),
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
            ("Address spaces", "One"),
            ("Model", "Qwen3-0.6B, int8"),
            ("Source", "89 files, ~36,000 lines"),
            ("Target machine", "MSI Thin GF63 12UC"),
            ("Licence", "All rights reserved"),
        ]),
        p("GLaDOS is a single-address-space operating system with a transformer "
          "running inside the kernel. Everything the machine can do lives at the "
          "same privilege level as everything else, including inference, and the "
          "model reaches the rest of the system by calling it."),
        h2("The structural choice"),
        p("Conventional systems put a privilege boundary between kernel and user "
          "code, and cross it with syscalls. The boundary buys isolation and costs a "
          "trip through the trap handler on every crossing. GLaDOS does without one: "
          "everything runs at <a href=\"ring-0.html\">ring 0</a> in a single "
          "identity-mapped address space, so any code can reach any address, and the "
          "hardware will not object."),
        p("What that changes, concretely, is what a tool call is. In a typical agent "
          "the model emits text, something parses the text, a dispatcher looks up a "
          "handler by name, arguments get deserialised, and the call crosses a "
          "process boundary to reach code that may itself be a network hop away. "
          "Here the model selects a function under "
          "<a href=\"constrained-decoding.html\">a grammar compiled from the live "
          "applet table</a> and the kernel calls it. Nothing is serialised and "
          "nothing is parsed, because the decode was structured: the sampler returns "
          "the index of the applet that was chosen."),
        p("The price is exactly what you would expect. A bug anywhere can corrupt "
          "anything, there is no adversary model, and the system does not pretend "
          "otherwise. What the design protects against is mistakes, not malice, and "
          "the one place that distinction is treated as load-bearing is the NVMe "
          "driver, which refuses to write until something explicitly unlocks it."),
        h2("Boot"),
        p("There is no bootloader. UEFI hands over a machine already in long mode, "
          "already at ring 0, already identity-mapped, with a working filesystem "
          "driver and a framebuffer, so the UEFI application <em>is</em> the kernel. "
          "That deletes ELF loading, relocation and the handoff ABI in one go. See "
          "<a href=\"uefi-kernel.html\">UEFI as the kernel</a>."),
        p("The model weights, the tokenizer and the root certificate bundle are all "
          "read before <code>ExitBootServices</code>, because that call is the last "
          "moment a filesystem exists. Everything after it runs on page tables this "
          "kernel built, with the firmware's services gone and unrecoverable, "
          "including the perfectly good USB stack the firmware was using ninety "
          "milliseconds earlier."),
        p("Boot then puts up a splash screen, which is not decoration. Reading "
          "570 MB of weights off a USB stick and running eleven sets of crypto test "
          "vectors takes long enough that a machine showing nothing is "
          "indistinguishable from a machine that has hung, and the GF63 has no "
          "serial port to ask. The console keeps writing to a shadow grid in RAM "
          "the whole time, and the splash hands the screen back by repainting the "
          "entire log, so nothing is actually hidden."),
        h2("Subsystems"),
        ul("<a href=\"llm-in-kernel.html\">Inference</a>: a transformer forward "
           "pass with no allocator underneath it and no libm to call, so "
           "<code>expf</code> and <code>sqrtf</code> are in the tree too.",
           "<a href=\"network-stack.html\">Networking</a>: ARP, IPv4, ICMP, UDP and "
           "TCP, with DHCP, DNS and <a href=\"tls.html\">TLS 1.3</a> above them.",
           "<a href=\"storage.html\">Storage</a>: an NVMe driver under a "
           "content-addressed object store where copying is free and a snapshot is "
           "one hash.",
           "<a href=\"usb-xhci.html\">USB</a>: xHCI rings, device enumeration, and a "
           "CDC Ethernet driver that carries real traffic.",
           "<a href=\"gui.html\">Graphics</a>: a Windows 3.1 desktop with a window "
           "manager, drawn pixel by pixel into the GOP framebuffer.",
           "A shell, an interpreted language with kernel builtins, a text editor, "
           "and a small browser that renders HTML and CSS over the kernel's own TLS."),
        h2("Results worth knowing about"),
        p("Two of the most useful things this project has established are negative, "
          "and both are still in the tree with the code that produced them. Training "
          "an adapter head on top of the model made held-out accuracy worse with every epoch (30% untrained, 10% after two epochs, 0% "
          "after eight), because forty examples spread across twenty-one classes gives gradient "
          "descent nothing to do except memorise. And a Product-of-Experts council "
          "combining three classifiers scored 76.9% where the best single classifier "
          "scored 77.8%, so the ensemble was quietly worse than one of its members."),
        p("The council stayed anyway, for a reason that only showed up in the same "
          "measurement: when all three of its cores agree they are right 90.3% of "
          "the time, and when they split, 50%. Their agreement is a usable "
          "confidence signal even though their combined vote is not a better answer. "
          "A router that knows when it is guessing can escalate or ask; one that is "
          "silently 78% accurate cannot."),
        ("seealso", ["rust-os", "ring-0", "templeos", "routing", "credits"]),
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
        p("<strong>Aperture Science</strong> is the fictional research company in "
          "Valve's <em>Portal</em>, and GLaDOS is the artificial intelligence that "
          "runs its facility, administers its tests, and is extremely encouraging "
          "about how well you are doing right up until the incinerator. This "
          "operating system takes the name and the look and points them at a real "
          "machine."),
        note(DISCLAIMER),
        h2("What the aesthetic actually is"),
        p("Reduced to things a renderer can act on, the Aperture look is four "
          "decisions:"),
        ul("Amber on black. The accent is <code>#F28C1E</code>, which is roughly "
           "the colour of an amber CRT phosphor and of every warning label in the "
           "facility. It reads as instrumentation.",
           "Monospace everywhere, which is less a font choice than an institutional "
           "one: the typography of test protocols, lab equipment and things printed "
           "by a machine that has no opinion about kerning.",
           "Cheerful text about dangerous things. The Enrichment Centre voice is "
           "procedural and encouraging, and describes hazards in the register of a "
           "workplace safety video.",
           "Industrial chrome: bevelled panels, hard edges, visible structure, "
           "nothing rounded. Which lands, very conveniently, about two pixels away "
           "from Windows 3.1."),
        h2("Where it shows up"),
        p("The boot screen draws the aperture mark as vector geometry, because the "
          "kernel has no image decoder and writing a PNG decoder in order to display "
          "a logo would be a strange allocation of effort. It is arcs and lines "
          "computed at startup from the same geometry the desktop wallpaper uses. "
          "The window chrome is <a href=\"gui.html\">Windows 3.1 construction</a>: "
          "two-pixel bevels with the light at the top-left, over a "
          "<code>#C0C0C0</code> face, with the title bar filling amber on focus."),
        p("That combination is not a coincidence, and the source says so: a raised "
          "panel with a sunken trough, four colours and one bitmap font is "
          "simultaneously the Windows 3.x dialog vocabulary and the cheapest thing a "
          "framebuffer can draw. No blending, no antialiasing, no gradients, no "
          "scaling. The aesthetic and the constraint turned out to be the same "
          "decision, which is why it still looks right now that there is a window "
          "manager behind it."),
        p("The overall effect is a machine that looks like it was requisitioned in "
          "1994 by a research facility with a generous budget and very little "
          "oversight."),
        h2("The voice, and where it is load-bearing"),
        p("The Enrichment Centre register is not only decoration, because a system "
          "that reports failures cheerfully has to actually report them. The boot log prints a pass or fail line for every self-test, "
          "including the ones that pass. The page-fault handler decodes the error code "
          "into English and prints the faulting address instead of halting. The "
          "storage layer announces that it found no unclaimed space and is therefore "
          "doing nothing, which is a non-event that a quieter system would omit and "
          "that is exactly the thing someone needs told."),
        p("There is a real design principle hiding in the joke: describe the hazard "
          "in the same tone as everything else, and do not hide the outcome you "
          "would rather not have had. The wiki is written the same way, which is why "
          "several of these pages are accounts of a measurement that went against "
          "the assumption behind it."),
        h2("Taking the joke seriously"),
        p("Naming an operating system after a fictional AI that kills its research "
          "staff is a joke about AI safety, and it is meant as one. It also sets a "
          "standard the system then has to meet, and the design is conspicuously "
          "unambitious about autonomy as a result. There is no agent loop: the model "
          "proposes and a keystroke adopts. Read-only mode works by deleting the "
          "mutating applets from the grammar before sampling begins, so a forbidden "
          "action is not rejected after the fact: there is no sequence of sampling "
          "outcomes that produces one. Self-modification, where it exists at all, is "
          "gated on measurement against a held-out split that is read once."),
        p("None of the engineering is a joke. The "
          "<a href=\"network-stack.html\">TCP/IP stack</a>, the "
          "<a href=\"tls.html\">TLS 1.3 client</a>, the "
          "<a href=\"usb-xhci.html\">xHCI driver</a> and the "
          "<a href=\"llm-in-kernel.html\">transformer</a> are all hand-written and "
          "all tested at every boot. The facility is fictional. The certificate "
          "chain validation is not."),
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
        p("GLaDOS is an operating system in the literal sense: it boots on bare "
          "metal, builds its own page tables, drives its own hardware, and has no "
          "Linux, no BSD and no kernel from anyone else underneath it. The Portal "
          "connection is in the name, in the "
          "<a href=\"aperture-science.html\">visual language</a>, and in the voice "
          "the resident model answers in. Underneath that, it is a kernel, and the "
          "table further down is the honest inventory of what that means."),
        p("The distinction matters mostly because the phrase gets used for theme "
          "packs, and someone arriving here from a search deserves to know within a "
          "paragraph which kind of thing they have found. This one requires you to "
          "turn Secure Boot off."),
        h2("The Portal-shaped parts"),
        ul("It is named for the facility AI, and the model answers in that "
           "register.",
           "The <a href=\"aperture-science.html\">Aperture palette</a> throughout: "
           "amber on black, monospace type, and the mark drawn as vector geometry "
           "at boot because there is no image decoder to draw it any other way.",
           "A tone that treats catastrophic failure as a procedural note. The fault "
           "handler reports a page fault by decoding the error code into English and "
           "printing the faulting address, which is the Enrichment Centre approach "
           "to hazards and also just good diagnostics.",
           "One design joke that is also the safety argument: the AI it is named "
           "after went autonomous, and this one categorically has not."),
        h2("The operating-system-shaped parts"),
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
        p("The only code in the tree that was not written for this project is Rust's "
          "<code>core</code> library, which the compiler supplies, and "
          "<a href=\"usb-wifi-driver.html\">509 hardware initialisation "
          "constants</a> transcribed from Linux's <code>rtl8xxxu</code> driver "
          "because no datasheet for that chip exists and there is nowhere else to "
          "get them. That is one file, it says so at the top, and it is listed on "
          "<a href=\"../credits.html\">the credits page</a>."),
        h2("What using it is like"),
        p("The action surface is shaped like busybox: one flat table of small "
          "applets, each with a name, an argument spec, a line of help, and a flag "
          "saying whether it can change anything. That shape was borrowed because it "
          "is the right shape for something that is not only typed at. The table renders directly "
          "into a prompt, and the mutation flag is the leash, so "
          "read-only applets can be handed to the model long before the rest."),
        p("Because the namespace underneath is content-addressed, the applet list "
          "reads slightly strangely to anyone expecting POSIX. <code>cp</code> is "
          "constant time at any size. <code>same</code> compares two whole subtrees "
          "in a single step. <code>rm</code> detaches a name and the content "
          "survives. <code>diff</code> compares two snapshots while skipping every "
          "subtree whose hashes already match, and <code>snap</code> commits the "
          "working tree for the price of one hash. None of that was designed in as "
          "a feature list; it is what content addressing already implies."),
        p("Around that sits a shell with about eighty commands. <code>gen</code> and "
          "<code>ask</code> drive the model, <code>route</code> and <code>act</code> "
          "are the two tool-routing paths, <code>teach</code> adds an example to the "
          "corpus and <code>fit</code> refits the probe from it. "
          "<code>ping</code>, <code>dhcp</code>, <code>dns</code> and "
          "<code>https</code> exercise the network; <code>pci</code>, "
          "<code>usb</code>, <code>nvme</code> and <code>cpu</code> report on "
          "hardware; <code>vi</code> is a text editor and <code>enternet</code> is a "
          "browser. There is also a small interpreted language, so a sequence of "
          "those can be a script."),
        h2("Where scepticism is warranted"),
        p("The parts worth being sceptical about are the ones where writing "
          "everything yourself is a liability. Hand-written "
          "cryptography is the clearest case: a mistake there produces output that "
          "works perfectly and is not secure, which is why every primitive is "
          "checked against published RFC vectors on every boot and why "
          "<a href=\"tls.html\">the TLS page</a> spends most of its length on what "
          "is still wrong with it."),
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
        fig("rust-logo",
            "Rust targets <code>x86_64-unknown-uefi</code> directly, so the "
            "compiler emits a PE binary the firmware will load. There is no linker "
            "script and no custom target JSON in this project.",
            "The Rust programming language logo"),
        p("GLaDOS targets <code>x86_64-unknown-uefi</code> with <code>no_std</code> "
          "and no runtime. The compiler emits a PE executable that UEFI loads "
          "directly, which means there is no linker script, no custom target "
          "specification and no assembler stub to get into long mode. What follows "
          "is what the rest of that arrangement costs and pays."),
        h2("What no_std takes away"),
        p("Dropping the standard library removes more than most people expect. "
          "There is no heap until you write an allocator, and therefore no "
          "<code>Vec</code>, <code>String</code> or <code>Box</code> until "
          "<code>alloc</code> is wired to it. No files, no threads, no "
          "<code>println!</code>, no <code>std::time</code>. Panicking needs a "
          "handler you supply, and so does the out-of-memory path."),
        p("The one that catches people building anything numeric: there are no "
          "floating-point maths functions. <code>exp</code>, <code>sqrt</code>, "
          "<code>tanh</code> and friends live in libm, which is a C library, which "
          "is not present. For an operating system with a transformer in it this is not a footnote. Softmax is defined in terms of "
          "<code>exp</code> and RMSNorm needs a reciprocal square root, so "
          "the "
          "<a href=\"llm-in-kernel.html\">inference code</a> carries its own "
          "implementations and the boot self-test checks them against known values."),
        h2("What ownership buys at ring 0"),
        p("Rust's ownership model does not stop being useful at the hardware "
          "boundary. A driver that hands out a buffer it still holds, a descriptor "
          "ring the allocator might free while the controller is reading it, a "
          "device structure aliased by two code paths that both think they own it: "
          "these are the classic kernel bugs, they are miserable to find because the "
          "symptom appears in a subsystem that did nothing wrong, and here most of "
          "them are compile errors."),
        p("The caveat is real, though. Everything that actually touches hardware is "
          "inside <code>unsafe</code>, and inside <code>unsafe</code> Rust does not "
          "check that a physical address is mapped, that a DMA buffer is where the "
          "device was told it is, that a volatile write reached a register rather "
          "than a cache line, or that a struct's field order matches the ABI it is "
          "standing in for. It narrows the blast radius without removing it, and the places where it narrows nothing are exactly the places that have cost "
          "the most debugging."),
        h2("Traps that cost real time"),
        ul("<code>extern \"C\"</code> on the UEFI target means Microsoft x64, not "
           "System V. Arguments arrive in <code>rcx, rdx, r8, r9</code> instead of "
           "<code>rdi, rsi, rdx, rcx</code>, and there are 32 bytes of shadow space. "
           "The context switch has to say <code>extern \"sysv64\"</code> explicitly. "
           "Getting this wrong looks exactly like memory corruption.",
           "A guarded match arm placed after the arms it guards is unreachable. The "
           "compiler said so, in a warning, for several commits, while the bug it "
           "described was being looked for elsewhere.",
           "Interior mutability is not a lock. The type used here is called "
           "<code>Racy</code> for the specific purpose of being greppable: on the "
           "day a second core exists, one search finds every place that quietly "
           "assumed there was only one.",
           "Struct field order in the UEFI bindings is the ABI. Every entry in "
           "<code>BootServices</code> has to be declared, in the specification's "
           "order, even the ones that are never called, or the function pointers "
           "that are called land at the wrong offsets and the machine jumps into "
           "the middle of an unrelated firmware routine. That failure presents as a "
           "spontaneous reboot with no diagnostic at all."),
        h2("The heap"),
        p("The kernel heap is one physically contiguous allocation taken from the "
          "UEFI memory map. It has to be contiguous because the identity map makes "
          "a heap pointer usable as a DMA address, which is what lets the NVMe and "
          "network drivers skip bounce buffers entirely."),
        p("Its size comes from a ladder that steps down until one rung fits, and it is a ladder for a specific reason: the largest "
          "contiguous region a machine can offer is not its free memory. A map can "
          "report 1.9 GB free spread across 84 separate regions and not contain "
          "320 MB in one piece. A fixed size that the GF63's memory map cannot "
          "satisfy is an unbootable system, on the one machine that cannot be "
          "debugged from here, so boot prints the size it got and says when it had "
          "to come down a rung."),
        p("The allocator underneath is an address-sorted free list with coalescing, "
          "resting on one invariant: every block address and every block size is a "
          "multiple of 16 bytes. That is what keeps it simple enough to trust. "
          "<code>dealloc</code> is handed the original <code>Layout</code> rather "
          "than a header the allocator wrote, so if <code>alloc</code> ever returned "
          "more than the rounded request, <code>dealloc</code> would give back less "
          "than was taken and leak the difference on every single allocation. "
          "Rounding both sides the same way makes them exact inverses."),
        note("A good bug lived in that ladder. It never actually degraded: the "
             "probing allocator advanced its cursor destructively, so once the "
             "largest rung failed, every smaller rung failed too, and the fallback "
             "path had never once run. The fix was a non-consuming "
             "<code>largest_span()</code> that looks without taking. Rewinding the "
             "cursor instead would have handed out the page tables' own frames a "
             "second time, which is a considerably more interesting failure."),
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
        fig("priv-rings",
            "x86 defines four privilege levels. Almost every operating system uses "
            "two of them, and this one uses the innermost.",
            "Diagram of the four x86 protection rings, ring 0 innermost"),
        p("x86 has offered four privilege levels since the 286, and in practice "
          "systems use two of them: ring 0 for the kernel, ring 3 for everything "
          "else. Rings 1 and 2 have been decorative for decades, partly because "
          "paging only distinguishes supervisor from user and cannot tell them "
          "apart. The boundary between 0 and 3 is what makes a process crash "
          "instead of a machine crash."),
        p("GLaDOS runs entirely in ring 0. There is no ring 3, no <code>syscall</code> "
          "instruction, no per-program address space, and no TSS entry for a ring-0 "
          "stack to switch to, because there is never anything to switch from."),
        h2("What the boundary costs"),
        p("Crossing it is not free. A syscall saves registers, switches to the "
          "kernel stack, validates every pointer that came across because userspace "
          "may be lying about them, and reverses all of it on the way back. On "
          "systems with kernel page-table isolation it also swaps page tables twice "
          "and takes the TLB misses that follow. For a call that reads a megabyte "
          "off a disk this is rounding error. For a great many very small calls it "
          "becomes the dominant cost."),
        p("A language model driving an operating system makes a great many very "
          "small calls. A tool invocation, a namespace lookup, a read of the applet "
          "table, the sampler asking which tokens the grammar still permits. In a "
          "conventional design every one of those is a boundary crossing, and the "
          "grammar question happens once per token."),
        p("Removing the boundary also removes the reason tool calls are text. When "
          "the model and the applet table are in the same address space, the sampler "
          "can be constrained by the real table and return an index into it. Nothing "
          "is rendered into a prompt and nothing is parsed back out, which falls out of the privilege decision and was not designed in "
          "separately."),
        h2("What removing it costs"),
        p("Precisely the things the boundary was buying, and it is worth being "
          "specific:"),
        ul("A null dereference is not a segfault in one process. It is a page fault "
           "in the only process, and if the fault handler cannot recover it is a "
           "triple fault and a reboot.",
           "A buffer overrun can reach the page tables, the interrupt descriptor "
           "table, or the model's weights, and nothing in the hardware will stop it.",
           "There is no <code>kill</code>. A runaway loop owns the machine until the "
           "timer interrupt preempts it, and if it holds interrupts off, not even "
           "then.",
           "A script that exhausts the heap exhausts <em>the</em> heap, and the "
           "allocator has nowhere to fail back to."),
        p("Some of that is mitigated. Virtual page zero is deliberately left unmapped, which costs one extra page table "
          "for the first 2 MiB, so a null dereference produces a hard fault with a legible "
          "message from the page-fault reporter instead of silently reading whatever "
          "the firmware happened to leave at address zero. Scripts run under "
          "capabilities, and a tame mode refuses raw port access by name before the "
          "arguments are even looked at. The NVMe driver keeps writes locked unless "
          "something explicitly unlocks them."),
        p("None of that is isolation. It is one interpreter in one address space, "
          "and it stops mistakes, not intent."),
        h2("Why that is a defensible trade here"),
        p("Because the threat model is small and stated. This is a research kernel "
          "for one laptop, booted from a USB stick, running one model that is not "
          "adversarial and is in any case too small to be much of anything. The "
          "failure mode being designed against is a bug, and the tool against bugs is the self-test output at boot."),
        p("The alternative, claiming a security boundary and enforcing it "
          "incompletely, is worse than having none, because it invites the system "
          "to be used as though the boundary were real. So the documentation says "
          "there is no isolation, the shell says it, and "
          "<a href=\"tls.html\">the TLS page</a> says the equivalent thing about "
          "certificate validation."),
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
        fig("uefi-logo",
            "UEFI replaced the BIOS with a specification that includes a "
            "filesystem driver, a memory allocator and a framebuffer, all available "
            "before your code starts.",
            "The UEFI logo"),
        p("A kernel is normally loaded by a bootloader (GRUB, systemd-boot, limine) which reads "
          "an ELF file off a filesystem, arranges long mode, gathers a "
          "memory map and jumps to an entry point under an agreed calling "
          "convention. Two codebases, one handoff contract between them, and a "
          "great deal of setup code."),
        p("GLaDOS skips the entire arrangement, because UEFI has already done the "
          "work. Firmware hands a UEFI application a machine in 64-bit long mode, "
          "at ring 0, with an identity-mapped address space, a working allocator, a "
          "filesystem driver, a console and a linear framebuffer. An application "
          "that starts in that state is already running under the conditions a "
          "kernel spends its first thousand lines establishing, so this one just "
          "keeps going."),
        h2("What that deletes"),
        ul("No ELF loader, no relocation processing, no linker script.",
           "No handoff ABI to define, document and then get subtly wrong.",
           "No second codebase living on its own release cycle.",
           "No multiboot header, no protected-mode stub, no A20 gate, and no "
           "real-mode assembly anywhere in the tree."),
        p("The bindings to the firmware are hand-written, not taken from "
          "<code>uefi-rs</code>, which is a decision with a cost attached. Field "
          "order and padding in those structs <em>are</em> the ABI: every entry in "
          "the <code>BootServices</code> table has to be declared, in the "
          "specification's order, including all the ones that are never called, or "
          "the handful that are called sit at the wrong offsets. The machine then "
          "jumps into the middle of an unrelated firmware routine, which presents as "
          "a spontaneous reboot with nothing printed."),
        h2("The one-way door"),
        p("Boot services, meaning the filesystem access, the allocator, "
          "the console, the device protocols and the firmware's own USB and "
          "disk drivers, exist only until <code>ExitBootServices</code>. After that call they are gone "
          "permanently, every pointer into them is dangling, and there is no way "
          "back. The firmware does not offer a re-entry."),
        p("So everything the kernel will ever need from disk has to be read before "
          "that moment: the 570 MB of model weights, the tokenizer, and the root "
          "certificate bundle. They go into a pool allocated as "
          "<code>LoaderData</code>, which survives the transition, and are then referenced in place for the rest of the machine's life. "
          "Copying half a gigabyte onto the heap to gain nothing would be an "
          "unusual way to spend the memory."),
        note("This is also why the firmware's perfectly serviceable USB stack gets "
             "thrown away and <a href=\"usb-xhci.html\">written again from "
             "scratch</a>. It lives in boot services. Being the kernel and not a guest of "
             "the firmware is the whole point, and this is the bill for it."),
        h2("A memory-map trap worth knowing"),
        p("Do not take the maximum over every UEFI memory descriptor to decide how "
          "much address space to map. OVMF describes reserved regions out to 1 TiB, "
          "and using that as a limit needs more than one page-directory-pointer "
          "table can cover. The identity map then fails to install, silently, and "
          "the firmware's own page tables stay active instead."),
        p("The reason this was expensive and not merely wrong: the "
          "firmware's "
          "tables map page zero. So the null-dereference self-test, which exists specifically to "
          "prove that page zero is unmapped, passed without faulting, because the mapping it was testing for had never been "
          "installed. A test that passes for the exact reason it should fail is the "
          "worst possible outcome, and it is the reason boot now prints the map "
          "limit it chose."),
        h2("What the kernel builds for itself"),
        p("Once the firmware is gone, the machine runs on page tables this kernel "
          "wrote: a straight identity map, physical address equal to virtual "
          "address, covering RAM out to 4 GiB, with one deliberate exception. "
          "Virtual page zero is left unmapped. That costs one extra page table to "
          "describe the first 2 MiB at 4 KiB granularity instead of one large page, "
          "and it buys a hard fault on every null dereference, which, given "
          "the page-fault reporter, turns the most common bug in the kernel into a legible message "
          "instead of a silent read of whatever the firmware happened to "
          "leave lying at address zero."),
        p("Non-RAM pages are mapped uncacheable, which is not optional: a doorbell "
          "write to an NVMe or xHCI controller that sits in a write-back cache line "
          "never reaches the device, and the resulting symptom is a controller that "
          "accepts commands and never completes them."),
        p("Then a GDT, an IDT with the fault handlers installed, the local APIC "
          "timer at 100 Hz for preemption, and a PS/2 keyboard, in that order. The "
          "frame allocator underneath all of it is a bump allocator over the UEFI "
          "memory map that never frees, which is exactly right for what it is used for: the page tables "
          "and the initial heap both live for the lifetime of "
          "the machine."),
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
        fig("transformer-arch",
            "The decoder-only stack this implements is the right-hand half of the "
            "original Transformer: embedding, then repeated attention and "
            "feed-forward blocks, then a projection back to vocabulary logits.",
            "The Transformer model architecture diagram from Attention Is All You "
            "Need",
            wide=True),
        p("The model is a decoder-only transformer, Qwen3-0.6B by default, "
          "quantised to int8 and executed inside the kernel's address space. It is "
          "not a service and not a process. The forward pass is a function that the "
          "shell, the router or the fault handler could all call directly, and the "
          "same page tables are in effect throughout."),
        h2("What is missing at ring 0"),
        p("Inference code normally assumes an enormous amount of infrastructure: an "
          "allocator, threads, a BLAS library, libm, and an operating system to "
          "provide all four. None of it is here, so each one had to be replaced or "
          "designed around."),
        ul("No libm. Softmax is defined in terms of <code>exp</code>, RMSNorm needs "
           "a square root, GELU needs <code>tanh</code>, and none of those exist in "
           "<code>core</code>. They are written out, and the boot self-test checks "
           "them against known values because a subtly wrong <code>expf</code> "
           "produces a subtly wrong distribution and no error.",
           "No BLAS. The matrix-vector product is essentially the entire cost of "
           "inference, so it is written directly with an AVX2 and FMA path, and "
           "falls back to scalar when the CPU does not advertise them.",
           "No memory to waste. The weights stay in the LoaderData pool they were "
           "read into before <code>ExitBootServices</code> and are referenced in "
           "place. Copying 570 MB onto a heap that is itself only a few hundred "
           "megabytes is not an option even if it were a good idea.",
           "No threads. One core, with cooperative yields inside the generation "
           "loop so the shell and the clock keep running while a token is being "
           "produced."),
        p("There is a subtler consequence of running on one core with no operating "
          "system underneath: extended CPU state has to be saved on context switch "
          "by this kernel, or the YMM registers holding a partly-computed dot "
          "product get clobbered by whatever the clock task does next. There is a "
          "guard for exactly this, which parks a known pattern in the vector "
          "registers across a spin and checks it survived. With extended state "
          "saved it never fires; without it, the error count starts climbing the "
          "moment two tasks touch floating point."),
        h2("Where the time goes"),
        p("Generation is bound by memory bandwidth, not arithmetic. "
          "Producing "
          "one token requires reading essentially every weight once, so the number "
          "that predicts throughput is the size of the model in bytes, not its FLOP "
          "count. That is why the 570 MB Qwen3 checkpoint costs about 4.4 times as long per token as the 135 MB SmolLM2 one, almost exactly the "
          "size ratio, "
          "which is the signature of a bandwidth-bound workload."),
        p("Around 155 MB of that 570 is the output classifier: one matrix multiply "
          "against the full vocabulary, performed once per token, to produce logits "
          "that are then mostly thrown away. When "
          "<a href=\"constrained-decoding.html\">constrained decoding</a> is active "
          "the grammar often admits a few dozen tokens out of 150,000, and logits "
          "for the rest are computed and discarded. Restricting that final product "
          "to the reachable set is therefore the single largest available "
          "optimisation, and it is available precisely because the sampler and the "
          "grammar are in the same address space."),
        h2("A feature gate that tested the wrong feature"),
        p("The AVX2 kernel was gated on <code>avx_enabled &amp;&amp; fma</code> and "
          "never on <code>avx2</code> itself. The two are separate CPUID bits and "
          "neither implies the other. So the fast path ran on hardware that had FMA "
          "without AVX2, where it produced wrong numbers, and refused to run on "
          "hardware that had AVX2 without reporting FMA the way the check expected."),
        p("The rule that came out of it is worth stating plainly: a feature gate "
          "must test the feature the code uses, not a feature that usually travels "
          "with it. Numerical code makes this failure especially unpleasant, because "
          "wrong-but-finite output looks like output."),
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
        fig("attention-heads",
            "Multi-head attention runs several attention operations in parallel and "
            "concatenates them. How wide each of those heads is turns out not to be "
            "something you can safely infer.",
            "Diagram of multi-head attention with parallel scaled dot-product "
            "attention blocks",
            wide=True),
        p("Qwen3-0.6B is the default checkpoint: 28 layers, a 1024-dimensional "
          "residual stream, 16 query heads against 8 key/value heads in "
          "grouped-query attention, and int8 quantisation bringing it to roughly "
          "570 MB on disk. Loading it revealed two ways in which it differs from a "
          "Llama, and the interesting property of both is that getting them wrong "
          "raises nothing."),
        h2("Head width is stated, not derived"),
        p("Almost every implementation computes the head dimension as "
          "<code>dim / n_heads</code>, because for Llama that is the definition. For "
          "Qwen3 that arithmetic gives 1024/16 = 64, and the correct answer is 128. "
          "The architecture states head width as an independent hyperparameter, "
          "which makes the query projection 2048 wide against a residual stream of "
          "1024. The attention path is deliberately wider than the thing it reads "
          "from, and narrows again on the way out."),
        p("Derive it instead and every attention tensor gets reshaped along the "
          "wrong axis. The shapes remain self-consistent, which is the trap, since 64 "
          "divides 1024 perfectly well, so nothing asserts, nothing overflows, "
          "and the model loads and runs and produces confident nonsense. The source "
          "comment on that field names it as the first thing to suspect when a "
          "converted model is fluent and wrong."),
        h2("QK-Norm"),
        p("Qwen3 also applies RMSNorm to each head's queries and keys, per head, "
          "before rotary embeddings go on. Omitting the step leaves the network numerically well-behaved, since the vectors "
          "come out only a little longer than they should be and softmax "
          "absorbs most of the difference, and it will keep producing fluent "
          "English. The text is coming from a network being fed activations "
          "at a scale it never saw in training, which is a model hallucinating for a mechanical reason, and the "
          "mechanical kind does not go away with a better prompt."),
        h2("How these get caught"),
        p("Not by exceptions, because there are none to catch. Three things are "
          "doing the work instead."),
        p("First, the checkpoint converter writes both facts into the file header, "
          "so head width and the QK-norm flag travel with the weights and "
          "are never guessed at load time. That is what version 3 of the "
          "<code>GLADOSM3</code> format added; version 2 files still load, and their "
          "defaults are exactly the Llama ones, which is correct for every "
          "checkpoint that predates the problem."),
        p("Second, there is a NumPy reference implementation that reads the "
          "<em>converted</em> file rather than the original safetensors. This "
          "matters more than it sounds: a converter bug shows up in the reference "
          "too, so a mismatch between reference and kernel isolates the fault to the "
          "Rust side, and agreement between them plus wrong output isolates it to "
          "the converter."),
        p("Third, and cheapest: read what comes out. An instruction-tuned model with "
          "a correctly wired attention path answers \"what is the capital of "
          "France\" with Paris. One with a subtly scrambled one writes a fluent "
          "paragraph that never quite arrives at the fact. Having a known answer to "
          "check against is worth a surprising amount when there is no error to "
          "catch."),
        h2("Thinking tokens"),
        p("Left to itself, Qwen3 reasons at length inside a "
          "<code>&lt;think&gt;</code> block before answering. That is the model "
          "working exactly as designed and completely useless at a 64-token budget, "
          "where the entire generation is spent deliberating and the answer never "
          "arrives. So <code>ask</code> closes the block itself unless given "
          "<code>-t</code>."),
        p("Whether a checkpoint has thinking tokens at all is decided by asking the "
          "tokenizer whether it knows <code>&lt;think&gt;</code> as a single token. "
          "That is a property of the vocabulary, which is a fact, as opposed to "
          "pattern-matching on the model's filename, which is a guess that fails the "
          "first time someone renames a file."),
        h2("Getting it onto the machine"),
        p("The checkpoint arrives as safetensors and is flattened by a converter "
          "into a layout the kernel indexes by arithmetic: a header followed by "
          "tensors in a fixed order, so loading is a matter of "
          "computing offsets against a buffer that was read before "
          "<code>ExitBootServices</code>. Nothing is rearranged at boot, because "
          "rearranging 570 MB at boot would require somewhere to put it."),
        p("The context window is a converter argument, and the <a href=\"kv-cache.html\">KV cache</a> is what bounds it, "
          "not the model. Qwen3 at "
          "512 tokens costs 112 MiB of kernel heap, and the converter prints that "
          "figure at conversion time so the decision is made where the trade-off is "
          "visible, instead of being discovered as an allocation failure on a laptop "
          "with no debugger attached."),
        p("Int8 quantisation is per-block with an f32 scale, which matters for the "
          "same reason it matters in the KV cache: transformer weights have "
          "outliers, and one scale across an entire tensor spends most of its "
          "dynamic range representing a handful of large values badly while "
          "flattening everything else."),
        h2("Running it at all"),
        p("Qwen3-0.6B only runs on real hardware. QEMU's built-in FAT support caps the whole emulated disk at 516 MB. (<code>fat:32:</code> "
          "raises that in principle, but QEMU describes its FAT32 as "
          "untested and the firmware cannot read the directory it produces.) "
          "So QEMU work uses the SmolLM2 "
          "checkpoint and the real model is only exercised on the GF63, or "
          "numerically against the reference implementation."),
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
        fig("rope-diagram",
            "RoPE rotates pairs of dimensions by an angle proportional to position. "
            "Each pair turns at its own frequency, so the vector encodes position "
            "the way the hands of a clock encode time.",
            "Diagram of rotary position embedding applied to a query vector",
            wide=True),
        p("Rotary Position Embedding encodes a token's position by rotating pairs of "
          "dimensions in its query and key vectors through an angle proportional to "
          "that position. Each pair gets its own frequency, geometrically spaced, so "
          "the fast pairs distinguish nearby positions and the slow ones carry "
          "long-range order."),
        p("The reason this is elegant and not merely clever: a dot "
          "product "
          "between two rotated vectors depends only on the <em>difference</em> of "
          "their angles. Rotate the query for position 400 and the key for position "
          "395 and the score you get is the score for a gap of five, wherever in the "
          "sequence that gap happens to sit. Relative position falls out of absolute "
          "rotation for free, with no extra parameters and no bias matrix."),
        h2("Two conventions, one right answer per checkpoint"),
        p("Rotating a pair requires deciding which dimensions constitute a pair, and "
          "there are two answers in circulation:"),
        ul("Interleaved: pair dimension <code>2i</code> with <code>2i+1</code>, so "
           "the pairs are adjacent. This is what the original llama2.c does, and it "
           "is the obvious reading of the paper.",
           "rotate_half: pair dimension <code>i</code> with "
           "<code>i + head_dim/2</code>, so the first half of the vector pairs with "
           "the second. This is what HuggingFace's <code>rotate_half</code> does, "
           "and therefore what essentially every published checkpoint was trained "
           "with, because essentially every published checkpoint was trained through "
           "transformers."),
        p("Neither is more correct in the abstract. They are the same rotation "
          "applied to a different permutation of the axes, and a model trained under "
          "either works fine. What matters is agreeing with whatever the weights "
          "were fitted under, and the weights do not come with a note."),
        h2("Why getting it wrong produces no error"),
        p("This is the part worth internalising. Both conventions are "
          "norm-preserving rotations through the same set of angles. Pick the wrong "
          "one and there is no NaN, no overflow, no gradual drift, no assertion and "
          "no warning. Vector lengths are unchanged. Attention scores stay in the "
          "range softmax expects. Every intermediate tensor looks exactly as healthy "
          "as it did before."),
        p("What has actually happened is that the model is attending by a scrambled "
          "notion of distance: tokens that should be five apart score as though they "
          "were some other distance apart, consistently but wrongly. From outside, "
          "the result is a model that is a bit vague, loses the thread over long "
          "spans, and repeats itself, which is an exact description of a small "
          "model being small. There is no observation that separates the two except "
          "comparing against a reference or checking a fact."),
        h2("What it cost, measured"),
        p("This kernel used the interleaved convention for a long time, on the "
          "strength of having started from llama2.c, and nothing looked broken. "
          "Switching to rotate_half with the same checkpoint, prompt and seed:"),
        table(
            ["Convention", "Output for \"The capital of France\""],
            [["Interleaved (wrong)", "\"The capital of France.\" then blank lines"],
             ["rotate_half (correct)", "\"The capital of France is Paris. Paris is a "
              "city known for...\""]]),
        p("At token level, the highest-probability first tokens went from "
          "<code>'\\n'</code>, <code>'\\n\\n'</code>, <code>'ity'</code> and "
          "<code>' capital'</code> to <code>'The'</code>, <code>'Paris'</code> and "
          "<code>'France'</code>. The wrong version was not producing garbage. It "
          "was producing a model that had lost track of what the prompt was about "
          "and defaulted to whitespace."),
        p("One checkpoint genuinely wants the interleaved convention: an actual "
          "llama2.c model, trained by that code. So the choice is recorded per "
          "checkpoint in <code>Config::rope_interleaved</code> and written into the file header at conversion time, so the "
          "kernel never decides it globally. A global constant here is a bug waiting for the second model."),
        note("If you are debugging an implementation that produces plausible but "
             "slightly-off text, check the RoPE pairing before anything else. It is "
             "the highest-probability silent failure in a hand-written transformer, "
             "it survives every sanity check that looks at magnitudes, and it will "
             "not announce itself at any point."),
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
          "has already read, so that generating token 500 does not require "
          "recomputing the first 499. Past a few thousand tokens it is comfortably "
          "larger than the model that produced it, and on a machine whose entire "
          "heap is one contiguous allocation, its size is a design constraint rather "
          "than an implementation detail."),
        h2("The arithmetic"),
        p("For Qwen3-0.6B: 2 (keys and values) × 28 layers × 1024 kv-dimensions × "
          "4 bytes comes to 224 KiB per token. At the default 512-token context that "
          "is 112 MiB of kernel heap, which is why <code>convert.py</code> prints "
          "the figure when the context window is chosen. The window is bounded by "
          "cache size and not by the model, and finding that out as an allocation "
          "failure at boot is a worse way to learn it."),
        table(
            ["Scheme", "Per token", "At 32,768 tokens"],
            [["f32, one allocation", "224 KiB", "7.0 GiB contiguous. Unreachable"],
             ["int8, one allocation", "56 KiB", "1.75 GiB contiguous. Very unlikely"],
             ["int8, split per layer", "56 KiB", "28 × ~66 MiB, achievable"]]),
        p("Splitting per layer is what makes long context reachable at all. As one "
          "allocation the cache needs a single unbroken physical region, and the "
          "largest contiguous span a real memory map offers is nothing like its "
          "total free memory. Split per layer, the largest single request drops "
          "28-fold, and a fragmented map can satisfy 28 requests of 66 MiB when it "
          "cannot satisfy one of 1.75 GiB."),
        h2("Quantisation, and an asymmetry"),
        p("Entries are stored as int8 in blocks of 32, with one f32 scale per block, "
          "so the overhead is one scale per 32 values, not one per tensor. Block "
          "scaling matters here because attention activations have outliers, and a "
          "single scale across a whole tensor spends most of its range representing "
          "one large value badly."),
        p("The measured error is not symmetric between keys and values: keys carry "
          "roughly fifteen times the impact. The reason is structural. A key goes into a dot product whose result then passes through "
          "softmax, which is exponential, so a small perturbation in a score becomes "
          "a disproportionate change in attention weight, and because "
          "softmax normalises, that error redistributes across every other position too. A "
          "value is only ever averaged, and averaging is forgiving. If you have "
          "quantisation budget to spend unevenly, spend it on the keys."),
        h2("Attention sinks and a sliding window"),
        fig("attention-sinks",
            "Attention logits averaged over 256 sentences in Llama-2-7B. Beyond the "
            "first couple of layers, an enormous share of the mass lands on the "
            "first few positions no matter what those positions contain.",
            "Heatmap of average attention logits showing large values at the "
            "initial token positions",
            wide=True),
        p("Past the allocated window the cache runs as a ring: a handful of initial "
          "tokens are pinned in place as attention sinks and everything after them "
          "slides. The sinks look like an arbitrary hack and are not. Transformers "
          "place a large amount of attention mass on the first few positions "
          "regardless of what those tokens actually say. The mechanism "
          "appears to be that softmax must sum to one, so so a head with nothing it wants to "
          "attend to has to put its mass somewhere, and the beginning of the "
          "sequence is the one place every query can see."),
        p("Evict those positions and every head that was using them as a null option "
          "is forced to redistribute onto real tokens it does not want, which "
          "degrades output sharply and immediately. Keep four of them pinned and "
          "generation continues past the nominal context length instead of "
          "falling apart at the boundary. This is the StreamingLLM result, and it costs four "
          "cache slots."),
        fig("streaming-kv",
            "Four strategies for a cache that has run out of room. Recomputing the "
            "window is correct and quadratic; naive eviction is cheap and degrades "
            "badly; keeping a few initial tokens pinned gets most of the first for "
            "the price of the second.",
            "Diagram comparing dense attention, window attention, sliding window "
            "with recomputation, and StreamingLLM",
            wide=True),
        h2("Why not simply recompute"),
        p("The obvious alternative to any of this is to drop the cache and recompute "
          "attention over a sliding window each time. That is correct, and it is "
          "quadratic in the window length, which turns every token into a "
          "prefill. On a machine that is already bandwidth-bound and single-core, "
          "the cache is not an optimisation to be traded away; it is what "
          "makes generation possible at a usable rate at all."),
        p("The int8 encoding and the sink-plus-window scheme are therefore both "
          "answers to the same question: how to keep the cache, given that doing without one is not an "
          "option."),
        note("Memory stops being the binding constraint before speed does. Attention "
             "is linear in the number of live positions, so at 32k tokens it comes "
             "to roughly 3.8 GMAC per token on top of the model's own 0.6 GMAC, which is several seconds "
             "per token at full context. That is a perfectly good "
             "rate for reading one long document once, and an unpleasant one for a "
             "conversation."),
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
        p("A byte-pair encoding tokenizer has two halves. There is the merge table, "
          "which is the part everyone pictures: a ranked list of byte pairs, applied "
          "greedily until nothing more can be joined. And there is a pre-tokenizer "
          "regex, which chops the input into candidate pieces <em>before</em> any "
          "merging happens and forbids merges from crossing the boundaries it "
          "creates. The second half is easy to forget, because when it is right it "
          "is invisible."),
        p("The kernel's tokenizer reads a flat binary: a maximum token length, then "
          "one record per vocabulary entry holding a merge score, a length and that "
          "many raw bytes. The vocabulary size deliberately is not in the file. It comes "
          "from the model header, which is what binds a tokenizer to a "
          "particular checkpoint. Hand it a mismatched pair and it parses "
          "cheerfully and produces nonsense."),
        h2("The regexes differ, and it matters"),
        p("GPT-2's pattern and the cl100k pattern disagree in ways that read as "
          "cosmetic:"),
        ul("Under cl100k a word may be led by any non-alphanumeric character, so <code>(x</code> is one piece and not two.",
           "Digits come out one at a time, never in runs, so <code>2048</code> "
           "is four pieces.",
           "Punctuation swallows the newlines that follow it, which changes how "
           "every line ending in a full stop tokenises."),
        p("SmolLM2 trained with the GPT-2 pattern. Qwen3 spells out the cl100k one "
          "explicitly as a Split rule in its tokenizer configuration. Using the "
          "wrong one moved about 12% of tokens across this project's training corpus, "
          "including, unhelpfully, the ChatML control structure that the "
          "instruction tuning depends on, where <code>&lt;|im_start|&gt;</code> and "
          "its neighbours have to land exactly as trained."),
        h2("Twelve percent of nothing visible"),
        p("There is no error anywhere in that. The tokenizer emits ids, the ids are "
          "in range, the model consumes them and produces fluent text. What is "
          "happening underneath is that the model is being handed sequences whose "
          "segmentation it never saw in training, and transformers degrade gently under that without failing, since a word split two "
          "ways instead of one still carries most of its meaning through the embedding."),
        p("The fix is the same shape as the one for "
          "<a href=\"qwen3.html\">head width and QK-norm</a>: the checkpoint carries "
          "which pre-tokenizer it needs as a flag in its own header, so the fact "
          "travels with the weights instead of being inferred. After that change, "
          "divergence across 407 texts and 66,598 tokens was zero."),
        h2("Verify against the reference, always"),
        p("The converter has a <code>--verify</code> mode that reimplements the "
          "kernel's algorithm in Python and diffs it, token for token, against the "
          "reference <code>tokenizers</code> library. Running it is not optional, "
          "for the reason that runs through this whole section of the wiki: a "
          "tokenizer that is subtly wrong produces text that still looks like text, "
          "so the only thing that catches it is a comparison against something known "
          "to be right."),
        h2("What the kernel's implementation actually does"),
        p("Encoding runs greedy byte-pair merging over the pieces the pre-tokenizer "
          "produced. Start with the individual bytes, repeatedly find the adjacent "
          "pair whose merged form has the best score in the vocabulary, and join it, "
          "until no pair can be joined. The vocabulary is kept sorted by byte string "
          "so a merge candidate is found by binary search and not by "
          "scanning 150,000 entries per step, which is the difference between "
          "tokenising a prompt instantly and tokenising it noticeably."),
        p("Bytes the vocabulary cannot express fall back to single-byte pieces. "
          "SentencePiece with byte fallback places those 256 pieces immediately "
          "after the three control ids, so any unrepresentable byte becomes "
          "<code>byte + 3</code>. That offset is a constant in the source with a "
          "comment explaining where the three comes from, because it is exactly the "
          "kind of number that looks arbitrary six months later."),
        p("Three further per-checkpoint behaviours ride along as flags in "
          "the header, where the code would otherwise have assumed them: whether to prepend a dummy space, "
          "whether digits are split individually, and which pre-tokenizer regex "
          "applies. Every one of them is a property of how the model was trained, "
          "and every one of them fails silently if guessed."),
        h2("An earlier design, and why it was abandoned"),
        p("The model code was originally written around a byte-level vocabulary of "
          "256 entries, which removes the tokenizer completely: no merge table, and no "
          "vocabulary file to get onto a machine that has no filesystem driver yet. "
          "For bootstrapping something that had to run before storage existed, that "
          "was the right trade."),
        p("It cost roughly four times the context per unit of text, since every "
          "character is a token. Once the kernel could read files, a real vocabulary "
          "became a 6 KB read that buys back all of it, and the byte-level path was "
          "retired. The trade was correct when it was made and wrong six months "
          "later, which is the usual shape of a bootstrapping decision."),
        note("A related trap, and an expensive one. Sizing the vocabulary from the "
             "merge table is wrong for Qwen3, whose 293 special tokens live "
             "<em>above</em> it. Doing that indexed the vocabulary past its end and "
             "resolved the end-of-turn token to id 2, so generation had no way to "
             "stop and simply ran to the token limit every time. Size from the "
             "highest added token id instead."),
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
        fig("dfa",
            "A grammar over applet names compiles to a state machine of this shape. "
            "At every state the set of admissible next symbols is known, which is "
            "the whole trick.",
            "A small deterministic finite automaton with three states and labelled "
            "transitions"),
        p("The standard way to get a tool call out of a language model is to ask for "
          "one in the prompt, generate text, parse the result, and handle the case "
          "where the model named a tool that does not exist. That approach is not a mistake; it is the only option available "
          "when the model sits behind an API and the sole thing you control is the text going in and out. It is "
          "also why tool calling is usually said to need a large model: a good fraction of the capability goes on producing well-formed "
          "syntax when it could have gone on choosing correctly."),
        p("GLaDOS owns the sampler, so none of that applies."),
        h2("The mechanism"),
        p("At each decoding step, the set of tokens that could extend the current "
          "prefix into a valid name from the live applet table is computable. The "
          "sampler is handed exactly that set, and every other logit is removed before sampling, with nothing "
          "checked afterwards."),
        p("The consequence is categorical. No sequence of sampling "
          "outcomes produces an invalid applet name, at any temperature, "
          "however badly calibrated the model happens to be. A "
          "260K-parameter story model driven "
          "through this grammar will choose absurdly and will still never emit "
          "anything other than a real applet name, which is a useful property to "
          "have tested against, since it proves the mechanism holds independently of "
          "whatever is loaded."),
        p("What is left for the model to be wrong about is <em>which</em> applet to "
          "pick, which is the part that actually requires intelligence. Model size "
          "stops gating correctness of form and starts gating only correctness of "
          "choice. And because the decode was structured by construction, nothing "
          "needs parsing at the end: the cursor reports the index of the applet that "
          "was decoded."),
        h2("Read-only mode is the same mechanism"),
        p("When the system runs at a read-only trust level, the mutating applets are "
          "removed from the reachable set before sampling begins. The model is not "
          "asked to behave and then audited afterwards; the tokens that spell "
          "<code>rm</code> have no path through the grammar at all. Permission "
          "enforcement and output validity are the same piece of code, which means "
          "there is no second place for them to disagree."),
        p("That is also why a script's mutating-ness is derived from its code rather "
          "than declared in a header. A script that claimed to be read-only and then "
          "called a mutating applet would insert a mutating entry into the read-only "
          "grammar, and the guarantee would leak through the one door it was not "
          "watching. So the system walks the syntax tree instead, and treats any "
          "call whose target is not a literal string as mutating, because a computed "
          "name cannot be resolved ahead of time, and guessing would break the property "
          "that makes the whole thing worth having."),
        h2("How it is tested"),
        p("The grammar self-test runs at every boot. It performs 200 random decodes "
          "at maximum temperature and asserts that not one escaped the reachable "
          "set, checks that the read-only grammar excludes every mutating applet and "
          "retains every read-only one, and confirms that a decode always terminates on a real applet."),
        h2("What it does not solve"),
        p("Constrained decoding guarantees the shape of the output and says nothing "
          "whatsoever about its quality. The model can still pick <code>rm</code> "
          "when it should have picked <code>ls</code>, at full confidence, and the "
          "grammar will help it spell that mistake correctly. Everything on "
          "<a href=\"routing.html\">the routing page</a> exists because that is the "
          "remaining problem and it is a harder one."),
        p("There is also a cost worth naming. Restricting the sampler means "
          "computing which tokens are admissible at every step, against a vocabulary "
          "of around 150,000. Done naively that is a scan per token; done properly "
          "it is a step bound over a compiled structure, which is why the grammar is compiled once from the applet table "
          "and not interpreted per token. The "
          "compensating win is much larger: the reachable set is usually a few dozen "
          "tokens, and the output classifier is 155 MB of the model, so the same "
          "information that constrains the sampler could restrict that matrix "
          "multiply too."),
        h2("A property worth stating precisely"),
        p("It also covers a case that a naive implementation gets wrong. If "
          "<code>snap</code> and <code>snaps</code> both exist, then after decoding "
          "<code>snap</code> the machine is simultaneously at an accepting state and "
          "able to continue. An implementation that treats \"can continue\" as \"must "
          "continue\" makes the shorter name impossible to finish, and the applet quietly becomes unreachable. That bug only appears "
          "when someone adds a name extending an existing one, which is to "
          "say, later."),
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
        fig("linear-regression",
            "Least squares fits a line by minimising squared error, and has a "
            "closed-form solution. Ridge regression is the same idea with a "
            "regularisation term, and it is what routes tool calls here.",
            "A scatter plot with a fitted least-squares regression line"),
        p("Given a task written in natural language, something has to decide which "
          "applet to call. There are two implementations of that in the tree, and the interesting result, interesting enough that it shaped how "
          "the rest of the project is evaluated, is that the much older idea "
          "wins "
          "comfortably."),
        h2("The two paths"),
        table(
            ["Approach", "Cost", "Held-out accuracy"],
            [["Decode the applet name token by token under a grammar",
              "A transformer forward pass per token, 191 of them across the "
              "evaluation", "5.7%"],
             ["Nearest neighbour on pooled embeddings",
              "One pass over the corpus per query", "32.1%"],
             ["Ridge regression on one hidden state",
              "12,672 parameters, ~1.6 ms, no forward pass", "54.7%"]]),
        p("The winner is a linear probe. Take the hidden state the model has already "
          "computed while reading the request, subtract a mean, and multiply by a "
          "576×22 matrix. That matrix is solved in closed form, <code>W = (A'A + "
          "λI)⁻¹A'Y</code>, by Cholesky decomposition, which is ridge "
          "regression onto one-hot targets, or Widrow-Hoff least mean squares "
          "as published in 1960, solved directly instead of by descent."),
        p("It costs effectively nothing, because the hidden state is a byproduct of "
          "work already done, and it beats asking the 135M transformer the same "
          "question by a factor of nine."),
        h2("Why the closed form matters more than the accuracy"),
        p("The probe replaced a gradient-descent head, and that head was measured "
          "getting <em>worse</em> as it trained: 30% held-out accuracy untrained, "
          "10% after two epochs, 0% after eight. With forty examples spread across "
          "twenty-one classes there is nothing for SGD to do except memorise, and it "
          "did, thoroughly."),
        p("Ridge regression has no such failure mode. There is no learning rate, no "
          "epoch count and no schedule, so there is no way for it to drift past the optimum; it <em>is</em> the "
          "optimum for the regularisation chosen. That property is also what makes fitting practical on the machine "
          "itself: a 576×576 Cholesky is about 64 million operations, a "
          "fraction of a second, so the system can refit from its own accumulated "
          "corpus every time someone runs <code>teach</code>."),
        p("One detail that turned out to be load-bearing: the mean subtraction is "
          "not cosmetic. Pooled embeddings share a large common component, because "
          "every sentence contains the same function words, and leaving it in makes "
          "every pair of examples about 0.99 similar. The argmax then falls out as "
          "whichever vector happens to be longest. Uncentred nearest-neighbour "
          "answered the same class for all twelve items it was first tried on."),
        h2("Agreement, not votes"),
        p("Three independent cores run over the same request: the ridge probe, a "
          "multinomial naive Bayes over hashed character trigrams, and a lexical one "
          "over exact token identity. The two Bayes cores are trained by counting, "
          "so there is no matrix to factorise, and each sees something the probe cannot. The character core "
          "notices that \"duplicating\" and "
          "\"duplicate\" share most of their trigrams even when they tokenise "
          "differently, and needs no tokenizer at all."),
        p("The obvious thing to do with three cores is combine them into a better "
          "answer. That was measured, and it does not work: across 108 held-out "
          "items the best single core scores 77.8% and an equal-weight product of "
          "all three scores 76.9%. The ensemble is slightly worse than its best "
          "member."),
        p("What the same measurement did show is that their agreement predicts "
          "correctness sharply. Where all three pick the same applet they are right "
          "90.3% of the time; where they split, 50%. So the probe answers, the other "
          "two corroborate, and their disagreement never changes the answer; it changes what the "
          "system says about the answer. A router that knows when "
          "it is guessing can ask, escalate or refuse. One that is silently 78% "
          "accurate cannot, and that is worth considerably more than the point or "
          "two the ensemble was supposed to buy."),
        h2("Measurement discipline"),
        p("This project got its evaluation wrong three separate times before it "
          "settled, and each failure is worth naming because each one produced "
          "encouraging numbers. A grid sweep was scored on the test set. "
          "Cross-validation folded by template family in a way that leaked. And the "
          "test set itself moved whenever the corpus was appended to, so results "
          "were not comparable across runs and nobody noticed for a while."),
        p("The current arrangement has three splits rather than two. Validation is "
          "spent freely; the test slice is read once, at the end. A configuration is "
          "adopted only when it measures better, and the corpus holds out whole template families rather than sampled instances, since "
          "instances within a family differ only by their slot values, so an instance split measures "
          "memorisation while looking exactly like generalisation."),
        note("Negative results stay in the tree next to the code that produced them. "
             "The adapter head hurts at this data scale; the Product-of-Experts "
             "council does not improve accuracy. Both are kept, because the reason "
             "to know them is the same reason they were worth measuring, and because "
             "a deleted experiment gets repeated."),
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
        p("There is no graphics library anywhere in this. UEFI's Graphics Output "
          "Protocol hands the kernel four numbers (a base address, a width, a height "
          "and a stride) and everything above them is written by hand: the "
          "glyph renderer, the bevel primitives, the window manager, the wallpaper "
          "geometry and the taskbar. There is no VGA text mode to retreat to "
          "either, because the machine boots UEFI with no CSM, so there is no "
          "INT 10h, no text buffer at 0xB8000, and no VGA BIOS. Pixels are the only "
          "output device that exists."),
        h2("Windows 3.1 chrome, precisely"),
        p("The look is not an approximation of the era, it is the actual "
          "construction:"),
        ul("A <code>#C0C0C0</code> face colour on every panel.",
           "Two-pixel bevels, white along the top and left and dark grey "
           "along the bottom and right, for a raised surface, and the same two colours "
           "swapped for a sunken one. That single trick is most of the visual "
           "language.",
           "Title bars that fill with the selection colour when focused and go grey "
           "when they are not.",
           "Hard rectangles throughout: no anti-aliasing, no shadows, no rounded "
           "corners, no alpha."),
        p("Over that sits the <a href=\"aperture-science.html\">Aperture palette</a> "
          "and its amber accents, with a wallpaper computed as vector geometry "
          "because the kernel has no image decoder."),
        p("The style and the constraint are the same decision. A raised panel with "
          "a sunken trough, four colours and one bitmap font is simultaneously the "
          "Windows 3.x dialog vocabulary and the cheapest set of operations a "
          "framebuffer can perform. Nothing here needs blending, scaling or "
          "sub-pixel positioning, which is why the boot splash and the window "
          "manager can share a drawing model despite one of them running before "
          "there is a heap."),
        h2("The window manager, and its one idea"),
        p("Z-order <em>is</em> focus. There is no separate focused-window pointer "
          "kept in agreement with the stacking order, because the front window is "
          "the focused window by definition, and raising a window is how you focus "
          "it. That is one fact where most systems carry two, and two facts about "
          "the same thing eventually disagree."),
        p("The terminal is a window on the desktop and not the other way round, "
          "which sounds like a detail and is the whole design. A shell that draws a "
          "dialog over itself is a program with a pop-up; a desktop that hosts a "
          "terminal alongside other windows is an environment. The console already "
          "knew how to live inside a rectangle and reflow to it, so the terminal "
          "needed no special case at all; it is a window whose contents "
          "happen to be a character grid."),
        p("Repainting is total, back to front, on every change: no damage tracking, "
          "no dirty rectangles. That is affordable because of what causes a change. "
          "Nothing here animates, so a repaint happens when a key is pressed or the "
          "mouse moves, which is at most a few times a second, and 1280×800 is about "
          "a million stores into a write-back-mapped aperture. Damage tracking would "
          "buy nothing measurable and would cost the property this code cannot "
          "afford to lose, which is being obviously correct. The entire category of "
          "\"stale pixels from a window that used to be there\" cannot occur if "
          "there is no such thing as a partial repaint."),
        ("shots", [
            ("mouse.png", "A pointer, and the system status menu",
             "The GLaDOS desktop with a mouse cursor and the Program Manager system "
             "status list open",
             "PS/2 mouse support. The cursor is drawn by saving the pixels "
             "underneath it and restoring them before the next move, which is the "
             "1990 solution and the correct one when every repaint is total anyway. "
             "The second button opens the menu under the pointer.", False),
            ("enternet.png", "Enternet",
             "A small web browser window in GLaDOS displaying example.com",
             "The browser, which is a window like any other. It has fetched a page "
             "over the kernel's own TLS 1.3 connection, parsed the HTML and a subset "
             "of CSS, and laid the result out in its rectangle. Links are amber and "
             "navigable from the keyboard.", False),
        ]),
        h2("A bug that only a screenshot could find"),
        p("The console wrote straight into its own rectangle with no knowledge of "
          "what might be stacked on top of it, so a file browser opened over the "
          "terminal was gradually eaten from underneath by shell output. Nothing in "
          "the serial log showed this. The log was correct, the window manager's "
          "internal state was correct, and the screen was wrong, which is a category "
          "of bug that only exists once you have pixels."),
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
        fig("ip-stack",
            "The layering, as everyone draws it. Each of these boxes is a module in "
            "<code>src/net/</code>, and the arrows are the part that turned out to "
            "need care.",
            "Diagram of the internet protocol stack showing application, transport, "
            "internet and link layers"),
        p("Every layer here is hand-written: ARP, IPv4, ICMP, UDP, TCP, DHCP, DNS "
          "and <a href=\"tls.html\">TLS 1.3</a>. Underneath them are drivers for the "
          "Intel e1000, the Realtek RTL8168 and "
          "<a href=\"usb-xhci.html\">USB CDC Ethernet</a>. Interfaces are named (<code>lo</code>, <code>eth0</code>, "
          "<code>wlan0</code>), routing picks one by destination, and every address belongs to an interface, not to the machine."),
        p("<code>ping</code> was the first milestone, deliberately, because it "
          "exercises the entire path and produces one visible result. PCI discovery, "
          "an MMIO mapping, DMA rings, frame parsing, ARP resolution, an IPv4 header "
          "with a correct checksum, and a reply that has to actually come back. "
          "Anything broken anywhere in that chain shows up as silence, and silence "
          "is straightforward to bisect. The alternative first milestone is a TCP "
          "state machine failing intermittently, which is not."),
        h2("The rule that shapes everything"),
        p("The poll function never dispatches into a transport state machine. It "
          "only queues."),
        p("This is not a style preference, it closes a re-entrancy trap that is very "
          "easy to walk into. Sending an IPv4 packet calls address resolution, and "
          "resolution has to poll the card while it waits for the ARP reply. If poll "
          "ran TCP's state machine directly, a connection could re-enter its own "
          "control block through that path while an earlier mutable borrow was still "
          "live. So poll pushes frames onto an inbox, and TCP and UDP drain their "
          "own inboxes from a context where nothing is borrowed. The cycle can only "
          "form in one place, and that is where it gets broken."),
        fig("tcp-states",
            "RFC 793's state machine. GLaDOS implements the active-open path "
            "through this diagram, with RFC 6298 retransmission on top of it.",
            "The TCP connection state transition diagram",
            wide=True),
        h2("What the TCP implementation is, and is not"),
        p("It handles one connection at a time, actively opened, which is enough to fetch something over HTTP, and a byte stream "
          "is the thing every useful protocol is built on, so until one exists the network can only answer "
          "questions about itself. What is there: the RFC 793 state machine for an "
          "active open, RFC 6298 retransmission with Jacobson/Karels round-trip "
          "estimation and Karn's algorithm, sequence arithmetic that survives "
          "wraparound, an MSS option, and a receive buffer that advertises a real "
          "window."),
        p("What is absent is absent on purpose, and each omission has a price:"),
        ul("No reassembly queue. An out-of-order segment is dropped and acknowledged "
           "with the sequence still wanted, which is a duplicate ACK and makes the "
           "peer resend. Correct, and slower than necessary at exactly the moment "
           "the network is already losing packets.",
           "No congestion control. A fixed four-segment cap on flight size stands in "
           "for it. That is not TCP-friendly and would be rude at scale; it is "
           "defensible while every transfer is a short request.",
           "No Nagle and no delayed ACK. Both trade latency for efficiency, and this "
           "stack has no bulk traffic to be inefficient with.",
           "A shortened TIME_WAIT of two seconds rather than twice the maximum "
           "segment lifetime, which is a real hazard traded against a single "
           "connection slot."),
        h2("No interrupt-driven receive"),
        p("The card is polled, never interrupt-driven: during a blocking "
          "call "
          "by the wait loop, and otherwise by a service function called from the "
          "shell's idle loop. So the stack advances at best a hundred times a second "
          "while the machine sits at a prompt, and not at all while a long command "
          "runs unless that command yields. A connection genuinely makes no progress "
          "during a long generation, and that is visible from the outside as a stall."),
        fig("udp-encap",
            "Encapsulation, layer by layer. Getting the trimming right at each "
            "boundary is where two of this stack's better bugs lived.",
            "Diagram showing UDP data encapsulated in an IP packet inside an "
            "Ethernet frame"),
        h2("Bugs worth knowing"),
        ul("Ethernet pads frames to 60 bytes. A bare 40-byte TCP ACK therefore "
           "arrives with 20 bytes of whatever was in the buffer trailing it. IPv4 "
           "payloads have to be trimmed to the length the header declares, never to "
           "the length the frame happens to be.",
           "A shared event ring needs demultiplexing. In the USB driver, waiting for "
           "\"a transfer event\" instead of \"a transfer event for <em>my</em> "
           "endpoint\" meant a send completed against an arriving receive. DHCP kept "
           "working, because it alternates send and receive and the mismatched "
           "events happened to pair up; ARP broke, because it has no send to "
           "piggyback on. That asymmetry is what identified the bug: the protocol that "
           "worked was the clue, not the one that failed."),
        h2("What it can actually do"),
        p("From the shell: <code>dhcp</code> obtains a lease and configures the "
          "interface, <code>ping</code> round-trips, <code>dns example.com</code> "
          "resolves, and <code>https example.com /</code> completes a TLS 1.3 "
          "handshake, validates the certificate chain against a bundled root store, "
          "and returns the page. The browser sits on top of the same path and "
          "renders it."),
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
        fig("chain-of-trust",
            "Chain validation: each certificate is verified with the public key of "
            "the one above it, terminating at a root the machine already trusts.",
            "Diagram of a certificate chain of trust from a root CA to an end-entity "
            "certificate",
            wide=True),
        p("The kernel speaks TLS 1.3 using its own SHA-256 and SHA-384, HMAC, HKDF, "
          "AES, ChaCha20-Poly1305, X25519, RSA and ECDSA, and validates certificate "
          "chains against a root bundle exported from the host's store at build "
          "time. One cipher suite and one key exchange group, because TLS 1.3 "
          "permits exactly that and every additional option is another code path "
          "that never gets exercised."),
        note("This is the one place in the project where writing "
             "everything yourself is a liability. A bug in a driver produces a "
             "device that does not work, which is annoying and obvious. A bug here "
             "produces a connection that works perfectly and is not secure."),
        h2("Primitives chosen for checkability"),
        p("The selections were made for how hard they are to get quietly wrong, "
          "rather than for speed:"),
        ul("ChaCha20-Poly1305 rather than AES-GCM, because it has no key-dependent "
           "table lookups and therefore no cache-timing side channel to reason "
           "about. Constant-time AES in software is achievable and it is not "
           "achievable by accident.",
           "X25519 rather than a NIST curve, because it needs no point validation, "
           "has no invalid-curve attack surface, and has substantially fewer ways to "
           "be subtly incorrect."),
        fig("dh-exchange",
            "The Diffie-Hellman idea both parties rely on: a shared secret derived "
            "from two private values that never cross the wire. X25519 is the same "
            "construction over an elliptic curve.",
            "Diagram of the Diffie-Hellman key exchange using mixed colours as an "
            "analogy"),
        p("Every primitive is checked against published RFC and FIPS test vectors at every boot, eleven vector sets in total, and the "
          "boot log prints a pass "
          "or fail line for each one."),
        h2("An ECDSA break that sat in plain sight"),
        p("That self-test output is easy to scroll past, and scrolling past it cost "
          "a full debugging cycle. A broken ECDSA implementation was printing "
          "<code>FAIL</code> in the crypto block the entire time, while the boot log "
          "was being sliced down to look at something else entirely. The test worked. "
          "The reader did not."),
        h2("What validation actually checks"),
        p("Four things, and a certificate is rejected if any of them fails. The "
          "chain: each certificate verified with the public key of the next one up. "
          "The root: identified by SHA-256 of its full DER encoding against a "
          "built-in list, which is fingerprint pinning, so there is no name "
          "canonicalisation to get wrong. The dates, against the "
          "CMOS clock. And the name, per RFC 6125, from subjectAltName only, with "
          "wildcards allowed solely in the leftmost label and dNSName entries kept "
          "strictly apart from iPAddress ones."),
        p("There is a fifth check that is easy to overlook and is the one that "
          "matters most: CertificateVerify, where the server signs the handshake "
          "transcript with the key in its certificate. That is the step proving the "
          "party at the other end of <em>this</em> connection holds the private key. "
          "Without it, an attacker can replay any certificate they have ever seen."),
        p("The DER parser in front of all this is deliberately strict, because it "
          "reads bytes chosen by whoever we are talking to before we know who that "
          "is, which makes it the most hostile input in the system. Every length is "
          "checked against the buffer, indefinite-length encodings are refused "
          "outright, nesting depth is bounded, and nothing is copied on the strength "
          "of a length field alone. Two parser bugs are recorded in the source: "
          "<code>expect(tag)</code> must not consume input on a mismatch, and "
          "\"try for the value, skip if that failed\" throws the value away when the "
          "optional field it was looking for is simply absent."),
        h2("Jacobian coordinates, and a trap"),
        p("ECDSA works in Jacobian coordinates because affine ones cost a modular "
          "inversion per point operation, and a modular inversion is a full modular "
          "exponentiation. For P-384 that came to roughly 460,000 allocating "
          "multiplications per signature, which exhausted the heap instead of merely being "
          "slow."),
        p("The trap in the replacement: the inversion routine takes and returns "
          "<em>ordinary</em> values, so handing it something already in Montgomery "
          "form computes the wrong answer and says nothing. There is a wrapper that "
          "converts correctly, and using it is not optional. This is the same shape "
          "of failure as the model bugs elsewhere in this wiki, a representation "
          "mismatch that produces well-formed output, which is why the boot "
          "vectors exist."),
        h2("What is still not safe"),
        ul("Validation reports, and does not enforce. The <code>https</code> command "
           "prints what was established and shows the body either way, which is a "
           "deliberate choice for a system whose purpose is inspection and the exact "
           "opposite of what a browser should do. A caller that cares has to check "
           "the result itself.",
           "There is no revocation checking of any kind, no CRL and no "
           "OCSP, so a certificate withdrawn by its issuer is accepted here until it expires.",
           "Key material comes from the timestamp counter and not from "
           "<code>RDRAND</code>. A counter is not a random number generator. This is "
           "a genuine weakness and it is on the list.",
           "Path length constraints and key usage bits are parsed, but only "
           "basicConstraints is enforced."),
        p("Writing these down is the point. A system that claimed to be secure here "
          "would be worse than one that states precisely where it is not, because "
          "the first invites someone to rely on it."),
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
        fig("merkle-tree",
            "A hash tree. Each leaf is named by the hash of its data and each "
            "interior node by the hash of its children, so the root hash names the "
            "entire structure.",
            "Diagram of a Merkle hash tree with data blocks, leaf hashes and a root "
            "hash",
            wide=True),
        p("The namespace here is not a tree of blocks with names attached to it. "
          "Objects are named by the SHA-256 of their own contents and assembled into "
          "Merkle trees, and several operations that are normally expensive collapse "
          "as a direct consequence."),
        ul("A copy is O(1) on any size. Identical content already has an identical "
           "name, so copying is adding a reference to something that exists.",
           "A snapshot is one hash. The root of the tree names the entire state of "
           "the namespace at that moment.",
           "Comparing two whole subtrees is one comparison, because if the root "
           "hashes match the contents match.",
           "Rolling back branches instead of rewriting. The old root remains valid "
           "and still points at everything it pointed at, because nothing was "
           "overwritten to produce the new one.",
           "Deletion cannot destroy content that something else still references, "
           "since names are derived from content and never assigned."),
        p("Those are not features that were added on top. They are what content "
          "addressing already implies, and the applet list reflects it: "
          "<code>cp</code>, <code>same</code> and <code>snap</code> exist because "
          "they became cheap."),
        h2("The rule that makes it work"),
        p("The content hash covers content only, and never block locations. This "
          "sounds too obvious to state and is easy to violate by including a block pointer somewhere in the hashed structure, at which point "
          "moving a block to defragment renames the object that contains it, every reference to "
          "that object breaks, and deduplication stops working without any error "
          "being raised."),
        h2("Surviving power loss"),
        p("Three properties, each chosen against a specific failure. The store is "
          "append-only, so chunks are never rewritten and a checkpoint that fails halfway cannot damage an earlier one, "
          "because it never touches the blocks an earlier one occupies. It is content-addressed, so every chunk carries the "
          "SHA-256 of its contents and reads verify it, which means corruption "
          "surfaces at read time instead of propagating into whatever is being "
          "restored."),
        p("And there are two superblocks. The obvious design is a "
          "single root pointer updated atomically, and it is not actually safe: a "
          "512-byte write is not guaranteed atomic across power loss, so a torn "
          "sector can destroy the only root and take every checkpoint with it. Two "
          "superblocks alternate, each carrying a sequence number and a checksum "
          "over itself, and mounting picks the highest sequence number that "
          "checksums. A torn write then costs the newest checkpoint and nothing "
          "else. This is the reasoning behind ZFS uberblocks and F2FS checkpoints, "
          "arrived at from the same starting point."),
        fig("nvme-ssd",
            "NVMe was chosen over USB Mass Storage because it is about a third of "
            "the work: some MMIO registers and a pair of ring buffers in ordinary "
            "memory, with no host controller stack underneath.",
            "An M.2 NVMe solid state drive"),
        h2("The driver, and why it is small"),
        p("The NVMe driver exists at all because of decisions made earlier. Identity "
          "mapping means virtual equals physical, so a heap pointer <em>is</em> a DMA address, with no IOMMU setup, "
          "no translation and no bounce buffers. PCI "
          "enumeration through ECAM finds the controller and its BAR. And non-RAM "
          "pages are mapped uncacheable, so doorbell writes actually reach the device and do not sit in a "
          "write-back cache line, which is the kind "
          "of thing that costs a day if it is not arranged in advance."),
        h2("Write locking, and why it stays that way"),
        p("NVMe writes are locked by default. They unlock only after a target region "
          "has been explicitly identified, the formatter re-checks before touching "
          "anything, and every error path locks them again on the way out."),
        p("The reason is specific: the only NVMe device in the "
          "development laptop is the internal drive holding Windows, and a stray LBA "
          "there is unrecoverable. Reads are harmless and prove everything except "
          "the write path, so the write path is exercised against a throwaway QEMU "
          "image instead. On a machine whose disk is fully allocated, storage "
          "initialisation fails and says so, which is the intended outcome rather "
          "than a bug to work around. Leaving the lock open on a failure path is "
          "precisely how a safety mechanism becomes decorative."),
        note("The development machine's boot disk is counterfeit. It advertises "
             "976 GB and holds 14.67, which is why the partitioning tooling uses MBR (a GPT backup header would "
             "be written to flash that does not exist) and why it carries an explicit safe-capacity limit. Nothing anyone "
             "cares about goes on that disk."),
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
        fig("usb3-contacts",
            "USB 3 added five SuperSpeed contacts behind the original four. The "
            "host controller behind them is xHCI, which is a very different animal "
            "from its predecessors.",
            "Diagram of a USB 3.0 connector showing the additional SuperSpeed "
            "contacts"),
        p("UEFI has a perfectly good USB stack, and "
          "<a href=\"uefi-kernel.html\"><code>ExitBootServices</code> destroys "
          "it</a> permanently along with everything else in boot services. So the "
          "kernel has its own."),
        p("It was written because it is the only route this machine has to a "
          "wireless dongle, the built-in card being "
          "<a href=\"usb-wifi-driver.html\">unusable for other reasons</a>. xHCI is "
          "the opposite kind of problem from that card: laborious, and specified. "
          "Intel publishes the register layout and the ring formats, so every question has an answer somewhere, and nobody has to read "
          "another project's driver as though it were a datasheet."),
        h2("Rings are the whole design"),
        p("xHCI is not register-poking in the style of an older NIC. Everything is a "
          "ring of 16-byte Transfer Request Blocks shared with the controller, and "
          "each ring carries a cycle bit that flips every time the producer wraps "
          "around."),
        p("That single bit is the entire synchronisation protocol between driver and "
          "controller. The controller consumes entries whose cycle matches its "
          "current state and stops at the first that does not, so both sides know "
          "which entries are new without any other coordination. Get it wrong and nothing is corrupted. The controller simply "
          "never sees the command, "
          "which presents as a device that enumerates and then goes quiet, with no "
          "error anywhere to attach a debugger to."),
        p("Three kinds of ring matter: a command ring the driver writes and the "
          "controller reads, an event ring the controller writes and the driver "
          "reads, and one transfer ring per endpoint. Everything completes "
          "asynchronously through the event ring, including all the operations that "
          "read like synchronous function calls in the driver source."),
        h2("Three bugs, each found by the layer above"),
        ol("Version 0.0. The capability-length and version fields share one 32-bit "
           "register, and that register block only answers dword reads, so a 16-bit "
           "read at offset 2 returns zero. The controller appeared to implement "
           "xHCI version 0.0, which would be impressive.",
           "Setup packet byte order. Little-endian means byte zero is the request "
           "type, so <code>SET_CONFIGURATION</code> is <code>0x0900</code> rather "
           "than <code>0x0009</code>. The descriptor read sitting immediately next "
           "to it in the source was correct entirely by luck, its two bytes being "
           "the same either way round.",
           "A stall halts the endpoint. Probing configuration descriptors past the "
           "last one stalls, which halts endpoint zero until something explicitly "
           "resets it. Every subsequent control transfer then failed, so the symptom "
           "appeared two steps downstream of the cause and looked like the device "
           "had simply stopped responding."),
        h2("The shared event ring"),
        p("The subtlest of them, and the one worth carrying to other drivers. The "
          "event ring is shared by every endpoint, so code that waits for \"a transfer event\" and not \"a transfer "
          "event for <em>this</em> endpoint\" is correct only while exactly one transfer is ever in flight."),
        p("That stops being true the moment a receive is left permanently armed, "
          "which is exactly what a network interface does. A send then completes "
          "against whichever event turns up first, so a transmit reports success on "
          "the strength of an arriving <em>receive</em>, and the frame that actually "
          "arrived gets dropped while its endpoint still looks armed."),
        p("What identified it was the asymmetry between two protocols. DHCP kept "
          "working, because it strictly alternates send and receive, so the "
          "mismatched events happened to pair up one-for-one. ARP did not, because "
          "it has no send of its own to piggyback on. The protocol that worked was "
          "the clue."),
        h2("CDC Ethernet"),
        p("Once bulk endpoints work, a USB Ethernet adapter becomes a network "
          "interface, and <code>eth0</code> can exist on a machine with no Ethernet "
          "port. There is one trap on the way in: having a CDC data interface is not "
          "the same as being an Ethernet adapter. A commonly emulated device offers "
          "two configurations that both qualify, and the first of them is RNDIS, "
          "which accepts bulk writes cheerfully and passes nothing at all, because "
          "it wants a control protocol spoken to it first."),
        p("The correct test is the presence of an Ethernet Networking Functional "
          "Descriptor. That is also where the MAC address lives, as a string index "
          "resolved to ASCII hex, because CDC provides no binary field for it "
          "anywhere in the specification."),
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
        fig("wifi-dongle",
            "A USB wireless adapter is a complete radio and MAC on the far side of "
            "a bus the kernel already drives, which is the entire reason this "
            "approach is tractable.",
            "A TP-Link USB WiFi dongle"),
        p("Wireless is the one major subsystem that does not work, and the reason is "
          "worth setting out, because it is not a question of effort."),
        h2("CNVi: the card is not a card"),
        p("The laptop's built-in wireless is CNVi, Intel's split architecture. The MAC, the part a driver would actually talk to, lives in the "
          "platform controller hub, and the M.2 module is only a radio. Reaching it means an "
          "undocumented signed-firmware protocol that is published nowhere."),
        p("It is worth being concrete about what a modern wireless driver costs, "
          "because \"add WiFi\" sounds like the same size of job as \"add "
          "Ethernet\" and is not. The e1000 driver is about 350 lines: map a BAR, "
          "set up two descriptor rings, poll them. For an Intel AX-series card you "
          "need a signed firmware blob of roughly a megabyte, loaded into the device "
          "over a bootstrap protocol before it will do anything at all, "
          "non-redistributable and undocumented and different between families. Then "
          "an asynchronous host command interface with its own versioned message "
          "formats. Then 802.11 itself: scanning, authentication, association, and "
          "the fact that a wireless frame carries three or four address fields "
          "depending on direction, plus fragmentation and aggregation. Then WPA2."),
        p("So the wireless module identifies the hardware and declines to pretend it "
          "can drive it. The <a href=\"tls.html\">WPA2 supplicant</a> is finished: "
          "PBKDF2-HMAC-SHA1 at 4,096 rounds for the pairwise master key, the "
          "PRF-384 expansion into confirmation, encryption and data keys, the "
          "four-way handshake, and AES key unwrap for the group key. It is written "
          "as pure functions over byte slices with no I/O anywhere, so that when a "
          "driver arrives it supplies frames and this supplies answers and none of "
          "it has to change. It passes the IEEE 802.11i test vectors at every single "
          "boot, and has never been within reach of an antenna."),
        h2("A USB dongle sidesteps all of it"),
        p("A USB wireless adapter is a complete radio and MAC behind a bus the "
          "kernel <a href=\"usb-xhci.html\">now drives</a>. The target is a Realtek "
          "RTL8188EU, chosen because its firmware situation is tractable."),
        p("It has no memory-mapped registers at all. Every register access is a "
          "vendor control transfer, which means a single register read is a full USB "
          "round trip of a few hundred microseconds. That one fact shapes the whole "
          "driver: anything written as a poll loop over a status register is orders "
          "of magnitude slower than the same code would be against MMIO, and the structure has to be arranged around batching, at some "
          "cost to how readable it is."),
        h2("Why the tables were copied, and why that is stated"),
        p("Bringing up the chip needs a power-on sequence, PHY and radio "
          "initialisation tables, and an efuse layout: <strong>509 specific "
          "register/value pairs</strong> that exist in Realtek's vendor driver and "
          "in Linux's rtl8xxxu, and nowhere else."),
        p("They were transcribed from Linux, mechanically, and not "
          "retyped. 509 hex pairs copied out by a human contain errors at "
          "some rate, and an error here does not fail loudly. A wrong AGC value "
          "costs sensitivity; a wrong PHY value skews a filter. Both present as "
          "\"wireless is a bit unreliable.\""),
        note("Those constants are GPL-2.0 and live in a single file that says so at "
             "the top. They are the only code in this kernel not written for it, "
             "apart from Rust's <code>core</code>, and they are listed on "
             "<a href=\"../credits.html\">the credits page</a>. Naming the one "
             "exception is cheaper and more useful than a blanket claim that "
             "quietly covers someone else's work."),
        h2("Order is content"),
        p("One property of those tables deserves its own warning. The AGC table "
          "writes the <em>same register</em> 130 times in a row, with the table "
          "index encoded into the value being written. It is a sequence, and the position of each entry is part of the "
          "data."),
        p("Which means sorting it, deduplicating it, or reordering it for tidiness "
          "produces a completely different table that still looks entirely "
          "reasonable to anyone reading the source. The tooling that transcribed it "
          "preserves order for that reason, and the file says so in case somebody "
          "later decides it looks untidy."),
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
        fig("compact-disc",
            "ISO 9660 was designed for physical media like this. The format long "
            "outlived the medium, and El Torito is the 1995 extension that taught "
            "it to boot.",
            "Diagram of the layers of a compact disc"),
        p("The ISO builder writes both filesystems itself. The practical reason is "
          "that xorriso and mkisofs are absent from most Windows machines, and the "
          "Windows ADK's equivalent is a 1.5 GB install in order to produce one "
          "600 MB file. Both formats are published, neither is large, and the same "
          "reasoning produced the FAT reader in the kernel."),
        h2("How UEFI boots an optical disc"),
        p("The firmware reads the El Torito boot catalog, looks for an entry whose "
          "platform id is <code>0xEF</code>, and treats the sectors that entry "
          "points at as a FAT filesystem: an EFI System Partition that "
          "happens to be living inside an ISO image. It then loads "
          "<code>\\EFI\\BOOT\\BOOTX64.EFI</code> from it and jumps."),
        p("Which makes the ISO 9660 structure wrapped around the whole thing almost "
          "ceremonial: it exists so the disc mounts and shows its contents, not so "
          "it boots. It gets built correctly regardless, because a disc that boots "
          "fine and appears empty when you mount it looks broken to everyone who "
          "checks."),
        p("El Torito is named after the Californian restaurant where the "
          "specification was drafted in 1994, which is the sort of detail that makes "
          "reading old standards worthwhile."),
        h2("Long file names are not optional"),
        p("The kernel opens <code>\\GLADOS\\tokenizer.bin</code>. That base name is "
          "nine characters and has no 8.3 representation, so a short-name-only image "
          "presents it as <code>TOKENI~1.BIN</code> and the kernel fails to find its tokenizer at boot, on real hardware, "
          "at the exact moment there is no filesystem left to debug from, since this all happens after "
          "<code>ExitBootServices</code>."),
        p("So the builder generates VFAT long-name entries, complete with the "
          "specified rotate-and-add checksum that ties each set of long entries to "
          "its short entry. Get that checksum wrong and the long names are silently "
          "ignored, which returns you to the previous paragraph."),
        h2("FAT32 has a floor"),
        p("FAT32 is <em>defined</em> as a volume with at least 65,525 clusters. "
          "Below that the volume is by definition FAT16, and firmware that trusts the count over the boot sector's claim "
          "will misparse the entire thing. Since the cluster count has a floor, the cluster size sets a "
          "minimum image size:"),
        table(
            ["Cluster size", "Smallest possible volume"],
            [["4 KiB", "~256 MB"],
             ["512 B", "~33 MB"]]),
        p("Before this was worked out, the kernel-only image was 269 MB to carry a "
          "1 MB kernel: 255 MB of zeroes, faithfully compressed by nothing and "
          "downloaded in full. Choosing the cluster size from the actual payload "
          "brings it to 33 MB, and even that is the format's floor: the volume is already as "
          "small as a FAT32 volume is permitted to be."),
        h2("Why write a FAT reader as well"),
        p("There is a matching FAT implementation inside the kernel, and it exists "
          "for a reason that only becomes obvious after "
          "<code>ExitBootServices</code>. The firmware can read files, and does; that is how the model gets "
          "loaded. But its filesystem driver dies with "
          "boot services, so from the moment the kernel owns the machine it can "
          "address every block on the disk and understand none of them. That gap is "
          "the difference between a system that boots and a system you could use to "
          "fix a machine."),
        p("FAT specifically, because FAT is what the ESP is, and the ESP is where "
          "boot configuration lives and therefore where things go wrong. It is also "
          "small enough to implement correctly: a chain of 32-bit numbers, a table "
          "of 32-byte records, and one awkward extension for long names."),
        p("It is deliberately read-only. Writing FAT means allocating clusters, "
          "keeping two copies of the allocation table in agreement, and holding "
          "directory entries consistent across a power cut. A rescue tool that can "
          "read a broken disk is useful; one that can half-write it is worse than "
          "nothing."),
        h2("Verifying the result"),
        p("The image is checked by booting it, which sounds circular and is the only "
          "test that means anything: firmware is the component least likely to agree "
          "with the specification. QEMU with OVMF catches structural mistakes, and "
          "the GF63's own firmware catches the rest, of which there have been "
          "enough to justify the trip. A disc that boots under OVMF and not on metal is a normal outcome "
          "here."),
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
        fig("templeos",
            "TempleOS: 640×480, sixteen colours, one address space, everything at "
            "ring 0, and a compiler you could call from anywhere in the system.",
            "A TempleOS desktop screenshot showing its text-mode interface"),
        p("TempleOS is the obvious ancestor of a project shaped like this one, and "
          "the parts it got right are the parts this system also relies on. Terry "
          "Davis wrote it alone over more than a decade: a 64-bit operating system "
          "in ring 0, in one address space, with its own compiler, its own graphics "
          "and its own language, and it worked."),
        h2("The inheritance"),
        ul("Ring 0, always. No user/kernel split and no syscall instruction, in "
           "either system.",
           "One identity-mapped address space, where everything can reach "
           "everything. The paging code here says so directly: physical equals "
           "virtual, one set of page tables for the whole machine, no higher half, "
           "no per-process tables and no TLB shootdown to design.",
           "No isolation, arrived at deliberately, with the costs written down "
           "where a reader will find them.",
           "An aesthetic committed to on purpose. TempleOS chose "
           "640×480 and sixteen colours and never apologised for it; this one "
           "chooses <a href=\"aperture-science.html\">amber on black and Windows 3.1 "
           "chrome</a> and does not either.",
           "One person, from scratch, with the tools included in the thing being "
           "built."),
        h2("Where the two diverge"),
        p("TempleOS had HolyC serving as compiler, shell and system language "
          "simultaneously, which is a genuinely radical unification: the command line was "
          "the language, and the language was compiled. GLaDOS has "
          "<a href=\"glados-os.html\">a small interpreted language</a> in the shell, "
          "but the system is Rust and stays Rust, and the interpreter is a guest in it."),
        p("The organising question is different too. TempleOS was built around one "
          "person's particular conviction about what a computer should be. This is "
          "built around a narrower technical question: what changes when a language model is a kernel primitive and not "
          "a userspace process? That is "
          "answerable by measurement, which is a different sort of project even "
          "where the architecture rhymes."),
        h2("What it got right"),
        p("That the boundary between kernel and user is a choice and not "
          "a law of nature, and that removing it makes some things genuinely simpler "
          "instead of merely more dangerous. A function call is a function call. "
          "There is no marshalling, no permission check on the way through, no "
          "context switch and no ABI that has to stay stable across a release."),
        p("For a system in which a model reaches for kernel functions constantly, "
          "that stops being a philosophical position and becomes an engineering one. "
          "The identity map, the single heap and the absence of a trap gate are all "
          "load-bearing here for the same reasons they were load-bearing there, and "
          "the null-page fault guard is the one deliberate exception in both."),
        h2("The parts that turned out to be practical, not just philosophical"),
        p("Three specific things fall out of the single-address-space choice, and "
          "each one saved real work here."),
        p("Identity mapping means a heap pointer is a DMA address. The NVMe driver "
          "and both network drivers hand device controllers pointers straight out of "
          "the allocator, with no IOMMU configuration, no address translation and no "
          "bounce buffers anywhere. A conventional kernel spends a meaningful amount "
          "of code on that translation layer; here it does not exist to be got "
          "wrong."),
        p("The context switch is six callee-saved registers and a stack pointer. "
          "There is no privilege transition, no reload of the ring-0 stack pointer "
          "in the task state segment, no page-table swap and no TLB flush. The "
          "caller-saved registers need no handling at all, because the calling "
          "convention already permits a function call to clobber them. Preemption "
          "works by calling the switch from inside the timer interrupt: the "
          "interrupted task's full state is sitting on its own stack, pushed by the "
          "handler prologue, so swapping the stack pointer underneath means each "
          "task carries its own suspended interrupt frame around with it and resumes "
          "into the middle of its own timer handler."),
        p("And there is no ABI to keep stable, anywhere, because nothing crosses a "
          "boundary. Changing the signature of a kernel function is a recompile and "
          "nothing more. For a system where the model is "
          "expected to reach into the applet table constantly, that is what keeps the table editable instead of versioned."),
        note("This is lineage, not continuation. It is not an attempt to "
             "carry on TempleOS and it is not a tribute; it is a different project "
             "that started from a different question and arrived at several of the "
             "same structural answers, which is the most useful kind of "
             "corroboration a design decision can get."),
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
        p("There is no <code>cargo test</code> here. This is a <code>no_std</code> "
          "UEFI binary with no host test runner and no operating system underneath "
          "it to host one, so the test suite had to go somewhere else. It went to "
          "three places, each catching a category the others cannot."),
        h2("1. Self-tests at boot"),
        p("Every boot exercises the heap, the timer, the clock, the namespace, "
          "eleven sets of published cryptographic vectors, "
          "<a href=\"constrained-decoding.html\">constrained decoding</a> and the "
          "linear probe, printing <code>ok</code> or <code>FAIL</code> on each line. "
          "That output is the test suite, and it runs on the real machine against "
          "the real hardware every time anyone starts it, which no host runner could "
          "manage."),
        p("Some of those checks are more interesting than a pass line suggests. The "
          "constrained-decoding test runs 200 random decodes at maximum temperature "
          "and asserts that none escaped the grammar. The null-dereference test "
          "confirms that page zero faults, which is what caught the memory-map bug "
          "described on <a href=\"uefi-kernel.html\">the UEFI page</a>. And there is "
          "a floating-point guard that parks a known pattern in the vector registers "
          "across a context switch and checks it survived. It never fires "
          "now, and it fires immediately if extended state saving is "
          "removed."),
        note("It is easy to scroll past, and it does catch real bugs. An ECDSA break "
             "was sitting in the crypto block printing <code>FAIL</code> for an "
             "entire debugging cycle while the log was being sliced down to look at "
             "something else. The Enrichment Centre reminds you that the testing "
             "protocol only works if someone reads the results."),
        h2("2. A scripted QEMU harness"),
        p("A driver script boots QEMU, stages the binary, resets the firmware's "
          "NVRAM to a pristine state, and drives the shell over a serial socket, "
          "sending commands and capturing everything that comes back. That makes a "
          "regression test out of any sequence someone can type."),
        p("Two details in there were not obvious in advance. QEMU's Windows stdio "
          "backend reads console handles directly, so piping a script into it does nothing whatsoever, silently, "
          "with no error and no output, whereas a TCP socket behaves like a socket on every platform. And a stale firmware "
          "boot entry sends the machine to the UEFI shell instead of the image, "
          "which is visually indistinguishable from the system failing to boot, so the NVRAM gets reset on every run and is never trusted."),
        p("The harness also captures the framebuffer through QEMU's monitor and "
          "converts it to PNG, because <a href=\"gui.html\">a serial log says "
          "nothing at all about whether the screen is right</a>. That capture found "
          "three layout bugs and produced one memorable false alarm: a screenshot "
          "taken mid-repaint looked exactly like a broken compositor and cost a "
          "bisect before the fix turned out to be waiting two seconds for the frame "
          "to settle."),
        h2("3. A numeric oracle"),
        p("For the model there is a NumPy reference implementation that reads the "
          "<em>converted</em> checkpoint and produces logits to diff against. "
          "Reading the converted file, and not the original safetensors, is "
          "the entire point of the design: a converter bug then shows up in both, so "
          "agreement plus wrong output indicts the converter, and disagreement "
          "indicts the kernel. Without that property a mismatch tells you only that "
          "something, somewhere, is wrong."),
        p("This is what caught <a href=\"rope.html\">the RoPE convention</a>, and no "
          "amount of reading the code would have. Both conventions are perfectly "
          "reasonable code that does exactly what it says. The difference only "
          "exists relative to what the weights were trained under, which is not a "
          "fact any single implementation contains."),
        p("QEMU cannot run the real model, incidentally, which shapes all of this. "
          "Its built-in FAT support caps the emulated disk at 516 MB and the Qwen3 "
          "image is larger, so the full checkpoint only runs on the actual laptop. "
          "The oracle is therefore the only way to check the real model's numerics "
          "without standing in front of the hardware."),
        h2("The habit underneath"),
        p("Measure, do not assume, and then go and look at the result. This project has "
          "produced a steady supply of confident beliefs that turned out to be "
          "false, and in almost every case a cheap deterministic check found it: the "
          "oracle found the RoPE pairing, screenshots found the layout bugs the "
          "serial log could not express, an A/B against a deliberately tiny context "
          "window isolated an off-by-one to a single token, and the evaluation "
          "harness exists because the measurement itself was got wrong three "
          "separate times before it settled."),
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
        p("Captured from the running system under QEMU, through the emulator's monitor and not with a camera pointed at a "
          "screen, so these are the "
          "framebuffer's exact contents at 1280x800."),
        ("shots", [
            ("desktop.png", "Desktop with terminal and Program Manager",
             "GLaDOS OS desktop showing a Windows 3.1 styled terminal window with "
             "the boot log, a Program Manager window, and a taskbar",
             "The desktop as it comes up. The terminal is showing its own boot log, "
             "with Program Manager beside it and a taskbar along the bottom. Every "
             "pixel here was drawn by the kernel into a UEFI framebuffer; there is "
             "no graphics library underneath.",
             True),
            ("desktop-clean.png", "Desktop wallpaper",
             "The GLaDOS desktop wallpaper showing the aperture mark drawn as "
             "vector geometry",
             "The same desktop with the terminal minimised. The mark on the "
             "wallpaper is computed as arcs and lines at boot, because the kernel "
             "has no image decoder and adding one to display a logo seemed like the "
             "wrong order to do things in.", False),
            ("model.png", "The model answering",
             "GLaDOS running a language model in kernel space, answering a question "
             "about operating systems",
             "The resident model answering <em>what is an operating system</em>. The "
             "forward pass is running in the same address space as the code that "
             "drew the window it is printing into.", False),
            ("mouse.png", "A pointer, and the system status menu",
             "The GLaDOS desktop with a mouse cursor and the Program Manager system "
             "status list open",
             "PS/2 mouse support, with the cursor drawn and undrawn by saving and "
             "restoring the pixels underneath it. The second button opens the menu "
             "under the pointer. Behind it the boot log records the model's shape: "
             "576-dimensional, 30 layers, nine query heads to three key/value heads, "
             "and 134,515,008 parameters in 132,402 KiB of int8 weights.", False),
            ("enternet.png", "Enternet, fetching a page",
             "A small web browser window in GLaDOS displaying example.com, fetched "
             "over the kernel's own TLS 1.3 connection",
             "Enternet, the browser. It has fetched <code>example.com</code> over a "
             "TLS 1.3 connection this kernel negotiated itself, parsed the HTML and "
             "the CSS, and laid the result out in a window. The status line counts "
             "what it found: three blocks and one link.", False),
            ("enternet-link.png", "Following a link",
             "The Enternet browser showing the IANA example-domains page after "
             "following a hyperlink",
             "The same browser after following that one link to IANA's page about "
             "reserved domains: 32 blocks and 30 links, with headings, paragraphs, "
             "a bulleted nav list and the anchors highlighted in amber. Keyboard "
             "navigation moves between links; there is no JavaScript engine and "
             "there is not going to be one.", False),
            ("enternet-back.png", "And back again",
             "The Enternet browser having navigated back to example.com from the "
             "IANA page",
             "Backspace walks the history back to where it started, which required keeping the parsed document around instead of "
             "refetching it. The shell "
             "behind shows the key sequence that got here: alt-tab, tab, enter, "
             "backspace.", False),
        ]),
        h2("What you are looking at"),
        p("The window chrome is Windows 3.1 construction, done properly: a "
          "<code>#C0C0C0</code> face, two-pixel bevels with the light source at the "
          "top-left, and a title bar that fills with the selection colour when it "
          "takes focus. Over that sits the "
          "<a href=\"../wiki/aperture-science.html\">Aperture palette</a>, which is "
          "where the amber comes from."),
        p("There is no graphics library beneath any of it. UEFI's Graphics Output "
          "Protocol hands the kernel a base address, a width, a height and a stride, "
          "and everything above those four numbers is written by hand: the glyph "
          "renderer, the bevel routines, the window manager, the wallpaper geometry "
          "and the compositor, such as it is. Every change repaints the entire "
          "screen back to front, which sounds wasteful and costs about a million "
          "stores at a rate of a few per second, because nothing here animates. See "
          "<a href=\"../wiki/gui.html\">the GUI page</a> for how it fits together, "
          "and for the bug that only a screenshot could have caught."),
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
          "distributed from, back when the way you got an operating system was to "
          "find a directory listing at a university and hope it was nearby. The "
          "listing below is live; the files behind it are served from GitHub "
          "Releases, because GitHub Pages caps at 100 MB per file and would turn "
          "away every ISO here."),
        p("Sizes are what they are for reasons the "
          "<a href=\"../wiki/iso-el-torito.html\">ISO page</a> explains: most of the "
          "575 MB image is model weights, and even the 33 MB kernel-only image is at the smallest size FAT32 permits, which is not the smallest "
          "size the payload needs."),
        p("Every release carries a <a href=\"" + REL + "SHA256SUMS\">SHA256SUMS</a> "
          "file. Checking it is worthwhile, since a truncated ISO still flashes and "
          "still boots before failing somewhere unrelated to the truncation."),
        p("If you only want the ISO, "
          "<a href=\"../download/\">the download page</a> has flashing instructions "
          "and hardware notes next to it."),
    ],
)

# --------------------------------------------------------------------------
# Credits
# --------------------------------------------------------------------------

page(
    "credits",
    "Credits. Where the diagrams and the borrowed code came from",
    "Sources for the diagrams used across the GLaDOS documentation, and the one "
    "file in the kernel that was not written for this project.",
    ["GLaDOS credits", "image credits", "attribution", "Wikimedia Commons"],
    blocks=[
        p("The screenshots on this site are of the system running. Everything "
          "else with a picture in it was drawn by someone else, and this page "
          "says who. The table is generated from the same record "
          "<code>tools/fetch-media.py</code> writes when it downloads a file, so "
          "it cannot fall out of step with what the pages actually show."),
        ("credits", None),
        h2("Code"),
        p("One file in the kernel was not written for this project: "
          "<code>src/dev/rtl8188eu_tables.rs</code>, which is 509 register values "
          "transcribed from Linux's GPL-2.0 <code>rtl8xxxu</code> driver. There is "
          "no datasheet for that chip and no other source for the numbers. The "
          "file carries the GPL-2.0 notice at the top and is the only thing in the "
          "tree that does. Everything else is this project's, apart from Rust's "
          "<code>core</code> library, which the compiler supplies."),
        p("The site's layout is a reconstruction of linux.org as it looked around "
          "2005, done from memory and screenshots. No markup or stylesheet was "
          "copied."),
        ("seealso", ["glados-os", "usb-wifi-driver", "wiki"]),
    ],
)

# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

CSS = """/* Layout copied from linux.org as it stood in 2005: a coloured masthead bar, a
   left column of boxed navigation sections, and a white content well. Aperture
   supplies the two colours and nothing else.

   Fluid rather than the original's fixed 850px. That width existed because
   screens were 1024x768; reproducing it on a modern display is not period
   accuracy, it is a narrow strip of text between two fields of grey. Type is
   scaled up for the same reason: 11px was legible on a 15 inch CRT. */

body {
  background: #e8e8e8;
  color: #000;
  margin: 0;
  padding: 0;
  font-family: Verdana, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.6;
}

a { color: #964b00; }
a:visited { color: #6b3600; }
a:hover { color: #f28c1e; }

#page {
  width: 100%;
  background: #fff;
}

#masthead {
  background: #b35900;
  padding: 12px 18px;
  overflow: hidden;
}
#masthead img { float: left; margin-right: 12px; }
#masthead .name {
  color: #fff;
  font-size: 26px;
  font-weight: bold;
  text-decoration: none;
}
#masthead .sub { color: #f4e2c8; font-size: 14px; display: block; }

#bar {
  background: #f4e6d0;
  border-top: 1px solid #b35900;
  border-bottom: 1px solid #b35900;
  padding: 5px 18px;
  font-size: 13px;
}
#bar a { text-decoration: none; }
#bar a:hover { text-decoration: underline; }

#body { overflow: hidden; padding: 14px 18px 24px; }
#sidebar { float: left; width: 210px; }
#content { margin-left: 228px; }

.box { margin-bottom: 10px; }
.box .t {
  background: #b35900;
  color: #fff;
  font-size: 13px;
  font-weight: bold;
  padding: 3px 7px;
}
.box ul {
  margin: 0;
  padding: 6px 6px 7px 24px;
  background: #f4e6d0;
  border: 1px solid #d8bc94;
  border-top: 0;
  font-size: 13px;
  list-style: square;
}
.box li { margin: 2px 0; }
.box a { text-decoration: none; }
.box a:hover { text-decoration: underline; }

h1 {
  font-size: 26px;
  font-weight: bold;
  margin: 0 0 6px;
  padding-bottom: 3px;
  border-bottom: 1px solid #b35900;
}
h2 {
  font-size: 19px;
  font-weight: bold;
  margin: 18px 0 5px;
  padding-bottom: 2px;
  border-bottom: 1px solid #ccc;
}
h3 { font-size: 16px; font-weight: bold; margin: 16px 0 5px; }

p { margin: 8px 0; }
ul, ol { margin: 8px 0; padding-left: 22px; }
li { margin: 3px 0; }

/* The page fills the screen; the prose does not. At 1600px an uncapped
   paragraph runs to about 157 characters a line, and the eye loses its place
   on the way back to the left margin. Capped here rather than by narrowing the
   whole column, so tables, code and screenshots still get the full width. */
#main > p,
#main > ul,
#main > ol,
#main > dl,
#main > .note { max-width: 48em; }

code {
  font-family: "Courier New", monospace;
  background: #f4f4f4;
  border: 1px solid #ddd;
  padding: 0 3px;
  font-size: 13px;
}
pre {
  font-family: "Courier New", monospace;
  background: #f8f8f8;
  border: 1px solid #ccc;
  padding: 7px 9px;
  overflow-x: auto;
  margin: 12px 0;
  font-size: 13px;
  line-height: 1.45;
}
pre code { background: none; border: 0; padding: 0; }

.toc {
  display: table;
  background: #f9f9f9;
  border: 1px solid #ccc;
  padding: 7px 18px 7px 7px;
  margin: 12px 0;
  font-size: 13px;
}
.toc .h { font-weight: bold; text-align: center; margin-bottom: 3px; }
.toc ol { margin: 0; padding-left: 22px; }

.note {
  background: #f4e6d0;
  border: 1px solid #d8bc94;
  padding: 6px 9px;
  margin: 10px 0;
}

table.data { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
table.data th, table.data td {
  border: 1px solid #ccc;
  padding: 3px 7px;
  text-align: left;
  vertical-align: top;
}
table.data th { background: #f4e6d0; }

.infobox {
  float: right;
  width: 280px;
  border: 1px solid #ccc;
  background: #f9f9f9;
  margin: 0 0 12px 16px;
  font-size: 13px;
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

table.dl { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
table.dl th, table.dl td {
  border: 1px solid #ccc;
  padding: 4px 7px;
  text-align: left;
  vertical-align: top;
}
table.dl th { background: #f4e6d0; }
table.dl td.size { white-space: nowrap; }

.shot { margin: 12px 0; }
.shot img {
  display: block;
  border: 1px solid #999;
  width: 100%;
  max-width: 1280px;
  height: auto;
}
.shot .cap { font-size: 13px; color: #444; margin-top: 4px; }

/* Borrowed diagrams, floated right the way a 2005 wiki floated a thumbnail.
   Width is fixed rather than a percentage because these sit beside prose that
   is itself capped at 48em: a figure that grows with the viewport would swallow
   the column it is supposed to annotate. Diagrams get a white plate behind them
   -- most were drawn for a white page and several have transparent
   backgrounds, so on any other colour the linework disappears. */
figure.fig {
  float: right;
  width: 320px;
  margin: 4px 0 12px 18px;
  border: 1px solid #ccc;
  background: #f9f9f9;
  padding: 6px;
  font-size: 12px;
  line-height: 1.5;
}
figure.fig img {
  display: block;
  width: 100%;
  height: auto;
  background: #fff;
}
figure.fig figcaption { color: #444; margin-top: 5px; }
figure.fig .credit { display: block; color: #777; margin-top: 3px; font-size: 11px; }
figure.fig .credit a { color: #777; }

/* A figure wider than the text it interrupts, for the two diagrams that are
   unreadable at 320px. */
figure.fig.wide { float: none; width: auto; max-width: 48em; margin: 14px 0; }

.wikilist { list-style: square; }
.wikilist span { color: #555; }

.seealso { margin-top: 18px; padding-top: 6px; border-top: 1px solid #ccc; }
dl.faq dt { font-weight: bold; margin-top: 10px; }
dl.faq dd { margin: 3px 0 0 20px; }

#footer {
  clear: both;
  background: #f4e6d0;
  border-top: 1px solid #b35900;
  padding: 8px 18px;
  font-size: 12px;
  color: #555;
}
#footer p { margin: 3px 0; }

.crumbs { font-size: 13px; color: #555; margin-bottom: 8px; }
.crumbs a { color: #555; }

.listing { border-collapse: collapse; width: 100%; font-size: 13px; }
.listing th {
  border-bottom: 1px solid #999;
  text-align: left;
  padding: 2px 8px 2px 0;
}
.listing td { padding: 2px 8px 2px 0; border-bottom: 1px solid #eee; }
.icon { color: #777; display: inline-block; width: 2.6em; }

.skip { position: absolute; left: -9999px; }

@media (max-width: 45rem) {
  #sidebar { float: none; width: auto; }
  #content { margin-left: 0; }
  .infobox { float: none; width: 100%; margin-left: 0; }
  figure.fig { float: none; width: auto; margin-left: 0; margin-right: 0; }
  #body { padding: 10px; }
  #masthead, #bar, #footer { padding-left: 10px; padding-right: 10px; }
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
        elif kind == "fig":
            key, cap, alt, wide = val
            m = MEDIA.get(key)
            if m is None:
                # Skipped rather than raised: see-also links are structural and
                # should fail the build, a missing decoration is not.
                continue
            src = m["url"]
            bits = [b for b in (esc(m["author"]), esc(m["licence"])) if b]
            credit = ", ".join(bits) + (", " if bits else "") + \
                f'via <a href="{src}" rel="noopener">{esc(m["source"])}</a>'
            out.append(
                f'<figure class="fig{" wide" if wide else ""}">'
                f'<img src="{up}img/{m["file"]}" '
                f'alt="{html.escape(alt, quote=True)}" '
                f'width="{m["width"]}" height="{m["height"]}" '
                f'loading="lazy" decoding="async">'
                f"<figcaption>{cap}"
                f'<span class="credit">{esc(m["title"])}. {credit}.</span>'
                f"</figcaption></figure>")
        elif kind == "credits":
            rows = []
            for k, m in sorted(MEDIA.items(), key=lambda kv: kv[1]["title"].lower()):
                where = PAGES.get("wiki/" + m["page"]) or PAGES.get(m["page"])
                seen = (f'<a href="{url_for(where["slug"], slug)}">'
                        f'{esc(short_title(where["slug"]))}</a>') if where else "&mdash;"
                rows.append(
                    f'<tr><td><a href="{m["url"]}" rel="noopener">'
                    f'{esc(m["title"])}</a></td><td>{esc(m["author"])}</td>'
                    f'<td>{esc(m["licence"]) or "&mdash;"}</td>'
                    f'<td>{esc(m["source"])}</td><td>{seen}</td></tr>')
            out.append(
                '<table class="data"><thead><tr><th>Work</th><th>Author</th>'
                "<th>Licence</th><th>Source</th><th>Used on</th></tr></thead>"
                f'<tbody>{"".join(rows)}</tbody></table>')
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
                elif s in PAGES:
                    target = s
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
            "wiki/index": "Wiki", "index": "Home", "credits": "Credits",
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
  <p>Copyright 2026. All rights reserved. The source is published to be read
  rather than reused. One file (<code>src/dev/rtl8188eu_tables.rs</code>) is
  GPL-2.0 from the Linux kernel and carries its own terms; the diagrams are
  other people's and are listed on <a href="{url_for("credits", slug)}">the
  credits page</a>.</p>
  <p>No cookies, no analytics, no telemetry of any kind. Last updated {UPDATED}.</p>
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
              blocks=[p("There is no page at that address. This is one of the few "
                        "failures on this site that is definitely not a silent one."),
                      p('Try <a href="/GLaDOS/wiki/">the wiki index</a>, which lists '
                        'everything, or <a href="/GLaDOS/download/">the '
                        'downloads</a>, or <a href="/GLaDOS/">the front page</a>.')])
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
