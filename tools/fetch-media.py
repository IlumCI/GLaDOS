#!/usr/bin/env python3
"""Fetch the third-party diagrams the documentation site embeds.

Why a script rather than a folder of files someone dropped in
------------------------------------------------------------
Every image here came from somewhere, and the somewhere has to survive. A file
copied into `docs/img/` by hand loses its author, its licence and its source
URL the moment the person who copied it stops remembering, and the credits page
then says whatever it said the last time anyone edited it. So the manifest
below is the record: `mksite.py` reads the JSON this emits and renders the
credit line from it, which means a caption cannot drift away from the file it
describes without the build noticing.

Two sources.

Wikimedia Commons entries fetch through the API, which hands back the author
and licence alongside the image. Note that the `/thumb/` URLs no longer serve
arbitrary widths -- Commons now 400s anything outside a standard set -- so this
pulls the original and resizes locally.

ar5iv entries pull a figure out of the HTML rendering of an arXiv paper. There
is no metadata API for those, so the manifest carries the citation.

Pillow is imported only to downscale rasters. It is a dependency of this
script, not of the site: `mksite.py` stays stdlib-only, and the images this
writes are committed, so nobody needs either one to build the site.

Usage:
    fetch-media.py [--out docs/img] [--credits tools/media-credits.json]
    fetch-media.py --only merkle-tree      # re-fetch one entry
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

UA = "GLaDOS-docs/1.0 (https://github.com/IlumCI/GLaDOS; research@euroswarms.eu)"
API = "https://commons.wikimedia.org/w/api.php"
AR5IV = "https://ar5iv.labs.arxiv.org/html/"

# Raster images are resized to this width. The content column caps prose at
# 48em and figures render at 320px, so anything wider is bytes the reader
# downloads and never sees. SVGs are left alone; they have no width.
RASTER_WIDTH = 720

# --- the manifest ---------------------------------------------------------
#
# key -> where it came from, and what page wants it. The `page` field is
# documentation for whoever reads this file; mksite.py places figures itself.

COMMONS = {
    "priv-rings":       ("File:Priv rings.svg", "ring-0"),
    "ip-stack":         ("File:IP stack connections.svg", "network-stack"),
    "tcp-states":       ("File:Tcp state diagram fixed new.svg", "network-stack"),
    "udp-encap":        ("File:UDP encapsulation.svg", "network-stack"),
    "chain-of-trust":   ("File:Chain Of Trust.svg", "tls"),
    "dh-exchange":      ("File:Diffie-Hellman Key Exchange.svg", "tls"),
    "merkle-tree":      ("File:Hash Tree.svg", "storage"),
    "nvme-ssd":         ("File:Samsung 980 PRO PCIe 4.0 NVMe SSD 1TB-top PNr°0915.jpg", "storage"),
    "dfa":              ("File:DFA example multiplies of 3.svg", "constrained-decoding"),
    "uefi-logo":        ("File:Uefi logo.svg", "uefi-kernel"),
    "wifi-dongle":      ("File:Tp-link usb wi-fi dongle tl-wn821n.jpg", "usb-wifi-driver"),
    "usb3-contacts":    ("File:USB 3.0 (zoom contacts).svg", "usb-xhci"),
    "templeos":         ("File:VirtualBox TempleOS x64 27 02 2021 20 43 48.png", "templeos"),
    "linear-regression": ("File:Linear regression.svg", "routing"),
    "compact-disc":     ("File:OD Compact disc.svg", "iso-el-torito"),
    "rust-logo":        ("File:Rust programming language black logo.svg", "rust-os"),
}

# key -> (arXiv id, asset path inside the ar5iv render, figure label,
#         paper title, authors, page)
AR5IV_FIGS = {
    "rope-diagram": (
        "2104.09864", "assets/roformer_RoPE_v2.svg", "Figure 1",
        "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        "Su, Lu, Pan, Murtadha, Wen and Liu", "rope"),
    "transformer-arch": (
        "1706.03762", "assets/Figures/ModalNet-21.png", "Figure 1",
        "Attention Is All You Need",
        "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin",
        "llm-in-kernel"),
    "attention-heads": (
        "1706.03762", "assets/Figures/ModalNet-20.png", "Figure 2 (right)",
        "Attention Is All You Need",
        "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin",
        "qwen3"),
    "attention-sinks": (
        "2309.17453", "assets/attention_weights.png", "Figure 2",
        "Efficient Streaming Language Models with Attention Sinks",
        "Xiao, Tian, Chen, Han and Lewis", "kv-cache"),
    "streaming-kv": (
        "2309.17453", "assets/scheme.png", "Figure 1",
        "Efficient Streaming Language Models with Attention Sinks",
        "Xiao, Tian, Chen, Han and Lewis", "kv-cache"),
}


def get(url, binary=False):
    """curl rather than urllib: the proxy in front of this environment is
    configured for it, and urllib picked up 429s that curl does not."""
    # --max-time is not optional. Without it a connection that stalls rather
    # than fails hangs the whole run with no output, which is what happened
    # here often enough to be worth a comment: the retry logic never gets a
    # chance to fire because curl is still politely waiting.
    cmd = ["curl", "-sSL", "--connect-timeout", "15", "--max-time", "90",
           "--retry", "3", "--retry-delay", "3", "-A", UA, url]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr.decode()[:200]}")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def commons_info(titles):
    """Batch the metadata query. Commons rate-limits hard enough that one
    request per file gets throttled halfway through a manifest this size."""
    q = urllib.parse.urlencode({
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "format": "json",
        "iiextmetadatafilter": "LicenseShortName|Artist|Credit",
    })
    for attempt in range(5):
        try:
            return json.loads(get(API + "?" + q))["query"]["pages"]
        except Exception:
            time.sleep(4 * (attempt + 1))
    raise RuntimeError("Commons API kept failing; try again in a minute")


def strip_html(s):
    return " ".join(re.sub(r"<[^>]+>", "", s or "").split())


def shrink(path):
    """Downscale a raster in place. SVGs and anything already narrow are left
    as they are."""
    if path.suffix.lower() == ".svg":
        return None
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if w <= RASTER_WIDTH:
            return (w, h)
        nh = round(h * RASTER_WIDTH / w)
        im = im.convert("RGBA") if im.mode == "P" else im
        im = im.resize((RASTER_WIDTH, nh), Image.LANCZOS)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            im.convert("RGB").save(path, quality=82, optimize=True)
        else:
            im.save(path, optimize=True)
        return (RASTER_WIDTH, nh)


def svg_size(path):
    """Pull width/height out of an SVG so the HTML can carry them and the page
    does not reflow as figures load.

    Only the root <svg> element is inspected. Scanning the first N bytes
    instead reads whichever viewBox comes first in the file, which on a drawing
    with an embedded symbol or pattern is not the drawing's -- two diagrams here
    came back as 135x101 that way."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<svg\b[^>]*>", text, re.S | re.I)
    root = m.group(0) if m else text[:2000]
    vb = re.search(r'viewBox\s*=\s*"\s*([\d.\-+eE]+)[ ,]+([\d.\-+eE]+)[ ,]+'
                   r'([\d.\-+eE]+)[ ,]+([\d.\-+eE]+)', root)
    w = re.search(r'\swidth\s*=\s*"([\d.]+)', root)
    h = re.search(r'\sheight\s*=\s*"([\d.]+)', root)
    if w and h:
        return (round(float(w.group(1))), round(float(h.group(1))))
    if vb:
        return (round(float(vb.group(3))), round(float(vb.group(4))))
    return (640, 400)


def fetch_commons(out, only):
    creds = {}
    keys = [k for k in COMMONS if not only or k in only]
    for i in range(0, len(keys), 8):
        batch = keys[i:i + 8]
        pages = commons_info([COMMONS[k][0] for k in batch])
        by_title = {p["title"]: p for p in pages.values()}
        for k in batch:
            title = COMMONS[k][0]
            # The API normalises underscores and some punctuation, so match on
            # the normalised title it hands back rather than the one we sent.
            pg = by_title.get(title) or by_title.get(title.replace("_", " "))
            if pg is None or "missing" in pg:
                print(f"  MISSING  {title}", file=sys.stderr)
                continue
            ii = pg["imageinfo"][0]
            em = ii.get("extmetadata", {})
            # The API appends utm_* query parameters to the file URL, which
            # makes the last path segment stop looking like a filename.
            clean = urllib.parse.urlsplit(ii["url"]).path
            ext = Path(urllib.parse.unquote(clean)).suffix.lower()
            dst = out / "wm" / (k + ext)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(get(ii["url"], binary=True))
            # Prefer the API's dimensions for anything not resized: for an SVG
            # it reports the size the drawing actually renders at, which beats
            # guessing from the markup.
            size = shrink(dst) or (ii.get("width"), ii.get("height"))
            if not all(size):
                size = svg_size(dst)
            creds[k] = {
                "file": f"wm/{dst.name}",
                "width": size[0], "height": size[1],
                "source": "Wikimedia Commons",
                "title": pg["title"].removeprefix("File:"),
                "author": strip_html(em.get("Artist", {}).get("value", "Unknown")),
                "licence": strip_html(em.get("LicenseShortName", {}).get("value", "see source")),
                "url": ii.get("descriptionurl", ""),
                "page": COMMONS[k][1],
            }
            print(f"  {k:20} {dst.name:34} {size[0]}x{size[1]}")
        time.sleep(2)
    return creds


def fetch_ar5iv(out, only):
    creds = {}
    for k, (aid, asset, label, paper, authors, page) in AR5IV_FIGS.items():
        if only and k not in only:
            continue
        url = f"{AR5IV}{aid}/{asset}"
        data = get(url, binary=True)
        if data[:15].lstrip().startswith(b"<!DOCTYPE html") or len(data) < 512:
            print(f"  MISSING  {url}", file=sys.stderr)
            continue
        dst = out / "paper" / (k + Path(asset).suffix.lower())
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        size = shrink(dst) or svg_size(dst)
        creds[k] = {
            "file": f"paper/{dst.name}",
            "width": size[0], "height": size[1],
            "source": "arXiv " + aid,
            "title": f"{label}, {paper}",
            "author": authors,
            "licence": "",
            "url": f"https://arxiv.org/abs/{aid}",
            "page": page,
        }
        print(f"  {k:20} {dst.name:34} {size[0]}x{size[1]}")
        time.sleep(1)
    return creds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--credits", default="tools/media-credits.json")
    ap.add_argument("--only", nargs="*", default=None,
                    help="fetch just these keys, and merge into the credits file")
    args = ap.parse_args()
    out = Path(args.out)
    only = set(args.only) if args.only else None

    print("Wikimedia Commons:")
    creds = fetch_commons(out, only)
    print("ar5iv:")
    creds.update(fetch_ar5iv(out, only))

    cf = Path(args.credits)
    if only and cf.exists():
        merged = json.loads(cf.read_text(encoding="utf-8"))
        merged.update(creds)
        creds = merged
    cf.write_text(json.dumps(dict(sorted(creds.items())), indent=1,
                             ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(creds)} images -> {out}/, credits -> {cf}")


if __name__ == "__main__":
    main()
