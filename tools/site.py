#!/usr/bin/env python3
"""Generate the derived half of docs/ and check that the result is not lying.

Why this exists rather than a careful edit. The download page and the archive
listing carry facts about release artifacts -- file names, sizes, versions --
and those facts go stale every release. They have gone stale twice already and
nobody noticed either time: the download page was written against V1.1.0 asset
names, the archive against V1.0.0's, and by the time anyone looked all four ISO
links on the site returned 404 while the page still described them confidently.

That is a class of bug, not an instance. A site that states derived facts by
hand will state them wrongly, and the failure is silent because a stale link
looks exactly like a fresh one until somebody clicks it. So the derived regions
are emitted from the releases API, and `--check` fetches every download link
the site offers and fails on anything that does not answer.

The same argument covers the chrome. There is no templating in docs/; the nav
bar, sidebar and footer are duplicated inline across 34 files at two different
relative-path depths, which is 34 chances to update 33 of them.

    site.py --build            regenerate chrome and every derived region
    site.py --build --releases-json out/rel.json      without the network
    site.py --check            resolve every link, exit non-zero on any break

Follows the convention the rest of tools/ uses: say what could not be done
rather than skipping it quietly.
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.normpath(os.path.join(HERE, "..", "docs"))
REPO = "https://github.com/IlumCI/GLaDOS"
API = "https://api.github.com/repos/IlumCI/GLaDOS/releases"
SITE = "https://aperture.institute"

# --- what each image actually is ----------------------------------------
#
# Keyed by what is left of an asset name after "glados-", the version, and
# ".iso" come off. Both naming schemes appear in the release history: assets
# were unversioned through V1.1.0 and version-stamped from v1.2.26, which is
# precisely what broke the old links, so both are recognised here.
#
# `order` is the display order, cheapest-to-run first, because the image most
# people should take is the small one and a table sorted by size buries it.
VARIANTS = {
    "": dict(model="Qwen3-0.6B", ctx="512", kv="112 MiB", ram="2 GB", order=2,
             note="The middle size. Small enough to be quick, large enough to "
                  "write coherent sentences."),
    "qwen35-2b": dict(model="Qwen3.5-2B hybrid", ctx="512", kv="12 MiB", ram="5 GB", order=3,
                      note="The flagship. Three layers in four run as linear "
                           "attention, which is what keeps the cache small at "
                           "this size."),
    "qwen35-2b-8k": dict(model="Qwen3.5-2B hybrid", ctx="8192", kv="192 MiB", ram="5 GB", order=4,
                         note="The same weights with room for a long "
                              "conversation before the cache becomes a ring."),
    "qwen35-2b-32k": dict(model="Qwen3.5-2B hybrid", ctx="32768", kv="768 MiB", ram="8 GB", order=5,
                          note="The largest context this kernel will hold. The "
                               "cache alone is 768 MiB of heap."),
    "smollm2-135m": dict(model="SmolLM2-135M", ctx="512", kv="112 MiB", ram="2 GB", order=1,
                         note="Four times faster per token than Qwen3, and the "
                              "only image that fits QEMU's 516 MB disk ceiling. "
                              "The one to reach for under emulation."),
    "nomodel": dict(model="Kernel only", ctx="", kv="", ram="1 GB", order=0,
                    note="Kernel only. Boots to a desktop, reports that it has "
                         "no model, and everything except inference works."),
    # Pre-1.2.26 spellings, kept so the news page can describe old releases.
    "qwen3-0.6b": dict(model="Qwen3-0.6B", ctx="512", kv="112 MiB", ram="2 GB", order=2,
                       note="The middle size."),
}

# The project box, in the shape a forge categorised a project: a fixed list of
# axes, every project answering the same ones, so two projects can be compared
# without reading either description. Stated once here rather than in 31 files,
# which is the whole reason the chrome is generated.
#
# Development status uses the forge's own ladder (1 Planning through 7
# Inactive). Alpha is the honest rung: it ships releases and boots on real
# hardware, and it has no fault recovery and no isolation.
PROJECT = [
    ("Development status", "3 - Alpha"),
    ("Environment", "Console, framebuffer"),
    ("Intended audience", "Developers, science/research"),
    ("Licence", "Other/proprietary, all rights reserved"),
    ("Natural language", "English"),
    ("Operating system", "x86-64 UEFI, bare metal"),
    ("Programming language", "Rust, Python"),
    ("Topic", "Operating systems, kernels, artificial intelligence"),
]

# Who is on the project. One row, and the forge convention is to say the role
# rather than leave it implied.
DEVELOPERS = [("IlumCI", "project admin, developer")]

WIKI_PICKS = [
    ("wiki/glados-os.html", "GLaDOS OS"),
    ("wiki/kernel.html", "The kernel"),
    ("wiki/model.html", "The model"),
    ("wiki/agent.html", "The agent"),
    ("wiki/rsi.html", "Self-improvement"),
    ("wiki/aiksi.html", "Aiksi, the language"),
]

# "Summary" rather than "Home", which is what a forge called the page that
# describes a project. Download keeps its own name instead of the forge's
# "Files": it is the one label a visitor acts on, and clarity beats the period
# reference at exactly that spot.
NAV = [
    ("./", "Summary"),
    ("download/", "Download"),
    ("news/", "News"),
    ("wiki/", "Wiki"),
    ("screenshots/", "Screenshots"),
    ("archive/", "Archive"),
    ("token/", "Token"),
]


# --- releases -----------------------------------------------------------

def fetch_releases(path=None):
    """Every release, newest first, as the API returns them."""
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    req = urllib.request.Request(API, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "glados-site-generator",
    })
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def size_str(n):
    """MiB, labelled MB, and GB past a thousand of them.

    Matches how the release notes and the old page both spelled these, which
    matters more than the pedantry: a reader comparing the two should not have
    to work out that 1.8 GB and 1810 MB are the same file.
    """
    mib = n / (1024 * 1024)
    if mib >= 1024:
        return "%.1f GB" % (mib / 1024)
    return "%d MB" % round(mib)


def variant_key(asset_name, tag):
    """Strip the wrapper off an asset name to get its variant key.

    Returns None for anything that is not an ISO, and raises for an ISO whose
    shape is not recognised -- an unknown image is a new image somebody added,
    and silently dropping it from the download table is how the table went
    wrong in the first place.
    """
    n = asset_name
    if not n.endswith(".iso"):
        return None
    n = n[:-len(".iso")]
    if not n.startswith("glados"):
        raise ValueError("unrecognised image name: " + asset_name)
    n = n[len("glados"):].lstrip("-")
    # Version-stamped from v1.2.26 onward; drop a leading N.N.N if present.
    ver = tag.lstrip("vV")
    if n == ver:
        n = ""
    elif n.startswith(ver + "-"):
        n = n[len(ver) + 1:]
    else:
        n = re.sub(r"^\d+\.\d+\.\d+-?", "", n)
    if n not in VARIANTS:
        raise ValueError(
            "image %r in %s has no entry in VARIANTS; add one rather than "
            "letting it fall off the download page" % (asset_name, tag))
    return n


def images_of(rel):
    """The ISO assets of one release, in display order."""
    out = []
    for a in rel.get("assets", []):
        k = variant_key(a["name"], rel["tag_name"])
        if k is None:
            continue
        v = dict(VARIANTS[k])
        v.update(name=a["name"], size=a["size"],
                 url=a["browser_download_url"], key=k)
        out.append(v)
    out.sort(key=lambda v: v["order"])
    return out


def sha_url(rel):
    for a in rel.get("assets", []):
        if a["name"] == "SHA256SUMS":
            return a["browser_download_url"]
    return None


def version_of(rel):
    return rel["tag_name"].lstrip("vV")


def date_of(rel):
    return (rel.get("published_at") or "")[:10]


# --- chrome -------------------------------------------------------------

def prefix_for(relpath):
    """How many ../ it takes to get from a page back to docs/."""
    depth = relpath.replace("\\", "/").count("/")
    return "../" * depth


def a(href, text, extra=""):
    return '<a href="%s"%s>%s</a>' % (href, extra, html.escape(text))


def masthead_html(p):
    """The wordmark, and the product name under it.

    The mark reads "Aperture Institute", which is the organisation rather than
    the software, so the tagline has to carry "GLaDOS" or the page stops saying
    what it is about. That split is deliberate and it is the one the domain
    already implies: the institute is the publisher, GLaDOS is the thing
    published.

    Drawn on white because the artwork is black on transparent and was made
    for a light ground. Recolouring somebody's mark to fit a band we chose is
    the wrong way round; the band moves instead. The amber has not gone
    anywhere -- it is the tab strip's borders, every section bar, every link
    and the footer.
    """
    home = p or "./"
    return "\n".join([
        '<div id="masthead">',
        '  <a class="mark" href="%s"><img src="%simg/wordmark.png" '
        'alt="Aperture Institute" width="666" height="169"></a>' % (home, p),
        '  <span class="sub">GLaDOS: an operating system in Rust, '
        'with a language model in the kernel</span>',
        '</div>',
    ])


def section_of(relpath):
    """Which tab a page belongs under, from its own path."""
    rl = relpath.replace("\\", "/")
    if "/" not in rl:
        return "./"                       # index.html, credits.html, 404.html
    return rl.split("/", 1)[0] + "/"


def nav_html(p, relpath):
    """The tab strip.

    A forge put tabs here rather than a line of links, and the active one is
    the part that does the work: it tells a reader which of eight areas they
    are standing in, which a pipe-separated list cannot. The active tab is
    drawn connected to the content below it, so the strip reads as the top
    edge of the page rather than as a separate widget.
    """
    here = section_of(relpath)
    items = []
    for href, label in NAV:
        on = ' class="on"' if href == here else ""
        items.append("<li%s>%s</li>"
                     % (on, a(p + href if href != "./" else (p or "./"), label)))
    items.append("<li>%s</li>" % a(REPO, "Source code", ' rel="noopener"'))
    return '<div id="tabs"><ul>%s</ul></div>' % "".join(items)


def stats_rows(releases):
    """The statistics a forge put on every project page.

    Downloads are the real count from the releases API rather than an estimate
    or an omission. It is a small number today and publishing it is the point:
    a figure that only appears once it is flattering is not a statistic, it is
    an advertisement, and nothing else on this site works that way.
    """
    total = 0
    for r in releases:
        for asset in r.get("assets", []):
            total += asset.get("download_count", 0)
    first = date_of(releases[-1]) if releases else ""
    return [
        ("Registered", first),
        ("Releases", "%d" % len(releases)),
        ("Downloads", "%d" % total),
        ("Source", "108 files, ~50,000 lines"),
    ]


def sidebar_html(p, rel, releases):
    """The forge's own column: what this release is, how the project is
    categorised, how much of it there is, who is on it, and the way in.

    It began as five boxes and 32 links, three of them wiki article lists --
    a table of contents, which is the right sidebar for a documentation site
    and the wrong one for a project page. The navigation list went with the
    tab strip, which now carries it and marks where the reader is standing;
    duplicating eight links directly under eight tabs said nothing twice.
    """
    b = []

    if rel:
        imgs = images_of(rel)
        rows = [
            ("Version", html.escape(version_of(rel))),
            ("Released", html.escape(date_of(rel))),
            ("Images", "%d" % len(imgs)),
        ]
        tbl = "".join("<tr><td>%s</td><td>%s</td></tr>" % r for r in rows)
        links = "<ul><li>%s</li><li>%s</li></ul>" % (
            a(p + "download/", "Download"),
            a(rel["html_url"], "Release notes"),
        )
        b.append(('Latest release',
                  '<table class="facts">%s</table>%s' % (tbl, links)))

    tbl = "".join("<tr><td>%s</td><td>%s</td></tr>"
                  % (html.escape(k), html.escape(v)) for k, v in PROJECT)
    b.append(('Project information', '<table class="facts">%s</table>' % tbl))

    tbl = "".join("<tr><td>%s</td><td>%s</td></tr>"
                  % (html.escape(k), html.escape(v))
                  for k, v in stats_rows(releases))
    b.append(('Statistics', '<table class="facts">%s</table>' % tbl))

    tbl = "".join("<tr><td>%s</td><td>%s</td></tr>"
                  % (html.escape(who), html.escape(role))
                  for who, role in DEVELOPERS)
    b.append(('Developers', '<table class="facts">%s</table>' % tbl))

    # Counted rather than written down. It said 25 while there were 23, which
    # is what a hand-maintained number does the first time an article is
    # merged into another.
    n = len([f for f in os.listdir(os.path.join(DOCS, "wiki"))
             if f.endswith(".html") and f != "index.html"])
    picks = "".join("<li>%s</li>" % a(p + h, t) for h, t in WIKI_PICKS)
    picks += "<li>%s</li>" % a(p + "wiki/", "All %d articles" % n)
    b.append(('Documentation', "<ul>%s</ul>" % picks))

    # One panel per box rather than one per element. The release box carries a
    # fact table and a list, and styling each of those as its own bordered
    # panel would stack two frames inside one heading.
    out = ['<div id="sidebar">']
    for title, body in b:
        out.append('<div class="box"><div class="t">%s</div>'
                   '<div class="bd">%s</div></div>' % (title, body))
    out.append("</div>")
    return "\n".join(out)


def footer_html(p, rel):
    """The trademark line no longer says non-commercial.

    It said "an independent, non-commercial homage". A token makes the project
    commercial, so that clause became untrue, and a disclaimer with a false
    clause in it is worse than a shorter one. The rest of the sentence -- not
    affiliated, not endorsed, not connected -- is what was doing the work.
    """
    updated = date_of(rel) if rel else ""
    return "\n".join([
        '<div id="footer">',
        '  <p>GLaDOS, Aperture Science and Portal are properties of Valve '
        'Corporation. This project is independent and is not affiliated with, '
        'endorsed by, or connected to Valve in any way.</p>',
        '  <p>Copyright 2026. All rights reserved. The source is published so '
        'it can be read; redistribution and derivatives need asking. '
        'One file (<code>src/dev/'
        'rtl8188eu_tables.rs</code>) is GPL-2.0 from the Linux kernel and '
        'carries its own terms; the diagrams are other people\'s and are '
        'listed on %s.</p>' % a(p + "credits.html", "the credits page"),
        '  <p>No cookies, no analytics, no telemetry of any kind.%s</p>'
        % (" Last updated %s." % updated if updated else ""),
        '</div>',
    ])


# --- region replacement -------------------------------------------------
#
# Markers make later runs unambiguous. The first run has none to find, so each
# region also carries a pattern matching the hand-written form it is replacing,
# and the replacement inserts the markers on the way past.

REGIONS = {
    "masthead": r'<div id="masthead">.*?\n</div>\n',
    "bar": r'<div id="bar">.*?</div>[ \t]*\n',
    "sidebar": r'<div id="sidebar">.*?</div>\n(?=<div id="content">)',
    "footer": r'<div id="footer">.*?\n</div>\n',
}


def replace_region(text, name, new):
    """Swap one chrome region, by marker if present and by shape if not."""
    block = "<!--%s-->\n%s\n<!--/%s-->\n" % (name, new, name)
    marker = re.compile(r"<!--%s-->.*?<!--/%s-->\n" % (name, name), re.S)
    if marker.search(text):
        return marker.sub(lambda m: block, text, count=1), True
    pat = re.compile(REGIONS[name], re.S)
    if pat.search(text):
        return pat.sub(lambda m: block, text, count=1), True
    return text, False


def pages():
    for root, _dirs, files in os.walk(DOCS):
        for f in sorted(files):
            if f.endswith(".html"):
                full = os.path.join(root, f)
                yield full, os.path.relpath(full, DOCS).replace("\\", "/")


def current(releases):
    """The release the site should present as the current one.

    The newest release that actually carries images, which is not always the
    newest release. The kernel image and its manifest are published by CI the
    moment a tag lands; the ISOs are built by hand afterwards, so there is a
    window where the latest release has no downloads in it. Taking
    `releases[0]` blindly during that window empties the download table and
    the front page, and the site then says the project has nothing to
    download -- which is how it went out once already.

    A release with no images is reported, so a release that never gets its
    ISOs uploaded is visible instead of silently skipped.
    """
    for i, r in enumerate(releases):
        if images_of(r):
            if i:
                print("  note: %s has no images yet; presenting %s"
                      % (releases[0]["tag_name"], r["tag_name"]))
            return r
    print("  warning: no release has images; the download table will be empty")
    return releases[0]


def build_chrome(releases):
    rel = current(releases)
    changed, missed = 0, []
    for full, rl in pages():
        p = prefix_for(rl)
        with open(full, "r", encoding="utf-8") as fh:
            text = orig = fh.read()
        for name, new in (("masthead", masthead_html(p)),
                          ("bar", nav_html(p, rl)),
                          ("sidebar", sidebar_html(p, rel, releases)),
                          ("footer", footer_html(p, rel))):
            text, ok = replace_region(text, name, new)
            if not ok:
                missed.append("%s: %s" % (rl, name))
        if text != orig:
            with open(full, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            changed += 1
    print("chrome: %d of %d pages rewritten" % (changed, len(list(pages()))))
    for m in missed:
        print("  no region found -- %s" % m)
    return not missed


# --- derived regions ----------------------------------------------------

def download_table(rel):
    rows = []
    for v in images_of(rel):
        ctx = v["ctx"] or "n/a"
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td class=\"size\">%s</td>"
            "<td>%s</td></tr>" % (
                a(v["url"], v["name"]),
                html.escape(v["model"]),
                ctx,
                size_str(v["size"]),
                html.escape(v["note"]),
            ))
    return ('<table class="dl"><thead><tr><th>File</th><th>Model</th>'
            '<th>Context</th><th>Size</th><th>Notes</th></tr></thead>'
            '<tbody>%s</tbody></table>' % "".join(rows))


def releases_table(rel):
    """The forge's Latest File Releases block, for the front page."""
    rows = []
    for v in images_of(rel):
        rows.append('<tr><td>%s</td><td>%s</td><td class="size">%s</td>'
                    '<td>%s</td></tr>' % (
                        html.escape(v["model"]),
                        html.escape(v["ctx"]) or "n/a",
                        size_str(v["size"]),
                        a(v["url"], "Download")))
    return ('<table class="forge"><thead><tr><th>Model</th><th>Context</th>'
            '<th>Size</th><th>&nbsp;</th></tr></thead><tbody>%s</tbody>'
            '</table>' % "".join(rows))


def archive_tree(rel):
    """The archive's TREE literal, regenerated.

    The listing is a client-side fake of an Apache index and is period-correct,
    so only its data is replaced, never its script.
    """
    isos = []
    for v in images_of(rel):
        isos.append("      {n:%s,t:'f',d:%s,\n       href:%s,size:%s}" % (
            json.dumps(v["name"]),
            json.dumps("kernel only" if v["key"] == "nomodel"
                       else "kernel + " + v["model"]),
            json.dumps(v["url"]),
            json.dumps(size_str(v["size"]).replace(" ", ""))))
    sha = sha_url(rel)
    return "\n".join([
        "  var TREE={",
        "    '':{kids:[",
        "      {n:'iso/',t:'dir',d:'bootable images'},",
        "      {n:'src/',t:'dir',d:'source tree'},",
        "      {n:'docs/',t:'dir',d:'documentation'},",
        "      {n:'checksums/',t:'dir',d:'digests and verification'}]},",
        "    'iso':{kids:[",
        ",\n".join(isos) + "]},",
        "    'src':{kids:[",
        "      {n:'glados-src.tar.gz',t:'f',d:'repository snapshot',",
        "       href:REPO+'/archive/refs/heads/main.tar.gz',size:'700K'},",
        "      {n:'browse/',t:'l',d:'read it on GitHub',href:REPO}]},",
        "    'docs':{kids:[",
        "      {n:'README.md',t:'f',d:'overview and build instructions',",
        "       href:REPO+'/blob/main/README.md',size:'13K'},",
        "      {n:'wiki/',t:'l',d:'the documentation wiki',href:'../wiki/'}]},",
        "    'checksums':{kids:[",
        "      {n:'SHA256SUMS',t:'f',d:'digests for every image',",
        "       href:%s,size:'562'}]}" % json.dumps(sha or ""),
        "  };",
    ])


def news_items(releases, limit=None):
    """Dated entries, newest first. The body is the release's own summary line.

    Deliberately not the whole release note: those run to a page each, and a
    news column that reproduces them is not a news column.
    """
    out = []
    for rel in releases[:limit]:
        ver, date = version_of(rel), date_of(rel)
        body = (rel.get("body") or "").strip()
        # The first paragraph of a release note is written as its summary.
        first = ""
        for para in body.split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                first = " ".join(para.split())
                break
        if len(first) > 320:
            first = first[:317].rsplit(" ", 1)[0] + "..."
        imgs = len(images_of(rel))
        out.append(
            '<div class="news">\n'
            '  <div class="nh"><span class="nd">%s</span> %s</div>\n'
            '  <p>%s</p>\n'
            '  <p class="nl">%s &middot; %s</p>\n'
            '</div>' % (
                html.escape(date),
                html.escape(rel.get("name") or ("GLaDOS " + ver)),
                html.escape(first) or "No summary was recorded for this release.",
                a(rel["html_url"], "Release notes"),
                "%d image%s" % (imgs, "" if imgs == 1 else "s"),
            ))
    return "\n".join(out)


def ld_app(rel):
    """The SoftwareApplication block, which carries a version number.

    Generated for the same reason the download table is: it said 0.1 while the
    project shipped 1.2.27, and a structured-data block is exactly the kind of
    thing nobody re-reads. The markers sit outside the script tag, because an
    HTML comment inside one would land in the JSON and break it.
    """
    doc = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "GLaDOS",
        "applicationCategory": "OperatingSystem",
        "operatingSystem": "x86-64 UEFI (bare metal)",
        "description": "GLaDOS is a from-scratch ring-0 operating system "
                       "written in Rust with a language model running inside "
                       "the kernel. Free bootable ISO, full source, and "
                       "documentation on how every part works.",
        "url": SITE + "/",
        "downloadUrl": SITE + "/download/",
        "softwareVersion": version_of(rel),
        "datePublished": date_of(rel),
        "programmingLanguage": "Rust",
        "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
    }
    return ('<script type="application/ld+json">%s</script>'
            % json.dumps(doc, separators=(",", ":")))


def flash_example(rel):
    """The dd line, naming an image that exists.

    Hardcoded prose is exactly the failure this generator was written for. The
    version sat at 1.2.27 in a code block a reader is meant to copy, which is
    the one place a stale name costs somebody a download that 404s.

    Names the default image, which is what the prose around it recommends for
    real hardware, and falls back to the tag when a release has no images so
    the block never reads "None".
    """
    imgs = [v for v in images_of(rel) if v["key"] == ""]
    name = imgs[0]["name"] if imgs else "glados-%s.iso" % version_of(rel)
    body = [
        "# Linux / macOS. Check the device name first, this overwrites it",
        "sudo dd if=%s of=/dev/sdX bs=4M status=progress oflag=sync"
        % html.escape(name),
    ]
    return "<pre><code>" + chr(10).join(body) + "</code></pre>"


DERIVED = {
    "download-table": download_table,
    "releases-table": releases_table,
    "archive-tree": archive_tree,
    "flash-example": flash_example,
    "ld-app": ld_app,
}


def build_sitemap(rel):
    """Emit sitemap.xml from the pages that actually exist.

    Hand-maintained, it listed every URL with a lastmod of 2026-08-14 and knew
    nothing about any page added since. Walking the directory means a new page
    is in the sitemap by virtue of existing, which is the only way this stays
    true.
    """
    when = date_of(rel)
    urls = []
    for _full, rl in pages():
        if rl == "404.html":                    # not a destination
            continue
        loc = SITE + "/" + ("" if rl == "index.html"
                            else rl[:-len("index.html")] if rl.endswith("/index.html")
                            else rl)
        if rl == "index.html":
            pri = "1.0"
        elif rl.startswith("wiki/") and rl != "wiki/index.html":
            pri = "0.6"
        else:
            pri = "0.8"
        urls.append("<url><loc>%s</loc><lastmod>%s</lastmod>"
                    "<changefreq>weekly</changefreq><priority>%s</priority>"
                    "</url>" % (html.escape(loc), when, pri))
    doc = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           + "".join(urls) + "</urlset>\n")
    with open(os.path.join(DOCS, "sitemap.xml"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    print("sitemap: %d urls" % len(urls))
    return True


def build_derived(releases):
    rel = current(releases)
    filled, missing = 0, []
    for full, rl in pages():
        with open(full, "r", encoding="utf-8") as fh:
            text = orig = fh.read()
        for name, fn in DERIVED.items():
            pat = re.compile(r"<!--%s-->.*?<!--/%s-->" % (name, name), re.S)
            if pat.search(text):
                body = fn(rel)
                # The body goes on its own lines, never sharing one with a
                # marker. `<!--` opens a legacy line comment in JavaScript, so
                # a marker sharing a line with the archive's `var TREE={`
                # commented out the declaration and left the object literal
                # after it as a syntax error -- an empty listing and a broken
                # page, from punctuation that is inert in HTML.
                text = pat.sub(
                    lambda m: "<!--{0}-->\n{1}\n<!--/{0}-->".format(name, body),
                    text, count=1)
                filled += 1
        for name, n in (("news-latest", 3), ("news-all", None)):
            pat = re.compile(r"<!--%s-->.*?<!--/%s-->" % (name, name), re.S)
            if pat.search(text):
                body = news_items(releases, n)
                text = pat.sub(
                    lambda m: "<!--%s-->\n%s\n<!--/%s-->" % (name, body, name),
                    text, count=1)
                filled += 1
        if text != orig:
            with open(full, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
    print("derived: %d region(s) filled" % filled)
    return True


# --- checking -----------------------------------------------------------

HREF = re.compile(r'(?:href|src)="([^"]+)"')


LATEST_DL = re.compile(r"/releases/latest/download/(?P<name>[^/?#]+)$")
TAG_DL = re.compile(r"/releases/download/(?P<tag>[^/]+)/(?P<name>[^/?#]+)$")
TAG_PAGE = re.compile(r"/releases/tag/(?P<tag>[^/?#]+)$")


def check(releases, fetch=False):
    """Resolve every link the site offers.

    Internal links are checked against the filesystem. Release links are
    checked against the releases API -- not by fetching them, which is how the
    first version worked and was a mistake worth recording: GitHub counts a
    request for an asset as a download, including a one-byte range request, so
    a checker that fetched every image inflated the very download figure the
    sidebar publishes. Three runs put the count from 5 to 19.

    Resolving against the API is also the better check. The URLs are generated
    from that API, so the question worth asking is whether a page names an
    asset the release actually has, and that is what caught the original
    breakage: `glados-qwen35-2b.iso` is absent from the current release's asset
    list, which is exactly why it 404s. It costs one request instead of one per
    link, and it cannot be fooled by a CDN that answers for a missing object.

    `--fetch` does the end-to-end version anyway, for when that is what is
    wanted. It is off by default because it is not free.
    """
    latest_names = {a["name"] for a in releases[0].get("assets", [])}
    tags = {r["tag_name"] for r in releases}
    asset_urls = {a["browser_download_url"]
                  for r in releases for a in r.get("assets", [])}
    page_urls = {r["html_url"] for r in releases}
    by_tag = {r["tag_name"]: {a["name"] for a in r.get("assets", [])}
              for r in releases}

    bad, ext_ok, int_ok, fetched = [], 0, 0, 0
    seen = set()
    for full, rl in pages():
        with open(full, "r", encoding="utf-8") as fh:
            text = fh.read()
        base = os.path.dirname(full)
        for href in HREF.findall(text):
            if href.startswith(("mailto:", "data:", "#")):
                continue
            if href.startswith(("http://", "https://")):
                if "/releases/" not in href or href in seen:
                    continue
                seen.add(href)
                m = LATEST_DL.search(href)
                mt = TAG_DL.search(href)
                mp = TAG_PAGE.search(href)
                if m:
                    ok = m.group("name") in latest_names
                    why = "not an asset of the latest release"
                elif mt:
                    ok = (mt.group("tag") in by_tag
                          and mt.group("name") in by_tag[mt.group("tag")])
                    why = "not an asset of that release"
                elif mp:
                    ok, why = mp.group("tag") in tags, "no such release tag"
                else:
                    ok = href in asset_urls or href in page_urls
                    why = "unrecognised release URL"
                if ok:
                    ext_ok += 1
                else:
                    bad.append("%s -> %s  (%s)" % (href, why, rl))
                    continue
                if fetch:
                    try:
                        req = urllib.request.Request(
                            href, headers={"Range": "bytes=0-0",
                                           "User-Agent": "glados-site-check"})
                        with urllib.request.urlopen(req, timeout=30) as r:
                            if r.status not in (200, 206):
                                bad.append("%s -> HTTP %d  (%s)"
                                           % (href, r.status, rl))
                            else:
                                fetched += 1
                    except urllib.error.HTTPError as e:
                        bad.append("%s -> HTTP %d  (%s)" % (href, e.code, rl))
                    except Exception as e:                   # noqa: BLE001
                        bad.append("%s -> %s  (%s)" % (href, e, rl))
                continue
            # A root-absolute href resolves against docs/, not against the page
            # holding it. 404.html is the reason: it is served in place of any
            # missing URL at any depth, so its links have to be absolute, and
            # resolving them relative to the file would call every one broken.
            path = href.split("#")[0]
            if path.startswith("/"):
                target = os.path.normpath(os.path.join(DOCS, path.lstrip("/")))
            else:
                target = os.path.normpath(os.path.join(base, path))
            if href.endswith("/") or os.path.isdir(target):
                target = os.path.join(target, "index.html")
            if href.split("#")[0] == "":
                continue
            if os.path.exists(target):
                int_ok += 1
            else:
                bad.append("%s -> missing  (%s)" % (href, rl))

    print("check: %d internal ok, %d release links ok%s, %d broken"
          % (int_ok, ext_ok,
             ", %d fetched" % fetched if fetch else "", len(bad)))
    for b in bad:
        print("  BROKEN  " + b)
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="regenerate chrome and every derived region")
    ap.add_argument("--check", action="store_true",
                    help="resolve every link; non-zero exit on any break")
    ap.add_argument("--releases-json",
                    help="read releases from a file instead of the API")
    ap.add_argument("--fetch", action="store_true",
                    help="with --check, also fetch each release asset. Off by "
                         "default: GitHub counts a fetch as a download, so "
                         "this moves the figure the site publishes")
    args = ap.parse_args()
    if not (args.build or args.check):
        ap.error("pick --build or --check")

    ok = True
    rels = fetch_releases(args.releases_json)
    if not rels:
        print("no releases returned; refusing to blank the site")
        return 1
    print("releases: %d, latest %s (%s)"
          % (len(rels), rels[0]["tag_name"], date_of(rels[0])))
    if args.build:
        ok &= build_chrome(rels)
        ok &= build_derived(rels)
        ok &= build_sitemap(rels[0])
    if args.check:
        ok &= check(rels, fetch=args.fetch)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
