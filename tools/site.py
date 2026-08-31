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

# The facts in the sidebar's project box. Stated once here rather than in 34
# files, which is the whole reason the chrome is generated.
PROJECT = [
    ("Status", "Active, research kernel"),
    ("Language", "Rust"),
    ("Platform", "x86-64 UEFI, bare metal"),
    ("Licence", "All rights reserved"),
    ("Size", "108 files, ~50,000 lines"),
    ("Developer", "IlumCI"),
]

WIKI_PICKS = [
    ("wiki/glados-os.html", "GLaDOS OS"),
    ("wiki/llm-in-kernel.html", "A model in the kernel"),
    ("wiki/ring-0.html", "Ring 0"),
    ("wiki/uefi-kernel.html", "UEFI as the kernel"),
    ("wiki/gui.html", "The desktop"),
    ("wiki/testing.html", "Testing without a runner"),
]

NAV = [
    ("./", "Home"),
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


def nav_html(p):
    """The horizontal bar. Home is ./ at the root and ../ below it."""
    parts = []
    for href, label in NAV:
        parts.append(a(p + href if href != "./" else (p or "./"), label))
    parts.append(a(REPO, "Source code", ' rel="noopener"'))
    return '<div id="bar">' + " | ".join(parts) + "</div>"


def sidebar_html(p, rel):
    """Four boxes: where to go, what the current release is, what the project
    is, and six ways into the wiki.

    It was five boxes and 32 links, three of them being wiki article lists.
    That is a table of contents, which is the right sidebar for a documentation
    site and the wrong one for a project page -- the full article list belongs
    on the wiki index, where somebody looking for it already is.
    """
    b = []

    nav = "".join("<li>%s</li>" % a(p + h if h != "./" else (p or "./"), t)
                  for h, t in NAV)
    nav += "<li>%s</li>" % a(REPO, "Source code")
    b.append(('Navigation', "<ul>%s</ul>" % nav))

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

    picks = "".join("<li>%s</li>" % a(p + h, t) for h, t in WIKI_PICKS)
    picks += "<li>%s</li>" % a(p + "wiki/", "All 25 articles")
    b.append(('Wiki', "<ul>%s</ul>" % picks))

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
        '  <p>Copyright 2026. All rights reserved. The source is published to '
        'be read rather than reused. One file (<code>src/dev/'
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


def build_chrome(rel):
    changed, missed = 0, []
    for full, rl in pages():
        p = prefix_for(rl)
        with open(full, "r", encoding="utf-8") as fh:
            text = orig = fh.read()
        for name, new in (("bar", nav_html(p)),
                          ("sidebar", sidebar_html(p, rel)),
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
        ctx = v["ctx"] or "&mdash;"
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
                        html.escape(v["ctx"]) or "&mdash;",
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


DERIVED = {
    "download-table": download_table,
    "releases-table": releases_table,
    "archive-tree": archive_tree,
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
    rel = releases[0]
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


def check():
    """Resolve every link the site offers.

    Internal ones are checked against the filesystem, external release
    downloads over the network with a one-byte range request. Only release
    assets are fetched: they are the links that rot, and hammering every
    Wikipedia reference in the wiki would make the check slow enough that
    nobody runs it.
    """
    bad, ext_ok, int_ok = [], 0, 0
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
                try:
                    req = urllib.request.Request(
                        href, headers={"Range": "bytes=0-0",
                                       "User-Agent": "glados-site-check"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        if r.status in (200, 206):
                            ext_ok += 1
                        else:
                            bad.append("%s -> HTTP %d  (%s)"
                                       % (href, r.status, rl))
                except urllib.error.HTTPError as e:
                    bad.append("%s -> HTTP %d  (%s)" % (href, e.code, rl))
                except Exception as e:                       # noqa: BLE001
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

    print("check: %d internal ok, %d release links ok, %d broken"
          % (int_ok, ext_ok, len(bad)))
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
    args = ap.parse_args()
    if not (args.build or args.check):
        ap.error("pick --build or --check")

    ok = True
    if args.build:
        rels = fetch_releases(args.releases_json)
        if not rels:
            print("no releases returned; refusing to blank the site")
            return 1
        print("releases: %d, latest %s (%s)"
              % (len(rels), rels[0]["tag_name"], date_of(rels[0])))
        ok &= build_chrome(rels[0])
        ok &= build_derived(rels)
        ok &= build_sitemap(rels[0])
    if args.check:
        ok &= check()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
