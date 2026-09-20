#!/usr/bin/env python3
"""Cut a thirty-second trailer out of a long capture.

    python tools/trailer.py out/demo-raw.mp4 out/trailer.mp4

### Why this is separate from `record.py`

`record.py` films a *session*: it drives the guest, waits for each command, and
lingers long enough that somebody could read the output. That is a walkthrough,
and the shortest honest one this machine can give is about four minutes, most
of which is text arriving at the speed a terminal prints it.

A trailer is not a shortened walkthrough. It asks nobody to read: each shot is
two to four seconds, the caption carries the meaning in four words, and the
footage is chosen for motion rather than for content. So the two cannot come
out of one pass, and the split is where the expensive part stops -- **booting
the guest costs six minutes and cutting costs about a minute**, so the capture
happens once and the edit is iterated against the file.

### The grammar

Nothing is a screen recording. Every shot is composed:

- **The guest is a card, not the frame.** 1280x800 sits inside 1920x1080 at
  1:1, so not one pixel of it is resampled, and the 320 pixels of margin either
  side are what make it read as a product shot rather than as somebody's
  desktop. A rim light and a soft drop shadow lift it off the plate.
- **Nothing holds still.** Each shot pushes in a little over its length. The
  amount is small on purpose: enough that the frame is alive, not so much that
  the upscale becomes visible.
- **Speed is per shot and it is the whole rhythm.** A terminal filling with
  green `ok` lines is dull at 1x and good at eight; a window arriving plays at
  1x because the animation is six frames and speeding it up removes the thing
  worth showing.
- **A punch-in crops the source and scales with `neighbor`.** This is a pixel
  font on a pixel UI, so a smooth upscale turns crisp glyphs into mush --
  nearest neighbour keeps the edges hard and reads as deliberate emphasis
  rather than as a zoom that lost focus.
- **Type is animated, not stamped.** It rises a few pixels and fades up over
  the first third of a second. A caption that simply appears reads as a
  subtitle; one that moves reads as design.
- **Hard cuts.** At three seconds a shot, a cross-fade eats a tenth of every
  clip and reads as hesitancy. Cutting on the caption change is what makes a
  trailer feel deliberate.

### The cut list is the edit

`tools/trailer.json` is a list of shots against the raw capture, in
raw-capture seconds. Keeping it out of this file is what makes the loop cheap:
a retime is an edit to data, and the tool never changes.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# 1920x1080 around a 1280x800 guest. The margin is the design.
W, H = 1920, 1080
GW, GH = 1280, 800
# The card sits a little above centre, so the caption has room under it without
# crowding the frame's bottom edge.
LIFT = 46

FPS = 30
# Deep teal into near-black. GLaDOS's own wallpaper is a horizon from deep
# water to gold, and this is the cold half of it so the guest's warm chrome has
# something to sit against.
PLATE_TOP = "0x0d2530"
PLATE_BOT = "0x05090c"
RIM = "0x3f94a8"

TITLE_PT = 74
CAPTION_PT = 50
FADE = 0.6


def ffmpeg():
    for p in (
        Path("C:/Program Files/ShareX/ffmpeg.exe"),
        Path("C:/Program Files/Krita (x64)/bin/ffmpeg.exe"),
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
    ):
        if p.exists():
            return p
    from shutil import which
    found = which("ffmpeg")
    if found:
        return Path(found)
    raise SystemExit("trailer.py: no ffmpeg found")


def fonts(into: Path):
    """Two weights beside the filter script, referenced by bare name.

    **Copied rather than referenced, because of one character.** A filter graph
    separates options with colons, so an absolute Windows path ends the value
    at the drive letter -- `No option name near '/Windows/Fonts/...'` -- and
    the documented escape does not survive a `-filter_complex_script` file. A
    relative name has no colon in it.
    """
    out = {}
    for key, names in (
        ("bold", ("seguibl.ttf", "segoeuib.ttf", "arialbd.ttf")),
        ("semi", ("seguisb.ttf", "segoeui.ttf", "arial.ttf")),
    ):
        for n in names:
            src = Path("C:/Windows/Fonts") / n
            if src.exists():
                dst = into / f"{key}.ttf"
                shutil.copyfile(src, dst)
                out[key] = dst.name
                break
        else:
            raise SystemExit(f"trailer.py: no {key} font under C:/Windows/Fonts")
    return out


def video_size(ff, raw):
    """How big the capture actually is.

    Asked rather than assumed. A capture that came out 640x480 instead of
    1280x800 -- the whole top-left quarter of the desktop and nothing else --
    would otherwise be composited as a tiny card in the middle of a large
    plate, which looks like a design decision rather than a broken take.
    """
    r = subprocess.run([str(ff), "-hide_banner", "-i", str(raw)],
                       capture_output=True, text=True, errors="replace")
    m = re.search(r",\s(\d{2,5})x(\d{2,5})[,\s]", r.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def esc(s):
    """Text for a `drawtext` written inline rather than in a file."""
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\\\'")


def title_card(fonts_, n, text, sub, secs):
    """A statement on the plate, with no footage under it.

    Trailers open and close on one, and the reason is pacing rather than
    decoration: a cut straight from black into a terminal gives the eye nothing
    to land on, and a cut from the last shot to black ends the film in the
    middle of a sentence.
    """
    g = [f"gradients=s={W}x{H}:c0={PLATE_TOP}:c1={PLATE_BOT}"
         f":x0={W//2}:y0=0:x1={W//2}:y1={H}:d={secs}:r={FPS}"
         f",noise=alls=3:allf=t+u[t{n}bg]"]
    chain = f"[t{n}bg]"
    chain += (f"drawtext=fontfile={fonts_['bold']}:text='{esc(text)}'"
              f":fontsize={TITLE_PT}:fontcolor=0xf4f9fa"
              f":x=(w-text_w)/2:y='(h-text_h)/2-30+18*(1-min(t/0.5,1))'"
              f":alpha='min(t/0.45,1)'")
    if sub:
        chain += (f",drawtext=fontfile={fonts_['semi']}:text='{esc(sub)}'"
                  f":fontsize=38:fontcolor=0x8fb3bd"
                  f":x=(w-text_w)/2:y='(h-text_h)/2+66'"
                  f":alpha='max(0,min((t-0.35)/0.5,1))'")
    chain += f",vignette=a=0.55[t{n}]"
    g.append(chain)
    return g, f"[t{n}]", secs


def shot(fonts_, n, c, src_w, src_h):
    """One composed shot: plate, shadow, card, push-in, caption."""
    at, ln, sp = float(c["at"]), float(c["len"]), float(c.get("speed", 1.0))
    shown = ln / sp
    push = float(c.get("push", 0.11))
    g = []

    # --- the footage -------------------------------------------------------
    v = (f"[0:v]trim={at:.3f}:{at + ln:.3f},setpts=(PTS-STARTPTS)/{sp:g}"
         f",fps={FPS}")
    focus = c.get("focus")
    if focus:
        # A punch-in. `neighbor` on purpose: this is a pixel font, and a smooth
        # upscale turns crisp glyphs into mush where nearest neighbour keeps
        # the edges hard.
        fx, fy, fw, fh = focus
        v += f",crop={fw}:{fh}:{fx}:{fy},scale={GW}:{GH}:flags=neighbor"
    elif (src_w, src_h) != (GW, GH):
        v += f",scale={GW}:{GH}:flags=neighbor"
    # Per-frame zoom rather than a scale, because `scale` takes no expression.
    rate = push / max(1.0, shown * FPS)
    v += (f",zoompan=z='min(1.0+{rate:.6f}*on,{1.0 + push:.4f})':d=1"
          f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={GW}x{GH}:fps={FPS}")
    v += f",pad=w=iw+3:h=ih+3:x=1:y=1:color={RIM}[card{n}]"
    g.append(v)

    # --- the plate and its shadow -----------------------------------------
    g.append(f"gradients=s={W}x{H}:c0={PLATE_TOP}:c1={PLATE_BOT}"
             f":x0={W//2}:y0=0:x1={W//2}:y1={H}:d={shown:.3f}:r={FPS}"
             f",noise=alls=3:allf=t+u[bg{n}]")
    g.append(f"color=c=black:s={GW + 120}x{GH + 100}:d={shown:.3f}:r={FPS}"
             f",format=rgba,colorchannelmixer=aa=0.6,gblur=sigma=34[sh{n}]")
    g.append(f"[bg{n}][sh{n}]overlay=(W-w)/2:(H-h)/2-{LIFT - 18}[plate{n}]")
    g.append(f"[plate{n}][card{n}]overlay=(W-w)/2:(H-h)/2-{LIFT}[comp{n}]")

    # --- the caption -------------------------------------------------------
    chain = f"[comp{n}]"
    if c.get("text"):
        chain += (f"drawtext=fontfile={fonts_['semi']}:text='{esc(c['text'])}'"
                  f":fontsize={c.get('size', CAPTION_PT)}:fontcolor=0xf2f7f8"
                  f":x=(w-text_w)/2:y='H-142+28*(1-min(t/0.45,1))'"
                  f":alpha='min(t/0.4,1)',")
    chain += f"vignette=a=0.62[s{n}]"
    g.append(chain)
    return g, f"[s{n}]", shown


def build(raw, out, spec):
    ff = ffmpeg()
    raw, out = Path(raw), Path(out)
    if not raw.exists():
        raise SystemExit(f"trailer.py: no capture at {raw}")
    work = out.parent / (out.stem + "-cut")
    work.mkdir(parents=True, exist_ok=True)
    f = fonts(work)

    sw, sh = video_size(ff, raw)
    if (sw, sh) != (GW, GH):
        print(f"[trailer] WARNING: the capture is {sw}x{sh}, not {GW}x{GH}. "
              f"It will be scaled, so the card is not pixel-perfect.",
              file=sys.stderr)

    graph, labels, total, n = [], [], 0.0, 0
    for c in spec:
        if c.get("kind") == "title":
            g, lab, secs = title_card(f, n, c["text"], c.get("sub", ""),
                                      float(c.get("len", 2.5)))
        else:
            g, lab, secs = shot(f, n, c, sw, sh)
        graph += g
        labels.append(lab)
        total += secs
        n += 1

    graph.append("".join(labels) + f"concat=n={n}:v=1:a=0[cat]")
    graph.append(f"[cat]fade=t=in:st=0:d={FADE},"
                 f"fade=t=out:st={max(0.0, total - FADE):.2f}:d={FADE}[v]")

    script = work / "filter.txt"
    script.write_text(";".join(graph), encoding="utf-8")

    print(f"[trailer] {n} shot(s), {total:.1f}s at {W}x{H}")
    t = 0.0
    for c in spec:
        if c.get("kind") == "title":
            secs = float(c.get("len", 2.5))
            print(f"  {t:5.1f} +{secs:4.1f}s  TITLE   {c['text']}")
        else:
            secs = float(c["len"]) / float(c.get("speed", 1.0))
            mark = "punch" if c.get("focus") else "     "
            print(f"  {t:5.1f} +{secs:4.1f}s  @{c['at']:>6.1f} "
                  f"x{c.get('speed', 1):<4g} {mark} {c.get('text', '')}")
        t += secs

    r = subprocess.run(
        [str(ff), "-hide_banner", "-loglevel", "error",
         "-i", str(raw.resolve()),
         "-filter_complex_script", str(script.resolve()),
         "-map", "[v]", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-y", str(out.resolve())],
        cwd=work)
    if r.returncode != 0 or not out.exists():
        raise SystemExit("trailer.py: the cut failed")
    print(f"[trailer] {out} ({out.stat().st_size / 1e6:.1f} MB, {total:.1f}s)")
    return out


def main():
    argv = sys.argv[1:]
    raw = argv[0] if argv else ROOT / "out/demo-raw.mp4"
    out = argv[1] if len(argv) > 1 else ROOT / "out/trailer.mp4"
    spec = json.loads((ROOT / "tools/trailer.json").read_text(encoding="utf-8"))
    build(raw, out, spec)


if __name__ == "__main__":
    main()
