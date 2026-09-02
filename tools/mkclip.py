"""Turn a recorded PNG sequence into something that plays.

`drive.py --record` writes numbered PNGs, which is what an editor wants and
what nothing else will open. This makes the preview: an animated WebP, and a
GIF beside it for anywhere WebP is refused.

WebP first because a GIF is 256 colours and this interface is gradients --
banding across the wallpaper is the one artefact that makes a real capture look
like a bad render. The GIF is written anyway, quantised once against a palette
taken from the whole sequence rather than per frame, because a per-frame
palette makes the background shimmer.

    python tools/mkclip.py out/trailer/clip-author --fps 24 --scale 0.5

There is no ffmpeg on this machine, and adding one to a repo whose whole point
is having no dependencies was not worth an mp4. Any editor imports the PNG
sequence directly.
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("needs Pillow: tools/venv/Scripts/pip install pillow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory of f0000.png frames")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--crop", default="", help="x,y,w,h in source pixels")
    ap.add_argument("--out", default="", help="basename, default the directory")
    args = ap.parse_args()

    d = Path(args.dir)
    frames = sorted(d.glob("f*.png"))
    if not frames:
        sys.exit(f"no frames in {d}")

    box = None
    if args.crop:
        x, y, w, h = (int(v) for v in args.crop.split(","))
        box = (x, y, x + w, y + h)

    imgs = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        if box:
            im = im.crop(box)
        if args.scale != 1.0:
            # NEAREST, not LANCZOS. Every glyph here is an 8x8 bitmap doubled
            # to 16x16, so a halving is exact and a smooth filter would blur
            # the one thing the capture exists to show.
            im = im.resize(
                (int(im.width * args.scale), int(im.height * args.scale)), Image.NEAREST
            )
        imgs.append(im)

    base = Path(args.out) if args.out else d
    ms = max(10, int(round(1000.0 / args.fps)))

    webp = base.with_suffix(".webp")
    imgs[0].save(
        webp, save_all=True, append_images=imgs[1:], duration=ms, loop=0, quality=88,
    )
    print(f"{webp}  {len(imgs)} frames  {imgs[0].width}x{imgs[0].height}  {args.fps}fps")

    # One palette for the whole sequence. Adaptive per frame re-quantises the
    # wallpaper every frame and the gradient crawls.
    gif = base.with_suffix(".gif")
    pal = imgs[0].quantize(colors=255, method=Image.MEDIANCUT)
    conv = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in imgs]
    conv[0].save(
        gif, save_all=True, append_images=conv[1:], duration=ms, loop=0, optimize=True,
    )
    print(f"{gif}  {gif.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
