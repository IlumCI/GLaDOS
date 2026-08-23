# career/

Job-hunting material. Nothing here is part of the kernel and nothing in the
kernel refers to it; the directory can be moved to its own repository or
deleted without touching anything else.

| File | What it is |
| --- | --- |
| `resume.tex` | The resume. Source of truth, edited here rather than in a PDF. |
| `resume.pdf` | Built from it by `build.sh`. One page, A4. |
| `portfolio/index.html`, `portfolio/style.css` | Standalone page for euroswarms.eu or GitHub Pages. Hand-written HTML and CSS, no JavaScript, no external requests, no build step. |
| `portfolio/resume.pdf` | Copy of the built PDF, so `portfolio/` can be uploaded on its own with its download link intact. `build.sh` refreshes it. |
| `jobs.md` | Shortlist of openings verified on 23 Aug 2026, with what to lead with for each. Re-check the links before writing anything long. |

## Building

```sh
./build.sh
```

Needs `pdflatex`. On Debian or Ubuntu, `texlive-latex-base`,
`texlive-latex-recommended` and `texlive-fonts-recommended` are enough:
the document deliberately avoids `titlesec`, `enumitem` and every CV template
package, so it also compiles on Overleaf with no setup.

## Facts that will go stale

The resume claims 138 public repositories and 387 merged pull requests, GLaDOS
at 40,000 lines across 93 files, and independent work from Aug 2026. All four
were true on 23 Aug 2026 and none of them updates itself.
