#!/bin/sh
# Build the resume and put a copy where the portfolio page expects it, so
# career/portfolio/ can be uploaded on its own and still have a working
# "Resume (PDF)" link.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode resume.tex >/dev/null
pdflatex -interaction=nonstopmode resume.tex >/dev/null   # second pass for hyperref
rm -f resume.aux resume.log resume.out build.log
cp resume.pdf portfolio/resume.pdf
echo "resume.pdf built and copied into portfolio/"
