# AutoDD poster — ICDM 2026

- **Board size:** 33.1 in x 46.8 in (841 x 1189 mm, A0 portrait)
- **Paper ID:** DM1326 (bottom-right corner, set via `\paperid` in `poster.tex`)
- **QR code:** generated at build time from `\paperurl`
  (currently https://github.com/TechyNilesh/AutoDD) — edit that macro to
  repoint it, no image file needed.

## Build

    pdflatex poster.tex     # run twice: the outer frame and paper ID use
    pdflatex poster.tex     # TikZ remember-picture

## Files

- `poster.tex` — the poster source
- `assets/icdm_logo.png` — ICDM 2026 logo, flattened onto black
- figures are pulled from `../images/` via `\graphicspath`, so the poster
  always tracks the figures used in the paper
