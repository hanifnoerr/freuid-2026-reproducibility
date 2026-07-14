# Technical report

Source: `freuid_technical_report.tex`

Bibliography: `references.bib`

Submission artifact: `freuid_technical_report.pdf`

The report documents the frozen ConvNeXtV2 checkpoint, Google Colab G4 96 GB
training run, public leaderboard result, and Docker inference contract.

Compile from this directory with a LaTeX installation that provides BibTeX:

```bash
pdflatex freuid_technical_report.tex
bibtex freuid_technical_report
pdflatex freuid_technical_report.tex
pdflatex freuid_technical_report.tex
```
