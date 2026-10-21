# Breaking ECDSA on a fault-tolerant quantum computer — companion document to shor/

Build (TeX Live 2023 or later; needs quantikz2, pgfplots ≥1.18, mathpazo, tcolorbox, algpseudocode):

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Layout:
- `main.tex`, `refs.bib` — document and bibliography (≈100 entries, arXiv/ePrint/company sources).
- `sections/` — Part I (01 target, 02 optimisations, 03 Pareto frontier, 04 shor/ implementation),
  Part II (05 QEC foundations, 06 small/surface/toric codes, 06b colour/qLDPC codes, 07 FT Clifford and T gates,
  08 physical estimates, 09 conclusion), appendices (A sources, B scripts).
- `scripts/` — Python (Stim, PyMatching, NumPy) that produced every "computed" number and plot.
- `data/` — the scripts' outputs (threshold data, BP+OSD data, Model S table, Pareto/trajectory data, syndrome tables).

Regenerate the data:  `python3 scripts/surface_threshold.py data/surface_threshold.dat` (hours),
`python3 scripts/bp_osd_gross.py`, `python3 scripts/physical_estimates.py && python3 scripts/modelS_table.py`,
`python3 scripts/syndrome_tables.py`, `python3 scripts/standard_form_encoder.py`, `python3 scripts/bb_codes.py`.
