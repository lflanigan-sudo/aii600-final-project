# Final audit checklist

Date: 2026-11-16. This list is the audit. It does not replace `CHECKLIST.md`, and it does not change the ticks on that file.

- [x] Catalogue price basis checked against the columns and eight public lot pages, including four booster-only flags. Finding recorded in `catalogue_price_audit.md`. Target not redefined.
- [x] Frozen booster rerun with `python -m scripts.closing_diagnostics`. 2024 pinball matched $3,168.7295. Report in `reproducibility_report.md`.
- [x] IQN search, CatBoost grid, and feature search were not rerun.
- [x] 2026 brand and price-band table was not rebuilt from scratch. The same script reprinted it and it matched the notebook.
- [x] Screening counts confirmed: 269, 344, 205, 64, 139, 354, and 203 changes.
- [x] Slides in `presentation.qmd` now lead with the 20.30% reduction, the feature-bundle result, 23.80% versus 33.33%, and interval coverage of 78.43% and 72.65%.
- [x] `presentation.html` keeps the original four-slide design and now contains the 20.30%, 23.80%, 78.43%, and 72.65% figures. It is not the Quarto render.
- [x] Notebook final section, `data/README.md`, and a new top entry in `decisions.md` use the recorded sale price rather than an unverified hammer.
- [x] Decision, finding, evidence, rejected alternatives, strongest check, and the two-minute opening are in `defense_summary.md`.
- [ ] The notebook was not executed from the top. Feature and closing cells have no saved outputs.
- [x] Four booster-only lots were opened: two Bonhams pages say the matching amount includes premium, and two Sotheby's pages match the recorded estimates but do not show the sold price. See `catalogue_price_audit.md`.
- [ ] Sotheby's sold prices remain unverified on the public page. The checked 2026 lot and the two booster-only Sotheby's lots do not display a result without login.
- [ ] No commit or push was made for this audit.
