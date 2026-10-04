# Reporting

isitfair generates self-contained HTML reports suitable for inclusion in regulatory submissions, manuscript supplementary materials, and clinical audit documentation.

## Generating a report

```python
audit.report("fairness_report.html", title="Infection Model Fairness Audit")
```

The `report()` method delegates to `isitfair.report.generate_report()` and returns the output `Path`. The report is a single HTML file with no external dependencies — all plots are embedded as base64-encoded SVG, and styles are inlined.

## Report sections

The generated report includes:

1. **Header** — title, generation timestamp, and optional metadata.
2. **Summary** — cohort size, number of events, prevalence, threshold, number of attributes, and bootstrap count.
3. **Discrimination** — per-attribute tables of AUROC, AUPRC, sensitivity, specificity, PPV, NPV with CIs. Gap statistics (equalized odds, predictive equality, statistical parity). Forest plots of AUROC and AUPRC per subgroup, and ROC curves with bootstrap bands.
4. **Calibration** — per-attribute tables of calibration intercept, slope, Brier score, ECE, ICI. LOESS-smoothed calibration curves with bootstrap confidence bands. Hosmer–Lemeshow goodness-of-fit with the per-subgroup decile tables of observed versus expected events, and the standing caveat that the statistic is not comparable across subgroups of different size.
5. **Clinical utility** — decision curves per attribute showing net benefit vs. threshold, with treat-all and treat-none reference lines.
6. **Mitigation** (conditional) — before-and-after metric tables when `mitigation_audits` is provided.
7. **Cell diagnostics** (conditional) — interaction cell statistics when intersectional attributes are defined.
8. **TRIPOD+AI item coverage** (appendix) — maps fairness-relevant TRIPOD+AI items (Collins et al., 2024) to report sections; flags items requiring user input.
9. **Footer** — TRIPOD+AI compliance note and generation metadata.

## Customisation

### Title and metadata

```python
audit.report(
    "report.html",
    title="Readmission Model — Fairness Audit",
    metadata={
        "Model": "Gradient boosting, v2.1",
        "Cohort": "Held-out test set, 2023–2024",
        "Analyst": "A. Analyst",
    },
)
```

Metadata key-value pairs appear in the report header.

### Including mitigation results

```python
from isitfair import GroupRecalibration

recalib = GroupRecalibration(method="isotonic")
mitigated = audit.mitigate(recalib, calibration_data=(y_c, p_c, s_c))

audit.report(
    "report.html",
    title="Audit with Mitigation",
    mitigation_audits=mitigated,
)
```

This adds a mitigation section comparing original and mitigated metrics per attribute.

## Determinism

Reports are reproducible. Given the same `FairnessAudit` object (which is deterministic by default when `random_state` is set), `report()` produces bit-identical HTML output across calls. This is achieved by:

- Fixing matplotlib's SVG hash salt to avoid random element IDs.
- Suppressing timestamp metadata in SVG output.
- Using deterministic Jinja2 template rendering.

This property is important for TRIPOD+AI compliance, where supplementary materials must be reproducible from the audit state.

## PDF export

`pdf=True` also writes a print-typeset PDF beside the HTML, with the same stem:

```python
audit.report("fairness_report.html", pdf=True)   # also writes fairness_report.pdf
```

An existing report can be converted without re-running its audit:

```python
from isitfair import report_to_pdf

report_to_pdf("fairness_report.html")            # -> fairness_report.pdf
```

The PDF is the HTML set for print: A4 landscape, ruled tables whose header row
repeats on every page, one page per section, and page numbers. Nothing is
recomputed. The output is deterministic, so the same report gives the same PDF
bytes.

PDF export needs the `pdf` extra (`pip install "isitfair[pdf]"`), which installs
WeasyPrint. WeasyPrint also needs the Pango system library. Without it the HTML
report works as before, and only the PDF call raises an `ImportError` naming
the extra.

## Programmatic access

If you need finer control than the report provides, access the underlying DataFrames directly:

```python
disc = audit.discrimination()    # DataFrame
cal = audit.calibration()        # DataFrame
curves = audit.calibration_curves()  # DataFrame
hl = audit.hosmer_lemeshow()     # DataFrame
hl_deciles = audit.hosmer_lemeshow(detail=True)  # DataFrame
roc = audit.roc_curves()         # DataFrame
dc = audit.decision_curve()      # DataFrame
diag = audit.cell_diagnostics()  # DataFrame (if intersections defined)
```

All DataFrames are long-format and suitable for custom plotting with matplotlib, seaborn, plotly, or export to CSV/Excel.

## TRIPOD+AI alignment

The report automatically generates an appendix mapping fairness-relevant TRIPOD+AI items
(Collins et al., 2024, *BMJ*) to report sections. The following items are addressed by the audit:

- **Item 12e**: All measures used to evaluate model performance → Discrimination, Calibration, and Clinical utility sections.
- **Item 14**: Fairness approaches and rationale → Audit summary and gap statistics.
- **Item 20b**: Report demographic characteristics and group differences → Discrimination section.
- **Item 23a**: Report model performance with CIs for key subgroups → All three analysis sections.

Conditionally addressed:

- **Item 12f**: Model updating for sociodemographic groups → Mitigation section (when included).

Items marked "User responsibility" (3c, 5a, 7, 8a, 8b, 9c, 20c, 25, 26) require information that the tool
cannot produce automatically, such as known health inequalities, data representativeness, data preparation
details, and overall interpretation. The appendix explicitly lists these so that manuscript authors can
check them off.
