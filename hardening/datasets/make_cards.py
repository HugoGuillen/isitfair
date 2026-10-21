"""Write CARDS.md from the registry's predictions/<name>.json files.

Usage: python -I hardening/datasets/make_cards.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

STATIC = {
    "support2": dict(
        url="https://archive.ics.uci.edu/dataset/880 (bundled copy examples/data/support2.csv.gz)",
        license="CC BY 4.0", stress="control; must match results/MANIFEST.csv",
        recode="sex as recorded; age_group = age binned [18,45) [45,65) [65,80) [80,inf) "
               "(examples/support2_data.py AGE_BINS); race_ethnicity = race with missing filled "
               "'other' (0.5% of rows)",
        split="committed reference split, seed 99, 60/20/20 (train/calibration/test); the "
              "calibration partition is predictions/support2_calib.csv"),
    "diabetes130": dict(
        url="https://archive.ics.uci.edu/dataset/296", license="CC BY 4.0",
        stress="large n, LOESS scaling, missing label, likely reproduces the hang",
        recode="race as written by UCI (the current static CSV writes the original '?' as the "
               "literal text 'NaN', which pandas reads as missing; kept missing so the audit's "
               "validation is exercised); gender as recorded (includes 'Unknown/Invalid'); age as "
               "UCI's 10-year bins '[0-10)'...'[90-100)'. Rows are encounters, not patients "
               "(patient_nbr repeats); the split is by row as specified, so a patient can sit in "
               "both halves"),
    "adult": dict(
        url="https://archive.ics.uci.edu/dataset/2", license="CC BY 4.0",
        stress="large n, multi-level attribute",
        recode="sex, race (5 levels) as recorded; age_group = [17,30) [30,45) [45,60) [60,inf); "
               "income labels '>50K' and '>50K.' (the original test file's trailing period) "
               "both count as the event"),
    "compas": dict(
        url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        license="no license file in propublica/compas-analysis (GitHub API reports license: null); "
                "used for research under ProPublica's publication of the data",
        stress="canonical fairness set, moderate n",
        recode="race (6 levels), sex, age_cat as recorded; rows filtered as in ProPublica's "
               "analysis (|days_b_screening_arrest| <= 30, is_recid != -1, c_charge_degree != 'O', "
               "score_text != 'N/A'), 7,214 -> 6,172"),
    "bank": dict(
        url="https://archive.ics.uci.edu/dataset/222", license="CC BY 4.0",
        stress="11.7% prevalence, 12-level attribute, intersections with empty cells",
        recode="age_group = [18,30) [30,40) [40,50) [50,60) [60,inf); job (12 levels; UCI's "
               "'unknown' arrives as missing and is recoded to 'unknown'); marital as recorded"),
    "german": dict(
        url="https://archive.ics.uci.edu/dataset/144", license="CC BY 4.0",
        stress="every event floor fails, `not estimable` paths",
        recode="sex from Attribute9 (A91/A93/A94 male; A92/A95 female); age_group from "
               "Attribute13 = [19,25) [25,35) [35,50) [50,inf); outcome class 2 = bad"),
    "heart_cleveland": dict(
        url="https://archive.ics.uci.edu/dataset/45", license="CC BY 4.0",
        stress="tiny n, degenerate bootstrap replicates",
        recode="sex 1 male / 0 female; age_group = [0,45) [45,55) [55,65) [65,inf)"),
    "oracle_calib": dict(
        url="synthetic (registry.make_oracle_calib, seed 42)", license="n/a",
        stress="planted subgroup calibration (a, b) = (0, 1) in A and (-0.8, 0.7) in B",
        recode="group A/B with equal probability; logit(p) ~ N(-1, 1.2^2); "
               "y ~ Bernoulli(expit(a_g + b_g logit p)); both prediction columns are p"),
    "oracle_sparse": dict(
        url="synthetic (registry.make_oracle_sparse, seed 42)", license="n/a",
        stress="a level with exactly 3 events (L5) and a level with 0 (L6): must be not estimable",
        recode="level L1..L6 with sizes 1800/1200/900/600/300/200; logit(p) ~ N(-2.2, 1); "
               "y ~ Bernoulli(p), then L5 forced to exactly 3 events and L6 to 0"),
}


def main() -> None:
    lines = [
        "# Dataset cards",
        "",
        "Built by `hardening/datasets/registry.py`; numbers below are read from",
        "`predictions/<name>.json`. Audited model for every real dataset except",
        "support2: `Pipeline(ColumnTransformer(median-impute + scale | mode-impute +",
        "one-hot(min_frequency=50)), LogisticRegression(max_iter=2000))` and, separately,",
        "`HistGradientBoostingClassifier(random_state=42)`; stratified 70/30 split with",
        "`random_state=42`; predictions on the 30% test half only. Sex and race are held out",
        "of the features (as in `examples/support2_data.py`); age stays in.",
        "",
    ]
    for name, s in STATIC.items():
        m = json.loads((HERE / "predictions" / f"{name}.json").read_text())
        dl = HERE / "raw" / name / "DOWNLOAD.json"
        date = json.loads(dl.read_text())["downloaded"] if dl.exists() else "n/a (in repository / generated)"
        lines += [
            f"## {name}",
            "",
            f"- **Source**: {s['url']}",
            f"- **License**: {s['license']}",
            f"- **Downloaded**: {date}",
            f"- **Outcome**: {m['outcome']}",
            f"- **Attributes and recoding**: {s['recode']}",
            f"- **Features**: {m['features_note']}",
            f"- **Sizes**: total {m['n_total']:,}; train {m['n_train']:,}; audited test {m['n_test']:,}"
            + (f"; {s['split']}" if "split" in s else ""),
            f"- **Test prevalence**: {m['prevalence_test']:.4f} ({m['events_test']:,} events); "
            f"audit threshold {m['threshold']}",
            f"- **Stress condition**: {s['stress']}",
            "",
            "| attribute | subgroup | n | events | prevalence |",
            "|---|---|---:|---:|---:|",
        ]
        for attr, groups in m["per_subgroup"].items():
            for g, v in groups.items():
                lines.append(f"| {attr} | {g} | {v['n']:,} | {v['events']:,} | {v['prevalence']:.4f} |")
        lines.append("")
    (HERE / "CARDS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
