"""Rebuild ``results/MANIFEST.csv`` from the files on disk.

The manifest records, for every executed artifact, its size, a content hash,
the notebook or script that produced it, the seed and bootstrap replicate count
it was built with, and the version of every package that can change an
artifact's bytes (optional ones included, as "not installed" when absent). Run
it after re-executing the examples, in the same environment:

    python examples/make_manifest.py

Rows are sorted by path. ``results/bern/`` is not listed: those files are
aggregated outputs of the clinical cohort, produced outside this repository,
and ``results/bern/README.md`` documents them. The manifest does not list
itself.

Examples
--------
>>> from make_manifest import produced_by
>>> produced_by("results/tables/table2_discrimination.csv")
('notebooks/02_paper_analysis_support2.ipynb', '2000')
>>> produced_by("results/notebooks_html/01_audit_report_support2.html")
('jupyter nbconvert --to html', '')
"""

from __future__ import annotations

import csv
import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from support2_data import SEED, repo_root  # noqa: E402

COLUMNS = [
    "artifact",
    "bytes",
    "sha256_16",
    "produced_by",
    "random_state",
    "n_bootstrap",
    "isitfair",
    "python",
    "numpy",
    "pandas",
    "scikit_learn",
    "statsmodels",
    "dcurves",
    "scipy",
    "matplotlib",
    "joblib",
    "jinja2",
    "tableone",
    "tabulate",
    "adjustText",
    "weasyprint",
    "data_source",
]
# Every distribution whose version can change an artifact's bytes, keyed by
# manifest column. tabulate writes the .md tables, adjustText places the
# volcano-figure labels and WeasyPrint typesets the PDF report; the last two
# are optional, and "not installed" is recorded rather than skipped, because a
# missing optional package changes the output too.
_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "statsmodels": "statsmodels",
    "dcurves": "dcurves",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "jinja2": "jinja2",
    "tableone": "tableone",
    "tabulate": "tabulate",
    "adjustText": "adjustText",
    "weasyprint": "weasyprint",
}
DATA_SOURCE = "UCI SUPPORT2 (id 880, CC BY 4.0)"
NB01 = "notebooks/01_audit_report_support2.ipynb"
NB02 = "notebooks/02_paper_analysis_support2.ipynb"
NB03 = "notebooks/03_tutorial_support2.ipynb"


def produced_by(artifact: str) -> tuple[str, str]:
    """Return ``(producer, n_bootstrap)`` for a repository-relative path.

    Parameters
    ----------
    artifact : str
        Path relative to the repository root, with forward slashes.

    Returns
    -------
    tuple of str
        The producing notebook or command, and the bootstrap replicate count
        it ran at (empty where no bootstrap is involved).

    Raises
    ------
    ValueError
        If the path belongs to no known producer, so a new artifact cannot
        slip into the manifest without an attribution.
    """
    if artifact.startswith(("results/figures/", "results/tables/")):
        return NB02, "2000"
    if artifact.startswith("results/support2_audit_report"):
        return NB01, "2000"
    if artifact.startswith("results/notebooks_html/"):
        return "jupyter nbconvert --to html", ""
    if artifact == NB01:
        return "examples/support2_mortality_audit.py (converted with jupytext)", "2000"
    if artifact == NB02:
        return "jupyter nbconvert --execute", "2000"
    if artifact == NB03:
        return "jupyter nbconvert --execute", "200"
    raise ValueError(f"no producer known for {artifact}")


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def _versions() -> dict[str, str]:
    import isitfair

    # The package version comes from the imported source, not from installed
    # metadata: an editable install can carry a stale dist-info record.
    return {
        "isitfair": isitfair.__version__,
        "python": platform.python_version(),
        **{column: _version(dist) for column, dist in _DISTRIBUTIONS.items()},
    }


def _artifacts(root: Path) -> list[str]:
    results = root / "results"
    paths = [
        p
        for p in results.rglob("*")
        if p.is_file()
        and p.name != "MANIFEST.csv"
        and "bern" not in p.relative_to(results).parts[:1]
    ]
    paths += sorted((root / "notebooks").glob("*.ipynb"))
    return sorted(p.relative_to(root).as_posix() for p in paths)


def build(root: Path | None = None) -> Path:
    """Write ``results/MANIFEST.csv`` and return its path."""
    root = root or repo_root()
    versions = _versions()
    rows = []
    for rel in _artifacts(root):
        data = (root / rel).read_bytes()
        producer, n_boot = produced_by(rel)
        rows.append(
            {
                "artifact": rel,
                "bytes": len(data),
                "sha256_16": hashlib.sha256(data).hexdigest()[:16],
                "produced_by": producer,
                "random_state": SEED,
                "n_bootstrap": n_boot,
                **versions,
                "data_source": DATA_SOURCE,
            }
        )
    out = root / "results" / "MANIFEST.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out


if __name__ == "__main__":
    path = build()
    n = sum(1 for _ in path.open(encoding="utf-8")) - 1
    print(f"{path.relative_to(repo_root())}: {n} artifacts")
