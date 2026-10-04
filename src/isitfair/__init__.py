__version__ = "0.4.0"

from isitfair.audit import BootstrapCell, FairnessAudit
from isitfair.effect_sizes import (
    cohens_d,
    cohens_h,
    cramers_v,
    effect_size_label,
    eta_squared,
    rank_biserial_r,
)
from isitfair.estimability import (
    DEFAULT_EVENT_FLOORS,
    EVENT_FLOOR_RATIONALE,
    FLOOR_FOR,
    GAP_SOURCE,
    floor_fails,
    resolve_event_floors,
    resolve_floor,
)
from isitfair.frailty import composite_frailty_proxy, compute_hfrs, compute_mfi4
from isitfair.inference import (
    InteractionResult,
    MetricDifferenceResult,
    OddsRatioResult,
    bootstrap_metric_difference,
    fdr_correct,
    interaction_test,
    lrt,
    odds_ratio_2x2,
)
from isitfair.metrics import HosmerLemeshowResult, hosmer_lemeshow, subgroup_metrics
from isitfair.mitigation import (
    GroupRecalibration,
    GroupThresholdOptimization,
    Mitigation,
    Reweighing,
    WassersteinPostprocessing,
)
from isitfair.periods import compare_periods
from isitfair.plots import (
    decision_curve_plot,
    forest_plot,
    lollipop_plot,
    recalibration_ladder_plot,
    reliability_plot,
    roc_plot,
    subgroup_forest_plot,
    volcano_plot,
)
from isitfair.report import generate_report, report_to_pdf
from isitfair.table_one import TableOneResult, table_one

__all__ = [
    "DEFAULT_EVENT_FLOORS",
    "EVENT_FLOOR_RATIONALE",
    "FLOOR_FOR",
    "GAP_SOURCE",
    "BootstrapCell",
    "FairnessAudit",
    "GroupRecalibration",
    "GroupThresholdOptimization",
    "HosmerLemeshowResult",
    "InteractionResult",
    "MetricDifferenceResult",
    "Mitigation",
    "OddsRatioResult",
    "Reweighing",
    "TableOneResult",
    "WassersteinPostprocessing",
    "__version__",
    "bootstrap_metric_difference",
    "cohens_d",
    "cohens_h",
    "compare_periods",
    "composite_frailty_proxy",
    "compute_hfrs",
    "compute_mfi4",
    "cramers_v",
    "decision_curve_plot",
    "effect_size_label",
    "eta_squared",
    "fdr_correct",
    "floor_fails",
    "forest_plot",
    "generate_report",
    "report_to_pdf",
    "hosmer_lemeshow",
    "interaction_test",
    "lollipop_plot",
    "lrt",
    "odds_ratio_2x2",
    "rank_biserial_r",
    "recalibration_ladder_plot",
    "reliability_plot",
    "resolve_event_floors",
    "resolve_floor",
    "roc_plot",
    "subgroup_forest_plot",
    "subgroup_metrics",
    "table_one",
    "volcano_plot",
]
