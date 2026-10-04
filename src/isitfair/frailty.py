"""
Frailty index ICD-10 code lists for the perioperative fairness audit.

HFRS weights and codes verified against:
    Gilbert T, Neuburger J, Kraindler J, et al. Lancet 2018;391:1775-82.
    Supplementary appendix, Table A2 (109 codes, version 5.6, 14 Mar 2018).

mFI-4 mappings from:
    Subramaniam S, Aalberg JJ, Soriano RP, et al. JAMA Surg 2018;153:160-7.
    (mFI-5 minus functional status, which ICD-10 discharge codes do not record.)

Both indices use 3-character ICD-10 category matching, following the
original Gilbert SAS macro. Each category contributes its weight at most
once per admission. Recorded codes can include any subcategory digits
(e.g. F00.1, F001, F00 all match the F00 entry).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# HFRS: Hospital Frailty Risk Score (Gilbert 2018), 109 codes from Table A2.
# ---------------------------------------------------------------------------
#     score < 5         → low risk (non-frail)
#     5 ≤ score ≤ 15    → intermediate risk
#     score > 15        → high risk (frail)

HFRS_WEIGHTS: dict[str, float] = {
    "F00": 7.1,  # Dementia in Alzheimer's disease
    "G81": 4.4,  # Hemiplegia
    "G30": 4.0,  # Alzheimer's disease
    "I69": 3.7,  # Sequelae of cerebrovascular disease (secondary)
    "R29": 3.6,  # Other symptoms involving nervous & musculoskeletal (R29.6 tendency to fall)
    "N39": 3.2,  # Other disorders of urinary system (incl. UTI, incontinence)
    "F05": 3.2,  # Delirium, not induced by alcohol/psychoactive
    "W19": 3.2,  # Unspecified fall
    "S00": 3.2,  # Superficial injury of head
    "R31": 3.0,  # Unspecified haematuria
    "B96": 2.9,  # Other bacterial agents (secondary)
    "R41": 2.7,  # Other symptoms involving cognitive functions
    "R26": 2.6,  # Abnormalities of gait and mobility
    "I67": 2.6,  # Other cerebrovascular diseases
    "R56": 2.6,  # Convulsions, NEC
    "R40": 2.5,  # Somnolence, stupor and coma
    "T83": 2.4,  # Complications of genitourinary prosthetic devices
    "S06": 2.4,  # Intracranial injury
    "S42": 2.3,  # Fracture of shoulder/upper arm
    "E87": 2.3,  # Other disorders of fluid/electrolyte/acid-base
    "M25": 2.3,  # Other joint disorders, NEC
    "E86": 2.3,  # Volume depletion
    "R54": 2.2,  # Senility
    "Z50": 2.1,  # Care involving rehabilitation procedures
    "F03": 2.1,  # Unspecified dementia
    "W18": 2.1,  # Other fall on same level
    "Z75": 2.0,  # Problems related to medical facilities
    "F01": 2.0,  # Vascular dementia
    "S80": 2.0,  # Superficial injury of lower leg
    "L03": 2.0,  # Cellulitis
    "H54": 1.9,  # Blindness and low vision
    "E53": 1.9,  # Deficiency of other B group vitamins
    "Z60": 1.8,  # Problems related to social environment
    "G20": 1.8,  # Parkinson's disease
    "R55": 1.8,  # Syncope and collapse
    "S22": 1.8,  # Fracture of rib(s)/sternum/thoracic spine
    "K59": 1.8,  # Other functional intestinal disorders
    "N17": 1.8,  # Acute renal failure
    "L89": 1.7,  # Decubitus ulcer
    "Z22": 1.7,  # Carrier of infectious disease
    "B95": 1.7,  # Streptococcus/staphylococcus as cause (secondary)
    "L97": 1.6,  # Ulcer of lower limb, NEC
    "R44": 1.6,  # Other symptoms involving general sensations
    "K26": 1.6,  # Duodenal ulcer
    "I95": 1.6,  # Hypotension
    "N19": 1.6,  # Unspecified renal failure
    "A41": 1.6,  # Other septicaemia
    "Z87": 1.5,  # Personal history of other diseases and conditions
    "J96": 1.5,  # Respiratory failure, NEC
    "X59": 1.5,  # Exposure to unspecified factor
    "M19": 1.5,  # Other arthrosis
    "G40": 1.5,  # Epilepsy
    "M81": 1.4,  # Osteoporosis without pathological fracture
    "S72": 1.4,  # Fracture of femur
    "S32": 1.4,  # Fracture of lumbar spine and pelvis
    "E16": 1.4,  # Other disorders of pancreatic internal secretion
    "R94": 1.4,  # Abnormal results of function studies
    "N18": 1.4,  # Chronic renal failure
    "R33": 1.3,  # Retention of urine
    "R69": 1.3,  # Unknown and unspecified causes of morbidity
    "N28": 1.3,  # Other disorders of kidney and ureter, NEC
    "R32": 1.2,  # Unspecified urinary incontinence
    "G31": 1.2,  # Other degenerative diseases of nervous system, NEC
    "Y95": 1.2,  # Nosocomial condition
    "S09": 1.2,  # Other and unspecified injuries of head
    "R45": 1.2,  # Symptoms/signs involving emotional state
    "G45": 1.2,  # Transient cerebral ischaemic attacks
    "Z74": 1.1,  # Problems related to care-provider dependency
    "M79": 1.1,  # Other soft tissue disorders, NEC
    "W06": 1.1,  # Fall involving bed
    "S01": 1.1,  # Open wound of head
    "A04": 1.1,  # Other bacterial intestinal infections
    "A09": 1.1,  # Diarrhoea and gastroenteritis of presumed infectious origin
    "J18": 1.1,  # Pneumonia, organism unspecified
    "J69": 1.0,  # Pneumonitis due to solids and liquids
    "R47": 1.0,  # Speech disturbances, NEC
    "E55": 1.0,  # Vitamin D deficiency
    "Z93": 1.0,  # Artificial opening status
    "R02": 1.0,  # Gangrene, NEC
    "R63": 0.9,  # Symptoms concerning food and fluid intake
    "H91": 0.9,  # Other hearing loss
    "W10": 0.9,  # Fall on/from stairs and steps
    "W01": 0.9,  # Fall on same level slipping/tripping
    "E05": 0.9,  # Thyrotoxicosis (hyperthyroidism)
    "M41": 0.9,  # Scoliosis
    "R13": 0.8,  # Dysphagia
    "Z99": 0.8,  # Dependence on enabling machines and devices
    "U80": 0.8,  # Agent resistant to penicillin/related antibiotics
    "M80": 0.8,  # Osteoporosis with pathological fracture
    "K92": 0.8,  # Other diseases of digestive system
    "I63": 0.8,  # Cerebral infarction
    "N20": 0.7,  # Calculus of kidney and ureter
    "F10": 0.7,  # Mental/behavioural disorders due to alcohol
    "Y84": 0.7,  # Other medical procedures as cause of abnormal reaction
    "R00": 0.7,  # Abnormalities of heart beat
    "J22": 0.7,  # Unspecified acute lower respiratory infection
    "Z73": 0.6,  # Problems related to life-management difficulty
    "R79": 0.6,  # Other abnormal findings of blood chemistry
    "Z91": 0.5,  # Personal history of risk-factors, NEC
    "S51": 0.5,  # Open wound of forearm
    "F32": 0.5,  # Depressive episode
    "M48": 0.5,  # Spinal stenosis (secondary code only)
    "E83": 0.4,  # Disorders of mineral metabolism
    "M15": 0.4,  # Polyarthrosis
    "D64": 0.4,  # Other anaemias
    "L08": 0.4,  # Other local infections of skin and subcutaneous tissue
    "R11": 0.3,  # Nausea and vomiting
    "K52": 0.3,  # Other noninfective gastroenteritis and colitis
    "R50": 0.1,  # Fever of unknown origin
}

HFRS_THRESHOLDS = {"low_max": 5.0, "intermediate_max": 15.0}


# ---------------------------------------------------------------------------
# mFI-4: Subramaniam 2018 mFI-5 minus functional status.
# ---------------------------------------------------------------------------

MFI4_ITEMS: dict[str, list[str]] = {
    "diabetes_mellitus": ["E10", "E11", "E12", "E13", "E14"],
    "copd": ["J41", "J42", "J43", "J44", "J47"],
    "congestive_heart_failure": ["I50"],
    "hypertension": ["I10", "I11", "I12", "I13", "I15"],
}
# IMPORTANT: mFI-5 conventionally requires hypertension *requiring medication*.
# Without medication linkage, ICD-only matching overestimates this item.
# Report this as a limitation; a sensitivity analysis on ASA + HFRS only avoids it.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ICD_PREFIX = re.compile(r"^([A-Z]\d{2})")  # 'F001' or 'F00.1' → 'F00'


def _category(code: str) -> str | None:
    if not code:
        return None
    cleaned = code.replace(".", "").upper().strip()
    m = _ICD_PREFIX.match(cleaned)
    return m.group(1) if m else None


def compute_hfrs(icd_codes: list[str]) -> tuple[float, str]:
    """Compute HFRS score and risk band from ICD-10 codes.

    Matching is by 3-character category, following Gilbert's SAS macro.
    Each category contributes its weight at most once per admission.

    Parameters
    ----------
    icd_codes : list[str]
        ICD-10 codes recorded for the admission.

    Returns
    -------
    tuple[float, str]
        ``(score, band)`` where *band* is ``'low'``, ``'intermediate'``,
        or ``'high'``.

    Examples
    --------
    >>> compute_hfrs(["F00.1", "W19", "N39.0", "I50.9", "I10"])
    (13.5, 'intermediate')
    """
    categories = {
        c for c in (_category(x) for x in icd_codes) if c is not None and c in HFRS_WEIGHTS
    }
    score = round(sum(HFRS_WEIGHTS[c] for c in categories), 1)

    if score < HFRS_THRESHOLDS["low_max"]:
        band = "low"
    elif score <= HFRS_THRESHOLDS["intermediate_max"]:
        band = "intermediate"
    else:
        band = "high"
    return score, band


def compute_mfi4(icd_codes: list[str]) -> tuple[int, dict[str, bool]]:
    """Compute mFI-4 score and per-item presence flags.

    Parameters
    ----------
    icd_codes : list[str]
        ICD-10 codes recorded for the admission.

    Returns
    -------
    tuple[int, dict[str, bool]]
        ``(score, items)`` where *score* is 0-4 and *items* maps each
        mFI-4 component to whether it was present.

    Examples
    --------
    >>> score, items = compute_mfi4(["I50.9", "I10"])
    >>> score
    2
    """
    categories = {c for c in (_category(x) for x in icd_codes) if c is not None}
    items = {name: any(p in categories for p in prefixes) for name, prefixes in MFI4_ITEMS.items()}
    return sum(items.values()), items


def composite_frailty_proxy(
    asa: int | None,
    hfrs_band: str | None,
    mfi4_score: int | None,
) -> str:
    """Three-level composite: ``'non_frail'`` / ``'pre_frail'`` / ``'frail'``.

    Frail if ANY of: ASA >= 4, HFRS high, mFI-4 >= 2.
    Pre-frail if ANY of: ASA = 3, HFRS intermediate, mFI-4 = 1.
    Else non-frail. Missing inputs are conservatively ignored.

    Parameters
    ----------
    asa : int | None
        ASA physical status classification (1-5), or ``None`` if unknown.
    hfrs_band : str | None
        HFRS risk band (``'low'``, ``'intermediate'``, ``'high'``),
        or ``None`` if unknown.
    mfi4_score : int | None
        mFI-4 score (0-4), or ``None`` if unknown.

    Returns
    -------
    str
        One of ``'non_frail'``, ``'pre_frail'``, ``'frail'``.

    Examples
    --------
    >>> composite_frailty_proxy(asa=3, hfrs_band='intermediate', mfi4_score=2)
    'frail'
    """
    if (
        (asa is not None and asa >= 4)
        or hfrs_band == "high"
        or (mfi4_score is not None and mfi4_score >= 2)
    ):
        return "frail"
    if asa == 3 or hfrs_band == "intermediate" or mfi4_score == 1:
        return "pre_frail"
    return "non_frail"
