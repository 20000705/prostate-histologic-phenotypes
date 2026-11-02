#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16c_test_peso_pc3_external_compartment_association.py

Step 16c v1.0.0

Formal statistical testing of the frozen PESO PC3 external compartment
validation.

EVIDENCE CHAIN
--------------
Step 16a:
    freezes cohort, target axis, endpoint, mask hierarchy, QC threshold,
    independent unit, association statistic, extreme-tail definition,
    bootstrap-over-slides inference.

Step 16b:
    performs measurement only and writes frozen per-slide NPZ files.

Step 16c:
    performs statistical inference only from Step-16b measurements.

ABSOLUTE RULES
--------------
Step 16c MUST NOT:
- read raw WSI;
- read raw masks;
- re-project embeddings;
- recompute epithelial fractions;
- redefine mask labels;
- redefine mask geometry;
- redefine minimum labeled fraction;
- redefine PC3;
- introduce tile-level pooled inference;
- select slides based on observed PC3 association;
- select a different PC axis;
- introduce a hybrid mask analysis.

FORMAL PRIMARY ANALYSIS
-----------------------
Population:
    62 PESO training slides.

Mask:
    frozen standard mask.

Within-slide estimand:
    Spearman rho(
        frozen PC3 tile score,
        epithelial_fraction_among_labeled_pixels
    )

Tiles:
    Step-16b frozen standard-mask QC-pass tiles only.

Independent unit:
    slide.

Cohort primary estimand:
    median slide-level Spearman rho.

Uncertainty:
    percentile bootstrap CI over slides.

Formal test:
    two-sided Wilcoxon signed-rank test of slide-level rho against zero.

Expected biological direction:
    rho < 0.

PRE-SPECIFIED CORRECTED-MASK SENSITIVITY
----------------------------------------
Repeat the same association analysis on the frozen 25-slide corrected-mask
subset, using corrected-mask Step-16b QC-pass measurements.

SUPPORTIVE EXTREME ANALYSIS
---------------------------
Within each slide:
    low  = PC3 <= within-slide 10th percentile
    high = PC3 >= within-slide 90th percentile

Effect:
    median(epithelial fraction | PC3-high)
    -
    median(epithelial fraction | PC3-low)

Expected direction:
    negative.

The 10th/90th percentile definition is inherited from Step 16a and is NOT
chosen in Step 16c.

POST-QC SUPPORTIVE COMMON-TILE SENSITIVITY
------------------------------------------
Step 16b revealed substantially lower spatial annotation coverage in the
corrected masks. Therefore, on the 25 corrected-mask slides, Step 16c also
performs a transparent post-QC supportive analysis restricted to identical
physical tiles satisfying BOTH:

    standard qc_pass == True
    corrected qc_pass == True

On those identical tiles, PC3 association is calculated separately using:
    1. standard epithelial fraction;
    2. corrected epithelial fraction.

This analysis is NOT treated as a pre-specified confirmatory endpoint.
Its purpose is to distinguish mask-source effects from spatial-support effects.

NO TILE-LEVEL P-VALUES ARE PERMITTED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    DATA_ROOT,
    RESULTS_ROOT,
    DISCOVERY_EMBEDDINGS,
    DISCOVERY_ANNOTATIONS,
    DISCOVERY_WSI,
    FROZEN_PCA_MODEL,
    FROZEN_DIRECTIONS,
    PESO_ROOT as CONFIG_PESO_ROOT,
    AGGC_ROOT as CONFIG_AGGC_ROOT,
    TCGA_ROOT as CONFIG_TCGA_ROOT,
)

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, wilcoxon
except ImportError as exc:
    raise ImportError(
        "Step 16c requires scipy for Spearman correlation and "
        "Wilcoxon signed-rank testing."
    ) from exc


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.0"
SCHEMA_VERSION = "16c-peso-pc3-association-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

STEP16A_DIR = (
    RESULTS_ROOT
    / "16a_peso_pc3_external_compartment_specification"
)

STEP16A_PLAN = (
    STEP16A_DIR
    / "peso_pc3_external_analysis_plan.json"
)

STEP16A_METADATA = (
    STEP16A_DIR
    / "analysis_metadata.json"
)

STEP16B_DIR = (
    RESULTS_ROOT
    / "16b_peso_pc3_external_compartment_measurements"
)

STEP16B_MANIFEST = (
    STEP16B_DIR
    / "peso_pc3_external_measurement_manifest.csv"
)

STEP16B_QC = (
    STEP16B_DIR
    / "peso_pc3_external_slide_qc_summary.csv"
)

STEP16B_FAILURES = (
    STEP16B_DIR
    / "peso_pc3_external_measurement_failures.csv"
)

STEP16B_METADATA = (
    STEP16B_DIR
    / "analysis_metadata.json"
)

OUT_DIR = (
    RESULTS_ROOT
    / "16c_peso_pc3_external_compartment_association"
)

OUT_PRIMARY_SLIDES = (
    OUT_DIR
    / "peso_pc3_primary_slide_associations.csv"
)

OUT_PRIMARY_COHORT = (
    OUT_DIR
    / "peso_pc3_primary_cohort_summary.csv"
)

OUT_CORRECTED_SLIDES = (
    OUT_DIR
    / "peso_pc3_corrected_sensitivity_slide_associations.csv"
)

OUT_CORRECTED_COHORT = (
    OUT_DIR
    / "peso_pc3_corrected_sensitivity_cohort_summary.csv"
)

OUT_COMMON_TILE = (
    OUT_DIR
    / "peso_pc3_common_tile_mask_source_sensitivity.csv"
)

OUT_COMMON_TILE_SUMMARY = (
    OUT_DIR
    / "peso_pc3_common_tile_mask_source_summary.csv"
)

OUT_BOOTSTRAP = (
    OUT_DIR
    / "peso_pc3_slide_bootstrap_distributions.npz"
)

OUT_METADATA = (
    OUT_DIR
    / "analysis_metadata.json"
)

OUT_README = (
    OUT_DIR
    / "README.txt"
)

CORE_OUTPUTS = [
    OUT_PRIMARY_SLIDES,
    OUT_PRIMARY_COHORT,
    OUT_CORRECTED_SLIDES,
    OUT_CORRECTED_COHORT,
    OUT_COMMON_TILE,
    OUT_COMMON_TILE_SUMMARY,
    OUT_BOOTSTRAP,
    OUT_METADATA,
    OUT_README,
]


# =============================================================================
# FIXED COMPUTATIONAL SETTINGS
#
# These do NOT redefine scientific endpoints.
# They control numerical reproducibility only.
# =============================================================================

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260807
CI_LEVEL = 0.95

EXPECTED_PRIMARY_SLIDES = 62
EXPECTED_CORRECTED_SLIDES = 25

EXPECTED_DIRECTION = "negative"

MIN_VALID_TILES_FOR_CORRELATION = 3


# =============================================================================
# BASIC HELPERS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow intentional replacement of existing Step-16c outputs.",
    )

    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {label}: {path}"
        )


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )


def guard_outputs(overwrite: bool):
    existing = [
        p for p in CORE_OUTPUTS
        if p.exists()
    ]

    if existing and not overwrite:
        raise RuntimeError(
            "Step-16c outputs already exist; refusing silent overwrite.\n"
            + "\n".join(
                f"  - {p}"
                for p in existing
            )
            + "\nUse --overwrite only if intentional."
        )


def finite_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan

    if not np.isfinite(x):
        return np.nan

    return x


# =============================================================================
# STEP-16A PLAN VALIDATION
# =============================================================================

def validate_step16a_plan(plan: Dict):
    """
    Step 16c does not choose the analysis.
    It verifies that the required analysis was frozen in Step 16a.
    """

    agg = plan.get(
        "aggregation_and_inference",
        {},
    )

    sampling = plan.get(
        "formal_sampling",
        {},
    )

    endpoint = plan.get(
        "primary_endpoint",
        {},
    )

    independent_unit = str(
        agg.get(
            "independent_unit",
            "",
        )
    ).strip().lower()

    if independent_unit != "slide":
        raise RuntimeError(
            "Step-16a independent unit is not frozen as slide."
        )

    continuous_effect = str(
        agg.get(
            "continuous_slide_effect",
            "",
        )
    ).strip().lower()

    if "spearman" not in continuous_effect:
        raise RuntimeError(
            "Step-16a continuous slide effect is not Spearman rho."
        )

    cohort_summary = str(
        agg.get(
            "cohort_summary",
            "",
        )
    ).strip().lower()

    if "median" not in cohort_summary:
        raise RuntimeError(
            "Step-16a cohort summary is not median slide-level rho."
        )

    confidence_interval = str(
        agg.get(
            "confidence_interval",
            "",
        )
    ).strip().lower()

    if "bootstrap" not in confidence_interval:
        raise RuntimeError(
            "Step-16a confidence interval is not bootstrap over slides."
        )

    two_sided = agg.get(
        "two_sided_testing",
        None,
    )

    if two_sided is not True:
        raise RuntimeError(
            "Step-16a did not freeze two-sided testing=True."
        )

    expected_direction = str(
        endpoint.get(
            "expected_pc3_direction",
            "",
        )
    ).strip().lower()

    if expected_direction != EXPECTED_DIRECTION:
        raise RuntimeError(
            "Step-16a expected PC3 direction is not negative."
        )

    extreme = sampling.get(
        "pc3_extreme_analysis",
        {},
    )

    definition = str(
        extreme.get(
            "definition",
            "",
        )
    ).lower()

    low_text = str(
        extreme.get(
            "low",
            "",
        )
    ).lower()

    high_text = str(
        extreme.get(
            "high",
            "",
        )
    ).lower()

    combined = " ".join(
        [
            definition,
            low_text,
            high_text,
        ]
    )

    if (
        "10th" not in combined
        or "90th" not in combined
    ):
        raise RuntimeError(
            "Step-16a did not freeze the 10th/90th percentile "
            "PC3 extreme analysis."
        )

    return {
        "independent_unit": independent_unit,
        "continuous_slide_effect": continuous_effect,
        "cohort_summary": cohort_summary,
        "confidence_interval": confidence_interval,
        "two_sided_testing": two_sided,
        "expected_direction": expected_direction,
        "extreme_low_quantile": 0.10,
        "extreme_high_quantile": 0.90,
    }


# =============================================================================
# STEP-16B VALIDATION
# =============================================================================

def validate_step16b_metadata(meta16b: Dict):
    if not bool(
        meta16b.get(
            "formal_full_run",
            False,
        )
    ):
        raise RuntimeError(
            "Step-16b is not marked formal_full_run=True."
        )

    if not bool(
        meta16b.get(
            "measurement_only",
            False,
        )
    ):
        raise RuntimeError(
            "Step-16b is not marked measurement_only=True."
        )

    if bool(
        meta16b.get(
            "inferential_statistics_performed",
            True,
        )
    ):
        raise RuntimeError(
            "Step-16b metadata indicates inference was already performed."
        )

    n_primary = int(
        meta16b.get(
            "n_primary_standard_slides",
            -1,
        )
    )

    n_corrected = int(
        meta16b.get(
            "n_corrected_sensitivity_slides",
            -1,
        )
    )

    failures = int(
        meta16b.get(
            "n_measurement_failure_records",
            -1,
        )
    )

    if n_primary != EXPECTED_PRIMARY_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIMARY_SLIDES} primary slides "
            f"from Step 16b; found {n_primary}."
        )

    if n_corrected != EXPECTED_CORRECTED_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_CORRECTED_SLIDES} corrected slides "
            f"from Step 16b; found {n_corrected}."
        )

    if failures != 0:
        raise RuntimeError(
            f"Step-16b contains {failures} measurement failure records."
        )


def validate_measurement_manifest(
    manifest: pd.DataFrame,
):
    required = {
        "slide_id",
        "analysis_role",
        "mask_source",
        "npz_path",
    }

    missing = required - set(
        manifest.columns
    )

    if missing:
        raise RuntimeError(
            "Step-16b measurement manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    primary = manifest[
        manifest["analysis_role"]
        .astype(str)
        .str.lower()
        .eq("primary")
    ].copy()

    sensitivity = manifest[
        manifest["analysis_role"]
        .astype(str)
        .str.lower()
        .eq("sensitivity")
    ].copy()

    if len(primary) != EXPECTED_PRIMARY_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIMARY_SLIDES} primary manifest rows; "
            f"found {len(primary)}."
        )

    if primary["slide_id"].nunique() != EXPECTED_PRIMARY_SLIDES:
        raise RuntimeError(
            "Primary manifest does not contain 62 unique slides."
        )

    if len(sensitivity) != EXPECTED_CORRECTED_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_CORRECTED_SLIDES} sensitivity rows; "
            f"found {len(sensitivity)}."
        )

    if sensitivity["slide_id"].nunique() != EXPECTED_CORRECTED_SLIDES:
        raise RuntimeError(
            "Corrected manifest does not contain 25 unique slides."
        )

    primary_sources = set(
        primary["mask_source"]
        .astype(str)
        .str.lower()
        .unique()
    )

    sensitivity_sources = set(
        sensitivity["mask_source"]
        .astype(str)
        .str.lower()
        .unique()
    )

    if primary_sources != {"standard"}:
        raise RuntimeError(
            f"Primary mask source is not standard-only: "
            f"{primary_sources}"
        )

    if sensitivity_sources != {"corrected"}:
        raise RuntimeError(
            f"Sensitivity mask source is not corrected-only: "
            f"{sensitivity_sources}"
        )

    return primary, sensitivity


# =============================================================================
# NPZ VALIDATION
# =============================================================================

REQUIRED_NPZ_ARRAYS = {
    "tile_idx",
    "x",
    "y",
    "pc3_score",
    "qc_pass",
    "epithelial_fraction_among_labeled_pixels",
}


def load_measurement_npz(
    path: Path,
):
    require_file(
        path,
        "Step-16b per-slide measurement NPZ",
    )

    z = np.load(
        path,
        allow_pickle=False,
    )

    missing = (
        REQUIRED_NPZ_ARRAYS
        - set(z.files)
    )

    if missing:
        z.close()

        raise RuntimeError(
            f"{path} missing arrays: "
            + ", ".join(sorted(missing))
        )

    arrays = {
        key: np.asarray(z[key])
        for key in REQUIRED_NPZ_ARRAYS
    }

    z.close()

    lengths = {
        len(v)
        for v in arrays.values()
    }

    if len(lengths) != 1:
        raise RuntimeError(
            f"Inconsistent array lengths in {path}: "
            f"{sorted(lengths)}"
        )

    return arrays


def validate_coordinate_identity(
    standard: Dict[str, np.ndarray],
    corrected: Dict[str, np.ndarray],
    slide_id: str,
):
    """
    Corrected and standard Step-16b measurements must refer to exactly
    the same original physical tile indexing.
    """

    for key in [
        "tile_idx",
        "x",
        "y",
    ]:
        if not np.array_equal(
            standard[key],
            corrected[key],
        ):
            raise RuntimeError(
                f"{slide_id}: standard/corrected {key} arrays "
                f"are not identical."
            )

    if not np.array_equal(
        standard["pc3_score"],
        corrected["pc3_score"],
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{slide_id}: standard/corrected PC3 arrays differ."
        )


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, int, str]:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    n = len(x)

    if n < MIN_VALID_TILES_FOR_CORRELATION:
        return (
            np.nan,
            n,
            "too_few_valid_tiles",
        )

    if np.all(
        x == x[0]
    ):
        return (
            np.nan,
            n,
            "constant_pc3",
        )

    if np.all(
        y == y[0]
    ):
        return (
            np.nan,
            n,
            "constant_epithelial_fraction",
        )

    rho = spearmanr(
        x,
        y,
    ).statistic

    return (
        finite_float(rho),
        n,
        "ok",
    )


def compute_extreme_effect(
    pc3: np.ndarray,
    epi: np.ndarray,
    low_quantile: float,
    high_quantile: float,
):
    pc3 = np.asarray(
        pc3,
        dtype=np.float64,
    )

    epi = np.asarray(
        epi,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(pc3)
        & np.isfinite(epi)
    )

    x = pc3[valid]
    y = epi[valid]

    if len(x) == 0:
        return {
            "pc3_q10": np.nan,
            "pc3_q90": np.nan,
            "n_low": 0,
            "n_high": 0,
            "median_epi_low": np.nan,
            "median_epi_high": np.nan,
            "high_minus_low_median_epi": np.nan,
        }

    q_low = float(
        np.quantile(
            x,
            low_quantile,
        )
    )

    q_high = float(
        np.quantile(
            x,
            high_quantile,
        )
    )

    low = (
        x <= q_low
    )

    high = (
        x >= q_high
    )

    med_low = (
        float(np.median(y[low]))
        if low.any()
        else np.nan
    )

    med_high = (
        float(np.median(y[high]))
        if high.any()
        else np.nan
    )

    effect = (
        med_high - med_low
        if np.isfinite(med_high)
        and np.isfinite(med_low)
        else np.nan
    )

    return {
        "pc3_q10": q_low,
        "pc3_q90": q_high,
        "n_low": int(low.sum()),
        "n_high": int(high.sum()),
        "median_epi_low": med_low,
        "median_epi_high": med_high,
        "high_minus_low_median_epi": effect,
    }


def analyze_slide(
    slide_id: str,
    mask_source: str,
    arrays: Dict[str, np.ndarray],
    low_quantile: float,
    high_quantile: float,
):
    qc = np.asarray(
        arrays["qc_pass"],
        dtype=bool,
    )

    pc3 = np.asarray(
        arrays["pc3_score"],
        dtype=np.float64,
    )

    epi = np.asarray(
        arrays[
            "epithelial_fraction_among_labeled_pixels"
        ],
        dtype=np.float64,
    )

    valid = (
        qc
        & np.isfinite(pc3)
        & np.isfinite(epi)
    )

    x = pc3[valid]
    y = epi[valid]

    rho, n_used, status = safe_spearman(
        x,
        y,
    )

    extreme = compute_extreme_effect(
        x,
        y,
        low_quantile,
        high_quantile,
    )

    return {
        "slide_id": slide_id,
        "mask_source": mask_source,
        "n_tiles_total": int(len(pc3)),
        "n_qc_pass_tiles": int(qc.sum()),
        "n_tiles_used_for_association": int(n_used),
        "association_status": status,
        "spearman_rho": rho,
        **extreme,
    }


def percentile_bootstrap_median(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise RuntimeError(
            "Cannot bootstrap empty values."
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    bootstrap = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        n_bootstrap
    ):
        sample_idx = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap[i] = np.median(
            values[sample_idx]
        )

    alpha = (
        1.0 - ci_level
    )

    lower = float(
        np.quantile(
            bootstrap,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap,
            1.0 - alpha / 2.0,
        )
    )

    return (
        bootstrap,
        lower,
        upper,
    )


def safe_wilcoxon_against_zero(
    values: np.ndarray,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    if np.allclose(
        values,
        0.0,
    ):
        return 1.0

    try:
        result = wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return finite_float(
            result.pvalue
        )

    except ValueError:
        return np.nan


def summarize_slide_effects(
    df: pd.DataFrame,
    analysis_name: str,
    bootstrap_seed: int,
):
    rho = pd.to_numeric(
        df["spearman_rho"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    rho = rho[
        np.isfinite(rho)
    ]

    extreme = pd.to_numeric(
        df[
            "high_minus_low_median_epi"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    extreme = extreme[
        np.isfinite(extreme)
    ]

    if len(rho) == 0:
        raise RuntimeError(
            f"{analysis_name}: no finite slide-level Spearman rho values."
        )

    rho_boot, rho_low, rho_high = (
        percentile_bootstrap_median(
            rho,
            BOOTSTRAP_REPLICATES,
            bootstrap_seed,
            CI_LEVEL,
        )
    )

    if len(extreme):
        (
            extreme_boot,
            extreme_low,
            extreme_high,
        ) = percentile_bootstrap_median(
            extreme,
            BOOTSTRAP_REPLICATES,
            bootstrap_seed + 1,
            CI_LEVEL,
        )
    else:
        extreme_boot = np.array(
            [],
            dtype=np.float64,
        )

        extreme_low = np.nan
        extreme_high = np.nan

    row = {
        "analysis": analysis_name,

        "n_slides_total": int(len(df)),
        "n_slides_finite_rho": int(len(rho)),

        "median_spearman_rho": float(
            np.median(rho)
        ),

        "rho_bootstrap_ci_level": CI_LEVEL,
        "rho_bootstrap_ci_lower": rho_low,
        "rho_bootstrap_ci_upper": rho_high,

        "fraction_slides_rho_negative": float(
            np.mean(rho < 0)
        ),

        "n_slides_rho_negative": int(
            np.sum(rho < 0)
        ),

        "n_slides_rho_positive": int(
            np.sum(rho > 0)
        ),

        "n_slides_rho_zero": int(
            np.sum(rho == 0)
        ),

        "wilcoxon_two_sided_p_rho_vs_zero":
            safe_wilcoxon_against_zero(
                rho
            ),

        "n_slides_finite_extreme_effect": int(
            len(extreme)
        ),

        "median_high_minus_low_median_epi":
            float(
                np.median(extreme)
            )
            if len(extreme)
            else np.nan,

        "extreme_bootstrap_ci_level":
            CI_LEVEL,

        "extreme_bootstrap_ci_lower":
            extreme_low,

        "extreme_bootstrap_ci_upper":
            extreme_high,

        "fraction_slides_extreme_effect_negative":
            float(
                np.mean(extreme < 0)
            )
            if len(extreme)
            else np.nan,

        "wilcoxon_two_sided_p_extreme_vs_zero":
            safe_wilcoxon_against_zero(
                extreme
            )
            if len(extreme)
            else np.nan,
    }

    return (
        row,
        rho_boot,
        extreme_boot,
    )


# =============================================================================
# COMMON-TILE SUPPORTIVE SENSITIVITY
# =============================================================================

def analyze_common_tiles(
    slide_id: str,
    standard: Dict[str, np.ndarray],
    corrected: Dict[str, np.ndarray],
):
    validate_coordinate_identity(
        standard,
        corrected,
        slide_id,
    )

    pc3 = np.asarray(
        standard["pc3_score"],
        dtype=np.float64,
    )

    s_epi = np.asarray(
        standard[
            "epithelial_fraction_among_labeled_pixels"
        ],
        dtype=np.float64,
    )

    c_epi = np.asarray(
        corrected[
            "epithelial_fraction_among_labeled_pixels"
        ],
        dtype=np.float64,
    )

    s_qc = np.asarray(
        standard["qc_pass"],
        dtype=bool,
    )

    c_qc = np.asarray(
        corrected["qc_pass"],
        dtype=bool,
    )

    common = (
        s_qc
        & c_qc
        & np.isfinite(pc3)
        & np.isfinite(s_epi)
        & np.isfinite(c_epi)
    )

    x = pc3[common]
    ys = s_epi[common]
    yc = c_epi[common]

    rho_s, n_s, status_s = safe_spearman(
        x,
        ys,
    )

    rho_c, n_c, status_c = safe_spearman(
        x,
        yc,
    )

    if n_s != n_c:
        raise RuntimeError(
            f"{slide_id}: common-tile standard/corrected "
            f"association sample sizes differ."
        )

    return {
        "slide_id": slide_id,

        "n_standard_qc_pass_tiles": int(
            s_qc.sum()
        ),

        "n_corrected_qc_pass_tiles": int(
            c_qc.sum()
        ),

        "n_common_qc_pass_tiles": int(
            common.sum()
        ),

        "standard_association_status":
            status_s,

        "corrected_association_status":
            status_c,

        "rho_standard_common":
            rho_s,

        "rho_corrected_common":
            rho_c,

        "delta_rho_corrected_minus_standard":
            (
                rho_c - rho_s
                if np.isfinite(rho_s)
                and np.isfinite(rho_c)
                else np.nan
            ),
    }


def summarize_common_tile_analysis(
    common_df: pd.DataFrame,
):
    s = pd.to_numeric(
        common_df[
            "rho_standard_common"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    c = pd.to_numeric(
        common_df[
            "rho_corrected_common"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    valid = (
        np.isfinite(s)
        & np.isfinite(c)
    )

    s = s[valid]
    c = c[valid]

    if len(s) == 0:
        raise RuntimeError(
            "No valid common-tile slide-level association pairs."
        )

    delta = c - s

    if (
        len(s) >= 3
        and not np.all(s == s[0])
        and not np.all(c == c[0])
    ):
        rho_agreement = finite_float(
            spearmanr(
                s,
                c,
            ).statistic
        )
    else:
        rho_agreement = np.nan

    return {
        "analysis":
            "post_qc_common_tile_mask_source_sensitivity",

        "status":
            "post_qc_supportive_not_prespecified_confirmatory",

        "n_slides_total":
            int(len(common_df)),

        "n_slides_valid_paired_rho":
            int(len(s)),

        "median_rho_standard_common":
            float(np.median(s)),

        "median_rho_corrected_common":
            float(np.median(c)),

        "median_delta_rho_corrected_minus_standard":
            float(np.median(delta)),

        "spearman_across_slides_standard_vs_corrected_rho":
            rho_agreement,

        "fraction_standard_common_rho_negative":
            float(np.mean(s < 0)),

        "fraction_corrected_common_rho_negative":
            float(np.mean(c < 0)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print(
        "=" * 124
    )

    print(
        "STEP 16C: PESO PC3 EXTERNAL COMPARTMENT ASSOCIATION"
    )

    print(
        "=" * 124
    )

    print(
        "[VERSION]      ",
        VERSION,
    )

    print(
        "[SCHEMA]       ",
        SCHEMA_VERSION,
    )

    print(
        "[STEP 16A]     ",
        STEP16A_DIR,
    )

    print(
        "[STEP 16B]     ",
        STEP16B_DIR,
    )

    print(
        "[OUTPUT]       ",
        OUT_DIR,
    )

    print(
        "[OVERWRITE]    ",
        args.overwrite,
    )

    # -------------------------------------------------------------------------
    # Required inputs
    # -------------------------------------------------------------------------

    required_files = [
        (
            STEP16A_PLAN,
            "Step-16a analysis plan",
        ),
        (
            STEP16A_METADATA,
            "Step-16a metadata",
        ),
        (
            STEP16B_MANIFEST,
            "Step-16b measurement manifest",
        ),
        (
            STEP16B_QC,
            "Step-16b QC summary",
        ),
        (
            STEP16B_FAILURES,
            "Step-16b failure audit",
        ),
        (
            STEP16B_METADATA,
            "Step-16b metadata",
        ),
    ]

    for path, label in required_files:
        require_file(
            path,
            label,
        )

    guard_outputs(
        args.overwrite
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load provenance
    # -------------------------------------------------------------------------

    plan16a = load_json(
        STEP16A_PLAN
    )

    meta16a = load_json(
        STEP16A_METADATA
    )

    meta16b = load_json(
        STEP16B_METADATA
    )

    frozen = validate_step16a_plan(
        plan16a
    )

    validate_step16b_metadata(
        meta16b
    )

    manifest = pd.read_csv(
        STEP16B_MANIFEST,
        low_memory=False,
    )

    primary_manifest, corrected_manifest = (
        validate_measurement_manifest(
            manifest
        )
    )

    # -------------------------------------------------------------------------
    # Verify Step-16b failure table
    # -------------------------------------------------------------------------

    try:
        failures = pd.read_csv(
            STEP16B_FAILURES,
            low_memory=False,
        )

        if len(failures) != 0:
            raise RuntimeError(
                f"Step-16b failure audit contains "
                f"{len(failures)} rows."
            )

    except pd.errors.EmptyDataError:
        failures = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Verify Step-16b still points back to current Step-16a
    # -------------------------------------------------------------------------

    step16a_recorded = meta16b.get(
        "step16a",
        {},
    )

    current_plan_hash = sha256_file(
        STEP16A_PLAN
    )

    recorded_plan_hash = step16a_recorded.get(
        "plan_sha256"
    )

    if (
        recorded_plan_hash is not None
        and str(recorded_plan_hash)
        != str(current_plan_hash)
    ):
        raise RuntimeError(
            "Step-16a analysis plan hash no longer matches "
            "the version consumed by Step-16b."
        )

    current_meta16a_hash = sha256_file(
        STEP16A_METADATA
    )

    recorded_meta16a_hash = step16a_recorded.get(
        "metadata_sha256"
    )

    if (
        recorded_meta16a_hash is not None
        and str(recorded_meta16a_hash)
        != str(current_meta16a_hash)
    ):
        raise RuntimeError(
            "Step-16a metadata hash no longer matches "
            "the version consumed by Step-16b."
        )

    print()
    print(
        "=" * 124
    )

    print(
        "FROZEN INFERENTIAL DESIGN"
    )

    print(
        "=" * 124
    )

    print(
        "Independent unit          :",
        frozen["independent_unit"],
    )

    print(
        "Primary slide effect      : Spearman rho"
    )

    print(
        "Cohort primary estimand   : median slide-level rho"
    )

    print(
        "Expected direction        : rho < 0"
    )

    print(
        "Bootstrap unit            : slide"
    )

    print(
        "Bootstrap replicates      :",
        BOOTSTRAP_REPLICATES,
    )

    print(
        "Bootstrap seed            :",
        BOOTSTRAP_SEED,
    )

    print(
        "Formal test               : two-sided Wilcoxon signed-rank vs 0"
    )

    print(
        "Extreme tails             : within-slide <=10th / >=90th PC3"
    )

    print(
        "Tile-level pooled p-values: PROHIBITED"
    )

    # =========================================================================
    # PRIMARY ANALYSIS
    # =========================================================================

    print()
    print(
        "=" * 124
    )

    print(
        "PRIMARY ANALYSIS — STANDARD MASKS"
    )

    print(
        "=" * 124
    )

    primary_rows = []

    primary_npz_by_slide = {}

    primary_manifest = (
        primary_manifest
        .sort_values(
            "slide_id"
        )
        .reset_index(
            drop=True
        )
    )

    for i, row in primary_manifest.iterrows():

        slide_id = str(
            row["slide_id"]
        )

        npz_path = Path(
            str(
                row["npz_path"]
            )
        )

        arrays = load_measurement_npz(
            npz_path
        )

        primary_npz_by_slide[
            slide_id
        ] = arrays

        result = analyze_slide(
            slide_id=slide_id,
            mask_source="standard",
            arrays=arrays,
            low_quantile=frozen[
                "extreme_low_quantile"
            ],
            high_quantile=frozen[
                "extreme_high_quantile"
            ],
        )

        primary_rows.append(
            result
        )

        print(
            f"[PRIMARY {i+1:02d}/{len(primary_manifest):02d}] "
            f"{slide_id} | "
            f"n={result['n_tiles_used_for_association']} | "
            f"rho={result['spearman_rho']:.4f} | "
            f"high-low={result['high_minus_low_median_epi']:.4f}"
        )

    primary_df = pd.DataFrame(
        primary_rows
    )

    primary_summary, primary_rho_boot, primary_extreme_boot = (
        summarize_slide_effects(
            primary_df,
            analysis_name=(
                "primary_standard_mask"
            ),
            bootstrap_seed=BOOTSTRAP_SEED,
        )
    )

    primary_summary_df = pd.DataFrame(
        [primary_summary]
    )

    # =========================================================================
    # CORRECTED SENSITIVITY
    # =========================================================================

    print()
    print(
        "=" * 124
    )

    print(
        "PRE-SPECIFIED SENSITIVITY — CORRECTED MASKS"
    )

    print(
        "=" * 124
    )

    corrected_rows = []

    corrected_npz_by_slide = {}

    corrected_manifest = (
        corrected_manifest
        .sort_values(
            "slide_id"
        )
        .reset_index(
            drop=True
        )
    )

    for i, row in corrected_manifest.iterrows():

        slide_id = str(
            row["slide_id"]
        )

        npz_path = Path(
            str(
                row["npz_path"]
            )
        )

        arrays = load_measurement_npz(
            npz_path
        )

        corrected_npz_by_slide[
            slide_id
        ] = arrays

        result = analyze_slide(
            slide_id=slide_id,
            mask_source="corrected",
            arrays=arrays,
            low_quantile=frozen[
                "extreme_low_quantile"
            ],
            high_quantile=frozen[
                "extreme_high_quantile"
            ],
        )

        corrected_rows.append(
            result
        )

        print(
            f"[CORRECTED {i+1:02d}/{len(corrected_manifest):02d}] "
            f"{slide_id} | "
            f"n={result['n_tiles_used_for_association']} | "
            f"rho={result['spearman_rho']:.4f} | "
            f"high-low={result['high_minus_low_median_epi']:.4f}"
        )

    corrected_df = pd.DataFrame(
        corrected_rows
    )

    (
        corrected_summary,
        corrected_rho_boot,
        corrected_extreme_boot,
    ) = summarize_slide_effects(
        corrected_df,
        analysis_name=(
            "prespecified_corrected_mask_sensitivity"
        ),
        bootstrap_seed=BOOTSTRAP_SEED + 100,
    )

    corrected_summary_df = pd.DataFrame(
        [corrected_summary]
    )

    # =========================================================================
    # POST-QC COMMON-TILE SUPPORTIVE SENSITIVITY
    # =========================================================================

    print()
    print(
        "=" * 124
    )

    print(
        "POST-QC SUPPORTIVE SENSITIVITY — COMMON PHYSICAL TILES"
    )

    print(
        "=" * 124
    )

    print(
        "[STATUS] Added after Step-16b revealed lower corrected-mask "
        "annotation coverage."
    )

    print(
        "[ROLE]   Supportive only; not a pre-specified confirmatory endpoint."
    )

    common_rows = []

    for i, slide_id in enumerate(
        sorted(
            corrected_npz_by_slide
        ),
        1,
    ):

        if slide_id not in primary_npz_by_slide:
            raise RuntimeError(
                f"{slide_id}: corrected slide missing corresponding "
                f"standard measurement."
            )

        result = analyze_common_tiles(
            slide_id=slide_id,
            standard=primary_npz_by_slide[
                slide_id
            ],
            corrected=corrected_npz_by_slide[
                slide_id
            ],
        )

        common_rows.append(
            result
        )

        print(
            f"[COMMON {i:02d}/{len(corrected_npz_by_slide):02d}] "
            f"{slide_id} | "
            f"n={result['n_common_qc_pass_tiles']} | "
            f"rho_std={result['rho_standard_common']:.4f} | "
            f"rho_corr={result['rho_corrected_common']:.4f} | "
            f"delta={result['delta_rho_corrected_minus_standard']:.4f}"
        )

    common_df = pd.DataFrame(
        common_rows
    )

    common_summary = (
        summarize_common_tile_analysis(
            common_df
        )
    )

    common_summary_df = pd.DataFrame(
        [common_summary]
    )

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    primary_df.to_csv(
        OUT_PRIMARY_SLIDES,
        index=False,
    )

    primary_summary_df.to_csv(
        OUT_PRIMARY_COHORT,
        index=False,
    )

    corrected_df.to_csv(
        OUT_CORRECTED_SLIDES,
        index=False,
    )

    corrected_summary_df.to_csv(
        OUT_CORRECTED_COHORT,
        index=False,
    )

    common_df.to_csv(
        OUT_COMMON_TILE,
        index=False,
    )

    common_summary_df.to_csv(
        OUT_COMMON_TILE_SUMMARY,
        index=False,
    )

    np.savez_compressed(
        OUT_BOOTSTRAP,

        primary_rho_median_bootstrap=
            primary_rho_boot,

        primary_extreme_median_bootstrap=
            primary_extreme_boot,

        corrected_rho_median_bootstrap=
            corrected_rho_boot,

        corrected_extreme_median_bootstrap=
            corrected_extreme_boot,

        bootstrap_replicates=np.array(
            [BOOTSTRAP_REPLICATES],
            dtype=np.int64,
        ),

        bootstrap_seed=np.array(
            [BOOTSTRAP_SEED],
            dtype=np.int64,
        ),

        ci_level=np.array(
            [CI_LEVEL],
            dtype=np.float64,
        ),
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "step": "16c",

        "version": VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),

        "analysis_role":
            "formal external statistical inference",

        "target":
            "PC3",

        "primary_endpoint":
            "epithelial_fraction_among_labeled_pixels",

        "expected_direction":
            "negative",

        "independent_unit":
            "slide",

        "tile_level_pooled_inference_performed":
            False,

        "feature_selection_performed":
            False,

        "pc_axis_screening_performed":
            False,

        "mask_measurement_recomputed":
            False,

        "raw_mask_accessed":
            False,

        "raw_wsi_accessed":
            False,

        "hybrid_mask_analysis_performed":
            False,

        "primary_analysis": {
            "mask_source":
                "standard",

            "n_expected_slides":
                EXPECTED_PRIMARY_SLIDES,

            "slide_effect":
                "Spearman rho",

            "cohort_estimand":
                "median slide-level Spearman rho",

            "confidence_interval":
                "percentile bootstrap over slides",

            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,

            "bootstrap_seed":
                BOOTSTRAP_SEED,

            "formal_test":
                "two-sided Wilcoxon signed-rank versus zero",
        },

        "supportive_extreme_analysis": {
            "low":
                "<= within-slide PC3 10th percentile",

            "high":
                ">= within-slide PC3 90th percentile",

            "effect":
                (
                    "median epithelial fraction high PC3 "
                    "minus median epithelial fraction low PC3"
                ),

            "expected_direction":
                "negative",
        },

        "prespecified_corrected_sensitivity": {
            "mask_source":
                "corrected",

            "n_expected_slides":
                EXPECTED_CORRECTED_SLIDES,
        },

        "post_qc_common_tile_sensitivity": {
            "status":
                (
                    "post-QC supportive sensitivity; "
                    "not pre-specified confirmatory"
                ),

            "reason":
                (
                    "Step 16b revealed substantially lower spatial "
                    "annotation coverage in corrected masks."
                ),

            "restriction":
                (
                    "identical physical tiles passing both standard "
                    "and corrected Step-16b QC"
                ),
        },

        "step16a": {
            "version":
                meta16a.get("version"),

            "schema_version":
                meta16a.get("schema_version"),

            "analysis_plan_sha256":
                sha256_file(
                    STEP16A_PLAN
                ),

            "metadata_sha256":
                sha256_file(
                    STEP16A_METADATA
                ),
        },

        "step16b": {
            "version":
                meta16b.get("version"),

            "schema_version":
                meta16b.get(
                    "schema_version"
                ),

            "measurement_manifest_sha256":
                sha256_file(
                    STEP16B_MANIFEST
                ),

            "qc_summary_sha256":
                sha256_file(
                    STEP16B_QC
                ),

            "failure_audit_sha256":
                sha256_file(
                    STEP16B_FAILURES
                ),

            "metadata_sha256":
                sha256_file(
                    STEP16B_METADATA
                ),
        },

        "n_primary_slides_analyzed":
            int(
                len(primary_df)
            ),

        "n_corrected_slides_analyzed":
            int(
                len(corrected_df)
            ),

        "n_common_tile_slides_analyzed":
            int(
                len(common_df)
            ),

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,
    }

    write_json(
        OUT_METADATA,
        metadata,
    )

    # =========================================================================
    # README
    # =========================================================================

    OUT_README.write_text(
        f"""STEP 16C — PESO PC3 EXTERNAL COMPARTMENT ASSOCIATION
=======================================================

Role
----
Formal statistical inference using only frozen Step-16b measurements.

No raw WSI, raw mask, embedding, or PCA projection is accessed or recomputed
in Step 16c.

Primary analysis
----------------
Cohort:
    {EXPECTED_PRIMARY_SLIDES} PESO training slides.

Mask:
    frozen standard mask.

Within-slide effect:
    Spearman rho between frozen PC3 tile score and
    epithelial_fraction_among_labeled_pixels among Step-16b QC-pass tiles.

Independent unit:
    slide.

Primary cohort estimand:
    median slide-level Spearman rho.

Expected direction:
    negative.

Confidence interval:
    percentile bootstrap over slides.

Bootstrap replicates:
    {BOOTSTRAP_REPLICATES}

Bootstrap seed:
    {BOOTSTRAP_SEED}

Formal test:
    two-sided Wilcoxon signed-rank test of slide-level rho versus zero.

No pooled tile-level inferential test is performed.

Supportive extreme analysis
---------------------------
Within each slide:

    low PC3:
        <= within-slide 10th percentile

    high PC3:
        >= within-slide 90th percentile

Effect:

    median epithelial fraction in high-PC3 tiles
    minus
    median epithelial fraction in low-PC3 tiles

Expected direction:
    negative.

Corrected-mask sensitivity
--------------------------
The frozen corrected-mask subset ({EXPECTED_CORRECTED_SLIDES} slides) is
analyzed independently using the same slide-level association framework.

Post-QC common-tile sensitivity
-------------------------------
Step 16b showed substantially lower spatial annotation coverage for corrected
masks. Therefore an additional supportive analysis is performed on the
{EXPECTED_CORRECTED_SLIDES} corrected-mask slides using only identical
physical tiles that pass BOTH standard-mask and corrected-mask QC.

This analysis is explicitly classified as:

    post-QC supportive sensitivity

and NOT as a pre-specified confirmatory endpoint.

Its purpose is to distinguish changes caused by mask source from changes caused
by different spatial support.

Evidence hierarchy
------------------
1. PRIMARY CONFIRMATORY:
       62 standard-mask slides.

2. PRE-SPECIFIED SENSITIVITY:
       25 corrected-mask slides.

3. POST-QC SUPPORTIVE SENSITIVITY:
       common physical tiles on the 25 corrected-mask slides.

Interpretation boundary
-----------------------
Step 16c tests only PC3.

No PC1-PC8 screening is performed.
No PC6 analysis is performed.
No feature selection is performed.
No hybrid mask analysis is performed.
No tile-level pseudoreplicated p-values are produced.
""",
        encoding="utf-8",
    )

    # =========================================================================
    # TERMINAL SUMMARY
    # =========================================================================

    print()
    print(
        "=" * 124
    )

    print(
        "PRIMARY RESULT"
    )

    print(
        "=" * 124
    )

    print(
        primary_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 124
    )

    print(
        "CORRECTED-MASK SENSITIVITY"
    )

    print(
        "=" * 124
    )

    print(
        corrected_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 124
    )

    print(
        "COMMON-TILE MASK-SOURCE SENSITIVITY"
    )

    print(
        "=" * 124
    )

    print(
        common_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 124
    )

    print(
        "STEP 16C SUMMARY"
    )

    print(
        "=" * 124
    )

    print(
        "Primary slides              :",
        len(primary_df),
    )

    print(
        "Corrected sensitivity slides:",
        len(corrected_df),
    )

    print(
        "Common-tile slides          :",
        len(common_df),
    )

    print(
        "Independent unit            : slide"
    )

    print(
        "Pooled tile inference       : NONE"
    )

    print(
        "PC-axis screening           : NONE"
    )

    print(
        "Mask remeasurement          : NONE"
    )

    print(
        "Hybrid mask analysis        : NONE"
    )

    print()
    print(
        "[OK] Saved:"
    )

    for path in CORE_OUTPUTS:
        print(
            " ",
            path,
        )

    print()
    print(
        "[DONE] Step-16c PESO PC3 external compartment "
        "association complete."
    )


if __name__ == "__main__":
    main()