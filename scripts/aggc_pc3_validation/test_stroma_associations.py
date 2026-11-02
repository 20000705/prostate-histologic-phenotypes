#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17c_test_aggc_pc3_external_stroma_associations.py

STEP 17C
========
Formal AGGC2022 Subset-1 PC3 external Stroma association / inference.

ROLE
----
ASSOCIATION + INFERENCE ONLY.

This script consumes ONLY:
    - frozen Step-17a specification
    - Step-17b tile-level measurements
    - Step-17b slide-level measurement QC

It MUST NOT:
    - redefine cohort
    - redefine masks
    - redefine annotation semantics
    - redefine geometry
    - redefine labeled-fraction threshold
    - redefine minimum tiles per slide
    - redefine PC3
    - refit PCA
    - flip PC3 sign
    - screen PC1-PC8
    - analyze PC6
    - modify Train/Test partition roles
    - create a post hoc replication pass threshold
    - treat tiles as independent units for cohort p-values
    - exclude slides based on observed association strength

FROZEN PRIMARY ANALYSIS
-----------------------
For every Step-17b-measured slide meeting the frozen minimum tile count:

    within-slide Spearman rho(
        PC3,
        stroma_fraction_among_annotated_pixels
    )

using only:
    primary_qc_pass == True

Expected direction:
    rho > 0

Independent unit:
    slide

TRAIN:
    formal external validation

TEST:
    separate replication using EXACT SAME specification

Cohort summaries:
    median slide-level rho
    95% bootstrap CI of median over slides
    two-sided Wilcoxon signed-rank test vs zero
    n / fraction slides rho > 0

FROZEN SUPPORTIVE EXTREME ANALYSIS
----------------------------------
Within every eligible slide:
    PC3 lowest 10% vs highest 10%
among primary-QC tiles.

Effect:
    median(stroma fraction in high-PC3 tiles)
    -
    median(stroma fraction in low-PC3 tiles)

Expected:
    > 0

FROZEN SUPPORTIVE OVERLAP SENSITIVITY
-------------------------------------
Use:
    primary_qc_pass == True
    AND
    overlap_free_tile == True

Repeat within-slide Spearman.

No new threshold is introduced.

NO POOLED TILE-LEVEL INFERENCE IS PERMITTED.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
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

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.0"
SCHEMA_VERSION = "17c-aggc-pc3-external-stroma-association-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

MEDIA_BASE = REPO_ROOT

STEP17A = (
    RESULTS_ROOT
    / "17a_aggc_pc3_external_stroma_specification"
)

STEP17B = (
    RESULTS_ROOT
    / "17b_aggc_pc3_external_stroma_measurements"
)

PLAN_JSON = STEP17A / "aggc_pc3_external_stroma_analysis_plan.json"
STEP17A_META = STEP17A / "analysis_metadata.json"

MEASUREMENT_MANIFEST = STEP17B / "aggc_pc3_measurement_manifest.csv"
SLIDE_QC = STEP17B / "aggc_pc3_slide_measurement_qc.csv"
STEP17B_META = STEP17B / "analysis_metadata.json"

OUT_DIR = (
    RESULTS_ROOT
    / "17c_aggc_pc3_external_stroma_association"
)

OUT_PRIMARY_PER_SLIDE = (
    OUT_DIR
    / "aggc_pc3_primary_association_per_slide.csv"
)

OUT_PRIMARY_SUMMARY = (
    OUT_DIR
    / "aggc_pc3_primary_association_summary.csv"
)

OUT_EXTREME_PER_SLIDE = (
    OUT_DIR
    / "aggc_pc3_extreme_analysis_per_slide.csv"
)

OUT_EXTREME_SUMMARY = (
    OUT_DIR
    / "aggc_pc3_extreme_analysis_summary.csv"
)

OUT_OVERLAP_PER_SLIDE = (
    OUT_DIR
    / "aggc_pc3_overlap_free_sensitivity_per_slide.csv"
)

OUT_OVERLAP_SUMMARY = (
    OUT_DIR
    / "aggc_pc3_overlap_free_sensitivity_summary.csv"
)

OUT_TRAIN_TEST_CONCORDANCE = (
    OUT_DIR
    / "aggc_pc3_train_test_concordance_summary.csv"
)

OUT_BOOTSTRAP_PRIMARY = (
    OUT_DIR
    / "aggc_pc3_primary_bootstrap_distribution.csv.gz"
)

OUT_BOOTSTRAP_EXTREME = (
    OUT_DIR
    / "aggc_pc3_extreme_bootstrap_distribution.csv.gz"
)

OUT_BOOTSTRAP_OVERLAP = (
    OUT_DIR
    / "aggc_pc3_overlap_free_bootstrap_distribution.csv.gz"
)

OUT_ANALYSIS_AUDIT = (
    OUT_DIR
    / "analysis_audit.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "analysis_metadata.json"
)

OUT_README = (
    OUT_DIR
    / "README.txt"
)


# =============================================================================
# BEHAVIOR
# =============================================================================

OVERWRITE = False

PARTITION_ORDER = [
    "train_validation",
    "test_replication",
]


# =============================================================================
# HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required {label}: {path}"
        )


def require_dir(path: Path, label: str):
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing required {label}: {path}"
        )


def prepare_output():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        if not OVERWRITE:
            raise RuntimeError(
                f"Step17c output already exists: {OUT_DIR}\n"
                "Do not overwrite formal inference casually."
            )

        for p in OUT_DIR.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_json(path: Path, obj):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def to_bool(x) -> bool:
    if pd.isna(x):
        return False

    if isinstance(
        x,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(x)

    text = str(
        x
    ).strip().lower()

    if text in {
        "true",
        "1",
        "1.0",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "0.0",
        "no",
        "",
    }:
        return False

    raise ValueError(
        f"Cannot parse Boolean value: {x!r}"
    )


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, int]:

    valid = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = x[
        valid
    ]

    y = y[
        valid
    ]

    n = len(
        x
    )

    if n < 3:
        return np.nan, n

    if (
        np.nanstd(
            x
        )
        == 0
        or
        np.nanstd(
            y
        )
        == 0
    ):
        return np.nan, n

    result = spearmanr(
        x,
        y,
    )

    return (
        float(
            result.statistic
        ),
        n,
    )


def safe_wilcoxon(
    values: np.ndarray,
    alternative: str,
) -> Tuple[float, float, int]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    n = len(
        values
    )

    if n == 0:
        return np.nan, np.nan, 0

    # scipy wilcoxon can fail when all differences are zero.
    if np.all(
        values
        ==
        0
    ):
        return 0.0, 1.0, n

    result = wilcoxon(
        values,
        alternative=alternative,
        zero_method="wilcox",
        correction=False,
        method="auto",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
        n,
    )


def bootstrap_median(
    values: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> Tuple[
    float,
    float,
    np.ndarray,
]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        return (
            np.nan,
            np.nan,
            np.array(
                [],
                dtype=float,
            ),
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        values
    )

    distribution = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        distribution[
            i
        ] = np.median(
            sample
        )

    alpha = (
        1.0
        -
        ci_level
    )

    low = float(
        np.quantile(
            distribution,
            alpha / 2.0,
        )
    )

    high = float(
        np.quantile(
            distribution,
            1.0 - alpha / 2.0,
        )
    )

    return (
        low,
        high,
        distribution,
    )


def summarize_slide_effects(
    partition: str,
    values: np.ndarray,
    expected_positive: bool,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
    wilcoxon_alternative: str,
    metric_name: str,
) -> Tuple[
    Dict,
    pd.DataFrame,
]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    n = len(
        values
    )

    if n == 0:

        summary = {
            "partition":
                partition,

            "metric":
                metric_name,

            "n_slides":
                0,

            "mean":
                np.nan,

            "median":
                np.nan,

            "std":
                np.nan,

            "q25":
                np.nan,

            "q75":
                np.nan,

            "min":
                np.nan,

            "max":
                np.nan,

            "bootstrap_ci_low":
                np.nan,

            "bootstrap_ci_high":
                np.nan,

            "bootstrap_replicates":
                n_bootstrap,

            "wilcoxon_statistic":
                np.nan,

            "wilcoxon_p_two_sided":
                np.nan,

            "n_expected_direction":
                0,

            "fraction_expected_direction":
                np.nan,
        }

        return (
            summary,
            pd.DataFrame(),
        )

    ci_low, ci_high, boot = bootstrap_median(
        values=values,
        n_bootstrap=n_bootstrap,
        seed=seed,
        ci_level=ci_level,
    )

    wilcox_stat, wilcox_p, wilcox_n = safe_wilcoxon(
        values=values,
        alternative=wilcoxon_alternative,
    )

    if expected_positive:

        expected_mask = (
            values
            >
            0
        )

    else:

        expected_mask = (
            values
            <
            0
        )

    summary = {
        "partition":
            partition,

        "metric":
            metric_name,

        "n_slides":
            int(
                n
            ),

        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if n > 1
            else 0.0,

        "q25":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),

        "q75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "bootstrap_ci_low":
            ci_low,

        "bootstrap_ci_high":
            ci_high,

        "bootstrap_replicates":
            int(
                n_bootstrap
            ),

        "bootstrap_ci_level":
            float(
                ci_level
            ),

        "wilcoxon_statistic":
            wilcox_stat,

        "wilcoxon_p_two_sided":
            wilcox_p,

        "wilcoxon_n":
            int(
                wilcox_n
            ),

        "n_expected_direction":
            int(
                expected_mask.sum()
            ),

        "fraction_expected_direction":
            float(
                expected_mask.mean()
            ),
    }

    boot_df = pd.DataFrame(
        {
            "partition":
                partition,

            "metric":
                metric_name,

            "bootstrap_index":
                np.arange(
                    len(
                        boot
                    ),
                    dtype=int,
                ),

            "bootstrap_median":
                boot,
        }
    )

    return (
        summary,
        boot_df,
    )


# =============================================================================
# FROZEN DESIGN
# =============================================================================

def read_frozen_design():

    plan = load_json(
        PLAN_JSON
    )

    step17a_meta = load_json(
        STEP17A_META
    )

    step17b_meta = load_json(
        STEP17B_META
    )

    if step17a_meta.get(
        "role"
    ) != "freeze_only":

        raise RuntimeError(
            "Step17a role is not freeze_only."
        )

    if step17b_meta.get(
        "role"
    ) != "measurement_only":

        raise RuntimeError(
            "Step17b role is not measurement_only."
        )

    if step17b_meta.get(
        "association_calculated"
    ) is not False:

        raise RuntimeError(
            "Step17b metadata indicates association was already calculated."
        )

    if step17b_meta.get(
        "spearman_calculated"
    ) is not False:

        raise RuntimeError(
            "Step17b metadata indicates Spearman was calculated."
        )

    if step17b_meta.get(
        "bootstrap_performed"
    ) is not False:

        raise RuntimeError(
            "Step17b metadata indicates bootstrap was already performed."
        )

    association = plan[
        "primary_association"
    ]

    train_inference = plan[
        "train_inference"
    ]

    extreme = plan[
        "supportive_extreme_analysis"
    ]

    overlap = plan[
        "supportive_overlap_sensitivity"
    ]

    frozen = {
        "endpoint":
            plan[
                "primary_endpoint"
            ],

        "within_slide_statistic":
            association[
                "within_slide_statistic"
            ],

        "expected_sign":
            association[
                "expected_sign"
            ],

        "min_tiles_per_slide":
            int(
                plan[
                    "primary_slide_qc"
                ][
                    "minimum_qc_passing_tiles"
                ]
            ),

        "bootstrap_replicates":
            int(
                train_inference[
                    "bootstrap_replicates"
                ]
            ),

        "bootstrap_seed":
            int(
                train_inference[
                    "bootstrap_seed"
                ]
            ),

        "bootstrap_ci_level":
            float(
                train_inference[
                    "bootstrap_ci_level"
                ]
            ),

        "wilcoxon_alternative":
            train_inference[
                "wilcoxon"
            ][
                "alternative"
            ],

        "extreme_enabled":
            bool(
                extreme[
                    "enabled"
                ]
            ),

        "extreme_quantile":
            float(
                extreme[
                    "within_slide_quantile"
                ]
            ),

        "extreme_endpoint":
            extreme[
                "endpoint"
            ],

        "overlap_enabled":
            bool(
                overlap[
                    "enabled"
                ]
            ),
    }

    if frozen[
        "endpoint"
    ] != (
        "stroma_fraction_among_annotated_pixels"
    ):

        raise RuntimeError(
            "17c does not match frozen endpoint."
        )

    if frozen[
        "within_slide_statistic"
    ].lower() != (
        "spearman rank correlation"
    ):

        raise RuntimeError(
            "17c does not match frozen primary statistic."
        )

    if frozen[
        "expected_sign"
    ] != "positive":

        raise RuntimeError(
            "17c expected sign does not match frozen positive direction."
        )

    if frozen[
        "extreme_endpoint"
    ] != frozen[
        "endpoint"
    ]:

        raise RuntimeError(
            "Frozen extreme endpoint differs from primary endpoint."
        )

    return (
        plan,
        step17a_meta,
        step17b_meta,
        frozen,
    )


# =============================================================================
# PER-SLIDE ANALYSES
# =============================================================================

def analyze_one_slide(
    measurement_path: Path,
    partition: str,
    slide_id: str,
    frozen: Dict,
) -> Tuple[
    Dict,
    Dict,
    Dict,
]:

    require_file(
        measurement_path,
        f"{slide_id} Step17b measurement",
    )

    df = pd.read_csv(
        measurement_path,
        compression="gzip",
        low_memory=False,
    )

    required = {
        "pc3",
        "stroma_fraction_among_annotated_pixels",
        "primary_qc_pass",
        "overlap_free_tile",
    }

    if not required.issubset(
        df.columns
    ):

        missing = sorted(
            required
            -
            set(
                df.columns
            )
        )

        raise RuntimeError(
            f"{slide_id}: missing measurement columns {missing}"
        )

    primary_mask = df[
        "primary_qc_pass"
    ].map(
        to_bool
    )

    primary_df = df.loc[
        primary_mask
    ].copy()

    n_primary = len(
        primary_df
    )

    if n_primary < frozen[
        "min_tiles_per_slide"
    ]:

        raise RuntimeError(
            f"{slide_id}: Step17b primary-QC tile count "
            f"{n_primary} < frozen minimum "
            f"{frozen['min_tiles_per_slide']}."
        )

    # -------------------------------------------------------------------------
    # PRIMARY
    # -------------------------------------------------------------------------

    rho, n_corr = safe_spearman(
        primary_df[
            "pc3"
        ].to_numpy(
            dtype=float
        ),
        primary_df[
            frozen[
                "endpoint"
            ]
        ].to_numpy(
            dtype=float
        ),
    )

    primary_row = {
        "partition":
            partition,

        "slide_id":
            slide_id,

        "n_primary_qc_tiles":
            int(
                n_primary
            ),

        "n_tiles_used_for_spearman":
            int(
                n_corr
            ),

        "spearman_rho":
            rho,

        "expected_direction_positive":
            bool(
                np.isfinite(
                    rho
                )
                and
                rho > 0
            ),

        "pc3_median_primary_qc":
            float(
                np.nanmedian(
                    primary_df[
                        "pc3"
                    ].to_numpy(
                        dtype=float
                    )
                )
            ),

        "endpoint_median_primary_qc":
            float(
                np.nanmedian(
                    primary_df[
                        frozen[
                            "endpoint"
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                )
            ),
    }

    # -------------------------------------------------------------------------
    # EXTREME SUPPORTIVE
    # -------------------------------------------------------------------------

    q = frozen[
        "extreme_quantile"
    ]

    pc3_vals = primary_df[
        "pc3"
    ].to_numpy(
        dtype=float
    )

    low_thr = float(
        np.quantile(
            pc3_vals,
            q,
        )
    )

    high_thr = float(
        np.quantile(
            pc3_vals,
            1.0 - q,
        )
    )

    low = primary_df.loc[
        primary_df[
            "pc3"
        ]
        <=
        low_thr
    ]

    high = primary_df.loc[
        primary_df[
            "pc3"
        ]
        >=
        high_thr
    ]

    low_endpoint = low[
        frozen[
            "endpoint"
        ]
    ].to_numpy(
        dtype=float
    )

    high_endpoint = high[
        frozen[
            "endpoint"
        ]
    ].to_numpy(
        dtype=float
    )

    low_med = float(
        np.nanmedian(
            low_endpoint
        )
    )

    high_med = float(
        np.nanmedian(
            high_endpoint
        )
    )

    extreme_effect = (
        high_med
        -
        low_med
    )

    extreme_row = {
        "partition":
            partition,

        "slide_id":
            slide_id,

        "n_primary_qc_tiles":
            int(
                n_primary
            ),

        "extreme_quantile":
            q,

        "pc3_low_threshold":
            low_thr,

        "pc3_high_threshold":
            high_thr,

        "n_low_tiles":
            int(
                len(
                    low
                )
            ),

        "n_high_tiles":
            int(
                len(
                    high
                )
            ),

        "low_pc3_endpoint_median":
            low_med,

        "high_pc3_endpoint_median":
            high_med,

        "high_minus_low_endpoint":
            extreme_effect,

        "expected_direction_positive":
            bool(
                np.isfinite(
                    extreme_effect
                )
                and
                extreme_effect > 0
            ),
    }

    # -------------------------------------------------------------------------
    # OVERLAP-FREE SUPPORTIVE SENSITIVITY
    # -------------------------------------------------------------------------

    overlap_free = primary_df[
        "overlap_free_tile"
    ].map(
        to_bool
    )

    overlap_df = primary_df.loc[
        overlap_free
    ].copy()

    overlap_rho, overlap_n = safe_spearman(
        overlap_df[
            "pc3"
        ].to_numpy(
            dtype=float
        ),
        overlap_df[
            frozen[
                "endpoint"
            ]
        ].to_numpy(
            dtype=float
        ),
    )

    overlap_row = {
        "partition":
            partition,

        "slide_id":
            slide_id,

        "n_primary_qc_tiles":
            int(
                n_primary
            ),

        "n_overlap_free_primary_qc_tiles":
            int(
                len(
                    overlap_df
                )
            ),

        "overlap_free_retention_fraction":
            float(
                len(
                    overlap_df
                )
                /
                n_primary
            )
            if n_primary
            else np.nan,

        "n_tiles_used_for_spearman":
            int(
                overlap_n
            ),

        "spearman_rho_overlap_free":
            overlap_rho,

        "expected_direction_positive":
            bool(
                np.isfinite(
                    overlap_rho
                )
                and
                overlap_rho > 0
            ),
    }

    return (
        primary_row,
        extreme_row,
        overlap_row,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 124)
    print("STEP 17C: AGGC PC3 EXTERNAL STROMA ASSOCIATION / INFERENCE")
    print("=" * 124)

    print("[VERSION]   ", VERSION)
    print("[SCHEMA]    ", SCHEMA_VERSION)
    print("[STEP 17A]  ", STEP17A)
    print("[STEP 17B]  ", STEP17B)
    print("[OUTPUT]    ", OUT_DIR)
    print("[OVERWRITE] ", OVERWRITE)

    print()
    print("[ROLE]      ASSOCIATION + INFERENCE")
    print("[UNIT]      SLIDE")
    print("[POOL TILES] NO")
    print("[PC REFIT]   NO")
    print("[PC SCREEN]  NO")
    print("[PC6]        NO")

    require_dir(
        STEP17A,
        "Step17a directory",
    )

    require_dir(
        STEP17B,
        "Step17b directory",
    )

    for path, label in [
        (
            PLAN_JSON,
            "Step17a analysis plan",
        ),
        (
            STEP17A_META,
            "Step17a metadata",
        ),
        (
            MEASUREMENT_MANIFEST,
            "Step17b measurement manifest",
        ),
        (
            SLIDE_QC,
            "Step17b slide QC",
        ),
        (
            STEP17B_META,
            "Step17b metadata",
        ),
    ]:

        require_file(
            path,
            label,
        )

    plan, step17a_meta, step17b_meta, frozen = read_frozen_design()

    prepare_output()

    print()
    print("[FROZEN DESIGN READ]")
    print(
        f"  Endpoint                  : {frozen['endpoint']}"
    )
    print(
        f"  Statistic                 : {frozen['within_slide_statistic']}"
    )
    print(
        f"  Expected sign             : {frozen['expected_sign']}"
    )
    print(
        f"  Minimum tiles / slide     : {frozen['min_tiles_per_slide']}"
    )
    print(
        f"  Bootstrap replicates      : {frozen['bootstrap_replicates']}"
    )
    print(
        f"  Bootstrap seed            : {frozen['bootstrap_seed']}"
    )
    print(
        f"  Extreme quantile          : {frozen['extreme_quantile']}"
    )
    print(
        f"  Overlap sensitivity       : {frozen['overlap_enabled']}"
    )

    # =========================================================================
    # Validate measurement manifest / QC
    # =========================================================================

    manifest = pd.read_csv(
        MEASUREMENT_MANIFEST,
        low_memory=False,
    )

    slide_qc = pd.read_csv(
        SLIDE_QC,
        low_memory=False,
    )

    if not manifest[
        "status"
    ].eq(
        "measured"
    ).all():

        bad = manifest.loc[
            ~manifest[
                "status"
            ].eq(
                "measured"
            )
        ]

        raise RuntimeError(
            "Step17b contains failed slides; formal 17c cannot proceed:\n"
            +
            bad.to_string(
                index=False
            )
        )

    if not slide_qc[
        "slide_meets_frozen_minimum_tile_count"
    ].map(
        to_bool
    ).all():

        bad = slide_qc.loc[
            ~slide_qc[
                "slide_meets_frozen_minimum_tile_count"
            ].map(
                to_bool
            )
        ]

        raise RuntimeError(
            "Some Step17b slides do not meet frozen minimum tile count:\n"
            +
            bad[
                [
                    "partition",
                    "slide_id",
                    "n_tiles_primary_qc_pass",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Analyze slides
    # =========================================================================

    primary_rows: List[
        Dict
    ] = []

    extreme_rows: List[
        Dict
    ] = []

    overlap_rows: List[
        Dict
    ] = []

    audit_rows: List[
        Dict
    ] = []

    for partition in PARTITION_ORDER:

        sub = manifest.loc[
            manifest[
                "partition"
            ].eq(
                partition
            )
        ].copy()

        expected_n = (
            int(
                step17a_meta[
                    "train_validation_n"
                ]
            )
            if partition
            ==
            "train_validation"
            else int(
                step17a_meta[
                    "test_replication_n"
                ]
            )
        )

        if len(
            sub
        ) != expected_n:

            raise RuntimeError(
                f"{partition}: Step17b measured slides={len(sub)} "
                f"but frozen Step17a count={expected_n}."
            )

        print()
        print(
            f"[PARTITION] {partition} | slides={len(sub)}"
        )

        for i, (_, row) in enumerate(
            sub.iterrows(),
            start=1,
        ):

            sid = str(
                row[
                    "slide_id"
                ]
            )

            measurement_path = Path(
                str(
                    row[
                        "per_tile_measurement_path"
                    ]
                )
            )

            print(
                f"[RUN] {partition} "
                f"[{i:03d}/{len(sub):03d}] "
                f"{sid}"
            )

            primary, extreme, overlap = analyze_one_slide(
                measurement_path=measurement_path,
                partition=partition,
                slide_id=sid,
                frozen=frozen,
            )

            primary_rows.append(
                primary
            )

            extreme_rows.append(
                extreme
            )

            overlap_rows.append(
                overlap
            )

            audit_rows.append(
                {
                    "partition":
                        partition,

                    "slide_id":
                        sid,

                    "measurement_path":
                        str(
                            measurement_path
                        ),

                    "measurement_sha256":
                        sha256_file(
                            measurement_path
                        ),

                    "primary_rho_finite":
                        bool(
                            np.isfinite(
                                primary[
                                    "spearman_rho"
                                ]
                            )
                        ),

                    "extreme_effect_finite":
                        bool(
                            np.isfinite(
                                extreme[
                                    "high_minus_low_endpoint"
                                ]
                            )
                        ),

                    "overlap_rho_finite":
                        bool(
                            np.isfinite(
                                overlap[
                                    "spearman_rho_overlap_free"
                                ]
                            )
                        ),
                }
            )

            print(
                f"[OK] {sid} | "
                f"rho={primary['spearman_rho']:.4f} | "
                f"extreme_delta={extreme['high_minus_low_endpoint']:.4f} | "
                f"overlap_rho={overlap['spearman_rho_overlap_free']:.4f}"
            )

    primary_df = pd.DataFrame(
        primary_rows
    )

    extreme_df = pd.DataFrame(
        extreme_rows
    )

    overlap_df = pd.DataFrame(
        overlap_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    # =========================================================================
    # Formal cohort-level summaries
    # =========================================================================

    primary_summary_rows = []
    extreme_summary_rows = []
    overlap_summary_rows = []

    primary_boot_frames = []
    extreme_boot_frames = []
    overlap_boot_frames = []

    for partition_index, partition in enumerate(
        PARTITION_ORDER
    ):

        # Independent deterministic bootstrap stream per partition/analysis.
        base_seed = (
            frozen[
                "bootstrap_seed"
            ]
            +
            partition_index
            *
            100000
        )

        # ---------------------------------------------------------------------
        # Primary
        # ---------------------------------------------------------------------

        vals = (
            primary_df.loc[
                primary_df[
                    "partition"
                ].eq(
                    partition
                ),
                "spearman_rho",
            ]
            .to_numpy(
                dtype=float
            )
        )

        summary, boot = summarize_slide_effects(
            partition=partition,
            values=vals,
            expected_positive=True,
            n_bootstrap=frozen[
                "bootstrap_replicates"
            ],
            seed=base_seed + 1,
            ci_level=frozen[
                "bootstrap_ci_level"
            ],
            wilcoxon_alternative=frozen[
                "wilcoxon_alternative"
            ],
            metric_name="slide_level_spearman_rho",
        )

        primary_summary_rows.append(
            summary
        )

        primary_boot_frames.append(
            boot
        )

        # ---------------------------------------------------------------------
        # Extreme
        # ---------------------------------------------------------------------

        vals = (
            extreme_df.loc[
                extreme_df[
                    "partition"
                ].eq(
                    partition
                ),
                "high_minus_low_endpoint",
            ]
            .to_numpy(
                dtype=float
            )
        )

        summary, boot = summarize_slide_effects(
            partition=partition,
            values=vals,
            expected_positive=True,
            n_bootstrap=frozen[
                "bootstrap_replicates"
            ],
            seed=base_seed + 2,
            ci_level=frozen[
                "bootstrap_ci_level"
            ],
            wilcoxon_alternative=frozen[
                "wilcoxon_alternative"
            ],
            metric_name="high_minus_low_stroma_fraction",
        )

        extreme_summary_rows.append(
            summary
        )

        extreme_boot_frames.append(
            boot
        )

        # ---------------------------------------------------------------------
        # Overlap-free sensitivity
        # ---------------------------------------------------------------------

        vals = (
            overlap_df.loc[
                overlap_df[
                    "partition"
                ].eq(
                    partition
                ),
                "spearman_rho_overlap_free",
            ]
            .to_numpy(
                dtype=float
            )
        )

        summary, boot = summarize_slide_effects(
            partition=partition,
            values=vals,
            expected_positive=True,
            n_bootstrap=frozen[
                "bootstrap_replicates"
            ],
            seed=base_seed + 3,
            ci_level=frozen[
                "bootstrap_ci_level"
            ],
            wilcoxon_alternative=frozen[
                "wilcoxon_alternative"
            ],
            metric_name="slide_level_spearman_rho_overlap_free",
        )

        overlap_summary_rows.append(
            summary
        )

        overlap_boot_frames.append(
            boot
        )

    primary_summary_df = pd.DataFrame(
        primary_summary_rows
    )

    extreme_summary_df = pd.DataFrame(
        extreme_summary_rows
    )

    overlap_summary_df = pd.DataFrame(
        overlap_summary_rows
    )

    primary_boot_df = pd.concat(
        primary_boot_frames,
        ignore_index=True,
    )

    extreme_boot_df = pd.concat(
        extreme_boot_frames,
        ignore_index=True,
    )

    overlap_boot_df = pd.concat(
        overlap_boot_frames,
        ignore_index=True,
    )

    # =========================================================================
    # Train-Test concordance summary — descriptive only
    # =========================================================================

    concordance_rows = []

    for metric_name, summary_df in [
        (
            "primary_spearman_rho",
            primary_summary_df,
        ),
        (
            "extreme_high_minus_low",
            extreme_summary_df,
        ),
        (
            "overlap_free_spearman_rho",
            overlap_summary_df,
        ),
    ]:

        train = summary_df.loc[
            summary_df[
                "partition"
            ].eq(
                "train_validation"
            )
        ].iloc[
            0
        ]

        test = summary_df.loc[
            summary_df[
                "partition"
            ].eq(
                "test_replication"
            )
        ].iloc[
            0
        ]

        concordance_rows.append(
            {
                "metric":
                    metric_name,

                "train_n":
                    int(
                        train[
                            "n_slides"
                        ]
                    ),

                "train_median":
                    float(
                        train[
                            "median"
                        ]
                    ),

                "train_ci_low":
                    float(
                        train[
                            "bootstrap_ci_low"
                        ]
                    ),

                "train_ci_high":
                    float(
                        train[
                            "bootstrap_ci_high"
                        ]
                    ),

                "train_fraction_expected_direction":
                    float(
                        train[
                            "fraction_expected_direction"
                        ]
                    ),

                "train_wilcoxon_p_two_sided":
                    float(
                        train[
                            "wilcoxon_p_two_sided"
                        ]
                    ),

                "test_n":
                    int(
                        test[
                            "n_slides"
                        ]
                    ),

                "test_median":
                    float(
                        test[
                            "median"
                        ]
                    ),

                "test_ci_low":
                    float(
                        test[
                            "bootstrap_ci_low"
                        ]
                    ),

                "test_ci_high":
                    float(
                        test[
                            "bootstrap_ci_high"
                        ]
                    ),

                "test_fraction_expected_direction":
                    float(
                        test[
                            "fraction_expected_direction"
                        ]
                    ),

                "test_wilcoxon_p_two_sided":
                    float(
                        test[
                            "wilcoxon_p_two_sided"
                        ]
                    ),

                "train_test_median_difference_test_minus_train":
                    float(
                        test[
                            "median"
                        ]
                        -
                        train[
                            "median"
                        ]
                    ),

                "new_replication_pass_threshold_introduced":
                    False,
            }
        )

    concordance_df = pd.DataFrame(
        concordance_rows
    )

    # =========================================================================
    # Save
    # =========================================================================

    primary_df.to_csv(
        OUT_PRIMARY_PER_SLIDE,
        index=False,
    )

    primary_summary_df.to_csv(
        OUT_PRIMARY_SUMMARY,
        index=False,
    )

    extreme_df.to_csv(
        OUT_EXTREME_PER_SLIDE,
        index=False,
    )

    extreme_summary_df.to_csv(
        OUT_EXTREME_SUMMARY,
        index=False,
    )

    overlap_df.to_csv(
        OUT_OVERLAP_PER_SLIDE,
        index=False,
    )

    overlap_summary_df.to_csv(
        OUT_OVERLAP_SUMMARY,
        index=False,
    )

    concordance_df.to_csv(
        OUT_TRAIN_TEST_CONCORDANCE,
        index=False,
    )

    primary_boot_df.to_csv(
        OUT_BOOTSTRAP_PRIMARY,
        index=False,
        compression="gzip",
    )

    extreme_boot_df.to_csv(
        OUT_BOOTSTRAP_EXTREME,
        index=False,
        compression="gzip",
    )

    overlap_boot_df.to_csv(
        OUT_BOOTSTRAP_OVERLAP,
        index=False,
        compression="gzip",
    )

    audit_df.to_csv(
        OUT_ANALYSIS_AUDIT,
        index=False,
    )

    # =========================================================================
    # Metadata
    # =========================================================================

    metadata = {
        "step":
            "17c",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),

        "role":
            "association_and_inference",

        "formal_independent_unit":
            "slide",

        "pooled_tile_inference":
            False,

        "primary_endpoint":
            frozen[
                "endpoint"
            ],

        "primary_within_slide_statistic":
            frozen[
                "within_slide_statistic"
            ],

        "expected_sign":
            frozen[
                "expected_sign"
            ],

        "bootstrap_replicates":
            frozen[
                "bootstrap_replicates"
            ],

        "bootstrap_seed_frozen_base":
            frozen[
                "bootstrap_seed"
            ],

        "bootstrap_ci_level":
            frozen[
                "bootstrap_ci_level"
            ],

        "wilcoxon_alternative":
            frozen[
                "wilcoxon_alternative"
            ],

        "supportive_extreme_analysis":
            frozen[
                "extreme_enabled"
            ],

        "supportive_overlap_sensitivity":
            frozen[
                "overlap_enabled"
            ],

        "new_threshold_introduced":
            False,

        "new_endpoint_introduced":
            False,

        "pc_screening":
            False,

        "pc6_analysis":
            False,

        "pca_refit":
            False,

        "pc3_sign_flip":
            False,

        "outcome_based_slide_exclusion":
            False,

        "train_test_pooled_primary_inference":
            False,

        "step17a_metadata_sha256":
            sha256_file(
                STEP17A_META
            ),

        "step17b_metadata_sha256":
            sha256_file(
                STEP17B_META
            ),

        "measurement_manifest_sha256":
            sha256_file(
                MEASUREMENT_MANIFEST
            ),

        "slide_qc_sha256":
            sha256_file(
                SLIDE_QC
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

    readme = f"""
STEP 17C — AGGC PC3 EXTERNAL STROMA ASSOCIATION / INFERENCE
============================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA_VERSION}


ROLE
----

Formal association and inference under the already-frozen Step17a plan.

Step17c does not modify the frozen design.


PRIMARY ANALYSIS
----------------

Within each eligible slide:

    Spearman rho(
        PC3,
        stroma_fraction_among_annotated_pixels
    )

using only Step17b:

    primary_qc_pass == True

Independent unit:

    slide

Expected direction:

    positive


TRAIN
-----

Role:

    formal external validation


TEST
----

Role:

    separate replication

The same endpoint, tile rules, slide rules, statistic, expected direction,
bootstrap framework, and Wilcoxon test are used.

No new Test-specific threshold is introduced.


COHORT INFERENCE
----------------

For each partition separately:

    median slide-level effect

    {int(frozen['bootstrap_ci_level'] * 100)}% slide-bootstrap CI

    bootstrap replicates:
        {frozen['bootstrap_replicates']}

    two-sided Wilcoxon signed-rank vs zero

    expected-direction consistency


SUPPORTIVE EXTREME ANALYSIS
---------------------------

Within each slide:

    lowest {int(frozen['extreme_quantile'] * 100)}% PC3

versus:

    highest {int(frozen['extreme_quantile'] * 100)}% PC3

Effect:

    median(high-PC3 Stroma fraction)
    -
    median(low-PC3 Stroma fraction)

Expected:

    positive


SUPPORTIVE OVERLAP-FREE SENSITIVITY
-----------------------------------

Use:

    primary_qc_pass == True
    AND
    overlap_free_tile == True

Then repeat slide-level Spearman.


PROHIBITED / NOT PERFORMED
--------------------------

No pooled tile-level p-value.

No PCA refit.

No PC1-PC8 screening.

No PC6 confirmatory analysis.

No PC3 sign flip.

No threshold tuning.

No endpoint tuning.

No outcome-based slide exclusion.

No Train/Test merging for primary inference.

No post hoc replication pass threshold.


OUTPUTS
-------

Primary per-slide:
    {OUT_PRIMARY_PER_SLIDE}

Primary summary:
    {OUT_PRIMARY_SUMMARY}

Extreme per-slide:
    {OUT_EXTREME_PER_SLIDE}

Extreme summary:
    {OUT_EXTREME_SUMMARY}

Overlap-free per-slide:
    {OUT_OVERLAP_PER_SLIDE}

Overlap-free summary:
    {OUT_OVERLAP_SUMMARY}

Train/Test concordance:
    {OUT_TRAIN_TEST_CONCORDANCE}

Bootstrap distributions:
    {OUT_BOOTSTRAP_PRIMARY}
    {OUT_BOOTSTRAP_EXTREME}
    {OUT_BOOTSTRAP_OVERLAP}

Analysis audit:
    {OUT_ANALYSIS_AUDIT}
""".strip() + "\n"

    OUT_README.write_text(
        readme,
        encoding="utf-8",
    )

    # =========================================================================
    # Console results
    # =========================================================================

    print()
    print("=" * 124)
    print("PRIMARY RESULTS")
    print("=" * 124)

    for _, row in primary_summary_df.iterrows():

        print()
        print(
            f"[{row['partition']}]"
        )

        print(
            f"Slides                     : {int(row['n_slides'])}"
        )

        print(
            f"Median Spearman rho        : {row['median']:+.6f}"
        )

        print(
            f"95% bootstrap CI           : "
            f"[{row['bootstrap_ci_low']:+.6f}, "
            f"{row['bootstrap_ci_high']:+.6f}]"
        )

        print(
            f"Slides rho > 0             : "
            f"{int(row['n_expected_direction'])}/"
            f"{int(row['n_slides'])} "
            f"({row['fraction_expected_direction']:.3%})"
        )

        print(
            f"Wilcoxon two-sided p       : "
            f"{row['wilcoxon_p_two_sided']:.6g}"
        )

    print()
    print("=" * 124)
    print("SUPPORTIVE EXTREME RESULTS")
    print("=" * 124)

    for _, row in extreme_summary_df.iterrows():

        print()
        print(
            f"[{row['partition']}]"
        )

        print(
            f"Slides                     : {int(row['n_slides'])}"
        )

        print(
            f"Median high-minus-low      : {row['median']:+.6f}"
        )

        print(
            f"95% bootstrap CI           : "
            f"[{row['bootstrap_ci_low']:+.6f}, "
            f"{row['bootstrap_ci_high']:+.6f}]"
        )

        print(
            f"Slides effect > 0          : "
            f"{int(row['n_expected_direction'])}/"
            f"{int(row['n_slides'])} "
            f"({row['fraction_expected_direction']:.3%})"
        )

        print(
            f"Wilcoxon two-sided p       : "
            f"{row['wilcoxon_p_two_sided']:.6g}"
        )

    print()
    print("=" * 124)
    print("OVERLAP-FREE SENSITIVITY")
    print("=" * 124)

    for _, row in overlap_summary_df.iterrows():

        print()
        print(
            f"[{row['partition']}]"
        )

        print(
            f"Slides                     : {int(row['n_slides'])}"
        )

        print(
            f"Median Spearman rho        : {row['median']:+.6f}"
        )

        print(
            f"95% bootstrap CI           : "
            f"[{row['bootstrap_ci_low']:+.6f}, "
            f"{row['bootstrap_ci_high']:+.6f}]"
        )

        print(
            f"Slides rho > 0             : "
            f"{int(row['n_expected_direction'])}/"
            f"{int(row['n_slides'])} "
            f"({row['fraction_expected_direction']:.3%})"
        )

        print(
            f"Wilcoxon two-sided p       : "
            f"{row['wilcoxon_p_two_sided']:.6g}"
        )

    print()
    print("=" * 124)
    print("TRAIN / TEST CONCORDANCE")
    print("=" * 124)

    print(
        concordance_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 124)
    print("ANALYSIS BOUNDARY AUDIT")
    print("=" * 124)

    print("Independent unit                 : SLIDE")
    print("Pooled tile-level inference      : NO")
    print("PCA refit                        : NO")
    print("PC screening                     : NO")
    print("PC6 analysis                     : NO")
    print("PC3 sign flip                    : NO")
    print("Threshold redefinition           : NO")
    print("Endpoint redefinition            : NO")
    print("Outcome-based slide exclusion    : NO")
    print("New replication threshold        : NO")

    print()
    print("[OK] Saved:")

    for path in [
        OUT_PRIMARY_PER_SLIDE,
        OUT_PRIMARY_SUMMARY,
        OUT_EXTREME_PER_SLIDE,
        OUT_EXTREME_SUMMARY,
        OUT_OVERLAP_PER_SLIDE,
        OUT_OVERLAP_SUMMARY,
        OUT_TRAIN_TEST_CONCORDANCE,
        OUT_BOOTSTRAP_PRIMARY,
        OUT_BOOTSTRAP_EXTREME,
        OUT_BOOTSTRAP_OVERLAP,
        OUT_ANALYSIS_AUDIT,
        OUT_METADATA,
        OUT_README,
    ]:

        print(
            " ",
            path,
        )

    print()
    print(
        "[DONE] Step-17c AGGC PC3 external Stroma association/inference complete."
    )


if __name__ == "__main__":
    main()
