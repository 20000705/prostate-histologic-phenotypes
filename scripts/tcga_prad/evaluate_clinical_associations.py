#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
28b_evaluate_frozen_tcga_prad_pc3_pc6_clinicopathologic_associations.py

STEP 28B
========
EVALUATE FROZEN TCGA-PRAD PC3 / PC6 CLINICOPATHOLOGIC ASSOCIATIONS

VERSION
-------
1.0.0

SCHEMA
------
28b-tcga-prad-pc3-pc6-clinicopathologic-association-schema-1.0

ROLE
----
PRESPECIFIED INFERENCE ONLY.

This step executes the clinicopathologic tests frozen in Step27G
using the analysis table assembled in Step28A.

NO endpoint discovery is permitted.

NO axis reselection is permitted.

NO PC sign tuning is permitted.

NO alternative patient aggregation is permitted.

NO exploratory covariate selection is permitted.

FROZEN TESTS
------------
PRIMARY_GRADE family:

    PC3 vs Grade Group
    PC6 vs Grade Group

SUPPORTIVE_PATHOLOGIC_T family:

    PC3 vs pathologic T stage
    PC6 vs pathologic T stage

STATISTIC
---------
Spearman rank correlation.

Alternative:
    two-sided

UNCERTAINTY
-----------
Patient-level bootstrap.

Replicates:
    10,000

Seed:
    2026

Confidence interval:
    percentile 95% CI

MULTIPLICITY
------------
Benjamini-Hochberg FDR within each prespecified two-test family.

INTERPRETATION
--------------
Because Step27G uses oriented PC scores:

    rho > 0
        higher oriented axis score is associated with
        more severe clinicopathologic endpoint.

    rho < 0
        higher oriented axis score is associated with
        less severe clinicopathologic endpoint.

IMPORTANT
---------
The script reports all four frozen tests regardless of significance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from scipy.stats import spearmanr


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "28b-tcga-prad-pc3-pc6-"
    "clinicopathologic-association-schema-1.0"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


DEFAULT_STEP28A_TABLE = (
    RESULTS
    / "28a_tcga_prad_clinicopathologic_analysis_table"
    / "tcga_prad_frozen_pc3_pc6_clinicopathologic_analysis_table.csv"
)

DEFAULT_STEP28A_SUMMARY = (
    RESULTS
    / "28a_tcga_prad_clinicopathologic_analysis_table"
    / "tcga_prad_clinicopathologic_analysis_table_summary.json"
)

DEFAULT_STEP28A_FIELD_MAPPING = (
    RESULTS
    / "28a_tcga_prad_clinicopathologic_analysis_table"
    / "tcga_prad_clinical_field_mapping.json"
)


DEFAULT_STEP27G_TEST_SPEC = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "tcga_prad_pc3_pc6_clinicopathologic_test_specification.csv"
)

DEFAULT_STEP27G_ENDPOINT_SPEC = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "tcga_prad_clinicopathologic_endpoint_specification.csv"
)

DEFAULT_STEP27G_SPEC = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "tcga_prad_clinicopathologic_evaluation_specification.json"
)


DEFAULT_OUT_DIR = (
    RESULTS
    / "28b_tcga_prad_pc3_pc6_clinicopathologic_associations"
)


# =============================================================================
# FROZEN SETTINGS
# =============================================================================

EXPECTED_PATIENTS = 403

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026

CI_LOW = 2.5
CI_HIGH = 97.5


AXIS_SCORE_COLUMNS = {
    "PC3": "patient_oriented_PC3",
    "PC6": "patient_oriented_PC6",
}


ENDPOINT_COLUMNS = {
    "GRADE_GROUP": {
        "value_column": "grade_group",
        "evaluable_column": "grade_group_evaluable",
        "family": "PRIMARY_GRADE",
        "family_label": "PRIMARY_GRADE_PC3_PC6",
        "role": "CONFIRMATORY",
    },

    "PATHOLOGIC_T_STAGE": {
        "value_column": "pathologic_t_ordinal",
        "evaluable_column": "pathologic_t_evaluable",
        "family": "SUPPORTIVE_PATHOLOGIC_T",
        "family_label": "SUPPORTIVE_PT_PC3_PC6",
        "role": "SUPPORTIVE",
    },
}


FROZEN_TEST_ORDER = [
    (
        "PC3",
        "GRADE_GROUP",
    ),
    (
        "PC6",
        "GRADE_GROUP",
    ),
    (
        "PC3",
        "PATHOLOGIC_T_STAGE",
    ),
    (
        "PC6",
        "PATHOLOGIC_T_STAGE",
    ),
]


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
    width: int = 112,
) -> None:

    print()
    print("=" * width)
    print(title)
    print("=" * width)


def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} missing: {path}"
        )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def save_json_atomic(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:

        while True:

            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# =============================================================================
# BH-FDR
# =============================================================================

def benjamini_hochberg(
    p_values: List[float],
) -> List[float]:

    p = np.asarray(
        p_values,
        dtype=np.float64,
    )

    if (
        p.ndim != 1
        or p.size == 0
    ):

        raise RuntimeError(
            "Invalid p-value vector for BH-FDR."
        )

    if not np.isfinite(
        p
    ).all():

        raise RuntimeError(
            "BH-FDR received non-finite p-values."
        )

    m = p.size

    order = np.argsort(
        p
    )

    ranked = p[
        order
    ]

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    q = np.empty_like(
        adjusted
    )

    q[
        order
    ] = adjusted

    return [
        float(
            value
        )
        for value in q
    ]


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[
    float,
    float,
    int,
]:

    if x.shape != y.shape:

        raise RuntimeError(
            "Bootstrap x/y shape mismatch."
        )

    n = x.shape[
        0
    ]

    if n < 3:

        raise RuntimeError(
            "Too few patients for bootstrap."
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_rho = np.full(
        n_bootstrap,
        np.nan,
        dtype=np.float64,
    )

    valid_count = 0

    for i in range(
        n_bootstrap
    ):

        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        xb = x[
            indices
        ]

        yb = y[
            indices
        ]

        # Some resamples can become constant due to ties / resampling.
        if (
            np.unique(
                xb
            ).size < 2
            or np.unique(
                yb
            ).size < 2
        ):

            continue

        result = spearmanr(
            xb,
            yb,
        )

        rho = float(
            result.statistic
        )

        if not np.isfinite(
            rho
        ):

            continue

        bootstrap_rho[
            i
        ] = rho

        valid_count += 1

    valid_values = bootstrap_rho[
        np.isfinite(
            bootstrap_rho
        )
    ]

    if valid_values.size < 0.95 * n_bootstrap:

        raise RuntimeError(
            "Too many invalid bootstrap replicates: "
            f"{valid_values.size}/{n_bootstrap}"
        )

    ci_low = float(
        np.percentile(
            valid_values,
            CI_LOW,
        )
    )

    ci_high = float(
        np.percentile(
            valid_values,
            CI_HIGH,
        )
    )

    return (
        ci_low,
        ci_high,
        int(
            valid_values.size
        ),
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_step27g(
    test_spec_path: Path,
    endpoint_spec_path: Path,
    spec_path: Path,
) -> None:

    require_file(
        test_spec_path,
        "Step27G test specification",
    )

    require_file(
        endpoint_spec_path,
        "Step27G endpoint specification",
    )

    require_file(
        spec_path,
        "Step27G JSON specification",
    )

    test_spec = pd.read_csv(
        test_spec_path
    )

    endpoint_spec = pd.read_csv(
        endpoint_spec_path
    )

    specification = load_json(
        spec_path
    )

    if (
        specification.get(
            "analysis_role"
        )
        != "PRE_EXECUTION_TCGA_CLINICOPATHOLOGIC_EVALUATION_FREEZE"
    ):

        raise RuntimeError(
            "Unexpected Step27G analysis role."
        )

    frozen_tests = set(
        zip(
            test_spec[
                "axis"
            ].astype(str),
            test_spec[
                "endpoint"
            ].astype(str),
        )
    )

    expected_tests = set(
        FROZEN_TEST_ORDER
    )

    if frozen_tests != expected_tests:

        raise RuntimeError(
            "Step27G frozen tests differ from expected tests."
        )

    if set(
        endpoint_spec[
            "endpoint"
        ].astype(str)
    ) != {
        "GRADE_GROUP",
        "PATHOLOGIC_T_STAGE",
    }:

        raise RuntimeError(
            "Step27G endpoint specification changed."
        )

    if int(
        specification.get(
            "uncertainty",
            {},
        ).get(
            "replicates",
            -1,
        )
    ) != BOOTSTRAP_REPLICATES:

        raise RuntimeError(
            "Step27G bootstrap replicate count changed."
        )

    if int(
        specification.get(
            "uncertainty",
            {},
        ).get(
            "seed",
            -1,
        )
    ) != BOOTSTRAP_SEED:

        raise RuntimeError(
            "Step27G bootstrap seed changed."
        )


def load_step28a_table(
    table_path: Path,
    summary_path: Path,
    field_mapping_path: Path,
) -> pd.DataFrame:

    require_file(
        table_path,
        "Step28A analysis table",
    )

    require_file(
        summary_path,
        "Step28A summary",
    )

    require_file(
        field_mapping_path,
        "Step28A field mapping",
    )

    df = pd.read_csv(
        table_path
    )

    summary = load_json(
        summary_path
    )

    field_mapping = load_json(
        field_mapping_path
    )

    if (
        summary.get(
            "analysis_role"
        )
        != "CLINICOPATHOLOGIC_TABLE_ASSEMBLY_ONLY"
    ):

        raise RuntimeError(
            "Unexpected Step28A analysis role."
        )

    if bool(
        summary.get(
            "association_calculated",
            True,
        )
    ):

        raise RuntimeError(
            "Step28A unexpectedly reports association calculation."
        )

    if bool(
        field_mapping.get(
            "association_calculated",
            True,
        )
    ):

        raise RuntimeError(
            "Step28A field mapping unexpectedly reports association."
        )

    required_columns = {
        "patient_id",

        "patient_oriented_PC3",
        "patient_oriented_PC6",

        "grade_group",
        "pathologic_t_ordinal",

        "grade_group_evaluable",
        "pathologic_t_evaluable",

        "inferential_unit",
        "association_calculated",
    }

    missing = sorted(
        required_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step28A analysis table schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        df
    ) != EXPECTED_PATIENTS:

        raise RuntimeError(
            "Step28A patient count changed."
        )

    if (
        df[
            "patient_id"
        ]
        .astype(str)
        .nunique()
        != EXPECTED_PATIENTS
    ):

        raise RuntimeError(
            "Step28A does not contain 403 unique patients."
        )

    if not (
        df[
            "inferential_unit"
        ].astype(str)
        == "patient"
    ).all():

        raise RuntimeError(
            "Step28A inferential unit is not uniformly patient."
        )

    if (
        df[
            "association_calculated"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
        .any()
    ):

        raise RuntimeError(
            "Step28A contains precomputed association flags."
        )

    return df


# =============================================================================
# TEST EXECUTION
# =============================================================================

def evaluate_one_test(
    df: pd.DataFrame,
    axis: str,
    endpoint: str,
    bootstrap_seed: int,
) -> Dict:

    if axis not in AXIS_SCORE_COLUMNS:

        raise RuntimeError(
            f"Unsupported axis: {axis}"
        )

    if endpoint not in ENDPOINT_COLUMNS:

        raise RuntimeError(
            f"Unsupported endpoint: {endpoint}"
        )

    score_column = (
        AXIS_SCORE_COLUMNS[
            axis
        ]
    )

    endpoint_info = (
        ENDPOINT_COLUMNS[
            endpoint
        ]
    )

    endpoint_column = (
        endpoint_info[
            "value_column"
        ]
    )

    evaluable_column = (
        endpoint_info[
            "evaluable_column"
        ]
    )

    mask = (
        df[
            evaluable_column
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )

    test_df = df.loc[
        mask,
        [
            "patient_id",
            score_column,
            endpoint_column,
        ],
    ].copy()

    test_df[
        score_column
    ] = pd.to_numeric(
        test_df[
            score_column
        ],
        errors="raise",
    )

    test_df[
        endpoint_column
    ] = pd.to_numeric(
        test_df[
            endpoint_column
        ],
        errors="raise",
    )

    finite = (
        np.isfinite(
            test_df[
                score_column
            ].to_numpy(
                dtype=np.float64
            )
        )
        &
        np.isfinite(
            test_df[
                endpoint_column
            ].to_numpy(
                dtype=np.float64
            )
        )
    )

    test_df = test_df.loc[
        finite
    ].copy()

    n = int(
        len(
            test_df
        )
    )

    if n < 3:

        raise RuntimeError(
            f"{axis} vs {endpoint}: insufficient evaluable patients."
        )

    x = test_df[
        score_column
    ].to_numpy(
        dtype=np.float64
    )

    y = test_df[
        endpoint_column
    ].to_numpy(
        dtype=np.float64
    )

    if np.unique(
        x
    ).size < 2:

        raise RuntimeError(
            f"{axis}: constant score vector."
        )

    if np.unique(
        y
    ).size < 2:

        raise RuntimeError(
            f"{endpoint}: constant endpoint vector."
        )

    result = spearmanr(
        x,
        y,
    )

    rho = float(
        result.statistic
    )

    p_value = float(
        result.pvalue
    )

    if not np.isfinite(
        rho
    ):

        raise RuntimeError(
            f"{axis} vs {endpoint}: non-finite Spearman rho."
        )

    if not np.isfinite(
        p_value
    ):

        raise RuntimeError(
            f"{axis} vs {endpoint}: non-finite p-value."
        )

    (
        ci_low,
        ci_high,
        bootstrap_valid,
    ) = bootstrap_spearman_ci(
        x=x,
        y=y,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        seed=bootstrap_seed,
    )

    if rho > 0:

        direction = (
            "higher_oriented_score_associated_with_"
            "greater_endpoint_severity"
        )

    elif rho < 0:

        direction = (
            "higher_oriented_score_associated_with_"
            "lower_endpoint_severity"
        )

    else:

        direction = "no_monotonic_direction"

    return {
        "axis": axis,

        "axis_score_column": (
            score_column
        ),

        "endpoint": endpoint,

        "endpoint_column": (
            endpoint_column
        ),

        "endpoint_family": (
            endpoint_info[
                "family"
            ]
        ),

        "multiplicity_family": (
            endpoint_info[
                "family_label"
            ]
        ),

        "analysis_role": (
            endpoint_info[
                "role"
            ]
        ),

        "n_patients": n,

        "spearman_rho": (
            rho
        ),

        "p_value": (
            p_value
        ),

        "bootstrap_ci_low": (
            ci_low
        ),

        "bootstrap_ci_high": (
            ci_high
        ),

        "bootstrap_replicates_requested": (
            BOOTSTRAP_REPLICATES
        ),

        "bootstrap_replicates_valid": (
            bootstrap_valid
        ),

        "bootstrap_seed": (
            bootstrap_seed
        ),

        "effect_direction": (
            direction
        ),
    }


# =============================================================================
# CLAIM BOUNDARY
# =============================================================================

def build_claim_boundary() -> str:

    return """STEP 28B — TCGA-PRAD CLINICOPATHOLOGIC INFERENCE CLAIM BOUNDARY

ROLE
----
PRESPECIFIED EXTERNAL CLINICOPATHOLOGIC ASSOCIATION TESTING.

FROZEN TESTS
------------
PRIMARY FAMILY:
    PC3 vs Grade Group
    PC6 vs Grade Group

SUPPORTIVE FAMILY:
    PC3 vs pathologic T
    PC6 vs pathologic T


IF SUPPORTED, RESULTS MAY ESTABLISH
----------------------------------
1. A frozen histologic axis is associated with clinicopathologic severity
   in the independent TCGA-PRAD cohort.

2. The direction and magnitude of the monotonic association may be reported.

3. PC3 and PC6 may show different strengths or directions of association,
   but formal PC3-vs-PC6 difference testing is NOT performed here.


THESE RESULTS DO NOT ESTABLISH
------------------------------
1. Tumor localization accuracy.
2. Diagnostic performance.
3. Grade prediction performance.
4. Prognostic value.
5. Clinical utility.
6. Added value beyond conventional pathology.
7. Molecular mechanism.
8. Causal biology.
9. Independence from Grade Group or stage.
10. Direct superiority of PC3 over PC6 or vice versa.


PROHIBITED POST-HOC ACTIONS
---------------------------
1. Reorienting PC3 or PC6 because of TCGA results.
2. Replacing PC3 or PC6 with another principal component.
3. Changing the Grade Group mapping after seeing results.
4. Changing the pathologic-T encoding after seeing results.
5. Changing patient aggregation after seeing results.
6. Removing a nonsignificant frozen test.
7. Promoting an unprespecified endpoint into the confirmatory family.
"""


# =============================================================================
# EXECUTION
# =============================================================================

def execute(
    step28a_table_path: Path,
    step28a_summary_path: Path,
    step28a_field_mapping_path: Path,
    step27g_test_spec_path: Path,
    step27g_endpoint_spec_path: Path,
    step27g_spec_path: Path,
    out_dir: Path,
) -> None:

    banner(
        "STEP 28B: EVALUATE FROZEN TCGA-PRAD "
        "PC3 / PC6 CLINICOPATHOLOGIC ASSOCIATIONS"
    )

    print(
        f"[VERSION]       {VERSION}"
    )

    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    print(
        f"[28A TABLE]     {step28a_table_path}"
    )

    # ------------------------------------------------------------------
    # Validate upstream frozen design.
    # ------------------------------------------------------------------

    validate_step27g(
        test_spec_path=(
            step27g_test_spec_path
        ),
        endpoint_spec_path=(
            step27g_endpoint_spec_path
        ),
        spec_path=(
            step27g_spec_path
        ),
    )

    df = load_step28a_table(
        table_path=(
            step28a_table_path
        ),
        summary_path=(
            step28a_summary_path
        ),
        field_mapping_path=(
            step28a_field_mapping_path
        ),
    )

    banner(
        "FROZEN TEST EXECUTION"
    )

    result_rows: List[Dict] = []

    for index, (
        axis,
        endpoint,
    ) in enumerate(
        FROZEN_TEST_ORDER
    ):

        # Different deterministic seed per frozen test while preserving
        # the frozen master seed.
        test_seed = (
            BOOTSTRAP_SEED
            + index
        )

        result = evaluate_one_test(
            df=df,
            axis=axis,
            endpoint=endpoint,
            bootstrap_seed=test_seed,
        )

        result_rows.append(
            result
        )

        print(
            f"{axis:<4} vs {endpoint:<22} "
            f"n={result['n_patients']:<4d} "
            f"rho={result['spearman_rho']:+.4f} "
            f"p={result['p_value']:.4g} "
            f"CI=[{result['bootstrap_ci_low']:+.4f}, "
            f"{result['bootstrap_ci_high']:+.4f}]"
        )

    results = pd.DataFrame(
        result_rows
    )

    # ------------------------------------------------------------------
    # BH-FDR within frozen families.
    # ------------------------------------------------------------------

    results[
        "q_value"
    ] = np.nan

    for family in [
        "PRIMARY_GRADE_PC3_PC6",
        "SUPPORTIVE_PT_PC3_PC6",
    ]:

        mask = (
            results[
                "multiplicity_family"
            ]
            == family
        )

        family_indices = results.index[
            mask
        ].tolist()

        if len(
            family_indices
        ) != 2:

            raise RuntimeError(
                f"Expected exactly 2 tests in family {family}."
            )

        p_values = [
            float(
                results.loc[
                    index,
                    "p_value",
                ]
            )
            for index
            in family_indices
        ]

        q_values = benjamini_hochberg(
            p_values
        )

        for index, q_value in zip(
            family_indices,
            q_values,
        ):

            results.loc[
                index,
                "q_value",
            ] = q_value

    if not np.isfinite(
        results[
            "q_value"
        ].to_numpy(
            dtype=np.float64
        )
    ).all():

        raise RuntimeError(
            "Non-finite q-values after BH-FDR."
        )

    # ------------------------------------------------------------------
    # Add frozen interpretation fields.
    # ------------------------------------------------------------------

    results[
        "significant_q_lt_0_05"
    ] = (
        results[
            "q_value"
        ]
        < 0.05
    )

    results[
        "ci_excludes_zero"
    ] = (
        (
            results[
                "bootstrap_ci_low"
            ]
            > 0
        )
        |
        (
            results[
                "bootstrap_ci_high"
            ]
            < 0
        )
    )

    # ------------------------------------------------------------------
    # Output paths.
    # ------------------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        out_dir
        / "tcga_prad_pc3_pc6_clinicopathologic_association_results.csv"
    )

    summary_path = (
        out_dir
        / "tcga_prad_pc3_pc6_clinicopathologic_association_summary.json"
    )

    claim_path = (
        out_dir
        / "tcga_prad_pc3_pc6_clinicopathologic_claim_boundary.txt"
    )

    hashes_path = (
        out_dir
        / "source_hashes.csv"
    )

    readme_path = (
        out_dir
        / "README.txt"
    )

    results.to_csv(
        result_path,
        index=False,
    )

    claim_path.write_text(
        build_claim_boundary(),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Source hashes.
    # ------------------------------------------------------------------

    source_files = [
        (
            "step28a_analysis_table",
            step28a_table_path,
        ),

        (
            "step28a_summary",
            step28a_summary_path,
        ),

        (
            "step28a_field_mapping",
            step28a_field_mapping_path,
        ),

        (
            "step27g_test_specification",
            step27g_test_spec_path,
        ),

        (
            "step27g_endpoint_specification",
            step27g_endpoint_spec_path,
        ),

        (
            "step27g_json_specification",
            step27g_spec_path,
        ),
    ]

    hash_rows = []

    for label, path in source_files:

        require_file(
            path,
            label,
        )

        hash_rows.append({
            "source": label,
            "path": str(
                path.resolve()
            ),
            "sha256": (
                sha256_file(
                    path
                )
            ),
        })

    pd.DataFrame(
        hash_rows
    ).to_csv(
        hashes_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary JSON.
    # ------------------------------------------------------------------

    summary = {
        "version": VERSION,
        "schema": SCHEMA,

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": "PASS",

        "analysis_role": (
            "PRESPECIFIED_TCGA_CLINICOPATHOLOGIC_INFERENCE_ONLY"
        ),

        "patients_total": int(
            len(
                df
            )
        ),

        "tests": (
            results
            .to_dict(
                orient="records"
            )
        ),

        "bootstrap": {
            "replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "master_seed": (
                BOOTSTRAP_SEED
            ),
            "confidence_interval": (
                "percentile_95_percent"
            ),
            "unit": (
                "patient"
            ),
        },

        "multiplicity": {
            "method": (
                "Benjamini-Hochberg FDR"
            ),
            "families": {
                "PRIMARY_GRADE_PC3_PC6": 2,
                "SUPPORTIVE_PT_PC3_PC6": 2,
            },
        },

        "firewall": {
            "pca_refit": False,
            "axis_reselection": False,
            "sign_tuning": False,
            "patient_aggregation_changed": False,
            "endpoint_mapping_changed": False,
            "pathologic_t_encoding_changed": False,
            "test_removed_after_result": False,
            "exploratory_endpoint_promoted": False,
            "rna_read": False,
        },

        "outputs": {
            "results": str(
                result_path.resolve()
            ),
            "claim_boundary": str(
                claim_path.resolve()
            ),
            "source_hashes": str(
                hashes_path.resolve()
            ),
        },
    }

    save_json_atomic(
        summary_path,
        summary,
    )

    # ------------------------------------------------------------------
    # README.
    # ------------------------------------------------------------------

    lines = [
        "STEP 28B",
        "TCGA-PRAD PC3 / PC6 CLINICOPATHOLOGIC ASSOCIATIONS",
        "",
        f"Version: {VERSION}",
        f"Schema: {SCHEMA}",
        "",
        "ROLE",
        "----",
        "Prespecified external clinicopathologic inference only.",
        "",
        "FROZEN TESTS",
        "------------",
        "PC3 vs Grade Group",
        "PC6 vs Grade Group",
        "PC3 vs pathologic T",
        "PC6 vs pathologic T",
        "",
        "STATISTICS",
        "----------",
        "Spearman rank correlation",
        "two-sided p-value",
        "10,000 patient bootstrap",
        "95% percentile CI",
        "BH-FDR within each two-test family",
        "",
        "RESULTS",
        "-------",
    ]

    for _, row in results.iterrows():

        lines.append(
            (
                f"{row['axis']} vs {row['endpoint']}: "
                f"n={int(row['n_patients'])}, "
                f"rho={row['spearman_rho']:+.4f}, "
                f"95% CI [{row['bootstrap_ci_low']:+.4f}, "
                f"{row['bootstrap_ci_high']:+.4f}], "
                f"p={row['p_value']:.6g}, "
                f"q={row['q_value']:.6g}"
            )
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "--------------",
        "Positive rho means higher oriented score is associated with",
        "greater clinicopathologic severity.",
        "",
        "Negative rho means higher oriented score is associated with",
        "lower clinicopathologic severity.",
        "",
        "No axis sign was modified using TCGA results.",
    ])

    readme_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------

    banner(
        "FINAL FROZEN RESULTS"
    )

    display_columns = [
        "axis",
        "endpoint",
        "analysis_role",
        "n_patients",
        "spearman_rho",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "p_value",
        "q_value",
        "significant_q_lt_0_05",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6g}"
            ),
        )
    )

    banner(
        "STEP 28B EXECUTION AUDIT"
    )

    audit = [
        (
            "Frozen Step27G test set inherited",
            "YES",
        ),

        (
            "Step28A patient table inherited",
            "YES",
        ),

        (
            "Patient inferential unit used",
            "YES",
        ),

        (
            "Four prespecified tests executed",
            "YES",
        ),

        (
            "Spearman statistic used",
            "YES",
        ),

        (
            "Two-sided p-values used",
            "YES",
        ),

        (
            "10,000 patient bootstrap used",
            "YES",
        ),

        (
            "BH-FDR within frozen families",
            "YES",
        ),

        (
            "PCA refit",
            "NO",
        ),

        (
            "Axis reselection",
            "NO",
        ),

        (
            "PC3 sign tuning",
            "NO",
        ),

        (
            "PC6 sign tuning",
            "NO",
        ),

        (
            "Patient aggregation changed",
            "NO",
        ),

        (
            "Endpoint encoding changed",
            "NO",
        ),

        (
            "Nonsignificant frozen test removed",
            "NO",
        ),

        (
            "RNA read",
            "NO",
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<58}: {value}"
        )

    banner(
        "STEP 28B: PASS"
    )

    print(
        f"[PATIENTS TOTAL]    {len(df)}"
    )

    print(
        "[TESTS]             4"
    )

    print(
        f"[BOOTSTRAP]         {BOOTSTRAP_REPLICATES:,}"
    )

    print(
        "[MULTIPLICITY]      BH-FDR within frozen families"
    )

    print()

    for _, row in results.iterrows():

        print(
            f"[{row['axis']} vs {row['endpoint']}] "
            f"n={int(row['n_patients'])} | "
            f"rho={row['spearman_rho']:+.4f} | "
            f"CI=[{row['bootstrap_ci_low']:+.4f}, "
            f"{row['bootstrap_ci_high']:+.4f}] | "
            f"p={row['p_value']:.4g} | "
            f"q={row['q_value']:.4g}"
        )

    print()

    print(
        f"[RESULTS]           {result_path}"
    )

    print(
        f"[SUMMARY]           {summary_path}"
    )

    print(
        f"[CLAIM BOUNDARY]    {claim_path}"
    )

    print(
        f"[HASHES]            {hashes_path}"
    )

    print(
        f"[README]            {readme_path}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen TCGA-PRAD PC3 / PC6 "
            "clinicopathologic associations."
        ),

        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--step28a_table",
        type=Path,
        default=DEFAULT_STEP28A_TABLE,
    )

    parser.add_argument(
        "--step28a_summary",
        type=Path,
        default=DEFAULT_STEP28A_SUMMARY,
    )

    parser.add_argument(
        "--step28a_field_mapping",
        type=Path,
        default=DEFAULT_STEP28A_FIELD_MAPPING,
    )

    parser.add_argument(
        "--step27g_test_spec",
        type=Path,
        default=DEFAULT_STEP27G_TEST_SPEC,
    )

    parser.add_argument(
        "--step27g_endpoint_spec",
        type=Path,
        default=DEFAULT_STEP27G_ENDPOINT_SPEC,
    )

    parser.add_argument(
        "--step27g_spec",
        type=Path,
        default=DEFAULT_STEP27G_SPEC,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    execute(
        step28a_table_path=(
            args.step28a_table.resolve()
        ),

        step28a_summary_path=(
            args.step28a_summary.resolve()
        ),

        step28a_field_mapping_path=(
            args.step28a_field_mapping.resolve()
        ),

        step27g_test_spec_path=(
            args.step27g_test_spec.resolve()
        ),

        step27g_endpoint_spec_path=(
            args.step27g_endpoint_spec.resolve()
        ),

        step27g_spec_path=(
            args.step27g_spec.resolve()
        ),

        out_dir=(
            args.out_dir.resolve()
        ),
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        banner(
            "STEP 28B: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)