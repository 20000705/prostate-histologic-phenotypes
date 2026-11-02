#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27g_freeze_tcga_prad_clinicopathologic_evaluation_specification.py

STEP 27G
========
FREEZE TCGA-PRAD CLINICOPATHOLOGIC EVALUATION SPECIFICATION

VERSION
-------
1.0.0

SCHEMA
------
27g-tcga-prad-clinicopathologic-evaluation-specification-schema-1.0

ROLE
----
PRE-EXECUTION CLINICOPATHOLOGIC ANALYSIS FIREWALL ONLY.

This script freezes the statistical design for evaluating whether the
already-established, discovery-frozen PC3 and PC6 histologic axes are
associated with clinicopathologic severity in TCGA-PRAD.

This script DOES NOT:

    - read TCGA Grade Group values,
    - read Gleason scores,
    - read pathologic stage,
    - read PSA,
    - read age,
    - read survival outcomes,
    - read RNA,
    - calculate any biological association,
    - select clinical endpoints based on results,
    - select covariates based on results,
    - refit PCA,
    - reorient PC3 or PC6,
    - reselect latent axes.

REPRESENTATION AUTHORITY
------------------------
Step27D is the only TCGA PC3 / PC6 score authority.

For each slide:

    oriented_PC3
    oriented_PC6

are already frozen.

Patient-level representation is defined BEFORE clinical outcome inspection:

    slide_score_PC3 =
        median(tile-level oriented_PC3 within slide)

    slide_score_PC6 =
        median(tile-level oriented_PC6 within slide)

    patient_score_PC3 =
        median(slide_score_PC3 across frozen slides from the same patient)

    patient_score_PC6 =
        median(slide_score_PC6 across frozen slides from the same patient)

Thus:

    tile -> slide median -> patient median

No alternative aggregation rule may be selected based on clinical results.

PATIENT IDENTIFIER
------------------
TCGA patient identifier is deterministically derived from slide_id using the
first three TCGA barcode fields:

    TCGA-XX-YYYY

equivalent to the first 12 barcode characters for standard TCGA barcodes.

CLINICOPATHOLOGIC ENDPOINT HIERARCHY
------------------------------------

PRIMARY ENDPOINT FAMILY
~~~~~~~~~~~~~~~~~~~~~~~
Tumor grade severity.

Preferred representation:
    ISUP / Grade Group, ordinal 1-5, if directly available or deterministically
    derivable from authoritative Gleason primary + secondary pattern fields.

Primary inferential test:
    Spearman rank correlation between patient-level oriented score and
    ordinal Grade Group.

PC3:
    PRIMARY confirmatory clinicopathologic axis.

PC6:
    SECONDARY prespecified clinicopathologic axis.

The same endpoint definition MUST be used for PC3 and PC6.

SUPPORTIVE ENDPOINT FAMILY
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pathologic T stage.

The stage representation must be reduced prospectively to an ordered
pathologic-T severity variable using source semantics only.

Examples of permissible deterministic ordering:

    pT2 < pT3 < pT4

Subcategories such as pT2a/pT2b/pT2c and pT3a/pT3b may be preserved only if
the authoritative source supports a deterministic ordinal ordering fixed
before any PC association result is inspected.

Primary supportive inferential test:
    Spearman rank correlation.

OPTIONAL DESCRIPTIVE ENDPOINTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
May be summarized but are NOT promoted to confirmatory evidence unless
separately frozen before testing:

    - pathologic N stage,
    - AJCC stage,
    - PSA,
    - age,
    - margin status,
    - recurrence,
    - survival.

These are outside the primary Step27G claim.

MULTIPLICITY
------------
Primary grade family:

    PC3 vs Grade Group
    PC6 vs Grade Group

These two tests form one prespecified family.

Benjamini-Hochberg FDR correction is applied across the two grade-axis tests.

Supportive pathologic-T family:

    PC3 vs pathologic T
    PC6 vs pathologic T

These form a separate supportive family and receive BH-FDR within that family.

No pooling of unrelated exploratory clinical endpoints into the confirmatory
families is permitted.

EFFECT SIZE REPORTING
---------------------
For every inferential endpoint report:

    n patients
    Spearman rho
    nominal p-value
    within-family BH q-value
    bootstrap 95% CI for Spearman rho

Bootstrap unit:
    patient

Bootstrap replicates:
    10,000

Bootstrap seed:
    2026

No tile-level or slide-level observations may be treated as independent
clinical replicates.

CLAIM CEILING
-------------
A statistically significant association may support statements such as:

    "The frozen histologic axis was associated with clinicopathologic
     severity in TCGA-PRAD."

It does NOT establish:

    - diagnostic performance,
    - tumor localization performance,
    - causal biology,
    - molecular mechanism,
    - prognostic utility,
    - clinical utility,
    - independence from standard pathology,
    - improved prediction beyond Grade Group or stage.

Those require separate analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "27g-tcga-prad-clinicopathologic-"
    "evaluation-specification-schema-1.0"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


DEFAULT_STEP27C_AXIS_SPEC = (
    RESULTS
    / "27c_tcga_prad_pc3_pc6_external_evaluation_specification"
    / "tcga_prad_pc3_pc6_frozen_axis_specification.csv"
)

DEFAULT_STEP27C_METADATA = (
    RESULTS
    / "27c_tcga_prad_pc3_pc6_external_evaluation_specification"
    / "tcga_prad_pc3_pc6_external_evaluation_specification.json"
)


DEFAULT_STEP27D_MANIFEST = (
    RESULTS
    / "27d_tcga_prad_frozen_pc3_pc6_scores"
    / "frozen_tcga_prad_pc3_pc6_score_manifest.csv"
)

DEFAULT_STEP27D_SLIDE_SUMMARY = (
    RESULTS
    / "27d_tcga_prad_frozen_pc3_pc6_scores"
    / "frozen_tcga_prad_pc3_pc6_slide_summary.csv"
)

DEFAULT_STEP27D_SPEC = (
    RESULTS
    / "27d_tcga_prad_frozen_pc3_pc6_scores"
    / "tcga_prad_pc3_pc6_score_materialization_specification.json"
)

DEFAULT_STEP27D_SUMMARY = (
    RESULTS
    / "27d_tcga_prad_frozen_pc3_pc6_scores"
    / "tcga_prad_pc3_pc6_score_materialization_summary.json"
)


DEFAULT_OUT_DIR = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
)


# =============================================================================
# EXPECTATIONS
# =============================================================================

EXPECTED_SLIDES = 403

TARGET_AXES = [
    "PC3",
    "PC6",
]

EXPECTED_AXIS_ROLES = {
    "PC3": "PRIMARY_INTERPRETED_AXIS",
    "PC6": "SECONDARY_INTERPRETED_AXIS",
}

SLIDE_SCORE_COLUMNS = {
    "PC3": "oriented_PC3_median",
    "PC6": "oriented_PC6_median",
}

PATIENT_AGGREGATION = "median"

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026

MULTIPLE_TESTING_METHOD = "Benjamini-Hochberg FDR"


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
# TCGA PATIENT IDENTIFIER
# =============================================================================

TCGA_SLIDE_RE = re.compile(
    r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})-"
)


def patient_id_from_slide_id(
    slide_id: str,
) -> str:

    match = TCGA_SLIDE_RE.match(
        str(
            slide_id
        )
    )

    if match is None:

        raise RuntimeError(
            f"Unable to derive TCGA patient ID from "
            f"slide_id={slide_id!r}"
        )

    return match.group(
        1
    )


# =============================================================================
# STEP27C AUTHORITY
# =============================================================================

def validate_step27c(
    axis_spec_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:

    require_file(
        axis_spec_path,
        "Step27C axis specification",
    )

    require_file(
        metadata_path,
        "Step27C metadata",
    )

    axis = pd.read_csv(
        axis_spec_path
    )

    metadata = load_json(
        metadata_path
    )

    if (
        metadata.get(
            "analysis_role"
        )
        != "PRE_EXECUTION_TCGA_EXTERNAL_EVALUATION_FREEZE_ONLY"
    ):

        raise RuntimeError(
            "Unexpected Step27C analysis role."
        )

    required_columns = {
        "axis",
        "axis_role",
        "frozen_direction",
        "score_multiplier",
    }

    missing = sorted(
        required_columns
        - set(
            axis.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step27C axis specification schema changed. "
            f"Missing columns: {missing}"
        )

    selected = (
        axis.loc[
            axis[
                "axis"
            ].astype(str)
            .isin(
                TARGET_AXES
            )
        ]
        .copy()
    )

    if len(
        selected
    ) != 2:

        raise RuntimeError(
            "Step27C does not contain exactly PC3 and PC6."
        )

    for axis_name in TARGET_AXES:

        row = selected.loc[
            selected[
                "axis"
            ].astype(str)
            == axis_name
        ]

        if len(
            row
        ) != 1:

            raise RuntimeError(
                f"Unable to resolve unique {axis_name} row."
            )

        role = str(
            row.iloc[0][
                "axis_role"
            ]
        )

        if (
            role
            != EXPECTED_AXIS_ROLES[
                axis_name
            ]
        ):

            raise RuntimeError(
                f"{axis_name}: unexpected axis role "
                f"{role!r}"
            )

    return (
        selected
        .sort_values(
            "axis"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# STEP27D AUTHORITY
# =============================================================================

def load_step27d(
    manifest_path: Path,
    slide_summary_path: Path,
    spec_path: Path,
    summary_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    require_file(
        manifest_path,
        "Step27D manifest",
    )

    require_file(
        slide_summary_path,
        "Step27D slide summary",
    )

    require_file(
        spec_path,
        "Step27D specification",
    )

    require_file(
        summary_path,
        "Step27D execution summary",
    )

    manifest = pd.read_csv(
        manifest_path
    )

    slide_summary = pd.read_csv(
        slide_summary_path
    )

    specification = load_json(
        spec_path
    )

    execution_summary = load_json(
        summary_path
    )

    # ------------------------------------------------------------------
    # Manifest checks
    # ------------------------------------------------------------------

    required_manifest_columns = {
        "slide_id",
        "n_tiles",
        "pc3_pc6_scores_path",
    }

    missing = sorted(
        required_manifest_columns
        - set(
            manifest.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step27D manifest schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        manifest
    ) != EXPECTED_SLIDES:

        raise RuntimeError(
            "Step27D slide count changed: "
            f"{len(manifest)} != {EXPECTED_SLIDES}"
        )

    if (
        manifest[
            "slide_id"
        ]
        .astype(str)
        .nunique()
        != EXPECTED_SLIDES
    ):

        raise RuntimeError(
            "Step27D does not contain 403 unique slides."
        )

    # ------------------------------------------------------------------
    # Slide-summary checks
    # ------------------------------------------------------------------

    required_summary_columns = {
        "slide_id",
        "n_tiles",
        "oriented_PC3_median",
        "oriented_PC6_median",
    }

    missing = sorted(
        required_summary_columns
        - set(
            slide_summary.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step27D slide-summary schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        slide_summary
    ) != EXPECTED_SLIDES:

        raise RuntimeError(
            "Step27D slide-summary row count changed."
        )

    if (
        slide_summary[
            "slide_id"
        ]
        .astype(str)
        .nunique()
        != EXPECTED_SLIDES
    ):

        raise RuntimeError(
            "Step27D slide summary does not contain "
            "403 unique slides."
        )

    if set(
        manifest[
            "slide_id"
        ].astype(str)
    ) != set(
        slide_summary[
            "slide_id"
        ].astype(str)
    ):

        raise RuntimeError(
            "Step27D manifest and slide summary slide sets differ."
        )

    # ------------------------------------------------------------------
    # Specification / execution checks
    # ------------------------------------------------------------------

    if (
        specification.get(
            "analysis_role"
        )
        != "FROZEN_TCGA_PC3_PC6_SCORE_MATERIALIZATION_ONLY"
    ):

        raise RuntimeError(
            "Unexpected Step27D analysis role."
        )

    if (
        str(
            execution_summary.get(
                "status",
                "",
            )
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Step27D execution status is not PASS."
        )

    if int(
        execution_summary.get(
            "unique_slides",
            -1,
        )
    ) != EXPECTED_SLIDES:

        raise RuntimeError(
            "Step27D execution summary unique-slide "
            "count is not 403."
        )

    return (
        manifest,
        slide_summary,
        specification,
    )


# =============================================================================
# REPRESENTATION COHORT
# =============================================================================

def build_patient_representation_manifest(
    slide_summary: pd.DataFrame,
) -> pd.DataFrame:

    df = slide_summary[
        [
            "slide_id",
            "n_tiles",
            "oriented_PC3_median",
            "oriented_PC6_median",
        ]
    ].copy()

    df[
        "slide_id"
    ] = df[
        "slide_id"
    ].astype(str)

    df[
        "patient_id"
    ] = df[
        "slide_id"
    ].map(
        patient_id_from_slide_id
    )

    for column in [
        "oriented_PC3_median",
        "oriented_PC6_median",
    ]:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="raise",
        )

        if not np.isfinite(
            df[
                column
            ].to_numpy()
        ).all():

            raise RuntimeError(
                f"{column} contains NaN/Inf."
            )

    grouped = (
        df
        .groupby(
            "patient_id",
            sort=True,
        )
    )

    patient = (
        grouped
        .agg(
            n_frozen_slides=(
                "slide_id",
                "size",
            ),
            total_tiles=(
                "n_tiles",
                "sum",
            ),
            patient_oriented_PC3=(
                "oriented_PC3_median",
                "median",
            ),
            patient_oriented_PC6=(
                "oriented_PC6_median",
                "median",
            ),
        )
        .reset_index()
    )

    patient[
        "patient_aggregation_rule"
    ] = (
        "median_of_slide_level_oriented_medians"
    )

    patient[
        "clinical_outcome_attached"
    ] = False

    return patient


# =============================================================================
# ENDPOINT SPECIFICATION
# =============================================================================

def build_endpoint_specification() -> pd.DataFrame:

    rows = [
        {
            "family": "PRIMARY_GRADE",
            "endpoint": "GRADE_GROUP",
            "status": "CONFIRMATORY",
            "clinical_unit": "patient",
            "endpoint_type": "ordinal",
            "preferred_encoding": (
                "ISUP/Grade Group 1-5"
            ),
            "source_mapping_rule": (
                "Use authoritative TCGA clinical/pathology field if "
                "directly available; otherwise deterministic derivation "
                "from authoritative Gleason primary and secondary patterns."
            ),
            "primary_test": (
                "Spearman rank correlation"
            ),
            "bootstrap_ci": True,
            "bootstrap_unit": "patient",
            "multiplicity_family": (
                "PRIMARY_GRADE_PC3_PC6"
            ),
            "allowed_claim_level": (
                "formal clinicopathologic association"
            ),
        },
        {
            "family": "SUPPORTIVE_PATHOLOGIC_T",
            "endpoint": "PATHOLOGIC_T_STAGE",
            "status": "SUPPORTIVE",
            "clinical_unit": "patient",
            "endpoint_type": "ordinal",
            "preferred_encoding": (
                "ordered pathologic T severity"
            ),
            "source_mapping_rule": (
                "Deterministically map authoritative pathologic T labels "
                "to an ordinal severity scale using clinical semantics only."
            ),
            "primary_test": (
                "Spearman rank correlation"
            ),
            "bootstrap_ci": True,
            "bootstrap_unit": "patient",
            "multiplicity_family": (
                "SUPPORTIVE_PT_PC3_PC6"
            ),
            "allowed_claim_level": (
                "supportive clinicopathologic association"
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def build_axis_test_specification(
    axis_spec: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for axis_name in TARGET_AXES:

        authority = axis_spec.loc[
            axis_spec[
                "axis"
            ].astype(str)
            == axis_name
        ].iloc[0]

        role = str(
            authority[
                "axis_role"
            ]
        )

        for family, endpoint, status in [
            (
                "PRIMARY_GRADE",
                "GRADE_GROUP",
                (
                    "PRIMARY_CONFIRMATORY"
                    if axis_name == "PC3"
                    else "SECONDARY_PRESPECIFIED"
                ),
            ),
            (
                "SUPPORTIVE_PATHOLOGIC_T",
                "PATHOLOGIC_T_STAGE",
                "SUPPORTIVE",
            ),
        ]:

            rows.append({
                "axis": (
                    axis_name
                ),
                "axis_role": (
                    role
                ),
                "frozen_direction": str(
                    authority[
                        "frozen_direction"
                    ]
                ),
                "score_multiplier": int(
                    float(
                        authority[
                            "score_multiplier"
                        ]
                    )
                ),
                "patient_score": (
                    f"patient_oriented_{axis_name}"
                ),
                "endpoint_family": (
                    family
                ),
                "endpoint": (
                    endpoint
                ),
                "test_status": (
                    status
                ),
                "test": (
                    "Spearman rank correlation"
                ),
                "alternative": (
                    "two-sided"
                ),
                "bootstrap_replicates": (
                    BOOTSTRAP_REPLICATES
                ),
                "bootstrap_seed": (
                    BOOTSTRAP_SEED
                ),
            })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CLAIM CEILING
# =============================================================================

def build_claim_ceiling() -> str:

    return """STEP 27G — TCGA-PRAD CLINICOPATHOLOGIC CLAIM CEILING

ROLE
----
PRE-EXECUTION CLINICOPATHOLOGIC ANALYSIS FREEZE.

NO CLINICAL OUTCOME VALUES ARE READ IN STEP27G.


REPRESENTATION
--------------
PC3:
    PRIMARY_INTERPRETED_AXIS

PC6:
    SECONDARY_INTERPRETED_AXIS

Both use Step27D oriented scores.

Patient-level representation:

    tile-level oriented score
        -> slide median
        -> patient median


PRIMARY CLINICOPATHOLOGIC FAMILY
--------------------------------
Grade Group / ordinal Gleason severity.

Formal tests:

    PC3 vs Grade Group
    PC6 vs Grade Group

Statistic:
    Spearman rho

Multiplicity:
    BH-FDR across these two tests.

Bootstrap:
    patient-level
    10,000 replicates
    seed 2026


SUPPORTIVE FAMILY
-----------------
Pathologic T stage.

Formal supportive tests:

    PC3 vs pathologic T
    PC6 vs pathologic T

Statistic:
    Spearman rho

Multiplicity:
    BH-FDR across these two tests separately from the grade family.


ALLOWED CLAIM IF SUPPORTED
--------------------------
"The frozen histologic axis was associated with clinicopathologic
severity in the independent TCGA-PRAD cohort."


NOT ESTABLISHED
---------------
1. Tumor localization.
2. Diagnostic accuracy.
3. Grade prediction accuracy.
4. Clinical utility.
5. Prognostic utility.
6. Added value beyond standard pathology.
7. Molecular mechanism.
8. Causal biology.
9. Independence from Grade Group or stage.
10. Treatment-response relevance.


PROHIBITED
----------
1. Refit PCA.
2. Reorient PC3 or PC6 using TCGA.
3. Replace PC3 or PC6 with another component.
4. Select patient aggregation after seeing clinical results.
5. Select clinical endpoint according to significance.
6. Select stage encoding according to significance.
7. Treat slides as independent clinical replicates when patients have
   multiple slides.
8. Treat tiles as independent clinical replicates.
9. Promote exploratory clinical outcomes into the frozen primary family.
"""


# =============================================================================
# README
# =============================================================================

def build_readme(
    n_slides: int,
    n_patients: int,
) -> str:

    return f"""STEP 27G
TCGA-PRAD CLINICOPATHOLOGIC EVALUATION FREEZE

Version:
    {VERSION}

Schema:
    {SCHEMA}


ROLE
----
PRE-EXECUTION FREEZE ONLY.

Clinical outcome values read:
    NO


REPRESENTATION COHORT
---------------------
Frozen slides:
    {n_slides}

Unique TCGA patients derived from frozen slide barcodes:
    {n_patients}


PATIENT SCORE DEFINITION
------------------------
PC3:

    tile oriented_PC3
        -> slide median
        -> median across frozen slides for patient

PC6:

    tile oriented_PC6
        -> slide median
        -> median across frozen slides for patient


PRIMARY ENDPOINT
----------------
Grade Group / ordinal Gleason severity.

Tests:

    PC3 vs Grade Group
    PC6 vs Grade Group

Statistic:
    Spearman rank correlation

Multiplicity:
    BH-FDR across the two axis tests


SUPPORTIVE ENDPOINT
-------------------
Pathologic T stage.

Tests:

    PC3 vs pathologic T
    PC6 vs pathologic T

Statistic:
    Spearman rank correlation

Multiplicity:
    separate BH-FDR family


UNCERTAINTY
-----------
Bootstrap unit:
    patient

Replicates:
    {BOOTSTRAP_REPLICATES}

Seed:
    {BOOTSTRAP_SEED}


NEXT STEP
---------
Build the clinicopathologic analysis table by joining authoritative TCGA
clinical/pathology metadata to the frozen patient representation manifest.

Endpoint mapping must follow semantic/provenance rules only.

No endpoint or encoding may be chosen based on PC3/PC6 association results.
"""


# =============================================================================
# EXECUTION
# =============================================================================

def execute(
    step27c_axis_spec_path: Path,
    step27c_metadata_path: Path,
    step27d_manifest_path: Path,
    step27d_slide_summary_path: Path,
    step27d_spec_path: Path,
    step27d_summary_path: Path,
    out_dir: Path,
) -> None:

    banner(
        "STEP 27G: FREEZE TCGA-PRAD "
        "CLINICOPATHOLOGIC EVALUATION SPECIFICATION"
    )

    print(
        f"[VERSION]       {VERSION}"
    )

    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    # ------------------------------------------------------------------
    # Validate upstream representation authorities.
    # ------------------------------------------------------------------

    axis_spec = validate_step27c(
        axis_spec_path=(
            step27c_axis_spec_path
        ),
        metadata_path=(
            step27c_metadata_path
        ),
    )

    (
        manifest27d,
        slide_summary27d,
        step27d_spec,
    ) = load_step27d(
        manifest_path=(
            step27d_manifest_path
        ),
        slide_summary_path=(
            step27d_slide_summary_path
        ),
        spec_path=(
            step27d_spec_path
        ),
        summary_path=(
            step27d_summary_path
        ),
    )

    # ------------------------------------------------------------------
    # Build representation-only patient manifest.
    # ------------------------------------------------------------------

    patient_manifest = (
        build_patient_representation_manifest(
            slide_summary27d
        )
    )

    n_patients = int(
        patient_manifest[
            "patient_id"
        ].nunique()
    )

    banner(
        "FROZEN REPRESENTATION COHORT"
    )

    print(
        f"Frozen slides    : {len(manifest27d)}"
    )

    print(
        f"Unique patients  : {n_patients}"
    )

    print(
        "Patient score PC3: median of slide-level "
        "oriented_PC3 medians"
    )

    print(
        "Patient score PC6: median of slide-level "
        "oriented_PC6 medians"
    )

    print(
        "Clinical outcomes read: NO"
    )

    # ------------------------------------------------------------------
    # Frozen endpoint / test design.
    # ------------------------------------------------------------------

    endpoint_spec = (
        build_endpoint_specification()
    )

    axis_test_spec = (
        build_axis_test_specification(
            axis_spec
        )
    )

    banner(
        "FROZEN CLINICOPATHOLOGIC DESIGN"
    )

    print(
        "Primary endpoint family    : Grade Group"
    )

    print(
        "Supportive endpoint family : Pathologic T stage"
    )

    print(
        "Primary statistic          : Spearman rho"
    )

    print(
        f"Bootstrap                  : "
        f"{BOOTSTRAP_REPLICATES:,} patient resamples"
    )

    print(
        f"Multiplicity               : "
        f"{MULTIPLE_TESTING_METHOD}"
    )

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_manifest_path = (
        out_dir
        / "frozen_tcga_prad_patient_representation_manifest.csv"
    )

    endpoint_spec_path = (
        out_dir
        / "tcga_prad_clinicopathologic_endpoint_specification.csv"
    )

    axis_test_spec_path = (
        out_dir
        / "tcga_prad_pc3_pc6_clinicopathologic_test_specification.csv"
    )

    specification_path = (
        out_dir
        / "tcga_prad_clinicopathologic_evaluation_specification.json"
    )

    claim_path = (
        out_dir
        / "tcga_prad_clinicopathologic_claim_ceiling.txt"
    )

    hashes_path = (
        out_dir
        / "source_hashes.csv"
    )

    readme_path = (
        out_dir
        / "README.txt"
    )

    patient_manifest.to_csv(
        patient_manifest_path,
        index=False,
    )

    endpoint_spec.to_csv(
        endpoint_spec_path,
        index=False,
    )

    axis_test_spec.to_csv(
        axis_test_spec_path,
        index=False,
    )

    claim_path.write_text(
        build_claim_ceiling(),
        encoding="utf-8",
    )

    readme_path.write_text(
        build_readme(
            n_slides=len(
                manifest27d
            ),
            n_patients=(
                n_patients
            ),
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Source hashes.
    # ------------------------------------------------------------------

    source_files = [
        (
            "step27c_axis_specification",
            step27c_axis_spec_path,
        ),
        (
            "step27c_metadata",
            step27c_metadata_path,
        ),
        (
            "step27d_manifest",
            step27d_manifest_path,
        ),
        (
            "step27d_slide_summary",
            step27d_slide_summary_path,
        ),
        (
            "step27d_specification",
            step27d_spec_path,
        ),
        (
            "step27d_summary",
            step27d_summary_path,
        ),
    ]

    hash_rows: List[dict] = []

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
    # JSON specification.
    # ------------------------------------------------------------------

    specification = {
        "version": VERSION,
        "schema": SCHEMA,

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "analysis_role": (
            "PRE_EXECUTION_TCGA_CLINICOPATHOLOGIC_EVALUATION_FREEZE"
        ),

        "freeze_only_audit": {
            "clinical_outcomes_read": False,
            "grade_group_read": False,
            "gleason_read": False,
            "pathologic_stage_read": False,
            "survival_read": False,
            "rna_read": False,
            "biological_association_calculated": False,
            "endpoint_selected_from_results": False,
            "covariate_selected_from_results": False,
            "axis_reselection": False,
            "sign_tuning": False,
            "pca_refit": False,
        },

        "representation_authority": {
            "source": "Step27D",
            "slides": int(
                len(
                    manifest27d
                )
            ),
            "patients": (
                n_patients
            ),

            "slide_scores": {
                "PC3": (
                    "oriented_PC3_median"
                ),
                "PC6": (
                    "oriented_PC6_median"
                ),
            },

            "patient_scores": {
                "PC3": (
                    "median of slide-level "
                    "oriented_PC3_median"
                ),
                "PC6": (
                    "median of slide-level "
                    "oriented_PC6_median"
                ),
            },

            "patient_identifier": (
                "TCGA participant barcode TCGA-XX-YYYY "
                "derived deterministically from slide_id"
            ),

            "alternative_patient_aggregation_allowed": False,
        },

        "axis_hierarchy": {
            "PC3": {
                "role": (
                    "PRIMARY_INTERPRETED_AXIS"
                ),
                "clinicopathologic_status": (
                    "PRIMARY_CONFIRMATORY"
                ),
            },

            "PC6": {
                "role": (
                    "SECONDARY_INTERPRETED_AXIS"
                ),
                "clinicopathologic_status": (
                    "SECONDARY_PRESPECIFIED"
                ),
            },
        },

        "primary_endpoint_family": {
            "name": "GRADE_GROUP",
            "type": "ordinal",
            "preferred_encoding": (
                "ISUP / Grade Group 1-5"
            ),

            "mapping_rule": (
                "Direct authoritative Grade Group if available; "
                "otherwise deterministic derivation from "
                "authoritative Gleason primary/secondary patterns."
            ),

            "tests": [
                "PC3 vs Grade Group",
                "PC6 vs Grade Group",
            ],

            "statistic": (
                "Spearman rank correlation"
            ),

            "alternative": (
                "two-sided"
            ),

            "multiplicity": {
                "method": (
                    MULTIPLE_TESTING_METHOD
                ),
                "family_size": 2,
            },
        },

        "supportive_endpoint_family": {
            "name": (
                "PATHOLOGIC_T_STAGE"
            ),

            "type": "ordinal",

            "mapping_rule": (
                "Deterministic clinically ordered pathologic-T "
                "severity encoding fixed from source semantics."
            ),

            "tests": [
                "PC3 vs pathologic T stage",
                "PC6 vs pathologic T stage",
            ],

            "statistic": (
                "Spearman rank correlation"
            ),

            "alternative": (
                "two-sided"
            ),

            "multiplicity": {
                "method": (
                    MULTIPLE_TESTING_METHOD
                ),
                "family_size": 2,
            },
        },

        "uncertainty": {
            "bootstrap_unit": (
                "patient"
            ),
            "replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "seed": (
                BOOTSTRAP_SEED
            ),
            "confidence_interval": (
                "95%"
            ),
        },

        "inferential_unit": (
            "patient"
        ),

        "forbidden_inferential_units": [
            "tile",
            "slide when multiple slides belong to same patient",
        ],

        "optional_descriptive_endpoints_not_promotable_without_new_freeze": [
            "pathologic N stage",
            "AJCC stage",
            "PSA",
            "age",
            "margin status",
            "biochemical recurrence",
            "progression-free interval",
            "overall survival",
        ],

        "claim_boundary": {
            "clinicopathologic_association_claim_allowed_if_supported": True,
            "tumor_localization_claim_allowed": False,
            "diagnostic_accuracy_claim_allowed": False,
            "grade_prediction_claim_allowed": False,
            "clinical_utility_claim_allowed": False,
            "prognostic_claim_allowed": False,
            "molecular_mechanism_claim_allowed": False,
            "causal_claim_allowed": False,
            "added_value_beyond_standard_pathology_claim_allowed": False,
        },

        "next_step_rule": (
            "Build the authoritative TCGA clinicopathologic analysis "
            "table by mapping frozen patient IDs to clinical/pathology "
            "metadata using semantic/provenance rules only. "
            "No PC association result may influence source-field mapping."
        ),

        "outputs": {
            "patient_representation_manifest": str(
                patient_manifest_path.resolve()
            ),
            "endpoint_specification": str(
                endpoint_spec_path.resolve()
            ),
            "axis_test_specification": str(
                axis_test_spec_path.resolve()
            ),
            "claim_ceiling": str(
                claim_path.resolve()
            ),
            "source_hashes": str(
                hashes_path.resolve()
            ),
            "readme": str(
                readme_path.resolve()
            ),
        },
    }

    save_json_atomic(
        specification_path,
        specification,
    )

    # ------------------------------------------------------------------
    # Audit.
    # ------------------------------------------------------------------

    banner(
        "STEP 27G FREEZE-ONLY AUDIT"
    )

    audit = [
        (
            "403 frozen slides inherited",
            "YES",
        ),
        (
            "Patient IDs derived deterministically",
            "YES",
        ),
        (
            "Patient-level PC3 aggregation frozen",
            "YES",
        ),
        (
            "Patient-level PC6 aggregation frozen",
            "YES",
        ),
        (
            "PC3 primary clinicopathologic role frozen",
            "YES",
        ),
        (
            "PC6 secondary clinicopathologic role frozen",
            "YES",
        ),
        (
            "Grade Group primary endpoint frozen",
            "YES",
        ),
        (
            "Pathologic T supportive endpoint frozen",
            "YES",
        ),
        (
            "Spearman statistic frozen",
            "YES",
        ),
        (
            "Patient bootstrap frozen",
            "YES",
        ),
        (
            "Multiplicity families frozen",
            "YES",
        ),
        (
            "Clinical outcomes read",
            "NO",
        ),
        (
            "Grade Group values read",
            "NO",
        ),
        (
            "Pathologic stage values read",
            "NO",
        ),
        (
            "Survival outcomes read",
            "NO",
        ),
        (
            "RNA outcomes read",
            "NO",
        ),
        (
            "Clinical association calculated",
            "NO",
        ),
        (
            "Endpoint selected from results",
            "NO",
        ),
        (
            "PCA refit",
            "NO",
        ),
        (
            "PC3/PC6 sign tuning",
            "NO",
        ),
        (
            "Claim ceiling frozen",
            "YES",
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<58}: {value}"
        )

    banner(
        "STEP 27G: PASS"
    )

    print(
        f"[SLIDES]            {len(manifest27d)}"
    )

    print(
        f"[PATIENTS]          {n_patients}"
    )

    print(
        "[PC3]               PRIMARY clinicopathologic axis"
    )

    print(
        "[PC6]               SECONDARY prespecified axis"
    )

    print(
        "[PRIMARY ENDPOINT]  Grade Group"
    )

    print(
        "[SUPPORTIVE]        Pathologic T stage"
    )

    print(
        "[STATISTIC]         Spearman rho"
    )

    print(
        f"[BOOTSTRAP]         {BOOTSTRAP_REPLICATES:,} patient resamples"
    )

    print(
        "[MULTIPLICITY]      BH-FDR within prespecified family"
    )

    print()

    print(
        f"[PATIENT MANIFEST]  {patient_manifest_path}"
    )

    print(
        f"[ENDPOINT SPEC]     {endpoint_spec_path}"
    )

    print(
        f"[TEST SPEC]         {axis_test_spec_path}"
    )

    print(
        f"[SPEC]              {specification_path}"
    )

    print(
        f"[CLAIM CEILING]     {claim_path}"
    )

    print(
        f"[HASHES]            {hashes_path}"
    )

    print(
        f"[README]            {readme_path}"
    )

    print()

    print(
        "[NEXT] Build the authoritative TCGA clinical/pathology "
        "analysis table without calculating PC associations."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the TCGA-PRAD PC3 / PC6 "
            "clinicopathologic evaluation design."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--step27c_axis_spec",
        type=Path,
        default=DEFAULT_STEP27C_AXIS_SPEC,
    )

    parser.add_argument(
        "--step27c_metadata",
        type=Path,
        default=DEFAULT_STEP27C_METADATA,
    )

    parser.add_argument(
        "--step27d_manifest",
        type=Path,
        default=DEFAULT_STEP27D_MANIFEST,
    )

    parser.add_argument(
        "--step27d_slide_summary",
        type=Path,
        default=DEFAULT_STEP27D_SLIDE_SUMMARY,
    )

    parser.add_argument(
        "--step27d_spec",
        type=Path,
        default=DEFAULT_STEP27D_SPEC,
    )

    parser.add_argument(
        "--step27d_summary",
        type=Path,
        default=DEFAULT_STEP27D_SUMMARY,
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
        step27c_axis_spec_path=(
            args.step27c_axis_spec.resolve()
        ),

        step27c_metadata_path=(
            args.step27c_metadata.resolve()
        ),

        step27d_manifest_path=(
            args.step27d_manifest.resolve()
        ),

        step27d_slide_summary_path=(
            args.step27d_slide_summary.resolve()
        ),

        step27d_spec_path=(
            args.step27d_spec.resolve()
        ),

        step27d_summary_path=(
            args.step27d_summary.resolve()
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
            "STEP 27G: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)