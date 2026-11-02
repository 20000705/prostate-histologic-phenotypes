#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27c_freeze_tcga_prad_pc3_pc6_external_evaluation_specification.py

STEP 27C
========
FREEZE TCGA-PRAD PC3 / PC6 EXTERNAL EVALUATION SPECIFICATION

VERSION
-------
1.0.0

SCHEMA
------
27c-tcga-prad-pc3-pc6-external-evaluation-specification-schema-1.0

ROLE
----
PRE-EXECUTION SCIENTIFIC / ANALYSIS FIREWALL ONLY.

This step freezes how the already-established discovery PC3 and PC6 axes
may be evaluated in the frozen TCGA-PRAD external cohort.

This script DOES NOT:
    - read TCGA PC3 or PC6 score values,
    - calculate any TCGA association,
    - calculate tumor localization,
    - calculate morphology associations,
    - read RNA / molecular outcomes,
    - inspect Gleason / clinical outcomes,
    - select thresholds,
    - select favorable slides,
    - fit or refit PCA,
    - screen PC1-PC8,
    - tune PC3 or PC6 signs,
    - create figures.

UPSTREAM AUTHORITIES
--------------------
Step06:
    Discovery-only frozen PC direction specification.

    Direction and score multiplier MUST be inherited exactly from:
        pc1_pc8_frozen_direction_specification.csv

Step23E:
    PC3 = primary interpreted histologic axis.
    PC6 = secondary interpreted histologic axis.

Step27A:
    Frozen authoritative TCGA-PRAD cohort.

Step27B:
    Frozen raw discovery-PCA projection:
        403 slides
        PC1-PC8
        no sign flip

CORE PRINCIPLE
--------------
TCGA is an external evaluation cohort.

The TCGA data must never:
    - redefine PC3 or PC6,
    - alter their direction,
    - select another PC,
    - alter the discovery PCA,
    - refine the phenotype definition,
    - choose a favorable endpoint after outcome inspection.

RAW VS ORIENTED SCORE
---------------------
Step27B stores the raw PCA orientation.

For an axis with discovery-frozen score_multiplier m:

    oriented_score = raw_pc_score * m

Thus larger oriented values always indicate the discovery-frozen
tumor-associated direction.

The multiplier must come from Step06 and must never be estimated from TCGA.

TCGA EVALUATION DOMAINS
-----------------------
1. AXIS TRANSFER / SCORE MATERIALIZATION
   Allowed.
   PC3 and PC6 may be materialized using the frozen Step06 multiplier.

2. DESCRIPTIVE SCORE DISTRIBUTION
   Allowed as technical/contextual characterization.
   This is NOT by itself biological external validation.

3. SPATIAL VISUALIZATION
   Allowed only as descriptive/supportive evidence.
   Case-selection rules must be frozen separately before visualization.
   No visually attractive or high-performing cases may be selected post hoc.

4. TUMOR LOCALIZATION
   NOT authorized by Step27C alone.
   Requires:
       - independent external tumor-label authority,
       - frozen evaluable cohort,
       - frozen predictor/sign,
       - frozen AUC definition,
       - separate pre-execution specification.

5. EXTERNAL MORPHOLOGY
   NOT authorized by Step27C alone.
   Requires a separately frozen external morphology endpoint and measurement
   authority before outcome inspection.

6. MOLECULAR / RNA ASSOCIATION
   NOT authorized by Step27C alone.
   Requires a separately frozen molecular analysis specification before
   reading PC-associated molecular results.

7. GLEASON / CLINICAL / SURVIVAL ASSOCIATION
   NOT authorized by Step27C alone.
   Requires separately frozen hypotheses, endpoints, covariates, and
   multiplicity rules.

CLAIM CEILING
-------------
Step27C itself establishes only:

    - the TCGA cohort and PCA projection are frozen;
    - PC3 and PC6 are prespecified axes;
    - their directions are inherited from discovery;
    - downstream TCGA analyses are firewall-protected from axis/sign selection.

Step27C itself does NOT establish:

    - external tumor localization,
    - external morphology replication,
    - molecular mechanism,
    - Gleason specificity,
    - survival association,
    - clinical utility,
    - independence of PC3 and PC6,
    - causal biological mechanisms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "27c-tcga-prad-pc3-pc6-external-evaluation-"
    "specification-schema-1.0"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT

# -------------------------------------------------------------------------
# Step06: discovery-only direction authority
# -------------------------------------------------------------------------

DEFAULT_STEP06_DIRECTION = (
    RESULTS
    / "06_frozen_pc_direction_specification"
    / "pc1_pc8_frozen_direction_specification.csv"
)

DEFAULT_STEP06_METADATA = (
    RESULTS
    / "06_frozen_pc_direction_specification"
    / "analysis_specification.json"
)

# -------------------------------------------------------------------------
# Step23E: final internal PC6 role / claim authority
# -------------------------------------------------------------------------

DEFAULT_STEP23E_DIR = (
    RESULTS
    / "23e_final_internal_pc6_phenotype_claim_freeze"
)

DEFAULT_STEP23E_METADATA = (
    DEFAULT_STEP23E_DIR
    / "analysis_metadata.json"
)

DEFAULT_STEP23E_CLAIM = (
    DEFAULT_STEP23E_DIR
    / "pc6_final_internal_claim_ceiling.txt"
)

# -------------------------------------------------------------------------
# Step27A: TCGA cohort freeze
# -------------------------------------------------------------------------

DEFAULT_STEP27A_MANIFEST = (
    RESULTS
    / "27a_tcga_prad_external_cohort_freeze"
    / "frozen_tcga_prad_external_cohort_manifest.csv"
)

DEFAULT_STEP27A_SUMMARY = (
    RESULTS
    / "27a_tcga_prad_external_cohort_freeze"
    / "tcga_prad_external_cohort_freeze_summary.json"
)

# -------------------------------------------------------------------------
# Step27B: raw frozen PCA projection
# -------------------------------------------------------------------------

DEFAULT_STEP27B_MANIFEST = (
    RESULTS
    / "27b_tcga_prad_frozen_pca_projection"
    / "frozen_tcga_prad_pca_projection_manifest.csv"
)

DEFAULT_STEP27B_SPEC = (
    RESULTS
    / "27b_tcga_prad_frozen_pca_projection"
    / "tcga_prad_pca_projection_specification.json"
)

DEFAULT_STEP27B_SUMMARY = (
    RESULTS
    / "27b_tcga_prad_frozen_pca_projection"
    / "tcga_prad_pca_projection_summary.json"
)

# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

DEFAULT_OUT_DIR = (
    RESULTS
    / "27c_tcga_prad_pc3_pc6_external_evaluation_specification"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_TCGA_SLIDES = 403
EXPECTED_PCS_PROJECTED = 8

TARGET_AXES = [
    "PC3",
    "PC6",
]

AXIS_ROLES = {
    "PC3": "PRIMARY_INTERPRETED_AXIS",
    "PC6": "SECONDARY_INTERPRETED_AXIS",
}


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


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
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


def bool_series(
    series: pd.Series,
) -> pd.Series:

    return (
        series
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


# =============================================================================
# STEP06 DIRECTION AUTHORITY
# =============================================================================

def load_pc_direction_authority(
    path: Path,
) -> pd.DataFrame:

    require_file(
        path,
        "Step06 direction specification",
    )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "component",
        "specification_status",
        "frozen_direction",
        "score_multiplier",
        "downstream_auc_source",
        "consensus_fraction",
        "direction_selection_dataset",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Step06 direction schema changed. "
            f"Missing columns: {missing}"
        )

    if (
        df["component"]
        .astype(str)
        .duplicated()
        .any()
    ):
        duplicates = df.loc[
            df["component"]
            .astype(str)
            .duplicated(
                keep=False
            )
        ]

        raise RuntimeError(
            "Step06 contains duplicate component rows:\n"
            + duplicates.to_string(
                index=False
            )
        )

    selected = (
        df.loc[
            df["component"]
            .astype(str)
            .isin(TARGET_AXES)
        ]
        .copy()
    )

    observed = set(
        selected[
            "component"
        ].astype(str)
    )

    expected = set(
        TARGET_AXES
    )

    if observed != expected:
        raise RuntimeError(
            "Step06 does not contain exactly PC3 and PC6 "
            "for the requested authority check. "
            f"Observed={sorted(observed)}"
        )

    rows = []

    for axis in TARGET_AXES:

        row = selected.loc[
            selected[
                "component"
            ].astype(str)
            == axis
        ].iloc[0]

        status = str(
            row[
                "specification_status"
            ]
        ).strip()

        direction = str(
            row[
                "frozen_direction"
            ]
        ).strip()

        multiplier = float(
            row[
                "score_multiplier"
            ]
        )

        auc_source = str(
            row[
                "downstream_auc_source"
            ]
        ).strip()

        dataset = str(
            row[
                "direction_selection_dataset"
            ]
        ).strip()

        consensus = float(
            row[
                "consensus_fraction"
            ]
        )

        if status != "frozen":
            raise RuntimeError(
                f"{axis}: Step06 status is "
                f"{status!r}, expected 'frozen'."
            )

        if direction not in {
            "high_in_tumor",
            "low_in_tumor",
        }:
            raise RuntimeError(
                f"{axis}: invalid frozen direction "
                f"{direction!r}"
            )

        expected_multiplier = (
            1.0
            if direction
            == "high_in_tumor"
            else -1.0
        )

        if not np.isclose(
            multiplier,
            expected_multiplier,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"{axis}: score_multiplier={multiplier} "
                f"is inconsistent with "
                f"frozen_direction={direction}"
            )

        expected_auc_source = (
            "auc_score_high_tumor"
            if direction
            == "high_in_tumor"
            else "auc_negative_score_low_tumor"
        )

        if auc_source != expected_auc_source:
            raise RuntimeError(
                f"{axis}: downstream_auc_source="
                f"{auc_source!r}; expected "
                f"{expected_auc_source!r}"
            )

        if (
            "discovery"
            not in dataset.lower()
        ):
            raise RuntimeError(
                f"{axis}: direction authority does not "
                f"appear discovery-only: {dataset!r}"
            )

        rows.append({
            "axis": axis,
            "axis_role": (
                AXIS_ROLES[axis]
            ),
            "step06_specification_status": (
                status
            ),
            "frozen_direction": (
                direction
            ),
            "score_multiplier": (
                int(multiplier)
            ),
            "downstream_auc_source": (
                auc_source
            ),
            "discovery_consensus_fraction": (
                consensus
            ),
            "direction_authority": (
                "Step06 discovery-only frozen direction"
            ),
            "tcga_sign_tuning_allowed": False,
            "tcga_axis_reselection_allowed": False,
        })

    out = pd.DataFrame(
        rows
    )

    return out


# =============================================================================
# STEP23E ROLE AUTHORITY
# =============================================================================

def validate_step23e(
    metadata_path: Path,
    claim_path: Path,
    axis_spec: pd.DataFrame,
) -> dict:

    require_file(
        metadata_path,
        "Step23E metadata",
    )

    require_file(
        claim_path,
        "Step23E claim ceiling",
    )

    metadata = load_json(
        metadata_path
    )

    pc3_role = str(
        metadata.get(
            "pc3_role",
            "",
        )
    )

    pc6_role = str(
        metadata.get(
            "pc6_role",
            "",
        )
    )

    if (
        pc3_role
        != "PRIMARY_INTERPRETED_AXIS"
    ):
        raise RuntimeError(
            "Step23E PC3 role changed: "
            f"{pc3_role!r}"
        )

    if (
        pc6_role
        != "SECONDARY_INTERPRETED_AXIS"
    ):
        raise RuntimeError(
            "Step23E PC6 role changed: "
            f"{pc6_role!r}"
        )

    step23e_pc6_direction = str(
        metadata.get(
            "pc6_frozen_direction",
            "",
        )
    )

    step06_pc6_direction = (
        axis_spec.loc[
            axis_spec["axis"]
            == "PC6",
            "frozen_direction",
        ]
        .iloc[0]
    )

    if (
        step23e_pc6_direction
        and step23e_pc6_direction
        != step06_pc6_direction
    ):
        raise RuntimeError(
            "PC6 direction mismatch between "
            "Step06 and Step23E: "
            f"Step06={step06_pc6_direction}, "
            f"Step23E={step23e_pc6_direction}"
        )

    return metadata


# =============================================================================
# STEP27A COHORT VALIDATION
# =============================================================================

def validate_step27a(
    manifest_path: Path,
    summary_path: Path,
) -> pd.DataFrame:

    require_file(
        manifest_path,
        "Step27A frozen cohort manifest",
    )

    require_file(
        summary_path,
        "Step27A freeze summary",
    )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "slide_id",
        "include_frozen_cohort",
        "embeddings_path",
        "coords_path",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Step27A manifest schema changed. "
            f"Missing columns: {missing}"
        )

    if len(df) != EXPECTED_TCGA_SLIDES:
        raise RuntimeError(
            "Step27A frozen cohort size changed: "
            f"{len(df)} != {EXPECTED_TCGA_SLIDES}"
        )

    if (
        df["slide_id"]
        .astype(str)
        .nunique()
        != EXPECTED_TCGA_SLIDES
    ):
        raise RuntimeError(
            "Step27A does not contain exactly "
            f"{EXPECTED_TCGA_SLIDES} unique slides."
        )

    if not bool_series(
        df[
            "include_frozen_cohort"
        ]
    ).all():
        raise RuntimeError(
            "Step27A contains rows outside the "
            "frozen cohort."
        )

    summary = load_json(
        summary_path
    )

    if (
        str(
            summary.get(
                "status",
                "",
            )
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Step27A summary status is not PASS."
        )

    return df


# =============================================================================
# STEP27B PROJECTION VALIDATION
# =============================================================================

def validate_step27b(
    manifest_path: Path,
    spec_path: Path,
    summary_path: Path,
    cohort27a: pd.DataFrame,
) -> pd.DataFrame:

    require_file(
        manifest_path,
        "Step27B projection manifest",
    )

    require_file(
        spec_path,
        "Step27B projection specification",
    )

    require_file(
        summary_path,
        "Step27B projection summary",
    )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "slide_id",
        "include_frozen_cohort",
        "pc1_pc8_path",
        "n_tiles",
        "n_projected_components",
        "projection_dtype",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Step27B manifest schema changed. "
            f"Missing columns: {missing}"
        )

    if len(df) != EXPECTED_TCGA_SLIDES:
        raise RuntimeError(
            "Step27B projection cohort size changed: "
            f"{len(df)} != {EXPECTED_TCGA_SLIDES}"
        )

    if (
        df["slide_id"]
        .astype(str)
        .nunique()
        != EXPECTED_TCGA_SLIDES
    ):
        raise RuntimeError(
            "Step27B projection manifest does not "
            "contain exactly 403 unique slides."
        )

    ids27a = set(
        cohort27a[
            "slide_id"
        ].astype(str)
    )

    ids27b = set(
        df[
            "slide_id"
        ].astype(str)
    )

    if ids27a != ids27b:
        raise RuntimeError(
            "Step27A and Step27B slide sets differ."
        )

    n_components = pd.to_numeric(
        df[
            "n_projected_components"
        ],
        errors="raise",
    )

    if not (
        n_components
        == EXPECTED_PCS_PROJECTED
    ).all():
        raise RuntimeError(
            "Step27B does not contain exactly "
            "PC1-PC8 projections for every slide."
        )

    spec = load_json(
        spec_path
    )

    orientation = (
        spec
        .get(
            "pc_orientation_policy",
            {},
        )
    )

    if bool(
        orientation.get(
            "sign_flipping_performed",
            True,
        )
    ):
        raise RuntimeError(
            "Step27B unexpectedly reports "
            "sign flipping."
        )

    if bool(
        orientation.get(
            "tumor_direction_used",
            True,
        )
    ):
        raise RuntimeError(
            "Step27B unexpectedly used tumor direction "
            "during raw PCA projection."
        )

    summary = load_json(
        summary_path
    )

    if (
        str(
            summary.get(
                "status",
                "",
            )
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Step27B summary status is not PASS."
        )

    if int(
        summary.get(
            "unique_slides",
            -1,
        )
    ) != EXPECTED_TCGA_SLIDES:
        raise RuntimeError(
            "Step27B summary unique-slide count "
            "is not 403."
        )

    if int(
        summary.get(
            "components_saved",
            -1,
        )
    ) != EXPECTED_PCS_PROJECTED:
        raise RuntimeError(
            "Step27B summary components_saved "
            "is not 8."
        )

    return df


# =============================================================================
# EVALUATION DOMAIN TABLE
# =============================================================================

def build_evaluation_domain_table() -> pd.DataFrame:

    rows: List[Dict] = [
        {
            "domain": (
                "axis_score_materialization"
            ),
            "status": "AUTHORIZED",
            "role": "TECHNICAL",
            "primary_unit": "tile",
            "frozen_rule": (
                "oriented_score = raw_PC_score * "
                "Step06 score_multiplier"
            ),
            "requires_separate_specification": False,
            "may_support_external_biological_claim": False,
            "notes": (
                "No sign estimation from TCGA."
            ),
        },
        {
            "domain": (
                "descriptive_score_distribution"
            ),
            "status": "AUTHORIZED",
            "role": "DESCRIPTIVE_CONTEXT",
            "primary_unit": "slide",
            "frozen_rule": (
                "Describe frozen PC3/PC6 distributions "
                "without hypothesis promotion."
            ),
            "requires_separate_specification": False,
            "may_support_external_biological_claim": False,
            "notes": (
                "Distribution alone is not phenotype validation."
            ),
        },
        {
            "domain": (
                "spatial_visualization"
            ),
            "status": "CONDITIONALLY_AUTHORIZED",
            "role": "SUPPORTIVE_VISUAL",
            "primary_unit": "slide",
            "frozen_rule": (
                "Visualization case-selection and display "
                "rules must be frozen before rendering."
            ),
            "requires_separate_specification": True,
            "may_support_external_biological_claim": False,
            "notes": (
                "No outcome-guided or attractiveness-guided "
                "case selection."
            ),
        },
        {
            "domain": (
                "tumor_localization"
            ),
            "status": "NOT_AUTHORIZED_YET",
            "role": "FORMAL_EXTERNAL_VALIDATION",
            "primary_unit": "slide",
            "frozen_rule": (
                "Requires independent frozen tumor-label "
                "authority and pre-execution AUC specification."
            ),
            "requires_separate_specification": True,
            "may_support_external_biological_claim": True,
            "notes": (
                "No AUC may be calculated under Step27C alone."
            ),
        },
        {
            "domain": (
                "external_morphology"
            ),
            "status": "NOT_AUTHORIZED_YET",
            "role": "FORMAL_EXTERNAL_VALIDATION",
            "primary_unit": "slide_or_patient",
            "frozen_rule": (
                "Morphology metric/state/threshold definitions "
                "must be frozen before outcome inspection."
            ),
            "requires_separate_specification": True,
            "may_support_external_biological_claim": True,
            "notes": (
                "TCGA must not redefine the internal phenotype."
            ),
        },
        {
            "domain": (
                "molecular_rna_association"
            ),
            "status": "NOT_AUTHORIZED_YET",
            "role": "FORMAL_OR_SUPPORTIVE_MOLECULAR",
            "primary_unit": "patient",
            "frozen_rule": (
                "Gene/pathway hypotheses, aggregation rule, "
                "covariates and multiplicity strategy must be "
                "frozen before reading PC-associated results."
            ),
            "requires_separate_specification": True,
            "may_support_external_biological_claim": True,
            "notes": (
                "No pathway or gene may be selected because it "
                "looks favorable after testing."
            ),
        },
        {
            "domain": (
                "gleason_clinical_association"
            ),
            "status": "NOT_AUTHORIZED_YET",
            "role": "FORMAL_OR_SUPPORTIVE_CLINICAL",
            "primary_unit": "patient",
            "frozen_rule": (
                "Endpoint, encoding, covariates and inferential "
                "model must be frozen separately."
            ),
            "requires_separate_specification": True,
            "may_support_external_biological_claim": True,
            "notes": (
                "No outcome-guided endpoint selection."
            ),
        },
        {
            "domain": (
                "survival_association"
            ),
            "status": "NOT_AUTHORIZED_YET",
            "role": "EXPLORATORY_UNLESS_SEPARATELY_FROZEN",
            "primary_unit": "patient",
            "frozen_rule": (
                "Time origin, event definition, censoring, "
                "aggregation and covariates must be frozen."
            ),
            "requires_separate_specification": True,
            "may_support_external_biological_claim": True,
            "notes": (
                "Not part of Step27C."
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CLAIM CEILING
# =============================================================================

def claim_ceiling_text(
    axis_spec: pd.DataFrame,
) -> str:

    pc3 = axis_spec.loc[
        axis_spec["axis"]
        == "PC3"
    ].iloc[0]

    pc6 = axis_spec.loc[
        axis_spec["axis"]
        == "PC6"
    ].iloc[0]

    return f"""STEP 27C — TCGA-PRAD PC3 / PC6 CLAIM CEILING

ROLE
----
PRE-EXECUTION EXTERNAL EVALUATION FREEZE ONLY.

FROZEN AXES
-----------
PC3:
    scientific role:
        PRIMARY_INTERPRETED_AXIS

    discovery-frozen direction:
        {pc3['frozen_direction']}

    score multiplier:
        {int(pc3['score_multiplier'])}

PC6:
    scientific role:
        SECONDARY_INTERPRETED_AXIS

    discovery-frozen direction:
        {pc6['frozen_direction']}

    score multiplier:
        {int(pc6['score_multiplier'])}


ALLOWED FROM STEP27C ITSELF
---------------------------
1. State that TCGA-PRAD uses a frozen 403-slide cohort.
2. State that all slides were projected using the frozen discovery PCA.
3. State that PC3 and PC6 were selected before TCGA outcome evaluation.
4. State that PC3 / PC6 directions are inherited from the discovery-only
   Step06 specification.
5. Materialize oriented PC3 / PC6 scores using the frozen Step06 multipliers.
6. Produce technical/descriptive score summaries.
7. Plan later analyses under separately frozen specifications.


NOT ESTABLISHED BY STEP27C
--------------------------
1. TCGA tumor-localization performance.
2. TCGA morphology replication.
3. TCGA molecular association.
4. A molecular mechanism for PC3 or PC6.
5. Gleason specificity.
6. Clinical utility.
7. Prognostic value.
8. Survival association.
9. Statistical independence of PC3 and PC6.
10. Causal independence of PC3 and PC6.
11. Completeness of PC3/PC6 as a representation of prostate histology.


PROHIBITED
----------
1. Refit PCA on TCGA.
2. Refit the scaler on TCGA.
3. Screen PC1-PC8 using TCGA outcomes.
4. Replace PC3 or PC6 because another PC performs better in TCGA.
5. Tune PC3 or PC6 sign using TCGA.
6. Select favorable TCGA thresholds.
7. Select favorable molecular genes/pathways after outcome inspection and
   describe them as prespecified.
8. Select visually attractive TCGA cases and present them as unbiased
   validation.
9. Use pooled tiles as independent biological replicates.
10. Treat descriptive spatial maps alone as proof of external phenotype
    replication.


DOWNSTREAM REQUIREMENT
----------------------
Every formal biological TCGA endpoint must receive its own frozen
pre-execution specification before the corresponding outcome is inspected.
"""


# =============================================================================
# README
# =============================================================================

def readme_text(
    axis_spec: pd.DataFrame,
) -> str:

    pc3 = axis_spec.loc[
        axis_spec["axis"]
        == "PC3"
    ].iloc[0]

    pc6 = axis_spec.loc[
        axis_spec["axis"]
        == "PC6"
    ].iloc[0]

    return f"""STEP 27C
TCGA-PRAD PC3 / PC6 EXTERNAL EVALUATION SPECIFICATION

Version
-------
{VERSION}

Schema
------
{SCHEMA}


ROLE
----
PRE-EXECUTION SCIENTIFIC FREEZE ONLY.

No TCGA PC3 / PC6 outcome values are read by this script.


FROZEN TCGA COHORT
------------------
403 unique slides.

Authority:
    Step27A


FROZEN PCA PROJECTION
---------------------
PC1-PC8 raw discovery-PCA projections.

Authority:
    Step27B

No PCA refit.
No scaler refit.
No sign flip in Step27B.


AXIS HIERARCHY
--------------
PC3:
    PRIMARY_INTERPRETED_AXIS

PC6:
    SECONDARY_INTERPRETED_AXIS


FROZEN DIRECTIONS
-----------------
PC3:
    direction       = {pc3['frozen_direction']}
    score_multiplier= {int(pc3['score_multiplier'])}

PC6:
    direction       = {pc6['frozen_direction']}
    score_multiplier= {int(pc6['score_multiplier'])}

Authority:
    Step06 discovery-only direction specification.


ORIENTED SCORE
--------------
For axis PCk:

    oriented_PCk = raw_PCk * frozen_score_multiplier

The multiplier is inherited from Step06.

TCGA may not change it.


AUTHORIZED NOW
--------------
1. Frozen PC3 / PC6 oriented-score materialization.
2. Technical/descriptive score summaries.


SEPARATE FREEZE REQUIRED
------------------------
1. Spatial visualization case selection.
2. Tumor localization.
3. External morphology.
4. Molecular / RNA association.
5. Gleason / clinical association.
6. Survival analysis.


FIREWALL
--------
PCA refit:
    NO

Scaler refit:
    NO

PC1-PC8 re-screen:
    NO

PC3 sign tuning:
    NO

PC6 sign tuning:
    NO

PC replacement:
    NO

TCGA-derived phenotype redefinition:
    NO

Outcome-guided threshold selection:
    NO

Outcome-guided slide selection:
    NO

Outcome-guided gene/pathway selection:
    NO

Manual visual cherry-picking:
    NO


NEXT
----
A suitable next technical step is to materialize frozen oriented PC3 / PC6
scores from the Step27B PC1-PC8 arrays without reading any biological TCGA
outcomes.

Formal biological endpoints should then be frozen individually before
execution.
"""


# =============================================================================
# MAIN
# =============================================================================

def run(
    step06_direction: Path,
    step06_metadata: Path,
    step23e_metadata: Path,
    step23e_claim: Path,
    step27a_manifest: Path,
    step27a_summary: Path,
    step27b_manifest: Path,
    step27b_spec: Path,
    step27b_summary: Path,
    out_dir: Path,
) -> None:

    banner(
        "STEP 27C: FREEZE TCGA-PRAD PC3 / PC6 "
        "EXTERNAL EVALUATION SPECIFICATION"
    )

    print(
        f"[VERSION]       {VERSION}"
    )
    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    # ------------------------------------------------------------------
    # Validate all source files.
    # ------------------------------------------------------------------

    require_file(
        step06_direction,
        "Step06 direction CSV",
    )

    require_file(
        step06_metadata,
        "Step06 metadata",
    )

    # ------------------------------------------------------------------
    # Step06 direction authority
    # ------------------------------------------------------------------

    banner(
        "VALIDATE DISCOVERY-FROZEN AXIS DIRECTIONS"
    )

    axis_spec = (
        load_pc_direction_authority(
            step06_direction
        )
    )

    print(
        axis_spec[
            [
                "axis",
                "axis_role",
                "frozen_direction",
                "score_multiplier",
                "discovery_consensus_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Step23E hierarchy / claim validation
    # ------------------------------------------------------------------

    banner(
        "VALIDATE INTERNAL AXIS HIERARCHY"
    )

    step23e_metadata_json = (
        validate_step23e(
            metadata_path=(
                step23e_metadata
            ),
            claim_path=(
                step23e_claim
            ),
            axis_spec=axis_spec,
        )
    )

    print(
        "PC3 primary interpreted axis : YES"
    )
    print(
        "PC6 secondary interpreted axis: YES"
    )
    print(
        "PC6 Step06/Step23E direction concordance: YES"
    )

    # ------------------------------------------------------------------
    # Step27A
    # ------------------------------------------------------------------

    banner(
        "VALIDATE FROZEN TCGA COHORT"
    )

    cohort27a = (
        validate_step27a(
            manifest_path=(
                step27a_manifest
            ),
            summary_path=(
                step27a_summary
            ),
        )
    )

    print(
        f"Frozen slides : {len(cohort27a)}"
    )
    print(
        "Unique slides : "
        f"{cohort27a['slide_id'].nunique()}"
    )

    # ------------------------------------------------------------------
    # Step27B
    # ------------------------------------------------------------------

    banner(
        "VALIDATE FROZEN TCGA PCA PROJECTION"
    )

    projection27b = (
        validate_step27b(
            manifest_path=(
                step27b_manifest
            ),
            spec_path=(
                step27b_spec
            ),
            summary_path=(
                step27b_summary
            ),
            cohort27a=(
                cohort27a
            ),
        )
    )

    total_tiles = int(
        pd.to_numeric(
            projection27b[
                "n_tiles"
            ],
            errors="raise",
        ).sum()
    )

    print(
        f"Projected slides: "
        f"{len(projection27b)}"
    )
    print(
        f"Total tiles     : "
        f"{total_tiles:,}"
    )
    print(
        "Raw PCs         : PC1-PC8"
    )
    print(
        "Step27B sign flip: NO"
    )

    # ------------------------------------------------------------------
    # Freeze evaluation domains
    # ------------------------------------------------------------------

    evaluation_domains = (
        build_evaluation_domain_table()
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    axis_path = (
        out_dir
        / "tcga_prad_pc3_pc6_frozen_axis_specification.csv"
    )

    domains_path = (
        out_dir
        / "tcga_prad_external_evaluation_domains.csv"
    )

    claim_path = (
        out_dir
        / "tcga_prad_pc3_pc6_claim_ceiling.txt"
    )

    metadata_path = (
        out_dir
        / "tcga_prad_pc3_pc6_external_evaluation_specification.json"
    )

    hashes_path = (
        out_dir
        / "source_hashes.csv"
    )

    readme_path = (
        out_dir
        / "README.txt"
    )

    axis_spec.to_csv(
        axis_path,
        index=False,
    )

    evaluation_domains.to_csv(
        domains_path,
        index=False,
    )

    claim_path.write_text(
        claim_ceiling_text(
            axis_spec
        ),
        encoding="utf-8",
    )

    readme_path.write_text(
        readme_text(
            axis_spec
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Source hashes
    # ------------------------------------------------------------------

    sources = [
        (
            "step06_direction_specification",
            step06_direction,
        ),
        (
            "step06_metadata",
            step06_metadata,
        ),
        (
            "step23e_metadata",
            step23e_metadata,
        ),
        (
            "step23e_claim_ceiling",
            step23e_claim,
        ),
        (
            "step27a_manifest",
            step27a_manifest,
        ),
        (
            "step27a_summary",
            step27a_summary,
        ),
        (
            "step27b_manifest",
            step27b_manifest,
        ),
        (
            "step27b_projection_specification",
            step27b_spec,
        ),
        (
            "step27b_summary",
            step27b_summary,
        ),
    ]

    hash_rows = []

    for label, path in sources:
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

    source_hashes = pd.DataFrame(
        hash_rows
    )

    source_hashes.to_csv(
        hashes_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    axis_records = (
        axis_spec
        .to_dict(
            orient="records"
        )
    )

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "analysis_role": (
            "PRE_EXECUTION_TCGA_EXTERNAL_EVALUATION_FREEZE_ONLY"
        ),
        "cohort": {
            "name": "TCGA-PRAD",
            "frozen_slides": (
                EXPECTED_TCGA_SLIDES
            ),
            "observed_frozen_slides": int(
                len(cohort27a)
            ),
            "observed_unique_slides": int(
                cohort27a[
                    "slide_id"
                ].nunique()
            ),
            "total_projected_tiles": (
                total_tiles
            ),
            "cohort_authority": (
                "Step27A"
            ),
        },
        "projection": {
            "authority": "Step27B",
            "raw_components": [
                f"PC{i}"
                for i in range(
                    1,
                    9,
                )
            ],
            "sign_flipping_in_step27b": False,
            "pca_refit_in_tcga": False,
            "scaler_refit_in_tcga": False,
        },
        "axis_hierarchy": {
            "PC3": (
                "PRIMARY_INTERPRETED_AXIS"
            ),
            "PC6": (
                "SECONDARY_INTERPRETED_AXIS"
            ),
        },
        "frozen_axes": (
            axis_records
        ),
        "oriented_score_definition": (
            "oriented_score = raw_discovery_PCA_score "
            "* Step06_score_multiplier"
        ),
        "direction_authority": (
            "Step06 discovery-only frozen direction specification"
        ),
        "internal_claim_authority": (
            "Step23E"
        ),
        "evaluation_domains": (
            evaluation_domains
            .to_dict(
                orient="records"
            )
        ),
        "freeze_only_audit": {
            "tcga_pc3_score_values_read": False,
            "tcga_pc6_score_values_read": False,
            "tcga_tumor_auc_calculated": False,
            "tcga_morphology_outcomes_read": False,
            "tcga_rna_outcomes_read": False,
            "tcga_clinical_outcomes_read": False,
            "tcga_survival_outcomes_read": False,
            "pca_refit": False,
            "scaler_refit": False,
            "pc1_pc8_rescreening": False,
            "pc3_sign_tuning": False,
            "pc6_sign_tuning": False,
            "pc3_pc6_role_swap": False,
            "pc_replacement": False,
            "threshold_selection": False,
            "gene_selection": False,
            "pathway_selection": False,
            "visual_case_selection": False,
            "manual_case_selection": False,
            "external_phenotype_redefinition": False,
        },
        "claim_ceiling": {
            "step27c_establishes_biological_external_validation": False,
            "tumor_localization_claim_allowed_now": False,
            "morphology_replication_claim_allowed_now": False,
            "molecular_claim_allowed_now": False,
            "gleason_claim_allowed_now": False,
            "clinical_claim_allowed_now": False,
            "survival_claim_allowed_now": False,
            "mechanistic_claim_allowed_now": False,
            "pc3_pc6_independence_claim_allowed_now": False,
        },
        "upstream": {
            "step06_direction": str(
                step06_direction.resolve()
            ),
            "step23e_metadata": str(
                step23e_metadata.resolve()
            ),
            "step27a_manifest": str(
                step27a_manifest.resolve()
            ),
            "step27b_manifest": str(
                step27b_manifest.resolve()
            ),
        },
        "outputs": {
            "axis_specification": str(
                axis_path.resolve()
            ),
            "evaluation_domains": str(
                domains_path.resolve()
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
        "next_step": (
            "Materialize frozen oriented PC3/PC6 scores "
            "from Step27B without reading biological TCGA outcomes."
        ),
    }

    save_json_atomic(
        metadata_path,
        metadata,
    )

    # ------------------------------------------------------------------
    # Final audit
    # ------------------------------------------------------------------

    banner(
        "STEP 27C FREEZE-ONLY AUDIT"
    )

    audit = [
        (
            "TCGA frozen cohort inherited",
            "YES",
        ),
        (
            "403-slide cohort retained",
            "YES",
        ),
        (
            "Step27B projection inherited",
            "YES",
        ),
        (
            "PC3 primary-axis role retained",
            "YES",
        ),
        (
            "PC6 secondary-axis role retained",
            "YES",
        ),
        (
            "PC3 direction inherited from Step06",
            "YES",
        ),
        (
            "PC6 direction inherited from Step06",
            "YES",
        ),
        (
            "TCGA PC3 outcome values read",
            "NO",
        ),
        (
            "TCGA PC6 outcome values read",
            "NO",
        ),
        (
            "PCA refit",
            "NO",
        ),
        (
            "Scaler refit",
            "NO",
        ),
        (
            "PC1-PC8 re-screening",
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
            "PC replacement",
            "NO",
        ),
        (
            "Tumor AUC calculated",
            "NO",
        ),
        (
            "Morphology outcomes read",
            "NO",
        ),
        (
            "RNA outcomes read",
            "NO",
        ),
        (
            "Clinical outcomes read",
            "NO",
        ),
        (
            "Gene/pathway selection",
            "NO",
        ),
        (
            "Threshold selection",
            "NO",
        ),
        (
            "Manual visual selection",
            "NO",
        ),
        (
            "Claim ceiling frozen",
            "YES",
        ),
    ]

    for label, value in audit:
        print(
            f"{label:<52}: {value}"
        )

    banner(
        "STEP 27C: PASS"
    )

    print(
        f"[TCGA SLIDES]    {len(cohort27a)}"
    )
    print(
        f"[TOTAL TILES]    {total_tiles:,}"
    )

    for _, row in axis_spec.iterrows():
        print(
            f"[{row['axis']}]"
            f"{'':<7}"
            f"role={row['axis_role']} | "
            f"direction={row['frozen_direction']} | "
            f"multiplier={row['score_multiplier']:+d}"
        )

    print()
    print(
        f"[AXIS SPEC]      {axis_path}"
    )
    print(
        f"[DOMAINS]        {domains_path}"
    )
    print(
        f"[CLAIM CEILING]  {claim_path}"
    )
    print(
        f"[METADATA]       {metadata_path}"
    )
    print(
        f"[HASHES]         {hashes_path}"
    )
    print(
        f"[README]         {readme_path}"
    )

    print()
    print(
        "[NEXT] Materialize frozen oriented PC3/PC6 "
        "scores from Step27B only."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Freeze TCGA-PRAD PC3/PC6 external "
            "evaluation specification."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--step06_direction",
        type=Path,
        default=DEFAULT_STEP06_DIRECTION,
    )

    parser.add_argument(
        "--step06_metadata",
        type=Path,
        default=DEFAULT_STEP06_METADATA,
    )

    parser.add_argument(
        "--step23e_metadata",
        type=Path,
        default=DEFAULT_STEP23E_METADATA,
    )

    parser.add_argument(
        "--step23e_claim",
        type=Path,
        default=DEFAULT_STEP23E_CLAIM,
    )

    parser.add_argument(
        "--step27a_manifest",
        type=Path,
        default=DEFAULT_STEP27A_MANIFEST,
    )

    parser.add_argument(
        "--step27a_summary",
        type=Path,
        default=DEFAULT_STEP27A_SUMMARY,
    )

    parser.add_argument(
        "--step27b_manifest",
        type=Path,
        default=DEFAULT_STEP27B_MANIFEST,
    )

    parser.add_argument(
        "--step27b_spec",
        type=Path,
        default=DEFAULT_STEP27B_SPEC,
    )

    parser.add_argument(
        "--step27b_summary",
        type=Path,
        default=DEFAULT_STEP27B_SUMMARY,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:

    args = parse_args()

    run(
        step06_direction=(
            args.step06_direction.resolve()
        ),
        step06_metadata=(
            args.step06_metadata.resolve()
        ),
        step23e_metadata=(
            args.step23e_metadata.resolve()
        ),
        step23e_claim=(
            args.step23e_claim.resolve()
        ),
        step27a_manifest=(
            args.step27a_manifest.resolve()
        ),
        step27a_summary=(
            args.step27a_summary.resolve()
        ),
        step27b_manifest=(
            args.step27b_manifest.resolve()
        ),
        step27b_spec=(
            args.step27b_spec.resolve()
        ),
        step27b_summary=(
            args.step27b_summary.resolve()
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
            "STEP 27C: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)