#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 24A: FREEZE AGGC EXTERNAL PC6 MORPHOLOGY TRANSFER SPECIFICATION

VERSION: 1.0.0
SCHEMA : 24a-aggc-external-pc6-morphology-transfer-freeze-schema-1.0

ROLE
----
PRE-EXECUTION EXTERNAL-VALIDATION FREEZE ONLY.

This step freezes how the INTERNAL PC6 phenotype established in Steps 23A-23E
will be tested in AGGC.

It MUST NOT calculate any AGGC PC6-morphology association.

SCIENTIFIC QUESTION
-------------------
Does the internally frozen PC6 morphology phenotype transfer directionally to
an independent AGGC cohort?

INTERNAL AUTHORITY
------------------
Step23E is the final INTERNAL PC6 phenotype / claim-boundary authority.

Step23C is consulted only to verify the complete frozen internal morphology
evidence set that Step23E summarizes. Step24A does NOT redefine the phenotype.

PC6 remains:
    raw Virchow2 PC6
    frozen direction = high_in_tumor

AGGC does not change the sign.

EXTERNAL PHENOTYPE TRANSFER DEFINITION
--------------------------------------
Within each AGGC slide:

    PC6_LOW  = raw PC6 <= within-slide q20
    PC6_HIGH = raw PC6 >= within-slide q80

External morphology effect:

    PC6_HIGH - PC6_LOW

This mirrors the frozen INTERNAL PC6 definition.

PRIMARY HELD-OUT REPLICATION PARTITION
--------------------------------------
AGGC Subset1_Test

SECONDARY CHARACTERIZATION PARTITION
------------------------------------
AGGC Subset1_Train

Train and Test are NEVER pooled for the primary inference.

FROZEN CONFIRMED TRANSFER ENDPOINTS
-----------------------------------
The following directions were established internally BEFORE AGGC evaluation
and are frozen as the formal external transfer targets:

    SEI_eosin_stroma          POSITIVE
        PC6_HIGH > PC6_LOW

    GOI_lumen_openness        NEGATIVE
        PC6_HIGH < PC6_LOW

    CLR_nuclear_to_lumen      POSITIVE
        PC6_HIGH > PC6_LOW

    lumen_area                NEGATIVE
        PC6_HIGH < PC6_LOW

    mean_lumen_area           NEGATIVE
        PC6_HIGH < PC6_LOW

    lumen_fragmentation       POSITIVE
        PC6_HIGH > PC6_LOW

NPI_nuclear_packing is NOT a confirmed PC6 phenotype and may not be promoted
if AGGC happens to show a favorable result.

EXTERNAL TRANSFER SUCCESS
-------------------------
The formal question is directional replication of the PRE-FROZEN endpoints.

Metric-level:
    estimate direction matches the internally frozen direction.

Primary cohort-level summary:
    direction-concordance fraction across all six frozen transfer endpoints
    in Subset1_Test.

Supportive:
    individual slide-level paired effects and bootstrap confidence intervals
    for each pre-frozen endpoint.

No new AGGC morphology metric may be added to the formal endpoint family.

AGGC TILE / SLIDE ELIGIBILITY
-----------------------------
Primary slide eligibility:
    >= 50 finite tissue tiles with raw PC6 and all six frozen transfer metrics.

Rationale:
    q20/q80 yields approximately >=10 tiles per tail at 50 eligible tiles.

Frozen deterministic sampling:
    up to 10 tiles / slide / stratum

Sampling seed:
    20260810

No patient cap is imposed because the formal AGGC external analysis unit is
the slide and the source does not require Step24A to infer unverified patient
groupings from slide names.

MEASUREMENT SOURCE
------------------
The already-created AGGC unified tile morphology table may be used in Step24B
only as a MEASUREMENT SOURCE.

Step24A may inspect:
    file existence
    schema/header
    static file provenance

Step24A MUST NOT read AGGC PC6 or morphology outcome rows.

NORMAL/G3/G4/G5
---------------
Not part of the Step24 PC6 morphology-transfer endpoint.

Histologic class labels may remain in the source file for provenance but are
not used to define PC6_LOW/PC6_HIGH or the formal morphology transfer.

STRICT FIREWALL
---------------
Step24A MUST NOT:
    - read AGGC PC6 values;
    - read AGGC morphology outcome values;
    - calculate PC6 quantiles;
    - calculate PC6_HIGH/PC6_LOW effects;
    - calculate p-values;
    - calculate bootstrap CIs;
    - calculate tumor AUC;
    - screen PC1-PC8;
    - refit PCA;
    - change PC6 sign;
    - choose a favorable AGGC partition;
    - choose a favorable morphology metric;
    - promote NPI from AGGC;
    - use Gleason states to redefine PC6;
    - use AGGC to revise the internal PC6 phenotype.

NEXT
----
24B = AGGC measurement only under this frozen contract.
24C = frozen AGGC morphology-transfer evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "24a-aggc-external-pc6-morphology-transfer-freeze-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

MEDIA = REPO_ROOT

RESULTS = RESULTS_ROOT

STEP23C = RESULTS / "23c_internal_pc6_morphology_inference_pc3_pc6_distinctness"
STEP23E = RESULTS / "23e_final_internal_pc6_phenotype_claim_freeze"

STEP23C_PRIMARY = STEP23C / "pc6_primary_patient_level_results.csv"
STEP23C_SECONDARY = STEP23C / "pc6_secondary_patient_level_results.csv"
STEP23C_METADATA = STEP23C / "analysis_metadata.json"

STEP23E_MORPH = STEP23E / "pc6_final_internal_morphology_evidence.csv"
STEP23E_CLAIMS = STEP23E / "pc6_final_internal_claim_boundary.csv"
STEP23E_CEILING = STEP23E / "pc6_final_internal_claim_ceiling.txt"
STEP23E_METADATA = STEP23E / "analysis_metadata.json"

AGGC_ROOT = CONFIG_AGGC_ROOT

AGGC_EMBED_ROOT = AGGC_ROOT / "aggc_virchow2_embeddings"

AGGC_MORPH_SOURCE = (
    AGGC_ROOT
    / "aggc_pc1_pc8_unified_screening_input"
    / "aggc_all_pc1_pc8_tiles_with_labels_morphology.csv.gz"
)

AGGC_LABEL_ROOT = AGGC_ROOT / "aggc_gleason_tile_labels"

OUT_DIR = RESULTS / "24a_aggc_external_pc6_morphology_transfer_specification"


# =============================================================================
# FROZEN SCIENTIFIC CONSTANTS
# =============================================================================

FROZEN_PC6_DIRECTION = "high_in_tumor"

PRIMARY_PARTITION = "Subset1_Test"
SECONDARY_PARTITION = "Subset1_Train"

LOW_QUANTILE = 0.20
HIGH_QUANTILE = 0.80

MIN_ELIGIBLE_TILES_PER_SLIDE = 50
MAX_TILES_PER_SLIDE_PER_STRATUM = 10
SAMPLING_SEED = 20260810

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260810

# Complete formal external transfer family.
FROZEN_TRANSFER_ENDPOINTS = [
    {
        "metric": "SEI_eosin_stroma",
        "family": "stroma_eosin",
        "internal_status": "CONFIRMED",
        "expected_sign": +1,
        "expected_direction": "POSITIVE",
        "external_role": "FORMAL_TRANSFER_ENDPOINT",
        "interpretation": "PC6_HIGH has greater eosin/stroma signal than PC6_LOW",
    },
    {
        "metric": "GOI_lumen_openness",
        "family": "gland_lumen",
        "internal_status": "CONFIRMED",
        "expected_sign": -1,
        "expected_direction": "NEGATIVE",
        "external_role": "FORMAL_TRANSFER_ENDPOINT",
        "interpretation": "PC6_HIGH has lower lumen openness than PC6_LOW",
    },
    {
        "metric": "CLR_nuclear_to_lumen",
        "family": "nuclear_lumen_balance",
        "internal_status": "CONFIRMED",
        "expected_sign": +1,
        "expected_direction": "POSITIVE",
        "external_role": "FORMAL_TRANSFER_ENDPOINT",
        "interpretation": "PC6_HIGH has greater nuclear-to-lumen ratio than PC6_LOW",
    },
    {
        "metric": "lumen_area",
        "family": "gland_lumen",
        "internal_status": "CONFIRMED",
        "expected_sign": -1,
        "expected_direction": "NEGATIVE",
        "external_role": "FORMAL_TRANSFER_ENDPOINT",
        "interpretation": "PC6_HIGH has lower total lumen area than PC6_LOW",
    },
    {
        "metric": "mean_lumen_area",
        "family": "gland_lumen",
        "internal_status": "CONFIRMED",
        "expected_sign": -1,
        "expected_direction": "NEGATIVE",
        "external_role": "FORMAL_TRANSFER_ENDPOINT",
        "interpretation": "PC6_HIGH has smaller mean lumen area than PC6_LOW",
    },
    {
        "metric": "lumen_fragmentation",
        "family": "gland_lumen",
        "internal_status": "CONFIRMED",
        "expected_sign": +1,
        "expected_direction": "POSITIVE",
        "external_role": "FORMAL_TRANSFER_ENDPOINT",
        "interpretation": "PC6_HIGH has greater lumen fragmentation than PC6_LOW",
    },
]

SPECIFICITY_REFERENCE = {
    "metric": "NPI_nuclear_packing",
    "internal_status": "NOT_CONFIRMED_PRIMARY_PHENOTYPE",
    "external_role": "SPECIFICITY_REFERENCE_ONLY",
    "may_be_promoted_from_aggc": False,
}

REQUIRED_SOURCE_COLUMNS = [
    "slide_id",
    "PC6",
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "CLR_nuclear_to_lumen",
    "NPI_nuclear_packing",
    "lumen_area",
    "mean_lumen_area",
    "lumen_fragmentation",
]

OPTIONAL_PROVENANCE_COLUMNS = [
    "tile_index",
    "x",
    "y",
    "histologic_class",
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC7",
    "PC8",
]


# =============================================================================
# HELPERS
# =============================================================================

def banner(title: str, width: int = 132) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}:\n{path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required {label}:\n{path}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty:\n{path}\n"
                "Use --overwrite only for an intentional re-freeze before "
                "Step24B/24C execution."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def normalize_direction(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def parse_partition(slide_id: str) -> Optional[str]:
    s = str(slide_id)

    if s.startswith("Subset1_Test_"):
        return PRIMARY_PARTITION

    if s.startswith("Subset1_Train_"):
        return SECONDARY_PARTITION

    return None


# =============================================================================
# INTERNAL AUTHORITY VALIDATION
# =============================================================================

def validate_step23_authority() -> Dict[str, Any]:

    required = [
        (STEP23C_PRIMARY, "Step23C primary morphology results"),
        (STEP23C_SECONDARY, "Step23C secondary morphology results"),
        (STEP23C_METADATA, "Step23C metadata"),
        (STEP23E_MORPH, "Step23E morphology evidence"),
        (STEP23E_CLAIMS, "Step23E claim boundary"),
        (STEP23E_CEILING, "Step23E claim ceiling"),
        (STEP23E_METADATA, "Step23E metadata"),
    ]

    for path, label in required:
        require_file(path, label)

    cmeta = load_json(STEP23C_METADATA)
    emeta = load_json(STEP23E_METADATA)

    pc6_direction = normalize_direction(
        emeta.get("pc6_frozen_direction", "")
    )

    if pc6_direction != FROZEN_PC6_DIRECTION:
        raise RuntimeError(
            "Step23E PC6 direction does not match frozen authority.\n"
            f"Expected={FROZEN_PC6_DIRECTION}\n"
            f"Observed={pc6_direction}"
        )

    if str(emeta.get("pc6_role", "")) != "SECONDARY_INTERPRETED_AXIS":
        raise RuntimeError(
            "Step23E no longer freezes PC6 as SECONDARY_INTERPRETED_AXIS."
        )

    # Step23C must remain internal-only.
    ctext = json.dumps(cmeta).lower()

    for forbidden in [
        '"aggc_files_read": true',
        '"peso_files_read": true',
        '"tcga_files_read": true',
    ]:
        if forbidden in ctext:
            raise RuntimeError(
                "Step23C provenance indicates external data were read. "
                "Step24A refuses to proceed."
            )

    # Verify internal results support the pre-specified external directions.
    primary = pd.read_csv(STEP23C_PRIMARY, low_memory=False)
    secondary = pd.read_csv(STEP23C_SECONDARY, low_memory=False)

    all_results = pd.concat(
        [primary.assign(_source="PRIMARY"), secondary.assign(_source="SECONDARY")],
        ignore_index=True,
        sort=False,
    )

    feature_col = None
    for c in ["feature", "metric", "morphology_metric"]:
        if c in all_results.columns:
            feature_col = c
            break

    if feature_col is None:
        raise RuntimeError(
            "Could not locate feature column in Step23C results."
        )

    effect_col = None

    for c in [
        "median_paired_difference_high_minus_low",
        "estimate",
        "effect",
        "median_effect",
    ]:
        if c in all_results.columns:
            effect_col = c
            break

    if effect_col is None:
        raise RuntimeError(
            "Could not locate effect estimate column in Step23C results."
        )

    verification_rows = []

    for endpoint in FROZEN_TRANSFER_ENDPOINTS:

        metric = endpoint["metric"]

        sub = all_results.loc[
            all_results[feature_col].astype(str).eq(metric)
        ].copy()

        if len(sub) != 1:
            raise RuntimeError(
                f"Step23C must contain exactly one frozen row for {metric}; "
                f"observed={len(sub)}"
            )

        estimate = float(
            pd.to_numeric(
                sub.iloc[0][effect_col],
                errors="raise",
            )
        )

        observed_sign = (
            +1
            if estimate > 0
            else -1
            if estimate < 0
            else 0
        )

        expected_sign = int(
            endpoint["expected_sign"]
        )

        if observed_sign != expected_sign:
            raise RuntimeError(
                "Attempted Step24A endpoint direction is not supported by "
                "the frozen Step23C result.\n"
                f"Metric={metric}\n"
                f"Expected sign={expected_sign}\n"
                f"Observed estimate={estimate}"
            )

        verification_rows.append(
            {
                "metric": metric,
                "step23c_source_family": sub.iloc[0]["_source"],
                "step23c_internal_estimate": estimate,
                "frozen_external_expected_sign": expected_sign,
                "direction_match": True,
                "used_to_select_from_aggc": False,
            }
        )

    # NPI must remain not-confirmed / non-promotable.
    claims = pd.read_csv(
        STEP23E_CLAIMS,
        low_memory=False,
    )

    claim_text = (
        STEP23E_CEILING.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        + "\n"
        + claims.to_csv(index=False)
    ).lower()

    if (
        "npi"
        not in claim_text
        or
        "not"
        not in claim_text
    ):
        raise RuntimeError(
            "Step23E claim package does not clearly preserve the NPI "
            "non-confirmed boundary."
        )

    return {
        "step23c_metadata": cmeta,
        "step23e_metadata": emeta,
        "verification": pd.DataFrame(verification_rows),
    }


# =============================================================================
# AGGC STATIC PROVENANCE / HEADER ONLY
# =============================================================================

def validate_aggc_static_inputs() -> Dict[str, Any]:

    require_dir(
        AGGC_ROOT,
        "AGGC root",
    )

    require_dir(
        AGGC_EMBED_ROOT,
        "AGGC Virchow2 embedding root",
    )

    require_file(
        AGGC_MORPH_SOURCE,
        "AGGC unified tile morphology source",
    )

    # HEADER ONLY. This must not read AGGC PC6/morphology rows.
    header = pd.read_csv(
        AGGC_MORPH_SOURCE,
        nrows=0,
    )

    cols = header.columns.astype(str).tolist()

    missing = [
        c
        for c in REQUIRED_SOURCE_COLUMNS
        if c not in cols
    ]

    if missing:
        raise RuntimeError(
            "AGGC morphology source lacks required frozen columns:\n"
            + "\n".join(missing)
        )

    # Inventory filenames only; no embedding arrays are opened.
    embedding_files = sorted(
        p
        for p in AGGC_EMBED_ROOT.iterdir()
        if p.is_file()
    )

    train_names = [
        p.name
        for p in embedding_files
        if "Subset1_Train_" in p.name
    ]

    test_names = [
        p.name
        for p in embedding_files
        if "Subset1_Test_" in p.name
    ]

    return {
        "source_columns": cols,
        "optional_columns_present": [
            c for c in OPTIONAL_PROVENANCE_COLUMNS if c in cols
        ],
        "embedding_file_count": len(embedding_files),
        "train_embedding_filename_count": len(train_names),
        "test_embedding_filename_count": len(test_names),
        "aggc_rows_read": 0,
    }


# =============================================================================
# FREEZE TABLES
# =============================================================================

def endpoint_table() -> pd.DataFrame:

    rows = []

    for rank, endpoint in enumerate(
        FROZEN_TRANSFER_ENDPOINTS,
        start=1,
    ):
        row = dict(endpoint)

        row.update(
            {
                "endpoint_order": rank,
                "contrast": "PC6_HIGH_MINUS_PC6_LOW",
                "primary_partition": PRIMARY_PARTITION,
                "secondary_partition": SECONDARY_PARTITION,
                "selection_from_aggc_outcome": False,
                "formal_external_endpoint": True,
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def specificity_table() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                **SPECIFICITY_REFERENCE,
                "contrast": "PC6_HIGH_MINUS_PC6_LOW",
                "selection_from_aggc_outcome": False,
                "formal_transfer_endpoint": False,
                "inferential_role": "SPECIFICITY_REFERENCE_ONLY",
            }
        ]
    )


def cohort_table() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "partition": PRIMARY_PARTITION,
                "formal_role": "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION",
                "primary_inference": True,
                "may_be_replaced_by_other_partition": False,
                "may_be_pooled_with_other_partition": False,
            },
            {
                "partition": SECONDARY_PARTITION,
                "formal_role": "SECONDARY_EXTERNAL_CHARACTERIZATION",
                "primary_inference": False,
                "may_be_replaced_by_other_partition": False,
                "may_be_pooled_with_other_partition": False,
            },
        ]
    )


def sampling_table() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "parameter": "low_quantile",
                "value": LOW_QUANTILE,
                "role": "FROZEN_PC6_LOW",
            },
            {
                "parameter": "high_quantile",
                "value": HIGH_QUANTILE,
                "role": "FROZEN_PC6_HIGH",
            },
            {
                "parameter": "min_eligible_tiles_per_slide",
                "value": MIN_ELIGIBLE_TILES_PER_SLIDE,
                "role": "PRIMARY_SLIDE_ELIGIBILITY",
            },
            {
                "parameter": "max_tiles_per_slide_per_stratum",
                "value": MAX_TILES_PER_SLIDE_PER_STRATUM,
                "role": "DETERMINISTIC_BALANCED_SAMPLING",
            },
            {
                "parameter": "sampling_seed",
                "value": SAMPLING_SEED,
                "role": "DETERMINISTIC_BALANCED_SAMPLING",
            },
            {
                "parameter": "bootstrap_resamples",
                "value": BOOTSTRAP_RESAMPLES,
                "role": "STEP24C_SLIDE_BOOTSTRAP",
            },
            {
                "parameter": "bootstrap_seed",
                "value": BOOTSTRAP_SEED,
                "role": "STEP24C_SLIDE_BOOTSTRAP",
            },
        ]
    )


def evaluation_table() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "endpoint": "metric_level_directional_replication",
                "role": "PRIMARY_METRIC_LEVEL",
                "definition": (
                    "sign(median slide-level PC6_HIGH-PC6_LOW effect) "
                    "matches the internally frozen expected sign"
                ),
                "partition": PRIMARY_PARTITION,
            },
            {
                "endpoint": "overall_direction_concordance_fraction",
                "role": "PRIMARY_FAMILY_LEVEL",
                "definition": (
                    "number of six frozen transfer endpoints with matching "
                    "direction divided by six"
                ),
                "partition": PRIMARY_PARTITION,
            },
            {
                "endpoint": "metric_level_bootstrap_ci",
                "role": "SUPPORTIVE",
                "definition": (
                    "95% slide-bootstrap CI of median slide-level "
                    "PC6_HIGH-PC6_LOW effect"
                ),
                "partition": PRIMARY_PARTITION,
            },
            {
                "endpoint": "train_directional_characterization",
                "role": "SECONDARY",
                "definition": (
                    "same frozen endpoint family evaluated descriptively "
                    "in Subset1_Train"
                ),
                "partition": SECONDARY_PARTITION,
            },
        ]
    )


def claim_ceiling_text() -> str:

    return """STEP24 AGGC PC6 MORPHOLOGY-TRANSFER CLAIM CEILING

ALLOWED IF SUPPORTED
--------------------
1. The internally defined PC6 morphology phenotype shows directional transfer
   to held-out AGGC Test.
2. Specific pre-frozen morphology directions replicate in AGGC Test.
3. AGGC provides independent external support for PC6 as a transferable
   secondary histologic representation axis.

NOT ALLOWED
-----------
1. AGGC defines or revises the PC6 phenotype.
2. AGGC selects the PC6 sign.
3. AGGC promotes NPI into the confirmed phenotype.
4. AGGC creates a new formal morphology endpoint after outcomes are inspected.
5. Train and Test may be pooled to rescue a weak Test result.
6. A favorable Train result may replace held-out Test as the primary result.
7. PC6 is a Gleason axis merely because AGGC contains Gleason annotations.
8. Morphology transfer alone establishes causal/statistical independence.
9. Morphology transfer alone establishes cross-foundation-model replication.
10. Tumor localization AUC is part of Step24; it belongs to a separately frozen
    future analysis.
"""


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Freeze AGGC external PC6 morphology-transfer specification "
            "before any AGGC PC6-morphology outcome evaluation."
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--skip-large-source-hash",
        action="store_true",
        help=(
            "Record size/mtime but do not SHA256 the large compressed AGGC "
            "morphology source. Scientific specification is unchanged."
        ),
    )

    return p.parse_args()


def main() -> None:

    args = parse_args()

    banner(
        "STEP 24A: FREEZE AGGC EXTERNAL PC6 MORPHOLOGY TRANSFER SPECIFICATION"
    )

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP23C]       {STEP23C}")
    print(f"[STEP23E]       {STEP23E}")
    print(f"[AGGC ROOT]     {AGGC_ROOT}")
    print(f"[MORPH SOURCE]  {AGGC_MORPH_SOURCE}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()

    print("[ROLE]          PRE-EXECUTION EXTERNAL FREEZE ONLY")
    print("[TARGET]        VIRCHOW2 PC6")
    print("[DIRECTION]     HIGH_IN_TUMOR / RAW PC6")
    print("[PRIMARY]       AGGC Subset1_Test")
    print("[SECONDARY]     AGGC Subset1_Train")
    print("[STATE MODEL]   ALL-TISSUE PC6 q20/q80; NOT G3/G4/G5")
    print("[AGGC ROWS]     NOT READ")
    print("[ASSOCIATION]   NONE")
    print("[BOOTSTRAP]     NONE")
    print("[P-VALUES]      NONE")
    print("[TUMOR AUC]     NONE")
    print("[PCA REFIT]     NONE")
    print("[PC SCREEN]     NONE")
    print("[SIGN TUNING]   NONE")

    prepare_output(
        args.output,
        args.overwrite,
    )

    # -------------------------------------------------------------------------
    # Internal authority.
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE FINAL INTERNAL PC6 PHENOTYPE AUTHORITY"
    )

    authority = validate_step23_authority()

    verification = authority[
        "verification"
    ]

    print(
        verification.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # AGGC static provenance only.
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE AGGC STATIC INPUT SCHEMA / PROVENANCE ONLY"
    )

    aggc = validate_aggc_static_inputs()

    print(
        f"AGGC source columns                 : {len(aggc['source_columns'])}"
    )

    print(
        f"Required frozen columns present     : YES"
    )

    print(
        f"Embedding files inventoried by name : {aggc['embedding_file_count']}"
    )

    print(
        f"Train filename matches              : {aggc['train_embedding_filename_count']}"
    )

    print(
        f"Test filename matches               : {aggc['test_embedding_filename_count']}"
    )

    print(
        f"AGGC outcome rows read              : {aggc['aggc_rows_read']}"
    )

    # -------------------------------------------------------------------------
    # Frozen contract.
    # -------------------------------------------------------------------------

    banner(
        "FROZEN AGGC PC6 MORPHOLOGY-TRANSFER CONTRACT"
    )

    endpoints = endpoint_table()

    specificity = specificity_table()

    cohorts = cohort_table()

    sampling = sampling_table()

    evaluation = evaluation_table()

    print(
        endpoints[
            [
                "endpoint_order",
                "metric",
                "expected_direction",
                "external_role",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Primary partition                    : {PRIMARY_PARTITION}"
    )

    print(
        f"Secondary partition                  : {SECONDARY_PARTITION}"
    )

    print(
        f"PC6 LOW                              : <= within-slide q20"
    )

    print(
        f"PC6 HIGH                             : >= within-slide q80"
    )

    print(
        f"Minimum eligible tiles / slide       : {MIN_ELIGIBLE_TILES_PER_SLIDE}"
    )

    print(
        f"Max sampled tiles / slide / stratum  : {MAX_TILES_PER_SLIDE_PER_STRATUM}"
    )

    print(
        f"NPI role                              : SPECIFICITY REFERENCE ONLY"
    )

    print(
        f"Primary family-level endpoint         : direction concordance / 6"
    )

    # -------------------------------------------------------------------------
    # Write specification.
    # -------------------------------------------------------------------------

    plan = {
        "version":
            VERSION,

        "schema":
            SCHEMA,

        "created_utc":
            utc_now(),

        "scientific_role":
            "pre_execution_aggc_external_pc6_morphology_transfer_freeze",

        "target_axis":
            "Virchow2_PC6",

        "frozen_pc6_direction":
            FROZEN_PC6_DIRECTION,

        "internal_authority":
            {
                "step23e":
                    str(
                        STEP23E
                    ),

                "step23c_verification_only":
                    str(
                        STEP23C
                    ),

                "aggc_may_redefine_internal_phenotype":
                    False,
            },

        "external_cohort":
            {
                "dataset":
                    "AGGC",

                "primary_partition":
                    PRIMARY_PARTITION,

                "secondary_partition":
                    SECONDARY_PARTITION,

                "pool_train_test":
                    False,

                "choose_best_partition":
                    False,
            },

        "pc6_strata":
            {
                "score":
                    "RAW_FROZEN_PC6",

                "low":
                    "within-slide <= q20",

                "high":
                    "within-slide >= q80",

                "effect":
                    "PC6_HIGH - PC6_LOW",

                "sign_flip":
                    False,
            },

        "eligibility":
            {
                "required_finite_fields":
                    [
                        "PC6"
                    ]
                    +
                    [
                        x[
                            "metric"
                        ]
                        for x in FROZEN_TRANSFER_ENDPOINTS
                    ],

                "min_eligible_tiles_per_slide":
                    MIN_ELIGIBLE_TILES_PER_SLIDE,

                "max_tiles_per_slide_per_stratum":
                    MAX_TILES_PER_SLIDE_PER_STRATUM,

                "sampling_seed":
                    SAMPLING_SEED,

                "patient_grouping_invented_from_slide_names":
                    False,
            },

        "formal_transfer_endpoints":
            FROZEN_TRANSFER_ENDPOINTS,

        "specificity_reference":
            SPECIFICITY_REFERENCE,

        "evaluation":
            {
                "primary_metric_level":
                    "directional_replication",

                "primary_family_level":
                    "overall_direction_concordance_fraction",

                "supportive":
                    "metric_level_slide_bootstrap_ci",

                "bootstrap_resamples":
                    BOOTSTRAP_RESAMPLES,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,

                "new_metric_selection_allowed":
                    False,

                "npi_promotion_allowed":
                    False,
            },

        "measurement_source":
            {
                "path":
                    str(
                        AGGC_MORPH_SOURCE
                    ),

                "step24a_rows_read":
                    0,

                "header_only_inspection":
                    True,

                "required_columns":
                    REQUIRED_SOURCE_COLUMNS,

                "normal_g3_g4_g5_used":
                    False,
            },

        "prohibited":
            {
                "aggc_outcome_read_in_step24a":
                    True,

                "aggc_pc6_quantile_calculation":
                    True,

                "aggc_morphology_effect_calculation":
                    True,

                "p_value":
                    True,

                "bootstrap":
                    True,

                "tumor_auc":
                    True,

                "pc_screen":
                    True,

                "pca_refit":
                    True,

                "sign_tuning":
                    True,

                "best_partition_selection":
                    True,

                "new_metric_selection":
                    True,

                "npi_promotion":
                    True,

                "gleason_redefinition":
                    True,

                "internal_phenotype_revision":
                    True,
            },

        "planned_steps":
            {
                "24B":
                    "AGGC PC6 morphology transfer measurement only",

                "24C":
                    "frozen AGGC PC6 morphology transfer evaluation",
            },
    }

    save_json(
        args.output
        /
        "aggc_pc6_external_morphology_transfer_plan.json",
        plan,
    )

    endpoints.to_csv(
        args.output
        /
        "aggc_pc6_frozen_transfer_endpoint_specification.csv",
        index=False,
    )

    specificity.to_csv(
        args.output
        /
        "aggc_pc6_specificity_reference_specification.csv",
        index=False,
    )

    cohorts.to_csv(
        args.output
        /
        "aggc_pc6_partition_role_specification.csv",
        index=False,
    )

    sampling.to_csv(
        args.output
        /
        "aggc_pc6_sampling_and_eligibility_specification.csv",
        index=False,
    )

    evaluation.to_csv(
        args.output
        /
        "aggc_pc6_external_evaluation_endpoint_specification.csv",
        index=False,
    )

    verification.to_csv(
        args.output
        /
        "internal_pc6_transfer_endpoint_verification.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "source_path":
                    str(
                        AGGC_MORPH_SOURCE
                    ),

                "file_exists":
                    AGGC_MORPH_SOURCE.is_file(),

                "file_size_bytes":
                    AGGC_MORPH_SOURCE.stat().st_size,

                "file_mtime_ns":
                    AGGC_MORPH_SOURCE.stat().st_mtime_ns,

                "sha256":
                    (
                        ""
                        if args.skip_large_source_hash
                        else
                        sha256_file(
                            AGGC_MORPH_SOURCE
                        )
                    ),

                "sha256_skipped":
                    bool(
                        args.skip_large_source_hash
                    ),

                "rows_read_in_step24a":
                    0,

                "header_columns":
                    "|".join(
                        aggc[
                            "source_columns"
                        ]
                    ),
            }
        ]
    ).to_csv(
        args.output
        /
        "aggc_pc6_measurement_source_snapshot.csv",
        index=False,
    )

    # Hash only consumed internal authority files + source specification.
    hash_rows = []

    for role, path in [
        (
            "step23c_primary",
            STEP23C_PRIMARY,
        ),
        (
            "step23c_secondary",
            STEP23C_SECONDARY,
        ),
        (
            "step23c_metadata",
            STEP23C_METADATA,
        ),
        (
            "step23e_morphology",
            STEP23E_MORPH,
        ),
        (
            "step23e_claim_boundary",
            STEP23E_CLAIMS,
        ),
        (
            "step23e_claim_ceiling",
            STEP23E_CEILING,
        ),
        (
            "step23e_metadata",
            STEP23E_METADATA,
        ),
    ]:

        hash_rows.append(
            {
                "role":
                    role,

                "path":
                    str(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    pd.DataFrame(
        hash_rows
    ).to_csv(
        args.output
        /
        "source_hashes.csv",
        index=False,
    )

    (
        args.output
        /
        "pc6_aggc_external_claim_ceiling.txt"
    ).write_text(
        claim_ceiling_text(),
        encoding="utf-8",
    )

    metadata = {
        "version":
            VERSION,

        "schema":
            SCHEMA,

        "created_utc":
            utc_now(),

        "python_version":
            platform.python_version(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "role":
            "PRE_EXECUTION_EXTERNAL_FREEZE_ONLY",

        "target":
            "Virchow2 PC6",

        "internal_authority":
            "STEP23E_FINAL_INTERNAL_PC6_FREEZE",

        "primary_external_partition":
            PRIMARY_PARTITION,

        "secondary_external_partition":
            SECONDARY_PARTITION,

        "n_formal_transfer_endpoints":
            len(
                FROZEN_TRANSFER_ENDPOINTS
            ),

        "formal_transfer_metrics":
            [
                x[
                    "metric"
                ]
                for x in FROZEN_TRANSFER_ENDPOINTS
            ],

        "specificity_reference":
            SPECIFICITY_REFERENCE,

        "aggc_static_input_audit":
            aggc,

        "freeze_only_audit":
            {
                "aggc_pc6_values_read":
                    False,

                "aggc_morphology_outcomes_read":
                    False,

                "aggc_rows_read":
                    0,

                "pc6_quantiles_calculated":
                    False,

                "morphology_effects_calculated":
                    False,

                "p_values_calculated":
                    False,

                "bootstrap_calculated":
                    False,

                "tumor_auc_calculated":
                    False,

                "pc_screening":
                    False,

                "pca_refit":
                    False,

                "sign_tuning":
                    False,

                "best_partition_selection":
                    False,

                "new_metric_selection":
                    False,

                "npi_promoted":
                    False,

                "gleason_states_used":
                    False,

                "internal_phenotype_revised":
                    False,

                "formal_endpoint_family_frozen":
                    True,
            },
    }

    save_json(
        args.output
        /
        "analysis_metadata.json",
        metadata,
    )

    readme = f"""STEP 24A
AGGC EXTERNAL PC6 MORPHOLOGY TRANSFER SPECIFICATION
===================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

Role
----
PRE-EXECUTION FREEZE ONLY.

Internal authority
------------------
{STEP23E}

PC6
---
Frozen raw PC6 direction:
  {FROZEN_PC6_DIRECTION}

External state definition
-------------------------
Within each AGGC slide:

  PC6_LOW  <= q20
  PC6_HIGH >= q80

Effect:
  PC6_HIGH - PC6_LOW

No Gleason-state definition is used.

Formal endpoint family
----------------------
SEI_eosin_stroma       : POSITIVE
GOI_lumen_openness     : NEGATIVE
CLR_nuclear_to_lumen   : POSITIVE
lumen_area             : NEGATIVE
mean_lumen_area        : NEGATIVE
lumen_fragmentation    : POSITIVE

NPI_nuclear_packing
-------------------
Specificity reference only.
It is NOT a confirmed primary PC6 phenotype and cannot be promoted from AGGC.

Partition hierarchy
-------------------
PRIMARY:
  {PRIMARY_PARTITION}
  held-out external replication

SECONDARY:
  {SECONDARY_PARTITION}
  characterization only

Train and Test are not pooled.

Eligibility / sampling
----------------------
Minimum eligible tiles per slide:
  {MIN_ELIGIBLE_TILES_PER_SLIDE}

Max tiles per slide per PC6 stratum:
  {MAX_TILES_PER_SLIDE_PER_STRATUM}

Sampling seed:
  {SAMPLING_SEED}

Formal transfer summary
-----------------------
Metric level:
  frozen-direction match

Family level:
  direction-concordance fraction across all six frozen transfer endpoints

Supportive:
  slide-level bootstrap confidence intervals

Measurement source
------------------
{AGGC_MORPH_SOURCE}

Step24A reads its header only.
No AGGC outcome rows are read.

Forbidden in Step24A
--------------------
AGGC PC6 values
AGGC morphology outcome values
PC6 quantiles
morphology effects
p-values
bootstrap
tumor AUC
PC screening
PCA refit
sign tuning
best-partition selection
new morphology metric selection
NPI promotion
Gleason-based PC6 redefinition

Next
----
24B:
  measurement only

24C:
  frozen external morphology-transfer evaluation
"""

    (
        args.output
        /
        "README.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Audit.
    # -------------------------------------------------------------------------

    banner(
        "STEP24A FREEZE-ONLY AUDIT"
    )

    audit = [
        (
            "Step23E final internal authority inherited",
            "YES",
        ),
        (
            "Step23C directions verified before AGGC outcome read",
            "YES",
        ),
        (
            "PC6 frozen direction retained",
            "YES",
        ),
        (
            "AGGC outcome rows read",
            "NO",
        ),
        (
            "AGGC PC6 values read",
            "NO",
        ),
        (
            "AGGC morphology outcome values read",
            "NO",
        ),
        (
            "AGGC PC6 quantiles calculated",
            "NO",
        ),
        (
            "AGGC morphology effects calculated",
            "NO",
        ),
        (
            "AGGC p-values calculated",
            "NO",
        ),
        (
            "AGGC bootstrap calculated",
            "NO",
        ),
        (
            "AGGC tumor AUC calculated",
            "NO",
        ),
        (
            "PC1-PC8 screening",
            "NO",
        ),
        (
            "PCA refit",
            "NO",
        ),
        (
            "PC6 sign tuning",
            "NO",
        ),
        (
            "Best AGGC partition selected",
            "NO",
        ),
        (
            "Train/Test pooling allowed",
            "NO",
        ),
        (
            "New morphology metric selected",
            "NO",
        ),
        (
            "NPI promoted",
            "NO",
        ),
        (
            "Normal/G3/G4/G5 used",
            "NO",
        ),
        (
            "Internal PC6 phenotype revised",
            "NO",
        ),
        (
            "Six transfer endpoints frozen",
            "YES",
        ),
        (
            "AGGC Test primary role frozen",
            "YES",
        ),
    ]

    for item, status in audit:
        print(
            f"{item:<62}: {status}"
        )

    print()
    print("[OK] Saved:")

    for path in sorted(
        args.output.iterdir()
    ):
        print(
            f"  {path}"
        )

    print()
    print(
        "[DONE] Step24A AGGC external PC6 morphology-transfer specification frozen."
    )

    print(
        "[NEXT] Step24B may read AGGC PC6/morphology rows for MEASUREMENT ONLY "
        "under this frozen contract. Do not calculate transfer inference until Step24C."
    )


if __name__ == "__main__":
    main()
