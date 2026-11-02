#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
21b_freeze_aggc_pc3_tumor_localization_auc_specification.py

STEP 21B
========
FREEZE AGGC PC3 TUMOR-LOCALIZATION AUC SPECIFICATION

VERSION
-------
1.0.0

SCHEMA
------
21b-aggc-pc3-tumor-localization-auc-freeze-schema-1.0

ROLE
----
PRE-OUTCOME FORMAL AUC SPECIFICATION FREEZE ONLY.

PURPOSE
-------
Freeze the formal AGGC PC3 tumor-localization AUC analysis BEFORE any new
PC3 AUC values are calculated.

SCIENTIFIC CONTEXT
------------------
PC3 was learned and frozen in the independent discovery cohort.

AGGC dataset partitions are therefore NOT model-training/testing partitions
for PC3. Their scientific roles here are:

    Subset1_Train -> AGGC characterization partition
    Subset1_Test  -> AGGC held-out replication partition

HISTORICAL ENDPOINT PROVENANCE
------------------------------
Step21A recovered the historical AGGC binary tumor-AUC definition from source:

    tumor-positive classes:
        G3, G4, G5

    non-tumor classes:
        Normal, Stroma

    minimum per-slide support:
        >=10 tumor tiles
        >=10 non-tumor tiles

    historical AUC analysis:
        per-slide

    historical screening endpoint:
        orientation-free AUC
        max(AUC(score), AUC(-score))

The historical orientation-free endpoint was suitable for multi-PC screening,
but it is NOT the formal primary endpoint here because the PC3 biological
direction has already been frozen upstream.

PRIMARY FORMAL ENDPOINT
-----------------------
Frozen biological direction authority:
    Step06 frozen PC3 direction

Expected frozen direction:
    low_in_tumor

Therefore the formal score is:
    -PC3

Per-slide primary endpoint:
    ROC AUC(tumor_label, -PC3)

Interpretation:
    AUC > 0.5 means lower frozen PC3 scores preferentially localize to
    tumor-associated histology.

PRIMARY CLASSES
---------------
Positive:
    G3 + G4 + G5

Negative:
    Normal + Stroma

Unlabeled / unassigned:
    EXCLUDED FROM BINARY AUC

No complement class is inferred.
No missing class is synthesized.

PER-SLIDE ELIGIBILITY
---------------------
A slide is primary-AUC evaluable only if:

    n_tumor_tiles >= 10
    AND
    n_non_tumor_tiles >= 10

These thresholds are inherited from the historical AGGC AUC source audit.

PRIMARY REPORTING
-----------------
For each AGGC partition separately:

    per-slide AUC(-PC3)

Across eligible slides:
    median AUC
    mean AUC
    interquartile range
    95% bootstrap CI for median

Characterization:
    descriptive / supportive

Held-out replication:
    formal external replication

No PC selection or direction selection is allowed from AGGC outcomes.

SUPPORTIVE HISTORICAL CONTINUITY ENDPOINT
-----------------------------------------
A secondary supportive endpoint may reproduce the historical orientation-free
discriminability:

    AUC_orientation_free =
        max(AUC(PC3), AUC(-PC3))

This supportive endpoint MUST NOT:
    - replace the primary fixed-direction AUC;
    - determine PC3 sign;
    - select PCs;
    - be called confirmatory directionality evidence.

STRICT FIREWALL
---------------
Step21B MUST NOT:
    - read historical AUC result values;
    - calculate PC3 AUC;
    - inspect Test AUC;
    - rank PC1-PC8;
    - screen PCs;
    - refit PCA;
    - flip PC3 sign from AGGC data;
    - change tumor/non-tumor classes;
    - change support thresholds;
    - select a best partition;
    - tune a threshold based on outcome.

OUTPUTS
-------
aggc_pc3_tumor_auc_analysis_plan.json
aggc_pc3_tumor_auc_class_specification.csv
aggc_pc3_tumor_auc_threshold_specification.csv
aggc_pc3_tumor_auc_partition_roles.csv
aggc_pc3_tumor_auc_endpoint_specification.csv
pc3_frozen_direction_snapshot.csv
historical_auc_definition_snapshot.csv
source_hashes.csv
analysis_metadata.json
README.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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

from typing import Any, Dict, List

import pandas as pd


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "21b-aggc-pc3-tumor-localization-auc-freeze-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

MEDIA = REPO_ROOT

RESULTS = RESULTS_ROOT

STEP06_DIR = (
    RESULTS
    / "06_frozen_pc_direction_specification"
)

STEP21A_DIR = (
    RESULTS
    / "21a_qc_aggc_historical_tumor_auc_definition"
)

OUT_DIR = (
    RESULTS
    / "21b_aggc_pc3_tumor_localization_auc_specification"
)

STEP21A_IDENTITY = (
    STEP21A_DIR
    / "historical_auc_script_identity.csv"
)

STEP21A_CONSTANTS = (
    STEP21A_DIR
    / "historical_auc_constants.csv"
)

STEP21A_ENDPOINT = (
    STEP21A_DIR
    / "historical_auc_endpoint_definition_audit.csv"
)

STEP21A_METADATA = (
    STEP21A_DIR
    / "analysis_metadata.json"
)


# =============================================================================
# FROZEN SCIENTIFIC SPECIFICATION
# =============================================================================

POSITIVE_CLASSES = [
    "G3",
    "G4",
    "G5",
]

NEGATIVE_CLASSES = [
    "Normal",
    "Stroma",
]

EXCLUDED_CLASSES = [
    "unlabeled",
    "unassigned",
    "background",
]

MIN_POSITIVE_TILES_PER_SLIDE = 10
MIN_NEGATIVE_TILES_PER_SLIDE = 10

PRIMARY_SCORE = "-PC3"
PRIMARY_ENDPOINT = "roc_auc_score(tumor_label, -PC3)"

SUPPORTIVE_HISTORICAL_ENDPOINT = (
    "max(roc_auc_score(tumor_label, PC3), "
    "roc_auc_score(tumor_label, -PC3))"
)

EXPECTED_FROZEN_PC3_DIRECTION = "low_in_tumor"

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260808

PARTITION_ROLES = {
    "train_validation":
        "AGGC_CHARACTERIZATION_PARTITION",

    "test_replication":
        "AGGC_HELD_OUT_REPLICATION_PARTITION",
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
    width: int = 124,
) -> None:

    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.is_file():

        raise FileNotFoundError(
            f"Missing required {label}:\n{path}"
        )


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def load_json(
    path: Path,
) -> Dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def save_json(
    path: Path,
    obj: Any,
) -> None:

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


def prepare_output(
    path: Path,
    overwrite: bool,
) -> None:

    if path.exists() and any(
        path.iterdir()
    ):

        if not overwrite:

            raise RuntimeError(
                f"Output directory is non-empty:\n{path}\n"
                "Use --overwrite only for an intentional pre-outcome rerun."
            )

        shutil.rmtree(
            path
        )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalize_text(
    x: Any,
) -> str:

    return str(
        x
    ).strip().lower()


# =============================================================================
# STEP06 PC3 DIRECTION RESOLUTION
# =============================================================================

def resolve_step06_pc3_direction(
    step06_dir: Path,
) -> Dict[str, Any]:
    """
    Resolve the frozen PC3 biological direction conservatively from Step06.

    The function searches CSV/JSON files in Step06, but does NOT inspect any
    AGGC AUC outcome.
    """

    if not step06_dir.is_dir():

        raise FileNotFoundError(
            f"Step06 directory not found:\n{step06_dir}"
        )

    candidates: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # CSVs
    # -------------------------------------------------------------------------

    for path in sorted(
        step06_dir.glob(
            "*.csv"
        )
    ):

        try:

            df = pd.read_csv(
                path,
                low_memory=False,
            )

        except Exception:

            continue

        if df.empty:

            continue

        # Component columns
        component_cols = [
            c
            for c in df.columns
            if normalize_text(c)
            in {
                "component",
                "pc",
                "principal_component",
                "pc_name",
            }
        ]

        direction_cols = [
            c
            for c in df.columns
            if (
                "direction"
                in normalize_text(c)
                or
                "tumor"
                in normalize_text(c)
                or
                "sign"
                in normalize_text(c)
            )
        ]

        for component_col in component_cols:

            pc3_rows = df.loc[
                df[
                    component_col
                ].astype(
                    str
                ).str.upper().eq(
                    "PC3"
                )
            ]

            if pc3_rows.empty:

                continue

            for direction_col in direction_cols:

                for value in pc3_rows[
                    direction_col
                ].dropna():

                    text = str(
                        value
                    ).strip()

                    if text:

                        candidates.append(
                            {
                                "source_path":
                                    str(
                                        path
                                    ),

                                "source_type":
                                    "CSV",

                                "component_column":
                                    component_col,

                                "direction_field":
                                    direction_col,

                                "raw_value":
                                    text,
                            }
                        )

    # -------------------------------------------------------------------------
    # JSONs
    # -------------------------------------------------------------------------

    for path in sorted(
        step06_dir.glob(
            "*.json"
        )
    ):

        try:

            obj = load_json(
                path
            )

        except Exception:

            continue

        serialized = json.dumps(
            obj,
            ensure_ascii=False,
        )

        low = serialized.lower()

        if "pc3" not in low:

            continue

        # Search only explicit expected phrase first.
        if EXPECTED_FROZEN_PC3_DIRECTION in low:

            candidates.append(
                {
                    "source_path":
                        str(
                            path
                        ),

                    "source_type":
                        "JSON",

                    "component_column":
                        "serialized_json",

                    "direction_field":
                        "serialized_json",

                    "raw_value":
                        EXPECTED_FROZEN_PC3_DIRECTION,
                }
            )

    # -------------------------------------------------------------------------
    # Strict resolution
    # -------------------------------------------------------------------------

    matching = [
        row
        for row in candidates
        if EXPECTED_FROZEN_PC3_DIRECTION
        in normalize_text(
            row[
                "raw_value"
            ]
        )
    ]

    if not matching:

        # Do not infer direction from numerical signs.
        preview = (
            pd.DataFrame(
                candidates
            ).to_string(
                index=False
            )
            if candidates
            else
            "No candidate direction rows found."
        )

        raise RuntimeError(
            "Could not explicitly verify Step06 frozen PC3 direction "
            f"as '{EXPECTED_FROZEN_PC3_DIRECTION}'.\n\n"
            "Refusing to freeze Step21B from inferred numerical signs.\n\n"
            f"Candidates:\n{preview}"
        )

    # Multiple matching sources are fine if they agree.
    return {
        "component":
            "PC3",

        "frozen_direction":
            EXPECTED_FROZEN_PC3_DIRECTION,

        "direction_authority":
            "STEP06_DISCOVERY_FROZEN_DIRECTION",

        "n_matching_sources":
            len(
                matching
            ),

        "matching_sources":
            matching,
    }


# =============================================================================
# STEP21A HISTORICAL DEFINITION VALIDATION
# =============================================================================

def validate_step21a(
    step21a_dir: Path,
) -> Dict[str, Any]:

    identity = (
        step21a_dir
        /
        "historical_auc_script_identity.csv"
    )

    constants = (
        step21a_dir
        /
        "historical_auc_constants.csv"
    )

    endpoint = (
        step21a_dir
        /
        "historical_auc_endpoint_definition_audit.csv"
    )

    metadata = (
        step21a_dir
        /
        "analysis_metadata.json"
    )

    for path, label in [
        (
            identity,
            "Step21A historical script identity",
        ),
        (
            constants,
            "Step21A constants",
        ),
        (
            endpoint,
            "Step21A endpoint audit",
        ),
        (
            metadata,
            "Step21A metadata",
        ),
    ]:

        require_file(
            path,
            label,
        )

    meta = load_json(
        metadata
    )

    if str(
        meta.get(
            "version",
            "",
        )
    ) != "1.0.0":

        raise RuntimeError(
            "Step21B expects Step21A v1.0.0."
        )

    guardrails = meta.get(
        "guardrails",
        {}
    )

    forbidden_true = [
        "historical_auc_result_csv_read",
        "historical_pc3_auc_value_read",
        "auc_calculated",
        "pc_ranking_performed",
        "test_auc_outcome_read",
        "tumor_endpoint_redefined",
        "threshold_tuned",
        "pc3_sign_flipped",
        "pca_refit",
        "pc_screening",
        "inference_performed",
    ]

    violations = [
        key
        for key in forbidden_true
        if bool(
            guardrails.get(
                key,
                False,
            )
        )
    ]

    if violations:

        raise RuntimeError(
            "Step21A firewall violation(s): "
            +
            ", ".join(
                violations
            )
        )

    constants_df = pd.read_csv(
        constants,
        low_memory=False,
    )

    if not {
        "name",
        "literal_value",
    }.issubset(
        constants_df.columns
    ):

        raise RuntimeError(
            "Step21A constants table lacks name/literal_value."
        )

    constant_map = {
        str(
            row[
                "name"
            ]
        ):
            str(
                row[
                    "literal_value"
                ]
            )
        for _, row in constants_df.iterrows()
    }

    # -------------------------------------------------------------------------
    # Historical support thresholds
    # -------------------------------------------------------------------------

    pos_raw = constant_map.get(
        "MIN_POSITIVE_TILES_PER_SLIDE",
        "",
    )

    neg_raw = constant_map.get(
        "MIN_NEGATIVE_TILES_PER_SLIDE",
        "",
    )

    try:

        pos_min = int(
            float(
                pos_raw
            )
        )

        neg_min = int(
            float(
                neg_raw
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not recover historical minimum positive/negative tile "
            "support from Step21A constants."
        ) from exc

    if pos_min != MIN_POSITIVE_TILES_PER_SLIDE:

        raise RuntimeError(
            f"Historical positive support differs from frozen expectation: "
            f"{pos_min} != {MIN_POSITIVE_TILES_PER_SLIDE}"
        )

    if neg_min != MIN_NEGATIVE_TILES_PER_SLIDE:

        raise RuntimeError(
            f"Historical negative support differs from frozen expectation: "
            f"{neg_min} != {MIN_NEGATIVE_TILES_PER_SLIDE}"
        )

    # -------------------------------------------------------------------------
    # Historical class definitions
    # -------------------------------------------------------------------------

    tumor_raw = constant_map.get(
        "TUMOR_CLASSES",
        "",
    )

    non_tumor_raw = constant_map.get(
        "NON_TUMOR_CLASSES",
        "",
    )

    for required_class in POSITIVE_CLASSES:

        if required_class not in tumor_raw:

            raise RuntimeError(
                "Historical TUMOR_CLASSES does not explicitly contain "
                f"{required_class}: {tumor_raw}"
            )

    for required_class in NEGATIVE_CLASSES:

        if required_class not in non_tumor_raw:

            raise RuntimeError(
                "Historical NON_TUMOR_CLASSES does not explicitly contain "
                f"{required_class}: {non_tumor_raw}"
            )

    # -------------------------------------------------------------------------
    # Historical identity
    # -------------------------------------------------------------------------

    identity_df = pd.read_csv(
        identity,
        low_memory=False,
    )

    if len(
        identity_df
    ) != 1:

        raise RuntimeError(
            "Step21A historical script identity is not unique."
        )

    historical_script = str(
        identity_df[
            "historical_script"
        ].iloc[0]
    )

    historical_script_sha256 = str(
        identity_df[
            "sha256"
        ].iloc[0]
    )

    return {
        "historical_script":
            historical_script,

        "historical_script_sha256":
            historical_script_sha256,

        "tumor_classes":
            POSITIVE_CLASSES,

        "non_tumor_classes":
            NEGATIVE_CLASSES,

        "min_positive_tiles_per_slide":
            pos_min,

        "min_negative_tiles_per_slide":
            neg_min,

        "historical_auc_unit":
            "PER_SLIDE",

        "historical_orientation_handling":
            "ORIENTATION_FREE_MAX_RAW_VS_NEGATED_SCORE",

        "historical_primary_context":
            "MULTI_PC_EXPLORATORY_SCREENING",
    }


# =============================================================================
# BUILD FROZEN TABLES
# =============================================================================

def build_class_specification() -> pd.DataFrame:

    rows = []

    for cls in [
        "Normal",
        "Stroma",
        "G3",
        "G4",
        "G5",
        "unlabeled",
    ]:

        if cls in POSITIVE_CLASSES:

            binary_role = "POSITIVE_TUMOR"

            binary_label = 1

            included = True

        elif cls in NEGATIVE_CLASSES:

            binary_role = "NEGATIVE_NON_TUMOR"

            binary_label = 0

            included = True

        else:

            binary_role = "EXCLUDED_FROM_BINARY_AUC"

            binary_label = ""

            included = False

        rows.append(
            {
                "histologic_class":
                    cls,

                "binary_auc_role":
                    binary_role,

                "binary_label":
                    binary_label,

                "included_in_primary_auc":
                    included,

                "definition_source":
                    (
                        "historical AGGC source recovered by Step21A; "
                        "frozen unchanged in Step21B"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_threshold_specification() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "threshold_name":
                    "minimum_tumor_tiles_per_slide",

                "threshold_value":
                    MIN_POSITIVE_TILES_PER_SLIDE,

                "threshold_role":
                    "PRIMARY_EVALUABILITY",

                "source":
                    "historical AGGC AUC definition",

                "outcome_selected":
                    False,
            },
            {
                "threshold_name":
                    "minimum_non_tumor_tiles_per_slide",

                "threshold_value":
                    MIN_NEGATIVE_TILES_PER_SLIDE,

                "threshold_role":
                    "PRIMARY_EVALUABILITY",

                "source":
                    "historical AGGC AUC definition",

                "outcome_selected":
                    False,
            },
        ]
    )


def build_partition_roles() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "partition":
                    "train_validation",

                "original_dataset_name":
                    "Subset1_Train",

                "scientific_role":
                    "AGGC_CHARACTERIZATION_PARTITION",

                "used_to_train_pc3":
                    False,

                "used_to_refit_pca":
                    False,

                "formal_replication":
                    False,
            },
            {
                "partition":
                    "test_replication",

                "original_dataset_name":
                    "Subset1_Test",

                "scientific_role":
                    "AGGC_HELD_OUT_REPLICATION_PARTITION",

                "used_to_train_pc3":
                    False,

                "used_to_refit_pca":
                    False,

                "formal_replication":
                    True,
            },
        ]
    )


def build_endpoint_specification() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "endpoint_id":
                    "PRIMARY_FIXED_DIRECTION_PC3_TUMOR_AUC",

                "endpoint_role":
                    "PRIMARY",

                "score":
                    "-PC3",

                "positive_class":
                    "G3|G4|G5",

                "negative_class":
                    "Normal|Stroma",

                "unit":
                    "slide",

                "definition":
                    PRIMARY_ENDPOINT,

                "direction_selected_from_aggc":
                    False,

                "pc_screening":
                    False,

                "interpretation":
                    (
                        "AUC > 0.5 indicates that lower frozen PC3 scores "
                        "preferentially localize to tumor-associated histology."
                    ),
            },
            {
                "endpoint_id":
                    "SUPPORTIVE_HISTORICAL_ORIENTATION_FREE_PC3_AUC",

                "endpoint_role":
                    "SUPPORTIVE_HISTORICAL_CONTINUITY",

                "score":
                    "PC3_and_-PC3",

                "positive_class":
                    "G3|G4|G5",

                "negative_class":
                    "Normal|Stroma",

                "unit":
                    "slide",

                "definition":
                    SUPPORTIVE_HISTORICAL_ENDPOINT,

                "direction_selected_from_aggc":
                    True,

                "pc_screening":
                    False,

                "interpretation":
                    (
                        "Orientation-free discriminability only; not evidence "
                        "for biological PC3 direction and not permitted to "
                        "replace the primary fixed-direction endpoint."
                    ),
            },
        ]
    )


# =============================================================================
# README
# =============================================================================

def build_readme(
    direction: Dict[str, Any],
    historical: Dict[str, Any],
) -> str:

    return f"""STEP 21B
FREEZE AGGC PC3 TUMOR-LOCALIZATION AUC SPECIFICATION

Version
-------
{VERSION}

Schema
------
{SCHEMA}


ROLE
----
PRE-OUTCOME FORMAL SPECIFICATION FREEZE ONLY.


PC3 ORIGIN
----------
PC3 was learned and frozen in the independent discovery cohort.

AGGC Subset1_Train is therefore NOT a model-training set.


AGGC PARTITION ROLES
--------------------
Subset1_Train:
    AGGC characterization partition

Subset1_Test:
    AGGC held-out replication partition


HISTORICAL AGGC AUC DEFINITION RECOVERED BY STEP21A
---------------------------------------------------
Tumor-positive:
    G3 + G4 + G5

Non-tumor negative:
    Normal + Stroma

Minimum tumor tiles per slide:
    {MIN_POSITIVE_TILES_PER_SLIDE}

Minimum non-tumor tiles per slide:
    {MIN_NEGATIVE_TILES_PER_SLIDE}

Historical analysis unit:
    per slide

Historical screening orientation handling:
    max(AUC(score), AUC(-score))


FORMAL PRIMARY ENDPOINT
-----------------------
Frozen PC3 biological direction:
    {direction['frozen_direction']}

Direction authority:
    Step06 discovery-frozen direction

Score:
    -PC3

Endpoint:
    ROC AUC(tumor_label, -PC3)

Interpretation:
    AUC > 0.5 means lower frozen PC3 values preferentially localize to
    tumor-associated histology.


PRIMARY BINARY CLASSES
----------------------
Positive:
    G3
    G4
    G5

Negative:
    Normal
    Stroma

Unlabeled:
    excluded


PRIMARY SLIDE EVALUABILITY
--------------------------
>= {MIN_POSITIVE_TILES_PER_SLIDE} tumor tiles
AND
>= {MIN_NEGATIVE_TILES_PER_SLIDE} non-tumor tiles


PRIMARY PARTITION-LEVEL SUMMARY
-------------------------------
Per partition:

    median per-slide AUC
    mean per-slide AUC
    IQR
    95% bootstrap CI for median

Bootstrap:
    {BOOTSTRAP_RESAMPLES} slide-level resamples

Seed:
    {BOOTSTRAP_SEED}


SUPPORTIVE HISTORICAL CONTINUITY ENDPOINT
-----------------------------------------
Orientation-free PC3 discriminability:

    max(AUC(PC3), AUC(-PC3))

This endpoint is supportive only.

It MUST NOT:
    select PC3 sign
    replace the primary fixed-direction AUC
    be called confirmatory directionality evidence


HISTORICAL SCRIPT
-----------------
{historical['historical_script']}

SHA256:
{historical['historical_script_sha256']}


STEP21B FIREWALL
----------------
Historical AUC values read:
    NO

New AUC calculated:
    NO

Test AUC outcome read:
    NO

PC1-PC8 ranking:
    NO

PC screening:
    NO

PCA refit:
    NO

AGGC-driven PC3 sign flip:
    NO

Tumor class redefinition:
    NO

Non-tumor class redefinition:
    NO

Support threshold tuning:
    NO

Best partition selection:
    NO


NEXT
----
Step21C may execute the frozen PC3 AUC analysis.

Step21C must not redefine:
    positive class
    negative class
    unlabeled handling
    per-slide unit
    >=10 / >=10 support
    fixed-direction score -PC3
    AGGC partition roles
    primary versus supportive endpoint roles
"""


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(
        formatter_class=
            argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--step06-dir",
        type=Path,
        default=STEP06_DIR,
    )

    p.add_argument(
        "--step21a-dir",
        type=Path,
        default=STEP21A_DIR,
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

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    banner(
        "STEP 21B: FREEZE AGGC PC3 TUMOR-LOCALIZATION AUC SPECIFICATION"
    )

    print(
        f"[VERSION]       {VERSION}"
    )

    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    print(
        f"[STEP06]        {args.step06_dir}"
    )

    print(
        f"[STEP21A]       {args.step21a_dir}"
    )

    print(
        f"[OUTPUT]        {args.output}"
    )

    print(
        f"[OVERWRITE]     {args.overwrite}"
    )

    print()

    print(
        "[ROLE]          PRE-OUTCOME FORMAL AUC FREEZE ONLY"
    )

    print(
        "[PRIMARY SCORE]  -PC3"
    )

    print(
        "[POSITIVE]       G3 + G4 + G5"
    )

    print(
        "[NEGATIVE]       Normal + Stroma"
    )

    print(
        "[MIN SUPPORT]    >=10 tumor AND >=10 non-tumor tiles/slide"
    )

    print(
        "[AUC CALC]       NONE"
    )

    print(
        "[TEST OUTCOME]   NOT READ"
    )

    print(
        "[PC SCREEN]      NONE"
    )

    print(
        "[PCA REFIT]      NONE"
    )

    print(
        "[SIGN TUNING]    NONE"
    )

    prepare_output(
        args.output,
        args.overwrite,
    )

    # -------------------------------------------------------------------------
    # Validate Step21A
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE HISTORICAL AUC DEFINITION AUTHORITY"
    )

    historical = validate_step21a(
        args.step21a_dir
    )

    print(
        "Historical tumor classes G3/G4/G5      : PASS"
    )

    print(
        "Historical non-tumor Normal/Stroma     : PASS"
    )

    print(
        "Historical min tumor support = 10      : PASS"
    )

    print(
        "Historical min non-tumor support = 10  : PASS"
    )

    print(
        "Historical analysis unit = per-slide   : PASS"
    )

    print(
        "Historical screening = orientation-free: PASS"
    )

    # -------------------------------------------------------------------------
    # Validate Step06 direction
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE DISCOVERY-FROZEN PC3 DIRECTION"
    )

    direction = resolve_step06_pc3_direction(
        args.step06_dir
    )

    print(
        f"Frozen PC3 direction : {direction['frozen_direction']}"
    )

    print(
        f"Direction authority  : {direction['direction_authority']}"
    )

    print(
        "Direction selected from AGGC outcome : NO"
    )

    # -------------------------------------------------------------------------
    # Build frozen specifications
    # -------------------------------------------------------------------------

    class_df = build_class_specification()

    threshold_df = build_threshold_specification()

    partition_df = build_partition_roles()

    endpoint_df = build_endpoint_specification()

    direction_df = pd.DataFrame(
        [
            {
                "component":
                    "PC3",

                "frozen_direction":
                    direction[
                        "frozen_direction"
                    ],

                "formal_auc_score":
                    "-PC3",

                "direction_authority":
                    direction[
                        "direction_authority"
                    ],

                "direction_selected_from_aggc":
                    False,
            }
        ]
    )

    historical_df = pd.DataFrame(
        [
            {
                "historical_script":
                    historical[
                        "historical_script"
                    ],

                "historical_script_sha256":
                    historical[
                        "historical_script_sha256"
                    ],

                "tumor_classes":
                    "|".join(
                        POSITIVE_CLASSES
                    ),

                "non_tumor_classes":
                    "|".join(
                        NEGATIVE_CLASSES
                    ),

                "minimum_positive_tiles_per_slide":
                    MIN_POSITIVE_TILES_PER_SLIDE,

                "minimum_negative_tiles_per_slide":
                    MIN_NEGATIVE_TILES_PER_SLIDE,

                "analysis_unit":
                    "slide",

                "orientation_handling":
                    (
                        "max(AUC(score), AUC(-score))"
                    ),

                "historical_context":
                    (
                        "multi-PC exploratory screening"
                    ),
            }
        ]
    )

    plan = {
        "version":
            VERSION,

        "schema":
            SCHEMA,

        "analysis_role":
            "pre_outcome_formal_auc_specification_freeze",

        "pc3": {
            "component":
                "PC3",

            "origin":
                "independent discovery cohort",

            "frozen_direction":
                direction[
                    "frozen_direction"
                ],

            "direction_authority":
                "Step06",

            "formal_auc_score":
                "-PC3",

            "aggc_sign_selection_allowed":
                False,

            "pca_refit_allowed":
                False,

            "pc_screening_allowed":
                False,
        },

        "binary_endpoint": {
            "positive_classes":
                POSITIVE_CLASSES,

            "negative_classes":
                NEGATIVE_CLASSES,

            "unlabeled_handling":
                "exclude",

            "binary_positive_label":
                1,

            "binary_negative_label":
                0,
        },

        "slide_evaluability": {
            "minimum_positive_tiles":
                MIN_POSITIVE_TILES_PER_SLIDE,

            "minimum_negative_tiles":
                MIN_NEGATIVE_TILES_PER_SLIDE,

            "rule":
                (
                    "n_tumor_tiles >= 10 AND "
                    "n_non_tumor_tiles >= 10"
                ),

            "outcome_based_exclusion_allowed":
                False,
        },

        "primary_endpoint": {
            "endpoint_id":
                "PRIMARY_FIXED_DIRECTION_PC3_TUMOR_AUC",

            "score":
                "-PC3",

            "definition":
                PRIMARY_ENDPOINT,

            "unit":
                "slide",

            "direction_selected_from_aggc":
                False,
        },

        "supportive_historical_endpoint": {
            "endpoint_id":
                "SUPPORTIVE_HISTORICAL_ORIENTATION_FREE_PC3_AUC",

            "definition":
                SUPPORTIVE_HISTORICAL_ENDPOINT,

            "role":
                "supportive historical continuity only",

            "may_replace_primary":
                False,

            "may_determine_pc3_direction":
                False,
        },

        "partition_roles":
            PARTITION_ROLES,

        "partition_summary": {
            "independent_unit":
                "slide",

            "statistics": [
                "n_evaluable_slides",
                "median_auc",
                "mean_auc",
                "q25_auc",
                "q75_auc",
                "bootstrap_95_ci_for_median",
            ],

            "bootstrap_resamples":
                BOOTSTRAP_RESAMPLES,

            "bootstrap_seed":
                BOOTSTRAP_SEED,
        },

        "forbidden": {
            "historical_auc_result_read":
                True,

            "pc1_pc8_ranking":
                True,

            "pc_screening":
                True,

            "pca_refit":
                True,

            "aggc_driven_pc3_sign_flip":
                True,

            "tumor_class_redefinition":
                True,

            "negative_class_redefinition":
                True,

            "support_threshold_tuning":
                True,

            "best_partition_selection":
                True,

            "outcome_based_slide_exclusion":
                True,
        },
    }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    p_plan = (
        args.output
        /
        "aggc_pc3_tumor_auc_analysis_plan.json"
    )

    p_classes = (
        args.output
        /
        "aggc_pc3_tumor_auc_class_specification.csv"
    )

    p_thresholds = (
        args.output
        /
        "aggc_pc3_tumor_auc_threshold_specification.csv"
    )

    p_partitions = (
        args.output
        /
        "aggc_pc3_tumor_auc_partition_roles.csv"
    )

    p_endpoints = (
        args.output
        /
        "aggc_pc3_tumor_auc_endpoint_specification.csv"
    )

    p_direction = (
        args.output
        /
        "pc3_frozen_direction_snapshot.csv"
    )

    p_historical = (
        args.output
        /
        "historical_auc_definition_snapshot.csv"
    )

    p_hashes = (
        args.output
        /
        "source_hashes.csv"
    )

    p_metadata = (
        args.output
        /
        "analysis_metadata.json"
    )

    p_readme = (
        args.output
        /
        "README.txt"
    )

    save_json(
        p_plan,
        plan,
    )

    class_df.to_csv(
        p_classes,
        index=False,
    )

    threshold_df.to_csv(
        p_thresholds,
        index=False,
    )

    partition_df.to_csv(
        p_partitions,
        index=False,
    )

    endpoint_df.to_csv(
        p_endpoints,
        index=False,
    )

    direction_df.to_csv(
        p_direction,
        index=False,
    )

    historical_df.to_csv(
        p_historical,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Source hashes
    # -------------------------------------------------------------------------

    hash_rows = []

    hash_sources = [
        (
            "step21a_historical_script_identity",
            args.step21a_dir
            /
            "historical_auc_script_identity.csv",
        ),
        (
            "step21a_historical_constants",
            args.step21a_dir
            /
            "historical_auc_constants.csv",
        ),
        (
            "step21a_endpoint_audit",
            args.step21a_dir
            /
            "historical_auc_endpoint_definition_audit.csv",
        ),
        (
            "step21a_metadata",
            args.step21a_dir
            /
            "analysis_metadata.json",
        ),
    ]

    for role, path in hash_sources:

        require_file(
            path,
            role,
        )

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

    # Capture matching Step06 sources used for direction authority.
    for i, match in enumerate(
        direction[
            "matching_sources"
        ],
        start=1,
    ):

        path = Path(
            match[
                "source_path"
            ]
        )

        if path.is_file():

            hash_rows.append(
                {
                    "role":
                        f"step06_pc3_direction_source_{i}",

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
        p_hashes,
        index=False,
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

        "scientific_role":
            "pre_outcome_formal_auc_specification_freeze",

        "pc3_origin":
            "independent discovery cohort",

        "aggc_train_role":
            "characterization_partition_not_model_training",

        "aggc_test_role":
            "held_out_replication_partition",

        "frozen_pc3_direction":
            direction[
                "frozen_direction"
            ],

        "formal_primary_score":
            "-PC3",

        "positive_classes":
            POSITIVE_CLASSES,

        "negative_classes":
            NEGATIVE_CLASSES,

        "minimum_positive_tiles_per_slide":
            MIN_POSITIVE_TILES_PER_SLIDE,

        "minimum_negative_tiles_per_slide":
            MIN_NEGATIVE_TILES_PER_SLIDE,

        "bootstrap_resamples":
            BOOTSTRAP_RESAMPLES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "historical_orientation_free_endpoint":
            SUPPORTIVE_HISTORICAL_ENDPOINT,

        "guardrails": {
            "historical_auc_result_values_read":
                False,

            "new_auc_calculated":
                False,

            "test_auc_outcome_read":
                False,

            "pc1_pc8_ranked":
                False,

            "pc_screening":
                False,

            "pca_refit":
                False,

            "aggc_driven_pc3_sign_flip":
                False,

            "tumor_endpoint_redefined":
                False,

            "negative_class_redefined":
                False,

            "support_threshold_tuned":
                False,

            "best_partition_selected":
                False,

            "outcome_based_slide_exclusion":
                False,
        },
    }

    save_json(
        p_metadata,
        metadata,
    )

    p_readme.write_text(
        build_readme(
            direction,
            historical,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    banner(
        "FROZEN FORMAL PC3 TUMOR-AUC CONTRACT"
    )

    print(
        f"PC3 origin                        : independent discovery cohort"
    )

    print(
        f"Frozen PC3 direction              : {direction['frozen_direction']}"
    )

    print(
        f"Formal AUC score                  : -PC3"
    )

    print(
        f"Positive classes                  : {POSITIVE_CLASSES}"
    )

    print(
        f"Negative classes                  : {NEGATIVE_CLASSES}"
    )

    print(
        f"Unlabeled                         : EXCLUDED"
    )

    print(
        f"Minimum tumor tiles / slide       : {MIN_POSITIVE_TILES_PER_SLIDE}"
    )

    print(
        f"Minimum non-tumor tiles / slide   : {MIN_NEGATIVE_TILES_PER_SLIDE}"
    )

    print(
        f"Primary endpoint                  : {PRIMARY_ENDPOINT}"
    )

    print(
        "Historical orientation-free AUC   : SUPPORTIVE ONLY"
    )

    print(
        f"Bootstrap resamples               : {BOOTSTRAP_RESAMPLES}"
    )

    print(
        f"Bootstrap seed                    : {BOOTSTRAP_SEED}"
    )

    print()
    print(
        "AGGC Subset1_Train role           : CHARACTERIZATION"
    )

    print(
        "AGGC Subset1_Test role            : HELD-OUT REPLICATION"
    )

    print(
        "AGGC Train used to train PC3      : NO"
    )

    banner(
        "FREEZE-ONLY AUDIT"
    )

    for item, status in [
        (
            "Historical AUC result values read",
            "NO",
        ),
        (
            "New PC3 AUC calculated",
            "NO",
        ),
        (
            "Held-out Test AUC inspected",
            "NO",
        ),
        (
            "PC1-PC8 ranking",
            "NO",
        ),
        (
            "PC screening",
            "NO",
        ),
        (
            "PCA refit",
            "NO",
        ),
        (
            "AGGC-driven PC3 sign flip",
            "NO",
        ),
        (
            "Tumor classes changed",
            "NO",
        ),
        (
            "Non-tumor classes changed",
            "NO",
        ),
        (
            "Minimum tile support changed",
            "NO",
        ),
        (
            "Best partition selected",
            "NO",
        ),
        (
            "Outcome-based slide exclusion",
            "NO",
        ),
        (
            "Primary endpoint frozen",
            "YES",
        ),
        (
            "Supportive historical endpoint frozen",
            "YES",
        ),
    ]:

        print(
            f"{item:46s}: {status}"
        )

    print()
    print(
        "[OK] Saved:"
    )

    for path in [
        p_plan,
        p_classes,
        p_thresholds,
        p_partitions,
        p_endpoints,
        p_direction,
        p_historical,
        p_hashes,
        p_metadata,
        p_readme,
    ]:

        print(
            f"  {path}"
        )

    print()
    print(
        "[DONE] Step21B AGGC PC3 tumor-localization AUC specification frozen."
    )

    print(
        "[NEXT] Step21C may execute the frozen per-slide AUC(-PC3) analysis. "
        "Do not redefine classes, support, direction, partition roles, or "
        "primary/supportive endpoint hierarchy."
    )


if __name__ == "__main__":
    main()
