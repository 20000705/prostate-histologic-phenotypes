#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
21c_evaluate_frozen_aggc_pc3_tumor_localization_auc.py

STEP 21C
========
EVALUATE FROZEN AGGC PC3 TUMOR-LOCALIZATION AUC

VERSION
-------
1.1.0

SCHEMA
------
21c-aggc-pc3-tumor-localization-auc-evaluation-schema-1.1

ROLE
----
FORMAL FROZEN AUC EXECUTION.

This is the first Step21 script allowed to calculate the formal AGGC PC3
tumor-localization AUC under the Step21B frozen contract.

FROZEN PRIMARY ENDPOINT
-----------------------
Positive classes:
    G3 + G4 + G5

Negative classes:
    Normal + Stroma

Excluded:
    unlabeled / unassigned / background / any class not explicitly frozen

Per-slide eligibility:
    >=10 tumor tiles
    AND
    >=10 non-tumor tiles

Frozen PC3 biological direction:
    low_in_tumor

Formal score:
    -PC3

Primary per-slide endpoint:
    roc_auc_score(tumor_label, -PC3)

Interpretation:
    AUC > 0.5 means lower frozen PC3 scores preferentially localize to
    tumor-associated histology.

SUPPORTIVE HISTORICAL CONTINUITY ENDPOINT
-----------------------------------------
For descriptive continuity with the historical multi-PC screening script:

    orientation_free_auc =
        max(
            roc_auc_score(tumor_label, PC3),
            roc_auc_score(tumor_label, -PC3)
        )

This endpoint is SUPPORTIVE ONLY.
It may not replace the fixed-direction primary endpoint and may not determine
PC3 sign.

AGGC PARTITION ROLES
--------------------
Subset1_Train:
    AGGC characterization partition

Subset1_Test:
    AGGC held-out replication partition

PC3 was NOT trained or refit on either AGGC partition.

PARTITION-LEVEL SUMMARY
-----------------------
For each partition separately:
    n_total_slides
    n_evaluable_slides
    median primary AUC
    mean primary AUC
    Q25
    Q75
    fraction AUC > 0.5
    fraction AUC >= 0.7
    fraction AUC >= 0.8
    95% bootstrap CI of median
    10,000 slide-level bootstrap resamples
    seed = 20260808

The same descriptive summary is also produced for the supportive
orientation-free endpoint.

STRICT FIREWALL
---------------
Step21C MUST NOT:
    - redefine positive/negative classes;
    - redefine support thresholds;
    - select PC3 sign from AGGC outcomes;
    - rank PC1-PC8;
    - screen PCs;
    - refit PCA;
    - select a best partition;
    - exclude slides based on AUC result;
    - alter the frozen tile-label definition;
    - alter PC3 values;
    - pool tiles across slides for inferential/summary reporting.

INPUT STRATEGY — v1.1.0
-----------------------------
Scientific endpoint authority:
    Step21B

Held-out Test tile authority:
    Step21B2
    results/21b2_complete_aggc_test_auc_input/
        aggc_subset1_test_pc3_auc_input.csv.gz

Held-out Test cohort authority:
    Step21B3
    results/21b3_frozen_aggc_test_auc_execution_cohort/
        frozen_aggc_test_auc_execution_cohort.csv
        frozen_aggc_test_auc_contract.json

Step21C requires the Step21B2 Test tile source and Step21B3 frozen cohort to
agree EXACTLY on slide IDs and per-slide clean/tumor/non-tumor support before
any AUC is calculated.

The raw Subset1 Test universe contains 45 slides, but Step21B3 froze the formal
execution cohort at 35 slides because 10 slides lacked the historical
Normal/Stroma fraction provenance required to reconstruct the frozen historical
five-class endpoint. Those exclusions were frozen before AUC outcomes and may
not be changed here.

For characterization, Step21C retains the historical Train per-slide morphology
tables under the AGGC morphology root unless an explicit combined Train source
is supplied.

Step20C morphology transfer is NOT an authority for the Test AUC cohort in
v1.1.0.

OUTPUTS
-------
aggc_pc3_tumor_auc_per_slide.csv
aggc_pc3_tumor_auc_partition_summary.csv
aggc_pc3_tumor_auc_supportive_orientation_free_summary.csv
aggc_pc3_tumor_auc_evaluability_qc.csv
aggc_pc3_tumor_auc_input_coverage.csv
analysis_audit.csv
source_hashes.csv
analysis_metadata.json
README.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception as exc:
    raise RuntimeError(
        "Step21C requires scikit-learn (sklearn.metrics.roc_auc_score)."
    ) from exc


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.1.0"
SCHEMA = "21c-aggc-pc3-tumor-localization-auc-evaluation-schema-1.1"


# =============================================================================
# PATHS
# =============================================================================

MEDIA = REPO_ROOT

RESULTS = RESULTS_ROOT

AGGC = CONFIG_AGGC_ROOT

STEP21B_DIR = (
    RESULTS
    / "21b_aggc_pc3_tumor_localization_auc_specification"
)

DEFAULT_TRAIN_MORPHOLOGY_ROOT = (
    AGGC
    / "aggc_pc3_pc6_morphology"
)

STEP21B2_DIR = (
    RESULTS
    / "21b2_complete_aggc_test_auc_input"
)

STEP21B3_DIR = (
    RESULTS
    / "21b3_frozen_aggc_test_auc_execution_cohort"
)

DEFAULT_TEST_TILE_SOURCE = (
    STEP21B2_DIR
    / "aggc_subset1_test_pc3_auc_input.csv.gz"
)

STEP21B3_COHORT = (
    STEP21B3_DIR
    / "frozen_aggc_test_auc_execution_cohort.csv"
)

STEP21B3_EXCLUDED = (
    STEP21B3_DIR
    / "frozen_aggc_test_auc_excluded_slides.csv"
)

STEP21B3_SUMMARY = (
    STEP21B3_DIR
    / "frozen_aggc_test_auc_cohort_summary.csv"
)

STEP21B3_CONTRACT = (
    STEP21B3_DIR
    / "frozen_aggc_test_auc_contract.json"
)

STEP21B3_METADATA_HASHES = (
    STEP21B3_DIR
    / "source_hashes.csv"
)

OUT_DIR = (
    RESULTS
    / "21c_aggc_pc3_tumor_localization_auc"
)


# =============================================================================
# STEP21B FILES
# =============================================================================

STEP21B_PLAN = (
    STEP21B_DIR
    / "aggc_pc3_tumor_auc_analysis_plan.json"
)

STEP21B_CLASSES = (
    STEP21B_DIR
    / "aggc_pc3_tumor_auc_class_specification.csv"
)

STEP21B_THRESHOLDS = (
    STEP21B_DIR
    / "aggc_pc3_tumor_auc_threshold_specification.csv"
)

STEP21B_PARTITIONS = (
    STEP21B_DIR
    / "aggc_pc3_tumor_auc_partition_roles.csv"
)

STEP21B_ENDPOINTS = (
    STEP21B_DIR
    / "aggc_pc3_tumor_auc_endpoint_specification.csv"
)

STEP21B_DIRECTION = (
    STEP21B_DIR
    / "pc3_frozen_direction_snapshot.csv"
)

STEP21B_METADATA = (
    STEP21B_DIR
    / "analysis_metadata.json"
)


# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

POSITIVE_CLASSES = {"G3", "G4", "G5"}
NEGATIVE_CLASSES = {"Normal", "Stroma"}

MIN_POSITIVE_TILES_PER_SLIDE = 10
MIN_NEGATIVE_TILES_PER_SLIDE = 10

PRIMARY_SCORE_NAME = "-PC3"
EXPECTED_FROZEN_DIRECTION = "low_in_tumor"

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260808

PARTITION_ROLE_MAP = {
    "train_validation": "AGGC_CHARACTERIZATION_PARTITION",
    "test_replication": "AGGC_HELD_OUT_REPLICATION_PARTITION",
}


# =============================================================================
# OUTPUT FILENAMES
# =============================================================================

OUT_PER_SLIDE = "aggc_pc3_tumor_auc_per_slide.csv"
OUT_PRIMARY_SUMMARY = "aggc_pc3_tumor_auc_partition_summary.csv"
OUT_SUPPORTIVE_SUMMARY = "aggc_pc3_tumor_auc_supportive_orientation_free_summary.csv"
OUT_EVALUABILITY = "aggc_pc3_tumor_auc_evaluability_qc.csv"
OUT_COVERAGE = "aggc_pc3_tumor_auc_input_coverage.csv"
OUT_AUDIT = "analysis_audit.csv"
OUT_HASHES = "source_hashes.csv"
OUT_METADATA = "analysis_metadata.json"
OUT_README = "README.txt"


# =============================================================================
# HELPERS
# =============================================================================

def banner(title: str, width: int = 124) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required {label}:\n{path}"
        )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )


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
                "Use --overwrite only for an intentional exact rerun."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False

    raise ValueError(f"Cannot interpret boolean: {value!r}")


def finite_array(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


# =============================================================================
# VALIDATE STEP21B CONTRACT
# =============================================================================

def validate_step21b(step21b_dir: Path) -> Dict[str, Any]:

    global STEP21B_PLAN
    global STEP21B_CLASSES
    global STEP21B_THRESHOLDS
    global STEP21B_PARTITIONS
    global STEP21B_ENDPOINTS
    global STEP21B_DIRECTION
    global STEP21B_METADATA

    STEP21B_PLAN = (
        step21b_dir
        / "aggc_pc3_tumor_auc_analysis_plan.json"
    )

    STEP21B_CLASSES = (
        step21b_dir
        / "aggc_pc3_tumor_auc_class_specification.csv"
    )

    STEP21B_THRESHOLDS = (
        step21b_dir
        / "aggc_pc3_tumor_auc_threshold_specification.csv"
    )

    STEP21B_PARTITIONS = (
        step21b_dir
        / "aggc_pc3_tumor_auc_partition_roles.csv"
    )

    STEP21B_ENDPOINTS = (
        step21b_dir
        / "aggc_pc3_tumor_auc_endpoint_specification.csv"
    )

    STEP21B_DIRECTION = (
        step21b_dir
        / "pc3_frozen_direction_snapshot.csv"
    )

    STEP21B_METADATA = (
        step21b_dir
        / "analysis_metadata.json"
    )

    for path, label in [
        (STEP21B_PLAN, "Step21B analysis plan"),
        (STEP21B_CLASSES, "Step21B class specification"),
        (STEP21B_THRESHOLDS, "Step21B threshold specification"),
        (STEP21B_PARTITIONS, "Step21B partition roles"),
        (STEP21B_ENDPOINTS, "Step21B endpoint specification"),
        (STEP21B_DIRECTION, "Step21B PC3 direction snapshot"),
        (STEP21B_METADATA, "Step21B metadata"),
    ]:
        require_file(path, label)

    metadata = load_json(STEP21B_METADATA)

    if str(metadata.get("version", "")) != "1.0.0":
        raise RuntimeError(
            "Step21C requires Step21B v1.0.0."
        )

    if str(metadata.get("schema", "")) != (
        "21b-aggc-pc3-tumor-localization-auc-freeze-schema-1.0"
    ):
        raise RuntimeError(
            "Unexpected Step21B schema."
        )

    if str(metadata.get("frozen_pc3_direction", "")) != EXPECTED_FROZEN_DIRECTION:
        raise RuntimeError(
            "Step21B frozen PC3 direction is not low_in_tumor."
        )

    if str(metadata.get("formal_primary_score", "")) != "-PC3":
        raise RuntimeError(
            "Step21B formal primary score is not -PC3."
        )

    pos = set(metadata.get("positive_classes", []))
    neg = set(metadata.get("negative_classes", []))

    if pos != POSITIVE_CLASSES:
        raise RuntimeError(
            f"Step21B positive classes differ from frozen set: {pos}"
        )

    if neg != NEGATIVE_CLASSES:
        raise RuntimeError(
            f"Step21B negative classes differ from frozen set: {neg}"
        )

    if int(metadata.get("minimum_positive_tiles_per_slide", -1)) != MIN_POSITIVE_TILES_PER_SLIDE:
        raise RuntimeError(
            "Step21B minimum positive support is not 10."
        )

    if int(metadata.get("minimum_negative_tiles_per_slide", -1)) != MIN_NEGATIVE_TILES_PER_SLIDE:
        raise RuntimeError(
            "Step21B minimum negative support is not 10."
        )

    if int(metadata.get("bootstrap_resamples", -1)) != BOOTSTRAP_RESAMPLES:
        raise RuntimeError(
            "Step21B bootstrap resamples are not 10,000."
        )

    if int(metadata.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED:
        raise RuntimeError(
            "Step21B bootstrap seed differs from 20260808."
        )

    # Explicit endpoint table check.
    endpoints = pd.read_csv(STEP21B_ENDPOINTS)

    primary = endpoints.loc[
        endpoints["endpoint_role"].astype(str).eq("PRIMARY")
    ]

    if len(primary) != 1:
        raise RuntimeError(
            "Step21B does not contain exactly one PRIMARY endpoint."
        )

    if str(primary["score"].iloc[0]) != "-PC3":
        raise RuntimeError(
            "Step21B primary endpoint score is not -PC3."
        )

    # Direction snapshot.
    direction = pd.read_csv(STEP21B_DIRECTION)

    pc3 = direction.loc[
        direction["component"].astype(str).eq("PC3")
    ]

    if len(pc3) != 1:
        raise RuntimeError(
            "Step21B PC3 direction snapshot does not contain exactly one PC3 row."
        )

    if str(pc3["frozen_direction"].iloc[0]) != EXPECTED_FROZEN_DIRECTION:
        raise RuntimeError(
            "Step21B direction snapshot is not low_in_tumor."
        )

    if normalize_bool(pc3["direction_selected_from_aggc"].iloc[0]):
        raise RuntimeError(
            "Step21B unexpectedly indicates AGGC-selected direction."
        )

    return {
        "metadata": metadata,
        "positive_classes": POSITIVE_CLASSES,
        "negative_classes": NEGATIVE_CLASSES,
        "frozen_direction": EXPECTED_FROZEN_DIRECTION,
    }



# =============================================================================
# VALIDATE STEP21B3 TEST EXECUTION COHORT
# =============================================================================

def validate_step21b3(
    step21b3_dir: Path,
    test_tile_source: Path,
) -> Dict[str, Any]:

    cohort_path = (
        step21b3_dir
        / "frozen_aggc_test_auc_execution_cohort.csv"
    )

    excluded_path = (
        step21b3_dir
        / "frozen_aggc_test_auc_excluded_slides.csv"
    )

    summary_path = (
        step21b3_dir
        / "frozen_aggc_test_auc_cohort_summary.csv"
    )

    contract_path = (
        step21b3_dir
        / "frozen_aggc_test_auc_contract.json"
    )

    hashes_path = (
        step21b3_dir
        / "source_hashes.csv"
    )

    for path, label in [
        (cohort_path, "Step21B3 frozen execution cohort"),
        (excluded_path, "Step21B3 excluded-slide manifest"),
        (summary_path, "Step21B3 cohort summary"),
        (contract_path, "Step21B3 frozen contract"),
        (hashes_path, "Step21B3 source hashes"),
        (test_tile_source, "Step21B2 frozen Test tile source"),
    ]:
        require_file(path, label)

    contract = load_json(contract_path)

    if str(contract.get("version", "")) != "1.0.0":
        raise RuntimeError(
            "Step21C v1.1.0 requires Step21B3 v1.0.0."
        )

    if str(contract.get("schema", "")) != (
        "21b3-aggc-subset1-test-pc3-auc-execution-cohort-freeze-schema-1.0"
    ):
        raise RuntimeError(
            "Unexpected Step21B3 schema."
        )

    guardrails = contract.get("guardrails", {})

    for key in [
        "auc_calculated",
        "auc_outcome_read",
        "bootstrap_performed",
        "p_values_calculated",
        "pc_ranking",
        "pc_screening",
        "pca_refit",
        "pca_reprojection",
        "pc3_sign_flip",
        "label_generation",
        "normal_stroma_regeneration",
        "threshold_tuning",
        "manual_slide_selection",
        "outcome_based_exclusion",
    ]:
        if normalize_bool(
            guardrails.get(key, False)
        ):
            raise RuntimeError(
                f"Step21B3 guardrail violation: {key}=True"
            )

    frozen = contract.get(
        "frozen_execution_cohort",
        {},
    )

    if int(frozen.get("n_slides", -1)) != 35:
        raise RuntimeError(
            "Step21B3 frozen execution cohort is not 35 slides."
        )

    if int(frozen.get("minimum_tumor_tiles", -1)) != (
        MIN_POSITIVE_TILES_PER_SLIDE
    ):
        raise RuntimeError(
            "Step21B3 tumor support threshold differs from Step21B."
        )

    if int(frozen.get("minimum_nontumor_tiles", -1)) != (
        MIN_NEGATIVE_TILES_PER_SLIDE
    ):
        raise RuntimeError(
            "Step21B3 non-tumor support threshold differs from Step21B."
        )

    if str(frozen.get("primary_score", "")) != "-PC3":
        raise RuntimeError(
            "Step21B3 primary score differs from Step21B."
        )

    if set(frozen.get("positive_classes", [])) != POSITIVE_CLASSES:
        raise RuntimeError(
            "Step21B3 positive classes differ from Step21B."
        )

    if set(frozen.get("negative_classes", [])) != NEGATIVE_CLASSES:
        raise RuntimeError(
            "Step21B3 negative classes differ from Step21B."
        )

    cohort = pd.read_csv(
        cohort_path,
        low_memory=False,
    )

    required = {
        "slide_id",
        "n_clean_five_class_tiles",
        "n_tumor_tiles",
        "n_nontumor_tiles",
        "minimum_support_10_10_met",
        "cohort_status",
    }

    missing = required - set(cohort.columns)

    if missing:
        raise RuntimeError(
            "Step21B3 cohort manifest missing columns: "
            f"{sorted(missing)}"
        )

    if len(cohort) != 35:
        raise RuntimeError(
            f"Step21B3 cohort manifest contains {len(cohort)} rows, not 35."
        )

    if cohort["slide_id"].duplicated().any():
        raise RuntimeError(
            "Step21B3 cohort manifest contains duplicate slide IDs."
        )

    if not cohort["minimum_support_10_10_met"].map(
        normalize_bool
    ).all():
        raise RuntimeError(
            "Step21B3 frozen cohort contains a slide failing 10/10 support."
        )

    expected_ids = set(
        cohort["slide_id"].astype(str)
    )

    contract_ids = set(
        str(x)
        for x in frozen.get("slide_ids", [])
    )

    if contract_ids != expected_ids:
        raise RuntimeError(
            "Step21B3 cohort CSV and JSON contract slide sets differ."
        )

    # Require Step21C to consume exactly the Step21B2 tile source frozen by B3.
    frozen_tile_source = Path(
        str(
            contract.get(
                "frozen_input_source_for_step21c",
                "",
            )
        )
    )

    if not frozen_tile_source:
        raise RuntimeError(
            "Step21B3 contract lacks frozen_input_source_for_step21c."
        )

    if (
        test_tile_source.resolve()
        !=
        frozen_tile_source.resolve()
    ):
        raise RuntimeError(
            "Test tile source does not equal the Step21B3-frozen "
            "Step21B2 source.\n"
            f"Requested: {test_tile_source}\n"
            f"Frozen:    {frozen_tile_source}"
        )

    return {
        "cohort": cohort,
        "contract": contract,
        "expected_slide_ids": expected_ids,
        "paths": {
            "cohort": cohort_path,
            "excluded": excluded_path,
            "summary": summary_path,
            "contract": contract_path,
            "hashes": hashes_path,
        },
    }


def validate_test_tiles_against_step21b3(
    test_tiles: pd.DataFrame,
    b3: Dict[str, Any],
) -> None:

    cohort = b3["cohort"]

    observed_ids = set(
        test_tiles[
            "slide_id"
        ].astype(str).unique()
    )

    expected_ids = set(
        b3[
            "expected_slide_ids"
        ]
    )

    if observed_ids != expected_ids:
        missing = sorted(
            expected_ids - observed_ids
        )
        extra = sorted(
            observed_ids - expected_ids
        )

        raise RuntimeError(
            "Step21B2 Test tile source slide set does not exactly match "
            "the Step21B3 frozen execution cohort.\n"
            f"Missing frozen slides: {missing}\n"
            f"Unexpected slides: {extra}"
        )

    # Recompute class support directly from the exact tile source, BEFORE AUC.
    classified = classify_auc_rows(
        test_tiles
    )

    support = (
        classified.loc[
            classified[
                "binary_auc_role"
            ].isin(
                [
                    "POSITIVE_TUMOR",
                    "NEGATIVE_NON_TUMOR",
                ]
            )
        ]
        .groupby(
            "slide_id",
            as_index=False,
        )
        .agg(
            n_clean_five_class_tiles=(
                "binary_auc_role",
                "size",
            ),
            n_tumor_tiles=(
                "binary_label",
                lambda x: int(
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    ).eq(1).sum()
                ),
            ),
            n_nontumor_tiles=(
                "binary_label",
                lambda x: int(
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    ).eq(0).sum()
                ),
            ),
        )
    )

    frozen_counts = cohort[
        [
            "slide_id",
            "n_clean_five_class_tiles",
            "n_tumor_tiles",
            "n_nontumor_tiles",
        ]
    ].copy()

    merged = frozen_counts.merge(
        support,
        on="slide_id",
        how="outer",
        validate="one_to_one",
        suffixes=(
            "_frozen",
            "_source",
        ),
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise RuntimeError(
            "Step21B3/source per-slide support reconciliation failed."
        )

    for col in [
        "n_clean_five_class_tiles",
        "n_tumor_tiles",
        "n_nontumor_tiles",
    ]:

        left = pd.to_numeric(
            merged[
                f"{col}_frozen"
            ],
            errors="raise",
        ).astype(int)

        right = pd.to_numeric(
            merged[
                f"{col}_source"
            ],
            errors="raise",
        ).astype(int)

        if not left.eq(right).all():

            bad = merged.loc[
                ~left.eq(right),
                [
                    "slide_id",
                    f"{col}_frozen",
                    f"{col}_source",
                ],
            ]

            raise RuntimeError(
                f"Step21B3/source {col} mismatch.\n"
                + bad.to_string(
                    index=False
                )
            )

    if not (
        (
            support[
                "n_tumor_tiles"
            ]
            >=
            MIN_POSITIVE_TILES_PER_SLIDE
        )
        &
        (
            support[
                "n_nontumor_tiles"
            ]
            >=
            MIN_NEGATIVE_TILES_PER_SLIDE
        )
    ).all():
        raise RuntimeError(
            "A Step21B3 frozen slide no longer satisfies the frozen 10/10 "
            "support when recomputed from the Step21B2 tile source."
        )

# =============================================================================
# SOURCE RESOLUTION
# =============================================================================

def extract_slide_from_name(path: Path, prefix: str) -> Optional[str]:
    m = re.match(
        rf"({re.escape(prefix)}\d+)",
        path.name,
    )
    return m.group(1) if m else None


def discover_per_slide_train_files(root: Path) -> Dict[str, Path]:

    if not root.exists():
        return {}

    candidates = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        name = p.name

        if not name.startswith("Subset1_Train_"):
            continue

        if not (
            name.endswith(".csv")
            or name.endswith(".csv.gz")
        ):
            continue

        # Prefer morphology tables because they are known to contain PC3 and class.
        if "morphology" not in name.lower():
            continue

        candidates.append(p.resolve())

    by_slide: Dict[str, List[Path]] = {}

    for p in sorted(set(candidates)):
        sid = extract_slide_from_name(
            p,
            "Subset1_Train_",
        )

        if sid is None:
            continue

        by_slide.setdefault(sid, []).append(p)

    ambiguous = {
        sid: paths
        for sid, paths in by_slide.items()
        if len(paths) != 1
    }

    if ambiguous:
        lines = []

        for sid, paths in sorted(ambiguous.items()):
            lines.append(
                f"{sid}: " + " | ".join(str(p) for p in paths)
            )

        raise RuntimeError(
            "Ambiguous historical Train tile sources detected. "
            "Step21C will not auto-select:\n"
            + "\n".join(lines)
        )

    return {
        sid: paths[0]
        for sid, paths in by_slide.items()
    }


def load_combined_tile_source(
    path: Path,
    expected_partition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    require_file(
        path,
        f"{expected_partition} combined tile source",
    )

    header = pd.read_csv(
        path,
        nrows=0,
    )

    required = {
        "slide_id",
        "histologic_class",
        "PC3",
    }

    missing = required - set(header.columns)

    if missing:
        raise RuntimeError(
            f"Combined tile source lacks required columns {sorted(missing)}:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path,
        usecols=[
            "slide_id",
            "histologic_class",
            "PC3",
        ],
        low_memory=False,
    )

    df["slide_id"] = df["slide_id"].astype(str)
    df["histologic_class"] = df["histologic_class"].astype(str)
    df["PC3"] = pd.to_numeric(df["PC3"], errors="coerce")

    inventory = pd.DataFrame(
        [
            {
                "partition": expected_partition,
                "source_mode": "COMBINED_TILE_FILE",
                "slide_id": "",
                "path": str(path),
                "sha256": sha256_file(path),
                "n_rows": int(len(df)),
                "n_slides": int(df["slide_id"].nunique()),
            }
        ]
    )

    return df, inventory


def load_per_slide_sources(
    resolved: Dict[str, Path],
    expected_partition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if not resolved:
        raise RuntimeError(
            f"No per-slide tile sources resolved for {expected_partition}."
        )

    dfs = []
    inv_rows = []

    for i, (sid, path) in enumerate(
        sorted(resolved.items()),
        start=1,
    ):
        header = pd.read_csv(
            path,
            nrows=0,
        )

        required = {
            "slide_id",
            "histologic_class",
            "PC3",
        }

        missing = required - set(header.columns)

        if missing:
            raise RuntimeError(
                f"{sid} source missing {sorted(missing)}:\n{path}"
            )

        df = pd.read_csv(
            path,
            usecols=[
                "slide_id",
                "histologic_class",
                "PC3",
            ],
            low_memory=False,
        )

        df["slide_id"] = df["slide_id"].astype(str)
        df["histologic_class"] = df["histologic_class"].astype(str)
        df["PC3"] = pd.to_numeric(df["PC3"], errors="coerce")

        observed = set(
            df["slide_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if observed != {sid}:
            raise RuntimeError(
                f"Unexpected slide IDs inside {path}: {observed}"
            )

        dfs.append(df)

        inv_rows.append(
            {
                "partition": expected_partition,
                "source_mode": "PER_SLIDE_FILES",
                "slide_id": sid,
                "path": str(path),
                "sha256": sha256_file(path),
                "n_rows": int(len(df)),
                "n_slides": 1,
            }
        )

        if (
            i % 20 == 0
            or i == len(resolved)
        ):
            print(
                f"[LOAD {expected_partition}] {i}/{len(resolved)} slides"
            )

    return (
        pd.concat(dfs, ignore_index=True),
        pd.DataFrame(inv_rows),
    )


# =============================================================================
# CLASS FILTERING / AUC
# =============================================================================

def classify_auc_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["binary_auc_role"] = "EXCLUDED"

    out.loc[
        out["histologic_class"].isin(POSITIVE_CLASSES),
        "binary_auc_role",
    ] = "POSITIVE_TUMOR"

    out.loc[
        out["histologic_class"].isin(NEGATIVE_CLASSES),
        "binary_auc_role",
    ] = "NEGATIVE_NON_TUMOR"

    out["binary_label"] = np.nan

    out.loc[
        out["binary_auc_role"].eq("POSITIVE_TUMOR"),
        "binary_label",
    ] = 1

    out.loc[
        out["binary_auc_role"].eq("NEGATIVE_NON_TUMOR"),
        "binary_label",
    ] = 0

    return out


def safe_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:

    finite = (
        np.isfinite(labels)
        &
        np.isfinite(scores)
    )

    y = labels[finite]
    s = scores[finite]

    if len(y) == 0:
        return np.nan

    if np.unique(y).size != 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            s,
        )
    )


def evaluate_partition(
    tiles: pd.DataFrame,
    partition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    classified = classify_auc_rows(
        tiles
    )

    slides = sorted(
        classified["slide_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []
    qc_rows = []

    for i, sid in enumerate(
        slides,
        start=1,
    ):

        g = classified.loc[
            classified["slide_id"].eq(sid)
        ].copy()

        observed_counts = (
            g["histologic_class"]
            .value_counts()
            .to_dict()
        )

        included = g.loc[
            g["binary_auc_role"].isin(
                [
                    "POSITIVE_TUMOR",
                    "NEGATIVE_NON_TUMOR",
                ]
            )
        ].copy()

        included = included.loc[
            np.isfinite(
                pd.to_numeric(
                    included["PC3"],
                    errors="coerce",
                )
            )
        ].copy()

        n_pos = int(
            included["binary_label"].eq(1).sum()
        )

        n_neg = int(
            included["binary_label"].eq(0).sum()
        )

        n_excluded = int(
            len(g)
            -
            len(included)
        )

        eligible = bool(
            n_pos >= MIN_POSITIVE_TILES_PER_SLIDE
            and
            n_neg >= MIN_NEGATIVE_TILES_PER_SLIDE
        )

        if eligible:
            labels = pd.to_numeric(
                included["binary_label"],
                errors="coerce",
            ).to_numpy(dtype=float)

            pc3 = pd.to_numeric(
                included["PC3"],
                errors="coerce",
            ).to_numpy(dtype=float)

            raw_auc_pc3 = safe_auc(
                labels,
                pc3,
            )

            fixed_direction_auc = safe_auc(
                labels,
                -pc3,
            )

            if np.isfinite(raw_auc_pc3) and np.isfinite(fixed_direction_auc):
                orientation_free_auc = float(
                    max(
                        raw_auc_pc3,
                        fixed_direction_auc,
                    )
                )

                historical_best_direction = (
                    "high_score_predicts_tumor"
                    if raw_auc_pc3 >= fixed_direction_auc
                    else
                    "low_score_predicts_tumor"
                )
            else:
                orientation_free_auc = np.nan
                historical_best_direction = ""

            status = "EVALUABLE"

        else:
            raw_auc_pc3 = np.nan
            fixed_direction_auc = np.nan
            orientation_free_auc = np.nan
            historical_best_direction = ""

            reasons = []

            if n_pos < MIN_POSITIVE_TILES_PER_SLIDE:
                reasons.append("TUMOR_SUPPORT_BELOW_10")

            if n_neg < MIN_NEGATIVE_TILES_PER_SLIDE:
                reasons.append("NON_TUMOR_SUPPORT_BELOW_10")

            status = "|".join(reasons)

        rows.append(
            {
                "partition":
                    partition,

                "scientific_role":
                    PARTITION_ROLE_MAP[
                        partition
                    ],

                "slide_id":
                    sid,

                "n_total_input_tiles":
                    int(
                        len(g)
                    ),

                "n_auc_included_tiles":
                    int(
                        len(included)
                    ),

                "n_excluded_tiles":
                    n_excluded,

                "n_tumor_tiles":
                    n_pos,

                "n_non_tumor_tiles":
                    n_neg,

                "n_normal_tiles":
                    int(
                        observed_counts.get(
                            "Normal",
                            0,
                        )
                    ),

                "n_stroma_tiles":
                    int(
                        observed_counts.get(
                            "Stroma",
                            0,
                        )
                    ),

                "n_g3_tiles":
                    int(
                        observed_counts.get(
                            "G3",
                            0,
                        )
                    ),

                "n_g4_tiles":
                    int(
                        observed_counts.get(
                            "G4",
                            0,
                        )
                    ),

                "n_g5_tiles":
                    int(
                        observed_counts.get(
                            "G5",
                            0,
                        )
                    ),

                "primary_auc_evaluable":
                    eligible,

                "evaluability_status":
                    status,

                "raw_auc_pc3":
                    raw_auc_pc3,

                "primary_fixed_direction_auc_minus_pc3":
                    fixed_direction_auc,

                "supportive_orientation_free_auc":
                    orientation_free_auc,

                "supportive_historical_best_direction":
                    historical_best_direction,

                "primary_direction_source":
                    "STEP06_DISCOVERY_FROZEN_LOW_IN_TUMOR",

                "primary_score":
                    "-PC3",

                "direction_selected_from_partition":
                    False,
            }
        )

        qc_rows.append(
            {
                "partition":
                    partition,

                "slide_id":
                    sid,

                "n_tumor_tiles":
                    n_pos,

                "n_non_tumor_tiles":
                    n_neg,

                "minimum_tumor_required":
                    MIN_POSITIVE_TILES_PER_SLIDE,

                "minimum_non_tumor_required":
                    MIN_NEGATIVE_TILES_PER_SLIDE,

                "eligible":
                    eligible,

                "reason":
                    status,
            }
        )

        if (
            i % 25 == 0
            or i == len(slides)
        ):
            print(
                f"[AUC {partition}] {i}/{len(slides)} slides"
            )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(qc_rows),
    )


# =============================================================================
# BOOTSTRAP SUMMARY
# =============================================================================

def bootstrap_median_ci(
    values: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:

    x = finite_array(
        values
    )

    n = len(x)

    if n == 0:
        return np.nan, np.nan

    if n == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(
        seed
    )

    idx = rng.integers(
        0,
        n,
        size=(
            n_bootstrap,
            n,
        ),
    )

    medians = np.median(
        x[idx],
        axis=1,
    )

    return (
        float(
            np.quantile(
                medians,
                0.025,
            )
        ),
        float(
            np.quantile(
                medians,
                0.975,
            )
        ),
    )


def summarize_endpoint(
    per_slide: pd.DataFrame,
    endpoint_col: str,
    endpoint_id: str,
    endpoint_role: str,
) -> pd.DataFrame:

    rows = []

    for partition in [
        "train_validation",
        "test_replication",
    ]:

        g = per_slide.loc[
            per_slide["partition"].eq(partition)
        ].copy()

        evaluable = g.loc[
            g["primary_auc_evaluable"]
        ].copy()

        values = finite_array(
            pd.to_numeric(
                evaluable[endpoint_col],
                errors="coerce",
            ).to_numpy(dtype=float)
        )

        ci_low, ci_high = bootstrap_median_ci(
            values,
            BOOTSTRAP_RESAMPLES,
            BOOTSTRAP_SEED,
        )

        rows.append(
            {
                "endpoint_id":
                    endpoint_id,

                "endpoint_role":
                    endpoint_role,

                "partition":
                    partition,

                "scientific_role":
                    PARTITION_ROLE_MAP[
                        partition
                    ],

                "n_total_slides":
                    int(
                        g["slide_id"].nunique()
                    ),

                "n_evaluable_slides":
                    int(
                        len(values)
                    ),

                "evaluable_fraction":
                    (
                        float(
                            len(values)
                            /
                            g["slide_id"].nunique()
                        )
                        if g["slide_id"].nunique()
                        else
                        np.nan
                    ),

                "median_auc":
                    (
                        float(
                            np.median(
                                values
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "mean_auc":
                    (
                        float(
                            np.mean(
                                values
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "q25_auc":
                    (
                        float(
                            np.quantile(
                                values,
                                0.25,
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "q75_auc":
                    (
                        float(
                            np.quantile(
                                values,
                                0.75,
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "fraction_auc_gt_0_5":
                    (
                        float(
                            np.mean(
                                values > 0.5
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "fraction_auc_ge_0_7":
                    (
                        float(
                            np.mean(
                                values >= 0.7
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "fraction_auc_ge_0_8":
                    (
                        float(
                            np.mean(
                                values >= 0.8
                            )
                        )
                        if len(values)
                        else
                        np.nan
                    ),

                "bootstrap_ci_low":
                    ci_low,

                "bootstrap_ci_high":
                    ci_high,

                "bootstrap_resamples":
                    BOOTSTRAP_RESAMPLES,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# INPUT COVERAGE
# =============================================================================

def build_coverage(
    train_tiles: pd.DataFrame,
    test_tiles: pd.DataFrame,
    train_inventory: pd.DataFrame,
    test_inventory: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for partition, tiles, inventory in [
        (
            "train_validation",
            train_tiles,
            train_inventory,
        ),
        (
            "test_replication",
            test_tiles,
            test_inventory,
        ),
    ]:

        rows.append(
            {
                "partition":
                    partition,

                "scientific_role":
                    PARTITION_ROLE_MAP[
                        partition
                    ],

                "n_input_tile_rows":
                    int(
                        len(
                            tiles
                        )
                    ),

                "n_input_slides":
                    int(
                        tiles[
                            "slide_id"
                        ].nunique()
                    ),

                "source_mode":
                    "|".join(
                        sorted(
                            inventory[
                                "source_mode"
                            ].astype(
                                str
                            ).unique()
                        )
                    ),

                "n_source_files":
                    int(
                        len(
                            inventory
                        )
                    ),

                "contains_required_columns":
                    True,

                "auc_cohort_scope_note":
                    (
                        "AUC evaluated on all slides represented in this input source; "
                        "no outcome-based slide exclusion."
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# README
# =============================================================================

def build_readme(
    coverage: pd.DataFrame,
) -> str:

    coverage_text = coverage.to_string(
        index=False
    )

    return f"""STEP 21C
EVALUATE FROZEN AGGC PC3 TUMOR-LOCALIZATION AUC

Version
-------
{VERSION}

Schema
------
{SCHEMA}


ROLE
----
FORMAL FROZEN AUC EXECUTION.


PRIMARY ENDPOINT
----------------
Positive:
    G3 + G4 + G5

Negative:
    Normal + Stroma

Excluded:
    any other class

Per-slide eligibility:
    >=10 tumor tiles
    AND
    >=10 non-tumor tiles

Frozen PC3 direction:
    low_in_tumor

Primary score:
    -PC3

Primary endpoint:
    roc_auc_score(tumor_label, -PC3)

Interpretation:
    AUC > 0.5 means lower frozen PC3 values preferentially localize to
    tumor-associated histology.


SUPPORTIVE HISTORICAL ENDPOINT
------------------------------
orientation-free AUC:
    max(AUC(PC3), AUC(-PC3))

Supportive only.
It cannot determine PC3 direction or replace the primary endpoint.


AGGC PARTITION ROLES
--------------------
Subset1_Train:
    AGGC characterization partition

Subset1_Test:
    AGGC held-out replication partition

Formal held-out execution cohort:
    Step21B3 frozen 35-slide cohort

Raw Subset1 Test universe:
    45 slides

Technical/provenance exclusions frozen before AUC:
    10 slides

Neither partition trained or refit PC3.


INPUT COVERAGE
--------------
{coverage_text}


PARTITION SUMMARY
-----------------
Per endpoint and partition:
    n total slides
    n evaluable slides
    median AUC
    mean AUC
    Q25/Q75
    fraction AUC > 0.5
    fraction AUC >= 0.7
    fraction AUC >= 0.8
    95% bootstrap CI of median

Bootstrap:
    {BOOTSTRAP_RESAMPLES}

Seed:
    {BOOTSTRAP_SEED}


NOT PERFORMED
-------------
PC1-PC8 ranking:
    NO

PC screening:
    NO

PCA refit:
    NO

AGGC-driven PC3 sign flip:
    NO

Tumor/non-tumor class redefinition:
    NO

Support threshold tuning:
    NO

Best partition selection:
    NO

Outcome-based slide exclusion:
    NO

Pooled tile-level inferential summary:
    NO


IMPORTANT INTERPRETATION
------------------------
The fixed-direction primary AUC is the formal endpoint.

The orientation-free AUC is retained only for continuity with historical
multi-PC screening and must not be presented as the confirmatory directional
result.
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
        "--step21b-dir",
        type=Path,
        default=STEP21B_DIR,
    )

    p.add_argument(
        "--train-morphology-root",
        type=Path,
        default=DEFAULT_TRAIN_MORPHOLOGY_ROOT,
        help=(
            "Historical characterization per-slide morphology root. "
            "Used only if --train-tile-source is not provided."
        ),
    )

    p.add_argument(
        "--train-tile-source",
        type=Path,
        default=None,
        help=(
            "Optional combined characterization tile CSV/CSV.GZ containing "
            "slide_id,histologic_class,PC3. Preferred if it represents a fuller "
            "AUC cohort than historical morphology outputs."
        ),
    )

    p.add_argument(
        "--step21b2-dir",
        type=Path,
        default=STEP21B2_DIR,
        help=(
            "Frozen Step21B2 complete Test input-construction directory."
        ),
    )

    p.add_argument(
        "--step21b3-dir",
        type=Path,
        default=STEP21B3_DIR,
        help=(
            "Frozen Step21B3 Test execution-cohort directory."
        ),
    )

    p.add_argument(
        "--test-tile-source",
        type=Path,
        default=None,
        help=(
            "Optional explicit Test tile source. If supplied, it MUST be "
            "exactly the Step21B2 source frozen by Step21B3; alternate Test "
            "sources are prohibited in formal v1.1 execution."
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

    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    banner(
        "STEP 21C: EVALUATE FROZEN AGGC PC3 TUMOR-LOCALIZATION AUC"
    )

    print(
        f"[VERSION]       {VERSION}"
    )

    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    print(
        f"[STEP21B]       {args.step21b_dir}"
    )

    print(
        f"[TRAIN ROOT]    {args.train_morphology_root}"
    )

    print(
        f"[TRAIN SOURCE]  {args.train_tile_source}"
    )

    frozen_default_test_source = (
        args.step21b2_dir
        / "aggc_subset1_test_pc3_auc_input.csv.gz"
    )

    test_tile_source = (
        args.test_tile_source
        if args.test_tile_source is not None
        else frozen_default_test_source
    )

    print(
        f"[STEP21B2]      {args.step21b2_dir}"
    )

    print(
        f"[STEP21B3]      {args.step21b3_dir}"
    )

    print(
        f"[TEST SOURCE]   {test_tile_source}"
    )

    print(
        f"[OUTPUT]        {args.output}"
    )

    print(
        f"[OVERWRITE]     {args.overwrite}"
    )

    print()
    print(
        "[ROLE]          FORMAL FROZEN AUC EXECUTION"
    )

    print(
        "[PRIMARY SCORE] -PC3"
    )

    print(
        "[POSITIVE]      G3 + G4 + G5"
    )

    print(
        "[NEGATIVE]      Normal + Stroma"
    )

    print(
        "[MIN SUPPORT]   >=10 tumor AND >=10 non-tumor tiles/slide"
    )

    print(
        "[PC SCREEN]     NONE"
    )

    print(
        "[PCA REFIT]     NONE"
    )

    print(
        "[SIGN TUNING]   NONE"
    )

    prepare_output(
        args.output,
        args.overwrite,
    )

    # -------------------------------------------------------------------------
    # Validate Step21B
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE STEP21B FROZEN CONTRACT"
    )

    contract = validate_step21b(
        args.step21b_dir
    )

    print(
        "Frozen positive classes G3/G4/G5       : PASS"
    )

    print(
        "Frozen negative classes Normal/Stroma  : PASS"
    )

    print(
        "Frozen minimum support 10 / 10         : PASS"
    )

    print(
        "Frozen PC3 direction low_in_tumor      : PASS"
    )

    print(
        "Frozen primary score -PC3              : PASS"
    )

    print(
        "Direction selected from AGGC           : NO"
    )

    # -------------------------------------------------------------------------
    # Validate Step21B3 Test cohort before any AUC outcome is calculated.
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE STEP21B3 FROZEN TEST EXECUTION COHORT"
    )

    b3_contract = validate_step21b3(
        args.step21b3_dir,
        test_tile_source,
    )

    print(
        "Frozen raw Test universe               : PASS (45)"
    )

    print(
        "Frozen Test execution cohort            : PASS (35)"
    )

    print(
        "Technical/provenance exclusions         : PASS (10)"
    )

    print(
        "Outcome-based exclusion                 : NO"
    )

    # -------------------------------------------------------------------------
    # Resolve characterization source
    # -------------------------------------------------------------------------

    banner(
        "LOAD AGGC CHARACTERIZATION TILE SOURCE"
    )

    if args.train_tile_source is not None:

        train_tiles, train_inventory = load_combined_tile_source(
            args.train_tile_source,
            "train_validation",
        )

    else:

        train_resolved = discover_per_slide_train_files(
            args.train_morphology_root
        )

        if not train_resolved:
            raise RuntimeError(
                "No characterization tile source found.\n\n"
                f"Searched historical morphology root:\n"
                f"  {args.train_morphology_root}\n\n"
                "Provide --train-tile-source if you have a fuller combined "
                "Train PC3+histologic_class table."
            )

        train_tiles, train_inventory = load_per_slide_sources(
            train_resolved,
            "train_validation",
        )

    print(
        f"Characterization tile rows : {len(train_tiles):,}"
    )

    print(
        f"Characterization slides    : {train_tiles['slide_id'].nunique()}"
    )

    # -------------------------------------------------------------------------
    # Resolve replication source
    # -------------------------------------------------------------------------

    banner(
        "LOAD AGGC HELD-OUT REPLICATION TILE SOURCE"
    )

    test_tiles, test_inventory = load_combined_tile_source(
        test_tile_source,
        "test_replication",
    )

    validate_test_tiles_against_step21b3(
        test_tiles,
        b3_contract,
    )

    print(
        f"Replication tile rows      : {len(test_tiles):,}"
    )

    print(
        f"Replication slides         : {test_tiles['slide_id'].nunique()}"
    )

    # -------------------------------------------------------------------------
    # Coverage warning
    # -------------------------------------------------------------------------

    coverage = build_coverage(
        train_tiles,
        test_tiles,
        train_inventory,
        test_inventory,
    )

    banner(
        "INPUT COVERAGE AUDIT"
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    test_input_slides = int(
        coverage.loc[
            coverage["partition"].eq("test_replication"),
            "n_input_slides",
        ].iloc[0]
    )

    if test_input_slides != 35:
        raise RuntimeError(
            f"Formal Step21C v1.1 requires exactly 35 Step21B3-frozen "
            f"Test slides, found {test_input_slides}."
        )

    print()
    print(
        "[TEST COHORT] Raw Subset1 Test universe = 45; formal frozen "
        "execution cohort = 35; 10 technical/provenance exclusions were "
        "frozen before AUC outcome."
    )

    # -------------------------------------------------------------------------
    # Evaluate per-slide AUC
    # -------------------------------------------------------------------------

    banner(
        "CALCULATE FROZEN PER-SLIDE PC3 TUMOR AUC"
    )

    train_auc, train_qc = evaluate_partition(
        train_tiles,
        "train_validation",
    )

    test_auc, test_qc = evaluate_partition(
        test_tiles,
        "test_replication",
    )

    per_slide = pd.concat(
        [
            train_auc,
            test_auc,
        ],
        ignore_index=True,
    )

    evaluability = pd.concat(
        [
            train_qc,
            test_qc,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Partition summaries
    # -------------------------------------------------------------------------

    primary_summary = summarize_endpoint(
        per_slide,
        endpoint_col=
            "primary_fixed_direction_auc_minus_pc3",
        endpoint_id=
            "PRIMARY_FIXED_DIRECTION_PC3_TUMOR_AUC",
        endpoint_role=
            "PRIMARY",
    )

    supportive_summary = summarize_endpoint(
        per_slide,
        endpoint_col=
            "supportive_orientation_free_auc",
        endpoint_id=
            "SUPPORTIVE_HISTORICAL_ORIENTATION_FREE_PC3_AUC",
        endpoint_role=
            "SUPPORTIVE_HISTORICAL_CONTINUITY",
    )

    # -------------------------------------------------------------------------
    # Save main outputs
    # -------------------------------------------------------------------------

    per_slide_path = (
        args.output
        /
        OUT_PER_SLIDE
    )

    primary_summary_path = (
        args.output
        /
        OUT_PRIMARY_SUMMARY
    )

    supportive_summary_path = (
        args.output
        /
        OUT_SUPPORTIVE_SUMMARY
    )

    evaluability_path = (
        args.output
        /
        OUT_EVALUABILITY
    )

    coverage_path = (
        args.output
        /
        OUT_COVERAGE
    )

    per_slide.to_csv(
        per_slide_path,
        index=False,
    )

    primary_summary.to_csv(
        primary_summary_path,
        index=False,
    )

    supportive_summary.to_csv(
        supportive_summary_path,
        index=False,
    )

    evaluability.to_csv(
        evaluability_path,
        index=False,
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Source hashes
    # -------------------------------------------------------------------------

    hash_rows = []

    frozen_sources = [
        (
            "step21b_analysis_plan",
            STEP21B_PLAN,
        ),
        (
            "step21b_class_specification",
            STEP21B_CLASSES,
        ),
        (
            "step21b_threshold_specification",
            STEP21B_THRESHOLDS,
        ),
        (
            "step21b_partition_roles",
            STEP21B_PARTITIONS,
        ),
        (
            "step21b_endpoint_specification",
            STEP21B_ENDPOINTS,
        ),
        (
            "step21b_direction_snapshot",
            STEP21B_DIRECTION,
        ),
        (
            "step21b_metadata",
            STEP21B_METADATA,
        ),
        (
            "step21b2_test_tile_source",
            test_tile_source,
        ),
        (
            "step21b3_frozen_cohort",
            b3_contract["paths"]["cohort"],
        ),
        (
            "step21b3_excluded_manifest",
            b3_contract["paths"]["excluded"],
        ),
        (
            "step21b3_cohort_summary",
            b3_contract["paths"]["summary"],
        ),
        (
            "step21b3_contract",
            b3_contract["paths"]["contract"],
        ),
    ]

    for role, path in frozen_sources:

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

    for inventory_name, inventory in [
        (
            "characterization_tile_source",
            train_inventory,
        ),
        (
            "replication_tile_source",
            test_inventory,
        ),
    ]:

        for i, row in inventory.iterrows():

            hash_rows.append(
                {
                    "role":
                        (
                            inventory_name
                            if len(inventory) == 1
                            else
                            f"{inventory_name}_{i+1}"
                        ),

                    "path":
                        str(
                            row[
                                "path"
                            ]
                        ),

                    "sha256":
                        str(
                            row[
                                "sha256"
                            ]
                        ),
                }
            )

    hashes_path = (
        args.output
        /
        OUT_HASHES
    )

    pd.DataFrame(
        hash_rows
    ).to_csv(
        hashes_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    audit_rows = [
        (
            "step21b_contract",
            "EXECUTED_AS_FROZEN",
            "v1.0.0",
        ),
        (
            "step21b2_test_tile_input",
            "EXECUTED_AS_FROZEN",
            str(test_tile_source),
        ),
        (
            "step21b3_test_execution_cohort",
            "EXECUTED_AS_FROZEN",
            "35 slides; 10 technical/provenance exclusions frozen before AUC",
        ),
        (
            "test_slide_set_reconciliation",
            "PASS",
            "Step21B2 tile source == Step21B3 frozen cohort",
        ),
        (
            "test_per_slide_support_reconciliation",
            "PASS",
            "clean/tumor/non-tumor counts exactly match Step21B3",
        ),
        (
            "positive_classes",
            "EXECUTED_AS_FROZEN",
            "G3|G4|G5",
        ),
        (
            "negative_classes",
            "EXECUTED_AS_FROZEN",
            "Normal|Stroma",
        ),
        (
            "minimum_tumor_tiles_per_slide",
            "EXECUTED_AS_FROZEN",
            "10",
        ),
        (
            "minimum_non_tumor_tiles_per_slide",
            "EXECUTED_AS_FROZEN",
            "10",
        ),
        (
            "primary_score",
            "EXECUTED_AS_FROZEN",
            "-PC3",
        ),
        (
            "primary_direction_authority",
            "EXECUTED_AS_FROZEN",
            "Step06 discovery low_in_tumor",
        ),
        (
            "orientation_free_endpoint",
            "SUPPORTIVE_ONLY",
            "",
        ),
        (
            "pc1_pc8_ranking",
            "NO",
            "",
        ),
        (
            "pc_screening",
            "NO",
            "",
        ),
        (
            "pca_refit",
            "NO",
            "",
        ),
        (
            "aggc_driven_pc3_sign_flip",
            "NO",
            "",
        ),
        (
            "tumor_class_redefinition",
            "NO",
            "",
        ),
        (
            "negative_class_redefinition",
            "NO",
            "",
        ),
        (
            "support_threshold_tuning",
            "NO",
            "",
        ),
        (
            "best_partition_selection",
            "NO",
            "",
        ),
        (
            "outcome_based_slide_exclusion",
            "NO",
            "",
        ),
        (
            "pooled_tile_level_partition_summary",
            "NO",
            "",
        ),
    ]

    audit_path = (
        args.output
        /
        OUT_AUDIT
    )

    pd.DataFrame(
        audit_rows,
        columns=[
            "item",
            "status",
            "detail",
        ],
    ).to_csv(
        audit_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

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

        "analysis_role":
            "formal_frozen_auc_execution",

        "step21b2_test_tile_source":
            str(
                test_tile_source
            ),

        "step21b3_execution_cohort":
            str(
                b3_contract[
                    "paths"
                ][
                    "cohort"
                ]
            ),

        "raw_subset1_test_universe_slides":
            45,

        "frozen_test_execution_cohort_slides":
            35,

        "technical_provenance_excluded_test_slides":
            10,

        "test_cohort_authority":
            "Step21B3 frozen before AUC outcome",

        "primary_endpoint":
            "roc_auc_score(tumor_label, -PC3)",

        "positive_classes":
            sorted(
                POSITIVE_CLASSES
            ),

        "negative_classes":
            sorted(
                NEGATIVE_CLASSES
            ),

        "minimum_positive_tiles_per_slide":
            MIN_POSITIVE_TILES_PER_SLIDE,

        "minimum_negative_tiles_per_slide":
            MIN_NEGATIVE_TILES_PER_SLIDE,

        "frozen_pc3_direction":
            EXPECTED_FROZEN_DIRECTION,

        "primary_score":
            "-PC3",

        "bootstrap_resamples":
            BOOTSTRAP_RESAMPLES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "characterization_input_slides":
            int(
                train_tiles[
                    "slide_id"
                ].nunique()
            ),

        "replication_input_slides":
            int(
                test_tiles[
                    "slide_id"
                ].nunique()
            ),

        "characterization_evaluable_slides":
            int(
                train_auc[
                    "primary_auc_evaluable"
                ].sum()
            ),

        "replication_evaluable_slides":
            int(
                test_auc[
                    "primary_auc_evaluable"
                ].sum()
            ),

        "guardrails": {
            "pc1_pc8_ranking":
                False,

            "pc_screening":
                False,

            "pca_refit":
                False,

            "aggc_driven_pc3_sign_flip":
                False,

            "tumor_class_redefinition":
                False,

            "negative_class_redefinition":
                False,

            "support_threshold_tuning":
                False,

            "best_partition_selection":
                False,

            "outcome_based_slide_exclusion":
                False,

            "pooled_tile_level_partition_summary":
                False,
        },
    }

    metadata_path = (
        args.output
        /
        OUT_METADATA
    )

    save_json(
        metadata_path,
        metadata,
    )

    readme_path = (
        args.output
        /
        OUT_README
    )

    readme_path.write_text(
        build_readme(
            coverage
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    banner(
        "PRIMARY FIXED-DIRECTION PC3 TUMOR-AUC RESULTS"
    )

    print(
        primary_summary.to_string(
            index=False
        )
    )

    banner(
        "SUPPORTIVE HISTORICAL ORIENTATION-FREE AUC"
    )

    print(
        supportive_summary.to_string(
            index=False
        )
    )

    banner(
        "EVALUABILITY SUMMARY"
    )

    eval_console = (
        per_slide.groupby(
            [
                "partition",
                "scientific_role",
            ],
            as_index=False,
        )
        .agg(
            n_total_slides=(
                "slide_id",
                "nunique",
            ),
            n_evaluable_slides=(
                "primary_auc_evaluable",
                "sum",
            ),
        )
    )

    eval_console[
        "evaluable_fraction"
    ] = (
        eval_console[
            "n_evaluable_slides"
        ]
        /
        eval_console[
            "n_total_slides"
        ]
    )

    print(
        eval_console.to_string(
            index=False
        )
    )

    banner(
        "STEP21C EXECUTION AUDIT"
    )

    print(
        "Step21B endpoint executed as frozen      : YES"
    )

    print(
        "Step21B2 Test tile input executed frozen : YES"
    )

    print(
        "Step21B3 Test cohort executed frozen     : YES"
    )

    print(
        "Test slide set matched Step21B3          : YES"
    )

    print(
        "Test support counts matched Step21B3     : YES"
    )

    print(
        "Positive classes changed                 : NO"
    )

    print(
        "Negative classes changed                 : NO"
    )

    print(
        "Minimum tile support changed             : NO"
    )

    print(
        "Primary score changed                    : NO"
    )

    print(
        "AGGC-driven PC3 sign flip                : NO"
    )

    print(
        "PC1-PC8 ranking                          : NO"
    )

    print(
        "PC screening                             : NO"
    )

    print(
        "PCA refit                                : NO"
    )

    print(
        "Best partition selected                  : NO"
    )

    print(
        "Outcome-based slide exclusion            : NO"
    )

    print(
        "Orientation-free endpoint promoted       : NO"
    )

    print()
    print(
        "[OK] Saved:"
    )

    for path in [
        per_slide_path,
        primary_summary_path,
        supportive_summary_path,
        evaluability_path,
        coverage_path,
        audit_path,
        hashes_path,
        metadata_path,
        readme_path,
    ]:
        print(
            f"  {path}"
        )

    print()
    print(
        "[DONE] Step21C frozen AGGC PC3 tumor-localization AUC evaluated."
    )

    print(
        "[NEXT] Review PRIMARY fixed-direction AUC first. The formal Test "
        "execution cohort is now fixed by Step21B3 at 35 slides; do not expand "
        "or shrink it based on AUC outcome. Visualization should remain a "
        "separate Step21D."
    )


if __name__ == "__main__":
    main()
