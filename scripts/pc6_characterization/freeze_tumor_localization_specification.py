#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 25A
==========
FREEZE AGGC PC6 EXTERNAL TUMOR-LOCALIZATION SPECIFICATION

VERSION
-------
1.0.0

SCHEMA
------
25a-aggc-pc6-external-tumor-localization-freeze-schema-1.0

ROLE
----
PRE-EXECUTION EXTERNAL TUMOR-LOCALIZATION FREEZE ONLY.

This step freezes the AGGC external tumor-localization analysis for the already
established Virchow2 PC6 secondary histologic axis.

It MUST NOT calculate tumor AUC, read historical PC6 AUC values, rank PCs, tune
the PC6 sign, or choose a favorable AGGC subset after outcomes are inspected.

SCIENTIFIC QUESTION
-------------------
Does the discovery-frozen Virchow2 PC6 axis localize prostate tumor tissue in
an independent AGGC cohort?

UPSTREAM SCIENTIFIC AUTHORITIES
-------------------------------
Step06:
    frozen PC6 direction = high_in_tumor

Step24D:
    PC6 external morphology evidence is already frozen / complete.
    Step25 must not alter the morphology phenotype.

PC6 ROLE
--------
PC3 remains the primary interpreted latent histologic axis.

PC6 remains:
    secondary interpreted axis
    frozen direction = high_in_tumor

AGGC TUMOR LABEL AUTHORITY
--------------------------
Use the already-constructed exact tile-level AGGC Gleason/tissue label files.

Frozen tumor definition:
    tumor = Gleason pattern 3 OR 4 OR 5

Frozen non-tumor definition:
    non_tumor = Normal OR Stroma

Ambiguous / unsupported labels:
    excluded from AUC evaluation by label definition only

No label threshold is selected from PC6 outcomes.

PRIMARY EXTERNAL COHORT
-----------------------
Subset1_Test

SECONDARY CHARACTERIZATION
--------------------------
Subset1_Train

Train and Test are never pooled.

AUC DEFINITION
--------------
Per slide:

    predictor:
        raw frozen PC6 score

    binary target:
        tumor = 1
        non_tumor = 0

Because PC6 direction is frozen as high_in_tumor, the primary AUC is the
ordinary ROC AUC using raw PC6 scores.

NO post-hoc sign flip is allowed.

Slide eligibility:
    >= 1 tumor tile
    >= 1 non-tumor tile

Primary summary:
    median per-slide AUC in Subset1_Test

Uncertainty:
    nonparametric bootstrap over slides

Secondary summaries:
    fraction of eligible Test slides with AUC > 0.5
    Subset1_Train median per-slide AUC
    pooled-tile AUC may be reported as supportive only, never primary

NO TILE-LEVEL PSEUDOREPLICATION
-------------------------------
The primary analysis unit is the slide.

Pooled tile AUC is supportive only.

BOOTSTRAP
---------
Resamples:
    10000

Seed:
    20260811

FREEZE-ONLY FIREWALL
--------------------
Step25A MUST NOT:
    - calculate any tumor AUC;
    - read historical PC6 AUC result values;
    - rank PC1-PC8;
    - compare PCs to choose a better axis;
    - change PC6 sign;
    - choose a favorable tumor definition;
    - choose a favorable Test subset;
    - use morphology outcomes to alter the localization analysis;
    - refit PCA;
    - reproject PCA;
    - pool Train/Test;
    - calculate p-values;
    - tune any score threshold;
    - define PC6 by AGGC tumor localization performance.

NEXT
----
Step25B:
    construct / measure the frozen AGGC PC6 tumor-localization input only

Step25C:
    calculate the frozen per-slide Test-primary AUC / bootstrap evaluation

Step25D:
    freeze final external tumor-localization evidence / claim boundary
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

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "25a-aggc-pc6-external-tumor-localization-freeze-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

MEDIA = REPO_ROOT

RESULTS = RESULTS_ROOT

STEP06 = RESULTS / "06_frozen_pc_direction_specification"
STEP24D = RESULTS / "24d_final_aggc_pc6_external_morphology_evidence_freeze"

STEP06_DIRECTION = STEP06 / "pc1_pc8_frozen_direction_specification.csv"
STEP24D_METADATA = STEP24D / "analysis_metadata.json"
STEP24D_CLAIM = STEP24D / "aggc_pc6_final_external_claim_ceiling.txt"

AGGC_ROOT = CONFIG_AGGC_ROOT

LABEL_ROOT = AGGC_ROOT / "aggc_gleason_tile_labels"
PCA_ROOT = AGGC_ROOT / "aggc_pc1_pc8_grid_heatmaps"
EMBED_ROOT = AGGC_ROOT / "aggc_virchow2_embeddings"

DEFAULT_OUT = RESULTS / "25a_aggc_pc6_external_tumor_localization_specification"


# =============================================================================
# FROZEN SCIENTIFIC CONSTANTS
# =============================================================================

TARGET_AXIS = "PC6"
FROZEN_DIRECTION = "high_in_tumor"

PRIMARY_PARTITION = "Subset1_Test"
SECONDARY_PARTITION = "Subset1_Train"

TUMOR_LABELS = ("G3", "G4", "G5")
NON_TUMOR_LABELS = ("Normal", "Stroma")

PRIMARY_ANALYSIS_UNIT = "slide"
PRIMARY_ESTIMATOR = "median_per_slide_auc"

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260811

MIN_TUMOR_TILES = 1
MIN_NON_TUMOR_TILES = 1


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        f.write("\n")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty:\n{path}\n"
                "Use --overwrite only for an intentional re-freeze BEFORE Step25B/25C."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x: Any) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# =============================================================================
# VALIDATE STEP06 FROZEN PC6 DIRECTION
# =============================================================================

def validate_step06_pc6_direction() -> Dict[str, Any]:
    require_file(
        STEP06_DIRECTION,
        "Step06 frozen direction specification",
    )

    df = pd.read_csv(
        STEP06_DIRECTION,
        low_memory=False,
    )

    pc_col = None
    for c in ["pc", "PC", "pc_number", "component", "component_number"]:
        if c in df.columns:
            pc_col = c
            break

    if pc_col is None:
        raise RuntimeError(
            "Could not locate PC/component column in Step06 direction file."
        )

    direction_col = None
    for c in [
        "frozen_direction",
        "direction",
        "tumor_direction",
        "frozen_tumor_direction",
    ]:
        if c in df.columns:
            direction_col = c
            break

    if direction_col is None:
        raise RuntimeError(
            "Could not locate frozen direction column in Step06 direction file."
        )

    def is_pc6(v: Any) -> bool:
        s = str(v).strip().upper()
        return s in {"6", "PC6", "PC_6"}

    sub = df.loc[df[pc_col].map(is_pc6)].copy()

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected exactly one Step06 PC6 direction row; observed={len(sub)}"
        )

    direction = normalize_text(
        sub.iloc[0][direction_col]
    )

    if direction != FROZEN_DIRECTION:
        raise RuntimeError(
            "Step06 PC6 frozen direction mismatch.\n"
            f"Expected={FROZEN_DIRECTION}\nObserved={direction}"
        )

    row = sub.iloc[0].to_dict()

    return {
        "direction": direction,
        "source_file": str(STEP06_DIRECTION),
        "source_row": row,
    }


# =============================================================================
# VALIDATE STEP24D CLAIM BOUNDARY
# =============================================================================

def validate_step24d() -> Dict[str, Any]:
    require_file(
        STEP24D_METADATA,
        "Step24D final external morphology metadata",
    )

    require_file(
        STEP24D_CLAIM,
        "Step24D external morphology claim ceiling",
    )

    meta = load_json(
        STEP24D_METADATA
    )

    if meta.get("role") != "FINAL_AGGC_PC6_EXTERNAL_MORPHOLOGY_EVIDENCE_FREEZE":
        raise RuntimeError(
            "Step24D metadata does not represent the final frozen AGGC PC6 "
            "external morphology evidence."
        )

    boundary = meta.get(
        "claim_boundary",
        {},
    )

    if boundary.get("tumor_localization") != "SEPARATE_FUTURE_ANALYSIS_REQUIRED":
        raise RuntimeError(
            "Step24D no longer preserves tumor localization as a separate "
            "future analysis."
        )

    return meta


# =============================================================================
# STATIC AGGC INPUT INVENTORY ONLY
# =============================================================================

def inventory_static_inputs() -> pd.DataFrame:
    require_dir(
        LABEL_ROOT,
        "AGGC tile-label directory",
    )

    require_dir(
        PCA_ROOT,
        "AGGC frozen PCA target directory",
    )

    require_dir(
        EMBED_ROOT,
        "AGGC Virchow2 embedding directory",
    )

    rows: List[Dict[str, Any]] = []

    label_files = sorted(
        LABEL_ROOT.glob("Subset1_*_tile_labels.csv")
    )

    pca_files = sorted(
        PCA_ROOT.glob("Subset1_*/*_pca_targets.npy")
    )

    coord_files = sorted(
        EMBED_ROOT.glob("Subset1_*_coords.npy")
    )

    for role, files in [
        ("tile_label_file", label_files),
        ("frozen_pca_target_file", pca_files),
        ("coordinate_file", coord_files),
    ]:
        for p in files:
            name = p.name
            m = re.search(r"(Subset1_(?:Test|Train)_\d+)", str(p))
            slide_id = m.group(1) if m else ""

            partition = (
                PRIMARY_PARTITION
                if slide_id.startswith("Subset1_Test_")
                else SECONDARY_PARTITION
                if slide_id.startswith("Subset1_Train_")
                else "OTHER"
            )

            rows.append({
                "role": role,
                "partition": partition,
                "slide_id": slide_id,
                "path": str(p),
            })

    return pd.DataFrame(rows)


# =============================================================================
# FREEZE TABLES
# =============================================================================

def build_label_definition_table() -> pd.DataFrame:
    rows = []

    for label in TUMOR_LABELS:
        rows.append({
            "histologic_label": label,
            "binary_tumor_target": 1,
            "auc_role": "TUMOR",
            "included_in_auc": True,
        })

    for label in NON_TUMOR_LABELS:
        rows.append({
            "histologic_label": label,
            "binary_tumor_target": 0,
            "auc_role": "NON_TUMOR",
            "included_in_auc": True,
        })

    rows.append({
        "histologic_label": "OTHER_OR_UNSUPPORTED",
        "binary_tumor_target": np.nan,
        "auc_role": "EXCLUDED_BY_LABEL_DEFINITION",
        "included_in_auc": False,
    })

    return pd.DataFrame(rows)


def build_partition_table() -> pd.DataFrame:
    return pd.DataFrame([
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
    ])


def build_endpoint_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "endpoint": "median_per_slide_auc",
            "role": "PRIMARY",
            "partition": PRIMARY_PARTITION,
            "analysis_unit": PRIMARY_ANALYSIS_UNIT,
            "definition": (
                "Median of eligible-slide ROC AUCs using raw frozen PC6 as "
                "predictor and G3/G4/G5 vs Normal/Stroma as binary target."
            ),
        },
        {
            "endpoint": "slide_bootstrap_95ci_of_median_auc",
            "role": "PRIMARY_UNCERTAINTY",
            "partition": PRIMARY_PARTITION,
            "analysis_unit": PRIMARY_ANALYSIS_UNIT,
            "definition": (
                "95% nonparametric bootstrap CI of median eligible-slide AUC."
            ),
        },
        {
            "endpoint": "fraction_slides_auc_gt_0_5",
            "role": "SUPPORTIVE",
            "partition": PRIMARY_PARTITION,
            "analysis_unit": PRIMARY_ANALYSIS_UNIT,
            "definition": (
                "Fraction of eligible Test slides with per-slide AUC > 0.5."
            ),
        },
        {
            "endpoint": "median_per_slide_auc",
            "role": "SECONDARY_CHARACTERIZATION",
            "partition": SECONDARY_PARTITION,
            "analysis_unit": PRIMARY_ANALYSIS_UNIT,
            "definition": (
                "Same frozen per-slide AUC definition in AGGC Train; "
                "cannot replace Test."
            ),
        },
        {
            "endpoint": "pooled_tile_auc",
            "role": "SUPPORTIVE_ONLY",
            "partition": "SEPARATE_BY_PARTITION",
            "analysis_unit": "tile",
            "definition": (
                "Pooled tile AUC may be reported only as a supportive descriptive "
                "quantity and never as the primary endpoint."
            ),
        },
    ])


def build_sampling_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "parameter": "min_tumor_tiles_per_slide",
            "value": MIN_TUMOR_TILES,
            "role": "SLIDE_ELIGIBILITY",
        },
        {
            "parameter": "min_non_tumor_tiles_per_slide",
            "value": MIN_NON_TUMOR_TILES,
            "role": "SLIDE_ELIGIBILITY",
        },
        {
            "parameter": "bootstrap_resamples",
            "value": BOOTSTRAP_RESAMPLES,
            "role": "PRIMARY_UNCERTAINTY",
        },
        {
            "parameter": "bootstrap_seed",
            "value": BOOTSTRAP_SEED,
            "role": "PRIMARY_UNCERTAINTY",
        },
    ])


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Freeze AGGC PC6 external tumor-localization analysis before any "
            "AUC evaluation."
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
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
        "STEP 25A: FREEZE AGGC PC6 EXTERNAL TUMOR-LOCALIZATION SPECIFICATION"
    )

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP06]        {STEP06}")
    print(f"[STEP24D]       {STEP24D}")
    print(f"[LABEL ROOT]    {LABEL_ROOT}")
    print(f"[PCA ROOT]      {PCA_ROOT}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          PRE-EXECUTION TUMOR-LOCALIZATION FREEZE ONLY")
    print("[TARGET]        VIRCHOW2 PC6")
    print("[DIRECTION]     HIGH_IN_TUMOR / RAW PC6")
    print("[PRIMARY]       AGGC Subset1_Test")
    print("[SECONDARY]     AGGC Subset1_Train")
    print("[TUMOR]         G3 / G4 / G5")
    print("[NON-TUMOR]     Normal / Stroma")
    print("[PRIMARY UNIT]  SLIDE")
    print("[PRIMARY]       MEDIAN PER-SLIDE ROC AUC")
    print("[AUC]           NOT CALCULATED")
    print("[PC RANKING]    NONE")
    print("[SIGN TUNING]   NONE")
    print("[PCA REFIT]     NONE")
    print("[THRESHOLD]     NONE")
    print("[TRAIN/TEST]    NEVER POOLED")

    prepare_output(
        args.output,
        args.overwrite,
    )

    # -------------------------------------------------------------------------
    # Validate frozen PC6 authority.
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE DISCOVERY-FROZEN PC6 DIRECTION"
    )

    pc6 = validate_step06_pc6_direction()

    print(f"Frozen PC6 direction                : {pc6['direction']}")
    print(f"Direction authority                 : STEP06_DISCOVERY_FROZEN_DIRECTION")
    print(f"Direction selected from AGGC        : NO")
    print(f"Direction selected from tumor AUC   : NO")

    # -------------------------------------------------------------------------
    # Validate Step24D separation.
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE STEP24D EXTERNAL MORPHOLOGY CLAIM BOUNDARY"
    )

    step24d = validate_step24d()

    print("Step24D external morphology frozen  : PASS")
    print("Tumor localization separate         : PASS")
    print("Step24 morphology altered           : NO")

    # -------------------------------------------------------------------------
    # Static input inventory.
    # -------------------------------------------------------------------------

    banner(
        "STATIC AGGC INPUT INVENTORY — FILENAMES ONLY"
    )

    inventory = inventory_static_inputs()

    summary = (
        inventory.groupby(["role", "partition"])
        .size()
        .rename("n_files")
        .reset_index()
    )

    print(summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Frozen contract tables.
    # -------------------------------------------------------------------------

    labels = build_label_definition_table()
    partitions = build_partition_table()
    endpoints = build_endpoint_table()
    sampling = build_sampling_table()

    banner(
        "FROZEN AGGC PC6 TUMOR-LOCALIZATION CONTRACT"
    )

    print(f"Predictor                            : raw frozen Virchow2 PC6")
    print(f"Frozen PC6 direction                 : {FROZEN_DIRECTION}")
    print(f"Tumor labels                         : {', '.join(TUMOR_LABELS)}")
    print(f"Non-tumor labels                     : {', '.join(NON_TUMOR_LABELS)}")
    print(f"Primary partition                    : {PRIMARY_PARTITION}")
    print(f"Secondary partition                  : {SECONDARY_PARTITION}")
    print(f"Primary analysis unit                : {PRIMARY_ANALYSIS_UNIT}")
    print(f"Primary endpoint                     : {PRIMARY_ESTIMATOR}")
    print(f"Bootstrap resamples                  : {BOOTSTRAP_RESAMPLES}")
    print(f"Bootstrap seed                       : {BOOTSTRAP_SEED}")
    print(f"Post-hoc sign flip allowed           : NO")
    print(f"Train/Test pooling allowed           : NO")
    print(f"Pooled tile AUC primary              : NO")

    # -------------------------------------------------------------------------
    # Save freeze.
    # -------------------------------------------------------------------------

    plan = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "scientific_role": "pre_execution_aggc_pc6_external_tumor_localization_freeze",
        "target_axis": TARGET_AXIS,
        "frozen_pc6_direction": FROZEN_DIRECTION,
        "direction_authority": "STEP06_DISCOVERY_FROZEN_DIRECTION",
        "morphology_authority": "STEP24D_FROZEN_COMPLETE",
        "primary_partition": PRIMARY_PARTITION,
        "secondary_partition": SECONDARY_PARTITION,
        "tumor_definition": {
            "positive_labels": list(TUMOR_LABELS),
            "negative_labels": list(NON_TUMOR_LABELS),
            "unsupported_labels": "excluded_by_label_definition_only",
        },
        "auc_definition": {
            "predictor": "raw_frozen_PC6",
            "binary_target": "tumor_vs_non_tumor",
            "per_slide_auc": True,
            "primary_analysis_unit": PRIMARY_ANALYSIS_UNIT,
            "primary_estimator": PRIMARY_ESTIMATOR,
            "posthoc_sign_flip": False,
            "pooled_tile_auc_role": "SUPPORTIVE_ONLY",
        },
        "eligibility": {
            "min_tumor_tiles_per_slide": MIN_TUMOR_TILES,
            "min_non_tumor_tiles_per_slide": MIN_NON_TUMOR_TILES,
            "outcome_based_exclusion": False,
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "slide",
        },
        "prohibited": {
            "auc_calculation_in_step25a": True,
            "historical_pc6_auc_value_read": True,
            "pc_ranking": True,
            "pc_screening": True,
            "pc6_sign_tuning": True,
            "tumor_definition_tuning": True,
            "test_subset_selection": True,
            "train_test_pooling": True,
            "pca_refit": True,
            "pca_reprojection": True,
            "threshold_tuning": True,
            "morphology_redefinition": True,
        },
        "planned_steps": {
            "25B": "construct_measure_frozen_aggc_pc6_tumor_localization_input",
            "25C": "evaluate_frozen_test_primary_train_secondary_auc",
            "25D": "freeze_final_external_tumor_localization_evidence",
        },
    }

    save_json(
        args.output / "aggc_pc6_tumor_localization_plan.json",
        plan,
    )

    labels.to_csv(
        args.output / "aggc_pc6_tumor_label_definition.csv",
        index=False,
    )

    partitions.to_csv(
        args.output / "aggc_pc6_tumor_localization_partition_specification.csv",
        index=False,
    )

    endpoints.to_csv(
        args.output / "aggc_pc6_tumor_localization_endpoint_specification.csv",
        index=False,
    )

    sampling.to_csv(
        args.output / "aggc_pc6_tumor_localization_bootstrap_specification.csv",
        index=False,
    )

    inventory.to_csv(
        args.output / "aggc_pc6_tumor_localization_static_input_inventory.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "pc": "PC6",
            "frozen_direction": pc6["direction"],
            "authority": "STEP06_DISCOVERY_FROZEN_DIRECTION",
            "selected_from_aggc": False,
            "selected_from_auc": False,
            "sign_tuned_from_outcome": False,
        }
    ]).to_csv(
        args.output / "pc6_frozen_direction_snapshot.csv",
        index=False,
    )

    claim_text = """STEP25 AGGC PC6 EXTERNAL TUMOR-LOCALIZATION CLAIM CEILING

ALLOWED IF SUPPORTED
--------------------
1. Frozen PC6 shows above-chance tumor localization in held-out AGGC Test.
2. The median per-slide Test AUC may be reported with its slide-bootstrap CI.
3. The fraction of eligible Test slides with AUC > 0.5 may be reported.
4. AGGC provides external spatial/localization support for PC6 as a secondary
   histologic representation axis.

NOT ALLOWED
-----------
1. AGGC tumor localization defines PC6.
2. AGGC tumor AUC selects or flips the PC6 sign.
3. Train may replace or rescue Test.
4. Train and Test may be pooled for the primary estimate.
5. Pooled tile AUC may replace median per-slide AUC.
6. Tumor localization establishes a unique biological mechanism.
7. Tumor localization establishes statistical independence from PC3.
8. PC6 is redefined as a Gleason axis.
9. Step25 modifies the frozen Step24 morphology phenotype.
10. A post-hoc score threshold is optimized and presented as pre-specified.
"""

    (args.output / "pc6_aggc_tumor_localization_claim_ceiling.txt").write_text(
        claim_text,
        encoding="utf-8",
    )

    hash_rows = []

    for role, path in [
        ("step06_pc6_direction", STEP06_DIRECTION),
        ("step24d_metadata", STEP24D_METADATA),
        ("step24d_claim_ceiling", STEP24D_CLAIM),
    ]:
        hash_rows.append({
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
        })

    pd.DataFrame(hash_rows).to_csv(
        args.output / "source_hashes.csv",
        index=False,
    )

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "role": "PRE_EXECUTION_AGGC_PC6_EXTERNAL_TUMOR_LOCALIZATION_FREEZE_ONLY",
        "target_axis": TARGET_AXIS,
        "frozen_direction": FROZEN_DIRECTION,
        "primary_partition": PRIMARY_PARTITION,
        "secondary_partition": SECONDARY_PARTITION,
        "primary_analysis_unit": PRIMARY_ANALYSIS_UNIT,
        "primary_endpoint": PRIMARY_ESTIMATOR,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "freeze_only_audit": {
            "tumor_auc_calculated": False,
            "historical_pc6_auc_values_read": False,
            "pc_ranking": False,
            "pc_screening": False,
            "pc6_sign_tuned": False,
            "tumor_definition_tuned": False,
            "test_subset_selected_by_outcome": False,
            "train_test_pooled": False,
            "pca_refit": False,
            "pca_reprojection": False,
            "threshold_tuned": False,
            "step24_morphology_redefined": False,
            "test_primary_role_frozen": True,
            "slide_primary_analysis_unit_frozen": True,
        },
    }

    save_json(
        args.output / "analysis_metadata.json",
        metadata,
    )

    readme = f"""STEP 25A
AGGC PC6 EXTERNAL TUMOR-LOCALIZATION SPECIFICATION
==================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

Role
----
PRE-EXECUTION FREEZE ONLY.

PC6
---
Frozen direction:
  {FROZEN_DIRECTION}

Authority:
  STEP06 discovery-frozen direction

Primary external cohort
-----------------------
{PRIMARY_PARTITION}

Secondary
---------
{SECONDARY_PARTITION}

Tumor definition
----------------
Tumor:
  {", ".join(TUMOR_LABELS)}

Non-tumor:
  {", ".join(NON_TUMOR_LABELS)}

Primary analysis
----------------
Per-slide ROC AUC using raw frozen PC6.

Primary summary:
  median per-slide AUC in Test

Uncertainty:
  {BOOTSTRAP_RESAMPLES} slide bootstrap resamples
  seed = {BOOTSTRAP_SEED}

Supportive
----------
fraction of eligible Test slides with AUC > 0.5

pooled tile AUC:
  supportive only

Slide eligibility
-----------------
>= {MIN_TUMOR_TILES} tumor tile
>= {MIN_NON_TUMOR_TILES} non-tumor tile

Freeze-only firewall
--------------------
Tumor AUC calculated         : NO
Historical PC6 AUC read      : NO
PC ranking                   : NO
PC screening                 : NO
PC6 sign tuning              : NO
tumor-definition tuning      : NO
Test subset outcome selection: NO
Train/Test pooling           : NO
PCA refit                    : NO
PCA reprojection             : NO
score-threshold tuning       : NO
Step24 morphology changed    : NO

Next
----
25B:
  construct / measure frozen AGGC PC6 tumor-localization input

25C:
  frozen AUC / bootstrap evaluation

25D:
  final evidence freeze
"""

    (args.output / "README.txt").write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final audit.
    # -------------------------------------------------------------------------

    banner("STEP25A FREEZE-ONLY AUDIT")

    audit = [
        ("Step06 discovery-frozen PC6 direction inherited", "YES"),
        ("Step24D external morphology freeze inherited", "YES"),
        ("PC6 raw direction retained", "YES"),
        ("AGGC Test primary role frozen", "YES"),
        ("AGGC Train secondary role frozen", "YES"),
        ("Tumor = G3/G4/G5 frozen", "YES"),
        ("Non-tumor = Normal/Stroma frozen", "YES"),
        ("Primary unit = slide frozen", "YES"),
        ("Primary endpoint = median per-slide AUC frozen", "YES"),
        ("Bootstrap 10,000 / slide frozen", "YES"),
        ("Tumor AUC calculated", "NO"),
        ("Historical PC6 AUC values read", "NO"),
        ("PC ranking", "NO"),
        ("PC screening", "NO"),
        ("PC6 sign tuning", "NO"),
        ("Tumor-definition tuning", "NO"),
        ("Test subset selected by outcome", "NO"),
        ("Train/Test pooled", "NO"),
        ("PCA refit", "NO"),
        ("PCA reprojection", "NO"),
        ("Score threshold tuned", "NO"),
        ("Step24 morphology redefined", "NO"),
    ]

    for item, status in audit:
        print(f"{item:<66}: {status}")

    print()
    print("[OK] Saved:")

    for p in sorted(args.output.iterdir()):
        print(f"  {p}")

    print()
    print(
        "[DONE] Step25A AGGC PC6 external tumor-localization specification frozen."
    )
    print(
        "[NEXT] Step25B may construct the frozen Test/Train PC6 tumor-localization "
        "measurement input without calculating AUC."
    )


if __name__ == "__main__":
    main()
