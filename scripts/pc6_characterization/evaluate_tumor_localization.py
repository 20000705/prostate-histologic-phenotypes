#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 25C
========
EVALUATE FROZEN AGGC SUBSET1 TEST PC6 TUMOR-LOCALIZATION AUC

VERSION
-------
1.0.0

SCHEMA
------
25c-frozen-aggc-test-pc6-tumor-localization-auc-schema-1.0

ROLE
----
FORMAL FROZEN INFERENCE ONLY.

AUTHORITATIVE INPUT
-------------------
Step25B2:
    results/25b2_frozen_aggc_test_pc6_auc_input_from_pc3_ground_truth/
        aggc_frozen_test_pc6_auc_input.csv.gz
        aggc_frozen_test_pc6_auc_slide_summary.csv
        analysis_metadata.json

This input already freezes:
    - exact 35-slide Step21B3 held-out Test cohort;
    - exact Step21B2 five-class ground truth;
    - exact tile identity;
    - tumor = G3/G4/G5;
    - non-tumor = Normal/Stroma;
    - frozen raw PC6 = pca_targets[tile_index,5];
    - frozen direction = high_in_tumor.

PRIMARY SCORE
-------------
+PC6

The sign is NOT chosen here.

PRIMARY PER-SLIDE ENDPOINT
--------------------------
ROC AUC for:
    binary_tumor_target vs +PC6

Independent external reporting unit:
    slide

PRIMARY PARTITION-LEVEL SUMMARY
-------------------------------
Across the frozen 35 held-out Test slides:
    - median per-slide ROC AUC;
    - IQR;
    - mean;
    - number/fraction with AUC > 0.5;
    - number/fraction with AUC >= 0.7;
    - slide-bootstrap 95% CI of the median per-slide AUC.

OPTIONAL SUPPORTIVE SUMMARY
---------------------------
A pooled tile-level AUC may be reported as DESCRIPTIVE ONLY if explicitly
allowed by the frozen Step25A endpoint specification. It is never the primary
external inference because tiles are clustered within slides.

BOOTSTRAP
---------
Resampling unit:
    slide

No tile bootstrap is used for the primary CI.

FIREWALL
--------
cohort change                 : NO
tile inclusion change         : NO
ground-truth change           : NO
Normal/Stroma regeneration    : NO
unlabeled -> negative         : NO
PC6 sign tuning               : NO
threshold tuning              : NO
PC screening                  : NO
PC ranking                    : NO
PCA refit                     : NO
PCA reprojection              : NO
Train/Test pooling            : NO
outcome-based exclusion       : NO
post-hoc slide removal        : NO
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "25c-frozen-aggc-test-pc6-tumor-localization-auc-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT

STEP25A = RESULTS / "25a_aggc_pc6_external_tumor_localization_specification"
STEP25B2 = RESULTS / "25b2_frozen_aggc_test_pc6_auc_input_from_pc3_ground_truth"

A_PLAN = STEP25A / "aggc_pc6_tumor_localization_plan.json"
A_LABELS = STEP25A / "aggc_pc6_tumor_label_definition.csv"
A_PARTITIONS = STEP25A / "aggc_pc6_tumor_localization_partition_specification.csv"
A_ENDPOINTS = STEP25A / "aggc_pc6_tumor_localization_endpoint_specification.csv"
A_BOOTSTRAP = STEP25A / "aggc_pc6_tumor_localization_bootstrap_specification.csv"
A_METADATA = STEP25A / "analysis_metadata.json"
A_CLAIM = STEP25A / "pc6_aggc_tumor_localization_claim_ceiling.txt"

B2_TILE = STEP25B2 / "aggc_frozen_test_pc6_auc_input.csv.gz"
B2_SLIDE = STEP25B2 / "aggc_frozen_test_pc6_auc_slide_summary.csv"
B2_ALIGNMENT = STEP25B2 / "aggc_frozen_test_pc6_auc_alignment_audit.csv"
B2_COHORT_AUDIT = STEP25B2 / "aggc_frozen_test_pc6_auc_cohort_audit.csv"
B2_METADATA = STEP25B2 / "analysis_metadata.json"
B2_HASHES = STEP25B2 / "source_hashes.csv"

DEFAULT_OUT = RESULTS / "25c_frozen_aggc_test_pc6_tumor_localization_auc"

EXPECTED_SLIDES = 35
EXPECTED_DIRECTION = "high_in_tumor"
PRIMARY_SCORE_COLUMN = "PC6"
TARGET_COLUMN = "binary_tumor_target"

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260811


# =============================================================================
# OUTPUTS
# =============================================================================

O_PER_SLIDE = "aggc_test_pc6_per_slide_tumor_localization_auc.csv"
O_SUMMARY = "aggc_test_pc6_tumor_localization_auc_summary.csv"
O_BOOTSTRAP = "aggc_test_pc6_median_auc_slide_bootstrap.csv.gz"
O_INPUT_AUDIT = "aggc_test_pc6_auc_input_integrity_audit.csv"
O_HASHES = "source_hashes.csv"
O_METADATA = "analysis_metadata.json"
O_README = "README.txt"


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
                "Use --overwrite only for intentional rerun under the unchanged frozen contract."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def test_number(slide_id: str) -> int:
    return int(str(slide_id).rsplit("_", 1)[1])


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


def percentile_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(x, alpha / 2)),
        float(np.quantile(x, 1 - alpha / 2)),
    )


# =============================================================================
# FROZEN CONTRACT VALIDATION
# =============================================================================

def resolve_bootstrap_contract() -> Tuple[int, int, pd.DataFrame]:
    """
    Read the frozen Step25A bootstrap file and resolve resample count/seed
    conservatively. If Step25A does not encode these fields, use script defaults
    and record that fact. No outcome-dependent choice is made.
    """
    require_file(A_BOOTSTRAP, "Step25A bootstrap specification")
    df = pd.read_csv(A_BOOTSTRAP, low_memory=False)

    n_boot = DEFAULT_BOOTSTRAP_RESAMPLES
    seed = DEFAULT_BOOTSTRAP_SEED

    # Search exact common field names only.
    for col in [
        "n_resamples", "bootstrap_resamples", "resamples", "n_bootstrap",
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                n_boot = int(vals.iloc[0])
                break

    for col in [
        "random_seed", "bootstrap_seed", "seed",
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                seed = int(vals.iloc[0])
                break

    if n_boot <= 0:
        raise RuntimeError(f"Invalid bootstrap resample count: {n_boot}")

    return n_boot, seed, df


def validate_step25a() -> Dict[str, Any]:
    for p, label in [
        (A_PLAN, "Step25A plan"),
        (A_LABELS, "Step25A label definition"),
        (A_PARTITIONS, "Step25A partition specification"),
        (A_ENDPOINTS, "Step25A endpoint specification"),
        (A_BOOTSTRAP, "Step25A bootstrap specification"),
        (A_METADATA, "Step25A metadata"),
        (A_CLAIM, "Step25A claim ceiling"),
    ]:
        require_file(p, label)

    plan = load_json(A_PLAN)
    meta = load_json(A_METADATA)

    if str(plan.get("target_axis")) not in {"PC6", "Virchow2_PC6"}:
        raise RuntimeError("Step25A target axis is not PC6.")

    if str(plan.get("frozen_pc6_direction")) != EXPECTED_DIRECTION:
        raise RuntimeError(
            "Step25A frozen direction is not high_in_tumor."
        )

    if str(plan.get("primary_partition")) != "Subset1_Test":
        raise RuntimeError(
            "Step25A primary partition is not Subset1_Test."
        )

    labels = pd.read_csv(A_LABELS, low_memory=False)
    required = {"histologic_label", "binary_tumor_target"}
    missing = required - set(labels.columns)
    if missing:
        raise RuntimeError(
            f"Step25A label definition missing columns: {sorted(missing)}"
        )

    positive = set(
        labels.loc[
            pd.to_numeric(labels["binary_tumor_target"], errors="coerce").eq(1),
            "histologic_label",
        ].astype(str)
    )
    negative = set(
        labels.loc[
            pd.to_numeric(labels["binary_tumor_target"], errors="coerce").eq(0),
            "histologic_label",
        ].astype(str)
    )

    if positive != {"G3", "G4", "G5"}:
        raise RuntimeError(f"Step25A positive classes changed: {positive}")
    if negative != {"Normal", "Stroma"}:
        raise RuntimeError(f"Step25A negative classes changed: {negative}")

    freeze = meta.get("freeze_only_audit", {})
    forbidden_true = [
        "tumor_auc_calculated",
        "historical_pc6_auc_values_read",
        "pc_ranking",
        "pc_screening",
        "pc6_sign_tuned",
        "tumor_definition_tuned",
        "test_subset_selected_by_outcome",
        "train_test_pooled",
        "pca_refit",
        "pca_reprojection",
        "threshold_tuned",
        "step24_morphology_redefined",
    ]
    bad = [k for k in forbidden_true if bool(freeze.get(k, False))]
    if bad:
        raise RuntimeError(
            "Step25A freeze-only firewall violated:\n" + "\n".join(bad)
        )

    n_boot, seed, boot_df = resolve_bootstrap_contract()

    return {
        "plan": plan,
        "metadata": meta,
        "n_boot": n_boot,
        "seed": seed,
        "bootstrap_table": boot_df,
    }


def validate_step25b2() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    for p, label in [
        (B2_TILE, "Step25B2 frozen tile input"),
        (B2_SLIDE, "Step25B2 slide summary"),
        (B2_ALIGNMENT, "Step25B2 alignment audit"),
        (B2_COHORT_AUDIT, "Step25B2 cohort audit"),
        (B2_METADATA, "Step25B2 metadata"),
        (B2_HASHES, "Step25B2 source hashes"),
    ]:
        require_file(p, label)

    tiles = pd.read_csv(B2_TILE, low_memory=False)
    slides = pd.read_csv(B2_SLIDE, low_memory=False)
    align = pd.read_csv(B2_ALIGNMENT, low_memory=False)
    cohort_audit = pd.read_csv(B2_COHORT_AUDIT, low_memory=False)
    meta = load_json(B2_METADATA)

    required_tile_cols = {
        "partition",
        "formal_partition_role",
        "slide_id",
        "tile_index",
        "x",
        "y",
        "histologic_class",
        TARGET_COLUMN,
        PRIMARY_SCORE_COLUMN,
        "pc6_direction",
        "included_in_auc",
    }
    missing = required_tile_cols - set(tiles.columns)
    if missing:
        raise RuntimeError(
            f"Step25B2 tile input missing columns: {sorted(missing)}"
        )

    if tiles.duplicated(["slide_id", "tile_index"]).any():
        raise RuntimeError(
            "Step25B2 frozen tile source contains duplicate slide_id+tile_index."
        )

    if slides["slide_id"].duplicated().any():
        raise RuntimeError(
            "Step25B2 slide summary contains duplicate slide_id."
        )

    if len(slides) != EXPECTED_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_SLIDES} Step25B2 slides; observed={len(slides)}"
        )

    tile_slide_ids = set(tiles["slide_id"].astype(str))
    summary_slide_ids = set(slides["slide_id"].astype(str))

    if tile_slide_ids != summary_slide_ids:
        raise RuntimeError(
            "Step25B2 tile and slide-summary slide sets differ."
        )

    if tiles["partition"].astype(str).nunique() != 1 or \
       not tiles["partition"].astype(str).eq("Subset1_Test").all():
        raise RuntimeError(
            "Step25B2 input is not exclusively Subset1_Test."
        )

    if not tiles["pc6_direction"].astype(str).eq(EXPECTED_DIRECTION).all():
        raise RuntimeError(
            "Step25B2 PC6 direction is not uniformly high_in_tumor."
        )

    included = tiles["included_in_auc"].astype(bool)
    if not included.all():
        raise RuntimeError(
            "Step25B2 frozen input contains rows not included in the AUC input."
        )

    target = finite_numeric(tiles[TARGET_COLUMN])
    score = finite_numeric(tiles[PRIMARY_SCORE_COLUMN])

    if not np.isfinite(target).all():
        raise RuntimeError(
            "Step25B2 target contains non-finite values."
        )
    if set(np.unique(target)) != {0.0, 1.0}:
        raise RuntimeError(
            f"Step25B2 target is not binary 0/1: {np.unique(target)}"
        )
    if not np.isfinite(score).all():
        raise RuntimeError(
            "Step25B2 PC6 score contains non-finite values."
        )

    class_target = (
        tiles[["histologic_class", TARGET_COLUMN]]
        .drop_duplicates()
        .sort_values("histologic_class")
    )
    expected_map = {
        "G3": 1,
        "G4": 1,
        "G5": 1,
        "Normal": 0,
        "Stroma": 0,
    }
    observed_map = dict(
        zip(
            class_target["histologic_class"].astype(str),
            pd.to_numeric(class_target[TARGET_COLUMN], errors="raise").astype(int),
        )
    )

    if observed_map != expected_map:
        raise RuntimeError(
            f"Step25B2 class/target mapping changed:\n{observed_map}"
        )

    if "support_10_10_met" not in slides.columns or \
       not slides["support_10_10_met"].astype(bool).all():
        raise RuntimeError(
            "At least one Step25B2 slide no longer satisfies frozen 10/10 support."
        )

    if "alignment_pass" not in align.columns or \
       not align["alignment_pass"].astype(bool).all():
        raise RuntimeError(
            "At least one Step25B2 alignment audit failed."
        )

    if "pass" not in cohort_audit.columns or \
       not cohort_audit["pass"].astype(bool).all():
        raise RuntimeError(
            "Step25B2 cohort audit is not fully PASS."
        )

    if int(meta.get("cohort", {}).get("observed_slides", -1)) != EXPECTED_SLIDES:
        raise RuntimeError(
            "Step25B2 metadata cohort size changed."
        )

    if bool(meta.get("ground_truth", {}).get("normal_stroma_regenerated", True)):
        raise RuntimeError(
            "Step25B2 metadata indicates Normal/Stroma regeneration."
        )

    if bool(meta.get("ground_truth", {}).get("unlabeled_treated_as_negative", True)):
        raise RuntimeError(
            "Step25B2 metadata indicates unlabeled->negative."
        )

    firewall = meta.get("firewall", {})
    forbidden_true = [
        "auc_calculated",
        "bootstrap_calculated",
        "p_values_calculated",
        "fdr_calculated",
        "threshold_tuned",
        "pc_ranking",
        "pc_screening",
        "pc6_sign_tuned",
        "pca_refit",
        "pca_reprojection",
        "label_regeneration",
        "normal_stroma_regeneration",
        "unlabeled_to_negative",
        "manual_slide_selection",
        "outcome_based_exclusion",
        "train_test_pooled",
        "approximate_coordinate_matching",
    ]
    bad = [k for k in forbidden_true if bool(firewall.get(k, False))]
    if bad:
        raise RuntimeError(
            "Step25B2 firewall violated:\n" + "\n".join(bad)
        )

    audit_rows = [
        ("frozen_slide_count", len(slides), EXPECTED_SLIDES, len(slides) == EXPECTED_SLIDES),
        ("duplicate_slide_tile_rows", int(tiles.duplicated(["slide_id","tile_index"]).sum()), 0, not tiles.duplicated(["slide_id","tile_index"]).any()),
        ("all_rows_subset1_test", bool(tiles["partition"].astype(str).eq("Subset1_Test").all()), True, bool(tiles["partition"].astype(str).eq("Subset1_Test").all())),
        ("all_pc6_direction_high_in_tumor", bool(tiles["pc6_direction"].astype(str).eq(EXPECTED_DIRECTION).all()), True, bool(tiles["pc6_direction"].astype(str).eq(EXPECTED_DIRECTION).all())),
        ("all_rows_auc_included", bool(included.all()), True, bool(included.all())),
        ("all_targets_binary_finite", bool(np.isfinite(target).all() and set(np.unique(target)) == {0.0,1.0}), True, bool(np.isfinite(target).all() and set(np.unique(target)) == {0.0,1.0})),
        ("all_pc6_finite", bool(np.isfinite(score).all()), True, bool(np.isfinite(score).all())),
        ("all_slides_support_10_10", bool(slides["support_10_10_met"].astype(bool).all()), True, bool(slides["support_10_10_met"].astype(bool).all())),
        ("all_alignment_pass", bool(align["alignment_pass"].astype(bool).all()), True, bool(align["alignment_pass"].astype(bool).all())),
        ("all_cohort_audit_pass", bool(cohort_audit["pass"].astype(bool).all()), True, bool(cohort_audit["pass"].astype(bool).all())),
    ]

    audit = pd.DataFrame(
        audit_rows,
        columns=["audit_item", "observed", "expected", "pass"],
    )

    return tiles, slides, meta, audit


# =============================================================================
# AUC
# =============================================================================

def calculate_per_slide_auc(tile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    order = sorted(
        tile_df["slide_id"].astype(str).unique(),
        key=test_number,
    )

    for slide_id in order:
        sub = tile_df.loc[
            tile_df["slide_id"].astype(str).eq(slide_id)
        ].copy()

        y = finite_numeric(sub[TARGET_COLUMN])
        score = finite_numeric(sub[PRIMARY_SCORE_COLUMN])

        if not np.isfinite(y).all() or not np.isfinite(score).all():
            raise RuntimeError(
                f"{slide_id}: non-finite target or PC6 reached formal inference."
            )

        n_tumor = int(np.sum(y == 1))
        n_nontumor = int(np.sum(y == 0))

        if n_tumor < 10 or n_nontumor < 10:
            raise RuntimeError(
                f"{slide_id}: frozen support rule violated at inference: "
                f"tumor={n_tumor}, non_tumor={n_nontumor}"
            )

        auc = float(roc_auc_score(y, score))

        rows.append({
            "slide_id": slide_id,
            "partition": "Subset1_Test",
            "formal_partition_role": "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION",
            "score_definition": "+PC6",
            "frozen_pc6_direction": EXPECTED_DIRECTION,
            "n_auc_tiles": int(len(sub)),
            "n_tumor_tiles": n_tumor,
            "n_nontumor_tiles": n_nontumor,
            "roc_auc": auc,
            "auc_above_0_5": bool(auc > 0.5),
            "auc_at_least_0_7": bool(auc >= 0.7),
            "sign_selected_from_aggc_outcome": False,
        })

        print(
            f"{slide_id:<16} | tiles={len(sub):>6,} | "
            f"tumor={n_tumor:>6,} | non_tumor={n_nontumor:>6,} | "
            f"AUC={auc:.6f}"
        )

    out = pd.DataFrame(rows)

    if len(out) != EXPECTED_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_SLIDES} per-slide AUCs; observed={len(out)}"
        )

    return out


# =============================================================================
# SLIDE-LEVEL BOOTSTRAP / SUMMARY
# =============================================================================

def slide_bootstrap_median_auc(
    per_slide: pd.DataFrame,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    aucs = finite_numeric(per_slide["roc_auc"])

    if len(aucs) != EXPECTED_SLIDES or not np.isfinite(aucs).all():
        raise RuntimeError(
            "Bootstrap requires 35 finite frozen per-slide AUC values."
        )

    rng = np.random.default_rng(seed)
    n = len(aucs)
    rows = []

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = aucs[idx]

        rows.append({
            "bootstrap_id": i,
            "n_resampled_slides": n,
            "median_auc": float(np.median(sample)),
            "mean_auc": float(np.mean(sample)),
            "fraction_auc_above_0_5": float(np.mean(sample > 0.5)),
            "fraction_auc_at_least_0_7": float(np.mean(sample >= 0.7)),
        })

    return pd.DataFrame(rows)


def build_summary(
    per_slide: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    aucs = finite_numeric(per_slide["roc_auc"])

    med = float(np.median(aucs))
    q1 = float(np.quantile(aucs, 0.25))
    q3 = float(np.quantile(aucs, 0.75))
    mean = float(np.mean(aucs))

    ci_lo, ci_hi = percentile_ci(
        finite_numeric(bootstrap["median_auc"])
    )

    mean_lo, mean_hi = percentile_ci(
        finite_numeric(bootstrap["mean_auc"])
    )

    frac05 = float(np.mean(aucs > 0.5))
    frac07 = float(np.mean(aucs >= 0.7))

    frac05_lo, frac05_hi = percentile_ci(
        finite_numeric(bootstrap["fraction_auc_above_0_5"])
    )

    frac07_lo, frac07_hi = percentile_ci(
        finite_numeric(bootstrap["fraction_auc_at_least_0_7"])
    )

    return pd.DataFrame([
        {
            "endpoint": "median_per_slide_roc_auc",
            "endpoint_role": "PRIMARY",
            "partition": "Subset1_Test",
            "n_slides": len(aucs),
            "estimate": med,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "q1": q1,
            "q3": q3,
            "n_slides_auc_above_0_5": int(np.sum(aucs > 0.5)),
            "fraction_slides_auc_above_0_5": frac05,
            "n_slides_auc_at_least_0_7": int(np.sum(aucs >= 0.7)),
            "fraction_slides_auc_at_least_0_7": frac07,
            "score_definition": "+PC6",
            "resampling_unit": "slide",
        },
        {
            "endpoint": "mean_per_slide_roc_auc",
            "endpoint_role": "SUPPORTIVE",
            "partition": "Subset1_Test",
            "n_slides": len(aucs),
            "estimate": mean,
            "ci_low": mean_lo,
            "ci_high": mean_hi,
            "q1": np.nan,
            "q3": np.nan,
            "n_slides_auc_above_0_5": int(np.sum(aucs > 0.5)),
            "fraction_slides_auc_above_0_5": frac05,
            "n_slides_auc_at_least_0_7": int(np.sum(aucs >= 0.7)),
            "fraction_slides_auc_at_least_0_7": frac07,
            "score_definition": "+PC6",
            "resampling_unit": "slide",
        },
        {
            "endpoint": "fraction_slides_auc_above_0_5",
            "endpoint_role": "SUPPORTIVE",
            "partition": "Subset1_Test",
            "n_slides": len(aucs),
            "estimate": frac05,
            "ci_low": frac05_lo,
            "ci_high": frac05_hi,
            "q1": np.nan,
            "q3": np.nan,
            "n_slides_auc_above_0_5": int(np.sum(aucs > 0.5)),
            "fraction_slides_auc_above_0_5": frac05,
            "n_slides_auc_at_least_0_7": int(np.sum(aucs >= 0.7)),
            "fraction_slides_auc_at_least_0_7": frac07,
            "score_definition": "+PC6",
            "resampling_unit": "slide",
        },
        {
            "endpoint": "fraction_slides_auc_at_least_0_7",
            "endpoint_role": "SUPPORTIVE",
            "partition": "Subset1_Test",
            "n_slides": len(aucs),
            "estimate": frac07,
            "ci_low": frac07_lo,
            "ci_high": frac07_hi,
            "q1": np.nan,
            "q3": np.nan,
            "n_slides_auc_above_0_5": int(np.sum(aucs > 0.5)),
            "fraction_slides_auc_above_0_5": frac05,
            "n_slides_auc_at_least_0_7": int(np.sum(aucs >= 0.7)),
            "fraction_slides_auc_at_least_0_7": frac07,
            "score_definition": "+PC6",
            "resampling_unit": "slide",
        },
    ])


# =============================================================================
# PROVENANCE
# =============================================================================

def source_hashes() -> pd.DataFrame:
    rows = []
    for role, path in [
        ("step25a_plan", A_PLAN),
        ("step25a_label_definition", A_LABELS),
        ("step25a_partition_specification", A_PARTITIONS),
        ("step25a_endpoint_specification", A_ENDPOINTS),
        ("step25a_bootstrap_specification", A_BOOTSTRAP),
        ("step25a_metadata", A_METADATA),
        ("step25a_claim_ceiling", A_CLAIM),
        ("step25b2_frozen_tile_input", B2_TILE),
        ("step25b2_slide_summary", B2_SLIDE),
        ("step25b2_alignment_audit", B2_ALIGNMENT),
        ("step25b2_cohort_audit", B2_COHORT_AUDIT),
        ("step25b2_metadata", B2_METADATA),
        ("step25b2_source_hashes", B2_HASHES),
    ]:
        rows.append({
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate the frozen AGGC Test PC6 per-slide tumor-localization "
            "ROC AUC without changing cohort, labels, score orientation, or PCA."
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


def main() -> None:
    args = parse_args()

    banner("STEP 25C: EVALUATE FROZEN AGGC TEST PC6 TUMOR-LOCALIZATION AUC")

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP25A]       {STEP25A}")
    print(f"[STEP25B2]      {STEP25B2}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          FORMAL FROZEN INFERENCE ONLY")
    print("[PARTITION]     Subset1_Test ONLY")
    print("[N SLIDES]      35 FROZEN")
    print("[GROUND TRUTH]  STEP21B2 via STEP25B2")
    print("[SCORE]         +PC6")
    print("[DIRECTION]     high_in_tumor")
    print("[PRIMARY]       MEDIAN PER-SLIDE ROC AUC")
    print("[BOOTSTRAP]     SLIDE LEVEL")
    print("[SIGN TUNING]   NONE")
    print("[THRESHOLD]     NONE")
    print("[PC SCREEN]     NONE")
    print("[PCA REFIT]     NONE")
    print("[COHORT CHANGE] NONE")
    print("[LABEL CHANGE]  NONE")
    print("[TRAIN/TEST]    NEVER POOLED")

    prepare_output(args.output, args.overwrite)

    banner("VALIDATE FROZEN STEP25A / STEP25B2 CONTRACT")

    contract = validate_step25a()
    tile_df, slide_df, b2_meta, integrity = validate_step25b2()

    print("Step25A target PC6                    : PASS")
    print("Step25A direction high_in_tumor       : PASS")
    print("Step25A Test primary role             : PASS")
    print("Step25B2 frozen 35-slide cohort       : PASS")
    print("Step25B2 exact inherited ground truth : PASS")
    print("Step25B2 exact PC6 input               : PASS")
    print(f"Bootstrap resamples                    : {contract['n_boot']}")
    print(f"Bootstrap seed                         : {contract['seed']}")

    integrity.to_csv(
        args.output / O_INPUT_AUDIT,
        index=False,
    )

    if not integrity["pass"].astype(bool).all():
        raise RuntimeError(
            "Input integrity audit failed:\n"
            + integrity.loc[~integrity["pass"].astype(bool)].to_string(index=False)
        )

    banner("FORMAL PER-SLIDE PC6 ROC AUC")

    per_slide = calculate_per_slide_auc(tile_df)

    banner("SLIDE-LEVEL BOOTSTRAP")

    bootstrap = slide_bootstrap_median_auc(
        per_slide,
        n_resamples=contract["n_boot"],
        seed=contract["seed"],
    )

    summary = build_summary(
        per_slide,
        bootstrap,
    )

    banner("FROZEN TEST PC6 AUC SUMMARY")

    print(summary.to_string(index=False))

    primary = summary.loc[
        summary["endpoint"].eq("median_per_slide_roc_auc")
    ].iloc[0]

    print()
    print(
        f"[PRIMARY] Median per-slide AUC = {primary['estimate']:.6f} "
        f"(95% slide-bootstrap CI {primary['ci_low']:.6f} to {primary['ci_high']:.6f})"
    )
    print(
        f"[IQR]     {primary['q1']:.6f} to {primary['q3']:.6f}"
    )
    print(
        f"[>0.5]    {int(primary['n_slides_auc_above_0_5'])}/{EXPECTED_SLIDES} "
        f"({primary['fraction_slides_auc_above_0_5']:.3f})"
    )
    print(
        f"[>=0.7]   {int(primary['n_slides_auc_at_least_0_7'])}/{EXPECTED_SLIDES} "
        f"({primary['fraction_slides_auc_at_least_0_7']:.3f})"
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    per_slide_path = args.output / O_PER_SLIDE
    summary_path = args.output / O_SUMMARY
    bootstrap_path = args.output / O_BOOTSTRAP
    hashes_path = args.output / O_HASHES
    metadata_path = args.output / O_METADATA
    readme_path = args.output / O_README

    per_slide.to_csv(
        per_slide_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
        compression="gzip",
    )

    source_hashes().to_csv(
        hashes_path,
        index=False,
    )

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_metric": "roc_auc_score",
        "role": "FORMAL_FROZEN_AGGC_TEST_PC6_TUMOR_LOCALIZATION_INFERENCE_ONLY",
        "authorities": {
            "step25a": str(STEP25A),
            "step25b2": str(STEP25B2),
            "frozen_tile_input": str(B2_TILE),
        },
        "partition": {
            "name": "Subset1_Test",
            "scientific_role": "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION",
            "n_slides": EXPECTED_SLIDES,
        },
        "endpoint": {
            "target": "binary_tumor_target",
            "positive_classes": ["G3", "G4", "G5"],
            "negative_classes": ["Normal", "Stroma"],
            "score": "+PC6",
            "pc6_direction": EXPECTED_DIRECTION,
            "per_slide_metric": "ROC AUC",
            "primary_partition_summary": "median per-slide ROC AUC",
        },
        "bootstrap": {
            "resampling_unit": "slide",
            "n_resamples": int(contract["n_boot"]),
            "seed": int(contract["seed"]),
            "ci": "percentile 95%",
            "tile_bootstrap_used_for_primary_ci": False,
        },
        "observed_results": {
            "median_per_slide_auc": float(primary["estimate"]),
            "median_auc_ci_low": float(primary["ci_low"]),
            "median_auc_ci_high": float(primary["ci_high"]),
            "q1": float(primary["q1"]),
            "q3": float(primary["q3"]),
            "n_slides_auc_above_0_5": int(primary["n_slides_auc_above_0_5"]),
            "fraction_slides_auc_above_0_5": float(primary["fraction_slides_auc_above_0_5"]),
            "n_slides_auc_at_least_0_7": int(primary["n_slides_auc_at_least_0_7"]),
            "fraction_slides_auc_at_least_0_7": float(primary["fraction_slides_auc_at_least_0_7"]),
        },
        "firewall": {
            "cohort_changed": False,
            "tile_inclusion_changed": False,
            "ground_truth_changed": False,
            "normal_stroma_regenerated": False,
            "unlabeled_to_negative": False,
            "pc6_sign_tuned": False,
            "threshold_tuned": False,
            "pc_screening": False,
            "pc_ranking": False,
            "pca_refit": False,
            "pca_reprojection": False,
            "train_test_pooled": False,
            "outcome_based_exclusion": False,
            "posthoc_slide_removal": False,
        },
        "next_step": (
            "Step25D may assemble the external PC6 tumor-localization evidence "
            "and, if prespecified, compare PC3 and PC6 on the identical frozen 35-slide cohort."
        ),
    }

    save_json(
        metadata_path,
        metadata,
    )

    readme = f"""STEP 25C
EVALUATE FROZEN AGGC TEST PC6 TUMOR-LOCALIZATION AUC
====================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

ROLE
----
FORMAL FROZEN INFERENCE ONLY.

INPUT
-----
{B2_TILE}

Frozen held-out Test slides:
  {EXPECTED_SLIDES}

Ground truth:
  Tumor     = G3 / G4 / G5
  Non-tumor = Normal / Stroma

Score:
  +PC6

Direction:
  high_in_tumor

No sign selection is performed in Step25C.

PRIMARY ENDPOINT
----------------
Per-slide ROC AUC.

Primary cohort summary:
  median per-slide ROC AUC

Observed:
  median = {float(primary["estimate"]):.6f}
  95% slide-bootstrap CI = [{float(primary["ci_low"]):.6f}, {float(primary["ci_high"]):.6f}]
  IQR = [{float(primary["q1"]):.6f}, {float(primary["q3"]):.6f}]

Slides with AUC > 0.5:
  {int(primary["n_slides_auc_above_0_5"])}/{EXPECTED_SLIDES}

Slides with AUC >= 0.7:
  {int(primary["n_slides_auc_at_least_0_7"])}/{EXPECTED_SLIDES}

BOOTSTRAP
---------
Resampling unit:
  slide

Resamples:
  {contract["n_boot"]}

Seed:
  {contract["seed"]}

Primary CI:
  percentile 95% CI of median per-slide AUC

FIREWALL
--------
cohort changed               : NO
tile inclusion changed       : NO
ground truth changed         : NO
Normal/Stroma regenerated    : NO
unlabeled -> negative        : NO
PC6 sign tuned               : NO
threshold tuned              : NO
PC screening                 : NO
PC ranking                   : NO
PCA refit                    : NO
PCA reprojection             : NO
Train/Test pooled            : NO
outcome-based exclusion      : NO
post-hoc slide removal       : NO

NEXT
----
Step25D may assemble/freeze the final external PC6 tumor-localization evidence.
"""

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    banner("STEP25C FINAL INFERENCE FIREWALL")

    for item, status in [
        ("Frozen 35-slide Test cohort retained", "YES"),
        ("Frozen Step21B2 ground truth retained", "YES"),
        ("Frozen +PC6 score retained", "YES"),
        ("Per-slide ROC AUC calculated", "YES"),
        ("Slide-level bootstrap calculated", "YES"),
        ("Cohort changed", "NO"),
        ("Tile inclusion changed", "NO"),
        ("Ground truth changed", "NO"),
        ("Normal/Stroma regenerated", "NO"),
        ("Unlabeled treated as negative", "NO"),
        ("PC6 sign tuning", "NO"),
        ("Threshold tuning", "NO"),
        ("PC screening", "NO"),
        ("PC ranking", "NO"),
        ("PCA refit", "NO"),
        ("PCA reprojection", "NO"),
        ("Train/Test pooling", "NO"),
        ("Outcome-based exclusion", "NO"),
        ("Post-hoc slide removal", "NO"),
    ]:
        print(f"{item:<66}: {status}")

    print()
    print("[OK] Saved:")
    for path in [
        per_slide_path,
        summary_path,
        bootstrap_path,
        args.output / O_INPUT_AUDIT,
        hashes_path,
        metadata_path,
        readme_path,
    ]:
        if not path.exists():
            raise RuntimeError(f"Expected output missing: {path}")
        print(f"  {path}")

    print()
    print("[DONE] Step25C frozen AGGC Test PC6 tumor-localization AUC evaluation complete.")
    print("[NEXT] Review the frozen Test-primary result before any PC3-vs-PC6 comparative assembly.")


if __name__ == "__main__":
    main()
