#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
31b_evaluate_combined_aggc_pc3_associations.py

STEP 31B
========
EVALUATE COMBINED AGGC PC3 ASSOCIATIONS

VERSION
-------
1.0.0

ROLE
----
COMBINED COHORT INFERENCE FROM FROZEN STEP31A TABLES.

PURPOSE
-------
Recalculate AGGC cohort-level summaries after combining the dataset-provided
Train and Test partitions into a single external cohort.

IMPORTANT
---------
This step changes ONLY the cohort aggregation used for manuscript reporting.
It does NOT change any scientific definition established upstream.

It MUST NOT:
    - refit PCA;
    - reproject embeddings;
    - change PC3 sign;
    - redefine tumor/non-tumor classes;
    - redefine histologic states;
    - change support thresholds;
    - select slides using outcomes;
    - pool tiles across slides for inference;
    - treat Train/Test labels as model-development labels.

Independent inferential unit:
    slide

Combined AGGC analyses
----------------------
1. Histologic-state available-case summaries at frozen threshold >=50 tiles/state.
2. Paired G4-G3 slide-level contrast:
       median PC3(G4) - median PC3(G3)
3. Stroma-Normal and Normal-G3 paired slide-level contrasts when both states
   are available in the same slide.
4. G5-G4 exploratory paired contrast.
5. Tumor-localization AUC summary using existing frozen per-slide AUC values.
6. Descriptive five-state cohort medians based on slide-level state summaries.
7. Within-state morphology table coverage summary from frozen Step20B outputs.

Statistical procedures
----------------------
For paired slide-level contrasts:
    median effect
    mean effect
    Q25 / Q75
    95% bootstrap CI of median over slides
    two-sided Wilcoxon signed-rank test vs zero
    fraction effect > 0
    fraction effect < 0

For tumor-localization AUC:
    n slides
    median / mean / Q25 / Q75
    95% bootstrap CI of median over slides
    fraction AUC > 0.5
    fraction AUC >= 0.7
    fraction AUC >= 0.8

Bootstrap:
    10,000 slide-level resamples
    seed = 20260830

No pooled tile-level inference is performed.

INPUT
-----
results/31a_combined_aggc_pc3_analysis/

OUTPUT
------
results/31b_combined_aggc_pc3_associations/

Files:
    combined_aggc_pc3_state_descriptive_summary.csv
    combined_aggc_pc3_paired_state_contrasts_per_slide.csv
    combined_aggc_pc3_paired_state_contrasts_summary.csv
    combined_aggc_pc3_paired_state_contrasts_bootstrap.csv.gz
    combined_aggc_pc3_tumor_auc_summary.csv
    combined_aggc_pc3_tumor_auc_bootstrap.csv.gz
    combined_aggc_pc3_morphology_coverage_summary.csv
    analysis_audit.csv
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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


VERSION = "1.0.2"
SCHEMA = "31b-combined-aggc-pc3-association-schema-1.0.2"

MEDIA = REPO_ROOT
RESULTS = RESULTS_ROOT

STEP31A = RESULTS / "31a_combined_aggc_pc3_analysis"
OUT = RESULTS / "31b_combined_aggc_pc3_associations"

F_UNIVERSE = "combined_aggc_slide_universe.csv"
F_STATE_PRIMARY = "combined_aggc_pc3_primary_state_measurements.csv"
F_AUC = "combined_aggc_pc3_tumor_auc_per_slide.csv"
F_MORPH = "combined_aggc_pc3_within_state_morphology_measurements.csv.gz"
F_COVERAGE = "combined_aggc_endpoint_coverage.csv"
F_AUDIT31A = "combined_aggc_partition_audit.csv"

OUT_STATE_DESC = "combined_aggc_pc3_state_descriptive_summary.csv"
OUT_CONTRAST_SLIDE = "combined_aggc_pc3_paired_state_contrasts_per_slide.csv"
OUT_CONTRAST_SUM = "combined_aggc_pc3_paired_state_contrasts_summary.csv"
OUT_CONTRAST_BOOT = "combined_aggc_pc3_paired_state_contrasts_bootstrap.csv.gz"
OUT_AUC_SUM = "combined_aggc_pc3_tumor_auc_summary.csv"
OUT_AUC_BOOT = "combined_aggc_pc3_tumor_auc_bootstrap.csv.gz"
OUT_MORPH_COVERAGE = "combined_aggc_pc3_morphology_coverage_summary.csv"
OUT_AUDIT = "analysis_audit.csv"
OUT_HASHES = "source_hashes.csv"
OUT_META = "analysis_metadata.json"
OUT_README = "README.txt"

EXPECTED_TOTAL = 150
N_BOOT = 10000
SEED = 20260830

STATE_ORDER = ["Stroma", "Normal", "G3", "G4", "G5"]

CONTRASTS = [
    ("stroma_minus_normal", "Stroma", "Normal", "median_pc3"),
    ("normal_minus_g3", "Normal", "G3", "median_pc3"),
    ("g4_minus_g3", "G4", "G3", "median_pc3"),
    ("g5_minus_g4", "G5", "G4", "median_pc3"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def banner(text: str, width: int = 118) -> None:
    print("\n" + "=" * width)
    print(text)
    print("=" * width)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}:\n{path}")


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
                "Use --overwrite only for intentional regeneration."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_state(x: Any) -> str:
    s = str(x).strip()
    mapping = {
        "stroma": "Stroma",
        "normal": "Normal",
        "g3": "G3",
        "g4": "G4",
        "g5": "G5",
        "Stroma": "Stroma",
        "Normal": "Normal",
        "G3": "G3",
        "G4": "G4",
        "G5": "G5",
    }
    if s in mapping:
        return mapping[s]
    low = s.lower()
    if low in mapping:
        return mapping[low]
    raise RuntimeError(f"Unexpected histologic state: {x!r}")


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError(
            f"Could not resolve required column from {list(candidates)}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def bootstrap_median(
    values: Sequence[float],
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.array([], dtype=float), np.nan, np.nan

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        out[i] = float(np.median(x[idx]))

    lo, hi = np.quantile(out, [0.025, 0.975])
    return out, float(lo), float(hi)


def safe_wilcoxon(values: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, 0

    if np.allclose(x, 0):
        return 0.0, 1.0, len(x)

    try:
        r = wilcoxon(
            x,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )
        return float(r.statistic), float(r.pvalue), len(x)
    except Exception:
        return np.nan, np.nan, len(x)


def summarize_effect(
    name: str,
    values: Sequence[float],
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    boot, ci_lo, ci_hi = bootstrap_median(x, N_BOOT, seed)
    wstat, wp, wn = safe_wilcoxon(x)

    row = {
        "contrast": name,
        "n_slides": int(len(x)),
        "mean_effect": float(np.mean(x)) if len(x) else np.nan,
        "median_effect": float(np.median(x)) if len(x) else np.nan,
        "q25_effect": float(np.quantile(x, 0.25)) if len(x) else np.nan,
        "q75_effect": float(np.quantile(x, 0.75)) if len(x) else np.nan,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "bootstrap_replicates": N_BOOT,
        "wilcoxon_statistic": wstat,
        "wilcoxon_p_two_sided": wp,
        "wilcoxon_n": int(wn),
        "n_effect_gt_0": int(np.sum(x > 0)) if len(x) else 0,
        "fraction_effect_gt_0": float(np.mean(x > 0)) if len(x) else np.nan,
        "n_effect_lt_0": int(np.sum(x < 0)) if len(x) else 0,
        "fraction_effect_lt_0": float(np.mean(x < 0)) if len(x) else np.nan,
    }

    boot_df = pd.DataFrame({
        "contrast": name,
        "bootstrap_index": np.arange(len(boot), dtype=int),
        "bootstrap_median_effect": boot,
    })

    return row, boot_df


def summarize_auc(
    values: Sequence[float],
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    boot, ci_lo, ci_hi = bootstrap_median(x, N_BOOT, seed)

    row = {
        "n_slides": int(len(x)),
        "mean_auc": float(np.mean(x)) if len(x) else np.nan,
        "median_auc": float(np.median(x)) if len(x) else np.nan,
        "q25_auc": float(np.quantile(x, 0.25)) if len(x) else np.nan,
        "q75_auc": float(np.quantile(x, 0.75)) if len(x) else np.nan,
        "min_auc": float(np.min(x)) if len(x) else np.nan,
        "max_auc": float(np.max(x)) if len(x) else np.nan,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "bootstrap_replicates": N_BOOT,
        "n_auc_gt_0_5": int(np.sum(x > 0.5)) if len(x) else 0,
        "fraction_auc_gt_0_5": float(np.mean(x > 0.5)) if len(x) else np.nan,
        "n_auc_ge_0_7": int(np.sum(x >= 0.7)) if len(x) else 0,
        "fraction_auc_ge_0_7": float(np.mean(x >= 0.7)) if len(x) else np.nan,
        "n_auc_ge_0_8": int(np.sum(x >= 0.8)) if len(x) else 0,
        "fraction_auc_ge_0_8": float(np.mean(x >= 0.8)) if len(x) else np.nan,
    }

    boot_df = pd.DataFrame({
        "bootstrap_index": np.arange(len(boot), dtype=int),
        "bootstrap_median_auc": boot,
    })

    return row, boot_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate combined AGGC PC3 associations from Step31A."
    )
    p.add_argument("--step31a", type=Path, default=STEP31A)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    banner("STEP 31B: EVALUATE COMBINED AGGC PC3 ASSOCIATIONS")
    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP31A]       {a.step31a}")
    print(f"[OUTPUT]        {a.output}")
    print("[UNIT]          SLIDE")
    print("[PARTITION USE] COMBINED FOR REPORTING")
    print("[PCA REFIT]     NO")
    print("[PC3 SIGN]      UNCHANGED")

    paths = {
        "universe": a.step31a / F_UNIVERSE,
        "state_primary": a.step31a / F_STATE_PRIMARY,
        "auc": a.step31a / F_AUC,
        "morphology": a.step31a / F_MORPH,
        "coverage": a.step31a / F_COVERAGE,
        "audit31a": a.step31a / F_AUDIT31A,
    }

    for role, path in paths.items():
        require_file(path, role)

    prepare_output(a.output, a.overwrite)

    universe = pd.read_csv(paths["universe"], low_memory=False)
    state = pd.read_csv(paths["state_primary"], low_memory=False)
    auc = pd.read_csv(paths["auc"], low_memory=False)
    morph = pd.read_csv(paths["morphology"], low_memory=False)
    coverage31a = pd.read_csv(paths["coverage"], low_memory=False)
    audit31a = pd.read_csv(paths["audit31a"], low_memory=False)

    # -------------------------------------------------------------------------
    # AUDIT STEP31A
    # -------------------------------------------------------------------------
    if "pass" in audit31a.columns:
        if not audit31a["pass"].astype(bool).all():
            raise RuntimeError("Step31A audit contains failed rows.")

    if "slide_id" not in universe.columns:
        raise RuntimeError("Step31A universe lacks slide_id.")

    if universe["slide_id"].astype(str).nunique() != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Step31A universe is not {EXPECTED_TOTAL} unique slides."
        )

    # -------------------------------------------------------------------------
    # HISTOLOGIC STATE TABLE
    # -------------------------------------------------------------------------
    banner("COMBINED HISTOLOGIC-STATE ANALYSIS")

    slide_col = find_col(state, ["slide_id"])
    state_col = find_col(state, ["histologic_state", "state"])
    median_col = find_col(state, ["median_pc3", "median_PC3"])

    state = state.rename(columns={
        slide_col: "slide_id",
        state_col: "histologic_state",
        median_col: "median_pc3",
    }).copy()

    state["slide_id"] = state["slide_id"].astype(str)
    state["histologic_state"] = state["histologic_state"].map(normalize_state)
    state["median_pc3"] = pd.to_numeric(state["median_pc3"], errors="coerce")

    state = state[np.isfinite(state["median_pc3"])].copy()

    if state.duplicated(["slide_id", "histologic_state"]).any():
        bad = state.loc[
            state.duplicated(["slide_id", "histologic_state"], keep=False)
        ].head(20)
        raise RuntimeError(
            "Duplicate slide-state rows in Step31A primary state table:\n"
            + bad.to_string(index=False)
        )

    # Descriptive cohort medians by state.
    desc_rows = []
    for st in STATE_ORDER:
        x = state.loc[state["histologic_state"].eq(st), "median_pc3"].to_numpy(float)
        x = x[np.isfinite(x)]
        boot, lo, hi = bootstrap_median(x, N_BOOT, SEED + STATE_ORDER.index(st))
        desc_rows.append({
            "histologic_state": st,
            "n_slides": int(len(x)),
            "mean_slide_median_pc3": float(np.mean(x)) if len(x) else np.nan,
            "median_slide_median_pc3": float(np.median(x)) if len(x) else np.nan,
            "q25_slide_median_pc3": float(np.quantile(x, 0.25)) if len(x) else np.nan,
            "q75_slide_median_pc3": float(np.quantile(x, 0.75)) if len(x) else np.nan,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "bootstrap_replicates": N_BOOT,
        })

    desc = pd.DataFrame(desc_rows)
    print(desc.to_string(index=False))

    # Wide slide x state table for paired contrasts.
    wide = (
        state.pivot(index="slide_id", columns="histologic_state", values="median_pc3")
        .reset_index()
    )

    contrast_slide_rows = []
    contrast_summary_rows = []
    contrast_boot_parts = []

    for i, (name, left, right, _) in enumerate(CONTRASTS):
        if left not in wide.columns or right not in wide.columns:
            continue

        z = wide[["slide_id", left, right]].dropna().copy()
        z["contrast"] = name
        z["left_state"] = left
        z["right_state"] = right
        z["left_median_pc3"] = z[left].astype(float)
        z["right_median_pc3"] = z[right].astype(float)
        z["effect"] = z["left_median_pc3"] - z["right_median_pc3"]

        contrast_slide_rows.append(
            z[
                [
                    "contrast",
                    "slide_id",
                    "left_state",
                    "right_state",
                    "left_median_pc3",
                    "right_median_pc3",
                    "effect",
                ]
            ]
        )

        s, b = summarize_effect(
            name=name,
            values=z["effect"].to_numpy(float),
            seed=SEED + 100 + i,
        )
        s["left_state"] = left
        s["right_state"] = right
        s["effect_definition"] = f"median_PC3({left}) - median_PC3({right})"
        contrast_summary_rows.append(s)
        contrast_boot_parts.append(b)

    contrast_slide = (
        pd.concat(contrast_slide_rows, ignore_index=True)
        if contrast_slide_rows else pd.DataFrame()
    )
    contrast_summary = pd.DataFrame(contrast_summary_rows)
    contrast_boot = (
        pd.concat(contrast_boot_parts, ignore_index=True)
        if contrast_boot_parts else pd.DataFrame()
    )

    print()
    print("Paired state contrasts:")
    print(contrast_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # TUMOR LOCALIZATION AUC
    # -------------------------------------------------------------------------
    banner("COMBINED TUMOR-LOCALIZATION AUC")

    auc_slide_col = find_col(auc, ["slide_id"])

    # Strictly inherit the frozen Step21C primary endpoint.
    # Do NOT fall back to raw_auc_pc3 or supportive_orientation_free_auc.
    auc_score_col = find_col(
        auc,
        ["primary_fixed_direction_auc_minus_pc3"],
    )

    auc = auc.rename(columns={
        auc_slide_col: "slide_id",
        auc_score_col: "primary_auc",
    }).copy()

    auc["slide_id"] = auc["slide_id"].astype(str)
    auc["primary_auc"] = pd.to_numeric(auc["primary_auc"], errors="coerce")

    if "primary_auc_evaluable" not in auc.columns:
        raise RuntimeError(
            "Frozen Step21C AUC table lacks required primary_auc_evaluable column."
        )

    evaluable = auc["primary_auc_evaluable"].copy()

    if evaluable.dtype != bool:
        evaluable = (
            evaluable.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    if evaluable.isna().any():
        bad_values = (
            auc.loc[evaluable.isna(), "primary_auc_evaluable"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise RuntimeError(
            "Could not interpret all primary_auc_evaluable values as boolean. "
            f"Unresolved values: {bad_values[:20]}"
        )

    finite_auc = np.isfinite(auc["primary_auc"])

    # Every upstream-frozen primary-evaluable slide must carry a finite
    # frozen fixed-direction (-PC3) AUC value.
    if (evaluable & ~finite_auc).any():
        cols = ["slide_id", "primary_auc_evaluable", "primary_auc"]
        if "evaluability_status" in auc.columns:
            cols.insert(2, "evaluability_status")
        bad = auc.loc[evaluable & ~finite_auc, cols]
        raise RuntimeError(
            "Primary AUC-evaluable slides contain missing/non-finite frozen "
            "primary_fixed_direction_auc_minus_pc3 values:\n"
            + bad.to_string(index=False)
        )

    # Conversely, a finite primary frozen AUC on a slide marked non-evaluable
    # would indicate an upstream schema/provenance inconsistency.
    if ((~evaluable) & finite_auc).any():
        cols = ["slide_id", "primary_auc_evaluable", "primary_auc"]
        if "evaluability_status" in auc.columns:
            cols.insert(2, "evaluability_status")
        bad = auc.loc[(~evaluable) & finite_auc, cols]
        raise RuntimeError(
            "Slides marked primary_auc_evaluable=False contain finite frozen "
            "primary AUC values:\n"
            + bad.to_string(index=False)
        )

    auc_analysis = auc.loc[evaluable & finite_auc].copy()

    # Inherit the primary AUC eligibility count directly from the upstream
    # frozen Step21C flag. Do not hard-code an endpoint N in Step31B.
    expected_auc_evaluable = int(evaluable.sum())
    observed_auc_evaluable = int(auc_analysis["slide_id"].nunique())

    if observed_auc_evaluable != expected_auc_evaluable:
        raise RuntimeError(
            "Frozen primary AUC eligibility mismatch: "
            f"flagged_evaluable={expected_auc_evaluable}, "
            f"finite_unique_slides={observed_auc_evaluable}."
        )

    if auc_analysis["slide_id"].duplicated().any():
        bad = auc_analysis.loc[
            auc_analysis["slide_id"].duplicated(keep=False)
        ].sort_values("slide_id")
        raise RuntimeError(
            "Duplicate slide rows in frozen primary AUC analysis table:\n"
            + bad.head(30).to_string(index=False)
        )

    auc_vals = auc_analysis["primary_auc"].to_numpy(float)

    auc_summary_row, auc_boot = summarize_auc(
        auc_vals,
        seed=SEED + 500,
    )
    auc_summary_row["endpoint"] = "frozen_G3_G4_G5_vs_Normal_Stroma"
    auc_summary_row["score"] = "-PC3"
    auc_summary_row["unit"] = "slide"
    auc_summary = pd.DataFrame([auc_summary_row])

    print(auc_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # MORPHOLOGY COVERAGE
    # -------------------------------------------------------------------------
    banner("COMBINED WITHIN-STATE MORPHOLOGY COVERAGE")

    morph_slide_col = find_col(morph, ["slide_id"])
    morph_state_col = find_col(morph, ["histologic_state", "state"], required=False)
    metric_col = find_col(
        morph,
        ["metric", "morphology_metric", "metric_name"],
        required=False,
    )

    morph = morph.rename(columns={morph_slide_col: "slide_id"}).copy()
    morph["slide_id"] = morph["slide_id"].astype(str)

    if morph_state_col is not None:
        morph = morph.rename(columns={morph_state_col: "histologic_state"})
        morph["histologic_state"] = morph["histologic_state"].map(normalize_state)

    morph_cov_rows = []

    if morph_state_col is not None and metric_col is not None:
        morph = morph.rename(columns={metric_col: "metric"})
        for (st, metric), d in morph.groupby(["histologic_state", "metric"], sort=False):
            morph_cov_rows.append({
                "histologic_state": st,
                "metric": str(metric),
                "n_slides": int(d["slide_id"].nunique()),
                "n_rows": int(len(d)),
            })
    elif morph_state_col is not None:
        for st, d in morph.groupby("histologic_state", sort=False):
            morph_cov_rows.append({
                "histologic_state": st,
                "metric": "ALL_FROZEN_METRICS",
                "n_slides": int(d["slide_id"].nunique()),
                "n_rows": int(len(d)),
            })
    else:
        morph_cov_rows.append({
            "histologic_state": "UNRESOLVED_FROM_SCHEMA",
            "metric": "ALL_FROZEN_METRICS",
            "n_slides": int(morph["slide_id"].nunique()),
            "n_rows": int(len(morph)),
        })

    morph_cov = pd.DataFrame(morph_cov_rows)
    print(morph_cov.to_string(index=False))

    # -------------------------------------------------------------------------
    # AUDIT
    # -------------------------------------------------------------------------
    audit_rows = [
        {
            "audit_item": "raw_aggc_unique_slides",
            "observed": int(universe["slide_id"].astype(str).nunique()),
            "expected": EXPECTED_TOTAL,
            "pass": int(universe["slide_id"].astype(str).nunique()) == EXPECTED_TOTAL,
        },
        {
            "audit_item": "state_table_unique_slides",
            "observed": int(state["slide_id"].nunique()),
            "expected": np.nan,
            "pass": True,
        },
        {
            "audit_item": "auc_evaluable_unique_slides",
            "observed": int(auc_analysis["slide_id"].nunique()),
            "expected": expected_auc_evaluable,
            "pass": int(auc_analysis["slide_id"].nunique()) == expected_auc_evaluable,
        },
        {
            "audit_item": "morphology_unique_slides",
            "observed": int(morph["slide_id"].nunique()),
            "expected": np.nan,
            "pass": True,
        },
        {
            "audit_item": "pca_refit",
            "observed": 0,
            "expected": 0,
            "pass": True,
        },
        {
            "audit_item": "pc3_sign_change",
            "observed": 0,
            "expected": 0,
            "pass": True,
        },
        {
            "audit_item": "tile_level_inferential_pooling",
            "observed": 0,
            "expected": 0,
            "pass": True,
        },
    ]
    audit = pd.DataFrame(audit_rows)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    banner("SAVE")

    desc.to_csv(a.output / OUT_STATE_DESC, index=False)
    contrast_slide.to_csv(a.output / OUT_CONTRAST_SLIDE, index=False)
    contrast_summary.to_csv(a.output / OUT_CONTRAST_SUM, index=False)
    contrast_boot.to_csv(
        a.output / OUT_CONTRAST_BOOT,
        index=False,
        compression="gzip",
    )
    auc_summary.to_csv(a.output / OUT_AUC_SUM, index=False)
    auc_boot.to_csv(
        a.output / OUT_AUC_BOOT,
        index=False,
        compression="gzip",
    )
    morph_cov.to_csv(a.output / OUT_MORPH_COVERAGE, index=False)
    audit.to_csv(a.output / OUT_AUDIT, index=False)

    hash_rows = []
    for role, path in paths.items():
        hash_rows.append({
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
        })
    pd.DataFrame(hash_rows).to_csv(a.output / OUT_HASHES, index=False)

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "role": "combined_cohort_inference_from_frozen_step31a_tables",
        "independent_unit": "slide",
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED,
        "combined_reporting_rule": (
            "Dataset-provided AGGC Train/Test labels are ignored for cohort-level "
            "summary estimation but remain preserved upstream for provenance."
        ),
        "guardrails": {
            "pca_refit": False,
            "pc3_reprojection": False,
            "pc3_sign_flip": False,
            "state_redefinition": False,
            "tumor_class_redefinition": False,
            "support_threshold_redefinition": False,
            "tile_level_pooled_inference": False,
            "outcome_based_exclusion": False,
        },
        "contrasts": [
            {
                "contrast": name,
                "left_state": left,
                "right_state": right,
                "effect": f"median_PC3({left}) - median_PC3({right})",
            }
            for name, left, right, _ in CONTRASTS
        ],
        "outputs": {
            "state_descriptive_summary": OUT_STATE_DESC,
            "paired_state_contrasts_per_slide": OUT_CONTRAST_SLIDE,
            "paired_state_contrasts_summary": OUT_CONTRAST_SUM,
            "paired_state_contrasts_bootstrap": OUT_CONTRAST_BOOT,
            "tumor_auc_summary": OUT_AUC_SUM,
            "tumor_auc_bootstrap": OUT_AUC_BOOT,
            "morphology_coverage_summary": OUT_MORPH_COVERAGE,
            "audit": OUT_AUDIT,
        },
    }

    with (a.output / OUT_META).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    readme = f"""STEP 31B: COMBINED AGGC PC3 ASSOCIATIONS

Version: {VERSION}

This step evaluates AGGC as one external cohort after the frozen Train/Test
partition-specific analyses were completed.

The underlying PC3 representation, sign, annotation definitions, support
thresholds, and endpoint-specific eligibility are unchanged.

Independent inferential unit:
    slide

Bootstrap:
    {N_BOOT} slide-level resamples
    seed = {SEED}

Paired state contrasts:
    Stroma - Normal
    Normal - G3
    G4 - G3
    G5 - G4

Tumor localization:
    existing frozen per-slide AUC values from Step31A / Step21C
    score = -PC3
    positive = G3/G4/G5
    negative = Normal/Stroma

No tile-level pooled inference is performed.
No PCA refit or PC3 sign selection is performed.
"""
    (a.output / OUT_README).write_text(readme, encoding="utf-8")

    print()
    print("STEP31B PASS")
    print(audit.to_string(index=False))
    print()
    print(f"Saved to: {a.output}")


if __name__ == "__main__":
    main()
