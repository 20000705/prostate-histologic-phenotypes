#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = RESULTS_ROOT

STEP15A_DIR = RESULTS_ROOT / "15a_pc3_quantitative_morphology_specification"
STEP15A_PLAN = STEP15A_DIR / "pc3_quantitative_morphology_analysis_plan.json"
STEP15A_THRESHOLDS = STEP15A_DIR / "pc3_quantitative_morphology_thresholds.json"
STEP15A_FEATURE_SPEC = STEP15A_DIR / "pc3_quantitative_morphology_feature_specification.csv"
STEP15A_METADATA = STEP15A_DIR / "analysis_metadata.json"

STEP15B_DIR = RESULTS_ROOT / "15b_pc3_quantitative_morphology"
STEP15B_PATIENT = STEP15B_DIR / "pc3_quantitative_morphology_patient_summary.csv"
STEP15B_AUDIT = STEP15B_DIR / "pc3_quantitative_morphology_sampling_audit.csv"
STEP15B_METADATA = STEP15B_DIR / "analysis_metadata.json"

OUT_DIR = RESULTS_ROOT / "15c_pc3_quantitative_morphology_inference"
OUT_PRIMARY = OUT_DIR / "pc3_primary_all_tissue_patient_level_results.csv"
OUT_TUMOR_CORE = OUT_DIR / "pc3_tumor_core_sensitivity_patient_level_results.csv"
OUT_TUMOR_ONLY = OUT_DIR / "pc3_tumor_only_sensitivity_patient_level_results.csv"
OUT_BORDER = OUT_DIR / "pc3_border_excluded_sensitivity_results.csv"
OUT_SECONDARY = OUT_DIR / "pc3_secondary_patient_level_results.csv"
OUT_PAIRED = OUT_DIR / "pc3_paired_patient_values.csv"
OUT_FAMILY_SUMMARY = OUT_DIR / "pc3_inference_family_summary.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"

CORE_OUTPUTS = [
    OUT_PRIMARY, OUT_TUMOR_CORE, OUT_TUMOR_ONLY, OUT_BORDER,
    OUT_SECONDARY, OUT_PAIRED, OUT_FAMILY_SUMMARY, OUT_METADATA, OUT_README,
]

ANALYSIS_NAME = "Internal-cohort PC3 quantitative morphology patient-level inference"
VERSION = "1.1.0"
EXPECTED_15A_VERSION_PREFIX = "2.2."

PRIMARY_FEATURES = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "NPI_nuclear_packing",
]

SECONDARY_FEATURES = [
    "CLR_nuclear_to_lumen",
    "CSR_nuclear_to_stroma",
    "dark_fraction",
    "hematoxylin_OD_proxy",
    "lumen_count",
    "lumen_area",
    "median_lumen_area",
    "mean_lumen_area",
    "lumen_component_density",
    "lumen_fragmentation",
]

BORDER_SENSITIVITY_FEATURES = [
    ("SEI_eosin_stroma", "SEI_eosin_stroma_border_excluded_denominator"),
    ("GOI_lumen_openness", "GOI_lumen_openness_border_excluded"),
    ("NPI_nuclear_packing", "NPI_nuclear_packing_border_excluded_denominator"),
]

COMPARISON_SPECS = {
    "primary_all_tissue": {
        "low_stratum": "all_tissue_low",
        "high_stratum": "all_tissue_high",
        "comparison_role": "primary",
    },
    "tumor_core_sensitivity": {
        "low_stratum": "tumor_core_low",
        "high_stratum": "tumor_core_high",
        "comparison_role": "prespecified_tumor_purity_sensitivity",
    },
    "tumor_only_sensitivity": {
        "low_stratum": "tumor_only_low",
        "high_stratum": "tumor_only_high",
        "comparison_role": "additional_tumor_only_sensitivity",
    },
}

BOOTSTRAP_RESAMPLES_DEFAULT = 10000
RANDOM_SEED_DEFAULT = 20260805


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_file(path: Path, label_text: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input ({label_text}): {path}")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_outputs(overwrite: bool):
    existing = [p for p in CORE_OUTPUTS if p.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Step-15c outputs already exist. Refusing silent overwrite. "
            "Use --overwrite if intentional."
        )


def load_and_validate_frozen_plan():
    for path, label_text in [
        (STEP15A_PLAN, "Step 15a frozen plan"),
        (STEP15A_THRESHOLDS, "Step 15a frozen thresholds"),
        (STEP15A_FEATURE_SPEC, "Step 15a frozen feature specification"),
        (STEP15A_METADATA, "Step 15a metadata"),
        (STEP15B_PATIENT, "Step 15b patient summary"),
        (STEP15B_AUDIT, "Step 15b QC audit"),
        (STEP15B_METADATA, "Step 15b metadata"),
    ]:
        require_file(path, label_text)

    with STEP15A_PLAN.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    with STEP15B_METADATA.open("r", encoding="utf-8") as f:
        metadata15b = json.load(f)

    version15a = str(plan.get("version", ""))
    if not version15a.startswith(EXPECTED_15A_VERSION_PREFIX):
        raise RuntimeError(
            f"Step-15c expects Step-15a {EXPECTED_15A_VERSION_PREFIX}x; found {version15a}"
        )

    if plan.get("primary_endpoints", []) != PRIMARY_FEATURES:
        raise RuntimeError("Frozen primary endpoint mismatch.")

    if plan.get("secondary_endpoints", []) != SECONDARY_FEATURES:
        raise RuntimeError("Frozen secondary endpoint mismatch.")

    if metadata15b.get("inferential_statistics_performed", False):
        raise RuntimeError("Step-15b unexpectedly reports inferential statistics.")

    if not metadata15b.get("formal_full_run", False):
        raise RuntimeError("Step-15b was not a formal full run.")

    recorded = metadata15b.get("step15a", {})
    current_hashes = {
        "plan": sha256_file(STEP15A_PLAN),
        "thresholds": sha256_file(STEP15A_THRESHOLDS),
        "feature_spec": sha256_file(STEP15A_FEATURE_SPEC),
    }
    recorded_hashes = {
        "plan": recorded.get("plan_sha256"),
        "thresholds": recorded.get("threshold_sha256"),
        "feature_spec": recorded.get("feature_spec_sha256"),
    }

    for key in ["plan", "thresholds", "feature_spec"]:
        if recorded_hashes[key] is None:
            raise RuntimeError(
                f"Step-15b metadata is missing recorded Step-15a {key} SHA256."
            )
        if recorded_hashes[key] != current_hashes[key]:
            raise RuntimeError(
                f"Step-15a {key} hash does not match the specification used by Step-15b.\n"
                f"Recorded by Step-15b: {recorded_hashes[key]}\n"
                f"Current Step-15a:      {current_hashes[key]}"
            )

    provenance = {
        key: {
            "recorded_by_step15b": recorded_hashes[key],
            "current_step15a": current_hashes[key],
            "match": recorded_hashes[key] == current_hashes[key],
        }
        for key in current_hashes
    }

    return plan, metadata15b, provenance


def validate_step15b_qc_gate(plan):
    audit = pd.read_csv(STEP15B_AUDIT, low_memory=False)

    required_strata = {
        "all_tissue_low", "all_tissue_high",
        "tumor_core_low", "tumor_core_high",
        "tumor_only_low", "tumor_only_high",
    }

    observed = set(audit["sampling_stratum"].astype(str))
    missing = required_strata - observed
    if missing:
        raise RuntimeError(f"Step-15b QC audit missing strata: {sorted(missing)}")

    gate = plan.get("inference_qc_gate", {})
    required = [
        "minimum_resolution_fraction",
        "minimum_qc_pass_fraction",
        "minimum_paired_patients",
    ]
    missing_gate = [k for k in required if k not in gate]
    if missing_gate:
        raise RuntimeError(
            "Step-15a plan missing frozen inference QC gate fields: "
            + ", ".join(missing_gate)
        )

    min_resolution = float(gate["minimum_resolution_fraction"])
    min_qc = float(gate["minimum_qc_pass_fraction"])
    min_paired = int(gate["minimum_paired_patients"])

    resolution = pd.to_numeric(audit["resolution_fraction"], errors="coerce")
    bad_resolution = audit.loc[resolution < min_resolution]
    if not bad_resolution.empty:
        raise RuntimeError(
            "Step-15b resolution QC gate failed.\n"
            + bad_resolution.to_string(index=False)
        )

    qc = pd.to_numeric(audit["qc_pass_fraction"], errors="coerce")
    bad_qc = audit.loc[qc < min_qc]
    if not bad_qc.empty:
        raise RuntimeError(
            "Step-15b image-QC retention gate failed.\n"
            + bad_qc.to_string(index=False)
        )

    return audit, min_paired, min_resolution, min_qc


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)

    mask = np.isfinite(p_values)
    valid = p_values[mask]
    m = len(valid)

    if m == 0:
        return adjusted

    order = np.argsort(valid)
    ranked = valid[order]
    raw = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)

    restored = np.empty_like(monotone)
    restored[order] = monotone
    adjusted[mask] = restored
    return adjusted


def paired_rank_biserial(differences: np.ndarray) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    if len(d) == 0:
        return np.nan

    ranks = rankdata(np.abs(d), method="average")
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    denom = w_plus + w_minus

    if denom == 0:
        return np.nan

    return float((w_plus - w_minus) / denom)


def bootstrap_median_difference_ci(
    differences: np.ndarray,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.median(d[idx])

    return (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def safe_wilcoxon(low: np.ndarray, high: np.ndarray) -> dict:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    valid = np.isfinite(low) & np.isfinite(high)
    low = low[valid]
    high = high[valid]

    d = high - low
    nonzero_n = int(np.sum(d != 0))

    if len(d) == 0:
        return {"wilcoxon_statistic": np.nan, "p_value": np.nan, "n_nonzero_differences": 0}

    if nonzero_n == 0:
        return {"wilcoxon_statistic": 0.0, "p_value": 1.0, "n_nonzero_differences": 0}

    result = wilcoxon(
        high,
        low,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return {
        "wilcoxon_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_nonzero_differences": nonzero_n,
    }


def make_patient_pairs(
    patient_df: pd.DataFrame,
    low_stratum: str,
    high_stratum: str,
    feature: str,
) -> pd.DataFrame:
    low = patient_df.loc[
        patient_df["sampling_stratum"] == low_stratum,
        ["patient_id", feature],
    ].copy().rename(columns={feature: "low_value"})

    high = patient_df.loc[
        patient_df["sampling_stratum"] == high_stratum,
        ["patient_id", feature],
    ].copy().rename(columns={feature: "high_value"})

    pairs = low.merge(
        high,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    pairs["low_value"] = pd.to_numeric(pairs["low_value"], errors="coerce")
    pairs["high_value"] = pd.to_numeric(pairs["high_value"], errors="coerce")

    pairs = pairs.loc[
        np.isfinite(pairs["low_value"]) & np.isfinite(pairs["high_value"])
    ].copy()

    pairs["difference_high_minus_low"] = (
        pairs["high_value"] - pairs["low_value"]
    )

    return pairs


def analyze_feature(
    patient_df: pd.DataFrame,
    comparison_name: str,
    comparison_role: str,
    low_stratum: str,
    high_stratum: str,
    feature: str,
    endpoint_class: str,
    bootstrap_resamples: int,
    random_seed: int,
    minimum_paired_patients: int,
):
    pairs = make_patient_pairs(
        patient_df,
        low_stratum,
        high_stratum,
        feature,
    )

    low = pairs["low_value"].to_numpy(dtype=float)
    high = pairs["high_value"].to_numpy(dtype=float)
    d = high - low
    n_pairs = int(len(pairs))

    if n_pairs < minimum_paired_patients:
        row = {
            "comparison_name": comparison_name,
            "comparison_role": comparison_role,
            "low_stratum": low_stratum,
            "high_stratum": high_stratum,
            "feature": feature,
            "endpoint_class": endpoint_class,
            "inference_status": "insufficient_paired_patients",
            "minimum_paired_patients_required": minimum_paired_patients,
            "n_paired_patients": n_pairs,
            "n_nonzero_differences": np.nan,
            "low_median_across_patients": float(np.median(low)) if n_pairs else np.nan,
            "high_median_across_patients": float(np.median(high)) if n_pairs else np.nan,
            "median_paired_difference_high_minus_low": float(np.median(d)) if n_pairs else np.nan,
            "mean_paired_difference_high_minus_low": float(np.mean(d)) if n_pairs else np.nan,
            "bootstrap_95ci_lower": np.nan,
            "bootstrap_95ci_upper": np.nan,
            "paired_rank_biserial": np.nan,
            "wilcoxon_statistic": np.nan,
            "p_value_raw": np.nan,
            "n_high_greater_than_low": int(np.sum(d > 0)),
            "n_high_less_than_low": int(np.sum(d < 0)),
            "n_equal": int(np.sum(d == 0)),
            "directional_sign_consistency_fraction": np.nan,
            "reported_difference_direction": "PC3-high minus PC3-low",
        }
        paired.insert(0, "comparison_name", comparison_name)
        paired.insert(1, "comparison_role", comparison_role)
        paired.insert(2, "feature", feature)
        paired.insert(3, "endpoint_class", endpoint_class)
        paired.insert(4, "low_stratum", low_stratum)
        paired.insert(5, "high_stratum", high_stratum)
        paired.insert(6, "minimum_paired_patients_required", minimum_paired_patients)
        return row, paired

    w = safe_wilcoxon(low, high)
    ci_low, ci_high = bootstrap_median_difference_ci(
        d,
        n_resamples=bootstrap_resamples,
        seed=random_seed,
    )

    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_zero = int(np.sum(d == 0))
    denom = n_pos + n_neg

    row = {
        "comparison_name": comparison_name,
        "comparison_role": comparison_role,
        "low_stratum": low_stratum,
        "high_stratum": high_stratum,
        "feature": feature,
        "endpoint_class": endpoint_class,
        "inference_status": "performed",
        "minimum_paired_patients_required": minimum_paired_patients,
        "n_paired_patients": int(len(pairs)),
        "n_nonzero_differences": w["n_nonzero_differences"],
        "low_median_across_patients": float(np.median(low)) if len(low) else np.nan,
        "high_median_across_patients": float(np.median(high)) if len(high) else np.nan,
        "median_paired_difference_high_minus_low": float(np.median(d)) if len(d) else np.nan,
        "mean_paired_difference_high_minus_low": float(np.mean(d)) if len(d) else np.nan,
        "bootstrap_95ci_lower": ci_low,
        "bootstrap_95ci_upper": ci_high,
        "paired_rank_biserial": paired_rank_biserial(d),
        "wilcoxon_statistic": w["wilcoxon_statistic"],
        "p_value_raw": w["p_value"],
        "n_high_greater_than_low": n_pos,
        "n_high_less_than_low": n_neg,
        "n_equal": n_zero,
        "directional_sign_consistency_fraction": (
            max(n_pos, n_neg) / denom if denom > 0 else np.nan
        ),
        "reported_difference_direction": "PC3-high minus PC3-low",
    }

    paired = pairs.copy()
    paired.insert(0, "comparison_name", comparison_name)
    paired.insert(1, "comparison_role", comparison_role)
    paired.insert(2, "feature", feature)
    paired.insert(3, "endpoint_class", endpoint_class)
    paired.insert(4, "low_stratum", low_stratum)
    paired.insert(5, "high_stratum", high_stratum)
    paired.insert(6, "minimum_paired_patients_required", minimum_paired_patients)

    return row, paired


def analyze_feature_family(
    patient_df: pd.DataFrame,
    comparison_name: str,
    comparison_role: str,
    low_stratum: str,
    high_stratum: str,
    features: List[str],
    endpoint_class: str,
    bootstrap_resamples: int,
    random_seed: int,
    minimum_paired_patients: int,
):
    rows = []
    pair_frames = []

    for i, feature in enumerate(features):
        row, paired = analyze_feature(
            patient_df,
            comparison_name,
            comparison_role,
            low_stratum,
            high_stratum,
            feature,
            endpoint_class,
            bootstrap_resamples,
            random_seed + i,
            minimum_paired_patients,
        )
        rows.append(row)
        pair_frames.append(paired)

    results = pd.DataFrame(rows)
    results["p_value_fdr_bh"] = benjamini_hochberg(
        results["p_value_raw"].to_numpy(dtype=float)
    )

    return results, pd.concat(pair_frames, ignore_index=True)


def analyze_border_sensitivity(patient_df, bootstrap_resamples, random_seed, minimum_paired_patients):
    all_results = []
    all_pairs = []

    for c_idx, (comparison_name, spec) in enumerate(COMPARISON_SPECS.items()):
        family_rows = []
        family_pairs = []

        for f_idx, (historical_feature, sensitivity_feature) in enumerate(
            BORDER_SENSITIVITY_FEATURES
        ):
            row, paired = analyze_feature(
                patient_df=patient_df,
                comparison_name=f"{comparison_name}_border_excluded",
                comparison_role=(
                    spec["comparison_role"]
                    + "_measurement_definition_sensitivity"
                ),
                low_stratum=spec["low_stratum"],
                high_stratum=spec["high_stratum"],
                feature=sensitivity_feature,
                endpoint_class="border_excluded_measurement_sensitivity",
                bootstrap_resamples=bootstrap_resamples,
                random_seed=random_seed + c_idx * 100 + f_idx,
                minimum_paired_patients=minimum_paired_patients,
            )
            row["corresponding_historical_feature"] = historical_feature
            paired["corresponding_historical_feature"] = historical_feature
            family_rows.append(row)
            family_pairs.append(paired)

        family = pd.DataFrame(family_rows)
        family["p_value_fdr_bh"] = benjamini_hochberg(
            family["p_value_raw"].to_numpy(dtype=float)
        )
        family["fdr_family_id"] = f"border_{comparison_name}"

        all_results.append(family)
        all_pairs.append(pd.concat(family_pairs, ignore_index=True))

    return (
        pd.concat(all_results, ignore_index=True),
        pd.concat(all_pairs, ignore_index=True),
    )


def analyze_secondary_endpoints(patient_df, bootstrap_resamples, random_seed, minimum_paired_patients):
    all_results = []
    all_pairs = []

    for c_idx, (comparison_name, spec) in enumerate(COMPARISON_SPECS.items()):
        results, paired = analyze_feature_family(
            patient_df=patient_df,
            comparison_name=f"{comparison_name}_secondary",
            comparison_role=spec["comparison_role"] + "_secondary_supportive",
            low_stratum=spec["low_stratum"],
            high_stratum=spec["high_stratum"],
            features=SECONDARY_FEATURES,
            endpoint_class="secondary",
            bootstrap_resamples=bootstrap_resamples,
            random_seed=random_seed + c_idx * 100,
            minimum_paired_patients=minimum_paired_patients,
        )
        results["fdr_family_id"] = f"secondary_{comparison_name}"
        all_results.append(results)
        all_pairs.append(paired)

    return (
        pd.concat(all_results, ignore_index=True),
        pd.concat(all_pairs, ignore_index=True),
    )


def _summarize_family(family_id, family_role, df):
    raw = pd.to_numeric(df["p_value_raw"], errors="coerce")
    fdr = pd.to_numeric(df["p_value_fdr_bh"], errors="coerce")
    return {
        "fdr_family_id": family_id,
        "family_role": family_role,
        "n_tests_total": int(len(df)),
        "n_tests_inference_performed": int((df["inference_status"] == "performed").sum()),
        "n_raw_p_lt_0_05": int((raw < 0.05).sum()),
        "n_fdr_q_lt_0_05": int((fdr < 0.05).sum()),
        "minimum_raw_p": float(raw.min()) if raw.notna().any() else np.nan,
        "minimum_fdr_q": float(fdr.min()) if fdr.notna().any() else np.nan,
    }


def build_family_summary(primary_df, core_df, tumor_only_df, border_df, secondary_df):
    rows = [
        _summarize_family("primary_all_tissue", "primary", primary_df),
        _summarize_family(
            "primary_tumor_core_sensitivity",
            "prespecified_tumor_purity_sensitivity",
            core_df,
        ),
        _summarize_family(
            "primary_tumor_only_sensitivity",
            "additional_tumor_only_sensitivity",
            tumor_only_df,
        ),
    ]

    for comparison_name in COMPARISON_SPECS:
        b = border_df.loc[
            border_df["fdr_family_id"] == f"border_{comparison_name}"
        ]
        rows.append(
            _summarize_family(
                f"border_{comparison_name}",
                "measurement_definition_sensitivity",
                b,
            )
        )

        s = secondary_df.loc[
            secondary_df["fdr_family_id"] == f"secondary_{comparison_name}"
        ]
        rows.append(
            _summarize_family(
                f"secondary_{comparison_name}",
                "secondary_supportive",
                s,
            )
        )

    return pd.DataFrame(rows)


def build_readme(
    bootstrap_resamples: int,
    random_seed: int,
    plan: dict,
    minimum_paired_patients: int,
    minimum_resolution_fraction: float,
    minimum_qc_pass_fraction: float,
) -> str:
    return f"""STEP 15C: PC3 QUANTITATIVE MORPHOLOGY PATIENT-LEVEL INFERENCE
===================================================================

Version: {VERSION}

Primary analysis:
  all_tissue_high vs all_tissue_low

Primary endpoints:
  SEI_eosin_stroma
  GOI_lumen_openness
  NPI_nuclear_packing

Inference unit:
  patient

Statistics:
  two-sided paired Wilcoxon signed-rank
  median paired difference = PC3-high minus PC3-low
  paired rank-biserial effect size
  95% patient-level bootstrap CI
  BH-FDR within each prespecified family

Bootstrap resamples: {bootstrap_resamples}
Random seed: {random_seed}

Frozen Step-15a inference release gates:
  minimum_resolution_fraction = {minimum_resolution_fraction}
  minimum_qc_pass_fraction     = {minimum_qc_pass_fraction}
  minimum_paired_patients      = {minimum_paired_patients}

Exact provenance chain:
  current Step-15a plan / threshold / feature-spec SHA256 hashes must exactly
  match the hashes recorded by the formal Step-15b run. Otherwise inference aborts.

Actual FDR families are reported separately in pc3_inference_family_summary.csv.

Continuous-PC3 sensitivity:
  {plan["continuous_pc3_sensitivity"]["deferred_statement"]}
  No continuous-PC3 inference is performed in Step 15c.

No tile-level inferential p-values are used.
No secondary endpoint is promoted to primary.
No measurement threshold or frozen sample definition is changed.
"""


def main():
    args = parse_args()

    print("=" * 124)
    print("STEP 15C: PC3 QUANTITATIVE MORPHOLOGY PATIENT-LEVEL INFERENCE")
    print("=" * 124)
    print("[VERSION]          ", VERSION)
    print("[STEP 15A PLAN]    ", STEP15A_PLAN)
    print("[STEP 15B PATIENT] ", STEP15B_PATIENT)
    print("[OUTPUT]           ", OUT_DIR)

    guard_outputs(args.overwrite)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan, metadata15b, provenance_checks = load_and_validate_frozen_plan()
    (
        audit15b,
        minimum_paired_patients,
        minimum_resolution_fraction,
        minimum_qc_pass_fraction,
    ) = validate_step15b_qc_gate(plan)

    patient_df = pd.read_csv(STEP15B_PATIENT, low_memory=False)

    required_features = set(
        PRIMARY_FEATURES
        + SECONDARY_FEATURES
        + [x[1] for x in BORDER_SENSITIVITY_FEATURES]
    )
    missing_features = required_features - set(patient_df.columns)

    if missing_features:
        raise RuntimeError(
            "Step-15b patient summary missing required features: "
            f"{sorted(missing_features)}"
        )

    stats_plan = plan.get("statistical_tests", {})
    bootstrap_resamples = int(
        stats_plan.get("bootstrap_resamples", BOOTSTRAP_RESAMPLES_DEFAULT)
    )
    random_seed = int(
        stats_plan.get("random_seed", RANDOM_SEED_DEFAULT)
    )

    primary_df, primary_pairs = analyze_feature_family(
        patient_df,
        "primary_all_tissue",
        "primary",
        "all_tissue_low",
        "all_tissue_high",
        PRIMARY_FEATURES,
        "primary",
        bootstrap_resamples,
        random_seed,
        minimum_paired_patients,
    )

    primary_df["fdr_family_id"] = "primary_all_tissue"

    core_df, core_pairs = analyze_feature_family(
        patient_df,
        "tumor_core_sensitivity",
        "prespecified_tumor_purity_sensitivity",
        "tumor_core_low",
        "tumor_core_high",
        PRIMARY_FEATURES,
        "primary_endpoint_sensitivity",
        bootstrap_resamples,
        random_seed + 100,
        minimum_paired_patients,
    )

    core_df["fdr_family_id"] = "primary_tumor_core_sensitivity"

    tumor_only_df, tumor_only_pairs = analyze_feature_family(
        patient_df,
        "tumor_only_sensitivity",
        "additional_tumor_only_sensitivity",
        "tumor_only_low",
        "tumor_only_high",
        PRIMARY_FEATURES,
        "primary_endpoint_sensitivity",
        bootstrap_resamples,
        random_seed + 200,
        minimum_paired_patients,
    )

    tumor_only_df["fdr_family_id"] = "primary_tumor_only_sensitivity"

    border_df, border_pairs = analyze_border_sensitivity(
        patient_df,
        bootstrap_resamples,
        random_seed + 1000,
        minimum_paired_patients,
    )

    secondary_df, secondary_pairs = analyze_secondary_endpoints(
        patient_df,
        bootstrap_resamples,
        random_seed + 5000,
        minimum_paired_patients,
    )

    paired_df = pd.concat(
        [
            primary_pairs,
            core_pairs,
            tumor_only_pairs,
            border_pairs,
            secondary_pairs,
        ],
        ignore_index=True,
        sort=False,
    )

    family_summary_df = build_family_summary(
        primary_df,
        core_df,
        tumor_only_df,
        border_df,
        secondary_df,
    )

    primary_df.to_csv(OUT_PRIMARY, index=False)
    core_df.to_csv(OUT_TUMOR_CORE, index=False)
    tumor_only_df.to_csv(OUT_TUMOR_ONLY, index=False)
    border_df.to_csv(OUT_BORDER, index=False)
    secondary_df.to_csv(OUT_SECONDARY, index=False)
    paired_df.to_csv(OUT_PAIRED, index=False)
    family_summary_df.to_csv(OUT_FAMILY_SUMMARY, index=False)

    metadata = {
        "analysis_name": ANALYSIS_NAME,
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "step15a_version": plan.get("version"),
        "step15a_plan_sha256": sha256_file(STEP15A_PLAN),
        "step15a_threshold_sha256": sha256_file(STEP15A_THRESHOLDS),
        "step15a_feature_spec_sha256": sha256_file(STEP15A_FEATURE_SPEC),
        "step15b_patient_sha256": sha256_file(STEP15B_PATIENT),
        "provenance_chain_verified": True,
        "provenance_checks": provenance_checks,
        "minimum_resolution_fraction": minimum_resolution_fraction,
        "minimum_qc_pass_fraction": minimum_qc_pass_fraction,
        "minimum_paired_patients": minimum_paired_patients,
        "step15b_audit_sha256": sha256_file(STEP15B_AUDIT),
        "step15b_qc_gate_passed": True,
        "step15b_qc_audit_rows": int(len(audit15b)),
        "primary_endpoints": PRIMARY_FEATURES,
        "secondary_endpoints": SECONDARY_FEATURES,
        "primary_analysis": "all_tissue_high vs all_tissue_low",
        "tumor_core_sensitivity": "tumor_core_high vs tumor_core_low",
        "tumor_only_sensitivity": "tumor_only_high vs tumor_only_low",
        "bootstrap_resamples": bootstrap_resamples,
        "random_seed": random_seed,
        "primary_test": "two-sided paired Wilcoxon signed-rank",
        "effect_size": "paired rank-biserial",
        "tile_level_inference_performed": False,
        "secondary_promoted_to_primary": False,
        "continuous_pc3_sensitivity_performed": False,
        "continuous_pc3_sensitivity_deferred_to": "Step 15d",
        "continuous_pc3_sensitivity_statement":
            plan["continuous_pc3_sensitivity"]["deferred_statement"],
    }

    with OUT_METADATA.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    OUT_README.write_text(
        build_readme(
            bootstrap_resamples,
            random_seed,
            plan,
            minimum_paired_patients,
            minimum_resolution_fraction,
            minimum_qc_pass_fraction,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "feature",
        "inference_status",
        "n_paired_patients",
        "low_median_across_patients",
        "high_median_across_patients",
        "median_paired_difference_high_minus_low",
        "bootstrap_95ci_lower",
        "bootstrap_95ci_upper",
        "paired_rank_biserial",
        "p_value_raw",
        "p_value_fdr_bh",
        "n_high_greater_than_low",
        "n_high_less_than_low",
    ]

    print("\n" + "=" * 124)
    print("PROVENANCE CHAIN")
    print("=" * 124)
    for key, info in provenance_checks.items():
        print(f"{key:15s} match={info['match']}")

    print("\n" + "=" * 124)
    print("PRIMARY ALL-TISSUE RESULTS")
    print("=" * 124)
    print(primary_df[display_cols].to_string(index=False))

    print("\n" + "=" * 124)
    print("TUMOR-CORE SENSITIVITY")
    print("=" * 124)
    print(core_df[display_cols].to_string(index=False))

    print("\n" + "=" * 124)
    print("TUMOR-ONLY SENSITIVITY")
    print("=" * 124)
    print(tumor_only_df[display_cols].to_string(index=False))

    print("\n" + "=" * 124)
    print("INFERENCE FAMILY SUMMARY")
    print("=" * 124)
    print(family_summary_df.to_string(index=False))

    print("\n[OK] Saved:")
    for p in [
        OUT_PRIMARY,
        OUT_TUMOR_CORE,
        OUT_TUMOR_ONLY,
        OUT_BORDER,
        OUT_SECONDARY,
        OUT_PAIRED,
        OUT_FAMILY_SUMMARY,
        OUT_METADATA,
        OUT_README,
    ]:
        print(" ", p)

    print("\n[CONTINUOUS-PC3]")
    print(plan["continuous_pc3_sensitivity"]["deferred_statement"])
    print("\n[DONE] Step-15c v1.1 frozen extreme-group inference complete.")


if __name__ == "__main__":
    main()
