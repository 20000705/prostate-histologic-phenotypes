#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 23C: INTERNAL PC6 MORPHOLOGY INFERENCE + PC3-vs-PC6 DISTINCTNESS

VERSION: 1.0.0
SCHEMA : 23c-internal-pc6-morphology-inference-pc3-pc6-distinctness-schema-1.0

ROLE
----
INFERENCE ONLY.

Step23B is the sole PC6 measurement authority. Step23C MUST NOT:
    - resample tiles;
    - remeasure morphology;
    - redefine masks / scale / thresholds / labels;
    - refit PCA;
    - screen PCs;
    - tune PC6 sign;
    - read AGGC / PESO / TCGA;
    - promote PC5;
    - select features from observed PC6 results.

PRIMARY PC6 INFERENCE
---------------------
Independent unit: patient.

For each frozen morphology feature:
    PC6 effect = patient-level PC6_HIGH - PC6_LOW

Statistics:
    - paired two-sided Wilcoxon signed-rank;
    - paired rank-biserial effect size;
    - median paired difference;
    - 95% patient-level bootstrap CI of median paired difference;
    - BH-FDR within prespecified feature families.

PC3-vs-PC6 DISTINCTNESS
-----------------------
PC3 reference is read ONLY from frozen INTERNAL Step15C all-tissue
PC3-high minus PC3-low patient-level inference.

The primary cross-axis profile uses paired rank-biserial effects because this
effect size is signed and dimensionless across heterogeneous morphology units.

Primary distinctness summary:
    Spearman correlation across the COMPLETE 13-feature signed effect profiles
    (3 primary + 10 secondary).

Supportive summaries:
    - Pearson correlation;
    - cosine similarity;
    - vector angle;
    - same-sign fraction;
    - primary-3-feature profile Spearman;
    - median absolute cross-axis effect-size difference.

A paired patient bootstrap resamples the same common patients for both axes and
recomputes the complete 13-feature profile concordance. No post-hoc binary
"distinct/not-distinct" cutoff is introduced.

STRICT DEDUPLICATION / INTEGRITY POLICY
---------------------------------------
NO silent drop_duplicates() is allowed.

Step23C validates:
    1. Step23B sampled physical tile identity is unique.
    2. Step23B sample_id is unique.
    3. Step23B QC-pass and QC-failure sample IDs are disjoint.
    4. QC-pass + QC-failure sample IDs exactly partition the frozen sample set.
    5. Step23B patient summary has exactly one row per patient x stratum.
    6. Every formal patient has exactly one PC6_LOW and one PC6_HIGH row.
    7. PC3 Step15C reference has exactly one all-tissue row per frozen feature.
    8. PC3 paired source has exactly one patient x feature row for the
       all-tissue comparison.

Any duplicate or conflicting record is written to an audit table and the run
ABORTS. Nothing is silently retained or averaged.

EXPECTED INPUTS
---------------
results/23a_pc6_internal_second_axis_multidimensional_validation_specification/
results/23b_internal_pc6_quantitative_morphology_measurement/
results/15c_pc3_quantitative_morphology_inference/

OUTPUTS
-------
results/23c_internal_pc6_morphology_inference_pc3_pc6_distinctness/

    23c_input_integrity_audit.csv
    23c_duplicate_audit.csv
    pc6_paired_patient_values.csv
    pc6_primary_patient_level_results.csv
    pc6_secondary_patient_level_results.csv
    pc6_sensitivity_patient_level_results.csv
    pc6_inference_family_summary.csv
    pc3_reference_morphology_effect_profile.csv
    pc3_pc6_effect_profile_table.csv
    pc3_pc6_effect_profile_distinctness_summary.csv
    pc3_pc6_profile_bootstrap.csv.gz
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
import shutil
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

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


# =============================================================================
# VERSION / PATHS
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "23c-internal-pc6-morphology-inference-pc3-pc6-distinctness-schema-1.0"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT

STEP23A_DIR = RESULTS / "23a_pc6_internal_second_axis_multidimensional_validation_specification"
STEP23A_PLAN = STEP23A_DIR / "pc6_internal_second_axis_analysis_plan.json"
STEP23A_METRICS = STEP23A_DIR / "pc6_internal_morphology_metric_specification.csv"
STEP23A_DISTINCT = STEP23A_DIR / "pc3_pc6_internal_distinctness_endpoint_specification.csv"
STEP23A_DIRECTION = STEP23A_DIR / "pc6_frozen_direction_snapshot.csv"
STEP23A_METADATA = STEP23A_DIR / "analysis_metadata.json"

STEP23B_DIR = RESULTS / "23b_internal_pc6_quantitative_morphology_measurement"
STEP23B_SAMPLE = STEP23B_DIR / "pc6_sampling_manifest.csv"
STEP23B_TILE = STEP23B_DIR / "pc6_quantitative_morphology_tile_features.csv"
STEP23B_FAILURE = STEP23B_DIR / "pc6_quantitative_morphology_qc_failures.csv"
STEP23B_PATIENT = STEP23B_DIR / "pc6_quantitative_morphology_patient_summary.csv"
STEP23B_AUDIT = STEP23B_DIR / "pc6_quantitative_morphology_sampling_audit.csv"
STEP23B_COHORT = STEP23B_DIR / "canonical_internal_cohort_snapshot.csv"
STEP23B_COHORT_AUDIT = STEP23B_DIR / "canonical_internal_cohort_audit.csv"
STEP23B_HASHES = STEP23B_DIR / "source_hashes.csv"
STEP23B_METADATA = STEP23B_DIR / "analysis_metadata.json"

STEP15A_DIR = RESULTS / "15a_pc3_quantitative_morphology_specification"
STEP15A_PLAN = STEP15A_DIR / "pc3_quantitative_morphology_analysis_plan.json"

STEP15C_DIR = RESULTS / "15c_pc3_quantitative_morphology_inference"
STEP15C_PRIMARY = STEP15C_DIR / "pc3_primary_all_tissue_patient_level_results.csv"
STEP15C_SECONDARY = STEP15C_DIR / "pc3_secondary_patient_level_results.csv"
STEP15C_PAIRED = STEP15C_DIR / "pc3_paired_patient_values.csv"
STEP15C_METADATA = STEP15C_DIR / "analysis_metadata.json"

OUT_DIR = RESULTS / "23c_internal_pc6_morphology_inference_pc3_pc6_distinctness"

OUT_INTEGRITY = OUT_DIR / "23c_input_integrity_audit.csv"
OUT_DUPLICATES = OUT_DIR / "23c_duplicate_audit.csv"
OUT_PAIRED = OUT_DIR / "pc6_paired_patient_values.csv"
OUT_PRIMARY = OUT_DIR / "pc6_primary_patient_level_results.csv"
OUT_SECONDARY = OUT_DIR / "pc6_secondary_patient_level_results.csv"
OUT_SENSITIVITY = OUT_DIR / "pc6_sensitivity_patient_level_results.csv"
OUT_FAMILY = OUT_DIR / "pc6_inference_family_summary.csv"
OUT_PC3_REF = OUT_DIR / "pc3_reference_morphology_effect_profile.csv"
OUT_PROFILE = OUT_DIR / "pc3_pc6_effect_profile_table.csv"
OUT_DISTINCT = OUT_DIR / "pc3_pc6_effect_profile_distinctness_summary.csv"
OUT_DISTINCT_BOOT = OUT_DIR / "pc3_pc6_profile_bootstrap.csv.gz"
OUT_HASHES = OUT_DIR / "source_hashes.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"


# =============================================================================
# FROZEN DESIGN
# =============================================================================

EXPECTED_PATIENTS = 15
EXPECTED_SLIDES = 65

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

SENSITIVITY_FEATURES = [
    "GOI_lumen_openness_border_excluded",
    "NPI_nuclear_packing_border_excluded_denominator",
    "SEI_eosin_stroma_border_excluded_denominator",
    "CLR_nuclear_to_lumen_border_excluded",
    "lumen_count_border_excluded",
    "lumen_area_border_excluded",
    "median_lumen_area_border_excluded",
    "mean_lumen_area_border_excluded",
    "lumen_component_density_border_excluded",
    "lumen_fragmentation_border_excluded",
    "border_connected_open_fraction_all",
]

FROZEN_PROFILE_FEATURES = PRIMARY_FEATURES + SECONDARY_FEATURES
ALL_INFERENCE_FEATURES = FROZEN_PROFILE_FEATURES + SENSITIVITY_FEATURES

LOW_STRATUM = "PC6_LOW"
HIGH_STRATUM = "PC6_HIGH"

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 20260809

PRIMARY_DISTINCTNESS_ENDPOINT = "cross_metric_signed_effect_profile_concordance"


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def banner(title: str, width: int = 132) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input ({label}):\n{path}")


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
                "Use --overwrite only for an intentional rerun under the unchanged frozen contract."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = str(base_seed) + "||" + "||".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    mask = np.isfinite(p)
    pv = p[mask]
    m = len(pv)
    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    raw = ranked * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    restored = np.empty_like(adj_sorted)
    restored[order] = adj_sorted
    out[mask] = restored
    return out


def paired_rank_biserial(differences: Sequence[float]) -> float:
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


def safe_wilcoxon(low: np.ndarray, high: np.ndarray) -> Dict[str, float]:
    low = np.asarray(low, float)
    high = np.asarray(high, float)
    ok = np.isfinite(low) & np.isfinite(high)
    low = low[ok]
    high = high[ok]
    d = high - low

    if len(d) == 0:
        return {
            "wilcoxon_statistic": np.nan,
            "p_value_raw": np.nan,
            "n_nonzero_differences": 0,
        }

    nonzero = int(np.sum(d != 0))
    if nonzero == 0:
        return {
            "wilcoxon_statistic": 0.0,
            "p_value_raw": 1.0,
            "n_nonzero_differences": 0,
        }

    res = wilcoxon(
        high,
        low,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return {
        "wilcoxon_statistic": float(res.statistic),
        "p_value_raw": float(res.pvalue),
        "n_nonzero_differences": nonzero,
    }


def bootstrap_median_ci(
    differences: Sequence[float],
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    if len(d) == 1:
        return float(d[0]), float(d[0])

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


def safe_spearman(x: Sequence[float], y: Sequence[float], minimum_n: int = 3) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < minimum_n:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def safe_pearson(x: Sequence[float], y: Sequence[float], minimum_n: int = 3) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < minimum_n or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cosine_and_angle(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) == 0:
        return np.nan, np.nan
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx == 0 or ny == 0:
        return np.nan, np.nan
    cosine = float(np.dot(x, y) / (nx * ny))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return cosine, angle


def mode_or_none(values: Iterable[Any]) -> Any:
    vals = [x for x in values if pd.notna(x)]
    if not vals:
        return None
    s = pd.Series(vals)
    modes = s.mode(dropna=True)
    return modes.iloc[0] if len(modes) else vals[0]


# =============================================================================
# CONTRACT VALIDATION
# =============================================================================

def load_contract() -> Dict[str, Any]:
    for p, label in [
        (STEP23A_PLAN, "Step23A plan"),
        (STEP23A_METRICS, "Step23A morphology metrics"),
        (STEP23A_DISTINCT, "Step23A distinctness endpoint"),
        (STEP23A_DIRECTION, "Step23A frozen direction"),
        (STEP23A_METADATA, "Step23A metadata"),
        (STEP15A_PLAN, "Step15A frozen plan"),
    ]:
        require_file(p, label)

    plan23a = load_json(STEP23A_PLAN)
    meta23a = load_json(STEP23A_METADATA)
    plan15a = load_json(STEP15A_PLAN)

    if str(plan23a.get("version", "")) != "1.2.0":
        raise RuntimeError(
            f"Step23C requires Step23A v1.2.0; observed={plan23a.get('version')}"
        )

    if str(meta23a.get("phenotype_definition_cohort", "")) != "INTERNAL_DISCOVERY_COHORT":
        raise RuntimeError("Step23A phenotype authority is not INTERNAL_DISCOVERY_COHORT.")

    metrics = pd.read_csv(STEP23A_METRICS, low_memory=False)
    feature_col = (
        "morphology_metric"
        if "morphology_metric" in metrics.columns
        else "feature_name"
        if "feature_name" in metrics.columns
        else None
    )
    if feature_col is None:
        raise RuntimeError("Step23A morphology metric table lacks morphology_metric/feature_name.")

    observed = metrics[feature_col].astype(str).tolist()
    if observed != FROZEN_PROFILE_FEATURES:
        raise RuntimeError(
            "Frozen Step23A 13-feature morphology profile changed.\n"
            f"Expected={FROZEN_PROFILE_FEATURES}\nObserved={observed}"
        )

    direction = pd.read_csv(STEP23A_DIRECTION, low_memory=False)
    if len(direction) != 1:
        raise RuntimeError("Step23A direction snapshot must contain exactly one row.")
    if str(direction.iloc[0]["frozen_direction"]) != "high_in_tumor":
        raise RuntimeError("Step23C refuses PC6 sign tuning: frozen direction is not high_in_tumor.")

    distinct = pd.read_csv(STEP23A_DISTINCT, low_memory=False)
    distinct_text = " ".join(distinct.astype(str).fillna("").to_numpy().ravel()).lower()
    if PRIMARY_DISTINCTNESS_ENDPOINT.lower() not in distinct_text:
        raise RuntimeError(
            "Step23A distinctness authority does not contain the frozen primary endpoint: "
            + PRIMARY_DISTINCTNESS_ENDPOINT
        )

    stats15a = plan15a.get("statistical_tests", {})
    n_boot = int(
        plan23a.get("bootstrap", {}).get(
            "n_resamples",
            plan23a.get("bootstrap", {}).get(
                "resamples",
                stats15a.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES),
            ),
        )
    )
    seed = int(
        plan23a.get("bootstrap", {}).get(
            "seed",
            stats15a.get("random_seed", DEFAULT_BOOTSTRAP_SEED),
        )
    )

    gate15a = plan15a.get("inference_qc_gate", {})
    min_pairs = int(gate15a.get("minimum_paired_patients", 10))

    return {
        "plan23a": plan23a,
        "meta23a": meta23a,
        "plan15a": plan15a,
        "n_boot": n_boot,
        "seed": seed,
        "minimum_paired_patients": min_pairs,
        "pc6_direction": "high_in_tumor",
    }


def validate_step23b_authority() -> Tuple[Dict[str, Any], pd.DataFrame]:
    for p, label in [
        (STEP23B_SAMPLE, "Step23B frozen sample manifest"),
        (STEP23B_TILE, "Step23B QC-pass tile morphology"),
        (STEP23B_FAILURE, "Step23B QC failures"),
        (STEP23B_PATIENT, "Step23B patient summary"),
        (STEP23B_AUDIT, "Step23B sampling/QC audit"),
        (STEP23B_COHORT, "Step23B canonical cohort snapshot"),
        (STEP23B_COHORT_AUDIT, "Step23B cohort audit"),
        (STEP23B_HASHES, "Step23B source hashes"),
        (STEP23B_METADATA, "Step23B metadata"),
    ]:
        require_file(p, label)

    meta = load_json(STEP23B_METADATA)

    checks = []

    def add(item: str, observed: Any, expected: Any, passed: bool) -> None:
        checks.append({
            "audit_domain": "step23b_authority",
            "audit_item": item,
            "observed": observed,
            "expected": expected,
            "pass": bool(passed),
        })

    add("formal_full_run", meta.get("formal_full_run"), True, meta.get("formal_full_run") is True)
    add(
        "inferential_statistics_performed",
        meta.get("firewall", {}).get("inferential_statistics_performed"),
        False,
        meta.get("firewall", {}).get("inferential_statistics_performed") is False,
    )
    add(
        "pc3_pc6_distinctness_performed",
        meta.get("firewall", {}).get("pc3_pc6_distinctness_performed"),
        False,
        meta.get("firewall", {}).get("pc3_pc6_distinctness_performed") is False,
    )
    add(
        "observed_slides",
        meta.get("canonical_cohort", {}).get("observed_slides"),
        EXPECTED_SLIDES,
        int(meta.get("canonical_cohort", {}).get("observed_slides", -1)) == EXPECTED_SLIDES,
    )
    add(
        "observed_patients",
        meta.get("canonical_cohort", {}).get("observed_patients"),
        EXPECTED_PATIENTS,
        int(meta.get("canonical_cohort", {}).get("observed_patients", -1)) == EXPECTED_PATIENTS,
    )
    add(
        "filesystem_discovery_used",
        meta.get("canonical_cohort", {}).get("filesystem_discovery_used"),
        False,
        meta.get("canonical_cohort", {}).get("filesystem_discovery_used") is False,
    )
    add(
        "pc6_sign_tuning",
        meta.get("firewall", {}).get("pc6_sign_tuning"),
        False,
        meta.get("firewall", {}).get("pc6_sign_tuning") is False,
    )

    for ext in ["aggc_files_read", "peso_files_read", "tcga_files_read"]:
        add(
            ext,
            meta.get("firewall", {}).get(ext),
            False,
            meta.get("firewall", {}).get(ext) is False,
        )

    audit = pd.DataFrame(checks)
    bad = audit.loc[~audit["pass"]]
    if not bad.empty:
        raise RuntimeError(
            "Step23B authority validation failed:\n" + bad.to_string(index=False)
        )
    return meta, audit


# =============================================================================
# STRICT DEDUP / INTEGRITY
# =============================================================================

def duplicate_rows(
    df: pd.DataFrame,
    subset: Sequence[str],
    source_name: str,
    duplicate_type: str,
) -> pd.DataFrame:
    missing = [c for c in subset if c not in df.columns]
    if missing:
        raise RuntimeError(f"{source_name} missing duplicate-key columns: {missing}")
    mask = df.duplicated(list(subset), keep=False)
    if not mask.any():
        return pd.DataFrame()
    out = df.loc[mask].copy()
    out.insert(0, "duplicate_type", duplicate_type)
    out.insert(0, "source_name", source_name)
    return out


def validate_step23b_dedup_and_partition(
    patient_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(STEP23B_SAMPLE, low_memory=False)
    tile = pd.read_csv(STEP23B_TILE, low_memory=False)
    failures = pd.read_csv(STEP23B_FAILURE, low_memory=False)

    duplicate_frames: List[pd.DataFrame] = []
    integrity_rows: List[Dict[str, Any]] = []

    def record(item: str, observed: Any, expected: Any, passed: bool) -> None:
        integrity_rows.append({
            "audit_domain": "dedup_and_partition",
            "audit_item": item,
            "observed": observed,
            "expected": expected,
            "pass": bool(passed),
        })

    for df, source, keys, dtype in [
        (sample, "step23b_sample", ["sample_id"], "duplicate_sample_id"),
        (
            sample,
            "step23b_sample",
            ["resolved_wsi_path", "x", "y", "patch_size"],
            "duplicate_physical_tile",
        ),
        (tile, "step23b_tile_qc_pass", ["sample_id"], "duplicate_qc_pass_sample_id"),
        (
            patient_df,
            "step23b_patient_summary",
            ["patient_id", "sampling_stratum"],
            "duplicate_patient_stratum",
        ),
    ]:
        dup = duplicate_rows(df, keys, source, dtype)
        if not dup.empty:
            duplicate_frames.append(dup)
        record(
            f"{source}:{dtype}",
            int(len(dup)),
            0,
            dup.empty,
        )

    if not failures.empty and "sample_id" in failures.columns:
        dup = duplicate_rows(
            failures,
            ["sample_id"],
            "step23b_qc_failures",
            "duplicate_failure_sample_id",
        )
        if not dup.empty:
            duplicate_frames.append(dup)
        record("step23b_qc_failures:duplicate_sample_id", int(len(dup)), 0, dup.empty)

    sample_ids = set(sample["sample_id"].astype(str))
    pass_ids = set(tile["sample_id"].astype(str))
    fail_ids = set(failures["sample_id"].astype(str)) if not failures.empty else set()

    overlap = pass_ids & fail_ids
    union = pass_ids | fail_ids

    record("qc_pass_failure_overlap", len(overlap), 0, len(overlap) == 0)
    record("qc_partition_matches_frozen_sample", len(union ^ sample_ids), 0, union == sample_ids)
    record("n_frozen_sample_ids", len(sample_ids), 1040, len(sample_ids) == 1040)

    patient_df = patient_df.copy()
    patient_df["patient_id"] = patient_df["patient_id"].astype(str)
    patient_df["sampling_stratum"] = patient_df["sampling_stratum"].astype(str)

    observed_strata = set(patient_df["sampling_stratum"])
    record(
        "patient_summary_strata",
        "|".join(sorted(observed_strata)),
        "PC6_HIGH|PC6_LOW",
        observed_strata == {LOW_STRATUM, HIGH_STRATUM},
    )

    patients = sorted(patient_df["patient_id"].unique())
    record("patient_summary_n_patients", len(patients), EXPECTED_PATIENTS, len(patients) == EXPECTED_PATIENTS)
    record(
        "patient_summary_n_rows",
        len(patient_df),
        EXPECTED_PATIENTS * 2,
        len(patient_df) == EXPECTED_PATIENTS * 2,
    )

    counts = (
        patient_df.groupby(["patient_id", "sampling_stratum"])
        .size()
        .rename("n_rows")
        .reset_index()
    )
    bad_counts = counts.loc[counts["n_rows"] != 1]
    record("patient_stratum_exactly_one_row", len(bad_counts), 0, bad_counts.empty)

    per_patient_strata = patient_df.groupby("patient_id")["sampling_stratum"].agg(lambda x: tuple(sorted(x)))
    bad_patient_strata = per_patient_strata.loc[
        per_patient_strata.map(lambda x: x != (HIGH_STRATUM, LOW_STRATUM))
    ]
    record("every_patient_has_both_strata_once", len(bad_patient_strata), 0, bad_patient_strata.empty)

    duplicates = (
        pd.concat(duplicate_frames, ignore_index=True, sort=False)
        if duplicate_frames
        else pd.DataFrame(columns=["source_name", "duplicate_type"])
    )
    integrity = pd.DataFrame(integrity_rows)

    if not bad_counts.empty:
        extra = bad_counts.copy()
        extra.insert(0, "duplicate_type", "patient_stratum_count_not_one")
        extra.insert(0, "source_name", "step23b_patient_summary")
        duplicates = pd.concat([duplicates, extra], ignore_index=True, sort=False)

    bad_integrity = integrity.loc[~integrity["pass"]]
    if not bad_integrity.empty:
        # Caller saves both audits before aborting.
        return integrity, duplicates

    return integrity, duplicates


# =============================================================================
# PC6 PATIENT-LEVEL INFERENCE
# =============================================================================

def make_patient_pairs(
    patient_df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    if feature not in patient_df.columns:
        raise RuntimeError(f"Step23B patient summary missing frozen feature: {feature}")

    low = patient_df.loc[
        patient_df["sampling_stratum"].eq(LOW_STRATUM),
        ["patient_id", feature],
    ].copy().rename(columns={feature: "low_value"})

    high = patient_df.loc[
        patient_df["sampling_stratum"].eq(HIGH_STRATUM),
        ["patient_id", feature],
    ].copy().rename(columns={feature: "high_value"})

    if low["patient_id"].duplicated().any() or high["patient_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate patient rows detected while pairing {feature}; dedup gate should have aborted earlier."
        )

    pairs = low.merge(high, on="patient_id", how="outer", validate="one_to_one", indicator=True)

    if not pairs["_merge"].eq("both").all():
        raise RuntimeError(
            f"Missing PC6_LOW/HIGH patient partner for feature {feature}:\n"
            + pairs.loc[~pairs["_merge"].eq("both")].to_string(index=False)
        )

    pairs = pairs.drop(columns="_merge")
    pairs["low_value"] = numeric(pairs["low_value"])
    pairs["high_value"] = numeric(pairs["high_value"])
    pairs["difference_high_minus_low"] = pairs["high_value"] - pairs["low_value"]
    pairs["finite_pair"] = (
        np.isfinite(pairs["low_value"]) & np.isfinite(pairs["high_value"])
    )
    return pairs


def analyze_feature(
    patient_df: pd.DataFrame,
    feature: str,
    endpoint_class: str,
    minimum_paired_patients: int,
    n_boot: int,
    seed: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    pairs_all = make_patient_pairs(patient_df, feature)
    pairs = pairs_all.loc[pairs_all["finite_pair"]].copy()

    low = pairs["low_value"].to_numpy(float)
    high = pairs["high_value"].to_numpy(float)
    d = high - low
    n = int(len(pairs))

    row: Dict[str, Any] = {
        "axis": "PC6",
        "comparison_name": "PC6_HIGH_vs_PC6_LOW",
        "comparison_role": "internal_frozen_extreme_group",
        "low_stratum": LOW_STRATUM,
        "high_stratum": HIGH_STRATUM,
        "feature": feature,
        "endpoint_class": endpoint_class,
        "minimum_paired_patients_required": minimum_paired_patients,
        "n_formal_patients": EXPECTED_PATIENTS,
        "n_paired_patients": n,
        "n_nonfinite_pairs": int((~pairs_all["finite_pair"]).sum()),
        "reported_difference_direction": "PC6_HIGH minus PC6_LOW",
    }

    if n:
        row.update({
            "low_median_across_patients": float(np.median(low)),
            "high_median_across_patients": float(np.median(high)),
            "median_paired_difference_high_minus_low": float(np.median(d)),
            "mean_paired_difference_high_minus_low": float(np.mean(d)),
            "n_high_greater_than_low": int(np.sum(d > 0)),
            "n_high_less_than_low": int(np.sum(d < 0)),
            "n_equal": int(np.sum(d == 0)),
        })
    else:
        row.update({
            "low_median_across_patients": np.nan,
            "high_median_across_patients": np.nan,
            "median_paired_difference_high_minus_low": np.nan,
            "mean_paired_difference_high_minus_low": np.nan,
            "n_high_greater_than_low": 0,
            "n_high_less_than_low": 0,
            "n_equal": 0,
        })

    if n < minimum_paired_patients:
        row.update({
            "inference_status": "insufficient_paired_patients",
            "bootstrap_95ci_lower": np.nan,
            "bootstrap_95ci_upper": np.nan,
            "paired_rank_biserial": np.nan,
            "wilcoxon_statistic": np.nan,
            "p_value_raw": np.nan,
            "n_nonzero_differences": np.nan,
            "directional_sign_consistency_fraction": np.nan,
        })
    else:
        w = safe_wilcoxon(low, high)
        ci_lo, ci_hi = bootstrap_median_ci(d, n_boot, seed)
        n_pos = int(np.sum(d > 0))
        n_neg = int(np.sum(d < 0))
        denom = n_pos + n_neg

        row.update({
            "inference_status": "performed",
            "bootstrap_95ci_lower": ci_lo,
            "bootstrap_95ci_upper": ci_hi,
            "paired_rank_biserial": paired_rank_biserial(d),
            "wilcoxon_statistic": w["wilcoxon_statistic"],
            "p_value_raw": w["p_value_raw"],
            "n_nonzero_differences": w["n_nonzero_differences"],
            "directional_sign_consistency_fraction": (
                max(n_pos, n_neg) / denom if denom else np.nan
            ),
        })

    paired_out = pairs_all.copy()
    paired_out.insert(0, "axis", "PC6")
    paired_out.insert(1, "feature", feature)
    paired_out.insert(2, "endpoint_class", endpoint_class)
    paired_out.insert(3, "comparison_name", "PC6_HIGH_vs_PC6_LOW")
    return row, paired_out


def analyze_family(
    patient_df: pd.DataFrame,
    features: Sequence[str],
    endpoint_class: str,
    family_id: str,
    minimum_paired_patients: int,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pairs = []

    for i, feature in enumerate(features):
        row, pair = analyze_feature(
            patient_df=patient_df,
            feature=feature,
            endpoint_class=endpoint_class,
            minimum_paired_patients=minimum_paired_patients,
            n_boot=n_boot,
            seed=stable_seed(seed, family_id, feature, i),
        )
        rows.append(row)
        pairs.append(pair)

    result = pd.DataFrame(rows)
    result["fdr_family_id"] = family_id
    result["p_value_fdr_bh"] = benjamini_hochberg(result["p_value_raw"].to_numpy(float))

    paired = pd.concat(pairs, ignore_index=True)
    paired["fdr_family_id"] = family_id
    return result, paired


def family_summary(*frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df in frames:
        if df.empty:
            continue
        for family_id, sub in df.groupby("fdr_family_id", sort=False):
            raw = numeric(sub["p_value_raw"])
            fdr = numeric(sub["p_value_fdr_bh"])
            rows.append({
                "fdr_family_id": family_id,
                "endpoint_class": mode_or_none(sub["endpoint_class"]),
                "n_tests_total": int(len(sub)),
                "n_tests_performed": int(sub["inference_status"].eq("performed").sum()),
                "n_raw_p_lt_0_05": int((raw < 0.05).sum()),
                "n_fdr_q_lt_0_05": int((fdr < 0.05).sum()),
                "minimum_raw_p": float(raw.min()) if raw.notna().any() else np.nan,
                "minimum_fdr_q": float(fdr.min()) if fdr.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


# =============================================================================
# PC3 REFERENCE
# =============================================================================

def select_exact_pc3_reference_rows(
    df: pd.DataFrame,
    features: Sequence[str],
    source_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"feature", "paired_rank_biserial", "median_paired_difference_high_minus_low"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{source_name} missing required columns: {sorted(missing)}")

    sub = df.loc[df["feature"].astype(str).isin(features)].copy()

    if "comparison_name" in sub.columns:
        sub = sub.loc[
            sub["comparison_name"].astype(str).str.contains("all_tissue", case=False, na=False)
        ].copy()

    dup = duplicate_rows(
        sub,
        ["feature"],
        source_name,
        "duplicate_pc3_reference_feature",
    )

    counts = sub.groupby("feature").size()
    missing_features = [f for f in features if int(counts.get(f, 0)) == 0]
    multiple_features = [f for f in features if int(counts.get(f, 0)) > 1]

    if missing_features or multiple_features:
        details = {
            "missing_features": missing_features,
            "multiple_features": multiple_features,
        }
        raise RuntimeError(
            f"PC3 frozen reference is not exactly one row per requested feature: {details}"
        )

    return sub.set_index("feature").loc[list(features)].reset_index(), dup


def load_pc3_reference() -> Tuple[pd.DataFrame, pd.DataFrame]:
    for p, label in [
        (STEP15C_PRIMARY, "Step15C primary results"),
        (STEP15C_SECONDARY, "Step15C secondary results"),
        (STEP15C_PAIRED, "Step15C paired patient values"),
        (STEP15C_METADATA, "Step15C metadata"),
    ]:
        require_file(p, label)

    meta = load_json(STEP15C_METADATA)
    if meta.get("tile_level_inference_performed", False):
        raise RuntimeError("Step15C metadata unexpectedly reports tile-level inference.")

    primary = pd.read_csv(STEP15C_PRIMARY, low_memory=False)
    secondary = pd.read_csv(STEP15C_SECONDARY, low_memory=False)

    p_ref, p_dup = select_exact_pc3_reference_rows(
        primary, PRIMARY_FEATURES, "step15c_primary"
    )
    s_ref, s_dup = select_exact_pc3_reference_rows(
        secondary, SECONDARY_FEATURES, "step15c_secondary"
    )

    ref = pd.concat([p_ref, s_ref], ignore_index=True)
    ref["axis"] = "PC3"
    ref["reference_role"] = "frozen_internal_step15c_all_tissue"

    duplicates = pd.concat(
        [x for x in [p_dup, s_dup] if not x.empty],
        ignore_index=True,
        sort=False,
    ) if (not p_dup.empty or not s_dup.empty) else pd.DataFrame(
        columns=["source_name", "duplicate_type"]
    )

    return ref, duplicates


def load_pc3_paired_for_profile() -> Tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(STEP15C_PAIRED, low_memory=False)

    required = {
        "patient_id", "feature", "difference_high_minus_low",
        "comparison_name",
    }
    missing = required - set(paired.columns)
    if missing:
        raise RuntimeError(f"Step15C paired source missing required columns: {sorted(missing)}")

    sub = paired.loc[
        paired["feature"].astype(str).isin(FROZEN_PROFILE_FEATURES)
        & paired["comparison_name"].astype(str).str.contains("all_tissue", case=False, na=False)
    ].copy()

    # Some Step15C versions may include the same feature in multiple endpoint blocks.
    # Exact patient x feature uniqueness is mandatory; no silent dedup.
    dup = duplicate_rows(
        sub,
        ["patient_id", "feature"],
        "step15c_paired",
        "duplicate_pc3_patient_feature",
    )
    if not dup.empty:
        return sub, dup

    counts = sub.groupby("feature")["patient_id"].nunique()
    missing_features = [f for f in FROZEN_PROFILE_FEATURES if int(counts.get(f, 0)) == 0]
    if missing_features:
        raise RuntimeError(
            "Step15C paired source lacks frozen all-tissue paired values for: "
            + ", ".join(missing_features)
        )

    sub["difference_high_minus_low"] = numeric(sub["difference_high_minus_low"])
    return sub, dup


# =============================================================================
# PROFILE DISTINCTNESS
# =============================================================================

def build_effect_profile(
    pc6_primary: pd.DataFrame,
    pc6_secondary: pd.DataFrame,
    pc3_ref: pd.DataFrame,
) -> pd.DataFrame:
    pc6 = pd.concat([pc6_primary, pc6_secondary], ignore_index=True)

    pc6_dup = duplicate_rows(
        pc6,
        ["feature"],
        "pc6_23c_profile",
        "duplicate_pc6_profile_feature",
    )
    if not pc6_dup.empty:
        raise RuntimeError(
            "PC6 profile contains duplicate feature rows:\n"
            + pc6_dup.to_string(index=False)
        )

    pc3_dup = duplicate_rows(
        pc3_ref,
        ["feature"],
        "pc3_step15c_profile",
        "duplicate_pc3_profile_feature",
    )
    if not pc3_dup.empty:
        raise RuntimeError(
            "PC3 profile contains duplicate feature rows:\n"
            + pc3_dup.to_string(index=False)
        )

    pc6_small = pc6[[
        "feature",
        "endpoint_class",
        "n_paired_patients",
        "median_paired_difference_high_minus_low",
        "paired_rank_biserial",
        "p_value_raw",
        "p_value_fdr_bh",
    ]].rename(columns={
        "n_paired_patients": "pc6_n_paired_patients",
        "median_paired_difference_high_minus_low": "pc6_median_difference",
        "paired_rank_biserial": "pc6_paired_rank_biserial",
        "p_value_raw": "pc6_p_value_raw",
        "p_value_fdr_bh": "pc6_p_value_fdr_bh",
    })

    pc3_small = pc3_ref[[
        "feature",
        "n_paired_patients",
        "median_paired_difference_high_minus_low",
        "paired_rank_biserial",
        "p_value_raw",
        "p_value_fdr_bh",
    ]].rename(columns={
        "n_paired_patients": "pc3_n_paired_patients",
        "median_paired_difference_high_minus_low": "pc3_median_difference",
        "paired_rank_biserial": "pc3_paired_rank_biserial",
        "p_value_raw": "pc3_p_value_raw",
        "p_value_fdr_bh": "pc3_p_value_fdr_bh",
    })

    out = pc6_small.merge(pc3_small, on="feature", how="outer", validate="one_to_one")

    order = {f: i for i, f in enumerate(FROZEN_PROFILE_FEATURES)}
    out["_order"] = out["feature"].map(order)
    out = out.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)

    observed = out["feature"].astype(str).tolist()
    if observed != FROZEN_PROFILE_FEATURES:
        raise RuntimeError(
            "Complete 13-feature profile is not evaluable in frozen order.\n"
            f"Expected={FROZEN_PROFILE_FEATURES}\nObserved={observed}"
        )

    out["pc6_minus_pc3_rank_biserial"] = (
        numeric(out["pc6_paired_rank_biserial"]) - numeric(out["pc3_paired_rank_biserial"])
    )
    out["absolute_pc6_minus_pc3_rank_biserial"] = np.abs(out["pc6_minus_pc3_rank_biserial"])

    s3 = np.sign(numeric(out["pc3_paired_rank_biserial"]).to_numpy(float))
    s6 = np.sign(numeric(out["pc6_paired_rank_biserial"]).to_numpy(float))
    out["same_effect_sign"] = s3 == s6
    out["significance_filter_used_for_profile"] = False
    out["profile_effect_measure"] = "paired_rank_biserial_signed_dimensionless"
    return out


def summarize_distinctness(profile: pd.DataFrame) -> pd.DataFrame:
    x = numeric(profile["pc3_paired_rank_biserial"]).to_numpy(float)
    y = numeric(profile["pc6_paired_rank_biserial"]).to_numpy(float)

    complete = np.isfinite(x) & np.isfinite(y)
    if int(complete.sum()) != len(FROZEN_PROFILE_FEATURES):
        raise RuntimeError(
            "Primary complete-profile endpoint requires all 13 finite PC3 and PC6 rank-biserial effects."
        )

    cosine, angle = cosine_and_angle(x, y)

    primary_mask = profile["feature"].isin(PRIMARY_FEATURES).to_numpy()
    p3 = x[primary_mask]
    p6 = y[primary_mask]

    rows = [
        {
            "endpoint": PRIMARY_DISTINCTNESS_ENDPOINT,
            "endpoint_role": "primary",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(x),
            "value": safe_spearman(x, y, minimum_n=3),
            "interpretation_scale": "Spearman rho; +1 identical rank-order/sign pattern, -1 opposing rank-order pattern",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
        {
            "endpoint": "signed_effect_profile_pearson",
            "endpoint_role": "supportive",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(x),
            "value": safe_pearson(x, y, minimum_n=3),
            "interpretation_scale": "Pearson r across complete signed effect-size vectors",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
        {
            "endpoint": "effect_vector_cosine_similarity",
            "endpoint_role": "supportive",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(x),
            "value": cosine,
            "interpretation_scale": "cosine similarity",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
        {
            "endpoint": "effect_vector_angle_degrees",
            "endpoint_role": "supportive",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(x),
            "value": angle,
            "interpretation_scale": "degrees",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
        {
            "endpoint": "same_sign_feature_fraction",
            "endpoint_role": "supportive",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(x),
            "value": float(profile["same_effect_sign"].mean()),
            "interpretation_scale": "fraction of 13 frozen features with same effect sign",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
        {
            "endpoint": "primary_three_feature_profile_spearman",
            "endpoint_role": "supportive",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(p3),
            "value": safe_spearman(p3, p6, minimum_n=3),
            "interpretation_scale": "Spearman rho across SEI/GOI/NPI only",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
        {
            "endpoint": "median_absolute_cross_axis_effect_difference",
            "endpoint_role": "supportive",
            "effect_measure": "paired_rank_biserial_signed_dimensionless",
            "n_features": len(x),
            "value": float(np.median(np.abs(y - x))),
            "interpretation_scale": "median absolute difference in paired rank-biserial effect",
            "posthoc_binary_cutoff_used": False,
            "significance_filter_used": False,
        },
    ]
    return pd.DataFrame(rows)


def bootstrap_profile_distinctness(
    pc3_pairs: pd.DataFrame,
    pc6_pairs: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Build patient x feature difference matrices for each axis.
    pc3 = pc3_pairs.pivot(index="patient_id", columns="feature", values="difference_high_minus_low")
    pc6 = pc6_pairs.loc[
        pc6_pairs["feature"].isin(FROZEN_PROFILE_FEATURES)
    ].pivot(index="patient_id", columns="feature", values="difference_high_minus_low")

    for feature in FROZEN_PROFILE_FEATURES:
        if feature not in pc3.columns or feature not in pc6.columns:
            raise RuntimeError(f"Bootstrap profile missing feature {feature} in PC3 or PC6 paired values.")

    common_patients = sorted(set(pc3.index.astype(str)) & set(pc6.index.astype(str)))
    if len(common_patients) < 3:
        raise RuntimeError("Too few common PC3/PC6 patients for profile bootstrap.")

    pc3.index = pc3.index.astype(str)
    pc6.index = pc6.index.astype(str)
    pc3 = pc3.loc[common_patients, FROZEN_PROFILE_FEATURES].apply(pd.to_numeric, errors="coerce")
    pc6 = pc6.loc[common_patients, FROZEN_PROFILE_FEATURES].apply(pd.to_numeric, errors="coerce")

    # Complete-case patient requirement for joint bootstrap prevents feature-specific
    # changing patient composition inside a profile replicate.
    complete_patient = np.isfinite(pc3.to_numpy(float)).all(axis=1) & np.isfinite(pc6.to_numpy(float)).all(axis=1)
    complete_patients = [p for p, keep in zip(common_patients, complete_patient) if keep]

    if len(complete_patients) < 3:
        raise RuntimeError(
            "Fewer than 3 common patients have complete paired values across all 13 features for both axes."
        )

    a = pc3.loc[complete_patients].to_numpy(float)
    b = pc6.loc[complete_patients].to_numpy(float)

    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_boot):
        idx = rng.integers(0, len(complete_patients), size=len(complete_patients))
        rb3 = np.array([paired_rank_biserial(a[idx, j]) for j in range(a.shape[1])], float)
        rb6 = np.array([paired_rank_biserial(b[idx, j]) for j in range(b.shape[1])], float)

        rho = safe_spearman(rb3, rb6, minimum_n=3)
        pear = safe_pearson(rb3, rb6, minimum_n=3)
        cos, angle = cosine_and_angle(rb3, rb6)
        same_sign = float(np.mean(np.sign(rb3) == np.sign(rb6)))
        mad = float(np.median(np.abs(rb6 - rb3)))

        rows.append({
            "bootstrap_id": i,
            "n_resampled_patients": len(complete_patients),
            "profile_spearman": rho,
            "profile_pearson": pear,
            "cosine_similarity": cos,
            "angle_degrees": angle,
            "same_sign_fraction": same_sign,
            "median_absolute_effect_difference": mad,
        })

    boot = pd.DataFrame(rows)

    def ci(col: str) -> Tuple[float, float]:
        x = numeric(boot[col]).dropna().to_numpy(float)
        if len(x) == 0:
            return np.nan, np.nan
        return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))

    summary = {
        "joint_bootstrap_resampling_unit": "patient",
        "joint_bootstrap_same_patient_indices_for_pc3_pc6": True,
        "n_common_patients_before_complete_case": len(common_patients),
        "n_complete_common_patients": len(complete_patients),
        "complete_common_patients": complete_patients,
        "profile_spearman_bootstrap_95ci": ci("profile_spearman"),
        "profile_pearson_bootstrap_95ci": ci("profile_pearson"),
        "cosine_bootstrap_95ci": ci("cosine_similarity"),
        "angle_bootstrap_95ci": ci("angle_degrees"),
        "same_sign_fraction_bootstrap_95ci": ci("same_sign_fraction"),
        "median_absolute_effect_difference_bootstrap_95ci": ci("median_absolute_effect_difference"),
    }
    return boot, summary


# =============================================================================
# PROVENANCE / README
# =============================================================================

def source_hash_table() -> pd.DataFrame:
    sources = [
        ("step23a_plan", STEP23A_PLAN),
        ("step23a_metrics", STEP23A_METRICS),
        ("step23a_distinctness", STEP23A_DISTINCT),
        ("step23a_direction", STEP23A_DIRECTION),
        ("step23a_metadata", STEP23A_METADATA),
        ("step23b_sample", STEP23B_SAMPLE),
        ("step23b_tile", STEP23B_TILE),
        ("step23b_failure", STEP23B_FAILURE),
        ("step23b_patient", STEP23B_PATIENT),
        ("step23b_audit", STEP23B_AUDIT),
        ("step23b_cohort", STEP23B_COHORT),
        ("step23b_cohort_audit", STEP23B_COHORT_AUDIT),
        ("step23b_source_hashes", STEP23B_HASHES),
        ("step23b_metadata", STEP23B_METADATA),
        ("step15a_plan", STEP15A_PLAN),
        ("step15c_primary", STEP15C_PRIMARY),
        ("step15c_secondary", STEP15C_SECONDARY),
        ("step15c_paired", STEP15C_PAIRED),
        ("step15c_metadata", STEP15C_METADATA),
    ]

    rows = []
    for role, path in sources:
        require_file(path, role)
        low = str(path).lower()
        if any(x in low for x in ["aggc", "peso", "tcga"]):
            raise RuntimeError(f"External source firewall violation: {path}")
        rows.append({
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)


def readme_text(
    contract: Dict[str, Any],
    bootstrap_summary: Dict[str, Any],
) -> str:
    return f"""STEP 23C - INTERNAL PC6 MORPHOLOGY INFERENCE + PC3-vs-PC6 DISTINCTNESS
================================================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

ROLE
----
INFERENCE ONLY.

PC6 measurement authority
-------------------------
{STEP23B_DIR}

No morphology is remeasured in Step23C.

Independent unit
----------------
patient

PC6 comparison
--------------
PC6_HIGH - PC6_LOW

Primary features
----------------
{chr(10).join(PRIMARY_FEATURES)}

Secondary features
------------------
{chr(10).join(SECONDARY_FEATURES)}

Sensitivity features
--------------------
{chr(10).join(SENSITIVITY_FEATURES)}

Statistics
----------
Two-sided paired Wilcoxon signed-rank.
Paired rank-biserial effect size.
Median paired difference.
95% patient bootstrap CI.
BH-FDR within primary, secondary, and sensitivity families.

Minimum paired patients
-----------------------
{contract['minimum_paired_patients']}

Bootstrap
---------
resamples = {contract['n_boot']}
seed      = {contract['seed']}
unit      = patient

PC3 reference
-------------
Frozen INTERNAL Step15C all-tissue PC3-high minus PC3-low patient-level inference:
{STEP15C_DIR}

PC3 is NOT redefined or remeasured here.

PC3-vs-PC6 morphology-profile distinctness
------------------------------------------
Complete frozen profile = 13 features:
  3 primary + 10 secondary.

Primary profile effect measure:
  paired rank-biserial effect size

Rationale:
  it is signed and dimensionless, so heterogeneous morphology units are not
  directly compared on incompatible raw scales.

Primary endpoint:
  {PRIMARY_DISTINCTNESS_ENDPOINT}
  operationalized as Spearman correlation across the complete 13-feature
  paired-rank-biserial profiles.

Supportive:
  Pearson profile correlation
  cosine similarity
  vector angle
  same-sign fraction
  primary-three-feature Spearman
  median absolute cross-axis effect-size difference

Joint PC3/PC6 profile bootstrap
-------------------------------
same patient bootstrap indices for both axes = YES
common patients before complete-case         = {bootstrap_summary['n_common_patients_before_complete_case']}
complete common patients                     = {bootstrap_summary['n_complete_common_patients']}

No binary post-hoc distinctness cutoff is used.

STRICT DEDUPLICATION
--------------------
No silent drop_duplicates().

The run aborts on:
  duplicate Step23B sample_id
  duplicate physical tile identity
  duplicate QC-pass sample_id
  duplicate QC-failure sample_id
  overlap between QC-pass and QC-failure IDs
  failure of pass+failure IDs to partition the frozen sample set
  duplicate patient x PC6 stratum
  missing patient stratum
  duplicate PC3 frozen reference feature
  duplicate PC3 patient x feature pair

FIREWALL
--------
AGGC read                    : NO
PESO read                    : NO
TCGA read                    : NO
PCA refit                    : NO
PC screening                 : NO
PC6 sign tuning              : NO
PC5 promotion                : NO
feature selection            : NO
measurement threshold change : NO
manual patient/tile selection: NO
tile-level inference         : NO

INTERPRETATION CEILING
----------------------
This analysis can quantify whether PC3 and PC6 have non-identical morphology
effect profiles. It does NOT prove causal biological independence and does not
use PCA orthogonality as biological evidence.
"""


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Internal PC6 patient-level inference and PC3-vs-PC6 morphology-profile distinctness."
    )
    p.add_argument("--output", type=Path, default=OUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    banner("STEP 23C: INTERNAL PC6 MORPHOLOGY INFERENCE + PC3-vs-PC6 DISTINCTNESS")
    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP23A]       {STEP23A_DIR}")
    print(f"[STEP23B]       {STEP23B_DIR}")
    print(f"[STEP15C PC3]   {STEP15C_DIR}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          INFERENCE ONLY")
    print("[UNIT]          PATIENT")
    print("[PC6 EFFECT]    PC6_HIGH - PC6_LOW")
    print("[DEDUP]         STRICT / ABORT ON DUPLICATE")
    print("[AGGC READ]     NONE")
    print("[PESO READ]     NONE")
    print("[TCGA READ]     NONE")
    print("[PCA REFIT]     NONE")
    print("[PC SCREEN]     NONE")
    print("[PC6 SIGN]      FROZEN high_in_tumor")
    print("[PC5]           NOT PROMOTED")
    print("[MEASUREMENT]   NONE")

    prepare_output(args.output, args.overwrite)

    banner("VALIDATE FROZEN AUTHORITIES")
    contract = load_contract()
    meta23b, authority_audit = validate_step23b_authority()
    print(f"Step23A version             : {contract['plan23a'].get('version')}")
    print(f"PC6 frozen direction        : {contract['pc6_direction']}")
    print(f"Bootstrap resamples         : {contract['n_boot']}")
    print(f"Bootstrap seed              : {contract['seed']}")
    print(f"Minimum paired patients     : {contract['minimum_paired_patients']}")
    print("Step23B formal full run     : PASS")

    banner("STRICT DEDUPLICATION / INPUT INTEGRITY")
    patient_df = pd.read_csv(STEP23B_PATIENT, low_memory=False)

    missing_features = [f for f in ALL_INFERENCE_FEATURES if f not in patient_df.columns]
    if missing_features:
        raise RuntimeError(
            "Step23B patient summary missing frozen inference features: "
            + ", ".join(missing_features)
        )

    integrity, duplicate_audit = validate_step23b_dedup_and_partition(patient_df)

    # Load PC3 sources now so their duplicates are included in the same audit.
    pc3_ref, pc3_ref_dup = load_pc3_reference()
    pc3_pairs, pc3_pair_dup = load_pc3_paired_for_profile()

    if not pc3_ref_dup.empty:
        duplicate_audit = pd.concat([duplicate_audit, pc3_ref_dup], ignore_index=True, sort=False)
    if not pc3_pair_dup.empty:
        duplicate_audit = pd.concat([duplicate_audit, pc3_pair_dup], ignore_index=True, sort=False)

    pc3_integrity_rows = [
        {
            "audit_domain": "pc3_reference",
            "audit_item": "duplicate_reference_feature_rows",
            "observed": int(len(pc3_ref_dup)),
            "expected": 0,
            "pass": pc3_ref_dup.empty,
        },
        {
            "audit_domain": "pc3_reference",
            "audit_item": "duplicate_patient_feature_rows",
            "observed": int(len(pc3_pair_dup)),
            "expected": 0,
            "pass": pc3_pair_dup.empty,
        },
        {
            "audit_domain": "pc3_reference",
            "audit_item": "complete_13_feature_reference",
            "observed": int(pc3_ref["feature"].nunique()),
            "expected": 13,
            "pass": int(pc3_ref["feature"].nunique()) == 13,
        },
    ]

    integrity = pd.concat(
        [authority_audit, integrity, pd.DataFrame(pc3_integrity_rows)],
        ignore_index=True,
        sort=False,
    )

    integrity.to_csv(OUT_INTEGRITY, index=False)
    duplicate_audit.to_csv(OUT_DUPLICATES, index=False)

    bad = integrity.loc[~integrity["pass"].astype(bool)]
    if not bad.empty or not duplicate_audit.empty:
        print("\n[DEDUP/INTEGRITY] FAIL")
        if not bad.empty:
            print(bad.to_string(index=False))
        if not duplicate_audit.empty:
            print("\n[DUPLICATES]")
            print(duplicate_audit.head(100).to_string(index=False))
        raise RuntimeError(
            "Step23C aborted because strict deduplication/input integrity failed. "
            f"Inspect:\n  {OUT_INTEGRITY}\n  {OUT_DUPLICATES}"
        )

    print(integrity.to_string(index=False))
    print("\nStrict deduplication gate: PASS")

    banner("PC6 PATIENT-LEVEL INFERENCE")
    primary_df, primary_pairs = analyze_family(
        patient_df,
        PRIMARY_FEATURES,
        "primary",
        "pc6_primary",
        contract["minimum_paired_patients"],
        contract["n_boot"],
        contract["seed"],
    )

    secondary_df, secondary_pairs = analyze_family(
        patient_df,
        SECONDARY_FEATURES,
        "secondary",
        "pc6_secondary",
        contract["minimum_paired_patients"],
        contract["n_boot"],
        contract["seed"],
    )

    sensitivity_df, sensitivity_pairs = analyze_family(
        patient_df,
        SENSITIVITY_FEATURES,
        "sensitivity",
        "pc6_sensitivity",
        contract["minimum_paired_patients"],
        contract["n_boot"],
        contract["seed"],
    )

    paired_df = pd.concat(
        [primary_pairs, secondary_pairs, sensitivity_pairs],
        ignore_index=True,
        sort=False,
    )

    # Inference output itself must remain unique at patient x feature.
    paired_dup = duplicate_rows(
        paired_df,
        ["patient_id", "feature"],
        "step23c_pc6_paired",
        "duplicate_pc6_patient_feature",
    )
    if not paired_dup.empty:
        paired_dup.to_csv(OUT_DUPLICATES, index=False)
        raise RuntimeError(
            "Internal Step23C construction produced duplicate patient x feature rows; aborting."
        )

    family_df = family_summary(primary_df, secondary_df, sensitivity_df)

    primary_df.to_csv(OUT_PRIMARY, index=False)
    secondary_df.to_csv(OUT_SECONDARY, index=False)
    sensitivity_df.to_csv(OUT_SENSITIVITY, index=False)
    paired_df.to_csv(OUT_PAIRED, index=False)
    family_df.to_csv(OUT_FAMILY, index=False)

    display_cols = [
        "feature",
        "inference_status",
        "n_paired_patients",
        "median_paired_difference_high_minus_low",
        "bootstrap_95ci_lower",
        "bootstrap_95ci_upper",
        "paired_rank_biserial",
        "p_value_raw",
        "p_value_fdr_bh",
        "n_high_greater_than_low",
        "n_high_less_than_low",
    ]

    print("\nPRIMARY")
    print(primary_df[display_cols].to_string(index=False))
    print("\nSECONDARY")
    print(secondary_df[display_cols].to_string(index=False))
    print("\nSENSITIVITY")
    print(sensitivity_df[display_cols].to_string(index=False))

    banner("BUILD FROZEN PC3 REFERENCE + PC3-vs-PC6 EFFECT PROFILE")
    pc3_ref.to_csv(OUT_PC3_REF, index=False)

    profile = build_effect_profile(primary_df, secondary_df, pc3_ref)
    profile.to_csv(OUT_PROFILE, index=False)
    print(profile[[
        "feature",
        "pc3_paired_rank_biserial",
        "pc6_paired_rank_biserial",
        "pc6_minus_pc3_rank_biserial",
        "same_effect_sign",
    ]].to_string(index=False))

    banner("PC3-vs-PC6 COMPLETE-PROFILE DISTINCTNESS")
    distinct = summarize_distinctness(profile)

    # Joint patient bootstrap needs only primary+secondary PC6 paired values.
    pc6_profile_pairs = paired_df.loc[
        paired_df["feature"].isin(FROZEN_PROFILE_FEATURES)
    ].copy()

    boot, boot_summary = bootstrap_profile_distinctness(
        pc3_pairs=pc3_pairs,
        pc6_pairs=pc6_profile_pairs,
        n_boot=contract["n_boot"],
        seed=stable_seed(contract["seed"], "pc3_pc6_joint_profile_bootstrap"),
    )

    # Attach bootstrap CIs to matching rows.
    ci_map = {
        PRIMARY_DISTINCTNESS_ENDPOINT: boot_summary["profile_spearman_bootstrap_95ci"],
        "signed_effect_profile_pearson": boot_summary["profile_pearson_bootstrap_95ci"],
        "effect_vector_cosine_similarity": boot_summary["cosine_bootstrap_95ci"],
        "effect_vector_angle_degrees": boot_summary["angle_bootstrap_95ci"],
        "same_sign_feature_fraction": boot_summary["same_sign_fraction_bootstrap_95ci"],
        "median_absolute_cross_axis_effect_difference":
            boot_summary["median_absolute_effect_difference_bootstrap_95ci"],
    }
    distinct["bootstrap_95ci_lower"] = np.nan
    distinct["bootstrap_95ci_upper"] = np.nan
    for endpoint, (lo, hi) in ci_map.items():
        mask = distinct["endpoint"].eq(endpoint)
        distinct.loc[mask, "bootstrap_95ci_lower"] = lo
        distinct.loc[mask, "bootstrap_95ci_upper"] = hi

    distinct["n_common_complete_patients_for_joint_bootstrap"] = (
        boot_summary["n_complete_common_patients"]
    )
    distinct.to_csv(OUT_DISTINCT, index=False)
    boot.to_csv(OUT_DISTINCT_BOOT, index=False, compression="gzip")

    print(distinct.to_string(index=False))

    banner("PROVENANCE + FIREWALL")
    hashes = source_hash_table()
    hashes.to_csv(OUT_HASHES, index=False)

    metadata = {
        "analysis_name": "Internal PC6 quantitative morphology inference and PC3-vs-PC6 morphology-profile distinctness",
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_required": True,
        "role": "inference_only",
        "step23a_version": contract["plan23a"].get("version"),
        "step23b_version": meta23b.get("version"),
        "step23b_formal_full_run_verified": True,
        "independent_unit": "patient",
        "expected_patients": EXPECTED_PATIENTS,
        "expected_slides": EXPECTED_SLIDES,
        "pc6_comparison": "PC6_HIGH minus PC6_LOW",
        "pc6_frozen_direction": contract["pc6_direction"],
        "primary_features": PRIMARY_FEATURES,
        "secondary_features": SECONDARY_FEATURES,
        "sensitivity_features": SENSITIVITY_FEATURES,
        "bootstrap_resamples": contract["n_boot"],
        "bootstrap_seed": contract["seed"],
        "minimum_paired_patients": contract["minimum_paired_patients"],
        "statistics": {
            "paired_test": "two-sided Wilcoxon signed-rank",
            "effect_size": "paired rank-biserial",
            "location_effect": "median paired difference high-minus-low",
            "confidence_interval": "95% patient bootstrap percentile CI",
            "multiple_testing": "BH-FDR within prespecified primary/secondary/sensitivity families",
        },
        "distinctness": {
            "primary_endpoint": PRIMARY_DISTINCTNESS_ENDPOINT,
            "profile_features": FROZEN_PROFILE_FEATURES,
            "profile_effect_measure": "paired_rank_biserial_signed_dimensionless",
            "complete_profile_required": True,
            "significance_filtering_used": False,
            "posthoc_binary_distinctness_cutoff_used": False,
            "joint_patient_bootstrap": boot_summary,
        },
        "deduplication": {
            "silent_drop_duplicates_used": False,
            "duplicate_policy": "audit_and_abort",
            "step23b_sample_id_unique_verified": True,
            "step23b_physical_tile_unique_verified": True,
            "step23b_pass_failure_disjoint_verified": True,
            "step23b_pass_failure_partition_verified": True,
            "patient_stratum_unique_verified": True,
            "pc3_reference_feature_unique_verified": True,
            "pc3_patient_feature_unique_verified": True,
        },
        "firewall": {
            "aggc_files_read": False,
            "peso_files_read": False,
            "tcga_files_read": False,
            "pca_refit": False,
            "pc_screening": False,
            "pc6_sign_tuning": False,
            "pc5_promotion": False,
            "morphology_remeasurement": False,
            "measurement_threshold_change": False,
            "manual_patient_selection": False,
            "manual_tile_selection": False,
            "tile_level_inference": False,
            "significance_based_feature_selection_for_distinctness": False,
        },
    }
    save_json(OUT_METADATA, metadata)
    OUT_README.write_text(readme_text(contract, boot_summary), encoding="utf-8")

    print("[DEDUP] Step23B sample_id unique                    : YES")
    print("[DEDUP] Step23B physical tile identity unique       : YES")
    print("[DEDUP] QC-pass / QC-failure IDs disjoint            : YES")
    print("[DEDUP] QC-pass + failure partition frozen sample    : YES")
    print("[DEDUP] patient x PC6 stratum exactly one row        : YES")
    print("[DEDUP] PC3 feature exactly one reference row        : YES")
    print("[DEDUP] PC3 patient x feature exactly one row        : YES")
    print("[AGGC READ]                                           : NO")
    print("[PESO READ]                                           : NO")
    print("[TCGA READ]                                           : NO")
    print("[PCA REFIT]                                           : NO")
    print("[PC SCREEN]                                           : NO")
    print("[SIGN TUNING]                                         : NO")
    print("[MEASUREMENT]                                         : NO")

    banner("SAVED OUTPUTS")
    expected_outputs = [
        OUT_INTEGRITY,
        OUT_DUPLICATES,
        OUT_PAIRED,
        OUT_PRIMARY,
        OUT_SECONDARY,
        OUT_SENSITIVITY,
        OUT_FAMILY,
        OUT_PC3_REF,
        OUT_PROFILE,
        OUT_DISTINCT,
        OUT_DISTINCT_BOOT,
        OUT_HASHES,
        OUT_METADATA,
        OUT_README,
    ]
    for p in expected_outputs:
        if not p.exists():
            raise RuntimeError(f"Expected output missing: {p}")
        print(f"  {p}")

    print("\n[DONE] Step23C INTERNAL PC6 inference + PC3-vs-PC6 morphology distinctness complete.")
    print("[NEXT] Freeze the internal PC6 phenotype/result before any external PC6 transfer analysis.")


if __name__ == "__main__":
    main()
