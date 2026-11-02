#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 24C
==========
EVALUATE FROZEN AGGC PC6 MORPHOLOGY TRANSFER

VERSION
-------
1.0.0

SCHEMA
------
24c-frozen-aggc-pc6-morphology-transfer-evaluation-schema-1.0

ROLE
----
FIRST FORMAL EXTERNAL MORPHOLOGY-TRANSFER INFERENCE.

This step is the first point in the Step24 chain where PC6_HIGH - PC6_LOW
morphology effects are calculated.

UPSTREAM AUTHORITIES
--------------------
Step24A:
    scientific freeze:
        - target axis = Virchow2 PC6
        - frozen direction = high_in_tumor
        - PC6_LOW  <= within-slide q20
        - PC6_HIGH >= within-slide q80
        - six formal morphology transfer endpoints
        - expected external direction for each endpoint
        - Test = PRIMARY held-out replication
        - Train = SECONDARY characterization
        - NPI = specificity reference only
        - bootstrap resamples / seed
        - no Train/Test pooling

Step24B3:
    final complete frozen measurement authority:
        - Test: 45 slides / 900 sampled tiles
        - Train: 80 slides / 1600 sampled tiles
        - slide x stratum morphology medians
        - no prior HIGH-minus-LOW effect calculation

FORMAL ENDPOINTS
----------------
Expected PC6_HIGH - PC6_LOW direction:

    SEI_eosin_stroma        > 0
    GOI_lumen_openness      < 0
    CLR_nuclear_to_lumen    > 0
    lumen_area              < 0
    mean_lumen_area         < 0
    lumen_fragmentation     > 0

PRIMARY INFERENCE
-----------------
Partition:
    Subset1_Test

Analysis unit:
    slide

For each endpoint and slide:
    effect = morphology_median(PC6_HIGH) - morphology_median(PC6_LOW)

Metric-level primary estimate:
    median slide-level effect

Uncertainty:
    nonparametric slide bootstrap of the median
    using Step24A-frozen resamples / seed

Directional replication:
    sign(median effect) matches frozen expected sign

Family-level primary summary:
    number of replicated formal endpoints / 6

SECONDARY
---------
Subset1_Train:
    same slide-level effect / median / bootstrap CI,
    characterization only.
    It may not replace or rescue Test.

SPECIFICITY REFERENCE
---------------------
NPI_nuclear_packing:
    evaluated descriptively / supportively only.
    It is not part of the 6-endpoint concordance family and cannot be promoted
    into the formal PC6 phenotype based on AGGC.

P-VALUES
--------
This Step24 chain froze directional replication + bootstrap CI, not a new
multiplicity-driven null-hypothesis testing family. Therefore the formal
external-transfer inference is based on effect direction and bootstrap
uncertainty.

No new p-value/FDR endpoint family is introduced here.

PROHIBITED
----------
- endpoint reselection;
- expected-direction tuning;
- PC6 sign tuning;
- NPI promotion;
- Train/Test pooling;
- Train result replacing Test;
- tumor AUC;
- PC1-PC8 screening;
- PCA refit / reprojection;
- Gleason-state analysis;
- post-outcome slide exclusion;
- morphology recomputation;
- deterministic sampling rerun.
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


# =============================================================================
# VERSION / PATHS
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "24c-frozen-aggc-pc6-morphology-transfer-evaluation-schema-1.0"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT

STEP24A = RESULTS / "24a_aggc_external_pc6_morphology_transfer_specification"
STEP24B3 = RESULTS / "24b3_complete_aggc_pc6_morphology_transfer_input_freeze"

A_PLAN = STEP24A / "aggc_pc6_external_morphology_transfer_plan.json"
A_ENDPOINTS = STEP24A / "aggc_pc6_frozen_transfer_endpoint_specification.csv"
A_SPECIFICITY = STEP24A / "aggc_pc6_specificity_reference_specification.csv"
A_PARTITIONS = STEP24A / "aggc_pc6_partition_role_specification.csv"
A_EVAL = STEP24A / "aggc_pc6_external_evaluation_endpoint_specification.csv"
A_SAMPLING = STEP24A / "aggc_pc6_sampling_and_eligibility_specification.csv"
A_METADATA = STEP24A / "analysis_metadata.json"
A_CLAIM = STEP24A / "pc6_aggc_external_claim_ceiling.txt"

B3_TILE = STEP24B3 / "aggc_pc6_final_frozen_sampled_tile_measurements.csv.gz"
B3_SLIDE = STEP24B3 / "aggc_pc6_final_slide_stratum_morphology_summary.csv"
B3_PARTITION = STEP24B3 / "aggc_pc6_final_partition_measurement_summary.csv"
B3_PROV = STEP24B3 / "aggc_pc6_final_measurement_provenance.csv"
B3_DUP = STEP24B3 / "aggc_pc6_final_duplicate_audit.csv"
B3_METADATA = STEP24B3 / "analysis_metadata.json"

DEFAULT_OUT = RESULTS / "24c_frozen_aggc_pc6_morphology_transfer_evaluation"

PRIMARY_PARTITION = "Subset1_Test"
SECONDARY_PARTITION = "Subset1_Train"

FORMAL_ENDPOINTS = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "CLR_nuclear_to_lumen",
    "lumen_area",
    "mean_lumen_area",
    "lumen_fragmentation",
]

EXPECTED_SIGNS = {
    "SEI_eosin_stroma": +1,
    "GOI_lumen_openness": -1,
    "CLR_nuclear_to_lumen": +1,
    "lumen_area": -1,
    "mean_lumen_area": -1,
    "lumen_fragmentation": +1,
}

SPECIFICITY = "NPI_nuclear_packing"

EXPECTED_TEST_SLIDES = 45
EXPECTED_TRAIN_SLIDES = 80


# =============================================================================
# OUTPUTS
# =============================================================================

O_PER_SLIDE = "aggc_pc6_per_slide_high_minus_low_effects.csv"
O_METRIC = "aggc_pc6_metric_level_transfer_results.csv"
O_FAMILY = "aggc_pc6_family_level_direction_concordance.csv"
O_NPI = "aggc_pc6_npi_specificity_reference_results.csv"
O_TEST = "aggc_pc6_primary_test_results.csv"
O_TRAIN = "aggc_pc6_secondary_train_results.csv"
O_BOOT = "aggc_pc6_bootstrap_audit.csv"
O_AUDIT = "24c_inference_firewall_audit.csv"
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
                "Use --overwrite only for intentional re-execution under the "
                "unchanged Step24A/24B3 frozen authorities."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sign_of(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x > 0:
        return +1
    if x < 0:
        return -1
    return 0


def direction_label(sign: int) -> str:
    if sign > 0:
        return "POSITIVE"
    if sign < 0:
        return "NEGATIVE"
    return "ZERO_OR_UNRESOLVED"


def ci_excludes_zero(lo: float, hi: float) -> bool:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))


def stable_bootstrap_seed(base_seed: int, *parts: Any) -> int:
    payload = str(base_seed) + "||" + "||".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


# =============================================================================
# CONTRACT VALIDATION
# =============================================================================

def load_contract() -> Dict[str, Any]:
    for path, label in [
        (A_PLAN, "Step24A plan"),
        (A_ENDPOINTS, "Step24A endpoints"),
        (A_SPECIFICITY, "Step24A specificity"),
        (A_PARTITIONS, "Step24A partitions"),
        (A_EVAL, "Step24A evaluation spec"),
        (A_SAMPLING, "Step24A sampling spec"),
        (A_METADATA, "Step24A metadata"),
        (A_CLAIM, "Step24A claim ceiling"),
        (B3_TILE, "Step24B3 final sampled tile measurements"),
        (B3_SLIDE, "Step24B3 slide-stratum summary"),
        (B3_PARTITION, "Step24B3 partition summary"),
        (B3_PROV, "Step24B3 provenance"),
        (B3_DUP, "Step24B3 duplicate audit"),
        (B3_METADATA, "Step24B3 metadata"),
    ]:
        require_file(path, label)

    plan = load_json(A_PLAN)
    a_meta = load_json(A_METADATA)
    b3_meta = load_json(B3_METADATA)

    if str(plan.get("target_axis")) != "Virchow2_PC6":
        raise RuntimeError("Step24A target axis is no longer Virchow2 PC6.")

    if str(plan.get("frozen_pc6_direction")) != "high_in_tumor":
        raise RuntimeError("Step24A PC6 frozen direction changed.")

    endpoints = pd.read_csv(A_ENDPOINTS, low_memory=False).sort_values("endpoint_order")

    observed_metrics = endpoints["metric"].astype(str).tolist()
    if observed_metrics != FORMAL_ENDPOINTS:
        raise RuntimeError(
            "Step24A formal endpoint family/order changed.\n"
            f"Expected={FORMAL_ENDPOINTS}\nObserved={observed_metrics}"
        )

    observed_signs = {
        str(r["metric"]): int(r["expected_sign"])
        for _, r in endpoints.iterrows()
    }

    if observed_signs != EXPECTED_SIGNS:
        raise RuntimeError(
            "Step24A frozen expected directions changed.\n"
            f"Expected={EXPECTED_SIGNS}\nObserved={observed_signs}"
        )

    specificity = pd.read_csv(A_SPECIFICITY, low_memory=False)
    if len(specificity) != 1:
        raise RuntimeError("Step24A specificity table must contain exactly one row.")
    if str(specificity.iloc[0]["metric"]) != SPECIFICITY:
        raise RuntimeError("Step24A specificity reference changed.")
    if bool(specificity.iloc[0].get("may_be_promoted_from_aggc", False)):
        raise RuntimeError("Step24A unexpectedly permits NPI promotion.")

    partitions = pd.read_csv(A_PARTITIONS, low_memory=False)

    ptest = partitions.loc[
        partitions["partition"].astype(str).eq(PRIMARY_PARTITION)
    ]
    ptrain = partitions.loc[
        partitions["partition"].astype(str).eq(SECONDARY_PARTITION)
    ]

    if len(ptest) != 1 or not bool(ptest.iloc[0]["primary_inference"]):
        raise RuntimeError("AGGC Test is not frozen as primary inference.")
    if len(ptrain) != 1 or bool(ptrain.iloc[0]["primary_inference"]):
        raise RuntimeError("AGGC Train role is not frozen as secondary.")

    sampling = pd.read_csv(A_SAMPLING, low_memory=False)
    smap = {
        str(r["parameter"]): float(r["value"])
        for _, r in sampling.iterrows()
    }

    bootstrap_resamples = int(smap["bootstrap_resamples"])
    bootstrap_seed = int(smap["bootstrap_seed"])

    if bootstrap_resamples != 10000:
        raise RuntimeError(
            f"Expected frozen 10,000 bootstrap resamples; observed={bootstrap_resamples}"
        )

    # Validate Step24B3 final measurement freeze.
    counts = b3_meta.get("counts", {})
    if int(counts.get("test_slides", -1)) != EXPECTED_TEST_SLIDES:
        raise RuntimeError("Step24B3 Test slide count is not 45.")
    if int(counts.get("train_slides", -1)) != EXPECTED_TRAIN_SLIDES:
        raise RuntimeError("Step24B3 Train slide count is not 80.")

    firewall = b3_meta.get("firewall", {})
    forbidden_true = [
        "high_minus_low_effect_calculated",
        "direction_concordance_calculated",
        "bootstrap_calculated",
        "p_values_calculated",
        "fdr_calculated",
        "tumor_auc_calculated",
        "train_test_inferential_comparison",
        "train_test_pooled",
        "pc_ranking",
        "pc_screening",
        "pca_refit",
        "pca_reprojection",
        "pc6_sign_flip",
        "new_metric_selection",
        "npi_promoted",
        "gleason_state_analysis",
        "outcome_based_slide_exclusion",
    ]

    violated = [x for x in forbidden_true if bool(firewall.get(x, False))]
    if violated:
        raise RuntimeError(
            "Step24B3 provenance shows inference/design actions before Step24C:\n"
            + "\n".join(violated)
        )

    dup = pd.read_csv(B3_DUP, low_memory=False)
    if len(dup) != 1 or not bool(dup.iloc[0]["pass"]):
        raise RuntimeError("Step24B3 duplicate audit did not pass.")

    return {
        "expected_signs": observed_signs,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "a_meta": a_meta,
        "b3_meta": b3_meta,
    }


# =============================================================================
# PER-SLIDE EFFECTS
# =============================================================================

def build_per_slide_effects(slide_summary: pd.DataFrame) -> pd.DataFrame:
    required = [
        "partition",
        "slide_id",
        "sampling_stratum",
        *[f"{m}__median" for m in FORMAL_ENDPOINTS],
        f"{SPECIFICITY}__median",
    ]

    missing = [c for c in required if c not in slide_summary.columns]
    if missing:
        raise RuntimeError(
            f"Step24B3 slide-stratum summary missing columns: {missing}"
        )

    rows: List[Dict[str, Any]] = []

    for (partition, slide_id), sub in slide_summary.groupby(
        ["partition", "slide_id"],
        sort=True,
    ):
        strata = set(sub["sampling_stratum"].astype(str))
        if strata != {"PC6_LOW", "PC6_HIGH"}:
            raise RuntimeError(
                f"{partition} {slide_id}: expected exactly PC6_LOW and PC6_HIGH; "
                f"observed={sorted(strata)}"
            )

        low = sub.loc[sub["sampling_stratum"].astype(str).eq("PC6_LOW")].iloc[0]
        high = sub.loc[sub["sampling_stratum"].astype(str).eq("PC6_HIGH")].iloc[0]

        for metric in [*FORMAL_ENDPOINTS, SPECIFICITY]:
            low_val = float(low[f"{metric}__median"])
            high_val = float(high[f"{metric}__median"])

            if not (np.isfinite(low_val) and np.isfinite(high_val)):
                raise RuntimeError(
                    f"{partition} {slide_id} {metric}: non-finite frozen stratum median."
                )

            effect = high_val - low_val

            rows.append({
                "partition": partition,
                "formal_partition_role": (
                    "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"
                    if partition == PRIMARY_PARTITION
                    else "SECONDARY_EXTERNAL_CHARACTERIZATION"
                ),
                "slide_id": slide_id,
                "metric": metric,
                "endpoint_role": (
                    "FORMAL_TRANSFER_ENDPOINT"
                    if metric in FORMAL_ENDPOINTS
                    else "SPECIFICITY_REFERENCE_ONLY"
                ),
                "pc6_low_median": low_val,
                "pc6_high_median": high_val,
                "high_minus_low_effect": effect,
                "expected_sign": (
                    EXPECTED_SIGNS[metric]
                    if metric in FORMAL_ENDPOINTS
                    else np.nan
                ),
                "direction_match": (
                    sign_of(effect) == EXPECTED_SIGNS[metric]
                    if metric in FORMAL_ENDPOINTS
                    else np.nan
                ),
            })

    out = pd.DataFrame(rows)

    # Verify complete slide x metric grid.
    for partition, expected_n in [
        (PRIMARY_PARTITION, EXPECTED_TEST_SLIDES),
        (SECONDARY_PARTITION, EXPECTED_TRAIN_SLIDES),
    ]:
        sub = out.loc[out["partition"].astype(str).eq(partition)]
        for metric in [*FORMAL_ENDPOINTS, SPECIFICITY]:
            n = sub.loc[sub["metric"].astype(str).eq(metric), "slide_id"].nunique()
            if n != expected_n:
                raise RuntimeError(
                    f"{partition} {metric}: expected {expected_n} slide effects; observed={n}"
                )

    return out.sort_values(
        ["partition", "metric", "slide_id"],
        kind="mergesort",
    ).reset_index(drop=True)


# =============================================================================
# BOOTSTRAP
# =============================================================================

def bootstrap_median(
    effects: np.ndarray,
    n_resamples: int,
    seed: int,
) -> Tuple[float, float, float, np.ndarray]:
    x = np.asarray(effects, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise RuntimeError("Cannot bootstrap an empty effect vector.")

    observed = float(np.median(x))

    rng = np.random.default_rng(seed)

    boot = np.empty(n_resamples, dtype=np.float64)
    n = len(x)

    # 10k x <=80 is small; loop is transparent and reproducible.
    for i in range(n_resamples):
        sample = x[rng.integers(0, n, size=n)]
        boot[i] = np.median(sample)

    lo, hi = np.quantile(boot, [0.025, 0.975])

    return observed, float(lo), float(hi), boot


def build_metric_results(
    per_slide: pd.DataFrame,
    contract: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    boot_audit_rows = []

    for partition, partition_role in [
        (PRIMARY_PARTITION, "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"),
        (SECONDARY_PARTITION, "SECONDARY_EXTERNAL_CHARACTERIZATION"),
    ]:
        for metric in FORMAL_ENDPOINTS:
            sub = per_slide.loc[
                per_slide["partition"].astype(str).eq(partition)
                & per_slide["metric"].astype(str).eq(metric)
            ].copy()

            effects = sub["high_minus_low_effect"].to_numpy(float)

            seed = stable_bootstrap_seed(
                contract["bootstrap_seed"],
                partition,
                metric,
                "median_high_minus_low",
            )

            estimate, lo, hi, boot = bootstrap_median(
                effects,
                contract["bootstrap_resamples"],
                seed,
            )

            expected_sign = contract["expected_signs"][metric]
            observed_sign = sign_of(estimate)
            match = observed_sign == expected_sign

            result_rows.append({
                "partition": partition,
                "formal_partition_role": partition_role,
                "metric": metric,
                "endpoint_role": "FORMAL_TRANSFER_ENDPOINT",
                "analysis_unit": "slide",
                "n_slides": int(len(sub)),
                "estimate_type": "median_slide_level_PC6_HIGH_minus_PC6_LOW",
                "estimate": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "ci_level": 0.95,
                "expected_sign": expected_sign,
                "expected_direction": direction_label(expected_sign),
                "observed_sign": observed_sign,
                "observed_direction": direction_label(observed_sign),
                "direction_match": bool(match),
                "ci_excludes_zero": ci_excludes_zero(lo, hi),
                "formal_replication_call": (
                    "DIRECTION_REPLICATED"
                    if match
                    else "DIRECTION_NOT_REPLICATED"
                ),
                "bootstrap_resamples": contract["bootstrap_resamples"],
                "bootstrap_seed": seed,
                "p_value_calculated": False,
                "fdr_calculated": False,
            })

            boot_audit_rows.append({
                "partition": partition,
                "metric": metric,
                "n_slides": int(len(sub)),
                "bootstrap_resamples": contract["bootstrap_resamples"],
                "bootstrap_seed": seed,
                "bootstrap_median_mean": float(np.mean(boot)),
                "bootstrap_median_sd": float(np.std(boot, ddof=1)),
                "bootstrap_min": float(np.min(boot)),
                "bootstrap_max": float(np.max(boot)),
            })

    return pd.DataFrame(result_rows), pd.DataFrame(boot_audit_rows)


def build_specificity_results(
    per_slide: pd.DataFrame,
    contract: Dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for partition, role in [
        (PRIMARY_PARTITION, "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"),
        (SECONDARY_PARTITION, "SECONDARY_EXTERNAL_CHARACTERIZATION"),
    ]:
        sub = per_slide.loc[
            per_slide["partition"].astype(str).eq(partition)
            & per_slide["metric"].astype(str).eq(SPECIFICITY)
        ].copy()

        effects = sub["high_minus_low_effect"].to_numpy(float)

        seed = stable_bootstrap_seed(
            contract["bootstrap_seed"],
            partition,
            SPECIFICITY,
            "specificity_reference",
        )

        estimate, lo, hi, _ = bootstrap_median(
            effects,
            contract["bootstrap_resamples"],
            seed,
        )

        rows.append({
            "partition": partition,
            "formal_partition_role": role,
            "metric": SPECIFICITY,
            "endpoint_role": "SPECIFICITY_REFERENCE_ONLY",
            "analysis_unit": "slide",
            "n_slides": int(len(sub)),
            "estimate_type": "median_slide_level_PC6_HIGH_minus_PC6_LOW",
            "estimate": estimate,
            "ci_low": lo,
            "ci_high": hi,
            "ci_level": 0.95,
            "expected_sign": np.nan,
            "direction_match": np.nan,
            "formal_replication_endpoint": False,
            "may_be_promoted_to_pc6_phenotype": False,
            "bootstrap_resamples": contract["bootstrap_resamples"],
            "bootstrap_seed": seed,
        })

    return pd.DataFrame(rows)


# =============================================================================
# FAMILY-LEVEL CONCORDANCE
# =============================================================================

def build_family_results(metric_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for partition, role in [
        (PRIMARY_PARTITION, "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"),
        (SECONDARY_PARTITION, "SECONDARY_EXTERNAL_CHARACTERIZATION"),
    ]:
        sub = metric_results.loc[
            metric_results["partition"].astype(str).eq(partition)
        ].copy()

        if len(sub) != len(FORMAL_ENDPOINTS):
            raise RuntimeError(
                f"{partition}: expected {len(FORMAL_ENDPOINTS)} formal metric results; "
                f"observed={len(sub)}"
            )

        n_match = int(sub["direction_match"].astype(bool).sum())
        fraction = n_match / len(FORMAL_ENDPOINTS)

        n_ci_zero_excluded = int(sub["ci_excludes_zero"].astype(bool).sum())
        n_match_and_ci = int(
            (
                sub["direction_match"].astype(bool)
                & sub["ci_excludes_zero"].astype(bool)
            ).sum()
        )

        rows.append({
            "partition": partition,
            "formal_partition_role": role,
            "family_endpoint": "overall_direction_concordance_fraction",
            "n_formal_endpoints": len(FORMAL_ENDPOINTS),
            "n_direction_replicated": n_match,
            "direction_concordance_fraction": fraction,
            "n_ci_excluding_zero": n_ci_zero_excluded,
            "n_direction_replicated_and_ci_excluding_zero": n_match_and_ci,
            "primary_inference": partition == PRIMARY_PARTITION,
            "train_may_replace_test": False,
        })

    return pd.DataFrame(rows)


# =============================================================================
# HASHES / README
# =============================================================================

def build_hashes() -> pd.DataFrame:
    rows = []

    for role, path in [
        ("step24a_plan", A_PLAN),
        ("step24a_endpoints", A_ENDPOINTS),
        ("step24a_specificity", A_SPECIFICITY),
        ("step24a_partitions", A_PARTITIONS),
        ("step24a_evaluation", A_EVAL),
        ("step24a_sampling", A_SAMPLING),
        ("step24a_metadata", A_METADATA),
        ("step24a_claim_ceiling", A_CLAIM),
        ("step24b3_final_sampled_tiles", B3_TILE),
        ("step24b3_slide_stratum_summary", B3_SLIDE),
        ("step24b3_partition_summary", B3_PARTITION),
        ("step24b3_provenance", B3_PROV),
        ("step24b3_duplicate_audit", B3_DUP),
        ("step24b3_metadata", B3_METADATA),
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
            "Evaluate the Step24A-frozen AGGC PC6 morphology-transfer endpoints "
            "using only the Step24B3 final frozen measurement authority."
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

    banner("STEP 24C: EVALUATE FROZEN AGGC PC6 MORPHOLOGY TRANSFER")

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP24A]       {STEP24A}")
    print(f"[STEP24B3]      {STEP24B3}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          FIRST FORMAL EXTERNAL MORPHOLOGY-TRANSFER INFERENCE")
    print("[PRIMARY]       Subset1_Test")
    print("[SECONDARY]     Subset1_Train")
    print("[ANALYSIS UNIT] SLIDE")
    print("[EFFECT]        PC6_HIGH - PC6_LOW")
    print("[ESTIMATE]      MEDIAN SLIDE-LEVEL EFFECT")
    print("[BOOTSTRAP]     10,000 SLIDE RESAMPLES")
    print("[P-VALUES]      NONE")
    print("[FDR]           NONE")
    print("[TUMOR AUC]     NONE")
    print("[TRAIN/TEST]    NEVER POOLED")
    print("[PC SCREEN]     NONE")
    print("[PCA REFIT]     NONE")
    print("[SIGN TUNING]   NONE")
    print("[NPI]           SPECIFICITY REFERENCE ONLY")

    prepare_output(args.output, args.overwrite)

    # -------------------------------------------------------------------------
    # Contract.
    # -------------------------------------------------------------------------

    banner("VALIDATE STEP24A / STEP24B3 FROZEN AUTHORITIES")

    contract = load_contract()

    print(f"Formal transfer endpoints             : {len(FORMAL_ENDPOINTS)}")
    print(f"Bootstrap resamples                   : {contract['bootstrap_resamples']:,}")
    print(f"Bootstrap master seed                 : {contract['bootstrap_seed']}")
    print("AGGC Test primary role                : PASS")
    print("AGGC Train secondary role             : PASS")
    print("Step24B3 pre-inference firewall       : PASS")

    print()
    frozen_spec = pd.DataFrame([
        {
            "metric": metric,
            "expected_sign": contract["expected_signs"][metric],
            "expected_direction": direction_label(contract["expected_signs"][metric]),
        }
        for metric in FORMAL_ENDPOINTS
    ])
    print(frozen_spec.to_string(index=False))

    # -------------------------------------------------------------------------
    # Per-slide effects.
    # -------------------------------------------------------------------------

    banner("CALCULATE FROZEN SLIDE-LEVEL PC6 HIGH-MINUS-LOW EFFECTS")

    slide_summary = pd.read_csv(B3_SLIDE, low_memory=False)

    per_slide = build_per_slide_effects(slide_summary)
    per_slide.to_csv(args.output / O_PER_SLIDE, index=False)

    print(
        per_slide.groupby(["partition", "metric"])
        .size()
        .rename("n_slides")
        .reset_index()
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Formal metric inference.
    # -------------------------------------------------------------------------

    banner("FORMAL METRIC-LEVEL EXTERNAL TRANSFER RESULTS")

    metric_results, bootstrap_audit = build_metric_results(
        per_slide,
        contract,
    )

    metric_results.to_csv(args.output / O_METRIC, index=False)
    bootstrap_audit.to_csv(args.output / O_BOOT, index=False)

    test_results = metric_results.loc[
        metric_results["partition"].astype(str).eq(PRIMARY_PARTITION)
    ].copy()

    train_results = metric_results.loc[
        metric_results["partition"].astype(str).eq(SECONDARY_PARTITION)
    ].copy()

    test_results.to_csv(args.output / O_TEST, index=False)
    train_results.to_csv(args.output / O_TRAIN, index=False)

    print("\nPRIMARY HELD-OUT TEST")
    print(
        test_results[
            [
                "metric",
                "n_slides",
                "estimate",
                "ci_low",
                "ci_high",
                "expected_direction",
                "observed_direction",
                "direction_match",
                "ci_excludes_zero",
            ]
        ].to_string(index=False)
    )

    print("\nSECONDARY TRAIN CHARACTERIZATION")
    print(
        train_results[
            [
                "metric",
                "n_slides",
                "estimate",
                "ci_low",
                "ci_high",
                "expected_direction",
                "observed_direction",
                "direction_match",
                "ci_excludes_zero",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Family concordance.
    # -------------------------------------------------------------------------

    banner("FAMILY-LEVEL DIRECTION CONCORDANCE")

    family_results = build_family_results(metric_results)
    family_results.to_csv(args.output / O_FAMILY, index=False)

    print(family_results.to_string(index=False))

    # -------------------------------------------------------------------------
    # NPI specificity.
    # -------------------------------------------------------------------------

    banner("NPI SPECIFICITY REFERENCE — SUPPORTIVE ONLY")

    npi = build_specificity_results(per_slide, contract)
    npi.to_csv(args.output / O_NPI, index=False)

    print(
        npi[
            [
                "partition",
                "n_slides",
                "estimate",
                "ci_low",
                "ci_high",
                "formal_replication_endpoint",
                "may_be_promoted_to_pc6_phenotype",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Firewall / provenance.
    # -------------------------------------------------------------------------

    audit = pd.DataFrame([
        {"audit_item": "Step24A frozen endpoint family inherited", "status": "YES"},
        {"audit_item": "Step24B3 final measurement authority inherited", "status": "YES"},
        {"audit_item": "Test primary role retained", "status": "YES"},
        {"audit_item": "Train secondary role retained", "status": "YES"},
        {"audit_item": "HIGH-minus-LOW effect calculated for first time", "status": "YES"},
        {"audit_item": "Slide-level analysis unit retained", "status": "YES"},
        {"audit_item": "10,000 slide bootstrap executed", "status": "YES"},
        {"audit_item": "P-values calculated", "status": "NO"},
        {"audit_item": "FDR calculated", "status": "NO"},
        {"audit_item": "Tumor AUC calculated", "status": "NO"},
        {"audit_item": "Train/Test pooled", "status": "NO"},
        {"audit_item": "Train result allowed to replace Test", "status": "NO"},
        {"audit_item": "PC screening", "status": "NO"},
        {"audit_item": "PCA refit", "status": "NO"},
        {"audit_item": "PCA reprojection", "status": "NO"},
        {"audit_item": "PC6 sign tuning", "status": "NO"},
        {"audit_item": "Endpoint reselection", "status": "NO"},
        {"audit_item": "Expected-direction tuning", "status": "NO"},
        {"audit_item": "NPI promoted", "status": "NO"},
        {"audit_item": "Gleason-state analysis", "status": "NO"},
        {"audit_item": "Morphology recomputed", "status": "NO"},
        {"audit_item": "Sampling rerun", "status": "NO"},
        {"audit_item": "Outcome-based slide exclusion", "status": "NO"},
    ])

    audit.to_csv(args.output / O_AUDIT, index=False)

    hashes = build_hashes()
    hashes.to_csv(args.output / O_HASHES, index=False)

    primary_family = family_results.loc[
        family_results["partition"].astype(str).eq(PRIMARY_PARTITION)
    ].iloc[0]

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "role": "FIRST_FORMAL_AGGC_EXTERNAL_PC6_MORPHOLOGY_TRANSFER_INFERENCE",
        "upstream": {
            "step24a": str(STEP24A),
            "step24b3": str(STEP24B3),
        },
        "primary_partition": PRIMARY_PARTITION,
        "secondary_partition": SECONDARY_PARTITION,
        "formal_endpoints": FORMAL_ENDPOINTS,
        "specificity_reference": SPECIFICITY,
        "effect_definition": "slide-level morphology median(PC6_HIGH) - median(PC6_LOW)",
        "primary_estimator": "median of slide-level effects",
        "bootstrap": {
            "resamples": int(contract["bootstrap_resamples"]),
            "master_seed": int(contract["bootstrap_seed"]),
            "unit": "slide",
            "ci": "percentile_95",
        },
        "primary_test_family_result": {
            "n_formal_endpoints": int(primary_family["n_formal_endpoints"]),
            "n_direction_replicated": int(primary_family["n_direction_replicated"]),
            "direction_concordance_fraction": float(
                primary_family["direction_concordance_fraction"]
            ),
            "n_ci_excluding_zero": int(primary_family["n_ci_excluding_zero"]),
            "n_direction_replicated_and_ci_excluding_zero": int(
                primary_family["n_direction_replicated_and_ci_excluding_zero"]
            ),
        },
        "inference_policy": {
            "directional_replication_primary": True,
            "ci_supportive": True,
            "p_values": False,
            "fdr": False,
            "train_can_rescue_test": False,
            "npi_formal_endpoint": False,
        },
        "firewall": {
            "tumor_auc_calculated": False,
            "train_test_pooled": False,
            "pc_screening": False,
            "pca_refit": False,
            "pca_reprojection": False,
            "pc6_sign_tuning": False,
            "endpoint_reselection": False,
            "expected_direction_tuning": False,
            "npi_promoted": False,
            "gleason_state_analysis": False,
            "morphology_recomputed": False,
            "sampling_rerun": False,
            "outcome_based_slide_exclusion": False,
        },
        "next_step": (
            "Freeze final AGGC PC6 external morphology evidence / claim boundary "
            "or proceed to separately frozen PC6 tumor-localization validation."
        ),
    }

    save_json(args.output / O_METADATA, metadata)

    readme = f"""STEP 24C
FROZEN AGGC PC6 MORPHOLOGY TRANSFER EVALUATION
==============================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

Role
----
FIRST FORMAL EXTERNAL PC6 MORPHOLOGY-TRANSFER INFERENCE.

Primary partition
-----------------
{PRIMARY_PARTITION}
Held-out external replication.

Secondary partition
-------------------
{SECONDARY_PARTITION}
Characterization only.

Formal endpoints
----------------
{chr(10).join("  - " + m + " : " + direction_label(EXPECTED_SIGNS[m]) for m in FORMAL_ENDPOINTS)}

Effect
------
For each slide and endpoint:

  PC6_HIGH median - PC6_LOW median

Primary metric-level estimate
-----------------------------
Median of slide-level effects.

Uncertainty
-----------
{contract['bootstrap_resamples']:,} slide-level nonparametric bootstrap resamples.

Family-level primary endpoint
-----------------------------
Direction concordance across the six pre-frozen morphology endpoints.

NPI
---
{SPECIFICITY}
Specificity reference only.
It is not part of the 6-endpoint formal replication family and cannot be
promoted from AGGC.

No new p-value/FDR family
-------------------------
Step24A froze directional replication plus bootstrap uncertainty.
Step24C does not introduce a new post-freeze multiplicity-testing family.

Outputs
-------
Per-slide effects:
  {O_PER_SLIDE}

Metric-level formal results:
  {O_METRIC}

Primary held-out Test:
  {O_TEST}

Secondary Train:
  {O_TRAIN}

Family-level concordance:
  {O_FAMILY}

NPI specificity:
  {O_NPI}

Inference firewall
------------------
Tumor AUC                      : NO
Train/Test pooling             : NO
Train replacing Test           : NO
PC screening                   : NO
PCA refit                      : NO
PCA reprojection               : NO
PC6 sign tuning                : NO
endpoint reselection           : NO
expected-direction tuning      : NO
NPI promotion                  : NO
Gleason-state analysis         : NO
morphology recomputation       : NO
sampling rerun                 : NO
outcome-based slide exclusion  : NO
"""

    (args.output / O_README).write_text(readme, encoding="utf-8")

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    banner("STEP24C FINAL INFERENCE AUDIT")

    for _, row in audit.iterrows():
        print(f"{row['audit_item']:<66}: {row['status']}")

    print()
    print("[PRIMARY TEST FAMILY RESULT]")
    print(
        f"Direction replicated             : "
        f"{int(primary_family['n_direction_replicated'])}/"
        f"{int(primary_family['n_formal_endpoints'])}"
    )
    print(
        f"Direction concordance fraction   : "
        f"{float(primary_family['direction_concordance_fraction']):.6f}"
    )
    print(
        f"CI excluding zero                : "
        f"{int(primary_family['n_ci_excluding_zero'])}/"
        f"{int(primary_family['n_formal_endpoints'])}"
    )
    print(
        f"Direction match + CI exclude 0   : "
        f"{int(primary_family['n_direction_replicated_and_ci_excluding_zero'])}/"
        f"{int(primary_family['n_formal_endpoints'])}"
    )

    print()
    print("[OK] Saved:")
    for name in [
        O_PER_SLIDE,
        O_METRIC,
        O_FAMILY,
        O_NPI,
        O_TEST,
        O_TRAIN,
        O_BOOT,
        O_AUDIT,
        O_HASHES,
        O_METADATA,
        O_README,
    ]:
        path = args.output / name
        if not path.exists():
            raise RuntimeError(f"Expected Step24C output missing: {path}")
        print(f"  {path}")

    print()
    print("[DONE] Step24C frozen AGGC PC6 morphology-transfer evaluation complete.")
    print("[STATUS] AGGC PC6 EXTERNAL MORPHOLOGY TRANSFER = EVALUATED.")
    print(
        "[NEXT] Interpret the held-out Test results under the Step24A claim ceiling; "
        "Train remains secondary regardless of outcome."
    )


if __name__ == "__main__":
    main()
