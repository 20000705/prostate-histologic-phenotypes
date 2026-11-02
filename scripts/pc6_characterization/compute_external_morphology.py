#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 24B: MEASURE AGGC EXTERNAL PC6 MORPHOLOGY TRANSFER

VERSION: 1.0.0
SCHEMA : 24b-aggc-external-pc6-morphology-transfer-measurement-schema-1.0

ROLE
----
MEASUREMENT / QC ONLY.

This step executes ONLY the measurement operations frozen in Step24A.

It is allowed to read AGGC PC6 and morphology rows for the first time, but it
MUST NOT perform external-transfer inference.

ALLOWED
-------
- validate Step24A frozen specification;
- read AGGC unified tile morphology table;
- classify slide partition as Subset1_Test or Subset1_Train;
- require finite raw PC6 + six frozen transfer metrics;
- require >= 50 eligible tiles per slide;
- calculate exact within-slide raw-PC6 q20 / q80 thresholds;
- assign PC6_LOW / PC6_HIGH;
- deterministic sampling up to 10 tiles / slide / stratum;
- retain NPI only as the frozen specificity reference;
- save sampled tile measurements;
- summarize tile measurements by slide x stratum using medians;
- report counts / QC / provenance.

PROHIBITED
----------
- PC6_HIGH - PC6_LOW effect calculation;
- direction-concordance calculation;
- bootstrap confidence intervals;
- p-values / FDR;
- Train-vs-Test inferential comparison;
- pooling Train and Test;
- tumor AUC;
- PC1-PC8 screening;
- PCA refit;
- PC6 sign tuning;
- morphology metric selection;
- NPI promotion;
- Gleason-state analysis.

IMPORTANT
---------
Step24B is deliberately two-pass and chunked:

PASS 1
    exact slide eligibility + exact within-slide PC6 q20/q80 thresholds.

PASS 2
    deterministic reservoir sampling from the frozen PC6 tails and measurement
    table assembly.

No association/effect statistic is calculated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from dataclasses import dataclass
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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / PATHS
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "24b-aggc-external-pc6-morphology-transfer-measurement-schema-1.0"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT

STEP24A = RESULTS / "24a_aggc_external_pc6_morphology_transfer_specification"

A_PLAN = STEP24A / "aggc_pc6_external_morphology_transfer_plan.json"
A_ENDPOINTS = STEP24A / "aggc_pc6_frozen_transfer_endpoint_specification.csv"
A_SPECIFICITY = STEP24A / "aggc_pc6_specificity_reference_specification.csv"
A_PARTITIONS = STEP24A / "aggc_pc6_partition_role_specification.csv"
A_SAMPLING = STEP24A / "aggc_pc6_sampling_and_eligibility_specification.csv"
A_EVALUATION = STEP24A / "aggc_pc6_external_evaluation_endpoint_specification.csv"
A_SOURCE = STEP24A / "aggc_pc6_measurement_source_snapshot.csv"
A_METADATA = STEP24A / "analysis_metadata.json"
A_CLAIM = STEP24A / "pc6_aggc_external_claim_ceiling.txt"

DEFAULT_OUT = RESULTS / "24b_aggc_external_pc6_morphology_transfer_measurements"

EXPECTED_24A_VERSION = "1.0.0"
EXPECTED_24A_SCHEMA = "24a-aggc-external-pc6-morphology-transfer-freeze-schema-1.0"

EXPECTED_PRIMARY_PARTITION = "Subset1_Test"
EXPECTED_SECONDARY_PARTITION = "Subset1_Train"
EXPECTED_PC6_DIRECTION = "high_in_tumor"

EXPECTED_ENDPOINTS = [
    ("SEI_eosin_stroma", +1),
    ("GOI_lumen_openness", -1),
    ("CLR_nuclear_to_lumen", +1),
    ("lumen_area", -1),
    ("mean_lumen_area", -1),
    ("lumen_fragmentation", +1),
]

EXPECTED_SPECIFICITY = "NPI_nuclear_packing"

DEFAULT_CHUNKSIZE = 250_000


# =============================================================================
# OUTPUT FILENAMES
# =============================================================================

O_INPUT_AUDIT = "24b_input_integrity_audit.csv"
O_THRESHOLDS = "aggc_pc6_slide_eligibility_and_thresholds.csv"
O_SAMPLE = "aggc_pc6_frozen_sample_manifest.csv"
O_TILE = "aggc_pc6_sampled_tile_measurements.csv.gz"
O_SLIDE = "aggc_pc6_slide_stratum_morphology_summary.csv"
O_PARTITION = "aggc_pc6_partition_measurement_summary.csv"
O_SAMPLING_AUDIT = "aggc_pc6_sampling_audit.csv"
O_DUP_AUDIT = "aggc_pc6_sampled_tile_duplicate_audit.csv"
O_SOURCE_SNAPSHOT = "aggc_pc6_measurement_source_execution_snapshot.csv"
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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty:\n{path}\n"
                "Use --overwrite only for intentional re-execution under the "
                "unchanged Step24A freeze."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_partition(slide_id: str) -> Optional[str]:
    s = str(slide_id)

    if s.startswith("Subset1_Test_"):
        return EXPECTED_PRIMARY_PARTITION

    if s.startswith("Subset1_Train_"):
        return EXPECTED_SECONDARY_PARTITION

    return None


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = str(base_seed) + "||" + "||".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def finite_mask(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)

    for c in columns:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)

    return mask


def median_or_nan(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return np.nan

    return float(np.median(v))


def resolve_identity_columns(header_columns: Sequence[str]) -> List[str]:
    cols = set(header_columns)

    if "tile_index" in cols:
        return ["slide_id", "tile_index"]

    if "x" in cols and "y" in cols:
        return ["slide_id", "x", "y"]

    return ["slide_id", "source_row_index"]


# =============================================================================
# FROZEN CONTRACT
# =============================================================================

@dataclass(frozen=True)
class Contract:
    source_path: Path
    formal_metrics: Tuple[str, ...]
    expected_signs: Dict[str, int]
    specificity_metric: str
    primary_partition: str
    secondary_partition: str
    low_q: float
    high_q: float
    min_tiles: int
    max_tiles_per_slide_stratum: int
    sampling_seed: int
    source_size_bytes: int
    source_mtime_ns: int
    source_sha256: str
    source_sha256_skipped: bool


def load_contract() -> Contract:
    required = [
        (A_PLAN, "Step24A plan"),
        (A_ENDPOINTS, "Step24A endpoint specification"),
        (A_SPECIFICITY, "Step24A specificity specification"),
        (A_PARTITIONS, "Step24A partition specification"),
        (A_SAMPLING, "Step24A sampling specification"),
        (A_EVALUATION, "Step24A evaluation specification"),
        (A_SOURCE, "Step24A measurement source snapshot"),
        (A_METADATA, "Step24A metadata"),
        (A_CLAIM, "Step24A claim ceiling"),
    ]

    for path, label in required:
        require_file(path, label)

    plan = load_json(A_PLAN)
    metadata = load_json(A_METADATA)

    if str(plan.get("version")) != EXPECTED_24A_VERSION:
        raise RuntimeError(
            f"Step24B requires Step24A v{EXPECTED_24A_VERSION}; "
            f"observed={plan.get('version')}"
        )

    if str(plan.get("schema")) != EXPECTED_24A_SCHEMA:
        raise RuntimeError(
            f"Unexpected Step24A schema: {plan.get('schema')}"
        )

    if str(plan.get("frozen_pc6_direction")) != EXPECTED_PC6_DIRECTION:
        raise RuntimeError(
            "Step24A PC6 direction changed; Step24B refuses to proceed."
        )

    endpoints = pd.read_csv(A_ENDPOINTS, low_memory=False)

    observed = []
    for _, r in endpoints.sort_values("endpoint_order").iterrows():
        observed.append((str(r["metric"]), int(r["expected_sign"])))

    if observed != EXPECTED_ENDPOINTS:
        raise RuntimeError(
            "Step24A formal endpoint family/order differs from the expected "
            "six frozen transfer endpoints.\n"
            f"Expected={EXPECTED_ENDPOINTS}\nObserved={observed}"
        )

    specificity = pd.read_csv(A_SPECIFICITY, low_memory=False)

    if len(specificity) != 1:
        raise RuntimeError("Step24A specificity specification must contain one row.")

    if str(specificity.iloc[0]["metric"]) != EXPECTED_SPECIFICITY:
        raise RuntimeError(
            "Step24A specificity metric changed unexpectedly."
        )

    if normalize_bool(specificity.iloc[0]["may_be_promoted_from_aggc"]):
        raise RuntimeError("NPI promotion is not allowed by the Step24A freeze.")

    partitions = pd.read_csv(A_PARTITIONS, low_memory=False)

    primary = partitions.loc[
        partitions["formal_role"].astype(str).eq("PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"),
        "partition",
    ].astype(str).tolist()

    secondary = partitions.loc[
        partitions["formal_role"].astype(str).eq("SECONDARY_EXTERNAL_CHARACTERIZATION"),
        "partition",
    ].astype(str).tolist()

    if primary != [EXPECTED_PRIMARY_PARTITION]:
        raise RuntimeError(f"Unexpected primary partition: {primary}")

    if secondary != [EXPECTED_SECONDARY_PARTITION]:
        raise RuntimeError(f"Unexpected secondary partition: {secondary}")

    sampling = pd.read_csv(A_SAMPLING, low_memory=False)
    smap = {
        str(r["parameter"]): float(r["value"])
        for _, r in sampling.iterrows()
    }

    required_sampling = [
        "low_quantile",
        "high_quantile",
        "min_eligible_tiles_per_slide",
        "max_tiles_per_slide_per_stratum",
        "sampling_seed",
    ]

    missing = [x for x in required_sampling if x not in smap]

    if missing:
        raise RuntimeError(
            f"Step24A sampling specification missing: {missing}"
        )

    source = pd.read_csv(A_SOURCE, low_memory=False)

    if len(source) != 1:
        raise RuntimeError("Step24A measurement source snapshot must contain one row.")

    src = source.iloc[0]

    source_path = Path(str(src["source_path"]))

    require_file(source_path, "AGGC frozen measurement source")

    return Contract(
        source_path=source_path,
        formal_metrics=tuple(x[0] for x in EXPECTED_ENDPOINTS),
        expected_signs={x[0]: x[1] for x in EXPECTED_ENDPOINTS},
        specificity_metric=EXPECTED_SPECIFICITY,
        primary_partition=EXPECTED_PRIMARY_PARTITION,
        secondary_partition=EXPECTED_SECONDARY_PARTITION,
        low_q=float(smap["low_quantile"]),
        high_q=float(smap["high_quantile"]),
        min_tiles=int(smap["min_eligible_tiles_per_slide"]),
        max_tiles_per_slide_stratum=int(
            smap["max_tiles_per_slide_per_stratum"]
        ),
        sampling_seed=int(smap["sampling_seed"]),
        source_size_bytes=int(src["file_size_bytes"]),
        source_mtime_ns=int(src["file_mtime_ns"]),
        source_sha256=str(src.get("sha256", "") or ""),
        source_sha256_skipped=normalize_bool(src.get("sha256_skipped", False)),
    )


# =============================================================================
# SOURCE PROVENANCE
# =============================================================================

def validate_source_provenance(
    contract: Contract,
    verify_hash: bool,
) -> pd.DataFrame:
    p = contract.source_path
    stat = p.stat()

    rows = [
        {
            "audit_item": "source_path_matches_step24a",
            "observed": str(p),
            "expected": str(contract.source_path),
            "pass": True,
        },
        {
            "audit_item": "source_size_bytes",
            "observed": int(stat.st_size),
            "expected": int(contract.source_size_bytes),
            "pass": int(stat.st_size) == int(contract.source_size_bytes),
        },
        {
            "audit_item": "source_mtime_ns",
            "observed": int(stat.st_mtime_ns),
            "expected": int(contract.source_mtime_ns),
            "pass": int(stat.st_mtime_ns) == int(contract.source_mtime_ns),
        },
    ]

    if int(stat.st_size) != int(contract.source_size_bytes):
        raise RuntimeError(
            "AGGC measurement source size changed after Step24A freeze."
        )

    if int(stat.st_mtime_ns) != int(contract.source_mtime_ns):
        raise RuntimeError(
            "AGGC measurement source mtime changed after Step24A freeze."
        )

    if verify_hash:
        if contract.source_sha256_skipped or not contract.source_sha256:
            raise RuntimeError(
                "--verify-source-hash requested, but Step24A did not freeze "
                "a SHA256 for the large source."
            )

        observed_hash = sha256_file(p)
        ok = observed_hash == contract.source_sha256

        rows.append(
            {
                "audit_item": "source_sha256",
                "observed": observed_hash,
                "expected": contract.source_sha256,
                "pass": ok,
            }
        )

        if not ok:
            raise RuntimeError(
                "AGGC measurement source SHA256 changed after Step24A freeze."
            )

    return pd.DataFrame(rows)


# =============================================================================
# PASS 1: ELIGIBILITY + EXACT q20/q80
# =============================================================================

def pass1_thresholds(
    contract: Contract,
    chunksize: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    usecols = ["slide_id", "PC6", *contract.formal_metrics]

    score_blocks: Dict[str, List[np.ndarray]] = {}

    total_rows = 0
    partition_rows = {
        contract.primary_partition: 0,
        contract.secondary_partition: 0,
    }
    unrecognized_partition_rows = 0
    eligible_rows = 0

    reader = pd.read_csv(
        contract.source_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        partition = chunk["slide_id"].astype(str).map(parse_partition)
        recognized = partition.notna()

        unrecognized_partition_rows += int((~recognized).sum())

        for part in [contract.primary_partition, contract.secondary_partition]:
            partition_rows[part] += int(partition.eq(part).sum())

        if not recognized.any():
            continue

        c = chunk.loc[recognized].copy()

        required = ["PC6", *contract.formal_metrics]
        fm = finite_mask(c, required)

        c = c.loc[fm].copy()
        eligible_rows += len(c)

        if len(c):
            c["PC6"] = pd.to_numeric(c["PC6"], errors="coerce")

            for slide_id, sub in c.groupby("slide_id", sort=False):
                arr = sub["PC6"].to_numpy(dtype=np.float64, copy=True)

                if slide_id not in score_blocks:
                    score_blocks[slide_id] = []

                score_blocks[slide_id].append(arr)

        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(
                f"[PASS1 CHUNK {chunk_idx:04d}] "
                f"rows_read={total_rows:,} | eligible={eligible_rows:,}"
            )

    threshold_rows = []
    pc6_values: Dict[str, np.ndarray] = {}

    for slide_id in sorted(score_blocks):
        values = np.concatenate(score_blocks[slide_id])

        partition = parse_partition(slide_id)

        n = len(values)

        eligible = n >= contract.min_tiles

        if eligible:
            q20 = float(np.quantile(values, contract.low_q))
            q80 = float(np.quantile(values, contract.high_q))

            if not np.isfinite(q20) or not np.isfinite(q80):
                raise RuntimeError(f"{slide_id}: non-finite PC6 quantiles.")

            if q20 >= q80:
                raise RuntimeError(
                    f"{slide_id}: degenerate PC6 q20/q80 thresholds "
                    f"({q20} >= {q80}); no rescue is permitted."
                )

            n_low = int(np.sum(values <= q20))
            n_high = int(np.sum(values >= q80))

            if n_low == 0 or n_high == 0:
                raise RuntimeError(
                    f"{slide_id}: frozen tail assignment produced an empty stratum."
                )
        else:
            q20 = np.nan
            q80 = np.nan
            n_low = 0
            n_high = 0

        threshold_rows.append(
            {
                "slide_id": slide_id,
                "partition": partition,
                "n_finite_required_tiles": n,
                "min_required_tiles": contract.min_tiles,
                "slide_eligible": bool(eligible),
                "pc6_q20": q20,
                "pc6_q80": q80,
                "n_pc6_low_candidates": n_low,
                "n_pc6_high_candidates": n_high,
                "low_quantile": contract.low_q,
                "high_quantile": contract.high_q,
                "effect_calculated": False,
            }
        )

        if eligible:
            # Keep only eligible arrays for final technical audit; no morphology.
            pc6_values[slide_id] = values

    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["partition", "slide_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    stats = {
        "total_rows_read": int(total_rows),
        "primary_partition_rows": int(
            partition_rows[contract.primary_partition]
        ),
        "secondary_partition_rows": int(
            partition_rows[contract.secondary_partition]
        ),
        "unrecognized_partition_rows": int(unrecognized_partition_rows),
        "finite_required_rows": int(eligible_rows),
        "slides_seen": int(len(score_blocks)),
        "eligible_slides": int(thresholds["slide_eligible"].sum()),
        "eligible_test_slides": int(
            (
                thresholds["slide_eligible"]
                & thresholds["partition"].eq(contract.primary_partition)
            ).sum()
        ),
        "eligible_train_slides": int(
            (
                thresholds["slide_eligible"]
                & thresholds["partition"].eq(contract.secondary_partition)
            ).sum()
        ),
    }

    return thresholds, pc6_values, stats


# =============================================================================
# PASS 2: DETERMINISTIC TAIL SAMPLING
# =============================================================================

@dataclass
class Reservoir:
    rng: np.random.Generator
    seen: int
    data: Optional[pd.DataFrame]


def reservoir_update(
    state: Reservoir,
    candidates: pd.DataFrame,
    k: int,
) -> None:
    if len(candidates) == 0:
        return

    c = candidates.copy()

    priorities = state.rng.random(len(c))
    c["_sampling_priority"] = priorities
    state.seen += len(c)

    if state.data is None or len(state.data) == 0:
        merged = c
    else:
        merged = pd.concat([state.data, c], ignore_index=True)

    if len(merged) > k:
        merged = merged.nsmallest(
            k,
            "_sampling_priority",
            keep="first",
        )

    state.data = merged.reset_index(drop=True)


def pass2_sampling(
    contract: Contract,
    thresholds: pd.DataFrame,
    header_columns: Sequence[str],
    chunksize: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    eligible = thresholds.loc[thresholds["slide_eligible"]].copy()

    threshold_map = eligible.set_index("slide_id")[
        ["partition", "pc6_q20", "pc6_q80"]
    ].to_dict(orient="index")

    identity_source_cols = [
        c for c in ["tile_index", "x", "y", "histologic_class"]
        if c in header_columns
    ]

    usecols = [
        "slide_id",
        "PC6",
        *contract.formal_metrics,
        contract.specificity_metric,
        *identity_source_cols,
    ]

    # Preserve order, remove duplicates.
    usecols = list(dict.fromkeys(usecols))

    reservoirs: Dict[Tuple[str, str, str], Reservoir] = {}

    for slide_id, info in threshold_map.items():
        for stratum in ["PC6_LOW", "PC6_HIGH"]:
            key = (str(info["partition"]), slide_id, stratum)

            reservoirs[key] = Reservoir(
                rng=np.random.default_rng(
                    stable_seed(
                        contract.sampling_seed,
                        info["partition"],
                        slide_id,
                        stratum,
                    )
                ),
                seen=0,
                data=None,
            )

    total_rows = 0
    eligible_rows = 0
    tail_rows = 0

    reader = pd.read_csv(
        contract.source_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        chunk_start = total_rows
        total_rows += len(chunk)

        chunk["source_row_index"] = (
            np.arange(len(chunk), dtype=np.int64)
            + int(chunk_start)
        )

        recognized_slides = chunk["slide_id"].astype(str).isin(threshold_map)

        if not recognized_slides.any():
            continue

        c = chunk.loc[recognized_slides].copy()

        fm = finite_mask(
            c,
            ["PC6", *contract.formal_metrics],
        )

        c = c.loc[fm].copy()

        eligible_rows += len(c)

        if len(c) == 0:
            continue

        c["PC6"] = pd.to_numeric(c["PC6"], errors="coerce")

        # Formal morphology metrics numeric.
        for metric in contract.formal_metrics:
            c[metric] = pd.to_numeric(c[metric], errors="coerce")

        # Specificity reference is carried but NOT required for eligibility.
        c[contract.specificity_metric] = pd.to_numeric(
            c[contract.specificity_metric],
            errors="coerce",
        )

        for slide_id, sub in c.groupby("slide_id", sort=False):
            info = threshold_map[str(slide_id)]

            low = sub.loc[sub["PC6"] <= float(info["pc6_q20"])].copy()
            high = sub.loc[sub["PC6"] >= float(info["pc6_q80"])].copy()

            if len(low):
                low["sampling_stratum"] = "PC6_LOW"
                low["partition"] = str(info["partition"])
                low["pc6_q20"] = float(info["pc6_q20"])
                low["pc6_q80"] = float(info["pc6_q80"])

                reservoir_update(
                    reservoirs[
                        (str(info["partition"]), str(slide_id), "PC6_LOW")
                    ],
                    low,
                    contract.max_tiles_per_slide_stratum,
                )

            if len(high):
                high["sampling_stratum"] = "PC6_HIGH"
                high["partition"] = str(info["partition"])
                high["pc6_q20"] = float(info["pc6_q20"])
                high["pc6_q80"] = float(info["pc6_q80"])

                reservoir_update(
                    reservoirs[
                        (str(info["partition"]), str(slide_id), "PC6_HIGH")
                    ],
                    high,
                    contract.max_tiles_per_slide_stratum,
                )

            tail_rows += len(low) + len(high)

        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(
                f"[PASS2 CHUNK {chunk_idx:04d}] "
                f"rows_read={total_rows:,} | finite_required={eligible_rows:,} | "
                f"tail_memberships_seen={tail_rows:,}"
            )

    frames = []
    audit_rows = []

    for (partition, slide_id, stratum), state in sorted(reservoirs.items()):
        if state.data is None:
            selected = pd.DataFrame()
        else:
            selected = state.data.copy()

        audit_rows.append(
            {
                "partition": partition,
                "slide_id": slide_id,
                "sampling_stratum": stratum,
                "n_tail_candidates_seen": int(state.seen),
                "max_tiles_per_slide_per_stratum": int(
                    contract.max_tiles_per_slide_stratum
                ),
                "n_sampled_tiles": int(len(selected)),
                "sampling_seed": int(contract.sampling_seed),
                "sampling_algorithm": (
                    "deterministic_uniform_priority_reservoir_per_slide_stratum"
                ),
                "effect_calculated": False,
            }
        )

        if len(selected) == 0:
            raise RuntimeError(
                f"Eligible slide {slide_id} has no sampled tiles for {stratum}."
            )

        selected["sample_id"] = selected.apply(
            lambda r: (
                f"24b__{partition}__{slide_id}__{stratum}__"
                f"row{int(r['source_row_index'])}"
            ),
            axis=1,
        )

        selected["sampling_role"] = (
            "PRIMARY_HELD_OUT_REPLICATION"
            if partition == contract.primary_partition
            else "SECONDARY_CHARACTERIZATION"
        )

        frames.append(selected)

    sampled = pd.concat(frames, ignore_index=True)

    sampled = sampled.sort_values(
        [
            "partition",
            "slide_id",
            "sampling_stratum",
            "source_row_index",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if "_sampling_priority" in sampled.columns:
        sampled = sampled.rename(
            columns={"_sampling_priority": "deterministic_sampling_priority"}
        )

    stats = {
        "total_rows_read_pass2": int(total_rows),
        "finite_required_rows_pass2": int(eligible_rows),
        "tail_memberships_seen": int(tail_rows),
        "sampled_rows": int(len(sampled)),
    }

    return sampled, pd.DataFrame(audit_rows), stats


# =============================================================================
# TECHNICAL DEDUP AUDIT
# =============================================================================

def build_duplicate_audit(
    sampled: pd.DataFrame,
    header_columns: Sequence[str],
) -> pd.DataFrame:
    identity_cols = resolve_identity_columns(header_columns)

    missing = [c for c in identity_cols if c not in sampled.columns]

    if missing:
        raise RuntimeError(
            f"Cannot establish sampled tile identity; missing columns: {missing}"
        )

    dup = sampled.duplicated(identity_cols, keep=False)

    rows = [
        {
            "identity_rule": " + ".join(identity_cols),
            "sampled_rows": int(len(sampled)),
            "unique_identities": int(
                sampled.drop_duplicates(identity_cols).shape[0]
            ),
            "duplicate_rows": int(dup.sum()),
            "duplicate_identity_groups": int(
                sampled.loc[dup].groupby(identity_cols).ngroups
                if dup.any()
                else 0
            ),
            "pass": bool(not dup.any()),
        }
    ]

    if dup.any():
        example = sampled.loc[
            dup,
            [
                *identity_cols,
                "partition",
                "sampling_stratum",
                "sample_id",
            ],
        ].head(20)

        raise RuntimeError(
            "Sampled AGGC physical/logical tile identities are duplicated.\n"
            "Step24B refuses silent deduplication because Step24A did not freeze "
            "an outcome-dependent rescue.\n"
            + example.to_string(index=False)
        )

    return pd.DataFrame(rows)


# =============================================================================
# MEASUREMENT SUMMARIES
# =============================================================================

def build_slide_stratum_summary(
    sampled: pd.DataFrame,
    contract: Contract,
) -> pd.DataFrame:
    measurement_metrics = [
        "PC6",
        *contract.formal_metrics,
        contract.specificity_metric,
    ]

    rows = []

    for (partition, slide_id, stratum), sub in sampled.groupby(
        ["partition", "slide_id", "sampling_stratum"],
        sort=False,
    ):
        row: Dict[str, Any] = {
            "partition": partition,
            "slide_id": slide_id,
            "sampling_stratum": stratum,
            "n_sampled_tiles": int(len(sub)),
            "pc6_q20": float(sub["pc6_q20"].iloc[0]),
            "pc6_q80": float(sub["pc6_q80"].iloc[0]),
            "high_minus_low_effect_calculated": False,
        }

        for metric in measurement_metrics:
            row[f"{metric}__median"] = median_or_nan(sub[metric])
            row[f"{metric}__n_finite"] = int(
                pd.to_numeric(sub[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .sum()
            )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ["partition", "slide_id", "sampling_stratum"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Strictly two strata for every eligible slide.
    counts = out.groupby(["partition", "slide_id"])["sampling_stratum"].nunique()

    bad = counts[counts != 2]

    if len(bad):
        raise RuntimeError(
            "At least one eligible slide lacks both frozen PC6 strata:\n"
            + bad.to_string()
        )

    return out


def build_partition_measurement_summary(
    thresholds: pd.DataFrame,
    sampled: pd.DataFrame,
    slide_summary: pd.DataFrame,
    contract: Contract,
) -> pd.DataFrame:
    rows = []

    for partition in [
        contract.primary_partition,
        contract.secondary_partition,
    ]:
        t = thresholds.loc[thresholds["partition"].eq(partition)]
        s = sampled.loc[sampled["partition"].eq(partition)]
        z = slide_summary.loc[slide_summary["partition"].eq(partition)]

        rows.append(
            {
                "partition": partition,
                "formal_role": (
                    "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"
                    if partition == contract.primary_partition
                    else "SECONDARY_EXTERNAL_CHARACTERIZATION"
                ),
                "n_slides_seen_with_finite_required_rows": int(len(t)),
                "n_eligible_slides": int(t["slide_eligible"].sum()),
                "n_sampled_tiles": int(len(s)),
                "n_pc6_low_sampled_tiles": int(
                    s["sampling_stratum"].eq("PC6_LOW").sum()
                ),
                "n_pc6_high_sampled_tiles": int(
                    s["sampling_stratum"].eq("PC6_HIGH").sum()
                ),
                "n_slide_stratum_summary_rows": int(len(z)),
                "effect_calculated": False,
                "direction_concordance_calculated": False,
                "bootstrap_calculated": False,
                "p_value_calculated": False,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# HASHES / README
# =============================================================================

def build_source_hashes() -> pd.DataFrame:
    rows = []

    for role, path in [
        ("step24a_plan", A_PLAN),
        ("step24a_endpoints", A_ENDPOINTS),
        ("step24a_specificity", A_SPECIFICITY),
        ("step24a_partitions", A_PARTITIONS),
        ("step24a_sampling", A_SAMPLING),
        ("step24a_evaluation", A_EVALUATION),
        ("step24a_source_snapshot", A_SOURCE),
        ("step24a_metadata", A_METADATA),
        ("step24a_claim_ceiling", A_CLAIM),
    ]:
        require_file(path, role)

        rows.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    return pd.DataFrame(rows)


def readme_text(contract: Contract, chunksize: int) -> str:
    metrics = "\n".join(f"  {x}" for x in contract.formal_metrics)

    return f"""STEP 24B
AGGC EXTERNAL PC6 MORPHOLOGY TRANSFER MEASUREMENT
=================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

Role
----
MEASUREMENT / QC ONLY.

Frozen authority
----------------
{STEP24A}

Measurement source
------------------
{contract.source_path}

Partitions
----------
PRIMARY:
  {contract.primary_partition}

SECONDARY:
  {contract.secondary_partition}

PC6 state definition
--------------------
PC6_LOW:
  raw PC6 <= within-slide q{int(contract.low_q * 100)}

PC6_HIGH:
  raw PC6 >= within-slide q{int(contract.high_q * 100)}

Eligibility
-----------
A tile must have finite:
  PC6
{metrics}

A slide must have at least:
  {contract.min_tiles} eligible tiles

NPI_nuclear_packing
-------------------
Carried as a specificity reference only.
It is NOT required for formal slide eligibility and is NOT promoted here.

Sampling
--------
Up to {contract.max_tiles_per_slide_stratum} tiles per slide / PC6 stratum.

Sampling seed:
  {contract.sampling_seed}

Algorithm:
  deterministic uniform-priority reservoir per slide / stratum

Chunk size:
  {chunksize:,}

Measurement outputs
-------------------
1. exact slide eligibility + q20/q80 thresholds
2. frozen sampled tile manifest
3. sampled tile measurements
4. slide x PC6-stratum morphology medians
5. technical sampling / duplicate / provenance audits

PROHIBITED IN STEP24B
---------------------
PC6_HIGH - PC6_LOW effect
direction concordance
bootstrap CI
p-values / FDR
Train/Test inferential comparison
Train/Test pooling
tumor AUC
PC screening
PCA refit
PC6 sign tuning
new morphology metric selection
NPI promotion
Gleason-state analysis

NEXT
----
Step24C may calculate the frozen AGGC morphology-transfer inference using only
the Step24B frozen measurement outputs and the Step24A endpoint family.
"""


# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Measure AGGC external PC6 morphology transfer under the "
            "Step24A frozen contract."
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
    )

    p.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--verify-source-hash",
        action="store_true",
        help=(
            "Recompute the potentially large AGGC source SHA256 and verify it "
            "against Step24A. Size and mtime are always verified."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive.")

    banner("STEP 24B: MEASURE AGGC EXTERNAL PC6 MORPHOLOGY TRANSFER")

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP24A]       {STEP24A}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[CHUNKSIZE]     {args.chunksize:,}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          MEASUREMENT / QC ONLY")
    print("[EFFECT]        NONE")
    print("[DIRECTION]     NOT EVALUATED")
    print("[BOOTSTRAP]     NONE")
    print("[P-VALUES]      NONE")
    print("[FDR]           NONE")
    print("[TUMOR AUC]     NONE")
    print("[TRAIN/TEST]    NEVER POOLED")
    print("[PCA REFIT]     NONE")
    print("[PC SCREEN]     NONE")
    print("[SIGN TUNING]   NONE")
    print("[NPI PROMOTION] NONE")
    print("[GLEASON STATE] NONE")

    prepare_output(args.output, args.overwrite)

    # -------------------------------------------------------------------------
    # Contract.
    # -------------------------------------------------------------------------

    banner("VALIDATE STEP24A FROZEN CONTRACT")

    contract = load_contract()

    print(f"Measurement source                   : {contract.source_path}")
    print(f"Primary partition                    : {contract.primary_partition}")
    print(f"Secondary partition                  : {contract.secondary_partition}")
    print(f"PC6 LOW                              : <= q{int(contract.low_q * 100)}")
    print(f"PC6 HIGH                             : >= q{int(contract.high_q * 100)}")
    print(f"Minimum eligible tiles / slide       : {contract.min_tiles}")
    print(
        f"Max sampled tiles / slide / stratum  : "
        f"{contract.max_tiles_per_slide_stratum}"
    )
    print(f"Sampling seed                        : {contract.sampling_seed}")
    print(f"Formal metrics                       : {len(contract.formal_metrics)}")
    print(f"NPI role                              : specificity reference only")

    # Header / schema.
    header = pd.read_csv(contract.source_path, nrows=0)
    header_columns = header.columns.astype(str).tolist()

    required_columns = [
        "slide_id",
        "PC6",
        *contract.formal_metrics,
        contract.specificity_metric,
    ]

    missing = [c for c in required_columns if c not in header_columns]

    if missing:
        raise RuntimeError(
            "AGGC measurement source no longer contains the frozen columns:\n"
            + "\n".join(missing)
        )

    provenance_audit = validate_source_provenance(
        contract,
        verify_hash=args.verify_source_hash,
    )

    # -------------------------------------------------------------------------
    # Pass 1.
    # -------------------------------------------------------------------------

    banner("PASS 1: EXACT SLIDE ELIGIBILITY + WITHIN-SLIDE PC6 q20/q80")

    thresholds, _pc6_arrays, pass1_stats = pass1_thresholds(
        contract,
        args.chunksize,
    )

    thresholds.to_csv(
        args.output / O_THRESHOLDS,
        index=False,
    )

    print()
    print("[PASS1 SUMMARY]")
    for k, v in pass1_stats.items():
        print(f"{k:<42}: {v:,}" if isinstance(v, int) else f"{k:<42}: {v}")

    print()
    print(
        thresholds.groupby(["partition", "slide_eligible"])
        .size()
        .rename("n_slides")
        .reset_index()
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Pass 2.
    # -------------------------------------------------------------------------

    banner("PASS 2: FROZEN PC6 TAIL SAMPLING / TILE MEASUREMENT")

    sampled, sampling_audit, pass2_stats = pass2_sampling(
        contract,
        thresholds,
        header_columns,
        args.chunksize,
    )

    dup_audit = build_duplicate_audit(
        sampled,
        header_columns,
    )

    # Preserve only frozen/source measurement columns + technical provenance.
    ordered_cols = [
        "sample_id",
        "partition",
        "sampling_role",
        "slide_id",
        "sampling_stratum",
        "source_row_index",
    ]

    for c in ["tile_index", "x", "y", "histologic_class"]:
        if c in sampled.columns:
            ordered_cols.append(c)

    ordered_cols += [
        "PC6",
        "pc6_q20",
        "pc6_q80",
        *contract.formal_metrics,
        contract.specificity_metric,
        "deterministic_sampling_priority",
    ]

    ordered_cols = [c for c in ordered_cols if c in sampled.columns]

    sampled_out = sampled[ordered_cols].copy()

    sampled_out.to_csv(
        args.output / O_TILE,
        index=False,
        compression="gzip",
    )

    sample_manifest_cols = [
        c
        for c in [
            "sample_id",
            "partition",
            "sampling_role",
            "slide_id",
            "sampling_stratum",
            "source_row_index",
            "tile_index",
            "x",
            "y",
            "pc6_q20",
            "pc6_q80",
            "deterministic_sampling_priority",
        ]
        if c in sampled_out.columns
    ]

    sampled_out[sample_manifest_cols].to_csv(
        args.output / O_SAMPLE,
        index=False,
    )

    sampling_audit.to_csv(
        args.output / O_SAMPLING_AUDIT,
        index=False,
    )

    dup_audit.to_csv(
        args.output / O_DUP_AUDIT,
        index=False,
    )

    print()
    print("[PASS2 SUMMARY]")
    for k, v in pass2_stats.items():
        print(f"{k:<42}: {v:,}" if isinstance(v, int) else f"{k:<42}: {v}")

    print()
    print(
        sampled_out.groupby(["partition", "sampling_stratum"])
        .size()
        .rename("n_sampled_tiles")
        .reset_index()
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Descriptive measurement summaries only.
    # -------------------------------------------------------------------------

    banner("DESCRIPTIVE SLIDE x PC6-STRATUM MEASUREMENT SUMMARIES")

    slide_summary = build_slide_stratum_summary(
        sampled_out,
        contract,
    )

    slide_summary.to_csv(
        args.output / O_SLIDE,
        index=False,
    )

    partition_summary = build_partition_measurement_summary(
        thresholds,
        sampled_out,
        slide_summary,
        contract,
    )

    partition_summary.to_csv(
        args.output / O_PARTITION,
        index=False,
    )

    print(partition_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Input integrity audit.
    # -------------------------------------------------------------------------

    input_audit_rows = [
        {
            "audit_item": "step24a_version",
            "observed": EXPECTED_24A_VERSION,
            "expected": EXPECTED_24A_VERSION,
            "pass": True,
        },
        {
            "audit_item": "formal_transfer_endpoint_count",
            "observed": len(contract.formal_metrics),
            "expected": 6,
            "pass": len(contract.formal_metrics) == 6,
        },
        {
            "audit_item": "specificity_reference",
            "observed": contract.specificity_metric,
            "expected": EXPECTED_SPECIFICITY,
            "pass": contract.specificity_metric == EXPECTED_SPECIFICITY,
        },
        {
            "audit_item": "primary_partition",
            "observed": contract.primary_partition,
            "expected": EXPECTED_PRIMARY_PARTITION,
            "pass": contract.primary_partition == EXPECTED_PRIMARY_PARTITION,
        },
        {
            "audit_item": "secondary_partition",
            "observed": contract.secondary_partition,
            "expected": EXPECTED_SECONDARY_PARTITION,
            "pass": contract.secondary_partition == EXPECTED_SECONDARY_PARTITION,
        },
        {
            "audit_item": "high_minus_low_effect_calculated",
            "observed": False,
            "expected": False,
            "pass": True,
        },
        {
            "audit_item": "direction_concordance_calculated",
            "observed": False,
            "expected": False,
            "pass": True,
        },
        {
            "audit_item": "bootstrap_calculated",
            "observed": False,
            "expected": False,
            "pass": True,
        },
        {
            "audit_item": "p_values_calculated",
            "observed": False,
            "expected": False,
            "pass": True,
        },
    ]

    input_audit = pd.concat(
        [
            pd.DataFrame(input_audit_rows),
            provenance_audit,
        ],
        ignore_index=True,
    )

    input_audit.to_csv(
        args.output / O_INPUT_AUDIT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Provenance package.
    # -------------------------------------------------------------------------

    source_execution = pd.DataFrame(
        [
            {
                "source_path": str(contract.source_path),
                "file_size_bytes": int(contract.source_path.stat().st_size),
                "file_mtime_ns": int(contract.source_path.stat().st_mtime_ns),
                "step24a_sha256": contract.source_sha256,
                "step24a_sha256_skipped": contract.source_sha256_skipped,
                "sha256_reverified_in_step24b": bool(args.verify_source_hash),
                "rows_read_pass1": int(pass1_stats["total_rows_read"]),
                "rows_read_pass2": int(pass2_stats["total_rows_read_pass2"]),
                "chunksize": int(args.chunksize),
            }
        ]
    )

    source_execution.to_csv(
        args.output / O_SOURCE_SNAPSHOT,
        index=False,
    )

    hashes = build_source_hashes()

    hashes.to_csv(
        args.output / O_HASHES,
        index=False,
    )

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "role": "AGGC_EXTERNAL_PC6_MORPHOLOGY_TRANSFER_MEASUREMENT_ONLY",
        "step24a_authority": str(STEP24A),
        "measurement_source": str(contract.source_path),
        "chunksize": int(args.chunksize),
        "formal_transfer_metrics": list(contract.formal_metrics),
        "specificity_reference": contract.specificity_metric,
        "primary_partition": contract.primary_partition,
        "secondary_partition": contract.secondary_partition,
        "pc6_strata": {
            "low": f"raw PC6 <= within-slide q{int(contract.low_q * 100)}",
            "high": f"raw PC6 >= within-slide q{int(contract.high_q * 100)}",
            "effect": "NOT_CALCULATED_IN_STEP24B",
        },
        "eligibility": {
            "required_finite_fields": ["PC6", *contract.formal_metrics],
            "npi_required_for_eligibility": False,
            "min_tiles_per_slide": int(contract.min_tiles),
        },
        "sampling": {
            "max_tiles_per_slide_per_stratum": int(
                contract.max_tiles_per_slide_stratum
            ),
            "sampling_seed": int(contract.sampling_seed),
            "algorithm": (
                "deterministic_uniform_priority_reservoir_per_slide_stratum"
            ),
        },
        "counts": {
            **pass1_stats,
            **pass2_stats,
            "slide_stratum_summary_rows": int(len(slide_summary)),
        },
        "firewall": {
            "high_minus_low_effect_calculated": False,
            "direction_concordance_calculated": False,
            "bootstrap_calculated": False,
            "p_values_calculated": False,
            "fdr_calculated": False,
            "train_test_inferential_comparison": False,
            "train_test_pooled": False,
            "tumor_auc_calculated": False,
            "pc_screening": False,
            "pca_refit": False,
            "pc6_sign_tuning": False,
            "new_metric_selection": False,
            "npi_promoted": False,
            "gleason_state_analysis": False,
        },
        "next_step": (
            "Step24C may evaluate the frozen AGGC PC6 morphology-transfer "
            "endpoints from Step24B measurements."
        ),
    }

    save_json(
        args.output / O_METADATA,
        metadata,
    )

    (args.output / O_README).write_text(
        readme_text(contract, args.chunksize),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final firewall.
    # -------------------------------------------------------------------------

    banner("STEP24B MEASUREMENT-ONLY FIREWALL AUDIT")

    audit_items = [
        ("Step24A frozen contract inherited", "YES"),
        ("AGGC rows read for measurement", "YES"),
        ("Exact within-slide PC6 q20/q80 calculated", "YES"),
        ("Frozen deterministic sampling executed", "YES"),
        ("Six formal morphology metrics measured", "YES"),
        ("NPI carried as specificity reference only", "YES"),
        ("PC6_HIGH - PC6_LOW effect calculated", "NO"),
        ("Direction concordance calculated", "NO"),
        ("Bootstrap calculated", "NO"),
        ("P-values calculated", "NO"),
        ("FDR calculated", "NO"),
        ("Train/Test inferential comparison", "NO"),
        ("Train/Test pooled", "NO"),
        ("Tumor AUC calculated", "NO"),
        ("PC1-PC8 screening", "NO"),
        ("PCA refit", "NO"),
        ("PC6 sign tuning", "NO"),
        ("New morphology metric selected", "NO"),
        ("NPI promoted", "NO"),
        ("Normal/G3/G4/G5 state analysis", "NO"),
    ]

    for item, status in audit_items:
        print(f"{item:<62}: {status}")

    print()
    print("[OK] Saved:")

    expected_outputs = [
        O_INPUT_AUDIT,
        O_THRESHOLDS,
        O_SAMPLE,
        O_TILE,
        O_SLIDE,
        O_PARTITION,
        O_SAMPLING_AUDIT,
        O_DUP_AUDIT,
        O_SOURCE_SNAPSHOT,
        O_HASHES,
        O_METADATA,
        O_README,
    ]

    for name in expected_outputs:
        p = args.output / name

        if not p.exists():
            raise RuntimeError(f"Expected Step24B output missing: {p}")

        print(f"  {p}")

    print()
    print("[DONE] Step24B AGGC external PC6 morphology-transfer measurement complete.")
    print(
        "[NEXT] Step24C may calculate ONLY the Step24A-frozen directional "
        "replication / bootstrap evaluation from these frozen measurements."
    )


if __name__ == "__main__":
    main()
