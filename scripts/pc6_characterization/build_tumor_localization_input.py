#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 25B
==========
BUILD AGGC PC6 EXTERNAL TUMOR-LOCALIZATION INPUT

VERSION
-------
1.0.0

SCHEMA
------
25b-aggc-pc6-external-tumor-localization-input-schema-1.0

ROLE
----
MEASUREMENT / INPUT-CONSTRUCTION / QC ONLY.

This step executes the Step25A frozen AGGC PC6 tumor-localization contract.

It may:
    - read exact frozen PC6 scores from pca_targets[:,5];
    - read exact AGGC tile labels;
    - reconcile tile_index/x/y exactly;
    - assign the frozen binary tumor target:
          G3/G4/G5      -> 1
          Normal/Stroma -> 0
          unsupported   -> excluded by label definition;
    - calculate per-slide tumor/non-tumor counts;
    - determine slide eligibility from the frozen requirement:
          >=1 tumor tile and >=1 non-tumor tile;
    - freeze the final Test/Train tumor-localization measurement input.

It MUST NOT calculate:
    - ROC AUC;
    - bootstrap confidence intervals;
    - p-values;
    - score thresholds;
    - PC ranking;
    - sign tuning;
    - pooled Train/Test inference.

PRIMARY / SECONDARY
-------------------
Subset1_Test:
    PRIMARY held-out external replication

Subset1_Train:
    SECONDARY external characterization

No Train/Test pooling.

EXACT TILE IDENTITY
-------------------
For each slide:

    coords:
        aggc_virchow2_embeddings/<slide>_coords.npy

    frozen PCA targets:
        aggc_pc1_pc8_grid_heatmaps/<slide>/<slide>_pca_targets.npy

    tile labels:
        aggc_gleason_tile_labels/<slide>_tile_labels.csv

Required:
    coords shape = (N,2)
    PCA rows = N
    PCA contains PC6
    tile labels contain exactly N rows
    tile_index = 0...N-1
    x/y exactly equal coords.npy
    PC6 = pca_targets[:,5]

No approximate coordinate matching.
No PCA refit.
No PCA reprojection.
No PC6 sign flip.

FROZEN LABEL DEFINITION
-----------------------
Tumor:
    G3
    G4
    G5

Non-tumor:
    Normal
    Stroma

Unsupported/other labels:
    excluded from AUC eligibility by label definition only

No outcome-based slide exclusion.

NEXT
----
Step25C may calculate the frozen per-slide ROC AUC / bootstrap evaluation from
the Step25B final measurement input only.
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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / PATHS
# =============================================================================

VERSION = "1.0.1"
SCHEMA = "25b-aggc-pc6-external-tumor-localization-input-schema-1.0.1"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT

STEP25A = RESULTS / "25a_aggc_pc6_external_tumor_localization_specification"

A_PLAN = STEP25A / "aggc_pc6_tumor_localization_plan.json"
A_LABELS = STEP25A / "aggc_pc6_tumor_label_definition.csv"
A_PARTITIONS = STEP25A / "aggc_pc6_tumor_localization_partition_specification.csv"
A_ENDPOINTS = STEP25A / "aggc_pc6_tumor_localization_endpoint_specification.csv"
A_BOOTSTRAP = STEP25A / "aggc_pc6_tumor_localization_bootstrap_specification.csv"
A_METADATA = STEP25A / "analysis_metadata.json"
A_CLAIM = STEP25A / "pc6_aggc_tumor_localization_claim_ceiling.txt"

AGGC_ROOT = CONFIG_AGGC_ROOT

LABEL_ROOT = AGGC_ROOT / "aggc_gleason_tile_labels"
PCA_ROOT = AGGC_ROOT / "aggc_pc1_pc8_grid_heatmaps"
COORD_ROOT = AGGC_ROOT / "aggc_virchow2_embeddings"

DEFAULT_OUT = RESULTS / "25b_aggc_pc6_external_tumor_localization_input"

PRIMARY_PARTITION = "Subset1_Test"
SECONDARY_PARTITION = "Subset1_Train"

EXPECTED_TEST_SLIDES = 45
EXPECTED_TRAIN_SLIDES = 104

PC6_INDEX_ZERO_BASED = 5

TUMOR_LABELS = {"G3", "G4", "G5"}
NON_TUMOR_LABELS = {"Normal", "Stroma"}

MIN_TUMOR_TILES = 1
MIN_NON_TUMOR_TILES = 1


# =============================================================================
# OUTPUTS
# =============================================================================

O_TILE = "aggc_pc6_tumor_localization_tile_input.csv.gz"
O_SLIDE = "aggc_pc6_tumor_localization_slide_eligibility.csv"
O_PARTITION = "aggc_pc6_tumor_localization_partition_summary.csv"
O_ALIGNMENT = "aggc_pc6_tumor_localization_alignment_audit.csv"
O_LABEL = "aggc_pc6_tumor_localization_label_audit.csv"
O_INVENTORY = "aggc_pc6_tumor_localization_resolved_input_inventory.csv"
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
                "Use --overwrite only for intentional regeneration under the "
                "unchanged Step25A freeze."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def natural_key(text: str):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(text))
    ]


def parse_partition(slide_id: str) -> str:
    s = str(slide_id)
    if s.startswith("Subset1_Test_"):
        return PRIMARY_PARTITION
    if s.startswith("Subset1_Train_"):
        return SECONDARY_PARTITION
    return "OTHER"


def normalize_label(value: Any) -> str:
    s = str(value).strip()

    aliases = {
        "3": "G3",
        "4": "G4",
        "5": "G5",
        "gleason3": "G3",
        "gleason_3": "G3",
        "gleason4": "G4",
        "gleason_4": "G4",
        "gleason5": "G5",
        "gleason_5": "G5",
        "normal": "Normal",
        "benign": "Normal",
        "stroma": "Stroma",
    }

    key = (
        s.lower()
        .replace(" ", "")
        .replace("-", "_")
    )

    if s in {"G3", "G4", "G5", "Normal", "Stroma"}:
        return s

    return aliases.get(key, s)


def exact_coords_path(slide_id: str) -> Path:
    candidates = [
        COORD_ROOT / f"{slide_id}_coords.npy",
        COORD_ROOT / slide_id / f"{slide_id}_coords.npy",
    ]

    found = [p for p in candidates if p.is_file()]

    if len(found) != 1:
        raise RuntimeError(
            f"{slide_id}: expected exactly one coords.npy; found={found}"
        )

    return found[0]


def exact_pca_path(slide_id: str) -> Path:
    candidates = [
        PCA_ROOT / slide_id / f"{slide_id}_pca_targets.npy",
        PCA_ROOT / f"{slide_id}_pca_targets.npy",
    ]

    found = [p for p in candidates if p.is_file()]

    if len(found) != 1:
        raise RuntimeError(
            f"{slide_id}: expected exactly one pca_targets.npy; found={found}"
        )

    return found[0]


def exact_label_path(slide_id: str) -> Path:
    candidates = [
        LABEL_ROOT / f"{slide_id}_tile_labels.csv",
        LABEL_ROOT / slide_id / f"{slide_id}_tile_labels.csv",
    ]

    found = [p for p in candidates if p.is_file()]

    if len(found) != 1:
        raise RuntimeError(
            f"{slide_id}: expected exactly one tile label CSV; found={found}"
        )

    return found[0]


def load_coords(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(
            f"Invalid coords shape {arr.shape}: {path}"
        )

    return np.asarray(arr, dtype=np.int64)


def load_pca(path: Path, expected_n: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")

    if arr.ndim != 2:
        raise RuntimeError(
            f"Invalid PCA shape {arr.shape}: {path}"
        )

    if arr.shape[0] != expected_n:
        raise RuntimeError(
            f"PCA/coords row mismatch: PCA={arr.shape[0]}, coords={expected_n}\n{path}"
        )

    if arr.shape[1] <= PC6_INDEX_ZERO_BASED:
        raise RuntimeError(
            f"PCA target array lacks PC6: shape={arr.shape}"
        )

    return np.asarray(arr)


def resolve_label_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)

    def first(candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    tile_col = first(
        ["tile_index", "tile_idx", "index", "idx"]
    )

    x_col = first(
        ["x", "tile_x", "coord_x", "x_coord"]
    )

    y_col = first(
        ["y", "tile_y", "coord_y", "y_coord"]
    )

    label_col = first(
        [
            "final_label",
            "histologic_class",
            "histologic_label",
            "gleason_label",
            "label",
            "class",
            "tissue_label",
        ]
    )

    missing = [
        name
        for name, value in [
            ("tile_index", tile_col),
            ("x", x_col),
            ("y", y_col),
            ("label", label_col),
        ]
        if value is None
    ]

    if missing:
        raise RuntimeError(
            f"Could not resolve required tile-label columns {missing}; "
            f"available columns={cols}"
        )

    return {
        "tile_index": tile_col,
        "x": x_col,
        "y": y_col,
        "label": label_col,
    }


# =============================================================================
# VALIDATE STEP25A
# =============================================================================

def validate_step25a() -> Dict[str, Any]:
    for path, label in [
        (A_PLAN, "Step25A plan"),
        (A_LABELS, "Step25A label definition"),
        (A_PARTITIONS, "Step25A partition specification"),
        (A_ENDPOINTS, "Step25A endpoint specification"),
        (A_BOOTSTRAP, "Step25A bootstrap specification"),
        (A_METADATA, "Step25A metadata"),
        (A_CLAIM, "Step25A claim ceiling"),
    ]:
        require_file(path, label)

    plan = load_json(A_PLAN)
    meta = load_json(A_METADATA)

    if plan.get("target_axis") != "PC6":
        raise RuntimeError("Step25A target axis changed.")

    if plan.get("frozen_pc6_direction") != "high_in_tumor":
        raise RuntimeError("Step25A frozen PC6 direction changed.")

    if plan.get("primary_partition") != PRIMARY_PARTITION:
        raise RuntimeError("Step25A primary partition changed.")

    if plan.get("secondary_partition") != SECONDARY_PARTITION:
        raise RuntimeError("Step25A secondary partition changed.")

    label_def = pd.read_csv(A_LABELS, low_memory=False)

    positive = set(
        label_def.loc[
            pd.to_numeric(label_def["binary_tumor_target"], errors="coerce").eq(1),
            "histologic_label",
        ].astype(str)
    )

    negative = set(
        label_def.loc[
            pd.to_numeric(label_def["binary_tumor_target"], errors="coerce").eq(0),
            "histologic_label",
        ].astype(str)
    )

    if positive != TUMOR_LABELS:
        raise RuntimeError(
            f"Step25A tumor labels changed: {positive}"
        )

    if negative != NON_TUMOR_LABELS:
        raise RuntimeError(
            f"Step25A non-tumor labels changed: {negative}"
        )

    freeze_audit = meta.get("freeze_only_audit", {})

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

    bad = [x for x in forbidden_true if bool(freeze_audit.get(x, False))]

    if bad:
        raise RuntimeError(
            "Step25A freeze-only firewall violated:\n"
            + "\n".join(bad)
        )

    return {
        "plan": plan,
        "metadata": meta,
    }


# =============================================================================
# DISCOVER SLIDE UNIVERSE
# =============================================================================

def discover_slide_ids() -> List[str]:
    require_dir(
        LABEL_ROOT,
        "AGGC tile-label root",
    )

    slide_ids = set()

    for p in LABEL_ROOT.glob("Subset1_*_tile_labels.csv"):
        m = re.fullmatch(
            r"(Subset1_(?:Test|Train)_\d+)_tile_labels\.csv",
            p.name,
        )

        if m is not None:
            slide_ids.add(m.group(1))

    ordered = sorted(
        slide_ids,
        key=natural_key,
    )

    n_test = sum(x.startswith("Subset1_Test_") for x in ordered)
    n_train = sum(x.startswith("Subset1_Train_") for x in ordered)

    if n_test != EXPECTED_TEST_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_SLIDES} Test label slides; found={n_test}"
        )

    if n_train != EXPECTED_TRAIN_SLIDES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_SLIDES} Train label slides; found={n_train}"
        )

    return ordered


# =============================================================================
# PROCESS ONE SLIDE
# =============================================================================

def process_slide(
    slide_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    partition = parse_partition(slide_id)

    coords_path = exact_coords_path(slide_id)
    pca_path = exact_pca_path(slide_id)
    label_path = exact_label_path(slide_id)

    coords = load_coords(
        coords_path,
    )

    pca = load_pca(
        pca_path,
        expected_n=len(coords),
    )

    labels = pd.read_csv(
        label_path,
        low_memory=False,
    )

    colmap = resolve_label_columns(
        labels,
    )

    labels = labels.rename(
        columns={
            colmap["tile_index"]: "tile_index",
            colmap["x"]: "x",
            colmap["y"]: "y",
            colmap["label"]: "histologic_label_raw",
        }
    ).copy()

    for c in ["tile_index", "x", "y"]:
        labels[c] = pd.to_numeric(
            labels[c],
            errors="raise",
        ).astype(np.int64)

    if len(labels) != len(coords):
        raise RuntimeError(
            f"{slide_id}: label/coords row mismatch: "
            f"labels={len(labels):,}, coords={len(coords):,}"
        )

    if labels.duplicated("tile_index").any():
        raise RuntimeError(
            f"{slide_id}: duplicate tile_index in label file."
        )

    labels = labels.sort_values(
        "tile_index"
    ).reset_index(drop=True)

    expected_idx = np.arange(
        len(coords),
        dtype=np.int64,
    )

    observed_idx = labels[
        "tile_index"
    ].to_numpy(np.int64)

    exact_index = np.array_equal(
        observed_idx,
        expected_idx,
    )

    if not exact_index:
        raise RuntimeError(
            f"{slide_id}: tile_index is not exact 0...N-1."
        )

    observed_xy = labels[
        ["x", "y"]
    ].to_numpy(np.int64)

    exact_xy = np.array_equal(
        observed_xy,
        coords,
    )

    if not exact_xy:
        bad = np.where(
            np.any(observed_xy != coords, axis=1)
        )[0][:10]

        detail = "\n".join(
            f"tile={int(i)} labels={tuple(observed_xy[i])} coords={tuple(coords[i])}"
            for i in bad
        )

        raise RuntimeError(
            f"{slide_id}: exact coordinate reconciliation failed.\n{detail}"
        )

    pc6 = np.asarray(
        pca[:, PC6_INDEX_ZERO_BASED],
        dtype=np.float64,
    )

    if not np.isfinite(pc6).all():
        raise RuntimeError(
            f"{slide_id}: frozen PC6 contains non-finite values."
        )

    labels["histologic_label"] = labels[
        "histologic_label_raw"
    ].map(normalize_label)

    binary = np.full(
        len(labels),
        np.nan,
        dtype=np.float64,
    )

    tumor_mask = labels[
        "histologic_label"
    ].isin(TUMOR_LABELS)

    non_tumor_mask = labels[
        "histologic_label"
    ].isin(NON_TUMOR_LABELS)

    binary[tumor_mask.to_numpy()] = 1.0
    binary[non_tumor_mask.to_numpy()] = 0.0

    included = np.isfinite(binary)

    out = pd.DataFrame({
        "partition": partition,
        "formal_partition_role": (
            "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"
            if partition == PRIMARY_PARTITION
            else "SECONDARY_EXTERNAL_CHARACTERIZATION"
        ),
        "slide_id": slide_id,
        "tile_index": expected_idx,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "PC6": pc6,
        "histologic_label_raw": labels["histologic_label_raw"].astype(str),
        "histologic_label": labels["histologic_label"].astype(str),
        "binary_tumor_target": binary,
        "included_in_auc": included,
        "exclusion_reason": np.where(
            included,
            "",
            "unsupported_histologic_label",
        ),
    })

    n_tumor = int(
        np.sum(binary == 1)
    )

    n_non_tumor = int(
        np.sum(binary == 0)
    )

    n_excluded = int(
        np.sum(~included)
    )

    eligible = (
        n_tumor >= MIN_TUMOR_TILES
        and n_non_tumor >= MIN_NON_TUMOR_TILES
    )

    slide_row = {
        "partition": partition,
        "formal_partition_role": (
            "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"
            if partition == PRIMARY_PARTITION
            else "SECONDARY_EXTERNAL_CHARACTERIZATION"
        ),
        "slide_id": slide_id,
        "n_total_tiles": int(len(out)),
        "n_auc_included_tiles": int(included.sum()),
        "n_tumor_tiles": n_tumor,
        "n_non_tumor_tiles": n_non_tumor,
        "n_excluded_label_tiles": n_excluded,
        "min_tumor_tiles_required": MIN_TUMOR_TILES,
        "min_non_tumor_tiles_required": MIN_NON_TUMOR_TILES,
        "slide_eligible_for_auc": bool(eligible),
        "eligibility_reason": (
            "ELIGIBLE"
            if eligible
            else (
                "NO_TUMOR_TILES"
                if n_tumor < MIN_TUMOR_TILES
                and n_non_tumor >= MIN_NON_TUMOR_TILES
                else "NO_NON_TUMOR_TILES"
                if n_non_tumor < MIN_NON_TUMOR_TILES
                and n_tumor >= MIN_TUMOR_TILES
                else "NO_TUMOR_OR_NON_TUMOR_TILES"
            )
        ),
        "auc_calculated": False,
    }

    align_row = {
        "partition": partition,
        "slide_id": slide_id,
        "n_coords": int(len(coords)),
        "n_pca_rows": int(pca.shape[0]),
        "n_label_rows": int(len(labels)),
        "resolved_label_column": str(colmap["label"]),
        "tile_index_exact_0_to_n_minus_1": bool(exact_index),
        "xy_exact_coords_match": bool(exact_xy),
        "pca_contains_pc6": bool(pca.shape[1] > PC6_INDEX_ZERO_BASED),
        "pc6_all_finite": bool(np.isfinite(pc6).all()),
        "alignment_pass": True,
    }

    inventory_row = {
        "partition": partition,
        "slide_id": slide_id,
        "coords_path": str(coords_path),
        "pca_path": str(pca_path),
        "label_path": str(label_path),
        "coords_sha256": sha256_file(coords_path),
        "pca_sha256": sha256_file(pca_path),
        "label_sha256": sha256_file(label_path),
    }

    return out, slide_row, align_row, inventory_row


# =============================================================================
# LABEL AUDIT
# =============================================================================

def build_label_audit(
    tile_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = (
        tile_df.groupby(
            [
                "partition",
                "histologic_label",
                "included_in_auc",
            ],
            dropna=False,
        )
        .size()
        .rename("n_tiles")
        .reset_index()
    )

    for _, r in grouped.iterrows():
        label = str(r["histologic_label"])

        if label in TUMOR_LABELS:
            target = 1
            role = "TUMOR"
        elif label in NON_TUMOR_LABELS:
            target = 0
            role = "NON_TUMOR"
        else:
            target = np.nan
            role = "EXCLUDED_BY_LABEL_DEFINITION"

        rows.append({
            "partition": r["partition"],
            "histologic_label": label,
            "frozen_target": target,
            "frozen_role": role,
            "included_in_auc": bool(r["included_in_auc"]),
            "n_tiles": int(r["n_tiles"]),
        })

    return pd.DataFrame(rows)


# =============================================================================
# PARTITION SUMMARY
# =============================================================================

def build_partition_summary(
    slide_df: pd.DataFrame,
    tile_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for partition, role in [
        (PRIMARY_PARTITION, "PRIMARY_HELD_OUT_EXTERNAL_REPLICATION"),
        (SECONDARY_PARTITION, "SECONDARY_EXTERNAL_CHARACTERIZATION"),
    ]:
        s = slide_df.loc[
            slide_df["partition"].astype(str).eq(partition)
        ].copy()

        t = tile_df.loc[
            tile_df["partition"].astype(str).eq(partition)
        ].copy()

        rows.append({
            "partition": partition,
            "formal_role": role,
            "n_slides_total": int(len(s)),
            "n_slides_auc_eligible": int(
                s["slide_eligible_for_auc"].sum()
            ),
            "n_slides_auc_ineligible": int(
                (~s["slide_eligible_for_auc"]).sum()
            ),
            "n_total_tiles": int(len(t)),
            "n_auc_included_tiles": int(
                t["included_in_auc"].sum()
            ),
            "n_tumor_tiles": int(
                (t["binary_tumor_target"] == 1).sum()
            ),
            "n_non_tumor_tiles": int(
                (t["binary_tumor_target"] == 0).sum()
            ),
            "n_excluded_label_tiles": int(
                (~t["included_in_auc"]).sum()
            ),
            "auc_calculated": False,
            "bootstrap_calculated": False,
            "train_test_pooled": False,
        })

    return pd.DataFrame(rows)


# =============================================================================
# HASHES
# =============================================================================

def build_source_hashes() -> pd.DataFrame:
    rows = []

    for role, path in [
        ("step25a_plan", A_PLAN),
        ("step25a_label_definition", A_LABELS),
        ("step25a_partition_specification", A_PARTITIONS),
        ("step25a_endpoint_specification", A_ENDPOINTS),
        ("step25a_bootstrap_specification", A_BOOTSTRAP),
        ("step25a_metadata", A_METADATA),
        ("step25a_claim_ceiling", A_CLAIM),
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
            "Build the frozen AGGC PC6 tumor-localization Test/Train input "
            "without calculating ROC AUC."
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

    banner(
        "STEP 25B: BUILD AGGC PC6 EXTERNAL TUMOR-LOCALIZATION INPUT"
    )

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP25A]       {STEP25A}")
    print(f"[LABEL ROOT]    {LABEL_ROOT}")
    print(f"[PCA ROOT]      {PCA_ROOT}")
    print(f"[COORD ROOT]    {COORD_ROOT}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          MEASUREMENT / INPUT CONSTRUCTION / QC ONLY")
    print("[PRIMARY]       Subset1_Test")
    print("[SECONDARY]     Subset1_Train")
    print("[PC6]           FROZEN RAW pca_targets[:,5]")
    print("[TUMOR]         G3 / G4 / G5")
    print("[NON-TUMOR]     Normal / Stroma")
    print("[AUC]           NONE")
    print("[BOOTSTRAP]     NONE")
    print("[P-VALUES]      NONE")
    print("[THRESHOLD]     NONE")
    print("[PC RANKING]    NONE")
    print("[SIGN TUNING]   NONE")
    print("[PCA REFIT]     NONE")
    print("[TRAIN/TEST]    NEVER POOLED")

    prepare_output(
        args.output,
        args.overwrite,
    )

    require_dir(
        LABEL_ROOT,
        "AGGC tile-label root",
    )

    require_dir(
        PCA_ROOT,
        "AGGC PCA root",
    )

    require_dir(
        COORD_ROOT,
        "AGGC coordinate root",
    )

    # -------------------------------------------------------------------------
    # Validate Step25A.
    # -------------------------------------------------------------------------

    banner(
        "VALIDATE STEP25A FROZEN CONTRACT"
    )

    validate_step25a()

    print("Step25A target PC6                  : PASS")
    print("Step25A direction high_in_tumor     : PASS")
    print("Step25A Test primary role           : PASS")
    print("Step25A Train secondary role        : PASS")
    print("Step25A label definition            : PASS")
    print("Step25A freeze-only firewall        : PASS")

    # -------------------------------------------------------------------------
    # Discover fixed slide universe.
    # -------------------------------------------------------------------------

    banner(
        "DISCOVER FROZEN AGGC LABEL/PCA/COORD SLIDE UNIVERSE"
    )

    slide_ids = discover_slide_ids()

    print(f"Total slides discovered             : {len(slide_ids)}")
    print(
        f"Subset1_Test                        : "
        f"{sum(x.startswith('Subset1_Test_') for x in slide_ids)}"
    )
    print(
        f"Subset1_Train                       : "
        f"{sum(x.startswith('Subset1_Train_') for x in slide_ids)}"
    )

    # -------------------------------------------------------------------------
    # Process slides.
    # -------------------------------------------------------------------------

    banner(
        "EXACT TILE RECONCILIATION / BINARY TARGET CONSTRUCTION"
    )

    tile_frames: List[pd.DataFrame] = []
    slide_rows: List[Dict[str, Any]] = []
    alignment_rows: List[Dict[str, Any]] = []
    inventory_rows: List[Dict[str, Any]] = []

    for i, slide_id in enumerate(
        slide_ids,
        start=1,
    ):
        tile, slide_row, align_row, inventory_row = process_slide(
            slide_id
        )

        tile_frames.append(
            tile
        )

        slide_rows.append(
            slide_row
        )

        alignment_rows.append(
            align_row
        )

        inventory_rows.append(
            inventory_row
        )

        print(
            f"[{i:03d}/{len(slide_ids):03d}] "
            f"{slide_id} | "
            f"tiles={slide_row['n_total_tiles']:,} | "
            f"tumor={slide_row['n_tumor_tiles']:,} | "
            f"non_tumor={slide_row['n_non_tumor_tiles']:,} | "
            f"excluded={slide_row['n_excluded_label_tiles']:,} | "
            f"eligible={slide_row['slide_eligible_for_auc']}"
        )

    tile_df = pd.concat(
        tile_frames,
        ignore_index=True,
        sort=False,
    )

    slide_df = pd.DataFrame(
        slide_rows
    ).sort_values(
        ["partition", "slide_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    alignment_df = pd.DataFrame(
        alignment_rows
    ).sort_values(
        ["partition", "slide_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    inventory_df = pd.DataFrame(
        inventory_rows
    ).sort_values(
        ["partition", "slide_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Global identity audit.
    # -------------------------------------------------------------------------

    banner(
        "GLOBAL TILE IDENTITY / LABEL AUDIT"
    )

    if tile_df.duplicated(
        ["slide_id", "tile_index"]
    ).any():
        dup = tile_df.loc[
            tile_df.duplicated(
                ["slide_id", "tile_index"],
                keep=False,
            ),
            [
                "slide_id",
                "tile_index",
                "x",
                "y",
            ],
        ].head(20)

        raise RuntimeError(
            "Duplicate slide_id + tile_index detected in Step25B output.\n"
            + dup.to_string(index=False)
        )

    print(f"Total tile rows                     : {len(tile_df):,}")
    print(
        f"Unique slide/tile identities         : "
        f"{tile_df.drop_duplicates(['slide_id','tile_index']).shape[0]:,}"
    )
    print("Duplicate slide/tile identities      : 0")
    print(
        f"All alignment audits pass            : "
        f"{bool(alignment_df['alignment_pass'].all())}"
    )

    label_audit = build_label_audit(
        tile_df
    )

    print()
    print(label_audit.to_string(index=False))

    # -------------------------------------------------------------------------
    # Partition summary.
    # -------------------------------------------------------------------------

    partition_summary = build_partition_summary(
        slide_df,
        tile_df,
    )

    banner(
        "FROZEN TUMOR-LOCALIZATION MEASUREMENT COVERAGE"
    )

    print(
        partition_summary.to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    tile_df.to_csv(
        args.output / O_TILE,
        index=False,
        compression="gzip",
    )

    slide_df.to_csv(
        args.output / O_SLIDE,
        index=False,
    )

    partition_summary.to_csv(
        args.output / O_PARTITION,
        index=False,
    )

    alignment_df.to_csv(
        args.output / O_ALIGNMENT,
        index=False,
    )

    label_audit.to_csv(
        args.output / O_LABEL,
        index=False,
    )

    inventory_df.to_csv(
        args.output / O_INVENTORY,
        index=False,
    )

    build_source_hashes().to_csv(
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
        "role": "AGGC_PC6_EXTERNAL_TUMOR_LOCALIZATION_INPUT_CONSTRUCTION_ONLY",
        "upstream": {
            "step25a": str(STEP25A),
        },
        "target_axis": "PC6",
        "frozen_direction": "high_in_tumor",
        "pc6_source": "pca_targets[:,5]",
        "primary_partition": PRIMARY_PARTITION,
        "secondary_partition": SECONDARY_PARTITION,
        "tumor_labels": sorted(TUMOR_LABELS),
        "non_tumor_labels": sorted(NON_TUMOR_LABELS),
        "label_column_resolution": {
            "preferred_order": [
                "final_label",
                "histologic_class",
                "histologic_label",
                "gleason_label",
                "label",
                "class",
                "tissue_label"
            ],
            "authoritative_existing_aggc_label_column_expected": "final_label",
        },
        "eligibility": {
            "min_tumor_tiles": MIN_TUMOR_TILES,
            "min_non_tumor_tiles": MIN_NON_TUMOR_TILES,
            "outcome_based_exclusion": False,
        },
        "counts": {
            "test_slides_total": int(
                (slide_df["partition"] == PRIMARY_PARTITION).sum()
            ),
            "test_slides_eligible": int(
                slide_df.loc[
                    slide_df["partition"] == PRIMARY_PARTITION,
                    "slide_eligible_for_auc",
                ].sum()
            ),
            "train_slides_total": int(
                (slide_df["partition"] == SECONDARY_PARTITION).sum()
            ),
            "train_slides_eligible": int(
                slide_df.loc[
                    slide_df["partition"] == SECONDARY_PARTITION,
                    "slide_eligible_for_auc",
                ].sum()
            ),
            "tile_rows": int(len(tile_df)),
            "auc_included_tiles": int(
                tile_df["included_in_auc"].sum()
            ),
        },
        "firewall": {
            "auc_calculated": False,
            "bootstrap_calculated": False,
            "p_values_calculated": False,
            "fdr_calculated": False,
            "threshold_tuned": False,
            "pc_ranking": False,
            "pc_screening": False,
            "pc6_sign_tuned": False,
            "pca_refit": False,
            "pca_reprojection": False,
            "train_test_pooled": False,
            "approximate_coordinate_matching": False,
            "outcome_based_slide_exclusion": False,
        },
        "next_step": (
            "Step25C may calculate the Step25A-frozen per-slide ROC AUC and "
            "slide-bootstrap uncertainty from this Step25B input."
        ),
    }

    save_json(
        args.output / O_METADATA,
        metadata,
    )

    readme = f"""STEP 25B
AGGC PC6 EXTERNAL TUMOR-LOCALIZATION INPUT
==========================================

Version
-------
{VERSION}

Schema
------
{SCHEMA}

Role
----
MEASUREMENT / INPUT-CONSTRUCTION / QC ONLY.

No ROC AUC is calculated here.

Primary partition
-----------------
{PRIMARY_PARTITION}

Secondary partition
-------------------
{SECONDARY_PARTITION}

PC6
---
Source:
  exact frozen pca_targets[:,5]

Direction:
  high_in_tumor

Frozen label definition
-----------------------
Tumor:
  G3
  G4
  G5

Non-tumor:
  Normal
  Stroma

Unsupported labels:
  excluded by frozen label definition only

Exact tile identity
-------------------
tile_index = exact 0...N-1
x/y        = exact coords.npy
PC6        = frozen PCA target PC6

No fuzzy coordinate matching.
No PCA refit.
No PCA reprojection.
No sign flip.

Slide eligibility
-----------------
>= {MIN_TUMOR_TILES} tumor tile
>= {MIN_NON_TUMOR_TILES} non-tumor tile

Outputs
-------
Tile-level frozen input:
  {O_TILE}

Slide eligibility:
  {O_SLIDE}

Partition coverage:
  {O_PARTITION}

Exact alignment audit:
  {O_ALIGNMENT}

Label audit:
  {O_LABEL}

Inference firewall
------------------
ROC AUC                 : NO
bootstrap               : NO
p-values                : NO
FDR                     : NO
score threshold tuning  : NO
PC ranking              : NO
PC screening            : NO
PC6 sign tuning         : NO
PCA refit               : NO
PCA reprojection        : NO
Train/Test pooling      : NO
outcome-based exclusion : NO

Next
----
Step25C may perform the first formal AGGC PC6 tumor-localization AUC evaluation.
"""

    (args.output / O_README).write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final firewall.
    # -------------------------------------------------------------------------

    banner(
        "STEP25B MEASUREMENT-ONLY FIREWALL AUDIT"
    )

    audit = [
        ("Step25A frozen contract inherited", "YES"),
        ("Exact coords/PCA/labels reconciliation", "YES"),
        ("Raw frozen PC6 retained", "YES"),
        ("Frozen tumor definition retained", "YES"),
        ("Frozen non-tumor definition retained", "YES"),
        ("Test primary / Train secondary retained", "YES"),
        ("Slide eligibility constructed", "YES"),
        ("ROC AUC calculated", "NO"),
        ("Bootstrap calculated", "NO"),
        ("P-values calculated", "NO"),
        ("FDR calculated", "NO"),
        ("Score threshold tuned", "NO"),
        ("PC ranking", "NO"),
        ("PC screening", "NO"),
        ("PC6 sign tuning", "NO"),
        ("PCA refit", "NO"),
        ("PCA reprojection", "NO"),
        ("Train/Test pooled", "NO"),
        ("Approximate coordinate matching", "NO"),
        ("Outcome-based slide exclusion", "NO"),
    ]

    for item, status in audit:
        print(f"{item:<66}: {status}")

    print()
    print("[OK] Saved:")

    for name in [
        O_TILE,
        O_SLIDE,
        O_PARTITION,
        O_ALIGNMENT,
        O_LABEL,
        O_INVENTORY,
        O_HASHES,
        O_METADATA,
        O_README,
    ]:
        path = args.output / name

        if not path.exists():
            raise RuntimeError(
                f"Expected Step25B output missing: {path}"
            )

        print(f"  {path}")

    print()
    print(
        "[DONE] Step25B AGGC PC6 external tumor-localization input construction complete."
    )
    print(
        "[NEXT] Step25C may calculate the frozen Test-primary / "
        "Train-secondary per-slide ROC AUC evaluation."
    )


if __name__ == "__main__":
    main()
