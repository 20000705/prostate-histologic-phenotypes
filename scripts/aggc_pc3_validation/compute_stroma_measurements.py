#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17b_compute_aggc_pc3_external_stroma_measurements.py

STEP 17B — MEASUREMENT ONLY

Consumes the frozen Step-17a AGGC specification and computes tile-level
annotation measurements for the frozen Train-validation and Test-replication
cohorts.

STRICTLY FORBIDDEN HERE
-----------------------
- redefine cohort / labels / masks / geometry / thresholds / endpoint
- refit PCA or flip PC3
- screen PCs or analyze PC6
- calculate Spearman/Pearson/AUC
- calculate p-values / bootstrap / inferential statistics
- exclude slides because of observed PC3 behavior
- treat unannotated pixels as non-Stroma

PRIMARY FROZEN MEASUREMENT
--------------------------
annotation_union = OR(Normal, Stroma, G3, G4, G5) for masks that exist
labeled_fraction = annotation_union_pixels / tile_pixels

stroma_fraction_among_annotated_pixels =
    Stroma_positive_pixels / annotation_union_pixels

Unannotated pixels are excluded from the denominator.

Missing optional class masks are recorded as unavailable and their class
measurements are NaN; they are not interpreted as biological absence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import traceback
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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VERSION = "1.0.0"
SCHEMA_VERSION = "17b-aggc-pc3-external-stroma-measurement-schema-1.0"

# =============================================================================
# PATHS
# =============================================================================

MEDIA_BASE = REPO_ROOT

AGGC_BASE = CONFIG_AGGC_ROOT

STEP17A = RESULTS_ROOT / "17a_aggc_pc3_external_stroma_specification"

TRAIN_COHORT = STEP17A / "aggc_pc3_train_validation_cohort.csv"
TEST_COHORT = STEP17A / "aggc_pc3_test_replication_cohort.csv"
PLAN_JSON = STEP17A / "aggc_pc3_external_stroma_analysis_plan.json"
LABEL_JSON = STEP17A / "aggc_pc3_external_stroma_label_specification.json"
GEOM_JSON = STEP17A / "aggc_pc3_external_stroma_geometry_specification.json"
META_JSON = STEP17A / "analysis_metadata.json"
SOURCE_HASHES = STEP17A / "source_hashes.csv"

TRAIN_MASK_MANIFEST = (
    AGGC_BASE
    / "step17_preflight_aggc_subset1_audit"
    / "aggc_subset1_mask_manifest.csv"
)

TEST_MASK_MANIFEST = (
    AGGC_BASE
    / "step17_preflight_aggc_subset1_test_audit"
    / "aggc_subset1_test_mask_manifest.csv"
)

OUT_DIR = RESULTS_ROOT / "17b_aggc_pc3_external_stroma_measurements"
PER_TILE_DIR = OUT_DIR / "per_tile"

OUT_MANIFEST = OUT_DIR / "aggc_pc3_measurement_manifest.csv"
OUT_SLIDE_QC = OUT_DIR / "aggc_pc3_slide_measurement_qc.csv"
OUT_FAILURES = OUT_DIR / "aggc_pc3_measurement_failures.csv"
OUT_SOURCE_AUDIT = OUT_DIR / "frozen_source_integrity_audit.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"

OVERWRITE = False

PARTITIONS = {
    "train_validation": TRAIN_COHORT,
    "test_replication": TEST_COHORT,
}

EXPECTED_LABELS = ["Normal", "Stroma", "G3", "G4", "G5"]


# =============================================================================
# HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def require_dir(path: Path, label: str):
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def prepare_output():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        if not OVERWRITE:
            raise RuntimeError(
                f"17b output already exists: {OUT_DIR}\n"
                "Do not overwrite a frozen measurement run casually."
            )
        for p in OUT_DIR.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    PER_TILE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"true", "1", "1.0", "yes"}:
        return True
    if s in {"false", "0", "0.0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {x!r}")


def safe_fraction(num, den) -> float:
    if den is None or den <= 0 or num is None:
        return np.nan
    return float(num / den)


def read_frozen_design():
    plan = load_json(PLAN_JSON)
    label_spec = load_json(LABEL_JSON)
    geom = load_json(GEOM_JSON)
    meta = load_json(META_JSON)

    if meta.get("role") != "freeze_only":
        raise RuntimeError("Step17a metadata does not say role=freeze_only.")
    if meta.get("pc_values_read") is not False:
        raise RuntimeError("Step17a metadata says PC values were read.")
    if meta.get("aggc_pc3_association_calculated") is not False:
        raise RuntimeError("Step17a metadata says AGGC association was calculated.")
    if meta.get("pca_refit") is not False:
        raise RuntimeError("Step17a metadata says PCA was refit.")
    if meta.get("pc_screening") is not False:
        raise RuntimeError("Step17a metadata says PC screening occurred.")

    latent = plan["latent_representation"]
    tile_qc = plan["primary_tile_qc"]
    slide_qc = plan["primary_slide_qc"]

    frozen = {
        "target_pc": latent["target"],
        "pc_index": int(latent["zero_based_component_index"]),
        "patch_size": int(geom["patch_size_pixels"]),
        "scale_x": float(geom["mask_to_wsi_scale_x"]),
        "scale_y": float(geom["mask_to_wsi_scale_y"]),
        "offset_x": int(geom["mask_to_wsi_offset_x"]),
        "offset_y": int(geom["mask_to_wsi_offset_y"]),
        "resampling": geom["resampling_for_primary_measurement"],
        "endpoint": plan["primary_endpoint"],
        "labeled_threshold": float(tile_qc["minimum_labeled_fraction"]),
        "min_tiles_per_slide": int(slide_qc["minimum_qc_passing_tiles"]),
        "labels": list(label_spec["official_labels"]),
        "primary_label": label_spec["primary_label"],
        "binary_rule": label_spec["mask_binary_definition"],
        "foreground_rule": label_spec["mask_foreground_definition"],
    }

    # Implementation compatibility checks only; these do not redefine 17a.
    if frozen["target_pc"] != "PC3":
        raise RuntimeError(f"Unexpected frozen target: {frozen['target_pc']}")
    if frozen["endpoint"] != "stroma_fraction_among_annotated_pixels":
        raise RuntimeError(f"Unexpected frozen endpoint: {frozen['endpoint']}")
    if frozen["labels"] != EXPECTED_LABELS:
        raise RuntimeError(f"Unexpected frozen labels: {frozen['labels']}")
    if frozen["primary_label"] != "Stroma":
        raise RuntimeError("Frozen primary label is not Stroma.")
    if frozen["scale_x"] != 1.0 or frozen["scale_y"] != 1.0:
        raise RuntimeError("17b does not permit non-1:1 mask scaling.")
    if frozen["offset_x"] != 0 or frozen["offset_y"] != 0:
        raise RuntimeError("17b does not permit mask offsets.")
    if str(frozen["resampling"]).lower() != "none":
        raise RuntimeError("17b does not permit primary mask resampling.")
    if frozen["binary_rule"] != "Pillow mode == '1'":
        raise RuntimeError("Frozen mask binary rule is not Pillow mode == '1'.")
    if frozen["foreground_rule"] != "Pillow Boolean True":
        raise RuntimeError("Frozen foreground rule is not Pillow Boolean True.")

    return meta, frozen


def verify_frozen_mask_manifests() -> pd.DataFrame:
    hashes = pd.read_csv(SOURCE_HASHES, low_memory=False)
    if not {"path", "sha256"}.issubset(hashes.columns):
        raise RuntimeError("Step17a source_hashes.csv lacks path/sha256.")

    frozen_hashes = {
        str(Path(r["path"])): str(r["sha256"])
        for _, r in hashes.iterrows()
    }

    rows = []
    for path, role in [
        (TRAIN_MASK_MANIFEST, "train_mask_manifest"),
        (TEST_MASK_MANIFEST, "test_mask_manifest"),
    ]:
        require_file(path, role)
        expected = frozen_hashes.get(str(path))
        actual = sha256_file(path)
        match = expected is not None and expected == actual

        rows.append({
            "role": role,
            "path": str(path),
            "frozen_hash_present": expected is not None,
            "frozen_sha256": expected,
            "actual_sha256": actual,
            "hash_match": match,
        })

        if expected is None:
            raise RuntimeError(f"{path} was not hashed by Step17a.")
        if not match:
            raise RuntimeError(
                f"Frozen source changed after Step17a: {path}"
            )

    return pd.DataFrame(rows)


def load_mask_map(path: Path) -> Dict[str, Dict[str, Optional[Path]]]:
    df = pd.read_csv(path, low_memory=False)
    need = {"slide_id", "label", "mask_exists", "mask_path"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"Bad mask manifest: {path}")

    result: Dict[str, Dict[str, Optional[Path]]] = {}

    for _, row in df.iterrows():
        sid = str(row["slide_id"])
        label = str(row["label"])
        exists = to_bool(row["mask_exists"])
        result.setdefault(sid, {})[label] = (
            Path(str(row["mask_path"])) if exists else None
        )

    return result


def open_mask(
    path: Path,
    expected_size: Tuple[int, int],
    slide_id: str,
    label: str,
) -> Image.Image:
    require_file(path, f"{slide_id} {label} mask")
    img = Image.open(path)

    if img.mode != "1":
        img.close()
        raise RuntimeError(
            f"{slide_id} {label}: expected mode '1', found {img.mode}"
        )

    if img.size != expected_size:
        size = img.size
        img.close()
        raise RuntimeError(
            f"{slide_id} {label}: geometry mismatch "
            f"expected={expected_size}, found={size}; no correction permitted."
        )

    return img


def crop_bool(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    crop = img.crop(box)
    arr = np.asarray(crop, dtype=np.bool_)
    crop.close()

    if arr.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected crop shape={arr.shape}, expected={expected_shape}"
        )
    return arr


# =============================================================================
# SLIDE MEASUREMENT
# =============================================================================

def measure_slide(
    row: pd.Series,
    partition: str,
    frozen: dict,
    mask_map: Dict[str, Dict[str, Optional[Path]]],
):
    sid = str(row["slide_id"])

    if not to_bool(row["formal_step17a_include"]):
        raise RuntimeError(f"{sid} is not Step17a-included.")

    coords_path = Path(str(row["coords_path"]))
    pca_path = Path(str(row["pca_path"]))
    wsi_path = Path(str(row["wsi_path"]))

    require_file(coords_path, f"{sid} coords")
    require_file(pca_path, f"{sid} PCA")
    require_file(wsi_path, f"{sid} WSI")

    coords = np.load(coords_path, mmap_mode="r")
    pca = np.load(pca_path, mmap_mode="r")

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise RuntimeError(f"{sid}: invalid coords shape={coords.shape}")
    if pca.ndim != 2:
        raise RuntimeError(f"{sid}: invalid PCA shape={pca.shape}")
    if coords.shape[0] != pca.shape[0]:
        raise RuntimeError(
            f"{sid}: coords/PCA row mismatch "
            f"{coords.shape[0]} vs {pca.shape[0]}"
        )

    pc_index = frozen["pc_index"]
    if pca.shape[1] <= pc_index:
        raise RuntimeError(
            f"{sid}: PCA lacks frozen PC index {pc_index}; shape={pca.shape}"
        )

    patch = frozen["patch_size"]
    wsi_w = int(float(row["wsi_width"]))
    wsi_h = int(float(row["wsi_height"]))
    expected_size = (wsi_w, wsi_h)

    slide_masks = mask_map.get(sid)
    if slide_masks is None:
        raise RuntimeError(f"{sid}: no frozen mask-manifest entry.")

    mask_images: Dict[str, Optional[Image.Image]] = {}
    available_labels: List[str] = []

    failures: List[Dict] = []

    try:
        for label in frozen["labels"]:
            path = slide_masks.get(label)
            if path is None:
                mask_images[label] = None
            else:
                mask_images[label] = open_mask(
                    path, expected_size, sid, label
                )
                available_labels.append(label)

        if "Stroma" not in available_labels:
            raise RuntimeError(f"{sid}: frozen included slide lacks Stroma mask.")

        rows = []
        n_tiles = int(coords.shape[0])
        expected_crop_shape = (patch, patch)
        tile_pixels = patch * patch

        for idx in range(n_tiles):
            x = int(coords[idx, 0])
            y = int(coords[idx, 1])
            pc3 = float(pca[idx, pc_index])

            x2 = x + patch
            y2 = y + patch

            measurement_ok = True
            failure_reason = ""

            counts: Dict[str, Optional[int]] = {
                label: None for label in frozen["labels"]
            }

            union_px = 0
            overlap_px = 0
            max_active = 0

            if x < 0 or y < 0 or x2 > wsi_w or y2 > wsi_h:
                measurement_ok = False
                failure_reason = "tile_outside_frozen_wsi_geometry"

            if measurement_ok:
                try:
                    active = np.zeros(
                        expected_crop_shape,
                        dtype=np.uint8,
                    )

                    for label in frozen["labels"]:
                        img = mask_images[label]
                        if img is None:
                            continue

                        arr = crop_bool(
                            img,
                            (x, y, x2, y2),
                            expected_crop_shape,
                        )

                        counts[label] = int(arr.sum())
                        active += arr.astype(np.uint8, copy=False)

                    union_px = int(np.count_nonzero(active >= 1))
                    overlap_px = int(np.count_nonzero(active >= 2))
                    max_active = int(active.max())

                except Exception as exc:
                    measurement_ok = False
                    failure_reason = "mask_crop_or_measurement_failure"

                    failures.append({
                        "partition": partition,
                        "slide_id": sid,
                        "tile_index": idx,
                        "x": x,
                        "y": y,
                        "failure_stage": "tile_measurement",
                        "failure_reason": failure_reason,
                        "exception": repr(exc),
                    })

            if (
                not measurement_ok
                and failure_reason == "tile_outside_frozen_wsi_geometry"
            ):
                failures.append({
                    "partition": partition,
                    "slide_id": sid,
                    "tile_index": idx,
                    "x": x,
                    "y": y,
                    "failure_stage": "tile_geometry",
                    "failure_reason": failure_reason,
                    "exception": "",
                })

            labeled_fraction = (
                float(union_px / tile_pixels)
                if measurement_ok
                else np.nan
            )

            class_fraction = {
                label: (
                    safe_fraction(counts[label], union_px)
                    if measurement_ok
                    else np.nan
                )
                for label in frozen["labels"]
            }

            stroma_fraction = class_fraction["Stroma"]

            overlap_fraction_tile = (
                float(overlap_px / tile_pixels)
                if measurement_ok
                else np.nan
            )

            overlap_fraction_annotated = (
                safe_fraction(overlap_px, union_px)
                if measurement_ok
                else np.nan
            )

            pc3_finite = bool(np.isfinite(pc3))

            frozen_labeled_qc = bool(
                measurement_ok
                and np.isfinite(labeled_fraction)
                and labeled_fraction >= frozen["labeled_threshold"]
            )

            primary_qc = bool(
                frozen_labeled_qc
                and pc3_finite
                and np.isfinite(stroma_fraction)
            )

            rows.append({
                "partition": partition,
                "slide_id": sid,
                "tile_index": idx,
                "x": x,
                "y": y,
                "patch_size": patch,
                "pc3": pc3,
                "pc3_finite": pc3_finite,

                "normal_mask_available": mask_images["Normal"] is not None,
                "stroma_mask_available": mask_images["Stroma"] is not None,
                "g3_mask_available": mask_images["G3"] is not None,
                "g4_mask_available": mask_images["G4"] is not None,
                "g5_mask_available": mask_images["G5"] is not None,

                "normal_positive_pixels": counts["Normal"],
                "stroma_positive_pixels": counts["Stroma"],
                "g3_positive_pixels": counts["G3"],
                "g4_positive_pixels": counts["G4"],
                "g5_positive_pixels": counts["G5"],

                "annotation_union_pixels": (
                    union_px if measurement_ok else np.nan
                ),
                "tile_pixels": tile_pixels,
                "labeled_fraction": labeled_fraction,

                "normal_fraction_among_annotated_pixels":
                    class_fraction["Normal"],
                "stroma_fraction_among_annotated_pixels":
                    stroma_fraction,
                "g3_fraction_among_annotated_pixels":
                    class_fraction["G3"],
                "g4_fraction_among_annotated_pixels":
                    class_fraction["G4"],
                "g5_fraction_among_annotated_pixels":
                    class_fraction["G5"],

                "overlap_positive_pixels": (
                    overlap_px if measurement_ok else np.nan
                ),
                "overlap_fraction_of_tile": overlap_fraction_tile,
                "overlap_fraction_among_annotated_pixels":
                    overlap_fraction_annotated,
                "max_active_annotation_labels": (
                    max_active if measurement_ok else np.nan
                ),
                "overlap_free_tile": (
                    bool(overlap_px == 0) if measurement_ok else False
                ),

                "measurement_ok": measurement_ok,
                "measurement_failure_reason": failure_reason,

                "frozen_labeled_fraction_threshold":
                    frozen["labeled_threshold"],
                "frozen_labeled_qc_pass": frozen_labeled_qc,
                "primary_qc_pass": primary_qc,
            })

        tile_df = pd.DataFrame(rows)

        out_path = (
            PER_TILE_DIR
            / f"{sid}_pc3_stroma_measurements.csv.gz"
        )

        tile_df.to_csv(
            out_path,
            index=False,
            compression="gzip",
        )

        n_measure_ok = int(tile_df["measurement_ok"].sum())
        n_primary = int(tile_df["primary_qc_pass"].sum())
        n_overlap_free_primary = int(
            (
                tile_df["primary_qc_pass"]
                & tile_df["overlap_free_tile"]
            ).sum()
        )

        slide_qc = {
            "partition": partition,
            "slide_id": sid,
            "n_tiles_total": n_tiles,
            "n_tiles_finite_pc3": int(tile_df["pc3_finite"].sum()),
            "n_tiles_measurement_ok": n_measure_ok,
            "n_tiles_frozen_labeled_qc_pass":
                int(tile_df["frozen_labeled_qc_pass"].sum()),
            "n_tiles_primary_qc_pass": n_primary,
            "n_tiles_overlap_free_primary_qc": n_overlap_free_primary,

            "measurement_ok_fraction":
                n_measure_ok / n_tiles if n_tiles else np.nan,
            "primary_qc_retention_fraction":
                n_primary / n_tiles if n_tiles else np.nan,
            "overlap_free_fraction_among_primary_qc":
                (
                    n_overlap_free_primary / n_primary
                    if n_primary
                    else np.nan
                ),

            "frozen_labeled_fraction_threshold":
                frozen["labeled_threshold"],
            "frozen_minimum_tiles_per_slide":
                frozen["min_tiles_per_slide"],
            "slide_meets_frozen_minimum_tile_count":
                bool(n_primary >= frozen["min_tiles_per_slide"]),

            "available_annotation_labels":
                "|".join(available_labels),
            "n_available_annotation_labels":
                len(available_labels),

            "n_failure_records": len(failures),
            "per_tile_measurement_path": str(out_path),

            # Descriptive measurement summaries only; no PC3 association.
            "median_labeled_fraction_measurement_ok":
                (
                    float(
                        tile_df.loc[
                            tile_df["measurement_ok"],
                            "labeled_fraction",
                        ].median()
                    )
                    if n_measure_ok
                    else np.nan
                ),

            "median_stroma_fraction_primary_qc":
                (
                    float(
                        tile_df.loc[
                            tile_df["primary_qc_pass"],
                            "stroma_fraction_among_annotated_pixels",
                        ].median()
                    )
                    if n_primary
                    else np.nan
                ),
        }

        return slide_qc, failures

    finally:
        for img in mask_images.values():
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 124)
    print("STEP 17B: AGGC PC3 EXTERNAL STROMA MEASUREMENTS")
    print("=" * 124)

    print("[VERSION]    ", VERSION)
    print("[SCHEMA]     ", SCHEMA_VERSION)
    print("[STEP 17A]   ", STEP17A)
    print("[OUTPUT]     ", OUT_DIR)
    print("[OVERWRITE]  ", OVERWRITE)
    print()
    print("[ROLE]       MEASUREMENT ONLY")
    print("[ASSOCIATION] NONE")
    print("[INFERENCE]   NONE")
    print("[PCA REFIT]   NONE")
    print("[PC SCREEN]   NONE")
    print("[PC6]         NONE")

    require_dir(STEP17A, "Step17a directory")

    for path, label in [
        (TRAIN_COHORT, "Step17a Train cohort"),
        (TEST_COHORT, "Step17a Test cohort"),
        (PLAN_JSON, "Step17a analysis plan"),
        (LABEL_JSON, "Step17a label specification"),
        (GEOM_JSON, "Step17a geometry specification"),
        (META_JSON, "Step17a metadata"),
        (SOURCE_HASHES, "Step17a source hashes"),
    ]:
        require_file(path, label)

    step17a_meta, frozen = read_frozen_design()

    print()
    print("[FROZEN DESIGN READ]")
    print(f"  Target PC                  : {frozen['target_pc']}")
    print(f"  PC index                   : {frozen['pc_index']}")
    print(f"  Patch size                 : {frozen['patch_size']}")
    print(f"  Primary endpoint           : {frozen['endpoint']}")
    print(f"  Labeled threshold          : {frozen['labeled_threshold']}")
    print(f"  Min tiles / slide          : {frozen['min_tiles_per_slide']}")
    print(f"  Geometry scale             : ({frozen['scale_x']}, {frozen['scale_y']})")
    print(f"  Geometry offset            : ({frozen['offset_x']}, {frozen['offset_y']})")
    print(f"  Resampling                 : {frozen['resampling']}")

    source_audit = verify_frozen_mask_manifests()

    prepare_output()
    source_audit.to_csv(OUT_SOURCE_AUDIT, index=False)

    mask_maps = {
        "train_validation": load_mask_map(TRAIN_MASK_MANIFEST),
        "test_replication": load_mask_map(TEST_MASK_MANIFEST),
    }

    slide_qc_rows = []
    failure_rows = []
    manifest_rows = []

    total_frozen = 0

    for partition, cohort_path in PARTITIONS.items():
        cohort = pd.read_csv(cohort_path, low_memory=False)

        if "formal_step17a_include" not in cohort.columns:
            raise RuntimeError(
                f"{partition}: missing formal_step17a_include."
            )

        included = cohort.loc[
            cohort["formal_step17a_include"].map(to_bool)
        ].copy()

        expected_n = (
            int(step17a_meta["train_validation_n"])
            if partition == "train_validation"
            else int(step17a_meta["test_replication_n"])
        )

        if len(included) != expected_n:
            raise RuntimeError(
                f"{partition}: frozen count changed; "
                f"expected={expected_n}, found={len(included)}"
            )

        total_frozen += len(included)

        print()
        print(f"[PARTITION] {partition} | slides={len(included)}")

        for i, (_, row) in enumerate(included.iterrows(), 1):
            sid = str(row["slide_id"])
            print(
                f"[RUN] {partition} [{i:03d}/{len(included):03d}] {sid}"
            )

            try:
                qc, failures = measure_slide(
                    row=row,
                    partition=partition,
                    frozen=frozen,
                    mask_map=mask_maps[partition],
                )

                slide_qc_rows.append(qc)
                failure_rows.extend(failures)

                manifest_rows.append({
                    "partition": partition,
                    "slide_id": sid,
                    "status": "measured",
                    "per_tile_measurement_path":
                        qc["per_tile_measurement_path"],
                    "n_tiles_total": qc["n_tiles_total"],
                    "n_tiles_measurement_ok":
                        qc["n_tiles_measurement_ok"],
                    "n_tiles_primary_qc_pass":
                        qc["n_tiles_primary_qc_pass"],
                    "slide_meets_frozen_minimum_tile_count":
                        qc["slide_meets_frozen_minimum_tile_count"],
                })

                print(
                    f"[OK] {sid} | "
                    f"tiles={qc['n_tiles_total']:,} | "
                    f"measurement_ok={qc['n_tiles_measurement_ok']:,} | "
                    f"primary_qc={qc['n_tiles_primary_qc_pass']:,} | "
                    f"retention={qc['primary_qc_retention_fraction']:.4f}"
                )

            except Exception as exc:
                failure_rows.append({
                    "partition": partition,
                    "slide_id": sid,
                    "tile_index": np.nan,
                    "x": np.nan,
                    "y": np.nan,
                    "failure_stage": "slide_measurement",
                    "failure_reason": "slide_measurement_failed",
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                })

                manifest_rows.append({
                    "partition": partition,
                    "slide_id": sid,
                    "status": "failed",
                    "per_tile_measurement_path": "",
                    "n_tiles_total": np.nan,
                    "n_tiles_measurement_ok": np.nan,
                    "n_tiles_primary_qc_pass": np.nan,
                    "slide_meets_frozen_minimum_tile_count": False,
                })

                print(f"[ERROR] {sid}: {exc}")

    slide_qc_df = pd.DataFrame(slide_qc_rows)
    failure_df = pd.DataFrame(failure_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    if failure_df.empty:
        failure_df = pd.DataFrame(columns=[
            "partition",
            "slide_id",
            "tile_index",
            "x",
            "y",
            "failure_stage",
            "failure_reason",
            "exception",
            "traceback",
        ])

    slide_qc_df.to_csv(OUT_SLIDE_QC, index=False)
    failure_df.to_csv(OUT_FAILURES, index=False)
    manifest_df.to_csv(OUT_MANIFEST, index=False)

    n_measured = int(manifest_df["status"].eq("measured").sum())
    n_failed = int(manifest_df["status"].eq("failed").sum())

    if len(manifest_df) != total_frozen:
        raise RuntimeError(
            "Measurement manifest rows != frozen included slide count."
        )

    metadata = {
        "step": "17b",
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "role": "measurement_only",
        "step17a_directory": str(STEP17A),
        "frozen_design": frozen,
        "formal_frozen_slides_requested": total_frozen,
        "slides_measured": n_measured,
        "slides_failed": n_failed,
        "failure_records": int(len(failure_df)),
        "association_calculated": False,
        "spearman_calculated": False,
        "pearson_calculated": False,
        "auc_calculated": False,
        "p_values_calculated": False,
        "bootstrap_performed": False,
        "extreme_group_inference_performed": False,
        "pca_refit": False,
        "pc_screening": False,
        "pc6_analysis": False,
        "mask_redefinition": False,
        "geometry_redefinition": False,
        "threshold_redefinition": False,
        "label_redefinition": False,
        "unannotated_pixels_treated_as_non_stroma": False,
        "source_integrity_verified":
            bool(source_audit["hash_match"].all()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    write_json(OUT_METADATA, metadata)

    OUT_README.write_text(
        f"""STEP 17B — AGGC PC3 EXTERNAL STROMA MEASUREMENTS
=================================================

Version: {VERSION}
Schema: {SCHEMA_VERSION}

ROLE
----
MEASUREMENT ONLY.

All cohort, PC, mask, label, geometry, endpoint, and threshold definitions
come from frozen Step 17a.

FROZEN PRIMARY ENDPOINT
-----------------------
{frozen['endpoint']}

FROZEN TILE QC
--------------
labeled_fraction >= {frozen['labeled_threshold']}

FROZEN SLIDE QC
---------------
minimum primary-QC tiles = {frozen['min_tiles_per_slide']}

ANNOTATION SEMANTICS
--------------------
annotation_union = OR of existing official Normal/Stroma/G3/G4/G5 masks

stroma_fraction_among_annotated_pixels =
    Stroma-positive pixels / annotation-union pixels

Unannotated pixels are excluded from the biological denominator.

Missing optional masks are recorded as unavailable and corresponding
class-specific values are NaN.

GEOMETRY
--------
patch = {frozen['patch_size']} x {frozen['patch_size']}
scale = ({frozen['scale_x']}, {frozen['scale_y']})
offset = ({frozen['offset_x']}, {frozen['offset_y']})
resampling = {frozen['resampling']}

No silent scaling, shifting, padding, clipping, or geometry repair.

OUTPUTS
-------
Per-tile:
    {PER_TILE_DIR}

Slide measurement QC:
    {OUT_SLIDE_QC}

Measurement manifest:
    {OUT_MANIFEST}

Failures:
    {OUT_FAILURES}

Source integrity:
    {OUT_SOURCE_AUDIT}

NOT PERFORMED
-------------
No Spearman.
No Pearson.
No AUC.
No p-values.
No bootstrap.
No high-vs-low inference.
No PC screening.
No PC6.
No PCA refit.
No threshold tuning.
No endpoint tuning.

NEXT
----
Step 17c may perform only the already-frozen association/inference using
these measurements.
""",
        encoding="utf-8",
    )

    print()
    print("=" * 124)
    print("FINAL MEASUREMENT SUMMARY")
    print("=" * 124)

    print(f"Frozen included slides requested : {total_frozen}")
    print(f"Slides measured                  : {n_measured}")
    print(f"Slides failed                    : {n_failed}")
    print(f"Failure records                  : {len(failure_df)}")

    if not slide_qc_df.empty:
        for partition in ["train_validation", "test_replication"]:
            sub = slide_qc_df[
                slide_qc_df["partition"].eq(partition)
            ]

            if len(sub):
                print()
                print(f"[{partition}]")
                print(f"Slides measured                  : {len(sub)}")
                print(
                    f"Total tiles                      : "
                    f"{int(sub['n_tiles_total'].sum()):,}"
                )
                print(
                    f"Measurement-ok tiles             : "
                    f"{int(sub['n_tiles_measurement_ok'].sum()):,}"
                )
                print(
                    f"Primary-QC tiles                 : "
                    f"{int(sub['n_tiles_primary_qc_pass'].sum()):,}"
                )
                print(
                    f"Median primary-QC retention      : "
                    f"{sub['primary_qc_retention_fraction'].median():.4f}"
                )
                print(
                    f"Min primary-QC retention         : "
                    f"{sub['primary_qc_retention_fraction'].min():.4f}"
                )
                print(
                    f"Slides meeting frozen min tiles  : "
                    f"{int(sub['slide_meets_frozen_minimum_tile_count'].sum())}"
                    f"/{len(sub)}"
                )

    print()
    print("=" * 124)
    print("MEASUREMENT-ONLY AUDIT")
    print("=" * 124)
    print("Association calculated            : NO")
    print("Spearman calculated               : NO")
    print("Pearson calculated                : NO")
    print("AUC calculated                    : NO")
    print("P-values calculated               : NO")
    print("Bootstrap performed               : NO")
    print("PCA refit                         : NO")
    print("PC screening                      : NO")
    print("PC6 analysis                      : NO")
    print("Mask redefinition                 : NO")
    print("Geometry redefinition             : NO")
    print("Threshold redefinition            : NO")
    print("Label redefinition                : NO")
    print("Unannotated -> non-Stroma         : NO")

    print()
    print("[OK] Saved:")
    for p in [
        OUT_MANIFEST,
        OUT_SLIDE_QC,
        OUT_FAILURES,
        OUT_SOURCE_AUDIT,
        OUT_METADATA,
        OUT_README,
    ]:
        print(" ", p)

    print()
    print(
        "[DONE] Step-17b AGGC PC3 external Stroma measurements complete."
    )


if __name__ == "__main__":
    main()
