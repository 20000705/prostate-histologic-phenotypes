#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18b_compute_aggc_pc3_gleason_histologic_state_measurements.py

STEP 18B: AGGC PC3 GLEASON / HISTOLOGIC-STATE MEASUREMENTS
VERSION 1.0.2 FINAL

PURPOSE
-------
Measurement-only implementation of the design frozen in Step 18a v1.1.0.

This FINAL v1.0.2 intentionally reuses the memory-safe AGGC implementation
pattern that was already validated in the user's previous code:

    PIL.Image.open(mask)
        -> tile-sized image.crop(...)
        -> np.asarray(crop, dtype=bool)

It DOES NOT:
    - decode full-resolution masks into NumPy
    - build integral images
    - call image.load() on the large mask
    - retain all slide DataFrames in memory

The scientific/statistical design remains fully controlled by Step 18a.

STEP-18A AUTHORITY
------------------
The script reads and enforces:

    results/18a_aggc_pc3_gleason_histologic_state_specification

Required Step18a version:
    1.1.0

Frozen rules consumed:
    - PC3 zero-based component index
    - minimum class fraction per tile
    - minimum class-positive pixels per tile
    - minimum class tiles per slide
    - formal frozen slide-membership flags
    - Train/Test partition
    - label hierarchy
    - no PCA refit
    - no PC screening
    - no PC6

AGGC ANNOTATION SEMANTICS
-------------------------
Official masks are sparse positive annotations.

    positive mask support
        = explicit annotated class/state

    missing mask or zero support
        != biological negative

Missing class masks are kept unknown (NaN), not converted to 0.

OVERLAP
-------
Step18a froze:
    overlapping annotation support is excluded from affected class comparisons

Therefore, for each tile, this script measures:

    raw class fraction
    overlap-excluded class fraction

Frozen class-specific eligibility uses:
    overlap-excluded class fraction

The overlap is computed only inside the current tile crop, never with a
full-slide label-count array.

MEMORY-SAFE MASK ACCESS
-----------------------
Each mask is opened once with Pillow.

For every embedding tile:
    crop only that tile from each available class mask
    convert only that crop to NumPy
    count raw and exclusive pixels

This follows the user's already-validated AGGC tile-crop approach.

MEASUREMENT-ONLY BOUNDARY
-------------------------
This script DOES NOT calculate:

    G4-G3 effect
    G5-G4 effect
    G5-G3 effect
    G3->G4->G5 ordinal statistic
    Stroma->Normal->G3->G4->G5 ordered statistic
    Spearman
    Pearson
    AUC
    p-values
    bootstrap
    PCA refit
    PC screening
    PC6 analysis
    Test association

OUTPUT STRATEGY
---------------
To keep RAM bounded:

    - tile records are streamed directly to gzip CSV
    - slide QC rows are accumulated only at slide level
    - each slide's mask handles are closed immediately
    - gc.collect() runs after each slide

No multi-million-row DataFrame is held in memory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import sys
import traceback
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
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.2"
SCHEMA = "18b-aggc-pc3-gleason-histologic-state-measurement-schema-1.0.2-final"


# =============================================================================
# PATHS
# =============================================================================

MEDIA_BASE = REPO_ROOT

STEP18A_DIR = (
    RESULTS_ROOT
    / "18a_aggc_pc3_gleason_histologic_state_specification"
)

AGGC_BASE = CONFIG_AGGC_ROOT

EMB_DIR = AGGC_BASE / "aggc_virchow2_embeddings"

PCA_CANDIDATE_ROOTS = [
    AGGC_BASE / "aggc_pc1_pc8_grid_heatmaps",
    CONFIG_AGGC_ROOT.parent / "pca_global_projected_from_main",
]

TRAIN_IMAGE_DIR = AGGC_BASE / "train" / "Subset1_Train_image"
TRAIN_MASK_ROOT = (
    AGGC_BASE
    / "train"
    / "Subset1_Train_annotation"
    / "Train"
)

TEST_IMAGE_DIR = AGGC_BASE / "test" / "Subset1_Test_image"
TEST_MASK_ROOT = (
    AGGC_BASE
    / "test"
    / "Subset1_Test_annotation"
    / "Test"
)

OUT_DIR = (
    RESULTS_ROOT
    / "18b_aggc_pc3_gleason_histologic_state_measurements"
)

CLASSES = ["Stroma", "Normal", "G3", "G4", "G5"]

MASK_FILENAMES = {
    "Stroma": "Stroma_Mask.tif",
    "Normal": "Normal_Mask.tif",
    "G3": "G3_Mask.tif",
    "G4": "G4_Mask.tif",
    "G5": "G5_Mask.tif",
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def banner(title: str, width: int = 124) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def norm_col(x: str) -> str:
    return "".join(
        ch.lower()
        for ch in str(x)
        if ch.isalnum()
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )


def sha256_file(
    path: Path,
    block_size: int = 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def as_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)

    if pd.isna(v):
        return False

    s = str(v).strip().lower()

    return s in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "include",
        "included",
        "eligible",
        "pass",
        "ready",
    }


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:

    lookup = {
        norm_col(c): c
        for c in df.columns
    }

    for candidate in candidates:
        key = norm_col(candidate)

        if key in lookup:
            return lookup[key]

    if required:
        raise RuntimeError(
            "Required column not found.\n"
            f"Candidates: {list(candidates)}\n"
            f"Available: {list(df.columns)}"
        )

    return None


def recursively_find_key(
    obj: Any,
    keys: Sequence[str],
) -> Optional[Any]:

    wanted = {
        norm_col(k)
        for k in keys
    }

    if isinstance(obj, dict):
        for k, v in obj.items():
            if norm_col(k) in wanted:
                return v

        for v in obj.values():
            found = recursively_find_key(
                v,
                keys,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):
        for v in obj:
            found = recursively_find_key(
                v,
                keys,
            )

            if found is not None:
                return found

    return None


def require_numeric_from_specs(
    specs: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    label: str,
    predicate=None,
) -> float:

    for spec in specs:
        value = recursively_find_key(
            spec,
            keys,
        )

        if value is None:
            continue

        try:
            x = float(value)
        except Exception:
            continue

        if predicate is not None and not predicate(x):
            continue

        return x

    raise RuntimeError(
        f"Frozen Step18a specification does not expose a usable {label!r}.\n"
        f"Searched keys: {list(keys)}\n"
        "Step18b refuses to invent or redefine the frozen rule."
    )


def require_int_from_specs(
    specs: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    label: str,
    predicate=None,
) -> int:

    x = require_numeric_from_specs(
        specs,
        keys,
        label,
        predicate,
    )

    if abs(x - round(x)) > 1e-8:
        raise RuntimeError(
            f"Frozen {label} must be integer-like, got {x}."
        )

    return int(round(x))


def first_existing(
    paths: Sequence[Path],
) -> Optional[Path]:

    for path in paths:
        if path.exists():
            return path

    return None


# =============================================================================
# STEP18A FREEZE PARSING
# =============================================================================

def load_frozen_design(
    step18a_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    if not step18a_dir.is_dir():
        raise FileNotFoundError(
            f"Step18a directory not found: {step18a_dir}"
        )

    manifest_path = (
        step18a_dir
        / "aggc_pc3_gleason_frozen_partition_manifest.csv"
    )

    plan_path = (
        step18a_dir
        / "aggc_pc3_gleason_analysis_plan.json"
    )

    label_path = (
        step18a_dir
        / "aggc_pc3_gleason_label_specification.json"
    )

    geometry_path = (
        step18a_dir
        / "aggc_pc3_gleason_geometry_specification.json"
    )

    metadata_path = (
        step18a_dir
        / "analysis_metadata.json"
    )

    summary_path = (
        step18a_dir
        / "aggc_pc3_gleason_cohort_summary.csv"
    )

    required = [
        manifest_path,
        plan_path,
        label_path,
        geometry_path,
        metadata_path,
    ]

    missing = [
        p
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing frozen Step18a files:\n"
            +
            "\n".join(
                str(p)
                for p in missing
            )
        )

    metadata = load_json(
        metadata_path
    )

    frozen_version = str(
        metadata.get(
            "version",
            "",
        )
    )

    frozen_schema = str(
        metadata.get(
            "schema_version",
            "",
        )
    )

    if frozen_version != "1.1.0":
        raise RuntimeError(
            "Step18b FINAL requires Step18a v1.1.0.\n"
            f"Found version: {frozen_version!r}"
        )

    if "1.1" not in frozen_schema:
        raise RuntimeError(
            "Unexpected Step18a schema.\n"
            f"Found: {frozen_schema}"
        )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    slide_col = find_column(
        manifest,
        ["slide_id"],
    )

    partition_col = find_column(
        manifest,
        ["partition"],
    )

    manifest = manifest.rename(
        columns={
            slide_col: "slide_id",
            partition_col: "partition",
        }
    ).copy()

    manifest["slide_id"] = (
        manifest["slide_id"].astype(str)
    )

    include_cols = [
        c
        for c in manifest.columns
        if norm_col(c).startswith(
            norm_col("formal_18a_")
        )
        and norm_col(c).endswith(
            norm_col("_include")
        )
    ]

    if not include_cols:
        raise RuntimeError(
            "No formal_18a_*_include fields were found in the frozen manifest."
        )

    any_include = np.zeros(
        len(manifest),
        dtype=bool,
    )

    for col in include_cols:
        any_include |= (
            manifest[col]
            .map(as_bool)
            .to_numpy(dtype=bool)
        )

    frozen = manifest.loc[
        any_include
    ].copy()

    frozen[
        "step18_any_frozen_analysis_include"
    ] = True

    # Reproduce the pre-outcome frozen primary counts exactly.
    primary_col = find_column(
        manifest,
        ["formal_18a_primary_g3_g4_include"],
    )

    primary = manifest.loc[
        manifest[primary_col].map(as_bool)
    ].copy()

    train_primary_n = int(
        (
            primary["partition"]
            ==
            "train_validation"
        ).sum()
    )

    test_primary_n = int(
        (
            primary["partition"]
            ==
            "test_replication"
        ).sum()
    )

    if train_primary_n != 85:
        raise RuntimeError(
            f"Frozen Train G3/G4 count={train_primary_n}; expected 85."
        )

    if test_primary_n != 39:
        raise RuntimeError(
            f"Frozen Test G3/G4 count={test_primary_n}; expected 39."
        )

    plan = load_json(
        plan_path
    )

    labels = load_json(
        label_path
    )

    geometry = load_json(
        geometry_path
    )

    specs = [
        plan,
        labels,
        geometry,
        metadata,
    ]

    min_class_fraction = require_numeric_from_specs(
        specs,
        [
            "minimum_class_fraction_per_tile",
            "minimum_class_fraction",
            "min_class_fraction",
            "class_fraction_threshold",
        ],
        "minimum class fraction per tile",
        predicate=lambda x: 0 < x <= 1,
    )

    min_positive_pixels = require_int_from_specs(
        specs,
        [
            "minimum_class_positive_pixels_per_tile",
            "minimum_positive_pixels_per_tile",
        ],
        "minimum class-positive pixels per tile",
        predicate=lambda x: x >= 1,
    )

    min_tiles_per_class = require_int_from_specs(
        specs,
        [
            "minimum_class_tiles_per_slide",
            "minimum_usable_tiles_per_class_per_slide",
            "minimum_tiles_per_class",
            "min_tiles_per_class",
        ],
        "minimum class tiles per slide",
        predicate=lambda x: x >= 1,
    )

    pc_index = require_int_from_specs(
        specs,
        [
            "zero_based_component_index",
            "pc3_index_zero_based",
            "pc_index_zero_based",
        ],
        "zero-based PC3 component index",
        predicate=lambda x: x >= 0,
    )

    if pc_index != 2:
        raise RuntimeError(
            f"Frozen PC3 zero-based index must be 2, got {pc_index}."
        )

    frozen_info = {
        "step18a_version":
            frozen_version,

        "step18a_schema":
            frozen_schema,

        "train_primary_g3_g4_n":
            train_primary_n,

        "test_primary_g3_g4_n":
            test_primary_n,

        "n_union_frozen_slides":
            int(len(frozen)),

        "n_union_train":
            int(
                (
                    frozen["partition"]
                    ==
                    "train_validation"
                ).sum()
            ),

        "n_union_test":
            int(
                (
                    frozen["partition"]
                    ==
                    "test_replication"
                ).sum()
            ),

        "minimum_class_fraction_per_tile":
            float(min_class_fraction),

        "minimum_class_positive_pixels_per_tile":
            int(min_positive_pixels),

        "minimum_class_tiles_per_slide":
            int(min_tiles_per_class),

        "pc3_index_zero_based":
            int(pc_index),

        "include_columns":
            include_cols,

        "source_files":
            {
                "partition_manifest":
                    str(manifest_path),

                "analysis_plan":
                    str(plan_path),

                "label_specification":
                    str(label_path),

                "geometry_specification":
                    str(geometry_path),

                "analysis_metadata":
                    str(metadata_path),

                "cohort_summary":
                    (
                        str(summary_path)
                        if summary_path.exists()
                        else None
                    ),
            },
    }

    return frozen, frozen_info


# =============================================================================
# EXACT FILE RESOLUTION
# =============================================================================

def exact_slide_file_candidates(
    root: Path,
    slide_id: str,
    filename: str,
) -> List[Path]:

    return [
        root / filename,
        root / slide_id / filename,
    ]


def first_existing_exact(
    candidates: Sequence[Path],
) -> Optional[Path]:

    found = [
        p
        for p in candidates
        if p.is_file()
    ]

    if len(found) > 1:
        raise RuntimeError(
            "Multiple exact candidate files exist; refusing to guess:\n"
            +
            "\n".join(
                str(p)
                for p in found
            )
        )

    return (
        found[0]
        if found
        else None
    )


def resolve_coords_path(
    slide_id: str,
    frozen_row: pd.Series,
) -> Path:

    # First use an exact frozen path if Step18a preserved one.
    for col in frozen_row.index:
        nc = norm_col(col)

        if (
            "coord" in nc
            and
            (
                "path" in nc
                or
                "file" in nc
            )
        ):
            value = frozen_row[col]

            if pd.notna(value) and str(value).strip():
                p = Path(str(value))

                if p.is_file():
                    return p

    filename = f"{slide_id}_coords.npy"

    p = first_existing_exact(
        exact_slide_file_candidates(
            EMB_DIR,
            slide_id,
            filename,
        )
    )

    if p is None:
        raise FileNotFoundError(
            f"Exact coords file not found for {slide_id}."
        )

    return p


def resolve_meta_path(
    slide_id: str,
    frozen_row: pd.Series,
) -> Path:

    for col in frozen_row.index:
        nc = norm_col(col)

        if (
            "meta" in nc
            and
            (
                "path" in nc
                or
                "file" in nc
            )
        ):
            value = frozen_row[col]

            if pd.notna(value) and str(value).strip():
                p = Path(str(value))

                if p.is_file():
                    return p

    filename = f"{slide_id}_meta.json"

    p = first_existing_exact(
        exact_slide_file_candidates(
            EMB_DIR,
            slide_id,
            filename,
        )
    )

    if p is None:
        raise FileNotFoundError(
            f"Exact embedding meta JSON not found for {slide_id}."
        )

    return p


def resolve_pca_path(
    slide_id: str,
    frozen_row: pd.Series,
) -> Path:

    for col in frozen_row.index:
        nc = norm_col(col)

        if (
            (
                "pca" in nc
                or
                "projection" in nc
            )
            and
            (
                "path" in nc
                or
                "file" in nc
            )
        ):
            value = frozen_row[col]

            if pd.notna(value) and str(value).strip():
                p = Path(str(value))

                if p.is_file():
                    return p

    filename = f"{slide_id}_pca_targets.npy"

    exact = []

    for root in PCA_CANDIDATE_ROOTS:
        exact.extend(
            exact_slide_file_candidates(
                root,
                slide_id,
                filename,
            )
        )

    p = first_existing_exact(
        exact
    )

    if p is not None:
        return p

    # Final locator fallback: unique recursive exact-suffix result only.
    matches: List[Path] = []

    for root in PCA_CANDIDATE_ROOTS:
        if root.exists():
            matches.extend(
                root.rglob(
                    filename
                )
            )

    matches = sorted(
        set(matches)
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous PCA projection for {slide_id}: {matches[:10]}"
        )

    raise FileNotFoundError(
        f"Frozen PCA projection not found for {slide_id}."
    )


def resolve_wsi_path(
    slide_id: str,
    partition: str,
    frozen_row: pd.Series,
) -> Path:

    for col in frozen_row.index:
        nc = norm_col(col)

        if (
            (
                "wsi" in nc
                or
                "image" in nc
                or
                "slide" in nc
            )
            and
            (
                "path" in nc
                or
                "file" in nc
            )
        ):
            value = frozen_row[col]

            if pd.notna(value) and str(value).strip():
                p = Path(str(value))

                if p.is_file():
                    return p

    root = (
        TRAIN_IMAGE_DIR
        if partition == "train_validation"
        else TEST_IMAGE_DIR
    )

    found = [
        p
        for p in (
            root / f"{slide_id}.tiff",
            root / f"{slide_id}.tif",
        )
        if p.is_file()
    ]

    if len(found) > 1:
        raise RuntimeError(
            f"Multiple WSI candidates for {slide_id}: {found}"
        )

    if not found:
        raise FileNotFoundError(
            f"WSI not found for {slide_id}."
        )

    return found[0]


def resolve_mask_dir(
    slide_id: str,
    partition: str,
) -> Path:

    root = (
        TRAIN_MASK_ROOT
        if partition == "train_validation"
        else TEST_MASK_ROOT
    )

    path = (
        root
        /
        slide_id
    )

    if not path.is_dir():
        raise FileNotFoundError(
            f"Mask directory not found: {path}"
        )

    return path


# =============================================================================
# COORD / WSI / MASK HELPERS
# =============================================================================

def ensure_coords(
    coords: np.ndarray,
    path: Path,
) -> np.ndarray:

    coords = np.asarray(coords)

    if (
        coords.ndim != 2
        or
        coords.shape[1] < 2
    ):
        raise ValueError(
            f"Coordinates must have shape (N,>=2); "
            f"{path} has {coords.shape}."
        )

    coords = coords[:, :2]

    if not np.issubdtype(
        coords.dtype,
        np.number,
    ):
        raise TypeError(
            f"Coordinates are nonnumeric: {path}"
        )

    if not np.all(
        np.isfinite(coords)
    ):
        raise ValueError(
            f"Coordinates contain NaN/Inf: {path}"
        )

    return np.rint(
        coords
    ).astype(
        np.int64
    )


def read_wsi_dimensions(
    path: Path,
) -> Tuple[int, int]:

    # Pillow reads metadata without converting the full WSI into NumPy.
    with Image.open(
        path
    ) as image:

        return (
            int(image.width),
            int(image.height),
        )


def estimate_foreground_value(
    image: Image.Image,
    preview_width: int = 1024,
) -> Tuple[bool, float]:

    """
    Same safeguard used in the user's prior successful AGGC code.

    AGGC annotation is sparse; minority Boolean value is treated as foreground.
    If the preview is completely empty, use the previously verified AGGC
    convention: foreground=True.
    """

    width, height = image.size

    preview_height = max(
        1,
        int(
            round(
                height
                *
                preview_width
                /
                max(
                    width,
                    1,
                )
            )
        ),
    )

    preview = image.resize(
        (
            preview_width,
            preview_height,
        ),
        resample=Image.Resampling.NEAREST,
    )

    arr = np.asarray(
        preview,
        dtype=bool,
    )

    n_true = int(
        np.count_nonzero(
            arr
        )
    )

    n_total = int(
        arr.size
    )

    n_false = (
        n_total
        -
        n_true
    )

    if n_true == 0:
        return True, 0.0

    if n_false == 0:
        return False, 0.0

    foreground_value = bool(
        n_true
        <
        n_false
    )

    foreground_count = (
        n_true
        if foreground_value
        else n_false
    )

    return (
        foreground_value,
        float(
            foreground_count
            /
            max(
                n_total,
                1,
            )
        ),
    )


def open_masks_memory_safe(
    mask_dir: Path,
    wsi_size: Tuple[int, int],
) -> Tuple[
    Dict[str, Optional[Image.Image]],
    Dict[str, Optional[bool]],
    Dict[str, Dict[str, Any]],
]:

    images: Dict[
        str,
        Optional[Image.Image],
    ] = {}

    foreground_values: Dict[
        str,
        Optional[bool],
    ] = {}

    metadata: Dict[
        str,
        Dict[str, Any],
    ] = {}

    wsi_w, wsi_h = wsi_size

    try:
        for cls in CLASSES:
            path = (
                mask_dir
                /
                MASK_FILENAMES[
                    cls
                ]
            )

            if not path.is_file():
                images[cls] = None
                foreground_values[cls] = None

                metadata[cls] = {
                    "exists": False,
                    "path": str(path),
                }

                continue

            image = Image.open(
                path
            )

            # Step17 audit established Pillow mode '1' as authoritative.
            if image.mode != "1":
                image.close()

                raise RuntimeError(
                    f"{cls} mask is not Pillow mode '1': "
                    f"{path} (mode={image.mode!r})"
                )

            if image.size != (
                wsi_w,
                wsi_h,
            ):
                image.close()

                raise RuntimeError(
                    f"{cls} mask geometry differs from WSI for {mask_dir.name}: "
                    f"mask={image.size}, WSI={(wsi_w, wsi_h)}. "
                    "Step18b refuses geometry reinterpretation."
                )

            foreground_value, preview_fraction = (
                estimate_foreground_value(
                    image
                )
            )

            images[cls] = image
            foreground_values[cls] = foreground_value

            metadata[cls] = {
                "exists":
                    True,

                "path":
                    str(path),

                "mode":
                    image.mode,

                "size":
                    [
                        int(
                            image.width
                        ),
                        int(
                            image.height
                        ),
                    ],

                "foreground_value":
                    bool(
                        foreground_value
                    ),

                "preview_foreground_fraction":
                    float(
                        preview_fraction
                    ),
            }

        return (
            images,
            foreground_values,
            metadata,
        )

    except Exception:
        close_masks(
            images
        )
        raise


def close_masks(
    images: Dict[
        str,
        Optional[Image.Image],
    ],
) -> None:

    for image in images.values():

        if image is None:
            continue

        try:
            image.close()
        except Exception:
            pass


def crop_mask_bool(
    image: Image.Image,
    foreground_value: bool,
    box: Tuple[int, int, int, int],
) -> np.ndarray:

    """
    Memory-safe primitive:
        only the tile crop enters NumPy.
    """

    crop = image.crop(
        box
    )

    arr = np.asarray(
        crop,
        dtype=bool,
    )

    if not foreground_value:
        arr = ~arr

    return arr


# =============================================================================
# PER-SLIDE MEASUREMENT
# =============================================================================

def measure_slide(
    frozen_row: pd.Series,
    frozen_info: Dict[str, Any],
    tile_writer: csv.DictWriter,
    progress_every: int,
) -> Dict[str, Any]:

    slide_id = str(
        frozen_row[
            "slide_id"
        ]
    )

    partition = str(
        frozen_row[
            "partition"
        ]
    )

    coords_path = resolve_coords_path(
        slide_id,
        frozen_row,
    )

    meta_path = resolve_meta_path(
        slide_id,
        frozen_row,
    )

    pca_path = resolve_pca_path(
        slide_id,
        frozen_row,
    )

    wsi_path = resolve_wsi_path(
        slide_id,
        partition,
        frozen_row,
    )

    mask_dir = resolve_mask_dir(
        slide_id,
        partition,
    )

    coords = ensure_coords(
        np.load(
            coords_path,
            mmap_mode="r",
        ),
        coords_path,
    )

    pca = np.load(
        pca_path,
        mmap_mode="r",
    )

    if pca.ndim != 2:
        raise RuntimeError(
            f"{slide_id}: PCA array is not 2-D: {pca.shape}"
        )

    if pca.shape[0] != coords.shape[0]:
        raise RuntimeError(
            f"{slide_id}: coords/PCA row mismatch: "
            f"{coords.shape[0]} vs {pca.shape[0]}"
        )

    pc_index = int(
        frozen_info[
            "pc3_index_zero_based"
        ]
    )

    if pca.shape[1] <= pc_index:
        raise RuntimeError(
            f"{slide_id}: PCA array has no frozen PC3 column {pc_index}."
        )

    meta = load_json(
        meta_path
    )

    if "patch_size" not in meta:
        raise RuntimeError(
            f"{slide_id}: embedding meta lacks patch_size."
        )

    patch_size = int(
        meta[
            "patch_size"
        ]
    )

    if patch_size <= 0:
        raise RuntimeError(
            f"{slide_id}: invalid patch_size={patch_size}."
        )

    wsi_w, wsi_h = read_wsi_dimensions(
        wsi_path
    )

    (
        mask_images,
        foreground_values,
        mask_metadata,
    ) = open_masks_memory_safe(
        mask_dir,
        (
            wsi_w,
            wsi_h,
        ),
    )

    n_tiles = int(
        coords.shape[0]
    )

    min_fraction = float(
        frozen_info[
            "minimum_class_fraction_per_tile"
        ]
    )

    min_positive_pixels = int(
        frozen_info[
            "minimum_class_positive_pixels_per_tile"
        ]
    )

    min_tiles_per_class = int(
        frozen_info[
            "minimum_class_tiles_per_slide"
        ]
    )

    pc3 = np.asarray(
        pca[
            :,
            pc_index,
        ],
        dtype=np.float64,
    )

    finite_pc3 = np.isfinite(
        pc3
    )

    counts_eligible = {
        cls:
        0
        for cls in CLASSES
    }

    counts_raw_positive_pixels = {
        cls:
        0
        for cls in CLASSES
    }

    counts_exclusive_positive_pixels = {
        cls:
        0
        for cls in CLASSES
    }

    n_geometry_ok = 0
    n_tiles_any_overlap = 0
    n_finite_pc3 = int(
        finite_pc3.sum()
    )

    full_tile_area = int(
        patch_size
        *
        patch_size
    )

    try:
        for tile_index, coord in enumerate(
            coords
        ):

            x = int(
                coord[
                    0
                ]
            )

            y = int(
                coord[
                    1
                ]
            )

            x1 = (
                x
                +
                patch_size
            )

            y1 = (
                y
                +
                patch_size
            )

            geometry_ok = bool(
                x >= 0
                and
                y >= 0
                and
                x1 <= wsi_w
                and
                y1 <= wsi_h
            )

            if geometry_ok:
                n_geometry_ok += 1

            rec: Dict[
                str,
                Any,
            ] = {
                "slide_id":
                    slide_id,

                "partition":
                    partition,

                "tile_index":
                    int(
                        tile_index
                    ),

                "x":
                    x,

                "y":
                    y,

                "patch_size":
                    patch_size,

                "geometry_ok":
                    geometry_ok,

                "pc3_finite":
                    bool(
                        finite_pc3[
                            tile_index
                        ]
                    ),

                "PC3":
                    (
                        float(
                            pc3[
                                tile_index
                            ]
                        )
                        if finite_pc3[
                            tile_index
                        ]
                        else ""
                    ),
            }

            if not geometry_ok:
                rec[
                    "any_annotation_overlap_fraction"
                ] = ""

                for cls in CLASSES:
                    lc = cls.lower()

                    rec[
                        f"{lc}_mask_exists"
                    ] = bool(
                        mask_images[
                            cls
                        ]
                        is not None
                    )

                    rec[
                        f"{lc}_raw_fraction"
                    ] = ""

                    rec[
                        f"{lc}_exclusive_fraction"
                    ] = ""

                    rec[
                        f"{lc}_eligible"
                    ] = False

                tile_writer.writerow(
                    rec
                )

                continue

            box = (
                x,
                y,
                x1,
                y1,
            )

            tile_masks: Dict[
                str,
                Optional[np.ndarray],
            ] = {}

            available_arrays: List[
                np.ndarray
            ] = []

            for cls in CLASSES:
                image = mask_images[
                    cls
                ]

                foreground_value = foreground_values[
                    cls
                ]

                if (
                    image
                    is None
                    or
                    foreground_value
                    is None
                ):
                    tile_masks[
                        cls
                    ] = None

                    continue

                arr = crop_mask_bool(
                    image,
                    bool(
                        foreground_value
                    ),
                    box,
                )

                if arr.shape != (
                    patch_size,
                    patch_size,
                ):
                    raise RuntimeError(
                        f"{slide_id} {cls}: unexpected crop shape "
                        f"{arr.shape} at tile {tile_index}."
                    )

                tile_masks[
                    cls
                ] = arr

                available_arrays.append(
                    arr
                )

            if available_arrays:
                label_count = np.zeros(
                    (
                        patch_size,
                        patch_size,
                    ),
                    dtype=np.uint8,
                )

                for arr in available_arrays:
                    label_count += arr.astype(
                        np.uint8,
                        copy=False,
                    )

                overlap_mask = (
                    label_count
                    >=
                    2
                )

                overlap_pixels = int(
                    overlap_mask.sum(
                        dtype=np.int64
                    )
                )

                if overlap_pixels > 0:
                    n_tiles_any_overlap += 1

                rec[
                    "any_annotation_overlap_fraction"
                ] = float(
                    overlap_pixels
                    /
                    full_tile_area
                )

            else:
                label_count = None

                rec[
                    "any_annotation_overlap_fraction"
                ] = 0.0

            for cls in CLASSES:
                lc = cls.lower()

                arr = tile_masks[
                    cls
                ]

                rec[
                    f"{lc}_mask_exists"
                ] = bool(
                    arr
                    is not None
                )

                if arr is None:
                    # Sparse/missing annotation remains unknown, not zero.
                    rec[
                        f"{lc}_raw_fraction"
                    ] = ""

                    rec[
                        f"{lc}_exclusive_fraction"
                    ] = ""

                    rec[
                        f"{lc}_eligible"
                    ] = False

                    continue

                raw_pixels = int(
                    arr.sum(
                        dtype=np.int64
                    )
                )

                counts_raw_positive_pixels[
                    cls
                ] += raw_pixels

                if label_count is None:
                    exclusive_pixels = raw_pixels

                else:
                    exclusive_pixels = int(
                        (
                            arr
                            &
                            (
                                label_count
                                ==
                                1
                            )
                        ).sum(
                            dtype=np.int64
                        )
                    )

                counts_exclusive_positive_pixels[
                    cls
                ] += exclusive_pixels

                raw_fraction = float(
                    raw_pixels
                    /
                    full_tile_area
                )

                exclusive_fraction = float(
                    exclusive_pixels
                    /
                    full_tile_area
                )

                eligible = bool(
                    finite_pc3[
                        tile_index
                    ]
                    and
                    exclusive_pixels
                    >=
                    min_positive_pixels
                    and
                    exclusive_fraction
                    >=
                    min_fraction
                )

                if eligible:
                    counts_eligible[
                        cls
                    ] += 1

                rec[
                    f"{lc}_raw_fraction"
                ] = raw_fraction

                rec[
                    f"{lc}_exclusive_fraction"
                ] = exclusive_fraction

                rec[
                    f"{lc}_eligible"
                ] = eligible

            tile_writer.writerow(
                rec
            )

            # Explicitly drop all per-tile arrays.
            tile_masks.clear()

            if (
                progress_every > 0
                and
                (
                    (
                        tile_index
                        +
                        1
                    )
                    %
                    progress_every
                    ==
                    0
                    or
                    tile_index
                    +
                    1
                    ==
                    n_tiles
                )
            ):
                print(
                    f"    [MEASURE] {tile_index + 1:,}/{n_tiles:,} "
                    f"({100.0 * (tile_index + 1) / max(n_tiles, 1):.1f}%)"
                )

    finally:
        close_masks(
            mask_images
        )

        del pca
        del coords
        del pc3

        gc.collect()

    row: Dict[
        str,
        Any,
    ] = {
        "slide_id":
            slide_id,

        "partition":
            partition,

        "n_tiles_total":
            n_tiles,

        "n_tiles_geometry_ok":
            int(
                n_geometry_ok
            ),

        "n_tiles_pc3_finite":
            int(
                n_finite_pc3
            ),

        "n_tiles_with_any_annotation_overlap":
            int(
                n_tiles_any_overlap
            ),

        "patch_size":
            patch_size,

        "wsi_width":
            wsi_w,

        "wsi_height":
            wsi_h,

        "coords_path":
            str(
                coords_path
            ),

        "pca_path":
            str(
                pca_path
            ),

        "meta_path":
            str(
                meta_path
            ),

        "wsi_path":
            str(
                wsi_path
            ),

        "mask_dir":
            str(
                mask_dir
            ),

        "measurement_backend":
            "Pillow_tile_crop_only",

        "frozen_min_class_fraction_per_tile":
            min_fraction,

        "frozen_min_class_positive_pixels_per_tile":
            min_positive_pixels,

        "frozen_min_class_tiles_per_slide":
            min_tiles_per_class,
    }

    # Preserve every pre-frozen Step18 analysis-membership flag.
    for col in frozen_row.index:
        if (
            norm_col(
                col
            ).startswith(
                norm_col(
                    "formal_18a_"
                )
            )
            and
            norm_col(
                col
            ).endswith(
                norm_col(
                    "_include"
                )
            )
        ):
            row[
                col
            ] = as_bool(
                frozen_row[
                    col
                ]
            )

    for cls in CLASSES:
        lc = cls.lower()

        row[
            f"{lc}_mask_exists"
        ] = bool(
            mask_metadata[
                cls
            ].get(
                "exists",
                False,
            )
        )

        row[
            f"n_{lc}_eligible_tiles"
        ] = int(
            counts_eligible[
                cls
            ]
        )

        row[
            f"{lc}_meets_frozen_min_tiles"
        ] = bool(
            counts_eligible[
                cls
            ]
            >=
            min_tiles_per_class
        )

    # QC conjunctions only. These are NOT outcomes/effects.
    row[
        "primary_g3_g4_meets_frozen_min_each"
    ] = bool(
        counts_eligible[
            "G3"
        ]
        >=
        min_tiles_per_class
        and
        counts_eligible[
            "G4"
        ]
        >=
        min_tiles_per_class
    )

    row[
        "g4_g5_meets_frozen_min_each"
    ] = bool(
        counts_eligible[
            "G4"
        ]
        >=
        min_tiles_per_class
        and
        counts_eligible[
            "G5"
        ]
        >=
        min_tiles_per_class
    )

    row[
        "g3_g5_meets_frozen_min_each"
    ] = bool(
        counts_eligible[
            "G3"
        ]
        >=
        min_tiles_per_class
        and
        counts_eligible[
            "G5"
        ]
        >=
        min_tiles_per_class
    )

    row[
        "all3_g3_g4_g5_meets_frozen_min_each"
    ] = bool(
        all(
            counts_eligible[
                c
            ]
            >=
            min_tiles_per_class
            for c in [
                "G3",
                "G4",
                "G5",
            ]
        )
    )

    row[
        "all5_states_meet_frozen_min_each"
    ] = bool(
        all(
            counts_eligible[
                c
            ]
            >=
            min_tiles_per_class
            for c in CLASSES
        )
    )

    return row


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Step18b v1.0.2 FINAL memory-safe measurement-only AGGC "
            "PC3 Gleason/histologic-state measurement."
        ),
    )

    parser.add_argument(
        "--step18a-dir",
        type=Path,
        default=STEP18A_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    out_dir = args.output_dir

    banner(
        "STEP 18B: AGGC PC3 GLEASON / HISTOLOGIC-STATE MEASUREMENTS"
    )

    print(f"[VERSION]      {VERSION}")
    print(f"[SCHEMA]       {SCHEMA}")
    print(f"[STEP 18A]     {args.step18a_dir}")
    print(f"[OUTPUT]       {out_dir}")
    print(f"[OVERWRITE]    {args.overwrite}")

    print()
    print("[ROLE]         MEASUREMENT ONLY")
    print("[BACKEND]      PILLOW TILE-CROP ONLY")
    print("[FULL MASK NP] NEVER")
    print("[IMAGE.LOAD]   NEVER")
    print("[INTEGRAL IMG] NEVER")
    print("[ASSOCIATION]  NONE")
    print("[INFERENCE]    NONE")
    print("[PCA REFIT]    NONE")
    print("[PC SCREENING] NONE")
    print("[PC6]          NONE")
    print("[TEST OUTCOME] NOT EVALUATED")

    if (
        out_dir.exists()
        and
        any(
            out_dir.iterdir()
        )
        and
        not args.overwrite
    ):
        raise RuntimeError(
            f"Output directory is non-empty: {out_dir}\n"
            "Use --overwrite only for intentional Step18b regeneration."
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen, frozen_info = load_frozen_design(
        args.step18a_dir
    )

    banner(
        "FROZEN MEASUREMENT RULES"
    )

    print(
        f"Step18a version                    : "
        f"{frozen_info['step18a_version']}"
    )

    print(
        f"Primary frozen Train G3/G4 slides : "
        f"{frozen_info['train_primary_g3_g4_n']}"
    )

    print(
        f"Primary frozen Test G3/G4 slides  : "
        f"{frozen_info['test_primary_g3_g4_n']}"
    )

    print(
        f"Union frozen Train slides         : "
        f"{frozen_info['n_union_train']}"
    )

    print(
        f"Union frozen Test slides          : "
        f"{frozen_info['n_union_test']}"
    )

    print(
        f"Union frozen slides total         : "
        f"{frozen_info['n_union_frozen_slides']}"
    )

    print(
        f"Frozen class fraction threshold   : "
        f"{frozen_info['minimum_class_fraction_per_tile']:.6f}"
    )

    print(
        f"Frozen min positive pixels/tile   : "
        f"{frozen_info['minimum_class_positive_pixels_per_tile']}"
    )

    print(
        f"Frozen minimum tiles/class/slide  : "
        f"{frozen_info['minimum_class_tiles_per_slide']}"
    )

    print(
        f"Frozen PC3 zero-based column      : "
        f"{frozen_info['pc3_index_zero_based']}"
    )

    print(
        "Patch size source                  : per-slide embedding meta"
    )

    print(
        "Classes measured                    : "
        +
        ", ".join(
            CLASSES
        )
    )

    print(
        "Overlap handling                    : raw + overlap-excluded fractions"
    )

    print(
        "Eligibility uses                    : overlap-excluded fraction"
    )

    print(
        "Absent class mask -> fraction 0     : NO"
    )

    tile_path = (
        out_dir
        /
        "aggc_pc3_histologic_state_tile_measurements.csv.gz"
    )

    slide_qc_path = (
        out_dir
        /
        "aggc_pc3_histologic_state_slide_measurement_qc.csv"
    )

    failure_path = (
        out_dir
        /
        "aggc_pc3_histologic_state_measurement_failures.csv"
    )

    integrity_path = (
        out_dir
        /
        "frozen_source_integrity_audit.csv"
    )

    metadata_path = (
        out_dir
        /
        "analysis_metadata.json"
    )

    readme_path = (
        out_dir
        /
        "README.txt"
    )

    tile_fields = [
        "slide_id",
        "partition",
        "tile_index",
        "x",
        "y",
        "patch_size",
        "geometry_ok",
        "pc3_finite",
        "PC3",
        "any_annotation_overlap_fraction",
    ]

    for cls in CLASSES:
        lc = cls.lower()

        tile_fields.extend(
            [
                f"{lc}_mask_exists",
                f"{lc}_raw_fraction",
                f"{lc}_exclusive_fraction",
                f"{lc}_eligible",
            ]
        )

    slide_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    failures: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    with gzip.open(
        tile_path,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=5,
    ) as gz:

        writer = csv.DictWriter(
            gz,
            fieldnames=tile_fields,
        )

        writer.writeheader()

        total = len(
            frozen
        )

        for index, (
            _,
            frozen_row,
        ) in enumerate(
            frozen.iterrows(),
            1,
        ):
            slide_id = str(
                frozen_row[
                    "slide_id"
                ]
            )

            partition = str(
                frozen_row[
                    "partition"
                ]
            )

            print(
                f"[SLIDE {index:03d}/{total:03d}] "
                f"{partition:16s} {slide_id}"
            )

            try:
                row = measure_slide(
                    frozen_row=frozen_row,
                    frozen_info=frozen_info,
                    tile_writer=writer,
                    progress_every=args.progress_every,
                )

                slide_rows.append(
                    row
                )

                print(
                    "  [OK] "
                    f"Stroma={row['n_stroma_eligible_tiles']:,} | "
                    f"Normal={row['n_normal_eligible_tiles']:,} | "
                    f"G3={row['n_g3_eligible_tiles']:,} | "
                    f"G4={row['n_g4_eligible_tiles']:,} | "
                    f"G5={row['n_g5_eligible_tiles']:,}"
                )

            except Exception as exc:
                failures.append(
                    {
                        "slide_id":
                            slide_id,

                        "partition":
                            partition,

                        "error_type":
                            type(
                                exc
                            ).__name__,

                        "error_message":
                            str(
                                exc
                            ),

                        "traceback":
                            traceback.format_exc(),
                    }
                )

                print(
                    f"  [ERROR] {type(exc).__name__}: {exc}"
                )

            finally:
                gc.collect()

    slide_df = pd.DataFrame(
        slide_rows
    )

    slide_df.to_csv(
        slide_qc_path,
        index=False,
    )

    failure_df = pd.DataFrame(
        failures,
        columns=[
            "slide_id",
            "partition",
            "error_type",
            "error_message",
            "traceback",
        ],
    )

    failure_df.to_csv(
        failure_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Frozen source integrity
    # -------------------------------------------------------------------------

    integrity_rows = []

    for role, path_string in frozen_info[
        "source_files"
    ].items():

        if not path_string:
            continue

        path = Path(
            path_string
        )

        integrity_rows.append(
            {
                "role":
                    role,

                "path":
                    str(
                        path
                    ),

                "exists":
                    path.exists(),

                "sha256":
                    (
                        sha256_file(
                            path
                        )
                        if path.exists()
                        else ""
                    ),
            }
        )

    pd.DataFrame(
        integrity_rows
    ).to_csv(
        integrity_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # FINAL QC SUMMARY
    # -------------------------------------------------------------------------

    banner(
        "FINAL MEASUREMENT SUMMARY"
    )

    print(
        f"Frozen union slides requested : {len(frozen)}"
    )

    print(
        f"Slides measured               : {len(slide_df)}"
    )

    print(
        f"Slides failed                 : {len(failure_df)}"
    )

    if not slide_df.empty:
        for partition in [
            "train_validation",
            "test_replication",
        ]:
            d = slide_df.loc[
                slide_df[
                    "partition"
                ]
                ==
                partition
            ].copy()

            if d.empty:
                continue

            print()
            print(
                f"[{partition}]"
            )

            print(
                f"Slides measured               : {len(d)}"
            )

            for cls in CLASSES:
                lc = cls.lower()

                print(
                    f"{cls:6s} >= frozen min tiles      : "
                    f"{int(d[f'{lc}_meets_frozen_min_tiles'].sum())}/{len(d)}"
                )

            primary_flag = (
                "formal_18a_primary_g3_g4_include"
            )

            if primary_flag in d.columns:
                primary_d = d.loc[
                    d[
                        primary_flag
                    ].map(
                        as_bool
                    )
                ]

                print(
                    "Primary frozen G3/G4 slides    : "
                    f"{len(primary_d)}"
                )

                print(
                    "Primary G3/G4 >= min each      : "
                    f"{int(primary_d['primary_g3_g4_meets_frozen_min_each'].sum())}"
                    f"/{len(primary_d)}"
                )

            all3_flag = (
                "formal_18a_g3_g4_g5_ordinal_exploratory_include"
            )

            if all3_flag in d.columns:
                all3_d = d.loc[
                    d[
                        all3_flag
                    ].map(
                        as_bool
                    )
                ]

                print(
                    "Frozen tumor all-3 slides      : "
                    f"{len(all3_d)}"
                )

                print(
                    "All-3 >= min each              : "
                    f"{int(all3_d['all3_g3_g4_g5_meets_frozen_min_each'].sum())}"
                    f"/{len(all3_d)}"
                )

            all5_flag = (
                "formal_18a_extended_all5_complete_case_include"
            )

            if all5_flag in d.columns:
                all5_d = d.loc[
                    d[
                        all5_flag
                    ].map(
                        as_bool
                    )
                ]

                print(
                    "Frozen all-five slides         : "
                    f"{len(all5_d)}"
                )

                print(
                    "All-five >= min each           : "
                    f"{int(all5_d['all5_states_meet_frozen_min_each'].sum())}"
                    f"/{len(all5_d)}"
                )

    # -------------------------------------------------------------------------
    # MEASUREMENT-ONLY AUDIT
    # -------------------------------------------------------------------------

    banner(
        "MEASUREMENT-ONLY AUDIT"
    )

    audit = [
        ("G4-G3 effect calculated", "NO"),
        ("G5-G4 effect calculated", "NO"),
        ("G5-G3 effect calculated", "NO"),
        ("Tumor ordinal trend calculated", "NO"),
        ("Five-state trend calculated", "NO"),
        ("Spearman calculated", "NO"),
        ("Pearson calculated", "NO"),
        ("AUC calculated", "NO"),
        ("P-values calculated", "NO"),
        ("Bootstrap performed", "NO"),
        ("PCA refit", "NO"),
        ("PC screening", "NO"),
        ("PC6 analysis", "NO"),
        ("Mask redefinition", "NO"),
        ("Geometry redefinition", "NO"),
        ("Threshold redefinition", "NO"),
        ("Label redefinition", "NO"),
        ("Absent annotation -> negative", "NO"),
        ("Test association evaluated", "NO"),
        ("Full-resolution NumPy mask", "NO"),
        ("Pillow image.load()", "NO"),
        ("Integral image", "NO"),
    ]

    for name, value in audit:
        print(
            f"{name:40s}: {value}"
        )

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    metadata = {
        "version":
            VERSION,

        "schema":
            SCHEMA,

        "role":
            "measurement_only",

        "implementation":
            "Pillow_tile_crop_only",

        "memory_safety":
            {
                "full_resolution_numpy_mask":
                    False,

                "pillow_image_load_called":
                    False,

                "integral_image":
                    False,

                "only_tile_crop_enters_numpy":
                    True,

                "tile_output_streamed":
                    True,

                "all_slide_tile_dataframe_retained":
                    False,
            },

        "step18a_dir":
            str(
                args.step18a_dir
            ),

        "output_dir":
            str(
                out_dir
            ),

        "frozen_design":
            frozen_info,

        "classes_measured":
            CLASSES,

        "fraction_fields":
            {
                "raw":
                    "raw annotated class fraction within tile",

                "exclusive":
                    (
                        "class fraction after removing pixels where "
                        "two or more available class masks overlap"
                    ),
            },

        "eligibility_fraction":
            "exclusive_fraction",

        "n_frozen_union_slides_requested":
            int(
                len(
                    frozen
                )
            ),

        "n_slides_measured":
            int(
                len(
                    slide_df
                )
            ),

        "n_slides_failed":
            int(
                len(
                    failure_df
                )
            ),

        "semantic_guardrails":
            {
                "aggc_masks_sparse_nonexhaustive":
                    True,

                "missing_mask_interpreted_as_zero":
                    False,

                "unannotated_interpreted_as_nonclass":
                    False,

                "overlap_silently_reassigned":
                    False,
            },

        "forbidden_analyses_performed":
            {
                "g4_minus_g3":
                    False,

                "g5_minus_g4":
                    False,

                "g5_minus_g3":
                    False,

                "tumor_ordinal_trend":
                    False,

                "five_state_trend":
                    False,

                "correlation":
                    False,

                "auc":
                    False,

                "p_values":
                    False,

                "bootstrap":
                    False,

                "pca_refit":
                    False,

                "pc_screening":
                    False,

                "pc6":
                    False,
            },

        "outputs":
            {
                "tile_measurements":
                    str(
                        tile_path
                    ),

                "slide_measurement_qc":
                    str(
                        slide_qc_path
                    ),

                "measurement_failures":
                    str(
                        failure_path
                    ),

                "frozen_source_integrity_audit":
                    str(
                        integrity_path
                    ),
            },
    }

    save_json(
        metadata,
        metadata_path,
    )

    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------

    readme = f"""STEP 18B: AGGC PC3 GLEASON / HISTOLOGIC-STATE MEASUREMENTS
Version: {VERSION}
Schema: {SCHEMA}

ROLE
----
Measurement only.

IMPLEMENTATION
--------------
This final v1.0.2 uses the memory-safe strategy already validated in the
user's prior AGGC scripts:

    PIL.Image.open(mask)
    -> image.crop(tile_box)
    -> np.asarray(tile_crop)

Only tile-sized mask crops enter NumPy.

It does NOT:
    decode a full-resolution mask into NumPy,
    call image.load(),
    build an integral image,
    or retain the full multi-slide tile table in RAM.

SPECIFICATION AUTHORITY
-----------------------
{args.step18a_dir}

Required Step18a version:
    1.1.0

FROZEN PRIMARY
--------------
G3 vs G4
within-slide paired
slide = independent unit

Step18b DOES NOT calculate the G4-G3 effect.

FROZEN EXPLORATORY ANALYSES
---------------------------
G5-G4
G5-G3
G3 -> G4 -> G5
Stroma -> Normal -> G3 -> G4 -> G5
all-five complete-case ordered summary

Step18b DOES NOT calculate any of these effects or trends.

MEASURED STATES
---------------
Stroma
Normal
G3
G4
G5

ANNOTATION SEMANTICS
--------------------
AGGC masks are sparse explicit positive annotations.

Missing/unannotated support is not converted to a biological negative.

OVERLAP
-------
Both raw and overlap-excluded fractions are measured.

Frozen class-specific eligibility uses overlap-excluded fraction.

OUTPUTS
-------
{tile_path}
    Tile-level PC3 + five-state measurement fields.

{slide_qc_path}
    Slide-level measurement/QC counts only.

{failure_path}
    Measurement failures.

{integrity_path}
    SHA256 audit of consumed Step18a frozen files.

NO INFERENCE
------------
No paired effects.
No ordinal trends.
No Spearman/Pearson.
No AUC.
No p-values.
No bootstrap.
No PCA refit.
No PC screening.
No PC6.
No Test association evaluation.
"""

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    print()
    print("[OK] Saved:")

    for path in [
        tile_path,
        slide_qc_path,
        failure_path,
        integrity_path,
        metadata_path,
        readme_path,
    ]:
        print(
            f"  {path}"
        )

    if len(failure_df):
        print()
        print(
            "[STOP] One or more frozen slides failed measurement. "
            "Do NOT proceed to Step18c until failures are audited."
        )

        sys.exit(
            2
        )

    print()
    print(
        "[DONE] Step18b v1.0.2 FINAL memory-safe measurement-only run complete."
    )

    print(
        "[NEXT] Review measurement/QC counts only before Step18c."
    )


if __name__ == "__main__":
    main()
