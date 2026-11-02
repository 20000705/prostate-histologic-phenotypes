#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13d_make_pc3_morphology_review_montages.py

Purpose
-------
Generate deterministic, patient-balanced review montages for frozen PC3
morphology samples.

Primary groups:
    all_tissue_low
    all_tissue_high
    tumor_only_low
    tumor_only_high

Tumor-core sensitivity groups:
    tumor_core_low
    tumor_core_high

Important
---------
Primary groups reuse step-13b extracted images.

Tumor-core groups use the COMPLETE frozen step-13c manifest. If a tumor-core
sample does not already have an extracted image from step 13b, this script
extracts it directly from the source WSI into a dedicated tumor-core cache.

Therefore tumor-core montages are NOT restricted to the intersection between
13a and 13c.

This script does NOT:
    - redefine PC3 extremes
    - change tumor thresholds
    - reselect samples using morphology
    - replace images based on appearance
    - compute quantitative morphology statistics
    - alter steps 13a, 13b, or 13c

Inputs
------
Step 13a:
    results/13a_pc3_morphology_sampling_manifest/
        pc3_sampling_manifest.csv

Step 13b:
    results/13b_pc3_morphology_roi_extraction/
        roi_extraction_manifest.csv

Step 13c:
    results/13c_pc3_tumor_core_sensitivity_sampling/
        pc3_tumor_core_sensitivity_manifest.csv

Outputs
-------
results/13d_pc3_morphology_review_montages/

    montages/
        all_tissue_low.png
        all_tissue_high.png
        tumor_only_low.png
        tumor_only_high.png
        tumor_core_low.png
        tumor_core_high.png

    paired_montages/
        all_tissue_low_vs_high.png
        tumor_only_low_vs_high.png
        tumor_core_low_vs_high.png

    tumor_core_extracted/
        model_tiles/
        context_rois/
        annotated_context_rois/

    montage_selection_manifest.csv
    tumor_core_extraction_manifest.csv
    montage_summary.csv
    missing_images.csv
    analysis_metadata.json
    README.txt
"""

import json
import re
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = RESULTS_ROOT

IN_13A = (
    RESULTS_ROOT
    / "13a_pc3_morphology_sampling_manifest"
    / "pc3_sampling_manifest.csv"
)

IN_13B = (
    RESULTS_ROOT
    / "13b_pc3_morphology_roi_extraction"
    / "roi_extraction_manifest.csv"
)

IN_13C = (
    RESULTS_ROOT
    / "13c_pc3_tumor_core_sensitivity_sampling"
    / "pc3_tumor_core_sensitivity_manifest.csv"
)

WSI_DIRS = [DISCOVERY_WSI]

OUT_DIR = (
    RESULTS_ROOT
    / "13d_pc3_morphology_review_montages"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTAGE_DIR = OUT_DIR / "montages"
PAIRED_DIR = OUT_DIR / "paired_montages"

CORE_EXTRACT_DIR = OUT_DIR / "tumor_core_extracted"
CORE_MODEL_TILE_DIR = CORE_EXTRACT_DIR / "model_tiles"
CORE_CONTEXT_DIR = CORE_EXTRACT_DIR / "context_rois"
CORE_ANNOTATED_CONTEXT_DIR = CORE_EXTRACT_DIR / "annotated_context_rois"

for directory in [
    MONTAGE_DIR,
    PAIRED_DIR,
    CORE_MODEL_TILE_DIR,
    CORE_CONTEXT_DIR,
    CORE_ANNOTATED_CONTEXT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

OUT_SELECTION = OUT_DIR / "montage_selection_manifest.csv"
OUT_CORE_EXTRACTION = OUT_DIR / "tumor_core_extraction_manifest.csv"
OUT_SUMMARY = OUT_DIR / "montage_summary.csv"
OUT_MISSING = OUT_DIR / "missing_images.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"


# =============================================================================
# CONFIGURATION
# =============================================================================
IMAGE_MODE = "annotated_context"  # annotated_context | context | model_tile

CONTEXT_SIZE = 896
MODEL_BOX_WIDTH = 4

N_IMAGES_PER_STRATUM = 24
MAX_IMAGES_PER_PATIENT = 2
MAX_IMAGES_PER_SLIDE = 1

N_COLS = 6
FIG_DPI = 250

SHOW_SAMPLE_ID = True
SHOW_PATIENT_ID = True
SHOW_PREFIX = False
SHOW_PC3_SCORE = False
SHOW_TUMOR_OVERLAP = False

IMAGE_INTERPOLATION = "nearest"

SKIP_EXISTING_CORE_EXTRACTIONS = True

ALLOWED_WSI_SUFFIXES = (
    ".tif",
    ".tiff",
)

ROUND_DIGITS = 4


# =============================================================================
# BASIC HELPERS
# =============================================================================
def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def normalize_name(value: str) -> str:
    stem = Path(str(value)).stem.lower()
    return re.sub(
        r"[^a-z0-9]",
        "",
        stem,
    )


def build_wsi_map(
    directories: List[Path],
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}

    for directory in directories:
        if not directory.exists():
            continue

        for path in sorted(
            directory.iterdir()
        ):
            if (
                path.is_file()
                and path.suffix.lower()
                in ALLOWED_WSI_SUFFIXES
            ):
                mapping[
                    normalize_name(
                        path.name
                    )
                ] = path

    return mapping


def find_wsi_by_prefix(
    prefix: str,
    wsi_map: Dict[str, Path],
) -> Optional[Path]:
    key = normalize_name(
        prefix
    )

    if key in wsi_map:
        return wsi_map[
            key
        ]

    candidates = [
        path
        for normalized, path in wsi_map.items()
        if (
            key in normalized
            or normalized in key
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        candidates = sorted(
            candidates,
            key=lambda path: abs(
                len(
                    normalize_name(
                        path.name
                    )
                )
                - len(key)
            ),
        )
        return candidates[0]

    return None


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def resolve_wsi_path(
    row: pd.Series,
    wsi_map: Dict[str, Path],
) -> Optional[Path]:
    meta_path_raw = str(
        row.get(
            "meta_path",
            "",
        )
    ).strip()

    if meta_path_raw:
        meta_path = Path(
            meta_path_raw
        )

        if meta_path.exists():
            try:
                meta = load_json(
                    meta_path
                )

                slide_path_raw = str(
                    meta.get(
                        "slide_path",
                        "",
                    )
                ).strip()

                if slide_path_raw:
                    slide_path = Path(
                        slide_path_raw
                    )

                    if slide_path.exists():
                        return slide_path

            except Exception:
                pass

    return find_wsi_by_prefix(
        str(
            row["prefix"]
        ),
        wsi_map,
    )


def resolve_image_column(
    dataframe: pd.DataFrame,
) -> str:
    mapping = {
        "annotated_context":
            "annotated_context_roi_path",

        "context":
            "context_roi_path",

        "model_tile":
            "model_tile_path",
    }

    if IMAGE_MODE not in mapping:
        raise ValueError(
            f"Unknown IMAGE_MODE: {IMAGE_MODE}"
        )

    column = mapping[
        IMAGE_MODE
    ]

    if column not in dataframe.columns:
        raise ValueError(
            f"Required image column '{column}' "
            f"not found for IMAGE_MODE={IMAGE_MODE}"
        )

    return column


# =============================================================================
# IMAGE EXTRACTION FOR COMPLETE 13C TUMOR-CORE SET
# =============================================================================
def compute_context_origin(
    tile_x: int,
    tile_y: int,
    patch_size: int,
    context_size: int,
) -> Tuple[int, int]:
    center_x = (
        tile_x
        + patch_size / 2.0
    )

    center_y = (
        tile_y
        + patch_size / 2.0
    )

    context_x = int(
        round(
            center_x
            - context_size / 2.0
        )
    )

    context_y = int(
        round(
            center_y
            - context_size / 2.0
        )
    )

    return (
        context_x,
        context_y,
    )


def crop_with_white_padding(
    slide_img: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Image.Image:
    slide_width, slide_height = (
        slide_img.size
    )

    source_left = max(
        0,
        x,
    )

    source_top = max(
        0,
        y,
    )

    source_right = min(
        slide_width,
        x + width,
    )

    source_bottom = min(
        slide_height,
        y + height,
    )

    canvas = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "white",
    )

    if (
        source_right > source_left
        and source_bottom > source_top
    ):
        crop = (
            slide_img.crop(
                (
                    source_left,
                    source_top,
                    source_right,
                    source_bottom,
                )
            )
            .convert(
                "RGB"
            )
        )

        paste_x = (
            source_left
            - x
        )

        paste_y = (
            source_top
            - y
        )

        canvas.paste(
            crop,
            (
                paste_x,
                paste_y,
            ),
        )

    return canvas


def draw_model_tile_box(
    context_img: Image.Image,
    tile_x: int,
    tile_y: int,
    patch_size: int,
    context_x: int,
    context_y: int,
) -> Image.Image:
    annotated = (
        context_img.copy()
    )

    draw = ImageDraw.Draw(
        annotated
    )

    local_x0 = (
        tile_x
        - context_x
    )

    local_y0 = (
        tile_y
        - context_y
    )

    local_x1 = (
        local_x0
        + patch_size
    )

    local_y1 = (
        local_y0
        + patch_size
    )

    for offset in range(
        MODEL_BOX_WIDTH
    ):
        draw.rectangle(
            [
                local_x0 - offset,
                local_y0 - offset,
                local_x1 + offset,
                local_y1 + offset,
            ],
            outline=(
                0,
                0,
                255,
            ),
        )

    return annotated


def extract_tumor_core_sample(
    row: pd.Series,
    wsi_map: Dict[str, Path],
) -> Tuple[
    Optional[dict],
    Optional[dict],
]:
    sample_id = str(
        row.get(
            "sensitivity_sample_id",
            row.get(
                "sample_id",
                "",
            ),
        )
    )

    patient_id = str(
        row[
            "patient_id"
        ]
    )

    prefix = str(
        row[
            "prefix"
        ]
    )

    tile_idx = int(
        row[
            "tile_idx"
        ]
    )

    tile_x = int(
        row[
            "x"
        ]
    )

    tile_y = int(
        row[
            "y"
        ]
    )

    patch_size = int(
        row[
            "patch_size"
        ]
    )

    wsi_path = resolve_wsi_path(
        row,
        wsi_map,
    )

    if (
        wsi_path is None
        or not wsi_path.exists()
    ):
        return (
            None,
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "prefix": prefix,
                "sampling_stratum": row[
                    "sampling_stratum"
                ],
                "reason": "missing_wsi",
                "details": "",
            },
        )

    model_tile_path = (
        CORE_MODEL_TILE_DIR
        / f"{sample_id}.png"
    )

    context_path = (
        CORE_CONTEXT_DIR
        / f"{sample_id}_context.png"
    )

    annotated_context_path = (
        CORE_ANNOTATED_CONTEXT_DIR
        / (
            f"{sample_id}"
            "_context_with_model_box.png"
        )
    )

    if (
        SKIP_EXISTING_CORE_EXTRACTIONS
        and model_tile_path.exists()
        and context_path.exists()
        and annotated_context_path.exists()
    ):
        return (
            {
                "sample_id": sample_id,
                "sensitivity_sample_id": sample_id,
                "dataset": row.get(
                    "dataset",
                    "",
                ),
                "patient_id": patient_id,
                "prefix": prefix,
                "sampling_stratum": row[
                    "sampling_stratum"
                ],
                "tile_idx": tile_idx,
                "x": tile_x,
                "y": tile_y,
                "patch_size": patch_size,
                "pc3_score": row[
                    "pc3_score"
                ],
                "pc3_within_slide_percentile": row[
                    "pc3_within_slide_percentile"
                ],
                "tumor_overlap_fraction": row[
                    "tumor_overlap_fraction"
                ],
                "model_tile_path": str(
                    model_tile_path
                ),
                "context_roi_path": str(
                    context_path
                ),
                "annotated_context_roi_path": str(
                    annotated_context_path
                ),
                "wsi_path": str(
                    wsi_path
                ),
                "status": "skipped_existing",
            },
            None,
        )

    try:
        slide_img = (
            Image.open(
                wsi_path
            )
            .convert(
                "RGB"
            )
        )

    except Exception as error:
        return (
            None,
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "prefix": prefix,
                "sampling_stratum": row[
                    "sampling_stratum"
                ],
                "reason": "wsi_open_error",
                "details": repr(
                    error
                ),
            },
        )

    try:
        model_tile = (
            crop_with_white_padding(
                slide_img=slide_img,
                x=tile_x,
                y=tile_y,
                width=patch_size,
                height=patch_size,
            )
        )

        (
            context_x,
            context_y,
        ) = compute_context_origin(
            tile_x=tile_x,
            tile_y=tile_y,
            patch_size=patch_size,
            context_size=CONTEXT_SIZE,
        )

        context = (
            crop_with_white_padding(
                slide_img=slide_img,
                x=context_x,
                y=context_y,
                width=CONTEXT_SIZE,
                height=CONTEXT_SIZE,
            )
        )

        annotated = (
            draw_model_tile_box(
                context_img=context,
                tile_x=tile_x,
                tile_y=tile_y,
                patch_size=patch_size,
                context_x=context_x,
                context_y=context_y,
            )
        )

        model_tile.save(
            model_tile_path,
            format="PNG",
            optimize=True,
        )

        context.save(
            context_path,
            format="PNG",
            optimize=True,
        )

        annotated.save(
            annotated_context_path,
            format="PNG",
            optimize=True,
        )

        return (
            {
                "sample_id": sample_id,
                "sensitivity_sample_id": sample_id,
                "dataset": row.get(
                    "dataset",
                    "",
                ),
                "patient_id": patient_id,
                "prefix": prefix,
                "sampling_stratum": row[
                    "sampling_stratum"
                ],
                "tile_idx": tile_idx,
                "x": tile_x,
                "y": tile_y,
                "patch_size": patch_size,
                "pc3_score": row[
                    "pc3_score"
                ],
                "pc3_within_slide_percentile": row[
                    "pc3_within_slide_percentile"
                ],
                "tumor_overlap_fraction": row[
                    "tumor_overlap_fraction"
                ],
                "model_tile_path": str(
                    model_tile_path
                ),
                "context_roi_path": str(
                    context_path
                ),
                "annotated_context_roi_path": str(
                    annotated_context_path
                ),
                "wsi_path": str(
                    wsi_path
                ),
                "status": "processed",
            },
            None,
        )

    except Exception as error:
        return (
            None,
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "prefix": prefix,
                "sampling_stratum": row[
                    "sampling_stratum"
                ],
                "reason": "extraction_error",
                "details": repr(
                    error
                ),
            },
        )

    finally:
        slide_img.close()


def build_complete_tumor_core_extractions(
    sensitivity_manifest: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    core = sensitivity_manifest.loc[
        sensitivity_manifest[
            "sampling_stratum"
        ].isin(
            [
                "tumor_core_low",
                "tumor_core_high",
            ]
        )
    ].copy()

    if core.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    required = {
        "sensitivity_sample_id",
        "dataset",
        "patient_id",
        "prefix",
        "tile_idx",
        "x",
        "y",
        "patch_size",
        "pc3_score",
        "pc3_within_slide_percentile",
        "tumor_overlap_fraction",
        "sampling_stratum",
    }

    missing = (
        required
        - set(
            core.columns
        )
    )

    if missing:
        raise ValueError(
            "13c tumor-core manifest missing columns: "
            f"{sorted(missing)}"
        )

    wsi_map = build_wsi_map(
        WSI_DIRS
    )

    success_rows = []
    failed_rows = []

    print(
        "\n[INFO] Ensuring complete tumor-core ROI extraction..."
    )

    for index, row in core.reset_index(
        drop=True
    ).iterrows():
        result, failure = (
            extract_tumor_core_sample(
                row=row,
                wsi_map=wsi_map,
            )
        )

        if result is not None:
            success_rows.append(
                result
            )

        else:
            failed_rows.append(
                failure
            )

        if (
            index < 10
            or (
                index + 1
            ) % 100 == 0
            or (
                index + 1
            ) == len(core)
        ):
            status = (
                "OK"
                if result is not None
                else "FAIL"
            )

            print(
                f"[{status} "
                f"{index + 1:04d}/"
                f"{len(core):04d}] "
                f"{row['sensitivity_sample_id']} "
                f"| {row['sampling_stratum']}"
            )

    success = pd.DataFrame(
        success_rows
    )

    failed = pd.DataFrame(
        failed_rows
    )

    if not success.empty:
        success = (
            success.sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )

    return (
        success,
        failed,
    )


# =============================================================================
# LOAD PRIMARY 13A + 13B
# =============================================================================
def load_primary_extracted_manifest() -> pd.DataFrame:
    check_file(
        IN_13A
    )

    check_file(
        IN_13B
    )

    sampling = pd.read_csv(
        IN_13A
    )

    extraction = pd.read_csv(
        IN_13B
    )

    if (
        "sample_id"
        not in sampling.columns
    ):
        raise ValueError(
            "13a manifest missing sample_id."
        )

    if (
        "sample_id"
        not in extraction.columns
    ):
        raise ValueError(
            "13b manifest missing sample_id."
        )

    merged = sampling.merge(
        extraction,
        on="sample_id",
        how="left",
        suffixes=(
            "",
            "_13b",
        ),
        validate="many_to_one",
    )

    return merged


# =============================================================================
# MONTAGE SELECTION
# =============================================================================
def format_label(
    row: pd.Series,
) -> str:
    parts = []

    if SHOW_SAMPLE_ID:
        parts.append(
            str(
                row.get(
                    "sample_id",
                    "",
                )
            )
        )

    if SHOW_PATIENT_ID:
        parts.append(
            str(
                row.get(
                    "patient_id",
                    "",
                )
            )
        )

    if SHOW_PREFIX:
        parts.append(
            str(
                row.get(
                    "prefix",
                    "",
                )
            )
        )

    if (
        SHOW_PC3_SCORE
        and pd.notna(
            row.get(
                "pc3_score",
                np.nan,
            )
        )
    ):
        parts.append(
            f"PC3="
            f"{float(row['pc3_score']):.2f}"
        )

    if (
        SHOW_TUMOR_OVERLAP
        and pd.notna(
            row.get(
                "tumor_overlap_fraction",
                np.nan,
            )
        )
    ):
        parts.append(
            f"tumor="
            f"{float(row['tumor_overlap_fraction']):.2f}"
        )

    return "\n".join(
        parts
    )


def direction_for_stratum(
    stratum: str,
) -> str:
    if stratum.endswith(
        "_low"
    ):
        return "low"

    if stratum.endswith(
        "_high"
    ):
        return "high"

    raise ValueError(
        f"Cannot infer direction: {stratum}"
    )


def sort_for_review(
    dataframe: pd.DataFrame,
    stratum: str,
) -> pd.DataFrame:
    out = dataframe.copy()

    rank_columns = [
        column
        for column in [
            "rank_within_patient_for_stratum",
            "rank_within_slide_for_stratum",
        ]
        if column in out.columns
    ]

    if rank_columns:
        return out.sort_values(
            [
                "patient_id",
            ]
            + rank_columns
            + [
                "prefix",
                "tile_idx",
            ],
            ascending=True,
            kind="mergesort",
        )

    direction = direction_for_stratum(
        stratum
    )

    return out.sort_values(
        [
            "patient_id",
            "pc3_score",
            "prefix",
            "tile_idx",
        ],
        ascending=[
            True,
            direction == "low",
            True,
            True,
        ],
        kind="mergesort",
    )


def balanced_select(
    dataframe: pd.DataFrame,
    stratum: str,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    ranked = (
        sort_for_review(
            dataframe,
            stratum,
        )
        .reset_index(
            drop=False
        )
    )

    patients = sorted(
        ranked[
            "patient_id"
        ]
        .astype(str)
        .unique()
    )

    groups = {
        patient: (
            ranked.loc[
                ranked[
                    "patient_id"
                ].astype(str)
                == patient
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )
        for patient in patients
    }

    cursors = {
        patient: 0
        for patient in patients
    }

    patient_counts: Dict[
        str,
        int,
    ] = {}

    slide_counts: Dict[
        Tuple[
            str,
            str,
        ],
        int,
    ] = {}

    selected_original_indices = []

    while (
        len(
            selected_original_indices
        )
        < N_IMAGES_PER_STRATUM
    ):
        progress = False

        for patient in patients:
            if (
                len(
                    selected_original_indices
                )
                >= N_IMAGES_PER_STRATUM
            ):
                break

            if (
                patient_counts.get(
                    patient,
                    0,
                )
                >= MAX_IMAGES_PER_PATIENT
            ):
                continue

            group = groups[
                patient
            ]

            while (
                cursors[
                    patient
                ]
                < len(group)
            ):
                row = group.iloc[
                    cursors[
                        patient
                    ]
                ]

                cursors[
                    patient
                ] += 1

                prefix = str(
                    row[
                        "prefix"
                    ]
                )

                slide_key = (
                    patient,
                    prefix,
                )

                if (
                    slide_counts.get(
                        slide_key,
                        0,
                    )
                    >= MAX_IMAGES_PER_SLIDE
                ):
                    continue

                selected_original_indices.append(
                    int(
                        row[
                            "index"
                        ]
                    )
                )

                patient_counts[
                    patient
                ] = (
                    patient_counts.get(
                        patient,
                        0,
                    )
                    + 1
                )

                slide_counts[
                    slide_key
                ] = (
                    slide_counts.get(
                        slide_key,
                        0,
                    )
                    + 1
                )

                progress = True
                break

        if not progress:
            break

    selected = dataframe.loc[
        selected_original_indices
    ].copy()

    selected = (
        sort_for_review(
            selected,
            stratum,
        )
        .reset_index(
            drop=True
        )
    )

    selected[
        "montage_rank"
    ] = np.arange(
        1,
        len(selected)
        + 1,
    )

    selected[
        "montage_stratum"
    ] = stratum

    return selected


def split_available_missing(
    dataframe: pd.DataFrame,
    image_column: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    available_rows = []
    missing_rows = []

    for _, row in dataframe.iterrows():
        raw = str(
            row.get(
                image_column,
                "",
            )
        ).strip()

        path = (
            Path(raw)
            if raw
            else None
        )

        if (
            path is not None
            and path.exists()
        ):
            available_rows.append(
                row
            )

        else:
            missing_rows.append({
                "sample_id": row.get(
                    "sample_id",
                    "",
                ),
                "patient_id": row.get(
                    "patient_id",
                    "",
                ),
                "prefix": row.get(
                    "prefix",
                    "",
                ),
                "sampling_stratum": row.get(
                    "sampling_stratum",
                    "",
                ),
                "image_column": image_column,
                "image_path": raw,
                "reason": "missing_image",
            })

    return (
        pd.DataFrame(
            available_rows
        ),
        pd.DataFrame(
            missing_rows
        ),
    )


# =============================================================================
# MONTAGE DRAWING
# =============================================================================
def load_rgb(
    path: Path,
) -> Image.Image:
    return (
        Image.open(
            path
        )
        .convert(
            "RGB"
        )
    )


def make_single_montage(
    selected: pd.DataFrame,
    image_column: str,
    stratum: str,
    out_path: Path,
) -> None:
    if selected.empty:
        return

    n_images = len(
        selected
    )

    n_cols = min(
        N_COLS,
        n_images,
    )

    n_rows = int(
        np.ceil(
            n_images
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            n_cols * 3.2,
            n_rows * 3.5 + 0.6,
        ),
        squeeze=False,
    )

    for ax in axes.flatten():
        ax.axis(
            "off"
        )

    for index, (
        _,
        row,
    ) in enumerate(
        selected.iterrows()
    ):
        ax = axes.flatten()[
            index
        ]

        image = load_rgb(
            Path(
                str(
                    row[
                        image_column
                    ]
                )
            )
        )

        ax.imshow(
            image,
            interpolation=(
                IMAGE_INTERPOLATION
            ),
        )

        ax.axis(
            "off"
        )

        label = format_label(
            row
        )

        if label:
            ax.set_title(
                label,
                fontsize=8,
                pad=4,
            )

    fig.suptitle(
        f"PC3 morphology review: "
        f"{stratum}",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def make_paired_montage(
    low_df: pd.DataFrame,
    high_df: pd.DataFrame,
    image_column: str,
    label: str,
    out_path: Path,
) -> None:
    n_each = min(
        len(low_df),
        len(high_df),
        N_IMAGES_PER_STRATUM,
    )

    if n_each == 0:
        return

    low_df = (
        low_df.head(
            n_each
        )
        .reset_index(
            drop=True
        )
    )

    high_df = (
        high_df.head(
            n_each
        )
        .reset_index(
            drop=True
        )
    )

    n_cols = min(
        6,
        n_each,
    )

    blocks = int(
        np.ceil(
            n_each
            / n_cols
        )
    )

    n_rows = (
        blocks
        * 2
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            n_cols * 3.2,
            n_rows * 3.3 + 0.8,
        ),
        squeeze=False,
    )

    for ax in axes.flatten():
        ax.axis(
            "off"
        )

    for i in range(
        n_each
    ):
        block = (
            i
            // n_cols
        )

        col = (
            i
            % n_cols
        )

        low_ax = axes[
            block * 2,
            col,
        ]

        high_ax = axes[
            block * 2 + 1,
            col,
        ]

        low_row = (
            low_df.iloc[
                i
            ]
        )

        high_row = (
            high_df.iloc[
                i
            ]
        )

        low_ax.imshow(
            load_rgb(
                Path(
                    str(
                        low_row[
                            image_column
                        ]
                    )
                )
            ),
            interpolation=(
                IMAGE_INTERPOLATION
            ),
        )

        high_ax.imshow(
            load_rgb(
                Path(
                    str(
                        high_row[
                            image_column
                        ]
                    )
                )
            ),
            interpolation=(
                IMAGE_INTERPOLATION
            ),
        )

        low_ax.axis(
            "off"
        )

        high_ax.axis(
            "off"
        )

        low_label = (
            format_label(
                low_row
            )
        )

        high_label = (
            format_label(
                high_row
            )
        )

        if low_label:
            low_ax.set_title(
                low_label,
                fontsize=7,
                pad=3,
            )

        if high_label:
            high_ax.set_title(
                high_label,
                fontsize=7,
                pad=3,
            )

        if col == 0:
            low_ax.text(
                -0.04,
                0.5,
                "LOW",
                transform=(
                    low_ax.transAxes
                ),
                rotation=90,
                ha="right",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

            high_ax.text(
                -0.04,
                0.5,
                "HIGH",
                transform=(
                    high_ax.transAxes
                ),
                rotation=90,
                ha="right",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    fig.suptitle(
        f"PC3 morphology review: "
        f"{label}",
        fontsize=18,
        fontweight="bold",
        y=0.997,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.975,
        ]
    )

    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    check_file(
        IN_13A
    )

    check_file(
        IN_13B
    )

    check_file(
        IN_13C
    )

    print(
        "=" * 110
    )

    print(
        "PC3 MORPHOLOGY REVIEW MONTAGES "
        "(COMPLETE TUMOR-CORE VERSION)"
    )

    print(
        "=" * 110
    )

    print(
        f"[STEP 13A]          "
        f"{IN_13A}"
    )

    print(
        f"[STEP 13B]          "
        f"{IN_13B}"
    )

    print(
        f"[STEP 13C]          "
        f"{IN_13C}"
    )

    print(
        f"[IMAGE MODE]        "
        f"{IMAGE_MODE}"
    )

    print(
        f"[N/STRATUM]         "
        f"{N_IMAGES_PER_STRATUM}"
    )

    print(
        f"[MAX/PATIENT]       "
        f"{MAX_IMAGES_PER_PATIENT}"
    )

    print(
        f"[MAX/SLIDE]         "
        f"{MAX_IMAGES_PER_SLIDE}"
    )

    print(
        f"[OUTPUT]            "
        f"{OUT_DIR}"
    )

    primary = (
        load_primary_extracted_manifest()
    )

    primary_image_column = (
        resolve_image_column(
            primary
        )
    )

    sensitivity = pd.read_csv(
        IN_13C
    )

    (
        core_extracted,
        core_failed,
    ) = (
        build_complete_tumor_core_extractions(
            sensitivity
        )
    )

    core_extracted.to_csv(
        OUT_CORE_EXTRACTION,
        index=False,
    )

    selected_tables = []
    missing_tables = []

    # -------------------------------------------------------------------------
    # PRIMARY 13A/13B GROUPS
    # -------------------------------------------------------------------------
    for stratum in [
        "all_tissue_low",
        "all_tissue_high",
        "tumor_only_low",
        "tumor_only_high",
    ]:
        group = primary.loc[
            primary[
                "sampling_stratum"
            ] == stratum
        ].copy()

        (
            available,
            missing,
        ) = split_available_missing(
            group,
            primary_image_column,
        )

        if not missing.empty:
            missing_tables.append(
                missing
            )

        selected = (
            balanced_select(
                available,
                stratum,
            )
        )

        if selected.empty:
            print(
                f"[WARNING] No selectable images: "
                f"{stratum}"
            )
            continue

        selected[
            "montage_image_path"
        ] = selected[
            primary_image_column
        ]

        selected_tables.append(
            selected
        )

        make_single_montage(
            selected=selected,
            image_column=(
                "montage_image_path"
            ),
            stratum=stratum,
            out_path=(
                MONTAGE_DIR
                / f"{stratum}.png"
            ),
        )

        print(
            f"[OK] {stratum}: "
            f"{len(selected)} images | "
            f"{selected['patient_id'].nunique()} patients | "
            f"{selected['prefix'].nunique()} slides"
        )

    # -------------------------------------------------------------------------
    # COMPLETE 13C TUMOR-CORE GROUPS
    # -------------------------------------------------------------------------
    if not core_extracted.empty:
        core_image_column = (
            resolve_image_column(
                core_extracted
            )
        )

        for stratum in [
            "tumor_core_low",
            "tumor_core_high",
        ]:
            group = (
                core_extracted.loc[
                    core_extracted[
                        "sampling_stratum"
                    ] == stratum
                ]
                .copy()
            )

            (
                available,
                missing,
            ) = split_available_missing(
                group,
                core_image_column,
            )

            if not missing.empty:
                missing_tables.append(
                    missing
                )

            selected = (
                balanced_select(
                    available,
                    stratum,
                )
            )

            if selected.empty:
                print(
                    f"[WARNING] No selectable images: "
                    f"{stratum}"
                )
                continue

            selected[
                "montage_image_path"
            ] = selected[
                core_image_column
            ]

            selected_tables.append(
                selected
            )

            make_single_montage(
                selected=selected,
                image_column=(
                    "montage_image_path"
                ),
                stratum=stratum,
                out_path=(
                    MONTAGE_DIR
                    / f"{stratum}.png"
                ),
            )

            print(
                f"[OK] {stratum}: "
                f"{len(selected)} images | "
                f"{selected['patient_id'].nunique()} patients | "
                f"{selected['prefix'].nunique()} slides"
            )

    if not selected_tables:
        raise RuntimeError(
            "No montage selections generated."
        )

    selection = pd.concat(
        selected_tables,
        ignore_index=True,
    )

    selection.to_csv(
        OUT_SELECTION,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Missing / failed
    # -------------------------------------------------------------------------
    if not core_failed.empty:
        core_failed_for_missing = (
            core_failed.copy()
        )

        core_failed_for_missing[
            "image_column"
        ] = IMAGE_MODE

        core_failed_for_missing[
            "image_path"
        ] = ""

        missing_tables.append(
            core_failed_for_missing
        )

    if missing_tables:
        missing_df = pd.concat(
            missing_tables,
            ignore_index=True,
            sort=False,
        )
    else:
        missing_df = pd.DataFrame(
            columns=[
                "sample_id",
                "patient_id",
                "prefix",
                "sampling_stratum",
                "image_column",
                "image_path",
                "reason",
                "details",
            ]
        )

    missing_df.to_csv(
        OUT_MISSING,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary = (
        selection.groupby(
            "montage_stratum",
            as_index=False,
        )
        .agg(
            n_images=(
                "sample_id",
                "count",
            ),
            n_patients=(
                "patient_id",
                "nunique",
            ),
            n_slides=(
                "prefix",
                "nunique",
            ),
            median_pc3_score=(
                "pc3_score",
                "median",
            ),
            median_tumor_overlap_fraction=(
                "tumor_overlap_fraction",
                "median",
            ),
        )
    )

    numeric_columns = (
        summary
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    summary[
        numeric_columns
    ] = (
        summary[
            numeric_columns
        ]
        .round(
            ROUND_DIGITS
        )
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Paired montages
    # -------------------------------------------------------------------------
    by_stratum = {
        name: (
            selection.loc[
                selection[
                    "montage_stratum"
                ] == name
            ]
            .copy()
        )
        for name in selection[
            "montage_stratum"
        ].unique()
    }

    for (
        label,
        low_name,
        high_name,
    ) in [
        (
            "all_tissue",
            "all_tissue_low",
            "all_tissue_high",
        ),
        (
            "tumor_only",
            "tumor_only_low",
            "tumor_only_high",
        ),
        (
            "tumor_core",
            "tumor_core_low",
            "tumor_core_high",
        ),
    ]:
        if (
            low_name in by_stratum
            and high_name in by_stratum
        ):
            make_paired_montage(
                low_df=(
                    by_stratum[
                        low_name
                    ]
                ),
                high_df=(
                    by_stratum[
                        high_name
                    ]
                ),
                image_column=(
                    "montage_image_path"
                ),
                label=(
                    f"{label}: low vs high"
                ),
                out_path=(
                    PAIRED_DIR
                    / (
                        f"{label}"
                        "_low_vs_high.png"
                    )
                ),
            )

    # -------------------------------------------------------------------------
    # Metadata / README
    # -------------------------------------------------------------------------
    metadata = {
        "analysis_name": (
            "PC3 morphology review montages "
            "with complete tumor-core extraction"
        ),

        "analysis_role": (
            "Deterministic patient-balanced visual review "
            "of frozen PC3 morphology samples"
        ),

        "image_mode": (
            IMAGE_MODE
        ),

        "context_size": int(
            CONTEXT_SIZE
        ),

        "n_images_per_stratum": int(
            N_IMAGES_PER_STRATUM
        ),

        "max_images_per_patient": int(
            MAX_IMAGES_PER_PATIENT
        ),

        "max_images_per_slide": int(
            MAX_IMAGES_PER_SLIDE
        ),

        "primary_groups_source": (
            "Step 13a frozen manifest + step 13b extracted ROIs"
        ),

        "tumor_core_groups_source": (
            "Complete step 13c frozen tumor-core samples; "
            "missing ROIs extracted directly from source WSI"
        ),

        "tumor_core_restricted_to_13a_intersection": False,

        "selection_uses_image_content": False,

        "selection_changes_frozen_sampling": False,

        "morphology_statistics_performed": False,

        "n_complete_tumor_core_extractions": int(
            len(
                core_extracted
            )
        ),

        "n_failed_tumor_core_extractions": int(
            len(
                core_failed
            )
        ),

        "outputs": {
            "montage_directory": str(
                MONTAGE_DIR
            ),

            "paired_montage_directory": str(
                PAIRED_DIR
            ),

            "tumor_core_extraction_manifest": str(
                OUT_CORE_EXTRACTION
            ),

            "tumor_core_extraction_root": str(
                CORE_EXTRACT_DIR
            ),

            "montage_selection_manifest": str(
                OUT_SELECTION
            ),

            "summary": str(
                OUT_SUMMARY
            ),

            "missing_images": str(
                OUT_MISSING
            ),

            "readme": str(
                OUT_README
            ),
        },
    }

    with OUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    OUT_README.write_text(
        "\n".join(
            [
                "PC3 Morphology Review Montages",
                "==============================",
                "",
                "Primary groups",
                "--------------",
                (
                    "all_tissue and tumor_only groups use frozen "
                    "step-13a samples and step-13b ROI extractions."
                ),
                "",
                "Tumor-core groups",
                "-----------------",
                (
                    "tumor_core_low/high use the COMPLETE frozen "
                    "step-13c sample set."
                ),
                (
                    "ROIs missing from step 13b are extracted directly "
                    "from the source WSI into tumor_core_extracted/."
                ),
                (
                    "Therefore tumor-core review is not restricted to "
                    "the 13a/13c intersection."
                ),
                "",
                "Selection",
                "---------",
                (
                    "Montage selection is deterministic and balanced "
                    "by patient and slide."
                ),
                (
                    "Image appearance is never used for selection."
                ),
                "",
                "Summary",
                "-------",
                summary.to_string(
                    index=False
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "MONTAGE SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nTumor-core extraction:"
    )

    print(
        f"  Successful: "
        f"{len(core_extracted)}"
    )

    print(
        f"  Failed:     "
        f"{len(core_failed)}"
    )

    print(
        "\n[OK] Saved:"
    )

    for path in [
        OUT_SELECTION,
        OUT_CORE_EXTRACTION,
        OUT_SUMMARY,
        OUT_MISSING,
        OUT_METADATA,
        OUT_README,
    ]:
        print(
            f"  {path}"
        )

    print(
        f"  {MONTAGE_DIR}"
    )

    print(
        f"  {PAIRED_DIR}"
    )

    print(
        f"  {CORE_EXTRACT_DIR}"
    )

    print(
        "\n[DONE] Complete PC3 morphology "
        "review montage generation finished."
    )


if __name__ == "__main__":
    main()
