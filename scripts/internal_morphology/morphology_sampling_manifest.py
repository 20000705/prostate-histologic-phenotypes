#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13a_build_pc3_morphology_sampling_manifest.py

Purpose
-------
Build a prespecified, patient-balanced, tumor-aware sampling manifest for PC3
morphology characterization.

This script intentionally separates SAMPLE DEFINITION from IMAGE EXTRACTION.

It does NOT:
    - crop H&E ROIs
    - make montages
    - compute morphology metrics
    - inspect pathology labels beyond tumor annotation overlap
    - use morphology to choose samples
    - use external cohorts
    - use PC6

The output manifest is intended to become the frozen source of truth for:
    - PC3 H&E ROI extraction
    - quantitative morphology
    - tumor-only characterization
    - blinded pathology review

Cohort
------
Slides are collected from three embedding directories with deterministic
priority-based deduplication:

    1. wsi_embeddings
    2. wsi_embeddings_single_slide
    3. wsi_embeddings_two_patients

Tumor definition
----------------
Tumor overlap fraction is computed as:

    intersection(tile, union_of_tumor_polygons).area / tile.area

A tile is considered tumor-associated when:

    tumor_overlap_fraction >= TUMOR_OVERLAP_THRESHOLD

PC3 extremes
------------
PC3 extremes are defined WITHIN SLIDE using percentile thresholds:

    low extreme  = PC3 percentile <= LOW_PERCENTILE
    high extreme = PC3 percentile >= HIGH_PERCENTILE

This avoids selecting only slides/patients with globally shifted PC3 score
distributions.

Balanced sampling
-----------------
The script samples separately for four analysis strata:

    all_tissue_low
    all_tissue_high
    tumor_only_low
    tumor_only_high

Sampling is deterministic.

Within each stratum:
    - each patient has a maximum contribution
    - each slide has a maximum contribution
    - lower/higher PC3 extremes are prioritized appropriately
    - no morphology information is used

Outputs
-------
MedIA_paper/results/13a_pc3_morphology_sampling_manifest/

    all_pc3_candidate_tiles.csv.gz
    pc3_sampling_manifest.csv
    pc3_sampling_summary.csv
    slide_sampling_summary.csv
    patient_sampling_summary.csv
    duplicate_manifest.csv
    excluded_slides.csv
    analysis_metadata.json
    README.txt
"""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
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
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = RESULTS_ROOT

BASE = DATA_ROOT / 'discovery'

DATASET_PRIORITY = [
    (
        "wsi_embeddings",
        BASE / "wsi_embeddings",
    ),
    (
        "wsi_embeddings_single_slide",
        BASE / "wsi_embeddings_single_slide",
    ),
    (
        "wsi_embeddings_two_patients",
        BASE / "wsi_embeddings_two_patients",
    ),
]

ANNOT_DIR = DISCOVERY_ANNOTATIONS

OUT_DIR = (
    RESULTS_ROOT
    / "13a_pc3_morphology_sampling_manifest"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_ALL_CANDIDATES = (
    OUT_DIR
    / "all_pc3_candidate_tiles.csv.gz"
)

OUT_SAMPLING_MANIFEST = (
    OUT_DIR
    / "pc3_sampling_manifest.csv"
)

OUT_SAMPLING_SUMMARY = (
    OUT_DIR
    / "pc3_sampling_summary.csv"
)

OUT_SLIDE_SUMMARY = (
    OUT_DIR
    / "slide_sampling_summary.csv"
)

OUT_PATIENT_SUMMARY = (
    OUT_DIR
    / "patient_sampling_summary.csv"
)

OUT_DUPLICATES = (
    OUT_DIR
    / "duplicate_manifest.csv"
)

OUT_EXCLUDED = (
    OUT_DIR
    / "excluded_slides.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "analysis_metadata.json"
)

OUT_README = (
    OUT_DIR
    / "README.txt"
)


# =============================================================================
# CONFIGURATION
# =============================================================================
PC3_INDEX = 2
PATCH_SIZE_DEFAULT = 224

# Use the same tumor-associated tile rule used upstream.
TUMOR_OVERLAP_THRESHOLD = 0.10

# Slide-wise extreme definition.
LOW_PERCENTILE = 0.10
HIGH_PERCENTILE = 0.90

# Sampling caps per stratum.
MAX_TILES_PER_SLIDE_PER_STRATUM = 10
MAX_TILES_PER_PATIENT_PER_STRATUM = 40

# Minimum number of candidates required from a slide before that slide can
# contribute to a given stratum.
MIN_CANDIDATES_PER_SLIDE_PER_STRATUM = 1

# Optional non-overlap constraint among sampled model tiles.
# False is recommended because embedding tiles are already non-overlapping
# in the original extraction pipeline.
ENFORCE_NO_DUPLICATE_TILE_IDS = True

RANDOM_STATE = 42
ROUND_DIGITS = 6
EPS = 1e-12


# =============================================================================
# HELPERS
# =============================================================================
def normalize_name(value: str) -> str:
    stem = Path(
        str(value)
    ).stem.lower()

    return re.sub(
        r"[^a-z0-9]",
        "",
        stem,
    )


def canonical_prefix(
    prefix: str,
) -> str:
    return normalize_name(
        prefix
    )


def build_file_map(
    directory: Path,
    suffixes,
) -> Dict[str, Path]:
    mapping = {}

    if not directory.exists():
        return mapping

    for path in sorted(
        directory.iterdir()
    ):
        if (
            path.is_file()
            and path.suffix.lower()
            in suffixes
        ):
            mapping[
                normalize_name(
                    path.name
                )
            ] = path

    return mapping


def find_file_by_prefix(
    prefix: str,
    file_map: Dict[str, Path],
) -> Optional[Path]:
    key = normalize_name(
        prefix
    )

    if key in file_map:
        return file_map[
            key
        ]

    candidates = [
        path
        for normalized, path
        in file_map.items()
        if (
            key in normalized
            or normalized in key
        )
    ]

    if len(
        candidates
    ) == 1:
        return candidates[
            0
        ]

    if len(
        candidates
    ) > 1:
        candidates = sorted(
            candidates,
            key=lambda path: abs(
                len(
                    normalize_name(
                        path.name
                    )
                )
                - len(
                    key
                )
            ),
        )

        return candidates[
            0
        ]

    return None


def find_pca_dir(
    emb_dir: Path,
) -> Optional[Path]:
    for name in [
        "pca_global_projected_from_main",
        "pca_global",
        "pca",
    ]:
        candidate = (
            emb_dir
            / name
        )

        if candidate.exists():
            return candidate

    return None


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


# =============================================================================
# ANNOTATION
# =============================================================================
def load_polygons(
    xml_path: Optional[Path],
) -> List[Polygon]:
    polygons = []

    if (
        xml_path is None
        or not xml_path.exists()
    ):
        return polygons

    tree = ET.parse(
        xml_path
    )

    root = (
        tree.getroot()
    )

    for region in root.iter(
        "Region"
    ):
        points = []

        for vertex in region.iter(
            "Vertex"
        ):
            if (
                "X" in vertex.attrib
                and "Y" in vertex.attrib
            ):
                points.append(
                    (
                        float(
                            vertex.attrib[
                                "X"
                            ]
                        ),
                        float(
                            vertex.attrib[
                                "Y"
                            ]
                        ),
                    )
                )

        if len(
            points
        ) < 3:
            continue

        polygon = Polygon(
            points
        )

        if not polygon.is_valid:
            polygon = (
                polygon.buffer(
                    0
                )
            )

        if (
            polygon.is_valid
            and polygon.area
            > 0
        ):
            polygons.append(
                polygon
            )

    return polygons


def compute_tumor_overlap_fractions(
    coords: np.ndarray,
    patch_size: int,
    polygons: List[Polygon],
) -> np.ndarray:
    overlaps = np.zeros(
        len(
            coords
        ),
        dtype=np.float32,
    )

    if not polygons:
        return overlaps

    tumor_union = unary_union(
        polygons
    )

    tile_area = float(
        patch_size
        * patch_size
    )

    for index, (
        x,
        y,
    ) in enumerate(
        coords
    ):
        tile_box = box(
            float(
                x
            ),
            float(
                y
            ),
            float(
                x
            )
            + patch_size,
            float(
                y
            )
            + patch_size,
        )

        overlap_area = (
            tumor_union
            .intersection(
                tile_box
            )
            .area
        )

        overlaps[
            index
        ] = float(
            overlap_area
            / tile_area
        )

    return overlaps


# =============================================================================
# DEDUPLICATION
# =============================================================================
def collect_unique_entries() -> Tuple[
    List[dict],
    pd.DataFrame,
]:
    selected = {}
    manifest_rows = []

    for priority, (
        dataset_name,
        emb_dir,
    ) in enumerate(
        DATASET_PRIORITY
    ):
        if not emb_dir.exists():
            print(
                f"[SKIP DATASET] Missing: "
                f"{emb_dir}"
            )
            continue

        pca_dir = find_pca_dir(
            emb_dir
        )

        if pca_dir is None:
            print(
                f"[SKIP DATASET] No PCA dir: "
                f"{emb_dir}"
            )
            continue

        for coord_path in sorted(
            emb_dir.glob(
                "*_coords.npy"
            )
        ):
            prefix = (
                coord_path.name
                .replace(
                    "_coords.npy",
                    "",
                )
            )

            key = canonical_prefix(
                prefix
            )

            meta_path = (
                emb_dir
                / f"{prefix}_meta.json"
            )

            pca_path = (
                pca_dir
                / f"{prefix}_pca_targets.npy"
            )

            candidate = {
                "dedup_key": (
                    key
                ),
                "dataset": (
                    dataset_name
                ),
                "priority": (
                    priority
                ),
                "prefix": (
                    prefix
                ),
                "patient_id": (
                    prefix.split(
                        "_"
                    )[0]
                ),
                "emb_dir": (
                    emb_dir
                ),
                "pca_dir": (
                    pca_dir
                ),
                "coord_path": (
                    coord_path
                ),
                "meta_path": (
                    meta_path
                ),
                "pca_path": (
                    pca_path
                ),
            }

            if key not in selected:
                selected[
                    key
                ] = candidate

                status = (
                    "kept"
                )
            else:
                kept = selected[
                    key
                ]

                status = (
                    "duplicate_skipped_"
                    f"kept_{kept['dataset']}"
                )

                print(
                    f"[DUPLICATE SKIP] "
                    f"{prefix} from "
                    f"{dataset_name} | "
                    f"kept "
                    f"{kept['prefix']} "
                    f"from "
                    f"{kept['dataset']}"
                )

            manifest_rows.append({
                "canonical_key": (
                    key
                ),
                "prefix": (
                    prefix
                ),
                "patient_id": (
                    candidate[
                        "patient_id"
                    ]
                ),
                "dataset": (
                    dataset_name
                ),
                "priority": (
                    priority
                ),
                "status": (
                    status
                ),
                "coord_path": str(
                    coord_path
                ),
                "meta_path": str(
                    meta_path
                ),
                "pca_path": str(
                    pca_path
                ),
            })

    duplicate_manifest = (
        pd.DataFrame(
            manifest_rows
        )
    )

    selected_entries = sorted(
        selected.values(),
        key=lambda item: (
            item[
                "patient_id"
            ],
            item[
                "prefix"
            ],
            item[
                "priority"
            ],
        ),
    )

    return (
        selected_entries,
        duplicate_manifest,
    )


# =============================================================================
# SLIDE CANDIDATES
# =============================================================================
def build_slide_candidate_table(
    entry: dict,
    xml_map: Dict[str, Path],
) -> Tuple[
    Optional[
        pd.DataFrame
    ],
    Optional[
        dict
    ],
]:
    prefix = entry[
        "prefix"
    ]

    patient_id = entry[
        "patient_id"
    ]

    coord_path = entry[
        "coord_path"
    ]

    meta_path = entry[
        "meta_path"
    ]

    pca_path = entry[
        "pca_path"
    ]

    if not meta_path.exists():
        return (
            None,
            {
                "patient_id": (
                    patient_id
                ),
                "prefix": (
                    prefix
                ),
                "reason": (
                    "missing_meta"
                ),
                "details": str(
                    meta_path
                ),
            },
        )

    if not pca_path.exists():
        return (
            None,
            {
                "patient_id": (
                    patient_id
                ),
                "prefix": (
                    prefix
                ),
                "reason": (
                    "missing_pca"
                ),
                "details": str(
                    pca_path
                ),
            },
        )

    coords = np.load(
        coord_path,
        mmap_mode="r",
    )

    pca = np.load(
        pca_path,
        mmap_mode="r",
    )

    if (
        pca.ndim != 2
        or pca.shape[
            1
        ]
        <= PC3_INDEX
    ):
        return (
            None,
            {
                "patient_id": (
                    patient_id
                ),
                "prefix": (
                    prefix
                ),
                "reason": (
                    "invalid_pca_shape"
                ),
                "details": str(
                    pca.shape
                ),
            },
        )

    if (
        coords.shape[
            0
        ]
        != pca.shape[
            0
        ]
    ):
        return (
            None,
            {
                "patient_id": (
                    patient_id
                ),
                "prefix": (
                    prefix
                ),
                "reason": (
                    "coords_pca_mismatch"
                ),
                "details": (
                    f"coords={coords.shape};"
                    f"pca={pca.shape}"
                ),
            },
        )

    meta = load_json(
        meta_path
    )

    patch_size = int(
        meta.get(
            "patch_size",
            PATCH_SIZE_DEFAULT,
        )
    )

    xml_path = find_file_by_prefix(
        prefix,
        xml_map,
    )

    polygons = []

    annotation_status = (
        "missing"
    )

    if (
        xml_path is not None
        and xml_path.exists()
    ):
        try:
            polygons = (
                load_polygons(
                    xml_path
                )
            )

            annotation_status = (
                "available"
                if len(
                    polygons
                ) > 0
                else "no_valid_polygons"
            )

        except Exception as error:
            return (
                None,
                {
                    "patient_id": (
                        patient_id
                    ),
                    "prefix": (
                        prefix
                    ),
                    "reason": (
                        "xml_parse_error"
                    ),
                    "details": repr(
                        error
                    ),
                },
            )

    overlaps = (
        compute_tumor_overlap_fractions(
            coords=np.asarray(
                coords
            ),
            patch_size=(
                patch_size
            ),
            polygons=(
                polygons
            ),
        )
    )

    scores = np.asarray(
        pca[
            :,
            PC3_INDEX,
        ],
        dtype=np.float64,
    )

    valid = np.isfinite(
        scores
    )

    if valid.sum() == 0:
        return (
            None,
            {
                "patient_id": (
                    patient_id
                ),
                "prefix": (
                    prefix
                ),
                "reason": (
                    "no_finite_pc3_scores"
                ),
                "details": "",
            },
        )

    ranks = np.full(
        len(
            scores
        ),
        np.nan,
        dtype=np.float64,
    )

    valid_scores = scores[
        valid
    ]

    order = np.argsort(
        valid_scores,
        kind="mergesort",
    )

    ordinal = np.empty(
        len(
            valid_scores
        ),
        dtype=np.int64,
    )

    ordinal[
        order
    ] = np.arange(
        len(
            valid_scores
        )
    )

    if len(
        valid_scores
    ) == 1:
        percentile_values = np.array(
            [
                0.5
            ],
            dtype=np.float64,
        )
    else:
        percentile_values = (
            ordinal.astype(
                np.float64
            )
            / (
                len(
                    valid_scores
                )
                - 1
            )
        )

    ranks[
        np.where(
            valid
        )[0]
    ] = (
        percentile_values
    )

    low_extreme = (
        valid
        & (
            ranks
            <= LOW_PERCENTILE
        )
    )

    high_extreme = (
        valid
        & (
            ranks
            >= HIGH_PERCENTILE
        )
    )

    tumor_associated = (
        overlaps
        >= TUMOR_OVERLAP_THRESHOLD
    )

    rows = pd.DataFrame({
        "dataset": (
            entry[
                "dataset"
            ]
        ),
        "patient_id": (
            patient_id
        ),
        "prefix": (
            prefix
        ),
        "dedup_key": (
            entry[
                "dedup_key"
            ]
        ),
        "tile_idx": np.arange(
            len(
                scores
            ),
            dtype=np.int64,
        ),
        "x": np.asarray(
            coords[
                :,
                0
            ],
            dtype=np.int64,
        ),
        "y": np.asarray(
            coords[
                :,
                1
            ],
            dtype=np.int64,
        ),
        "patch_size": (
            patch_size
        ),
        "pc3_score": (
            scores
        ),
        "pc3_within_slide_percentile": (
            ranks
        ),
        "is_pc3_low_extreme": (
            low_extreme
        ),
        "is_pc3_high_extreme": (
            high_extreme
        ),
        "tumor_overlap_fraction": (
            overlaps
        ),
        "is_tumor_associated": (
            tumor_associated
        ),
        "annotation_status": (
            annotation_status
        ),
        "xml_path": (
            str(
                xml_path
            )
            if xml_path is not None
            else ""
        ),
        "coord_path": str(
            coord_path
        ),
        "pca_path": str(
            pca_path
        ),
        "meta_path": str(
            meta_path
        ),
    })

    return (
        rows,
        None,
    )


# =============================================================================
# SAMPLING
# =============================================================================
def candidate_subset(
    all_candidates: pd.DataFrame,
    stratum: str,
) -> pd.DataFrame:
    if (
        stratum
        == "all_tissue_low"
    ):
        mask = (
            all_candidates[
                "is_pc3_low_extreme"
            ]
        )

    elif (
        stratum
        == "all_tissue_high"
    ):
        mask = (
            all_candidates[
                "is_pc3_high_extreme"
            ]
        )

    elif (
        stratum
        == "tumor_only_low"
    ):
        mask = (
            all_candidates[
                "is_pc3_low_extreme"
            ]
            & all_candidates[
                "is_tumor_associated"
            ]
        )

    elif (
        stratum
        == "tumor_only_high"
    ):
        mask = (
            all_candidates[
                "is_pc3_high_extreme"
            ]
            & all_candidates[
                "is_tumor_associated"
            ]
        )

    else:
        raise ValueError(
            f"Unknown stratum: {stratum}"
        )

    subset = (
        all_candidates.loc[
            mask
        ]
        .copy()
    )

    return subset


def deterministic_rank(
    dataframe: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    out = dataframe.copy()

    if direction == "low":
        out = out.sort_values(
            [
                "patient_id",
                "prefix",
                "pc3_score",
                "tile_idx",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
        )

    elif direction == "high":
        out = out.sort_values(
            [
                "patient_id",
                "prefix",
                "pc3_score",
                "tile_idx",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            kind="mergesort",
        )

    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    return out


def sample_balanced_stratum(
    candidates: pd.DataFrame,
    stratum: str,
    direction: str,
) -> pd.DataFrame:
    """
    Deterministic patient-balanced / slide-balanced sampling.

    Strategy:
      1. Rank extreme candidates within each slide.
      2. Apply per-slide cap.
      3. Pool capped slide candidates per patient.
      4. Rank again and apply per-patient cap.
    """
    if candidates.empty:
        return candidates.copy()

    ranked = deterministic_rank(
        candidates,
        direction,
    )

    slide_selected = []

    for (
        patient_id,
        prefix,
    ), group in ranked.groupby(
        [
            "patient_id",
            "prefix",
        ],
        sort=True,
    ):
        if (
            len(
                group
            )
            < MIN_CANDIDATES_PER_SLIDE_PER_STRATUM
        ):
            continue

        selected_group = (
            group.head(
                MAX_TILES_PER_SLIDE_PER_STRATUM
            )
            .copy()
        )

        selected_group[
            "rank_within_slide_for_stratum"
        ] = np.arange(
            1,
            len(
                selected_group
            )
            + 1,
        )

        slide_selected.append(
            selected_group
        )

    if not slide_selected:
        return pd.DataFrame(
            columns=ranked.columns
        )

    after_slide_cap = pd.concat(
        slide_selected,
        ignore_index=True,
    )

    patient_selected = []

    for patient_id, group in (
        after_slide_cap.groupby(
            "patient_id",
            sort=True,
        )
    ):
        ranked_patient = (
            deterministic_rank(
                group,
                direction,
            )
        )

        selected_group = (
            ranked_patient.head(
                MAX_TILES_PER_PATIENT_PER_STRATUM
            )
            .copy()
        )

        selected_group[
            "rank_within_patient_for_stratum"
        ] = np.arange(
            1,
            len(
                selected_group
            )
            + 1,
        )

        patient_selected.append(
            selected_group
        )

    sampled = pd.concat(
        patient_selected,
        ignore_index=True,
    )

    sampled[
        "sampling_stratum"
    ] = stratum

    sampled[
        "sampling_direction"
    ] = direction

    return sampled


def assign_frozen_sample_ids(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    out = sampled.copy()

    stratum_order = {
        "all_tissue_low": 1,
        "all_tissue_high": 2,
        "tumor_only_low": 3,
        "tumor_only_high": 4,
    }

    out[
        "_stratum_order"
    ] = out[
        "sampling_stratum"
    ].map(
        stratum_order
    )

    out = out.sort_values(
        [
            "_stratum_order",
            "patient_id",
            "prefix",
            "rank_within_patient_for_stratum",
            "tile_idx",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    out[
        "sample_id"
    ] = [
        f"PC3ROI_{index:04d}"
        for index in range(
            1,
            len(
                out
            )
            + 1,
        )
    ]

    out[
        "model_tile_uid"
    ] = (
        out[
            "prefix"
        ].astype(
            str
        )
        + "__tile"
        + out[
            "tile_idx"
        ].astype(
            str
        )
    )

    if ENFORCE_NO_DUPLICATE_TILE_IDS:
        duplicate = out[
            [
                "sampling_stratum",
                "model_tile_uid",
            ]
        ].duplicated(
            keep=False
        )

        if duplicate.any():
            raise ValueError(
                "Duplicate tile detected within a sampling stratum."
            )

    return out.drop(
        columns=[
            "_stratum_order"
        ]
    )


# =============================================================================
# SUMMARIES
# =============================================================================
def build_sampling_summary(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for stratum, group in sampled.groupby(
        "sampling_stratum",
        sort=True,
    ):
        rows.append({
            "sampling_stratum": (
                stratum
            ),
            "n_samples": int(
                len(
                    group
                )
            ),
            "n_patients": int(
                group[
                    "patient_id"
                ].nunique()
            ),
            "n_slides": int(
                group[
                    "prefix"
                ].nunique()
            ),
            "median_pc3_score": float(
                group[
                    "pc3_score"
                ].median()
            ),
            "median_pc3_within_slide_percentile": float(
                group[
                    "pc3_within_slide_percentile"
                ].median()
            ),
            "median_tumor_overlap_fraction": float(
                group[
                    "tumor_overlap_fraction"
                ].median()
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_slide_summary(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    return (
        sampled.groupby(
            [
                "sampling_stratum",
                "patient_id",
                "prefix",
            ],
            as_index=False,
        )
        .agg(
            n_samples=(
                "sample_id",
                "count",
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


def build_patient_summary(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    return (
        sampled.groupby(
            [
                "sampling_stratum",
                "patient_id",
            ],
            as_index=False,
        )
        .agg(
            n_samples=(
                "sample_id",
                "count",
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


def round_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    out = dataframe.copy()

    numeric = (
        out
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    out[
        numeric
    ] = (
        out[
            numeric
        ]
        .round(
            ROUND_DIGITS
        )
    )

    return out


# =============================================================================
# README
# =============================================================================
def make_readme_text(
    summary: pd.DataFrame,
) -> str:
    lines = [
        "PC3 Morphology Sampling Manifest",
        "================================",
        "",
        "Purpose",
        "-------",
        (
            "Freeze a patient-balanced, slide-balanced, tumor-aware sampling "
            "set for downstream PC3 morphology characterization."
        ),
        "",
        "PC3 extreme definition",
        "----------------------",
        (
            f"Low extreme: within-slide percentile <= "
            f"{LOW_PERCENTILE:.2f}"
        ),
        (
            f"High extreme: within-slide percentile >= "
            f"{HIGH_PERCENTILE:.2f}"
        ),
        "",
        "Tumor-associated tile definition",
        "--------------------------------",
        (
            f"Tumor overlap fraction >= "
            f"{TUMOR_OVERLAP_THRESHOLD:.2f}"
        ),
        "",
        "Sampling caps",
        "-------------",
        (
            f"Maximum tiles per slide per stratum: "
            f"{MAX_TILES_PER_SLIDE_PER_STRATUM}"
        ),
        (
            f"Maximum tiles per patient per stratum: "
            f"{MAX_TILES_PER_PATIENT_PER_STRATUM}"
        ),
        "",
        "Strata",
        "------",
        "all_tissue_low",
        "all_tissue_high",
        "tumor_only_low",
        "tumor_only_high",
        "",
        "Important",
        "---------",
        (
            "No H&E morphology was inspected by this script when selecting "
            "samples."
        ),
        (
            "Downstream ROI extraction and blinded pathology review should "
            "use sample_id from pc3_sampling_manifest.csv."
        ),
        "",
        "Sampling summary",
        "----------------",
        summary.to_string(
            index=False
        ),
        "",
    ]

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    if not ANNOT_DIR.exists():
        raise FileNotFoundError(
            f"Missing annotation directory: "
            f"{ANNOT_DIR}"
        )

    print(
        "=" * 110
    )
    print(
        "PC3 MORPHOLOGY SAMPLING MANIFEST"
    )
    print(
        "=" * 110
    )

    print(
        f"[ANNOTATIONS]                  "
        f"{ANNOT_DIR}"
    )

    print(
        f"[LOW PERCENTILE]               "
        f"{LOW_PERCENTILE:.2f}"
    )

    print(
        f"[HIGH PERCENTILE]              "
        f"{HIGH_PERCENTILE:.2f}"
    )

    print(
        f"[TUMOR OVERLAP THRESHOLD]      "
        f"{TUMOR_OVERLAP_THRESHOLD:.2f}"
    )

    print(
        f"[MAX TILES/SLIDE/STRATUM]      "
        f"{MAX_TILES_PER_SLIDE_PER_STRATUM}"
    )

    print(
        f"[MAX TILES/PATIENT/STRATUM]    "
        f"{MAX_TILES_PER_PATIENT_PER_STRATUM}"
    )

    print(
        f"[OUTPUT]                       "
        f"{OUT_DIR}"
    )

    xml_map = build_file_map(
        ANNOT_DIR,
        (
            ".xml",
        ),
    )

    (
        unique_entries,
        duplicate_manifest,
    ) = collect_unique_entries()

    duplicate_manifest.to_csv(
        OUT_DUPLICATES,
        index=False,
    )

    print(
        f"[UNIQUE SLIDES]                "
        f"{len(unique_entries)}"
    )

    all_candidate_tables = []
    excluded_rows = []

    for index, entry in enumerate(
        unique_entries,
        start=1,
    ):
        print(
            f"[SLIDE {index:03d}/"
            f"{len(unique_entries):03d}] "
            f"{entry['prefix']} "
            f"| {entry['dataset']}"
        )

        (
            candidate_table,
            exclusion,
        ) = build_slide_candidate_table(
            entry=entry,
            xml_map=xml_map,
        )

        if exclusion is not None:
            excluded_rows.append(
                exclusion
            )
            print(
                f"    [EXCLUDE] "
                f"{exclusion['reason']}"
            )
            continue

        all_candidate_tables.append(
            candidate_table
        )

        print(
            f"    tiles="
            f"{len(candidate_table)} | "
            f"low="
            f"{candidate_table['is_pc3_low_extreme'].sum()} | "
            f"high="
            f"{candidate_table['is_pc3_high_extreme'].sum()} | "
            f"tumor="
            f"{candidate_table['is_tumor_associated'].sum()}"
        )

    if not all_candidate_tables:
        raise RuntimeError(
            "No eligible slides produced candidate tiles."
        )

    all_candidates = pd.concat(
        all_candidate_tables,
        ignore_index=True,
    )

    all_candidates = (
        round_numeric_columns(
            all_candidates
        )
    )

    all_candidates.to_csv(
        OUT_ALL_CANDIDATES,
        index=False,
        compression="gzip",
    )

    strata = [
        (
            "all_tissue_low",
            "low",
        ),
        (
            "all_tissue_high",
            "high",
        ),
        (
            "tumor_only_low",
            "low",
        ),
        (
            "tumor_only_high",
            "high",
        ),
    ]

    sampled_tables = []

    for (
        stratum,
        direction,
    ) in strata:
        candidates = candidate_subset(
            all_candidates,
            stratum,
        )

        sampled = sample_balanced_stratum(
            candidates=candidates,
            stratum=stratum,
            direction=direction,
        )

        if sampled.empty:
            print(
                f"[WARNING] No samples for "
                f"{stratum}"
            )
            continue

        sampled_tables.append(
            sampled
        )

        print(
            f"[SAMPLED] "
            f"{stratum}: "
            f"{len(sampled)} tiles | "
            f"{sampled['patient_id'].nunique()} patients | "
            f"{sampled['prefix'].nunique()} slides"
        )

    if not sampled_tables:
        raise RuntimeError(
            "No morphology sampling strata produced samples."
        )

    sampled = pd.concat(
        sampled_tables,
        ignore_index=True,
    )

    sampled = assign_frozen_sample_ids(
        sampled
    )

    sampled = round_numeric_columns(
        sampled
    )

    sampled.to_csv(
        OUT_SAMPLING_MANIFEST,
        index=False,
    )

    sampling_summary = (
        build_sampling_summary(
            sampled
        )
    )

    sampling_summary = (
        round_numeric_columns(
            sampling_summary
        )
    )

    sampling_summary.to_csv(
        OUT_SAMPLING_SUMMARY,
        index=False,
    )

    slide_summary = (
        build_slide_summary(
            sampled
        )
    )

    slide_summary = (
        round_numeric_columns(
            slide_summary
        )
    )

    slide_summary.to_csv(
        OUT_SLIDE_SUMMARY,
        index=False,
    )

    patient_summary = (
        build_patient_summary(
            sampled
        )
    )

    patient_summary = (
        round_numeric_columns(
            patient_summary
        )
    )

    patient_summary.to_csv(
        OUT_PATIENT_SUMMARY,
        index=False,
    )

    if excluded_rows:
        excluded = pd.DataFrame(
            excluded_rows
        )
    else:
        excluded = pd.DataFrame(
            columns=[
                "patient_id",
                "prefix",
                "reason",
                "details",
            ]
        )

    excluded.to_csv(
        OUT_EXCLUDED,
        index=False,
    )

    metadata = {
        "analysis_name": (
            "PC3 morphology sampling manifest"
        ),

        "analysis_role": (
            "Prespecified sampling before ROI extraction, "
            "quantitative morphology, and blinded pathology review"
        ),

        "pc_component": (
            "PC3"
        ),

        "pc_index_zero_based": (
            PC3_INDEX
        ),

        "dataset_priority": [
            {
                "dataset": name,
                "path": str(
                    path
                ),
            }
            for name, path
            in DATASET_PRIORITY
        ],

        "n_unique_slides": int(
            len(
                unique_entries
            )
        ),

        "n_candidate_tiles": int(
            len(
                all_candidates
            )
        ),

        "n_sampled_rows": int(
            len(
                sampled
            )
        ),

        "extreme_definition": {
            "scope": (
                "within-slide percentile"
            ),
            "low_percentile_max": (
                LOW_PERCENTILE
            ),
            "high_percentile_min": (
                HIGH_PERCENTILE
            ),
        },

        "tumor_definition": {
            "metric": (
                "tile area overlap with union of XML tumor polygons"
            ),
            "threshold": (
                TUMOR_OVERLAP_THRESHOLD
            ),
        },

        "sampling_caps": {
            "max_tiles_per_slide_per_stratum": (
                MAX_TILES_PER_SLIDE_PER_STRATUM
            ),
            "max_tiles_per_patient_per_stratum": (
                MAX_TILES_PER_PATIENT_PER_STRATUM
            ),
        },

        "sampling_strata": [
            "all_tissue_low",
            "all_tissue_high",
            "tumor_only_low",
            "tumor_only_high",
        ],

        "selection_uses_morphology": False,

        "selection_uses_external_cohort": False,

        "selection_uses_pathologist_review": False,

        "selection_is_deterministic": True,

        "random_state": (
            RANDOM_STATE
        ),

        "outputs": {
            "all_candidate_tiles": str(
                OUT_ALL_CANDIDATES
            ),
            "sampling_manifest": str(
                OUT_SAMPLING_MANIFEST
            ),
            "sampling_summary": str(
                OUT_SAMPLING_SUMMARY
            ),
            "slide_summary": str(
                OUT_SLIDE_SUMMARY
            ),
            "patient_summary": str(
                OUT_PATIENT_SUMMARY
            ),
            "duplicate_manifest": str(
                OUT_DUPLICATES
            ),
            "excluded_slides": str(
                OUT_EXCLUDED
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
        make_readme_text(
            sampling_summary
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "SAMPLING SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        sampling_summary.to_string(
            index=False
        )
    )

    print(
        "\n[OK] Saved:"
    )

    for path in [
        OUT_ALL_CANDIDATES,
        OUT_SAMPLING_MANIFEST,
        OUT_SAMPLING_SUMMARY,
        OUT_SLIDE_SUMMARY,
        OUT_PATIENT_SUMMARY,
        OUT_DUPLICATES,
        OUT_EXCLUDED,
        OUT_METADATA,
        OUT_README,
    ]:
        print(
            f"  {path}"
        )

    print(
        "\n[DONE] PC3 morphology sampling manifest complete."
    )


if __name__ == "__main__":
    main()
