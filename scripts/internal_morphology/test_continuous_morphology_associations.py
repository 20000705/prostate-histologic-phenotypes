#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15d_test_pc3_continuous_morphology_associations.py

Step 15d v1.1.0
Prespecified continuous-PC3 morphology sensitivity analysis.

PURPOSE
-------
Step 15c tested frozen PC3 extreme groups. Step 15d addresses a different
question:

    Does morphology vary continuously with PC3 within slides, rather than
    appearing only because top/bottom tails were selected?

IMPORTANT DESIGN
----------------
This script DOES NOT reuse only the Step-15b extreme-group measurements.

Instead, for every Step-13a all-tissue eligible slide, it:
    1. loads the complete frozen PC3 tile-score vector and tile coordinates;
    2. spans the full within-slide PC3 distribution using deterministic
       quantile-balanced sampling;
    3. computes morphology with the SAME Step-15b v1.4 morphology engine;
    4. computes within-slide Spearman rho:
           PC3 vs SEI
           PC3 vs GOI
           PC3 vs NPI
    5. aggregates slide rho values to patient median rho;
    6. performs patient-level two-sided Wilcoxon signed-rank tests vs rho=0;
    7. applies BH-FDR across the three primary morphology dimensions.

No tile-level inferential p-values are used.

EXPECTED DIRECTIONS FROM THE PREVIOUSLY DECLARED BIOLOGICAL HYPOTHESIS
-----------------------------------------------------------------------
    PC3 vs SEI : positive
    PC3 vs GOI : negative
    PC3 vs NPI : negative

Formal tests remain TWO-SIDED.

TIMING / TRANSPARENCY
---------------------
The existence of a continuous-PC3 sensitivity analysis was already declared
in Step-15a before Step-15c inference. The detailed deterministic sampling
implementation in this script was finalized after Step-15c results were
observed. Therefore this analysis must be described as a prespecified
sensitivity objective with post-Step-15c implementation finalization, not as
a fully outcome-naive preregistration.

SAMPLING
--------
Within each slide:
    - divide the empirical PC3 distribution into 10 equal-frequency bins;
    - target up to 5 QC-passing tiles per bin;
    - candidate tiles are ordered deterministically by distance to each
      bin's PC3 median, then tile index;
    - no morphology value is used to select candidates;
    - frozen Step-15a image QC is applied;
    - no replacement is made using morphology direction.

Maximum target:
    50 tiles / slide.

This produces coverage across the entire PC3 distribution while keeping the
runtime similar to Step 15b.

FORMAL INFERENCE UNIT
---------------------
Patient.

Per-slide rho values are descriptive intermediate quantities.
Per-patient median rho is the inferential observation.

OUTPUT
------
results/15d_pc3_continuous_morphology_sensitivity/
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

BASE = DATA_ROOT / 'discovery'

RESULTS_ROOT = RESULTS_ROOT

STEP13A_MANIFEST = (
    RESULTS_ROOT
    / "13a_pc3_morphology_sampling_manifest"
    / "pc3_sampling_manifest.csv"
)

STEP15A_DIR = (
    RESULTS_ROOT
    / "15a_pc3_quantitative_morphology_specification"
)

STEP15A_PLAN = (
    STEP15A_DIR
    / "pc3_quantitative_morphology_analysis_plan.json"
)

STEP15A_THRESHOLDS = (
    STEP15A_DIR
    / "pc3_quantitative_morphology_thresholds.json"
)

STEP15A_FEATURE_SPEC = (
    STEP15A_DIR
    / "pc3_quantitative_morphology_feature_specification.csv"
)

STEP15B_SCRIPT = (
    SCRIPT_DIR
    / "15b_compute_pc3_quantitative_morphology.py"
)

STEP15B_METADATA = (
    RESULTS_ROOT
    / "15b_pc3_quantitative_morphology"
    / "analysis_metadata.json"
)

OUT_DIR = (
    RESULTS_ROOT
    / "15d_pc3_continuous_morphology_sensitivity"
)

OUT_SAMPLE_MANIFEST = (
    OUT_DIR
    / "pc3_continuous_quantile_sampling_manifest.csv"
)

OUT_QC_FAILURES = (
    OUT_DIR
    / "pc3_continuous_sampling_qc_failures.csv"
)

OUT_TILE_FEATURES = (
    OUT_DIR
    / "pc3_continuous_tile_features.csv"
)

OUT_SLIDE_RHO = (
    OUT_DIR
    / "pc3_continuous_within_slide_spearman.csv"
)

OUT_PATIENT_RHO = (
    OUT_DIR
    / "pc3_continuous_patient_median_rho.csv"
)

OUT_RESULTS = (
    OUT_DIR
    / "pc3_continuous_patient_level_inference.csv"
)

OUT_COVERAGE = (
    OUT_DIR
    / "pc3_continuous_sampling_coverage.csv"
)

OUT_SPEC = (
    OUT_DIR
    / "pc3_continuous_analysis_specification.json"
)

OUT_METADATA = (
    OUT_DIR
    / "analysis_metadata.json"
)

OUT_README = (
    OUT_DIR
    / "README.txt"
)

CORE_OUTPUTS = [
    OUT_SAMPLE_MANIFEST,
    OUT_QC_FAILURES,
    OUT_TILE_FEATURES,
    OUT_SLIDE_RHO,
    OUT_PATIENT_RHO,
    OUT_RESULTS,
    OUT_COVERAGE,
    OUT_SPEC,
    OUT_METADATA,
    OUT_README,
]


# =============================================================================
# VERSION / FROZEN DESIGN
# =============================================================================

VERSION = "1.1.0"

EXPECTED_15A_VERSION_PREFIX = "2.2."
EXPECTED_15B_VERSION_PREFIX = "1.4."

N_QUANTILE_BINS = 10
TARGET_TILES_PER_BIN = 5
MAX_TILES_PER_SLIDE = (
    N_QUANTILE_BINS
    * TARGET_TILES_PER_BIN
)

MIN_TILES_PER_SLIDE_FOR_RHO = 20

PRIMARY_FEATURES = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "NPI_nuclear_packing",
]

EXPECTED_DIRECTIONS = {
    "SEI_eosin_stroma": "positive",
    "GOI_lumen_openness": "negative",
    "NPI_nuclear_packing": "negative",
}

PCA_COMPONENT_INDEX_ZERO_BASED = 2



# =============================================================================
# CLI / BASIC HELPERS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help=(
            "Engineering/debugging only. "
            "Do not use for the formal Step-15d run."
        ),
    )

    return parser.parse_args()


def require_file(
    path: Path,
    label_text: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input ({label_text}): {path}"
        )


def sha256_file(
    path: Path,
) -> Optional[str]:
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def guard_outputs(
    overwrite: bool,
):
    existing = [
        path
        for path in CORE_OUTPUTS
        if path.exists()
    ]

    if (
        existing
        and not overwrite
    ):
        raise RuntimeError(
            "Step-15d outputs already exist. "
            "Refusing silent overwrite. "
            "Use --overwrite if intentional."
        )


# =============================================================================
# LOAD / VALIDATE FROZEN DEPENDENCIES
# =============================================================================

def load_step15b_module():
    require_file(
        STEP15B_SCRIPT,
        "Step 15b v1.4 morphology engine",
    )

    spec = importlib.util.spec_from_file_location(
        "step15b_morphology_engine",
        STEP15B_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None

    spec.loader.exec_module(
        module
    )

    version = str(
        getattr(
            module,
            "VERSION",
            "",
        )
    )

    if not version.startswith(
        EXPECTED_15B_VERSION_PREFIX
    ):
        raise RuntimeError(
            "Step-15d requires Step-15b morphology engine "
            f"{EXPECTED_15B_VERSION_PREFIX}x; found {version}"
        )

    return module


def load_and_validate_dependencies():
    for path, label_text in [
        (
            STEP13A_MANIFEST,
            "Step 13a sampling manifest",
        ),
        (
            STEP15A_PLAN,
            "Step 15a analysis plan",
        ),
        (
            STEP15A_THRESHOLDS,
            "Step 15a thresholds",
        ),
        (
            STEP15A_FEATURE_SPEC,
            "Step 15a feature specification",
        ),
        (
            STEP15B_METADATA,
            "Step 15b metadata",
        ),
    ]:
        require_file(
            path,
            label_text,
        )

    with STEP15A_PLAN.open(
        "r",
        encoding="utf-8",
    ) as handle:
        plan = json.load(
            handle
        )

    with STEP15A_THRESHOLDS.open(
        "r",
        encoding="utf-8",
    ) as handle:
        thresholds = json.load(
            handle
        )

    with STEP15B_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata15b = json.load(
            handle
        )

    version15a = str(
        plan.get(
            "version",
            "",
        )
    )

    if not version15a.startswith(
        EXPECTED_15A_VERSION_PREFIX
    ):
        raise RuntimeError(
            "Step-15d expects Step-15a "
            f"{EXPECTED_15A_VERSION_PREFIX}x; found {version15a}"
        )

    if plan.get(
        "primary_endpoints",
        [],
    ) != PRIMARY_FEATURES:
        raise RuntimeError(
            "Frozen primary endpoint mismatch."
        )

    continuous = plan.get(
        "continuous_pc3_sensitivity",
        {},
    )

    if continuous.get(
        "analysis_step"
    ) != "15d":
        raise RuntimeError(
            "Step-15a does not designate Step 15d "
            "for continuous-PC3 sensitivity."
        )

    # Exact Step15a -> Step15b provenance chain must still hold.
    recorded = metadata15b.get(
        "step15a",
        {},
    )

    checks = {
        "plan":
            (
                recorded.get(
                    "plan_sha256"
                ),
                sha256_file(
                    STEP15A_PLAN
                ),
            ),

        "thresholds":
            (
                recorded.get(
                    "threshold_sha256"
                ),
                sha256_file(
                    STEP15A_THRESHOLDS
                ),
            ),

        "feature_spec":
            (
                recorded.get(
                    "feature_spec_sha256"
                ),
                sha256_file(
                    STEP15A_FEATURE_SPEC
                ),
            ),
    }

    for label_text, (
        recorded_hash,
        current_hash,
    ) in checks.items():
        if (
            recorded_hash is None
            or recorded_hash != current_hash
        ):
            raise RuntimeError(
                f"Step-15a / Step-15b provenance mismatch for {label_text}."
            )

    if metadata15b.get(
        "inferential_statistics_performed",
        False,
    ):
        raise RuntimeError(
            "Unexpected Step-15b inference flag."
        )

    morphology_engine = load_step15b_module()

    return (
        plan,
        thresholds,
        metadata15b,
        morphology_engine,
        checks,
    )


# =============================================================================
# DISCOVERY SLIDE SET
# =============================================================================

def _resolve_unique_text_value(
    sub: pd.DataFrame,
    column: str,
    prefix: str,
) -> str:
    if column not in sub.columns:
        raise RuntimeError(
            f"Step-13a manifest missing required provenance column: {column}"
        )

    values = (
        sub[column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    values = values.loc[values != ""].unique().tolist()

    if len(values) != 1:
        raise RuntimeError(
            f"{prefix}: expected exactly one unique {column} across "
            f"all all-tissue Step-13a rows, found {len(values)}: {values}"
        )

    return values[0]


def _resolve_wsi_from_meta(
    meta_path: Path,
    prefix: str,
) -> Path:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{prefix}: meta_path does not exist: {meta_path}"
        )

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    slide_path_value = meta.get("slide_path")

    if not slide_path_value:
        raise RuntimeError(
            f"{prefix}: metadata JSON has no slide_path: {meta_path}"
        )

    slide_path = Path(str(slide_path_value))

    if not slide_path.exists():
        raise FileNotFoundError(
            f"{prefix}: resolved native WSI does not exist: {slide_path}"
        )

    return slide_path.resolve(strict=False)


def load_eligible_slides():
    """
    Define the Step-15d slide universe and exact file provenance directly from
    the frozen Step-13a all-tissue sampling manifest.

    Step-13a is authoritative for:
        dataset
        patient_id
        prefix
        coord_path
        pca_path
        meta_path

    Native WSI path is resolved from:
        meta_path -> metadata["slide_path"]

    No directory-wide rediscovery of PCA / coordinate / metadata files is
    performed in Step 15d.
    """

    df = pd.read_csv(
        STEP13A_MANIFEST,
        low_memory=False,
    )

    required_columns = [
        "dataset",
        "patient_id",
        "prefix",
        "coord_path",
        "pca_path",
        "meta_path",
        "sampling_stratum",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Step-13a manifest missing required columns: "
            f"{missing_columns}"
        )

    df = df.loc[
        df["sampling_stratum"].isin(
            ["all_tissue_low", "all_tissue_high"]
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No frozen all_tissue_low/high rows found in Step-13a manifest."
        )

    rows = []

    for (patient_id, prefix), sub in df.groupby(
        ["patient_id", "prefix"],
        dropna=False,
    ):
        dataset_values = (
            sub["dataset"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(dataset_values) != 1:
            raise RuntimeError(
                f"{prefix}: ambiguous dataset values: {dataset_values}"
            )

        coord_path = Path(
            _resolve_unique_text_value(
                sub,
                "coord_path",
                str(prefix),
            )
        )

        pca_path = Path(
            _resolve_unique_text_value(
                sub,
                "pca_path",
                str(prefix),
            )
        )

        meta_path = Path(
            _resolve_unique_text_value(
                sub,
                "meta_path",
                str(prefix),
            )
        )

        for path, label_text in [
            (coord_path, "coord_path"),
            (pca_path, "pca_path"),
            (meta_path, "meta_path"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{prefix}: frozen Step-13a {label_text} does not exist: "
                    f"{path}"
                )

        wsi_path = _resolve_wsi_from_meta(
            meta_path,
            str(prefix),
        )

        rows.append(
            {
                "dataset": dataset_values[0],
                "patient_id": str(patient_id),
                "prefix": str(prefix),
                "coord_path": str(coord_path.resolve(strict=False)),
                "pca_path": str(pca_path.resolve(strict=False)),
                "meta_path": str(meta_path.resolve(strict=False)),
                "wsi_path": str(wsi_path),
                "n_frozen_all_tissue_manifest_rows": int(len(sub)),
            }
        )

    slides = (
        pd.DataFrame(rows)
        .sort_values(["patient_id", "prefix"])
        .reset_index(drop=True)
    )

    if slides.duplicated(
        subset=["patient_id", "prefix"]
    ).any():
        raise RuntimeError(
            "Duplicate patient/prefix rows remain after Step-13a slide "
            "provenance resolution."
        )

    print(f"[ELIGIBLE SLIDES]    {len(slides)}")
    print(f"[ELIGIBLE PATIENTS]  {slides['patient_id'].nunique()}")
    print(
        f"[PCA PATHS RESOLVED] "
        f"{slides['pca_path'].notna().sum()}/{len(slides)}"
    )
    print(
        f"[COORDS RESOLVED]    "
        f"{slides['coord_path'].notna().sum()}/{len(slides)}"
    )
    print(
        f"[META PATHS]         "
        f"{slides['meta_path'].notna().sum()}/{len(slides)}"
    )
    print(
        f"[WSI PATHS]          "
        f"{slides['wsi_path'].notna().sum()}/{len(slides)}"
    )

    return slides


# =============================================================================
# DETERMINISTIC FULL-DISTRIBUTION SAMPLING
# =============================================================================

def make_quantile_bins(
    pc3_scores: np.ndarray,
) -> np.ndarray:
    """
    Equal-frequency bins using rank-based quantile assignment.

    This avoids qcut duplicate-edge problems.
    """
    scores = np.asarray(
        pc3_scores,
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            scores
        )
    ):
        raise ValueError(
            "PC3 scores contain non-finite values."
        )

    ranks = rankdata(
        scores,
        method="ordinal",
    )

    n = len(
        scores
    )

    bins = np.floor(
        (
            ranks
            - 1
        )
        * N_QUANTILE_BINS
        / n
    ).astype(
        int
    )

    bins = np.clip(
        bins,
        0,
        N_QUANTILE_BINS - 1,
    )

    return bins


def deterministic_candidates_for_bin(
    pc3_scores: np.ndarray,
    bin_ids: np.ndarray,
    bin_id: int,
) -> List[int]:
    indices = np.where(
        bin_ids == bin_id
    )[
        0
    ]

    if len(
        indices
    ) == 0:
        return []

    bin_scores = pc3_scores[
        indices
    ]

    center = float(
        np.median(
            bin_scores
        )
    )

    order = np.lexsort(
        (
            indices,
            np.abs(
                bin_scores
                - center
            ),
        )
    )

    return indices[
        order
    ].tolist()


# =============================================================================
# SLIDE SAMPLING / FEATURE EXTRACTION
# =============================================================================

def process_slide(
    slide_row: pd.Series,
    thresholds,
    morphology_engine,
):
    patient_id = str(
        slide_row[
            "patient_id"
        ]
    )

    prefix = str(
        slide_row[
            "prefix"
        ]
    )

    dataset = str(
        slide_row.get(
            "dataset",
            "",
        )
    )

    wsi_path = Path(
        str(
            slide_row[
                "wsi_path"
            ]
        )
    )

    if not wsi_path.exists():
        raise FileNotFoundError(
            f"WSI missing for {prefix}: {wsi_path}"
        )

    pca_path = Path(str(slide_row["pca_path"]))
    coords_path = Path(str(slide_row["coord_path"]))
    meta_path = Path(str(slide_row["meta_path"]))

    for path, label_text in [
        (pca_path, "pca_path"),
        (coords_path, "coord_path"),
        (meta_path, "meta_path"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{prefix}: frozen Step-13a {label_text} missing: {path}"
            )

    pca_targets = np.load(
        pca_path,
        mmap_mode="r",
    )

    coords = np.load(
        coords_path,
        mmap_mode="r",
    )

    if pca_targets.ndim != 2:
        raise RuntimeError(
            f"{prefix}: PCA targets must be 2D, got {pca_targets.shape}"
        )

    if (
        PCA_COMPONENT_INDEX_ZERO_BASED
        >= pca_targets.shape[
            1
        ]
    ):
        raise RuntimeError(
            f"{prefix}: PC3 column unavailable."
        )

    if (
        coords.ndim != 2
        or coords.shape[
            1
        ] < 2
    ):
        raise RuntimeError(
            f"{prefix}: coords must be Nx2+, got {coords.shape}"
        )

    if len(
        pca_targets
    ) != len(
        coords
    ):
        raise RuntimeError(
            f"{prefix}: PCA/coords row mismatch "
            f"{len(pca_targets)} vs {len(coords)}"
        )

    pc3 = np.asarray(
        pca_targets[
            :,
            PCA_COMPONENT_INDEX_ZERO_BASED,
        ],
        dtype=float,
    )

    bin_ids = make_quantile_bins(
        pc3
    )

    patch_size = int(
        thresholds[
            "patch_size_default"
        ]
    )

    tile_rows = []
    failure_rows = []
    sampling_rows = []

    for bin_id in range(
        N_QUANTILE_BINS
    ):
        candidates = deterministic_candidates_for_bin(
            pc3,
            bin_ids,
            bin_id,
        )

        accepted_in_bin = 0
        attempted_in_bin = 0

        for tile_index in candidates:
            if (
                accepted_in_bin
                >= TARGET_TILES_PER_BIN
            ):
                break

            attempted_in_bin += 1

            x = int(
                coords[
                    tile_index,
                    0,
                ]
            )

            y = int(
                coords[
                    tile_index,
                    1,
                ]
            )

            base = {
                "dataset":
                    dataset,

                "patient_id":
                    patient_id,

                "prefix":
                    prefix,

                "wsi_path":
                    str(
                        wsi_path
                    ),

                "pca_targets_path":
                    str(
                        pca_path
                    ),

                "coords_path":
                    str(
                        coords_path
                    ),

                "meta_path":
                    str(
                        meta_path
                    ),

                "tile_idx":
                    int(
                        tile_index
                    ),

                "x":
                    x,

                "y":
                    y,

                "patch_size":
                    patch_size,

                "pc3_score":
                    float(
                        pc3[
                            tile_index
                        ]
                    ),

                "quantile_bin":
                    int(
                        bin_id
                    ),

                "candidate_rank_within_bin":
                    int(
                        attempted_in_bin
                    ),
            }

            try:
                patch = morphology_engine.crop_native_with_bounds_check(
                    str(
                        wsi_path
                    ),
                    x,
                    y,
                    patch_size,
                )

                features = morphology_engine.compute_features(
                    patch,
                    thresholds,
                )

                qc_pass, qc_reasons = morphology_engine.evaluate_qc(
                    features,
                    thresholds,
                )

            except Exception as error:
                failure_rows.append(
                    {
                        **base,

                        "failure_stage":
                            "crop_or_feature_extraction",

                        "failure_reason":
                            repr(
                                error
                            ),
                    }
                )

                continue

            if not qc_pass:
                failure_rows.append(
                    {
                        **base,

                        "failure_stage":
                            "frozen_image_qc",

                        "failure_reason":
                            ";".join(
                                qc_reasons
                            ),

                        "tissue_fraction":
                            features[
                                "tissue_fraction"
                            ],

                        "white_fraction":
                            features[
                                "white_fraction"
                            ],

                        "black_fraction":
                            features[
                                "black_fraction"
                            ],
                    }
                )

                continue

            accepted_in_bin += 1

            sampling_rows.append(
                {
                    **base,

                    "sampling_status":
                        "selected_qc_pass",

                    "selected_order_within_bin":
                        accepted_in_bin,
                }
            )

            tile_rows.append(
                {
                    **base,
                    **features,
                }
            )

    coverage = {
        "dataset":
            dataset,

        "patient_id":
            patient_id,

        "prefix":
            prefix,

        "n_source_tiles":
            int(
                len(
                    pc3
                )
            ),

        "n_selected_qc_pass_tiles":
            int(
                len(
                    tile_rows
                )
            ),

        "n_target_tiles":
            int(
                MAX_TILES_PER_SLIDE
            ),

        "n_bins_with_at_least_one_selected":
            int(
                len(
                    set(
                        row[
                            "quantile_bin"
                        ]
                        for row in tile_rows
                    )
                )
            ),

        "pc3_min_source":
            float(
                np.min(
                    pc3
                )
            ),

        "pc3_max_source":
            float(
                np.max(
                    pc3
                )
            ),

        "pc3_min_selected":
            (
                float(
                    min(
                        row[
                            "pc3_score"
                        ]
                        for row in tile_rows
                    )
                )
                if tile_rows
                else np.nan
            ),

        "pc3_max_selected":
            (
                float(
                    max(
                        row[
                            "pc3_score"
                        ]
                        for row in tile_rows
                    )
                )
                if tile_rows
                else np.nan
            ),
    }

    return (
        pd.DataFrame(
            tile_rows
        ),
        pd.DataFrame(
            failure_rows
        ),
        pd.DataFrame(
            sampling_rows
        ),
        coverage,
    )


# =============================================================================
# SPEARMAN / PATIENT INFERENCE
# =============================================================================

def safe_spearman(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(
        x
    ) < MIN_TILES_PER_SLIDE_FOR_RHO:
        return (
            np.nan,
            "insufficient_tiles",
        )

    if (
        np.unique(
            x
        ).size
        < 3
        or np.unique(
            y
        ).size
        < 3
    ):
        return (
            np.nan,
            "insufficient_variation",
        )

    rho = spearmanr(
        x,
        y,
    ).statistic

    return (
        float(
            rho
        ),
        "performed",
    )


def build_slide_rho(
    tile_df,
):
    rows = []

    for (
        patient_id,
        prefix,
    ), sub in tile_df.groupby(
        [
            "patient_id",
            "prefix",
        ],
        dropna=False,
    ):
        for feature in PRIMARY_FEATURES:
            rho, status = safe_spearman(
                sub[
                    "pc3_score"
                ],
                sub[
                    feature
                ],
            )

            rows.append(
                {
                    "patient_id":
                        patient_id,

                    "prefix":
                        prefix,

                    "feature":
                        feature,

                    "expected_direction":
                        EXPECTED_DIRECTIONS[
                            feature
                        ],

                    "n_tiles":
                        int(
                            len(
                                sub
                            )
                        ),

                    "n_quantile_bins":
                        int(
                            sub[
                                "quantile_bin"
                            ].nunique()
                        ),

                    "pc3_min":
                        float(
                            sub[
                                "pc3_score"
                            ].min()
                        ),

                    "pc3_max":
                        float(
                            sub[
                                "pc3_score"
                            ].max()
                        ),

                    "spearman_rho":
                        rho,

                    "rho_status":
                        status,
                }
            )

    return pd.DataFrame(
        rows
    )


def build_patient_rho(
    slide_rho_df,
):
    performed = slide_rho_df.loc[
        slide_rho_df[
            "rho_status"
        ] == "performed"
    ].copy()

    patient = (
        performed
        .groupby(
            [
                "patient_id",
                "feature",
                "expected_direction",
            ],
            dropna=False,
        )
        .agg(
            n_slides=(
                "prefix",
                "nunique",
            ),

            median_slide_rho=(
                "spearman_rho",
                "median",
            ),

            mean_slide_rho=(
                "spearman_rho",
                "mean",
            ),
        )
        .reset_index()
    )

    return patient


def paired_rank_biserial_one_sample(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    values = values[
        values != 0
    ]

    if len(
        values
    ) == 0:
        return np.nan

    ranks = rankdata(
        np.abs(
            values
        ),
        method="average",
    )

    w_plus = float(
        ranks[
            values > 0
        ].sum()
    )

    w_minus = float(
        ranks[
            values < 0
        ].sum()
    )

    denominator = (
        w_plus
        + w_minus
    )

    if denominator == 0:
        return np.nan

    return float(
        (
            w_plus
            - w_minus
        )
        / denominator
    )


def bootstrap_median_ci(
    values,
    n_resamples,
    seed,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    rng = np.random.default_rng(
        seed
    )

    n = len(
        values
    )

    if n == 0:
        return (
            np.nan,
            np.nan,
        )

    boots = np.empty(
        n_resamples,
        dtype=float,
    )

    for index in range(
        n_resamples
    ):
        draw = rng.integers(
            0,
            n,
            size=n,
        )

        boots[
            index
        ] = np.median(
            values[
                draw
            ]
        )

    return (
        float(
            np.quantile(
                boots,
                0.025,
            )
        ),
        float(
            np.quantile(
                boots,
                0.975,
            )
        ),
    )


def benjamini_hochberg(
    p_values,
):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    mask = np.isfinite(
        p_values
    )

    valid = p_values[
        mask
    ]

    m = len(
        valid
    )

    if m == 0:
        return adjusted

    order = np.argsort(
        valid
    )

    ranked = valid[
        order
    ]

    raw = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    monotone = np.minimum.accumulate(
        raw[
            ::-1
        ]
    )[
        ::-1
    ]

    monotone = np.clip(
        monotone,
        0,
        1,
    )

    restored = np.empty_like(
        monotone
    )

    restored[
        order
    ] = monotone

    adjusted[
        mask
    ] = restored

    return adjusted


def infer_patient_rho(
    patient_rho_df,
    plan,
):
    minimum_patients = int(
        plan[
            "inference_qc_gate"
        ][
            "minimum_paired_patients"
        ]
    )

    n_bootstrap = int(
        plan[
            "statistical_tests"
        ][
            "bootstrap_resamples"
        ]
    )

    base_seed = int(
        plan[
            "statistical_tests"
        ][
            "random_seed"
        ]
    )

    rows = []

    for feature_index, feature in enumerate(
        PRIMARY_FEATURES
    ):
        sub = patient_rho_df.loc[
            patient_rho_df[
                "feature"
            ] == feature
        ].copy()

        values = pd.to_numeric(
            sub[
                "median_slide_rho"
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        n_patients = len(
            values
        )

        if n_patients < minimum_patients:
            rows.append(
                {
                    "feature":
                        feature,

                    "expected_direction":
                        EXPECTED_DIRECTIONS[
                            feature
                        ],

                    "inference_status":
                        "insufficient_patients",

                    "minimum_patients_required":
                        minimum_patients,

                    "n_patients":
                        n_patients,

                    "median_patient_rho":
                        (
                            float(
                                np.median(
                                    values
                                )
                            )
                            if n_patients
                            else np.nan
                        ),

                    "bootstrap_95ci_lower":
                        np.nan,

                    "bootstrap_95ci_upper":
                        np.nan,

                    "rank_biserial_vs_zero":
                        np.nan,

                    "wilcoxon_statistic":
                        np.nan,

                    "p_value_raw":
                        np.nan,

                    "n_positive_rho":
                        int(
                            np.sum(
                                values > 0
                            )
                        ),

                    "n_negative_rho":
                        int(
                            np.sum(
                                values < 0
                            )
                        ),

                    "n_zero_rho":
                        int(
                            np.sum(
                                values == 0
                            )
                        ),
                }
            )

            continue

        nonzero = values[
            values != 0
        ]

        if len(
            nonzero
        ) == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            result = wilcoxon(
                values,
                np.zeros_like(
                    values
                ),
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        ci_low, ci_high = bootstrap_median_ci(
            values,
            n_resamples=n_bootstrap,
            seed=(
                base_seed
                + 15000
                + feature_index
            ),
        )

        rows.append(
            {
                "feature":
                    feature,

                "expected_direction":
                    EXPECTED_DIRECTIONS[
                        feature
                    ],

                "inference_status":
                    "performed",

                "minimum_patients_required":
                    minimum_patients,

                "n_patients":
                    n_patients,

                "median_patient_rho":
                    float(
                        np.median(
                            values
                        )
                    ),

                "mean_patient_rho":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "bootstrap_95ci_lower":
                    ci_low,

                "bootstrap_95ci_upper":
                    ci_high,

                "rank_biserial_vs_zero":
                    paired_rank_biserial_one_sample(
                        values
                    ),

                "wilcoxon_statistic":
                    statistic,

                "p_value_raw":
                    p_value,

                "n_positive_rho":
                    int(
                        np.sum(
                            values > 0
                        )
                    ),

                "n_negative_rho":
                    int(
                        np.sum(
                            values < 0
                        )
                    ),

                "n_zero_rho":
                    int(
                        np.sum(
                            values == 0
                        )
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    results[
        "p_value_fdr_bh"
    ] = benjamini_hochberg(
        results[
            "p_value_raw"
        ].to_numpy(
            dtype=float
        )
    )

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print(
        "=" * 124
    )

    print(
        "STEP 15D v1.1: CONTINUOUS-PC3 MORPHOLOGY SENSITIVITY"
    )

    print(
        "=" * 124
    )

    print(
        "[VERSION]      ",
        VERSION,
    )

    print(
        "[BINS]         ",
        N_QUANTILE_BINS,
    )

    print(
        "[TILES/BIN]    ",
        TARGET_TILES_PER_BIN,
    )

    print(
        "[MAX/SLIDE]    ",
        MAX_TILES_PER_SLIDE,
    )

    print(
        "[OUTPUT]       ",
        OUT_DIR,
    )

    guard_outputs(
        args.overwrite
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        plan,
        thresholds,
        metadata15b,
        morphology_engine,
        provenance_checks,
    ) = load_and_validate_dependencies()

    slides = load_eligible_slides()

    if args.max_slides is not None:
        if args.max_slides <= 0:
            raise ValueError(
                "--max-slides must be positive."
            )

        print(
            "[WARNING] DEBUG / ENGINEERING RUN ONLY"
        )

        slides = slides.head(
            args.max_slides
        ).copy()

    all_tiles = []
    all_failures = []
    all_sampling = []
    coverage_rows = []

    for slide_number, (
        _,
        slide_row,
    ) in enumerate(
        slides.iterrows(),
        start=1,
    ):
        prefix = str(
            slide_row[
                "prefix"
            ]
        )

        print(
            f"[SLIDE {slide_number:02d}/{len(slides):02d}] {prefix}"
        )

        (
            tile_df,
            failure_df,
            sampling_df,
            coverage,
        ) = process_slide(
            slide_row,
            thresholds,
            morphology_engine,
        )

        if not tile_df.empty:
            all_tiles.append(
                tile_df
            )

        if not failure_df.empty:
            all_failures.append(
                failure_df
            )

        if not sampling_df.empty:
            all_sampling.append(
                sampling_df
            )

        coverage_rows.append(
            coverage
        )

    tile_df = (
        pd.concat(
            all_tiles,
            ignore_index=True,
        )
        if all_tiles
        else pd.DataFrame()
    )

    failure_df = (
        pd.concat(
            all_failures,
            ignore_index=True,
        )
        if all_failures
        else pd.DataFrame()
    )

    sampling_df = (
        pd.concat(
            all_sampling,
            ignore_index=True,
        )
        if all_sampling
        else pd.DataFrame()
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    if tile_df.empty:
        raise RuntimeError(
            "No continuous-sensitivity tiles passed measurement/QC."
        )

    slide_rho_df = build_slide_rho(
        tile_df
    )

    patient_rho_df = build_patient_rho(
        slide_rho_df
    )

    results_df = infer_patient_rho(
        patient_rho_df,
        plan,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    sampling_df.to_csv(
        OUT_SAMPLE_MANIFEST,
        index=False,
    )

    failure_df.to_csv(
        OUT_QC_FAILURES,
        index=False,
    )

    tile_df.to_csv(
        OUT_TILE_FEATURES,
        index=False,
    )

    slide_rho_df.to_csv(
        OUT_SLIDE_RHO,
        index=False,
    )

    patient_rho_df.to_csv(
        OUT_PATIENT_RHO,
        index=False,
    )

    results_df.to_csv(
        OUT_RESULTS,
        index=False,
    )

    coverage_df.to_csv(
        OUT_COVERAGE,
        index=False,
    )

    analysis_spec = {
        "version":
            VERSION,

        "analysis_role":
            "continuous-PC3 sensitivity",

        "declared_pre_step15c":
            True,

        "detailed_sampling_implementation_finalized_post_step15c":
            True,

        "slide_universe_and_file_provenance":
            (
                "Frozen Step-13a all-tissue manifest defines eligible slides "
                "and exact coord_path / pca_path / meta_path; meta JSON resolves "
                "the native slide_path. No directory-wide file rediscovery."
            ),

        "sampling":
            {
                "quantile_bins":
                    N_QUANTILE_BINS,

                "target_qc_pass_tiles_per_bin":
                    TARGET_TILES_PER_BIN,

                "maximum_tiles_per_slide":
                    MAX_TILES_PER_SLIDE,

                "candidate_order":
                    (
                        "absolute distance to within-bin PC3 median, "
                        "then tile index"
                    ),

                "morphology_used_for_sampling":
                    False,
            },

        "slide_level_association":
            "Spearman rho between tile PC3 and morphology",

        "formal_inference_unit":
            "patient",

        "patient_aggregation":
            "median of slide-level Spearman rho",

        "formal_test":
            "two-sided Wilcoxon signed-rank vs rho=0",

        "primary_features":
            PRIMARY_FEATURES,

        "expected_directions":
            EXPECTED_DIRECTIONS,

        "multiple_testing":
            "BH-FDR across three primary morphology dimensions",

        "tile_level_inferential_p_values":
            False,
    }

    with OUT_SPEC.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            analysis_spec,
            handle,
            indent=2,
        )

    metadata = {
        "version":
            VERSION,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "formal_full_run":
            args.max_slides is None,

        "debug_max_slides":
            args.max_slides,

        "step15a_version":
            plan[
                "version"
            ],

        "step15b_version":
            metadata15b[
                "version"
            ],

        "step15a_plan_sha256":
            sha256_file(
                STEP15A_PLAN
            ),

        "step15a_threshold_sha256":
            sha256_file(
                STEP15A_THRESHOLDS
            ),

        "step15a_feature_spec_sha256":
            sha256_file(
                STEP15A_FEATURE_SPEC
            ),

        "step15b_script_sha256":
            sha256_file(
                STEP15B_SCRIPT
            ),

        "provenance_chain_verified":
            True,

        "step13a_manifest_sha256":
            sha256_file(
                STEP13A_MANIFEST
            ),

        "n_slides_requested":
            int(
                len(
                    slides
                )
            ),

        "n_slides_with_selected_tiles":
            int(
                tile_df[
                    "prefix"
                ].nunique()
            ),

        "n_patients":
            int(
                tile_df[
                    "patient_id"
                ].nunique()
            ),

        "n_selected_tiles":
            int(
                len(
                    tile_df
                )
            ),

        "n_qc_or_processing_failures":
            int(
                len(
                    failure_df
                )
            ),

        "no_tile_level_inferential_p_values":
            True,

        "transparency_statement":
            (
                "The continuous-PC3 sensitivity objective was declared in "
                "Step-15a before Step-15c. The detailed deterministic sampling "
                "implementation was finalized after Step-15c results were "
                "observed and is therefore treated as sensitivity analysis, "
                "not fully outcome-naive preregistration."
            ),

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,
    }

    with OUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    OUT_README.write_text(
        """STEP 15D v1.1: CONTINUOUS-PC3 MORPHOLOGY SENSITIVITY
=================================================

Purpose
-------
Test whether the PC3 morphology relationship extends continuously across the
within-slide PC3 distribution instead of being restricted to frozen extreme
groups.

Slide universe / file provenance
-------------------------------
Frozen Step-13a all-tissue manifest defines:
  patient_id / prefix
  coord_path
  pca_path
  meta_path

meta_path -> meta["slide_path"] resolves the native WSI.
No directory-wide rediscovery is performed.

Sampling
--------
10 equal-frequency PC3 bins per slide.
Up to 5 QC-passing tiles per bin.
Candidate ordering is deterministic and uses PC3 only, never morphology.

Morphology
----------
Uses the same Step-15b v1.4 morphology engine and frozen Step-15a thresholds.

Association
-----------
Within each slide:
  Spearman rho between PC3 score and each primary morphology proxy.

Patient aggregation:
  median slide rho.

Formal inference:
  two-sided Wilcoxon signed-rank test of patient median rho vs zero.

Multiplicity:
  BH-FDR across SEI / GOI / NPI.

No tile-level p-values are used.

Transparency
------------
The continuous sensitivity objective was declared before Step-15c.
The detailed deterministic sampling implementation was finalized after Step-15c
results were observed. This must be described as sensitivity analysis rather
than fully outcome-naive preregistration.
""",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal
    # -------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 124
    )

    print(
        "SAMPLING COVERAGE"
    )

    print(
        "=" * 124
    )

    print(
        coverage_df[
            [
                "patient_id",
                "prefix",
                "n_source_tiles",
                "n_selected_qc_pass_tiles",
                "n_bins_with_at_least_one_selected",
                "pc3_min_source",
                "pc3_max_source",
                "pc3_min_selected",
                "pc3_max_selected",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 124
    )

    print(
        "PATIENT-LEVEL CONTINUOUS-PC3 INFERENCE"
    )

    print(
        "=" * 124
    )

    display_columns = [
        "feature",
        "expected_direction",
        "inference_status",
        "n_patients",
        "median_patient_rho",
        "bootstrap_95ci_lower",
        "bootstrap_95ci_upper",
        "rank_biserial_vs_zero",
        "p_value_raw",
        "p_value_fdr_bh",
        "n_positive_rho",
        "n_negative_rho",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n[DONE] Step-15d continuous-PC3 sensitivity complete."
    )


if __name__ == "__main__":
    main()
