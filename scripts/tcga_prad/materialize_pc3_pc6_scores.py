#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27d_materialize_frozen_tcga_prad_pc3_pc6_scores.py

STEP 27D
========
MATERIALIZE FROZEN TCGA-PRAD PC3 / PC6 SCORES

VERSION
-------
1.0.0

SCHEMA
------
27d-tcga-prad-pc3-pc6-score-materialization-schema-1.0

ROLE
----
REPRESENTATION-SIDE MATERIALIZATION ONLY.

This step combines:

    Step27B:
        frozen raw PC1-PC8 projection

with:

    Step27C:
        frozen PC3 / PC6 axis roles and discovery-derived score multipliers

to generate, for every frozen TCGA tile:

    raw_PC3
    oriented_PC3
    raw_PC6
    oriented_PC6

Definition
----------
For axis PCk:

    oriented_PCk = raw_PCk * frozen_score_multiplier

The multiplier is inherited exclusively from Step27C / Step06.

Expected current frozen directions
----------------------------------
PC3:
    discovery-frozen direction = low_in_tumor
    multiplier = -1

PC6:
    discovery-frozen direction = high_in_tumor
    multiplier = +1

Thus larger ORIENTED values correspond to the discovery-frozen
tumor-associated direction for both axes.

IMPORTANT
---------
This script DOES NOT:

    - fit or refit PCA,
    - modify the scaler,
    - re-project 2560-D embeddings,
    - reselect PC3 / PC6,
    - tune score signs,
    - inspect tumor masks,
    - calculate AUC,
    - inspect morphology,
    - inspect RNA / gene expression,
    - inspect Gleason score,
    - inspect clinical outcomes,
    - inspect survival,
    - choose thresholds,
    - select favorable cases,
    - create figures.

INPUT AUTHORITY
---------------
All slides come ONLY from the Step27B frozen projection manifest.

All PC3 / PC6 score multipliers come ONLY from the Step27C frozen
axis specification.

No filesystem scanning is used to discover slides.

OUTPUTS
-------
results/27d_tcga_prad_frozen_pc3_pc6_scores/

    scores/
        <slide_id>_pc3_pc6_scores.npy

    frozen_tcga_prad_pc3_pc6_score_manifest.csv
    frozen_tcga_prad_pc3_pc6_slide_summary.csv
    tcga_prad_pc3_pc6_score_materialization_specification.json
    tcga_prad_pc3_pc6_score_materialization_summary.json
    source_hashes.csv
    README.txt

Per-slide score array
---------------------
Shape:
    (n_tiles, 4)

Column order:
    column 0 = raw_PC3
    column 1 = oriented_PC3
    column 2 = raw_PC6
    column 3 = oriented_PC6

Coordinates are NOT duplicated.

The corresponding coords_path is retained in the manifest and tile order
is identical to Step27B / Step27A.

Downstream rule
---------------
All later TCGA PC3 / PC6 analyses should use the Step27D score arrays.

They should not independently:
    - reopen PC1-PC8 arrays to redefine score orientation,
    - multiply signs again,
    - re-project embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "27d-tcga-prad-pc3-pc6-"
    "score-materialization-schema-1.0"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


DEFAULT_STEP27B_MANIFEST = (
    RESULTS
    / "27b_tcga_prad_frozen_pca_projection"
    / "frozen_tcga_prad_pca_projection_manifest.csv"
)

DEFAULT_STEP27B_SPEC = (
    RESULTS
    / "27b_tcga_prad_frozen_pca_projection"
    / "tcga_prad_pca_projection_specification.json"
)

DEFAULT_STEP27B_SUMMARY = (
    RESULTS
    / "27b_tcga_prad_frozen_pca_projection"
    / "tcga_prad_pca_projection_summary.json"
)


DEFAULT_STEP27C_AXIS_SPEC = (
    RESULTS
    / "27c_tcga_prad_pc3_pc6_external_evaluation_specification"
    / "tcga_prad_pc3_pc6_frozen_axis_specification.csv"
)

DEFAULT_STEP27C_METADATA = (
    RESULTS
    / "27c_tcga_prad_pc3_pc6_external_evaluation_specification"
    / "tcga_prad_pc3_pc6_external_evaluation_specification.json"
)


DEFAULT_OUT_DIR = (
    RESULTS
    / "27d_tcga_prad_frozen_pc3_pc6_scores"
)


# =============================================================================
# EXPECTATIONS
# =============================================================================

EXPECTED_SLIDES = 403
EXPECTED_INPUT_COMPONENTS = 8

PC3_ZERO_INDEX = 2
PC6_ZERO_INDEX = 5

OUTPUT_COLUMNS = [
    "raw_PC3",
    "oriented_PC3",
    "raw_PC6",
    "oriented_PC6",
]


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
    width: int = 112,
) -> None:

    print()
    print("=" * width)
    print(title)
    print("=" * width)


def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} missing: {path}"
        )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def save_json_atomic(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def save_npy_atomic(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "wb",
    ) as handle:

        np.save(
            handle,
            array,
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp,
        path,
    )


# =============================================================================
# STEP27C AXIS AUTHORITY
# =============================================================================

def load_axis_specification(
    path: Path,
) -> pd.DataFrame:

    require_file(
        path,
        "Step27C axis specification",
    )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "axis",
        "axis_role",
        "frozen_direction",
        "score_multiplier",
        "direction_authority",
        "tcga_sign_tuning_allowed",
        "tcga_axis_reselection_allowed",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Step27C axis specification schema changed. "
            f"Missing columns: {missing}"
        )

    targets = (
        df.loc[
            df["axis"]
            .astype(str)
            .isin(
                ["PC3", "PC6"]
            )
        ]
        .copy()
    )

    if len(targets) != 2:
        raise RuntimeError(
            "Step27C must contain exactly one PC3 row "
            "and one PC6 row."
        )

    if (
        targets["axis"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate PC3 / PC6 rows in Step27C."
        )

    observed = set(
        targets[
            "axis"
        ].astype(str)
    )

    if observed != {
        "PC3",
        "PC6",
    }:
        raise RuntimeError(
            f"Unexpected Step27C axes: {observed}"
        )

    expected_roles = {
        "PC3": "PRIMARY_INTERPRETED_AXIS",
        "PC6": "SECONDARY_INTERPRETED_AXIS",
    }

    for axis in [
        "PC3",
        "PC6",
    ]:

        row = targets.loc[
            targets["axis"]
            == axis
        ].iloc[0]

        role = str(
            row["axis_role"]
        )

        if role != expected_roles[axis]:

            raise RuntimeError(
                f"{axis}: role changed: "
                f"{role!r}"
            )

        direction = str(
            row["frozen_direction"]
        )

        multiplier = float(
            row["score_multiplier"]
        )

        if direction not in {
            "high_in_tumor",
            "low_in_tumor",
        }:

            raise RuntimeError(
                f"{axis}: invalid frozen direction "
                f"{direction!r}"
            )

        expected_multiplier = (
            1.0
            if direction
            == "high_in_tumor"
            else -1.0
        )

        if multiplier != expected_multiplier:

            raise RuntimeError(
                f"{axis}: multiplier={multiplier} "
                f"inconsistent with "
                f"direction={direction}"
            )

        if str(
            row[
                "tcga_sign_tuning_allowed"
            ]
        ).strip().lower() not in {
            "false",
            "0",
            "no",
        }:

            raise RuntimeError(
                f"{axis}: TCGA sign tuning unexpectedly allowed."
            )

        if str(
            row[
                "tcga_axis_reselection_allowed"
            ]
        ).strip().lower() not in {
            "false",
            "0",
            "no",
        }:

            raise RuntimeError(
                f"{axis}: TCGA axis reselection unexpectedly allowed."
            )

    return (
        targets
        .sort_values(
            "axis"
        )
        .reset_index(
            drop=True
        )
    )


def get_multiplier(
    axis_spec: pd.DataFrame,
    axis: str,
) -> int:

    row = axis_spec.loc[
        axis_spec[
            "axis"
        ].astype(str)
        == axis
    ]

    if len(row) != 1:

        raise RuntimeError(
            f"Unable to resolve unique multiplier for {axis}"
        )

    multiplier = int(
        float(
            row.iloc[0][
                "score_multiplier"
            ]
        )
    )

    if multiplier not in {
        -1,
        1,
    }:

        raise RuntimeError(
            f"{axis}: invalid multiplier {multiplier}"
        )

    return multiplier


# =============================================================================
# STEP27B PROJECTION AUTHORITY
# =============================================================================

def load_projection_manifest(
    manifest_path: Path,
    spec_path: Path,
    summary_path: Path,
) -> pd.DataFrame:

    require_file(
        manifest_path,
        "Step27B projection manifest",
    )

    require_file(
        spec_path,
        "Step27B projection specification",
    )

    require_file(
        summary_path,
        "Step27B projection summary",
    )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "slide_id",
        "include_frozen_cohort",
        "n_tiles",
        "n_projected_components",
        "projection_dtype",
        "coords_path",
        "pc1_pc8_path",
    }

    missing = sorted(
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Step27B manifest schema changed. "
            f"Missing columns: {missing}"
        )

    if len(df) != EXPECTED_SLIDES:

        raise RuntimeError(
            "Step27B cohort size changed: "
            f"{len(df)} != {EXPECTED_SLIDES}"
        )

    if (
        df["slide_id"]
        .astype(str)
        .nunique()
        != EXPECTED_SLIDES
    ):

        raise RuntimeError(
            "Step27B does not contain exactly "
            f"{EXPECTED_SLIDES} unique slides."
        )

    components = pd.to_numeric(
        df[
            "n_projected_components"
        ],
        errors="raise",
    )

    if not (
        components
        == EXPECTED_INPUT_COMPONENTS
    ).all():

        raise RuntimeError(
            "Step27B input does not contain exactly "
            "PC1-PC8 for every slide."
        )

    spec = load_json(
        spec_path
    )

    orientation = (
        spec
        .get(
            "pc_orientation_policy",
            {},
        )
    )

    if bool(
        orientation.get(
            "sign_flipping_performed",
            True,
        )
    ):

        raise RuntimeError(
            "Step27B unexpectedly reports sign flipping."
        )

    if bool(
        orientation.get(
            "tumor_direction_used",
            True,
        )
    ):

        raise RuntimeError(
            "Step27B unexpectedly reports tumor-direction "
            "use during raw projection."
        )

    summary = load_json(
        summary_path
    )

    if str(
        summary.get(
            "status",
            "",
        )
    ) != "PASS":

        raise RuntimeError(
            "Step27B status is not PASS."
        )

    if int(
        summary.get(
            "unique_slides",
            -1,
        )
    ) != EXPECTED_SLIDES:

        raise RuntimeError(
            "Step27B unique-slide count changed."
        )

    return (
        df
        .copy()
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# EXISTING OUTPUT VALIDATION
# =============================================================================

def validate_existing_score_file(
    path: Path,
    expected_n_tiles: int,
) -> bool:

    if not path.is_file():
        return False

    try:

        arr = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if arr.shape != (
            expected_n_tiles,
            4,
        ):
            return False

        if arr.dtype != np.float32:
            return False

    except Exception:

        return False

    return True


# =============================================================================
# MATERIALIZATION
# =============================================================================

def materialize_scores(
    pc1_pc8_path: Path,
    n_tiles: int,
    pc3_multiplier: int,
    pc6_multiplier: int,
) -> np.ndarray:

    raw = np.load(
        pc1_pc8_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if raw.ndim != 2:

        raise RuntimeError(
            f"Bad Step27B array shape: "
            f"{raw.shape} | {pc1_pc8_path}"
        )

    if raw.shape != (
        n_tiles,
        EXPECTED_INPUT_COMPONENTS,
    ):

        raise RuntimeError(
            f"Step27B array shape mismatch: "
            f"{raw.shape}, expected "
            f"({n_tiles}, {EXPECTED_INPUT_COMPONENTS})"
        )

    raw_pc3 = np.asarray(
        raw[
            :,
            PC3_ZERO_INDEX,
        ],
        dtype=np.float32,
    )

    raw_pc6 = np.asarray(
        raw[
            :,
            PC6_ZERO_INDEX,
        ],
        dtype=np.float32,
    )

    if not np.isfinite(
        raw_pc3
    ).all():

        raise RuntimeError(
            f"PC3 contains NaN/Inf: {pc1_pc8_path}"
        )

    if not np.isfinite(
        raw_pc6
    ).all():

        raise RuntimeError(
            f"PC6 contains NaN/Inf: {pc1_pc8_path}"
        )

    oriented_pc3 = (
        raw_pc3
        * np.float32(
            pc3_multiplier
        )
    )

    oriented_pc6 = (
        raw_pc6
        * np.float32(
            pc6_multiplier
        )
    )

    output = np.empty(
        (
            n_tiles,
            4,
        ),
        dtype=np.float32,
    )

    output[
        :,
        0,
    ] = raw_pc3

    output[
        :,
        1,
    ] = oriented_pc3

    output[
        :,
        2,
    ] = raw_pc6

    output[
        :,
        3,
    ] = oriented_pc6

    return output


# =============================================================================
# TECHNICAL SUMMARY
# =============================================================================

def summarize_vector(
    values: np.ndarray,
    prefix: str,
) -> Dict[str, float]:

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.ndim != 1:

        raise RuntimeError(
            f"{prefix}: expected 1D vector."
        )

    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"{prefix}: NaN/Inf detected."
        )

    return {
        f"{prefix}_mean": float(
            np.mean(
                values
            )
        ),
        f"{prefix}_median": float(
            np.median(
                values
            )
        ),
        f"{prefix}_std": float(
            np.std(
                values
            )
        ),
        f"{prefix}_q01": float(
            np.quantile(
                values,
                0.01,
            )
        ),
        f"{prefix}_q05": float(
            np.quantile(
                values,
                0.05,
            )
        ),
        f"{prefix}_q10": float(
            np.quantile(
                values,
                0.10,
            )
        ),
        f"{prefix}_q25": float(
            np.quantile(
                values,
                0.25,
            )
        ),
        f"{prefix}_q75": float(
            np.quantile(
                values,
                0.75,
            )
        ),
        f"{prefix}_q90": float(
            np.quantile(
                values,
                0.90,
            )
        ),
        f"{prefix}_q95": float(
            np.quantile(
                values,
                0.95,
            )
        ),
        f"{prefix}_q99": float(
            np.quantile(
                values,
                0.99,
            )
        ),
        f"{prefix}_min": float(
            np.min(
                values
            )
        ),
        f"{prefix}_max": float(
            np.max(
                values
            )
        ),
    }


def summarize_score_array(
    scores: np.ndarray,
) -> Dict[str, float]:

    if (
        scores.ndim != 2
        or scores.shape[1] != 4
    ):

        raise RuntimeError(
            f"Unexpected score shape: "
            f"{scores.shape}"
        )

    output: Dict[str, float] = {}

    mapping = [
        (
            0,
            "raw_PC3",
        ),
        (
            1,
            "oriented_PC3",
        ),
        (
            2,
            "raw_PC6",
        ),
        (
            3,
            "oriented_PC6",
        ),
    ]

    for column, name in mapping:

        output.update(
            summarize_vector(
                scores[
                    :,
                    column,
                ],
                name,
            )
        )

    return output


# =============================================================================
# EXECUTION
# =============================================================================

def execute(
    projection_manifest: pd.DataFrame,
    axis_spec: pd.DataFrame,
    step27b_manifest_path: Path,
    step27b_spec_path: Path,
    step27b_summary_path: Path,
    step27c_axis_spec_path: Path,
    step27c_metadata_path: Path,
    out_dir: Path,
    overwrite: bool,
) -> None:

    score_dir = (
        out_dir
        / "scores"
    )

    score_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pc3_multiplier = get_multiplier(
        axis_spec,
        "PC3",
    )

    pc6_multiplier = get_multiplier(
        axis_spec,
        "PC6",
    )

    pc3_direction = str(
        axis_spec.loc[
            axis_spec["axis"]
            == "PC3",
            "frozen_direction",
        ].iloc[0]
    )

    pc6_direction = str(
        axis_spec.loc[
            axis_spec["axis"]
            == "PC6",
            "frozen_direction",
        ].iloc[0]
    )

    banner(
        "FROZEN AXIS MATERIALIZATION RULE"
    )

    print(
        f"PC3: direction={pc3_direction} | "
        f"multiplier={pc3_multiplier:+d}"
    )

    print(
        f"PC6: direction={pc6_direction} | "
        f"multiplier={pc6_multiplier:+d}"
    )

    print()
    print(
        "oriented_PC3 = raw_PC3 * "
        f"({pc3_multiplier:+d})"
    )

    print(
        "oriented_PC6 = raw_PC6 * "
        f"({pc6_multiplier:+d})"
    )

    manifest_rows: List[dict] = []
    summary_rows: List[dict] = []

    processed = 0
    skipped = 0
    total_tiles = 0

    banner(
        "MATERIALIZE TCGA PC3 / PC6 SCORES"
    )

    for index, row in projection_manifest.iterrows():

        slide_id = str(
            row["slide_id"]
        )

        n_tiles = int(
            row["n_tiles"]
        )

        pc1_pc8_path = Path(
            str(
                row[
                    "pc1_pc8_path"
                ]
            )
        )

        coords_path = Path(
            str(
                row[
                    "coords_path"
                ]
            )
        )

        require_file(
            pc1_pc8_path,
            f"{slide_id} Step27B projection",
        )

        require_file(
            coords_path,
            f"{slide_id} coordinates",
        )

        coords = np.load(
            coords_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if (
            coords.ndim != 2
            or coords.shape[1] != 2
        ):

            raise RuntimeError(
                f"{slide_id}: bad coordinate shape "
                f"{coords.shape}"
            )

        if coords.shape[0] != n_tiles:

            raise RuntimeError(
                f"{slide_id}: coordinate count "
                f"{coords.shape[0]} != "
                f"frozen n_tiles={n_tiles}"
            )

        score_path = (
            score_dir
            / f"{slide_id}_pc3_pc6_scores.npy"
        )

        print(
            f"[{index + 1:03d}/"
            f"{len(projection_manifest)}] "
            f"{slide_id} | "
            f"tiles={n_tiles:,}",
            end="",
        )

        if (
            not overwrite
            and validate_existing_score_file(
                score_path,
                n_tiles,
            )
        ):

            print(
                " | SKIP valid existing"
            )

            scores = np.load(
                score_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            status = (
                "skipped_valid_existing"
            )

            skipped += 1

        else:

            scores = materialize_scores(
                pc1_pc8_path=(
                    pc1_pc8_path
                ),
                n_tiles=n_tiles,
                pc3_multiplier=(
                    pc3_multiplier
                ),
                pc6_multiplier=(
                    pc6_multiplier
                ),
            )

            save_npy_atomic(
                score_path,
                scores,
            )

            saved = np.load(
                score_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if saved.shape != (
                n_tiles,
                4,
            ):

                raise RuntimeError(
                    f"{slide_id}: saved score shape "
                    f"mismatch: {saved.shape}"
                )

            if saved.dtype != np.float32:

                raise RuntimeError(
                    f"{slide_id}: saved dtype "
                    f"{saved.dtype}, expected float32"
                )

            # ----------------------------------------------------------
            # Exact mathematical consistency checks.
            # ----------------------------------------------------------

            raw_pc3 = saved[
                :,
                0,
            ]

            oriented_pc3 = saved[
                :,
                1,
            ]

            raw_pc6 = saved[
                :,
                2,
            ]

            oriented_pc6 = saved[
                :,
                3,
            ]

            if not np.array_equal(
                oriented_pc3,
                raw_pc3
                * np.float32(
                    pc3_multiplier
                ),
            ):

                raise RuntimeError(
                    f"{slide_id}: PC3 orientation "
                    "consistency check failed."
                )

            if not np.array_equal(
                oriented_pc6,
                raw_pc6
                * np.float32(
                    pc6_multiplier
                ),
            ):

                raise RuntimeError(
                    f"{slide_id}: PC6 orientation "
                    "consistency check failed."
                )

            scores = saved

            print(
                " | OK"
            )

            status = "processed"

            processed += 1

        total_tiles += n_tiles

        manifest_row = {
            "slide_id": slide_id,
            "include_frozen_cohort": True,
            "n_tiles": n_tiles,

            "pc3_axis_role": (
                "PRIMARY_INTERPRETED_AXIS"
            ),
            "pc3_frozen_direction": (
                pc3_direction
            ),
            "pc3_score_multiplier": (
                pc3_multiplier
            ),

            "pc6_axis_role": (
                "SECONDARY_INTERPRETED_AXIS"
            ),
            "pc6_frozen_direction": (
                pc6_direction
            ),
            "pc6_score_multiplier": (
                pc6_multiplier
            ),

            "score_array_columns": (
                "0=raw_PC3;"
                "1=oriented_PC3;"
                "2=raw_PC6;"
                "3=oriented_PC6"
            ),

            "score_dtype": "float32",

            "pc1_pc8_path": str(
                pc1_pc8_path.resolve()
            ),

            "coords_path": str(
                coords_path.resolve()
            ),

            "pc3_pc6_scores_path": str(
                score_path.resolve()
            ),

            "materialization_status": (
                status
            ),
        }

        passthrough = [
            "assigned_part",
            "canonical_part",
            "location_mismatch",
            "cross_directory_duplicate",
            "patient_id",
            "slide_type",
            "filename",
            "gdc_uuid",
            "file_path",
            "source_cohort_csv",
        ]

        for column in passthrough:

            if column in row.index:

                manifest_row[
                    column
                ] = row.get(
                    column
                )

        manifest_rows.append(
            manifest_row
        )

        technical_summary = {
            "slide_id": slide_id,
            "n_tiles": n_tiles,
        }

        technical_summary.update(
            summarize_score_array(
                np.asarray(
                    scores
                )
            )
        )

        summary_rows.append(
            technical_summary
        )

    # =========================================================================
    # Assemble outputs
    # =========================================================================

    manifest = pd.DataFrame(
        manifest_rows
    )

    slide_summary = pd.DataFrame(
        summary_rows
    )

    if len(
        manifest
    ) != EXPECTED_SLIDES:

        raise RuntimeError(
            f"Output manifest has {len(manifest)} rows; "
            f"expected {EXPECTED_SLIDES}."
        )

    if (
        manifest[
            "slide_id"
        ]
        .astype(str)
        .nunique()
        != EXPECTED_SLIDES
    ):

        raise RuntimeError(
            "Output manifest does not contain "
            "403 unique slides."
        )

    if len(
        slide_summary
    ) != EXPECTED_SLIDES:

        raise RuntimeError(
            "Slide summary row count mismatch."
        )

    if set(
        manifest[
            "slide_id"
        ].astype(str)
    ) != set(
        projection_manifest[
            "slide_id"
        ].astype(str)
    ):

        raise RuntimeError(
            "Step27D slide set differs from Step27B."
        )

    manifest_path = (
        out_dir
        / "frozen_tcga_prad_pc3_pc6_score_manifest.csv"
    )

    slide_summary_path = (
        out_dir
        / "frozen_tcga_prad_pc3_pc6_slide_summary.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    slide_summary.to_csv(
        slide_summary_path,
        index=False,
    )

    # =========================================================================
    # Hash sources
    # =========================================================================

    source_files = [
        (
            "step27b_projection_manifest",
            step27b_manifest_path,
        ),
        (
            "step27b_projection_specification",
            step27b_spec_path,
        ),
        (
            "step27b_projection_summary",
            step27b_summary_path,
        ),
        (
            "step27c_axis_specification",
            step27c_axis_spec_path,
        ),
        (
            "step27c_evaluation_metadata",
            step27c_metadata_path,
        ),
    ]

    hash_rows = []

    for label, path in source_files:

        require_file(
            path,
            label,
        )

        hash_rows.append({
            "source": label,
            "path": str(
                path.resolve()
            ),
            "sha256": (
                sha256_file(
                    path
                )
            ),
        })

    hashes = pd.DataFrame(
        hash_rows
    )

    hashes_path = (
        out_dir
        / "source_hashes.csv"
    )

    hashes.to_csv(
        hashes_path,
        index=False,
    )

    # =========================================================================
    # Specification
    # =========================================================================

    specification = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "analysis_role": (
            "FROZEN_TCGA_PC3_PC6_SCORE_MATERIALIZATION_ONLY"
        ),

        "input_authority": {
            "slide_and_projection_authority": (
                "Step27B"
            ),
            "axis_and_direction_authority": (
                "Step27C / Step06"
            ),
        },

        "cohort": {
            "expected_slides": (
                EXPECTED_SLIDES
            ),
            "observed_slides": int(
                len(manifest)
            ),
            "observed_unique_slides": int(
                manifest[
                    "slide_id"
                ].nunique()
            ),
            "total_tiles": int(
                total_tiles
            ),
        },

        "input_projection": {
            "source": (
                "Step27B raw discovery-PCA PC1-PC8"
            ),
            "input_components": [
                f"PC{i}"
                for i in range(
                    1,
                    9,
                )
            ],
            "pc3_zero_based_column": (
                PC3_ZERO_INDEX
            ),
            "pc6_zero_based_column": (
                PC6_ZERO_INDEX
            ),
        },

        "axis_roles": {
            "PC3": (
                "PRIMARY_INTERPRETED_AXIS"
            ),
            "PC6": (
                "SECONDARY_INTERPRETED_AXIS"
            ),
        },

        "frozen_directions": {
            "PC3": {
                "direction": (
                    pc3_direction
                ),
                "score_multiplier": (
                    pc3_multiplier
                ),
            },
            "PC6": {
                "direction": (
                    pc6_direction
                ),
                "score_multiplier": (
                    pc6_multiplier
                ),
            },
        },

        "score_definition": {
            "PC3": (
                "oriented_PC3 = "
                "raw_PC3 * frozen_PC3_score_multiplier"
            ),
            "PC6": (
                "oriented_PC6 = "
                "raw_PC6 * frozen_PC6_score_multiplier"
            ),
            "interpretation": (
                "Larger oriented score indicates the "
                "discovery-frozen tumor-associated direction."
            ),
        },

        "per_slide_array": {
            "dtype": "float32",
            "shape": (
                "(n_tiles, 4)"
            ),
            "columns": {
                "0": "raw_PC3",
                "1": "oriented_PC3",
                "2": "raw_PC6",
                "3": "oriented_PC6",
            },
            "tile_order": (
                "Identical to Step27B PC1-PC8 array "
                "and associated coords_path."
            ),
        },

        "technical_summary_only": {
            "slide_level_statistics": [
                "mean",
                "median",
                "std",
                "q01",
                "q05",
                "q10",
                "q25",
                "q75",
                "q90",
                "q95",
                "q99",
                "min",
                "max",
            ],
            "biological_inference_performed": False,
        },

        "firewall": {
            "embedding_reprojection": False,
            "pca_refit": False,
            "scaler_refit": False,
            "axis_reselection": False,
            "pc3_sign_tuning": False,
            "pc6_sign_tuning": False,
            "tumor_mask_read": False,
            "tumor_auc_calculated": False,
            "morphology_read": False,
            "rna_read": False,
            "gleason_read": False,
            "clinical_outcome_read": False,
            "survival_read": False,
            "threshold_selected": False,
            "case_selected": False,
            "figure_generated": False,
        },

        "downstream_rule": (
            "All TCGA PC3/PC6 analyses must use "
            "pc3_pc6_scores_path from the Step27D manifest. "
            "Do not independently reapply PCA or sign multipliers."
        ),
    }

    specification_path = (
        out_dir
        / "tcga_prad_pc3_pc6_score_materialization_specification.json"
    )

    save_json_atomic(
        specification_path,
        specification,
    )

    # =========================================================================
    # Summary
    # =========================================================================

    execution_summary = {
        "version": VERSION,
        "schema": SCHEMA,
        "status": "PASS",

        "slides_expected": (
            EXPECTED_SLIDES
        ),

        "slides_materialized": int(
            len(manifest)
        ),

        "unique_slides": int(
            manifest[
                "slide_id"
            ].nunique()
        ),

        "processed_this_run": int(
            processed
        ),

        "skipped_valid_existing": int(
            skipped
        ),

        "failed": 0,

        "total_tiles": int(
            total_tiles
        ),

        "score_array_columns": (
            OUTPUT_COLUMNS
        ),

        "score_dtype": "float32",

        "pc3_direction": (
            pc3_direction
        ),

        "pc3_multiplier": (
            pc3_multiplier
        ),

        "pc6_direction": (
            pc6_direction
        ),

        "pc6_multiplier": (
            pc6_multiplier
        ),

        "biological_outcomes_read": False,

        "manifest": str(
            manifest_path.resolve()
        ),

        "slide_summary": str(
            slide_summary_path.resolve()
        ),

        "specification": str(
            specification_path.resolve()
        ),

        "source_hashes": str(
            hashes_path.resolve()
        ),
    }

    execution_summary_path = (
        out_dir
        / "tcga_prad_pc3_pc6_score_materialization_summary.json"
    )

    save_json_atomic(
        execution_summary_path,
        execution_summary,
    )

    # =========================================================================
    # README
    # =========================================================================

    readme_path = (
        out_dir
        / "README.txt"
    )

    readme = f"""STEP 27D
TCGA-PRAD FROZEN PC3 / PC6 SCORE MATERIALIZATION

Version:
    {VERSION}

Schema:
    {SCHEMA}


COHORT
------
Slides:
    {len(manifest)}

Unique slides:
    {manifest['slide_id'].nunique()}

Total tiles:
    {total_tiles:,}


PC3
---
Role:
    PRIMARY_INTERPRETED_AXIS

Frozen direction:
    {pc3_direction}

Multiplier:
    {pc3_multiplier:+d}

Definition:
    oriented_PC3 = raw_PC3 * ({pc3_multiplier:+d})


PC6
---
Role:
    SECONDARY_INTERPRETED_AXIS

Frozen direction:
    {pc6_direction}

Multiplier:
    {pc6_multiplier:+d}

Definition:
    oriented_PC6 = raw_PC6 * ({pc6_multiplier:+d})


PER-SLIDE ARRAY
---------------
Shape:
    (n_tiles, 4)

Columns:
    0 = raw_PC3
    1 = oriented_PC3
    2 = raw_PC6
    3 = oriented_PC6

dtype:
    float32


TILE ALIGNMENT
--------------
The row order is identical to:
    Step27B pc1_pc8_path

Coordinates must be read from:
    coords_path

stored in:
    frozen_tcga_prad_pc3_pc6_score_manifest.csv


FIREWALL
--------
PCA refit:
    NO

Scaler refit:
    NO

Embedding re-projection:
    NO

Axis reselection:
    NO

PC3 sign tuning:
    NO

PC6 sign tuning:
    NO

Tumor outcome read:
    NO

Morphology outcome read:
    NO

RNA outcome read:
    NO

Clinical outcome read:
    NO

Threshold tuning:
    NO

Case selection:
    NO


DOWNSTREAM AUTHORITY
--------------------
All downstream TCGA PC3 / PC6 analyses should use:

    frozen_tcga_prad_pc3_pc6_score_manifest.csv

and its:

    pc3_pc6_scores_path

Do not independently re-project embeddings or reapply direction multipliers.
"""

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    # =========================================================================
    # Final audit
    # =========================================================================

    banner(
        "STEP 27D MATERIALIZATION AUDIT"
    )

    audit = [
        (
            "403 frozen slides inherited",
            "YES",
        ),
        (
            "Step27B raw projection used",
            "YES",
        ),
        (
            "Step27C PC3 multiplier inherited",
            "YES",
        ),
        (
            "Step27C PC6 multiplier inherited",
            "YES",
        ),
        (
            "Raw PC3 retained",
            "YES",
        ),
        (
            "Oriented PC3 materialized",
            "YES",
        ),
        (
            "Raw PC6 retained",
            "YES",
        ),
        (
            "Oriented PC6 materialized",
            "YES",
        ),
        (
            "PCA refit",
            "NO",
        ),
        (
            "Scaler refit",
            "NO",
        ),
        (
            "Embedding reprojection",
            "NO",
        ),
        (
            "Axis reselection",
            "NO",
        ),
        (
            "TCGA sign tuning",
            "NO",
        ),
        (
            "Tumor outcome read",
            "NO",
        ),
        (
            "Morphology outcome read",
            "NO",
        ),
        (
            "RNA outcome read",
            "NO",
        ),
        (
            "Clinical outcome read",
            "NO",
        ),
        (
            "Threshold selection",
            "NO",
        ),
        (
            "Visual case selection",
            "NO",
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<52}: {value}"
        )

    banner(
        "STEP 27D: PASS"
    )

    print(
        f"[SLIDES]         {len(manifest)}"
    )

    print(
        "[UNIQUE]         "
        f"{manifest['slide_id'].nunique()}"
    )

    print(
        f"[TOTAL TILES]    {total_tiles:,}"
    )

    print(
        f"[PROCESSED]      {processed}"
    )

    print(
        f"[SKIP EXISTING]  {skipped}"
    )

    print(
        "[FAILED]         0"
    )

    print()

    print(
        f"[PC3]            "
        f"direction={pc3_direction} | "
        f"multiplier={pc3_multiplier:+d}"
    )

    print(
        f"[PC6]            "
        f"direction={pc6_direction} | "
        f"multiplier={pc6_multiplier:+d}"
    )

    print()

    print(
        f"[MANIFEST]       {manifest_path}"
    )

    print(
        f"[SLIDE SUMMARY]  {slide_summary_path}"
    )

    print(
        f"[SPEC]           {specification_path}"
    )

    print(
        f"[SUMMARY]        {execution_summary_path}"
    )

    print(
        f"[HASHES]         {hashes_path}"
    )

    print(
        f"[README]         {readme_path}"
    )

    print()

    print(
        "Downstream rule:"
    )

    print(
        "  Read pc3_pc6_scores_path from the Step27D "
        "manifest."
    )

    print(
        "  Do not independently re-project embeddings "
        "or reapply PC3/PC6 sign multipliers."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Materialize frozen TCGA-PRAD PC3/PC6 "
            "raw and oriented scores."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--step27b_manifest",
        type=Path,
        default=DEFAULT_STEP27B_MANIFEST,
    )

    parser.add_argument(
        "--step27b_spec",
        type=Path,
        default=DEFAULT_STEP27B_SPEC,
    )

    parser.add_argument(
        "--step27b_summary",
        type=Path,
        default=DEFAULT_STEP27B_SUMMARY,
    )

    parser.add_argument(
        "--step27c_axis_spec",
        type=Path,
        default=DEFAULT_STEP27C_AXIS_SPEC,
    )

    parser.add_argument(
        "--step27c_metadata",
        type=Path,
        default=DEFAULT_STEP27C_METADATA,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recompute and replace valid existing "
            "per-slide PC3/PC6 score arrays."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    step27b_manifest = (
        args.step27b_manifest.resolve()
    )

    step27b_spec = (
        args.step27b_spec.resolve()
    )

    step27b_summary = (
        args.step27b_summary.resolve()
    )

    step27c_axis_spec = (
        args.step27c_axis_spec.resolve()
    )

    step27c_metadata = (
        args.step27c_metadata.resolve()
    )

    out_dir = (
        args.out_dir.resolve()
    )

    banner(
        "STEP 27D: MATERIALIZE FROZEN TCGA-PRAD "
        "PC3 / PC6 SCORES"
    )

    print(
        f"[VERSION]       {VERSION}"
    )

    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    print(
        f"[27B MANIFEST]  {step27b_manifest}"
    )

    print(
        f"[27C AXIS SPEC] {step27c_axis_spec}"
    )

    print(
        f"[OUTPUT]        {out_dir}"
    )

    print(
        f"[OVERWRITE]     {args.overwrite}"
    )

    axis_spec = (
        load_axis_specification(
            step27c_axis_spec
        )
    )

    require_file(
        step27c_metadata,
        "Step27C metadata",
    )

    metadata27c = load_json(
        step27c_metadata
    )

    if (
        metadata27c.get(
            "analysis_role"
        )
        != "PRE_EXECUTION_TCGA_EXTERNAL_EVALUATION_FREEZE_ONLY"
    ):

        raise RuntimeError(
            "Unexpected Step27C analysis role."
        )

    projection_manifest = (
        load_projection_manifest(
            manifest_path=(
                step27b_manifest
            ),
            spec_path=(
                step27b_spec
            ),
            summary_path=(
                step27b_summary
            ),
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    execute(
        projection_manifest=(
            projection_manifest
        ),
        axis_spec=(
            axis_spec
        ),
        step27b_manifest_path=(
            step27b_manifest
        ),
        step27b_spec_path=(
            step27b_spec
        ),
        step27b_summary_path=(
            step27b_summary
        ),
        step27c_axis_spec_path=(
            step27c_axis_spec
        ),
        step27c_metadata_path=(
            step27c_metadata
        ),
        out_dir=(
            out_dir
        ),
        overwrite=(
            args.overwrite
        ),
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        banner(
            "STEP 27D: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)