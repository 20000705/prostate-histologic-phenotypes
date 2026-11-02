#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27b_project_frozen_tcga_prad_pca.py

Project the frozen TCGA-PRAD external cohort into the discovery-fitted,
frozen Virchow2 PCA space.

ROLE
----
This script performs PROJECTION ONLY.

It reads:
    1. The frozen TCGA-PRAD external cohort manifest from Step 27A.
    2. The original discovery-cohort PCA model.

It writes:
    1. One PC1-PC8 score array per frozen TCGA slide.
    2. A projection manifest containing provenance and output paths.
    3. A technical per-slide PC summary table.
    4. A frozen projection specification and execution summary.

This script DOES NOT:
    - scan embedding directories to discover slides,
    - add or remove TCGA slides,
    - fit PCA,
    - refit the scaler,
    - orient PCs by tumor association,
    - flip PC3 or PC6 signs,
    - inspect tumor annotations,
    - inspect morphology,
    - inspect RNA / molecular outcomes,
    - create heatmaps,
    - perform statistical inference.

The projection is exactly:

    X_scaled = (X - scaler_mean) / scaler_scale
    X_pca    = (X_scaled - pca_mean) @ pca_components.T

Only PC1-PC8 are retained.

IMPORTANT
---------
The PCA orientation in this step is the RAW orientation stored in the
discovery-fitted PCA model.

Any later tumor-oriented sign convention for PC3 / PC6 must be applied
explicitly in a downstream frozen specification. It must NOT be silently
applied here.

Expected frozen cohort:
    403 unique TCGA-PRAD slides

Expected embedding dimension:
    2560

Outputs
-------
results/27b_tcga_prad_frozen_pca_projection/

    projected_pc1_pc8/
        <slide_id>_pc1_pc8.npy

    frozen_tcga_prad_pca_projection_manifest.csv
    frozen_tcga_prad_pc1_pc8_slide_summary.csv
    tcga_prad_pca_projection_specification.json
    tcga_prad_pca_projection_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
SCHEMA = "27b-tcga-prad-frozen-pca-projection-schema-1.0"


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = REPO_ROOT

DEFAULT_FROZEN_MANIFEST = (
    RESULTS_ROOT
    / "27a_tcga_prad_external_cohort_freeze"
    / "frozen_tcga_prad_external_cohort_manifest.csv"
)

DEFAULT_MODEL_NPZ = FROZEN_PCA_MODEL

DEFAULT_OUT_DIR = (
    RESULTS_ROOT
    / "27b_tcga_prad_frozen_pca_projection"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_SLIDES = 403
EXPECTED_EMBEDDING_DIM = 2560
N_COMPONENTS_TO_SAVE = 8

DEFAULT_CHUNK_SIZE = 50000


# =============================================================================
# HELPERS
# =============================================================================

def print_header(
    title: str,
    width: int = 100,
) -> None:
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def sha256_file(
    path: Path,
    chunk_bytes: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


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

    with open(
        tmp,
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

    with open(
        tmp,
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
# FROZEN COHORT
# =============================================================================

def load_frozen_manifest(
    path: Path,
) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen Step-27A manifest not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "slide_id",
        "include_frozen_cohort",
        "embeddings_path",
        "coords_path",
        "n_tiles",
        "embedding_dim",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Frozen manifest missing required columns: "
            f"{missing}"
        )

    if len(df) != EXPECTED_SLIDES:
        raise RuntimeError(
            "Frozen TCGA cohort size changed: "
            f"observed={len(df)}, "
            f"expected={EXPECTED_SLIDES}"
        )

    if (
        df["slide_id"]
        .astype(str)
        .duplicated()
        .any()
    ):
        duplicates = df.loc[
            df["slide_id"]
            .astype(str)
            .duplicated(keep=False),
            ["slide_id"],
        ]

        raise RuntimeError(
            "Duplicate slide IDs in frozen manifest:\n"
            + duplicates.to_string(
                index=False
            )
        )

    if (
        df["slide_id"]
        .astype(str)
        .nunique()
        != EXPECTED_SLIDES
    ):
        raise RuntimeError(
            "Frozen manifest does not contain exactly "
            f"{EXPECTED_SLIDES} unique slide IDs."
        )

    include_values = (
        df["include_frozen_cohort"]
        .astype(str)
        .str.lower()
    )

    valid_true = {
        "true",
        "1",
        "yes",
    }

    if not include_values.isin(
        valid_true
    ).all():
        bad = df.loc[
            ~include_values.isin(valid_true),
            [
                "slide_id",
                "include_frozen_cohort",
            ],
        ]

        raise RuntimeError(
            "Frozen manifest contains rows not included "
            "in the frozen cohort:\n"
            + bad.to_string(
                index=False
            )
        )

    if not (
        pd.to_numeric(
            df["embedding_dim"],
            errors="raise",
        )
        == EXPECTED_EMBEDDING_DIM
    ).all():
        bad = df.loc[
            pd.to_numeric(
                df["embedding_dim"],
                errors="coerce",
            )
            != EXPECTED_EMBEDDING_DIM,
            [
                "slide_id",
                "embedding_dim",
            ],
        ]

        raise RuntimeError(
            "Unexpected embedding dimension in frozen manifest:\n"
            + bad.to_string(
                index=False
            )
        )

    return (
        df
        .copy()
        .reset_index(drop=True)
    )


# =============================================================================
# PCA MODEL
# =============================================================================

def load_pca_model(
    path: Path,
) -> Dict[str, np.ndarray]:

    if not path.is_file():
        raise FileNotFoundError(
            f"PCA model not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as model:

        required = {
            "scaler_mean",
            "scaler_scale",
            "pca_mean",
            "pca_components",
        }

        missing = sorted(
            required - set(model.files)
        )

        if missing:
            raise ValueError(
                "PCA model missing required arrays: "
                f"{missing}"
            )

        output = {
            "scaler_mean": np.asarray(
                model["scaler_mean"],
                dtype=np.float32,
            ),
            "scaler_scale": np.asarray(
                model["scaler_scale"],
                dtype=np.float32,
            ),
            "pca_mean": np.asarray(
                model["pca_mean"],
                dtype=np.float32,
            ),
            "pca_components": np.asarray(
                model["pca_components"],
                dtype=np.float32,
            ),
        }

        # Preserve optional model information if available.
        optional_keys = [
            "explained_variance",
            "explained_variance_ratio",
            "singular_values",
        ]

        for key in optional_keys:
            if key in model.files:
                output[key] = np.asarray(
                    model[key]
                )

    dim = output[
        "scaler_mean"
    ].shape[0]

    if dim != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            "Discovery scaler dimension changed: "
            f"{dim} != {EXPECTED_EMBEDDING_DIM}"
        )

    if output[
        "scaler_scale"
    ].shape != (dim,):
        raise RuntimeError(
            "scaler_scale shape mismatch: "
            f"{output['scaler_scale'].shape}"
        )

    if output[
        "pca_mean"
    ].shape != (dim,):
        raise RuntimeError(
            "pca_mean shape mismatch: "
            f"{output['pca_mean'].shape}"
        )

    if output[
        "pca_components"
    ].ndim != 2:
        raise RuntimeError(
            "pca_components must be 2D"
        )

    if (
        output[
            "pca_components"
        ].shape[1]
        != dim
    ):
        raise RuntimeError(
            "PCA loading dimension mismatch: "
            f"{output['pca_components'].shape}"
        )

    if (
        output[
            "pca_components"
        ].shape[0]
        < N_COMPONENTS_TO_SAVE
    ):
        raise RuntimeError(
            "Discovery PCA contains fewer than "
            f"{N_COMPONENTS_TO_SAVE} components."
        )

    if not np.isfinite(
        output["scaler_mean"]
    ).all():
        raise RuntimeError(
            "scaler_mean contains NaN/Inf"
        )

    if not np.isfinite(
        output["scaler_scale"]
    ).all():
        raise RuntimeError(
            "scaler_scale contains NaN/Inf"
        )

    if np.any(
        output["scaler_scale"] == 0
    ):
        raise RuntimeError(
            "scaler_scale contains zero values"
        )

    if not np.isfinite(
        output["pca_mean"]
    ).all():
        raise RuntimeError(
            "pca_mean contains NaN/Inf"
        )

    if not np.isfinite(
        output["pca_components"]
    ).all():
        raise RuntimeError(
            "pca_components contains NaN/Inf"
        )

    return output


# =============================================================================
# PROJECTION
# =============================================================================

def project_embeddings_in_chunks(
    emb_path: Path,
    model: Dict[str, np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    """
    Apply the frozen discovery scaler and PCA model.

    Projection:
        X_scaled = (X - scaler_mean) / scaler_scale
        X_pca    = (X_scaled - pca_mean) @ pca_components.T

    Only PC1-PC8 are retained.
    """

    embeddings = np.load(
        emb_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Embedding array must be 2D: "
            f"{emb_path} | shape={embeddings.shape}"
        )

    n_tiles, dim = embeddings.shape

    if dim != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dim mismatch: "
            f"{emb_path} | "
            f"{dim} != {EXPECTED_EMBEDDING_DIM}"
        )

    if n_tiles <= 0:
        raise RuntimeError(
            f"Zero-tile embedding array: {emb_path}"
        )

    components = model[
        "pca_components"
    ][:N_COMPONENTS_TO_SAVE]

    output = np.empty(
        (
            n_tiles,
            N_COMPONENTS_TO_SAVE,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_tiles,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_tiles,
        )

        xb = np.asarray(
            embeddings[start:stop],
            dtype=np.float32,
        )

        if not np.isfinite(
            xb
        ).all():
            raise RuntimeError(
                f"Embedding NaN/Inf detected: "
                f"{emb_path} | rows={start}:{stop}"
            )

        xb_scaled = (
            xb
            - model["scaler_mean"]
        ) / model["scaler_scale"]

        xb_pca = (
            xb_scaled
            - model["pca_mean"]
        ) @ components.T

        if not np.isfinite(
            xb_pca
        ).all():
            raise RuntimeError(
                f"Projected PCA NaN/Inf detected: "
                f"{emb_path} | rows={start}:{stop}"
            )

        output[
            start:stop
        ] = xb_pca.astype(
            np.float32,
            copy=False,
        )

    return output


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_existing_projection(
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
            N_COMPONENTS_TO_SAVE,
        ):
            return False

        if arr.dtype != np.float32:
            return False

    except Exception:
        return False

    return True


# =============================================================================
# TECHNICAL SUMMARY
# =============================================================================

def summarize_projection(
    pca_scores: np.ndarray,
) -> Dict[str, float]:

    row: Dict[str, float] = {}

    for index in range(
        N_COMPONENTS_TO_SAVE
    ):
        values = pca_scores[
            :,
            index,
        ]

        pc = f"PC{index + 1}"

        row[
            f"{pc}_mean"
        ] = float(
            np.mean(values)
        )

        row[
            f"{pc}_median"
        ] = float(
            np.median(values)
        )

        row[
            f"{pc}_std"
        ] = float(
            np.std(values)
        )

        row[
            f"{pc}_q01"
        ] = float(
            np.quantile(
                values,
                0.01,
            )
        )

        row[
            f"{pc}_q05"
        ] = float(
            np.quantile(
                values,
                0.05,
            )
        )

        row[
            f"{pc}_q25"
        ] = float(
            np.quantile(
                values,
                0.25,
            )
        )

        row[
            f"{pc}_q75"
        ] = float(
            np.quantile(
                values,
                0.75,
            )
        )

        row[
            f"{pc}_q95"
        ] = float(
            np.quantile(
                values,
                0.95,
            )
        )

        row[
            f"{pc}_q99"
        ] = float(
            np.quantile(
                values,
                0.99,
            )
        )

        row[
            f"{pc}_min"
        ] = float(
            np.min(values)
        )

        row[
            f"{pc}_max"
        ] = float(
            np.max(values)
        )

    return row


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def execute_projection(
    frozen_manifest: pd.DataFrame,
    model: Dict[str, np.ndarray],
    model_path: Path,
    frozen_manifest_path: Path,
    out_dir: Path,
    chunk_size: int,
    overwrite: bool,
) -> None:

    projected_dir = (
        out_dir
        / "projected_pc1_pc8"
    )

    projected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows: List[dict] = []
    summary_rows: List[dict] = []

    processed = 0
    skipped_existing = 0

    total_tiles = 0

    print_header(
        "PROJECTING FROZEN TCGA-PRAD COHORT"
    )

    for index, row in frozen_manifest.iterrows():

        slide_id = str(
            row["slide_id"]
        )

        emb_path = Path(
            str(row["embeddings_path"])
        )

        coords_path = Path(
            str(row["coords_path"])
        )

        expected_n_tiles = int(
            row["n_tiles"]
        )

        if not emb_path.is_file():
            raise FileNotFoundError(
                f"{slide_id}: frozen embedding path missing: "
                f"{emb_path}"
            )

        if not coords_path.is_file():
            raise FileNotFoundError(
                f"{slide_id}: frozen coords path missing: "
                f"{coords_path}"
            )

        embeddings = np.load(
            emb_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        coords = np.load(
            coords_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"{slide_id}: bad embedding shape "
                f"{embeddings.shape}"
            )

        if coords.ndim != 2:
            raise RuntimeError(
                f"{slide_id}: bad coords shape "
                f"{coords.shape}"
            )

        if coords.shape[1] != 2:
            raise RuntimeError(
                f"{slide_id}: coords second dimension "
                f"is {coords.shape[1]}, expected 2"
            )

        if embeddings.shape[0] != coords.shape[0]:
            raise RuntimeError(
                f"{slide_id}: embeddings/coords tile mismatch: "
                f"{embeddings.shape[0]} vs {coords.shape[0]}"
            )

        if embeddings.shape[0] != expected_n_tiles:
            raise RuntimeError(
                f"{slide_id}: current embedding tile count "
                f"{embeddings.shape[0]} differs from frozen "
                f"manifest n_tiles={expected_n_tiles}"
            )

        if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"{slide_id}: embedding dim "
                f"{embeddings.shape[1]} != "
                f"{EXPECTED_EMBEDDING_DIM}"
            )

        output_path = (
            projected_dir
            / f"{slide_id}_pc1_pc8.npy"
        )

        print(
            f"[{index + 1:03d}/{len(frozen_manifest)}] "
            f"{slide_id} | "
            f"tiles={expected_n_tiles:,}",
            end="",
        )

        if (
            not overwrite
            and validate_existing_projection(
                output_path,
                expected_n_tiles,
            )
        ):
            print(
                " | SKIP valid existing"
            )

            pca_scores = np.load(
                output_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            status = "skipped_valid_existing"
            skipped_existing += 1

        else:
            pca_scores = (
                project_embeddings_in_chunks(
                    emb_path=emb_path,
                    model=model,
                    chunk_size=chunk_size,
                )
            )

            if pca_scores.shape != (
                expected_n_tiles,
                N_COMPONENTS_TO_SAVE,
            ):
                raise RuntimeError(
                    f"{slide_id}: unexpected projected shape "
                    f"{pca_scores.shape}"
                )

            save_npy_atomic(
                output_path,
                pca_scores,
            )

            # Reload after writing.
            saved = np.load(
                output_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            if saved.shape != pca_scores.shape:
                raise RuntimeError(
                    f"{slide_id}: saved projection shape mismatch"
                )

            if saved.dtype != np.float32:
                raise RuntimeError(
                    f"{slide_id}: saved projection dtype "
                    f"{saved.dtype}, expected float32"
                )

            print(
                " | OK"
            )

            pca_scores = saved
            status = "processed"
            processed += 1

        total_tiles += expected_n_tiles

        projection_row = {
            "slide_id": slide_id,
            "include_frozen_cohort": True,
            "assigned_part": row.get(
                "assigned_part",
                None,
            ),
            "canonical_part": row.get(
                "canonical_part",
                None,
            ),
            "location_mismatch": row.get(
                "location_mismatch",
                None,
            ),
            "cross_directory_duplicate": row.get(
                "cross_directory_duplicate",
                None,
            ),
            "n_tiles": expected_n_tiles,
            "embedding_dim": (
                EXPECTED_EMBEDDING_DIM
            ),
            "n_projected_components": (
                N_COMPONENTS_TO_SAVE
            ),
            "projection_dtype": "float32",
            "embeddings_path": str(
                emb_path.resolve()
            ),
            "coords_path": str(
                coords_path.resolve()
            ),
            "pc1_pc8_path": str(
                output_path.resolve()
            ),
            "projection_status": status,
        }

        # Preserve useful cohort identifiers if available.
        passthrough_columns = [
            "patient_id",
            "slide_type",
            "filename",
            "gdc_uuid",
            "file_path",
            "source_cohort_csv",
        ]

        for column in passthrough_columns:
            if column in row.index:
                projection_row[
                    column
                ] = row.get(
                    column
                )

        manifest_rows.append(
            projection_row
        )

        technical_summary = {
            "slide_id": slide_id,
            "n_tiles": expected_n_tiles,
        }

        technical_summary.update(
            summarize_projection(
                np.asarray(
                    pca_scores
                )
            )
        )

        summary_rows.append(
            technical_summary
        )

    projection_manifest = pd.DataFrame(
        manifest_rows
    )

    slide_summary = pd.DataFrame(
        summary_rows
    )

    if len(
        projection_manifest
    ) != EXPECTED_SLIDES:
        raise RuntimeError(
            "Projection manifest row count mismatch: "
            f"{len(projection_manifest)} != "
            f"{EXPECTED_SLIDES}"
        )

    if (
        projection_manifest[
            "slide_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate slide IDs in projection manifest"
        )

    if len(
        slide_summary
    ) != EXPECTED_SLIDES:
        raise RuntimeError(
            "PC summary row count mismatch"
        )

    manifest_path = (
        out_dir
        / "frozen_tcga_prad_pca_projection_manifest.csv"
    )

    slide_summary_path = (
        out_dir
        / "frozen_tcga_prad_pc1_pc8_slide_summary.csv"
    )

    projection_manifest.to_csv(
        manifest_path,
        index=False,
    )

    slide_summary.to_csv(
        slide_summary_path,
        index=False,
    )

    frozen_manifest_hash = (
        sha256_file(
            frozen_manifest_path
        )
    )

    model_hash = (
        sha256_file(
            model_path
        )
    )

    specification = {
        "version": VERSION,
        "schema": SCHEMA,
        "analysis_role": (
            "Projection of frozen TCGA-PRAD external "
            "Virchow2 embeddings into the frozen "
            "discovery PCA space"
        ),
        "input_cohort": {
            "frozen_manifest": str(
                frozen_manifest_path.resolve()
            ),
            "frozen_manifest_sha256": (
                frozen_manifest_hash
            ),
            "expected_slides": (
                EXPECTED_SLIDES
            ),
        },
        "pca_model": {
            "path": str(
                model_path.resolve()
            ),
            "sha256": model_hash,
            "embedding_dim": (
                EXPECTED_EMBEDDING_DIM
            ),
            "components_available": int(
                model[
                    "pca_components"
                ].shape[0]
            ),
            "components_saved": [
                f"PC{i}"
                for i in range(
                    1,
                    N_COMPONENTS_TO_SAVE + 1,
                )
            ],
        },
        "projection_formula": (
            "X_scaled = "
            "(X - scaler_mean) / scaler_scale; "
            "X_pca = "
            "(X_scaled - pca_mean) @ pca_components.T"
        ),
        "projection_dtype": "float32",
        "chunk_size": int(
            chunk_size
        ),
        "pc_orientation_policy": {
            "orientation": (
                "raw discovery-fitted PCA orientation"
            ),
            "sign_flipping_performed": False,
            "tumor_direction_used": False,
            "note": (
                "Any tumor-oriented PC3/PC6 sign convention "
                "must be applied explicitly downstream."
            ),
        },
        "forbidden_operations": [
            "No TCGA PCA fitting",
            "No TCGA scaler fitting",
            "No slide discovery from embedding directories",
            "No slide inclusion/exclusion",
            "No PC3/PC6 result-based selection",
            "No tumor outcome inspection",
            "No morphology outcome inspection",
            "No molecular outcome inspection",
        ],
    }

    specification_path = (
        out_dir
        / "tcga_prad_pca_projection_specification.json"
    )

    save_json_atomic(
        specification_path,
        specification,
    )

    execution_summary = {
        "version": VERSION,
        "schema": SCHEMA,
        "status": "PASS",
        "frozen_slides_expected": (
            EXPECTED_SLIDES
        ),
        "slides_in_projection_manifest": int(
            len(projection_manifest)
        ),
        "unique_slides": int(
            projection_manifest[
                "slide_id"
            ].nunique()
        ),
        "processed_this_run": int(
            processed
        ),
        "skipped_valid_existing": int(
            skipped_existing
        ),
        "failed": 0,
        "total_tiles": int(
            total_tiles
        ),
        "embedding_dim": (
            EXPECTED_EMBEDDING_DIM
        ),
        "components_saved": (
            N_COMPONENTS_TO_SAVE
        ),
        "projection_manifest": str(
            manifest_path.resolve()
        ),
        "slide_summary": str(
            slide_summary_path.resolve()
        ),
        "specification": str(
            specification_path.resolve()
        ),
        "frozen_manifest_sha256": (
            frozen_manifest_hash
        ),
        "pca_model_sha256": (
            model_hash
        ),
    }

    execution_summary_path = (
        out_dir
        / "tcga_prad_pca_projection_summary.json"
    )

    save_json_atomic(
        execution_summary_path,
        execution_summary,
    )

    print_header(
        "STEP 27B: FROZEN TCGA-PRAD PCA PROJECTION — PASS"
    )

    print(
        f"[VERSION]        {VERSION}"
    )
    print(
        f"[SCHEMA]         {SCHEMA}"
    )
    print(
        f"[SLIDES]         {len(projection_manifest)}"
    )
    print(
        f"[UNIQUE]         "
        f"{projection_manifest['slide_id'].nunique()}"
    )
    print(
        f"[PROCESSED]      {processed}"
    )
    print(
        f"[SKIP EXISTING]  {skipped_existing}"
    )
    print(
        "[FAILED]         0"
    )
    print(
        f"[TOTAL TILES]    {total_tiles:,}"
    )
    print(
        f"[DIM]            {EXPECTED_EMBEDDING_DIM}"
    )
    print(
        f"[PCS]            PC1-PC{N_COMPONENTS_TO_SAVE}"
    )
    print(
        "[SIGN FLIP]      False"
    )

    print()
    print(
        f"[MANIFEST]       {manifest_path}"
    )
    print(
        f"[PC SUMMARY]     {slide_summary_path}"
    )
    print(
        f"[SPEC]           {specification_path}"
    )
    print(
        f"[SUMMARY]        {execution_summary_path}"
    )

    print()
    print(
        "Downstream rule:"
    )
    print(
        "  Use pc1_pc8_path from the Step-27B projection "
        "manifest. Do not re-project embeddings independently."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Project the frozen 403-slide TCGA-PRAD "
            "external cohort into the frozen discovery PCA."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--frozen_manifest",
        type=Path,
        default=DEFAULT_FROZEN_MANIFEST,
    )

    parser.add_argument(
        "--model_npz",
        type=Path,
        default=DEFAULT_MODEL_NPZ,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Recompute and replace existing valid "
            "per-slide PC1-PC8 projection arrays."
        ),
    )

    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error(
            "--chunk_size must be >= 1"
        )

    return args


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    frozen_manifest_path = (
        args.frozen_manifest.resolve()
    )

    model_path = (
        args.model_npz.resolve()
    )

    out_dir = (
        args.out_dir.resolve()
    )

    print_header(
        "STEP 27B: PROJECT FROZEN TCGA-PRAD "
        "COHORT INTO DISCOVERY PCA"
    )

    print(
        f"[VERSION]      {VERSION}"
    )
    print(
        f"[SCHEMA]       {SCHEMA}"
    )
    print(
        f"[27A MANIFEST] {frozen_manifest_path}"
    )
    print(
        f"[PCA MODEL]    {model_path}"
    )
    print(
        f"[OUTPUT]       {out_dir}"
    )
    print(
        f"[CHUNK]        {args.chunk_size:,}"
    )
    print(
        f"[OVERWRITE]    {args.overwrite}"
    )

    frozen_manifest = (
        load_frozen_manifest(
            frozen_manifest_path
        )
    )

    print()
    print(
        f"[FROZEN COHORT] "
        f"{len(frozen_manifest)} slides"
    )
    print(
        f"[UNIQUE]        "
        f"{frozen_manifest['slide_id'].nunique()}"
    )

    model = load_pca_model(
        model_path
    )

    print()
    print(
        "[MODEL EMB DIM] "
        f"{model['pca_components'].shape[1]}"
    )
    print(
        "[MODEL PCS]     "
        f"{model['pca_components'].shape[0]}"
    )
    print(
        "[OUTPUT PCS]    PC1-PC8"
    )
    print(
        "[SIGN POLICY]   Raw discovery PCA orientation"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    execute_projection(
        frozen_manifest=frozen_manifest,
        model=model,
        model_path=model_path,
        frozen_manifest_path=(
            frozen_manifest_path
        ),
        out_dir=out_dir,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 100)
        print(
            "STEP 27B: FROZEN TCGA-PRAD "
            "PCA PROJECTION — FAIL"
        )
        print("=" * 100)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)