#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
27a_freeze_tcga_prad_external_cohort_manifest.py

Freeze the authoritative TCGA-PRAD external validation cohort and establish
one deterministic canonical Virchow2 embedding source per slide.

Purpose
-------
This script is a COHORT / PROVENANCE FREEZE ONLY.

It does NOT:
    - fit or refit PCA,
    - project embeddings,
    - inspect PC3 / PC6 values,
    - inspect tumor localization,
    - inspect morphology,
    - inspect molecular outcomes,
    - exclude slides based on downstream results,
    - modify, delete, move, or regenerate embeddings.

Authoritative cohort
--------------------
The TCGA-PRAD DX1 primary cohort is defined by the union of:

    part1: 200 slides
    part2: 203 slides

Expected total:
    403 unique slides

Canonical embedding rule
------------------------
For every authoritative slide:

1. Prefer the complete embedding output located in the output directory
   corresponding to the slide's prespecified cohort part.

2. If the assigned-part output is absent but exactly one complete output is
   available in the alternate output directory, use that valid pre-existing
   extraction and explicitly flag:
       location_mismatch = True

3. If a slide exists in both output directories, retain the slide only once.
   The assigned-part copy is selected deterministically.

4. No selection is permitted using PC3, PC6, PCA projection, morphology,
   tumor localization, molecular association, or any downstream endpoint.

5. Missing authoritative slides are fatal.

6. True extra slide outputs outside the authoritative 403-slide cohort are
   fatal, because they indicate output-directory contamination relative to
   the frozen cohort.

Expected extraction specification
---------------------------------
    model                    : Virchow2
    embedding dimension      : 2560
    patch size               : 224
    stride                   : 224
    source level             : 0
    tissue fraction threshold: 0.35
    embedding definition     :
        concat(output[:,0], mean(output[:,5:], dim=1))

Outputs
-------
results/27a_tcga_prad_external_cohort_freeze/

    frozen_tcga_prad_external_cohort_manifest.csv
    cross_directory_duplicate_audit.csv
    location_mismatch_audit.csv
    tcga_prad_external_cohort_freeze_specification.json
    tcga_prad_external_cohort_freeze_summary.json

Optional
--------
--deep_qc

When enabled, this scans every element of every canonical NumPy array and
checks for NaN / Inf. This is intentionally optional because the embedding
files are large and the structural/provenance freeze itself does not require
a full-array scan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"
SCHEMA = "27a-tcga-prad-external-cohort-freeze-schema-1.0"


# =============================================================================
# DEFAULT PATHS
# =============================================================================

PROJECT_ROOT = REPO_ROOT

TCGA_ROOT = CONFIG_TCGA_ROOT

DEFAULT_PART1_COHORT = (
    TCGA_ROOT / "tcga_prad_dx1_primary_cohort_part1_200.csv"
)

DEFAULT_PART2_COHORT = (
    TCGA_ROOT / "tcga_prad_dx1_primary_cohort_part2_203.csv"
)

DEFAULT_PART1_EMBEDDINGS = (
    TCGA_ROOT / "tcga_prad_virchow2_embeddings_part1"
)

DEFAULT_PART2_EMBEDDINGS = (
    TCGA_ROOT / "tcga_prad_virchow2_embeddings_part2"
)

DEFAULT_OUTPUT_DIR = (
    RESULTS_ROOT
    / "27a_tcga_prad_external_cohort_freeze"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_PART1_N = 200
EXPECTED_PART2_N = 203
EXPECTED_TOTAL_N = 403

EXPECTED_EMBEDDING_DIM = 2560
EXPECTED_PATCH_SIZE = 224
EXPECTED_STRIDE = 224
EXPECTED_SOURCE_LEVEL = 0

EXPECTED_TISSUE_FRACTION_THRESHOLD = 0.35
EXPECTED_PRESCREEN_THRESHOLD = 0.10

EXPECTED_MODEL = "Virchow2"
EXPECTED_MODEL_SOURCE = "hf-hub:paige-ai/Virchow2"

EXPECTED_EMBEDDING_DEFINITION = (
    "concat(output[:,0], mean(output[:,5:], dim=1))"
)


# =============================================================================
# HELPERS
# =============================================================================

def sanitize_id(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._")

    if not value:
        raise ValueError("Encountered empty slide identifier")

    return value


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp, path)


def output_paths(
    directory: Path,
    slide_id: str,
) -> Dict[str, Path]:

    return {
        "embeddings": directory / f"{slide_id}_embeddings.npy",
        "coords": directory / f"{slide_id}_coords.npy",
        "tissue_fraction": (
            directory / f"{slide_id}_tissue_fraction.npy"
        ),
        "meta": directory / f"{slide_id}_meta.json",
    }


def complete_output(
    directory: Path,
    slide_id: str,
) -> bool:

    paths = output_paths(directory, slide_id)
    return all(path.is_file() for path in paths.values())


def all_output_slide_ids(directory: Path) -> set[str]:
    """
    Define output slide IDs from *_embeddings.npy files.

    Completeness is evaluated separately.
    """
    return {
        path.name[:-len("_embeddings.npy")]
        for path in directory.glob("*_embeddings.npy")
    }


def values_equal_float(
    a,
    b,
    atol: float = 1e-12,
) -> bool:

    try:
        return math.isclose(
            float(a),
            float(b),
            rel_tol=0.0,
            abs_tol=atol,
        )
    except (TypeError, ValueError):
        return False


# =============================================================================
# COHORT
# =============================================================================

def load_authoritative_cohort(
    part1_csv: Path,
    part2_csv: Path,
) -> pd.DataFrame:

    datasets = []

    for part, path, expected_n in [
        ("part1", part1_csv, EXPECTED_PART1_N),
        ("part2", part2_csv, EXPECTED_PART2_N),
    ]:

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cohort CSV: {path}"
            )

        df = pd.read_csv(path).copy()

        if "slide_id" not in df.columns:
            raise ValueError(
                f"{path} does not contain slide_id"
            )

        if len(df) != expected_n:
            raise ValueError(
                f"{part} cohort row count changed: "
                f"observed={len(df)}, expected={expected_n}"
            )

        df["slide_id"] = (
            df["slide_id"]
            .astype(str)
            .map(sanitize_id)
        )

        if df["slide_id"].duplicated().any():
            duplicates = df.loc[
                df["slide_id"].duplicated(keep=False),
                ["slide_id"],
            ]

            raise ValueError(
                f"Duplicate slide IDs inside {part}:\n"
                + duplicates.to_string(index=False)
            )

        df["assigned_part"] = part
        df["source_cohort_csv"] = str(path.resolve())

        datasets.append(df)

    cohort = pd.concat(
        datasets,
        ignore_index=True,
        sort=False,
    )

    if len(cohort) != EXPECTED_TOTAL_N:
        raise RuntimeError(
            f"Combined cohort row count changed: "
            f"{len(cohort)} != {EXPECTED_TOTAL_N}"
        )

    duplicates = cohort[
        cohort["slide_id"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate slide IDs across authoritative parts:\n"
            + duplicates[
                ["slide_id", "assigned_part"]
            ].to_string(index=False)
        )

    if cohort["slide_id"].nunique() != EXPECTED_TOTAL_N:
        raise RuntimeError(
            "Authoritative cohort does not contain exactly "
            f"{EXPECTED_TOTAL_N} unique slide IDs."
        )

    return cohort.reset_index(drop=True)


# =============================================================================
# METADATA VALIDATION
# =============================================================================

def validate_meta(
    meta: dict,
    slide_id: str,
    n_tiles: int,
    embedding_dim: int,
) -> List[str]:

    errors: List[str] = []

    if str(meta.get("slide_id")) != slide_id:
        errors.append(
            f"meta.slide_id={meta.get('slide_id')!r}"
        )

    if meta.get("model") != EXPECTED_MODEL:
        errors.append(
            f"model={meta.get('model')!r}"
        )

    if meta.get("model_source") != EXPECTED_MODEL_SOURCE:
        errors.append(
            f"model_source={meta.get('model_source')!r}"
        )

    if (
        meta.get("embedding_definition")
        != EXPECTED_EMBEDDING_DEFINITION
    ):
        errors.append(
            "embedding_definition mismatch"
        )

    if meta.get("patch_size") != EXPECTED_PATCH_SIZE:
        errors.append(
            f"patch_size={meta.get('patch_size')!r}"
        )

    if meta.get("stride") != EXPECTED_STRIDE:
        errors.append(
            f"stride={meta.get('stride')!r}"
        )

    if meta.get("source_level") != EXPECTED_SOURCE_LEVEL:
        errors.append(
            f"source_level={meta.get('source_level')!r}"
        )

    if not values_equal_float(
        meta.get("tissue_fraction_threshold"),
        EXPECTED_TISSUE_FRACTION_THRESHOLD,
    ):
        errors.append(
            "tissue_fraction_threshold="
            f"{meta.get('tissue_fraction_threshold')!r}"
        )

    if not values_equal_float(
        meta.get("prescreen_tissue_fraction"),
        EXPECTED_PRESCREEN_THRESHOLD,
    ):
        errors.append(
            "prescreen_tissue_fraction="
            f"{meta.get('prescreen_tissue_fraction')!r}"
        )

    if meta.get("embedding_dim") != embedding_dim:
        errors.append(
            f"meta.embedding_dim={meta.get('embedding_dim')!r}, "
            f"array={embedding_dim}"
        )

    if meta.get("n_tiles") != n_tiles:
        errors.append(
            f"meta.n_tiles={meta.get('n_tiles')!r}, "
            f"array={n_tiles}"
        )

    return errors


# =============================================================================
# ARRAY VALIDATION
# =============================================================================

def validate_complete_output(
    directory: Path,
    slide_id: str,
    deep_qc: bool,
) -> dict:

    paths = output_paths(directory, slide_id)

    missing = [
        key
        for key, path in paths.items()
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{slide_id}: incomplete output in {directory}; "
            f"missing={missing}"
        )

    try:
        embeddings = np.load(
            paths["embeddings"],
            mmap_mode="r",
            allow_pickle=False,
        )

        coords = np.load(
            paths["coords"],
            mmap_mode="r",
            allow_pickle=False,
        )

        tissue = np.load(
            paths["tissue_fraction"],
            mmap_mode="r",
            allow_pickle=False,
        )

    except Exception as exc:
        raise RuntimeError(
            f"{slide_id}: unable to load NumPy outputs from "
            f"{directory}: {type(exc).__name__}: {exc}"
        ) from exc

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"{slide_id}: embeddings shape invalid: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"{slide_id}: embedding dimension changed: "
            f"{embeddings.shape[1]} != "
            f"{EXPECTED_EMBEDDING_DIM}"
        )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError(
            f"{slide_id}: coords shape invalid: "
            f"{coords.shape}"
        )

    if tissue.ndim != 1:
        raise RuntimeError(
            f"{slide_id}: tissue_fraction shape invalid: "
            f"{tissue.shape}"
        )

    n_tiles = int(embeddings.shape[0])

    if n_tiles <= 0:
        raise RuntimeError(
            f"{slide_id}: zero retained tiles"
        )

    if not (
        embeddings.shape[0]
        == coords.shape[0]
        == tissue.shape[0]
    ):
        raise RuntimeError(
            f"{slide_id}: row-count mismatch: "
            f"embeddings={embeddings.shape[0]}, "
            f"coords={coords.shape[0]}, "
            f"tissue={tissue.shape[0]}"
        )

    try:
        with open(
            paths["meta"],
            "r",
            encoding="utf-8",
        ) as handle:
            meta = json.load(handle)

    except Exception as exc:
        raise RuntimeError(
            f"{slide_id}: invalid metadata JSON: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    metadata_errors = validate_meta(
        meta=meta,
        slide_id=slide_id,
        n_tiles=n_tiles,
        embedding_dim=int(embeddings.shape[1]),
    )

    if metadata_errors:
        raise RuntimeError(
            f"{slide_id}: frozen extraction metadata mismatch "
            f"in {directory}:\n  - "
            + "\n  - ".join(metadata_errors)
        )

    if deep_qc:
        if not np.isfinite(embeddings).all():
            raise RuntimeError(
                f"{slide_id}: embeddings contain NaN/Inf"
            )

        if not np.isfinite(coords).all():
            raise RuntimeError(
                f"{slide_id}: coords contain NaN/Inf"
            )

        if not np.isfinite(tissue).all():
            raise RuntimeError(
                f"{slide_id}: tissue_fraction contains NaN/Inf"
            )

        if np.any(tissue < 0) or np.any(tissue > 1):
            raise RuntimeError(
                f"{slide_id}: tissue_fraction outside [0,1]"
            )

    return {
        "n_tiles": n_tiles,
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "coords_dtype": str(coords.dtype),
        "tissue_fraction_dtype": str(tissue.dtype),
        "slide_width": meta.get("slide_width"),
        "slide_height": meta.get("slide_height"),
        "objective_power": meta.get("objective_power"),
        "mpp_x": meta.get("mpp_x"),
        "mpp_y": meta.get("mpp_y"),
        "thumbnail_tissue_fraction": meta.get(
            "thumbnail_tissue_fraction"
        ),
        "patient_id_meta": meta.get("patient_id"),
        "slide_type_meta": meta.get("slide_type"),
        "source_slide_path_meta": meta.get("slide_path"),
        "filename_meta": meta.get("filename"),
        "gdc_uuid_meta": meta.get("gdc_uuid"),
    }


# =============================================================================
# MAIN FREEZE
# =============================================================================

def freeze_cohort(
    cohort: pd.DataFrame,
    part1_dir: Path,
    part2_dir: Path,
    output_dir: Path,
    deep_qc: bool,
) -> None:

    expected_ids = set(cohort["slide_id"])

    observed_part1 = all_output_slide_ids(part1_dir)
    observed_part2 = all_output_slide_ids(part2_dir)

    observed_union = observed_part1 | observed_part2

    missing_global = sorted(
        expected_ids - observed_union
    )

    true_extra = sorted(
        observed_union - expected_ids
    )

    if missing_global:
        raise RuntimeError(
            "Authoritative slides are missing from both output "
            "directories:\n"
            + "\n".join(
                f"  {slide_id}"
                for slide_id in missing_global
            )
        )

    if true_extra:
        raise RuntimeError(
            "True extra embedding slide IDs were found outside "
            "the authoritative cohort:\n"
            + "\n".join(
                f"  {slide_id}"
                for slide_id in true_extra
            )
        )

    directories = {
        "part1": part1_dir,
        "part2": part2_dir,
    }

    alternate_part = {
        "part1": "part2",
        "part2": "part1",
    }

    frozen_rows = []
    duplicate_rows = []
    mismatch_rows = []

    print()
    print("=" * 100)
    print("FREEZING CANONICAL TCGA-PRAD COHORT")
    print("=" * 100)

    for index, row in cohort.iterrows():

        slide_id = row["slide_id"]
        assigned_part = row["assigned_part"]

        assigned_dir = directories[assigned_part]

        other_part = alternate_part[assigned_part]
        other_dir = directories[other_part]

        assigned_complete = complete_output(
            assigned_dir,
            slide_id,
        )

        alternate_complete = complete_output(
            other_dir,
            slide_id,
        )

        present_part1 = complete_output(
            part1_dir,
            slide_id,
        )

        present_part2 = complete_output(
            part2_dir,
            slide_id,
        )

        is_cross_directory_duplicate = (
            present_part1 and present_part2
        )

        if assigned_complete:
            canonical_part = assigned_part
            canonical_dir = assigned_dir
            location_mismatch = False

            selection_reason = (
                "assigned_part_output"
                if not is_cross_directory_duplicate
                else
                "assigned_part_preferred_over_duplicate_copy"
            )

        elif alternate_complete:
            canonical_part = other_part
            canonical_dir = other_dir
            location_mismatch = True

            selection_reason = (
                "assigned_part_missing_use_unique_valid_"
                "alternate_output"
            )

        else:
            raise RuntimeError(
                f"{slide_id}: no complete output available "
                f"in either part directory"
            )

        qc = validate_complete_output(
            directory=canonical_dir,
            slide_id=slide_id,
            deep_qc=deep_qc,
        )

        paths = output_paths(
            canonical_dir,
            slide_id,
        )

        frozen = row.to_dict()

        frozen.update({
            "include_frozen_cohort": True,
            "canonical_part": canonical_part,
            "canonical_output_dir": str(
                canonical_dir.resolve()
            ),
            "canonical_selection_reason": (
                selection_reason
            ),
            "location_mismatch": bool(
                location_mismatch
            ),
            "cross_directory_duplicate": bool(
                is_cross_directory_duplicate
            ),
            "present_part1": bool(
                present_part1
            ),
            "present_part2": bool(
                present_part2
            ),
            "n_tiles": qc["n_tiles"],
            "embedding_dim": qc[
                "embedding_dim"
            ],
            "embedding_dtype": qc[
                "embedding_dtype"
            ],
            "coords_dtype": qc[
                "coords_dtype"
            ],
            "tissue_fraction_dtype": qc[
                "tissue_fraction_dtype"
            ],
            "slide_width": qc[
                "slide_width"
            ],
            "slide_height": qc[
                "slide_height"
            ],
            "objective_power": qc[
                "objective_power"
            ],
            "mpp_x": qc["mpp_x"],
            "mpp_y": qc["mpp_y"],
            "thumbnail_tissue_fraction": qc[
                "thumbnail_tissue_fraction"
            ],
            "embeddings_path": str(
                paths["embeddings"].resolve()
            ),
            "coords_path": str(
                paths["coords"].resolve()
            ),
            "tissue_fraction_path": str(
                paths["tissue_fraction"].resolve()
            ),
            "meta_path": str(
                paths["meta"].resolve()
            ),
        })

        frozen_rows.append(frozen)

        if is_cross_directory_duplicate:
            duplicate_rows.append({
                "slide_id": slide_id,
                "assigned_part": assigned_part,
                "canonical_part": canonical_part,
                "part1_output_dir": str(
                    part1_dir.resolve()
                ),
                "part2_output_dir": str(
                    part2_dir.resolve()
                ),
                "canonical_selection_reason": (
                    selection_reason
                ),
                "action": (
                    "retain_one_canonical_copy;"
                    "do_not_exclude_slide"
                ),
            })

        if location_mismatch:
            mismatch_rows.append({
                "slide_id": slide_id,
                "assigned_part": assigned_part,
                "canonical_part": canonical_part,
                "assigned_output_complete": False,
                "alternate_output_complete": True,
                "canonical_selection_reason": (
                    selection_reason
                ),
                "action": (
                    "retain_valid_preexisting_alternate_"
                    "extraction"
                ),
            })

        print(
            f"[{index + 1:03d}/{len(cohort)}] "
            f"{slide_id} | "
            f"assigned={assigned_part} | "
            f"canonical={canonical_part} | "
            f"tiles={qc['n_tiles']:,} | "
            f"duplicate={is_cross_directory_duplicate} | "
            f"mismatch={location_mismatch}"
        )

    frozen_df = pd.DataFrame(frozen_rows)

    if len(frozen_df) != EXPECTED_TOTAL_N:
        raise RuntimeError(
            f"Frozen cohort count is {len(frozen_df)}, "
            f"expected {EXPECTED_TOTAL_N}"
        )

    if frozen_df["slide_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate slide IDs remain in frozen manifest"
        )

    if frozen_df["slide_id"].nunique() != EXPECTED_TOTAL_N:
        raise RuntimeError(
            "Frozen manifest does not contain exactly "
            f"{EXPECTED_TOTAL_N} unique slides"
        )

    if not frozen_df[
        "include_frozen_cohort"
    ].all():
        raise RuntimeError(
            "Unexpected excluded rows in frozen cohort"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "frozen_tcga_prad_external_cohort_manifest.csv"
    )

    duplicate_path = (
        output_dir
        / "cross_directory_duplicate_audit.csv"
    )

    mismatch_path = (
        output_dir
        / "location_mismatch_audit.csv"
    )

    frozen_df.to_csv(
        manifest_path,
        index=False,
    )

    pd.DataFrame(
        duplicate_rows,
        columns=[
            "slide_id",
            "assigned_part",
            "canonical_part",
            "part1_output_dir",
            "part2_output_dir",
            "canonical_selection_reason",
            "action",
        ],
    ).to_csv(
        duplicate_path,
        index=False,
    )

    pd.DataFrame(
        mismatch_rows,
        columns=[
            "slide_id",
            "assigned_part",
            "canonical_part",
            "assigned_output_complete",
            "alternate_output_complete",
            "canonical_selection_reason",
            "action",
        ],
    ).to_csv(
        mismatch_path,
        index=False,
    )

    specification = {
        "version": VERSION,
        "schema": SCHEMA,
        "analysis_role": (
            "TCGA-PRAD external cohort and embedding "
            "provenance freeze"
        ),
        "authoritative_cohort": {
            "part1_expected_slides": (
                EXPECTED_PART1_N
            ),
            "part2_expected_slides": (
                EXPECTED_PART2_N
            ),
            "total_expected_slides": (
                EXPECTED_TOTAL_N
            ),
            "part1_cohort_csv": str(
                DEFAULT_PART1_COHORT.resolve()
            ),
            "part2_cohort_csv": str(
                DEFAULT_PART2_COHORT.resolve()
            ),
        },
        "canonicalization_rule": {
            "primary_rule": (
                "Use the complete output in the slide's "
                "prespecified assigned part."
            ),
            "duplicate_rule": (
                "If complete outputs exist in both directories, "
                "retain exactly one canonical copy using the "
                "assigned-part output."
            ),
            "location_mismatch_rule": (
                "If assigned-part output is absent but exactly "
                "one complete valid alternate output exists, "
                "retain that pre-existing alternate extraction "
                "and flag location_mismatch=True."
            ),
            "downstream_result_blinding": [
                "No PCA score inspected",
                "No PC3 value inspected",
                "No PC6 value inspected",
                "No tumor localization inspected",
                "No morphology inspected",
                "No molecular association inspected",
            ],
            "physical_file_policy": (
                "Original outputs are retained unchanged; "
                "canonicalization occurs only through the "
                "frozen manifest."
            ),
        },
        "frozen_extraction_expectations": {
            "model": EXPECTED_MODEL,
            "model_source": (
                EXPECTED_MODEL_SOURCE
            ),
            "embedding_dim": (
                EXPECTED_EMBEDDING_DIM
            ),
            "patch_size": (
                EXPECTED_PATCH_SIZE
            ),
            "stride": EXPECTED_STRIDE,
            "source_level": (
                EXPECTED_SOURCE_LEVEL
            ),
            "tissue_fraction_threshold": (
                EXPECTED_TISSUE_FRACTION_THRESHOLD
            ),
            "prescreen_threshold": (
                EXPECTED_PRESCREEN_THRESHOLD
            ),
            "embedding_definition": (
                EXPECTED_EMBEDDING_DEFINITION
            ),
        },
        "deep_qc_enabled": bool(
            deep_qc
        ),
    }

    specification_path = (
        output_dir
        / "tcga_prad_external_cohort_freeze_specification.json"
    )

    save_json_atomic(
        specification_path,
        specification,
    )

    n_duplicates = int(
        frozen_df[
            "cross_directory_duplicate"
        ].sum()
    )

    n_mismatches = int(
        frozen_df[
            "location_mismatch"
        ].sum()
    )

    total_tiles = int(
        frozen_df["n_tiles"].sum()
    )

    summary = {
        "version": VERSION,
        "schema": SCHEMA,
        "status": "PASS",
        "authoritative_rows": int(
            len(cohort)
        ),
        "authoritative_unique_slides": int(
            cohort["slide_id"].nunique()
        ),
        "frozen_slides": int(
            len(frozen_df)
        ),
        "frozen_unique_slides": int(
            frozen_df["slide_id"].nunique()
        ),
        "missing_authoritative_slides": 0,
        "true_extra_output_slides": 0,
        "cross_directory_duplicate_slides": (
            n_duplicates
        ),
        "location_mismatch_slides": (
            n_mismatches
        ),
        "excluded_slides": 0,
        "total_tiles": total_tiles,
        "embedding_dim": (
            EXPECTED_EMBEDDING_DIM
        ),
        "deep_qc_enabled": bool(
            deep_qc
        ),
        "frozen_manifest": str(
            manifest_path.resolve()
        ),
        "duplicate_audit": str(
            duplicate_path.resolve()
        ),
        "location_mismatch_audit": str(
            mismatch_path.resolve()
        ),
        "specification": str(
            specification_path.resolve()
        ),
    }

    summary_path = (
        output_dir
        / "tcga_prad_external_cohort_freeze_summary.json"
    )

    save_json_atomic(
        summary_path,
        summary,
    )

    print()
    print("=" * 100)
    print("TCGA-PRAD EXTERNAL COHORT FREEZE: PASS")
    print("=" * 100)

    print(
        f"[VERSION]        {VERSION}"
    )
    print(
        f"[SCHEMA]         {SCHEMA}"
    )
    print(
        f"[AUTHORITATIVE]  {len(cohort)} slides"
    )
    print(
        f"[FROZEN]         {len(frozen_df)} slides"
    )
    print(
        "[MISSING]        0"
    )
    print(
        "[TRUE EXTRA]     0"
    )
    print(
        f"[DUPLICATES]     {n_duplicates}"
    )
    print(
        f"[LOC MISMATCH]   {n_mismatches}"
    )
    print(
        "[EXCLUDED]       0"
    )
    print(
        f"[TOTAL TILES]    {total_tiles:,}"
    )
    print(
        f"[DEEP QC]        {deep_qc}"
    )

    print()
    print(
        f"[MANIFEST]       {manifest_path}"
    )
    print(
        f"[DUP AUDIT]      {duplicate_path}"
    )
    print(
        f"[LOC AUDIT]      {mismatch_path}"
    )
    print(
        f"[SPEC]           {specification_path}"
    )
    print(
        f"[SUMMARY]        {summary_path}"
    )

    print()
    print(
        "Frozen downstream rule:"
    )
    print(
        "  All subsequent TCGA analyses must read the "
        "canonical embedding paths from the frozen manifest."
    )
    print(
        "  Do not rediscover slides by scanning part1/part2 "
        "embedding directories."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the authoritative 403-slide TCGA-PRAD "
            "external Virchow2 cohort."
        )
    )

    parser.add_argument(
        "--part1_cohort",
        type=Path,
        default=DEFAULT_PART1_COHORT,
    )

    parser.add_argument(
        "--part2_cohort",
        type=Path,
        default=DEFAULT_PART2_COHORT,
    )

    parser.add_argument(
        "--part1_embeddings",
        type=Path,
        default=DEFAULT_PART1_EMBEDDINGS,
    )

    parser.add_argument(
        "--part2_embeddings",
        type=Path,
        default=DEFAULT_PART2_EMBEDDINGS,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--deep_qc",
        action="store_true",
        help=(
            "Scan all canonical arrays for NaN/Inf. "
            "This can be slow for the full TCGA cohort."
        ),
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:

    args = parse_args()

    part1_cohort = (
        args.part1_cohort.resolve()
    )

    part2_cohort = (
        args.part2_cohort.resolve()
    )

    part1_embeddings = (
        args.part1_embeddings.resolve()
    )

    part2_embeddings = (
        args.part2_embeddings.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    print("=" * 100)
    print(
        "STEP 27A: FREEZE TCGA-PRAD EXTERNAL COHORT "
        "AND CANONICAL EMBEDDING PROVENANCE"
    )
    print("=" * 100)

    print(f"[VERSION]      {VERSION}")
    print(f"[SCHEMA]       {SCHEMA}")
    print(f"[PART1 CSV]    {part1_cohort}")
    print(f"[PART2 CSV]    {part2_cohort}")
    print(f"[PART1 EMB]    {part1_embeddings}")
    print(f"[PART2 EMB]    {part2_embeddings}")
    print(f"[OUTPUT]       {output_dir}")
    print(f"[DEEP QC]      {args.deep_qc}")

    for directory in [
        part1_embeddings,
        part2_embeddings,
    ]:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Embedding directory not found: {directory}"
            )

    cohort = load_authoritative_cohort(
        part1_csv=part1_cohort,
        part2_csv=part2_cohort,
    )

    print()
    print("=" * 100)
    print("AUTHORITATIVE COHORT")
    print("=" * 100)

    print(
        f"Rows               : {len(cohort)}"
    )
    print(
        "Unique slide IDs   : "
        f"{cohort['slide_id'].nunique()}"
    )
    print(
        "Part1              : "
        f"{(cohort['assigned_part'] == 'part1').sum()}"
    )
    print(
        "Part2              : "
        f"{(cohort['assigned_part'] == 'part2').sum()}"
    )

    freeze_cohort(
        cohort=cohort,
        part1_dir=part1_embeddings,
        part2_dir=part2_embeddings,
        output_dir=output_dir,
        deep_qc=args.deep_qc,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print("=" * 100)
        print("TCGA-PRAD EXTERNAL COHORT FREEZE: FAIL")
        print("=" * 100)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)