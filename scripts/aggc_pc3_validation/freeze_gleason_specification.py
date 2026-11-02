#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18a_freeze_aggc_pc3_gleason_validation_specification.py

STEP 18A
========
Freeze the AGGC2022 Subset-1 PC3–Gleason / histologic-state validation
specification BEFORE formal outcome analysis.

THIS STEP IS FREEZE-ONLY.

It may read:
    - preflight WSI / mask / geometry / embedding-PCA technical metadata

It must NOT read:
    - PC3 values
    - any prior PC3-vs-Gleason result
    - any historical exploratory PC3 outcome
    - any PC1–PC8 screening result
    - any Test PC3-vs-label outcome


SCIENTIFIC HIERARCHY FROZEN HERE
================================

PRIMARY
-------
Clinically relevant contrast:
    Gleason pattern 3 vs Gleason pattern 4

Design:
    within-slide paired

Effect:
    median_PC3_G4 - median_PC3_G3

Independent unit:
    slide

Direction:
    none preassigned

Inference:
    two-sided


EXPLORATORY 1 — G5 EXTENSIONS
-----------------------------
Within-slide paired:
    G5 - G4
    G5 - G3

G5 is exploratory only and cannot be promoted after results.


EXPLORATORY 2 — TUMOR-ONLY ORDINAL
-----------------------------------
G3 -> G4 -> G5

Allowed only as exploratory/supportive.

For slides with all three technically valid patterns:
    median PC3 in G3
    median PC3 in G4
    median PC3 in G5

Then:
    Spearman rho(
        class order [3,4,5],
        class-level median PC3
    )

This is NOT a primary endpoint.


EXPLORATORY 3 — EXTENDED HISTOLOGIC-STATE PROFILE
--------------------------------------------------
Stroma -> Normal -> G3 -> G4 -> G5

IMPORTANT:
    This is NOT called:
        disease progression
        Gleason ordinal progression
        biological progression

Why:
    Stroma is a tissue compartment,
    Normal is benign epithelium,
    G3/G4/G5 are tumor differentiation patterns.

Therefore this five-state sequence is frozen only as an exploratory ordered
histologic-state profile.

Two outputs are prespecified:

A. State-profile description
   For each histologic state, summarize slide-level median PC3 using all slides
   technically eligible for that state.

   This does NOT require all five states to be present in the same slide.

B. All-five complete-case ordered-state summary
   For slides containing technically valid:
       Stroma + Normal + G3 + G4 + G5

   compute within-slide Spearman rho between:
       state order [0,1,2,3,4]
   and:
       median PC3 for [Stroma, Normal, G3, G4, G5]

   This is exploratory only.

No monotonic direction is prespecified.


TRAIN / TEST
------------
Train:
    formal external validation

Test:
    separate replication

The exact same frozen protocol must be applied to Test.
No Test-specific tuning is permitted.


ANNOTATION SEMANTICS
--------------------
AGGC masks are sparse positive annotation layers designed primarily for
automatic Gleason grading.

Therefore:
    explicit positive mask support
        = annotated state / class

    absence of mask support
        != biological negative

Unannotated tissue is not assigned a label.

Overlapping annotation support is never silently reassigned.


NO PC SCREENING
---------------
Only frozen discovery-cohort PC3 is permitted.

PC1–PC8 screening is forbidden.
PC6 belongs to a separate future module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
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

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.1.0"
SCHEMA_VERSION = "18a-aggc-pc3-gleason-validation-schema-1.1"


# =============================================================================
# PATHS
# =============================================================================

MEDIA_BASE = REPO_ROOT

AGGC_BASE = CONFIG_AGGC_ROOT

TRAIN_PREFLIGHT = (
    AGGC_BASE
    / "step17_preflight_aggc_subset1_audit"
)

TEST_PREFLIGHT = (
    AGGC_BASE
    / "step17_preflight_aggc_subset1_test_audit"
)

TRAIN_WSI_MANIFEST = TRAIN_PREFLIGHT / "aggc_subset1_wsi_manifest.csv"
TRAIN_EMB_PCA_AUDIT = TRAIN_PREFLIGHT / "aggc_subset1_embedding_pca_audit.csv"
TRAIN_MASK_MANIFEST = TRAIN_PREFLIGHT / "aggc_subset1_mask_manifest.csv"
TRAIN_GEOM_AUDIT = TRAIN_PREFLIGHT / "aggc_subset1_geometry_audit.csv"
TRAIN_MASK_ENCODING = TRAIN_PREFLIGHT / "aggc_subset1_mask_encoding_audit.csv"

TEST_WSI_MANIFEST = TEST_PREFLIGHT / "aggc_subset1_test_wsi_manifest.csv"
TEST_EMB_PCA_AUDIT = TEST_PREFLIGHT / "aggc_subset1_test_embedding_pca_audit.csv"
TEST_MASK_MANIFEST = TEST_PREFLIGHT / "aggc_subset1_test_mask_manifest.csv"
TEST_GEOM_AUDIT = TEST_PREFLIGHT / "aggc_subset1_test_geometry_audit.csv"
TEST_MASK_ENCODING = TEST_PREFLIGHT / "aggc_subset1_test_mask_encoding_audit.csv"

# IMPORTANT:
# use a NEW versioned output folder so the already-frozen v1.0.0 package remains intact.
OUT_DIR = (
    RESULTS_ROOT
    / "18a_aggc_pc3_gleason_histologic_state_specification"
)

OUT_TRAIN_PRIMARY = OUT_DIR / "aggc_pc3_g3_g4_train_validation_cohort.csv"
OUT_TEST_PRIMARY = OUT_DIR / "aggc_pc3_g3_g4_test_replication_cohort.csv"

OUT_TRAIN_G5 = OUT_DIR / "aggc_pc3_g5_train_exploratory_cohort.csv"
OUT_TEST_G5 = OUT_DIR / "aggc_pc3_g5_test_exploratory_cohort.csv"

OUT_TRAIN_EXTENDED = OUT_DIR / "aggc_pc3_extended_state_train_exploratory_cohort.csv"
OUT_TEST_EXTENDED = OUT_DIR / "aggc_pc3_extended_state_test_exploratory_cohort.csv"

OUT_PARTITION_MANIFEST = OUT_DIR / "aggc_pc3_gleason_frozen_partition_manifest.csv"

OUT_PLAN = OUT_DIR / "aggc_pc3_gleason_analysis_plan.json"
OUT_LABEL_SPEC = OUT_DIR / "aggc_pc3_gleason_label_specification.json"
OUT_GEOM_SPEC = OUT_DIR / "aggc_pc3_gleason_geometry_specification.json"
OUT_COHORT_SUMMARY = OUT_DIR / "aggc_pc3_gleason_cohort_summary.csv"

OUT_PROVENANCE = OUT_DIR / "evidence_provenance.csv"
OUT_HASHES = OUT_DIR / "source_hashes.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"


# =============================================================================
# BEHAVIOR / FROZEN CONSTANTS
# =============================================================================

OVERWRITE = False

TARGET_PC = "PC3"
PC_INDEX_ZERO_BASED = 2

PRIMARY_CONTRAST = "G4_minus_G3"
PRIMARY_WITHIN_SLIDE_SUMMARY = "median"
PRIMARY_INDEPENDENT_UNIT = "slide"
PRIMARY_EXPECTED_DIRECTION = "none_preassigned_two_sided"

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_SEED = 18032026

# Technical tile/class rules frozen before outcome analysis.
MIN_CLASS_POSITIVE_PIXELS_PER_TILE = 1
MIN_CLASS_FRACTION_PER_TILE = 0.10
MIN_CLASS_TILES_PER_SLIDE = 50

EXCLUDE_OVERLAPPING_CLASS_SUPPORT = True

GLEASON_LABELS = ["G3", "G4", "G5"]
EXTENDED_STATE_LABELS = ["Stroma", "Normal", "G3", "G4", "G5"]

EXTENDED_STATE_ORDER = {
    "Stroma": 0,
    "Normal": 1,
    "G3": 2,
    "G4": 3,
    "G5": 4,
}


# =============================================================================
# HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required {label}: {path}"
        )


def prepare_output():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        if not OVERWRITE:
            raise RuntimeError(
                f"Step18a v1.1.0 output already exists: {OUT_DIR}\n"
                "A frozen specification must not be silently overwritten."
            )

        for p in OUT_DIR.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_json(path: Path, obj):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def to_bool(x) -> bool:
    if pd.isna(x):
        return False

    if isinstance(
        x,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(x)

    s = str(x).strip().lower()

    if s in {
        "true",
        "1",
        "1.0",
        "yes",
        "y",
    }:
        return True

    if s in {
        "false",
        "0",
        "0.0",
        "no",
        "n",
        "",
    }:
        return False

    raise ValueError(
        f"Cannot parse bool: {x!r}"
    )


def normalize_path_value(x) -> Optional[str]:
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s == "":
        return None

    return s


def normalize_label(x: str) -> str:
    s = str(x).strip()

    mapping = {
        "Gleason3": "G3",
        "Gleason4": "G4",
        "Gleason5": "G5",
        "Gleason_3": "G3",
        "Gleason_4": "G4",
        "Gleason_5": "G5",
        "normal": "Normal",
        "Normal": "Normal",
        "stroma": "Stroma",
        "Stroma": "Stroma",
    }

    return mapping.get(
        s,
        s,
    )


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
    label: str = "",
) -> Optional[str]:

    exact = {
        str(c): str(c)
        for c in df.columns
    }

    lower = {
        str(c).lower(): str(c)
        for c in df.columns
    }

    for c in candidates:

        if c in exact:
            return exact[c]

        if c.lower() in lower:
            return lower[c.lower()]

    if required:
        raise RuntimeError(
            f"Could not find required column for "
            f"{label or list(candidates)}.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


# =============================================================================
# NORMALIZE PREFLIGHT TABLES
# =============================================================================

def normalize_wsi_manifest(
    path: Path,
    partition: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    sid_col = find_column(
        df,
        ["slide_id", "prefix"],
        label="slide_id",
    )

    wsi_path_col = find_column(
        df,
        [
            "wsi_path",
            "slide_path",
            "image_path",
        ],
        required=False,
    )

    width_col = find_column(
        df,
        [
            "wsi_width",
            "slide_width",
            "image_width",
            "width",
        ],
        required=False,
    )

    height_col = find_column(
        df,
        [
            "wsi_height",
            "slide_height",
            "image_height",
            "height",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "slide_id":
                df[sid_col].astype(str),

            "partition":
                partition,
        }
    )

    out["wsi_path"] = (
        df[wsi_path_col].map(normalize_path_value)
        if wsi_path_col is not None
        else None
    )

    out["wsi_width"] = (
        pd.to_numeric(
            df[width_col],
            errors="coerce",
        )
        if width_col is not None
        else np.nan
    )

    out["wsi_height"] = (
        pd.to_numeric(
            df[height_col],
            errors="coerce",
        )
        if height_col is not None
        else np.nan
    )

    if out["slide_id"].duplicated().any():
        raise RuntimeError(
            f"{partition}: duplicated slide IDs in WSI manifest."
        )

    return out


def normalize_embedding_pca_audit(
    path: Path,
    partition: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    sid_col = find_column(
        df,
        [
            "slide_id",
            "prefix",
        ],
        label="slide_id",
    )

    emb_exists_col = find_column(
        df,
        [
            "embedding_exists",
            "embeddings_exists",
        ],
        label="embedding_exists",
    )

    coords_exists_col = find_column(
        df,
        [
            "coords_exists",
            "coordinates_exists",
        ],
        label="coords_exists",
    )

    pca_exists_col = find_column(
        df,
        [
            "pca_exists",
            "frozen_pca_exists",
            "pca_projection_exists",
        ],
        label="pca_exists",
    )

    consistent_col = find_column(
        df,
        [
            "embedding_coord_pca_rows_consistent",
            "embedding_coords_pca_rows_consistent",
            "rows_consistent",
            "shape_consistent",
            "embedding_coord_pca_consistent",
        ],
        required=False,
    )

    emb_path_col = find_column(
        df,
        [
            "embedding_path",
            "embeddings_path",
        ],
        required=False,
    )

    coords_path_col = find_column(
        df,
        [
            "coords_path",
            "coordinates_path",
        ],
        required=False,
    )

    pca_path_col = find_column(
        df,
        [
            "pca_path",
            "pca_targets_path",
            "frozen_pca_path",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "slide_id":
                df[sid_col].astype(str),

            "partition":
                partition,

            "embedding_exists":
                df[emb_exists_col].map(to_bool),

            "coords_exists":
                df[coords_exists_col].map(to_bool),

            "pca_exists":
                df[pca_exists_col].map(to_bool),
        }
    )

    if consistent_col is not None:

        out[
            "embedding_coord_pca_rows_consistent"
        ] = (
            df[consistent_col].map(to_bool)
        )

    else:

        out[
            "embedding_coord_pca_rows_consistent"
        ] = (
            out["embedding_exists"]
            &
            out["coords_exists"]
            &
            out["pca_exists"]
        )

    out["embedding_path"] = (
        df[emb_path_col].map(normalize_path_value)
        if emb_path_col is not None
        else None
    )

    out["coords_path"] = (
        df[coords_path_col].map(normalize_path_value)
        if coords_path_col is not None
        else None
    )

    out["pca_path"] = (
        df[pca_path_col].map(normalize_path_value)
        if pca_path_col is not None
        else None
    )

    if out["slide_id"].duplicated().any():
        raise RuntimeError(
            f"{partition}: duplicated slide IDs in embedding/PCA audit."
        )

    return out


def normalize_mask_manifest(
    path: Path,
    partition: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    sid_col = find_column(
        df,
        [
            "slide_id",
            "prefix",
        ],
        label="slide_id",
    )

    label_col = find_column(
        df,
        [
            "label",
            "mask_label",
            "class_name",
            "class",
        ],
        label="mask label",
    )

    exists_col = find_column(
        df,
        [
            "mask_exists",
            "exists",
        ],
        label="mask_exists",
    )

    path_col = find_column(
        df,
        [
            "mask_path",
            "path",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "slide_id":
                df[sid_col].astype(str),

            "partition":
                partition,

            "label":
                df[label_col].map(normalize_label),

            "mask_exists":
                df[exists_col].map(to_bool),
        }
    )

    out["mask_path"] = (
        df[path_col].map(normalize_path_value)
        if path_col is not None
        else None
    )

    return out


def normalize_geometry_audit(
    path: Path,
    partition: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    sid_col = find_column(
        df,
        [
            "slide_id",
            "prefix",
        ],
        label="slide_id",
    )

    label_col = find_column(
        df,
        [
            "label",
            "mask_label",
            "class_name",
            "class",
        ],
        required=False,
    )

    compatible_col = find_column(
        df,
        [
            "direct_geometry_compatible",
            "geometry_compatible",
            "same_dimensions_as_wsi",
            "same_dimensions",
        ],
        label="geometry compatibility",
    )

    scale_x_col = find_column(
        df,
        [
            "scale_x",
            "mask_to_wsi_scale_x",
        ],
        required=False,
    )

    scale_y_col = find_column(
        df,
        [
            "scale_y",
            "mask_to_wsi_scale_y",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "slide_id":
                df[sid_col].astype(str),

            "partition":
                partition,

            "geometry_compatible":
                df[compatible_col].map(to_bool),
        }
    )

    out["label"] = (
        df[label_col].map(normalize_label)
        if label_col is not None
        else ""
    )

    out["scale_x"] = (
        pd.to_numeric(
            df[scale_x_col],
            errors="coerce",
        )
        if scale_x_col is not None
        else np.nan
    )

    out["scale_y"] = (
        pd.to_numeric(
            df[scale_y_col],
            errors="coerce",
        )
        if scale_y_col is not None
        else np.nan
    )

    return out


def normalize_mask_encoding(
    path: Path,
    partition: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    sid_col = find_column(
        df,
        [
            "slide_id",
            "prefix",
        ],
        label="slide_id",
    )

    label_col = find_column(
        df,
        [
            "label",
            "mask_label",
            "class_name",
            "class",
        ],
        required=False,
    )

    binary_col = find_column(
        df,
        [
            "binary_compatible",
            "mask_binary_compatible",
        ],
        required=False,
    )

    mode_col = find_column(
        df,
        [
            "pillow_mode",
            "mode",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "slide_id":
                df[sid_col].astype(str),

            "partition":
                partition,
        }
    )

    out["label"] = (
        df[label_col].map(normalize_label)
        if label_col is not None
        else ""
    )

    if binary_col is not None:

        out["binary_compatible"] = (
            df[binary_col].map(to_bool)
        )

    elif mode_col is not None:

        # tolerate CSV round-trip 1 / 1.0 / "1"
        mode_numeric = pd.to_numeric(
            df[mode_col],
            errors="coerce",
        )

        out["binary_compatible"] = (
            mode_numeric.eq(1.0)
        )

    else:

        raise RuntimeError(
            f"{partition}: no authoritative binary mask field in {path}"
        )

    out["pillow_mode"] = (
        df[mode_col].astype(str)
        if mode_col is not None
        else np.nan
    )

    return out


# =============================================================================
# CLASS-LEVEL TECHNICAL READINESS
# =============================================================================

def build_class_readiness(
    slide_ids: List[str],
    mask_df: pd.DataFrame,
    geom_df: pd.DataFrame,
    encoding_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for sid in slide_ids:

        row = {
            "slide_id":
                sid,
        }

        for label in EXTENDED_STATE_LABELS:

            key = label.lower()

            m = mask_df.loc[
                mask_df["slide_id"].eq(sid)
                &
                mask_df["label"].eq(label)
            ]

            mask_exists = bool(
                len(m) == 1
                and
                to_bool(
                    m.iloc[0]["mask_exists"]
                )
            )

            mask_path = (
                normalize_path_value(
                    m.iloc[0]["mask_path"]
                )
                if len(m) == 1
                else None
            )

            g = geom_df.loc[
                geom_df["slide_id"].eq(sid)
            ]

            if (
                "label" in g.columns
                and
                g["label"].astype(str).str.len().gt(0).any()
            ):

                g_label = g.loc[
                    g["label"].eq(label)
                ]

            else:

                g_label = g

            geometry_compatible = bool(
                len(g_label) >= 1
                and
                g_label[
                    "geometry_compatible"
                ].map(to_bool).all()
            )

            e = encoding_df.loc[
                encoding_df["slide_id"].eq(sid)
            ]

            if (
                "label" in e.columns
                and
                e["label"].astype(str).str.len().gt(0).any()
            ):

                e_label = e.loc[
                    e["label"].eq(label)
                ]

            else:

                e_label = e

            binary_compatible = bool(
                len(e_label) >= 1
                and
                e_label[
                    "binary_compatible"
                ].map(to_bool).all()
            )

            row[
                f"{key}_mask_exists"
            ] = mask_exists

            row[
                f"{key}_mask_path"
            ] = mask_path

            row[
                f"{key}_geometry_compatible"
            ] = geometry_compatible

            row[
                f"{key}_binary_compatible"
            ] = binary_compatible

            row[
                f"{key}_technically_ready"
            ] = bool(
                mask_exists
                and
                geometry_compatible
                and
                binary_compatible
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# PARTITION FREEZE
# =============================================================================

def build_partition(
    partition: str,
    wsi_path: Path,
    emb_path: Path,
    mask_path: Path,
    geom_path: Path,
    encoding_path: Path,
) -> pd.DataFrame:

    wsi = normalize_wsi_manifest(
        wsi_path,
        partition,
    )

    emb = normalize_embedding_pca_audit(
        emb_path,
        partition,
    )

    masks = normalize_mask_manifest(
        mask_path,
        partition,
    )

    geom = normalize_geometry_audit(
        geom_path,
        partition,
    )

    encoding = normalize_mask_encoding(
        encoding_path,
        partition,
    )

    class_ready = build_class_readiness(
        slide_ids=wsi["slide_id"].tolist(),
        mask_df=masks,
        geom_df=geom,
        encoding_df=encoding,
    )

    df = (
        wsi
        .merge(
            emb,
            on=[
                "slide_id",
                "partition",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            class_ready,
            on="slide_id",
            how="left",
            validate="one_to_one",
        )
    )

    bool_cols = [
        "embedding_exists",
        "coords_exists",
        "pca_exists",
        "embedding_coord_pca_rows_consistent",
    ]

    for label in EXTENDED_STATE_LABELS:

        key = label.lower()

        bool_cols.extend(
            [
                f"{key}_mask_exists",
                f"{key}_geometry_compatible",
                f"{key}_binary_compatible",
                f"{key}_technically_ready",
            ]
        )

    for col in bool_cols:

        if col not in df.columns:

            df[col] = False

        else:

            df[col] = (
                df[col]
                .fillna(False)
                .map(to_bool)
            )

    df[
        "base_embedding_pca_ready"
    ] = (
        df["embedding_exists"]
        &
        df["coords_exists"]
        &
        df["pca_exists"]
        &
        df["embedding_coord_pca_rows_consistent"]
    )

    # -------------------------------------------------------------------------
    # PRIMARY
    # -------------------------------------------------------------------------

    df[
        "formal_18a_primary_g3_g4_include"
    ] = (
        df["base_embedding_pca_ready"]
        &
        df["g3_technically_ready"]
        &
        df["g4_technically_ready"]
    )

    # -------------------------------------------------------------------------
    # G5 exploratory
    # -------------------------------------------------------------------------

    df[
        "formal_18a_g4_g5_exploratory_include"
    ] = (
        df["base_embedding_pca_ready"]
        &
        df["g4_technically_ready"]
        &
        df["g5_technically_ready"]
    )

    df[
        "formal_18a_g3_g5_exploratory_include"
    ] = (
        df["base_embedding_pca_ready"]
        &
        df["g3_technically_ready"]
        &
        df["g5_technically_ready"]
    )

    df[
        "formal_18a_g3_g4_g5_ordinal_exploratory_include"
    ] = (
        df["base_embedding_pca_ready"]
        &
        df["g3_technically_ready"]
        &
        df["g4_technically_ready"]
        &
        df["g5_technically_ready"]
    )

    # -------------------------------------------------------------------------
    # Extended histologic-state profile
    # -------------------------------------------------------------------------

    for label in EXTENDED_STATE_LABELS:

        key = label.lower()

        df[
            f"formal_18a_state_{key}_profile_include"
        ] = (
            df["base_embedding_pca_ready"]
            &
            df[f"{key}_technically_ready"]
        )

    df[
        "formal_18a_extended_all5_complete_case_include"
    ] = (
        df["base_embedding_pca_ready"]
        &
        df["stroma_technically_ready"]
        &
        df["normal_technically_ready"]
        &
        df["g3_technically_ready"]
        &
        df["g4_technically_ready"]
        &
        df["g5_technically_ready"]
    )

    # -------------------------------------------------------------------------
    # Primary technical exclusion reason
    # -------------------------------------------------------------------------

    reasons = []

    for row in df.itertuples(
        index=False
    ):

        r = []

        if not row.base_embedding_pca_ready:
            r.append(
                "embedding_coords_pca_not_ready"
            )

        if not row.g3_technically_ready:
            r.append(
                "G3_not_technically_ready"
            )

        if not row.g4_technically_ready:
            r.append(
                "G4_not_technically_ready"
            )

        reasons.append(
            "|".join(r)
        )

    df[
        "primary_exclusion_reason"
    ] = reasons

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 124)
    print("STEP 18A: FREEZE AGGC PC3 GLEASON / HISTOLOGIC-STATE VALIDATION")
    print("=" * 124)

    print("[VERSION]         ", VERSION)
    print("[SCHEMA]          ", SCHEMA_VERSION)
    print("[TRAIN PREFLIGHT] ", TRAIN_PREFLIGHT)
    print("[TEST PREFLIGHT]  ", TEST_PREFLIGHT)
    print("[OUTPUT]          ", OUT_DIR)
    print("[OVERWRITE]       ", OVERWRITE)

    print()
    print("[ROLE]            FREEZE ONLY")
    print("[PC VALUES]       NOT READ")
    print("[ASSOCIATION]     NOT CALCULATED")
    print("[TEST OUTCOME]    NOT READ")
    print("[PCA REFIT]       NONE")
    print("[PC SCREENING]    NONE")
    print("[PC6]             NONE")
    print("[INFERENCE]       NONE")

    source_files = [
        TRAIN_WSI_MANIFEST,
        TRAIN_EMB_PCA_AUDIT,
        TRAIN_MASK_MANIFEST,
        TRAIN_GEOM_AUDIT,
        TRAIN_MASK_ENCODING,
        TEST_WSI_MANIFEST,
        TEST_EMB_PCA_AUDIT,
        TEST_MASK_MANIFEST,
        TEST_GEOM_AUDIT,
        TEST_MASK_ENCODING,
    ]

    for p in source_files:
        require_file(
            p,
            p.name,
        )

    prepare_output()

    train = build_partition(
        partition="train_validation",
        wsi_path=TRAIN_WSI_MANIFEST,
        emb_path=TRAIN_EMB_PCA_AUDIT,
        mask_path=TRAIN_MASK_MANIFEST,
        geom_path=TRAIN_GEOM_AUDIT,
        encoding_path=TRAIN_MASK_ENCODING,
    )

    test = build_partition(
        partition="test_replication",
        wsi_path=TEST_WSI_MANIFEST,
        emb_path=TEST_EMB_PCA_AUDIT,
        mask_path=TEST_MASK_MANIFEST,
        geom_path=TEST_GEOM_AUDIT,
        encoding_path=TEST_MASK_ENCODING,
    )

    # -------------------------------------------------------------------------
    # Frozen cohorts
    # -------------------------------------------------------------------------

    train_primary = train.loc[
        train[
            "formal_18a_primary_g3_g4_include"
        ]
    ].copy()

    test_primary = test.loc[
        test[
            "formal_18a_primary_g3_g4_include"
        ]
    ].copy()

    train_g5 = train.loc[
        train[
            [
                "formal_18a_g4_g5_exploratory_include",
                "formal_18a_g3_g5_exploratory_include",
                "formal_18a_g3_g4_g5_ordinal_exploratory_include",
            ]
        ].any(
            axis=1
        )
    ].copy()

    test_g5 = test.loc[
        test[
            [
                "formal_18a_g4_g5_exploratory_include",
                "formal_18a_g3_g5_exploratory_include",
                "formal_18a_g3_g4_g5_ordinal_exploratory_include",
            ]
        ].any(
            axis=1
        )
    ].copy()

    train_extended = train.loc[
        train[
            [
                "formal_18a_state_stroma_profile_include",
                "formal_18a_state_normal_profile_include",
                "formal_18a_state_g3_profile_include",
                "formal_18a_state_g4_profile_include",
                "formal_18a_state_g5_profile_include",
                "formal_18a_extended_all5_complete_case_include",
            ]
        ].any(
            axis=1
        )
    ].copy()

    test_extended = test.loc[
        test[
            [
                "formal_18a_state_stroma_profile_include",
                "formal_18a_state_normal_profile_include",
                "formal_18a_state_g3_profile_include",
                "formal_18a_state_g4_profile_include",
                "formal_18a_state_g5_profile_include",
                "formal_18a_extended_all5_complete_case_include",
            ]
        ].any(
            axis=1
        )
    ].copy()

    if len(train_primary) == 0:
        raise RuntimeError(
            "No technically eligible Train G3/G4 paired slides."
        )

    if len(test_primary) == 0:
        raise RuntimeError(
            "No technically eligible Test G3/G4 paired slides."
        )

    train_primary.to_csv(
        OUT_TRAIN_PRIMARY,
        index=False,
    )

    test_primary.to_csv(
        OUT_TEST_PRIMARY,
        index=False,
    )

    train_g5.to_csv(
        OUT_TRAIN_G5,
        index=False,
    )

    test_g5.to_csv(
        OUT_TEST_G5,
        index=False,
    )

    train_extended.to_csv(
        OUT_TRAIN_EXTENDED,
        index=False,
    )

    test_extended.to_csv(
        OUT_TEST_EXTENDED,
        index=False,
    )

    partition_manifest = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    partition_manifest.to_csv(
        OUT_PARTITION_MANIFEST,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Frozen analysis plan
    # -------------------------------------------------------------------------

    analysis_plan = {
        "step":
            "18a",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "analysis_status":
            "frozen_before_formal_PC3_label_outcome_analysis",

        "latent_representation":
            {
                "target":
                    TARGET_PC,

                "zero_based_component_index":
                    PC_INDEX_ZERO_BASED,

                "source":
                    "discovery-cohort frozen PCA projection",

                "pca_refit_permitted":
                    False,

                "pc_sign_flip_permitted":
                    False,

                "pc_screening_permitted":
                    False,

                "pc6_in_this_module":
                    False,
            },

        "primary":
            {
                "scientific_question":
                    (
                        "Does frozen PC3 differ between pathologist-annotated "
                        "Gleason pattern 3 and Gleason pattern 4 within the same WSI?"
                    ),

                "contrast":
                    "G3_vs_G4",

                "clinical_role":
                    "primary clinically relevant contrast",

                "design":
                    "within-slide paired",

                "independent_unit":
                    PRIMARY_INDEPENDENT_UNIT,

                "class_summary":
                    PRIMARY_WITHIN_SLIDE_SUMMARY,

                "effect_definition":
                    "median_PC3_G4 - median_PC3_G3",

                "expected_direction":
                    PRIMARY_EXPECTED_DIRECTION,

                "inference":
                    {
                        "cohort_statistic":
                            "median within-slide paired effect",

                        "bootstrap_ci_level":
                            BOOTSTRAP_CI_LEVEL,

                        "bootstrap_replicates":
                            BOOTSTRAP_REPLICATES,

                        "bootstrap_seed":
                            BOOTSTRAP_SEED,

                        "wilcoxon":
                            {
                                "test":
                                    "slide-level paired effect vs zero",

                                "alternative":
                                    "two-sided",
                            },

                        "direction_consistency":
                            "report positive/negative/zero slide counts",

                        "pooled_tile_level_p_value":
                            False,
                    },
            },

        "train_test_roles":
            {
                "train":
                    {
                        "partition":
                            "Subset1 Train",

                        "role":
                            "formal external validation",
                    },

                "test":
                    {
                        "partition":
                            "Subset1 Test",

                        "role":
                            "separate replication",

                        "same_analysis_as_train":
                            True,

                        "test_specific_threshold_permitted":
                            False,

                        "test_specific_sign_flip_permitted":
                            False,

                        "test_specific_endpoint_change_permitted":
                            False,

                        "test_specific_slide_selection_after_outcome":
                            False,
                    },

                "train_test_pooled_primary_inference":
                    False,
            },

        "measurement_rules_for_step18b":
            {
                "annotation_source":
                    "official AGGC masks",

                "annotation_semantics":
                    "sparse positive annotations",

                "absence_of_mask_support_means_biological_negative":
                    False,

                "minimum_class_positive_pixels_per_tile":
                    MIN_CLASS_POSITIVE_PIXELS_PER_TILE,

                "minimum_class_fraction_per_tile":
                    MIN_CLASS_FRACTION_PER_TILE,

                "minimum_class_tiles_per_slide":
                    MIN_CLASS_TILES_PER_SLIDE,

                "overlap_policy":
                    (
                        "exclude overlapping support from class-specific "
                        "paired comparisons; do not silently reassign"
                    ),

                "silent_overlap_reassignment_permitted":
                    False,

                "silent_geometry_repair_permitted":
                    False,

                "silent_mask_rescaling_permitted":
                    False,
            },

        "g5_exploratory":
            {
                "status":
                    "exploratory_extension_only",

                "cannot_be_promoted_after_results":
                    True,

                "paired_contrasts":
                    [
                        {
                            "name":
                                "G5_minus_G4",

                            "effect_definition":
                                "median_PC3_G5 - median_PC3_G4",

                            "design":
                                "within-slide paired",

                            "status":
                                "exploratory",
                        },
                        {
                            "name":
                                "G5_minus_G3",

                            "effect_definition":
                                "median_PC3_G5 - median_PC3_G3",

                            "design":
                                "within-slide paired",

                            "status":
                                "exploratory",
                        },
                    ],
            },

        "tumor_only_ordinal":
            {
                "sequence":
                    [
                        "G3",
                        "G4",
                        "G5",
                    ],

                "allowed":
                    True,

                "status":
                    "prespecified_exploratory_supportive_only",

                "primary_or_confirmatory":
                    False,

                "eligible_slides":
                    (
                        "slides with technically valid G3/G4/G5 and sufficient "
                        "Step18b class-specific tile support"
                    ),

                "implementation":
                    {
                        "class_order":
                            [
                                3,
                                4,
                                5,
                            ],

                        "per_class_value":
                            "median frozen PC3",

                        "within_slide_statistic":
                            (
                                "Spearman rho between class order [3,4,5] and "
                                "the three class-level median PC3 values"
                            ),

                        "n_class_observations_per_slide":
                            3,
                    },

                "direction_preassigned":
                    False,

                "interpretation_boundary":
                    (
                        "Exploratory tumor-pattern ordering only; cannot by "
                        "itself establish a continuous clinical progression axis."
                    ),
            },

        "extended_histologic_state_profile":
            {
                "sequence":
                    [
                        "Stroma",
                        "Normal",
                        "G3",
                        "G4",
                        "G5",
                    ],

                "state_order":
                    EXTENDED_STATE_ORDER,

                "status":
                    "prespecified_exploratory_descriptive_and_supportive",

                "not_called":
                    [
                        "disease progression",
                        "Gleason ordinal progression",
                        "biological progression",
                    ],

                "why_not_progression":
                    (
                        "Stroma is a tissue compartment, Normal is benign "
                        "epithelium, and G3/G4/G5 are tumor differentiation "
                        "patterns; they are not a single clinical grading scale."
                    ),

                "analysis_A_state_profile":
                    {
                        "role":
                            "descriptive exploratory",

                        "complete_case_required":
                            False,

                        "per_state_eligibility":
                            (
                                "all slides technically valid for that state "
                                "and meeting Step18b minimum class tile support"
                            ),

                        "per_slide_state_value":
                            "median frozen PC3",

                        "cohort_outputs":
                            [
                                "n slides per state",
                                "median of slide-level state medians",
                                "bootstrap CI over slides",
                                "distribution plots",
                            ],

                        "cross_state_pooled_tile_inference":
                            False,
                    },

                "analysis_B_all5_complete_case_ordered_summary":
                    {
                        "role":
                            "exploratory supportive",

                        "complete_case_required":
                            True,

                        "required_states":
                            [
                                "Stroma",
                                "Normal",
                                "G3",
                                "G4",
                                "G5",
                            ],

                        "per_state_value":
                            "median frozen PC3",

                        "within_slide_statistic":
                            (
                                "Spearman rho between state order [0,1,2,3,4] "
                                "and the five state-level median PC3 values"
                            ),

                        "expected_direction":
                            "none_preassigned",

                        "cannot_be_promoted_after_results":
                            True,
                    },

                "interpretation_boundary":
                    (
                        "This profile describes how frozen PC3 is distributed "
                        "across pathologist-defined histologic states. It must "
                        "not be interpreted as a temporal or causal disease "
                        "progression sequence."
                    ),
            },

        "forbidden":
            [
                "PC1-PC8 screening",
                "PC6 confirmatory analysis",
                "PCA refitting on AGGC",
                "PC3 sign flipping after AGGC outcomes",
                "pooled tile-level inferential p-values",
                "using sparse mask absence as biological negative",
                "changing G3/G4 primary endpoint after Train results",
                "changing Test rules after Train results",
                "promoting G5 after results",
                "promoting G3-G4-G5 ordinal trend after results",
                "promoting extended five-state profile to confirmatory after results",
                "calling the five-state profile disease progression",
                "outcome-based slide exclusion",
            ],
    }

    write_json(
        OUT_PLAN,
        analysis_plan,
    )

    # -------------------------------------------------------------------------
    # Label specification
    # -------------------------------------------------------------------------

    label_spec = {
        "official_labels":
            EXTENDED_STATE_LABELS,

        "primary_labels":
            [
                "G3",
                "G4",
            ],

        "primary_contrast":
            "G4_minus_G3",

        "g5_status":
            "exploratory_only",

        "tumor_only_ordinal":
            [
                "G3",
                "G4",
                "G5",
            ],

        "extended_histologic_state_profile":
            [
                "Stroma",
                "Normal",
                "G3",
                "G4",
                "G5",
            ],

        "annotation_semantics":
            {
                "source":
                    "official AGGC2022 Subset-1 masks",

                "dataset_context":
                    "automatic Gleason grading",

                "positive_annotation_layer":
                    True,

                "whole_slide_exhaustive_segmentation":
                    False,

                "unannotated_pixel_interpreted_as_negative":
                    False,
            },

        "class_assignment":
            {
                "minimum_class_fraction_per_tile":
                    MIN_CLASS_FRACTION_PER_TILE,

                "minimum_class_positive_pixels_per_tile":
                    MIN_CLASS_POSITIVE_PIXELS_PER_TILE,

                "minimum_class_tiles_per_slide":
                    MIN_CLASS_TILES_PER_SLIDE,

                "overlap_rule":
                    (
                        "class-overlapping annotation support is excluded from "
                        "the affected class comparison; no silent reassignment"
                    ),
            },
    }

    write_json(
        OUT_LABEL_SPEC,
        label_spec,
    )

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    geometry_spec = {
        "source":
            "Step17 preflight technical geometry audit",

        "allowed_geometry":
            "direct native WSI/mask geometry only",

        "rescaling_permitted":
            False,

        "offset_inference_permitted":
            False,

        "manual_registration_permitted":
            False,

        "geometry_repair_permitted":
            False,

        "step18b_requirement":
            (
                "re-assert frozen WSI/mask dimensions and coords/PCA row "
                "consistency before measuring; fail rather than repair"
            ),
    }

    write_json(
        OUT_GEOM_SPEC,
        geometry_spec,
    )

    # -------------------------------------------------------------------------
    # Cohort summary
    # -------------------------------------------------------------------------

    def ntrue(
        df: pd.DataFrame,
        col: str,
    ) -> int:

        return int(
            df[col].map(to_bool).sum()
        )

    summary_rows = []

    for partition, df in [
        (
            "train_validation",
            train,
        ),
        (
            "test_replication",
            test,
        ),
    ]:

        metrics = {
            "n_total_subset1_wsis":
                len(df),

            "n_base_embedding_pca_ready":
                ntrue(
                    df,
                    "base_embedding_pca_ready",
                ),

            "n_Stroma_technically_ready":
                ntrue(
                    df,
                    "stroma_technically_ready",
                ),

            "n_Normal_technically_ready":
                ntrue(
                    df,
                    "normal_technically_ready",
                ),

            "n_G3_technically_ready":
                ntrue(
                    df,
                    "g3_technically_ready",
                ),

            "n_G4_technically_ready":
                ntrue(
                    df,
                    "g4_technically_ready",
                ),

            "n_G5_technically_ready":
                ntrue(
                    df,
                    "g5_technically_ready",
                ),

            "n_primary_G3_G4_paired_frozen":
                ntrue(
                    df,
                    "formal_18a_primary_g3_g4_include",
                ),

            "n_exploratory_G4_G5_paired_frozen":
                ntrue(
                    df,
                    "formal_18a_g4_g5_exploratory_include",
                ),

            "n_exploratory_G3_G5_paired_frozen":
                ntrue(
                    df,
                    "formal_18a_g3_g5_exploratory_include",
                ),

            "n_exploratory_G3_G4_G5_ordinal_frozen":
                ntrue(
                    df,
                    "formal_18a_g3_g4_g5_ordinal_exploratory_include",
                ),

            "n_state_Stroma_profile_frozen":
                ntrue(
                    df,
                    "formal_18a_state_stroma_profile_include",
                ),

            "n_state_Normal_profile_frozen":
                ntrue(
                    df,
                    "formal_18a_state_normal_profile_include",
                ),

            "n_state_G3_profile_frozen":
                ntrue(
                    df,
                    "formal_18a_state_g3_profile_include",
                ),

            "n_state_G4_profile_frozen":
                ntrue(
                    df,
                    "formal_18a_state_g4_profile_include",
                ),

            "n_state_G5_profile_frozen":
                ntrue(
                    df,
                    "formal_18a_state_g5_profile_include",
                ),

            "n_extended_all5_complete_case_frozen":
                ntrue(
                    df,
                    "formal_18a_extended_all5_complete_case_include",
                ),
        }

        for metric, value in metrics.items():

            summary_rows.append(
                {
                    "partition":
                        partition,

                    "metric":
                        metric,

                    "value":
                        int(value),
                }
            )

    cohort_summary = pd.DataFrame(
        summary_rows
    )

    cohort_summary.to_csv(
        OUT_COHORT_SUMMARY,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    provenance_rows = [
        {
            "evidence_domain":
                "Primary design",

            "evidence_item":
                "Clinically relevant contrast",

            "frozen_value":
                "G3 vs G4",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Primary design",

            "evidence_item":
                "Design",

            "frozen_value":
                "within-slide paired",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Primary design",

            "evidence_item":
                "Effect",

            "frozen_value":
                "median_PC3_G4 - median_PC3_G3",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Primary design",

            "evidence_item":
                "Direction",

            "frozen_value":
                "none preassigned; two-sided",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Exploratory design",

            "evidence_item":
                "G5",

            "frozen_value":
                "exploratory extension only",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Exploratory design",

            "evidence_item":
                "Tumor-only ordinal",

            "frozen_value":
                "G3 -> G4 -> G5; exploratory/supportive only",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Exploratory design",

            "evidence_item":
                "Extended histologic-state profile",

            "frozen_value":
                "Stroma -> Normal -> G3 -> G4 -> G5",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Claim boundary",

            "evidence_item":
                "Five-state profile interpretation",

            "frozen_value":
                "not disease progression / not Gleason ordinal progression",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Partition role",

            "evidence_item":
                "Train",

            "frozen_value":
                "formal external validation",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Partition role",

            "evidence_item":
                "Test",

            "frozen_value":
                "separate exact-protocol replication",

            "status":
                "FROZEN",
        },
        {
            "evidence_domain":
                "Latent representation",

            "evidence_item":
                "Target",

            "frozen_value":
                "discovery-cohort frozen PC3",

            "status":
                "FROZEN",
        },
    ]

    pd.DataFrame(
        provenance_rows
    ).to_csv(
        OUT_PROVENANCE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Source hashes
    # -------------------------------------------------------------------------

    hash_rows = []

    for p in source_files:

        hash_rows.append(
            {
                "path":
                    str(p),

                "sha256":
                    sha256_file(p),
            }
        )

    pd.DataFrame(
        hash_rows
    ).to_csv(
        OUT_HASHES,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = {
        "step":
            "18a",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),

        "role":
            "freeze_only",

        "pc_values_read":
            False,

        "pc3_label_association_calculated":
            False,

        "test_pc3_label_outcome_read":
            False,

        "pca_refit":
            False,

        "pc_screening":
            False,

        "pc6_analysis":
            False,

        "inference_performed":
            False,

        "primary_direction_learned_from_previous_AGGC_results":
            False,

        "primary_contrast":
            "G3_vs_G4",

        "primary_design":
            "within_slide_paired",

        "primary_effect":
            "median_PC3_G4_minus_median_PC3_G3",

        "primary_inference_two_sided":
            True,

        "G5_status":
            "exploratory_only",

        "tumor_only_ordinal_status":
            "exploratory_supportive_only",

        "extended_histologic_state_profile_status":
            "exploratory_descriptive_and_supportive",

        "extended_histologic_state_sequence":
            EXTENDED_STATE_LABELS,

        "extended_histologic_state_called_progression":
            False,

        "train_role":
            "formal_external_validation",

        "test_role":
            "separate_replication",

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,
    }

    write_json(
        OUT_METADATA,
        metadata,
    )

    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------

    readme = f"""
STEP 18A — AGGC PC3 GLEASON / HISTOLOGIC-STATE VALIDATION SPECIFICATION
=======================================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA_VERSION}


STATUS
------
FROZEN BEFORE FORMAL PC3–LABEL OUTCOME ANALYSIS.


PRIMARY
=======

Question:
    Does frozen PC3 differ between pathologist-annotated Gleason pattern 3
    and Gleason pattern 4 within the same WSI?

Contrast:
    G3 vs G4

Effect:
    median_PC3_G4 - median_PC3_G3

Design:
    within-slide paired

Independent unit:
    slide

Direction:
    none preassigned

Inference:
    two-sided


WHY G3 VS G4 IS PRIMARY
-----------------------
The G3-to-G4 distinction is directly relevant to prostate-cancer grading and
is more clinically interpretable than an undirected multi-class PC screen.


TRAIN / TEST
============
Train:
    formal external validation

Test:
    separate replication

Exactly the same frozen protocol must be used in Test.

No Test-specific tuning is permitted.


G5 EXPLORATORY
==============
G5 is exploratory only.

Prespecified paired contrasts:
    G5 - G4
    G5 - G3

G5 cannot be promoted after outcome inspection.


TUMOR-ONLY ORDINAL
==================
Sequence:
    G3 -> G4 -> G5

Status:
    exploratory/supportive only

For eligible all-three slides:
    compute median PC3 in G3/G4/G5
    then Spearman rho between [3,4,5] and those three medians

No direction is prespecified.


EXTENDED HISTOLOGIC-STATE PROFILE
=================================
Sequence:
    Stroma -> Normal -> G3 -> G4 -> G5

This is deliberately NOT called:
    disease progression
    Gleason ordinal progression
    biological progression

Reason:
    Stroma is a tissue compartment,
    Normal is benign epithelium,
    and G3/G4/G5 are tumor differentiation patterns.

Two prespecified exploratory outputs are allowed.

A. STATE PROFILE
----------------
For each state separately:
    Stroma
    Normal
    G3
    G4
    G5

use all technically eligible slides for that state.

Per-slide value:
    median frozen PC3

Complete-case all-five support is NOT required for this descriptive profile.

B. ALL-FIVE COMPLETE-CASE ORDERED SUMMARY
-----------------------------------------
Only slides with all five technically valid states may enter.

Per slide:
    median PC3(Stroma)
    median PC3(Normal)
    median PC3(G3)
    median PC3(G4)
    median PC3(G5)

Then compute exploratory Spearman rho between:

    [0,1,2,3,4]

and the five state-level medians.

No expected monotonic direction is prespecified.

This analysis cannot be promoted to a primary conclusion.


ANNOTATION SEMANTICS
====================
AGGC annotations are sparse positive labels developed primarily for automatic
Gleason grading.

Explicit annotation support:
    means that state/class was annotated.

Absence of annotation support:
    does NOT mean biological negative.

Unannotated tissue is not assigned a class.


FROZEN TECHNICAL RULES FOR STEP18B
==================================
Minimum class-positive pixels per tile:
    {MIN_CLASS_POSITIVE_PIXELS_PER_TILE}

Minimum class fraction per tile:
    {MIN_CLASS_FRACTION_PER_TILE:.2f}

Minimum usable tiles per class per slide:
    {MIN_CLASS_TILES_PER_SLIDE}

Overlapping annotation support:
    excluded from affected class-specific comparisons

No silent reassignment.
No silent geometry repair.
No silent rescaling.


PRIMARY INFERENCE FOR STEP18C
=============================
Train and Test separately:

    Delta_slide
        = median_PC3_G4 - median_PC3_G3

    cohort effect
        = median Delta_slide

    uncertainty
        {int(BOOTSTRAP_CI_LEVEL * 100)}% slide-bootstrap CI
        {BOOTSTRAP_REPLICATES} replicates

    formal test
        two-sided Wilcoxon signed-rank vs zero

No pooled tile-level inferential p-value.


FORBIDDEN
=========
No PC1-PC8 screening.
No PC6 confirmatory analysis.
No PCA refit on AGGC.
No PC3 sign flip after outcomes.
No endpoint switching after Train.
No Test-specific tuning.
No outcome-based slide exclusion.
No promotion of G5 after results.
No promotion of tumor-only ordinal trend after results.
No promotion of the five-state profile after results.
Do not call the five-state profile disease progression.


OUTPUTS
=======
Primary Train cohort:
    {OUT_TRAIN_PRIMARY}

Primary Test cohort:
    {OUT_TEST_PRIMARY}

G5 exploratory Train cohort:
    {OUT_TRAIN_G5}

G5 exploratory Test cohort:
    {OUT_TEST_G5}

Extended-state Train cohort:
    {OUT_TRAIN_EXTENDED}

Extended-state Test cohort:
    {OUT_TEST_EXTENDED}

Full frozen partition manifest:
    {OUT_PARTITION_MANIFEST}

Analysis plan:
    {OUT_PLAN}

Label specification:
    {OUT_LABEL_SPEC}

Geometry specification:
    {OUT_GEOM_SPEC}

Cohort summary:
    {OUT_COHORT_SUMMARY}
""".strip() + "\n"

    OUT_README.write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 124)
    print("FROZEN SCIENTIFIC DESIGN")
    print("=" * 124)

    print("PRIMARY")
    print("  Contrast                         : G3 vs G4")
    print("  Design                           : within-slide paired")
    print("  Effect                           : median PC3(G4) - median PC3(G3)")
    print("  Direction                        : NONE preassigned; two-sided")
    print("  Independent unit                 : slide")

    print()
    print("EXPLORATORY")
    print("  G5                               : G5-G4 and G5-G3")
    print("  Tumor-only ordinal               : G3 -> G4 -> G5")
    print("  Extended histologic-state profile: Stroma -> Normal -> G3 -> G4 -> G5")
    print("  Five-state called progression    : NO")

    print()
    print("PARTITIONS")
    print("  Train                            : formal external validation")
    print("  Test                             : separate replication")

    print()
    print("=" * 124)
    print("FROZEN TECHNICAL COHORTS")
    print("=" * 124)

    for name, df in [
        (
            "Train",
            train,
        ),
        (
            "Test",
            test,
        ),
    ]:

        print()
        print(f"[{name}]")
        print(f"Total Subset1 WSIs                 : {len(df)}")
        print(
            "Primary G3/G4 paired              : "
            f"{int(df['formal_18a_primary_g3_g4_include'].sum())}"
        )
        print(
            "Exploratory G4/G5 paired          : "
            f"{int(df['formal_18a_g4_g5_exploratory_include'].sum())}"
        )
        print(
            "Exploratory G3/G5 paired          : "
            f"{int(df['formal_18a_g3_g5_exploratory_include'].sum())}"
        )
        print(
            "Tumor-only all-3 ordinal          : "
            f"{int(df['formal_18a_g3_g4_g5_ordinal_exploratory_include'].sum())}"
        )
        print(
            "State Stroma profile              : "
            f"{int(df['formal_18a_state_stroma_profile_include'].sum())}"
        )
        print(
            "State Normal profile              : "
            f"{int(df['formal_18a_state_normal_profile_include'].sum())}"
        )
        print(
            "State G3 profile                  : "
            f"{int(df['formal_18a_state_g3_profile_include'].sum())}"
        )
        print(
            "State G4 profile                  : "
            f"{int(df['formal_18a_state_g4_profile_include'].sum())}"
        )
        print(
            "State G5 profile                  : "
            f"{int(df['formal_18a_state_g5_profile_include'].sum())}"
        )
        print(
            "Extended all-5 complete case      : "
            f"{int(df['formal_18a_extended_all5_complete_case_include'].sum())}"
        )

    print()
    print("=" * 124)
    print("FREEZE BOUNDARY AUDIT")
    print("=" * 124)

    print("PC values read                       : NO")
    print("PC3-label association calculated     : NO")
    print("Test outcome read                    : NO")
    print("Inference performed                  : NO")
    print("PCA refit                            : NO")
    print("PC screening                         : NO")
    print("PC6 analysis                         : NO")
    print("Primary direction learned from AGGC  : NO")
    print("Five-state progression claim frozen  : NO")

    print()
    print("[OK] Saved:")

    for p in [
        OUT_TRAIN_PRIMARY,
        OUT_TEST_PRIMARY,
        OUT_TRAIN_G5,
        OUT_TEST_G5,
        OUT_TRAIN_EXTENDED,
        OUT_TEST_EXTENDED,
        OUT_PARTITION_MANIFEST,
        OUT_PLAN,
        OUT_LABEL_SPEC,
        OUT_GEOM_SPEC,
        OUT_COHORT_SUMMARY,
        OUT_PROVENANCE,
        OUT_HASHES,
        OUT_METADATA,
        OUT_README,
    ]:
        print(
            " ",
            p,
        )

    print()
    print(
        "[DONE] Step-18a v1.1.0 AGGC PC3 Gleason / histologic-state specification frozen."
    )

    print()
    print(
        "[NEXT] Step 18b = measurement only; do not calculate G3/G4 effects yet."
    )


if __name__ == "__main__":
    main()
