#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16b_compute_peso_pc3_external_compartment_measurements.py

Step 16b v1.1.0
================

Measurement-only implementation of the frozen Step-16a PESO PC3
external compartment specification.

ABSOLUTE RULE
-------------
Step 16b is NOT allowed to redefine:

- cohort membership
- PC index
- PCA space
- mask source
- corrected-mask role
- mask scale
- mask labels
- patch size
- labeled-pixel denominator
- minimum labeled fraction
- independent statistical unit
- expected biological direction
- inferential procedure

All scientific quantities are read from frozen Step 16a outputs.

Missing, inconsistent, modified, or conflicting frozen information
causes an immediate abort.

FROZEN PRIMARY DESIGN
---------------------
Primary:
    62 / 62 PESO standard training masks.

Sensitivity:
    corrected masks on the frozen 25-slide corrected subset.

Hybrid:
    prohibited / not performed.

STEP 16B PERFORMS ONLY
----------------------
- Step-16a provenance/hash validation
- frozen design validation
- exact H&E-coordinate -> mask-coordinate mapping
- categorical mask pixel counting
- frozen labeled-fraction QC
- per-slide measurement NPZ generation
- slide-level measurement coverage QC
- failure audit
- standard-vs-corrected measurement agreement QC

STEP 16B DOES NOT PERFORM
-------------------------
- Spearman correlation with PC3
- Pearson correlation with PC3
- PC3 low/high comparison
- effect-direction testing
- p-values
- confidence intervals
- bootstrap
- FDR
- feature selection
- morphology interpretation
- hybrid mask analysis

Formal inference is reserved for Step 16c.
"""

from __future__ import annotations

import argparse
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

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# =============================================================================
# GLOBAL PIL CONFIG
# =============================================================================

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.1.0"

SCHEMA_VERSION = (
    "16b-peso-pc3-measurement-schema-1.1"
)


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

STEP16A_DIR = (
    RESULTS_ROOT
    / "16a_peso_pc3_external_compartment_specification"
)

STEP16A_MANIFEST = (
    STEP16A_DIR
    / "peso_pc3_external_mask_manifest.csv"
)

STEP16A_PLAN = (
    STEP16A_DIR
    / "peso_pc3_external_analysis_plan.json"
)

STEP16A_LABELS = (
    STEP16A_DIR
    / "peso_pc3_external_label_specification.json"
)

STEP16A_GEOMETRY = (
    STEP16A_DIR
    / "peso_pc3_external_geometry_audit.csv"
)

STEP16A_METADATA = (
    STEP16A_DIR
    / "analysis_metadata.json"
)


OUT_DIR = (
    RESULTS_ROOT
    / "16b_peso_pc3_external_compartment_measurements"
)

PER_SLIDE_DIR = (
    OUT_DIR
    / "per_slide"
)

OUT_MANIFEST = (
    OUT_DIR
    / "peso_pc3_external_measurement_manifest.csv"
)

OUT_QC = (
    OUT_DIR
    / "peso_pc3_external_slide_qc_summary.csv"
)

OUT_AGREEMENT = (
    OUT_DIR
    / "peso_pc3_external_corrected_agreement_qc.csv"
)

OUT_FAILURES = (
    OUT_DIR
    / "peso_pc3_external_measurement_failures.csv"
)

OUT_COVERAGE = (
    OUT_DIR
    / "peso_pc3_external_feature_coverage.csv"
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
    OUT_MANIFEST,
    OUT_QC,
    OUT_AGREEMENT,
    OUT_FAILURES,
    OUT_COVERAGE,
    OUT_METADATA,
    OUT_README,
]


# =============================================================================
# EXPECTED FROZEN DESIGN
# =============================================================================

EXPECTED_STEP16A_SCHEMA = (
    "16a-peso-pc3-external-schema-2.0"
)

EXPECTED_PRIMARY_SLIDES = 62

EXPECTED_CORRECTED_SLIDES = 25

EXPECTED_PC_NAME = "PC3"

EXPECTED_PRIMARY_MASK_SOURCE = "standard"

EXPECTED_SENSITIVITY_MASK_SOURCE = "corrected"


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing Step-16b output directory. "
            "Existing per-slide NPZ files are removed first."
        ),
    )

    parser.add_argument(
        "--max-slides",
        type=int,
        default=None,
        help=(
            "Engineering/debug only. "
            "Limits the number of frozen manifest rows processed. "
            "Any run using this flag is marked formal_full_run=False."
        ),
    )

    return parser.parse_args()


# =============================================================================
# BASIC HELPERS
# =============================================================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def require_file(
    path: Path,
    label: str,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required {label}:\n"
            f"{path}"
        )


def sha256_file(
    path: Path,
) -> Optional[str]:

    if (
        not path.exists()
        or not path.is_file()
    ):
        return None

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def write_json(
    path: Path,
    obj,
):

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


def clean_path(
    value,
) -> Optional[Path]:

    if value is None:
        return None

    try:

        if pd.isna(
            value
        ):
            return None

    except Exception:
        pass

    text = str(
        value
    ).strip()

    if not text:
        return None

    return Path(
        text
    )


def as_bool(
    value,
) -> bool:

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return bool(
            value
        )

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "",
        "nan",
        "none",
    }:
        return False

    raise ValueError(
        "Cannot interpret boolean: "
        f"{value!r}"
    )


# =============================================================================
# OUTPUT PROTECTION
# =============================================================================

def prepare_output_directory(
    overwrite: bool,
):

    existing = [
        path
        for path in CORE_OUTPUTS
        if path.exists()
    ]

    existing_npz = []

    if PER_SLIDE_DIR.exists():

        existing_npz = list(
            PER_SLIDE_DIR.glob(
                "*.npz"
            )
        )

    if (
        (
            existing
            or existing_npz
        )
        and
        not overwrite
    ):

        details = [
            str(path)
            for path in existing
        ]

        if existing_npz:

            details.append(
                f"{PER_SLIDE_DIR} "
                f"({len(existing_npz)} existing NPZ files)"
            )

        raise RuntimeError(
            "Step-16b outputs already exist; "
            "refusing silent overwrite.\n"
            +
            "\n".join(
                f"  - {x}"
                for x in details
            )
            +
            "\n\nUse --overwrite only if replacement "
            "is intentional."
        )

    if (
        overwrite
        and
        OUT_DIR.exists()
    ):

        print(
            "[OVERWRITE] Removing existing "
            "Step-16b output directory:"
        )

        print(
            f"            {OUT_DIR}"
        )

        shutil.rmtree(
            OUT_DIR
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PER_SLIDE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# FROZEN-VALUE EXTRACTION
# =============================================================================

def first_existing_column(
    df: pd.DataFrame,
    names: Iterable[str],
    label: str,
) -> str:

    found = [
        name
        for name in names
        if name in df.columns
    ]

    if not found:

        raise RuntimeError(
            "Step-16a manifest missing "
            f"{label}; expected one of:\n"
            f"{list(names)}"
        )

    return found[0]


def nested_get(
    obj: Dict[str, Any],
    paths: Iterable[Tuple[str, ...]],
    label: str,
):

    found = []

    for path in paths:

        cur = obj

        ok = True

        for key in path:

            if (
                not isinstance(
                    cur,
                    dict,
                )
                or key not in cur
            ):

                ok = False
                break

            cur = cur[
                key
            ]

        if ok:

            found.append(
                (
                    path,
                    cur,
                )
            )

    if not found:

        raise RuntimeError(
            "Step-16a did not freeze "
            f"{label}; tried paths:\n"
            f"{list(paths)}"
        )

    serialized = [
        json.dumps(
            value,
            sort_keys=True,
        )
        for _, value in found
    ]

    if len(
        set(
            serialized
        )
    ) > 1:

        raise RuntimeError(
            "Conflicting Step-16a "
            f"values for {label}:\n"
            f"{found}"
        )

    return found[0][1]


# =============================================================================
# STEP-16A HASH CHAIN
# =============================================================================

def validate_hash_chain(
    meta16a: Dict[str, Any],
):

    recorded = (
        meta16a.get(
            "output_sha256",
            {},
        )
    )

    mapping = {
        "manifest":
            STEP16A_MANIFEST,

        "analysis_plan":
            STEP16A_PLAN,

        "label_specification":
            STEP16A_LABELS,

        "geometry_audit":
            STEP16A_GEOMETRY,
    }

    aliases = {
        "manifest": [
            "mask_manifest",
        ],

        "analysis_plan": [
            "plan",
        ],

        "label_specification": [
            "labels",
            "label_spec",
        ],

        "geometry_audit": [
            "geometry",
        ],
    }

    out = {}

    for key, path in (
        mapping.items()
    ):

        recorded_hash = (
            recorded.get(
                key
            )
        )

        if recorded_hash is None:

            for alias in (
                aliases.get(
                    key,
                    [],
                )
            ):

                if alias in recorded:

                    recorded_hash = (
                        recorded[
                            alias
                        ]
                    )

                    break

        if recorded_hash is None:

            raise RuntimeError(
                "Step-16a metadata lacks "
                f"frozen SHA256 for {key}."
            )

        current_hash = (
            sha256_file(
                path
            )
        )

        match = (
            str(
                recorded_hash
            )
            ==
            str(
                current_hash
            )
        )

        out[
            key
        ] = {
            "recorded":
                recorded_hash,

            "current":
                current_hash,

            "match":
                match,
        }

        if not match:

            raise RuntimeError(
                "Step-16a hash mismatch "
                f"for {key}:\n"
                f"recorded={recorded_hash}\n"
                f"current ={current_hash}"
            )

    return out


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

def extract_labels(
    labels16a: Dict[str, Any],
):

    """
    Compatible with frozen Step-16a schema 2.0.

    Primary expected source:

        formal_label_mapping:
            "0": ...
            "1": ...
            "2": ...

    Legacy aliases are retained only for backward-readable diagnostics.
    """

    epi = int(
        nested_get(
            labels16a,
            [
                (
                    "formal_label_mapping",
                    "2",
                    "pixel_value",
                ),
                (
                    "mask_semantics",
                    "epithelium",
                    "pixel_value",
                ),
                (
                    "labels",
                    "epithelium",
                ),
                (
                    "pixel_values",
                    "epithelium",
                ),
            ],
            "epithelium label",
        )
    )

    background = int(
        nested_get(
            labels16a,
            [
                (
                    "formal_label_mapping",
                    "1",
                    "pixel_value",
                ),
                (
                    "mask_semantics",
                    "background_or_non_epithelial",
                    "pixel_value",
                ),
                (
                    "mask_semantics",
                    "background",
                    "pixel_value",
                ),
                (
                    "labels",
                    "background",
                ),
                (
                    "pixel_values",
                    "background",
                ),
            ],
            "background/non-epithelial label",
        )
    )

    unlabeled = int(
        nested_get(
            labels16a,
            [
                (
                    "formal_label_mapping",
                    "0",
                    "pixel_value",
                ),
                (
                    "mask_semantics",
                    "unlabeled",
                    "pixel_value",
                ),
                (
                    "labels",
                    "unlabeled",
                ),
                (
                    "pixel_values",
                    "unlabeled",
                ),
            ],
            "unlabeled label",
        )
    )

    if len(
        {
            epi,
            background,
            unlabeled,
        }
    ) != 3:

        raise RuntimeError(
            "Frozen mask labels "
            "are not distinct."
        )

    return {
        "epithelium_label":
            epi,

        "background_label":
            background,

        "unlabeled_label":
            unlabeled,
    }


# =============================================================================
# ANALYSIS-PLAN EXTRACTION
# =============================================================================

def extract_plan(
    plan16a: Dict[str, Any],
):

    pc_idx = int(
        nested_get(
            plan16a,
            [
                (
                    "latent_representation",
                    "target_axis_zero_based_index",
                ),
                (
                    "target_axis_zero_based_index",
                ),
                (
                    "pc3_zero_based_index",
                ),
            ],
            "PC3 zero-based index",
        )
    )

    min_labeled = float(
        nested_get(
            plan16a,
            [
                (
                    "measurement_qc",
                    "formal_minimum_labeled_fraction",
                ),
                (
                    "measurement_qc",
                    "minimum_labeled_fraction",
                ),
                (
                    "qc",
                    "minimum_labeled_fraction",
                ),
            ],
            "minimum labeled fraction",
        )
    )

    independent_unit = str(
        nested_get(
            plan16a,
            [
                (
                    "aggregation_and_inference",
                    "independent_unit",
                ),
                (
                    "formal_independent_unit",
                ),
            ],
            "independent unit",
        )
    ).strip().lower()

    if independent_unit != "slide":

        raise RuntimeError(
            "Frozen independent unit "
            "must be slide; found "
            f"{independent_unit!r}"
        )

    if not (
        0
        <= min_labeled
        <= 1
    ):

        raise RuntimeError(
            "Frozen minimum labeled "
            "fraction is outside [0,1]."
        )

    if pc_idx < 0:

        raise RuntimeError(
            "Frozen PC index cannot "
            "be negative."
        )

    return {
        "pc_index_zero_based":
            pc_idx,

        "minimum_labeled_fraction":
            min_labeled,

        "independent_unit":
            independent_unit,
    }


# =============================================================================
# MANIFEST COLUMN RESOLUTION
# =============================================================================

def resolve_columns(
    df: pd.DataFrame,
):

    return {
        "slide_id":
            first_existing_column(
                df,
                [
                    "slide_id",
                ],
                "slide_id",
            ),

        "coords_path":
            first_existing_column(
                df,
                [
                    "coords_path",
                    "coord_path",
                ],
                "coords path",
            ),

        "pca_path":
            first_existing_column(
                df,
                [
                    "pca_path",
                    "pca_targets_path",
                ],
                "PCA path",
            ),

        "patch_size":
            first_existing_column(
                df,
                [
                    "patch_size",
                ],
                "patch size",
            ),

        "standard_mask_path":
            first_existing_column(
                df,
                [
                    "standard_mask_path",
                    "primary_standard_mask_path",
                    "primary_mask_path",
                ],
                "standard mask path",
            ),

        "standard_scale_x":
            first_existing_column(
                df,
                [
                    "standard_scale_x",
                    "primary_standard_scale_x",
                    "primary_scale_x",
                ],
                "standard scale_x",
            ),

        "standard_scale_y":
            first_existing_column(
                df,
                [
                    "standard_scale_y",
                    "primary_standard_scale_y",
                    "primary_scale_y",
                ],
                "standard scale_y",
            ),

        "corrected_available":
            first_existing_column(
                df,
                [
                    "corrected_mask_available",
                    "has_corrected_mask",
                ],
                "corrected availability",
            ),

        "corrected_mask_path":
            first_existing_column(
                df,
                [
                    "corrected_mask_path",
                    "sensitivity_corrected_mask_path",
                    "sensitivity_mask_path",
                ],
                "corrected mask path",
            ),

        "corrected_scale_x":
            first_existing_column(
                df,
                [
                    "corrected_scale_x",
                    "sensitivity_corrected_scale_x",
                    "sensitivity_scale_x",
                ],
                "corrected scale_x",
            ),

        "corrected_scale_y":
            first_existing_column(
                df,
                [
                    "corrected_scale_y",
                    "sensitivity_corrected_scale_y",
                    "sensitivity_scale_y",
                ],
                "corrected scale_y",
            ),
    }


# =============================================================================
# FROZEN DESIGN VALIDATION
# =============================================================================

def validate_design(
    df: pd.DataFrame,
    columns: Dict[str, str],
    plan16a: Dict[str, Any],
    meta16a: Dict[str, Any],
):

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    schema16a = str(
        meta16a.get(
            "schema_version",
            "",
        )
    )

    if schema16a != EXPECTED_STEP16A_SCHEMA:

        raise RuntimeError(
            "Step-16b expects frozen "
            f"Step-16a schema:\n"
            f"  {EXPECTED_STEP16A_SCHEMA}\n"
            f"but found:\n"
            f"  {schema16a}"
        )


    # -------------------------------------------------------------------------
    # Cohort
    # -------------------------------------------------------------------------

    n_rows = int(
        len(
            df
        )
    )

    n_unique = int(
        df[
            columns[
                "slide_id"
            ]
        ].nunique()
    )

    if (
        n_rows
        != EXPECTED_PRIMARY_SLIDES
        or
        n_unique
        != EXPECTED_PRIMARY_SLIDES
    ):

        raise RuntimeError(
            "Frozen Step-16a manifest "
            "must contain exactly "
            f"{EXPECTED_PRIMARY_SLIDES} "
            "unique slides.\n"
            f"rows={n_rows}, "
            f"unique={n_unique}"
        )

    if (
        df[
            columns[
                "slide_id"
            ]
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Duplicate slide_id in "
            "frozen Step-16a manifest."
        )


    # -------------------------------------------------------------------------
    # Corrected subset
    # -------------------------------------------------------------------------

    n_corrected = int(
        df[
            columns[
                "corrected_available"
            ]
        ]
        .map(
            as_bool
        )
        .sum()
    )

    if (
        n_corrected
        != EXPECTED_CORRECTED_SLIDES
    ):

        raise RuntimeError(
            "Frozen corrected subset "
            "must contain exactly "
            f"{EXPECTED_CORRECTED_SLIDES} "
            f"slides; found {n_corrected}."
        )


    # -------------------------------------------------------------------------
    # Primary mask role
    # -------------------------------------------------------------------------

    for col in [
        "primary_mask_type",
        "primary_analysis_mask_type",
    ]:

        if col in df.columns:

            values = set(
                df[
                    col
                ]
                .dropna()
                .astype(
                    str
                )
                .str.strip()
                .str.lower()
                .unique()
            )

            if values != {
                EXPECTED_PRIMARY_MASK_SOURCE
            }:

                raise RuntimeError(
                    "Step-16a primary "
                    "mask role is not "
                    "standard-only:\n"
                    f"{values}"
                )


    # -------------------------------------------------------------------------
    # Plan role consistency
    # -------------------------------------------------------------------------

    primary_source = (
        plan16a
        .get(
            "primary_mask_source",
            {},
        )
    )

    if not isinstance(
        primary_source,
        dict,
    ):

        raise RuntimeError(
            "Step-16a primary_mask_source "
            "is malformed."
        )

    primary_text = (
        json.dumps(
            primary_source,
            sort_keys=True,
        )
        .lower()
    )

    if "standard" not in primary_text:

        raise RuntimeError(
            "Step-16a primary analysis "
            "does not explicitly freeze "
            "standard masks."
        )


    corrected_section = (
        plan16a
        .get(
            "corrected_mask_sensitivity",
            {},
        )
    )

    corrected_text = (
        json.dumps(
            corrected_section,
            sort_keys=True,
        )
        .lower()
    )

    if (
        "corrected"
        not in corrected_text
        or
        "sensitivity"
        not in corrected_text
    ):

        raise RuntimeError(
            "Step-16a does not explicitly "
            "freeze corrected masks as "
            "sensitivity-only."
        )


    if corrected_section.get(
        "no_hybrid_primary_masking"
    ) is not True:

        raise RuntimeError(
            "Step-16a does not explicitly "
            "prohibit hybrid primary masking."
        )


# =============================================================================
# MASK IO
# =============================================================================

def open_mask(
    path: Path,
    allowed_labels: set,
) -> Image.Image:

    require_file(
        path,
        "frozen categorical mask",
    )

    img = Image.open(
        path
    )

    if img.mode in {
        "L",
        "I",
        "I;16",
    }:

        return img


    # -------------------------------------------------------------------------
    # Unexpected multi-channel mask.
    #
    # Only accept if all channels are identical.
    # -------------------------------------------------------------------------

    arr = np.asarray(
        img
    )

    if arr.ndim != 3:

        img.close()

        raise RuntimeError(
            "Unsupported mask mode "
            f"{img.mode}: "
            f"{path}"
        )

    first = arr[
        ...,
        0
    ]

    for channel in range(
        1,
        arr.shape[-1],
    ):

        if not np.array_equal(
            first,
            arr[
                ...,
                channel
            ],
        ):

            img.close()

            raise RuntimeError(
                "Mask channels are "
                "not identical:\n"
                f"{path}"
            )

    unique = set(
        np.unique(
            first
        )
        .astype(
            int
        )
        .tolist()
    )

    if not unique.issubset(
        allowed_labels
    ):

        img.close()

        raise RuntimeError(
            "Unexpected mask labels "
            f"{sorted(unique)}:\n"
            f"{path}"
        )

    out = Image.fromarray(
        first
    )

    img.close()

    return out


# =============================================================================
# GEOMETRY
# =============================================================================

def crop_mask_rect(
    img: Image.Image,
    x: int,
    y: int,
    patch_size: int,
    scale_x: float,
    scale_y: float,
) -> Image.Image:

    """
    Frozen Step-16a mapping:

        x0 = round(x * scale_x)
        y0 = round(y * scale_y)

        x1 = round((x + patch_size) * scale_x)
        y1 = round((y + patch_size) * scale_y)

    No interpolation is performed.
    """

    x0 = int(
        round(
            x
            * scale_x
        )
    )

    y0 = int(
        round(
            y
            * scale_y
        )
    )

    x1 = int(
        round(
            (
                x
                + patch_size
            )
            * scale_x
        )
    )

    y1 = int(
        round(
            (
                y
                + patch_size
            )
            * scale_y
        )
    )

    if (
        x1 <= x0
        or
        y1 <= y0
    ):

        raise ValueError(
            "Degenerate mapped "
            "mask rectangle: "
            f"{(x0, y0, x1, y1)}"
        )

    width, height = (
        img.size
    )

    if (
        x0 < 0
        or
        y0 < 0
        or
        x1 > width
        or
        y1 > height
    ):

        raise ValueError(
            "Mapped rectangle exceeds "
            "mask bounds: "
            f"rect={(x0, y0, x1, y1)}, "
            f"mask={(width, height)}"
        )

    return img.crop(
        (
            x0,
            y0,
            x1,
            y1,
        )
    )


# =============================================================================
# CATEGORICAL MEASUREMENT
# =============================================================================

def measure_patch(
    patch: Image.Image,
    labels: Dict[str, int],
):

    arr = np.asarray(
        patch
    )

    if arr.ndim == 3:

        first = arr[
            ...,
            0
        ]

        for channel in range(
            1,
            arr.shape[-1],
        ):

            if not np.array_equal(
                first,
                arr[
                    ...,
                    channel
                ],
            ):

                raise RuntimeError(
                    "Non-identical channels "
                    "in categorical mask crop."
                )

        arr = first


    epi_label = (
        labels[
            "epithelium_label"
        ]
    )

    background_label = (
        labels[
            "background_label"
        ]
    )

    unlabeled_label = (
        labels[
            "unlabeled_label"
        ]
    )


    unique = set(
        np.unique(
            arr
        )
        .astype(
            int
        )
        .tolist()
    )

    allowed = {
        epi_label,
        background_label,
        unlabeled_label,
    }

    if not unique.issubset(
        allowed
    ):

        raise RuntimeError(
            "Unexpected categorical "
            "mask crop labels: "
            f"{sorted(unique)}"
        )


    n_total = int(
        arr.size
    )

    n_epi = int(
        np.sum(
            arr
            == epi_label
        )
    )

    n_background = int(
        np.sum(
            arr
            == background_label
        )
    )

    n_unlabeled = int(
        np.sum(
            arr
            == unlabeled_label
        )
    )

    n_labeled = (
        n_epi
        + n_background
    )


    if (
        n_epi
        + n_background
        + n_unlabeled
        != n_total
    ):

        raise RuntimeError(
            "Mask pixel counts "
            "do not sum to total "
            "crop pixels."
        )


    labeled_fraction = (
        n_labeled
        / n_total
        if n_total
        else np.nan
    )


    epithelial_fraction = (
        n_epi
        / n_labeled
        if n_labeled
        else np.nan
    )


    non_epithelial_fraction = (
        n_background
        / n_labeled
        if n_labeled
        else np.nan
    )


    return {
        "n_total":
            n_total,

        "n_labeled":
            n_labeled,

        "n_epi":
            n_epi,

        "n_bg":
            n_background,

        "n_unl":
            n_unlabeled,

        "labeled_fraction":
            labeled_fraction,

        "epithelial_fraction":
            epithelial_fraction,

        "non_epithelial_fraction":
            non_epithelial_fraction,
    }


# =============================================================================
# SLIDE ARRAY LOADING
# =============================================================================

def load_slide_arrays(
    row: pd.Series,
    columns: Dict[str, str],
    pc_idx: int,
):

    slide_id = str(
        row[
            columns[
                "slide_id"
            ]
        ]
    )


    coords_path = clean_path(
        row[
            columns[
                "coords_path"
            ]
        ]
    )

    pca_path = clean_path(
        row[
            columns[
                "pca_path"
            ]
        ]
    )


    if (
        coords_path is None
        or
        pca_path is None
    ):

        raise RuntimeError(
            f"{slide_id}: frozen "
            "coords/PCA path is empty."
        )


    require_file(
        coords_path,
        f"{slide_id} coords",
    )

    require_file(
        pca_path,
        f"{slide_id} PCA",
    )


    coords = np.load(
        coords_path,
        mmap_mode="r",
    )

    pca = np.load(
        pca_path,
        mmap_mode="r",
    )


    if (
        coords.ndim != 2
        or
        coords.shape[1] < 2
    ):

        raise RuntimeError(
            f"{slide_id}: invalid "
            f"coords shape "
            f"{coords.shape}"
        )


    if (
        pca.ndim != 2
        or
        pca.shape[0]
        != coords.shape[0]
    ):

        raise RuntimeError(
            f"{slide_id}: invalid "
            "PCA/coords relation "
            f"pca={pca.shape}, "
            f"coords={coords.shape}"
        )


    if pc_idx >= pca.shape[1]:

        raise RuntimeError(
            f"{slide_id}: PC index "
            f"{pc_idx} unavailable "
            f"in PCA array "
            f"{pca.shape}"
        )


    coords_xy = np.asarray(
        coords[
            :,
            :2,
        ]
    )


    pc3 = np.asarray(
        pca[
            :,
            pc_idx,
        ],
        dtype=np.float32,
    )


    return (
        slide_id,
        coords_xy,
        pc3,
    )


# =============================================================================
# SOURCE MEASUREMENT
# =============================================================================

def measure_source(
    slide_id: str,
    source_name: str,
    mask_path: Path,
    scale_x: float,
    scale_y: float,
    coords: np.ndarray,
    pc3: np.ndarray,
    patch_size: int,
    labels: Dict[str, int],
    min_labeled: float,
):

    if not np.isfinite(
        scale_x
    ):

        raise RuntimeError(
            f"{slide_id}: non-finite "
            f"{source_name} scale_x."
        )

    if not np.isfinite(
        scale_y
    ):

        raise RuntimeError(
            f"{slide_id}: non-finite "
            f"{source_name} scale_y."
        )

    if (
        scale_x <= 0
        or
        scale_y <= 0
    ):

        raise RuntimeError(
            f"{slide_id}: invalid "
            f"{source_name} scales "
            f"({scale_x}, {scale_y})."
        )


    img = open_mask(
        mask_path,
        set(
            labels.values()
        ),
    )


    n_tiles = int(
        len(
            pc3
        )
    )


    tile_idx = np.arange(
        n_tiles,
        dtype=np.int32,
    )


    x = (
        coords[
            :,
            0
        ]
        .astype(
            np.int64,
            copy=False,
        )
    )

    y = (
        coords[
            :,
            1
        ]
        .astype(
            np.int64,
            copy=False,
        )
    )


    finite_pc3 = np.isfinite(
        pc3
    )


    geometry_ok = np.zeros(
        n_tiles,
        dtype=bool,
    )

    measurement_ok = np.zeros(
        n_tiles,
        dtype=bool,
    )

    qc_pass = np.zeros(
        n_tiles,
        dtype=bool,
    )


    labeled_fraction = np.full(
        n_tiles,
        np.nan,
        dtype=np.float32,
    )

    epithelial_fraction = np.full(
        n_tiles,
        np.nan,
        dtype=np.float32,
    )

    non_epithelial_fraction = np.full(
        n_tiles,
        np.nan,
        dtype=np.float32,
    )


    n_labeled_pixels = np.zeros(
        n_tiles,
        dtype=np.int32,
    )

    n_epithelial_pixels = np.zeros(
        n_tiles,
        dtype=np.int32,
    )

    n_background_pixels = np.zeros(
        n_tiles,
        dtype=np.int32,
    )

    n_unlabeled_pixels = np.zeros(
        n_tiles,
        dtype=np.int32,
    )


    failures = []


    for i in range(
        n_tiles
    ):

        if not finite_pc3[
            i
        ]:

            failures.append(
                {
                    "slide_id":
                        slide_id,

                    "mask_source":
                        source_name,

                    "tile_idx":
                        int(
                            i
                        ),

                    "x":
                        int(
                            x[
                                i
                            ]
                        ),

                    "y":
                        int(
                            y[
                                i
                            ]
                        ),

                    "failure_stage":
                        "finite_pc3",

                    "failure_reason":
                        "non_finite_pc3",
                }
            )

            continue


        try:

            patch = crop_mask_rect(
                img=img,
                x=int(
                    x[
                        i
                    ]
                ),
                y=int(
                    y[
                        i
                    ]
                ),
                patch_size=patch_size,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            geometry_ok[
                i
            ] = True


            measurement = (
                measure_patch(
                    patch,
                    labels,
                )
            )


            measurement_ok[
                i
            ] = True


            labeled_fraction[
                i
            ] = (
                measurement[
                    "labeled_fraction"
                ]
            )


            epithelial_fraction[
                i
            ] = (
                measurement[
                    "epithelial_fraction"
                ]
            )


            non_epithelial_fraction[
                i
            ] = (
                measurement[
                    "non_epithelial_fraction"
                ]
            )


            n_labeled_pixels[
                i
            ] = (
                measurement[
                    "n_labeled"
                ]
            )


            n_epithelial_pixels[
                i
            ] = (
                measurement[
                    "n_epi"
                ]
            )


            n_background_pixels[
                i
            ] = (
                measurement[
                    "n_bg"
                ]
            )


            n_unlabeled_pixels[
                i
            ] = (
                measurement[
                    "n_unl"
                ]
            )


            qc_pass[
                i
            ] = bool(
                np.isfinite(
                    epithelial_fraction[
                        i
                    ]
                )
                and
                labeled_fraction[
                    i
                ]
                >=
                min_labeled
            )


        except Exception as exc:

            failures.append(
                {
                    "slide_id":
                        slide_id,

                    "mask_source":
                        source_name,

                    "tile_idx":
                        int(
                            i
                        ),

                    "x":
                        int(
                            x[
                                i
                            ]
                        ),

                    "y":
                        int(
                            y[
                                i
                            ]
                        ),

                    "failure_stage":
                        "mask_crop_or_measurement",

                    "failure_reason":
                        repr(
                            exc
                        ),
                }
            )


    img.close()


    arrays = {

        "tile_idx":
            tile_idx,

        "x":
            x,

        "y":
            y,

        "pc3_score":
            pc3.astype(
                np.float32,
                copy=False,
            ),

        "finite_pc3":
            finite_pc3,

        "geometry_ok":
            geometry_ok,

        "measurement_ok":
            measurement_ok,

        "qc_pass":
            qc_pass,

        "labeled_fraction":
            labeled_fraction,

        "epithelial_fraction_among_labeled_pixels":
            epithelial_fraction,

        "non_epithelial_fraction_among_labeled_pixels":
            non_epithelial_fraction,

        "n_labeled_pixels":
            n_labeled_pixels,

        "n_epithelial_pixels":
            n_epithelial_pixels,

        "n_non_epithelial_labeled_pixels":
            n_background_pixels,

        "n_unlabeled_pixels":
            n_unlabeled_pixels,
    }


    n_finite = int(
        finite_pc3.sum()
    )

    n_geometry_ok = int(
        geometry_ok.sum()
    )

    n_measurement_ok = int(
        measurement_ok.sum()
    )

    n_qc_pass = int(
        qc_pass.sum()
    )


    summary = {

        "slide_id":
            slide_id,

        "mask_source":
            source_name,

        "mask_path":
            str(
                mask_path
            ),

        "scale_x":
            float(
                scale_x
            ),

        "scale_y":
            float(
                scale_y
            ),

        "n_tiles_total":
            n_tiles,

        "n_tiles_finite_pc3":
            n_finite,

        "n_tiles_geometry_ok":
            n_geometry_ok,

        "n_tiles_measurement_ok":
            n_measurement_ok,

        "n_tiles_qc_pass":
            n_qc_pass,

        "fraction_geometry_ok_of_finite_pc3":
            (
                n_geometry_ok
                / n_finite
                if n_finite
                else np.nan
            ),

        "fraction_measurement_ok_of_finite_pc3":
            (
                n_measurement_ok
                / n_finite
                if n_finite
                else np.nan
            ),

        "fraction_qc_pass_of_finite_pc3":
            (
                n_qc_pass
                / n_finite
                if n_finite
                else np.nan
            ),

        "median_labeled_fraction_measurement_ok":
            (
                float(
                    np.nanmedian(
                        labeled_fraction[
                            measurement_ok
                        ]
                    )
                )
                if measurement_ok.any()
                else np.nan
            ),
    }


    return (
        arrays,
        summary,
        failures,
    )


# =============================================================================
# STANDARD-vs-CORRECTED MEASUREMENT AGREEMENT
# =============================================================================

def agreement_row(
    slide_id: str,
    standard: Dict[str, np.ndarray],
    corrected: Dict[str, np.ndarray],
):

    standard_epi = (
        standard[
            "epithelial_fraction_among_labeled_pixels"
        ]
    )

    corrected_epi = (
        corrected[
            "epithelial_fraction_among_labeled_pixels"
        ]
    )


    if (
        len(
            standard_epi
        )
        !=
        len(
            corrected_epi
        )
    ):

        raise RuntimeError(
            f"{slide_id}: standard "
            "and corrected measurement "
            "arrays have different lengths."
        )


    common = (
        standard[
            "qc_pass"
        ]
        &
        corrected[
            "qc_pass"
        ]
        &
        np.isfinite(
            standard_epi
        )
        &
        np.isfinite(
            corrected_epi
        )
    )


    n_common = int(
        common.sum()
    )


    if n_common == 0:

        return {
            "slide_id":
                slide_id,

            "n_common_qc_pass_tiles":
                0,

            "median_abs_epithelial_fraction_difference":
                np.nan,

            "mean_abs_epithelial_fraction_difference":
                np.nan,

            "median_corrected_minus_standard":
                np.nan,

            "mean_corrected_minus_standard":
                np.nan,
        }


    difference = (
        corrected_epi[
            common
        ]
        -
        standard_epi[
            common
        ]
    )


    return {

        "slide_id":
            slide_id,

        "n_common_qc_pass_tiles":
            n_common,

        "median_abs_epithelial_fraction_difference":
            float(
                np.median(
                    np.abs(
                        difference
                    )
                )
            ),

        "mean_abs_epithelial_fraction_difference":
            float(
                np.mean(
                    np.abs(
                        difference
                    )
                )
            ),

        "median_corrected_minus_standard":
            float(
                np.median(
                    difference
                )
            ),

        "mean_corrected_minus_standard":
            float(
                np.mean(
                    difference
                )
            ),
    }


# =============================================================================
# FORMAL FINAL VALIDATION
# =============================================================================

def validate_formal_full_run(
    measurement_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
):

    n_primary_rows = int(
        (
            measurement_df[
                "analysis_role"
            ]
            ==
            "primary"
        ).sum()
    )


    n_sensitivity_rows = int(
        (
            measurement_df[
                "analysis_role"
            ]
            ==
            "sensitivity"
        ).sum()
    )


    n_primary_slides = int(
        measurement_df.loc[
            measurement_df[
                "analysis_role"
            ]
            ==
            "primary",
            "slide_id",
        ].nunique()
    )


    n_sensitivity_slides = int(
        measurement_df.loc[
            measurement_df[
                "analysis_role"
            ]
            ==
            "sensitivity",
            "slide_id",
        ].nunique()
    )


    if (
        n_primary_rows
        != EXPECTED_PRIMARY_SLIDES
    ):

        raise RuntimeError(
            "Formal Step-16b run "
            "did not produce exactly "
            f"{EXPECTED_PRIMARY_SLIDES} "
            "primary measurement rows; "
            f"found {n_primary_rows}."
        )


    if (
        n_primary_slides
        != EXPECTED_PRIMARY_SLIDES
    ):

        raise RuntimeError(
            "Formal Step-16b run "
            "did not measure exactly "
            f"{EXPECTED_PRIMARY_SLIDES} "
            "unique primary slides; "
            f"found {n_primary_slides}."
        )


    if (
        n_sensitivity_rows
        != EXPECTED_CORRECTED_SLIDES
    ):

        raise RuntimeError(
            "Formal Step-16b run "
            "did not produce exactly "
            f"{EXPECTED_CORRECTED_SLIDES} "
            "corrected sensitivity rows; "
            f"found {n_sensitivity_rows}."
        )


    if (
        n_sensitivity_slides
        != EXPECTED_CORRECTED_SLIDES
    ):

        raise RuntimeError(
            "Formal Step-16b run "
            "did not measure exactly "
            f"{EXPECTED_CORRECTED_SLIDES} "
            "corrected sensitivity slides; "
            f"found {n_sensitivity_slides}."
        )


    expected_qc_rows = (
        EXPECTED_PRIMARY_SLIDES
        +
        EXPECTED_CORRECTED_SLIDES
    )

    if len(
        qc_df
    ) != expected_qc_rows:

        raise RuntimeError(
            "Formal Step-16b QC table "
            "must contain "
            f"{expected_qc_rows} rows; "
            f"found {len(qc_df)}."
        )


    if len(
        agreement_df
    ) != EXPECTED_CORRECTED_SLIDES:

        raise RuntimeError(
            "Corrected agreement QC "
            "must contain exactly "
            f"{EXPECTED_CORRECTED_SLIDES} "
            f"slides; found "
            f"{len(agreement_df)}."
        )


    primary_npz = list(
        PER_SLIDE_DIR.glob(
            "*_standard_measurements.npz"
        )
    )

    corrected_npz = list(
        PER_SLIDE_DIR.glob(
            "*_corrected_measurements.npz"
        )
    )


    if len(
        primary_npz
    ) != EXPECTED_PRIMARY_SLIDES:

        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_PRIMARY_SLIDES} "
            "primary NPZ files; found "
            f"{len(primary_npz)}."
        )


    if len(
        corrected_npz
    ) != EXPECTED_CORRECTED_SLIDES:

        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_CORRECTED_SLIDES} "
            "corrected NPZ files; found "
            f"{len(corrected_npz)}."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()


    print(
        "=" * 124
    )

    print(
        "STEP 16B: PESO PC3 EXTERNAL "
        "COMPARTMENT MEASUREMENT"
    )

    print(
        "=" * 124
    )


    print(
        "[VERSION]      ",
        VERSION,
    )

    print(
        "[SCHEMA]       ",
        SCHEMA_VERSION,
    )

    print(
        "[STEP 16A]     ",
        STEP16A_DIR,
    )

    print(
        "[OUTPUT]       ",
        OUT_DIR,
    )

    print(
        "[OVERWRITE]    ",
        args.overwrite,
    )

    print(
        "[MAX SLIDES]   ",
        args.max_slides,
    )


    # =========================================================================
    # Required Step-16a inputs
    # =========================================================================

    required = [

        (
            STEP16A_MANIFEST,
            "16a manifest",
        ),

        (
            STEP16A_PLAN,
            "16a plan",
        ),

        (
            STEP16A_LABELS,
            "16a labels",
        ),

        (
            STEP16A_GEOMETRY,
            "16a geometry",
        ),

        (
            STEP16A_METADATA,
            "16a metadata",
        ),
    ]


    for path, label in required:

        require_file(
            path,
            label,
        )


    # =========================================================================
    # Output protection / cleanup
    # =========================================================================

    prepare_output_directory(
        overwrite=args.overwrite
    )


    # =========================================================================
    # Load frozen Step 16a
    # =========================================================================

    manifest = pd.read_csv(
        STEP16A_MANIFEST,
        low_memory=False,
    )

    plan16a = load_json(
        STEP16A_PLAN
    )

    labels16a = load_json(
        STEP16A_LABELS
    )

    meta16a = load_json(
        STEP16A_METADATA
    )


    # =========================================================================
    # Provenance/hash chain
    # =========================================================================

    hash_checks = (
        validate_hash_chain(
            meta16a
        )
    )


    # =========================================================================
    # Extract frozen rules
    # =========================================================================

    labels = extract_labels(
        labels16a
    )

    frozen = extract_plan(
        plan16a
    )

    columns = resolve_columns(
        manifest
    )


    # =========================================================================
    # Design validation
    # =========================================================================

    validate_design(
        df=manifest,
        columns=columns,
        plan16a=plan16a,
        meta16a=meta16a,
    )


    # =========================================================================
    # Print frozen provenance
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "STEP-16A PROVENANCE CHAIN"
    )

    print(
        "=" * 124
    )


    for key, info in (
        hash_checks.items()
    ):

        print(
            f"{key:<24} "
            f"match={info['match']}"
        )


    print()

    print(
        "=" * 124
    )

    print(
        "FROZEN MEASUREMENT RULES"
    )

    print(
        "=" * 124
    )


    print(
        "PC index zero-based       :",
        frozen[
            "pc_index_zero_based"
        ],
    )

    print(
        "Minimum labeled fraction :",
        frozen[
            "minimum_labeled_fraction"
        ],
    )

    print(
        "Epithelium label          :",
        labels[
            "epithelium_label"
        ],
    )

    print(
        "Background/non-epi label  :",
        labels[
            "background_label"
        ],
    )

    print(
        "Unlabeled label           :",
        labels[
            "unlabeled_label"
        ],
    )

    print(
        "Primary mask source       :",
        "standard",
    )

    print(
        "Sensitivity mask source   :",
        "corrected subset",
    )

    print(
        "Hybrid                     :",
        "prohibited / not performed",
    )

    print(
        "Independent unit          :",
        frozen[
            "independent_unit"
        ],
    )

    print(
        "Inference                  :",
        "NONE",
    )


    # =========================================================================
    # Formal vs debug run
    # =========================================================================

    formal_full_run = (
        args.max_slides
        is None
    )


    if (
        args.max_slides
        is not None
    ):

        if (
            args.max_slides
            <= 0
        ):

            raise ValueError(
                "--max-slides must "
                "be positive."
            )

        print()

        print(
            "[WARNING] DEBUG / "
            "ENGINEERING RUN ONLY"
        )

        print(
            "[WARNING] formal_full_run=False"
        )


        manifest = (
            manifest
            .head(
                args.max_slides
            )
            .copy()
        )


    # =========================================================================
    # Measurement containers
    # =========================================================================

    measurement_rows = []

    qc_rows = []

    agreement_rows = []

    failure_rows = []


    # =========================================================================
    # Measurement loop
    # =========================================================================

    for number, (
        _,
        row,
    ) in enumerate(
        manifest.iterrows(),
        1,
    ):


        (
            slide_id,
            coords,
            pc3,
        ) = load_slide_arrays(
            row=row,
            columns=columns,
            pc_idx=frozen[
                "pc_index_zero_based"
            ],
        )


        patch_size = int(
            row[
                columns[
                    "patch_size"
                ]
            ]
        )


        standard_path = (
            clean_path(
                row[
                    columns[
                        "standard_mask_path"
                    ]
                ]
            )
        )


        if standard_path is None:

            raise RuntimeError(
                f"{slide_id}: empty "
                "frozen standard mask path."
            )


        standard_scale_x = float(
            row[
                columns[
                    "standard_scale_x"
                ]
            ]
        )

        standard_scale_y = float(
            row[
                columns[
                    "standard_scale_y"
                ]
            ]
        )


        has_corrected = as_bool(
            row[
                columns[
                    "corrected_available"
                ]
            ]
        )


        print(
            f"[SLIDE "
            f"{number:02d}/"
            f"{len(manifest):02d}] "
            f"{slide_id} | "
            f"tiles={len(pc3)} | "
            f"corrected={has_corrected}"
        )


        # =====================================================================
        # PRIMARY: standard mask
        # =====================================================================

        (
            standard_arrays,
            standard_summary,
            standard_failures,
        ) = measure_source(

            slide_id=
                slide_id,

            source_name=
                "standard_primary",

            mask_path=
                standard_path,

            scale_x=
                standard_scale_x,

            scale_y=
                standard_scale_y,

            coords=
                coords,

            pc3=
                pc3,

            patch_size=
                patch_size,

            labels=
                labels,

            min_labeled=
                frozen[
                    "minimum_labeled_fraction"
                ],
        )


        standard_npz = (
            PER_SLIDE_DIR
            / f"{slide_id}_standard_measurements.npz"
        )


        np.savez_compressed(
            standard_npz,
            **standard_arrays,
        )


        standard_summary.update(
            {
                "analysis_role":
                    "primary",

                "npz_path":
                    str(
                        standard_npz
                    ),
            }
        )


        qc_rows.append(
            standard_summary
        )


        failure_rows.extend(
            standard_failures
        )


        measurement_rows.append(
            {
                "slide_id":
                    slide_id,

                "analysis_role":
                    "primary",

                "mask_source":
                    "standard",

                "mask_path":
                    str(
                        standard_path
                    ),

                "scale_x":
                    standard_scale_x,

                "scale_y":
                    standard_scale_y,

                "patch_size":
                    patch_size,

                "n_tiles":
                    len(
                        pc3
                    ),

                "npz_path":
                    str(
                        standard_npz
                    ),

                "corrected_mask_available":
                    has_corrected,
            }
        )


        # =====================================================================
        # SENSITIVITY: corrected mask
        # =====================================================================

        if has_corrected:


            corrected_path = clean_path(
                row[
                    columns[
                        "corrected_mask_path"
                    ]
                ]
            )


            if corrected_path is None:

                raise RuntimeError(
                    f"{slide_id}: corrected "
                    "mask frozen as available "
                    "but path is empty."
                )


            corrected_scale_x = float(
                row[
                    columns[
                        "corrected_scale_x"
                    ]
                ]
            )

            corrected_scale_y = float(
                row[
                    columns[
                        "corrected_scale_y"
                    ]
                ]
            )


            (
                corrected_arrays,
                corrected_summary,
                corrected_failures,
            ) = measure_source(

                slide_id=
                    slide_id,

                source_name=
                    "corrected_sensitivity",

                mask_path=
                    corrected_path,

                scale_x=
                    corrected_scale_x,

                scale_y=
                    corrected_scale_y,

                coords=
                    coords,

                pc3=
                    pc3,

                patch_size=
                    patch_size,

                labels=
                    labels,

                min_labeled=
                    frozen[
                        "minimum_labeled_fraction"
                    ],
            )


            corrected_npz = (
                PER_SLIDE_DIR
                / f"{slide_id}_corrected_measurements.npz"
            )


            np.savez_compressed(
                corrected_npz,
                **corrected_arrays,
            )


            corrected_summary.update(
                {
                    "analysis_role":
                        "sensitivity",

                    "npz_path":
                        str(
                            corrected_npz
                        ),
                }
            )


            qc_rows.append(
                corrected_summary
            )


            failure_rows.extend(
                corrected_failures
            )


            measurement_rows.append(
                {
                    "slide_id":
                        slide_id,

                    "analysis_role":
                        "sensitivity",

                    "mask_source":
                        "corrected",

                    "mask_path":
                        str(
                            corrected_path
                        ),

                    "scale_x":
                        corrected_scale_x,

                    "scale_y":
                        corrected_scale_y,

                    "patch_size":
                        patch_size,

                    "n_tiles":
                        len(
                            pc3
                        ),

                    "npz_path":
                        str(
                            corrected_npz
                        ),

                    "corrected_mask_available":
                        True,
                }
            )


            agreement_rows.append(
                agreement_row(
                    slide_id=
                        slide_id,

                    standard=
                        standard_arrays,

                    corrected=
                        corrected_arrays,
                )
            )


    # =========================================================================
    # Assemble tables
    # =========================================================================

    measurement_df = pd.DataFrame(
        measurement_rows
    )

    qc_df = pd.DataFrame(
        qc_rows
    )

    agreement_df = pd.DataFrame(
        agreement_rows
    )

    failure_df = pd.DataFrame(
        failure_rows
    )


    # =========================================================================
    # Coverage summary
    # =========================================================================

    coverage_rows = []


    if not qc_df.empty:

        for (
            analysis_role,
            mask_source,
        ), subset in (
            qc_df.groupby(
                [
                    "analysis_role",
                    "mask_source",
                ],
                dropna=False,
            )
        ):


            n_finite = int(
                subset[
                    "n_tiles_finite_pc3"
                ].sum()
            )


            n_measurement_ok = int(
                subset[
                    "n_tiles_measurement_ok"
                ].sum()
            )


            n_qc_pass = int(
                subset[
                    "n_tiles_qc_pass"
                ].sum()
            )


            coverage_rows.append(
                {
                    "analysis_role":
                        analysis_role,

                    "mask_source":
                        mask_source,

                    "n_slides":
                        int(
                            subset[
                                "slide_id"
                            ].nunique()
                        ),

                    "n_tiles_total":
                        int(
                            subset[
                                "n_tiles_total"
                            ].sum()
                        ),

                    "n_tiles_finite_pc3":
                        n_finite,

                    "n_tiles_measurement_ok":
                        n_measurement_ok,

                    "n_tiles_qc_pass":
                        n_qc_pass,

                    "overall_measurement_ok_fraction_of_finite_pc3":
                        (
                            n_measurement_ok
                            / n_finite
                            if n_finite
                            else np.nan
                        ),

                    "overall_qc_pass_fraction_of_finite_pc3":
                        (
                            n_qc_pass
                            / n_finite
                            if n_finite
                            else np.nan
                        ),

                    "minimum_slide_qc_pass_fraction_of_finite_pc3":
                        float(
                            subset[
                                "fraction_qc_pass_of_finite_pc3"
                            ].min()
                        ),

                    "median_slide_qc_pass_fraction_of_finite_pc3":
                        float(
                            subset[
                                "fraction_qc_pass_of_finite_pc3"
                            ].median()
                        ),
                }
            )


    coverage_df = pd.DataFrame(
        coverage_rows
    )


    # =========================================================================
    # Save CSV outputs
    # =========================================================================

    measurement_df.to_csv(
        OUT_MANIFEST,
        index=False,
    )

    qc_df.to_csv(
        OUT_QC,
        index=False,
    )

    agreement_df.to_csv(
        OUT_AGREEMENT,
        index=False,
    )

    failure_df.to_csv(
        OUT_FAILURES,
        index=False,
    )

    coverage_df.to_csv(
        OUT_COVERAGE,
        index=False,
    )


    # =========================================================================
    # Formal full-run checks
    # =========================================================================

    if formal_full_run:

        validate_formal_full_run(
            measurement_df=
                measurement_df,

            qc_df=
                qc_df,

            agreement_df=
                agreement_df,
        )


    # =========================================================================
    # Metadata
    # =========================================================================

    metadata = {

        "step":
            "16b",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),


        "formal_full_run":
            formal_full_run,

        "debug_max_slides":
            args.max_slides,


        "measurement_only":
            True,

        "inferential_statistics_performed":
            False,

        "pc3_direction_inspected_by_script":
            False,

        "feature_selection_performed":
            False,

        "hybrid_mask_analysis_performed":
            False,


        "step16a": {

            "version":
                meta16a.get(
                    "version"
                ),

            "schema_version":
                meta16a.get(
                    "schema_version"
                ),

            "manifest_sha256":
                sha256_file(
                    STEP16A_MANIFEST
                ),

            "plan_sha256":
                sha256_file(
                    STEP16A_PLAN
                ),

            "label_specification_sha256":
                sha256_file(
                    STEP16A_LABELS
                ),

            "geometry_audit_sha256":
                sha256_file(
                    STEP16A_GEOMETRY
                ),

            "metadata_sha256":
                sha256_file(
                    STEP16A_METADATA
                ),

            "provenance_hash_chain_verified":
                True,
        },


        "frozen_rules_used": {

            "pc_index_zero_based":
                frozen[
                    "pc_index_zero_based"
                ],

            "minimum_labeled_fraction":
                frozen[
                    "minimum_labeled_fraction"
                ],

            **labels,

            "primary_mask_source":
                "standard",

            "sensitivity_mask_source":
                "corrected",

            "independent_unit":
                frozen[
                    "independent_unit"
                ],
        },


        "n_unique_slides_measured":
            (
                int(
                    measurement_df[
                        "slide_id"
                    ].nunique()
                )
                if not measurement_df.empty
                else 0
            ),


        "n_primary_standard_slides":
            (
                int(
                    measurement_df.loc[
                        measurement_df[
                            "analysis_role"
                        ]
                        ==
                        "primary",
                        "slide_id",
                    ].nunique()
                )
                if not measurement_df.empty
                else 0
            ),


        "n_corrected_sensitivity_slides":
            (
                int(
                    measurement_df.loc[
                        measurement_df[
                            "analysis_role"
                        ]
                        ==
                        "sensitivity",
                        "slide_id",
                    ].nunique()
                )
                if not measurement_df.empty
                else 0
            ),


        "n_measurement_failure_records":
            int(
                len(
                    failure_df
                )
            ),


        "n_primary_npz_files":
            len(
                list(
                    PER_SLIDE_DIR.glob(
                        "*_standard_measurements.npz"
                    )
                )
            ),


        "n_corrected_npz_files":
            len(
                list(
                    PER_SLIDE_DIR.glob(
                        "*_corrected_measurements.npz"
                    )
                )
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


    write_json(
        OUT_METADATA,
        metadata,
    )


    # =========================================================================
    # README
    # =========================================================================

    OUT_README.write_text(
        f"""
STEP 16B — PESO PC3 EXTERNAL COMPARTMENT MEASUREMENT
=====================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA_VERSION}


ROLE
----

Measurement and QC only.

No PC3 association inference is performed in Step 16b.


FROZEN STEP-16A DESIGN
----------------------

Primary:

    standard masks for all 62 PESO training WSIs

Sensitivity:

    corrected masks for the frozen 25-slide corrected subset

Hybrid:

    prohibited / not performed


FROZEN LABELS
-------------

Epithelium:

    {labels['epithelium_label']}

Background / non-epithelial labeled compartment:

    {labels['background_label']}

Unlabeled:

    {labels['unlabeled_label']}


FROZEN ENDPOINT
---------------

epithelial_fraction_among_labeled_pixels =

        N(epithelium label)
    ---------------------------
    N(epithelium OR background label)

With the frozen Step-16a mapping:

        N(mask == 2)
    -------------------
      N(mask in {{1,2}})

Unlabeled pixels are excluded from the denominator.


FROZEN TILE QC
--------------

Minimum labeled fraction:

    {frozen['minimum_labeled_fraction']}

A tile passes measurement QC only when:

    PC3 is finite

    geometry mapping succeeds

    labeled pixels exist

    epithelial fraction is finite

    labeled fraction >= frozen minimum

No additional morphology or data-driven QC threshold is introduced
in Step 16b.


GEOMETRY
--------

H&E tile footprints are mapped into masks using the per-slide
scale_x and scale_y frozen by Step 16a.

Categorical masks are cropped directly.

No continuous interpolation is performed.


PRIMARY MEASUREMENTS
--------------------

For each of 62 standard-mask slides, Step 16b stores:

    tile_idx
    x
    y
    pc3_score

    finite_pc3
    geometry_ok
    measurement_ok
    qc_pass

    labeled_fraction

    epithelial_fraction_among_labeled_pixels

    non_epithelial_fraction_among_labeled_pixels

    n_labeled_pixels
    n_epithelial_pixels
    n_non_epithelial_labeled_pixels
    n_unlabeled_pixels


CORRECTED-MASK SENSITIVITY
--------------------------

For each of the frozen 25 corrected-mask slides, the exact same
physical H&E tile footprints are measured using the corrected mask
coordinate scale frozen by Step 16a.


CORRECTED AGREEMENT QC
----------------------

Standard-vs-corrected epithelial-fraction differences are reported only
as measurement agreement QC on common QC-passing tiles.

PC3 is NOT used in the agreement calculation.

This is not the formal PC3 association analysis.


INFERENCE
---------

None.

Step 16b does NOT calculate:

    Spearman rho
    Pearson r
    p-values
    bootstrap confidence intervals
    high/low PC3 effects
    direction tests
    FDR


FORMAL STATISTICAL UNIT
-----------------------

Frozen by Step 16a as:

    {frozen['independent_unit']}

Tile observations remain repeated image measurements nested within slides.


FORMAL FULL-RUN REQUIREMENTS
----------------------------

A complete formal run must produce:

    62 primary standard measurement NPZ files

    25 corrected sensitivity measurement NPZ files

    62 primary manifest rows

    25 corrected sensitivity manifest rows

    25 corrected agreement-QC rows


DEBUG RUNS
----------

Using:

    --max-slides

creates an engineering/debug run only.

Such a run is recorded as:

    formal_full_run = False

It must not be used as the formal Step-16 measurement output.


NEXT STEP
---------

Step 16c may perform the frozen slide-level external inference only
after Step-16b measurement/QC output has been reviewed and accepted.

Step 16c must not redefine:

    cohort
    PC index
    mask source
    corrected-mask role
    mask scale
    mask labels
    labeled-pixel denominator
    minimum labeled fraction
    independent unit
""".strip()
        + "\n",
        encoding="utf-8",
    )


    # =========================================================================
    # Console summary
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "MEASUREMENT COVERAGE"
    )

    print(
        "=" * 124
    )


    if coverage_df.empty:

        print(
            "No coverage rows."
        )

    else:

        print(
            coverage_df.to_string(
                index=False
            )
        )


    print()

    print(
        "=" * 124
    )

    print(
        "CORRECTED-MASK "
        "MEASUREMENT AGREEMENT QC"
    )

    print(
        "=" * 124
    )


    if agreement_df.empty:

        print(
            "No corrected sensitivity "
            "rows in this run."
        )

    else:

        print(
            agreement_df.to_string(
                index=False
            )
        )


    print()

    print(
        "=" * 124
    )

    print(
        "FAILURE AUDIT"
    )

    print(
        "=" * 124
    )


    print(
        "Failure records:",
        len(
            failure_df
        ),
    )


    if not failure_df.empty:

        print(
            failure_df[
                "failure_stage"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


    print()

    print(
        "=" * 124
    )

    print(
        "STEP 16B SUMMARY"
    )

    print(
        "=" * 124
    )


    print(
        "Formal full run             :",
        formal_full_run,
    )


    if not measurement_df.empty:

        n_primary = int(
            measurement_df.loc[
                measurement_df[
                    "analysis_role"
                ]
                ==
                "primary",
                "slide_id",
            ].nunique()
        )

        n_sensitivity = int(
            measurement_df.loc[
                measurement_df[
                    "analysis_role"
                ]
                ==
                "sensitivity",
                "slide_id",
            ].nunique()
        )

    else:

        n_primary = 0

        n_sensitivity = 0


    print(
        "Primary standard slides     :",
        n_primary,
    )

    print(
        "Corrected sensitivity slides:",
        n_sensitivity,
    )

    print(
        "Measurement failure records :",
        len(
            failure_df
        ),
    )


    print()

    print(
        "[OK] Saved:"
    )


    for path in CORE_OUTPUTS:

        print(
            " ",
            path,
        )


    print(
        " ",
        PER_SLIDE_DIR,
    )


    print()

    if formal_full_run:

        print(
            "[PASS] Formal Step-16b "
            "measurement structure complete."
        )

    else:

        print(
            "[DEBUG] Engineering run only; "
            "not a formal Step-16b result."
        )


    print()

    print(
        "[DONE] Step-16b PESO PC3 "
        "external compartment "
        "measurement complete."
    )

    print(
        "[INFERENCE] None performed."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()