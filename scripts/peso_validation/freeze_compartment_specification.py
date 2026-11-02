#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16a_freeze_peso_pc3_external_compartment_specification.py

Step 16a
========

Freeze the formal PESO external-validation specification for PC3
histologic compartment interpretation.

This step DOES NOT compute PC3-compartment associations.

The formal design is intentionally conservative:

1. External cohort
------------------
PESO training cohort only:
    62 prostatectomy H&E WSIs.

2. Frozen latent representation
--------------------------------
Virchow2 embeddings are projected into the discovery-cohort PCA space.
PCA is NOT refit on PESO.

3. PRIMARY mask source
----------------------
Use the standard PESO training mask for ALL 62 training slides.

Reason:
    The standard masks provide a homogeneous reference source across the
    complete 62-slide training cohort.

4. CORRECTED-MASK sensitivity subset
------------------------------------
For the 25 slides with a corrected mask, repeat the same analysis using
the corrected mask.

The corrected masks are NOT mixed with standard masks in the primary
analysis.

5. Mask label semantics
-----------------------
Observed PESO training masks contain categorical values {0, 1, 2}.

The official PESO test-set ground-truth documentation defines:
    0 = unlabeled
    1 = background
    2 = epithelial tissue

Because the training masks use the identical {0,1,2} encoding and are
provided as epithelial segmentation masks, the same mapping is adopted
for the formal Step-16 analysis.

This provenance distinction is explicitly recorded. The script does not
claim that the training-mask documentation independently restates the
label mapping verbatim.

6. Geometry
-----------
Standard masks:
    approximately 1:1 resolution with H&E.

Corrected masks:
    approximately 2x resolution relative to H&E.

Per-slide scale_x and scale_y are explicitly frozen.

7. PRIMARY biological endpoint
------------------------------
Epithelial fraction among labeled pixels:

    N(mask == 2)
    -----------------
    N(mask in {1,2})

Label 0 pixels are excluded.

8. Derived complementary endpoint
---------------------------------
Non-epithelial labeled fraction:

    N(mask == 1)
    -----------------
    N(mask in {1,2})

This endpoint is exactly complementary to epithelial fraction.

IMPORTANT:
Label 1 must NOT automatically be described as a gold-standard stromal
segmentation.

9. Formal statistical unit
--------------------------
Slide.

Tiles are repeated measurements nested within slides and are not treated
as independent external-cohort samples.

10. Primary statistical analysis
--------------------------------
For each slide:

    frozen tile-level PC3
        versus
    epithelial fraction

using Spearman correlation.

The external cohort is summarized using slide-level effects.

11. Corrected-mask sensitivity analysis
---------------------------------------
Repeat the same PC3 association in the 25-slide corrected-mask subset.

12. Provenance
--------------
Prior exploratory PESO analyses existed before this formal Step-16
re-analysis.

Therefore this is NOT described as an independent preregistration.

Instead, the formal cohort, PCA projection, endpoint, mask-source roles,
geometry handling, QC rules, and statistical procedures are frozen before
the formal Step-16 re-analysis.

Expected output
---------------
MedIA_paper/results/16a_peso_pc3_external_compartment_specification/

    peso_pc3_external_mask_manifest.csv
    peso_pc3_external_analysis_plan.json
    peso_pc3_external_label_specification.json
    peso_pc3_external_geometry_audit.csv
    peso_pc3_external_cohort_summary.csv
    evidence_provenance.csv
    source_hashes.csv
    analysis_metadata.json
    README.txt
"""

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

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# VERSION
# =============================================================================

VERSION = "2.0.0"

SCHEMA_VERSION = (
    "16a-peso-pc3-external-schema-2.0"
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT = CONFIG_PESO_ROOT.parent

MEDIA_ROOT = REPO_ROOT

PESO_ROOT = (
    PROJECT
    / "PESO_dataset"
)

EMB_DIR = (
    PESO_ROOT
    / "peso_virchow2_embeddings"
)

PCA_DIR = (
    PROJECT
    / "pca_global_projected_from_main"
)

DISCOVERY_PCA_MODEL = FROZEN_PCA_MODEL


# =============================================================================
# MASK PATHS
# =============================================================================

# Homogeneous primary reference:
# all 62 training masks.
STANDARD_MASK_ROOTS = [
    PESO_ROOT
    / "peso_training_masks",
]


# Corrected-mask sensitivity subset:
# exactly the audited 25 corrected masks.
CORRECTED_MASK_ROOTS = [
    PESO_ROOT
    / "peso_training_masks_corrected",
]


# =============================================================================
# OPTIONAL PRIOR EXPLORATORY FILES
# =============================================================================

PRIOR_EXPLORATORY_FILES = [
    PROJECT
    / "peso_pc3_compartment_validation.py",

    PROJECT
    / "validate_pc3_epithelium_stroma_fraction.py",

    PROJECT
    / "validate_peso_pc3_epithelium_fraction.py",

    PROJECT
    / "analyze_peso_pc3_pc6_relationship.py",
]


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = (
    RESULTS_ROOT
    / "16a_peso_pc3_external_compartment_specification"
)

OUT_MANIFEST = (
    OUT_DIR
    / "peso_pc3_external_mask_manifest.csv"
)

OUT_PLAN = (
    OUT_DIR
    / "peso_pc3_external_analysis_plan.json"
)

OUT_LABELS = (
    OUT_DIR
    / "peso_pc3_external_label_specification.json"
)

OUT_GEOMETRY = (
    OUT_DIR
    / "peso_pc3_external_geometry_audit.csv"
)

OUT_COHORT = (
    OUT_DIR
    / "peso_pc3_external_cohort_summary.csv"
)

OUT_PROVENANCE = (
    OUT_DIR
    / "evidence_provenance.csv"
)

OUT_HASHES = (
    OUT_DIR
    / "source_hashes.csv"
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
    OUT_PLAN,
    OUT_LABELS,
    OUT_GEOMETRY,
    OUT_COHORT,
    OUT_PROVENANCE,
    OUT_HASHES,
    OUT_METADATA,
    OUT_README,
]


# =============================================================================
# FROZEN SPECIFICATION
# =============================================================================

PC_INDEX_ZERO_BASED = 2

PC_NAME = "PC3"

PATCH_SIZE_EXPECTED = 224


# -----------------------------------------------------------------------------
# Mask labels
#
# IMPORTANT:
# These values are formally adopted for Step 16 based on:
#
# 1. observed training-mask support {0,1,2};
# 2. PESO masks being epithelial segmentation masks;
# 3. official PESO test-set ground-truth documentation defining
#    0 / 1 / 2 as unlabeled / background / epithelial tissue.
#
# Provenance distinction is explicitly recorded below.
# -----------------------------------------------------------------------------

UNLABELED_LABEL = 0

BACKGROUND_LABEL = 1

EPITHELIUM_LABEL = 2

EXPECTED_LABELS = {
    UNLABELED_LABEL,
    BACKGROUND_LABEL,
    EPITHELIUM_LABEL,
}


# -----------------------------------------------------------------------------
# Geometry compatibility
# -----------------------------------------------------------------------------

STANDARD_SCALE_RANGE = (
    0.995,
    1.005,
)

CORRECTED_SCALE_RANGE = (
    1.95,
    2.05,
)


# -----------------------------------------------------------------------------
# Formal measurement QC
# -----------------------------------------------------------------------------

FORMAL_MIN_LABELED_FRACTION = 0.10


# =============================================================================
# HELPERS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite-freeze",
        action="store_true",
        help=(
            "Explicitly allow replacement of an "
            "existing Step-16a frozen specification."
        ),
    )

    return parser.parse_args()


def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):

    path = Path(path)

    if (
        not path.exists()
        or not path.is_file()
    ):
        return None

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def write_json(
    path,
    obj,
):

    with open(
        path,
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


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def normalize_slide_id(value):

    s = Path(
        str(value)
    ).stem.lower()

    return re.sub(
        r"[^a-z0-9]",
        "",
        s,
    )


def clean_slide_key(value):

    norm = normalize_slide_id(
        value
    )

    match = re.search(
        r"(pds\d+he)",
        norm,
    )

    if match is None:
        return None

    return match.group(1)


def discover_tiff_files(
    roots,
):

    files = []

    seen = set()

    for root in roots:

        root = Path(root)

        if not root.exists():
            continue

        for pattern in (
            "*.tif",
            "*.tiff",
        ):

            for path in root.rglob(
                pattern
            ):

                resolved = str(
                    path.resolve()
                )

                if resolved in seen:
                    continue

                seen.add(
                    resolved
                )

                files.append(
                    path
                )

    return sorted(
        files
    )


def classify_mask_candidate(path):

    text = str(
        path
    ).lower()

    if "correct" in text:
        return "corrected"

    if "training_mask" in text:
        return "standard"

    if "mask" in text:
        return "standard"

    return "unknown"


def build_mask_index(
    paths,
    expected_type,
):

    index = {}

    for path in paths:

        key = clean_slide_key(
            path.name
        )

        if key is None:
            continue

        inferred = (
            classify_mask_candidate(
                path
            )
        )

        if (
            expected_type
            == "corrected"
            and inferred
            != "corrected"
        ):
            continue

        if (
            expected_type
            == "standard"
            and inferred
            == "corrected"
        ):
            continue

        index.setdefault(
            key,
            [],
        ).append(
            path
        )

    return index


def choose_unique_mask(
    index,
    slide_key,
    mask_type,
):

    candidates = index.get(
        slide_key,
        [],
    )

    unique = {}

    for path in candidates:

        unique[
            str(
                path.resolve()
            )
        ] = path

    candidates = list(
        unique.values()
    )

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:

        raise RuntimeError(
            f"Ambiguous {mask_type} masks "
            f"for {slide_key}:\n"
            +
            "\n".join(
                f"  - {x}"
                for x in candidates
            )
        )

    return candidates[0]


def get_image_info(path):

    with Image.open(
        path
    ) as img:

        width, height = (
            img.size
        )

        mode = img.mode

        extrema = (
            img.getextrema()
        )

    return {
        "width":
            int(width),

        "height":
            int(height),

        "mode":
            str(mode),

        "extrema":
            str(extrema),
    }


def inspect_mask_labels(
    path,
    max_size=2048,
):

    """
    Audit categorical labels using a nearest-neighbor
    downsampled representation.

    This is only a label-support audit.

    Formal Step-16b measurements will operate directly
    on the corresponding mask regions.
    """

    with Image.open(
        path
    ) as img:

        width, height = (
            img.size
        )

        scale = min(
            1.0,
            max_size
            / max(
                width,
                height,
            ),
        )

        if scale < 1.0:

            new_size = (
                max(
                    1,
                    int(
                        round(
                            width
                            * scale
                        )
                    ),
                ),
                max(
                    1,
                    int(
                        round(
                            height
                            * scale
                        )
                    ),
                ),
            )

            img = img.resize(
                new_size,
                resample=(
                    Image.Resampling.NEAREST
                ),
            )

        arr = np.asarray(
            img
        )

        if arr.ndim == 3:

            # PESO masks are expected to be single-channel.
            # If a future file is RGB, this audit uses the
            # first channel but preserves the unexpected mode
            # in the manifest for review.
            arr = arr[:, :, 0]

    unique = np.unique(
        arr
    )

    return [
        int(x)
        for x in unique.tolist()
    ]


def scale_is_compatible(
    scale_x,
    scale_y,
    mask_type,
):

    if mask_type == "standard":

        lo, hi = (
            STANDARD_SCALE_RANGE
        )

    elif mask_type == "corrected":

        lo, hi = (
            CORRECTED_SCALE_RANGE
        )

    else:

        return False

    return (
        lo
        <= scale_x
        <= hi
        and
        lo
        <= scale_y
        <= hi
    )


def validate_frozen_pca(
    prefix,
    meta,
    coords_path,
    pca_path,
):

    if not coords_path.exists():

        raise FileNotFoundError(
            f"Missing coordinates for "
            f"{prefix}: "
            f"{coords_path}"
        )

    if not pca_path.exists():

        raise FileNotFoundError(
            f"Missing frozen PCA "
            f"projection for "
            f"{prefix}: "
            f"{pca_path}"
        )

    coords = np.load(
        coords_path,
        mmap_mode="r",
    )

    pca = np.load(
        pca_path,
        mmap_mode="r",
    )

    if coords.ndim != 2:

        raise RuntimeError(
            f"Coordinates are not "
            f"2D for {prefix}: "
            f"{coords.shape}"
        )

    if pca.ndim != 2:

        raise RuntimeError(
            f"PCA targets are not "
            f"2D for {prefix}: "
            f"{pca.shape}"
        )

    if (
        coords.shape[0]
        != pca.shape[0]
    ):

        raise RuntimeError(
            f"Coordinate/PCA tile "
            f"count mismatch for "
            f"{prefix}: "
            f"coords={coords.shape}, "
            f"pca={pca.shape}"
        )

    n_meta = int(
        meta.get(
            "n_tiles",
            coords.shape[0],
        )
    )

    if (
        n_meta
        != coords.shape[0]
    ):

        raise RuntimeError(
            f"Metadata tile count "
            f"mismatch for "
            f"{prefix}: "
            f"meta={n_meta}, "
            f"coords={coords.shape[0]}"
        )

    if (
        pca.shape[1]
        <= PC_INDEX_ZERO_BASED
    ):

        raise RuntimeError(
            f"PCA projection does "
            f"not contain {PC_NAME}: "
            f"{prefix}, "
            f"shape={pca.shape}"
        )

    return (
        int(
            coords.shape[0]
        ),
        int(
            pca.shape[1]
        ),
    )


# =============================================================================
# COHORT DISCOVERY
# =============================================================================

def load_training_slides():

    meta_paths = sorted(
        EMB_DIR.glob(
            "*_meta.json"
        )
    )

    if len(meta_paths) == 0:

        raise RuntimeError(
            f"No metadata files "
            f"found in {EMB_DIR}"
        )

    rows = []

    for meta_path in meta_paths:

        meta = load_json(
            meta_path
        )

        split = str(
            meta.get(
                "split",
                "",
            )
        ).strip().lower()

        if split != "training":
            continue

        prefix = (
            meta.get(
                "slide_id"
            )
            or
            meta_path.name.replace(
                "_meta.json",
                "",
            )
        )

        prefix = str(
            prefix
        )

        slide_path = meta.get(
            "slide_path"
        )

        if not slide_path:

            raise RuntimeError(
                f"Training metadata "
                f"lacks slide_path: "
                f"{meta_path}"
            )

        slide_path = Path(
            slide_path
        )

        if not slide_path.exists():

            raise FileNotFoundError(
                f"Training WSI does "
                f"not exist: "
                f"{slide_path}"
            )

        coords_path = (
            EMB_DIR
            / f"{prefix}_coords.npy"
        )

        pca_path = (
            PCA_DIR
            / f"{prefix}_pca_targets.npy"
        )

        (
            n_tiles,
            n_components,
        ) = validate_frozen_pca(
            prefix=prefix,
            meta=meta,
            coords_path=coords_path,
            pca_path=pca_path,
        )

        slide_width = int(
            meta["slide_width"]
        )

        slide_height = int(
            meta["slide_height"]
        )

        patch_size = int(
            meta.get(
                "patch_size",
                PATCH_SIZE_EXPECTED,
            )
        )

        rows.append(
            {
                "slide_id":
                    prefix,

                "slide_key":
                    clean_slide_key(
                        prefix
                    ),

                "split":
                    split,

                "group_dir":
                    meta.get(
                        "group_dir"
                    ),

                "wsi_path":
                    str(
                        slide_path
                    ),

                "meta_path":
                    str(
                        meta_path
                    ),

                "coords_path":
                    str(
                        coords_path
                    ),

                "pca_path":
                    str(
                        pca_path
                    ),

                "slide_width":
                    slide_width,

                "slide_height":
                    slide_height,

                "patch_size":
                    patch_size,

                "n_tiles":
                    n_tiles,

                "n_pca_components":
                    n_components,

                "embedding_model":
                    meta.get(
                        "model"
                    ),

                "embedding_dtype":
                    meta.get(
                        "dtype"
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    if df.empty:

        raise RuntimeError(
            "No PESO training "
            "slides were identified."
        )

    if (
        df[
            "slide_id"
        ].duplicated().any()
    ):

        dup = df.loc[
            df[
                "slide_id"
            ].duplicated(
                False
            ),
            [
                "slide_id",
                "meta_path",
            ],
        ]

        raise RuntimeError(
            "Duplicate PESO training "
            "slide IDs detected:\n"
            +
            dup.to_string(
                index=False
            )
        )

    if (
        df[
            "slide_key"
        ].isna().any()
    ):

        bad = df.loc[
            df[
                "slide_key"
            ].isna(),
            [
                "slide_id"
            ],
        ]

        raise RuntimeError(
            "Could not normalize "
            "some slide IDs:\n"
            +
            bad.to_string(
                index=False
            )
        )

    return (
        df
        .sort_values(
            "slide_id"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# MASK MANIFEST
# =============================================================================

def build_mask_manifest(
    training_df,
):

    standard_files = (
        discover_tiff_files(
            STANDARD_MASK_ROOTS
        )
    )

    corrected_files = (
        discover_tiff_files(
            CORRECTED_MASK_ROOTS
        )
    )

    print(
        "[MASK DISCOVERY] "
        f"standard masks: "
        f"{len(standard_files)}"
    )

    print(
        "[MASK DISCOVERY] "
        f"corrected masks: "
        f"{len(corrected_files)}"
    )

    standard_index = (
        build_mask_index(
            standard_files,
            expected_type="standard",
        )
    )

    corrected_index = (
        build_mask_index(
            corrected_files,
            expected_type="corrected",
        )
    )

    rows = []

    geometry_rows = []

    for _, slide in (
        training_df.iterrows()
    ):

        slide_id = (
            slide["slide_id"]
        )

        slide_key = (
            slide["slide_key"]
        )

        standard_mask = (
            choose_unique_mask(
                standard_index,
                slide_key,
                "standard",
            )
        )

        corrected_mask = (
            choose_unique_mask(
                corrected_index,
                slide_key,
                "corrected",
            )
        )

        if standard_mask is None:

            raise RuntimeError(
                f"No standard "
                f"training mask "
                f"found for "
                f"{slide_id}"
            )

        standard_info = (
            get_image_info(
                standard_mask
            )
        )

        corrected_info = (
            get_image_info(
                corrected_mask
            )
            if corrected_mask
            is not None
            else None
        )

        w = int(
            slide[
                "slide_width"
            ]
        )

        h = int(
            slide[
                "slide_height"
            ]
        )


        # ---------------------------------------------------------------------
        # Standard-mask geometry
        # ---------------------------------------------------------------------

        std_scale_x = (
            standard_info[
                "width"
            ]
            / w
        )

        std_scale_y = (
            standard_info[
                "height"
            ]
            / h
        )

        std_geometry_ok = (
            scale_is_compatible(
                std_scale_x,
                std_scale_y,
                "standard",
            )
        )

        if not std_geometry_ok:

            raise RuntimeError(
                f"Standard mask "
                f"geometry incompatible "
                f"for {slide_id}: "
                f"H&E=({w},{h}), "
                f"mask=("
                f"{standard_info['width']},"
                f"{standard_info['height']}), "
                f"scale=("
                f"{std_scale_x:.6f},"
                f"{std_scale_y:.6f})"
            )


        # ---------------------------------------------------------------------
        # Corrected-mask geometry
        # ---------------------------------------------------------------------

        if (
            corrected_mask
            is not None
        ):

            corr_scale_x = (
                corrected_info[
                    "width"
                ]
                / w
            )

            corr_scale_y = (
                corrected_info[
                    "height"
                ]
                / h
            )

            corr_geometry_ok = (
                scale_is_compatible(
                    corr_scale_x,
                    corr_scale_y,
                    "corrected",
                )
            )

            if not corr_geometry_ok:

                raise RuntimeError(
                    f"Corrected mask "
                    f"geometry incompatible "
                    f"for {slide_id}: "
                    f"H&E=({w},{h}), "
                    f"mask=("
                    f"{corrected_info['width']},"
                    f"{corrected_info['height']}), "
                    f"scale=("
                    f"{corr_scale_x:.6f},"
                    f"{corr_scale_y:.6f})"
                )

        else:

            corr_scale_x = np.nan

            corr_scale_y = np.nan

            corr_geometry_ok = np.nan


        # ---------------------------------------------------------------------
        # Label-support audit
        # ---------------------------------------------------------------------

        standard_labels = (
            inspect_mask_labels(
                standard_mask
            )
        )

        if not set(
            standard_labels
        ).issubset(
            EXPECTED_LABELS
        ):

            raise RuntimeError(
                f"Unexpected labels "
                f"in standard mask "
                f"{slide_id}: "
                f"{standard_labels}"
            )

        if (
            corrected_mask
            is not None
        ):

            corrected_labels = (
                inspect_mask_labels(
                    corrected_mask
                )
            )

            if not set(
                corrected_labels
            ).issubset(
                EXPECTED_LABELS
            ):

                raise RuntimeError(
                    f"Unexpected labels "
                    f"in corrected mask "
                    f"{slide_id}: "
                    f"{corrected_labels}"
                )

        else:

            corrected_labels = None


        # ---------------------------------------------------------------------
        # Formal manifest row
        #
        # PRIMARY = standard mask for all 62.
        # Corrected masks = sensitivity only.
        # ---------------------------------------------------------------------

        rows.append(
            {
                "slide_id":
                    slide_id,

                "slide_key":
                    slide_key,

                "split":
                    slide["split"],

                "group_dir":
                    slide["group_dir"],


                # -------------------------------------------------------------
                # Frozen latent inputs
                # -------------------------------------------------------------

                "wsi_path":
                    slide["wsi_path"],

                "meta_path":
                    slide["meta_path"],

                "coords_path":
                    slide["coords_path"],

                "pca_path":
                    slide["pca_path"],

                "slide_width":
                    w,

                "slide_height":
                    h,

                "patch_size":
                    int(
                        slide[
                            "patch_size"
                        ]
                    ),

                "n_tiles":
                    int(
                        slide[
                            "n_tiles"
                        ]
                    ),

                "n_pca_components":
                    int(
                        slide[
                            "n_pca_components"
                        ]
                    ),


                # -------------------------------------------------------------
                # PRIMARY mask
                # -------------------------------------------------------------

                "primary_mask_type":
                    "standard",

                "primary_mask_path":
                    str(
                        standard_mask
                    ),

                "primary_mask_width":
                    int(
                        standard_info[
                            "width"
                        ]
                    ),

                "primary_mask_height":
                    int(
                        standard_info[
                            "height"
                        ]
                    ),

                "primary_scale_x":
                    float(
                        std_scale_x
                    ),

                "primary_scale_y":
                    float(
                        std_scale_y
                    ),


                # -------------------------------------------------------------
                # Standard mask explicit fields
                # -------------------------------------------------------------

                "standard_mask_path":
                    str(
                        standard_mask
                    ),

                "standard_mask_width":
                    int(
                        standard_info[
                            "width"
                        ]
                    ),

                "standard_mask_height":
                    int(
                        standard_info[
                            "height"
                        ]
                    ),

                "standard_scale_x":
                    float(
                        std_scale_x
                    ),

                "standard_scale_y":
                    float(
                        std_scale_y
                    ),

                "standard_geometry_compatible":
                    bool(
                        std_geometry_ok
                    ),


                # -------------------------------------------------------------
                # Corrected sensitivity subset
                # -------------------------------------------------------------

                "corrected_mask_available":
                    bool(
                        corrected_mask
                        is not None
                    ),

                "corrected_mask_path":
                    (
                        str(
                            corrected_mask
                        )
                        if corrected_mask
                        is not None
                        else ""
                    ),

                "corrected_mask_width":
                    (
                        int(
                            corrected_info[
                                "width"
                            ]
                        )
                        if corrected_info
                        is not None
                        else np.nan
                    ),

                "corrected_mask_height":
                    (
                        int(
                            corrected_info[
                                "height"
                            ]
                        )
                        if corrected_info
                        is not None
                        else np.nan
                    ),

                "corrected_scale_x":
                    (
                        float(
                            corr_scale_x
                        )
                        if corrected_mask
                        is not None
                        else np.nan
                    ),

                "corrected_scale_y":
                    (
                        float(
                            corr_scale_y
                        )
                        if corrected_mask
                        is not None
                        else np.nan
                    ),

                "corrected_geometry_compatible":
                    (
                        bool(
                            corr_geometry_ok
                        )
                        if corrected_mask
                        is not None
                        else np.nan
                    ),

                "corrected_sensitivity_eligible":
                    bool(
                        corrected_mask
                        is not None
                    ),


                # -------------------------------------------------------------
                # Label audit
                # -------------------------------------------------------------

                "standard_mask_labels_audit":
                    ",".join(
                        map(
                            str,
                            standard_labels,
                        )
                    ),

                "corrected_mask_labels_audit":
                    (
                        ",".join(
                            map(
                                str,
                                corrected_labels,
                            )
                        )
                        if corrected_labels
                        is not None
                        else ""
                    ),

                "epithelium_label":
                    EPITHELIUM_LABEL,

                "background_label":
                    BACKGROUND_LABEL,

                "unlabeled_label":
                    UNLABELED_LABEL,

                "formal_primary_external_validation_eligible":
                    True,
            }
        )


        # ---------------------------------------------------------------------
        # Geometry audit row
        # ---------------------------------------------------------------------

        geometry_rows.append(
            {
                "slide_id":
                    slide_id,

                "slide_width":
                    w,

                "slide_height":
                    h,


                "standard_mask_width":
                    standard_info[
                        "width"
                    ],

                "standard_mask_height":
                    standard_info[
                        "height"
                    ],

                "standard_scale_x":
                    std_scale_x,

                "standard_scale_y":
                    std_scale_y,

                "standard_geometry_compatible":
                    bool(
                        std_geometry_ok
                    ),


                "corrected_mask_available":
                    bool(
                        corrected_mask
                        is not None
                    ),

                "corrected_mask_width":
                    (
                        corrected_info[
                            "width"
                        ]
                        if corrected_info
                        is not None
                        else np.nan
                    ),

                "corrected_mask_height":
                    (
                        corrected_info[
                            "height"
                        ]
                        if corrected_info
                        is not None
                        else np.nan
                    ),

                "corrected_scale_x":
                    corr_scale_x,

                "corrected_scale_y":
                    corr_scale_y,

                "corrected_geometry_compatible":
                    (
                        bool(
                            corr_geometry_ok
                        )
                        if corrected_mask
                        is not None
                        else np.nan
                    ),

                "primary_mask_type":
                    "standard",

                "primary_scale_x":
                    std_scale_x,

                "primary_scale_y":
                    std_scale_y,
            }
        )

    manifest = pd.DataFrame(
        rows
    )

    geometry = pd.DataFrame(
        geometry_rows
    )

    return (
        manifest,
        geometry,
    )


# =============================================================================
# LABEL SPECIFICATION
# =============================================================================

def build_label_specification():

    return {
        "schema_version":
            SCHEMA_VERSION,

        "formal_label_mapping": {
            "0": {
                "name":
                    "unlabeled",

                "pixel_value":
                    UNLABELED_LABEL,

                "formal_role":
                    (
                        "excluded from the "
                        "labeled-compartment denominator"
                    ),
            },

            "1": {
                "name":
                    "background_or_non_epithelial",

                "pixel_value":
                    BACKGROUND_LABEL,

                "formal_role":
                    (
                        "non-epithelial labeled compartment"
                    ),
            },

            "2": {
                "name":
                    "epithelial_tissue",

                "pixel_value":
                    EPITHELIUM_LABEL,

                "formal_role":
                    (
                        "epithelial labeled compartment"
                    ),
            },
        },


        "label_mapping_provenance": {

            "officially_documented_context":
                (
                    "PESO test-set ground-truth masks "
                    "document the categorical mapping "
                    "0=unlabeled, 1=background, "
                    "2=epithelial tissue."
                ),

            "training_mask_observation":
                (
                    "PESO standard and corrected training "
                    "masks were independently audited and "
                    "found to use the same categorical "
                    "value support {0,1,2}."
                ),

            "formal_step16_decision":
                (
                    "The documented PESO test-set label "
                    "mapping is applied to the identically "
                    "encoded PESO training masks for the "
                    "formal Step-16 analysis."
                ),

            "claim_boundary":
                (
                    "This does not imply that the training-mask "
                    "documentation independently restates the "
                    "label semantics verbatim."
                ),
        },


        "primary_endpoint_definition": {

            "name":
                "epithelial_fraction_among_labeled_pixels",

            "numerator":
                "number of pixels with mask value == 2",

            "denominator":
                "number of pixels with mask value in {1,2}",

            "formula":
                "N(mask == 2) / N(mask in {1,2})",

            "interpretation":
                (
                    "fraction of labeled mask pixels "
                    "assigned to epithelium"
                ),
        },


        "derived_complementary_endpoint": {

            "name":
                "non_epithelial_fraction_among_labeled_pixels",

            "formula":
                "N(mask == 1) / N(mask in {1,2})",

            "relationship":
                (
                    "1 - epithelial_fraction_"
                    "among_labeled_pixels"
                ),

            "claim_boundary":
                (
                    "This complementary compartment must "
                    "not be described as a gold-standard "
                    "stromal segmentation without independent "
                    "validation of that semantic."
                ),
        },


        "prohibited_definition": {

            "mask_greater_than_zero":
                (
                    "Do not define epithelium as mask > 0 "
                    "because this would combine labels 1 and 2."
                ),
        },
    }


# =============================================================================
# ANALYSIS PLAN
# =============================================================================

def build_analysis_plan(
    manifest,
):

    n_slides = int(
        manifest[
            "slide_id"
        ].nunique()
    )

    n_corrected = int(
        manifest[
            "corrected_mask_available"
        ].sum()
    )

    return {

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,


        "analysis_name":
            (
                "PESO external PC3 "
                "epithelial-compartment validation"
            ),


        "analysis_status":
            (
                "frozen_before_formal_"
                "step16_reanalysis"
            ),


        "provenance_statement":
            (
                "Endpoint direction and biological hypotheses "
                "were informed by prior exploratory PESO analyses. "
                "The formal external cohort, frozen-PC projection, "
                "primary mask source, corrected-mask sensitivity "
                "role, endpoint definition, geometry handling, "
                "measurement QC, aggregation strategy, and "
                "inferential procedures were frozen before the "
                "formal Step-16 re-analysis."
            ),


        # =====================================================================
        # Cohort
        # =====================================================================

        "cohort": {

            "dataset":
                "PESO",

            "formal_subset":
                "training",

            "n_training_slides":
                n_slides,

            "expected_n_training_slides":
                62,

            "primary_mask_complete_cohort":
                62,

            "corrected_mask_sensitivity_subset":
                n_corrected,

            "test_set_role":
                (
                    "The 40-slide PESO test cohort is excluded "
                    "from this whole-slide epithelial-compartment "
                    "analysis and reserved for a separate "
                    "region-level external-validation analysis."
                ),
        },


        # =====================================================================
        # Frozen representation
        # =====================================================================

        "latent_representation": {

            "embedding_model":
                "Virchow2",

            "target_axis":
                PC_NAME,

            "target_axis_zero_based_index":
                PC_INDEX_ZERO_BASED,

            "pca_rule":
                (
                    "Use PESO PCA scores obtained by projection "
                    "into the discovery-cohort frozen PCA space. "
                    "PCA must not be refit on PESO."
                ),

            "discovery_pca_model":
                str(
                    DISCOVERY_PCA_MODEL
                ),
        },


        # =====================================================================
        # Primary mask
        # =====================================================================

        "primary_mask_source": {

            "mask_type":
                "standard_training_mask",

            "n_slides":
                n_slides,

            "rule":
                (
                    "Use the PESO standard training mask "
                    "for every one of the 62 training slides."
                ),

            "rationale":
                (
                    "The primary external analysis uses a "
                    "homogeneous mask source across the complete "
                    "PESO training cohort and therefore does not "
                    "mix standard and corrected mask-generation "
                    "procedures or resolutions."
                ),
        },


        # =====================================================================
        # Corrected-mask sensitivity
        # =====================================================================

        "corrected_mask_sensitivity": {

            "name":
                "corrected_mask_subset_sensitivity",

            "n_slides":
                n_corrected,

            "expected_n_slides":
                25,

            "rule":
                (
                    "For slides with a corrected PESO mask, "
                    "repeat the same PC3-compartment analysis "
                    "using the corrected mask instead of the "
                    "standard mask."
                ),

            "role":
                "sensitivity_analysis",

            "rationale":
                (
                    "Assess whether the direction and magnitude "
                    "of the PC3 epithelial-compartment association "
                    "remain consistent when measurement uses the "
                    "corrected-mask subset."
                ),

            "not_primary":
                True,

            "no_hybrid_primary_masking":
                True,
        },


        # =====================================================================
        # Geometry
        # =====================================================================

        "geometry": {

            "coordinate_system":
                (
                    "Virchow2 tile coordinates are defined "
                    "in H&E level-0 pixel coordinates."
                ),

            "standard_mask_rule":
                (
                    "Map H&E tile coordinates into the "
                    "standard mask using per-slide scale_x "
                    "and scale_y frozen in the Step-16a manifest."
                ),

            "corrected_mask_rule":
                (
                    "For the corrected-mask sensitivity analysis, "
                    "map the same physical H&E tile footprint into "
                    "the corrected mask using its per-slide scale_x "
                    "and scale_y. Do not assume an exact integer "
                    "factor of two."
                ),

            "mapping_formula": {

                "mask_x0":
                    "round(x_he * scale_x)",

                "mask_y0":
                    "round(y_he * scale_y)",

                "mask_x1":
                    (
                        "round((x_he + patch_size) "
                        "* scale_x)"
                    ),

                "mask_y1":
                    (
                        "round((y_he + patch_size) "
                        "* scale_y)"
                    ),
            },

            "standard_expected_scale_range":
                list(
                    STANDARD_SCALE_RANGE
                ),

            "corrected_expected_scale_range":
                list(
                    CORRECTED_SCALE_RANGE
                ),

            "categorical_mask_rule":
                (
                    "No continuous interpolation of categorical "
                    "mask labels is permitted for formal "
                    "measurement. The corresponding mask rectangle "
                    "is cropped directly in mask coordinates."
                ),
        },


        # =====================================================================
        # Primary endpoint
        # =====================================================================

        "primary_endpoint": {

            "feature":
                (
                    "epithelial_fraction_"
                    "among_labeled_pixels"
                ),

            "expected_pc3_direction":
                "negative",

            "hypothesis":
                (
                    "Higher frozen PC3 scores are associated "
                    "with lower epithelial compartment fraction."
                ),

            "numerator":
                "mask == 2",

            "denominator":
                "mask in {1,2}",

            "unlabeled_pixels":
                "excluded",
        },


        # =====================================================================
        # Supportive endpoint
        # =====================================================================

        "supportive_endpoint": {

            "feature":
                (
                    "non_epithelial_fraction_"
                    "among_labeled_pixels"
                ),

            "expected_pc3_direction":
                "positive",

            "relationship_to_primary":
                (
                    "Exact complement of the primary endpoint "
                    "when only labels 1 and 2 enter the denominator."
                ),

            "multiplicity_role":
                (
                    "Supportive descriptive endpoint only; "
                    "not treated as an independent biological "
                    "hypothesis."
                ),
        },


        # =====================================================================
        # Sampling
        # =====================================================================

        "formal_sampling": {

            "tile_population":
                (
                    "All Virchow2 tiles with finite frozen PC3 "
                    "scores and sufficient valid labeled-mask "
                    "coverage."
                ),


            "continuous_analysis": {

                "role":
                    "primary_external_association",

                "definition":
                    (
                        "Within each slide, quantify the "
                        "monotonic association between frozen "
                        "tile-level PC3 and epithelial fraction."
                    ),
            },


            "pc3_extreme_analysis": {

                "role":
                    "supportive_effect_size_summary",

                "definition":
                    (
                        "Within each slide, compare low and high "
                        "PC3 tails defined by the slide-specific "
                        "10th and 90th percentiles."
                    ),

                "low":
                    (
                        "<= within-slide PC3 "
                        "10th percentile"
                    ),

                "high":
                    (
                        ">= within-slide PC3 "
                        "90th percentile"
                    ),
            },
        },


        # =====================================================================
        # Aggregation and inference
        # =====================================================================

        "aggregation_and_inference": {

            "independent_unit":
                "slide",

            "pseudoreplication_rule":
                (
                    "Tile-level observations are repeated image "
                    "samples nested within slides and must not be "
                    "treated as independent external-cohort "
                    "observations for formal inference."
                ),

            "continuous_slide_effect":
                "Spearman rho",

            "cohort_summary":
                (
                    "median slide-level "
                    "Spearman rho"
                ),

            "formal_null":
                (
                    "slide-level PC3 association "
                    "centered at zero"
                ),

            "extreme_effect_summary":
                (
                    "within-slide high-minus-low "
                    "epithelial-fraction difference "
                    "summarized across slides"
                ),

            "confidence_interval":
                "bootstrap over slides",

            "two_sided_testing":
                True,
        },


        # =====================================================================
        # QC
        # =====================================================================

        "measurement_qc": {

            "require_finite_pc3":
                True,

            "require_mask_geometry_compatible":
                True,

            "exclude_unlabeled_pixels":
                True,

            "require_positive_labeled_pixel_count":
                True,

            "formal_minimum_labeled_fraction":
                FORMAL_MIN_LABELED_FRACTION,

            "note":
                (
                    "Tiles with less than 10% labeled-mask "
                    "coverage are excluded from formal "
                    "compartment measurement. This threshold "
                    "is frozen before Step-16b measurement."
                ),
        },


        # =====================================================================
        # Claim boundary
        # =====================================================================

        "claim_boundary": {

            "supported": [

                (
                    "Frozen PC3 is associated with epithelial "
                    "versus non-epithelial labeled compartment "
                    "composition in the external PESO cohort."
                ),

                (
                    "The direction of the external compartment "
                    "association is consistent or inconsistent "
                    "with the internal PC3 morphology "
                    "interpretation."
                ),

                (
                    "The association is robust or not robust "
                    "to measurement using the corrected-mask "
                    "sensitivity subset."
                ),
            ],

            "not_supported_by_this_analysis_alone": [

                "single-cell nuclear morphology",

                "cell-type-specific nuclear density",

                "gold-standard stromal segmentation",

                "causal biological interpretation",

                "clinical outcome prediction",
            ],
        },
    }


# =============================================================================
# PROVENANCE
# =============================================================================

def build_provenance():

    rows = [

        {
            "evidence_source":
                (
                    "internal discovery PC3 "
                    "quantitative morphology"
                ),

            "role":
                "prior biological hypothesis",

            "status":
                (
                    "previously observed before "
                    "formal PESO re-analysis"
                ),

            "formal_step16_use":
                (
                    "Defines the directional external "
                    "hypothesis; internal RGB-derived "
                    "measurements are not reused as PESO "
                    "compartment labels."
                ),
        },


        {
            "evidence_source":
                (
                    "PESO standard training masks"
                ),

            "role":
                (
                    "primary external whole-slide "
                    "epithelial-compartment annotation"
                ),

            "status":
                "formal Step-16 primary input",

            "formal_step16_use":
                (
                    "Used as the primary mask source "
                    "for all 62 PESO training slides."
                ),
        },


        {
            "evidence_source":
                (
                    "PESO corrected training masks"
                ),

            "role":
                (
                    "corrected-mask sensitivity "
                    "reference subset"
                ),

            "status":
                (
                    "formal Step-16 sensitivity input"
                ),

            "formal_step16_use":
                (
                    "Used only for sensitivity analysis "
                    "on the 25 slides for which a corrected "
                    "mask is available."
                ),
        },


        {
            "evidence_source":
                (
                    "PESO official test-set "
                    "ground-truth label documentation"
                ),

            "role":
                "label-semantics provenance",

            "status":
                "external documentation",

            "formal_step16_use":
                (
                    "Provides the documented categorical "
                    "mapping 0=unlabeled, 1=background, "
                    "2=epithelial tissue. The same mapping "
                    "is adopted for training masks after "
                    "confirming identical {0,1,2} value support."
                ),
        },


        {
            "evidence_source":
                (
                    "discovery-cohort frozen PCA model"
                ),

            "role":
                "latent-space definition",

            "status":
                (
                    "frozen before PESO "
                    "external analysis"
                ),

            "formal_step16_use":
                (
                    "PESO embeddings are projected "
                    "into this frozen space; PCA is "
                    "not refit externally."
                ),
        },
    ]


    for path in (
        PRIOR_EXPLORATORY_FILES
    ):

        rows.append(
            {
                "evidence_source":
                    str(
                        path
                    ),

                "role":
                    (
                        "prior exploratory "
                        "PESO analysis"
                    ),

                "status":
                    (
                        "available"
                        if path.exists()
                        else "not_found"
                    ),

                "formal_step16_use":
                    (
                        "Recorded for provenance only; "
                        "formal Step-16 results must be "
                        "regenerated from the frozen "
                        "Step-16 specification."
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# HASH TABLE
# =============================================================================

def build_hash_table(
    manifest,
):

    """
    Hash small/high-value provenance files only.

    We intentionally do NOT SHA256 every large mask, coordinate,
    or PCA array because doing so would add substantial filesystem
    I/O without materially improving the frozen analysis specification.

    Large input files remain identified by:
        absolute path
        file size
        modification time

    The frozen PCA model and metadata JSONs are SHA256 hashed.
    """

    rows = []


    # -------------------------------------------------------------------------
    # Frozen discovery PCA model
    # -------------------------------------------------------------------------

    path = DISCOVERY_PCA_MODEL

    stat = path.stat()

    rows.append(
        {
            "path":
                str(
                    path
                ),

            "file_role":
                "frozen_discovery_pca_model",

            "size_bytes":
                int(
                    stat.st_size
                ),

            "mtime_ns":
                int(
                    stat.st_mtime_ns
                ),

            "sha256":
                sha256_file(
                    path
                ),
        }
    )


    # -------------------------------------------------------------------------
    # Metadata JSON files
    # -------------------------------------------------------------------------

    for value in (
        manifest[
            "meta_path"
        ]
        .dropna()
        .unique()
    ):

        path = Path(
            str(
                value
            )
        )

        stat = path.stat()

        rows.append(
            {
                "path":
                    str(
                        path
                    ),

                "file_role":
                    "PESO_embedding_metadata",

                "size_bytes":
                    int(
                        stat.st_size
                    ),

                "mtime_ns":
                    int(
                        stat.st_mtime_ns
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )


    # -------------------------------------------------------------------------
    # Large data arrays/masks:
    # record provenance without expensive hashing.
    # -------------------------------------------------------------------------

    large_file_columns = {

        "coords_path":
            "Virchow2_tile_coordinates",

        "pca_path":
            "frozen_PCA_projection",

        "standard_mask_path":
            "PESO_standard_training_mask",

        "corrected_mask_path":
            "PESO_corrected_mask",
    }

    seen = set()

    for (
        column,
        role,
    ) in (
        large_file_columns.items()
    ):

        for value in (
            manifest[
                column
            ]
            .dropna()
        ):

            value = str(
                value
            ).strip()

            if not value:
                continue

            path = Path(
                value
            )

            resolved = str(
                path.resolve()
            )

            if resolved in seen:
                continue

            seen.add(
                resolved
            )

            if not path.exists():

                rows.append(
                    {
                        "path":
                            str(
                                path
                            ),

                        "file_role":
                            role,

                        "size_bytes":
                            np.nan,

                        "mtime_ns":
                            np.nan,

                        "sha256":
                            None,
                    }
                )

                continue

            stat = path.stat()

            rows.append(
                {
                    "path":
                        str(
                            path
                        ),

                    "file_role":
                        role,

                    "size_bytes":
                        int(
                            stat.st_size
                        ),

                    "mtime_ns":
                        int(
                            stat.st_mtime_ns
                        ),

                    "sha256":
                        None,
                }
            )


    # -------------------------------------------------------------------------
    # Prior exploratory scripts
    # -------------------------------------------------------------------------

    for path in (
        PRIOR_EXPLORATORY_FILES
    ):

        if not path.exists():
            continue

        stat = path.stat()

        rows.append(
            {
                "path":
                    str(
                        path
                    ),

                "file_role":
                    (
                        "prior_exploratory_script"
                    ),

                "size_bytes":
                    int(
                        stat.st_size
                    ),

                "mtime_ns":
                    int(
                        stat.st_mtime_ns
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# README
# =============================================================================

def build_readme(
    manifest,
    geometry,
):

    n_total = int(
        len(
            manifest
        )
    )

    n_corrected = int(
        manifest[
            "corrected_mask_available"
        ].sum()
    )

    n_standard_geometry_ok = int(
        geometry[
            "standard_geometry_compatible"
        ]
        .fillna(
            False
        )
        .sum()
    )

    corrected_geometry = (
        geometry.loc[
            geometry[
                "corrected_mask_available"
            ]
        ]
    )

    n_corrected_geometry_ok = int(
        corrected_geometry[
            "corrected_geometry_compatible"
        ]
        .fillna(
            False
        )
        .sum()
    )

    text = f"""
STEP 16A — PESO PC3 EXTERNAL COMPARTMENT SPECIFICATION
======================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA_VERSION}


PURPOSE
-------

Freeze the formal PESO external-validation specification for testing
whether discovery-cohort frozen PC3 is associated with epithelial
compartment composition.

This step does NOT calculate the PC3-compartment association.


COHORT
------

PESO training WSIs:

    {n_total}

Primary whole-slide analysis:

    62 / 62 standard PESO training masks

Corrected-mask sensitivity subset:

    {n_corrected} slides


IMPORTANT MASK-SOURCE DECISION
------------------------------

The PRIMARY analysis does NOT mix corrected and standard masks.

Primary:

    all 62 standard training masks

Sensitivity:

    25 corrected masks

This design preserves a homogeneous reference source across the complete
primary external cohort.

Corrected masks are used only to determine whether the association remains
consistent under the alternative corrected-mask measurement.


FROZEN PCA
----------

PESO PCA scores must come from projection into the original
discovery-cohort frozen PCA space.

PCA must NOT be refit on PESO.

Target axis:

    PC3


MASK LABELS
-----------

Observed standard and corrected training masks use:

    {{0, 1, 2}}

The official PESO test-set ground-truth documentation defines:

    0 = unlabeled
    1 = background
    2 = epithelial tissue

Because the PESO training masks use the same categorical value support and
are epithelial segmentation masks, this same mapping is adopted for the
formal Step-16 training-mask analysis.

The provenance distinction is explicitly retained:

    the test-set mapping is directly documented;
    the identical mapping is applied to the training masks after
    empirical confirmation of the same encoding.


PRIMARY ENDPOINT
----------------

Epithelial fraction among labeled pixels:

        N(mask == 2)
    -------------------
      N(mask in {{1,2}})

Pixels with label 0 are excluded.


DO NOT USE
----------

    mask > 0

as an epithelial definition.

That would incorrectly combine labels 1 and 2.


COMPLEMENTARY ENDPOINT
----------------------

Non-epithelial labeled fraction:

        N(mask == 1)
    -------------------
      N(mask in {{1,2}})

This is exactly:

    1 - epithelial fraction


CLAIM BOUNDARY
--------------

Label 1 is treated formally as the non-epithelial labeled compartment.

It must NOT automatically be described as a gold-standard stromal
segmentation without independent validation of that interpretation.


MASK GEOMETRY
-------------

Standard masks:

    approximately 1:1 with H&E

Compatible standard-mask geometries:

    {n_standard_geometry_ok}/{n_total}


Corrected masks:

    approximately 2x H&E resolution

Compatible corrected-mask geometries:

    {n_corrected_geometry_ok}/{n_corrected}


Per-slide mapping uses:

    scale_x = mask_width  / H&E_width
    scale_y = mask_height / H&E_height


For H&E tile:

    [x, x + patch_size)
    [y, y + patch_size)

corresponding mask rectangle:

    x0 = round(x * scale_x)
    y0 = round(y * scale_y)

    x1 = round((x + patch_size) * scale_x)
    y1 = round((y + patch_size) * scale_y)


No continuous interpolation of categorical mask labels is permitted.


PRIMARY EXTERNAL HYPOTHESIS
---------------------------

Higher frozen PC3 is associated with lower epithelial fraction.

Expected direction:

    PC3 higher
        |
        v
    epithelial fraction lower


FORMAL STATISTICAL UNIT
-----------------------

Slide.

Tiles are repeated measurements nested within slides.

Tile-level observations must NOT be treated as independent
external-cohort samples for formal inference.


PRIMARY CONTINUOUS ANALYSIS
---------------------------

Within each of the 62 primary slides:

    tile-level frozen PC3
              versus
    epithelial fraction

using:

    Spearman correlation


Cohort-level summary:

    distribution of slide-level Spearman rho
    median slide-level rho
    slide-level bootstrap confidence interval


SUPPORTIVE EXTREME ANALYSIS
---------------------------

Within each slide:

    PC3 <= 10th percentile

versus

    PC3 >= 90th percentile


This is a supportive effect-size analysis.

It does NOT replace the continuous-PC3 primary analysis.


CORRECTED-MASK SENSITIVITY
--------------------------

For the 25 corrected-mask slides:

    repeat the exact same analysis
    using corrected mask coordinates

Then compare:

    direction
    effect magnitude
    cohort-level consistency

against the corresponding standard-mask analysis.


FORMAL TILE QC
--------------

A tile requires:

    finite frozen PC3

    compatible mask geometry

    positive labeled-pixel count

    labeled fraction >= {FORMAL_MIN_LABELED_FRACTION:.2f}

Label 0 pixels are excluded from the endpoint denominator.


PROVENANCE
----------

Prior exploratory PESO analyses existed before this formal Step-16
re-analysis.

Therefore this analysis is NOT described as an independent
preregistration.

Appropriate wording:

    Endpoint direction and biological hypotheses were informed by prior
    exploratory PESO analyses; the formal cohort, frozen-PC projection,
    primary mask source, corrected-mask sensitivity role, endpoint
    definition, geometry handling, measurement QC, aggregation strategy,
    and inferential procedures were frozen before the formal Step-16
    re-analysis.


TEST SET
--------

The PESO 40-slide test cohort is intentionally excluded from this
whole-slide epithelial-compartment analysis.

It is reserved for separate region-level external validation.


NEXT STEP
---------

Step 16b should perform measurement and QC only.

It must read this frozen manifest and analysis plan.

It must NOT silently redefine:

    cohort
    primary mask source
    corrected-mask role
    mask labels
    endpoint denominator
    mask scale factors
    labeled-pixel threshold
    PC3 direction
    statistical unit
"""

    return (
        text.strip()
        + "\n"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print(
        "=" * 120
    )

    print(
        "STEP 16A: FREEZE PESO PC3 "
        "EXTERNAL COMPARTMENT VALIDATION"
    )

    print(
        "=" * 120
    )

    print(
        f"[VERSION]       "
        f"{VERSION}"
    )

    print(
        f"[SCHEMA]        "
        f"{SCHEMA_VERSION}"
    )

    print(
        f"[PESO EMB]      "
        f"{EMB_DIR}"
    )

    print(
        f"[PESO PCA]      "
        f"{PCA_DIR}"
    )

    print(
        f"[STANDARD MASK] "
        f"{STANDARD_MASK_ROOTS[0]}"
    )

    print(
        f"[CORRECTED]     "
        f"{CORRECTED_MASK_ROOTS[0]}"
    )

    print(
        f"[OUTPUT]        "
        f"{OUT_DIR}"
    )

    print(
        f"[OVERWRITE]     "
        f"{args.overwrite_freeze}"
    )


    # =========================================================================
    # Overwrite protection
    # =========================================================================

    existing = [
        path
        for path in CORE_OUTPUTS
        if path.exists()
    ]

    if (
        existing
        and
        not args.overwrite_freeze
    ):

        raise RuntimeError(
            "Frozen Step-16a outputs "
            "already exist.\n"
            "Refusing to overwrite:\n"
            +
            "\n".join(
                f"  - {path}"
                for path in existing
            )
            +
            "\n\nUse --overwrite-freeze "
            "only if you intentionally "
            "want to replace the frozen "
            "specification."
        )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================================
    # Frozen PCA
    # =========================================================================

    if not (
        DISCOVERY_PCA_MODEL.exists()
    ):

        raise FileNotFoundError(
            "Missing discovery frozen "
            "PCA model:\n"
            f"{DISCOVERY_PCA_MODEL}"
        )


    # =========================================================================
    # Training cohort
    # =========================================================================

    training = (
        load_training_slides()
    )

    print()

    print(
        "=" * 120
    )

    print(
        "PESO TRAINING COHORT"
    )

    print(
        "=" * 120
    )

    n_training = int(
        training[
            "slide_id"
        ].nunique()
    )

    print(
        f"Training slides: "
        f"{n_training}"
    )

    if n_training != 62:

        raise RuntimeError(
            "Expected exactly "
            "62 PESO training "
            f"slides, found "
            f"{n_training}."
        )


    bad_patch = (
        training.loc[
            training[
                "patch_size"
            ]
            !=
            PATCH_SIZE_EXPECTED
        ]
    )

    if not bad_patch.empty:

        raise RuntimeError(
            "Unexpected patch "
            "sizes detected:\n"
            +
            bad_patch[
                [
                    "slide_id",
                    "patch_size",
                ]
            ].to_string(
                index=False
            )
        )


    # =========================================================================
    # Mask manifest
    # =========================================================================

    (
        manifest,
        geometry,
    ) = build_mask_manifest(
        training
    )


    if len(
        manifest
    ) != 62:

        raise RuntimeError(
            "Expected 62 manifest "
            f"rows, found "
            f"{len(manifest)}."
        )


    n_corrected = int(
        manifest[
            "corrected_mask_available"
        ].sum()
    )


    print()

    print(
        "=" * 120
    )

    print(
        "FROZEN MASK SOURCE DESIGN"
    )

    print(
        "=" * 120
    )

    print(
        "Primary standard masks   : "
        f"{len(manifest)}"
    )

    print(
        "Corrected sensitivity    : "
        f"{n_corrected}"
    )

    print(
        "Hybrid primary masks     : "
        "0"
    )


    if n_corrected != 25:

        raise RuntimeError(
            "Expected 25 corrected "
            "masks based on audited "
            "PESO training cohort, "
            f"found {n_corrected}."
        )


    # =========================================================================
    # Geometry
    # =========================================================================

    print()

    print(
        "=" * 120
    )

    print(
        "GEOMETRY AUDIT"
    )

    print(
        "=" * 120
    )


    print(
        "Standard scale_x range:",
        f"{geometry['standard_scale_x'].min():.6f}",
        "to",
        f"{geometry['standard_scale_x'].max():.6f}",
    )

    print(
        "Standard scale_y range:",
        f"{geometry['standard_scale_y'].min():.6f}",
        "to",
        f"{geometry['standard_scale_y'].max():.6f}",
    )


    corrected_geometry = (
        geometry.loc[
            geometry[
                "corrected_mask_available"
            ]
        ]
    )


    if not (
        corrected_geometry.empty
    ):

        print(
            "Corrected scale_x range:",
            f"{corrected_geometry['corrected_scale_x'].min():.6f}",
            "to",
            f"{corrected_geometry['corrected_scale_x'].max():.6f}",
        )

        print(
            "Corrected scale_y range:",
            f"{corrected_geometry['corrected_scale_y'].min():.6f}",
            "to",
            f"{corrected_geometry['corrected_scale_y'].max():.6f}",
        )


    # =========================================================================
    # Frozen specification objects
    # =========================================================================

    labels = (
        build_label_specification()
    )

    plan = (
        build_analysis_plan(
            manifest
        )
    )

    provenance = (
        build_provenance()
    )


    # =========================================================================
    # Cohort summary
    # =========================================================================

    cohort_summary = (
        pd.DataFrame(
            [
                {
                    "metric":
                        "n_training_slides",

                    "value":
                        len(
                            manifest
                        ),
                },

                {
                    "metric":
                        "n_primary_standard_masks",

                    "value":
                        len(
                            manifest
                        ),
                },

                {
                    "metric":
                        "n_corrected_mask_sensitivity_slides",

                    "value":
                        n_corrected,
                },

                {
                    "metric":
                        "n_hybrid_primary_masks",

                    "value":
                        0,
                },

                {
                    "metric":
                        "n_frozen_pca_projections",

                    "value":
                        int(
                            manifest[
                                "pca_path"
                            ]
                            .astype(
                                bool
                            )
                            .sum()
                        ),
                },
            ]
        )
    )


    # =========================================================================
    # Save specification
    # =========================================================================

    manifest.to_csv(
        OUT_MANIFEST,
        index=False,
    )

    geometry.to_csv(
        OUT_GEOMETRY,
        index=False,
    )

    cohort_summary.to_csv(
        OUT_COHORT,
        index=False,
    )

    provenance.to_csv(
        OUT_PROVENANCE,
        index=False,
    )

    write_json(
        OUT_PLAN,
        plan,
    )

    write_json(
        OUT_LABELS,
        labels,
    )


    # =========================================================================
    # Provenance / hashes
    # =========================================================================

    hashes = (
        build_hash_table(
            manifest
        )
    )

    hashes.to_csv(
        OUT_HASHES,
        index=False,
    )


    # =========================================================================
    # README
    # =========================================================================

    readme = (
        build_readme(
            manifest,
            geometry,
        )
    )

    with open(
        OUT_README,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            readme
        )


    # =========================================================================
    # Metadata
    # =========================================================================

    metadata = {

        "step":
            "16a",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),

        "formal_status":
            "frozen",

        "target_axis":
            PC_NAME,

        "target_axis_zero_based_index":
            PC_INDEX_ZERO_BASED,

        "primary_mask_source":
            (
                "standard PESO "
                "training mask"
            ),

        "primary_mask_n_slides":
            62,

        "corrected_mask_role":
            "sensitivity_only",

        "corrected_mask_n_slides":
            n_corrected,

        "hybrid_primary_masking":
            False,

        "formal_minimum_labeled_fraction":
            FORMAL_MIN_LABELED_FRACTION,

        "discovery_pca_model":
            str(
                DISCOVERY_PCA_MODEL
            ),

        "discovery_pca_model_sha256":
            sha256_file(
                DISCOVERY_PCA_MODEL
            ),


        "outputs": {

            "manifest":
                str(
                    OUT_MANIFEST
                ),

            "analysis_plan":
                str(
                    OUT_PLAN
                ),

            "label_specification":
                str(
                    OUT_LABELS
                ),

            "geometry_audit":
                str(
                    OUT_GEOMETRY
                ),

            "cohort_summary":
                str(
                    OUT_COHORT
                ),

            "provenance":
                str(
                    OUT_PROVENANCE
                ),

            "source_hashes":
                str(
                    OUT_HASHES
                ),

            "readme":
                str(
                    OUT_README
                ),
        },


        "output_sha256": {

            "manifest":
                sha256_file(
                    OUT_MANIFEST
                ),

            "analysis_plan":
                sha256_file(
                    OUT_PLAN
                ),

            "label_specification":
                sha256_file(
                    OUT_LABELS
                ),

            "geometry_audit":
                sha256_file(
                    OUT_GEOMETRY
                ),

            "cohort_summary":
                sha256_file(
                    OUT_COHORT
                ),

            "provenance":
                sha256_file(
                    OUT_PROVENANCE
                ),

            "source_hashes":
                sha256_file(
                    OUT_HASHES
                ),

            "readme":
                sha256_file(
                    OUT_README
                ),
        },


        "label_semantics_provenance":
            (
                "The official PESO test-set "
                "ground-truth documentation "
                "defines 0=unlabeled, "
                "1=background, and "
                "2=epithelial tissue. "
                "The same mapping is formally "
                "applied to PESO training masks "
                "after confirming identical "
                "{0,1,2} categorical encoding."
            ),


        "provenance_statement":
            (
                "Prior exploratory PESO analyses "
                "informed the biological hypothesis. "
                "Formal Step-16 cohort definition, "
                "frozen-PC projection, homogeneous "
                "primary mask source, corrected-mask "
                "sensitivity role, geometry handling, "
                "endpoint definition, measurement QC, "
                "aggregation strategy, and statistical "
                "procedures were frozen before the "
                "formal re-analysis."
            ),
    }


    write_json(
        OUT_METADATA,
        metadata,
    )


    # =========================================================================
    # Final validation
    # =========================================================================

    n_primary = int(
        manifest[
            "primary_mask_type"
        ]
        .eq(
            "standard"
        )
        .sum()
    )

    n_geometry_ok = int(
        manifest[
            "standard_geometry_compatible"
        ]
        .fillna(
            False
        )
        .sum()
    )


    corrected_geometry_ok = int(
        manifest.loc[
            manifest[
                "corrected_mask_available"
            ],
            "corrected_geometry_compatible",
        ]
        .fillna(
            False
        )
        .sum()
    )


    print()

    print(
        "=" * 120
    )

    print(
        "STEP 16A FREEZE SUMMARY"
    )

    print(
        "=" * 120
    )


    print(
        f"Training WSIs                 : "
        f"{len(manifest)}"
    )

    print(
        f"Primary standard masks       : "
        f"{n_primary}"
    )

    print(
        f"Corrected sensitivity masks  : "
        f"{n_corrected}"
    )

    print(
        f"Hybrid primary masks         : "
        f"0"
    )

    print(
        f"Standard geometry compatible : "
        f"{n_geometry_ok}/62"
    )

    print(
        f"Corrected geometry compatible: "
        f"{corrected_geometry_ok}/"
        f"{n_corrected}"
    )

    print(
        f"Target                        : "
        f"{PC_NAME}"
    )

    print(
        "Primary endpoint              : "
        "epithelial_fraction_among_labeled_pixels"
    )

    print(
        "Expected direction            : "
        "PC3 higher -> epithelial fraction lower"
    )

    print(
        "Formal independent unit       : "
        "slide"
    )


    # =========================================================================
    # Hard final checks
    # =========================================================================

    if n_primary != 62:

        raise RuntimeError(
            "Primary mask freeze failed: "
            f"expected 62 standard masks, "
            f"found {n_primary}."
        )


    if n_corrected != 25:

        raise RuntimeError(
            "Corrected sensitivity "
            "freeze failed: expected "
            f"25 masks, found "
            f"{n_corrected}."
        )


    if n_geometry_ok != 62:

        raise RuntimeError(
            "Not all standard-mask "
            "geometries are compatible."
        )


    if corrected_geometry_ok != 25:

        raise RuntimeError(
            "Not all corrected-mask "
            "geometries are compatible."
        )


    print()

    print(
        "[OK] Saved:"
    )

    for path in CORE_OUTPUTS:

        print(
            f"  {path}"
        )


    print()

    print(
        "[DONE] Step-16a PESO PC3 external "
        "compartment specification frozen."
    )

    print()

    print(
        "[FROZEN DESIGN]"
    )

    print(
        "  PRIMARY     : "
        "62/62 standard training masks"
    )

    print(
        "  SENSITIVITY : "
        "25 corrected-mask slides"
    )

    print(
        "  HYBRID      : "
        "not permitted"
    )


if __name__ == "__main__":
    main()