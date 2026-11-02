#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
16d_generate_peso_pc3_external_validation_figures.py

Step 16d v1.1.0
================

Reporting / visualization only for the formal PESO PC3 external validation.

EVIDENCE CHAIN
--------------
16a:
    frozen cohort / mask / geometry / labels / endpoint / QC / inference plan

16b:
    measurement only

16c:
    formal slide-level statistical inference

16d:
    visualization / reporting only

ABSOLUTE RULES
--------------
16d MUST NOT:

- recompute epithelial fraction from raw masks;
- redefine mask labels;
- redefine mask scale;
- redefine labeled-fraction QC;
- redefine PC3;
- test another PC;
- calculate new biological association statistics;
- introduce pooled tile-level inference;
- optimize thresholds;
- manually choose "best-looking" slides;
- manually choose "best-looking" ROIs;
- perform hybrid-mask analysis.

Raw H&E is accessed ONLY for visualization.

FIGURES
-------

Panel A
-------
Primary 62-slide Spearman-rho distribution.

Input:
    Step-16c primary slide association table.

Displays:
    ordered slide-level rho values
    cohort median rho
    frozen bootstrap 95% CI
    rho = 0 reference

No statistic is recomputed.

Panel B
-------
Post-QC common-tile mask-source robustness.

Input:
    Step-16c common-tile sensitivity table.

Displays:
    x = standard-mask rho on common physical tiles
    y = corrected-mask rho on same tiles
    identity line

No new mask comparison test is introduced.

Panel C
-------
Spatial external-validation visualization.

Representative slides are selected algorithmically from the PRIMARY
62-slide rho distribution at approximately:

    25th percentile
    50th percentile
    75th percentile

The nearest observed slide-level rho is selected for each target.

For each representative slide:

    H&E overview
    frozen PC3 heatmap
    frozen epithelial-fraction heatmap

Both heatmaps use exactly the Step-16b STANDARD-mask QC-pass tile support.

PC3 plotting scale:
    one cohort-wide symmetric display range derived from the pooled
    primary QC-pass PC3 values using 1st / 99th percentiles.

This is a VISUALIZATION scale only and does NOT change any endpoint,
threshold, tile inclusion rule, or inferential result.

Epithelial-fraction plotting scale:
    fixed [0, 1].

Representative H&E ROIs:
    frozen PC3-low tail  <= Step-16c within-slide q10
    frozen PC3-high tail >= Step-16c within-slide q90

Within each frozen tail, five ROIs are selected deterministically at
5%, 25%, 50%, 75%, and 95% of the score-ranked tail.

This prevents manual cherry-picking.

OUTPUT
------
results/16d_peso_pc3_external_validation_figures/

    panel_A_primary_slide_spearman.png
    panel_A_primary_slide_spearman.pdf

    panel_B_common_tile_mask_robustness.png
    panel_B_common_tile_mask_robustness.pdf

    panel_C_spatial_q25_<slide>.png
    panel_C_spatial_q25_<slide>.pdf

    panel_C_spatial_q50_<slide>.png
    panel_C_spatial_q50_<slide>.pdf

    panel_C_spatial_q75_<slide>.png
    panel_C_spatial_q75_<slide>.pdf

    representative_slide_selection.csv
    representative_roi_selection.csv

    panel_A_source_data.csv
    panel_B_source_data.csv

    visualization_metadata.json
    README.txt
"""

from __future__ import annotations

import argparse
import hashlib
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

from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import TwoSlopeNorm, Normalize


# =============================================================================
# OPTIONAL OPENSLIDE
# =============================================================================

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except Exception:
    openslide = None
    OPENSLIDE_AVAILABLE = False


Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.2.0"

SCHEMA_VERSION = (
    "16d-peso-pc3-external-visualization-schema-1.2"
)


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


# -----------------------------------------------------------------------------
# Step 16a
# -----------------------------------------------------------------------------

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

STEP16A_METADATA = (
    STEP16A_DIR
    / "analysis_metadata.json"
)


# -----------------------------------------------------------------------------
# Step 16b
# -----------------------------------------------------------------------------

STEP16B_DIR = (
    RESULTS_ROOT
    / "16b_peso_pc3_external_compartment_measurements"
)

STEP16B_MANIFEST = (
    STEP16B_DIR
    / "peso_pc3_external_measurement_manifest.csv"
)

STEP16B_METADATA = (
    STEP16B_DIR
    / "analysis_metadata.json"
)


# -----------------------------------------------------------------------------
# Step 16c
# -----------------------------------------------------------------------------

STEP16C_DIR = (
    RESULTS_ROOT
    / "16c_peso_pc3_external_compartment_association"
)

STEP16C_PRIMARY_SLIDES = (
    STEP16C_DIR
    / "peso_pc3_primary_slide_associations.csv"
)

STEP16C_PRIMARY_COHORT = (
    STEP16C_DIR
    / "peso_pc3_primary_cohort_summary.csv"
)

STEP16C_COMMON_TILE = (
    STEP16C_DIR
    / "peso_pc3_common_tile_mask_source_sensitivity.csv"
)

STEP16C_COMMON_TILE_SUMMARY = (
    STEP16C_DIR
    / "peso_pc3_common_tile_mask_source_summary.csv"
)

STEP16C_METADATA = (
    STEP16C_DIR
    / "analysis_metadata.json"
)


# -----------------------------------------------------------------------------
# Step 16d outputs
# -----------------------------------------------------------------------------

OUT_DIR = (
    RESULTS_ROOT
    / "16d_peso_pc3_external_validation_figures"
)

OUT_PANEL_A_PNG = (
    OUT_DIR
    / "panel_A_primary_slide_spearman.png"
)

OUT_PANEL_A_PDF = (
    OUT_DIR
    / "panel_A_primary_slide_spearman.pdf"
)

OUT_PANEL_B_PNG = (
    OUT_DIR
    / "panel_B_common_tile_mask_robustness.png"
)

OUT_PANEL_B_PDF = (
    OUT_DIR
    / "panel_B_common_tile_mask_robustness.pdf"
)

OUT_REP_SLIDES = (
    OUT_DIR
    / "representative_slide_selection.csv"
)

OUT_REP_ROIS = (
    OUT_DIR
    / "representative_roi_selection.csv"
)

OUT_PANEL_A_DATA = (
    OUT_DIR
    / "panel_A_source_data.csv"
)

OUT_PANEL_B_DATA = (
    OUT_DIR
    / "panel_B_source_data.csv"
)

OUT_METADATA = (
    OUT_DIR
    / "visualization_metadata.json"
)

OUT_README = (
    OUT_DIR
    / "README.txt"
)

CORE_OUTPUTS = [
    OUT_PANEL_A_PNG,
    OUT_PANEL_A_PDF,
    OUT_PANEL_B_PNG,
    OUT_PANEL_B_PDF,
    OUT_REP_SLIDES,
    OUT_REP_ROIS,
    OUT_PANEL_A_DATA,
    OUT_PANEL_B_DATA,
    OUT_METADATA,
    OUT_README,
]


# =============================================================================
# EXPECTED FORMAL DESIGN
# =============================================================================

EXPECTED_PRIMARY_SLIDES = 62

EXPECTED_CORRECTED_SLIDES = 25

EXPECTED_PC_NAME = "PC3"

PC3_ZERO_BASED_INDEX = 2


# =============================================================================
# VISUALIZATION SETTINGS ONLY
# =============================================================================

DPI = 300

WSI_THUMBNAIL_MAX_SIZE = 1800

# Visualization-only crop around the spatial support used by BOTH heatmaps.
# This removes large scanner-background margins from Panel C while preserving
# the exact same PC3 / epithelial measurement support.
SPATIAL_DISPLAY_PADDING_TILES = 2

PC3_CMAP = "RdBu_r"

EPITHELIAL_CMAP = "viridis"

PC3_DISPLAY_LOWER_PERCENTILE = 1.0

PC3_DISPLAY_UPPER_PERCENTILE = 99.0

REPRESENTATIVE_RHO_QUANTILES = [
    0.25,
    0.50,
    0.75,
]

REPRESENTATIVE_QUANTILE_LABELS = [
    "q25",
    "q50",
    "q75",
]

ROI_WITHIN_TAIL_POSITIONS = [
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
]

N_ROI_PER_TAIL = len(
    ROI_WITHIN_TAIL_POSITIONS
)


# =============================================================================
# FIGURE STYLE
# =============================================================================

BG = "#FFFFFF"

TEXT = "#111111"

MUTED = "#555555"

GRID = "#E6E6E6"

ZERO_LINE = "#555555"

PRIMARY_POINT = "#255F85"

MEDIAN_LINE = "#A23B72"

CI_COLOR = "#D9E6EF"

COMMON_POINT = "#2E6F9E"

IDENTITY_LINE = "#555555"


plt.rcParams.update(
    {
        "font.family":
            "DejaVu Sans",

        "font.size":
            12,

        "axes.titlesize":
            14,

        "axes.labelsize":
            13,

        "xtick.labelsize":
            10,

        "ytick.labelsize":
            10,

        "axes.spines.top":
            False,

        "axes.spines.right":
            False,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,

        "savefig.dpi":
            DPI,

        "savefig.facecolor":
            BG,

        "figure.facecolor":
            BG,
    }
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow intentional replacement "
            "of existing Step-16d outputs."
        ),
    )

    return parser.parse_args()


# =============================================================================
# HELPERS
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


def guard_outputs(
    overwrite: bool,
):

    existing = [
        path
        for path in CORE_OUTPUTS
        if path.exists()
    ]

    spatial_existing = list(
        OUT_DIR.glob(
            "panel_C_spatial_*.png"
        )
    ) + list(
        OUT_DIR.glob(
            "panel_C_spatial_*.pdf"
        )
    )

    existing.extend(
        spatial_existing
    )

    if (
        existing
        and
        not overwrite
    ):

        raise RuntimeError(
            "Step-16d outputs already exist; "
            "refusing silent overwrite.\n"
            +
            "\n".join(
                f"  - {x}"
                for x in existing
            )
            +
            "\n\nUse --overwrite only if intentional."
        )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_inputs():

    required = [

        (
            STEP16A_MANIFEST,
            "Step-16a mask manifest",
        ),

        (
            STEP16A_PLAN,
            "Step-16a analysis plan",
        ),

        (
            STEP16A_METADATA,
            "Step-16a metadata",
        ),

        (
            STEP16B_MANIFEST,
            "Step-16b measurement manifest",
        ),

        (
            STEP16B_METADATA,
            "Step-16b metadata",
        ),

        (
            STEP16C_PRIMARY_SLIDES,
            "Step-16c primary slide association table",
        ),

        (
            STEP16C_PRIMARY_COHORT,
            "Step-16c primary cohort summary",
        ),

        (
            STEP16C_COMMON_TILE,
            "Step-16c common-tile sensitivity table",
        ),

        (
            STEP16C_COMMON_TILE_SUMMARY,
            "Step-16c common-tile summary",
        ),

        (
            STEP16C_METADATA,
            "Step-16c metadata",
        ),
    ]


    for path, label in required:

        require_file(
            path,
            label,
        )


# =============================================================================
# FORMAL CHAIN VALIDATION
# =============================================================================

def validate_formal_chain(
    meta16b: Dict,
    meta16c: Dict,
):

    if not bool(
        meta16b.get(
            "formal_full_run",
            False,
        )
    ):

        raise RuntimeError(
            "Step-16b is not marked "
            "formal_full_run=True."
        )


    if not bool(
        meta16b.get(
            "measurement_only",
            False,
        )
    ):

        raise RuntimeError(
            "Step-16b is not marked "
            "measurement_only=True."
        )


    if bool(
        meta16b.get(
            "inferential_statistics_performed",
            True,
        )
    ):

        raise RuntimeError(
            "Step-16b metadata indicates "
            "that inferential statistics "
            "were performed."
        )


    if int(
        meta16b.get(
            "n_primary_standard_slides",
            -1,
        )
    ) != EXPECTED_PRIMARY_SLIDES:

        raise RuntimeError(
            "Step-16b primary slide count "
            "is not 62."
        )


    if int(
        meta16b.get(
            "n_corrected_sensitivity_slides",
            -1,
        )
    ) != EXPECTED_CORRECTED_SLIDES:

        raise RuntimeError(
            "Step-16b corrected slide count "
            "is not 25."
        )


    if int(
        meta16c.get(
            "n_primary_slides_analyzed",
            -1,
        )
    ) != EXPECTED_PRIMARY_SLIDES:

        raise RuntimeError(
            "Step-16c primary slide count "
            "is not 62."
        )


    if int(
        meta16c.get(
            "n_corrected_slides_analyzed",
            -1,
        )
    ) != EXPECTED_CORRECTED_SLIDES:

        raise RuntimeError(
            "Step-16c corrected slide count "
            "is not 25."
        )


    if bool(
        meta16c.get(
            "tile_level_pooled_inference_performed",
            True,
        )
    ):

        raise RuntimeError(
            "Step-16c indicates pooled "
            "tile-level inference."
        )


    if bool(
        meta16c.get(
            "mask_measurement_recomputed",
            True,
        )
    ):

        raise RuntimeError(
            "Step-16c indicates mask "
            "remeasurement."
        )


# =============================================================================
# LOAD TABLES
# =============================================================================

def load_formal_tables():

    manifest16a = pd.read_csv(
        STEP16A_MANIFEST,
        low_memory=False,
    )

    manifest16b = pd.read_csv(
        STEP16B_MANIFEST,
        low_memory=False,
    )

    primary = pd.read_csv(
        STEP16C_PRIMARY_SLIDES,
        low_memory=False,
    )

    primary_summary = pd.read_csv(
        STEP16C_PRIMARY_COHORT,
        low_memory=False,
    )

    common = pd.read_csv(
        STEP16C_COMMON_TILE,
        low_memory=False,
    )

    common_summary = pd.read_csv(
        STEP16C_COMMON_TILE_SUMMARY,
        low_memory=False,
    )


    if len(
        primary
    ) != EXPECTED_PRIMARY_SLIDES:

        raise RuntimeError(
            f"Expected 62 primary rows; "
            f"found {len(primary)}."
        )


    if len(
        common
    ) != EXPECTED_CORRECTED_SLIDES:

        raise RuntimeError(
            f"Expected 25 common-tile rows; "
            f"found {len(common)}."
        )


    return (
        manifest16a,
        manifest16b,
        primary,
        primary_summary,
        common,
        common_summary,
    )


# =============================================================================
# STEP-16B STANDARD MEASUREMENT INDEX
# =============================================================================

def build_standard_measurement_index(
    manifest16b: pd.DataFrame,
):

    required = {
        "slide_id",
        "analysis_role",
        "mask_source",
        "npz_path",
    }

    missing = (
        required
        - set(
            manifest16b.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step-16b manifest missing columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )


    primary = manifest16b.loc[
        manifest16b[
            "analysis_role"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            "primary"
        )
    ].copy()


    if len(
        primary
    ) != EXPECTED_PRIMARY_SLIDES:

        raise RuntimeError(
            "Expected 62 primary Step-16b "
            "measurement rows."
        )


    index = {}

    for _, row in (
        primary.iterrows()
    ):

        slide_id = str(
            row[
                "slide_id"
            ]
        )

        path = Path(
            str(
                row[
                    "npz_path"
                ]
            )
        )

        require_file(
            path,
            f"{slide_id} standard measurement NPZ",
        )

        index[
            slide_id
        ] = path


    return index


# =============================================================================
# LOAD 16B NPZ
# =============================================================================

def load_standard_measurements(
    npz_path: Path,
):

    required = {
        "tile_idx",
        "x",
        "y",
        "pc3_score",
        "qc_pass",
        "epithelial_fraction_among_labeled_pixels",
    }


    z = np.load(
        npz_path,
        allow_pickle=False,
    )


    missing = (
        required
        - set(
            z.files
        )
    )


    if missing:

        z.close()

        raise RuntimeError(
            f"{npz_path} missing arrays: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )


    out = {
        key:
            np.asarray(
                z[
                    key
                ]
            )
        for key in required
    }


    z.close()


    lengths = {
        len(
            value
        )
        for value in out.values()
    }


    if len(
        lengths
    ) != 1:

        raise RuntimeError(
            f"Inconsistent array lengths "
            f"in {npz_path}: "
            f"{sorted(lengths)}"
        )


    return out


# =============================================================================
# PANEL A
# =============================================================================

def plot_panel_a(
    primary: pd.DataFrame,
    summary: pd.DataFrame,
):

    df = (
        primary
        .copy()
    )


    df[
        "spearman_rho"
    ] = pd.to_numeric(
        df[
            "spearman_rho"
        ],
        errors="coerce",
    )


    if (
        df[
            "spearman_rho"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Panel A contains non-finite "
            "primary Spearman rho."
        )


    df = (
        df
        .sort_values(
            "spearman_rho"
        )
        .reset_index(
            drop=True
        )
    )


    df[
        "ordered_slide_rank"
    ] = np.arange(
        1,
        len(
            df
        )
        + 1,
    )


    summary_row = (
        summary.iloc[
            0
        ]
    )


    median_rho = float(
        summary_row[
            "median_spearman_rho"
        ]
    )


    ci_low = float(
        summary_row[
            "rho_bootstrap_ci_lower"
        ]
    )


    ci_high = float(
        summary_row[
            "rho_bootstrap_ci_upper"
        ]
    )


    n_negative = int(
        summary_row[
            "n_slides_rho_negative"
        ]
    )


    fig, ax = plt.subplots(
        figsize=(
            8.6,
            5.8,
        )
    )


    ax.axhspan(
        ci_low,
        ci_high,
        color=CI_COLOR,
        alpha=0.55,
        zorder=0,
        label="95% slide-bootstrap CI",
    )


    ax.axhline(
        0,
        color=ZERO_LINE,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )


    ax.axhline(
        median_rho,
        color=MEDIAN_LINE,
        linewidth=2.0,
        zorder=2,
        label="Median slide-level ρ",
    )


    ax.scatter(
        df[
            "ordered_slide_rank"
        ],
        df[
            "spearman_rho"
        ],
        s=38,
        color=PRIMARY_POINT,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )


    ax.set_xlabel(
        "PESO slide, ordered by within-slide Spearman ρ"
    )


    ax.set_ylabel(
        "Within-slide Spearman ρ\n"
        "(PC3 vs epithelial fraction)"
    )


    ax.set_title(
        "External PC3–epithelial association across 62 PESO slides",
        fontweight="bold",
        pad=12,
    )


    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.7,
        zorder=0,
    )


    ax.set_axisbelow(
        True
    )


    annotation = (
        f"Median ρ = {median_rho:.3f}\n"
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]\n"
        f"{n_negative}/{len(df)} slides ρ < 0"
    )


    ax.text(
        0.985,
        0.965,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle":
                "round,pad=0.45",

            "facecolor":
                "white",

            "edgecolor":
                "#BBBBBB",

            "linewidth":
                0.8,
        },
    )


    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=9,
    )


    fig.tight_layout()


    fig.savefig(
        OUT_PANEL_A_PNG,
        dpi=DPI,
        bbox_inches="tight",
    )


    fig.savefig(
        OUT_PANEL_A_PDF,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    df.to_csv(
        OUT_PANEL_A_DATA,
        index=False,
    )


    print(
        f"[OK] Panel A: "
        f"{OUT_PANEL_A_PNG}"
    )


# =============================================================================
# PANEL B
# =============================================================================

def plot_panel_b(
    common: pd.DataFrame,
    summary: pd.DataFrame,
):

    df = (
        common
        .copy()
    )


    x = pd.to_numeric(
        df[
            "rho_standard_common"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    y = pd.to_numeric(
        df[
            "rho_corrected_common"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


    valid = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )


    if (
        valid.sum()
        != EXPECTED_CORRECTED_SLIDES
    ):

        raise RuntimeError(
            "Panel B does not contain "
            "25 finite paired rho values."
        )


    x = x[
        valid
    ]

    y = y[
        valid
    ]


    row = (
        summary.iloc[
            0
        ]
    )


    across_slide_rho = float(
        row[
            "spearman_across_slides_standard_vs_corrected_rho"
        ]
    )


    median_standard = float(
        row[
            "median_rho_standard_common"
        ]
    )


    median_corrected = float(
        row[
            "median_rho_corrected_common"
        ]
    )


    median_delta = float(
        row[
            "median_delta_rho_corrected_minus_standard"
        ]
    )


    minimum = float(
        min(
            np.min(
                x
            ),
            np.min(
                y
            ),
        )
    )


    maximum = float(
        max(
            np.max(
                x
            ),
            np.max(
                y
            ),
        )
    )


    pad = max(
        0.03,
        0.08
        * (
            maximum
            - minimum
        ),
    )


    lo = minimum - pad

    hi = maximum + pad


    fig, ax = plt.subplots(
        figsize=(
            6.4,
            6.1,
        )
    )


    ax.plot(
        [
            lo,
            hi,
        ],
        [
            lo,
            hi,
        ],
        linestyle="--",
        linewidth=1.4,
        color=IDENTITY_LINE,
        zorder=1,
        label="Identity",
    )


    ax.scatter(
        x,
        y,
        s=58,
        color=COMMON_POINT,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )


    ax.set_xlim(
        lo,
        hi,
    )


    ax.set_ylim(
        lo,
        hi,
    )


    ax.set_aspect(
        "equal",
        adjustable="box",
    )


    ax.set_xlabel(
        "Standard-mask Spearman ρ\n"
        "(common physical tiles)"
    )


    ax.set_ylabel(
        "Corrected-mask Spearman ρ\n"
        "(same physical tiles)"
    )


    ax.set_title(
        "PC3 association is preserved across mask sources",
        fontweight="bold",
        pad=12,
    )


    ax.grid(
        color=GRID,
        linewidth=0.7,
    )


    annotation = (
        f"Across-slide agreement ρ = {across_slide_rho:.3f}\n"
        f"Median standard = {median_standard:.3f}\n"
        f"Median corrected = {median_corrected:.3f}\n"
        f"Median Δρ = {median_delta:+.4f}"
    )


    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle":
                "round,pad=0.45",

            "facecolor":
                "white",

            "edgecolor":
                "#BBBBBB",

            "linewidth":
                0.8,
        },
    )


    fig.tight_layout()


    fig.savefig(
        OUT_PANEL_B_PNG,
        dpi=DPI,
        bbox_inches="tight",
    )


    fig.savefig(
        OUT_PANEL_B_PDF,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    df.to_csv(
        OUT_PANEL_B_DATA,
        index=False,
    )


    print(
        f"[OK] Panel B: "
        f"{OUT_PANEL_B_PNG}"
    )


# =============================================================================
# REPRESENTATIVE SLIDE SELECTION
# =============================================================================

def select_representative_slides(
    primary: pd.DataFrame,
):

    df = (
        primary
        .copy()
    )


    df[
        "spearman_rho"
    ] = pd.to_numeric(
        df[
            "spearman_rho"
        ],
        errors="coerce",
    )


    if (
        df[
            "spearman_rho"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Representative-slide selection "
            "requires finite rho for all 62 slides."
        )


    rows = []


    used_slides = set()


    for (
        quantile_value,
        quantile_label,
    ) in zip(
        REPRESENTATIVE_RHO_QUANTILES,
        REPRESENTATIVE_QUANTILE_LABELS,
    ):


        target_rho = float(
            df[
                "spearman_rho"
            ]
            .quantile(
                quantile_value
            )
        )


        candidates = (
            df
            .assign(
                distance_to_target=
                    np.abs(
                        df[
                            "spearman_rho"
                        ]
                        -
                        target_rho
                    )
            )
            .sort_values(
                [
                    "distance_to_target",
                    "slide_id",
                ]
            )
        )


        selected_row = None


        for _, candidate in (
            candidates.iterrows()
        ):

            slide_id = str(
                candidate[
                    "slide_id"
                ]
            )


            if slide_id not in used_slides:

                selected_row = candidate

                used_slides.add(
                    slide_id
                )

                break


        if selected_row is None:

            raise RuntimeError(
                "Could not select unique "
                "representative slide."
            )


        rows.append(
            {
                "selection_label":
                    quantile_label,

                "target_rho_quantile":
                    float(
                        quantile_value
                    ),

                "target_rho":
                    target_rho,

                "slide_id":
                    str(
                        selected_row[
                            "slide_id"
                        ]
                    ),

                "observed_rho":
                    float(
                        selected_row[
                            "spearman_rho"
                        ]
                    ),

                "distance_to_target":
                    float(
                        selected_row[
                            "distance_to_target"
                        ]
                    ),

                "pc3_q10":
                    float(
                        selected_row[
                            "pc3_q10"
                        ]
                    ),

                "pc3_q90":
                    float(
                        selected_row[
                            "pc3_q90"
                        ]
                    ),

                "selection_rule":
                    (
                        "nearest observed primary "
                        "slide-level Spearman rho "
                        "to requested cohort quantile; "
                        "ties resolved by slide_id"
                    ),
            }
        )


    out = pd.DataFrame(
        rows
    )


    out.to_csv(
        OUT_REP_SLIDES,
        index=False,
    )


    return out


# =============================================================================
# PC3 COHORT-WIDE VISUALIZATION SCALE
# =============================================================================

def compute_cohort_pc3_display_limit(
    primary_measurement_index: Dict[str, Path],
):

    pooled = []


    for slide_id in sorted(
        primary_measurement_index
    ):

        arrays = load_standard_measurements(
            primary_measurement_index[
                slide_id
            ]
        )


        pc3 = np.asarray(
            arrays[
                "pc3_score"
            ],
            dtype=np.float32,
        )


        qc = np.asarray(
            arrays[
                "qc_pass"
            ],
            dtype=bool,
        )


        values = pc3[
            qc
            &
            np.isfinite(
                pc3
            )
        ]


        if len(
            values
        ):

            pooled.append(
                values
            )


    if not pooled:

        raise RuntimeError(
            "No primary QC-pass PC3 values "
            "available for visualization scaling."
        )


    pooled = np.concatenate(
        pooled
    )


    low = float(
        np.percentile(
            pooled,
            PC3_DISPLAY_LOWER_PERCENTILE,
        )
    )


    high = float(
        np.percentile(
            pooled,
            PC3_DISPLAY_UPPER_PERCENTILE,
        )
    )


    vmax = max(
        abs(
            low
        ),
        abs(
            high
        ),
    )


    if (
        not np.isfinite(
            vmax
        )
        or
        vmax <= 0
    ):

        raise RuntimeError(
            "Invalid cohort-wide PC3 "
            "visualization range."
        )


    return {
        "pooled_q01":
            low,

        "pooled_q99":
            high,

        "symmetric_vmax":
            float(
                vmax
            ),

        "n_values":
            int(
                len(
                    pooled
                )
            ),
    }


# =============================================================================
# GRID CONSTRUCTION
# =============================================================================

def make_grid(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    slide_width: int,
    slide_height: int,
    patch_size: int,
):

    grid_height = int(
        math.ceil(
            slide_height
            /
            patch_size
        )
    )


    grid_width = int(
        math.ceil(
            slide_width
            /
            patch_size
        )
    )


    grid = np.full(
        (
            grid_height,
            grid_width,
        ),
        np.nan,
        dtype=np.float32,
    )


    for (
        xx,
        yy,
        value,
        keep,
    ) in zip(
        x,
        y,
        values,
        valid,
    ):

        if not keep:
            continue


        gx = int(
            int(
                xx
            )
            //
            patch_size
        )


        gy = int(
            int(
                yy
            )
            //
            patch_size
        )


        if (
            0
            <= gx
            < grid_width
            and
            0
            <= gy
            < grid_height
        ):

            grid[
                gy,
                gx,
            ] = float(
                value
            )


    return grid


# =============================================================================
# H&E READER
# =============================================================================

class SlideReader:

    def __init__(
        self,
        path: Path,
    ):

        self.path = Path(
            path
        )

        require_file(
            self.path,
            "H&E WSI",
        )

        self.backend = None

        self.slide = None


        if OPENSLIDE_AVAILABLE:

            try:

                self.slide = (
                    openslide.OpenSlide(
                        str(
                            self.path
                        )
                    )
                )

                self.backend = (
                    "openslide"
                )

            except Exception:

                self.slide = None


        if self.slide is None:

            self.slide = (
                Image.open(
                    self.path
                )
            )

            self.backend = "PIL"


    @property
    def dimensions(
        self,
    ):

        if self.backend == "openslide":

            return tuple(
                map(
                    int,
                    self.slide.dimensions,
                )
            )

        return tuple(
            map(
                int,
                self.slide.size,
            )
        )


    def thumbnail(
        self,
        max_size: int,
    ) -> Image.Image:

        width, height = (
            self.dimensions
        )


        scale = min(
            max_size
            / width,

            max_size
            / height,

            1.0,
        )


        target = (
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


        if self.backend == "openslide":

            thumb = (
                self.slide
                .get_thumbnail(
                    target
                )
                .convert(
                    "RGB"
                )
            )

            return thumb


        img = (
            self.slide
            .convert(
                "RGB"
            )
        )


        if img.size != target:

            img = img.resize(
                target,
                Image.Resampling.LANCZOS,
            )


        return img


    def patch(
        self,
        x: int,
        y: int,
        patch_size: int,
    ) -> Image.Image:

        x = int(
            x
        )

        y = int(
            y
        )

        patch_size = int(
            patch_size
        )


        if self.backend == "openslide":

            return (
                self.slide
                .read_region(
                    (
                        x,
                        y,
                    ),
                    0,
                    (
                        patch_size,
                        patch_size,
                    ),
                )
                .convert(
                    "RGB"
                )
            )


        return (
            self.slide
            .crop(
                (
                    x,
                    y,
                    x
                    + patch_size,
                    y
                    + patch_size,
                )
            )
            .convert(
                "RGB"
            )
        )


    def close(
        self,
    ):

        if self.slide is not None:

            try:
                self.slide.close()
            except Exception:
                pass


# =============================================================================
# REPRESENTATIVE ROI SELECTION
# =============================================================================

def deterministic_tail_selection(
    tile_indices: np.ndarray,
    scores: np.ndarray,
    positions: List[float],
):

    if len(
        tile_indices
    ) == 0:

        raise RuntimeError(
            "Cannot select ROI from "
            "an empty frozen PC3 tail."
        )


    order = np.argsort(
        scores
    )


    tile_indices = (
        tile_indices[
            order
        ]
    )


    scores = (
        scores[
            order
        ]
    )


    selected = []


    for position in positions:

        rank = int(
            round(
                position
                *
                (
                    len(
                        tile_indices
                    )
                    - 1
                )
            )
        )


        rank = min(
            max(
                rank,
                0,
            ),
            len(
                tile_indices
            )
            - 1,
        )


        selected.append(
            (
                int(
                    tile_indices[
                        rank
                    ]
                ),
                float(
                    scores[
                        rank
                    ]
                ),
                float(
                    position
                ),
                int(
                    rank
                ),
            )
        )


    return selected


def select_rois_for_slide(
    slide_id: str,
    arrays: Dict[str, np.ndarray],
    q10: float,
    q90: float,
):

    pc3 = np.asarray(
        arrays[
            "pc3_score"
        ],
        dtype=float,
    )


    epi = np.asarray(
        arrays[
            "epithelial_fraction_among_labeled_pixels"
        ],
        dtype=float,
    )


    qc = np.asarray(
        arrays[
            "qc_pass"
        ],
        dtype=bool,
    )


    x = np.asarray(
        arrays[
            "x"
        ],
        dtype=np.int64,
    )


    y = np.asarray(
        arrays[
            "y"
        ],
        dtype=np.int64,
    )


    valid = (
        qc
        &
        np.isfinite(
            pc3
        )
        &
        np.isfinite(
            epi
        )
    )


    low_mask = (
        valid
        &
        (
            pc3
            <= q10
        )
    )


    high_mask = (
        valid
        &
        (
            pc3
            >= q90
        )
    )


    low_idx = np.where(
        low_mask
    )[0]


    high_idx = np.where(
        high_mask
    )[0]


    low_selected = (
        deterministic_tail_selection(
            tile_indices=
                low_idx,

            scores=
                pc3[
                    low_idx
                ],

            positions=
                ROI_WITHIN_TAIL_POSITIONS,
        )
    )


    high_selected = (
        deterministic_tail_selection(
            tile_indices=
                high_idx,

            scores=
                pc3[
                    high_idx
                ],

            positions=
                ROI_WITHIN_TAIL_POSITIONS,
        )
    )


    rows = []


    for tail, selected in [
        (
            "PC3_low",
            low_selected,
        ),
        (
            "PC3_high",
            high_selected,
        ),
    ]:

        for (
            tile_idx,
            score,
            position,
            rank,
        ) in selected:

            rows.append(
                {
                    "slide_id":
                        slide_id,

                    "tail":
                        tail,

                    "tile_idx":
                        int(
                            tile_idx
                        ),

                    "x":
                        int(
                            x[
                                tile_idx
                            ]
                        ),

                    "y":
                        int(
                            y[
                                tile_idx
                            ]
                        ),

                    "pc3_score":
                        float(
                            score
                        ),

                    "epithelial_fraction":
                        float(
                            epi[
                                tile_idx
                            ]
                        ),

                    "within_tail_position":
                        float(
                            position
                        ),

                    "within_tail_sorted_rank":
                        int(
                            rank
                        ),

                    "frozen_pc3_q10":
                        float(
                            q10
                        ),

                    "frozen_pc3_q90":
                        float(
                            q90
                        ),

                    "selection_rule":
                        (
                            "deterministic 5/25/50/75/95% "
                            "position within frozen "
                            "PC3 tail, sorted by PC3 score"
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# SPATIAL PANEL
# =============================================================================

def compute_spatial_display_bbox(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    patch_size: int,
    slide_width: int,
    slide_height: int,
    padding_tiles: int = SPATIAL_DISPLAY_PADDING_TILES,
):
    """
    Visualization-only bounding box around the shared PC3 / epithelial
    spatial support.

    IMPORTANT:
    This does NOT change tile inclusion, measurement, or inference.
    It only removes large scanner-background margins from the rendered
    Panel-C overview.
    """

    valid = np.asarray(valid, dtype=bool)

    if not valid.any():
        raise RuntimeError(
            "Cannot construct spatial display bbox: no valid shared tiles."
        )

    xv = np.asarray(x[valid], dtype=np.int64)
    yv = np.asarray(y[valid], dtype=np.int64)

    pad = int(max(0, padding_tiles) * patch_size)

    x0 = max(0, int(np.min(xv)) - pad)
    y0 = max(0, int(np.min(yv)) - pad)
    x1 = min(slide_width, int(np.max(xv)) + patch_size + pad)
    y1 = min(slide_height, int(np.max(yv)) + patch_size + pad)

    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(
            f"Invalid spatial display bbox: {(x0, y0, x1, y1)}"
        )

    return x0, y0, x1, y1


def crop_thumbnail_to_wsi_bbox(
    thumbnail: np.ndarray,
    slide_width: int,
    slide_height: int,
    bbox: Tuple[int, int, int, int],
):
    """Crop a WSI thumbnail to the same level-0 display bbox."""

    if thumbnail.ndim != 3:
        raise RuntimeError(
            f"Expected RGB thumbnail, found shape {thumbnail.shape}."
        )

    thumb_h, thumb_w = thumbnail.shape[:2]
    x0, y0, x1, y1 = bbox

    sx = thumb_w / float(slide_width)
    sy = thumb_h / float(slide_height)

    tx0 = max(0, int(math.floor(x0 * sx)))
    ty0 = max(0, int(math.floor(y0 * sy)))
    tx1 = min(thumb_w, int(math.ceil(x1 * sx)))
    ty1 = min(thumb_h, int(math.ceil(y1 * sy)))

    if tx1 <= tx0 or ty1 <= ty0:
        raise RuntimeError(
            "Thumbnail crop became empty after mapping the WSI display bbox."
        )

    return thumbnail[ty0:ty1, tx0:tx1]


def crop_grid_to_wsi_bbox(
    grid: np.ndarray,
    bbox: Tuple[int, int, int, int],
    patch_size: int,
):
    """Crop a tile grid to the grid cells overlapping a level-0 bbox."""

    x0, y0, x1, y1 = bbox

    gx0 = max(0, int(math.floor(x0 / patch_size)))
    gy0 = max(0, int(math.floor(y0 / patch_size)))
    gx1 = min(grid.shape[1], int(math.ceil(x1 / patch_size)))
    gy1 = min(grid.shape[0], int(math.ceil(y1 / patch_size)))

    if gx1 <= gx0 or gy1 <= gy0:
        raise RuntimeError(
            "Heatmap crop became empty after mapping the WSI display bbox."
        )

    return grid[gy0:gy1, gx0:gx1]


def plot_spatial_panel(
    selection_label: str,
    slide_row: pd.Series,
    manifest16a: pd.DataFrame,
    measurement_index: Dict[str, Path],
    pc3_vmax: float,
):

    slide_id = str(
        slide_row[
            "slide_id"
        ]
    )


    match16a = (
        manifest16a.loc[
            manifest16a[
                "slide_id"
            ]
            .astype(
                str
            )
            .eq(
                slide_id
            )
        ]
    )


    if len(
        match16a
    ) != 1:

        raise RuntimeError(
            f"{slide_id}: expected exactly "
            "one Step-16a manifest row."
        )


    row16a = (
        match16a.iloc[
            0
        ]
    )


    wsi_path = Path(
        str(
            row16a[
                "wsi_path"
            ]
        )
    )


    slide_width = int(
        row16a[
            "slide_width"
        ]
    )


    slide_height = int(
        row16a[
            "slide_height"
        ]
    )


    patch_size = int(
        row16a[
            "patch_size"
        ]
    )


    arrays = (
        load_standard_measurements(
            measurement_index[
                slide_id
            ]
        )
    )


    pc3 = np.asarray(
        arrays[
            "pc3_score"
        ],
        dtype=float,
    )


    epi = np.asarray(
        arrays[
            "epithelial_fraction_among_labeled_pixels"
        ],
        dtype=float,
    )


    qc = np.asarray(
        arrays[
            "qc_pass"
        ],
        dtype=bool,
    )


    x = np.asarray(
        arrays[
            "x"
        ],
        dtype=np.int64,
    )


    y = np.asarray(
        arrays[
            "y"
        ],
        dtype=np.int64,
    )


    common_support = (
        qc
        &
        np.isfinite(
            pc3
        )
        &
        np.isfinite(
            epi
        )
    )


    pc3_grid = make_grid(

        values=
            pc3,

        x=
            x,

        y=
            y,

        valid=
            common_support,

        slide_width=
            slide_width,

        slide_height=
            slide_height,

        patch_size=
            patch_size,
    )


    epi_grid = make_grid(

        values=
            epi,

        x=
            x,

        y=
            y,

        valid=
            common_support,

        slide_width=
            slide_width,

        slide_height=
            slide_height,

        patch_size=
            patch_size,
    )


    # -------------------------------------------------------------------------
    # Visualization-only spatial crop.
    #
    # The bbox is derived ONLY from the shared Step-16b QC-pass support and
    # is applied identically to H&E, PC3, and epithelial maps.
    # -------------------------------------------------------------------------

    display_bbox = compute_spatial_display_bbox(
        x=x,
        y=y,
        valid=common_support,
        patch_size=patch_size,
        slide_width=slide_width,
        slide_height=slide_height,
        padding_tiles=SPATIAL_DISPLAY_PADDING_TILES,
    )


    pc3_grid_display = crop_grid_to_wsi_bbox(
        grid=pc3_grid,
        bbox=display_bbox,
        patch_size=patch_size,
    )


    epi_grid_display = crop_grid_to_wsi_bbox(
        grid=epi_grid,
        bbox=display_bbox,
        patch_size=patch_size,
    )


    q10 = float(
        slide_row[
            "pc3_q10"
        ]
    )


    q90 = float(
        slide_row[
            "pc3_q90"
        ]
    )


    roi_df = (
        select_rois_for_slide(
            slide_id=
                slide_id,

            arrays=
                arrays,

            q10=
                q10,

            q90=
                q90,
        )
    )


    reader = SlideReader(
        wsi_path
    )


    actual_width, actual_height = (
        reader.dimensions
    )


    if (
        actual_width
        != slide_width
        or
        actual_height
        != slide_height
    ):

        reader.close()

        raise RuntimeError(
            f"{slide_id}: H&E dimensions "
            "do not match frozen Step-16a geometry.\n"
            f"Frozen: {(slide_width, slide_height)}\n"
            f"Actual: {(actual_width, actual_height)}"
        )


    thumbnail_full = np.asarray(
        reader.thumbnail(
            WSI_THUMBNAIL_MAX_SIZE
        )
    )


    thumbnail = crop_thumbnail_to_wsi_bbox(
        thumbnail=thumbnail_full,
        slide_width=slide_width,
        slide_height=slide_height,
        bbox=display_bbox,
    )


    roi_images = {}


    for _, roi in (
        roi_df.iterrows()
    ):

        tile_idx = int(
            roi[
                "tile_idx"
            ]
        )


        roi_images[
            tile_idx
        ] = (
            reader.patch(
                x=int(
                    roi[
                        "x"
                    ]
                ),
                y=int(
                    roi[
                        "y"
                    ]
                ),
                patch_size=
                    patch_size,
            )
        )


    reader.close()


    # =========================================================================
    # Figure layout — v1.1
    #
    # Changes are VISUAL ONLY:
    #   - one clean title line + one subtitle line;
    #   - larger spatial maps;
    #   - shared tissue-support crop removes scanner-background margins;
    #   - ROI row labels occupy their own grid column;
    #   - less vertical whitespace between overview and ROIs;
    #   - five deterministic ROIs per frozen tail (5/25/50/75/95%).
    # =========================================================================

    fig = plt.figure(
        figsize=(
            15.8,
            8.6,
        ),
        facecolor="white",
    )


    rho = float(
        slide_row[
            "observed_rho"
        ]
    )


    clean_slide_name = (
        slide_id
        .replace(
            "_HE",
            "",
        )
    )


    # -------------------------------------------------------------------------
    # Header — separated from axes titles to prevent overlap.
    # -------------------------------------------------------------------------

    fig.text(
        0.5,
        0.975,
        f"PESO spatial validation — {clean_slide_name}",
        ha="center",
        va="top",
        fontsize=16.5,
        fontweight="bold",
        color=TEXT,
    )


    fig.text(
        0.5,
        0.942,
        (
            f"{selection_label.upper()} representative  |  "
            f"within-slide Spearman ρ = {rho:.3f}"
        ),
        ha="center",
        va="top",
        fontsize=11.5,
        color=MUTED,
    )


    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[
            1.32,
            1.08,
        ],
        hspace=0.13,
        left=0.055,
        right=0.965,
        top=0.885,
        bottom=0.07,
    )


    # -------------------------------------------------------------------------
    # Top row: H&E / PC3 / epithelial fraction.
    # -------------------------------------------------------------------------

    top = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[
            0
        ],
        wspace=0.17,
    )


    ax_he = fig.add_subplot(
        top[
            0,
            0,
        ]
    )


    ax_pc3 = fig.add_subplot(
        top[
            0,
            1,
        ]
    )


    ax_epi = fig.add_subplot(
        top[
            0,
            2,
        ]
    )


    ax_he.imshow(
        thumbnail
    )


    ax_he.set_title(
        "H&E",
        fontweight="bold",
        fontsize=13.5,
        pad=7,
    )


    ax_he.axis(
        "off"
    )


    # PC3 ---------------------------------------------------------------------

    pc3_cmap = (
        plt.get_cmap(
            PC3_CMAP
        )
        .copy()
    )

    pc3_cmap.set_bad(
        "white"
    )


    pc3_norm = TwoSlopeNorm(
        vmin=
            -pc3_vmax,

        vcenter=
            0.0,

        vmax=
            pc3_vmax,
    )


    im_pc3 = ax_pc3.imshow(
        np.ma.masked_invalid(
            pc3_grid_display
        ),
        cmap=
            pc3_cmap,
        norm=
            pc3_norm,
        interpolation=
            "nearest",
        origin=
            "upper",
        aspect=
            "equal",
    )


    ax_pc3.set_title(
        "Frozen PC3",
        fontweight="bold",
        fontsize=13.5,
        pad=7,
    )


    ax_pc3.axis(
        "off"
    )


    cbar_pc3 = fig.colorbar(
        im_pc3,
        ax=ax_pc3,
        fraction=0.035,
        pad=0.018,
        shrink=0.86,
    )


    cbar_pc3.ax.tick_params(
        labelsize=8.5
    )


    cbar_pc3.set_label(
        "PC3 score",
        fontsize=9.5,
    )


    # Epithelial fraction ------------------------------------------------------

    epi_cmap = (
        plt.get_cmap(
            EPITHELIAL_CMAP
        )
        .copy()
    )

    epi_cmap.set_bad(
        "white"
    )


    epi_norm = Normalize(
        vmin=0.0,
        vmax=1.0,
    )


    im_epi = ax_epi.imshow(
        np.ma.masked_invalid(
            epi_grid_display
        ),
        cmap=
            epi_cmap,
        norm=
            epi_norm,
        interpolation=
            "nearest",
        origin=
            "upper",
        aspect=
            "equal",
    )


    ax_epi.set_title(
        "Epithelial fraction",
        fontweight="bold",
        fontsize=13.5,
        pad=7,
    )


    ax_epi.axis(
        "off"
    )


    cbar_epi = fig.colorbar(
        im_epi,
        ax=ax_epi,
        fraction=0.035,
        pad=0.018,
        shrink=0.86,
    )


    cbar_epi.ax.tick_params(
        labelsize=8.5
    )


    cbar_epi.set_label(
        "Fraction among labeled pixels",
        fontsize=9.5,
    )


    # -------------------------------------------------------------------------
    # Bottom row: row-label column + 5 deterministic H&E ROIs.
    # -------------------------------------------------------------------------

    bottom = gridspec.GridSpecFromSubplotSpec(
        2,
        6,
        subplot_spec=outer[
            1
        ],
        width_ratios=[
            0.20,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        wspace=0.055,
        hspace=0.24,
    )


    low_df = (
        roi_df.loc[
            roi_df[
                "tail"
            ]
            .eq(
                "PC3_low"
            )
        ]
        .sort_values(
            "within_tail_position"
        )
    )


    high_df = (
        roi_df.loc[
            roi_df[
                "tail"
            ]
            .eq(
                "PC3_high"
            )
        ]
        .sort_values(
            "within_tail_position"
        )
    )


    # Dedicated label axes avoid the previous large left-gap / detached labels.

    ax_low_label = fig.add_subplot(
        bottom[
            0,
            0,
        ]
    )

    ax_low_label.axis(
        "off"
    )

    ax_low_label.text(
        0.98,
        0.5,
        "PC3-low\n(bottom 10%)",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )


    ax_high_label = fig.add_subplot(
        bottom[
            1,
            0,
        ]
    )

    ax_high_label.axis(
        "off"
    )

    ax_high_label.text(
        0.98,
        0.5,
        "PC3-high\n(top 10%)",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )


    for col, (
        _,
        roi,
    ) in enumerate(
        low_df.iterrows(),
        start=1,
    ):

        ax = fig.add_subplot(
            bottom[
                0,
                col,
            ]
        )


        tile_idx = int(
            roi[
                "tile_idx"
            ]
        )


        ax.imshow(
            roi_images[
                tile_idx
            ]
        )


        ax.set_title(
            (
                f"PC3 {roi['pc3_score']:.2f}  |  "
                f"Epi {roi['epithelial_fraction']:.2f}"
            ),
            fontsize=9.5,
            pad=4,
        )


        ax.axis(
            "off"
        )


    for col, (
        _,
        roi,
    ) in enumerate(
        high_df.iterrows(),
        start=1,
    ):

        ax = fig.add_subplot(
            bottom[
                1,
                col,
            ]
        )


        tile_idx = int(
            roi[
                "tile_idx"
            ]
        )


        ax.imshow(
            roi_images[
                tile_idx
            ]
        )


        ax.set_title(
            (
                f"PC3 {roi['pc3_score']:.2f}  |  "
                f"Epi {roi['epithelial_fraction']:.2f}"
            ),
            fontsize=9.5,
            pad=4,
        )


        ax.axis(
            "off"
        )


    # Footer records that all visual comparison maps use identical support.

    fig.text(
        0.5,
        0.025,
        (
            "PC3 and epithelial maps use identical Step-16b standard-mask "
            "QC-pass tile support; H&E ROIs are selected deterministically "
            "from the frozen PC3 tails."
        ),
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=MUTED,
    )


    output_png = (
        OUT_DIR
        /
        (
            f"panel_C_spatial_{selection_label}_"
            f"{slide_id}.png"
        )
    )


    output_pdf = (
        OUT_DIR
        /
        (
            f"panel_C_spatial_{selection_label}_"
            f"{slide_id}.pdf"
        )
    )


    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )


    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    roi_df[
        "selection_label"
    ] = (
        selection_label
    )


    roi_df[
        "wsi_path"
    ] = str(
        wsi_path
    )


    roi_df[
        "patch_size"
    ] = int(
        patch_size
    )


    roi_df[
        "spatial_display_bbox_x0"
    ] = int(
        display_bbox[
            0
        ]
    )


    roi_df[
        "spatial_display_bbox_y0"
    ] = int(
        display_bbox[
            1
        ]
    )


    roi_df[
        "spatial_display_bbox_x1"
    ] = int(
        display_bbox[
            2
        ]
    )


    roi_df[
        "spatial_display_bbox_y1"
    ] = int(
        display_bbox[
            3
        ]
    )


    roi_df[
        "spatial_display_padding_tiles"
    ] = int(
        SPATIAL_DISPLAY_PADDING_TILES
    )


    roi_df[
        "spatial_panel_png"
    ] = str(
        output_png
    )


    print(
        f"[OK] Panel C "
        f"{selection_label}: "
        f"{output_png}"
    )


    return roi_df


# =============================================================================
# README
# =============================================================================

def build_readme(
    representative_slides: pd.DataFrame,
    pc3_scale: Dict,
):

    lines = [
        "STEP 16D — PESO PC3 EXTERNAL VALIDATION FIGURES",
        "================================================",
        "",
        f"Version: {VERSION}",
        f"Schema : {SCHEMA_VERSION}",
        "",
        "ROLE",
        "----",
        "",
        "Visualization / reporting only.",
        "",
        "No new biological hypothesis test is performed.",
        "",
        "Panel A",
        "-------",
        "",
        "62 primary standard-mask slide-level Spearman rho values",
        "from Step 16c, ordered for display.",
        "",
        "The cohort median and bootstrap CI are read directly from",
        "Step 16c and are not recomputed.",
        "",
        "Panel B",
        "-------",
        "",
        "Common-physical-tile standard-mask versus corrected-mask",
        "slide-level rho values from Step 16c.",
        "",
        "The identity line is visual only.",
        "",
        "Panel C",
        "-------",
        "",
        "Algorithmically selected representative slides.",
        "",
        "Target rho quantiles:",
        "    0.25",
        "    0.50",
        "    0.75",
        "",
        "Representative slide selection:",
    ]


    for _, row in (
        representative_slides.iterrows()
    ):

        lines.append(
            (
                f"    {row['selection_label']}: "
                f"{row['slide_id']} "
                f"(rho={row['observed_rho']:.4f})"
            )
        )


    lines.extend(
        [
            "",
            "Spatial heatmaps",
            "----------------",
            "",
            "PC3 and epithelial-fraction maps are generated directly",
            "from Step-16b standard-mask measurement NPZ files.",
            "",
            "Only tiles with:",
            "",
            "    qc_pass == True",
            "",
            "are visualized in both heatmaps.",
            "",
            "Panel-C overview images are cropped for DISPLAY ONLY to the",
            "bounding box of the shared QC-pass spatial support plus a fixed",
            "two-tile padding. The same crop is applied to H&E, PC3, and",
            "epithelial-fraction views.",
            "",
            "This crop does not change tile inclusion, measurement, or inference.",
            "",
            "No raw mask is read or remeasured in Step 16d.",
            "",
            "PC3 visualization scale",
            "-----------------------",
            "",
            (
                f"Pooled primary QC-pass values: "
                f"{pc3_scale['n_values']}"
            ),
            (
                f"Pooled PC3 1st percentile: "
                f"{pc3_scale['pooled_q01']:.6f}"
            ),
            (
                f"Pooled PC3 99th percentile: "
                f"{pc3_scale['pooled_q99']:.6f}"
            ),
            (
                f"Symmetric plotting vmax: "
                f"{pc3_scale['symmetric_vmax']:.6f}"
            ),
            "",
            "This range is used ONLY for figure display.",
            "",
            "It is not a biological threshold and does not affect",
            "tile inclusion or inference.",
            "",
            "Epithelial visualization scale",
            "------------------------------",
            "",
            "Fixed from 0 to 1.",
            "",
            "Representative ROI selection",
            "----------------------------",
            "",
            "Low tail:",
            "    PC3 <= frozen Step-16c within-slide q10",
            "",
            "High tail:",
            "    PC3 >= frozen Step-16c within-slide q90",
            "",
            "Within each frozen tail, five H&E ROIs are selected",
            "deterministically at 5%, 25%, 50%, 75%, and 95% positions of the",
            "PC3-score-ranked tail.",
            "",
            "No manual ROI selection is performed.",
            "",
            "NOT PERFORMED",
            "-------------",
            "",
            "No PC6 analysis.",
            "No PC1-PC8 screening.",
            "No raw-mask measurement.",
            "No new correlation test.",
            "No pooled tile-level inference.",
            "No threshold optimization.",
            "No manual representative-slide selection.",
            "No manual ROI selection.",
            "No hybrid mask analysis.",
            "",
        ]
    )


    return "\n".join(
        lines
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
        "STEP 16D: PESO PC3 EXTERNAL VALIDATION FIGURES"
    )

    print(
        "=" * 124
    )


    print(
        "[VERSION]   ",
        VERSION,
    )


    print(
        "[SCHEMA]    ",
        SCHEMA_VERSION,
    )


    print(
        "[STEP 16A]  ",
        STEP16A_DIR,
    )


    print(
        "[STEP 16B]  ",
        STEP16B_DIR,
    )


    print(
        "[STEP 16C]  ",
        STEP16C_DIR,
    )


    print(
        "[OUTPUT]    ",
        OUT_DIR,
    )


    print(
        "[OVERWRITE] ",
        args.overwrite,
    )


    print(
        "[OPENSLIDE] ",
        OPENSLIDE_AVAILABLE,
    )


    # =========================================================================
    # Validation
    # =========================================================================

    validate_inputs()


    guard_outputs(
        args.overwrite
    )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    meta16b = load_json(
        STEP16B_METADATA
    )


    meta16c = load_json(
        STEP16C_METADATA
    )


    validate_formal_chain(
        meta16b=
            meta16b,

        meta16c=
            meta16c,
    )


    (
        manifest16a,
        manifest16b,
        primary,
        primary_summary,
        common,
        common_summary,
    ) = load_formal_tables()


    measurement_index = (
        build_standard_measurement_index(
            manifest16b
        )
    )


    # =========================================================================
    # Panel A
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "PANEL A — PRIMARY SLIDE-LEVEL ASSOCIATION"
    )

    print(
        "=" * 124
    )


    plot_panel_a(
        primary=
            primary,

        summary=
            primary_summary,
    )


    # =========================================================================
    # Panel B
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "PANEL B — COMMON-TILE MASK-SOURCE ROBUSTNESS"
    )

    print(
        "=" * 124
    )


    plot_panel_b(
        common=
            common,

        summary=
            common_summary,
    )


    # =========================================================================
    # Representative selection
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "REPRESENTATIVE SLIDE SELECTION"
    )

    print(
        "=" * 124
    )


    representative_slides = (
        select_representative_slides(
            primary
        )
    )


    print(
        representative_slides.to_string(
            index=False
        )
    )


    # =========================================================================
    # Cohort-wide PC3 plotting scale
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "COHORT-WIDE PC3 VISUALIZATION SCALE"
    )

    print(
        "=" * 124
    )


    pc3_scale = (
        compute_cohort_pc3_display_limit(
            measurement_index
        )
    )


    print(
        "Pooled QC-pass PC3 values:",
        f"{pc3_scale['n_values']:,}",
    )


    print(
        "PC3 pooled q01:",
        f"{pc3_scale['pooled_q01']:.4f}",
    )


    print(
        "PC3 pooled q99:",
        f"{pc3_scale['pooled_q99']:.4f}",
    )


    print(
        "Symmetric vmax:",
        f"{pc3_scale['symmetric_vmax']:.4f}",
    )


    # =========================================================================
    # Panel C spatial figures
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "PANEL C — SPATIAL VALIDATION"
    )

    print(
        "=" * 124
    )


    all_roi_rows = []


    for _, representative in (
        representative_slides.iterrows()
    ):

        roi_df = (
            plot_spatial_panel(

                selection_label=
                    str(
                        representative[
                            "selection_label"
                        ]
                    ),

                slide_row=
                    representative,

                manifest16a=
                    manifest16a,

                measurement_index=
                    measurement_index,

                pc3_vmax=
                    pc3_scale[
                        "symmetric_vmax"
                    ],
            )
        )


        all_roi_rows.append(
            roi_df
        )


    roi_selection = pd.concat(
        all_roi_rows,
        ignore_index=True,
    )


    roi_selection.to_csv(
        OUT_REP_ROIS,
        index=False,
    )


    # =========================================================================
    # Metadata
    # =========================================================================

    metadata = {

        "step":
            "16d",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),

        "role":
            "visualization_and_reporting_only",

        "new_inferential_statistics_performed":
            False,

        "raw_masks_accessed":
            False,

        "epithelial_fraction_recomputed":
            False,

        "pc_axis_screening_performed":
            False,

        "pc6_analyzed":
            False,

        "manual_slide_selection_performed":
            False,

        "manual_roi_selection_performed":
            False,

        "hybrid_mask_analysis_performed":
            False,


        "panel_A": {

            "source":
                str(
                    STEP16C_PRIMARY_SLIDES
                ),

            "summary_source":
                str(
                    STEP16C_PRIMARY_COHORT
                ),

            "n_slides":
                EXPECTED_PRIMARY_SLIDES,

            "display":
                (
                    "ordered slide-level Spearman rho "
                    "with Step-16c median and bootstrap CI"
                ),
        },


        "panel_B": {

            "source":
                str(
                    STEP16C_COMMON_TILE
                ),

            "summary_source":
                str(
                    STEP16C_COMMON_TILE_SUMMARY
                ),

            "n_slides":
                EXPECTED_CORRECTED_SLIDES,

            "display":
                (
                    "standard vs corrected slide-level "
                    "rho on identical common physical tiles"
                ),
        },


        "panel_C": {

            "representative_slide_quantiles":
                REPRESENTATIVE_RHO_QUANTILES,

            "representative_slide_selection":
                (
                    "nearest observed primary slide-level rho "
                    "to requested cohort rho quantile; "
                    "ties resolved by slide_id"
                ),

            "pc3_visualization_percentiles":
                [
                    PC3_DISPLAY_LOWER_PERCENTILE,
                    PC3_DISPLAY_UPPER_PERCENTILE,
                ],

            "pc3_symmetric_vmax":
                pc3_scale[
                    "symmetric_vmax"
                ],

            "epithelial_visualization_range":
                [
                    0.0,
                    1.0,
                ],

            "shared_heatmap_support":
                "Step-16b standard-mask qc_pass tiles",

            "spatial_display_crop":
                (
                    "visualization-only bounding box of shared qc_pass "
                    "support plus fixed tile padding; applied identically "
                    "to H&E, PC3, and epithelial views"
                ),

            "spatial_display_padding_tiles":
                SPATIAL_DISPLAY_PADDING_TILES,

            "roi_tail_threshold_source":
                "Step-16c frozen within-slide q10/q90",

            "roi_within_tail_positions":
                ROI_WITHIN_TAIL_POSITIONS,

            "n_rois_per_tail":
                N_ROI_PER_TAIL,

            "raw_H_and_E_accessed_for_visualization_only":
                True,
        },


        "input_hashes": {

            "step16a_manifest":
                sha256_file(
                    STEP16A_MANIFEST
                ),

            "step16a_plan":
                sha256_file(
                    STEP16A_PLAN
                ),

            "step16a_metadata":
                sha256_file(
                    STEP16A_METADATA
                ),

            "step16b_manifest":
                sha256_file(
                    STEP16B_MANIFEST
                ),

            "step16b_metadata":
                sha256_file(
                    STEP16B_METADATA
                ),

            "step16c_primary_slides":
                sha256_file(
                    STEP16C_PRIMARY_SLIDES
                ),

            "step16c_primary_cohort":
                sha256_file(
                    STEP16C_PRIMARY_COHORT
                ),

            "step16c_common_tile":
                sha256_file(
                    STEP16C_COMMON_TILE
                ),

            "step16c_common_tile_summary":
                sha256_file(
                    STEP16C_COMMON_TILE_SUMMARY
                ),

            "step16c_metadata":
                sha256_file(
                    STEP16C_METADATA
                ),
        },


        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "matplotlib_version":
            matplotlib.__version__,

        "openslide_available":
            OPENSLIDE_AVAILABLE,
    }


    write_json(
        OUT_METADATA,
        metadata,
    )


    # =========================================================================
    # README
    # =========================================================================

    readme = build_readme(
        representative_slides=
            representative_slides,

        pc3_scale=
            pc3_scale,
    )


    OUT_README.write_text(
        readme,
        encoding="utf-8",
    )


    # =========================================================================
    # Final console summary
    # =========================================================================

    print()

    print(
        "=" * 124
    )

    print(
        "STEP 16D SUMMARY"
    )

    print(
        "=" * 124
    )


    print(
        "Panel A:",
        OUT_PANEL_A_PNG,
    )


    print(
        "Panel B:",
        OUT_PANEL_B_PNG,
    )


    print()

    print(
        "Panel C representative slides:"
    )


    for _, row in (
        representative_slides.iterrows()
    ):

        print(
            f"  {row['selection_label']}: "
            f"{row['slide_id']} | "
            f"rho={row['observed_rho']:.4f}"
        )


    print()

    print(
        "Representative slide selection:",
        OUT_REP_SLIDES,
    )


    print(
        "Representative ROI selection:",
        OUT_REP_ROIS,
    )


    print(
        "Metadata:",
        OUT_METADATA,
    )


    print(
        "README:",
        OUT_README,
    )


    print()

    print(
        "[INFERENCE] None performed."
    )


    print(
        "[PC SCREENING] None performed."
    )


    print(
        "[MASK REMEASUREMENT] None performed."
    )


    print(
        "[MANUAL SELECTION] None performed."
    )


    print()

    print(
        "[DONE] Step-16d PESO PC3 external "
        "validation figures generated."
    )


if __name__ == "__main__":
    main()