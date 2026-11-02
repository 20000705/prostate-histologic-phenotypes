#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13c_build_pc3_tumor_core_sensitivity_manifest.py

Purpose
-------
Extend the frozen PC3 morphology sampling framework with a stricter
tumor-purity sensitivity analysis.

This script does NOT alter or overwrite the primary step-13a sampling design.

It reads the frozen candidate table from step 13a and reproduces the original
four strata for comparison:

    all_tissue_low
    all_tissue_high
    tumor_only_low          tumor overlap >= 0.10
    tumor_only_high         tumor overlap >= 0.10

and adds two stricter sensitivity strata:

    tumor_core_low          tumor overlap >= 0.80
    tumor_core_high         tumor overlap >= 0.80

The goal is descriptive:
    - How many patients remain represented?
    - How many slides remain represented?
    - How many candidate tiles exist?
    - How many balanced sampled tiles remain?
    - How does tumor overlap differ across strata?

No images are extracted and no morphology metrics are computed.

Input
-----
results/13a_pc3_morphology_sampling_manifest/
    all_pc3_candidate_tiles.csv.gz

Output
------
results/13c_pc3_tumor_core_sensitivity_sampling/
    pc3_tumor_core_sensitivity_manifest.csv
    pc3_tumor_core_candidate_summary.csv
    pc3_tumor_core_sampling_summary.csv
    pc3_tumor_core_slide_summary.csv
    pc3_tumor_core_patient_summary.csv
    analysis_metadata.json
    README.txt
"""

import json
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


import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = RESULTS_ROOT

IN_CANDIDATES = (
    RESULTS_ROOT
    / "13a_pc3_morphology_sampling_manifest"
    / "all_pc3_candidate_tiles.csv.gz"
)

OUT_DIR = (
    RESULTS_ROOT
    / "13c_pc3_tumor_core_sensitivity_sampling"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST = (
    OUT_DIR
    / "pc3_tumor_core_sensitivity_manifest.csv"
)

OUT_CANDIDATE_SUMMARY = (
    OUT_DIR
    / "pc3_tumor_core_candidate_summary.csv"
)

OUT_SAMPLING_SUMMARY = (
    OUT_DIR
    / "pc3_tumor_core_sampling_summary.csv"
)

OUT_SLIDE_SUMMARY = (
    OUT_DIR
    / "pc3_tumor_core_slide_summary.csv"
)

OUT_PATIENT_SUMMARY = (
    OUT_DIR
    / "pc3_tumor_core_patient_summary.csv"
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
PRIMARY_TUMOR_THRESHOLD = 0.10
TUMOR_CORE_THRESHOLD = 0.80

MAX_TILES_PER_SLIDE_PER_STRATUM = 10
MAX_TILES_PER_PATIENT_PER_STRATUM = 40

MIN_CANDIDATES_PER_SLIDE_PER_STRATUM = 1

ROUND_DIGITS = 6


# =============================================================================
# HELPERS
# =============================================================================
def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def validate_candidates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "dataset",
        "patient_id",
        "prefix",
        "tile_idx",
        "x",
        "y",
        "patch_size",
        "pc3_score",
        "pc3_within_slide_percentile",
        "is_pc3_low_extreme",
        "is_pc3_high_extreme",
        "tumor_overlap_fraction",
        "is_tumor_associated",
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Candidate table missing columns: "
            f"{sorted(missing)}"
        )

    out = dataframe.copy()

    out["patient_id"] = (
        out["patient_id"]
        .astype(str)
    )

    out["prefix"] = (
        out["prefix"]
        .astype(str)
    )

    for column in [
        "tile_idx",
        "x",
        "y",
        "patch_size",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="raise",
        ).astype(int)

    for column in [
        "pc3_score",
        "pc3_within_slide_percentile",
        "tumor_overlap_fraction",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    for column in [
        "is_pc3_low_extreme",
        "is_pc3_high_extreme",
        "is_tumor_associated",
    ]:
        if out[column].dtype != bool:
            out[column] = (
                out[column]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    }
                )
            )

    return out


def round_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    out = dataframe.copy()

    numeric = (
        out
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    out[numeric] = (
        out[numeric]
        .round(ROUND_DIGITS)
    )

    return out


# =============================================================================
# STRATA
# =============================================================================
def get_stratum_candidates(
    all_candidates: pd.DataFrame,
    stratum: str,
) -> pd.DataFrame:
    low = all_candidates[
        "is_pc3_low_extreme"
    ]

    high = all_candidates[
        "is_pc3_high_extreme"
    ]

    overlap = all_candidates[
        "tumor_overlap_fraction"
    ]

    if stratum == "all_tissue_low":
        mask = low

    elif stratum == "all_tissue_high":
        mask = high

    elif stratum == "tumor_only_low":
        mask = (
            low
            & (
                overlap
                >= PRIMARY_TUMOR_THRESHOLD
            )
        )

    elif stratum == "tumor_only_high":
        mask = (
            high
            & (
                overlap
                >= PRIMARY_TUMOR_THRESHOLD
            )
        )

    elif stratum == "tumor_core_low":
        mask = (
            low
            & (
                overlap
                >= TUMOR_CORE_THRESHOLD
            )
        )

    elif stratum == "tumor_core_high":
        mask = (
            high
            & (
                overlap
                >= TUMOR_CORE_THRESHOLD
            )
        )

    else:
        raise ValueError(
            f"Unknown stratum: {stratum}"
        )

    return (
        all_candidates.loc[
            mask
        ]
        .copy()
    )


def direction_from_stratum(
    stratum: str,
) -> str:
    if stratum.endswith("_low"):
        return "low"

    if stratum.endswith("_high"):
        return "high"

    raise ValueError(
        f"Cannot infer direction: {stratum}"
    )


def deterministic_rank(
    dataframe: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    out = dataframe.copy()

    if direction == "low":
        ascending = [
            True,
            True,
            True,
            True,
        ]

    elif direction == "high":
        ascending = [
            True,
            True,
            False,
            True,
        ]

    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    return out.sort_values(
        [
            "patient_id",
            "prefix",
            "pc3_score",
            "tile_idx",
        ],
        ascending=ascending,
        kind="mergesort",
    )


def sample_balanced_stratum(
    candidates: pd.DataFrame,
    stratum: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    direction = direction_from_stratum(
        stratum
    )

    ranked = deterministic_rank(
        candidates,
        direction,
    )

    slide_blocks = []

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
            len(group)
            < MIN_CANDIDATES_PER_SLIDE_PER_STRATUM
        ):
            continue

        selected = (
            group.head(
                MAX_TILES_PER_SLIDE_PER_STRATUM
            )
            .copy()
        )

        selected[
            "rank_within_slide_for_stratum"
        ] = np.arange(
            1,
            len(selected) + 1,
        )

        slide_blocks.append(
            selected
        )

    if not slide_blocks:
        return pd.DataFrame()

    after_slide_cap = pd.concat(
        slide_blocks,
        ignore_index=True,
    )

    patient_blocks = []

    for patient_id, group in (
        after_slide_cap.groupby(
            "patient_id",
            sort=True,
        )
    ):
        ranked_patient = deterministic_rank(
            group,
            direction,
        )

        selected = (
            ranked_patient.head(
                MAX_TILES_PER_PATIENT_PER_STRATUM
            )
            .copy()
        )

        selected[
            "rank_within_patient_for_stratum"
        ] = np.arange(
            1,
            len(selected) + 1,
        )

        patient_blocks.append(
            selected
        )

    sampled = pd.concat(
        patient_blocks,
        ignore_index=True,
    )

    sampled[
        "sampling_stratum"
    ] = stratum

    sampled[
        "sampling_direction"
    ] = direction

    sampled[
        "tumor_threshold_used"
    ] = np.where(
        stratum.startswith(
            "tumor_core"
        ),
        TUMOR_CORE_THRESHOLD,
        np.where(
            stratum.startswith(
                "tumor_only"
            ),
            PRIMARY_TUMOR_THRESHOLD,
            np.nan,
        ),
    )

    return sampled


def assign_sample_ids(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    out = sampled.copy()

    stratum_order = {
        "all_tissue_low": 1,
        "all_tissue_high": 2,
        "tumor_only_low": 3,
        "tumor_only_high": 4,
        "tumor_core_low": 5,
        "tumor_core_high": 6,
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
        "sensitivity_sample_id"
    ] = [
        f"PC3SENS_{index:04d}"
        for index in range(
            1,
            len(out) + 1,
        )
    ]

    out[
        "model_tile_uid"
    ] = (
        out["prefix"].astype(str)
        + "__tile"
        + out["tile_idx"].astype(str)
    )

    return out.drop(
        columns=[
            "_stratum_order"
        ]
    )


# =============================================================================
# SUMMARIES
# =============================================================================
def build_candidate_summary(
    all_candidates: pd.DataFrame,
    strata,
) -> pd.DataFrame:
    rows = []

    for stratum in strata:
        group = get_stratum_candidates(
            all_candidates,
            stratum,
        )

        rows.append({
            "sampling_stratum": (
                stratum
            ),

            "n_candidate_tiles": int(
                len(group)
            ),

            "n_patients_with_candidates": int(
                group[
                    "patient_id"
                ].nunique()
            )
            if not group.empty
            else 0,

            "n_slides_with_candidates": int(
                group[
                    "prefix"
                ].nunique()
            )
            if not group.empty
            else 0,

            "median_pc3_score": float(
                group[
                    "pc3_score"
                ].median()
            )
            if not group.empty
            else np.nan,

            "median_tumor_overlap_fraction": float(
                group[
                    "tumor_overlap_fraction"
                ].median()
            )
            if not group.empty
            else np.nan,

            "q25_tumor_overlap_fraction": float(
                group[
                    "tumor_overlap_fraction"
                ].quantile(
                    0.25
                )
            )
            if not group.empty
            else np.nan,

            "q75_tumor_overlap_fraction": float(
                group[
                    "tumor_overlap_fraction"
                ].quantile(
                    0.75
                )
            )
            if not group.empty
            else np.nan,
        })

    return pd.DataFrame(
        rows
    )


def build_sampling_summary(
    sampled: pd.DataFrame,
    strata,
) -> pd.DataFrame:
    rows = []

    for stratum in strata:
        group = sampled.loc[
            sampled[
                "sampling_stratum"
            ] == stratum
        ]

        rows.append({
            "sampling_stratum": (
                stratum
            ),

            "n_sampled_tiles": int(
                len(group)
            ),

            "n_patients_sampled": int(
                group[
                    "patient_id"
                ].nunique()
            )
            if not group.empty
            else 0,

            "n_slides_sampled": int(
                group[
                    "prefix"
                ].nunique()
            )
            if not group.empty
            else 0,

            "median_pc3_score": float(
                group[
                    "pc3_score"
                ].median()
            )
            if not group.empty
            else np.nan,

            "median_pc3_within_slide_percentile": float(
                group[
                    "pc3_within_slide_percentile"
                ].median()
            )
            if not group.empty
            else np.nan,

            "median_tumor_overlap_fraction": float(
                group[
                    "tumor_overlap_fraction"
                ].median()
            )
            if not group.empty
            else np.nan,

            "minimum_tumor_overlap_fraction": float(
                group[
                    "tumor_overlap_fraction"
                ].min()
            )
            if not group.empty
            else np.nan,
        })

    return pd.DataFrame(
        rows
    )


def build_slide_summary(
    sampled: pd.DataFrame,
) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame()

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
            n_sampled_tiles=(
                "sensitivity_sample_id",
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
    if sampled.empty:
        return pd.DataFrame()

    return (
        sampled.groupby(
            [
                "sampling_stratum",
                "patient_id",
            ],
            as_index=False,
        )
        .agg(
            n_sampled_tiles=(
                "sensitivity_sample_id",
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


def make_readme_text(
    candidate_summary: pd.DataFrame,
    sampling_summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "PC3 Tumor-Core Sensitivity Sampling",
            "===================================",
            "",
            "Purpose",
            "-------",
            (
                "Extend the step-13a morphology sampling framework with a "
                "strict tumor-core sensitivity threshold."
            ),
            "",
            "Thresholds",
            "----------",
            (
                f"Primary tumor-associated threshold: "
                f"{PRIMARY_TUMOR_THRESHOLD:.2f}"
            ),
            (
                f"Tumor-core threshold: "
                f"{TUMOR_CORE_THRESHOLD:.2f}"
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
            "Important",
            "---------",
            (
                "The original step-13a outputs are not modified. "
                "This is a sensitivity extension only."
            ),
            "",
            "Candidate summary",
            "-----------------",
            candidate_summary.to_string(
                index=False
            ),
            "",
            "Balanced sampling summary",
            "-------------------------",
            sampling_summary.to_string(
                index=False
            ),
            "",
        ]
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    check_file(
        IN_CANDIDATES
    )

    print(
        "=" * 110
    )
    print(
        "PC3 TUMOR-CORE SENSITIVITY SAMPLING"
    )
    print(
        "=" * 110
    )

    print(
        f"[INPUT CANDIDATES]        "
        f"{IN_CANDIDATES}"
    )

    print(
        f"[PRIMARY TUMOR THRESHOLD] "
        f"{PRIMARY_TUMOR_THRESHOLD:.2f}"
    )

    print(
        f"[TUMOR-CORE THRESHOLD]    "
        f"{TUMOR_CORE_THRESHOLD:.2f}"
    )

    print(
        f"[MAX TILES/SLIDE]         "
        f"{MAX_TILES_PER_SLIDE_PER_STRATUM}"
    )

    print(
        f"[MAX TILES/PATIENT]       "
        f"{MAX_TILES_PER_PATIENT_PER_STRATUM}"
    )

    print(
        f"[OUTPUT]                  "
        f"{OUT_DIR}"
    )

    all_candidates = pd.read_csv(
        IN_CANDIDATES
    )

    all_candidates = validate_candidates(
        all_candidates
    )

    strata = [
        "all_tissue_low",
        "all_tissue_high",
        "tumor_only_low",
        "tumor_only_high",
        "tumor_core_low",
        "tumor_core_high",
    ]

    candidate_summary = (
        build_candidate_summary(
            all_candidates,
            strata,
        )
    )

    sampled_blocks = []

    for stratum in strata:
        candidates = get_stratum_candidates(
            all_candidates,
            stratum,
        )

        sampled = sample_balanced_stratum(
            candidates,
            stratum,
        )

        if sampled.empty:
            print(
                f"[WARNING] No balanced samples "
                f"for {stratum}"
            )
            continue

        sampled_blocks.append(
            sampled
        )

        print(
            f"[SAMPLED] "
            f"{stratum}: "
            f"{len(sampled)} tiles | "
            f"{sampled['patient_id'].nunique()} patients | "
            f"{sampled['prefix'].nunique()} slides"
        )

    if not sampled_blocks:
        raise RuntimeError(
            "No sensitivity samples were generated."
        )

    sampled_all = pd.concat(
        sampled_blocks,
        ignore_index=True,
    )

    sampled_all = assign_sample_ids(
        sampled_all
    )

    sampled_all = round_numeric_columns(
        sampled_all
    )

    candidate_summary = round_numeric_columns(
        candidate_summary
    )

    sampling_summary = (
        build_sampling_summary(
            sampled_all,
            strata,
        )
    )

    sampling_summary = round_numeric_columns(
        sampling_summary
    )

    slide_summary = (
        build_slide_summary(
            sampled_all
        )
    )

    slide_summary = round_numeric_columns(
        slide_summary
    )

    patient_summary = (
        build_patient_summary(
            sampled_all
        )
    )

    patient_summary = round_numeric_columns(
        patient_summary
    )

    sampled_all.to_csv(
        OUT_MANIFEST,
        index=False,
    )

    candidate_summary.to_csv(
        OUT_CANDIDATE_SUMMARY,
        index=False,
    )

    sampling_summary.to_csv(
        OUT_SAMPLING_SUMMARY,
        index=False,
    )

    slide_summary.to_csv(
        OUT_SLIDE_SUMMARY,
        index=False,
    )

    patient_summary.to_csv(
        OUT_PATIENT_SUMMARY,
        index=False,
    )

    metadata = {
        "analysis_name": (
            "PC3 tumor-core sensitivity sampling"
        ),

        "analysis_role": (
            "Strict tumor-purity sensitivity extension "
            "of frozen PC3 morphology sampling"
        ),

        "input_candidate_table": str(
            IN_CANDIDATES
        ),

        "primary_tumor_threshold": float(
            PRIMARY_TUMOR_THRESHOLD
        ),

        "tumor_core_threshold": float(
            TUMOR_CORE_THRESHOLD
        ),

        "sampling_caps": {
            "max_tiles_per_slide_per_stratum": int(
                MAX_TILES_PER_SLIDE_PER_STRATUM
            ),
            "max_tiles_per_patient_per_stratum": int(
                MAX_TILES_PER_PATIENT_PER_STRATUM
            ),
        },

        "strata": strata,

        "modifies_step13a_outputs": False,

        "uses_morphology_for_selection": False,

        "uses_pathologist_review_for_selection": False,

        "uses_external_cohorts": False,

        "outputs": {
            "sampling_manifest": str(
                OUT_MANIFEST
            ),
            "candidate_summary": str(
                OUT_CANDIDATE_SUMMARY
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
            candidate_summary,
            sampling_summary,
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "CANDIDATE COVERAGE SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        candidate_summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "BALANCED SAMPLING SUMMARY"
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
        OUT_MANIFEST,
        OUT_CANDIDATE_SUMMARY,
        OUT_SAMPLING_SUMMARY,
        OUT_SLIDE_SUMMARY,
        OUT_PATIENT_SUMMARY,
        OUT_METADATA,
        OUT_README,
    ]:
        print(
            f"  {path}"
        )

    print(
        "\n[DONE] PC3 tumor-core sensitivity sampling complete."
    )


if __name__ == "__main__":
    main()
