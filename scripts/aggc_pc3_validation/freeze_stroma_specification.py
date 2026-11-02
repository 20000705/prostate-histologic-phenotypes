#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
17a_freeze_aggc_pc3_external_stroma_validation_specification.py

STEP 17A
========
Freeze the AGGC2022 Subset-1 PC3 external Stroma-validation specification.

ROLE
----
SPECIFICATION / FREEZE ONLY.

This script defines, validates, records, and freezes the formal analysis
design BEFORE Step 17b measures PC3 against AGGC annotations and BEFORE
any formal Test-partition PC3-label outcome is read.

It consumes ONLY technical / annotation preflight outputs.

It MUST NOT:
    - read PC3 values
    - calculate PC3-label association
    - calculate AUC
    - screen PC1-PC8
    - analyze PC6
    - calculate p-values
    - bootstrap outcomes
    - select slides by phenotype
    - select thresholds from observed PC3 effects
    - re-fit PCA
    - change AGGC masks
    - infer missing labels
    - resolve label overlap post hoc

DESIGN
------
Dataset:
    AGGC2022 Subset 1 only
    Whole-mount prostatectomy images

Partitions:
    TRAIN = formal external validation partition
    TEST  = separately analyzed replication partition

Expected technical cohorts from preflight:
    TRAIN Stroma-ready: 92
    TEST  Stroma-ready: 42

Frozen latent variable:
    discovery-cohort frozen PC3
    PC index = 2 (zero-based)

Frozen primary annotation endpoint:
    stroma_fraction_among_annotated_pixels

For every Virchow2 tile:
    annotation_union =
        Normal OR Stroma OR G3 OR G4 OR G5

    labeled_fraction =
        pixels(annotation_union) / tile_pixels

    stroma_fraction_among_annotated_pixels =
        pixels(Stroma) / pixels(annotation_union)

Unannotated pixels are excluded from the denominator.

This is critical because AGGC annotations are sparse positive annotation
layers. Pixels outside Stroma_Mask.tif are NOT automatically treated as
non-Stroma.

Primary tile QC:
    labeled_fraction >= 0.10

Primary slide-level statistic:
    Spearman rho(
        frozen PC3,
        stroma_fraction_among_annotated_pixels
    )

Formal independent unit:
    slide

Expected direction:
    rho > 0

Minimum usable annotated tiles per slide:
    50

TRAIN cohort inference:
    - distribution of slide-level rho
    - median rho
    - 95% slide-bootstrap CI of the median
    - two-sided Wilcoxon signed-rank test vs 0
    - direction consistency: fraction/slides with rho > 0

TEST replication:
    EXACT SAME measurement and inferential specification.
    No threshold or endpoint changes are permitted after Train results.

Supportive analysis:
    Within-slide frozen PC3 extreme comparison:
        PC3 lowest 10% vs highest 10%
    endpoint:
        median stroma_fraction_among_annotated_pixels
    effect:
        high-PC3 minus low-PC3
    expected direction:
        > 0

Overlap sensitivity:
    repeat primary slide-level Spearman after removing every tile that
    contains any pixel assigned to >=2 official AGGC annotation masks.

Important:
    overlap sensitivity is supportive, not primary.

Secondary descriptive characterization:
    Normal / G3 / G4 / G5 fractions among annotated pixels may be measured
    and summarized, but they are NOT confirmatory primary endpoints and
    must not be used to re-select the primary hypothesis.

No pooled tile-level p-values are permitted.
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

from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.1"
SCHEMA_VERSION = "17a-aggc-pc3-external-stroma-schema-1.0.1"


# =============================================================================
# PATHS
# =============================================================================

MEDIA_BASE = REPO_ROOT

AGGC_BASE = CONFIG_AGGC_ROOT

TRAIN_PREFLIGHT_DIR = (
    AGGC_BASE
    / "step17_preflight_aggc_subset1_audit"
)

TEST_PREFLIGHT_DIR = (
    AGGC_BASE
    / "step17_preflight_aggc_subset1_test_audit"
)

TRAIN_ELIGIBILITY = (
    TRAIN_PREFLIGHT_DIR
    / "aggc_subset1_prefreeze_eligibility_manifest.csv"
)

TRAIN_ARRAY_AUDIT = (
    TRAIN_PREFLIGHT_DIR
    / "aggc_subset1_embedding_pca_audit.csv"
)

TRAIN_MASK_MANIFEST = (
    TRAIN_PREFLIGHT_DIR
    / "aggc_subset1_mask_manifest.csv"
)

TRAIN_GEOMETRY = (
    TRAIN_PREFLIGHT_DIR
    / "aggc_subset1_geometry_audit.csv"
)

TRAIN_ENCODING = (
    TRAIN_PREFLIGHT_DIR
    / "aggc_subset1_mask_encoding_audit.csv"
)

TRAIN_SUMMARY = (
    TRAIN_PREFLIGHT_DIR
    / "aggc_subset1_prefreeze_summary.csv"
)

TRAIN_PREFLIGHT_METADATA = (
    TRAIN_PREFLIGHT_DIR
    / "analysis_metadata.json"
)

TEST_ELIGIBILITY = (
    TEST_PREFLIGHT_DIR
    / "aggc_subset1_test_prefreeze_eligibility_manifest.csv"
)

TEST_ARRAY_AUDIT = (
    TEST_PREFLIGHT_DIR
    / "aggc_subset1_test_embedding_pca_audit.csv"
)

TEST_MASK_MANIFEST = (
    TEST_PREFLIGHT_DIR
    / "aggc_subset1_test_mask_manifest.csv"
)

TEST_GEOMETRY = (
    TEST_PREFLIGHT_DIR
    / "aggc_subset1_test_geometry_audit.csv"
)

TEST_ENCODING = (
    TEST_PREFLIGHT_DIR
    / "aggc_subset1_test_mask_encoding_audit.csv"
)

TEST_SUMMARY = (
    TEST_PREFLIGHT_DIR
    / "aggc_subset1_test_prefreeze_summary.csv"
)

TEST_PREFLIGHT_METADATA = (
    TEST_PREFLIGHT_DIR
    / "analysis_metadata.json"
)

OUT_DIR = (
    RESULTS_ROOT
    / "17a_aggc_pc3_external_stroma_specification"
)

OUT_TRAIN_COHORT = (
    OUT_DIR
    / "aggc_pc3_train_validation_cohort.csv"
)

OUT_TEST_COHORT = (
    OUT_DIR
    / "aggc_pc3_test_replication_cohort.csv"
)

OUT_COMBINED_COHORT = (
    OUT_DIR
    / "aggc_pc3_frozen_partition_manifest.csv"
)

OUT_ANALYSIS_PLAN = (
    OUT_DIR
    / "aggc_pc3_external_stroma_analysis_plan.json"
)

OUT_LABEL_SPEC = (
    OUT_DIR
    / "aggc_pc3_external_stroma_label_specification.json"
)

OUT_GEOMETRY_SPEC = (
    OUT_DIR
    / "aggc_pc3_external_stroma_geometry_specification.json"
)

OUT_COHORT_SUMMARY = (
    OUT_DIR
    / "aggc_pc3_external_stroma_cohort_summary.csv"
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


# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

DATASET = "AGGC2022"
SUBSET = "Subset1"

EXPECTED_TRAIN_TOTAL = 105
EXPECTED_TEST_TOTAL = 45

EXPECTED_TRAIN_STROMA_READY = 92
EXPECTED_TEST_STROMA_READY = 42

KNOWN_CURRENT_TECHNICAL_EXCLUSION = "Subset1_Train_96"

TARGET_PC = "PC3"
TARGET_PC_ZERO_BASED_INDEX = 2

PATCH_SIZE = 224

OFFICIAL_LABELS = [
    "Normal",
    "Stroma",
    "G3",
    "G4",
    "G5",
]

PRIMARY_LABEL = "Stroma"

PRIMARY_ENDPOINT = (
    "stroma_fraction_among_annotated_pixels"
)

PRIMARY_LABELED_FRACTION_THRESHOLD = 0.10

MIN_PRIMARY_TILES_PER_SLIDE = 50

PRIMARY_ASSOCIATION = "spearman"

PRIMARY_EXPECTED_DIRECTION = (
    "PC3 higher -> Stroma fraction among explicitly annotated pixels higher"
)

PRIMARY_EXPECTED_SIGN = "positive"

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260807
BOOTSTRAP_LEVEL = 0.95

WILCOXON_ALTERNATIVE = "two-sided"

EXTREME_QUANTILE = 0.10

OVERLAP_SENSITIVITY = True

OVERWRITE = False


# =============================================================================
# HELPERS
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {label}: {path}"
        )
    if not path.is_file():
        raise RuntimeError(
            f"Expected file for {label}: {path}"
        )


def require_dir(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {label}: {path}"
        )
    if not path.is_dir():
        raise RuntimeError(
            f"Expected directory for {label}: {path}"
        )


def prepare_output():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        if not OVERWRITE:
            raise RuntimeError(
                "Step 17a output already exists and is non-empty:\n"
                f"{OUT_DIR}\n\n"
                "This is a frozen specification. "
                "Do not overwrite casually. "
                "If a new specification is scientifically necessary, "
                "create a new version explicitly."
            )

        for path in OUT_DIR.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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


def bool_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda x:
            bool(x)
            if pd.notna(x)
            else False
    )


def extract_summary_value(
    summary_df: pd.DataFrame,
    metric: str,
):
    row = summary_df.loc[
        summary_df["metric"].eq(metric)
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly one summary row for metric={metric}, "
            f"found {len(row)}"
        )

    return row.iloc[0]["value"]


def validate_preflight_metadata(
    path: Path,
    expected_partition: str,
):

    meta = load_json(path)

    if meta.get("pc_values_read") is not False:
        raise RuntimeError(
            f"Preflight metadata does not confirm PC values unread: {path}"
        )

    if meta.get("pc3_association_calculated") is not False:
        raise RuntimeError(
            f"Preflight metadata indicates PC3 association was calculated: {path}"
        )

    if meta.get("pc6_analyzed") is not False:
        raise RuntimeError(
            f"Preflight metadata indicates PC6 analysis occurred: {path}"
        )

    if meta.get("pc1_pc8_screening_performed") is not False:
        raise RuntimeError(
            f"Preflight metadata indicates PC screening occurred: {path}"
        )

    if expected_partition == "test":
        if meta.get("partition") != "test":
            raise RuntimeError(
                "Test preflight metadata does not identify partition=test."
            )

    return meta


def build_frozen_partition(
    partition: str,
    eligibility_path: Path,
    array_path: Path,
    mask_path: Path,
    geometry_path: Path,
    encoding_path: Path,
) -> pd.DataFrame:

    eligibility = pd.read_csv(
        eligibility_path,
        low_memory=False,
    )

    arrays = pd.read_csv(
        array_path,
        low_memory=False,
    )

    masks = pd.read_csv(
        mask_path,
        low_memory=False,
    )

    geometry = pd.read_csv(
        geometry_path,
        low_memory=False,
    )

    encoding = pd.read_csv(
        encoding_path,
        low_memory=False,
    )

    # -------------------------------------------------------------------------
    # Exact Stroma mask rows
    # -------------------------------------------------------------------------

    stroma_masks = (
        masks.loc[
            masks["label"].eq("Stroma")
        ]
        .copy()
    )

    if stroma_masks["slide_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate Stroma mask rows in {partition} preflight."
        )

    stroma_masks = stroma_masks[
        [
            "slide_id",
            "mask_exists",
            "mask_path",
        ]
    ].rename(
        columns={
            "mask_exists":
                "stroma_mask_exists_from_manifest",
            "mask_path":
                "stroma_mask_path",
        }
    )

    # -------------------------------------------------------------------------
    # Exact Stroma geometry rows
    # -------------------------------------------------------------------------

    stroma_geometry = (
        geometry.loc[
            geometry["label"].eq("Stroma")
        ]
        .copy()
    )

    if stroma_geometry["slide_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate Stroma geometry rows in {partition} preflight."
        )

    stroma_geometry = stroma_geometry[
        [
            "slide_id",
            "wsi_width",
            "wsi_height",
            "mask_width",
            "mask_height",
            "scale_x",
            "scale_y",
            "same_dimensions",
            "direct_geometry_compatible",
        ]
    ]

    # -------------------------------------------------------------------------
    # Exact Stroma encoding rows
    # -------------------------------------------------------------------------

    stroma_encoding = (
        encoding.loc[
            encoding["label"].eq("Stroma")
        ]
        .copy()
    )

    if stroma_encoding["slide_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate Stroma encoding rows in {partition} preflight."
        )

    stroma_encoding = stroma_encoding[
        [
            "slide_id",
            "mode",
            "binary_compatible",
            "binary_compatibility_source",
            "polarity_rule",
            "tifffile_decode_status",
        ]
    ]

    # -------------------------------------------------------------------------
    # Array audit columns
    # -------------------------------------------------------------------------

    arrays_small = arrays[
        [
            "slide_id",
            "embedding_exists",
            "coords_exists",
            "meta_exists",
            "pca_exists",
            "embedding_path",
            "coords_path",
            "meta_path",
            "pca_path",
            "embedding_shape",
            "coords_shape",
            "pca_shape",
            "row_count_consistent",
            "pca_has_at_least_3_components",
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    df = eligibility.merge(
        arrays_small,
        on="slide_id",
        how="left",
        suffixes=("", "_array"),
        validate="one_to_one",
    )

    df = df.merge(
        stroma_masks,
        on="slide_id",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        stroma_geometry,
        on="slide_id",
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        stroma_encoding,
        on="slide_id",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Formal technical inclusion rule
    # -------------------------------------------------------------------------

    wsi_ok = bool_series(
        df["wsi_geometry_ok"]
    )

    emb_ok = bool_series(
        df["embedding_exists"]
    )

    coords_ok = bool_series(
        df["coords_exists"]
    )

    pca_ok = bool_series(
        df["pca_exists"]
    )

    rows_ok = bool_series(
        df["row_count_consistent"]
    )

    pc3_shape_ok = bool_series(
        df["pca_has_at_least_3_components"]
    )

    stroma_exists = bool_series(
        df["stroma_mask_exists_from_manifest"]
    )

    stroma_binary = bool_series(
        df["binary_compatible"]
    )

    stroma_geometry_ok = bool_series(
        df["direct_geometry_compatible"]
    )

    formal_include = (
        wsi_ok
        &
        emb_ok
        &
        coords_ok
        &
        pca_ok
        &
        rows_ok
        &
        pc3_shape_ok
        &
        stroma_exists
        &
        stroma_binary
        &
        stroma_geometry_ok
    )

    df["partition"] = partition

    df["formal_step17a_include"] = formal_include

    reasons: List[str] = []

    for i in range(len(df)):

        fail = []

        if not wsi_ok.iloc[i]:
            fail.append("wsi_geometry_not_readable")

        if not emb_ok.iloc[i]:
            fail.append("embedding_missing")

        if not coords_ok.iloc[i]:
            fail.append("coords_missing")

        if not pca_ok.iloc[i]:
            fail.append("frozen_pca_missing")

        if not rows_ok.iloc[i]:
            fail.append("array_row_count_inconsistent")

        if not pc3_shape_ok.iloc[i]:
            fail.append("pca_does_not_contain_pc3")

        if not stroma_exists.iloc[i]:
            fail.append("official_stroma_mask_missing")

        if not stroma_binary.iloc[i]:
            fail.append("stroma_mask_not_binary_compatible")

        if not stroma_geometry_ok.iloc[i]:
            fail.append("stroma_geometry_not_directly_compatible")

        reasons.append(
            "INCLUDE"
            if not fail
            else "|".join(fail)
        )

    df["formal_step17a_reason"] = reasons

    # -------------------------------------------------------------------------
    # Geometry invariants for included slides
    # -------------------------------------------------------------------------

    included = df.loc[
        df["formal_step17a_include"]
    ].copy()

    if len(included):

        if not bool_series(
            included["same_dimensions"]
        ).all():
            raise RuntimeError(
                f"{partition}: included cohort contains non-1:1 mask geometry."
            )

        sx = pd.to_numeric(
            included["scale_x"],
            errors="coerce",
        )

        sy = pd.to_numeric(
            included["scale_y"],
            errors="coerce",
        )

        if not np.allclose(
            sx.to_numpy(),
            1.0,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{partition}: included Stroma scale_x is not exactly 1."
            )

        if not np.allclose(
            sy.to_numpy(),
            1.0,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{partition}: included Stroma scale_y is not exactly 1."
            )

        # CSV round-tripping may deserialize Pillow mode "1" as numeric
        # 1 or 1.0.  Normalize robustly instead of requiring the literal
        # string "1" after pandas CSV parsing.
        mode_raw = included["mode"]
        mode_numeric = pd.to_numeric(
            mode_raw,
            errors="coerce",
        )
        mode_string = (
            mode_raw
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mode_is_one = (
            mode_numeric.eq(1.0)
            | mode_string.isin({"1", "1.0"})
        )

        if not mode_is_one.all():
            bad_modes = included.loc[
                ~mode_is_one,
                ["slide_id", "mode", "binary_compatible",
                 "binary_compatibility_source"],
            ]
            raise RuntimeError(
                f"{partition}: included Stroma masks are not all Pillow mode 1 "
                "after robust CSV normalization. Offending rows:\n"
                + bad_modes.to_string(index=False)
            )

    # -------------------------------------------------------------------------
    # Frozen measurement constants
    # -------------------------------------------------------------------------

    df["frozen_target_pc"] = TARGET_PC
    df["frozen_target_pc_zero_based_index"] = TARGET_PC_ZERO_BASED_INDEX

    df["frozen_patch_size"] = PATCH_SIZE

    df["frozen_primary_endpoint"] = PRIMARY_ENDPOINT

    df["frozen_primary_labeled_fraction_threshold"] = (
        PRIMARY_LABELED_FRACTION_THRESHOLD
    )

    df["frozen_min_primary_tiles_per_slide"] = (
        MIN_PRIMARY_TILES_PER_SLIDE
    )

    df["frozen_expected_direction"] = (
        PRIMARY_EXPECTED_SIGN
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print("STEP 17A: FREEZE AGGC PC3 EXTERNAL STROMA VALIDATION")
    print("=" * 120)

    print("[VERSION]        ", VERSION)
    print("[SCHEMA]         ", SCHEMA_VERSION)
    print("[TRAIN PREFLIGHT]", TRAIN_PREFLIGHT_DIR)
    print("[TEST PREFLIGHT] ", TEST_PREFLIGHT_DIR)
    print("[OUTPUT]         ", OUT_DIR)
    print("[OVERWRITE]      ", OVERWRITE)

    print()
    print("[PC VALUES]      NOT READ")
    print("[INFERENCE]      NONE")
    print("[PCA REFIT]      NONE")
    print("[PC SCREENING]   NONE")
    print("[TEST OUTCOME]   NOT READ")

    # =========================================================================
    # Preconditions
    # =========================================================================

    require_dir(
        TRAIN_PREFLIGHT_DIR,
        "Train preflight directory",
    )

    require_dir(
        TEST_PREFLIGHT_DIR,
        "Test preflight directory",
    )

    required_files = [
        TRAIN_ELIGIBILITY,
        TRAIN_ARRAY_AUDIT,
        TRAIN_MASK_MANIFEST,
        TRAIN_GEOMETRY,
        TRAIN_ENCODING,
        TRAIN_SUMMARY,
        TRAIN_PREFLIGHT_METADATA,
        TEST_ELIGIBILITY,
        TEST_ARRAY_AUDIT,
        TEST_MASK_MANIFEST,
        TEST_GEOMETRY,
        TEST_ENCODING,
        TEST_SUMMARY,
        TEST_PREFLIGHT_METADATA,
    ]

    for path in required_files:
        require_file(
            path,
            path.name,
        )

    train_meta = validate_preflight_metadata(
        TRAIN_PREFLIGHT_METADATA,
        expected_partition="train",
    )

    test_meta = validate_preflight_metadata(
        TEST_PREFLIGHT_METADATA,
        expected_partition="test",
    )

    # =========================================================================
    # Validate preflight cohort sizes
    # =========================================================================

    train_summary = pd.read_csv(
        TRAIN_SUMMARY,
        low_memory=False,
    )

    test_summary = pd.read_csv(
        TEST_SUMMARY,
        low_memory=False,
    )

    train_total = int(
        float(
            extract_summary_value(
                train_summary,
                "n_subset1_wsis_discovered",
            )
        )
    )

    test_total = int(
        float(
            extract_summary_value(
                test_summary,
                "n_subset1_test_wsis_discovered",
            )
        )
    )

    train_ready_preflight = int(
        float(
            extract_summary_value(
                train_summary,
                "n_wsis_preliminary_primary_stroma_ready",
            )
        )
    )

    test_ready_preflight = int(
        float(
            extract_summary_value(
                test_summary,
                "n_test_wsis_preliminary_replication_stroma_ready",
            )
        )
    )

    if train_total != EXPECTED_TRAIN_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_TOTAL} Train WSIs, "
            f"preflight reports {train_total}."
        )

    if test_total != EXPECTED_TEST_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_TOTAL} Test WSIs, "
            f"preflight reports {test_total}."
        )

    if train_ready_preflight != EXPECTED_TRAIN_STROMA_READY:
        raise RuntimeError(
            "Train preflight Stroma-ready count changed: "
            f"expected {EXPECTED_TRAIN_STROMA_READY}, "
            f"found {train_ready_preflight}."
        )

    if test_ready_preflight != EXPECTED_TEST_STROMA_READY:
        raise RuntimeError(
            "Test preflight Stroma-ready count changed: "
            f"expected {EXPECTED_TEST_STROMA_READY}, "
            f"found {test_ready_preflight}."
        )

    # =========================================================================
    # Freeze output
    # =========================================================================

    prepare_output()

    train_df = build_frozen_partition(
        partition="train_validation",
        eligibility_path=TRAIN_ELIGIBILITY,
        array_path=TRAIN_ARRAY_AUDIT,
        mask_path=TRAIN_MASK_MANIFEST,
        geometry_path=TRAIN_GEOMETRY,
        encoding_path=TRAIN_ENCODING,
    )

    test_df = build_frozen_partition(
        partition="test_replication",
        eligibility_path=TEST_ELIGIBILITY,
        array_path=TEST_ARRAY_AUDIT,
        mask_path=TEST_MASK_MANIFEST,
        geometry_path=TEST_GEOMETRY,
        encoding_path=TEST_ENCODING,
    )

    train_n = int(
        train_df["formal_step17a_include"].sum()
    )

    test_n = int(
        test_df["formal_step17a_include"].sum()
    )

    if train_n != EXPECTED_TRAIN_STROMA_READY:
        raise RuntimeError(
            f"Formal Train inclusion={train_n}, "
            f"expected={EXPECTED_TRAIN_STROMA_READY}."
        )

    if test_n != EXPECTED_TEST_STROMA_READY:
        raise RuntimeError(
            f"Formal Test inclusion={test_n}, "
            f"expected={EXPECTED_TEST_STROMA_READY}."
        )

    # Current known Train technical exclusion must remain objective.
    row_96 = train_df.loc[
        train_df["slide_id"].eq(
            KNOWN_CURRENT_TECHNICAL_EXCLUSION
        )
    ]

    if len(row_96) != 1:
        raise RuntimeError(
            f"Could not uniquely locate {KNOWN_CURRENT_TECHNICAL_EXCLUSION}."
        )

    if bool(
        row_96.iloc[0]["formal_step17a_include"]
    ):
        raise RuntimeError(
            f"{KNOWN_CURRENT_TECHNICAL_EXCLUSION} unexpectedly included."
        )

    # Historical Train_94 must not be automatically excluded if it now passes.
    row_94 = train_df.loc[
        train_df["slide_id"].eq(
            "Subset1_Train_94"
        )
    ]

    if len(row_94) == 1:
        if not bool(
            row_94.iloc[0]["formal_step17a_include"]
        ):
            print(
                "[WARN] Subset1_Train_94 does not meet current formal "
                "technical inclusion despite historical recovery."
            )

    # =========================================================================
    # Save partition manifests
    # =========================================================================

    train_df.to_csv(
        OUT_TRAIN_COHORT,
        index=False,
    )

    test_df.to_csv(
        OUT_TEST_COHORT,
        index=False,
    )

    combined = pd.concat(
        [
            train_df,
            test_df,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        OUT_COMBINED_COHORT,
        index=False,
    )

    # =========================================================================
    # Frozen label specification
    # =========================================================================

    label_spec = {

        "dataset":
            DATASET,

        "subset":
            SUBSET,

        "official_labels":
            OFFICIAL_LABELS,

        "primary_label":
            PRIMARY_LABEL,

        "mask_foreground_definition":
            "Pillow Boolean True",

        "mask_binary_definition":
            "Pillow mode == '1'",

        "annotation_semantics":
            {
                "sparse_positive_annotations":
                    True,

                "unannotated_pixels_are_negative":
                    False,

                "missing_class_mask_means_class_absent":
                    False,

                "outside_stroma_means_non_stroma":
                    False,

                "outside_normal_means_stroma":
                    False,

                "mask_overlap_allowed":
                    True,
            },

        "tile_level_quantities":
            {
                "annotation_union":
                    "pixelwise OR of Normal, Stroma, G3, G4, G5 masks that exist for that slide",

                "tile_pixels":
                    PATCH_SIZE * PATCH_SIZE,

                "labeled_fraction":
                    "annotation_union_positive_pixels / tile_pixels",

                "stroma_fraction_among_annotated_pixels":
                    "Stroma_positive_pixels / annotation_union_positive_pixels",

                "normal_fraction_among_annotated_pixels":
                    "Normal_positive_pixels / annotation_union_positive_pixels",

                "g3_fraction_among_annotated_pixels":
                    "G3_positive_pixels / annotation_union_positive_pixels",

                "g4_fraction_among_annotated_pixels":
                    "G4_positive_pixels / annotation_union_positive_pixels",

                "g5_fraction_among_annotated_pixels":
                    "G5_positive_pixels / annotation_union_positive_pixels",
            },

        "overlap_policy_primary":
            (
                "Retain official overlapping annotations. "
                "The denominator is the annotation UNION, not the sum of class pixels."
            ),

        "overlap_policy_sensitivity":
            (
                "Exclude any tile containing >=1 pixel positive in two or more "
                "official annotation masks, then repeat the primary slide-level association."
            ),
    }

    write_json(
        OUT_LABEL_SPEC,
        label_spec,
    )

    # =========================================================================
    # Frozen geometry specification
    # =========================================================================

    geometry_spec = {

        "coordinate_space":
            "native level-0 WSI pixel coordinates",

        "tile_coordinate_definition":
            "Virchow2 tile top-left (x, y)",

        "patch_size_pixels":
            PATCH_SIZE,

        "mask_to_wsi_scale_x":
            1.0,

        "mask_to_wsi_scale_y":
            1.0,

        "mask_to_wsi_offset_x":
            0,

        "mask_to_wsi_offset_y":
            0,

        "resampling_for_primary_measurement":
            "none",

        "geometry_rule":
            (
                "Formal inclusion requires exact WSI-mask width and height equality "
                "for Stroma_Mask.tif."
            ),

        "silent_geometry_correction_permitted":
            False,
    }

    write_json(
        OUT_GEOMETRY_SPEC,
        geometry_spec,
    )

    # =========================================================================
    # Frozen formal analysis plan
    # =========================================================================

    analysis_plan = {

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "dataset":
            DATASET,

        "subset":
            SUBSET,

        "specimen_type":
            "whole-mount prostatectomy",

        "latent_representation":
            {
                "target":
                    TARGET_PC,

                "zero_based_component_index":
                    TARGET_PC_ZERO_BASED_INDEX,

                "pca_space":
                    "discovery-cohort frozen PCA space",

                "pca_refit_on_aggc":
                    False,

                "pc_screening_in_step17":
                    False,
            },

        "partition_roles":
            {
                "train":
                    "formal external validation",

                "test":
                    "separately analyzed replication using identical frozen specification",

                "pooled_train_test":
                    "supportive summary only; not primary inferential analysis",
            },

        "frozen_cohort_sizes":
            {
                "train_total_wsis":
                    EXPECTED_TRAIN_TOTAL,

                "train_stroma_ready":
                    train_n,

                "test_total_wsis":
                    EXPECTED_TEST_TOTAL,

                "test_stroma_ready":
                    test_n,

                "combined_stroma_ready":
                    train_n + test_n,
            },

        "primary_endpoint":
            PRIMARY_ENDPOINT,

        "primary_endpoint_definition":
            (
                "Within each QC-passing tile, Stroma-positive pixels divided by "
                "the union of all explicitly annotated class pixels in that tile."
            ),

        "primary_tile_qc":
            {
                "minimum_labeled_fraction":
                    PRIMARY_LABELED_FRACTION_THRESHOLD,

                "labeled_fraction_denominator":
                    "all tile pixels",

                "labeled_fraction_numerator":
                    "union of official Normal/Stroma/G3/G4/G5 annotation pixels",

                "unannotated_pixels_in_primary_endpoint_denominator":
                    False,
            },

        "primary_slide_qc":
            {
                "minimum_qc_passing_tiles":
                    MIN_PRIMARY_TILES_PER_SLIDE,

                "slides_below_minimum":
                    "excluded from formal slide-level association with reason recorded; no replacement threshold permitted",
            },

        "primary_association":
            {
                "within_slide_statistic":
                    "Spearman rank correlation",

                "x":
                    TARGET_PC,

                "y":
                    PRIMARY_ENDPOINT,

                "expected_sign":
                    PRIMARY_EXPECTED_SIGN,

                "expected_direction_text":
                    PRIMARY_EXPECTED_DIRECTION,
            },

        "formal_independent_unit":
            "slide",

        "train_inference":
            {
                "cohort_summary":
                    "median of slide-level Spearman rho",

                "bootstrap_unit":
                    "slide",

                "bootstrap_replicates":
                    BOOTSTRAP_REPLICATES,

                "bootstrap_seed":
                    BOOTSTRAP_SEED,

                "bootstrap_ci_level":
                    BOOTSTRAP_LEVEL,

                "bootstrap_statistic":
                    "median slide-level rho",

                "wilcoxon":
                    {
                        "test":
                            "Wilcoxon signed-rank vs zero",

                        "alternative":
                            WILCOXON_ALTERNATIVE,
                    },

                "direction_consistency":
                    "number and fraction of usable slides with rho > 0",
            },

        "test_replication":
            {
                "analysis_identical_to_train":
                    True,

                "endpoint_change_after_train_results":
                    False,

                "threshold_change_after_train_results":
                    False,

                "tile_rule_change_after_train_results":
                    False,

                "expected_direction_change_after_train_results":
                    False,

                "formal_replication_criterion_reporting":
                    (
                        "Report Test median slide-level rho, 95% slide-bootstrap CI, "
                        "Wilcoxon signed-rank vs zero, and direction consistency. "
                        "No new pass/fail threshold is introduced in Step 17a."
                    ),
            },

        "supportive_extreme_analysis":
            {
                "enabled":
                    True,

                "within_slide_quantile":
                    EXTREME_QUANTILE,

                "low_group":
                    "lowest 10% frozen PC3 among primary QC-passing tiles",

                "high_group":
                    "highest 10% frozen PC3 among primary QC-passing tiles",

                "endpoint":
                    PRIMARY_ENDPOINT,

                "slide_effect":
                    "median(high-PC3 endpoint) - median(low-PC3 endpoint)",

                "expected_sign":
                    "positive",

                "cohort_inference":
                    (
                        "median slide-level effect, 95% slide-bootstrap CI, "
                        "two-sided Wilcoxon signed-rank vs zero"
                    ),
            },

        "supportive_overlap_sensitivity":
            {
                "enabled":
                    OVERLAP_SENSITIVITY,

                "rule":
                    (
                        "remove any tile containing one or more pixels that are "
                        "positive in >=2 official AGGC masks"
                    ),

                "repeat":
                    "primary within-slide Spearman analysis",

                "role":
                    "supportive sensitivity only",
            },

        "secondary_characterization":
            {
                "labels":
                    [
                        "Normal",
                        "G3",
                        "G4",
                        "G5",
                    ],

                "fractions":
                    [
                        "normal_fraction_among_annotated_pixels",
                        "g3_fraction_among_annotated_pixels",
                        "g4_fraction_among_annotated_pixels",
                        "g5_fraction_among_annotated_pixels",
                    ],

                "role":
                    (
                        "descriptive / secondary characterization only; "
                        "not used to redefine the primary Stroma hypothesis"
                    ),

                "confirmatory_multiplicity_claim":
                    False,
            },

        "prohibited_in_step17b_17c":
            [
                "re-fit PCA",
                "flip PC3 sign based on AGGC results",
                "screen PC1-PC8",
                "select PC6",
                "change 10% labeled-fraction threshold",
                "change minimum tile count after seeing association",
                "treat unannotated pixels as non-Stroma",
                "pool millions of tiles as independent observations for p-values",
                "merge Train and Test into the primary inferential unit",
                "change Test endpoint after inspecting Train/Test outcome",
                "manually exclude slides because their PC3-Stroma association is weak",
            ],
    }

    write_json(
        OUT_ANALYSIS_PLAN,
        analysis_plan,
    )

    # =========================================================================
    # Cohort summary
    # =========================================================================

    cohort_summary = pd.DataFrame(
        [
            {
                "partition":
                    "train_validation",

                "total_wsis":
                    EXPECTED_TRAIN_TOTAL,

                "formal_stroma_ready":
                    train_n,

                "not_formally_stroma_ready":
                    EXPECTED_TRAIN_TOTAL - train_n,

                "role":
                    "formal external validation",
            },

            {
                "partition":
                    "test_replication",

                "total_wsis":
                    EXPECTED_TEST_TOTAL,

                "formal_stroma_ready":
                    test_n,

                "not_formally_stroma_ready":
                    EXPECTED_TEST_TOTAL - test_n,

                "role":
                    "separate replication",
            },

            {
                "partition":
                    "combined_descriptive",

                "total_wsis":
                    EXPECTED_TRAIN_TOTAL + EXPECTED_TEST_TOTAL,

                "formal_stroma_ready":
                    train_n + test_n,

                "not_formally_stroma_ready":
                    (
                        EXPECTED_TRAIN_TOTAL
                        +
                        EXPECTED_TEST_TOTAL
                        -
                        train_n
                        -
                        test_n
                    ),

                "role":
                    "supportive descriptive only",
            },
        ]
    )

    cohort_summary.to_csv(
        OUT_COHORT_SUMMARY,
        index=False,
    )

    # =========================================================================
    # Evidence provenance
    # =========================================================================

    provenance = pd.DataFrame(
        [
            {
                "item":
                    "PC3 definition",

                "source":
                    "discovery cohort frozen PCA",

                "status":
                    "FROZEN",

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "AGGC Train cohort",

                "source":
                    str(TRAIN_PREFLIGHT_DIR),

                "status":
                    f"FROZEN n={train_n}",

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "AGGC Test cohort",

                "source":
                    str(TEST_PREFLIGHT_DIR),

                "status":
                    f"FROZEN n={test_n}",

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "Primary endpoint",

                "source":
                    "Step 17a specification",

                "status":
                    PRIMARY_ENDPOINT,

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "Primary tile QC threshold",

                "source":
                    "Step 17a specification",

                "status":
                    f"labeled_fraction >= {PRIMARY_LABELED_FRACTION_THRESHOLD:.2f}",

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "Primary association",

                "source":
                    "Step 17a specification",

                "status":
                    "within-slide Spearman rho; expected > 0",

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "Independent unit",

                "source":
                    "Step 17a specification",

                "status":
                    "slide",

                "used_AGGC_outcome":
                    False,
            },

            {
                "item":
                    "Test partition outcome",

                "source":
                    "not accessed in Step 17a",

                "status":
                    "SEALED BEFORE FORMAL ANALYSIS",

                "used_AGGC_outcome":
                    False,
            },
        ]
    )

    provenance.to_csv(
        OUT_PROVENANCE,
        index=False,
    )

    # =========================================================================
    # Hash all specification inputs
    # =========================================================================

    hash_rows = []

    for path in required_files:
        hash_rows.append(
            {
                "path":
                    str(path),

                "sha256":
                    sha256_file(path),
            }
        )

    pd.DataFrame(
        hash_rows
    ).to_csv(
        OUT_HASHES,
        index=False,
    )

    # =========================================================================
    # Metadata
    # =========================================================================

    metadata = {

        "step":
            "17a",

        "version":
            VERSION,

        "schema_version":
            SCHEMA_VERSION,

        "created_utc":
            utc_now(),

        "role":
            "freeze_only",

        "dataset":
            DATASET,

        "subset":
            SUBSET,

        "train_validation_n":
            train_n,

        "test_replication_n":
            test_n,

        "combined_stroma_ready_n":
            train_n + test_n,

        "target_pc":
            TARGET_PC,

        "pc_values_read":
            False,

        "aggc_pc3_association_calculated":
            False,

        "test_pc3_outcome_read":
            False,

        "pca_refit":
            False,

        "pc_screening":
            False,

        "pc6_analysis":
            False,

        "new_inference":
            False,

        "primary_endpoint":
            PRIMARY_ENDPOINT,

        "primary_labeled_fraction_threshold":
            PRIMARY_LABELED_FRACTION_THRESHOLD,

        "minimum_primary_tiles_per_slide":
            MIN_PRIMARY_TILES_PER_SLIDE,

        "formal_independent_unit":
            "slide",

        "expected_direction":
            PRIMARY_EXPECTED_DIRECTION,

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "pandas_version":
            pd.__version__,

        "train_preflight_metadata":
            train_meta,

        "test_preflight_metadata":
            test_meta,
    }

    write_json(
        OUT_METADATA,
        metadata,
    )

    # =========================================================================
    # README
    # =========================================================================

    readme = f"""
STEP 17A — AGGC2022 SUBSET-1 PC3 EXTERNAL STROMA VALIDATION
============================================================

Version
-------
{VERSION}

Schema
------
{SCHEMA_VERSION}


STATUS
------

FROZEN SPECIFICATION.

This step performs no PC3-label outcome analysis.


PARTITIONS
----------

TRAIN:
    role = formal external validation
    total WSIs = {EXPECTED_TRAIN_TOTAL}
    formal Stroma-ready = {train_n}

TEST:
    role = separate replication
    total WSIs = {EXPECTED_TEST_TOTAL}
    formal Stroma-ready = {test_n}

Combined Stroma-ready = {train_n + test_n}

Train and Test MUST remain separately analyzed for formal inference.


CURRENT TECHNICAL EXCLUSION
---------------------------

Subset1_Train_96 is not formally eligible because required frozen
embedding / coordinate / PCA outputs are unavailable following a
documented source-TIFF JPEG decode failure.

Historical exclusion of Subset1_Train_94 is NOT automatically retained.
Current technical readiness determines inclusion.


FROZEN PC
---------

Target:

    PC3

Source:

    discovery-cohort frozen PCA space

Zero-based column:

    2

PCA is NOT re-fit on AGGC.

PC sign is NOT changed using AGGC results.


AGGC ANNOTATION SEMANTICS
-------------------------

Official masks:

    Normal
    Stroma
    G3
    G4
    G5

These are sparse positive annotation layers.

Therefore:

    Stroma mask == 0 does NOT automatically mean non-Stroma.

    Missing Stroma mask does NOT automatically mean no Stroma.

    Unannotated tissue is not assigned a biological class.


PRIMARY TILE ENDPOINT
---------------------

For each 224 x 224 tile:

    annotation_union =
        Normal OR Stroma OR G3 OR G4 OR G5

    labeled_fraction =
        annotation_union pixels / all tile pixels

Primary QC:

    labeled_fraction >= {PRIMARY_LABELED_FRACTION_THRESHOLD:.2f}

Primary endpoint:

    stroma_fraction_among_annotated_pixels =
        Stroma-positive pixels / annotation-union pixels

Thus unannotated pixels are excluded from the biological denominator.


PRIMARY SLIDE-LEVEL ANALYSIS
----------------------------

For each formal slide with at least:

    {MIN_PRIMARY_TILES_PER_SLIDE}

QC-passing annotated tiles:

    Spearman rho(
        frozen PC3,
        stroma_fraction_among_annotated_pixels
    )

Expected direction:

    rho > 0


FORMAL INDEPENDENT UNIT
-----------------------

Slide.

Tile-level observations MUST NOT be treated as independent observations
for cohort p-values.


TRAIN INFERENCE
---------------

Across Train validation slides:

    median slide-level rho

    {int(BOOTSTRAP_LEVEL * 100)}% bootstrap CI
    bootstrap unit = slide
    bootstrap replicates = {BOOTSTRAP_REPLICATES}
    bootstrap seed = {BOOTSTRAP_SEED}

    two-sided Wilcoxon signed-rank vs zero

    direction consistency:
        number / fraction of slides with rho > 0


TEST REPLICATION
----------------

The EXACT SAME specification must be used in Test.

No endpoint change.

No threshold change.

No tile-inclusion change.

No sign change.

No PC screening.

No slide removal based on outcome.


SUPPORTIVE EXTREME ANALYSIS
---------------------------

Within each slide, among primary QC-passing tiles:

    lowest {int(EXTREME_QUANTILE * 100)}% PC3
    versus
    highest {int(EXTREME_QUANTILE * 100)}% PC3

Slide effect:

    median Stroma fraction in high-PC3 tiles
    minus
    median Stroma fraction in low-PC3 tiles

Expected:

    positive


OVERLAP SENSITIVITY
-------------------

Primary analysis retains the official annotation masks.

Because official masks may overlap, the primary denominator uses the
pixelwise annotation UNION rather than summing class-mask pixels.

Supportive sensitivity:

    remove every tile containing any pixel positive in >=2 masks

then repeat the primary slide-level Spearman analysis.


SECONDARY CHARACTERIZATION
---------------------------

Normal / G3 / G4 / G5 fractions among explicitly annotated pixels may
be summarized as secondary/descriptive characterization.

They are NOT permitted to redefine the primary Stroma hypothesis.


PROHIBITED
----------

No PCA refitting.

No PC1-PC8 screening.

No PC6 confirmatory analysis.

No AGGC-based PC3 sign flip.

No treatment of unannotated pixels as non-Stroma.

No threshold tuning after outcome inspection.

No pooled tile-level p-values.

No merging Train and Test into the primary inference.

No manual exclusion based on weak or unexpected PC3-Stroma effects.


NEXT STEP
---------

17b must be MEASUREMENT ONLY.

It may:
    read frozen PC3 values
    read frozen coordinates
    read official masks
    compute the quantities defined here

It may NOT:
    redefine masks
    redefine scale
    redefine labels
    redefine thresholds
    perform inferential testing
""".strip() + "\n"

    OUT_README.write_text(
        readme,
        encoding="utf-8",
    )

    # =========================================================================
    # Console summary
    # =========================================================================

    print()
    print("=" * 120)
    print("FROZEN COHORT")
    print("=" * 120)

    print(
        f"Train total WSIs              : {EXPECTED_TRAIN_TOTAL}"
    )

    print(
        f"Train formal Stroma-ready     : {train_n}"
    )

    print(
        f"Test total WSIs               : {EXPECTED_TEST_TOTAL}"
    )

    print(
        f"Test formal Stroma-ready      : {test_n}"
    )

    print(
        f"Combined Stroma-ready         : {train_n + test_n}"
    )

    print()
    print("=" * 120)
    print("FROZEN PRIMARY ANALYSIS")
    print("=" * 120)

    print(
        f"Target                         : {TARGET_PC}"
    )

    print(
        f"Primary endpoint               : {PRIMARY_ENDPOINT}"
    )

    print(
        "Annotation denominator         : union of explicit AGGC annotations"
    )

    print(
        f"Minimum labeled fraction       : {PRIMARY_LABELED_FRACTION_THRESHOLD:.2f}"
    )

    print(
        f"Minimum usable tiles / slide   : {MIN_PRIMARY_TILES_PER_SLIDE}"
    )

    print(
        "Within-slide statistic         : Spearman rho"
    )

    print(
        "Expected direction             : rho > 0"
    )

    print(
        "Formal independent unit        : slide"
    )

    print(
        "Train role                     : external validation"
    )

    print(
        "Test role                      : separate replication"
    )

    print()
    print("=" * 120)
    print("FREEZE AUDIT")
    print("=" * 120)

    print(
        "PC values read                 : NO"
    )

    print(
        "AGGC association calculated    : NO"
    )

    print(
        "Test outcome read              : NO"
    )

    print(
        "PCA refit                      : NO"
    )

    print(
        "PC screening                   : NO"
    )

    print(
        "PC6 analysis                   : NO"
    )

    print(
        "Inference                      : NONE"
    )

    print()
    print("[OK] Saved:")

    for path in [
        OUT_TRAIN_COHORT,
        OUT_TEST_COHORT,
        OUT_COMBINED_COHORT,
        OUT_ANALYSIS_PLAN,
        OUT_LABEL_SPEC,
        OUT_GEOMETRY_SPEC,
        OUT_COHORT_SUMMARY,
        OUT_PROVENANCE,
        OUT_HASHES,
        OUT_METADATA,
        OUT_README,
    ]:
        print(" ", path)

    print()
    print(
        "[DONE] Step-17a AGGC PC3 external Stroma-validation "
        "specification frozen."
    )

    print()
    print("[FROZEN DESIGN]")
    print(
        f"TRAIN VALIDATION : {train_n} slides"
    )
    print(
        f"TEST REPLICATION : {test_n} slides"
    )
    print(
        f"PRIMARY          : PC3 vs {PRIMARY_ENDPOINT}"
    )
    print(
        "EXPECTED         : positive slide-level Spearman rho"
    )
    print(
        "NEXT             : Step 17b measurement only"
    )


if __name__ == "__main__":
    main()
