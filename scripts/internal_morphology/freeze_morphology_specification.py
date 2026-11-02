#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15a_freeze_pc3_quantitative_morphology_specification.py

Step-15a v2.1.0
Formal INTERNAL-COHORT PC3 quantitative morphology specification freeze.

Scientific status
-----------------
This is a formal prespecified re-analysis informed by prior exploratory internal
morphology analyses. It is not represented as a hypothesis-naive preregistration.

v2.1.0 change relative to v2.0.0
--------------------------------
SOFTWARE-SCHEMA HARDENING ONLY:
    - machine-readable basic white/black RGB thresholds are now stored;
    - no endpoint hierarchy changed;
    - no biological hypothesis changed;
    - no sampling rule changed;
    - no inferential procedure changed.

PRIMARY ENDPOINTS
-----------------
    SEI_eosin_stroma
    GOI_lumen_openness
    NPI_nuclear_packing

SECONDARY / SUPPORTIVE
----------------------
    CLR_nuclear_to_lumen
    CSR_nuclear_to_stroma
    dark_fraction
    hematoxylin_OD_proxy
    lumen_count
    lumen_area
    median_lumen_area
    mean_lumen_area
    lumen_component_density
    lumen_fragmentation
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = DATA_ROOT / 'discovery'
RESULTS_ROOT = RESULTS_ROOT

STEP13A_MANIFEST = (
    RESULTS_ROOT / "13a_pc3_morphology_sampling_manifest" / "pc3_sampling_manifest.csv"
)
STEP13C_MANIFEST = (
    RESULTS_ROOT
    / "13c_pc3_tumor_core_sensitivity_sampling"
    / "pc3_tumor_core_sensitivity_manifest.csv"
)

HIST_DIR = BASE / "heatmap_interpretation"
SRC_PC6_ARCH = HIST_DIR / "quantify_pc6_prostate_architecture.py"
SRC_PC3_2X2 = HIST_DIR / "analyze_pc3_2x2_gland_stroma_tumor.py"
SRC_PC3_LUMEN = HIST_DIR / "plot_pc3_lumen_fraction_boxplot.py"

OUT_DIR = RESULTS_ROOT / "15a_pc3_quantitative_morphology_specification"
OUT_FEATURES = OUT_DIR / "pc3_quantitative_morphology_feature_specification.csv"
OUT_PLAN = OUT_DIR / "pc3_quantitative_morphology_analysis_plan.json"
OUT_THRESHOLDS = OUT_DIR / "pc3_quantitative_morphology_thresholds.json"
OUT_PROVENANCE = OUT_DIR / "evidence_provenance.csv"
OUT_SAMPLING = OUT_DIR / "frozen_sampling_snapshot.csv"
OUT_DUPLICATES = OUT_DIR / "frozen_sampling_duplicate_audit.csv"
OUT_HASHES = OUT_DIR / "source_hashes.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"

CORE_FREEZE_OUTPUTS = [
    OUT_FEATURES,
    OUT_PLAN,
    OUT_THRESHOLDS,
    OUT_PROVENANCE,
    OUT_SAMPLING,
    OUT_HASHES,
    OUT_METADATA,
]

ANALYSIS_NAME = "Internal-cohort PC3 quantitative histomorphometric re-analysis"
VERSION = "2.2.0"
SCHEMA_VERSION = "15a-morphology-schema-2.2"

FREEZE_STATEMENT = (
    "Endpoint definitions and hypotheses were informed by prior exploratory "
    "internal morphology analyses; endpoint hierarchy, measurement specification, "
    "aggregation rules, sampling provenance, sensitivity analyses, and statistical "
    "procedures were frozen before the formal prespecified Step-15 re-analysis "
    "and inferential testing."
)

# -------------------------------------------------------------------------
# Frozen machine-readable parameters
# -------------------------------------------------------------------------
PATCH_SIZE_DEFAULT = 224
EPS = 1e-6

WHITE_RGB_MIN = 220
BLACK_RGB_MAX = 30

MIN_TISSUE_FRACTION = 0.30
MAX_WHITE_FRACTION = 0.65
MAX_BLACK_FRACTION = 0.10

PURPLE_B_OVER_R_RATIO = 0.85
PURPLE_B_OVER_G_RATIO = 0.85
PURPLE_R_MAX = 190
PURPLE_G_MAX = 170

DARK_RGB_MEAN_MAX = 150

LUMEN_R_MIN = 210
LUMEN_G_MIN = 210
LUMEN_B_MIN = 210

EOSIN_R_OVER_G_RATIO = 1.05
EOSIN_R_OVER_B_RATIO = 1.05
EOSIN_R_MIN = 120
EOSIN_G_MIN = 70

MIN_LUMEN_AREA_PIXELS = 20
LUMEN_CONNECTIVITY = "2D 4-connectivity"

TUMOR_ONLY_THRESHOLD = 0.10
TUMOR_CORE_THRESHOLD = 0.80
MAX_TILES_PER_SLIDE_PER_STRATUM = 10
MAX_TILES_PER_PATIENT_PER_STRATUM = 40

BOOTSTRAP_RESAMPLES = 10000
RANDOM_SEED = 20260805

# Prespecified release gates for formal Step-15c inference.
# The minimum paired-patient rule is defined as at least half of the
# 16-patient discovery cohort, preventing formal inference on small accidental
# subsets while remaining below the frozen tumor-core paired availability.
MINIMUM_RESOLUTION_FRACTION = 1.00
MINIMUM_QC_PASS_FRACTION = 0.95
MINIMUM_PAIRED_PATIENTS = 8

PRIMARY_FEATURES = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "NPI_nuclear_packing",
]

SECONDARY_FEATURES = [
    "CLR_nuclear_to_lumen",
    "CSR_nuclear_to_stroma",
    "dark_fraction",
    "hematoxylin_OD_proxy",
    "lumen_count",
    "lumen_area",
    "median_lumen_area",
    "mean_lumen_area",
    "lumen_component_density",
    "lumen_fragmentation",
]

FEATURES = [
    {
        "feature_name": "SEI_eosin_stroma",
        "short_name": "SEI",
        "endpoint_tier": "primary",
        "proxy_category": "eosin/stroma-like composition",
        "formal_segmentation": False,
        "mathematical_definition":
            "eosin_mask_pixels / (tissue_mask OR lumen_mask)_pixels",
        "claim_boundary":
            "Image-derived eosin/stroma-like composition proxy; not formal stromal segmentation.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "GOI_lumen_openness",
        "short_name": "GOI",
        "endpoint_tier": "primary",
        "proxy_category": "lumen/open-space architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "lumen_mask_pixels / (tissue_mask OR lumen_mask)_pixels",
        "claim_boundary":
            "Image-derived lumen/open-space proxy; not formal lumen/gland segmentation.",
        "known_limitation":
            "Historical lumen threshold overlaps pure-white background threshold; "
            "border-connected-open-space sensitivity is prespecified.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "NPI_nuclear_packing",
        "short_name": "NPI",
        "endpoint_tier": "primary",
        "proxy_category": "hematoxylin/nuclear-richness",
        "formal_segmentation": False,
        "mathematical_definition":
            "purple_mask_pixels / (tissue_mask OR lumen_mask)_pixels",
        "claim_boundary":
            "Image-derived hematoxylin/nuclear-richness proxy; not nuclei count/density.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "CLR_nuclear_to_lumen",
        "short_name": "CLR",
        "endpoint_tier": "secondary",
        "proxy_category": "cellularity-to-lumen contrast",
        "formal_segmentation": False,
        "mathematical_definition":
            "NPI_nuclear_packing / (GOI_lumen_openness + 1e-6)",
        "claim_boundary": "Supportive composite image proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "CSR_nuclear_to_stroma",
        "short_name": "CSR",
        "endpoint_tier": "secondary",
        "proxy_category": "nuclear-to-eosin contrast",
        "formal_segmentation": False,
        "mathematical_definition":
            "NPI_nuclear_packing / (SEI_eosin_stroma + 1e-6)",
        "claim_boundary": "Supportive composite image proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "dark_fraction",
        "short_name": "dark_fraction",
        "endpoint_tier": "secondary",
        "proxy_category": "dark tissue pixels",
        "formal_segmentation": False,
        "mathematical_definition":
            "dark_mask_pixels / (tissue_mask OR lumen_mask)_pixels",
        "claim_boundary": "Supportive image-intensity proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "hematoxylin_OD_proxy",
        "short_name": "hema_OD",
        "endpoint_tier": "secondary",
        "proxy_category": "optical-density intensity",
        "formal_segmentation": False,
        "mathematical_definition":
            "mean[-log(B/255)] over tissue_mask",
        "claim_boundary":
            "Supportive optical-density proxy; not formal stain deconvolution.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "lumen_count",
        "short_name": "lumen_count",
        "endpoint_tier": "secondary",
        "proxy_category": "lumen-object architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "count(retained lumen-like connected components >=20 pixels)",
        "claim_boundary": "Supportive native-pixel architecture proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "lumen_area",
        "short_name": "lumen_area",
        "endpoint_tier": "secondary",
        "proxy_category": "lumen-object architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "sum(retained lumen-like component pixel areas)",
        "claim_boundary":
            "Pixel-area metric only; physical-area interpretation prohibited.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "median_lumen_area",
        "short_name": "median_lumen_area",
        "endpoint_tier": "secondary",
        "proxy_category": "lumen-object architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "median(retained lumen-like component areas in pixels)",
        "claim_boundary": "Supportive pixel-scale architecture proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "mean_lumen_area",
        "short_name": "mean_lumen_area",
        "endpoint_tier": "secondary",
        "proxy_category": "lumen-object architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "mean(retained lumen-like component areas in pixels)",
        "claim_boundary": "Supportive pixel-scale architecture proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "lumen_component_density",
        "short_name": "lumen_component_density",
        "endpoint_tier": "secondary",
        "proxy_category": "lumen-object architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "lumen_count / [(tissue_mask OR lumen_mask)_pixels + 1e-6]",
        "claim_boundary": "Supportive pixel-scale architecture proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
    {
        "feature_name": "lumen_fragmentation",
        "short_name": "lumen_fragmentation",
        "endpoint_tier": "secondary",
        "proxy_category": "lumen-object architecture",
        "formal_segmentation": False,
        "mathematical_definition":
            "lumen_count / (lumen_area_pixels + 1e-6)",
        "claim_boundary": "Supportive fragmentation proxy.",
        "source_definition": "quantify_pc6_prostate_architecture.py",
    },
]

THRESHOLDS = {
    "schema_version": SCHEMA_VERSION,
    "patch_size_default": PATCH_SIZE_DEFAULT,
    "eps": EPS,
    "basic_masks": {
        "white_rgb_min": WHITE_RGB_MIN,
        "black_rgb_max": BLACK_RGB_MAX,
        "white_mask_definition":
            "(R > white_rgb_min) & (G > white_rgb_min) & (B > white_rgb_min)",
        "black_mask_definition":
            "(R < black_rgb_max) & (G < black_rgb_max) & (B < black_rgb_max)",
        "tissue_mask_definition":
            "(~white_mask) & (~black_mask)",
    },
    "quality_filters": {
        "minimum_tissue_fraction": MIN_TISSUE_FRACTION,
        "maximum_white_fraction": MAX_WHITE_FRACTION,
        "maximum_black_fraction": MAX_BLACK_FRACTION,
    },
    "purple_mask": {
        "b_over_r_min": PURPLE_B_OVER_R_RATIO,
        "b_over_g_min": PURPLE_B_OVER_G_RATIO,
        "r_max": PURPLE_R_MAX,
        "g_max": PURPLE_G_MAX,
    },
    "dark_mask": {
        "mean_rgb_max": DARK_RGB_MEAN_MAX,
    },
    "historical_lumen_mask": {
        "r_min": LUMEN_R_MIN,
        "g_min": LUMEN_G_MIN,
        "b_min": LUMEN_B_MIN,
        "known_overlap_with_white_background": True,
        "primary_status":
            "retained unchanged for formal historical-definition re-analysis",
    },
    "eosin_mask": {
        "r_over_g_min": EOSIN_R_OVER_G_RATIO,
        "r_over_b_min": EOSIN_R_OVER_B_RATIO,
        "r_min": EOSIN_R_MIN,
        "g_min": EOSIN_G_MIN,
        "exclude_purple_mask": True,
    },
    "lumen_connected_components": {
        "minimum_component_area_pixels": MIN_LUMEN_AREA_PIXELS,
        "connectivity": LUMEN_CONNECTIVITY,
        "scipy_structure_definition":
            "scipy.ndimage.generate_binary_structure(rank=2, connectivity=1)",
    },
    "lumen_background_sensitivity": {
        "exclude_border_connected_open_components": True,
        "connectivity": LUMEN_CONNECTIVITY,
        "status": "prespecified sensitivity only",
        "claim_boundary":
            "border-connected background sensitivity; not corrected lumen segmentation",
    },
    "sampling": {
        "tumor_only_overlap_threshold": TUMOR_ONLY_THRESHOLD,
        "tumor_core_overlap_threshold": TUMOR_CORE_THRESHOLD,
        "maximum_tiles_per_slide_per_stratum":
            MAX_TILES_PER_SLIDE_PER_STRATUM,
        "maximum_tiles_per_patient_per_stratum":
            MAX_TILES_PER_PATIENT_PER_STRATUM,
    },
    "pixel_scale": {
        "measurement_statement":
            "Measurements are performed on frozen sampled image patches.",
        "reliable_mpp_available": False,
        "reliable_objective_magnification_available": False,
        "physical_area_interpretation": False,
    },
}

ANALYSIS_PLAN = {
    "analysis_name": ANALYSIS_NAME,
    "version": VERSION,
    "schema_version": SCHEMA_VERSION,
    "cohort_scope": "internal/discovery cohort only",
    "analysis_status":
        "formal prespecified re-analysis informed by prior exploratory internal morphology analyses",
    "freeze_statement": FREEZE_STATEMENT,
    "primary_endpoints": PRIMARY_FEATURES,
    "secondary_endpoints": SECONDARY_FEATURES,
    "comparison_hierarchy": {
        "primary_analysis": {
            "comparison": "all_tissue_high vs all_tissue_low",
            "unit": "patient",
            "paired": True,
            "reported_difference": "PC3-high minus PC3-low",
        },
        "prespecified_tumor_purity_sensitivity": {
            "comparison": "tumor_core_high vs tumor_core_low",
            "tumor_overlap_threshold": TUMOR_CORE_THRESHOLD,
            "unit": "patient",
            "paired": True,
        },
        "additional_sensitivity": {
            "comparison": "tumor_only_high vs tumor_only_low",
            "tumor_overlap_threshold": TUMOR_ONLY_THRESHOLD,
            "unit": "patient",
            "paired": True,
        },
    },
    "aggregation": {
        "tile_level":
            "feature extraction only; no primary tile-level inferential p-values",
        "slide_level":
            "median across retained tiles within slide/stratum",
        "patient_level":
            "median across slide medians within patient/stratum",
        "primary_inference_unit": "patient",
    },
    "statistical_tests": {
        "primary_test": "two-sided Wilcoxon signed-rank test",
        "primary_effect":
            "median paired difference, PC3-high minus PC3-low",
        "effect_size": "paired rank-biserial effect size",
        "confidence_interval":
            "95% patient-level bootstrap CI for median paired difference",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "random_seed": RANDOM_SEED,
    },
    "multiple_testing": {
        "primary_adjustment":
            "Benjamini-Hochberg FDR across the three primary endpoints for the all-tissue primary analysis",
        "tumor_core_adjustment":
            "Benjamini-Hochberg FDR across the three primary endpoints within the tumor-core sensitivity family",
        "tumor_only_adjustment":
            "Benjamini-Hochberg FDR across the three primary endpoints within the tumor-only sensitivity family",
        "border_sensitivity_adjustment":
            "Benjamini-Hochberg FDR across the three border-excluded primary-dimension sensitivity endpoints within each comparison",
        "secondary_adjustment":
            "Benjamini-Hochberg FDR across secondary endpoints within each comparison",
        "no_promotion_rule":
            "secondary endpoints cannot be promoted to primary after outcome inspection",
    },
    "inference_qc_gate": {
        "minimum_resolution_fraction": MINIMUM_RESOLUTION_FRACTION,
        "minimum_qc_pass_fraction": MINIMUM_QC_PASS_FRACTION,
        "minimum_paired_patients": MINIMUM_PAIRED_PATIENTS,
        "minimum_paired_patients_rationale":
            "At least half of the 16-patient discovery cohort; prespecified as an interpretability/stability gate rather than selected from inferential significance.",
    },
    "continuous_pc3_sensitivity": {
        "status": "prespecified sensitivity",
        "analysis_step": "15d",
        "association": "two-sided Spearman rank correlation",
        "analysis_unit": "slide-level and/or patient-level aggregated summaries",
        "tile_level_inference": "prohibited",
        "deferred_statement":
            "Prespecified continuous-PC3 morphology association sensitivity is deferred to Step 15d.",
    },
    "lumen_background_sensitivity": {
        "status": "prespecified sensitivity",
        "primary_measurement":
            "historical GOI definition remains unchanged",
        "sensitivity_measurement":
            "exclude border-connected open components using 2D 4-connectivity",
        "claim_boundary":
            "border-connected background sensitivity only; not corrected lumen segmentation",
    },
}

PROVENANCE = [
    {
        "evidence_item":
            "SEI/GOI/NPI/CLR/CSR and lumen-object feature definitions",
        "evidence_type":
            "pre-existing exploratory internal implementation",
        "source": str(SRC_PC6_ARCH),
        "role_in_step15":
            "historical measurement implementation informing formal Step-15 re-analysis",
    },
    {
        "evidence_item": "PC3 tumor/non-tumor morphology hypothesis",
        "evidence_type": "prior exploratory internal morphology analysis",
        "source": str(SRC_PC3_2X2),
        "role_in_step15":
            "informed biological hypothesis; not formal inferential dataset",
    },
    {
        "evidence_item": "Prior PC3 lumen morphology analysis",
        "evidence_type": "prior exploratory internal morphology analysis",
        "source": str(SRC_PC3_LUMEN),
        "role_in_step15":
            "informed lumen/open-space hypothesis; explicitly recorded",
    },
    {
        "evidence_item": "Frozen all-tissue PC3 sampling",
        "evidence_type": "frozen internal sampling",
        "source": str(STEP13A_MANIFEST),
        "role_in_step15": "source for all_tissue_low/high",
    },
    {
        "evidence_item": "Frozen tumor-specific PC3 sampling",
        "evidence_type": "frozen internal sensitivity sampling",
        "source": str(STEP13C_MANIFEST),
        "role_in_step15": "source for tumor_only/tumor_core low/high",
    },
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--overwrite-freeze",
        action="store_true",
        help="Explicitly replace an existing Step-15a freeze.",
    )
    return p.parse_args()

def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required source ({label}): {path}")

def guard_freeze(overwrite: bool):
    existing = [p for p in CORE_FREEZE_OUTPUTS if p.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Frozen Step-15a outputs already exist. "
            "Refusing silent overwrite. Use --overwrite-freeze."
        )

def stable_tile_key(row: pd.Series) -> str:
    for col in ["model_tile_uid", "sample_id", "sensitivity_sample_id"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    prefix = str(row.get("prefix", ""))
    tile_idx = row.get("tile_idx", np.nan)
    if pd.notna(tile_idx):
        return f"{prefix}__tile{int(tile_idx)}"
    x = row.get("x", row.get("tile_x", -1))
    y = row.get("y", row.get("tile_y", -1))
    return f"{prefix}__x{int(x)}__y{int(y)}"

def standardize(df: pd.DataFrame, source_step: str) -> pd.DataFrame:
    df = df.copy()
    if "x" not in df.columns and "tile_x" in df.columns:
        df["x"] = df["tile_x"]
    if "y" not in df.columns and "tile_y" in df.columns:
        df["y"] = df["tile_y"]
    df["snapshot_source_step"] = source_step
    df["_tile_key"] = df.apply(stable_tile_key, axis=1)
    return df

def build_sampling_snapshot():
    a = standardize(pd.read_csv(STEP13A_MANIFEST, low_memory=False), "13a")
    c = standardize(pd.read_csv(STEP13C_MANIFEST, low_memory=False), "13c")

    a = a[a["sampling_stratum"].isin({"all_tissue_low", "all_tissue_high"})].copy()
    c = c[c["sampling_stratum"].isin({
        "tumor_only_low", "tumor_only_high",
        "tumor_core_low", "tumor_core_high"
    })].copy()

    combined = pd.concat([a, c], ignore_index=True, sort=False)

    dup_cols = ["sampling_stratum", "patient_id", "prefix", "_tile_key"]
    dup_mask = combined.duplicated(subset=dup_cols, keep=False)
    duplicate_rows = combined.loc[dup_mask].copy()

    if not duplicate_rows.empty:
        duplicate_rows.to_csv(OUT_DUPLICATES, index=False)
        raise RuntimeError(
            "Same-stratum duplicate frozen samples detected. "
            f"See {OUT_DUPLICATES}"
        )

    summary = (
        combined
        .groupby(["snapshot_source_step", "sampling_stratum"], dropna=False)
        .agg(
            n_sampled_tiles=("_tile_key", "size"),
            n_patients_sampled=("patient_id", "nunique"),
            n_slides_sampled=("prefix", "nunique"),
            median_pc3_score=("pc3_score", "median"),
            median_pc3_within_slide_percentile=("pc3_within_slide_percentile", "median"),
            median_tumor_overlap_fraction=("tumor_overlap_fraction", "median"),
            minimum_tumor_overlap_fraction=("tumor_overlap_fraction", "min"),
        )
        .reset_index()
    )
    return combined, summary, duplicate_rows

def main():
    args = parse_args()

    require_file(STEP13A_MANIFEST, "Step 13a manifest")
    require_file(STEP13C_MANIFEST, "Step 13c manifest")
    guard_freeze(args.overwrite_freeze)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(FEATURES).to_csv(OUT_FEATURES, index=False)
    with OUT_PLAN.open("w") as f:
        json.dump(ANALYSIS_PLAN, f, indent=2)
    with OUT_THRESHOLDS.open("w") as f:
        json.dump(THRESHOLDS, f, indent=2)
    pd.DataFrame(PROVENANCE).to_csv(OUT_PROVENANCE, index=False)

    combined, summary, duplicates = build_sampling_snapshot()
    summary.to_csv(OUT_SAMPLING, index=False)

    if duplicates.empty:
        duplicates = pd.DataFrame(columns=[
            "sampling_stratum", "patient_id", "prefix", "_tile_key"
        ])
    duplicates.to_csv(OUT_DUPLICATES, index=False)

    source_specs = [
        ("step13a_sampling_manifest", STEP13A_MANIFEST, True),
        ("step13c_sampling_manifest", STEP13C_MANIFEST, True),
        ("historical_pc6_architecture", SRC_PC6_ARCH, False),
        ("historical_pc3_2x2", SRC_PC3_2X2, False),
        ("historical_pc3_lumen", SRC_PC3_LUMEN, False),
        ("step15a_script", Path(__file__).resolve(), True),
    ]

    hash_rows = []
    for label_name, path, required in source_specs:
        if required and not path.exists():
            raise FileNotFoundError(path)
        hash_rows.append({
            "source_label": label_name,
            "path": str(path),
            "required": required,
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
        })
    pd.DataFrame(hash_rows).to_csv(OUT_HASHES, index=False)

    metadata = {
        "analysis_name": ANALYSIS_NAME,
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status":
            "formal prespecified re-analysis informed by prior exploratory internal morphology analyses",
        "freeze_statement": FREEZE_STATEMENT,
        "v2_2_change":
            "inference-protocol hardening only: adds prespecified QC release gates, minimum paired-patient gate, and explicit Step-15d continuous sensitivity deferral; no morphology threshold, endpoint, or sampling change",
        "overwrite_freeze_requested": bool(args.overwrite_freeze),
        "primary_endpoints": PRIMARY_FEATURES,
        "secondary_endpoints": SECONDARY_FEATURES,
        "inference_qc_gate": ANALYSIS_PLAN["inference_qc_gate"],
        "continuous_pc3_sensitivity": ANALYSIS_PLAN["continuous_pc3_sensitivity"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "n_sampling_rows": int(len(combined)),
        "same_stratum_duplicate_rows": int(len(duplicates)),
    }
    with OUT_METADATA.open("w") as f:
        json.dump(metadata, f, indent=2)

    OUT_README.write_text(
        f"""STEP 15A PC3 QUANTITATIVE MORPHOLOGY SPECIFICATION
=================================================

Version: {VERSION}
Schema:  {SCHEMA_VERSION}

{FREEZE_STATEMENT}

v2.2.0 is an inference-protocol hardening revision only.
No morphology threshold, endpoint hierarchy, or frozen sampling changed.

INFERENCE QC RELEASE GATE
-------------------------
minimum_resolution_fraction = {MINIMUM_RESOLUTION_FRACTION}
minimum_qc_pass_fraction     = {MINIMUM_QC_PASS_FRACTION}
minimum_paired_patients      = {MINIMUM_PAIRED_PATIENTS}

Minimum paired-patient rationale:
  at least half of the 16-patient discovery cohort; this is an
  interpretability/stability gate, not a significance-driven threshold.

PRIMARY:
  SEI_eosin_stroma
  GOI_lumen_openness
  NPI_nuclear_packing

PRIMARY ANALYSIS:
  all_tissue_high vs all_tissue_low

SENSITIVITY:
  tumor_core_high vs tumor_core_low
  tumor_only_high vs tumor_only_low
  border-connected-open-space exclusion

CONTINUOUS-PC3 SENSITIVITY:
  Prespecified continuous-PC3 morphology association sensitivity is deferred
  to Step 15d.

Because the analysis-plan hash changes in v2.2, Step 15b must be rerun so its
provenance metadata records this exact Step-15a specification.
""",
        encoding="utf-8",
    )

    print("=" * 110)
    print("STEP 15A v2.2 FREEZE COMPLETE")
    print("=" * 110)
    print("Version:", VERSION)
    print("Schema :", SCHEMA_VERSION)
    print(summary.to_string(index=False))
    print("\n[OK]", OUT_DIR)

if __name__ == "__main__":
    main()
