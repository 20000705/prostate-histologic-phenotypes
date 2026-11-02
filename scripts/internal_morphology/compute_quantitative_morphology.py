#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
15b_compute_pc3_quantitative_morphology.py

Step-15b v1.4.0
Hardened INTERNAL-COHORT PC3 morphology measurement engine.

Key rules
---------
1. ALL morphology thresholds are read from Step-15a v2.2 JSON.
   Step-15b owns no independent morphology threshold.
2. Frozen sample sources:
      Step 13a -> all_tissue_low/high
      Step 13c -> tumor_only_low/high + tumor_core_low/high
3. Native TIFF crops are bounds-checked BEFORE cropping.
4. Resolution method / pixel-source distribution is audited by stratum.
5. Where both an extracted tile and a native TIFF crop are available,
   pixel-equivalence QC is attempted as OPTIONAL audit only. Audit failure
   can never remove an otherwise valid formal morphology measurement.
6. Duplicate resolution keys are never silently overwritten. Physical identity
   is defined preferentially by WSI path + x + y + patch_size, so different
   exported PNG filenames from the same WSI location are treated as legitimate
   duplicate exports of one physical tile.
7. No inferential statistics are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import OrderedDict
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

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

Image.MAX_IMAGE_PIXELS = None

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = DATA_ROOT / 'discovery'
RESULTS_ROOT = RESULTS_ROOT

STEP15A_DIR = RESULTS_ROOT / "15a_pc3_quantitative_morphology_specification"
STEP15A_PLAN = STEP15A_DIR / "pc3_quantitative_morphology_analysis_plan.json"
STEP15A_THRESHOLDS = STEP15A_DIR / "pc3_quantitative_morphology_thresholds.json"
STEP15A_FEATURE_SPEC = STEP15A_DIR / "pc3_quantitative_morphology_feature_specification.csv"
STEP15A_METADATA = STEP15A_DIR / "analysis_metadata.json"

STEP13A_MANIFEST = (
    RESULTS_ROOT / "13a_pc3_morphology_sampling_manifest" / "pc3_sampling_manifest.csv"
)
STEP13B_MANIFEST = (
    RESULTS_ROOT / "13b_pc3_morphology_roi_extraction" / "roi_extraction_manifest.csv"
)
STEP13C_MANIFEST = (
    RESULTS_ROOT
    / "13c_pc3_tumor_core_sensitivity_sampling"
    / "pc3_tumor_core_sensitivity_manifest.csv"
)
STEP13D_CORE_MANIFEST = (
    RESULTS_ROOT
    / "13d_pc3_morphology_review_montages"
    / "tumor_core_extraction_manifest.csv"
)

OUT_DIR = RESULTS_ROOT / "15b_pc3_quantitative_morphology"
OUT_TILE = OUT_DIR / "pc3_quantitative_morphology_tile_features.csv"
OUT_QC_FAILURES = OUT_DIR / "pc3_quantitative_morphology_qc_failures.csv"
OUT_SLIDE = OUT_DIR / "pc3_quantitative_morphology_slide_summary.csv"
OUT_PATIENT = OUT_DIR / "pc3_quantitative_morphology_patient_summary.csv"
OUT_COVERAGE = OUT_DIR / "pc3_quantitative_morphology_feature_coverage.csv"
OUT_AUDIT = OUT_DIR / "pc3_quantitative_morphology_sampling_audit.csv"
OUT_RESOLUTION = OUT_DIR / "pc3_quantitative_morphology_resolution_manifest.csv"
OUT_RESOLUTION_AUDIT = OUT_DIR / "pc3_quantitative_morphology_resolution_method_audit.csv"
OUT_EQUIVALENCE = OUT_DIR / "pc3_quantitative_morphology_pixel_source_equivalence_audit.csv"
OUT_RESOLUTION_DUPLICATES = OUT_DIR / "pc3_quantitative_morphology_resolution_duplicate_audit.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"

CORE_OUTPUTS = [
    OUT_TILE, OUT_QC_FAILURES, OUT_SLIDE, OUT_PATIENT, OUT_COVERAGE,
    OUT_AUDIT, OUT_RESOLUTION, OUT_RESOLUTION_AUDIT, OUT_EQUIVALENCE,
    OUT_RESOLUTION_DUPLICATES, OUT_METADATA, OUT_README,
]

VERSION = "1.4.0"
EXPECTED_15A_VERSION_PREFIX = "2.2."
EXPECTED_SCHEMA_VERSION = "15a-morphology-schema-2.2"

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

SENSITIVITY_FEATURES = [
    "GOI_lumen_openness_border_excluded",
    "NPI_nuclear_packing_border_excluded_denominator",
    "SEI_eosin_stroma_border_excluded_denominator",
    "CLR_nuclear_to_lumen_border_excluded",
    "lumen_count_border_excluded",
    "lumen_area_border_excluded",
    "median_lumen_area_border_excluded",
    "mean_lumen_area_border_excluded",
    "lumen_component_density_border_excluded",
    "lumen_fragmentation_border_excluded",
    "border_connected_open_fraction_all",
]

ALL_FEATURES = PRIMARY_FEATURES + SECONDARY_FEATURES + SENSITIVITY_FEATURES

SLIDE_CACHE: OrderedDict[str, Image.Image] = OrderedDict()
MAX_CACHED_SLIDES = 1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Engineering-debugging cap only. Do not use for formal full run.",
    )
    return p.parse_args()

def require_file(path: Path, label_text: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input ({label_text}): {path}")

def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def guard_outputs(overwrite: bool):
    existing = [p for p in CORE_OUTPUTS if p.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Step-15b outputs already exist. Refusing silent overwrite. "
            "Use --overwrite if intentional."
        )

def as_int(value, default=None):
    if value is None or pd.isna(value):
        return default
    try:
        return int(round(float(value)))
    except Exception:
        return default

def clean_path(value) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return Path(s)

def first_value(row: pd.Series, cols: Iterable[str]):
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip() != "":
                return v
    return None

def stable_tile_key(row: pd.Series) -> str:
    for c in ["model_tile_uid", "sample_id", "sensitivity_sample_id"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    prefix = str(row.get("prefix", ""))
    tile_idx = row.get("tile_idx", np.nan)
    if pd.notna(tile_idx):
        return f"{prefix}__tile{int(tile_idx)}"
    x = as_int(first_value(row, ["x", "tile_x"]), -1)
    y = as_int(first_value(row, ["y", "tile_y"]), -1)
    return f"{prefix}__x{x}__y{y}"

def load_frozen_specification():
    require_file(STEP15A_PLAN, "Step 15a plan")
    require_file(STEP15A_THRESHOLDS, "Step 15a thresholds")
    require_file(STEP15A_FEATURE_SPEC, "Step 15a feature specification")

    plan = json.load(STEP15A_PLAN.open())
    thresholds = json.load(STEP15A_THRESHOLDS.open())
    spec = pd.read_csv(STEP15A_FEATURE_SPEC, low_memory=False)

    version = str(plan.get("version", ""))
    schema = str(plan.get("schema_version", thresholds.get("schema_version", "")))

    if not version.startswith(EXPECTED_15A_VERSION_PREFIX):
        raise RuntimeError(
            f"Step-15b expects Step-15a {EXPECTED_15A_VERSION_PREFIX}x; found {version}"
        )
    if schema != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Step-15a schema mismatch. Expected {EXPECTED_SCHEMA_VERSION}; found {schema}"
        )

    observed_primary = spec.loc[
        spec["endpoint_tier"] == "primary", "feature_name"
    ].tolist()
    observed_secondary = spec.loc[
        spec["endpoint_tier"] == "secondary", "feature_name"
    ].tolist()

    if observed_primary != PRIMARY_FEATURES:
        raise RuntimeError(
            f"Primary feature mismatch.\nExpected {PRIMARY_FEATURES}\nObserved {observed_primary}"
        )
    if observed_secondary != SECONDARY_FEATURES:
        raise RuntimeError(
            f"Secondary feature mismatch.\nExpected {SECONDARY_FEATURES}\nObserved {observed_secondary}"
        )

    expected_all = set(PRIMARY_FEATURES + SECONDARY_FEATURES)
    observed_all = set(spec["feature_name"].astype(str))
    if observed_all != expected_all:
        raise RuntimeError(
            f"Frozen feature set mismatch.\nMissing={expected_all-observed_all}\n"
            f"Unexpected={observed_all-expected_all}"
        )

    required_threshold_paths = [
        ("basic_masks", "white_rgb_min"),
        ("basic_masks", "black_rgb_max"),
        ("quality_filters", "minimum_tissue_fraction"),
        ("quality_filters", "maximum_white_fraction"),
        ("quality_filters", "maximum_black_fraction"),
        ("purple_mask", "b_over_r_min"),
        ("purple_mask", "b_over_g_min"),
        ("purple_mask", "r_max"),
        ("purple_mask", "g_max"),
        ("dark_mask", "mean_rgb_max"),
        ("historical_lumen_mask", "r_min"),
        ("historical_lumen_mask", "g_min"),
        ("historical_lumen_mask", "b_min"),
        ("eosin_mask", "r_over_g_min"),
        ("eosin_mask", "r_over_b_min"),
        ("eosin_mask", "r_min"),
        ("eosin_mask", "g_min"),
        ("lumen_connected_components", "minimum_component_area_pixels"),
    ]
    missing = []
    for section, key in required_threshold_paths:
        if section not in thresholds or key not in thresholds[section]:
            missing.append(f"{section}.{key}")
    if missing:
        raise RuntimeError(
            "Step-15a frozen threshold schema missing required keys: " + ", ".join(missing)
        )

    return plan, thresholds, spec

def standardize_sampling(df: pd.DataFrame, source_step: str) -> pd.DataFrame:
    df = df.copy()
    if "x" not in df.columns and "tile_x" in df.columns:
        df["x"] = df["tile_x"]
    if "y" not in df.columns and "tile_y" in df.columns:
        df["y"] = df["tile_y"]
    df["source_step"] = source_step
    df["frozen_tile_key"] = df.apply(stable_tile_key, axis=1)
    return df

def load_frozen_samples():
    a = standardize_sampling(pd.read_csv(STEP13A_MANIFEST, low_memory=False), "13a")
    c = standardize_sampling(pd.read_csv(STEP13C_MANIFEST, low_memory=False), "13c")

    a = a[a["sampling_stratum"].isin({"all_tissue_low", "all_tissue_high"})].copy()
    c = c[c["sampling_stratum"].isin({
        "tumor_only_low", "tumor_only_high",
        "tumor_core_low", "tumor_core_high"
    })].copy()

    combined = pd.concat([a, c], ignore_index=True, sort=False)
    dup = combined.duplicated(
        subset=["sampling_stratum", "patient_id", "prefix", "frozen_tile_key"],
        keep=False,
    )
    if dup.any():
        raise RuntimeError(
            "Unexpected same-stratum duplicate frozen samples:\n"
            + combined.loc[dup, [
                "source_step", "sampling_stratum", "patient_id", "prefix", "frozen_tile_key"
            ]].to_string(index=False)
        )
    return combined

def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

def standardize_resolution(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "x" not in df.columns and "tile_x" in df.columns:
        df["x"] = df["tile_x"]
    if "y" not in df.columns and "tile_y" in df.columns:
        df["y"] = df["tile_y"]
    df["_resolution_source"] = label_name
    return df

def _resolution_identity(row: pd.Series):
    """
    Return a conservative PHYSICAL image identity for duplicate-key auditing.

    IMPORTANT
    ---------
    Different exported PNG filenames are NOT automatically different physical
    tiles. The preferred identity is:

        WSI path + x + y + patch_size

    Therefore, for example:
        PC3ROI_0021.png
        PC3ROI_1181.png
        PC3SENS_2115.png

    are considered the same physical tile when all three were extracted from
    the same WSI coordinate with the same patch size.

    Only when WSI-coordinate identity cannot be established do we fall back to
    the extracted image path. If neither identity can be established, return
    None so duplicate ambiguity cannot be silently resolved.
    """

    # -----------------------------------------------------------------
    # Preferred identity: native physical WSI location.
    # -----------------------------------------------------------------
    wsi_path = None

    for column in WSI_PATH_COLUMNS:
        if column in row.index:
            candidate = clean_path(
                row[column]
            )

            if candidate is not None:
                try:
                    wsi_path = str(
                        candidate.resolve(
                            strict=False
                        )
                    )
                except Exception:
                    wsi_path = str(
                        candidate
                    )

                break

    x = as_int(
        first_value(
            row,
            [
                "x",
                "tile_x",
            ],
        ),
        None,
    )

    y = as_int(
        first_value(
            row,
            [
                "y",
                "tile_y",
            ],
        ),
        None,
    )

    patch_size = as_int(
        first_value(
            row,
            [
                "patch_size",
            ],
        ),
        None,
    )

    if (
        wsi_path is not None
        and x is not None
        and y is not None
        and patch_size is not None
    ):
        return (
            "wsi_coordinate",
            wsi_path,
            int(x),
            int(y),
            int(patch_size),
        )

    # -----------------------------------------------------------------
    # Fallback identity: extracted image path.
    #
    # This fallback is used only when WSI-coordinate identity is missing.
    # -----------------------------------------------------------------
    extracted_path = find_existing_model_tile(
        row
    )

    if extracted_path is not None:
        try:
            extracted_path = str(
                extracted_path.resolve(
                    strict=False
                )
            )
        except Exception:
            extracted_path = str(
                extracted_path
            )

        return (
            "extracted_path_fallback",
            extracted_path,
        )

    return None

def _unique_resolution_index(all_rows, key_fn, name, audit_rows):
    groups = {}
    for idx, row in all_rows.iterrows():
        key = key_fn(row)
        if key is not None:
            groups.setdefault(key, []).append(idx)

    out = {}
    for key, idxs in groups.items():
        if len(idxs) == 1:
            out[key] = idxs[0]
            continue

        identities = [_resolution_identity(all_rows.loc[i]) for i in idxs]
        if any(x is None for x in identities) or len(set(identities)) != 1:
            raise RuntimeError(
                f"Ambiguous duplicate resolution key in {name}: {key!r}\n"
                f"Physical identities={identities}\n"
                + all_rows.loc[idxs].to_string(index=False)
            )

        kept = min(idxs)  # deterministic
        out[key] = kept
        for i, identity in zip(idxs, identities):
            audit_rows.append({
                "index_name": name,
                "logical_key": repr(key),
                "row_index": int(i),
                "resolution_source": str(
                    all_rows.loc[i].get(
                        "_resolution_source",
                        "",
                    )
                ),
                "sample_id": all_rows.loc[i].get(
                    "sample_id",
                    np.nan,
                ),
                "sensitivity_sample_id": all_rows.loc[i].get(
                    "sensitivity_sample_id",
                    np.nan,
                ),
                "sampling_stratum": all_rows.loc[i].get(
                    "sampling_stratum",
                    np.nan,
                ),
                "prefix": all_rows.loc[i].get(
                    "prefix",
                    np.nan,
                ),
                "x": all_rows.loc[i].get(
                    "x",
                    np.nan,
                ),
                "y": all_rows.loc[i].get(
                    "y",
                    np.nan,
                ),
                "patch_size": all_rows.loc[i].get(
                    "patch_size",
                    np.nan,
                ),
                "model_tile_path": all_rows.loc[i].get(
                    "model_tile_path",
                    np.nan,
                ),
                "physical_identity": repr(
                    identity
                ),
                "same_physical_location": True,
                "status": (
                    "identical_physical_tile_kept"
                    if i == kept
                    else "identical_physical_tile_duplicate_deduplicated"
                ),
                "kept_row_index": int(
                    kept
                ),
            })
    return out

def build_resolution_indexes():
    b = standardize_resolution(load_optional(STEP13B_MANIFEST), "13b_roi_extraction")
    d = standardize_resolution(load_optional(STEP13D_CORE_MANIFEST), "13d_tumor_core_extraction")
    frames = [x for x in [b, d] if not x.empty]
    all_rows = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    indexes = {
        "all_rows": all_rows,
        "by_sample_id": {},
        "by_sensitivity_id": {},
        "by_model_tile_uid": {},
        "by_coord": {},
        "duplicate_audit": pd.DataFrame(),
    }
    if all_rows.empty:
        return indexes

    audit_rows = []

    def scalar(col):
        def f(row):
            v = row.get(col, np.nan)
            return None if pd.isna(v) or str(v).strip() == "" else str(v)
        return f

    def coord(row):
        if pd.isna(row.get("prefix", np.nan)) or pd.isna(row.get("x", np.nan)) or pd.isna(row.get("y", np.nan)):
            return None
        return (str(row["prefix"]), as_int(row["x"], -1), as_int(row["y"], -1))

    indexes["by_sample_id"] = _unique_resolution_index(all_rows, scalar("sample_id"), "by_sample_id", audit_rows)
    indexes["by_sensitivity_id"] = _unique_resolution_index(all_rows, scalar("sensitivity_sample_id"), "by_sensitivity_id", audit_rows)
    indexes["by_model_tile_uid"] = _unique_resolution_index(all_rows, scalar("model_tile_uid"), "by_model_tile_uid", audit_rows)
    indexes["by_coord"] = _unique_resolution_index(all_rows, coord, "by_coord", audit_rows)
    indexes["duplicate_audit"] = pd.DataFrame(audit_rows)
    return indexes

def find_resolution_row(sample_row: pd.Series, indexes: Dict) -> Optional[pd.Series]:
    all_rows = indexes["all_rows"]
    if all_rows.empty:
        return None

    for col, idxname in [
        ("sample_id", "by_sample_id"),
        ("sensitivity_sample_id", "by_sensitivity_id"),
        ("model_tile_uid", "by_model_tile_uid"),
    ]:
        if col in sample_row.index and pd.notna(sample_row[col]):
            idx = indexes[idxname].get(str(sample_row[col]))
            if idx is not None:
                return all_rows.loc[idx]

    if (
        pd.notna(sample_row.get("prefix", np.nan))
        and pd.notna(sample_row.get("x", np.nan))
        and pd.notna(sample_row.get("y", np.nan))
    ):
        key = (
            str(sample_row["prefix"]),
            as_int(sample_row["x"], -1),
            as_int(sample_row["y"], -1),
        )
        idx = indexes["by_coord"].get(key)
        if idx is not None:
            return all_rows.loc[idx]
    return None

MODEL_TILE_PATH_COLUMNS = [
    "model_tile_path", "tile_path", "image_path", "extracted_tile_path", "output_tile_path"
]
WSI_PATH_COLUMNS = ["wsi_path", "slide_path"]

def find_existing_model_tile(row: pd.Series) -> Optional[Path]:
    for col in MODEL_TILE_PATH_COLUMNS:
        if col in row.index:
            p = clean_path(row[col])
            if p is not None and p.exists():
                return p
    return None

def resolve_sources(sample_row: pd.Series, indexes: Dict, default_patch_size: int) -> Dict:
    resolution_row = find_resolution_row(sample_row, indexes)

    direct_extracted = find_existing_model_tile(sample_row)
    manifest_extracted = (
        find_existing_model_tile(resolution_row)
        if resolution_row is not None
        else None
    )

    combined_sources = [sample_row]
    if resolution_row is not None:
        combined_sources.append(resolution_row)

    wsi_path = None
    x = None
    y = None
    patch_size = None

    for source in combined_sources:
        if wsi_path is None:
            for col in WSI_PATH_COLUMNS:
                if col in source.index:
                    p = clean_path(source[col])
                    if p is not None and p.exists():
                        wsi_path = p
                        break
        if x is None:
            x = first_value(source, ["x", "tile_x"])
        if y is None:
            y = first_value(source, ["y", "tile_y"])
        if patch_size is None:
            patch_size = first_value(source, ["patch_size"])

    x = as_int(x, None)
    y = as_int(y, None)
    patch_size = as_int(patch_size, default_patch_size)

    native_available = (
        wsi_path is not None and wsi_path.exists()
        and x is not None and y is not None and patch_size is not None
    )

    if direct_extracted is not None:
        chosen_method = "direct_extracted_model_tile"
        chosen_path = direct_extracted
        pixel_source_class = "extracted_model_tile"
    elif manifest_extracted is not None:
        chosen_method = "extraction_manifest_model_tile"
        chosen_path = manifest_extracted
        pixel_source_class = "extracted_model_tile"
    elif native_available:
        chosen_method = "native_tiff_crop"
        chosen_path = None
        pixel_source_class = "native_tiff_crop"
    else:
        chosen_method = None
        chosen_path = None
        pixel_source_class = None

    return {
        "resolution_status": "resolved" if chosen_method else "unresolved",
        "resolution_method": chosen_method,
        "pixel_source_class": pixel_source_class,
        "chosen_extracted_path": str(chosen_path) if chosen_path else None,
        "direct_extracted_path": str(direct_extracted) if direct_extracted else None,
        "manifest_extracted_path": str(manifest_extracted) if manifest_extracted else None,
        "native_wsi_path": str(wsi_path) if wsi_path else None,
        "native_x": x,
        "native_y": y,
        "native_patch_size": patch_size,
        "native_crop_available": bool(native_available),
        "both_extracted_and_native_available":
            bool((direct_extracted or manifest_extracted) and native_available),
    }

def open_cached_slide(slide_path: str) -> Image.Image:
    if slide_path in SLIDE_CACHE:
        img = SLIDE_CACHE.pop(slide_path)
        SLIDE_CACHE[slide_path] = img
        return img

    print(f"[OPEN WSI] {slide_path}")
    img = Image.open(slide_path).convert("RGB")
    SLIDE_CACHE[slide_path] = img

    while len(SLIDE_CACHE) > MAX_CACHED_SLIDES:
        _, old = SLIDE_CACHE.popitem(last=False)
        try:
            old.close()
        except Exception:
            pass
    return img

def crop_native_with_bounds_check(
    slide_path: str,
    x: int,
    y: int,
    size: int,
) -> Image.Image:
    img = open_cached_slide(slide_path)

    if (
        x < 0
        or y < 0
        or x + size > img.width
        or y + size > img.height
    ):
        raise ValueError(
            "Requested native TIFF crop exceeds slide bounds: "
            f"x={x}, y={y}, size={size}, "
            f"slide_size=({img.width}, {img.height})"
        )

    return img.crop((x, y, x + size, y + size)).convert("RGB")

def load_chosen_patch(resolution: Dict) -> Image.Image:
    method = resolution["resolution_method"]

    if method in {"direct_extracted_model_tile", "extraction_manifest_model_tile"}:
        return Image.open(resolution["chosen_extracted_path"]).convert("RGB")

    if method == "native_tiff_crop":
        return crop_native_with_bounds_check(
            resolution["native_wsi_path"],
            int(resolution["native_x"]),
            int(resolution["native_y"]),
            int(resolution["native_patch_size"]),
        )

    raise RuntimeError(f"Unknown resolution method: {method}")

def load_native_patch_if_available(resolution: Dict) -> Optional[Image.Image]:
    if not resolution["native_crop_available"]:
        return None
    return crop_native_with_bounds_check(
        resolution["native_wsi_path"],
        int(resolution["native_x"]),
        int(resolution["native_y"]),
        int(resolution["native_patch_size"]),
    )

def rgb_to_od(rgb):
    rgb = rgb.astype(np.float32)
    rgb = np.maximum(rgb, 1.0)
    return -np.log(rgb / 255.0)

def remove_border_connected_components(binary_mask, structure):
    lab, n = label(binary_mask, structure=structure)
    if n == 0:
        empty = np.zeros_like(binary_mask, dtype=bool)
        return empty.copy(), empty
    border_labels = np.unique(np.concatenate([
        lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]
    ]))
    border_labels = border_labels[border_labels != 0]
    border = np.isin(lab, border_labels)
    filtered = binary_mask & (~border)
    return filtered, border

def component_areas(binary_mask, minimum_area_pixels, structure):
    lab, n = label(binary_mask, structure=structure)
    if n == 0:
        return []
    counts = np.bincount(lab.ravel())
    return [
        int(area)
        for cid, area in enumerate(counts)
        if cid != 0 and area >= minimum_area_pixels
    ]

def compute_features(patch: Image.Image, thresholds: Dict) -> Dict:
    rgb = np.asarray(patch.convert("RGB")).astype(np.uint8)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    eps = float(thresholds["eps"])

    # Every morphology threshold comes from Step 15a.
    basic = thresholds["basic_masks"]
    qc = thresholds["quality_filters"]
    purple = thresholds["purple_mask"]
    dark = thresholds["dark_mask"]
    lumen = thresholds["historical_lumen_mask"]
    eosin = thresholds["eosin_mask"]
    comp = thresholds["lumen_connected_components"]

    white_thr = float(basic["white_rgb_min"])
    black_thr = float(basic["black_rgb_max"])

    white_mask = (r > white_thr) & (g > white_thr) & (b > white_thr)
    black_mask = (r < black_thr) & (g < black_thr) & (b < black_thr)
    tissue_mask = (~white_mask) & (~black_mask)

    total_area = float(rgb.shape[0] * rgb.shape[1])
    tissue_area = float(tissue_mask.sum())
    if tissue_area < 1:
        raise ValueError("No tissue pixels detected.")

    tissue_fraction = tissue_area / total_area
    white_fraction = float(white_mask.mean())
    black_fraction = float(black_mask.mean())

    purple_mask = (
        tissue_mask
        & (b > r * float(purple["b_over_r_min"]))
        & (b > g * float(purple["b_over_g_min"]))
        & (r < float(purple["r_max"]))
        & (g < float(purple["g_max"]))
    )

    dark_mask = (
        tissue_mask
        & (((r + g + b) / 3.0) < float(dark["mean_rgb_max"]))
    )

    lumen_mask = (
        (r > float(lumen["r_min"]))
        & (g > float(lumen["g_min"]))
        & (b > float(lumen["b_min"]))
    )

    eosin_mask = (
        tissue_mask
        & (r > g * float(eosin["r_over_g_min"]))
        & (r > b * float(eosin["r_over_b_min"]))
        & (r > float(eosin["r_min"]))
        & (g > float(eosin["g_min"]))
        & (~purple_mask)
    )

    od = rgb_to_od(rgb)
    hema_od = float(np.mean(od[:, :, 2][tissue_mask]))

    denominator_mask = tissue_mask | lumen_mask
    denominator = float(denominator_mask.sum()) + eps

    npi = float(purple_mask.sum()) / denominator
    goi = float(lumen_mask.sum()) / denominator
    sei = float(eosin_mask.sum()) / denominator
    dark_fraction = float(dark_mask.sum()) / denominator
    clr = npi / (goi + eps)
    csr = npi / (sei + eps)

    structure = generate_binary_structure(rank=2, connectivity=1)
    min_area = int(comp["minimum_component_area_pixels"])

    areas = component_areas(lumen_mask, min_area, structure)
    lumen_count = len(areas)
    lumen_area = float(np.sum(areas))
    median_lumen_area = float(np.median(areas)) if areas else 0.0
    mean_lumen_area = float(np.mean(areas)) if areas else 0.0
    lumen_component_density = lumen_count / denominator
    lumen_fragmentation = lumen_count / (lumen_area + eps)

    lumen_be, border_open = remove_border_connected_components(lumen_mask, structure)
    sens_den_mask = tissue_mask | lumen_be
    sens_den = float(sens_den_mask.sum()) + eps

    goi_be = float(lumen_be.sum()) / sens_den
    npi_be = float(purple_mask.sum()) / sens_den
    sei_be = float(eosin_mask.sum()) / sens_den
    clr_be = npi_be / (goi_be + eps)

    areas_be = component_areas(lumen_be, min_area, structure)
    count_be = len(areas_be)
    area_be = float(np.sum(areas_be))
    med_be = float(np.median(areas_be)) if areas_be else 0.0
    mean_be = float(np.mean(areas_be)) if areas_be else 0.0

    return {
        "image_width_pixels": int(rgb.shape[1]),
        "image_height_pixels": int(rgb.shape[0]),
        "tissue_fraction": tissue_fraction,
        "white_fraction": white_fraction,
        "black_fraction": black_fraction,

        "SEI_eosin_stroma": sei,
        "GOI_lumen_openness": goi,
        "NPI_nuclear_packing": npi,

        "CLR_nuclear_to_lumen": clr,
        "CSR_nuclear_to_stroma": csr,
        "dark_fraction": dark_fraction,
        "hematoxylin_OD_proxy": hema_od,
        "lumen_count": lumen_count,
        "lumen_area": lumen_area,
        "median_lumen_area": median_lumen_area,
        "mean_lumen_area": mean_lumen_area,
        "lumen_component_density": lumen_component_density,
        "lumen_fragmentation": lumen_fragmentation,

        "GOI_lumen_openness_border_excluded": goi_be,
        "NPI_nuclear_packing_border_excluded_denominator": npi_be,
        "SEI_eosin_stroma_border_excluded_denominator": sei_be,
        "CLR_nuclear_to_lumen_border_excluded": clr_be,
        "lumen_count_border_excluded": count_be,
        "lumen_area_border_excluded": area_be,
        "median_lumen_area_border_excluded": med_be,
        "mean_lumen_area_border_excluded": mean_be,
        "lumen_component_density_border_excluded": count_be / sens_den,
        "lumen_fragmentation_border_excluded": count_be / (area_be + eps),
        "border_connected_open_fraction_all": float(border_open.mean()),
    }

def evaluate_qc(features: Dict, thresholds: Dict):
    qc = thresholds["quality_filters"]
    reasons = []
    if features["tissue_fraction"] < float(qc["minimum_tissue_fraction"]):
        reasons.append("tissue_fraction_below_minimum")
    if features["white_fraction"] > float(qc["maximum_white_fraction"]):
        reasons.append("white_fraction_above_maximum")
    if features["black_fraction"] > float(qc["maximum_black_fraction"]):
        reasons.append("black_fraction_above_maximum")
    return len(reasons) == 0, reasons

def pixel_equivalence_record(
    extracted: Image.Image,
    native: Image.Image,
    thresholds: Dict,
) -> Dict:
    a = np.asarray(extracted.convert("RGB")).astype(np.int16)
    b = np.asarray(native.convert("RGB")).astype(np.int16)

    if a.shape != b.shape:
        return {
            "equivalence_tested": True,
            "shape_equal": False,
            "exact_rgb_equal": False,
            "mean_abs_rgb_difference": np.nan,
            "max_abs_rgb_difference": np.nan,
            "SEI_abs_difference": np.nan,
            "GOI_abs_difference": np.nan,
            "NPI_abs_difference": np.nan,
        }

    diff = np.abs(a - b)
    fa = compute_features(extracted, thresholds)
    fb = compute_features(native, thresholds)

    return {
        "equivalence_tested": True,
        "shape_equal": True,
        "exact_rgb_equal": bool(np.array_equal(a, b)),
        "mean_abs_rgb_difference": float(diff.mean()),
        "max_abs_rgb_difference": int(diff.max()),
        "SEI_abs_difference": abs(fa["SEI_eosin_stroma"] - fb["SEI_eosin_stroma"]),
        "GOI_abs_difference": abs(fa["GOI_lumen_openness"] - fb["GOI_lumen_openness"]),
        "NPI_abs_difference": abs(fa["NPI_nuclear_packing"] - fb["NPI_nuclear_packing"]),
    }

def process_samples(samples, thresholds, indexes):
    tile_rows = []
    failure_rows = []
    resolution_rows = []
    equivalence_rows = []

    total = len(samples)

    for pos, (_, sample) in enumerate(samples.iterrows(), start=1):
        resolution = resolve_sources(
            sample,
            indexes=indexes,
            default_patch_size=int(thresholds["patch_size_default"]),
        )

        base = {
            "source_step": sample.get("source_step", np.nan),
            "sampling_stratum": sample.get("sampling_stratum", np.nan),
            "dataset": sample.get("dataset", np.nan),
            "patient_id": sample.get("patient_id", np.nan),
            "prefix": sample.get("prefix", np.nan),
            "frozen_tile_key": sample.get("frozen_tile_key", np.nan),
            "sample_id": sample.get("sample_id", np.nan),
            "sensitivity_sample_id": sample.get("sensitivity_sample_id", np.nan),
            "model_tile_uid": sample.get("model_tile_uid", np.nan),
            "tile_idx": sample.get("tile_idx", np.nan),
            "x": sample.get("x", np.nan),
            "y": sample.get("y", np.nan),
            "patch_size": sample.get("patch_size", np.nan),
            "pc3_score": sample.get("pc3_score", np.nan),
            "pc3_within_slide_percentile":
                sample.get("pc3_within_slide_percentile", np.nan),
            "tumor_overlap_fraction": sample.get("tumor_overlap_fraction", np.nan),
        }

        rr = dict(base)
        rr.update(resolution)
        resolution_rows.append(rr)

        if resolution["resolution_status"] != "resolved":
            failure_rows.append({
                **base,
                "failure_stage": "image_resolution",
                "failure_reason": "no valid extracted tile or native TIFF crop source",
            })
            continue

        # CORE formal measurement: only failures here can exclude the tile.
        try:
            chosen_patch = load_chosen_patch(resolution)
            features = compute_features(chosen_patch, thresholds)
            qc_pass, qc_reasons = evaluate_qc(features, thresholds)

        except Exception as e:
            failure_rows.append({
                **base,
                "failure_stage": "feature_extraction_or_bounds",
                "failure_reason": repr(e),
                "resolution_method": resolution["resolution_method"],
                "pixel_source_class": resolution["pixel_source_class"],
            })
            continue

        if not qc_pass:
            failure_rows.append({
                **base,
                "failure_stage": "quality_filter",
                "failure_reason": ";".join(qc_reasons),
                "resolution_method": resolution["resolution_method"],
                "pixel_source_class": resolution["pixel_source_class"],
                "tissue_fraction": features["tissue_fraction"],
                "white_fraction": features["white_fraction"],
                "black_fraction": features["black_fraction"],
            })
            continue

        tile_rows.append({
            **base,
            "resolution_method": resolution["resolution_method"],
            "pixel_source_class": resolution["pixel_source_class"],
            "chosen_extracted_path": resolution["chosen_extracted_path"],
            "native_wsi_path": resolution["native_wsi_path"],
            "qc_pass": True,
            **features,
        })

        # OPTIONAL equivalence audit: failure MUST NOT remove a valid measurement.
        if (
            resolution["pixel_source_class"] == "extracted_model_tile"
            and resolution["native_crop_available"]
        ):
            eq_row = {
                **base,
                "resolution_method": resolution["resolution_method"],
                "pixel_source_class": resolution["pixel_source_class"],
                "extracted_path": resolution["chosen_extracted_path"],
                "native_wsi_path": resolution["native_wsi_path"],
                "equivalence_audit_status": "not_attempted",
                "equivalence_audit_error": None,
            }
            try:
                native_patch = load_native_patch_if_available(resolution)
                eq_row.update(pixel_equivalence_record(chosen_patch, native_patch, thresholds))
                eq_row["equivalence_audit_status"] = "success"
            except Exception as e:
                eq_row.update({
                    "equivalence_tested": False,
                    "shape_equal": np.nan,
                    "exact_rgb_equal": np.nan,
                    "mean_abs_rgb_difference": np.nan,
                    "max_abs_rgb_difference": np.nan,
                    "SEI_abs_difference": np.nan,
                    "GOI_abs_difference": np.nan,
                    "NPI_abs_difference": np.nan,
                    "equivalence_audit_status": "failed_optional_audit",
                    "equivalence_audit_error": repr(e),
                })
            equivalence_rows.append(eq_row)

        if pos % 250 == 0 or pos == total:
            print(
                f"[PROGRESS] {pos}/{total} | kept={len(tile_rows)} "
                f"| failed={len(failure_rows)} | equivalence={len(equivalence_rows)}"
            )

    return (
        pd.DataFrame(tile_rows),
        pd.DataFrame(failure_rows),
        pd.DataFrame(resolution_rows),
        pd.DataFrame(equivalence_rows),
    )

def median_value(s):
    x = pd.to_numeric(s, errors="coerce")
    return float(x.median()) if x.notna().sum() else np.nan

def aggregate_slide(tile_df):
    rows = []
    group_cols = ["source_step", "sampling_stratum", "dataset", "patient_id", "prefix"]
    metrics = [
        "pc3_score", "pc3_within_slide_percentile", "tumor_overlap_fraction",
        "tissue_fraction", "white_fraction", "black_fraction"
    ] + ALL_FEATURES

    for keys, sub in tile_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_tiles"] = len(sub)
        row["n_pixel_source_classes"] = sub["pixel_source_class"].nunique()
        row["pixel_source_classes"] = ";".join(
            sorted(sub["pixel_source_class"].dropna().astype(str).unique())
        )
        for m in metrics:
            if m in sub.columns:
                row[m] = median_value(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_patient(slide_df):
    rows = []
    group_cols = ["source_step", "sampling_stratum", "patient_id"]
    metrics = [
        "pc3_score", "pc3_within_slide_percentile", "tumor_overlap_fraction",
        "tissue_fraction", "white_fraction", "black_fraction"
    ] + ALL_FEATURES

    for keys, sub in slide_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_slides"] = len(sub)
        row["n_tiles_total"] = int(pd.to_numeric(sub["n_tiles"], errors="coerce").sum())
        for m in metrics:
            if m in sub.columns:
                row[m] = median_value(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)

def build_resolution_method_audit(resolution_df):
    if resolution_df.empty:
        return pd.DataFrame()
    resolved = resolution_df[
        resolution_df["resolution_status"] == "resolved"
    ].copy()
    return (
        resolved
        .groupby(
            ["source_step", "sampling_stratum", "resolution_method", "pixel_source_class"],
            dropna=False,
        )
        .agg(
            n_tiles=("frozen_tile_key", "size"),
            n_patients=("patient_id", "nunique"),
            n_slides=("prefix", "nunique"),
        )
        .reset_index()
    )

def build_sampling_audit(samples, tile_df, failures_df, resolution_df):
    requested = (
        samples.groupby(["source_step", "sampling_stratum"], dropna=False)
        .agg(
            n_requested_tiles=("frozen_tile_key", "size"),
            n_requested_patients=("patient_id", "nunique"),
            n_requested_slides=("prefix", "nunique"),
        )
        .reset_index()
    )

    resolved = (
        resolution_df[resolution_df["resolution_status"] == "resolved"]
        .groupby(["source_step", "sampling_stratum"], dropna=False)
        .size().rename("n_resolved_tiles").reset_index()
        if not resolution_df.empty
        else pd.DataFrame(columns=["source_step", "sampling_stratum", "n_resolved_tiles"])
    )

    kept = (
        tile_df.groupby(["source_step", "sampling_stratum"], dropna=False)
        .agg(
            n_qc_pass_tiles=("frozen_tile_key", "size"),
            n_qc_pass_patients=("patient_id", "nunique"),
            n_qc_pass_slides=("prefix", "nunique"),
        )
        .reset_index()
        if not tile_df.empty
        else pd.DataFrame(columns=[
            "source_step","sampling_stratum","n_qc_pass_tiles",
            "n_qc_pass_patients","n_qc_pass_slides"
        ])
    )

    failed = (
        failures_df.groupby(["source_step", "sampling_stratum"], dropna=False)
        .size().rename("n_failed_tiles").reset_index()
        if not failures_df.empty
        else pd.DataFrame(columns=["source_step","sampling_stratum","n_failed_tiles"])
    )

    out = requested.merge(resolved, how="left").merge(kept, how="left").merge(failed, how="left")
    for c in [
        "n_resolved_tiles","n_qc_pass_tiles","n_qc_pass_patients",
        "n_qc_pass_slides","n_failed_tiles"
    ]:
        out[c] = out[c].fillna(0).astype(int)

    out["resolution_fraction"] = out["n_resolved_tiles"] / out["n_requested_tiles"]
    out["qc_pass_fraction"] = out["n_qc_pass_tiles"] / out["n_requested_tiles"]
    return out

def build_feature_coverage(tile_df, slide_df, patient_df):
    rows = []
    for feature in ALL_FEATURES:
        endpoint_class = (
            "primary" if feature in PRIMARY_FEATURES
            else "secondary" if feature in SECONDARY_FEATURES
            else "sensitivity"
        )
        for stratum in sorted(tile_df["sampling_stratum"].dropna().unique()):
            t = tile_df[tile_df["sampling_stratum"] == stratum]
            s = slide_df[slide_df["sampling_stratum"] == stratum]
            p = patient_df[patient_df["sampling_stratum"] == stratum]
            rows.append({
                "sampling_stratum": stratum,
                "feature": feature,
                "endpoint_class": endpoint_class,
                "n_tile_values": pd.to_numeric(t[feature], errors="coerce").notna().sum(),
                "n_slide_values": pd.to_numeric(s[feature], errors="coerce").notna().sum(),
                "n_patient_values": pd.to_numeric(p[feature], errors="coerce").notna().sum(),
            })
    return pd.DataFrame(rows)

def main():
    args = parse_args()

    print("=" * 120)
    print("STEP 15B v1.4 HARDENED PC3 QUANTITATIVE MORPHOLOGY")
    print("=" * 120)

    guard_outputs(args.overwrite)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plan, thresholds, feature_spec = load_frozen_specification()
    samples = load_frozen_samples()

    if args.max_tiles is not None:
        if args.max_tiles <= 0:
            raise ValueError("--max-tiles must be positive")
        print("[WARNING] ENGINEERING DEBUG RUN ONLY")
        samples = samples.head(args.max_tiles).copy()

    indexes = build_resolution_indexes()
    duplicate_audit_df = indexes.get("duplicate_audit", pd.DataFrame())

    tile_df, failures_df, resolution_df, equivalence_df = process_samples(
        samples, thresholds, indexes
    )

    if tile_df.empty:
        resolution_df.to_csv(OUT_RESOLUTION, index=False)
        failures_df.to_csv(OUT_QC_FAILURES, index=False)
        equivalence_df.to_csv(OUT_EQUIVALENCE, index=False)
        raise RuntimeError("No tiles passed measurement/QC.")

    slide_df = aggregate_slide(tile_df)
    patient_df = aggregate_patient(slide_df)
    audit_df = build_sampling_audit(samples, tile_df, failures_df, resolution_df)
    resolution_audit_df = build_resolution_method_audit(resolution_df)
    coverage_df = build_feature_coverage(tile_df, slide_df, patient_df)

    tile_df.to_csv(OUT_TILE, index=False)
    failures_df.to_csv(OUT_QC_FAILURES, index=False)
    slide_df.to_csv(OUT_SLIDE, index=False)
    patient_df.to_csv(OUT_PATIENT, index=False)
    coverage_df.to_csv(OUT_COVERAGE, index=False)
    audit_df.to_csv(OUT_AUDIT, index=False)
    resolution_df.to_csv(OUT_RESOLUTION, index=False)
    resolution_audit_df.to_csv(OUT_RESOLUTION_AUDIT, index=False)
    equivalence_df.to_csv(OUT_EQUIVALENCE, index=False)
    duplicate_audit_df.to_csv(OUT_RESOLUTION_DUPLICATES, index=False)

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_full_run": args.max_tiles is None,
        "debug_max_tiles": args.max_tiles,
        "step15a": {
            "version": plan["version"],
            "schema_version": plan["schema_version"],
            "plan_path": str(STEP15A_PLAN),
            "plan_sha256": sha256_file(STEP15A_PLAN),
            "threshold_path": str(STEP15A_THRESHOLDS),
            "threshold_sha256": sha256_file(STEP15A_THRESHOLDS),
            "feature_spec_path": str(STEP15A_FEATURE_SPEC),
            "feature_spec_sha256": sha256_file(STEP15A_FEATURE_SPEC),
            "metadata_path": str(STEP15A_METADATA),
            "metadata_sha256": sha256_file(STEP15A_METADATA),
        },
        "step13a_sha256": sha256_file(STEP13A_MANIFEST),
        "step13c_sha256": sha256_file(STEP13C_MANIFEST),
        "n_requested_tiles": len(samples),
        "n_qc_pass_tiles": len(tile_df),
        "n_failed_tiles": len(failures_df),
        "n_equivalence_pairs": len(equivalence_df),
        "n_equivalence_audit_failures": int(
            (equivalence_df.get("equivalence_audit_status", pd.Series(dtype=str)) == "failed_optional_audit").sum()
        ),
        "n_resolution_duplicate_audit_rows": len(duplicate_audit_df),

        "resolution_duplicate_identity_rule": (
            "Prefer WSI path + x + y + patch_size as physical identity; "
            "different exported PNG filenames at the same physical location "
            "are legitimate duplicate exports and are deterministically "
            "deduplicated for lookup only. Conflicting physical identities "
            "for the same logical lookup key abort the run."
        ),

        "optional_equivalence_audit_rule": (
            "Failure of extracted-vs-native pixel-equivalence audit cannot "
            "remove an otherwise valid formal morphology measurement."
        ),

        "inferential_statistics_performed": False,
        "threshold_ownership":
            "all morphology thresholds read from Step-15a frozen JSON",
        "pixel_scale_statement":
            "Measurements are performed on frozen sampled image patches. "
            "Pixel-area features are interpreted only in pixel units; no physical-area "
            "interpretation is made because reliable MPP metadata are unavailable.",
    }
    json.dump(metadata, OUT_METADATA.open("w"), indent=2)

    OUT_README.write_text(
        """STEP 15B v1.4 HARDENED PC3 MORPHOLOGY MEASUREMENT
================================================

No inferential statistics are performed here.

Critical implementation guarantees:
  - all morphology thresholds come from Step-15a v2.2 JSON;
  - native TIFF crops are bounds-checked;
  - resolution method and pixel source class are audited by sampling stratum;
  - extracted-vs-native equivalence is OPTIONAL QC and can never remove a successful formal measurement;
  - duplicate resolution keys are explicitly audited;
  - physical identity is preferentially defined by WSI path + x + y + patch_size;
  - different exported PNG filenames from the same WSI coordinate are treated as legitimate duplicate exports of one physical tile;
  - identical physical references are deterministically deduplicated for lookup only;
  - conflicting physical identities for the same logical key abort the run;
  - border-excluded lumen analysis is only a border-connected background sensitivity,
    not corrected lumen segmentation;
  - pixel-area features are reported only in pixel units.

Review these files BEFORE Step 15c:
  pc3_quantitative_morphology_sampling_audit.csv
  pc3_quantitative_morphology_feature_coverage.csv
  pc3_quantitative_morphology_resolution_method_audit.csv
  pc3_quantitative_morphology_pixel_source_equivalence_audit.csv
  pc3_quantitative_morphology_resolution_duplicate_audit.csv
  pc3_quantitative_morphology_qc_failures.csv
""",
        encoding="utf-8",
    )

    print("\nSAMPLING AUDIT")
    print(audit_df.to_string(index=False))

    print("\nRESOLUTION METHOD AUDIT")
    print(resolution_audit_df.to_string(index=False))

    print("\nPIXEL SOURCE EQUIVALENCE")
    if equivalence_df.empty:
        print("No extracted/native pairs available for equivalence testing.")
    else:
        cols = [
            "sampling_stratum","pixel_source_class","exact_rgb_equal",
            "mean_abs_rgb_difference","max_abs_rgb_difference",
            "SEI_abs_difference","GOI_abs_difference","NPI_abs_difference"
        ]
        print(equivalence_df[[c for c in cols if c in equivalence_df.columns]].head(50).to_string(index=False))

    print("\n[DONE] Step 15b v1.4 measurement complete.")
    print("[NEXT] Review QC/audit only; do not inspect PC3 high-vs-low morphology before 15c.")

if __name__ == "__main__":
    main()
