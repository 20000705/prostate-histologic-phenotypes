#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_nested_leave_one_patient_out_pca_tumor_validation.py

Purpose
-------
Perform leakage-controlled nested leave-one-patient-out (LOPO) validation of
tumor-associated latent pathology axes.

For each outer held-out patient:

    1. Remove the held-out patient completely.
    2. Fit StandardScaler + PCA using only the remaining training patients.
    3. Match fold-specific PCA components to the frozen full-cohort reference
       axes using loading-space Hungarian matching.
    4. Determine the tumor-associated direction for each matched component
       using TRAINING PATIENTS ONLY.
    5. Freeze that training-derived direction within the fold.
    6. Project the held-out patient's tile embeddings using the fold-specific
       scaler + PCA.
    7. Evaluate tumor localization AUC on held-out slides using only the
       training-frozen direction.

Critical anti-leakage rule
--------------------------
The held-out patient's:
    - embeddings do not contribute to scaler fitting,
    - embeddings do not contribute to PCA fitting,
    - tumor labels do not contribute to component direction selection,
    - tumor AUC does not influence sign flipping or component selection.

This directly addresses circularity concerns that arise when PCA is fit once on
the complete cohort before leave-one-patient-out evaluation.

Component identity
------------------
Fold-specific PCs are matched to the full-cohort frozen PC1-PC8 reference basis
using one-to-one Hungarian assignment on absolute cosine similarity between
effective loading vectors. The full-cohort reference is used only to define
axis identity, not to estimate fold-specific scaler/PCA parameters or tumor
direction.

Direction specification
-----------------------
Within each outer fold, tumor direction is derived from TRAINING slides only.

A direction is frozen in a fold when:
    - at least MIN_TRAIN_DIRECTION_CONSENSUS of eligible training slides agree,
    - at least MIN_TRAIN_SLIDES_FOR_DIRECTION eligible training slides exist.

Otherwise the component is marked ambiguous in that fold and no held-out
fixed-direction AUC is reported for that component.

Tumor tile definition
---------------------
A tile is tumor-associated when at least 10% of its area overlaps the union of
XML tumor annotation polygons.

Inputs
------
../wsi_embeddings/embedding_manifest.csv
../wsi_embeddings/pca_global/global_pca_model.npz
../wsi_embeddings/*_embeddings.npy
../wsi_embeddings/*_coords.npy
../wsi_embeddings/*_meta.json
XML annotations from ANNOT_DIR

Outputs
-------
MedIA_paper/results/10_nested_lopo_pca_tumor_validation/
    analysis_metadata.json
    outer_fold_summary.csv
    nested_component_summary.csv
    heldout_slide_results.csv
    training_direction_results.csv
    excluded_slides.csv
    component_match_frequency.csv

    holdout_aaaXXXX/
        fold_pca_model.npz
        fold_component_matches.csv
        training_direction_summary.csv
        heldout_slide_results.csv
        fit_sampling_manifest.csv
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    RESULTS_ROOT,
    DISCOVERY_EMBEDDINGS,
    DISCOVERY_ANNOTATIONS,
    FROZEN_PCA_MODEL,
)

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EMB_DIR = DISCOVERY_EMBEDDINGS
MANIFEST_PATH = EMB_DIR / "embedding_manifest.csv"
REFERENCE_PCA_DIR = FROZEN_PCA_MODEL.parent
REFERENCE_MODEL_PATH = FROZEN_PCA_MODEL
ANNOT_DIR = DISCOVERY_ANNOTATIONS

OUT_DIR = RESULTS_ROOT / "10_nested_lopo_pca_tumor_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_FOLD_SUMMARY = OUT_DIR / "outer_fold_summary.csv"
OUT_COMPONENT_SUMMARY = OUT_DIR / "nested_component_summary.csv"
OUT_HELDOUT_RESULTS = OUT_DIR / "heldout_slide_results.csv"
OUT_TRAIN_DIRECTION = OUT_DIR / "training_direction_results.csv"
OUT_EXCLUDED = OUT_DIR / "excluded_slides.csv"
OUT_MATCH_FREQUENCY = OUT_DIR / "component_match_frequency.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================
N_COMPONENTS = 8

PATCH_SIZE_DEFAULT = 224
TUMOR_OVERLAP_THRESHOLD = 0.10
MIN_TUMOR_TILES = 5
MIN_NON_TUMOR_TILES = 20

# Same cap used in 08/09 for consistency.
MAX_TILES_PER_SLIDE_FOR_FIT = 2000

# Direction is determined only from training slides in each outer fold.
MIN_TRAIN_DIRECTION_CONSENSUS = 0.80
MIN_TRAIN_SLIDES_FOR_DIRECTION = 20

RANDOM_STATE = 42
USE_FLOAT32 = True
EPS = 1e-12


# =============================================================================
# GENERIC HELPERS
# =============================================================================
def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def check_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required directory: {path}")


def normalize_name(value: str) -> str:
    stem = Path(str(value)).stem.lower()
    return re.sub(r"[^a-z0-9]", "", stem)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_manifest_path(path_value) -> str:
    raw = Path(str(path_value).strip())

    if raw.is_absolute():
        return str(raw)

    if raw.exists():
        return str(raw.resolve())

    if raw.parts and raw.parts[0] == "wsi_embeddings":
        candidate = EMB_DIR / Path(*raw.parts[1:])
        if candidate.exists():
            return str(candidate.resolve())

    candidate = MANIFEST_PATH.parent / raw
    if candidate.exists():
        return str(candidate.resolve())

    candidate = EMB_DIR / raw.name
    if candidate.exists():
        return str(candidate.resolve())

    if raw.parts and raw.parts[0] == "wsi_embeddings":
        return str(
            (EMB_DIR / Path(*raw.parts[1:])).resolve()
        )

    return str(
        (MANIFEST_PATH.parent / raw).resolve()
    )


def build_xml_map(directory: Path) -> Dict[str, Path]:
    xml_map: Dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() == ".xml":
            xml_map[normalize_name(path.name)] = path

    return xml_map


def find_xml_by_prefix(
    prefix: str,
    xml_map: Dict[str, Path],
) -> Optional[Path]:
    key = normalize_name(prefix)

    if key in xml_map:
        return xml_map[key]

    candidates = [
        path
        for normalized, path in xml_map.items()
        if key in normalized or normalized in key
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        candidates = sorted(
            candidates,
            key=lambda path: abs(
                len(normalize_name(path.name)) - len(key)
            ),
        )
        return candidates[0]

    return None


def safe_auc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    if len(np.unique(labels)) < 2:
        return np.nan

    if np.std(scores) <= EPS:
        return 0.5

    try:
        return float(
            roc_auc_score(
                labels,
                scores,
            )
        )
    except Exception:
        return np.nan


# =============================================================================
# MANIFEST
# =============================================================================
def validate_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "patient_id",
        "wsi_label",
        "prefix",
        "embeddings_path",
        "coords_path",
        "meta_path",
    }

    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError("Embedding manifest is empty.")

    out = manifest.copy()

    for column in [
        "patient_id",
        "wsi_label",
        "prefix",
    ]:
        out[column] = out[column].astype(str)

    for column in [
        "embeddings_path",
        "coords_path",
        "meta_path",
    ]:
        out[column] = (
            out[column]
            .astype(str)
            .map(resolve_manifest_path)
        )

    if out["prefix"].duplicated().any():
        duplicated = out.loc[
            out["prefix"].duplicated(keep=False),
            "prefix",
        ].tolist()
        raise ValueError(
            f"Duplicate prefixes in manifest: {duplicated[:20]}"
        )

    for column in [
        "embeddings_path",
        "coords_path",
        "meta_path",
    ]:
        missing_paths = [
            path
            for path in out[column]
            if not Path(path).exists()
        ]

        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            raise FileNotFoundError(
                f"{len(missing_paths)} missing files in "
                f"{column}.\n{preview}"
            )

    return (
        out
        .sort_values(
            [
                "patient_id",
                "wsi_label",
                "prefix",
            ]
        )
        .reset_index(drop=True)
    )


# =============================================================================
# DATA LOADING
# =============================================================================
def load_embeddings(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got {x.shape}: {path}"
        )

    if USE_FLOAT32:
        x = np.asarray(x, dtype=np.float32)
    else:
        x = np.asarray(x)

    if not np.isfinite(x).all():
        raise ValueError(
            f"Non-finite embedding values: {path}"
        )

    return x


def build_embedding_cache(
    manifest: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    cache = {}

    for _, row in manifest.iterrows():
        x = load_embeddings(
            row["embeddings_path"]
        )
        cache[row["prefix"]] = x

        print(
            f"[LOAD] {row['prefix']} | "
            f"shape={x.shape}"
        )

    return cache


# =============================================================================
# REFERENCE PCA / MATCHING
# =============================================================================
def load_reference_model(
    path: Path,
) -> Dict[str, np.ndarray]:
    model = np.load(path)

    required = {
        "scaler_mean",
        "scaler_scale",
        "pca_components",
        "pca_mean",
        "pca_explained_variance_ratio",
    }

    missing = required - set(model.files)

    if missing:
        raise ValueError(
            f"Reference model missing arrays: {sorted(missing)}"
        )

    output = {
        key: np.asarray(model[key])
        for key in model.files
    }

    if (
        output["pca_components"].shape[0]
        < N_COMPONENTS
    ):
        raise ValueError(
            f"Reference PCA has fewer than {N_COMPONENTS} PCs."
        )

    return output


def effective_loadings(
    pca_components: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    if np.any(scaler_scale <= EPS):
        raise ValueError(
            "Scaler contains zero or near-zero feature scales."
        )

    loadings = (
        pca_components
        / scaler_scale[np.newaxis, :]
    )

    norms = np.linalg.norm(
        loadings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= EPS):
        raise ValueError(
            "Zero-norm effective loading encountered."
        )

    return loadings / norms


def match_components(
    reference_effective: np.ndarray,
    fold_effective: np.ndarray,
) -> pd.DataFrame:
    signed_cosine = (
        reference_effective
        @ fold_effective.T
    )

    absolute_cosine = np.abs(
        signed_cosine
    )

    ref_indices, fold_indices = (
        linear_sum_assignment(
            -absolute_cosine
        )
    )

    rows = []

    for ref_idx, fold_idx in zip(
        ref_indices,
        fold_indices,
    ):
        cosine = float(
            signed_cosine[
                ref_idx,
                fold_idx,
            ]
        )

        sign_multiplier = (
            1 if cosine >= 0 else -1
        )

        rows.append({
            "reference_component_index": int(ref_idx),
            "reference_component": f"PC{ref_idx + 1}",
            "fold_component_index": int(fold_idx),
            "fold_component": f"PC{fold_idx + 1}",
            "loading_cosine_before_sign_alignment": cosine,
            "absolute_loading_cosine": abs(cosine),
            "sign_multiplier_for_reference_alignment": (
                int(sign_multiplier)
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "reference_component_index"
        )
        .reset_index(drop=True)
    )


# =============================================================================
# XML / TILE LABELING
# =============================================================================
def load_annotation_polygons(
    xml_path: Optional[Path],
) -> List[Polygon]:
    polygons: List[Polygon] = []

    if xml_path is None or not xml_path.exists():
        return polygons

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for region in root.iter("Region"):
        points = []

        for vertex in region.iter("Vertex"):
            if (
                "X" in vertex.attrib
                and "Y" in vertex.attrib
            ):
                points.append(
                    (
                        float(vertex.attrib["X"]),
                        float(vertex.attrib["Y"]),
                    )
                )

        if len(points) < 3:
            continue

        polygon = Polygon(points)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)

    return polygons


def classify_tiles_by_overlap(
    coords: np.ndarray,
    patch_size: int,
    polygons: List[Polygon],
) -> np.ndarray:
    tumor_mask = np.zeros(
        len(coords),
        dtype=bool,
    )

    if not polygons:
        return tumor_mask

    tumor_union = unary_union(
        polygons
    )
    tile_area = float(
        patch_size * patch_size
    )

    for index, (x, y) in enumerate(coords):
        tile = box(
            float(x),
            float(y),
            float(x) + patch_size,
            float(y) + patch_size,
        )

        overlap = (
            tumor_union
            .intersection(tile)
            .area
            / tile_area
        )

        tumor_mask[index] = (
            overlap
            >= TUMOR_OVERLAP_THRESHOLD
        )

    return tumor_mask


def load_slide_labels(
    row: pd.Series,
    xml_map: Dict[str, Path],
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    prefix = str(row["prefix"])
    patient_id = str(row["patient_id"])

    xml_path = find_xml_by_prefix(
        prefix,
        xml_map,
    )

    if xml_path is None:
        return None, {
            "patient_id": patient_id,
            "prefix": prefix,
            "stage": "annotation",
            "reason": "missing_xml",
            "details": "",
        }

    try:
        polygons = load_annotation_polygons(
            xml_path
        )
    except Exception as error:
        return None, {
            "patient_id": patient_id,
            "prefix": prefix,
            "stage": "annotation",
            "reason": "xml_parse_error",
            "details": repr(error),
        }

    if len(polygons) == 0:
        return None, {
            "patient_id": patient_id,
            "prefix": prefix,
            "stage": "annotation",
            "reason": "xml_without_valid_polygons",
            "details": str(xml_path),
        }

    coords = np.load(
        row["coords_path"],
        mmap_mode="r",
    )

    meta = load_json(
        Path(row["meta_path"])
    )

    patch_size = int(
        meta.get(
            "patch_size",
            PATCH_SIZE_DEFAULT,
        )
    )

    tumor_mask = classify_tiles_by_overlap(
        coords=np.asarray(coords),
        patch_size=patch_size,
        polygons=polygons,
    )

    n_tumor = int(
        tumor_mask.sum()
    )
    n_non = int(
        (~tumor_mask).sum()
    )

    if n_tumor < MIN_TUMOR_TILES:
        return None, {
            "patient_id": patient_id,
            "prefix": prefix,
            "stage": "eligibility",
            "reason": "insufficient_tumor_tiles",
            "details": (
                f"n_tumor={n_tumor};"
                f"required={MIN_TUMOR_TILES}"
            ),
        }

    if n_non < MIN_NON_TUMOR_TILES:
        return None, {
            "patient_id": patient_id,
            "prefix": prefix,
            "stage": "eligibility",
            "reason": "insufficient_non_tumor_tiles",
            "details": (
                f"n_non={n_non};"
                f"required={MIN_NON_TUMOR_TILES}"
            ),
        }

    info = {
        "xml_path": str(xml_path),
        "patch_size": patch_size,
        "n_tumor_tiles": n_tumor,
        "n_non_tumor_tiles": n_non,
        "n_tiles": int(
            len(tumor_mask)
        ),
    }

    return tumor_mask, info


# =============================================================================
# FOLD FITTING
# =============================================================================
def build_fit_matrix(
    training_manifest: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(
        random_state
    )

    blocks = []
    rows = []

    for _, row in training_manifest.iterrows():
        prefix = row["prefix"]
        x = embedding_cache[prefix]

        n_available = int(
            x.shape[0]
        )

        n_used = min(
            n_available,
            MAX_TILES_PER_SLIDE_FOR_FIT,
        )

        if n_available > n_used:
            indices = rng.choice(
                n_available,
                size=n_used,
                replace=False,
            )
            indices.sort()
            x_use = x[indices]
            sampling_mode = (
                "random_subsample_without_replacement"
            )
        else:
            x_use = x
            sampling_mode = (
                "all_available_tiles"
            )

        blocks.append(
            x_use
        )

        rows.append({
            "patient_id": row["patient_id"],
            "wsi_label": row["wsi_label"],
            "prefix": prefix,
            "n_tiles_available": n_available,
            "n_tiles_used_for_fit": n_used,
            "sampling_mode": sampling_mode,
        })

    x_fit = np.concatenate(
        blocks,
        axis=0,
    )

    return (
        x_fit,
        pd.DataFrame(rows),
    )


def fit_fold_pca(
    x_fit: np.ndarray,
    random_state: int,
) -> Tuple[StandardScaler, PCA]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(
        x_fit
    )

    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="auto",
        random_state=random_state,
    )

    pca.fit(
        x_scaled
    )

    return scaler, pca


def save_fold_model(
    path: Path,
    scaler: StandardScaler,
    pca: PCA,
    holdout_patient: str,
    random_state: int,
    n_fit_tiles: int,
) -> None:
    np.savez(
        path,
        scaler_mean=np.asarray(
            scaler.mean_
        ),
        scaler_scale=np.asarray(
            scaler.scale_
        ),
        pca_components=np.asarray(
            pca.components_
        ),
        pca_mean=np.asarray(
            pca.mean_
        ),
        pca_explained_variance=np.asarray(
            pca.explained_variance_
        ),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_
        ),
        holdout_patient=np.array(
            [holdout_patient]
        ),
        random_state=np.array(
            [random_state],
            dtype=np.int64,
        ),
        n_fit_tiles=np.array(
            [n_fit_tiles],
            dtype=np.int64,
        ),
    )


# =============================================================================
# TRAINING-ONLY DIRECTION SPECIFICATION
# =============================================================================
def project_aligned_reference_scores(
    x: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    matches: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Project tiles using fold PCA and return scores indexed by reference PC name,
    sign-aligned to the reference PCA orientation.
    """
    raw_scores = pca.transform(
        scaler.transform(x)
    )

    output: Dict[str, np.ndarray] = {}

    for _, match in matches.iterrows():
        ref_pc = str(
            match["reference_component"]
        )
        fold_idx = int(
            match["fold_component_index"]
        )
        sign_multiplier = int(
            match[
                "sign_multiplier_for_reference_alignment"
            ]
        )

        output[ref_pc] = (
            raw_scores[:, fold_idx]
            * sign_multiplier
        )

    return output


def compute_training_direction_summary(
    training_manifest: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    xml_map: Dict[str, Path],
    scaler: StandardScaler,
    pca: PCA,
    matches: pd.DataFrame,
    holdout_patient: str,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Determine one tumor-associated direction per reference PC using only
    eligible training slides.
    """
    per_slide_rows = []
    exclusions = []

    for _, row in training_manifest.iterrows():
        tumor_mask, label_info = (
            load_slide_labels(
                row=row,
                xml_map=xml_map,
            )
        )

        if tumor_mask is None:
            exclusion = dict(
                label_info
            )
            exclusion[
                "holdout_patient"
            ] = holdout_patient
            exclusion[
                "dataset_role"
            ] = "training"
            exclusions.append(
                exclusion
            )
            continue

        x = embedding_cache[
            row["prefix"]
        ]

        if (
            x.shape[0]
            != len(tumor_mask)
        ):
            exclusions.append({
                "holdout_patient": holdout_patient,
                "dataset_role": "training",
                "patient_id": row["patient_id"],
                "prefix": row["prefix"],
                "stage": "validation",
                "reason": "embedding_label_length_mismatch",
                "details": (
                    f"embeddings={x.shape[0]};"
                    f"labels={len(tumor_mask)}"
                ),
            })
            continue

        aligned_scores = (
            project_aligned_reference_scores(
                x=x,
                scaler=scaler,
                pca=pca,
                matches=matches,
            )
        )

        for ref_pc, scores in aligned_scores.items():
            tumor_mean = float(
                np.mean(
                    scores[tumor_mask]
                )
            )
            non_tumor_mean = float(
                np.mean(
                    scores[
                        ~tumor_mask
                    ]
                )
            )

            delta = (
                tumor_mean
                - non_tumor_mean
            )

            direction = (
                "high_in_tumor"
                if delta >= 0
                else "low_in_tumor"
            )

            per_slide_rows.append({
                "holdout_patient": holdout_patient,
                "training_patient_id": row["patient_id"],
                "prefix": row["prefix"],
                "reference_component": ref_pc,
                "direction": direction,
                "delta_tumor_minus_non": delta,
                "n_tumor_tiles": (
                    label_info[
                        "n_tumor_tiles"
                    ]
                ),
                "n_non_tumor_tiles": (
                    label_info[
                        "n_non_tumor_tiles"
                    ]
                ),
            })

    per_slide = pd.DataFrame(
        per_slide_rows
    )

    if per_slide.empty:
        raise RuntimeError(
            f"No eligible training slides in fold {holdout_patient}."
        )

    rows = []

    for pc in [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]:
        group = per_slide.loc[
            per_slide[
                "reference_component"
            ] == pc
        ].copy()

        n_slides = int(
            len(group)
        )

        n_high = int(
            (
                group["direction"]
                == "high_in_tumor"
            ).sum()
        )
        n_low = int(
            (
                group["direction"]
                == "low_in_tumor"
            ).sum()
        )

        if n_high > n_low:
            majority = (
                "high_in_tumor"
            )
            majority_count = (
                n_high
            )
        elif n_low > n_high:
            majority = (
                "low_in_tumor"
            )
            majority_count = (
                n_low
            )
        else:
            majority = "tie"
            majority_count = (
                n_high
            )

        consensus = (
            majority_count / n_slides
            if n_slides > 0
            else np.nan
        )

        if (
            n_slides
            < MIN_TRAIN_SLIDES_FOR_DIRECTION
        ):
            status = (
                "insufficient_data"
            )
            frozen_direction = ""
        elif majority == "tie":
            status = "ambiguous"
            frozen_direction = ""
        elif (
            consensus
            < MIN_TRAIN_DIRECTION_CONSENSUS
        ):
            status = "ambiguous"
            frozen_direction = ""
        else:
            status = "frozen"
            frozen_direction = (
                majority
            )

        rows.append({
            "holdout_patient": holdout_patient,
            "reference_component": pc,
            "n_training_slides": n_slides,
            "n_high_in_tumor": n_high,
            "n_low_in_tumor": n_low,
            "majority_direction": majority,
            "consensus_fraction": consensus,
            "minimum_consensus_required": (
                MIN_TRAIN_DIRECTION_CONSENSUS
            ),
            "minimum_training_slides_required": (
                MIN_TRAIN_SLIDES_FOR_DIRECTION
            ),
            "specification_status": status,
            "training_frozen_direction": (
                frozen_direction
            ),
        })

    return (
        pd.DataFrame(rows),
        exclusions,
    )


# =============================================================================
# HELD-OUT EVALUATION
# =============================================================================
def evaluate_heldout_patient(
    holdout_manifest: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    xml_map: Dict[str, Path],
    scaler: StandardScaler,
    pca: PCA,
    matches: pd.DataFrame,
    direction_summary: pd.DataFrame,
    holdout_patient: str,
) -> Tuple[pd.DataFrame, List[dict]]:
    rows = []
    exclusions = []

    direction_map = {
        str(row["reference_component"]): {
            "status": str(
                row[
                    "specification_status"
                ]
            ),
            "direction": str(
                row[
                    "training_frozen_direction"
                ]
            ),
            "consensus": float(
                row[
                    "consensus_fraction"
                ]
            ),
        }
        for _, row in direction_summary.iterrows()
    }

    for _, row in holdout_manifest.iterrows():
        tumor_mask, label_info = (
            load_slide_labels(
                row=row,
                xml_map=xml_map,
            )
        )

        if tumor_mask is None:
            exclusion = dict(
                label_info
            )
            exclusion[
                "holdout_patient"
            ] = holdout_patient
            exclusion[
                "dataset_role"
            ] = "heldout"
            exclusions.append(
                exclusion
            )
            continue

        x = embedding_cache[
            row["prefix"]
        ]

        if (
            x.shape[0]
            != len(tumor_mask)
        ):
            exclusions.append({
                "holdout_patient": holdout_patient,
                "dataset_role": "heldout",
                "patient_id": row["patient_id"],
                "prefix": row["prefix"],
                "stage": "validation",
                "reason": "embedding_label_length_mismatch",
                "details": (
                    f"embeddings={x.shape[0]};"
                    f"labels={len(tumor_mask)}"
                ),
            })
            continue

        aligned_scores = (
            project_aligned_reference_scores(
                x=x,
                scaler=scaler,
                pca=pca,
                matches=matches,
            )
        )

        labels = (
            tumor_mask.astype(int)
        )

        for ref_pc, scores in aligned_scores.items():
            direction_info = (
                direction_map[
                    ref_pc
                ]
            )

            if (
                direction_info["status"]
                != "frozen"
            ):
                fixed_auc = np.nan
                score_multiplier = np.nan
            else:
                direction = (
                    direction_info[
                        "direction"
                    ]
                )

                if (
                    direction
                    == "high_in_tumor"
                ):
                    score_multiplier = 1
                    eval_scores = scores
                elif (
                    direction
                    == "low_in_tumor"
                ):
                    score_multiplier = -1
                    eval_scores = -scores
                else:
                    score_multiplier = np.nan
                    eval_scores = (
                        np.full_like(
                            scores,
                            np.nan,
                        )
                    )

                fixed_auc = safe_auc(
                    labels,
                    eval_scores,
                )

            raw_auc = safe_auc(
                labels,
                scores,
            )

            inverse_auc = safe_auc(
                labels,
                -scores,
            )

            tumor_mean = float(
                np.mean(
                    scores[tumor_mask]
                )
            )
            non_tumor_mean = float(
                np.mean(
                    scores[
                        ~tumor_mask
                    ]
                )
            )

            rows.append({
                "holdout_patient": holdout_patient,
                "heldout_slide_patient_id": row["patient_id"],
                "prefix": row["prefix"],
                "reference_component": ref_pc,
                "training_specification_status": (
                    direction_info[
                        "status"
                    ]
                ),
                "training_frozen_direction": (
                    direction_info[
                        "direction"
                    ]
                ),
                "training_direction_consensus": (
                    direction_info[
                        "consensus"
                    ]
                ),
                "score_multiplier": (
                    score_multiplier
                ),
                "heldout_auc_raw_score": (
                    raw_auc
                ),
                "heldout_auc_negative_score": (
                    inverse_auc
                ),
                "heldout_fixed_direction_auc": (
                    fixed_auc
                ),
                "heldout_tumor_mean_score": (
                    tumor_mean
                ),
                "heldout_non_tumor_mean_score": (
                    non_tumor_mean
                ),
                "heldout_delta_tumor_minus_non": (
                    tumor_mean
                    - non_tumor_mean
                ),
                "n_tiles": (
                    label_info[
                        "n_tiles"
                    ]
                ),
                "n_tumor_tiles": (
                    label_info[
                        "n_tumor_tiles"
                    ]
                ),
                "n_non_tumor_tiles": (
                    label_info[
                        "n_non_tumor_tiles"
                    ]
                ),
            })

    return (
        pd.DataFrame(rows),
        exclusions,
    )


# =============================================================================
# SUMMARIES
# =============================================================================
def make_outer_fold_summary(
    heldout_results: pd.DataFrame,
    training_directions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for holdout_patient in sorted(
        heldout_results[
            "holdout_patient"
        ].unique()
    ):
        held = heldout_results.loc[
            heldout_results[
                "holdout_patient"
            ] == holdout_patient
        ]

        train = training_directions.loc[
            training_directions[
                "holdout_patient"
            ] == holdout_patient
        ]

        rows.append({
            "holdout_patient": (
                holdout_patient
            ),
            "n_heldout_slides": int(
                held[
                    "prefix"
                ].nunique()
            ),
            "n_components_with_frozen_training_direction": int(
                (
                    train[
                        "specification_status"
                    ]
                    == "frozen"
                ).sum()
            ),
            "n_heldout_component_slide_rows": int(
                len(held)
            ),
            "n_heldout_fixed_direction_auc_values": int(
                held[
                    "heldout_fixed_direction_auc"
                ].notna().sum()
            ),
        })

    return pd.DataFrame(
        rows
    )


def make_nested_component_summary(
    heldout_results: pd.DataFrame,
    matches_all: pd.DataFrame,
    training_directions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for pc in [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]:
        held = heldout_results.loc[
            heldout_results[
                "reference_component"
            ] == pc
        ].copy()

        match = matches_all.loc[
            matches_all[
                "reference_component"
            ] == pc
        ].copy()

        train = training_directions.loc[
            training_directions[
                "reference_component"
            ] == pc
        ].copy()

        auc = pd.to_numeric(
            held[
                "heldout_fixed_direction_auc"
            ],
            errors="coerce",
        )

        frozen_fold_fraction = float(
            (
                train[
                    "specification_status"
                ]
                == "frozen"
            ).mean()
        )

        n_auc = int(
            auc.notna().sum()
        )

        if n_auc > 0:
            mean_auc = float(
                auc.mean()
            )
            median_auc = float(
                auc.median()
            )
            sd_auc = float(
                auc.std(
                    ddof=1
                )
            ) if n_auc > 1 else np.nan
            q25_auc = float(
                auc.quantile(
                    0.25
                )
            )
            q75_auc = float(
                auc.quantile(
                    0.75
                )
            )
            auc_gt_half_fraction = float(
                (
                    auc.dropna()
                    > 0.5
                ).mean()
            )
        else:
            mean_auc = np.nan
            median_auc = np.nan
            sd_auc = np.nan
            q25_auc = np.nan
            q75_auc = np.nan
            auc_gt_half_fraction = np.nan

        rows.append({
            "Component": pc,
            "N outer folds": int(
                train[
                    "holdout_patient"
                ].nunique()
            ),
            "Fraction of folds with frozen training direction": (
                frozen_fold_fraction
            ),
            "N held-out slide AUC values": (
                n_auc
            ),
            "Mean held-out fixed-direction AUC": (
                mean_auc
            ),
            "Median held-out fixed-direction AUC": (
                median_auc
            ),
            "SD held-out fixed-direction AUC": (
                sd_auc
            ),
            "Q25 held-out fixed-direction AUC": (
                q25_auc
            ),
            "Q75 held-out fixed-direction AUC": (
                q75_auc
            ),
            "Fraction held-out AUC > 0.5": (
                auc_gt_half_fraction
            ),
            "Mean LOPO loading cosine": float(
                match[
                    "absolute_loading_cosine"
                ].mean()
            ),
            "Median LOPO loading cosine": float(
                match[
                    "absolute_loading_cosine"
                ].median()
            ),
            "Minimum LOPO loading cosine": float(
                match[
                    "absolute_loading_cosine"
                ].min()
            ),
            "Most frequent fold component": (
                match[
                    "fold_component"
                ]
                .mode()
                .iloc[0]
            ),
            "Fold component consistency": float(
                (
                    match[
                        "fold_component"
                    ]
                    == match[
                        "fold_component"
                    ]
                    .mode()
                    .iloc[0]
                ).mean()
            ),
        })

    return pd.DataFrame(
        rows
    )


def make_component_match_frequency(
    matches_all: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    n_folds = int(
        matches_all[
            "holdout_patient"
        ].nunique()
    )

    for ref_pc in [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]:
        group = matches_all.loc[
            matches_all[
                "reference_component"
            ] == ref_pc
        ]

        for fold_pc in [
            f"PC{i}"
            for i in range(
                1,
                N_COMPONENTS + 1
            )
        ]:
            count = int(
                (
                    group[
                        "fold_component"
                    ]
                    == fold_pc
                ).sum()
            )

            rows.append({
                "reference_component": (
                    ref_pc
                ),
                "fold_component": (
                    fold_pc
                ),
                "match_count": count,
                "match_fraction": float(
                    count / n_folds
                ),
            })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    check_file(
        MANIFEST_PATH
    )
    check_file(
        REFERENCE_MODEL_PATH
    )
    check_directory(
        ANNOT_DIR
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest = validate_manifest(
        manifest
    )

    reference = (
        load_reference_model(
            REFERENCE_MODEL_PATH
        )
    )

    reference_effective = (
        effective_loadings(
            reference[
                "pca_components"
            ][
                :N_COMPONENTS
            ],
            reference[
                "scaler_scale"
            ],
        )
    )

    xml_map = build_xml_map(
        ANNOT_DIR
    )

    print("=" * 110)
    print(
        "NESTED LEAVE-ONE-PATIENT-OUT "
        "PCA TUMOR VALIDATION"
    )
    print("=" * 110)
    print(
        f"[MANIFEST]                         "
        f"{MANIFEST_PATH}"
    )
    print(
        f"[REFERENCE PCA]                    "
        f"{REFERENCE_MODEL_PATH}"
    )
    print(
        f"[ANNOTATIONS]                      "
        f"{ANNOT_DIR}"
    )
    print(
        f"[OUTPUT]                           "
        f"{OUT_DIR}"
    )
    print(
        f"[SLIDES]                           "
        f"{len(manifest)}"
    )
    print(
        f"[PATIENTS]                         "
        f"{manifest['patient_id'].nunique()}"
    )
    print(
        f"[COMPONENTS]                       "
        f"{N_COMPONENTS}"
    )
    print(
        f"[MAX TILES/SLIDE FOR FIT]          "
        f"{MAX_TILES_PER_SLIDE_FOR_FIT}"
    )
    print(
        f"[TRAIN DIRECTION CONSENSUS]        "
        f"{MIN_TRAIN_DIRECTION_CONSENSUS:.0%}"
    )
    print(
        f"[MIN TRAIN SLIDES FOR DIRECTION]   "
        f"{MIN_TRAIN_SLIDES_FOR_DIRECTION}"
    )

    print(
        "\n[INFO] Loading discovery embeddings..."
    )

    embedding_cache = (
        build_embedding_cache(
            manifest
        )
    )

    patients = sorted(
        manifest[
            "patient_id"
        ].unique()
    )

    all_matches = []
    all_training_directions = []
    all_heldout_results = []
    all_exclusions = []

    for fold_index, holdout_patient in enumerate(
        patients,
        start=1,
    ):
        fold_seed = (
            RANDOM_STATE
            + fold_index
        )

        fold_dir = (
            OUT_DIR
            / f"holdout_{holdout_patient}"
        )
        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n" + "-" * 110)
        print(
            f"[OUTER FOLD {fold_index:02d}/{len(patients)}] "
            f"Hold out {holdout_patient}"
        )
        print("-" * 110)

        training_manifest = (
            manifest.loc[
                manifest[
                    "patient_id"
                ] != holdout_patient
            ]
            .copy()
            .reset_index(drop=True)
        )

        heldout_manifest = (
            manifest.loc[
                manifest[
                    "patient_id"
                ] == holdout_patient
            ]
            .copy()
            .reset_index(drop=True)
        )

        x_fit, fit_sampling = (
            build_fit_matrix(
                training_manifest=(
                    training_manifest
                ),
                embedding_cache=(
                    embedding_cache
                ),
                random_state=(
                    fold_seed
                ),
            )
        )

        fit_sampling.to_csv(
            fold_dir
            / "fit_sampling_manifest.csv",
            index=False,
        )

        print(
            f"[FIT MATRIX] {x_fit.shape} | "
            f"train_patients="
            f"{training_manifest['patient_id'].nunique()} | "
            f"heldout_slides={len(heldout_manifest)}"
        )

        scaler, pca = fit_fold_pca(
            x_fit=x_fit,
            random_state=(
                fold_seed
            ),
        )

        save_fold_model(
            path=(
                fold_dir
                / "fold_pca_model.npz"
            ),
            scaler=scaler,
            pca=pca,
            holdout_patient=(
                holdout_patient
            ),
            random_state=(
                fold_seed
            ),
            n_fit_tiles=int(
                x_fit.shape[0]
            ),
        )

        fold_effective = (
            effective_loadings(
                pca.components_,
                scaler.scale_,
            )
        )

        matches = match_components(
            reference_effective=(
                reference_effective
            ),
            fold_effective=(
                fold_effective
            ),
        )

        matches[
            "holdout_patient"
        ] = holdout_patient

        matches[
            "fold_index"
        ] = int(
            fold_index
        )

        matches[
            "reference_explained_variance_ratio"
        ] = matches[
            "reference_component_index"
        ].map(
            lambda idx: float(
                reference[
                    "pca_explained_variance_ratio"
                ][
                    int(idx)
                ]
            )
        )

        matches[
            "fold_explained_variance_ratio"
        ] = matches[
            "fold_component_index"
        ].map(
            lambda idx: float(
                pca.explained_variance_ratio_[
                    int(idx)
                ]
            )
        )

        matches.to_csv(
            fold_dir
            / "fold_component_matches.csv",
            index=False,
        )

        all_matches.append(
            matches
        )

        print(
            "\n[COMPONENT MATCHING]"
        )
        print(
            matches[
                [
                    "reference_component",
                    "fold_component",
                    "absolute_loading_cosine",
                ]
            ].to_string(
                index=False
            )
        )

        training_direction, train_exclusions = (
            compute_training_direction_summary(
                training_manifest=(
                    training_manifest
                ),
                embedding_cache=(
                    embedding_cache
                ),
                xml_map=(
                    xml_map
                ),
                scaler=scaler,
                pca=pca,
                matches=matches,
                holdout_patient=(
                    holdout_patient
                ),
            )
        )

        training_direction.to_csv(
            fold_dir
            / "training_direction_summary.csv",
            index=False,
        )

        all_training_directions.append(
            training_direction
        )

        print(
            "\n[TRAINING-ONLY DIRECTION]"
        )
        print(
            training_direction[
                [
                    "reference_component",
                    "n_training_slides",
                    "n_high_in_tumor",
                    "n_low_in_tumor",
                    "consensus_fraction",
                    "specification_status",
                    "training_frozen_direction",
                ]
            ].to_string(
                index=False
            )
        )

        heldout_results, held_exclusions = (
            evaluate_heldout_patient(
                holdout_manifest=(
                    heldout_manifest
                ),
                embedding_cache=(
                    embedding_cache
                ),
                xml_map=(
                    xml_map
                ),
                scaler=scaler,
                pca=pca,
                matches=matches,
                direction_summary=(
                    training_direction
                ),
                holdout_patient=(
                    holdout_patient
                ),
            )
        )

        heldout_results.to_csv(
            fold_dir
            / "heldout_slide_results.csv",
            index=False,
        )

        all_heldout_results.append(
            heldout_results
        )

        all_exclusions.extend(
            train_exclusions
        )
        all_exclusions.extend(
            held_exclusions
        )

        print(
            "\n[HELD-OUT RESULTS]"
        )

        if heldout_results.empty:
            print(
                "No eligible held-out slides."
            )
        else:
            display = (
                heldout_results[
                    [
                        "prefix",
                        "reference_component",
                        "training_specification_status",
                        "training_frozen_direction",
                        "heldout_fixed_direction_auc",
                    ]
                ]
            )
            print(
                display.to_string(
                    index=False
                )
            )

        del x_fit

    matches_all = pd.concat(
        all_matches,
        ignore_index=True,
    )

    training_all = pd.concat(
        all_training_directions,
        ignore_index=True,
    )

    heldout_all = pd.concat(
        all_heldout_results,
        ignore_index=True,
    )

    matches_all.to_csv(
        OUT_DIR
        / "fold_component_matches_all.csv",
        index=False,
    )

    training_all.to_csv(
        OUT_TRAIN_DIRECTION,
        index=False,
    )

    heldout_all.to_csv(
        OUT_HELDOUT_RESULTS,
        index=False,
    )

    if all_exclusions:
        exclusions_df = pd.DataFrame(
            all_exclusions
        )
    else:
        exclusions_df = pd.DataFrame(
            columns=[
                "holdout_patient",
                "dataset_role",
                "patient_id",
                "prefix",
                "stage",
                "reason",
                "details",
            ]
        )

    exclusions_df.to_csv(
        OUT_EXCLUDED,
        index=False,
    )

    outer_summary = (
        make_outer_fold_summary(
            heldout_results=(
                heldout_all
            ),
            training_directions=(
                training_all
            ),
        )
    )

    outer_summary.to_csv(
        OUT_FOLD_SUMMARY,
        index=False,
    )

    component_summary = (
        make_nested_component_summary(
            heldout_results=(
                heldout_all
            ),
            matches_all=(
                matches_all
            ),
            training_directions=(
                training_all
            ),
        )
    )

    component_summary.to_csv(
        OUT_COMPONENT_SUMMARY,
        index=False,
    )

    match_frequency = (
        make_component_match_frequency(
            matches_all
        )
    )

    match_frequency.to_csv(
        OUT_MATCH_FREQUENCY,
        index=False,
    )

    metadata = {
        "analysis_name": (
            "Nested leave-one-patient-out PCA tumor validation"
        ),
        "analysis_role": (
            "Leakage-controlled discovery-cohort validation"
        ),
        "manifest_path": str(
            MANIFEST_PATH
        ),
        "reference_pca_model": str(
            REFERENCE_MODEL_PATH
        ),
        "annotation_directory": str(
            ANNOT_DIR
        ),
        "n_patients": int(
            manifest[
                "patient_id"
            ].nunique()
        ),
        "n_slides": int(
            len(manifest)
        ),
        "n_components": int(
            N_COMPONENTS
        ),
        "max_tiles_per_slide_for_fit": int(
            MAX_TILES_PER_SLIDE_FOR_FIT
        ),
        "tumor_overlap_threshold": float(
            TUMOR_OVERLAP_THRESHOLD
        ),
        "minimum_tumor_tiles": int(
            MIN_TUMOR_TILES
        ),
        "minimum_non_tumor_tiles": int(
            MIN_NON_TUMOR_TILES
        ),
        "minimum_training_direction_consensus": float(
            MIN_TRAIN_DIRECTION_CONSENSUS
        ),
        "minimum_training_slides_for_direction": int(
            MIN_TRAIN_SLIDES_FOR_DIRECTION
        ),
        "outer_fold_rule": (
            "One patient is completely excluded from scaler/PCA fitting "
            "and tumor-direction specification."
        ),
        "component_identity_rule": (
            "Fold PCs are matched to the frozen full-cohort reference axes "
            "using one-to-one Hungarian assignment on absolute effective-"
            "loading cosine similarity."
        ),
        "reference_basis_role": (
            "Used only to define component identity; not used to estimate "
            "fold scaler/PCA parameters or tumor direction."
        ),
        "direction_rule": (
            "Tumor direction is derived only from eligible training slides "
            "within each outer fold."
        ),
        "heldout_direction_optimization": False,
        "heldout_component_selection_from_auc": False,
        "heldout_labels_used_for_training": False,
        "outputs": {
            "outer_fold_summary": str(
                OUT_FOLD_SUMMARY
            ),
            "nested_component_summary": str(
                OUT_COMPONENT_SUMMARY
            ),
            "heldout_slide_results": str(
                OUT_HELDOUT_RESULTS
            ),
            "training_direction_results": str(
                OUT_TRAIN_DIRECTION
            ),
            "excluded_slides": str(
                OUT_EXCLUDED
            ),
            "component_match_frequency": str(
                OUT_MATCH_FREQUENCY
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

    print("\n" + "=" * 110)
    print(
        "NESTED LOPO PCA TUMOR VALIDATION SUMMARY"
    )
    print("=" * 110)

    print(
        component_summary.to_string(
            index=False
        )
    )

    print("\n[OK] Saved:")
    print(f"  {OUT_FOLD_SUMMARY}")
    print(f"  {OUT_COMPONENT_SUMMARY}")
    print(f"  {OUT_HELDOUT_RESULTS}")
    print(f"  {OUT_TRAIN_DIRECTION}")
    print(f"  {OUT_EXCLUDED}")
    print(f"  {OUT_MATCH_FREQUENCY}")
    print(f"  {OUT_METADATA}")

    print(
        "\n[DONE] Nested LOPO PCA tumor validation complete."
    )


if __name__ == "__main__":
    main()
