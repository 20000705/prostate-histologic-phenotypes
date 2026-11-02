#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_leave_one_patient_out_pca_stability.py

Purpose
-------
Evaluate whether any single discovery patient disproportionately determines the
frozen PCA basis.

For each leave-one-patient-out (LOPO) fold:

1. Remove one patient from the discovery cohort.
2. Fit a new StandardScaler + PCA on all remaining discovery patients.
3. Cap each slide at MAX_TILES_PER_SLIDE_FOR_FIT tiles for tractability.
4. Compare the refitted PC1-PC8 basis with the original frozen PCA basis.
5. Perform one-to-one Hungarian matching using absolute cosine similarity
   between effective loading vectors.
6. Align component signs after matching.
7. Compare tile scores on one fixed common evaluation tile sample.

This is a structural PCA stability / patient-influence analysis.

It does NOT:
    - compute tumor association
    - choose tumor direction
    - evaluate held-out-patient AUC
    - perform nested validation

Those are reserved for the subsequent nested-PCA validation step.

Cohort definition
-----------------
The cohort is defined exclusively by:

    ../wsi_embeddings/embedding_manifest.csv

Inputs
------
../wsi_embeddings/embedding_manifest.csv
../wsi_embeddings/pca_global/global_pca_model.npz
../wsi_embeddings/*_embeddings.npy

Outputs
-------
MedIA_paper/results/09_leave_one_patient_out_pca_stability/
    analysis_metadata.json
    lopo_component_matches.csv
    component_stability_summary.csv
    component_match_frequency.csv
    patient_influence_summary.csv
    score_comparison_sampling_manifest.csv
    fit_sampling_manifest.csv

    holdout_aaaXXXX/
        lopo_pca_model.npz
        lopo_explained_variance.csv
        component_matches.csv
        fit_sampling_manifest.csv
"""

import json
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

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EMB_DIR = DISCOVERY_EMBEDDINGS
MANIFEST_PATH = EMB_DIR / "embedding_manifest.csv"
PRIMARY_PCA_DIR = FROZEN_PCA_MODEL.parent
PRIMARY_MODEL_PATH = FROZEN_PCA_MODEL

OUT_DIR = RESULTS_ROOT / "09_leave_one_patient_out_pca_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_ALL_MATCHES = OUT_DIR / "lopo_component_matches.csv"
OUT_STABILITY_SUMMARY = OUT_DIR / "component_stability_summary.csv"
OUT_MATCH_FREQUENCY = OUT_DIR / "component_match_frequency.csv"
OUT_PATIENT_INFLUENCE = OUT_DIR / "patient_influence_summary.csv"
OUT_SCORE_SAMPLE = OUT_DIR / "score_comparison_sampling_manifest.csv"
OUT_FIT_SAMPLING = OUT_DIR / "fit_sampling_manifest.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================
N_COMPONENTS = 8

# Use the same per-slide cap as the bootstrap analysis for comparability.
MAX_TILES_PER_SLIDE_FOR_FIT = 2000

# Fixed common tile sample used only for score-correlation comparison.
MAX_EVAL_TILES_PER_SLIDE = 1000
EVAL_SAMPLE_RANDOM_STATE = 271828

RANDOM_STATE = 42
USE_FLOAT32 = True
EPS = 1e-12


# =============================================================================
# HELPERS
# =============================================================================
def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


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
            (
                EMB_DIR
                / Path(*raw.parts[1:])
            ).resolve()
        )

    return str(
        (
            MANIFEST_PATH.parent
            / raw
        ).resolve()
    )


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
            f"Non-finite values found: {path}"
        )

    return x


def validate_manifest(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "patient_id",
        "wsi_label",
        "prefix",
        "embeddings_path",
    }

    missing = required - set(
        manifest.columns
    )

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    if manifest.empty:
        raise ValueError(
            "Embedding manifest is empty."
        )

    out = manifest.copy()

    out["patient_id"] = (
        out["patient_id"].astype(str)
    )
    out["wsi_label"] = (
        out["wsi_label"].astype(str)
    )
    out["prefix"] = (
        out["prefix"].astype(str)
    )
    out["embeddings_path"] = (
        out["embeddings_path"]
        .astype(str)
        .map(resolve_manifest_path)
    )

    if out["prefix"].duplicated().any():
        duplicated = out.loc[
            out["prefix"].duplicated(
                keep=False
            ),
            "prefix",
        ].tolist()

        raise ValueError(
            "Duplicate prefixes in discovery manifest: "
            f"{duplicated[:20]}"
        )

    missing_paths = [
        path
        for path in out["embeddings_path"]
        if not Path(path).exists()
    ]

    if missing_paths:
        preview = "\n".join(
            missing_paths[:10]
        )
        raise FileNotFoundError(
            f"{len(missing_paths)} embedding files "
            f"are missing after path resolution.\n"
            f"{preview}"
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


def load_primary_model(
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

    missing = required - set(
        model.files
    )

    if missing:
        raise ValueError(
            "Primary model missing arrays: "
            f"{sorted(missing)}"
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
            "Primary PCA has fewer than "
            f"{N_COMPONENTS} components."
        )

    if np.any(
        output["scaler_scale"] <= EPS
    ):
        raise ValueError(
            "Primary scaler contains zero or "
            "near-zero scales."
        )

    return output


def inspect_slide_tile_counts(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    expected_dim = None

    for _, row in manifest.iterrows():
        x = np.load(
            row["embeddings_path"],
            mmap_mode="r",
        )

        if x.ndim != 2:
            raise ValueError(
                f"Invalid embedding shape: {x.shape}"
            )

        n_tiles, embedding_dim = x.shape

        if expected_dim is None:
            expected_dim = int(
                embedding_dim
            )
        elif (
            int(embedding_dim)
            != expected_dim
        ):
            raise ValueError(
                "Inconsistent embedding dimension: "
                f"{row['prefix']}"
            )

        rows.append({
            "patient_id": row["patient_id"],
            "wsi_label": row["wsi_label"],
            "prefix": row["prefix"],
            "embeddings_path": (
                row["embeddings_path"]
            ),
            "n_tiles_available": int(
                n_tiles
            ),
            "embedding_dim": int(
                embedding_dim
            ),
        })

    return pd.DataFrame(rows)


def effective_loadings(
    pca_components: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """
    Map standardized-space PCA components back into original feature space.
    """
    if np.any(
        scaler_scale <= EPS
    ):
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

    if np.any(
        norms <= EPS
    ):
        raise ValueError(
            "Zero-norm effective loading encountered."
        )

    return loadings / norms


def transform_with_npz_model(
    x: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> np.ndarray:
    x_scaled = (
        x - scaler_mean
    ) / scaler_scale

    return (
        x_scaled - pca_mean
    ) @ pca_components.T


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 3:
        return np.nan, np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if (
        np.std(x_valid) <= EPS
        or np.std(y_valid) <= EPS
    ):
        return np.nan, np.nan

    result = spearmanr(
        x_valid,
        y_valid,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


# =============================================================================
# COMMON SCORE-COMPARISON SAMPLE
# =============================================================================
def build_common_evaluation_sample(
    tile_counts: pd.DataFrame,
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build one fixed slide-aware tile sample used for all LOPO score comparisons.

    This sample is not used to fit any fold.
    """
    rng = np.random.default_rng(
        random_state
    )

    blocks: List[np.ndarray] = []
    rows = []

    for _, row in tile_counts.iterrows():
        x = load_embeddings(
            row["embeddings_path"]
        )

        n_available = int(
            x.shape[0]
        )

        n_eval = min(
            MAX_EVAL_TILES_PER_SLIDE,
            n_available,
        )

        indices = rng.choice(
            n_available,
            size=n_eval,
            replace=False,
        )
        indices.sort()

        blocks.append(
            x[indices]
        )

        rows.append({
            "patient_id": row["patient_id"],
            "wsi_label": row["wsi_label"],
            "prefix": row["prefix"],
            "n_tiles_available": n_available,
            "n_tiles_used_for_score_comparison": (
                n_eval
            ),
        })

    x_eval = np.concatenate(
        blocks,
        axis=0,
    )

    if not np.isfinite(
        x_eval
    ).all():
        raise ValueError(
            "Non-finite values in evaluation sample."
        )

    return (
        x_eval,
        pd.DataFrame(rows),
    )


# =============================================================================
# EMBEDDING CACHE / LOPO FIT SAMPLING
# =============================================================================
def build_embedding_cache(
    tile_counts: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    cache = {}

    for _, row in tile_counts.iterrows():
        prefix = row["prefix"]

        x = load_embeddings(
            row["embeddings_path"]
        )

        cache[prefix] = x

        print(
            f"[LOAD] {prefix} | "
            f"shape={x.shape}"
        )

    return cache


def build_lopo_fit_matrix(
    tile_counts: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    holdout_patient: str,
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build one LOPO PCA fitting matrix from all non-held-out patients.
    """
    rng = np.random.default_rng(
        random_state
    )

    blocks: List[np.ndarray] = []
    rows = []

    training_slides = (
        tile_counts.loc[
            tile_counts[
                "patient_id"
            ] != holdout_patient
        ]
        .copy()
        .sort_values(
            [
                "patient_id",
                "wsi_label",
                "prefix",
            ]
        )
    )

    for _, row in training_slides.iterrows():
        prefix = row["prefix"]
        x = embedding_cache[
            prefix
        ]

        n_available = int(
            x.shape[0]
        )

        n_used = min(
            n_available,
            MAX_TILES_PER_SLIDE_FOR_FIT,
        )

        if (
            n_available
            > n_used
        ):
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
            "holdout_patient": (
                holdout_patient
            ),
            "patient_id": row[
                "patient_id"
            ],
            "wsi_label": row[
                "wsi_label"
            ],
            "prefix": prefix,
            "n_tiles_available": (
                n_available
            ),
            "n_tiles_used_for_fit": (
                n_used
            ),
            "sampling_mode": (
                sampling_mode
            ),
        })

    if not blocks:
        raise RuntimeError(
            f"No training embeddings for holdout {holdout_patient}."
        )

    x_fit = np.concatenate(
        blocks,
        axis=0,
    )

    if not np.isfinite(
        x_fit
    ).all():
        raise ValueError(
            "Non-finite values in LOPO fit matrix."
        )

    return (
        x_fit,
        pd.DataFrame(rows),
    )


# =============================================================================
# PCA FITTING / COMPONENT MATCHING
# =============================================================================
def fit_lopo_pca(
    x_fit: np.ndarray,
    random_state: int,
) -> Tuple[StandardScaler, PCA]:
    if (
        x_fit.shape[0]
        <= N_COMPONENTS
    ):
        raise ValueError(
            "Too few samples for PCA fitting."
        )

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


def match_components(
    primary_effective: np.ndarray,
    lopo_effective: np.ndarray,
) -> pd.DataFrame:
    """
    One-to-one Hungarian matching maximizing total absolute cosine similarity.
    """
    signed_cosine = (
        primary_effective
        @ lopo_effective.T
    )

    absolute_cosine = np.abs(
        signed_cosine
    )

    primary_indices, lopo_indices = (
        linear_sum_assignment(
            -absolute_cosine
        )
    )

    rows = []

    for primary_idx, lopo_idx in zip(
        primary_indices,
        lopo_indices,
    ):
        cosine = float(
            signed_cosine[
                primary_idx,
                lopo_idx,
            ]
        )

        sign_multiplier = (
            1
            if cosine >= 0
            else -1
        )

        rows.append({
            "primary_component_index": int(
                primary_idx
            ),
            "primary_component": (
                f"PC{primary_idx + 1}"
            ),
            "lopo_component_index": int(
                lopo_idx
            ),
            "lopo_component": (
                f"PC{lopo_idx + 1}"
            ),
            "loading_cosine_before_sign_alignment": (
                cosine
            ),
            "absolute_loading_cosine": (
                abs(cosine)
            ),
            "sign_multiplier_for_alignment": (
                int(sign_multiplier)
            ),
            "loading_cosine_after_sign_alignment": (
                abs(cosine)
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "primary_component_index"
        )
        .reset_index(drop=True)
    )


def save_lopo_model(
    path: Path,
    scaler: StandardScaler,
    pca: PCA,
    holdout_patient: str,
    random_state: int,
    n_fit_tiles: int,
    n_training_patients: int,
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
        pca_singular_values=np.asarray(
            pca.singular_values_
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
        n_training_patients=np.array(
            [n_training_patients],
            dtype=np.int32,
        ),
    )


# =============================================================================
# SUMMARY TABLES
# =============================================================================
def make_component_stability_summary(
    all_matches: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for component in [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]:
        group = all_matches.loc[
            all_matches[
                "primary_component"
            ] == component
        ].copy()

        matched_counts = (
            group[
                "lopo_component"
            ]
            .value_counts()
            .sort_values(
                ascending=False
            )
        )

        most_common = (
            matched_counts.index[0]
        )
        most_common_count = int(
            matched_counts.iloc[0]
        )

        rows.append({
            "Primary component": component,
            "N LOPO folds": int(
                group[
                    "holdout_patient"
                ].nunique()
            ),
            "Most frequent matched LOPO component": (
                most_common
            ),
            "Matched component consistency": float(
                most_common_count
                / group[
                    "holdout_patient"
                ].nunique()
            ),
            "Mean absolute loading cosine": float(
                group[
                    "absolute_loading_cosine"
                ].mean()
            ),
            "Median absolute loading cosine": float(
                group[
                    "absolute_loading_cosine"
                ].median()
            ),
            "Minimum absolute loading cosine": float(
                group[
                    "absolute_loading_cosine"
                ].min()
            ),
            "Maximum absolute loading cosine": float(
                group[
                    "absolute_loading_cosine"
                ].max()
            ),
            "SD absolute loading cosine": float(
                group[
                    "absolute_loading_cosine"
                ].std(
                    ddof=1
                )
            ),
            "Mean tile-score Spearman rho": float(
                group[
                    "tile_score_spearman_rho"
                ].mean()
            ),
            "Median tile-score Spearman rho": float(
                group[
                    "tile_score_spearman_rho"
                ].median()
            ),
            "Minimum tile-score Spearman rho": float(
                group[
                    "tile_score_spearman_rho"
                ].min()
            ),
            "Maximum tile-score Spearman rho": float(
                group[
                    "tile_score_spearman_rho"
                ].max()
            ),
            "SD tile-score Spearman rho": float(
                group[
                    "tile_score_spearman_rho"
                ].std(
                    ddof=1
                )
            ),
            "Mean matched LOPO explained variance ratio": float(
                group[
                    "lopo_explained_variance_ratio"
                ].mean()
            ),
            "Median matched LOPO explained variance ratio": float(
                group[
                    "lopo_explained_variance_ratio"
                ].median()
            ),
            "Primary explained variance ratio": float(
                group[
                    "primary_explained_variance_ratio"
                ].iloc[0]
            ),
            "Cosine >= 0.80 fraction": float(
                (
                    group[
                        "absolute_loading_cosine"
                    ] >= 0.80
                ).mean()
            ),
            "Cosine >= 0.90 fraction": float(
                (
                    group[
                        "absolute_loading_cosine"
                    ] >= 0.90
                ).mean()
            ),
            "Score rho >= 0.80 fraction": float(
                (
                    group[
                        "tile_score_spearman_rho"
                    ] >= 0.80
                ).mean()
            ),
            "Score rho >= 0.90 fraction": float(
                (
                    group[
                        "tile_score_spearman_rho"
                    ] >= 0.90
                ).mean()
            ),
        })

    summary = pd.DataFrame(
        rows
    )

    numeric_cols = (
        summary
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    summary[
        numeric_cols
    ] = summary[
        numeric_cols
    ].round(6)

    return summary


def make_match_frequency_table(
    all_matches: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    total_folds = int(
        all_matches[
            "holdout_patient"
        ].nunique()
    )

    for primary_component in [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]:
        group = all_matches.loc[
            all_matches[
                "primary_component"
            ] == primary_component
        ]

        for lopo_component in [
            f"PC{i}"
            for i in range(
                1,
                N_COMPONENTS + 1
            )
        ]:
            count = int(
                (
                    group[
                        "lopo_component"
                    ]
                    == lopo_component
                ).sum()
            )

            rows.append({
                "primary_component": (
                    primary_component
                ),
                "lopo_component": (
                    lopo_component
                ),
                "match_count": count,
                "match_fraction": float(
                    count / total_folds
                ),
            })

    return pd.DataFrame(
        rows
    )


def make_patient_influence_summary(
    all_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each primary PC, identify the held-out patient that caused the largest
    structural deviation from the full-cohort axis.
    """
    rows = []

    for component in [
        f"PC{i}"
        for i in range(
            1,
            N_COMPONENTS + 1
        )
    ]:
        group = all_matches.loc[
            all_matches[
                "primary_component"
            ] == component
        ].copy()

        min_cos_idx = group[
            "absolute_loading_cosine"
        ].idxmin()

        min_rho_idx = group[
            "tile_score_spearman_rho"
        ].idxmin()

        min_cos_row = group.loc[
            min_cos_idx
        ]

        min_rho_row = group.loc[
            min_rho_idx
        ]

        rows.append({
            "Primary component": component,
            "Most influential patient by loading cosine": (
                min_cos_row[
                    "holdout_patient"
                ]
            ),
            "Minimum absolute loading cosine": float(
                min_cos_row[
                    "absolute_loading_cosine"
                ]
            ),
            "Matched LOPO component at minimum cosine": (
                min_cos_row[
                    "lopo_component"
                ]
            ),
            "Most influential patient by score rho": (
                min_rho_row[
                    "holdout_patient"
                ]
            ),
            "Minimum tile-score Spearman rho": float(
                min_rho_row[
                    "tile_score_spearman_rho"
                ]
            ),
            "Matched LOPO component at minimum rho": (
                min_rho_row[
                    "lopo_component"
                ]
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
        PRIMARY_MODEL_PATH
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    manifest = validate_manifest(
        manifest
    )

    primary = load_primary_model(
        PRIMARY_MODEL_PATH
    )

    tile_counts = inspect_slide_tile_counts(
        manifest
    )

    embedding_dim = int(
        tile_counts[
            "embedding_dim"
        ].iloc[0]
    )

    if (
        primary[
            "pca_components"
        ].shape[1]
        != embedding_dim
    ):
        raise ValueError(
            "Primary PCA embedding dimension "
            "does not match manifest embeddings."
        )

    patients = sorted(
        tile_counts[
            "patient_id"
        ].unique()
    )

    n_patients = len(
        patients
    )

    if (
        n_patients < 3
    ):
        raise ValueError(
            "LOPO analysis requires at least three patients."
        )

    print("=" * 100)
    print(
        "LEAVE-ONE-PATIENT-OUT PCA STABILITY"
    )
    print("=" * 100)
    print(
        f"[MANIFEST]                   "
        f"{MANIFEST_PATH}"
    )
    print(
        f"[PRIMARY MODEL]              "
        f"{PRIMARY_MODEL_PATH}"
    )
    print(
        f"[OUTPUT]                     "
        f"{OUT_DIR}"
    )
    print(
        f"[SLIDES]                     "
        f"{len(tile_counts)}"
    )
    print(
        f"[PATIENTS]                   "
        f"{n_patients}"
    )
    print(
        f"[LOPO FOLDS]                 "
        f"{n_patients}"
    )
    print(
        f"[EMBEDDING DIMENSION]        "
        f"{embedding_dim}"
    )
    print(
        f"[COMPONENTS]                 "
        f"{N_COMPONENTS}"
    )
    print(
        f"[MAX TILES/SLIDE FOR FIT]    "
        f"{MAX_TILES_PER_SLIDE_FOR_FIT}"
    )

    print(
        "\n[INFO] Building fixed score-comparison sample..."
    )

    x_eval, eval_sampling = (
        build_common_evaluation_sample(
            tile_counts=tile_counts,
            random_state=(
                EVAL_SAMPLE_RANDOM_STATE
            ),
        )
    )

    eval_sampling.to_csv(
        OUT_SCORE_SAMPLE,
        index=False,
    )

    primary_scores = (
        transform_with_npz_model(
            x=x_eval,
            scaler_mean=primary[
                "scaler_mean"
            ],
            scaler_scale=primary[
                "scaler_scale"
            ],
            pca_mean=primary[
                "pca_mean"
            ],
            pca_components=primary[
                "pca_components"
            ][
                :N_COMPONENTS
            ],
        )
    )

    primary_effective = effective_loadings(
        pca_components=primary[
            "pca_components"
        ][
            :N_COMPONENTS
        ],
        scaler_scale=primary[
            "scaler_scale"
        ],
    )

    print(
        "\n[INFO] Loading discovery embeddings..."
    )

    embedding_cache = build_embedding_cache(
        tile_counts
    )

    all_matches = []
    all_sampling = []

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

        print("\n" + "-" * 100)
        print(
            f"[FOLD {fold_index:02d}/{n_patients}] "
            f"Hold out {holdout_patient}"
        )
        print("-" * 100)

        x_fit, sampling_df = (
            build_lopo_fit_matrix(
                tile_counts=tile_counts,
                embedding_cache=(
                    embedding_cache
                ),
                holdout_patient=(
                    holdout_patient
                ),
                random_state=(
                    fold_seed
                ),
            )
        )

        sampling_df.to_csv(
            fold_dir
            / "fit_sampling_manifest.csv",
            index=False,
        )

        all_sampling.append(
            sampling_df
        )

        training_patients = int(
            sampling_df[
                "patient_id"
            ].nunique()
        )

        print(
            f"[FIT MATRIX] {x_fit.shape} | "
            f"training_patients={training_patients}"
        )

        scaler, pca = fit_lopo_pca(
            x_fit=x_fit,
            random_state=(
                fold_seed
            ),
        )

        save_lopo_model(
            path=(
                fold_dir
                / "lopo_pca_model.npz"
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
            n_training_patients=(
                training_patients
            ),
        )

        explained_df = pd.DataFrame({
            "lopo_component": [
                f"PC{i + 1}"
                for i in range(
                    N_COMPONENTS
                )
            ],
            "component_index": (
                np.arange(
                    N_COMPONENTS
                )
            ),
            "explained_variance": (
                pca.explained_variance_
            ),
            "explained_variance_ratio": (
                pca.explained_variance_ratio_
            ),
            "cumulative_explained_variance_ratio": (
                np.cumsum(
                    pca.explained_variance_ratio_
                )
            ),
        })

        explained_df.to_csv(
            fold_dir
            / "lopo_explained_variance.csv",
            index=False,
        )

        lopo_effective = (
            effective_loadings(
                pca_components=(
                    pca.components_
                ),
                scaler_scale=(
                    scaler.scale_
                ),
            )
        )

        matches = match_components(
            primary_effective=(
                primary_effective
            ),
            lopo_effective=(
                lopo_effective
            ),
        )

        lopo_scores = pca.transform(
            scaler.transform(
                x_eval
            )
        )

        score_rows = []

        for _, match in matches.iterrows():
            primary_idx = int(
                match[
                    "primary_component_index"
                ]
            )

            lopo_idx = int(
                match[
                    "lopo_component_index"
                ]
            )

            sign_multiplier = int(
                match[
                    "sign_multiplier_for_alignment"
                ]
            )

            aligned_lopo_score = (
                lopo_scores[
                    :,
                    lopo_idx,
                ]
                * sign_multiplier
            )

            rho, p_value = safe_spearman(
                primary_scores[
                    :,
                    primary_idx,
                ],
                aligned_lopo_score,
            )

            score_rows.append({
                "primary_component_index": (
                    primary_idx
                ),
                "tile_score_spearman_rho": (
                    rho
                ),
                "tile_score_spearman_p_value": (
                    p_value
                ),
                "n_tiles_for_score_comparison": (
                    int(
                        x_eval.shape[0]
                    )
                ),
            })

        score_df = pd.DataFrame(
            score_rows
        )

        matches = matches.merge(
            score_df,
            on="primary_component_index",
            how="left",
            validate="one_to_one",
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
            "fold_random_state"
        ] = int(
            fold_seed
        )

        matches[
            "n_training_patients"
        ] = int(
            training_patients
        )

        matches[
            "n_fit_tiles"
        ] = int(
            x_fit.shape[0]
        )

        matches[
            "primary_explained_variance_ratio"
        ] = matches[
            "primary_component_index"
        ].map(
            lambda idx: float(
                primary[
                    "pca_explained_variance_ratio"
                ][
                    int(idx)
                ]
            )
        )

        matches[
            "lopo_explained_variance_ratio"
        ] = matches[
            "lopo_component_index"
        ].map(
            lambda idx: float(
                pca.explained_variance_ratio_[
                    int(idx)
                ]
            )
        )

        ordered_cols = [
            "holdout_patient",
            "fold_index",
            "fold_random_state",
            "n_training_patients",
            "n_fit_tiles",
            "primary_component",
            "primary_component_index",
            "lopo_component",
            "lopo_component_index",
            "sign_multiplier_for_alignment",
            "loading_cosine_before_sign_alignment",
            "absolute_loading_cosine",
            "loading_cosine_after_sign_alignment",
            "tile_score_spearman_rho",
            "tile_score_spearman_p_value",
            "primary_explained_variance_ratio",
            "lopo_explained_variance_ratio",
            "n_tiles_for_score_comparison",
        ]

        matches = matches[
            ordered_cols
        ]

        matches.to_csv(
            fold_dir
            / "component_matches.csv",
            index=False,
        )

        all_matches.append(
            matches
        )

        print(
            matches[
                [
                    "primary_component",
                    "lopo_component",
                    "absolute_loading_cosine",
                    "tile_score_spearman_rho",
                ]
            ].to_string(
                index=False
            )
        )

        del x_fit

    all_matches_df = pd.concat(
        all_matches,
        ignore_index=True,
    )

    all_matches_df[
        "primary_component_number"
    ] = (
        all_matches_df[
            "primary_component"
        ]
        .str.replace(
            "PC",
            "",
            regex=False,
        )
        .astype(int)
    )

    all_matches_df = (
        all_matches_df
        .sort_values(
            [
                "primary_component_number",
                "fold_index",
            ]
        )
        .drop(
            columns=[
                "primary_component_number"
            ]
        )
        .reset_index(drop=True)
    )

    all_matches_df.to_csv(
        OUT_ALL_MATCHES,
        index=False,
    )

    all_sampling_df = pd.concat(
        all_sampling,
        ignore_index=True,
    )

    all_sampling_df.to_csv(
        OUT_FIT_SAMPLING,
        index=False,
    )

    stability_summary = (
        make_component_stability_summary(
            all_matches_df
        )
    )

    stability_summary.to_csv(
        OUT_STABILITY_SUMMARY,
        index=False,
    )

    match_frequency = (
        make_match_frequency_table(
            all_matches_df
        )
    )

    match_frequency.to_csv(
        OUT_MATCH_FREQUENCY,
        index=False,
    )

    patient_influence = (
        make_patient_influence_summary(
            all_matches_df
        )
    )

    patient_influence.to_csv(
        OUT_PATIENT_INFLUENCE,
        index=False,
    )

    metadata = {
        "analysis_name": (
            "Leave-one-patient-out PCA stability"
        ),
        "analysis_role": (
            "Discovery patient-influence and latent-space structural stability"
        ),
        "manifest_path": str(
            MANIFEST_PATH
        ),
        "primary_model_path": str(
            PRIMARY_MODEL_PATH
        ),
        "output_directory": str(
            OUT_DIR
        ),
        "n_slides": int(
            len(tile_counts)
        ),
        "n_patients": int(
            n_patients
        ),
        "n_lopo_folds": int(
            n_patients
        ),
        "embedding_dimension": int(
            embedding_dim
        ),
        "n_components": int(
            N_COMPONENTS
        ),
        "max_tiles_per_slide_for_fit": int(
            MAX_TILES_PER_SLIDE_FOR_FIT
        ),
        "random_state": int(
            RANDOM_STATE
        ),
        "score_comparison_max_tiles_per_slide": int(
            MAX_EVAL_TILES_PER_SLIDE
        ),
        "score_comparison_random_state": int(
            EVAL_SAMPLE_RANDOM_STATE
        ),
        "score_comparison_total_tiles": int(
            x_eval.shape[0]
        ),
        "component_matching_method": (
            "One-to-one Hungarian assignment maximizing absolute cosine "
            "similarity between effective loading vectors."
        ),
        "effective_loading_definition": (
            "PCA component divided elementwise by scaler scale, followed "
            "by L2 normalization."
        ),
        "sign_alignment_rule": (
            "Matched LOPO components are multiplied by +1 or -1 so their "
            "loading cosine with the primary component is non-negative."
        ),
        "tile_score_similarity": (
            "Spearman correlation between primary frozen PCA scores and "
            "sign-aligned LOPO PCA scores on one fixed common tile sample."
        ),
        "tumor_labels_used": False,
        "held_out_patient_evaluation_used": False,
        "nested_validation": False,
        "hard_stability_threshold": None,
        "interpretation_rule": (
            "Report patient influence and structural stability metrics "
            "without post hoc threshold-based classification."
        ),
        "outputs": {
            "all_component_matches": str(
                OUT_ALL_MATCHES
            ),
            "component_stability_summary": str(
                OUT_STABILITY_SUMMARY
            ),
            "component_match_frequency": str(
                OUT_MATCH_FREQUENCY
            ),
            "patient_influence_summary": str(
                OUT_PATIENT_INFLUENCE
            ),
            "score_comparison_sampling_manifest": str(
                OUT_SCORE_SAMPLE
            ),
            "fit_sampling_manifest": str(
                OUT_FIT_SAMPLING
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

    print("\n" + "=" * 100)
    print(
        "LEAVE-ONE-PATIENT-OUT PCA "
        "STABILITY SUMMARY"
    )
    print("=" * 100)

    print(
        stability_summary.to_string(
            index=False
        )
    )

    print(
        "\nPATIENT INFLUENCE SUMMARY"
    )
    print(
        patient_influence.to_string(
            index=False
        )
    )

    print("\n[OK] Saved:")
    print(f"  {OUT_ALL_MATCHES}")
    print(f"  {OUT_STABILITY_SUMMARY}")
    print(f"  {OUT_MATCH_FREQUENCY}")
    print(f"  {OUT_PATIENT_INFLUENCE}")
    print(f"  {OUT_SCORE_SAMPLE}")
    print(f"  {OUT_FIT_SAMPLING}")
    print(f"  {OUT_METADATA}")

    print(
        "\n[DONE] Leave-one-patient-out "
        "PCA stability analysis complete."
    )


if __name__ == "__main__":
    main()
