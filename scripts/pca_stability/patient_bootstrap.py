#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_patient_bootstrap_pca_stability.py

Purpose
-------
Evaluate the stability of the frozen discovery PCA basis under patient-level
bootstrap resampling.

This script extends the earlier PC3-only bootstrap analysis to all PC1-PC8.

For each bootstrap iteration:
1. Resample discovery patients with replacement.
2. Include all slides belonging to each sampled patient occurrence.
3. Randomly cap each slide at MAX_TILES_PER_SLIDE_FOR_FIT tiles.
4. Fit a new StandardScaler + PCA.
5. Compare bootstrap components with the original frozen PCA components using
   effective loading vectors in the original embedding feature space.
6. Perform one-to-one Hungarian matching that maximizes total absolute cosine
   similarity.
7. Align component signs after matching.
8. Record loading cosine, matched rank, explained variance, and score correlation.

Cohort definition
-----------------
The bootstrap cohort is defined exclusively by:
    ../wsi_embeddings/embedding_manifest.csv

Outputs
-------
MedIA_paper/results/08_patient_bootstrap_pca_stability/
    analysis_metadata.json
    bootstrap_sampling_manifest.csv
    bootstrap_component_matches.csv
    component_stability_summary.csv
    component_match_frequency.csv
    score_comparison_sampling_manifest.csv

    bootstrap_0001/
        bootstrap_pca_model.npz
        bootstrap_explained_variance.csv
        component_matches.csv
        patient_sampling.csv
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

OUT_DIR = RESULTS_ROOT / "08_patient_bootstrap_pca_stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_BOOTSTRAP_SAMPLING = OUT_DIR / "bootstrap_sampling_manifest.csv"
OUT_ALL_MATCHES = OUT_DIR / "bootstrap_component_matches.csv"
OUT_STABILITY_SUMMARY = OUT_DIR / "component_stability_summary.csv"
OUT_MATCH_FREQUENCY = OUT_DIR / "component_match_frequency.csv"
OUT_SCORE_SAMPLE = OUT_DIR / "score_comparison_sampling_manifest.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================
N_COMPONENTS = 8
N_BOOTSTRAP = 1000
MAX_TILES_PER_SLIDE_FOR_FIT = 2000

MAX_EVAL_TILES_PER_SLIDE = 1000
EVAL_SAMPLE_RANDOM_STATE = 314159

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
        return str((EMB_DIR / Path(*raw.parts[1:])).resolve())

    return str((MANIFEST_PATH.parent / raw).resolve())


def load_embeddings(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {x.shape}: {path}")

    if USE_FLOAT32:
        x = np.asarray(x, dtype=np.float32)
    else:
        x = np.asarray(x)

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite values found: {path}")

    return x


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "wsi_label",
        "prefix",
        "embeddings_path",
    }

    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    if manifest.empty:
        raise ValueError("Embedding manifest is empty.")

    out = manifest.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["wsi_label"] = out["wsi_label"].astype(str)
    out["prefix"] = out["prefix"].astype(str)
    out["embeddings_path"] = (
        out["embeddings_path"].astype(str).map(resolve_manifest_path)
    )

    if out["prefix"].duplicated().any():
        duplicated = out.loc[
            out["prefix"].duplicated(keep=False),
            "prefix",
        ].tolist()
        raise ValueError(
            f"Duplicate prefixes in discovery manifest: {duplicated[:20]}"
        )

    missing_paths = [
        path
        for path in out["embeddings_path"]
        if not Path(path).exists()
    ]
    if missing_paths:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} embedding files are missing after path "
            f"resolution.\n{preview}"
        )

    return out.sort_values(
        ["patient_id", "wsi_label", "prefix"]
    ).reset_index(drop=True)


def load_primary_model(path: Path) -> Dict[str, np.ndarray]:
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
            f"Primary model missing arrays: {sorted(missing)}"
        )

    output = {key: np.asarray(model[key]) for key in model.files}

    if output["pca_components"].shape[0] < N_COMPONENTS:
        raise ValueError(
            f"Primary PCA has fewer than {N_COMPONENTS} components."
        )

    if np.any(output["scaler_scale"] <= EPS):
        raise ValueError(
            "Primary scaler contains zero or near-zero scales."
        )

    return output


def inspect_slide_tile_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected_dim = None

    for _, row in manifest.iterrows():
        x = np.load(row["embeddings_path"], mmap_mode="r")

        if x.ndim != 2:
            raise ValueError(f"Invalid embedding shape: {x.shape}")

        n_tiles, embedding_dim = x.shape

        if expected_dim is None:
            expected_dim = int(embedding_dim)
        elif int(embedding_dim) != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimension: {row['prefix']}"
            )

        rows.append({
            "patient_id": row["patient_id"],
            "wsi_label": row["wsi_label"],
            "prefix": row["prefix"],
            "embeddings_path": row["embeddings_path"],
            "n_tiles_available": int(n_tiles),
            "embedding_dim": int(embedding_dim),
        })

    return pd.DataFrame(rows)


def effective_loadings(
    pca_components: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    if np.any(scaler_scale <= EPS):
        raise ValueError(
            "Scaler contains zero or near-zero feature scales."
        )

    loadings = pca_components / scaler_scale[np.newaxis, :]
    norms = np.linalg.norm(loadings, axis=1, keepdims=True)

    if np.any(norms <= EPS):
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
    x_scaled = (x - scaler_mean) / scaler_scale
    return (x_scaled - pca_mean) @ pca_components.T


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 3:
        return np.nan, np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if np.std(x_valid) <= EPS or np.std(y_valid) <= EPS:
        return np.nan, np.nan

    result = spearmanr(x_valid, y_valid)
    return float(result.statistic), float(result.pvalue)


# =============================================================================
# COMMON EVALUATION SAMPLE
# =============================================================================
def build_common_evaluation_sample(
    tile_counts: pd.DataFrame,
    random_state: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    blocks: List[np.ndarray] = []
    rows = []

    for _, row in tile_counts.iterrows():
        x = load_embeddings(row["embeddings_path"])

        n_available = int(x.shape[0])
        n_eval = min(MAX_EVAL_TILES_PER_SLIDE, n_available)

        indices = rng.choice(
            n_available,
            size=n_eval,
            replace=False,
        )
        indices.sort()

        blocks.append(x[indices])

        rows.append({
            "patient_id": row["patient_id"],
            "wsi_label": row["wsi_label"],
            "prefix": row["prefix"],
            "n_tiles_available": n_available,
            "n_tiles_used_for_score_comparison": n_eval,
        })

    x_eval = np.concatenate(blocks, axis=0)

    if not np.isfinite(x_eval).all():
        raise ValueError(
            "Non-finite values in evaluation sample."
        )

    return x_eval, pd.DataFrame(rows)


# =============================================================================
# BOOTSTRAP SAMPLING
# =============================================================================
def build_embedding_cache(
    tile_counts: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    cache = {}

    for _, row in tile_counts.iterrows():
        prefix = row["prefix"]
        x = load_embeddings(row["embeddings_path"])
        cache[prefix] = x
        print(f"[LOAD] {prefix} | shape={x.shape}")

    return cache


def sample_bootstrap_fit_matrix(
    tile_counts: pd.DataFrame,
    embedding_cache: Dict[str, np.ndarray],
    sampled_patients: np.ndarray,
    rng: np.random.Generator,
    bootstrap_index: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    blocks: List[np.ndarray] = []
    rows = []

    occurrence_counter = {}

    for draw_index, patient_id in enumerate(sampled_patients):
        patient_id = str(patient_id)

        occurrence_counter[patient_id] = (
            occurrence_counter.get(patient_id, 0) + 1
        )
        occurrence = occurrence_counter[patient_id]

        patient_slides = (
            tile_counts.loc[
                tile_counts["patient_id"] == patient_id
            ]
            .copy()
            .sort_values(["wsi_label", "prefix"])
        )

        for _, row in patient_slides.iterrows():
            prefix = row["prefix"]
            x = embedding_cache[prefix]

            n_available = int(x.shape[0])
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
                sampling_mode = "all_available_tiles"

            blocks.append(x_use)

            rows.append({
                "bootstrap": int(bootstrap_index),
                "draw_index": int(draw_index),
                "patient_id": patient_id,
                "patient_occurrence": int(occurrence),
                "wsi_label": row["wsi_label"],
                "prefix": prefix,
                "n_tiles_available": n_available,
                "n_tiles_used_for_fit": n_used,
                "sampling_mode": sampling_mode,
            })

    if not blocks:
        raise RuntimeError(
            "Bootstrap sample produced no embeddings."
        )

    x_fit = np.concatenate(blocks, axis=0)

    if not np.isfinite(x_fit).all():
        raise ValueError(
            "Non-finite values in bootstrap fit matrix."
        )

    return x_fit, pd.DataFrame(rows)


# =============================================================================
# PCA FITTING AND MATCHING
# =============================================================================
def fit_bootstrap_pca(
    x_fit: np.ndarray,
    random_state: int,
) -> Tuple[StandardScaler, PCA]:
    if x_fit.shape[0] <= N_COMPONENTS:
        raise ValueError(
            "Too few samples for PCA fitting."
        )

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_fit)

    pca = PCA(
        n_components=N_COMPONENTS,
        svd_solver="auto",
        random_state=random_state,
    )
    pca.fit(x_scaled)

    return scaler, pca


def match_components(
    primary_effective: np.ndarray,
    bootstrap_effective: np.ndarray,
) -> pd.DataFrame:
    signed_cosine = (
        primary_effective
        @ bootstrap_effective.T
    )
    absolute_cosine = np.abs(signed_cosine)

    primary_indices, bootstrap_indices = (
        linear_sum_assignment(
            -absolute_cosine
        )
    )

    rows = []

    for primary_idx, bootstrap_idx in zip(
        primary_indices,
        bootstrap_indices,
    ):
        cosine = float(
            signed_cosine[
                primary_idx,
                bootstrap_idx,
            ]
        )

        sign_multiplier = 1 if cosine >= 0 else -1

        rows.append({
            "primary_component_index": int(primary_idx),
            "primary_component": f"PC{primary_idx + 1}",
            "bootstrap_component_index": int(bootstrap_idx),
            "bootstrap_component": f"PC{bootstrap_idx + 1}",
            "loading_cosine_before_sign_alignment": cosine,
            "absolute_loading_cosine": abs(cosine),
            "sign_multiplier_for_alignment": int(sign_multiplier),
            "loading_cosine_after_sign_alignment": abs(cosine),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("primary_component_index")
        .reset_index(drop=True)
    )


def save_bootstrap_model(
    path: Path,
    scaler: StandardScaler,
    pca: PCA,
    bootstrap_index: int,
    random_state: int,
    n_fit_tiles: int,
    n_unique_patients: int,
) -> None:
    np.savez(
        path,
        scaler_mean=np.asarray(scaler.mean_),
        scaler_scale=np.asarray(scaler.scale_),
        pca_components=np.asarray(pca.components_),
        pca_mean=np.asarray(pca.mean_),
        pca_explained_variance=np.asarray(
            pca.explained_variance_
        ),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_
        ),
        pca_singular_values=np.asarray(
            pca.singular_values_
        ),
        bootstrap_index=np.array(
            [bootstrap_index],
            dtype=np.int32,
        ),
        random_state=np.array(
            [random_state],
            dtype=np.int64,
        ),
        n_fit_tiles=np.array(
            [n_fit_tiles],
            dtype=np.int64,
        ),
        n_unique_patients=np.array(
            [n_unique_patients],
            dtype=np.int32,
        ),
    )


# =============================================================================
# SUMMARY
# =============================================================================
def make_component_stability_summary(
    all_matches: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for component in [
        f"PC{i}"
        for i in range(1, N_COMPONENTS + 1)
    ]:
        group = all_matches.loc[
            all_matches["primary_component"] == component
        ].copy()

        matched_counts = (
            group["bootstrap_component"]
            .value_counts()
            .sort_values(ascending=False)
        )

        most_common = matched_counts.index[0]
        most_common_count = int(
            matched_counts.iloc[0]
        )

        rows.append({
            "Primary component": component,
            "N bootstrap resamples": int(
                group["bootstrap"].nunique()
            ),
            "Most frequent matched bootstrap component": (
                most_common
            ),
            "Matched component consistency": float(
                most_common_count
                / group["bootstrap"].nunique()
            ),
            "Mean absolute loading cosine": float(
                group["absolute_loading_cosine"].mean()
            ),
            "Median absolute loading cosine": float(
                group["absolute_loading_cosine"].median()
            ),
            "Q25 absolute loading cosine": float(
                group["absolute_loading_cosine"].quantile(0.25)
            ),
            "Q75 absolute loading cosine": float(
                group["absolute_loading_cosine"].quantile(0.75)
            ),
            "Minimum absolute loading cosine": float(
                group["absolute_loading_cosine"].min()
            ),
            "Mean tile-score Spearman rho": float(
                group["tile_score_spearman_rho"].mean()
            ),
            "Median tile-score Spearman rho": float(
                group["tile_score_spearman_rho"].median()
            ),
            "Q25 tile-score Spearman rho": float(
                group["tile_score_spearman_rho"].quantile(0.25)
            ),
            "Q75 tile-score Spearman rho": float(
                group["tile_score_spearman_rho"].quantile(0.75)
            ),
            "Minimum tile-score Spearman rho": float(
                group["tile_score_spearman_rho"].min()
            ),
            "Mean matched bootstrap explained variance ratio": float(
                group[
                    "bootstrap_explained_variance_ratio"
                ].mean()
            ),
            "Median matched bootstrap explained variance ratio": float(
                group[
                    "bootstrap_explained_variance_ratio"
                ].median()
            ),
            "Primary explained variance ratio": float(
                group[
                    "primary_explained_variance_ratio"
                ].iloc[0]
            ),
            "Cosine >= 0.70 fraction": float(
                (
                    group["absolute_loading_cosine"]
                    >= 0.70
                ).mean()
            ),
            "Cosine >= 0.80 fraction": float(
                (
                    group["absolute_loading_cosine"]
                    >= 0.80
                ).mean()
            ),
            "Cosine >= 0.90 fraction": float(
                (
                    group["absolute_loading_cosine"]
                    >= 0.90
                ).mean()
            ),
            "Score rho >= 0.80 fraction": float(
                (
                    group["tile_score_spearman_rho"]
                    >= 0.80
                ).mean()
            ),
            "Score rho >= 0.90 fraction": float(
                (
                    group["tile_score_spearman_rho"]
                    >= 0.90
                ).mean()
            ),
        })

    summary = pd.DataFrame(rows)

    numeric_cols = (
        summary
        .select_dtypes(include=[np.number])
        .columns
    )

    summary[numeric_cols] = (
        summary[numeric_cols].round(6)
    )

    return summary


def make_match_frequency_table(
    all_matches: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    total_bootstraps = int(
        all_matches["bootstrap"].nunique()
    )

    for primary_component in [
        f"PC{i}"
        for i in range(1, N_COMPONENTS + 1)
    ]:
        group = all_matches.loc[
            all_matches["primary_component"]
            == primary_component
        ]

        for bootstrap_component in [
            f"PC{i}"
            for i in range(1, N_COMPONENTS + 1)
        ]:
            count = int(
                (
                    group["bootstrap_component"]
                    == bootstrap_component
                ).sum()
            )

            rows.append({
                "primary_component": primary_component,
                "bootstrap_component": bootstrap_component,
                "match_count": count,
                "match_fraction": float(
                    count / total_bootstraps
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    check_file(MANIFEST_PATH)
    check_file(PRIMARY_MODEL_PATH)

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = validate_manifest(manifest)

    primary = load_primary_model(
        PRIMARY_MODEL_PATH
    )

    tile_counts = inspect_slide_tile_counts(
        manifest
    )

    embedding_dim = int(
        tile_counts["embedding_dim"].iloc[0]
    )

    if (
        primary["pca_components"].shape[1]
        != embedding_dim
    ):
        raise ValueError(
            "Primary PCA embedding dimension does not "
            "match manifest embeddings."
        )

    print("=" * 100)
    print("PATIENT-LEVEL BOOTSTRAP PCA STABILITY")
    print("=" * 100)
    print(f"[MANIFEST]                   {MANIFEST_PATH}")
    print(f"[PRIMARY MODEL]              {PRIMARY_MODEL_PATH}")
    print(f"[OUTPUT]                     {OUT_DIR}")
    print(f"[SLIDES]                     {len(tile_counts)}")
    print(
        f"[PATIENTS]                   "
        f"{tile_counts['patient_id'].nunique()}"
    )
    print(f"[EMBEDDING DIMENSION]        {embedding_dim}")
    print(f"[COMPONENTS]                 {N_COMPONENTS}")
    print(f"[BOOTSTRAPS]                 {N_BOOTSTRAP}")
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
            random_state=EVAL_SAMPLE_RANDOM_STATE,
        )
    )

    eval_sampling.to_csv(
        OUT_SCORE_SAMPLE,
        index=False,
    )

    primary_scores = transform_with_npz_model(
        x=x_eval,
        scaler_mean=primary["scaler_mean"],
        scaler_scale=primary["scaler_scale"],
        pca_mean=primary["pca_mean"],
        pca_components=primary[
            "pca_components"
        ][:N_COMPONENTS],
    )

    primary_effective = effective_loadings(
        pca_components=primary[
            "pca_components"
        ][:N_COMPONENTS],
        scaler_scale=primary["scaler_scale"],
    )

    print(
        "\n[INFO] Loading discovery embeddings..."
    )

    embedding_cache = build_embedding_cache(
        tile_counts
    )

    patients = np.array(
        sorted(
            tile_counts["patient_id"].unique()
        )
    )
    n_patients = int(len(patients))

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    all_matches = []
    all_sampling_rows = []

    for bootstrap_index in range(
        1,
        N_BOOTSTRAP + 1,
    ):
        sampled_patients = rng.choice(
            patients,
            size=n_patients,
            replace=True,
        )

        bootstrap_seed = (
            RANDOM_STATE + bootstrap_index
        )

        bootstrap_dir = (
            OUT_DIR
            / f"bootstrap_{bootstrap_index:04d}"
        )
        bootstrap_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        x_fit, sampling_df = (
            sample_bootstrap_fit_matrix(
                tile_counts=tile_counts,
                embedding_cache=embedding_cache,
                sampled_patients=sampled_patients,
                rng=rng,
                bootstrap_index=bootstrap_index,
            )
        )

        unique_patients = int(
            len(set(sampled_patients.tolist()))
        )

        patient_sampling = pd.DataFrame({
            "draw_index": np.arange(
                n_patients
            ),
            "sampled_patient": sampled_patients,
        })
        patient_sampling["bootstrap"] = int(
            bootstrap_index
        )

        patient_sampling.to_csv(
            bootstrap_dir
            / "patient_sampling.csv",
            index=False,
        )

        sampling_df.to_csv(
            bootstrap_dir
            / "fit_sampling_manifest.csv",
            index=False,
        )

        all_sampling_rows.append(
            sampling_df
        )

        scaler, pca = fit_bootstrap_pca(
            x_fit=x_fit,
            random_state=bootstrap_seed,
        )

        save_bootstrap_model(
            path=(
                bootstrap_dir
                / "bootstrap_pca_model.npz"
            ),
            scaler=scaler,
            pca=pca,
            bootstrap_index=bootstrap_index,
            random_state=bootstrap_seed,
            n_fit_tiles=int(x_fit.shape[0]),
            n_unique_patients=unique_patients,
        )

        explained_df = pd.DataFrame({
            "bootstrap_component": [
                f"PC{i + 1}"
                for i in range(N_COMPONENTS)
            ],
            "component_index": np.arange(
                N_COMPONENTS
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
            bootstrap_dir
            / "bootstrap_explained_variance.csv",
            index=False,
        )

        bootstrap_effective = (
            effective_loadings(
                pca_components=pca.components_,
                scaler_scale=scaler.scale_,
            )
        )

        matches = match_components(
            primary_effective=primary_effective,
            bootstrap_effective=bootstrap_effective,
        )

        bootstrap_scores = pca.transform(
            scaler.transform(x_eval)
        )

        score_rows = []

        for _, match in matches.iterrows():
            primary_idx = int(
                match[
                    "primary_component_index"
                ]
            )
            bootstrap_idx = int(
                match[
                    "bootstrap_component_index"
                ]
            )
            sign_multiplier = int(
                match[
                    "sign_multiplier_for_alignment"
                ]
            )

            aligned_bootstrap_score = (
                bootstrap_scores[:, bootstrap_idx]
                * sign_multiplier
            )

            rho, p_value = safe_spearman(
                primary_scores[:, primary_idx],
                aligned_bootstrap_score,
            )

            score_rows.append({
                "primary_component_index": (
                    primary_idx
                ),
                "tile_score_spearman_rho": rho,
                "tile_score_spearman_p_value": (
                    p_value
                ),
                "n_tiles_for_score_comparison": int(
                    x_eval.shape[0]
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

        matches["bootstrap"] = int(
            bootstrap_index
        )
        matches[
            "bootstrap_random_state"
        ] = int(
            bootstrap_seed
        )
        matches["n_unique_patients"] = int(
            unique_patients
        )
        matches["n_fit_tiles"] = int(
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
                ][int(idx)]
            )
        )

        matches[
            "bootstrap_explained_variance_ratio"
        ] = matches[
            "bootstrap_component_index"
        ].map(
            lambda idx: float(
                pca.explained_variance_ratio_[
                    int(idx)
                ]
            )
        )

        ordered_cols = [
            "bootstrap",
            "bootstrap_random_state",
            "n_unique_patients",
            "n_fit_tiles",
            "primary_component",
            "primary_component_index",
            "bootstrap_component",
            "bootstrap_component_index",
            "sign_multiplier_for_alignment",
            "loading_cosine_before_sign_alignment",
            "absolute_loading_cosine",
            "loading_cosine_after_sign_alignment",
            "tile_score_spearman_rho",
            "tile_score_spearman_p_value",
            "primary_explained_variance_ratio",
            "bootstrap_explained_variance_ratio",
            "n_tiles_for_score_comparison",
        ]

        matches = matches[
            ordered_cols
        ]

        matches.to_csv(
            bootstrap_dir
            / "component_matches.csv",
            index=False,
        )

        all_matches.append(matches)

        if (
            bootstrap_index <= 10
            or bootstrap_index % 25 == 0
            or bootstrap_index == N_BOOTSTRAP
        ):
            print(
                f"[BOOT {bootstrap_index:04d}/"
                f"{N_BOOTSTRAP}] "
                f"unique_patients={unique_patients} | "
                f"fit_tiles={x_fit.shape[0]}"
            )

            print(
                matches[
                    [
                        "primary_component",
                        "bootstrap_component",
                        "absolute_loading_cosine",
                        "tile_score_spearman_rho",
                    ]
                ].to_string(index=False)
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
                "bootstrap",
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
        all_sampling_rows,
        ignore_index=True,
    )

    all_sampling_df.to_csv(
        OUT_BOOTSTRAP_SAMPLING,
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

    metadata = {
        "analysis_name": (
            "Patient-level bootstrap PCA stability"
        ),
        "analysis_role": (
            "Discovery latent-space structural stability"
        ),
        "manifest_path": str(MANIFEST_PATH),
        "primary_model_path": str(
            PRIMARY_MODEL_PATH
        ),
        "output_directory": str(OUT_DIR),
        "n_slides": int(len(tile_counts)),
        "n_patients": int(
            tile_counts["patient_id"].nunique()
        ),
        "embedding_dimension": int(
            embedding_dim
        ),
        "n_components": int(
            N_COMPONENTS
        ),
        "n_bootstrap": int(
            N_BOOTSTRAP
        ),
        "patient_resampling": (
            "Patients sampled with replacement; sample "
            "size equals number of discovery patients."
        ),
        "slide_handling_within_sampled_patient": (
            "All discovery slides from each sampled patient "
            "occurrence are included."
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
            "One-to-one Hungarian assignment maximizing absolute "
            "cosine similarity between effective loading vectors."
        ),
        "effective_loading_definition": (
            "PCA component divided elementwise by scaler scale, "
            "followed by L2 normalization."
        ),
        "sign_alignment_rule": (
            "Matched bootstrap components are multiplied by +1 or "
            "-1 so their loading cosine with the primary component "
            "is non-negative."
        ),
        "tile_score_similarity": (
            "Spearman correlation between primary frozen PCA scores "
            "and sign-aligned bootstrap PCA scores on one fixed "
            "common tile sample."
        ),
        "hard_stability_threshold": None,
        "interpretation_rule": (
            "Report stability metrics without post hoc "
            "threshold-based classification."
        ),
        "outputs": {
            "bootstrap_sampling_manifest": str(
                OUT_BOOTSTRAP_SAMPLING
            ),
            "all_component_matches": str(
                OUT_ALL_MATCHES
            ),
            "component_stability_summary": str(
                OUT_STABILITY_SUMMARY
            ),
            "component_match_frequency": str(
                OUT_MATCH_FREQUENCY
            ),
            "score_comparison_sampling_manifest": str(
                OUT_SCORE_SAMPLE
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
        "PATIENT-LEVEL BOOTSTRAP PCA STABILITY SUMMARY"
    )
    print("=" * 100)

    print(
        stability_summary.to_string(
            index=False
        )
    )

    print("\n[OK] Saved:")
    print(f"  {OUT_ALL_MATCHES}")
    print(f"  {OUT_STABILITY_SUMMARY}")
    print(f"  {OUT_MATCH_FREQUENCY}")
    print(f"  {OUT_BOOTSTRAP_SAMPLING}")
    print(f"  {OUT_SCORE_SAMPLE}")
    print(f"  {OUT_METADATA}")

    print(
        "\n[DONE] Patient-level bootstrap PCA "
        "stability analysis complete."
    )


if __name__ == "__main__":
    main()
