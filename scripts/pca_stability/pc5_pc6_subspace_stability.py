#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
12_pc5_pc6_principal_angle_subspace_stability.py

Purpose
-------
Quantify the stability of the two-dimensional PC5-PC6 latent subspace using
principal angles.

Motivation
----------
Previous bootstrap and LOPO analyses showed frequent PC5 <-> PC6 component
ordering swaps despite strong matched loading similarity and near-complete
containment within the PC5-PC6 pair.

Individual PCA axes can rotate or exchange order when adjacent eigenvalues are
close. Therefore, exact component-number consistency alone may underestimate
stability of the underlying two-dimensional latent structure.

This script directly compares:

    Reference subspace:
        span(reference PC5, reference PC6)

with:

    Resampled subspace:
        span(resampled PC5, resampled PC6)

for:
    - all saved patient-bootstrap PCA refits from step 08
    - all saved leave-one-patient-out PCA refits from step 09

For two 2D subspaces, two principal angles are returned:

    theta_1 <= theta_2

Small angles indicate strong subspace similarity.

Method
------
PCA components are first converted from standardized feature space back to
effective loading vectors in the original embedding feature space:

    effective_loading = pca_component / scaler_scale

The two effective loading vectors are orthonormalized with QR decomposition.
Principal angles are then obtained from the singular values of:

    Q_reference.T @ Q_resampled

where Q_reference and Q_resampled are orthonormal bases.

If singular values are sigma_1 >= sigma_2, then:

    theta_i = arccos(sigma_i)

Angles are reported in degrees.

Important
---------
This analysis evaluates stability of the PC5-PC6 subspace only.

It does NOT:
    - change primary-axis qualification
    - relabel PC5 or PC6
    - use tumor annotations
    - use morphology
    - use external cohorts
    - refit any PCA model

Inputs
------
../wsi_embeddings/pca_global/global_pca_model.npz

results/08_patient_bootstrap_pca_stability/
    bootstrap_0001/bootstrap_pca_model.npz
    ...
    bootstrap_1000/bootstrap_pca_model.npz

results/09_leave_one_patient_out_pca_stability/
    holdout_*/lopo_pca_model.npz

Outputs
-------
results/12_pc5_pc6_principal_angle_subspace_stability/
    bootstrap_pc5_pc6_principal_angles.csv
    lopo_pc5_pc6_principal_angles.csv
    pc5_pc6_subspace_stability_summary.csv
    analysis_metadata.json
    README.txt
"""

import hashlib
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

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_MODEL = FROZEN_PCA_MODEL

BOOTSTRAP_ROOT = (
    RESULTS_ROOT
    / "08_patient_bootstrap_pca_stability"
)

LOPO_ROOT = (
    RESULTS_ROOT
    / "09_leave_one_patient_out_pca_stability"
)

OUT_DIR = (
    RESULTS_ROOT
    / "12_pc5_pc6_principal_angle_subspace_stability"
)
OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_BOOTSTRAP = (
    OUT_DIR
    / "bootstrap_pc5_pc6_principal_angles.csv"
)

OUT_LOPO = (
    OUT_DIR
    / "lopo_pc5_pc6_principal_angles.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "pc5_pc6_subspace_stability_summary.csv"
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
PC5_INDEX = 4
PC6_INDEX = 5

EXPECTED_BOOTSTRAPS = 1000
EXPECTED_LOPO_FOLDS = 15

EPS = 1e-12
ROUND_DIGITS = 6


# =============================================================================
# HELPERS
# =============================================================================
def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def check_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required directory: {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_pca_model(
    path: Path,
) -> Dict[str, np.ndarray]:
    check_file(path)

    model = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "scaler_scale",
        "pca_components",
    }

    missing = required - set(
        model.files
    )

    if missing:
        raise ValueError(
            f"{path} missing arrays: "
            f"{sorted(missing)}"
        )

    scaler_scale = np.asarray(
        model["scaler_scale"],
        dtype=np.float64,
    )

    pca_components = np.asarray(
        model["pca_components"],
        dtype=np.float64,
    )

    if pca_components.ndim != 2:
        raise ValueError(
            f"Invalid PCA component shape in {path}: "
            f"{pca_components.shape}"
        )

    if (
        pca_components.shape[0]
        <= PC6_INDEX
    ):
        raise ValueError(
            f"Model has fewer than 6 PCs: {path}"
        )

    if (
        pca_components.shape[1]
        != scaler_scale.shape[0]
    ):
        raise ValueError(
            f"Component/scaler dimension mismatch: {path}"
        )

    if np.any(
        scaler_scale <= EPS
    ):
        raise ValueError(
            f"Zero or near-zero scaler scale: {path}"
        )

    return {
        "scaler_scale": scaler_scale,
        "pca_components": pca_components,
    }


def effective_pc56_matrix(
    model: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Return embedding_dim x 2 matrix whose columns are effective PC5 and PC6
    loading vectors in the original embedding feature space.
    """
    scale = model[
        "scaler_scale"
    ]

    components = model[
        "pca_components"
    ]

    pc5 = (
        components[PC5_INDEX]
        / scale
    )

    pc6 = (
        components[PC6_INDEX]
        / scale
    )

    matrix = np.column_stack(
        [
            pc5,
            pc6,
        ]
    )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            "Non-finite values in PC5-PC6 "
            "effective loading matrix."
        )

    return matrix


def orthonormal_basis(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Orthonormalize the two-dimensional subspace with reduced QR.
    """
    q, r = np.linalg.qr(
        matrix,
        mode="reduced",
    )

    diagonal = np.abs(
        np.diag(r)
    )

    if (
        len(diagonal) < 2
        or np.any(
            diagonal <= EPS
        )
    ):
        raise ValueError(
            "PC5-PC6 effective loading matrix "
            "is rank deficient."
        )

    return q


def principal_angles_degrees(
    q_reference: np.ndarray,
    q_test: np.ndarray,
) -> Tuple[
    float,
    float,
    float,
    float,
]:
    """
    Compute the two principal angles between 2D subspaces.

    Returns
    -------
    theta1_deg
    theta2_deg
    cos_theta1
    cos_theta2
    """
    cross = (
        q_reference.T
        @ q_test
    )

    singular_values = np.linalg.svd(
        cross,
        compute_uv=False,
    )

    singular_values = np.clip(
        singular_values,
        -1.0,
        1.0,
    )

    singular_values = np.sort(
        singular_values
    )[::-1]

    if len(
        singular_values
    ) != 2:
        raise ValueError(
            "Expected exactly two singular values."
        )

    angles_rad = np.arccos(
        singular_values
    )

    angles_deg = np.degrees(
        angles_rad
    )

    return (
        float(
            angles_deg[0]
        ),
        float(
            angles_deg[1]
        ),
        float(
            singular_values[0]
        ),
        float(
            singular_values[1]
        ),
    )


def subspace_distance_metrics(
    theta1_deg: float,
    theta2_deg: float,
) -> Dict[str, float]:
    """
    Additional commonly useful summaries.

    spectral_sin_distance:
        sin(max principal angle)

    chordal_distance:
        sqrt(sin(theta1)^2 + sin(theta2)^2)
    """
    theta1 = np.radians(
        theta1_deg
    )

    theta2 = np.radians(
        theta2_deg
    )

    sin1 = np.sin(
        theta1
    )

    sin2 = np.sin(
        theta2
    )

    return {
        "maximum_principal_angle_deg": float(
            max(
                theta1_deg,
                theta2_deg,
            )
        ),
        "mean_principal_angle_deg": float(
            (
                theta1_deg
                + theta2_deg
            )
            / 2.0
        ),
        "spectral_sin_distance": float(
            max(
                sin1,
                sin2,
            )
        ),
        "chordal_distance": float(
            np.sqrt(
                sin1**2
                + sin2**2
            )
        ),
    }


# =============================================================================
# BOOTSTRAP ANALYSIS
# =============================================================================
def analyze_bootstraps(
    q_reference: np.ndarray,
) -> pd.DataFrame:
    bootstrap_dirs = sorted(
        path
        for path in BOOTSTRAP_ROOT.glob(
            "bootstrap_*"
        )
        if path.is_dir()
    )

    if len(
        bootstrap_dirs
    ) == 0:
        raise FileNotFoundError(
            "No bootstrap directories found in "
            f"{BOOTSTRAP_ROOT}"
        )

    rows = []

    for directory in bootstrap_dirs:
        model_path = (
            directory
            / "bootstrap_pca_model.npz"
        )

        if not model_path.exists():
            print(
                f"[SKIP] Missing model: "
                f"{model_path}"
            )
            continue

        model = load_pca_model(
            model_path
        )

        q_test = orthonormal_basis(
            effective_pc56_matrix(
                model
            )
        )

        (
            theta1,
            theta2,
            cos1,
            cos2,
        ) = principal_angles_degrees(
            q_reference,
            q_test,
        )

        distances = (
            subspace_distance_metrics(
                theta1,
                theta2,
            )
        )

        bootstrap_number = int(
            directory.name
            .replace(
                "bootstrap_",
                "",
            )
        )

        rows.append({
            "bootstrap": (
                bootstrap_number
            ),
            "model_path": str(
                model_path
            ),
            "principal_angle_1_deg": (
                theta1
            ),
            "principal_angle_2_deg": (
                theta2
            ),
            "cos_principal_angle_1": (
                cos1
            ),
            "cos_principal_angle_2": (
                cos2
            ),
            **distances,
        })

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        raise RuntimeError(
            "No bootstrap models were analyzed."
        )

    return (
        output
        .sort_values(
            "bootstrap"
        )
        .reset_index(drop=True)
    )


# =============================================================================
# LOPO ANALYSIS
# =============================================================================
def analyze_lopo(
    q_reference: np.ndarray,
) -> pd.DataFrame:
    fold_dirs = sorted(
        path
        for path in LOPO_ROOT.glob(
            "holdout_*"
        )
        if path.is_dir()
    )

    if len(
        fold_dirs
    ) == 0:
        raise FileNotFoundError(
            "No LOPO holdout directories found in "
            f"{LOPO_ROOT}"
        )

    rows = []

    for directory in fold_dirs:
        model_path = (
            directory
            / "lopo_pca_model.npz"
        )

        if not model_path.exists():
            print(
                f"[SKIP] Missing model: "
                f"{model_path}"
            )
            continue

        model = load_pca_model(
            model_path
        )

        q_test = orthonormal_basis(
            effective_pc56_matrix(
                model
            )
        )

        (
            theta1,
            theta2,
            cos1,
            cos2,
        ) = principal_angles_degrees(
            q_reference,
            q_test,
        )

        distances = (
            subspace_distance_metrics(
                theta1,
                theta2,
            )
        )

        holdout_patient = (
            directory.name
            .replace(
                "holdout_",
                "",
            )
        )

        rows.append({
            "holdout_patient": (
                holdout_patient
            ),
            "model_path": str(
                model_path
            ),
            "principal_angle_1_deg": (
                theta1
            ),
            "principal_angle_2_deg": (
                theta2
            ),
            "cos_principal_angle_1": (
                cos1
            ),
            "cos_principal_angle_2": (
                cos2
            ),
            **distances,
        })

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        raise RuntimeError(
            "No LOPO models were analyzed."
        )

    return (
        output
        .sort_values(
            "holdout_patient"
        )
        .reset_index(drop=True)
    )


# =============================================================================
# SUMMARY
# =============================================================================
def summarize_one(
    dataframe: pd.DataFrame,
    analysis_type: str,
) -> Dict:
    theta1 = dataframe[
        "principal_angle_1_deg"
    ]

    theta2 = dataframe[
        "principal_angle_2_deg"
    ]

    theta_max = dataframe[
        "maximum_principal_angle_deg"
    ]

    return {
        "analysis_type": (
            analysis_type
        ),
        "n_resamples": int(
            len(
                dataframe
            )
        ),

        "mean_principal_angle_1_deg": float(
            theta1.mean()
        ),
        "median_principal_angle_1_deg": float(
            theta1.median()
        ),
        "q25_principal_angle_1_deg": float(
            theta1.quantile(
                0.25
            )
        ),
        "q75_principal_angle_1_deg": float(
            theta1.quantile(
                0.75
            )
        ),
        "maximum_principal_angle_1_deg": float(
            theta1.max()
        ),

        "mean_principal_angle_2_deg": float(
            theta2.mean()
        ),
        "median_principal_angle_2_deg": float(
            theta2.median()
        ),
        "q25_principal_angle_2_deg": float(
            theta2.quantile(
                0.25
            )
        ),
        "q75_principal_angle_2_deg": float(
            theta2.quantile(
                0.75
            )
        ),
        "maximum_principal_angle_2_deg": float(
            theta2.max()
        ),

        "median_maximum_principal_angle_deg": float(
            theta_max.median()
        ),
        "q75_maximum_principal_angle_deg": float(
            theta_max.quantile(
                0.75
            )
        ),
        "maximum_observed_principal_angle_deg": float(
            theta_max.max()
        ),

        "fraction_max_angle_le_5_deg": float(
            (
                theta_max <= 5.0
            ).mean()
        ),

        "fraction_max_angle_le_10_deg": float(
            (
                theta_max <= 10.0
            ).mean()
        ),

        "fraction_max_angle_le_15_deg": float(
            (
                theta_max <= 15.0
            ).mean()
        ),

        "median_cos_principal_angle_1": float(
            dataframe[
                "cos_principal_angle_1"
            ].median()
        ),

        "median_cos_principal_angle_2": float(
            dataframe[
                "cos_principal_angle_2"
            ].median()
        ),

        "median_spectral_sin_distance": float(
            dataframe[
                "spectral_sin_distance"
            ].median()
        ),

        "median_chordal_distance": float(
            dataframe[
                "chordal_distance"
            ].median()
        ),
    }


def build_summary(
    bootstrap_df: pd.DataFrame,
    lopo_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            summarize_one(
                bootstrap_df,
                "patient_bootstrap",
            ),
            summarize_one(
                lopo_df,
                "leave_one_patient_out",
            ),
        ]
    )

    numeric_columns = (
        summary
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    summary[
        numeric_columns
    ] = (
        summary[
            numeric_columns
        ]
        .round(
            ROUND_DIGITS
        )
    )

    return summary


# =============================================================================
# DOCUMENTATION
# =============================================================================
def make_readme_text(
    summary: pd.DataFrame,
) -> str:
    lines = [
        "PC5-PC6 Principal-Angle Subspace Stability",
        "==========================================",
        "",
        "Purpose",
        "-------",
        (
            "Determine whether the reduced exact component-number stability "
            "of PC5 and PC6 reflects true loss of latent structure or rotation/"
            "permutation within a stable two-dimensional PC5-PC6 subspace."
        ),
        "",
        "Method",
        "------",
        (
            "PC5 and PC6 effective loading vectors are mapped back to original "
            "embedding feature space, orthonormalized, and compared with the "
            "reference PC5-PC6 subspace using principal angles."
        ),
        "",
        "Interpretation",
        "--------------",
        (
            "Two principal angles are returned for every resample. "
            "Angles close to 0 degrees indicate strong subspace preservation."
        ),
        (
            "The larger second principal angle is the more conservative "
            "measure of two-dimensional subspace similarity."
        ),
        "",
        "Important",
        "---------",
        (
            "This analysis is descriptive. It does not alter the frozen "
            "PC3/PC6 primary-axis qualification from step 11."
        ),
        "",
        "Summary",
        "-------",
        summary.to_string(
            index=False
        ),
        "",
    ]

    return "\n".join(
        lines
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    check_file(
        REFERENCE_MODEL
    )

    check_directory(
        BOOTSTRAP_ROOT
    )

    check_directory(
        LOPO_ROOT
    )

    print(
        "=" * 110
    )
    print(
        "PC5-PC6 PRINCIPAL-ANGLE "
        "SUBSPACE STABILITY"
    )
    print(
        "=" * 110
    )

    print(
        f"[REFERENCE MODEL]     "
        f"{REFERENCE_MODEL}"
    )

    print(
        f"[BOOTSTRAP ROOT]      "
        f"{BOOTSTRAP_ROOT}"
    )

    print(
        f"[LOPO ROOT]           "
        f"{LOPO_ROOT}"
    )

    print(
        f"[OUTPUT]              "
        f"{OUT_DIR}"
    )

    reference_model = (
        load_pca_model(
            REFERENCE_MODEL
        )
    )

    reference_matrix = (
        effective_pc56_matrix(
            reference_model
        )
    )

    q_reference = (
        orthonormal_basis(
            reference_matrix
        )
    )

    print(
        "\n[INFO] Analyzing bootstrap PCA models..."
    )

    bootstrap_df = (
        analyze_bootstraps(
            q_reference
        )
    )

    print(
        f"[BOOTSTRAP MODELS ANALYZED] "
        f"{len(bootstrap_df)}"
    )

    if (
        len(bootstrap_df)
        != EXPECTED_BOOTSTRAPS
    ):
        print(
            f"[WARNING] Expected "
            f"{EXPECTED_BOOTSTRAPS} bootstrap models "
            f"but analyzed {len(bootstrap_df)}."
        )

    print(
        "\n[INFO] Analyzing LOPO PCA models..."
    )

    lopo_df = (
        analyze_lopo(
            q_reference
        )
    )

    print(
        f"[LOPO MODELS ANALYZED] "
        f"{len(lopo_df)}"
    )

    if (
        len(lopo_df)
        != EXPECTED_LOPO_FOLDS
    ):
        print(
            f"[WARNING] Expected "
            f"{EXPECTED_LOPO_FOLDS} LOPO models "
            f"but analyzed {len(lopo_df)}."
        )

    bootstrap_df = (
        bootstrap_df.round(
            ROUND_DIGITS
        )
    )

    lopo_df = (
        lopo_df.round(
            ROUND_DIGITS
        )
    )

    bootstrap_df.to_csv(
        OUT_BOOTSTRAP,
        index=False,
    )

    lopo_df.to_csv(
        OUT_LOPO,
        index=False,
    )

    summary = build_summary(
        bootstrap_df,
        lopo_df,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    metadata = {
        "analysis_name": (
            "PC5-PC6 principal-angle "
            "subspace stability"
        ),

        "analysis_role": (
            "Resolve whether PC5-PC6 axis swapping "
            "reflects local rotation within a stable "
            "two-dimensional latent subspace"
        ),

        "reference_model": str(
            REFERENCE_MODEL
        ),

        "reference_model_sha256": (
            sha256_file(
                REFERENCE_MODEL
            )
        ),

        "reference_components": [
            "PC5",
            "PC6",
        ],

        "component_indices_zero_based": [
            PC5_INDEX,
            PC6_INDEX,
        ],

        "effective_loading_definition": (
            "PCA component divided elementwise by "
            "StandardScaler scale."
        ),

        "subspace_basis_method": (
            "Reduced QR orthonormalization."
        ),

        "principal_angle_method": (
            "Singular values of Q_reference.T @ Q_resampled; "
            "theta_i = arccos(sigma_i)."
        ),

        "angles_reported_in": (
            "degrees"
        ),

        "n_bootstrap_models_analyzed": int(
            len(
                bootstrap_df
            )
        ),

        "n_lopo_models_analyzed": int(
            len(
                lopo_df
            )
        ),

        "tumor_labels_used": False,

        "morphology_used": False,

        "external_validation_used": False,

        "pca_refitting_performed_by_this_script": False,

        "changes_step11_axis_qualification": False,

        "outputs": {
            "bootstrap_principal_angles": str(
                OUT_BOOTSTRAP
            ),
            "lopo_principal_angles": str(
                OUT_LOPO
            ),
            "summary": str(
                OUT_SUMMARY
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
            summary
        ),
        encoding="utf-8",
    )

    print(
        "\n" + "=" * 110
    )

    print(
        "PC5-PC6 SUBSPACE STABILITY SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nWorst bootstrap resamples "
        "by maximum principal angle:"
    )

    print(
        bootstrap_df.nlargest(
            10,
            "maximum_principal_angle_deg",
        )[
            [
                "bootstrap",
                "principal_angle_1_deg",
                "principal_angle_2_deg",
                "maximum_principal_angle_deg",
                "spectral_sin_distance",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nLOPO folds:"
    )

    print(
        lopo_df[
            [
                "holdout_patient",
                "principal_angle_1_deg",
                "principal_angle_2_deg",
                "maximum_principal_angle_deg",
                "spectral_sin_distance",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n[OK] Saved:"
    )

    print(
        f"  {OUT_BOOTSTRAP}"
    )

    print(
        f"  {OUT_LOPO}"
    )

    print(
        f"  {OUT_SUMMARY}"
    )

    print(
        f"  {OUT_METADATA}"
    )

    print(
        f"  {OUT_README}"
    )

    print(
        "\n[DONE] PC5-PC6 principal-angle "
        "subspace stability analysis complete."
    )


if __name__ == "__main__":
    main()
