#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
29c_evaluate_frozen_tcga_prad_pc3_pc6_gene_associations.py

STEP 29C
========
EVALUATE FROZEN TCGA-PRAD PC3 / PC6 GENOME-WIDE GENE ASSOCIATIONS

VERSION
-------
1.0.0

SCHEMA
------
29c-tcga-prad-pc3-pc6-genomewide-gene-association-schema-1.0

ROLE
----
PRESPECIFIED GENOME-WIDE MOLECULAR INFERENCE.

This is the first formal step that combines:

    frozen patient-level PC3 / PC6 scores

with:

    frozen Step29B RNA expression matrix.

FROZEN AUTHORITY
----------------
Patient-level representation:
    Step27G

RNA patient order:
    Step29B

Eligible gene universe:
    Step29B

Expression:
    frozen_matched_patient_by_eligible_gene_log2_tpm_plus1.npy

Expected frozen molecular cohort:
    401 patients

Expected frozen gene universe:
    20,066 genes

TESTS
-----
For every eligible gene:

    Spearman(
        patient_oriented_PC3,
        log2(TPM + 1)
    )

and:

    Spearman(
        patient_oriented_PC6,
        log2(TPM + 1)
    )

Alternative:
    two-sided

MULTIPLICITY
------------
PC3:
    BH-FDR across ALL eligible genes.

PC6:
    BH-FDR across ALL eligible genes.

The two axes define separate frozen genome-wide families.

Threshold:
    q < 0.05

BOOTSTRAP
---------
Genome-wide per-gene bootstrap:
    NO

This follows Step29A v1.0.1.

IMPORTANT
---------
This script MUST NOT:

    - change patient inclusion;
    - change patient order;
    - change gene inclusion;
    - change gene order;
    - recalculate RNA preprocessing;
    - reorient PC3;
    - reorient PC6;
    - select genes before multiplicity correction;
    - remove nonsignificant genes;
    - perform pathway analysis.

All eligible genes are retained in the authoritative result tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import t as student_t


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "29c-tcga-prad-pc3-pc6-"
    "genomewide-gene-association-schema-1.0"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


DEFAULT_STEP29A_SPEC = (
    RESULTS
    / "29a_tcga_prad_pc3_pc6_molecular_association_specification"
    / "tcga_prad_pc3_pc6_molecular_association_specification.json"
)


DEFAULT_STEP29B_DIR = (
    RESULTS
    / "29b_frozen_tcga_prad_rna_analysis_matrix"
)

DEFAULT_STEP29B_SUMMARY = (
    DEFAULT_STEP29B_DIR
    / "tcga_prad_frozen_rna_analysis_matrix_summary.json"
)

DEFAULT_STEP29B_PATIENT_ORDER = (
    DEFAULT_STEP29B_DIR
    / "frozen_matched_rna_patient_order.csv"
)

DEFAULT_STEP29B_ELIGIBLE_GENES = (
    DEFAULT_STEP29B_DIR
    / "eligible_ensembl_gene_universe.csv"
)

DEFAULT_STEP29B_EXPRESSION = (
    DEFAULT_STEP29B_DIR
    / "frozen_matched_patient_by_eligible_gene_log2_tpm_plus1.npy"
)


DEFAULT_STEP27G_PATIENT_MANIFEST = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "frozen_tcga_prad_patient_representation_manifest.csv"
)


DEFAULT_OUT_DIR = (
    RESULTS
    / "29c_tcga_prad_pc3_pc6_genomewide_gene_associations"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_FROZEN_WSI_PATIENTS = 403

EXPECTED_RNA_PATIENTS = 401

EXPECTED_ELIGIBLE_GENES = 20066


PC3_COLUMN = "patient_oriented_PC3"
PC6_COLUMN = "patient_oriented_PC6"

AXES = [
    ("PC3", PC3_COLUMN),
    ("PC6", PC6_COLUMN),
]

FDR_THRESHOLD = 0.05


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
    width: int = 122,
) -> None:

    print()
    print("=" * width)
    print(title)
    print("=" * width)


def require_file(
    path: Path,
    label: str,
) -> None:

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} missing: {path}"
        )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def save_json_atomic(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:

        while True:

            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def parse_boolish_series(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


# =============================================================================
# BH-FDR
# =============================================================================

def benjamini_hochberg(
    p_values: np.ndarray,
) -> np.ndarray:

    p = np.asarray(
        p_values,
        dtype=np.float64,
    )

    if p.ndim != 1:

        raise RuntimeError(
            "BH-FDR requires a one-dimensional p-value vector."
        )

    if p.size == 0:

        raise RuntimeError(
            "BH-FDR received zero p-values."
        )

    if not np.isfinite(
        p
    ).all():

        raise RuntimeError(
            "BH-FDR received NaN/Inf p-values."
        )

    if (
        (p < 0)
        | (p > 1)
    ).any():

        raise RuntimeError(
            "BH-FDR received p-values outside [0, 1]."
        )

    m = p.size

    order = np.argsort(
        p,
        kind="mergesort",
    )

    ranked_p = p[
        order
    ]

    ranks = np.arange(
        1,
        m + 1,
        dtype=np.float64,
    )

    adjusted = (
        ranked_p
        * m
        / ranks
    )

    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    output = np.empty(
        m,
        dtype=np.float64,
    )

    output[
        order
    ] = adjusted

    return output


# =============================================================================
# STEP29A VALIDATION
# =============================================================================

def validate_step29a(
    path: Path,
) -> dict:

    require_file(
        path,
        "Step29A specification",
    )

    spec = load_json(
        path
    )

    if (
        spec.get(
            "version"
        )
        != "1.0.1"
    ):

        raise RuntimeError(
            "Step29A version is not frozen v1.0.1."
        )

    if (
        spec.get(
            "schema"
        )
        !=
        "29a-tcga-prad-pc3-pc6-molecular-association-"
        "specification-schema-1.0.1"
    ):

        raise RuntimeError(
            "Unexpected Step29A schema."
        )

    representation = spec.get(
        "representation",
        {},
    )

    if int(
        representation.get(
            "n_frozen_patients",
            -1,
        )
    ) != EXPECTED_FROZEN_WSI_PATIENTS:

        raise RuntimeError(
            "Step29A frozen patient count changed."
        )

    rna = spec.get(
        "rna",
        {},
    )

    if (
        rna.get(
            "unit"
        )
        != "TPM"
    ):

        raise RuntimeError(
            "Step29A RNA authority changed."
        )

    if (
        rna.get(
            "transformation"
        )
        != "log2(TPM + 1)"
    ):

        raise RuntimeError(
            "Step29A RNA transformation changed."
        )

    gene_filter = spec.get(
        "gene_filter",
        {},
    )

    if not np.isclose(
        float(
            gene_filter.get(
                "tpm_threshold",
                -1,
            )
        ),
        1.0,
    ):

        raise RuntimeError(
            "Step29A TPM threshold changed."
        )

    if not np.isclose(
        float(
            gene_filter.get(
                "patient_fraction",
                -1,
            )
        ),
        0.20,
    ):

        raise RuntimeError(
            "Step29A patient-fraction threshold changed."
        )

    association = spec.get(
        "gene_association",
        {},
    )

    if (
        association.get(
            "statistic"
        )
        != "Spearman rho"
    ):

        raise RuntimeError(
            "Step29A gene statistic changed."
        )

    if (
        association.get(
            "sidedness"
        )
        != "two-sided"
    ):

        raise RuntimeError(
            "Step29A sidedness changed."
        )

    if (
        association.get(
            "fdr_method"
        )
        != "Benjamini-Hochberg"
    ):

        raise RuntimeError(
            "Step29A FDR method changed."
        )

    if not np.isclose(
        float(
            association.get(
                "q_threshold",
                -1,
            )
        ),
        FDR_THRESHOLD,
    ):

        raise RuntimeError(
            "Step29A FDR threshold changed."
        )

    bootstrap = spec.get(
        "bootstrap",
        {},
    )

    if bool(
        bootstrap.get(
            "genomewide_per_gene_bootstrap",
            True,
        )
    ):

        raise RuntimeError(
            "Step29A unexpectedly requires genome-wide bootstrap."
        )

    return spec


# =============================================================================
# STEP29B VALIDATION
# =============================================================================

def validate_and_load_step29b(
    summary_path: Path,
    patient_order_path: Path,
    gene_order_path: Path,
    expression_path: Path,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    dict,
]:

    require_file(
        summary_path,
        "Step29B summary",
    )

    require_file(
        patient_order_path,
        "Step29B patient order",
    )

    require_file(
        gene_order_path,
        "Step29B eligible gene order",
    )

    require_file(
        expression_path,
        "Step29B eligible expression matrix",
    )

    summary = load_json(
        summary_path
    )

    if (
        summary.get(
            "status"
        )
        != "PASS"
    ):

        raise RuntimeError(
            "Step29B summary does not report PASS."
        )

    if (
        summary.get(
            "analysis_role"
        )
        != "RNA_PROVENANCE_QC_AND_MATERIALIZATION_ONLY"
    ):

        raise RuntimeError(
            "Unexpected Step29B analysis role."
        )

    frozen_info = summary.get(
        "frozen_wsi_cohort",
        {},
    )

    if int(
        frozen_info.get(
            "patients",
            -1,
        )
    ) != EXPECTED_FROZEN_WSI_PATIENTS:

        raise RuntimeError(
            "Step29B frozen WSI patient count changed."
        )

    if int(
        frozen_info.get(
            "matched_rna_patients",
            -1,
        )
    ) != EXPECTED_RNA_PATIENTS:

        raise RuntimeError(
            "Step29B matched RNA patient count changed."
        )

    expression_info = summary.get(
        "expression",
        {},
    )

    if int(
        expression_info.get(
            "genes_after_filter",
            -1,
        )
    ) != EXPECTED_ELIGIBLE_GENES:

        raise RuntimeError(
            "Step29B eligible gene count changed."
        )

    firewall = summary.get(
        "firewall",
        {},
    )

    if bool(
        firewall.get(
            "step27g_pc3_numerical_values_read",
            True,
        )
    ):

        raise RuntimeError(
            "Step29B unexpectedly read PC3 values."
        )

    if bool(
        firewall.get(
            "step27g_pc6_numerical_values_read",
            True,
        )
    ):

        raise RuntimeError(
            "Step29B unexpectedly read PC6 values."
        )

    # ------------------------------------------------------------------
    # Patient order.
    # ------------------------------------------------------------------

    patient_order = pd.read_csv(
        patient_order_path
    )

    required_patient_columns = {
        "patient_row_index",
        "patient_id",
    }

    missing = sorted(
        required_patient_columns
        - set(
            patient_order.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step29B patient-order schema changed. "
            f"Missing: {missing}"
        )

    if len(
        patient_order
    ) != EXPECTED_RNA_PATIENTS:

        raise RuntimeError(
            "Step29B patient-order row count changed."
        )

    if (
        patient_order[
            "patient_id"
        ].astype(str)
        .nunique()
        != EXPECTED_RNA_PATIENTS
    ):

        raise RuntimeError(
            "Step29B patient order is not unique."
        )

    expected_row_index = np.arange(
        EXPECTED_RNA_PATIENTS,
        dtype=int,
    )

    observed_row_index = (
        patient_order[
            "patient_row_index"
        ]
        .to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        observed_row_index,
        expected_row_index,
    ):

        raise RuntimeError(
            "Step29B patient row index is not exactly 0..400."
        )

    # ------------------------------------------------------------------
    # Gene order.
    # ------------------------------------------------------------------

    gene_order = pd.read_csv(
        gene_order_path
    )

    required_gene_columns = {
        "eligible_gene_column_index",
        "ensembl_gene_id",
        "gene_name",
        "gene_type",
        "passes_frozen_detectability_filter",
    }

    missing = sorted(
        required_gene_columns
        - set(
            gene_order.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step29B eligible gene schema changed. "
            f"Missing: {missing}"
        )

    if len(
        gene_order
    ) != EXPECTED_ELIGIBLE_GENES:

        raise RuntimeError(
            "Step29B eligible gene-order row count changed."
        )

    if (
        gene_order[
            "ensembl_gene_id"
        ]
        .astype(str)
        .nunique()
        != EXPECTED_ELIGIBLE_GENES
    ):

        raise RuntimeError(
            "Step29B eligible gene IDs are not unique."
        )

    expected_gene_index = np.arange(
        EXPECTED_ELIGIBLE_GENES,
        dtype=int,
    )

    observed_gene_index = (
        gene_order[
            "eligible_gene_column_index"
        ]
        .to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        observed_gene_index,
        expected_gene_index,
    ):

        raise RuntimeError(
            "Step29B eligible gene column order changed."
        )

    passes_filter = parse_boolish_series(
        gene_order[
            "passes_frozen_detectability_filter"
        ]
    )

    if not passes_filter.all():

        raise RuntimeError(
            "Step29B eligible gene table contains genes "
            "not passing the frozen detectability filter."
        )

    # ------------------------------------------------------------------
    # Expression matrix.
    # ------------------------------------------------------------------

    expression = np.load(
        expression_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_shape = (
        EXPECTED_RNA_PATIENTS,
        EXPECTED_ELIGIBLE_GENES,
    )

    if expression.shape != expected_shape:

        raise RuntimeError(
            "Step29B RNA matrix shape changed: "
            f"{expression.shape} != {expected_shape}"
        )

    if not np.issubdtype(
        expression.dtype,
        np.floating,
    ):

        raise RuntimeError(
            "RNA matrix is not floating point."
        )

    # Materialize in memory for rank computation.
    expression_array = np.asarray(
        expression,
        dtype=np.float64,
    )

    if not np.isfinite(
        expression_array
    ).all():

        raise RuntimeError(
            "Step29B eligible log2(TPM+1) matrix contains NaN/Inf."
        )

    if (
        expression_array < 0
    ).any():

        raise RuntimeError(
            "Step29B log2(TPM+1) matrix contains negative values."
        )

    return (
        patient_order,
        gene_order,
        expression_array,
        summary,
    )


# =============================================================================
# STEP27G PC SCORE AUTHORITY
# =============================================================================

def load_frozen_scores_in_step29b_order(
    path: Path,
    patient_order: pd.DataFrame,
) -> pd.DataFrame:

    require_file(
        path,
        "Step27G patient manifest",
    )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "patient_id",
        PC3_COLUMN,
        PC6_COLUMN,
        "patient_aggregation_rule",
    }

    missing = sorted(
        required_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step27G schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        df
    ) != EXPECTED_FROZEN_WSI_PATIENTS:

        raise RuntimeError(
            "Step27G patient count changed."
        )

    if (
        df[
            "patient_id"
        ].astype(str)
        .nunique()
        != EXPECTED_FROZEN_WSI_PATIENTS
    ):

        raise RuntimeError(
            "Step27G patient IDs are not unique."
        )

    indexed = (
        df
        .set_index(
            "patient_id",
            verify_integrity=True,
        )
    )

    required_ids = (
        patient_order[
            "patient_id"
        ]
        .astype(str)
        .tolist()
    )

    missing_ids = [
        patient_id
        for patient_id
        in required_ids
        if patient_id
        not in indexed.index
    ]

    if missing_ids:

        raise RuntimeError(
            "Step29B RNA patient IDs missing from Step27G:\n"
            + "\n".join(
                missing_ids
            )
        )

    # IMPORTANT:
    # Reindex EXACTLY to Step29B order.
    # No independent sorting or dropping.
    aligned = (
        indexed.loc[
            required_ids,
            [
                PC3_COLUMN,
                PC6_COLUMN,
                "patient_aggregation_rule",
            ],
        ]
        .copy()
    )

    aligned.insert(
        0,
        "patient_id",
        required_ids,
    )

    aligned = aligned.reset_index(
        drop=True
    )

    if not np.array_equal(
        aligned[
            "patient_id"
        ].astype(str)
        .to_numpy(),
        patient_order[
            "patient_id"
        ].astype(str)
        .to_numpy(),
    ):

        raise RuntimeError(
            "PC score order does not exactly match Step29B patient order."
        )

    for column in [
        PC3_COLUMN,
        PC6_COLUMN,
    ]:

        aligned[
            column
        ] = pd.to_numeric(
            aligned[
                column
            ],
            errors="coerce",
        )

        values = aligned[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(
            values
        ).all():

            raise RuntimeError(
                f"{column} contains NaN/Inf in the matched RNA cohort."
            )

        if np.unique(
            values
        ).size < 2:

            raise RuntimeError(
                f"{column} is constant."
            )

    return aligned


# =============================================================================
# VECTORIZED SPEARMAN
# =============================================================================

def rank_columns(
    expression: np.ndarray,
) -> np.ndarray:

    """
    Rank each gene across patients using average ranks for ties.

    Shape:
        patients x genes

    scipy.stats.rankdata supports axis=0.
    """

    ranked = rankdata(
        expression,
        axis=0,
        method="average",
    )

    return np.asarray(
        ranked,
        dtype=np.float64,
    )


def vectorized_spearman(
    score: np.ndarray,
    ranked_expression: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    """
    Spearman correlation is Pearson correlation of ranks.

    Returns:
        rho
        two-sided p-value

    P-value follows the conventional t approximation:

        t = rho * sqrt((n - 2) / (1 - rho^2))

        df = n - 2
    """

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    if score.ndim != 1:

        raise RuntimeError(
            "Score vector must be one-dimensional."
        )

    n = score.shape[
        0
    ]

    if (
        ranked_expression.shape[
            0
        ]
        != n
    ):

        raise RuntimeError(
            "Score / RNA patient dimension mismatch."
        )

    score_rank = rankdata(
        score,
        method="average",
    ).astype(
        np.float64
    )

    score_centered = (
        score_rank
        - score_rank.mean()
    )

    score_ss = np.sum(
        score_centered ** 2
    )

    if score_ss <= 0:

        raise RuntimeError(
            "Ranked score has zero variance."
        )

    expression_centered = (
        ranked_expression
        - ranked_expression.mean(
            axis=0,
            keepdims=True,
        )
    )

    gene_ss = np.sum(
        expression_centered ** 2,
        axis=0,
    )

    constant_gene_mask = (
        gene_ss <= 0
    )

    if constant_gene_mask.any():

        indices = np.flatnonzero(
            constant_gene_mask
        )

        raise RuntimeError(
            "One or more frozen eligible genes are constant across "
            f"the 401 matched patients. First indices: "
            f"{indices[:20].tolist()}"
        )

    numerator = np.sum(
        expression_centered
        * score_centered[
            :,
            None
        ],
        axis=0,
    )

    denominator = np.sqrt(
        score_ss
        * gene_ss
    )

    rho = (
        numerator
        / denominator
    )

    # Numerical protection only.
    rho = np.clip(
        rho,
        -1.0,
        1.0,
    )

    dof = n - 2

    one_minus_r2 = np.maximum(
        1.0 - rho ** 2,
        np.finfo(
            np.float64
        ).tiny,
    )

    t_stat = (
        rho
        * np.sqrt(
            dof
            / one_minus_r2
        )
    )

    p_value = (
        2.0
        * student_t.sf(
            np.abs(
                t_stat
            ),
            df=dof,
        )
    )

    if not np.isfinite(
        rho
    ).all():

        raise RuntimeError(
            "Non-finite Spearman rho values produced."
        )

    if not np.isfinite(
        p_value
    ).all():

        raise RuntimeError(
            "Non-finite Spearman p-values produced."
        )

    return (
        rho,
        p_value,
    )


# =============================================================================
# SPEARMAN IMPLEMENTATION QC
# =============================================================================

def validate_vectorized_spearman_against_scipy(
    score: np.ndarray,
    expression: np.ndarray,
    rho_vector: np.ndarray,
    p_vector: np.ndarray,
    axis_name: str,
) -> pd.DataFrame:

    """
    Deterministic spot checks using fixed gene indices.

    These indices are chosen independently of observed associations.
    """

    n_genes = expression.shape[
        1
    ]

    fixed_indices = sorted(
        set(
            [
                0,
                n_genes // 4,
                n_genes // 2,
                (3 * n_genes) // 4,
                n_genes - 1,
            ]
        )
    )

    rows = []

    for gene_index in fixed_indices:

        scipy_result = spearmanr(
            score,
            expression[
                :,
                gene_index
            ],
        )

        scipy_rho = float(
            scipy_result.statistic
        )

        scipy_p = float(
            scipy_result.pvalue
        )

        vector_rho = float(
            rho_vector[
                gene_index
            ]
        )

        vector_p = float(
            p_vector[
                gene_index
            ]
        )

        rho_difference = abs(
            scipy_rho
            - vector_rho
        )

        p_difference = abs(
            scipy_p
            - vector_p
        )

        rows.append({
            "axis": (
                axis_name
            ),

            "gene_column_index": (
                gene_index
            ),

            "scipy_rho": (
                scipy_rho
            ),

            "vectorized_rho": (
                vector_rho
            ),

            "absolute_rho_difference": (
                rho_difference
            ),

            "scipy_p": (
                scipy_p
            ),

            "vectorized_p": (
                vector_p
            ),

            "absolute_p_difference": (
                p_difference
            ),
        })

    audit = pd.DataFrame(
        rows
    )

    if (
        audit[
            "absolute_rho_difference"
        ].max()
        > 1e-10
    ):

        raise RuntimeError(
            f"{axis_name}: vectorized Spearman rho does not "
            "match scipy.stats.spearmanr."
        )

    # scipy p-value implementation can differ at tiny floating-point
    # scale; this tolerance remains extremely strict.
    if (
        audit[
            "absolute_p_difference"
        ].max()
        > 1e-10
    ):

        raise RuntimeError(
            f"{axis_name}: vectorized Spearman p-value does not "
            "match scipy.stats.spearmanr."
        )

    return audit


# =============================================================================
# RESULT TABLE
# =============================================================================

def build_axis_result_table(
    axis_name: str,
    score_column: str,
    gene_table: pd.DataFrame,
    rho: np.ndarray,
    p_value: np.ndarray,
    q_value: np.ndarray,
    n_patients: int,
) -> pd.DataFrame:

    if len(
        gene_table
    ) != len(
        rho
    ):

        raise RuntimeError(
            f"{axis_name}: gene/result length mismatch."
        )

    result = pd.DataFrame({
        "axis": (
            axis_name
        ),

        "axis_score_column": (
            score_column
        ),

        "gene_column_index": (
            gene_table[
                "eligible_gene_column_index"
            ].to_numpy(
                dtype=int
            )
        ),

        "ensembl_gene_id": (
            gene_table[
                "ensembl_gene_id"
            ].astype(str)
            .to_numpy()
        ),

        "gene_name": (
            gene_table[
                "gene_name"
            ].fillna(
                ""
            )
            .astype(str)
            .to_numpy()
        ),

        "gene_type": (
            gene_table[
                "gene_type"
            ].fillna(
                ""
            )
            .astype(str)
            .to_numpy()
        ),

        "n_patients": (
            n_patients
        ),

        "spearman_rho": (
            rho
        ),

        "p_value": (
            p_value
        ),

        "q_value": (
            q_value
        ),

        "significant_q_lt_0_05": (
            q_value
            < FDR_THRESHOLD
        ),

        "association_direction": np.where(
            rho > 0,
            "POSITIVE",
            np.where(
                rho < 0,
                "NEGATIVE",
                "ZERO",
            ),
        ),
    })

    # ------------------------------------------------------------------
    # Rankings.
    #
    # These are descriptive ordering fields only.
    # They do NOT alter significance.
    # ------------------------------------------------------------------

    result[
        "rank_by_abs_rho"
    ] = (
        result[
            "spearman_rho"
        ]
        .abs()
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    result[
        "rank_by_positive_rho"
    ] = (
        result[
            "spearman_rho"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    result[
        "rank_by_negative_rho"
    ] = (
        result[
            "spearman_rho"
        ]
        .rank(
            method="first",
            ascending=True,
        )
        .astype(int)
    )

    return result


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_axis(
    result: pd.DataFrame,
) -> Dict[str, Any]:

    significant = result.loc[
        result[
            "significant_q_lt_0_05"
        ]
    ].copy()

    positive_significant = significant.loc[
        significant[
            "spearman_rho"
        ]
        > 0
    ]

    negative_significant = significant.loc[
        significant[
            "spearman_rho"
        ]
        < 0
    ]

    strongest_positive = (
        result
        .sort_values(
            "spearman_rho",
            ascending=False,
            kind="mergesort",
        )
        .iloc[
            0
        ]
    )

    strongest_negative = (
        result
        .sort_values(
            "spearman_rho",
            ascending=True,
            kind="mergesort",
        )
        .iloc[
            0
        ]
    )

    strongest_absolute = (
        result
        .assign(
            _abs_rho=lambda x: (
                x[
                    "spearman_rho"
                ].abs()
            )
        )
        .sort_values(
            [
                "_abs_rho",
                "ensembl_gene_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .iloc[
            0
        ]
    )

    return {
        "genes_tested": int(
            len(
                result
            )
        ),

        "genes_significant_q_lt_0_05": int(
            len(
                significant
            )
        ),

        "significant_positive": int(
            len(
                positive_significant
            )
        ),

        "significant_negative": int(
            len(
                negative_significant
            )
        ),

        "minimum_p_value": float(
            result[
                "p_value"
            ].min()
        ),

        "minimum_q_value": float(
            result[
                "q_value"
            ].min()
        ),

        "median_absolute_rho": float(
            result[
                "spearman_rho"
            ].abs()
            .median()
        ),

        "maximum_absolute_rho": float(
            result[
                "spearman_rho"
            ].abs()
            .max()
        ),

        "strongest_positive": {
            "ensembl_gene_id": str(
                strongest_positive[
                    "ensembl_gene_id"
                ]
            ),

            "gene_name": str(
                strongest_positive[
                    "gene_name"
                ]
            ),

            "rho": float(
                strongest_positive[
                    "spearman_rho"
                ]
            ),

            "p": float(
                strongest_positive[
                    "p_value"
                ]
            ),

            "q": float(
                strongest_positive[
                    "q_value"
                ]
            ),
        },

        "strongest_negative": {
            "ensembl_gene_id": str(
                strongest_negative[
                    "ensembl_gene_id"
                ]
            ),

            "gene_name": str(
                strongest_negative[
                    "gene_name"
                ]
            ),

            "rho": float(
                strongest_negative[
                    "spearman_rho"
                ]
            ),

            "p": float(
                strongest_negative[
                    "p_value"
                ]
            ),

            "q": float(
                strongest_negative[
                    "q_value"
                ]
            ),
        },

        "strongest_absolute": {
            "ensembl_gene_id": str(
                strongest_absolute[
                    "ensembl_gene_id"
                ]
            ),

            "gene_name": str(
                strongest_absolute[
                    "gene_name"
                ]
            ),

            "rho": float(
                strongest_absolute[
                    "spearman_rho"
                ]
            ),

            "p": float(
                strongest_absolute[
                    "p_value"
                ]
            ),

            "q": float(
                strongest_absolute[
                    "q_value"
                ]
            ),
        },
    }


# =============================================================================
# CLAIM BOUNDARY
# =============================================================================

def claim_boundary_text() -> str:

    return """STEP 29C — CLAIM BOUNDARY

ROLE
----
PRESPECIFIED GENOME-WIDE TCGA-PRAD MOLECULAR ASSOCIATION INFERENCE.


IF SUPPORTED, STEP29C MAY ESTABLISH
----------------------------------
1. A frozen histologic axis is associated with transcriptomic variation
   in the independent TCGA-PRAD matched RNA cohort.

2. Genes satisfying the prespecified axis-specific genome-wide
   BH-FDR threshold may be described as expression features associated
   with that frozen axis.

3. The sign of Spearman rho may be used to describe whether expression
   increases or decreases with the oriented histologic axis.

4. Complete PC3 and PC6 genome-wide association profiles may be compared
   descriptively.


STEP29C DOES NOT ESTABLISH
--------------------------
1. Causal molecular mechanism.
2. Direct transcriptional regulation.
3. Cell-type-specific origin.
4. Tumor-cell-specific expression.
5. Pathway activation.
6. Mediation of clinicopathologic associations.
7. Independence from Grade Group.
8. Independence from pathologic T stage.
9. Prognostic utility.
10. Clinical utility.
11. Formal statistical difference between PC3 and PC6 molecular profiles.


PROHIBITED POST-HOC ACTIONS
---------------------------
1. Reorient PC3 or PC6.
2. Replace PC3 or PC6 with another component.
3. Change the 401-patient molecular cohort.
4. Change the 20,066-gene frozen eligible universe.
5. Remove nonsignificant genes from the authoritative genome-wide result.
6. Change FDR families.
7. Apply bootstrap to rescue nonsignificant genes.
8. Introduce pathway interpretation in Step29C.
"""


# =============================================================================
# EXECUTION
# =============================================================================

def execute(
    step29a_spec_path: Path,
    step29b_summary_path: Path,
    step29b_patient_order_path: Path,
    step29b_gene_order_path: Path,
    step29b_expression_path: Path,
    step27g_patient_manifest_path: Path,
    out_dir: Path,
) -> None:

    banner(
        "STEP 29C: EVALUATE FROZEN TCGA-PRAD "
        "PC3 / PC6 GENOME-WIDE GENE ASSOCIATIONS"
    )

    print(
        f"[VERSION]          {VERSION}"
    )

    print(
        f"[SCHEMA]           {SCHEMA}"
    )

    print(
        f"[29A SPEC]         {step29a_spec_path}"
    )

    print(
        f"[29B MATRIX]       {step29b_expression_path}"
    )

    # ------------------------------------------------------------------
    # Validate frozen molecular specification.
    # ------------------------------------------------------------------

    step29a_spec = validate_step29a(
        step29a_spec_path
    )

    # ------------------------------------------------------------------
    # Validate frozen RNA matrix.
    # ------------------------------------------------------------------

    (
        patient_order,
        gene_order,
        expression,
        step29b_summary,
    ) = validate_and_load_step29b(
        summary_path=(
            step29b_summary_path
        ),

        patient_order_path=(
            step29b_patient_order_path
        ),

        gene_order_path=(
            step29b_gene_order_path
        ),

        expression_path=(
            step29b_expression_path
        ),
    )

    # ------------------------------------------------------------------
    # Read PC scores only now.
    # ------------------------------------------------------------------

    banner(
        "LOAD FROZEN PC3 / PC6 PATIENT SCORES"
    )

    scores = load_frozen_scores_in_step29b_order(
        path=(
            step27g_patient_manifest_path
        ),

        patient_order=(
            patient_order
        ),
    )

    print(
        f"[PATIENTS]          {len(scores)}"
    )

    print(
        f"[PC3 COLUMN]        {PC3_COLUMN}"
    )

    print(
        f"[PC6 COLUMN]        {PC6_COLUMN}"
    )

    print(
        "[PATIENT ORDER]     inherited exactly from Step29B"
    )

    print(
        "[PATIENT DROPPING]  NO"
    )

    # ------------------------------------------------------------------
    # Matrix audit.
    # ------------------------------------------------------------------

    banner(
        "FROZEN MOLECULAR MATRIX AUDIT"
    )

    print(
        f"Patients                 : {expression.shape[0]}"
    )

    print(
        f"Eligible genes           : {expression.shape[1]:,}"
    )

    print(
        f"Matrix shape             : {expression.shape}"
    )

    print(
        "Expected shape           : "
        f"({EXPECTED_RNA_PATIENTS}, "
        f"{EXPECTED_ELIGIBLE_GENES})"
    )

    print(
        "Expression               : log2(TPM + 1)"
    )

    print(
        "Gene universe changed    : NO"
    )

    print(
        "Patient cohort changed   : NO"
    )

    # ------------------------------------------------------------------
    # Rank RNA matrix once.
    # ------------------------------------------------------------------

    banner(
        "RANK FROZEN RNA MATRIX"
    )

    print(
        "Ranking expression across 401 patients "
        "for all 20,066 frozen eligible genes..."
    )

    ranked_expression = rank_columns(
        expression
    )

    print(
        "[OK] RNA rank matrix materialized."
    )

    # ------------------------------------------------------------------
    # Output directory.
    # ------------------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Execute both frozen genome-wide families.
    # ------------------------------------------------------------------

    result_tables = {}

    qc_tables = []

    axis_summary = {}

    banner(
        "EXECUTE FROZEN GENOME-WIDE ASSOCIATIONS"
    )

    for axis_name, score_column in AXES:

        score = scores[
            score_column
        ].to_numpy(
            dtype=np.float64
        )

        (
            rho,
            p_value,
        ) = vectorized_spearman(
            score=score,
            ranked_expression=(
                ranked_expression
            ),
        )

        # --------------------------------------------------------------
        # Independent axis-specific genome-wide BH-FDR family.
        # --------------------------------------------------------------

        q_value = benjamini_hochberg(
            p_value
        )

        # --------------------------------------------------------------
        # Deterministic implementation QC.
        # --------------------------------------------------------------

        qc = (
            validate_vectorized_spearman_against_scipy(
                score=score,
                expression=expression,
                rho_vector=rho,
                p_vector=p_value,
                axis_name=axis_name,
            )
        )

        qc_tables.append(
            qc
        )

        # --------------------------------------------------------------
        # Authoritative complete table.
        # --------------------------------------------------------------

        result = build_axis_result_table(
            axis_name=axis_name,
            score_column=score_column,
            gene_table=gene_order,
            rho=rho,
            p_value=p_value,
            q_value=q_value,
            n_patients=(
                EXPECTED_RNA_PATIENTS
            ),
        )

        result_tables[
            axis_name
        ] = result

        axis_summary[
            axis_name
        ] = summarize_axis(
            result
        )

        n_sig = int(
            result[
                "significant_q_lt_0_05"
            ].sum()
        )

        n_pos = int(
            (
                result[
                    "significant_q_lt_0_05"
                ]
                &
                (
                    result[
                        "spearman_rho"
                    ]
                    > 0
                )
            ).sum()
        )

        n_neg = int(
            (
                result[
                    "significant_q_lt_0_05"
                ]
                &
                (
                    result[
                        "spearman_rho"
                    ]
                    < 0
                )
            ).sum()
        )

        print(
            f"[{axis_name}] genes={len(result):,} | "
            f"q<0.05={n_sig:,} | "
            f"positive={n_pos:,} | "
            f"negative={n_neg:,} | "
            f"max|rho|="
            f"{result['spearman_rho'].abs().max():.4f}"
        )

    # ------------------------------------------------------------------
    # Save complete authoritative results.
    # ------------------------------------------------------------------

    pc3_result_path = (
        out_dir
        / "tcga_prad_pc3_genomewide_gene_associations.csv"
    )

    pc6_result_path = (
        out_dir
        / "tcga_prad_pc6_genomewide_gene_associations.csv"
    )

    result_tables[
        "PC3"
    ].to_csv(
        pc3_result_path,
        index=False,
    )

    result_tables[
        "PC6"
    ].to_csv(
        pc6_result_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Significant-only convenience tables.
    #
    # These are NOT authoritative substitutes for full tables.
    # ------------------------------------------------------------------

    pc3_sig_path = (
        out_dir
        / "tcga_prad_pc3_fdr_significant_gene_subset.csv"
    )

    pc6_sig_path = (
        out_dir
        / "tcga_prad_pc6_fdr_significant_gene_subset.csv"
    )

    (
        result_tables[
            "PC3"
        ]
        .loc[
            result_tables[
                "PC3"
            ][
                "significant_q_lt_0_05"
            ]
        ]
        .sort_values(
            [
                "q_value",
                "p_value",
                "ensembl_gene_id",
            ],
            kind="mergesort",
        )
        .to_csv(
            pc3_sig_path,
            index=False,
        )
    )

    (
        result_tables[
            "PC6"
        ]
        .loc[
            result_tables[
                "PC6"
            ][
                "significant_q_lt_0_05"
            ]
        ]
        .sort_values(
            [
                "q_value",
                "p_value",
                "ensembl_gene_id",
            ],
            kind="mergesort",
        )
        .to_csv(
            pc6_sig_path,
            index=False,
        )
    )

    # ------------------------------------------------------------------
    # Complete signed-rho rankings for Step29D.
    # ------------------------------------------------------------------

    pc3_ranking_path = (
        out_dir
        / "tcga_prad_pc3_complete_signed_rho_ranking.csv"
    )

    pc6_ranking_path = (
        out_dir
        / "tcga_prad_pc6_complete_signed_rho_ranking.csv"
    )

    for axis_name, output_path in [
        (
            "PC3",
            pc3_ranking_path,
        ),
        (
            "PC6",
            pc6_ranking_path,
        ),
    ]:

        ranking = (
            result_tables[
                axis_name
            ][
                [
                    "ensembl_gene_id",
                    "gene_name",
                    "gene_type",
                    "spearman_rho",
                    "p_value",
                    "q_value",
                ]
            ]
            .sort_values(
                [
                    "spearman_rho",
                    "ensembl_gene_id",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        ranking.insert(
            0,
            "signed_rho_rank",
            np.arange(
                1,
                len(
                    ranking
                )
                + 1,
                dtype=int,
            ),
        )

        ranking.to_csv(
            output_path,
            index=False,
        )

    # ------------------------------------------------------------------
    # Spearman implementation QC.
    # ------------------------------------------------------------------

    spearman_qc_path = (
        out_dir
        / "vectorized_spearman_implementation_qc.csv"
    )

    spearman_qc = pd.concat(
        qc_tables,
        ignore_index=True,
    )

    spearman_qc.to_csv(
        spearman_qc_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Frozen matched-score manifest.
    # ------------------------------------------------------------------

    score_manifest_path = (
        out_dir
        / "frozen_401_patient_pc3_pc6_score_manifest.csv"
    )

    scores.to_csv(
        score_manifest_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Claim boundary.
    # ------------------------------------------------------------------

    claim_path = (
        out_dir
        / "tcga_prad_pc3_pc6_genomewide_gene_claim_boundary.txt"
    )

    claim_path.write_text(
        claim_boundary_text(),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Source hashes.
    # ------------------------------------------------------------------

    hashes_path = (
        out_dir
        / "source_hashes.csv"
    )

    source_files = [
        (
            "step29a_specification",
            step29a_spec_path,
        ),

        (
            "step29b_summary",
            step29b_summary_path,
        ),

        (
            "step29b_patient_order",
            step29b_patient_order_path,
        ),

        (
            "step29b_eligible_gene_order",
            step29b_gene_order_path,
        ),

        (
            "step29b_eligible_log2_tpm_matrix",
            step29b_expression_path,
        ),

        (
            "step27g_patient_manifest",
            step27g_patient_manifest_path,
        ),
    ]

    hash_rows = []

    for label, path in source_files:

        require_file(
            path,
            label,
        )

        hash_rows.append({
            "source": (
                label
            ),

            "path": str(
                path.resolve()
            ),

            "sha256": (
                sha256_file(
                    path
                )
            ),
        })

    pd.DataFrame(
        hash_rows
    ).to_csv(
        hashes_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary JSON.
    # ------------------------------------------------------------------

    summary_path = (
        out_dir
        / "tcga_prad_pc3_pc6_genomewide_gene_association_summary.json"
    )

    summary = {
        "version": (
            VERSION
        ),

        "schema": (
            SCHEMA
        ),

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "PASS"
        ),

        "analysis_role": (
            "PRESPECIFIED_TCGA_GENOMEWIDE_GENE_ASSOCIATION_INFERENCE"
        ),

        "cohort": {
            "frozen_wsi_patients": (
                EXPECTED_FROZEN_WSI_PATIENTS
            ),

            "matched_rna_patients": (
                EXPECTED_RNA_PATIENTS
            ),

            "patient_order_source": str(
                step29b_patient_order_path.resolve()
            ),

            "patient_reselection": (
                False
            ),

            "patient_dropping_in_step29c": (
                False
            ),
        },

        "gene_universe": {
            "eligible_genes": (
                EXPECTED_ELIGIBLE_GENES
            ),

            "gene_order_source": str(
                step29b_gene_order_path.resolve()
            ),

            "gene_filter_reapplied": (
                False
            ),

            "gene_reselection": (
                False
            ),
        },

        "expression": {
            "matrix": str(
                step29b_expression_path.resolve()
            ),

            "unit_before_transformation": (
                "TPM"
            ),

            "transformation": (
                "log2(TPM + 1)"
            ),

            "matrix_shape": [
                int(
                    expression.shape[
                        0
                    ]
                ),

                int(
                    expression.shape[
                        1
                    ]
                ),
            ],
        },

        "inference": {
            "statistic": (
                "Spearman rho"
            ),

            "alternative": (
                "two-sided"
            ),

            "tie_method": (
                "average ranks"
            ),

            "p_value_method": (
                "Spearman rank correlation t approximation "
                "with df=n-2; implementation QC against "
                "scipy.stats.spearmanr"
            ),

            "multiple_testing": (
                "Benjamini-Hochberg FDR"
            ),

            "families": {
                "PC3": (
                    EXPECTED_ELIGIBLE_GENES
                ),

                "PC6": (
                    EXPECTED_ELIGIBLE_GENES
                ),
            },

            "families_corrected_separately": (
                True
            ),

            "q_threshold": (
                FDR_THRESHOLD
            ),

            "genomewide_bootstrap": (
                False
            ),
        },

        "results": {
            "PC3": (
                axis_summary[
                    "PC3"
                ]
            ),

            "PC6": (
                axis_summary[
                    "PC6"
                ]
            ),
        },

        "firewall": {
            "pca_refit": (
                False
            ),

            "axis_reselection": (
                False
            ),

            "pc3_sign_tuning": (
                False
            ),

            "pc6_sign_tuning": (
                False
            ),

            "rna_preprocessing_changed": (
                False
            ),

            "patient_cohort_changed": (
                False
            ),

            "gene_universe_changed": (
                False
            ),

            "gene_selection_before_fdr": (
                False
            ),

            "nonsignificant_genes_removed_from_authoritative_table": (
                False
            ),

            "pathway_analysis_performed": (
                False
            ),

            "genomewide_bootstrap_performed": (
                False
            ),
        },

        "outputs": {
            "pc3_complete_results": str(
                pc3_result_path.resolve()
            ),

            "pc6_complete_results": str(
                pc6_result_path.resolve()
            ),

            "pc3_significant_subset": str(
                pc3_sig_path.resolve()
            ),

            "pc6_significant_subset": str(
                pc6_sig_path.resolve()
            ),

            "pc3_signed_rho_ranking": str(
                pc3_ranking_path.resolve()
            ),

            "pc6_signed_rho_ranking": str(
                pc6_ranking_path.resolve()
            ),

            "spearman_qc": str(
                spearman_qc_path.resolve()
            ),

            "score_manifest": str(
                score_manifest_path.resolve()
            ),

            "claim_boundary": str(
                claim_path.resolve()
            ),

            "source_hashes": str(
                hashes_path.resolve()
            ),
        },
    }

    save_json_atomic(
        summary_path,
        summary,
    )

    # ------------------------------------------------------------------
    # README.
    # ------------------------------------------------------------------

    readme_path = (
        out_dir
        / "README.txt"
    )

    pc3_summary = axis_summary[
        "PC3"
    ]

    pc6_summary = axis_summary[
        "PC6"
    ]

    readme = f"""STEP 29C
TCGA-PRAD PC3 / PC6 GENOME-WIDE GENE ASSOCIATIONS

Version:
    {VERSION}

Schema:
    {SCHEMA}


ROLE
----
Prespecified genome-wide molecular association inference.


FROZEN COHORT
-------------
WSI patients:
    {EXPECTED_FROZEN_WSI_PATIENTS}

Matched RNA patients:
    {EXPECTED_RNA_PATIENTS}

No patient was added or dropped in Step29C.


FROZEN GENE UNIVERSE
--------------------
Eligible genes:
    {EXPECTED_ELIGIBLE_GENES}

No gene was added or removed in Step29C.


EXPRESSION
----------
Matrix:
    {step29b_expression_path}

Values:
    log2(TPM + 1)


STATISTIC
---------
For every eligible gene:

    Spearman rho

Alternative:
    two-sided

Ties:
    average ranks


MULTIPLICITY
------------
PC3:
    BH-FDR across all {EXPECTED_ELIGIBLE_GENES:,} genes.

PC6:
    BH-FDR across all {EXPECTED_ELIGIBLE_GENES:,} genes.

These are separate prespecified genome-wide families.

Threshold:
    q < {FDR_THRESHOLD}


GENOME-WIDE BOOTSTRAP
---------------------
Performed:
    NO

Consistent with Step29A v1.0.1.


PC3 SUMMARY
-----------
Genes tested:
    {pc3_summary['genes_tested']}

Genes q < 0.05:
    {pc3_summary['genes_significant_q_lt_0_05']}

Positive significant:
    {pc3_summary['significant_positive']}

Negative significant:
    {pc3_summary['significant_negative']}

Maximum |rho|:
    {pc3_summary['maximum_absolute_rho']:.6f}


PC6 SUMMARY
-----------
Genes tested:
    {pc6_summary['genes_tested']}

Genes q < 0.05:
    {pc6_summary['genes_significant_q_lt_0_05']}

Positive significant:
    {pc6_summary['significant_positive']}

Negative significant:
    {pc6_summary['significant_negative']}

Maximum |rho|:
    {pc6_summary['maximum_absolute_rho']:.6f}


AUTHORITATIVE RESULTS
---------------------
PC3:
    {pc3_result_path}

PC6:
    {pc6_result_path}

These complete genome-wide tables are authoritative.

Significant-only files are convenience subsets and are NOT substitutes
for the complete result tables.


PATHWAY ANALYSIS
----------------
Performed in Step29C:
    NO

The complete signed-rho rankings are prepared for the separately frozen
Hallmark analysis in Step29D.
"""

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console final results.
    # ------------------------------------------------------------------

    banner(
        "FINAL FROZEN GENOME-WIDE RESULTS"
    )

    for axis_name in [
        "PC3",
        "PC6",
    ]:

        axis_result = result_tables[
            axis_name
        ]

        summary_axis = axis_summary[
            axis_name
        ]

        print(
            f"[{axis_name}]"
        )

        print(
            f"  patients                   : "
            f"{EXPECTED_RNA_PATIENTS}"
        )

        print(
            f"  genes tested               : "
            f"{EXPECTED_ELIGIBLE_GENES:,}"
        )

        print(
            "  genes q<0.05               : "
            f"{summary_axis['genes_significant_q_lt_0_05']:,}"
        )

        print(
            "  significant positive       : "
            f"{summary_axis['significant_positive']:,}"
        )

        print(
            "  significant negative       : "
            f"{summary_axis['significant_negative']:,}"
        )

        print(
            "  median |rho|               : "
            f"{summary_axis['median_absolute_rho']:.4f}"
        )

        print(
            "  maximum |rho|              : "
            f"{summary_axis['maximum_absolute_rho']:.4f}"
        )

        positive = (
            summary_axis[
                "strongest_positive"
            ]
        )

        negative = (
            summary_axis[
                "strongest_negative"
            ]
        )

        print(
            "  strongest positive         : "
            f"{positive['gene_name']} "
            f"({positive['ensembl_gene_id']}) | "
            f"rho={positive['rho']:+.4f} | "
            f"q={positive['q']:.4g}"
        )

        print(
            "  strongest negative         : "
            f"{negative['gene_name']} "
            f"({negative['ensembl_gene_id']}) | "
            f"rho={negative['rho']:+.4f} | "
            f"q={negative['q']:.4g}"
        )

        print()

    # ------------------------------------------------------------------
    # Execution audit.
    # ------------------------------------------------------------------

    banner(
        "STEP 29C EXECUTION AUDIT"
    )

    audit = [
        (
            "Step29A v1.0.1 specification inherited",
            "YES",
        ),

        (
            "Step29B 401-patient RNA cohort inherited",
            "YES",
        ),

        (
            "Step29B patient order inherited exactly",
            "YES",
        ),

        (
            "Step29B 20,066-gene universe inherited",
            "YES",
        ),

        (
            "Step29B gene order inherited exactly",
            "YES",
        ),

        (
            "Step29B log2(TPM+1) matrix inherited",
            "YES",
        ),

        (
            "Step27G frozen PC3 score used",
            "YES",
        ),

        (
            "Step27G frozen PC6 score used",
            "YES",
        ),

        (
            "Patient cohort changed",
            "NO",
        ),

        (
            "Gene universe changed",
            "NO",
        ),

        (
            "Gene filtering repeated",
            "NO",
        ),

        (
            "RNA preprocessing repeated",
            "NO",
        ),

        (
            "PC3 sign tuning",
            "NO",
        ),

        (
            "PC6 sign tuning",
            "NO",
        ),

        (
            "Spearman statistic used",
            "YES",
        ),

        (
            "Vectorized Spearman checked against scipy",
            "YES",
        ),

        (
            "Two-sided p-values used",
            "YES",
        ),

        (
            "PC3 BH-FDR across all 20,066 genes",
            "YES",
        ),

        (
            "PC6 BH-FDR across all 20,066 genes",
            "YES",
        ),

        (
            "PC3 / PC6 FDR families kept separate",
            "YES",
        ),

        (
            "Genome-wide bootstrap performed",
            "NO",
        ),

        (
            "Nonsignificant genes retained",
            "YES",
        ),

        (
            "Pathway analysis performed",
            "NO",
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<72}: {value}"
        )

    # ------------------------------------------------------------------
    # Final PASS.
    # ------------------------------------------------------------------

    banner(
        "STEP 29C: PASS"
    )

    print(
        f"[PATIENTS]          {EXPECTED_RNA_PATIENTS}"
    )

    print(
        f"[GENES]             {EXPECTED_ELIGIBLE_GENES:,}"
    )

    print(
        "[STATISTIC]         Spearman rho"
    )

    print(
        "[ALTERNATIVE]       two-sided"
    )

    print(
        "[PC3 FDR FAMILY]    20,066 genes"
    )

    print(
        "[PC6 FDR FAMILY]    20,066 genes"
    )

    print(
        "[FDR METHOD]        Benjamini-Hochberg"
    )

    print(
        f"[q THRESHOLD]       < {FDR_THRESHOLD}"
    )

    print(
        "[BOOTSTRAP]         NO genome-wide bootstrap"
    )

    print(
        "[PATHWAYS]          NOT ANALYZED"
    )

    print()

    print(
        "[PC3 SIGNIFICANT]   "
        f"{axis_summary['PC3']['genes_significant_q_lt_0_05']:,}"
    )

    print(
        "[PC6 SIGNIFICANT]   "
        f"{axis_summary['PC6']['genes_significant_q_lt_0_05']:,}"
    )

    print()

    print(
        f"[PC3 RESULTS]       {pc3_result_path}"
    )

    print(
        f"[PC6 RESULTS]       {pc6_result_path}"
    )

    print(
        f"[PC3 SIG SUBSET]    {pc3_sig_path}"
    )

    print(
        f"[PC6 SIG SUBSET]    {pc6_sig_path}"
    )

    print(
        f"[PC3 RANKING]       {pc3_ranking_path}"
    )

    print(
        f"[PC6 RANKING]       {pc6_ranking_path}"
    )

    print(
        f"[SPEARMAN QC]       {spearman_qc_path}"
    )

    print(
        f"[SCORE MANIFEST]    {score_manifest_path}"
    )

    print(
        f"[SUMMARY]           {summary_path}"
    )

    print(
        f"[CLAIM BOUNDARY]    {claim_path}"
    )

    print(
        f"[HASHES]            {hashes_path}"
    )

    print(
        f"[README]            {readme_path}"
    )

    print()

    print(
        "[NEXT] Step29D may execute the frozen MSigDB Hallmark "
        "analysis from the complete signed-rho rankings."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen TCGA-PRAD PC3 / PC6 "
            "genome-wide gene association analysis."
        ),

        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--step29a_spec",
        type=Path,
        default=DEFAULT_STEP29A_SPEC,
    )

    parser.add_argument(
        "--step29b_summary",
        type=Path,
        default=DEFAULT_STEP29B_SUMMARY,
    )

    parser.add_argument(
        "--step29b_patient_order",
        type=Path,
        default=DEFAULT_STEP29B_PATIENT_ORDER,
    )

    parser.add_argument(
        "--step29b_gene_order",
        type=Path,
        default=DEFAULT_STEP29B_ELIGIBLE_GENES,
    )

    parser.add_argument(
        "--step29b_expression",
        type=Path,
        default=DEFAULT_STEP29B_EXPRESSION,
    )

    parser.add_argument(
        "--step27g_patient_manifest",
        type=Path,
        default=DEFAULT_STEP27G_PATIENT_MANIFEST,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    execute(
        step29a_spec_path=(
            args.step29a_spec
            .expanduser()
            .resolve()
        ),

        step29b_summary_path=(
            args.step29b_summary
            .expanduser()
            .resolve()
        ),

        step29b_patient_order_path=(
            args.step29b_patient_order
            .expanduser()
            .resolve()
        ),

        step29b_gene_order_path=(
            args.step29b_gene_order
            .expanduser()
            .resolve()
        ),

        step29b_expression_path=(
            args.step29b_expression
            .expanduser()
            .resolve()
        ),

        step27g_patient_manifest_path=(
            args.step27g_patient_manifest
            .expanduser()
            .resolve()
        ),

        out_dir=(
            args.out_dir
            .expanduser()
            .resolve()
        ),
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        banner(
            "STEP 29C: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)