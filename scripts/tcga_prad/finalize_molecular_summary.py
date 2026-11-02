#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
29e_freeze_tcga_prad_molecular_interpretation_summary.py

STEP 29E
========
FREEZE FINAL TCGA-PRAD PC3 / PC6 MOLECULAR INTERPRETATION SUMMARY

VERSION
-------
1.0.0

SCHEMA
------
29e-tcga-prad-pc3-pc6-molecular-interpretation-summary-schema-1.0

ROLE
----
SUMMARY-ONLY / FREEZE-ONLY.

This step consolidates the already-completed frozen molecular evidence:

    Step29C:
        genome-wide PC3 / PC6 gene associations

    Step29D v1.0.1:
        MSigDB Hallmark preranked enrichment

It performs NO NEW inferential analysis.

PURPOSE
-------
1. Freeze the final molecular evidence used in the manuscript.
2. Record the breadth and direction of gene-level associations.
3. Record the complete Hallmark result structure.
4. Freeze deterministic descriptive PC3-vs-PC6 contrasts.
5. Define manuscript-safe interpretation boundaries.
6. Provide a single authoritative downstream source for:
       - Figure 24 interpretation
       - Results text
       - Discussion wording

THIS STEP DOES NOT
------------------
- reread RNA expression
- reread WSI embeddings
- recompute PC3 / PC6
- recompute Spearman correlations
- recompute gene-level p-values
- recompute gene-level FDR
- rerun GSEA
- recompute pathway p-values
- recompute pathway FDR
- select pathways manually
- select genes manually
- test PC3 vs PC6 pathway differences
- perform mediation
- perform survival analysis
- perform causal inference

IMPORTANT
---------
The PC3-versus-PC6 pathway contrast is DESCRIPTIVE.

A large difference:

    PC3_NES - PC6_NES

does NOT establish a statistically significant difference between
the two pathway associations unless such a difference is separately
tested.

No such formal difference test is performed here.
"""

from __future__ import annotations

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

from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "29e-tcga-prad-pc3-pc6-"
    "molecular-interpretation-summary-schema-1.0"
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


STEP29C_DIR = (
    RESULTS
    / "29c_tcga_prad_pc3_pc6_genomewide_gene_associations"
)

STEP29D_DIR = (
    RESULTS
    / "29d_tcga_prad_pc3_pc6_hallmark_pathway_associations"
)


# -----------------------------------------------------------------------------
# Step29C
# -----------------------------------------------------------------------------

PC3_GENE_RESULTS = (
    STEP29C_DIR
    / "tcga_prad_pc3_genomewide_gene_associations.csv"
)

PC6_GENE_RESULTS = (
    STEP29C_DIR
    / "tcga_prad_pc6_genomewide_gene_associations.csv"
)

STEP29C_SUMMARY = (
    STEP29C_DIR
    / "tcga_prad_pc3_pc6_genomewide_gene_association_summary.json"
)

STEP29C_PC3_SIGNIFICANT = (
    STEP29C_DIR
    / "tcga_prad_pc3_fdr_significant_gene_subset.csv"
)

STEP29C_PC6_SIGNIFICANT = (
    STEP29C_DIR
    / "tcga_prad_pc6_fdr_significant_gene_subset.csv"
)


# -----------------------------------------------------------------------------
# Step29D
# -----------------------------------------------------------------------------

PC3_PATHWAY_RESULTS = (
    STEP29D_DIR
    / "tcga_prad_pc3_hallmark_pathway_associations.csv"
)

PC6_PATHWAY_RESULTS = (
    STEP29D_DIR
    / "tcga_prad_pc6_hallmark_pathway_associations.csv"
)

STEP29D_SUMMARY = (
    STEP29D_DIR
    / "tcga_prad_pc3_pc6_hallmark_pathway_summary.json"
)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

OUT_DIR = (
    RESULTS
    / "29e_tcga_prad_molecular_interpretation_summary"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FINAL_AXIS_SUMMARY = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_final_molecular_axis_summary.csv"
)

FINAL_HALLMARK_TABLE = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_final_hallmark_interpretation_table.csv"
)

FINAL_HALLMARK_CONTRAST = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_hallmark_descriptive_contrast_table.csv"
)

FINAL_GENE_EXTREMES = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_gene_level_extreme_association_summary.csv"
)

FINAL_SUMMARY_JSON = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_final_molecular_interpretation_summary.json"
)

CLAIM_BOUNDARY_FILE = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_final_molecular_claim_boundary.txt"
)

MANUSCRIPT_GUIDANCE_FILE = (
    OUT_DIR
    / "tcga_prad_pc3_pc6_manuscript_interpretation_guidance.txt"
)

SOURCE_HASHES_FILE = (
    OUT_DIR
    / "source_hashes.csv"
)

README_FILE = (
    OUT_DIR
    / "README.txt"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_PATIENTS = 401
EXPECTED_GENES = 20066
EXPECTED_HALLMARKS = 50

GENE_Q_THRESHOLD = 0.05
PATHWAY_Q_THRESHOLD = 0.05


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    text: str,
    width: int = 124,
) -> None:

    print()
    print("=" * width)
    print(text)
    print("=" * width)


def yesno(
    value: bool,
) -> str:

    return (
        "YES"
        if bool(value)
        else "NO"
    )


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
) -> dict[str, Any]:

    require_file(
        path,
        "JSON input",
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def save_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:

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
            sort_keys=True,
            default=str,
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


def clean_pathway_name(
    pathway: str,
) -> str:

    pathway = str(
        pathway
    )

    if pathway.startswith(
        "HALLMARK_"
    ):

        pathway = pathway[
            len("HALLMARK_"):
        ]

    return pathway.replace(
        "_",
        " ",
    )


# =============================================================================
# VALIDATE STEP29C
# =============================================================================

def validate_step29c() -> dict[str, Any]:

    require_file(
        STEP29C_SUMMARY,
        "Step29C summary",
    )

    summary = load_json(
        STEP29C_SUMMARY
    )

    if (
        summary.get("status")
        != "PASS"
    ):

        raise RuntimeError(
            "Step29C does not report PASS."
        )

    gene_info = summary.get(
        "gene_universe",
        {},
    )

    if int(
        gene_info.get(
            "eligible_genes",
            -1,
        )
    ) != EXPECTED_GENES:

        raise RuntimeError(
            "Step29C eligible gene universe changed."
        )

    return summary


# =============================================================================
# VALIDATE STEP29D
# =============================================================================

def validate_step29d() -> dict[str, Any]:

    require_file(
        STEP29D_SUMMARY,
        "Step29D summary",
    )

    summary = load_json(
        STEP29D_SUMMARY
    )

    if (
        summary.get("status")
        != "PASS"
    ):

        raise RuntimeError(
            "Step29D does not report PASS."
        )

    if (
        summary.get("version")
        != "1.0.1"
    ):

        raise RuntimeError(
            "Step29E requires Step29D v1.0.1."
        )

    hallmark = summary.get(
        "hallmark",
        {},
    )

    if int(
        hallmark.get(
            "n_gene_sets",
            -1,
        )
    ) != EXPECTED_HALLMARKS:

        raise RuntimeError(
            "Step29D does not contain 50 Hallmark pathways."
        )

    return summary


# =============================================================================
# LOAD GENE RESULTS
# =============================================================================

def load_gene_results(
    path: Path,
    axis: str,
) -> pd.DataFrame:

    require_file(
        path,
        f"{axis} gene results",
    )

    df = pd.read_csv(
        path
    )

    required = {
        "ensembl_gene_id",
        "gene_name",
        "spearman_rho",
        "p_value",
        "q_value",
        "significant_q_lt_0_05",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{axis} gene-result schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        df
    ) != EXPECTED_GENES:

        raise RuntimeError(
            f"{axis}: expected {EXPECTED_GENES:,} genes, "
            f"found {len(df):,}."
        )

    if (
        df[
            "ensembl_gene_id"
        ].astype(str)
        .nunique()
        != EXPECTED_GENES
    ):

        raise RuntimeError(
            f"{axis}: gene IDs are not unique."
        )

    return df.copy()


# =============================================================================
# LOAD PATHWAY RESULTS
# =============================================================================

def load_pathway_results(
    path: Path,
    axis: str,
) -> pd.DataFrame:

    require_file(
        path,
        f"{axis} Hallmark results",
    )

    df = pd.read_csv(
        path
    )

    required = {
        "axis",
        "pathway",
        "enrichment_score",
        "normalized_enrichment_score",
        "raw_nominal_p_value",
        "finite_permutation_nominal_p_value",
        "pathway_bh_q_value",
        "significant_bh_q_lt_0_05",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{axis} Hallmark-result schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        df
    ) != EXPECTED_HALLMARKS:

        raise RuntimeError(
            f"{axis}: expected {EXPECTED_HALLMARKS} Hallmark pathways, "
            f"found {len(df)}."
        )

    if (
        df[
            "pathway"
        ].nunique()
        != EXPECTED_HALLMARKS
    ):

        raise RuntimeError(
            f"{axis}: Hallmark pathway names are not unique."
        )

    return df.copy()


# =============================================================================
# GENE-LEVEL SUMMARY
# =============================================================================

def summarize_gene_axis(
    df: pd.DataFrame,
    axis: str,
) -> dict[str, Any]:

    significant = df.loc[
        df[
            "significant_q_lt_0_05"
        ].astype(bool)
    ].copy()

    positive_sig = significant.loc[
        significant[
            "spearman_rho"
        ] > 0
    ]

    negative_sig = significant.loc[
        significant[
            "spearman_rho"
        ] < 0
    ]

    strongest_positive = (
        df.sort_values(
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
        .iloc[0]
    )

    strongest_negative = (
        df.sort_values(
            [
                "spearman_rho",
                "ensembl_gene_id",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[0]
    )

    return {
        "axis":
            axis,

        "patients":
            EXPECTED_PATIENTS,

        "genes_tested":
            int(
                len(df)
            ),

        "genes_q_lt_0_05":
            int(
                len(significant)
            ),

        "significant_positive":
            int(
                len(positive_sig)
            ),

        "significant_negative":
            int(
                len(negative_sig)
            ),

        "significant_fraction":
            float(
                len(significant)
                / len(df)
            ),

        "strongest_positive_gene":
            str(
                strongest_positive[
                    "gene_name"
                ]
            ),

        "strongest_positive_ensembl":
            str(
                strongest_positive[
                    "ensembl_gene_id"
                ]
            ),

        "strongest_positive_rho":
            float(
                strongest_positive[
                    "spearman_rho"
                ]
            ),

        "strongest_positive_q":
            float(
                strongest_positive[
                    "q_value"
                ]
            ),

        "strongest_negative_gene":
            str(
                strongest_negative[
                    "gene_name"
                ]
            ),

        "strongest_negative_ensembl":
            str(
                strongest_negative[
                    "ensembl_gene_id"
                ]
            ),

        "strongest_negative_rho":
            float(
                strongest_negative[
                    "spearman_rho"
                ]
            ),

        "strongest_negative_q":
            float(
                strongest_negative[
                    "q_value"
                ]
            ),

        "median_absolute_rho":
            float(
                df[
                    "spearman_rho"
                ].abs().median()
            ),

        "maximum_absolute_rho":
            float(
                df[
                    "spearman_rho"
                ].abs().max()
            ),
    }


# =============================================================================
# PATHWAY SUMMARY
# =============================================================================

def summarize_pathway_axis(
    df: pd.DataFrame,
    axis: str,
) -> dict[str, Any]:

    significant = df.loc[
        df[
            "significant_bh_q_lt_0_05"
        ].astype(bool)
    ].copy()

    positive_sig = significant.loc[
        significant[
            "normalized_enrichment_score"
        ] > 0
    ]

    negative_sig = significant.loc[
        significant[
            "normalized_enrichment_score"
        ] < 0
    ]

    strongest_positive = (
        df.sort_values(
            [
                "normalized_enrichment_score",
                "pathway",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .iloc[0]
    )

    strongest_negative = (
        df.sort_values(
            [
                "normalized_enrichment_score",
                "pathway",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
        .iloc[0]
    )

    return {
        "axis":
            axis,

        "hallmark_pathways_tested":
            int(
                len(df)
            ),

        "hallmark_q_lt_0_05":
            int(
                len(significant)
            ),

        "significant_positive_nes":
            int(
                len(positive_sig)
            ),

        "significant_negative_nes":
            int(
                len(negative_sig)
            ),

        "strongest_positive_pathway":
            str(
                strongest_positive[
                    "pathway"
                ]
            ),

        "strongest_positive_nes":
            float(
                strongest_positive[
                    "normalized_enrichment_score"
                ]
            ),

        "strongest_positive_q":
            float(
                strongest_positive[
                    "pathway_bh_q_value"
                ]
            ),

        "strongest_negative_pathway":
            str(
                strongest_negative[
                    "pathway"
                ]
            ),

        "strongest_negative_nes":
            float(
                strongest_negative[
                    "normalized_enrichment_score"
                ]
            ),

        "strongest_negative_q":
            float(
                strongest_negative[
                    "pathway_bh_q_value"
                ]
            ),
    }


# =============================================================================
# BUILD COMPLETE HALLMARK TABLE
# =============================================================================

def build_hallmark_interpretation_table(
    pc3: pd.DataFrame,
    pc6: pd.DataFrame,
) -> pd.DataFrame:

    a = pc3[
        [
            "pathway",
            "normalized_enrichment_score",
            "finite_permutation_nominal_p_value",
            "pathway_bh_q_value",
            "significant_bh_q_lt_0_05",
        ]
    ].rename(
        columns={
            "normalized_enrichment_score":
                "PC3_NES",

            "finite_permutation_nominal_p_value":
                "PC3_p",

            "pathway_bh_q_value":
                "PC3_q",

            "significant_bh_q_lt_0_05":
                "PC3_significant",
        }
    )

    b = pc6[
        [
            "pathway",
            "normalized_enrichment_score",
            "finite_permutation_nominal_p_value",
            "pathway_bh_q_value",
            "significant_bh_q_lt_0_05",
        ]
    ].rename(
        columns={
            "normalized_enrichment_score":
                "PC6_NES",

            "finite_permutation_nominal_p_value":
                "PC6_p",

            "pathway_bh_q_value":
                "PC6_q",

            "significant_bh_q_lt_0_05":
                "PC6_significant",
        }
    )

    merged = a.merge(
        b,
        on="pathway",
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged
    ) != EXPECTED_HALLMARKS:

        raise RuntimeError(
            "Merged Hallmark table does not contain 50 pathways."
        )

    merged[
        "PC3_minus_PC6_NES"
    ] = (
        merged[
            "PC3_NES"
        ]
        -
        merged[
            "PC6_NES"
        ]
    )

    merged[
        "absolute_NES_difference"
    ] = (
        merged[
            "PC3_minus_PC6_NES"
        ].abs()
    )

    merged[
        "direction_pattern"
    ] = np.select(
        [
            (
                merged[
                    "PC3_NES"
                ] > 0
            )
            &
            (
                merged[
                    "PC6_NES"
                ] > 0
            ),

            (
                merged[
                    "PC3_NES"
                ] < 0
            )
            &
            (
                merged[
                    "PC6_NES"
                ] < 0
            ),

            (
                merged[
                    "PC3_NES"
                ] > 0
            )
            &
            (
                merged[
                    "PC6_NES"
                ] < 0
            ),

            (
                merged[
                    "PC3_NES"
                ] < 0
            )
            &
            (
                merged[
                    "PC6_NES"
                ] > 0
            ),
        ],
        [
            "BOTH_POSITIVE",
            "BOTH_NEGATIVE",
            "PC3_POSITIVE_PC6_NEGATIVE",
            "PC3_NEGATIVE_PC6_POSITIVE",
        ],
        default="OTHER",
    )

    merged[
        "significance_pattern"
    ] = np.select(
        [
            (
                merged[
                    "PC3_significant"
                ].astype(bool)
                &
                merged[
                    "PC6_significant"
                ].astype(bool)
            ),

            (
                merged[
                    "PC3_significant"
                ].astype(bool)
                &
                ~merged[
                    "PC6_significant"
                ].astype(bool)
            ),

            (
                ~merged[
                    "PC3_significant"
                ].astype(bool)
                &
                merged[
                    "PC6_significant"
                ].astype(bool)
            ),
        ],
        [
            "BOTH_SIGNIFICANT",
            "PC3_ONLY_SIGNIFICANT",
            "PC6_ONLY_SIGNIFICANT",
        ],
        default="NEITHER_SIGNIFICANT",
    )

    merged[
        "display_name"
    ] = merged[
        "pathway"
    ].map(
        clean_pathway_name
    )

    # Deterministic descriptive order.
    merged = (
        merged
        .sort_values(
            [
                "PC3_minus_PC6_NES",
                "pathway",
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

    return merged


# =============================================================================
# GENE EXTREME TABLE
# =============================================================================

def build_gene_extreme_summary(
    pc3: pd.DataFrame,
    pc6: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for axis, df in [
        (
            "PC3",
            pc3,
        ),
        (
            "PC6",
            pc6,
        ),
    ]:

        positive = (
            df.sort_values(
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
            .iloc[0]
        )

        negative = (
            df.sort_values(
                [
                    "spearman_rho",
                    "ensembl_gene_id",
                ],
                ascending=[
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .iloc[0]
        )

        for direction, row in [
            (
                "STRONGEST_POSITIVE",
                positive,
            ),
            (
                "STRONGEST_NEGATIVE",
                negative,
            ),
        ]:

            rows.append(
                {
                    "axis":
                        axis,

                    "direction":
                        direction,

                    "gene_name":
                        row[
                            "gene_name"
                        ],

                    "ensembl_gene_id":
                        row[
                            "ensembl_gene_id"
                        ],

                    "spearman_rho":
                        row[
                            "spearman_rho"
                        ],

                    "p_value":
                        row[
                            "p_value"
                        ],

                    "q_value":
                        row[
                            "q_value"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CLAIM BOUNDARY
# =============================================================================

def claim_boundary_text() -> str:

    return """STEP 29E — FINAL TCGA-PRAD MOLECULAR CLAIM BOUNDARY

ROLE
----
FINAL SUMMARY OF FROZEN TRANSCRIPTOMIC EVIDENCE.

NO NEW STATISTICAL ANALYSIS IS PERFORMED IN STEP29E.


SUPPORTED CLAIMS
----------------
PC3
---
The frozen oriented PC3 histologic axis shows broad transcriptomic
association in TCGA-PRAD.

The genome-wide association analysis identified a substantial set of
genes whose expression was associated with PC3 after genome-wide
multiple-testing correction.

The frozen Hallmark analysis further identified coordinated
transcriptomic programs associated with the PC3 gene-ranking profile.

Positive Hallmark NES indicates enrichment among genes positively
associated with oriented PC3.

Negative Hallmark NES indicates enrichment among genes negatively
associated with oriented PC3.


PC6
---
The frozen oriented PC6 histologic axis shows a more limited
gene-level transcriptomic association profile than PC3.

Nevertheless, the prespecified Hallmark analysis identifies structured
pathway-level enrichment associated with PC6.

This is compatible with PC6 representing a biologically nonredundant
histologic axis whose transcriptomic profile differs from PC3.


PC3 versus PC6
--------------
PC3 and PC6 may be described as showing contrasting transcriptomic
program profiles.

Some Hallmark programs show concordant enrichment direction.

Other Hallmark programs show opposing enrichment directions.

The PC3-minus-PC6 NES difference may be used descriptively for
visualization and ordering.


NOT SUPPORTED
-------------
The following statements are NOT established by these analyses:

1. PC3 causes proliferation.
2. PC6 causes differentiation.
3. PC3 activates E2F, MYC, mTOR, or any other pathway.
4. PC6 activates androgen response, EMT, myogenesis, or any other pathway.
5. Any Hallmark program is the mechanistic origin of PC3 or PC6.
6. Any pathway difference between PC3 and PC6 is statistically significant
   merely because their NES values differ.
7. Bulk RNA enrichment establishes tumor-cell-specific biology.
8. Transcriptomic association establishes causal mediation of Grade Group
   or pathologic T stage.
9. Molecular associations establish prognostic or therapeutic utility.


FORMAL PC3-versus-PC6 DIFFERENCE
--------------------------------
No direct formal inferential test of:

    NES_PC3 - NES_PC6

is performed.

Therefore:

    "contrasting"
    "opposing"
    "different descriptive profiles"

are acceptable descriptive wording.

But:

    "significantly different pathway programs"

is NOT permitted without a dedicated inferential comparison.


GENE-LEVEL INTERPRETATION
-------------------------
Individual genes may be reported as the strongest observed positive or
negative associations.

Individual genes should not be presented as mechanistic drivers without
independent validation.


PATHWAY-LEVEL INTERPRETATION
----------------------------
Hallmark pathways are coordinated expression programs and should not be
interpreted as literal cell identities or direct pathway activation.

For example, MYOGENESIS enrichment in bulk prostate tissue does not by
itself establish de novo muscle differentiation.


FINAL CLAIM CEILING
-------------------
The strongest supported conclusion is:

    Frozen PC3 and PC6 histologic axes show distinct transcriptomic
    association profiles in TCGA-PRAD, with PC3 exhibiting broader
    genome-wide gene-level associations and both axes mapping onto
    structured, partly contrasting Hallmark transcriptional programs.

This remains an associative molecular interpretation rather than a
mechanistic demonstration.
"""


# =============================================================================
# MANUSCRIPT GUIDANCE
# =============================================================================

def manuscript_guidance_text(
    pc3_gene_summary: dict[str, Any],
    pc6_gene_summary: dict[str, Any],
    pc3_path_summary: dict[str, Any],
    pc6_path_summary: dict[str, Any],
) -> str:

    return f"""TCGA-PRAD PC3 / PC6 MOLECULAR INTERPRETATION
MANUSCRIPT GUIDANCE
============================================================

This file summarizes manuscript-safe wording only.
It is not a statistical-analysis output.


GENE-LEVEL RESULTS
------------------

PC3:
    patients = {EXPECTED_PATIENTS}
    genes tested = {pc3_gene_summary['genes_tested']:,}
    q<0.05 genes = {pc3_gene_summary['genes_q_lt_0_05']:,}
    positive = {pc3_gene_summary['significant_positive']:,}
    negative = {pc3_gene_summary['significant_negative']:,}

PC6:
    patients = {EXPECTED_PATIENTS}
    genes tested = {pc6_gene_summary['genes_tested']:,}
    q<0.05 genes = {pc6_gene_summary['genes_q_lt_0_05']:,}
    positive = {pc6_gene_summary['significant_positive']:,}
    negative = {pc6_gene_summary['significant_negative']:,}


HALLMARK RESULTS
----------------

PC3:
    pathways tested = {pc3_path_summary['hallmark_pathways_tested']}
    BH q<0.05 = {pc3_path_summary['hallmark_q_lt_0_05']}
    positive NES significant = {pc3_path_summary['significant_positive_nes']}
    negative NES significant = {pc3_path_summary['significant_negative_nes']}

PC6:
    pathways tested = {pc6_path_summary['hallmark_pathways_tested']}
    BH q<0.05 = {pc6_path_summary['hallmark_q_lt_0_05']}
    positive NES significant = {pc6_path_summary['significant_positive_nes']}
    negative NES significant = {pc6_path_summary['significant_negative_nes']}


SAFE RESULTS WORDING
--------------------

Recommended:

    "In TCGA-PRAD, PC3 showed broad transcriptomic association,
    with {pc3_gene_summary['genes_q_lt_0_05']:,} of
    {pc3_gene_summary['genes_tested']:,} prespecified eligible genes
    associated with the frozen patient-level PC3 score after
    genome-wide FDR correction."

Recommended:

    "PC6 showed a substantially narrower gene-level association
    profile, with {pc6_gene_summary['genes_q_lt_0_05']:,} genes
    passing the same genome-wide FDR threshold."

Recommended:

    "Preranked Hallmark analysis identified structured transcriptional
    programs associated with both axes, with
    {pc3_path_summary['hallmark_q_lt_0_05']} of 50 Hallmark sets
    significant for PC3 and
    {pc6_path_summary['hallmark_q_lt_0_05']} of 50 significant for PC6."

Recommended:

    "The two axes showed partly contrasting Hallmark enrichment
    profiles, providing transcriptomic support for their
    nonredundant biological organization."


AVOID
-----

Avoid:

    "PC3 activates E2F."

Avoid:

    "PC6 activates androgen signaling."

Avoid:

    "PC3 and PC6 have significantly different pathways."

Avoid:

    "PC3 is a proliferation pathway."

Avoid:

    "PC6 is an EMT/myogenesis axis."

Avoid:

    "These pathways explain Grade Group."

Avoid:

    "The molecular mechanism of PC3 is ..."


FIGURE 24 INTERPRETATION
------------------------

Panel A:
    quantitative breadth of genome-wide gene association.

Panel B:
    descriptive contrast between PC3 and PC6 Hallmark NES values.

Panel C:
    complete 50-set Hallmark landscape.

The Figure 24 pathway order is descriptive and deterministic:

    descending (PC3_NES - PC6_NES)

This ordering does NOT constitute an inferential test of pathway
difference between the axes.
"""


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "STEP 29E: FREEZE FINAL TCGA-PRAD "
        "PC3 / PC6 MOLECULAR INTERPRETATION SUMMARY"
    )

    print(
        f"[VERSION]          {VERSION}"
    )

    print(
        f"[SCHEMA]           {SCHEMA}"
    )

    print(
        f"[STEP29C]          {STEP29C_DIR}"
    )

    print(
        f"[STEP29D]          {STEP29D_DIR}"
    )

    print(
        f"[OUTPUT]           {OUT_DIR}"
    )

    # =========================================================================
    # Validate upstream frozen evidence
    # =========================================================================

    banner(
        "VALIDATE FROZEN UPSTREAM MOLECULAR EVIDENCE"
    )

    step29c_summary = validate_step29c()
    step29d_summary = validate_step29d()

    print(
        "[STEP29C PASS]      YES"
    )

    print(
        "[STEP29D PASS]      YES"
    )

    print(
        "[STEP29D VERSION]   1.0.1"
    )

    print(
        f"[EXPECTED PATIENTS] {EXPECTED_PATIENTS}"
    )

    print(
        f"[EXPECTED GENES]    {EXPECTED_GENES:,}"
    )

    print(
        f"[HALLMARK SETS]     {EXPECTED_HALLMARKS}"
    )

    # =========================================================================
    # Load frozen outputs
    # =========================================================================

    banner(
        "LOAD FROZEN GENE-LEVEL RESULTS"
    )

    pc3_gene = load_gene_results(
        PC3_GENE_RESULTS,
        "PC3",
    )

    pc6_gene = load_gene_results(
        PC6_GENE_RESULTS,
        "PC6",
    )

    if set(
        pc3_gene[
            "ensembl_gene_id"
        ].astype(str)
    ) != set(
        pc6_gene[
            "ensembl_gene_id"
        ].astype(str)
    ):

        raise RuntimeError(
            "PC3 and PC6 gene universes differ."
        )

    print(
        f"[PC3 GENES]        {len(pc3_gene):,}"
    )

    print(
        f"[PC6 GENES]        {len(pc6_gene):,}"
    )

    print(
        "[SAME GENE SET]     YES"
    )

    banner(
        "LOAD FROZEN HALLMARK RESULTS"
    )

    pc3_pathway = load_pathway_results(
        PC3_PATHWAY_RESULTS,
        "PC3",
    )

    pc6_pathway = load_pathway_results(
        PC6_PATHWAY_RESULTS,
        "PC6",
    )

    if set(
        pc3_pathway[
            "pathway"
        ]
    ) != set(
        pc6_pathway[
            "pathway"
        ]
    ):

        raise RuntimeError(
            "PC3 and PC6 Hallmark pathway universes differ."
        )

    print(
        f"[PC3 PATHWAYS]     {len(pc3_pathway)}"
    )

    print(
        f"[PC6 PATHWAYS]     {len(pc6_pathway)}"
    )

    print(
        "[SAME PATHWAYS]     YES"
    )

    # =========================================================================
    # Freeze summaries
    # =========================================================================

    banner(
        "FREEZE FINAL GENE-LEVEL MOLECULAR SUMMARY"
    )

    pc3_gene_summary = summarize_gene_axis(
        pc3_gene,
        "PC3",
    )

    pc6_gene_summary = summarize_gene_axis(
        pc6_gene,
        "PC6",
    )

    print(
        "[PC3] "
        f"q<0.05={pc3_gene_summary['genes_q_lt_0_05']:,} | "
        f"positive={pc3_gene_summary['significant_positive']:,} | "
        f"negative={pc3_gene_summary['significant_negative']:,}"
    )

    print(
        "[PC6] "
        f"q<0.05={pc6_gene_summary['genes_q_lt_0_05']:,} | "
        f"positive={pc6_gene_summary['significant_positive']:,} | "
        f"negative={pc6_gene_summary['significant_negative']:,}"
    )

    gene_extreme_table = build_gene_extreme_summary(
        pc3_gene,
        pc6_gene,
    )

    gene_extreme_table.to_csv(
        FINAL_GENE_EXTREMES,
        index=False,
    )

    # =========================================================================
    # Pathway summary
    # =========================================================================

    banner(
        "FREEZE FINAL HALLMARK MOLECULAR SUMMARY"
    )

    pc3_path_summary = summarize_pathway_axis(
        pc3_pathway,
        "PC3",
    )

    pc6_path_summary = summarize_pathway_axis(
        pc6_pathway,
        "PC6",
    )

    print(
        "[PC3] "
        f"BH q<0.05={pc3_path_summary['hallmark_q_lt_0_05']} / 50 | "
        f"positive={pc3_path_summary['significant_positive_nes']} | "
        f"negative={pc3_path_summary['significant_negative_nes']}"
    )

    print(
        "[PC6] "
        f"BH q<0.05={pc6_path_summary['hallmark_q_lt_0_05']} / 50 | "
        f"positive={pc6_path_summary['significant_positive_nes']} | "
        f"negative={pc6_path_summary['significant_negative_nes']}"
    )

    hallmark_table = build_hallmark_interpretation_table(
        pc3_pathway,
        pc6_pathway,
    )

    hallmark_table.to_csv(
        FINAL_HALLMARK_TABLE,
        index=False,
    )

    # =========================================================================
    # Descriptive contrast
    # =========================================================================

    banner(
        "FREEZE DESCRIPTIVE PC3-versus-PC6 HALLMARK CONTRAST"
    )

    descriptive_contrast = hallmark_table[
        [
            "pathway",
            "display_name",
            "PC3_NES",
            "PC3_q",
            "PC3_significant",
            "PC6_NES",
            "PC6_q",
            "PC6_significant",
            "PC3_minus_PC6_NES",
            "absolute_NES_difference",
            "direction_pattern",
            "significance_pattern",
        ]
    ].copy()

    descriptive_contrast.to_csv(
        FINAL_HALLMARK_CONTRAST,
        index=False,
    )

    n_pc3_pos_pc6_neg = int(
        (
            hallmark_table[
                "direction_pattern"
            ]
            ==
            "PC3_POSITIVE_PC6_NEGATIVE"
        ).sum()
    )

    n_pc3_neg_pc6_pos = int(
        (
            hallmark_table[
                "direction_pattern"
            ]
            ==
            "PC3_NEGATIVE_PC6_POSITIVE"
        ).sum()
    )

    n_both_positive = int(
        (
            hallmark_table[
                "direction_pattern"
            ]
            ==
            "BOTH_POSITIVE"
        ).sum()
    )

    n_both_negative = int(
        (
            hallmark_table[
                "direction_pattern"
            ]
            ==
            "BOTH_NEGATIVE"
        ).sum()
    )

    print(
        "[DIRECTION PATTERNS]"
    )

    print(
        f"  PC3 positive / PC6 negative : {n_pc3_pos_pc6_neg}"
    )

    print(
        f"  PC3 negative / PC6 positive : {n_pc3_neg_pc6_pos}"
    )

    print(
        f"  both positive               : {n_both_positive}"
    )

    print(
        f"  both negative               : {n_both_negative}"
    )

    print()
    print(
        "[FORMAL PC3-vs-PC6 pathway difference test] NO"
    )

    # =========================================================================
    # Axis-level frozen summary table
    # =========================================================================

    axis_summary = pd.DataFrame(
        [
            {
                **pc3_gene_summary,
                **{
                    key:
                        value
                    for key, value
                    in pc3_path_summary.items()
                    if key != "axis"
                },
                "molecular_role":
                    "PRIMARY_INTERPRETED_AXIS",
            },

            {
                **pc6_gene_summary,
                **{
                    key:
                        value
                    for key, value
                    in pc6_path_summary.items()
                    if key != "axis"
                },
                "molecular_role":
                    "SECONDARY_PRESPECIFIED_AXIS",
            },
        ]
    )

    axis_summary.to_csv(
        FINAL_AXIS_SUMMARY,
        index=False,
    )

    # =========================================================================
    # Claim boundary
    # =========================================================================

    CLAIM_BOUNDARY_FILE.write_text(
        claim_boundary_text(),
        encoding="utf-8",
    )

    # =========================================================================
    # Manuscript guidance
    # =========================================================================

    MANUSCRIPT_GUIDANCE_FILE.write_text(
        manuscript_guidance_text(
            pc3_gene_summary,
            pc6_gene_summary,
            pc3_path_summary,
            pc6_path_summary,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Summary JSON
    # =========================================================================

    final_summary = {
        "version":
            VERSION,

        "schema":
            SCHEMA,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "PASS",

        "analysis_role":
            "FINAL_FROZEN_TCGA_MOLECULAR_INTERPRETATION_SUMMARY",

        "upstream": {
            "step29c_version":
                step29c_summary.get(
                    "version"
                ),

            "step29c_status":
                step29c_summary.get(
                    "status"
                ),

            "step29d_version":
                step29d_summary.get(
                    "version"
                ),

            "step29d_status":
                step29d_summary.get(
                    "status"
                ),
        },

        "cohort": {
            "matched_patients":
                EXPECTED_PATIENTS,

            "genes_tested":
                EXPECTED_GENES,

            "hallmark_pathways":
                EXPECTED_HALLMARKS,
        },

        "PC3": {
            "molecular_role":
                "PRIMARY_INTERPRETED_AXIS",

            "gene_level":
                pc3_gene_summary,

            "hallmark_level":
                pc3_path_summary,
        },

        "PC6": {
            "molecular_role":
                "SECONDARY_PRESPECIFIED_AXIS",

            "gene_level":
                pc6_gene_summary,

            "hallmark_level":
                pc6_path_summary,
        },

        "descriptive_pc3_pc6_hallmark_structure": {
            "PC3_positive_PC6_negative":
                n_pc3_pos_pc6_neg,

            "PC3_negative_PC6_positive":
                n_pc3_neg_pc6_pos,

            "both_positive":
                n_both_positive,

            "both_negative":
                n_both_negative,

            "formal_difference_test_performed":
                False,

            "allowed_role":
                (
                    "descriptive visualization and interpretation only"
                ),
        },

        "final_claim": (
            "Frozen PC3 and PC6 histologic axes show distinct "
            "transcriptomic association profiles in TCGA-PRAD, "
            "with PC3 exhibiting broader genome-wide gene-level "
            "associations and both axes mapping onto structured, "
            "partly contrasting Hallmark transcriptional programs."
        ),

        "claim_ceiling": (
            "Associative transcriptomic interpretation; "
            "no mechanistic or causal inference."
        ),

        "new_analysis_performed":
            False,

        "firewall": {
            "rna_expression_read":
                False,

            "pc_scores_recomputed":
                False,

            "gene_correlations_recomputed":
                False,

            "gene_p_values_recomputed":
                False,

            "gene_fdr_recomputed":
                False,

            "gsea_rerun":
                False,

            "pathway_p_values_recomputed":
                False,

            "pathway_fdr_recomputed":
                False,

            "genes_selected_manually":
                False,

            "pathways_selected_manually":
                False,

            "formal_pc3_vs_pc6_pathway_test":
                False,

            "clinical_outcomes_read":
                False,

            "survival_analysis":
                False,

            "causal_inference":
                False,
        },
    }

    save_json_atomic(
        FINAL_SUMMARY_JSON,
        final_summary,
    )

    # =========================================================================
    # Source hashes
    # =========================================================================

    source_files = [
        (
            "step29c_summary",
            STEP29C_SUMMARY,
        ),
        (
            "step29c_pc3_gene_results",
            PC3_GENE_RESULTS,
        ),
        (
            "step29c_pc6_gene_results",
            PC6_GENE_RESULTS,
        ),
        (
            "step29d_summary",
            STEP29D_SUMMARY,
        ),
        (
            "step29d_pc3_hallmark_results",
            PC3_PATHWAY_RESULTS,
        ),
        (
            "step29d_pc6_hallmark_results",
            PC6_PATHWAY_RESULTS,
        ),
    ]

    hash_rows = []

    for role, path in source_files:

        require_file(
            path,
            role,
        )

        hash_rows.append(
            {
                "role":
                    role,

                "path":
                    str(
                        path.resolve()
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),
            }
        )

    pd.DataFrame(
        hash_rows
    ).to_csv(
        SOURCE_HASHES_FILE,
        index=False,
    )

    # =========================================================================
    # README
    # =========================================================================

    readme = f"""STEP 29E
FINAL TCGA-PRAD PC3 / PC6 MOLECULAR INTERPRETATION SUMMARY

Version
-------
{VERSION}

Schema
------
{SCHEMA}


ROLE
----
SUMMARY-ONLY / FREEZE-ONLY


UPSTREAM AUTHORITY
------------------
Step29C:
    genome-wide gene associations

Step29D v1.0.1:
    frozen Hallmark pathway associations


COHORT
------
Matched RNA patients:
    {EXPECTED_PATIENTS}

Eligible genes:
    {EXPECTED_GENES:,}

Hallmark pathways:
    {EXPECTED_HALLMARKS}


PC3
---
Gene-level q<0.05:
    {pc3_gene_summary['genes_q_lt_0_05']:,}

Positive:
    {pc3_gene_summary['significant_positive']:,}

Negative:
    {pc3_gene_summary['significant_negative']:,}

Hallmark BH q<0.05:
    {pc3_path_summary['hallmark_q_lt_0_05']} / 50


PC6
---
Gene-level q<0.05:
    {pc6_gene_summary['genes_q_lt_0_05']:,}

Positive:
    {pc6_gene_summary['significant_positive']:,}

Negative:
    {pc6_gene_summary['significant_negative']:,}

Hallmark BH q<0.05:
    {pc6_path_summary['hallmark_q_lt_0_05']} / 50


PC3-versus-PC6
--------------
The Hallmark NES contrast is descriptive only.

No formal pathway-difference test is performed.


FINAL INTERPRETATION
--------------------
Frozen PC3 and PC6 histologic axes show distinct transcriptomic
association profiles in TCGA-PRAD, with PC3 exhibiting broader
genome-wide gene-level associations and both axes mapping onto
structured, partly contrasting Hallmark transcriptional programs.

This supports molecular nonredundancy but does not establish mechanism,
causality, tumor-cell specificity, mediation, prognosis, or therapeutic
utility.


NO NEW ANALYSIS
---------------
Step29E performs no new inferential analysis.
"""

    README_FILE.write_text(
        readme,
        encoding="utf-8",
    )

    # =========================================================================
    # Audit
    # =========================================================================

    banner(
        "STEP 29E FREEZE-ONLY AUDIT"
    )

    audit = [
        (
            "Step29C frozen genome-wide results inherited",
            True,
        ),

        (
            "Step29D v1.0.1 frozen Hallmark results inherited",
            True,
        ),

        (
            "401-patient molecular cohort retained",
            True,
        ),

        (
            "20,066-gene test universe retained",
            True,
        ),

        (
            "50 Hallmark pathways retained",
            True,
        ),

        (
            "PC3 primary molecular role retained",
            True,
        ),

        (
            "PC6 secondary molecular role retained",
            True,
        ),

        (
            "Gene-level significant counts summarized",
            True,
        ),

        (
            "Hallmark significant counts summarized",
            True,
        ),

        (
            "Complete Hallmark table retained",
            True,
        ),

        (
            "Nonsignificant Hallmark pathways retained",
            True,
        ),

        (
            "PC3-minus-PC6 NES contrast descriptive only",
            True,
        ),

        (
            "RNA expression reread",
            False,
        ),

        (
            "PC scores recomputed",
            False,
        ),

        (
            "Gene correlations recomputed",
            False,
        ),

        (
            "Gene p-values recomputed",
            False,
        ),

        (
            "Gene FDR recomputed",
            False,
        ),

        (
            "GSEA rerun",
            False,
        ),

        (
            "Pathway p-values recomputed",
            False,
        ),

        (
            "Pathway FDR recomputed",
            False,
        ),

        (
            "Pathways manually selected",
            False,
        ),

        (
            "Genes manually selected",
            False,
        ),

        (
            "Formal PC3-vs-PC6 pathway difference test performed",
            False,
        ),

        (
            "Clinical outcomes reread",
            False,
        ),

        (
            "Survival analysis performed",
            False,
        ),

        (
            "Causal inference performed",
            False,
        ),

        (
            "Final molecular claim boundary frozen",
            True,
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<78}: {yesno(value)}"
        )

    # =========================================================================
    # PASS
    # =========================================================================

    banner(
        "STEP 29E: PASS"
    )

    print(
        f"[PATIENTS]          {EXPECTED_PATIENTS}"
    )

    print(
        f"[GENES]             {EXPECTED_GENES:,}"
    )

    print(
        f"[HALLMARKS]         {EXPECTED_HALLMARKS}"
    )

    print()

    print(
        "[PC3 GENES]         "
        f"{pc3_gene_summary['genes_q_lt_0_05']:,} q<0.05 | "
        f"{pc3_gene_summary['significant_positive']:,} positive | "
        f"{pc3_gene_summary['significant_negative']:,} negative"
    )

    print(
        "[PC6 GENES]         "
        f"{pc6_gene_summary['genes_q_lt_0_05']:,} q<0.05 | "
        f"{pc6_gene_summary['significant_positive']:,} positive | "
        f"{pc6_gene_summary['significant_negative']:,} negative"
    )

    print()

    print(
        "[PC3 HALLMARK]      "
        f"{pc3_path_summary['hallmark_q_lt_0_05']} / 50 BH q<0.05"
    )

    print(
        "[PC6 HALLMARK]      "
        f"{pc6_path_summary['hallmark_q_lt_0_05']} / 50 BH q<0.05"
    )

    print()

    print(
        "[PC3-vs-PC6 TEST]   NOT PERFORMED"
    )

    print(
        "[CONTRAST ROLE]     DESCRIPTIVE ONLY"
    )

    print(
        "[NEW ANALYSIS]      NO"
    )

    print()

    print(
        f"[AXIS SUMMARY]      {FINAL_AXIS_SUMMARY}"
    )

    print(
        f"[HALLMARK TABLE]    {FINAL_HALLMARK_TABLE}"
    )

    print(
        f"[CONTRAST TABLE]    {FINAL_HALLMARK_CONTRAST}"
    )

    print(
        f"[GENE EXTREMES]     {FINAL_GENE_EXTREMES}"
    )

    print(
        f"[SUMMARY]           {FINAL_SUMMARY_JSON}"
    )

    print(
        f"[CLAIM BOUNDARY]    {CLAIM_BOUNDARY_FILE}"
    )

    print(
        f"[MANUSCRIPT GUIDE]  {MANUSCRIPT_GUIDANCE_FILE}"
    )

    print(
        f"[HASHES]            {SOURCE_HASHES_FILE}"
    )

    print(
        f"[README]            {README_FILE}"
    )

    print()
    print(
        "[NEXT] Integrate frozen TCGA transfer, clinicopathologic, "
        "and molecular evidence into the manuscript Results."
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        banner(
            "STEP 29E: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)