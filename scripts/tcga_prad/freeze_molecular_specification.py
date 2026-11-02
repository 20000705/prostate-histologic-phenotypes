#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 29A
Freeze the TCGA-PRAD PC3 / PC6 molecular association specification.

VERSION: 1.0.1

PURPOSE
-------
Freeze the molecular-analysis design before any RNA expression values
are inspected in relation to PC3 or PC6.

This step defines:

1. The frozen PC3 / PC6 patient-level representation authority.
2. The molecular roles of PC3 and PC6.
3. RNA expression unit and transformation.
4. Gene identity and detectability rules.
5. Genome-wide gene-level association statistic.
6. Axis-specific genome-wide multiplicity correction.
7. Bootstrap scope.
8. Hallmark pathway analysis framework.
9. Claim boundaries.

IMPORTANT
---------
This is a specification-only step.

It MUST NOT:
- read RNA expression values;
- calculate PC3/gene or PC6/gene associations;
- calculate gene-level p-values or FDR;
- inspect pathway results;
- select genes or pathways from outcomes;
- refit PCA;
- change PC3 / PC6 directions.

VERSION 1.0.1 CHANGE
--------------------
The genome-wide primary inferential procedure is explicitly defined as:

    Spearman rho
    + two-sided asymptotic/exact scipy p-value as applicable
    + BH-FDR separately for PC3 and PC6.

10,000-resample patient bootstrap is NOT applied genome-wide to every
eligible gene.

Bootstrap is reserved only for explicitly defined downstream stability
summaries / leading-association characterization, if performed.

Bootstrap results MUST NOT be used to:
- define genome-wide significance;
- select genes;
- change the eligible gene universe;
- modify FDR families;
- promote otherwise nonsignificant genes.

This clarification is made before any PC3/PC6-to-RNA association results
are calculated.
"""

from __future__ import annotations

import hashlib
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


import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.1"
SCHEMA = (
    "29a-tcga-prad-pc3-pc6-molecular-association-"
    "specification-schema-1.0.1"
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = REPO_ROOT

RESULTS_ROOT = RESULTS_ROOT

STEP27G_DIR = (
    RESULTS_ROOT
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
)

PATIENT_MANIFEST = (
    STEP27G_DIR
    / "frozen_tcga_prad_patient_representation_manifest.csv"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "29a_tcga_prad_pc3_pc6_molecular_association_specification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AXIS_SPEC_PATH = (
    OUTPUT_DIR
    / "tcga_prad_pc3_pc6_molecular_axis_specification.csv"
)

GENE_SPEC_PATH = (
    OUTPUT_DIR
    / "tcga_prad_gene_level_molecular_analysis_specification.csv"
)

PATHWAY_SPEC_PATH = (
    OUTPUT_DIR
    / "tcga_prad_hallmark_pathway_analysis_specification.csv"
)

SPEC_JSON_PATH = (
    OUTPUT_DIR
    / "tcga_prad_pc3_pc6_molecular_association_specification.json"
)

CLAIM_CEILING_PATH = (
    OUTPUT_DIR
    / "tcga_prad_pc3_pc6_molecular_claim_ceiling.txt"
)

HASH_PATH = OUTPUT_DIR / "source_hashes.csv"

README_PATH = OUTPUT_DIR / "README.txt"


# =============================================================================
# CONSTANTS
# =============================================================================

EXPECTED_PATIENTS = 403

RNA_UNIT = "TPM"
RNA_COLUMN = "tpm_unstranded"
RNA_TRANSFORM = "log2(TPM + 1)"

GENE_ID_AUTHORITY = "Ensembl gene ID"
STRIP_ENSEMBL_VERSION = True
PRESERVE_PAR_Y = True

GENE_FILTER_TPM_THRESHOLD = 1.0
GENE_FILTER_PATIENT_FRACTION = 0.20

GENE_STATISTIC = "Spearman rho"
GENE_TEST_SIDEDNESS = "two-sided"

GENE_FDR_METHOD = "Benjamini-Hochberg"
GENE_FDR_THRESHOLD = 0.05
GENE_FDR_FAMILY = "separate genome-wide family per axis"

# -------------------------------------------------------------------------
# v1.0.1 bootstrap clarification
# -------------------------------------------------------------------------

GENOMEWIDE_BOOTSTRAP = False

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_UNIT = "patient"
BOOTSTRAP_SCOPE = (
    "downstream prespecified stability/leading-association "
    "characterization only; not genome-wide primary inference"
)

BOOTSTRAP_SELECTION_ALLOWED = False
BOOTSTRAP_SIGNIFICANCE_DEFINITION_ALLOWED = False

PATHWAY_COLLECTION = "MSigDB Hallmark"
PATHWAY_RANK_METRIC = "signed Spearman rho"
PATHWAY_METHOD = "rank-based enrichment"
PATHWAY_FDR_METHOD = "Benjamini-Hochberg"
PATHWAY_FDR_THRESHOLD = 0.05
PATHWAY_FDR_FAMILY = "separate Hallmark family per axis"


# =============================================================================
# HELPERS
# =============================================================================

def banner(text: str, width: int = 118) -> None:
    print("\n" + "=" * width)
    print(text)
    print("=" * width)


def yesno(value: bool) -> str:
    return "YES" if value else "NO"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def resolve_patient_column(df: pd.DataFrame) -> str:
    candidates = [
        "patient_id",
        "tcga_patient_id",
        "case_submitter_id",
        "submitter_id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not identify patient ID column in frozen patient manifest. "
        f"Columns: {df.columns.tolist()}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "STEP 29A: FREEZE TCGA-PRAD PC3 / PC6 "
        "MOLECULAR ASSOCIATION SPECIFICATION"
    )

    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")

    # -------------------------------------------------------------------------
    # Validate frozen representation authority
    # -------------------------------------------------------------------------

    require_file(PATIENT_MANIFEST)

    patient_df = pd.read_csv(PATIENT_MANIFEST)

    patient_col = resolve_patient_column(patient_df)

    patient_ids = (
        patient_df[patient_col]
        .astype(str)
        .str.strip()
    )

    if patient_ids.duplicated().any():
        dup = sorted(
            patient_ids[
                patient_ids.duplicated(keep=False)
            ].unique()
        )

        raise ValueError(
            "Frozen patient manifest contains duplicate patient IDs:\n"
            + "\n".join(dup[:20])
        )

    n_patients = len(patient_df)
    n_unique = patient_ids.nunique()

    if n_patients != EXPECTED_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_PATIENTS} frozen patients, "
            f"found {n_patients}."
        )

    if n_unique != EXPECTED_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_PATIENTS} unique patients, "
            f"found {n_unique}."
        )

    # -------------------------------------------------------------------------
    # Axis specification
    # -------------------------------------------------------------------------

    axis_spec = pd.DataFrame(
        [
            {
                "axis": "PC3",
                "molecular_role": "PRIMARY",
                "representation_level": "patient",
                "representation_authority":
                    "Step27G frozen patient representation",
                "score_definition":
                    "median of slide-level oriented_PC3 medians",
                "direction_reselection_allowed": False,
                "axis_reselection_allowed": False,
            },
            {
                "axis": "PC6",
                "molecular_role": "SECONDARY_PRESPECIFIED",
                "representation_level": "patient",
                "representation_authority":
                    "Step27G frozen patient representation",
                "score_definition":
                    "median of slide-level oriented_PC6 medians",
                "direction_reselection_allowed": False,
                "axis_reselection_allowed": False,
            },
        ]
    )

    axis_spec.to_csv(AXIS_SPEC_PATH, index=False)

    # -------------------------------------------------------------------------
    # Gene-level specification
    # -------------------------------------------------------------------------

    gene_spec = pd.DataFrame(
        [
            {
                "analysis_level": "gene",
                "rna_unit": RNA_UNIT,
                "rna_column": RNA_COLUMN,
                "rna_transform": RNA_TRANSFORM,

                "gene_identifier_authority":
                    GENE_ID_AUTHORITY,

                "strip_ensembl_version":
                    STRIP_ENSEMBL_VERSION,

                "preserve_PAR_Y_distinction":
                    PRESERVE_PAR_Y,

                "detectability_rule":
                    "TPM >= 1 in >= 20% of matched patients",

                "detectability_tpm_threshold":
                    GENE_FILTER_TPM_THRESHOLD,

                "detectability_patient_fraction":
                    GENE_FILTER_PATIENT_FRACTION,

                "same_gene_universe_for_pc3_pc6":
                    True,

                "association_statistic":
                    GENE_STATISTIC,

                "test_sidedness":
                    GENE_TEST_SIDEDNESS,

                "genomewide_primary_inference":
                    (
                        "Spearman rho + two-sided p-value + "
                        "axis-specific BH-FDR"
                    ),

                "multiple_testing_method":
                    GENE_FDR_METHOD,

                "multiple_testing_family":
                    GENE_FDR_FAMILY,

                "significance_q_threshold":
                    GENE_FDR_THRESHOLD,

                "genomewide_bootstrap":
                    GENOMEWIDE_BOOTSTRAP,

                "bootstrap_resamples_if_used":
                    BOOTSTRAP_RESAMPLES,

                "bootstrap_unit":
                    BOOTSTRAP_UNIT,

                "bootstrap_scope":
                    BOOTSTRAP_SCOPE,

                "bootstrap_may_select_genes":
                    BOOTSTRAP_SELECTION_ALLOWED,

                "bootstrap_may_define_significance":
                    BOOTSTRAP_SIGNIFICANCE_DEFINITION_ALLOWED,
            }
        ]
    )

    gene_spec.to_csv(GENE_SPEC_PATH, index=False)

    # -------------------------------------------------------------------------
    # Hallmark pathway specification
    # -------------------------------------------------------------------------

    pathway_spec = pd.DataFrame(
        [
            {
                "analysis_level": "pathway",
                "gene_set_collection":
                    PATHWAY_COLLECTION,

                "gene_ranking_metric":
                    PATHWAY_RANK_METRIC,

                "ranking_source":
                    (
                        "complete eligible-gene genome-wide "
                        "Spearman ranking"
                    ),

                "ranking_direction":
                    "signed",

                "analysis_method":
                    PATHWAY_METHOD,

                "multiple_testing_method":
                    PATHWAY_FDR_METHOD,

                "multiple_testing_family":
                    PATHWAY_FDR_FAMILY,

                "significance_q_threshold":
                    PATHWAY_FDR_THRESHOLD,

                "separate_pc3_pc6_analysis":
                    True,

                "pathway_selection_from_results_allowed":
                    False,

                "gene_selection_before_ranking_allowed":
                    False,
            }
        ]
    )

    pathway_spec.to_csv(PATHWAY_SPEC_PATH, index=False)

    # -------------------------------------------------------------------------
    # Claim ceiling
    # -------------------------------------------------------------------------

    claim_text = """TCGA-PRAD PC3 / PC6 molecular claim ceiling
================================================

This analysis evaluates whether discovery-defined, frozen histologic
axes show transcriptomic associations in TCGA-PRAD.

Permitted claims
----------------
1. PC3 and/or PC6 may be described as showing transcriptomic
   associations if supported by the prespecified genome-wide
   analysis.

2. Individual genes may be described as associated with a frozen
   histologic axis when they satisfy the prespecified genome-wide
   BH-FDR criterion.

3. Hallmark pathways may be described as enriched along the signed
   genome-wide molecular ranking when supported by the prespecified
   Hallmark analysis.

4. PC3 and PC6 molecular association patterns may be compared
   descriptively because both use the same frozen eligible-gene
   universe and the same statistical framework.

Not permitted
-------------
1. Molecular associations do not establish causality.

2. Molecular associations do not establish that PC3 or PC6 directly
   measures a specific biological pathway.

3. Molecular results may not be used to redefine PC3 or PC6.

4. Genes or pathways may not be selected post hoc and presented as
   though they were prespecified confirmatory targets.

5. Bootstrap stability summaries, if performed, may not be used to
   redefine genome-wide significance or select genes.

6. Nonsignificant genome-wide genes may not be promoted on the basis
   of bootstrap behavior alone.

Primary genome-wide inference
-----------------------------
For every eligible gene and separately for PC3 and PC6:

    Spearman rho
    two-sided p-value
    Benjamini-Hochberg FDR

The BH correction is performed separately for the PC3 genome-wide
family and the PC6 genome-wide family.

Genome-wide significance is defined by q < 0.05.

Bootstrap clarification
-----------------------
A 10,000-resample patient bootstrap is NOT part of the primary
genome-wide inferential procedure for every eligible gene.

If bootstrap analysis is subsequently performed, it is restricted to
an explicitly defined downstream stability or leading-association
characterization analysis.

Bootstrap results cannot:
- select genes;
- alter the eligible gene universe;
- redefine statistical significance;
- alter FDR families;
- rescue genes failing the frozen genome-wide FDR criterion.

This clarification was frozen before calculation of PC3/PC6-to-RNA
association results.
"""

    CLAIM_CEILING_PATH.write_text(
        claim_text,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # JSON specification
    # -------------------------------------------------------------------------

    specification = {
        "version": VERSION,
        "schema": SCHEMA,

        "representation": {
            "source":
                str(PATIENT_MANIFEST),

            "n_frozen_patients":
                n_patients,

            "n_unique_patients":
                n_unique,

            "pc3_role":
                "PRIMARY",

            "pc6_role":
                "SECONDARY_PRESPECIFIED",
        },

        "rna": {
            "unit":
                RNA_UNIT,

            "column":
                RNA_COLUMN,

            "transformation":
                RNA_TRANSFORM,
        },

        "gene_identity": {
            "authority":
                GENE_ID_AUTHORITY,

            "strip_ensembl_version":
                STRIP_ENSEMBL_VERSION,

            "preserve_PAR_Y":
                PRESERVE_PAR_Y,
        },

        "gene_filter": {
            "rule":
                "TPM >= 1 in >= 20% of matched patients",

            "tpm_threshold":
                GENE_FILTER_TPM_THRESHOLD,

            "patient_fraction":
                GENE_FILTER_PATIENT_FRACTION,

            "same_universe_for_pc3_pc6":
                True,
        },

        "gene_association": {
            "statistic":
                GENE_STATISTIC,

            "sidedness":
                GENE_TEST_SIDEDNESS,

            "primary_inference":
                (
                    "Spearman rho + two-sided p-value + "
                    "axis-specific BH-FDR"
                ),

            "fdr_method":
                GENE_FDR_METHOD,

            "fdr_family":
                GENE_FDR_FAMILY,

            "q_threshold":
                GENE_FDR_THRESHOLD,
        },

        "bootstrap": {
            "genomewide_per_gene_bootstrap":
                False,

            "resamples_if_used":
                BOOTSTRAP_RESAMPLES,

            "resampling_unit":
                BOOTSTRAP_UNIT,

            "scope":
                BOOTSTRAP_SCOPE,

            "may_select_genes":
                False,

            "may_define_genomewide_significance":
                False,

            "may_change_fdr_family":
                False,
        },

        "pathway_analysis": {
            "collection":
                PATHWAY_COLLECTION,

            "ranking_metric":
                PATHWAY_RANK_METRIC,

            "method":
                PATHWAY_METHOD,

            "fdr_method":
                PATHWAY_FDR_METHOD,

            "fdr_family":
                PATHWAY_FDR_FAMILY,

            "q_threshold":
                PATHWAY_FDR_THRESHOLD,
        },

        "prohibited_operations": [
            "PCA refit",
            "axis reselection",
            "PC3 sign tuning",
            "PC6 sign tuning",
            "RNA-driven gene filtering",
            "gene selection from association outcomes",
            "pathway selection from enrichment outcomes",
            "bootstrap-driven gene selection",
            "bootstrap-defined genome-wide significance",
        ],

        "revision_from_v1_0_0": {
            "reason":
                (
                    "Clarify bootstrap scope before any "
                    "PC3/PC6-to-RNA association calculation."
                ),

            "gene_universe_changed":
                False,

            "rna_processing_changed":
                False,

            "gene_statistic_changed":
                False,

            "fdr_procedure_changed":
                False,

            "pathway_collection_changed":
                False,

            "bootstrap_scope_changed":
                True,
        },
    }

    with SPEC_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            specification,
            f,
            indent=2,
            sort_keys=True,
        )

    # -------------------------------------------------------------------------
    # Source hashes
    # -------------------------------------------------------------------------

    hash_rows = []

    for source in [
        PATIENT_MANIFEST,
    ]:
        hash_rows.append(
            {
                "source_path": str(source),
                "sha256": sha256_file(source),
            }
        )

    pd.DataFrame(hash_rows).to_csv(
        HASH_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------

    readme = f"""STEP 29A
TCGA-PRAD PC3 / PC6 Molecular Association Specification

Version
-------
{VERSION}

Schema
------
{SCHEMA}

Frozen patient cohort
---------------------
{n_patients} patients.

PC3
---
Primary molecular axis.

PC6
---
Secondary prespecified molecular axis.

RNA
---
Unit:
    TPM

Authority column:
    tpm_unstranded

Transformation:
    log2(TPM + 1)

Gene eligibility
----------------
TPM >= 1 in >=20% of matched RNA patients.

The same eligible-gene universe must be used for PC3 and PC6.

Genome-wide gene analysis
-------------------------
For every eligible gene:

    Spearman rho
    two-sided p-value

BH-FDR is applied separately to:

    PC3 genome-wide gene family
    PC6 genome-wide gene family

Significance threshold:

    q < 0.05

Bootstrap
---------
Genome-wide per-gene bootstrap:
    NO

10,000 patient resamples are reserved only for explicitly defined
downstream stability / leading-association characterization if such
an analysis is performed.

Bootstrap must not be used for gene selection or genome-wide
significance determination.

Hallmark analysis
-----------------
Collection:
    MSigDB Hallmark

Ranking:
    signed genome-wide Spearman rho

PC3 and PC6 are analyzed separately.

Important
---------
This specification was frozen before PC3/PC6-to-RNA association
results were calculated.

Version 1.0.1 changes only the bootstrap scope. RNA preprocessing,
gene filtering, gene statistic, FDR families, pathway collection,
and axis definitions are unchanged.
"""

    README_PATH.write_text(
        readme,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console audit
    # -------------------------------------------------------------------------

    banner(
        "FROZEN MOLECULAR REPRESENTATION AUTHORITY"
    )

    print(f"Frozen patients        : {n_patients}")
    print("PC3                    : PRIMARY molecular axis")
    print(
        "PC6                    : "
        "SECONDARY prespecified molecular axis"
    )
    print("RNA expression values  : NOT READ")
    print(
        "Clinicopathologic result values used to "
        "design molecular tests: NO"
    )

    banner("STEP 29A FREEZE-ONLY AUDIT")

    audit_rows = [
        (
            "403-patient representation authority inherited",
            True,
        ),
        (
            "PC3 primary molecular role frozen",
            True,
        ),
        (
            "PC6 secondary molecular role frozen",
            True,
        ),
        (
            "RNA unit frozen as TPM",
            True,
        ),
        (
            "RNA transformation frozen",
            True,
        ),
        (
            "Gene identifier rule frozen",
            True,
        ),
        (
            "Gene detectability filter frozen",
            True,
        ),
        (
            "Same eligible genes required for PC3/PC6",
            True,
        ),
        (
            "Spearman gene-level statistic frozen",
            True,
        ),
        (
            "Separate genome-wide PC3/PC6 FDR families frozen",
            True,
        ),
        (
            "Genome-wide per-gene bootstrap required",
            False,
        ),
        (
            "Bootstrap excluded from genome-wide significance",
            True,
        ),
        (
            "Bootstrap prohibited from gene selection",
            True,
        ),
        (
            "Hallmark-only pathway collection frozen",
            True,
        ),
        (
            "Signed-rho pathway ranking frozen",
            True,
        ),
        (
            "Pathway FDR families frozen",
            True,
        ),
        (
            "RNA expression values read",
            False,
        ),
        (
            "Gene associations calculated",
            False,
        ),
        (
            "Gene p-values calculated",
            False,
        ),
        (
            "Gene FDR calculated",
            False,
        ),
        (
            "Pathway outcomes read",
            False,
        ),
        (
            "Genes selected from outcomes",
            False,
        ),
        (
            "Pathways selected from outcomes",
            False,
        ),
        (
            "PCA refit",
            False,
        ),
        (
            "Axis reselection",
            False,
        ),
        (
            "PC3/PC6 sign tuning",
            False,
        ),
        (
            "Claim ceiling frozen",
            True,
        ),
    ]

    for label, value in audit_rows:
        print(
            f"{label:<70} : {yesno(value)}"
        )

    banner("STEP 29A: PASS")

    print(f"[PATIENTS]          {n_patients}")
    print("[PC3]               PRIMARY molecular axis")
    print(
        "[PC6]               "
        "SECONDARY prespecified molecular axis"
    )
    print(f"[RNA UNIT]          {RNA_UNIT}")
    print(f"[TRANSFORM]         {RNA_TRANSFORM}")
    print(
        "[GENE FILTER]       "
        "TPM >= 1 in >= 20% patients"
    )
    print(
        "[GENE TEST]         "
        "Spearman rho, two-sided"
    )
    print(
        "[GENE FDR]          "
        "BH separately for PC3 / PC6"
    )
    print(
        f"[GENE q]            < {GENE_FDR_THRESHOLD}"
    )
    print(
        "[GENOME BOOTSTRAP]  NO"
    )
    print(
        "[STABILITY BOOT]    "
        "10,000 patient resamples if prespecified downstream"
    )
    print(
        f"[PATHWAYS]          {PATHWAY_COLLECTION} only"
    )
    print(
        f"[PATHWAY RANK]      {PATHWAY_RANK_METRIC}"
    )
    print(
        "[PATHWAY FDR]       "
        "BH separately for PC3 / PC6"
    )
    print(
        f"[PATHWAY q]         < {PATHWAY_FDR_THRESHOLD}"
    )
    print("[RNA OUTCOMES]      NOT READ")

    print()
    print(f"[AXIS SPEC]          {AXIS_SPEC_PATH}")
    print(f"[GENE SPEC]          {GENE_SPEC_PATH}")
    print(f"[PATHWAY SPEC]       {PATHWAY_SPEC_PATH}")
    print(f"[SPEC]               {SPEC_JSON_PATH}")
    print(f"[CLAIM CEILING]      {CLAIM_CEILING_PATH}")
    print(f"[HASHES]             {HASH_PATH}")
    print(f"[README]             {README_PATH}")

    print()
    print(
        "[REVISION] Version 1.0.1 changes bootstrap scope only; "
        "RNA processing, gene filtering, gene statistic, FDR "
        "families, pathway collection, and axis definitions "
        "are unchanged."
    )

    print()
    print(
        "[NEXT] Step29C may join the frozen PC3/PC6 patient "
        "scores to the Step29B RNA matrix and execute the "
        "frozen genome-wide gene associations."
    )


if __name__ == "__main__":
    main()