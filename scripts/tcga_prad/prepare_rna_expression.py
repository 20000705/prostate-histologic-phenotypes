#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
29b_audit_and_materialize_frozen_tcga_prad_rna_analysis_matrix.py

STEP 29B
========
AUDIT AND MATERIALIZE FROZEN TCGA-PRAD RNA ANALYSIS MATRIX

VERSION
-------
1.0.0

SCHEMA
------
29b-frozen-tcga-prad-rna-analysis-matrix-schema-1.0

ROLE
----
RNA PROVENANCE / MATCHING / QC / MATERIALIZATION ONLY.

This step inherits the previously established TCGA-PRAD RNA workflow and
materializes the exact RNA matrix that will be used for the frozen
PC3 / PC6 genome-wide molecular analysis.

THIS STEP DOES
--------------
1. Inherit the previously selected one-row-per-patient RNA metadata.
2. Resolve each selected metadata row to the already-downloaded GDC
   augmented STAR-count file.
3. Require the frozen TPM authority:
       tpm_unstranded
4. Retain ALL Ensembl genes.
5. Strip Ensembl version suffixes while preserving _PAR_Y.
6. Validate gene identity/order across all selected RNA profiles.
7. Intersect the RNA cohort with the frozen 403-patient WSI cohort.
8. Materialize matched-frozen patient × gene TPM matrix.
9. Materialize log2(TPM + 1).
10. Apply the Step29A frozen detectability filter:
       TPM >= 1 in >=20% of matched RNA patients.
11. Write explicit patient order, gene order, provenance, QC and hashes.

THIS STEP DOES NOT
------------------
- read patient_oriented_PC3 numerical values,
- read patient_oriented_PC6 numerical values,
- calculate Spearman correlations,
- calculate gene-level p-values,
- calculate gene-level FDR,
- perform pathway analysis,
- reselect RNA samples,
- re-run duplicate-aliquot selection,
- use clinicopathologic results,
- select genes using PC3 or PC6 results.

IMPORTANT PROVENANCE RULE
-------------------------
RNA sample selection authority is inherited from:

    tcga_prad_rnaseq_patient_level_selected_metadata.csv

This script does NOT decide again between 01A / 01B or other replicates.

GENE UNIVERSE
-------------
ALL Ensembl genes quantified by the GDC STAR-count files are retained before
the frozen detectability filter.

No protein_coding-only restriction is introduced here because Step29A did
not freeze such a restriction.

FROZEN GENE FILTER
------------------
A gene is eligible if:

    TPM >= 1

in at least:

    20%

of the final matched frozen RNA cohort.

The exact same eligible gene set must subsequently be used for PC3 and PC6.

MATRIX CONVENTION
-----------------
Rows:
    frozen TCGA patients with selected RNA

Columns:
    stripped Ensembl gene IDs

Values:
    TPM
    log2(TPM + 1)

Explicit patient and gene order files are written.

Downstream Step29C MUST inherit these orders and MUST NOT independently
reconstruct them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
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

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.0"

SCHEMA = (
    "29b-frozen-tcga-prad-rna-analysis-matrix-schema-1.0"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


DEFAULT_BASE = CONFIG_TCGA_ROOT

DEFAULT_RNA_ROOT = (
    DEFAULT_BASE
    / "tcga_prad_rnaseq"
)

DEFAULT_SELECTED_METADATA = (
    DEFAULT_RNA_ROOT
    / "metadata"
    / "tcga_prad_rnaseq_patient_level_selected_metadata.csv"
)

DEFAULT_STEP29A_SPEC = (
    RESULTS
    / "29a_tcga_prad_pc3_pc6_molecular_association_specification"
    / "tcga_prad_pc3_pc6_molecular_association_specification.json"
)

DEFAULT_STEP27G_PATIENT_MANIFEST = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "frozen_tcga_prad_patient_representation_manifest.csv"
)

DEFAULT_OUT_DIR = (
    RESULTS
    / "29b_frozen_tcga_prad_rna_analysis_matrix"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_FROZEN_PATIENTS = 403

TPM_COLUMN = "tpm_unstranded"

MIN_TPM = 1.0
MIN_PATIENT_FRACTION = 0.20


# =============================================================================
# METADATA COLUMN CANDIDATES
# =============================================================================

PATIENT_COLUMN_CANDIDATES = [
    "patient_id",
    "case_submitter_id",
    "case_submitter_submitter_id",
    "cases.submitter_id",
    "case_barcode",
    "submitter_id",
]

FILE_UUID_COLUMN_CANDIDATES = [
    "file_id",
    "file_uuid",
    "gdc_file_uuid",
    "id",
]

FILE_NAME_COLUMN_CANDIDATES = [
    "file_name",
    "file_name_api",
    "file_name_manifest",
    "gdc_file_name",
    "filename",
]

LOCAL_PATH_COLUMN_CANDIDATES = [
    "local_file_path",
    "resolved_file_path",
    "file_path",
    "path",
    "downloaded_file_path",
]

SAMPLE_COLUMN_CANDIDATES = [
    "sample_submitter_id",
    "sample_id",
    "sample_barcode",
]

SAMPLE_TYPE_COLUMN_CANDIDATES = [
    "sample_type",
    "cases.samples.sample_type",
]


# =============================================================================
# FILE PATTERNS
# =============================================================================

STAR_FILE_PATTERNS = (
    "*.rna_seq.augmented_star_gene_counts.tsv",
    "*.rna_seq.augmented_star_gene_counts.tsv.gz",
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def banner(
    title: str,
    width: int = 120,
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


def require_dir(
    path: Path,
    label: str,
) -> None:

    if not path.is_dir():

        raise FileNotFoundError(
            f"{label} missing: {path}"
        )


def normalize_text(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    try:

        if pd.isna(
            value
        ):
            return None

    except Exception:

        pass

    text = str(
        value
    ).strip()

    if (
        not text
        or text.lower()
        in {
            "nan",
            "none",
            "null",
            "na",
            "n/a",
            "unknown",
        }
    ):

        return None

    return text


def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for candidate in candidates:

        if candidate in df.columns:

            return candidate

    return None


def normalize_patient_id(
    value: Any,
) -> Optional[str]:

    text = normalize_text(
        value
    )

    if text is None:
        return None

    text = text.upper()

    match = re.search(
        r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",
        text,
    )

    if match is None:
        return None

    return match.group(
        1
    )


def strip_ensembl_version(
    gene_id: Any,
) -> Optional[str]:

    """
    Preserve _PAR_Y.

    Examples:
        ENSG00000141510.18
            -> ENSG00000141510

        ENSG00000002586.20_PAR_Y
            -> ENSG00000002586_PAR_Y
    """

    text = normalize_text(
        gene_id
    )

    if text is None:
        return None

    match = re.match(
        r"^(ENSG\d+)\.\d+(_PAR_Y)?$",
        text,
    )

    if match:

        base = match.group(
            1
        )

        suffix = (
            match.group(
                2
            )
            or ""
        )

        return (
            base
            + suffix
        )

    # Also accept already stripped Ensembl IDs.
    if re.match(
        r"^ENSG\d+(_PAR_Y)?$",
        text,
    ):

        return text

    return None


def open_text(
    path: Path,
):

    if path.suffix.lower() == ".gz":

        return gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
        )

    return path.open(
        mode="rt",
        encoding="utf-8",
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
        path.name
        + ".tmp"
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


def save_npy_atomic(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        + ".tmp"
    )

    with tmp.open(
        "wb",
    ) as handle:

        np.save(
            handle,
            array,
            allow_pickle=False,
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
            "analysis_role"
        )
        !=
        "PRE_RNA_OUTCOME_TCGA_MOLECULAR_ASSOCIATION_FREEZE"
    ):

        raise RuntimeError(
            "Unexpected Step29A analysis role."
        )

    expression = spec.get(
        "expression_authority",
        {},
    )

    if (
        expression.get(
            "unit"
        )
        != "TPM"
    ):

        raise RuntimeError(
            "Step29A RNA unit is not TPM."
        )

    gene_filter = spec.get(
        "gene_filtering",
        {},
    )

    observed_tpm = float(
        gene_filter.get(
            "minimum_tpm",
            -1,
        )
    )

    observed_fraction = float(
        gene_filter.get(
            "minimum_patient_fraction",
            -1,
        )
    )

    if not np.isclose(
        observed_tpm,
        MIN_TPM,
    ):

        raise RuntimeError(
            "Step29A minimum TPM changed."
        )

    if not np.isclose(
        observed_fraction,
        MIN_PATIENT_FRACTION,
    ):

        raise RuntimeError(
            "Step29A patient-fraction filter changed."
        )

    return spec


# =============================================================================
# LOAD FROZEN PATIENT IDS ONLY
# =============================================================================

def load_frozen_patient_ids_only(
    path: Path,
) -> list[str]:

    """
    Explicit firewall:
    only patient_id is loaded from Step27G.

    PC3 and PC6 numerical columns are never read.
    """

    require_file(
        path,
        "Step27G frozen patient manifest",
    )

    header = pd.read_csv(
        path,
        nrows=0,
    )

    if (
        "patient_id"
        not in header.columns
    ):

        raise RuntimeError(
            "Step27G manifest lacks patient_id."
        )

    df = pd.read_csv(
        path,
        usecols=[
            "patient_id",
        ],
    )

    ids = []

    for value in df[
        "patient_id"
    ]:

        patient_id = normalize_patient_id(
            value
        )

        if patient_id is None:

            raise RuntimeError(
                "Invalid Step27G patient ID: "
                f"{value!r}"
            )

        ids.append(
            patient_id
        )

    if len(
        ids
    ) != EXPECTED_FROZEN_PATIENTS:

        raise RuntimeError(
            "Frozen patient count changed: "
            f"{len(ids)} != "
            f"{EXPECTED_FROZEN_PATIENTS}"
        )

    if len(
        set(
            ids
        )
    ) != EXPECTED_FROZEN_PATIENTS:

        raise RuntimeError(
            "Frozen Step27G patient IDs are not unique."
        )

    return sorted(
        ids
    )


# =============================================================================
# SELECTED RNA METADATA
# =============================================================================

def resolve_metadata_columns(
    metadata: pd.DataFrame,
) -> dict[str, Optional[str]]:

    patient_col = first_existing_column(
        metadata,
        PATIENT_COLUMN_CANDIDATES,
    )

    file_uuid_col = first_existing_column(
        metadata,
        FILE_UUID_COLUMN_CANDIDATES,
    )

    file_name_col = first_existing_column(
        metadata,
        FILE_NAME_COLUMN_CANDIDATES,
    )

    local_path_col = first_existing_column(
        metadata,
        LOCAL_PATH_COLUMN_CANDIDATES,
    )

    sample_col = first_existing_column(
        metadata,
        SAMPLE_COLUMN_CANDIDATES,
    )

    sample_type_col = first_existing_column(
        metadata,
        SAMPLE_TYPE_COLUMN_CANDIDATES,
    )

    if patient_col is None:

        raise RuntimeError(
            "Selected RNA metadata lacks a recognizable "
            "patient ID column."
        )

    if (
        file_uuid_col is None
        and file_name_col is None
        and local_path_col is None
    ):

        raise RuntimeError(
            "Selected RNA metadata lacks file UUID/name/path authority."
        )

    return {
        "patient": patient_col,
        "file_uuid": file_uuid_col,
        "file_name": file_name_col,
        "local_path": local_path_col,
        "sample": sample_col,
        "sample_type": sample_type_col,
    }


def read_selected_metadata(
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Optional[str]],
]:

    require_file(
        path,
        "Selected RNA metadata",
    )

    df = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    )

    columns = resolve_metadata_columns(
        df
    )

    df = df.copy()

    df[
        "patient_id_canonical"
    ] = df[
        columns[
            "patient"
        ]
    ].map(
        normalize_patient_id
    )

    if df[
        "patient_id_canonical"
    ].isna().any():

        bad = df.loc[
            df[
                "patient_id_canonical"
            ].isna()
        ].head(
            20
        )

        raise RuntimeError(
            "Selected metadata contains invalid patient IDs:\n"
            + bad.to_string(
                index=False
            )
        )

    if df[
        "patient_id_canonical"
    ].duplicated().any():

        duplicates = (
            df.loc[
                df[
                    "patient_id_canonical"
                ].duplicated(
                    keep=False
                ),
                "patient_id_canonical",
            ]
            .value_counts()
        )

        raise RuntimeError(
            "Selected RNA metadata is not one row per patient.\n"
            + duplicates.head(
                20
            ).to_string()
        )

    return (
        df,
        columns,
    )


# =============================================================================
# LOCAL RNA FILE INDEX
# =============================================================================

def index_local_rna_files(
    root: Path,
) -> pd.DataFrame:

    require_dir(
        root,
        "RNA root",
    )

    rows = []

    seen = set()

    for pattern in STAR_FILE_PATTERNS:

        for path in root.rglob(
            pattern
        ):

            if not path.is_file():
                continue

            resolved = path.resolve()

            if resolved in seen:
                continue

            seen.add(
                resolved
            )

            rows.append({
                "indexed_path": str(
                    resolved
                ),

                "indexed_file_name": (
                    resolved.name
                ),

                "indexed_parent_name": (
                    resolved.parent.name
                ),

                "indexed_size_bytes": int(
                    resolved.stat().st_size
                ),
            })

    index = pd.DataFrame(
        rows
    )

    if index.empty:

        raise RuntimeError(
            f"No GDC STAR-count files found under {root}"
        )

    return (
        index
        .sort_values(
            "indexed_path"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# STAR FILE VALIDATION
# =============================================================================

def is_star_tpm_file(
    path: Path,
) -> bool:

    required = {
        "gene_id",
        "gene_name",
        "gene_type",
        TPM_COLUMN,
    }

    try:

        with open_text(
            path
        ) as handle:

            for _ in range(
                100
            ):

                line = handle.readline()

                if line == "":
                    break

                line = line.strip()

                if not line:
                    continue

                if line.startswith(
                    "#"
                ):
                    continue

                header = set(
                    line.split(
                        "\t"
                    )
                )

                return required.issubset(
                    header
                )

    except Exception:

        return False

    return False


# =============================================================================
# RESOLVE SELECTED ROW -> FILE
# =============================================================================

def resolve_selected_file(
    row: pd.Series,
    columns: dict[str, Optional[str]],
    rna_root: Path,
    file_index: pd.DataFrame,
) -> tuple[
    Optional[Path],
    str,
    list[str],
]:

    candidates: list[Path] = []
    attempted: list[str] = []

    local_path = (
        normalize_text(
            row[
                columns[
                    "local_path"
                ]
            ]
        )
        if columns[
            "local_path"
        ] is not None
        else None
    )

    file_uuid = (
        normalize_text(
            row[
                columns[
                    "file_uuid"
                ]
            ]
        )
        if columns[
            "file_uuid"
        ] is not None
        else None
    )

    file_name = (
        normalize_text(
            row[
                columns[
                    "file_name"
                ]
            ]
        )
        if columns[
            "file_name"
        ] is not None
        else None
    )

    # ------------------------------------------------------------------
    # 1. Explicit local path.
    # ------------------------------------------------------------------

    if local_path is not None:

        p = Path(
            local_path
        ).expanduser()

        if not p.is_absolute():

            p = (
                rna_root
                / p
            )

        candidates.append(
            p
        )

        attempted.append(
            f"explicit_path:{p}"
        )

        # Historical manifest compatibility:
        # Try locating the same file beneath the configured RNA root.
        # This avoids relying on the original server mount location.
        if p.is_absolute():
            alternative = rna_root / p.name

            if alternative != p:
                candidates.append(alternative)
                attempted.append(
                    f"configured_rna_root_path:{alternative}"
                )

    # ------------------------------------------------------------------
    # 2. UUID/name expected path.
    # ------------------------------------------------------------------

    if (
        file_uuid is not None
        and file_name is not None
    ):

        p = (
            rna_root
            / file_uuid
            / file_name
        )

        candidates.append(
            p
        )

        attempted.append(
            f"uuid_and_name:{p}"
        )

    # ------------------------------------------------------------------
    # 3. UUID directory index.
    # ------------------------------------------------------------------

    if file_uuid is not None:

        matches = file_index.loc[
            file_index[
                "indexed_parent_name"
            ].astype(str)
            == file_uuid
        ]

        attempted.append(
            f"indexed_uuid_matches:{len(matches)}"
        )

        for path_text in matches[
            "indexed_path"
        ]:

            candidates.append(
                Path(
                    path_text
                )
            )

    # ------------------------------------------------------------------
    # 4. Exact filename index.
    # ------------------------------------------------------------------

    if file_name is not None:

        matches = file_index.loc[
            file_index[
                "indexed_file_name"
            ].astype(str)
            == file_name
        ]

        attempted.append(
            f"indexed_name_matches:{len(matches)}"
        )

        for path_text in matches[
            "indexed_path"
        ]:

            candidates.append(
                Path(
                    path_text
                )
            )

    # ------------------------------------------------------------------
    # Deduplicate candidates.
    # ------------------------------------------------------------------

    unique_candidates = []

    seen = set()

    for candidate in candidates:

        candidate = candidate.expanduser()

        try:

            candidate = candidate.resolve()

        except Exception:

            pass

        key = str(
            candidate
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_candidates.append(
            candidate
        )

    existing = [
        p
        for p in unique_candidates
        if p.exists()
        and p.is_file()
    ]

    probable = [
        p
        for p in existing
        if is_star_tpm_file(
            p
        )
    ]

    if len(
        probable
    ) == 1:

        return (
            probable[
                0
            ],
            "RESOLVED",
            attempted,
        )

    if len(
        probable
    ) > 1:

        if file_name is not None:

            exact_name = [
                p
                for p in probable
                if p.name
                == file_name
            ]

            if len(
                exact_name
            ) == 1:

                return (
                    exact_name[
                        0
                    ],
                    "RESOLVED_EXACT_FILENAME",
                    attempted,
                )

        raise RuntimeError(
            "Selected metadata resolves to multiple valid "
            "STAR-count files. Sample selection must NOT be "
            "re-decided in Step29B.\n"
            + "\n".join(
                str(
                    p
                )
                for p in probable
            )
        )

    if existing:

        return (
            None,
            "FOUND_FILES_BUT_NO_REQUIRED_TPM_SCHEMA",
            attempted,
        )

    return (
        None,
        "FILE_NOT_FOUND",
        attempted,
    )


# =============================================================================
# READ ALL-GENE TPM
# =============================================================================

def read_all_gene_tpm(
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype={
            "gene_id": "string",
            "gene_name": "string",
            "gene_type": "string",
        },
        low_memory=False,
    )

    required = {
        "gene_id",
        "gene_name",
        "gene_type",
        TPM_COLUMN,
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{path}: missing required columns {missing}"
        )

    n_rows_original = int(
        len(
            df
        )
    )

    # Remove STAR summary rows.
    df = df.loc[
        df[
            "gene_id"
        ].notna()
        &
        ~df[
            "gene_id"
        ]
        .astype(str)
        .str.startswith(
            "N_"
        )
    ].copy()

    n_gene_rows = int(
        len(
            df
        )
    )

    # ------------------------------------------------------------------
    # No gene_type filtering.
    # ------------------------------------------------------------------

    df[
        "ensembl_gene_id_versioned"
    ] = (
        df[
            "gene_id"
        ]
        .astype(str)
    )

    df[
        "ensembl_gene_id"
    ] = df[
        "ensembl_gene_id_versioned"
    ].map(
        strip_ensembl_version
    )

    invalid_ensembl = df[
        "ensembl_gene_id"
    ].isna()

    n_non_ensembl = int(
        invalid_ensembl.sum()
    )

    df = df.loc[
        ~invalid_ensembl
    ].copy()

    # ------------------------------------------------------------------
    # Numeric TPM validation.
    # ------------------------------------------------------------------

    df[
        TPM_COLUMN
    ] = pd.to_numeric(
        df[
            TPM_COLUMN
        ],
        errors="coerce",
    )

    if df[
        TPM_COLUMN
    ].isna().any():

        raise RuntimeError(
            f"{path}: nonnumeric/missing TPM values detected."
        )

    tpm = df[
        TPM_COLUMN
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        tpm
    ).all():

        raise RuntimeError(
            f"{path}: TPM contains NaN/Inf."
        )

    if (
        tpm < 0
    ).any():

        raise RuntimeError(
            f"{path}: TPM contains negative values."
        )

    # ------------------------------------------------------------------
    # Duplicate stripped IDs are forbidden.
    # ------------------------------------------------------------------

    duplicated = df[
        "ensembl_gene_id"
    ].duplicated(
        keep=False
    )

    if duplicated.any():

        duplicate_table = (
            df.loc[
                duplicated,
                [
                    "ensembl_gene_id",
                    "ensembl_gene_id_versioned",
                    "gene_name",
                    "gene_type",
                ],
            ]
            .sort_values(
                "ensembl_gene_id"
            )
        )

        raise RuntimeError(
            "Duplicate Ensembl IDs after version stripping in "
            f"{path}:\n"
            + duplicate_table.head(
                30
            ).to_string(
                index=False
            )
        )

    df[
        "gene_name"
    ] = (
        df[
            "gene_name"
        ]
        .fillna(
            ""
        )
        .astype(str)
    )

    df[
        "gene_type"
    ] = (
        df[
            "gene_type"
        ]
        .fillna(
            ""
        )
        .astype(str)
    )

    df = (
        df[
            [
                "ensembl_gene_id",
                "ensembl_gene_id_versioned",
                "gene_name",
                "gene_type",
                TPM_COLUMN,
            ]
        ]
        .sort_values(
            "ensembl_gene_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    tpm = df[
        TPM_COLUMN
    ].to_numpy(
        dtype=np.float64
    )

    qc = {
        "n_rows_original": (
            n_rows_original
        ),

        "n_gene_rows_before_ensembl_filter": (
            n_gene_rows
        ),

        "n_non_ensembl_rows_removed": (
            n_non_ensembl
        ),

        "n_ensembl_genes": int(
            len(
                df
            )
        ),

        "n_gene_types": int(
            df[
                "gene_type"
            ].nunique()
        ),

        "n_protein_coding": int(
            (
                df[
                    "gene_type"
                ]
                == "protein_coding"
            ).sum()
        ),

        "detected_genes_tpm_gt_0": int(
            (
                tpm > 0
            ).sum()
        ),

        "detected_genes_tpm_ge_1": int(
            (
                tpm >= 1
            ).sum()
        ),

        "fraction_zero_tpm": float(
            (
                tpm == 0
            ).mean()
        ),

        "sum_tpm": float(
            tpm.sum()
        ),

        "max_tpm": float(
            tpm.max()
        ),
    }

    return (
        df,
        qc,
    )


# =============================================================================
# MATERIALIZE RNA MATRICES
# =============================================================================

def materialize_matrices(
    selected_frozen: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:

    if selected_frozen.empty:

        raise RuntimeError(
            "No frozen patients with selected RNA."
        )

    selected_frozen = (
        selected_frozen
        .sort_values(
            "patient_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    n_patients = int(
        len(
            selected_frozen
        )
    )

    patient_ids = (
        selected_frozen[
            "patient_id"
        ]
        .astype(str)
        .tolist()
    )

    banner(
        "READ SELECTED RNA AND MATERIALIZE ALL-GENE TPM MATRIX"
    )

    tpm_rows = []

    qc_rows = []

    reference_gene_ids = None

    reference_annotation = None

    start_time = time.time()

    for i, row in selected_frozen.iterrows():

        patient_id = str(
            row[
                "patient_id"
            ]
        )

        path = Path(
            row[
                "resolved_file_path"
            ]
        )

        expression, qc = read_all_gene_tpm(
            path
        )

        current_gene_ids = (
            expression[
                "ensembl_gene_id"
            ]
            .astype(str)
            .to_numpy()
        )

        if reference_gene_ids is None:

            reference_gene_ids = (
                current_gene_ids.copy()
            )

            reference_annotation = (
                expression[
                    [
                        "ensembl_gene_id",
                        "ensembl_gene_id_versioned",
                        "gene_name",
                        "gene_type",
                    ]
                ]
                .copy()
            )

            print(
                "[REFERENCE ALL-GENE UNIVERSE] "
                f"{len(reference_gene_ids):,}"
            )

        else:

            if (
                current_gene_ids.shape
                != reference_gene_ids.shape
            ):

                raise RuntimeError(
                    f"{patient_id}: gene count differs "
                    "from reference."
                )

            if not np.array_equal(
                current_gene_ids,
                reference_gene_ids,
            ):

                reference_set = set(
                    reference_gene_ids.tolist()
                )

                current_set = set(
                    current_gene_ids.tolist()
                )

                missing = sorted(
                    reference_set
                    - current_set
                )

                extra = sorted(
                    current_set
                    - reference_set
                )

                raise RuntimeError(
                    f"{patient_id}: gene identity/order differs.\n"
                    f"Missing={len(missing)} "
                    f"Extra={len(extra)}\n"
                    f"First missing={missing[:10]}\n"
                    f"First extra={extra[:10]}"
                )

        tpm_rows.append(
            expression[
                TPM_COLUMN
            ].to_numpy(
                dtype=np.float32
            )
        )

        qc_rows.append({
            "patient_id": (
                patient_id
            ),

            "selected_sample_submitter_id": (
                row.get(
                    "sample_submitter_id",
                    None,
                )
            ),

            "selected_file_uuid": (
                row.get(
                    "file_uuid",
                    None,
                )
            ),

            "resolved_file_path": str(
                path.resolve()
            ),

            "file_size_bytes": int(
                path.stat().st_size
            ),

            **qc,
        })

        done = i + 1

        if (
            done == 1
            or done % 25 == 0
            or done == n_patients
        ):

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                done / elapsed
                if elapsed > 0
                else 0
            )

            eta = (
                (
                    n_patients
                    - done
                )
                / rate
                if rate > 0
                else np.nan
            )

            print(
                f"[READ] {done:>4}/{n_patients} "
                f"| elapsed={elapsed / 60:.1f} min "
                f"| ETA={eta / 60:.1f} min"
            )

    if (
        reference_gene_ids is None
        or reference_annotation is None
    ):

        raise RuntimeError(
            "Reference gene universe was not initialized."
        )

    # ------------------------------------------------------------------
    # Stack matrices.
    # ------------------------------------------------------------------

    tpm = np.vstack(
        tpm_rows
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        tpm
    ).all():

        raise RuntimeError(
            "Final TPM matrix contains NaN/Inf."
        )

    if (
        tpm < 0
    ).any():

        raise RuntimeError(
            "Final TPM matrix contains negative values."
        )

    log2_tpm = np.log2(
        tpm
        + np.float32(
            1.0
        )
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------------
    # Frozen detectability filter.
    # ------------------------------------------------------------------

    min_patient_count = int(
        math.ceil(
            MIN_PATIENT_FRACTION
            * n_patients
        )
    )

    detectable_count = (
        tpm
        >= np.float32(
            MIN_TPM
        )
    ).sum(
        axis=0
    )

    detectable_fraction = (
        detectable_count
        / n_patients
    )

    eligible_mask = (
        detectable_count
        >= min_patient_count
    )

    n_genes_all = int(
        len(
            reference_gene_ids
        )
    )

    n_genes_eligible = int(
        eligible_mask.sum()
    )

    if n_genes_eligible == 0:

        raise RuntimeError(
            "Frozen gene filter retained zero genes."
        )

    eligible_tpm = tpm[
        :,
        eligible_mask
    ]

    eligible_log2_tpm = log2_tpm[
        :,
        eligible_mask
    ]

    # ------------------------------------------------------------------
    # Annotation.
    # ------------------------------------------------------------------

    gene_table = (
        reference_annotation
        .copy()
        .reset_index(
            drop=True
        )
    )

    gene_table.insert(
        0,
        "all_gene_column_index",
        np.arange(
            n_genes_all,
            dtype=int,
        ),
    )

    gene_table[
        "patients_tpm_ge_1"
    ] = detectable_count.astype(
        int
    )

    gene_table[
        "fraction_patients_tpm_ge_1"
    ] = detectable_fraction

    gene_table[
        "passes_frozen_detectability_filter"
    ] = eligible_mask

    eligible_gene_table = (
        gene_table.loc[
            eligible_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    eligible_gene_table.insert(
        0,
        "eligible_gene_column_index",
        np.arange(
            n_genes_eligible,
            dtype=int,
        ),
    )

    # ------------------------------------------------------------------
    # Output paths.
    # ------------------------------------------------------------------

    patient_order_path = (
        out_dir
        / "frozen_matched_rna_patient_order.csv"
    )

    all_gene_path = (
        out_dir
        / "all_ensembl_gene_universe.csv"
    )

    eligible_gene_path = (
        out_dir
        / "eligible_ensembl_gene_universe.csv"
    )

    patient_qc_path = (
        out_dir
        / "frozen_matched_rna_patient_qc.csv"
    )

    all_tpm_path = (
        out_dir
        / "frozen_matched_patient_by_all_gene_tpm.npy"
    )

    all_log2_path = (
        out_dir
        / "frozen_matched_patient_by_all_gene_log2_tpm_plus1.npy"
    )

    eligible_tpm_path = (
        out_dir
        / "frozen_matched_patient_by_eligible_gene_tpm.npy"
    )

    eligible_log2_path = (
        out_dir
        / "frozen_matched_patient_by_eligible_gene_log2_tpm_plus1.npy"
    )

    eligible_mask_path = (
        out_dir
        / "frozen_eligible_gene_mask.npy"
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    pd.DataFrame({
        "patient_row_index": np.arange(
            n_patients,
            dtype=int,
        ),

        "patient_id": (
            patient_ids
        ),
    }).to_csv(
        patient_order_path,
        index=False,
    )

    gene_table.to_csv(
        all_gene_path,
        index=False,
    )

    eligible_gene_table.to_csv(
        eligible_gene_path,
        index=False,
    )

    pd.DataFrame(
        qc_rows
    ).sort_values(
        "patient_id"
    ).to_csv(
        patient_qc_path,
        index=False,
    )

    save_npy_atomic(
        all_tpm_path,
        tpm,
    )

    save_npy_atomic(
        all_log2_path,
        log2_tpm,
    )

    save_npy_atomic(
        eligible_tpm_path,
        eligible_tpm,
    )

    save_npy_atomic(
        eligible_log2_path,
        eligible_log2_tpm,
    )

    save_npy_atomic(
        eligible_mask_path,
        eligible_mask.astype(
            np.bool_
        ),
    )

    return {
        "n_patients": (
            n_patients
        ),

        "n_genes_all": (
            n_genes_all
        ),

        "n_genes_eligible": (
            n_genes_eligible
        ),

        "minimum_patient_count": (
            min_patient_count
        ),

        "patient_order_path": (
            patient_order_path
        ),

        "all_gene_path": (
            all_gene_path
        ),

        "eligible_gene_path": (
            eligible_gene_path
        ),

        "patient_qc_path": (
            patient_qc_path
        ),

        "all_tpm_path": (
            all_tpm_path
        ),

        "all_log2_path": (
            all_log2_path
        ),

        "eligible_tpm_path": (
            eligible_tpm_path
        ),

        "eligible_log2_path": (
            eligible_log2_path
        ),

        "eligible_mask_path": (
            eligible_mask_path
        ),
    }


# =============================================================================
# CLAIM BOUNDARY
# =============================================================================

def claim_boundary_text() -> str:

    return """STEP 29B — CLAIM BOUNDARY

ROLE
----
RNA PROVENANCE / MATCHING / QC / MATERIALIZATION ONLY.


STEP29B MAY ESTABLISH
---------------------
1. Which frozen WSI patients have a previously selected TCGA RNA profile.
2. Which GDC STAR-count file corresponds to each selected RNA patient.
3. The available all-Ensembl-gene TPM expression matrix.
4. The frozen TPM >=1 in >=20% detectability gene universe.
5. RNA matching and missingness.
6. RNA QC and expression provenance.


STEP29B DOES NOT ESTABLISH
--------------------------
1. PC3-associated genes.
2. PC6-associated genes.
3. Gene-level statistical significance.
4. Molecular pathways associated with PC3 or PC6.
5. Molecular mechanism.
6. Transcriptomic mediation of clinicopathologic associations.
7. Causal biology.


PROHIBITED
----------
1. Reading PC3 / PC6 numerical values.
2. Re-selecting RNA aliquots.
3. Re-running 01A / 01B selection.
4. Selecting genes by PC association.
5. Applying protein-coding-only filtering.
6. Changing the frozen TPM >=1 in >=20% rule.
"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def execute(
    rna_root: Path,
    selected_metadata_path: Path,
    step29a_spec_path: Path,
    frozen_patient_manifest_path: Path,
    out_dir: Path,
) -> None:

    banner(
        "STEP 29B: AUDIT AND MATERIALIZE FROZEN "
        "TCGA-PRAD RNA ANALYSIS MATRIX"
    )

    print(
        f"[VERSION]          {VERSION}"
    )

    print(
        f"[SCHEMA]           {SCHEMA}"
    )

    print(
        f"[RNA ROOT]         {rna_root}"
    )

    print(
        f"[SELECTED META]    {selected_metadata_path}"
    )

    print(
        f"[OUTPUT]           {out_dir}"
    )

    # ------------------------------------------------------------------
    # Validate Step29A.
    # ------------------------------------------------------------------

    validate_step29a(
        step29a_spec_path
    )

    frozen_ids = load_frozen_patient_ids_only(
        frozen_patient_manifest_path
    )

    frozen_set = set(
        frozen_ids
    )

    # ------------------------------------------------------------------
    # Read selected RNA authority.
    # ------------------------------------------------------------------

    banner(
        "1. INHERIT SELECTED PATIENT-LEVEL RNA METADATA"
    )

    (
        selected,
        columns,
    ) = read_selected_metadata(
        selected_metadata_path
    )

    print(
        f"[SELECTED ROWS]      {len(selected)}"
    )

    print(
        "[UNIQUE PATIENTS]    "
        f"{selected['patient_id_canonical'].nunique()}"
    )

    print(
        f"[PATIENT COLUMN]     {columns['patient']}"
    )

    print(
        f"[FILE UUID COLUMN]   {columns['file_uuid']}"
    )

    print(
        f"[FILE NAME COLUMN]   {columns['file_name']}"
    )

    print(
        f"[LOCAL PATH COLUMN]  {columns['local_path']}"
    )

    print(
        f"[SAMPLE COLUMN]      {columns['sample']}"
    )

    print(
        f"[SAMPLE TYPE COLUMN] {columns['sample_type']}"
    )

    # ------------------------------------------------------------------
    # Index files.
    # ------------------------------------------------------------------

    banner(
        "2. INDEX LOCAL GDC STAR-COUNT FILES"
    )

    file_index = index_local_rna_files(
        rna_root
    )

    print(
        f"[INDEXED FILES]      {len(file_index)}"
    )

    # ------------------------------------------------------------------
    # Resolve all selected metadata rows.
    # ------------------------------------------------------------------

    banner(
        "3. RESOLVE SELECTED RNA AUTHORITY TO LOCAL FILES"
    )

    resolution_rows = []

    for row_number, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        patient_id = row[
            "patient_id_canonical"
        ]

        (
            resolved_path,
            status,
            attempted,
        ) = resolve_selected_file(
            row=row,
            columns=columns,
            rna_root=rna_root,
            file_index=file_index,
        )

        resolution_rows.append({
            "patient_id": (
                patient_id
            ),

            "in_frozen_403": (
                patient_id
                in frozen_set
            ),

            "resolution_status": (
                status
            ),

            "resolved_file_path": (
                str(
                    resolved_path
                )
                if resolved_path
                is not None
                else None
            ),

            "selected_file_uuid": (
                normalize_text(
                    row[
                        columns[
                            "file_uuid"
                        ]
                    ]
                )
                if columns[
                    "file_uuid"
                ] is not None
                else None
            ),

            "selected_file_name": (
                normalize_text(
                    row[
                        columns[
                            "file_name"
                        ]
                    ]
                )
                if columns[
                    "file_name"
                ] is not None
                else None
            ),

            "sample_submitter_id": (
                normalize_text(
                    row[
                        columns[
                            "sample"
                        ]
                    ]
                )
                if columns[
                    "sample"
                ] is not None
                else None
            ),

            "sample_type": (
                normalize_text(
                    row[
                        columns[
                            "sample_type"
                        ]
                    ]
                )
                if columns[
                    "sample_type"
                ] is not None
                else None
            ),

            "attempted_resolution": (
                " | ".join(
                    attempted
                )
            ),
        })

        if (
            row_number == 1
            or row_number % 50 == 0
            or row_number == len(
                selected
            )
        ):

            print(
                f"[RESOLVE] "
                f"{row_number:>4}/{len(selected)}"
            )

    resolution = pd.DataFrame(
        resolution_rows
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolution_path = (
        out_dir
        / "selected_rna_file_resolution_audit.csv"
    )

    resolution.to_csv(
        resolution_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Frozen cohort matching.
    # ------------------------------------------------------------------

    banner(
        "4. MATCH SELECTED RNA TO FROZEN 403-PATIENT COHORT"
    )

    selected_frozen = resolution.loc[
        resolution[
            "in_frozen_403"
        ]
    ].copy()

    unresolved_frozen = selected_frozen.loc[
        ~selected_frozen[
            "resolution_status"
        ].astype(str)
        .str.startswith(
            "RESOLVED"
        )
    ].copy()

    if not unresolved_frozen.empty:

        raise RuntimeError(
            "One or more frozen patients have a selected RNA "
            "record that could not be resolved.\n"
            + unresolved_frozen[
                [
                    "patient_id",
                    "resolution_status",
                ]
            ].to_string(
                index=False
            )
        )

    selected_frozen = selected_frozen.loc[
        selected_frozen[
            "resolution_status"
        ].astype(str)
        .str.startswith(
            "RESOLVED"
        )
    ].copy()

    if selected_frozen[
        "patient_id"
    ].duplicated().any():

        raise RuntimeError(
            "Frozen RNA resolution is not one row per patient."
        )

    matched_ids = set(
        selected_frozen[
            "patient_id"
        ].astype(str)
    )

    missing_frozen_ids = sorted(
        frozen_set
        - matched_ids
    )

    rna_only_ids = sorted(
        set(
            resolution.loc[
                resolution[
                    "resolution_status"
                ].astype(str)
                .str.startswith(
                    "RESOLVED"
                ),
                "patient_id",
            ].astype(str)
        )
        - frozen_set
    )

    frozen_matching_path = (
        out_dir
        / "frozen_403_rna_matching_audit.csv"
    )

    frozen_match_rows = []

    resolution_by_patient = (
        resolution
        .set_index(
            "patient_id"
        )
    )

    for patient_id in frozen_ids:

        if patient_id in resolution_by_patient.index:

            row = resolution_by_patient.loc[
                patient_id
            ]

            frozen_match_rows.append({
                "patient_id": (
                    patient_id
                ),

                "rna_selected": bool(
                    str(
                        row[
                            "resolution_status"
                        ]
                    ).startswith(
                        "RESOLVED"
                    )
                ),

                "resolution_status": (
                    row[
                        "resolution_status"
                    ]
                ),

                "sample_submitter_id": (
                    row[
                        "sample_submitter_id"
                    ]
                ),

                "resolved_file_path": (
                    row[
                        "resolved_file_path"
                    ]
                ),
            })

        else:

            frozen_match_rows.append({
                "patient_id": (
                    patient_id
                ),

                "rna_selected": (
                    False
                ),

                "resolution_status": (
                    "NO_SELECTED_RNA_PROFILE"
                ),

                "sample_submitter_id": (
                    None
                ),

                "resolved_file_path": (
                    None
                ),
            })

    pd.DataFrame(
        frozen_match_rows
    ).to_csv(
        frozen_matching_path,
        index=False,
    )

    print(
        f"[FROZEN PATIENTS]    {len(frozen_ids)}"
    )

    print(
        f"[MATCHED RNA]        {len(matched_ids)}"
    )

    print(
        f"[WITHOUT RNA]        {len(missing_frozen_ids)}"
    )

    print(
        f"[RNA-ONLY OTHERS]    {len(rna_only_ids)}"
    )

    if missing_frozen_ids:

        print()
        print(
            "[FROZEN PATIENTS WITHOUT SELECTED RNA]"
        )

        for patient_id in missing_frozen_ids:

            print(
                f"  {patient_id}"
            )

    # ------------------------------------------------------------------
    # Materialize matrices.
    # ------------------------------------------------------------------

    matrix_info = materialize_matrices(
        selected_frozen=(
            selected_frozen
        ),

        out_dir=(
            out_dir
        ),
    )

    # ------------------------------------------------------------------
    # Claim boundary.
    # ------------------------------------------------------------------

    claim_path = (
        out_dir
        / "tcga_prad_rna_materialization_claim_boundary.txt"
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

    hash_rows = []

    for label, path in [
        (
            "step29a_specification",
            step29a_spec_path,
        ),

        (
            "step27g_patient_manifest",
            frozen_patient_manifest_path,
        ),

        (
            "selected_rna_metadata",
            selected_metadata_path,
        ),
    ]:

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
    # Summary.
    # ------------------------------------------------------------------

    summary_path = (
        out_dir
        / "tcga_prad_frozen_rna_analysis_matrix_summary.json"
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
            "RNA_PROVENANCE_QC_AND_MATERIALIZATION_ONLY"
        ),

        "sample_selection_authority": {
            "source": str(
                selected_metadata_path.resolve()
            ),

            "selection_reperformed_in_step29b": False,

            "duplicate_aliquot_selection_reperformed": False,

            "expression_used_to_select_rna_profile": False,
        },

        "frozen_wsi_cohort": {
            "patients": (
                len(
                    frozen_ids
                )
            ),

            "matched_rna_patients": (
                len(
                    matched_ids
                )
            ),

            "missing_rna_patients": (
                len(
                    missing_frozen_ids
                )
            ),

            "missing_patient_ids": (
                missing_frozen_ids
            ),
        },

        "expression": {
            "authority_column": (
                TPM_COLUMN
            ),

            "unit": (
                "TPM"
            ),

            "transformation": (
                "log2(TPM + 1)"
            ),

            "gene_universe_before_filter": (
                "all quantified Ensembl genes"
            ),

            "protein_coding_only_filter_applied": (
                False
            ),

            "genes_before_filter": (
                matrix_info[
                    "n_genes_all"
                ]
            ),

            "genes_after_filter": (
                matrix_info[
                    "n_genes_eligible"
                ]
            ),

            "filter": {
                "minimum_tpm": (
                    MIN_TPM
                ),

                "minimum_patient_fraction": (
                    MIN_PATIENT_FRACTION
                ),

                "minimum_patient_count": (
                    matrix_info[
                        "minimum_patient_count"
                    ]
                ),
            },
        },

        "firewall": {
            "step27g_patient_id_column_read": (
                True
            ),

            "step27g_pc3_numerical_values_read": (
                False
            ),

            "step27g_pc6_numerical_values_read": (
                False
            ),

            "pc3_rna_association_calculated": (
                False
            ),

            "pc6_rna_association_calculated": (
                False
            ),

            "gene_p_values_calculated": (
                False
            ),

            "gene_fdr_calculated": (
                False
            ),

            "pathway_analysis_performed": (
                False
            ),

            "clinical_results_used": (
                False
            ),

            "genes_selected_by_association_results": (
                False
            ),
        },

        "outputs": {
            "file_resolution": str(
                resolution_path.resolve()
            ),

            "frozen_matching": str(
                frozen_matching_path.resolve()
            ),

            "patient_order": str(
                matrix_info[
                    "patient_order_path"
                ].resolve()
            ),

            "all_gene_universe": str(
                matrix_info[
                    "all_gene_path"
                ].resolve()
            ),

            "eligible_gene_universe": str(
                matrix_info[
                    "eligible_gene_path"
                ].resolve()
            ),

            "patient_qc": str(
                matrix_info[
                    "patient_qc_path"
                ].resolve()
            ),

            "all_gene_tpm": str(
                matrix_info[
                    "all_tpm_path"
                ].resolve()
            ),

            "all_gene_log2_tpm": str(
                matrix_info[
                    "all_log2_path"
                ].resolve()
            ),

            "eligible_gene_tpm": str(
                matrix_info[
                    "eligible_tpm_path"
                ].resolve()
            ),

            "eligible_gene_log2_tpm": str(
                matrix_info[
                    "eligible_log2_path"
                ].resolve()
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

    readme = f"""STEP 29B
FROZEN TCGA-PRAD RNA ANALYSIS MATRIX

Version:
    {VERSION}

Schema:
    {SCHEMA}


ROLE
----
RNA provenance, matching, QC and matrix materialization only.


RNA SAMPLE AUTHORITY
--------------------
Inherited from:

    {selected_metadata_path}

RNA sample selection was NOT repeated.

Duplicate 01A / 01B selection was NOT repeated.


FROZEN WSI COHORT
-----------------
Frozen patients:
    {len(frozen_ids)}

Matched selected RNA:
    {len(matched_ids)}

Without selected RNA:
    {len(missing_frozen_ids)}

Missing IDs:
    {", ".join(missing_frozen_ids) if missing_frozen_ids else "NONE"}


EXPRESSION
----------
Source column:
    {TPM_COLUMN}

Unit:
    TPM

Transformation:
    log2(TPM + 1)


GENE UNIVERSE
-------------
Input:
    ALL quantified Ensembl genes.

Protein-coding-only restriction:
    NO

Genes before frozen filter:
    {matrix_info['n_genes_all']}


FROZEN DETECTABILITY FILTER
---------------------------
TPM >= {MIN_TPM:g}

in at least:

    {MIN_PATIENT_FRACTION:.0%}

of matched frozen RNA patients.

Minimum required patients:
    {matrix_info['minimum_patient_count']}

Genes retained:
    {matrix_info['n_genes_eligible']}


INFERENCE FIREWALL
------------------
PC3 numerical values read:
    NO

PC6 numerical values read:
    NO

PC3 / gene correlations:
    NOT CALCULATED

PC6 / gene correlations:
    NOT CALCULATED

Gene p-values:
    NOT CALCULATED

Gene FDR:
    NOT CALCULATED

Pathway analysis:
    NOT PERFORMED


NEXT
----
Step29C must inherit:

    frozen_matched_rna_patient_order.csv

    eligible_ensembl_gene_universe.csv

    frozen_matched_patient_by_eligible_gene_log2_tpm_plus1.npy

It may then join frozen Step27G PC3 / PC6 patient scores and execute
the prespecified genome-wide Spearman tests.

Step29C must NOT change patient inclusion or gene eligibility.
"""

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console audit.
    # ------------------------------------------------------------------

    banner(
        "STEP 29B EXECUTION AUDIT"
    )

    audit = [
        (
            "Step29A frozen RNA specification inherited",
            "YES",
        ),

        (
            "Selected patient-level RNA metadata inherited",
            "YES",
        ),

        (
            "RNA sample selection repeated",
            "NO",
        ),

        (
            "01A / 01B duplicate selection repeated",
            "NO",
        ),

        (
            "Frozen 403 patient IDs inherited",
            "YES",
        ),

        (
            "Step27G PC3 numerical values read",
            "NO",
        ),

        (
            "Step27G PC6 numerical values read",
            "NO",
        ),

        (
            "tpm_unstranded authority enforced",
            "YES",
        ),

        (
            "FPKM substituted",
            "NO",
        ),

        (
            "Raw counts substituted",
            "NO",
        ),

        (
            "All Ensembl genes retained before frozen filter",
            "YES",
        ),

        (
            "Protein-coding-only filter applied",
            "NO",
        ),

        (
            "Ensembl version suffix stripped",
            "YES",
        ),

        (
            "_PAR_Y distinction preserved",
            "YES",
        ),

        (
            "Gene identity/order validated across patients",
            "YES",
        ),

        (
            "TPM matrix materialized",
            "YES",
        ),

        (
            "log2(TPM+1) matrix materialized",
            "YES",
        ),

        (
            "Frozen TPM>=1 / >=20% filter applied",
            "YES",
        ),

        (
            "PC3 / gene correlations calculated",
            "NO",
        ),

        (
            "PC6 / gene correlations calculated",
            "NO",
        ),

        (
            "Gene p-values calculated",
            "NO",
        ),

        (
            "Gene FDR calculated",
            "NO",
        ),

        (
            "Pathway analysis performed",
            "NO",
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<70}: {value}"
        )

    banner(
        "STEP 29B: PASS"
    )

    print(
        f"[FROZEN PATIENTS]   {len(frozen_ids)}"
    )

    print(
        f"[RNA MATCHED]       {len(matched_ids)}"
    )

    print(
        f"[RNA MISSING]       {len(missing_frozen_ids)}"
    )

    print(
        f"[GENES ALL]         {matrix_info['n_genes_all']:,}"
    )

    print(
        "[GENE FILTER]       "
        f"TPM >= {MIN_TPM:g} in >= "
        f"{MIN_PATIENT_FRACTION:.0%} matched patients"
    )

    print(
        f"[MIN PATIENT COUNT] {matrix_info['minimum_patient_count']}"
    )

    print(
        f"[GENES ELIGIBLE]    {matrix_info['n_genes_eligible']:,}"
    )

    print(
        "[RNA UNIT]          TPM"
    )

    print(
        "[TRANSFORM]         log2(TPM + 1)"
    )

    print(
        "[PROTEIN CODING]    NO RESTRICTION"
    )

    print(
        "[PC3 VALUES READ]   NO"
    )

    print(
        "[PC6 VALUES READ]   NO"
    )

    print(
        "[ASSOCIATIONS]      NOT CALCULATED"
    )

    print()

    print(
        f"[RESOLUTION]        {resolution_path}"
    )

    print(
        f"[FROZEN MATCHING]   {frozen_matching_path}"
    )

    print(
        f"[PATIENT ORDER]     {matrix_info['patient_order_path']}"
    )

    print(
        f"[GENES ALL]         {matrix_info['all_gene_path']}"
    )

    print(
        f"[GENES ELIGIBLE]    {matrix_info['eligible_gene_path']}"
    )

    print(
        f"[PATIENT QC]        {matrix_info['patient_qc_path']}"
    )

    print(
        f"[ALL TPM]           {matrix_info['all_tpm_path']}"
    )

    print(
        f"[ALL LOG2 TPM]      {matrix_info['all_log2_path']}"
    )

    print(
        f"[ELIGIBLE TPM]      {matrix_info['eligible_tpm_path']}"
    )

    print(
        f"[ELIGIBLE LOG2]     {matrix_info['eligible_log2_path']}"
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
        "[NEXT] Step29C may join the frozen PC3/PC6 patient "
        "scores and execute the frozen genome-wide gene associations."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Audit and materialize the frozen TCGA-PRAD RNA "
            "analysis matrix without reading PC3/PC6 values."
        ),

        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--rna_root",
        type=Path,
        default=DEFAULT_RNA_ROOT,
    )

    parser.add_argument(
        "--selected_metadata",
        type=Path,
        default=DEFAULT_SELECTED_METADATA,
    )

    parser.add_argument(
        "--step29a_spec",
        type=Path,
        default=DEFAULT_STEP29A_SPEC,
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
        rna_root=(
            args.rna_root
            .expanduser()
            .resolve()
        ),

        selected_metadata_path=(
            args.selected_metadata
            .expanduser()
            .resolve()
        ),

        step29a_spec_path=(
            args.step29a_spec
            .expanduser()
            .resolve()
        ),

        frozen_patient_manifest_path=(
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
            "STEP 29B: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)