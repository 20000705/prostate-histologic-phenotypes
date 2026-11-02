#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
28a_build_tcga_prad_clinicopathologic_analysis_table.py

STEP 28A
========
BUILD TCGA-PRAD CLINICOPATHOLOGIC ANALYSIS TABLE

VERSION
-------
1.0.1

SCHEMA
------
28a-tcga-prad-clinicopathologic-analysis-table-schema-1.0.1

ROLE
----
MEASUREMENT / TABLE-ASSEMBLY ONLY.

This step joins the frozen Step27G patient-level PC3 / PC6 representation
to the authoritative Step27H2 TCGA-PRAD patient-level clinical table.

This script MAY:
    - resolve patient identifier columns by prespecified semantic aliases,
    - resolve direct Grade Group columns if present,
    - resolve Gleason primary and secondary pattern/grade fields,
    - deterministically derive Grade Group when direct Grade Group is absent,
    - resolve pathologic T stage,
    - deterministically encode pathologic-T severity,
    - audit missingness and provenance,
    - write the final evaluable analysis table.

This script MUST NOT:
    - calculate PC3 / PC6 correlations,
    - calculate p-values,
    - calculate bootstrap confidence intervals,
    - calculate FDR,
    - select an endpoint because it gives a stronger result,
    - change PC3 / PC6 orientation,
    - change patient aggregation,
    - refit PCA,
    - inspect RNA,
    - inspect survival outcomes.

VERSION 1.0.1 AMENDMENT
-----------------------
Pre-inference implementation correction only.

Changes relative to v1.0.0:

    1. Fix duplicate patient_id column creation during clinical-table
       collapse.

    2. Add authoritative GDC prostate fields:

           primary_gleason_grade
           secondary_gleason_grade

       to the prespecified semantic alias resolution.

No clinical association had been calculated before this correction.

No analytical hypothesis, endpoint, score, aggregation, or inferential
procedure was changed.

REPRESENTATION AUTHORITY
------------------------
Step27G:

    frozen_tcga_prad_patient_representation_manifest.csv

The inferential unit is patient.

GRADE GROUP RULE
----------------
Frozen source precedence:

    1. Direct authoritative Grade Group / ISUP field, if available and valid.

    2. Otherwise derive from authoritative Gleason primary and secondary
       patterns / grades.

Deterministic derivation:

    Gleason 3+3 -> Grade Group 1
    Gleason 3+4 -> Grade Group 2
    Gleason 4+3 -> Grade Group 3
    Gleason total 8 -> Grade Group 4
    Gleason total 9 or 10 -> Grade Group 5

The script records whether each patient uses:

    DIRECT
    DERIVED_FROM_GLEASON
    MISSING

PATHOLOGIC T RULE
-----------------
Canonical labels are parsed from authoritative pathologic-T strings.

Examples:

    T2, pT2, T2A, pT2a -> T2 / T2A
    T3A -> T3A
    T3B -> T3B
    T4 -> T4

Frozen ordinal encoding:

    T2 / T2A / T2B / T2C -> 2.0
    T3 / T3A / T3B       -> 3.0
    T4                    -> 4.0

The primary supportive analysis therefore reflects broad pathologic-T
severity.

IMPORTANT
---------
No association statistics are calculated in Step28A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / SCHEMA
# =============================================================================

VERSION = "1.0.1"

SCHEMA = (
    "28a-tcga-prad-clinicopathologic-"
    "analysis-table-schema-1.0.1"
)


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = RESULTS_ROOT


DEFAULT_STEP27G_PATIENT_MANIFEST = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "frozen_tcga_prad_patient_representation_manifest.csv"
)

DEFAULT_STEP27G_ENDPOINT_SPEC = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "tcga_prad_clinicopathologic_endpoint_specification.csv"
)

DEFAULT_STEP27G_TEST_SPEC = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "tcga_prad_pc3_pc6_clinicopathologic_test_specification.csv"
)

DEFAULT_STEP27G_SPEC = (
    RESULTS
    / "27g_tcga_prad_clinicopathologic_evaluation_specification"
    / "tcga_prad_clinicopathologic_evaluation_specification.json"
)


DEFAULT_CLINICAL_FILE = (
    RESULTS
    / "27h2_tcga_prad_authoritative_clinical_table"
    / "tcga_prad_authoritative_patient_clinical_table.csv"
)


DEFAULT_OUT_DIR = (
    RESULTS
    / "28a_tcga_prad_clinicopathologic_analysis_table"
)


# =============================================================================
# EXPECTATIONS
# =============================================================================

EXPECTED_PATIENTS = 403


PATIENT_COLUMN_ALIASES = [
    "patient_id",
    "case_submitter_id",
    "submitter_id",
    "bcr_patient_barcode",
    "participant_id",
    "case_id",
    "patient",
]


DIRECT_GRADE_GROUP_ALIASES = [
    "grade_group",
    "isup_grade_group",
    "isup_grade",
    "gleason_grade_group",
    "primary_diagnosis_grade_group",
]


# IMPORTANT:
# GDC TCGA-PRAD uses primary_gleason_grade / secondary_gleason_grade.
GLEASON_PRIMARY_ALIASES = [
    "primary_gleason_grade",
    "gleason_primary_pattern",
    "primary_gleason_pattern",
    "gleason_primary",
    "primary_pattern",
]


GLEASON_SECONDARY_ALIASES = [
    "secondary_gleason_grade",
    "gleason_secondary_pattern",
    "secondary_gleason_pattern",
    "gleason_secondary",
    "secondary_pattern",
]


GLEASON_TOTAL_ALIASES = [
    "gleason_score",
    "gleason_sum",
    "total_gleason_score",
    "combined_gleason_score",
]


PATHOLOGIC_T_ALIASES = [
    "ajcc_pathologic_t",
    "ajcc_pathologic_t_stage",
    "pathologic_t",
    "pathologic_t_stage",
    "pathologic_tumor_stage",
    "tumor_pathologic_t",
    "pt_stage",
]


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
    width: int = 112,
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


def normalize_column_name(
    value: str,
) -> str:

    value = str(
        value
    ).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip(
        "_"
    )


def normalize_string(
    value,
) -> Optional[str]:

    if pd.isna(
        value
    ):
        return None

    text = str(
        value
    ).strip()

    if text == "":
        return None

    if text.lower() in {
        "nan",
        "na",
        "n/a",
        "none",
        "null",
        "not reported",
        "not available",
        "unknown",
        "--",
    }:

        return None

    return text


# =============================================================================
# CLINICAL FILE
# =============================================================================

def read_clinical_file(
    path: Path,
) -> pd.DataFrame:

    require_file(
        path,
        "Clinical metadata file",
    )

    suffix = path.suffix.lower()

    if suffix in {
        ".tsv",
        ".txt",
    }:

        df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
        )

    elif suffix == ".csv":

        df = pd.read_csv(
            path,
            low_memory=False,
        )

    else:

        raise RuntimeError(
            "Unsupported clinical file extension. "
            "Use CSV or TSV."
        )

    if len(
        df
    ) == 0:

        raise RuntimeError(
            "Clinical metadata table is empty."
        )

    original_columns = list(
        df.columns
    )

    normalized_columns = [
        normalize_column_name(
            column
        )
        for column in original_columns
    ]

    if len(
        normalized_columns
    ) != len(
        set(
            normalized_columns
        )
    ):

        duplicates = pd.Series(
            normalized_columns
        ).value_counts()

        duplicates = duplicates.loc[
            duplicates > 1
        ]

        raise RuntimeError(
            "Clinical columns become duplicated after normalization:\n"
            + duplicates.to_string()
        )

    df.columns = normalized_columns

    return df


# =============================================================================
# COLUMN RESOLUTION
# =============================================================================

def resolve_column(
    columns: Sequence[str],
    aliases: Sequence[str],
    label: str,
    required: bool,
) -> Optional[str]:

    columns_set = set(
        columns
    )

    normalized_aliases = [
        normalize_column_name(
            alias
        )
        for alias in aliases
    ]

    matches = [
        alias
        for alias in normalized_aliases
        if alias in columns_set
    ]

    if len(
        matches
    ) == 0:

        if required:

            raise RuntimeError(
                f"Unable to resolve required clinical field: "
                f"{label}\n"
                f"Accepted aliases: {normalized_aliases}"
            )

        return None

    # Prespecified semantic precedence:
    # first matching alias wins.
    return matches[
        0
    ]


# =============================================================================
# TCGA PATIENT ID
# =============================================================================

TCGA_PATIENT_RE = re.compile(
    r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})",
    re.IGNORECASE,
)


def canonical_tcga_patient_id(
    value,
) -> Optional[str]:

    text = normalize_string(
        value
    )

    if text is None:
        return None

    match = TCGA_PATIENT_RE.search(
        text
    )

    if match is None:
        return None

    return match.group(
        1
    ).upper()


# =============================================================================
# NUMERIC PARSING
# =============================================================================

def parse_integer_like(
    value,
) -> Optional[int]:

    text = normalize_string(
        value
    )

    if text is None:
        return None

    match = re.search(
        r"(?<!\d)([0-9]+)(?!\d)",
        text,
    )

    if match is None:
        return None

    try:

        return int(
            match.group(
                1
            )
        )

    except Exception:

        return None


def parse_grade_group(
    value,
) -> Optional[int]:

    parsed = parse_integer_like(
        value
    )

    if parsed is None:
        return None

    if parsed not in {
        1,
        2,
        3,
        4,
        5,
    }:

        return None

    return parsed


def parse_gleason_pattern(
    value,
) -> Optional[int]:

    parsed = parse_integer_like(
        value
    )

    if parsed is None:
        return None

    if parsed not in {
        3,
        4,
        5,
    }:

        return None

    return parsed


def parse_gleason_total(
    value,
) -> Optional[int]:

    parsed = parse_integer_like(
        value
    )

    if parsed is None:
        return None

    if parsed < 6 or parsed > 10:

        return None

    return parsed


# =============================================================================
# GRADE GROUP DERIVATION
# =============================================================================

def derive_grade_group(
    primary: Optional[int],
    secondary: Optional[int],
    total: Optional[int],
) -> Optional[int]:

    # ------------------------------------------------------------------
    # Prefer pattern-aware derivation whenever both Gleason components
    # are available.
    # ------------------------------------------------------------------

    if (
        primary is not None
        and secondary is not None
    ):

        gleason_sum = (
            primary
            + secondary
        )

        if (
            primary == 3
            and secondary == 3
        ):

            return 1

        if (
            primary == 3
            and secondary == 4
        ):

            return 2

        if (
            primary == 4
            and secondary == 3
        ):

            return 3

        if gleason_sum == 8:

            return 4

        if gleason_sum in {
            9,
            10,
        }:

            return 5

    # ------------------------------------------------------------------
    # Total-only fallback is allowed only when Grade Group is unambiguous.
    # ------------------------------------------------------------------

    if total is not None:

        if total == 6:

            return 1

        # Total 7 alone cannot distinguish:
        #
        #   3+4 -> GG2
        #   4+3 -> GG3
        #
        # so it is deliberately NOT mapped.
        if total == 7:

            return None

        if total == 8:

            return 4

        if total in {
            9,
            10,
        }:

            return 5

    return None


# =============================================================================
# PATHOLOGIC T
# =============================================================================

def canonicalize_pathologic_t(
    value,
) -> Optional[str]:

    text = normalize_string(
        value
    )

    if text is None:
        return None

    text = (
        text
        .upper()
        .replace(
            " ",
            ""
        )
    )

    # pT3a -> PT3A -> T3A
    if text.startswith(
        "PT"
    ):

        text = text[
            1:
        ]

    match = re.search(
        r"T([234])([ABC]?)",
        text,
    )

    if match is None:

        return None

    return (
        "T"
        + match.group(
            1
        )
        + match.group(
            2
        )
    )


def pathologic_t_ordinal(
    canonical_stage: Optional[str],
) -> Optional[float]:

    if canonical_stage is None:

        return None

    if canonical_stage.startswith(
        "T2"
    ):

        return 2.0

    if canonical_stage.startswith(
        "T3"
    ):

        return 3.0

    if canonical_stage.startswith(
        "T4"
    ):

        return 4.0

    return None


# =============================================================================
# STEP27G VALIDATION
# =============================================================================

def load_step27g(
    patient_manifest_path: Path,
    endpoint_spec_path: Path,
    test_spec_path: Path,
    spec_path: Path,
) -> pd.DataFrame:

    require_file(
        patient_manifest_path,
        "Step27G patient manifest",
    )

    require_file(
        endpoint_spec_path,
        "Step27G endpoint specification",
    )

    require_file(
        test_spec_path,
        "Step27G test specification",
    )

    require_file(
        spec_path,
        "Step27G JSON specification",
    )

    patient = pd.read_csv(
        patient_manifest_path
    )

    endpoint = pd.read_csv(
        endpoint_spec_path
    )

    test = pd.read_csv(
        test_spec_path
    )

    specification = load_json(
        spec_path
    )

    if (
        specification.get(
            "analysis_role"
        )
        !=
        "PRE_EXECUTION_TCGA_CLINICOPATHOLOGIC_EVALUATION_FREEZE"
    ):

        raise RuntimeError(
            "Unexpected Step27G analysis role."
        )

    required_patient_columns = {
        "patient_id",
        "n_frozen_slides",
        "total_tiles",
        "patient_oriented_PC3",
        "patient_oriented_PC6",
        "patient_aggregation_rule",
    }

    missing = sorted(
        required_patient_columns
        - set(
            patient.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Step27G patient manifest schema changed. "
            f"Missing columns: {missing}"
        )

    if len(
        patient
    ) != EXPECTED_PATIENTS:

        raise RuntimeError(
            "Step27G patient count changed: "
            f"{len(patient)} != {EXPECTED_PATIENTS}"
        )

    if (
        patient[
            "patient_id"
        ]
        .astype(str)
        .nunique()
        != EXPECTED_PATIENTS
    ):

        raise RuntimeError(
            "Step27G patient manifest does not contain "
            "403 unique patients."
        )

    expected_endpoints = {
        "GRADE_GROUP",
        "PATHOLOGIC_T_STAGE",
    }

    observed_endpoints = set(
        endpoint[
            "endpoint"
        ].astype(str)
    )

    if not expected_endpoints.issubset(
        observed_endpoints
    ):

        raise RuntimeError(
            "Step27G endpoint freeze is inconsistent."
        )

    expected_tests = {
        (
            "PC3",
            "GRADE_GROUP",
        ),
        (
            "PC6",
            "GRADE_GROUP",
        ),
        (
            "PC3",
            "PATHOLOGIC_T_STAGE",
        ),
        (
            "PC6",
            "PATHOLOGIC_T_STAGE",
        ),
    }

    observed_tests = set(
        zip(
            test[
                "axis"
            ].astype(str),
            test[
                "endpoint"
            ].astype(str),
        )
    )

    if not expected_tests.issubset(
        observed_tests
    ):

        raise RuntimeError(
            "Step27G test specification is incomplete."
        )

    return patient


# =============================================================================
# CLINICAL DEDUPLICATION
# =============================================================================

def collapse_clinical_to_patient(
    clinical: pd.DataFrame,
    patient_column: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    """
    Collapse authoritative clinical input to one row per canonical TCGA patient.

    IMPORTANT v1.0.1 FIX
    --------------------
    The original patient identifier column is NOT carried into the clinical
    payload after canonicalization.

    This prevents creation of two columns both named patient_id.
    """

    df = clinical.copy()

    # ------------------------------------------------------------------
    # Deterministically derive canonical TCGA patient identifier.
    # ------------------------------------------------------------------

    df[
        "_canonical_patient_id"
    ] = df[
        patient_column
    ].map(
        canonical_tcga_patient_id
    )

    invalid_mask = df[
        "_canonical_patient_id"
    ].isna()

    df = df.loc[
        ~invalid_mask
    ].copy()

    # ------------------------------------------------------------------
    # Do not carry the original patient identifier column into the
    # collapsed clinical payload.
    #
    # Example:
    #
    #   patient_id
    #   _canonical_patient_id
    #
    # must NOT later both become patient_id.
    # ------------------------------------------------------------------

    payload_columns = [
        column
        for column in df.columns
        if column not in {
            patient_column,
            "_canonical_patient_id",
        }
    ]

    collapsed_rows = []
    audit_rows = []

    for patient_id, group in df.groupby(
        "_canonical_patient_id",
        sort=True,
    ):

        output = {
            "patient_id": (
                patient_id
            )
        }

        conflict_columns = []

        for column in payload_columns:

            values = [
                normalize_string(
                    value
                )
                for value
                in group[
                    column
                ].tolist()
            ]

            values = [
                value
                for value
                in values
                if value is not None
            ]

            unique_values = list(
                dict.fromkeys(
                    values
                )
            )

            if len(
                unique_values
            ) == 0:

                output[
                    column
                ] = np.nan

            elif len(
                unique_values
            ) == 1:

                output[
                    column
                ] = unique_values[
                    0
                ]

            else:

                # Never silently choose conflicting clinical values.
                conflict_columns.append(
                    column
                )

                output[
                    column
                ] = np.nan

        collapsed_rows.append(
            output
        )

        audit_rows.append({
            "patient_id": (
                patient_id
            ),

            "source_rows": int(
                len(
                    group
                )
            ),

            "duplicate_source_rows": (
                len(
                    group
                )
                > 1
            ),

            "conflict_count": int(
                len(
                    conflict_columns
                )
            ),

            "conflict_columns": ";".join(
                conflict_columns
            ),
        })

    collapsed = pd.DataFrame(
        collapsed_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    # ------------------------------------------------------------------
    # Hard uniqueness checks.
    # ------------------------------------------------------------------

    if (
        "patient_id"
        not in collapsed.columns
    ):

        raise RuntimeError(
            "Collapsed clinical table lacks patient_id."
        )

    if list(
        collapsed.columns
    ).count(
        "patient_id"
    ) != 1:

        raise RuntimeError(
            "Collapsed clinical table contains duplicate "
            "patient_id column labels."
        )

    if not collapsed[
        "patient_id"
    ].is_unique:

        raise RuntimeError(
            "Collapsed clinical table is not one row per patient."
        )

    return (
        collapsed,
        audit,
    )


# =============================================================================
# ANALYSIS TABLE
# =============================================================================

def build_analysis_table(
    patient_manifest: pd.DataFrame,
    clinical: pd.DataFrame,
    patient_column: str,
    direct_grade_column: Optional[str],
    gleason_primary_column: Optional[str],
    gleason_secondary_column: Optional[str],
    gleason_total_column: Optional[str],
    pathologic_t_column: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    (
        clinical_patient,
        duplicate_audit,
    ) = collapse_clinical_to_patient(
        clinical=clinical,
        patient_column=patient_column,
    )

    # IMPORTANT:
    # No rename of _patient_id -> patient_id is needed in v1.0.1.
    # collapse_clinical_to_patient() already returns exactly one
    # canonical patient_id column.

    if list(
        clinical_patient.columns
    ).count(
        "patient_id"
    ) != 1:

        raise RuntimeError(
            "Clinical patient table contains duplicate "
            "patient_id labels before merge."
        )

    if not clinical_patient[
        "patient_id"
    ].is_unique:

        raise RuntimeError(
            "Clinical patient table has duplicate patient rows."
        )

    if not patient_manifest[
        "patient_id"
    ].is_unique:

        raise RuntimeError(
            "Step27G frozen patient manifest is not unique."
        )

    merged = patient_manifest.merge(
        clinical_patient,
        on="patient_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=(
            "",
            "_clinical",
        ),
    )

    # ------------------------------------------------------------------
    # Preserve raw source fields.
    # ------------------------------------------------------------------

    if direct_grade_column is not None:

        merged[
            "source_grade_group_raw"
        ] = merged[
            direct_grade_column
        ]

    else:

        merged[
            "source_grade_group_raw"
        ] = np.nan

    if gleason_primary_column is not None:

        merged[
            "source_gleason_primary_raw"
        ] = merged[
            gleason_primary_column
        ]

    else:

        merged[
            "source_gleason_primary_raw"
        ] = np.nan

    if gleason_secondary_column is not None:

        merged[
            "source_gleason_secondary_raw"
        ] = merged[
            gleason_secondary_column
        ]

    else:

        merged[
            "source_gleason_secondary_raw"
        ] = np.nan

    if gleason_total_column is not None:

        merged[
            "source_gleason_total_raw"
        ] = merged[
            gleason_total_column
        ]

    else:

        merged[
            "source_gleason_total_raw"
        ] = np.nan

    merged[
        "source_pathologic_t_raw"
    ] = merged[
        pathologic_t_column
    ]

    # ------------------------------------------------------------------
    # Parse Grade Group / Gleason.
    # ------------------------------------------------------------------

    merged[
        "grade_group_direct"
    ] = merged[
        "source_grade_group_raw"
    ].map(
        parse_grade_group
    )

    merged[
        "gleason_primary"
    ] = merged[
        "source_gleason_primary_raw"
    ].map(
        parse_gleason_pattern
    )

    merged[
        "gleason_secondary"
    ] = merged[
        "source_gleason_secondary_raw"
    ].map(
        parse_gleason_pattern
    )

    merged[
        "gleason_total"
    ] = merged[
        "source_gleason_total_raw"
    ].map(
        parse_gleason_total
    )

    derived_grade_values = []
    final_grade_values = []
    grade_source_values = []

    for _, row in merged.iterrows():

        direct_raw = row[
            "grade_group_direct"
        ]

        primary_raw = row[
            "gleason_primary"
        ]

        secondary_raw = row[
            "gleason_secondary"
        ]

        total_raw = row[
            "gleason_total"
        ]

        direct = (
            None
            if pd.isna(
                direct_raw
            )
            else int(
                direct_raw
            )
        )

        primary = (
            None
            if pd.isna(
                primary_raw
            )
            else int(
                primary_raw
            )
        )

        secondary = (
            None
            if pd.isna(
                secondary_raw
            )
            else int(
                secondary_raw
            )
        )

        total = (
            None
            if pd.isna(
                total_raw
            )
            else int(
                total_raw
            )
        )

        derived = derive_grade_group(
            primary=primary,
            secondary=secondary,
            total=total,
        )

        derived_grade_values.append(
            derived
        )

        if direct is not None:

            final_grade_values.append(
                direct
            )

            grade_source_values.append(
                "DIRECT"
            )

        elif derived is not None:

            final_grade_values.append(
                derived
            )

            grade_source_values.append(
                "DERIVED_FROM_GLEASON"
            )

        else:

            final_grade_values.append(
                np.nan
            )

            grade_source_values.append(
                "MISSING"
            )

    merged[
        "grade_group_derived"
    ] = derived_grade_values

    merged[
        "grade_group"
    ] = final_grade_values

    merged[
        "grade_group_source"
    ] = grade_source_values

    # ------------------------------------------------------------------
    # Direct-vs-derived cross-check.
    # ------------------------------------------------------------------

    both_grade_sources = (
        merged[
            "grade_group_direct"
        ].notna()
        &
        merged[
            "grade_group_derived"
        ].notna()
    )

    concordance_values = []

    for _, row in merged.iterrows():

        if (
            pd.notna(
                row[
                    "grade_group_direct"
                ]
            )
            and pd.notna(
                row[
                    "grade_group_derived"
                ]
            )
        ):

            concordance_values.append(
                bool(
                    int(
                        row[
                            "grade_group_direct"
                        ]
                    )
                    ==
                    int(
                        row[
                            "grade_group_derived"
                        ]
                    )
                )
            )

        else:

            concordance_values.append(
                np.nan
            )

    merged[
        "grade_group_direct_derived_concordant"
    ] = concordance_values

    # ------------------------------------------------------------------
    # Pathologic T.
    # ------------------------------------------------------------------

    merged[
        "pathologic_t_canonical"
    ] = merged[
        "source_pathologic_t_raw"
    ].map(
        canonicalize_pathologic_t
    )

    merged[
        "pathologic_t_ordinal"
    ] = merged[
        "pathologic_t_canonical"
    ].map(
        pathologic_t_ordinal
    )

    # ------------------------------------------------------------------
    # Evaluable flags.
    # ------------------------------------------------------------------

    merged[
        "has_clinical_match"
    ] = (
        merged[
            "_merge"
        ]
        == "both"
    )

    merged[
        "grade_group_evaluable"
    ] = merged[
        "grade_group"
    ].notna()

    merged[
        "pathologic_t_evaluable"
    ] = merged[
        "pathologic_t_ordinal"
    ].notna()

    merged[
        "joint_grade_t_evaluable"
    ] = (
        merged[
            "grade_group_evaluable"
        ]
        &
        merged[
            "pathologic_t_evaluable"
        ]
    )

    # ------------------------------------------------------------------
    # Explicit firewall.
    # ------------------------------------------------------------------

    merged[
        "association_calculated"
    ] = False

    merged[
        "inferential_unit"
    ] = "patient"

    # ------------------------------------------------------------------
    # Formal table.
    # ------------------------------------------------------------------

    formal_columns = [
        "patient_id",

        "n_frozen_slides",
        "total_tiles",

        "patient_oriented_PC3",
        "patient_oriented_PC6",

        "source_grade_group_raw",

        "source_gleason_primary_raw",
        "source_gleason_secondary_raw",
        "source_gleason_total_raw",

        "grade_group_direct",
        "grade_group_derived",
        "grade_group",
        "grade_group_source",
        "grade_group_direct_derived_concordant",

        "source_pathologic_t_raw",
        "pathologic_t_canonical",
        "pathologic_t_ordinal",

        "has_clinical_match",
        "grade_group_evaluable",
        "pathologic_t_evaluable",
        "joint_grade_t_evaluable",

        "inferential_unit",
        "association_calculated",
    ]

    formal = merged[
        formal_columns
    ].copy()

    return (
        formal,
        merged,
        duplicate_audit,
    )


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def build_endpoint_availability(
    analysis: pd.DataFrame,
) -> pd.DataFrame:

    n_total = len(
        analysis
    )

    rows = [
        {
            "endpoint": (
                "CLINICAL_MATCH"
            ),
            "n_total": (
                n_total
            ),
            "n_evaluable": int(
                analysis[
                    "has_clinical_match"
                ].sum()
            ),
        },
        {
            "endpoint": (
                "GRADE_GROUP"
            ),
            "n_total": (
                n_total
            ),
            "n_evaluable": int(
                analysis[
                    "grade_group_evaluable"
                ].sum()
            ),
        },
        {
            "endpoint": (
                "PATHOLOGIC_T_STAGE"
            ),
            "n_total": (
                n_total
            ),
            "n_evaluable": int(
                analysis[
                    "pathologic_t_evaluable"
                ].sum()
            ),
        },
        {
            "endpoint": (
                "JOINT_GRADE_AND_T"
            ),
            "n_total": (
                n_total
            ),
            "n_evaluable": int(
                analysis[
                    "joint_grade_t_evaluable"
                ].sum()
            ),
        },
    ]

    df = pd.DataFrame(
        rows
    )

    df[
        "fraction_evaluable"
    ] = (
        df[
            "n_evaluable"
        ]
        /
        df[
            "n_total"
        ]
    )

    return df


def build_grade_distribution(
    analysis: pd.DataFrame,
) -> pd.DataFrame:

    evaluable = analysis.loc[
        analysis[
            "grade_group_evaluable"
        ]
    ].copy()

    if len(
        evaluable
    ) == 0:

        return pd.DataFrame({
            "grade_group": [
                1,
                2,
                3,
                4,
                5,
            ],
            "n": [
                0,
                0,
                0,
                0,
                0,
            ],
            "fraction": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        })

    counts = (
        evaluable[
            "grade_group"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    rows = []

    for grade_group in range(
        1,
        6,
    ):

        rows.append({
            "grade_group": (
                grade_group
            ),

            "n": int(
                counts.get(
                    grade_group,
                    0,
                )
            ),
        })

    output = pd.DataFrame(
        rows
    )

    total = int(
        output[
            "n"
        ].sum()
    )

    output[
        "fraction"
    ] = np.where(
        total > 0,
        output[
            "n"
        ]
        /
        total,
        np.nan,
    )

    return output


def build_pt_distribution(
    analysis: pd.DataFrame,
) -> pd.DataFrame:

    evaluable = analysis.loc[
        analysis[
            "pathologic_t_evaluable"
        ]
    ].copy()

    counts = (
        evaluable[
            "pathologic_t_canonical"
        ]
        .value_counts()
        .sort_index()
    )

    return (
        counts
        .rename_axis(
            "pathologic_t_canonical"
        )
        .reset_index(
            name="n"
        )
    )


# =============================================================================
# CLAIM BOUNDARY
# =============================================================================

def claim_boundary_text() -> str:

    return """STEP 28A — CLAIM BOUNDARY

ROLE
----
CLINICOPATHOLOGIC TABLE ASSEMBLY / MEASUREMENT ONLY.

STEP28A MAY ESTABLISH
---------------------
1. Which frozen TCGA patients have authoritative clinicopathologic metadata.
2. Which Grade Group / Gleason fields were used.
3. Which pathologic-T field was used.
4. Which patients are evaluable for the frozen Step27G endpoints.
5. Deterministic Grade Group and pathologic-T mappings.
6. Missingness and provenance.

STEP28A DOES NOT ESTABLISH
--------------------------
1. PC3 association with Grade Group.
2. PC6 association with Grade Group.
3. PC3 association with pathologic T stage.
4. PC6 association with pathologic T stage.
5. Statistical significance.
6. Effect size.
7. Clinical utility.
8. Molecular association.
9. Prognostic value.

Those require Step28B or later.

PROHIBITED
----------
No endpoint may be added, removed, re-encoded, or replaced based on
PC3 / PC6 association results.
"""


# =============================================================================
# EXECUTION
# =============================================================================

def execute(
    clinical_file: Path,
    step27g_patient_manifest_path: Path,
    step27g_endpoint_spec_path: Path,
    step27g_test_spec_path: Path,
    step27g_spec_path: Path,
    out_dir: Path,
    patient_column_override: Optional[str],
    grade_group_column_override: Optional[str],
    gleason_primary_column_override: Optional[str],
    gleason_secondary_column_override: Optional[str],
    gleason_total_column_override: Optional[str],
    pathologic_t_column_override: Optional[str],
) -> None:

    banner(
        "STEP 28A: BUILD TCGA-PRAD "
        "CLINICOPATHOLOGIC ANALYSIS TABLE"
    )

    print(
        f"[VERSION]       {VERSION}"
    )

    print(
        f"[SCHEMA]        {SCHEMA}"
    )

    print(
        f"[CLINICAL]      {clinical_file}"
    )

    # ------------------------------------------------------------------
    # Frozen Step27G authority.
    # ------------------------------------------------------------------

    patient_manifest = load_step27g(
        patient_manifest_path=(
            step27g_patient_manifest_path
        ),

        endpoint_spec_path=(
            step27g_endpoint_spec_path
        ),

        test_spec_path=(
            step27g_test_spec_path
        ),

        spec_path=(
            step27g_spec_path
        ),
    )

    # ------------------------------------------------------------------
    # Authoritative clinical table.
    # ------------------------------------------------------------------

    clinical = read_clinical_file(
        clinical_file
    )

    banner(
        "RESOLVE AUTHORITATIVE CLINICAL FIELDS"
    )

    # ------------------------------------------------------------------
    # Patient ID.
    # ------------------------------------------------------------------

    if patient_column_override is None:

        patient_column = resolve_column(
            columns=clinical.columns,
            aliases=PATIENT_COLUMN_ALIASES,
            label="patient identifier",
            required=True,
        )

    else:

        patient_column = normalize_column_name(
            patient_column_override
        )

        if patient_column not in clinical.columns:

            raise RuntimeError(
                f"Requested patient column not found: "
                f"{patient_column}"
            )

    # ------------------------------------------------------------------
    # Direct Grade Group.
    # ------------------------------------------------------------------

    if grade_group_column_override is None:

        direct_grade_column = resolve_column(
            columns=clinical.columns,
            aliases=DIRECT_GRADE_GROUP_ALIASES,
            label="direct Grade Group",
            required=False,
        )

    else:

        direct_grade_column = normalize_column_name(
            grade_group_column_override
        )

        if direct_grade_column not in clinical.columns:

            raise RuntimeError(
                "Requested Grade Group column not found: "
                f"{direct_grade_column}"
            )

    # ------------------------------------------------------------------
    # Gleason primary.
    # ------------------------------------------------------------------

    if gleason_primary_column_override is None:

        gleason_primary_column = resolve_column(
            columns=clinical.columns,
            aliases=GLEASON_PRIMARY_ALIASES,
            label="Gleason primary grade/pattern",
            required=False,
        )

    else:

        gleason_primary_column = normalize_column_name(
            gleason_primary_column_override
        )

        if gleason_primary_column not in clinical.columns:

            raise RuntimeError(
                "Requested Gleason primary column not found: "
                f"{gleason_primary_column}"
            )

    # ------------------------------------------------------------------
    # Gleason secondary.
    # ------------------------------------------------------------------

    if gleason_secondary_column_override is None:

        gleason_secondary_column = resolve_column(
            columns=clinical.columns,
            aliases=GLEASON_SECONDARY_ALIASES,
            label="Gleason secondary grade/pattern",
            required=False,
        )

    else:

        gleason_secondary_column = normalize_column_name(
            gleason_secondary_column_override
        )

        if gleason_secondary_column not in clinical.columns:

            raise RuntimeError(
                "Requested Gleason secondary column not found: "
                f"{gleason_secondary_column}"
            )

    # ------------------------------------------------------------------
    # Gleason total.
    # ------------------------------------------------------------------

    if gleason_total_column_override is None:

        gleason_total_column = resolve_column(
            columns=clinical.columns,
            aliases=GLEASON_TOTAL_ALIASES,
            label="Gleason total score",
            required=False,
        )

    else:

        gleason_total_column = normalize_column_name(
            gleason_total_column_override
        )

        if gleason_total_column not in clinical.columns:

            raise RuntimeError(
                "Requested Gleason total column not found: "
                f"{gleason_total_column}"
            )

    # ------------------------------------------------------------------
    # Pathologic T.
    # ------------------------------------------------------------------

    if pathologic_t_column_override is None:

        pathologic_t_column = resolve_column(
            columns=clinical.columns,
            aliases=PATHOLOGIC_T_ALIASES,
            label="pathologic T stage",
            required=True,
        )

    else:

        pathologic_t_column = normalize_column_name(
            pathologic_t_column_override
        )

        if pathologic_t_column not in clinical.columns:

            raise RuntimeError(
                "Requested pathologic-T column not found: "
                f"{pathologic_t_column}"
            )

    # ------------------------------------------------------------------
    # Grade source availability.
    # ------------------------------------------------------------------

    if (
        direct_grade_column is None
        and gleason_primary_column is None
        and gleason_secondary_column is None
        and gleason_total_column is None
    ):

        raise RuntimeError(
            "No Grade Group or Gleason source fields were found."
        )

    print(
        f"Patient ID       : {patient_column}"
    )

    print(
        "Direct GradeGroup: "
        f"{direct_grade_column}"
    )

    print(
        "Gleason primary  : "
        f"{gleason_primary_column}"
    )

    print(
        "Gleason secondary: "
        f"{gleason_secondary_column}"
    )

    print(
        "Gleason total    : "
        f"{gleason_total_column}"
    )

    print(
        "Pathologic T     : "
        f"{pathologic_t_column}"
    )

    # ------------------------------------------------------------------
    # Build analysis table.
    # ------------------------------------------------------------------

    (
        analysis_table,
        full_joined_table,
        duplicate_audit,
    ) = build_analysis_table(
        patient_manifest=(
            patient_manifest
        ),

        clinical=(
            clinical
        ),

        patient_column=(
            patient_column
        ),

        direct_grade_column=(
            direct_grade_column
        ),

        gleason_primary_column=(
            gleason_primary_column
        ),

        gleason_secondary_column=(
            gleason_secondary_column
        ),

        gleason_total_column=(
            gleason_total_column
        ),

        pathologic_t_column=(
            pathologic_t_column
        ),
    )

    # ------------------------------------------------------------------
    # Strong patient-level integrity checks.
    # ------------------------------------------------------------------

    if len(
        analysis_table
    ) != EXPECTED_PATIENTS:

        raise RuntimeError(
            "Final analysis table does not contain "
            "403 frozen patients."
        )

    if (
        analysis_table[
            "patient_id"
        ].nunique()
        != EXPECTED_PATIENTS
    ):

        raise RuntimeError(
            "Final analysis table does not contain "
            "403 unique patients."
        )

    # ------------------------------------------------------------------
    # Descriptive endpoint tables only.
    # ------------------------------------------------------------------

    endpoint_availability = build_endpoint_availability(
        analysis_table
    )

    grade_distribution = build_grade_distribution(
        analysis_table
    )

    pt_distribution = build_pt_distribution(
        analysis_table
    )

    # ------------------------------------------------------------------
    # Direct-vs-derived Grade Group concordance.
    # ------------------------------------------------------------------

    grade_crosscheck_mask = (
        analysis_table[
            "grade_group_direct_derived_concordant"
        ].notna()
    )

    n_grade_crosscheck = int(
        grade_crosscheck_mask.sum()
    )

    if n_grade_crosscheck > 0:

        grade_source_concordance = float(
            analysis_table.loc[
                grade_crosscheck_mask,
                "grade_group_direct_derived_concordant",
            ]
            .astype(bool)
            .mean()
        )

    else:

        grade_source_concordance = None

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_path = (
        out_dir
        / "tcga_prad_frozen_pc3_pc6_clinicopathologic_analysis_table.csv"
    )

    full_joined_path = (
        out_dir
        / "tcga_prad_full_clinical_join_audit_table.csv"
    )

    availability_path = (
        out_dir
        / "tcga_prad_clinicopathologic_endpoint_availability.csv"
    )

    grade_distribution_path = (
        out_dir
        / "tcga_prad_grade_group_distribution.csv"
    )

    pt_distribution_path = (
        out_dir
        / "tcga_prad_pathologic_t_distribution.csv"
    )

    duplicate_audit_path = (
        out_dir
        / "tcga_prad_clinical_duplicate_patient_audit.csv"
    )

    field_mapping_path = (
        out_dir
        / "tcga_prad_clinical_field_mapping.json"
    )

    summary_path = (
        out_dir
        / "tcga_prad_clinicopathologic_analysis_table_summary.json"
    )

    hashes_path = (
        out_dir
        / "source_hashes.csv"
    )

    claim_path = (
        out_dir
        / "tcga_prad_clinicopathologic_table_claim_boundary.txt"
    )

    readme_path = (
        out_dir
        / "README.txt"
    )

    analysis_table.to_csv(
        analysis_path,
        index=False,
    )

    full_joined_table.to_csv(
        full_joined_path,
        index=False,
    )

    endpoint_availability.to_csv(
        availability_path,
        index=False,
    )

    grade_distribution.to_csv(
        grade_distribution_path,
        index=False,
    )

    pt_distribution.to_csv(
        pt_distribution_path,
        index=False,
    )

    duplicate_audit.to_csv(
        duplicate_audit_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Clinical field provenance.
    # ------------------------------------------------------------------

    field_mapping = {
        "version": (
            VERSION
        ),

        "schema": (
            SCHEMA
        ),

        "clinical_file": str(
            clinical_file.resolve()
        ),

        "selected_fields": {
            "patient_id": (
                patient_column
            ),

            "direct_grade_group": (
                direct_grade_column
            ),

            "gleason_primary": (
                gleason_primary_column
            ),

            "gleason_secondary": (
                gleason_secondary_column
            ),

            "gleason_total": (
                gleason_total_column
            ),

            "pathologic_t": (
                pathologic_t_column
            ),
        },

        "selection_rule": (
            "Prespecified semantic alias precedence or explicit "
            "CLI override. No PC3/PC6 association result was used."
        ),

        "grade_group_precedence": [
            "direct_authoritative_grade_group",
            "deterministic_derivation_from_primary_secondary_gleason",
            "unambiguous_total_gleason_fallback",
            "missing",
        ],

        "grade_group_mapping": {
            "3+3": 1,
            "3+4": 2,
            "4+3": 3,
            "total_8": 4,
            "total_9_or_10": 5,
            "total_7_without_patterns": (
                "not_evaluable"
            ),
        },

        "pathologic_t_ordinal_rule": {
            "T2/T2A/T2B/T2C": 2,
            "T3/T3A/T3B": 3,
            "T4": 4,
        },

        "v1_0_1_amendment": {
            "patient_id_duplicate_column_bug_fixed": True,
            "primary_gleason_grade_alias_added": True,
            "secondary_gleason_grade_alias_added": True,
            "association_had_been_calculated_before_fix": False,
            "inferential_design_changed": False,
        },

        "association_calculated": False,
    }

    save_json_atomic(
        field_mapping_path,
        field_mapping,
    )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------

    summary = {
        "version": (
            VERSION
        ),

        "schema": (
            SCHEMA
        ),

        "status": (
            "PASS"
        ),

        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "analysis_role": (
            "CLINICOPATHOLOGIC_TABLE_ASSEMBLY_ONLY"
        ),

        "frozen_patients": int(
            len(
                analysis_table
            )
        ),

        "clinical_file_rows": int(
            len(
                clinical
            )
        ),

        "clinical_matches": int(
            analysis_table[
                "has_clinical_match"
            ].sum()
        ),

        "grade_group_evaluable": int(
            analysis_table[
                "grade_group_evaluable"
            ].sum()
        ),

        "pathologic_t_evaluable": int(
            analysis_table[
                "pathologic_t_evaluable"
            ].sum()
        ),

        "joint_grade_t_evaluable": int(
            analysis_table[
                "joint_grade_t_evaluable"
            ].sum()
        ),

        "grade_group_source_counts": {
            str(
                key
            ): int(
                value
            )
            for key, value
            in (
                analysis_table[
                    "grade_group_source"
                ]
                .value_counts(
                    dropna=False
                )
                .items()
            )
        },

        "grade_direct_derived_crosscheck_n": (
            n_grade_crosscheck
        ),

        "grade_direct_derived_concordance": (
            grade_source_concordance
        ),

        "association_calculated": False,

        "pc3_pc6_results_inspected": False,

        "v1_0_1_amendment": {
            "pre_inference_bugfix_only": True,
            "patient_id_collision_fixed": True,
            "gdc_gleason_field_aliases_fixed": True,
            "analytical_design_changed": False,
        },

        "outputs": {
            "analysis_table": str(
                analysis_path.resolve()
            ),

            "full_join_audit": str(
                full_joined_path.resolve()
            ),

            "endpoint_availability": str(
                availability_path.resolve()
            ),

            "grade_distribution": str(
                grade_distribution_path.resolve()
            ),

            "pathologic_t_distribution": str(
                pt_distribution_path.resolve()
            ),

            "duplicate_patient_audit": str(
                duplicate_audit_path.resolve()
            ),

            "field_mapping": str(
                field_mapping_path.resolve()
            ),
        },
    }

    save_json_atomic(
        summary_path,
        summary,
    )

    claim_path.write_text(
        claim_boundary_text(),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Source hashes.
    # ------------------------------------------------------------------

    source_files = [
        (
            "clinical_file",
            clinical_file,
        ),

        (
            "step27g_patient_manifest",
            step27g_patient_manifest_path,
        ),

        (
            "step27g_endpoint_specification",
            step27g_endpoint_spec_path,
        ),

        (
            "step27g_test_specification",
            step27g_test_spec_path,
        ),

        (
            "step27g_json_specification",
            step27g_spec_path,
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
    # README.
    # ------------------------------------------------------------------

    readme = f"""STEP 28A
TCGA-PRAD CLINICOPATHOLOGIC ANALYSIS TABLE

Version:
    {VERSION}

Schema:
    {SCHEMA}


ROLE
----
TABLE ASSEMBLY / ENDPOINT MATERIALIZATION ONLY.

Association statistics calculated:
    NO


V1.0.1 IMPLEMENTATION CORRECTION
--------------------------------
1. Duplicate patient_id column creation was fixed.

2. GDC fields:

       primary_gleason_grade
       secondary_gleason_grade

   are now explicitly recognized.

No PC3/PC6 association had been calculated before this correction.

No inferential design was changed.


FROZEN REPRESENTATION
---------------------
Patients:
    {len(analysis_table)}

PC3:
    patient_oriented_PC3

PC6:
    patient_oriented_PC6


CLINICAL FIELD MAPPING
----------------------
Patient ID:
    {patient_column}

Direct Grade Group:
    {direct_grade_column}

Gleason primary:
    {gleason_primary_column}

Gleason secondary:
    {gleason_secondary_column}

Gleason total:
    {gleason_total_column}

Pathologic T:
    {pathologic_t_column}


EVALUABILITY
------------
Clinical matches:
    {int(analysis_table['has_clinical_match'].sum())}

Grade Group evaluable:
    {int(analysis_table['grade_group_evaluable'].sum())}

Pathologic T evaluable:
    {int(analysis_table['pathologic_t_evaluable'].sum())}

Joint Grade + T evaluable:
    {int(analysis_table['joint_grade_t_evaluable'].sum())}


GRADE GROUP RULE
----------------
Direct authoritative Grade Group is preferred if available.

Otherwise deterministic Gleason derivation is used:

    3+3 -> GG1
    3+4 -> GG2
    4+3 -> GG3
    sum 8 -> GG4
    sum 9-10 -> GG5

Total score 7 alone is insufficient to distinguish GG2 from GG3.


PATHOLOGIC T RULE
-----------------
Canonical subcategory is retained.

Formal ordinal encoding:

    T2* -> 2
    T3* -> 3
    T4  -> 4


NEXT
----
Step28B may perform only the inferential tests frozen in Step27G:

    PC3 vs Grade Group
    PC6 vs Grade Group
    PC3 vs pathologic T
    PC6 vs pathologic T

using patient-level Spearman correlation, patient bootstrap, and
prespecified BH-FDR families.
"""

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console audit.
    # ------------------------------------------------------------------

    banner(
        "STEP 28A TABLE-ASSEMBLY AUDIT"
    )

    audit = [
        (
            "Frozen patient representation inherited",
            "YES",
        ),

        (
            "Clinical metadata joined by canonical TCGA patient ID",
            "YES",
        ),

        (
            "Duplicate patient_id column bug fixed",
            "YES",
        ),

        (
            "GDC primary_gleason_grade recognized",
            "YES",
        ),

        (
            "GDC secondary_gleason_grade recognized",
            "YES",
        ),

        (
            "Clinical field provenance recorded",
            "YES",
        ),

        (
            "Grade Group rule deterministic",
            "YES",
        ),

        (
            "Pathologic T rule deterministic",
            "YES",
        ),

        (
            "Patient inferential unit preserved",
            "YES",
        ),

        (
            "PC3 numerical scores changed",
            "NO",
        ),

        (
            "PC6 numerical scores changed",
            "NO",
        ),

        (
            "PC3/PC6 association calculated",
            "NO",
        ),

        (
            "p-values calculated",
            "NO",
        ),

        (
            "Bootstrap performed",
            "NO",
        ),

        (
            "FDR calculated",
            "NO",
        ),

        (
            "RNA read",
            "NO",
        ),

        (
            "Survival outcomes analyzed",
            "NO",
        ),

        (
            "Endpoint selected from results",
            "NO",
        ),
    ]

    for label, value in audit:

        print(
            f"{label:<62}: {value}"
        )

    banner(
        "STEP 28A: PASS"
    )

    print(
        f"[VERSION]           {VERSION}"
    )

    print(
        f"[PATIENTS]          {len(analysis_table)}"
    )

    print(
        "[CLINICAL MATCH]    "
        f"{int(analysis_table['has_clinical_match'].sum())}"
    )

    print(
        "[GRADE EVALUABLE]   "
        f"{int(analysis_table['grade_group_evaluable'].sum())}"
    )

    print(
        "[pT EVALUABLE]      "
        f"{int(analysis_table['pathologic_t_evaluable'].sum())}"
    )

    print(
        "[JOINT EVALUABLE]   "
        f"{int(analysis_table['joint_grade_t_evaluable'].sum())}"
    )

    print(
        "[GRADE SOURCE]      "
        + ", ".join(
            f"{key}={value}"
            for key, value
            in (
                analysis_table[
                    "grade_group_source"
                ]
                .value_counts(
                    dropna=False
                )
                .items()
            )
        )
    )

    print(
        "[ASSOCIATIONS]      NOT CALCULATED"
    )

    print()

    print(
        f"[ANALYSIS TABLE]    {analysis_path}"
    )

    print(
        f"[FULL AUDIT]        {full_joined_path}"
    )

    print(
        f"[AVAILABILITY]      {availability_path}"
    )

    print(
        f"[GRADE DIST]        {grade_distribution_path}"
    )

    print(
        f"[pT DIST]           {pt_distribution_path}"
    )

    print(
        f"[DUPLICATE AUDIT]   {duplicate_audit_path}"
    )

    print(
        f"[FIELD MAPPING]     {field_mapping_path}"
    )

    print(
        f"[SUMMARY]           {summary_path}"
    )

    print(
        f"[HASHES]            {hashes_path}"
    )

    print(
        f"[CLAIM BOUNDARY]    {claim_path}"
    )

    print(
        f"[README]            {readme_path}"
    )

    print()

    print(
        "[NEXT] Inspect Grade Group and pathologic-T "
        "provenance / missingness only. "
        "Do not inspect PC associations until Step28B."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen TCGA-PRAD PC3/PC6 "
            "clinicopathologic analysis table without "
            "performing association tests."
        ),

        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--clinical_file",
        type=Path,
        default=DEFAULT_CLINICAL_FILE,
        help=(
            "Authoritative Step27H2 TCGA-PRAD "
            "patient-level clinical CSV/TSV."
        ),
    )

    parser.add_argument(
        "--step27g_patient_manifest",
        type=Path,
        default=DEFAULT_STEP27G_PATIENT_MANIFEST,
    )

    parser.add_argument(
        "--step27g_endpoint_spec",
        type=Path,
        default=DEFAULT_STEP27G_ENDPOINT_SPEC,
    )

    parser.add_argument(
        "--step27g_test_spec",
        type=Path,
        default=DEFAULT_STEP27G_TEST_SPEC,
    )

    parser.add_argument(
        "--step27g_spec",
        type=Path,
        default=DEFAULT_STEP27G_SPEC,
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    # Optional explicit provenance overrides.
    parser.add_argument(
        "--patient_column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--grade_group_column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--gleason_primary_column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--gleason_secondary_column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--gleason_total_column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--pathologic_t_column",
        type=str,
        default=None,
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    execute(
        clinical_file=(
            args.clinical_file.resolve()
        ),

        step27g_patient_manifest_path=(
            args.step27g_patient_manifest.resolve()
        ),

        step27g_endpoint_spec_path=(
            args.step27g_endpoint_spec.resolve()
        ),

        step27g_test_spec_path=(
            args.step27g_test_spec.resolve()
        ),

        step27g_spec_path=(
            args.step27g_spec.resolve()
        ),

        out_dir=(
            args.out_dir.resolve()
        ),

        patient_column_override=(
            args.patient_column
        ),

        grade_group_column_override=(
            args.grade_group_column
        ),

        gleason_primary_column_override=(
            args.gleason_primary_column
        ),

        gleason_secondary_column_override=(
            args.gleason_secondary_column
        ),

        gleason_total_column_override=(
            args.gleason_total_column
        ),

        pathologic_t_column_override=(
            args.pathologic_t_column
        ),
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        banner(
            "STEP 28A: FAIL"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)