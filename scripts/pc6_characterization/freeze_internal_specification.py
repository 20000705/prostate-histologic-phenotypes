#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 23A: FREEZE PC6 INTERNAL SECOND-AXIS / MULTIDIMENSIONAL VALIDATION SPECIFICATION

VERSION: 1.2.0
SCHEMA : 23a-pc6-internal-second-axis-multidimensional-validation-freeze-schema-1.2

PRE-EXECUTION SCIENTIFIC FREEZE ONLY.

INTERNAL-FIRST RULE
-------------------
The INTERNAL discovery cohort is the sole authority for defining the PC6
histomorphologic phenotype and for primary PC3-vs-PC6 morphology distinctness.

Step23A MUST NOT read AGGC or PESO files. External cohorts are reserved for
future Step23C validation after the internal PC6 phenotype has been frozen.

PC6 INTERNAL MORPHOLOGY
-----------------------
PC6_LOW  = within-slide raw PC6 <= q20
PC6_HIGH = within-slide raw PC6 >= q80
Primary morphology effect = PC6_HIGH - PC6_LOW

The morphology family is inherited from INTERNAL Step15A:
Primary:
  SEI_eosin_stroma
  GOI_lumen_openness
  NPI_nuclear_packing
Secondary/supportive:
  CLR_nuclear_to_lumen
  CSR_nuclear_to_stroma
  dark_fraction
  hematoxylin_OD_proxy
  lumen_count
  lumen_area
  median_lumen_area
  mean_lumen_area
  lumen_component_density
  lumen_fragmentation

Normal/G3/G4/G5 is NOT the internal PC6 phenotype framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
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

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

VERSION = "1.2.0"
SCHEMA = "23a-pc6-internal-second-axis-multidimensional-validation-freeze-schema-1.2"

MEDIA = REPO_ROOT
RESULTS = RESULTS_ROOT

# INTERNAL authorities only.
STEP06_DIR = RESULTS / "06_frozen_pc_direction_specification"
STEP11_DIR = RESULTS / "11_axis_evidence"
STEP12_DIR = RESULTS / "12_pc5_pc6_principal_angle_stability"
STEP15A_DIR = RESULTS / "15a_pc3_quantitative_morphology_specification"

# Future methodology reference only; it cannot define PC6 phenotype.
STEP22A_DIR = RESULTS / "22a_cross_foundation_model_pc3_correspondence_specification"

STEP15B_CANDIDATES = [
    MEDIA / "15b_compute_pc3_quantitative_morphology.py",
    MEDIA / "15b_compute_pc3_quantitative_morphology_v1_4.py",
    MEDIA / "15b_compute_pc3_quantitative_morphology_v1_2.py",
]

OUT_DIR = RESULTS / "23a_pc6_internal_second_axis_multidimensional_validation_specification"

PRIMARY_AXIS = "PC3"
SECOND_AXIS = "PC6"
EXPLORATORY_AXIS = "PC5"

PHENOTYPE_DEFINITION_COHORT = "INTERNAL_DISCOVERY_COHORT"
PRIMARY_DISTINCTNESS_COHORT = "INTERNAL_DISCOVERY_COHORT"

PRIMARY_MORPHOLOGY_FEATURES = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "NPI_nuclear_packing",
]
SECONDARY_MORPHOLOGY_FEATURES = [
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
MORPHOLOGY_METRICS = PRIMARY_MORPHOLOGY_FEATURES + SECONDARY_MORPHOLOGY_FEATURES

LOW_QUANTILE = 0.20
HIGH_QUANTILE = 0.80
MAX_TILES_PER_SLIDE_PER_STRATUM = 10
MAX_TILES_PER_PATIENT_PER_STRATUM = 40
SAMPLING_SEED = 20260809
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260809

FORMAL_EXTERNAL_COHORTS = ["PESO", "AGGC"]
EXTERNAL_COHORT_ROLE = "FUTURE_EXTERNAL_VALIDATION_ONLY"
TCGA_ROLE = "CONTEXTUAL_OR_SUPPORTIVE_ONLY"

CROSS_FM_TARGET = "Virchow2_PC6"
CROSS_FM_MODEL = "UNI2-h"
CROSS_FM_CANDIDATES = "INHERIT_STEP22_FROZEN_PC1_PC20_CANDIDATE_SET"

PRIMARY_DISTINCTNESS = "cross_metric_signed_effect_profile_concordance"
SUPPORTIVE_DISTINCTNESS = [
    "effect_vector_cosine_similarity",
    "effect_vector_angle_degrees",
    "metric_wise_direction_agreement",
    "primary_feature_effect_profile_comparison",
]

FORMAL_LABEL = "internal_first_pc6_second_axis_multidimensional_validation"


def banner(title: str, width: int = 124) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty:\n{path}\n"
                "Use --overwrite only for intentional pre-execution rerun."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_direction(value: Any) -> str:
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "low_in_tumor": "low_in_tumor",
        "lower_in_tumor": "low_in_tumor",
        "tumor_low": "low_in_tumor",
        "low_score_predicts_tumor": "low_in_tumor",
        "high_in_tumor": "high_in_tumor",
        "higher_in_tumor": "high_in_tumor",
        "tumor_high": "high_in_tumor",
        "high_score_predicts_tumor": "high_in_tumor",
    }
    return aliases.get(s, s)


def locate_step15b() -> Optional[Path]:
    for path in STEP15B_CANDIDATES:
        if path.is_file():
            return path
    matches = sorted(MEDIA.glob("15b_compute_pc3_quantitative_morphology*.py"))
    return matches[0] if len(matches) == 1 else None


def resolve_frozen_pc6_direction(step06_dir: Path) -> Dict[str, Any]:
    if not step06_dir.is_dir():
        raise FileNotFoundError(f"Step06 directory not found:\n{step06_dir}")

    candidates: List[Dict[str, Any]] = []
    for path in sorted(step06_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue

        component_cols = [
            c for c in df.columns
            if str(c).strip().lower() in {
                "pc", "component", "principal_component", "pc_name",
                "pc_number", "component_number"
            }
        ]
        direction_cols = [
            c for c in df.columns
            if (
                "direction" in str(c).lower()
                or "tumor" in str(c).lower()
                or "sign" in str(c).lower()
            )
        ]

        for cpc in component_cols:
            text = df[cpc].astype(str).str.upper().str.replace(" ", "", regex=False)
            num = pd.to_numeric(df[cpc], errors="coerce")
            sub = df.loc[text.eq("PC6") | num.eq(6)]
            if sub.empty:
                continue
            for cdir in direction_cols:
                for raw in sub[cdir].dropna():
                    norm = normalize_direction(raw)
                    if norm in {"low_in_tumor", "high_in_tumor"}:
                        candidates.append({
                            "path": path,
                            "component_column": cpc,
                            "direction_column": cdir,
                            "raw_value": str(raw),
                            "normalized_direction": norm,
                        })

    if not candidates:
        raise RuntimeError(
            "Could not explicitly recover a frozen PC6 direction from Step06. "
            "Step23A refuses to infer PC6 sign from later outcomes."
        )

    directions = {x["normalized_direction"] for x in candidates}
    if len(directions) != 1:
        raise RuntimeError(
            "Conflicting PC6 directions in Step06:\n" +
            pd.DataFrame(candidates).to_string(index=False)
        )

    direction = next(iter(directions))
    sources = sorted({Path(x["path"]) for x in candidates}, key=lambda p: str(p))
    return {
        "component": "PC6",
        "frozen_direction": direction,
        "direction_authority": "STEP06_INTERNAL_DISCOVERY_FROZEN_DIRECTION",
        "direction_selected_from_step23_outcome": False,
        "sources": [
            {"path": str(p), "sha256": sha256_file(p)}
            for p in sources
        ],
    }


def upstream_inventory() -> pd.DataFrame:
    step15b = locate_step15b()
    return pd.DataFrame([
        {
            "role": "step06_frozen_pc6_direction",
            "cohort_scope": "INTERNAL",
            "path": str(STEP06_DIR),
            "required": True,
            "exists": STEP06_DIR.exists(),
            "authority_role": "FROZEN_DIRECTION_ONLY",
        },
        {
            "role": "step11_axis_evidence",
            "cohort_scope": "INTERNAL",
            "path": str(STEP11_DIR),
            "required": False,
            "exists": STEP11_DIR.exists(),
            "authority_role": "UPSTREAM_AXIS_EVIDENCE",
        },
        {
            "role": "step12_pc5_pc6_stability",
            "cohort_scope": "INTERNAL",
            "path": str(STEP12_DIR),
            "required": False,
            "exists": STEP12_DIR.exists(),
            "authority_role": "UPSTREAM_STABILITY_EVIDENCE",
        },
        {
            "role": "step15a_internal_morphology_specification",
            "cohort_scope": "INTERNAL",
            "path": str(STEP15A_DIR),
            "required": True,
            "exists": STEP15A_DIR.exists(),
            "authority_role": "FROZEN_INTERNAL_MORPHOLOGY_DEFINITION",
        },
        {
            "role": "step15b_internal_morphology_implementation",
            "cohort_scope": "INTERNAL",
            "path": "" if step15b is None else str(step15b),
            "required": False,
            "exists": bool(step15b is not None and step15b.is_file()),
            "authority_role": "FUTURE_23B_EXECUTION_IMPLEMENTATION",
        },
        {
            "role": "step22a_cross_fm_framework",
            "cohort_scope": "FUTURE_METHOD_REFERENCE",
            "path": str(STEP22A_DIR),
            "required": False,
            "exists": STEP22A_DIR.exists(),
            "authority_role": "FUTURE_23D_METHOD_FRAMEWORK_ONLY",
        },
    ])


def role_table(pc6_direction: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "axis": "PC3",
            "formal_role": "PRIMARY_INTERPRETED_AXIS",
            "step23_role": "REFERENCE_AXIS",
            "frozen_direction": "INHERIT_UPSTREAM",
            "may_replace_pc6": False,
            "may_be_replaced_by_pc6": False,
        },
        {
            "axis": "PC6",
            "formal_role": "SECOND_REPRODUCIBLE_AXIS_TARGET",
            "step23_role": "FORMAL_INTERNAL_VALIDATION_TARGET",
            "frozen_direction": pc6_direction,
            "may_replace_pc3": False,
            "selected_from_step23_outcome": False,
        },
        {
            "axis": "PC5",
            "formal_role": "EXPLORATORY",
            "step23_role": "NOT_PROMOTED",
            "frozen_direction": "NOT_APPLICABLE",
            "may_replace_pc6": False,
            "selected_from_step23_outcome": False,
        },
    ])


def metric_table() -> pd.DataFrame:
    rows = []
    for feature in PRIMARY_MORPHOLOGY_FEATURES:
        rows.append({
            "morphology_metric": feature,
            "endpoint_tier": "PRIMARY",
            "included_in_frozen_pc6_family": True,
            "authority": "INTERNAL_STEP15A",
            "selected_from_pc6_outcome": False,
            "selected_from_external_outcome": False,
        })
    for feature in SECONDARY_MORPHOLOGY_FEATURES:
        rows.append({
            "morphology_metric": feature,
            "endpoint_tier": "SECONDARY_SUPPORTIVE",
            "included_in_frozen_pc6_family": True,
            "authority": "INTERNAL_STEP15A",
            "selected_from_pc6_outcome": False,
            "selected_from_external_outcome": False,
        })
    return pd.DataFrame(rows)


def stratum_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "stratum": "PC6_LOW",
            "definition": "within-slide raw PC6 <= q20",
            "quantile": LOW_QUANTILE,
            "analysis_role": "PRIMARY",
            "cohort": PHENOTYPE_DEFINITION_COHORT,
            "selected_from_outcome": False,
        },
        {
            "stratum": "PC6_HIGH",
            "definition": "within-slide raw PC6 >= q80",
            "quantile": HIGH_QUANTILE,
            "analysis_role": "PRIMARY",
            "cohort": PHENOTYPE_DEFINITION_COHORT,
            "selected_from_outcome": False,
        },
    ])


def sampling_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"parameter": "low_quantile", "value": LOW_QUANTILE},
        {"parameter": "high_quantile", "value": HIGH_QUANTILE},
        {"parameter": "max_tiles_per_slide_per_stratum", "value": MAX_TILES_PER_SLIDE_PER_STRATUM},
        {"parameter": "max_tiles_per_patient_per_stratum", "value": MAX_TILES_PER_PATIENT_PER_STRATUM},
        {"parameter": "sampling_seed", "value": SAMPLING_SEED},
        {"parameter": "bootstrap_resamples", "value": BOOTSTRAP_RESAMPLES},
        {"parameter": "bootstrap_seed", "value": BOOTSTRAP_SEED},
    ])


def cohort_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cohort": "INTERNAL_DISCOVERY_COHORT",
            "formal_role": "PHENOTYPE_DEFINITION_AND_PRIMARY_DISTINCTNESS",
            "read_in_step23a": True,
            "may_define_pc6_phenotype": True,
            "may_change_pc6_sign": False,
        },
        {
            "cohort": "PESO",
            "formal_role": EXTERNAL_COHORT_ROLE,
            "read_in_step23a": False,
            "may_define_pc6_phenotype": False,
            "may_change_pc6_sign": False,
        },
        {
            "cohort": "AGGC",
            "formal_role": EXTERNAL_COHORT_ROLE,
            "read_in_step23a": False,
            "may_define_pc6_phenotype": False,
            "may_change_pc6_sign": False,
        },
        {
            "cohort": "TCGA-PRAD",
            "formal_role": TCGA_ROLE,
            "read_in_step23a": False,
            "may_define_pc6_phenotype": False,
            "may_change_pc6_sign": False,
        },
    ])


def distinctness_table() -> pd.DataFrame:
    rows = [{
        "endpoint_id": "PRIMARY_PC3_PC6_SIGNED_EFFECT_PROFILE_CONCORDANCE",
        "endpoint_role": "PRIMARY_DISTINCTNESS_SUMMARY",
        "definition": PRIMARY_DISTINCTNESS,
        "unit": "complete_frozen_morphology_effect_profile",
        "feature_scope": "ALL_13_INTERNAL_STEP15_FEATURES",
        "cohort": PRIMARY_DISTINCTNESS_COHORT,
        "uses_significance_filter": False,
        "allows_metric_selection": False,
    }]
    for endpoint in SUPPORTIVE_DISTINCTNESS:
        rows.append({
            "endpoint_id": endpoint.upper(),
            "endpoint_role": "SUPPORTIVE_DISTINCTNESS",
            "definition": endpoint,
            "unit": "complete_frozen_morphology_effect_profile",
            "feature_scope": "ALL_13_INTERNAL_STEP15_FEATURES",
            "cohort": PRIMARY_DISTINCTNESS_COHORT,
            "uses_significance_filter": False,
            "allows_metric_selection": False,
        })
    return pd.DataFrame(rows)


def cross_fm_table() -> pd.DataFrame:
    return pd.DataFrame([
        ("target_axis", CROSS_FM_TARGET, "FUTURE_23D"),
        ("comparison_model", CROSS_FM_MODEL, "FUTURE_23D"),
        ("candidate_axis_set", CROSS_FM_CANDIDATES, "FUTURE_23D"),
        ("alignment", "EXACT_TILE_COORDINATES_ONLY", "FUTURE_23D"),
        ("primary_similarity", "SLIDE_LEVEL_SPEARMAN", "FUTURE_23D_PRIMARY"),
        ("correspondence_strength", "ABSOLUTE_SPEARMAN", "FUTURE_23D_PRIMARY"),
        ("tumor_auc_for_component_selection", "PROHIBITED", "GUARDRAIL"),
    ], columns=["item", "value", "role"])


def claim_ceiling_text() -> str:
    return """STEP23 PC6 CLAIM CEILING\n\nALLOWED IF SUPPORTED\n--------------------\n1. PC6 defines a second reproducible histologic representation axis.\n2. PC3 and PC6 show distinguishable frozen quantitative morphology fingerprints.\n3. PC3 and PC6 support a multidimensional organization of prostate histology.\n4. A stronger cross-FM multidimensionality claim requires a later distinct PC6 counterpart.\n\nNOT ALLOWED\n-----------\n1. PC6 is superior to PC3.\n2. PCA orthogonality alone proves biological independence.\n3. PC3/PC6 explain the complete latent space.\n4. PC6 defines a causal/molecular mechanism without direct evidence.\n5. PC6 was prospectively discovered in Step23.\n6. PC5 is promoted because PC6 gives inconvenient results.\n7. AGGC/PESO may define, refine, or rescue the internal PC6 phenotype.\n8. Normal/G3/G4/G5 may substitute for internal PC6 high/low morphology.\n"""


def readme_text(direction: str) -> str:
    return f"""STEP 23A\nPC6 INTERNAL SECOND-AXIS / MULTIDIMENSIONAL VALIDATION SPECIFICATION\n\nVersion: {VERSION}\nSchema: {SCHEMA}\n\nINTERNAL-FIRST\n--------------\nPhenotype authority: INTERNAL DISCOVERY COHORT ONLY\nAGGC read: NO\nPESO read: NO\nNormal/G3/G4/G5 framework: NO\nInternal morphology authority: Step15A\n\nPC6 direction\n-------------\n{direction}\nAuthority: Step06 internal discovery-frozen direction\n\nPC6 morphology strata\n--------------------\nPC6_LOW  = within-slide raw PC6 <= q20\nPC6_HIGH = within-slide raw PC6 >= q80\nPrimary effect = PC6_HIGH - PC6_LOW\n\nPrimary morphology\n------------------\n{chr(10).join('  ' + x for x in PRIMARY_MORPHOLOGY_FEATURES)}\n\nSecondary/supportive morphology\n-------------------------------\n{chr(10).join('  ' + x for x in SECONDARY_MORPHOLOGY_FEATURES)}\n\nSampling\n--------\nMax tiles/slide/stratum: {MAX_TILES_PER_SLIDE_PER_STRATUM}\nMax tiles/patient/stratum: {MAX_TILES_PER_PATIENT_PER_STRATUM}\nSampling seed: {SAMPLING_SEED}\n\nPC3-vs-PC6 distinctness\n----------------------\nPrimary: {PRIMARY_DISTINCTNESS}\nSupportive: {', '.join(SUPPORTIVE_DISTINCTNESS)}\n\nExternal validation\n-------------------\nPESO + AGGC are reserved for Step23C only.\nThey cannot define/refine/rescue the internal phenotype.\n\nNext\n----\n23B: INTERNAL PC6 high-vs-low quantitative morphology + INTERNAL PC3-vs-PC6 profile distinctness\n23C: external validation in PESO/AGGC\n23D: PC6 cross-FM correspondence\n23E: visualization only\n"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step06-dir", type=Path, default=STEP06_DIR)
    p.add_argument("--output", type=Path, default=OUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    banner("STEP 23A: FREEZE PC6 INTERNAL SECOND-AXIS / MULTIDIMENSIONAL VALIDATION SPECIFICATION")
    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP06]        {args.step06_dir}")
    print(f"[STEP15A]       {STEP15A_DIR}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          PRE-EXECUTION SCIENTIFIC FREEZE ONLY")
    print("[PRIMARY AXIS]  PC3")
    print("[SECOND AXIS]   PC6")
    print("[PC5]           EXPLORATORY / NOT PROMOTED")
    print("[PHENOTYPE]     INTERNAL DISCOVERY COHORT ONLY")
    print("[MORPH AUTH]    INTERNAL STEP15A")
    print("[AGGC READ]     NONE")
    print("[PESO READ]     NONE")
    print("[GLEASON STATE] NONE")
    print("[PC SCREEN]     NONE")
    print("[PCA REFIT]     NONE")
    print("[SIGN TUNING]   NONE")

    prepare_output(args.output, args.overwrite)

    banner("VALIDATE INTERNAL DISCOVERY-FROZEN PC6 DIRECTION")
    pc6 = resolve_frozen_pc6_direction(args.step06_dir)
    print(f"Frozen PC6 direction        : {pc6['frozen_direction']}")
    print(f"Direction authority         : {pc6['direction_authority']}")
    print("Direction selected from Step23 outcome : NO")

    inventory = upstream_inventory()
    banner("INTERNAL AUTHORITY INVENTORY")
    print(inventory.to_string(index=False))

    if not STEP15A_DIR.exists():
        raise RuntimeError("Required INTERNAL Step15A morphology specification is missing.")

    plan = {
        "version": VERSION,
        "schema": SCHEMA,
        "formal_label": FORMAL_LABEL,
        "analysis_role": "pre_execution_internal_pc6_second_axis_multidimensional_validation_freeze",
        "axis_roles": {
            "PC3": "PRIMARY_INTERPRETED_AXIS",
            "PC6": "SECOND_AXIS_FORMAL_INTERNAL_VALIDATION_TARGET",
            "PC5": "EXPLORATORY_NOT_PROMOTED",
        },
        "pc6": {
            "component": "PC6",
            "frozen_direction": pc6["frozen_direction"],
            "direction_authority": pc6["direction_authority"],
            "direction_selected_from_step23_outcome": False,
            "prospective_new_axis_discovery_claim": False,
        },
        "cohort_hierarchy": {
            "phenotype_definition_cohort": PHENOTYPE_DEFINITION_COHORT,
            "primary_distinctness_cohort": PRIMARY_DISTINCTNESS_COHORT,
            "step23a_external_files_read": False,
            "external_validation_cohorts": FORMAL_EXTERNAL_COHORTS,
            "external_role": EXTERNAL_COHORT_ROLE,
            "external_may_define_or_refine_phenotype": False,
        },
        "internal_pc6_morphology": {
            "low_stratum": "within-slide raw PC6 <= q20",
            "high_stratum": "within-slide raw PC6 >= q80",
            "primary_effect": "PC6_HIGH_MINUS_PC6_LOW",
            "primary_features": PRIMARY_MORPHOLOGY_FEATURES,
            "secondary_supportive_features": SECONDARY_MORPHOLOGY_FEATURES,
            "n_features": len(MORPHOLOGY_METRICS),
            "authority": "INTERNAL_STEP15A",
            "normal_g3_g4_g5_used": False,
        },
        "sampling": {
            "max_tiles_per_slide_per_stratum": MAX_TILES_PER_SLIDE_PER_STRATUM,
            "max_tiles_per_patient_per_stratum": MAX_TILES_PER_PATIENT_PER_STRATUM,
            "sampling_seed": SAMPLING_SEED,
        },
        "pc3_pc6_distinctness": {
            "cohort": PRIMARY_DISTINCTNESS_COHORT,
            "primary_endpoint": PRIMARY_DISTINCTNESS,
            "supportive_endpoints": SUPPORTIVE_DISTINCTNESS,
            "feature_scope": MORPHOLOGY_METRICS,
            "significance_filtering_for_profile": False,
            "definition_by_p_value_difference": False,
            "metric_selection_allowed": False,
        },
        "bootstrap": {
            "resampling_unit": "slide",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "external_transfer": {
            "formal_cohorts": FORMAL_EXTERNAL_COHORTS,
            "role": EXTERNAL_COHORT_ROLE,
            "read_in_step23a": False,
            "external_phenotype_definition_allowed": False,
        },
        "cross_fm": {
            "planned_step": "23D",
            "target": CROSS_FM_TARGET,
            "comparison_model": CROSS_FM_MODEL,
            "candidate_set": CROSS_FM_CANDIDATES,
            "exact_coordinate_alignment": True,
            "primary_similarity": "SLIDE_LEVEL_SPEARMAN",
            "correspondence_strength": "ABSOLUTE_SPEARMAN",
            "tumor_auc_for_component_selection": False,
        },
        "prohibited": {
            "pca_refit": True,
            "pc1_pc8_rescreening": True,
            "pc6_sign_tuning": True,
            "pc5_promotion": True,
            "morphology_metric_selection": True,
            "gleason_state_framework_for_internal_pc6": True,
            "aggc_phenotype_definition": True,
            "peso_phenotype_definition": True,
            "best_external_cohort_selection": True,
            "tumor_auc_based_cross_fm_component_selection": True,
            "manual_case_selection_before_quantitative_freeze": True,
        },
        "planned_steps": {
            "23B": "INTERNAL PC6 high-vs-low morphology + INTERNAL PC3-vs-PC6 profile distinctness",
            "23C": "external validation in PESO/AGGC of internally frozen PC6 phenotype",
            "23D": "PC6 cross-FM correspondence + counterpart separation",
            "23E": "visualization only",
        },
    }

    files = {
        "plan": args.output / "pc6_internal_second_axis_analysis_plan.json",
        "roles": args.output / "pc6_scientific_role_specification.csv",
        "metrics": args.output / "pc6_internal_morphology_metric_specification.csv",
        "strata": args.output / "pc6_internal_score_stratum_specification.csv",
        "sampling": args.output / "pc6_internal_sampling_specification.csv",
        "cohorts": args.output / "pc6_cohort_hierarchy_specification.csv",
        "cross_fm": args.output / "pc6_future_cross_fm_specification.csv",
        "distinctness": args.output / "pc3_pc6_internal_distinctness_endpoint_specification.csv",
        "claim": args.output / "pc6_claim_ceiling.txt",
        "direction": args.output / "pc6_frozen_direction_snapshot.csv",
        "inventory": args.output / "internal_upstream_authority_inventory.csv",
        "hashes": args.output / "source_hashes.csv",
        "metadata": args.output / "analysis_metadata.json",
        "readme": args.output / "README.txt",
    }

    save_json(files["plan"], plan)
    role_table(pc6["frozen_direction"]).to_csv(files["roles"], index=False)
    metric_table().to_csv(files["metrics"], index=False)
    stratum_table().to_csv(files["strata"], index=False)
    sampling_table().to_csv(files["sampling"], index=False)
    cohort_table().to_csv(files["cohorts"], index=False)
    cross_fm_table().to_csv(files["cross_fm"], index=False)
    distinctness_table().to_csv(files["distinctness"], index=False)
    files["claim"].write_text(claim_ceiling_text(), encoding="utf-8")
    inventory.to_csv(files["inventory"], index=False)

    pd.DataFrame([{
        "component": "PC6",
        "frozen_direction": pc6["frozen_direction"],
        "direction_authority": pc6["direction_authority"],
        "direction_selected_from_step23_outcome": False,
        "source_paths": "|".join(x["path"] for x in pc6["sources"]),
    }]).to_csv(files["direction"], index=False)

    # Hash only INTERNAL authorities plus future Step22 method reference.
    hash_rows = [
        {
            "role": "step06_internal_pc6_direction_authority",
            "cohort_scope": "INTERNAL",
            "path": x["path"],
            "sha256": x["sha256"],
        }
        for x in pc6["sources"]
    ]

    candidates = [
        ("step11_internal_axis_evidence_metadata", STEP11_DIR / "analysis_metadata.json", "INTERNAL"),
        ("step12_internal_pc5_pc6_stability_metadata", STEP12_DIR / "analysis_metadata.json", "INTERNAL"),
        ("step15a_internal_morphology_metadata", STEP15A_DIR / "analysis_metadata.json", "INTERNAL"),
        ("step15a_internal_morphology_plan", STEP15A_DIR / "pc3_quantitative_morphology_plan.json", "INTERNAL"),
        ("step22a_future_cross_fm_framework", STEP22A_DIR / "cross_fm_pc3_analysis_plan.json", "FUTURE_METHOD_REFERENCE"),
    ]
    for role, path, scope in candidates:
        if path.is_file():
            hash_rows.append({
                "role": role,
                "cohort_scope": scope,
                "path": str(path),
                "sha256": sha256_file(path),
            })

    step15b = locate_step15b()
    if step15b is not None and step15b.is_file():
        hash_rows.append({
            "role": "step15b_internal_morphology_implementation",
            "cohort_scope": "INTERNAL",
            "path": str(step15b),
            "sha256": sha256_file(step15b),
        })

    # Hard external-cohort firewall.
    bad = [
        row for row in hash_rows
        if "aggc" in str(row["path"]).lower() or "peso" in str(row["path"]).lower()
    ]
    if bad:
        raise RuntimeError(
            "Step23A INTERNAL-only firewall violated by external source hashes:\n" +
            pd.DataFrame(bad).to_string(index=False)
        )
    pd.DataFrame(hash_rows).to_csv(files["hashes"], index=False)

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scientific_role": "pre_execution_internal_pc6_second_axis_multidimensional_validation_freeze",
        "formal_label": FORMAL_LABEL,
        "primary_axis": PRIMARY_AXIS,
        "second_axis": SECOND_AXIS,
        "exploratory_axis": EXPLORATORY_AXIS,
        "pc6_frozen_direction": pc6["frozen_direction"],
        "phenotype_definition_cohort": PHENOTYPE_DEFINITION_COHORT,
        "primary_distinctness_cohort": PRIMARY_DISTINCTNESS_COHORT,
        "n_frozen_morphology_metrics": len(MORPHOLOGY_METRICS),
        "freeze_only_audit": {
            "pc6_outcome_values_read": False,
            "external_pc6_outcomes_read": False,
            "aggc_files_read": False,
            "peso_files_read": False,
            "external_morphology_specification_inherited": False,
            "normal_g3_g4_g5_framework_used": False,
            "step15a_internal_morphology_specification_used": True,
            "pc6_phenotype_definition_internal_only": True,
            "pca_refit": False,
            "pc_screening": False,
            "pc6_sign_tuning": False,
            "pc5_promotion": False,
            "morphology_metric_selection": False,
            "best_external_cohort_selection": False,
            "tumor_auc_cross_fm_selection": False,
            "manual_case_selection": False,
            "prospective_new_axis_discovery_claim": False,
            "second_axis_hypothesis_frozen": True,
            "pc3_pc6_internal_distinctness_endpoint_frozen": True,
        },
    }
    save_json(files["metadata"], metadata)
    files["readme"].write_text(readme_text(pc6["frozen_direction"]), encoding="utf-8")

    banner("FROZEN INTERNAL PC6 SECOND-AXIS CONTRACT")
    for k, v in [
        ("Primary interpreted axis", "PC3"),
        ("Second-axis validation target", "PC6"),
        ("Frozen PC6 direction", pc6["frozen_direction"]),
        ("Phenotype-definition cohort", PHENOTYPE_DEFINITION_COHORT),
        ("Internal morphology authority", "STEP15A INTERNAL"),
        ("PC6 low stratum", "<= within-slide q20"),
        ("PC6 high stratum", ">= within-slide q80"),
        ("Primary PC6 morphology effect", "PC6_HIGH - PC6_LOW"),
        ("Frozen morphology family", "13 INTERNAL Step15 metrics"),
        ("External cohorts", "PESO + AGGC = FUTURE VALIDATION ONLY"),
        ("Normal/G3/G4/G5 framework", "NOT USED"),
        ("Primary distinctness endpoint", PRIMARY_DISTINCTNESS),
    ]:
        print(f"{k:<46}: {v}")

    banner("FREEZE-ONLY AUDIT")
    for item, status in [
        ("PC6 outcome values read", "NO"),
        ("AGGC files read", "NO"),
        ("PESO files read", "NO"),
        ("External morphology specification inherited", "NO"),
        ("Internal Step15A morphology specification used", "YES"),
        ("Normal/G3/G4/G5 state framework used", "NO"),
        ("PC6 phenotype defined internally", "YES"),
        ("PCA refit", "NO"),
        ("PC1-PC8 re-screening", "NO"),
        ("PC6 sign tuning", "NO"),
        ("PC5 promotion", "NO"),
        ("Morphology metric selection", "NO"),
        ("Best external cohort selection", "NO"),
        ("Tumor-AUC-based cross-FM matching", "NO"),
        ("Manual case selection", "NO"),
        ("Prospective new-PC6 claim", "NO"),
        ("PC6 second-axis hypothesis frozen", "YES"),
        ("PC3-vs-PC6 internal distinctness endpoint frozen", "YES"),
        ("Claim ceiling frozen", "YES"),
    ]:
        print(f"{item:<58}: {status}")

    print("\n[OK] Saved:")
    for p in files.values():
        if not p.exists():
            raise RuntimeError(f"Expected output missing: {p}")
        print(f"  {p}")

    print()
    print("[DONE] Step23A INTERNAL PC6 second-axis specification frozen.")
    print(
        "[NEXT] Step23B must use INTERNAL discovery cohort only: "
        "PC6_LOW vs PC6_HIGH quantitative morphology + INTERNAL PC3-vs-PC6 morphology-profile distinctness."
    )


if __name__ == "__main__":
    main()
