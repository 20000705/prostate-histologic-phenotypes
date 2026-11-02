#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 23B: INTERNAL PC6 QUANTITATIVE MORPHOLOGY MEASUREMENT

VERSION: 1.1.0
SCHEMA : 23b-internal-pc6-quantitative-morphology-measurement-schema-1.1

MEASUREMENT ONLY.

Canonical cohort authority:
    Discovery embedding manifest configured through project_paths.py
    (DISCOVERY_EMBEDDINGS / "embedding_manifest.csv").

The formal cohort MUST be exactly 65 slides / 15 patients.
Filesystem scanning is prohibited for cohort definition.

PC6 strata (frozen in Step23A):
    PC6_LOW  = within-slide raw PC6 <= q20
    PC6_HIGH = within-slide raw PC6 >= q80

Sampling (frozen in Step23A):
    <=10 tiles / slide / stratum
    <=40 tiles / patient / stratum

Morphology measurement:
    Reuses the exact INTERNAL Step15B v1.4.0 measurement/QC engine.
    Step23B defines no morphology thresholds or image-processing rules.

Physical tile identity:
    resolved WSI path + x + y + patch_size

NO INFERENCE IN STEP23B:
    no Wilcoxon, Spearman outcome analysis, bootstrap CI, p-values, FDR,
    or PC3-vs-PC6 distinctness. Those belong in Step23C.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "1.1.1"
SCHEMA = "23b-internal-pc6-quantitative-morphology-measurement-schema-1.1.1"

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = DATA_ROOT / 'discovery'
RESULTS = RESULTS_ROOT

# Canonical discovery-cohort authority used by the original PCA/tumor-association pipeline.
CANONICAL_MANIFEST = BASE / "wsi_embeddings" / "embedding_manifest.csv"
CANONICAL_PCA_DIR = BASE / "wsi_embeddings" / "pca_global"
CANONICAL_PCA_MANIFEST = CANONICAL_PCA_DIR / "pca_manifest.csv"

STEP23A_DIR = RESULTS / "23a_pc6_internal_second_axis_multidimensional_validation_specification"
STEP23A_PLAN = STEP23A_DIR / "pc6_internal_second_axis_analysis_plan.json"
STEP23A_METRICS = STEP23A_DIR / "pc6_internal_morphology_metric_specification.csv"
STEP23A_STRATA = STEP23A_DIR / "pc6_internal_score_stratum_specification.csv"
STEP23A_SAMPLING = STEP23A_DIR / "pc6_internal_sampling_specification.csv"
STEP23A_DIRECTION = STEP23A_DIR / "pc6_frozen_direction_snapshot.csv"
STEP23A_METADATA = STEP23A_DIR / "analysis_metadata.json"

STEP15A_DIR = RESULTS / "15a_pc3_quantitative_morphology_specification"
STEP15B_CANDIDATES = [
    SCRIPT_DIR / "15b_compute_pc3_quantitative_morphology.py",
    SCRIPT_DIR / "15b_compute_pc3_quantitative_morphology_v1_4.py",
    SCRIPT_DIR / "15b_compute_pc3_quantitative_morphology_v1_2.py",
]

OUT_DIR = RESULTS / "23b_internal_pc6_quantitative_morphology_measurement"

EXPECTED_SLIDES = 65
EXPECTED_PATIENTS = 15
PC6_INDEX = 5  # zero-based

PRIMARY_FEATURES = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "NPI_nuclear_packing",
]
SECONDARY_FEATURES = [
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
SENSITIVITY_FEATURES = [
    "GOI_lumen_openness_border_excluded",
    "NPI_nuclear_packing_border_excluded_denominator",
    "SEI_eosin_stroma_border_excluded_denominator",
    "CLR_nuclear_to_lumen_border_excluded",
    "lumen_count_border_excluded",
    "lumen_area_border_excluded",
    "median_lumen_area_border_excluded",
    "mean_lumen_area_border_excluded",
    "lumen_component_density_border_excluded",
    "lumen_fragmentation_border_excluded",
    "border_connected_open_fraction_all",
]
ALL_FEATURES = PRIMARY_FEATURES + SECONDARY_FEATURES + SENSITIVITY_FEATURES

OUT_COHORT = OUT_DIR / "canonical_internal_cohort_snapshot.csv"
OUT_COHORT_AUDIT = OUT_DIR / "canonical_internal_cohort_audit.csv"
OUT_THRESHOLDS = OUT_DIR / "pc6_slide_score_thresholds.csv"
OUT_SAMPLE = OUT_DIR / "pc6_sampling_manifest.csv"
OUT_PRE_SAMPLE_AUDIT = OUT_DIR / "pc6_sampling_audit_pre_measurement.csv"
OUT_PHYSICAL_DEDUP = OUT_DIR / "pc6_physical_tile_dedup_audit.csv"
OUT_TILE = OUT_DIR / "pc6_quantitative_morphology_tile_features.csv"
OUT_QC_FAILURES = OUT_DIR / "pc6_quantitative_morphology_qc_failures.csv"
OUT_SLIDE = OUT_DIR / "pc6_quantitative_morphology_slide_summary.csv"
OUT_PATIENT = OUT_DIR / "pc6_quantitative_morphology_patient_summary.csv"
OUT_COVERAGE = OUT_DIR / "pc6_quantitative_morphology_feature_coverage.csv"
OUT_AUDIT = OUT_DIR / "pc6_quantitative_morphology_sampling_audit.csv"
OUT_RESOLUTION = OUT_DIR / "pc6_quantitative_morphology_resolution_manifest.csv"
OUT_RESOLUTION_AUDIT = OUT_DIR / "pc6_quantitative_morphology_resolution_method_audit.csv"
OUT_EQUIVALENCE = OUT_DIR / "pc6_quantitative_morphology_pixel_source_equivalence_audit.csv"
OUT_RESOLUTION_DUP = OUT_DIR / "pc6_quantitative_morphology_resolution_duplicate_audit.csv"
OUT_HASHES = OUT_DIR / "source_hashes.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"


def banner(title: str, width: int = 132) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input ({label}):\n{path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required directory ({label}):\n{path}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is non-empty:\n{path}\n"
                "Use --overwrite only for intentional re-execution."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_path(value: Any) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return Path(s)


def resolve_manifest_input_path(value: Any, manifest_dir: Path) -> Optional[Path]:
    """Resolve historical manifest paths deterministically.

    The canonical internal manifest stores paths such as
    ``wsi_embeddings/<prefix>_coords.npy`` relative to the virchow-histo BASE,
    not relative to MedIA_paper.  This function only resolves file location; it
    never changes cohort membership or sample selection.
    """
    raw = clean_path(value)
    if raw is None:
        return None
    if raw.is_absolute():
        return raw

    candidates = [
        BASE / raw,
        manifest_dir / raw,
        SCRIPT_DIR / raw,
        Path.cwd() / raw,
    ]
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    # Preserve the most likely historical interpretation for a useful error.
    return BASE / raw


def resolved_path(value: Any) -> str:
    p = clean_path(value)
    if p is None:
        return ""
    try:
        return str(p.resolve(strict=False))
    except Exception:
        return str(p)


def patient_from_prefix(prefix: str) -> str:
    return str(prefix).split("_")[0]


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = str(base_seed) + "||" + "||".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2 ** 32)


def deterministic_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=n, replace=False))
    return df.iloc[idx].copy()


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_prefix_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prefix" in out.columns:
        return out
    for c in ["slide_id", "slide", "slide_name"]:
        if c in out.columns:
            out["prefix"] = out[c].astype(str)
            return out
    raise RuntimeError("Canonical manifest lacks a prefix/slide_id column.")


def load_step23a_contract() -> Dict[str, Any]:
    for p in [
        STEP23A_PLAN, STEP23A_METRICS, STEP23A_STRATA,
        STEP23A_SAMPLING, STEP23A_DIRECTION, STEP23A_METADATA,
    ]:
        require_file(p, f"Step23A authority: {p.name}")

    plan = load_json(STEP23A_PLAN)
    meta = load_json(STEP23A_METADATA)
    if str(plan.get("version")) != "1.2.0":
        raise RuntimeError(f"Step23B requires Step23A v1.2.0; observed={plan.get('version')}")
    if str(meta.get("phenotype_definition_cohort", "")) != "INTERNAL_DISCOVERY_COHORT":
        raise RuntimeError("Step23A phenotype authority is no longer INTERNAL discovery cohort.")

    metrics = pd.read_csv(STEP23A_METRICS, low_memory=False)
    strata = pd.read_csv(STEP23A_STRATA, low_memory=False)
    sampling = pd.read_csv(STEP23A_SAMPLING, low_memory=False)
    direction = pd.read_csv(STEP23A_DIRECTION, low_memory=False)

    if len(direction) != 1:
        raise RuntimeError("Step23A direction snapshot must contain exactly one row.")
    pc6_direction = str(direction.iloc[0]["frozen_direction"])
    if pc6_direction != "high_in_tumor":
        raise RuntimeError(
            "Unexpected frozen PC6 direction; Step23B refuses sign tuning. "
            f"Observed={pc6_direction}"
        )

    feature_col = "morphology_metric" if "morphology_metric" in metrics.columns else "feature_name"
    frozen_features = metrics[feature_col].astype(str).tolist()
    expected = PRIMARY_FEATURES + SECONDARY_FEATURES
    if frozen_features != expected:
        raise RuntimeError(
            "Step23A morphology family/order changed.\n"
            f"Expected={expected}\nObserved={frozen_features}"
        )

    strata_map = {str(r["stratum"]): float(r["quantile"]) for _, r in strata.iterrows()}
    if set(strata_map) != {"PC6_LOW", "PC6_HIGH"}:
        raise RuntimeError(f"Unexpected frozen PC6 strata: {strata_map}")

    smap = {str(r["parameter"]): float(r["value"]) for _, r in sampling.iterrows()}
    for key in [
        "max_tiles_per_slide_per_stratum",
        "max_tiles_per_patient_per_stratum",
        "sampling_seed",
    ]:
        if key not in smap:
            raise RuntimeError(f"Step23A sampling specification missing {key}")

    return {
        "plan": plan,
        "metadata": meta,
        "features": frozen_features,
        "low_q": strata_map["PC6_LOW"],
        "high_q": strata_map["PC6_HIGH"],
        "max_slide": int(smap["max_tiles_per_slide_per_stratum"]),
        "max_patient": int(smap["max_tiles_per_patient_per_stratum"]),
        "sampling_seed": int(smap["sampling_seed"]),
        "pc6_direction": pc6_direction,
    }


def locate_step15b() -> Path:
    for p in STEP15B_CANDIDATES:
        if p.is_file():
            return p
    matches = sorted(SCRIPT_DIR.glob("15b_compute_pc3_quantitative_morphology*.py"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("Could not uniquely locate INTERNAL Step15B morphology engine.")


def import_step15b(path: Path):
    spec = importlib.util.spec_from_file_location("step15b_internal_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Step15B: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = str(getattr(module, "VERSION", ""))
    if observed != "1.4.0":
        raise RuntimeError(f"Step23B requires hardened Step15B v1.4.0; observed={observed}")
    required = [
        "load_frozen_specification", "standardize_sampling",
        "build_resolution_indexes", "process_samples",
        "build_resolution_method_audit",
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise RuntimeError(f"Step15B missing required functions: {missing}")
    return module


def load_canonical_internal_cohort() -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_file(CANONICAL_MANIFEST, "canonical INTERNAL embedding manifest")
    require_dir(CANONICAL_PCA_DIR, "canonical INTERNAL PCA directory")

    m = normalize_prefix_column(pd.read_csv(CANONICAL_MANIFEST, low_memory=False))
    m["prefix"] = m["prefix"].astype(str).str.strip()
    if m["prefix"].duplicated().any():
        dup = m.loc[m["prefix"].duplicated(False)]
        raise RuntimeError(
            "Canonical embedding manifest itself contains duplicate prefixes; "
            "Step23B refuses to resolve them from filesystem state:\n"
            + dup.to_string(index=False)
        )

    if "patient_id" not in m.columns:
        m["patient_id"] = m["prefix"].map(patient_from_prefix)
    else:
        m["patient_id"] = m["patient_id"].astype(str).str.strip()

    n_slides = len(m)
    n_patients = int(m["patient_id"].nunique())
    aaa0044_present = "aaa0044" in set(m["patient_id"])

    audit = pd.DataFrame([
        {"audit_item": "n_slides", "observed": n_slides, "expected": EXPECTED_SLIDES, "pass": n_slides == EXPECTED_SLIDES},
        {"audit_item": "n_patients", "observed": n_patients, "expected": EXPECTED_PATIENTS, "pass": n_patients == EXPECTED_PATIENTS},
        {"audit_item": "aaa0044_absent", "observed": not aaa0044_present, "expected": True, "pass": not aaa0044_present},
        {"audit_item": "filesystem_discovery_used", "observed": False, "expected": False, "pass": True},
    ])

    if n_slides != EXPECTED_SLIDES or n_patients != EXPECTED_PATIENTS:
        raise RuntimeError(
            "CANONICAL INTERNAL COHORT VIOLATION.\n"
            f"Expected {EXPECTED_SLIDES} slides / {EXPECTED_PATIENTS} patients.\n"
            f"Observed {n_slides} slides / {n_patients} patients.\n"
            f"Authority: {CANONICAL_MANIFEST}"
        )
    if aaa0044_present:
        raise RuntimeError("CANONICAL INTERNAL COHORT VIOLATION: aaa0044 is present.")

    coords_col = first_existing_column(m, ["coords_path", "coordinates_path"])
    meta_col = first_existing_column(m, ["meta_path", "metadata_path"])
    slide_col = first_existing_column(m, ["slide_path", "wsi_path"])

    rows: List[Dict[str, Any]] = []
    for _, r in m.iterrows():
        prefix = str(r["prefix"])
        patient_id = str(r["patient_id"])

        coords_path = resolve_manifest_input_path(r[coords_col], CANONICAL_MANIFEST.parent) if coords_col else BASE / "wsi_embeddings" / f"{prefix}_coords.npy"
        meta_path = resolve_manifest_input_path(r[meta_col], CANONICAL_MANIFEST.parent) if meta_col else BASE / "wsi_embeddings" / f"{prefix}_meta.json"
        if coords_path is None or not coords_path.is_file():
            raise FileNotFoundError(f"{prefix}: canonical coords missing: {coords_path}")
        if meta_path is None or not meta_path.is_file():
            raise FileNotFoundError(f"{prefix}: canonical metadata missing: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        wsi_path = resolve_manifest_input_path(r[slide_col], CANONICAL_MANIFEST.parent) if slide_col else None
        if wsi_path is None or not wsi_path.is_file():
            wsi_path = resolve_manifest_input_path(meta.get("slide_path"), meta_path.parent)
        if wsi_path is None or not wsi_path.is_file():
            raise FileNotFoundError(f"{prefix}: canonical WSI missing")

        pca_path = CANONICAL_PCA_DIR / f"{prefix}_pca_targets.npy"
        if not pca_path.is_file():
            raise FileNotFoundError(
                f"{prefix}: canonical frozen PCA target missing from {CANONICAL_PCA_DIR}:\n{pca_path}"
            )

        rows.append({
            "patient_id": patient_id,
            "prefix": prefix,
            "wsi_path": str(wsi_path),
            "coords_path": str(coords_path),
            "pca_path": str(pca_path),
            "meta_path": str(meta_path),
            "patch_size": int(meta.get("patch_size", r.get("patch_size", 224))),
            "cohort_authority": "CANONICAL_INTERNAL_EMBEDDING_MANIFEST",
            "filesystem_discovery_used": False,
        })

    cohort = pd.DataFrame(rows).sort_values(["patient_id", "prefix"], kind="mergesort").reset_index(drop=True)
    return cohort, audit


def load_pc6_scores(cohort: pd.DataFrame, contract: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    thresholds = []

    for i, r in cohort.iterrows():
        prefix = str(r["prefix"])
        coords = np.load(Path(r["coords_path"]), mmap_mode="r")
        pca = np.load(Path(r["pca_path"]), mmap_mode="r")

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise RuntimeError(f"{prefix}: invalid coords shape {coords.shape}")
        if pca.ndim != 2 or pca.shape[0] != coords.shape[0] or pca.shape[1] <= PC6_INDEX:
            raise RuntimeError(f"{prefix}: invalid PCA target shape {pca.shape}")

        pc6 = np.asarray(pca[:, PC6_INDEX], dtype=float)
        finite = np.isfinite(pc6)
        if not finite.any():
            raise RuntimeError(f"{prefix}: no finite PC6 scores")

        vals = pc6[finite]
        q20 = float(np.quantile(vals, contract["low_q"]))
        q80 = float(np.quantile(vals, contract["high_q"]))
        low = finite & (pc6 <= q20)
        high = finite & (pc6 >= q80)
        overlap = int((low & high).sum())
        if overlap:
            raise RuntimeError(f"{prefix}: PC6_LOW/HIGH overlap={overlap}; refusing rescue")

        pct = np.full(len(pc6), np.nan, float)
        pct[finite] = pd.Series(vals).rank(method="average", pct=True).to_numpy(float)

        frames.append(pd.DataFrame({
            "patient_id": str(r["patient_id"]),
            "prefix": prefix,
            "tile_idx": np.arange(len(pc6), dtype=np.int64),
            "x": np.asarray(coords[:, 0], dtype=np.int64),
            "y": np.asarray(coords[:, 1], dtype=np.int64),
            "patch_size": int(r["patch_size"]),
            "wsi_path": str(r["wsi_path"]),
            "pc6_score": pc6,
            "pc6_within_slide_percentile": pct,
            "pc6_q20": q20,
            "pc6_q80": q80,
            "PC6_LOW": low,
            "PC6_HIGH": high,
        }))

        thresholds.append({
            "patient_id": r["patient_id"],
            "prefix": prefix,
            "n_tiles_total": len(pc6),
            "n_pc6_finite": int(finite.sum()),
            "pc6_q20": q20,
            "pc6_q80": q80,
            "n_PC6_LOW": int(low.sum()),
            "n_PC6_HIGH": int(high.sum()),
            "strata_overlap_count": overlap,
        })

        print(
            f"[SCORES {i+1:02d}/{len(cohort):02d}] {prefix} | "
            f"tiles={len(pc6):,} | low={int(low.sum()):,} | high={int(high.sum()):,}"
        )

    return pd.concat(frames, ignore_index=True), pd.DataFrame(thresholds)


def build_sampling(score_df: pd.DataFrame, contract: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stage1 = []
    audits = []

    for stratum in ["PC6_LOW", "PC6_HIGH"]:
        for prefix, sub in score_df.groupby("prefix", sort=False):
            cand = sub.loc[sub[stratum].astype(bool)].sort_values(["tile_idx", "x", "y"], kind="mergesort").reset_index(drop=True)
            if cand.empty:
                raise RuntimeError(f"{prefix}: no candidates for {stratum}")
            picked = deterministic_sample(
                cand,
                contract["max_slide"],
                stable_seed(contract["sampling_seed"], "slide", stratum, prefix),
            )
            picked["sampling_stratum"] = stratum
            picked["sampling_role"] = "PRIMARY_INTERNAL_PC6"
            stage1.append(picked)
            audits.append({
                "audit_level": "slide",
                "sampling_stratum": stratum,
                "patient_id": cand.iloc[0]["patient_id"],
                "prefix": prefix,
                "n_candidates": len(cand),
                "n_after_cap": len(picked),
                "cap": contract["max_slide"],
            })

    stage1_df = pd.concat(stage1, ignore_index=True)
    final = []
    for (stratum, patient_id), sub in stage1_df.groupby(["sampling_stratum", "patient_id"], sort=False):
        sub = sub.sort_values(["prefix", "tile_idx"], kind="mergesort").reset_index(drop=True)
        picked = deterministic_sample(
            sub,
            contract["max_patient"],
            stable_seed(contract["sampling_seed"], "patient", stratum, patient_id),
        )
        final.append(picked)
        audits.append({
            "audit_level": "patient",
            "sampling_stratum": stratum,
            "patient_id": patient_id,
            "prefix": "",
            "n_candidates": len(sub),
            "n_after_cap": len(picked),
            "cap": contract["max_patient"],
        })

    sampled = pd.concat(final, ignore_index=True).sort_values(
        ["sampling_stratum", "patient_id", "prefix", "tile_idx"], kind="mergesort"
    ).reset_index(drop=True)

    sampled["source_step"] = "23b"
    sampled["dataset"] = "internal_discovery"
    sampled["sample_id"] = sampled.apply(
        lambda r: f"PC6__{r['sampling_stratum']}__{r['prefix']}__tile{int(r['tile_idx'])}", axis=1
    )
    sampled["model_tile_uid"] = sampled.apply(
        lambda r: f"{r['prefix']}__x{int(r['x'])}__y{int(r['y'])}__p{int(r['patch_size'])}", axis=1
    )
    return sampled, pd.DataFrame(audits)


def deduplicate_physical_tiles(sampled: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = sampled.copy()
    df["resolved_wsi_path"] = df["wsi_path"].map(resolved_path)
    if df["resolved_wsi_path"].eq("").any():
        raise RuntimeError("Could not establish WSI identity for all sampled tiles")

    df["physical_tile_identity"] = df.apply(
        lambda r: (
            str(r["resolved_wsi_path"]), int(r["x"]), int(r["y"]), int(r["patch_size"])
        ),
        axis=1,
    )

    kept = []
    audit_rows = []
    for identity, sub in df.groupby("physical_tile_identity", sort=False):
        sub = sub.sort_values(["sampling_stratum", "prefix", "tile_idx"], kind="mergesort")
        strata = sorted(sub["sampling_stratum"].astype(str).unique())
        if len(strata) > 1:
            raise RuntimeError(
                "Same physical tile entered multiple PC6 strata; this must not happen:\n"
                + sub.to_string(index=False)
            )
        keep_idx = int(sub.index[0])
        kept.append(keep_idx)
        for idx, r in sub.iterrows():
            audit_rows.append({
                "physical_tile_identity": repr(identity),
                "sampling_stratum": r["sampling_stratum"],
                "patient_id": r["patient_id"],
                "prefix": r["prefix"],
                "tile_idx": int(r["tile_idx"]),
                "x": int(r["x"]),
                "y": int(r["y"]),
                "patch_size": int(r["patch_size"]),
                "sample_id": r["sample_id"],
                "status": (
                    "kept_unique_physical_tile" if len(sub) == 1
                    else "kept_physical_duplicate" if idx == keep_idx
                    else "deduplicated_physical_duplicate"
                ),
                "kept_row_index": keep_idx,
            })

    unique = df.loc[kept].copy().sort_values(
        ["sampling_stratum", "patient_id", "prefix", "tile_idx"], kind="mergesort"
    ).reset_index(drop=True)
    return unique, pd.DataFrame(audit_rows)


def run_step15b(step15b, sampled: pd.DataFrame):
    plan, thresholds, feature_spec = step15b.load_frozen_specification()

    observed_primary = feature_spec.loc[
        feature_spec["endpoint_tier"].astype(str).eq("primary"), "feature_name"
    ].astype(str).tolist()
    observed_secondary = feature_spec.loc[
        feature_spec["endpoint_tier"].astype(str).eq("secondary"), "feature_name"
    ].astype(str).tolist()
    if observed_primary != PRIMARY_FEATURES or observed_secondary != SECONDARY_FEATURES:
        raise RuntimeError("Step15B feature family differs from Step23A-frozen PC6 family")

    standardized = step15b.standardize_sampling(sampled.copy(), "23b")
    indexes = step15b.build_resolution_indexes()
    duplicate_audit = indexes.get("duplicate_audit", pd.DataFrame())
    tile, failures, resolution, equivalence = step15b.process_samples(
        standardized, thresholds, indexes
    )
    if tile.empty:
        raise RuntimeError("No PC6 tiles passed frozen Step15B measurement/QC")
    return tile, failures, resolution, equivalence, duplicate_audit, plan, thresholds, feature_spec


def attach_sampling_metadata(df: pd.DataFrame, sampled: pd.DataFrame, how: str) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [
        "sample_id", "patient_id", "prefix", "tile_idx", "x", "y", "patch_size",
        "sampling_stratum", "pc6_score", "pc6_within_slide_percentile",
        "pc6_q20", "pc6_q80", "wsi_path", "physical_tile_identity",
    ]
    lookup = sampled[keep].copy()
    if lookup["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id after physical dedup")

    # Remove fields duplicated by Step15B so the frozen sample manifest is authoritative.
    drop_cols = [
        c for c in ["patient_id", "prefix", "tile_idx", "x", "y", "patch_size", "sampling_stratum"]
        if c in df.columns
    ]
    left = df.drop(columns=drop_cols).copy()
    merged = lookup.merge(left, on="sample_id", how=how, validate="one_to_one" if how == "inner" else "one_to_many")
    return merged


def median_value(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.median()) if x.notna().any() else np.nan


def aggregate_slide(tile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "pc6_score", "pc6_within_slide_percentile", "tissue_fraction",
        "white_fraction", "black_fraction",
    ] + ALL_FEATURES

    for keys, sub in tile_df.groupby(["sampling_stratum", "patient_id", "prefix"], dropna=False, sort=False):
        row = dict(zip(["sampling_stratum", "patient_id", "prefix"], keys))
        row["n_tiles"] = len(sub)
        row["n_pixel_source_classes"] = sub["pixel_source_class"].nunique()
        row["pixel_source_classes"] = ";".join(sorted(sub["pixel_source_class"].dropna().astype(str).unique()))
        for m in metrics:
            if m in sub.columns:
                row[m] = median_value(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_patient(slide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "pc6_score", "pc6_within_slide_percentile", "tissue_fraction",
        "white_fraction", "black_fraction",
    ] + ALL_FEATURES

    for keys, sub in slide_df.groupby(["sampling_stratum", "patient_id"], dropna=False, sort=False):
        row = dict(zip(["sampling_stratum", "patient_id"], keys))
        row["n_slides"] = len(sub)
        row["n_tiles_total"] = int(pd.to_numeric(sub["n_tiles"], errors="coerce").sum())
        for m in metrics:
            if m in sub.columns:
                row[m] = median_value(sub[m])
        rows.append(row)
    return pd.DataFrame(rows)


def build_sampling_audit(samples, tile_df, failures_df, resolution_df) -> pd.DataFrame:
    requested = samples.groupby("sampling_stratum", dropna=False).agg(
        n_requested_tiles=("sample_id", "size"),
        n_requested_patients=("patient_id", "nunique"),
        n_requested_slides=("prefix", "nunique"),
    ).reset_index()

    resolved = (
        resolution_df.loc[resolution_df["resolution_status"].eq("resolved")]
        .groupby("sampling_stratum", dropna=False).size().rename("n_resolved_tiles").reset_index()
        if not resolution_df.empty
        else pd.DataFrame(columns=["sampling_stratum", "n_resolved_tiles"])
    )
    kept = tile_df.groupby("sampling_stratum", dropna=False).agg(
        n_qc_pass_tiles=("sample_id", "size"),
        n_qc_pass_patients=("patient_id", "nunique"),
        n_qc_pass_slides=("prefix", "nunique"),
    ).reset_index()
    failed = (
        failures_df.groupby("sampling_stratum", dropna=False).size().rename("n_failed_tiles").reset_index()
        if not failures_df.empty
        else pd.DataFrame(columns=["sampling_stratum", "n_failed_tiles"])
    )

    out = requested.merge(resolved, how="left", on="sampling_stratum").merge(
        kept, how="left", on="sampling_stratum"
    ).merge(failed, how="left", on="sampling_stratum")

    for c in [
        "n_resolved_tiles", "n_qc_pass_tiles", "n_qc_pass_patients",
        "n_qc_pass_slides", "n_failed_tiles",
    ]:
        out[c] = out[c].fillna(0).astype(int)
    out["resolution_fraction"] = out["n_resolved_tiles"] / out["n_requested_tiles"]
    out["qc_pass_fraction"] = out["n_qc_pass_tiles"] / out["n_requested_tiles"]
    return out


def build_feature_coverage(tile_df, slide_df, patient_df) -> pd.DataFrame:
    rows = []
    for feature in ALL_FEATURES:
        endpoint_class = (
            "primary" if feature in PRIMARY_FEATURES
            else "secondary" if feature in SECONDARY_FEATURES
            else "sensitivity"
        )
        for stratum in sorted(tile_df["sampling_stratum"].dropna().unique()):
            t = tile_df.loc[tile_df["sampling_stratum"].eq(stratum)]
            s = slide_df.loc[slide_df["sampling_stratum"].eq(stratum)]
            p = patient_df.loc[patient_df["sampling_stratum"].eq(stratum)]
            rows.append({
                "sampling_stratum": stratum,
                "feature": feature,
                "endpoint_class": endpoint_class,
                "n_tile_values": pd.to_numeric(t[feature], errors="coerce").notna().sum(),
                "n_slide_values": pd.to_numeric(s[feature], errors="coerce").notna().sum(),
                "n_patient_values": pd.to_numeric(p[feature], errors="coerce").notna().sum(),
            })
    return pd.DataFrame(rows)


def build_hashes(cohort: pd.DataFrame, step15b_path: Path) -> pd.DataFrame:
    rows = []
    fixed = [
        ("canonical_internal_embedding_manifest", CANONICAL_MANIFEST),
        ("step23a_plan", STEP23A_PLAN),
        ("step23a_metrics", STEP23A_METRICS),
        ("step23a_strata", STEP23A_STRATA),
        ("step23a_sampling", STEP23A_SAMPLING),
        ("step23a_direction", STEP23A_DIRECTION),
        ("step23a_metadata", STEP23A_METADATA),
        ("step15b_measurement_engine", step15b_path),
    ]
    if CANONICAL_PCA_MANIFEST.is_file():
        fixed.append(("canonical_internal_pca_manifest", CANONICAL_PCA_MANIFEST))

    for role, p in fixed:
        require_file(p, role)
        low = str(p).lower()
        if any(x in low for x in ["aggc", "peso", "tcga"]):
            raise RuntimeError(f"External source firewall violation: {p}")
        rows.append({"role": role, "prefix": "", "path": str(p), "sha256": sha256_file(p)})

    for _, r in cohort.iterrows():
        for role, col in [
            ("canonical_internal_coords", "coords_path"),
            ("canonical_internal_pca_targets", "pca_path"),
            ("canonical_internal_meta", "meta_path"),
        ]:
            p = Path(r[col])
            require_file(p, f"{role}:{r['prefix']}")
            low = str(p).lower()
            if any(x in low for x in ["aggc", "peso", "tcga"]):
                raise RuntimeError(f"External source firewall violation: {p}")
            rows.append({"role": role, "prefix": r["prefix"], "path": str(p), "sha256": sha256_file(p)})
    return pd.DataFrame(rows)


def readme_text(contract: Dict[str, Any], step15b_path: Path) -> str:
    return f"""STEP 23B - INTERNAL PC6 QUANTITATIVE MORPHOLOGY MEASUREMENT

Version: {VERSION}
Schema: {SCHEMA}

ROLE
----
MEASUREMENT ONLY.

CANONICAL COHORT
----------------
Authority:
  {CANONICAL_MANIFEST}

Required:
  {EXPECTED_SLIDES} slides
  {EXPECTED_PATIENTS} patients

Filesystem scanning is not used to define cohort membership.

PC6 STRATA
----------
PC6_LOW  = within-slide raw PC6 <= q20
PC6_HIGH = within-slide raw PC6 >= q80
Frozen direction = {contract['pc6_direction']}

SAMPLING
--------
max tiles / slide / stratum   = {contract['max_slide']}
max tiles / patient / stratum = {contract['max_patient']}
seed                          = {contract['sampling_seed']}

PHYSICAL TILE IDENTITY
----------------------
resolved WSI path + x + y + patch_size

MORPHOLOGY
----------
Exact INTERNAL Step15B v1.4.0 engine:
  {step15b_path}

All thresholds come from frozen Step15A.
No morphology definition is changed for PC6.

AGGREGATION
-----------
tile -> slide/stratum median -> patient/stratum median

NO INFERENCE HERE
-----------------
Wilcoxon: NO
Spearman outcome analysis: NO
Bootstrap CI: NO
P-values: NO
FDR: NO
PC3-vs-PC6 distinctness: NO

FIREWALL
--------
AGGC: NO
PESO: NO
TCGA: NO
Normal/G3/G4/G5: NO
PCA refit: NO
PC screening: NO
PC6 sign tuning: NO
PC5 promotion: NO
manual case/tile selection: NO
outcome-based cohort selection: NO

NEXT
----
Step23C should perform INTERNAL PC6_HIGH-vs-PC6_LOW inference and
PC3-vs-PC6 morphology-profile distinctness. External PC6 validation should
start only after that internal result is frozen.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Measurement-only INTERNAL PC6 morphology using canonical 65-slide cohort.",
    )
    p.add_argument("--output", type=Path, default=OUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--max-tiles", type=int, default=None,
        help="Engineering debug cap after frozen sampling; any capped run is non-formal.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    banner("STEP 23B: INTERNAL PC6 QUANTITATIVE MORPHOLOGY MEASUREMENT")
    print(f"[VERSION]       {VERSION}")
    print(f"[SCHEMA]        {SCHEMA}")
    print(f"[STEP23A]       {STEP23A_DIR}")
    print(f"[COHORT]        {CANONICAL_MANIFEST}")
    print(f"[PCA DIR]       {CANONICAL_PCA_DIR}")
    print(f"[STEP15A]       {STEP15A_DIR}")
    print(f"[OUTPUT]        {args.output}")
    print(f"[OVERWRITE]     {args.overwrite}")
    print()
    print("[ROLE]          MEASUREMENT ONLY")
    print("[EXPECTED]      65 slides / 15 patients")
    print("[COHORT SCAN]   NONE")
    print("[PATH RESOLVE]  absolute / BASE-relative / manifest-relative only")
    print("[AGGC READ]     NONE")
    print("[PESO READ]     NONE")
    print("[TCGA READ]     NONE")
    print("[GLEASON STATE] NONE")
    print("[INFERENCE]     NONE")
    print("[BOOTSTRAP]     NONE")
    print("[PC3-vs-PC6]    NONE")
    print("[PCA REFIT]     NONE")
    print("[PC SCREEN]     NONE")
    print("[SIGN TUNING]   NONE")

    prepare_output(args.output, args.overwrite)

    banner("VALIDATE STEP23A + INTERNAL STEP15B AUTHORITIES")
    contract = load_step23a_contract()
    step15b_path = locate_step15b()
    step15b = import_step15b(step15b_path)
    print(f"Frozen PC6 direction        : {contract['pc6_direction']}")
    print(f"PC6 tails                   : q{int(contract['low_q']*100)} / q{int(contract['high_q']*100)}")
    print(f"Max tiles/slide/stratum     : {contract['max_slide']}")
    print(f"Max tiles/patient/stratum   : {contract['max_patient']}")
    print(f"Sampling seed               : {contract['sampling_seed']}")
    print(f"Step15B engine              : {step15b_path}")
    print(f"Step15B version             : {getattr(step15b, 'VERSION', None)}")

    banner("LOAD CANONICAL INTERNAL DISCOVERY COHORT")
    cohort, cohort_audit = load_canonical_internal_cohort()
    cohort.to_csv(OUT_COHORT, index=False)
    cohort_audit.to_csv(OUT_COHORT_AUDIT, index=False)
    print(f"Canonical slides            : {len(cohort)}")
    print(f"Canonical patients          : {cohort['patient_id'].nunique()}")
    print(f"aaa0044 present             : {'YES' if 'aaa0044' in set(cohort['patient_id']) else 'NO'}")
    print(cohort[["patient_id", "prefix", "coords_path", "pca_path"]].to_string(index=False))

    banner("ASSIGN FROZEN WITHIN-SLIDE PC6 q20/q80 STRATA")
    score_df, threshold_df = load_pc6_scores(cohort, contract)
    threshold_df.to_csv(OUT_THRESHOLDS, index=False)
    print(threshold_df.to_string(index=False))

    banner("DETERMINISTIC SLIDE/PATIENT-BALANCED PC6 SAMPLING")
    sampled, pre_audit = build_sampling(score_df, contract)
    sampled, physical_audit = deduplicate_physical_tiles(sampled)

    if args.max_tiles is not None:
        if args.max_tiles <= 0:
            raise ValueError("--max-tiles must be positive")
        print("[WARNING] ENGINEERING DEBUG RUN ONLY")
        sampled = sampled.head(args.max_tiles).copy()

    sampled.to_csv(OUT_SAMPLE, index=False)
    pre_audit.to_csv(OUT_PRE_SAMPLE_AUDIT, index=False)
    physical_audit.to_csv(OUT_PHYSICAL_DEDUP, index=False)

    sample_summary = sampled.groupby("sampling_stratum").agg(
        n_sampled_tiles=("sample_id", "size"),
        n_unique_physical_tiles=("physical_tile_identity", "nunique"),
        n_slides=("prefix", "nunique"),
        n_patients=("patient_id", "nunique"),
    ).reset_index()
    print(sample_summary.to_string(index=False))

    banner("RUN IDENTICAL INTERNAL STEP15B v1.4 MORPHOLOGY ENGINE")
    (
        tile_raw, failures_raw, resolution_raw, equivalence_df,
        resolution_dup_df, step15a_plan, thresholds, feature_spec,
    ) = run_step15b(step15b, sampled)

    tile_df = attach_sampling_metadata(tile_raw, sampled, "inner")

    # Failures/resolution may contain multiple rows per sample only in unusual audit cases;
    # attach needed PC6 columns conservatively by left merge.
    if failures_raw.empty:
        failures_df = failures_raw
    else:
        lookup = sampled[[
            "sample_id", "pc6_score", "pc6_within_slide_percentile",
            "pc6_q20", "pc6_q80", "physical_tile_identity",
        ]].copy()
        failures_df = failures_raw.merge(lookup, on="sample_id", how="left", validate="many_to_one")

    if resolution_raw.empty:
        resolution_df = resolution_raw
    else:
        lookup = sampled[[
            "sample_id", "pc6_score", "pc6_within_slide_percentile",
            "pc6_q20", "pc6_q80", "physical_tile_identity",
        ]].copy()
        resolution_df = resolution_raw.merge(lookup, on="sample_id", how="left", validate="many_to_one")

    slide_df = aggregate_slide(tile_df)
    patient_df = aggregate_patient(slide_df)
    sampling_audit = build_sampling_audit(sampled, tile_df, failures_df, resolution_df)
    resolution_method_audit = step15b.build_resolution_method_audit(resolution_df)
    coverage_df = build_feature_coverage(tile_df, slide_df, patient_df)

    tile_df.to_csv(OUT_TILE, index=False)
    failures_df.to_csv(OUT_QC_FAILURES, index=False)
    slide_df.to_csv(OUT_SLIDE, index=False)
    patient_df.to_csv(OUT_PATIENT, index=False)
    coverage_df.to_csv(OUT_COVERAGE, index=False)
    sampling_audit.to_csv(OUT_AUDIT, index=False)
    resolution_df.to_csv(OUT_RESOLUTION, index=False)
    resolution_method_audit.to_csv(OUT_RESOLUTION_AUDIT, index=False)
    equivalence_df.to_csv(OUT_EQUIVALENCE, index=False)
    resolution_dup_df.to_csv(OUT_RESOLUTION_DUP, index=False)

    hashes = build_hashes(cohort, step15b_path)
    hashes.to_csv(OUT_HASHES, index=False)

    metadata = {
        "version": VERSION,
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "formal_full_run": args.max_tiles is None,
        "debug_max_tiles": args.max_tiles,
        "role": "internal_pc6_quantitative_morphology_measurement_only",
        "canonical_cohort": {
            "authority": str(CANONICAL_MANIFEST),
            "expected_slides": EXPECTED_SLIDES,
            "observed_slides": int(len(cohort)),
            "expected_patients": EXPECTED_PATIENTS,
            "observed_patients": int(cohort["patient_id"].nunique()),
            "aaa0044_present": bool("aaa0044" in set(cohort["patient_id"])),
            "filesystem_discovery_used": False,
        },
        "pc6": {
            "frozen_direction": contract["pc6_direction"],
            "low_stratum": f"within-slide raw PC6 <= q{int(contract['low_q']*100)}",
            "high_stratum": f"within-slide raw PC6 >= q{int(contract['high_q']*100)}",
            "sign_tuning": False,
        },
        "sampling": {
            "max_tiles_per_slide_per_stratum": contract["max_slide"],
            "max_tiles_per_patient_per_stratum": contract["max_patient"],
            "seed": contract["sampling_seed"],
            "n_requested_after_physical_dedup": int(len(sampled)),
            "physical_identity_rule": "resolved WSI path + x + y + patch_size",
        },
        "morphology": {
            "step15b_path": str(step15b_path),
            "step15b_version": str(getattr(step15b, "VERSION", "")),
            "step15a_version": step15a_plan.get("version"),
            "threshold_ownership": "all thresholds inherited from frozen Step15A",
            "features": PRIMARY_FEATURES + SECONDARY_FEATURES,
        },
        "counts": {
            "n_qc_pass_tiles": int(len(tile_df)),
            "n_failed_tiles": int(len(failures_df)),
            "n_slide_summary_rows": int(len(slide_df)),
            "n_patient_summary_rows": int(len(patient_df)),
            "n_resolution_duplicate_audit_rows": int(len(resolution_dup_df)),
            "n_pixel_equivalence_rows": int(len(equivalence_df)),
        },
        "firewall": {
            "aggc_files_read": False,
            "peso_files_read": False,
            "tcga_files_read": False,
            "normal_g3_g4_g5_framework_used": False,
            "pca_refit": False,
            "pc_screening": False,
            "pc6_sign_tuning": False,
            "pc5_promotion": False,
            "outcome_based_cohort_selection": False,
            "manual_slide_selection": False,
            "manual_tile_selection": False,
            "inferential_statistics_performed": False,
            "pc3_pc6_distinctness_performed": False,
        },
    }
    save_json(OUT_METADATA, metadata)
    OUT_README.write_text(readme_text(contract, step15b_path), encoding="utf-8")

    banner("SAMPLING / QC AUDIT")
    print(sampling_audit.to_string(index=False))

    banner("RESOLUTION METHOD AUDIT")
    print("No resolved tiles." if resolution_method_audit.empty else resolution_method_audit.to_string(index=False))

    banner("MEASUREMENT-ONLY FIREWALL AUDIT")
    for item, status in [
        ("Canonical manifest used", "YES"),
        ("Canonical cohort = 65 slides", "YES"),
        ("Canonical cohort = 15 patients", "YES"),
        ("aaa0044 included", "NO"),
        ("Filesystem cohort scan used", "NO"),
        ("Physical tile dedup performed", "YES"),
        ("Physical identity = WSI+x+y+patch", "YES"),
        ("Step15B v1.4 engine reused", "YES"),
        ("AGGC files read", "NO"),
        ("PESO files read", "NO"),
        ("TCGA files read", "NO"),
        ("Normal/G3/G4/G5 used", "NO"),
        ("PCA refit", "NO"),
        ("PC screening", "NO"),
        ("PC6 sign tuning", "NO"),
        ("PC5 promotion", "NO"),
        ("Bootstrap performed", "NO"),
        ("P-values calculated", "NO"),
        ("PC3-vs-PC6 distinctness calculated", "NO"),
    ]:
        print(f"{item:<58}: {status}")

    expected_outputs = [
        OUT_COHORT, OUT_COHORT_AUDIT, OUT_THRESHOLDS, OUT_SAMPLE,
        OUT_PRE_SAMPLE_AUDIT, OUT_PHYSICAL_DEDUP, OUT_TILE, OUT_QC_FAILURES,
        OUT_SLIDE, OUT_PATIENT, OUT_COVERAGE, OUT_AUDIT, OUT_RESOLUTION,
        OUT_RESOLUTION_AUDIT, OUT_EQUIVALENCE, OUT_RESOLUTION_DUP,
        OUT_HASHES, OUT_METADATA, OUT_README,
    ]
    print("\n[OK] Saved:")
    for p in expected_outputs:
        if not p.exists():
            raise RuntimeError(f"Expected output missing: {p}")
        print(f"  {p}")

    print("\n[DONE] Step23B INTERNAL PC6 morphology measurement complete.")
    print(
        "[NEXT] Step23C may perform frozen INTERNAL PC6_HIGH-vs-PC6_LOW inference "
        "and PC3-vs-PC6 morphology-profile distinctness."
    )


if __name__ == "__main__":
    main()
