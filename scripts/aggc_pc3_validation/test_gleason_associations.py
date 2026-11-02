#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
18c_test_aggc_pc3_gleason_histologic_state_associations.py

Formal Step-18 outcome analysis.

Frozen authorities
------------------
18a v1.1.0 : scientific design
18a v1.1.1 : primary evaluability amendment
18a v1.1.2 : tile-support sensitivity freeze
18b v1.0.2 : measurement-only tile table

Primary
-------
Within each slide:
    effect = median PC3(G4) - median PC3(G3)

Independent unit:
    slide

Primary support:
    >=50 eligible G3 tiles AND >=50 eligible G4 tiles

Primary cohorts:
    Train n=66
    Test  n=30

Sensitivity:
    25 / 50 / 100 / 200 tiles per class
    50 remains PRIMARY

Exploratory:
    G5-G4
    G5-G3
    G3->G4->G5 state-level ordinal profile
    Stroma->Normal->G3->G4->G5 extended histologic-state profile

No pooled tile-level inference.
No PCA refit.
No PC screening.
No PC6.
No post-outcome exclusion or threshold tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

VERSION = "1.0.0"
SCHEMA = "18c-aggc-pc3-gleason-histologic-state-association-schema-1.0"

MEDIA_BASE = REPO_ROOT
R = RESULTS_ROOT

STEP18A_V110 = R / "18a_aggc_pc3_gleason_histologic_state_specification"
STEP18A_V111 = R / "18a_aggc_pc3_gleason_histologic_state_specification_v1_1_1_amendment"
STEP18A_V112 = R / "18a_aggc_pc3_g3_g4_tile_support_sensitivity_specification_v1_1_2"
STEP18B = R / "18b_aggc_pc3_gleason_histologic_state_measurements"
OUT_DIR = R / "18c_aggc_pc3_gleason_histologic_state_association"

PRIMARY_T = 50
SENS_T = [25, 50, 100, 200]
N_BOOT = 10000
SEED = 20260808

EXPECTED_PRIMARY = {
    "train_validation": 66,
    "test_replication": 30,
}
EXPECTED_SENS = {
    25: {"train_validation": 72, "test_replication": 34},
    50: {"train_validation": 66, "test_replication": 30},
    100: {"train_validation": 59, "test_replication": 28},
    200: {"train_validation": 49, "test_replication": 24},
}

CLASSES = ["stroma", "normal", "g3", "g4", "g5"]


def banner(s: str) -> None:
    print("\n" + "=" * 120)
    print(s)
    print("=" * 120)


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, x: Any) -> None:
    with p.open("w", encoding="utf-8") as f:
        json.dump(x, f, indent=2, sort_keys=True)


def require(p: Path) -> None:
    if not p.is_file():
        raise FileNotFoundError(p)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def as_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse bool: {x!r}")


def part(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"train", "training", "validation", "train_validation"}:
        return "train_validation"
    if s in {"test", "testing", "replication", "test_replication"}:
        return "test_replication"
    raise ValueError(f"Unexpected partition {x!r}")


def wilcoxon_p(x: Sequence[float]) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    if np.allclose(a, 0):
        return 1.0
    try:
        return float(wilcoxon(a, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def bootstrap_median(x: Sequence[float], n: int, seed: int) -> Tuple[np.ndarray, float, float]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.array([], float), np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(n, float)
    for i in range(n):
        out[i] = np.median(a[rng.integers(0, len(a), len(a))])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return out, float(lo), float(hi)


def summarize(x: Sequence[float], seed: int) -> Tuple[Dict[str, Any], np.ndarray]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    boot, lo, hi = bootstrap_median(a, N_BOOT, seed)
    d = {
        "n_slides": int(len(a)),
        "median_effect": float(np.median(a)) if len(a) else np.nan,
        "mean_effect": float(np.mean(a)) if len(a) else np.nan,
        "q25_effect": float(np.quantile(a, .25)) if len(a) else np.nan,
        "q75_effect": float(np.quantile(a, .75)) if len(a) else np.nan,
        "fraction_effect_gt_0": float(np.mean(a > 0)) if len(a) else np.nan,
        "fraction_effect_lt_0": float(np.mean(a < 0)) if len(a) else np.nan,
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "wilcoxon_p_two_sided": wilcoxon_p(a),
    }
    return d, boot


def source_audit(args) -> Dict[str, Path]:
    p = {
        "v110_metadata": args.step18a_v110 / "analysis_metadata.json",
        "v111_evaluable": args.step18a_v111 / "aggc_pc3_g3_g4_measurement_evaluable_primary_cohort.csv",
        "v111_amendment": args.step18a_v111 / "evaluability_amendment.json",
        "v112_manifest": args.step18a_v112 / "aggc_pc3_g3_g4_tile_support_sensitivity_manifest.csv",
        "v112_plan": args.step18a_v112 / "tile_support_sensitivity_analysis_plan.json",
        "b_tile": args.step18b / "aggc_pc3_histologic_state_tile_measurements.csv.gz",
        "b_qc": args.step18b / "aggc_pc3_histologic_state_slide_measurement_qc.csv",
        "b_metadata": args.step18b / "analysis_metadata.json",
    }
    for x in p.values():
        require(x)
    if str(load_json(p["v110_metadata"]).get("version")) != "1.1.0":
        raise RuntimeError("Expected Step18a v1.1.0.")
    if str(load_json(p["v111_amendment"]).get("version")) != "1.1.1":
        raise RuntimeError("Expected Step18a v1.1.1 amendment.")
    if str(load_json(p["v112_plan"]).get("version")) != "1.1.2":
        raise RuntimeError("Expected Step18a v1.1.2 sensitivity freeze.")
    if str(load_json(p["b_metadata"]).get("version")) != "1.0.2":
        raise RuntimeError("Expected Step18b v1.0.2.")
    return p


def load_primary(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False)
    d["slide_id"] = d["slide_id"].astype(str)
    d["partition"] = d["partition"].map(part)
    d = d[["slide_id", "partition"]].drop_duplicates()
    for k, n in EXPECTED_PRIMARY.items():
        obs = int((d["partition"] == k).sum())
        if obs != n:
            raise RuntimeError(f"Primary {k}: {obs} != {n}")
    return d


def load_sens(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False)
    d["slide_id"] = d["slide_id"].astype(str)
    d["partition"] = d["partition"].map(part)
    d["threshold_tiles_per_class"] = pd.to_numeric(d["threshold_tiles_per_class"]).astype(int)
    d["included_at_threshold"] = d["included_at_threshold"].map(as_bool)
    for t, parts in EXPECTED_SENS.items():
        for k, n in parts.items():
            obs = int(((d.threshold_tiles_per_class == t) &
                       (d.partition == k) &
                       d.included_at_threshold).sum())
            if obs != n:
                raise RuntimeError(f"Sensitivity t={t} {k}: {obs} != {n}")
    return d


def load_qc(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False)
    d["slide_id"] = d["slide_id"].astype(str)
    d["partition"] = d["partition"].map(part)
    for c in [f"n_{x}_eligible_tiles" for x in CLASSES]:
        d[c] = pd.to_numeric(d[c], errors="raise").astype(int)
    return d


def load_tiles(p: Path) -> pd.DataFrame:
    use = ["slide_id", "partition", "PC3"] + [f"{x}_eligible" for x in CLASSES]
    d = pd.read_csv(p, usecols=use, low_memory=False)
    d["slide_id"] = d["slide_id"].astype(str)
    d["partition"] = d["partition"].map(part)
    d["PC3"] = pd.to_numeric(d["PC3"], errors="coerce")
    for x in CLASSES:
        d[f"{x}_eligible"] = d[f"{x}_eligible"].map(as_bool)
    d = d[np.isfinite(d["PC3"])].copy()
    print(f"[LOAD] finite PC3 tile rows: {len(d):,}")
    return d


def slide_state_summary(tile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = tile.groupby(["partition", "slide_id"], sort=False)
    total = groups.ngroups
    for i, ((p, sid), d) in enumerate(groups, 1):
        r = {"partition": p, "slide_id": sid}
        for x in CLASSES:
            v = d.loc[d[f"{x}_eligible"], "PC3"].to_numpy(float)
            v = v[np.isfinite(v)]
            r[f"n_{x}_tiles"] = int(len(v))
            r[f"median_pc3_{x}"] = float(np.median(v)) if len(v) else np.nan
            r[f"mean_pc3_{x}"] = float(np.mean(v)) if len(v) else np.nan
        rows.append(r)
        if i % 25 == 0 or i == total:
            print(f"[SLIDE SUMMARY] {i}/{total}")
    return pd.DataFrame(rows)


def primary_tables(state: pd.DataFrame, cohort: pd.DataFrame):
    d = cohort.merge(state, on=["slide_id", "partition"], how="left", validate="one_to_one")
    if not ((d.n_g3_tiles >= 50) & (d.n_g4_tiles >= 50)).all():
        raise RuntimeError("Primary cohort violates frozen >=50 support.")
    d["g4_minus_g3_median_pc3"] = d.median_pc3_g4 - d.median_pc3_g3

    sr, br = [], []
    for j, k in enumerate(["train_validation", "test_replication"]):
        x = d.loc[d.partition == k, "g4_minus_g3_median_pc3"].to_numpy(float)
        s, b = summarize(x, SEED + j)
        sr.append({
            "analysis": "PRIMARY_G3_vs_G4",
            "partition": k,
            "threshold_tiles_per_class": 50,
            "effect_definition": "median_PC3_G4_minus_median_PC3_G3",
            **s,
        })
        br += [{
            "analysis": "PRIMARY_G3_vs_G4",
            "partition": k,
            "bootstrap_index": i,
            "bootstrap_median_effect": float(v),
        } for i, v in enumerate(b)]
    return d, pd.DataFrame(sr), pd.DataFrame(br)


def sensitivity_tables(state: pd.DataFrame, manifest: pd.DataFrame):
    m = manifest[manifest.included_at_threshold].copy()
    d = m.merge(state, on=["slide_id", "partition"], how="left", validate="many_to_one")
    d["g4_minus_g3_median_pc3"] = d.median_pc3_g4 - d.median_pc3_g3
    sr, br = [], []
    q = 100
    for t in SENS_T:
        for k in ["train_validation", "test_replication"]:
            z = d[(d.threshold_tiles_per_class == t) & (d.partition == k)]
            if not ((z.n_g3_tiles >= t) & (z.n_g4_tiles >= t)).all():
                raise RuntimeError(f"Sensitivity t={t} violates frozen support.")
            s, b = summarize(z.g4_minus_g3_median_pc3.to_numpy(float), SEED + q)
            q += 1
            role = "PRIMARY" if t == 50 else "SENSITIVITY"
            sr.append({
                "analysis": "G3_G4_TILE_SUPPORT",
                "threshold_tiles_per_class": t,
                "threshold_role": role,
                "partition": k,
                "effect_definition": "median_PC3_G4_minus_median_PC3_G3",
                **s,
            })
            br += [{
                "threshold_tiles_per_class": t,
                "threshold_role": role,
                "partition": k,
                "bootstrap_index": i,
                "bootstrap_median_effect": float(v),
            } for i, v in enumerate(b)]
    return d, pd.DataFrame(sr), pd.DataFrame(br)


def g5_tables(state: pd.DataFrame):
    rows = []
    for name, high, low in [("G5_minus_G4", "g5", "g4"), ("G5_minus_G3", "g5", "g3")]:
        z = state[(state[f"n_{high}_tiles"] >= 50) & (state[f"n_{low}_tiles"] >= 50)]
        for _, r in z.iterrows():
            rows.append({
                "analysis": name,
                "partition": r.partition,
                "slide_id": r.slide_id,
                "n_high_tiles": int(r[f"n_{high}_tiles"]),
                "n_low_tiles": int(r[f"n_{low}_tiles"]),
                "median_pc3_high": float(r[f"median_pc3_{high}"]),
                "median_pc3_low": float(r[f"median_pc3_{low}"]),
                "high_minus_low_median_pc3": float(r[f"median_pc3_{high}"] - r[f"median_pc3_{low}"]),
            })
    d = pd.DataFrame(rows)
    sr, br = [], []
    q = 300
    for name in ["G5_minus_G4", "G5_minus_G3"]:
        for k in ["train_validation", "test_replication"]:
            z = d[(d.analysis == name) & (d.partition == k)] if len(d) else pd.DataFrame()
            vals = z.high_minus_low_median_pc3.to_numpy(float) if len(z) else np.array([])
            s, b = summarize(vals, SEED + q); q += 1
            sr.append({"analysis": name, "role": "EXPLORATORY", "partition": k, **s})
            br += [{
                "analysis": name, "partition": k,
                "bootstrap_index": i, "bootstrap_median_effect": float(v)
            } for i, v in enumerate(b)]
    return d, pd.DataFrame(sr), pd.DataFrame(br)


def ordinal_tables(state: pd.DataFrame, classes: List[str], name: str, seed_base: int):
    elig = np.ones(len(state), dtype=bool)
    for x in classes:
        elig &= state[f"n_{x}_tiles"].to_numpy(int) >= 50
    d = state.loc[elig].copy()
    rows = []
    order = np.arange(1, len(classes) + 1, dtype=float)

    for _, r in d.iterrows():
        med = [float(r[f"median_pc3_{x}"]) for x in classes]
        rho = float(spearmanr(order, med).statistic) if len(set(med)) > 1 else 0.0
        rows.append({
            "partition": r.partition,
            "slide_id": r.slide_id,
            **{f"n_{x}_tiles": int(r[f"n_{x}_tiles"]) for x in classes},
            **{f"median_pc3_{x}": float(r[f"median_pc3_{x}"]) for x in classes},
            "state_spearman_rho": rho,
            "monotonic_increasing": bool(all(med[i] < med[i+1] for i in range(len(med)-1))),
            "monotonic_decreasing": bool(all(med[i] > med[i+1] for i in range(len(med)-1))),
        })

    out = pd.DataFrame(rows)
    sr, br = [], []
    q = seed_base
    for k in ["train_validation", "test_replication"]:
        z = out[out.partition == k] if len(out) else pd.DataFrame()
        vals = z.state_spearman_rho.to_numpy(float) if len(z) else np.array([])
        s, b = summarize(vals, SEED + q); q += 1
        sr.append({
            "analysis": name,
            "role": "EXPLORATORY",
            "partition": k,
            "statistic": "state_level_spearman_rho",
            **s,
            "fraction_monotonic_increasing": float(z.monotonic_increasing.mean()) if len(z) else np.nan,
            "fraction_monotonic_decreasing": float(z.monotonic_decreasing.mean()) if len(z) else np.nan,
        })
        br += [{
            "analysis": name, "partition": k,
            "bootstrap_index": i, "bootstrap_median_rho": float(v)
        } for i, v in enumerate(b)]
    return out, pd.DataFrame(sr), pd.DataFrame(br)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--step18a-v110", type=Path, default=STEP18A_V110)
    p.add_argument("--step18a-v111", type=Path, default=STEP18A_V111)
    p.add_argument("--step18a-v112", type=Path, default=STEP18A_V112)
    p.add_argument("--step18b", type=Path, default=STEP18B)
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()

    banner("STEP 18C: AGGC PC3 GLEASON / HISTOLOGIC-STATE FORMAL ASSOCIATION")
    print(f"[VERSION]      {VERSION}")
    print(f"[SCHEMA]       {SCHEMA}")
    print(f"[18A v1.1.0]  {a.step18a_v110}")
    print(f"[18A v1.1.1]  {a.step18a_v111}")
    print(f"[18A v1.1.2]  {a.step18a_v112}")
    print(f"[STEP 18B]    {a.step18b}")
    print(f"[OUTPUT]      {a.output_dir}")
    print("[PRIMARY]     median PC3(G4) - median PC3(G3)")
    print("[PRIMARY N]   Train=66 / Test=30")
    print("[UNIT]        SLIDE")
    print("[SENSITIVITY] 25 / 50 / 100 / 200")
    print("[PC SCREEN]   NONE")
    print("[PC6]         NONE")

    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.overwrite:
        raise RuntimeError("Output directory non-empty; use --overwrite intentionally.")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    src = source_audit(a)
    primary = load_primary(src["v111_evaluable"])
    sens = load_sens(src["v112_manifest"])
    qc = load_qc(src["b_qc"])

    banner("FIRST FORMAL PC3 OUTCOME READ")
    tiles = load_tiles(src["b_tile"])
    state = slide_state_summary(tiles)
    del tiles

    banner("PRIMARY G3 vs G4")
    p_slide, p_sum, p_boot = primary_tables(state, primary)
    print(p_sum.to_string(index=False))

    banner("PRESPECIFIED TILE-SUPPORT SENSITIVITY")
    s_slide, s_sum, s_boot = sensitivity_tables(state, sens)
    print(s_sum[
        ["threshold_tiles_per_class","threshold_role","partition","n_slides",
         "median_effect","bootstrap_ci_low","bootstrap_ci_high",
         "fraction_effect_gt_0","wilcoxon_p_two_sided"]
    ].to_string(index=False))

    banner("EXPLORATORY G5")
    g_slide, g_sum, g_boot = g5_tables(state)
    print(g_sum.to_string(index=False))

    banner("EXPLORATORY G3 -> G4 -> G5")
    o_slide, o_sum, o_boot = ordinal_tables(
        state, ["g3","g4","g5"], "G3_G4_G5_ORDINAL", 500
    )
    print(o_sum.to_string(index=False))

    banner("EXPLORATORY STROMA -> NORMAL -> G3 -> G4 -> G5")
    e_slide, e_sum, e_boot = ordinal_tables(
        state, ["stroma","normal","g3","g4","g5"],
        "STROMA_NORMAL_G3_G4_G5_PROFILE", 700
    )
    print(e_sum.to_string(index=False))

    # Save.
    p_slide.to_csv(a.output_dir/"aggc_pc3_g3_g4_primary_per_slide.csv", index=False)
    p_sum.to_csv(a.output_dir/"aggc_pc3_g3_g4_primary_summary.csv", index=False)
    p_boot.to_csv(a.output_dir/"aggc_pc3_g3_g4_primary_bootstrap_distribution.csv.gz",
                  index=False, compression="gzip")

    s_slide.to_csv(a.output_dir/"aggc_pc3_g3_g4_tile_support_sensitivity_per_slide.csv", index=False)
    s_sum.to_csv(a.output_dir/"aggc_pc3_g3_g4_tile_support_sensitivity_summary.csv", index=False)
    s_boot.to_csv(a.output_dir/"aggc_pc3_g3_g4_tile_support_sensitivity_bootstrap.csv.gz",
                  index=False, compression="gzip")

    g_slide.to_csv(a.output_dir/"aggc_pc3_g5_exploratory_per_slide.csv", index=False)
    g_sum.to_csv(a.output_dir/"aggc_pc3_g5_exploratory_summary.csv", index=False)
    g_boot.to_csv(a.output_dir/"aggc_pc3_g5_exploratory_bootstrap.csv.gz",
                  index=False, compression="gzip")

    o_slide.to_csv(a.output_dir/"aggc_pc3_tumor_ordinal_per_slide.csv", index=False)
    o_sum.to_csv(a.output_dir/"aggc_pc3_tumor_ordinal_summary.csv", index=False)
    o_boot.to_csv(a.output_dir/"aggc_pc3_tumor_ordinal_bootstrap.csv.gz",
                  index=False, compression="gzip")

    e_slide.to_csv(a.output_dir/"aggc_pc3_extended_histologic_state_per_slide.csv", index=False)
    e_sum.to_csv(a.output_dir/"aggc_pc3_extended_histologic_state_summary.csv", index=False)
    e_boot.to_csv(a.output_dir/"aggc_pc3_extended_histologic_state_bootstrap.csv.gz",
                  index=False, compression="gzip")

    audit = pd.DataFrame([
        ["primary_contrast","EXECUTED_AS_FROZEN","G3 vs G4"],
        ["primary_effect","EXECUTED_AS_FROZEN","median PC3(G4)-median PC3(G3)"],
        ["primary_threshold","EXECUTED_AS_FROZEN",">=50 tiles/class/slide"],
        ["independent_unit","SLIDE",""],
        ["pooled_tile_level_inference","NO",""],
        ["test_specific_tuning","NO",""],
        ["outcome_based_slide_exclusion","NO",""],
        ["threshold_redefinition","NO",""],
        ["endpoint_redefinition","NO",""],
        ["pc3_sign_flip","NO",""],
        ["pca_refit","NO",""],
        ["pc_screening","NO",""],
        ["pc6_analysis","NO",""],
        ["g5_role","EXPLORATORY_ONLY",""],
        ["tumor_ordinal_role","EXPLORATORY_ONLY","G3->G4->G5"],
        ["five_state_role","EXPLORATORY_ONLY","histologic-state profile, not progression"],
        ["support_sensitivity","PRESPECIFIED","25/50/100/200; 50 remains primary"],
    ], columns=["item","status","detail"])
    audit.to_csv(a.output_dir/"analysis_audit.csv", index=False)

    pd.DataFrame([
        {"role": k, "path": str(v), "sha256": sha256(v)}
        for k, v in src.items()
    ]).to_csv(a.output_dir/"source_hashes.csv", index=False)

    save_json(a.output_dir/"analysis_metadata.json", {
        "version": VERSION,
        "schema": SCHEMA,
        "python_version": platform.python_version(),
        "n_bootstrap": N_BOOT,
        "bootstrap_seed": SEED,
        "primary": {
            "effect": "median_PC3_G4_minus_median_PC3_G3",
            "threshold": 50,
            "train_n": 66,
            "test_n": 30,
            "independent_unit": "slide",
            "two_sided": True,
        },
        "sensitivity_thresholds": SENS_T,
        "guardrails": {
            "pooled_tile_level_inference": False,
            "outcome_based_slide_exclusion": False,
            "threshold_redefinition": False,
            "endpoint_redefinition": False,
            "pc3_sign_flip": False,
            "pca_refit": False,
            "pc_screening": False,
            "pc6": False,
        },
    })

    (a.output_dir/"README.txt").write_text(
        """STEP18C formal association/inference.

PRIMARY:
  within-slide median PC3(G4) - median PC3(G3)
  slide = independent unit
  >=50 eligible tiles per class
  Train n=66
  Test n=30
  two-sided Wilcoxon vs 0
  slide-bootstrap 95% CI for median effect

PRESPECIFIED SUPPORT SENSITIVITY:
  25 / 50 / 100 / 200 tiles per class
  50 remains PRIMARY

EXPLORATORY:
  G5-G4
  G5-G3
  G3->G4->G5
  Stroma->Normal->G3->G4->G5 histologic-state profile

NO pooled tile-level inference.
NO PCA refit.
NO PC screening.
NO PC6.
NO threshold tuning.
NO outcome-based exclusion.
""",
        encoding="utf-8"
    )

    banner("PRIMARY RESULT")
    for _, r in p_sum.iterrows():
        print(
            f"[{r.partition}]\n"
            f"Slides                  : {int(r.n_slides)}\n"
            f"Median G4-G3 PC3 effect : {r.median_effect:+.6f}\n"
            f"95% bootstrap CI        : [{r.bootstrap_ci_low:+.6f}, {r.bootstrap_ci_high:+.6f}]\n"
            f"Slides effect > 0       : {r.fraction_effect_gt_0:.3%}\n"
            f"Slides effect < 0       : {r.fraction_effect_lt_0:.3%}\n"
            f"Wilcoxon two-sided p    : {r.wilcoxon_p_two_sided:.6g}\n"
        )

    banner("ANALYSIS BOUNDARY AUDIT")
    for _, r in audit.iterrows():
        print(f"{r['item']:38s}: {r['status']}")

    print("\n[DONE] Step18c formal association analysis complete.")
    print(f"[OUTPUT] {a.output_dir}")


if __name__ == "__main__":
    main()
