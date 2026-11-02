#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, hashlib, json, platform, sys
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

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = RESULTS_ROOT

STEP15C_DIR = RESULTS_ROOT / "15c_pc3_quantitative_morphology_inference"
STEP15C_PRIMARY = STEP15C_DIR / "pc3_primary_all_tissue_patient_level_results.csv"
STEP15C_TUMOR_CORE = STEP15C_DIR / "pc3_tumor_core_sensitivity_patient_level_results.csv"
STEP15C_TUMOR_ONLY = STEP15C_DIR / "pc3_tumor_only_sensitivity_patient_level_results.csv"
STEP15C_METADATA = STEP15C_DIR / "analysis_metadata.json"

STEP15D_DIR = RESULTS_ROOT / "15d_pc3_continuous_morphology_sensitivity"
STEP15D_RESULTS = STEP15D_DIR / "pc3_continuous_patient_level_inference.csv"
STEP15D_METADATA = STEP15D_DIR / "analysis_metadata.json"

OUT_DIR = RESULTS_ROOT / "15e_pc3_quantitative_morphology_summary"
OUT_SUMMARY = OUT_DIR / "pc3_quantitative_morphology_integrated_summary.csv"
OUT_DIRECTION = OUT_DIR / "pc3_quantitative_morphology_direction_consistency.csv"
OUT_PNG = OUT_DIR / "pc3_quantitative_morphology_integrated_figure.png"
OUT_PDF = OUT_DIR / "pc3_quantitative_morphology_integrated_figure.pdf"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"

CORE_OUTPUTS = [
    OUT_SUMMARY, OUT_DIRECTION, OUT_PNG, OUT_PDF, OUT_METADATA, OUT_README
]

VERSION = "1.0.0"

PRIMARY_FEATURES = [
    "SEI_eosin_stroma",
    "GOI_lumen_openness",
    "NPI_nuclear_packing",
]

FEATURE_LABELS = {
    "SEI_eosin_stroma": "SEI\nEosin/stroma-like",
    "GOI_lumen_openness": "GOI\nLumen openness",
    "NPI_nuclear_packing": "NPI\nNuclear-richness",
}

EXPECTED_SIGNS = {
    "SEI_eosin_stroma": +1,
    "GOI_lumen_openness": -1,
    "NPI_nuclear_packing": -1,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def guard_outputs(overwrite: bool):
    existing = [p for p in CORE_OUTPUTS if p.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Step-15e outputs already exist. Use --overwrite if intentional."
        )

def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input ({label}): {path}")

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_df(df: pd.DataFrame, label: str):
    if "feature" not in df.columns:
        raise RuntimeError(f"{label} missing feature column.")
    sub = df[df["feature"].isin(PRIMARY_FEATURES)]
    if set(sub["feature"]) != set(PRIMARY_FEATURES):
        raise RuntimeError(f"{label} missing one or more primary endpoints.")
    if sub.groupby("feature").size().gt(1).any():
        raise RuntimeError(f"{label} contains duplicate primary endpoint rows.")

def load_inputs():
    for p, label in [
        (STEP15C_PRIMARY, "15c primary"),
        (STEP15C_TUMOR_CORE, "15c tumor-core"),
        (STEP15C_TUMOR_ONLY, "15c tumor-only"),
        (STEP15C_METADATA, "15c metadata"),
        (STEP15D_RESULTS, "15d results"),
        (STEP15D_METADATA, "15d metadata"),
    ]:
        require_file(p, label)

    primary = pd.read_csv(STEP15C_PRIMARY)
    core = pd.read_csv(STEP15C_TUMOR_CORE)
    tumor = pd.read_csv(STEP15C_TUMOR_ONLY)
    continuous = pd.read_csv(STEP15D_RESULTS)
    meta15c = load_json(STEP15C_METADATA)
    meta15d = load_json(STEP15D_METADATA)

    for df, label in [
        (primary, "15c primary"),
        (core, "15c tumor-core"),
        (tumor, "15c tumor-only"),
        (continuous, "15d continuous"),
    ]:
        validate_df(df, label)

    if not meta15c.get("provenance_chain_verified", False):
        raise RuntimeError("Step-15c provenance chain was not verified.")
    if not meta15d.get("provenance_chain_verified", False):
        raise RuntimeError("Step-15d provenance chain was not verified.")
    if not meta15d.get("formal_full_run", False):
        raise RuntimeError("Step-15d is not marked as a formal full run.")

    return primary, core, tumor, continuous, meta15c, meta15d

def one_row(df: pd.DataFrame, feature: str) -> pd.Series:
    sub = df[df["feature"] == feature]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one row for {feature}, found {len(sub)}")
    return sub.iloc[0]

def build_summary(primary, core, tumor, continuous):
    rows = []
    for feature in PRIMARY_FEATURES:
        p = one_row(primary, feature)
        c = one_row(core, feature)
        t = one_row(tumor, feature)
        d = one_row(continuous, feature)
        positive = EXPECTED_SIGNS[feature] > 0

        rows.append({
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature].replace("\n", " "),
            "all_tissue_n_patients": int(p["n_paired_patients"]),
            "all_tissue_median_high_minus_low": float(p["median_paired_difference_high_minus_low"]),
            "all_tissue_ci_lower": float(p["bootstrap_95ci_lower"]),
            "all_tissue_ci_upper": float(p["bootstrap_95ci_upper"]),
            "all_tissue_rank_biserial": float(p["paired_rank_biserial"]),
            "all_tissue_p_raw": float(p["p_value_raw"]),
            "all_tissue_q_fdr": float(p["p_value_fdr_bh"]),
            "all_tissue_n_expected_direction": int(
                p["n_high_greater_than_low"] if positive else p["n_high_less_than_low"]
            ),
            "tumor_core_n_patients": int(c["n_paired_patients"]),
            "tumor_core_median_high_minus_low": float(c["median_paired_difference_high_minus_low"]),
            "tumor_core_ci_lower": float(c["bootstrap_95ci_lower"]),
            "tumor_core_ci_upper": float(c["bootstrap_95ci_upper"]),
            "tumor_core_rank_biserial": float(c["paired_rank_biserial"]),
            "tumor_core_p_raw": float(c["p_value_raw"]),
            "tumor_core_q_fdr": float(c["p_value_fdr_bh"]),
            "tumor_core_n_expected_direction": int(
                c["n_high_greater_than_low"] if positive else c["n_high_less_than_low"]
            ),
            "tumor_only_n_patients": int(t["n_paired_patients"]),
            "tumor_only_median_high_minus_low": float(t["median_paired_difference_high_minus_low"]),
            "tumor_only_ci_lower": float(t["bootstrap_95ci_lower"]),
            "tumor_only_ci_upper": float(t["bootstrap_95ci_upper"]),
            "tumor_only_rank_biserial": float(t["paired_rank_biserial"]),
            "tumor_only_p_raw": float(t["p_value_raw"]),
            "tumor_only_q_fdr": float(t["p_value_fdr_bh"]),
            "tumor_only_n_expected_direction": int(
                t["n_high_greater_than_low"] if positive else t["n_high_less_than_low"]
            ),
            "continuous_n_patients": int(d["n_patients"]),
            "continuous_median_patient_rho": float(d["median_patient_rho"]),
            "continuous_ci_lower": float(d["bootstrap_95ci_lower"]),
            "continuous_ci_upper": float(d["bootstrap_95ci_upper"]),
            "continuous_rank_biserial_vs_zero": float(d["rank_biserial_vs_zero"]),
            "continuous_p_raw": float(d["p_value_raw"]),
            "continuous_q_fdr": float(d["p_value_fdr_bh"]),
            "continuous_n_expected_direction": int(
                d["n_positive_rho"] if positive else d["n_negative_rho"]
            ),
        })
    return pd.DataFrame(rows)

def build_direction(summary):
    rows = []
    for _, r in summary.iterrows():
        for analysis, ncol, ecol in [
            ("All-tissue extreme", "all_tissue_n_patients", "all_tissue_n_expected_direction"),
            ("Tumor-core extreme", "tumor_core_n_patients", "tumor_core_n_expected_direction"),
            ("Tumor-only extreme", "tumor_only_n_patients", "tumor_only_n_expected_direction"),
            ("Continuous PC3", "continuous_n_patients", "continuous_n_expected_direction"),
        ]:
            n = int(r[ncol]); e = int(r[ecol])
            rows.append({
                "feature": r["feature"],
                "analysis": analysis,
                "n_patients": n,
                "n_expected_direction": e,
                "direction_consistency_fraction": e / n if n else np.nan,
            })
    return pd.DataFrame(rows)

def make_figure(summary, direction):
    fig = plt.figure(figsize=(13.2, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    x = np.arange(len(PRIMARY_FEATURES))

    for label, ecol, lcol, ucol, off, marker in [
        ("All tissue", "all_tissue_median_high_minus_low", "all_tissue_ci_lower", "all_tissue_ci_upper", -0.18, "o"),
        ("Tumor core", "tumor_core_median_high_minus_low", "tumor_core_ci_lower", "tumor_core_ci_upper", 0.00, "s"),
        ("Tumor only", "tumor_only_median_high_minus_low", "tumor_only_ci_lower", "tumor_only_ci_upper", 0.18, "^"),
    ]:
        eff = summary[ecol].to_numpy(float)
        lo = summary[lcol].to_numpy(float)
        hi = summary[ucol].to_numpy(float)
        ax_a.errorbar(
            x + off, eff, yerr=np.vstack([eff-lo, hi-eff]),
            fmt=marker, capsize=3, linewidth=1.2, markersize=6, label=label
        )

    ax_a.axhline(0, linewidth=1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([FEATURE_LABELS[f] for f in PRIMARY_FEATURES], fontsize=10)
    ax_a.set_ylabel("Median paired difference\n(PC3-high − PC3-low)", fontsize=11)
    ax_a.set_title("A  Frozen extreme-group morphology", loc="left", fontweight="bold", fontsize=12)
    ax_a.legend(frameon=False, fontsize=9)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    eff = summary["continuous_median_patient_rho"].to_numpy(float)
    lo = summary["continuous_ci_lower"].to_numpy(float)
    hi = summary["continuous_ci_upper"].to_numpy(float)
    ax_b.errorbar(
        x, eff, yerr=np.vstack([eff-lo, hi-eff]),
        fmt="o", capsize=4, linewidth=1.3, markersize=7
    )
    ax_b.axhline(0, linewidth=1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([FEATURE_LABELS[f] for f in PRIMARY_FEATURES], fontsize=10)
    ax_b.set_ylabel("Median patient-level\nwithin-slide Spearman ρ", fontsize=11)
    ax_b.set_title("B  Continuous-PC3 morphology gradient", loc="left", fontweight="bold", fontsize=12)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    analyses = ["All-tissue extreme", "Tumor-core extreme", "Tumor-only extreme", "Continuous PC3"]
    offsets = [-0.24, 0.0, 0.24]
    markers = ["o", "s", "^"]
    xx = np.arange(len(analyses))

    for feature, off, marker in zip(PRIMARY_FEATURES, offsets, markers):
        sub = direction[direction["feature"] == feature].set_index("analysis").loc[analyses].reset_index()
        frac = sub["direction_consistency_fraction"].to_numpy(float)
        ax_c.scatter(xx + off, frac, marker=marker, s=70, label=FEATURE_LABELS[feature].replace("\n", " "))
        for i, row in sub.iterrows():
            ax_c.text(
                xx[i] + off, float(row["direction_consistency_fraction"]) + 0.018,
                f"{int(row['n_expected_direction'])}/{int(row['n_patients'])}",
                ha="center", va="bottom", fontsize=8
            )

    ax_c.set_xticks(xx)
    ax_c.set_xticklabels(analyses, fontsize=10)
    ax_c.set_ylim(0, 1.12)
    ax_c.set_ylabel("Fraction of patients\nin expected direction", fontsize=11)
    ax_c.set_title("C  Patient-level directional consistency", loc="left", fontweight="bold", fontsize=12)
    ax_c.legend(frameon=False, fontsize=9, ncol=3, loc="lower center")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig.suptitle(
        "PC3 quantitative morphology is consistent across extreme-group and continuous analyses",
        fontsize=15, fontweight="bold", y=0.98
    )
    fig.text(
        0.5, 0.015,
        "Image-derived morphology proxies; error bars show patient-level bootstrap 95% confidence intervals. "
        "No new inference is performed in Step 15e.",
        ha="center", va="bottom", fontsize=9
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    args = parse_args()
    print("=" * 124)
    print("STEP 15E: INTEGRATED PC3 QUANTITATIVE MORPHOLOGY SUMMARY")
    print("=" * 124)
    print("[VERSION]", VERSION)
    print("[OUTPUT] ", OUT_DIR)

    guard_outputs(args.overwrite)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    primary, core, tumor, continuous, meta15c, meta15d = load_inputs()
    summary = build_summary(primary, core, tumor, continuous)
    direction = build_direction(summary)

    summary.to_csv(OUT_SUMMARY, index=False)
    direction.to_csv(OUT_DIRECTION, index=False)
    make_figure(summary, direction)

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "reporting/integration only",
        "new_inferential_tests_performed": False,
        "feature_selection_performed": False,
        "primary_features": PRIMARY_FEATURES,
        "step15c_version": meta15c.get("version"),
        "step15d_version": meta15d.get("version"),
        "step15c_primary_sha256": sha256_file(STEP15C_PRIMARY),
        "step15c_tumor_core_sha256": sha256_file(STEP15C_TUMOR_CORE),
        "step15c_tumor_only_sha256": sha256_file(STEP15C_TUMOR_ONLY),
        "step15d_results_sha256": sha256_file(STEP15D_RESULTS),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    with OUT_METADATA.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    OUT_README.write_text(
        """STEP 15E: INTEGRATED PC3 QUANTITATIVE MORPHOLOGY SUMMARY

Reporting/integration only. No new statistical tests are performed.

Panel A:
  Step-15c paired PC3-high minus PC3-low morphology effects.

Panel B:
  Step-15d continuous-PC3 patient-level Spearman associations.

Panel C:
  Patient-level directional consistency across all-tissue, tumor-core,
  tumor-only, and continuous analyses.

Claim boundary:
  SEI = eosin/stroma-like image proxy.
  GOI = lumen/open-space image proxy.
  NPI = hematoxylin/nuclear-richness image proxy.

Step 15e does not alter sampling, endpoints, p-values, FDR values,
confidence intervals, or effect estimates from Steps 15c/15d.
""",
        encoding="utf-8",
    )

    print("\nINTEGRATED SUMMARY")
    print(summary[
        [
            "feature",
            "all_tissue_median_high_minus_low",
            "all_tissue_q_fdr",
            "tumor_core_median_high_minus_low",
            "tumor_core_q_fdr",
            "tumor_only_median_high_minus_low",
            "tumor_only_q_fdr",
            "continuous_median_patient_rho",
            "continuous_q_fdr",
        ]
    ].to_string(index=False))

    print("\n[OK] Saved:")
    for p in CORE_OUTPUTS:
        print(" ", p)

    print("\n[DONE] Step-15e integrated reporting complete.")

if __name__ == "__main__":
    main()
