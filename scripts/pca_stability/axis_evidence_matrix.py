#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

IN_03A = RESULTS_ROOT / "03a_strict_equal_slide_pca_sensitivity" / "component_stability_summary.csv"
IN_03B = RESULTS_ROOT / "03b_capped_per_slide_pca_sensitivity" / "component_stability_summary.csv"
IN_07 = RESULTS_ROOT / "07_discovery_fixed_direction_tables" / "discovery_fixed_direction_summary.csv"
IN_08_SUMMARY = RESULTS_ROOT / "08_patient_bootstrap_pca_stability" / "component_stability_summary.csv"
IN_08_MATCHES = RESULTS_ROOT / "08_patient_bootstrap_pca_stability" / "bootstrap_component_matches.csv"
IN_09_SUMMARY = RESULTS_ROOT / "09_leave_one_patient_out_pca_stability" / "component_stability_summary.csv"
IN_09_MATCHES = RESULTS_ROOT / "09_leave_one_patient_out_pca_stability" / "lopo_component_matches.csv"
IN_10 = RESULTS_ROOT / "10_nested_lopo_pca_tumor_validation" / "nested_component_summary.csv"

OUT_DIR = RESULTS_ROOT / "11_pc1_pc8_axis_evidence_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATRIX = OUT_DIR / "pc1_pc8_axis_evidence_matrix.csv"
OUT_PRIMARY = OUT_DIR / "primary_qualified_axes.csv"
OUT_EXPLORATORY = OUT_DIR / "exploratory_axes.csv"
OUT_NOT_QUALIFIED = OUT_DIR / "not_qualified_axes.csv"
OUT_PC56 = OUT_DIR / "pc5_pc6_subspace_summary.csv"
OUT_SOURCES = OUT_DIR / "evidence_sources.csv"
OUT_METADATA = OUT_DIR / "analysis_metadata.json"
OUT_README = OUT_DIR / "README.txt"

EXPECTED_COMPONENTS = [f"PC{i}" for i in range(1, 9)]

MIN_NESTED_FROZEN_FOLD_FRACTION = 0.80
MIN_NESTED_MEDIAN_AUC = 0.60
MIN_HELDOUT_AUC_GT_HALF_FRACTION = 0.80
MIN_LOPO_MEDIAN_COSINE = 0.90
MIN_BOOTSTRAP_MEDIAN_COSINE = 0.80
MIN_EXPLORATORY_NESTED_MEDIAN_AUC = 0.60

PC56_SET = {"PC5", "PC6"}
ROUND_DIGITS = 4


def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_component_column(df, candidates):
    out = df.copy()
    src = next((c for c in candidates if c in out.columns), None)
    if src is None:
        raise ValueError(f"No component column found among: {candidates}")
    out["Component"] = out[src].astype(str)
    missing = set(EXPECTED_COMPONENTS) - set(out["Component"].unique())
    if missing:
        raise ValueError(f"Missing expected components: {sorted(missing)}")
    return out


def select_and_rename(df, rename_map):
    out = df.rename(columns=rename_map)
    keep = ["Component"] + [new for new in rename_map.values() if new in out.columns]
    return out[keep].copy()


def round_numeric_columns(df):
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].round(ROUND_DIGITS)
    return out


def load_03a():
    df = normalize_component_column(pd.read_csv(IN_03A), ["Primary component", "primary_component", "Component", "component"])
    return select_and_rename(df, {
        "Mean absolute loading cosine": "Strict-equal mean loading cosine",
        "Median absolute loading cosine": "Strict-equal median loading cosine",
        "Minimum absolute loading cosine": "Strict-equal minimum loading cosine",
        "Mean tile-score Spearman rho": "Strict-equal mean score rho",
        "Median tile-score Spearman rho": "Strict-equal median score rho",
        "Minimum tile-score Spearman rho": "Strict-equal minimum score rho",
        "Matched component consistency": "Strict-equal exact-axis match consistency",
        "Most frequent matched balanced component": "Strict-equal most frequent matched component",
    })


def load_03b():
    df = normalize_component_column(pd.read_csv(IN_03B), ["Primary component", "primary_component", "Component", "component"])
    return select_and_rename(df, {
        "Mean absolute loading cosine": "Capped mean loading cosine",
        "Median absolute loading cosine": "Capped median loading cosine",
        "Minimum absolute loading cosine": "Capped minimum loading cosine",
        "Mean tile-score Spearman rho": "Capped mean score rho",
        "Median tile-score Spearman rho": "Capped median score rho",
        "Minimum tile-score Spearman rho": "Capped minimum score rho",
        "Matched component consistency": "Capped exact-axis match consistency",
        "Most frequent matched capped component": "Capped most frequent matched component",
    })


def load_07():
    df = normalize_component_column(pd.read_csv(IN_07), ["Component", "component", "pc"])
    return select_and_rename(df, {
        "Specification status": "Discovery specification status",
        "Frozen direction": "Discovery frozen direction",
        "Frozen AUC source": "Discovery frozen AUC source",
        "Direction consensus fraction": "Discovery direction consensus",
        "Mean fixed-direction AUC": "Discovery mean fixed-direction AUC",
        "Median fixed-direction AUC": "Discovery median fixed-direction AUC",
        "SD fixed-direction AUC": "Discovery SD fixed-direction AUC",
        "Mean direction-optimized AUC": "Discovery mean direction-optimized AUC",
        "Median direction-optimized AUC": "Discovery median direction-optimized AUC",
        "Decision reason": "Discovery decision reason",
    })


def load_08():
    df = normalize_component_column(pd.read_csv(IN_08_SUMMARY), ["Primary component", "primary_component", "Component", "component"])
    return select_and_rename(df, {
        "Matched component consistency": "Bootstrap exact-axis match consistency",
        "Most frequent matched bootstrap component": "Bootstrap most frequent matched component",
        "Mean absolute loading cosine": "Bootstrap mean loading cosine",
        "Median absolute loading cosine": "Bootstrap median loading cosine",
        "Minimum absolute loading cosine": "Bootstrap minimum loading cosine",
        "Mean tile-score Spearman rho": "Bootstrap mean score rho",
        "Median tile-score Spearman rho": "Bootstrap median score rho",
        "Minimum tile-score Spearman rho": "Bootstrap minimum score rho",
        "Cosine >= 0.80 fraction": "Bootstrap cosine >=0.80 fraction",
        "Cosine >= 0.90 fraction": "Bootstrap cosine >=0.90 fraction",
        "Score rho >= 0.80 fraction": "Bootstrap score rho >=0.80 fraction",
        "Score rho >= 0.90 fraction": "Bootstrap score rho >=0.90 fraction",
    })


def load_09():
    df = normalize_component_column(pd.read_csv(IN_09_SUMMARY), ["Primary component", "primary_component", "Component", "component"])
    return select_and_rename(df, {
        "Matched component consistency": "LOPO exact-axis match consistency",
        "Most frequent matched LOPO component": "LOPO most frequent matched component",
        "Mean absolute loading cosine": "LOPO mean loading cosine",
        "Median absolute loading cosine": "LOPO median loading cosine",
        "Minimum absolute loading cosine": "LOPO minimum loading cosine",
        "Mean tile-score Spearman rho": "LOPO mean score rho",
        "Median tile-score Spearman rho": "LOPO median score rho",
        "Minimum tile-score Spearman rho": "LOPO minimum score rho",
        "Cosine >= 0.80 fraction": "LOPO cosine >=0.80 fraction",
        "Cosine >= 0.90 fraction": "LOPO cosine >=0.90 fraction",
        "Score rho >= 0.80 fraction": "LOPO score rho >=0.80 fraction",
        "Score rho >= 0.90 fraction": "LOPO score rho >=0.90 fraction",
    })


def load_10():
    df = normalize_component_column(pd.read_csv(IN_10), ["Component", "component", "reference_component"])
    return select_and_rename(df, {
        "Fraction of folds with frozen training direction": "Nested frozen-fold fraction",
        "N held-out slide AUC values": "Nested N held-out AUC values",
        "Mean held-out fixed-direction AUC": "Nested mean held-out fixed-direction AUC",
        "Median held-out fixed-direction AUC": "Nested median held-out fixed-direction AUC",
        "SD held-out fixed-direction AUC": "Nested SD held-out fixed-direction AUC",
        "Q25 held-out fixed-direction AUC": "Nested Q25 held-out fixed-direction AUC",
        "Q75 held-out fixed-direction AUC": "Nested Q75 held-out fixed-direction AUC",
        "Fraction held-out AUC > 0.5": "Nested fraction held-out AUC >0.5",
        "Mean LOPO loading cosine": "Nested mean loading cosine",
        "Median LOPO loading cosine": "Nested median loading cosine",
        "Minimum LOPO loading cosine": "Nested minimum loading cosine",
        "Most frequent fold component": "Nested most frequent fold component",
        "Fold component consistency": "Nested exact-axis match consistency",
    })


def make_pc56_subspace_summary():
    boot = pd.read_csv(IN_08_MATCHES)
    lopo = pd.read_csv(IN_09_MATCHES)

    rows = []
    for pc in ["PC5", "PC6"]:
        bg = boot.loc[boot["primary_component"] == pc].copy()
        lg = lopo.loc[lopo["primary_component"] == pc].copy()

        b_exact = bg["bootstrap_component"].eq(pc)
        b_pair = bg["bootstrap_component"].isin(PC56_SET)
        l_exact = lg["lopo_component"].eq(pc)
        l_pair = lg["lopo_component"].isin(PC56_SET)

        rows.append({
            "Component": pc,
            "Bootstrap exact-axis match fraction": float(b_exact.mean()),
            "Bootstrap PC5-PC6 pair-contained match fraction": float(b_pair.mean()),
            "Bootstrap swap fraction within PC5-PC6 pair": float((b_pair & ~b_exact).mean()),
            "Bootstrap outside-PC5-PC6 match fraction": float((~b_pair).mean()),
            "Bootstrap median loading cosine": float(bg["absolute_loading_cosine"].median()),
            "Bootstrap median score rho": float(bg["tile_score_spearman_rho"].median()),
            "LOPO exact-axis match fraction": float(l_exact.mean()),
            "LOPO PC5-PC6 pair-contained match fraction": float(l_pair.mean()),
            "LOPO swap fraction within PC5-PC6 pair": float((l_pair & ~l_exact).mean()),
            "LOPO outside-PC5-PC6 match fraction": float((~l_pair).mean()),
            "LOPO median loading cosine": float(lg["absolute_loading_cosine"].median()),
            "LOPO median score rho": float(lg["tile_score_spearman_rho"].median()),
        })

    return round_numeric_columns(pd.DataFrame(rows))


def merge_pc56_metrics(evidence, pc56):
    rename = {
        "Bootstrap exact-axis match fraction": "PC5-PC6 bootstrap exact-axis match fraction",
        "Bootstrap PC5-PC6 pair-contained match fraction": "PC5-PC6 bootstrap pair-contained match fraction",
        "Bootstrap swap fraction within PC5-PC6 pair": "PC5-PC6 bootstrap swap fraction",
        "Bootstrap outside-PC5-PC6 match fraction": "PC5-PC6 bootstrap outside-pair fraction",
        "LOPO exact-axis match fraction": "PC5-PC6 LOPO exact-axis match fraction",
        "LOPO PC5-PC6 pair-contained match fraction": "PC5-PC6 LOPO pair-contained match fraction",
        "LOPO swap fraction within PC5-PC6 pair": "PC5-PC6 LOPO swap fraction",
        "LOPO outside-PC5-PC6 match fraction": "PC5-PC6 LOPO outside-pair fraction",
    }
    tmp = pc56.rename(columns=rename)
    cols = ["Component"] + list(rename.values())
    return evidence.merge(tmp[cols], on="Component", how="left", validate="one_to_one")


def apply_qualification(df):
    out = df.copy()

    out["Criterion A: discovery direction frozen"] = out["Discovery specification status"].astype(str).eq("frozen")
    out["Criterion B: nested frozen-fold fraction"] = pd.to_numeric(out["Nested frozen-fold fraction"], errors="coerce") >= MIN_NESTED_FROZEN_FOLD_FRACTION
    out["Criterion C: nested median AUC"] = pd.to_numeric(out["Nested median held-out fixed-direction AUC"], errors="coerce") >= MIN_NESTED_MEDIAN_AUC
    out["Criterion D: held-out AUC >0.5 fraction"] = pd.to_numeric(out["Nested fraction held-out AUC >0.5"], errors="coerce") >= MIN_HELDOUT_AUC_GT_HALF_FRACTION
    out["Criterion E: LOPO median cosine"] = pd.to_numeric(out["LOPO median loading cosine"], errors="coerce") >= MIN_LOPO_MEDIAN_COSINE
    out["Criterion F: bootstrap median cosine"] = pd.to_numeric(out["Bootstrap median loading cosine"], errors="coerce") >= MIN_BOOTSTRAP_MEDIAN_COSINE

    criteria = [
        "Criterion A: discovery direction frozen",
        "Criterion B: nested frozen-fold fraction",
        "Criterion C: nested median AUC",
        "Criterion D: held-out AUC >0.5 fraction",
        "Criterion E: LOPO median cosine",
        "Criterion F: bootstrap median cosine",
    ]

    out["N primary criteria met"] = out[criteria].fillna(False).astype(bool).sum(axis=1)
    out["Primary qualification"] = out[criteria].fillna(False).astype(bool).all(axis=1)

    nested_frozen = pd.to_numeric(out["Nested frozen-fold fraction"], errors="coerce")
    nested_auc = pd.to_numeric(out["Nested median held-out fixed-direction AUC"], errors="coerce")

    out["Exploratory evidence"] = (
        nested_frozen.gt(0)
        & nested_auc.ge(MIN_EXPLORATORY_NESTED_MEDIAN_AUC)
        & ~out["Primary qualification"]
    )

    labels = []
    reasons = []

    for _, row in out.iterrows():
        failed = [
            c.replace("Criterion ", "")
            for c in criteria
            if not bool(row[c])
        ]

        if bool(row["Primary qualification"]):
            labels.append("Primary-qualified")
            if row["Component"] in PC56_SET:
                reasons.append(
                    "Met all primary criteria. Exact-axis swapping within PC5-PC6 "
                    "is reported separately from matched-axis stability."
                )
            else:
                reasons.append("Met all prespecified primary qualification criteria.")
        elif bool(row["Exploratory evidence"]):
            labels.append("Exploratory")
            reasons.append(
                "Did not meet all primary criteria but retained nontrivial nested "
                "tumor-association evidence. Failed: " + "; ".join(failed)
            )
        else:
            labels.append("Not qualified")
            reasons.append(
                "Did not meet all primary criteria and lacked sufficient nested "
                "tumor-association evidence. Failed: " + "; ".join(failed)
            )

    out["Axis qualification"] = labels
    out["Qualification rationale"] = reasons
    return out


def make_sources_table():
    return pd.DataFrame([
        {"Step": "03a", "Evidence type": "Strict equal-slide PCA sensitivity", "Path": str(IN_03A)},
        {"Step": "03b", "Evidence type": "Capped per-slide PCA sensitivity", "Path": str(IN_03B)},
        {"Step": "07", "Evidence type": "Discovery fixed-direction tumor association", "Path": str(IN_07)},
        {"Step": "08", "Evidence type": "Patient bootstrap PCA stability", "Path": str(IN_08_SUMMARY)},
        {"Step": "08 matches", "Evidence type": "Bootstrap component matches", "Path": str(IN_08_MATCHES)},
        {"Step": "09", "Evidence type": "LOPO PCA stability", "Path": str(IN_09_SUMMARY)},
        {"Step": "09 matches", "Evidence type": "LOPO component matches", "Path": str(IN_09_MATCHES)},
        {"Step": "10", "Evidence type": "Nested LOPO tumor validation", "Path": str(IN_10)},
    ])


def main():
    required = [
        IN_03A, IN_03B, IN_07, IN_08_SUMMARY, IN_08_MATCHES,
        IN_09_SUMMARY, IN_09_MATCHES, IN_10
    ]

    for path in required:
        check_file(path)

    evidence = pd.DataFrame({"Component": EXPECTED_COMPONENTS})

    for loader in [load_03a, load_03b, load_07, load_08, load_09, load_10]:
        evidence = evidence.merge(
            loader(),
            on="Component",
            how="left",
            validate="one_to_one",
        )

    pc56 = make_pc56_subspace_summary()
    evidence = merge_pc56_metrics(evidence, pc56)
    evidence = apply_qualification(evidence)

    evidence["Component number"] = (
        evidence["Component"].str.replace("PC", "", regex=False).astype(int)
    )

    evidence = evidence.sort_values("Component number").reset_index(drop=True)
    evidence = round_numeric_columns(evidence)

    primary = evidence.loc[evidence["Axis qualification"] == "Primary-qualified"].copy()
    exploratory = evidence.loc[evidence["Axis qualification"] == "Exploratory"].copy()
    not_qualified = evidence.loc[evidence["Axis qualification"] == "Not qualified"].copy()

    evidence.to_csv(OUT_MATRIX, index=False)
    primary.to_csv(OUT_PRIMARY, index=False)
    exploratory.to_csv(OUT_EXPLORATORY, index=False)
    not_qualified.to_csv(OUT_NOT_QUALIFIED, index=False)
    pc56.to_csv(OUT_PC56, index=False)

    sources = make_sources_table()
    sources.to_csv(OUT_SOURCES, index=False)

    metadata = {
        "analysis_name": "PC1-PC8 integrated axis evidence matrix",
        "exact_axis_consistency_is_hard_criterion": False,
        "primary_thresholds": {
            "nested_frozen_fold_fraction": MIN_NESTED_FROZEN_FOLD_FRACTION,
            "nested_median_auc": MIN_NESTED_MEDIAN_AUC,
            "heldout_auc_gt_half_fraction": MIN_HELDOUT_AUC_GT_HALF_FRACTION,
            "lopo_median_loading_cosine": MIN_LOPO_MEDIAN_COSINE,
            "bootstrap_median_loading_cosine": MIN_BOOTSTRAP_MEDIAN_COSINE,
            "discovery_direction_must_be_frozen": True,
        },
        "pc5_pc6_reporting": {
            "enabled": True,
            "pair": ["PC5", "PC6"],
            "definition": "Matched component remains within the PC5-PC6 pair.",
            "principal_angle_subspace_analysis_performed": False,
        },
        "morphology_used_for_qualification": False,
        "external_validation_used_for_qualification": False,
        "manual_override_used": False,
        "input_hashes": {str(path): sha256_file(path) for path in required},
    }

    with OUT_METADATA.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    primary_names = primary["Component"].tolist()
    exploratory_names = exploratory["Component"].tolist()
    not_names = not_qualified["Component"].tolist()

    OUT_README.write_text(
        "\n".join([
            "PC1-PC8 Axis Evidence Matrix",
            "============================",
            "",
            "Primary-qualified: " + (", ".join(primary_names) if primary_names else "None"),
            "Exploratory: " + (", ".join(exploratory_names) if exploratory_names else "None"),
            "Not qualified: " + (", ".join(not_names) if not_names else "None"),
            "",
            "Exact component-number consistency is reported but is not a hard criterion.",
            "PC5-PC6 pair-contained matching is reported separately.",
            "This is not a true principal-angle subspace stability analysis.",
        ]),
        encoding="utf-8",
    )

    print("=" * 130)
    print("PC1-PC8 INTEGRATED AXIS EVIDENCE MATRIX")
    print("=" * 130)

    cols = [
        "Component",
        "Discovery specification status",
        "Discovery direction consensus",
        "Discovery median fixed-direction AUC",
        "Bootstrap exact-axis match consistency",
        "Bootstrap median loading cosine",
        "LOPO exact-axis match consistency",
        "LOPO median loading cosine",
        "Nested frozen-fold fraction",
        "Nested median held-out fixed-direction AUC",
        "Nested fraction held-out AUC >0.5",
        "N primary criteria met",
        "Axis qualification",
    ]

    print(evidence[[c for c in cols if c in evidence.columns]].to_string(index=False))

    print("\nPC5-PC6 SUBSPACE-AWARE SUMMARY")
    print(pc56.to_string(index=False))

    print("\nPrimary-qualified axes:", primary_names)
    print("Exploratory axes:", exploratory_names)
    print("Not qualified:", not_names)

    print("\n[OK] Saved:")
    for path in [
        OUT_MATRIX, OUT_PRIMARY, OUT_EXPLORATORY, OUT_NOT_QUALIFIED,
        OUT_PC56, OUT_SOURCES, OUT_METADATA, OUT_README
    ]:
        print(f"  {path}")

    print("\n[DONE] Axis evidence integration complete.")


if __name__ == "__main__":
    main()
