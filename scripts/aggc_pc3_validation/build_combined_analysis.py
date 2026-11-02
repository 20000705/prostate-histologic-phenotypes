#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 31A: assemble combined AGGC PC3 Train+Test tables from frozen outputs.

ASSEMBLY/AUDIT ONLY. No PCA refit, PC3 sign change, endpoint redefinition,
AUC recalculation, morphology recalculation, p-values, or model fitting.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil
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

import numpy as np
import pandas as pd

VERSION="1.0.0"
SCHEMA="31a-combined-aggc-pc3-analysis-assembly-schema-1.0"
MEDIA = REPO_ROOT
RESULTS=RESULTS_ROOT
STEP19B=RESULTS/"19b_aggc_pc3_available_case_histologic_state_measurements"
STEP20B=RESULTS/"20b_aggc_pc3_within_class_morphology_measurements"
STEP20E=RESULTS/"20e_aggc_test_pc3_morphology_associations"
STEP21C=RESULTS/"21c_aggc_pc3_tumor_localization_auc"
OUT=RESULTS/"31a_combined_aggc_pc3_analysis"
EXPECTED={"train_validation":105,"test_replication":45}
PRIMARY_THRESHOLD=50

def norm_part(x):
    s=str(x).strip().lower()
    m={"train":"train_validation","training":"train_validation","validation":"train_validation","train_validation":"train_validation",
       "test":"test_replication","testing":"test_replication","replication":"test_replication","test_replication":"test_replication"}
    if s not in m: raise RuntimeError(f"Unexpected partition: {x!r}")
    return m[s]

def req(p,label):
    if not p.is_file(): raise FileNotFoundError(f"Missing {label}:\n{p}")

def load(p):
    d=pd.read_csv(p,low_memory=False)
    if "partition" in d: d["partition"]=d["partition"].map(norm_part)
    if "slide_id" in d: d["slide_id"]=d["slide_id"].astype(str).str.strip()
    return d

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def prep(p,overwrite):
    if p.exists() and any(p.iterdir()):
        if not overwrite: raise RuntimeError(f"Output non-empty: {p}\nUse --overwrite intentionally.")
        shutil.rmtree(p)
    p.mkdir(parents=True,exist_ok=True)

def validate_universe(u):
    if not {"slide_id","partition"}.issubset(u): raise RuntimeError("Universe requires slide_id, partition")
    u=u[["slide_id","partition"]].drop_duplicates().copy()
    if u["slide_id"].duplicated().any(): raise RuntimeError("Cross-partition duplicate slide IDs in AGGC universe")
    counts=u.groupby("partition")["slide_id"].nunique().to_dict()
    obs=(int(counts.get("train_validation",0)),int(counts.get("test_replication",0)),int(u.slide_id.nunique()))
    if obs!=(105,45,150): raise RuntimeError(f"AGGC universe mismatch: observed Train/Test/Total={obs}; expected (105,45,150)")
    return u.sort_values(["partition","slide_id"]).reset_index(drop=True)

def against(df,u,label):
    if not {"slide_id","partition"}.issubset(df): raise RuntimeError(f"{label} lacks slide_id/partition")
    allowed=set(zip(u.partition,u.slide_id)); seen=set(zip(df.partition,df.slide_id)); extra=seen-allowed
    if extra: raise RuntimeError(f"{label} has identities outside frozen universe: {sorted(extra)[:10]}")

def coverage(endpoint,df,u,note):
    rows=[]
    for p in EXPECTED:
        den=int((u.partition==p).sum()); n=int(df.loc[df.partition==p,"slide_id"].nunique())
        rows.append(dict(endpoint=endpoint,partition=p,n_raw_universe_slides=den,n_endpoint_slides=n,
                         coverage_fraction=n/den if den else np.nan,note=note))
    return rows

def args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step19b",type=Path,default=STEP19B)
    p.add_argument("--step20b",type=Path,default=STEP20B)
    p.add_argument("--step20e",type=Path,default=STEP20E)
    p.add_argument("--step21c",type=Path,default=STEP21C)
    p.add_argument("--universe-csv",type=Path,default=None,help="Optional frozen 150-slide manifest with slide_id,partition")
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()

def main():
    a=args()
    files={
      "state_unique":a.step19b/"aggc_pc3_available_case_slide_state_measurements.csv",
      "state_threshold":a.step19b/"aggc_pc3_available_case_threshold_measurements.csv",
      "morph":a.step20b/"aggc_pc3_within_class_slide_state_metric_measurements.csv.gz",
      "morph_qc":a.step20b/"aggc_pc3_within_class_slide_state_qc.csv",
      "auc":a.step21c/"aggc_pc3_tumor_auc_per_slide.csv",
      "auc_summary":a.step21c/"aggc_pc3_tumor_auc_partition_summary.csv",
      "test_cont":a.step20e/"aggc_test_pc3_morphology_continuous_per_slide.csv.gz",
      "test_hilo":a.step20e/"aggc_test_pc3_morphology_high_low_per_slide.csv.gz",
    }
    for k in ["state_unique","state_threshold","morph","morph_qc","auc","auc_summary"]: req(files[k],k)
    prep(a.output,a.overwrite)
    state=load(files["state_unique"]); sth=load(files["state_threshold"]); morph=load(files["morph"]); mqc=load(files["morph_qc"]); auc=load(files["auc"])

    # Prefer explicit raw-universe manifest. Otherwise reconstruct only if Step19B contains all 150 slides.
    if a.universe_csv:
        req(a.universe_csv,"explicit AGGC universe"); universe=validate_universe(load(a.universe_csv))
    else:
        universe=state[["slide_id","partition"]].drop_duplicates()
        try: universe=validate_universe(universe)
        except RuntimeError as e:
            raise RuntimeError(str(e)+"\nStep19B is available-case and may not contain the raw 150-slide universe. Rerun with --universe-csv <frozen_manifest.csv>.")

    for d,l in [(state,"Step19B state"),(sth,"Step19B threshold"),(morph,"Step20B morphology"),(mqc,"Step20B QC"),(auc,"Step21C AUC")]: against(d,universe,l)

    if "threshold_tiles_per_state" not in sth: raise RuntimeError("Step19B threshold table lacks threshold_tiles_per_state")
    primary=sth.loc[pd.to_numeric(sth.threshold_tiles_per_state,errors="raise").astype(int)==PRIMARY_THRESHOLD].copy()
    if "threshold_role" in primary and not primary.threshold_role.astype(str).str.upper().eq("PRIMARY").all():
        raise RuntimeError("Threshold 50 is not uniformly marked PRIMARY")
    key=["partition","slide_id","histologic_state"]
    if primary.duplicated(key).any(): raise RuntimeError("Duplicate primary slide-state rows")

    if "threshold_tiles_per_state" in morph:
        morph_primary=morph.loc[pd.to_numeric(morph.threshold_tiles_per_state,errors="raise").astype(int)==PRIMARY_THRESHOLD].copy()
    else: morph_primary=morph.copy()

    cov=[]
    cov+=coverage("histologic_state_threshold50",primary,universe,"Frozen available-case state support; >=50 eligible tiles/state/slide.")
    cov+=coverage("within_state_morphology_step20b",morph_primary,universe,"Frozen morphology availability and state-support rules.")
    cov+=coverage("tumor_localization_auc_step21c",auc,universe,"Frozen G3-G5 vs Normal/Stroma endpoint; >=10 tumor and >=10 non-tumor; Test provenance restrictions retained.")
    for k,name in [("test_cont","test_pc3_morphology_continuous_step20e"),("test_hilo","test_pc3_morphology_high_low_step20e")]:
        if files[k].is_file():
            d=load(files[k]); against(d,universe,k); cov+=coverage(name,d,universe,"Formal Test-only Step20E endpoint.")
    cov=pd.DataFrame(cov)

    audit=[]
    for p,nexp in EXPECTED.items():
        n=int(universe.loc[universe.partition==p,"slide_id"].nunique()); audit.append(dict(audit_item=f"{p}_raw_unique_slides",observed=n,expected=nexp,passed=n==nexp))
    audit.append(dict(audit_item="combined_raw_unique_slides",observed=int(universe.slide_id.nunique()),expected=150,passed=int(universe.slide_id.nunique())==150))
    audit=pd.DataFrame(audit)
    if not audit.passed.all(): raise RuntimeError(audit.to_string(index=False))

    universe.to_csv(a.output/"combined_aggc_slide_universe.csv",index=False)
    state.to_csv(a.output/"combined_aggc_pc3_slide_state_measurements.csv",index=False)
    primary.to_csv(a.output/"combined_aggc_pc3_primary_state_measurements.csv",index=False)
    auc.to_csv(a.output/"combined_aggc_pc3_tumor_auc_per_slide.csv",index=False)
    morph_primary.to_csv(a.output/"combined_aggc_pc3_within_state_morphology_measurements.csv.gz",index=False,compression="gzip")
    cov.to_csv(a.output/"combined_aggc_endpoint_coverage.csv",index=False)
    audit.to_csv(a.output/"combined_aggc_partition_audit.csv",index=False)

    hashes=[]
    for role,p in files.items():
        if p.is_file(): hashes.append(dict(role=role,path=str(p),sha256=sha(p)))
    if a.universe_csv: hashes.append(dict(role="explicit_universe",path=str(a.universe_csv),sha256=sha(a.universe_csv)))
    pd.DataFrame(hashes).to_csv(a.output/"source_hashes.csv",index=False)

    meta=dict(version=VERSION,schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),python_version=platform.python_version(),
              role="assembly_and_audit_only",raw_universe=dict(train=105,test=45,total=150),primary_state_threshold=50,
              guardrails=dict(pca_refit=False,pc3_sign_flip=False,state_redefinition=False,tumor_class_redefinition=False,
                              support_threshold_redefinition=False,auc_recalculated=False,morphology_recalculated=False,
                              new_p_values=False,new_model_fit=False,outcome_based_exclusion=False))
    (a.output/"analysis_metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
    (a.output/"README.txt").write_text("""STEP31A combines already-frozen AGGC Train and Test PC3 outputs while preserving partition labels and endpoint-specific eligibility.\n\nRaw universe: Train 105, Test 45, total 150 unique slides.\n\nNo PCA refit, sign selection, endpoint redefinition, morphology/AUC recalculation, p-values, or model fitting occurs here. Use combined_aggc_endpoint_coverage.csv for endpoint-specific N.\n""",encoding="utf-8")

    print("\nSTEP31A PASS")
    print(audit.to_string(index=False))
    print("\nEndpoint coverage:")
    print(cov.to_string(index=False))
    print(f"\nSaved to: {a.output}")

if __name__=="__main__": main()
