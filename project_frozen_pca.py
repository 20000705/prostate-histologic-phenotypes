#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project Virchow2 tile embeddings into the frozen discovery PCA space.

Input:
    NumPy array of shape (N, 2560)

Output:
    Raw PCA scores of shape (N, 8)

The PCA model is never refitted.
Raw PC scores retain the original discovery PCA orientation.

Tumor-oriented scores are optional and use the frozen
discovery-only direction specification.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = ROOT / "frozen_parameters" / "global_pca_model.npz"
DEFAULT_DIRECTIONS = (
    ROOT / "frozen_parameters"
    / "pc1_pc8_frozen_direction_specification.csv"
)


def project_frozen_pca(embeddings, model_path, chunk_size=50000):
    with np.load(model_path, allow_pickle=False) as model:
        mean = np.asarray(model["scaler_mean"], dtype=np.float64)
        scale = np.asarray(model["scaler_scale"], dtype=np.float64)
        pca_mean = np.asarray(model["pca_mean"], dtype=np.float64)
        components = np.asarray(model["pca_components"], dtype=np.float64)

    if embeddings.ndim != 2 or embeddings.shape[1] != mean.size:
        raise ValueError(
            f"Expected embeddings (N, {mean.size}), "
            f"received {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("No embedding rows supplied.")

    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Invalid frozen scaler parameters.")

    if np.any(scale <= 0):
        raise ValueError("Invalid scaler scale.")

    if not np.isfinite(pca_mean).all() or not np.isfinite(components).all():
        raise ValueError("Invalid frozen PCA parameters.")

    if components.shape[1] != mean.size:
        raise ValueError("PCA component dimension mismatch.")

    scores = np.empty(
        (embeddings.shape[0], components.shape[0]),
        dtype=np.float32,
    )

    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])

        x = np.asarray(embeddings[start:stop], dtype=np.float64)

        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite embeddings at rows {start}:{stop}")

        x_scaled = (x - mean) / scale
        x_pca = (x_scaled - pca_mean) @ components.T

        scores[start:stop] = x_pca.astype(np.float32)

    return scores


def apply_frozen_tumor_orientation(raw_scores, specification_path):
    df = pd.read_csv(specification_path)

    oriented = raw_scores.copy()

    for _, row in df.iterrows():
        if row["specification_status"] != "frozen":
            continue

        component = str(row["component"])
        index = int(component.replace("PC", "")) - 1
        multiplier = float(row["score_multiplier"])

        if index < 0 or index >= oriented.shape[1]:
            raise ValueError(f"Invalid component: {component}")

        if multiplier not in (-1.0, 1.0):
            raise ValueError(f"Invalid multiplier for {component}")

        oriented[:, index] *= multiplier

    return oriented


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--directions",
        type=Path,
        default=DEFAULT_DIRECTIONS,
    )
    parser.add_argument(
        "--tumor-oriented",
        action="store_true",
        help="Apply discovery-frozen direction multipliers.",
    )
    parser.add_argument("--chunk-size", type=int, default=50000)

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    embeddings = np.load(
        args.embeddings,
        mmap_mode="r",
        allow_pickle=False,
    )

    scores = project_frozen_pca(
        embeddings,
        args.model,
        args.chunk_size,
    )

    if args.tumor_oriented:
        scores = apply_frozen_tumor_orientation(
            scores,
            args.directions,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, scores)

    print(f"Input: {args.embeddings}")
    print(f"Output: {args.output}")
    print(f"Shape: {scores.shape}")
    print(f"Tumor-oriented: {args.tumor_oriented}")
    print("[PASS] Frozen PCA projection completed.")


if __name__ == "__main__":
    main()
