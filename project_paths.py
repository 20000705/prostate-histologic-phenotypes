"""
Repository path configuration.

External datasets and intermediate results are not distributed
with this repository.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


DATA_ROOT = env_path(
    "PROSTATE_DATA_ROOT",
    REPO_ROOT / "data",
)

RESULTS_ROOT = env_path(
    "PROSTATE_RESULTS_ROOT",
    REPO_ROOT / "results",
)

FROZEN_PARAMETERS = REPO_ROOT / "frozen_parameters"

FROZEN_PCA_MODEL = (
    FROZEN_PARAMETERS / "global_pca_model.npz"
)

FROZEN_DIRECTIONS = (
    FROZEN_PARAMETERS
    / "pc1_pc8_frozen_direction_specification.csv"
)

DISCOVERY_EMBEDDINGS = env_path(
    "PROSTATE_DISCOVERY_EMBEDDINGS",
    DATA_ROOT / "discovery" / "wsi_embeddings",
)

DISCOVERY_WSI = env_path(
    "PROSTATE_DISCOVERY_WSI",
    DATA_ROOT / "discovery" / "slides",
)

DISCOVERY_ANNOTATIONS = env_path(
    "PROSTATE_DISCOVERY_ANNOTATIONS",
    DATA_ROOT / "discovery" / "annotations",
)

PESO_ROOT = env_path(
    "PROSTATE_PESO_ROOT",
    DATA_ROOT / "peso",
)

AGGC_ROOT = env_path(
    "PROSTATE_AGGC_ROOT",
    DATA_ROOT / "aggc2022",
)

TCGA_ROOT = env_path(
    "PROSTATE_TCGA_ROOT",
    DATA_ROOT / "tcga_prad",
)
