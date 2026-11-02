# Continuous Histologic Phenotypes in Prostate Pathology Foundation-Model Representations

Code and frozen PCA parameters accompanying a study of continuous histologic variation in prostate H&E whole-slide images using Virchow2 representations.

## Overview

Pathology foundation models are commonly evaluated through downstream prediction tasks, while the histologic variation represented by individual embedding dimensions remains less well characterized.

We applied principal component analysis (PCA) to 2,560-dimensional Virchow2 tile embeddings from a prostate histology discovery cohort. PCA stability, spatial organization, tumor associations, and quantitative morphology were examined. The fitted PCA transformation was then applied without refitting to independent PESO, AGGC2022, and TCGA-PRAD cohorts.

The analysis focuses on two components:

- **PC3:** A stable component associated with glandular–stromal variation and epithelial composition.
- **PC6:** A secondary component associated with histologic variation distinct from PC3.

The study characterizes histologic patterns within foundation-model representations rather than developing a supervised tumor classifier.

## Datasets

| Dataset | Analysis |
|---|---|
| Prostate Fused-MRI-Pathology | PCA discovery, stability, internal morphology, and tumor associations |
| PESO | Epithelial-compartment validation |
| AGGC2022 | Stromal composition, Gleason-associated histologic states, morphology, and tumor localization |
| TCGA-PRAD | Clinicopathologic and molecular associations |

Datasets are not redistributed in this repository. Access and usage conditions are determined by the respective data providers.

## Repository Structure

    prostate-histologic-phenotypes/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── project_frozen_pca.py
    ├── frozen_parameters/
    │   ├── global_pca_model.npz
    │   └── pc1_pc8_frozen_direction_specification.csv
    └── scripts/
        ├── pca_stability/
        ├── internal_morphology/
        ├── peso_validation/
        ├── aggc_pc3_validation/
        ├── pc6_characterization/
        └── tcga_prad/

## Installation

Install the Python dependencies:

    python -m pip install -r requirements.txt

Additional dependencies may be required for individual analysis scripts.

## Frozen PCA Projection

The frozen PCA model is provided in `frozen_parameters/global_pca_model.npz`. It contains the feature standardization parameters, PCA loadings, centering parameters, and explained variance ratios for eight components.

The projection utility accepts NumPy arrays of compatible 2,560-dimensional Virchow2 tile embeddings.

To calculate raw PC1–PC8 scores:

    python project_frozen_pca.py \
        --embeddings /path/to/virchow2_embeddings.npy \
        --output /path/to/pc_scores.npy

To apply the frozen tumor-orientation convention:

    python project_frozen_pca.py \
        --embeddings /path/to/virchow2_embeddings.npy \
        --output /path/to/oriented_pc_scores.npy \
        --tumor-oriented

The orientation rules are stored in `frozen_parameters/pc1_pc8_frozen_direction_specification.csv`.

| Component | Discovery tumor association | Orientation multiplier |
|---|---|---|
| PC3 | Lower scores in tumor | -1 |
| PC4 | Lower scores in tumor | -1 |
| PC6 | Higher scores in tumor | +1 |

Components without a specified tumor direction retain their original PCA signs.

Raw and tumor-oriented scores should not be mixed within an analysis.

## Analysis Scripts

| Directory | Contents |
|---|---|
| `pca_stability/` | Patient bootstrap, leave-one-patient-out PCA, nested validation, axis stability, and PC5–PC6 subspace analysis |
| `internal_morphology/` | PC3 morphology sampling, quantitative measurements, and association analyses |
| `peso_validation/` | External epithelial-compartment validation |
| `aggc_pc3_validation/` | PC3 stromal, Gleason-state, and tumor-localization analyses |
| `pc6_characterization/` | Internal and external PC6 morphology and tumor-localization analyses |
| `tcga_prad/` | Frozen PCA transfer, clinicopathologic associations, and gene-level molecular analyses |

All directories are located under `scripts/`.

## Reproducibility

The repository includes the frozen PCA parameters, direction specification, and selected analysis scripts supporting the study.

The PCA projection utility was checked against saved discovery-cohort scores. Other scripts retain dependencies on the original research workflow, including local data paths and intermediate analysis files, and have not yet been validated as a standalone pipeline.

Whole-slide images, tile embeddings, clinical records, and intermediate analysis outputs are not included.

## Citation

Citation details will be added upon manuscript publication.

## License

The original code in this repository is released under the MIT License.
See [LICENSE](LICENSE).

Third-party datasets, whole-slide images, and other external resources
remain subject to their respective licenses and data-use terms.
