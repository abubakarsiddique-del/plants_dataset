# Plant Disease Clustering Project

## Overview

This project builds a traceable data pipeline for plant leaf disease analysis using a hybrid feature extraction approach and unsupervised clustering.

Key components:
- `augmentation_pipeline.py`: ingest the Malabar dataset, validate originals, generate 12 deterministic augmentations per image, and save QA review grids.
- `build_manifest.py`: build a master manifest CSV and JSON summary with metadata, SHA256 hashes, and a run-level `run_id`.
- `feature_extraction.py`: compute hybrid features from letterboxed images using ResNet18 embeddings plus classical color/texture descriptors, then standardize and PCA-reduce features.
- `fuzzy_cmeans_pipeline.py`: run Fuzzy C-Means clustering on originals-only PCA features and save cluster centers, membership data, confusion matrices, and run summaries.
- `fcm_evaluation_artifacts.py`: generate embeddings, cluster montages, stability reports, and high-entropy uncertainty artifacts.
- `validate_outputs.py`: assert manifest consistency and summary integrity.

## Data and Artifacts

Dataset structure:
- `Malabar_Dataset/`: 3,006 original images across 5 disease classes.
- `augmented_dataset/`: 12 augmentations per original, yielding 36,072 augmented images.
- `augmentation_review_grids/`: QA grids for sampled augmentations.

Manifest and summary:
- `logs/master_manifest.csv`
- `logs/master_manifest_summary.json`

Feature extraction artifacts:
- `features/feature_matrix_raw.npy`
- `features/feature_matrix_scaled.npy`
- `features/feature_matrix.npy`
- `features/originals_pca64_features.npy`
- `features/feature_index.csv`
- `features/scaler.joblib`
- `features/pca.joblib`

FCM clustering artifacts:
- `features/fcm_results/fcm_c5/summary.json`
- `features/fcm_results/fcm_best_run.json`
- `features/fcm_results/fcm_c5/confusion_matrix.csv`
- `features/fcm_results/fcm_c5/membership_matrix.npy`
- `features/fcm_results/fcm_c5/hard_labels.npy`
- `features/fcm_results/eval/` QA artifacts

## Pipeline Details

### Augmentation

The augmentation pipeline applies 12 isolated Albumentations transforms:
- horizontal_flip
- vertical_flip
- rotation
- zoom_scale
- brightness
- contrast
- hue_saturation
- gaussian_blur
- gaussian_noise
- shear
- clahe
- translation

It produces deterministic outputs using a per-image seed derived from the filename.

### Feature extraction

Hybrid feature vector composition:
- Deep embedding: ResNet18 global pooling output (512 dims)
- Classical features: 1067 dims
  - HSV histogram (8x8x8 = 512)
  - LAB histogram (8x8x8 = 512)
  - GLCM statistics (4)
  - LBP histogram (26)
  - Hu moments (7)
  - lesion/leaf heuristic stats (6)

Post-processing:
- StandardScaler fit on hybrid feature vectors
- PCA reduction to 128 dims
- Recorded explained variance: ~0.705 total variance preserved

### Clustering

FCM is executed using originals-only PCA features.
- Configured clusters: `[5, 6, 7]`
- Best run selected by FPC
- FCM hyperparameters: `m=2.0`, `error=1e-5`, `maxiter=1000`, `seed=42`

## Results

### Extraction results

From `features/extraction_summary.json`:
- `manifest_rows_requested`: 39,078
- `rows_processed`: 39,078
- `rows_failed`: 0
- `final_feature_shape`: [39,078, 128]
- PCA captured `70.48%` of the variance in 128 components

### Fuzzy C-Means results

From `features/fcm_results/fcm_c5/summary.json` and `features/fcm_results/fcm_best_run.json`:
- Best cluster count: `c=5`
- `fpc`: 0.2000
- `nmi`: 0.00744
- `ari`: 0.00419
- `membership_entropy_mean`: ~1.0
- Cluster sizes:
  - `cluster_0`: 1,618
  - `cluster_4`: 1,387
  - `cluster_1`: 1

### Evaluation metrics

From `features/fcm_results/eval/evaluation_summary.json`:
- `stable_percent`: 32.44%
- Only ~1/3 of parent image groups have stable cluster assignments across augmentations.
- Generated artifacts include:
  - t-SNE and UMAP plots by true class and cluster assignment
  - per-cluster top-10 image montages
  - augmentation stability CSV
  - top-200 highest-entropy image report
  - high-entropy montage

## Interpretation

The current unsupervised pipeline is producing weak class alignment.

Evidence:
- Near-zero `NMI` and `ARI` values indicate cluster assignments are almost uncorrelated with true labels.
- Extremely high membership entropy (~1.0) means the FCM model is highly uncertain about sample assignments.
- Confusion matrix shows clusters contain mixed classes and one cluster is effectively unused.
- Only 32% of augmentation parent groups are stable across the cluster assignment of their children.

This suggests the current feature representation and / or clustering strategy is not capturing disease-specific structure strongly enough.

## Suggested improvements

### Data and features

1. Separate augmented and original feature processing.
   - Ensure original-only features are not biased by augmentation artifacts.

2. Rebalance or reweight feature contributions.
   - The 512-d deep embedding may dominate the classical descriptors.
   - Consider feature selection, per-block PCA, or normalized concatenation.

3. Improve classical descriptors.
   - Validate if HSV/LAB histograms and texture features actually separate disease classes.
   - Experiment with additional leaf-specific shape or segmentation features.

4. Use a more task-specific embedding.
   - Fine-tune ResNet18 on leaf images with labels, or use a model pretrained on plant pathology data.

### Clustering strategy

1. Try supervised or semi-supervised approaches.
   - Given ground truth labels are known for originals, classification may be more appropriate than pure clustering.

2. Add baseline clustering methods for comparison.
   - KMeans, GaussianMixture, spectral clustering, or hierarchical clustering.
   - Use silhouette and Davies-Bouldin score in addition to NMI/ARI.

3. Tune FCM hyperparameters more broadly.
   - Explore `m` values beyond 2.0 and different initialization methods.
   - Search cluster counts in a wider range (2-12) and compare stability.

### Pipeline and production hardening

1. Add config files for all pipeline stages, not just FCM.
   - Current `config.yaml` only covers FCM.

2. Version outputs with `run_id` and add artifact metadata across extraction and evaluation.

3. Add automated validation and unit tests.
   - `validate_outputs.py` is a good start; extend it with feature dimension and cluster artifact checks.

4. Add documentation and reproducible commands.
   - This `PROJECT.md` should be supplemented with a root `README.md` if desired.

## Recommended next steps

1. Re-run feature extraction with a tighter feature selection or reduced classical feature set.
2. Compare FCM to KMeans on the same original PCA features.
3. Inspect `features/fcm_results/eval/umap_cluster_assignment.png` and `tsne_cluster_assignment.png` to see whether clusters are just split by global appearance, not disease.
4. Consider a label-aware model or downstream classifier if unsupervised clustering is not the core objective.

## Execution commands

```bash
./venv/bin/python3 augmentation_pipeline.py
./venv/bin/python3 build_manifest.py
./venv/bin/python3 feature_extraction.py --no-resume
./venv/bin/python3 fuzzy_cmeans_pipeline.py --config config.yaml
./venv/bin/python3 fcm_evaluation_artifacts.py
./venv/bin/python3 validate_outputs.py
```

## Notes

- The project currently uses `torch==2.1.1`, `opencv-python-headless==4.11.0.72`, `scikit-fuzzy==0.4.2`, and `PyYAML==6.0`.
- The current pipeline is strong on traceability and artifact generation, but the clustering quality needs more targeted feature or model work.
