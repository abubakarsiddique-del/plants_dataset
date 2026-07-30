# Project Metrics

## Dataset and augmentation

- Original images: `3006`
- Augmented images: `36072`
- Total manifest rows: `39078`
- Unique sources: `3006`
- Augmentations per image: `12`

## Feature extraction

- Hybrid feature dimension before PCA: `1579`
- PCA components: `128`
- Final feature matrix shape: `39078 x 128`
- Variance retained by PCA: `70.48%`
- Runtime: `2657.73` seconds

## Fuzzy C-Means clustering results

- Best cluster count: `5`
- Number of original samples: `3006`
- Feature dimensionality used for clustering: `64`
- FPC: `0.2000`
- NMI: `0.00744`
- ARI: `0.00419`
- Mean membership entropy: `0.9999999999999413`
- Cluster counts:
  - `cluster_0`: `1618`
  - `cluster_4`: `1387`
  - `cluster_1`: `1`

## Evaluation summary

- Stable augmentation parents: `975`
- Total parents evaluated: `3006`
- Stability percentage: `32.44%`

## Interpretation

- Clusters are poorly aligned with true disease labels.
- High entropy indicates fuzzy assignments are nearly uniform.
- Only one cluster is effectively small, making the solution behave like a 2-way split.
- The current feature representation and clustering strategy need refinement.

## Artifact locations

- Manifest: `logs/master_manifest.csv`
- Manifest summary: `logs/master_manifest_summary.json`
- Feature index: `features/feature_index.csv`
- PCA features: `features/originals_pca64_features.npy`
- Best FCM summary: `features/fcm_results/fcm_best_run.json`
- Evaluation summary: `features/fcm_results/eval/evaluation_summary.json`
- Evaluation plots: `features/fcm_results/eval/*.png`
- Stability report: `features/fcm_results/eval/augmentation_stability.csv`
- High entropy report: `features/fcm_results/eval/entropy_highest_200.csv`
