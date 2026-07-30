# Plant Disease Clustering Pipeline

This repository builds a traceable plant disease dataset pipeline for image augmentation, hybrid feature extraction, and unsupervised clustering.

## What it does

- Validates the original `Malabar_Dataset` image collection
- Generates 12 deterministic augmented versions per original image
- Builds a manifest with SHA256 hashes and metadata
- Extracts hybrid image features using frozen ResNet18 embeddings plus classical descriptors
- Applies PCA and runs Fuzzy C-Means clustering on originals-only feature embeddings
- Produces evaluation artifacts: t-SNE/UMAP plots, cluster montages, stability reports, and entropy analyses

## Key files

- `augmentation_pipeline.py`: augmentation and review grid generation
- `build_manifest.py`: master manifest CSV and summary JSON creation
- `feature_extraction.py`: hybrid feature extraction and PCA pipeline
- `fuzzy_cmeans_pipeline.py`: FCM clustering with summary output
- `fcm_evaluation_artifacts.py`: cluster QA and visualization artifacts
- `validate_outputs.py`: output consistency checks
- `config.yaml`: FCM hyperparameters
- `PROJECT.md`: detailed project analysis and improvement suggestions

## Requirements

Use the project virtual environment at `venv`.

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Running the pipeline

1. Augment dataset and build QA grids:

```bash
./venv/bin/python3 augmentation_pipeline.py
```

2. Build the manifest CSV/JSON:

```bash
./venv/bin/python3 build_manifest.py
```

3. Extract features:

```bash
./venv/bin/python3 feature_extraction.py --no-resume
```

4. Run Fuzzy C-Means clustering:

```bash
./venv/bin/python3 fuzzy_cmeans_pipeline.py --config config.yaml
```

5. Generate evaluation artifacts:

```bash
./venv/bin/python3 fcm_evaluation_artifacts.py
```

6. Validate manifest and summary consistency:

```bash
./venv/bin/python3 validate_outputs.py
```

## Notes

- The current pipeline is optimized for reproducibility.
- `config.yaml` contains the FCM hyperparameter configuration.
- The clustering results indicate weak label alignment, so supervised or feature-tuning improvements are recommended.
