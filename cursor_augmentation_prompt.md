# Cursor Prompt — Full-Dataset Leaf Image Augmentation Pipeline

Paste everything below into Cursor's chat/composer (with your venv active in the project containing the extracted `Malabar_Dataset/` folder).

---

## ROLE

Act as a senior computer vision engineer. Build and then **execute** a production-quality Python augmentation pipeline for a plant leaf disease image dataset. This is preprocessing for a downstream fuzzy clustering pipeline, so correctness, traceability, and reproducibility matter more than speed.

## DATASET CONTEXT (already verified — do not re-derive, just confirm on run)

- Root folder: `Malabar_Dataset/` containing exactly 5 class subfolders, each with images directly inside (no further nesting):
  - `Anthracnose(102)` — 102 images
  - `Bacterial-Spot(752)` — 752 images
  - `Downy-Mildew(240)` — 240 images
  - `Healthy-Leaf(1399)` — 1,399 images
  - `Pest-Damage(513)` — 513 images
- **3,006 images total.** All `.jpg`/`.JPG`, RGB.
- Filenames follow the pattern `ClassName (N).jpg`, e.g. `Anthracnose (17).jpg`.
- **Resolution is highly inconsistent across the dataset** (observed range: ~300×400 up to ~3060×4080). This MUST be normalized to a fixed working size before augmentation, or transform strength (blur radius, noise sigma, shift %) will not behave consistently image-to-image.
- If you extract the zip and the root folder is named differently or nested one level deeper, auto-detect it — don't hardcode a path that breaks silently.

## THE JOB — RUN ON THE ENTIRE DATASET, NOT A SAMPLE

Process **all 3,006 images across all 5 class folders**. No shortcuts, no "process one image as a demo and stop." Optionally do a fast internal test on 2–3 images first purely to confirm the pipeline logic works before launching the full batch — but the full 3,006-image run is the actual deliverable, and it must complete unattended.

Execute these steps **sequentially**:

### Step 0 — Environment setup
Install/verify: `albumentations`, `opencv-python-headless`, `pillow`, `numpy`, `tqdm`. Print installed versions. If any Albumentations parameter name below has changed in the installed version, use the closest current equivalent that preserves the described effect — test on one image before the full run.

### Step 1 — Config block (top of script, all tunables in one place)
```python
INPUT_DIR = "Malabar_Dataset"
OUTPUT_AUGMENTED_DIR = "augmented_dataset"       # individual augmented images
OUTPUT_GRIDS_DIR = "augmentation_review_grids"   # the requested comparison files
LOG_DIR = "logs"
WORKING_SIZE = (640, 640)     # letterbox pad+resize target before augmenting
THUMB_SIZE = (300, 300)       # size of each cell in the review grid
GRID_COLS = 4
JPEG_QUALITY = 90
RANDOM_SEED = 42
```

### Step 2 — Dataset discovery & validation
Walk `INPUT_DIR`, confirm 5 class folders and per-class counts match what's expected. Attempt to open every file with PIL; log (don't crash on) any unreadable/corrupt file to `logs/corrupt_files.csv` and skip it in the run.

### Step 3 — Standardize each image before augmenting
Resize each image so it fits within `WORKING_SIZE` preserving aspect ratio, then letterbox-pad to exactly `WORKING_SIZE` with white fill (do not center-crop — leaf lesions near the edge must not be cut off). All downstream augmentation and grid-building operates on this standardized version.

### Step 4 — Define exactly 12 augmentation techniques (isolated, not stacked)
Each of the 12 outputs must reflect **one technique in isolation** so the review grid clearly attributes each visible effect to a single named transform. Use Albumentations with `p=1.0` on the single transform, applied to the Step 3 standardized image:

| # | Name | Transform | Suggested params |
|---|------|-----------|-------------------|
| 1 | Horizontal Flip | `HorizontalFlip` | — |
| 2 | Vertical Flip | `VerticalFlip` | — |
| 3 | Rotation | `Rotate` | `limit=40, border_mode=cv2.BORDER_REFLECT_101` |
| 4 | Zoom / Scale | `Affine` | `scale=(0.75, 1.25)` |
| 5 | Brightness | `RandomBrightnessContrast` | `brightness_limit=0.4, contrast_limit=0.0` |
| 6 | Contrast | `RandomBrightnessContrast` | `brightness_limit=0.0, contrast_limit=0.4` |
| 7 | Hue/Saturation Jitter | `HueSaturationValue` | `hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=0` |
| 8 | Gaussian Blur | `GaussianBlur` | `blur_limit=(3,7)` |
| 9 | Gaussian Noise | `GaussNoise` | std/var equivalent to sigma ≈ 10–25 on 0–255 scale |
| 10 | Shear | `Affine` | `shear=(-15, 15)` |
| 11 | CLAHE | `CLAHE` | `clip_limit=2.0, tile_grid_size=(8,8)` — enhances lesion/texture contrast |
| 12 | Translation/Shift | `Affine` | `translate_percent=(-0.15, 0.15)` |

Use a **per-image deterministic seed** (e.g. `RANDOM_SEED + hash(filename) % 100000`) so reruns are reproducible per file while still varying across the dataset.

### Step 5 — Per-image processing function
For each original image:
1. Load, validate, standardize (Step 3).
2. Apply each of the 12 transforms independently → 12 augmented images.
3. Save each to `augmented_dataset/<ClassName>/<original_stem>_aug##_<technique_slug>.jpg` (needed later for the fuzzy clustering stage — this is the actual expanded dataset, distinct from the review file below).
4. Build **one composite review file**: a labeled grid with the original in the first cell (bordered/labeled "ORIGINAL" distinctly) followed by the 12 augmented cells, each labeled with its technique name underneath. `GRID_COLS=4` → 4 rows (13 cells used, 3 left blank). Add a header bar with `<ClassName> / <filename>` for traceability. Save to `augmentation_review_grids/<ClassName>/<original_stem>_review.jpg`.

### Step 6 — Batch loop over the full dataset
Iterate every image in every class folder (all 3,006) with a `tqdm` progress bar. Wrap each image in `try/except` so one bad file never kills the run — log failures and continue. **Skip images whose review grid already exists** so the script is safely resumable if interrupted midway.

### Step 7 — Logging & summary
- `logs/augmentation_log.csv`: one row per original image (class, filename, status, processing time, error if any).
- `logs/augmentation_summary.json`: totals — images processed, skipped, failed; per-class counts; total runtime; average time/image; total output disk usage; total files generated (expected: 3,006 × 12 = 36,072 augmented images + 3,006 review grids = 39,078 files).

### Step 8 — Post-run sanity check
Re-open 10 random generated review grids and 10 random augmented images to confirm they're valid, non-corrupt image files. Print a final human-readable summary to console.

## ACCEPTANCE CHECKLIST

- [ ] Full 3,006-image dataset processed — not a subset
- [ ] Exactly 12 distinct, isolated augmentation techniques per image, each visually labeled
- [ ] One review grid file per original image (original + 12 labeled augmented cells)
- [ ] Individual augmented images also saved per class (for the fuzzy clustering stage)
- [ ] Class folder structure and filename traceability preserved throughout
- [ ] Corrupt/unreadable files logged and skipped, not crashing the run
- [ ] Run is resumable (safe to rerun without duplicating completed work)
- [ ] Final summary (counts, runtime, disk usage) printed and saved to `logs/`
