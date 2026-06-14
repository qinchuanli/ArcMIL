# ArcMIL

This repository contains the current public implementation of **ArcMIL** used for manuscript review.
Some components are still being cleaned and documented for a fuller release. The current version
prioritizes the core angular representation, offline center computation, abnormal-center screening,
pseudo-label generation, and the main training pipeline.

## Current Release Scope

The current public code focuses on the following modules:

- angular coordinate transformation
- offline reduced-angular center computation
- abnormal center screening
- pseudo-label generation
- ArcMIL training entry

This is a **partial release for the review stage**. The codebase is partially cleaned for release,
some engineering modules remain under reorganization, and the final data pipeline together with some
internal naming will be further refined.

## Minimal Workflow

1. Prepare pre-extracted features for the target dataset.
2. Run `compute_centers.py` to compute class-wise centers, abnormal-center indices, and selected dimensions.
3. Run `main.py` for ArcMIL training and evaluation.

## Minimal Dependencies

The current public release includes a minimal `requirements.txt` for the main experimental entry points.
You can install the basic dependencies with:

```bash
pip install -r requirements.txt
```

## Example Commands

Offline center computation:

```bash
python compute_centers.py --dataset camelyon16 --kn 5 --kt 5
```

Training:

```bash
python main.py --dataset camelyon16 --k0 5 --k1 5
```

Dataset-specific settings are currently managed through `config.py` together with CLI arguments.

## Key Output Files

The offline center computation stage typically produces:

- `centers.npy`: class-wise centers in the released angular pipeline
- `abnormal_indices.npy`: retained abnormal-center indices
- `selected_dims_top128.json`: selected discriminative dimensions

Training typically produces:

- checkpoints
- logs
- evaluation metrics

## Current Limitations

- This repository does not yet provide a fully cleaned end-to-end data preprocessing platform.
- Some internal modules still preserve legacy naming or experimental branches that are being reorganized.
- The current release is intended to expose the main ArcMIL pipeline rather than a finalized reproducibility package.
