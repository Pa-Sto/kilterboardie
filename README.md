# Kilterboardie

Kilterboard route-generation research with a tensor dataset, conditional VAE, conditional diffusion model, and hierarchical graph Transformer.

This repository contains model implementations, training and local inference tools, dataset preparation utilities, and research documentation. See [DATASET.md](DATASET.md) for the data representation and collection limitations.

## Website

The [Kilterboardie generator](https://pa-sto.github.io/kilterboardie/) offers three models for 50-degree V3-V13 climbs: the conditional VAE, diffusion (the default), and the hierarchical graph Transformer. Select a grade and model, then generate a new climb on the reference board. Generated routes are experimental proposals, not validated climbs.

## Download the NumPy Dataset

The **Boardsesh dataset** contains **37,051 unique Kilter Original routes at 50 degrees, V3-V13**, converted into training-ready NumPy arrays.

- [Download the dataset ZIP (226 MB)](https://github.com/Pa-Sto/kilterboardie/releases/download/dataset-boardsesh-50degree-20260930/kilterboardie-boardsesh-50degree-20260930-numpy.zip)
- [Release notes and checksums](https://github.com/Pa-Sto/kilterboardie/releases/tag/dataset-boardsesh-50degree-20260930)
- [Source and snapshot provenance](https://github.com/Pa-Sto/kilterboardie/releases/download/dataset-boardsesh-50degree-20260930/SOURCE.json)

The losslessly compressed archive expands to approximately **1.8 GB**. It is distributed as a GitHub Release asset, not included in a repository clone. Extract it from the project root:

```bash
unzip /path/to/kilterboardie-boardsesh-50degree-20260930-numpy.zip -d .
```

Route files are placed in `ImageData/50Degree/ExportBoardsesh/`. Each route has a `float32` `.npy` array of shape `(34, 35, 10)` and a matching `.json` metadata file. Both orientation sin/cos pairs are included. The archive also contains documentation, the importer, validation reports, source attribution, and per-file SHA-256 checksums.

```python
from pathlib import Path
import json
import numpy as np

path = next(Path("ImageData/50Degree/ExportBoardsesh").glob("*.npy"))
matrix = np.load(path, allow_pickle=False)
metadata = json.loads(path.with_suffix(".json").read_text())
print(matrix.shape, matrix.dtype)  # (34, 35, 10) float32
```

**Source:** Route roles and metadata were converted from the [Boardsesh Kilter Original database snapshot built on September 30, 2026](https://boardsesh-board-snapshots.t3.tigrisfiles.io/board-snapshots/v1-gzip/kilter/1/2026-09-30T18-37-44-688Z.db). Boardsesh supplied a database, not these NumPy arrays. Hold presence, normalized size, and manually annotated orientations come from the existing Kilterboardie reference. Each route's metadata preserves its setter and source UUID. Third-party data rights remain with their respective owners; the repository's software license does not grant additional rights to that data.

All 37,051 matrices and the compressed archive were validated before publication. Do not combine this dataset with the older `ExportClean` dataset without deduplication. See [DATASET.md](DATASET.md) for filtering and encoding details.

## Local Setup

Use Python 3.11 or newer and install the model dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The exported tensors can be used directly for training. Rebuilding exports with `dataset.py` additionally requires `opencv-python`, `tqdm`, and `pynput`; OCR requires `pytesseract` and a local Tesseract installation. Raw source screenshots are not included in the public dataset.


## Dataset Overview

Each route is encoded as a tensor with shape `rows x cols x channels`.

- `rows`: 34
- `cols`: 35
- `channels`: 10

### Channel Labels

Channel order (axis 2) is:

- `start` (binary)
- `finish` (binary)
- `hand` (binary)
- `foot` (binary)
- `hold_presence` (binary, 1 if a hold exists at that grid cell)
- `hold_size` (float in [0, 1], normalized hold area)
- `orient_sin1` (float, `sin(theta1)` for primary orientation)
- `orient_cos1` (float, `cos(theta1)` for primary orientation)
- `orient_sin2` (float, `sin(theta2)` for secondary orientation)
- `orient_cos2` (float, `cos(theta2)` for secondary orientation)

Orientation angles are stored per hold in `ImageData/References/holds.json` and are also encoded per grid cell as `sin/cos` channels.
The exported matrices are `34 x 35 x 10` with the channel list above.

### File Format

For each route in `ImageData/50Degree/ExportBoardsesh` or the older `ImageData/50Degree/ExportClean`:

- `<route>.npy`: `H x W x 10` float32 matrix
- `<route>.json`: metadata with `rows`, `cols`, `channels`, `grade_v`, and ring counts

The older screenshot-derived clean dataset contains 1,000 canonical routes. The original 30,000 screenshots
are 30 captures of those same 1,000 route slots. Ring roles are classified at the
476 calibrated hold centers using HSV coverage in a 25-35 pixel annulus; each hold
can therefore receive at most one role. `dataset_audit.json` records count-rule and
overlap validation for every export.

Rebuild it without repeating OCR:

```bash
python dataset.py export \
  --image-dir ImageData/50Degree \
  --hold-map ImageData/References/holds.json \
  --output-dir ImageData/50Degree/ExportClean \
  --ring-method fixed-centers \
  --max-images 1000 \
  --metadata-dir ImageData/50Degree/Export
```

### Hold Grid + Labeled Rings (Overlay)

![Hold grid overlay with labeled rings](ImageData/References/debug_overlay.png)

Legend:
- Green ring: `start`
- Magenta ring: `finish`
- Cyan ring: `hand`
- Orange ring: `foot`
- Gray dots: detected hold centers
- Light grid: row/column centers used for the matrix layout

### Hold Grid Maps

#### Hold Presence (Binary)

![Hold presence grid](ImageData/References/hold_grid_presence.png)

Legend:
- Light orange: hold present (1)
- Light gray: no hold (0)

#### Hold Size (Normalized)

![Hold size grid](ImageData/References/hold_grid_size.png)

Legend:
- Light gray: no hold
- Light orange: smaller holds
- Darker orange: larger holds

### Hold Orientations

Each hold can have up to two orientation angles (in radians) stored in `ImageData/References/holds.json` under `holds[*].orientations`. Angles are measured using `atan2(dy, dx)` in image coordinates, so values are in `[-pi, pi]` relative to the +x axis. These are encoded into the matrix as `orient_sin1/cos1` and `orient_sin2/cos2`.

#### Orientation Input (Annotated Board)

![Annotated orientation input](ImageData/References/empty_board_orientations.png)

#### Orientation Overall Bias Check

![Hold orientation bias check overlay](ImageData/References/hold_orientations_overlay_empty.png)

Legend:
- Red arrows: detected hold orientation vectors (up to two per hold)

### Notes

- The grid is derived from the detected hold centers stored in `ImageData/References/holds.json`.
- `hold_size` is normalized by the maximum hold area in the board so values are in `[0, 1]`.
- The downloadable Boardsesh dataset contains 50-degree climbs, grades `V3` through `V13`; historical screenshot exports are separate.

## Channel Split Used By Models

The grid-based CVAE and diffusion models split the 10-channel tensor into:

- `route`: 4 dynamic channels (`start`, `finish`, `hand`, `foot`)
- `static`: 6 conditioning channels (`hold_presence`, `hold_size`, `orient_sin1`, `orient_cos1`, `orient_sin2`, `orient_cos2`)

## Model (Conditional VAE)

The model is defined in `cvae_model.py` as `KilterCVAE`.

**Inputs**
- `route`: `(B, 4, H, W)` for the 4 dynamic channels (`start`, `finish`, `hand`, `foot`)
- `static`: `(B, 6, H, W)` for hold presence, hold size, and both orientation sin/cos pairs
- `grade`: `(B,)` int64 in `[0, num_grades-1]`

**Output**
- `logits`: `(B, 4, H, W)` for the 4 dynamic channels

**Loss**
- Reconstruction: BCEWithLogitsLoss over hold positions
- KL divergence (beta-scaled)
- Optional count loss to encourage realistic counts for `start` and `finish`
- Optional focal loss
- Optional path loss to encourage reachable sequences from start to finish
- Optional upward loss to discourage implausible hand placements below starts or above finishes

## Training

Training entry point: `cvae_train.py`.

Example:

```bash
python cvae_train.py --data-dir ImageData/50Degree/Export --epochs 30 --batch-size 64
```

Key options:
- `--beta`: KL weight
- `--count-weight`: start/finish count regularizer
- `--focal-gamma`: focal loss gamma
- `--path-weight`, `--path-reach`, `--path-steps`: path connectivity regularization
- `--upward-weight`: vertical hand-placement regularization

Artifacts are saved under `runs/cvae/<timestamp>/`.

## Generation

Generate routes from a trained checkpoint with `cvae_generate.py`:

```bash
python cvae_generate.py \
  --checkpoint runs/cvae/<run>/best.pt \
  --data-dir ImageData/50Degree/Export \
  --grade 6 \
  --n 4 \
  --out generated_route.npy
```

The output is a full `H x W x (4 + static_channels)` matrix (route + static channels) plus a JSON sidecar.

## Diffusion Model (Conditional DDPM)

Alternative generator implemented in:
- `diffusion_model.py`
- `diffusion_train.py`
- `diffusion_generate.py`

The diffusion model is implemented as a grade-conditioned U-Net denoiser with a Gaussian DDPM scheduler.

This model denoises the 4 dynamic route channels (`start`, `finish`, `hand`, `foot`) conditioned on:
- static channels (`hold_presence`, `hold_size`, `orient_sin1`, `orient_cos1`, `orient_sin2`, `orient_cos2`)
- grade embedding
- timestep embedding

Training losses:
- masked denoising MSE over valid hold cells
- masked reconstruction BCE on reconstructed route probabilities
- count loss for realistic start/finish counts
- path loss for start-to-finish reachability
- upward loss for vertical plausibility
- hand-density and foot-density losses to control occupancy

Train:

```bash
python diffusion_train.py \
  --data-dir ImageData/50Degree/Export \
  --epochs 40 \
  --batch-size 64
```

Useful options:

- `--timesteps`, `--beta-start`, `--beta-end`: diffusion schedule
- `--base-channels`, `--grade-emb-dim`, `--time-emb-dim`: model capacity
- `--eps-weight`, `--recon-weight`: denoising vs reconstruction balance
- `--count-weight`, `--path-weight`, `--upward-weight`: structure regularizers
- `--hand-density-weight`, `--foot-density-weight`: occupancy regularizers

Run visualization:

```bash
python diffusion_visualize.py \
  --run-dir runs/diffusion/<run> \
  --data-dir ImageData/50Degree/Export
```

Generate:

```bash
python diffusion_generate.py \
  --checkpoint runs/diffusion/<run>/best.pt \
  --data-dir ImageData/50Degree/Export \
  --grade 6 \
  --n 4 \
  --out generated_route.npy
```

The output format matches the CVAE generator: full `H x W x (4 + static_channels)` tensor(s) and a JSON sidecar. Sampling is then decoded with empirical per-grade count priors for start, finish, hand, and foot placements.

## Hierarchical Graph Transformer

The graph model generates a route as a bottom-to-top sequence of hold groups rather than predicting the complete matrix at once.

- Each of the 476 real board holds is a graph node.
- Node features use calibrated hold-center coordinates, hold size, and orientation.
- Graph edges use image-coordinate distances normalized by median hold spacing, not row/column grid distance.
- The GNN creates one contextual embedding per hold.
- A causal Transformer tracks the generated route and grade.
- Separate heads predict the next action, point to a real hold, and assign its route role.
- Group-boundary tokens keep starts, vertical movement bands, and finishes distinct.
- Holds labeled in multiple exported route channels are converted to one LED role using `start > finish > hand > foot` precedence.
- Intermediate groups are ordered by handhold height; footholds are attached to the nearest hand-led group and do not drive path order.

The current image coordinates are more accurate than matrix-grid distances but are not yet measurements in centimeters. The graph data loader isolates this coordinate source so measured board coordinates can replace it later without changing the model.

Train:

```bash
python graph_transformer_train.py \
  --data-dir ImageData/50Degree/ExportClean \
  --holds-path ImageData/References/holds.json \
  --epochs 30 \
  --batch-size 64
```

Generate:

```bash
python graph_transformer_generate.py \
  --checkpoint runs/graph_transformer/<run>/best.pt \
  --data-dir ImageData/50Degree/ExportClean \
  --grade 7 \
  --n 4 \
  --out generated_graph_route.npy
```

Start and finish counts are constrained to one or two during generation. Pair distance is calculated from calibrated hold centers in approximate hold-spacing units. Use `--pair-max-distance` to adjust it.

For a quick pipeline test before a full run:

```bash
python graph_transformer_train.py \
  --max-samples 128 \
  --epochs 1 \
  --hidden-dim 32 \
  --graph-layers 1 \
  --transformer-layers 1 \
  --feedforward-dim 64
```

Inspect the pseudo-sequence preprocessing before training:

```bash
python graph_sequence_visualize.py \
  --grades 3 5 7 9 11 13 \
  --per-grade 4 \
  --save-individuals
```

This produces a contact sheet and manifest under `runs/graph_sequence_preview/`. Numbered groups show the bottom-to-top sequence given to the Transformer; arrows connect consecutive groups. The reported gap uses calibrated hold-center coordinates and is normalized by median nearest-hold spacing.

## Project Layout

- `ImageData/50Degree/ExportBoardsesh/`: downloadable Boardsesh dataset (37,051 routes; extract the release ZIP)
- `ImageData/50Degree/ExportClean/`: canonical cleaned dataset (`.npy` + `.json` per route)
- `ImageData/50Degree/Export/`: legacy 30-pass Hough export
- `ImageData/References/`: hold grid, overlays, orientation assets, `holds.json`
- `dataset.py`: utilities for building hold maps, overlays, and exporting matrices
- `cvae_data.py`: dataset loader for training
- `cvae_model.py`: CVAE model + loss
- `cvae_train.py`: training loop
- `cvae_generate.py`: sampling/generation
- `cvae_predict.py`: lightweight CVAE inference bundle loader
- `diffusion_model.py`: diffusion denoiser + scheduler + losses
- `diffusion_train.py`: diffusion training loop
- `diffusion_generate.py`: diffusion sampling/generation
- `diffusion_visualize.py`: training-curve and sample-grid rendering
- `graph_transformer_data.py`: physical hold graph and pseudo-sequence conversion
- `graph_transformer_model.py`: GNN encoder, causal Transformer, and pointer/role heads
- `graph_transformer_train.py`: graph Transformer training loop
- `graph_transformer_generate.py`: constrained autoregressive route generation
- `graph_sequence_visualize.py`: pseudo-sequence inspection on the real board image

## Included Checkpoints and Local Inference

- `models/best.pt`: CVAE checkpoint.
- `models/diffusion_best.pt`: diffusion checkpoint.
- Graph Transformer: deployed from `runs/graph_transformer_clean/20260716_101603/best.pt`; its checkpoint is not included in the public clone.
- `inference_bundle/`: standalone CVAE inference code, checkpoint, static board tensor, hold metadata, and count priors.

Training and generation run locally using the commands above. Model outputs are experimental route proposals; the documented structural constraints are not a validation of climbing quality.

## Grade Distribution Statistics

Historical source screenshot distribution (including repeated captures).

Format: `V grade/French grade` (example: `V3/6a`).

Source screenshot counts: **45° = 32813**, **50° = 30000**. The 50° counts include repeated captures, not independent routes; see [DATASET.md](DATASET.md).

| Grade | 45° Count | 45° Percent | 50° Count | 50° Percent |
|---|---:|---:|---:|---:|
| V3/6a | 4655 | 14.19% | 3210 | 10.7% |
| V4/6b | 4654 | 14.18% | 3570 | 11.9% |
| V5/6c | 5314 | 16.19% | 3870 | 12.9% |
| V6/7a | 4883 | 14.88% | 3840 | 12.8% |
| V7/7a+ | 3676 | 11.2% | 2970 | 9.9% |
| V8/7b | 4396 | 13.4% | 4890 | 16.3% |
| V9/7c | 2403 | 7.32% | 3000 | 10.0% |
| V10/7c+ | 1547 | 4.71% | 2160 | 7.2% |
| V11/8a | 627 | 1.91% | 1470 | 4.9% |
| V12/8a+ | 99 | 0.3% | 690 | 2.3% |
| V13/8b | 0 | 0% | 90 | 0.3% |
| Unknown | 559 | 1.7% | 240 | 0.8% |

### Larger Boardsesh Training Dataset

The [downloadable Boardsesh export](#download-the-numpy-dataset) at
`ImageData/50Degree/ExportBoardsesh/` contains 37,051 unique
50-degree V3-V13 routes in the existing ten-channel format. See [DATASET.md](DATASET.md)
for provenance, filtering, validation, and the reproducible importer command.
For graph training, use `--data-dir ImageData/50Degree/ExportBoardsesh`
and `--max-sequence-length 64` (the longest route needs 49 events).

### Shared three-model benchmark (Boardsesh)

`comparison_data.py` creates an 80/10/10 grade-stratified split, grouping identical
hold occupancy (including role variants) to prevent layout leakage. All three
trainers accept `--split-manifest`; the test partition is never used for training
or checkpoint selection.

`train_comparison.py` runs 30 epochs each of CVAE, diffusion, and graph Transformer
on MPS, sequentially, with batch size 32. The graph sequence limit is 64. This
experiment uses `runs/model_comparison_20260930/split.json`; its runner refuses to
overwrite an existing status file. Keep the computer awake while training.
Check `status.json`, each model's log, and timestamped `metrics.jsonl` for progress.
Best and latest weights are saved separately. The comparison does not replace
website model bundles.

After all three jobs succeed, `evaluate_comparison.py` automatically generates
220 routes per model, measures warmed single-route latency (including decoding),
and compares role/count statistics with the held-out data. Count priors are
computed from the training partition only. Outputs include `comparison.json`,
`REPORT.md`, `routes.png`, `training.png`, and generated route archives in the
experiment folder. Reachability uses calibrated image coordinates rather than
matrix-index distances. These structural scores do not establish climbing grade
or actual climbability; imposed decoder constraints are not learned-rule scores.
Training losses have different meanings and must not be ranked across models.
