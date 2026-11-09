# GFM4DBench

Dynamic Points and Scene Flow evaluation for 3D point tracking on the
TapVid3D / WorldTrack and SynthVerse benchmarks.

The Dynamic Points evaluation follows [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt),
an open implementation of the D4RT evaluation: the TapVid3D mini subsets, the
frame-0 world reference frame (`coord_convention: opend4rt`), the APD / EPE
metrics and the dynamic-query split come from it, and SynthVerse is an
additional subset. The Scene Flow task, its prediction format and its metrics
are implemented in this benchmark. We thank the OpenD4RT authors for releasing
their implementation and evaluation tools. See [Citation](#citation).

Compared with OpenD4RT, the main change is the frame-0 query set: **points that
are not visible in the first RGB frame — occluded or projecting outside the
image — are not tracked or scored.** A frame-0 query is kept only when its
visibility flag is set, its depth is finite, and its projected pixel lies
inside the real RGB resolution of the sequence. The model has no image
evidence for such points at the query frame and their trajectories cannot be
followed reliably, so including them would make the metrics less reliable.
For out-of-image projections this is stricter than OpenD4RT, which clamps them
to the image border.

## Data

The benchmark data is hosted on Hugging Face:

```text
https://huggingface.co/datasets/KosmoResearch/GFM4DBenchMark
```

The repository stores the extracted package directly (`MANIFEST.json`,
`README.md`, `SHA256SUMS`, `Tapvid3d_mini/` and `SynthVerse/`), about 5.5 GB in
total. It contains the RGB frames and the per-frame arrays referenced by the
JSON (3D trajectories and visibility); depth maps and unused annotation arrays
are not part of the benchmark.

### Option 1: download into the repository (recommended)

Run these commands from the Kosmo-GFM repository root. Downloading into
`bench/GFM4DBench/data/4dbenchmark_data` produces exactly the layout that
`script/eval.sh` looks for by default, so no other configuration is needed:

```bash
pip install "bench/GFM4DBench[download]"   # installs huggingface_hub

mkdir -p bench/GFM4DBench/data/4dbenchmark_data
hf download KosmoResearch/GFM4DBenchMark \
  --repo-type dataset \
  --local-dir bench/GFM4DBench/data/4dbenchmark_data
(cd bench/GFM4DBench/data/4dbenchmark_data && sha256sum -c SHA256SUMS)
rm -rf bench/GFM4DBench/data/4dbenchmark_data/.cache
```

The `sha256sum` line verifies every file against `SHA256SUMS` and takes a few
minutes; omit it for a quick trial. `bench/GFM4DBench/data/` is git-ignored, so
the download does not end up in a commit. An interrupted download resumes when
the same command is re-run. The repository contains many small files, so
`--max-workers 16` speeds up the download, and running `hf auth login` first
avoids anonymous rate limits. With older `huggingface_hub` releases (`<1.0`),
use `huggingface-cli download` instead; the arguments are identical.

### Option 2: keep the data on another disk

If the repository is on a small disk, download the package to a larger volume,
then expose it at the default location with a symlink:

```bash
DATA_DIR=/data/gfm4d   # an existing directory on the larger disk
mkdir -p "$DATA_DIR/4dbenchmark_data"
hf download KosmoResearch/GFM4DBenchMark \
  --repo-type dataset --local-dir "$DATA_DIR/4dbenchmark_data"
(cd "$DATA_DIR/4dbenchmark_data" && sha256sum -c SHA256SUMS)

mkdir -p bench/GFM4DBench/data
ln -s "$DATA_DIR/4dbenchmark_data" bench/GFM4DBench/data/4dbenchmark_data
```

Alternatively, skip the symlink and set `GFM4D_DATA_ROOT` in each shell instead:

```bash
export GFM4D_DATA_ROOT="$DATA_DIR/4dbenchmark_data"
```

`GFM4D_DATA_ROOT` must point at the directory that directly contains
`Tapvid3d_mini/` and `SynthVerse/`.

### Expected layout

With either option, `script/eval.sh` expects this layout (at the default
location or wherever `GFM4D_DATA_ROOT` points):

```text
4dbenchmark_data/
├── Tapvid3d_mini/
│   └── Tapvid3d_mini_extracted/
│       ├── adt_mini.json
│       ├── pstudio_mini.json
│       ├── po_mini.json
│       ├── ds_mini.json
│       ├── adt_mini/
│       ├── pstudio_mini/
│       ├── po_mini/
│       └── ds_mini/
└── SynthVerse/
    ├── synthverse_mini.json
    └── converted/
        ├── converted_subset50_64f_dynamic/   # trajectory arrays
        │   └── <sequence>/{trajs_3d,visibs}
        └── <sequence>/rgbs/
```

### Checkpoint

Place the GFM4D checkpoint at `weights/gfm_4d.pth` in the Kosmo-GFM
repository root. If it lives elsewhere, point `GFM4D_CHECKPOINT` at it:

```bash
export GFM4D_CHECKPOINT=/path/to/gfm_4d.pth
```

`script/eval.sh` reads `GFM4D_CHECKPOINT` and falls back to
`weights/gfm_4d.pth` at the repository root.

## Installation

Requirements: Python 3.10+, a CUDA GPU, and PyTorch / torchvision built for
your CUDA version. From the Kosmo-GFM repository root:

```bash
pip install -e .
pip install -e "bench/GFM4DBench[gfm]"
```

The first command installs the `gfm` package used by the GFM4D adapter. The
`gfm` extra adds PyTorch and torchvision. To download the released files with
the Hugging Face CLI, install `huggingface_hub` (or the `download` extra):

```bash
pip install "bench/GFM4DBench[download]"
```

## Evaluation

Quick check on one sequence and one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash bench/GFM4DBench/script/eval.sh dynamic-points \
  --subset pstudio --limit-seqs 1
```

Run the full benchmark (ADT, PStudio, PO, DS and SynthVerse):

```bash
bash bench/GFM4DBench/script/eval.sh dynamic-points
bash bench/GFM4DBench/script/eval.sh scene-flow
```

SynthVerse is reported separately and is not part of the TapVid3D average.

Results are written to `bench/GFM4DBench/results/<task>/`. Set
`GFM4D_OUTPUT_DIR` to override the output directory. The top-level
`summary.json` reports

- Dynamic Points: `avg_pts_global` (APD), `tau_global` (tau@0.1m),
  `epe_global` (EPE).
- Scene Flow: `avg_sf_global` (tau@0.1m), `epe_sf_global` (EPE).

The default clip length is 64 frames. Useful options include `--subset`,
`--limit-seqs`, `--seq-name`, `--num-frames`, `--output-dir`, and `--resume`.
List every option with:

```bash
gfm-4d-evaluate --help
```

## Offline scoring

Predictions are saved as `<subset>/predictions/<sequence>.npz`. To re-score a
previous run without a GPU:

```bash
gfm-4d-score --task dynamic-points \
  --subset adt --subset pstudio --subset po --subset ds --subset synthverse \
  --predictions-dir bench/GFM4DBench/results/dynamic-points \
  --registry bench/GFM4DBench/configs/subsets.yaml \
  --output-dir bench/GFM4DBench/results/dynamic-points_rescored
```

Add `--data-root "$GFM4D_DATA_ROOT"` when the data package is not at the
default location.

### Prediction format

Each `.npz` file describes one sequence in the frame-0 reference frame:

| Field | Shape | Meaning |
|---|---|---|
| `sequence_name` | scalar string | Sequence name, matching the JSON key. |
| `tracks_ref0` | `[T, Q, 3]` | Absolute 3D positions (Dynamic Points). |
| `flow_ref0` | `[T, Q, 3]` | Displacement from frame 0 (Scene Flow); `flow_ref0[0] == 0`. |
| `query_indices` | `[Q]` | Indices of the scored frame-0 queries in the JSON track arrays. |

`T` is the clip length (64 by default) and `Q` is the number of frame-0
queries. The arrays are written with `numpy.savez_compressed`.

## Citation

The Dynamic Points evaluation builds on the OpenD4RT protocol. Please cite the
OpenD4RT repository and the D4RT paper it implements:

```bibtex
@misc{opend4rt,
  title        = {{OpenD4RT}: An open PyTorch implementation of {D4RT} for
                  dynamic 4D reconstruction and tracking},
  howpublished = {\url{https://github.com/Lijiaxin0111/Open-d4rt}},
  note         = {Unofficial implementation of the D4RT evaluation protocol}
}

@article{zhang2025d4rt,
  title   = {Efficiently Reconstructing Dynamic Scenes One {D4RT} at a Time},
  author  = {Zhang, Chuhan and Le Moing, Guillaume and Koppula, Skanda and
             Rocco, Ignacio and Momeni, Liliane and Xie, Junyu and Sun, Shuyang and
             Sukthankar, Rahul and Barral, Jo{\"e}lle K. and Hadsell, Raia and
             Ghahramani, Zoubin and Zisserman, Andrew and Zhang, Junlin and
             Sajjadi, Mehdi S. M.},
  journal = {arXiv preprint arXiv:2512.08924},
  year    = {2025}
}
```
