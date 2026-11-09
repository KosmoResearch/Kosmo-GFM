# SpatialBench for GFM

Adapted from [SpatialBench](https://github.com/Ropedia/SpatialBench).
Evaluates depth, camera pose, and trajectory using GFM3D, Camera Condition GFM,
and GFM4D from this repository.

## Setup

Install PyTorch / torchvision for your CUDA version, then run all commands
from the repository root:

```bash
pip install -e .
pip install -e bench/SpatialBench
```

Download the dataset from
[SpatialBenchmark on Hugging Face](https://huggingface.co/datasets/ropedia-ai/SpatialBenchmark).
Place it at `bench/SpatialBenchmark/` (or use a symlink), or specify its location
with `--benchmark-root /path/to/SpatialBenchmark`.

Place checkpoints in `weights/`:

- `gfm_3d.pth`
- `gfm_3d_camera_cond.pth`
- `gfm_4d.pth`

## Usage

```bash
bash bench/SpatialBench/script/eval.sh gfm3d
bash bench/SpatialBench/script/eval.sh gfm3d_camera_cond
bash bench/SpatialBench/script/eval.sh gfm4d

# Custom data directory and medium scenes
bash bench/SpatialBench/script/eval.sh gfm3d \
  --benchmark-root /path/to/SpatialBenchmark --tags medium

# Quick check on one scene
bash bench/SpatialBench/script/eval.sh gfm4d --max-scenes 1
```

The default tag is `sparse`. Results are saved to
`bench/SpatialBench/results/<model>/<tags>/`. Set `--output-dir` to change
the output base directory; tags are appended automatically.
Set `CUDA_VISIBLE_DEVICES` to select a GPU. For all options:

```bash
bash bench/SpatialBench/script/eval.sh gfm3d --help
```
