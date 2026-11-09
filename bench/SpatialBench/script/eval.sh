#!/usr/bin/env bash
set -euo pipefail

# Usage: bash script/eval.sh [gfm3d|gfm3d_camera_cond|gfm4d] [benchmark arguments...]
# Environment overrides: PYTHON, CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS,
# BENCHMARK_ROOT, OUTPUT_DIR, SCENE_INDEX, TAGS.
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: eval.sh [gfm3d|gfm3d_camera_cond|gfm4d] [benchmark arguments...]

Default model: gfm3d. Additional arguments override the selected YAML and defaults.
Results: results/<model>/<effective-tags>/ (OUTPUT_DIR sets the base directory).
Examples:
  bash script/eval.sh gfm3d --max-scenes 1
  bash script/eval.sh gfm3d --tags medium
  CUDA_VISIBLE_DEVICES=1 bash script/eval.sh gfm3d_camera_cond
  BENCHMARK_ROOT=/path/to/SpatialBenchmark bash script/eval.sh gfm4d

Environment: PYTHON (python), CUDA_VISIBLE_DEVICES (0), OMP_NUM_THREADS (1),
BENCHMARK_ROOT, OUTPUT_DIR, SCENE_INDEX, TAGS (sparse).
Pass a model followed by --help to see all benchmark arguments.
USAGE
    exit 0
fi

model=gfm3d
if [[ $# -gt 0 && "$1" != -* ]]; then
    model="$1"
    shift
fi
case "$model" in
    gfm3d|gfm3d_camera_cond|gfm4d) ;;
    *) printf 'Unsupported model: %s\nExpected gfm3d, gfm3d_camera_cond, or gfm4d.\n' "$model" >&2; exit 2 ;;
esac

BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Data sits next to the SpatialBench code directory, under bench/SpatialBenchmark.
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${BENCH_DIR}/../SpatialBenchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_DIR}/results/${model}}"
SCENE_INDEX="${SCENE_INDEX:-${BENCH_DIR}/benchmark/scene_indices/all_scenes.json}"
TAGS="${TAGS-sparse}"

exec "$PYTHON" "${BENCH_DIR}/benchmark/evaluation/run_benchmark.py" \
    --config "${BENCH_DIR}/benchmark/configs/end2end/${model}_eval.yaml" \
    --benchmark-root "$BENCHMARK_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --scene-index "$SCENE_INDEX" \
    --tags "$TAGS" \
    "$@"
