#!/usr/bin/env bash
# Usage:
#   bash eval.sh [dynamic-points|scene-flow] [extra arguments]
#
# The data package is expected at bench/GFM4DBench/data/4dbenchmark_data by
# default. Override it with GFM4D_DATA_ROOT or --data-root.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GFM_ROOT="$(cd "${BENCH_DIR}/../.." && pwd)"
export PYTHONPATH="${GFM_ROOT}:${BENCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TASK=dynamic-points
case "${1:-}" in
  dynamic-points|scene-flow) TASK="$1"; shift ;;
esac

args=(
  --task "$TASK"
  --registry "${BENCH_DIR}/configs/subsets.yaml"
  --config "${BENCH_DIR}/configs/gfm4d.yaml"
  --output-dir "${GFM4D_OUTPUT_DIR:-${BENCH_DIR}/results/${TASK}}"
)

user_has() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$flag" || "$arg" == "$flag="* ]] && return 0
  done
  return 1
}

# With no explicit selection, run all benchmark subsets.
if ! user_has --subset "$@" && ! user_has --json "$@"; then
  for subset in adt pstudio po ds synthverse; do
    args+=(--subset "$subset")
  done
fi

offline=0
user_has --predictions-dir "$@" && offline=1
user_data_root=0
user_has --data-root "$@" && user_data_root=1

if [[ -n "${GFM4D_DATA_ROOT:-}" ]]; then
  DATA_ROOT="$GFM4D_DATA_ROOT"
elif [[ "$user_data_root" == 1 ]]; then
  DATA_ROOT=""
else
  DATA_ROOT="${BENCH_DIR}/data/4dbenchmark_data"
  if [[ ! -d "$DATA_ROOT" ]]; then
    printf 'error: benchmark data not found at %s\n' "$DATA_ROOT" >&2
    printf 'Download the GFM4DBench data package, e.g.:\n' >&2
    printf '  hf download KosmoResearch/GFM4DBenchMark --repo-type dataset \\\n' >&2
    printf '    --local-dir %s\n' "$DATA_ROOT" >&2
    printf 'or set GFM4D_DATA_ROOT to the package directory.\n' >&2
    exit 2
  fi
fi
[[ -n "$DATA_ROOT" ]] && args+=(--data-root "$DATA_ROOT")

if [[ "$offline" == 0 ]] && ! user_has --load-from "$@"; then
  CHECKPOINT="${GFM4D_CHECKPOINT:-${GFM_ROOT}/weights/gfm_4d.pth}"
  if [[ ! -f "$CHECKPOINT" ]]; then
    printf 'error: GFM4D checkpoint not found at %s\n' "$CHECKPOINT" >&2
    printf 'Download the checkpoint and set GFM4D_CHECKPOINT, or pass --load-from.\n' >&2
    exit 2
  fi
  args+=(--load-from "$CHECKPOINT")
fi

printf 'data root: %s\nregistry: %s\noutput: %s\n' \
  "${DATA_ROOT:-<from --data-root>}" \
  "${BENCH_DIR}/configs/subsets.yaml" \
  "${GFM4D_OUTPUT_DIR:-${BENCH_DIR}/results/${TASK}}" >&2

exec python -m eval4d.cli.evaluate "${args[@]}" "$@"
