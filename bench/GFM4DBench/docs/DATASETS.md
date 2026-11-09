# Datasets

## The data package

The released benchmark package is hosted on Hugging Face
(`https://huggingface.co/datasets/KosmoResearch/GFM4DBenchMark`) as extracted
files, about 5.5 GB in total. It downloads into a directory
(`4dbenchmark_data/`) with one nested dataset root per release:

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

`--data-root` is the directory that contains `Tapvid3d_mini/` and
`SynthVerse/`. In `script/eval.sh` it defaults to
`bench/GFM4DBench/data/4dbenchmark_data`, and it can be overridden with
`GFM4D_DATA_ROOT`.

The SynthVerse RGB frames live in `converted/<sequence>/rgbs/`, and the
trajectory arrays live in `converted/converted_subset50_64f_dynamic/`, holding
`trajs_3d/` and `visibs/` for the 64-frame clip. The benchmark covers the 49
sequences that contain dynamic queries (`animal_3716` has none).

## The sequence JSON

A subset is described by an `mf_files` JSON: a dictionary that maps a sequence
name to a list of frames, and every frame to a list of views (one view per
frame in these benchmarks). A view carries

| Key | Content |
|---|---|
| `rgb` | Path to the RGB frame. |
| `trajs_3d` | Path to the `[N, 3]` 3D trajectories. |
| `visibs` | Path to the `[N]` visibility flags. |
| `cam_in` | The `[4]` intrinsics `fx, fy, cx, cy`. |
| `extrinsics` | The `[4, 4]` camera extrinsics. |

The paths are relative to the subset's dataset root (`--data-root` plus the
registry's `data_root`). The release does not include depth maps or unused
annotation arrays (`trajs_2d`, `valids`, `query_track_indices`) because the
evaluator never reads them.

## The subsets

| Subset | Label | File | Convention |
|---|---|---|---|
| `adt` | ADT | `Tapvid3d_mini/Tapvid3d_mini_extracted/adt_mini.json` | `opend4rt` |
| `pstudio` | PStudio | `Tapvid3d_mini/Tapvid3d_mini_extracted/pstudio_mini.json` | `opend4rt` |
| `po` | PO | `Tapvid3d_mini/Tapvid3d_mini_extracted/po_mini.json` | `opend4rt` |
| `ds` | DS | `Tapvid3d_mini/Tapvid3d_mini_extracted/ds_mini.json` | `opend4rt` |
| `synthverse` | SynthVerse | `SynthVerse/synthverse_mini.json` | `pointodyssey` |

SynthVerse is reported separately from the TapVid3D average.

## Acceptance values

The number of scored queries is a strong check of the loader, the query rules
and the coordinate convention:

| Subset | `total_queries` |
|---|---|
| `adt` | 22187 |
| `pstudio` | 8720 |
| `po` | 53465 |
| `ds` | 45149 |
| `synthverse` | 15762 |

These values assume the default clip length of 64 frames. The runner warns
when the count differs from the reference value in the registry. The Dynamic
Points and the Scene Flow runs of the same subset must report the **same**
`total_queries`.

## Adding a subset

Add an entry to `configs/subsets.yaml`:

```yaml
subsets:
  my_subset:
    label: My Subset
    data_root: my_dataset          # relative to --data-root
    json: sequences/mf_with_tracking.json
    coord_convention: opend4rt
    dynamic_query_threshold: 0.01  # optional
    reference_queries: 12345       # optional
```

Then run it by name:

```bash
gfm-4d-evaluate --task dynamic-points --subset my_subset \
    --data-root /path/to/datasets
```

A subset can also be evaluated without touching the registry:

```bash
gfm-4d-evaluate --task dynamic-points \
    --json /path/to/sequences.json --subset-name my_subset \
    --data-root /path/to/datasets --coord-convention opend4rt
```
