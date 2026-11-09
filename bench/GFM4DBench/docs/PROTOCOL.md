# Protocol

This document specifies the evaluation protocol implemented by `eval4d`. It is
a description of the *numbers*: which points are scored, how the prediction and
the ground truth are aligned, and what each metric means.

The alignment rules below define the reported numbers.

---

## 1. Frame-0 query selection

Every sequence contributes a set of **frame-0 queries**. A track becomes a
query when all of the following hold:

1. It is **visible at frame 0** (`visibility[0] == True`).
2. Its frame-0 depth is finite and `|z| > 1e-8`.
3. Its UV, **projected from `trajs_3d` with the camera intrinsics**, is finite.
4. The projected UV is inside the image: `0 <= u < W` and `0 <= v < H`, where
   `W x H` is the **real resolution of the frame-0 RGB image** (read from the
   image header, never assumed).

The selection never reads the on-disk `trajs_2d` files or the JSON `valids`
array (neither is part of the benchmark package).

Implementation: `eval4d/data/queries.py`.

### 1.1 Optional dynamic-query filter

A subset may declare a dynamic-query threshold (metres). When it is set, only
the queries whose **maximum** displacement from frame 0 exceeds the threshold
are kept:

```text
keep(q)  <=>  max_t || gt[t, q] - gt[0, q] || > threshold
```

This is the *query filter*. It is **not** the same definition as the dynamic
subset used by the metrics -- see §5.

## 2. Ground truth and coordinate conventions

The ground truth is `trajs_3d`, transformed into the **frame-0 reference
(world) frame**: `gt_tracks_world[T, N, 3]`.

Two layouts are supported:

| Convention | `trajs_3d` on disk | `extrinsics` on disk | Used by |
|---|---|---|---|
| `opend4rt` (default) | frame-0 reference world | world-to-camera | TapVid3D / WorldTrack |
| `pointodyssey` | global world | camera-to-world | PointOdyssey-style datasets |

For `pointodyssey` the conversion happens in memory:

```text
gt_tracks_world[t] = w2c_0 @ world[t]
extrinsics_w2c[t]  = inv(c2w[t])
```

Using the default `opend4rt` convention on a PointOdyssey-style JSON is wrong:
the camera-space projection drops most of the points, so the scored query count
falls far below the expected value.

## 3. The prediction

A prediction is expressed in the **frame-0 reference frame**, the same frame as
the ground truth.

### 3.1 Dynamic Points

```text
pred_tracks_ref0[t, q] = the absolute 3D position of query q at frame t
```

### 3.2 Scene Flow

```text
pred_flow_ref0[t, q] = the displacement of query q between frame 0 and frame t
pred_flow_ref0[0, q] = 0
```

The ground-truth flow is `gt[t, q] - gt[0, q]`.

Scene Flow additionally needs the **absolute** positions, because the global
scale is estimated on positions, not on flow. When the positions are missing,
the flow is anchored on the ground-truth frame-0 positions instead.

## 4. Scale alignment

A single **global median-norm scale** is fitted on **all finite `(t, q)` 3D
positions**:

```text
scale = median(|| gt ||) / median(|| pred ||)
```

and applied to the prediction before any error is computed. The scale is
estimated on the whole sequence, not per trajectory.

Three diagnostic alignments are also computed, but they are **not** the
reported result: `per_traj` (per-trajectory median scale), `sim3` (RANSAC
Sim3) and `sim3_closed` (closed-form Sim3).

## 5. Two definitions of "dynamic"

They are deliberately different and are **not** unified:

| | Definition | Where |
|---|---|---|
| Query filter | **maximum** displacement from frame 0 | §1.1 |
| Metric subset | **sum** of the consecutive displacements `sum_t || gt[t,q] - gt[t-1,q] ||` | §6.2 |

The default threshold is 0.01 m for both.

## 6. Metrics

### 6.1 Dynamic Points

Let `d[t, q] = || scale * pred[t, q] - gt[t, q] ||`.

| Key | Meaning |
|---|---|
| `avg_pts_global` | **APD**: the mean of the hit rates at 0.1 m, 0.3 m, 0.5 m and 1.0 m. Reported as a fraction (multiply by 100 for a percentage). |
| `tau_global` | **τ@0.1 m**: the hit rate at 0.1 m only (`fractions_global[1]`). |
| `epe_global` | The mean 3D error over all finite `(t, q)`, in metres. |

The dynamic subset (`avg_pts_global_dyn`, `tau_global_dyn`, `epe_global_dyn`)
re-fits the scale **on the dynamic queries only**.

### 6.2 Scene Flow

Let `f[t, q] = || scale * pred_flow[t, q] - (gt[t, q] - gt[0, q]) ||`.

| Key | Meaning |
|---|---|
| `avg_sf_global` | **τ@0.1 m only** -- the hit rate of the flow error at 0.1 m. The key name suggests an APD, but it is a single-threshold hit rate. |
| `epe_sf_global` | The mean flow error, in metres. |
| `avg_sf_global_dyn` / `epe_sf_global_dyn` | The same metrics on the GT-dynamic queries. They **reuse the global scale**; the scale is not re-fitted on the subset. |
| `avg_sf_global_dyn_micro` / `epe_sf_global_dyn_micro` | The same metrics, averaged over the sequences weighted by the number of flow vectors. |
| `avg_sf_pred_dyn` / `epe_sf_pred_dyn` | The same metrics on the queries that the *prediction* considers dynamic. Diagnostic only. |

**The asymmetry is intentional.** Dynamic Points re-fits the scale on its
dynamic subset, Scene Flow does not.

### 6.3 Aggregation

* Within a subset, the metrics are the **macro mean over the sequences**.
* Across subsets, the Dynamic Points tables use a **query-weighted** average
  (each subset is weighted by its number of scored queries); the Scene Flow
  tables use a plain **macro** average. The two policies differ on purpose.

## 7. Serialization

* `fractions_global` and friends use **integer** keys in memory and **string**
  keys in JSON. The aggregation recovers them with `int(thr)`.
* `NaN` values are written as bare `NaN` tokens, so the JSON is not strictly
  standard but stays compatible with the existing downstream parsers.
* `num_queries` in a subset summary counts the queries that were actually
  scored, which may be smaller than the number of tracks in the JSON.

## 8. Invariants

The evaluation must hold these properties:

1. Dynamic Points and Scene Flow of the same run share **the same query set**
   and the same number of queries.
2. The prediction is never derived from the ground truth, from a dense
   pointmap, or from `warp3d - warp3d_delta`.
3. The frame-0 position comes from the model's own `src_points` (or, failing
   that, from the `(0, 0)` pair), never from the ground truth.
4. The scale is global, estimated on positions, and applied to the prediction
   before the errors are computed.
