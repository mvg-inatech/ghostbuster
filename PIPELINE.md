# Pipeline overview

What runs, in what order, to go from a recorded session to a trained confidence
model. `README.md` has the minimal reproduction; this file says what each script
does and which stage it belongs to.

```
  1. SLAM          bag            -> per-scan clouds, trajectory, channels A, B, C
  2. Labelling     cloud + RTC360 -> C2C distance per point + evaluation domain
  3. Features      labelled cloud -> channels D and E added
  4. Model         featured clouds-> confidence model and metrics
```

A point keeps its coordinates from stage 1 onwards. Later stages only add
channels and labels.

---

## 1. SLAM

`VoxelSLAM/` — our instrumented fork of VoxelSLAM. The confidence channels are
written here, not added afterwards.

| what | where | notes |
|---|---|---|
| frontend, BA, loop closure | `src/voxelslam.{hpp,cpp}` | writes `ekf_res`, `balm_res`, `plane_quality`, `pose_unc_rot`, `ang_rate` |
| within-scan family B | `src/range_image.hpp` | computed on the raw organised scan, **before** downsampling |
| per-point struct | `src/tools.hpp` | the channel list and what each one means |
| sensor handlers | `src/feature_point.hpp` | Ouster and Hesai; rebuilds the range image for the Hesai |
| merge to a global map | `scripts/global_pcd_converter.py` | applies the final trajectory, computes `cell_id`, `obs_count`, `view_diversity`, `res_spread` |

`launch/` and `config/` hold the Ouster and Hesai setups used in the paper.

## 2. Labelling

Turns a SLAM cloud plus a Leica RTC360 project into per-point distances and an
evaluation domain.

| script | does |
|---|---|
| `merge_gt_e57.py` | merge a per-station RTC360 export into one project-frame LAS, keeping the stations separate |
| `prepare_labels_e57.py` | the main labelling run: coarse alignment, ICP per station group, scanner visibility, C2C |
| `stretch_labels.py` | re-label in 15 m trajectory pieces so the labels do not carry drift. The cloud is not moved, only the frame each label is computed in |

**The `gtlabel/` package** holds the logic those three call:

| module | does |
|---|---|
| `e57.py` | read RTC360 per-station exports |
| `coarse.py` | automatic coarse alignment, so ICP starts in the right basin |
| `register.py` | SLAM → GT registration |
| `visibility.py` | the scanner-visibility model: observed / free-space violation / occluded |
| `domain.py` | assembles the evaluation-domain mask from all stations |
| `c2c.py` | tiled cloud-to-cloud distance for clouds that do not fit in RAM |
| `geom.py` | multi-scale kNN and roughness (family E) |
| `knn.py` | kNN distance without holding one big KD-tree |
| `aggregate.py`, `aggregate_fast.py` | family D: deviation of each channel from its neighbourhood |
| `lasio.py` | chunked LAS I/O |
| `pcd.py` | streaming reader for the PCDs VoxelSLAM writes |
| `slices.py`, `qa.py` | renders for checking a registration by eye |

## 3. Features

Run in this order; each adds channels to the labelled cloud.

| script | adds |
|---|---|
| `add_geom_features.py` | family E: kNN distance at k = 6, 30, 100 and roughness at r = 5, 15, 40 cm |
| `add_ratio_features.py` | the scale-free ratios of family E |
| `add_aggregate_features.py` | family D: each A/B/C channel against its k = 30 neighbourhood |

## 4. Model

| script | does |
|---|---|
| `compare_filters.py` | the main experiment: loads scenes, holds one out, fits the model, scores the baselines, writes `filter_comparison.json` and `ranking_summary.txt`. The family ablations and the online variant are arms of the same run |
| `outlier_analysis.py` | cloud loading and per-channel AUC; imported by the above and runs standalone |
| `predict_cloud.py` | apply a saved model to a whole cloud, adding `pred_*` and `keep_*` channels |

## Configuration

`scripts/configs/` holds one YAML per scene: where the SLAM cloud and the RTC360
stations live, the station groups, and the registration and visibility settings.
The paths in them are absolute and point at our storage, so they need adjusting
before a run.
