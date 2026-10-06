# Automated GT labelling

Replaces the manual CloudCompare loop for producing training and test labels:

    run SLAM -> [ manual coarse align + ICP -> C2C -> delete unseen areas ->
    ICP again -> C2C again ] -> outlier_analysis.py

The bracketed part is one command:

```bash
python prepare_labels_e57.py --config configs/<scene>.yaml
python outlier_analysis.py <output>_labeled.las --out figures/
```

The only human input left per dataset is in the config, plus a look at the QA
images afterwards.

## Output

`<slam>_labeled.las` carries every input point and channel, plus:

| field | meaning |
|---|---|
| `C2C_distance` | nearest-neighbour distance to the GT, `-1` where no GT within `c2c_max_dist` |
| `domain_code` | 0 in domain, 1 no GT within radius, 2 outside footprint, 3 outside z range, 4 exclusion volume |
| `eval_domain` | 1 where `domain_code == 0` |

Points are **flagged, never deleted**. `outlier_analysis.py` filters on
`eval_domain` automatically, and the same flag defines the region for CD_sym, so
one run serves both training and metrics and nothing is destroyed.

`<slam>_gt_crop.las` is the GT restricted to the common volume, for the
completeness term `d(Q→P)`. Fix it once from the **unfiltered** SLAM cloud and
reuse it for every filter variant — otherwise an aggressive filter improves its
own score by shrinking the reference it is measured against.

`<slam>_labels.json` records the transform, per-round ICP fitness, and domain
composition.

## Why the old coverage rule was replaced

The previous rule dropped points with `C2C > 4 cm` **and** `mean_knn_dist < 1.5 cm`
— exactly the points where the kNN channel disagreed with the label, i.e. kNN's
false negatives. Removing them raises kNN's TPR at every threshold, and raises it
more for the with-kNN model than the without-kNN model. Since the headline
experiment is the gap between those two, the rule contaminated the result it was
meant to support.

Everything in `domain.py` is computed from GT geometry and point position only.
No confidence channel is involved, so no channel is favoured.

## The three gates

1. **Footprint** (regional) — 2D occupancy raster of the GT, dilated. Rejects
   space the scanner never visited. Built from the GT automatically, or drawn
   once as a polygon.
2. **Coverage radius** (local) — no GT point within `coverage_radius`. Catches
   occlusion shadows the footprint is too coarse to see. Reuses the C2C distance
   that is computed anyway.
3. **Exclusion volumes** (declared) — see below.

## The limitation that gate 3 exists for

Gates 1 and 2 are pure geometry, and there is a case they probably cannot handle.

Hand-cleaning an indoor scene removes points concentrated in slabs at door and
window openings. Those points sit a few centimetres to a few tens of
centimetres off the wall plane — inside the coverage radius — so gate 2 keeps
them, even though many of them are close enough to a reference surface to look
clean.

Geometry alone cannot separate "15 cm into an unscanned neighbouring room" from
"15 cm off a scanned wall, i.e. a genuine SLAM error". Both are *a point 15 cm
from a GT surface*. Only a visibility model built from the RTC station positions
can decide, by asking whether the scanner could see that spot at all: cast a ray
from each station, and a point in front of the returned surface is a free-space
violation (a real outlier) while a point behind it was never observable
(exclude). That needs the per-station clouds and their registration transforms,
which the merged export does not carry.

Until then, openings are declared explicitly in the dataset config, as boxes,
prisms or a voxel mask (see `domain.py`). They do not move between runs, so the
judgement is made once and versioned with the config rather than repeated by
hand for every run.

## Validation

| check | result |
|---|---|
| C2C vs. CloudCompare's `C2C_distance` | max deviation 0.84 mm, p99 0.58 mm — the 1 mm LAS quantisation bound |
| Domain mask vs. hand-cleaned cloud | 99.01% agreement, 99.9% recall of the manual deletions |
| Scale | 343 M point / 6.9 GB GT streamed at 2.8 GB peak RSS |
| Runtime | 368 s end-to-end for 15 M SLAM points against a 91 M point GT |

## Design notes

* **The GT is never loaded whole.** A cKDTree over 343 M points needs tens of GB.
  `c2c.py` bins the GT into spatial tiles on disk once, then queries each tile
  against the SLAM points inside it; peak memory is one tile. Tile size is chosen
  from a GT density pass so the worst tile stays under `tile_budget`.
* **The GT is never downsampled for C2C.** Distances are exact, because the
  smallest label threshold in use is τ = 1 cm and even a 5 mm GT downsample would
  eat half of it.
* **The search cap does double duty.** Points with no GT neighbour inside
  `c2c_max_dist` come back as `inf`, which is exactly the "never scanned here"
  signal gate 2 needs.
* **ICP cannot make things worse.** After the multiscale run the result is
  compared against the initial transform on the finest level, and the initial one
  is kept if it wins. The coarse level's wide correspondence distance can
  otherwise pull an already-good alignment off a sharp optimum.
* **Round 2 replaces "delete, then ICP again."** The first round aligns on
  everything, the second realigns using only in-domain points. It stops early
  once the transform stops moving.
* **The GT for ICP is streamed at half the finest ICP voxel.** Re-voxelising an
  already voxelised cloud at the same resolution silently discards ~35% of it,
  because the second grid has a different phase.
