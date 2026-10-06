# Visibility-based labelling (E57 stations)

The second pipeline. Where [README.md](README.md) works from one merged GT cloud
and decides the evaluation domain by proximity, this one reads the RTC360 project
as separate E57 stations and decides it by **whether any scanner could have seen
the spot**.

```bash
python prepare_labels_e57.py --config configs/<scene>.yaml
```

No pre-registration. No hand-drawn exclusion volumes.

## What it fixes

The proximity gate has a failure it cannot escape: a SLAM point 15 cm off a
scanned wall and a SLAM point 15 cm behind it in an unscanned room are both *a
point 15 cm from a GT surface*. In an indoor scene that case is the points at door and
window openings, and the only way to handle it was to declare the openings by
hand.

Asking the scanner instead: cast the ray, compare against what came back.

| condition | verdict |
|---|---|
| `r < r_near − ε` | the beam flew through this spot → **free-space violation** (a real outlier) |
| `r_near − ε ≤ r ≤ r_far + ε + behind` | **observed** |
| `r > r_far + ε + behind` | something opaque was in the way → **occluded** |
| no returns nearby | **unknown** |

Fused across stations by priority: **any** station observing the point puts it in
the domain; failing that, **any** free-space violation puts it in the domain as an
outlier; otherwise nobody could see it and it is excluded. Observation outranks
free space on purpose — at grazing angles two stations disagree, and declining to
call a point an outlier when some scanner saw a surface there is the conservative
choice.

## The tripod shadow

The unscanned disc under each setup comes out for free. The RTC360 has a blind
cone below the instrument: on this dataset elevations stop at **−61.6°**, a 28.4°
cone, which at ~1.7 m instrument height leaves an unscanned disc roughly 1.8 m
across. Those directions hold no returns, so they read as *unknown* from that
station — and the fusion rule lets a neighbouring setup fill them in. Only discs
that no station covered stay excluded, which is exactly right.

Verified directly: 200 k synthetic points on a 0.9 m disc 1.7 m under Setup 001
classify **100% unknown**.

## Self-test

Run against Setup 001's own points:

| input | expected | result |
|---|---|---|
| the station's own returns | observed | **100.0%** |
| same points pulled 1 m toward the scanner | free-space violation | **99.3%** |
| same points pushed 1 m away | occluded | **98.9%** |
| disc under the tripod | unknown | **100.0%** |

## `behind_tolerance`

The one parameter that needs judgement. Strict visibility is *too* strict: a SLAM
point 5 cm behind a scanned wall is formally unobservable, so it would be dropped
— but that is ordinary range noise on a surface the scanner did measure, and it
is the kind of error the filter exists to catch. Without the tolerance, pushing
points 5 cm away already reads 77% occluded.

`behind_tolerance` widens the observed band on the far side only. Keep it well
under a wall thickness (default 0.10 m): raise it too far and the gate stops
rejecting the neighbouring-room points it was built for. There is no setting that
is right for both cases — that is a real limit of the method, and worth stating
in the write-up rather than tuning away.

## Automatic coarse alignment

`coarse.py` replaces the manual CloudCompare transform with a 4-DoF search — yaw
plus translation, not the full 6, because both clouds are already gravity-aligned
and searching roll and pitch only adds room for noise.

Both clouds are projected to a top-down binary raster of **above-ground
structure**; the ground plane is deliberately excluded, because it covers
everything, correlates with everything and flattens the score. Yaw is swept, and
for each angle the XY shift comes from an FFT cross correlation.

Two things had to be right for the score to mean anything:

* The GT is rasterised at the **centre of the SLAM grid**, not at its own world
  position. Otherwise the number of in-bounds GT cells changes with yaw and the
  per-yaw scores are not comparable — the first version scored a perfect 1.0 at a
  dozen unrelated angles for exactly this reason.
* The score is a **masked normalised cross correlation**, not an overlap count.
  A raw count rewards dropping the GT onto any densely built-up patch regardless
  of pattern. The normaliser runs over the GT's support — every cell the scanner
  reached, ground included — so a window where every SLAM cell is occupied has
  zero variance and scores 0 rather than 1.

On this dataset it recovered **yaw = 0.00°, shift (256.42, 222.57, −0.16) m** from
no input, taking the SLAM cloud from 0% to **98.4% of points within 2 m** of the
GT. The runner-up was the ±2° neighbour of the same peak, which is what a genuine
peak looks like; a flat top-5 means the scene is rotationally ambiguous and you
should fall back to a manual `initial_matrix`.

## Scale

Nothing is held whole. The SLAM cloud is 194 M points / 12.8 GB, the GT 234 M
points across 10 stations.

* Stations merge into one project-frame LAS once (71 s), cached.
* Panoramas are built one station at a time and kept resident (~1.5 GB total).
* The SLAM cloud is streamed for coarse alignment, for visibility, and again for
  the output write, via `lasio.write_las_with_fields`.
* C2C runs only on points within reach of a station — the rest of a 400 m
  footprint would otherwise be tiled and searched for nothing.

## E57 conventions

Poses live in `data3D → pose` and map local to project:

    p_project = R @ p_local + t

so `t` is the station origin in the project frame, which is what the visibility
model needs. Confirmed on this data: `R @ x + t` lands Setup 002 on Setup 001 at
6 cm median, the transposed convention at 30 cm. Setup 001 is identity — it
defines the project frame.

Two gotchas:

* `pye57` headers hold a **weak reference** into the E57 document. Let the reader
  go out of scope before copying the pose out and it raises `bad_weak_ptr`.
* The exports carry `rowIndex`/`columnIndex`, the native scan grid. The panorama
  deliberately does **not** use it — the index-to-angle mapping is not a clean
  linear function (a linear fit leaves 1.16° of residual against a 0.036° bin),
  so angles are recomputed from the coordinates instead.

## Output

Same fields as the other pipeline, plus `visibility_code` (0 unknown, 1 occluded,
2 free-space violation, 3 observed). `domain_code` gains value 5, *not visible to
any station*. Points are flagged, never deleted.
