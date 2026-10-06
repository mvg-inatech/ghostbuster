#!/usr/bin/env python3
"""
Automatic coarse alignment of a SLAM cloud onto the ground truth, so that ICP
has something to converge from without a hand-made CloudCompare transform.

The search is 4-DoF — yaw plus 3D translation — not the full 6. Both clouds are
already gravity-aligned (the SLAM by its IMU, the RTC by its own levelling), so
roll and pitch are known to be near zero and searching them would only add
dimensions for noise to hide in.

Method: project both clouds to a top-down binary raster of *above-ground
structure*, sweep yaw, and for each yaw find the XY shift by FFT cross
correlation. The ground plane itself is deliberately excluded from the raster —
it covers everything and correlates with everything, so leaving it in makes the
score flat. Walls, buildings and vehicles are what carry the signal.

The score reported per yaw is the fraction of occupied GT cells that land on an
occupied SLAM cell, which is comparable across yaws and interpretable: a decisive
alignment on this data scores far above the runner-up, and a flat top-5 means the
scene is rotationally ambiguous and you should fall back to a manual transform.
"""

import numpy as np

from . import lasio


class Raster:
    """Top-down grid with an explicit world origin, so shifts convert to metres."""

    def __init__(self, grid, origin, cell):
        self.grid = grid
        self.origin = np.asarray(origin, dtype=np.float64)  # world XY of cell (0,0)
        self.cell = float(cell)

    @property
    def shape(self):
        return self.grid.shape


def _accumulate(path_or_xyz, cell, origin, shape, reduce_min_z=False,
                ground=None, band=(1.0, 8.0), verbose=True, tag=""):
    """One streaming pass: either the per-cell minimum z, or the structure mask.

    With `reduce_min_z` the result is the lowest z seen in each cell, used as a
    local ground estimate — local rather than global because outdoor sites slope.
    Otherwise the result counts points whose height above that local ground falls
    inside `band`.
    """
    nx, ny = int(shape[0]), int(shape[1])
    if reduce_min_z:
        acc = np.full((nx, ny), np.inf, dtype=np.float64)
    else:
        acc = np.zeros((nx, ny), dtype=np.int64)

    if isinstance(path_or_xyz, np.ndarray):
        chunks = [path_or_xyz]
        total = len(path_or_xyz)
    else:
        chunks = lasio.iter_xyz(path_or_xyz)
        total = lasio.point_count(path_or_xyz)

    tick = lasio.Ticker(enabled=verbose)
    seen = 0
    for xyz in chunks:
        seen += len(xyz)
        ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        if not ok.any():
            continue
        ij, z = ij[ok], xyz[ok, 2]
        flat = ij[:, 0] * ny + ij[:, 1]
        if reduce_min_z:
            np.minimum.at(acc.reshape(-1), flat, z)
        else:
            g = ground.reshape(-1)[flat]
            keep = np.isfinite(g) & (z >= g + band[0]) & (z <= g + band[1])
            if keep.any():
                acc += np.bincount(flat[keep], minlength=nx * ny).reshape(nx, ny)
        tick(f"    {tag}{seen:,} / {total:,} points")
    tick.done(f"    {tag}{seen:,} points")
    return acc


def dense_extent(path, percentile=0.1, sample_target=2_000_000, verbose=True,
                 tag=""):
    """XY extent of the dense core of a cloud, ignoring a thin tail of flyers.

    A SLAM cloud routinely carries a sparse spray of points far outside the
    mapped scene — sky returns, a divergence excursion, reflections through
    glass. They are a negligible fraction of the points but they set the
    bounding box, and the raster is sized from the bounding box, so a handful of
    them can leave the real content occupying a percent of the grid.

    The percentile is taken over *points*, not voxels. A sparse spray occupies
    roughly one voxel per point, so on a voxelised sample it can outnumber the
    surfaces and survive any percentile you would want to use.
    """
    total = lasio.point_count(path)
    stride = max(1, total // max(sample_target, 1))
    acc = []
    tick = lasio.Ticker(enabled=verbose)
    seen = 0
    for xyz in lasio.iter_xyz(path):
        seen += len(xyz)
        acc.append(xyz[::stride, :2].astype(np.float32))
        tick(f"    {tag}extent scan: {seen:,} / {total:,} points")
    tick.done(f"    {tag}extent scan: {seen:,} points")
    sample = np.concatenate(acc)
    lo = np.percentile(sample, percentile, axis=0).astype(np.float64)
    hi = np.percentile(sample, 100.0 - percentile, axis=0).astype(np.float64)
    return lo, hi


def structure_raster(source, cell, origin, shape, band=(1.0, 8.0),
                     min_points=2, verbose=True, tag=""):
    """Binary raster of above-ground structure, plus the local ground heights."""
    ground = _accumulate(source, cell, origin, shape, reduce_min_z=True,
                         verbose=verbose, tag=tag + "ground ")
    counts = _accumulate(source, cell, origin, shape, ground=ground, band=band,
                         verbose=verbose, tag=tag + "structure ")
    return Raster((counts >= min_points), origin, cell), ground


def _rasterise_points(xy, origin, shape, cell):
    nx, ny = int(shape[0]), int(shape[1])
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    grid = np.zeros((nx, ny), dtype=bool)
    if ok.any():
        grid[ij[ok, 0], ij[ok, 1]] = True
    return grid


def align_yaw_xy(slam_raster, gt_struct_xy, gt_support_xy, gt_centre, yaw_steps,
                 verbose=True):
    """Sweep yaw; for each, find the best XY shift by FFT cross correlation.

    Scoring is a masked normalised cross correlation, not a raw overlap count.
    The raw count is unusable here for two reasons: it rewards dropping the GT
    onto any densely built-up patch of the SLAM cloud regardless of pattern, and
    it says nothing about SLAM structure sitting where the GT has none.

    The normaliser is evaluated over the GT's *support* — the cells the scanner
    covered at all, ground included — so the null hypothesis is "what fraction of
    SLAM cells inside the scanned region are structure". A window where every
    SLAM cell is occupied has zero variance and therefore carries no information;
    those score 0 rather than 1.

    Returns a list of {yaw, score, shift} sorted by score, best first.
    """
    S = slam_raster.grid.astype(np.float32)
    nx, ny = S.shape
    cell = slam_raster.cell
    fx, fy = 1 << int(np.ceil(np.log2(2 * nx))), 1 << int(np.ceil(np.log2(2 * ny)))
    # The SLAM side is fixed, so its transform is computed once and reused.
    FS = np.fft.rfft2(S, s=(fx, fy))

    # The rotated GT is rasterised at the centre of the SLAM grid so that it is
    # always fully in bounds; otherwise the number of in-bounds GT cells changes
    # with yaw and the per-yaw scores stop being comparable. The placement offset
    # is folded back into the reported shift.
    raster_centre = slam_raster.origin + 0.5 * np.array([nx, ny]) * cell
    place = raster_centre - gt_centre

    results = []
    tick = lasio.Ticker(enabled=verbose)
    for k, yaw in enumerate(yaw_steps):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        rot = np.array([[c, -s], [s, c]])

        struct = (gt_struct_xy - gt_centre) @ rot.T + gt_centre + place
        support = (gt_support_xy - gt_centre) @ rot.T + gt_centre + place
        G = _rasterise_points(struct, slam_raster.origin, S.shape, cell)
        M = _rasterise_points(support, slam_raster.origin, S.shape, cell)
        n_g, n_m = int(G.sum()), int(M.sum())
        if n_g == 0 or n_m <= 1:
            continue

        FG = np.fft.rfft2(G.astype(np.float32), s=(fx, fy))
        FM = np.fft.rfft2(M.astype(np.float32), s=(fx, fy))
        # A = overlap with GT structure, B = SLAM structure inside the GT support
        A = np.fft.irfft2(FS * np.conj(FG), s=(fx, fy))
        B = np.fft.irfft2(FS * np.conj(FM), s=(fx, fy))

        var_s = B - B * B / n_m           # S is binary, so sum(S^2) == sum(S)
        var_g = n_g - n_g * n_g / n_m
        denom = np.sqrt(np.clip(var_s, 1e-9, None) * max(var_g, 1e-9))
        ncc = (A - B * (n_g / n_m)) / denom
        ncc[var_s <= 1e-6] = 0.0          # saturated window: no information

        peak = int(np.argmax(ncc))
        pi, pj = np.unravel_index(peak, ncc.shape)
        # Wrap the circular-correlation index into a signed shift.
        di = pi - fx if pi > fx // 2 else pi
        dj = pj - fy if pj > fy // 2 else pj
        results.append({
            "yaw": float(yaw),
            "score": float(ncc[pi, pj]),
            "shift": (float(place[0] + di * cell), float(place[1] + dj * cell)),
        })
        tick(f"    yaw sweep {k + 1}/{len(yaw_steps)}  "
             f"best so far {max(r['score'] for r in results):.4f}")
    tick.done(f"    yaw sweep over {len(results)} angles")
    return sorted(results, key=lambda r: -r["score"])


def transform_from(yaw, shift_xy, dz, gt_centre):
    """Assemble the GT -> SLAM 4x4 used during the search."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([gt_centre[0], gt_centre[1], 0.0])
    t = centre - R @ centre + np.array([shift_xy[0], shift_xy[1], dz])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def distinct_candidates(results, n=5, min_sep=10.0):
    """Top-scoring candidates that are genuinely different hypotheses.

    The angles either side of a peak belong to that same peak and always score
    highly, so a plain top-N is one hypothesis listed N times. Separating by
    yaw gives the caller real alternatives to test.
    """
    out = []
    for r in results:
        if all(min(abs(r["yaw"] - k["yaw"]) % 360.0,
                   360.0 - abs(r["yaw"] - k["yaw"]) % 360.0) > min_sep
               for k in out):
            out.append(r)
        if len(out) >= n:
            break
    return out


def align(slam_path, gt_path, cell=0.30, band=(1.0, 8.0), yaw_step=2.0,
          refine_step=0.25, gt_percentile=1.0, slam_percentile=0.1,
          n_candidates=1, try_flip=True, verbose=True):
    """Coarse-align a SLAM cloud onto the GT. Returns (4x4 SLAM->GT, info).

    With `n_candidates` > 1, `info["alternatives"]` carries additional 4x4
    transforms for the next-best *distinct* yaw peaks. A top-down structure
    raster cannot always tell the right peak from a plausible one — a warehouse
    of repetitive racking, or a long corridor, produces several near-equal
    scores — so the caller can test them with ICP and keep whichever actually
    fits, instead of committing to the highest correlation.
    """
    if verbose:
        print("  Building top-down structure rasters ...")

    # The GT extent can be inflated by a handful of very distant returns; clip to
    # the dense core so the raster stays small and the correlation stays sharp.
    gt_sample = lasio.stream_voxel_downsample(gt_path, cell, verbose=False)
    lo = np.percentile(gt_sample[:, :2], gt_percentile, axis=0)
    hi = np.percentile(gt_sample[:, :2], 100 - gt_percentile, axis=0)
    core = np.all((gt_sample[:, :2] >= lo) & (gt_sample[:, :2] <= hi), axis=1)
    gt_sample = gt_sample[core]
    if verbose:
        print(f"    GT core extent {np.round(lo, 1)} .. {np.round(hi, 1)} "
              f"({int(core.sum()):,} voxels)")

    # Same treatment for the SLAM side. Points outside the raster are simply
    # not accumulated, so clipping to the dense core drops the flyers and keeps
    # the grid on the scene. Keep the percentile gentle: the SLAM cloud legitimately
    # extends beyond the GT, and anything holding more than `slam_percentile` of
    # the points survives.
    slam_mins, slam_maxs = lasio.bounds(slam_path)
    raw_lo, raw_hi = slam_mins[:2].copy(), slam_maxs[:2].copy()
    if slam_percentile and slam_percentile > 0:
        lo_s, hi_s = dense_extent(slam_path, slam_percentile, verbose=verbose,
                                  tag="SLAM ")
        slam_mins = np.concatenate([lo_s, slam_mins[2:]])
        slam_maxs = np.concatenate([hi_s, slam_maxs[2:]])
    origin = slam_mins[:2] - cell
    shape = np.ceil((slam_maxs[:2] + cell - origin) / cell).astype(np.int64) + 1
    if verbose:
        raw_shape = np.ceil((raw_hi + 2 * cell - raw_lo) / cell).astype(np.int64) + 1
        print(f"    SLAM raster {int(shape[0])} x {int(shape[1])} at {cell:.2f} m")
        if int(raw_shape[0]) * int(raw_shape[1]) > 2 * int(shape[0]) * int(shape[1]):
            print(f"      (full bounding box would have been "
                  f"{int(raw_shape[0])} x {int(raw_shape[1])} — "
                  f"{raw_shape.prod() / max(shape.prod(), 1):.0f}x larger; "
                  f"a flyer tail was setting the extent)")

    slam_raster, slam_ground = structure_raster(
        slam_path, cell, origin, shape, band=band, verbose=verbose, tag="SLAM ")

    # Gravity sign. Both clouds are gravity-aligned, but "aligned" does not fix
    # the *direction*: a SLAM cloud can come out with +z pointing at the floor
    # instead of the ceiling, and then the true transform is a 180 deg roll,
    # which a yaw-only search can never reach. Left unhandled, ICP settles into
    # the best floor-on-ceiling fit -- on an indoor scene that looks superficially
    # fine and left every station 21-27 cm out.
    #
    # Only ONE extra hypothesis is needed. R_x(180) and R_y(180) differ by a
    # 180 deg yaw, and yaw is already swept over the full circle, so flipping
    # about x covers the whole inverted-gravity case.
    #
    # The GT sample is flipped rather than the SLAM cloud: it is a small
    # in-memory array, so the expensive SLAM rasters are built once and reused
    # for both hypotheses. Flipping the GT makes both clouds share the same
    # (possibly inverted) convention, so "structure above the per-cell minimum
    # z" means the same thing on each side, which is what the score compares.
    FLIP = np.diag([1.0, -1.0, -1.0])
    hypotheses = [False, True] if try_flip else [False]
    searched = []
    for flip in hypotheses:
        sample = gt_sample * np.array([1.0, -1.0, -1.0]) if flip else gt_sample
        gt_lo = sample.min(axis=0)[:2] - cell
        gt_shape = np.ceil((sample.max(axis=0)[:2] + cell - gt_lo) / cell).astype(np.int64) + 1
        gt_raster, gt_ground = structure_raster(
            sample, cell, gt_lo, gt_shape, band=band, min_points=1,
            verbose=verbose, tag=f"GT{'(flipped)' if flip else ''} ")
        gt_xy = (np.argwhere(gt_raster.grid) + 0.5) * cell + gt_lo
        # Support = every cell the scanner reached, ground included. This is the
        # region the normaliser is evaluated over.
        gt_support_xy = (np.argwhere(np.isfinite(gt_ground)) + 0.5) * cell + gt_lo
        if len(gt_xy) == 0:
            raise RuntimeError("no above-ground structure in the GT; adjust `band`")
        if verbose:
            print(f"    SLAM structure cells {int(slam_raster.grid.sum()):,}   "
                  f"GT structure cells {len(gt_xy):,}   "
                  f"GT support cells {len(gt_support_xy):,}"
                  f"{'   [z-flipped hypothesis]' if flip else ''}")
        gt_centre = gt_xy.mean(axis=0)
        res = align_yaw_xy(slam_raster, gt_xy, gt_support_xy, gt_centre,
                           np.arange(0.0, 360.0, yaw_step), verbose=verbose)
        searched.append({"flip": flip, "coarse": res, "gt_xy": gt_xy,
                         "gt_support_xy": gt_support_xy, "gt_centre": gt_centre,
                         "gt_ground": gt_ground, "gt_lo": gt_lo})

    searched.sort(key=lambda h: -h["coarse"][0]["score"])
    if try_flip and verbose:
        for h in searched:
            print(f"    gravity {'DOWN (flipped)' if h['flip'] else 'UP  (as-is)'}: "
                  f"best raster score {h['coarse'][0]['score']:.4f} "
                  f"at yaw {h['coarse'][0]['yaw']:.1f} deg")
        if searched[0]["flip"]:
            print("    NOTE: the z-flipped hypothesis scores higher — this SLAM "
                  "cloud appears to have gravity inverted relative to the GT.")

    chosen = searched[0]
    flip = chosen["flip"]
    gt_xy, gt_support_xy = chosen["gt_xy"], chosen["gt_support_xy"]
    gt_centre, gt_ground, gt_lo = chosen["gt_centre"], chosen["gt_ground"], chosen["gt_lo"]
    coarse = chosen["coarse"]
    best = coarse[0]
    if verbose:
        print("    top yaw candidates:")
        for r in coarse[:5]:
            print(f"      yaw={r['yaw']:7.2f} deg  score={r['score']:.4f}  "
                  f"shift=({r['shift'][0]:8.2f}, {r['shift'][1]:8.2f}) m")

    fine_range = np.arange(best["yaw"] - yaw_step, best["yaw"] + yaw_step + 1e-9,
                           refine_step)
    fine = align_yaw_xy(slam_raster, gt_xy, gt_support_xy, gt_centre, fine_range,
                        verbose=verbose)
    best = fine[0] if fine[0]["score"] >= best["score"] else best
    if verbose:
        print(f"    refined: yaw={best['yaw']:.2f} deg  score={best['score']:.4f}")

    # Height offset from the ground rasters, over cells both clouds observed.
    T_gt_to_slam_xy = transform_from(best["yaw"], best["shift"], 0.0, gt_centre)
    gt_pts = np.column_stack([gt_xy, np.zeros(len(gt_xy))])
    moved = gt_pts @ T_gt_to_slam_xy[:3, :3].T + T_gt_to_slam_xy[:3, 3]
    ij = np.floor((moved[:, :2] - origin) / cell).astype(np.int64)
    ok = ((ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) &
          (ij[:, 1] >= 0) & (ij[:, 1] < shape[1]))
    gt_ij = np.floor((gt_xy - gt_lo) / cell).astype(np.int64)
    sg = slam_ground[ij[ok, 0], ij[ok, 1]]
    gg = gt_ground[gt_ij[ok, 0], gt_ij[ok, 1]]
    both = np.isfinite(sg) & np.isfinite(gg)
    dz = float(np.median(sg[both] - gg[both])) if both.any() else 0.0
    if verbose:
        print(f"    ground offset dz = {dz:.3f} m  (from {int(both.sum()):,} cells)")

    # If the search ran against a flipped GT, undo the flip on the way back.
    # The search solved  x_slam ~ T' F x_gt, so  x_gt ~ F inv(T') x_slam
    # (F is its own inverse).
    F4 = np.eye(4)
    F4[:3, :3] = FLIP

    def _to_gt(T_gt_to_slam_local):
        T = np.linalg.inv(T_gt_to_slam_local)
        return F4 @ T if flip else T

    T_gt_to_slam = transform_from(best["yaw"], best["shift"], dz, gt_centre)
    T_slam_to_gt = _to_gt(T_gt_to_slam)

    # The runner-up must be a genuinely different hypothesis. The angles either
    # side of the winner belong to the same peak and always score highly, so
    # comparing against them would flag every correct alignment as ambiguous.
    rival = 0.0
    for r in coarse:
        sep = abs(r["yaw"] - best["yaw"]) % 360.0
        if min(sep, 360.0 - sep) > 5.0:
            rival = float(r["score"])
            break
    if verbose:
        print(f"    best score {best['score']:.4f}, best rival more than 5 deg "
              f"away {rival:.4f}")

    alternatives = []
    if n_candidates > 1:
        for cand in distinct_candidates(coarse, n=n_candidates)[1:]:
            Ta = transform_from(cand["yaw"], cand["shift"], dz, gt_centre)
            alternatives.append({"yaw": cand["yaw"], "score": cand["score"],
                                 "flip": flip,
                                 "transform": _to_gt(Ta).tolist()})
        if verbose:
            print(f"    {len(alternatives)} distinct alternative peak(s) to test: "
                  + ", ".join(f"{a['yaw']:.0f} deg ({a['score']:.3f})"
                              for a in alternatives))

    if try_flip and len(searched) > 1 and n_candidates > 1:
        other = searched[1]
        ob = other["coarse"][0]
        oT = transform_from(ob["yaw"], ob["shift"], dz, other["gt_centre"])
        oT = np.linalg.inv(oT)
        if other["flip"]:
            oT = F4 @ oT
        alternatives.append({"yaw": ob["yaw"], "score": ob["score"],
                             "flip": other["flip"], "transform": oT.tolist()})
        if verbose:
            print(f"    also testing the losing gravity hypothesis "
                  f"({'flipped' if other['flip'] else 'as-is'}, "
                  f"score {ob['score']:.4f}) — the raster score alone is not "
                  f"decisive enough to discard it")

    info = {
        "flip": bool(flip),
        "alternatives": alternatives,
        "yaw_deg": best["yaw"],
        "shift_xy": best["shift"],
        "dz": dz,
        "score": best["score"],
        "rival_score": rival,
        "cell": cell,
        "band": list(band),
        "candidates": coarse[:5],
    }
    return T_slam_to_gt, info
