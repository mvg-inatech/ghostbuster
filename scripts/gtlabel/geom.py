#!/usr/bin/env python3
"""
Multi-scale geometric features: the strong version of the conventional baseline.

Two families, both computed post-hoc on the assembled cloud, both belonging to
the *geometric* bucket rather than the SLAM-internal one:

  mean_knn_dist_k{K}   mean distance to the K nearest neighbours — the SOR
                       statistic at several scales
  roughness_r{R}       distance from the point to a plane fitted over its
                       neighbours within radius R

Why several scales. Small K measures sensor noise and isolated flyers; large K
measures regional density. The case that separates them is a compact cluster of
wrong points — a multipath blob, a ghost surface — where every point has close
neighbours and looks healthy at K=6, while K=100 exposes the whole blob. A
single-scale baseline is blind to exactly the errors the confidence channels are
supposed to catch, so beating it proves less than it appears to.

Cost is not a reason to pick one scale: all K come out of a single query as
prefixes of the same sorted distance array.

Why roughness is radius-based rather than fixed-K. Point density here varies by
orders of magnitude across the scan (range 0.5 m to 140 m), so a fixed-K patch
spans centimetres near the sensor and metres at the far end — a fixed-K
roughness would partly be measuring range, which is already a channel. To keep
the neighbourhood physically sized without unbounded neighbour counts, each
scale fits its plane against a copy of the cloud voxelised at R/5, so an R-ball
holds ~80 points whatever the local density.
"""

import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from . import lasio
from .c2c import TileIndex
from .knn import _TILE_DTYPE, _probe_counts, _choose_tile_size, _bin_to_tiles

SENTINEL = -1.0
BALL_K = 100          # neighbours pulled per radius from the R/5-voxelised copy
MIN_FIT = 4           # points needed before a plane means anything


def _voxel_reduce(pts, voxel):
    """One representative point per voxel — a physically sized thinning."""
    ijk = np.floor(pts / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, idx = np.unique(key, return_index=True)
    return pts[idx]


def _roughness(query, reference, radius, block=400_000):
    """|distance to the plane fitted over neighbours within `radius`|."""
    tree = cKDTree(reference)
    out = np.full(len(query), SENTINEL, dtype=np.float64)
    k = min(BALL_K, len(reference))
    if k < MIN_FIT:
        return out

    for start in range(0, len(query), block):
        q = query[start:start + block]
        d, idx = tree.query(q, k=k, distance_upper_bound=radius, workers=-1)
        if k == 1:
            d, idx = d[:, None], idx[:, None]
        valid = np.isfinite(d)
        n = valid.sum(axis=1)
        ok = n >= MIN_FIT
        if not ok.any():
            continue

        # Out-of-range hits come back as len(reference); clamp then mask them
        # out by weight so the gather stays in bounds.
        safe = np.where(valid, idx, 0)
        nb = reference[safe]                              # (B, k, 3)
        w = valid.astype(np.float64)[..., None]
        cnt = w.sum(axis=1)
        centroid = (nb * w).sum(axis=1) / np.maximum(cnt, 1)
        cen = (nb - centroid[:, None, :]) * w
        cov = np.einsum("bki,bkj->bij", cen, cen) / np.maximum(cnt[:, None], 1)

        sub = np.flatnonzero(ok)
        evals, evecs = np.linalg.eigh(cov[sub])
        normal = evecs[:, :, 0]                           # smallest eigenvalue
        out[start + sub] = np.abs(
            ((q[sub] - centroid[sub]) * normal).sum(axis=1))
    return out


def _roughness_k(query, tree, pts, k, block=400_000):
    """Fixed-K roughness: plane fitted over the K nearest neighbours.

    The CloudCompare-style definition, kept for comparison against the
    radius-based one. On a scan whose density varies with range, a fixed K
    spans a different physical size everywhere, so this partly encodes range
    rather than surface roughness — which is why the radius version exists.
    """
    out = np.full(len(query), SENTINEL, dtype=np.float64)
    kq = min(k, len(pts))
    if kq < MIN_FIT:
        return out
    for start in range(0, len(query), block):
        q = query[start:start + block]
        _, idx = tree.query(q, k=kq, workers=-1)
        nb = pts[idx]                                    # (B, kq, 3)
        centroid = nb.mean(axis=1)
        cen = nb - centroid[:, None, :]
        cov = np.einsum("bki,bkj->bij", cen, cen) / kq
        _, evecs = np.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        out[start:start + len(q)] = np.abs(
            ((q - centroid) * normal).sum(axis=1))
    return out


def geometric_features(path, ks=(6, 30, 100), radii=(0.05, 0.15, 0.40),
                       k_rough=(), tile_budget=4_000_000, scratch_dir=None,
                       keep_tiles=False, rows=None, verbose=True):
    """Compute every scale in one tiled pass. Returns {name: array}.

    With `rows` (global point indices) only those points are QUERIED, and the
    returned arrays have length len(rows) rather than one entry per point. Every
    tile is still built from every point, so each neighbourhood is exactly what
    it would have been -- a kNN distance does not depend on how many points you
    ask about.

    Worth being clear about what this does and does not save, because it is not
    proportional. Three things happen per tile: the cloud is streamed and binned
    (all points, unavoidable), a KD-tree is built over the tile including its
    halo (all points, unavoidable), and the tree is queried (only the points
    asked about). Only the third scales with `rows`. The filter comparison never
    reads more than a few million rows per scene, so querying all of a
    600 M-point cloud to use 4 M of them is the one part worth avoiding.""" 
    if scratch_dir is None:
        raise ValueError("scratch_dir is required")
    ks = sorted(ks)
    radii = sorted(radii)
    k_rough = sorted(k_rough)
    margin = max(max(radii) * 1.5, 1.0)

    total = lasio.point_count(path)
    mins, maxs = lasio.bounds(path)

    if rows is None:
        row_of, sel_mask, n_out = None, None, total
    else:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.min() < 0 or rows.max() >= total:
            raise ValueError("rows out of range for this cloud")
        row_of = np.full(total, -1, dtype=np.int64)
        row_of[rows] = np.arange(len(rows), dtype=np.int64)
        sel_mask = np.zeros(total, dtype=bool)
        sel_mask[rows] = True
        n_out = len(rows)

    if verbose:
        print(f"  Geometric features over {total:,} points"
              + (f", queried for {n_out:,} ({100.0 * n_out / total:.2f} %)"
                 if rows is not None else ""))
        print(f"    kNN scales k={list(ks)}   roughness radii={list(radii)} m"
              + (f"   roughness k={list(k_rough)}" if k_rough else ""))

    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    probe = max(2.0 * margin, span / 512.0)
    counts = _probe_counts(path, mins, maxs, probe, verbose=verbose)
    tile_size = _choose_tile_size(counts, probe, tile_budget, margin, verbose)

    tiles = TileIndex(mins, maxs, tile_size, margin)
    if verbose:
        print(f"    tiling {tiles.shape[0]} x {tiles.shape[1]} "
              f"= {tiles.n_tiles} tiles at {tile_size:.1f} m")
    origin = mins.copy()
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    written, _ = _bin_to_tiles(path, tiles, origin, scratch_dir, verbose=verbose)

    out = {f"mean_knn_dist_k{k}": np.full(n_out, SENTINEL) for k in ks}
    out.update({f"roughness_r{int(round(r * 100)):03d}": np.full(n_out, SENTINEL)
                for r in radii})
    out.update({f"roughness_k{k}": np.full(n_out, SENTINEL) for k in k_rough})

    tick = lasio.Ticker(enabled=verbose)
    done = 0
    for tid in range(tiles.n_tiles):
        f = os.path.join(scratch_dir, f"k_{tid:06d}.bin")
        if written[tid] == 0 or not os.path.exists(f):
            continue
        rec = np.fromfile(f, dtype=_TILE_DTYPE)
        pts = rec["xyz"].astype(np.float64)
        owner = tiles.flat(tiles.cell_of(pts[:, :2] + origin[:2]))
        inside = owner == tid
        if sel_mask is not None:
            # Narrow the QUERY set only. `pts` -- and therefore the tree built
            # from it below -- still holds every point of the tile and its halo.
            inside &= sel_mask[rec["idx"]]
        if not inside.any():
            del rec, pts
            continue
        q = pts[inside]
        qi = rec["idx"][inside]
        if row_of is not None:
            qi = row_of[qi]          # global index -> output row

        # All k at once: the distances come back sorted, so every requested k is
        # a prefix of the same query.
        #
        # Blocked, because the result is (len(q), kq) float64 and scipy also
        # builds the matching index array: unblocked at kq=101 that is ~1.6 kB
        # per query point, so a tile holding 10 M points would want ~16 GB just
        # for this call. Blocking makes the tile budget -- and therefore the
        # tile size -- a free parameter, and tile size is what governs how much
        # halo overlap gets recomputed (3.24x at 2.5 m, 1.56x at 8 m).
        kq = min(max(ks) + 1, len(pts))
        tree = cKDTree(pts)
        qblock = 500_000
        for k in ks:
            if k + 1 <= kq:
                out[f"mean_knn_dist_k{k}"][qi] = 0.0
        for start in range(0, len(q), qblock):
            sl = slice(start, start + qblock)
            d, _ = tree.query(q[sl], k=kq, workers=-1)
            for k in ks:
                if k + 1 <= kq:
                    out[f"mean_knn_dist_k{k}"][qi[sl]] = d[:, 1:k + 1].mean(axis=1)
            del d

        for k in k_rough:
            out[f"roughness_k{k}"][qi] = _roughness_k(q, tree, pts, k)
        del tree

        for r in radii:
            ref = _voxel_reduce(pts, r / 5.0)
            out[f"roughness_r{int(round(r * 100)):03d}"][qi] = _roughness(q, ref, r)
            del ref

        done += int(inside.sum())
        del rec, pts, q, qi
        tick(f"    tile {tid + 1}/{tiles.n_tiles}: {done:,} / {total:,}")
    tick.done(f"    computed {done:,} / {n_out:,} points")

    if not keep_tiles and os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)

    if verbose:
        for name, arr in out.items():
            good = arr[arr >= 0]
            if len(good):
                print(f"    {name:22s} median {np.median(good) * 100:7.4f} cm  "
                      f"valid {100 * len(good) / total:5.1f} %")
    return out
