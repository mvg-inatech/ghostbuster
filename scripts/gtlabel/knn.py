#!/usr/bin/env python3
"""
Mean distance to the k nearest neighbours, computed without holding a KD-tree
over the whole cloud.

This is the `mean_knn_dist` channel — the one signal with no SLAM origin, and
the stand-in for the SOR baseline the confidence channels are measured against.

A single cKDTree over 194 M points needs roughly 10 GB on top of the
coordinates, so the cloud is tiled the same way `c2c.py` tiles the ground truth:
each tile is loaded with a margin, a tree is built over tile+margin, and only
the points strictly inside the tile are queried. Peak memory is one tile.

Exactness: for a point inside the tile, every neighbour within `margin` of it is
present in the loaded set, so any k-th neighbour distance at or below `margin`
is exact. Distances above it could in principle be overestimates — the true
neighbour would have to lie beyond the margin — so those points are counted and
reported. They are isolated points by construction, which the channel is
supposed to score as suspicious anyway, and an overestimate keeps them there.

Matches the CloudCompare definition used previously: k neighbours excluding
self.
"""

import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from . import lasio
from .c2c import TileIndex

# (int64 index, float32 x, float32 y, float32 z) per tile entry.
_TILE_DTYPE = np.dtype([("idx", "<i8"), ("xyz", "<f4", (3,))])


def _probe_counts(path, mins, maxs, probe, verbose=True):
    """Point counts on a coarse XY grid, for choosing a tile size."""
    span = maxs[:2] - mins[:2]
    shape = np.maximum(np.ceil(span / probe).astype(np.int64), 1)
    counts = np.zeros((int(shape[0]), int(shape[1])), dtype=np.int64)
    tick = lasio.Ticker(enabled=verbose)
    seen = 0
    for xyz in lasio.iter_xyz(path):
        seen += len(xyz)
        ij = np.floor((xyz[:, :2] - mins[:2]) / probe).astype(np.int64)
        np.clip(ij, [0, 0], [shape[0] - 1, shape[1] - 1], out=ij)
        flat = ij[:, 0] * shape[1] + ij[:, 1]
        counts += np.bincount(flat, minlength=int(np.prod(shape))
                              ).reshape(int(shape[0]), int(shape[1]))
        tick(f"    density scan: {seen:,} points")
    tick.done(f"    density scan: {seen:,} points")
    return counts


def _choose_tile_size(counts, probe, budget, margin, verbose=True):
    """Largest tile whose worst case stays under `budget`, and above the margin."""
    for mult in (64, 32, 16, 8, 4, 2, 1):
        ts = probe * mult
        if ts <= 2.0 * margin:
            continue
        pad_x = (-counts.shape[0]) % mult
        pad_y = (-counts.shape[1]) % mult
        padded = np.pad(counts, ((0, pad_x), (0, pad_y)))
        blocks = padded.reshape(padded.shape[0] // mult, mult,
                                padded.shape[1] // mult, mult).sum(axis=(1, 3))
        worst = int(blocks.max()) if blocks.size else 0
        if worst <= budget:
            if verbose:
                print(f"    tile size {ts:.1f} m -> worst tile {worst:,} points "
                      f"(budget {budget:,})")
            return ts
    ts = max(probe, 2.5 * margin)
    if verbose:
        print(f"    WARNING: no tile size fits the {budget:,} point budget; "
              f"using {ts:.1f} m")
    return ts


def _bin_to_tiles(path, tiles, origin, scratch_dir, verbose=True):
    """One pass over the cloud, writing (index, xyz) into per-tile files."""
    os.makedirs(scratch_dir, exist_ok=True)
    handles = {}
    written = np.zeros(tiles.n_tiles, dtype=np.int64)
    total = lasio.point_count(path)
    tick = lasio.Ticker(enabled=verbose)
    at = 0
    try:
        for xyz in lasio.iter_xyz(path):
            idx_block = np.arange(at, at + len(xyz), dtype=np.int64)
            at += len(xyz)
            lo, hi = tiles.cell_range(xyz[:, :2])
            # margin < tile size, so a point reaches at most a 2x2 neighbourhood
            for dx in (0, 1):
                for dy in (0, 1):
                    ix, iy = lo[:, 0] + dx, lo[:, 1] + dy
                    take = (ix <= hi[:, 0]) & (iy <= hi[:, 1])
                    if not take.any():
                        continue
                    flat = ix[take] * tiles.shape[1] + iy[take]
                    rec = np.empty(int(take.sum()), dtype=_TILE_DTYPE)
                    rec["idx"] = idx_block[take]
                    rec["xyz"] = (xyz[take] - origin).astype(np.float32)
                    order = np.argsort(flat, kind="stable")
                    flat, rec = flat[order], rec[order]
                    edges = np.flatnonzero(np.diff(flat)) + 1
                    ids = flat[np.concatenate([[0], edges])]
                    for grp, tid in zip(np.split(rec, edges), ids):
                        tid = int(tid)
                        if tid not in handles:
                            handles[tid] = open(
                                os.path.join(scratch_dir, f"k_{tid:06d}.bin"), "wb")
                        handles[tid].write(grp.tobytes())
                        written[tid] += len(grp)
            tick(f"    binning: {at:,} / {total:,} points")
    finally:
        for h in handles.values():
            h.close()
    tick.done(f"    binned {at:,} points into {int((written > 0).sum())} tiles")
    return written, at


def mean_knn_distance(path, k=6, margin=1.0, tile_budget=6_000_000,
                      scratch_dir=None, keep_tiles=False, verbose=True):
    """Mean distance to the k nearest neighbours, in source point order.

    Returns (values, n_uncertain) where `n_uncertain` counts points whose k-th
    neighbour came out beyond `margin`.
    """
    if scratch_dir is None:
        raise ValueError("scratch_dir is required")

    total = lasio.point_count(path)
    mins, maxs = lasio.bounds(path)
    if verbose:
        print(f"  Mean kNN distance (k={k}) over {total:,} points ...")

    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    probe = max(2.0 * margin, span / 512.0)
    counts = _probe_counts(path, mins, maxs, probe, verbose=verbose)
    tile_size = _choose_tile_size(counts, probe, tile_budget, margin,
                                  verbose=verbose)

    tiles = TileIndex(mins, maxs, tile_size, margin)
    if verbose:
        print(f"    tiling {tiles.shape[0]} x {tiles.shape[1]} "
              f"= {tiles.n_tiles} tiles at {tile_size:.1f} m")

    origin = mins.copy()
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    written, _ = _bin_to_tiles(path, tiles, origin, scratch_dir, verbose=verbose)

    out = np.full(total, -1.0, dtype=np.float64)
    uncertain = 0
    done = 0
    tick = lasio.Ticker(enabled=verbose)
    for tid in range(tiles.n_tiles):
        if written[tid] == 0:
            continue
        tile_file = os.path.join(scratch_dir, f"k_{tid:06d}.bin")
        if not os.path.exists(tile_file):
            continue
        rec = np.fromfile(tile_file, dtype=_TILE_DTYPE)
        pts = rec["xyz"].astype(np.float64)

        # Only points owned by this tile are queried; the rest are margin.
        owner = tiles.flat(tiles.cell_of(pts[:, :2] + origin[:2]))
        inside = owner == tid
        if not inside.any():
            del rec, pts
            continue

        tree = cKDTree(pts)
        kq = min(k + 1, len(pts))
        dists, _ = tree.query(pts[inside], k=kq, workers=-1)
        if kq < k + 1:
            # Fewer points in the tile than k+1: average what exists.
            neigh = dists[:, 1:] if dists.ndim > 1 else dists[:, None]
        else:
            neigh = dists[:, 1:]
        values = neigh.mean(axis=1)
        out[rec["idx"][inside]] = values
        uncertain += int((neigh[:, -1] > margin).sum())

        done += int(inside.sum())
        del tree, rec, pts
        tick(f"    tile {tid + 1}/{tiles.n_tiles}: {done:,} / {total:,} queried")
    tick.done(f"    queried {done:,} / {total:,} points")

    if not keep_tiles and os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)

    missing = int((out < 0).sum())
    if missing and verbose:
        print(f"    WARNING: {missing:,} points were never assigned a value")
    if uncertain and verbose:
        print(f"    {uncertain:,} points ({100.0 * uncertain / max(done, 1):.3f} %) "
              f"had their k-th neighbour beyond the {margin:.2f} m margin; "
              f"their value may be an overestimate")
    return out, uncertain
