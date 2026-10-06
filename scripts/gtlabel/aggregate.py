#!/usr/bin/env python3
"""
Local aggregation of the SLAM channels over each point's kNN neighbourhood.

Every SLAM channel describes one point at one instant. `mean_knn_dist` and 
roughness are the only features that know anything about neighbours, which 
is most of why plain SOR is hard to beat: it uses spatial context and the 
confidence channels do not. This module gives them context.

The features that matter are the **deviations**, not the neighbourhood means:

    <ch>__dev_k{K}   c - mean_K(c)        how far this point sits from its
                                          neighbourhood on this channel
    <ch>__z_k{K}     dev / (std_K(c)+eps) the same, in units of local spread
    <ch>__nmean_k{K} mean_K(c)            the neighbourhood level itself
    <ch>__nstd_k{K}  std_K(c)             local disagreement
    <ch>__ninv_k{K}  fraction of neighbours whose value is the -1 sentinel

A wall region with uniformly high `balm_res` is a badly modelled surface; one
point with high `balm_res` among low ones is a flyer. The raw channel cannot
tell those apart and `dev` can.

Grouping is by kNN in the assembled cloud rather than by SLAM voxel. The voxel
grouping would keep the features purely SLAM-internal, but (a) `voxel_id` is
currently reused across the file so it cannot be used as a grouping key, (b) the
voxel partition is the SLAM's own bookkeeping, not a surface neighbourhood, and
(c) voxel membership is fixed at capture time, so two points that ended up
adjacent in the final cloud but came from different voxels are never compared —
which is exactly where inconsistency shows up.

Reproducibility note: LAS coordinates are quantised to 1 mm, so exact ties at
the k-th neighbour are common — about 0.17% of points on HMS. Two KD-tree builds
over the same cloud can break such a tie differently, which shifts that point's
neighbourhood mean by one neighbour out of k. This is an ambiguity in the
definition of "the k nearest neighbours", not an implementation defect; do not
spend time chasing it when a brute-force check disagrees on a fraction of a
percent of points.

Honest bookkeeping for the write-up: these features borrow the final cloud's
*topology* (who is next to whom) while the values being aggregated are all
SLAM-internal. They are therefore neither purely SLAM-internal nor geometric,
and should be reported as their own bucket.
"""

import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from . import lasio
from .c2c import TileIndex
from .knn import _TILE_DTYPE, _probe_counts, _choose_tile_size, _bin_to_tiles

SENTINEL = -1.0        # the marker used by the *input* channels
SENTINEL_TOL = 1e-6
# The outputs use NaN, not -1: `dev` and `z` are signed, so a legitimate
# deviation of exactly -1 would be indistinguishable from "invalid". HGBR splits
# on NaN natively, so nothing is lost.
INVALID = np.nan
EPS = 1e-6
DEFAULT_STATS = ("dev", "z")
ALL_STATS = ("nmean", "nstd", "dev", "z", "ninv")


def _load_channels(path, names, verbose=True):
    """Read the named dimensions into float32 arrays, streamed."""
    import laspy
    total = lasio.point_count(path)
    out = {n: np.empty(total, dtype=np.float32) for n in names}
    at = 0
    tick = lasio.Ticker(enabled=verbose)
    with laspy.open(path) as f:
        for chunk in f.chunk_iterator(10_000_000):
            for n in names:
                out[n][at:at + len(chunk)] = np.asarray(chunk[n], dtype=np.float32)
            at += len(chunk)
            tick(f"    loading channels: {at:,} / {total:,}")
    tick.done(f"    loaded {len(names)} channels over {total:,} points")
    return out


def aggregate_features(path, channels, ks=(30,), stats=DEFAULT_STATS,
                       tile_budget=4_000_000, scratch_dir=None, block=300_000,
                       keep_tiles=False, verbose=True):
    """Aggregate `channels` over each point's k nearest neighbours."""
    if scratch_dir is None:
        raise ValueError("scratch_dir is required")
    bad = [s for s in stats if s not in ALL_STATS]
    if bad:
        raise ValueError(f"unknown stats {bad}; choose from {ALL_STATS}")
    ks = sorted(ks)
    total = lasio.point_count(path)
    mins, maxs = lasio.bounds(path)

    if verbose:
        print(f"  Local aggregation over {total:,} points")
        print(f"    channels: {', '.join(channels)}")
        print(f"    k={list(ks)}   stats={list(stats)}   "
              f"-> {len(channels) * len(ks) * len(stats)} new columns")

    values = _load_channels(path, channels, verbose=verbose)

    margin = 1.0
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

    out = {}
    for ch in channels:
        for k in ks:
            for st in stats:
                out[f"{ch}__{st}_k{k}"] = np.full(total, INVALID, dtype=np.float32)

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
        if not inside.any():
            del rec, pts
            continue

        tree = cKDTree(pts)
        q_pos = np.flatnonzero(inside)
        tile_idx = rec["idx"]                       # global index of every tile point

        for k in ks:
            kq = min(k + 1, len(pts))
            for start in range(0, len(q_pos), block):
                sel = q_pos[start:start + block]
                _, nb = tree.query(pts[sel], k=kq, workers=-1)
                if kq == 1:
                    nb = nb[:, None]
                nb = nb[:, 1:] if kq > 1 else nb     # drop self
                gidx = tile_idx[nb]                  # (B, k) global indices
                own = tile_idx[sel]

                for ch in channels:
                    v = values[ch]
                    nbv = v[gidx].astype(np.float32)
                    invalid = np.abs(nbv - SENTINEL) <= SENTINEL_TOL
                    good = ~invalid
                    n_good = good.sum(axis=1)
                    safe = np.where(good, nbv, 0.0)
                    s1 = safe.sum(axis=1)
                    mean = s1 / np.maximum(n_good, 1)
                    var = (np.where(good, (nbv - mean[:, None]) ** 2, 0.0).sum(axis=1)
                           / np.maximum(n_good, 1))
                    std = np.sqrt(np.maximum(var, 0.0))

                    # Points whose own value is the sentinel, or with no valid
                    # neighbour, keep the sentinel rather than a fabricated zero.
                    ov = v[own]
                    usable = (n_good > 0) & (np.abs(ov - SENTINEL) > SENTINEL_TOL)
                    dev = np.where(usable, ov - mean, INVALID)

                    for st in stats:
                        key = f"{ch}__{st}_k{k}"
                        if st == "nmean":
                            out[key][own] = np.where(n_good > 0, mean, INVALID)
                        elif st == "nstd":
                            out[key][own] = np.where(n_good > 0, std, INVALID)
                        elif st == "dev":
                            out[key][own] = dev
                        elif st == "z":
                            out[key][own] = np.where(
                                usable, (ov - mean) / (std + EPS), INVALID)
                        elif st == "ninv":
                            out[key][own] = invalid.mean(axis=1)
                    del nbv, invalid, good, safe
                del gidx, nb

        done += int(inside.sum())
        del tree, rec, pts
        tick(f"    tile {tid + 1}/{tiles.n_tiles}: {done:,} / {total:,}")
    tick.done(f"    aggregated {done:,} / {total:,} points")

    if not keep_tiles and os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    return out
