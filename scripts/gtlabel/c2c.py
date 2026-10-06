#!/usr/bin/env python3
"""
Cloud-to-cloud nearest-neighbour distance from the SLAM cloud to a ground truth
cloud that is far too large for a single KD-tree.

A cKDTree over 343 M points would need tens of GB. Instead the GT is binned once
into spatial tiles on disk, and each tile is then queried against the SLAM points
that fall inside it. Peak memory is one tile, not one cloud.

Two details that matter for the labels:

* Distances are exact. The GT is tiled, never downsampled, so there is no
  quantisation bias — this matters because the smallest label threshold in use
  is tau = 1 cm and a 5 mm GT downsample would already eat half of it.
* The search is capped at `max_dist`. Points with no GT neighbour inside the cap
  come back as `inf`, which is exactly the "RTC never scanned here" signal the
  coverage mask needs, so the cap does double duty.
"""

import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from . import lasio


class TileIndex:
    """Uniform XY tiling of the SLAM cloud's footprint."""

    def __init__(self, mins, maxs, tile_size, margin):
        self.mins = np.asarray(mins, dtype=np.float64)
        self.maxs = np.asarray(maxs, dtype=np.float64)
        self.tile_size = float(tile_size)
        self.margin = float(margin)
        span = self.maxs[:2] - self.mins[:2]
        self.shape = np.maximum(np.ceil(span / self.tile_size).astype(np.int64), 1)

    @property
    def n_tiles(self):
        return int(self.shape[0] * self.shape[1])

    def cell_of(self, xy):
        """Tile index (ix, iy) containing each point, clamped to the grid."""
        ij = np.floor((xy - self.mins[:2]) / self.tile_size).astype(np.int64)
        return np.clip(ij, 0, self.shape - 1)

    def flat(self, ij):
        return ij[:, 0] * self.shape[1] + ij[:, 1]

    def cell_range(self, xy):
        """Tile index range each point belongs to once `margin` is added.

        A GT point within `margin` of a tile border is needed by the neighbouring
        tile too, so it is written to both.
        """
        lo = np.floor((xy - self.margin - self.mins[:2]) / self.tile_size).astype(np.int64)
        hi = np.floor((xy + self.margin - self.mins[:2]) / self.tile_size).astype(np.int64)
        return np.clip(lo, 0, self.shape - 1), np.clip(hi, 0, self.shape - 1)


def choose_tile_size(gt_path, slam_mins, slam_maxs, margin, budget, probe=0.5,
                     verbose=True):
    """Pick the largest tile size whose worst tile stays inside `budget` points.

    Runs one counting pass over the GT at a fine probe resolution, then evaluates
    candidate tile sizes against the resulting density map. Uniform tiles keep the
    binning pass simple; the only thing that has to adapt is their size.
    """
    span = np.asarray(slam_maxs[:2]) - np.asarray(slam_mins[:2])
    nx, ny = (np.maximum(np.ceil(span / probe).astype(np.int64), 1))
    counts = np.zeros((int(nx), int(ny)), dtype=np.int64)

    lo = np.asarray(slam_mins[:2]) - margin
    hi = np.asarray(slam_maxs[:2]) + margin
    zlo = slam_mins[2] - margin
    zhi = slam_maxs[2] + margin

    for xyz in lasio.iter_xyz(gt_path):
        m = ((xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
             (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
             (xyz[:, 2] >= zlo) & (xyz[:, 2] <= zhi))
        if not m.any():
            continue
        ij = np.floor((xyz[m, :2] - slam_mins[:2]) / probe).astype(np.int64)
        np.clip(ij, [0, 0], [nx - 1, ny - 1], out=ij)
        flat = ij[:, 0] * ny + ij[:, 1]
        counts += np.bincount(flat, minlength=int(nx * ny)).reshape(int(nx), int(ny))

    relevant = int(counts.sum())
    if verbose:
        print(f"    {relevant:,} GT points inside the SLAM footprint (+{margin:.2f} m)")

    if relevant == 0:
        raise RuntimeError(
            "no GT points overlap the SLAM cloud — check the initial transform")

    # Candidate tile sizes, coarse to fine; take the first that fits.
    for mult in (32, 16, 8, 4, 2, 1):
        ts = probe * mult
        k = mult
        # worst-case points in any ts x ts window, via a box sum over the probe grid
        pad_x = (-counts.shape[0]) % k
        pad_y = (-counts.shape[1]) % k
        padded = np.pad(counts, ((0, pad_x), (0, pad_y)))
        blocks = padded.reshape(padded.shape[0] // k, k,
                                padded.shape[1] // k, k).sum(axis=(1, 3))
        # neighbouring-tile margin can pull in up to one extra probe ring
        worst = int(blocks.max()) if blocks.size else 0
        if worst <= budget:
            if verbose:
                print(f"    tile size {ts:.2f} m -> worst tile {worst:,} points "
                      f"(budget {budget:,})")
            return ts

    if verbose:
        print(f"    WARNING: even {probe:.2f} m tiles exceed the {budget:,} point "
              f"budget; proceeding with {probe:.2f} m")
    return probe


def _bin_gt_to_tiles(gt_path, tiles, origin, scratch_dir, verbose=True):
    """One pass over the GT, appending float32 XYZ into per-tile binary files."""
    os.makedirs(scratch_dir, exist_ok=True)
    handles = {}
    written = np.zeros(tiles.n_tiles, dtype=np.int64)

    lo_xy = tiles.mins[:2] - tiles.margin
    hi_xy = tiles.maxs[:2] + tiles.margin
    lo_z = tiles.mins[2] - tiles.margin
    hi_z = tiles.maxs[2] + tiles.margin

    seen = 0
    tick = lasio.Ticker(enabled=verbose)
    try:
        for xyz in lasio.iter_xyz(gt_path):
            seen += len(xyz)
            m = ((xyz[:, 0] >= lo_xy[0]) & (xyz[:, 0] <= hi_xy[0]) &
                 (xyz[:, 1] >= lo_xy[1]) & (xyz[:, 1] <= hi_xy[1]) &
                 (xyz[:, 2] >= lo_z) & (xyz[:, 2] <= hi_z))
            if not m.any():
                continue
            sel = xyz[m]
            lo, hi = tiles.cell_range(sel[:, :2])

            # A point straddles at most 2x2 tiles for margin < tile_size.
            for dx in (0, 1):
                for dy in (0, 1):
                    ix = lo[:, 0] + dx
                    iy = lo[:, 1] + dy
                    take = (ix <= hi[:, 0]) & (iy <= hi[:, 1])
                    if not take.any():
                        continue
                    flat = ix[take] * tiles.shape[1] + iy[take]
                    pts = (sel[take] - origin).astype(np.float32)
                    order = np.argsort(flat, kind="stable")
                    flat, pts = flat[order], pts[order]
                    edges = np.flatnonzero(np.diff(flat)) + 1
                    for grp_pts, grp_id in zip(np.split(pts, edges),
                                               flat[np.concatenate([[0], edges])]):
                        tid = int(grp_id)
                        if tid not in handles:
                            handles[tid] = open(
                                os.path.join(scratch_dir, f"tile_{tid:06d}.f32"), "wb")
                        handles[tid].write(grp_pts.tobytes())
                        written[tid] += len(grp_pts)
            tick(f"    binning GT: {seen:,} points read, {written.sum():,} tiled")
    finally:
        for h in handles.values():
            h.close()

    tick.done(f"    binned {seen:,} GT points into {int((written > 0).sum())} "
              f"tiles ({written.sum():,} entries incl. margin overlap)")
    return written


def compute_c2c(slam_xyz, gt_path, max_dist=0.5, tile_size=None,
                tile_budget=8_000_000, scratch_dir=None, keep_tiles=False,
                verbose=True):
    """Nearest-neighbour distance from each SLAM point to the GT cloud.

    Returns (N,) float64. Points with no GT neighbour within `max_dist` get inf.
    """
    if scratch_dir is None:
        raise ValueError("scratch_dir is required")

    slam_mins = slam_xyz.min(axis=0)
    slam_maxs = slam_xyz.max(axis=0)
    margin = float(max_dist)

    if tile_size is None:
        if verbose:
            print("  Sizing tiles from GT density ...")
        tile_size = choose_tile_size(gt_path, slam_mins, slam_maxs, margin,
                                     tile_budget, verbose=verbose)
    if tile_size <= margin:
        # cell_range assumes a point spills into at most the neighbouring tile
        tile_size = margin * 2.0
        if verbose:
            print(f"    tile size raised to {tile_size:.2f} m to exceed the margin")

    tiles = TileIndex(slam_mins, slam_maxs, tile_size, margin)
    if verbose:
        print(f"  Tiling: {tiles.shape[0]} x {tiles.shape[1]} = {tiles.n_tiles} tiles "
              f"at {tile_size:.2f} m")

    origin = slam_mins.copy()  # keeps float32 tile storage sub-micrometre
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    written = _bin_gt_to_tiles(gt_path, tiles, origin, scratch_dir, verbose=verbose)

    # Assign SLAM points to tiles (exact bounds, no margin).
    slam_flat = tiles.flat(tiles.cell_of(slam_xyz[:, :2]))
    order = np.argsort(slam_flat, kind="stable")
    slam_sorted = slam_flat[order]
    starts = np.searchsorted(slam_sorted, np.arange(tiles.n_tiles), side="left")
    ends = np.searchsorted(slam_sorted, np.arange(tiles.n_tiles), side="right")

    dist = np.full(len(slam_xyz), np.inf, dtype=np.float64)
    n_done = 0
    tick = lasio.Ticker(enabled=verbose)
    for tid in range(tiles.n_tiles):
        idx = order[starts[tid]:ends[tid]]
        if len(idx) == 0:
            continue
        tile_file = os.path.join(scratch_dir, f"tile_{tid:06d}.f32")
        if written[tid] == 0 or not os.path.exists(tile_file):
            continue  # no GT here: stays inf -> flagged as uncovered
        gt_tile = np.fromfile(tile_file, dtype=np.float32).reshape(-1, 3)
        tree = cKDTree(gt_tile.astype(np.float64) + origin)
        d, _ = tree.query(slam_xyz[idx], k=1, distance_upper_bound=max_dist,
                          workers=-1)
        dist[idx] = d
        n_done += len(idx)
        del tree, gt_tile
        tick(f"    tile {tid + 1}/{tiles.n_tiles}: {n_done:,} / "
             f"{len(slam_xyz):,} SLAM points queried")

    finite = np.isfinite(dist)
    tick.done(f"    queried {n_done:,} points; {int(finite.sum()):,} found a GT "
              f"neighbour within {max_dist:.2f} m")

    if not keep_tiles and os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)

    return dist
