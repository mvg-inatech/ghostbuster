"""Local aggregation, restructured to stop repeating the neighbour search.

Same statistics and same column names as `aggregate.aggregate_features`; this
only changes how the work is ordered. Two redundancies were costing ~4x:

1. One query per k. cKDTree.query returns neighbours sorted by distance, so the
   k=101 result already contains the k=31 result in its first 31 columns. The
   old loop issued both queries. Here a single query at max(k)+1 is sliced.

2. One neighbour search per *batch of channels*. Channels were split into
   batches to bound memory, and each batch rebuilt every tile's kd-tree from
   scratch -- measured at ~20 min per batch on a 74.6 M point cloud regardless
   of whether it carried 5 channels or 2, because the query is essentially the
   whole cost. The memory that forced batching is the OUTPUT (16 channels x 2 ks
   x 3 stats x 74.6 M x 4 B = 28.6 GB), not the input (~4.8 GB), so the fix is
   to write outputs to a memmap rather than to split the inputs.

Returns a dict of column name -> memmap-backed array, usable exactly like the
in-RAM dict the original returns.
"""
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from .aggregate import (ALL_STATS, DEFAULT_STATS, EPS, INVALID, SENTINEL,
                        SENTINEL_TOL, _load_channels)
from .c2c import TileIndex
from . import lasio
from .knn import _TILE_DTYPE, _probe_counts, _choose_tile_size, _bin_to_tiles


def _quiet_unlink(path):
    """Remove `path` if it is there, without caring if it is not."""
    try:
        os.unlink(path)
    except OSError:
        pass




def aggregate_features_fast(path, channels, ks=(30,), stats=DEFAULT_STATS,
                            tile_budget=4_000_000, scratch_dir=None,
                            block=300_000, keep_tiles=False, rows=None,
                            verbose=True):
    """Local aggregation of `channels` over the cloud's own kNN topology.

    With `rows` (global point indices) only those points get an output value,
    and the returned arrays have length len(rows) rather than one entry per
    point in the cloud. The NEIGHBOURHOODS are unaffected: every tile is still
    built from every point, so a selected point sees exactly the neighbours it
    would have seen otherwise. Only the output is restricted.

    That distinction is what makes this worth having. These 96 columns are a
    deterministic function of the cloud, not a measurement, and nothing ever
    reads all of them at once: compare_filters fits on a few hundred thousand
    rows and evaluates on a few million. Materialising them for every point of
    a 625 M-point cloud costs hundreds of gigabytes to store something that is
    cheaper to recompute for the rows that are actually used.
    """
    if scratch_dir is None:
        raise ValueError("scratch_dir is required")
    bad = [s for s in stats if s not in ALL_STATS]
    if bad:
        raise ValueError(f"unknown stats {bad}; choose from {ALL_STATS}")
    ks = sorted(ks)
    kmax = max(ks)
    total = lasio.point_count(path)
    mins, maxs = lasio.bounds(path)

    if rows is None:
        row_of, n_out = None, total
    else:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.min() < 0 or rows.max() >= total:
            raise ValueError("rows out of range for this cloud")
        row_of = np.full(total, -1, dtype=np.int64)
        row_of[rows] = np.arange(len(rows), dtype=np.int64)
        n_out = len(rows)

    keys = [f"{ch}__{st}_k{k}" for ch in channels for k in ks for st in stats]
    if verbose:
        print(f"  Local aggregation (fast) over {total:,} points")
        print(f"    channels: {', '.join(channels)}")
        print(f"    k={list(ks)}   stats={list(stats)}   "
              f"-> {len(keys)} new columns, one query at k={kmax}")

    values = _load_channels(path, channels, verbose=verbose)

    os.makedirs(scratch_dir, exist_ok=True)
    mm_path = os.path.join(scratch_dir, "..", os.path.basename(scratch_dir) + "_agg.f32")
    mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(len(keys), n_out))
    mm[:] = INVALID

    # Unlink the backing file straight away and keep only the mapping. The
    # caller needs these arrays alive until it has written them into the LAS,
    # so the file cannot be deleted at the end of this function -- but on POSIX
    # an unlinked file stays fully readable and writable through an open
    # mapping, and the kernel reclaims the space as soon as the last reference
    # goes, including after a crash or a kill.
    #
    # This matters at the scale the pipeline runs at: the memmap is
    # len(keys) x n_points x 4 B, which reaches tens of gigabytes on a large
    # outdoor scene and is easy to leave behind on disk,
    # because nothing ever removed them, and the file is opened "w+" (truncate)
    # on every run so they were never of any use as a cache either.
    try:
        os.unlink(mm_path)
    except OSError:
        # Windows refuses to unlink a mapped file. Fall back to removing it when
        # the interpreter exits, which still beats leaving it behind forever.
        import atexit
        atexit.register(lambda p=mm_path: _quiet_unlink(p))

    out = {key: mm[i] for i, key in enumerate(keys)}

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

    tick = lasio.Ticker(enabled=verbose)
    done = 0
    for tid in range(tiles.n_tiles):
        f = os.path.join(scratch_dir, f"k_{tid:06d}.bin")
        if written[tid] == 0 or not os.path.exists(f):
            continue
        rec = np.fromfile(f, dtype=_TILE_DTYPE)
        if len(rec) == 0:
            continue
        pts = rec["xyz"].astype(np.float64)
        owner = tiles.flat(tiles.cell_of(pts[:, :2] + origin[:2]))
        inside = owner == tid
        if not inside.any():
            del rec, pts
            continue

        tree = cKDTree(pts)
        q_pos = np.flatnonzero(inside)
        tile_idx = rec["idx"]

        kq = min(kmax + 1, len(pts))
        for start in range(0, len(q_pos), block):
            sel = q_pos[start:start + block]
            _, nb_all = tree.query(pts[sel], k=kq, workers=-1)
            if kq == 1:
                nb_all = nb_all[:, None]
            nb_all = nb_all[:, 1:] if kq > 1 else nb_all   # drop self
            own = tile_idx[sel]
            # Where this block's results go, and which of them are wanted.
            # With rows=None this is the identity and costs nothing.
            if row_of is None:
                dst, keep = own, slice(None)
            else:
                orow = row_of[own]
                wanted = orow >= 0
                if not wanted.any():
                    continue
                dst, keep = orow[wanted], wanted

            for k in ks:
                # prefix slice: neighbours come back sorted by distance, so the
                # first k of the k=max query ARE the k nearest.
                nb = nb_all[:, :min(k, nb_all.shape[1])]
                gidx = tile_idx[nb]
                for ch in channels:
                    v = values[ch]
                    nbv = v[gidx].astype(np.float32)
                    invalid = np.abs(nbv - SENTINEL) <= SENTINEL_TOL
                    good = ~invalid
                    n_good = good.sum(axis=1)
                    safe = np.where(good, nbv, 0.0)
                    mean = safe.sum(axis=1) / np.maximum(n_good, 1)
                    var = (np.where(good, (nbv - mean[:, None]) ** 2, 0.0).sum(axis=1)
                           / np.maximum(n_good, 1))
                    std = np.sqrt(np.maximum(var, 0.0))
                    ov = v[own]
                    usable = (n_good > 0) & (np.abs(ov - SENTINEL) > SENTINEL_TOL)
                    dev = np.where(usable, ov - mean, INVALID)
                    for st in stats:
                        key = f"{ch}__{st}_k{k}"
                        if st == "nmean":
                            out[key][dst] = np.where(n_good > 0, mean, INVALID)[keep]
                        elif st == "nstd":
                            out[key][dst] = np.where(n_good > 0, std, INVALID)[keep]
                        elif st == "dev":
                            out[key][dst] = dev[keep]
                        elif st == "z":
                            out[key][dst] = np.where(
                                usable, (ov - mean) / (std + EPS), INVALID)[keep]
                        elif st == "ninv":
                            out[key][dst] = invalid.mean(axis=1)[keep]
                    del nbv, invalid, good, safe
                del gidx, nb
            del nb_all

        done += int(inside.sum())
        del tree, rec, pts
        tick(f"    tile {tid + 1}/{tiles.n_tiles}: {done:,} / {total:,}")
    tick.done(f"    aggregated {done:,} / {total:,} points")

    mm.flush()
    if not keep_tiles:
        for tid in range(tiles.n_tiles):
            _quiet_unlink(os.path.join(scratch_dir, f"k_{tid:06d}.bin"))
        # The tile files were the only contents; drop the directory too rather
        # than leaving an empty dotted directory next to the output.
        try:
            os.rmdir(scratch_dir)
        except OSError:
            pass          # not empty: something else is in there, leave it
    return out
