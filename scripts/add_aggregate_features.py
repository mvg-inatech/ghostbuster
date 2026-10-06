#!/usr/bin/env python3
"""
Add locally aggregated SLAM-channel features to a labelled cloud.

For each SLAM channel and each point, how far that point deviates from its kNN
neighbourhood on that channel. See gtlabel/aggregate.py for why the deviation,
not the neighbourhood mean, is the feature that carries information.

Usage:
    python add_aggregate_features.py --in labeled.las --out labeled_agg.las
    python add_aggregate_features.py --in labeled.las --ks 30 100 --stats dev z nstd
"""

import argparse
import os
import sys
import time

import numpy as np
import laspy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtlabel import aggregate, aggregate_fast, lasio  # noqa: E402

# Never aggregate these: pipeline outputs, bookkeeping, or geometry (which
# already is a neighbourhood statistic — aggregating it again says little).
# Identifiers and pipeline outputs: averaging an ID over a neighbourhood is
# meaningless. cell_id replaced voxel_id in the SLAM; both listed so clouds from
# either side of that change behave.
SKIP = {"C2C_distance", "domain_code", "eval_domain", "visibility_code",
        "gt_group", "predicted_c2c", "voxel_id", "cell_id"}
SKIP_PREFIX = ("mean_knn_dist", "roughness")


def default_channels(path):
    with laspy.open(path) as f:
        extra = list(f.header.point_format.extra_dimension_names)
    return [d for d in extra
            if d not in SKIP and not d.startswith(SKIP_PREFIX) and "__" not in d]


def main():
    ap = argparse.ArgumentParser(
        description="Add kNN-aggregated SLAM channel features to a cloud.")
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", dest="output", default=None,
                    help="Default: overwrite the input (staged safely).")
    ap.add_argument("--channels", nargs="+", default=None,
                    help="Channels to aggregate (default: every SLAM channel).")
    ap.add_argument("--ks", nargs="+", type=int, default=[30],
                    help="Neighbourhood sizes (default: 30). Matching the "
                         "geometric scales keeps the buckets comparable.")
    ap.add_argument("--stats", nargs="+", default=list(aggregate.DEFAULT_STATS),
                    choices=list(aggregate.ALL_STATS),
                    help="Which statistics to emit (default: dev z).")
    ap.add_argument("--tile-budget", type=int, default=4_000_000)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--slow", action="store_true",
                    help="Use the original in-RAM implementation. The default "
                         "path writes results to a memmap and issues ONE "
                         "neighbour query at max(k) instead of one per k, which "
                         "lets every channel go in a single pass. Measured on "
                         "a 75 M-point cloud with 16 channels: 4 batches "
                         "took 80 min against ~27 min predicted for one pass. "
                         "Results are identical except where an exact distance "
                         "tie at the k-th neighbour is broken differently "
                         "(0.09%% of points).")
    ap.add_argument("--out-npz",
                    help="write the aggregates to this .npz instead of into "
                         "the LAS. Required with --rows/--sample.")
    ap.add_argument("--rows",
                    help=".npy of global point indices to compute for; the "
                         "neighbourhoods still come from the whole cloud")
    ap.add_argument("--sample", type=int,
                    help="compute for this many uniformly sampled points "
                         "instead of all of them")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if lasio.is_pcd(args.input) and not args.output:
        sys.exit("ERROR: --out is required for a PCD input")
    out_path = args.output or args.input
    channels = args.channels or default_channels(args.input)
    if not channels:
        sys.exit("ERROR: no SLAM channels found to aggregate")
    scratch = args.scratch or os.path.join(
        os.path.dirname(os.path.abspath(out_path)),
        f".{os.path.splitext(os.path.basename(args.input))[0]}_agg_tiles")

    t0 = time.time()

    # --rows / --out-npz: compute the aggregates only for a subset of points and
    # keep them beside the cloud instead of inside it.
    #
    # These columns are a deterministic function of the cloud, and nothing reads
    # all of them at once -- the filter comparison fits on a few hundred thousand
    # rows and evaluates on a few million. Writing 96 float32 columns for every
    # point of a large cloud costs hundreds of gigabytes to store something
    # cheaper to recompute, and it doubles again during the stage-and-swap that
    # an in-place LAS write requires. The neighbourhoods are unchanged either
    # way: every tile is still built from every point.
    rows = None
    if args.rows:
        rows = np.load(args.rows)
        print(f"  restricted to {len(rows):,} of "
              f"{lasio.point_count(args.input):,} points "
              f"({100.0 * len(rows) / lasio.point_count(args.input):.2f} %)")
    elif args.sample:
        n = lasio.point_count(args.input)
        if args.sample < n:
            rng = np.random.default_rng(args.seed)
            rows = np.sort(rng.choice(n, args.sample, replace=False))
            print(f"  sampling {len(rows):,} of {n:,} points "
                  f"(seed {args.seed})")

    if args.slow:
        if rows is not None:
            sys.exit("ERROR: --slow does not support --rows/--sample")
        feats = aggregate.aggregate_features(
            args.input, channels, ks=args.ks, stats=tuple(args.stats),
            tile_budget=args.tile_budget, scratch_dir=scratch)
    else:
        feats = aggregate_fast.aggregate_features_fast(
            args.input, channels, ks=args.ks, stats=tuple(args.stats),
            tile_budget=args.tile_budget, scratch_dir=scratch, rows=rows)

    if args.out_npz:
        np.savez_compressed(
            args.out_npz,
            rows=(rows if rows is not None
                  else np.arange(lasio.point_count(args.input), dtype=np.int64)),
            source=np.array(os.path.abspath(args.input)),
            **{k: np.asarray(v, dtype=np.float32) for k, v in feats.items()})
        size = os.path.getsize(args.out_npz) / 1e9
        print(f"Wrote {len(feats)} channels for "
              f"{len(rows) if rows is not None else 'all'} rows -> "
              f"{args.out_npz}  ({size:.2f} GB, {time.time() - t0:.1f} s)")
        return

    if rows is not None:
        sys.exit("ERROR: --rows/--sample only make sense with --out-npz; "
                 "a partial column cannot be written back into the LAS")

    lasio.write_las_with_fields(
        args.input, out_path,
        {name: (arr, np.float32) for name, arr in feats.items()})
    print(f"Added {len(feats)} channels -> {out_path}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
