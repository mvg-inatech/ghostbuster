#!/usr/bin/env python3
"""
Add multi-scale geometric channels to a labelled cloud.

These are the strong version of the conventional baseline — multi-scale SOR plus
radius-based roughness. They belong to the geometric bucket, not the
SLAM-internal one, so adding them makes the baseline harder to beat rather than
making the confidence channels look better.

Usage:
    python add_geom_features.py --in labeled.las --out labeled_geom.las
    python add_geom_features.py --in labeled.las --ks 6 30 100 --radii 0.05 0.15 0.40
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtlabel import geom, lasio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Add multi-scale kNN and roughness channels to a cloud.")
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", dest="output", default=None,
                    help="Default: overwrite the input (staged safely).")
    ap.add_argument("--ks", nargs="+", type=int, default=[6, 30, 100],
                    help="kNN scales (default: 6 30 100)")
    ap.add_argument("--radii", nargs="+", type=float, default=[0.05, 0.15, 0.40],
                    help="Roughness radii in metres (default: 0.05 0.15 0.40)")
    ap.add_argument("--k-rough", nargs="+", type=int, default=[],
                    help="Fixed-K roughness scales, e.g. 128. Kept for "
                         "comparison with the radius-based version; on a scan "
                         "with range-varying density a fixed K partly encodes "
                         "range rather than roughness.")
    ap.add_argument("--tile-budget", type=int, default=4_000_000)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out-npz",
                    help="write the features to this .npz instead of into the "
                         "LAS. Required with --rows/--sample.")
    ap.add_argument("--rows",
                    help=".npy of global point indices to compute for. The "
                         "neighbourhoods still come from the whole cloud; only "
                         "the query set shrinks.")
    ap.add_argument("--sample", type=int,
                    help="compute for this many uniformly sampled points "
                         "instead of all of them")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if lasio.is_pcd(args.input) and not args.output:
        sys.exit("ERROR: --out is required for a PCD input")
    out_path = args.output or args.input
    scratch = args.scratch or os.path.join(
        os.path.dirname(os.path.abspath(out_path)),
        f".{os.path.splitext(os.path.basename(args.input))[0]}_geom_tiles")

    t0 = time.time()
    # --rows / --sample / --out-npz: keep the features beside the cloud instead
    # of inside it. The filter comparison never reads more than a few million
    # rows per scene, so computing nine columns for every point of a 600 M-point
    # cloud stores -- and, during the stage-and-swap an in-place LAS write needs,
    # stores twice -- something that is cheaper to compute for the rows actually
    # used. The neighbourhoods are identical either way.
    rows = None
    total = lasio.point_count(args.input)
    if args.rows:
        rows = np.load(args.rows)
        print(f"  restricted to {len(rows):,} of {total:,} points "
              f"({100.0 * len(rows) / total:.2f} %)")
    elif args.sample and args.sample < total:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(total, args.sample, replace=False))
        print(f"  sampling {len(rows):,} of {total:,} points (seed {args.seed})")

    feats = geom.geometric_features(
        args.input, ks=args.ks, radii=args.radii, k_rough=args.k_rough,
        tile_budget=args.tile_budget, scratch_dir=scratch, rows=rows)

    if args.out_npz:
        np.savez_compressed(
            args.out_npz,
            rows=(rows if rows is not None
                  else np.arange(total, dtype=np.int64)),
            source=np.array(os.path.abspath(args.input)),
            **{k: np.asarray(v, dtype=np.float32) for k, v in feats.items()})
        print(f"Wrote {len(feats)} channels for "
              f"{len(rows) if rows is not None else total:,} rows -> "
              f"{args.out_npz}  "
              f"({os.path.getsize(args.out_npz) / 1e9:.2f} GB, "
              f"{time.time() - t0:.1f} s)")
        return

    if rows is not None:
        sys.exit("ERROR: --rows/--sample only make sense with --out-npz; a "
                 "partial column cannot be written back into the LAS")

    lasio.write_las_with_fields(
        args.input, out_path,
        {name: (arr, np.float32) for name, arr in feats.items()})
    print(f"Added {len(feats)} channels -> {out_path}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
