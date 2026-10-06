#!/usr/bin/env python3
"""Score a labelled cloud with a saved model and write the prediction back.

The model bundles written by compare_filters --save-models were fitted on
features rank-normalised *within each scene*, so a raw-channel prediction would
be silently wrong. This applies the recorded recipe to the target cloud before
predicting, which is the same transductive treatment SOR gets: the normalisation
uses this cloud's own distribution, never stored statistics.

Adds, for each model, two columns:
    pred_<name>     the model's score, higher = more suspicious
    keep_<name>     1 if the point survives at --removed, else 0

With a single --model the legacy names `predicted_c2c` / `filter_keep` are kept
so existing consumers do not break. Several --model arguments are scored in one
pass over the cloud, which matters: the input is tens of GB and writing one copy
per model would multiply that.

`--denormalise` multiplies the score back by this cloud's own median C2C, so the
channel reads in metres instead of "multiples of the scene median". That uses the
held-out scene's labels, which is fine for inspection but is NOT how the filter
is evaluated -- the ranking, and therefore every metric, is unchanged by it.

Usage:
    python predict_cloud.py --model .../models/slam_plus_q90.joblib \
        --in .../global_map_feat.las --out .../scored.las --removed 0.20
"""
import argparse
import os
import sys

import joblib
import laspy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlabel import lasio  # noqa: E402

SENTINEL, SENTINEL_TOL = -1.0, 1e-6
N_EDGES = 1001


def quantile_edges(path, features, sample_target=2_000_000, verbose=True,
                   chunk_size=2_000_000, sentinel=frozenset()):
    """Per-feature quantile breakpoints from a streamed subsample."""
    total = lasio.point_count(path)
    stride = max(1, total // max(sample_target, 1))
    cols = {f: [] for f in features}
    with laspy.open(path) as fh:
        for chunk in fh.chunk_iterator(chunk_size):
            for f in features:
                cols[f].append(np.asarray(chunk[f])[::stride].astype(np.float32))
    qs = np.linspace(0.0, 1.0, N_EDGES)
    edges = {}
    for f in features:
        v = np.concatenate(cols[f])
        good = np.isfinite(v)
        if f in sentinel:
            good &= v > SENTINEL_TOL - 1.0
        if not good.any():
            edges[f] = None
            continue
        e = np.quantile(v[good], qs)
        e = np.maximum.accumulate(e)
        edges[f] = None if e[-1] <= e[0] else (e, qs)
    if verbose:
        n_flat = sum(1 for f in features if edges[f] is None)
        print(f"  quantile edges from {total // stride:,} sampled points"
              + (f"  ({n_flat} constant channels left raw)" if n_flat else ""))
    return edges


def main():
    ap = argparse.ArgumentParser(description="Score a cloud with a saved model.")
    ap.add_argument("--model", required=True, action="append",
                    help="a .joblib from --save-models; repeat to score "
                         "several models in a single pass over the cloud")
    ap.add_argument("--name", action="append", default=None,
                    help="Channel suffix for the matching --model "
                         "(default: the joblib stem minus any _q90/_mean).")
    ap.add_argument("--denormalise", action="store_true",
                    help="Scale scores back to metres using this cloud's "
                         "median C2C_distance over the eval domain.")
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", dest="output", default=None,
                    help="Default: overwrite the input (staged safely).")
    ap.add_argument("--removed", type=float, default=0.20,
                    help="Removal fraction for the filter_keep flag.")
    ap.add_argument("--chunk", type=int, default=2_000_000,
                    help="Points per scoring chunk.")
    ap.add_argument("--drop-aggregates", action="store_true",
                    help="Leave the aggregated channels (any name containing "
                         "'__') out of the output. They are pipeline "
                         "scaffolding: the models need them to predict, the "
                         "result does not need to carry them, and on these "
                         "clouds they are ~75%% of the file.")
    ap.add_argument("--drop-channels", default=None,
                    help="Comma-separated extra channels to leave out of the "
                         "output, beyond --drop-aggregates.")
    ap.add_argument("--write-chunk", type=int, default=2_000_000,
                    help="Points per output chunk. lasio defaults to 20M, "
                         "which on a ~400 byte/point cloud holds an 8 GB "
                         "reader chunk and an 8 GB output record at once and "
                         "will exhaust a 64 GB machine. Keep this small.")
    args = ap.parse_args()

    bundles = [joblib.load(m) for m in args.model]
    if args.name and len(args.name) != len(args.model):
        sys.exit("ERROR: --name must be given once per --model")
    names = args.name or [os.path.basename(m).replace(".joblib", "")
                          for m in args.model]

    for m, b, nm in zip(args.model, bundles, names):
        print(f"Model    : {os.path.basename(m)}  -> pred_{nm}")
        print(f"  trained on {b['training']['scenes']}, "
              f"held out {b['training']['held_out']}")
        print(f"  {len(b['features'])} features, recipe {b['recipe']}")

    modes = {b["recipe"].get("normalise_per_scene", "none") for b in bundles}
    if modes - {"none", "rank"}:
        sys.exit(f"ERROR: recipe normalisation {modes} not supported here")
    if len(modes) > 1:
        sys.exit("ERROR: mixing normalisation recipes in one pass is unsafe")
    mode = modes.pop()

    # One quantile table shared by every model: the transform is per-channel and
    # per-scene, so it does not depend on which model consumes the column.
    features = sorted({f for b in bundles for f in b["features"]})
    # compare_filters marks "no plane could be fitted" (value -1) two ways at
    # training time: the value is blanked to NaN before the per-scene ranking,
    # and a raw 0/1 column <name>__nofit is added after it. It does both only
    # for channels that carry the marker, which are exactly the channels the
    # model has a __nofit column for -- so that set drives both steps here too.
    # The __nofit columns are built, not read: they are not stored in the cloud.
    nofit = [f for f in features if f.endswith("__nofit")]
    sentinel = frozenset(f[:-len("__nofit")] for f in nofit)
    features = sorted(set(f for f in features if not f.endswith("__nofit")) | sentinel)
    with laspy.open(args.input) as fh:
        available = set(fh.header.point_format.dimension_names)
        total = fh.header.point_count
    missing = [f for f in features if f not in available]
    if missing:
        sys.exit(f"ERROR: {args.input} lacks {len(missing)} channels the models "
                 f"need, first few: {missing[:5]}")

    edges = (quantile_edges(args.input, features, chunk_size=args.chunk,
                            sentinel=sentinel)
             if mode == "rank" else None)

    preds = {nm: np.empty(total, dtype=np.float32) for nm in names}
    at = 0
    tick = lasio.Ticker()
    with laspy.open(args.input) as fh:
        for chunk in fh.chunk_iterator(args.chunk):
            n = len(chunk)
            cols = {}
            for f in features:
                v = np.asarray(chunk[f]).astype(np.float32)
                if f in sentinel:
                    bad = v <= SENTINEL_TOL - 1.0
                    cols[f + "__nofit"] = bad.astype(np.float32)   # raw 0/1, not ranked
                    v[bad] = np.nan
                if edges is not None and edges[f] is not None:
                    e, qs = edges[f]
                    good = np.isfinite(v)
                    v[good] = np.interp(v[good], e, qs).astype(np.float32)
                cols[f] = v
            for b, nm in zip(bundles, names):
                X = np.empty((n, len(b["features"])), dtype=np.float32)
                for j, f in enumerate(b["features"]):
                    X[:, j] = cols[f]
                preds[nm][at:at + n] = b["model"].predict(X).astype(np.float32)
            at += n
            tick(f"    scored {at:,} / {total:,}")
            del cols
    tick.done(f"    scored {at:,} points for {len(names)} model(s)")

    scale = 1.0
    if args.denormalise:
        with laspy.open(args.input) as fh:
            c2c_s, dom_s = [], []
            for chunk in fh.chunk_iterator(args.chunk):
                c = np.asarray(chunk["C2C_distance"]).astype(np.float32)
                d = np.asarray(chunk["eval_domain"])
                m = (d == 1) & np.isfinite(c) & (c >= 0)
                c2c_s.append(c[m][::17])
        scale = float(np.median(np.concatenate(c2c_s)))
        print(f"  denormalise: x median C2C over eval domain = {scale:.5f} m")

    fields = {}
    for nm in list(preds):
        p = preds.pop(nm)          # hand the array over; never hold two copies
        if scale != 1.0:
            p *= scale
        keep = np.ones(total, dtype=np.uint8)
        if 0.0 < args.removed < 1.0:
            cut = np.quantile(p, 1.0 - args.removed)
            keep[p > cut] = 0
            print(f"  {nm:16s} keep at {args.removed * 100:.0f}% removed: "
                  f"threshold {cut:.5f}, {int((keep == 0).sum()):,} flagged")
        fields[f"pred_{nm}"] = (p, np.float32)
        fields[f"keep_{nm}"] = (keep, np.uint8)
        del p, keep
    if len(names) == 1:
        # keep the original column names for single-model callers
        nm = names[0]
        fields["predicted_c2c"] = fields.pop(f"pred_{nm}")
        fields["filter_keep"] = fields.pop(f"keep_{nm}")

    out = args.output or args.input
    drop = set()
    if args.drop_aggregates:
        drop |= {d for d in available if "__" in d}
    if args.drop_channels:
        drop |= {c.strip() for c in args.drop_channels.split(",") if c.strip()}
    drop -= set(fields)            # never drop what we are writing
    if drop:
        print(f"  dropping {len(drop)} channel(s) from the output")
    lasio.write_las_with_fields(args.input, out, fields,
                                chunk_size=args.write_chunk, drop=sorted(drop))
    print(f"Wrote {out}  ({len(fields)} new channels, "
          f"{len(drop)} dropped)")


if __name__ == "__main__":
    main()
