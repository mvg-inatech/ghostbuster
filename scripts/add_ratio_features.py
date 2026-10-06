#!/usr/bin/env python3
"""Add scale-free ratios of the geometric channels.

The multi-scale kNN and roughness channels each carry an absolute length, and
which of them is most diagnostic depends on the scene: on our indoor scenes
the best single scale is k30, on Keble it is k6. A model trained on the
warehouses therefore learns to weight k30 and arrives at a scene that wants k6,
which is how a learned combination ends up losing to one of its own inputs.

A ratio removes the scale from the feature, so the quantity the model keys on is
the same one in every scene -- the *shape* of the multi-scale profile rather
than its position.

  roughness_ratio_015_005  On a flat surface with range noise roughness barely
                         grows with radius, so the ratio sits near 1; on curved
                         or ornamented geometry it grows. Separates "noisy
                         plane" from "real structure".

                         r040/r015 was tried and dropped: as a standalone
                         scorer it ran BELOW chance on Keble (lift -0.072),
                         because roughness still growing at 40 cm means genuine
                         curvature, and curved structure is real geometry rather
                         than error. The signal is real but points the wrong
                         way, and it is already carried by r015/r005.

  mean_knn_dist_ratio_*  k6/k30 and k30/k100, each divided by the sqrt(k) law a
                         uniformly sampled surface obeys. 1.0 means "sampled
                         like a surface" at any density; above 1 means locally
                         isolated, which is what a flyer is. It measures what
                         SOR reaches for, without SOR's density dependence.

  roughness_over_knn_*   roughness in units of local point spacing, so a
                         deviation means the same thing in a dense cloud and a
                         sparse one.

Names begin with `roughness` or `mean_knn_dist` so compare_filters buckets them
as geometry, not as a new family: they are derived from family E and belong to
it. A sentinel in either operand propagates -- a ratio built on "no plane could
be fitted" is not a number.
"""

import argparse
import os
import sys

import laspy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlabel import lasio                      # noqa: E402
from gtlabel.geom import SENTINEL              # noqa: E402

TOL = 1e-6
EPS = 1e-9


def ratio(num, den, scale=1.0):
    """num/den * scale, propagating sentinels and guarding the denominator."""
    bad = (num <= SENTINEL + TOL) | (den <= SENTINEL + TOL) \
        | ~np.isfinite(num) | ~np.isfinite(den) | (np.abs(den) < EPS)
    out = np.full(len(num), SENTINEL, dtype=np.float32)
    ok = ~bad
    if ok.any():
        out[ok] = (num[ok] / den[ok] * scale).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--out", dest="output", required=True)
    ap.add_argument("--chunk", type=int, default=5_000_000)
    a = ap.parse_args()

    need = ["roughness_r005", "roughness_r015",
            "mean_knn_dist_k6", "mean_knn_dist_k30", "mean_knn_dist_k100"]
    with laspy.open(a.input) as f:
        have = set(f.header.point_format.dimension_names)
        n = f.header.point_count
    missing = [c for c in need if c not in have]
    if missing:
        sys.exit(f"ERROR: {a.input} lacks {', '.join(missing)} -- "
                 f"run add_geom_features.py first")
    print(f"  {n:,} points")

    # sqrt(k) is how mean kNN distance grows on a uniformly sampled surface, so
    # dividing it out makes 1.0 mean "sampled like a surface" at any density.
    s_6_30 = np.sqrt(30.0 / 6.0)
    s_30_100 = np.sqrt(100.0 / 30.0)

    made = {}

    def build(chunk):
        r5 = np.asarray(chunk["roughness_r005"], dtype=np.float64)
        r15 = np.asarray(chunk["roughness_r015"], dtype=np.float64)
        k6 = np.asarray(chunk["mean_knn_dist_k6"], dtype=np.float64)
        k30 = np.asarray(chunk["mean_knn_dist_k30"], dtype=np.float64)
        k100 = np.asarray(chunk["mean_knn_dist_k100"], dtype=np.float64)
        return {
            "roughness_ratio_015_005":      ratio(r15, r5),
            "mean_knn_dist_ratio_k6_k30":   ratio(k6, k30, s_6_30),
            "mean_knn_dist_ratio_k30_k100": ratio(k30, k100, s_30_100),
            "roughness_over_knn_005_k6":    ratio(r5, k6),
            "roughness_over_knn_015_k30":   ratio(r15, k30),
        }

    parts = {}
    with laspy.open(a.input) as f:
        done = 0
        for chunk in f.chunk_iterator(a.chunk):
            for k, v in build(chunk).items():
                parts.setdefault(k, []).append(v)
            done += len(chunk)
            print(f"\r    computed {done:,} / {n:,}", end="", flush=True)
    print()
    for k, v in parts.items():
        made[k] = np.concatenate(v)
    del parts

    for k, v in sorted(made.items()):
        valid = v > SENTINEL + TOL
        if valid.any():
            q = np.percentile(v[valid], [50, 90])
            print(f"    {k:30s} {100 * (~valid).mean():5.2f} % sentinel   "
                  f"median {q[0]:7.3f}  p90 {q[1]:7.3f}")

    fields = {k: (v, np.float32) for k, v in made.items()}
    lasio.write_las_with_fields(a.input, a.output, fields, chunk_size=a.chunk)
    print(f"  wrote {a.output}")


if __name__ == "__main__":
    main()
