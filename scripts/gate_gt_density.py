#!/usr/bin/env python3
"""Drop points from the evaluation domain where the reference is too sparse.

A C2C label answers "how far is this point from the reference surface?". Where
the reference samples that surface at spacing s, a point lying exactly on it
still reports a distance of order s/2 -- so once s approaches tau, the label
stops measuring the cloud and starts measuring the scan pattern. On Keble the
emptiest regions carry a median C2C of 9.4 cm and 69 % "outliers"; on
holzkirchen, 23.2 cm and 88 %. Those points are not bad.

This sets domain_code to OUT_NO_GT wherever the local reference density falls
below a threshold, so the evaluation domain covers only what the reference can
actually resolve. C2C itself is left alone -- like every other domain gate, the
point leaves the evaluation rather than carrying a fabricated value into it.

Density is counted per voxel rather than by a KD-tree: the references run to
500 M points, and only the relative number matters.

    python gate_gt_density.py --in feat.las --report labels.json --out gated.las
"""

import argparse
import json
import os
import sys

import laspy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlabel import domain as domain_mod        # noqa: E402


def gt_voxel_counts(gt_path, cell, stride, lo, hi):
    keys = counts = None
    span = np.floor((hi - lo) / cell).astype(np.int64) + 2
    with laspy.open(gt_path) as f:
        for chunk in f.chunk_iterator(20_000_000):
            p = np.column_stack([chunk.x, chunk.y, chunk.z])[::stride]
            m = np.all((p >= lo) & (p <= hi), axis=1)
            if not m.any():
                continue
            k = np.floor((p[m] - lo) / cell).astype(np.int64)
            flat = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
            u, c = np.unique(flat, return_counts=True)
            if keys is None:
                keys, counts = u, c
            else:
                keys = np.concatenate([keys, u])
                counts = np.concatenate([counts, c])
                keys, inv = np.unique(keys, return_inverse=True)
                counts = np.bincount(inv, weights=counts).astype(np.int64)
    return keys, counts, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default=None)
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="drop points below this percentile of the GT density that "
                         "in-domain points actually see (default 10)")
    ap.add_argument("--chunk", type=int, default=10_000_000)
    a = ap.parse_args()

    report = json.load(open(a.report))
    g = (report["groups"][0] if a.group is None
         else next(x for x in report["groups"] if x["name"] == a.group))
    same = report.get("output_frame") == g["name"]
    T = np.eye(4) if same else np.asarray(g["transform"], dtype=np.float64)
    gt_path = g["merged_gt_las"]
    print(f"  group {g['name']}  GT {os.path.basename(gt_path)}  "
          f"{'already in GT frame' if same else 'applying transform'}")

    with laspy.open(a.input) as f:
        hdr = f.header
        lo = np.array(hdr.mins, dtype=np.float64)
        hi = np.array(hdr.maxs, dtype=np.float64)
    corners = np.array([[x, y, z] for x in lo[[0]].tolist() + hi[[0]].tolist()
                        for y in lo[[1]].tolist() + hi[[1]].tolist()
                        for z in lo[[2]].tolist() + hi[[2]].tolist()])
    corners = corners @ T[:3, :3].T + T[:3, 3]
    glo, ghi = corners.min(axis=0) - 2.0, corners.max(axis=0) + 2.0

    print("  scanning the reference ...")
    keys, counts, span = gt_voxel_counts(gt_path, a.cell, a.stride, glo, ghi)
    # The percentile has to be taken over the density the SLAM points actually
    # experience, not over occupied voxels. The voxel distribution is dominated
    # by near-empty cells at cloud edges, so its 10th percentile is ~1 point --
    # a 22 cm spacing that gates almost nothing.
    def density_of(p_slam):
        k = np.floor((p_slam - glo) / a.cell).astype(np.int64)
        flat = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
        pos = np.clip(np.searchsorted(keys, flat), 0, len(keys) - 1)
        return np.where(keys[pos] == flat, counts[pos], 0)

    rng = np.random.default_rng(0)
    probe = []
    with laspy.open(a.input) as f:
        want = 3_000_000 / max(f.header.point_count, 1)
        for chunk in f.chunk_iterator(a.chunk):
            m = np.asarray(chunk["domain_code"]) == domain_mod.IN_DOMAIN
            if want < 1.0:
                m &= rng.random(len(chunk)) < want
            if m.any():
                probe.append(np.column_stack(
                    [chunk.x, chunk.y, chunk.z])[m] @ T[:3, :3].T + T[:3, 3])
    probe = np.concatenate(probe)
    n_probe = density_of(probe)
    thresh = np.percentile(n_probe[n_probe > 0], a.percentile)
    spacing = a.cell / np.sqrt(max(thresh * a.stride, 1.0))
    print(f"  {len(keys):,} occupied voxels; {len(probe):,} probe points")
    print(f"  p{a.percentile:.0f} of the density SLAM points see = "
          f"{thresh:.0f} per {a.cell} m voxel (~{spacing * 100:.1f} cm spacing)")
    del probe, n_probe

    dropped = kept = total = 0
    with laspy.open(a.input) as f, \
            laspy.open(a.out, mode="w", header=f.header) as w:
        for chunk in f.chunk_iterator(a.chunk):
            total += len(chunk)
            n_gt = density_of(np.column_stack([chunk.x, chunk.y, chunk.z])
                              @ T[:3, :3].T + T[:3, 3])

            code = np.asarray(chunk["domain_code"])
            sparse = (code == domain_mod.IN_DOMAIN) & (n_gt < thresh)
            if sparse.any():
                code = code.copy()
                code[sparse] = domain_mod.OUT_NO_GT
                chunk["domain_code"] = code
                # eval_domain is the field the evaluation actually filters on
                # (outlier_analysis.DOMAIN_FIELD); prepare_labels writes it as
                # code == IN_DOMAIN. Updating domain_code alone leaves the two
                # inconsistent and the gate silently does nothing.
                ed = np.asarray(chunk["eval_domain"]).copy()
                ed[sparse] = 0
                chunk["eval_domain"] = ed
                dropped += int(sparse.sum())
            kept += int((code == domain_mod.IN_DOMAIN).sum())
            w.write_points(chunk)
            print(f"\r    {total:,} written, {dropped:,} gated out",
                  end="", flush=True)
    print(f"\n  {dropped:,} points left the domain; {kept:,} remain")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
