#!/usr/bin/env python3
"""Re-label a SLAM cloud with one rigid registration per trajectory stretch.

A single rigid transform per GT group cannot remove trajectory drift, so on a
long session every label also carries the drift at the moment the point was
captured. Drift is out of scope for a per-point filter -- and so are the double
walls it causes -- so the labels should not contain it.

This keeps the cloud exactly as the SLAM produced it: no point moves, no channel
changes. Only the frame each label is computed in changes. The session is cut
into stretches of consecutive scans, each stretch is registered to the reference
on its own, starting from the group transform the labelling pipeline already
found, and every point is labelled in its stretch's frame.

Stretches are cut in time, not in space: two passes over the same wall are two
stretches with two transforms, so neither pass is labelled an outlier merely for
disagreeing with the other.

Each point's scan is recovered exactly from the two per-scan channels the
converter writes (pose_unc_rot, ang_rate), whose float32 pair is unique per
trajectory row. The evaluation domain (visibility) is left untouched: it comes
from the stations, not from the stretches.

Validation built in: in the same GT pass, a random sample of in-domain points is
re-labelled under the ORIGINAL group transform and compared against the stored
label. If the two do not agree, nothing written by this script can be trusted.

    python stretch_labels.py --cloud feat.las --traj alidarState.txt \
        --labels global_map_labels.json --config configs/x.yaml \
        --stretch-scans 300 --out out.las
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import laspy
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "VoxelSLAM", "scripts"))
from gtlabel import register, lasio                             # noqa: E402
from gtlabel import c2c as c2c_mod                              # noqa: E402
from gtlabel.visibility import VIS_OBSERVED                     # noqa: E402
from outlier_analysis import C2C_FIELD, DOMAIN_FIELD            # noqa: E402
import global_pcd_converter as gpc                              # noqa: E402


def fingerprint(pu, ar):
    pu = np.asarray(pu, np.float32)
    ar = np.asarray(ar, np.float32)
    return (pu.view(np.uint32).astype(np.uint64) << np.uint64(32)) | \
        ar.view(np.uint32).astype(np.uint64)


def apply(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def delta(Ta, Tb, xyz):
    """How far the stretch's own points move between the two transforms (p95),
    and the rotation angle between them.

    Measured on the points, not on the bounding box. The first version used
    the box corners, which in a large scene sit tens of metres from any point:
    a 0.5 deg rotation moves them half a metre even where there is nothing to
    label, and good fits (fitness 0.994) were rejected for it. What matters is
    how much the stretch's labels change, and that is where its points are."""
    shift = np.percentile(np.linalg.norm(apply(Ta, xyz) - apply(Tb, xyz), axis=1), 95)
    R = Ta[:3, :3].T @ Tb[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(shift), float(ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True, help="labelled (featured) LAS")
    ap.add_argument("--traj", required=True,
                    help="the alidarState.txt the cloud was merged from")
    ap.add_argument("--labels", required=True, help="global_map_labels.json")
    ap.add_argument("--config", required=True, help="the labelling config")
    ap.add_argument("--stretch-scans", type=int, default=None,
                    help="cut stretches every N scans")
    ap.add_argument("--stretch-metres", type=float, default=None,
                    help="cut stretches every M metres of walked path. Drift is "
                         "quoted per distance travelled, and scenes walked at "
                         "different speeds get pieces of comparable extent "
                         "(a fixed time is 10 m in one scene and 20 m in another)")
    ap.add_argument("--min-points", type=int, default=20_000,
                    help="observed points needed to register a stretch; fewer "
                         "and it keeps the group transform")
    ap.add_argument("--max-shift", type=float, default=0.5,
                    help="guard: a stretch transform that moves the stretch's "
                         "points by more than this [m] (p95) from the group "
                         "transform is rejected as a failed fit")
    ap.add_argument("--max-rot", type=float, default=2.0,
                    help="guard, in degrees")
    ap.add_argument("--n-validate", type=int, default=1_000_000)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--from-npz", default=None,
                    help="skip registration and C2C; write the labels saved by "
                         "an earlier run (the *_labels.npz next to --out)")
    a = ap.parse_args()
    if (a.stretch_scans is None) == (a.stretch_metres is None) and not a.from_npz:
        sys.exit("ERROR: pass exactly one of --stretch-scans / --stretch-metres")
    if a.from_npz:
        z = np.load(a.from_npz)
        write_out(a.cloud, a.out, z["c2c_new"], z["c2c_old"], z["stretch"])
        return
    t_start = time.time()
    rng = np.random.default_rng(0)
    os.makedirs(a.scratch, exist_ok=True)

    cfg = yaml.safe_load(open(a.config))
    lab = json.load(open(a.labels))
    groups = {int(g["index"]): g for g in lab["groups"]}
    # The group transforms map the SLAM frame to each group's GT frame. A cloud
    # written in 'original' is already in the SLAM frame; one written in a
    # group's frame (output_frame = that group's name) is mapped back first.
    # Getting this wrong cannot slip through: the validation below re-labels
    # under the original transform and must reproduce the stored labels.
    frame = lab.get("output_frame")
    if frame == "original":
        T_frame = np.eye(4)
    else:
        hit = [g for g in lab["groups"] if g["name"] == frame]
        if len(hit) != 1:
            sys.exit(f"ERROR: output_frame '{frame}' is neither 'original' nor "
                     "the name of exactly one group")
        T_frame = np.asarray(hit[0]["transform"], np.float64)
    print(f"  cloud frame: {frame}")
    max_dist = float(cfg["c2c_max_dist"])
    levels = [tuple(l) for l in cfg["icp"]["levels"]][1:]   # start is ~cm off
    finest = min(l[0] for l in levels)
    print(f"  ICP levels {levels}, C2C cap {max_dist} m")

    # ---- scan fingerprint -> trajectory row -------------------------------
    poses, ts, ch, _ = gpc.read_trajectory(a.traj)
    poses = np.asarray(poses); ts = np.asarray(ts, float)
    key = fingerprint(ch["pose_unc_rot"], ch["ang_rate"])
    if len(np.unique(key)) != len(key):
        sys.exit("ERROR: per-scan fingerprint is not unique in this trajectory")
    order = np.argsort(key)
    skey = key[order]

    # ---- read the cloud ---------------------------------------------------
    print("  reading the cloud ...")
    cols = {k: [] for k in ("xyz", "grp", "vis", "dom", "c2c", "row")}
    with laspy.open(a.cloud) as f:
        n = f.header.point_count
        for c in f.chunk_iterator(10_000_000):
            cols["xyz"].append(np.c_[np.asarray(c.x), np.asarray(c.y),
                                     np.asarray(c.z)])
            cols["grp"].append(np.asarray(c["gt_group"]).astype(np.int16))
            cols["vis"].append(np.asarray(c["visibility_code"]).astype(np.int8))
            cols["dom"].append(np.asarray(c[DOMAIN_FIELD]) != 0)
            cols["c2c"].append(np.asarray(c[C2C_FIELD]).astype(np.float64))
            k = fingerprint(c["pose_unc_rot"], c["ang_rate"])
            i = np.searchsorted(skey, k).clip(0, len(skey) - 1)
            if not np.all(skey[i] == k):
                sys.exit("ERROR: a point's scan fingerprint is not in the "
                         "trajectory -- wrong --traj for this cloud?")
            cols["row"].append(order[i])
    xyz = np.concatenate(cols["xyz"]); grp = np.concatenate(cols["grp"])
    if frame != "original":
        xyz = apply(np.linalg.inv(T_frame), xyz)
    vis = np.concatenate(cols["vis"]); dom = np.concatenate(cols["dom"])
    c2c_old = np.concatenate(cols["c2c"]); row = np.concatenate(cols["row"])
    del cols
    if a.stretch_metres:
        pos = poses[:, :3, 3]
        cum = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        row_stretch = np.floor(cum / a.stretch_metres).astype(np.int32)
        unit = f"{a.stretch_metres:g} m"
    else:
        row_stretch = (np.arange(len(poses)) // a.stretch_scans).astype(np.int32)
        unit = f"{a.stretch_scans} scans"
    stretch = row_stretch[row]
    # how long each piece is in time -- a piece where the walker stood still or
    # turned on the spot covers little path but can span a long time
    ids = np.unique(row_stretch)
    secs = np.array([np.ptp(ts[row_stretch == i]) for i in ids])
    print(f"  stretches of {unit}: {len(ids)} over the session, duration "
          f"min {secs.min():.0f} s, median {np.median(secs):.0f} s, "
          f"max {secs.max():.0f} s")
    print(f"  {n:,} points, {len(np.unique(stretch))} stretches, "
          f"groups {sorted(set(np.unique(grp).tolist()))}")

    c2c_new = c2c_old.copy()
    report = {"stretch_scans": a.stretch_scans,
              "stretch_metres": a.stretch_metres, "cloud_frame": frame,
              "stretch_seconds": dict(min=float(secs.min()),
                                      median=float(np.median(secs)),
                                      max=float(secs.max())),
              "groups": {}}

    for g in sorted(set(np.unique(grp).tolist())):
        if g not in groups:
            print(f"\n  group {g}: not in labels.json (unassigned), left as is")
            continue
        G = groups[g]
        Tg = np.asarray(G["transform"], np.float64)
        gm = grp == g
        print(f"\n=== group {g} {G['name']}: {gm.sum():,} points, "
              f"{(gm & dom).sum():,} in domain")
        gt_icp = register.load_gt_for_icp(G["merged_gt_las"], finest)

        Ts, rows_rep = {}, []
        for s in np.unique(stretch[gm]):
            ms = gm & (stretch == s)
            # observed AND in the domain: with a cropped reference, points on
            # cropped-away surfaces still count as observed (the panoramas come
            # from the uncropped cloud) but have nothing to register against
            obs = ms & (vis == VIS_OBSERVED) & dom
            info = dict(stretch=int(s), points=int(ms.sum()),
                        observed=int(obs.sum()))
            T = Tg
            if obs.sum() >= a.min_points:
                src = xyz[obs]
                Tc, ri = register.register(
                    src, gt_icp, init=Tg, levels=levels,
                    max_iter=int(cfg["icp"]["max_iter"]),
                    tukey_k=float(cfg["icp"]["tukey_k"]), verbose=False)
                sh, ang = delta(Tg, Tc, src)
                info.update(shift_m=sh, rot_deg=ang, used=ri["used"],
                            fitness=ri["fitness"], rmse=ri["inlier_rmse"])
                if sh > a.max_shift or ang > a.max_rot:
                    info["used"] = "rejected"
                else:
                    T = Tc
            else:
                info["used"] = "too_few_points"
            Ts[int(s)] = T
            rows_rep.append(info)
        used = [r["used"] for r in rows_rep]
        moved = np.array([r.get("shift_m", 0.0) for r in rows_rep
                          if r["used"] == "icp"])
        print(f"  stretches: {len(rows_rep)}  icp {used.count('icp')}  "
              f"kept-group-transform {used.count('initial')}  "
              f"rejected {used.count('rejected')}  "
              f"too-few {used.count('too_few_points')}")
        if len(moved):
            print(f"  stretch vs group transform, point shift p95: median "
                  f"{np.median(moved) * 100:.2f} cm, p95 "
                  f"{np.percentile(moved, 95) * 100:.2f} cm, max "
                  f"{moved.max() * 100:.2f} cm")

        # one GT pass: every in-domain point in its stretch frame, plus a
        # validation sample in the ORIGINAL group frame
        idx = np.flatnonzero(gm & dom)
        pts = np.empty((len(idx), 3))
        for s, T in Ts.items():
            sel = stretch[idx] == s
            pts[sel] = apply(T, xyz[idx[sel]])
        val = rng.choice(idx, min(a.n_validate, len(idx)), replace=False)
        pts_all = np.vstack([pts, apply(Tg, xyz[val])])
        d = c2c_mod.compute_c2c(
            pts_all, G["merged_gt_las"], max_dist=max_dist,
            tile_size=cfg.get("tile_size"),
            tile_budget=int(cfg.get("tile_budget", 8_000_000)),
            scratch_dir=os.path.join(a.scratch, f"tiles_g{g}"))
        d_new, d_val = d[:len(idx)], d[len(idx):]

        # validation: the script must reproduce the stored label
        old_v = c2c_old[val]
        both = np.isfinite(d_val) & (old_v < max_dist - 1e-6)
        err = np.abs(d_val[both] - old_v[both])
        v = dict(n=int(both.sum()), max_mm=float(err.max() * 1000),
                 p99_mm=float(np.percentile(err, 99) * 1000))
        print(f"  VALIDATION vs stored label ({v['n']:,} pts): "
              f"p99 {v['p99_mm']:.3f} mm, max {v['max_mm']:.3f} mm")

        lost = ~np.isfinite(d_new)
        d_new = np.where(lost, max_dist, d_new)     # same clamp as the pipeline
        c2c_new[idx] = d_new
        o, nw = c2c_old[idx] * 100, d_new * 100
        print(f"  in-domain C2C  median {np.median(o):.3f} -> {np.median(nw):.3f} cm"
              f"   <2cm {100*(o<2).mean():.2f} -> {100*(nw<2).mean():.2f} %"
              f"   <5cm {100*(o<5).mean():.2f} -> {100*(nw<5).mean():.2f} %"
              f"   p99 {np.percentile(o,99):.2f} -> {np.percentile(nw,99):.2f} cm")
        print(f"  in-domain points that lost their GT neighbour (clamped): "
              f"{int(lost.sum()):,}")
        report["groups"][G["name"]] = dict(
            stretches=rows_rep, validation=v, clamped=int(lost.sum()),
            median_cm=[float(np.median(o)), float(np.median(nw))],
            lt2_pct=[float(100*(o<2).mean()), float(100*(nw<2).mean())],
            lt5_pct=[float(100*(o<5).mean()), float(100*(nw<5).mean())],
            p99_cm=[float(np.percentile(o, 99)), float(np.percentile(nw, 99))])
        if v["max_mm"] > 2.0:
            sys.exit("ERROR: validation failed -- this script does not "
                     "reproduce the stored labels, so its new ones cannot be "
                     "trusted. Nothing written.")

    # Save everything computed BEFORE the write. The write streams the whole
    # source cloud and is the step most likely to be killed on a shared machine;
    # the first run lost 40 minutes of ICP and C2C that way, with no traceback.
    # With these saved, a failed write is retried with --from-npz in minutes.
    stem = os.path.splitext(a.out)[0]
    rep = a.report or stem + "_stretch_report.json"
    json.dump(report, open(rep, "w"), indent=1)
    npz = stem + "_labels.npz"
    np.savez(npz, c2c_new=c2c_new, c2c_old=c2c_old,
             stretch=stretch.astype(np.uint16))
    print(f"\n  report {rep}\n  labels {npz}")
    del xyz, grp, vis, dom, row            # the write does not need them
    write_out(a.cloud, a.out, c2c_new, c2c_old, stretch)
    print(f"  {time.time() - t_start:.0f} s")


def write_out(cloud, out, c2c_new, c2c_old, stretch, chunk=2_000_000):
    print("\n  writing ...")
    # No drop=: write_las_with_fields copies the existing C2C column and then
    # overwrites it with the values given here. Dropping it and re-adding the
    # same name does NOT work -- the helper skips any name the source already
    # had, so the column would silently vanish from the output.
    with laspy.open(cloud) as f:
        c2c_dtype = np.dtype(f.header.point_format.dimension_by_name(C2C_FIELD).dtype)
    # Small chunks: each chunk is materialised as a full point record over every
    # dimension of the cloud, which is the write's peak memory.
    lasio.write_las_with_fields(
        cloud, out,
        {C2C_FIELD: (c2c_new.astype(c2c_dtype), c2c_dtype),
         "c2c_rigid": (c2c_old.astype(np.float32), np.float32),
         "stretch_id": (np.asarray(stretch).astype(np.uint16), np.uint16)},
        chunk_size=chunk)
    print(f"  wrote {out}")

if __name__ == "__main__":
    main()
