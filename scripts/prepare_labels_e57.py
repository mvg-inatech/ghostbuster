#!/usr/bin/env python3
"""
Label a SLAM cloud against a Leica RTC360 project exported as separate E57
stations, using the scanner's own visibility instead of a proximity heuristic.

What it does, and why it is built this way:

  * No manual pre-registration. The coarse transform is found automatically by
    a 4-DoF search (yaw + translation) on top-down structure rasters, so the
    only thing you supply is the two clouds.
  * The evaluation domain comes from a per-station visibility model rather than
    a coverage radius. It answers "could any scanner have seen this spot", which
    a proximity test cannot: it separates a genuine SLAM error 15 cm off a
    scanned wall from a point 15 cm behind it in an unscanned room.
  * The scanner's blind cone below the tripod — the unscanned disc on the
    ground under each setup — is handled for free, because those directions
    simply hold no returns.
  * Everything streams. A cloud of a few hundred million points is never held
    in memory.

Usage:
    python prepare_labels_e57.py --config configs/barn.yaml
    python prepare_labels_e57.py --config configs/barn.yaml --rounds 1
    python prepare_labels_e57.py --config configs/barn.yaml --skip-c2c
"""

import argparse
import json
import os
import math
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtlabel import (lasio, register, coarse, e57, visibility as vis,  # noqa: E402
                     c2c as c2c_mod, domain as domain_mod,              # noqa: E402
                     knn as knn_mod, qa)                                # noqa: E402

C2C_SENTINEL = -1.0

DEFAULTS = {
    "rounds": 2,
    # Each group is one RTC project (or any subset of stations) with its own
    # coordinate frame and its own registration to the SLAM cloud.
    "gt_groups": None,
    "output_frame": None,
    "tau": 0.05,
    "c2c_max_dist": 0.50,
    "drop_cropped_away": False,
    "cropped_away_margin": 0.01,
    "coarse": {
        "enabled": True,
        # >1 tests that many distinct yaw peaks with a short ICP and keeps the
        # one that actually fits, rather than trusting the correlation score.
        "n_candidates": 1,
        "cell": 0.30,
        "band": [1.0, 8.0],
        "yaw_step": 2.0,
        "refine_step": 0.25,
        # Percentile clip on each side of the SLAM extent before rastering, so a
        # thin spray of flyers cannot set the grid size. 0 disables it.
        "slam_percentile": 0.1,
        # Gravity-aligned does not mean gravity-signed: test a 180 deg roll too.
        "try_flip": True,
    },
    "visibility": {
        "resolution": 0.05,
        "eps0": 0.02,
        "eps_rate": 0.002,
        "behind_tolerance": 0.10,
    },
    "icp": {
        "levels": [[0.40, 1.20], [0.20, 0.60], [0.10, 0.25]],
        "max_iter": 60,
        "tukey_k": 0.10,
    },
    # Section 3.4: do not hunt for the "right" radius -- sweep it and show the
    # ranking is stable. Off by default because it costs a pass over the GT.
    "gt_crop": {"enabled": False, "radius": 0.50, "radii": [0.3, 0.5, 1.0],
                "voxel": None},
    "tile_budget": 8_000_000,
    "tile_size": None,
    # The SOR stand-in. Computed here so the labelled cloud carries every
    # channel the analysis needs, rather than requiring a separate pass.
    "knn": {
        "enabled": True,
        "k": 6,
        "field_name": "mean_knn_dist",
        "margin": 1.0,
        "tile_budget": 6_000_000,
        "recompute": False,
    },
}


def deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    with open(path) as f:
        cfg = deep_merge(DEFAULTS, yaml.safe_load(f) or {})
    if not cfg.get("slam_las"):
        sys.exit("ERROR: config is missing required key 'slam_las'")
    if not os.path.exists(cfg["slam_las"]):
        sys.exit(f"ERROR: slam_las does not exist: {cfg['slam_las']}")

    # `gt_e57_dir` stays valid as the single-group shorthand.
    groups = cfg.get("gt_groups")
    if not groups:
        if not cfg.get("gt_e57_dir") and not cfg.get("merged_gt_las"):
            sys.exit("ERROR: config needs 'gt_groups', or 'gt_e57_dir' / "
                     "'merged_gt_las' as the single-group shorthand")
        base = cfg.get("gt_e57_dir") or cfg["merged_gt_las"]
        groups = [{"name": os.path.splitext(os.path.basename(
                       base.rstrip("/")))[0],
                   "e57_dir": cfg.get("gt_e57_dir"),
                   "merged_gt_las": cfg.get("merged_gt_las"),
                   "initial_matrix": cfg.get("initial_matrix")}]
    normalised = []
    for i, g in enumerate(groups):
        g = dict(g)
        g.setdefault("name", f"group{i + 1}")
        # A group may be driven either from a station-tagged merged cloud (the
        # preferred form, built by merge_gt_e57.py) or from the per-station
        # E57s it would otherwise be built from.
        merged = g.get("merged_gt_las")
        has_merged = bool(merged) and os.path.exists(merged)
        if not g.get("e57_dir") and not g.get("files") and not has_merged:
            sys.exit(f"ERROR: group '{g['name']}' needs 'e57_dir', 'files', or "
                     f"an existing 'merged_gt_las'. Build the merged cloud "
                     f"with merge_gt_e57.py.")
        if has_merged and not g.get("e57_dir") and not g.get("files"):
            sidecar = e57.sidecar_path(merged)
            if not os.path.exists(sidecar):
                sys.exit(f"ERROR: group '{g['name']}': {merged} has no station "
                         f"sidecar at {sidecar}, and no E57 source to fall back "
                         f"on. Rebuild it with merge_gt_e57.py.")
            if not lasio.has_field(merged, e57.STATION_FIELD):
                sys.exit(f"ERROR: group '{g['name']}': {merged} carries no "
                         f"'{e57.STATION_FIELD}' field, so the visibility model "
                         f"cannot tell its stations apart. Rebuild it with "
                         f"merge_gt_e57.py.")
        if g.get("e57_dir") and not os.path.exists(g["e57_dir"]):
            sys.exit(f"ERROR: group '{g['name']}': {g['e57_dir']} does not exist")
        for f in g.get("files", []):
            if not os.path.exists(f):
                sys.exit(f"ERROR: group '{g['name']}': {f} does not exist")
        normalised.append(g)
    names = [g["name"] for g in normalised]
    if len(set(names)) != len(names):
        sys.exit(f"ERROR: duplicate group names: {names}")
    cfg["gt_groups"] = normalised
    return cfg


def group_source(group, scratch_root):
    """Return (station_paths, merged_las_path) for one group.

    `station_paths` is None when the group works from a merged cloud that
    already carries station ids, which is the preferred form -- see
    `group_is_merged`.

    A group given as an explicit file list is staged into its own directory of
    symlinks, because the E57 helpers take a directory.
    """
    merged = group.get("merged_gt_las") or os.path.join(
        scratch_root, "merged", f"{group['name']}_gt.las")

    if group_is_merged(group, merged):
        return None, merged

    if group.get("files"):
        staged = os.path.join(scratch_root, f"stations_{group['name']}")
        os.makedirs(staged, exist_ok=True)
        for f in group["files"]:
            link = os.path.join(staged, os.path.basename(f))
            if not os.path.exists(link):
                os.symlink(os.path.abspath(f), link)
        src = staged
    else:
        src = group.get("e57_dir")
        if not src:
            sys.exit(f"ERROR: group '{group['name']}' has neither a merged GT "
                     f"with station ids nor an 'e57_dir'/'files' to build one "
                     f"from. Build the merged cloud with merge_gt_e57.py.")
    return src, merged


def group_is_merged(group, merged):
    """True if this group should be driven from a merged, station-tagged cloud.

    The merged form is preferred whenever it is available, because then the
    labels, the ICP and the visibility verdicts all derive from the same points
    -- so cropping something out of the ground truth (people who walked through
    the scene, say) removes it from the visibility model as well, instead of
    leaving the scanner apparently able to see something that is no longer
    there.
    """
    if group.get("use_e57_stations"):
        return False                     # explicit opt-out
    if not os.path.exists(merged):
        return False
    if not os.path.exists(e57.sidecar_path(merged)):
        return False
    return lasio.has_field(merged, e57.STATION_FIELD)


def ensure_merged_gt(cfg, verbose=True):
    """Merge the stations into one project-frame LAS, once, and cache it."""
    merged = cfg.get("merged_gt_las")
    if not merged:
        merged = os.path.join(os.path.dirname(cfg["gt_e57_dir"]), "merged",
                              "gt_merged.las")
    if os.path.exists(merged):
        if verbose:
            print(f"  Merged GT already present: {merged} "
                  f"({lasio.point_count(merged):,} points)")
        return merged
    os.makedirs(os.path.dirname(merged), exist_ok=True)
    e57.merge_to_las(cfg["gt_e57_dir"], merged, verbose=verbose)
    return merged


def near_station_mask(slam_path, panoramas, transform, margin=2.0, verbose=True):
    """Points within reach of at least one station, streamed.

    Only these can be observed, so only these need a C2C distance; the rest of a
    400 m SLAM footprint would otherwise be tiled and searched for nothing.
    """
    origins = np.array([p.t for p in panoramas])
    radii = np.array([p.max_range for p in panoramas]) + margin
    T = np.asarray(transform, dtype=np.float64)
    total = lasio.point_count(slam_path)
    mask = np.zeros(total, dtype=bool)
    tick = lasio.Ticker(enabled=verbose)
    at = 0
    for xyz in lasio.iter_xyz(slam_path):
        moved = xyz @ T[:3, :3].T + T[:3, 3]
        hit = np.zeros(len(moved), dtype=bool)
        for o, rad in zip(origins, radii):
            np.logical_or(hit, np.linalg.norm(moved - o, axis=1) <= rad, out=hit)
        mask[at:at + len(moved)] = hit
        at += len(moved)
        tick(f"    station reach: {at:,} / {total:,} points")
    tick.done(f"    {int(mask.sum()):,} / {total:,} points within reach of a station")
    return mask


def gather_subset(slam_path, mask, transform=None, verbose=True):
    """Pull the masked points into memory, optionally transformed."""
    T = None if transform is None else np.asarray(transform, dtype=np.float64)
    out = np.empty((int(mask.sum()), 3), dtype=np.float64)
    at = 0
    src = 0
    for xyz in lasio.iter_xyz(slam_path):
        sel = mask[src:src + len(xyz)]
        src += len(xyz)
        if not sel.any():
            continue
        block = xyz[sel]
        if T is not None:
            block = block @ T[:3, :3].T + T[:3, 3]
        out[at:at + len(block)] = block
        at += len(block)
    return out[:at]


def _pick_by_fit(slam_path, merged, transform, coarse_info, cfg, name):
    """Choose among distinct coarse peaks by which one ICP can actually fit.

    Correlation score ranks how well two occupancy rasters overlap, which is not
    the same as the clouds aligning. Where several peaks score alike, a short
    ICP on downsampled clouds settles it for a fraction of the cost of finding
    out later from a bad domain mask.
    """
    import open3d as o3d
    finest = min(l[0] for l in cfg["icp"]["levels"])
    gt = lasio.stream_voxel_downsample(merged, finest, verbose=False)
    sl = lasio.stream_voxel_downsample(slam_path, finest, verbose=False)
    src = register.to_o3d(sl).voxel_down_sample(finest * 2)
    tgt = register.to_o3d(gt).voxel_down_sample(finest * 2)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=finest * 6, max_nn=30))

    trials = [("best", coarse_info["yaw_deg"], coarse_info["score"], transform)]
    for a in coarse_info["alternatives"]:
        trials.append(("alt", a["yaw"], a["score"],
                       np.asarray(a["transform"], dtype=np.float64)))

    print(f"  Testing {len(trials)} distinct coarse peaks with a short ICP ...")
    scored = []
    for kind, yaw, score, T in trials:
        Ticp, _ = register.register(
            sl, gt, init=T, levels=[tuple(l) for l in cfg["icp"]["levels"]],
            max_iter=30, tukey_k=float(cfg["icp"]["tukey_k"]), verbose=False)
        r = o3d.pipelines.registration.evaluate_registration(
            src, tgt, finest * 2.4, Ticp)
        scored.append((r.fitness, r.inlier_rmse, yaw, score, Ticp, kind))
        print(f"    yaw {yaw:6.1f} deg  raster score {score:.4f}  "
              f"-> ICP fitness {r.fitness:.4f}  rmse {r.inlier_rmse * 100:5.2f} cm"
              f"  [{kind}]")

    scored.sort(key=lambda t: -t[0])
    fit, rmse, yaw, score, Ticp, kind = scored[0]
    if kind != "best":
        print(f"  NOTE [{name}]: the highest-correlation peak was NOT the best "
              f"fit. Using yaw {yaw:.1f} deg (fitness {fit:.4f}) instead.")
    # Do not relax this threshold. A weak coarse probe means a weak lock, and
    # the full ICP will not recover from it: its coarsest level uses a 60 cm
    # correspondence distance, so a high fitness there only means "aligned to
    # within 60 cm". Read the finest level, and check the per-station distances
    # before trusting a registration whose probe scored low.
    if fit < 0.5:
        print(f"  WARNING [{name}]: even the best peak only probes at fitness "
              f"{fit:.4f}. The scene is probably too repetitive for a top-down "
              f"raster search — set this group's initial_matrix by hand.")
    coarse_info = dict(coarse_info)
    coarse_info.update({"chosen_yaw_deg": yaw, "chosen_fitness": float(fit),
                        "peak_was_best_fit": kind == "best"})
    return Ticp, coarse_info


def process_group(group, cfg, slam_path, scratch_root, rounds, corners,
                  verbose=True):
    """Register one station group to the SLAM cloud and classify visibility.

    Every group is an independent RTC project with its own coordinate frame, so
    it gets its own coarse alignment and its own ICP. Only the per-point verdict
    comes back out; the panoramas are freed with the function's scope.
    """
    name = group["name"]
    src_dir, merged = group_source(group, scratch_root)
    print(f"\n{'=' * 70}\nGroup '{name}'\n{'=' * 70}")

    if os.path.exists(merged):
        print(f"  Merged GT already present: {merged} "
              f"({lasio.point_count(merged):,} points)")
    else:
        os.makedirs(os.path.dirname(merged), exist_ok=True)
        e57.merge_to_las(src_dir, merged, verbose=verbose)
        src_dir = None if group_is_merged(group, merged) else src_dir

    vcfg = cfg["visibility"]
    # The panoramas answer "what did the scanner see along this ray?" and the
    # C2C answers "how far is this point from the reference surface?". They do
    # not have to come from the same cloud, and must not when the reference has
    # been cropped: a point on something cropped away sits in front of whatever
    # is behind it, so a panorama built from the cropped cloud calls it a flyer.
    # Built from the UNCROPPED cloud it is 'observed' instead, finds no
    # reference neighbour within c2c_max_dist, and drops out as OUT_NO_GT --
    # which is right, while a genuine flyer still reads as free space.
    pano_las = group.get("panorama_gt_las") or cfg.get("panorama_gt_las")
    if src_dir is None:
        pano_src = pano_las or merged
        if pano_las:
            if not os.path.exists(pano_las):
                sys.exit(f"ERROR [{name}]: panorama_gt_las not found: {pano_las}")
            print(f"  Visibility from a SEPARATE cloud: "
                  f"{os.path.basename(pano_las)}\n"
                  f"    (C2C still measured against {os.path.basename(merged)})")
        meta = e57.read_sidecar(pano_src)
        print(f"  Visibility from {'that cloud' if pano_las else 'the merged cloud itself'} "
              f"({meta['n_stations']} stations in "
              f"{os.path.basename(e57.sidecar_path(pano_src))})")
        panoramas = vis.build_panoramas_from_las(
            pano_src, meta["stations"], resolution=float(vcfg["resolution"]),
            station_field=meta.get("station_field"))
    else:
        print("  Visibility from the per-station E57s "
              "(no station-tagged merged cloud for this group)")
        panoramas = vis.build_panoramas(src_dir,
                                        resolution=float(vcfg["resolution"]))

    # ------------------------------------------------------- coarse alignment
    coarse_info = None
    if group.get("initial_matrix") is not None:
        transform = np.asarray(group["initial_matrix"], dtype=np.float64)
        print("  Using initial_matrix from the config")
    elif cfg["coarse"]["enabled"]:
        ccfg = cfg["coarse"]
        transform, coarse_info = coarse.align(
            slam_path, merged,
            cell=float(ccfg["cell"]), band=tuple(ccfg["band"]),
            yaw_step=float(ccfg["yaw_step"]),
            refine_step=float(ccfg["refine_step"]),
            slam_percentile=float(ccfg.get("slam_percentile", 0.1)),
            try_flip=bool(ccfg.get("try_flip", True)),
            n_candidates=int(ccfg.get("n_candidates", 1)))
        print(f"  yaw={coarse_info['yaw_deg']:.2f} deg  "
              f"score={coarse_info['score']:.4f}  "
              f"best rival >5 deg away={coarse_info['rival_score']:.4f}")
        if coarse_info.get("alternatives"):
            transform, coarse_info = _pick_by_fit(
                slam_path, merged, transform, coarse_info, cfg, name)
        elif coarse_info["score"] < 1.5 * max(coarse_info["rival_score"], 1e-6):
            print(f"  WARNING [{name}]: the best yaw is not clearly separated "
                  f"from a competing angle. Check the QA render, set "
                  f"coarse.n_candidates above 1 so the alternatives are tested, "
                  f"or set this group's initial_matrix by hand.")
    else:
        transform = np.eye(4)
        print("  Starting from identity")

    # ------------------------------------------- visibility / ICP rounds
    gt_icp = register.load_gt_for_icp(
        merged, min(l[0] for l in cfg["icp"]["levels"]))

    history = []
    reg_info = None
    for rnd in range(rounds):
        print(f"\n  --- {name} round {rnd + 1}/{rounds} ---")
        prev = transform.copy()

        verdict = vis.classify_las(
            slam_path, panoramas, transform,
            eps0=float(vcfg["eps0"]), eps_rate=float(vcfg["eps_rate"]),
            behind_tolerance=float(vcfg["behind_tolerance"]))
        stats = vis.summarise(verdict)
        print("  Visibility:")
        for label, (count, frac) in stats.items():
            print(f"    {label:<32s} {count:>13,}  ({frac * 100:5.2f} %)")

        observed = verdict == vis.VIS_OBSERVED
        if not observed.any():
            sys.exit(f"ERROR: group '{name}' sees no SLAM point at all — its "
                     f"coarse alignment is probably wrong.")

        # ICP realigns on the observed points only. Free-space points are left
        # out deliberately: they do not lie on any scanned surface, so letting
        # them pull correspondences would drag the fit toward the very errors
        # the registration is meant to expose.
        src = gather_subset(slam_path, observed, transform=None)
        print(f"  Registering on {len(src):,} observed points ...")
        transform, reg_info = register.register(
            src, gt_icp, init=transform,
            levels=[tuple(l) for l in cfg["icp"]["levels"]],
            max_iter=int(cfg["icp"]["max_iter"]),
            tukey_k=float(cfg["icp"]["tukey_k"]))
        del src

        angle, trans = register.transform_delta(prev, transform, corners)
        history.append({
            "round": rnd + 1,
            "observed_points": int(observed.sum()),
            "fitness": reg_info["fitness"],
            "inlier_rmse": reg_info["inlier_rmse"],
            "delta_rotation_deg": angle,
            "delta_translation_m": trans,
            "visibility": {k: v[0] for k, v in stats.items()},
        })
        print(f"  Transform moved by {angle:.4f} deg / "
              f"{trans * 1000:.1f} mm (max over the cloud extent)")
        if angle < 1e-3 and trans < 1e-3:
            print("  Transform converged; stopping early.")
            break

    tilt = math.degrees(math.acos(min(1.0, max(-1.0, transform[2, 2]))))
    max_tilt = float(cfg.get("max_tilt_deg", 5.0))
    if tilt > max_tilt:
        fitness_note = "" if reg_info is None else (
            f"  ICP reported fitness {reg_info['fitness']:.4f} — if that looks "
            f"healthy, note that fitness is a local measure and says nothing "
            f"about whether the pose is globally right.")
        sys.exit(
            f"ERROR [{name}]: the registered transform tilts the SLAM cloud by "
            f"{tilt:.1f} deg out of level (limit {max_tilt:.1f} deg).\n"
            f"  Both clouds are gravity-levelled — a terrestrial scanner levels "
            f"its project frame, and the SLAM cloud is IMU-gravity-aligned — so "
            f"a large tilt is a registration failure, not a fit. A tilt near "
            f"180 deg means the z-flipped hypothesis won the raster search; set "
            f"coarse.try_flip to false. Anything else means ICP walked out of "
            f"the basin: set this group's initial_matrix by hand.\n"
            + fitness_note)

    if reg_info is not None and reg_info["fitness"] < 0.5:
        print(f"  WARNING [{name}]: the converged ICP fitness is only "
              f"{reg_info['fitness']:.4f} (rmse {reg_info['inlier_rmse'] * 100:.1f} cm). "
              f"Check the slice render before trusting these labels — a bad "
              f"registration mislabels every point this group decides.")

    print(f"\n  Final visibility pass for '{name}' ...")
    verdict = vis.classify_las(
        slam_path, panoramas, transform,
        eps0=float(vcfg["eps0"]), eps_rate=float(vcfg["eps_rate"]),
        behind_tolerance=float(vcfg["behind_tolerance"]))
    stats = vis.summarise(verdict)
    for label, (count, frac) in stats.items():
        print(f"    {label:<32s} {count:>13,}  ({frac * 100:5.2f} %)")

    del panoramas, gt_icp
    return {"name": name, "transform": transform, "verdict": verdict,
            "merged": merged, "full_gt": pano_las, "coarse": coarse_info,
            "rounds": history, "reg_info": reg_info,
            "visibility": {k: {"count": v[0], "fraction": v[1]}
                           for k, v in stats.items()}}


def main():
    ap = argparse.ArgumentParser(
        description="Label a SLAM cloud against E57 RTC stations using a "
                    "scanner-visibility model. Several station groups, each "
                    "its own RTC project with its own frame, are registered "
                    "independently and their verdicts fused.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--no-gt-crop", action="store_true",
                    help="skip the CD_sym reference crop even if the config "
                         "enables it")
    ap.add_argument("--recompute-gt-crop", action="store_true",
                    help="rebuild the crop even if the output already exists")
    ap.add_argument("--skip-c2c", action="store_true",
                    help="Domain mask only; no C2C labels")
    ap.add_argument("--no-qa", action="store_true")
    ap.add_argument("--skip-knn", action="store_true",
                    help="Do not compute the mean_knn_dist channel")
    ap.add_argument("--only-group", default=None,
                    help="Process just this group (by name). Useful for "
                         "checking one section's registration before "
                         "committing to a full run.")
    ap.add_argument("--output-frame", default=None,
                    help="Group name whose frame the output coordinates use, "
                         "or 'original' for the untransformed SLAM frame. "
                         "Defaults to the single group's frame when there is "
                         "one, and to 'original' when there are several — with "
                         "several groups no single registered frame is correct "
                         "for the whole cloud.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slam_path = cfg["slam_las"]
    rounds = args.rounds if args.rounds is not None else int(cfg["rounds"])
    tau = float(cfg["tau"])
    max_dist = float(cfg["c2c_max_dist"])
    drop_cropped_away = bool(cfg.get("drop_cropped_away", False))
    cropped_away_margin = float(cfg.get("cropped_away_margin", 0.01))

    groups = cfg["gt_groups"]
    if args.only_group:
        groups = [g for g in groups if g["name"] == args.only_group]
        if not groups:
            sys.exit(f"ERROR: no group named '{args.only_group}' in the config")

    base = os.path.splitext(os.path.basename(slam_path))[0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(slam_path)),
                                           "labeled")
    os.makedirs(out_dir, exist_ok=True)
    scratch_root = args.scratch or os.path.join(out_dir, ".work")
    os.makedirs(scratch_root, exist_ok=True)

    t_start = time.time()
    n_slam = lasio.point_count(slam_path)
    print(f"Dataset      : {cfg.get('name', base)}")
    print(f"SLAM cloud   : {slam_path}  ({n_slam:,} points)")
    print(f"Groups       : {', '.join(g['name'] for g in groups)}")
    print(f"Output       : {out_dir}")

    slam_mins, slam_maxs = lasio.bounds(slam_path)
    corners = register.bbox_corners(slam_mins, slam_maxs)

    # --------------------------------------------------- per-group processing
    results = [process_group(g, cfg, slam_path, scratch_root, rounds, corners)
               for g in groups]

    # ------------------------------------------------------------- fuse
    # Verdict codes are ordered by priority, so the winner is the maximum. The
    # group that produced it is recorded, which is what lets the analysis train
    # on one section and test on another.
    print(f"\n{'=' * 70}\nFusing {len(results)} group(s)\n{'=' * 70}")
    verdict = np.zeros(n_slam, dtype=np.uint8)
    gt_group = np.zeros(n_slam, dtype=np.uint8)     # 0 = seen by nobody
    for gi, res in enumerate(results, start=1):
        better = res["verdict"] > verdict
        verdict[better] = res["verdict"][better]
        gt_group[better] = gi
        res["verdict"] = None                        # free 194 MB per group
    for gi, res in enumerate(results, start=1):
        owned = int((gt_group == gi).sum())
        print(f"  {res['name']:<20s} decides {owned:>13,} points "
              f"({100.0 * owned / n_slam:5.2f} %)")

    seen = (verdict == vis.VIS_OBSERVED) | (verdict == vis.VIS_FREE_SPACE)
    code = np.full(n_slam, domain_mod.IN_DOMAIN, dtype=np.uint8)
    code[~seen] = domain_mod.OUT_NOT_VISIBLE

    exc_cfg = cfg.get("exclude") or {}
    exclusions = domain_mod.ExclusionVolumes(boxes=exc_cfg.get("boxes"),
                                             prisms=exc_cfg.get("prisms"),
                                             voxels=exc_cfg.get("voxels"))

    # --------------------------------------------------------------- C2C
    # Each point is measured against the GT of the group that actually saw it,
    # in that group's own frame.
    print("\nC2C labels ...")
    c2c_full = np.full(n_slam, np.inf, dtype=np.float64)
    if args.skip_c2c:
        print("  skipped (--skip-c2c)")
    else:
        if len(exclusions):
            print(f"  Exclusion volumes: {exclusions.describe()}")
        for gi, res in enumerate(results, start=1):
            subset = seen & (gt_group == gi)
            if not subset.any():
                print(f"  {res['name']}: no points to measure")
                continue
            print(f"  {res['name']}: {int(subset.sum()):,} points ...")
            pts = gather_subset(slam_path, subset, transform=res["transform"])
            if len(exclusions):
                inside = exclusions.contains(pts)
                idx = np.flatnonzero(subset)
                code[idx[inside]] = domain_mod.OUT_EXCLUDED
            d = c2c_mod.compute_c2c(
                pts, res["merged"], max_dist=max_dist,
                tile_size=cfg["tile_size"], tile_budget=int(cfg["tile_budget"]),
                scratch_dir=os.path.join(scratch_root, f"tiles_{res['name']}"))
            c2c_full[subset] = d
            # A cropped reference deletes surfaces the SLAM cloud still has.
            # Points on one of them keep a neighbour if the crop face happens
            # to be within max_dist, and are then labelled with the distance to
            # the wrong surface -- the bright rind along every crop boundary.
            # Measuring the same points against the UNCROPPED cloud identifies
            # them: their true surface is closer than anything that survived.
            full_gt = res.get("full_gt")
            if full_gt and drop_cropped_away and full_gt != res["merged"]:
                d_full = c2c_mod.compute_c2c(
                    pts, full_gt, max_dist=max_dist,
                    tile_size=cfg["tile_size"],
                    tile_budget=int(cfg["tile_budget"]),
                    scratch_dir=os.path.join(scratch_root,
                                             f"tiles_{res['name']}_full"),
                    verbose=False)
                rind = np.isfinite(d_full) & (d_full < d - cropped_away_margin)
                if rind.any():
                    idx = np.flatnonzero(subset)
                    code[idx[rind]] = domain_mod.OUT_NO_GT
                    # Drop the distance too. It was measured against the wrong
                    # surface, and every other gate leaves an excluded point at
                    # the sentinel -- so a viewer coloured by C2C keeps showing
                    # the domain, which is the only reason that view is useful.
                    c2c_full[idx[rind]] = np.inf
                    print(f"    nearest surface cropped away (margin "
                          f"{cropped_away_margin * 100:.0f} cm): "
                          f"{int(rind.sum()):,} points -> OUT_NO_GT")
                del d_full
            del d
            del pts
            finite = np.isfinite(c2c_full) & subset & (code == domain_mod.IN_DOMAIN)
            if finite.any():
                print(f"    in-domain median C2C: "
                      f"{np.median(c2c_full[finite]) * 100:.2f} cm")

    # ------------------------------------------------- observed but unmeasurable
    # A point can be VIS_OBSERVED and still have no GT neighbour inside the cap.
    # "Observed" is a statement about the ray: the band is [near, far] taken over
    # a 3x3 panorama window, which is wide wherever the scene has depth structure,
    # and in vegetation the GT is a sparse 3D medium whose gaps exceed the cap.
    # Neither verdict is wrong — the scanner did return something that way, and
    # there really is no GT point nearby — the spot is simply not measurable.
    #
    # It has to leave the evaluation rather than enter it carrying C2C_SENTINEL,
    # which is negative and therefore sorts as the most accurate value in the
    # cloud. On one outdoor scene that was 274 k points, clustered in a handful of 2 m
    # columns at a median height of 2.5 m: tree canopy.
    if not args.skip_c2c:
        unmeasured = ~np.isfinite(c2c_full) & (code == domain_mod.IN_DOMAIN)
        no_gt = unmeasured & (verdict == vis.VIS_OBSERVED)
        if no_gt.any():
            code[no_gt] = domain_mod.OUT_NO_GT
            print(f"\n  observed but no GT within {max_dist:.2f} m: "
                  f"{int(no_gt.sum()):,} points -> OUT_NO_GT")
        # The free-space half is kept, but clamped. Those points sit where a beam
        # flew through, so they are wrong rather than unmeasurable — dropping
        # them would delete the easiest true positives in the label set. They
        # must not keep the sentinel either: -1 sorts below every real distance,
        # so the most certainly-wrong points in the cloud would train as the most
        # accurate. max_dist is where every other capped point already sits.
        fs = unmeasured & (verdict == vis.VIS_FREE_SPACE)
        if fs.any():
            if bool(cfg.get("drop_unmeasured_free_space", False)):
                # With a cropped reference the clamp stops being a conservative
                # stand-in and becomes the majority of the label set: on
                # one scene it went from 2.9 % of the domain to 33.7 %, all
                # sitting at exactly max_dist and therefore counted as outliers
                # at every tau. A flyer with a reference surface nearby is still
                # measured and kept; one with nothing nearby is in a region we
                # cropped away, and we have no distance to assign it.
                code[fs] = domain_mod.OUT_NO_GT
                print(f"  free-space with no GT within {max_dist:.2f} m: "
                      f"{int(fs.sum()):,} points -> OUT_NO_GT "
                      f"(drop_unmeasured_free_space)")
            else:
                c2c_full[fs] = max_dist
                print(f"  free-space with no GT within {max_dist:.2f} m: "
                      f"{int(fs.sum()):,} points -> C2C clamped to {max_dist:.2f} m")
        good = np.isfinite(c2c_full) & (code == domain_mod.IN_DOMAIN)
        if good.any():
            print(f"  in-domain median C2C after the gate: "
                  f"{np.median(c2c_full[good]) * 100:.2f} cm")

    stats = domain_mod.summarise(code)
    print("\n  Domain composition:")
    for label, (count, frac) in stats.items():
        print(f"    {label:<32s} {count:>13,}  ({frac * 100:5.2f} %)")

    # ------------------------------------------------------------- gt_crop
    # The completeness half of CD_sym, d(GT -> SLAM), is unbounded: a GT point
    # the SLAM never reached contributes its full distance. Cropping the GT to
    # the volume the SLAM actually visited bounds it by construction.
    #
    # It is built here, once, from the UNFILTERED cloud, and every filter
    # variant is then scored against the same reference. Rebuilding it per
    # variant would let an aggressive filter improve its own score by shrinking
    # the thing it is measured against.
    crop_cfg = cfg.get("gt_crop") or {}
    crop_paths = {}
    if crop_cfg.get("enabled") and not args.no_gt_crop:
        radii = crop_cfg.get("radii") or [float(crop_cfg.get("radius", 0.50))]
        print(f"\nGT crop for CD_sym (radii {', '.join(f'{r:g}' for r in radii)} m) ...")
        for res in results:
            for r in radii:
                tag = f"{res['name']}_r{str(r).replace('.', 'p')}"
                dst = os.path.join(out_dir, f"gt_crop_{tag}.las")
                if os.path.exists(dst) and not args.recompute_gt_crop:
                    print(f"  {tag}: already present, skipping")
                    crop_paths.setdefault(res["name"], {})[str(r)] = dst
                    continue
                domain_mod.crop_gt_to_slam_streaming(
                    res["merged"], slam_path, float(r), dst,
                    transform=res["transform"],
                    voxel=crop_cfg.get("voxel"))
                crop_paths.setdefault(res["name"], {})[str(r)] = dst

    # ------------------------------------------------------------- kNN channel
    knn_cfg = cfg["knn"]
    knn_info = None
    knn_field = knn_cfg["field_name"]
    fields = {
        "C2C_distance": (np.where(np.isfinite(c2c_full), c2c_full,
                                  C2C_SENTINEL), np.float32),
        "domain_code": (code, np.uint8),
        "eval_domain": ((code == domain_mod.IN_DOMAIN).astype(np.uint8), np.uint8),
        "visibility_code": (verdict, np.uint8),
        "gt_group": (gt_group, np.uint8),
    }
    if args.skip_knn or not knn_cfg["enabled"]:
        print(f"\n  Skipping {knn_field}.")
    else:
        print(f"\n  Computing {knn_field} ...")
        values, uncertain = knn_mod.mean_knn_distance(
            slam_path, k=int(knn_cfg["k"]), margin=float(knn_cfg["margin"]),
            tile_budget=int(knn_cfg["tile_budget"]),
            scratch_dir=os.path.join(scratch_root, "knn_tiles"))
        fields[knn_field] = (values, np.float32)
        knn_info = {"k": int(knn_cfg["k"]), "margin": float(knn_cfg["margin"]),
                    "median": float(np.median(values[values >= 0])),
                    "beyond_margin": int(uncertain)}
        print(f"  median {knn_info['median'] * 100:.3f} cm")

    # -------------------------------------------------------------- write out
    frame = args.output_frame or cfg.get("output_frame")
    if frame is None:
        frame = results[0]["name"] if len(results) == 1 else "original"
    if frame == "original":
        out_transform = None
        print(f"\nWriting in the original SLAM frame — with several groups no "
              f"single registered frame is correct for the whole cloud. Each "
              f"group's transform is in the report; pass --output-frame "
              f"<group> to write in one of them instead.")
    else:
        match = [r for r in results if r["name"] == frame]
        if not match:
            sys.exit(f"ERROR: --output-frame '{frame}' is not a processed group")
        out_transform = match[0]["transform"]
        print(f"\nWriting in the frame of group '{frame}'.")

    print("Writing ...")
    out_las = os.path.join(out_dir, f"{base}_labeled.las")
    lasio.write_las_with_fields(slam_path, out_las, fields=fields,
                                transform=out_transform)

    report = {
        "dataset": cfg.get("name", base),
        "slam_las": slam_path,
        "output_las": out_las,
        "output_frame": frame,
        "gt_crop": crop_paths or None,
        "groups": [{
            "index": gi,
            "name": r["name"],
            "merged_gt_las": r["merged"],
            "transform": np.asarray(r["transform"]).tolist(),
            "coarse": r["coarse"],
            "rounds": r["rounds"],
            "visibility": r["visibility"],
            "points_decided": int((gt_group == gi).sum()),
        } for gi, r in enumerate(results, start=1)],
        "domain": {k: {"count": v[0], "fraction": v[1]} for k, v in stats.items()},
        "mean_knn_dist": knn_info,
        "settings": {"visibility": cfg["visibility"], "icp": cfg["icp"],
                     "tau": tau, "c2c_max_dist": max_dist},
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    report_path = os.path.join(out_dir, f"{base}_labels.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  {report_path}")

    if not args.no_qa:
        print("\n  QA renders ...")
        step = max(1, n_slam // 20_000_000)
        pts = gather_subset(slam_path, np.ones(n_slam, dtype=bool),
                            transform=out_transform)[::step]
        qa.render(pts, c2c_full[::step], code[::step], out_dir, base, cell=0.25,
                  tau=tau, reg_info=results[0]["reg_info"])
        del pts

    print(f"\nDone in {time.time() - t_start:.1f} s.")
    n_in = int((code == domain_mod.IN_DOMAIN).sum())
    print(f"Evaluation domain: {n_in:,} / {n_slam:,} points "
          f"({100.0 * n_in / n_slam:.2f} %)")


if __name__ == "__main__":
    main()
