#!/usr/bin/env python3
"""
SLAM -> GT registration, replacing the manual CloudCompare ICP step.

The coarse alignment stays a one-time human decision: it is read from the
dataset config as a 4x4 matrix (the same one CloudCompare hands you) and reused
for every rerun of that dataset. Everything after it is automatic.

Fine alignment is multiscale point-to-plane ICP with a robust kernel on the
finest level. The kernel matters here: the SLAM cloud always contains regions
the RTC never scanned, and without down-weighting they pull the fit. That, plus
re-running ICP on the masked cloud, is what the manual "delete, then ICP again"
round was doing by hand.
"""

import numpy as np
import open3d as o3d

from . import lasio

# (voxel size, max correspondence distance) per level, coarse to fine.
DEFAULT_LEVELS = [(0.20, 0.60), (0.10, 0.30), (0.05, 0.12)]


def to_o3d(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    return pcd


def load_gt_for_icp(gt_path, finest_voxel, verbose=True):
    """Stream-downsample the GT once, for reuse across all ICP levels.

    Streamed at half the finest ICP voxel on purpose: re-voxelising an already
    voxelised cloud at the *same* resolution silently discards ~35% of it,
    because the second grid has a different phase and merges neighbours the
    first grid kept apart. Streaming finer keeps the finest ICP level at its
    intended density.
    """
    voxel = finest_voxel / 2.0
    if verbose:
        print(f"  Downsampling GT for ICP (voxel={voxel:.3f} m, "
              f"half of the finest ICP level) ...")
    return lasio.stream_voxel_downsample(gt_path, voxel, verbose=verbose)


def register(slam_xyz, gt_icp_xyz, init=None, levels=None, max_iter=60,
             tukey_k=0.05, verbose=True):
    """Multiscale point-to-plane ICP of `slam_xyz` onto `gt_icp_xyz`.

    `init` is the 4x4 coarse transform from the dataset config. Returns
    (transform, info) where transform maps SLAM coordinates into the GT frame.
    """
    levels = levels or DEFAULT_LEVELS
    init = np.eye(4) if init is None else np.asarray(init, dtype=np.float64)

    src_full = to_o3d(slam_xyz)
    tgt_full = to_o3d(gt_icp_xyz)

    transform = init.copy()
    info = {"levels": []}

    for i, (voxel, max_corr) in enumerate(levels):
        src = src_full.voxel_down_sample(voxel)
        tgt = tgt_full.voxel_down_sample(voxel)
        tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30))

        # Robust kernel only on the finest level: on the coarse levels the
        # residuals are still dominated by the initial misalignment, and
        # down-weighting them there would stall convergence.
        if i == len(levels) - 1:
            kernel = o3d.pipelines.registration.TukeyLoss(k=tukey_k)
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        result = o3d.pipelines.registration.registration_icp(
            src, tgt, max_corr, transform, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

        transform = result.transformation.copy()
        level_info = {
            "voxel": voxel,
            "max_correspondence_distance": max_corr,
            "source_points": len(src.points),
            "target_points": len(tgt.points),
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
        }
        info["levels"].append(level_info)
        if verbose:
            print(f"    level {i + 1}/{len(levels)} voxel={voxel:.3f} "
                  f"corr={max_corr:.3f}: fitness={result.fitness:.4f} "
                  f"rmse={result.inlier_rmse * 100:.2f} cm "
                  f"({len(src.points):,} vs {len(tgt.points):,} pts)")

    # Guard: ICP must never leave the cloud worse aligned than the coarse
    # transform it started from. The first level uses a wide correspondence
    # distance, which can pull an already-good alignment off a sharp optimum;
    # if that happened, fall back rather than silently degrading the labels.
    voxel, max_corr = levels[-1]
    src = src_full.voxel_down_sample(voxel)
    tgt = tgt_full.voxel_down_sample(voxel)
    before = o3d.pipelines.registration.evaluate_registration(src, tgt, max_corr, init)
    after = o3d.pipelines.registration.evaluate_registration(src, tgt, max_corr, transform)
    info["initial"] = {"fitness": float(before.fitness),
                       "inlier_rmse": float(before.inlier_rmse)}
    info["final"] = {"fitness": float(after.fitness),
                     "inlier_rmse": float(after.inlier_rmse)}

    if _is_better(before, after):
        if verbose:
            print(f"    ICP did not improve on the initial transform "
                  f"(fitness {before.fitness:.4f} vs {after.fitness:.4f}, "
                  f"rmse {before.inlier_rmse * 100:.2f} vs "
                  f"{after.inlier_rmse * 100:.2f} cm) — keeping the initial one")
        transform = init.copy()
        info["used"] = "initial"
        info["fitness"] = float(before.fitness)
        info["inlier_rmse"] = float(before.inlier_rmse)
    else:
        info["used"] = "icp"
        info["fitness"] = float(after.fitness)
        info["inlier_rmse"] = float(after.inlier_rmse)

    info["transform"] = transform.tolist()
    return transform, info


def _is_better(a, b, fitness_tol=0.01):
    """True if registration result `a` beats `b`.

    Fitness (overlap) dominates; when the two are within `fitness_tol` the
    lower correspondence RMSE wins.
    """
    if a.fitness > b.fitness + fitness_tol:
        return True
    if b.fitness > a.fitness + fitness_tol:
        return False
    return a.inlier_rmse < b.inlier_rmse


def apply_transform(xyz, transform):
    """Apply a 4x4 rigid transform to (N, 3) points."""
    t = np.asarray(transform, dtype=np.float64)
    return xyz @ t[:3, :3].T + t[:3, 3]


def transform_delta(a, b, points=None):
    """Rotation (deg) and displacement (m) between two 4x4 transforms.

    Used as the convergence test for the mask -> ICP -> mask loop.

    When `points` is given (typically the cloud's bounding-box corners) the
    displacement is the largest distance any of them moves. Comparing the raw
    translation vectors instead is misleading for a cloud far from the origin:
    on the outdoor dataset, sitting ~350 m out, a 1.7 deg rotation traded against
    5 m of translation while the points themselves barely shifted.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dr = a[:3, :3].T @ b[:3, :3]
    cos = (np.trace(dr) - 1.0) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    if points is None:
        trans = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    else:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pa = p @ a[:3, :3].T + a[:3, 3]
        pb = p @ b[:3, :3].T + b[:3, 3]
        trans = float(np.linalg.norm(pa - pb, axis=1).max())
    return angle, trans


def bbox_corners(mins, maxs):
    """The eight corners of a bounding box, for use with `transform_delta`."""
    return np.array(np.meshgrid(*zip(np.asarray(mins), np.asarray(maxs)))
                    ).reshape(3, -1).T
