#!/usr/bin/env python3
"""
Horizontal slice renders for checking a registration by eye.

Loading a 17 GB cloud next to a 12 GB ground truth in a viewer to see whether
they line up is impractical. A thin horizontal slice through both, drawn from
above, answers the same question in a PNG: walls that agree print as a single
line, a misregistration prints as a doubled line, and a yaw error shows up as a
wedge that opens across the scene.

The two clouds go into separate colour channels of one image — ground truth in
magenta, SLAM in green — so agreement reads as white and disagreement as colour
fringing. That is far easier to judge than two point clouds in similar colours.

Both clouds are streamed; nothing is held whole.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import lasio

MAX_RASTER = 3000          # keep the output image a sane size


def _overlap_extent(gt_pts, slam_pts, cell=1.0, margin=3.0):
    """Bounding box of the XY cells both clouds occupy.

    Framing on the ground truth alone is close to useless here: the RTC picks up
    returns a hundred metres out, so the building ends up a speck in a mostly
    black image. What matters for a registration check is where the two clouds
    are supposed to agree, which is their overlap.
    """
    lo = np.minimum(gt_pts[:, :2].min(axis=0), slam_pts[:, :2].min(axis=0))
    hi = np.maximum(gt_pts[:, :2].max(axis=0), slam_pts[:, :2].max(axis=0))
    shape = np.maximum(np.ceil((hi - lo) / cell).astype(np.int64) + 1, 1)

    def occ(pts):
        ij = np.floor((pts[:, :2] - lo) / cell).astype(np.int64)
        np.clip(ij, [0, 0], shape - 1, out=ij)
        g = np.zeros(tuple(int(v) for v in shape), dtype=bool)
        g[ij[:, 0], ij[:, 1]] = True
        return g

    def counts(pts):
        ij = np.floor((pts[:, :2] - lo) / cell).astype(np.int64)
        np.clip(ij, [0, 0], shape - 1, out=ij)
        flat = ij[:, 0] * shape[1] + ij[:, 1]
        return np.bincount(flat, minlength=int(np.prod(shape))
                           ).reshape(tuple(int(v) for v in shape))

    # Frame on where the ground truth is *dense*, not merely present. A
    # terrestrial scanner returns something a hundred metres out, but a thin
    # horizontal slice through those grazing-angle returns is nearly empty, so
    # including them buys black pixels and shrinks the part worth looking at.
    gt_n = counts(gt_pts)
    occupied = gt_n[gt_n > 0]
    if occupied.size == 0:
        raise RuntimeError("the ground truth has no points in this extent")
    dense = gt_n >= max(np.percentile(occupied, 75), 2)
    both = dense & (counts(slam_pts) > 0)
    if not both.any():
        both = dense
    if not both.any():
        raise RuntimeError("the two clouds share no ground area — is the "
                           "transform right?")
    ij = np.argwhere(both)
    lo_o = lo + np.percentile(ij, 1, axis=0) * cell - margin
    hi_o = lo + (np.percentile(ij, 99, axis=0) + 1) * cell + margin
    return lo_o, hi_o


def slice_raster(path, bands, lo, hi, cell, transform=None, verbose=True):
    """Top-down count rasters for several z bands, in one streaming pass.

    `bands` is a list of (z_lo, z_hi). Returns a list of 2-D count arrays.
    """
    nx = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    grids = [np.zeros((nx, ny), dtype=np.int32) for _ in bands]
    T = None if transform is None else np.asarray(transform, dtype=np.float64)

    tick = lasio.Ticker(enabled=verbose)
    seen = 0
    total = lasio.point_count(path)
    for xyz in lasio.iter_xyz(path):
        seen += len(xyz)
        if T is not None:
            xyz = xyz @ T[:3, :3].T + T[:3, 3]
        inxy = ((xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
                (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]))
        if not inxy.any():
            continue
        sub = xyz[inxy]
        ij = np.floor((sub[:, :2] - lo) / cell).astype(np.int64)
        np.clip(ij, [0, 0], [nx - 1, ny - 1], out=ij)
        flat = ij[:, 0] * ny + ij[:, 1]
        for grid, (z0, z1) in zip(grids, bands):
            band = (sub[:, 2] >= z0) & (sub[:, 2] < z1)
            if band.any():
                grid += np.bincount(flat[band], minlength=nx * ny
                                    ).reshape(nx, ny).astype(np.int32)
        tick(f"    slicing {seen:,} / {total:,} points")
    tick.done(f"    sliced {seen:,} points")
    return grids


def render(slam_path, gt_path, out_png, transform=None, heights=None,
           thickness=0.20, cell=None, title="", full_extent=False,
           centre=None, size=None, stations=None, verbose=True):
    """Write one figure with a row per height: GT magenta, SLAM green.

    `heights` are absolute z in the ground truth frame. If omitted they are
    placed at 0.5 / 1.5 / 3.0 m above the ground truth's own floor level, which
    is meaningful because the RTC cloud is levelled.

    `stations` is an optional list of sidecar entries; each origin is drawn on
    every slice. Worth having: a misregistration shows up as scanner positions
    landing inside walls or outside the building, which is easier to judge at a
    glance than the magenta/green overlap alone, and the scanners are also the
    one thing in the figure whose true position is known exactly.
    """
    if verbose:
        print("    sampling both clouds to find their overlap ...")
    gt_sample = lasio.stream_voxel_downsample(gt_path, 0.5, verbose=False)
    slam_sample = lasio.stream_voxel_downsample(slam_path, 0.5, verbose=False)
    if transform is not None:
        T = np.asarray(transform, dtype=np.float64)
        slam_sample = slam_sample @ T[:3, :3].T + T[:3, 3]
    if centre is not None and size is not None:
        c = np.asarray(centre, dtype=np.float64)
        lo, hi = c - size / 2.0, c + size / 2.0
    elif full_extent:
        lo = np.minimum(gt_sample[:, :2].min(axis=0), slam_sample[:, :2].min(axis=0))
        hi = np.maximum(gt_sample[:, :2].max(axis=0), slam_sample[:, :2].max(axis=0))
    else:
        lo, hi = _overlap_extent(gt_sample, slam_sample)
    span = hi - lo
    if cell is None:
        cell = max(float(span.max()) / MAX_RASTER, 0.02)

    if heights is None:
        core = np.all((gt_sample[:, :2] >= lo) & (gt_sample[:, :2] <= hi), axis=1)
        ground = float(np.percentile(gt_sample[core, 2], 1.0))
        heights = [ground + 0.5, ground + 1.5, ground + 3.0]
        if verbose:
            print(f"    GT floor level {ground:.2f} m -> slices at "
                  f"{', '.join(f'{h:.2f}' for h in heights)} m")

    bands = [(h - thickness / 2.0, h + thickness / 2.0) for h in heights]
    if verbose:
        print(f"    extent {np.round(lo, 1)} .. {np.round(hi, 1)} "
              f"at {cell * 100:.1f} cm cells")
        print("    rasterising ground truth ...")
    gt_grids = slice_raster(gt_path, bands, lo, hi, cell, None, verbose)
    if verbose:
        print("    rasterising SLAM ...")
    slam_grids = slice_raster(slam_path, bands, lo, hi, cell, transform, verbose)

    n = len(heights)
    fig, axes = plt.subplots(n, 1, figsize=(14, 6.5 * n), squeeze=False)
    extent_xy = [lo[0], lo[0] + gt_grids[0].shape[0] * cell,
                 lo[1], lo[1] + gt_grids[0].shape[1] * cell]

    for row, (h, gtg, smg) in enumerate(zip(heights, gt_grids, slam_grids)):
        ax = axes[row][0]
        # Presence, not density: a wall must read the same whether it was
        # scanned from 2 m or 30 m away.
        g = (gtg > 0).T.astype(np.float32)
        s = (smg > 0).T.astype(np.float32)
        rgb = np.zeros(g.shape + (3,), dtype=np.float32)
        rgb[..., 0] = g            # magenta = GT
        rgb[..., 2] = g
        rgb[..., 1] = s            # green   = SLAM
        ax.imshow(rgb, origin="lower", extent=extent_xy, interpolation="nearest")
        both = int(((g > 0) & (s > 0)).sum())
        gt_only = int(((g > 0) & (s == 0)).sum())
        sl_only = int(((g == 0) & (s > 0)).sum())
        agree = 100.0 * both / max(both + gt_only + sl_only, 1)
        ax.set_title(f"z = {h:.2f} m  ({thickness * 100:.0f} cm slice)   "
                     f"cells: both {both:,} | GT only {gt_only:,} | "
                     f"SLAM only {sl_only:,}   overlap {agree:.1f} %")
        if stations:
            org = np.array([st["origin"] for st in stations], dtype=np.float64)
            vis_st = ((org[:, 0] >= lo[0]) & (org[:, 0] <= hi[0]) &
                      (org[:, 1] >= lo[1]) & (org[:, 1] <= hi[1]))
            if vis_st.any():
                ax.scatter(org[vis_st, 0], org[vis_st, 1], s=70, marker="x",
                           c="yellow", linewidths=1.6, zorder=5)
                for st, o in zip(np.asarray(stations, dtype=object)[vis_st],
                                 org[vis_st]):
                    ax.annotate(str(st.get("id", "")), (o[0], o[1]),
                                textcoords="offset points", xytext=(5, 4),
                                fontsize=7, color="yellow", zorder=6)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          color=c, label=l)
               for c, l in (("magenta", "ground truth only"),
                            ("lime", "SLAM only"),
                            ("white", "both (registered)"))]
    if stations:
        handles.append(plt.Line2D([], [], marker="x", linestyle="", markersize=9,
                                  color="yellow", label="scanner setup"))
    leg = axes[0][0].legend(handles=handles, loc="upper right", fontsize=9)
    leg.get_frame().set_facecolor("black")
    for text in leg.get_texts():          # matplotlib 3.1 has no labelcolor=
        text.set_color("white")
    fig.suptitle(title or "Registration check — horizontal slices", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=120, facecolor="black")
    plt.close(fig)
    if verbose:
        print(f"    wrote {out_png}")
    return out_png
