#!/usr/bin/env python3
"""
QA renders for the labelling pipeline.

The point of these is to keep a human in the loop as a reviewer rather than an
operator: instead of driving CloudCompare for every run, you look at three
images and decide whether the registration and the domain mask are sane.

Everything is rendered as top-down rasters rather than 3D screenshots — Open3D's
offscreen rendering is unreliable headless, and a raster is more readable for
"where did the mask cut" anyway.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from .domain import (IN_DOMAIN, OUT_NO_GT, OUT_FOOTPRINT, OUT_Z, OUT_EXCLUDED,
                     OUT_NOT_VISIBLE, CODE_NAMES)

CLASS_COLORS = {
    IN_DOMAIN: "#2c7fb8",
    OUT_NO_GT: "#d95f02",
    OUT_FOOTPRINT: "#7570b3",
    OUT_Z: "#666666",
    OUT_EXCLUDED: "#e7298a",
    OUT_NOT_VISIBLE: "#1b9e77",
}


def _raster(xy, values, cell, reduce="mean"):
    """Aggregate scattered values onto a top-down grid. Empty cells are NaN."""
    mins = xy.min(axis=0)
    shape = np.maximum(np.ceil((xy.max(axis=0) - mins) / cell).astype(np.int64) + 1, 1)
    ij = np.floor((xy - mins) / cell).astype(np.int64)
    np.clip(ij, [0, 0], [shape[0] - 1, shape[1] - 1], out=ij)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    size = int(shape[0] * shape[1])

    counts = np.bincount(flat, minlength=size).astype(np.float64)
    if reduce == "count":
        grid = counts
    else:
        sums = np.bincount(flat, weights=values, minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = sums / counts
    grid[counts == 0] = np.nan
    extent = [mins[0], mins[0] + shape[0] * cell, mins[1], mins[1] + shape[1] * cell]
    return grid.reshape(int(shape[0]), int(shape[1])).T, extent


def _majority_raster(xy, codes, cell):
    """Per-cell dominant domain code, for the mask overview."""
    mins = xy.min(axis=0)
    shape = np.maximum(np.ceil((xy.max(axis=0) - mins) / cell).astype(np.int64) + 1, 1)
    ij = np.floor((xy - mins) / cell).astype(np.int64)
    np.clip(ij, [0, 0], [shape[0] - 1, shape[1] - 1], out=ij)
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    size = int(shape[0] * shape[1])

    per_code = np.stack([np.bincount(flat[codes == c], minlength=size)
                         for c in sorted(CODE_NAMES)], axis=1)
    total = per_code.sum(axis=1)
    winner = np.argmax(per_code, axis=1).astype(np.float64)
    winner[total == 0] = np.nan
    extent = [mins[0], mins[0] + shape[0] * cell, mins[1], mins[1] + shape[1] * cell]
    return winner.reshape(int(shape[0]), int(shape[1])).T, extent


def render(slam_xyz, c2c, code, out_dir, base, cell=0.10, tau=0.05,
           reg_info=None, verbose=True):
    """Write the QA figures. Returns the list of paths written."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    in_dom = code == IN_DOMAIN
    finite = np.isfinite(c2c)

    # ---------------------------------------------------------------- domain
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    codes_sorted = sorted(CODE_NAMES)
    grid, extent = _majority_raster(slam_xyz[:, :2], code, cell)
    cmap = ListedColormap([CLASS_COLORS[c] for c in codes_sorted])
    norm = BoundaryNorm(np.arange(len(codes_sorted) + 1) - 0.5, len(codes_sorted))
    ax = axes[0, 0]
    ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest")
    ax.set_title(f"Evaluation domain (dominant class per {cell:.2f} m cell)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=10,
                          color=CLASS_COLORS[c], label=CODE_NAMES[c])
               for c in codes_sorted]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax = axes[0, 1]
    if in_dom.any():
        vals = np.clip(c2c[in_dom & finite], 0, 4 * tau)
        grid, extent = _raster(slam_xyz[in_dom & finite, :2], vals, cell)
        im = ax.imshow(grid, origin="lower", extent=extent, cmap="viridis",
                       interpolation="nearest", vmin=0, vmax=4 * tau)
        plt.colorbar(im, ax=ax, label="mean C2C [m]")
    ax.set_title(f"Mean C2C, in-domain points only (clipped at {4 * tau * 100:.0f} cm)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")

    ax = axes[1, 0]
    bins = np.linspace(0, max(4 * tau, 0.05), 120)
    for c in codes_sorted:
        sel = (code == c) & finite
        if sel.sum() == 0:
            continue
        # sel excludes points with no GT neighbour at all (C2C = inf), so this
        # count is below the class total in the composition panel.
        ax.hist(np.clip(c2c[sel], bins[0], bins[-1]), bins=bins, histtype="step",
                linewidth=1.5, color=CLASS_COLORS[c],
                label=f"{CODE_NAMES[c]}  (n={int(sel.sum()):,} with a GT neighbour)")
    ax.axvline(tau, color="red", linestyle="--", linewidth=1,
               label=f"tau = {tau * 100:.0f} cm")
    ax.set_yscale("log")
    ax.set_xlabel("C2C distance [m]"); ax.set_ylabel("points")
    ax.set_title("C2C distribution by domain class")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    counts = [int((code == c).sum()) for c in codes_sorted]
    fracs = [100.0 * v / len(code) for v in counts]
    ax.barh([CODE_NAMES[c] for c in codes_sorted], fracs,
            color=[CLASS_COLORS[c] for c in codes_sorted])
    for i, (f, n) in enumerate(zip(fracs, counts)):
        ax.text(f, i, f"  {f:.1f}%  ({n:,})", va="center", fontsize=9)
    ax.set_xlabel("share of SLAM points [%]")
    ax.set_xlim(0, max(fracs) * 1.35 if max(fracs) > 0 else 1)
    ax.set_title("Domain composition")

    title = f"Domain mask QA — {base}"
    if reg_info:
        title += (f"\nICP fitness={reg_info.get('fitness', float('nan')):.4f}  "
                  f"inlier RMSE={reg_info.get('inlier_rmse', float('nan')) * 100:.2f} cm")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"{base}_qa_domain.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    # ---------------------------------------------------------- registration
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sel = in_dom & finite
    ax = axes[0]
    if sel.any():
        ax.hist(c2c[sel], bins=200, color="#2c7fb8")
        ax.axvline(tau, color="red", linestyle="--", label=f"tau = {tau * 100:.0f} cm")
        ax.axvline(float(np.median(c2c[sel])), color="black", linestyle=":",
                   label=f"median = {np.median(c2c[sel]) * 100:.2f} cm")
        ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("C2C distance [m]"); ax.set_ylabel("points")
    ax.set_title("In-domain C2C histogram")

    ax = axes[1]
    if sel.any():
        s = np.sort(c2c[sel])
        ax.plot(s, np.linspace(0, 1, len(s)), color="#2c7fb8")
        for t in (0.01, 0.02, 0.05, 0.10):
            if t <= s[-1]:
                ax.axvline(t, color="grey", linewidth=0.7, linestyle=":")
                ax.text(t, 0.02, f" {t * 100:.0f} cm", fontsize=7, rotation=90)
    ax.set_xlim(0, max(4 * tau, 0.05))
    ax.set_xlabel("C2C distance [m]"); ax.set_ylabel("cumulative fraction")
    ax.set_title("In-domain C2C CDF")

    fig.suptitle(f"Registration QA — {base}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, f"{base}_qa_registration.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    if verbose:
        for p in written:
            print(f"    wrote {p}")
    return written
