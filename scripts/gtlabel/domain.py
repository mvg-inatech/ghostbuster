#!/usr/bin/env python3
"""
Evaluation-domain mask: which SLAM points the RTC could plausibly have scanned.

This replaces the manual "delete the areas the scanner never saw" step, and it
deliberately replaces the earlier C2C + mean_knn_dist rule as well. That rule
dropped points with C2C > 4 cm and mean_knn_dist < 1.5 cm, i.e. exactly the
points where the kNN channel disagreed with the label — which mechanically
inflates the kNN AUC, and inflates it more for the with-kNN model than the
without-kNN model. Since the headline experiment is precisely the gap between
those two, the rule contaminated the result it was meant to support.

Everything here is computed from GT geometry and point position only. No
confidence channel is involved, so no channel is favoured.

Two gates, coarse to fine:

1. Footprint (regional). A 2D occupancy raster of the GT, dilated. Answers
   "did the scanner visit this part of the building at all", and is what
   rejects an adjacent room the SLAM leaked into through a doorway. Derived
   from the GT automatically; a hand-drawn polygon can be supplied instead for
   datasets where the automatic version misfires.

2. Coverage radius (local). No GT point within `radius` of the SLAM point.
   Answers "was this particular surface resolved", and catches occlusion
   shadows behind furniture that the footprint is far too coarse to see.
   Reuses the C2C distance that is computed anyway, so it costs nothing.

3. Exclusion volumes (optional, one-time). Hand-specified boxes or prisms.

Why gate 3 exists — measured, not assumed. On an indoor scene the manual deletion
removed 608 k points concentrated in slabs at door and window openings. Those
points sit 5-30 cm off the wall plane, i.e. *inside* the coverage radius, so
gate 2 keeps them. Geometry alone cannot tell them apart from a genuine SLAM
error at a scanned wall: both are "a point 15 cm from a GT surface". Only a
visibility model built from the scanner station positions can separate the two,
by asking whether the RTC could see that spot at all.

Until station data is available, openings are handled as explicit exclusion
volumes. They do not move between runs, so they are declared once in the
dataset config and versioned with it, so the human judgement is kept while
the per-run manual step is not.
"""

import os

import numpy as np
from scipy import ndimage

from . import lasio

# Per-point classification codes written to the output LAS as `domain_code`.
IN_DOMAIN = 0
OUT_NO_GT = 1          # no GT point within the coverage radius
OUT_FOOTPRINT = 2      # outside the scanned footprint
OUT_Z = 3              # outside the configured height band
OUT_EXCLUDED = 4       # inside a declared exclusion volume
OUT_NOT_VISIBLE = 5    # no station could have seen this spot

CODE_NAMES = {
    IN_DOMAIN: "in domain",
    OUT_NO_GT: "no GT within radius",
    OUT_FOOTPRINT: "outside footprint",
    OUT_Z: "outside z range",
    OUT_EXCLUDED: "exclusion volume",
    OUT_NOT_VISIBLE: "not visible to any station",
}


class ExclusionVolumes:
    """Declared regions to drop from the evaluation domain.

    Three shapes, all given once per dataset in the config:

      boxes:    [xmin, ymin, zmin, xmax, ymax, zmax]
      prisms:   {polygon: [[x, y], ...], z_range: [zmin, zmax]}
      voxels:   path to an .npz voxel mask, one bool per occupied cell

    Intended for openings — doorways, windows, hatches — where the SLAM sees
    into space the scanner never covered, and where the resulting points land
    too close to a real GT surface for the coverage radius to reject them.

    Prefer `voxels` when the volumes come from a hand-cleaned cloud: a deletion
    around an opening is a hollow shell, and its bounding box also contains the
    wall around it. In one measured case the box form removed an extra 2.9% of the cloud
    that the original selection had kept; the voxel form is exact.
    """

    def __init__(self, boxes=None, prisms=None, voxels=None):
        self.boxes = [np.asarray(b, dtype=np.float64) for b in (boxes or [])]
        for b in self.boxes:
            if b.shape != (6,):
                raise ValueError(f"exclusion box must have 6 values, got {b.shape}")
        self.prisms = prisms or []
        self.voxel_masks = []
        for path in ([voxels] if isinstance(voxels, str) else (voxels or [])):
            data = np.load(path)
            self.voxel_masks.append({
                "origin": data["origin"].astype(np.float64),
                "cell": float(data["cell"]),
                "keys": np.sort(_pack_ijk(data["ijk"].astype(np.int64))),
                "path": path,
            })

    def __len__(self):
        return len(self.boxes) + len(self.prisms) + len(self.voxel_masks)

    def describe(self):
        parts = []
        if self.boxes:
            parts.append(f"{len(self.boxes)} box(es)")
        if self.prisms:
            parts.append(f"{len(self.prisms)} prism(s)")
        for vm in self.voxel_masks:
            parts.append(f"{len(vm['keys']):,} voxels at {vm['cell']:.2f} m "
                         f"({vm['path']})")
        return ", ".join(parts)

    def contains(self, xyz):
        """Boolean mask of points inside any declared volume."""
        hit = np.zeros(len(xyz), dtype=bool)
        for b in self.boxes:
            hit |= np.all((xyz >= b[:3]) & (xyz <= b[3:]), axis=1)

        if self.prisms:
            from matplotlib.path import Path
            for prism in self.prisms:
                poly = np.asarray(prism["polygon"], dtype=np.float64)
                zlo, zhi = prism.get("z_range", (-np.inf, np.inf))
                band = (xyz[:, 2] >= zlo) & (xyz[:, 2] <= zhi)
                if band.any():
                    inside = Path(poly).contains_points(xyz[band, :2])
                    hit[np.flatnonzero(band)[inside]] = True

        for vm in self.voxel_masks:
            ijk = np.floor((xyz - vm["origin"]) / vm["cell"]).astype(np.int64)
            keys = _pack_ijk(ijk)
            pos = np.searchsorted(vm["keys"], keys)
            np.clip(pos, 0, len(vm["keys"]) - 1, out=pos)
            hit |= vm["keys"][pos] == keys

        return hit


# Voxel indices are packed with a large offset so that negative indices (points
# outside the mask's own extent) stay distinct rather than aliasing onto valid
# cells.
_VOX_OFFSET = np.int64(1 << 20)
_VOX_STRIDE = np.int64(1 << 21)


def _pack_ijk(ijk):
    ijk = np.asarray(ijk, dtype=np.int64) + _VOX_OFFSET
    if ijk.min() < 0 or ijk.max() >= _VOX_STRIDE:
        ijk = np.clip(ijk, 0, _VOX_STRIDE - 1)
    return (ijk[:, 0] * _VOX_STRIDE + ijk[:, 1]) * _VOX_STRIDE + ijk[:, 2]


class Footprint:
    """2D occupancy raster of the GT cloud in the GT coordinate frame."""

    def __init__(self, mask, origin, cell):
        self.mask = mask
        self.origin = np.asarray(origin, dtype=np.float64)
        self.cell = float(cell)

    def contains(self, xy):
        ij = np.floor((np.asarray(xy) - self.origin) / self.cell).astype(np.int64)
        inside = ((ij[:, 0] >= 0) & (ij[:, 0] < self.mask.shape[0]) &
                  (ij[:, 1] >= 0) & (ij[:, 1] < self.mask.shape[1]))
        out = np.zeros(len(ij), dtype=bool)
        out[inside] = self.mask[ij[inside, 0], ij[inside, 1]]
        return out

    @property
    def area(self):
        return float(self.mask.sum()) * self.cell ** 2


def build_footprint(gt_path, cell=0.25, dilate=1.0, min_points=4,
                    fill_holes=True, verbose=True):
    """Build the scanned-footprint raster by streaming the GT once.

    `min_points` rejects isolated stray GT returns, `dilate` grows the footprint
    so that genuine gaps at the edge of the scanned area are not clipped away.
    """
    mins, maxs = lasio.bounds(gt_path)
    span = maxs[:2] - mins[:2]
    shape = np.maximum(np.ceil(span / cell).astype(np.int64) + 1, 1)
    counts = np.zeros((int(shape[0]), int(shape[1])), dtype=np.int64)

    total = lasio.point_count(gt_path)
    seen = 0
    tick = lasio.Ticker(enabled=verbose)
    for xyz in lasio.iter_xyz(gt_path):
        seen += len(xyz)
        ij = np.floor((xyz[:, :2] - mins[:2]) / cell).astype(np.int64)
        np.clip(ij, [0, 0], [shape[0] - 1, shape[1] - 1], out=ij)
        flat = ij[:, 0] * shape[1] + ij[:, 1]
        counts += np.bincount(flat, minlength=int(shape[0] * shape[1])
                              ).reshape(int(shape[0]), int(shape[1]))
        tick(f"    footprint: {seen:,} / {total:,} GT points")

    mask = counts >= min_points
    raw_cells = int(mask.sum())

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if dilate > 0:
        r = int(np.ceil(dilate / cell))
        if r > 0:
            mask = ndimage.binary_dilation(mask, iterations=r)

    fp = Footprint(mask, mins[:2], cell)
    tick.done(f"    footprint: {raw_cells:,} occupied cells at {cell:.2f} m "
              f"-> {int(mask.sum()):,} after fill/dilate ({fp.area:.1f} m^2)")
    return fp


def polygon_footprint(polygon, cell=0.25):
    """Rasterise a hand-drawn polygon (list of [x, y]) into a Footprint."""
    from matplotlib.path import Path

    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("polygon must be a list of at least 3 [x, y] pairs")

    mins = poly.min(axis=0) - cell
    maxs = poly.max(axis=0) + cell
    shape = np.maximum(np.ceil((maxs - mins) / cell).astype(np.int64) + 1, 1)
    gx, gy = np.meshgrid(
        mins[0] + (np.arange(shape[0]) + 0.5) * cell,
        mins[1] + (np.arange(shape[1]) + 0.5) * cell,
        indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = Path(poly).contains_points(pts).reshape(int(shape[0]), int(shape[1]))
    return Footprint(inside, mins, cell)


def classify(slam_xyz, c2c, radius=0.30, footprint=None, z_range=None,
             exclusions=None):
    """Assign every SLAM point a domain code.

    `c2c` may contain inf (no GT neighbour inside the search cap), which counts
    as uncovered. Gates are applied coarse to fine so `domain_code` reports the
    most structural reason a point was excluded.
    """
    n = len(slam_xyz)
    code = np.full(n, IN_DOMAIN, dtype=np.uint8)

    if z_range is not None:
        zlo, zhi = z_range
        bad = (slam_xyz[:, 2] < zlo) | (slam_xyz[:, 2] > zhi)
        code[bad & (code == IN_DOMAIN)] = OUT_Z

    if footprint is not None:
        outside = ~footprint.contains(slam_xyz[:, :2])
        code[outside & (code == IN_DOMAIN)] = OUT_FOOTPRINT

    if exclusions is not None and len(exclusions):
        inside = exclusions.contains(slam_xyz)
        code[inside & (code == IN_DOMAIN)] = OUT_EXCLUDED

    uncovered = ~(np.isfinite(c2c) & (c2c < radius))
    code[uncovered & (code == IN_DOMAIN)] = OUT_NO_GT

    return code


def summarise(code):
    """Return {label: (count, fraction)} for logging and the run report."""
    n = len(code)
    out = {}
    for value, name in CODE_NAMES.items():
        c = int((code == value).sum())
        out[name] = (c, c / n if n else 0.0)
    return out


def crop_gt_to_slam_streaming(gt_path, slam_path, radius, out_path,
                              transform=None, voxel=None, verbose=True):
    """`crop_gt_to_slam` for clouds too large to hold in memory.

    The in-memory version builds a KD-tree over every SLAM point, which the
    per-station driver cannot do: it never loads the SLAM cloud, and at 194 M
    points the coordinates alone are 4.7 GB before the tree.

    Here the SLAM side is voxel-downsampled on the way in. That is sound for
    this particular query -- the question is only "is there any SLAM point
    within `radius`", so replacing a cluster of points by one representative per
    voxel can move the boundary by at most half a voxel diagonal. The default
    voxel is `radius / 10`, which bounds that at under 9% of the radius, and
    CD_sym is meant to be reported over a sweep of radii precisely because the
    exact value should not matter (see the write-up's discussion of gt_crop).

    `transform` is the group's 4x4 taking SLAM coordinates into the GT frame.

    The GT is copied chunk by chunk with its point format intact, so a merged
    cloud keeps its `station_id`; the pose sidecar is copied alongside, so the
    cropped cloud is still a valid input to the visibility model.
    """
    import json
    import shutil

    import laspy
    from scipy.spatial import cKDTree

    from . import e57 as e57_mod

    voxel = float(voxel if voxel else radius / 10.0)
    if verbose:
        print(f"  Cropping GT to within {radius:.2f} m of the SLAM cloud "
              f"(SLAM sampled at {voxel * 100:.1f} cm) ...")
    slam_xyz = lasio.stream_voxel_downsample(slam_path, voxel, verbose=verbose)
    if transform is not None:
        T = np.asarray(transform, dtype=np.float64)
        slam_xyz = slam_xyz @ T[:3, :3].T + T[:3, 3]
    if verbose:
        print(f"    {len(slam_xyz):,} SLAM voxels in the GT frame")
    tree = cKDTree(slam_xyz)

    header = lasio.read_header(gt_path)
    with laspy.open(gt_path) as fin:
        out_header = laspy.LasHeader(version=header.version,
                                     point_format=header.point_format)
        out_header.scales = header.scales
        out_header.offsets = header.offsets
        # No need to re-add the extra dimensions: LasHeader was constructed with
        # the source point format, which already carries them, so `station_id`
        # travels into the crop by itself. Adding them explicitly is not merely
        # redundant, it does not terminate -- the new header SHARES the source
        # point-format object, so appending to it extends the very list being
        # iterated, while laspy rebuilds the whole ExtraBytes VLR on each add.
        kept = total = 0
        tick = lasio.Ticker(enabled=verbose)
        with laspy.open(out_path, mode="w", header=out_header) as fout:
            for chunk in fin.chunk_iterator(lasio.DEFAULT_CHUNK):
                total += len(chunk)
                xyz = np.stack([chunk.x, chunk.y, chunk.z], axis=1)
                d, _ = tree.query(xyz, k=1, distance_upper_bound=radius,
                                  workers=-1)
                keep = np.isfinite(d)
                if keep.any():
                    fout.write_points(chunk[keep])
                    kept += int(keep.sum())
                tick(f"    {total:,} GT points read, {kept:,} kept")
    tick.done(f"    kept {kept:,} / {total:,} GT points -> {out_path}")

    src_side = e57_mod.sidecar_path(gt_path)
    if os.path.exists(src_side):
        dst_side = e57_mod.sidecar_path(out_path)
        with open(src_side) as f:
            meta = json.load(f)
        meta["cropped_from"] = os.path.abspath(gt_path)
        meta["crop_radius"] = float(radius)
        # Point counts and extents describe the uncropped cloud. The visibility
        # model only uses them to bound a grid generously, so leaving them is
        # safe; saying so here stops anyone reading them as post-crop counts.
        meta["counts_are_pre_crop"] = True
        with open(dst_side, "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"    carried the station sidecar -> {dst_side}")
    return kept


def crop_gt_to_slam(gt_path, slam_xyz, radius, out_path, chunk_origin=None,
                    verbose=True):
    """Write the GT points within `radius` of the SLAM cloud to a new LAS.

    This is the other half of the common evaluation volume: it stops GT regions
    the SLAM never visited from dominating the completeness term d(Q->P) of the
    symmetric Chamfer distance. Fix it once from the *unfiltered* SLAM cloud and
    reuse it for every filter variant, otherwise an aggressive filter can
    improve its own score by shrinking the reference.
    """
    import laspy
    from scipy.spatial import cKDTree

    if verbose:
        print(f"  Cropping GT to within {radius:.2f} m of the SLAM cloud ...")
    tree = cKDTree(slam_xyz)

    header = lasio.read_header(gt_path)
    with laspy.open(gt_path) as fin:
        out_header = laspy.LasHeader(version=header.version,
                                     point_format=header.point_format)
        out_header.scales = header.scales
        out_header.offsets = header.offsets
        kept = total = 0
        tick = lasio.Ticker(enabled=verbose)
        with laspy.open(out_path, mode="w", header=out_header) as fout:
            for chunk in fin.chunk_iterator(lasio.DEFAULT_CHUNK):
                total += len(chunk)
                xyz = np.stack([chunk.x, chunk.y, chunk.z], axis=1)
                d, _ = tree.query(xyz, k=1, distance_upper_bound=radius,
                                  workers=-1)
                keep = np.isfinite(d)
                if keep.any():
                    fout.write_points(chunk[keep])
                    kept += int(keep.sum())
                tick(f"    {total:,} GT points read, {kept:,} kept")
    tick.done(f"    kept {kept:,} / {total:,} GT points -> {out_path}")
    return kept
