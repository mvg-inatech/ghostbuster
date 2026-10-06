#!/usr/bin/env python3
"""
Chunked LAS I/O for point clouds that do not fit in RAM.

The ground truth clouds in this project reach several hundred million points
(a 343 M-point reference: 343 M points / 6.9 GB), so every GT-side operation streams
the file and keeps only a bounded summary in memory. The SLAM clouds are small
enough (~15 M points) to be held whole, which is what the rest of the pipeline
assumes for the estimate side.

Nothing here depends on the confidence channels — this is pure geometry I/O.
"""

import os
import sys
import time

import numpy as np
import laspy

# 20 M points x 24 B (float64 xyz) = 480 MB per chunk, comfortably inside the
# memory budget while keeping the number of passes low.
DEFAULT_CHUNK = 20_000_000


class Ticker:
    """Progress reporting that behaves in both a terminal and a log file.

    Interactively it rewrites one line; redirected to a file it emits a line
    every `interval` seconds, so a long GT pass leaves a readable trail instead
    of a wall of carriage returns.
    """

    def __init__(self, interval=15.0, enabled=True):
        self.interval = interval
        self.enabled = enabled
        self.tty = sys.stdout.isatty()
        self.last = 0.0

    def __call__(self, msg):
        if not self.enabled:
            return
        if self.tty:
            print(msg, end="\r", flush=True)
        else:
            now = time.time()
            if now - self.last >= self.interval:
                self.last = now
                print(msg, flush=True)

    def done(self, msg):
        if self.enabled:
            print(msg + (" " * 20 if self.tty else ""), flush=True)


# --------------------------------------------------------------------------- #
# Headers and bounds
# --------------------------------------------------------------------------- #
def is_pcd(path):
    """PCD files are read by gtlabel.pcd; everything else goes through laspy.

    The pipeline stages call the helpers below rather than laspy directly, so
    pointing a config at `global_map.pcd` straight out of VoxelSLAM works
    without a conversion step.
    """
    return str(path).lower().endswith(".pcd")


def read_header(path):
    """Return the LAS header without loading any points."""
    with laspy.open(path) as f:
        return f.header


def bounds(path):
    """Return (mins, maxs) as float64 (3,) arrays.

    Free for LAS, where the header carries them; a full pass for PCD, which
    does not — that result is cached in a sidecar file.
    """
    if is_pcd(path):
        from . import pcd as pcd_mod
        return pcd_mod.bounds(path)
    h = read_header(path)
    return np.asarray(h.mins, dtype=np.float64), np.asarray(h.maxs, dtype=np.float64)


def point_count(path):
    if is_pcd(path):
        from . import pcd as pcd_mod
        return pcd_mod.point_count(path)
    return read_header(path).point_count


# --------------------------------------------------------------------------- #
# Streaming reads
# --------------------------------------------------------------------------- #
def iter_xyz(path, chunk_size=DEFAULT_CHUNK, origin=None):
    """Yield (N, 3) float64 XYZ blocks.

    If `origin` is given it is subtracted from every block, which keeps
    coordinates small enough to be stored as float32 downstream without
    losing sub-millimetre precision on projected/UTM-style coordinates.
    """
    if is_pcd(path):
        from . import pcd as pcd_mod
        for block in pcd_mod.iter_xyz(path, chunk_size, origin):
            yield block
        return

    with laspy.open(path) as f:
        for chunk in f.chunk_iterator(chunk_size):
            xyz = np.empty((len(chunk), 3), dtype=np.float64)
            xyz[:, 0] = chunk.x
            xyz[:, 1] = chunk.y
            xyz[:, 2] = chunk.z
            if origin is not None:
                xyz -= origin
            yield xyz


def iter_xyz_fields(path, fields, chunk_size=DEFAULT_CHUNK, origin=None):
    """Yield (xyz, {name: array}) blocks, carrying extra per-point dimensions.

    Same streaming contract as `iter_xyz`, but each block also brings the named
    extra dimensions along. Used by the visibility model, which has to know
    which station each point of a merged ground-truth cloud came from.
    """
    if is_pcd(path):
        raise NotImplementedError(
            "iter_xyz_fields is LAS-only; the merged GT is always written as LAS")

    with laspy.open(path) as f:
        available = set(f.header.point_format.dimension_names)
        missing = [n for n in fields if n not in available]
        if missing:
            raise KeyError(f"{path} has no dimension(s) {missing}; "
                           f"available: {sorted(available)}")
        for chunk in f.chunk_iterator(chunk_size):
            xyz = np.empty((len(chunk), 3), dtype=np.float64)
            xyz[:, 0] = chunk.x
            xyz[:, 1] = chunk.y
            xyz[:, 2] = chunk.z
            if origin is not None:
                xyz -= origin
            yield xyz, {n: np.asarray(chunk[n]) for n in fields}


def has_field(path, name):
    """True if the LAS carries an extra dimension called `name`."""
    if is_pcd(path):
        return False
    with laspy.open(path) as f:
        return name in set(f.header.point_format.dimension_names)


def read_xyz(path):
    """Load the full cloud as (N, 3) float64. Only for the small SLAM clouds."""
    las = laspy.read(path)
    return np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)],
                    axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# Voxel key packing
# --------------------------------------------------------------------------- #
class VoxelLattice:
    """Maps XYZ onto a fixed integer lattice and packs indices into int64 keys.

    Used for streaming voxel downsampling and for coarse occupancy tests. Points
    outside the lattice get key -1 rather than wrapping onto a valid cell.
    """

    def __init__(self, origin, voxel, shape):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel = float(voxel)
        self.shape = np.asarray(shape, dtype=np.int64)
        nx, ny, nz = (int(v) for v in self.shape)
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"degenerate lattice shape {self.shape}")
        # int64 has 63 usable bits; refuse silently-wrapping lattices.
        if float(nx) * float(ny) * float(nz) >= 2.0 ** 62:
            raise ValueError(
                f"lattice {nx}x{ny}x{nz} at voxel={voxel} exceeds int64 packing; "
                "use a coarser voxel size")
        self._stride_y = np.int64(nz)
        self._stride_x = np.int64(ny) * np.int64(nz)

    @classmethod
    def from_bounds(cls, mins, maxs, voxel, pad=2):
        """Lattice covering [mins, maxs] with `pad` spare cells on each side."""
        mins = np.asarray(mins, dtype=np.float64)
        maxs = np.asarray(maxs, dtype=np.float64)
        origin = mins - pad * voxel
        shape = np.floor((maxs - origin) / voxel).astype(np.int64) + pad + 1
        return cls(origin, voxel, shape)

    def indices(self, xyz):
        """Return (N, 3) int64 lattice indices (may fall outside the lattice)."""
        return np.floor((xyz - self.origin) / self.voxel).astype(np.int64)

    def keys(self, xyz):
        """Return (N,) int64 packed keys; -1 for points outside the lattice."""
        ijk = self.indices(xyz)
        inside = np.all((ijk >= 0) & (ijk < self.shape), axis=1)
        keys = np.full(len(ijk), -1, dtype=np.int64)
        ij = ijk[inside]
        keys[inside] = (ij[:, 0] * self._stride_x
                        + ij[:, 1] * self._stride_y
                        + ij[:, 2])
        return keys


def stream_voxel_downsample(path, voxel, chunk_size=DEFAULT_CHUNK, verbose=True):
    """Voxel-downsample a LAS file without loading it.

    Keeps one representative point per occupied voxel (the first one seen)
    rather than the voxel centre, so the result carries no quantisation offset.

    Returns (M, 3) float64.
    """
    mins, maxs = bounds(path)
    lattice = VoxelLattice.from_bounds(mins, maxs, voxel)
    total = point_count(path)

    acc_keys = np.empty(0, dtype=np.int64)
    acc_pts = np.empty((0, 3), dtype=np.float64)
    seen = 0
    tick = Ticker(enabled=verbose)

    for xyz in iter_xyz(path, chunk_size):
        seen += len(xyz)
        keys = lattice.keys(xyz)
        valid = keys >= 0
        keys, xyz = keys[valid], xyz[valid]

        uniq, idx = np.unique(keys, return_index=True)
        acc_keys = np.concatenate([acc_keys, uniq])
        acc_pts = np.concatenate([acc_pts, xyz[idx]])
        # collapse duplicates introduced by the merge
        uniq, idx = np.unique(acc_keys, return_index=True)
        acc_keys, acc_pts = uniq, acc_pts[idx]

        tick(f"    {seen:,} / {total:,} points -> {len(acc_keys):,} voxels")

    tick.done(f"    {seen:,} points -> {len(acc_keys):,} voxels at {voxel:.3f} m")
    return acc_pts


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def add_or_overwrite(las, name, values, dtype):
    """Add an extra dimension, or overwrite it if the name already exists."""
    if name in list(las.point_format.dimension_names):
        las[name] = values.astype(dtype)
    else:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        las[name] = values.astype(dtype)


def write_las_with_fields(src_path, out_path, fields, transform=None,
                          chunk_size=DEFAULT_CHUNK, verbose=True, drop=None):
    """Copy a LAS, adding extra dimensions, without loading it into memory.

    `fields` maps name -> (full-length array, numpy dtype). `transform` is an
    optional 4x4 applied to the coordinates on the way through. `drop` is an
    iterable of existing extra-dimension names to leave out of the copy, for
    when the input carries scaffolding the result does not need.

    The in-memory route (laspy.read, add dims, write) needs roughly twice the
    file size in RAM, which is fine for a 1 GB indoor cloud and not fine for the
    12.8 GB outdoor one.

    A PCD source is converted to LAS on the way out, so the labelled result is
    always a LAS regardless of what went in.

    Writing back onto the source path is safe: the data is staged beside it and
    swapped in at the end. Doing it directly would truncate the file that is
    still being streamed, which destroys it — the reader gets an empty buffer on
    its next chunk and the original contents are already gone.
    """
    in_place = os.path.abspath(src_path) == os.path.abspath(out_path)
    if in_place:
        target = out_path + ".partial"
        if verbose:
            print(f"    in-place write: staging via {os.path.basename(target)}")
    else:
        target = out_path

    try:
        written = _write_las_impl(src_path, target, fields, transform,
                                  chunk_size, verbose, drop)
    except BaseException:
        if in_place and os.path.exists(target):
            os.remove(target)          # never leave a half-written stand-in
        raise
    if in_place:
        os.replace(target, out_path)
    return written


def _write_las_impl(src_path, out_path, fields, transform, chunk_size, verbose,
                    drop=None):
    if is_pcd(src_path):
        from . import pcd as pcd_mod
        if drop:
            raise NotImplementedError("drop= is only supported for LAS input")
        return pcd_mod.write_las(src_path, out_path, fields=fields,
                                 transform=transform, chunk_size=chunk_size,
                                 verbose=verbose)

    src_header = read_header(src_path)
    total = src_header.point_count
    for name, (values, _) in fields.items():
        if len(values) != total:
            raise ValueError(f"field '{name}' has {len(values):,} values, "
                             f"cloud has {total:,} points")

    drop = set(drop or ())
    if drop:
        src_pf = src_header.point_format
        unknown = drop - {d.name for d in src_pf.extra_dimensions}
        if unknown:
            raise ValueError(f"drop= names not in {src_path}: {sorted(unknown)}")
        out_header = laspy.LasHeader(version=src_header.version,
                                     point_format=laspy.PointFormat(src_pf.id))
        for d in src_pf.extra_dimensions:
            if d.name not in drop:
                out_header.add_extra_dim(laspy.ExtraBytesParams(
                    name=d.name, type=src_pf.dimension_by_name(d.name).dtype,
                    description=d.description))
        if verbose:
            print(f"    dropping {len(drop)} extra dimension(s) from the copy")
    else:
        out_header = laspy.LasHeader(version=src_header.version,
                                     point_format=src_header.point_format)
    out_header.scales = src_header.scales
    out_header.offsets = src_header.offsets
    if transform is not None:
        T = np.asarray(transform, dtype=np.float64)
        corners = np.array(np.meshgrid(*zip(src_header.mins, src_header.maxs))
                           ).reshape(3, -1).T
        moved = corners @ T[:3, :3].T + T[:3, 3]
        # Offsets must be set before any coordinate is written: LAS stores XYZ as
        # int32 relative to the offset, and a stale offset silently overflows.
        out_header.offsets = np.floor(moved.min(axis=0))

    existing = list(src_header.point_format.dimension_names)
    for name, (values, dtype) in fields.items():
        if name not in existing:
            out_header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

    src_dims = [d for d in existing if d not in ("X", "Y", "Z") and d not in drop]
    tick = Ticker(enabled=verbose)
    at = 0
    with laspy.open(src_path) as reader, \
            laspy.open(out_path, mode="w", header=out_header) as writer:
        for chunk in reader.chunk_iterator(chunk_size):
            n = len(chunk)
            record = laspy.ScaleAwarePointRecord.zeros(n, header=out_header)
            for dim in src_dims:
                record[dim] = chunk[dim]

            xyz = np.empty((n, 3), dtype=np.float64)
            xyz[:, 0] = chunk.x
            xyz[:, 1] = chunk.y
            xyz[:, 2] = chunk.z
            if transform is not None:
                xyz = xyz @ T[:3, :3].T + T[:3, 3]
            record.x = xyz[:, 0]
            record.y = xyz[:, 1]
            record.z = xyz[:, 2]

            for name, (values, dtype) in fields.items():
                record[name] = values[at:at + n].astype(dtype)

            writer.write_points(record)
            at += n
            tick(f"    writing: {at:,} / {total:,} points")
    tick.done(f"    wrote {at:,} points -> {out_path}")
    return at


def set_xyz(las, xyz):
    """Replace the coordinates of an in-memory LasData.

    Header offsets are updated *before* the assignment: LAS stores coordinates
    as int32 relative to the offset, so writing registered coordinates against
    a stale offset can silently overflow.
    """
    mins = xyz.min(axis=0)
    las.header.offsets = mins
    las.header.mins = mins
    las.header.maxs = xyz.max(axis=0)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
