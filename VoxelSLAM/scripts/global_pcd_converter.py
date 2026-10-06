#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import os
import argparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Confidence channels
#
# Each channel is a float32 array with one value per point. A channel is either
# absent (its key is missing from the dict) or fully populated. The sentinel
# below is written when a channel cannot be filled; a scan in which every value
# equals the sentinel is treated as one where the SLAM did not produce that
# channel at all.
# ---------------------------------------------------------------------------

CHANNEL_SENTINELS = {
    'intensity':         0.0,
    'balm_res':         -1.0,
    'ekf_res':          -1.0,
    'reflectivity':      0.0,
    'plane_quality':    -1.0,
    'cell_id':          -1.0,
    'obs_count':         0.0,
    'view_diversity':   -1.0,
    'temp_consistency': -1.0,
    'ambient':           0.0,
    'ring':             -1.0,
    'scan_time':        -1.0,
    'range_detach':      0.0,
    'range_edge':        0.0,
    'int_detach':        0.0,
    'ring_rough':        0.0,
    'range':            -1.0,
    'pose_unc_rot':     -1.0,
    'ang_rate':         -1.0,
}

# Channels read out of the per-scan PCDs written by VoxelSLAM. Adding a channel
# means updating three tables: this one (what is parsed), CHANNEL_SENTINELS (its
# missing value) and CHANNEL_ORDER (whether it reaches the merged file). A
# channel left out of this one is skipped silently.
PCD_CHANNELS = ('intensity', 'balm_res', 'ekf_res', 'reflectivity', 'plane_quality',
                'cell_id', 'obs_count', 'view_diversity', 'temp_consistency',
                'ambient', 'ring', 'scan_time',
                'range_detach', 'range_edge', 'int_detach', 'ring_rough')

# Per-scan channels derived from alidarState.txt and broadcast to every point of
# the scan. These are constant within a scan, so a model trained on them must use
# a spatial or dataset-wise train/test split — under a random split a booster can
# use them as a near scan-ID lookup and report an inflated AUC.
SCAN_CHANNELS = ('pose_unc_rot', 'ang_rate')

# Field order in written PCDs. 'rgb' is the only non-float field.
CHANNEL_ORDER = ('intensity', 'rgb', 'balm_res', 'ekf_res', 'reflectivity',
                 'plane_quality', 'cell_id', 'obs_count', 'view_diversity',
                 'temp_consistency', 'ambient', 'ring', 'scan_time',
                 'range_detach', 'range_edge', 'int_detach', 'ring_rough',
                 'range', 'pose_unc_rot', 'ang_rate')


def read_pcd_channels(pcd_file):
    """
    Read a PCD file and extract every supported scalar channel.
    Handles both ASCII and binary PCD formats.
    Returns (pcd, channels) where channels maps a name in PCD_CHANNELS to a
    float32 array; channels that are absent or disabled are simply not present.
    """
    pcd = o3d.io.read_point_cloud(pcd_file)
    channels = {}

    try:
        with open(pcd_file, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline()
                if line.startswith(b'DATA'):
                    data_format = line.decode('ascii').strip().split()[1]
                    header_lines.append(line.decode('ascii').strip())
                    data_start_pos = f.tell()
                    break
                else:
                    header_lines.append(line.decode('ascii').strip())

        fields_line = next((l for l in header_lines if l.startswith('FIELDS')), None)
        if fields_line is None:
            return pcd, channels

        fields = fields_line.split()[1:]
        size_line = next((l for l in header_lines if l.startswith('SIZE')), None)
        type_line = next((l for l in header_lines if l.startswith('TYPE')), None)
        count_line = next((l for l in header_lines if l.startswith('COUNT')), None)

        num_points = len(pcd.points)

        if data_format.lower() == 'ascii':
            field_indices = {f.lower(): i for i, f in enumerate(fields)}
            with open(pcd_file, 'r') as f:
                lines = f.readlines()
            data_start = next(i + 1 for i, l in enumerate(lines) if l.startswith('DATA'))
            data_lines = [l.strip().split() for l in lines[data_start:] if l.strip()]

            for name in PCD_CHANNELS:
                if name not in field_indices:
                    continue
                idx = field_indices[name]
                sentinel = CHANNEL_SENTINELS[name]
                channels[name] = np.array(
                    [float(row[idx]) if len(row) > idx else sentinel for row in data_lines],
                    dtype=np.float32)

        elif data_format.lower() == 'binary' and size_line and type_line and count_line:
            sizes = [int(value) for value in size_line.split()[1:]]
            types = type_line.split()[1:]
            counts = [int(value) for value in count_line.split()[1:]]
            channels = read_binary_pcd_fields(
                pcd_file, data_start_pos, num_points, fields, sizes, types, counts)

    except Exception as e:
        print(f"  Warning: Could not extract scalar fields from {pcd_file}: {e}")

    # Drop channels whose every value equals the disabled-field sentinel written
    # by C++ (SaveFields/<field>: 0), so they are treated the same as absent.
    return pcd, {name: values for name, values in channels.items()
                 if not _all_sentinel(values, CHANNEL_SENTINELS[name])}


def _all_sentinel(arr, sentinel):
    """True if every value equals sentinel (i.e. the field was disabled in C++)."""
    return arr is None or bool(np.all(arr == sentinel))


def read_binary_pcd_fields(pcd_file, data_start_pos, num_points,
                           fields, sizes, types, counts):
    """Read all supported fields from an interleaved binary PCD in one pass."""
    type_codes = {
        ('F', 4): '<f4', ('F', 8): '<f8',
        ('I', 1): '<i1', ('I', 2): '<i2', ('I', 4): '<i4', ('I', 8): '<i8',
        ('U', 1): '<u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8',
    }
    names = []
    formats = []
    offsets = []
    offset = 0
    for name, size, field_type, count in zip(fields, sizes, types, counts):
        base_type = type_codes.get((field_type.upper(), size))
        if base_type is None:
            raise ValueError(f"Unsupported PCD field type {field_type}{size} for {name}")
        names.append(name.lower())
        formats.append(base_type if count == 1 else (base_type, (count,)))
        offsets.append(offset)
        offset += size * count

    dtype = np.dtype({
        'names': names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': offset,
    })
    records = np.fromfile(pcd_file, dtype=dtype, count=num_points, offset=data_start_pos)
    return {
        name: np.asarray(records[name], dtype=np.float32)
        for name in names if name in PCD_CHANNELS
    }


def save_pcd_with_channels(pcd, channels, output_file):
    """
    Save a point cloud with all populated scalar channels.
    Preserves color information if available and stays CloudCompare compatible.
    """
    points = np.asarray(pcd.points)
    n = len(points)
    has_colors = n > 0 and len(pcd.colors) == n
    present = [name for name in CHANNEL_ORDER
               if name != 'rgb' and name in channels and len(channels[name]) == n]

    if not present and not has_colors:
        return o3d.io.write_point_cloud(output_file, pcd)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for name in CHANNEL_ORDER:
        if name == 'rgb':
            if has_colors:
                dtype.append(('rgb', 'u4'))
        elif name in present:
            dtype.append((name, 'f4'))

    structured_points = np.zeros(n, dtype=dtype)
    structured_points['x'] = points[:, 0].astype(np.float32)
    structured_points['y'] = points[:, 1].astype(np.float32)
    structured_points['z'] = points[:, 2].astype(np.float32)

    for name in present:
        structured_points[name] = np.nan_to_num(
            np.asarray(channels[name], dtype=np.float32), nan=CHANNEL_SENTINELS[name])

    if has_colors:
        colors = np.asarray(pcd.colors)
        r = (colors[:, 0] * 255).astype(np.uint8)
        g = (colors[:, 1] * 255).astype(np.uint8)
        b = (colors[:, 2] * 255).astype(np.uint8)
        structured_points['rgb'] = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)

    write_pcd_with_fields(structured_points, output_file)
    return True


def write_pcd_with_fields(structured_points, output_file):
    """
    Write PCD file with custom fields including intensity and RGB
    Optimized for CloudCompare compatibility
    """
    num_points = len(structured_points)
    field_names = structured_points.dtype.names

    with open(output_file, 'w') as f:
        # Write header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(field_names)}\n")

        # Write field sizes and types
        sizes = []
        types = []
        counts = []
        for name in field_names:
            dtype = structured_points.dtype.fields[name][0]
            if dtype == np.float32:
                sizes.append("4")
                types.append("F")
            elif dtype == np.uint8:
                sizes.append("1")
                types.append("U")
            elif dtype == np.uint32:
                sizes.append("4")
                types.append("U")
            else:
                sizes.append("4")
                types.append("F")
            counts.append("1")

        f.write(f"SIZE {' '.join(sizes)}\n")
        f.write(f"TYPE {' '.join(types)}\n")
        f.write(f"COUNT {' '.join(counts)}\n")
        f.write(f"WIDTH {num_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {num_points}\n")
        f.write("DATA ascii\n")

        # Write data
        for point in structured_points:
            values = []
            for name in field_names:
                val = point[name]
                if name == 'rgb':
                    # Write RGB as unsigned integer for proper CloudCompare interpretation
                    values.append(str(int(val)))
                elif isinstance(val, (np.float32, np.float64)):
                    values.append(f"{float(val)}")
                else:
                    values.append(str(int(val)))
            f.write(' '.join(values) + '\n')


def _stream_schema(available_fields):
    return ['x', 'y', 'z'] + [name for name in CHANNEL_ORDER if name in available_fields]


def _write_stream_header(output, fields, binary=False):
    """Write a fixed-width PCD header and return offsets for the final point count."""
    def write(value):
        output.write(value.encode('ascii') if binary else value)

    write("# .PCD v0.7 - Point Cloud Data file format\n")
    write("VERSION 0.7\n")
    write(f"FIELDS {' '.join(fields)}\n")
    write(f"SIZE {' '.join(['4'] * len(fields))}\n")
    write(f"TYPE {' '.join('U' if name == 'rgb' else 'F' for name in fields)}\n")
    write(f"COUNT {' '.join(['1'] * len(fields))}\n")
    write("WIDTH ")
    width_offset = output.tell()
    write("00000000000000000000\n")
    write("HEIGHT 1\n")
    write("VIEWPOINT 0 0 0 1 0 0 0\n")
    write("POINTS ")
    points_offset = output.tell()
    write("00000000000000000000\n")
    write(f"DATA {'binary' if binary else 'ascii'}\n")
    return width_offset, points_offset


def _append_stream_points(output, fields, pcd, channels, binary=False):
    """Append one transformed scan to an open PCD without retaining it."""
    points = np.asarray(pcd.points)
    point_count = len(points)

    prepared = {}
    for name in fields[3:]:
        if name == 'rgb':
            if len(pcd.colors) == point_count:
                colors = np.clip(np.asarray(pcd.colors) * 255, 0, 255).astype(np.uint8)
            else:
                colors = np.full((point_count, 3), 128, dtype=np.uint8)
            prepared['rgb'] = ((colors[:, 0].astype(np.uint32) << 16)
                               | (colors[:, 1].astype(np.uint32) << 8)
                               | colors[:, 2].astype(np.uint32))
        else:
            values = channels.get(name)
            prepared[name] = (np.asarray(values, dtype=np.float32) if values is not None
                              else np.full(point_count, CHANNEL_SENTINELS[name], dtype=np.float32))

    if binary:
        dtype = [(name, '<u4' if name == 'rgb' else '<f4') for name in fields]
        structured = np.empty(point_count, dtype=dtype)
        structured['x'] = points[:, 0]
        structured['y'] = points[:, 1]
        structured['z'] = points[:, 2]
        for name in fields[3:]:
            structured[name] = prepared[name]
        output.write(structured.tobytes())
        return

    for index, point in enumerate(points):
        values = []
        for name in fields:
            if name == 'x':
                values.append(str(float(point[0])))
            elif name == 'y':
                values.append(str(float(point[1])))
            elif name == 'z':
                values.append(str(float(point[2])))
            elif name == 'rgb':
                values.append(str(int(prepared[name][index])))
            else:
                values.append(str(float(prepared[name][index])))
        output.write(' '.join(values) + '\n')


def _first_per_id(ids, *values):
    """Unique ids plus one representative value each, without a full-size temp.

    Every channel here is constant within a leaf at a given save time, and a
    cell is finer than any leaf, so the first occurrence is the cell's value.
    """
    order = np.argsort(ids, kind="stable")
    sid = ids[order]
    first = np.empty(len(sid), dtype=bool)
    first[0] = True
    np.not_equal(sid[1:], sid[:-1], out=first[1:])
    idx = order[first]
    return ids[idx], tuple(v[idx] for v in values)


# What the merged per-cell values mean. The names are shorter than the
# definitions, so read these before using the channels:
#
#   obs_count        total scans observing the cell, summed over visits. A new
#                    visit is detected only by the count dropping, so a short
#                    visit followed by a revisit that starts higher is missed
#                    and such cells undercount.
#   view_diversity   the maximum over visits, i.e. the most diverse single
#                    visit, not the spread over all of them. The eigenvalue is
#                    not additive and the scatter matrices behind it are not
#                    saved per scan.
#   temp_consistency the value from the visit with the most scans: the spread of
#                    |point-to-plane residual| over the points inserted during
#                    that visit. It describes one visit, not a time series,
#                    which is why the paper calls it res_spread.
#   cell_id          computed by the SLAM from the pose at save time, so before
#                    loop closure and global bundle adjustment, and never
#                    recomputed. Where the trajectory drifts between visits by
#                    more than one cell (6.25 cm), a revisit is given new ids
#                    and is not merged with the earlier one.
def _collect_global_cell_statistics(pcd_files):
    """Merge-wide obs_count / view_diversity / temp_consistency, per grid cell.

    Keyed on cell_id: a fixed grid at the finest octree resolution, taken from
    the point's position rather than from the octree node holding it. That
    matters twice over.

    First, resolution: the previous version keyed on voxel_id, the ROOT voxel, so
    it overwrote every point in a 1 m cube with a single value and destroyed
    whatever detail the SLAM exported.

    Second, correctness of obs_count. A voxel's accumulator only lives as long as
    the scanner keeps looking at it — margi() drops any node that stops receiving
    points — so revisiting a surface starts a fresh count from zero. Taking a
    plain maximum would therefore report the longest single visit rather than the
    total. Scans are processed in order, so a drop in a leaf's obs_count marks the
    end of an episode: sum the episode maxima instead.

    view_diversity keeps a maximum (lambda_min of a scatter matrix is not additive
    across episodes), and temp_consistency keeps the value observed at the highest
    obs_count.

    Returns (stats, available_fields) where stats maps leaf id -> arrays.
    """
    n = 0
    prev = np.zeros(0, dtype=np.float64)     # obs_count at the end of the current episode
    total = np.zeros(0, dtype=np.float64)    # summed maxima of completed episodes
    vmax = np.full(0, np.nan)
    tcval = np.full(0, np.nan)
    tcobs = np.full(0, -1.0)
    available_fields = set()

    for _, pcd_file in tqdm(pcd_files, desc="Collecting cell statistics"):
        pcd, channels = read_pcd_channels(pcd_file)
        available_fields.update(channels.keys())
        if len(pcd.colors) > 0:
            available_fields.add('rgb')

        cell_id = channels.get('cell_id')
        if cell_id is None:
            continue
        ids_all = cell_id.astype(np.int64)
        keep = ids_all >= 0
        if not np.any(keep):
            continue
        ids_all = ids_all[keep]

        obs = channels.get('obs_count')
        vd = channels.get('view_diversity')
        tc = channels.get('temp_consistency')
        obs_all = obs[keep] if obs is not None else np.zeros(len(ids_all))
        vd_all = vd[keep] if vd is not None else np.full(len(ids_all), np.nan)
        tc_all = tc[keep] if tc is not None else np.full(len(ids_all), np.nan)

        ids, (o, v, t) = _first_per_id(ids_all, obs_all.astype(np.float64),
                                       vd_all.astype(np.float64),
                                       tc_all.astype(np.float64))
        top = int(ids.max()) + 1
        if top > n:
            grow = top - n
            prev = np.concatenate([prev, np.zeros(grow)])
            total = np.concatenate([total, np.zeros(grow)])
            vmax = np.concatenate([vmax, np.full(grow, np.nan)])
            tcval = np.concatenate([tcval, np.full(grow, np.nan)])
            tcobs = np.concatenate([tcobs, np.full(grow, -1.0)])
            n = top

        if obs is not None:
            restarted = o < prev[ids]
            total[ids[restarted]] += prev[ids[restarted]]
            prev[ids] = o
        if vd is not None:
            vmax[ids] = np.fmax(vmax[ids], v)
        if tc is not None:
            better = (t != -1.0) & np.isfinite(t) & (o > tcobs[ids])
            sel = ids[better]
            tcval[sel] = t[better]
            tcobs[sel] = o[better]

    total = total + prev          # close the final episode of every leaf
    return {'obs_count': total, 'view_diversity': vmax,
            'temp_consistency': tcval}, available_fields


def _apply_global_cell_statistics(stats, channels):
    """Replace per-scan snapshots with the merge-wide value for each leaf."""
    cell_id = channels.get('cell_id')
    if cell_id is None:
        return channels
    corrected = dict(channels)
    ids = cell_id.astype(np.int64)
    ok = (ids >= 0) & (ids < len(stats['obs_count']))
    sel = ids[ok]
    for name in ('obs_count', 'view_diversity', 'temp_consistency'):
        src = channels.get(name)
        if src is None:
            continue
        out = src.copy()
        vals = stats[name][sel]
        good = np.isfinite(vals)
        idx = np.flatnonzero(ok)[good]
        out[idx] = vals[good].astype(out.dtype)
        corrected[name] = out
    return corrected


# A diverged scan still gets a row in alidarState.txt. Any pose further than this
# from the median of the trajectory is treated as garbage rather than as a real
# position, and its scan is left out of the conversion.
POSE_SANITY_RADIUS = 1e4  # metres


def _validate_poses(positions, timestamps):
    """Flag trajectory rows that cannot be a real pose.

    VoxelSLAM writes one row per scan even when the EKF has just diverged. On
    one of our recordings the Ouster clock jumps from sensor uptime to UTC
    part way through, so the scan spanning the jump is propagated over a dt of
    ~1.75e9 s and lands at ~1e17 m before the degeneracy check resets the
    session. Transforming a scan by such a pose does not fail — it quietly puts
    a few thousand points 1e17 m away, and every viewer that fits its view to
    the bounding box then renders the real map as a sub-pixel dot.

    Returns a boolean mask over the rows; indices are preserved, because scan
    `i.pcd` is matched to pose `i`.
    """
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(timestamps)
    within = np.zeros(len(positions), dtype=bool)
    if not finite.any():
        return within
    centre = np.median(positions[finite], axis=0)
    within[finite] = np.linalg.norm(positions[finite] - centre, axis=1) <= POSE_SANITY_RADIUS
    return within


def read_trajectory(trajectory_file):
    """
    Read alidarState.txt and extract poses plus the per-scan channels.

    Column layout (0-indexed), written by FileReaderWriter::save_pose:
        0       timestamp
        1-3     position x y z
        4-7     quaternion qx qy qz qw
        8-10    velocity
        11-13   gyro bias
        14-16   accelerometer bias
        17-19   gravity
        20-22   BA pose looseness, rotation (roll pitch yaw)
        23-25   BA pose looseness, translation (x y z)

    Columns 20-25 are ScanPose::v6 = 1/|diag(H[0:6, DIM:DIM+6])|, the element-wise
    reciprocal of the sliding-window BA information matrix block coupling the two
    oldest keyframes (voxelslam.cpp:1836). It is an uncalibrated *relative*
    stiffness score, not a marginal covariance in m² / rad²: small means the BA
    constrained that step weakly, e.g. in a geometrically degenerate corridor.

    Only the rotation half is used. The three translation entries are mutually
    correlated at r > 0.998 in every dataset checked, so they carry roughly one
    effective degree of freedom rather than three, and on the Ouster handheld data
    they are additionally saturated by the IMU preintegration term (std 1.5e-4
    around a mean of 4.3e-3) and carry almost no signal.

    Returns (poses, timestamps, scan_channels, valid) where scan_channels maps a
    name in SCAN_CHANNELS to an array with one value per pose, and valid is the
    per-pose mask from _validate_poses.
    """
    rows = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 8:  # Need at least timestamp + pose
                rows.append([float(value) for value in data])

    if not rows:
        return [], [], {}, np.zeros(0, dtype=bool)

    timestamps = [row[0] for row in rows]
    # scipy expects quaternions as [x, y, z, w]
    quaternions = np.array([row[4:8] for row in rows], dtype=np.float64)

    # A diverged row can carry a zero-norm or non-finite quaternion, which scipy
    # rejects outright and which would abort the whole run. Neutralise those
    # rows here and let the validity mask below drop them.
    quat_norms = np.linalg.norm(quaternions, axis=1)
    quat_bad = ~np.isfinite(quat_norms) | (quat_norms < 1e-9)
    if quat_bad.any():
        quaternions[quat_bad] = (0.0, 0.0, 0.0, 1.0)

    rotations = R.from_quat(quaternions)
    matrices = rotations.as_matrix()

    positions = np.array([row[1:4] for row in rows], dtype=np.float64)
    valid = _validate_poses(positions, np.asarray(timestamps, dtype=np.float64))
    valid &= ~quat_bad

    poses = []
    for index, row in enumerate(rows):
        transform = np.eye(4)
        transform[:3, :3] = matrices[index]
        transform[:3, 3] = row[1:4]
        poses.append(transform)

    scan_channels = {}
    count = len(rows)

    # pose_unc_rot: rms of the three rotation entries, so the value is a
    # standard-deviation-like quantity in rad rather than a variance-like one.
    # Boosting is invariant to monotone transforms, so this matters only for the
    # linear baseline and for readable partial-dependence plots.
    if all(len(row) >= 26 for row in rows):
        rotation_var = np.array([row[20:23] for row in rows], dtype=np.float64)
        with np.errstate(invalid='ignore'):
            looseness = np.sqrt(rotation_var.mean(axis=1))
        scan_channels['pose_unc_rot'] = np.where(
            np.isfinite(looseness), looseness,
            CHANNEL_SENTINELS['pose_unc_rot']).astype(np.float32)

    # ang_rate: |Log(R_k^T · R_k+1)| / dt, the angular rate over the scan interval
    # in rad/s. Deskewing error scales with how far the sensor rotated during the
    # sweep, so this targets motion distortion, which no other channel sees. It is
    # uncorrelated with pose_unc_rot (r = -0.013 on the Ouster handheld data), so
    # the two are independent signals rather than proxies for each other.
    if count >= 2:
        deltas = np.diff(np.asarray(timestamps))
        relative = (rotations[:-1].inv() * rotations[1:]).magnitude()
        # A step touching a rejected pose has a meaningless dt (1.75e9 s across
        # the Ouster clock jump), which would otherwise report ~0 rad/s for the
        # last good scan as well.
        usable = (deltas > 1e-9) & valid[:-1] & valid[1:]
        step = np.where(usable, relative / np.where(usable, deltas, 1.0),
                        CHANNEL_SENTINELS['ang_rate'])
        rates = np.empty(count, dtype=np.float64)
        rates[:-1] = step
        rates[-1] = step[-1]  # last scan has no successor; reuse the previous step
        scan_channels['ang_rate'] = rates.astype(np.float32)

    return poses, timestamps, scan_channels, valid


def transform_pcd(pcd_file, transform_matrix):
    """
    Load a PCD file and transform it to global coordinates.
    Returns (pcd, channels), or (None, {}) for an empty cloud.

    `range` is computed here, before the transform, because the per-scan PCDs
    written by VoxelSLAM are in the body frame: var_init applies the LiDAR->IMU
    extrinsic to pv.pnt and pvec_update never writes world coordinates back into
    it (voxelslam.hpp:190-226). So ||xyz|| is the measured range, up to the few-cm
    offset between the IMU origin and the LiDAR optical centre.
    """
    pcd, channels = read_pcd_channels(pcd_file)

    if len(pcd.points) == 0:
        return None, {}

    channels['range'] = np.linalg.norm(np.asarray(pcd.points), axis=1).astype(np.float32)

    original_has_colors = len(pcd.colors) > 0
    if original_has_colors:
        original_color_count = len(pcd.colors)

    # Geometry and normals transform; scalar fields are unaffected by rigid transform.
    pcd.transform(transform_matrix)

    if original_has_colors:
        colors_preserved = len(pcd.colors) == original_color_count
        if not colors_preserved:
            print(f"  Warning: Color information may have been lost during transformation of {pcd_file}")

    if len(pcd.normals) > 0:
        rotation_matrix = transform_matrix[:3, :3]
        normals = np.asarray(pcd.normals)
        transformed_normals = (rotation_matrix @ normals.T).T
        pcd.normals = o3d.utility.Vector3dVector(transformed_normals)

    return pcd, channels


def downsample_with_color_preservation(pcd, voxel_size):
    """
    Downsample point cloud while preserving color.
    Note: per-point scalar channels are lost during Open3D voxel downsampling.
    Returns (pcd, {}); per-scan channels are broadcast by the caller afterwards.
    """
    print(f"  Original points: {len(pcd.points)}")

    has_colors = len(pcd.colors) > 0
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    print(f"  Downsampled points: {len(downsampled_pcd.points)}")

    if has_colors:
        colors_preserved = len(downsampled_pcd.colors) == len(downsampled_pcd.points)
        if not colors_preserved:
            print(f"  Warning: Color information may have been affected by downsampling")

    return downsampled_pcd, {}


def match_pcds_to_poses(input_dir, timestamps, ts_tol=0.01, pose_offset=0, valid=None):
    """Match PCD files to trajectory poses by timestamp.

    Supports two naming conventions:
      - Integer index:  0.pcd, 1.pcd, ...   (original VoxelSLAM output)
      - Timestamp:      3431.836751000.pcd   (colored_pcd_saver output)

    pose_offset: shift the pose assigned to each matched PCD by this many
                 positions (e.g. +1 means each PCD uses the next pose).
    valid:       optional per-pose mask; a PCD whose pose is not valid is left
                 out entirely rather than transformed by a garbage pose.

    Returns a list of (pose_index, pcd_path) in pose order.
    """
    def pose_ok(index):
        return valid is None or valid[index]

    all_files = [f for f in os.listdir(input_dir) if f.endswith('.pcd')]
    if not all_files:
        return []

    # Detect naming convention from first file.
    # Timestamp PCDs have a decimal point in the stem (e.g. "525.092635436");
    # integer-indexed PCDs are whole numbers (e.g. "0", "1", "2053").
    stem = os.path.splitext(all_files[0])[0]
    timestamp_named = '.' in stem

    if not timestamp_named:
        # Integer index mode — direct 1:1 mapping (with optional offset)
        pcd_files = []
        for i in range(len(timestamps)):
            pose_i = i + pose_offset
            if pose_i < 0 or pose_i >= len(timestamps):
                continue
            if not pose_ok(pose_i):
                continue
            pcd_file = os.path.join(input_dir, f"{i}.pcd")
            if os.path.exists(pcd_file):
                pcd_files.append((pose_i, pcd_file))
            else:
                print(f"Warning: {pcd_file} not found")
        return pcd_files

    # Timestamp mode — nearest-neighbour match within tolerance
    pcd_ts = {}
    for f in all_files:
        try:
            pcd_ts[float(os.path.splitext(f)[0])] = os.path.join(input_dir, f)
        except ValueError:
            pass

    pcd_ts_arr = np.array(sorted(pcd_ts.keys()))
    pcd_files = []
    used = set()
    for i, ts in enumerate(timestamps):
        idx = np.searchsorted(pcd_ts_arr, ts)
        candidates = []
        if idx < len(pcd_ts_arr):
            candidates.append(pcd_ts_arr[idx])
        if idx > 0:
            candidates.append(pcd_ts_arr[idx - 1])
        if not candidates:
            continue
        best = min(candidates, key=lambda t: abs(t - ts))
        if abs(best - ts) <= ts_tol and best not in used:
            pose_i = i + pose_offset
            if 0 <= pose_i < len(timestamps) and pose_ok(pose_i):
                used.add(best)
                pcd_files.append((pose_i, pcd_ts[best]))

    expected = len(timestamps) if valid is None else int(np.count_nonzero(valid))
    unmatched = expected - len(pcd_files)
    if unmatched:
        print(f"  {unmatched} trajectory poses had no matching PCD (tolerance={ts_tol} s)")

    # Diagnostic: show distribution of timestamp deltas
    if pcd_files:
        deltas = []
        for pose_i, pcd_path in pcd_files:
            pcd_stem = os.path.splitext(os.path.basename(pcd_path))[0]
            try:
                pcd_t = float(pcd_stem)
                deltas.append(abs(pcd_t - timestamps[pose_i]))
            except ValueError:
                pass
        if deltas:
            deltas = np.array(deltas)
            print(f"  Timestamp delta: mean={deltas.mean()*1e6:.3f} µs, "
                  f"max={deltas.max()*1e6:.3f} µs, "
                  f"min={deltas.min()*1e6:.3f} µs")
            print(f"  (Differences >1 µs may indicate a real mismatch; <1 µs is just float precision)")

    return pcd_files


def convert_to_global_frame(input_dir, output_dir, trajectory_file, merge_all=False,
                            downsample_voxel=None, pose_offset=0,
                            binary=False, merged_only=False,
                            voxel_aggregate=True):
    """
    Convert all PCD files to global frame using trajectory data.
    Preserves intensity and RGB information if available.

    Supports both integer-indexed PCDs (0.pcd, 1.pcd, ...) and
    timestamp-named PCDs (<timestamp>.pcd) from colored_pcd_saver.

    Args:
        input_dir: Directory containing PCD files
        output_dir: Directory to save transformed PCD files
        trajectory_file: Path to alidarState.txt
        merge_all: If True, merge all point clouds into one file
        downsample_voxel: If specified, downsample with this voxel size
    """

    # Read trajectory
    print("Reading trajectory file...")
    poses, timestamps, scan_channels, valid = read_trajectory(trajectory_file)
    print(f"Loaded {len(poses)} poses")

    rejected = np.flatnonzero(~valid)
    if rejected.size:
        listed = ', '.join(str(i) for i in rejected[:10])
        if rejected.size > 10:
            listed += ', ...'
        print(f"Warning: {rejected.size} of {len(poses)} poses are unusable and their "
              f"scans are skipped (indices: {listed})")
        print(f"  A pose is rejected when it is non-finite or more than "
              f"{POSE_SANITY_RADIUS:g} m from the median of the trajectory, which means "
              f"the EKF had diverged for that scan. Keeping one would scatter its points "
              f"far enough to flatten the bounding box of the whole map.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Match PCDs to poses
    if pose_offset != 0:
        print(f"Pose offset: {pose_offset:+d} (each PCD uses pose shifted by {pose_offset})")
    pcd_files = match_pcds_to_poses(input_dir, timestamps, pose_offset=pose_offset,
                                    valid=valid)

    print(f"Found {len(pcd_files)} PCD files")

    # Check first file to see what data is available
    if pcd_files:
        sample_pcd, sample_channels = read_pcd_channels(pcd_files[0][1])
        info_parts = []
        if len(sample_pcd.colors) > 0:
            info_parts.append("RGB")
        info_parts.extend(name for name in CHANNEL_ORDER if name in sample_channels)
        print(f"Point cloud data: {', '.join(info_parts) if info_parts else 'XYZ only'}")

        derived = ['range'] + [name for name in SCAN_CHANNELS if name in scan_channels]
        print(f"Derived channels: {', '.join(derived)}")
        missing = [name for name in SCAN_CHANNELS if name not in scan_channels]
        if missing:
            print(f"  Not derivable from this trajectory file: {', '.join(missing)}")

    voxel_stats = None
    stream_output = None
    stream_count_offsets = None
    stream_fields = None
    if merge_all:
        # The aggregate is what turns a per-scan snapshot into a merge-wide
        # value: obs_count in a per-scan PCD is only what that leaf had counted
        # at the instant the scan was written, so points on one surface disagree
        # depending on when they were saved. Now keyed on cell_id, it fixes that
        # without flattening the cloud — the earlier version keyed on the root
        # voxel and overwrote a whole 1 m cube with one number.
        # --no-voxel-aggregate keeps the raw per-scan snapshots instead.
        print("Pass 1/2: collecting global voxel statistics ...")
        voxel_stats, available_fields = _collect_global_cell_statistics(pcd_files)
        if not voxel_aggregate:
            # Pass 1 still has to run: `available_fields` is the union of what
            # read_pcd_channels yields across EVERY scan, and that is not the
            # same as the PCD header. read_pcd_channels filters by content, so
            # early scans — written before the map has planes — expose fewer
            # channels than later ones despite identical FIELDS lines. Taking
            # the schema from one file silently drops temp_consistency, ekf_res,
            # reflectivity and intensity from the merged cloud.
            print("  --no-voxel-aggregate: discarding the per-voxel maxima; "
                  "obs_count / view_diversity / temp_consistency keep their "
                  "per-scan leaf-resolution values")
            voxel_stats = None
        # Derived per point from xyz, so it is never seen by pass 1 — but it does
        # not survive downsampling, so it is added before the intersection below.
        available_fields.add('range')
        if downsample_voxel is not None:
            available_fields.intersection_update({'rgb'})
        # Per-scan channels are broadcast after downsampling and always survive.
        available_fields.update(name for name in SCAN_CHANNELS if name in scan_channels)
        stream_fields = _stream_schema(available_fields)
        output_file = os.path.join(output_dir, "global_map.pcd")
        stream_output = open(output_file, 'wb+' if binary else 'w+')
        stream_count_offsets = _write_stream_header(stream_output, stream_fields, binary)
        if voxel_stats is not None:
            print(f"Collected statistics for {len(voxel_stats['obs_count']):,} cells")
        print("Pass 2/2: transforming and streaming point clouds ...")

    # Transform and save scans one at a time. In merge mode, append each scan to
    # the output file immediately instead of retaining its points in memory.
    print("Processing point clouds...")
    saved_count = 0
    merged_point_count = 0

    for i, pcd_file in tqdm(pcd_files, desc="Processing PCDs"):
        t_pcd, channels = transform_pcd(pcd_file, poses[i])
        if t_pcd is None:
            continue

        if downsample_voxel is not None:
            t_pcd, channels = downsample_with_color_preservation(t_pcd, downsample_voxel)

        if voxel_stats is not None:
            channels = _apply_global_cell_statistics(voxel_stats, channels)

        # Per-scan channels are constant over the scan. Broadcast after any
        # downsampling so the length always matches the surviving point count.
        point_count = len(t_pcd.points)
        for name in SCAN_CHANNELS:
            values = scan_channels.get(name)
            if values is not None:
                channels[name] = np.full(point_count, values[i], dtype=np.float32)

        if not merged_only:
            output_file = os.path.join(output_dir, f"global_{i}.pcd")
            success = save_pcd_with_channels(t_pcd, channels, output_file)
            if not success:
                print(f"Error saving {output_file}")

        if stream_output is not None:
            _append_stream_points(stream_output, stream_fields, t_pcd, channels, binary)
            merged_point_count += point_count
        saved_count += 1

    if merged_only:
        print(f"Processed {saved_count} point clouds")
    else:
        print(f"Saved {saved_count} individual point clouds")

    if saved_count == 0:
        print("Error: no point clouds to merge")
        if stream_output is not None:
            stream_output.close()
        return

    if stream_output is not None:
        for offset in stream_count_offsets:
            stream_output.seek(offset)
            count = f"{merged_point_count:020d}"
            stream_output.write(count.encode('ascii') if binary else count)
        stream_output.close()
        print(f"Saved global map with {merged_point_count} points")


def main():
    parser = argparse.ArgumentParser(description="Convert VoxelSLAM PCD files to global frame")
    parser.add_argument("input_dir", help="Directory containing numbered PCD files")
    parser.add_argument("trajectory_file", help="Path to alidarState.txt file")
    parser.add_argument("-o", "--output_dir",
                        help="Output directory (default: <input_dir>_global alongside input)",
                        default=None)
    parser.add_argument("--merge", action="store_true", help="Merge all point clouds into one file")
    parser.add_argument("--no-voxel-aggregate", dest="voxel_aggregate",
                        action="store_false",
                        help="Do not overwrite obs_count / view_diversity / "
                             "temp_consistency with one merge-wide value per "
                             "finest-grid cell (cell_id): obs_count summed over "
                             "visits, view_diversity the maximum over snapshots, "
                             "temp_consistency the snapshot with the highest "
                             "obs_count. Without it each point keeps the snapshot "
                             "taken when its own scan was saved.")
    parser.add_argument("--binary", action="store_true",
                        help="Write the merged PCD in faster, smaller binary format")
    parser.add_argument("--merged_only", action="store_true",
                        help="With --merge, skip redundant global_<index>.pcd files")
    parser.add_argument("--downsample", type=float, help="Downsample voxel size (e.g., 0.1)")
    parser.add_argument("--pose_offset", type=int, default=0,
                        help="Shift pose assignment by N positions (e.g. +1 or -1) to test alignment")

    args = parser.parse_args()

    if args.output_dir is None:
        abs_input = os.path.abspath(args.input_dir.rstrip('/\\'))
        args.output_dir = os.path.join(os.path.dirname(abs_input), os.path.basename(abs_input) + "_global")

    # Check if input files exist
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return

    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file {args.trajectory_file} does not exist")
        return

    if args.merged_only and not args.merge:
        parser.error("--merged_only requires --merge")

    # Convert to global frame
    convert_to_global_frame(
        args.input_dir,
        args.output_dir,
        args.trajectory_file,
        merge_all=args.merge,
        downsample_voxel=args.downsample,
        pose_offset=args.pose_offset,
        binary=args.binary,
        merged_only=args.merged_only,
        voxel_aggregate=args.voxel_aggregate,
    )


if __name__ == "__main__":
    main()
