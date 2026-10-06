#!/usr/bin/env python3
"""
Reading Leica RTC360 exports as separate E57 stations.

Each E57 holds one scan in the scanner's own local frame — the sensor sits at
the origin, so ranges and directions are read straight off the coordinates —
plus the pose that maps it into the project frame:

    p_project = R @ p_local + t

`t` is therefore the station origin in the project frame, which is exactly what
the visibility model needs. Verified on this dataset: transforming Setup 002 by
`R @ x + t` lands it on Setup 001 (median NN 6 cm on a random subsample), while
the transposed convention gives 30 cm.

The exports also carry `rowIndex` / `columnIndex` (the native scan grid) and
`cartesianInvalidState`. This dataset exports only valid returns, so absent grid
cells are the non-returns. The visibility model does not rely on that grid: it
rebuilds its own angular panorama, which avoids depending on the scanner's
internal index-to-angle mapping.
"""

import glob
import json
import os

import numpy as np

try:
    import pye57
except ImportError:  # pragma: no cover
    pye57 = None

XYZ_FIELDS = ("cartesianX", "cartesianY", "cartesianZ")


def _require_pye57():
    if pye57 is None:
        raise ImportError("pye57 is required to read E57 stations: pip install pye57")


def list_stations(path):
    """Return the sorted list of E57 files under `path` (file or directory)."""
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.e57")))
    if not files:
        raise FileNotFoundError(f"no .e57 files under {path}")
    return files


def read_pose(path, index=0):
    """Return (R, t) mapping this station's local frame into the project frame."""
    _require_pye57()
    # The header holds a weak reference into the E57 document, so the reader has
    # to stay alive until the pose has been copied out into numpy.
    reader = pye57.E57(path)
    header = reader.get_header(index)
    R = np.asarray(header.rotation_matrix, dtype=np.float64)
    t = np.asarray(header.translation, dtype=np.float64)
    del header, reader
    return R, t


def read_station(path, index=0, with_intensity=False):
    """Load one station. Coordinates stay in the scanner's local frame."""
    _require_pye57()
    e = pye57.E57(path)
    header = e.get_header(index)
    data = e.read_scan_raw(index)
    xyz = np.stack([np.asarray(data[f], dtype=np.float64) for f in XYZ_FIELDS], axis=1)

    # Only present in some exports; when it is, non-zero means "no return here".
    invalid = data.get("cartesianInvalidState")
    if invalid is not None:
        valid = np.asarray(invalid) == 0
        if not valid.all():
            xyz = xyz[valid]

    out = {
        "name": os.path.splitext(os.path.basename(path))[0],
        "path": path,
        "index": index,
        "xyz_local": xyz,
        "R": np.asarray(header.rotation_matrix, dtype=np.float64),
        "t": np.asarray(header.translation, dtype=np.float64),
    }
    if with_intensity and "intensity" in data:
        inten = np.asarray(data["intensity"], dtype=np.float32)
        out["intensity"] = inten[valid] if invalid is not None and not valid.all() else inten
    return out


def to_project(xyz_local, R, t):
    return xyz_local @ np.asarray(R).T + np.asarray(t)


def station_table(station_dir, cache=None, verbose=True):
    """Collect name / origin / rotation for every station, with a JSON cache."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            table = json.load(f)
        if verbose:
            print(f"  {len(table)} stations (cached from {cache})")
        return table

    table = []
    for path in list_stations(station_dir):
        R, t = read_pose(path)
        table.append({
            "name": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "origin": t.tolist(),
            "rotation": R.tolist(),
        })
        if verbose:
            print(f"    {table[-1]['name']:40s} origin={np.round(t, 3)}")
    if cache:
        with open(cache, "w") as f:
            json.dump(table, f, indent=2)
    return table


STATION_FIELD = "station_id"


def sidecar_path(merged_path):
    """Path of the pose sidecar that belongs to a merged GT cloud."""
    return os.path.splitext(merged_path)[0] + "_stations.json"


def read_sidecar(path):
    """Load a pose sidecar written by `merge_to_las`.

    Accepts either the sidecar itself or the merged LAS it belongs to.
    """
    if not path.endswith(".json"):
        path = sidecar_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no station sidecar at {path}. The merged GT must be built by "
            f"merge_gt_e57.py, which records each station's pose alongside it.")
    with open(path) as f:
        meta = json.load(f)
    for st in meta["stations"]:
        st["R"] = np.asarray(st["rotation"], dtype=np.float64)
        st["t"] = np.asarray(st["origin"], dtype=np.float64)
    return meta


# 0.1 mm rather than the usual 1 mm. The merged cloud is not only measured
# against (where 1 mm is plenty) but also re-projected into each scanner's frame
# to rebuild its range panorama, and there the quantisation turns into angular
# jitter: at 20 m, 0.5 mm of position error is 0.0014 deg, against a 0.05 deg
# bin. Points within that of a bin edge migrate, and at a depth discontinuity a
# migrated point changes that bin's near/far by the whole depth of the edge.
# Measured on an indoor scene (14 stations, 486 M reference points):
# at 1 mm the merged-cloud panoramas disagreed with the per-E57 panoramas on
# 0.59% of verdicts; at 0.1 mm on 0.058%, and on 0.010% of domain memberships.
# Costs nothing -- LAS coordinates are int32 either way, and 0.1 mm still spans
# +/-214 km. The residue is irreducible: a point on a bin boundary can fall
# either side at any precision, and at a depth edge that swaps the bin's
# near/far by the depth of the edge.
GT_SCALE = 0.0001


def merge_to_las(station_dir, out_path, chunk_stations=True, with_station_id=True,
                 scale=GT_SCALE, verbose=True):
    """Write every station, transformed into the project frame, into one LAS.

    Streams station by station: only one scan (~25 M points) is resident at a
    time.

    With `with_station_id` (the default) each point also carries the id of the
    station it came from, and a JSON sidecar records that station's pose, its
    elevation span and its farthest return. That is what lets the visibility
    model work from the merged cloud instead of from the original E57s -- which
    in turn means the merged cloud can be edited (cropping people out of the
    ground truth, say) and the visibility model follows the edit, because it is
    rebuilt from the same points the labels are measured against.

    The elevation span and maximum range are recorded here, while each station
    is in memory, so the visibility model needs only ONE pass over the merged
    cloud rather than one pass to size the panoramas and another to fill them.
    Both are only used to bound a grid, so an edit that removes points can leave
    them slightly generous without changing any verdict.
    """
    import laspy

    files = list_stations(station_dir)
    if verbose:
        print(f"  Merging {len(files)} stations -> {out_path}")
        if with_station_id:
            print(f"    carrying '{STATION_FIELD}' + pose sidecar "
                  f"{os.path.basename(sidecar_path(out_path))}")

    # Project-frame extent, from each station's transformed local bounding box.
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    poses = []
    for path in files:
        R, t = read_pose(path)
        poses.append((path, R, t))
    for path, R, t in poses:
        st = read_station(path)
        corners = np.array(np.meshgrid(*zip(st["xyz_local"].min(axis=0),
                                            st["xyz_local"].max(axis=0)))).reshape(3, -1).T
        p = to_project(corners, R, t)
        lo = np.minimum(lo, p.min(axis=0))
        hi = np.maximum(hi, p.max(axis=0))
        del st

    if with_station_id:
        # 1.4 / format 6 so the station id can travel as an extra dimension.
        header = laspy.LasHeader(version="1.4", point_format=6)
    else:
        header = laspy.LasHeader(version="1.2", point_format=0)
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.floor(lo)
    if with_station_id:
        header.add_extra_dim(laspy.ExtraBytesParams(
            name=STATION_FIELD, type=np.uint16,
            # The LAS ExtraBytes VLR caps this at 32 bytes; the sidecar holds
            # the real description (1-based id, 0 = unknown).
            description="station id, see _stations.json"))
    if verbose:
        print(f"    project extent {np.round(lo, 2)} .. {np.round(hi, 2)}")

    stations = []
    total = 0
    with laspy.open(out_path, mode="w", header=header) as writer:
        for sid, (path, R, t) in enumerate(poses, start=1):
            st = read_station(path, with_intensity=True)
            local = st["xyz_local"]
            xyz = to_project(local, R, t)

            # Recorded from the local frame, which is the frame the visibility
            # model rebuilds its panorama in.
            r = np.linalg.norm(local, axis=1)
            good = r > 1e-6
            el = np.degrees(np.arcsin(np.clip(local[good, 2] / r[good], -1.0, 1.0)))

            record = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
            record.x = xyz[:, 0]
            record.y = xyz[:, 1]
            record.z = xyz[:, 2]
            if "intensity" in st:
                inten = np.clip(st["intensity"], 0.0, 1.0) * 65535.0
                record.intensity = inten.astype(np.uint16)
            if with_station_id:
                record[STATION_FIELD] = np.full(len(xyz), sid, dtype=np.uint16)
            writer.write_points(record)

            stations.append({
                "id": sid,
                "name": st["name"],
                "source": os.path.abspath(path),
                "rotation": np.asarray(R).tolist(),
                "origin": np.asarray(t).tolist(),
                "points": int(len(xyz)),
                "el_min": float(el.min()) if el.size else 0.0,
                "el_max": float(el.max()) if el.size else 0.0,
                "max_range": float(r.max()) if r.size else 0.0,
            })
            total += len(xyz)
            if verbose:
                print(f"    [{sid:>3}] {st['name']:40s} +{len(xyz):>11,}  "
                      f"total {total:,}")
            del st, xyz, local, record

    if with_station_id:
        meta = {
            "merged_las": os.path.abspath(out_path),
            "station_field": STATION_FIELD,
            "source_dir": os.path.abspath(station_dir),
            "n_stations": len(stations),
            "points": total,
            "stations": stations,
        }
        with open(sidecar_path(out_path), "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"  wrote pose sidecar -> {sidecar_path(out_path)}")

    if verbose:
        print(f"  merged {total:,} points -> {out_path}")
    return total
