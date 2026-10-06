#!/usr/bin/env python3
"""
Streaming reader for the PCD files VoxelSLAM writes directly.

`global_map.pcd` comes straight out of the SLAM with every confidence channel
already attached, so the labelling pipeline can start from it and skip the
PCD -> LAS conversion entirely. The files are large (10.9 GB / 194 M points for
a kilometre-long walk), so nothing here loads a whole cloud.

Conventions were confirmed by matching record 0 of `global_map.pcd` against
record 0 of the converted LAS — identical `pv_var`, `balm_res` and `obs_count`,
same point order:

  * PCD `intensity` (float32) becomes the LAS extra dimension `intensity_orig`,
    which is the name `outlier_analysis.py` reads.
  * PCD `rgb` is PCL's packed uint32, `(r << 16) | (g << 8) | b` with 8-bit
    components. LAS colours are 16-bit, so each component is shifted up by 8:
    0x808080 in the PCD is (32768, 32768, 32768) in the LAS.

Unlike LAS, PCD carries no bounding box in its header, so `bounds()` costs a
full pass. The result is cached in a sidecar JSON next to the file.
"""

import json
import os

import numpy as np

DEFAULT_CHUNK = 20_000_000

_NUMPY_TYPE = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
}


def is_pcd(path):
    return str(path).lower().endswith(".pcd")


def read_header(path):
    """Parse the ASCII header. Returns a dict including the data byte offset."""
    header = {"fields": None, "size": None, "type": None, "count": None,
              "points": None, "width": None, "height": None, "data": None}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line — not a PCD file?")
            text = line.decode("ascii", "replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, rest = text.partition(" ")
            key = key.upper()
            if key == "FIELDS":
                header["fields"] = rest.split()
            elif key == "SIZE":
                header["size"] = [int(v) for v in rest.split()]
            elif key == "TYPE":
                header["type"] = rest.split()
            elif key == "COUNT":
                header["count"] = [int(v) for v in rest.split()]
            elif key in ("WIDTH", "HEIGHT", "POINTS"):
                header[key.lower()] = int(rest)
            elif key == "DATA":
                header["data"] = rest.strip().lower()
                header["data_offset"] = f.tell()
                break

    if header["fields"] is None or header["data"] is None:
        raise ValueError(f"{path}: incomplete PCD header")
    if header["count"] is None:
        header["count"] = [1] * len(header["fields"])
    if header["points"] is None:
        header["points"] = int(header["width"]) * int(header["height"])
    if header["data"] == "binary_compressed":
        raise NotImplementedError(
            f"{path} is binary_compressed; re-save it as DATA binary "
            "(pcl_convert_pcd_ascii_binary <in> <out> 1) or export a LAS")
    if header["data"] not in ("binary", "ascii"):
        raise ValueError(f"{path}: unsupported DATA {header['data']}")
    return header


def dtype_of(header):
    """Structured dtype for one record."""
    entries = []
    for name, size, typ, count in zip(header["fields"], header["size"],
                                      header["type"], header["count"]):
        key = (typ.upper(), int(size))
        if key not in _NUMPY_TYPE:
            raise ValueError(f"unsupported PCD field {name}: TYPE {typ} SIZE {size}")
        base = _NUMPY_TYPE[key]
        entries.append((name, base) if count == 1 else (name, base, (count,)))
    return np.dtype(entries)


def point_count(path):
    return int(read_header(path)["points"])


def field_names(path):
    return list(read_header(path)["fields"])


def iter_records(path, chunk_size=DEFAULT_CHUNK):
    """Yield structured arrays of at most `chunk_size` records."""
    header = read_header(path)
    dt = dtype_of(header)
    total = int(header["points"])

    if header["data"] == "ascii":
        names = header["fields"]
        with open(path, "r") as f:
            f.seek(header["data_offset"])
            block, seen = [], 0
            for line in f:
                if seen >= total:
                    break
                block.append(line.split())
                seen += 1
                if len(block) >= chunk_size:
                    yield _ascii_block(block, dt, names)
                    block = []
            if block:
                yield _ascii_block(block, dt, names)
        return

    offset = int(header["data_offset"])
    read = 0
    while read < total:
        n = min(chunk_size, total - read)
        rec = np.fromfile(path, dtype=dt, count=n,
                          offset=offset + read * dt.itemsize)
        if len(rec) == 0:
            break
        read += len(rec)
        yield rec


def _ascii_block(rows, dt, names):
    arr = np.zeros(len(rows), dtype=dt)
    cols = np.array(rows, dtype=np.float64).T
    for i, name in enumerate(names):
        arr[name] = cols[i].astype(dt[name])
    return arr


def iter_xyz(path, chunk_size=DEFAULT_CHUNK, origin=None):
    """Yield (N, 3) float64 XYZ blocks, matching lasio.iter_xyz."""
    for rec in iter_records(path, chunk_size):
        xyz = np.empty((len(rec), 3), dtype=np.float64)
        xyz[:, 0] = rec["x"]
        xyz[:, 1] = rec["y"]
        xyz[:, 2] = rec["z"]
        if origin is not None:
            xyz -= origin
        yield xyz


def bounds(path, cache=True, verbose=True):
    """(mins, maxs) as float64. Costs a full pass; cached in a sidecar JSON."""
    side = f"{path}.bounds.json"
    if cache and os.path.exists(side) and os.path.getmtime(side) >= os.path.getmtime(path):
        with open(side) as f:
            data = json.load(f)
        return np.asarray(data["mins"]), np.asarray(data["maxs"])

    if verbose:
        print(f"    scanning {os.path.basename(path)} for its extent "
              f"(PCD headers carry no bounding box) ...")
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for xyz in iter_xyz(path):
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.any():
            continue
        block = xyz[finite]
        lo = np.minimum(lo, block.min(axis=0))
        hi = np.maximum(hi, block.max(axis=0))

    if cache:
        try:
            with open(side, "w") as f:
                json.dump({"mins": lo.tolist(), "maxs": hi.tolist()}, f)
        except OSError:
            pass
    return lo, hi


def channel_fields(header):
    """Float channels to carry across into the LAS, excluding coordinates.

    `intensity` is renamed to `intensity_orig` so it survives as a float; the
    LAS `intensity` slot is only 16-bit and is filled separately.
    """
    out = []
    for name, typ, count in zip(header["fields"], header["type"], header["count"]):
        if name in ("x", "y", "z", "rgb") or count != 1:
            continue
        if typ.upper() != "F":
            continue
        out.append((name, "intensity_orig" if name == "intensity" else name))
    return out


def write_las(pcd_path, out_path, fields=None, transform=None,
              chunk_size=DEFAULT_CHUNK, verbose=True):
    """Convert a PCD to LAS, optionally transformed and with extra fields.

    `fields` maps name -> (full-length array, dtype), same as
    lasio.write_las_with_fields.
    """
    import laspy

    header = read_header(pcd_path)
    total = int(header["points"])
    fields = fields or {}
    for name, (values, _) in fields.items():
        if len(values) != total:
            raise ValueError(f"field '{name}' has {len(values):,} values, "
                             f"cloud has {total:,} points")

    mins, maxs = bounds(pcd_path, verbose=verbose)
    if transform is not None:
        T = np.asarray(transform, dtype=np.float64)
        corners = np.array(np.meshgrid(*zip(mins, maxs))).reshape(3, -1).T
        moved = corners @ T[:3, :3].T + T[:3, 3]
        origin = np.floor(moved.min(axis=0))
    else:
        T = None
        origin = np.floor(mins)

    has_rgb = "rgb" in header["fields"]
    out_header = laspy.LasHeader(version="1.2", point_format=2 if has_rgb else 0)
    out_header.scales = np.array([0.001, 0.001, 0.001])
    out_header.offsets = origin

    channels = channel_fields(header)
    for _, las_name in channels:
        if las_name not in fields:
            out_header.add_extra_dim(
                laspy.ExtraBytesParams(name=las_name, type=np.float32))
    for name, (_, dtype) in fields.items():
        out_header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

    from .lasio import Ticker
    tick = Ticker(enabled=verbose)
    at = 0
    # Decided once from the first chunk and frozen: choosing per chunk would
    # scale different parts of the same cloud differently.
    intensity_scale = None
    with laspy.open(out_path, mode="w", header=out_header) as writer:
        for rec in iter_records(pcd_path, chunk_size):
            n = len(rec)
            record = laspy.ScaleAwarePointRecord.zeros(n, header=out_header)

            xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
            if T is not None:
                xyz = xyz @ T[:3, :3].T + T[:3, 3]
            record.x = xyz[:, 0]
            record.y = xyz[:, 1]
            record.z = xyz[:, 2]

            if has_rgb:
                packed = rec["rgb"].astype(np.uint32)
                record.red = (((packed >> 16) & 0xFF) << 8).astype(np.uint16)
                record.green = (((packed >> 8) & 0xFF) << 8).astype(np.uint16)
                record.blue = ((packed & 0xFF) << 8).astype(np.uint16)

            if "intensity" in header["fields"]:
                inten = np.nan_to_num(rec["intensity"].astype(np.float64), nan=0.0)
                if intensity_scale is None:
                    peak = float(inten.max()) if len(inten) else 0.0
                    intensity_scale = 65535.0 if peak <= 1.0 else 1.0
                record.intensity = np.clip(inten * intensity_scale, 0,
                                           65535).astype(np.uint16)

            for pcd_name, las_name in channels:
                if las_name not in fields:
                    record[las_name] = rec[pcd_name].astype(np.float32)
            for name, (values, dtype) in fields.items():
                record[name] = values[at:at + n].astype(dtype)

            writer.write_points(record)
            at += n
            tick(f"    writing: {at:,} / {total:,} points")
    tick.done(f"    wrote {at:,} points -> {out_path}")
    return at
