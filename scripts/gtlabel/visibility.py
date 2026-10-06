#!/usr/bin/env python3
"""
Scanner-visibility model: decide, per SLAM point, whether the RTC could have
seen that spot at all.

This is the gate that pure geometry cannot provide. A coverage radius asks "is
there a GT point nearby", which cannot tell a genuine SLAM error 15 cm off a
scanned wall from a point 15 cm behind it in a room the scanner never entered.
Both are *a point 15 cm from a GT surface*. Asking whether a beam could have
reached the spot separates them.

Per station, the scan is rebuilt as a range panorama in the scanner's own frame:
a regular (azimuth, elevation) grid holding the nearest return per direction. A
SLAM point is then converted to that station's spherical coordinates and compared
against what came back along the same ray:

    r  <  r_near - eps   the beam flew through this spot   -> free space violation
    r_near-eps <= r <= r_far+eps                           -> observed
    r  >  r_far + eps    something opaque was in the way   -> occluded
    no returns nearby                                      -> unknown

`r_near` / `r_far` are the min and max range over a 3x3 neighbourhood of bins,
not a single bin. That matters more than it looks: on a surface seen at grazing
incidence one bin spans a large range extent, and comparing against a single
value manufactures free-space violations along every oblique floor. The
neighbourhood widens the tolerance exactly where the geometry demands it, and it
absorbs isolated one-bin dropouts without a separate hole-filling pass.

Fusing stations, in priority order:

    any station observed  -> in domain, label by C2C
    else any free space   -> in domain, and an outlier whatever C2C says
    else                  -> excluded, nobody could see it

Observation outranks free space deliberately: at grazing angles and on thin
structures two stations will disagree, and declining to call a point an outlier
when some scanner saw a surface there is the conservative choice.

The tripod shadow falls out of this for free. The RTC360 has a blind cone below
the instrument — on this dataset elevations stop at -61.6 degrees, a 28.4 degree
cone that leaves an unscanned disc of roughly 1.8 m across on the ground under
each setup. Those directions simply hold no returns, so they read as unknown from
that station, and the fusion rule lets a neighbouring station fill them in.
"""

import os

import numpy as np
from scipy import ndimage

from . import e57, lasio

# Per-station verdicts, ordered by priority so that fusing across stations is a
# running maximum.
VIS_UNKNOWN = 0
VIS_OCCLUDED = 1
VIS_FREE_SPACE = 2
VIS_OBSERVED = 3

VIS_NAMES = {
    VIS_UNKNOWN: "unknown (no return that way)",
    VIS_OCCLUDED: "occluded",
    VIS_FREE_SPACE: "free-space violation",
    VIS_OBSERVED: "observed",
}


class StationPanorama:
    """Nearest/farthest return per direction, in one station's local frame."""

    def __init__(self, name, R, t, resolution, el_min, el_max, r_near, r_far,
                 max_range):
        self.name = name
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)
        self.resolution = float(resolution)
        self.el_min = float(el_min)
        self.el_max = float(el_max)
        self.r_near = r_near          # float32, +inf where nothing came back
        self.r_far = r_far            # float32, -inf where nothing came back
        self.max_range = float(max_range)

    @property
    def n_az(self):
        return self.r_near.shape[0]

    @property
    def n_el(self):
        return self.r_near.shape[1]

    @property
    def nbytes(self):
        return self.r_near.nbytes + self.r_far.nbytes

    # ------------------------------------------------- build (merged-LAS path)
    @classmethod
    def empty(cls, name, R, t, resolution, el_min, el_max, max_range,
              el_margin=1.0):
        """Allocate the grids for a station whose extent is already known.

        The merged-cloud path cannot measure a station's elevation span before
        allocating its grid -- the points arrive in whatever order the file
        holds them. The span is therefore taken from the sidecar, which recorded
        it at merge time when the station was in memory. It only bounds the
        grid, so a slightly generous value costs a few empty bins and nothing
        else, which is what keeps the sidecar valid after the merged cloud has
        been edited.
        """
        el_min = max(float(el_min) - el_margin, -90.0)
        el_max = min(float(el_max) + el_margin, 90.0)
        n_az = int(np.ceil(360.0 / resolution))
        n_el = int(np.ceil((el_max - el_min) / resolution)) + 1
        near = np.full((n_az, n_el), np.inf, dtype=np.float32)
        far = np.full((n_az, n_el), -np.inf, dtype=np.float32)
        return cls(name, R, t, resolution, el_min, el_max, near, far,
                   float(max_range))

    def accumulate(self, xyz_project):
        """Fold one block of this station's points into the raw near/far grids.

        Called once per chunk of the merged cloud. The 3x3 neighbourhood filter
        is deliberately NOT applied here -- it has to see the finished grid, so
        `finalise` applies it once after every chunk has been folded in.
        """
        local = (np.asarray(xyz_project, dtype=np.float64) - self.t) @ self.R
        r = np.linalg.norm(local, axis=1)
        keep = r > 1e-6
        if not keep.any():
            return
        local, r = local[keep], r[keep]

        az = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
        el = np.degrees(np.arcsin(np.clip(local[:, 2] / r, -1.0, 1.0)))
        ia = np.floor((az + 180.0) / self.resolution).astype(np.int64) % self.n_az
        ie = np.clip(np.floor((el - self.el_min) / self.resolution).astype(np.int64),
                     0, self.n_el - 1)
        flat = ia * self.n_el + ie

        # First/last of each run after sorting by (cell, range) gives this
        # block's min and max per touched cell in one pass. The resulting cell
        # indices are unique, so the merge back into the grid is plain fancy
        # indexing rather than np.minimum.at, which is far too slow here.
        order = np.lexsort((r, flat))
        fs, rs = flat[order], r[order].astype(np.float32)
        first = np.empty(len(fs), dtype=bool)
        first[0] = True
        np.not_equal(fs[1:], fs[:-1], out=first[1:])
        last = np.empty(len(fs), dtype=bool)
        last[-1] = True
        np.not_equal(fs[:-1], fs[1:], out=last[:-1])

        cells = fs[first]
        nf = self.r_near.reshape(-1)
        ff = self.r_far.reshape(-1)
        nf[cells] = np.minimum(nf[cells], rs[first])
        ff[cells] = np.maximum(ff[cells], rs[last])

    def finalise(self, verbose=False):
        """Apply the 3x3 neighbourhood filter once the grids are complete."""
        self.r_near = _filter3(self.r_near, ndimage.minimum_filter)
        self.r_far = _filter3(self.r_far, ndimage.maximum_filter)
        if verbose:
            filled = np.isfinite(self.r_near)
            print(f"    {self.name:40s} grid {self.n_az}x{self.n_el}  "
                  f"el [{self.el_min:.1f},{self.el_max:.1f}]  "
                  f"{100.0 * filled.mean():.1f}% of bins reachable  "
                  f"max range {self.max_range:.1f} m")
        return self

    # ------------------------------------------------- build (per-E57 path)
    @classmethod
    def build(cls, station, resolution=0.05, el_margin=1.0, verbose=True):
        xyz = station["xyz_local"]
        r = np.linalg.norm(xyz, axis=1)
        good = r > 1e-6
        xyz, r = xyz[good], r[good]

        az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
        el = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0)))

        el_min = max(float(el.min()) - el_margin, -90.0)
        el_max = min(float(el.max()) + el_margin, 90.0)
        n_az = int(np.ceil(360.0 / resolution))
        n_el = int(np.ceil((el_max - el_min) / resolution)) + 1

        ia = np.floor((az + 180.0) / resolution).astype(np.int64) % n_az
        ie = np.clip(np.floor((el - el_min) / resolution).astype(np.int64), 0, n_el - 1)
        flat = ia * n_el + ie

        # np.minimum.at is far too slow at this scale; sorting by cell and taking
        # the first/last of each run does the same job in a fraction of the time.
        near = np.full(n_az * n_el, np.inf, dtype=np.float32)
        far = np.full(n_az * n_el, -np.inf, dtype=np.float32)
        order = np.lexsort((r, flat))
        fs, rs = flat[order], r[order].astype(np.float32)
        first = np.empty(len(fs), dtype=bool)
        first[0] = True
        np.not_equal(fs[1:], fs[:-1], out=first[1:])
        last = np.empty(len(fs), dtype=bool)
        last[-1] = True
        np.not_equal(fs[:-1], fs[1:], out=last[:-1])
        near[fs[first]] = rs[first]
        far[fs[last]] = rs[last]

        near = near.reshape(n_az, n_el)
        far = far.reshape(n_az, n_el)

        # 3x3 neighbourhood, wrapping in azimuth (the seam at +/-180 is a real
        # neighbourhood, not an edge) and clamped in elevation.
        near = _filter3(near, ndimage.minimum_filter)
        far = _filter3(far, ndimage.maximum_filter)

        filled = np.isfinite(near)
        if verbose:
            print(f"    {station['name']:40s} {len(r):>11,} pts  "
                  f"grid {n_az}x{n_el}  el [{el_min:.1f},{el_max:.1f}]  "
                  f"{100.0 * filled.mean():.1f}% of bins reachable  "
                  f"max range {r.max():.1f} m")
        return cls(station["name"], station["R"], station["t"], resolution,
                   el_min, el_max, near, far, float(r.max()))

    # ------------------------------------------------------------------ query
    def classify(self, xyz_project, eps0=0.02, eps_rate=0.002,
                 behind_tolerance=0.10):
        """Per-point verdict for this station. Input is in the project frame.

        `behind_tolerance` widens the observed band on the far side only. Without
        it the model is too strict to be useful: a SLAM point 5 cm behind a
        scanned wall is formally unobservable, so it would be dropped from the
        evaluation — but that is ordinary range noise on a surface the scanner
        did measure, and it is exactly the kind of error the filter is meant to
        catch. Keep it well under a wall thickness, or the gate stops rejecting
        the neighbouring-room points it exists for.
        """
        local = (xyz_project - self.t) @ self.R          # R is orthonormal: R^T x
        r = np.linalg.norm(local, axis=1)

        verdict = np.zeros(len(local), dtype=np.uint8)   # VIS_UNKNOWN
        # Beyond what this station ever returned, it has nothing to say.
        cand = (r > 1e-6) & (r <= self.max_range)
        if not cand.any():
            return verdict

        lr = r[cand]
        lp = local[cand]
        az = np.degrees(np.arctan2(lp[:, 1], lp[:, 0]))
        el = np.degrees(np.arcsin(np.clip(lp[:, 2] / lr, -1.0, 1.0)))

        inside = (el >= self.el_min) & (el <= self.el_max)
        ia = np.floor((az + 180.0) / self.resolution).astype(np.int64) % self.n_az
        ie = np.floor((el - self.el_min) / self.resolution).astype(np.int64)
        np.clip(ie, 0, self.n_el - 1, out=ie)

        near = self.r_near[ia, ie]
        far = self.r_far[ia, ie]
        eps = eps0 + eps_rate * lr

        has_return = np.isfinite(near) & inside
        far_limit = far + eps + behind_tolerance
        sub = np.zeros(len(lr), dtype=np.uint8)
        sub[has_return & (lr < near - eps)] = VIS_FREE_SPACE
        sub[has_return & (lr > far_limit)] = VIS_OCCLUDED
        observed = has_return & (lr >= near - eps) & (lr <= far_limit)
        sub[observed] = VIS_OBSERVED

        verdict[cand] = sub
        return verdict


def _filter3(grid, filt):
    """3x3 filter with azimuth wrapped and elevation clamped."""
    padded = np.concatenate([grid[-1:], grid, grid[:1]], axis=0)
    out = filt(padded, size=(3, 3), mode="nearest")
    return out[1:-1]


def build_panoramas(station_dir, resolution=0.05, verbose=True):
    """Build one panorama per station. Scans are loaded and freed one at a time."""
    paths = e57.list_stations(station_dir)
    if verbose:
        print(f"  Building {len(paths)} station panoramas at {resolution:.3f} deg ...")
    panos = []
    for path in paths:
        station = e57.read_station(path)
        panos.append(StationPanorama.build(station, resolution=resolution,
                                           verbose=verbose))
        del station
    if verbose:
        total = sum(p.nbytes for p in panos) / 1e9
        print(f"    {len(panos)} panoramas resident, {total:.2f} GB")
    return panos


def build_panoramas_from_las(merged_path, stations, resolution=0.05,
                             station_field=None, chunk_size=None, verbose=True):
    """Build one panorama per station from a merged, project-frame GT cloud.

    The per-E57 path (`build_panoramas`) reads each station's own file. This one
    reads the merged cloud instead, splitting it by the per-point station id and
    folding each block into the panorama it belongs to. One streaming pass over
    the file serves every station, which matters because the merged GT is the
    largest single input in the pipeline.

    The reason to prefer it is not speed but provenance: the labels, the ICP and
    the visibility verdicts then all derive from exactly the same points. If
    people walking through the scene are cropped out of the merged cloud, the
    scanner stops "seeing" them here too -- those directions simply hold no
    return, so the affected SLAM points fall out of the evaluation domain
    instead of being scored against ground truth that no longer exists.
    """
    from . import e57 as e57_mod

    field = station_field or e57_mod.STATION_FIELD
    panos = {}
    for st in stations:
        panos[int(st["id"])] = StationPanorama.empty(
            st["name"], st["R"], st["t"], resolution,
            st["el_min"], st["el_max"], st["max_range"])
    if verbose:
        print(f"  Building {len(panos)} station panoramas at "
              f"{resolution:.3f} deg from {os.path.basename(merged_path)} ...")

    kwargs = {} if chunk_size is None else {"chunk_size": chunk_size}
    total = lasio.point_count(merged_path)
    tick = lasio.Ticker(enabled=verbose)
    seen = 0
    unknown = 0
    for xyz, extra in lasio.iter_xyz_fields(merged_path, [field], **kwargs):
        sid = extra[field]
        # A handful of ids per chunk, so grouping by sorted unique beats
        # testing every station against every chunk.
        for s in np.unique(sid):
            s = int(s)
            pano = panos.get(s)
            if pano is None:
                unknown += int((sid == s).sum())
                continue
            pano.accumulate(xyz[sid == s])
        seen += len(xyz)
        tick(f"    panorama fill: {seen:,} / {total:,} points")
    tick.done(f"    panorama fill: {seen:,} points")
    if unknown and verbose:
        print(f"    WARNING: {unknown:,} points carry a station id that is not "
              f"in the sidecar and were ignored")

    out = [panos[k].finalise(verbose=verbose) for k in sorted(panos)]
    if verbose:
        print(f"    {len(out)} panoramas resident, "
              f"{sum(p.nbytes for p in out) / 1e9:.2f} GB")
    return out


def classify_cloud(xyz_project, panoramas, eps0=0.02, eps_rate=0.002,
                   behind_tolerance=0.10, verbose=True):
    """Fuse every station's verdict for a block of points in the project frame."""
    fused = np.zeros(len(xyz_project), dtype=np.uint8)
    for pano in panoramas:
        np.maximum(fused,
                   pano.classify(xyz_project, eps0, eps_rate, behind_tolerance),
                   out=fused)
    return fused


def summarise(verdict):
    n = len(verdict)
    return {name: (int((verdict == v).sum()),
                   float((verdict == v).mean()) if n else 0.0)
            for v, name in VIS_NAMES.items()}


def classify_las(slam_path, panoramas, transform, eps0=0.02, eps_rate=0.002,
                 behind_tolerance=0.10, chunk_size=lasio.DEFAULT_CHUNK,
                 verbose=True):
    """Stream a SLAM LAS and return the fused verdict for every point.

    The cloud is streamed rather than loaded: at a few hundred million points
    the coordinates
    alone are 4.7 GB, and all the panoramas have to stay resident meanwhile.
    """
    T = np.asarray(transform, dtype=np.float64)
    total = lasio.point_count(slam_path)
    out = np.empty(total, dtype=np.uint8)
    tick = lasio.Ticker(enabled=verbose)
    at = 0
    for xyz in lasio.iter_xyz(slam_path, chunk_size):
        moved = xyz @ T[:3, :3].T + T[:3, 3]
        out[at:at + len(moved)] = classify_cloud(moved, panoramas, eps0, eps_rate,
                                                 behind_tolerance, verbose=False)
        at += len(moved)
        tick(f"    visibility: {at:,} / {total:,} points")
    tick.done(f"    visibility: {at:,} points classified")
    return out[:at]
