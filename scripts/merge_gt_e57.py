#!/usr/bin/env python3
"""
Merge a Leica RTC360 project exported as per-station E57 files into ONE
project-frame LAS, keeping the information the labelling pipeline needs.

Why this exists
---------------
The pipeline used to read the merged cloud for C2C and ICP but go back to the
original E57s for the visibility model. That made the merged cloud a derived
artefact nobody could edit: cropping something out of it (people walking
through the scene, a parked vehicle -- anything present in the ground truth but
absent from the SLAM run) left the visibility model still convinced the scanner
had seen it, so those SLAM points kept being scored against ground truth that
was no longer there.

Here the merged LAS becomes the single source of truth. Every point carries the
id of the station that measured it, and a JSON sidecar records each station's
pose, elevation span and farthest return. The visibility model rebuilds its
panoramas from exactly the points in the file, so an edit to the merged cloud
propagates into the labels: cropped directions hold no return, read as
"unknown", and the affected SLAM points leave the evaluation domain rather than
being counted as errors.

It is also the form the dataset should be published in -- one cloud plus one
small pose file, rather than a directory of proprietary per-station exports.

Usage
-----
    python merge_gt_e57.py --e57-dir <project>/stations \
                           --out     <project>/merged_gt.las

Then edit the LAS if you need to (CloudCompare, whatever), keeping the
station_id field intact, and point the dataset config at it:

    gt_groups:
      - name: <group>
        merged_gt_las: <project>/merged_gt.las

The sidecar is found automatically next to it as <name>_stations.json.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtlabel import e57, lasio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Merge per-station E57s into one LAS with station ids and "
                    "a pose sidecar.")
    ap.add_argument("--e57-dir", required=True,
                    help="directory of per-station .e57 files (or one file)")
    ap.add_argument("--out", required=True, help="output merged .las")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if the output already exists")
    ap.add_argument("--no-station-id", action="store_true",
                    help="legacy plain merge: no station id, no sidecar. The "
                         "visibility model cannot use the result.")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"ERROR: {args.out} exists. Pass --overwrite to rebuild it.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    total = e57.merge_to_las(args.e57_dir, args.out,
                             with_station_id=not args.no_station_id)

    if args.no_station_id:
        return
    meta = e57.read_sidecar(args.out)
    print(f"\n{len(meta['stations'])} stations, {total:,} points")
    print(f"{'id':>4} {'station':40s} {'points':>13s} {'el range':>18s} {'max r':>8s}")
    for st in meta["stations"]:
        print(f"{st['id']:>4} {st['name'][:40]:40s} {st['points']:>13,} "
              f"{st['el_min']:>8.1f}..{st['el_max']:<8.1f} {st['max_range']:>8.1f}")
    print(f"\nmerged LAS : {args.out}")
    print(f"sidecar    : {e57.sidecar_path(args.out)}")
    print("\nThe merged cloud can now be edited. Keep the "
          f"'{e57.STATION_FIELD}' field; the visibility model is rebuilt from "
          "whatever points remain.")


if __name__ == "__main__":
    main()
