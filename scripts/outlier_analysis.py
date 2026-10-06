#!/usr/bin/env python3
"""
C2C regression + per-channel AUC analysis for point cloud confidence channels.

Steps:
  1. Compute per-channel ROC/AUC at each threshold τ (binarized C2C) for interpretability
  2. Fit one HistGradientBoostingRegressor (quantile=0.9) to predict raw C2C distance
  3. Use HGBR predictions as outlier score → compute model ROC/AUC at each τ
  4. Figures: predicted vs actual hexbin, permutation feature importances,
              ROC curves per τ, AUC summary vs τ
  5. Write predicted_c2c back to LAS as a new scalar field

Sentinel handling: balm_res / ekf_res / plane_quality / temp_consistency use
-1 as an invalid sentinel. These are encoded as NaN for HGBR (which handles
NaN natively via optimal missing-value splits) and masked for per-channel AUC.

Usage:
    python outlier_analysis.py <scene>_labeled.las
    python outlier_analysis.py <scene>_labeled.las --tau 0.01 0.02 0.05 0.10
    python outlier_analysis.py <scene>_labeled.las --out figures/ --out-las out.las
"""

import argparse
import json
import os
import sys

import numpy as np
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlabel import lasio  # noqa: E402  (streaming LAS/PCD writer)


# --------------------------------------------------------------------------- #
# Field definitions
# --------------------------------------------------------------------------- #
SENTINEL       = -1.0
SENTINEL_TOL   = 1e-6

# Channels are discovered from the file rather than hard-coded, so a SLAM rerun
# that adds or drops a confidence channel is picked up automatically. The old
# fixed list silently went stale: it still named `pv_var`, which the current
# SLAM no longer writes, and knew nothing about `range`, `pose_unc_rot` or
# `ang_rate`, which it now does.
NON_FEATURE_FIELDS = {
    # bookkeeping and pipeline outputs, never model inputs.
    # leaf_id replaced voxel_id in the SLAM (octree leaf rather than 1 m root
    # voxel); both are listed so clouds written either side of that change load.
    "cell_id", "leaf_id", "voxel_id", "C2C_distance", "domain_code", "eval_domain",
    "visibility_code", "gt_group", "predicted_c2c",
    # spatial train/test blocks for an in-scene split. These ARE the split, so
    # letting them in as features would be perfect leakage.
    "splitx_block", "splitx_ok", "splity_block", "splity_ok",
    # written by stretch_labels.py. c2c_rigid IS a label -- the pre-stretch
    # C2C, nearly identical to the new one wherever there is little drift -- and
    # stretch_id is trajectory position. Both went in as features the first
    # time every cloud in a run carried them, and lift@5 came out at 0.996.
    "c2c_rigid", "stretch_id",
}

# Extra dimensions whose logical name differs from the stored name.
FIELD_ALIASES = {"intensity_orig": "intensity"}

# ROC *curves* are only ever plotted, so they are traced on a subsample: one
# curve per channel per tau over tens of millions of points does not fit in
# memory. The AUC scalar still uses every point.
CURVE_MAX_POINTS = 2_000_000

C2C_FIELD = "C2C_distance"

# Written by the label pipelines: 1 where the RTC could plausibly have scanned
# the point, 0 where its C2C value says nothing about SLAM quality.
DOMAIN_FIELD = "eval_domain"


def discover_channels(path, explicit=None):
    """Return [(logical, stored)] for every usable confidence channel."""
    with laspy.open(path) as f:
        extra = list(f.header.point_format.extra_dimension_names)
    if explicit:
        missing = [c for c in explicit if c not in extra]
        if missing:
            sys.exit(f"ERROR: --channels not present in {path}: {missing}")
        chosen = list(explicit)
    else:
        chosen = [d for d in extra if d not in NON_FEATURE_FIELDS]
    return [(FIELD_ALIASES.get(d, d), d) for d in chosen]


def detect_sentinels(arrays, threshold=0.001):
    """Channels using -1 as an 'invalid' marker, found rather than declared.

    A channel qualifies when -1 is its minimum and lands on exactly -1 often
    enough to be a flag rather than a coincidence. The old hard-coded set could
    not cover channels added by a later SLAM run.
    """
    found = set()
    for logical, arr in arrays.items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        exact = np.mean(np.abs(finite - SENTINEL) <= SENTINEL_TOL)
        if exact > threshold and finite.min() >= SENTINEL - SENTINEL_TOL:
            found.add(logical)
    return found


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def get_field(las, name):
    if name in las.point_format.dimension_names:
        return np.asarray(las[name], dtype=np.float64)
    return None


def add_or_overwrite(las, name, values, dtype):
    if name in las.point_format.dimension_names:
        las[name] = values.astype(dtype)
    else:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        las[name] = values.astype(dtype)


# --------------------------------------------------------------------------- #
# Feature matrix — NaN encoding for sentinels (HGBR handles natively)
# --------------------------------------------------------------------------- #
def build_feature_matrix(arrays, sentinel_fields=frozenset()):
    cols, names = [], []
    for logical, arr in arrays.items():
        col = arr.copy()
        if logical in sentinel_fields:
            col[arr <= SENTINEL + SENTINEL_TOL] = np.nan
        cols.append(col)
        names.append(logical)
    return np.column_stack(cols), names


# --------------------------------------------------------------------------- #
# Per-channel AUC at a single τ
# --------------------------------------------------------------------------- #
def _curve(y_true, y_score, rng=np.random.default_rng(0)):
    """ROC curve on at most CURVE_MAX_POINTS samples, for plotting only."""
    if len(y_true) > CURVE_MAX_POINTS:
        sel = rng.choice(len(y_true), CURVE_MAX_POINTS, replace=False)
        y_true, y_score = y_true[sel], y_score[sel]
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)
    return fpr.astype(np.float32), tpr.astype(np.float32)


def compute_channel_aucs(c2c, arrays, tau, sentinel_fields=frozenset()):
    label = (c2c > tau).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return {}

    channel_info = {}
    for logical, arr in arrays.items():
        # Aggregated channels mark "no valid neighbourhood" with NaN, which
        # roc_auc_score cannot consume; sentinel channels use -1.
        valid = np.isfinite(arr)
        if logical in sentinel_fields:
            valid &= arr > SENTINEL + SENTINEL_TOL
        if valid.all():
            y_true, y_score = label, arr
        else:
            y_true, y_score = label[valid], arr[valid]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        auc       = roc_auc_score(y_true, y_score)
        direction = 1 if auc >= 0.5 else -1
        auc_norm  = auc if auc >= 0.5 else 1.0 - auc
        fpr, tpr = _curve(y_true, y_score * direction)
        channel_info[logical] = dict(
            auc=float(auc), auc_norm=float(auc_norm),
            direction=direction, fpr=fpr, tpr=tpr,
        )
    return channel_info


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_roc_per_tau(c2c, channel_infos_all, predicted_c2c, taus, out_dir, base):
    for tau in taus:
        channel_info = channel_infos_all.get(tau, {})
        if not channel_info:
            continue

        label   = (c2c > tau).astype(int)
        n_pos   = int(label.sum())
        n       = len(label)
        tau_cm  = tau * 100
        tau_tag = f"{int(round(tau_cm)):03d}cm"

        auc_hgbr = roc_auc_score(label, predicted_c2c)
        fpr_hgbr, tpr_hgbr = _curve(label, predicted_c2c)

        # One curve per channel is unreadable past ~10 channels: the feature set
        # is now 123 wide, the tab10 cycle repeats every 10, and the legend was
        # longer than the plot. Draw the best TOP_N in colour, the rest as a grey
        # envelope, and put a ranked AUC bar chart beside it.
        TOP_N = 10
        ranked = sorted(channel_info.items(), key=lambda kv: -kv[1]["auc_norm"])
        top, rest = ranked[:TOP_N], ranked[TOP_N:]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
        ax = axes[0]
        for logical, info in rest:
            ax.plot(info["fpr"], info["tpr"], color="#cccccc", linewidth=0.6, zorder=1)
        cmap = plt.cm.tab10
        for i, (logical, info) in enumerate(top):
            arrow = "\u2191" if info["direction"] > 0 else "\u2193"
            ax.plot(info["fpr"], info["tpr"], color=cmap(i % 10), linewidth=1.8,
                    zorder=3, label=f"{arrow} {logical}  {info['auc_norm']:.3f}")
        ax.plot(fpr_hgbr, tpr_hgbr, color="black", linewidth=2.5, linestyle="--",
                zorder=4, label=f"HGBR (q90)  {auc_hgbr:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC  \u2014  tau = {tau_cm:.1f} cm   "
                     f"outliers {n_pos:,} ({100*n_pos/n:.1f}%)\n"
                     f"top {len(top)} of {len(channel_info)}; rest in grey",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")

        ax2 = axes[1]
        names = [k for k, _ in ranked]
        aucs = [v["auc_norm"] for _, v in ranked]
        fams = [_family_of(k) for k in names]
        ax2.barh(np.arange(len(names))[::-1], aucs,
                 color=[FAMILY_COLOUR[f] for f in fams], height=0.9)
        ax2.axvline(0.5, color="#444444", linewidth=0.8)
        ax2.axvline(auc_hgbr, color="black", linewidth=1.5, linestyle="--")
        step = max(1, len(names) // 40)
        ax2.set_yticks(np.arange(len(names))[::-1][::step])
        ax2.set_yticklabels(names[::step], fontsize=6, family="monospace")
        ax2.set_xlim(0.45, max(0.75, max(aucs + [auc_hgbr]) + 0.03))
        ax2.set_xlabel("AUC (direction-normalised)")
        ax2.set_title(f"all {len(names)} channels by AUC", fontsize=10)
        from matplotlib.patches import Patch
        ax2.legend(handles=[Patch(facecolor=FAMILY_COLOUR[f], label=FAMILY_LABEL[f])
                            for f in ("geom", "agg", "sensor", "slam") if f in set(fams)]
                           + [plt.Line2D([0], [0], color="black", ls="--",
                                         label=f"HGBR {auc_hgbr:.3f}")],
                   fontsize=7, loc="lower right")
        ax2.grid(axis="x", alpha=0.25)

        out_path = os.path.join(out_dir, f"{base}_roc_{tau_tag}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"    saved → {out_path}")


FAMILY_COLOUR = {"geom": "#1b9e77", "slam": "#d95f02",
                 "sensor": "#e7298a", "agg": "#7570b3"}
FAMILY_LABEL = {"geom": "E geometric", "slam": "C SLAM-internal",
                "sensor": "A/B sensor + within-scan", "agg": "D local aggregation"}
_SENSOR = {"intensity", "intensity_orig", "reflectivity", "ambient", "ring",
           "scan_time", "range", "range_detach", "range_edge", "int_detach",
           "ring_rough", "ring_detach"}


def _family_of(name):
    """Which of the section-2 families a channel belongs to."""
    if "__" in name:
        return "agg"
    if name.startswith(("mean_knn_dist", "roughness")):
        return "geom"
    return "sensor" if name in _SENSOR else "slam"


def plot_predicted_vs_actual(c2c, predicted_c2c, out_dir, base):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    lim = max(float(np.percentile(c2c, 99)), float(np.percentile(predicted_c2c, 99)))
    hb  = ax.hexbin(c2c, predicted_c2c, gridsize=80, bins="log", cmap="plasma",
                    extent=[0, lim, 0, lim])
    ax.plot([0, lim], [0, lim], color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    plt.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("Actual C2C distance (m)")
    ax.set_ylabel("Predicted C2C — 90th quantile (m)")
    coverage = float((c2c < predicted_c2c).mean())
    ax.set_title(f"HGBR quantile=0.9 — coverage: {coverage:.1%}  (target 90.0%)")
    out_path = os.path.join(out_dir, f"{base}_predicted_vs_actual.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved predicted vs actual → {out_path}")


def plot_feature_importance(model, X_val, y_val, feat_names, out_dir, base, rng, n_rep=5):
    print(f"  Computing permutation importance on {len(y_val):,} points ...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_rep,
        random_state=int(rng.integers(0, 2**31)),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    imp_mean = result.importances_mean
    imp_std  = result.importances_std

    # Was figsize=(7, 0.4*n+1): at 123 features that is a 50-inch-tall figure.
    # Show the top TOP_N, coloured by family, and say what was left off.
    TOP_N = 25
    order = np.argsort(imp_mean)[-TOP_N:]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.30 * len(order) + 1.6)),
                           constrained_layout=True)
    y_pos = np.arange(len(order))
    ax.barh(y_pos, imp_mean[order], xerr=imp_std[order],
            color=[FAMILY_COLOUR[_family_of(feat_names[i])] for i in order],
            edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(feat_names)[order], fontsize=8, family="monospace")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation importance  (mean decrease in neg-MAE)")
    ax.set_title(f"HGBR feature importances \u2014 top {len(order)} of {len(feat_names)}")
    from matplotlib.patches import Patch
    fams = {_family_of(feat_names[i]) for i in order}
    ax.legend(handles=[Patch(facecolor=FAMILY_COLOUR[f], label=FAMILY_LABEL[f])
                       for f in ("geom", "agg", "sensor", "slam") if f in fams],
              fontsize=8, loc="lower right")
    out_path = os.path.join(out_dir, f"{base}_feature_importance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance → {out_path}")

    return {feat_names[i]: float(imp_mean[i]) for i in range(len(feat_names))}


def plot_auc_summary(all_results, hgbr_aucs, out_dir, base):
    taus_cm  = [r["tau"] * 100 for r in all_results]
    channels = list(all_results[0]["per_channel_auc"].keys())

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    cmap    = plt.cm.tab10
    colors  = cmap(np.linspace(0, 0.9, len(channels)))
    for ch, color in zip(channels, colors):
        aucs = [r["per_channel_auc"].get(ch, np.nan) for r in all_results]
        ax.plot(taus_cm, aucs, marker="o", color=color, linewidth=1.5, label=ch)

    hgbr_vals = [hgbr_aucs.get(r["tau"], np.nan) for r in all_results]
    ax.plot(taus_cm, hgbr_vals, marker="s", color="black", linewidth=2.5,
            linestyle="--", label="HGBR (q90)")

    ax.set_xlabel("τ (cm)")
    ax.set_ylabel("AUC  (normalized ≥ 0.5)")
    ax.set_title("Discriminative power vs. outlier threshold τ")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, f"{base}_auc_vs_tau.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved AUC summary → {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _distinct_names(paths):
    """Short, distinguishing names for the datasets being pooled.

    The basename alone is not enough: the convention is
    <scene>/labeled/global_map_feat.las, so every file is called the same thing
    and --holdout would have nothing to match on. Walk up the path until the
    names separate, skipping directories that carry no information.
    """
    skip = {"labeled", "labelled", "labels", "out", "output", "."}
    parts = [os.path.abspath(p).split(os.sep) for p in paths]
    stems = [os.path.splitext(q[-1])[0] for q in parts]
    if len(set(stems)) == len(stems):
        return stems
    out = []
    for q, stem in zip(parts, stems):
        for d in reversed(q[:-1]):
            if d and d.lower() not in skip:
                out.append(d)
                break
        else:
            out.append(stem)
    if len(set(out)) != len(out):        # still colliding: fall back to full stems
        out = [f"{d}/{s}" for d, s in zip(out, stems)]
    return out


def load_many(paths, domain_field, no_domain, max_per_dataset, rng,
              split_field=None, explicit_channels=None, group_field=None):
    """Load and pool several labelled clouds for joint training.

    Each file is filtered to its evaluation domain, dropped of unlabelled
    points, subsampled to `max_per_dataset`, and tagged with a dataset index so
    results can be split per dataset afterwards. Only the pooled subsample is
    held, so several hundred-million-point clouds stay inside memory.

    `group_field` (typically `voxel_id`) is carried through unmodified as a
    fifth return value, for callers that need to split rows by spatial group
    rather than at random. It is never a model input. Values are per file, so
    the caller must combine them with the dataset index before using them.
    """
    chunks_c2c, chunks_arrays, chunks_id, chunks_grp, names = [], [], [], [], []
    names_sub, all_channels = [], []
    common = None
    read_chunk = 10_000_000
    dataset_names = _distinct_names(paths)

    for di, path in enumerate(paths):
        name = dataset_names[di]
        print(f"\nReading {path} ...")

        # Streamed, not laspy.read: a labelled outdoor cloud is 14 GB on disk
        # and would otherwise be resident in full before any subsampling.
        with laspy.open(path) as f:
            total = f.header.point_count
            available = list(f.header.point_format.dimension_names)
        if C2C_FIELD not in available:
            sys.exit(f"ERROR: '{C2C_FIELD}' not found in {path}")
        channels = discover_channels(path, explicit_channels)
        all_channels.append(channels)
        print(f"  channels: {', '.join(l for l, _ in channels)}")
        present = {logical for logical, raw in channels if raw in available}
        common = present if common is None else (common & present)
        if split_field and split_field not in available:
            sys.exit(f"ERROR: --split-field '{split_field}' not in {path}")

        # Pass 1: how many points survive the domain filter, so pass 2 knows
        # what fraction to keep.
        usable = 0
        with laspy.open(path) as f:
            for chunk in f.chunk_iterator(read_chunk):
                keep = np.asarray(chunk[C2C_FIELD]) >= 0
                if not no_domain and domain_field in available:
                    keep &= np.asarray(chunk[domain_field]) != 0
                usable += int(keep.sum())
        if usable == 0:
            sys.exit(f"ERROR: {path} has no usable points")

        frac = 1.0 if not max_per_dataset else min(1.0, max_per_dataset / usable)
        parts_c2c, parts_arr, parts_split, parts_grp, kept = [], [], [], [], 0
        with laspy.open(path) as f:
            for chunk in f.chunk_iterator(read_chunk):
                keep = np.asarray(chunk[C2C_FIELD]) >= 0
                if not no_domain and domain_field in available:
                    keep &= np.asarray(chunk[domain_field]) != 0
                if frac < 1.0:
                    keep &= rng.random(len(keep)) < frac
                if not keep.any():
                    continue
                parts_c2c.append(np.asarray(chunk[C2C_FIELD])[keep].astype(np.float64))
                parts_arr.append({
                    logical: np.asarray(chunk[raw])[keep].astype(np.float64)
                    for logical, raw in channels if raw in available})
                if split_field:
                    parts_split.append(np.asarray(chunk[split_field])[keep])
                if group_field and group_field in available:
                    parts_grp.append(np.asarray(chunk[group_field])[keep]
                                     .astype(np.int64))
                kept += int(keep.sum())

        print(f"  {total:,} points, {usable:,} usable -> {kept:,} loaded"
              + (f" ({100 * frac:.1f}% sample)" if frac < 1.0 else ""))
        file_c2c = np.concatenate(parts_c2c)
        file_arr = {k: np.concatenate([p[k] for p in parts_arr])
                    for k in parts_arr[0]}
        file_grp = (np.concatenate(parts_grp) if parts_grp
                    else np.full(len(file_c2c), -1, dtype=np.int64))
        del parts_c2c, parts_arr, parts_grp

        if split_field:
            # Each distinct value of the split field becomes its own pseudo
            # dataset, which lets the existing --holdout machinery hold out a
            # section of one cloud rather than a whole file.
            values = np.concatenate(parts_split)
            del parts_split
            for v in np.unique(values):
                rows = values == v
                sub = f"{name}|{split_field}={v}"
                names_sub.append(sub)
                chunks_c2c.append(file_c2c[rows])
                chunks_arrays.append({k: a[rows] for k, a in file_arr.items()})
                chunks_id.append(np.full(int(rows.sum()),
                                         len(names_sub) - 1, dtype=np.int32))
                chunks_grp.append(file_grp[rows])
                print(f"    {sub}: {int(rows.sum()):,} points")
        else:
            names_sub.append(name)
            chunks_c2c.append(file_c2c)
            chunks_arrays.append(file_arr)
            chunks_id.append(np.full(kept, len(names_sub) - 1, dtype=np.int32))
            chunks_grp.append(file_grp)

    ordered = [k for k in dict.fromkeys(
        l for group in all_channels for l, _ in group) if k in common]
    dropped = sorted({k for a in chunks_arrays for k in a} - set(ordered))
    if dropped:
        print(f"\n  NOTE: dropping channels missing from at least one dataset: "
              f"{', '.join(dropped)}")

    pooled = {k: np.concatenate([a[k] for a in chunks_arrays]) for k in ordered}
    return (np.concatenate(chunks_c2c), pooled,
            np.concatenate(chunks_id), names_sub,
            np.concatenate(chunks_grp))


def predict_full_cloud(path, model, channel_order, domain_field,
                       sentinel_fields=frozenset(),
                       chunk_size=10_000_000, verbose=True):
    """Score every point of a cloud, streaming.

    `channel_order` must be the exact channel list the model was fitted on:
    build_feature_matrix lays columns out in dict order, so a different order
    here would silently feed the model the wrong features.
    """
    with laspy.open(path) as f:
        total = f.header.point_count
        available = list(f.header.point_format.dimension_names)
    raw_of = {logical: raw for logical, raw in discover_channels(path)}
    missing = [c for c in channel_order if raw_of[c] not in available]
    if missing:
        sys.exit(f"ERROR: {path} lacks channels the model needs: {missing}")

    out = np.empty(total, dtype=np.float64)
    at = 0
    with laspy.open(path) as f:
        for chunk in f.chunk_iterator(chunk_size):
            block = {c: np.asarray(chunk[raw_of[c]]).astype(np.float64)
                     for c in channel_order}
            X_chunk, _ = build_feature_matrix(block, sentinel_fields)
            out[at:at + len(X_chunk)] = model.predict(X_chunk)
            at += len(X_chunk)
            if verbose:
                print(f"  predicting: {at:,} / {total:,}", end="\r", flush=True)
    if verbose:
        print(f"  predicted for all {total:,} points" + " " * 20)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="HGBR C2C regression + per-channel AUC for point cloud confidence channels.")
    ap.add_argument("input", nargs="+",
                    help="Input LAS file(s). Give several to pool datasets for "
                         "training; each is subsampled to --max-per-dataset "
                         "and tagged so results can be broken down per dataset.")
    ap.add_argument("--tau", nargs="+", type=float,
                    default=[0.01, 0.02, 0.05, 0.10],
                    help="C2C thresholds in metres for ROC/AUC plots (default: 0.01 0.02 0.05 0.10)")
    ap.add_argument("--out",     default=".", help="Output directory for figures / JSON")
    ap.add_argument("--out-las", default=None,
                    help="Path to write updated LAS (default: overwrite input)")
    ap.add_argument("--fit-n", type=int, default=500_000,
                    help="Points to subsample for HGBR training (0 = all, default: 500 000)")
    ap.add_argument("--imp-n", type=int, default=50_000,
                    help="Points for permutation importance (default: 50 000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domain-field", default=DOMAIN_FIELD,
                    help=f"Restrict the analysis to points where this field is "
                         f"non-zero (default: {DOMAIN_FIELD}, written by "
                         f"prepare_labels_e57.py; "
                         f"ignored if the field is absent)")
    ap.add_argument("--no-domain", action="store_true",
                    help="Use every point, including those outside the "
                         "evaluation domain (not recommended: labels there are "
                         "not trustworthy)")
    ap.add_argument("--max-per-dataset", type=int, default=20_000_000,
                    help="Random subsample cap per input file (0 = no cap, "
                         "default: 20 000 000). Keeps several 100 M-point "
                         "clouds inside memory.")
    ap.add_argument("--channels", nargs="+", default=None,
                    help="Explicit list of stored dimension names to use as "
                         "features. Default: every extra dimension that is not "
                         "a pipeline output or bookkeeping field.")
    ap.add_argument("--split-field", default=None,
                    help="Per-point field to split each cloud on, e.g. "
                         "gt_group from prepare_labels_e57.py. Each distinct "
                         "value becomes its own pseudo-dataset, so one section "
                         "of a cloud can be held out from another.")
    ap.add_argument("--holdout", default=None,
                    help="Dataset name (input file stem) to exclude from "
                         "fitting and score separately. This is the "
                         "dataset-level train/test split: a random split over "
                         "points is meaningless here because neighbouring "
                         "points are highly correlated.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng  = np.random.default_rng(args.seed)
    multi = len(args.input) > 1
    base = ("+".join(os.path.splitext(os.path.basename(p))[0][:20]
                     for p in args.input) if multi
            else os.path.splitext(os.path.basename(args.input[0]))[0])

    # One streaming loader for both cases. The domain filter and the per-dataset
    # subsample are applied while reading, so a 14 GB labelled cloud never has
    # to be resident.
    c2c, arrays, dataset, dataset_names, _groups = load_many(
        args.input, args.domain_field, args.no_domain,
        args.max_per_dataset, rng, split_field=args.split_field,
        explicit_channels=args.channels)
    n = len(c2c)
    if multi:
        print(f"\n  Pooled: {n:,} points from {len(dataset_names)} datasets")
    print(f"  Fields loaded: {', '.join(arrays)}")
    sentinels = detect_sentinels(arrays)
    print(f"  Sentinel (-1 = invalid) channels detected: "
          f"{', '.join(sorted(sentinels)) if sentinels else 'none'}")

    # ------------------------------------------------------------------ #
    # Evaluation domain
    #
    # Points the RTC could not have scanned carry a C2C value that says
    # nothing about SLAM quality, so training on them injects label noise and
    # scoring on them biases the comparison. prepare_labels_e57.py flags them
    # rather than deleting them; here they are dropped from the analysis.
    # ------------------------------------------------------------------ #
    # The loader has already applied the domain filter and dropped C2C = -1
    # (no GT neighbour inside the search cap, so no distance label exists).

    # ------------------------------------------------------------------ #
    # Dataset-level train/test split
    #
    # Splitting randomly over points would leak: neighbouring points share a
    # scan, a pose and a surface, so a random holdout is almost the training
    # set. Holding out a whole dataset is the honest version.
    # ------------------------------------------------------------------ #
    fit_mask = np.ones(n, dtype=bool)
    if args.holdout is not None:
        # Accept the full name, or just the trailing value when unambiguous —
        # "--holdout 2" instead of "--holdout cloud|gt_group=2".
        if args.holdout in dataset_names:
            held = dataset_names.index(args.holdout)
        else:
            hits = [i for i, nm in enumerate(dataset_names)
                    if nm.endswith(f"={args.holdout}") or nm.endswith(args.holdout)]
            if len(hits) != 1:
                sys.exit(f"ERROR: --holdout '{args.holdout}' matched {len(hits)} "
                         f"of {dataset_names}; give the full name")
            held = hits[0]
            print(f"  --holdout '{args.holdout}' resolved to "
                  f"'{dataset_names[held]}'")
        args.holdout = dataset_names[held]
        fit_mask = dataset != held
        print(f"\n  Holdout: '{args.holdout}' — fitting on "
              f"{int(fit_mask.sum()):,} points, scoring the model on "
              f"{int((~fit_mask).sum()):,} held-out points")
        if fit_mask.sum() == 0:
            sys.exit("ERROR: holding out that dataset leaves nothing to fit on.")

    # ------------------------------------------------------------------ #
    # Per-channel AUC at each τ
    # ------------------------------------------------------------------ #
    print("\nPer-channel AUC analysis ...")
    all_results       = []
    channel_infos_all = {}
    for tau in sorted(args.tau):
        info   = compute_channel_aucs(c2c, arrays, tau, sentinels)
        label  = (c2c > tau).astype(int)
        n_pos  = int(label.sum())
        n_neg  = n - n_pos
        tau_cm = tau * 100
        print(f"\n  τ = {tau_cm:.1f} cm  |  "
              f"outliers {n_pos:,} ({100*n_pos/n:.1f}%)  "
              f"inliers {n_neg:,} ({100*n_neg/n:.1f}%)")
        for logical, d in info.items():
            arrow = "↑ higher" if d["direction"] > 0 else "↓ lower"
            print(f"    {logical:15s}  AUC={d['auc_norm']:.4f}  ({arrow} → outlier)")
        channel_infos_all[tau] = info
        all_results.append(dict(
            tau=tau, n_outlier=n_pos, n_inlier=n_neg,
            outlier_rate=float(n_pos / n),
            per_channel_auc={k: v["auc_norm"] for k, v in info.items()},
            per_channel_direction={k: v["direction"] for k, v in info.items()},
        ))

    if not all_results:
        sys.exit("No valid results produced.")

    # ------------------------------------------------------------------ #
    # HGBR: fit on raw C2C distance (regression, not classification)
    # ------------------------------------------------------------------ #
    print("\nBuilding feature matrix ...")
    X, feat_names = build_feature_matrix(arrays, sentinels)
    y = c2c

    fit_idx = np.flatnonzero(fit_mask)
    if args.fit_n and args.fit_n < len(fit_idx):
        sel = rng.choice(fit_idx, args.fit_n, replace=False)
        X_train, y_train = X[sel], y[sel]
        print(f"  Training on {len(sel):,} points sampled from the fitting set ...")
    else:
        X_train, y_train = X[fit_idx], y[fit_idx]
        print(f"  Training on all {len(fit_idx):,} points of the fitting set ...")

    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9,
        max_iter=300, learning_rate=0.1,
        early_stopping=True, validation_fraction=0.1,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)
    print(f"  Fitted in {model.n_iter_} iterations")

    print("  Predicting on full dataset ...")
    predicted_c2c = model.predict(X)
    coverage = float((y < predicted_c2c).mean())
    print(f"  Coverage (actual < predicted): {coverage:.1%}  (target: 90.0%)")

    # ------------------------------------------------------------------ #
    # HGBR AUC at each τ — use predicted_c2c as the outlier score
    # ------------------------------------------------------------------ #
    print("\nHGBR AUC at each τ ...")
    hgbr_aucs = {}
    dataset_summary = None
    for tau in sorted(args.tau):
        label = (c2c > tau).astype(int)
        if label.sum() == 0 or label.sum() == n:
            continue
        auc_hgbr     = float(roc_auc_score(label, predicted_c2c))
        hgbr_aucs[tau] = auc_hgbr
        for r in all_results:
            if r["tau"] == tau:
                r["hgbr_auc"] = auc_hgbr
        print(f"  τ = {tau*100:.1f} cm  HGBR AUC = {auc_hgbr:.4f}")

    # Per-dataset breakdown. With --holdout, the held-out rows are the only
    # honest estimate of how the model transfers; the rest were fitted on.
    if len(dataset_names) > 1:
        print("\nPer-dataset AUC (HGBR score) ...")
        per_dataset = {}
        for di, name in enumerate(dataset_names):
            rows = dataset == di
            if rows.sum() == 0:
                continue
            role = "HELD OUT" if (args.holdout == name) else "fitted on"
            entry = {"role": role, "n": int(rows.sum())}
            parts = []
            for tau in sorted(args.tau):
                label = (c2c[rows] > tau).astype(int)
                if label.sum() in (0, len(label)):
                    continue
                a = float(roc_auc_score(label, predicted_c2c[rows]))
                entry[f"auc_tau_{tau}"] = a
                parts.append(f"τ={tau*100:.0f}cm AUC={a:.4f}")
            per_dataset[name] = entry
            print(f"  {name:<28s} [{role:<9s}] {int(rows.sum()):>10,} pts  "
                  + "  ".join(parts))
        # Kept separate: all_results is a list of per-tau records that the
        # summary plot iterates, so a differently shaped entry cannot go in it.
        dataset_summary = {"per_dataset": per_dataset, "holdout": args.holdout}

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    plot_roc_per_tau(c2c, channel_infos_all, predicted_c2c,
                     sorted(args.tau), args.out, base)

    plot_predicted_vs_actual(c2c, predicted_c2c, args.out, base)

    imp_n = min(args.imp_n, n)
    val_idx = rng.choice(n, imp_n, replace=False)
    feat_imp = plot_feature_importance(
        model, X[val_idx], y[val_idx], feat_names, args.out, base, rng)
    for r in all_results:
        r["feature_importances"] = feat_imp

    plot_auc_summary(all_results, hgbr_aucs, args.out, base)

    # ------------------------------------------------------------------ #
    # Save JSON
    # ------------------------------------------------------------------ #
    json_path = os.path.join(args.out, f"{base}_outlier_analysis.json")
    with open(json_path, "w") as fh:
        json.dump({"per_tau": all_results, "datasets": dataset_summary}
                  if dataset_summary else all_results, fh, indent=2)
    print(f"Saved JSON → {json_path}")

    # ------------------------------------------------------------------ #
    # Write predicted_c2c back to LAS
    # ------------------------------------------------------------------ #
    if multi:
        print("\nSkipping the LAS write-back: several inputs were pooled, so "
              "there is no single cloud to attach predicted_c2c to. Re-run on "
              "one file to score it, or use --holdout to score a dataset the "
              "model never saw.")
        print("\nDone.")
        return

    src = args.input[0]
    out_las = args.out_las or src
    print(f"\nWriting predicted_c2c to {out_las} ...")

    # The model is fitted on the evaluation domain only, but the filter has to
    # run on the whole cloud, so predict everywhere. Both the prediction and the
    # write stream, so this works on clouds far larger than memory.
    predicted_full = predict_full_cloud(src, model, list(arrays),
                                        args.domain_field, sentinels)
    lasio.write_las_with_fields(
        src, out_las, {"predicted_c2c": (predicted_full, np.float32)})
    print(f"Saved LAS → {out_las}")
    print("\nDone.")


if __name__ == "__main__":
    main()
