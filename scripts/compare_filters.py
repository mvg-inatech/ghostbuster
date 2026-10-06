#!/usr/bin/env python3
"""
The headline experiment: can raw Sensor and SLAM-internal confidence channels together 
with geometric features be a good predictor of outliers, and does combining them beat 
either alone?

Four scorers are compared on identical data at identical removal budgets:

  none        no filtering, the reference point
  sor         classical Statistical Outlier Removal — mean distance to the k
              nearest neighbours. `mean_knn_dist` *is* the SOR statistic, so
              sweeping a threshold on it traces out every SOR operating point;
              the textbook mean + n*std cut-offs are marked on the curve.
  slam_only   boosted model over the SLAM-internal channels, with
              mean_knn_dist deliberately withheld
  slam_plus   the same model with mean_knn_dist added — SOR *combined with* the
              confidence channels

Each learned scorer is fitted under two objectives, because the two metrics
reported here reward different things and a model can only be optimal for one:

  q90    quantile regression at 0.9 — targets the tail, so it is the right
         objective for "how many gross outliers are left"
  mean   squared-error regression — predicts E[C2C | channels], which is
         exactly the score that minimises the mean C2C of whatever survives

Fitting only q90 and then reporting mean C2C, as the first version did, judged
the models on an objective none of them was trained for.

Comparing at a matched removal fraction is the point. A filter that deletes more
points will always look better on a mean-error metric, so "which is better" is
only meaningful at equal budget; the curves are all plotted against the fraction
removed.

Evaluation honesty: pass --split-field/--holdout to fit on one section and score
on another. Without it the models are scored on data they were fitted on and the
learned curves are optimistic — the SOR curve is unaffected either way, since it
has nothing to fit.

Usage:
    python compare_filters.py labeled.las --out figures_filters/
    python compare_filters.py labeled.las --split-field gt_group --holdout 2 \\
           --out figures_filters/
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from outlier_analysis import (load_many, build_feature_matrix,  # noqa: E402
                              detect_sentinels, DOMAIN_FIELD, SENTINEL_TOL)

KNN_CHANNEL = "mean_knn_dist"

# Thresholds for the AP-versus-tau figure, in metres. 1 cm is below the median
# C2C of every scene here, so it is the regime where "outlier" means ordinary
# surface noise; 10 cm is gross error only. The headline 2 cm and 5 cm sit
# inside the sweep so the figure shows them in context.
SWEEP_TAUS = (0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10)

# Anything matching these is post-hoc cloud geometry, not a SLAM-internal
# signal. The split has to be by *bucket*, not by single channel: once
# multi-scale kNN and roughness exist, "without kNN" is no longer the same as
# "without geometry".
GEOM_PREFIXES = ("mean_knn_dist", "roughness")

# Raw per-point sensor readings. These are NOT SLAM-internal confidence: the
# driver would hand them over with no SLAM running at all. Reported as their own
# bucket because "sensor channels help" and "SLAM confidence channels help" are
# different claims. The split affects only how importance is attributed; every
# model is fitted on the same channels either way.
SENSOR_CHANNELS = frozenset({
    "intensity", "intensity_orig", "reflectivity",   # returned signal
    "ambient", "ring", "scan_time",                  # acquisition conditions
    "range",                                         # derived from xyz by the converter
})

# Family B: computed on the raw organised scan, before down_sampling_voxel and
# before merging, from the sensor's own returns and its detector topology (same
# ring, adjacent azimuth). No SLAM state is involved, so these are NOT family C,
# and the `geom` bucket means post-hoc 3D neighbourhoods on the assembled cloud,
# which is a different neighbourhood entirely -- measured correlation with
# mean_knn_dist is +0.04 to +0.05.
#
# Two generations of names are listed. The in-SLAM implementation writes
# range_detach / range_edge / ring_rough; ring_detach / ring_edge are the older
# converter-side names, kept so clouds written before the move into the SLAM
# are still bucketed as family B rather than as SLAM-internal.
WITHIN_SCAN_CHANNELS = frozenset({
    "range_detach", "range_edge", "int_detach", "ring_rough",
    "ring_detach", "ring_edge",                      # older converter-side names
})

# Columns that duplicate another column exactly and are dropped before fitting.
# Verified bit-identical: intensity_orig vs intensity (max |diff| 0.0 over 3 M
# points) and mean_knn_dist vs mean_knn_dist_k6. Keeping both of a pair adds no
# information, splits that channel's importance across two bars, and lets a
# feature-count ablation spend one of its slots on a copy.
#
# Only the RAW intensity_orig is dropped: the aggregates are named
# intensity_orig__* and are the only aggregates of intensity there are, so they
# stay and CHANNEL_ALIASES rolls them onto `intensity`.
REDUNDANT_CHANNELS = frozenset({"intensity_orig", "mean_knn_dist"})


def is_geometric(name):
    return name.startswith(GEOM_PREFIXES)


def bucket_of(name):
    """Which feature family a channel belongs to.

    Five families, matching the write-up: A raw sensor, B within-scan, C
    SLAM-internal, D local aggregation, E purely geometric. The buckets are
    named sensor / within / slam / agg / geom.

    The aggregated channels are a family of their own: their *values* are sensor
    or SLAM quantities but their *topology* is the assembled cloud's, so they
    borrow neighbourhood context the raw channels do not have. Counting them as
    SLAM would overstate what the SLAM alone can do, which is the claim
    under test.
    """
    if "__" in name:
        return "agg"
    if is_geometric(name):
        return "geom"
    if name in WITHIN_SCAN_CHANNELS:
        return "within"
    if name in SENSOR_CHANNELS:
        return "sensor"
    return "slam"


# `intensity_orig` is written by the label pipeline as a float32 copy of the
# LAS `intensity` field. Verified bit-identical on one indoor scene (max |diff| 0.0
# over 3 M points), so the two are one channel and must roll up as one -- the
# raw value is called `intensity` and its aggregates `intensity_orig__*`, which
# would otherwise appear as two half-empty bars.
CHANNEL_ALIASES = {"intensity_orig": "intensity"}


def base_channel(name):
    """The channel an aggregate was derived from: plane_quality__dev_k30 ->
    plane_quality. A raw channel is its own base. Aliases are resolved, so
    intensity_orig__dev_k30 and intensity share one base."""
    b = name.split("__", 1)[0]
    return CHANNEL_ALIASES.get(b, b)


def buckets(names, fold_aggregates=False, merge_ab=False):
    """Column indices per bucket.

    With `fold_aggregates`, an aggregate joins the bucket of the channel it was
    derived from rather than a bucket of its own: obs_count__dev_k30 counts as
    SLAM, ambient__dev_k30 as sensor. That answers "what does this family
    contribute in total", which is the question when the aggregates are part of
    your pipeline. Note it is NOT interchangeable with the 4-way split: the
    aggregates borrow the final cloud's topology, so folding them in credits
    SLAM with neighbourhood context the SLAM alone never had. Both are reported.
    """
    out = {"geom": [], "slam": [], "sensor": [], "within": [], "agg": []}
    for i, n in enumerate(names):
        b = bucket_of(base_channel(n)) if fold_aggregates else bucket_of(n)
        out[b].append(i)
    if merge_ab:
        # A and B are both per-scan quantities available before any pose is
        # estimated -- the split between "what the driver reported" and "how the
        # return sits on the range image" is a distinction about provenance, not
        # about when the information exists. Permuting them together answers
        # "what is the scan worth", which is the question for an online filter.
        out["sensor_within"] = out.pop("sensor", []) + out.pop("within", [])
    return {k: v for k, v in out.items() if v}

METHOD_STYLE = {
    "none":           ("#666666", ":",  "no filter"),
    "oracle":         ("#000000", "-",  "oracle (best possible)"),
    "sor":            ("#d95f02", "-",  "SOR, single scale"),
    "geom_only_q90":  ("#e6ab02", "--", "geometry only, q90"),
    "sensor_only_q90":   ("#8c6d31", "--", "sensor channels only, q90"),
    "sensor_only_mean":  ("#8c6d31", "-",  "sensor channels only, mean"),
    "slam_int_only_q90": ("#7570b3", "--", "SLAM internals only, q90"),
    "slam_int_only_mean":("#7570b3", "-",  "SLAM internals only, mean"),
    "geom_only_mean": ("#e6ab02", "-",  "geometry only, mean"),
    # These two names predate the sensor/SLAM split and understated what is in
    # them: "SLAM + geometry" is EVERY channel -- geometry, SLAM-internal,
    # sensor and all the aggregates, which are the large majority of the columns
    # -- and "SLAM only" silently included the sensor channels too. The tags are
    # unchanged so stored results stay comparable; only the plot labels moved.
    "slam_only_q90":  ("#1b9e77", "--", "sensor+SLAM raw, no geometry, q90"),
    "slam_only_mean": ("#1b9e77", "-",  "sensor+SLAM raw, no geometry, mean"),
    "slam_plus_q90":  ("#2c7fb8", "--", "ALL channels, q90"),
    "slam_plus_mean": ("#2c7fb8", "-",  "ALL channels, mean"),
    "slam_agg_q90":   ("#e7298a", "--", "all but geometry, q90"),
    "slam_agg_mean":  ("#e7298a", "-",  "all but geometry, mean"),
}
_SINGLE_COLOURS = ["#7570b3", "#e7298a", "#66a61e", "#a6761d", "#666666"]


def style_of(tag):
    """Style for a scorer, inventing one for baselines added at run time."""
    if tag in METHOD_STYLE:
        return METHOD_STYLE[tag]
    if tag.startswith("single_"):
        name = tag[len("single_"):]
        i = sum(map(ord, name)) % len(_SINGLE_COLOURS)
        return (_SINGLE_COLOURS[i], "-.", f"{name} threshold")
    return ("#999999", ":", tag)


LEARNED = ["geom_only_q90", "geom_only_mean",
           "sensor_only_q90", "sensor_only_mean",
           "slam_int_only_q90", "slam_int_only_mean",
           "slam_only_q90", "slam_only_mean",
           "slam_plus_q90", "slam_plus_mean"]


def _grouped_holdout(groups, fraction, rng):
    """Split row indices so that no group straddles the boundary."""
    uniq = rng.permutation(np.unique(groups))
    n_val = max(1, int(round(len(uniq) * fraction)))
    if n_val >= len(uniq):                       # only one group: cannot split
        return None, None
    is_val = np.isin(groups, uniq[:n_val])
    return ~is_val, is_val


def _val_loss(model, X, y, objective, quantile):
    """Validation loss in the same units the model is optimising."""
    pred = model.predict(X)
    if objective == "q90":
        d = y - pred
        return float(np.mean(np.maximum(quantile * d, (quantile - 1.0) * d)))
    return float(np.mean((y - pred) ** 2))


def fit_scorer(X, y, seed, objective="q90", quantile=0.9, groups=None,
               max_iter=300, val_fraction=0.1, patience=3, step=10,
               verbose=False):
    """Fit one scorer. `objective` picks what the model is optimal for.

    Early stopping is done here rather than inside HistGradientBoostingRegressor,
    because sklearn carves its internal validation set out with a plain random
    `train_test_split`. That is invisible to the outer, group-aware protocol and
    it leaks badly: measured on one indoor scene, 99.6% of the rows in a random 10%
    validation split share a 6.25 cm `cell_id` with a training row -- roughly 46
    points per cell, so a random split almost never separates one. The stopping
    criterion was therefore reading a validation score inflated by neighbourhood
    memorisation, and stopping too late.

    Worse, that internal set is drawn from the *training scenes*, so it measures
    in-distribution error while the quantity of interest is error on a held-out
    scene. That is a distribution mismatch: more data does not shrink it, and
    the leak measurement above in fact rises slightly with sample size.

    With `groups` given, the validation set is instead held out whole-group. The
    caller should pass the scene index when the fit set spans more than one
    scene -- then the stopping criterion approximates transfer, which is what the
    experiment is about -- and `cell_id` otherwise, which at least removes the
    neighbourhood leak. The number of iterations chosen on that split is then
    used to refit on all the fit rows, so no data is spent on validation.

    Passing `groups=None` restores the old behaviour exactly, for comparison.
    """
    params = dict(max_iter=max_iter, learning_rate=0.1, random_state=seed)
    if objective == "q90":
        params.update(loss="quantile", quantile=quantile)
    elif objective == "mean":
        params.update(loss="squared_error")
    else:
        raise ValueError(f"unknown objective {objective}")

    if groups is None:
        model = HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=val_fraction, **params)
        model.fit(X, y)
        model.stopping_ = "sklearn-internal (random split)"
        return model

    tr, va = _grouped_holdout(np.asarray(groups),
                              val_fraction, np.random.default_rng(seed))
    if tr is None or tr.sum() == 0 or va.sum() == 0:
        model = HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=val_fraction, **params)
        model.fit(X, y)
        model.stopping_ = "sklearn-internal (too few groups to hold out)"
        return model

    # warm_start lets the same trees be extended, so the sweep costs about one
    # full fit rather than one per candidate iteration count.
    probe = HistGradientBoostingRegressor(
        **{**params, "max_iter": step}, early_stopping=False, warm_start=True)
    best, best_iter, waited = np.inf, step, 0
    for it in range(step, max_iter + 1, step):
        probe.set_params(max_iter=it)
        probe.fit(X[tr], y[tr])
        v = _val_loss(probe, X[va], y[va], objective, quantile)
        if v < best - 1e-12:
            best, best_iter, waited = v, it, 0
        else:
            waited += 1
            if waited >= patience:
                break
    if verbose:
        print(f"      grouped early stopping: {best_iter} iters "
              f"({int(va.sum()):,} rows in {len(np.unique(np.asarray(groups)[va])):,} "
              f"held-out groups)")

    # Refit on every fit row with the chosen depth of ensemble: the validation
    # split existed only to choose it.
    model = HistGradientBoostingRegressor(
        **{**params, "max_iter": best_iter}, early_stopping=False)
    model.fit(X, y)
    model.stopping_ = f"grouped holdout, {best_iter} iters"
    return model


def stopping_groups(dataset, cell_groups, fit_idx, mode="auto", verbose=True):
    """Pick what the early-stopping validation set should be held out by.

    Scene when the fit rows span more than one, because then the stopping
    criterion is measuring something close to cross-scene transfer. Otherwise
    the cell, which removes the neighbourhood leak even though it cannot say
    anything about transfer. `mode` forces one of them; "random" restores
    sklearn's leaky internal split, kept so the change can be measured.
    """
    if mode == "random":
        if verbose:
            print("  early stopping: sklearn internal random split "
                  "(leaky -- for comparison only)")
        return None, "random"
    if mode == "cell":
        if cell_groups is None:
            sys.exit("ERROR: --early-stopping-split cell needs --group-field")
        if verbose:
            print("  early stopping: holding out whole cells (forced)")
        return cell_groups[fit_idx], "cell"
    if mode == "scene":
        if len(np.unique(dataset[fit_idx])) < 2:
            sys.exit("ERROR: --early-stopping-split scene needs the fit set to "
                     "span at least two scenes")
        if verbose:
            print("  early stopping: holding out whole scenes (forced)")
        return dataset[fit_idx], "scene"

    # Scene-holdout needs at least three scenes in the fit set to be worth it.
    # With two, holding one out leaves a single scene to probe on, so the
    # iteration count would be chosen from half the data and from a fit that
    # cannot see any between-scene variation at all -- the cell split is the
    # better trade there, since it keeps every scene in training and still
    # removes the neighbourhood leak.
    scenes = np.unique(dataset[fit_idx])
    if len(scenes) > 2:
        if verbose:
            print(f"  early stopping: holding out whole scenes "
                  f"({len(scenes)} in the fit set)")
        return dataset[fit_idx], "scene"
    if len(scenes) == 2 and cell_groups is not None and verbose:
        print("  early stopping: 2 scenes in the fit set -- holding out whole "
              "cells rather than a scene (see stopping_groups)")
    if cell_groups is None:
        if verbose:
            print("  early stopping: no groups available, sklearn default "
                  "(random split -- leaky)")
        return None, "random"
    if verbose:
        print("  early stopping: single scene, holding out whole cells")
    return cell_groups[fit_idx], "cell"


def ranking_metrics(score, c2c, tau, extra_taus=(0.02,), sweep=SWEEP_TAUS,
                    spearman_n=2_000_000, seed=0):
    """Threshold-free quality of the ranking itself.

    The operating-point sweep answers "how clean is what remains after
    discarding f?", which depends on choosing f. AP and ROC AUC ask instead
    whether outliers are ranked above inliers at all, with no operating point
    to pick -- which is what makes them comparable across scenes whose outlier
    rates differ by 4x. AP is the more informative of the two here: the
    positive class is the minority (9 % in the warehouses), and ROC AUC is
    optimistic on imbalanced data because the true-negative pool is huge.
    """
    def at(t):
        y = (c2c > t).astype(np.int8)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            return {"ap": float("nan"), "roc_auc": float("nan"),
                    "positive_rate": float(n_pos) / len(y)}
        return {
            "ap": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "positive_rate": float(n_pos) / len(y),
        }

    # Spearman needs no threshold at all: it asks whether the predicted C2C
    # orders the points the way the true C2C does, over the whole range rather
    # than either side of one cut. Ties are not a problem -- tied values take
    # the mean of the ranks they span -- which matters for a method like
    # PointCleanNet whose scores are >50 % exact zeros: the tied block genuinely
    # carries no ordering, and Spearman reporting that is correct.
    #
    # Subsampled because the full O(n log n) sort over 6 M points is repeated
    # for every scorer, and the estimate is stable long before that.
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(c2c), spearman_n, replace=False)
           if len(c2c) > spearman_n else slice(None))
    rho = float(spearmanr(score[idx], c2c[idx]).correlation)

    out = at(tau)
    out["spearman"] = rho
    # The 5 cm threshold is the headline, but 2 cm is the one that says whether
    # a method separates ordinary surface noise from real error rather than
    # only catching gross flyers. Both are reported so neither can be chosen
    # after the fact.
    out["by_tau"] = {f"{t:.3f}": at(t) for t in sorted({tau, *extra_taus})}
    # A sweep for the AP-versus-tau figure. One threshold is a choice the reader
    # has to take on trust; the curve shows where each method wins and where the
    # orderings cross, and makes the two headline thresholds visibly ordinary
    # points on it rather than picked after the fact.
    out["sweep"] = {f"{t:.3f}": at(t) for t in sweep}
    return out


def evaluate(score, c2c, fractions, tau):
    """Metrics after removing the worst-scoring `f` fraction, for each f.

    `score` is "higher = more suspicious". Ties are broken by the sort, so the
    realised removal fraction can differ slightly from the requested one on
    heavily tied scores; the realised value is what gets reported.
    """
    n = len(score)
    order = np.argsort(score, kind="stable")   # best first, worst last
    ordered_c2c = c2c[order]
    total_inliers = int((c2c <= tau).sum())

    rows = []
    for f in fractions:
        keep_n = int(round(n * (1.0 - f)))
        keep_n = max(keep_n, 1)
        kept = ordered_c2c[:keep_n]
        inliers = int((kept <= tau).sum())
        precision = inliers / keep_n
        recall = inliers / total_inliers if total_inliers else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        rows.append({
            "requested_removed": float(f),
            "removed": float(1.0 - keep_n / n),
            "kept": int(keep_n),
            "mean_c2c": float(kept.mean()),
            "median_c2c": float(np.median(kept)),
            "p90_c2c": float(np.percentile(kept, 90)),
            "outlier_rate": float((kept > tau).mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    return rows


def normalise_per_scene(arrays, dataset, n_scenes, mode, n_edges=1001,
                        sub=2_000_000, seed=0, verbose=True):
    """Put every feature on a per-scene scale before training.

    Without this the comparison is rigged in SOR's favour. SOR's threshold is
    computed from the *test* scene's own mean and standard deviation, so it
    recalibrates to each scene for free, while a learned model carries the
    training scenes' absolute thresholds with it. `mean_knn_dist_k100 > 3 cm`
    means different things in clouds of different density, so the model is
    asked to transfer a number that does not transfer.

    `rank` maps each value to its within-scene quantile, which removes scale and
    distribution shape alike. `zscore` is the gentler robust version, dividing
    the deviation from the median by the IQR.

    This is transductive: it uses the held-out scene's feature *distribution*,
    never its labels. SOR is transductive in exactly the same way and to the
    same degree, which is the point -- it equalises the two, and it leaves SOR's
    own curve untouched because a within-scene monotone map cannot reorder it.
    """
    if mode == "none":
        return
    rng = np.random.default_rng(seed)
    qs = np.linspace(0.0, 1.0, n_edges)
    for si in range(n_scenes):
        m = dataset == si
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        for name, arr in arrays.items():
            v = arr[idx]
            good = np.isfinite(v)
            if not good.any():
                continue
            gv = v[good]
            samp = gv if len(gv) <= sub else gv[rng.choice(len(gv), sub, replace=False)]
            if mode == "rank":
                edges = np.quantile(samp, qs)
                edges = np.maximum.accumulate(edges)
                if edges[-1] <= edges[0]:
                    continue                      # constant in this scene
                v[good] = np.interp(gv, edges, qs)
            elif mode == "zscore":
                med = float(np.median(samp))
                q1, q3 = np.percentile(samp, [25, 75])
                iqr = float(q3 - q1)
                if iqr <= 0:
                    continue
                v[good] = (gv - med) / iqr
            else:
                raise ValueError(f"unknown normalisation {mode}")
            arr[idx] = v
    if verbose:
        print(f"  per-scene feature normalisation: {mode} "
              f"over {n_scenes} scene(s), {len(arrays)} channels")


def tau_for_rate(c2c, rate):
    """The tau that makes exactly `rate` of these points outliers."""
    return float(np.quantile(c2c, 1.0 - rate))


def rank_features(X, y, feat, seed, objective, quantile, rng,
                  n_score=150_000, n_repeats=3, groups=None):
    """Rank features by permutation importance, using training rows only.

    Selection must not see the evaluation split. If features are chosen by
    looking at every scene and the result is then reported as leave-one-out,
    the held-out scene has informed the model and the number is optimistic.
    So the fit rows are split again: an inner train fits the ranking model, an
    inner score slice measures importance, and the evaluation rows are untouched.

    `groups` (voxel_id) makes that inner split *grouped* rather than random, and
    this matters more than it sounds. obs_count, view_diversity and
    temp_consistency are read off the 2 m ROOT voxel (voxel_map.hpp:1849), so
    they are constant inside it: a random split puts points from the same voxel
    on both sides, and the model can memorise "voxel 417 is bad" instead of
    learning anything transferable. Measured on one indoor scene, obs_count's
    importance falls 10.5x (30.3% -> 5.9% of the total, rank 1 -> rank 6) when
    the split is grouped. Those channels were being ranked first on leakage, and
    the resulting model then failed on the held-out scene.
    """
    if groups is None:
        idx = rng.permutation(len(y))
        cut = max(int(0.75 * len(idx)), 1)
        tr, sc = idx[:cut], idx[cut:cut + n_score]
    else:
        uniq = rng.permutation(np.unique(groups))
        cut = max(int(0.75 * len(uniq)), 1)
        in_train = np.isin(groups, uniq[:cut])
        tr = np.flatnonzero(in_train)
        sc = np.flatnonzero(~in_train)[:n_score]
        if len(sc) == 0 or len(tr) == 0:      # degenerate: fall back to random
            idx = rng.permutation(len(y))
            c = max(int(0.75 * len(idx)), 1)
            tr, sc = idx[:c], idx[c:c + n_score]
    # The inner ranking split is already group-aware; carry the same groups into
    # the fit so its early stopping is not decided on a leaky split either.
    model = fit_scorer(X[tr], y[tr], seed, objective, quantile,
                       groups=None if groups is None else np.asarray(groups)[tr])
    r = permutation_importance(model, X[sc], y[sc], n_repeats=n_repeats,
                               random_state=seed, n_jobs=1,
                               scoring="neg_mean_absolute_error")
    order = np.argsort(-r.importances_mean)
    return [(feat[i], float(r.importances_mean[i])) for i in order]


def grouped_importance(model, X, y, names, seed, n_repeats=3,
                       fold_aggregates=False, merge_ab=False):
    """Permutation importance over whole buckets rather than single features.

    With 44 mostly-redundant features a per-feature permutation understates
    everything: knock out one kNN scale and the others cover for it. Permuting a
    whole bucket with one shared permutation destroys that bucket's information
    while preserving the correlations inside it, which is the quantity the
    research question actually asks about.
    """
    rng = np.random.default_rng(seed)
    base = -mean_absolute_error(y, model.predict(X))
    out = {}
    for tag, cols in buckets(names, fold_aggregates, merge_ab).items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[np.ix_(perm, cols)]
            drops.append(base - (-mean_absolute_error(y, model.predict(Xp))))
        out[tag] = {"n_features": len(cols),
                    "importance": float(np.mean(drops)),
                    "std": float(np.std(drops))}
    return out


# Five families, in the order the write-up introduces them: A raw sensor,
# B within-scan, C SLAM-internal, D local aggregation, E purely geometric.
BUCKET_COLOUR = {"geom": "#1b9e77", "slam": "#d95f02", "sensor": "#e7298a",
                 "within": "#e6ab02", "agg": "#7570b3",
                 "sensor_within": "#c4562b"}
BUCKET_LABEL = {"geom": "E  geometry (post-hoc)", "slam": "C  SLAM-internal",
                "sensor": "A  raw sensor", "within": "B  within-scan",
                "agg": "D  local aggregation",
                "sensor_within": "A+B  raw sensor\n+ within-scan"}
BUCKET_LABEL_FOLDED = {"geom": "E  geometry (post-hoc)",
                       "slam": "C  SLAM-internal\n+ its aggregates",
                       "sensor": "A  raw sensor\n+ its aggregates",
                       "within": "B  within-scan\n+ its aggregates",
                       "agg": "D  aggregated",
                       "sensor_within": "A+B  raw sensor + within-scan"
                                        "\n+ their aggregates"}


def plot_importance(ranking, group_imp, out_dir, holdout, top_n=28,
                    top_bases=20):
    """Per-feature permutation importance, coloured by bucket, plus two rollups.

    Per-feature numbers understate everything when features are redundant --
    knock out one kNN scale and the others cover for it -- so the bucket panel
    is the one to read for "does this family matter". Both are shown because
    they answer different questions and disagreeing is informative.

    The middle panel rolls the aggregates back onto the channel they were
    derived from, so `plane_quality` is scored against every other channel
    rather than against its own six children. Without it a raw channel with six
    aggregates has its importance split seven ways and reads as weak purely
    because it was expanded. The split between the solid and hatched segment is
    the point of the panel: it says whether a channel matters as a raw value or
    only once neighbourhood context is attached to it.

    Caveat that belongs on the figure and in any caption: permutation
    importances are not additive. Summing them over redundant columns
    understates the group, which is exactly why the bucket panel permutes whole
    families at once. Read the middle panel as a lower bound and an ordering,
    not as a share of a total.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    shown = ranking[:top_n]
    names = [n for n, _ in shown][::-1]
    vals = [v for _, v in shown][::-1]
    cols = [BUCKET_COLOUR[bucket_of(n)] for n in names]

    # roll every aggregate back onto its parent channel
    raw_of, agg_of, nagg_of = {}, {}, {}
    for n, v in ranking:
        b = base_channel(n)
        if n == b:
            raw_of[b] = raw_of.get(b, 0.0) + v
        else:
            agg_of[b] = agg_of.get(b, 0.0) + v
            nagg_of[b] = nagg_of.get(b, 0) + 1
    bases = sorted(set(raw_of) | set(agg_of),
                   key=lambda b: raw_of.get(b, 0.0) + agg_of.get(b, 0.0))
    bases = bases[-top_bases:]

    has_groups = bool(group_imp)
    ncol = 3 if has_groups else 2
    fig, axes = plt.subplots(
        1, ncol,
        figsize=(19 if has_groups else 14,
                 max(6, 0.28 * max(len(names), len(bases)) + 2)),
        gridspec_kw={"width_ratios": [3, 2, 1][:ncol]},
        constrained_layout=True)
    ax = axes[0]

    ax.barh(range(len(names)), vals, color=cols)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8, family="monospace")
    ax.set_xlabel("permutation importance  (mean increase in MAE, m)")
    ax.set_title(f"Per-feature — top {len(names)} of {len(ranking)}", fontsize=11)
    ax.axvline(0.0, color="#444444", lw=0.8)
    ax.grid(axis="x", alpha=0.25)
    present = [b for b in ("sensor", "within", "slam", "agg", "geom")
               if any(bucket_of(n) == b for n in names)]
    ax.legend(handles=[Patch(facecolor=BUCKET_COLOUR[b], label=BUCKET_LABEL[b])
                       for b in present],
              loc="lower right", fontsize=9, framealpha=0.9)

    axb = axes[1]
    rawv = [raw_of.get(b, 0.0) for b in bases]
    aggv = [agg_of.get(b, 0.0) for b in bases]
    bcol = [BUCKET_COLOUR[bucket_of(b)] for b in bases]
    axb.barh(range(len(bases)), rawv, color=bcol)
    axb.barh(range(len(bases)), aggv, left=rawv, color=bcol, alpha=0.45,
             hatch="///", edgecolor="white", linewidth=0.4)
    axb.set_yticks(range(len(bases)))
    axb.set_yticklabels(
        [f"{b}" + (f"  (+{nagg_of[b]})" if nagg_of.get(b) else "")
         for b in bases], fontsize=8, family="monospace")
    axb.set_xlabel("summed permutation importance")
    axb.set_title("Per channel — aggregates rolled onto their parent\n"
                  "(+n = how many aggregate columns)", fontsize=11)
    axb.axvline(0.0, color="#444444", lw=0.8)
    axb.grid(axis="x", alpha=0.25)
    axb.legend(handles=[
        Patch(facecolor="#777777", label="raw channel"),
        Patch(facecolor="#777777", alpha=0.45, hatch="///",
              edgecolor="white", label="its aggregates")],
        loc="lower right", fontsize=9, framealpha=0.9)

    if has_groups:
        # a folded grouping has no standalone "agg" bucket left
        folded = "agg" not in group_imp
        g = sorted(group_imp.items(), key=lambda kv: kv[1]["importance"])
        ax2 = axes[2]
        ax2.barh(range(len(g)), [d["importance"] for _, d in g],
                 xerr=[d["std"] for _, d in g],
                 color=[BUCKET_COLOUR[t] for t, _ in g], capsize=3)
        ax2.set_yticks(range(len(g)))
        lbl = BUCKET_LABEL_FOLDED if folded else BUCKET_LABEL
        ax2.set_yticklabels([f"{lbl[t]}\n({d['n_features']} features)"
                             for t, d in g], fontsize=9)
        ax2.set_xlabel("grouped importance")
        ax2.set_title("Whole bucket permuted together"
                      + ("\naggregates folded onto their parent" if folded
                         else ""), fontsize=11)
        ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Feature importance — held out: {holdout or 'none (in-sample)'}"
                 "     (middle panel sums non-additive importances: "
                 "a lower bound, read the ordering)", fontsize=12)
    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_model(out_dir, tag, model, features, args, names, held, n_fit):
    """Persist one fitted scorer with everything needed to reuse it.

    The model alone is not usable. It was fitted on features that were
    rank-normalised *within each scene* and on a target divided by that scene's
    median C2C, so applying it to a new cloud means reproducing both steps. The
    normalisation is deliberately transductive — it needs the new scene's own
    feature distribution, not stored statistics — so what has to travel with the
    model is the recipe and the exact feature order, not fitted scalers.
    """
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        "model": model,
        "features": list(features),
        "recipe": {
            "normalise_per_scene": args.normalise_per_scene,
            "normalise_target": bool(args.normalise_target),
            "target": "C2C_distance / per-scene median" if args.normalise_target
                      else "C2C_distance",
            "objective": "q90" if tag.endswith("q90") else "mean",
            "quantile": args.quantile,
        },
        "training": {
            "scenes": [n for i, n in enumerate(names) if i not in (held or [])],
            "held_out": [names[i] for i in (held or [])],
            "n_fit_rows": int(n_fit),
            "tau": float(args.tau),
            "seed": args.seed,
        },
        "sklearn": __import__("sklearn").__version__,
    }
    path = os.path.join(out_dir, f"{tag}.joblib")
    joblib.dump(bundle, path, compress=3)
    return path


def sor_operating_points(knn, n_std=(1.0, 2.0, 3.0)):
    """Where the textbook SOR cut-offs land, as removal fractions."""
    mu, sd = float(knn.mean()), float(knn.std())
    out = []
    for k in n_std:
        thr = mu + k * sd
        out.append({"n_std": k, "threshold": thr,
                    "removed": float((knn > thr).mean())})
    return out


def write_ranking_summary(out_dir, rank_metrics, holdout, n_eval, inputs):
    """Human-readable ranking table beside the plot and the JSON.

    filter_comparison.json holds everything, which makes it awkward to read the
    one thing usually wanted: how each scorer ranks, at both thresholds. This
    writes that as a table.

    `lift` normalises AP by its own floor. AP cannot fall below the positive
    rate, so a scene with more outliers scores higher for free and raw AP is not
    comparable across scenes:

        lift = (AP - positive_rate) / (1 - positive_rate)

    0 is chance, 1 is perfect, and the number means the same thing on a scene
    with 9 % outliers and one with 64 %.
    """
    taus = sorted({t for m in rank_metrics.values()
                   for t in (m.get("by_tau") or {})})
    if not taus:
        return None
    path = os.path.join(out_dir, "ranking_summary.txt")
    with open(path, "w") as f:
        f.write(f"held out : {holdout or 'none (in-sample)'}\n")
        f.write(f"evaluated: {n_eval:,} points\n")
        for i in inputs:
            f.write(f"input    : {i}\n")
        f.write("\nlift = (AP - positive_rate) / (1 - positive_rate)"
                "   -- 0 is chance, 1 is perfect\n")
        f.write("rho  = Spearman correlation with the true C2C"
                "   -- threshold-free, +1 is a perfect ordering\n")

        rows = sorted(((m.get("spearman", float("nan")), name)
                       for name, m in rank_metrics.items()),
                      reverse=True)
        f.write("\n=== threshold-free ranking quality\n")
        f.write(f"{'scorer':32s} {'Spearman rho':>13s}\n")
        f.write("-" * 46 + "\n")
        for rho, name in rows:
            if np.isfinite(rho):
                f.write(f"{name:32s} {rho:13.4f}\n")
        for t in taus:
            any_m = next(iter(rank_metrics.values()))
            base = any_m["by_tau"][t]["positive_rate"]
            f.write(f"\n=== tau = {float(t) * 100:.0f} cm"
                    f"   outlier rate {base * 100:.2f} %\n")
            f.write(f"{'scorer':32s} {'AP':>9s} {'lift':>9s} {'ROC AUC':>9s}\n")
            f.write("-" * 62 + "\n")
            rows = []
            for name, m in rank_metrics.items():
                e = (m.get("by_tau") or {}).get(t)
                if not e or not np.isfinite(e["ap"]):
                    continue
                lift = (e["ap"] - base) / (1 - base) if base < 1 else float("nan")
                rows.append((lift, name, e["ap"], e["roc_auc"]))
            for lift, name, ap, auc in sorted(rows, reverse=True):
                f.write(f"{name:32s} {ap:9.4f} {lift:9.4f} {auc:9.4f}\n")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Compare SOR against SLAM-internal confidence channels, "
                    "with and without kNN in the mix.")
    ap.add_argument("input", nargs="+")
    ap.add_argument("--out", default="filter_comparison")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Outlier threshold in metres (default: 0.05)")
    ap.add_argument("--fractions", nargs="+", type=float, default=None,
                    help="Removal fractions to evaluate "
                         "(default: 0 to 0.50 in 26 steps)")
    ap.add_argument("--split-field", default=None)
    ap.add_argument("--group-field", default=None,
                    help="Field used to group rows for the inner ranking split "
                         "(normally 'cell_id', or 'voxel_id' on older clouds). Without it the split is random, "
                         "which lets voxel-constant channels — obs_count, "
                         "view_diversity, temp_consistency, all read off the 2 m "
                         "root voxel — memorise region identity and be ranked "
                         "first on leakage. Never a model input.")
    ap.add_argument("--no-sentinel-indicators", dest="sentinel_indicators",
                    action="store_false", default=True,
                    help="do not add <channel>__nofit columns for sentinel "
                         "channels. They are on by default: 'no plane could be "
                         "fitted' is a strong outlier signal, not missing data.")
    ap.add_argument("--keep-aggregates", action="store_true",
                    help="When excluding a raw channel, keep its __dev/__z "
                         "aggregates instead of dropping them with it.")
    ap.add_argument("--sor-from", default=None,
                    help="Channel to use as the SOR baseline even when it is "
                         "excluded from the models. The online ablation drops "
                         "every geometry channel -- family E needs the finished "
                         "map -- but SOR is the thing being compared against, "
                         "not a model input, so it still has to be computed.")
    ap.add_argument("--exclude-channels", default=None,
                    help="Comma-separated channels to drop from the feature set, "
                         "e.g. obs_count,view_diversity,temp_consistency.")
    ap.add_argument("--holdout", default=None,
                    help="Dataset or split value to hold out. Comma-separate "
                         "to hold out several at once (e.g. '3,4').")
    ap.add_argument("--max-per-dataset", type=int, default=20_000_000)
    ap.add_argument("--fit-n", type=int, default=500_000)
    ap.add_argument("--domain-field", default=DOMAIN_FIELD)
    ap.add_argument("--no-domain", action="store_true")
    ap.add_argument("--sor-channel", default=None,
                    help="Channel used as the plain SOR baseline (default: "
                         "mean_knn_dist, else the first mean_knn_dist_k*).")
    ap.add_argument("--normalise-per-scene", default="none",
                    choices=("none", "rank", "zscore"),
                    help="Rescale every feature within each scene before "
                         "training, so the model is not asked to transfer "
                         "absolute thresholds. SOR is unaffected.")
    ap.add_argument("--normalise-target", action="store_true",
                    help="Divide C2C by its per-scene median before fitting. "
                         "Without it a scene with 4x the noise dominates the "
                         "regression; the filter only needs a within-scene "
                         "ranking. Evaluation always uses raw C2C.")
    ap.add_argument("--tau-rate", type=float, default=None,
                    help="Set tau from the held-out scene's own C2C "
                         "distribution so this fraction are outliers, instead "
                         "of a fixed --tau. Makes folds comparable.")
    ap.add_argument("--single-baselines", nargs="*", default=None,
                    help="Conventional one-statistic filters to score beside "
                         "SOR: threshold a single channel, remove the worst "
                         "fraction. Default: every geometric channel, so a "
                         "plain roughness filter is compared on the same "
                         "footing as plain kNN. Pass with no names to disable.")
    ap.add_argument("--online-only", action="store_true",
                    help="keep only what exists while mapping: families A, B "
                         "and C. Drops D (aggregates over the finished cloud) "
                         "and E (geometry), which both need the assembled map, "
                         "so the result is what a filter could do in real time")
    ap.add_argument("--loo-families-agg", action="store_true",
                    help="also fit no_A_all / no_B_all / no_C_all: the family "
                         "removed together with its own k=30 aggregates")
    ap.add_argument("--only-arms", default=None,
                    help="comma list of arm names to fit (e.g. "
                         "slam_plus,no_C_all); all others are skipped")
    ap.add_argument("--loo-families", action="store_true",
                    help="also fit one arm per family with that family REMOVED "
                         "(no_A .. no_E), the complement of the *_only arms")
    ap.add_argument("--save-models", action="store_true",
                    help="Write every fitted scorer to <out>/models/ as a "
                         "joblib bundle carrying the feature order and the "
                         "preprocessing recipe needed to reuse it.")
    ap.add_argument("--geom-plus", nargs="+", type=int, default=None,
                    help="The direct test: geometry (all of it) plus the top-N "
                         "non-geometric features, for each N given. Compared "
                         "against geometry alone this isolates 'do the SLAM "
                         "channels add anything' from 'do 47 extra features "
                         "hurt a model fitted on three scenes'.")
    ap.add_argument("--ablate", nargs="+", type=int, default=None,
                    help="Feature-count sweep: refit using only the top-N "
                         "features for each N given, ranked inside the "
                         "training split. Shows how much the long tail of "
                         "weak channels is really worth.")
    ap.add_argument("--grouped-importance", action="store_true",
                    help="Permutation importance over whole buckets "
                         "(geometry / SLAM raw / SLAM aggregated).")
    ap.add_argument("--no-merge-ab", dest="merge_ab", action="store_false",
                    default=True,
                    help="keep A (raw sensor) and B (within-scan) as separate "
                         "buckets in the folded importance panel. They are "
                         "merged by default: both are per-scan quantities "
                         "available before any pose exists, so 'what is the "
                         "scan worth' is the question an online filter asks.")
    ap.add_argument("--plot-methods", default="key",
                    help="Which curves to draw: 'key' (default) shows one "
                         "objective per family and stays readable; 'all' draws "
                         "every fitted scorer, which on the current feature set "
                         "is 11+ overlapping lines; or a comma-separated list of "
                         "tags. Every scorer always reaches the JSON either way.")
    ap.add_argument("--early-stopping-split",
                    choices=("auto", "scene", "cell", "random"), default="auto",
                    help="what the early-stopping validation set is held out "
                         "by. auto = whole scenes when the fit set spans more "
                         "than one, else whole cells. random = sklearn's "
                         "internal split, which leaks (99.6%% of its validation "
                         "rows share a cell with a training row) and is kept "
                         "only so the change can be measured.")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # 0 to 50 % in 2 % steps. The old sweep stopped at 30 %, which is exactly
    # where the learned score first drags the retained mean C2C under 1 cm --
    # the baseline had not yet failed visibly at the right-hand edge, so the
    # figure cut off the comparison just as it became interesting.
    fractions = (np.asarray(args.fractions) if args.fractions is not None
                 else np.linspace(0.0, 0.50, 26))

    c2c, arrays, dataset, names, raw_groups = load_many(
        args.input, args.domain_field, args.no_domain, args.max_per_dataset,
        rng, split_field=args.split_field, group_field=args.group_field)
    # voxel_id is per file, so fold in the dataset index to make it globally
    # unique, and give every unassigned point (-1) a group of its own so it
    # behaves like a random row rather than joining one enormous pseudo-voxel.
    if args.group_field:
        span = int(raw_groups.max()) + 2
        groups = dataset.astype(np.int64) * span + raw_groups
        lone = raw_groups < 0
        if lone.any():
            groups[lone] = span * (int(dataset.max()) + 1) + np.arange(int(lone.sum()))
        print(f"  grouping rows by {args.group_field}: "
              f"{len(np.unique(groups)):,} groups "
              f"({int(lone.sum()):,} unassigned points isolated)")
    else:
        groups = None
    n = len(c2c)
    # Classical filters (ROR, DROR, SDOR), if a cloud carries them as
    # classical_* fields, are scores to compare against, never model inputs.
    # Taken out of `arrays` before anything else touches it: every model is
    # built from that dict, so a classical_* field left in it would silently
    # become a feature -- the same way c2c_rigid once did.
    classical_raw = {k: arrays.pop(k) for k in list(arrays)
                     if k.startswith("classical_")}
    if classical_raw:
        print(f"  classical filters, scored only: {', '.join(sorted(classical_raw))}")

    # Held aside before exclusion so the baseline survives an ablation that
    # removes every geometry channel from the models.
    sor_rescue = None
    if args.sor_from:
        if args.sor_from not in arrays:
            sys.exit(f"ERROR: --sor-from '{args.sor_from}' is not a channel")
        sor_rescue = arrays[args.sor_from].copy()
    if args.exclude_channels:
        drop = [c.strip() for c in args.exclude_channels.split(",") if c.strip()]
        gone = [c for c in drop if c in arrays]
        missing = [c for c in drop if c not in arrays]
        for c in gone:
            del arrays[c]
        # Aggregates are derived from the raw channel, so dropping the raw
        # channel without its aggregates normally leaves the same information in
        # place. --keep-aggregates suspends that, for the one question where the
        # distinction is the point: the leaf-constant SLAM channels are what
        # make a prediction blocky, while their k=30 aggregates are smooth by
        # construction, so "drop the blocky ones, keep the smooth ones" is a
        # real ablation rather than a half-done exclusion.
        derived = ([] if args.keep_aggregates
                   else [k for k in list(arrays) if k.split("__")[0] in gone])
        for c in derived:
            del arrays[c]
        print(f"  excluding {len(gone) + len(derived)} channels: "
              f"{', '.join(gone + derived)}")
        if missing:
            print(f"  NOTE: --exclude-channels named absent channels: "
                  f"{', '.join(missing)}")
    sentinels = detect_sentinels(arrays)
    # normalise_per_scene rewrites `arrays` in place, and a rank transform maps
    # -1 to rank 0 -- the marker is gone by the time the single-channel
    # baselines are built. Capture it here while it still exists.
    sentinel_masks = {name: (arrays[name] <= SENTINEL_TOL - 1.0)
                      for name in sentinels}
    # Blank the marker to NaN *here*, before normalisation. build_feature_matrix
    # also tries this, but it runs after normalise_per_scene, and a rank map
    # sends -1 to quantile 0 -- so that conversion never fired and the booster
    # was reading "no plane could be fitted" as the most planar value in the
    # cloud. normalise_per_scene skips non-finite entries, so NaN survives it
    # and reaches HistGradientBoostingRegressor, which learns a split direction
    # for missing values instead of being told one.
    for name, bad in sentinel_masks.items():
        if bad.any():
            arrays[name] = arrays[name].copy()
            arrays[name][bad] = np.nan
    n_sent = {k: int(v.sum()) for k, v in sentinel_masks.items() if v.any()}
    if n_sent:
        print("  sentinels blanked to NaN: "
              + ", ".join(f"{k} {v:,}" for k, v in sorted(n_sent.items())))
    print(f"\n  {n:,} points, channels: {', '.join(arrays)}")
    print(f"  sentinel channels: {', '.join(sorted(sentinels)) or 'none'}")
    if sor_rescue is not None and args.sor_from not in arrays:
        # Deliberately NOT put back into `arrays`: the models are built from
        # that dict, so restoring it there would hand the excluded channel
        # straight back to them. It lives beside the dict, used only for the
        # baseline score and the operating points.
        args.sor_channel = args.sor_from
        print(f"  SOR baseline kept from '{args.sor_from}' "
              f"(excluded from the models, used for comparison only)")
    if args.sor_channel is None:
        cands = [c for c in arrays if c.startswith("mean_knn_dist")]
        if not cands and sor_rescue is None:
            sys.exit("ERROR: no mean_knn_dist* channel — there is no SOR "
                     "baseline to compare against. Run the label pipeline with "
                     "the knn stage enabled, or add_geom_features.py.")
        # Prefer the classic single-scale channel when it is present.
        args.sor_channel = (KNN_CHANNEL if KNN_CHANNEL in cands
                            else sorted(cands)[0])
    if args.sor_channel not in arrays and sor_rescue is None:
        sys.exit(f"ERROR: --sor-channel '{args.sor_channel}' not in {list(arrays)}")

    # ---------------------------------------------------------- train / test
    fit_mask = np.ones(n, dtype=bool)
    held = []
    if args.holdout is not None:
        # Comma-separated, so a whole section made of several groups can be held
        # out at once -- the first building is gt_group 1 and 2, the second is 3 and 4, and
        # holding out only half of a section leaves the other half in training.
        held = []
        for want in [w.strip() for w in args.holdout.split(",") if w.strip()]:
            if want in names:
                held.append(names.index(want))
                continue
            hits = [i for i, nm in enumerate(names)
                    if nm.endswith(f"={want}") or nm.endswith(want)]
            if len(hits) != 1:
                sys.exit(f"ERROR: --holdout '{want}' matched {len(hits)} "
                         f"of {names}")
            held.append(hits[0])
        fit_mask = ~np.isin(dataset, held)
        print(f"  fitting on {int(fit_mask.sum()):,}, scoring on "
              f"{int((~fit_mask).sum()):,} held-out points "
              f"({', '.join(names[i] for i in held)})")
        eval_mask = ~fit_mask
    else:
        print("  NOTE: no --holdout given, so the learned scorers are evaluated "
              "on the data they were fitted on. Treat their curves as an upper "
              "bound; SOR is unaffected.")
        eval_mask = np.ones(n, dtype=bool)

    # ------------------------------------------ per-scene rescaling
    # Keep the raw SOR statistic: the textbook mu + k*sigma cut-offs are only
    # meaningful on the original scale (rank normalisation makes the
    # distribution uniform, where mu + 2*sigma means nothing). The SOR *curve*
    # is unaffected either way -- a within-scene monotone map cannot reorder it.
    # Drop exact duplicates before anything reads them: they add no information,
    # split a channel's importance across two bars, and would let the
    # feature-count ablation spend a slot on a copy. Done here so the drop
    # applies to fitting, evaluation, ranking and every plot alike.
    dropped = [k for k in list(arrays) if k in REDUNDANT_CHANNELS]
    for k in dropped:
        if k == args.sor_channel:      # never drop the SOR baseline itself
            continue
        del arrays[k]
    if dropped:
        print(f"  dropped {len(dropped)} redundant channel(s): "
              f"{', '.join(sorted(dropped))} "
              f"(bit-identical to intensity / mean_knn_dist_k6)")

    raw_sor = (sor_rescue.copy() if sor_rescue is not None
               and args.sor_channel not in arrays
               else arrays[args.sor_channel].copy())
    normalise_per_scene(arrays, dataset, len(names), args.normalise_per_scene)

    # "No plane could be fitted here" is evidence in its own right, not just an
    # absent measurement: on one indoor scene those points are 89.6 % outliers against an
    # 8.8 % background. NaN lets the booster learn a direction per split; an
    # explicit 0/1 column lets it use the fact directly, and lets the importance
    # attribution show how much it is worth. Added after normalisation so the
    # indicator stays 0/1 rather than being rank-mapped.
    if args.sentinel_indicators:
        for name, bad in sentinel_masks.items():
            if bad.any():
                arrays[f"{name}__nofit"] = bad.astype(np.float64)
        made = [f"{k}__nofit" for k, v in sentinel_masks.items() if v.any()]
        if made:
            print(f"  sentinel indicator channels: {', '.join(sorted(made))}")

    fit_target = c2c
    if args.normalise_target:
        fit_target = c2c.copy()
        for si in range(len(names)):
            m = dataset == si
            if m.any():
                med = float(np.median(c2c[m]))
                if med > 0:
                    fit_target[m] = c2c[m] / med
        print("  target normalised per scene (divided by its median C2C)")

    if args.tau_rate is not None:
        args.tau = tau_for_rate(c2c[eval_mask], args.tau_rate)
        print(f"  tau set from the evaluated scene: {args.tau * 100:.2f} cm "
              f"-> {args.tau_rate * 100:.1f} % outliers")

    fit_idx = np.flatnonzero(fit_mask)
    if args.fit_n and args.fit_n < len(fit_idx):
        fit_idx = rng.choice(fit_idx, args.fit_n, replace=False)

    stop_groups, stop_kind = stopping_groups(dataset, groups, fit_idx,
                                             mode=args.early_stopping_split)

    model_dir = os.path.join(args.out, "models")
    saved = []

    # ----------------------------------------------------------- the scorers
    # "slam_only" is families A + B + C: every non-geometric, non-aggregated
    # channel, sensor readings included. The three buckets are listed explicitly
    # because each has its own name; they are fitted as one set.
    slam_channels = {k: v for k, v in arrays.items()
                     if bucket_of(k) in ("slam", "sensor", "within")}
    agg_channels = {k: v for k, v in arrays.items() if bucket_of(k) == "agg"}
    scores = {}

    # SOR: the statistic itself is the score, higher = more isolated.
    # raw_sor, not arrays[...]: under --sor-from the channel is not in the
    # dict at all. A within-scene monotone map cannot reorder a ranking, so
    # using the un-normalised values leaves SOR's AP and curves unchanged.
    scores["sor"] = (raw_sor[eval_mask] if args.sor_channel not in arrays
                     else arrays[args.sor_channel][eval_mask])
    # higher = more suspicious for every classical_* field, by construction
    for name, col in classical_raw.items():
        scores[name] = col[eval_mask]

    # Conventional single-statistic filters. SOR is one of these -- a threshold
    # on mean_knn_dist -- and a roughness threshold is another, just as standard.
    # The honest conventional baseline is therefore the *best* of them, not kNN
    # alone, which is all this script compared against before.
    if args.single_baselines is None:
        singles = [k for k in arrays
                   if bucket_of(k) == "geom" and not k.endswith("__nofit")]
    else:
        singles = list(args.single_baselines)
    for name in singles:
        if name not in arrays:
            sys.exit(f"ERROR: --single-baselines '{name}' is not a channel")
        if name == args.sor_channel:
            continue
        # A sentinel means "no plane could be fitted here", which is not a
        # low roughness -- it is the strongest evidence the channel has. On
        # one indoor scene those points are 89.6 % outliers against an 8.8 % background.
        # Left at -1 they sort as the cleanest points in the cloud and the
        # baseline scores below chance, which understates it badly. The model
        # never sees this: build_feature_matrix turns sentinels into NaN and
        # the booster learns their direction.
        col = arrays[name][eval_mask].astype(np.float64, copy=True)
        bad = sentinel_masks.get(name)
        if bad is not None:
            bad = bad[eval_mask]
            if bad.any():
                finite = col[~bad & np.isfinite(col)]
                col[bad] = (finite.max() if len(finite) else 0.0) + 1.0
        scores[f"single_{name}"] = col
    extra = [n for n in singles if n != args.sor_channel]
    if extra:
        print(f"  single-statistic baselines: {', '.join(sorted(extra))}")
    scores["none"] = np.zeros(int(eval_mask.sum()))   # removes nothing useful

    geom_channels = {k: v for k, v in arrays.items() if bucket_of(k) == "geom"}
    print(f"\n  geometric bucket : {len(geom_channels):3d}  "
          f"{', '.join(sorted(geom_channels)) or '(none)'}")
    print(f"  SLAM raw bucket  : {len(slam_channels):3d}  "
          f"{', '.join(sorted(slam_channels)) or '(none)'}")
    print(f"  SLAM agg bucket  : {len(agg_channels):3d}  "
          f"{', '.join(sorted(agg_channels)[:4]) or '(none)'}"
          f"{' ...' if len(agg_channels) > 4 else ''}")
    print(f"    of which sensor: "
          f"{len([k for k in slam_channels if bucket_of(k) in ('sensor', 'within')]):3d}  "
          f"{', '.join(sorted(k for k in slam_channels if bucket_of(k) in ('sensor', 'within')))}")
    print(f"    of which SLAM  : "
          f"{len([k for k in slam_channels if bucket_of(k) == 'slam']):3d}  "
          f"{', '.join(sorted(k for k in slam_channels if bucket_of(k) == 'slam'))}")
    print(f"  SOR baseline     : {args.sor_channel}")

    # The raw bucket split into its two halves, in addition to `slam_only`
    # rather than instead of it: they answer whether it is the sensor readings
    # or the SLAM internals that carry the raw-channel signal.
    if args.online_only:
        keep = {"sensor", "within", "slam"}
        dropped = sorted(k for k in arrays if bucket_of(k) not in keep)
        for k in dropped:
            arrays.pop(k)
        geom_channels = {k: v for k, v in geom_channels.items() if k in arrays}
        agg_channels = {k: v for k, v in agg_channels.items() if k in arrays}
        slam_channels = {k: v for k, v in slam_channels.items() if k in arrays}
        print(f"  --online-only: dropped {len(dropped)} channels that need the "
              f"finished cloud, {len(arrays)} left")

    sensor_channels = {k: v for k, v in arrays.items()
                       if bucket_of(k) == "sensor"}
    slam_int_channels = {k: v for k, v in arrays.items()
                         if bucket_of(k) == "slam"}

    sets = [("slam_only", slam_channels), ("slam_plus", arrays)]
    if len(sensor_channels) > 1:
        sets.insert(0, ("sensor_only", sensor_channels))
    if len(slam_int_channels) > 1:
        sets.insert(0, ("slam_int_only", slam_int_channels))
    if agg_channels:
        # SLAM values over final-cloud topology: its own bucket, because it is
        # neither purely SLAM-internal nor conventional geometry.
        sets.insert(1, ("slam_agg", {**slam_channels, **agg_channels}))
    if len(geom_channels) > 1:
        # Only worth fitting when there is more than one geometric channel;
        # with a single one it would just relearn the SOR threshold.
        sets.insert(0, ("geom_only", geom_channels))

    if args.loo_families:
        # The complement of the "only" arms: everything EXCEPT one family. An
        # "only" arm says what a family carries on its own; this says what it
        # still adds once the others are present, which is the question worth
        # asking of a family that is expensive to produce (C needs the SLAM's
        # internals) or redundant with geometry.
        FAMILY = {"sensor": "A", "within": "B", "slam": "C",
                  "agg": "D", "geom": "E"}
        for bucket, letter in FAMILY.items():
            rest = {k: v for k, v in arrays.items() if bucket_of(k) != bucket}
            dropped = len(arrays) - len(rest)
            if dropped == 0 or len(rest) < 2:
                continue
            sets.append((f"no_{letter}", rest))

    if args.loo_families_agg:
        # As --loo-families, but A, B and C leave TOGETHER with their own k=30
        # aggregates. The aggregates (family D) are computed from A/B/C values,
        # so plain no_C still carries the SLAM internals' neighbourhood
        # summaries -- it cannot answer "does the method need the SLAM's
        # internals at all". Here a family goes with everything derived from it.
        def base_family(name):
            base = name.split("__")[0]
            return bucket_of(CHANNEL_ALIASES.get(base, base))
        for bucket, letter in (("sensor", "A"), ("within", "B"), ("slam", "C")):
            rest = {k: v for k, v in arrays.items()
                    if bucket_of(k) != bucket and base_family(k) != bucket}
            if len(rest) < len(arrays) and len(rest) >= 2:
                sets.append((f"no_{letter}_all", rest))

    if args.only_arms:
        # Fit just the arms asked for. The data loading and sampling are
        # deterministic, so arms fitted in separate runs on the same clouds and
        # holdout are directly comparable, and a missing arm need not cost a
        # full rerun of every other one.
        want = set(args.only_arms.split(","))
        sets = [(tag, ch) for tag, ch in sets if tag in want]
        print(f"  --only-arms: fitting {', '.join(tag for tag, _ in sets)}")

    for base_tag, chans in sets:
        X, feat = build_feature_matrix(chans, sentinels)
        print(f"\n  {base_tag}: {len(feat)} channels ({', '.join(feat)})")
        for obj in ("q90", "mean"):
            tag = f"{base_tag}_{obj}"
            model = fit_scorer(X[fit_idx], fit_target[fit_idx], args.seed, obj,
                               args.quantile, groups=stop_groups,
                               verbose=(base_tag == sets[0][0] and obj == "q90"))
            scores[tag] = model.predict(X[eval_mask])
            print(f"    {tag:16s} fitted in {model.n_iter_} iterations "
                  f"[{getattr(model, 'stopping_', '?')}]")
            if args.save_models:
                saved.append(save_model(model_dir, tag, model, feat, args,
                                        names, held if args.holdout else [],
                                        len(fit_idx)))
        del X

    # ------------------------------------------- ablation and attribution
    ablation, group_imp, ranking = None, None, None
    group_imp_split = group_imp_folded = None
    if args.ablate or args.grouped_importance or args.geom_plus:
        X, feat = build_feature_matrix(arrays, sentinels)
        Xf, yf = X[fit_idx], fit_target[fit_idx]
        ranking = rank_features(Xf, yf, feat, args.seed, "q90",
                                args.quantile, rng,
                                groups=None if groups is None else groups[fit_idx])
        print(f"\n  feature ranking (top 12 of {len(feat)}, "
              f"from the training split only):")
        for name, imp in ranking[:12]:
            print(f"    {name:<34s} {bucket_of(name):>5s}  {imp: .5f}")

        if args.ablate:
            ablation = []
            order = [n for n, _ in ranking]
            for n_feat in sorted(set(args.ablate)):
                if n_feat > len(order):
                    continue
                keep = order[:n_feat]
                cols = [feat.index(k) for k in keep]
                model = fit_scorer(Xf[:, cols], yf, args.seed, "q90",
                                   args.quantile, groups=stop_groups)
                tag = f"top{n_feat}_q90"
                scores[tag] = model.predict(X[eval_mask][:, cols])
                if args.save_models:
                    saved.append(save_model(model_dir, tag, model, keep, args,
                                            names, held if args.holdout else [],
                                            len(fit_idx)))
                by_bucket = {}
                for k in keep:
                    by_bucket[bucket_of(k)] = by_bucket.get(bucket_of(k), 0) + 1
                ablation.append({"n_features": n_feat, "tag": tag,
                                 "features": keep, "by_bucket": by_bucket})
                print(f"    top{n_feat:<3d} fitted  "
                      + "  ".join(f"{b}:{c}" for b, c in sorted(by_bucket.items())))

        if args.geom_plus:
            geom_names = [f for f in feat if bucket_of(f) == "geom"]
            other = [n for n, _ in ranking if bucket_of(n) != "geom"]
            for n_add in sorted(set(args.geom_plus)):
                keep = geom_names + other[:n_add]
                cols = [feat.index(k) for k in keep]
                model = fit_scorer(Xf[:, cols], yf, args.seed, "q90",
                                   args.quantile, groups=stop_groups)
                tag = f"geomplus{n_add}_q90"
                scores[tag] = model.predict(X[eval_mask][:, cols])
                if args.save_models:
                    saved.append(save_model(model_dir, tag, model, keep, args,
                                            names, held if args.holdout else [],
                                            len(fit_idx)))
                print(f"    geometry({len(geom_names)}) + top{n_add} "
                      f"non-geometric -> {len(keep)} features: "
                      + ", ".join(other[:n_add][:4])
                      + (" ..." if n_add > 4 else ""))

        if args.grouped_importance:
            model = fit_scorer(Xf, yf, args.seed, "q90", args.quantile,
                               groups=stop_groups)
            sub = rng.choice(len(fit_idx), min(150_000, len(fit_idx)),
                             replace=False)
            group_imp_split = grouped_importance(model, Xf[sub], yf[sub],
                                                 feat, args.seed)
            group_imp_folded = grouped_importance(model, Xf[sub], yf[sub],
                                                  feat, args.seed,
                                                  fold_aggregates=True)
            for lab, gi in (("4-way (aggregates their own bucket)",
                             group_imp_split),
                            ("3-way (aggregates folded onto their parent)",
                             group_imp_folded)):
                print(f"\n  grouped permutation importance -- {lab}:")
                for tag, d in sorted(gi.items(),
                                     key=lambda kv: -kv[1]["importance"]):
                    print(f"    {tag:<6s} {d['n_features']:3d} features   "
                          f"{d['importance']: .5f} +/- {d['std']:.5f}")
            # the folded view is what the third panel shows; both reach the JSON
            group_imp = group_imp_folded
        del X, Xf

    eval_c2c = c2c[eval_mask]
    sor_points = sor_operating_points(raw_sor[eval_mask])

    # -------------------------------------------------------------- evaluate
    results = {}
    rank_metrics = {}
    for tag, score in scores.items():
        if tag == "none":
            # A single row: nothing removed.
            results[tag] = evaluate(score, eval_c2c, [0.0], args.tau)
        else:
            results[tag] = evaluate(score, eval_c2c, fractions, args.tau)
            rank_metrics[tag] = ranking_metrics(score, eval_c2c, args.tau)

    # The best any filter could possibly do at each budget: rank by the TRUE
    # C2C and drop the worst points first. This is what makes the removal-budget
    # figure honest. A curve that reaches 1 cm at 30 % removal says nothing on
    # its own about whether the right 30 % went -- a method could get there by
    # removing good points that happen to sit near bad ones. The vertical gap to
    # the oracle is exactly the waste: how much of the budget was spent on points
    # that did not need removing. It is not attainable (it needs the ground
    # truth) and it is not a competitor; it is the axis the others are read
    # against, which is the role AP's precision term played in the table.
    results["oracle"] = evaluate(eval_c2c, eval_c2c, fractions, args.tau)

    baseline = results["none"][0]
    print(f"\n  unfiltered: mean C2C {baseline['mean_c2c'] * 100:.3f} cm, "
          f"outlier rate {baseline['outlier_rate'] * 100:.2f} % "
          f"(tau = {args.tau * 100:.0f} cm)")
    # Excluding a whole family leaves some model tags unfitted -- no SLAM
    # channels means no slam_int_only model at all -- so the table takes the
    # tags that exist rather than the full list.
    cols = [t for t in ["sor"] + LEARNED + ["oracle"] if t in results]
    for key, label, scale, prec in (
            ("mean_c2c", "Mean C2C [cm] of the retained points", 100.0, 4),
            ("outlier_rate", f"Remaining outliers [%] (C2C > {args.tau * 100:.0f} cm)",
             100.0, 4),
            ("f1", "Inlier F1", 1.0, 4)):
        print(f"\n  {label}, at matched removal:")
        print("    removed" + "".join(f"{style_of(t)[2][:19]:>21s}" for t in cols))
        for i, f in enumerate(fractions):
            # width and precision have to be separate: an f-string spec of
            # ">18" + "9.4f" parses as width 189, not width 18.
            cells = "".join(f"{results[t][i][key] * scale:>21.{prec}f}"
                            for t in cols)
            print(f"    {f * 100:6.1f}%{cells}")

    print("\n  Textbook SOR operating points on this data:")
    for op in sor_points:
        print(f"    mean + {op['n_std']:.0f}*std = {op['threshold'] * 100:7.3f} cm "
              f"-> removes {op['removed'] * 100:5.2f} %")

    # ------------------------------------------------------------- headline
    def at(tag, frac):
        return min(results[tag], key=lambda r: abs(r["removed"] - frac))

    print("\n  Headline: every learned scorer against SOR, as % change "
          "(negative = better than SOR)")
    for frac in (0.05, 0.10, 0.20, 0.30):
        base = at("sor", frac)
        print(f"    at ~{frac * 100:.0f} % removed  "
              f"(SOR: mean {base['mean_c2c'] * 100:.3f} cm, "
              f"outliers {base['outlier_rate'] * 100:.3f} %)")
        # cols[1:] rather than LEARNED: an ablation that drops a whole family
        # never trains the variants built on it (no geometry -> no geom_only),
        # and this table is the last thing before the JSON is written, so a
        # KeyError here throws away the entire run.
        for tag in cols[1:]:
            r = at(tag, frac)
            dm = 100 * (r["mean_c2c"] / base["mean_c2c"] - 1)
            do = (100 * (r["outlier_rate"] / base["outlier_rate"] - 1)
                  if base["outlier_rate"] > 0 else float("nan"))
            print(f"      {style_of(tag)[2]:24s} mean {dm:+6.1f} %   "
                  f"outliers {do:+6.1f} %")

    # ---------------------------------------------------------------- plots
    # With five model families x two objectives plus the ablation sweeps, drawing
    # everything puts 20+ lines on one axis and the figure stops being readable.
    # Default to one curve per family (the `mean` objective, which has beaten q90
    # in every run) plus the two baselines. Nothing is lost: the JSON keeps all.
    KEY = ["sor", "geom_only_mean", "sensor_only_mean", "slam_int_only_mean",
           "slam_only_mean", "slam_agg_mean", "slam_plus_mean", "slam_plus_q90",
           "oracle"]
    if args.plot_methods == "all":
        plot_tags = list(cols)
    elif args.plot_methods == "key":
        plot_tags = [t for t in KEY if t in cols]
        if len(plot_tags) < 3:                    # nothing matched: draw it all
            plot_tags = list(cols)
    else:
        want = [t.strip() for t in args.plot_methods.split(",") if t.strip()]
        plot_tags = [t for t in want if t in cols]
        missing = [t for t in want if t not in cols]
        if missing:
            print(f"  --plot-methods: no such scorer {missing}, ignored")
    print(f"  plotting {len(plot_tags)} of {len(cols)} scorers "
          f"({args.plot_methods}); all are in the JSON")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    panels = [("mean_c2c", "mean C2C of retained points [cm]", 100.0),
              ("outlier_rate", f"fraction with C2C > {args.tau * 100:.0f} cm [%]", 100.0),
              ("f1", "inlier F1", 1.0)]
    for ax, (key, ylabel, scale) in zip(axes, panels):
        for tag in plot_tags:
            colour, style, label = style_of(tag)
            xs = [r["removed"] * 100 for r in results[tag]]
            ys = [r[key] * scale for r in results[tag]]
            ax.plot(xs, ys, style, color=colour, linewidth=2, label=label,
                    marker="o", markersize=3)
        ax.axhline(baseline[key] * scale, color="#666666", linestyle=":",
                   linewidth=1.5, label="no filter")
        for op in sor_points:
            ax.axvline(op["removed"] * 100, color="#d95f02", alpha=0.25,
                       linewidth=1, linestyle="--")
        ax.set_xlabel("points removed [%]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    # legend under the figure: inside the axes it covered the curves it explains
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    title = "Filter comparison at matched removal budget"
    if args.holdout:
        title += f"  —  held out: {args.holdout}"
    else:
        title += "  —  IN-SAMPLE (no holdout), learned curves are optimistic"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    path = os.path.join(args.out, "filter_comparison.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\n  wrote {path}")

    report = {
        "inputs": args.input,
        "tau": args.tau,
        "holdout": args.holdout,
        "split_field": args.split_field,
        "in_sample": args.holdout is None,
        "n_eval": int(eval_mask.sum()),
        "channels": list(arrays),
        "sentinel_channels": sorted(sentinels),
        # What the early-stopping validation set was held out by. "random" is
        # sklearn's internal split and is leaky; see fit_scorer.
        "early_stopping_split": stop_kind,
        "unfiltered": baseline,
        "sor_operating_points": sor_points,
        # Computed inside the fold; persisted so the figures can be redrawn and
        # the numbers quoted without re-running the fit.
        "ranking": [{"feature": n, "bucket": bucket_of(n), "importance": v}
                    for n, v in (ranking or [])],
        "grouped_importance": group_imp,
        "grouped_importance_split": group_imp_split,
        "grouped_importance_folded": group_imp_folded,
        "ablation": ablation,
        "curves": results,
        "ranking_metrics": rank_metrics,
    }
    jpath = os.path.join(args.out, "filter_comparison.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    spath = write_ranking_summary(args.out, rank_metrics, args.holdout,
                                  int(len(eval_c2c)), args.input)
    if spath:
        print(f"  wrote {spath}")
    if ranking is not None:
        p = plot_importance(ranking, group_imp, args.out, args.holdout)
        print(f"  wrote {p}")
    if args.save_models and saved:
        with open(os.path.join(model_dir, "manifest.json"), "w") as f:
            json.dump({"models": [os.path.basename(x) for x in saved],
                       "holdout": args.holdout,
                       "note": "each .joblib holds {model, features, recipe, "
                               "training}. Reuse needs the same per-scene "
                               "normalisation applied to the new cloud."},
                      f, indent=2)
        print(f"Saved {len(saved)} models -> {model_dir}")
    print(f"  wrote {jpath}")


if __name__ == "__main__":
    main()
