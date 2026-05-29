"""Tests for the cross-validated nonparametric 1-D twist-map ceiling.

``cv_binned_predict`` fits a GLOBAL map theta -> (per-position de-meaned) c by
quantile-binned means, with k-fold cross-validation OVER POSITIONS (so a
position's own candidates never leak into its own prediction). This is the
strongest honest ceiling for "best smooth function of the 1-D latent": if even
this cannot beat a global temperature retune, no log-linear/scalar twist over
this latent can.
"""
from __future__ import annotations

import numpy as np

from decoding_decoding.full_sequence.metrics import cv_binned_predict


def test_recovers_deterministic_monotonic_function_of_theta():
    rng = np.random.default_rng(0)
    thetas, cs = [], []
    for _ in range(40):
        th = np.sort(rng.uniform(-2, 2, size=12))
        level = rng.normal(0, 3.0)          # per-position gauge (must cancel)
        c = 1.5 * th + level
        thetas.append(th)
        cs.append(c)
    preds = cv_binned_predict(thetas, cs, n_bins=8, n_folds=5, seed=0)
    true_dm = np.concatenate([c - c.mean() for c in cs])
    pred_dm = np.concatenate(preds)
    assert np.corrcoef(true_dm, pred_dm)[0, 1] > 0.9


def test_predicts_near_zero_when_c_independent_of_theta():
    rng = np.random.default_rng(1)
    thetas, cs = [], []
    for _ in range(60):
        thetas.append(rng.uniform(-2, 2, size=12))
        cs.append(rng.normal(0, 1.0, size=12))   # independent of theta
    preds = cv_binned_predict(thetas, cs, n_bins=8, n_folds=5, seed=0)
    true_dm = np.concatenate([c - c.mean() for c in cs])
    pred_dm = np.concatenate(preds)
    assert np.var(pred_dm) < 0.25 * np.var(true_dm)
    assert abs(np.corrcoef(true_dm, pred_dm)[0, 1]) < 0.3


def test_output_shapes_match_inputs():
    rng = np.random.default_rng(2)
    thetas = [rng.uniform(-1, 1, size=k) for k in (5, 8, 3, 10)]
    cs = [rng.normal(size=k) for k in (5, 8, 3, 10)]
    preds = cv_binned_predict(thetas, cs, n_bins=4, n_folds=2, seed=0)
    assert len(preds) == len(thetas)
    for pr, th in zip(preds, thetas):
        assert pr.shape == th.shape


def test_predictions_are_per_position_demeaned():
    rng = np.random.default_rng(3)
    thetas, cs = [], []
    for _ in range(30):
        th = np.sort(rng.uniform(-2, 2, 10))
        cs.append(2.0 * th + rng.normal(0, 5.0))   # big per-position offsets
        thetas.append(th)
    preds = cv_binned_predict(thetas, cs, n_bins=6, n_folds=5, seed=0)
    for pr in preds:
        assert abs(float(pr.mean())) < 1.0
