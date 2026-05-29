"""Integration test: aggregate() must discriminate the three correction regimes.

This is the scrutiny test for the go/no-go decision logic. We synthesize
PositionRecords with a KNOWN structure for the correction c_v = log F(prefix, v)
and check that the metric:

  R1 (affine in logit)      -> beyond-temperature increment ~ 0   (temperature suffices)
  R2 (nonlinear in logit)   -> beyond-temperature increment > 0   (1-D twist helps)
  R3 (identity-dependent)   -> beyond-temperature increment ~ 0   (no 1-D twist can help),
                               while D_total stays large (correction is real but unreachable)

These are exactly the cases the go/no-go must tell apart: R2 is GO, R1/R3 are not.
The filter latent theta is set to the centered logit (the affine case b=1), so the
filter-CV and logit-CV ceilings should agree here.
"""
from __future__ import annotations

import numpy as np

from decoding_decoding.full_sequence.analyze import (
    PositionRecord,
    aggregate,
    go_no_go_decision,
)


def make_record(**kw) -> PositionRecord:
    return PositionRecord(**kw)


def _build(regime: str, *, n_pos: int, K: int, T: float, seed: int):
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_pos):
        ell = np.sort(rng.normal(0, 2.0, size=K))[::-1].copy()  # candidate logits
        log_p = ell - np.log(np.exp(ell).sum())                 # within-candidate logprob
        E_q = float(np.mean(ell))                               # proxy full-vocab mean
        s = ell - E_q
        if regime == "affine":
            c = 0.8 * s
        elif regime == "nonlinear":
            # Inverted-U in the logit: boosts mid-probability tokens, suppresses BOTH
            # the head and the tail. q_g then peaks at an INTERIOR logit -> no monotone
            # temperature exp(beta*ell) can represent it, but a free function of the
            # 1-D latent can. This is the genuine "beyond-temperature" regime.
            c = -0.8 * s ** 2
        elif regime == "identity":
            c = rng.normal(0, 1.2, size=K)        # depends on token identity, not logit
        else:
            raise ValueError(regime)
        theta = s.copy()  # pretend the filter latent IS the centered logit (b=1)
        records.append(
            make_record(
                corpus="syn", passage_id=p, position=24, candidates=list(range(K)),
                ell=ell.tolist(), log_p=log_p.tolist(), E_q=E_q,
                covered_mass={str(T): 1.0}, c_by_T={str(T): c.tolist()},
                theta=theta.tolist(),
            )
        )
    return records


def _agg(records, T, seed=0):
    return aggregate(records, [T], seed=seed)[str(T)]


def test_affine_regime_no_beyond_temperature_signal():
    recs = _build("affine", n_pos=80, K=24, T=0.6, seed=1)
    m = _agg(recs, 0.6)
    # Temperature alone recovers almost everything; CV ceilings add ~nothing.
    assert m["frac_bestT"] > 0.9
    lo, pt, hi = m["frac_filter_cv_minus_bestT"]
    assert pt < 0.05          # below the GO bar
    assert m["affine_residual_fraction_mean"] < 0.05


def test_nonlinear_regime_has_beyond_temperature_signal():
    recs = _build("nonlinear", n_pos=120, K=24, T=0.6, seed=2)
    m = _agg(recs, 0.6)
    assert m["D_total_ci"][0] > 1e-3
    flo, fpt, fhi = m["frac_filter_cv_minus_bestT"]
    # The CV map of the 1-D latent recovers a real beyond-temperature fraction.
    assert flo > 0.0
    assert fpt >= 0.05


def test_identity_regime_real_correction_but_unreachable_by_1d():
    recs = _build("identity", n_pos=120, K=24, T=0.6, seed=3)
    m = _agg(recs, 0.6)
    # The correction is real ...
    assert m["D_total_ci"][0] > 1e-3
    # ... but no function of the 1-D latent can recover it beyond temperature.
    _flo, fpt, _fhi = m["frac_filter_cv_minus_bestT"]
    _llo, lpt, _lhi = m["frac_logit_cv_minus_bestT"]
    assert fpt < 0.05
    assert lpt < 0.05


def test_linear_filter_twist_is_just_temperature():
    # The scalar-LINEAR filter twist must not beat best-T (it IS a temperature change).
    recs = _build("nonlinear", n_pos=100, K=24, T=0.6, seed=4)
    m = _agg(recs, 0.6)
    _lo, pt, _hi = m["frac_filter_linear_minus_bestT"]
    assert pt < 0.05
