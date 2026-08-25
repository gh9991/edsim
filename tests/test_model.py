import dataclasses

import numpy as np
import pandas as pd
import pytest

from edsim import features, model
from edsim.calibrate import SimParams, size_capacity
from edsim.sim import simulate


def _sim(days=150, seed=11):
    return simulate(size_capacity(SimParams(daily_arrivals=180, source="test")),
                    days=days, seed=seed)


def test_outcome_columns_are_rejected_as_features():
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(["ats", "age", "los_min"])


def test_built_features_contain_no_outcomes():
    X = features.build(_sim(days=20))
    assert not set(X.columns) & set(features.LEAKY)
    assert "ats" in X.columns


def test_split_is_chronological_not_random():
    df = _sim(days=40)
    tr, ca, te = model.temporal_split(df)
    assert tr["arrival_ts"].max() <= ca["arrival_ts"].min()
    assert ca["arrival_ts"].max() <= te["arrival_ts"].min()


def _ceiling_auc(p: SimParams) -> float:
    """Best achievable AUC when admission depends only on ATS."""
    ats = sorted(p.acuity_mix)
    pos = np.array([p.acuity_mix[a] * p.admit_prob[a] for a in ats])
    neg = np.array([p.acuity_mix[a] * (1 - p.admit_prob[a]) for a in ats])
    pos, neg = pos / pos.sum(), neg / neg.sum()
    auc = 0.0
    for i, a in enumerate(ats):
        for j, b in enumerate(ats):
            if p.admit_prob[a] > p.admit_prob[b]:
                auc += pos[i] * neg[j]
            elif p.admit_prob[a] == p.admit_prob[b]:
                auc += 0.5 * pos[i] * neg[j]
    return float(auc)


def test_model_reaches_the_information_ceiling_on_simulated_data():
    """In the simulator, admission depends ONLY on ATS - so there is a hard
    upper bound on AUC. Landing near it says the pipeline extracts the real
    signal; landing above it would mean something leaked."""
    p = size_capacity(SimParams(daily_arrivals=180, source="test"))
    m, test = model.train(simulate(p, days=150, seed=11))
    auc = model.evaluate(m, test)["auc"]
    ceiling = _ceiling_auc(p)
    assert auc <= ceiling + 0.02, f"AUC {auc} above ceiling {ceiling} - leakage?"
    assert auc >= ceiling - 0.06, f"AUC {auc} far below ceiling {ceiling}"


def test_predictions_are_calibrated_enough_to_forecast_beds():
    m, test = model.train(_sim())
    ev = model.evaluate(m, test)
    assert abs(ev["calibration_bias"]) < 0.05
    # daily bed forecast should be within ~25% of actual demand
    assert ev["daily_mae_beds"] < 0.25 * ev["mean_actual_admissions_per_day"]


def test_calibration_table_tracks_the_diagonal():
    m, test = model.train(_sim())
    t = model.calibration_table(m, test)
    assert t["gap"].abs().max() < 0.15


def test_ats_dominates_importance_because_the_simulator_says_so():
    m, test = model.train(_sim())
    imp = model.importances(m, test, n_repeats=5)
    assert imp.iloc[0]["feature"] == "ats"


def test_date_shifted_data_gets_a_forecast_warning():
    df = _sim(days=40)
    shifted = df.copy()
    # one encounter per calendar day, as MIMIC's per-patient shifting produces
    shifted["arrival_ts"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    m, test = model.train(df)
    out = model.daily_forecast_error(m, shifted.iloc[-len(test):])
    assert "daily_forecast_warning" in out


def test_patient_explanation_names_the_driver():
    m, test = model.train(_sim())
    e = model.explain_patient(m, test, i=0)
    assert "contribution" in e.columns
    assert e["contribution"].abs().iloc[0] > 0
