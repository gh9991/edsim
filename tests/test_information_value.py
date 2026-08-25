"""Tests run on simulated data - MIMIC cannot be committed."""
import numpy as np
import pandas as pd

from edsim import information_value as iv
from edsim.calibrate import SimParams, size_capacity
from edsim.sim import simulate


def _sim(days=120, seed=3):
    df = simulate(size_capacity(SimParams(daily_arrivals=200, source="test")),
                  days=days, seed=seed)
    # give patients repeat visits so the group split has something to separate
    df["patient_id"] = "P" + (np.arange(len(df)) // 2).astype(str)
    df["presenting_complaint"] = np.where(df["ats"] <= 2, "chest pain", "sore throat")
    return df


def test_split_never_puts_a_patient_in_both_train_and_test():
    df = _sim()
    tr, ca, te = iv._split_by_patient(df, seed=7)
    pids = df["patient_id"].to_numpy()
    assert not (set(pids[tr]) & set(pids[te]))
    assert not (set(pids[tr]) & set(pids[ca]))
    assert len(tr) + len(ca) + len(te) == len(df)


def test_sharper_probabilities_have_a_lower_floor():
    """The mechanical claim behind the headline result: a model that pushes
    probabilities towards 0 and 1 lowers its own irreducible cohort error,
    without forecasting any better."""
    flat = np.full(10_000, 0.4)
    sharp = np.where(np.arange(10_000) % 2 == 0, 0.05, 0.75)
    assert abs(flat.mean() - sharp.mean()) < 0.01      # same expected count
    assert iv.cohort_floor_pct(sharp) < iv.cohort_floor_pct(flat)


def test_floor_matches_a_monte_carlo_draw():
    rng = np.random.default_rng(0)
    p = np.full(5_000, 0.372)
    size = 150
    errs = []
    for _ in range(4_000):
        s = rng.choice(len(p), size=size, replace=False)
        y = rng.binomial(1, p[s])
        if y.sum():
            errs.append(abs(p[s].sum() - y.sum()) / y.sum() * 100)
    assert abs(np.mean(errs) - iv.cohort_floor_pct(p, size)) < 1.0


def test_run_returns_three_arms_with_gains_measured_against_A():
    res = iv.run(_sim(days=60), verbose=False)
    assert len(res) == 3
    assert res.loc[0, "auc_gain_vs_A"] == 0
    assert (res["cohort_mae_pct"] > 0).all()
    for c in ("auc", "brier", "cohort_floor_pct", "excess_over_floor_pct"):
        assert c in res.columns


def test_no_gain_when_the_extra_columns_carry_no_signal():
    """The simulator has no vitals at all, so arms B and C cannot beat A by
    much. A large gain here would mean the harness is inventing signal."""
    res = iv.run(_sim(days=90), verbose=False)
    assert res.loc[1, "auc_gain_vs_A"] < 0.05
