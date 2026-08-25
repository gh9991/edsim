import numpy as np
import pandas as pd

from edsim import dataqc, schema


def _frame(n=4000, seed=0, **override):
    rng = np.random.default_rng(seed)
    ats = rng.choice([1, 2, 3, 4, 5], n, p=[0.01, 0.18, 0.42, 0.34, 0.05])
    arrive = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 365 * 24 * 60, n)), unit="m")
    wait = pd.Series({1: 0.5, 2: 8, 3: 40, 4: 70, 5: 90}).reindex(ats).to_numpy() \
        * rng.lognormal(0, 0.3, n)
    admit_p = pd.Series({1: .75, 2: .55, 3: .30, 4: .10, 5: .03}).reindex(ats).to_numpy()
    df = pd.DataFrame({
        "encounter_id": [f"E{i}" for i in range(n)],
        "arrival_ts": arrive,
        "seen_ts": arrive + pd.to_timedelta(wait, unit="m"),
        "depart_ts": arrive + pd.to_timedelta(wait + rng.uniform(30, 300, n), unit="m"),
        "acuity": ats, "acuity_scale": "ATS",
        "disposition": np.where(rng.random(n) < admit_p, "admitted", "discharged"),
    })
    for k, v in override.items():
        df[k] = v
    return schema.validate(df)


def _named(df, name):
    return next(f for f in dataqc.health_check(df) if f.check == name)


def test_a_plausible_extract_passes_everything():
    findings = dataqc.health_check(_frame())
    assert all(f.passed for f in findings), [str(f) for f in findings if not f.passed]


def test_catches_wait_time_independent_of_triage():
    """The artefact found in a 2025 participant's generated data: every ATS
    category has the same median wait."""
    df = _frame()
    flat = pd.to_timedelta(np.full(len(df), 32.0), unit="m")
    df["seen_ts"] = df["arrival_ts"] + flat
    df = schema.validate(df)
    assert not _named(df, "wait varies with acuity").passed


def test_catches_deterministic_admission():
    df = _frame()
    df["disposition"] = np.where(df["ats"] <= 2, "admitted", "discharged")
    df = schema.validate(df)
    assert not _named(df, "admission is probabilistic").passed


def test_catches_date_shifting():
    df = _frame(n=1000)
    df["arrival_ts"] = pd.date_range("2100-01-01", periods=len(df), freq="36D")
    df = schema.validate(df)
    f = _named(df, "dates not shifted")
    assert not f.passed and "span" in f.detail


def test_catches_out_of_order_timestamps():
    df = _frame()
    df.loc[df.index[:20], "seen_ts"] = df["arrival_ts"].iloc[:20] - pd.Timedelta(minutes=5)
    df = schema.validate(df)
    assert not _named(df, "timestamps ordered").passed


def test_catches_implausible_resus_wait():
    df = _frame()
    m = df["ats"] == 1
    df.loc[m, "seen_ts"] = df.loc[m, "arrival_ts"] + pd.Timedelta(minutes=32)
    df = schema.validate(df)
    assert not _named(df, "ATS 1 seen immediately").passed


def test_checks_are_skippable_not_crashable():
    """A source missing a column must degrade, never blow up the run."""
    df = _frame().drop(columns=["seen_ts"])
    findings = dataqc.health_check(df)
    assert len(findings) == len(dataqc.CHECKS)
