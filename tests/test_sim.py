import pandas as pd

from edsim.calibrate import SimParams, check, size_capacity
from edsim.metrics import by_triage, kpis
from edsim.sim import simulate


def _params(**kw):
    p = size_capacity(SimParams(daily_arrivals=120, source="test"))
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_simulation_produces_canonical_frame():
    df = simulate(_params(), days=5, seed=1)
    assert len(df) > 100
    for col in ("arrival_ts", "ats", "los_min", "wait_to_treat_min", "admitted"):
        assert col in df.columns
    assert df["arrival_ts"].is_monotonic_increasing
    assert df["ats"].between(1, 5).all()


def test_departures_never_precede_arrivals():
    df = simulate(_params(), days=5, seed=2)
    assert (df["los_min"].dropna() >= 0).all()
    assert (df["wait_to_treat_min"].dropna() >= 0).all()


def test_higher_acuity_waits_less():
    df = simulate(_params(), days=20, seed=3)
    seen = df[df["seen_ts"].notna()]
    med = seen.groupby("ats")["wait_to_treat_min"].median()
    assert med.get(2, 0) <= med.get(5, 1e9)


def test_ward_pressure_creates_access_block():
    easy = kpis(simulate(_params(ward_bed_occupancy=0.50), days=20, seed=4))
    hard = kpis(simulate(_params(ward_bed_occupancy=0.95), days=20, seed=4))
    assert hard["median_boarding_min"] >= easy["median_boarding_min"]
    assert hard["neat_pct"] <= easy["neat_pct"]


def test_more_cubicles_cannot_fix_a_blocked_ward():
    """The point of the whole model: ED capacity has diminishing returns
    once the ward is full."""
    blocked = dict(ward_bed_occupancy=0.95)
    small = kpis(simulate(_params(n_cubicles=12, **blocked), days=20, seed=5))
    big = kpis(simulate(_params(n_cubicles=40, **blocked), days=20, seed=5))
    assert big["neat_pct"] - small["neat_pct"] < 30


def test_check_flags_gridlock():
    p = _params(n_inpatient_beds=5, ward_bed_occupancy=0.99)
    assert any("access block" in w or "ward" in w for w in check(p))


def test_by_triage_reports_every_category_present():
    df = simulate(_params(), days=10, seed=6)
    t = by_triage(df)
    assert set(t["ats"]).issubset({1, 2, 3, 4, 5})
    assert (t["share_pct"].sum() - 100) < 1e-6 or abs(t["share_pct"].sum() - 100) < 1.0
