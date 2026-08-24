import numpy as np
import pandas as pd

from edsim import schema
from edsim.calibrate import SimParams, from_encounters, size_capacity


def _encounters(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    arrive = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 30 * 24 * 60, n)), unit="m")
    ats = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.02, 0.2, 0.4, 0.33, 0.05])
    wait = rng.exponential(20, n)
    svc = rng.lognormal(4.2, 0.6, n)
    return schema.validate(pd.DataFrame({
        "encounter_id": [f"E{i}" for i in range(n)],
        "arrival_ts": arrive,
        "seen_ts": arrive + pd.to_timedelta(wait, unit="m"),
        "depart_ts": arrive + pd.to_timedelta(wait + svc, unit="m"),
        "acuity": ats,
        "acuity_scale": "ATS",
        "disposition": np.where(rng.random(n) < 0.25, "admitted", "discharged"),
    }))


def test_fits_volume_and_mix_from_data():
    df = _encounters()
    p = from_encounters(df)
    assert 50 < p.daily_arrivals < 100
    assert abs(sum(p.acuity_mix.values()) - 1.0) < 1e-6
    assert p.acuity_mix[3] > p.acuity_mix[1]


def test_records_provenance_and_warns_on_los_proxy():
    df = _encounters()
    assert "from seen_ts" in from_encounters(df).notes
    df2 = df.copy()
    df2["seen_ts"] = pd.NaT
    assert "LOS PROXY" in from_encounters(df2).notes


def test_params_round_trip_through_json(tmp_path):
    p = size_capacity(SimParams(daily_arrivals=210))
    f = tmp_path / "p.json"
    p.to_json(f)
    q = SimParams.from_json(f)
    assert q.n_cubicles == p.n_cubicles
    assert q.acuity_mix == p.acuity_mix
    assert all(isinstance(k, int) for k in q.acuity_mix)


def test_size_capacity_scales_with_demand():
    small = size_capacity(SimParams(daily_arrivals=80))
    big = size_capacity(SimParams(daily_arrivals=400))
    assert big.n_cubicles > small.n_cubicles
    assert big.n_inpatient_beds > small.n_inpatient_beds


def test_date_shifted_timestamps_are_rejected_not_silently_used():
    """MIMIC shifts each patient into a random future year. A naive rate
    calculation gives ~0 arrivals/day and an empty simulation."""
    df = _encounters(n=200)
    shifted = df.copy()
    shifted.loc[shifted.index[-1], "arrival_ts"] += pd.Timedelta(days=365 * 100)
    p = from_encounters(shifted)
    assert "REJECTED empirical rate" in p.notes
    assert p.daily_arrivals == SimParams().daily_arrivals
