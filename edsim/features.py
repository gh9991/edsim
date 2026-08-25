"""Features known **at triage time** - and a guard against using anything else.

The single most common way a hackathon admission model gets a great score and
is worthless: training on columns that only exist after the patient's visit
ended. `los_min`, `depart_ts`, `boarding_min`, `disposition` are all outcomes.
Use them and you will report an AUC of 0.99 and get taken apart in questions.

Everything here is computable from what the ED knows the moment triage
finishes. That includes crowding: how busy the department already is when this
patient walks in is one of the strongest predictors of what happens next, and
it is knowable in real time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that only exist once the visit is over. Never features.
LEAKY = [
    "seen_ts", "depart_ts", "disposition", "admitted",
    "los_min", "wait_to_treat_min", "boarding_min", "space", "preempted",
]

VITALS = ["hr", "sbp", "dbp", "rr", "temp", "spo2", "pain"]


class LeakageError(ValueError):
    pass


def assert_no_leakage(feature_names) -> None:
    bad = sorted(set(feature_names) & set(LEAKY))
    if bad:
        raise LeakageError(
            f"outcome columns used as features: {bad}. These are not known at "
            "triage time - a model using them cannot be deployed and its score "
            "is meaningless."
        )


def _crowding(df: pd.DataFrame) -> pd.DataFrame:
    """How full is the ED when each patient arrives?

    Census is reconstructed from arrival/departure of *earlier* patients only,
    so nothing leaks backwards from the future.
    """
    arr = df["arrival_ts"].to_numpy()
    dep = df["depart_ts"].to_numpy()

    order = np.argsort(arr)
    census = np.zeros(len(df), dtype=float)
    waiting = np.zeros(len(df), dtype=float)
    seen = df["seen_ts"].to_numpy()

    # occupancy at t = arrivals before t that had not yet departed
    dep_sorted = np.sort(dep[~pd.isna(dep)])
    for i in order:
        t = arr[i]
        arrived_before = np.searchsorted(arr[order], t, side="left")
        departed_before = np.searchsorted(dep_sorted, t, side="left")
        census[i] = arrived_before - departed_before
        # still in the waiting room: arrived before t, not yet seen at t
        waiting[i] = np.sum((arr[order][:arrived_before] <= t)
                            & (pd.isna(seen[order][:arrived_before])
                               | (seen[order][:arrived_before] > t)))
    return pd.DataFrame({"ed_census_on_arrival": census,
                         "waiting_room_on_arrival": waiting}, index=df.index)


def build(df: pd.DataFrame, *, include_crowding: bool = True) -> pd.DataFrame:
    """Canonical encounters -> triage-time feature matrix."""
    df = df.sort_values("arrival_ts").reset_index(drop=True)
    X = pd.DataFrame(index=df.index)

    X["ats"] = pd.to_numeric(df["ats"], errors="coerce")
    X["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    X["is_female"] = df.get("sex", pd.Series(index=df.index)).eq("F").astype(float)
    X["by_ambulance"] = (
        df.get("arrival_mode", pd.Series(index=df.index))
        .astype(str).str.lower().str.contains("ambulance").astype(float)
    )

    for v in VITALS:
        X[v] = pd.to_numeric(df.get(v), errors="coerce")

    # simple physiological derangement count - a crude but very robust signal
    X["n_abnormal_vitals"] = (
        X["hr"].gt(100).fillna(False).astype(int)
        + X["sbp"].lt(100).fillna(False).astype(int)
        + X["rr"].gt(20).fillna(False).astype(int)
        + X["spo2"].lt(94).fillna(False).astype(int)
        + X["temp"].gt(38).fillna(False).astype(int)
    )

    t = df["arrival_ts"]
    X["hour"] = t.dt.hour
    X["dow"] = t.dt.dayofweek
    X["is_weekend"] = t.dt.dayofweek.ge(5).astype(float)
    X["is_overnight"] = t.dt.hour.between(0, 6).astype(float)
    # cyclical encoding so 23:00 and 00:00 are adjacent
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

    if include_crowding and df["depart_ts"].notna().any():
        X = X.join(_crowding(df))

    assert_no_leakage(X.columns)
    return X


def target(df: pd.DataFrame, name: str = "admitted") -> pd.Series:
    return df.sort_values("arrival_ts").reset_index(drop=True)[name].astype(int)
