"""ED performance metrics, using the measures WA actually reports on.

Deliberately the same names the sponsors use, so a judge from DoH or EMHS
recognises the numbers instantly:

  ATS on-time %   ACEM target: treatment commenced within the ATS time target
  NEAT            % of patients who depart the ED within 4 hours of arrival
  Access block    admitted patients boarding in ED > 8 hours
  DNW             did-not-wait rate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from edsim.triage import (
    ATS_NAME,
    ATS_PERFORMANCE_TARGET,
    ATS_TARGET_MIN,
    NEAT_TARGET_MIN,
    seen_on_time,
)

ACCESS_BLOCK_MIN = 8 * 60


def kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"presentations": 0}

    wait = df["wait_to_treat_min"]
    los = df["los_min"]
    board = df.get("boarding_min", pd.Series(dtype=float))

    out = {
        "presentations": int(len(df)),
        "per_day": round(len(df) / max(
            (df["arrival_ts"].max() - df["arrival_ts"].min()).total_seconds() / 86400, 1e-9), 1),
        "median_wait_min": round(float(wait.median()), 1),
        "p90_wait_min": round(float(wait.quantile(0.90)), 1),
        "on_time_pct": round(float(seen_on_time(wait, df["ats"]).mean() * 100), 1),
        "median_los_min": round(float(los.median()), 1),
        "p90_los_min": round(float(los.quantile(0.90)), 1),
        "neat_pct": round(float(los.le(NEAT_TARGET_MIN).mean() * 100), 1),
        "admitted_pct": round(float(df["admitted"].mean() * 100), 1),
        "dnw_pct": round(float(df["disposition"].eq("did_not_wait").mean() * 100), 1),
    }
    if len(board):
        adm = df[df["admitted"]]
        out["median_boarding_min"] = round(float(adm["boarding_min"].median()), 1) if len(adm) else 0.0
        out["access_block_pct"] = round(
            float(adm["boarding_min"].ge(ACCESS_BLOCK_MIN).mean() * 100), 1) if len(adm) else 0.0
    return out


def by_triage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ATS breakdown against the ACEM performance thresholds."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["on_time"] = seen_on_time(df["wait_to_treat_min"], df["ats"])

    g = df.groupby("ats", dropna=True)
    out = pd.DataFrame({
        "category": [ATS_NAME.get(int(a), "?") for a in g.groups],
        "presentations": g.size(),
        "share_pct": (g.size() / len(df) * 100).round(1),
        "target_min": [ATS_TARGET_MIN.get(int(a), np.nan) for a in g.groups],
        "median_wait_min": g["wait_to_treat_min"].median().round(1),
        "p90_wait_min": g["wait_to_treat_min"].quantile(0.90).round(1),
        "on_time_pct": (g["on_time"].mean() * 100).round(1),
        "acem_target_pct": [ATS_PERFORMANCE_TARGET.get(int(a), np.nan) * 100 for a in g.groups],
        "median_los_min": g["los_min"].median().round(1),
        "admitted_pct": (g["admitted"].mean() * 100).round(1),
    })
    out["meets_target"] = out["on_time_pct"] >= out["acem_target_pct"]
    return out.reset_index()


def hourly_load(df: pd.DataFrame) -> pd.DataFrame:
    """Arrivals and waits by hour of day - the shape every ED exec asks for."""
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["hour"] = d["arrival_ts"].dt.hour
    g = d.groupby("hour")
    return pd.DataFrame({
        "arrivals": g.size(),
        "median_wait_min": g["wait_to_treat_min"].median().round(1),
        "on_time_pct": (seen_on_time(d["wait_to_treat_min"], d["ats"])
                        .groupby(d["hour"]).mean() * 100).round(1),
    }).reset_index()


def occupancy_series(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Reconstruct ED census over time from arrival/departure timestamps."""
    if df.empty:
        return pd.DataFrame()
    starts = df["arrival_ts"].dropna()
    ends = df["depart_ts"].dropna()
    events = pd.concat([
        pd.DataFrame({"ts": starts, "delta": 1}),
        pd.DataFrame({"ts": ends, "delta": -1}),
    ]).sort_values("ts")
    events["census"] = events["delta"].cumsum()
    s = events.set_index("ts")["census"].resample(freq).max().ffill()
    return s.rename("census").reset_index()


def compare_to_aihw(sim_df: pd.DataFrame, reporting_unit_code: str,
                    period_end: str | None = None) -> pd.DataFrame:
    """Back-test: simulated on-time % vs what AIHW actually observed.

    A simulation nobody has validated is a guess with extra steps. This is the
    table that turns "we built a model" into "we built a model that reproduces
    Royal Perth to within N points". Expect gaps on ATS 3-4: the default
    service times and capacity are not from data, so tune those first.
    """
    from edsim.aihw import ed_triage_profile

    obs = ed_triage_profile(reporting_unit_code, period_end)
    if obs.empty:
        raise ValueError(f"no AIHW data for {reporting_unit_code}")

    sim = by_triage(sim_df)[["ats", "presentations", "on_time_pct"]]
    sim = sim.rename(columns={"presentations": "sim_presentations",
                              "on_time_pct": "sim_on_time_pct"})
    obs = obs.rename(columns={"presentations": "obs_presentations",
                              "pct_treated_on_time": "obs_on_time_pct"})

    out = obs.merge(sim, on="ats", how="outer").sort_values("ats")
    out["category"] = out["ats"].map(ATS_NAME)
    out["on_time_gap_pp"] = (out["sim_on_time_pct"] - out["obs_on_time_pct"]).round(1)
    return out[["ats", "category", "obs_presentations", "sim_presentations",
                "obs_on_time_pct", "sim_on_time_pct", "on_time_gap_pp"]]
