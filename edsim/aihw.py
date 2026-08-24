"""AIHW MyHospitals API client - public, open, no API key.

Base: https://myhospitalsapi.aihw.gov.au/api/v1

Why this matters for the hackathon: it gives you *real Western Australian*
ED volumes and triage-category mix per hospital, so your simulation is
calibrated to the actual system the sponsors run - not to invented numbers.
You can filter straight to East Metropolitan Health Service, this year's
partner (Royal Perth, Armadale-Kelmscott, Bentley, Kalamunda).
"""
from __future__ import annotations

import json
import pathlib
import time

import pandas as pd
import requests

BASE = "https://myhospitalsapi.aihw.gov.au/api/v1"
CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "aihw_cache"

# The AIHW edge rejects the default python-requests User-Agent with a 403.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "edsim/0.1 (WA Health Hackathon prototype; +https://wadsih.org.au)",
}

ED_WAITS = "MYH-ED-WAITS"   # presentations + % treated within recommended time, by triage
ED_TIME = "MYH-ED-TIME"     # time in ED: 4-hour (NEAT) compliance, median / 90th percentile
ED_PRES = "MYH-ED"          # total presentations


def _get(path: str, params: dict | None = None, *, cache_key: str | None = None) -> dict:
    if cache_key:
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{cache_key}.json"
        if f.exists():
            return json.loads(f.read_text())

    for attempt in range(3):
        r = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            data = r.json()
            if cache_key:
                (CACHE / f"{cache_key}.json").write_text(json.dumps(data))
            return data
        time.sleep(1 + attempt)
    r.raise_for_status()
    return {}


def _rows(payload: dict) -> list[dict]:
    result = payload.get("result", payload)
    if isinstance(result, dict):
        return result.get("data", [])
    return result or []


# --- reporting units -------------------------------------------------------

def reporting_units(refresh: bool = False) -> pd.DataFrame:
    """All reporting units, flattened with their state and LHN mapping."""
    key = None if refresh else "reporting_units"
    payload = _get("reporting-units", {"skip": 0, "top": 2000}, cache_key=key)
    out = []
    for u in _rows(payload):
        state = lhn = phn = None
        for m in u.get("mapped_reporting_units") or []:
            code = m["map_type"]["mapped_reporting_unit_code"]
            target = m["mapped_reporting_unit"]
            if code == "STATE_MAPPING":
                state = target["reporting_unit_code"]
            elif code == "H_LHN":
                lhn = target["reporting_unit_name"]
            elif code == "H_PHN":
                phn = target["reporting_unit_name"]
        out.append({
            "code": u["reporting_unit_code"],
            "name": u["reporting_unit_name"],
            "type": u["reporting_unit_type"]["reporting_unit_type_code"],
            "state": state,
            "lhn": lhn,
            "phn": phn,
            "private": u.get("private"),
            "closed": u.get("closed"),
            "lat": u.get("latitude"),
            "lon": u.get("longitude"),
        })
    return pd.DataFrame(out)


def wa_hospitals(lhn_contains: str | None = None) -> pd.DataFrame:
    """WA public hospitals, optionally filtered to one health service.

    >>> wa_hospitals("East Metropolitan")   # this year's EMHS partner
    """
    df = reporting_units()
    df = df[(df["state"] == "WA") & (df["type"] == "H") & (~df["closed"].fillna(False))]
    if lhn_contains:
        df = df[df["lhn"].fillna("").str.contains(lhn_contains, case=False)]
    return df.sort_values("name").reset_index(drop=True)


# --- measures --------------------------------------------------------------

def flat_extract(category: str, reporting_unit_code: str | None = None,
                 *, max_rows: int = 5000, refresh: bool = False) -> pd.DataFrame:
    """Paged flat data extract for a measure category (max 1000 rows/request)."""
    key = None if refresh else f"{category}_{reporting_unit_code or 'ALL'}"
    frames, skip = [], 0
    while skip < max_rows:
        params = {"skip": skip, "top": 1000}
        if reporting_unit_code:
            params["reporting_unit_code"] = reporting_unit_code
        payload = _get(f"flat-data-extract/{category}", params,
                       cache_key=f"{key}_{skip}" if key else None)
        rows = _rows(payload)
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < 1000:
            break
        skip += 1000
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    keep = ["reporting_unit_code", "reporting_unit_name", "mapped_state",
            "measure_code", "measure_name", "reported_measure_name",
            "reported_measure_category_name", "reported_measure_category_type_name",
            "reporting_start_date", "reporting_end_date", "value", "units_name",
            "suppression"]
    return df[[c for c in keep if c in df.columns]]


def ed_triage_profile(reporting_unit_code: str, period_end: str | None = None) -> pd.DataFrame:
    """Presentations and on-time-treatment rate by ATS category for one hospital.

    period_end: 'YYYY-MM-DD' reporting_end_date; defaults to the latest available.
    """
    from edsim.triage import aihw_name_to_ats

    df = flat_extract(ED_WAITS, reporting_unit_code)
    if df.empty:
        return df

    df = df[df["reported_measure_category_type_name"] == "Triage category"].copy()
    df["ats"] = aihw_name_to_ats(df["reported_measure_category_name"])
    df = df[df["ats"].notna()]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if period_end is None:
        period_end = df["reporting_end_date"].max()
    df = df[df["reporting_end_date"] == period_end]

    wide = df.pivot_table(index="ats", columns="measure_code",
                          values="value", aggfunc="first")
    wide = wide.rename(columns={"MYH0011": "presentations",
                                "MYH0010": "pct_treated_on_time"})
    wide["period_end"] = period_end
    wide["hospital"] = df["reporting_unit_name"].iloc[0]
    return wide.reset_index()
