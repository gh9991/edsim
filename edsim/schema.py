"""The canonical encounter schema.

Every loader returns a DataFrame with these columns. The simulator, the
calibrator, the metrics and any model you train read only from here, so
swapping the data source is a one-file change.
"""
from __future__ import annotations

import pandas as pd

# --- column contract -------------------------------------------------------

REQUIRED = ["encounter_id", "arrival_ts", "acuity"]

OPTIONAL = [
    "patient_id",
    "triage_ts",       # triage completed
    "seen_ts",         # first definitive clinician contact  (often absent!)
    "depart_ts",       # physically left the ED
    "acuity_scale",    # "ATS" (AU) or "ESI" (US) - see edsim.triage
    "age",
    "sex",             # M / F / U
    "presenting_complaint",
    "arrival_mode",    # ambulance / walk-in / other
    "hr", "sbp", "dbp", "rr", "temp", "spo2", "pain",
    "disposition",
    "site",            # hospital / campus identifier
]

DERIVED = ["triage_wait_min", "wait_to_treat_min", "los_min", "admitted", "ats"]

COLUMNS = REQUIRED + OPTIONAL

DISPOSITIONS = (
    "admitted",
    "discharged",
    "transferred",
    "died",
    "did_not_wait",   # LWBS / eloped / left against medical advice
    "other",
)

_TS_COLS = ["arrival_ts", "triage_ts", "seen_ts", "depart_ts"]
_NUM_COLS = ["age", "hr", "sbp", "dbp", "rr", "temp", "spo2", "pain"]


class SchemaError(ValueError):
    pass


def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    for c in _TS_COLS:
        df[c] = pd.Series(dtype="datetime64[ns]")
    for c in _NUM_COLS:
        df[c] = pd.Series(dtype="float64")
    df["acuity"] = pd.Series(dtype="float64")
    return df


def _minutes(a: pd.Series, b: pd.Series) -> pd.Series:
    return (b - a).dt.total_seconds() / 60.0


def validate(df: pd.DataFrame, *, strict: bool = True, keep_extra: bool = True) -> pd.DataFrame:
    """Coerce a loader's output into the canonical schema and derive metrics.

    strict=True raises on missing required columns; strict=False fills them
    with NA so you can inspect a half-mapped source.
    """
    df = df.copy()

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        if strict:
            raise SchemaError(f"missing required columns: {missing}")
        for c in missing:
            df[c] = pd.NA

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    for c in _TS_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in _NUM_COLS + ["acuity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df["acuity_scale"].isna().all():
        df["acuity_scale"] = "ATS"

    bad = df.loc[df["acuity"].notna() & ~df["acuity"].between(1, 5), "acuity"]
    if len(bad) and strict:
        raise SchemaError(f"acuity must be 1-5 (1 = most urgent); saw {sorted(bad.unique())[:5]}")

    df["disposition"] = (
        df["disposition"].astype("object").where(df["disposition"].isin(DISPOSITIONS), "other")
    )

    # derived
    df["triage_wait_min"] = _minutes(df["arrival_ts"], df["triage_ts"])
    df["wait_to_treat_min"] = _minutes(df["arrival_ts"], df["seen_ts"])
    df["los_min"] = _minutes(df["arrival_ts"], df["depart_ts"])
    df["admitted"] = df["disposition"].eq("admitted")

    from edsim.triage import to_ats  # local import avoids a cycle
    df["ats"] = to_ats(df["acuity"], df["acuity_scale"])

    cols = COLUMNS + DERIVED
    if keep_extra:
        cols = cols + [c for c in df.columns if c not in cols]
    return df[cols].sort_values("arrival_ts").reset_index(drop=True)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage report - which canonical fields did this source actually fill?"""
    rows = []
    for c in COLUMNS + DERIVED:
        if c not in df.columns:
            rows.append({"column": c, "present": False, "non_null": 0, "coverage": 0.0})
            continue
        n = int(df[c].notna().sum())
        rows.append(
            {"column": c, "present": True, "non_null": n,
             "coverage": round(n / len(df), 3) if len(df) else 0.0}
        )
    return pd.DataFrame(rows)
