"""WA Health Hackathon Data Hosting Portal - fill this in on 15 September.

This is the ONE file that should need writing on the day. Everything else -
simulator, calibration, metrics, models, UI - already speaks the canonical
schema.

Workflow:
  1. Download the challenge extract from the Portal.
  2. Run:  edsim inspect --path challenge.csv
     to print the source columns next to the canonical ones.
  3. Fill MAPPING below (source column -> canonical column).
  4. Run:  edsim calibrate --source portal --path challenge.csv
     Nothing downstream changes.

Keep the raw extract OUT of version control - see .gitignore. Hackathon data
is provided under the sponsor's terms; treat it as unpublishable.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from edsim import schema

# source column name -> canonical column name. TODO on the day.
MAPPING: dict[str, str] = {
    # "PRESENTATION_ID":      "encounter_id",
    # "UMRN":                 "patient_id",
    # "ARRIVAL_DTTM":         "arrival_ts",
    # "TRIAGE_DTTM":          "triage_ts",
    # "TREAT_DTTM":           "seen_ts",
    # "DEPARTURE_DTTM":       "depart_ts",
    # "TRIAGE_CATEGORY":      "acuity",
    # "AGE_YEARS":            "age",
    # "SEX":                  "sex",
    # "PRESENTING_PROBLEM":   "presenting_complaint",
    # "ARRIVAL_MODE":         "arrival_mode",
    # "ED_DEPARTURE_STATUS":  "disposition",
}

# source disposition value -> canonical disposition
DISPOSITION_MAP: dict[str, str] = {
    # "Admitted":                     "admitted",
    # "Discharged":                   "discharged",
    # "Transferred to other hospital":"transferred",
    # "Died in ED":                   "died",
    # "Did not wait":                 "did_not_wait",
}


def load(path: str | pathlib.Path, *, mapping: dict | None = None,
         disposition_map: dict | None = None, acuity_scale: str = "ATS",
         strict: bool = True, **read_kwargs) -> pd.DataFrame:
    path = pathlib.Path(path)
    reader = pd.read_parquet if path.suffix in (".parquet", ".pq") else pd.read_csv
    raw = reader(path, **read_kwargs)

    mapping = mapping or MAPPING
    if not mapping:
        raise NotImplementedError(
            "edsim/loaders/portal.py MAPPING is empty. Run `edsim inspect --path "
            f"{path}` to see the source columns, then fill it in."
        )

    out = raw.rename(columns=mapping)[[c for c in mapping.values() if c in
                                       raw.rename(columns=mapping).columns]].copy()
    out["acuity_scale"] = acuity_scale
    out["site"] = out.get("site", path.stem)

    dmap = disposition_map or DISPOSITION_MAP
    if dmap and "disposition" in out.columns:
        out["disposition"] = out["disposition"].map(dmap).fillna("other")

    return schema.validate(out, strict=strict)


def inspect(path: str | pathlib.Path, n: int = 5) -> pd.DataFrame:
    """Print source columns beside the canonical ones to speed up mapping."""
    path = pathlib.Path(path)
    reader = pd.read_parquet if path.suffix in (".parquet", ".pq") else pd.read_csv
    raw = reader(path) if path.suffix in (".parquet", ".pq") else reader(path, nrows=2000)
    return pd.DataFrame({
        "source_column": raw.columns,
        "dtype": [str(t) for t in raw.dtypes],
        "non_null": raw.notna().sum().values,
        "sample": [", ".join(map(str, raw[c].dropna().head(n).tolist()))[:80]
                   for c in raw.columns],
    })
