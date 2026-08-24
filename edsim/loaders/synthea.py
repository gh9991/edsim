"""Synthea CSV output.

    java -jar synthea-with-dependencies.jar -p 5000 --exporter.csv.export true

Use Synthea for: schema shakeout, FHIR pipeline work, filling a UI with
plausible-looking patients, load-testing.

Do NOT use it to fit waiting-time or queueing behaviour. Synthea generates
encounters from disease-progression modules, not from a queueing model - there
is no waiting room, no bed contention, no diurnal arrival surge. A wait-time
model trained on Synthea learns artefacts of the generator.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from edsim import schema

# Synthea has no triage scale. Crude severity proxy so the frame is usable.
_DEFAULT_ACUITY = 3


def load(path: str | pathlib.Path, *, acuity_column: str | None = None,
         strict: bool = True) -> pd.DataFrame:
    root = pathlib.Path(path)
    enc = pd.read_csv(root / "encounters.csv", low_memory=False)
    enc = enc[enc["ENCOUNTERCLASS"].str.lower().eq("emergency")]

    out = pd.DataFrame({
        "encounter_id": enc["Id"].astype(str),
        "patient_id": enc["PATIENT"].astype(str),
        "arrival_ts": enc["START"],
        "depart_ts": enc["STOP"],
        "acuity": enc[acuity_column] if acuity_column else _DEFAULT_ACUITY,
        "acuity_scale": "ATS",
        "presenting_complaint": enc.get("DESCRIPTION"),
        "disposition": "discharged",
        "site": "synthea",
    })

    pt = root / "patients.csv"
    if pt.exists():
        pts = pd.read_csv(pt)[["Id", "BIRTHDATE", "GENDER"]]
        out = out.merge(pts, left_on="patient_id", right_on="Id", how="left")
        out["age"] = (
            (pd.to_datetime(out["arrival_ts"], errors="coerce", utc=True)
             - pd.to_datetime(out["BIRTHDATE"], errors="coerce", utc=True)).dt.days / 365.25
        )
        out["sex"] = out["GENDER"].map({"M": "M", "F": "F"}).fillna("U")
        out = out.drop(columns=["Id", "BIRTHDATE", "GENDER"])

    return schema.validate(out, strict=strict)
