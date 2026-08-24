"""MIMIC-IV-ED (and its 100-patient open-access demo).

Download the demo - no credentialing, no account:
    https://physionet.org/content/mimic-iv-ed-demo/2.2/
    wget -r -N -c -np https://physionet.org/files/mimic-iv-ed-demo/2.2/

The full MIMIC-IV-ED (~425k ED stays) has an identical schema, so anything
you build against the demo runs unchanged against the full set once your
PhysioNet credentialing comes through.

KNOWN LIMITATION - read this before you trust a wait-time model:
MIMIC-IV-ED has `intime` and `outtime` but **no first-clinician-contact
timestamp**. You can compute ED length of stay, you cannot directly compute
time-to-treatment. `seen_ts` is therefore left null here. Do not silently
substitute LOS for wait; `calibrate.from_encounters` will warn you.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from edsim import schema

DISPOSITION_MAP = {
    "HOME": "discharged",
    "ADMITTED": "admitted",
    "TRANSFER": "transferred",
    "EXPIRED": "died",
    "LEFT WITHOUT BEING SEEN": "did_not_wait",
    "ELOPED": "did_not_wait",
    "LEFT AGAINST MEDICAL ADVICE": "did_not_wait",
    "OTHER": "other",
}


def _read(root: pathlib.Path, name: str) -> pd.DataFrame:
    for cand in (root / f"{name}.csv.gz", root / f"{name}.csv",
                 root / "ed" / f"{name}.csv.gz", root / "ed" / f"{name}.csv"):
        if cand.exists():
            return pd.read_csv(cand, low_memory=False)
    raise FileNotFoundError(f"{name}.csv[.gz] not found under {root}")


def load(path: str | pathlib.Path, *, strict: bool = True) -> pd.DataFrame:
    root = pathlib.Path(path)
    stays = _read(root, "edstays")
    triage = _read(root, "triage")

    df = stays.merge(triage, on=["subject_id", "stay_id"], how="left",
                     suffixes=("", "_triage"))

    out = pd.DataFrame({
        "encounter_id": df["stay_id"].astype(str),
        "patient_id": df["subject_id"].astype(str),
        "arrival_ts": df["intime"],
        "triage_ts": pd.NaT,
        "seen_ts": pd.NaT,          # not recorded in MIMIC-IV-ED - see docstring
        "depart_ts": df["outtime"],
        "acuity": df.get("acuity"),
        "acuity_scale": "ESI",
        "sex": df.get("gender", pd.Series(dtype=object)).map(
            {"M": "M", "F": "F"}).fillna("U"),
        "presenting_complaint": df.get("chiefcomplaint"),
        "arrival_mode": df.get("arrival_transport"),
        "hr": df.get("heartrate"),
        "sbp": df.get("sbp"),
        "dbp": df.get("dbp"),
        "rr": df.get("resprate"),
        "temp": df.get("temperature"),
        "spo2": df.get("o2sat"),
        "pain": pd.to_numeric(df.get("pain"), errors="coerce"),
        "disposition": df.get("disposition", pd.Series(dtype=object))
            .astype(str).str.upper().map(DISPOSITION_MAP).fillna("other"),
        "site": "MIMIC-IV-ED",
    })

    # age lives in the MIMIC-IV core module, not in -ED. Join it if present.
    for cand in (root / "patients.csv.gz", root / "hosp" / "patients.csv.gz",
                 root.parent / "hosp" / "patients.csv.gz"):
        if cand.exists():
            pts = pd.read_csv(cand)[["subject_id", "anchor_age"]]
            out = out.merge(
                pts.assign(subject_id=lambda d: d.subject_id.astype(str)),
                left_on="patient_id", right_on="subject_id", how="left")
            out["age"] = out.pop("anchor_age")
            out = out.drop(columns=["subject_id"])
            break

    return schema.validate(out, strict=strict)
