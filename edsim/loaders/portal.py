"""WA Health Hackathon portal data: EDDC + HMDC.

Written against the Department of Health's published field definitions
(*Linked Representative Synthetic EDDC / HMDC 2022 Data Dictionary*, July 2025)
rather than guessed from a sample, so the codes below are the real ones.

Two collections, linked by person id:

    EDDC   Emergency Department Data Collection - one row per ED presentation
    HMDC   Hospital Morbidity Data Collection   - one row per inpatient episode

EDDC carries four timestamps, which is what makes it far better than MIMIC for
operational work:

    presentation_datetime                arrival
    clinical_care_commencement_datetime  first seen by a practitioner
    bed_request_datetime                 an inpatient bed was requested
    discharge_datetime                   left the ED

`clinical_care_commencement_datetime` is the field MIMIC does not have, so ATS
on-time performance is directly measurable here. `discharge_datetime` minus
`bed_request_datetime` is access block, measured rather than modelled.

What EDDC does **not** have: any clinical measurement at all. No heart rate, no
blood pressure, no chief complaint. See `edsim.information_value` for what that
costs.

Field names drift between extracts (`synth_person_ID` in the published sample,
`person_ID` elsewhere; `metropolitan_hospital_flag` vs `metropolitan_flag`), so
every column is looked up through an alias list. Codes may arrive numeric or
already decoded to text; both are handled.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from edsim import schema

# ── column aliases: canonical name -> names seen in the wild ────────────────
EDDC_ALIASES: dict[str, list[str]] = {
    "patient_id":  ["synth_person_ID", "person_ID", "person_id"],
    "arrival_ts":  ["presentation_datetime"],
    "seen_ts":     ["clinical_care_commencement_datetime"],
    "bed_request_ts": ["bed_request_datetime"],
    "depart_ts":   ["discharge_datetime"],
    "acuity":      ["triage_category"],
    "age":         ["age"],
    "sex":         ["sex"],
    "ethnicity":   ["ethnicity", "aboriginal_status"],
    "arrival_mode": ["mode_of_arrival"],
    "referral_source": ["referral_source"],
    "site":        ["establishment_code"],
    "metro":       ["metropolitan_hospital_flag", "metropolitan_flag"],
    "_departure_status": ["departure_status"],
    "_dx_chapter": ["primary_diagnosis_ICD10AM_chapter"],
    "_mental_health": ["mental_health_attendance"],
    "_avoidable_gp": ["potentially_avoidable_general_practitioner_type_attendance"],
    "_self_harm":   ["self_harm_attendance", "Self_harm_attendance"],
    "_mental_health_admission": ["mental_health_admission"],
    "_drugs_alcohol": ["affected_by_drugs_and_or_alcohol"],
}

HMDC_ALIASES: dict[str, list[str]] = {
    "patient_id":     ["synth_person_ID", "person_ID", "person_id"],
    "admission_ts":   ["admission_datetime"],
    "separation_ts":  ["separation_datetime"],
    "admission_status": ["admission_status"],
    "care_type":      ["care_type"],
    "mdc":            ["major_diagnostic_categ_current"],
    "procedure":      ["principal_procedure"],
    "site":           ["establishment_code"],
}

# ── code sets, straight out of the data dictionary ─────────────────────────
DEPARTURE_STATUS = {
    1:  "Admitted to ward/other admitted patient unit",
    2:  "ED service event completed; departed under own care",
    3:  "Transferred to another hospital for admission",
    4:  "Did not wait to be attended by medical officer",
    5:  "Left at own risk",
    6:  "Died in ED",
    8:  "Referred to After Hours General Practitioner",
    9:  "Unknown",
    10: "Admitted to ED Observation Ward",
    14: "Returned to Hospital in the Home",
    19: "Discharged After Admission",
    20: "Reversal",
}

DEPARTURE_TO_DISPOSITION = {
    1: "admitted", 2: "discharged", 3: "transferred", 4: "did_not_wait",
    5: "did_not_wait", 6: "died", 8: "discharged", 9: "other",
    10: "admitted", 14: "other", 19: "admitted", 20: "other",
}

# Which departure statuses mean "this patient consumed an inpatient bed".
# A modelling choice, so it is named and adjustable rather than buried:
# ward admission, ED observation ward, transfer *for admission*, and
# discharged-after-admission all consume a bed somewhere.
NEEDS_BED_CODES = frozenset({1, 3, 10, 19})

MODE_OF_ARRIVAL = {
    1: "private transport", 2: "public transport", 3: "ambulance",
    4: "hospital transport", 5: "police", 6: "helicopter rescue",
    7: "royal flying doctor service", 8: "other", 9: "unknown", 10: "taxi",
}

SEX = {1: "M", 2: "F"}
ETHNICITY = {0: "non-indigenous", 1: "aboriginal and/or torres strait islander"}

# ── leakage classification (see FINDINGS.md) ───────────────────────────────
# Not known at triage. Using any of these to predict admission inflates the
# score and produces a model that cannot run in production.
EDDC_LEAKY = {
    "bed_request_datetime":
        "requesting a bed IS the admission decision",
    "departure_status":
        "the outcome",
    "discharge_datetime":
        "the outcome",
    "primary_diagnosis_ICD10AM_chapter":
        "data dictionary: 'Chapter level roll of the Principal Diagnosis' - "
        "coded after the episode ends",
    "mental_health_admission":
        "the worst of them - data dictionary: 'based on DEPARTURE STATUS OF "
        "ADMITTED and Principal Diagnoses'. It literally contains the target",
}
EDDC_SUSPECT = {
    "mental_health_attendance":
        "'based on ... primary diagnosis codes, or ... symptoms' - partly "
        "derived from the post-hoc diagnosis",
    "potentially_avoidable_general_practitioner_type_attendance":
        "'pre-calculated' from undisclosed variables",
}


def _pick(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return df[lower[n.lower()]]
    return None


def _decode(s: pd.Series, mapping: dict) -> pd.Series:
    """Accept either the numeric codes or already-decoded text."""
    if s is None:
        return None
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.map(mapping)
    return s.astype(str).str.strip().str.lower()


def _read(path: str | pathlib.Path, **kw) -> pd.DataFrame:
    path = pathlib.Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, **kw)


def load_eddc(path, *, strict: bool = True, **kw) -> pd.DataFrame:
    """EDDC extract -> canonical encounters."""
    raw = _read(path, **kw)
    out = pd.DataFrame(index=raw.index)

    for canon, aliases in EDDC_ALIASES.items():
        col = _pick(raw, aliases)
        if col is not None:
            out[canon] = col.values

    if "arrival_ts" not in out:
        raise KeyError(
            f"no presentation_datetime in {path}. Columns present: "
            f"{list(raw.columns)}. Add the name to EDDC_ALIASES."
        )

    out["encounter_id"] = (
        out.get("patient_id", pd.Series(range(len(out)))).astype(str)
        + "_" + out["arrival_ts"].astype(str)
    )
    out["acuity_scale"] = "ATS"          # triage_category IS the ATS, no mapping

    ds_raw = (out.pop("_departure_status") if "_departure_status" in out.columns
              else pd.Series(index=out.index, dtype=float))
    ds = pd.to_numeric(ds_raw, errors="coerce")
    out["disposition"] = ds.map(DEPARTURE_TO_DISPOSITION).fillna("other")
    out["needs_bed"] = ds.isin(NEEDS_BED_CODES)
    out["departure_status_label"] = ds.map(DEPARTURE_STATUS)

    if "sex" in out:
        # canonical schema wants M / F / U regardless of how it arrived
        out["sex"] = (_decode(out["sex"], SEX).astype(str).str.upper()
                      .str[0].where(lambda s: s.isin(["M", "F"]), "U"))
    if "arrival_mode" in out:
        out["arrival_mode"] = _decode(out["arrival_mode"], MODE_OF_ARRIVAL)
    if "ethnicity" in out:
        out["ethnicity"] = _decode(out["ethnicity"], ETHNICITY)

    for c in ("bed_request_ts",):
        if c in out:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    df = schema.validate(out, strict=strict)

    # Access block, measured rather than modelled. MIMIC cannot do this.
    if "bed_request_ts" in df.columns:
        df["boarding_min"] = (
            (df["depart_ts"] - df["bed_request_ts"]).dt.total_seconds() / 60
        ).where(lambda s: s >= 0)
    return df


def load_hmdc(path, **kw) -> pd.DataFrame:
    """HMDC extract -> inpatient episodes, keyed by patient_id."""
    raw = _read(path, **kw)
    out = pd.DataFrame(index=raw.index)
    for canon, aliases in HMDC_ALIASES.items():
        col = _pick(raw, aliases)
        if col is not None:
            out[canon] = col.values
    for c in ("admission_ts", "separation_ts"):
        if c in out:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    if {"admission_ts", "separation_ts"} <= set(out.columns):
        out["inpatient_los_hours"] = (
            (out["separation_ts"] - out["admission_ts"]).dt.total_seconds() / 3600
        )
    return out


def link(eddc: pd.DataFrame, hmdc: pd.DataFrame, *,
         max_gap_hours: float = 24.0) -> pd.DataFrame:
    """Attach each ED presentation to the inpatient episode it led to.

    The data dictionary is explicit about the join: *"Where a patient is
    assessed in an ED and the decision to admit is made, the admission
    commencement time should be the time the patient leaves the ED."* So the
    right episode is the first one starting at or shortly after ED departure -
    not merely the same person, who may have unrelated admissions all year.
    """
    if "patient_id" not in eddc.columns or "patient_id" not in hmdc.columns:
        raise KeyError("both frames need patient_id to link")

    e = eddc.sort_values("depart_ts").reset_index(drop=True)
    h = (hmdc.dropna(subset=["admission_ts"])
             .sort_values("admission_ts").reset_index(drop=True))

    merged = pd.merge_asof(
        e, h.add_prefix("ip_"),
        left_on="depart_ts", right_on="ip_admission_ts",
        left_by="patient_id", right_by="ip_patient_id",
        direction="forward",
        tolerance=pd.Timedelta(hours=max_gap_hours),
    )
    merged["linked_inpatient"] = merged["ip_admission_ts"].notna()
    return merged


def load(path=None, *, eddc: str | None = None, hmdc: str | None = None,
         strict: bool = True, **kw) -> pd.DataFrame:
    """Entry point used by `edsim.loaders.load("portal", ...)`.

    Pass `path` (or `eddc`) for the ED extract alone; add `hmdc` to link the
    inpatient side.
    """
    eddc_path = eddc or path
    if eddc_path is None:
        raise ValueError("give path= (EDDC extract) and optionally hmdc=")
    df = load_eddc(eddc_path, strict=strict, **kw)
    if hmdc:
        df = link(df, load_hmdc(hmdc))
    return df


def inspect(path, n: int = 5) -> pd.DataFrame:
    """Source columns beside what they map to, plus a leakage verdict."""
    raw = _read(path, nrows=2000) if str(path).endswith(".csv") else _read(path)
    rev = {a.lower(): c for c, al in EDDC_ALIASES.items() for a in al}
    rows = []
    for c in raw.columns:
        verdict = ("LEAKY - " + EDDC_LEAKY[c]) if c in EDDC_LEAKY else (
            ("SUSPECT - " + EDDC_SUSPECT[c]) if c in EDDC_SUSPECT else "ok")
        rows.append({
            "source_column": c,
            "maps_to": rev.get(c.lower(), "—"),
            "leakage": verdict,
            "sample": ", ".join(map(str, raw[c].dropna().head(n).tolist()))[:60],
        })
    return pd.DataFrame(rows)
