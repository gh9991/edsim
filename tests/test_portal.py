"""Portal loader tests.

Fixtures are written in the official EDDC/HMDC field names rather than loading
the Department's sample file, which cannot be committed.
"""
import pandas as pd
import pytest

from edsim.loaders import portal


def _eddc(tmp_path, rows=None):
    rows = rows or [
        # arrival, seen, bed request, discharge, ATS, departure_status
        ("2022-01-16 19:01:03", "2022-01-16 19:58:44", "2022-01-16 20:47:42",
         "2022-01-16 21:54:03", 3, 1),                       # admitted, boarded
        ("2022-02-17 00:14:07", "2022-02-17 00:14:38", "",
         "2022-02-17 03:48:07", 4, 2),                       # discharged
        ("2022-03-11 20:07:34", "2022-03-11 20:48:52", "",
         "2022-03-11 21:00:00", 5, 4),                       # did not wait
        ("2022-04-01 08:00:00", "2022-04-01 08:02:00", "2022-04-01 09:00:00",
         "2022-04-01 15:00:00", 2, 10),                      # ED obs ward
    ]
    df = pd.DataFrame(rows, columns=[
        "presentation_datetime", "clinical_care_commencement_datetime",
        "bed_request_datetime", "discharge_datetime",
        "triage_category", "departure_status"])
    df["synth_person_ID"] = [f"S{i}" for i in range(len(df))]
    df["sex"] = [1, 2, 1, 2]
    df["age"] = [65, 45, 75, 30]
    df["mode_of_arrival"] = [1, 1, 3, 3]
    df["ethnicity"] = [0, 0, 1, 0]
    df["establishment_code"] = 8005.0
    p = tmp_path / "eddc.csv"
    df.to_csv(p, index=False)
    return p


def test_official_field_names_map_to_canonical(tmp_path):
    df = portal.load_eddc(_eddc(tmp_path))
    assert len(df) == 4
    for c in ("arrival_ts", "seen_ts", "depart_ts", "bed_request_ts", "ats"):
        assert c in df.columns
    assert df["arrival_ts"].notna().all()


def test_triage_category_is_ats_not_esi(tmp_path):
    """WA records the ATS directly - no scale conversion should happen."""
    df = portal.load_eddc(_eddc(tmp_path))
    assert (df["acuity_scale"] == "ATS").all()
    assert df["ats"].tolist() == [3, 4, 5, 2]


def test_wait_and_boarding_are_measurable(tmp_path):
    """The two things MIMIC cannot give us."""
    df = portal.load_eddc(_eddc(tmp_path)).sort_values("arrival_ts")
    assert round(df["wait_to_treat_min"].iloc[0], 1) == 57.7
    assert round(df["boarding_min"].iloc[0], 1) == 66.4     # discharge - bed request
    assert pd.isna(df["boarding_min"].iloc[1])              # no bed requested


def test_departure_status_codes_decode(tmp_path):
    df = portal.load_eddc(_eddc(tmp_path)).sort_values("arrival_ts")
    assert df["disposition"].tolist() == [
        "admitted", "discharged", "did_not_wait", "admitted"]
    # ED observation ward (10) still consumes a bed
    assert df["needs_bed"].tolist() == [True, False, False, True]


def test_numeric_codes_and_decoded_text_both_work(tmp_path):
    numeric = portal.load_eddc(_eddc(tmp_path))
    raw = pd.read_csv(_eddc(tmp_path))
    raw["sex"] = raw["sex"].map({1: "M", 2: "F"})
    raw["mode_of_arrival"] = raw["mode_of_arrival"].map({1: "private transport",
                                                         3: "ambulance"})
    p = tmp_path / "decoded.csv"
    raw.to_csv(p, index=False)
    decoded = portal.load_eddc(p)
    assert numeric["sex"].tolist() == decoded["sex"].tolist()


def test_column_name_drift_is_tolerated(tmp_path):
    """`person_ID` instead of `synth_person_ID`, `metropolitan_flag` instead
    of `metropolitan_hospital_flag` - both have been seen in real extracts."""
    raw = pd.read_csv(_eddc(tmp_path)).rename(
        columns={"synth_person_ID": "person_ID"})
    raw["metropolitan_flag"] = 1
    p = tmp_path / "drift.csv"
    raw.to_csv(p, index=False)
    df = portal.load_eddc(p)
    assert df["patient_id"].notna().all()


def test_missing_arrival_column_fails_loudly(tmp_path):
    raw = pd.read_csv(_eddc(tmp_path)).drop(columns=["presentation_datetime"])
    p = tmp_path / "broken.csv"
    raw.to_csv(p, index=False)
    with pytest.raises(KeyError, match="presentation_datetime"):
        portal.load_eddc(p)


def test_inspect_flags_the_leaky_fields(tmp_path):
    rep = portal.inspect(_eddc(tmp_path)).set_index("source_column")
    assert rep.loc["bed_request_datetime", "leakage"].startswith("LEAKY")
    assert rep.loc["departure_status", "leakage"].startswith("LEAKY")
    assert rep.loc["presentation_datetime", "leakage"] == "ok"
    assert rep.loc["triage_category", "leakage"] == "ok"


def test_hmdc_links_on_the_episode_that_follows_ed_departure(tmp_path):
    """The data dictionary says inpatient admission time is when the patient
    leaves the ED, so the link is forward-in-time and bounded - not merely
    'same person', who may have unrelated admissions all year."""
    eddc = portal.load_eddc(_eddc(tmp_path))
    hmdc = pd.DataFrame({
        "synth_person_ID": ["S0", "S0"],
        "admission_datetime": ["2022-01-16 22:10:00",   # right after ED exit
                               "2022-09-01 10:00:00"],  # unrelated, months later
        "separation_datetime": ["2022-01-20 10:00:00", "2022-09-04 10:00:00"],
    })
    p = tmp_path / "hmdc.csv"
    hmdc.to_csv(p, index=False)

    linked = portal.link(eddc, portal.load_hmdc(p))
    s0 = linked[linked["patient_id"] == "S0"].iloc[0]
    assert s0["linked_inpatient"]
    assert str(s0["ip_admission_ts"]).startswith("2022-01-16")
    assert linked["linked_inpatient"].sum() == 1     # the September one is not picked up
