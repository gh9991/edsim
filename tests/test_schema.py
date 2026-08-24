import pandas as pd
import pytest

from edsim import schema
from edsim.triage import seen_on_time


def _raw():
    return pd.DataFrame({
        "encounter_id": ["a", "b"],
        "arrival_ts": ["2026-09-19 08:00", "2026-09-19 09:00"],
        "seen_ts": ["2026-09-19 08:20", "2026-09-19 11:00"],
        "depart_ts": ["2026-09-19 11:00", "2026-09-19 16:00"],
        "acuity": [3, 4],
        "disposition": ["discharged", "admitted"],
    })


def test_validate_derives_waits_and_los():
    df = schema.validate(_raw())
    assert df["wait_to_treat_min"].tolist() == [20.0, 120.0]
    assert df["los_min"].tolist() == [180.0, 420.0]
    assert df["admitted"].tolist() == [False, True]


def test_missing_required_column_raises():
    with pytest.raises(schema.SchemaError):
        schema.validate(_raw().drop(columns=["acuity"]))


def test_unknown_disposition_becomes_other():
    raw = _raw()
    raw.loc[0, "disposition"] = "WALKED OUT THE BACK"
    assert schema.validate(raw)["disposition"].iloc[0] == "other"


def test_esi_is_mapped_onto_ats():
    raw = _raw()
    raw["acuity_scale"] = "ESI"
    assert schema.validate(raw)["ats"].tolist() == [3, 4]


def test_on_time_uses_ats_targets():
    df = schema.validate(_raw())
    # ATS 3 target is 30 min (20 -> on time); ATS 4 target is 60 (120 -> late)
    assert seen_on_time(df["wait_to_treat_min"], df["ats"]).tolist() == [True, False]


def test_extra_columns_survive_validation():
    raw = _raw()
    raw["boarding_min"] = [0.0, 90.0]
    assert "boarding_min" in schema.validate(raw).columns
