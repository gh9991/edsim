"""ATS (Australia) vs ESI (US) - and why you must not silently equate them.

ATS  = Australasian Triage Scale. Defined by *maximum time to treatment*.
ESI  = Emergency Severity Index. Defined by acuity **plus predicted resource
       use** - ESI 3/4/5 separate on "how many resources", not on time.

Both run 1 (most urgent) to 5, and both are ordinal, so an identity map is
"close enough" to prototype with. It is not clinically equivalent, and the
divergence is concentrated in the 3-4-5 band. Say this out loud in the pitch:
judges from DoH/EMHS will know, and owning the limitation reads as competence.
"""
from __future__ import annotations

import pandas as pd

# ATS category -> maximum time to treatment, minutes (ACEM standard)
ATS_TARGET_MIN = {1: 0, 2: 10, 3: 30, 4: 60, 5: 120}

# ACEM performance thresholds - % of patients that should be seen within target
ATS_PERFORMANCE_TARGET = {1: 1.00, 2: 0.80, 3: 0.75, 4: 0.70, 5: 0.70}

ATS_NAME = {
    1: "Resuscitation",
    2: "Emergency",
    3: "Urgent",
    4: "Semi-urgent",
    5: "Non-urgent",
}

# AIHW MyHospitals reports triage categories by name. Casing and hyphenation
# are NOT stable across measures ("Semi-Urgent" vs "Semi-urgent"), so match on
# a normalised key - a silent .map() miss here drops ATS 4 and 5 entirely.
AIHW_TRIAGE_TO_ATS = {
    "resuscitation": 1,
    "emergency": 2,
    "urgent": 3,
    "semiurgent": 4,
    "nonurgent": 5,
}


def normalise_triage_name(name: str) -> str:
    return str(name).lower().replace("-", "").replace(" ", "").strip()


def aihw_name_to_ats(names: pd.Series) -> pd.Series:
    return names.map(lambda n: AIHW_TRIAGE_TO_ATS.get(normalise_triage_name(n)))

# Ordinal pass-through. Kept explicit so it is a decision, not an accident.
ESI_TO_ATS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

NEAT_TARGET_MIN = 240  # National Emergency Access Target: depart ED within 4h


def to_ats(acuity: pd.Series, scale: pd.Series | str) -> pd.Series:
    """Normalise any supported acuity scale onto ATS 1-5."""
    acuity = pd.to_numeric(acuity, errors="coerce")
    if isinstance(scale, str):
        scale = pd.Series([scale] * len(acuity), index=acuity.index)
    scale = scale.fillna("ATS").astype(str).str.upper()

    out = acuity.copy()
    is_esi = scale.eq("ESI")
    out.loc[is_esi] = acuity.loc[is_esi].map(ESI_TO_ATS)
    return out


def seen_on_time(wait_min: pd.Series, ats: pd.Series) -> pd.Series:
    """Boolean: treatment commenced within the ATS target for that category."""
    target = ats.map(ATS_TARGET_MIN)
    return wait_min.le(target)
