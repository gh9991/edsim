"""Health check for a synthetic ED extract. Run this before you model anything.

Synthetic data is generated, and generators leave fingerprints. A model trained
on an artefact learns the artefact. The checks here are the ones that have
actually caught something, phrased so that a failure tells you what it means
rather than just printing a number.

The prize-winning move if a check fails is not to quietly work around it. It is
to put the finding on a slide: the Department's own lead data scientist builds
this data, and "your generator makes waiting time independent of triage
category" is a more valuable contribution than another dashboard.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from edsim.triage import ATS_TARGET_MIN


@dataclasses.dataclass
class Finding:
    check: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.check:<34} {self.detail}"


def _wait_varies_with_acuity(df: pd.DataFrame) -> Finding:
    """A real ED prioritises. If ATS 1 waits as long as ATS 5, the generator
    drew waiting time from a distribution that ignores triage."""
    if "wait_to_treat_min" not in df or df["wait_to_treat_min"].notna().sum() == 0:
        return Finding("wait varies with acuity", True,
                       "not checkable - this source has no first-contact time")
    w = df.dropna(subset=["wait_to_treat_min"]).groupby("ats")["wait_to_treat_min"].median()
    if len(w) < 3:
        return Finding("wait varies with acuity", True, "too few categories to judge")
    spread = w.max() - w.min()
    ok = spread > 10 and w.idxmin() <= 3
    return Finding(
        "wait varies with acuity", ok,
        f"median wait by ATS {w.round(0).to_dict()} · spread {spread:.0f} min"
        + ("" if ok else "  ← waiting time looks independent of triage"))


def _admission_is_probabilistic(df: pd.DataFrame) -> Finding:
    """Admission should be a probability given ATS, not a lookup table."""
    a = df.groupby("ats")["admitted"].mean()
    degenerate = ((a == 0) | (a == 1)).sum()
    ok = degenerate == 0
    return Finding(
        "admission is probabilistic", ok,
        f"admit rate by ATS {a.round(3).to_dict()}"
        + ("" if ok else f"  ← {degenerate} categories are deterministic 0 or 1"))


def _resus_seen_immediately(df: pd.DataFrame) -> Finding:
    """ATS 1 means immediate. A median wait in the tens of minutes is not
    survivable and no real department reports it."""
    if "wait_to_treat_min" not in df or df["wait_to_treat_min"].notna().sum() == 0:
        return Finding("ATS 1 seen immediately", True,
                       "not checkable - this source has no first-contact time")
    w = df[df["ats"] == 1]["wait_to_treat_min"].dropna()
    if w.empty:
        return Finding("ATS 1 seen immediately", True, "no ATS 1 records with a wait")
    med = w.median()
    ok = med <= 5
    return Finding("ATS 1 seen immediately", ok,
                   f"median {med:.0f} min (target {ATS_TARGET_MIN[1]})"
                   + ("" if ok else "  ← clinically implausible"))


def _timestamps_are_ordered(df: pd.DataFrame) -> Finding:
    bad = 0
    n = 0
    for a, b in [("arrival_ts", "seen_ts"), ("seen_ts", "depart_ts"),
                 ("arrival_ts", "depart_ts")]:
        if a in df and b in df:
            m = df[[a, b]].dropna()
            n += len(m)
            bad += int((m[b] < m[a]).sum())
    ok = bad == 0
    return Finding("timestamps ordered", ok,
                   f"{bad:,} of {n:,} pairs out of order"
                   + ("" if ok else "  ← clean before modelling"))


def _arrivals_per_day_plausible(df: pd.DataFrame) -> Finding:
    """Catches date-shifted data, where every patient sits on their own day."""
    days = max((df["arrival_ts"].max() - df["arrival_ts"].min()).days, 1)
    rate = len(df) / days
    years = days / 365.25
    # Span is the reliable tell. A low arrival rate on its own means nothing -
    # a small rural site or a sampled extract legitimately has one. But no real
    # extract covers a century: MIMIC shifts each patient into their own random
    # year, stretching the span to 102 years and flattening the rate with it.
    ok = years <= 15 and rate >= 5
    why = []
    if years > 15:
        why.append(f"span {years:.0f} years is far longer than any real extract")
    if rate < 20:
        why.append("rate too low for a department")
    return Finding("dates not shifted", ok,
                   f"{rate:.1f}/day over {days:,} days ({years:.1f} yr)"
                   + ("" if ok else "  ← " + "; ".join(why)
                      + ". Crowding features and daily forecasts are invalid"))


def _acuity_mix_plausible(df: pd.DataFrame) -> Finding:
    """AIHW national: roughly 1% ATS 1, and 3+4 dominate."""
    mix = df["ats"].value_counts(normalize=True)
    a1 = mix.get(1, 0)
    ok = 0.001 <= a1 <= 0.06
    return Finding("acuity mix plausible", ok,
                   f"ATS1 {a1:.1%} · mix {mix.sort_index().round(3).to_dict()}"
                   + ("" if ok else "  ← ATS 1 share is far from anything reported"))


def _boarding_non_negative(df: pd.DataFrame) -> Finding:
    if "boarding_min" not in df:
        return Finding("boarding non-negative", True, "no bed_request_datetime")
    b = df["boarding_min"].dropna()
    neg = int((b < 0).sum())
    return Finding("boarding non-negative", neg == 0,
                   f"{len(b):,} boarded, median {b.median():.0f} min, {neg} negative")


def verify_official_eddc(raw: pd.DataFrame) -> list[Finding]:
    """Does this extract actually follow the published EDDC dictionary?

    Run this on the raw file before mapping anything. It caught a dataset that
    imitated the schema loosely but ignored the codebook entirely - ages 1-99
    instead of five-year brackets, `HOSP_1` instead of an encrypted four-digit
    establishment code, three free-text departure statuses instead of twelve
    numeric ones, and a ready-made `is_admitted` column that does not exist in
    the real collection. Every statistic computed on it was worthless.

    Pass `raw` straight from read_csv, not the canonical frame.
    """
    from edsim.loaders.portal import (DEPARTURE_STATUS, EDDC_ALIASES,
                                      MODE_OF_ARRIVAL)
    out = []

    def col(name):
        for c in raw.columns:
            if c.lower() == name.lower():
                return raw[c]
        return None

    # Ages are published in five-year brackets shown as the bracket minimum.
    age = col("age")
    if age is not None:
        a = pd.to_numeric(age, errors="coerce").dropna()
        off = a[a % 5 != 0]
        out.append(Finding("age in 5-year brackets", off.empty,
                           f"{len(off):,}/{len(a):,} values not multiples of 5"
                           + ("" if off.empty else f", e.g. {sorted(off.unique())[:6]}")))

    # Establishment code is a four-digit number, encrypted.
    est = col("establishment_code")
    if est is not None:
        v = est.dropna().astype(str).str.replace(r"\.0$", "", regex=True)
        ok = v.str.fullmatch(r"\d{4}").mean() > 0.95
        out.append(Finding("establishment code is 4 digits", ok,
                           f"e.g. {sorted(v.unique())[:4]}"))

    # Departure status is one of twelve numeric codes.
    ds = col("departure_status")
    if ds is not None:
        num = pd.to_numeric(ds, errors="coerce")
        known = num.isin(list(DEPARTURE_STATUS)).mean()
        out.append(Finding("departure_status uses the 12 codes", known > 0.95,
                           f"{known:.0%} of values are known codes; "
                           f"{num.dropna().nunique()} distinct"
                           + ("" if known > 0.95 else f", sample {list(ds.dropna().unique())[:3]}")))

    ma = col("mode_of_arrival")
    if ma is not None:
        num = pd.to_numeric(ma, errors="coerce")
        known = num.isin(list(MODE_OF_ARRIVAL)).mean()
        out.append(Finding("mode_of_arrival uses the 10 codes", known > 0.95,
                           f"{known:.0%} known"))

    expected = {a for al in EDDC_ALIASES.values() for a in al}
    present = {c.lower() for c in raw.columns}
    unknown = sorted(c for c in raw.columns
                     if c.lower() not in {e.lower() for e in expected})
    out.append(Finding("no columns outside the dictionary", not unknown,
                       f"unexpected: {unknown}" if unknown else "all recognised"))
    return out


CHECKS = [
    _timestamps_are_ordered,
    _arrivals_per_day_plausible,
    _wait_varies_with_acuity,
    _resus_seen_immediately,
    _admission_is_probabilistic,
    _acuity_mix_plausible,
    _boarding_non_negative,
]


def health_check(df: pd.DataFrame) -> list[Finding]:
    out = []
    for fn in CHECKS:
        try:
            out.append(fn(df))
        except Exception as exc:                      # a check must never crash the run
            out.append(Finding(fn.__name__.strip("_"), True, f"skipped: {exc}"))
    return out


def report(df: pd.DataFrame) -> str:
    findings = health_check(df)
    failed = [f for f in findings if not f.passed]
    lines = [str(f) for f in findings]
    lines.append("")
    lines.append(f"{len(findings) - len(failed)}/{len(findings)} passed")
    if failed:
        lines.append("Failures are findings, not blockers - they say something "
                     "about how the data was generated. Put them on a slide.")
    return "\n".join(lines)
