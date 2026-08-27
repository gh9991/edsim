"""Each check has to fire in both directions.

A check that always reported MISSING would tell us nothing, so every test below
comes in a pair: one fixture where the relationship really exists and the check
must find it, one where it does not and the check must say so. Each fixture
seeds its own generator - a shared one makes the results depend on test order.
"""
import numpy as np
import pandas as pd

from edsim import artefacts

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2022-01-01")
SITES = ("A", "B")

# how long each triage category waits when priority is real - roughly the
# shape of a department that streams properly
PRIORITY_SCALE = {1: 1, 2: 5, 3: 20, 4: 45, 5: 70}


def _ed(n=20_000, *, prioritised=False, congested=False, clustered=False, seed=0):
    """An ED extract carrying only the relationships asked for.

    Arrival rate varies by day and by site, so time-of-day and hospital are
    confounded in *every* fixture. That matters: it means the queueing check
    has something to strip away even when there is no queueing underneath.
    """
    rng = np.random.default_rng(seed)
    n_index = n if not clustered else int(n * 0.7)

    day = rng.integers(0, 364, n_index)
    site = rng.choice(list(SITES), n_index)
    hour = rng.integers(0, 24, n_index)
    pid = rng.integers(0, max(n // 3, 1), n_index)
    arrival = (T0 + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h")
               + pd.to_timedelta(rng.integers(0, 60, n_index), "m"))

    if clustered:   # a return visit a few days after some of the index ones
        k = n - n_index
        back = rng.choice(n_index, k, replace=False)
        follow = arrival[back] + pd.to_timedelta(rng.exponential(1.5, k), "D")
        arrival = arrival.append(follow)
        pid = np.concatenate([pid, pid[back]])
        site = np.concatenate([site, site[back]])

    m = len(arrival)
    ats = rng.choice([1, 2, 3, 4, 5], m, p=[.01, .12, .40, .37, .10])
    scale = (np.vectorize(PRIORITY_SCALE.get)(ats) if prioritised
             else np.full(m, 35.0))
    wait = rng.exponential(scale)
    los = rng.exponential(180, m)

    df = pd.DataFrame({
        "encounter_id": np.arange(m), "patient_id": pid, "arrival_ts": arrival,
        "ats": ats, "acuity": ats, "wait_to_treat_min": wait, "site": site,
        "los": los,
    })
    if congested:
        # a real queue: how long you wait depends on how many people are
        # already in the department, which varies day to day within the same
        # hour at the same site - exactly the variation the check looks for.
        # Crowding falls on the low-acuity end; a resus patient is seen at once
        # however full the place is, which is why priority survives congestion.
        df = df.sort_values(["site", "arrival_ts"]).reset_index(drop=True)
        occ = np.zeros(len(df))
        for _, g in df.groupby("site"):
            t = g.arrival_ts.values.astype("datetime64[m]").astype(float)
            out = t + g.los.values
            occ[g.index] = (np.searchsorted(np.sort(t), t, "right")
                            - np.searchsorted(np.sort(out), t, "left"))
        df["wait_to_treat_min"] = df.wait_to_treat_min + 1.5 * occ * (df.ats - 2).clip(lower=0)
    df["depart_ts"] = df.arrival_ts + pd.to_timedelta(
        df.wait_to_treat_min + df.los, "m")
    return df.drop(columns="los")


def _hm(ed, *, linked=False, sequenced=False, repeats=False, seed=1):
    """One admission per patient, either following their ED visit or not."""
    rng = np.random.default_rng(seed)
    one = ed.drop_duplicates("patient_id")
    n = len(one)
    adm = (one.depart_ts + pd.to_timedelta(rng.exponential(2, n), "h") if linked
           else T0 + pd.to_timedelta(rng.integers(0, 364, n), "D"))
    rows = pd.DataFrame({"patient_id": one.patient_id.values,
                         "admission_ts": pd.Series(adm).values,
                         "care_type": rng.choice([21, 22], n, p=[.9, .1]),
                         "admission_status": 6})
    rows["separation_ts"] = rows.admission_ts + pd.to_timedelta(
        rng.exponential(4, n), "D")
    if repeats:     # readmitted months later, keeping the person's own type -
        again = rows.sample(frac=0.4, random_state=2).copy()   # not a handover
        again["admission_ts"] += pd.to_timedelta(rng.integers(30, 200, len(again)), "D")
        again["separation_ts"] = again.admission_ts + pd.to_timedelta(
            rng.exponential(4, len(again)), "D")
        rows = pd.concat([rows, again], ignore_index=True)
    if sequenced:   # acute closes and rehab opens ten minutes later
        moves = rows[rows.care_type == 21].sample(frac=0.3, random_state=1).copy()
        moves["admission_ts"] = moves.separation_ts + pd.Timedelta(minutes=10)
        moves["separation_ts"] = moves.admission_ts + pd.to_timedelta(
            rng.exponential(10, len(moves)), "D")
        moves["care_type"] = 22
        rows = pd.concat([rows, moves], ignore_index=True)
    return rows


# --- 1 priority ------------------------------------------------------------

def test_priority_found_when_categories_have_their_own_waits():
    assert artefacts.check_priority(_ed(prioritised=True)).preserved


def test_priority_missing_when_one_pooled_distribution():
    assert not artefacts.check_priority(_ed(prioritised=False, congested=False,
                                            clustered=False)).preserved


# --- 2 queueing ------------------------------------------------------------

def test_queueing_found_when_occupancy_drives_the_wait():
    assert artefacts.check_queueing(_ed(congested=True)).preserved


def test_queueing_missing_when_waits_ignore_occupancy():
    r = artefacts.check_queueing(_ed())
    assert not r.preserved
    assert "hour&hospital" in r.statistic


def test_queueing_strips_the_confounder_not_the_signal():
    """Both fixtures share the site/hour structure; only one has real queueing,
    so a check that merely detected grouping would pass both."""
    congested = _ed(congested=True)
    flat = _ed()
    assert artefacts.check_queueing(congested).preserved
    assert not artefacts.check_queueing(flat).preserved


# --- 3 event linkage -------------------------------------------------------

def test_linkage_found_when_admission_follows_the_visit():
    ed = _ed()
    r = artefacts.check_event_linkage(ed, _hm(ed, linked=True, sequenced=False))
    assert r.preserved
    assert "0.0%" in r.statistic or "before" in r.statistic.lower()


def test_linkage_missing_when_dates_are_drawn_independently():
    ed = _ed()
    assert not artefacts.check_event_linkage(
        ed, _hm(ed, linked=False, sequenced=False)).preserved


def test_linkage_uses_only_unambiguous_one_to_one_pairs():
    """People with several visits are excluded, so the pairing never guesses."""
    ed = _ed()
    hm = _hm(ed, linked=True, sequenced=False)
    used = int(artefacts.check_event_linkage(ed, hm).statistic
               .split("over ")[1].split(" ")[0].replace(",", ""))
    assert used == (ed.groupby("patient_id").size() == 1).sum()


# --- 4 sequence ------------------------------------------------------------

def test_sequence_found_when_acute_hands_over_to_rehab():
    ed = _ed()
    assert artefacts.check_sequence(_hm(ed, linked=True, sequenced=True)).preserved


def test_sequence_missing_when_care_type_is_just_a_person_attribute():
    """Repeat admissions exist, but nothing hands over from one to the next."""
    ed = _ed()
    assert not artefacts.check_sequence(
        _hm(ed, linked=True, repeats=True)).preserved


def test_sequence_abstains_when_nobody_is_admitted_twice():
    """With one admission each there is no sequence to judge either way."""
    ed = _ed()
    r = artefacts.check_sequence(_hm(ed, linked=True))
    assert r.preserved and "not checkable" in r.statistic


def test_sequence_ignores_same_type_persistence():
    """A patient readmitted to rehab twice is a rehab patient, not a stay that
    progressed - repeating one type must not count as a transition."""
    rng = np.random.default_rng(7)
    n = 4000
    pid = np.repeat(np.arange(n // 2), 2)
    adm = T0 + pd.to_timedelta(np.tile([0, 30], n // 2) + rng.integers(0, 200, n), "D")
    hm = pd.DataFrame({"patient_id": pid, "admission_ts": adm, "care_type": 22,
                       "separation_ts": adm + pd.Timedelta(minutes=1)})
    assert not artefacts.check_sequence(hm).preserved


# --- 5 revisit -------------------------------------------------------------

def test_revisit_found_when_return_visits_cluster():
    assert artefacts.check_revisit(_ed(clustered=True)).preserved


def test_revisit_missing_when_visits_are_memoryless():
    assert not artefacts.check_revisit(_ed()).preserved


def test_revisit_is_independent_of_the_observation_window():
    """The raw 72-hour rate rises when the extract covers less time; CV must
    not, or 2025 and 2026 extracts of different lengths cannot be compared."""
    ed = _ed(n=30_000, clustered=True)
    half = ed[ed.arrival_ts < T0 + 182 * DAY]
    full = float(artefacts.check_revisit(ed).statistic.split("CV ")[1][:4])
    part = float(artefacts.check_revisit(half).statistic.split("CV ")[1][:4])
    assert abs(full - part) < 0.35


# --- the suite as a whole --------------------------------------------------

def test_report_runs_without_hmdc():
    """Without admissions data, checks 3 and 4 are skipped rather than failed -
    an extract we cannot test is not the same as one that fails the test."""
    ed = _ed(prioritised=True, congested=True, clustered=True)
    numbered = [a.n for a in artefacts.run_all(ed, None)]
    assert numbered == [1, 2, 5]
    assert artefacts.report(ed, None).endswith("3/3 preserved")


def test_report_orders_by_number_and_counts_survivors():
    ed = _ed(prioritised=True, congested=True, clustered=True)
    res = artefacts.run_all(ed, _hm(ed, linked=True, sequenced=True))
    assert [a.n for a in res] == [1, 2, 3, 4, 5]
    assert artefacts.report(ed, _hm(ed, linked=True, sequenced=True)).endswith(
        "5/5 preserved")


def test_checks_abstain_rather_than_guess_on_tiny_inputs():
    """Too little data must read 'not checkable', never a false MISSING."""
    tiny = _ed(n=300, prioritised=False, congested=False, clustered=False)
    for a in artefacts.run_all(tiny, None):
        assert a.preserved and "not checkable" in a.statistic
