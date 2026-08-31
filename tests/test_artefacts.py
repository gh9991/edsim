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


def _ed(n=20_000, *, prioritised=False, congested=False, clustered=False,
        blocked=False, seed=0):
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

    # a third of visits ask for a ward bed once treatment is done, and only
    # then start waiting - the request comes first and the wait delays leaving,
    # so build it in that order or the two decorrelate
    asks = rng.random(len(df)) < 0.35
    req = df.depart_ts
    board = rng.exponential(90, len(df))
    if blocked:
        # wards fill and empty through the day, so everyone whose bed is
        # requested while they are full waits together - access block belongs
        # to the hospital at a moment, not to the patient
        block = (req.dt.floor("4h").astype("int64") // (10 ** 9 * 14400)
                 + df.site.map({s: i * 7919 for i, s in enumerate(SITES)}))
        pressure = pd.Series(block).map(
            dict(zip(block.unique(),
                     rng.random(block.nunique()) ** 2))).values
        board = board * (0.2 + 8 * pressure)
    df["bed_request_ts"] = req.where(asks)
    df["depart_ts"] = req + pd.to_timedelta(np.where(asks, board, 0), "m")
    return df.drop(columns="los")


def _hm(ed, *, linked=False, sequenced=False, repeats=False, pressured=False,
        seed=1):
    """One admission per patient, either following their ED visit or not."""
    rng = np.random.default_rng(seed)
    one = ed.drop_duplicates("patient_id")
    n = len(one)
    adm = (one.depart_ts + pd.to_timedelta(rng.exponential(2, n), "h") if linked
           else T0 + pd.to_timedelta(rng.integers(0, 364, n), "D"))
    rows = pd.DataFrame({"patient_id": one.patient_id.values,
                         "admission_ts": pd.Series(adm).values,
                         "site": one.site.values,
                         "care_type": rng.choice([21, 22], n, p=[.9, .1]),
                         "admission_status": 6})
    los = rng.exponential(4, n)
    if pressured:      # a fuller ward holds its patients longer
        a = rows.admission_ts.values.astype("datetime64[h]").astype(float)
        occ = np.zeros(n)
        for _, g in rows.groupby("site"):
            aa = a[g.index]
            occ[g.index] = (np.searchsorted(np.sort(aa), aa, "right")
                            - np.searchsorted(np.sort(aa + los[g.index] * 24), aa, "left"))
        los = np.clip(los * (1 + 0.03 * (occ - occ.mean())), 0.1, None)
    rows["separation_ts"] = rows.admission_ts + pd.to_timedelta(los, "D")
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


# --- 6 boarding -----------------------------------------------------------

def test_boarding_found_when_the_ward_blocks_for_everyone_at_once():
    r = artefacts.check_boarding(_ed(n=40_000, blocked=True))
    assert r.preserved
    assert "median boarding" in r.statistic


def test_boarding_missing_when_each_wait_is_drawn_alone():
    assert not artefacts.check_boarding(_ed(n=40_000)).preserved


def test_boarding_abstains_without_a_bed_request_timestamp():
    ed = _ed(n=40_000, blocked=True).drop(columns="bed_request_ts")
    r = artefacts.check_boarding(ed)
    assert r.preserved and "not checkable" in r.statistic


def test_boarding_is_a_different_question_from_queueing():
    """Check 2 is the wait to be seen, check 6 the wait for a bed. A dataset
    can keep one and lose the other, so neither may stand in for the other."""
    ed = _ed(n=40_000, congested=True, blocked=False)
    assert artefacts.check_queueing(ed).preserved
    assert not artefacts.check_boarding(ed).preserved


# --- the suite as a whole --------------------------------------------------

def test_report_runs_without_hmdc():
    """Without admissions data, checks 3 and 4 are skipped rather than failed -
    an extract we cannot test is not the same as one that fails the test."""
    ed = _ed(n=40_000, prioritised=True, congested=True, clustered=True,
             blocked=True)
    numbered = [a.n for a in artefacts.run_all(ed, None)]
    assert numbered == [1, 2, 5, 6, 8]


def test_report_orders_by_number_and_counts_survivors():
    ed = _ed(n=40_000, prioritised=True, congested=True, clustered=True,
             blocked=True)
    hm = _hm(ed, linked=True, sequenced=True, pressured=True)
    assert [a.n for a in artefacts.run_all(ed, hm)] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # checks 8 and 9 abstain here - this fixture carries neither a diagnosis
    # chapter nor a patient region - and an abstention counts as preserved
    assert artefacts.report(ed, hm).endswith("9/9 preserved")


def test_checks_abstain_rather_than_guess_on_tiny_inputs():
    """Too little data must read 'not checkable', never a false MISSING."""
    tiny = _ed(n=300, prioritised=False, congested=False, clustered=False)
    for a in artefacts.run_all(tiny, None):
        assert a.preserved and "not checkable" in a.statistic


# --- 7 ward pressure -------------------------------------------------------

def _ward(n=30_000, *, pressured=False, seed=5):
    """Admissions at two hospitals, with or without occupancy affecting stay."""
    rng = np.random.default_rng(seed)
    site = rng.choice(list(SITES), n)
    adm = (T0 + pd.to_timedelta(rng.integers(0, 364, n), "D")
           + pd.to_timedelta(rng.integers(0, 24, n), "h"))
    # an elective list pulls weekend admissions back onto the Friday; derive
    # the day from the timestamp rather than arithmetic on the offset, since
    # 1 January 2022 was a Saturday and day % 7 does not mean what it looks like
    move = (adm.dayofweek >= 5) & (rng.random(n) < 0.55)
    back = np.clip(adm.dayofweek.to_numpy() - 4, 0, None)
    adm = adm - pd.to_timedelta(np.where(move, back, 0), "D")
    los = rng.exponential(4, n)
    df = pd.DataFrame({"patient_id": np.arange(n), "admission_ts": adm,
                       "site": site, "los": los})
    if pressured:
        df = df.sort_values(["site", "admission_ts"]).reset_index(drop=True)
        occ = np.zeros(len(df))
        for _, g in df.groupby("site"):
            a = g.admission_ts.values.astype("datetime64[h]").astype(float)
            out = a + g.los.values * 24
            occ[g.index] = (np.searchsorted(np.sort(a), a, "right")
                            - np.searchsorted(np.sort(out), a, "left"))
        df["los"] = df.los * (1 + 0.02 * (occ - occ.mean()))   # fuller -> longer
        df["los"] = df.los.clip(lower=0.1)
    df["separation_ts"] = df.admission_ts + pd.to_timedelta(df.los, "D")
    return df.drop(columns="los")


def test_ward_pressure_found_when_a_full_hospital_holds_patients():
    r = artefacts.check_ward_pressure(_ward(pressured=True))
    assert r.preserved
    assert "weekday/weekend" in r.statistic


def test_ward_pressure_missing_when_stays_ignore_the_hospital():
    assert not artefacts.check_ward_pressure(_ward()).preserved


def test_ward_pressure_abstains_without_timestamps():
    w = _ward().drop(columns="separation_ts")
    r = artefacts.check_ward_pressure(w)
    assert r.preserved and "not checkable" in r.statistic


def test_ward_pressure_reports_the_weekday_rhythm_separately():
    """A weekday bias is a property of one row - a generator can reproduce the
    calendar and still miss the physics, so the two are reported side by side
    and only the correlation decides the verdict."""
    r = artefacts.check_ward_pressure(_ward())
    weekday = float(r.statistic.split("admissions ")[1].split("x")[0])
    assert weekday > 1.1                      # calendar rhythm present
    assert not r.preserved                    # physics still absent


# --- 8 clock, not count ----------------------------------------------------

CHAPTERS = ["S0", "R0", "Z0", "J0", "I0", "F0", "A0", "K0", "M0", "N0"]
CH_P = np.array([.28, .19, .10, .09, .06, .06, .06, .06, .05, .05])


def _three(n=12_000, *, drifts_with_time=False, drifts_with_position=False, seed=9):
    """Three visits each. Concordance can be made to depend on the gap between
    two visits, on which visit it is, both, or neither."""
    rng = np.random.default_rng(seed)
    theta = rng.choice(len(CHAPTERS), n, p=CH_P)
    rows = []
    t = T0 + pd.to_timedelta(rng.integers(0, 120, n), "D")
    for pos in range(3):
        gap = rng.exponential(90, n) + 1
        t = t + pd.to_timedelta(gap, "D")
        follow = np.full(n, 0.35)
        if drifts_with_time:                 # close together -> same problem
            follow = np.clip(0.75 - gap / 220, 0.10, 0.75)
        if drifts_with_position and pos == 0:
            follow = follow * 0.35           # the opening visit follows less
        use = rng.random(n) < follow
        ch = np.where(use, theta, rng.choice(len(CHAPTERS), n, p=CH_P))
        rows.append(pd.DataFrame({"patient_id": np.arange(n), "arrival_ts": t,
                                  "diagnosis_chapter": [CHAPTERS[i] for i in ch]}))
    return pd.concat(rows, ignore_index=True)


def test_clock_found_when_agreement_falls_with_the_gap_alone():
    r = artefacts.check_clock_not_count(_three(drifts_with_time=True))
    assert r.preserved


def test_clock_missing_when_visit_position_moves_agreement():
    """The body has no idea which visit this is - if position matters once the
    gap is held fixed, something other than physiology is deciding."""
    r = artefacts.check_clock_not_count(
        _three(drifts_with_time=True, drifts_with_position=True))
    assert not r.preserved
    assert "visit position" in r.statistic


def test_clock_missing_when_time_does_not_move_agreement():
    assert not artefacts.check_clock_not_count(_three()).preserved


def test_clock_abstains_without_a_diagnosis_chapter():
    r = artefacts.check_clock_not_count(
        _three(drifts_with_time=True).drop(columns="diagnosis_chapter"))
    assert r.preserved and "not checkable" in r.statistic


def test_clock_ignores_rows_coded_missing():
    """'Missing' is a data-completeness pattern concentrated in first visits;
    counting it as a chapter would manufacture a positional effect."""
    d = _three(drifts_with_time=True)
    first = d.sort_values(["patient_id", "arrival_ts"]).groupby("patient_id").head(1).index
    d.loc[first[:3000], "diagnosis_chapter"] = "Missing"
    assert artefacts.check_clock_not_count(d).preserved


# --- 9 geography -----------------------------------------------------------

REGIONS = ["metro", "north", "south", "other"]


def _geo(n=20_000, *, people_live_near_care=False, seed=11):
    """Admissions carrying where the patient lives and whether the hospital is
    metropolitan. Western Australia sends many country patients to Perth, so a
    high metro share everywhere is realistic - an identical share everywhere
    is not."""
    rng = np.random.default_rng(seed)
    region = rng.choice(REGIONS, n, p=[.55, .15, .27, .03])
    if people_live_near_care:
        p_metro = pd.Series(region).map(
            {"metro": .95, "north": .45, "south": .55, "other": .60}).to_numpy()
    else:
        p_metro = np.full(n, .76)          # same everywhere - the artefact
    return pd.DataFrame({
        "patient_id": np.arange(n), "patient_region": region,
        "metro": (rng.random(n) < p_metro).astype(int),
        "admission_ts": T0 + pd.to_timedelta(rng.integers(0, 364, n), "D")})


def test_geography_found_when_where_you_live_predicts_where_you_are_treated():
    r = artefacts.check_geography(_geo(people_live_near_care=True))
    assert r.preserved
    assert "metro" in r.statistic


def test_geography_missing_when_every_region_looks_identical():
    r = artefacts.check_geography(_geo())
    assert not r.preserved
    spread = float(r.statistic.split("spread of ")[1].split(" pp")[0])
    assert spread < 5


def test_geography_tolerates_a_high_metro_share_everywhere():
    """WA really does treat many country patients in Perth. A high level is
    not the failure - a flat one is."""
    g = _geo(people_live_near_care=True, seed=12)
    assert g.metro.mean() > 0.6
    assert artefacts.check_geography(g).preserved


def test_geography_abstains_without_a_patient_region():
    r = artefacts.check_geography(_geo().drop(columns="patient_region"))
    assert r.preserved and "not checkable" in r.statistic
