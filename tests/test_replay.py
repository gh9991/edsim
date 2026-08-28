"""The replay engine has to be right about the arithmetic before it is useful
about anything else, so most of these check it against a queue small enough to
work out by hand."""
import numpy as np
import pandas as pd
import pytest

from edsim.replay import (occupancy_at_arrival, offered_load, replay, sweep)


def test_matches_a_queue_worked_out_by_hand():
    """Three beds, five patients. Patients 4 and 5 arrive to a full department
    and wait for the first space to free, which is patient 3's at t=50 and
    patient 1's at t=60."""
    arr = np.array([0, 5, 10, 12, 20.0])
    svc = np.array([60, 90, 40, 30, 25.0])
    got = replay(arr, svc, np.ones(5), beds=3)
    assert got.tolist() == [0, 0, 0, 38, 40]


def test_nobody_waits_when_there_are_beds_to_spare():
    arr = np.arange(0, 500, 10.0)
    got = replay(arr, np.full(50, 5.0), np.ones(50), beds=50)
    assert (got == 0).all()


def test_one_bed_serialises_everyone():
    arr = np.zeros(4)
    svc = np.array([10, 10, 10, 10.0])
    got = replay(arr, svc, np.ones(4), beds=1)
    assert sorted(got.tolist()) == [0, 10, 20, 30]


def test_better_priority_jumps_the_queue():
    """One bed, busy until t=100. Two people wait; the sicker one is seated
    first even though they arrived ten minutes later.

    Compare seating times, not waits - the later arrival records a shorter wait
    whichever order they are seated in, so waits cannot show who went first."""
    arr = np.array([0, 10, 20.0])
    svc = np.array([100, 5, 5.0])
    seated = arr + replay(arr, svc, np.array([3, 3, 1.0]), beds=1)
    assert seated[2] < seated[1]                 # ATS 1 before ATS 3
    assert seated.tolist() == [0, 105, 100]


def test_ties_on_priority_go_to_whoever_arrived_first():
    arr = np.array([0, 10, 20.0])
    svc = np.array([100, 5, 5.0])
    seated = arr + replay(arr, svc, np.array([3, 3, 3.0]), beds=1)
    assert seated[1] < seated[2]
    assert seated.tolist() == [0, 100, 105]


def test_result_is_aligned_to_input_order_not_arrival_order():
    """Shuffling the rows must permute the answers and nothing else - getting
    this wrong is the single easiest way to produce a plausible lie."""
    rng = np.random.default_rng(0)
    arr = rng.random(400) * 1000
    svc = rng.exponential(30, 400)
    pri = rng.integers(1, 6, 400).astype(float)
    straight = replay(arr, svc, pri, beds=4)
    o = rng.permutation(400)
    shuffled = replay(arr[o], svc[o], pri[o], beds=4)
    assert np.allclose(shuffled, straight[o])


def test_fewer_beds_never_shortens_anyone_wait():
    rng = np.random.default_rng(1)
    arr = np.sort(rng.random(600) * 2000)
    svc = rng.exponential(40, 600)
    pri = np.full(600, 3.0)
    assert replay(arr, svc, pri, 3).sum() > replay(arr, svc, pri, 6).sum()


def test_waiting_appears_only_once_capacity_bites():
    """Below the offered load the queue never clears; well above it nobody
    waits. The interesting behaviour is the transition between."""
    rng = np.random.default_rng(2)
    arr = np.sort(rng.random(3000) * 10_000)
    svc = rng.exponential(50, 3000)
    load = offered_load(svc, arr)
    assert np.median(replay(arr, svc, np.ones(3000), int(load * 3))) == 0
    assert np.median(replay(arr, svc, np.ones(3000), max(int(load), 1))) > 0


def test_replay_creates_the_relationship_the_extract_lacks():
    """The whole point: waits that come out of a queue correlate with how full
    the department was, and independently drawn ones do not."""
    rng = np.random.default_rng(3)
    arr = np.sort(rng.random(4000) * 20_000)
    svc = rng.exponential(60, 4000)
    load = offered_load(svc, arr)
    w = replay(arr, svc, np.ones(4000), int(load * 1.15))
    drawn = rng.exponential(np.mean(w) + 1, 4000)      # same scale, no queue
    r_queue = np.corrcoef(occupancy_at_arrival(arr, w, svc), w)[0, 1]
    r_drawn = np.corrcoef(occupancy_at_arrival(arr, drawn, svc), drawn)[0, 1]
    assert r_queue > 0.3
    assert abs(r_drawn) < 0.1


def test_rejects_mismatched_and_empty_inputs():
    with pytest.raises(ValueError):
        replay([0, 1], [1], [1, 1], beds=1)
    with pytest.raises(ValueError):
        replay([0.0], [1.0], [1.0], beds=0)
    assert len(replay([], [], [], beds=1)) == 0


def _frame(n=4000, seed=4):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-01-01")
    arr = t0 + pd.to_timedelta(np.sort(rng.random(n) * 200_000), "m")
    wait = rng.exponential(30, n)
    svc = rng.exponential(90, n)
    return pd.DataFrame({
        "arrival_ts": arr,
        "seen_ts": arr + pd.to_timedelta(wait, "m"),
        "depart_ts": arr + pd.to_timedelta(wait + svc, "m"),
        "ats": rng.integers(1, 6, n).astype(float)})


def test_sweep_reports_recorded_alongside_recomputed():
    out = sweep(_frame())
    assert "as recorded" in out.index
    assert out.attrs["offered_load"] > 0
    assert out.loc["as recorded"].isna()[["beds", "utilisation"]].all()


def test_sweep_utilisation_falls_as_beds_rise():
    out = sweep(_frame()).drop("as recorded")
    assert out.utilisation.is_monotonic_decreasing
    assert out.median_min.is_monotonic_decreasing


def test_sweep_refuses_too_little_data():
    with pytest.raises(ValueError, match="need at least"):
        sweep(_frame(n=200))
