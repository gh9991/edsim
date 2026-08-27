import numpy as np
import pandas as pd

from edsim import featurescan


def _frame(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    ats = rng.choice([1, 2, 3, 4, 5], n, p=[.01, .17, .36, .38, .08])
    p = pd.Series({1: .65, 2: .43, 3: .29, 4: .13, 5: .03}).reindex(ats).to_numpy()
    y = rng.random(n) < p
    df = pd.DataFrame({
        "triage_category": ats,
        "sex": rng.integers(1, 3, n),
        "noise": rng.normal(size=n),
        "needs_bed": y,
    })
    df["departure_status"] = np.where(y, 1, 2)          # wide leak: the answer
    narrow = (rng.random(n) < 0.05) & y                  # narrow leak: fires rarely
    df["mental_health_admission"] = narrow.astype(int)
    return df


def test_finds_the_obvious_leak():
    s = featurescan.scan(_frame(), min_support=40).set_index("field")
    assert s.loc["departure_status", "auc"] > 0.95
    assert s.loc["departure_status", "verdict"].startswith("\U0001F534")


def test_finds_the_narrow_leak_that_auc_alone_would_miss():
    """Why this module exists. mental_health_admission's own definition says
    'based on departure status of admitted', yet its AUC is barely above
    chance because it covers a few percent of rows."""
    s = featurescan.scan(_frame(), min_support=40).set_index("field")
    r = s.loc["mental_health_admission"]
    assert r.auc < 0.60, "the point is that AUC does not catch this"
    assert r.lift > 2.5
    assert r.coverage < 0.10
    assert r.verdict.startswith("\U0001F534")


def test_timing_separates_a_leak_from_a_rare_genuine_predictor():
    """ATS 1 and mental_health_admission have near-identical statistical
    signatures - narrow coverage, high lift. Only timing tells them apart."""
    s = featurescan.scan(_frame(), min_support=40).set_index("field")
    ats, leak = s.loc["triage_category"], s.loc["mental_health_admission"]
    assert ats.lift > 2.0 and leak.lift > 2.0
    assert ats.verdict.startswith("\U0001F7E2")
    assert leak.verdict.startswith("\U0001F534")


def test_unknown_fields_are_flagged_not_assumed_safe():
    s = featurescan.scan(_frame(), min_support=40).set_index("field")
    assert s.loc["noise", "timing"] == "unknown"
    assert s.loc["noise", "verdict"].startswith("⚪")


def test_auc_is_out_of_fold():
    """A unique-per-row column encoded in-fold would score near 1.
    Out of fold it must sit near chance."""
    df = _frame()
    df["random_id"] = np.arange(len(df))
    s = featurescan.scan(df, min_support=40).set_index("field")
    assert s.loc["random_id", "auc"] < 0.60


def test_pairs_find_an_interaction():
    out = featurescan.scan_pairs(_frame()[["triage_category", "sex", "needs_bed"]],
                                 top=5, min_support=40)
    assert len(out) >= 1
    assert (out.lift >= 1).all()
    assert out.n.min() >= 40
