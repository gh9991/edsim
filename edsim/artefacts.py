"""Generator-artefact suite. Run this first on any new extract, in this order.

The 2025 official synthetic extract preserves what a single row looks like and
loses almost everything that happens *between* rows. Row-wise generation models
"what does this patient look like" and cannot model "what happened between this
record and that one" - which is where nearly all of emergency medicine lives:
patients competing for the same bed, one visit causing the next, one phase of
an admission following another.

Each check answers one question and, if it fails, closes off a specific set of
analyses. They are ordered by how much they constrain the project, so a failure
early tells you to stop planning work that cannot be done.

  1  priority          does triage order affect waiting time?
  2  queueing          does congestion affect waiting time, within a hospital?
  3  event_linkage     does an ED visit connect to the admission it caused?
  4  sequence          do care phases follow one another within a stay?
  5  revisit           do unresolved visits bring people back?
  6  boarding          is waiting for a ward bed a shared condition?

Every check compares against a null rather than against intuition - a pooled
distribution, a stripped-away confounder, an unambiguous subset, a marginal
rate, a Poisson process. The question is always the same: how much does this
exceed chance?
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from edsim.triage import ATS_TARGET_MIN


@dataclasses.dataclass
class Artefact:
    n: int
    name: str
    question: str
    preserved: bool
    statistic: str
    blocks: str = ""

    def __str__(self) -> str:
        mark = "PRESERVED" if self.preserved else "MISSING  "
        out = f"{self.n}. {mark}  {self.name:<14} {self.statistic}"
        if not self.preserved and self.blocks:
            out += f"\n                              blocks: {self.blocks}"
        return out


def check_priority(ed: pd.DataFrame) -> Artefact:
    """1. Does triage category change how long you wait?

    Compare each category's on-time rate against what a single pooled waiting
    distribution would produce. The ATS targets alone (0/10/30/60/120 min)
    manufacture a gradient, so a rising on-time rate proves nothing on its own.
    Small residuals mean every category was drawn from one distribution.
    """
    w = ed.dropna(subset=["wait_to_treat_min"])
    if len(w) < 1000 or w.ats.nunique() < 4:
        return Artefact(1, "priority", "does triage order affect waiting?",
                        True, "not checkable - too few records or categories")
    pooled = w.wait_to_treat_min
    res = [((w[w.ats == a].wait_to_treat_min <= ATS_TARGET_MIN[a]).mean()
            - (pooled <= ATS_TARGET_MIN[a]).mean()) * 100
           for a in sorted(w.ats.dropna().unique().astype(int)) if a in ATS_TARGET_MIN]
    mae = float(np.mean(np.abs(res)))
    return Artefact(
        1, "priority", "does triage order affect waiting?",
        preserved=mae > 15,
        statistic=f"mean |residual| vs one pooled distribution = {mae:.1f} pp "
                  f"(real WA: ATS 1 +100, ATS 2 +54)",
        blocks="wait-time modelling, ATS performance, streaming and fast-track studies")


def check_queueing(ed: pd.DataFrame) -> Artefact:
    """2. Within one hospital at one hour, does a fuller department mean a
    longer wait?

    The raw correlation is confounded by both time of day and by hospital -
    busy hospitals have both. Strip each layer and watch what survives.
    """
    need = {"wait_to_treat_min", "depart_ts", "arrival_ts", "site"}
    if not need <= set(ed.columns):
        return Artefact(2, "queueing", "does congestion lengthen the wait?",
                        True, "not checkable - missing columns")
    w = ed.dropna(subset=["wait_to_treat_min", "depart_ts", "arrival_ts"])
    if len(w) < 5000:
        return Artefact(2, "queueing", "does congestion lengthen the wait?",
                        True, "not checkable - too few records")
    w = w.reset_index(drop=True).copy()
    occ = pd.Series(index=w.index, dtype=float)
    for _, g in w.groupby("site"):                 # assign back by index, never
        st = np.sort(g.arrival_ts.values)          # by groupby iteration order
        en = np.sort(g.depart_ts.values)
        q = g.arrival_ts.values
        occ.loc[g.index] = (np.searchsorted(st, q, "right")
                            - np.searchsorted(en, q, "left"))
    w["occ"], w["hour"] = occ, w.arrival_ts.dt.hour

    def wavg(groups):
        x = [(len(g), np.corrcoef(g.occ, g.wait_to_treat_min)[0, 1])
             for _, g in groups if len(g) > 200 and g.occ.std() > 0]
        if not x:
            return float("nan")
        return float(np.average([b for _, b in x], weights=[a for a, _ in x]))

    raw = float(np.corrcoef(w.occ, w.wait_to_treat_min)[0, 1])
    by_hour = wavg(w.groupby("hour"))
    within = wavg(w.groupby(["hour", "site"]))
    return Artefact(
        2, "queueing", "does congestion lengthen the wait?",
        preserved=within > 0.05,
        statistic=f"corr(occupancy, wait): raw {raw:+.3f} -> +hour {by_hour:+.3f} "
                  f"-> +hour&hospital {within:+.3f}",
        blocks="crowding as a feature, access-block dynamics, early-bed-request counterfactuals")


def check_event_linkage(ed: pd.DataFrame, hm: pd.DataFrame) -> Artefact:
    """3. Does an ED visit connect to the admission it produced?

    Restricted to people with exactly one ED visit and one admission, so the
    pairing carries no ambiguity at all. If the two are the same episode their
    dates must correlate near +1. An admission dated before the ED visit is
    impossible for a real link, which makes direction the cleanest evidence
    available - it needs no threshold and no assumption about transfer delay.

    Reported against its own null: shuffling who is paired with whom gives the
    share of backwards pairs that pure chance produces here, so the two numbers
    can be read without knowing anything about this extract in advance. A real
    linkage sits near zero; an independently generated pair of files sits on
    the shuffled figure.
    """
    e1 = ed.groupby("patient_id").filter(lambda g: len(g) == 1)
    h1 = (hm.dropna(subset=["admission_ts"])
            .groupby("patient_id").filter(lambda g: len(g) == 1))
    both = set(e1.patient_id) & set(h1.patient_id)
    if len(both) < 500:
        return Artefact(3, "event_linkage", "does a visit connect to its admission?",
                        True, "not checkable - too few unambiguous pairs")
    cols = ["admission_ts"] + (["admission_status"] if "admission_status" in hm else [])
    j = (e1[e1.patient_id.isin(both)].set_index("patient_id")
         .join(h1[h1.patient_id.isin(both)].set_index("patient_id")[cols], how="inner"))
    if "admission_status" in j:                    # 6 = admitted via the ED
        via = pd.to_numeric(j.admission_status, errors="coerce") == 6
        if via.sum() > 500:
            j = j[via]
    origin = pd.Timestamp(j.depart_ts.min().date())
    x = (j.depart_ts - origin).dt.total_seconds() / 86400
    y = (j.admission_ts - origin).dt.total_seconds() / 86400
    r = float(np.corrcoef(x, y)[0, 1])
    neg = float((y < x).mean())
    rng = np.random.default_rng(0)
    yv = y.values
    null = float(np.mean([(rng.permutation(yv) < x.values).mean() for _ in range(50)]))
    return Artefact(
        3, "event_linkage", "does a visit connect to its admission?",
        preserved=r > 0.5,
        statistic=f"corr(ED departure date, admission date) = {r:+.3f} over {len(j):,} "
                  f"unambiguous pairs; {neg:.1%} admitted BEFORE the ED visit "
                  f"(shuffling the pairing gives {null:.1%}; a real linkage gives ~0%)",
        blocks="ED-to-inpatient flow, transfer timing, the value of predicting bed need early")


def check_sequence(hm: pd.DataFrame) -> Artefact:
    """4. Do care phases follow one another within a stay?

    A care-type change is supposed to close one episode and open another
    minutes later - acute care becomes rehabilitation, the patient never leaves
    the building. Two signatures: consecutive rows minutes apart, and
    acute -> rehab enriched over rehab's marginal rate. Person-level
    persistence (rehab -> rehab) is a different thing and does not count: it
    says this is a rehab patient, not that this stay progressed.
    """
    if "care_type" not in hm or len(hm) < 2000:
        return Artefact(4, "sequence", "do care phases follow one another?",
                        True, "not checkable - no care_type")
    s = hm.dropna(subset=["admission_ts"]).sort_values(
        ["patient_id", "admission_ts"]).copy()
    s["ct"] = pd.to_numeric(s.care_type, errors="coerce")
    s["prev_ct"] = s.groupby("patient_id").ct.shift()
    s["prev_sep"] = s.groupby("patient_id").separation_ts.shift()
    s["gap_h"] = (s.admission_ts - s.prev_sep).dt.total_seconds() / 3600
    p = s.dropna(subset=["prev_ct"])
    if len(p) < 500:
        return Artefact(4, "sequence", "do care phases follow one another?",
                        True, "not checkable - too few repeat admissions")
    contiguous = int(((p.ct != p.prev_ct) & (p.gap_h.abs() <= 1)).sum())
    marg = s.ct.value_counts(normalize=True)
    a2r = float("nan")                             # 21 = acute, 22 = rehab
    sub = p[p.prev_ct == 21]
    if len(sub) > 100 and 22 in marg.index and marg[22] > 0:
        a2r = float((sub.ct == 22).mean() / marg[22])
    return Artefact(
        4, "sequence", "do care phases follow one another?",
        preserved=contiguous > 0.01 * len(p),
        statistic=f"{contiguous:,} of {len(p):,} consecutive pairs change type within 1h; "
                  f"acute->rehab enrichment {a2r:.2f}x marginal",
        blocks="within-stay phase progression, true total length of stay")


def check_revisit(ed: pd.DataFrame) -> Artefact:
    """5. Do unresolved visits bring people back?

    A real ED clusters return visits: whatever was not fixed brings the patient
    back within days. Measured against a Poisson process, which has no memory,
    so its interval coefficient of variation is 1 and its short-interval mass
    follows the exponential. Both statistics are window-independent, so they
    compare across extracts covering different periods - the raw 72-hour rate
    does not.
    """
    s = ed.dropna(subset=["arrival_ts"]).sort_values(
        ["patient_id", "arrival_ts"]).copy()
    s["prev"] = s.groupby("patient_id").arrival_ts.shift()
    g = ((s.arrival_ts - s["prev"]).dt.total_seconds() / 86400).dropna()
    if len(g) < 2000:
        return Artefact(5, "revisit", "do unresolved visits bring people back?",
                        True, "not checkable - too few repeat visits")
    m = float(g.mean())
    cv = float(g.std() / m)
    obs, exp = float((g < 3).mean()), float(1 - np.exp(-3 / m))
    return Artefact(
        5, "revisit", "do unresolved visits bring people back?",
        preserved=cv > 1.3,
        statistic=f"interval CV {cv:.2f} (Poisson = 1.00); 72h revisits {obs / exp:.1f}x "
                  f"the exponential null  [MIMIC real: CV 1.50, 5.8x]",
        blocks="return-visit quality indicators, care-continuity analysis")


def check_boarding(ed: pd.DataFrame, window_min: float = 120) -> Artefact:
    """6. When a hospital has no ward beds, does everyone waiting feel it?

    Access block is a shared condition, not a private one. If the wards are
    full at six in the evening, every patient whose bed was requested around
    then waits, and their waits move together. So compare each boarding time
    against the mean of everyone else who asked for a bed at the same hospital
    within a couple of hours. In a real queue this correlation runs +0.8 or
    higher - our own simulator gives +0.80 to +0.95 depending on ward
    occupancy. Drawn independently per record it sits at zero.

    This is a separate question from check 2. That one asks whether the wait to
    be *seen* responds to crowding; this asks whether the wait for a *bed*
    does. An extract can plausibly keep one and lose the other, and the two
    block different work.
    """
    need = {"bed_request_ts", "depart_ts", "site"}
    if not need <= set(ed.columns):
        return Artefact(6, "boarding", "is waiting for a bed shared?",
                        True, "not checkable - no bed-request timestamp")
    b = ed.dropna(subset=["bed_request_ts", "depart_ts"]).copy()
    b["board"] = (b.depart_ts - b.bed_request_ts).dt.total_seconds() / 60
    b = b[(b.board >= 0) & (b.board < 3000)]
    if len(b) < 5000:
        return Artefact(6, "boarding", "is waiting for a bed shared?",
                        True, "not checkable - too few boarding episodes")
    rs, ws = [], []
    for _, g in b.groupby("site"):
        if len(g) < 3000:
            continue
        g = g.sort_values("bed_request_ts")
        t = g.bed_request_ts.values.astype("datetime64[m]").astype(float)
        y = g.board.values
        cs = np.concatenate([[0.0], np.cumsum(y)])
        lo = np.searchsorted(t, t - window_min)
        hi = np.searchsorted(t, t + window_min)
        cnt = hi - lo - 1                       # everyone else in the window
        others = np.where(cnt > 0, (cs[hi] - cs[lo] - y) / np.maximum(cnt, 1), np.nan)
        m = cnt >= 3
        if m.sum() > 1000 and np.std(others[m]) > 0:
            rs.append(float(np.corrcoef(others[m], y[m])[0, 1]))
            ws.append(int(m.sum()))
    if not rs:
        return Artefact(6, "boarding", "is waiting for a bed shared?",
                        True, "not checkable - no hospital busy enough")
    r = float(np.average(rs, weights=ws))
    return Artefact(
        6, "boarding", "is waiting for a bed shared?",
        preserved=r > 0.2,
        statistic=f"corr(neighbours\' boarding, own) = {r:+.3f} over {len(rs)} hospitals, "
                  f"median boarding {b.board.median():.0f} min  [real queue: +0.80 to +0.95]",
        blocks="access-block drivers, ward-discharge timing, anything predicting how long a bed takes")


def run_all(ed: pd.DataFrame, hm: pd.DataFrame | None = None) -> list[Artefact]:
    """The five checks, in the order they should be read."""
    out = [check_priority(ed), check_queueing(ed)]
    if hm is not None:
        out += [check_event_linkage(ed, hm), check_sequence(hm)]
    out += [check_revisit(ed), check_boarding(ed)]
    return sorted(out, key=lambda a: a.n)


def report(ed: pd.DataFrame, hm: pd.DataFrame | None = None) -> str:
    res = run_all(ed, hm)
    missing = [a for a in res if not a.preserved]
    lines = ["cross-record relationships", ""]
    lines += [str(a) for a in res]
    lines += ["", f"{len(res) - len(missing)}/{len(res)} preserved"]
    if missing:
        lines += ["",
                  "Row-wise generation reproduces what one record looks like, not what",
                  "happens between records. Everything under 'blocks' is off the table."]
    return "\n".join(lines)
