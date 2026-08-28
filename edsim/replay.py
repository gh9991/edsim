"""Replay real patients through a finite ED and recompute what they waited.

The 2025 extract records a waiting time for every patient that bears no
relation to anyone else's - correlation with how full the department was at
the time is -0.006. But waiting is not an independent fact about a patient. It
is what happens when more people want a bed than there are beds, and it is
computed, not observed.

Everything needed to compute it survived: when each patient arrived, how sick
they were, and how long they occupied a space. Only the consequence of
contention is missing. So take the real patients, discard the recorded wait,
and re-derive it by making them compete for a fixed number of spaces.

    waits = replay(arrivals, service_min, priority, beds=27)

What this does not do is tell you how many beds there are. That is not in the
extract either, and it cannot be recovered by fitting to the recorded waits -
those were not produced by a queue, so no capacity reproduces both their
middle and their tail. Take capacity from the hospital or from published
figures, and report a range rather than a point.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd


def replay(arrival_min, service_min, priority, beds: int) -> np.ndarray:
    """Waiting time for each patient, in the order given.

    A non-preemptive priority queue with `beds` identical servers: whenever a
    space frees, the waiting patient with the best priority takes it, ties
    going to whoever arrived first. Returns minutes waited, aligned to the
    input order - never to arrival order, which is where this kind of code
    usually goes wrong.
    """
    a0 = np.asarray(arrival_min, dtype=float)
    s0 = np.asarray(service_min, dtype=float)
    p0 = np.asarray(priority, dtype=float)
    n = len(a0)
    if not n:
        return np.empty(0)
    if beds < 1:
        raise ValueError("beds must be at least 1")
    if not (len(s0) == len(p0) == n):
        raise ValueError("arrival, service and priority must be the same length")

    order = np.argsort(a0, kind="stable")
    a, s, p = a0[order], s0[order], p0[order]

    start = np.empty(n)
    busy: list[float] = []        # when each occupied space frees
    queue: list[tuple] = []       # (priority, arrival, index) - best pops first
    nxt = 0
    t = a[0]
    seated = 0

    while seated < n:
        while nxt < n and a[nxt] <= t:
            heapq.heappush(queue, (p[nxt], a[nxt], nxt))
            nxt += 1
        while busy and busy[0] <= t:
            heapq.heappop(busy)
        while queue and len(busy) < beds:
            _, _, k = heapq.heappop(queue)
            start[k] = t
            heapq.heappush(busy, t + s[k])
            seated += 1
        when = []
        if nxt < n:
            when.append(a[nxt])       # someone new turns up
        if busy and queue:
            when.append(busy[0])      # a space frees for someone waiting
        if not when:
            break
        t = min(when)

    out = np.empty(n)
    out[order] = start - a
    return out


def offered_load(service_min, arrival_min) -> float:
    """Spaces needed on average for the queue never to build.

    Total occupied time divided by the period it is spread over. Capacity below
    this cannot keep up at all; just above it the queue is unstable; comfortable
    departments run somewhere around 0.8 of capacity.
    """
    a = np.asarray(arrival_min, dtype=float)
    span = a.max() - a.min()
    return float(np.sum(service_min) / span) if span > 0 else float("nan")


def occupancy_at_arrival(arrival_min, wait_min, service_min) -> np.ndarray:
    """How many patients were in the department as each one arrived."""
    a = np.asarray(arrival_min, dtype=float)
    depart = a + np.asarray(wait_min) + np.asarray(service_min)
    return (np.searchsorted(np.sort(a), a, "right")
            - np.searchsorted(np.sort(depart), a, "left"))


def sweep(ed: pd.DataFrame, beds: list[int] | None = None) -> pd.DataFrame:
    """Recompute waits across a range of capacities.

    Needs arrival_ts, a service time, and an acuity. Reports the recorded wait
    on the same rows for comparison - the gap between the two is the point.
    """
    d = ed.dropna(subset=["arrival_ts", "seen_ts", "depart_ts", "ats"]).copy()
    d["svc"] = (d.depart_ts - d.seen_ts).dt.total_seconds() / 60
    d["recorded"] = (d.seen_ts - d.arrival_ts).dt.total_seconds() / 60
    d = d[(d.svc > 0) & (d.svc < 1440) & (d.recorded >= 0)]
    if len(d) < 1000:
        raise ValueError(f"only {len(d)} usable rows - need at least 1000")
    a = (d.arrival_ts - d.arrival_ts.min()).dt.total_seconds().values / 60
    load = offered_load(d.svc.values, a)
    beds = beds or [max(int(load * k), 1) for k in (1.0, 1.1, 1.25, 1.5, 2.0)]

    rows = []
    for c in beds:
        w = replay(a, d.svc.values, d.ats.values.astype(float), c)
        occ = occupancy_at_arrival(a, w, d.svc.values)
        rows.append({"beds": c, "utilisation": load / c,
                     "median_min": float(np.median(w)),
                     "p90_min": float(np.percentile(w, 90)),
                     "corr_occupancy": float(np.corrcoef(occ, w)[0, 1])})
    occ0 = occupancy_at_arrival(a, d.recorded.values, d.svc.values)
    rows.append({"beds": np.nan, "utilisation": np.nan,
                 "median_min": float(d.recorded.median()),
                 "p90_min": float(d.recorded.quantile(0.9)),
                 "corr_occupancy": float(np.corrcoef(occ0, d.recorded)[0, 1])})
    out = pd.DataFrame(rows, index=[f"{c} beds" for c in beds] + ["as recorded"])
    out.attrs["offered_load"] = load
    return out
