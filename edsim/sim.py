"""Discrete-event simulation of an emergency department (SimPy).

Flow modelled:

    arrival -> triage nurse -> [wait for cubicle, priority by ATS]
            -> treatment -> (admitted?) -> [BOARDING: hold the cubicle while
               waiting for a ward bed] -> depart

Two things here are deliberate and are what makes this more than a queue toy:

1. **Access block / boarding.** An admitted patient keeps occupying the ED
   cubicle until a ward bed frees up. This is the dominant driver of ED
   congestion in WA and the reason "just add ED staff" does not fix it.
2. **Did-not-wait.** Low-acuity patients abandon the queue after a patience
   threshold. Ignore this and you overestimate waiting-room load.

Output is a canonical encounter frame, so simulated and real data flow through
exactly the same metrics and models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import simpy

from edsim import schema
from edsim.calibrate import SimParams, sample_treat_minutes

EPOCH = pd.Timestamp("2026-09-19 00:00:00")


class EDModel:
    def __init__(self, params: SimParams, *, seed: int = 7, warm_up_hours: float = 24.0):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.warm_up_hours = warm_up_hours
        self.env = simpy.Environment()
        self.triage = simpy.Resource(self.env, capacity=params.n_triage_nurses)
        self.cubicles = simpy.PriorityResource(self.env, capacity=params.n_cubicles)
        self.resus = simpy.PriorityResource(self.env, capacity=params.n_resus_bays)
        self.ward = simpy.Resource(self.env, capacity=params.n_inpatient_beds)
        self.records: list[dict] = []
        self._n = 0

    # --- demand ------------------------------------------------------------

    def _rate_at(self, t_hours: float) -> float:
        """Arrivals per hour at simulation time t (non-homogeneous Poisson)."""
        hour = int(t_hours % 24)
        base = self.p.daily_arrivals / 24.0
        return base * self.p.hourly_profile[hour]

    def _arrivals(self):
        """Thinning algorithm for the non-homogeneous Poisson process."""
        lam_max = max(self._rate_at(h) for h in range(24)) or 1e-9
        acuities = list(self.p.acuity_mix.keys())
        weights = np.array(list(self.p.acuity_mix.values()), dtype=float)
        weights = weights / weights.sum()
        while True:
            yield self.env.timeout(self.rng.exponential(60.0 / lam_max))
            if self.rng.random() <= self._rate_at(self.env.now / 60.0) / lam_max:
                self._n += 1
                ats = int(self.rng.choice(acuities, p=weights))
                self.env.process(self._patient(f"E{self._n:07d}", ats))

    def _background_admissions(self):
        """Non-ED demand on ward beds (elective, transfers) holding occupancy."""
        target = self.p.n_inpatient_beds * self.p.ward_bed_occupancy
        rate_per_hour = target / max(self.p.inpatient_los_hours, 1.0)
        # pre-load the wards so we do not start the day with an empty hospital
        for _ in range(int(target)):
            self.env.process(self._ward_stay(
                self.rng.exponential(self.p.inpatient_los_hours * 60.0) * self.rng.random()))
        while True:
            yield self.env.timeout(self.rng.exponential(60.0 / max(rate_per_hour, 1e-9)))
            self.env.process(self._ward_stay(
                self.rng.exponential(self.p.inpatient_los_hours * 60.0)))

    def _ward_stay(self, minutes: float):
        with self.ward.request() as req:
            yield req
            yield self.env.timeout(minutes)

    # --- patient pathway ---------------------------------------------------

    def _patient(self, eid: str, ats: int):
        rec = {
            "encounter_id": eid,
            "patient_id": eid,
            "arrival_ts": self.env.now,
            "acuity": ats,
            "acuity_scale": "ATS",
            "triage_ts": None, "seen_ts": None, "depart_ts": None,
            "disposition": None, "boarding_min": 0.0, "space": None,
        }

        # ATS 1 bypasses the triage queue - they go straight to a resus bay.
        if ats == 1:
            rec["triage_ts"] = self.env.now
        else:
            with self.triage.request() as req:
                yield req
                yield self.env.timeout(self.rng.exponential(self.p.triage_time_min))
            rec["triage_ts"] = self.env.now

        space, req = yield from self._acquire_space(ats, rec)
        if req is None:                       # abandoned the queue
            rec["disposition"] = "did_not_wait"
            rec["depart_ts"] = self.env.now
            self.records.append(rec)
            return

        rec["seen_ts"] = self.env.now
        rec["space"] = space
        pool = self.resus if space == "resus" else self.cubicles
        yield self.env.timeout(sample_treat_minutes(self.rng, self.p, ats))

        if self.rng.random() < self.p.admit_prob.get(ats, 0.2):
            board_start = self.env.now
            bed = self.ward.request()
            yield bed                          # still holding the ED space: access block
            rec["boarding_min"] = self.env.now - board_start
            rec["disposition"] = "admitted"
            rec["depart_ts"] = self.env.now
            pool.release(req)
            self.records.append(rec)
            yield self.env.timeout(self.rng.exponential(self.p.inpatient_los_hours * 60.0))
            self.ward.release(bed)
            return

        rec["disposition"] = "discharged"
        rec["depart_ts"] = self.env.now
        pool.release(req)
        self.records.append(rec)

    def _acquire_space(self, ats: int, rec: dict):
        """Get a treatment space. ATS 1-2 may use a reserved resus bay.

        Returns (space_name, request) or (None, None) if the patient gave up.
        """
        patience = self.p.dnw_patience_min.get(ats, float("inf"))

        if ats <= 2 and self.p.n_resus_bays > 0:
            r_req = self.resus.request(priority=ats)
            c_req = self.cubicles.request(priority=ats)
            timeout = self.env.timeout(patience)
            res = yield r_req | c_req | timeout

            if r_req in res:
                if c_req in res:
                    self.cubicles.release(c_req)
                else:
                    c_req.cancel()
                return "resus", r_req
            if c_req in res:
                r_req.cancel()
                return "cubicle", c_req
            r_req.cancel()
            c_req.cancel()
            return None, None

        c_req = self.cubicles.request(priority=ats)
        res = yield c_req | self.env.timeout(patience)
        if c_req in res:
            return "cubicle", c_req
        c_req.cancel()
        return None, None

    # --- run ---------------------------------------------------------------

    def run(self, days: float = 30.0) -> pd.DataFrame:
        self.env.process(self._arrivals())
        self.env.process(self._background_admissions())
        self.env.run(until=(days * 24 + self.warm_up_hours) * 60)

        df = pd.DataFrame(self.records)
        if df.empty:
            return schema.empty_frame()

        warm_up_min = self.warm_up_hours * 60
        df = df[df["arrival_ts"] >= warm_up_min].copy()
        for c in ("arrival_ts", "triage_ts", "seen_ts", "depart_ts"):
            df[c] = EPOCH + pd.to_timedelta(df[c] - warm_up_min, unit="m")
        df["site"] = self.p.source
        return schema.validate(df)


def simulate(params: SimParams, days: float = 30.0, seed: int = 7) -> pd.DataFrame:
    return EDModel(params, seed=seed).run(days=days)


def scenario_sweep(params: SimParams, *, cubicles: list[int] | None = None,
                   ward_occupancy: list[float] | None = None,
                   days: float = 30.0, seed: int = 7) -> pd.DataFrame:
    """What-if grid. This is the table that wins the pitch."""
    import dataclasses

    from edsim.metrics import kpis

    rows = []
    for n in (cubicles or [params.n_cubicles]):
        for occ in (ward_occupancy or [params.ward_bed_occupancy]):
            p = dataclasses.replace(params, n_cubicles=n, ward_bed_occupancy=occ)
            k = kpis(simulate(p, days=days, seed=seed))
            rows.append({"n_cubicles": n, "ward_occupancy": occ, **k})
    return pd.DataFrame(rows)
