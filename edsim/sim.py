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
3. **Resus and fast track never board.** When an admitted patient finishes
   treatment in a resus bay, they release it immediately and wait for their
   ward bed in a corridor space instead. Real EDs decant resus first, because
   the bay is too valuable to hold - which is why a completely gridlocked
   department can still report near-100% on-time for ATS 1. Only the main
   cubicles absorb access block. Without this the model collapses ATS 1 along
   with everything else, which no real ED does.
4. **Streaming.** Three separate pools of space, not one queue: resus bays
   (ATS 1-2), a fast track open only part of the day (ATS 4-5), and the main
   cubicles (everyone else, plus overflow). This is what produces the real
   signature of a congested Australian ED - excellent ATS 1, tolerable ATS 5,
   and ATS 3 collapsing in the middle, because ATS 3 is the only category
   protected by neither stream and therefore competes directly with admitted
   patients boarding for a ward bed. Call it the ATS 3 squeeze.

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


def _probit(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = (-2 * np.log(1 - p)) ** 0.5
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


class EDModel:
    def __init__(self, params: SimParams, *, seed: int = 7, warm_up_hours: float = 24.0):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.warm_up_hours = warm_up_hours
        self.env = simpy.Environment()
        self.triage = simpy.Resource(self.env, capacity=params.n_triage_nurses)
        self.cubicles = simpy.PriorityResource(self.env, capacity=params.n_cubicles)
        # Preemptive: an arriving ATS 1 takes a bay from a stabilised ATS 2,
        # who is bumped out to finish treatment in a main cubicle.
        self.resus = simpy.PreemptiveResource(self.env, capacity=params.n_resus_bays)
        self.fasttrack = simpy.PriorityResource(
            self.env, capacity=max(params.n_fasttrack_spaces, 0) or 1)
        # Priority = the moment the bed was asked for. Requesting at triage
        # buys a place in the queue, not a bed held empty while the patient is
        # still being treated - which is what actually happens, and modelling
        # it the other way makes early requests catastrophically bad for a
        # reason that has nothing to do with reality.
        self.ward = simpy.PriorityResource(self.env, capacity=params.n_inpatient_beds)
        self.records: list[dict] = []
        self.background_waits: list[float] = []
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
        # Total target occupancy minus what ED admissions will hold themselves.
        from edsim.calibrate import mean_admit_prob
        ed_beds = (self.p.daily_arrivals * mean_admit_prob(self.p)
                   * self.p.inpatient_los_hours / 24.0)
        target = max(self.p.n_inpatient_beds * self.p.ward_bed_occupancy - ed_beds, 0.0)
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
        """Non-ED demand: elective, direct and transfer admissions.

        They queue on the same beds with a priority of 'now', so an ED patient
        who claimed a position at triage outranks an elective admission that
        appeared later. This is the actual mechanism by which early requesting
        helps, and it is a transfer between demand streams rather than new
        capacity - which is why the elective wait is reported too.
        """
        with self.ward.request(priority=self.env.now) as req:
            t0 = self.env.now
            yield req
            self.background_waits.append(self.env.now - t0)
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

        # Will this patient actually need a bed? Decided now so the predictor
        # can be scored against it, but not revealed to the ED until treatment
        # finishes - exactly as in reality.
        truly_admitted = self.rng.random() < self.p.admit_prob.get(ats, 0.2)
        rec["truly_admitted"] = truly_admitted

        claim_ts = None
        if self.p.bed_request_policy == "at_triage" and \
                self._predict_admission(truly_admitted) > 0.5:
            rec["early_request"] = True
            rec["wasted_request"] = not truly_admitted
            claim_ts = self.env.now                      # queue position, not a bed

        space, req = yield from self._acquire_space(ats, rec)
        if req is None:                       # abandoned the queue
            rec["disposition"] = "did_not_wait"
            rec["depart_ts"] = self.env.now
            self.records.append(rec)
            return

        rec["seen_ts"] = self.env.now
        rec["space"] = space
        pool = {"resus": self.resus, "fasttrack": self.fasttrack}.get(space, self.cubicles)

        remaining = sample_treat_minutes(self.rng, self.p, ats)
        while remaining > 0:
            started = self.env.now
            try:
                yield self.env.timeout(remaining)
                remaining = 0
            except simpy.Interrupt:
                # Bumped out of resus by an ATS 1. Finish in a main cubicle.
                remaining -= self.env.now - started
                rec["preempted"] = True
                req = self.cubicles.request(priority=ats)
                yield req
                space, pool = "cubicle", self.cubicles

        if truly_admitted:
            board_start = self.env.now
            # Resus bays and fast track are decanted immediately - the patient
            # boards in a corridor. Only main cubicles are held during boarding,
            # which is what makes access block bite the ATS 3 queue specifically.
            if space in ("resus", "fasttrack"):
                pool.release(req)
                pool = None
            # If the request went in at triage, the queue position is already
            # held and the wait may be over before treatment even finished.
            bed = self.ward.request(priority=claim_ts if claim_ts is not None
                                    else self.env.now)
            yield bed
            rec["boarding_min"] = self.env.now - board_start
            rec["disposition"] = "admitted"
            rec["depart_ts"] = self.env.now
            if pool is not None:
                pool.release(req)
            self.records.append(rec)
            yield self.env.timeout(self.rng.exponential(self.p.inpatient_los_hours * 60.0))
            self.ward.release(bed)
            return

        # Not admitted after all. Any bed requested at triage was a false
        # positive: it held a queue slot a real admission could have used.
        rec["disposition"] = "discharged"
        rec["depart_ts"] = self.env.now
        pool.release(req)
        self.records.append(rec)

    def _predict_admission(self, truth: bool) -> float:
        """Stand-in for the triage-time model, parameterised by target AUC.

        A latent normal score: N(d, 1) if the patient really needs a bed,
        N(0, 1) if not. Separation d = sqrt(2) * Phi^-1(AUC), which reproduces
        the requested AUC exactly. The score is squashed to (0, 1) so the
        threshold reads like a probability.

        The earlier version mixed truth with uniform noise, which separates
        *perfectly* for any skill above 0.5 - so "skill 0.7" and an oracle
        gave identical results. Worth recording: a predictor stub that cannot
        make mistakes silently turns the whole experiment into a best case.
        """
        from math import sqrt
        auc = min(max(self.p.predictor_skill, 0.5), 0.9999)
        d = sqrt(2) * _probit(auc)
        z = self.rng.normal(d if truth else 0.0, 1.0)
        # Threshold expressed as the false-positive rate we accept, not as a
        # probability: Phi(N(0,1)) is uniform, so thresholding it at 0.5 flags
        # half of all non-admissions no matter how good the model is.
        z_star = _probit(1 - self.p.early_request_fpr)
        return 1.0 if z >= z_star else 0.0

    def _fasttrack_open(self) -> bool:
        if self.p.n_fasttrack_spaces <= 0:
            return False
        hour = (self.env.now / 60.0) % 24
        return self.p.fasttrack_open_hour <= hour < self.p.fasttrack_close_hour

    def _race(self, requests: list[tuple[str, object, object]], patience: float):
        """Wait on several space requests at once; keep the first granted."""
        timeout = self.env.timeout(patience)
        res = yield simpy.events.AnyOf(self.env, [r for _, r, _ in requests] + [timeout])

        winner = None
        for name, req, pool in requests:
            if req in res and winner is None:
                winner = (name, req, pool)
            elif req in res:
                pool.release(req)          # granted but not taken
            else:
                req.cancel()
        return winner

    def _acquire_space(self, ats: int, rec: dict):
        """Get a treatment space from whichever stream the patient qualifies for.

        Returns (space_name, request) or (None, None) if the patient gave up.
        """
        patience = self.p.dnw_patience_min.get(ats, float("inf"))
        candidates = []

        if ats in self.p.resus_ats and self.p.n_resus_bays > 0:
            candidates.append(
                ("resus", self.resus.request(priority=ats, preempt=(ats == 1)), self.resus))
        if ats in self.p.fasttrack_ats and self._fasttrack_open():
            candidates.append(("fasttrack", self.fasttrack.request(priority=ats), self.fasttrack))
        candidates.append(("cubicle", self.cubicles.request(priority=ats), self.cubicles))

        winner = yield from self._race(candidates, patience)
        if winner is None:
            return None, None
        name, req, _ = winner
        return name, req

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
