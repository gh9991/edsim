"""Turn data (real, synthetic, or AIHW aggregate) into simulator parameters.

Two entry points, because you will have two very different kinds of data:

  from_encounters(df)   patient-level rows -> fit everything empirically
  from_aihw(code)       WA aggregate only  -> fit volume + triage mix,
                        fall back to literature defaults for service times

Both return the same SimParams, so the simulator neither knows nor cares.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd

from edsim.triage import ATS_TARGET_MIN

# Literature-ish defaults, used only where data cannot inform a parameter.
# Replace these the moment the hackathon portal data lands.
DEFAULT_TREAT_MEAN_MIN = {1: 180.0, 2: 150.0, 3: 110.0, 4: 70.0, 5: 45.0}
DEFAULT_ADMIT_PROB = {1: 0.75, 2: 0.55, 3: 0.30, 4: 0.10, 5: 0.03}
DEFAULT_ACUITY_MIX = {1: 0.01, 2: 0.18, 3: 0.42, 4: 0.34, 5: 0.05}

# Typical metro ED diurnal shape (relative weight per hour, midnight-first).
DEFAULT_HOURLY_PROFILE = [
    0.55, 0.42, 0.34, 0.29, 0.28, 0.32, 0.45, 0.70,
    1.05, 1.35, 1.50, 1.52, 1.45, 1.40, 1.38, 1.35,
    1.30, 1.25, 1.20, 1.10, 1.00, 0.90, 0.78, 0.66,
]


@dataclasses.dataclass
class SimParams:
    """Everything the ED simulator needs. Serialisable, diffable, reviewable."""

    # demand
    daily_arrivals: float = 200.0
    hourly_profile: list[float] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_HOURLY_PROFILE))
    acuity_mix: dict[int, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_ACUITY_MIX))

    # capacity
    n_triage_nurses: int = 2
    n_cubicles: int = 30
    n_resus_bays: int = 4          # ring-fenced; see resus_ats
    # Which categories may use a resus bay. Defaults to ATS 1 only: opening
    # resus to all of ATS 2 saturates it (ATS 2 is ~25% of arrivals) and the
    # ATS 1 patients then queue behind them, which no real ED tolerates.
    # ATS 1 preempts ATS 2 out of a bay (see sim.EDModel). Without preemption,
    # ATS 2 saturates resus and ATS 1 queues behind it - which no real ED
    # tolerates, and which shows up immediately as ATS 1 on-time far below 100%.
    resus_ats: list[int] = dataclasses.field(default_factory=lambda: [1, 2])
    n_inpatient_beds: int = 500

    # Fast track / ambulatory stream. A separate pool of spaces and staff for
    # low-acuity patients, so they do not queue behind admitted patients who
    # are boarding in the main cubicles. Almost every metro ED runs one, and
    # almost no ED runs it 24/7 - the opening hours matter as much as the size.
    n_fasttrack_spaces: int = 6
    fasttrack_ats: list[int] = dataclasses.field(default_factory=lambda: [4, 5])
    fasttrack_open_hour: int = 8
    fasttrack_close_hour: int = 22

    # service
    triage_time_min: float = 5.0
    treat_mean_min: dict[int, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_TREAT_MEAN_MIN))
    treat_cv: float = 0.7               # coefficient of variation -> lognormal sigma
    admit_prob: dict[int, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_ADMIT_PROB))

    # access block: how long an admitted patient holds an ED cubicle waiting
    # for a ward bed. This is the WA system's actual pain point.
    inpatient_los_hours: float = 72.0
    # TARGET TOTAL ward occupancy, ED admissions included. The simulator
    # generates only the non-ED remainder as background demand, so this number
    # means what a hospital executive means by "we run at 92%".
    #
    # Previously this was documented as the non-ED share while size_capacity
    # treated it as the total, so background demand and ED demand were counted
    # twice and every configuration came out unstable. WA's access-block
    # problem lives in this single number, so it had better mean one thing.
    ward_bed_occupancy: float = 0.92

    # What fraction of all inpatient admissions arrive via the ED. Sets how big
    # the ward has to be relative to ED demand. 50-70% is typical for a metro
    # hospital; the rest are elective, direct and transfer admissions.
    ed_share_of_admissions: float = 0.60

    # Bed-request policy. "on_decision" is current practice: the request goes in
    # once the doctor has finished working the patient up. "at_triage" issues it
    # the moment triage predicts admission, so the coordination runs in parallel
    # with treatment instead of after it.
    #
    # The catch, and the reason prediction quality matters operationally: a
    # request issued at triage for a patient who turns out NOT to need a bed
    # holds a queue position that a real admission needed. Sensitivity and
    # specificity stop being abstract and start costing beds.
    bed_request_policy: str = "on_decision"      # or "at_triage"
    # Operating point for the early-request rule, as an accepted false-positive
    # rate. 0.10 = flag 10% of patients who will not need a bed.
    early_request_fpr: float = 0.10
    # Quality of the triage-time predictor, given as its AUC. 0.5 = coin flip,
    # 0.83 = what we actually measured on MIMIC with vitals + chief complaint,
    # 0.99 ~ oracle.
    predictor_skill: float = 0.83

    # behaviour
    dnw_patience_min: dict[int, float] = dataclasses.field(
        default_factory=lambda: {1: 1e9, 2: 1e9, 3: 240.0, 4: 180.0, 5: 150.0})

    # provenance - always record where the numbers came from
    source: str = "defaults"
    notes: str = ""

    def to_json(self, path: str | pathlib.Path) -> None:
        d = dataclasses.asdict(self)
        for k in ("acuity_mix", "treat_mean_min", "admit_prob", "dnw_patience_min"):
            d[k] = {str(kk): vv for kk, vv in d[k].items()}
        pathlib.Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "SimParams":
        d = json.loads(pathlib.Path(path).read_text())
        for k in ("acuity_mix", "treat_mean_min", "admit_prob", "dnw_patience_min"):
            d[k] = {int(kk): vv for kk, vv in d[k].items()}
        return cls(**d)


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Convert (mean, coefficient of variation) to numpy lognormal (mu, sigma)."""
    sigma = float(np.sqrt(np.log(1.0 + cv ** 2)))
    mu = float(np.log(max(mean, 1e-6)) - 0.5 * sigma ** 2)
    return mu, sigma


def from_encounters(df: pd.DataFrame, *, source: str = "encounters") -> SimParams:
    """Fit parameters empirically from canonical patient-level rows."""
    p = SimParams(source=source)
    notes = []

    if df.empty:
        p.notes = "empty frame - defaults only"
        return p

    span_days = max(
        (df["arrival_ts"].max() - df["arrival_ts"].min()).total_seconds() / 86400.0, 1.0
    )
    rate = float(len(df) / span_days)

    # GUARD: MIMIC date-shifts every patient into a random future year to
    # de-identify them, so the observed span is ~100 years and the implied
    # arrival rate collapses to nearly zero. Silently calibrating on that
    # produces an empty simulation that looks like it ran fine.
    if rate < 1.0:
        notes.append(
            f"daily_arrivals: REJECTED empirical rate {rate:.3f}/day over {span_days:,.0f} days "
            f"- timestamps are almost certainly date-shifted (MIMIC does this per patient). "
            f"Kept default {p.daily_arrivals}. Set daily_arrivals from a real volume source "
            f"(e.g. `edsim aihw <code>`) instead."
        )
    else:
        p.daily_arrivals = rate
        notes.append("daily_arrivals: empirical")

    # Hour-of-day survives date shifting (MIMIC preserves time-of-day), so the
    # diurnal profile is still usable even when the arrival rate is not.
    hours = df["arrival_ts"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    if hours.sum() > 0:
        profile = (hours / hours.mean()).astype(float)
        p.hourly_profile = [round(float(x), 4) for x in profile]
        notes.append("hourly_profile: empirical")

    mix = df["ats"].value_counts(normalize=True)
    if not mix.empty:
        p.acuity_mix = {int(k): float(v) for k, v in mix.items() if 1 <= k <= 5}
        notes.append("acuity_mix: empirical")

    # Service time. Prefer seen->depart; fall back to LOS, which overstates
    # treatment because it contains the wait. Flag it loudly when we do.
    if df["seen_ts"].notna().any():
        svc = (df["depart_ts"] - df["seen_ts"]).dt.total_seconds() / 60
        notes.append("treat_mean_min: from seen_ts->depart_ts")
    else:
        svc = df["los_min"]
        notes.append("treat_mean_min: LOS PROXY (no seen_ts in source) - overstates service time")

    by_ats = pd.DataFrame({"ats": df["ats"], "svc": svc}).dropna()
    by_ats = by_ats[by_ats["svc"].between(1, 60 * 24)]
    if not by_ats.empty:
        med = by_ats.groupby("ats")["svc"].median()
        p.treat_mean_min = {int(k): float(v) for k, v in med.items() if 1 <= k <= 5}
        for a in range(1, 6):
            p.treat_mean_min.setdefault(a, DEFAULT_TREAT_MEAN_MIN[a])
        pooled = by_ats["svc"]
        if pooled.mean() > 0:
            p.treat_cv = float(min(pooled.std() / pooled.mean(), 2.0))

    if df["disposition"].notna().any():
        adm = df.groupby("ats")["admitted"].mean()
        p.admit_prob = {int(k): float(v) for k, v in adm.items() if 1 <= k <= 5}
        for a in range(1, 6):
            p.admit_prob.setdefault(a, DEFAULT_ADMIT_PROB[a])
        notes.append("admit_prob: empirical")

    # Capacity is never in the data. Size cubicles so utilisation lands ~85%.
    size_capacity(p)
    notes.append(
        f"capacity: sized for ~85% utilisation -> {p.n_cubicles} cubicles, "
        f"{p.n_inpatient_beds} ward beds (NOT from data - sweep it)")

    p.notes = "; ".join(notes)
    return p


def from_aihw(reporting_unit_code: str, period_end: str | None = None) -> SimParams:
    """Calibrate volume + triage mix from public AIHW data for a real WA hospital."""
    from edsim.aihw import ed_triage_profile

    prof = ed_triage_profile(reporting_unit_code, period_end)
    if prof.empty:
        raise ValueError(f"no AIHW ED data for {reporting_unit_code}")

    p = SimParams(source=f"AIHW MyHospitals {reporting_unit_code}")
    total = float(prof["presentations"].sum())
    # AIHW ED reporting periods are financial years
    p.daily_arrivals = total / 365.0
    p.acuity_mix = {
        int(r.ats): float(r.presentations / total)
        for r in prof.itertuples() if total > 0
    }

    size_capacity(p)

    observed = {int(r.ats): r.pct_treated_on_time for r in prof.itertuples()
                if not pd.isna(getattr(r, "pct_treated_on_time", np.nan))}
    p.notes = (
        f"hospital={prof['hospital'].iloc[0]}; period_end={prof['period_end'].iloc[0]}; "
        f"annual_presentations={total:,.0f}; "
        f"AIHW observed on-time % by ATS={observed}; "
        "service times + capacity are DEFAULTS - aggregate data cannot inform them"
    )
    return p


def mean_admit_prob(p: SimParams) -> float:
    return sum(p.acuity_mix.get(a, 0.0) * p.admit_prob.get(a, 0.0) for a in range(1, 6))


def size_capacity(p: SimParams, *, ed_target_util: float = 0.85,
                  ward_target_util: float = 0.85) -> SimParams:
    """Size ED cubicles and ward beds to the demand the params imply.

    Capacity is never in the data. Rather than let the simulation silently
    gridlock on a made-up bed count, derive it from offered load, then let the
    user sweep around it. Erlang-style: beds = offered_load / target_utilisation.
    """
    def _load(acuities, hours=24.0, share=1.0):
        """Offered load in beds = arrival rate per hour x service time in hours."""
        return sum(
            p.daily_arrivals * share * p.acuity_mix.get(a, 0) * p.treat_mean_min.get(a, 60) / 60
            for a in acuities
        ) / hours

    ft_ats = [a for a in p.fasttrack_ats if 1 <= a <= 5]
    open_hours = max(p.fasttrack_close_hour - p.fasttrack_open_hour, 1)
    open_share = sum(p.hourly_profile[h] for h in
                     range(p.fasttrack_open_hour, p.fasttrack_close_hour)) / sum(p.hourly_profile)

    if ft_ats and p.n_fasttrack_spaces > 0:
        ft_load = _load(ft_ats, hours=open_hours, share=open_share)
        p.n_fasttrack_spaces = max(1, int(np.ceil(ft_load / ed_target_util)))
        # what the main ED still has to absorb: everything not diverted
        main_load = _load([a for a in range(1, 6) if a not in ft_ats])
        main_load += _load(ft_ats, share=1 - open_share)
    else:
        main_load = _load(range(1, 6))

    ed_load = main_load
    p.n_cubicles = max(4, int(np.ceil(ed_load / ed_target_util)))

    # Ward sizing. The earlier version required ED demand to fit inside the
    # slice of beds NOT held by anyone else, which for a 90%-occupied hospital
    # implies a ward ten times larger than any real one - Royal Perth has
    # roughly 450 beds, not 1,700. A pool that large also never queues, so
    # boarding came out at zero however hard the hospital was pushed.
    #
    # ED admissions are typically 50-70% of all admissions at a metro hospital,
    # so total bed demand is ED demand divided by that share, and the ward is
    # sized to carry it at the target occupancy.
    ed_bed_demand = p.daily_arrivals * mean_admit_prob(p) * (p.inpatient_los_hours / 24.0)
    total_demand = ed_bed_demand / max(p.ed_share_of_admissions, 0.05)
    p.n_inpatient_beds = max(10, int(np.ceil(total_demand / max(p.ward_bed_occupancy, 0.5))))
    return p


def check(p: SimParams) -> list[str]:
    """Warn about parameter sets that will gridlock. Run this before you pitch."""
    warnings = []
    ed_load = sum(
        p.daily_arrivals * p.acuity_mix.get(a, 0) * p.treat_mean_min.get(a, 60) / 60
        for a in range(1, 6)
    ) / 24.0
    util = ed_load / max(p.n_cubicles, 1)
    if util > 0.95:
        warnings.append(f"ED cubicle utilisation {util:.0%} - queue will grow without bound")

    ed_bed_demand = p.daily_arrivals * mean_admit_prob(p) * (p.inpatient_los_hours / 24.0)
    free_beds = p.n_inpatient_beds * (1 - p.ward_bed_occupancy)
    if free_beds <= 0:
        warnings.append("no ward beds available to ED - every admission will board forever")
    elif ed_bed_demand / free_beds > 0.95:
        warnings.append(
            f"ward utilisation for ED admissions {ed_bed_demand / free_beds:.0%} "
            f"(demand {ed_bed_demand:.0f} beds vs {free_beds:.0f} free) - severe access block"
        )
    if sum(p.acuity_mix.values()) < 0.99 or sum(p.acuity_mix.values()) > 1.01:
        warnings.append(f"acuity_mix sums to {sum(p.acuity_mix.values()):.3f}, not 1")
    return warnings


def sample_treat_minutes(rng: np.random.Generator, params: SimParams, ats: int) -> float:
    mu, sigma = _lognormal_params(params.treat_mean_min.get(ats, 60.0), params.treat_cv)
    return float(rng.lognormal(mu, sigma))


def tune_to_aihw(reporting_unit_code: str, *, days: float = 45, seed: int = 7,
                 cubicles: list[int] | None = None,
                 occupancies: list[float] | None = None,
                 treat_scales: list[float] | None = None,
                 fasttrack: list[int] | None = None,
                 resus_bays: list[int] | None = None,
                 verbose: bool = True) -> tuple[SimParams, "pd.DataFrame"]:
    """Coarse grid search for the capacity parameters AIHW cannot tell us.

    Volume and triage mix come from real data. Service time, cubicle count and
    ward pressure do not - so search them for the combination that best
    reproduces the hospital's published on-time performance. That is the
    difference between "a simulation" and "a simulation of Royal Perth".

    Returns (best_params, all_results_sorted).
    """
    import dataclasses

    import pandas as pd

    from edsim.metrics import compare_to_aihw
    from edsim.sim import simulate

    base = from_aihw(reporting_unit_code)
    cubicles = cubicles or [12, 16, 20]
    occupancies = occupancies or [0.88, 0.92]
    treat_scales = treat_scales or [1.2, 1.6]
    fasttrack = fasttrack or [3, 6]
    resus_bays = resus_bays or [8, 12]

    import itertools

    rows = []
    for n, occ, ts, ft, rb in itertools.product(
            cubicles, occupancies, treat_scales, fasttrack, resus_bays):
            if True:
                p = dataclasses.replace(
                    base,
                    n_cubicles=n,
                    n_fasttrack_spaces=ft,
                    n_resus_bays=rb,
                    ward_bed_occupancy=occ,
                    treat_mean_min={k: v * ts for k, v in base.treat_mean_min.items()},
                )
                try:
                    cmp = compare_to_aihw(simulate(p, days=days, seed=seed),
                                          reporting_unit_code)
                    mae = float(cmp["on_time_gap_pp"].abs().mean())
                except Exception as exc:      # a gridlocked combo can yield no rows
                    mae = float("inf")
                    if verbose:
                        print(f"  skip n={n} occ={occ} ts={ts}: {exc}")
                rows.append({"n_cubicles": n, "n_fasttrack_spaces": ft,
                             "n_resus_bays": rb, "ward_bed_occupancy": occ,
                             "treat_scale": ts, "mae_pp": round(mae, 2)})
                if verbose:
                    print(f"  cub={n:>3} ft={ft:>2} resus={rb:>2} occ={occ:.2f} "
                          f"treat x{ts:<4} -> MAE {mae:6.2f} pp")

    results = pd.DataFrame(rows).sort_values("mae_pp").reset_index(drop=True)
    best = results.iloc[0]
    tuned = dataclasses.replace(
        base,
        n_cubicles=int(best.n_cubicles),
        n_fasttrack_spaces=int(best.n_fasttrack_spaces),
        n_resus_bays=int(best.n_resus_bays),
        ward_bed_occupancy=float(best.ward_bed_occupancy),
        treat_mean_min={k: v * float(best.treat_scale) for k, v in base.treat_mean_min.items()},
        notes=base.notes + f"; TUNED to AIHW (MAE {best.mae_pp} pp, treat_scale {best.treat_scale})",
    )
    return tuned, results
