"""Command line entry point:  edsim <command> [options]"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def _print(title: str, obj) -> None:
    print(f"\n=== {title} ===")
    print(obj if not isinstance(obj, dict) else
          "\n".join(f"  {k:>22}: {v}" for k, v in obj.items()))


def cmd_units(args):
    from edsim.aihw import wa_hospitals
    df = wa_hospitals(args.lhn)
    _print(f"WA public hospitals{f' - {args.lhn}' if args.lhn else ''}",
           df[["code", "name", "lhn"]].to_string(index=False))


def cmd_aihw(args):
    from edsim.aihw import ed_triage_profile
    from edsim.calibrate import from_aihw
    _print("AIHW triage profile", ed_triage_profile(args.code).to_string(index=False))
    p = from_aihw(args.code)
    _print("calibrated params", {"daily_arrivals": round(p.daily_arrivals, 1),
                                 "acuity_mix": {k: round(v, 4) for k, v in p.acuity_mix.items()},
                                 "n_cubicles": p.n_cubicles, "notes": p.notes})
    if args.out:
        p.to_json(args.out)
        print(f"\nwrote {args.out}")


def cmd_inspect(args):
    from edsim.loaders.portal import inspect
    _print(f"columns in {args.path}", inspect(args.path).to_string(index=False))
    from edsim.schema import COLUMNS
    _print("canonical targets", ", ".join(COLUMNS))


def cmd_calibrate(args):
    from edsim.calibrate import from_encounters
    from edsim.loaders import load
    from edsim.schema import describe
    df = load(args.source, path=args.path)
    _print("schema coverage", describe(df).to_string(index=False))
    p = from_encounters(df, source=f"{args.source}:{args.path}")
    _print("calibrated params", {"daily_arrivals": round(p.daily_arrivals, 2),
                                 "acuity_mix": {k: round(v, 3) for k, v in p.acuity_mix.items()},
                                 "treat_mean_min": {k: round(v, 1) for k, v in p.treat_mean_min.items()},
                                 "admit_prob": {k: round(v, 3) for k, v in p.admit_prob.items()},
                                 "n_cubicles": p.n_cubicles, "notes": p.notes})
    if args.out:
        p.to_json(args.out)
        print(f"\nwrote {args.out}")


def cmd_simulate(args):
    from edsim.calibrate import SimParams
    from edsim.metrics import by_triage, hourly_load, kpis
    from edsim.sim import simulate

    p = SimParams.from_json(args.params) if args.params else SimParams()
    if args.cubicles:
        p.n_cubicles = args.cubicles
    if args.ward_occupancy is not None:
        p.ward_bed_occupancy = args.ward_occupancy

    df = simulate(p, days=args.days, seed=args.seed)
    _print(f"KPIs ({args.days:g} simulated days, source={p.source})", kpis(df))
    _print("by ATS category", by_triage(df).to_string(index=False))
    if args.hourly:
        _print("by hour of day", hourly_load(df).to_string(index=False))
    if args.out:
        df.to_parquet(args.out) if args.out.endswith(".parquet") else df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(df):,} encounters)")


def cmd_sweep(args):
    from edsim.calibrate import SimParams
    from edsim.sim import scenario_sweep
    p = SimParams.from_json(args.params) if args.params else SimParams()
    grid = scenario_sweep(
        p,
        cubicles=[int(x) for x in args.cubicles.split(",")] if args.cubicles else None,
        ward_occupancy=[float(x) for x in args.ward_occupancy.split(",")] if args.ward_occupancy else None,
        days=args.days, seed=args.seed,
    )
    cols = ["n_cubicles", "ward_occupancy", "median_wait_min", "p90_wait_min",
            "on_time_pct", "neat_pct", "median_boarding_min", "access_block_pct", "dnw_pct"]
    _print("scenario sweep", grid[[c for c in cols if c in grid.columns]].to_string(index=False))


def cmd_validate(args):
    from edsim.calibrate import check, from_aihw
    from edsim.metrics import compare_to_aihw, kpis
    from edsim.sim import simulate

    p = from_aihw(args.code)
    if args.params_override:
        import json
        for k, v in json.loads(args.params_override).items():
            setattr(p, k, v)
    for w in check(p):
        print(f"  ! {w}")

    df = simulate(p, days=args.days, seed=args.seed)
    _print("simulated KPIs", kpis(df))
    cmp = compare_to_aihw(df, args.code)
    _print("simulated vs AIHW observed", cmp.to_string(index=False))
    mae = cmp["on_time_gap_pp"].abs().mean()
    print(f"\n  mean absolute error on on-time %: {mae:.1f} pp")
    print("  tune treat_mean_min / n_cubicles / ward_bed_occupancy to close the gap")


def cmd_tune(args):
    from edsim.calibrate import tune_to_aihw
    print(f"grid searching capacity params against AIHW {args.code} ...")
    tuned, results = tune_to_aihw(args.code, days=args.days, seed=args.seed)
    _print("top 10 parameter sets", results.head(10).to_string(index=False))
    _print("tuned params", {"daily_arrivals": round(tuned.daily_arrivals, 1),
                            "n_cubicles": tuned.n_cubicles,
                            "ward_bed_occupancy": tuned.ward_bed_occupancy,
                            "treat_mean_min": {k: round(v, 1) for k, v in tuned.treat_mean_min.items()},
                            "notes": tuned.notes})
    if args.out:
        tuned.to_json(args.out)
        print(f"\nwrote {args.out} - now run: edsim simulate --params {args.out}")


def cmd_demo(args):
    """End-to-end smoke test with zero downloads."""
    from edsim.calibrate import SimParams
    from edsim.metrics import by_triage, kpis
    from edsim.sim import simulate
    from edsim.calibrate import check, size_capacity
    p = size_capacity(SimParams(source="built-in defaults"))
    for w in check(p):
        print(f"  ! {w}")
    _print("capacity", {"n_cubicles": p.n_cubicles, "n_resus_bays": p.n_resus_bays,
                        "n_inpatient_beds": p.n_inpatient_beds,
                        "ward_bed_occupancy (non-ED)": p.ward_bed_occupancy})
    df = simulate(p, days=args.days, seed=args.seed)
    _print("KPIs", kpis(df))
    _print("by ATS category", by_triage(df).to_string(index=False))
    print("\nNext: `edsim units --lhn 'East Metropolitan'` to calibrate on a real WA hospital.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="edsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="run the simulator on built-in defaults")
    s.add_argument("--days", type=float, default=30)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("units", help="list WA public hospitals from AIHW")
    s.add_argument("--lhn", help="filter, e.g. 'East Metropolitan'")
    s.set_defaults(func=cmd_units)

    s = sub.add_parser("aihw", help="calibrate from public AIHW data for one hospital")
    s.add_argument("code", help="reporting unit code, e.g. H0632 (Royal Perth)")
    s.add_argument("--out")
    s.set_defaults(func=cmd_aihw)

    s = sub.add_parser("validate", help="back-test the sim against AIHW observed data")
    s.add_argument("code", help="reporting unit code, e.g. H0632")
    s.add_argument("--days", type=float, default=60)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--params-override", dest="params_override",
                   help='JSON, e.g. \'{"n_cubicles": 40, "ward_bed_occupancy": 0.8}\'')
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("tune", help="grid search capacity params to match AIHW observed data")
    s.add_argument("code")
    s.add_argument("--days", type=float, default=45)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out")
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("inspect", help="show source columns next to canonical ones")
    s.add_argument("--path", required=True)
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("calibrate", help="fit params from patient-level data")
    s.add_argument("--source", required=True, choices=["mimic_demo", "synthea", "portal"])
    s.add_argument("--path", required=True)
    s.add_argument("--out")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("simulate", help="run the ED simulation")
    s.add_argument("--params", help="params.json from calibrate/aihw")
    s.add_argument("--days", type=float, default=30)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--cubicles", type=int)
    s.add_argument("--ward-occupancy", type=float, dest="ward_occupancy")
    s.add_argument("--hourly", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("sweep", help="what-if grid over capacity and ward pressure")
    s.add_argument("--params")
    s.add_argument("--cubicles", help="comma separated, e.g. 24,30,36")
    s.add_argument("--ward-occupancy", dest="ward_occupancy", help="e.g. 0.85,0.92,0.97")
    s.add_argument("--days", type=float, default=14)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
