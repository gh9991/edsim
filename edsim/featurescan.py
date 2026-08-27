"""Systematic screen of every field against the target - before any modelling.

Written after a process failure: `mental_health_admission` carries the phrase
"departure status of **admitted**" in its own definition, yet its univariate
AUC is only 0.53 because it covers under 2% of presentations. Screening on AUC
alone would never have flagged it. It surfaced only because someone asked about
that specific field, which is not a method.

So the scan reports three quantities per field, because each catches a
different kind of relationship:

  auc       ranking power across the whole cohort. Catches broad, strong
            predictors. Blind to anything that only fires on a small subgroup.
  lift      P(y | best level) / P(y). Catches a level that nearly determines
            the outcome even when almost nobody has it.
  coverage  what share of rows sit in that level. High lift with tiny coverage
            is the signature of a hidden leak - the model overfits that
            subgroup, the overall metric never moves, and the field is empty
            at inference time.

AUC uses out-of-fold target encoding. Encoding on the same rows you score makes
every high-cardinality field look predictive, which would bury the real signals
under noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SUPPORT = 200          # a level below this is noise, not a finding
# ⚠️ Set this relative to the extract. Too high and it silently drops the rare
# levels that matter most - ATS 1 is under 1% of presentations, and a narrow
# leak is narrow by definition. Both vanish if the floor is above their count.
CONT_BINS = 10


def _as_levels(s: pd.Series) -> pd.Series:
    """Categorical stays as-is; continuous becomes quantile bins."""
    if s.dtype.kind in "biufc" and s.nunique(dropna=True) > 20:
        return pd.qcut(s, CONT_BINS, duplicates="drop").astype(str)
    return s.astype(str)


def _oof_auc(levels: pd.Series, y: np.ndarray, seed: int = 7) -> float:
    """Target-encode out of fold, then score. In-fold encoding inflates
    high-cardinality fields to near-1 and makes the scan useless."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(levels))
    score = np.full(len(levels), y.mean(), dtype=float)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        enc = pd.Series(y[tr]).groupby(levels[tr].to_numpy()).mean()
        score[te] = levels[te].map(enc).fillna(y[tr].mean()).to_numpy()
    if len(np.unique(score)) < 2:
        return 0.5
    return float(roc_auc_score(y, score))


def scan(df: pd.DataFrame, target: str = "needs_bed",
         exclude: list[str] | None = None,
         min_support: int | None = None) -> pd.DataFrame:
    """One row per field: coverage, cardinality, AUC, best level and its lift."""
    y = df[target].astype(int).to_numpy()
    base = y.mean()
    skip = set(exclude or []) | {target}
    floor = min_support if min_support is not None else MIN_SUPPORT
    rows = []

    for col in df.columns:
        if col in skip or df[col].isna().all():
            continue
        s = df[col]
        if s.dtype.kind == "M":                       # timestamps handled elsewhere
            continue
        levels = _as_levels(s)
        g = pd.DataFrame({"lv": levels, "y": y}).groupby("lv")["y"]
        stat = pd.DataFrame({"n": g.size(), "rate": g.mean()})
        stat = stat[stat.n >= floor]
        if stat.empty:
            continue
        best = stat.rate.idxmax()
        rows.append({
            "field": col,
            "n_levels": int(s.nunique(dropna=True)),
            "missing": round(float(s.isna().mean()), 4),
            "auc": round(_oof_auc(levels, y), 4),
            "best_level": str(best)[:34],
            "best_rate": round(float(stat.rate.max()), 4),
            "lift": round(float(stat.rate.max() / base), 2),
            "coverage": round(float(stat.n[best] / len(df)), 4),
        })

    out = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    out["timing"] = out.field.map(TIMING).fillna("unknown")
    out["verdict"] = [_verdict(r) for _, r in out.iterrows()]
    return out


# When each field comes into existence, from the data dictionary. The scan
# cannot derive this - statistics alone cannot tell a rare-but-genuine
# predictor (ATS 1, helicopter retrieval) from a leak. Only timing can.
TIMING = {
    "triage_category": "triage", "mode_of_arrival": "triage",
    "referral_source": "triage", "age": "triage", "sex": "triage",
    "ethnicity": "triage", "establishment_code": "triage",
    "metropolitan_hospital_flag": "triage",
    "self_harm_attendance": "triage?",
    "affected_by_drugs_and_or_alcohol": "triage?",
    "primary_diagnosis_ICD10AM_chapter": "POST-HOC",
    "mental_health_attendance": "POST-HOC?",
    "mental_health_admission": "POST-HOC",
    "potentially_avoidable_general_practitioner_type_attendance": "POST-HOC?",
    "departure_status": "OUTCOME", "discharge_datetime": "OUTCOME",
    "bed_request_datetime": "OUTCOME",
    "clinical_care_commencement_datetime": "OUTCOME",
}


def _signal(r) -> str:
    """What the numbers say, independent of when the field exists."""
    if r.auc >= 0.90:
        return "near-perfect ranking"
    if r.lift >= 3.0 and r.coverage <= 0.10:
        return "narrow + high lift"
    if r.auc >= 0.70:
        return "strong"
    if r.lift >= 2.0:
        return "informative"
    return ""


def _verdict(r) -> str:
    """Signal and timing together. Neither alone decides it.

    A rare level with a big lift is a leak only if the field does not exist
    when the decision is made. ATS 1 lifts admission 3.4x on 0.9% of
    presentations and is perfectly legitimate; mental_health_admission lifts
    3.7x on 2.0% and is not, because its own definition says "based on
    departure status of admitted".
    """
    t = TIMING.get(r.field, "unknown")
    sig = _signal(r)
    if t == "OUTCOME":
        return "🔴 outcome - never a feature"
    if t == "POST-HOC":
        return f"🔴 leak - post-hoc ({sig})" if sig else "🔴 leak - post-hoc"
    if t.endswith("?"):
        return f"🟡 timing unclear ({sig})" if sig else "🟡 timing unclear"
    if t == "unknown":
        return f"⚪ timing unknown ({sig})" if sig else "⚪ timing unknown"
    return f"🟢 usable - {sig}" if sig else "🟢 usable"


def scan_pairs(df: pd.DataFrame, target: str = "needs_bed",
               fields: list[str] | None = None, top: int = 15,
               min_support: int | None = None) -> pd.DataFrame:
    """Interactions: the cell of a field-pair with the highest outcome rate.

    A pair can separate the outcome when neither field does alone, so scanning
    fields one at a time is not enough.
    """
    y = df[target].astype(int).to_numpy()
    base = y.mean()
    cols = fields or [c for c in df.columns
                      if c != target and df[c].dtype.kind != "M"
                      and 1 < df[c].nunique(dropna=True) <= 30]
    floor = min_support if min_support is not None else MIN_SUPPORT
    rows = []
    for i, a in enumerate(cols):
        la = _as_levels(df[a])
        for b in cols[i + 1:]:
            lb = _as_levels(df[b])
            g = pd.DataFrame({"a": la, "b": lb, "y": y}).groupby(["a", "b"])["y"]
            stat = pd.DataFrame({"n": g.size(), "rate": g.mean()})
            stat = stat[stat.n >= floor]
            if stat.empty:
                continue
            k = stat.rate.idxmax()
            rows.append({"pair": f"{a} × {b}",
                         "cell": f"{str(k[0])[:18]} | {str(k[1])[:18]}",
                         "n": int(stat.n[k]),
                         "rate": round(float(stat.rate.max()), 4),
                         "lift": round(float(stat.rate.max() / base), 2),
                         "coverage": round(float(stat.n[k] / len(df)), 4)})
    return (pd.DataFrame(rows).sort_values("lift", ascending=False)
            .head(top).reset_index(drop=True))


def report(df: pd.DataFrame, target: str = "needs_bed", **kw) -> str:
    s = scan(df, target, **kw)
    floor = kw.get("min_support") or MIN_SUPPORT
    base = df[target].astype(int).mean()
    lines = [f"target = {target}   base rate {base:.2%}   n = {len(df):,}   "
             f"min level support {floor}", ""]
    lines.append(s.to_string(index=False))
    bad = s[s.verdict.str.startswith(("🔴", "🟡", "⚪"))]
    if len(bad):
        lines += ["", "do not use as features without resolving timing:"]
        lines += [f"  {r.field:<34} {r.verdict}" for _, r in bad.iterrows()]
    return "\n".join(lines)
