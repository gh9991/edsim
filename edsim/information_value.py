"""How much predictive power does EDDC give up by not recording clinical data?

WA's Emergency Department Data Collection is an administrative extract. It has
four excellent timestamps and no clinical measurements at all - no heart rate,
no blood pressure, no chief complaint. MIMIC-IV-ED is the mirror image: rich
clinical detail, no first-clinician-contact time.

That makes MIMIC the one place this question can be answered quickly:

    A   what EDDC records          demographics, triage category, arrival mode, time
    B   A + vitals                 heart rate, BP, resp, temp, SpO2, pain
    C   B + chief complaint        what the patient says is wrong

Same task, same test set, same algorithm. The only thing that changes is how
much the model is allowed to see, so the gap between them is attributable to
the information itself.

Two results matter, and they answer different questions:

    AUC          "which patients will need a bed"  - needs individual resolution
    cohort MAE   "how many beds will we need"      - group averages may suffice

If the AUC gap is large but the cohort gap is small, the honest advice to the
Department is: your existing extract is fine for bed forecasting, and you need
vitals only if you want per-patient early warning. That is a far more useful
answer than "collect more data".

THREE LIMITS, STATE THEM OUT LOUD:
  1. MIMIC is a US tertiary centre using ESI, not ATS. Indicative magnitude only.
  2. MIMIC-IV-ED has no age (it lives in the MIMIC-IV core module), and no
     crowding is computable because every patient is date-shifted into a
     different year. Real EDDC has both, so model A here is *weaker* than a
     true EDDC model - the measured gap is an upper bound on what is missing.
  3. Patients revisit (2.07 stays each), so the split is by patient, not by
     time. A random row split would leak the same person into train and test.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from edsim import features

# What EDDC actually records, expressed in canonical column names.
ADMIN = ["ats", "is_female", "by_ambulance", "hour", "dow",
         "is_weekend", "is_overnight", "hour_sin", "hour_cos"]
VITALS = features.VITALS + ["n_abnormal_vitals"]


@dataclasses.dataclass
class ArmResult:
    name: str
    n_features: int
    auc: float
    pr_auc: float
    brier: float
    calibration_bias: float
    cohort_mae_pct: float
    cohort_bias_pct: float
    cohort_floor_pct: float      # irreducible, given this arm's probabilities
    excess_over_floor_pct: float # how much of the error is the model's fault


def _text_matrix(train_text: pd.Series, all_text: pd.Series,
                 max_features: int = 150) -> tuple[np.ndarray, list[str]]:
    """TF-IDF over chief complaint, fitted on the training slice only."""
    vec = TfidfVectorizer(max_features=max_features, min_df=20,
                          ngram_range=(1, 2), strip_accents="unicode")
    vec.fit(train_text.fillna(""))
    M = vec.transform(all_text.fillna("")).toarray().astype(np.float32)
    return M, [f"cc__{t}" for t in vec.get_feature_names_out()]


def _split_by_patient(df: pd.DataFrame, seed: int = 7):
    """60/20/20 by patient, never by row - patients revisit."""
    groups = df["patient_id"].to_numpy()
    idx = np.arange(len(df))
    tr_i, rest_i = next(GroupShuffleSplit(n_splits=1, test_size=0.4,
                                          random_state=seed).split(idx, groups=groups))
    rest = idx[rest_i]
    ca_rel, te_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.5,
                                            random_state=seed).split(rest, groups=groups[rest]))
    return idx[tr_i], rest[ca_rel], rest[te_rel]


def cohort_floor_pct(p: np.ndarray, size: int = 150) -> float:
    """The irreducible cohort error, given these predicted probabilities.

    Even a perfectly calibrated model cannot forecast a bed count exactly,
    because the actual number admitted is a random draw. For a cohort of
    `size` patients the count has variance sum(p_i * (1 - p_i)), so the
    expected absolute error has a hard floor of sqrt(2/pi) * sd.

    Crucially this floor is **model-specific**. A Bernoulli variance is
    largest at p = 0.5, so a model that pushes probabilities towards 0 and 1
    - i.e. one that discriminates better - mechanically lowers its own floor.
    That is why a sharper model posts a smaller cohort error even when its
    calibration is no better. Comparing each arm against its own floor
    separates "forecasts better" from "is merely sharper".
    """
    var_per_patient = float(np.mean(p * (1 - p)))
    mean_p = float(np.mean(p))
    sd = np.sqrt(size * var_per_patient)
    return float(0.7979 * sd / (size * mean_p) * 100)


def _cohort_error(p: np.ndarray, y: np.ndarray, *, size: int = 150,
                  n: int = 400, seed: int = 7) -> tuple[float, float]:
    """Aggregate accuracy: for a random cohort of `size` patients, how far off
    is the predicted bed count? Reported as % of actual admissions.

    MIMIC's date shifting makes a real per-day forecast impossible, so we
    sample cohorts instead. It tests the same property - whether summed
    probabilities land on the truth.
    """
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        s = rng.choice(len(p), size=min(size, len(p)), replace=False)
        actual = y[s].sum()
        if actual == 0:
            continue
        errs.append((p[s].sum() - actual) / actual * 100)
    errs = np.array(errs)
    return float(np.abs(errs).mean()), float(errs.mean())


def _fit_arm(name: str, X: np.ndarray, y: np.ndarray, splits, seed: int = 7) -> ArmResult:
    tr, ca, te = splits
    # A column with no observed values anywhere (the simulator records no
    # vitals) breaks the histogram binner. Drop rather than impute - there is
    # nothing to impute from.
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, random_state=seed)
    base.fit(X[tr], y[tr])

    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(X[ca], y[ca])

    p, yt = model.predict_proba(X[te])[:, 1], y[te]
    mae, bias = _cohort_error(p, yt, seed=seed)
    floor = cohort_floor_pct(p)
    return ArmResult(
        name=name, n_features=X.shape[1],
        auc=round(float(roc_auc_score(yt, p)), 4),
        pr_auc=round(float(average_precision_score(yt, p)), 4),
        brier=round(float(brier_score_loss(yt, p)), 4),
        calibration_bias=round(float(p.mean() - yt.mean()), 4),
        cohort_mae_pct=round(mae, 2),
        cohort_bias_pct=round(bias, 2),
        cohort_floor_pct=round(floor, 2),
        excess_over_floor_pct=round(mae - floor, 2),
    )


def run(df: pd.DataFrame, *, seed: int = 7, cohort_size: int = 150,
        max_text_features: int = 150, verbose: bool = True) -> pd.DataFrame:
    """Run the A/B/C comparison. `df` must be a canonical encounter frame."""
    df = df.sort_values("arrival_ts").reset_index(drop=True)
    X_all = features.build(df, include_crowding=False)   # date-shifted: no crowding
    y = df["admitted"].astype(int).to_numpy()
    splits = _split_by_patient(df, seed)
    tr = splits[0]

    if verbose:
        print(f"{len(df):,} encounters · {df.patient_id.nunique():,} patients · "
              f"admitted {y.mean():.1%}")
        print(f"split by patient: train {len(splits[0]):,} / calib {len(splits[1]):,} "
              f"/ test {len(splits[2]):,}")
        overlap = set(df.patient_id.iloc[splits[0]]) & set(df.patient_id.iloc[splits[2]])
        print(f"train∩test patient overlap: {len(overlap)}  (must be 0)\n")

    A = X_all[ADMIN].to_numpy(dtype=np.float32)
    B = X_all[ADMIN + VITALS].to_numpy(dtype=np.float32)
    text, text_names = _text_matrix(
        df["presenting_complaint"].iloc[tr], df["presenting_complaint"],
        max_features=max_text_features)
    C = np.hstack([B, text])

    arms = [
        _fit_arm("A · EDDC-equivalent (admin only)", A, y, splits, seed),
        _fit_arm("B · A + vitals", B, y, splits, seed),
        _fit_arm("C · B + chief complaint", C, y, splits, seed),
    ]
    out = pd.DataFrame([dataclasses.asdict(a) for a in arms])
    out["auc_gain_vs_A"] = (out["auc"] - out.loc[0, "auc"]).round(4)
    out["cohort_mae_gain_vs_A"] = (out.loc[0, "cohort_mae_pct"] - out["cohort_mae_pct"]).round(2)
    return out
