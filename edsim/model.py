"""Predicting admission from triage - and turning that into a bed forecast.

The 2025 Department of Health challenge asked two things that look the same
and are not:

  "Which types of patients arriving at an ED will require hospital beds?"
      -> a per-patient classifier. Ranking matters. AUC is the right metric.

  "Predict the NUMBER of inpatient admissions from ED triage."
      -> a count forecast. Ranking is irrelevant. What matters is whether the
         predicted probabilities are *calibrated*, because the expected number
         of admissions is just their sum.

A model can have excellent AUC and still be useless for bed planning, if it is
systematically over- or under-confident. Most teams will report AUC and stop.
This module reports both, and treats the daily count error as the headline
number, because that is what a bed manager actually acts on.

Two other things done deliberately:

- **Temporal split, never random.** ED behaviour drifts (seasons, staffing,
  policy). A random split lets the model see the future and flatters it.
- **Calibrated on a separate slice.** Fitting the calibrator on the training
  data would undo the point of calibrating.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from edsim import features


@dataclasses.dataclass
class AdmissionModel:
    pipeline: Pipeline
    feature_names: list[str]
    train_period: tuple[pd.Timestamp, pd.Timestamp]
    notes: str = ""

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = features.build(df)[self.feature_names]
        return self.pipeline.predict_proba(X)[:, 1]


def temporal_split(df: pd.DataFrame, *, train_frac: float = 0.6,
                   calib_frac: float = 0.2) -> tuple[pd.DataFrame, ...]:
    """Split by arrival time: train | calibrate | test, in chronological order."""
    df = df.sort_values("arrival_ts").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train_frac), int(n * (train_frac + calib_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def train(df: pd.DataFrame, *, target: str = "admitted", seed: int = 7,
          n_estimators: int = 300) -> tuple[AdmissionModel, pd.DataFrame]:
    """Fit on the earliest slice, calibrate on the next, return both + test set."""
    train_df, calib_df, test_df = temporal_split(df)

    X_tr = features.build(train_df)
    # Drop features with no observed values at all - a source with no vitals
    # (MIMIC-IV-ED has no age; the simulator has neither) should simply not
    # carry those columns rather than feed the imputer nothing.
    X_tr = X_tr.loc[:, X_tr.notna().any()]
    y_tr = features.target(train_df, target)
    X_ca = features.build(calib_df)[X_tr.columns]
    y_ca = features.target(calib_df, target)

    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
    ])
    base.fit(X_tr, y_tr)

    # isotonic calibration on the held-out middle slice - this is what makes
    # the summed probabilities a usable bed forecast
    # FrozenEstimator: calibrate the already-fitted model without refitting it
    # (replaces the removed cv="prefit")
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(X_ca, y_ca)

    model = AdmissionModel(
        pipeline=calibrated,
        feature_names=list(X_tr.columns),
        train_period=(train_df["arrival_ts"].min(), train_df["arrival_ts"].max()),
        notes=(f"RandomForest({n_estimators}) + isotonic calibration; "
               f"temporal split {len(train_df)}/{len(calib_df)}/{len(test_df)}"),
    )
    return model, test_df


def evaluate(model: AdmissionModel, test_df: pd.DataFrame,
             target: str = "admitted") -> dict:
    p = model.predict_proba(test_df)
    y = features.target(test_df, target).to_numpy()

    out = {
        "n_test": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        # ranking - "which patients"
        "auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        # calibration - "how many"
        "brier": round(float(brier_score_loss(y, p)), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "mean_actual": round(float(y.mean()), 4),
        "calibration_bias": round(float(p.mean() - y.mean()), 4),
    }
    out.update(daily_forecast_error(model, test_df, target))
    return out


def daily_forecast_error(model: AdmissionModel, test_df: pd.DataFrame,
                         target: str = "admitted") -> dict:
    """The headline number: how far off is the predicted bed demand per day?"""
    d = test_df.sort_values("arrival_ts").reset_index(drop=True).copy()
    d["p"] = model.predict_proba(test_df)
    d["y"] = features.target(test_df, target).to_numpy()
    daily = d.groupby(d["arrival_ts"].dt.date).agg(
        predicted=("p", "sum"), actual=("y", "sum"))
    err = daily["predicted"] - daily["actual"]

    # Same trap as in calibrate.from_encounters: MIMIC shifts every patient to
    # a different year, so "per day" becomes one patient per day and the bed
    # forecast is meaningless. Say so rather than printing a tidy fake number.
    per_day = len(d) / max(len(daily), 1)
    warning = None
    if per_day < 5:
        warning = (f"only {per_day:.1f} encounters per calendar day - timestamps "
                   "are probably date-shifted (MIMIC does this), so the daily "
                   "bed forecast below is not meaningful")

    return {
        "days": int(len(daily)),
        **({"daily_forecast_warning": warning} if warning else {}),
        "mean_actual_admissions_per_day": round(float(daily["actual"].mean()), 1),
        "daily_mae_beds": round(float(err.abs().mean()), 2),
        "daily_bias_beds": round(float(err.mean()), 2),
    }


def calibration_table(model: AdmissionModel, test_df: pd.DataFrame,
                      bins: int = 10, target: str = "admitted") -> pd.DataFrame:
    """Predicted vs observed by probability decile. The plot judges ask for."""
    p = model.predict_proba(test_df)
    y = features.target(test_df, target).to_numpy()
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
    t["gap"] = (t["mean_predicted"] - t["observed"]).round(3)
    return t.round(3).reset_index()


def importances(model: AdmissionModel, test_df: pd.DataFrame,
                target: str = "admitted", seed: int = 7,
                n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance - model-agnostic, and honest about correlated inputs.

    This is the drill-down that won the 2024 Technical Achievement prize: a
    clinician can see which variables drove the call before acting on it.
    """
    X = features.build(test_df)[model.feature_names]
    y = features.target(test_df, target)
    r = permutation_importance(model.pipeline, X, y, n_repeats=n_repeats,
                               random_state=seed, scoring="roc_auc", n_jobs=-1)
    return (pd.DataFrame({"feature": model.feature_names,
                          "importance": r.importances_mean.round(4),
                          "std": r.importances_std.round(4)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def explain_patient(model: AdmissionModel, df: pd.DataFrame, i: int = 0,
                    top: int = 6) -> pd.DataFrame:
    """Why this patient? Shift each feature to the cohort median, see what moves.

    Crude next to SHAP, but it needs no extra dependency, runs instantly, and
    a clinician can follow the logic out loud: "if their heart rate had been
    normal, this would have dropped from 71% to 48%."
    """
    X = features.build(df)[model.feature_names]
    base = model.pipeline.predict_proba(X.iloc[[i]])[:, 1][0]
    med = X.median(numeric_only=True)

    rows = []
    for f in model.feature_names:
        x = X.iloc[[i]].copy()
        if pd.isna(med.get(f)):
            continue
        x[f] = med[f]
        rows.append({"feature": f,
                     "patient_value": X.iloc[i][f],
                     "cohort_median": round(float(med[f]), 2),
                     "prob_if_median": round(float(
                         model.pipeline.predict_proba(x)[:, 1][0]), 4)})
    out = pd.DataFrame(rows)
    out["contribution"] = (base - out["prob_if_median"]).round(4)
    out.insert(0, "predicted_prob", round(float(base), 4))
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top)
