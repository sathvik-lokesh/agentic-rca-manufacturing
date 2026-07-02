"""Train the defect detector and calibrate the abstention thresholds.

Run: uv run python -m rca.train
Saves model + preprocessor + thresholds to outputs/artifacts.joblib and
metrics to outputs/metrics.json.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import Preprocessor, load_raw

OUT = Path(__file__).resolve().parent.parent / "outputs"
SEED = 42

# Abstention band targets (set on cross-validated train probabilities, never on the holdout):
# - above t_high: flag as FAIL — precision there must be worth an engineer's attention
# - below t_low: treat as PASS — we want to miss almost no true fails down there
T_HIGH_MIN_PRECISION = 0.50
T_LOW_TARGET_RECALL = 0.90
OOD_MISSING_FRAC = 0.30  # lot missing >30% of its sensors -> model shouldn't judge it
OOD_EXTREME_SIGMA = 6.0  # any sensor this far out of the training envelope -> defer
OOD_EXTREME_COUNT = 3


def pick_thresholds(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_true, p)
    # t_high: smallest threshold whose precision clears the bar (fall back to top decile)
    ok = np.where(prec[:-1] >= T_HIGH_MIN_PRECISION)[0]
    t_high = float(thr[ok[0]]) if len(ok) else float(np.quantile(p, 0.90))
    # t_low: largest threshold that still keeps recall >= target
    ok = np.where(rec[:-1] >= T_LOW_TARGET_RECALL)[0]
    t_low = float(thr[ok[-1]]) if len(ok) else float(np.quantile(p, 0.50))
    if t_low >= t_high:  # degenerate curve on tiny minority class — keep a real band
        t_low = t_high * 0.4
    return t_low, t_high


def main() -> None:
    X, y = load_raw()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    prep = Preprocessor().fit(X_tr, y_tr)
    Xt_tr = prep.transform(X_tr)

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_cv = cross_val_predict(model, Xt_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    t_low, t_high = pick_thresholds(y_tr.to_numpy(), p_cv)

    model.fit(Xt_tr, y_tr)

    p_te = model.predict_proba(prep.transform(X_te))[:, 1]
    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "fail_prevalence": round(float(y.mean()), 4),
        "n_features_used": len(prep.features),
        "holdout_pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
        "holdout_roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "cv_pr_auc_train": round(float(average_precision_score(y_tr, p_cv)), 4),
        "t_low": round(t_low, 4),
        "t_high": round(t_high, 4),
    }

    OUT.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "prep": prep,
            "t_low": t_low,
            "t_high": t_high,
            "test_index": X_te.index.to_list(),
            "seed": SEED,
        },
        OUT / "artifacts.joblib",
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nPR-AUC baseline (random) = prevalence = {y.mean():.3f}")
    print(f"artifacts -> {OUT / 'artifacts.joblib'}")


if __name__ == "__main__":
    main()
