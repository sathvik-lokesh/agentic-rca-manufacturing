"""Load and preprocess the SECOM dataset (UCI).

SECOM: 1567 production lots x 590 anonymized process sensors, label 1 = FAIL
(defect), -1 = PASS. Heavily imbalanced (~6.6% fail) and full of missing values
— which is exactly the reliability story this prototype is about.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"

MAX_MISSING_FRAC = 0.40  # drop sensors that are missing in >40% of lots


def _download_if_missing() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name in ("secom.data", "secom_labels.data"):
        path = DATA_DIR / name
        if not path.exists():
            import requests

            print(f"downloading {name} from UCI ...")
            resp = requests.get(f"{UCI_BASE}/{name}", timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    _download_if_missing()
    X = pd.read_csv(DATA_DIR / "secom.data", sep=" ", header=None)
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]
    labels = pd.read_csv(DATA_DIR / "secom_labels.data", sep=" ", header=None)
    y = (labels[0] == 1).astype(int)  # 1 = FAIL, 0 = PASS
    return X, y


def select_features(X_train: pd.DataFrame) -> list[str]:
    """Feature selection fitted on TRAIN ONLY: drop mostly-missing and constant sensors."""
    missing_ok = X_train.columns[X_train.isna().mean() <= MAX_MISSING_FRAC]
    keep = [c for c in missing_ok if X_train[c].nunique(dropna=True) > 1]
    return keep


class Preprocessor:
    """Median imputation + PASS-population statistics, fitted on train only.

    The PASS-population mean/std per sensor is what lets us later say
    "sensor_059 is +3.2 sigma away from normal production" for a flagged lot.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "Preprocessor":
        self.features = select_features(X_train)
        Xf = X_train[self.features]
        self.medians = Xf.median()
        pass_rows = Xf[y_train == 0]
        self.pass_mean = pass_rows.mean()
        self.pass_std = pass_rows.std().replace(0, np.nan)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features].fillna(self.medians)

    def z_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Deviation of each sensor from normal (PASS) production, in sigmas.
        Computed on raw values so missing sensors stay NaN (imputation would
        fake a 'normal' reading)."""
        return (X[self.features] - self.pass_mean) / self.pass_std

    def missing_frac(self, X: pd.DataFrame) -> pd.Series:
        return X[self.features].isna().mean(axis=1)
