"""Abstention gate: decide FLAG / PASS / DEFER-TO-HUMAN for one lot.

The point of the prototype: an agent that acts autonomously only when it has
the right to be confident, and says WHY it defers otherwise.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .train import OOD_EXTREME_COUNT, OOD_EXTREME_SIGMA, OOD_MISSING_FRAC


@dataclass
class Decision:
    verdict: str          # "FAIL-FLAG" | "PASS" | "DEFER"
    p_fail: float
    reason: str
    top_signals: list[tuple[str, float]]  # (sensor, z-score) — NaN z = sensor missing


def top_deviations(z_row: pd.Series, importances: pd.Series, k: int = 5) -> list[tuple[str, float]]:
    """Rank sensors by |deviation from normal production| weighted by model importance."""
    score = z_row.abs().fillna(0) * (importances + importances[importances > 0].min())
    top = score.sort_values(ascending=False).head(k)
    return [(name, float(z_row[name])) for name in top.index]


def decide(
    p_fail: float,
    z_row: pd.Series,
    missing_frac: float,
    importances: pd.Series,
    t_low: float,
    t_high: float,
) -> Decision:
    signals = top_deviations(z_row, importances)

    n_extreme = int((z_row.abs() > OOD_EXTREME_SIGMA).sum())
    if missing_frac > OOD_MISSING_FRAC:
        return Decision(
            "DEFER", p_fail,
            f"out-of-distribution: {missing_frac:.0%} of sensors missing — "
            "the model has no basis to judge this lot", signals,
        )
    if n_extreme >= OOD_EXTREME_COUNT:
        return Decision(
            "DEFER", p_fail,
            f"out-of-distribution: {n_extreme} sensors beyond {OOD_EXTREME_SIGMA:.0f} sigma of "
            "normal production — outside the training envelope", signals,
        )
    if p_fail >= t_high:
        return Decision("FAIL-FLAG", p_fail, f"p_fail={p_fail:.2f} >= {t_high:.2f}", signals)
    if p_fail <= t_low:
        return Decision("PASS", p_fail, f"p_fail={p_fail:.2f} <= {t_low:.2f}", signals)
    return Decision(
        "DEFER", p_fail,
        f"uncertain: p_fail={p_fail:.2f} sits in the abstention band "
        f"({t_low:.2f}, {t_high:.2f}) — not confident enough to act", signals,
    )


def format_signals(signals: list[tuple[str, float]]) -> str:
    parts = []
    for name, z in signals:
        parts.append(f"{name} (missing)" if np.isnan(z) else f"{name} {z:+.1f}σ")
    return ", ".join(parts)
