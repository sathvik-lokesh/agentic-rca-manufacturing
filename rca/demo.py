"""End-to-end demo: detect -> decide (act/abstain) -> LLM root-cause.

Run: uv run python -m rca.demo  (after rca.train)
Picks holdout lots spanning the confidence spectrum (confident pass, flagged
fail, abstention band, high-missing) so every branch of the gate is visible.
Writes outputs/demo_run.md.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_raw
from .explain import explain
from .gate import decide, format_signals

OUT = Path(__file__).resolve().parent.parent / "outputs"


def pick_demo_lots(p: pd.Series, missing: pd.Series, t_low: float, t_high: float) -> list[int]:
    """A cross-section of the holdout, NOT cherry-picked successes: the point is
    to show each decision branch, including deferrals."""
    lots: list[int] = []

    def take(mask: pd.Series, n: int, by: pd.Series, ascending: bool) -> None:
        cand = by[mask & ~by.index.isin(lots)].sort_values(ascending=ascending)
        lots.extend(cand.head(n).index.to_list())

    take(p >= t_high, 3, p, ascending=False)                      # strongest FAIL flags
    take((p > t_low) & (p < t_high), 3, p, ascending=False)       # abstention band
    take(missing > missing.quantile(0.97), 1, missing, ascending=False)  # OOD / gappy lot
    take(p <= t_low, 3, p, ascending=True)                        # confident passes
    return lots


def main() -> None:
    art = joblib.load(OUT / "artifacts.joblib")
    model, prep = art["model"], art["prep"]
    t_low, t_high = art["t_low"], art["t_high"]

    X, y = load_raw()
    X_te = X.loc[art["test_index"]]
    y_te = y.loc[art["test_index"]]

    p = pd.Series(model.predict_proba(prep.transform(X_te))[:, 1], index=X_te.index)
    z = prep.z_scores(X_te)
    missing = prep.missing_frac(X_te)
    importances = pd.Series(model.feature_importances_, index=prep.features)

    rows = []
    for lot in pick_demo_lots(p, missing, t_low, t_high):
        d = decide(p[lot], z.loc[lot], float(missing[lot]), importances, t_low, t_high)
        note = explain(lot, d) if d.verdict != "PASS" else "—"
        truth = "FAIL" if y_te[lot] == 1 else "PASS"
        rows.append((lot, truth, f"{d.p_fail:.2f}", d.verdict, format_signals(d.top_signals[:3]), note))
        print(f"lot {lot:4d} | truth {truth} | p_fail {d.p_fail:.2f} | {d.verdict:9s} | {d.reason}")

    header = "| lot | ground truth | p_fail | decision | top deviating sensors | agent output |"
    sep = "|---|---|---|---|---|---|"
    body = "\n".join(
        "| " + " | ".join(str(c).replace("\n", " ").replace("|", "/") for c in r) + " |" for r in rows
    )
    md = (
        "# Demo run — agentic RCA on SECOM holdout lots\n\n"
        f"Gate: PASS below p={t_low:.2f}, FAIL-FLAG above p={t_high:.2f}, DEFER in between "
        f"or when out-of-distribution. Lots chosen to span all decision branches.\n\n"
        f"{header}\n{sep}\n{body}\n"
    )
    (OUT / "demo_run.md").write_text(md)
    print(f"\nwrote {OUT / 'demo_run.md'}")


if __name__ == "__main__":
    main()
