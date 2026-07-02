"""Export real prototype results to JSON for the portfolio site.

Run: uv run python -m rca.export_site
Writes outputs/site_data.json — aggregate decision counts over the full holdout,
plus a curated set of lots (spanning every decision branch) WITH their LLM
root-cause text. The site ships this file; no backend needed at view time.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import load_raw
from .demo import pick_demo_lots
from .explain import explain
from .gate import decide

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    art = joblib.load(OUT / "artifacts.joblib")
    model, prep = art["model"], art["prep"]
    t_low, t_high = art["t_low"], art["t_high"]
    metrics = json.loads((OUT / "metrics.json").read_text())

    X, y = load_raw()
    X_te, y_te = X.loc[art["test_index"]], y.loc[art["test_index"]]
    p = pd.Series(model.predict_proba(prep.transform(X_te))[:, 1], index=X_te.index)
    z = prep.z_scores(X_te)
    missing = prep.missing_frac(X_te)
    importances = pd.Series(model.feature_importances_, index=prep.features)

    # aggregate: how the gate splits the entire holdout
    counts = {"FAIL-FLAG": 0, "PASS": 0, "DEFER": 0}
    for lot in X_te.index:
        d = decide(p[lot], z.loc[lot], float(missing[lot]), importances, t_low, t_high)
        counts[d.verdict] += 1

    # curated lots (with LLM explanation) for the interactive table
    lots = []
    for lot in pick_demo_lots(p, missing, t_low, t_high):
        d = decide(p[lot], z.loc[lot], float(missing[lot]), importances, t_low, t_high)
        lots.append({
            "lot": int(lot),
            "truth": "FAIL" if y_te[lot] == 1 else "PASS",
            "p_fail": round(d.p_fail, 3),
            "verdict": d.verdict,
            "reason": d.reason,
            "signals": [
                {"sensor": s, "z": (None if pd.isna(zv) else round(zv, 1))}
                for s, zv in d.top_signals
            ],
            "agent": None if d.verdict == "PASS" else explain(lot, d),
        })

    data = {
        "metrics": metrics,
        "thresholds": {"t_low": round(t_low, 3), "t_high": round(t_high, 3)},
        "holdout_decision_counts": counts,
        "lots": lots,
    }
    (OUT / "site_data.json").write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT / 'site_data.json'}  ({len(lots)} demo lots, holdout split {counts})")


if __name__ == "__main__":
    main()
