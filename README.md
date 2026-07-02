# Agentic RCA on Manufacturing Data — detect → explain → *know when to abstain*

A small, honest prototype of the loop a manufacturing-AI agent has to get right:
**detect a defect-risk anomaly in noisy production data → propose a root cause →
and, crucially, defer to a human when it isn't confident enough to act.**

Built on the public **SECOM** semiconductor dataset (UCI). No private data.

> Why this shape? The hard part of autonomous process-steering isn't flagging the
> obvious fails — it's not acting on the uncertain ones. This prototype makes the
> abstention decision a first-class output, not an afterthought.

## What it does

```
lot sensors ──▶ detector ──▶ p_fail ──┐
(590 signals)  (RandomForest)         ├─▶ abstention gate ──▶ FAIL-FLAG ─▶ LLM root-cause agent
                                      │   (confidence band  ─▶ PASS         (hypothesis + next check)
   raw sensor deviations ────────────┘    + OOD checks)    ─▶ DEFER ──────▶ "why I can't judge this"
```

- **Detector** — RandomForest with balanced class weights on 442 usable sensors
  (of 590; the rest are mostly-missing or constant). Reported with **PR-AUC**, the
  metric that's honest under 6.6% class imbalance — not accuracy.
- **Abstention gate** — three verdicts. `FAIL-FLAG` above a precision-calibrated
  threshold, `PASS` below a high-recall threshold, and `DEFER` for (a) the uncertain
  middle band and (b) out-of-distribution lots — too many missing sensors, or sensors
  beyond ±6σ of the training envelope. Thresholds are picked on **cross-validated
  training** probabilities, never on the holdout.
- **Root-cause agent** — for flagged/deferred lots, ranks the most-deviating sensors
  (deviation × model importance) and an LLM (Groq `llama-3.3-70b`) writes a plain-language
  hypothesis + the single most informative next check. Falls back to a deterministic
  template if no API key, so the demo always runs.

## Results (holdout, 392 lots, seed 42)

| metric | value | note |
|---|---|---|
| Fail prevalence | 6.6% | heavily imbalanced |
| PR-AUC (holdout) | **0.247** | vs **0.066** random baseline → 3.7× lift |
| ROC-AUC (holdout) | 0.825 | |

SECOM is a genuinely hard dataset — these are realistic numbers, not a polished-up
demo score. The contribution here is the *decision layer around* the detector, not a
state-of-the-art detector.

See [`outputs/demo_run.md`](outputs/demo_run.md) for a run spanning every decision
branch — confident passes, a flagged fail, uncertain-band deferrals, and an
out-of-distribution lot (8 sensors past 6σ) the model correctly refuses to judge.

## Run it

```bash
uv sync
uv run python -m rca.train   # trains detector, calibrates thresholds, writes metrics
uv run python -m rca.demo    # end-to-end demo over holdout lots -> outputs/demo_run.md
```

The SECOM data auto-downloads from UCI on first run. For LLM root-cause text set
`GROQ_API_KEY` (optional — a template fallback runs otherwise).

## What this deliberately does NOT solve (the thesis territory)

- **Calibration of the abstention band** — right now the thresholds are picked from
  a PR curve; the real question is *calibrated* uncertainty (does "p_fail = 0.22"
  mean a 22% failure rate?) and cost-aware thresholds tied to scrap vs inspection cost.
- **Grounded root causes** — the LLM reasons over anonymized sensor IDs. On real
  ERP/MES data it could be grounded in actual process steps and equipment.
- **Closing the loop** — this recommends; it doesn't steer the process. Autonomous
  action needs the reliability guarantees above first.

That gap — *how reliable does agentic root-cause analysis have to be before you let
it act autonomously on noisy production data, and when should it defer?* — is exactly
the master's-thesis question this prototype is meant to open.

## Tech
Python, scikit-learn, pandas; Groq (`llama-3.3-70b`) for the root-cause agent, with an
offline template fallback. Dataset: [SECOM (UCI)](https://archive.ics.uci.edu/dataset/179/secom).
