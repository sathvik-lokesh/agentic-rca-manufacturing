"""LLM root-cause agent: turn a flagged/deferred lot into an engineer-readable
hypothesis + recommended next check.

Backend: Groq (llama-3.3-70b-versatile) if GROQ_API_KEY is set; otherwise a
deterministic template so the demo always runs offline. Calls are sequential —
Groq free tier is ~12k tokens/min, never parallelize.
"""
import os
import textwrap

import requests

from .gate import Decision, format_signals

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM = textwrap.dedent("""\
    You are a manufacturing process engineer's assistant reviewing semiconductor
    production lots (SECOM dataset: sensors are anonymized, so reason about
    signal PATTERNS, not physics you can't know). Be concise and concrete.
    Given a lot's defect-risk assessment and its most deviating sensors, write:
    1. Hypothesis: (1-2 sentences — what kind of process deviation this pattern suggests)
    2. Next check: (1 sentence — the single most informative thing to verify first)
    Never invent sensor meanings; refer to sensors by their IDs and deviations.""")


def _prompt(lot_id: int, d: Decision) -> str:
    return (
        f"Lot {lot_id}: verdict={d.verdict}, p_fail={d.p_fail:.2f} ({d.reason}). "
        f"Most deviating sensors vs normal production: {format_signals(d.top_signals)}."
    )


def _template_fallback(d: Decision) -> str:
    sig = format_signals(d.top_signals)
    if d.verdict == "DEFER":
        return (f"Hypothesis: not established — {d.reason}. "
                f"Next check: have an engineer review this lot manually, starting with {sig}.")
    return (f"Hypothesis: co-deviation of {sig} suggests a shared upstream process shift. "
            f"Next check: inspect the process step feeding these sensors on this lot's tool/chamber.")


def explain(lot_id: int, d: Decision) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return _template_fallback(d)
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0.3,
                "max_tokens": 180,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt(lot_id, d)},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # demo must not die on a rate limit
        return _template_fallback(d) + f"  [LLM unavailable: {type(e).__name__}]"
