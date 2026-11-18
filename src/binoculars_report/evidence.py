"""Secondary evidence signals attached to flagged sentences: low burstiness and AI idioms.

Ported from clarity (https://github.com/roowus/clarity, MIT License, Copyright (c) 2026 roowus): `clarity.evidence`.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

# Phrases statistically overrepresented in LLM output; a stylistic signal only, never enough to flag a sentence alone.
AI_IDIOMS = [
    "delve into",
    "delves into",
    "it's important to note",
    "it is important to note",
    "in today's fast-paced world",
    "in the ever-evolving landscape",
    "tapestry of",
    "underscores the importance",
    "plays a crucial role",
    "plays a pivotal role",
    "a testament to",
    "navigate the complexities",
    "in conclusion,",
    "furthermore,",
    "moreover,",
    "harness the power",
    "unlock the potential",
    "seamlessly integrate",
]
IDIOM_PATTERN = re.compile("|".join(re.escape(p) for p in AI_IDIOMS), re.IGNORECASE)
BURSTINESS_MIN_SENTENCES = 3
BURSTINESS_MAX_STDEV = 0.15  # nats/token
BURSTINESS_MAX_MEAN = 2.5  # nats/token


@dataclass
class Signal:
    kind: str  # machine-readable, e.g. "low_binoculars"
    detail: str  # human-readable explanation with the numbers that fired


def sentence_signals(text: str, neighbor_log_ppls: list[float]) -> list[Signal]:
    """Signals for one sentence, given the mean log-perplexities of the sentence and its neighbors."""
    signals: list[Signal] = []
    if len(neighbor_log_ppls) >= BURSTINESS_MIN_SENTENCES:
        stdev = statistics.stdev(neighbor_log_ppls)
        if stdev < BURSTINESS_MAX_STDEV and statistics.fmean(neighbor_log_ppls) < BURSTINESS_MAX_MEAN:
            signals.append(Signal(
                kind="low_burstiness",
                detail=(
                    f"This sentence sits in a run of {len(neighbor_log_ppls)} sentences with near-identical predictability "
                    f"(σ={stdev:.2f}) — human writing varies more sentence-to-sentence."
                ),
            ))
    if match := IDIOM_PATTERN.search(text):
        signals.append(Signal(
            kind="ai_idiom",
            detail=f'Contains "{match.group(0)}", a phrase statistically overrepresented in LLM output (stylistic signal only).',
        ))
    return signals
