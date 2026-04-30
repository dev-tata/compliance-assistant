from __future__ import annotations

import math


def normalize_retrieval_score(score: float) -> float:
    bounded_score = max(min(float(score), 60.0), -60.0)
    normalized = 1.0 / (1.0 + math.exp(-bounded_score))
    return min(1.0, max(0.0, normalized))
