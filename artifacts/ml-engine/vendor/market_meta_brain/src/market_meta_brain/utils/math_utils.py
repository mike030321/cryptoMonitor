from __future__ import annotations

import math


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def softmax_dict(values: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not values:
        return {}
    temperature = max(1e-6, temperature)
    max_v = max(values.values())
    exps = {k: math.exp((v - max_v) / temperature) for k, v in values.items()}
    total = sum(exps.values())
    if total <= 0.0:
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: v / total for k, v in exps.items()}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
