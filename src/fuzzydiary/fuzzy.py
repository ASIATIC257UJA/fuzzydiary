from __future__ import annotations

import math
from typing import Sequence


def membership_degree(value: float, mf: dict | Sequence[float]) -> float:
    if isinstance(mf, dict):
        return _dispatch(value, mf)
    if len(mf) == 4:
        return _trapezoid(value, mf)
    raise ValueError(f"Cannot infer membership function type from sequence of length {len(mf)}.")


def _dispatch(value: float, mf: dict) -> float:
    kind = mf.get("type", "trapezoid")
    if kind == "trapezoid":
        return _trapezoid(value, (mf["a"], mf["b"], mf["c"], mf["d"]))
    if kind == "triangle":
        return _triangle(value, mf["a"], mf["b"], mf["c"])
    if kind == "gaussian":
        return _gaussian(value, mf["mean"], mf["sigma"])
    if kind == "singleton":
        return _singleton(value, mf["value"], mf.get("tolerance", 0.0))
    raise ValueError(f"Unknown membership function type: {kind!r}")


def _trapezoid(value: float, params: Sequence[float]) -> float:
    a, b, c, d = map(float, params)
    if not (a <= b <= c <= d):
        raise ValueError(f"Trapezoid breakpoints must be non-decreasing; got {params}.")
    if value < a or value > d:
        return 0.0
    if b <= value <= c:
        return 1.0
    if a <= value < b:
        return 1.0 if b == a else (value - a) / (b - a)
    return 1.0 if d == c else (d - value) / (d - c)


def _triangle(value: float, a: float, b: float, c: float) -> float:
    if not (a <= b <= c):
        raise ValueError(f"Triangle breakpoints must satisfy a<=b<=c; got {(a,b,c)}.")
    if value <= a or value >= c:
        return 0.0
    if value == b:
        return 1.0
    if value < b:
        return (value - a) / (b - a) if b != a else 1.0
    return (c - value) / (c - b) if c != b else 1.0


def _gaussian(value: float, mean: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError(f"Gaussian sigma must be > 0; got {sigma}.")
    return math.exp(-0.5 * ((value - mean) / sigma) ** 2)


def _singleton(value: float, centre: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 1.0 if value == centre else 0.0
    diff = abs(value - centre)
    return max(0.0, 1.0 - diff / tolerance)
