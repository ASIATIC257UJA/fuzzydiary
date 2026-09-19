from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from rdp import rdp

from fuzzydiary.io import Series


def _pldist2d(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    ux, uy = end[0] - start[0], end[1] - start[1]
    vx, vy = start[0] - point[0], start[1] - point[1]
    return abs(ux * vy - uy * vx) / np.hypot(ux, uy)


@dataclass
class SimplifiedSeries:
    days: dict[pd.Timestamp, pd.DataFrame]
    epsilon_used: dict[pd.Timestamp, float]

    @property
    def n_days(self) -> int:
        return len(self.days)

    def __len__(self) -> int:
        return self.n_days



def simplify(
    series: Series,
    epsilon: float | Literal["auto"] = "auto",
    auto_std_fraction: float = 0.2,
) -> SimplifiedSeries:
    if not isinstance(series, Series):
        raise TypeError(f"Expected a Series, got {type(series).__name__}.")
    if epsilon != "auto" and not isinstance(epsilon, (int, float)):
        raise TypeError(f"epsilon must be a float or 'auto', got {epsilon!r}.")
    if epsilon != "auto" and epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")

    out_days: dict[pd.Timestamp, pd.DataFrame] = {}
    used_eps: dict[pd.Timestamp, float] = {}

    for day, day_df in series.days.items():
        eps = _resolve_epsilon(day_df, epsilon, auto_std_fraction)
        used_eps[day] = eps
        out_days[day] = _simplify_one_day(day_df, eps)

    return SimplifiedSeries(days=out_days, epsilon_used=used_eps)



_EPSILON_FLOOR: float = 1e-4


def _resolve_epsilon(
    day_df: pd.DataFrame,
    epsilon: float | Literal["auto"],
    auto_std_fraction: float,
) -> float:
    if epsilon != "auto":
        return float(epsilon)

    values = day_df["value"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return _EPSILON_FLOOR

    vrange = float(np.nanmax(finite) - np.nanmin(finite))
    if vrange <= 0:
        return _EPSILON_FLOOR

    eps = float(auto_std_fraction * (np.nanstd(finite) / vrange))
    return max(eps, _EPSILON_FLOOR)


def _simplify_one_day(day_df: pd.DataFrame, eps: float) -> pd.DataFrame:
    if day_df.empty:
        return day_df.iloc[0:0].copy()

    runs = _split_on_nan(day_df)
    if not runs:
        return day_df.iloc[0:0].copy()

    pieces: list[pd.DataFrame] = []
    for run in runs:
        pieces.append(_rdp_run(run, eps))

    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _split_on_nan(day_df: pd.DataFrame) -> list[pd.DataFrame]:
    values = day_df["value"]
    valid = values.notna().to_numpy()
    if not valid.any():
        return []

    edges = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    runs: list[pd.DataFrame] = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            runs.append(day_df.iloc[s:e])
        elif e - s == 1:
            runs.append(day_df.iloc[s:e])
    return runs


def _rdp_run(run: pd.DataFrame, eps: float) -> pd.DataFrame:
    n = len(run)
    if n <= 2:
        return run.copy()

    values = run["value"].to_numpy(dtype=float)

    x_idx = np.arange(n, dtype=float)
    x_norm = x_idx / (n - 1)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = vmax - vmin
    if vrange <= 0:
        return run.iloc[[0, -1]].copy()
    y_norm = (values - vmin) / vrange

    points = np.column_stack([x_norm, y_norm])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        mask = rdp(points, epsilon=eps, dist=_pldist2d, return_mask=True)

    kept = run.iloc[np.where(mask)[0]].copy()
    return kept
