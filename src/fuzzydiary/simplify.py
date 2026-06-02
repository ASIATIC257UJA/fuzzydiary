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


@dataclass
class SimplifiedSeries:
    """
    Output of `simplify`: per-day piecewise-linear approximation of the
    original signal. Each day maps to a DataFrame holding the surviving
    breakpoints (timestamp + value) selected by RDP.
    """

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
    """
    Reduce each daily window of `series` to a compact piecewise-linear
    approximation using the Ramer-Douglas-Peucker algorithm.

    Parameters
    ----------
    series : Series
        Output of `load_series`.
    epsilon : float or 'auto'
        RDP tolerance on the normalised representation (values in [0, 1] on
        both axes). When 'auto', for each day epsilon is set to
        `auto_std_fraction * (std/range)`, clipped to a sensible minimum.
        Typical good values: 0.01 - 0.05.
    auto_std_fraction : float
        Multiplier used only when `epsilon == 'auto'`.

    Returns
    -------
    SimplifiedSeries
    """
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
    """Return the RDP tolerance to use for one day."""
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
    """
    Apply RDP to a single day, handling NaN runs gracefully.

    The day is split at NaN gaps; each contiguous non-NaN run is simplified
    independently, and the surviving breakpoints are concatenated.
    """
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
    """Split a daily DataFrame into contiguous non-NaN runs."""
    values = day_df["value"]
    valid = values.notna().to_numpy()
    if not valid.any():
        return []

    edges = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive

    runs: list[pd.DataFrame] = []
    for s, e in zip(starts, ends):
        if e - s >= 2:  # need at least two points for RDP
            runs.append(day_df.iloc[s:e])
        elif e - s == 1:
            runs.append(day_df.iloc[s:e])
    return runs


def _rdp_run(run: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Apply RDP to a single non-NaN run and return surviving breakpoints."""
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
        mask = rdp(points, epsilon=eps, return_mask=True)

    kept = run.iloc[np.where(mask)[0]].copy()
    return kept
