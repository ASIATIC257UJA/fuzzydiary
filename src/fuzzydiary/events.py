from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fuzzydiary.config import FuzzyDiaryConfig, load_config
from fuzzydiary.pattern_events import (
    Event,
    EventCollection,
    detect_events as _detect,
)
from fuzzydiary.simplify import SimplifiedSeries

__all__ = ["Event", "EventCollection", "detect_events"]


def detect_events(
    simplified: SimplifiedSeries,
    config: FuzzyDiaryConfig | str | Path,
    series=None,
) -> EventCollection:
    cfg = config if isinstance(config, FuzzyDiaryConfig) else load_config(config)
    return _detect(_segmented_days(simplified, series), cfg)


def _segmented_days(
    simplified: SimplifiedSeries,
    series=None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for day, vertices in simplified.days.items():
        original = series.days.get(day) if series is not None else None
        if original is None or len(vertices) < 2:
            out[day] = vertices
            continue
        grid = original.index
        values = np.interp(
            grid.asi8.astype(float),
            vertices.index.asi8.astype(float),
            vertices["value"].to_numpy(dtype=float),
        )
        values[original["value"].isna().to_numpy()] = np.nan
        out[day] = pd.DataFrame({"value": values}, index=grid)
    return out
