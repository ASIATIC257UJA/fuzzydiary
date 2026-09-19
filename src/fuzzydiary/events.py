from __future__ import annotations

from pathlib import Path

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
    day_data_map = series.days if series is not None else simplified.days
    return _detect(day_data_map, cfg)
