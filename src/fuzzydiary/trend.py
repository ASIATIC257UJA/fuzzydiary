from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fuzzydiary.config import LinguisticVariable
from fuzzydiary.fuzzy import membership_degree


@dataclass
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp
    start_value: float
    end_value: float
    slope_normalised: float
    trend_label: str
    trend_md: float

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    @property
    def amplitude(self) -> float:
        return self.end_value - self.start_value


def build_segments(
    day_rdp: pd.DataFrame,
    trend_variable: LinguisticVariable,
) -> list[Segment]:
    n = len(day_rdp)
    if n < 2:
        return []

    timestamps = day_rdp.index.to_list()
    values = day_rdp["value"].to_numpy(dtype=float)

    vmin, vmax = float(np.min(values)), float(np.max(values))
    vrange = max(vmax - vmin, 1e-9)
    v_norm = (values - vmin) / vrange

    total_span = (timestamps[-1] - timestamps[0]).total_seconds()
    if total_span <= 0:
        return []

    segments: list[Segment] = []
    for i in range(n - 1):
        t1, t2 = timestamps[i], timestamps[i + 1]
        dx = (t2 - t1).total_seconds() / total_span
        if dx <= 0:
            continue
        dy = float(v_norm[i + 1] - v_norm[i])
        slope = float(np.clip(dy / dx, -1.0, 1.0))
        label, md = _classify(slope, trend_variable)
        segments.append(
            Segment(
                start=t1, end=t2,
                start_value=float(values[i]),
                end_value=float(values[i + 1]),
                slope_normalised=slope,
                trend_label=label,
                trend_md=md,
            )
        )
    return segments


def _classify(slope: float, variable: LinguisticVariable) -> tuple[str, float]:
    best_term, best_md = "", -1.0
    for term in variable.terms:
        md = membership_degree(slope, variable.mf_as_dict(term))
        if md > best_md:
            best_term, best_md = term, md
    return best_term, best_md
