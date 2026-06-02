from __future__ import annotations

import pandas as pd

from fuzzydiary.config import ContextDimension
from fuzzydiary.events import Event
from fuzzydiary.fuzzy import membership_degree


def label_event_with_contexts(
    event: Event,
    contexts: list[ContextDimension],
) -> dict[str, tuple[str, float]]:
    if not contexts:
        return {}
    midpoint = _midpoint(event)
    hour = midpoint.hour + midpoint.minute / 60.0 + midpoint.second / 3600.0
    out: dict[str, tuple[str, float]] = {}
    for dim in contexts:
        best_term, best_md = "", -1.0
        for term in dim.terms:
            md = membership_degree(hour, dim.mf_as_dict(term))
            if md > best_md:
                best_term, best_md = term, md
        out[dim.name] = (best_term, best_md)
    return out


def _midpoint(event: Event) -> pd.Timestamp:
    span = (event.end - event.start).total_seconds()
    return event.start + pd.Timedelta(seconds=span / 2.0)
