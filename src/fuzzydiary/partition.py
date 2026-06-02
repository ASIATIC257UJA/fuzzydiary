from __future__ import annotations

import pandas as pd

from fuzzydiary.events import Event
from fuzzydiary.trend import Segment


def partition_day(
    day: pd.Timestamp,
    day_rdp: pd.DataFrame,
    events: list[Event],
    segments: list[Segment],
    describe_lines: bool = False,
) -> list[Event]:
    atomic = [e for e in events if not e.composite]
    composites = [e for e in events if e.composite]

    if not describe_lines:
        return sorted(atomic + composites, key=lambda e: (e.start, e.end))

    if day_rdp is None or day_rdp.empty or not segments:
        return sorted(atomic + composites, key=lambda e: (e.start, e.end))

    atomic_sorted = sorted(atomic, key=lambda e: e.start)
    day_start = day_rdp.index[0]
    day_end = day_rdp.index[-1]

    result: list[Event] = []
    cursor = day_start
    for ev in atomic_sorted:
        if ev.start > cursor:
            line = _make_line(cursor, ev.start, segments)
            if line is not None:
                result.append(line)
        result.append(ev)
        cursor = max(cursor, ev.end)

    if cursor < day_end:
        line = _make_line(cursor, day_end, segments)
        if line is not None:
            result.append(line)

    result.extend(composites)
    return sorted(result, key=lambda e: (e.start, e.end))


def _make_line(
    start: pd.Timestamp,
    end: pd.Timestamp,
    segments: list[Segment],
) -> Event | None:
    if end <= start:
        return None
    covered = [s for s in segments if s.start < end and s.end > start]
    if not covered:
        return None

    duration = (end - start).total_seconds() / 60.0
    amplitude = covered[-1].end_value - covered[0].start_value

    by_label: dict[str, float] = {}
    for s in covered:
        dur = (min(s.end, end) - max(s.start, start)).total_seconds() / 60.0
        if dur <= 0:
            continue
        by_label[s.trend_label] = by_label.get(s.trend_label, 0.0) + dur
    dominant = max(by_label.items(), key=lambda kv: kv[1])[0] if by_label else "steady"

    return Event(
        kind="line",
        start=start,
        end=end,
        attributes={
            "duration_minutes": duration,
            "amplitude": amplitude,
            "dominant_trend": dominant,
        },
    )
