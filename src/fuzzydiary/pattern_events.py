from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fuzzydiary.config import (
    EventDef,
    FuzzyDiaryConfig,
    LevelCondition,
    LinguisticVariable,
)
from fuzzydiary.fuzzy import membership_degree

_DOWNWARD = {"very_low", "low"}
_UPWARD   = {"high", "very_high"}


@dataclass
class Event:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    attributes: dict[str, Any] = field(default_factory=dict)
    composite: bool = False
    components: list["Event"] | None = None

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


@dataclass
class EventCollection:
    days: dict[pd.Timestamp, list[Event]]

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def n_events(self) -> int:
        return sum(len(evs) for evs in self.days.values())

    def by_kind(self, kind: str) -> list[Event]:
        return [e for evs in self.days.values() for e in evs if e.kind == kind]


def detect_events(
    day_data_map: dict[pd.Timestamp, pd.DataFrame],
    cfg: FuzzyDiaryConfig,
) -> EventCollection:
    enabled = [e for e in cfg.events.catalogue if e.enabled]
    simple_defs    = [e for e in enabled if not e.is_composite and not e.is_anchored]
    anchored_defs  = [e for e in enabled if e.is_anchored]
    composite_defs = [e for e in enabled if e.is_composite]
    max_dur = {e.label: e.max_duration_minutes for e in enabled}

    out: dict[pd.Timestamp, list[Event]] = {}
    for day, day_data in day_data_map.items():
        labels = _label_samples(day_data, cfg.linguistic_variable)

        simple    = _detect_simple(day_data, labels, simple_defs)
        composite = _detect_composite(simple, composite_defs)

        pre_anchored = sorted(simple + composite, key=lambda e: (e.start, e.end))
        pre_anchored = [
            e for e in pre_anchored
            if not (max_dur.get(e.kind, 0) > 0 and e.duration_minutes > max_dur[e.kind])
        ]
        pre_anchored = _suppress_overlapping(pre_anchored)

        anchored = _detect_anchored(day_data, labels, pre_anchored, [], anchored_defs)

        all_events = _suppress_overlapping(
            sorted(pre_anchored + anchored, key=lambda e: (e.start, e.end))
        )
        out[day] = all_events

    return EventCollection(days=out)


def _label_samples(
    day_data: pd.DataFrame,
    variable: LinguisticVariable,
) -> list[str]:
    labels: list[str] = []
    for v in day_data["value"].to_numpy(dtype=float):
        if np.isnan(v):
            labels.append("__nan__")
            continue
        best, best_md = "__nan__", -1.0
        for term in variable.terms:
            md = membership_degree(v, variable.mf_as_dict(term))
            if md > best_md:
                best, best_md = term, md
        labels.append(best)
    return labels


def _detect_simple(
    day_data: pd.DataFrame,
    labels: list[str],
    defs: list[EventDef],
) -> list[Event]:
    events: list[Event] = []
    for edef in defs:
        events.extend(_detect_one_simple(day_data, labels, edef))
    return events


def _detect_one_simple(
    day_data: pd.DataFrame,
    labels: list[str],
    edef: EventDef,
) -> list[Event]:
    steps = edef.pattern
    step_runs = [_find_runs(day_data, labels, step.level) for step in steps]

    candidates: list[Event] = []
    if len(steps) == 1:
        step = steps[0]
        for run in step_runs[0]:
            dur = (run["end"] - run["start"]).total_seconds() / 60.0
            if dur < step.min_duration_minutes:
                continue
            ev = _run_to_event(run, edef.label)
            ev.attributes["_min_dur"] = edef.min_duration_minutes
            candidates.append(ev)
    else:
        candidates = _match_sequence(step_runs, steps, edef)

    if edef.merge_gap_minutes > 0:
        candidates = _merge(candidates, edef.merge_gap_minutes, edef.label)

    return [ev for ev in candidates if ev.duration_minutes >= edef.min_duration_minutes]


def _find_runs(
    day_data: pd.DataFrame,
    labels: list[str],
    accepted: list[str],
) -> list[dict]:
    accepted_set = set(accepted)
    idx = day_data.index
    values = day_data["value"].to_numpy(dtype=float)
    runs: list[dict] = []
    start_i = None
    for i, lbl in enumerate(labels):
        in_run = lbl in accepted_set
        if in_run and start_i is None:
            start_i = i
        elif not in_run and start_i is not None:
            runs.append(_build_run(idx, values, start_i, i - 1, accepted))
            start_i = None
    if start_i is not None:
        runs.append(_build_run(idx, values, start_i, len(labels) - 1, accepted))
    return runs


def _build_run(idx, values, start_i: int, end_i: int, accepted: list[str]) -> dict:
    seg_vals = values[start_i : end_i + 1]
    seg_idx  = idx[start_i : end_i + 1]
    accepted_set = set(accepted)
    if accepted_set <= _DOWNWARD:
        ci = int(np.argmin(seg_vals))
    elif accepted_set <= _UPWARD:
        ci = int(np.argmax(seg_vals))
    else:
        ci = len(seg_vals) // 2
    return {
        "start": seg_idx[0],
        "end":   seg_idx[-1],
        "values": seg_vals,
        "times":  seg_idx,
        "characteristic_time":  seg_idx[ci],
        "characteristic_value": float(seg_vals[ci]),
        "n": end_i - start_i + 1,
    }


def _run_to_event(run: dict, label: str) -> Event:
    return Event(
        kind=label,
        start=run["start"],
        end=run["end"],
        attributes={
            "characteristic_time":  run["characteristic_time"],
            "characteristic_value": run["characteristic_value"],
            "duration_minutes": (run["end"] - run["start"]).total_seconds() / 60.0,
            "n_samples": run["n"],
        },
    )


def _match_sequence(
    step_runs: list[list[dict]],
    steps: list[LevelCondition],
    edef: EventDef,
) -> list[Event]:
    events: list[Event] = []
    for run0 in step_runs[0]:
        if (run0["end"] - run0["start"]).total_seconds() / 60.0 < steps[0].min_duration_minutes:
            continue
        chain = [run0]
        ok = True
        for s_idx in range(1, len(steps)):
            step = steps[s_idx]
            nxt = None
            for run in step_runs[s_idx]:
                if run["start"] <= chain[-1]["end"]:
                    continue
                gap = (run["start"] - chain[-1]["end"]).total_seconds() / 60.0
                if gap <= edef.max_gap_minutes:
                    if (run["end"] - run["start"]).total_seconds() / 60.0 >= step.min_duration_minutes:
                        nxt = run
                        break
            if nxt is None:
                ok = False
                break
            chain.append(nxt)

        if ok:
            all_vals = np.concatenate([r["values"] for r in chain])
            accepted_flat = [l for step in steps for l in step.level]
            accepted_set = set(accepted_flat)
            if accepted_set <= _DOWNWARD:
                ci_val = float(np.min(all_vals))
                ci_time = chain[np.argmin([r["characteristic_value"] for r in chain])]["characteristic_time"]
            else:
                ci_val = float(np.max(all_vals))
                ci_time = chain[np.argmax([r["characteristic_value"] for r in chain])]["characteristic_time"]
            events.append(Event(
                kind=edef.label,
                start=chain[0]["start"],
                end=chain[-1]["end"],
                attributes={
                    "characteristic_time": ci_time,
                    "characteristic_value": ci_val,
                    "duration_minutes": (chain[-1]["end"] - chain[0]["start"]).total_seconds() / 60.0,
                    "n_steps": len(chain),
                    "_min_dur": edef.min_duration_minutes,
                },
            ))
    return events


def _merge(events: list[Event], gap_minutes: float, label: str) -> list[Event]:
    if not events:
        return events
    merged = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        gap = (ev.start - last.end).total_seconds() / 60.0
        if ev.kind == label and gap <= gap_minutes:
            last.end = ev.end
            last.attributes["duration_minutes"] = (
                (last.end - last.start).total_seconds() / 60.0
            )
            last.attributes["n_samples"] = (
                last.attributes.get("n_samples", 0) + ev.attributes.get("n_samples", 0)
            )
            cur_v = last.attributes.get("characteristic_value")
            new_v = ev.attributes.get("characteristic_value")
            if cur_v is not None and new_v is not None:
                if "hypo" in label or "low" in label:
                    take = new_v < cur_v
                else:
                    take = new_v > cur_v
                if take:
                    last.attributes["characteristic_value"] = new_v
                    last.attributes["characteristic_time"] = ev.attributes.get("characteristic_time")
        else:
            merged.append(ev)
    return merged


def _detect_anchored(
    day_data: pd.DataFrame,
    labels: list[str],
    simple: list[Event],
    composite: list[Event],
    defs: list[EventDef],
) -> list[Event]:
    if not defs:
        return []
    pool = sorted(simple + composite, key=lambda e: e.start)
    results: list[Event] = []
    for edef in defs:
        results.extend(_detect_one_anchored(day_data, labels, pool, edef))
    return results


def _detect_one_anchored(
    day_data: pd.DataFrame,
    labels: list[str],
    pool: list[Event],
    edef: EventDef,
) -> list[Event]:
    if edef.anchor_labels:
        level_steps = [s for s in edef.pattern if isinstance(s, LevelCondition)]
        anchors = [ev for ev in pool if ev.kind in edef.anchor_labels]
        results: list[Event] = []
        for anchor in anchors:
            results.extend(
                _anchored_level_events(day_data, labels, anchor, level_steps, edef)
            )
        return results

    first_level_idx = next(
        i for i, s in enumerate(edef.pattern) if isinstance(s, LevelCondition)
    )
    label_steps = edef.pattern[:first_level_idx]
    level_steps = edef.pattern[first_level_idx:]
    label_pattern = [s.label for s in label_steps]

    anchors: list[Event] = []
    for ev in pool:
        if ev.kind != label_pattern[0]:
            continue
        chain = [ev]
        for next_label in label_pattern[1:]:
            nxt = None
            for cand in pool:
                if cand.start <= chain[-1].end:
                    continue
                if cand.kind != next_label:
                    continue
                gap = (cand.start - chain[-1].end).total_seconds() / 60.0
                if gap <= edef.max_gap_minutes:
                    nxt = cand
                    break
                break
            if nxt is None:
                chain = []
                break
            chain.append(nxt)
        if chain and len(chain) == len(label_pattern):
            anchors.append(chain[-1])

    results: list[Event] = []
    for anchor in anchors:
        results.extend(
            _anchored_level_events(day_data, labels, anchor, level_steps, edef)
        )
    return results


def _anchored_level_events(
    day_data: pd.DataFrame,
    labels: list[str],
    anchor: Event,
    level_steps: list,
    edef: EventDef,
) -> list[Event]:
    anchor_end_idx = day_data.index.searchsorted(anchor.end, side="right")
    if anchor_end_idx >= len(day_data):
        return []

    slice_data   = day_data.iloc[anchor_end_idx:]
    slice_labels = labels[anchor_end_idx:]
    if slice_data.empty:
        return []

    step_runs = [
        _find_runs(slice_data, slice_labels, step.level)
        for step in level_steps
    ]
    if not all(step_runs):
        return []

    results: list[Event] = []
    if len(level_steps) == 1:
        step = level_steps[0]
        for run in step_runs[0]:
            gap = (run["start"] - anchor.end).total_seconds() / 60.0
            if gap > edef.max_gap_minutes:
                break
            dur = (run["end"] - run["start"]).total_seconds() / 60.0
            if dur < step.min_duration_minutes:
                continue
            if edef.min_duration_minutes > 0 and dur < edef.min_duration_minutes:
                continue
            results.append(Event(
                kind=edef.label,
                start=run["start"],
                end=run["end"],
                attributes={
                    "anchor_label": anchor.kind,
                    "anchor_end":   anchor.end,
                    "characteristic_time":  run["characteristic_time"],
                    "characteristic_value": run["characteristic_value"],
                    "duration_minutes": dur,
                    "_min_dur": edef.min_duration_minutes,
                },
            ))
            break
    else:
        matched = _match_sequence(step_runs, list(level_steps), edef)
        for ev in matched:
            gap = (ev.start - anchor.end).total_seconds() / 60.0
            if gap <= edef.max_gap_minutes:
                ev.attributes["anchor_label"] = anchor.kind
                ev.attributes["anchor_end"]   = anchor.end
                results.append(ev)
    return results


def _detect_composite(
    simple: list[Event],
    defs: list[EventDef],
) -> list[Event]:
    events: list[Event] = []
    by_time = sorted(simple, key=lambda e: e.start)
    for cdef in defs:
        events.extend(_detect_one_composite(by_time, cdef))
    return events


def _detect_one_composite(pool: list[Event], cdef: EventDef) -> list[Event]:
    pattern = [step.label for step in cdef.pattern]
    max_gap = cdef.max_gap_minutes
    results: list[Event] = []

    for ev in pool:
        if ev.kind != pattern[0]:
            continue
        chain = [ev]
        for next_label in pattern[1:]:
            nxt = None
            for cand in pool:
                if cand.start <= chain[-1].end:
                    continue
                if cand.kind != next_label:
                    continue
                gap = (cand.start - chain[-1].end).total_seconds() / 60.0
                if gap <= max_gap:
                    nxt = cand
                    break
                break
            if nxt is None:
                chain = []
                break
            chain.append(nxt)

        if not chain or len(chain) != len(pattern):
            continue
        dur = (chain[-1].end - chain[0].start).total_seconds() / 60.0
        if dur < cdef.min_duration_minutes:
            continue
        results.append(Event(
            kind=cdef.label,
            start=chain[0].start,
            end=chain[-1].end,
            attributes={"pattern": pattern},
            composite=True,
            components=chain,
        ))
    return results


def _suppress_overlapping(events: list[Event]) -> list[Event]:
    if len(events) < 2:
        return events

    by_dur = sorted(
        events,
        key=lambda e: (e.duration_minutes, e.attributes.get("_min_dur", 0)),
        reverse=True,
    )
    kept: list[Event] = []

    for ev in by_dur:
        dominated = False
        for other in kept:
            inter_start = max(ev.start, other.start)
            inter_end   = min(ev.end,   other.end)
            if inter_end <= inter_start:
                continue
            inter_min = (inter_end - inter_start).total_seconds() / 60.0
            shorter_min = min(ev.duration_minutes, other.duration_minutes)
            if shorter_min > 0 and inter_min / shorter_min > 0.5:
                dominated = True
                break
        if not dominated:
            kept.append(ev)

    return sorted(kept, key=lambda e: (e.start, e.end))
