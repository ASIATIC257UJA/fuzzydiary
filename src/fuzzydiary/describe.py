from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fuzzydiary.aggregation_fuzzy import AggregationResult, aggregate_labels
from fuzzydiary.config import FuzzyDiaryConfig, ProtoformTemplate, load_config
from fuzzydiary.context import label_event_with_contexts
from fuzzydiary.events import Event, EventCollection
from fuzzydiary.fuzzy import membership_degree
from fuzzydiary.io import Series
from fuzzydiary.partition import partition_day
from fuzzydiary.simplify import SimplifiedSeries
from fuzzydiary.trend import build_segments


@dataclass
class Statement:
    text: str
    template: str
    truth: float
    simplicity: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventDescription:
    event: Event
    characteristic_level: str | None
    characteristic_md: float
    interval_aggregation: AggregationResult
    contexts: dict[str, tuple[str, float]] = field(default_factory=dict)
    statements: list[Statement] = field(default_factory=list)


SegmentDescription = EventDescription


@dataclass
class DailySummary:
    day: pd.Timestamp
    events: list[EventDescription] = field(default_factory=list)
    day_protoforms: list[Statement] = field(default_factory=list)
    narrative: str = ""

    @property
    def segments(self) -> list[EventDescription]:
        return self.events

    @property
    def n_segments(self) -> int:
        return len(self.events)

    @property
    def all_statements(self) -> list[Statement]:
        per_event = [s for ev in self.events for s in ev.statements]
        return per_event + self.day_protoforms


@dataclass
class DailySummaryCollection:
    days: dict[pd.Timestamp, DailySummary]

    @property
    def n_days(self) -> int:
        return len(self.days)


def describe(
    series: Series,
    events: EventCollection,
    config: FuzzyDiaryConfig | str | Path,
    simplified: SimplifiedSeries | None = None,
    acceptance_threshold: float = 0.5,
    dot_threshold: float = 0.7,
) -> DailySummaryCollection:
    cfg = config if isinstance(config, FuzzyDiaryConfig) else load_config(config)

    out: dict[pd.Timestamp, DailySummary] = {}
    for day, day_events in events.days.items():
        day_data = series.days.get(day)
        day_rdp = simplified.days.get(day) if simplified else None
        segments_geom = build_segments(day_rdp, cfg.trend) if day_rdp is not None else []

        partition = partition_day(
            day, day_rdp, day_events, segments_geom,
            describe_lines=cfg.events.describe_lines,
        )
        event_descs: list[EventDescription] = []
        for elem in partition:
            event_descs.append(
                _describe_event(elem, day_data, cfg, acceptance_threshold, dot_threshold)
            )

        day_protoforms = _describe_day_protoforms(day_data, cfg, dot_threshold)

        out[day] = DailySummary(day=day, events=event_descs, day_protoforms=day_protoforms)

    return DailySummaryCollection(days=out)


def _describe_event(
    elem: Event,
    day_data: pd.DataFrame | None,
    cfg: FuzzyDiaryConfig,
    acceptance_threshold: float,
    dot_threshold: float,
) -> EventDescription:
    char_level = elem.attributes.get("level")
    char_md = float(elem.attributes.get("level_md", 0.0))
    if char_level is None:
        char_value = elem.attributes.get(
            "characteristic_value",
            elem.attributes.get("apex_value",
            elem.attributes.get("nadir_value")),
        )
        if char_value is not None:
            char_level, char_md = _dominant_level(float(char_value), cfg)

    sample_memberships = _sample_memberships(elem, day_data, cfg)
    agg = aggregate_labels(
        sample_memberships=sample_memberships,
        variable=cfg.linguistic_variable,
        quantifiers=cfg.quantifiers,
        acceptance_threshold=acceptance_threshold,
    )

    contexts = label_event_with_contexts(elem, cfg.contexts)
    statements = _instantiate_event_statements(
        elem, char_level, agg, contexts, cfg, dot_threshold
    )

    return EventDescription(
        event=elem,
        characteristic_level=char_level,
        characteristic_md=char_md,
        interval_aggregation=agg,
        contexts=contexts,
        statements=statements,
    )


def _dominant_level(value: float, cfg: FuzzyDiaryConfig) -> tuple[str, float]:
    best_term, best_md = "", -1.0
    for term in cfg.linguistic_variable.terms:
        md = membership_degree(value, cfg.linguistic_variable.mf_as_dict(term))
        if md > best_md:
            best_term, best_md = term, md
    return best_term, max(best_md, 0.0)


def _sample_memberships(
    elem: Event,
    day_data: pd.DataFrame | None,
    cfg: FuzzyDiaryConfig,
) -> list[dict[str, float]]:
    if day_data is None or day_data.empty:
        return []
    mask = (day_data.index >= elem.start) & (day_data.index <= elem.end)
    slice_ = day_data.loc[mask]
    if slice_.empty:
        return []

    out: list[dict[str, float]] = []
    for value in slice_["value"].to_numpy(dtype=float):
        if pd.isna(value):
            continue
        memberships = {
            term: membership_degree(value, cfg.linguistic_variable.mf_as_dict(term))
            for term in cfg.linguistic_variable.terms
        }
        out.append(memberships)
    return out


def _instantiate_event_statements(
    elem: Event,
    char_level: str | None,
    agg: AggregationResult,
    contexts: dict[str, tuple[str, float]],
    cfg: FuzzyDiaryConfig,
    dot_threshold: float,
) -> list[Statement]:
    statements: list[Statement] = []

    if char_level is not None:
        char_time = elem.attributes.get("characteristic_time")
        if char_time is None:
            char_time = elem.start + (elem.end - elem.start) / 2
        ctx_phrase = ""
        if "day_moment" in contexts:
            ctx_phrase = f" in the {contexts['day_moment'][0]}"
        text = (
            f"{elem.kind}{ctx_phrase} "
            f"({char_level} at {char_time.strftime('%H:%M')})"
        )
        statements.append(
            Statement(
                text=text,
                template="<event-characteristic>",
                truth=float(elem.attributes.get("level_md", 1.0)),
                simplicity=1.0,
                evidence={
                    "kind": elem.kind,
                    "start": elem.start,
                    "end": elem.end,
                    "characteristic_level": char_level,
                },
            )
        )

    if agg.quantifier is not None:
        dot = float(agg.quantifier_md)
        if dot >= dot_threshold:
            placeholders: dict[str, str] = {
                "Q": agg.quantifier,
                "A": agg.dominant_term,
                "B": agg.dominant_term,
                "kind": elem.kind,
                "level": str(char_level or agg.dominant_term),
            }
            for ctx_name, (ctx_term, _md) in contexts.items():
                placeholders[ctx_name] = ctx_term

            ctx_names = {dim.name for dim in cfg.contexts}
            for tmpl in cfg.protoforms:
                if any(("{" + name + "}") in tmpl.template for name in ctx_names):
                    continue  # handled at day level
                try:
                    rendered = tmpl.template.format(**placeholders)
                except KeyError:
                    continue
                n_ph = tmpl.template.count("{")
                statements.append(
                    Statement(
                        text=rendered,
                        template=tmpl.template,
                        truth=dot,
                        simplicity=max(0.0, 1.0 - 0.1 * n_ph),
                        evidence={
                            "kind": elem.kind,
                            "start": elem.start,
                            "end": elem.end,
                            "dominant_term": agg.dominant_term,
                            "quantifier": agg.quantifier,
                            "scope": "event_interval",
                        },
                    )
                )
    return statements


def _describe_day_protoforms(
    day_data: pd.DataFrame | None,
    cfg: FuzzyDiaryConfig,
    dot_threshold: float,
) -> list[Statement]:
    if day_data is None or day_data.empty or not cfg.contexts:
        return []

    hours = day_data.index.hour + day_data.index.minute / 60.0
    values = day_data["value"].to_numpy(dtype=float)

    statements: list[Statement] = []

    for ctx_dim in cfg.contexts:
        ctx_templates = [
            t for t in cfg.protoforms if "{" + ctx_dim.name + "}" in t.template
        ]
        if not ctx_templates:
            continue

        for r_term in ctx_dim.terms:
            r_md = [membership_degree(h, ctx_dim.mf_as_dict(r_term)) for h in hours]
            r_sum = sum(r_md)
            if r_sum <= 0:
                continue

            for a_term in cfg.linguistic_variable.terms:
                num = 0.0
                for md_r, val in zip(r_md, values):
                    if pd.isna(val) or md_r <= 0:
                        continue
                    a_md = membership_degree(float(val), cfg.linguistic_variable.mf_as_dict(a_term))
                    num += min(md_r, a_md)
                ratio = num / r_sum

                q_name, q_md = _select_quantifier(ratio, cfg)
                if q_name is None or q_md < dot_threshold:
                    continue

                placeholders = {
                    "Q": q_name,
                    "A": a_term,
                    "B": a_term,
                    ctx_dim.name: r_term,
                }
                for tmpl in ctx_templates:
                    try:
                        rendered = tmpl.template.format(**placeholders)
                    except KeyError:
                        continue
                    statements.append(
                        Statement(
                            text=rendered,
                            template=tmpl.template,
                            truth=q_md,
                            simplicity=max(0.0, 1.0 - 0.1 * tmpl.template.count("{")),
                            evidence={
                                "scope": "day",
                                "qualifier": r_term,
                                "term": a_term,
                                "quantifier": q_name,
                                "ratio": ratio,
                            },
                        )
                    )
    return statements


def _select_quantifier(ratio: float, cfg: FuzzyDiaryConfig) -> tuple[str | None, float]:
    best_name, best_md = None, -1.0
    for q in cfg.quantifiers:
        md = membership_degree(ratio, q.membership_as_dict())
        if md > best_md:
            best_name, best_md = q.name, md
    return best_name, max(best_md, 0.0)
