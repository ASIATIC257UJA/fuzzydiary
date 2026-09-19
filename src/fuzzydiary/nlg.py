from __future__ import annotations

from dataclasses import dataclass, field

from fuzzydiary.config import FuzzyDiaryConfig
from fuzzydiary.describe import DailySummary, Statement
from fuzzydiary.synthesis import (
    DaySynthesis,
    EventNode,
    LabelHierarchy,
    ScopeGroup,
    build_label_hierarchy,
    quantifier_order,
    synthesize_day,
    term_order,
)

_MAIN_ADVERBS = {
    "few": "occasionally",
    "some": "sometimes",
    "many": "often",
    "most": "mostly",
    "almost_all": "almost always",
    "all": "always",
}

_SUBORDINATE_ADVERBS = {
    "few": "occasionally",
    "some": "sometimes",
    "many": "at times",
    "most": "frequently",
    "almost_all": "almost always",
    "all": "throughout",
}

_NUMERALS = {
    2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

@dataclass
class Clause:
    scopes: list[str]
    main: Statement | None
    intensifiers: list[Statement] = field(default_factory=list)
    others: list[Statement] = field(default_factory=list)
    modifiers: list[Statement] = field(default_factory=list)
    events: list[EventNode] = field(default_factory=list)
    relation: str = "opening"

    @property
    def support(self) -> list[Statement]:
        out = [self.main] if self.main is not None else []
        return out + self.intensifiers + self.others + self.modifiers


@dataclass
class RealizedClause:
    text: str
    support: list[Statement] = field(default_factory=list)
    events: list[EventNode] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)


def lexicalize(label: str, kind: str = "term", cfg: FuzzyDiaryConfig | None = None) -> str:
    overrides: dict[str, str] = {}
    lexicon = getattr(cfg, "lexicon", None) if cfg is not None else None
    if lexicon is not None:
        overrides = getattr(lexicon, kind + "s", {}) or {}
    if label in overrides:
        return overrides[label]

    if kind == "quantifier":
        return _MAIN_ADVERBS.get(label, label.replace("_", " "))
    if kind == "subordinate":
        return _SUBORDINATE_ADVERBS.get(label, "at times")
    return label.replace("_", " ")

def plan_document(day: DaySynthesis, cfg: FuzzyDiaryConfig) -> list[Clause]:
    ranks = term_order(cfg.linguistic_variable)
    q_ranks = quantifier_order(cfg)
    hierarchy = build_label_hierarchy(cfg)

    clauses: list[Clause] = []
    previous_rank: int | None = None

    for group in sorted(day.groups, key=lambda g: g.order):
        main_node = group.roots[0] if group.roots else None
        main = main_node.statement if main_node else None

        relation = "opening"
        if clauses:
            rank = ranks.get(main_node.term) if main_node else None
            if rank is None or previous_rank is None or rank == previous_rank:
                relation = "continuation"
            elif rank > previous_rank:
                relation = "rise"
            else:
                relation = "fall"

        clauses.append(
            Clause(
                scopes=[group.qualifier],
                main=main,
                intensifiers=[child.statement for child in main_node.children] if main_node else [],
                others=[node.statement for node in group.roots[1:]],
                modifiers=_best_modifiers(
                    group, q_ranks, main_node.term if main_node else "", hierarchy
                ),
                events=group.events,
                relation=relation,
            )
        )
        if main_node is not None:
            previous_rank = ranks.get(main_node.term, previous_rank)

    return clauses


def _best_modifiers(
    group: ScopeGroup,
    q_ranks: dict[str, int],
    main_term: str,
    hierarchy: LabelHierarchy,
) -> list[Statement]:
    by_qualifier: dict[str, Statement] = {}
    for statement in group.modifiers:
        qualifier = str(statement.evidence.get("qualifier", ""))
        current = by_qualifier.get(qualifier)
        if current is None or _agreement(statement, q_ranks, main_term, hierarchy) > _agreement(
            current, q_ranks, main_term, hierarchy
        ):
            by_qualifier[qualifier] = statement
    return list(by_qualifier.values())

def _agreement(
    statement: Statement,
    q_ranks: dict[str, int],
    main_term: str,
    hierarchy: LabelHierarchy,
) -> tuple[int, int, int, float]:
    term = str(statement.evidence.get("term", ""))
    return (
        int(term == main_term),
        int(hierarchy.same_family(term, main_term)),
        q_ranks.get(str(statement.evidence.get("quantifier", "")), 0),
        float(statement.truth),
    )

def aggregate_clauses(clauses: list[Clause]) -> list[Clause]:
    merged: list[Clause] = []
    for clause in clauses:
        previous = merged[-1] if merged else None
        if previous is not None and _mergeable(previous, clause):
            previous.scopes.extend(clause.scopes)
            continue
        merged.append(clause)
    return merged


def _mergeable(first: Clause, second: Clause) -> bool:
    if first.main is None or second.main is None:
        return False
    if first.events or second.events:
        return False
    if first.intensifiers or second.intensifiers:
        return False
    if first.others or second.others or first.modifiers or second.modifiers:
        return False
    return (
        first.main.evidence.get("term") == second.main.evidence.get("term")
        and first.main.evidence.get("quantifier") == second.main.evidence.get("quantifier")
    )

def realize_clauses(day: DaySynthesis, cfg: FuzzyDiaryConfig) -> list[RealizedClause]:
    clauses = aggregate_clauses(plan_document(day, cfg))
    signal = cfg.signal.signal_name or "the signal"
    hierarchy = build_label_hierarchy(cfg)

    out: list[RealizedClause] = []
    for index, clause in enumerate(clauses):
        text = _realize_one(clause, cfg, signal, first=index == 0, hierarchy=hierarchy)
        if text:
            out.append(RealizedClause(
                text=text,
                support=clause.support,
                events=clause.events,
                scopes=list(clause.scopes),
            ))
    return out


def realize_day(
    day: DaySynthesis,
    cfg: FuzzyDiaryConfig,
    include_date: bool = True,
) -> str:
    realized = realize_clauses(day, cfg)
    if not realized:
        return f"No describable activity on {day.day.date()}."

    sentences = [clause.text for clause in realized]
    if include_date:
        sentences[0] = f"On {day.day.date()}, {sentences[0][0].lower()}{sentences[0][1:]}"
    return " ".join(f"{s}." for s in sentences)


def narrate_day_paragraph(
    summary: DailySummary,
    cfg: FuzzyDiaryConfig,
    include_date: bool = True,
    primary_context: str | None = None,
) -> str:
    settings = getattr(cfg, "synthesis", None)
    if primary_context is None and settings is not None:
        primary_context = settings.primary_context
    threshold = settings.inclusion_threshold if settings is not None else 0.5

    synthesis = synthesize_day(
        summary, cfg,
        primary_context=primary_context,
        inclusion_threshold=threshold,
    )
    return realize_day(synthesis, cfg, include_date=include_date)

_PREPOSITIONS = ("in ", "at ", "on ", "during ", "through ", "towards ", "over ")


def _scope_phrase(labels: list[str], cfg: FuzzyDiaryConfig) -> str:
    phrases = []
    for label in labels:
        realized = lexicalize(label, "scope", cfg)
        if realized.lower().startswith(_PREPOSITIONS):
            phrases.append(realized)
        else:
            phrases.append(f"in the {realized}")
    return _join(phrases)


def _realize_one(
    clause: Clause,
    cfg: FuzzyDiaryConfig,
    signal: str,
    first: bool,
    hierarchy: LabelHierarchy,
) -> str:
    scope = _scope_phrase(clause.scopes, cfg)
    events = _events_to_report(clause, hierarchy)

    if clause.main is None:
        if not events:
            return ""
        return _capitalize(f"{scope}, {_realize_events(events, cfg)}")

    main_quantifier = str(clause.main.evidence.get("quantifier", ""))
    main_term = str(clause.main.evidence.get("term", ""))
    adverb = lexicalize(main_quantifier, "quantifier", cfg)
    coordinated = [main_term]
    remaining: list[Statement] = []
    for statement in clause.others:
        if str(statement.evidence.get("quantifier", "")) == main_quantifier:
            coordinated.append(str(statement.evidence.get("term", "")))
        else:
            remaining.append(statement)
    terms = _join([lexicalize(term, "term", cfg) for term in coordinated])

    if first:
        lead = f"{scope}, {signal} was {adverb} {terms}"
    elif clause.relation == "rise":
        lead = f"it rose {scope}, where it was {adverb} {terms}"
    elif clause.relation == "fall":
        lead = f"it fell {scope}, where it was {adverb} {terms}"
    else:
        lead = f"it remained {adverb} {terms} {scope}"

    parts = [lead]
    parts.extend(_realize_intensifiers(clause.intensifiers, cfg))
    parts.extend(_realize_modifiers(clause.modifiers, cfg, main_term, main_quantifier))
    parts.extend(_realize_others(remaining, cfg))

    sentence = ", ".join(parts)
    if events:
        sentence = f"{sentence}; {_realize_events(events, cfg)}"
    return _capitalize(sentence)


def _events_to_report(clause: Clause, hierarchy: LabelHierarchy) -> list[EventNode]:
    return [
        node for node in clause.events
        if hierarchy.transitions.get(node.kind) != clause.relation
    ]


def _realize_intensifiers(statements: list[Statement], cfg: FuzzyDiaryConfig) -> list[str]:
    if not statements:
        return []

    by_adverb: dict[str, list[str]] = {}
    for statement in statements:
        adverb = lexicalize(str(statement.evidence.get("quantifier", "")), "subordinate", cfg)
        term = lexicalize(str(statement.evidence.get("term", "")), "term", cfg)
        by_adverb.setdefault(adverb, []).append(term)

    return [f"{adverb} reaching {_join(terms)} values" for adverb, terms in by_adverb.items()]


def _realize_modifiers(
    statements: list[Statement],
    cfg: FuzzyDiaryConfig,
    main_term: str,
    main_quantifier: str,
) -> list[str]:

    if not statements:
        return []

    fragments = []
    for statement in statements:
        term = str(statement.evidence.get("term", ""))
        quantifier = str(statement.evidence.get("quantifier", ""))
        scope = lexicalize(str(statement.evidence.get("qualifier", "")), "scope", cfg)

        if term == main_term and quantifier == main_quantifier:
            fragments.append(f"also around {scope}")
        elif term == main_term:
            fragments.append(f"{lexicalize(quantifier, 'quantifier', cfg)} around {scope}")
        else:
            fragments.append(
                f"{lexicalize(quantifier, 'quantifier', cfg)} "
                f"{lexicalize(term, 'term', cfg)} around {scope}"
            )
    return [_join(fragments)]


def _realize_others(statements: list[Statement], cfg: FuzzyDiaryConfig) -> list[str]:
    out = []
    for statement in statements:
        adverb = lexicalize(str(statement.evidence.get("quantifier", "")), "quantifier", cfg)
        term = lexicalize(str(statement.evidence.get("term", "")), "term", cfg)
        out.append(f"and also {adverb} {term}")
    return out


def _realize_events(events: list[EventNode], cfg: FuzzyDiaryConfig) -> str:
    counts: dict[str, int] = {}
    for node in events:
        kind = lexicalize(node.kind, "event", cfg)
        counts[kind] = counts.get(kind, 0) + 1

    if len(counts) == 1:
        kind, count = next(iter(counts.items()))
        if count == 1:
            return f"a period of {kind} was detected"
        return f"{_NUMERALS.get(count, str(count))} periods of {kind} were detected"

    phrases = [
        kind if count == 1 else f"{_NUMERALS.get(count, str(count))} periods of {kind}"
        for kind, count in counts.items()
    ]
    return f"{_join(phrases)} were detected"


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text else text
