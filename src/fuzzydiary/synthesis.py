from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fuzzydiary.config import (
    ContextDimension,
    FuzzyDiaryConfig,
    LabelCondition,
    LevelCondition,
    LinguisticVariable,
)
from fuzzydiary.describe import DailySummary, EventDescription, Statement
from fuzzydiary.fuzzy import membership_degree

_GRID = 512

@dataclass
class SynthesisNode:
    statement: Statement
    children: list["SynthesisNode"] = field(default_factory=list)

    @property
    def term(self) -> str:
        return str(self.statement.evidence.get("term", ""))

    @property
    def quantifier(self) -> str:
        return str(self.statement.evidence.get("quantifier", ""))

    @property
    def truth(self) -> float:
        return float(self.statement.truth)

    def flatten(self) -> list[Statement]:
        """This node's statement plus every statement below it."""
        out = [self.statement]
        for child in self.children:
            out.extend(child.flatten())
        return out

@dataclass
class EventNode:
    description: EventDescription
    children: list["EventNode"] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.description.event.kind

@dataclass
class ScopeGroup:
    dimension: str
    qualifier: str
    order: int
    roots: list[SynthesisNode] = field(default_factory=list)
    events: list[EventNode] = field(default_factory=list)
    modifiers: list[Statement] = field(default_factory=list)

    @property
    def statements(self) -> list[Statement]:
        return [s for root in self.roots for s in root.flatten()]

@dataclass
class DaySynthesis:
    day: pd.Timestamp
    groups: list[ScopeGroup] = field(default_factory=list)
    unscoped: list[Statement] = field(default_factory=list)
    primary_dimension: str = ""

    @property
    def n_subordinated(self) -> int:
        return sum(
            len(root.flatten()) - 1
            for group in self.groups
            for root in group.roots
        )


@dataclass(frozen=True)
class LabelHierarchy:
    families: dict[str, frozenset[str]]
    term_family: dict[str, str]
    specializes: dict[str, frozenset[str]]
    transitions: dict[str, str] = field(default_factory=dict)

    def same_family(self, term_a: str, term_b: str) -> bool:
        fam_a = self.term_family.get(term_a)
        fam_b = self.term_family.get(term_b)
        return fam_a is not None and fam_a == fam_b

def _grid(lo: float, hi: float, n: int = _GRID) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]

def term_inclusion(
    variable: LinguisticVariable,
    term_a: str,
    term_b: str,
    n: int = _GRID,
) -> float:
    mf_a = variable.mf_as_dict(term_a)
    mf_b = variable.mf_as_dict(term_b)
    num = 0.0
    den = 0.0
    for x in _grid(variable.universe_min, variable.universe_max, n):
        mu_a = membership_degree(x, mf_a)
        if mu_a <= 0.0:
            continue
        den += mu_a
        num += min(mu_a, membership_degree(x, mf_b))
    return num / den if den > 0 else 0.0

def _centroid(mf: dict, lo: float, hi: float, n: int = _GRID) -> float:
    num = 0.0
    den = 0.0
    for x in _grid(lo, hi, n):
        mu = membership_degree(x, mf)
        num += x * mu
        den += mu
    return num / den if den > 0 else lo

def term_order(variable: LinguisticVariable) -> dict[str, int]:
    centroids = {
        term: _centroid(variable.mf_as_dict(term), variable.universe_min, variable.universe_max)
        for term in variable.terms
    }
    ordered = sorted(centroids, key=lambda t: centroids[t])
    return {term: rank for rank, term in enumerate(ordered)}

def quantifier_order(cfg: FuzzyDiaryConfig) -> dict[str, int]:
    centroids = {
        q.name: _centroid(q.membership_as_dict(), 0.0, 1.0) for q in cfg.quantifiers
    }
    ordered = sorted(centroids, key=lambda q: centroids[q])
    return {name: rank for rank, name in enumerate(ordered)}

def _extremity(term: str, order: dict[str, int]) -> float:
    if term not in order:
        return 0.0
    center = (len(order) - 1) / 2.0
    return abs(order[term] - center)

def _normalize(label: str) -> str:
    return str(label).strip().lower().replace("_", " ")


def build_label_hierarchy(cfg: FuzzyDiaryConfig) -> LabelHierarchy:
    known = {_normalize(t): t for t in cfg.linguistic_variable.terms}
    families: dict[str, frozenset[str]] = {}
    durations: dict[str, float] = {}
    for event in cfg.events.catalogue:
        if not event.enabled:
            continue
        terms: set[str] = set()
        for step in event.pattern:
            if isinstance(step, LevelCondition):
                for raw in step.level:
                    resolved = known.get(_normalize(raw))
                    if resolved is not None:
                        terms.add(resolved)
        if terms:
            families[event.label] = frozenset(terms)
            durations[event.label] = float(event.min_duration_minutes)

    declared = list(families)
    term_family: dict[str, str] = {}
    for term in cfg.linguistic_variable.terms:
        candidates = [label for label in declared if term in families[label]]
        if not candidates:
            continue
        candidates.sort(key=lambda label: (-len(families[label]), declared.index(label)))
        term_family[term] = candidates[0]

    specializes: dict[str, frozenset[str]] = {}
    for label, terms in families.items():
        parents = {
            other
            for other, other_terms in families.items()
            if other != label
            and terms <= other_terms
            and durations[label] > durations[other]
        }
        if parents:
            specializes[label] = frozenset(parents)

    return LabelHierarchy(
        families=families,
        term_family=term_family,
        specializes=specializes,
        transitions=_build_transitions(cfg, families),
    )


def _build_transitions(
    cfg: FuzzyDiaryConfig,
    families: dict[str, frozenset[str]],
) -> dict[str, str]:
    ranks = term_order(cfg.linguistic_variable)

    def mean_rank(terms: frozenset[str]) -> float | None:
        known = [ranks[t] for t in terms if t in ranks]
        return sum(known) / len(known) if known else None

    out: dict[str, str] = {}
    for event in cfg.events.catalogue:
        if not event.enabled or event.label not in families:
            continue
        anchors = list(event.anchor_labels)
        anchors += [s.label for s in event.pattern if isinstance(s, LabelCondition)]
        anchor_terms = frozenset().union(
            *(families.get(label, frozenset()) for label in anchors)
        ) if anchors else frozenset()
        if not anchor_terms:
            continue

        target = mean_rank(families[event.label])
        source = mean_rank(anchor_terms)
        if target is None or source is None or target == source:
            continue
        out[event.label] = "rise" if target > source else "fall"
    return out

@dataclass(frozen=True)
class Orderings:
    hierarchy: LabelHierarchy
    terms: dict[str, int]
    quantifiers: dict[str, int]
    variable: LinguisticVariable
    inclusion_threshold: float


def build_orderings(
    cfg: FuzzyDiaryConfig,
    inclusion_threshold: float = 0.5,
) -> Orderings:
    return Orderings(
        hierarchy=build_label_hierarchy(cfg),
        terms=term_order(cfg.linguistic_variable),
        quantifiers=quantifier_order(cfg),
        variable=cfg.linguistic_variable,
        inclusion_threshold=inclusion_threshold,
    )


def subsumes(
    specific: Statement,
    general: Statement,
    orderings: Orderings,
) -> bool:
    ev_s, ev_g = specific.evidence, general.evidence
    if ev_s.get("scope") != "day" or ev_g.get("scope") != "day":
        return False
    if ev_s.get("qualifier") != ev_g.get("qualifier"):
        return False

    term_s = str(ev_s.get("term", ""))
    term_g = str(ev_g.get("term", ""))
    if not term_s or not term_g or term_s == term_g:
        return False

    related = orderings.hierarchy.same_family(term_s, term_g)
    if not related:
        related = (
            term_inclusion(orderings.variable, term_s, term_g)
            >= orderings.inclusion_threshold
        )
    if not related:
        return False

    if _extremity(term_s, orderings.terms) <= _extremity(term_g, orderings.terms):
        return False

    rank_s = orderings.quantifiers.get(str(ev_s.get("quantifier", "")), 0)
    rank_g = orderings.quantifiers.get(str(ev_g.get("quantifier", "")), 0)
    return rank_g >= rank_s

def synthesize_day(
    summary: DailySummary,
    cfg: FuzzyDiaryConfig,
    primary_context: str | None = None,
    inclusion_threshold: float = 0.5,
) -> DaySynthesis:
    if not cfg.contexts:
        return DaySynthesis(day=summary.day, unscoped=list(summary.day_protoforms))

    orderings = build_orderings(cfg, inclusion_threshold)

    primary = _primary_dimension(cfg, primary_context)
    others = [dim for dim in cfg.contexts if dim.name != primary.name]

    groups = [
        ScopeGroup(dimension=primary.name, qualifier=term, order=order)
        for order, term in enumerate(primary.terms)
    ]
    by_qualifier = {group.qualifier: group for group in groups}

    unscoped: list[Statement] = []
    for statement in summary.day_protoforms:
        qualifier = str(statement.evidence.get("qualifier", ""))
        group = by_qualifier.get(qualifier)
        if group is not None:
            group.roots.append(SynthesisNode(statement=statement))
            continue
        host = _host_group(qualifier, others, primary, by_qualifier)
        if host is not None:
            host.modifiers.append(statement)
        else:
            unscoped.append(statement)

    for group in groups:
        group.roots = _build_forest(group.roots, orderings)
        group.events = _build_event_forest(
            [
                ed for ed in summary.events
                if ed.contexts.get(primary.name, ("", 0.0))[0] == group.qualifier
            ],
            orderings.hierarchy,
        )

    populated = [g for g in groups if g.roots or g.events or g.modifiers]
    return DaySynthesis(
        day=summary.day,
        groups=populated,
        unscoped=unscoped,
        primary_dimension=primary.name,
    )

def synthesize(
    daily,
    cfg: FuzzyDiaryConfig,
    primary_context: str | None = None,
    inclusion_threshold: float = 0.5,
) -> dict[pd.Timestamp, DaySynthesis]:
    return {
        day: synthesize_day(summary, cfg, primary_context, inclusion_threshold)
        for day, summary in daily.days.items()
    }

def _primary_dimension(
    cfg: FuzzyDiaryConfig,
    requested: str | None,
) -> ContextDimension:
    if requested:
        for dim in cfg.contexts:
            if dim.name == requested:
                return dim
    return cfg.contexts[0]


def _support_midpoint(dim: ContextDimension, term: str) -> float:
    mf = dim.mf_as_dict(term)
    best_x, best_mu = dim.universe_min, -1.0
    total_x, total_w = 0.0, 0.0
    for x in _grid(dim.universe_min, dim.universe_max):
        mu = membership_degree(x, mf)
        if mu > best_mu:
            best_x, best_mu = x, mu
        total_x += x * mu
        total_w += mu
    return total_x / total_w if total_w > 0 else best_x


def _host_group(
    qualifier: str,
    others: list[ContextDimension],
    primary: ContextDimension,
    by_qualifier: dict[str, ScopeGroup],
) -> ScopeGroup | None:
    for dim in others:
        if qualifier not in dim.terms:
            continue
        midpoint = _support_midpoint(dim, qualifier)
        best_term, best_mu = None, 0.0
        for term in primary.terms:
            mu = membership_degree(midpoint, primary.mf_as_dict(term))
            if mu > best_mu:
                best_term, best_mu = term, mu
        if best_term is not None:
            return by_qualifier.get(best_term)
    return None


def _build_forest(
    nodes: list[SynthesisNode],
    orderings: Orderings,
) -> list[SynthesisNode]:
    ranked = sorted(
        nodes,
        key=lambda node: (
            _extremity(node.term, orderings.terms),
            -orderings.quantifiers.get(node.quantifier, 0),
            -node.truth,
        ),
    )

    roots: list[SynthesisNode] = []
    for node in ranked:
        parent = None
        for candidate in roots:
            if subsumes(node.statement, candidate.statement, orderings):
                parent = candidate
                break
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)

    roots.sort(key=lambda node: (-orderings.quantifiers.get(node.quantifier, 0), -node.truth))
    for root in roots:
        root.children.sort(key=lambda node: -_extremity(node.term, orderings.terms))
    return roots


def _build_event_forest(
    descriptions: list[EventDescription],
    hierarchy: LabelHierarchy,
) -> list[EventNode]:
    ordered = sorted(
        descriptions,
        key=lambda ed: (ed.event.start, -(ed.event.end - ed.event.start).total_seconds()),
    )
    nodes = [EventNode(description=ed) for ed in ordered]

    roots: list[EventNode] = []
    for node in nodes:
        parent = None
        for candidate in roots:
            if _specializes(candidate, node, hierarchy) and _contains(candidate, node):
                parent = candidate
                break
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
    return roots


def _specializes(specific: EventNode, general: EventNode, hierarchy: LabelHierarchy) -> bool:
    return general.kind in hierarchy.specializes.get(specific.kind, frozenset())


def _contains(outer: EventNode, inner: EventNode) -> bool:
    return (
        outer.description.event.start <= inner.description.event.start
        and outer.description.event.end >= inner.description.event.end
    )
