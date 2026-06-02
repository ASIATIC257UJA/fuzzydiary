from __future__ import annotations

from dataclasses import dataclass

from fuzzydiary.config import LinguisticVariable, Quantifier
from fuzzydiary.fuzzy import membership_degree


@dataclass
class AggregationResult:
    proportions: dict[str, float]
    quantifier: str | None
    quantifier_md: float
    dominant_term: str


def aggregate_labels(
    sample_memberships: list[dict[str, float]],
    variable: LinguisticVariable,
    quantifiers: list[Quantifier],
    acceptance_threshold: float = 0.5,
) -> AggregationResult:
    proportions = _compute_proportions(sample_memberships, variable)
    dominant_term = max(proportions.items(), key=lambda kv: kv[1])[0]
    selected_q, selected_md = _select_rightmost_passing(
        proportions[dominant_term], quantifiers, acceptance_threshold
    )
    return AggregationResult(
        proportions=proportions,
        quantifier=selected_q,
        quantifier_md=selected_md,
        dominant_term=dominant_term,
    )


def _compute_proportions(
    sample_memberships: list[dict[str, float]],
    variable: LinguisticVariable,
) -> dict[str, float]:
    n = len(sample_memberships)
    proportions: dict[str, float] = {term: 0.0 for term in variable.terms}
    if n == 0:
        return proportions
    for memb in sample_memberships:
        for term in proportions:
            proportions[term] += float(memb.get(term, 0.0))
    return {term: v / n for term, v in proportions.items()}


def _select_rightmost_passing(
    proportion: float,
    quantifiers: list[Quantifier],
    threshold: float,
) -> tuple[str | None, float]:
    candidates: list[tuple[Quantifier, float, float]] = []
    for q in quantifiers:
        md = membership_degree(proportion, q.membership_as_dict())
        if md >= threshold:
            centre = _centre_of_mass(q)
            candidates.append((q, md, centre))
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda t: (t[2], t[1]), reverse=True)
    chosen = candidates[0]
    return chosen[0].name, chosen[1]


def _centre_of_mass(q: Quantifier) -> float:
    mf = q.membership
    mf_type = getattr(mf, "type", "trapezoid")
    if mf_type == "trapezoid":
        return (mf.a + mf.b + mf.c + mf.d) / 4.0
    if mf_type == "triangle":
        return (mf.a + mf.b + mf.c) / 3.0
    if mf_type == "gaussian":
        return mf.mean
    if mf_type == "singleton":
        return mf.value
    return 0.5
