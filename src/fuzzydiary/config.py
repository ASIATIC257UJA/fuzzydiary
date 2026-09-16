from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, model_validator


class Trapezoid(BaseModel):
    type: Literal["trapezoid"] = "trapezoid"
    a: float
    b: float
    c: float
    d: float

    @model_validator(mode="after")
    def _check_order(self) -> "Trapezoid":
        if not (self.a <= self.b <= self.c <= self.d):
            raise ValueError(f"Trapezoid breakpoints must satisfy a<=b<=c<=d; got {self}")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.a, self.b, self.c, self.d)

    def as_dict(self) -> dict:
        return {"type": "trapezoid", "a": self.a, "b": self.b, "c": self.c, "d": self.d}


class Triangle(BaseModel):
    type: Literal["triangle"] = "triangle"
    a: float
    b: float
    c: float

    @model_validator(mode="after")
    def _check_order(self) -> "Triangle":
        if not (self.a <= self.b <= self.c):
            raise ValueError(f"Triangle breakpoints must satisfy a<=b<=c; got {self}")
        return self

    def as_dict(self) -> dict:
        return {"type": "triangle", "a": self.a, "b": self.b, "c": self.c}


class Gaussian(BaseModel):
    type: Literal["gaussian"] = "gaussian"
    mean: float
    sigma: float

    @model_validator(mode="after")
    def _check_sigma(self) -> "Gaussian":
        if self.sigma <= 0:
            raise ValueError(f"Gaussian sigma must be > 0; got {self.sigma}")
        return self

    def as_dict(self) -> dict:
        return {"type": "gaussian", "mean": self.mean, "sigma": self.sigma}


class Singleton(BaseModel):
    type: Literal["singleton"] = "singleton"
    value: float
    tolerance: float = 0.0

    def as_dict(self) -> dict:
        return {"type": "singleton", "value": self.value, "tolerance": self.tolerance}


MembershipFunction = Annotated[
    Union[Trapezoid, Triangle, Gaussian, Singleton],
    Field(discriminator="type"),
]


def _mf_from_raw(raw: Any) -> MembershipFunction:
    if isinstance(raw, dict):
        kind = raw.get("type", "trapezoid")
        if kind == "trapezoid":
            return Trapezoid.model_validate(raw)
        if kind == "triangle":
            return Triangle.model_validate(raw)
        if kind == "gaussian":
            return Gaussian.model_validate(raw)
        if kind == "singleton":
            return Singleton.model_validate(raw)
        raise ValueError(f"Unknown membership function type: {kind!r}")
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return Trapezoid(a=raw[0], b=raw[1], c=raw[2], d=raw[3])
    raise ValueError(f"Cannot parse membership function from {raw!r}")


class LinguisticVariable(BaseModel):
    universe_min: float
    universe_max: float
    terms: dict[str, MembershipFunction] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce_terms(cls, data: Any) -> Any:
        if isinstance(data, dict) and "terms" in data:
            raw_terms = data["terms"]
            if isinstance(raw_terms, dict):
                coerced: dict[str, Any] = {}
                for name, raw_mf in raw_terms.items():
                    if isinstance(raw_mf, dict) and "type" not in raw_mf:
                        keys = set(raw_mf.keys())
                        if keys == {"a", "b", "c", "d"}:
                            raw_mf = {**raw_mf, "type": "trapezoid"}
                        elif keys == {"a", "b", "c"}:
                            raw_mf = {**raw_mf, "type": "triangle"}
                        elif keys >= {"mean", "sigma"}:
                            raw_mf = {**raw_mf, "type": "gaussian"}
                        elif "value" in keys:
                            raw_mf = {**raw_mf, "type": "singleton"}
                    coerced[name] = raw_mf
                data = {**data, "terms": coerced}
        return data

    def mf_as_dict(self, term: str) -> dict:
        return self.terms[term].as_dict()


class Quantifier(BaseModel):
    name: str
    membership: MembershipFunction

    @model_validator(mode="before")
    @classmethod
    def _coerce_membership(cls, data: Any) -> Any:
        if isinstance(data, dict) and "membership" in data:
            raw = data["membership"]
            if isinstance(raw, dict) and "type" not in raw:
                keys = set(raw.keys())
                if keys == {"a", "b", "c", "d"}:
                    raw = {**raw, "type": "trapezoid"}
                elif keys == {"a", "b", "c"}:
                    raw = {**raw, "type": "triangle"}
                elif keys >= {"mean", "sigma"}:
                    raw = {**raw, "type": "gaussian"}
                elif "value" in keys:
                    raw = {**raw, "type": "singleton"}
                data = {**data, "membership": raw}
        return data

    def membership_as_dict(self) -> dict:
        return self.membership.as_dict()


class LevelCondition(BaseModel):
    level: list[str] = Field(..., min_length=1)
    min_duration_minutes: float = 0.0


class LabelCondition(BaseModel):
    label: str


PatternStep = LevelCondition | LabelCondition


class EventDef(BaseModel):
    label: str
    pattern: list[PatternStep] = Field(..., min_length=1)
    anchor_labels: list[str] = Field(default_factory=list)
    min_duration_minutes: float = 0.0
    max_duration_minutes: float = 0.0
    merge_gap_minutes: float = 0.0
    max_gap_minutes: float = 60.0
    enabled: bool = True

    @property
    def is_composite(self) -> bool:
        return not self.anchor_labels and all(
            isinstance(s, LabelCondition) for s in self.pattern
        )

    @property
    def is_anchored(self) -> bool:
        if self.anchor_labels:
            return all(isinstance(s, LevelCondition) for s in self.pattern)
        label_steps = [isinstance(s, LabelCondition) for s in self.pattern]
        level_steps = [isinstance(s, LevelCondition) for s in self.pattern]
        if not any(label_steps) or not any(level_steps):
            return False
        first_level = next(
            i for i, s in enumerate(self.pattern) if isinstance(s, LevelCondition)
        )
        return all(isinstance(s, LabelCondition) for s in self.pattern[:first_level])


class EventsConfig(BaseModel):
    catalogue: list[EventDef] = Field(default_factory=list)
    describe_lines: bool = False
    event_colors: dict[str, str] = Field(default_factory=dict)


class ContextDimension(BaseModel):
    name: str
    universe_min: float = 0.0
    universe_max: float = 24.0
    terms: dict[str, MembershipFunction] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce_terms(cls, data: Any) -> Any:
        if isinstance(data, dict) and "terms" in data:
            raw_terms = data["terms"]
            if isinstance(raw_terms, dict):
                coerced: dict[str, Any] = {}
                for name, raw_mf in raw_terms.items():
                    if isinstance(raw_mf, dict) and "type" not in raw_mf:
                        keys = set(raw_mf.keys())
                        if keys == {"a", "b", "c", "d"}:
                            raw_mf = {**raw_mf, "type": "trapezoid"}
                        elif keys == {"a", "b", "c"}:
                            raw_mf = {**raw_mf, "type": "triangle"}
                        elif keys >= {"mean", "sigma"}:
                            raw_mf = {**raw_mf, "type": "gaussian"}
                        elif "value" in keys:
                            raw_mf = {**raw_mf, "type": "singleton"}
                    coerced[name] = raw_mf
                data = {**data, "terms": coerced}
        return data

    def mf_as_dict(self, term: str) -> dict:
        return self.terms[term].as_dict()


class ProtoformTemplate(BaseModel):
    template: str
    description: str | None = None


class LexiconConfig(BaseModel):
    terms: dict[str, str] = Field(default_factory=dict)
    scopes: dict[str, str] = Field(default_factory=dict)
    quantifiers: dict[str, str] = Field(default_factory=dict)
    subordinates: dict[str, str] = Field(default_factory=dict)
    events: dict[str, str] = Field(default_factory=dict)


class SynthesisConfig(BaseModel):
    enabled: bool = True
    primary_context: str | None = None
    inclusion_threshold: float = 0.5


class SignalConfig(BaseModel):
    signal_name: str = "signal"
    sampling_period_minutes: float = 5.0
    unit: str | None = None


class RDPConfig(BaseModel):
    epsilon: float | Literal["auto"] = "auto"
    auto_std_fraction: float = 0.2


class IOConfig(BaseModel):
    timestamp_column: str = "timestamp"
    value_column: str = "value"
    timezone: str | None = None
    empty_interval_threshold: int = 3
    max_gap_minutes: float = 30.0


class FuzzyDiaryConfig(BaseModel):
    signal: SignalConfig = Field(default_factory=SignalConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    rdp: RDPConfig = Field(default_factory=RDPConfig)
    linguistic_variable: LinguisticVariable
    quantifiers: list[Quantifier] = Field(..., min_length=1)
    trend: LinguisticVariable = Field(
        default_factory=lambda: LinguisticVariable(
            universe_min=-1.0,
            universe_max=1.0,
            terms={
                "sharply_decreasing": Trapezoid(a=-1.0, b=-1.0, c=-0.75, d=-0.5),
                "decreasing":         Trapezoid(a=-0.75, b=-0.5, c=-0.25, d=-0.10),
                "steady":             Trapezoid(a=-0.25, b=-0.10, c=0.10, d=0.25),
                "increasing":         Trapezoid(a=0.10, b=0.25, c=0.5, d=0.75),
                "sharply_increasing": Trapezoid(a=0.5, b=0.75, c=1.0, d=1.0),
            },
        )
    )
    contexts: list[ContextDimension] = Field(default_factory=list)
    events: EventsConfig = Field(default_factory=EventsConfig)
    protoforms: list[ProtoformTemplate] = Field(default_factory=list)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    lexicon: LexiconConfig = Field(default_factory=LexiconConfig)


def load_config(path: str | Path) -> FuzzyDiaryConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return FuzzyDiaryConfig.model_validate(raw)
