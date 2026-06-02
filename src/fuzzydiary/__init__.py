from fuzzydiary.io import load_series
from fuzzydiary.config import load_config, FuzzyDiaryConfig

from fuzzydiary.simplify import simplify
from fuzzydiary.events import detect_events
from fuzzydiary.describe import describe
from fuzzydiary.report import render_report

__version__ = "0.1.0"

__all__ = [
    "load_series",
    "simplify",
    "detect_events",
    "describe",
    "render_report",
    "load_config",
    "FuzzyDiaryConfig",
    "__version__",
]
