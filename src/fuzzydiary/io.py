from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from fuzzydiary.config import FuzzyDiaryConfig, IOConfig, SignalConfig, load_config



@dataclass
class Series:
    """
    Canonical representation of a loaded univariate time series.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame indexed by datetime, with a single column 'value'.
    days : dict[pd.Timestamp, pd.DataFrame]
        Per-day segmentation; each key is a calendar day (midnight),
        each value is the slice of `data` falling on that day.
    signal : SignalConfig
        Signal metadata (name, unit, sampling period).
    """

    data: pd.DataFrame
    days: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    signal: SignalConfig = field(default_factory=SignalConfig)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_days(self) -> int:
        return len(self.days)



PathLike = Union[str, Path]


def load_series(
    path: PathLike | pd.DataFrame,
    config: FuzzyDiaryConfig | PathLike | None = None,
) -> Series:
    """
    Ingest a univariate time series and return the canonical representation.

    Parameters
    ----------
    path : str, Path, or pandas.DataFrame
        Source of the data. Supported formats:
          - CSV file with at least a timestamp column and a value column.
          - JSON file (records orientation) with the same columns.
          - In-memory pandas.DataFrame.
    config : FuzzyDiaryConfig, str, Path, or None
        Configuration object or path to a YAML configuration. If None,
        defaults are used (sampling period 5 min, 'timestamp' and 'value'
        columns expected, etc.).

    Returns
    -------
    Series
        Canonical representation with resampled data and per-day segmentation.
    """
    cfg = _resolve_config(config)

    raw = _read_raw(path)
    df = _normalise_columns(raw, cfg.io)
    df = _resample_and_impute(df, cfg.io, cfg.signal.sampling_period_minutes)
    days = _split_by_day(df)

    return Series(data=df, days=days, signal=cfg.signal)



def _resolve_config(
    config: FuzzyDiaryConfig | PathLike | None,
) -> FuzzyDiaryConfig:
    """Return a FuzzyDiaryConfig from any acceptable input."""
    if isinstance(config, FuzzyDiaryConfig):
        return config
    if config is not None:
        return load_config(config)

    from fuzzydiary.config import LinguisticVariable, Quantifier, Trapezoid

    return FuzzyDiaryConfig(
        linguistic_variable=LinguisticVariable(
            universe_min=0.0,
            universe_max=1.0,
            terms={"any": Trapezoid(a=0, b=0, c=1, d=1)},
        ),
        quantifiers=[
            Quantifier(name="some", membership=Trapezoid(a=0, b=0.2, c=0.8, d=1.0))
        ],
    )


def _read_raw(path: PathLike | pd.DataFrame) -> pd.DataFrame:
    """Read raw data from disk or accept an in-memory DataFrame."""
    if isinstance(path, pd.DataFrame):
        return path.copy()

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Data file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
        try:
            pd.to_datetime(df.columns[0])
            df = pd.read_csv(p, header=None, names=["timestamp", "value"])
        except (ValueError, TypeError):
            pass
        return df
    if suffix == ".json":
        return pd.read_json(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)

    raise ValueError(
        f"Unsupported file format: {suffix}. Use CSV, JSON, Parquet, or a pandas DataFrame."
    )


def _normalise_columns(df: pd.DataFrame, io_cfg: IOConfig) -> pd.DataFrame:
    """
    Identify the timestamp and value columns according to the IO config,
    parse the timestamp, drop duplicates, and return a DataFrame indexed
    by datetime with a single 'value' column.
    """
    if df.empty:
        raise ValueError("Input data is empty.")

    cols = list(df.columns)
    ts_col, val_col = io_cfg.timestamp_column, io_cfg.value_column

    if ts_col not in cols or val_col not in cols:
        if len(cols) < 2:
            raise ValueError(
                f"Cannot identify timestamp/value columns. Expected '{ts_col}' "
                f"and '{val_col}'; got {cols}."
            )
        ts_col, val_col = cols[0], cols[1]

    out = df[[ts_col, val_col]].copy()
    out.columns = ["timestamp", "value"]

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    if io_cfg.timezone:
        if out["timestamp"].dt.tz is None:
            out["timestamp"] = out["timestamp"].dt.tz_localize(io_cfg.timezone)
        else:
            out["timestamp"] = out["timestamp"].dt.tz_convert(io_cfg.timezone)

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
    out = out.sort_values("timestamp").set_index("timestamp")

    return out


def _resample_and_impute(
    df: pd.DataFrame,
    io_cfg: IOConfig,
    sampling_period_minutes: float,
) -> pd.DataFrame:
    """
    Resample to the target period, impute short gaps via centred rolling mean,
    and leave longer gaps as NaN (to be picked up by `_split_by_day`).
    """
    rule = f"{int(sampling_period_minutes)}min"
    resampled = df["value"].resample(rule).mean().to_frame("value")

    window = 7  # samples; mirrors the legacy preprocess.py heuristic.
    min_periods = max(1, int(io_cfg.empty_interval_threshold))
    rolled = (
        resampled["value"]
        .rolling(window, center=True, min_periods=min_periods)
        .mean()
    )

    filled = resampled["value"].copy()
    mask = filled.isna()
    filled.loc[mask] = rolled.loc[mask]

    if io_cfg.max_gap_minutes and io_cfg.max_gap_minutes > 0:
        max_gap_samples = int(np.ceil(io_cfg.max_gap_minutes / sampling_period_minutes))
        if max_gap_samples > 0:
            gap_id = (~mask).cumsum()  # contiguous NaN runs share an id
            sizes = mask.groupby(gap_id).transform("sum")
            wide = mask & (sizes > max_gap_samples)
            filled.loc[wide] = np.nan

    resampled["value"] = filled
    return resampled


def _split_by_day(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Partition the resampled series into a dict keyed by calendar day.

    Days are kept even if partially missing; the downstream modules decide
    what to do with a sparse day.
    """
    if df.empty:
        return {}

    days: dict[pd.Timestamp, pd.DataFrame] = {}
    for day, group in df.groupby(df.index.normalize()):
        days[pd.Timestamp(day)] = group
    return days
