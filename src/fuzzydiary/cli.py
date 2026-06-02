from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fuzzydiary import (
    __version__,
    describe,
    detect_events,
    load_series,
    render_report,
    simplify,
)
from fuzzydiary.config import load_config


def _resolve_eps(value: str) -> float | str:
    return "auto" if value == "auto" else float(value)


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"fuzzydiary {__version__}")
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else None
    series = load_series(args.series, config=cfg)
    print(f"Loaded {len(series)} samples across {series.n_days} day(s).")
    print(f"Signal name : {series.signal.signal_name}")
    print(f"First sample: {series.data.index[0]}")
    print(f"Last sample : {series.data.index[-1]}")
    return 0


def _cmd_simplify(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else None
    series = load_series(args.series, config=cfg)
    simplified = simplify(series, epsilon=_resolve_eps(args.epsilon))
    print(f"Simplified {simplified.n_days} day(s).")
    for day, df in simplified.days.items():
        used = simplified.epsilon_used[day]
        original = len(series.days[day])
        print(f"  {day.date()}: {original} -> {len(df)} points (epsilon={used:.5f})")
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    if not args.config:
        print("[fuzzydiary] 'events' requires --config.", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    series = load_series(args.series, config=cfg)
    simplified = simplify(series, epsilon=_resolve_eps(args.epsilon))
    events = detect_events(simplified, config=cfg, series=series)

    print(f"Detected {events.n_events} event(s) across {events.n_days} day(s).")
    for day, evs in events.days.items():
        kinds: dict[str, int] = {}
        for e in evs:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "(none)"
        print(f"  {day.date()}: {len(evs)} event(s) -> {summary}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    if not args.config:
        print("[fuzzydiary] 'describe' requires --config.", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    series = load_series(args.series, config=cfg)
    simplified = simplify(series, epsilon=_resolve_eps(args.epsilon))
    events = detect_events(simplified, config=cfg, series=series)
    daily = describe(series, events, cfg, simplified=simplified)

    total_stmts = sum(len(d.all_statements) for d in daily.days.values())
    print(f"Described {daily.n_days} day(s), {total_stmts} instantiated statement(s).")
    for day, summary in daily.days.items():
        print(f"  {day.date()}: {summary.n_segments} segment(s), "
              f"{len(summary.all_statements)} statement(s).")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    series = load_series(args.series, config=cfg)
    simplified = simplify(series, epsilon=_resolve_eps(args.epsilon))
    events = detect_events(simplified, config=cfg, series=series)
    daily = describe(series, events, cfg, simplified=simplified)

    output = Path(args.output)
    if output.is_dir() or str(output).endswith("/"):
        output = output / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    path = render_report(series=series, events=events, daily=daily, output=output, cfg=cfg)
    print(f"Report written to: {path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fuzzydiary",
        description="Protoform-based linguistic summarization of univariate time series.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full pipeline and render an HTML report.")
    p_run.add_argument("--config", required=True, type=Path)
    p_run.add_argument("--series", required=True, type=Path)
    p_run.add_argument("--output", required=True, type=Path)
    p_run.add_argument("--epsilon", default="auto")
    p_run.set_defaults(func=_cmd_run)

    p_load = sub.add_parser("load", help="Ingest and inspect a series.")
    p_load.add_argument("--config", type=Path, default=None)
    p_load.add_argument("--series", required=True, type=Path)
    p_load.set_defaults(func=_cmd_load)

    p_simp = sub.add_parser("simplify", help="RDP geometric simplification.")
    p_simp.add_argument("--config", type=Path, default=None)
    p_simp.add_argument("--series", required=True, type=Path)
    p_simp.add_argument("--epsilon", default="auto")
    p_simp.set_defaults(func=_cmd_simplify)

    p_ev = sub.add_parser("events", help="Event detection.")
    p_ev.add_argument("--config", required=True, type=Path)
    p_ev.add_argument("--series", required=True, type=Path)
    p_ev.add_argument("--epsilon", default="auto")
    p_ev.set_defaults(func=_cmd_events)

    p_desc = sub.add_parser("describe", help="Per-day linguistic description.")
    p_desc.add_argument("--config", required=True, type=Path)
    p_desc.add_argument("--series", required=True, type=Path)
    p_desc.add_argument("--epsilon", default="auto")
    p_desc.set_defaults(func=_cmd_describe)

    p_ver = sub.add_parser("version", help="Print version.")
    p_ver.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
