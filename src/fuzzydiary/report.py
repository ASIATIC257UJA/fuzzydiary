from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template

from fuzzydiary.describe import DailySummaryCollection
from fuzzydiary.events import EventCollection
from fuzzydiary.io import Series
from fuzzydiary.narrate import narrate_daily


def render_report(
    series: Series,
    events: EventCollection | None = None,
    daily: DailySummaryCollection | None = None,
    output: str | Path = "./report/index.html",
    plotly_cdn: bool = True,
    cfg=None,
) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sections: list[dict] = []
    sections.append({
        "title": "Signal overview",
        "html": _signal_overview_figure(series, events, plotly_cdn, cfg),
    })

    if daily is not None and daily.n_days > 0:
        sections.append({"title": "Daily analysis", "html": _daily_section(daily, cfg)})

    html = _render_template(
        title=f"FuzzyDiary report — {series.signal.signal_name}",
        sections=sections,
        plotly_cdn=plotly_cdn,
    )
    output.write_text(html, encoding="utf-8")
    return output


_DEFAULT_PALETTE = [
    "#3D6FA8", "#2E7D54", "#C0392B", "#8C5320",
    "#6E3E70", "#A0522D", "#5C8A8A", "#B7950B",
]


def _signal_overview_figure(
    series: Series,
    events: EventCollection | None,
    use_cdn: bool,
    cfg=None,
) -> str:
    config_colors: dict[str, str] = {}
    if cfg is not None and hasattr(cfg, "events"):
        config_colors = cfg.events.event_colors or {}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.data.index,
            y=series.data["value"],
            mode="lines",
            name=series.signal.signal_name,
            line=dict(color="black", width=1.4),
        )
    )

    if events is not None:
        kind_colors: dict[str, str] = {}
        _palette_idx = 0

        def _color_for(kind: str) -> str:
            nonlocal _palette_idx
            if kind not in kind_colors:
                if kind in config_colors:
                    kind_colors[kind] = config_colors[kind]
                else:
                    kind_colors[kind] = _DEFAULT_PALETTE[_palette_idx % len(_DEFAULT_PALETTE)]
                    _palette_idx += 1
            return kind_colors[kind]

        for day_events in events.days.values():
            for ev in day_events:
                if not ev.composite:
                    _color_for(ev.kind)

        for kind, color in kind_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=kind,
                showlegend=True,
            ))

        for day_events in events.days.values():
            for ev in day_events:
                if ev.composite:
                    continue
                color = _color_for(ev.kind)
                fig.add_vrect(
                    x0=ev.start, x1=ev.end,
                    fillcolor=color, opacity=0.18, line_width=0,
                )

    unit = series.signal.unit or "value"
    fig.update_layout(
        template='simple_white',
        margin=dict(l=40, r=20, t=30, b=80),
        height=400,
        xaxis_title="Time",
        yaxis_title=unit,
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if use_cdn else True)


def _daily_section(daily: DailySummaryCollection, cfg=None) -> str:
    context_order: list[str] = []
    if cfg is not None and cfg.contexts:
        context_order = list(cfg.contexts[0].terms.keys())

    parts: list[str] = []
    for day in sorted(daily.days):
        summary = daily.days[day]

        ctx_dims = list(summary.events[0].contexts.keys()) if summary.events else []

        header_cols = "".join(f"<th>{_escape(dim)}</th>" for dim in ctx_dims)
        ev_rows: list[str] = []
        for ed in summary.events:
            ctx_cells = "".join(
                f"<td>{_escape(ed.contexts[dim][0]) if ed.contexts.get(dim, ('', 0))[1] > 0.1 else '—'}</td>"
                for dim in ctx_dims
            )
            ev_rows.append(f"<tr><td>{_escape(ed.event.kind)}</td>{ctx_cells}</tr>")

        ev_table = (
            "<table class='fd-table'>"
            f"<thead><tr><th>event</th>{header_cols}</tr></thead>"
            "<tbody>" + "".join(ev_rows) + "</tbody></table>"
        ) if ev_rows else "<p><em>No qualified events detected.</em></p>"

        def _ctx_rank(st):
            text_lower = st.text.lower()
            for i, term in enumerate(context_order):
                if term.lower() in text_lower:
                    return i
            return len(context_order)

        all_stmts = [s for s in summary.all_statements if s.template != "<event-characteristic>"]
        proto_sorted = (
            sorted(all_stmts, key=_ctx_rank)
            if context_order
            else sorted(all_stmts, key=lambda s: s.truth, reverse=True)
        )

        proto_rows: list[str] = []
        for st in proto_sorted:
            proto_rows.append(
                f"<tr><td>{_escape(st.text)}</td>"
                f"<td>{st.truth:.2f}</td></tr>"
            )
        proto_table = (
            "<table class='fd-table'>"
            "<thead><tr><th>linguistic summary</th><th>DoT</th></tr></thead>"
            "<tbody>" + "".join(proto_rows) + "</tbody></table>"
        ) if proto_rows else "<p><em>No protoforms above threshold.</em></p>"

        parts.append(
            f"<details open><summary><b>{day.date()}</b></summary>"
            f"<p><b>Detected events</b></p>{ev_table}"
            f"<p><b>Daily linguistic summary</b></p>{proto_table}"
            f"</details>"
        )
    return "\n".join(parts)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
           max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }
    h1 { font-size: 1.6em; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
    h2 { font-size: 1.2em; margin-top: 32px; color: #345; }
    details { margin: 12px 0; padding: 10px 14px; border: 1px solid #e3e6ea;
              border-radius: 8px; background: #fafbfd; }
    summary { cursor: pointer; padding: 4px 0; font-size: 1.02em; }
    .fd-narrative { line-height: 1.5; white-space: pre-wrap; }
    .fd-table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.92em; }
    .fd-table th, .fd-table td { border: 1px solid #e3e6ea; padding: 6px 8px; text-align: left; }
    .fd-table th { background: #f0f3f7; }
    footer { margin-top: 40px; padding-top: 8px; border-top: 1px solid #ddd;
             color: #777; font-size: 0.85em; text-align: right; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  {% for section in sections %}
    <h2>{{ section.title }}</h2>
    {{ section.html | safe }}
  {% endfor %}
  <footer>Generated by FuzzyDiary.</footer>
</body>
</html>
"""


def _render_template(title: str, sections: list[dict], plotly_cdn: bool) -> str:
    return Template(_HTML_TEMPLATE).render(title=title, sections=sections)


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
