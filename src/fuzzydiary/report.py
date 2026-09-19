from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

from fuzzydiary.config import FuzzyDiaryConfig, load_config
from fuzzydiary.describe import DailySummaryCollection
from fuzzydiary.events import EventCollection
from fuzzydiary.fuzzy import membership_degree
from fuzzydiary.io import Series
from fuzzydiary.nlg import realize_clauses
from fuzzydiary.synthesis import synthesize_day
_SIGNAL_DIV_ID = "fd-signal"


def render_report(
    series: Series,
    events: EventCollection | None = None,
    daily: DailySummaryCollection | None = None,
    output: str | Path = "./report/index.html",
    plotly_cdn: bool = False,
    config: FuzzyDiaryConfig | str | Path | None = None,
) -> Path:
    cfg = config if config is None or isinstance(config, FuzzyDiaryConfig) else load_config(config)
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
        height=250,
        xaxis_title="Time",
        yaxis_title=unit,
        legend=dict(orientation="h", yanchor="top", y=-0.5, xanchor="center", x=0.5),
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if use_cdn else True,
        div_id=_SIGNAL_DIV_ID,
    )


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
            "<thead><tr><th>Protoform</th><th>DoT</th></tr></thead>"
            "<tbody>" + "".join(proto_rows) + "</tbody></table>"
        ) if proto_rows else "<p><em>No protoforms above threshold.</em></p>"

        narrative = _daily_narrative_block(summary, cfg)

        parts.append(
            f"<details open><summary><b>{day.date()}</b></summary>"
            f"{narrative}"
            f"<p><b>Detected events</b></p>{ev_table}"
            f"<p><b>Activated protoforms across data</b></p>{proto_table}"
            f"</details>"
        )
    return "\n".join(parts)


def _daily_narrative_block(summary, cfg=None) -> str:
    if cfg is None or not getattr(getattr(cfg, "synthesis", None), "enabled", False):
        return ""

    synthesis = synthesize_day(
        summary, cfg,
        primary_context=cfg.synthesis.primary_context,
        inclusion_threshold=cfg.synthesis.inclusion_threshold,
    )
    clauses = realize_clauses(synthesis, cfg)
    if not clauses:
        return ""

    spans = []
    for clause in clauses:
        evidence = [f"{st.text} (DoT={st.truth:.2f})" for st in clause.support]
        if clause.events:
            kinds = ", ".join(node.kind for node in clause.events)
            evidence.append(f"events: {kinds}")
        tooltip = _escape_attr(" | ".join(evidence))
        spans_attr = _spans_attr(_clause_spans(clause, summary, cfg))
        spans.append(
            f'<span class="fd-clause fd-linked" tabindex="0" '
            f'title="{tooltip}" data-fd-spans="{spans_attr}">'
            f"{_escape(clause.text)}.</span>"
        )

    return (
        "<p><b>Narrative summary</b></p>"
        f"<p class='fd-narrative'>{' '.join(spans)}</p>"
    )


def _spans_attr(spans: list[tuple]) -> str:
    payload = [
        [pd.Timestamp(start).isoformat(), pd.Timestamp(end).isoformat()]
        for start, end in _merge_spans(spans)
    ]
    return _escape_attr(json.dumps(payload, separators=(",", ":")))


def _merge_spans(spans: list[tuple]) -> list[tuple]:
    clean = [(s, e) for s, e in spans if s is not None and e is not None]
    if not clean:
        return []
    merged: list[list] = []
    for start, end in sorted(clean):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _clause_spans(clause, summary, cfg=None) -> list[tuple]:
    spans: list[tuple] = []
    for node in clause.events:
        spans.extend(_event_node_spans(node))
    for statement in clause.support:
        start = statement.evidence.get("start")
        end = statement.evidence.get("end")
        if start is not None and end is not None:
            spans.append((start, end))
    if not spans:
        spans.extend(_scope_spans(clause.scopes, summary.day, cfg))
    return _merge_spans(spans)


def _event_node_spans(node) -> list[tuple]:
    spans = [(node.description.event.start, node.description.event.end)]
    for child in node.children:
        spans.extend(_event_node_spans(child))
    return spans


def _scope_spans(scopes: list[str], day, cfg=None, alpha: float = 0.5) -> list[tuple]:
    if not scopes or cfg is None or not cfg.contexts:
        return []

    requested = getattr(getattr(cfg, "synthesis", None), "primary_context", None)
    dimension = None
    for dim in cfg.contexts:
        if requested and dim.name == requested:
            dimension = dim
            break
    if dimension is None:
        dimension = cfg.contexts[0]

    day_start = pd.Timestamp(day).normalize()
    spans: list[tuple] = []
    for scope in scopes:
        if scope not in dimension.terms:
            continue
        cut = _alpha_cut(dimension, scope, alpha)
        if cut is None:
            continue
        low, high = cut
        spans.append((
            day_start + pd.Timedelta(hours=float(low)),
            day_start + pd.Timedelta(hours=float(high)),
        ))
    return spans


def _alpha_cut(dimension, term: str, alpha: float, grid: int = 512):
    mf = dimension.mf_as_dict(term)
    low, high = float(dimension.universe_min), float(dimension.universe_max)
    step = (high - low) / (grid - 1) if grid > 1 else 0.0
    hits = [
        low + i * step
        for i in range(grid)
        if membership_degree(low + i * step, mf) >= alpha
    ]
    if not hits:
        return None
    return min(hits), max(hits)


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
    .fd-narrative { line-height: 1.6; }
    .fd-clause { border-bottom: 1px dotted #c3ccd8; }
    .fd-linked { cursor: pointer; }
    .fd-linked:hover { background: #eef3fa; }
    .fd-linked:focus { outline: 2px solid #3D6FA8; outline-offset: 1px; }
    .fd-linked.fd-active { background: #ffe9a8; border-bottom-color: #E8A200; }
    .fd-hint { color: #777; font-size: 0.85em; margin: 4px 0 0; }
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
    {% if loop.first %}
    <p class="fd-hint">Click any sentence of a daily narrative to highlight the
    segment of the signal that supports it; click it again to restore the full
    view.</p>
    {% endif %}
  {% endfor %}
  <footer>Generated by FuzzyDiary.</footer>
  <script>
  (function () {
    var DIV_ID = "__SIGNAL_DIV_ID__";
    var HIGHLIGHT = {
      type: "rect", xref: "x", yref: "paper", y0: 0, y1: 1,
      fillcolor: "rgba(0,0,0,0)", layer: "above",
      line: { width: 2, color: "#111111" }
    };

    function isoLocal(ms) {
      var d = new Date(ms);
      return new Date(ms - d.getTimezoneOffset() * 60000)
        .toISOString()
        .slice(0, 19);
    }

    function chart() {
      var gd = document.getElementById(DIV_ID);
      return (gd && typeof Plotly !== "undefined" && gd.layout) ? gd : null;
    }

    function baseShapes(gd) {
      if (gd._fdBaseShapes === undefined) {
        gd._fdBaseShapes = (gd.layout.shapes || []).slice();
      }
      return gd._fdBaseShapes;
    }

    function clear(gd) {
      Plotly.relayout(gd, { shapes: baseShapes(gd), "xaxis.autorange": true });
    }

    function highlight(gd, spans) {
      var shapes = baseShapes(gd).slice();
      var lo = null, hi = null;
      spans.forEach(function (span) {
        var rect = Object.assign({}, HIGHLIGHT, { x0: span[0], x1: span[1] });
        shapes.push(rect);
        var a = Date.parse(span[0]), b = Date.parse(span[1]);
        if (lo === null || a < lo) { lo = a; }
        if (hi === null || b > hi) { hi = b; }
      });
      var update = { shapes: shapes };
      if (lo !== null && hi !== null) {
        var pad = Math.max((hi - lo) * 0.5, 30 * 60 * 1000);
        update["xaxis.range"] = [isoLocal(lo - pad), isoLocal(hi + pad)];
      }
      Plotly.relayout(gd, update);
    }

    function activate(el) {
      var gd = chart();
      if (!gd) { return; }
      var spans;
      try {
        spans = JSON.parse(el.getAttribute("data-fd-spans") || "[]");
      } catch (err) {
        spans = [];
      }
      var wasActive = el.classList.contains("fd-active");
      document.querySelectorAll(".fd-linked.fd-active").forEach(function (other) {
        other.classList.remove("fd-active");
      });
      if (wasActive || !spans.length) {
        clear(gd);
        return;
      }
      el.classList.add("fd-active");
      highlight(gd, spans);
      gd.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    document.addEventListener("click", function (event) {
      var el = event.target.closest(".fd-linked");
      if (el) { activate(el); }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") { return; }
      var el = event.target.closest && event.target.closest(".fd-linked");
      if (el) { event.preventDefault(); activate(el); }
    });
  })();
  </script>
</body>
</html>
""".replace("__SIGNAL_DIV_ID__", _SIGNAL_DIV_ID)


def _render_template(title: str, sections: list[dict]) -> str:
    return Template(_HTML_TEMPLATE).render(title=title, sections=sections)


def _escape_attr(text: str) -> str:
    return _escape(text).replace('"', "&quot;")


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
