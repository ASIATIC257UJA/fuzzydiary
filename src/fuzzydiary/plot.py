from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuzzydiary.config import FuzzyDiaryConfig
    from fuzzydiary.events import EventCollection
    from fuzzydiary.io import Series
    from fuzzydiary.simplify import SimplifiedSeries

_DEFAULT_PALETTE = [
    "#3D6FA8", "#2E7D54", "#C0392B", "#8C5320",
    "#6E3E70", "#A0522D", "#5C8A8A", "#B7950B",
]


def plot_events(
    series: "Series",
    simplified: "SimplifiedSeries",
    events: "EventCollection",
    cfg: "FuzzyDiaryConfig",
    figsize: tuple[float, float] = (7.4, 6.6),
    dpi: int = 130,
):
    """
    Generate a three-panel figure:
      1. Original signal (black line).
      2. RDP piecewise-linear simplification.
      3. Declarative events as coloured bands with extremum markers.

    Colours are taken from ``cfg.events.event_colors``; unlisted events
    receive colours from the default palette.

    Parameters
    ----------
    series : Series
    simplified : SimplifiedSeries
    events : EventCollection
    cfg : FuzzyDiaryConfig
    figsize, dpi : matplotlib figure parameters

    Returns
    -------
    matplotlib.figure.Figure
        The figure object; call ``fig.savefig(...)`` or ``fig.show()``
        as needed.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    config_colors: dict[str, str] = cfg.events.event_colors or {}

    def _color_for(kind: str, seen: dict) -> str:
        if kind not in seen:
            if kind in config_colors:
                seen[kind] = config_colors[kind]
            else:
                idx = len([k for k in seen if k not in config_colors])
                seen[kind] = _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]
        return seen[kind]

    day = next(iter(series.days))
    raw = series.days[day]
    rdp = simplified.days[day]
    day_events = events.days.get(day, [])

    def H(idx):
        return np.array([t.hour + t.minute / 60.0 for t in idx])

    x_raw, y_raw = H(raw.index), raw["value"].to_numpy()
    x_rdp, y_rdp = H(rdp.index), rdp["value"].to_numpy()

    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(x_raw, y_raw, color="black", lw=1.0)
    axes[0].set_ylabel(f"{cfg.signal.signal_name} ({cfg.signal.unit or 'value'})")
    axes[0].set_title(f"Original signal  ({len(x_raw)} samples @ "
                      f"{cfg.signal.sampling_period_minutes:.0f} min)", fontsize=9, loc="left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(x_raw, y_raw, color="#cccccc", lw=0.8, label="original")
    axes[1].plot(x_rdp, y_rdp, color="#2E7D54", lw=1.3, marker="o", ms=3.5,
                 label=f"RDP ({len(x_rdp)} pts, ε auto)")
    axes[1].set_ylabel(f"{cfg.signal.signal_name} ({cfg.signal.unit or 'value'})")
    axes[1].set_title("RDP piecewise-linear simplification", fontsize=9, loc="left")
    axes[1].legend(fontsize=7.5, loc="upper right")
    axes[1].grid(alpha=0.2)

    axes[2].plot(x_raw, y_raw, color="black", lw=0.9)
    seen: dict[str, str] = {}
    for ev in day_events:
        if ev.composite:
            continue
        color = _color_for(ev.kind, seen)
        x0 = ev.start.hour + ev.start.minute / 60
        x1 = ev.end.hour + ev.end.minute / 60
        lbl = ev.kind if ev.kind not in [k for k in seen if k != ev.kind] else "_nolegend_"
        legend_label = ev.kind if ev.kind not in getattr(axes[2], "_fd_seen", set()) else "_nolegend_"
        if not hasattr(axes[2], "_fd_seen"):
            axes[2]._fd_seen = set()
        if ev.kind not in axes[2]._fd_seen:
            axes[2]._fd_seen.add(ev.kind)
            axes[2].axvspan(x0, x1, color=color, alpha=0.22, label=ev.kind)
        else:
            axes[2].axvspan(x0, x1, color=color, alpha=0.22)
        ct = ev.attributes.get("characteristic_time")
        cv = ev.attributes.get("characteristic_value")
        if ct is not None and cv is not None and "recovery" not in ev.kind.lower():
            xt = ct.hour + ct.minute / 60
            marker = "v" if any(x in ev.kind.lower() for x in ("hypo", "low")) else "^"
            axes[2].plot(xt, cv, marker=marker, color=color, ms=8, zorder=5,
                         markeredgecolor="white", markeredgewidth=0.5)

    axes[2].set_ylabel(f"{cfg.signal.signal_name} ({cfg.signal.unit or 'value'})")
    axes[2].set_xlabel("hour of day")
    axes[2].set_title("Declarative event catalogue (level-based)", fontsize=9, loc="left")
    axes[2].set_xlim(0, 24)
    axes[2].set_xticks(range(0, 25, 3))
    axes[2].grid(alpha=0.2)
    axes[2].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.28),
        ncol=3, fontsize=7.5, frameon=True,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    return fig
