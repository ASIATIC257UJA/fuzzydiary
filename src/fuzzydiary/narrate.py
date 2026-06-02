from __future__ import annotations

from fuzzydiary.describe import DailySummary


def narrate_daily(daily: DailySummary, context_order: list[str] | None = None) -> str:
    has_events = any(ed.statements for ed in daily.events)
    has_protoforms = bool(daily.day_protoforms)
    if not has_events and not has_protoforms:
        return f"No describable activity on {daily.day.date()}."

    lines: list[str] = [f"On {daily.day.date()}:"]

    if has_events:
        lines.append("  Detected events:")
        for ed in daily.events:
            ctx_phrase = ""
            if ed.contexts and "day_moment" in ed.contexts:
                ctx_phrase = f" in the {ed.contexts['day_moment'][0]}"
            elif ed.contexts:
                first_ctx = next(iter(ed.contexts.values()))
                ctx_phrase = f" in the {first_ctx[0]}"
            lines.append(f"    {ed.event.kind}{ctx_phrase}.")

    if has_protoforms:
        lines.append("  Daily linguistic summary:")
        if context_order:
            def _order_key(st):
                text_lower = st.text.lower()
                for i, term in enumerate(context_order):
                    if term.lower() in text_lower:
                        return i
                return len(context_order)
            ordered = sorted(daily.day_protoforms, key=_order_key)
        else:
            ordered = sorted(daily.day_protoforms, key=lambda s: s.truth, reverse=True)
        for st in ordered:
            lines.append(f"    {st.text} (DoT={st.truth:.2f}).")

    return "\n".join(lines)
