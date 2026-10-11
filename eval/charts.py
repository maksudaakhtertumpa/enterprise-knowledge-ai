"""Tiny dependency-free SVG charts for the evaluation report.

Two chart types are enough here: grouped bars (compare settings or datasets) and
lines (a metric against a threshold). Values are printed on the chart and every
series has its own marker shape, so nothing depends on colour alone. Output is
plain SVG text that GitHub renders inline in Markdown.
"""

from html import escape

# Okabe-Ito palette: distinguishable with the common forms of colour blindness.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d1d9e0"


def _text(x, y, s, size=12, anchor="middle", color=INK, weight="normal", extra=""):
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' "
        f"text-anchor='{anchor}' font-weight='{weight}' {extra}>{escape(str(s))}</text>"
    )


def _frame(width, height, title, subtitle, desc):
    head = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-labelledby='t d'>",
        f"<title id='t'>{escape(title)}</title><desc id='d'>{escape(desc)}</desc>",
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
        _text(16, 26, title, 16, "start", weight="600"),
    ]
    if subtitle:
        head.append(_text(16, 46, subtitle, 12, "start", MUTED))
    return head


def _y_axis(parts, left, top, plot_w, plot_h, y_max, ticks, label, pct):
    for i in range(ticks + 1):
        v = y_max * i / ticks
        y = top + plot_h - plot_h * i / ticks
        parts.append(
            f"<line x1='{left}' x2='{left + plot_w}' y1='{y:.1f}' y2='{y:.1f}' "
            f"stroke='{GRID}' stroke-width='1'/>"
        )
        parts.append(_text(left - 8, y + 4, f"{v:.0%}" if pct else f"{v:g}", 11, "end", MUTED))
    if label:
        cy = top + plot_h / 2
        parts.append(
            _text(16, cy, label, 12, "middle", MUTED, extra=f"transform='rotate(-90 16 {cy:.1f})'")
        )


def _legend(parts, names, x, y, markers=False):
    for i, name in enumerate(names):
        color = PALETTE[i % len(PALETTE)]
        if markers:
            parts.append(_marker(i, x + 6, y - 4, color))
        else:
            parts.append(f"<rect x='{x}' y='{y - 10}' width='12' height='12' fill='{color}'/>")
        parts.append(_text(x + 18, y, name, 12, "start"))
        x += 30 + 7 * len(name)


def _marker(i, x, y, color):
    shape = i % 3
    if shape == 0:
        return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>"
    if shape == 1:
        return f"<rect x='{x - 4:.1f}' y='{y - 4:.1f}' width='8' height='8' fill='{color}'/>"
    return (
        f"<path d='M{x:.1f} {y - 5:.1f} L{x + 5:.1f} {y + 4:.1f} L{x - 5:.1f} {y + 4:.1f} Z' "
        f"fill='{color}'/>"
    )


def bar_chart(
    title: str,
    categories: list[str],
    series: dict[str, list[float | None]],
    *,
    subtitle: str = "",
    y_label: str = "",
    y_max: float = 1.0,
    pct: bool = True,
    width: int = 720,
    height: int = 360,
) -> str:
    """Grouped vertical bars: one group per category, one bar per series. None = no bar."""
    names = list(series)
    left, right, top, bottom = 64, 16, 84, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = _frame(
        width,
        height,
        title,
        subtitle,
        "; ".join(
            f"{c}: " + ", ".join(f"{n} {_fmt(series[n][i], pct)}" for n in names)
            for i, c in enumerate(categories)
        ),
    )
    _legend(parts, names, left, 68)
    _y_axis(parts, left, top, plot_w, plot_h, y_max, 4, y_label, pct)
    group_w = plot_w / max(len(categories), 1)
    bar_w = min(42.0, group_w * 0.8 / max(len(names), 1))
    for i, cat in enumerate(categories):
        gx = left + group_w * i + (group_w - bar_w * len(names)) / 2
        for j, name in enumerate(names):
            v = series[name][i]
            x = gx + j * bar_w
            if v is None:
                parts.append(_text(x + bar_w / 2, top + plot_h - 4, "n/a", 10, color=MUTED))
                continue
            h = plot_h * max(0.0, min(v, y_max)) / y_max
            color = PALETTE[j % len(PALETTE)]
            parts.append(
                f"<rect x='{x + 1:.1f}' y='{top + plot_h - h:.1f}' width='{bar_w - 2:.1f}' "
                f"height='{h:.1f}' fill='{color}'/>"
            )
            parts.append(_text(x + bar_w / 2, top + plot_h - h - 4, _fmt(v, pct, short=True), 10))
        parts.append(_text(left + group_w * (i + 0.5), top + plot_h + 20, cat, 12))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def line_chart(
    title: str,
    x: list[float],
    series: dict[str, list[float | None]],
    *,
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    y_max: float = 1.0,
    pct: bool = True,
    marker_x: float | None = None,
    marker_label: str = "",
    width: int = 720,
    height: int = 380,
) -> str:
    """Lines over a numeric x axis (e.g. a threshold). Optional vertical marker line."""
    names = list(series)
    left, right, top, bottom = 64, 24, 84, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = min(x), max(x)
    span = (x_max - x_min) or 1.0

    def px(v):
        return left + plot_w * (v - x_min) / span

    def py(v):
        return top + plot_h - plot_h * max(0.0, min(v, y_max)) / y_max

    parts = _frame(
        width,
        height,
        title,
        subtitle,
        "; ".join(
            f"{n}: "
            + ", ".join(f"{xv:g}→{_fmt(v, pct)}" for xv, v in zip(x, series[n], strict=True))
            for n in names
        ),
    )
    _legend(parts, names, left, 68, markers=True)
    _y_axis(parts, left, top, plot_w, plot_h, y_max, 4, y_label, pct)
    for xv in x:
        parts.append(_text(px(xv), top + plot_h + 18, f"{xv:g}", 11, color=MUTED))
    if x_label:
        parts.append(_text(left + plot_w / 2, height - 14, x_label, 12, color=MUTED))
    if marker_x is not None:
        mx = px(marker_x)
        parts.append(
            f"<line x1='{mx:.1f}' x2='{mx:.1f}' y1='{top}' y2='{top + plot_h}' "
            f"stroke='{MUTED}' stroke-dasharray='4 4'/>"
        )
        if marker_label:
            parts.append(_text(mx + 6, top + 12, marker_label, 11, "start", MUTED))
    for j, name in enumerate(names):
        color = PALETTE[j % len(PALETTE)]
        points = [(px(xv), py(v)) for xv, v in zip(x, series[name], strict=True) if v is not None]
        if len(points) > 1:
            path = " ".join(
                f"{'M' if k == 0 else 'L'}{a:.1f} {b:.1f}" for k, (a, b) in enumerate(points)
            )
            parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        parts.extend(_marker(j, a, b, color) for a, b in points)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _fmt(v: float | None, pct: bool, short: bool = False) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v:.0%}" if short else f"{v:.1%}"
    return f"{v:.2f}" if v < 10 else f"{v:.0f}"
