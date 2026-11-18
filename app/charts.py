"""Altair charts for the Streamlit app (dataviz rules: fixed categorical order,
thin marks with rounded data ends, tooltips on every mark, legend for >= 2 series,
recessive guides, a table view next to each chart)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# Reference palette, categorical slots 1-2 (validated adjacent pair, light mode).
SERIES = ["#2a78d6", "#eb6834"]
GUIDE = "#8a8984"  # recessive reference lines (quantile markers)


#: Pixels per category row in horizontal bar charts, and room for axes / legend.
ROW = 26
FRAME = 110
#: Long right tails (wide recruitment priors) are cut at this quantile on screen.
TAIL_QUANTILE = 0.99


def _money_axis(title: str) -> alt.Axis:
    return alt.Axis(title=title, format="~s", labelExpr="'€' + datum.label")


def _category_axis() -> alt.Axis:
    """Every category label shown (no overlap culling), long names allowed."""
    return alt.Axis(title=None, labelOverlap=False, labelLimit=260)


def element_height(n_rows: int) -> int:
    """Pixel height to pass to st.altair_chart for an n-row horizontal bar chart."""
    return ROW * n_rows + FRAME


def tail_cut(*arrays: np.ndarray) -> float:
    """Upper display limit shared by several distributions (largest P99)."""
    return float(max(np.quantile(a, TAIL_QUANTILE) for a in arrays))


def histogram(values: np.ndarray, title: str, money: bool = True) -> alt.LayerChart:
    """Distribution of one quantity with P10 / P50 / P90 markers.

    The x-axis stops at P99 so a long right tail does not squash the body; the
    caller says so in a caption. Quantile labels sit on three rows so close
    quantiles never overlap.
    """
    upper = tail_cut(values)
    df = pd.DataFrame({"value": values[values <= upper]})
    q = pd.DataFrame(
        {
            "label": ["P10", "P50", "P90"],
            "value": np.quantile(values, [0.1, 0.5, 0.9]),
            "ypix": [0, 13, 26],  # staggered label rows, in pixels from the top
        }
    )
    axis = _money_axis(title) if money else alt.Axis(title=title)
    fmt = ",.0f" if money else ".1f"
    bins = alt.Bin(maxbins=40)
    bars = (
        alt.Chart(df)
        .mark_bar(color=SERIES[0], cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("value:Q", bin=bins, axis=axis),
            y=alt.Y("count():Q", title="Simulated trials"),
            tooltip=[
                alt.Tooltip("value:Q", bin=bins, title=title, format=fmt),
                alt.Tooltip("count():Q", title="Trials"),
            ],
        )
    )
    rules = (
        alt.Chart(q)
        .mark_rule(color=GUIDE, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="value:Q", tooltip=["label", alt.Tooltip("value:Q", format=fmt)])
    )
    labels = (
        alt.Chart(q)
        .mark_text(align="left", dx=4, baseline="top", color=GUIDE, fontSize=11)
        .encode(x="value:Q", y=alt.Y("ypix:Q", scale=None), text="label")
    )
    return (bars + rules + labels).properties(height=240)


def breakdown(df: pd.DataFrame, dimension: str) -> alt.LayerChart:
    """Mean cost per group with the P10-P90 range as a whisker. ``df`` from report.breakdown."""
    data = df.reset_index().rename(columns={df.index.name or "index": "group"})
    data["share_pct"] = 100 * data["share"]
    order = data.sort_values("mean", ascending=False)["group"].tolist()
    y = alt.Y("group:N", sort=order, axis=_category_axis())
    tooltip = [
        alt.Tooltip("group:N", title=dimension.capitalize()),
        alt.Tooltip("mean:Q", title="Mean €", format=",.0f"),
        alt.Tooltip("P10:Q", format=",.0f"),
        alt.Tooltip("P50:Q", format=",.0f"),
        alt.Tooltip("P90:Q", format=",.0f"),
        alt.Tooltip("share_pct:Q", title="Share of mean total, %", format=".1f"),
    ]
    bars = (
        alt.Chart(data)
        .mark_bar(color=SERIES[0], cornerRadiusEnd=4, height={"band": 0.7})
        .encode(x=alt.X("mean:Q", axis=_money_axis("Mean cost")), y=y, tooltip=tooltip)
    )
    whiskers = (
        alt.Chart(data)
        .mark_rule(color=GUIDE, strokeWidth=1.5)
        .encode(x="P10:Q", x2="P90:Q", y=y, tooltip=tooltip)
    )
    return (bars + whiskers).properties(height=ROW * len(data))


TORNADO_SIDES = ["Input low (P10, or 0.5x if fixed)", "Input high (P90, or 1.5x if fixed)"]


def tornado(df: pd.DataFrame) -> alt.LayerChart:
    """Change in mean total cost when each input moves to its low / high value.

    ``df`` from report.tornado: one bar segment from the base to the cost at the
    low input (slot 1) and one to the cost at the high input (slot 2).
    """
    base = float(df["base"].iloc[0])
    rows = []
    for _, r in df.iterrows():
        for side, cost, label in (
            (TORNADO_SIDES[0], r.cost_at_low, r.low_input),
            (TORNADO_SIDES[1], r.cost_at_high, r.high_input),
        ):
            rows.append(
                {
                    "parameter": r.parameter,
                    "side": side,
                    "start": base,
                    "end": cost,
                    "delta": cost - base,
                    "input": label,
                }
            )
    data = pd.DataFrame(rows)
    order = df["parameter"].tolist()  # already sorted by swing
    bars = (
        alt.Chart(data)
        .mark_bar(height={"band": 0.6}, cornerRadius=2)
        .encode(
            y=alt.Y("parameter:N", sort=order, axis=_category_axis()),
            x=alt.X("start:Q", axis=_money_axis("Mean total cost"), scale=alt.Scale(zero=False)),
            x2="end:Q",
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=TORNADO_SIDES, range=SERIES),
                legend=alt.Legend(title=None, orient="top", labelLimit=320),
            ),
            tooltip=[
                "parameter",
                "side",
                alt.Tooltip("input:N", title="Input value"),
                alt.Tooltip("end:Q", title="Mean total €", format=",.0f"),
                alt.Tooltip("delta:Q", title="Δ vs base €", format="+,.0f"),
            ],
        )
    )
    base_rule = (
        alt.Chart(pd.DataFrame({"x": [base]}))
        .mark_rule(color=GUIDE, strokeWidth=1.5)
        .encode(x="x:Q")
    )
    return (bars + base_rule).properties(height=ROW * len(order))


# Categorical slot 3 (aqua) for a third scenario; the first three slots validate
# all-pairs in both modes (palette reference), so overlays are capped at three.
SCENARIOS = [*SERIES, "#1baf7a"]
# Diverging poles (reference palette): blue = cheaper than baseline, red = dearer.
DIVERGING = {"lower": "#2a78d6", "higher": "#e34948"}


def ecdf(values: dict[str, np.ndarray]) -> alt.Chart:
    """Cumulative distribution of total cost per scenario: read P(cost <= x) directly."""
    upper = tail_cut(*values.values())
    frames = []
    for name, x in values.items():
        xs = np.sort(x)
        xs = xs[xs <= upper]  # display cut at P99; probabilities still over all trials
        idx = np.linspace(0, len(xs) - 1, min(len(xs), 400)).astype(int)  # thin for the browser
        frames.append(pd.DataFrame({"scenario": name, "cost": xs[idx], "p": (idx + 1) / len(x)}))
    data = pd.concat(frames)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2, interpolate="step-after")
        .encode(
            x=alt.X(
                "cost:Q", axis=_money_axis("Total cost"), scale=alt.Scale(zero=False, nice=False)
            ),
            y=alt.Y("p:Q", title="P(total cost ≤ x)", axis=alt.Axis(format="%")),
            color=alt.Color(
                "scenario:N",
                sort=list(values),
                scale=alt.Scale(domain=list(values), range=SCENARIOS[: len(values)]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                "scenario",
                alt.Tooltip("cost:Q", title="Total cost €", format=",.0f"),
                alt.Tooltip("p:Q", title="P(cost ≤ x)", format=".0%"),
            ],
        )
        .properties(height=280)
    )


def delta_bars(delta: pd.Series, title: str) -> alt.Chart:
    """Change in mean cost per group vs the baseline (diverging colours by sign)."""
    data = delta.rename("delta").reset_index()
    data.columns = ["group", "delta"]
    data["direction"] = np.where(data["delta"] < 0, "lower", "higher")
    order = data["group"].tolist()
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
        .encode(
            y=alt.Y("group:N", sort=order, axis=_category_axis()),
            x=alt.X("delta:Q", axis=_money_axis(title)),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=list(DIVERGING), range=list(DIVERGING.values())),
                legend=alt.Legend(title="vs baseline", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("delta:Q", title="Δ mean €", format="+,.0f"),
            ],
        )
        .properties(height=ROW * len(data))
    )
