from __future__ import annotations

import altair as alt
import pandas as pd


GRID_COLOR = "#E5EAF0"
AXIS_COLOR = "#667085"
TITLE_COLOR = "#102A43"

FACTOR_COLORS = {
    "Value": "#2563EB",
    "Quality": "#159A67",
    "Growth": "#F97316",
    "Momentum": "#8B5CF6",
    "Revisions": "#DC315B",
    "Profitability momentum": "#C98200",
    "Low volatility": "#64748B",
}


def _color_encoding(domain: list[str], colors: list[str], title: str = "Configuración") -> alt.Color:
    return alt.Color(
        "experiment:N",
        title=title,
        scale=alt.Scale(domain=domain, range=colors),
        legend=alt.Legend(orient="bottom", columns=2, labelLimit=380, symbolStrokeWidth=3),
    )


def _configure(chart: alt.Chart | alt.LayerChart | alt.FacetChart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelColor=AXIS_COLOR,
            titleColor=TITLE_COLOR,
            gridColor=GRID_COLOR,
            domainColor="#C9D3DF",
            tickColor="#C9D3DF",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(labelColor=TITLE_COLOR, titleColor=TITLE_COLOR, labelFontSize=11)
        .configure_header(labelColor=TITLE_COLOR, titleColor=TITLE_COLOR, labelFontSize=12)
    )


def capital_growth_chart(data: pd.DataFrame, domain: list[str], colors: list[str]) -> alt.Chart:
    chart = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", grid=False, tickCount=13)),
            y=alt.Y(
                "base_100:Q",
                title="Capital (Base 100)",
                scale=alt.Scale(zero=False, nice=True),
            ),
            color=_color_encoding(domain, colors),
            tooltip=[
                alt.Tooltip("date:T", title="Fecha", format="%d/%m/%Y"),
                alt.Tooltip("experiment:N", title="Configuración"),
                alt.Tooltip("base_100:Q", title="Base 100", format=",.2f"),
                alt.Tooltip("nav:Q", title="Capital", format="$,.0f"),
            ],
        )
        .properties(height=360)
        .interactive(bind_y=False)
    )
    return _configure(chart)


def subfactor_timeline_chart(data: pd.DataFrame) -> alt.Chart:
    active_factors = [factor for factor in FACTOR_COLORS if factor in set(data["Factor"])]
    factor_colors = [FACTOR_COLORS[factor] for factor in active_factors]

    first_year = int(data["year"].min())
    last_year = int(data["year"].max())
    label_year = last_year + 0.12
    x_domain = [first_year, last_year + 1.8]
    latest = (
        data.sort_values("year")
        .groupby("Subfactor", as_index=False)
        .tail(1)
        .sort_values("weight")
        .copy()
    )
    latest["label_year"] = label_year
    minimum_gap = max(float(data["weight"].max()) * 0.03, 0.01)
    label_positions: list[float] = []
    for weight in latest["weight"]:
        next_position = max(float(weight), minimum_gap * 0.55)
        if label_positions:
            next_position = max(next_position, label_positions[-1] + minimum_gap)
        label_positions.append(next_position)
    latest["label_weight"] = label_positions
    latest["line_label"] = latest.apply(
        lambda row: f"{int(row['rank'])}. {row['Subfactor']}",
        axis=1,
    )

    hover = alt.selection_point(
        fields=["Subfactor"],
        nearest=True,
        on="pointerover",
        empty=False,
    )
    x = alt.X(
        "year:Q",
        title=None,
        scale=alt.Scale(domain=x_domain, nice=False),
        axis=alt.Axis(
            values=list(range(first_year, last_year + 1)),
            format="d",
            labelAngle=0,
            grid=False,
        ),
    )
    y_domain_end = max(float(data["weight"].max()), max(label_positions)) * 1.05
    y = alt.Y(
        "weight:Q",
        title="Peso medio",
        scale=alt.Scale(domain=[0, y_domain_end], nice=True),
        axis=alt.Axis(format=".0%"),
    )
    color = alt.Color(
        "Factor:N",
        title="Factor",
        scale=alt.Scale(domain=active_factors, range=factor_colors),
        legend=alt.Legend(
            orient="top",
            direction="horizontal",
            columns=4,
            symbolStrokeWidth=4,
            labelLimit=220,
        ),
    )

    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=x,
            y=y,
            color=color,
            detail=alt.Detail("Subfactor:N"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.78)),
            strokeWidth=alt.condition(hover, alt.value(3.4), alt.value(2.2)),
            tooltip=[
                alt.Tooltip("year:O", title="Año"),
                alt.Tooltip("rank:Q", title="Ranking", format=".0f"),
                alt.Tooltip("Subfactor:N", title="Subfactor"),
                alt.Tooltip("Factor:N", title="Factor"),
                alt.Tooltip("weight:Q", title="Peso", format=".2%"),
                alt.Tooltip("mean_weight:Q", title="Peso medio período", format=".2%"),
            ],
        )
    )
    annual_points = (
        alt.Chart(data)
        .mark_circle(size=42, filled=True, stroke="white", strokeWidth=0.6)
        .encode(
            x=x,
            y=y,
            color=color,
            detail=alt.Detail("Subfactor:N"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0.78)),
        )
    )
    selectors = (
        alt.Chart(data)
        .mark_point(opacity=0, size=80)
        .encode(x=x, y=y, detail="Subfactor:N")
        .add_params(hover)
    )
    endpoints = (
        alt.Chart(latest)
        .mark_circle(size=42, stroke="white", strokeWidth=1)
        .encode(
            x=x,
            y=y,
            color=color,
            tooltip=[
                alt.Tooltip("Subfactor:N", title="Subfactor"),
                alt.Tooltip("Factor:N", title="Factor"),
                alt.Tooltip("weight:Q", title="Último peso", format=".2%"),
            ],
        )
    )
    connectors = (
        alt.Chart(latest)
        .mark_rule(strokeWidth=0.8, opacity=0.55)
        .encode(
            x=alt.X("year:Q", scale=alt.Scale(domain=x_domain, nice=False)),
            x2="label_year:Q",
            y=alt.Y("weight:Q", scale=alt.Scale(domain=[0, y_domain_end], nice=True)),
            y2="label_weight:Q",
            color=color,
        )
    )
    labels = (
        alt.Chart(latest)
        .mark_text(align="left", baseline="middle", fontSize=10, fontWeight=600)
        .encode(
            x=alt.X("label_year:Q", scale=alt.Scale(domain=x_domain, nice=False)),
            y=alt.Y("label_weight:Q", scale=alt.Scale(domain=[0, y_domain_end], nice=True)),
            text="line_label:N",
            color=color,
        )
    )
    return _configure(
        (lines + annual_points + selectors + endpoints + connectors + labels)
        .properties(height=510)
        .interactive(bind_y=False)
    )


def subfactor_comparison_bar_chart(
    data: pd.DataFrame,
    subfactor_order: list[str],
    experiment_order: list[str],
) -> alt.Chart:
    """Compare annual subfactor weights in small multiples by experiment."""
    active_factors = [factor for factor in FACTOR_COLORS if factor in set(data["Factor"])]
    factor_colors = [FACTOR_COLORS[factor] for factor in active_factors]
    max_weight = max(float(data["weight"].max()) * 1.08, 0.01)
    chart_data = data.copy()
    chart_data["zero"] = 0.0
    chart_data["bar_label"] = chart_data.apply(
        lambda row: f"{row['Subfactor']} · {row['weight']:.1%}",
        axis=1,
    )

    x_scale = alt.Scale(domain=[0, max_weight], nice=True)
    color = alt.Color(
        "Factor:N",
        title="Factor",
        scale=alt.Scale(domain=active_factors, range=factor_colors),
        legend=alt.Legend(
            orient="top",
            direction="horizontal",
            columns=4,
            symbolStrokeWidth=4,
            labelLimit=220,
        ),
    )
    shared = alt.Chart(chart_data).encode(
        y=alt.Y(
            "Subfactor:N",
            sort=subfactor_order,
            title=None,
            axis=None,
        ),
        tooltip=[
            alt.Tooltip("experiment:N", title="Configuración"),
            alt.Tooltip("year:O", title="Año"),
            alt.Tooltip("rank:O", title="Ranking"),
            alt.Tooltip("Subfactor:N", title="Subfactor"),
            alt.Tooltip("Factor:N", title="Factor"),
            alt.Tooltip("weight:Q", title="Peso medio anual", format=".2%"),
        ],
    )
    bars = shared.mark_bar(
        height=24,
        opacity=0.52,
        strokeWidth=1.1,
        cornerRadiusEnd=2,
    ).encode(
        x=alt.X(
            "weight:Q",
            title="Peso medio anual",
            scale=x_scale,
            axis=alt.Axis(format=".0%", grid=True),
        ),
        color=color,
        stroke=color,
    )
    labels = shared.mark_text(
        align="left",
        baseline="middle",
        dx=7,
        color=TITLE_COLOR,
        fontSize=10,
        fontWeight=600,
    ).encode(
        x=alt.X("zero:Q", scale=x_scale),
        text=alt.Text("bar_label:N"),
    )
    panel = (bars + labels).properties(
        width=570,
        height=max(220, len(subfactor_order) * 31),
    )
    faceted = panel.facet(
        facet=alt.Facet(
            "experiment:N",
            title=None,
            sort=experiment_order,
            header=alt.Header(
                labelFontWeight=600,
                labelFontSize=11,
                labelLimit=560,
                labelPadding=10,
            ),
        ),
        columns=2,
    ).resolve_scale(x="shared")
    return _configure(faceted)


def factor_chart(
    data: pd.DataFrame,
    domain: list[str],
    colors: list[str],
    factor: str | None,
) -> alt.Chart:
    color = _color_encoding(domain, colors)
    if factor:
        filtered = data[data["Factor"] == factor]
        chart = (
            alt.Chart(filtered)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", grid=False, tickCount=13)),
                y=alt.Y("weight:Q", title="Peso", axis=alt.Axis(format=".0%")),
                color=color,
                tooltip=[
                    alt.Tooltip("date:T", title="Fecha", format="%b %Y"),
                    alt.Tooltip("experiment:N", title="Configuración"),
                    alt.Tooltip("Factor:N", title="Factor"),
                    alt.Tooltip("weight:Q", title="Peso", format=".2%"),
                ],
            )
            .properties(height=300)
            .interactive(bind_y=False)
        )
        return _configure(chart)

    base = (
        alt.Chart(data)
        .mark_line(strokeWidth=1.7)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", grid=False, tickCount=13)),
            y=alt.Y("weight:Q", title="Peso", axis=alt.Axis(format=".0%")),
            color=color,
            tooltip=[
                alt.Tooltip("date:T", title="Fecha", format="%b %Y"),
                alt.Tooltip("experiment:N", title="Configuración"),
                alt.Tooltip("Factor:N", title="Factor"),
                alt.Tooltip("weight:Q", title="Peso", format=".2%"),
            ],
        )
        .properties(width=520, height=150)
    )
    faceted = base.facet(
        facet=alt.Facet("Factor:N", title=None, header=alt.Header(labelFontWeight=600)),
        columns=2,
    ).resolve_scale(y="independent")
    return _configure(faceted)


def subfactor_bar_chart(
    data: pd.DataFrame,
    domain: list[str],
    colors: list[str],
) -> alt.Chart:
    base = (
        alt.Chart(data)
        .mark_bar(size=11, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0, grid=False)),
            xOffset=alt.XOffset("experiment:N", sort=domain),
            y=alt.Y("weight:Q", title="Peso medio", axis=alt.Axis(format=".0%")),
            color=_color_encoding(domain, colors),
            tooltip=[
                alt.Tooltip("year:O", title="Año"),
                alt.Tooltip("experiment:N", title="Configuración"),
                alt.Tooltip("Subfactor:N", title="Subfactor"),
                alt.Tooltip("weight:Q", title="Peso medio", format=".2%"),
            ],
        )
        .properties(width=520, height=220)
    )
    faceted = base.facet(
        facet=alt.Facet("Subfactor:N", title=None, header=alt.Header(labelFontWeight=600)),
        columns=2,
    ).resolve_scale(y="independent")
    return _configure(faceted)
