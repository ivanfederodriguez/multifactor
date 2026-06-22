from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from charts import FACTOR_COLORS, capital_growth_chart, subfactor_comparison_bar_chart
from data_loader import (
    annual_subfactor_comparison,
    build_short_label,
    load_experiment_catalog,
    load_nav_series,
    load_weight_series,
)


APP_DIR = Path(__file__).resolve().parent
# Check if a local 'data' directory exists (for self-contained deployment)
if (APP_DIR / "data").exists():
    DEFAULT_EXPERIMENTS_ROOT = APP_DIR / "data"
else:
    DEFAULT_EXPERIMENTS_ROOT = APP_DIR.parent / "Prometheus-git" / "Output" / "Experiments"


PALETTE = [
    "#2563EB",
    "#159A67",
    "#F97316",
    "#8B5CF6",
    "#DC315B",
    "#0F766E",
    "#C98200",
    "#4772A8",
    "#B65C8A",
    "#6B7280",
    "#65A30D",
    "#0891B2",
]

FACTOR_LIGHT_COLORS = {
    "Value": "#DBEAFE",
    "Quality": "#D1FAE5",
    "Growth": "#FFEDD5",
    "Momentum": "#EDE9FE",
    "Revisions": "#FFE4E6",
    "Profitability momentum": "#FEF3C7",
    "Low volatility": "#E2E8F0",
}


st.set_page_config(
    page_title="Laboratorio Multifactor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #102a43;
        --muted: #64748b;
        --rule: #d7dee8;
        --soft: #f6f8fb;
        --primary: #1d4ed8;
    }
    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--ink);
    }
    .stApp { background: #ffffff; }
    .block-container {
        max-width: 1540px;
        padding-top: 1.45rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    h1 { font-size: 2rem !important; line-height: 1.08 !important; margin: 0 0 .15rem !important; }
    h2 { font-size: 1.18rem !important; margin-top: .4rem !important; }
    h3 { font-size: 1rem !important; }
    p, label, .stCaption { color: var(--muted); }
    div[data-testid="stHorizontalBlock"] { gap: .72rem; }
    div[data-testid="stDataEditor"] {
        border: 1px solid var(--rule);
        border-radius: 6px;
        overflow: hidden;
    }
    div[data-testid="stDataEditor"] [role="columnheader"] {
        background: #f5f7fa;
        color: var(--ink);
        font-weight: 650;
    }
    .matrix-scroll {
        max-height: 720px;
        overflow: auto;
        border: 1px solid var(--rule);
        border-radius: 6px;
    }
    .matrix-scroll table {
        border-collapse: separate;
        border-spacing: 0;
        min-width: 1320px;
        width: 100%;
        font-size: .72rem;
    }
    .matrix-scroll thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #f5f7fa;
        color: var(--ink);
        font-weight: 700;
        white-space: nowrap;
    }
    .matrix-scroll th, .matrix-scroll td {
        border-right: 1px solid #e3e8ef;
        border-bottom: 1px solid #e3e8ef;
        padding: .42rem .5rem;
        text-align: right;
        white-space: nowrap;
    }
    .matrix-scroll th:nth-child(2),
    .matrix-scroll th:nth-child(3),
    .matrix-scroll th:nth-child(4),
    .matrix-scroll td:nth-child(2),
    .matrix-scroll td:nth-child(3),
    .matrix-scroll td:nth-child(4) { text-align: left; }
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stTextInput"] > div > div > input,
    div[data-testid="stMultiSelect"] > div > div {
        border-radius: 6px !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid var(--rule);
        border-radius: 6px;
        box-shadow: none;
    }
    .section-rule {
        border-top: 1px solid var(--rule);
        margin: 1.1rem 0 .75rem;
    }
    .section-title {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin: 0 0 .5rem;
    }
    .section-title h2 { margin: 0 !important; }
    .section-title span { color: var(--muted); font-size: .8rem; }
    .selected-rail {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: .55rem;
        margin: .3rem 0 .2rem;
    }
    .selected-item {
        border: 1px solid var(--rule);
        border-radius: 6px;
        min-height: 52px;
        display: flex;
        align-items: stretch;
        background: #fff;
        overflow: hidden;
    }
    .selected-swatch { width: 5px; flex: 0 0 5px; }
    .selected-copy { padding: .45rem .65rem; min-width: 0; }
    .selected-key {
        color: var(--ink);
        font-size: .76rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .selected-meta {
        color: var(--muted);
        font-size: .7rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-top: .1rem;
    }
    .empty-state {
        border: 1px dashed #bcc7d4;
        border-radius: 6px;
        padding: 2.2rem 1rem;
        text-align: center;
        color: var(--muted);
        background: #fbfcfe;
    }
    div[data-testid="stMetric"] { border-left: 2px solid #dbe5f2; padding-left: .75rem; }
    #MainMenu, footer { visibility: hidden; }
    @media (max-width: 800px) {
        .block-container { padding: 1rem .8rem 3rem; }
        h1 { font-size: 1.65rem !important; }
        .selected-rail { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def catalog_version(root: Path) -> str:
    """Change the cache key when experiments are added or regenerated."""
    tracked = [root / "experiments_summary.csv"]
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        tracked.extend(
            [
                folder / "analytics.json",
                folder / "config.yaml",
                folder / "portfolio_nav.csv",
                folder / "trades.csv",
            ]
        )
    signature = []
    for path in tracked:
        if path.exists():
            stat = path.stat()
            signature.append((str(path.relative_to(root)), stat.st_mtime_ns, stat.st_size))
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:12]


@st.cache_data(show_spinner="Leyendo los experimentos…")
def cached_catalog(root: str, version: str) -> pd.DataFrame:
    return load_experiment_catalog(root)


@st.cache_data(show_spinner=False)
def cached_nav(root: str, run_keys: tuple[str, ...]) -> pd.DataFrame:
    return load_nav_series(root, run_keys)


@st.cache_data(show_spinner=False)
def cached_weights(root: str, run_keys: tuple[str, ...]) -> pd.DataFrame:
    return load_weight_series(root, run_keys)


def select_filter(label: str, values: list, key: str, format_func=None):
    options = ["Todos", *values]
    formatter = format_func or (lambda value: str(value))
    if st.session_state.get(key) not in options:
        st.session_state[key] = "Todos"
    return st.selectbox(
        label,
        options,
        key=key,
        format_func=lambda value: value if value == "Todos" else formatter(value),
    )


def reset_filters() -> None:
    for key in [
        "filter_schema",
        "filter_positions",
        "filter_vol",
        "filter_leverage",
        "filter_mode",
        "filter_window",
        "filter_dynamic",
    ]:
        st.session_state[key] = "Todos"


def filter_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    filtered = catalog.copy()
    mappings = {
        "filter_schema": "schema_label",
        "filter_positions": "n_positions",
        "filter_vol": "target_vol",
        "filter_leverage": "max_leverage",
        "filter_mode": "weights_mode_label",
        "filter_window": "training_years",
    }
    for state_key, column in mappings.items():
        value = st.session_state.get(state_key)
        if value not in (None, "Todos"):
            filtered = filtered[filtered[column] == value]
    dynamic_value = st.session_state.get("filter_dynamic")
    if dynamic_value not in (None, "Todos"):
        filtered = filtered[filtered["is_dynamic"] == (dynamic_value == "Sí")]
    return filtered.sort_values(
        ["cagr", "sharpe_ratio", "run_key"],
        ascending=[False, False, True],
    )


def table_frame(filtered: pd.DataFrame, selected: set[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "_run_key": filtered["run_key"],
            "Comparar": filtered["run_key"].isin(selected),
            "Esquema": filtered["schema_label"],
            "Pesos dinámicos": filtered["is_dynamic"].map({True: "Sí", False: "No"}),
            "Posiciones": filtered["n_positions"],
            "Vol. objetivo": filtered["target_vol"] * 100,
            "Apalancamiento": filtered["max_leverage"],
            "Ponderación": filtered["weights_mode_label"],
            "Ventana": filtered["training_years"],
            "CAGR": filtered["cagr"] * 100,
            "Sharpe": filtered["sharpe_ratio"],
            "Max drawdown": filtered["max_drawdown"] * 100,
            "Vol. anual": filtered["annual_volatility"] * 100,
            "Calmar": filtered["calmar_ratio"],
            "Sortino": filtered["sortino_ratio"],
            "Win rate": filtered["monthly_win_rate_pct"],
            "Retorno total": filtered["total_return"] * 100,
            "Días": filtered["total_trading_days"],
            "Trades": filtered["total_trades"],
        }
    )
    return table.reset_index(drop=True)


def render_selected_rail(catalog: pd.DataFrame, run_keys: list[str], colors: dict[str, str]) -> None:
    rows = catalog.set_index("run_key")
    items = []
    for run_key in run_keys:
        if run_key not in rows.index:
            continue
        row = rows.loc[run_key]
        items.append(
            f'<div class="selected-item">'
            f'<div class="selected-swatch" style="background:{colors[run_key]}"></div>'
            f'<div class="selected-copy">'
            f'<div class="selected-key">{build_short_label(row)}</div>'
            f'<div class="selected-meta">CAGR {row["cagr"]:.2%} · Sharpe {row["sharpe_ratio"]:.2f} · '
            f'Max DD {row["max_drawdown"]:.2%}</div>'
            f'</div></div>'
        )
    st.markdown(f'<div class="selected-rail">{"".join(items)}</div>', unsafe_allow_html=True)


st.title("Laboratorio Multifactor")
st.caption("Filtrá hiperparámetros, ordená resultados y compará las configuraciones seleccionadas")

if not DEFAULT_EXPERIMENTS_ROOT.exists():
    st.error(f"No encontré la carpeta de experimentos en: {DEFAULT_EXPERIMENTS_ROOT}")
    st.stop()

current_catalog_version = catalog_version(DEFAULT_EXPERIMENTS_ROOT)
catalog = cached_catalog(str(DEFAULT_EXPERIMENTS_ROOT), current_catalog_version)

if catalog.empty:
    st.warning("Todavía no hay experimentos completos para mostrar.")
    st.stop()

if "selected_experiments" not in st.session_state:
    best_per_family = (
        catalog.sort_values("cagr", ascending=False)
        .groupby("schema", sort=False, as_index=False)
        .head(1)
        .nlargest(3, "cagr")
    )
    st.session_state.selected_experiments = set(
        best_per_family["run_key"].tolist()
    )

filter_columns = st.columns([1.25, .9, .78, .9, .9, 1.1, .8, .68])
with filter_columns[0]:
    select_filter("Esquema", sorted(catalog["schema_label"].unique()), "filter_schema")
with filter_columns[1]:
    select_filter("Pesos dinámicos", ["Sí", "No"], "filter_dynamic")
with filter_columns[2]:
    select_filter("Posiciones", sorted(catalog["n_positions"].unique()), "filter_positions")
with filter_columns[3]:
    select_filter("Vol. objetivo", sorted(catalog["target_vol"].unique()), "filter_vol", lambda x: f"{x:.0%}")
with filter_columns[4]:
    select_filter("Apalancamiento", sorted(catalog["max_leverage"].unique()), "filter_leverage", lambda x: f"{x:.1f}x")
with filter_columns[5]:
    select_filter("Ponderación", sorted(catalog["weights_mode_label"].unique()), "filter_mode")
with filter_columns[6]:
    select_filter("Ventana", sorted(catalog["training_years"].unique()), "filter_window", lambda x: f"{x} años")
with filter_columns[7]:
    st.markdown("<div style='height:1.74rem'></div>", unsafe_allow_html=True)
    st.button("Limpiar", use_container_width=True, on_click=reset_filters)

filtered = filter_catalog(catalog)

filter_signature = tuple(
    st.session_state.get(key, "Todos")
    for key in [
        "filter_schema",
        "filter_dynamic",
        "filter_positions",
        "filter_vol",
        "filter_leverage",
        "filter_mode",
        "filter_window",
    ]
)

# Store the table's initial selection when the filters change
if "last_filter_signature" not in st.session_state or st.session_state.last_filter_signature != filter_signature:
    st.session_state.last_filter_signature = filter_signature
    st.session_state.table_initial_selection = set(st.session_state.selected_experiments)

selected_before = set(st.session_state.selected_experiments)
table = table_frame(filtered, st.session_state.table_initial_selection)

column_config = {
    "_run_key": None,
    "Esquema": st.column_config.TextColumn(width="medium"),
    "Pesos dinámicos": st.column_config.TextColumn(width="small"),
    "Posiciones": st.column_config.NumberColumn(format="%d"),
    "Vol. objetivo": st.column_config.NumberColumn(format="%.0f%%"),
    "Apalancamiento": st.column_config.NumberColumn(format="%.1fx"),
    "Ponderación": st.column_config.TextColumn(width="medium"),
    "Ventana": st.column_config.NumberColumn(format="%d años"),
    "CAGR": st.column_config.NumberColumn(format="%.2f%%"),
    "Sharpe": st.column_config.NumberColumn(format="%.2f"),
    "Max drawdown": st.column_config.NumberColumn(format="%.2f%%"),
    "Vol. anual": st.column_config.NumberColumn(format="%.2f%%"),
    "Calmar": st.column_config.NumberColumn(format="%.2f"),
    "Sortino": st.column_config.NumberColumn(format="%.2f"),
    "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
    "Retorno total": st.column_config.NumberColumn(format="%.1f%%"),
    "Días": st.column_config.NumberColumn(format="%d"),
    "Trades": st.column_config.NumberColumn(format="%d"),
    "Comparar": st.column_config.CheckboxColumn(width="small"),
}

editor_version = hashlib.sha1(repr(filter_signature).encode("utf-8")).hexdigest()[:10]
edited = st.data_editor(
    table,
    key=f"experiment_comparison_table_{current_catalog_version}_{editor_version}",
    hide_index=True,
    use_container_width=True,
    height=455,
    column_config=column_config,
    disabled=[column for column in table.columns if column != "Comparar"],
)

visible_keys = set(table["_run_key"])
checked_visible = set(edited.loc[edited["Comparar"], "_run_key"])
selected_after = (selected_before - visible_keys) | checked_visible
st.session_state.selected_experiments = selected_after

left_caption, right_action = st.columns([5, 1])
left_caption.caption(
    f"Mostrando {len(filtered)} de {len(catalog)} configuraciones · "
    "orden inicial: CAGR de mayor a menor"
)
if right_action.button("Actualizar", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

selected = [run_key for run_key in catalog["run_key"] if run_key in selected_after]
color_map = {run_key: PALETTE[index % len(PALETTE)] for index, run_key in enumerate(selected)}

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-title"><h2>Configuraciones seleccionadas</h2><span>{len(selected)} activas</span></div>',
    unsafe_allow_html=True,
)

if not selected:
    st.markdown(
        '<div class="empty-state">Marcá <strong>Comparar</strong> en una o más filas para activar los gráficos.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

render_selected_rail(catalog, selected, color_map)

if len(selected) > len(PALETTE):
    st.warning("Hay más de 12 configuraciones seleccionadas; algunos colores se repetirán. Para leer mejor los gráficos, conviene comparar entre 2 y 8.")

run_keys_tuple = tuple(selected)
colors = [color_map[run_key] for run_key in selected]
catalog_by_key = catalog.set_index("run_key")
label_map = {
    run_key: build_short_label(catalog_by_key.loc[run_key])
    for run_key in selected
}
display_domain = [label_map[run_key] for run_key in selected]
comparison_code_map = {
    run_key: f"C{index + 1}"
    for index, run_key in enumerate(selected)
}
comparison_label_map = {
    run_key: f"{comparison_code_map[run_key]} · {label_map[run_key]}"
    for run_key in selected
}
comparison_domain = [comparison_label_map[run_key] for run_key in selected]

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title"><h2>Crecimiento del capital</h2><span>Capital normalizado · Base 100</span></div>',
    unsafe_allow_html=True,
)
nav_long = cached_nav(str(DEFAULT_EXPERIMENTS_ROOT), run_keys_tuple)
nav_long["experiment"] = nav_long["experiment"].map(label_map)
st.altair_chart(capital_growth_chart(nav_long, display_domain, colors), use_container_width=True)

weights = cached_weights(str(DEFAULT_EXPERIMENTS_ROOT), run_keys_tuple)
if not weights.empty:
    weights["experiment"] = weights["experiment"].map(comparison_label_map)
annual_comparison = annual_subfactor_comparison(weights)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title"><h2>Comparación anual de subfactores</h2>'
    f'<span>{len(selected)} configuraciones · valores sin promediar entre configuraciones</span></div>',
    unsafe_allow_html=True,
)
if annual_comparison.empty:
    st.info("Las configuraciones seleccionadas no tienen series de pesos disponibles.")
else:
    available_years = sorted(annual_comparison["year"].unique(), reverse=True)
    available_factors = [
        factor
        for factor in FACTOR_COLORS
        if factor in set(annual_comparison["Factor"].dropna())
    ]
    year_control, factor_control, _ = st.columns([1, 1.55, 3.45])
    with year_control:
        selected_year = st.selectbox(
            "Año",
            available_years,
            key="subfactor_bar_year",
        )
    with factor_control:
        selected_factor = st.selectbox(
            "Factor",
            ["Todos", *available_factors],
            key="subfactor_factor_filter",
        )

    selected_year_data = annual_comparison[
        annual_comparison["year"] == selected_year
    ].copy()
    if selected_factor != "Todos":
        selected_year_data = selected_year_data[
            selected_year_data["Factor"] == selected_factor
        ]

    top_scores = (
        selected_year_data.groupby(
            ["subfactor_key", "Subfactor"],
            as_index=False,
        )["weight"]
        .max()
        .sort_values(
            ["weight", "Subfactor"],
            ascending=[False, True],
        )
        .head(10)
    )
    top_keys = top_scores["subfactor_key"].tolist()
    subfactor_order = top_scores["Subfactor"].tolist()
    rank_map = {
        subfactor: rank
        for rank, subfactor in enumerate(subfactor_order, start=1)
    }
    chart_data = selected_year_data[
        selected_year_data["subfactor_key"].isin(top_keys)
    ].copy()
    chart_data["rank"] = chart_data["Subfactor"].map(rank_map)

    st.caption(
        "El top 10 se define por el mayor peso individual observado en el año y factor elegidos. "
        "Cada panel conserva los valores propios de una configuración; no se promedian entre sí."
    )
    st.altair_chart(
        subfactor_comparison_bar_chart(
            chart_data,
            subfactor_order,
            comparison_domain,
        ),
        use_container_width=True,
    )

    st.markdown("### Matriz histórica de los subfactores comparados")
    st.caption(
        "Una fila por configuración y subfactor; una columna por año. "
        "El fondo de cada valor identifica el factor del subfactor. "
        "C1, C2, etc. corresponden a los paneles del gráfico."
    )
    history = annual_comparison[
        annual_comparison["subfactor_key"].isin(top_keys)
    ].copy()
    history["Ranking"] = history["Subfactor"].map(rank_map)
    year_columns = sorted(annual_comparison["year"].unique())
    annual_matrix = history.pivot(
        index=["Ranking", "experiment", "Subfactor", "Factor"],
        columns="year",
        values="weight",
    ).reset_index()
    annual_matrix = annual_matrix.rename(columns={"experiment": "Configuración"})
    year_labels = [str(year) for year in year_columns]
    annual_matrix = annual_matrix.rename(
        columns={year: str(year) for year in year_columns}
    )
    experiment_order = {
        experiment: order
        for order, experiment in enumerate(comparison_domain)
    }
    annual_matrix["_experiment_order"] = annual_matrix["Configuración"].map(
        experiment_order
    )
    annual_matrix = annual_matrix.sort_values(
        ["Ranking", "_experiment_order"]
    ).drop(columns="_experiment_order").reset_index(drop=True)
    comparison_code_by_label = {
        comparison_label_map[run_key]: comparison_code_map[run_key]
        for run_key in selected
    }
    annual_matrix["Configuración"] = annual_matrix["Configuración"].map(
        comparison_code_by_label
    )
    annual_matrix = annual_matrix[
        ["Ranking", "Configuración", "Subfactor", "Factor", *year_labels]
    ]

    matrix_styles = pd.DataFrame(
        "",
        index=annual_matrix.index,
        columns=annual_matrix.columns,
    )
    for row_index, factor in annual_matrix["Factor"].items():
        light_color = FACTOR_LIGHT_COLORS.get(factor, "#F1F5F9")
        dark_color = FACTOR_COLORS.get(factor, "#475569")
        matrix_styles.loc[row_index, "Factor"] = (
            f"color: {dark_color}; font-weight: 700;"
        )
        for year in year_labels:
            matrix_styles.loc[row_index, year] = (
                f"background-color: {light_color}; color: #102A43; font-weight: 600;"
            )

    styled_matrix = (
        annual_matrix.style
        .format({year: "{:.2%}" for year in year_labels}, na_rep="—")
        .apply(lambda _: matrix_styles, axis=None)
    )
    st.markdown(
        f'<div class="matrix-scroll">{styled_matrix.hide(axis="index").to_html()}</div>',
        unsafe_allow_html=True,
    )

st.caption(
    "Fuente: analytics.json, portfolio_nav.csv, trades.csv y dqi_active_weights_evolution.csv "
    "de cada configuración. Los promedios anuales se calculan por configuración, sin combinarlas."
)
