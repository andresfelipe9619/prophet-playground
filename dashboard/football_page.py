"""The football page: the data layer and the market baseline, and nothing more.

Run through `dashboard/app.py`, which owns the sidebar's domain selector.

**What this page deliberately does not do.** There is no football model in this
repository yet, and no scoring rule for a three-way outcome, so nothing here
compares a forecast against the market. Every number on this page describes
either the data or the baseline itself. That distinction is the whole point of
the football half of the project: the bar is the closing line, and a page that
implied otherwise — by showing a "model" column, or by scoring anything — would
be claiming a result that does not exist.

What it *is* useful for: seeing which odds source a file resolved to before you
build anything on it, seeing how big the bookmaker's margin is, and seeing that
the market is calibrated — which is what makes it a hard baseline rather than a
convenient one.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football.common import DEFAULT_DATA_DIR, ODDS_COLUMNS, OUTCOMES, PROBABILITY_COLUMNS
from football.market import METHODS, compare_methods, market_probabilities, overround
from football.processor import (
    CLOSING_SOURCES,
    MatchFormatError,
    check_match_format,
    load_seasons,
    odds_coverage,
    preprocess_matches,
)
from football.sample_data import generate_matches

from dashboard.ui import HELP, chart, section

# Spanish labels for the three outcomes, since OUTCOME_LABELS is English (it is
# code-facing). The order follows OUTCOMES, which is load-bearing everywhere.
OUTCOME_ES = {"H": "Local", "D": "Empate", "A": "Visitante"}

# Bins for the calibration curve. Wide enough that each carries enough matches
# for its observed rate to mean something on one season of data.
CALIBRATION_BINS = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


def season_files(directory):
    """The season CSVs sitting in `directory`, newest name last."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".csv"))


@st.cache_data(show_spinner=False)
def load_matches(uploaded_bytes, paths, closing_odds_only):
    """Load matches from an upload, from season files, or fall back to synthetic ones.

    Returns `(matches, is_demo, report)`. `load_seasons` is allowed to raise
    through: refusing to concatenate files that resolve to different odds
    sources is the guard this domain exists around, and swallowing it here to
    show *something* would defeat it. The caller renders the message.
    """
    if uploaded_bytes is not None:
        matches = preprocess_matches(pd.read_csv(io.BytesIO(uploaded_bytes)), validate=False)
        is_demo = False
    elif paths:
        matches = load_seasons(list(paths), validate=False, closing_odds_only=closing_odds_only)
        is_demo = False
    else:
        matches = preprocess_matches(
            generate_matches(seed=0).drop(columns=["TrueH", "TrueD", "TrueA"]), validate=False)
        is_demo = True

    return matches, is_demo, check_match_format(matches)


def render():
    st.title("⚽ Fútbol")
    st.caption(
        "Capa de datos y línea base del mercado. La barra que un modelo de fútbol tiene que superar "
        "es la **cuota de cierre**, no un Elo ni un 50/50 — y todavía no hay ningún modelo en este "
        "repositorio, así que nada de esta página compara un pronóstico contra el mercado."
    )

    with st.sidebar:
        st.header("Datos")
        uploaded = st.file_uploader("CSV de football-data.co.uk", type="csv")
        directory = st.text_input("Carpeta de temporadas", value=DEFAULT_DATA_DIR)
        available = season_files(directory)
        chosen = st.multiselect(
            "Temporadas", available, default=available[-1:],
            help=HELP["fb_seasons"],
        )
        closing_odds_only = st.checkbox(
            "Solo temporadas con cuotas de cierre", value=False, help=HELP["fb_closing_only"])
        method = st.selectbox(
            "Método para quitar el margen", METHODS, index=0, help=HELP["fb_method"])

    paths = tuple(os.path.join(directory, name) for name in chosen) if not uploaded else ()

    try:
        matches, is_demo, report = load_matches(
            uploaded.getvalue() if uploaded else None, paths, closing_odds_only)
    except MatchFormatError as exc:
        st.error(
            "**No se pudieron cargar estos archivos juntos.** Cada archivo resuelve a una sola "
            "fuente de cuotas, y juntar dos fuentes distintas pondría un precio de cierre y uno de "
            "apertura en la misma columna: todo lo que se midiera ahí se estaría comparando contra "
            "dos barras a la vez. Elige las temporadas que comparten fuente, o marca **Solo "
            "temporadas con cuotas de cierre**."
        )
        # The original message names the sources involved; it is English because
        # it comes from the domain layer, where everything is.
        st.caption(f"Detalle: {exc}")
        st.stop()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if is_demo:
        st.info(
            "No hay archivos de temporada en la carpeta indicada ni se subió uno — mostrando "
            "**datos sintéticos** (una temporada generada con fuerzas de ataque y defensa por equipo "
            "y una casa de apuestas simulada). Sirven para recorrer la página, no para concluir nada. "
            "Descarga temporadas reales con `python -m football.downloader --seasons 2023/24 "
            "--leagues E0`."
        )

    source = matches.attrs.get("odds_source")
    if source is None:
        st.error(
            "**Estos partidos no traen cuotas.** Se puede entrenar un modelo con ellos, pero no se "
            "puede medir: sin precio no hay línea base, y en fútbol superar el azar no es la barra."
        )
    elif source not in CLOSING_SOURCES:
        st.warning(
            f"**Las mejores cuotas de este archivo son de apertura ({source}).** football-data "
            "publica cuotas de cierre (las columnas con «C») solo desde 2019/20. Las de apertura son "
            "blandas: superarlas es superar la primera estimación de la casa, no al mercado."
        )
    if report and report.get("odds_coverage", 1.0) < 1.0:
        st.info(
            f"Solo **{report['odds_coverage']:.1%}** de los partidos tienen un precio utilizable de "
            f"{source}. El resto queda en NaN a propósito: rellenarlo desde otra columna mezclaría "
            "dos mercados distintos en una sola línea base."
        )

    priced = matches.dropna(subset=list(ODDS_COLUMNS)) if source else matches.iloc[:0]
    with_probabilities = market_probabilities(priced, method=method) if len(priced) else priced

    tabs = st.tabs(["Datos", "Mercado", "Resultados"])

    # ------------------------------------------------------------------ Datos
    with tabs[0]:
        section("Estado de los datos", "fb_tab_datos")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidos", len(matches), help=HELP["fb_n_matches"])
        c2.metric("Desde", matches["ds"].min().strftime("%Y-%m-%d") if len(matches) else "—")
        c3.metric("Hasta", matches["ds"].max().strftime("%Y-%m-%d") if len(matches) else "—")
        c4.metric("Equipos", matches["home_team"].nunique())

        c1, c2, c3 = st.columns(3)
        c1.metric("Fuente de cuotas", source or "ninguna", help=HELP["fb_odds_source"])
        c2.metric("¿De cierre?", "Sí" if matches.attrs.get("odds_are_closing") else "No",
                  help=HELP["fb_closing"])
        c3.metric("Con precio", f"{odds_coverage(matches):.1%}", help=HELP["fb_coverage"])

        st.markdown("**Primeros partidos**", help=HELP["fb_table"])
        st.dataframe(matches.head(25), use_container_width=True)

    # ---------------------------------------------------------------- Mercado
    with tabs[1]:
        if not len(priced):
            st.warning("Sin cuotas utilizables no hay línea base que mostrar.")
        else:
            section("El margen de la casa", "fb_tab_mercado")
            margins = pd.Series(
                [overround(row) for row in priced[list(ODDS_COLUMNS)].to_numpy()],
                index=priced.index)
            c1, c2, c3 = st.columns(3)
            c1.metric("Margen medio", f"{margins.mean():.2%}", help=HELP["fb_overround"])
            c2.metric("Mínimo", f"{margins.min():.2%}")
            c3.metric("Máximo", f"{margins.max():.2%}")

            figure = go.Figure(go.Histogram(x=margins, nbinsx=30))
            figure.update_layout(xaxis_title="Sobrerredondeo", yaxis_title="Partidos", height=320)
            chart(figure, "Distribución del margen", "fb_overround_chart")

            section("¿Está calibrado el mercado?", "fb_tab_calibracion")
            probability = with_probabilities["p_home"]
            observed = (with_probabilities["outcome"] == "H").astype(float)
            buckets = pd.cut(probability, bins=list(CALIBRATION_BINS), include_lowest=True)
            grouped = pd.DataFrame({"p": probability, "y": observed, "bucket": buckets})
            summary = grouped.groupby("bucket", observed=True).agg(
                predicha=("p", "mean"), observada=("y", "mean"), partidos=("y", "size"))

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                                        line=dict(dash="dash")))
            figure.add_trace(go.Scatter(
                x=summary["predicha"], y=summary["observada"], mode="markers+lines",
                name="Mercado", text=[f"{n} partidos" for n in summary["partidos"]]))
            figure.update_layout(xaxis_title="Probabilidad del mercado (gana el local)",
                                 yaxis_title="Frecuencia observada", height=380)
            chart(figure, "Calibración de la cuota de cierre (victoria local)", "fb_calibration")
            st.dataframe(summary.reset_index().astype({"bucket": str}), use_container_width=True)

            st.caption(
                "Los puntos cerca de la diagonal significan que cuando el mercado dice 60%, gana el "
                "local ~60% de las veces. Eso es exactamente lo que hace difícil la barra: no hay un "
                "sesgo obvio que explotar. Con una sola temporada cada punto tiene pocos partidos, "
                "así que la dispersión es ruido, no señal."
            )

            section("Los tres métodos, lado a lado", "fb_tab_metodos")
            options = [
                f"{r.ds:%Y-%m-%d} · {r.home_team} vs {r.away_team}"
                for r in priced.itertuples()
            ]
            picked = st.selectbox("Partido", options, index=0, help=HELP["fb_match_pick"])
            row = priced.iloc[options.index(picked)]
            table = compare_methods(row[list(ODDS_COLUMNS)].to_numpy(dtype=float))
            table = table.rename(columns={"method": "método", **{
                column: OUTCOME_ES[outcome]
                for column, outcome in zip(PROBABILITY_COLUMNS, OUTCOMES)}})
            st.dataframe(table, use_container_width=True)
            st.caption(
                "Los tres discrepan más en los no favoritos, que es donde importa. Ninguno es "
                "correcto: se elige uno, se dice cuál, y se comprueba que la conclusión no cambie al "
                "cambiarlo. **Si cambia, el hallazgo es sobre el modelo de margen, no sobre el modelo.**"
            )

    # -------------------------------------------------------------- Resultados
    with tabs[2]:
        section("Cómo terminaron los partidos", "fb_tab_resultados")
        counts = matches["outcome"].value_counts()
        shares = [float(counts.get(outcome, 0)) / max(len(matches), 1) for outcome in OUTCOMES]

        columns = st.columns(3)
        for column, outcome, share in zip(columns, OUTCOMES, shares):
            column.metric(OUTCOME_ES[outcome], f"{share:.1%}",
                          help=HELP["fb_outcome_share"])

        figure = go.Figure(go.Bar(x=[OUTCOME_ES[o] for o in OUTCOMES], y=shares))
        figure.update_layout(yaxis_title="Proporción", height=320, yaxis_tickformat=".0%")
        chart(figure, "Reparto de resultados", "fb_outcomes")

        c1, c2, c3 = st.columns(3)
        c1.metric("Goles del local por partido", f"{matches['home_goals'].mean():.2f}",
                  help=HELP["fb_goals"])
        c2.metric("Goles del visitante", f"{matches['away_goals'].mean():.2f}")
        c3.metric("Total por partido",
                  f"{(matches['home_goals'] + matches['away_goals']).mean():.2f}")

        if source and len(with_probabilities):
            mean_market = [float(with_probabilities[column].mean())
                           for column in PROBABILITY_COLUMNS]
            figure = go.Figure()
            figure.add_trace(go.Bar(x=[OUTCOME_ES[o] for o in OUTCOMES], y=shares,
                                    name="Observado"))
            figure.add_trace(go.Bar(x=[OUTCOME_ES[o] for o in OUTCOMES], y=mean_market,
                                    name="Media del mercado"))
            figure.update_layout(barmode="group", yaxis_title="Proporción", height=340,
                                 yaxis_tickformat=".0%")
            chart(figure, "Observado contra la media del mercado", "fb_observed_vs_market")
            st.caption(
                "Dos barras casi iguales es lo esperado, y no es un resultado: el mercado acierta la "
                "frecuencia global de cada resultado sin esfuerzo. La pregunta que importa — si un "
                "modelo le gana **partido a partido** — necesita una regla de puntuación y un modelo, "
                "y ninguno de los dos existe todavía."
            )
