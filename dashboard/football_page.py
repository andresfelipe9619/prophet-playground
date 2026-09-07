"""The football page: the data layer, the market baseline, and Dixon-Coles.

Run through `dashboard/app.py`, which owns the sidebar's domain selector.

**How the model is shown.** The Pronóstico tab puts Dixon-Coles beside the
de-margined market for a single match — never a bare "model" column implying it
is good on its own. The Resultados tab runs the walk-forward backtest and shows
the measured verdict (naive and Bonferroni-corrected) against the closing line;
that verdict is the only number on the page that says whether the model is
worth anything. Everything else describes the data or the baseline itself: the
bar is the closing line, and beating chance is not the bar here.

What the data/baseline half is useful for: seeing which odds source a file
resolved to before you build anything on it, seeing how big the bookmaker's
margin is, and seeing that the market is calibrated — which is what makes it a
hard baseline rather than a convenient one.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football.common import DEFAULT_DATA_DIR, ODDS_COLUMNS, OUTCOMES, PROBABILITY_COLUMNS
from football.dixon_coles import UnknownTeamError
from football.extra_processor import available_leagues, load_extra
from football.h2h import head_to_head, team_form
from football.market import (
    METHODS,
    compare_methods,
    implied_probabilities,
    market_probabilities,
    overround,
)
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


@st.cache_data(show_spinner=False)
def load_extra_cached(path, league):
    """Load one football-data 'extra' file (Colombia and friends: opening odds only)."""
    return load_extra(path, league=league)


@st.cache_resource(show_spinner="Ajustando el modelo…")
def _fit_dixon_coles(_matches, cache_key):
    """Fit Dixon-Coles once per (data, half_life).

    Streamlit cannot reliably hash a match frame carrying `.attrs`, so the frame
    is passed underscore-prefixed (unhashed) and the cache is keyed on
    `cache_key` — a cheap tuple the caller builds from the frame's fingerprint
    and the half-life.
    """
    from football.dixon_coles import DixonColes

    half_life = cache_key[-1]
    return DixonColes.fit(_matches, half_life=half_life or None)


@st.cache_data(show_spinner=False)
def _run_backtest(cache_key, _matches, n_windows, half_life, method):
    """Walk-forward model-vs-market backtest, cached on the cheap `cache_key`.

    `_matches` is underscore-prefixed so Streamlit does not try to hash the
    frame (it carries `.attrs`); identity comes from `cache_key` plus the
    parameters. Slow: it refits Dixon-Coles once per evaluated match.
    """
    from football.backtest import run_all

    # Floor on the training set; window_bounds already takes max(min_train,
    # n - n_windows), so this is just "never fit on fewer than 100 matches".
    return run_all(_matches, n_windows=n_windows, min_train=100,
                   half_life=half_life, method=method)


def render():
    st.title("⚽ Fútbol")
    st.caption(
        "Datos, línea base del mercado y un modelo Dixon-Coles. La barra que tiene que superar "
        "es la **cuota de cierre**, no un Elo ni un 50/50. El modelo se muestra junto al mercado "
        "en **Pronóstico**; el veredicto medido, fuera de muestra y corregido, está en **Resultados**."
    )

    with st.sidebar:
        st.header("Datos")
        source_kind = st.radio(
            "Fuente", ["Europa (football-data)", "Colombia (archivo extra)"],
            help=HELP["fb_source_toggle"],
        )
        colombia = source_kind.startswith("Colombia")
        method = st.selectbox(
            "Método para quitar el margen", METHODS, index=0, help=HELP["fb_method"])

        col_path = chosen_league = None
        uploaded = None
        directory = DEFAULT_DATA_DIR
        chosen = []
        closing_odds_only = False
        if colombia:
            col_path = st.text_input(
                "Archivo COL.csv", value=os.path.join(DEFAULT_DATA_DIR, "COL.csv"))
            try:
                leagues = available_leagues(col_path)
            except (FileNotFoundError, OSError, ValueError, KeyError):
                leagues = []
            chosen_league = st.selectbox("Liga", leagues) if leagues else None
        else:
            uploaded = st.file_uploader("CSV de football-data.co.uk", type="csv")
            directory = st.text_input("Carpeta de temporadas", value=DEFAULT_DATA_DIR)
            available = season_files(directory)
            chosen = st.multiselect(
                "Temporadas", available, default=available[-1:],
                help=HELP["fb_seasons"],
            )
            closing_odds_only = st.checkbox(
                "Solo temporadas con cuotas de cierre", value=False, help=HELP["fb_closing_only"])

    if colombia:
        try:
            matches = load_extra_cached(col_path, chosen_league)
            is_demo = False
        except FileNotFoundError:
            matches = preprocess_matches(
                generate_matches(seed=0).drop(columns=["TrueH", "TrueD", "TrueA"]), validate=False)
            is_demo = True
            st.info(
                f"No se encontró el archivo **{col_path}** — mostrando **datos sintéticos**. "
                "Descárgalo con `python -m football.downloader --extra --leagues COL`."
            )
        report = check_match_format(matches)
    else:
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

    if is_demo and not colombia:
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

    tabs = st.tabs(["Datos", "Mercado", "Pronóstico", "Resultados"])

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

            section("Calibración del modelo (victoria local)", "fb_model_calibration")
            if len(matches) < 150:
                st.info(
                    "Hacen falta ~150 partidos para ajustar el modelo y medir su calibración; "
                    "carga más temporadas."
                )
            else:
                model_cache_key = (
                    len(matches),
                    str(matches["ds"].min()),
                    str(matches["ds"].max()),
                    int(matches["home_goals"].sum()),
                    int(matches["away_goals"].sum()),
                    0,
                )
                try:
                    model = _fit_dixon_coles(matches, model_cache_key)
                    model_probs = model.predict_matches(priced)
                    p_home_model = pd.Series(model_probs[:, 0], index=priced.index)
                    observed_model = (priced["outcome"] == "H").astype(float)
                    m_buckets = pd.cut(p_home_model, bins=list(CALIBRATION_BINS),
                                       include_lowest=True)
                    m_grouped = pd.DataFrame(
                        {"p": p_home_model, "y": observed_model, "bucket": m_buckets})
                    m_summary = m_grouped.groupby("bucket", observed=True).agg(
                        predicha=("p", "mean"), observada=("y", "mean"), partidos=("y", "size"))

                    figure = go.Figure()
                    figure.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                        line=dict(dash="dash")))
                    figure.add_trace(go.Scatter(
                        x=m_summary["predicha"], y=m_summary["observada"], mode="markers+lines",
                        name="Modelo",
                        text=[f"{n} partidos" for n in m_summary["partidos"]]))
                    figure.update_layout(
                        xaxis_title="Probabilidad del modelo (gana el local)",
                        yaxis_title="Frecuencia observada", height=380)
                    chart(figure, "Curva de calibración del modelo", "fb_model_calibration_curve")
                    st.dataframe(m_summary.reset_index().astype({"bucket": str}),
                                 use_container_width=True)
                    st.caption(
                        "Ajustado sobre las mismas temporadas que se muestran (dentro de muestra), "
                        "así que esto favorece al modelo. El veredicto fuera de muestra está en "
                        "**Resultados**."
                    )
                except Exception as exc:  # noqa: BLE001 - a fit warning must not kill the page
                    st.warning(f"No se pudo ajustar el modelo para la curva de calibración: {exc}")

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

    # ------------------------------------------------------------- Pronóstico
    with tabs[2]:
        render_forecast_tab(matches, method)

    # -------------------------------------------------------------- Resultados
    with tabs[3]:
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
                "frecuencia global de cada resultado sin esfuerzo. La regla de puntuación y el modelo "
                "ya existen; este gráfico sigue mostrando solo frecuencias globales. El veredicto "
                "**partido a partido** está en la sección «¿Le gana este modelo al mercado?» de abajo."
            )

        section("¿Le gana este modelo al mercado?", "fb_eval_tab")
        if source is None:
            st.warning("Sin cuotas no hay contra qué medir el modelo.")
        elif len(matches) < 150:
            st.info(
                "Hacen falta ~150 partidos para un backtest con sentido; carga más temporadas."
            )
        else:
            if source not in CLOSING_SOURCES:
                st.warning(
                    "La línea base de estos datos es de **apertura**. Cualquier ventaja que "
                    "aparezca aquí es contra un mercado blando y no es prueba de una ventaja real."
                )
            n_windows = st.slider("Partidos a evaluar (walk-forward)", 20, 200, 40, step=10)
            half_life = st.number_input(
                "Vida media (días), 0 = sin decaimiento",
                min_value=0, value=180, step=30, help=HELP["fb_half_life"])
            eval_key = (
                len(matches),
                str(matches["ds"].min()),
                str(matches["ds"].max()),
                int(matches["home_goals"].sum()),
                int(matches["away_goals"].sum()),
                source,
                n_windows,
                int(half_life),
                method,
            )
            if st.button("Correr backtest"):
                with st.spinner("Reajustando el modelo por ventana…"):
                    result = _run_backtest(
                        eval_key, matches, n_windows, int(half_life) or None, method)
                v1, v2, v3 = st.columns(3)
                v1.metric("Skill score (RPS)", f"{result['skill_score']:+.3f}",
                          help=HELP["fb_skill_score"])
                v2.metric("Efecto", f"{result['effect']:+.4f}",
                          help=f"IC 95%: [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
                v3.metric("Partidos", result["n_windows_scored"])
                b1, b2 = st.columns(2)
                b1.metric("Supera al mercado (naive)",
                          "Sí" if result["beats_market"] else "No", help=HELP["fb_beats_market"])
                b2.metric("Supera al mercado (corregido)",
                          "Sí" if result["beats_market_corrected"] else "No")
                st.dataframe(pd.DataFrame({
                    "Métrica": ["RPS"],
                    "Modelo": [f"{result.get('model_score', float('nan')):.4f}"],
                    "Mercado": [f"{result.get('market_score', float('nan')):.4f}"],
                }), use_container_width=True, hide_index=True)
                st.caption(
                    "Mira la columna **corregida**. Un skill score positivo con IC que cruza 0 "
                    "no es una ventaja: es ruido con el signo favorable."
                )


def render_forecast_tab(matches, method):
    """A single-match forecast: head-to-head, then Dixon-Coles beside the market.

    Model probabilities are only ever shown next to the de-margined market
    prices, or under an explicit "no baseline for this match" caption — never on
    their own implying the model is good by itself.
    """
    section("Pronóstico de un partido", "fb_forecast_tab")
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    if len(teams) < 2:
        st.warning("Hacen falta al menos dos equipos en los datos cargados.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    home_team = c1.selectbox("Local", teams, index=0)
    away_team = c2.selectbox("Visitante", teams, index=1)
    half_life = c3.number_input("Vida media (días)", min_value=0, value=180, step=30,
                                help=HELP["fb_half_life"])
    if home_team == away_team:
        st.warning("Elige dos equipos distintos.")
        return

    with st.form("odds_form"):
        st.markdown("**Cuotas actuales (opcional)**", help=HELP["fb_your_odds"])
        o1, o2, o3 = st.columns(3)
        odd_home = o1.number_input("Local", min_value=0.0, value=0.0, step=0.05)
        odd_draw = o2.number_input("Empate", min_value=0.0, value=0.0, step=0.05)
        odd_away = o3.number_input("Visitante", min_value=0.0, value=0.0, step=0.05)
        submitted = st.form_submit_button("Calcular pronóstico")

    if not submitted:
        st.info("Elige los equipos y pulsa **Calcular pronóstico**.")
        return

    # --- head to head ---
    section("Cómo llegan", "fb_h2h")
    h2h = head_to_head(matches, home_team, away_team)
    fc, ac = st.columns(2)
    for col, team in ((fc, home_team), (ac, away_team)):
        form = team_form(matches, team, last_n=5)
        col.metric(team,
                   "".join({"W": "V", "D": "E", "L": "D"}[r] for r in form["results"]) or "—",
                   help=HELP["fb_form"])
        col.caption(f"{form['wins']}V {form['draws']}E {form['losses']}D · "
                    f"{form['goals_for']}-{form['goals_against']} goles · {form['points']} pts")
    st.caption(
        f"{h2h['meetings']} enfrentamientos: {h2h['home_wins']} {home_team}, "
        f"{h2h['draws']} empates, {h2h['away_wins']} {away_team}. "
        f"Media de goles {h2h['avg_goals']:.2f}." if h2h["meetings"] else "Sin enfrentamientos previos."
    )

    # --- model ---
    cache_key = (
        len(matches),
        str(matches["ds"].min()),
        str(matches["ds"].max()),
        int(matches["home_goals"].sum()),
        int(matches["away_goals"].sum()),
        int(half_life),
    )
    try:
        model = _fit_dixon_coles(matches, cache_key)
        p_model = model.predict_outcome(home_team, away_team)
        grid = model.scoreline_matrix(home_team, away_team, max_goals=6)
        scores = model.most_likely_scores(home_team, away_team, n=5)
        p_over, p_under = model.over_under(home_team, away_team, 2.5)
        p_btts = model.both_teams_to_score(home_team, away_team)
    except UnknownTeamError as exc:
        st.error(f"El modelo no conoce a ese equipo en las temporadas cargadas. Detalle: {exc}")
        return

    section("El modelo", "fb_model_1x2")
    m1, m2, m3 = st.columns(3)
    for col, label, value in zip((m1, m2, m3), ("Local", "Empate", "Visitante"), p_model):
        col.metric(label, f"{value:.1%}")
    if not all(o > 1.0 for o in (odd_home, odd_draw, odd_away)):
        st.caption(
            "Estas cifras son solo el modelo: no hay cuotas para este partido, así que no hay "
            "mercado contra el que contrastarlas."
        )

    figure = go.Figure(go.Heatmap(
        z=grid, x=list(range(grid.shape[1])), y=list(range(grid.shape[0])), colorscale="Blues"))
    figure.update_layout(xaxis_title=f"Goles {away_team}", yaxis_title=f"Goles {home_team}",
                         height=380)
    chart(figure, "Probabilidad de cada marcador", "fb_scoreline_grid")

    st.markdown("**Marcadores más probables**", help=HELP["fb_most_likely_scores"])
    st.dataframe(pd.DataFrame(
        [{"Marcador": f"{h}-{a}", "Probabilidad": f"{p:.1%}"} for (h, a), p in scores]),
        use_container_width=True, hide_index=True)

    ou1, ou2, ou3 = st.columns(3)
    ou1.metric("Más de 2.5", f"{p_over:.1%}", help=HELP["fb_over_under"])
    ou2.metric("Menos de 2.5", f"{p_under:.1%}")
    ou3.metric("Ambos marcan", f"{p_btts:.1%}", help=HELP["fb_btts"])

    # --- market ---
    odds = (odd_home, odd_draw, odd_away)
    if all(o > 1.0 for o in odds):
        section("Modelo contra mercado", "fb_model_vs_market")
        p_market = implied_probabilities(np.array(odds), method=method)
        figure = go.Figure()
        figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_model, name="Modelo"))
        figure.add_trace(go.Bar(x=["Local", "Empate", "Visitante"], y=p_market, name="Mercado"))
        figure.update_layout(barmode="group", yaxis_tickformat=".0%", height=340)
        chart(figure, "Probabilidades: modelo y mercado", "fb_model_vs_market")
        st.dataframe(pd.DataFrame({
            "Resultado": ["Local", "Empate", "Visitante"],
            "Modelo": [f"{p:.1%}" for p in p_model],
            "Mercado": [f"{p:.1%}" for p in p_market],
            "Modelo − Mercado": [f"{m - k:+.1%}" for m, k in zip(p_model, p_market)],
        }), use_container_width=True, hide_index=True)
        st.caption(
            "Un partido, comparación **sin corregir**. Si el modelo se desvía mucho del mercado "
            "aquí, lo interesante es por qué, no que tenga razón. El veredicto medido está en "
            "**Resultados**."
        )
    else:
        st.info(
            "Sin cuotas para este partido no hay línea base para este partido: el pronóstico de "
            "arriba es solo el modelo. Pega las tres cuotas decimales para compararlo con el mercado."
        )
